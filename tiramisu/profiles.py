import time
import logging
import math
import pathlib
import typing as t
from concurrent.futures import ThreadPoolExecutor
from functools import partial, lru_cache

import numba
import numpy as np
import polars as pl
from astropy import constants as ac, units as u

from pyarrow import parquet as pq
from dask import dataframe as dd
from numpy import typing as npt
from scipy.special import roots_hermite, erf

from .config import _DEFAULT_NUM_THREADS, _INTENSITY_CUTOFF, _N_GH_QUAD_POINTS
from .numerics import loglinear_normalise_2d_nonnegative

log = logging.getLogger(__name__)

# Constants.
ac_h_on_8_c = ac.h.cgs / (8 * ac.c.cgs)
ac_4_pi_c = 4 * np.pi * ac.c.cgs
ac_8_pi_c = 2 * ac_4_pi_c
ac_8_pi_five_halves_c = ac_8_pi_c * (np.pi ** 1.5)
ac_16_pi_c = 2 * ac_8_pi_c
ac_h_c_on_4_pi = ac.h.cgs * ac.c.cgs / (4 * np.pi)
ac_h_c_on_4_pi_five_halves = ac_h_c_on_4_pi / (np.pi ** 1.5)
ac_h_c_on_8_pi = ac_h_c_on_4_pi / 2.0
ac_sqrt_NA_kB_on_c = (np.sqrt(ac.N_A * ac.k_B.cgs) / ac.c.cgs).to(
    u.g ** 0.5 / (u.K ** 0.5 * u.mol ** 0.5), equivalencies=u.spectral()
)
ac_sqrt_2_NA_kB_log2_on_c = ac_sqrt_NA_kB_on_c * np.sqrt(2 * np.log(2))
ac_c_sq_on_2h = (ac.c ** 2) / (2 * ac.h)
# Constant values.
const_amu = ac.u.cgs.value
const_h_on_8_c = ac_h_on_8_c.value
const_4_pi_c = ac_4_pi_c.value
const_8_pi_c = ac_8_pi_c.value
const_16_pi_c = ac_16_pi_c.value
const_8_pi_five_halves_c = ac_8_pi_five_halves_c.value
const_h_c_on_4_pi = ac_h_c_on_4_pi.value
const_h_c_on_4_pi_five_halves = ac_h_c_on_4_pi_five_halves.value
const_h_c_on_8_pi = ac_h_c_on_8_pi.value
const_sqrt_NA_kB_on_c = ac_sqrt_NA_kB_on_c.value
const_sqrt_2_NA_kB_log2_on_c = ac_sqrt_2_NA_kB_log2_on_c.value
const_c_sq_on_2h = ac_c_sq_on_2h.value


def create_erf_lut(
        n_points: int = 20000, arg_max: float = 6.0
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Create lookup table for erf() for sampling by :func:`tiramisu.nlte.get_erf_lut`.
    Only stores [0, arg_max] due to symmetry.

    Parameters
    ----------
    n_points: int
        Number of linearly spaced points up to arg_max.
    arg_max: float
        Maximum argument value for erf().
    Returns
    -------
        Values of erf() at n_points linearly spaced points up to arg_max.
    """
    args = np.linspace(0, arg_max, n_points, dtype=np.float64)
    erf_values = erf(args)
    return args, erf_values


global_erf_arg_max = 6.0

# --------------- ERF LOOKUP TABLE ---------------
_, global_erf_lut = create_erf_lut(n_points=20000, arg_max=global_erf_arg_max)


@numba.njit(cache=True, error_model="numpy", inline="always")
def get_erf_lut(erf_arg: float, erf_lut: npt.NDArray[np.float64], erf_arg_max: float) -> float:
    """
    Fast erf lookup with linear interpolation between points. Uses symmetry; erf(-x) = -erf(x).

    Parameters
    ----------
    erf_arg: float
        Argument to look up erf() result for.
    erf_lut: ndarray
        Lookup table to search.
    erf_arg_max: float
        Maximum erf() argument value associated with the lookup table

    Returns
    -------
        Value of erf() at given argument, linearly interpolated between lookup table points.
    """
    n_points: int = erf_lut.shape[0]

    sign = 1.0 if erf_arg >= 0 else -1.0
    abs_arg = abs(erf_arg)

    # Saturation for large arguments.
    if abs_arg >= erf_arg_max:
        return sign * 1.0

    # Map to index; linear interpolation.
    float_idx = (abs_arg / erf_arg_max) * (n_points - 1)
    idx = int(float_idx)
    frac = float_idx - idx

    # Bounds check for safety.
    if idx >= n_points - 1:
        return sign * erf_lut[n_points - 1]

    # Linear interpolation.
    val = erf_lut[idx] * (1.0 - frac) + erf_lut[idx + 1] * frac
    return sign * val


# ------------------------------------- CALC BAND PROFILE WRAPPERS -------------------------------------


def calc_band_profile_layered(
        wn_grid: npt.NDArray[np.float64],
        temperature_profile: npt.NDArray[np.float64],
        pressure_profile: npt.NDArray[np.float64],
        species_mass: float,
        broad_n: npt.NDArray[np.float64],
        broad_gamma: npt.NDArray[np.float64],
        n_nlte_layers: int,
        group: pl.DataFrame,
) -> pl.DataFrame:
    """
    Layer-aware replacement for calc_band_profile.

    Called once per (id_agg_f, id_agg_i) band group. Extracts the contiguous
    arrays for transitions in this group, slices the matching n_frac columns
    from the pre-assembled (n_layers, n_trans_total) population arrays, and
    delegates to _band_profile_binned_voigt_variable_width_layered.

    The returned DataFrame has one row per band with profile columns holding
    (n_nlte_layers, n_grid) arrays rather than (n_grid,) arrays.

    Parameters
    ----------
    wn_grid : ndarray, shape (n_grid,)
        The full wavenumber grid.
    temperature_profile : ndarray, shape (n_nlte_layers,)
        Kinetic temperatures for NLTE layers, float values in Kelvin.
    pressure_profile : ndarray, shape (n_nlte_layers,)
        Pressure for NLTE layers, float values in bars.
    species_mass : float
        Mass of the species for broadening.
    broad_n : ndarray, shape (n_broadeners,)
        Exponential temperature factor for pressure broadening.
    broad_gamma : ndarray, shape (n_broadeners, n_nlte_layers)
        gamma_0 for pressure broadening, weighted by mixing ratio at each layer.
    n_nlte_layers: int
        Number of NLTE layers.
    group : pl.DataFrame
        Subset of trans_batch for one (id_agg_f, id_agg_i) band.
    """
    a_fi = np.ascontiguousarray(group["A_fi"].to_numpy())
    g_i = np.ascontiguousarray(group["g_i"].to_numpy())
    g_f = np.ascontiguousarray(group["g_f"].to_numpy())
    energy_fi = np.ascontiguousarray(group["energy_fi"].to_numpy())
    tau_f = np.ascontiguousarray(group["tau_f"].to_numpy())

    # Stack per-layer population columns into (n_nlte_layers, n_trans) arrays.
    n_frac_i = np.ascontiguousarray(
        np.stack([group[f"n_frac_nL{l}_i"].fill_null(0.0).to_numpy() for l in range(n_nlte_layers)])
    )
    n_frac_f = np.ascontiguousarray(
        np.stack([group[f"n_frac_nL{l}_f"].fill_null(0.0).to_numpy() for l in range(n_nlte_layers)])
    )

    if energy_fi.min() > wn_grid.max() or energy_fi.max() < wn_grid.min():
        raise RuntimeError("Computing band profile for band with no transitions on grid: check workflow logic.")

    if (broad_n is None) ^ (broad_gamma is None):
        raise RuntimeError(f"Either broadening n or gamma missing: n={broad_n}, gamma={broad_gamma}.")

    # is_fixed_width = np.all(np.isclose(np.diff(wn_grid), np.abs(wn_grid[1] - wn_grid[0]), atol=0))

    gh_roots, gh_weights = roots_hermite(_N_GH_QUAD_POINTS)

    _abs_xsec, _ste_xsec, _spe_xsec = _band_profile_binned_voigt_variable_width_layered(
        wn_grid=wn_grid,
        n_i=n_frac_i,
        n_f=n_frac_f,
        a_fi=a_fi,
        g_f=g_f,
        g_i=g_i,
        energy_fi=energy_fi,
        lifetimes=tau_f,
        temperatures=temperature_profile,
        pressures=pressure_profile,
        broad_n=broad_n,
        broad_gamma=broad_gamma,
        species_mass=species_mass,
        gh_roots=gh_roots,
        gh_weights=gh_weights,
    )
    return pl.DataFrame({
        "id_agg_f": [group["id_agg_f"][0]],
        "id_agg_i": [group["id_agg_i"][0]],
        "abs_profile": [_abs_xsec.ravel()],  # flattened to (n_nlte_layers * n_grid,)
        "ste_profile": [_ste_xsec.ravel()],
        "spe_profile": [_spe_xsec.ravel()],
        "n_layers": [_abs_xsec.shape[0]],  # n_nlte_layers, for reconstruction
        "n_grid": [_abs_xsec.shape[1]],  # grid length, for reconstruction
    })


def calc_continuum_band_profile_layered(
        wn_grid: npt.NDArray[np.float64],
        temperature_profile: npt.NDArray[np.float64],
        species_mass: float,
        box_length: float,
        n_nlte_layers: int,
        group: pl.DataFrame,
) -> pl.DataFrame:
    """
    Called once per (id_agg_f, id_agg_i) band group. Extracts the contiguous arrays for transitions in this group,
    slices the matching n_frac columns from the pre-assembled (n_layers, n_trans_total) population arrays, and
    delegates to _band_profile_binned_voigt_variable_width_layered.

    The returned DataFrame has one row per band with profile columns holding (n_nlte_layers, n_grid) arrays rather than
    (n_grid,) arrays.

    Parameters
    ----------
    wn_grid : ndarray, shape (n_grid,)
        The full wavenumber grid.
    temperature_profile : ndarray, shape (n_nlte_layers,)
        Kinetic temperatures for NLTE layers, plain float values in Kelvin.
    species_mass : float
        Mass of the species for broadening.
    box_length : float
        Box length off to use in continuum box broadening.
    n_nlte_layers: int
        Number of NLTE layers.
    group : pl.DataFrame
        Subset of trans_batch for one (id_agg_f, id_agg_i) band.
    """
    a_fi = np.ascontiguousarray(group["A_fi"].to_numpy())
    g_i = np.ascontiguousarray(group["g_i"].to_numpy())
    g_f = np.ascontiguousarray(group["g_f"].to_numpy())
    energy_fi = np.ascontiguousarray(group["energy_fi"].to_numpy())
    v_f = np.ascontiguousarray(group["v_f"].to_numpy())

    # Stack per-layer population columns into (n_nlte_layers, n_trans) arrays.
    n_frac_i = np.ascontiguousarray(
        np.stack([group[f"n_frac_nL{l}_i"].fill_null(0.0).to_numpy() for l in range(n_nlte_layers)])
    )

    if energy_fi.min() > wn_grid.max() or energy_fi.max() < wn_grid.min():
        raise RuntimeError("Computing band profile for band with no transitions on grid: check workflow logic.")

    # is_fixed_width = np.all(np.isclose(np.diff(wn_grid), np.abs(wn_grid[1] - wn_grid[0]), atol=0))

    _abs_xsec = _continuum_profile_binned_gauss_variable_width_layered(
        wn_grid=wn_grid,
        n_i=n_frac_i,
        a_fi=a_fi,
        g_f=g_f,
        g_i=g_i,
        energy_fi=energy_fi,
        v_f=v_f,
        temperatures=temperature_profile,
        species_mass=species_mass,
        box_length=box_length,
    )
    return pl.DataFrame({
        "id_agg_f": [group["id_agg_f"][0]],
        "id_agg_i": [group["id_agg_i"][0]],
        "abs_profile": [_abs_xsec.ravel()],  # flattened to (n_nlte_layers * n_grid,)
        "n_layers": [_abs_xsec.shape[0]],  # n_nlte_layers, for reconstruction
        "n_grid": [_abs_xsec.shape[1]],  # grid length, for reconstruction
    })


# ------------------------------------- EINSTEIN COEFFICIENTS -------------------------------------

@numba.njit(parallel=False, cache=True, error_model="numpy")
def calc_einstein_b_fi(a_fi: npt.NDArray[np.float64], energy_fi: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Here the Einstein B coefficient is given in :math:`\text{m}^2 / (\text{J·s})`.

    .. math::
        B_{fi}=\\frac{A_{fi}}{2hc\\tilde{\\nu}^{3}_{fi}}

    Parameters
    ----------
    a_fi
    energy_fi

    Returns
    -------

    """
    # return a_fi / (2 * ac.h * ac.c * (energy_fi ** 3))  # WAVENUMBERS
    # return (a_fi * (ac.c ** 2)) / (2 * ac.h * (energy_fi ** 3))  # FREQUENCY
    return (a_fi * const_c_sq_on_2h) / (energy_fi ** 3)  # FREQUENCY
    # return a_fi * (energy_fi ** 3) / (2 * ac.h * ac.c)  # LAMBDA


@numba.njit(parallel=False, cache=True, error_model="numpy")
def calc_einstein_b_if(
        b_fi: npt.NDArray[np.float64],
        g_f: npt.NDArray[np.float64],
        g_i: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return b_fi * g_f / g_i


@numba.njit(parallel=True, cache=True, error_model="numpy")
def calc_einstein_bs(
        id_i: npt.NDArray[np.int32],
        id_f: npt.NDArray[np.int32],
        a_fi: npt.NDArray[np.float64],
        energy_fi: npt.NDArray[np.float64],
        g_lookup: npt.NDArray[np.float64],
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    n_trans = a_fi.shape[0]
    b_fi = np.zeros(n_trans, dtype=np.float64)
    b_if = np.zeros(n_trans, dtype=np.float64)
    inv_g_lookup = 1.0 / g_lookup
    inv_energy_fi_cubed = 1.0 / (energy_fi ** 3)

    for t in numba.prange(n_trans):
        a_fi_t = a_fi[t]
        g_f_t = g_lookup[id_f[t]]
        inv_g_i_t = inv_g_lookup[id_i[t]]
        inv_energy_fi_cubed_t = inv_energy_fi_cubed[t]
        b_fi_t = a_fi_t * const_c_sq_on_2h * inv_energy_fi_cubed_t
        b_fi[t] = b_fi_t
        b_if[t] = b_fi_t * g_f_t * inv_g_i_t

    return b_fi, b_if


# ------------------------------------- PROCESS BATCH HELPERS -------------------------------------
@numba.njit(parallel=True, cache=True, error_model="numpy")
def _accumulate_superline_batch(
        buffer_abs: npt.NDArray[np.float64],
        buffer_emi: npt.NDArray[np.float64],
        bin_indices: npt.NDArray[np.int32],
        g_lookup: npt.NDArray[np.float64],
        abs_prefactor: npt.NDArray[np.float64],
        emi_prefactor: npt.NDArray[np.float64],
        id_i: npt.NDArray[np.int32],
        id_f: npt.NDArray[np.int32],
        n_lookup: npt.NDArray[np.float64],
):
    """

    Parameters
    ----------
    buffer_abs : ndarray, shape (n_layers, n_grid)
    buffer_emi : ndarray, shape (n_layers, n_grid)
    bin_indices : ndarray, shape (n_trans,)
    g_lookup : ndarray, shape (n_trans, n_layers)
    abs_prefactor : ndarray, shape (n_trans,)
    emi_prefactor : ndarray, shape (n_trans,)
    id_i : ndarray, shape (n_trans,)
    id_f : ndarray, shape (n_trans,)
    n_lookup : ndarray, shape (n_trans, n_layers)

    Returns
    -------

    """
    n_trans = bin_indices.shape[0]
    n_layers = n_lookup.shape[1]

    inv_g_lookup = 1.0 / g_lookup

    for l in numba.prange(n_layers):
        for t in range(n_trans):
            bin_t = bin_indices[t]
            abs_prefactor_t = abs_prefactor[t]
            emi_prefactor_t = emi_prefactor[t]
            # Populations.
            n_il = n_lookup[id_i[t], l]
            n_fl = n_lookup[id_f[t], l]
            # Degeneracies.
            g_f_t = g_lookup[id_f[t]]
            inv_g_i_t = inv_g_lookup[id_i[t]]
            # Accumulate into buffers.
            buffer_abs[l, bin_t] += (
                    abs_prefactor_t * ((n_il * g_f_t * inv_g_i_t) - n_fl)
            )
            buffer_emi[l, bin_t] += (
                    emi_prefactor_t * n_fl
            )


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _accumulate_superline_band_batch(
        profile_buffer: npt.NDArray[np.float64],
        band_indices: npt.NDArray[np.int32],
        bin_indices: npt.NDArray[np.int32],
        g_lookup: npt.NDArray[np.float64],
        abs_ste_prefactor: npt.NDArray[np.float64],
        spe_prefactor: npt.NDArray[np.float64],
        id_i: npt.NDArray[np.int32],
        id_f: npt.NDArray[np.int32],
        n_frac_lookup: npt.NDArray[np.float64],
) -> None:
    """

    Parameters
    ----------
    profile_buffer : ndarray, shape (3, n_bands, n_layers, n_grid)
    band_indices : ndarray, shape (n_trans,)
    bin_indices : ndarray, shape (n_trans,)
    g_lookup : ndarray, shape (n_states + 1, )
    abs_ste_prefactor : ndarray, shape (n_trans,)
    spe_prefactor : ndarray, shape (n_trans,)
    id_i : ndarray, shape (n_trans,)
    id_f : ndarray, shape (n_trans,)
    n_frac_lookup : ndarray, shape (n_states + 1, n_layers)
    """
    n_trans = bin_indices.shape[0]
    n_layers = n_frac_lookup.shape[1]

    inv_g_lookup = 1.0 / g_lookup

    for l in numba.prange(n_layers):
        for t in range(n_trans):
            band_t = band_indices[t]
            bin_t = bin_indices[t]
            abs_ste_prefactor_t = abs_ste_prefactor[t]
            abs_prefactor_t = abs_ste_prefactor_t * g_lookup[id_f[t]] * inv_g_lookup[id_i[t]]
            ste_prefactor_t = abs_ste_prefactor_t
            spe_prefactor_t = spe_prefactor[t]
            # Accumulate into buffers.
            profile_buffer[0, band_t, l, bin_t] += (
                    abs_prefactor_t * n_frac_lookup[id_i[t], l]
            )
            profile_buffer[1, band_t, l, bin_t] += (
                    ste_prefactor_t * n_frac_lookup[id_f[t], l]
            )
            profile_buffer[2, band_t, l, bin_t] += (
                    spe_prefactor_t * n_frac_lookup[id_f[t], l]
            )


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _accumulate_continuum_superline_band_batch(
        profile_buffer: npt.NDArray[np.float64],
        band_indices: npt.NDArray[np.int32],
        bin_indices: npt.NDArray[np.int32],
        n_frac_lookup: npt.NDArray[np.float64],
        g_lookup: npt.NDArray[np.float64],
        v_lookup: npt.NDArray[np.float64],
        abs_prefactor: npt.NDArray[np.float64],
        id_i: npt.NDArray[np.int32],
        id_f: npt.NDArray[np.int32],
        reduced_mass: float,
        box_length: float,
) -> None:
    """
    Can be modified to add in photoassociation cross-section if desired.

    Parameters
    ----------
    profile_buffer : ndarray, shape (2, n_bands, n_layers, n_grid)
    band_indices : ndarray, shape (n_trans,)
    bin_indices : ndarray, shape (n_trans,)
    n_frac_lookup : ndarray, shape (n_states + 1, n_layers)
    g_lookup : ndarray, shape (n_states + 1, )
    v_lookup : ndarray, shape (n_states + 1, )
    abs_prefactor : ndarray, shape (n_trans,)
    id_i : ndarray, shape (n_trans,)
    id_f : ndarray, shape (n_trans,)
    reduced_mass : float
    box_length : floatawdasdasda

    """
    n_trans = bin_indices.shape[0]
    n_layers = n_frac_lookup.shape[1]
    n_states = v_lookup.shape[0]

    inv_g_lookup = 1.0 / g_lookup

    alpha_box_lookup = np.empty(n_states, dtype=np.float64)
    alpha_box_prefactor = const_h_on_8_c / (const_amu * box_length * box_length * reduced_mass)
    for s in range(n_states):
        alpha_box_lookup[s] = alpha_box_prefactor * (2.0 * v_lookup[s] + 1.0)

    for l in numba.prange(n_layers):
        for t in range(n_trans):
            band_t = band_indices[t]
            bin_t = bin_indices[t]
            abs_tl = abs_prefactor[t] * g_lookup[id_f[t]] * inv_g_lookup[id_i[t]] * n_frac_lookup[id_i[t], l]
            # Accumulate into buffers.
            profile_buffer[0, band_t, l, bin_t] += abs_tl
            profile_buffer[1, band_t, l, bin_t] += abs_tl * alpha_box_lookup[id_f[t]]


def _process_trans_batch_layered(
        trans_batch: pl.DataFrame,
        profile_buffer: npt.NDArray[np.float64],
        band_indices: npt.NDArray[np.int32],
        n_frac_lookup: npt.NDArray[np.float64],
        g_lookup: npt.NDArray[np.float64],
        tau_lookup: npt.NDArray[np.float64],
        broad_n: npt.NDArray[np.float64],
        broad_gamma: npt.NDArray[np.float64],
        species_mass: float,
        wn_grid: npt.NDArray[np.float64],
        temperature_profile: npt.NDArray[np.float64],
        pressure_profile: npt.NDArray[np.float64],
) -> None:
    """
    Called from :func:`~xsec.NLTEProcessor.compute_rates_profiles`. Input trans_batch is joined on states prior to this
    function call and must be sorted on the columns ["id_f_agg", "id_i_agg"] so numpy slicing maintains group order.

    Computes the band profiles for every band across all layers, on the fixed wn_grid. Absorption, stimulated and
    spontaneous emission are computed separately and returned in a profile buffer to be accumulated with the results of
    other transition files or file chunks.

    Parameters
    ----------
    trans_batch : polars.DataFrame
        Polars DataFrame of raw transitions (id_f, id_i, A_fi) and joined states data.
    profile_buffer : ndarray, shape (3, n_bands, n_layers, n_grid)
    band_indices : ndarray, shape (n_trans, )
    n_frac_lookup : ndarray, shape (n_states + 1, n_layers)
        Lookup table for n_frac, avoiding extremely memory intensive joins.
    g_lookup : ndarray, shape (n_states + 1, )
        Lookup table for g, avoiding extremely memory intensive joins.
    tau_lookup : ndarray, shape (n_states + 1, )
        Lookup table for tau (lifetimes), avoiding extremely memory intensive joins.
    broad_n : ndarray, shape (n_broadeners,)
        Pressure broadening exponent n.
    broad_gamma : ndarray, shape (n_broadeners, n_layers)
        Pressure broadening gamma, per layer due to mixing of broadeners.
    species_mass : float
        Mass of the species for broadening, in Daltons.
    wn_grid : ndarray, shape (n_grid,)
        The full wavenumber grid.
    temperature_profile : ndarray, shape (n_nlte_layers,)
        Kinetic temperatures for NLTE layers, plain float values in Kelvin.
    pressure_profile : ndarray, shape (n_nlte_layers,)
        Pressures for NLTE layers, plain float values in Pascals.
    """
    # Compute band profiles for all bands and all layers at once.
    start_time = time.perf_counter()
    id_agg_f = trans_batch["id_agg_f"].to_numpy()
    id_agg_i = trans_batch["id_agg_i"].to_numpy()
    # Finds indices where either id_agg value changes.
    group_boundaries = np.where(
        (id_agg_f[1:] != id_agg_f[:-1]) |
        (id_agg_i[1:] != id_agg_i[:-1])
    )[0] + 1
    group_starts = np.concatenate([[0], group_boundaries], dtype=np.int32)
    group_ends = np.concatenate([group_boundaries, [len(id_agg_f)]], dtype=np.int32)
    band_group_indices = np.ascontiguousarray(band_indices[group_starts])

    _band_profile_sampled_voigt_layered(
        profile_buffer=profile_buffer,
        wn_grid=wn_grid,
        id_f=np.ascontiguousarray(trans_batch["id_f"].to_numpy()),
        id_i=np.ascontiguousarray(trans_batch["id_i"].to_numpy()),
        band_group_indices=band_group_indices,
        group_starts=group_starts,
        group_ends=group_ends,
        n_lookup=n_frac_lookup,
        g_lookup=g_lookup,
        tau_lookup=tau_lookup,
        a_fi=np.ascontiguousarray(trans_batch["A_fi"].to_numpy()),
        energy_fi=np.ascontiguousarray(trans_batch["energy_fi"].to_numpy()),
        temperatures=temperature_profile,
        pressures=pressure_profile,
        broad_n=broad_n,
        broad_gamma=broad_gamma,
        species_mass=species_mass,
    )
    log.info(f"Numba accumulate duration = {time.perf_counter() - start_time:.3f}s.")


def _process_continuum_trans_batch_layered(
        trans_batch: pl.DataFrame,
        profile_buffer: npt.NDArray[np.float64],
        band_indices: npt.NDArray[np.int32],
        n_frac_lookup: npt.NDArray[np.float64],
        g_lookup: npt.NDArray[np.float64],
        v_lookup: npt.NDArray[np.float64],
        species_mass: float,
        reduced_mass: float,
        cont_box_length: float,
        wn_grid: npt.NDArray[np.float64],
        temperature_profile: npt.NDArray[np.float64],
) -> None:
    """
    Called from :func:`~xsec.NLTEProcessor.compute_continuum_rates_profiles`. Input trans_batch is joined on states
    prior to this function call and must be sorted on the column ["id_i_agg"] so numpy slicing maintains group order.
    The column "id_f_agg" is not used here as the continuum is state is not tracked.

    Computes the band profiles for every band across all layers, on the fixed wn_grid. Absorption is computed separately
    and returned in a profile buffer to be accumulated with the results of other transition files or file chunks.
    Stimulated and spontaneous emission are not yet implemented.

    Parameters
    ----------
    trans_batch : polars.DataFrame
        Polars DataFrame of raw transitions (id_f, id_i, A_fi).
    profile_buffer : ndarray, shape (3, n_bands, n_layers, n_grid)
    band_indices : ndarray, shape (n_trans, )
    n_frac_lookup : ndarray, shape (n_states + 1, n_layers)
        Lookup table for n_frac, avoiding extremely memory intensive joins.
    g_lookup : ndarray, shape (n_states + 1, )
        Lookup table for g, avoiding extremely memory intensive joins.
    v_lookup : ndarray, shape (n_states + 1, )
        Lookup table for v for box broadening, avoiding extremely memory intensive joins.
    species_mass : float
        Mass of the species for Doppler broadening, in Daltons.
    reduced_mass : float
        Reduced mass of the species for box broadening, in Daltons.
    cont_box_length : float
        Box length off to use in continuum box broadening.
    wn_grid : ndarray, shape (n_grid,)
        The full wavenumber grid.
    temperature_profile : ndarray, shape (n_nlte_layers,)
        Kinetic temperatures for NLTE layers, plain float values in Kelvin.
    """
    trans_batch = trans_batch.fill_null(strategy="zero")

    start_time = time.perf_counter()
    id_agg_f = trans_batch["id_agg_f"].to_numpy()
    id_agg_i = trans_batch["id_agg_i"].to_numpy()
    # Finds indices where either id_agg value changes.
    group_boundaries = np.where(
        # (id_agg_f_arr[1:] != id_agg_f_arr[:-1]) |
        (id_agg_i[1:] != id_agg_i[:-1])
    )[0] + 1
    group_starts = np.concatenate([[0], group_boundaries])
    group_ends = np.concatenate([group_boundaries, [len(id_agg_f)]])
    band_group_indices = np.ascontiguousarray(band_indices[group_starts])

    _continuum_band_profile_sampled_gauss_layered(
        profile_buffer=profile_buffer,
        wn_grid=wn_grid,
        id_f=np.ascontiguousarray(trans_batch["id_f"].to_numpy()),
        id_i=np.ascontiguousarray(trans_batch["id_i"].to_numpy()),
        band_group_indices=band_group_indices,
        group_starts=group_starts,
        group_ends=group_ends,
        n_lookup=n_frac_lookup,
        g_lookup=g_lookup,
        v_lookup=v_lookup,
        a_fi=np.ascontiguousarray(trans_batch["A_fi"].to_numpy()),
        energy_fi=np.ascontiguousarray(trans_batch["energy_fi"].to_numpy()),
        temperatures=temperature_profile,
        species_mass=species_mass,
        reduced_mass=reduced_mass,
        box_length=cont_box_length,
    )
    log.info(f"Numba accumulate continuum duration = {time.perf_counter() - start_time:.3f}s.")


# ------------------------------------- COMPACT PROFILE & STORE CLASSES -------------------------------------

@numba.njit(parallel=True, cache=True, error_model="numpy")
def _analyse_profiles(
        profile_matrix_2d: npt.NDArray[np.float64], cutoff: float
) -> t.Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
    # profile_matrix_2d: (n_profiles, n_grid) - caller reshapes
    n_profiles, n_grid = profile_matrix_2d.shape
    start_idxs = np.empty(n_profiles, dtype=np.int64)
    end_idxs = np.empty(n_profiles, dtype=np.int64)
    valid = np.zeros(n_profiles, dtype=np.bool_)

    for i in numba.prange(n_profiles):
        profile = profile_matrix_2d[i]
        first = -1
        last = -1
        for j in range(n_grid):
            if profile[j] >= cutoff:
                if first == -1:
                    first = j
                last = j
        if first != -1:
            start_idxs[i] = first
            end_idxs[i] = last + 1
            valid[i] = True

    return start_idxs, end_idxs, valid


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _write_profiles(
        profile_matrix_2d: npt.NDArray[np.float64],
        profiles_out: npt.NDArray[np.float64],
        offsets_full: npt.NDArray[np.int64],
        start_idxs: npt.NDArray[np.int64],
        end_idxs: npt.NDArray[np.int64],
        valid: npt.NDArray[np.bool_]
) -> None:
    n_profiles = valid.shape[0]
    for i in numba.prange(n_profiles):
        if not valid[i]:
            continue
        s = start_idxs[i]
        e = end_idxs[i]
        off = offsets_full[i]
        for k in range(e - s):
            profiles_out[off + k] = profile_matrix_2d[i, s + k]


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _build_xsec(
        profiles: npt.NDArray[np.float64],
        offsets: npt.NDArray[np.int64],
        start_idxs: npt.NDArray[np.int64],
        key_idx_array: npt.NDArray[np.int64],
        pop_matrix: npt.NDArray[np.float64],
        wn_grid_len: int,
        is_abs: bool,
        numba_num_threads: int = _DEFAULT_NUM_THREADS,
) -> npt.NDArray[np.float64]:
    """
    For use only by :class:`tiramisu.nlte.CompactProfile` instances.

    Parameters
    ----------
    profiles : ndarray
        1-dimensional array of combined profiles from CompactProfile.
    offsets : ndarray
        1-dimensional array of offsets from CompactProfile.
    start_idxs : ndarray
        1-dimensional array of starting indices from CompactProfile.
    key_idx_array : ndarray
        2-dimensional array storing ID_u, ID_l mapping for profile at corresponding index.
    pop_matrix : ndarray
        1-dimensional array storing populations of state ID corresponding to array index.
    wn_grid_len : int
        Number of points on the wavenumber grid.
    is_abs : bool
        Boolean controlling whether population is chosen from ID_u (Emission) or ID_l (Absorption).

    Returns
    -------
    ndarray
        Combined cross-section from all population-adjusted band profiles.
    """
    n_profiles = start_idxs.shape[0]

    # Allocate per-thread accumulation buffer
    # Note: This allocation is performed inside the njit function; it is ok and reused only for the scope of this call.
    buffers = np.zeros((numba_num_threads, wn_grid_len), dtype=np.float64)

    # Accumulate profile in buffers[thread_id,:] per thread.
    for i in numba.prange(n_profiles):
        thread_id = numba.get_thread_id()  # Thread ID [0..n_threads-1]

        # Absorption depends on ID_l population; Emission depends on ID_u population.
        if is_abs:
            pop_val = pop_matrix[key_idx_array[i, 1]]
        else:
            pop_val = pop_matrix[key_idx_array[i, 0]]

        offset_start = offsets[i]
        offset_end = offsets[i + 1]
        profile_len = offset_end - offset_start
        start_idx = start_idxs[i]

        # Element-wise accumulation into the thread's buffer row.
        for j in range(profile_len):
            buffers[thread_id, start_idx + j] += pop_val * profiles[offset_start + j]

    # Parallelize over grid indices, avoids race conditions.
    xsec_out = np.zeros(wn_grid_len, dtype=np.float64)
    for k in numba.prange(wn_grid_len):
        xsec_point = 0.0
        for t in range(numba_num_threads):
            xsec_point += buffers[t, k]
        xsec_out[k] = xsec_point

    return xsec_out


@numba.njit(parallel=False, cache=True, error_model="numpy")
def _rebuild_all_ox_profiles(
        id_agg_cutoff: int,
        key_idx_map: npt.NDArray[np.int64],
        profiles: npt.NDArray[np.float64],
        offsets: npt.NDArray[np.int64],
        start_idxs: npt.NDArray[np.int64],
        num_grid: int,
) -> npt.NDArray[np.float64]:
    """
    Rebuild full spontaneous emission profiles for all upper states in a single pass.

    Scanning key_idx_map once, accumulating each band's contribution directly into the row of the output matrix
    corresponding to its upper state.

    Could be refactored to accumulate into a 3D, per-thread buffer if performance struggles for polyatomics.

    Parameters
    ----------
    id_agg_cutoff : int
        ID cutoff; sets the number of rows in the output as id_agg_cutoff + 1.
    key_idx_map : np.ndarray, shape (n_profiles, 2)
        Each row is (upper_state_id, lower_state_id) for the stored band.
    profiles : np.ndarray, shape (total_points,)
        Contiguous array of trimmed profile values.
    offsets : np.ndarray, shape (n_profiles + 1,)
        Start index of each profile in `profiles`, with a terminator at the end.
    start_idxs : np.ndarray, shape (n_profiles,)
        Position of each trimmed profile on the full wavenumber grid.
    num_grid : int
        Length of the full wavenumber grid.

    Returns
    -------
    all_profiles : np.ndarray, shape (id_agg_cutoff + 1, num_grid)
        Row o_idx contains the summed emission profile from upper state o_idx
        to all lower states. Unnormalised.
    """
    all_profiles = np.zeros((id_agg_cutoff + 1, num_grid), dtype=np.float64)

    num_profiles = key_idx_map.shape[0]
    for idx in range(num_profiles):
        upper_state_id = key_idx_map[idx, 0]
        if upper_state_id < 0 or upper_state_id > id_agg_cutoff:
            continue

        offset_start = offsets[idx]
        offset_end = offsets[idx + 1]
        wn_start = start_idxs[idx]
        profile_len = offset_end - offset_start
        wn_end = min(wn_start + profile_len, num_grid)

        for j in range(wn_end - wn_start):
            all_profiles[upper_state_id, wn_start + j] += profiles[offset_start + j]

    return all_profiles


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _compute_all_cross_terms_vectorized(
        emission_profiles: npt.NDArray[np.float64],
        chem_factor: float,
) -> npt.NDArray[np.float64]:
    """

    Parameters
    ----------
    emission_profiles
    chem_factor

    Returns
    -------

    """
    n_agg_states, num_grid = emission_profiles.shape
    result = np.zeros((n_agg_states, num_grid), dtype=np.float64)

    for o_idx in numba.prange(n_agg_states):
        for wn_idx in range(num_grid):
            result[o_idx, wn_idx] = (
                    emission_profiles[o_idx, wn_idx] * chem_factor
            )

    return result


class CompactProfile:
    __slots__ = ["profiles", "offsets", "start_idxs", "key_idx_map", "key_lookup"]

    def __init__(self):
        self.profiles: npt.NDArray[np.float64] | None = None
        self.offsets: npt.NDArray[np.int64] | None = None
        self.start_idxs: npt.NDArray[np.int64] | None = None
        self.key_idx_map: npt.NDArray[np.int64] | None = None
        self.key_lookup: t.Dict[t.Tuple[int, int], int] = {}

    def finalise_from_buffer(self, profile_matrix: npt.NDArray[np.float64], keys: npt.NDArray[np.int64]):
        """
        Finalise a set of band profiles stored in the compact super-line representation.

        The starting indices within self.profiles of each individual profile is contained within self.offsets. An extra
        terminator is stored at the end of self.offsets equal to the total length of self.profiles; this is so the start
        and end indices of a given profile can always be obtained by looking at the current and next offset.

        The starting position of each profile on the main wavenumber grid is stored at the corresponding index in
        self.start_idxs.

        The upper and lower state IDs are stored in self.key_idx_map; the index of the first dimension of this array
        matches the corresponding index in self.offsets and self.start_idxs. This is used for fast cross-section
        reconstruction in :func:`tiramisu.nlte.CompactProfile.build_xsec`. A fast dictionary lookup for accessing
        individual bands is stored in self.key_lookup, used by :func:`tiramisu.nlte.CompactProfile.get_profile`.

        Parameters
        ----------
        profile_matrix : ndarray, shape (n_bands, n_grid)
            Profile values for every populated band on the common wavenumber grid.

        keys : ndarray, shape (n_bands, 2)
            Mapping from profile index -> (id_f_agg, id_i_agg).
            Row i corresponds to profile_matrix[i].

        Notes
        -----
        For each populated profile only the section exceeding _INTENSITY_CUTOFF is stored. The resulting compressed
        arrays are:

        profiles
            Concatenated profile values.

        offsets
            Start position of each stored profile within profiles.

        start_idxs
            Starting wavenumber-grid index of each stored profile.

        key_idx_map
            (id_f_agg, id_i_agg) identifiers corresponding to each stored
            profile.

        key_lookup
            Dictionary mapping (id_f_agg, id_i_agg) -> profile index.
        """
        log.info(f"CompactProfile: profile shape = {profile_matrix.shape}")
        n_profiles, n_grid = profile_matrix.shape

        if keys.shape[0] != n_profiles:
            raise ValueError(
                f"Number of keys ({keys.shape[0]}) does not match number of profiles ({n_profiles})."
            )

        start_idxs, end_idxs, valid = _analyse_profiles(
            profile_matrix_2d=profile_matrix,
            cutoff=_INTENSITY_CUTOFF,
        )
        log.info(f"valids = {valid}")
        lengths = np.where(valid, end_idxs - start_idxs, 0)

        # Prefix sum gives output offsets.
        offsets_full = np.zeros(n_profiles + 1, dtype=np.int64)
        offsets_full[1:] = np.cumsum(lengths)
        total_len = offsets_full[-1]

        profiles_out = np.empty(total_len, dtype=np.float64)
        _write_profiles(
            profile_matrix_2d=profile_matrix,
            profiles_out=profiles_out,
            offsets_full=offsets_full,
            start_idxs=start_idxs,
            end_idxs=end_idxs,
            valid=valid,
        )
        # Build compact index arrays (only valid entries).
        n_valid = int(valid.sum())

        self.offsets = np.empty(n_valid + 1, dtype=np.int64)
        self.start_idxs = np.empty(n_valid, dtype=np.int64)
        self.key_idx_map = np.empty((n_valid, 2), dtype=np.int64)
        self.key_lookup.clear()

        store_idx = 0
        for profile_idx in range(n_profiles):
            if not valid[profile_idx]:
                continue
            self.offsets[store_idx] = offsets_full[profile_idx]
            self.start_idxs[store_idx] = start_idxs[profile_idx]
            self.key_idx_map[store_idx] = keys[profile_idx]
            self.key_lookup[(int(keys[profile_idx, 0]), int(keys[profile_idx, 1]))] = store_idx
            store_idx += 1

        self.offsets[n_valid] = total_len
        self.profiles = profiles_out
        log.info(f"Finalised CompactProfile with {total_len} points for {n_valid} bands.")

    def finalise(self, profile_matrix: npt.NDArray[np.float64]) -> None:
        """
        Used when super-lines is disabled.

        The starting indices within self.profiles of each individual profile is contained within self.offsets. An extra
        terminator is stored at the end of self.offsets equal to the total length of self.profiles; this is so the start
        and end indices of a given profile can always be obtained by looking at the current and next offset.

        The starting position of each profile on the main wavenumber grid is stored at the corresponding index in
        self.start_idxs.

        The upper and lower state IDs are stored in self.key_idx_map; the index of theh first dimension of this array
        matches the corresponding index in self.offsets and self.start_idxs. This is used for fast cross-section
        reconstruction in :func:`tiramisu.nlte.CompactProfile.build_xsec`. A fast dictionary lookup for accessing
        individual bands is stored in self.key_lookup, used by :func:`tiramisu.nlte.CompactProfile.get_profile`.

        Parameters
        ----------
        profile_matrix : ndarray, shape (n_agg_states, n_agg_states, n_grid)
            Array containing the profiles for every band in the layer on the same fixed grid.
        """
        if profile_matrix.ndim == 3:
            n0, n1, n_grid = profile_matrix.shape
            flat = profile_matrix.reshape(n0 * n1, n_grid)
            # Generate keys: (i, j) pairs in ndindex order
            keys = np.array([(i, j) for i in range(n0) for j in range(n1)], dtype=np.int64)
        elif profile_matrix.ndim == 2:
            flat = profile_matrix  # (n_agg, n_grid)
            keys = np.array([(-1, i) for i in range(flat.shape[0])], dtype=np.int64)
        else:
            raise ValueError(f"Unsupported number of columns in CompactProfile input ({profile_matrix.ndim}).")

        n_profiles, n_grid = flat.shape
        start_idxs, end_idxs, valid = _analyse_profiles(profile_matrix_2d=flat, cutoff=_INTENSITY_CUTOFF)

        lengths = np.where(valid, end_idxs - start_idxs, 0)
        # Prefix sum gives each profile's offset in the output array.
        offsets_full = np.zeros(n_profiles + 1, dtype=np.int64)
        offsets_full[1:] = np.cumsum(lengths)
        total_len = offsets_full[-1]

        profiles_out = np.empty(total_len, dtype=np.float64)
        _write_profiles(
            profile_matrix_2d=flat,
            profiles_out=profiles_out,
            offsets_full=offsets_full,
            start_idxs=start_idxs,
            end_idxs=end_idxs,
            valid=valid
        )
        # Build compact index arrays (only valid entries).
        n_valid = int(valid.sum())
        self.offsets = np.empty(n_valid + 1, dtype=np.int64)
        self.start_idxs = np.empty(n_valid, dtype=np.int64)
        self.key_idx_map = np.empty((n_valid, 2), dtype=np.int64)

        store_idx = 0
        for i in range(n_profiles):
            if not valid[i]:
                continue
            self.offsets[store_idx] = offsets_full[i]
            self.start_idxs[store_idx] = start_idxs[i]
            self.key_idx_map[store_idx] = keys[i]
            self.key_lookup[tuple(keys[i])] = store_idx
            store_idx += 1
        self.offsets[n_valid] = total_len
        self.profiles = profiles_out
        log.info(f"Finalised CompactProfile with {total_len} points for {n_valid} bands.")

    def get_profile(self, key: t.Tuple[int, int] | int) -> t.Tuple[npt.NDArray[np.float64], int] | None:
        profile_idx = self.key_lookup.get(key)
        if profile_idx is None:
            return None
        offset_start, offset_end = self.offsets[profile_idx: profile_idx + 2]
        start_idx = self.start_idxs[profile_idx]
        profile = self.profiles[offset_start:offset_end]
        return profile, start_idx

    def build_xsec(
            self, pop_matrix: npt.NDArray[np.float64], wn_grid: npt.NDArray[np.float64], is_abs: bool
    ) -> npt.NDArray[np.float64]:
        # Ensure arrays are the right dtype and contiguous
        profiles = np.ascontiguousarray(self.profiles, dtype=np.float64)
        offsets = np.ascontiguousarray(self.offsets, dtype=np.int64)
        start_idxs = np.ascontiguousarray(self.start_idxs, dtype=np.int64)
        key_idx_map = np.ascontiguousarray(self.key_idx_map, dtype=np.int64)

        return _build_xsec(
            profiles, offsets, start_idxs, key_idx_map, pop_matrix, wn_grid.shape[0], is_abs
        )

    def get_all_emission_from_upper(
            self, id_agg_cutoff: int, num_grid: int
    ) -> npt.NDArray[np.float64]:
        """
        Rebuild spontaneous emission profiles for all upper states in a single pass.

        Returns
        -------
        all_profiles : np.ndarray, shape (id_agg_cutoff, num_grid)
        """
        return _rebuild_all_ox_profiles(
            id_agg_cutoff=id_agg_cutoff,
            key_idx_map=np.ascontiguousarray(self.key_idx_map, dtype=np.int64),
            profiles=np.ascontiguousarray(self.profiles, dtype=np.float64),
            offsets=np.ascontiguousarray(self.offsets, dtype=np.int64),
            start_idxs=np.ascontiguousarray(self.start_idxs, dtype=np.int64),
            num_grid=num_grid,
        )


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _scatter_add_profiles(
        abs_accum: npt.NDArray[np.float64],  # (n_layers, n_agg, n_agg, n_grid)
        ste_accum: npt.NDArray[np.float64],
        spe_accum: npt.NDArray[np.float64],
        profile_buffer: npt.NDArray[np.float64],  # (3, n_bands, n_layers, n_grid)
        band_id_f: npt.NDArray[np.float64],
        band_id_i: npt.NDArray[np.float64],
):
    """
    Takes the input profile buffer from :func:`~profiles._process_trans_batch_layered` and the associated
    id_agg arrays defining the bands and adds them to the relevant abs/ste/spe accumulation buffers. Used by
    :func:`~profiles.ProfileStore.add_batch` to avoid storing multiple buffer objects before calling
    :func:`~profiles.ProfileStore.finalise`.

    Parameters
    ----------
    abs_accum : ndarray, shape (n_layers, n_agg, n_agg, n_grid)
        Accumulation buffer for absorption profiles across all bands and layers.
    ste_accum : ndarray, shape (n_layers, n_agg, n_agg, n_grid)
        Accumulation buffer for stimulated emission profiles across all bands and layers.
    spe_accum : ndarray, shape (n_layers, n_agg, n_agg, n_grid)
        Accumulation buffer for spontaneous emission profiles across all bands and layers.
    profile_buffer : ndarray, shape (3, n_bands, n_layers, n_grid)
        Profile buffer where first dimension tracks abs/ste/spe profile type.
    band_id_f : ndarray, shape (n_agg, )
        1D array containing the id_f_agg values for each band.
    band_id_i : ndarray, shape (n_agg, )
        1D array containing the id_f_agg values for each band.

    Returns
    -------

    """
    n_layers = abs_accum.shape[0]
    n_bands = band_id_f.shape[0]
    n_grid = abs_accum.shape[3]

    for layer_idx in numba.prange(n_layers):
        for band_idx in range(n_bands):
            f = band_id_f[band_idx]
            i = band_id_i[band_idx]
            for k in range(n_grid):
                abs_accum[layer_idx, f, i, k] += profile_buffer[0, band_idx, layer_idx, k]
                ste_accum[layer_idx, f, i, k] += profile_buffer[1, band_idx, layer_idx, k]
                spe_accum[layer_idx, f, i, k] += profile_buffer[2, band_idx, layer_idx, k]


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _scatter_add_profiles_continuum(
        abs_accum: npt.NDArray[np.float64],  # (n_layers, n_agg, n_agg, n_grid)
        # ste_accum: npt.NDArray[np.float64],
        # spe_accum: npt.NDArray[np.float64],
        profile_buffer: npt.NDArray[np.float64],  # (1, n_bands, n_layers, n_grid)
        band_id_f: npt.NDArray[np.float64],
        band_id_i: npt.NDArray[np.float64],
):
    """
    Takes the input profile buffer from :func:`~profiles._process_trans_batch_layered` and the associated
    id_agg arrays defining the bands and adds them to the relevant abs/ste/spe accumulation buffers. Used by
    :func:`~profiles.ProfileStore.add_batch` to avoid storing multiple buffer objects before calling
    :func:`~profiles.ProfileStore.finalise`.

    Stimulated and spontaneous emission are not currently implemented for continuum bands.

    Parameters
    ----------
    abs_accum : ndarray, shape (n_layers, n_agg, n_agg, n_grid)
        Accumulation buffer for absorption profiles across all bands and layers.
    # ste_accum : ndarray, shape (n_layers, n_agg, n_agg, n_grid)
    #     Accumulation buffer for stimulated emission profiles across all bands and layers.
    # spe_accum : ndarray, shape (n_layers, n_agg, n_agg, n_grid)
    #     Accumulation buffer for spontaneous emission profiles across all bands and layers.
    profile_buffer : ndarray, shape (3, n_bands, n_layers, n_grid)
        Profile buffer where first dimension tracks abs/ste/spe profile type.
    band_id_f : ndarray, shape (n_agg, )
        1D array containing the id_f_agg values for each band.
    band_id_i : ndarray, shape (n_agg, )
        1D array containing the id_f_agg values for each band.

    Returns
    -------

    """
    n_layers = abs_accum.shape[0]
    n_bands = band_id_f.shape[0]
    n_grid = abs_accum.shape[3]

    for layer_idx in numba.prange(n_layers):
        for band_idx in range(n_bands):
            f = band_id_f[band_idx]
            i = band_id_i[band_idx]
            for k in range(n_grid):
                abs_accum[layer_idx, f, i, k] += profile_buffer[0, band_idx, layer_idx, k]
                # ste_accum[layer_idx, f, i, k] += profile_buffer[1, band_idx, layer_idx, k]
                # spe_accum[layer_idx, f, i, k] += profile_buffer[2, band_idx, layer_idx, k]


class ProfileStore:
    """
    Store :class:`tiramisu.nlte.CompactProfile` objects representing the species' absorption, stimulated emission and
    spontaneous emission profiles. Used for bound-bound transitions (including quasi-bound).
    """
    __slots__ = ["n_layers", "abs_profiles", "ste_profiles", "spe_profiles"]

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        # Storage for final profiles.
        self.abs_profiles = [CompactProfile() for _ in range(n_layers)]  # Absorption
        self.ste_profiles = [CompactProfile() for _ in range(n_layers)]  # Stimulated Emission
        self.spe_profiles = [CompactProfile() for _ in range(n_layers)]  # Spontaneous Emission

    def finalise_from_buffer(
            self,
            profile_buffer: npt.NDArray[np.float64],
            band_keys: npt.NDArray[np.int64],
            save: bool = False,
            species: str = None,
    ) -> None:
        """

        Parameters
        ----------
        profile_buffer : ndarray, shape (3, n_bands, n_layers, n_grid)
        band_keys : ndarray, shape (n_bands, 2)
            Contains the (id_f_agg, id_i_agg) pairs for each band.
        save : bool
            Flag for whether compressed ProfileStore outputs should be saved.
        species : str
            String label for the species for naming saved outputs.

        """
        # TEMP below!
        export_dir = pathlib.Path(r"/mnt/c/PhD/programs/TIRAMISU/tests/outputs/")
        export_dir.mkdir(parents=True, exist_ok=True)
        #####
        for l in range(self.n_layers):
            self.abs_profiles[l].finalise_from_buffer(
                profile_matrix=profile_buffer[0, :, l, :],
                keys=band_keys,
            )
            self.ste_profiles[l].finalise_from_buffer(
                profile_matrix=profile_buffer[1, :, l, :],
                keys=band_keys,
            )
            self.spe_profiles[l].finalise_from_buffer(
                profile_matrix=profile_buffer[2, :, l, :],
                keys=band_keys,
            )
            if save:
                for label, store in (
                        ("abs", self.abs_profiles[l]),
                        ("ste", self.ste_profiles[l]),
                        ("spe", self.spe_profiles[l]),
                ):
                    if store.profiles is None:
                        continue
                    np.savez_compressed(
                        export_dir / f"{species}_L{l:03d}_{label}.npz",
                        profiles=store.profiles,  # flat 1D array of all profile data
                        offsets=store.offsets,  # ragged array boundaries
                        start_idxs=store.start_idxs,  # position of each profile on wn_grid
                        key_idx_map=store.key_idx_map,
                    )

    def get_profiles(
            self, layer_idx: int, key: t.Tuple[int, int]
    ) -> t.Tuple[
        t.Tuple[npt.NDArray[np.float64], int],
        t.Tuple[npt.NDArray[np.float64], int],
        t.Tuple[npt.NDArray[np.float64], int]
    ]:
        return (
            self.abs_profiles[layer_idx].get_profile(key),
            self.ste_profiles[layer_idx].get_profile(key),
            self.spe_profiles[layer_idx].get_profile(key),
        )

    @lru_cache(maxsize=1000)
    def get_profile(
            self, layer_idx: int, key: t.Tuple[int, int], profile_type: str
    ) -> t.Tuple[npt.NDArray[np.float64], int]:
        if profile_type == "abs":
            return self.abs_profiles[layer_idx].get_profile(key)
        if profile_type == "ste":
            return self.ste_profiles[layer_idx].get_profile(key)
        if profile_type == "spe":
            return self.spe_profiles[layer_idx].get_profile(key)
        else:
            raise RuntimeError(f"ProfileStore profile type {profile_type} not implemented.")

    def build_abs_emi(
            self, layer_idx: int, pop_matrix: npt.NDArray[np.float64], wn_grid: npt.NDArray[np.float64]
    ) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        abs_profile = self.abs_profiles[layer_idx].build_xsec(pop_matrix=pop_matrix, wn_grid=wn_grid, is_abs=True)
        ste_profile = self.ste_profiles[layer_idx].build_xsec(pop_matrix=pop_matrix, wn_grid=wn_grid, is_abs=False)
        spe_profile = self.spe_profiles[layer_idx].build_xsec(pop_matrix=pop_matrix, wn_grid=wn_grid, is_abs=False)
        return abs_profile - ste_profile, spe_profile

    def precompute_normalised_downward_emission_profiles(
            self,
            layer_idx: int,
            id_agg_cutoff: int,
            wn_grid: u.Quantity,
    ) -> u.Quantity:
        """
        Precompute normalised spontaneous emission profiles for all upper states.

        Each profile represents the total downward emission from a given upper state summed over all lower states, then
        integral normalised over wn_grid so. Profiles with zero integral (no emission from that state) are left as zero.

        Parameters
        ----------
        layer_idx : int
        id_agg_cutoff : int
            id above which to cutoff calculations.
        wn_grid : astropy.units.Quantity, shape (num_grid,)
            Wavenumber grid used for normalisation. Must match the grid against which
            profiles were originally built.

        Returns
        -------
        all_profiles : astropy.units.Quantity, shape (n_agg_states, num_grid)
            Normalised emission profiles; each row integrates to 1 (or is zero).
            Units are 1/wn_grid.unit.
        """
        all_profiles = self.spe_profiles[layer_idx].get_all_emission_from_upper(
            id_agg_cutoff=id_agg_cutoff,
            num_grid=wn_grid.shape[0],
        )
        wn_dx = np.diff(wn_grid.value)
        # return simpson_normalise_2d(y_data=all_profiles, x_data=wn_grid.value) << 1 / wn_grid.unit
        return loglinear_normalise_2d_nonnegative(y_data=all_profiles, dx=wn_dx) << 1 / wn_grid.unit

    def get_sorted_band_keys(
            self, num_max: int, id_cutoff: int, agg_energies: npt.NDArray[np.float64], profile_type: str = "abs",
    ) -> t.List[t.Tuple[int, int]]:
        """
        Returns band keys (id_agg_f, id_agg_i) from the CompactProfile for the given layer, sorted by energy gap
        (id_agg_f - id_agg_i) ascending, then by id_agg_i, then by id_agg_f. Returns at most num_max keys.

        Parameters
        ----------
        num_max : int
            Maximum number of keys to return.
        id_cutoff : int
            ID above which states are fixed, so key pairs containing IDs above this should be excluded.
        agg_energies : ndarray
            Array containing the energies for each aggregate state ordered on id_agg (ascending).
        profile_type : str
            One of "abs", "ste", "spe" - selects which CompactProfile to inspect.

        Returns
        -------
        List of (id_agg_f, id_agg_i) tuples in the above order.
        """
        match profile_type:
            case "abs":
                profiles = self.abs_profiles
            case "ste":
                profiles = self.ste_profiles
            case "spe":
                profiles = self.spe_profiles
            case _:
                raise ValueError(f"Unknown profile_type '{profile_type}': expected 'abs', 'ste' or 'spe'.")

        key_sets = [set(compact.key_lookup.keys()) for compact in profiles]
        common_keys = key_sets[0].intersection(*key_sets[1:])
        common_keys = [k for k in common_keys if id_cutoff >= k[0] > k[1] >= 0]

        def sort_key(k):
            id_f, id_i = k
            e_i = agg_energies[id_i]  # lower state energy
            e_f = agg_energies[id_f]  # upper state energy
            nu = e_f - e_i  # band centre frequency
            return e_i, nu

        common_keys.sort(key=sort_key)

        return common_keys[:num_max]


class ContinuumProfileStore:
    __slots__ = [
        "n_layers", "abs_profiles",
        # "ste_profiles", "spe_profiles"
    ]

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        # Storage for final profiles.
        self.abs_profiles = [CompactProfile() for _ in range(n_layers)]  # Absorption
        # self.ste_profiles = [CompactProfile() for _ in range(n_layers)]  # Stimulated Emission
        # self.spe_profiles = [CompactProfile() for _ in range(n_layers)]  # Spontaneous Emission

    def finalise_from_buffer(
            self,
            profile_buffer: npt.NDArray[np.float64],
            band_keys: npt.NDArray[np.int64],
            save: bool = False,
            species: str = None,
    ) -> None:
        """

        Parameters
        ----------
        profile_buffer : ndarray, shape (1, n_bands, n_layers, n_grid)
        band_keys : ndarray, shape (n_bands, 2)
            Contains the (id_f_agg, id_i_agg) pairs for each band.
        save : bool
            Flag for whether compressed ProfileStore outputs should be saved.
        species : str
            String label for the species for naming saved outputs.
        """
        # TEMP below!
        export_dir = pathlib.Path(r"/mnt/c/PhD/programs/TIRAMISU/tests/outputs/")
        export_dir.mkdir(parents=True, exist_ok=True)
        #####
        for l in range(self.n_layers):
            self.abs_profiles[l].finalise_from_buffer(
                profile_matrix=profile_buffer[0, :, l, :],
                keys=band_keys,
            )
            if save:
                for label, store in (
                        ("abs", self.abs_profiles[l]),
                ):
                    if store.profiles is None:
                        continue
                    np.savez_compressed(
                        export_dir / f"{species}_L{l:03d}_{label}.npz",
                        profiles=store.profiles,  # flat 1D array of all profile data
                        offsets=store.offsets,  # ragged array boundaries
                        start_idxs=store.start_idxs,  # position of each profile on wn_grid
                        key_idx_map=store.key_idx_map,
                    )

    @lru_cache(maxsize=1000)
    def get_profile(self, layer_idx: int, key: int, profile_type: str) -> t.Tuple[npt.NDArray, int]:
        if type(key) is int:
            get_key = (-1, key)
        else:
            get_key = key
        if profile_type == "abs":
            # Continuum profiles are stored with an arbitrary index for the upper continuum state (-1).
            return self.abs_profiles[layer_idx].get_profile(get_key)
        else:
            raise RuntimeError(f"ContinuumProfileStore profile type {profile_type} not implemented.")

    def get_keys(self, layer_idx: int, profile_type: str):
        if profile_type == "abs":
            return self.abs_profiles[layer_idx].key_lookup.keys()
        else:
            raise RuntimeError(f"ContinuumProfileStore profile type {profile_type} not implemented.")

    def build_profiles(
            self, layer_idx: int, pop_matrix: npt.NDArray[np.float64], wn_grid: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        abs_profile = self.abs_profiles[layer_idx].build_xsec(pop_matrix=pop_matrix, wn_grid=wn_grid, is_abs=True)
        return abs_profile


# ------------------------------------- FINAL XSEC CALCULATIONS -------------------------------------

def abs_emi_xsec(
        states: pl.DataFrame,
        trans_files: t.List[pathlib.Path],
        n_lte_layers: int,
        n_nlte_layers: int,
        temperature_profile: npt.NDArray[np.float64],
        pressure_profile: npt.NDArray[np.float64],
        wn_grid: npt.NDArray[np.float64],
        species_mass: float,
        do_super_lines: bool,
        broadening_params: t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] = None,
        # n_gh_quad_points: int = _N_GH_QUAD_POINTS,
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Uses sampled profiles for final computations, so GH quadrature is deprecated.

    Parameters
    ----------
    states
    trans_files
    n_lte_layers
    n_nlte_layers
    temperature_profile
    pressure_profile
    wn_grid
    species_mass
    broadening_params
    # n_gh_quad_points: int
    #     Deprecated Gauss-Hermite quadrature for binned Voigt profiles.

    Returns
    -------

    """
    half_bin_width = abs(wn_grid[1] - wn_grid[0]) / 2.0
    # TODO: Check is_fixed_width rigour!
    is_fixed_width = np.all(np.isclose(np.diff(wn_grid), np.abs(wn_grid[1] - wn_grid[0]), atol=0))
    n_grid = wn_grid.shape[0]

    wn_min = wn_grid[0]
    wn_max = wn_grid[-1]

    n_cols = [f"n_nlte_L{n_lte_layers + nlte_idx}" for nlte_idx in range(n_nlte_layers)]

    # invariant_cols_i = ["id", "energy", "g"] + n_cols
    # invariant_cols_f = ["id", "energy", "g", "tau"] + n_cols
    invariant_cols = ["id", "energy", "id_agg"]

    states_i = (
        states
        .select(invariant_cols)
        .rename({col: f"{col}_i" for col in invariant_cols})
    )
    states_f = (
        states
        .select(invariant_cols)
        .rename({col: f"{col}_f" for col in invariant_cols})
    )
    # Plain float arrays - no astropy units - passed directly to Numba.
    temperature_slice = temperature_profile[n_lte_layers:]  # (n_nlte_layers,)
    pressure_slice = pressure_profile[n_lte_layers:]  # (n_nlte_layers,)

    broad_n = np.zeros(1, dtype=np.float64)
    broad_gamma = np.zeros((1, n_nlte_layers), dtype=np.float64)
    if broadening_params is not None:
        broad_n = broadening_params[1]
        broad_gamma = broadening_params[0][:, n_lte_layers:]

    trans_columns = ["id_f", "id_i", "A_fi"]
    dask_dtypes = {"id_f": "int64", "id_i": "int64", "A_fi": "float64", }
    dask_blocksize = "256MB"

    # States lookup - much more memory efficient than duplicating it all with polars joins!
    state_ids = states["id"].to_numpy()
    max_state_id = int(state_ids.max())
    # State ID lookup offset by 1 as IDs are 1-indexed!
    n_lookup = np.zeros(
        (max_state_id + 1, n_nlte_layers),
        dtype=np.float64,
    )
    for l in range(n_nlte_layers):
        n_lookup[state_ids, l] = states[f"n_nlte_L{n_lte_layers + l}"].to_numpy()
    n_lookup = np.ascontiguousarray(n_lookup)

    g_lookup = np.zeros(max_state_id + 1, dtype=np.float64)
    g_lookup[state_ids] = states["g"].to_numpy()
    g_lookup = np.ascontiguousarray(g_lookup)

    tau_lookup = np.zeros(max_state_id + 1, dtype=np.float64)
    tau_lookup[state_ids] = states["tau"].to_numpy()
    tau_lookup = np.ascontiguousarray(tau_lookup)

    process_time = time.perf_counter()
    parquet_batch_size = 20_000_000
    if do_super_lines:
        # ---------------------------------------------- SUPER-LINES ----------------------------------------------
        # States lookup
        state_ids = states["id"].to_numpy()
        max_state_id = int(state_ids.max())
        # State ID lookup offset by 1 as IDs are 1-indexed!
        n_lookup = np.zeros(
            (max_state_id + 1, n_nlte_layers),
            dtype=np.float64,
        )
        for l in range(n_nlte_layers):
            n_lookup[state_ids, l] = states[f"n_nlte_L{n_lte_layers + l}"].to_numpy()
        n_lookup = np.ascontiguousarray(n_lookup)

        # Super-lines accumulator.
        profile_buffer = np.zeros((2, n_nlte_layers, n_grid), dtype=np.float64)
        profile_buffer = np.ascontiguousarray(profile_buffer)
        buffer_abs = profile_buffer[0]
        buffer_emi = profile_buffer[1]

        # Bin edges for finding trans bins.
        wn_arr = wn_grid.value
        bin_edges = np.empty(n_grid + 1, dtype=np.float64)
        bin_edges[0] = wn_arr[0] - (wn_arr[1] - wn_arr[0]) * 0.5
        for j in range(1, n_grid):
            bin_edges[j] = (wn_arr[j - 1] + wn_arr[j]) * 0.5
        bin_edges[-1] = wn_arr[-1] + (wn_arr[-1] - wn_arr[-2]) * 0.5

        for trans_file in trans_files:
            log.info(f"Processing file {trans_file}.")
            # Preprocessed Parquet files (MUCH faster).
            if str(trans_file).endswith(".parquet"):
                with pq.ParquetFile(trans_file) as pq_file:
                    batch_iter = pq_file.iter_batches(
                        batch_size=parquet_batch_size,
                        columns=trans_columns,
                        use_threads=True,
                    )
                    for arrow_batch in batch_iter:
                        trans_batch = pl.from_arrow(arrow_batch)
                        trans_batch = (
                            trans_batch
                            .with_columns([
                                pl.col("id_i").cast(pl.Int32),
                                pl.col("id_f").cast(pl.Int32),
                            ])
                            .join(states_i, on="id_i", how="inner")
                            .join(states_f, on="id_f", how="inner")
                            .with_columns(
                                (pl.col("energy_f") - pl.col("energy_i")).alias("energy_fi")
                            )
                            .filter(
                                (pl.col("energy_fi") >= wn_min)
                                & (pl.col("energy_fi") <= wn_max)
                            )
                            .fill_null(strategy="zero")
                        )
                        if trans_batch.height == 0:
                            log.info("No valid trans in batch.")
                            continue
                        # Extract numpy arrays.
                        id_i = np.ascontiguousarray(trans_batch["id_i"].to_numpy())
                        id_f = np.ascontiguousarray(trans_batch["id_f"].to_numpy())
                        a_fi = trans_batch["A_fi"].to_numpy()

                        energy_fi = trans_batch["energy_fi"].to_numpy()
                        abs_prefactor = np.ascontiguousarray(a_fi / (const_8_pi_c * energy_fi ** 2))
                        emi_prefactor = np.ascontiguousarray(a_fi / (energy_fi * const_h_c_on_4_pi))

                        # Get bin indices per transition.
                        bin_indices = np.searchsorted(bin_edges, energy_fi, side="right") - 1
                        bin_indices = np.clip(bin_indices, 0, n_grid - 1, dtype=np.int32)
                        bin_indices = np.ascontiguousarray(bin_indices)

                        # Write all transitions to respective bins.
                        _accumulate_superline_batch(
                            buffer_abs=buffer_abs,
                            buffer_emi=buffer_emi,
                            bin_indices=bin_indices,
                            g_lookup=g_lookup,
                            abs_prefactor=abs_prefactor,
                            emi_prefactor=emi_prefactor,
                            id_i=id_i,
                            id_f=id_f,
                            n_lookup=n_lookup,
                        )
            else:
                # ExoMol format text files (*.trans files from website).
                ddf = dd.read_csv(
                    trans_file,
                    sep=r"\s+",
                    engine="python",
                    header=None,
                    names=trans_columns,
                    usecols=[0, 1, 2],
                    dtype=dask_dtypes,
                    blocksize=dask_blocksize,
                )
                for delayed_batch in ddf.to_delayed():
                    trans_batch = pl.from_pandas(delayed_batch.compute())
                    trans_batch = (
                        trans_batch
                        .with_columns([
                            pl.col("id_i").cast(pl.Int32),
                            pl.col("id_f").cast(pl.Int32),
                        ])
                        .join(states_i, on="id_i", how="inner")
                        .join(states_f, on="id_f", how="inner")
                        .with_columns(
                            (pl.col("energy_f") - pl.col("energy_i")).alias("energy_fi")
                        )
                        .filter(
                            (pl.col("energy_fi") >= wn_min)
                            & (pl.col("energy_fi") <= wn_max)
                        )
                        .fill_null(strategy="zero")
                    )
                    if trans_batch.height == 0:
                        continue
                    # Extract numpy arrays.
                    id_i = np.ascontiguousarray(trans_batch["id_i"].to_numpy())
                    id_f = np.ascontiguousarray(trans_batch["id_f"].to_numpy())
                    a_fi = trans_batch["A_fi"].to_numpy()

                    energy_fi = trans_batch["energy_fi"].to_numpy()
                    abs_prefactor = np.ascontiguousarray(a_fi / (const_8_pi_c * energy_fi ** 2))
                    emi_prefactor = np.ascontiguousarray(a_fi / (energy_fi * const_h_c_on_4_pi))

                    # Get bin indices per transition.
                    bin_indices = np.searchsorted(bin_edges, energy_fi, side="right") - 1
                    bin_indices = np.clip(bin_indices, 0, n_grid - 1, dtype=np.int32)
                    bin_indices = np.ascontiguousarray(bin_indices)

                    # Write all transitions to respective bins.
                    _accumulate_superline_batch(
                        buffer_abs=buffer_abs,
                        buffer_emi=buffer_emi,
                        bin_indices=bin_indices,
                        g_lookup=g_lookup,
                        abs_prefactor=abs_prefactor,
                        emi_prefactor=emi_prefactor,
                        id_i=id_i,
                        id_f=id_f,
                        n_lookup=n_lookup,
                    )
        abs_xsec, emi_xsec = _broaden_superline_buffer(
            buffer_abs=buffer_abs,
            buffer_emi=buffer_emi,
            wn_grid=wn_grid,
            temperatures=temperature_profile.value,
            pressures=pressure_profile.value,
            broad_n=broad_n,
            broad_gamma=broad_gamma,
            species_mass=species_mass,
        )
    else:
        # ------------------------------------------ REGULAR LINE-BY-LINE ------------------------------------------
        # Accumulate each iteration.
        abs_xsec = np.zeros((n_nlte_layers, n_grid), dtype=np.float64)
        emi_xsec = np.zeros_like(abs_xsec, dtype=np.float64)

        for trans_file in trans_files:
            log.info(f"Processing file {trans_file}.")
            if str(trans_file).endswith(".parquet"):
                with pq.ParquetFile(trans_file) as pq_file, ThreadPoolExecutor(max_workers=1) as executor:
                    num_rg = pq_file.num_row_groups

                    def read_rg(idx):
                        return pl.from_arrow(
                            pq_file.read_row_group(idx, columns=trans_columns, use_threads=True)
                        )

                    future = executor.submit(read_rg, 0)
                    for rg_idx in range(num_rg):
                        trans_batch = future.result()
                        # Prefetch next batch.
                        if rg_idx + 1 < num_rg:
                            future = executor.submit(read_rg, rg_idx + 1)
                        trans_batch = (
                            trans_batch
                            .with_columns([
                                pl.col("id_i").cast(pl.Int32),
                                pl.col("id_f").cast(pl.Int32),
                            ])
                            .join(states_i, on="id_i", how="inner")
                            .join(states_f, on="id_f", how="inner")
                            .with_columns(
                                (pl.col("energy_f") - pl.col("energy_i")).alias("energy_fi")
                            )
                            .filter(
                                (pl.col("energy_fi") >= wn_min)
                                & (pl.col("energy_fi") <= wn_max)
                            )
                            .fill_null(strategy="zero")
                        )
                        if trans_batch.height == 0:
                            continue
                        # Get all contiguous arrays.
                        id_f = np.ascontiguousarray(trans_batch["id_f"].to_numpy(), dtype=np.float64)
                        id_i = np.ascontiguousarray(trans_batch["id_i"].to_numpy())
                        a_fi = np.ascontiguousarray(trans_batch["A_fi"].to_numpy())
                        energy_fi = np.ascontiguousarray(trans_batch["energy_fi"].to_numpy())

                        # All layer calculation.
                        # If we want to use integrated xsecs, replace with checks on is_fixed_width and call binned fucntions.
                        _abs_xsec, _emi_xsec = _abs_emi_sampled_voigt_layered(
                            wn_grid=wn_grid,
                            id_f=id_f,
                            id_i=id_i,
                            n_lookup=n_lookup,
                            g_lookup=g_lookup,
                            tau_lookup=tau_lookup,
                            a_fi=a_fi,
                            energy_fi=energy_fi,
                            temperatures=temperature_slice,
                            pressures=pressure_slice,
                            broad_n=broad_n,
                            broad_gamma=broad_gamma,
                            species_mass=species_mass,
                        )
                        abs_xsec += _abs_xsec
                        emi_xsec += _emi_xsec
            else:
                ddf = dd.read_csv(
                    trans_file,
                    sep=r"\s+",
                    engine="python",
                    header=None,
                    names=trans_columns,
                    usecols=[0, 1, 2],
                    dtype=dask_dtypes,
                    blocksize=dask_blocksize,
                )
                for delayed_batch in ddf.to_delayed():
                    trans_batch = pl.from_pandas(delayed_batch.compute())
                    trans_batch = (
                        trans_batch
                        .with_columns([
                            pl.col("id_i").cast(pl.Int32),
                            pl.col("id_f").cast(pl.Int32),
                        ])
                        .join(states_i, on="id_i", how="inner")
                        .join(states_f, on="id_f", how="inner")
                        .with_columns(
                            (pl.col("energy_f") - pl.col("energy_i")).alias("energy_fi")
                        )
                        .filter(
                            (pl.col("energy_fi") >= wn_min)
                            & (pl.col("energy_fi") <= wn_max)
                        )
                        .fill_null(strategy="zero")
                    )
                    if trans_batch.height == 0:
                        continue
                    # Get all contiguous arrays.
                    id_f = np.ascontiguousarray(trans_batch["id_f"].to_numpy())
                    id_i = np.ascontiguousarray(trans_batch["id_i"].to_numpy())
                    a_fi = np.ascontiguousarray(trans_batch["A_fi"].to_numpy())
                    energy_fi = np.ascontiguousarray(trans_batch["energy_fi"].to_numpy())

                    # All layer calculation.
                    # If we want to use integrated xsecs, replace with checks on is_fixed_width and call binned fucntions.
                    _abs_xsec, _emi_xsec = _abs_emi_sampled_voigt_layered(
                        wn_grid=wn_grid,
                        id_f=id_f,
                        id_i=id_i,
                        n_lookup=n_lookup,
                        g_lookup=g_lookup,
                        tau_lookup=tau_lookup,
                        a_fi=a_fi,
                        energy_fi=energy_fi,
                        temperatures=temperature_slice,
                        pressures=pressure_slice,
                        broad_n=broad_n,
                        broad_gamma=broad_gamma,
                        species_mass=species_mass,
                    )
                    abs_xsec += _abs_xsec
                    emi_xsec += _emi_xsec
    log.info(f"DEBUG: new rates/profiles duration = {time.perf_counter() - process_time:.3f}.")
    return abs_xsec, emi_xsec


def continuum_xsec(
        states: pl.DataFrame,
        continuum_states: pl.DataFrame,
        continuum_trans_files: t.List[pathlib.Path],
        n_lte_layers: int,
        n_nlte_layers: int,
        temperature_profile: npt.NDArray[np.float64],
        wn_grid: npt.NDArray[np.float64],
        species_mass: float,
        reduced_mass: float,
        cont_box_length: float,
) -> npt.NDArray[np.float64]:
    # TODO This whole thing is now wrong now, needs a complete overhaul.
    is_fixed_width = np.all(np.isclose(np.diff(wn_grid), np.abs(wn_grid[1] - wn_grid[0]), atol=0))
    abs_xsec = np.zeros(wn_grid.shape, dtype=np.float64)

    wn_min = wn_grid[0]
    wn_max = wn_grid[-1]

    n_cols = [f"n_nlte_L{n_lte_layers + nlte_idx}" for nlte_idx in range(n_nlte_layers)]
    select_cols = ["id"] + n_cols

    cont_states = continuum_states.join(states.select(select_cols), on="id", how="left")

    invariant_cols_i = ["id", "energy", "g", ] + n_cols
    invariant_cols_f = ["id", "energy", "g", "v"] + n_cols

    states_i = (
        cont_states
        .select(invariant_cols_i)
        .rename({col: f"{col}_i" for col in invariant_cols_i})
    )
    states_f = (
        cont_states
        .select(invariant_cols_f)
        .rename({col: f"{col}_f" for col in invariant_cols_f})
    )
    # Plain float arrays - no astropy units - passed directly to Numba.
    temperature_slice = temperature_profile[n_lte_layers:]  # (n_nlte_layers,)

    trans_columns = ["id_f", "id_i", "A_fi"]
    dask_dtypes = {"id_f": "int64", "id_i": "int64", "A_fi": "float64"}
    dask_blocksize = "256MB"

    process_time = time.perf_counter()
    for trans_file in continuum_trans_files:
        log.info(f"Processing file {trans_file}.")
        if str(trans_file).endswith(".parquet"):
            with pq.ParquetFile(trans_file) as pq_file, ThreadPoolExecutor(max_workers=1) as executor:
                num_rg = pq_file.num_row_groups

                def read_rg(idx):
                    return pl.from_arrow(
                        pq_file.read_row_group(idx, columns=trans_columns, use_threads=True)
                    )

                future = executor.submit(read_rg, 0)
                for rg_idx in range(num_rg):
                    trans_batch = future.result()
                    # Prefetch next batch.
                    if rg_idx + 1 < num_rg:
                        future = executor.submit(read_rg, rg_idx + 1)
                    trans_batch = (
                        trans_batch
                        .join(states_i, on="id_i", how="inner")
                        .join(states_f, on="id_f", how="inner")
                        .with_columns((pl.col("energy_f") - pl.col("energy_i")).alias("energy_fi"))
                        .filter(
                            # (pl.col("energy_f") >= wn_min)
                            # & (pl.col("energy_i") <= wn_max)
                            (pl.col("energy_fi") >= wn_min)
                            & (pl.col("energy_fi") <= wn_max)
                        )
                        .fill_null(strategy="zero")
                    )
                    if trans_batch.height == 0:
                        continue
                    # Get all contiguous arrays.
                    a_fi_all = np.ascontiguousarray(trans_batch["A_fi"].to_numpy())
                    g_i_all = np.ascontiguousarray(trans_batch["g_i"].to_numpy())
                    g_f_all = np.ascontiguousarray(trans_batch["g_f"].to_numpy())
                    energy_fi_all = np.ascontiguousarray(trans_batch["energy_fi"].to_numpy())
                    v_f_all = np.ascontiguousarray(trans_batch["v_f"].to_numpy())
                    # Stack per-layer population columns into (n_nlte_layers, n_trans) arrays.
                    n_i_all = np.ascontiguousarray(
                        np.stack(
                            [trans_batch[f"n_nlte_L{n_lte_layers + l}_i"].to_numpy() for l in range(n_nlte_layers)])
                    )
                    # All layer calculation.
                    # If we want to use integrated xsecs, replace with checks on is_fixed_width and call binned fucntions.
                    _abs_xsec, _emi_xsec = _continuum_band_profile_sampled_gauss_layered(
                        wn_grid=wn_grid,
                        n_i=n_i_all,
                        a_fi=a_fi_all,
                        g_f=g_f_all,
                        g_i=g_i_all,
                        energy_fi=energy_fi_all,
                        v_f=v_f_all,
                        temperatures=temperature_slice,
                        species_mass=species_mass,
                        box_length=cont_box_length,
                    )
                    abs_xsec += _abs_xsec
        else:
            ddf = dd.read_csv(
                trans_file,
                sep=r"\s+",
                engine="python",
                header=None,
                names=trans_columns,
                usecols=[0, 1, 2],
                dtype=dask_dtypes,
                blocksize=dask_blocksize,
            )
            for delayed_batch in ddf.to_delayed():
                trans_batch = pl.from_pandas(delayed_batch.compute())
                trans_batch = (
                    trans_batch
                    .join(states_i, on="id_i", how="inner")
                    .join(states_f, on="id_f", how="inner")
                    .with_columns((pl.col("energy_f") - pl.col("energy_i")).alias("energy_fi"))
                    .filter(
                        # (pl.col("energy_f") >= wn_min)
                        # & (pl.col("energy_i") <= wn_max)
                        (pl.col("energy_fi") >= wn_min)
                        & (pl.col("energy_fi") <= wn_max)
                    )
                    .fill_null(strategy="zero")
                )
                if trans_batch.height == 0:
                    continue
                # Get all contiguous arrays.
                a_fi_all = np.ascontiguousarray(trans_batch["A_fi"].to_numpy())
                g_i_all = np.ascontiguousarray(trans_batch["g_i"].to_numpy())
                g_f_all = np.ascontiguousarray(trans_batch["g_f"].to_numpy())
                energy_fi_all = np.ascontiguousarray(trans_batch["energy_fi"].to_numpy())
                v_f_all = np.ascontiguousarray(trans_batch["v_f"].to_numpy())
                # Stack per-layer population columns into (n_nlte_layers, n_trans) arrays.
                n_i_all = np.ascontiguousarray(
                    np.stack(
                        [trans_batch[f"n_nlte_L{n_lte_layers + l}_i"].to_numpy() for l in range(n_nlte_layers)])
                )
                # All layer calculation.
                # If we want to use integrated xsecs, replace with checks on is_fixed_width and call binned fucntions.
                _abs_xsec, _emi_xsec = _continuum_band_profile_sampled_gauss_layered(
                    wn_grid=wn_grid,
                    n_i=n_i_all,
                    a_fi=a_fi_all,
                    g_f=g_f_all,
                    g_i=g_i_all,
                    energy_fi=energy_fi_all,
                    v_f=v_f_all,
                    temperatures=temperature_slice,
                    species_mass=species_mass,
                    box_length=cont_box_length,
                )
                abs_xsec += _abs_xsec
    log.info(f"DEBUG: new rates/profiles duration = {time.perf_counter() - process_time:.3f}.")
    # for trans_file in continuum_trans_files:
    #     ddf = dd.read_csv(
    #         trans_file,
    #         sep=r"\s+",
    #         engine="python",
    #         header=None,
    #         names=["id_f", "id_i", "A_fi"],
    #         usecols=[0, 1, 2],
    #         dtype=dask_dtypes,
    #         blocksize=dask_blocksize,
    #     )
    #     delayed_batches = ddf.to_delayed()
    #     for delayed_batch in delayed_batches:
    #         trans_batch = pl.from_pandas(delayed_batch.compute())
    #         trans_batch = trans_batch.join(pl_states_i, left_on="id_i", right_on="id", how="left")
    #         trans_batch = trans_batch.join(pl_states_f, left_on="id_f", right_on="id", how="left")
    #         trans_batch = trans_batch.with_columns(
    #             (pl.col("energy_f") - pl.col("energy_i")).alias("energy_fi")
    #         )
    #         trans_batch = trans_batch.filter(
    #             (pl.col("energy_f") >= wn_min)
    #             & (pl.col("energy_i") <= wn_max)
    #             & (pl.col("energy_fi") >= wn_min)
    #             & (pl.col("energy_fi") <= wn_max)
    #         )
    #         final_cols = ["n_nlte_f", "n_nlte_i", "A_fi", "g_f", "g_i", "energy_fi", "v_f"]
    #         trans_batch = trans_batch.select(final_cols)
    #         trans_chunk_np = trans_batch.to_numpy()
    #
    #         # Matches final_cols ordering.
    #         n_f = np.ascontiguousarray(trans_chunk_np[:, 0])
    #         n_i = np.ascontiguousarray(trans_chunk_np[:, 1])
    #         a_fi = np.ascontiguousarray(trans_chunk_np[:, 2])
    #         g_f = np.ascontiguousarray(trans_chunk_np[:, 3])
    #         g_i = np.ascontiguousarray(trans_chunk_np[:, 4])
    #         energy_fi = np.ascontiguousarray(trans_chunk_np[:, 5])
    #         v_f = np.ascontiguousarray(trans_chunk_np[:, 6])
    #
    #         if is_fixed_width:
    #             half_bin_width = abs(wn_grid[1] - wn_grid[0]) / 2.0
    #             cont_xsec += _continuum_binned_gauss_fixed_width(
    #                 wn_grid=wn_grid,
    #                 n_f=n_f,
    #                 n_i=n_i,
    #                 a_fi=a_fi,
    #                 g_f=g_f,
    #                 g_i=g_i,
    #                 energy_fi=energy_fi,
    #                 v_f=v_f,
    #                 temperature=temperature,
    #                 species_mass=species_mass,
    #                 box_length=cont_box_length,
    #                 half_bin_width=half_bin_width,
    #             )
    #         else:
    #             cont_xsec += _continuum_binned_gauss_variable_width(
    #                 wn_grid=wn_grid,
    #                 n_f=n_f,
    #                 n_i=n_i,
    #                 a_fi=a_fi,
    #                 g_f=g_f,
    #                 g_i=g_i,
    #                 energy_fi=energy_fi,
    #                 v_f=v_f,
    #                 temperature=temperature,
    #                 species_mass=species_mass,
    #                 box_length=cont_box_length,
    #             )
    return abs_xsec


# ------------------------------------- NUMBA XSEC CALCULATIONS -------------------------------------


@numba.njit(cache=True, error_model="numpy", inline="always")
def _binary_search_left(arr: npt.NDArray[np.float64], value: float, start: int = 0) -> int:
    """Fast binary search for left insertion point with optional start hint."""
    left, right = start, len(arr)
    while left < right:
        mid = (left + right) >> 1
        if arr[mid] < value:
            left = mid + 1
        else:
            right = mid
    return left


@numba.njit(cache=True, error_model="numpy", inline="always")
def _binary_search_right(arr: npt.NDArray[np.float64], value: float, start: int = 0) -> int:
    """Fast binary search for right insertion point with optional start hint."""
    left, right = start, len(arr)
    while left < right:
        mid = (left + right) >> 1
        if arr[mid] <= value:
            left = mid + 1
        else:
            right = mid
    return left


# ------------------------------------- VOIGT HUMLICEK HELPER -------------------------------------

@numba.njit(cache=True, error_model="numpy", inline="always")
def _voigt_humlicek_w(x: float, y: float) -> float:
    """
    Real part of the Faddeeva function w(z) = exp(-z²) * erfc(-iz) via the Humlíček (1979) 4-region rational
    approximation.

    Relative accuracy ~1e-6. x and y are the real and imaginary parts of z = (nu - nu0) / (sigma_D * sqrt(2)), where
    y > 0 always (it is the ratio of Lorentzian to Gaussian halfwidths).

    Parameters
    ----------
    x : float
       (nu - nu0) / (sigma_D * sqrt(2)).
    y : float
       gamma_L / (sigma_D * sqrt(2)). Always positive.

    Returns
    -------
    float  Re[w(z)] - proportional to the Voigt profile at this point.
    """
    humlicek_t = y - 1j * x  # Numba supports complex arithmetic in njit

    # Region selection on |x| + y following Humlíček (1979).
    humlicek_s = abs(x) + y

    if humlicek_s >= 15.0:
        # Region 1: far wings, single-term approximation.
        humlicek_w = humlicek_t * 0.5641896 / (0.5 + humlicek_t * humlicek_t)
    elif humlicek_s >= 5.5:
        # Region 2.
        humlicek_u = humlicek_t * humlicek_t
        humlicek_w = humlicek_t * (1.410474 + humlicek_u * 0.5641896) / (0.75 + humlicek_u * (3.0 + humlicek_u))
    elif y >= 0.195 * abs(x) - 0.176:
        # Region 3: near line centre.
        humlicek_w = (16.4955 + humlicek_t * (
                20.20933 + humlicek_t * (11.96482 + humlicek_t * (3.778987 + humlicek_t * 0.5642236)))) / (
                             16.4955 + humlicek_t * (38.82363 + humlicek_t * (
                             39.27121 + humlicek_t * (21.69274 + humlicek_t * (6.699398 + humlicek_t))))
                     )
    else:
        # Region 4: intermediate, exponential term dominates.
        humlicek_u = humlicek_t * humlicek_t
        humlicek_w = np.exp(humlicek_u) - humlicek_t * (
                36183.31 - humlicek_u * (3321.99 - humlicek_u * (1540.787 - humlicek_u * (
                219.031 - humlicek_u * (35.7668 - humlicek_u * (1.320522 - humlicek_u * 0.56419)))))
        ) / (
                             32066.6 - humlicek_u * (24322.84 - humlicek_u * (
                             9022.228 - humlicek_u * (2186.181 - humlicek_u * (
                             364.2191 - humlicek_u * (61.57037 - humlicek_u * (1.841439 - humlicek_u))))))
                     )

    return humlicek_w.real


# ------------------------------------------- BAND PROFILE ACCUMULATORS -------------------------------------------
@numba.njit(parallel=True, cache=True, error_model="numpy")
def _broaden_superline_buffer(
        buffer_abs: npt.NDArray[np.float64],
        buffer_emi: npt.NDArray[np.float64],
        wn_grid: npt.NDArray[np.float64],
        temperatures: npt.NDArray[np.float64],
        pressures: npt.NDArray[np.float64],
        broad_n: npt.NDArray[np.float64],
        broad_gamma: npt.NDArray[np.float64],
        species_mass: float,
        t_ref=296.0,
        pressure_ref=1.0,
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Parameters
    ----------
    buffer_abs : (n_layers, n_grid)
        Accumulated absorption super-line coefficients per band per layer per grid bin.
    buffer_emi : (n_layers, n_grid)
        Accumulated emission super-line coefficients per band per layer per grid bin.
    wn_grid : ndarray, shape (n_grid,)
    temperatures : ndarray, shape (n_layers,)
    pressures : ndarray, shape (n_layers,)
    broad_n : ndarray, shape (n_broadeners,)
    broad_gamma : ndarray, shape (n_broadeners, n_layers)
        Note: layers on axis-1, matching broadening_params[0].
    species_mass : float

    Returns
    -------
    out_abs, out_emi : each (n_layers, n_grid)
    """
    n_layers = temperatures.shape[0]
    n_grid = wn_grid.shape[0]
    n_broad = broad_n.shape[0]
    cutoff = 25.0

    out_abs = np.zeros((n_layers, n_grid), dtype=np.float64)
    out_emi = np.zeros((n_layers, n_grid), dtype=np.float64)

    sqrt2 = np.sqrt(2.0)
    sqrt2_NA_kB_on_c = sqrt2 * const_sqrt_NA_kB_on_c
    inv_sqrt_pi = 1 / np.sqrt(np.pi)

    inv_sigma_sqrt2 = np.empty((n_grid, n_layers), dtype=np.float64)
    # gamma_total = np.empty((n_layers, n_grid), dtype=np.float64)
    gamma_total = np.empty((n_layers,), dtype=np.float64)
    # Voigt y-parameter: gamma_L / (sigma_D * sqrt(2)).
    # sigma here is the Gaussian sigma (standard deviation), so
    # sigma_D * sqrt(2) = sigma * sqrt(2).
    y_voigt = np.empty((n_grid, n_layers), dtype=np.float64)
    # gamma_lifetime = 1.0 / (const_4_pi_c * lifetimes)  # (n_trans,)

    for l in range(n_layers):
        temp_l = temperatures[l]
        pres_l = pressures[l]
        inv_sigma_sqrt2_l = 1 / (wn_grid * sqrt2_NA_kB_on_c * np.sqrt(temp_l / species_mass))
        inv_sigma_sqrt2[:, l] = inv_sigma_sqrt2_l

        gamma_pressure_l = 0.0
        for b in range(n_broad):
            gamma_pressure_l += broad_gamma[b, l] * pres_l * (t_ref / temp_l) ** broad_n[b] / pressure_ref
        # gamma_total[l] = gamma_lifetime + gamma_pressure_l
        gamma_total[l] = gamma_pressure_l
        y_voigt[:, l] = gamma_total[l] * inv_sigma_sqrt2_l

    # Bin edges for natural linewidth from accumulated A_fi.
    # gamma_lifetime[band, bin] = A_fi_sum / (4 * pi * c) — approximation for super-line.
    # Only needed for occupied bins so we compute inline below.

    # Precompute occupation mask: True if any layer has nonzero coefficients.
    occupied = np.zeros(n_grid, dtype=numba.boolean)
    for i in range(n_grid):
        for l in range(n_layers):
            # Loop and l/i index ordering could maybe be changed.
            if buffer_abs[l, i] != 0.0 or buffer_emi[l, i] != 0.0:
                occupied[i] = True
                break

    # As in other implementations, "i" tracks transitions.
    for i in numba.prange(n_grid):
        # Super-line centre is the bin centre.
        if not occupied[i]:
            continue
        energy_i = wn_grid[i]
        transition_min = energy_i - cutoff
        transition_max = energy_i + cutoff
        j_start = max(0, _binary_search_left(wn_grid, transition_min))
        j_end = min(n_grid, _binary_search_right(wn_grid, transition_max) + 1)

        for l in range(n_layers):
            abs_li = buffer_abs[l, i]
            emi_li = buffer_emi[l, i]
            # Skip empty bins.
            if abs_li == 0.0 and emi_li == 0.0:
                continue
            inv_sigma_sqrt2_il = inv_sigma_sqrt2[i, l]
            y_il = y_voigt[i, l]
            # Integral of Re[w(x,y)] dx = sqrt(pi), so the normalised Voigt is Re[w(z)] / (sigma * sqrt(2*pi)).
            norm = inv_sigma_sqrt2_il * inv_sqrt_pi

            for j in range(j_start, j_end):
                wn_j = wn_grid[j]
                x_ij = (wn_j - energy_i) * inv_sigma_sqrt2_il
                voigt_val = _voigt_humlicek_w(x_ij, y_il) * norm

                out_abs[l, j] += abs_li * voigt_val
                out_emi[l, j] += emi_li * voigt_val

    return out_abs, out_emi


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _broaden_superline_band_buffer(
        profile_buffer: npt.NDArray[np.float64],
        wn_grid: npt.NDArray[np.float64],
        temperatures: npt.NDArray[np.float64],
        pressures: npt.NDArray[np.float64],
        broad_n: npt.NDArray[np.float64],
        broad_gamma: npt.NDArray[np.float64],
        species_mass: float,
        t_ref=296.0,
        pressure_ref=1.0,
) -> npt.NDArray[np.float64]:
    """
    Parameters
    ----------
    profile_buffer : ndarray, shape (3, n_bands, n_layers, n_trans)
    wn_grid : ndarray, shape (n_grid,)
    temperatures : ndarray, shape (n_layers,)
    pressures : ndarray, shape (n_layers,)
    broad_n : ndarray, shape (n_broadeners,)
    broad_gamma : ndarray, shape (n_broadeners, n_layers)
        Note: layers on axis-1, matching broadening_params[0].
    species_mass : float
    # n_bands_used : int
    #     How many entries in axis-0 of the buffers actually contain data.

    Returns
    -------
    out_buffer : ndarray, shape (3, n_bands, n_layers, n_grid)
    """
    n_layers = temperatures.shape[0]
    n_grid = wn_grid.shape[0]
    n_broad = broad_n.shape[0]
    n_bands = profile_buffer.shape[1]
    cutoff = 25.0

    buffer_out = np.zeros_like(profile_buffer, dtype=np.float64)

    sqrt2 = np.sqrt(2.0)
    sqrt2_NA_kB_on_c = sqrt2 * const_sqrt_NA_kB_on_c
    inv_sqrt_pi = 1 / np.sqrt(np.pi)

    inv_sigma_sqrt2 = np.empty((n_grid, n_layers), dtype=np.float64)
    # gamma_total = np.empty((n_layers, n_grid), dtype=np.float64)
    gamma_total = np.empty((n_layers,), dtype=np.float64)
    # Voigt y-parameter: gamma_L / (sigma_D * sqrt(2)).
    # sigma here is the Gaussian sigma (standard deviation), so
    # sigma_D * sqrt(2) = sigma * sqrt(2).
    y_voigt = np.empty((n_grid, n_layers), dtype=np.float64)
    # gamma_lifetime = 1.0 / (const_4_pi_c * lifetimes)  # (n_trans,)

    for l in range(n_layers):
        temp_l = temperatures[l]
        pres_l = pressures[l]
        inv_sigma_sqrt2_l = 1 / (wn_grid * sqrt2_NA_kB_on_c * np.sqrt(temp_l / species_mass))
        inv_sigma_sqrt2[:, l] = inv_sigma_sqrt2_l

        gamma_pressure_l = 0.0
        for b in range(n_broad):
            gamma_pressure_l += broad_gamma[b, l] * pres_l * (t_ref / temp_l) ** broad_n[b] / pressure_ref
        # gamma_total[l] = gamma_lifetime + gamma_pressure_l
        gamma_total[l] = gamma_pressure_l
        y_voigt[:, l] = gamma_total[l] * inv_sigma_sqrt2_l

    # Bin edges for natural linewidth from accumulated A_fi.
    # gamma_lifetime[band, bin] = A_fi_sum / (4 * pi * c) — approximation for super-line.
    # Only needed for occupied bins so we compute inline below.

    # Precompute occupation mask: True if any layer has nonzero coefficients.
    occupied = np.zeros((n_bands, n_grid), dtype=numba.boolean)
    for band in numba.prange(n_bands):
        for i in range(n_grid):
            for l in range(n_layers):
                # Loop and l/i index ordering could maybe be changed.
                if profile_buffer[0, band, l, i] != 0.0 or profile_buffer[1, band, l, i] != 0.0 or profile_buffer[
                    2, band, l, i] != 0.0:
                    occupied[band, i] = True
                    break

    # Parallelise over bands — each band is fully independent.
    for band in numba.prange(n_bands):
        # As in other implementations, "i" tracks transitions.
        for i in range(n_grid):
            # Super-line centre is the bin centre.
            if not occupied[band, i]:
                continue
            energy_i = wn_grid[i]
            transition_min = energy_i - cutoff
            transition_max = energy_i + cutoff
            j_start = max(0, _binary_search_left(wn_grid, transition_min))
            j_end = min(n_grid, _binary_search_right(wn_grid, transition_max) + 1)

            for l in range(n_layers):
                abs_li = profile_buffer[0, band, l, i]
                ste_li = profile_buffer[1, band, l, i]
                spe_li = profile_buffer[2, band, l, i]
                # Skip empty bins.
                if abs_li == 0.0 and ste_li == 0.0 and spe_li == 0.0:
                    continue
                inv_sigma_sqrt2_il = inv_sigma_sqrt2[i, l]
                y_il = y_voigt[i, l]
                # Integral of Re[w(x,y)] dx = sqrt(pi), so the normalised Voigt is Re[w(z)] / (sigma * sqrt(2*pi)).
                norm = inv_sigma_sqrt2_il * inv_sqrt_pi

                for j in range(j_start, j_end):
                    wn_j = wn_grid[j]
                    x_ij = (wn_j - energy_i) * inv_sigma_sqrt2_il
                    voigt_val = _voigt_humlicek_w(x_ij, y_il) * norm

                    buffer_out[0, band, l, j] += abs_li * voigt_val
                    buffer_out[1, band, l, j] += ste_li * voigt_val
                    buffer_out[2, band, l, j] += spe_li * voigt_val

    return buffer_out


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _broaden_continuum_superline_band_buffer(
        profile_buffer: npt.NDArray[np.float64],
        wn_grid: npt.NDArray[np.float64],
        temperatures: npt.NDArray[np.float64],
        species_mass: float,
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Parameters
    ----------
    profile_buffer : ndarray, shape (2, n_bands, n_layers, n_trans)
    wn_grid : ndarray, shape (n_grid,)
    temperatures : ndarray, shape (n_layers,)
    species_mass : float

    Returns
    -------
    out_buffer : ndarray, shape (1, n_bands, n_layers, n_grid)
    """
    n_layers = temperatures.shape[0]
    n_grid = wn_grid.shape[0]
    n_bands = profile_buffer.shape[1]
    min_cutoff = 25.0
    max_cutoff = 5000.0
    cutoff_fwhm_multiple = 5.0

    n_layers_buffer = profile_buffer.shape[2]  # Buffer only tracks varying NLTE layers.
    buffer_out = np.zeros_like((1, n_bands, n_layers_buffer, n_grid), dtype=np.float64)

    sqrtln2 = np.sqrt(np.log(2.0))
    inv_sqrt_pi = 1 / np.sqrt(np.pi)

    doppler_prefactor = np.empty((n_layers,), dtype=np.float64)

    inv_mass = 1 / species_mass
    doppler_coef = const_sqrt_2_NA_kB_log2_on_c * math.sqrt(inv_mass)
    temp_max = temperatures[0]
    for l in range(n_layers):
        temp_l = temperatures[l]
        doppler_prefactor[l] = doppler_coef * math.sqrt(temp_l)
        if temp_l > temp_max:
            temp_max = temp_l

    doppler_prefactor_max = doppler_coef * np.sqrt(temp_max)

    # Precompute occupation mask: True if any layer has nonzero coefficients.
    occupied = np.zeros((n_bands, n_grid), dtype=numba.boolean)
    for band in numba.prange(n_bands):
        for i in range(n_grid):
            for l in range(n_layers):
                # Loop and l/i index ordering could maybe be changed.
                if profile_buffer[0, band, l, i] != 0.0:
                    occupied[band, i] = True
                    break

    # Parallelise over bands — each band is fully independent.
    for band in numba.prange(n_bands):
        # As in other implementations, "i" tracks transitions.
        for i in range(n_grid):
            # Super-line centre is the bin centre.
            if not occupied[band, i]:
                continue
            energy_i = wn_grid[i]

            for l in range(n_layers):
                abs_li = profile_buffer[0, band, l, i]
                # Skip empty bins.
                if abs_li == 0.0:
                    continue

                weighted_broad = profile_buffer[1, band, l, i]
                alpha_box_li = weighted_broad / abs_li
                doppler_prefactor_l = doppler_prefactor[l]
                alpha_doppler_li = doppler_prefactor_l * energy_i
                alpha_total_li = alpha_box_li + alpha_doppler_li
                cutoff_li = cutoff_fwhm_multiple * alpha_total_li
                if cutoff_li < min_cutoff:
                    cutoff_li = min_cutoff
                elif cutoff_li > max_cutoff:
                    cutoff_li = max_cutoff

                transition_min = energy_i - cutoff_li
                transition_max = energy_i + cutoff_li
                j_start = max(0, _binary_search_left(wn_grid, transition_min))
                j_end = min(n_grid, _binary_search_right(wn_grid, transition_max) + 1)

                if j_start >= j_end:
                    continue

                sqrtln2_on_alpha_li = sqrtln2 / alpha_total_li
                norm = sqrtln2_on_alpha_li * inv_sqrt_pi

                for j in range(j_start, j_end):
                    wn_j = wn_grid[j]
                    arg = sqrtln2_on_alpha_li * (wn_j - energy_i)
                    gauss_val = norm * math.exp(-(arg * arg))

                    buffer_out[0, band, l, j] += abs_li * gauss_val

    return buffer_out


# ------------------------------------- ALL LAYER, BINNED VOIGT CALCULATIONS -------------------------------------

@numba.njit(parallel=True, cache=True, error_model="numpy")
def _band_profile_binned_voigt_variable_width_layered(
        wn_grid: npt.NDArray[np.float64],
        n_i: npt.NDArray[np.float64],  # (n_layers, n_trans)
        n_f: npt.NDArray[np.float64],  # (n_layers, n_trans)
        a_fi: npt.NDArray[np.float64],  # (n_trans,)
        g_f: npt.NDArray[np.float64],  # (n_trans,)
        g_i: npt.NDArray[np.float64],  # (n_trans,)
        energy_fi: npt.NDArray[np.float64],  # (n_trans,)
        lifetimes: npt.NDArray[np.float64],  # (n_trans,)
        temperatures: npt.NDArray[np.float64],  # (n_layers,)
        pressures: npt.NDArray[np.float64],  # (n_layers,)
        broad_n: npt.NDArray[np.float64],  # (n_broadeners,)
        broad_gamma: npt.NDArray[np.float64],  # (n_broadeners, n_layers)
        species_mass: float,
        gh_roots: npt.NDArray[np.float64],
        gh_weights: npt.NDArray[np.float64],
        t_ref: float = 296.0,
        pressure_ref: float = 1.0,
        numba_num_threads: int = _DEFAULT_NUM_THREADS,
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Vectorised multi-layer variant of _band_profile_binned_voigt_variable_width.

    Computes absorption, stimulated emission, and spontaneous emission profiles for all layers simultaneously. The
    bin_term inner loop - the dominant cost - is computed once per transition per grid point and reused across all 
    layers, with only the per-layer coefficients (which depend on populations, temperature, and pressure) applied on 
    top.

    Parameters
    ----------
    wn_grid : ndarray, shape (n_grid,)
    n_i : ndarray, shape (n_layers, n_trans)
       LTE population fractions of lower states, per layer.
    n_f : ndarray, shape (n_layers, n_trans)
       LTE population fractions of upper states, per layer.
    a_fi : ndarray, shape (n_trans,)
    g_f : ndarray, shape (n_trans,)
    g_i : ndarray, shape (n_trans,)
    energy_fi : ndarray, shape (n_trans,)
    lifetimes : ndarray, shape (n_trans,)
    temperatures : ndarray, shape (n_layers,)
    pressures : ndarray, shape (n_layers,)
    broad_n : ndarray, shape (n_broadeners,)
    broad_gamma : ndarray, shape (n_broadeners, n_layers)
        Note: layers on axis-1, matching broadening_params[0].
    species_mass : float
    gh_roots : ndarray, shape (n_gh,)
    gh_weights : ndarray, shape (n_gh,)

    Returns
    -------
    abs_xsec : ndarray, shape (n_layers, n_grid)
    ste_xsec : ndarray, shape (n_layers, n_grid)
    spe_xsec : ndarray, shape (n_layers, n_grid)
    """
    n_layers = temperatures.shape[0]
    num_grid = wn_grid.shape[0]
    num_trans = energy_fi.shape[0]
    num_broad = broad_n.shape[0]
    cutoff = 25

    # Bin geometry - identical to the 1-D version, computed once.
    bin_edges = np.empty(num_grid + 1, dtype=np.float64)
    bin_edges[0] = wn_grid[0] - (wn_grid[1] - wn_grid[0]) * 0.5
    for j in range(1, num_grid):
        bin_edges[j] = (wn_grid[j - 1] + wn_grid[j]) * 0.5
    bin_edges[-1] = wn_grid[-1] + (wn_grid[-1] - wn_grid[-2]) * 0.5

    bin_widths_lower = wn_grid - bin_edges[:-1]
    bin_widths_upper = bin_edges[1:] - wn_grid
    inv_bin_widths = 1.0 / (bin_widths_lower + bin_widths_upper)

    # Per-layer scalars: sigma (Doppler width) and gamma (total damping), vary with T/P.
    sigma = np.empty((n_layers, num_trans), dtype=np.float64)
    gamma_total = np.empty((n_layers, num_trans), dtype=np.float64)
    gamma_lifetime = 1.0 / (const_4_pi_c * lifetimes)  # (n_trans,)  layer-invariant
    # Per-layer, per-transition profile coefficients (n_f/n_i vary per layer).
    abs_coef = np.empty((n_layers, num_trans), dtype=np.float64)
    ste_coef = np.empty((n_layers, num_trans), dtype=np.float64)
    spe_coef = np.empty((n_layers, num_trans), dtype=np.float64)

    for l in range(n_layers):
        temp_l = temperatures[l]
        pres_l = pressures[l]
        sigma[l] = energy_fi * const_sqrt_NA_kB_on_c * np.sqrt(temp_l / species_mass)

        # Scalar pressure-broadening sum for this layer.
        gamma_pressure_l = 0.0
        for b in range(num_broad):
            gamma_pressure_l += broad_gamma[b, l] * pres_l * (t_ref / temp_l) ** broad_n[b] / pressure_ref
        gamma_total[l] = gamma_lifetime + gamma_pressure_l

        # Bake the 1 / (pi^(3/2)) from integral into the coefficients.
        abs_coef[l] = a_fi * (n_i[l] * g_f / g_i) / (const_8_pi_five_halves_c * energy_fi * energy_fi)
        ste_coef[l] = a_fi * n_f[l] / (const_8_pi_five_halves_c * energy_fi * energy_fi)
        spe_coef[l] = n_f[l] * a_fi * energy_fi * const_h_c_on_4_pi_five_halves

    gamma_inv = 1.0 / gamma_total  # (n_layers, n_trans)

    # Thread-local output buffers.
    abs_xsec_buffer = np.zeros((numba_num_threads, n_layers, num_grid), dtype=np.float64)
    ste_xsec_buffer = np.zeros((numba_num_threads, n_layers, num_grid), dtype=np.float64)
    spe_xsec_buffer = np.zeros((numba_num_threads, n_layers, num_grid), dtype=np.float64)

    grid_min = wn_grid[0]
    grid_max = wn_grid[-1]

    # ------------------------------------------------------------------
    # Main loop: parallelised over transitions, vectorised over layers.
    #
    # The bin_term for a given (transition i, grid point j) depends only on energy_fi[i], sigma[l,i], gamma_inv[l,i],
    # and the bin geometry. sigma and gamma_inv are layer-dependent, so bin_term is computed per layer. This is still a
    # net win: the binary search bounds (j_start, j_end) are computed from energy_fi alone and shared across all layers,
    # avoiding redundant range computation.
    # ------------------------------------------------------------------
    for i in numba.prange(num_trans):
        thread_id = numba.get_thread_id()

        energy_fi_i = energy_fi[i]
        transition_min = energy_fi_i - cutoff
        transition_max = energy_fi_i + cutoff

        # Grid range is the same for all layers - compute once.
        j_start = _binary_search_right(bin_edges, transition_min) - 1
        j_start = max(0, j_start)
        j_end = _binary_search_left(bin_edges, transition_max, start=j_start)
        j_end = min(num_grid, j_end)

        if j_start >= j_end:
            continue

        for l in range(n_layers):
            sigma_il = sigma[l, i]
            gamma_inv_il = gamma_inv[l, i]
            abs_coef_il = abs_coef[l, i]
            ste_coef_il = ste_coef[l, i]
            spe_coef_il = spe_coef[l, i]

            gh_roots_sigma = gh_roots * sigma_il
            start_sigma = grid_min - gh_roots_sigma
            end_sigma = grid_max - gh_roots_sigma
            b_corr = np.pi / (
                    np.arctan((end_sigma - energy_fi_i) * gamma_inv_il)
                    - np.arctan((start_sigma - energy_fi_i) * gamma_inv_il)
            )
            gh_weights_b_corr = gh_weights * b_corr

            for j in range(j_start, j_end):
                upper_width_j = bin_widths_upper[j]
                lower_width_j = bin_widths_lower[j]
                inv_bin_width_j = inv_bin_widths[j]
                wn_j = wn_grid[j]

                shift_sigma = wn_j - energy_fi_i - gh_roots_sigma
                bin_term = np.sum(
                    gh_weights_b_corr
                    * (
                            np.arctan((shift_sigma + upper_width_j) * gamma_inv_il)
                            - np.arctan((shift_sigma - lower_width_j) * gamma_inv_il)
                    )
                ) * inv_bin_width_j

                abs_xsec_buffer[thread_id, l, j] += abs_coef_il * bin_term
                ste_xsec_buffer[thread_id, l, j] += ste_coef_il * bin_term
                spe_xsec_buffer[thread_id, l, j] += spe_coef_il * bin_term

    # Accumulate thread buffers into final output.
    abs_xsec = np.zeros((n_layers, num_grid), dtype=np.float64)
    ste_xsec = np.zeros((n_layers, num_grid), dtype=np.float64)
    spe_xsec = np.zeros((n_layers, num_grid), dtype=np.float64)

    for l in numba.prange(n_layers):
        for k in range(num_grid):
            abs_point = 0.0
            ste_point = 0.0
            spe_point = 0.0
            for t in range(numba_num_threads):
                abs_point += abs_xsec_buffer[t, l, k]
                ste_point += ste_xsec_buffer[t, l, k]
                spe_point += spe_xsec_buffer[t, l, k]
            abs_xsec[l, k] = abs_point
            ste_xsec[l, k] = ste_point
            spe_xsec[l, k] = spe_point

    return abs_xsec, ste_xsec, spe_xsec


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _band_profile_binned_voigt_fixed_width_layered(
        wn_grid: npt.NDArray[np.float64],
        n_i: npt.NDArray[np.float64],  # (n_layers, n_trans)
        n_f: npt.NDArray[np.float64],  # (n_layers, n_trans)
        a_fi: npt.NDArray[np.float64],  # (n_trans,)
        g_f: npt.NDArray[np.float64],  # (n_trans,)
        g_i: npt.NDArray[np.float64],  # (n_trans,)
        energy_fi: npt.NDArray[np.float64],  # (n_trans,)
        lifetimes: npt.NDArray[np.float64],  # (n_trans,)
        temperatures: npt.NDArray[np.float64],  # (n_layers,)
        pressures: npt.NDArray[np.float64],  # (n_layers,)
        broad_n: npt.NDArray[np.float64],  # (n_broadeners,)
        broad_gamma: npt.NDArray[np.float64],  # (n_broadeners, n_layers)
        species_mass: float,
        half_bin_width: float,
        gh_roots: npt.NDArray[np.float64],
        gh_weights: npt.NDArray[np.float64],
        t_ref: float = 296.0,
        pressure_ref: float = 1.0,
        numba_num_threads: int = _DEFAULT_NUM_THREADS,
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Vectorised multi-layer variant of _band_profile_binned_voigt_variable_width.

    Computes absorption, stimulated emission, and spontaneous emission profiles
    for all layers simultaneously. The bin_term inner loop - the dominant cost -
    is computed once per transition per grid point and reused across all layers,
    with only the per-layer coefficients (which depend on populations,
    temperature, and pressure) applied on top.

    Parameters
    ----------
    wn_grid : ndarray, shape (n_grid,)
    n_i : ndarray, shape (n_layers, n_trans)
       LTE population fractions of lower states, per layer.
    n_f : ndarray, shape (n_layers, n_trans)
       LTE population fractions of upper states, per layer.
    a_fi : ndarray, shape (n_trans,)
    g_f : ndarray, shape (n_trans,)
    g_i : ndarray, shape (n_trans,)
    energy_fi : ndarray, shape (n_trans,)
    lifetimes : ndarray, shape (n_trans,)
    temperatures : ndarray, shape (n_layers,)
    pressures : ndarray, shape (n_layers,)
    broad_n : ndarray, shape (n_broadeners,)
    broad_gamma : ndarray, shape (n_broadeners, n_layers)
        Note: layers on axis-1, matching broadening_params[0].
    species_mass : float
    gh_roots : ndarray, shape (n_gh,)
    gh_weights : ndarray, shape (n_gh,)

    Returns
    -------
    abs_xsec : ndarray, shape (n_layers, n_grid)
    ste_xsec : ndarray, shape (n_layers, n_grid)
    spe_xsec : ndarray, shape (n_layers, n_grid)
    """
    n_layers = temperatures.shape[0]
    num_grid = wn_grid.shape[0]
    num_trans = energy_fi.shape[0]
    num_broad = broad_n.shape[0]
    bin_width = 2.0 * half_bin_width
    cutoff = 25 + half_bin_width

    # Bin geometry - identical to the 1-D version, computed once.
    bin_edges = np.empty(num_grid + 1, dtype=np.float64)
    bin_edges[0] = wn_grid[0] - (wn_grid[1] - wn_grid[0]) * 0.5
    for j in range(1, num_grid):
        bin_edges[j] = (wn_grid[j - 1] + wn_grid[j]) * 0.5
    bin_edges[-1] = wn_grid[-1] + (wn_grid[-1] - wn_grid[-2]) * 0.5

    # Per-layer scalars: sigma (Doppler width) and gamma (total damping), vary with T/P.
    sigma = np.empty((n_layers, num_trans), dtype=np.float64)
    gamma_total = np.empty((n_layers, num_trans), dtype=np.float64)
    gamma_lifetime = 1.0 / (const_4_pi_c * lifetimes)  # (n_trans,)  layer-invariant
    # Per-layer, per-transition profile coefficients (n_f/n_i vary per layer).
    abs_coef = np.empty((n_layers, num_trans), dtype=np.float64)
    ste_coef = np.empty((n_layers, num_trans), dtype=np.float64)
    spe_coef = np.empty((n_layers, num_trans), dtype=np.float64)

    for l in range(n_layers):
        temp_l = temperatures[l]
        pres_l = pressures[l]
        sigma[l] = energy_fi * const_sqrt_NA_kB_on_c * np.sqrt(temp_l / species_mass)

        # Scalar pressure-broadening sum for this layer.
        gamma_pressure_l = 0.0
        for b in range(num_broad):
            gamma_pressure_l += broad_gamma[b, l] * pres_l * (t_ref / temp_l) ** broad_n[b] / pressure_ref
        gamma_total[l] = gamma_lifetime + gamma_pressure_l

        # Bake the 1 / (pi^(3/2) * bin_width) from integral into the coefficients.
        abs_coef[l] = a_fi * (n_i[l] * g_f / g_i) / (const_8_pi_five_halves_c * bin_width * energy_fi * energy_fi)
        ste_coef[l] = a_fi * n_f[l] / (const_8_pi_five_halves_c * bin_width * energy_fi * energy_fi)
        spe_coef[l] = n_f[l] * a_fi * energy_fi * const_h_c_on_4_pi_five_halves / bin_width

    gamma_inv = 1.0 / gamma_total  # (n_layers, n_trans)

    # Thread-local output buffers.
    abs_xsec_buffer = np.zeros((numba_num_threads, n_layers, num_grid), dtype=np.float64)
    ste_xsec_buffer = np.zeros((numba_num_threads, n_layers, num_grid), dtype=np.float64)
    spe_xsec_buffer = np.zeros((numba_num_threads, n_layers, num_grid), dtype=np.float64)

    grid_min = wn_grid[0]
    grid_max = wn_grid[-1]

    for i in numba.prange(num_trans):
        thread_id = numba.get_thread_id()

        energy_fi_i = energy_fi[i]
        transition_min = energy_fi_i - cutoff
        transition_max = energy_fi_i + cutoff

        # Grid range is the same for all layers - compute once.
        j_start = _binary_search_right(bin_edges, transition_min) - 1
        j_start = max(0, j_start)
        j_end = _binary_search_left(bin_edges, transition_max, start=j_start)
        j_end = min(num_grid, j_end)

        if j_start >= j_end:
            continue

        for l in range(n_layers):
            sigma_il = sigma[l, i]
            gamma_inv_il = gamma_inv[l, i]
            abs_coef_il = abs_coef[l, i]
            ste_coef_il = ste_coef[l, i]
            spe_coef_il = spe_coef[l, i]

            gh_roots_sigma = gh_roots * sigma_il
            start_sigma = grid_min - gh_roots_sigma
            end_sigma = grid_max - gh_roots_sigma
            b_corr = np.pi / (
                    np.arctan((end_sigma - energy_fi_i) * gamma_inv_il)
                    - np.arctan((start_sigma - energy_fi_i) * gamma_inv_il)
            )
            gh_weights_b_corr = gh_weights * b_corr

            for j in range(j_start, j_end):
                wn_j = wn_grid[j]

                shift_sigma = wn_j - energy_fi_i - gh_roots_sigma
                bin_term = np.sum(
                    gh_weights_b_corr
                    * (
                            np.arctan((shift_sigma + half_bin_width) * gamma_inv_il)
                            - np.arctan((shift_sigma - half_bin_width) * gamma_inv_il)
                    )
                )

                abs_xsec_buffer[thread_id, l, j] += abs_coef_il * bin_term
                ste_xsec_buffer[thread_id, l, j] += ste_coef_il * bin_term
                spe_xsec_buffer[thread_id, l, j] += spe_coef_il * bin_term

    # Accumulate thread buffers into final output.
    abs_xsec = np.zeros((n_layers, num_grid), dtype=np.float64)
    ste_xsec = np.zeros((n_layers, num_grid), dtype=np.float64)
    spe_xsec = np.zeros((n_layers, num_grid), dtype=np.float64)

    for l in numba.prange(n_layers):
        for k in range(num_grid):
            abs_point = 0.0
            ste_point = 0.0
            spe_point = 0.0
            for t in range(numba_num_threads):
                abs_point += abs_xsec_buffer[t, l, k]
                ste_point += ste_xsec_buffer[t, l, k]
                spe_point += spe_xsec_buffer[t, l, k]
            abs_xsec[l, k] = abs_point
            ste_xsec[l, k] = ste_point
            spe_xsec[l, k] = spe_point

    return abs_xsec, ste_xsec, spe_xsec


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _abs_emi_binned_voigt_variable_width_layered(
        wn_grid: npt.NDArray[np.float64],
        n_i: npt.NDArray[np.float64],  # (n_layers, n_trans)
        n_f: npt.NDArray[np.float64],  # (n_layers, n_trans)
        a_fi: npt.NDArray[np.float64],  # (n_trans,)
        g_f: npt.NDArray[np.float64],  # (n_trans,)
        g_i: npt.NDArray[np.float64],  # (n_trans,)
        energy_fi: npt.NDArray[np.float64],  # (n_trans,)
        lifetimes: npt.NDArray[np.float64],  # (n_trans,)
        temperatures: npt.NDArray[np.float64],  # (n_layers,)
        pressures: npt.NDArray[np.float64],  # (n_layers,)
        broad_n: npt.NDArray[np.float64],  # (n_broadeners,)
        broad_gamma: npt.NDArray[np.float64],  # (n_broadeners, n_layers)
        species_mass: float,
        gh_roots: npt.NDArray[np.float64],
        gh_weights: npt.NDArray[np.float64],
        t_ref: float = 296.0,
        pressure_ref: float = 1.0,
        numba_num_threads: int = _DEFAULT_NUM_THREADS,
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Computes absorption (including stimulated emission) and spontaneous emission profiles for all layers simultaneously.
    The bin_term inner loop - the dominant cost - is computed once per transition per grid point and reused across all
    layers, with only the per-layer coefficients (which depend on populations, temperature, and pressure) applied on
    top.

    Used for wn_grid inputs with variable grid point spacing.

    Parameters
    ----------
    wn_grid : ndarray, shape (n_grid,)
    n_i : ndarray, shape (n_layers, n_trans)
       LTE population fractions of lower states, per layer.
    n_f : ndarray, shape (n_layers, n_trans)
       LTE population fractions of upper states, per layer.
    a_fi : ndarray, shape (n_trans,)
    g_f : ndarray, shape (n_trans,)
    g_i : ndarray, shape (n_trans,)
    energy_fi : ndarray, shape (n_trans,)
    lifetimes : ndarray, shape (n_trans,)
    temperatures : ndarray, shape (n_layers,)
    pressures : ndarray, shape (n_layers,)
    broad_n : ndarray, shape (n_broadeners,)
    broad_gamma : ndarray, shape (n_broadeners, n_layers)
        Note: layers on axis-1, matching broadening_params[0].
    species_mass : float
    gh_roots : ndarray, shape (n_gh,)
    gh_weights : ndarray, shape (n_gh,)

    Returns
    -------
    abs_xsec : ndarray, shape (n_layers, n_grid)
    emi_xsec : ndarray, shape (n_layers, n_grid)
    """
    n_layers = temperatures.shape[0]
    num_grid = wn_grid.shape[0]
    num_trans = energy_fi.shape[0]
    num_broad = broad_n.shape[0]
    cutoff = 25

    # Bin geometry - identical to the 1-D version, computed once.
    bin_edges = np.empty(num_grid + 1, dtype=np.float64)
    bin_edges[0] = wn_grid[0] - (wn_grid[1] - wn_grid[0]) * 0.5
    for j in range(1, num_grid):
        bin_edges[j] = (wn_grid[j - 1] + wn_grid[j]) * 0.5
    bin_edges[-1] = wn_grid[-1] + (wn_grid[-1] - wn_grid[-2]) * 0.5

    bin_widths_lower = wn_grid - bin_edges[:-1]
    bin_widths_upper = bin_edges[1:] - wn_grid
    inv_bin_widths = 1.0 / (bin_widths_lower + bin_widths_upper)

    # Per-layer scalars: sigma (Doppler width) and gamma (total damping), vary with T/P.
    sigma = np.empty((n_layers, num_trans), dtype=np.float64)
    gamma_total = np.empty((n_layers, num_trans), dtype=np.float64)
    gamma_lifetime = 1.0 / (const_4_pi_c * lifetimes)  # (n_trans,)  layer-invariant
    # Per-layer, per-transition profile coefficients (n_f/n_i vary per layer).
    abs_coef = np.empty((n_layers, num_trans), dtype=np.float64)
    emi_coef = np.empty((n_layers, num_trans), dtype=np.float64)

    for l in range(n_layers):
        temp_l = temperatures[l]
        pres_l = pressures[l]
        sigma[l] = energy_fi * const_sqrt_NA_kB_on_c * np.sqrt(temp_l / species_mass)

        # Scalar pressure-broadening sum for this layer.
        gamma_pressure_l = 0.0
        for b in range(num_broad):
            gamma_pressure_l += broad_gamma[b, l] * pres_l * (t_ref / temp_l) ** broad_n[b] / pressure_ref
        gamma_total[l] = gamma_lifetime + gamma_pressure_l

        # Bake the 1 / (pi^(3/2)) from integral into the coefficients.
        abs_coef[l] = a_fi * ((n_i[l] * g_f / g_i) - n_f[l]) / (const_8_pi_five_halves_c * energy_fi * energy_fi)
        emi_coef[l] = n_f[l] * a_fi * energy_fi * const_h_c_on_4_pi_five_halves

    gamma_inv = 1.0 / gamma_total  # (n_layers, n_trans)

    # Thread-local output buffers.
    abs_xsec_buffer = np.zeros((numba_num_threads, n_layers, num_grid), dtype=np.float64)
    emi_xsec_buffer = np.zeros((numba_num_threads, n_layers, num_grid), dtype=np.float64)

    grid_min = wn_grid[0]
    grid_max = wn_grid[-1]

    # The bin_term for a given (transition i, grid point j) depends only on energy_fi[i], sigma[l,i], gamma_inv[l,i],
    # and the bin geometry. sigma and gamma_inv are layer-dependent, so bin_term is computed per layer. This is still a
    # net win: the binary search bounds (j_start, j_end) are computed from energy_fi alone and shared across all layers,
    # avoiding redundant range computation.
    for i in numba.prange(num_trans):
        thread_id = numba.get_thread_id()

        energy_fi_i = energy_fi[i]
        transition_min = energy_fi_i - cutoff
        transition_max = energy_fi_i + cutoff

        # Grid range is the same for all layers - compute once.
        j_start = _binary_search_right(bin_edges, transition_min) - 1
        j_start = max(0, j_start)
        j_end = _binary_search_left(bin_edges, transition_max, start=j_start)
        j_end = min(num_grid, j_end)

        if j_start >= j_end:
            continue

        for l in range(n_layers):
            sigma_il = sigma[l, i]
            gamma_inv_il = gamma_inv[l, i]
            abs_coef_il = abs_coef[l, i]
            emi_coef_il = emi_coef[l, i]

            gh_roots_sigma = gh_roots * sigma_il
            start_sigma = grid_min - gh_roots_sigma
            end_sigma = grid_max - gh_roots_sigma
            b_corr = np.pi / (
                    np.arctan((end_sigma - energy_fi_i) * gamma_inv_il)
                    - np.arctan((start_sigma - energy_fi_i) * gamma_inv_il)
            )
            gh_weights_b_corr = gh_weights * b_corr

            for j in range(j_start, j_end):
                upper_width_j = bin_widths_upper[j]
                lower_width_j = bin_widths_lower[j]
                inv_bin_width_j = inv_bin_widths[j]
                wn_j = wn_grid[j]

                shift_sigma = wn_j - energy_fi_i - gh_roots_sigma
                bin_term = np.sum(
                    gh_weights_b_corr
                    * (
                            np.arctan((shift_sigma + upper_width_j) * gamma_inv_il)
                            - np.arctan((shift_sigma - lower_width_j) * gamma_inv_il)
                    )
                ) * inv_bin_width_j

                abs_xsec_buffer[thread_id, l, j] += abs_coef_il * bin_term
                emi_xsec_buffer[thread_id, l, j] += emi_coef_il * bin_term

    # Accumulate thread buffers into final output.
    abs_xsec = np.zeros((n_layers, num_grid), dtype=np.float64)
    emi_xsec = np.zeros((n_layers, num_grid), dtype=np.float64)

    for l in numba.prange(n_layers):
        for k in range(num_grid):
            abs_point = 0.0
            emi_point = 0.0
            for t in range(numba_num_threads):
                abs_point += abs_xsec_buffer[t, l, k]
                emi_point += emi_xsec_buffer[t, l, k]
            abs_xsec[l, k] = abs_point
            emi_xsec[l, k] = emi_point

    return abs_xsec, emi_xsec


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _abs_emi_binned_voigt_fixed_width_layered(
        wn_grid: npt.NDArray[np.float64],
        n_i: npt.NDArray[np.float64],  # (n_layers, n_trans)
        n_f: npt.NDArray[np.float64],  # (n_layers, n_trans)
        a_fi: npt.NDArray[np.float64],  # (n_trans,)
        g_f: npt.NDArray[np.float64],  # (n_trans,)
        g_i: npt.NDArray[np.float64],  # (n_trans,)
        energy_fi: npt.NDArray[np.float64],  # (n_trans,)
        lifetimes: npt.NDArray[np.float64],  # (n_trans,)
        temperatures: npt.NDArray[np.float64],  # (n_layers,)
        pressures: npt.NDArray[np.float64],  # (n_layers,)
        broad_n: npt.NDArray[np.float64],  # (n_broadeners,)
        broad_gamma: npt.NDArray[np.float64],  # (n_broadeners, n_layers)
        species_mass: float,
        half_bin_width: float,
        gh_roots: npt.NDArray[np.float64],
        gh_weights: npt.NDArray[np.float64],
        t_ref: float = 296.0,
        pressure_ref: float = 1.0,
        numba_num_threads: int = _DEFAULT_NUM_THREADS,
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Computes absorption (including stimulated emission) and spontaneous emission profiles for all layers simultaneously.
    The bin_term inner loop - the dominant cost - is computed once per transition per grid point and reused across all
    layers, with only the per-layer coefficients (which depend on populations, temperature, and pressure) applied on
    top.

    Used for wn_grid inputs with fixed grid point spacing.

    Parameters
    ----------
    wn_grid : ndarray, shape (n_grid,)
    n_i : ndarray, shape (n_layers, n_trans)
       LTE population fractions of lower states, per layer.
    n_f : ndarray, shape (n_layers, n_trans)
       LTE population fractions of upper states, per layer.
    a_fi : ndarray, shape (n_trans,)
    g_f : ndarray, shape (n_trans,)
    g_i : ndarray, shape (n_trans,)
    energy_fi : ndarray, shape (n_trans,)
    lifetimes : ndarray, shape (n_trans,)
    temperatures : ndarray, shape (n_layers,)
    pressures : ndarray, shape (n_layers,)
    broad_n : ndarray, shape (n_broadeners,)
    broad_gamma : ndarray, shape (n_broadeners, n_layers)
        Note: layers on axis-1, matching broadening_params[0].
    species_mass : float
    gh_roots : ndarray, shape (n_gh,)
    gh_weights : ndarray, shape (n_gh,)

    Returns
    -------
    abs_xsec : ndarray, shape (n_layers, n_grid)
    emi_xsec : ndarray, shape (n_layers, n_grid)
    """
    n_layers = temperatures.shape[0]
    num_grid = wn_grid.shape[0]
    num_trans = energy_fi.shape[0]
    num_broad = broad_n.shape[0]
    bin_width = 2.0 * half_bin_width
    cutoff = 25 + half_bin_width

    # Bin geometry - identical to the 1-D version, computed once.
    bin_edges = np.empty(num_grid + 1, dtype=np.float64)
    bin_edges[0] = wn_grid[0] - (wn_grid[1] - wn_grid[0]) * 0.5
    for j in range(1, num_grid):
        bin_edges[j] = (wn_grid[j - 1] + wn_grid[j]) * 0.5
    bin_edges[-1] = wn_grid[-1] + (wn_grid[-1] - wn_grid[-2]) * 0.5

    # Per-layer scalars: sigma (Doppler width) and gamma (total damping), vary with T/P.
    sigma = np.empty((n_layers, num_trans), dtype=np.float64)
    gamma_total = np.empty((n_layers, num_trans), dtype=np.float64)
    gamma_lifetime = 1.0 / (const_4_pi_c * lifetimes)  # (n_trans,)  layer-invariant
    # Per-layer, per-transition profile coefficients (n_f/n_i vary per layer).
    abs_coef = np.empty((n_layers, num_trans), dtype=np.float64)
    emi_coef = np.empty((n_layers, num_trans), dtype=np.float64)

    for l in range(n_layers):
        temp_l = temperatures[l]
        pres_l = pressures[l]
        sigma[l] = energy_fi * const_sqrt_NA_kB_on_c * np.sqrt(temp_l / species_mass)

        # Scalar pressure-broadening sum for this layer.
        gamma_pressure_l = 0.0
        for b in range(num_broad):
            gamma_pressure_l += broad_gamma[b, l] * pres_l * (t_ref / temp_l) ** broad_n[b] / pressure_ref
        gamma_total[l] = gamma_lifetime + gamma_pressure_l

        # Bake the 1 / (pi^(3/2) * bin_width) from integral into the coefficients.
        abs_coef[l] = a_fi * ((n_i[l] * g_f / g_i) - n_f[l]) / (
                const_8_pi_five_halves_c * bin_width * energy_fi * energy_fi)
        emi_coef[l] = n_f[l] * a_fi * energy_fi * const_h_c_on_4_pi_five_halves / bin_width

    gamma_inv = 1.0 / gamma_total  # (n_layers, n_trans)

    # Thread-local output buffers.
    abs_xsec_buffer = np.zeros((numba_num_threads, n_layers, num_grid), dtype=np.float64)
    emi_xsec_buffer = np.zeros((numba_num_threads, n_layers, num_grid), dtype=np.float64)

    grid_min = wn_grid[0]
    grid_max = wn_grid[-1]

    for i in numba.prange(num_trans):
        thread_id = numba.get_thread_id()

        energy_fi_i = energy_fi[i]
        transition_min = energy_fi_i - cutoff
        transition_max = energy_fi_i + cutoff

        # Grid range is the same for all layers - compute once.
        j_start = _binary_search_right(bin_edges, transition_min) - 1
        j_start = max(0, j_start)
        j_end = _binary_search_left(bin_edges, transition_max, start=j_start)
        j_end = min(num_grid, j_end)

        if j_start >= j_end:
            continue

        for l in range(n_layers):
            sigma_il = sigma[l, i]
            gamma_inv_il = gamma_inv[l, i]
            abs_coef_il = abs_coef[l, i]
            emi_coef_il = emi_coef[l, i]

            gh_roots_sigma = gh_roots * sigma_il
            start_sigma = grid_min - gh_roots_sigma
            end_sigma = grid_max - gh_roots_sigma
            b_corr = np.pi / (
                    np.arctan((end_sigma - energy_fi_i) * gamma_inv_il)
                    - np.arctan((start_sigma - energy_fi_i) * gamma_inv_il)
            )
            gh_weights_b_corr = gh_weights * b_corr

            for j in range(j_start, j_end):
                wn_j = wn_grid[j]

                shift_sigma = wn_j - energy_fi_i - gh_roots_sigma
                bin_term = np.sum(
                    gh_weights_b_corr
                    * (
                            np.arctan((shift_sigma + half_bin_width) * gamma_inv_il)
                            - np.arctan((shift_sigma - half_bin_width) * gamma_inv_il)
                    )
                )

                abs_xsec_buffer[thread_id, l, j] += abs_coef_il * bin_term
                emi_xsec_buffer[thread_id, l, j] += emi_coef_il * bin_term

    # Accumulate thread buffers into final output.
    abs_xsec = np.zeros((n_layers, num_grid), dtype=np.float64)
    emi_xsec = np.zeros((n_layers, num_grid), dtype=np.float64)

    for l in numba.prange(n_layers):
        for k in range(num_grid):
            abs_point = 0.0
            emi_point = 0.0
            for t in range(numba_num_threads):
                abs_point += abs_xsec_buffer[t, l, k]
                emi_point += emi_xsec_buffer[t, l, k]
            abs_xsec[l, k] = abs_point
            emi_xsec[l, k] = emi_point

    return abs_xsec, emi_xsec


# ------------------------------------- ALL LAYER, SAMPLED VOIGT CALCULATIONS -------------------------------------

@numba.njit(parallel=True, cache=True, error_model="numpy")
def _band_profile_sampled_voigt_layered(
        profile_buffer: npt.NDArray[np.float64],
        wn_grid: npt.NDArray[np.float64],
        id_f: npt.NDArray[np.int32],
        id_i: npt.NDArray[np.int32],
        band_group_indices: npt.NDArray[np.int32],
        group_starts: npt.NDArray[np.int32],
        group_ends: npt.NDArray[np.int32],
        n_lookup: npt.NDArray[np.float64],
        g_lookup: npt.NDArray[np.float64],
        tau_lookup: npt.NDArray[np.float64],
        a_fi: npt.NDArray[np.float64],
        energy_fi: npt.NDArray[np.float64],
        temperatures: npt.NDArray[np.float64],
        pressures: npt.NDArray[np.float64],
        broad_n: npt.NDArray[np.float64],
        broad_gamma: npt.NDArray[np.float64],
        species_mass: float,
        t_ref: float = 296.0,
        pressure_ref: float = 1.0,
) -> None:
    """
    Sampled multi-layer Voigt cross-section using the Humlíček 4-region approximation for the Faddeeva function.

    Evaluates the Voigt profile at each wn_grid point directly rather than integrating over bins. This is faster per
    grid point than the binned version (no quadrature loop) but less accurate for coarse grids where the bin width is
    comparable to the line width - in that regime the binned version is preferred. For high-resolution grids (resolving
    power, R >> line width / grid spacing) the two should converge.

    Can be used for wn_grid input with variable or fixed grid spacing, as profiles are compute at grid points and are
    agnostic of distance to adjacent points (no integral limits).

    Parameters
    ----------
    profile_buffer : ndarray, shape (3, n_bands, n_layers, n_trans)
    wn_grid : ndarray, shape (n_grid,)
    id_f : ndarray, shape (n_trans,)
    id_i : ndarray, shape (n_trans,)
    band_group_indices : ndarray, shape (n_bands_in_batch,)
        Maps the bands in the current batch to the band index stored in the external band_indices map.
    group_starts : ndarray, shape (n_bands_in_batch,)
        Start index in the transition arrays for each band.
    group_ends : ndarray, shape (n_bands_in_batch,)
        End index (exclusive) in the transition arrays for each band.
    n_lookup : ndarray, shape (n_states + 1, n_layers)
        Lookup table for n_frac, avoiding extremely memory intensive joins.
    g_lookup : ndarray, shape (n_states + 1, )
        Lookup table for g, avoiding extremely memory intensive joins.
    tau_lookup : ndarray, shape (n_states + 1, )
        Lookup table for tau (lifetimes), avoiding extremely memory intensive joins.
    a_fi : ndarray, shape (n_trans,)
    energy_fi : ndarray, shape (n_trans,)
    temperatures : ndarray, shape (n_layers,)
    pressures : ndarray, shape (n_layers,)
    broad_n : ndarray, shape (n_broadeners,)
    broad_gamma : ndarray, shape (n_broadeners, n_layers)
        Note: layers on axis-1, matching broadening_params[0].
    species_mass : float

    """
    n_layers = temperatures.shape[0]
    n_grid = wn_grid.shape[0]
    n_trans = energy_fi.shape[0]
    n_broad = broad_n.shape[0]
    n_states = n_lookup.shape[0]
    n_bands_in_batch = group_starts.shape[0]

    cutoff = 25.0
    sqrt2 = np.sqrt(2.0)
    sqrt2_NA_kB_on_c = sqrt2 * const_sqrt_NA_kB_on_c
    inv_sqrt_pi = 1 / np.sqrt(np.pi)

    # Per-layer per-transition: sigma_D, gamma_total, and profile coefficients
    inv_sigma_sqrt2 = np.empty((n_layers, n_trans), dtype=np.float64)  # Doppler sigma (std dev)
    gamma_total_lookup = np.empty((n_layers, n_states), dtype=np.float64)

    inv_g_lookup = 1.0 / g_lookup

    # gamma_lifetime = 1.0 / (const_4_pi_c * lifetimes)  # (n_trans,)
    gamma_lifetime_lookup = 1.0 / (const_4_pi_c * tau_lookup)

    for l in range(n_layers):
        temp_l = temperatures[l]
        pres_l = pressures[l]
        inv_sigma_sqrt2[l] = 1 / (energy_fi * sqrt2_NA_kB_on_c * np.sqrt(temp_l / species_mass))

        gamma_pressure_l = 0.0
        for b in range(n_broad):
            gamma_pressure_l += broad_gamma[b, l] * pres_l * (t_ref / temp_l) ** broad_n[b] / pressure_ref
        gamma_total_lookup[l] = gamma_lifetime_lookup + gamma_pressure_l

    # Voigt y-parameter: gamma_L / (sigma_D * sqrt(2)).
    # sigma here is the Gaussian sigma (standard deviation), so
    # sigma_D * sqrt(2) = sigma * sqrt(2).
    # y_voigt = gamma_total * inv_sigma_sqrt2  # (n_layers, n_trans)

    abs_ste_prefactor = a_fi / (const_8_pi_c * energy_fi * energy_fi)
    spe_prefactor = a_fi * energy_fi * const_h_c_on_4_pi

    # for t in numba.prange(n_trans):
    #     thread_id = numba.get_thread_id()
    for bl in numba.prange(n_bands_in_batch * n_layers):
        b_local = bl // n_layers
        l = bl % n_layers

        b_global = band_group_indices[b_local]
        gs = group_starts[b_local]
        ge = group_ends[b_local]

        abs_bl = profile_buffer[0, b_global, l]
        ste_bl = profile_buffer[1, b_global, l]
        spe_bl = profile_buffer[2, b_global, l]

        for t in range(gs, ge):
            energy_fi_t = energy_fi[t]
            transition_min = energy_fi_t - cutoff
            transition_max = energy_fi_t + cutoff
            j_start = max(0, _binary_search_left(wn_grid, transition_min))
            j_end = min(n_grid, _binary_search_right(wn_grid, transition_max) + 1)

            if j_start >= j_end:
                continue

            id_f_t = id_f[t]
            id_i_t = id_i[t]
            g_f_t = g_lookup[id_f_t]
            inv_g_i_t = inv_g_lookup[id_i_t]
            g_f_on_g_i_t = g_f_t * inv_g_i_t
            # Layer-independent coeficient prefactors.
            abs_ste_prefactor_t = abs_ste_prefactor[t]
            abs_prefactor_t = g_f_on_g_i_t * abs_ste_prefactor_t
            ste_prefactor_t = abs_ste_prefactor_t
            spe_prefactor_t = spe_prefactor[t]

            # for l in range(n_layers):
            gamma_total_tl = gamma_total_lookup[l, id_f_t]
            inv_sigma_sqrt2_tl = inv_sigma_sqrt2[l, t]
            y_tl = gamma_total_tl * inv_sigma_sqrt2_tl
            # y_tl = y_voigt[l, t]

            n_f_tl = n_lookup[id_f_t, l]
            n_i_tl = n_lookup[id_i_t, l]
            abs_tl = n_i_tl * abs_prefactor_t
            ste_tl = n_f_tl * ste_prefactor_t
            spe_tl = n_f_tl * spe_prefactor_t

            # Integral of Re[w(x,y)] dx = sqrt(pi), so the normalised Voigt is Re[w(z)] / (sigma * sqrt(2*pi)).
            norm = inv_sigma_sqrt2_tl * inv_sqrt_pi

            for j in range(j_start, j_end):
                # x = (nu - nu0) / (sigma * sqrt(2))
                wn_j = wn_grid[j]
                x_tj = (wn_j - energy_fi_t) * inv_sigma_sqrt2_tl
                voigt_val = _voigt_humlicek_w(x_tj, y_tl) * norm

                abs_bl[j] += abs_tl * voigt_val
                ste_bl[j] += ste_tl * voigt_val
                spe_bl[j] += spe_tl * voigt_val


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _abs_emi_sampled_voigt_layered(
        wn_grid: npt.NDArray[np.float64],
        id_f: npt.NDArray[np.float64],
        id_i: npt.NDArray[np.float64],
        n_lookup: npt.NDArray[np.float64],
        g_lookup: npt.NDArray[np.float64],
        tau_lookup: npt.NDArray[np.float64],
        a_fi: npt.NDArray[np.float64],
        energy_fi: npt.NDArray[np.float64],
        temperatures: npt.NDArray[np.float64],
        pressures: npt.NDArray[np.float64],
        broad_n: npt.NDArray[np.float64],
        broad_gamma: npt.NDArray[np.float64],
        species_mass: float,
        t_ref: float = 296.0,
        pressure_ref: float = 1.0,
        numba_num_threads: int = _DEFAULT_NUM_THREADS,
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Sampled multi-layer Voigt cross-section using the Humlíček 4-region approximation for the Faddeeva function.

    Evaluates the Voigt profile at each wn_grid point directly rather than integrating over bins. This is faster per
    grid point than the binned version (no quadrature loop) but less accurate for coarse grids where the bin width is
    comparable to the line width - in that regime the binned version is preferred. For high-resolution grids (resolving
    power, R >> line width / grid spacing) the two should converge.

    Can be used for wn_grid input with variable or fixed grid spacing, as profiles are compute at grid points and are
    agnostic of distance to adjacent points (no integral limits).

    Parameters
    ----------
    wn_grid : ndarray, shape (n_grid,)
    id_f : ndarray, shape (n_trans,)
    id_i : ndarray, shape (n_trans,)
    n_lookup : ndarray, shape (n_states + 1, n_layers)
        Lookup table for n_frac, avoiding extremely memory intensive joins.
    g_lookup : ndarray, shape (n_states + 1, )
        Lookup table for g, avoiding extremely memory intensive joins.
    tau_lookup : ndarray, shape (n_states + 1, )
        Lookup table for tau (lifetimes), avoiding extremely memory intensive joins.
    a_fi : ndarray, shape (n_trans,)
    energy_fi : ndarray, shape (n_trans,)
    temperatures : ndarray, shape (n_layers,)
    pressures : ndarray, shape (n_layers,)
    broad_n : ndarray, shape (n_broadeners,)
    broad_gamma : ndarray, shape (n_broadeners, n_layers)
        Note: layers on axis-1, matching broadening_params[0].
    species_mass : float

    Returns
    -------
    abs_xsec : ndarray, shape (n_layers, n_grid)
    emi_xsec : ndarray, shape (n_layers, n_grid)
    """
    n_layers = temperatures.shape[0]
    n_grid = wn_grid.shape[0]
    n_trans = energy_fi.shape[0]
    n_broad = broad_n.shape[0]
    n_states = n_lookup.shape[0]

    cutoff = 25.0
    sqrt2 = np.sqrt(2.0)
    sqrt2_NA_kB_on_c = sqrt2 * const_sqrt_NA_kB_on_c
    inv_sqrt_pi = 1 / np.sqrt(np.pi)

    # Per-layer per-transition: sigma_D, gamma_total, and profile coefficients
    inv_sigma_sqrt2 = np.empty((n_layers, n_trans), dtype=np.float64)  # Doppler sigma (std dev)
    # gamma_total = np.empty((n_layers, num_trans), dtype=np.float64)
    gamma_total_lookup = np.empty((n_layers, n_states), dtype=np.float64)
    # abs_coef = np.empty((n_layers, num_trans), dtype=np.float64)
    # emi_coef = np.empty((n_layers, num_trans), dtype=np.float64)

    inv_g_lookup = 1.0 / g_lookup

    # gamma_lifetime = 1.0 / (const_4_pi_c * lifetimes)  # (n_trans,)
    gamma_lifetime_lookup = 1.0 / (const_4_pi_c * tau_lookup)

    for l in range(n_layers):
        temp_l = temperatures[l]
        pres_l = pressures[l]
        inv_sigma_sqrt2[l] = 1 / (energy_fi * sqrt2_NA_kB_on_c * np.sqrt(temp_l / species_mass))

        gamma_pressure_l = 0.0
        for b in range(n_broad):
            gamma_pressure_l += broad_gamma[b, l] * pres_l * (t_ref / temp_l) ** broad_n[b] / pressure_ref
        # gamma_total[l] = gamma_lifetime + gamma_pressure_l
        gamma_total_lookup[l] = gamma_lifetime_lookup + gamma_pressure_l

        # abs_coef[l] = a_fi * ((n_i[l] * g_f / g_i) - n_f[l]) / (const_8_pi_c * energy_fi * energy_fi)
        # emi_coef[l] = n_f[l] * a_fi * energy_fi * const_h_c_on_4_pi

    # Voigt y-parameter: gamma_L / (sigma_D * sqrt(2)).
    # sigma here is the Gaussian sigma (standard deviation), so
    # sigma_D * sqrt(2) = sigma * sqrt(2).
    # y_voigt = gamma_total * inv_sigma_sqrt2  # (n_layers, n_trans)

    # Thread-local output buffers.
    abs_xsec_buffer = np.zeros((numba_num_threads, n_layers, n_grid), dtype=np.float64)
    emi_xsec_buffer = np.zeros((numba_num_threads, n_layers, n_grid), dtype=np.float64)

    abs_prefactor = a_fi / (const_8_pi_c * energy_fi * energy_fi)
    emi_prefactor = a_fi * energy_fi * const_h_c_on_4_pi

    for t in numba.prange(n_trans):
        thread_id = numba.get_thread_id()

        energy_fi_t = energy_fi[t]
        transition_min = energy_fi_t - cutoff
        transition_max = energy_fi_t + cutoff
        j_start = max(0, _binary_search_left(wn_grid, transition_min))
        j_end = min(n_grid, _binary_search_right(wn_grid, transition_max) + 1)

        if j_start >= j_end:
            continue

        id_f_t = id_f[t]
        id_i_t = id_i[t]
        g_f_t = g_lookup[id_f_t]
        inv_g_i_t = inv_g_lookup[id_i_t]
        g_f_on_g_i_t = g_f_t * inv_g_i_t
        # Layer-independent coeficient prefactors.
        abs_prefactor_t = abs_prefactor[t]
        emi_prefactor_t = emi_prefactor[t]

        for l in range(n_layers):
            gamma_total_tl = gamma_total_lookup[l, id_f_t]
            inv_sigma_sqrt2_tl = inv_sigma_sqrt2[l, t]
            y_tl = gamma_total_tl * inv_sigma_sqrt2_tl
            # y_tl = y_voigt[l, t]

            n_f_l = n_lookup[id_f_t, l]
            n_i_l = n_lookup[id_i_t, l]
            abs_tl = abs_prefactor_t * ((n_i_l * g_f_on_g_i_t) - n_f_l)
            emi_tl = emi_prefactor_t

            # Integral of Re[w(x,y)] dx = sqrt(pi), so the normalised Voigt is Re[w(z)] / (sigma * sqrt(2*pi)).
            norm = inv_sigma_sqrt2_tl * inv_sqrt_pi

            for j in range(j_start, j_end):
                # x = (nu - nu0) / (sigma * sqrt(2))
                wn_j = wn_grid[j]
                x_tj = (wn_j - energy_fi_t) * inv_sigma_sqrt2_tl
                voigt_val = _voigt_humlicek_w(x_tj, y_tl) * norm

                abs_xsec_buffer[thread_id, l, j] += abs_tl * voigt_val
                emi_xsec_buffer[thread_id, l, j] += emi_tl * voigt_val

    # Accumulate thread buffers into final output.
    abs_xsec = np.zeros((n_layers, n_grid), dtype=np.float64)
    emi_xsec = np.zeros((n_layers, n_grid), dtype=np.float64)

    for l in numba.prange(n_layers):
        for k in range(n_grid):
            abs_point = 0.0
            emi_point = 0.0
            for t in range(numba_num_threads):
                abs_point += abs_xsec_buffer[t, l, k]
                emi_point += emi_xsec_buffer[t, l, k]
            abs_xsec[l, k] = abs_point
            emi_xsec[l, k] = emi_point

    return abs_xsec, emi_xsec


# ------------------------------------- ALL LAYER, BINNED GAUSSIAN CALCULATIONS -------------------------------------

@numba.njit(parallel=True, cache=True, error_model="numpy")
def _continuum_profile_binned_gauss_variable_width_layered(
        wn_grid: npt.NDArray[np.float64],
        n_i: npt.NDArray[np.float64],  # (n_layers, n_trans)
        a_fi: npt.NDArray[np.float64],  # (n_trans,)
        g_f: npt.NDArray[np.float64],  # (n_trans,)
        g_i: npt.NDArray[np.float64],  # (n_trans,)
        energy_fi: npt.NDArray[np.float64],  # (n_trans,)
        v_f: npt.NDArray[np.float64],  # (n_trans,)
        temperatures: npt.NDArray[np.float64],  # (n_layers,)
        species_mass: float,
        box_length: float,
        erf_lut: npt.NDArray[np.float64] = global_erf_lut,
        erf_arg_max: float = global_erf_arg_max,
        numba_num_threads: int = _DEFAULT_NUM_THREADS,
) -> npt.NDArray[np.float64]:
    """

    Parameters
    ----------
    wn_grid : ndarray, shape (n_grid,)
    n_i : ndarray, shape (n_layers, n_trans)
    a_fi : ndarray, shape (n_trans,)
    g_f : ndarray, shape (n_trans,)
    g_i : ndarray, shape (n_trans,)
    energy_fi : ndarray, shape (n_trans,)
    v_f : ndarray, shape (n_trans,)
    temperatures : ndarray, shape (n_layers,)
    species_mass : float
    box_length : float
    erf_lut : ndarray
        Precomputed error function look-up table.
    erf_arg_max : float
        Maximum argument value for the precomputed error function look-up table.

    Returns
    -------
    abs_xsec : ndarray, shape (n_layers, n_grid)
    """
    n_layers = temperatures.shape[0]
    num_grid = wn_grid.shape[0]
    num_trans = energy_fi.shape[0]

    sqrtln2 = math.sqrt(math.log(2))

    min_cutoff = 25.0
    max_cutoff = 3000.0
    cutoff_fwhm_multiple = 5.0

    # Bin geometry - identical to the 1-D version, computed once.
    bin_edges = np.empty(num_grid + 1, dtype=np.float64)
    bin_edges[0] = wn_grid[0] - (wn_grid[1] - wn_grid[0]) * 0.5
    for j in range(1, num_grid):
        bin_edges[j] = (wn_grid[j - 1] + wn_grid[j]) * 0.5
    bin_edges[-1] = wn_grid[-1] + (wn_grid[-1] - wn_grid[-2]) * 0.5

    bin_widths_lower = wn_grid - bin_edges[:-1]
    bin_widths_upper = bin_edges[1:] - wn_grid
    inv_bin_widths = 1.0 / (bin_widths_lower + bin_widths_upper)

    alpha_box = const_h_on_8_c * (2 * v_f + 1) / (species_mass * const_amu * box_length * box_length)

    # Per-layer, per-transition profile coefficients (n_f/n_i/T vary per layer).
    alpha_doppler = np.empty((n_layers, num_trans), dtype=np.float64)
    alpha_total = np.empty((n_layers, num_trans), dtype=np.float64)
    sqrtln2_on_alpha = np.empty((n_layers, num_trans), dtype=np.float64)
    abs_coef = np.empty((n_layers, num_trans), dtype=np.float64)

    for l in range(n_layers):
        temp_l = temperatures[l]
        alpha_doppler[l] = energy_fi * const_sqrt_2_NA_kB_log2_on_c * np.sqrt(temp_l / species_mass)
        alpha_total[l] = alpha_box + alpha_doppler[l]
        sqrtln2_on_alpha[l] = sqrtln2 / alpha_total[l]
        # Bake the 1 / 2 from integral into the coefficients.
        abs_coef[l] = a_fi * (n_i[l] * g_f / g_i) / (const_16_pi_c * energy_fi * energy_fi)

    # Set the cutoff from the maximum value of alpha_total across each layer. Useful in cases where doppler broadening
    # is comparable to box broadening and there might be variation across layers. Need single cutoff per transition to
    # allow for j_start, j_end to be set in the first transition loop.
    alpha_total_max = np.empty(num_trans, dtype=np.float64)
    for i in range(num_trans):
        a_max = alpha_total[0, i]
        for l in range(1, n_layers):
            if alpha_total[l, i] > a_max:
                a_max = alpha_total[l, i]
        alpha_total_max[i] = a_max

    cutoff = np.empty(num_trans, dtype=np.float64)
    for i in range(num_trans):
        c = alpha_total_max[i] * cutoff_fwhm_multiple
        if c < min_cutoff:
            c = min_cutoff
        elif c > max_cutoff:
            c = max_cutoff
        cutoff[i] = c

    # Thread-local output buffers.
    abs_xsec_buffer = np.zeros((numba_num_threads, n_layers, num_grid), dtype=np.float64)

    for i in numba.prange(num_trans):
        thread_id = numba.get_thread_id()

        energy_fi_i = energy_fi[i]
        cutoff_i = cutoff[i]

        transition_min = energy_fi_i - cutoff_i
        transition_max = energy_fi_i + cutoff_i

        # Grid range is the same for all layers - compute once.
        j_start = _binary_search_right(bin_edges, transition_min) - 1
        j_start = max(0, j_start)
        j_end = _binary_search_left(bin_edges, transition_max, start=j_start)
        j_end = min(num_grid, j_end)

        if j_start >= j_end:
            continue

        for l in range(n_layers):
            abs_coef_il = abs_coef[l, i]
            sqrtln2_on_alpha_il = sqrtln2_on_alpha[l, i]

            for j in range(j_start, j_end):
                upper_width_j = bin_widths_upper[j]
                lower_width_j = bin_widths_lower[j]
                inv_bin_width_j = inv_bin_widths[j]
                wn_j = wn_grid[j]
                wn_shift = wn_j - energy_fi_i

                erf_plus = get_erf_lut(
                    sqrtln2_on_alpha_il * (wn_shift + upper_width_j), erf_lut, erf_arg_max
                )
                erf_minus = get_erf_lut(
                    sqrtln2_on_alpha_il * (wn_shift - lower_width_j), erf_lut, erf_arg_max
                )
                abs_xsec_buffer[thread_id, l, j] += abs_coef_il * (erf_plus - erf_minus) * inv_bin_width_j

    # Accumulate thread buffers into final output.
    abs_xsec = np.zeros((n_layers, num_grid), dtype=np.float64)

    for l in numba.prange(n_layers):
        for k in range(num_grid):
            abs_point = 0.0
            for t in range(numba_num_threads):
                abs_point += abs_xsec_buffer[t, l, k]
            abs_xsec[l, k] = abs_point

    return abs_xsec


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _continuum_profile_binned_gauss_fixed_width_layered(
        wn_grid: npt.NDArray[np.float64],
        n_i: npt.NDArray[np.float64],  # (n_layers, n_trans)
        a_fi: npt.NDArray[np.float64],  # (n_trans,)
        g_f: npt.NDArray[np.float64],  # (n_trans,)
        g_i: npt.NDArray[np.float64],  # (n_trans,)
        energy_fi: npt.NDArray[np.float64],  # (n_trans,)
        v_f: npt.NDArray[np.float64],  # (n_trans,)
        temperatures: npt.NDArray[np.float64],  # (n_layers,)
        species_mass: float,
        half_bin_width: float,
        box_length: float,
        erf_lut: npt.NDArray[np.float64] = global_erf_lut,
        erf_arg_max: float = global_erf_arg_max,
        numba_num_threads: int = _DEFAULT_NUM_THREADS,
) -> npt.NDArray[np.float64]:
    """

    Parameters
    ----------
    wn_grid : ndarray, shape (n_grid,)
    n_i : ndarray, shape (n_layers, n_trans)
    a_fi : ndarray, shape (n_trans,)
    g_f : ndarray, shape (n_trans,)
    g_i : ndarray, shape (n_trans,)
    energy_fi : ndarray, shape (n_trans,)
    v_f : ndarray, shape (n_trans,)
    temperatures : ndarray, shape (n_layers,)
    species_mass : float
    half_bin_width: float
    box_length : float
    erf_lut: np.ndarray
        Precomputed error function look-up table.
    erf_arg_max: float
        Maximum argument value for the precomputed error function look-up table.

    Returns
    -------
    abs_xsec : ndarray, shape (n_layers, n_grid)
    """
    n_layers = temperatures.shape[0]
    num_grid = wn_grid.shape[0]
    num_trans = energy_fi.shape[0]

    twice_bin_width = 4.0 * half_bin_width
    sqrtln2 = math.sqrt(math.log(2))

    min_cutoff = 25.0
    max_cutoff = 3000.0
    cutoff_fwhm_multiple = 5.0

    # Bin geometry - identical to the 1-D version, computed once.
    bin_edges = np.empty(num_grid + 1, dtype=np.float64)
    bin_edges[0] = wn_grid[0] - (wn_grid[1] - wn_grid[0]) * 0.5
    for j in range(1, num_grid):
        bin_edges[j] = (wn_grid[j - 1] + wn_grid[j]) * 0.5
    bin_edges[-1] = wn_grid[-1] + (wn_grid[-1] - wn_grid[-2]) * 0.5

    alpha_box = const_h_on_8_c * (2 * v_f + 1) / (species_mass * const_amu * box_length * box_length)

    # Per-layer, per-transition profile coefficients (n_f/n_i/T vary per layer).
    alpha_doppler = np.empty((n_layers, num_trans), dtype=np.float64)
    alpha_total = np.empty((n_layers, num_trans), dtype=np.float64)
    sqrtln2_on_alpha = np.empty((n_layers, num_trans), dtype=np.float64)
    abs_coef = np.empty((n_layers, num_trans), dtype=np.float64)

    for l in range(n_layers):
        temp_l = temperatures[l]
        alpha_doppler[l] = energy_fi * const_sqrt_2_NA_kB_log2_on_c * np.sqrt(temp_l / species_mass)
        alpha_total[l] = alpha_box + alpha_doppler[l]
        sqrtln2_on_alpha[l] = sqrtln2 / alpha_total[l]
        # Bake the 1 / (2 * bin_width) from integral into the coefficients.
        abs_coef[l] = a_fi * (n_i[l] * g_f / g_i) / (const_8_pi_c * twice_bin_width * energy_fi * energy_fi)

    # Set the cutoff from the maximum value of alpha_total across each layer. Useful in cases where doppler broadening
    # is comparable to box broadening and there might be variation across layers. Need single cutoff per transition to
    # allow for j_start, j_end to be set in the first transition loop.
    alpha_total_max = np.empty(num_trans, dtype=np.float64)
    for i in range(num_trans):
        a_max = alpha_total[0, i]
        for l in range(1, n_layers):
            if alpha_total[l, i] > a_max:
                a_max = alpha_total[l, i]
        alpha_total_max[i] = a_max

    cutoff = np.empty(num_trans, dtype=np.float64)
    for i in range(num_trans):
        c = alpha_total_max[i] * cutoff_fwhm_multiple
        if c < min_cutoff:
            c = min_cutoff
        elif c > max_cutoff:
            c = max_cutoff
        cutoff[i] = c

    # Thread-local output buffers.
    abs_xsec_buffer = np.zeros((numba_num_threads, n_layers, num_grid), dtype=np.float64)

    for i in numba.prange(num_trans):
        thread_id = numba.get_thread_id()

        energy_fi_i = energy_fi[i]
        cutoff_i = cutoff[i]

        transition_min = energy_fi_i - cutoff_i
        transition_max = energy_fi_i + cutoff_i

        # Grid range is the same for all layers - compute once.
        j_start = _binary_search_right(bin_edges, transition_min) - 1
        j_start = max(0, j_start)
        j_end = _binary_search_left(bin_edges, transition_max, start=j_start)
        j_end = min(num_grid, j_end)

        if j_start >= j_end:
            continue

        for l in range(n_layers):
            abs_coef_il = abs_coef[l, i]
            sqrtln2_on_alpha_il = sqrtln2_on_alpha[l, i]

            for j in range(j_start, j_end):
                wn_j = wn_grid[j]
                wn_shift = wn_j - energy_fi_i

                erf_plus = get_erf_lut(
                    sqrtln2_on_alpha_il * (wn_shift + half_bin_width), erf_lut, erf_arg_max
                )
                erf_minus = get_erf_lut(
                    sqrtln2_on_alpha_il * (wn_shift - half_bin_width), erf_lut, erf_arg_max
                )
                abs_xsec_buffer[thread_id, l, j] += abs_coef_il * (erf_plus - erf_minus)
                # Accumulate thread buffers into final output.
    abs_xsec = np.zeros((n_layers, num_grid), dtype=np.float64)
    for l in numba.prange(n_layers):
        for k in range(num_grid):
            abs_point = 0.0
            for t in range(numba_num_threads):
                abs_point += abs_xsec_buffer[t, l, k]
            abs_xsec[l, k] = abs_point

    return abs_xsec


# ------------------------------------- ALL LAYER, SAMPLED GAUSSIAN CALCULATIONS -------------------------------------


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _continuum_band_profile_sampled_gauss_layered(
        profile_buffer: npt.NDArray[np.float64],
        wn_grid: npt.NDArray[np.float64],
        id_f: npt.NDArray[np.float32],
        id_i: npt.NDArray[np.float32],
        band_group_indices: npt.NDArray[np.int32],
        group_starts: npt.NDArray[np.int32],
        group_ends: npt.NDArray[np.int32],
        n_lookup: npt.NDArray[np.float64],
        g_lookup: npt.NDArray[np.float64],
        v_lookup: npt.NDArray[np.float64],
        a_fi: npt.NDArray[np.float64],
        energy_fi: npt.NDArray[np.float64],
        temperatures: npt.NDArray[np.float64],
        species_mass: float,
        reduced_mass: float,
        box_length: float,
) -> npt.NDArray[np.float64]:
    """
    Sampled multi-layer Gaussian continuum cross-section.

    Evaluates the Gaussian profile at each wn_grid point directly rather than integrating over bins. Grid spacing is
    irrelevant - only the values of wn_grid matter, so this function works equally for uniform, logarithmic, or
    irregular grids.

    The cutoff strategy mirrors the binned version: a fixed window per transition derived from the maximum alpha_total
    across all layers, clamped to [min_cutoff, max_cutoff] in cm^-1. Here the cutoff is expressed in units of HWHM
    multiples (cutoff_fwhm_multiple) rather than sigma multiples.

    Parameters
    ----------
    profile_buffer : ndarray, shape (3, n_bands, n_layers, n_trans)
    wn_grid : ndarray, shape (n_grid,)
    id_f : ndarray, shape (n_trans,)
    id_i : ndarray, shape (n_trans,)
    band_group_indices : ndarray, shape (n_bands_in_batch,)
        Maps the bands in the current batch to the band index stored in the external band_indices map.
    group_starts : ndarray, shape (n_bands_in_batch,)
        Start index in the transition arrays for each band.
    group_ends : ndarray, shape (n_bands_in_batch,)
        End index (exclusive) in the transition arrays for each band.
    n_lookup : ndarray, shape (n_states + 1, n_layers)
        Lookup table for n_frac, avoiding extremely memory intensive joins.
    g_lookup : ndarray, shape (n_states + 1, )
        Lookup table for g, avoiding extremely memory intensive joins.
    v_lookup : ndarray, shape (n_states + 1, )
        Lookup table for v for box broadening, avoiding extremely memory intensive joins.
    a_fi : ndarray, shape (n_trans,)
    energy_fi : ndarray, shape (n_trans,)
    temperatures : ndarray, shape (n_layers,)
    species_mass : float
    reduced_mass : float
    box_length : float
    """
    n_layers = temperatures.shape[0]
    n_grid = wn_grid.shape[0]
    n_states = n_lookup.shape[0]
    n_bands_in_batch = group_starts.shape[0]

    sqrtln2 = np.sqrt(np.log(2.0))
    inv_sqrt_pi = 1 / np.sqrt(np.pi)

    min_cutoff = 25.0
    max_cutoff = 3000.0
    cutoff_fwhm_multiple = 5.0

    inv_mass = 1 / species_mass

    alpha_box_lookup = np.empty(n_states, dtype=np.float64)
    alpha_box_prefactor = const_h_on_8_c / (const_amu * box_length * box_length * reduced_mass)
    for s in range(n_states):
        alpha_box_lookup[s] = alpha_box_prefactor * (2.0 * v_lookup[s] + 1.0)

    inv_g_lookup = 1.0 / g_lookup
    doppler_prefactor = np.empty((n_layers,), dtype=np.float64)

    doppler_coef = const_sqrt_2_NA_kB_log2_on_c * math.sqrt(inv_mass)
    temp_max = temperatures[0]
    for l in range(n_layers):
        temp_l = temperatures[l]
        doppler_prefactor[l] = doppler_coef * math.sqrt(temp_l)
        if temp_l > temp_max:
            temp_max = temp_l

    # alpha_doppler_max per transition (layer-independent bound).
    alpha_doppler_max = energy_fi * doppler_coef * np.sqrt(temp_max)
    abs_prefactor = a_fi / (const_8_pi_c * energy_fi * energy_fi)

    for bl in numba.prange(n_bands_in_batch * n_layers):
        b_local = bl // n_layers
        l = bl % n_layers

        b_global = band_group_indices[b_local]
        gs = group_starts[b_local]
        ge = group_ends[b_local]

        abs_bl = profile_buffer[0, b_global, l]

        doppler_prefactor_l = doppler_prefactor[l]

        for t in range(gs, ge):
            energy_fi_t = energy_fi[t]
            id_f_t = id_f[t]
            # Broadening.
            alpha_box_t = alpha_box_lookup[id_f_t]
            alpha_doppler_max_t = alpha_doppler_max[t]
            alpha_total_max_t = alpha_box_t + alpha_doppler_max_t
            cutoff_t = cutoff_fwhm_multiple * alpha_total_max_t
            if cutoff_t < min_cutoff:
                cutoff_t = min_cutoff
            elif cutoff_t > max_cutoff:
                cutoff_t = max_cutoff

            transition_min = energy_fi_t - cutoff_t
            transition_max = energy_fi_t + cutoff_t

            j_start = max(0, _binary_search_left(wn_grid, transition_min))
            j_end = min(n_grid, _binary_search_right(wn_grid, transition_max) + 1)

            if j_start >= j_end:
                continue

            id_i_t = id_i[t]
            g_f_t = g_lookup[id_f_t]
            inv_g_i_t = inv_g_lookup[id_i_t]
            g_f_on_g_i_t = g_f_t * inv_g_i_t
            # Layer-independent coeficient prefactors.
            abs_prefactor_t = g_f_on_g_i_t * abs_prefactor[t]
            # for l in range(n_layers):
            alpha_doppler_tl = energy_fi_t * doppler_prefactor_l
            alpha_total_tl = alpha_box_t + alpha_doppler_tl
            sqrtln2_on_alpha_tl = sqrtln2 / alpha_total_tl

            n_i_tl = n_lookup[id_i_t, l]
            abs_tl = n_i_tl * abs_prefactor_t

            # Normalised Gaussian: G(nu) = (sqrt(ln2) / (alpha * sqrt(pi))) * exp(-ln2 * ((nu - nu0) / alpha)^2)
            # Rewritten with sqrtln2_on_alpha precomputed:
            #   exponent arg  = sqrtln2_on_alpha * (nu - nu0)
            #   prefactor     = sqrtln2_on_alpha / sqrt(pi)
            norm = sqrtln2_on_alpha_tl * inv_sqrt_pi

            for j in range(j_start, j_end):
                wn_j = wn_grid[j]
                wn_shift = wn_j - energy_fi_t
                arg = sqrtln2_on_alpha_tl * wn_shift
                gauss_val = norm * math.exp(-(arg * arg))

                abs_bl[j] += abs_tl * gauss_val


# ------------------------------------- OLD, PER LAYER CALCULATIONS -------------------------------------


# @numba.njit(parallel=True, cache=True, error_model="numpy")
# def _band_profile_binned_doppler_variable_width(
#         wn_grid: npt.NDArray[np.float64],
#         n_i: npt.NDArray[np.float64],
#         n_f: npt.NDArray[np.float64],
#         a_fi: npt.NDArray[np.float64],
#         g_f: npt.NDArray[np.float64],
#         g_i: npt.NDArray[np.float64],
#         energy_fi: npt.NDArray[np.float64],
#         temperature: float,
#         species_mass: float,
# ) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
#     assert n_i.shape[0] == n_f.shape[0] == a_fi.shape[0] == g_f.shape[0] == g_i.shape[0] == energy_fi.shape[0]
#
#     sqrtln2 = math.sqrt(math.log(2))
#     num_grid = wn_grid.shape[0]
#     _abs_xsec = np.zeros(num_grid, dtype=np.float64)
#     _emi_xsec = np.zeros(num_grid, dtype=np.float64)
#     num_trans = energy_fi.shape[0]
#     cutoff = 25
#
#     bin_widths = np.zeros(num_grid + 1, dtype=np.float64)
#     bin_widths[1:-1] = (wn_grid[:-1] + wn_grid[1:]) / 2.0 - wn_grid[:-1]
#
#     abs_coef = a_fi * (n_i * g_f / g_i) / (const_16_pi_c * energy_fi * energy_fi)
#     emi_coef = n_f * a_fi * energy_fi * const_h_c_on_8_pi
#     sqrtln2_on_alpha = sqrtln2 / (energy_fi * const_sqrt_2_NA_kB_log2_on_c * np.sqrt(temperature / species_mass))
#
#     for i in numba.prange(num_trans):
#         energy_fi_i = energy_fi[i]
#         sqrtln2_on_alpha_i = sqrtln2_on_alpha[i]
#         abs_coef_i = abs_coef[i]
#         emi_coef_i = emi_coef[i]
#
#         for j in range(num_grid):
#             wn_shift = wn_grid[j] - energy_fi_i
#             upper_width = bin_widths[j + 1]
#             lower_width = bin_widths[j]
#             if -upper_width - cutoff <= wn_shift <= lower_width + cutoff:
#                 bin_term = (
#                                    math.erf(sqrtln2_on_alpha_i * (wn_shift + upper_width))
#                                    - math.erf(sqrtln2_on_alpha_i * (wn_shift - lower_width))
#                            ) / (upper_width + lower_width)
#                 _abs_xsec[j] += abs_coef_i * bin_term
#                 _emi_xsec[j] += emi_coef_i * bin_term
#     return _abs_xsec, _emi_xsec


# @numba.njit(parallel=True, cache=True, error_model="numpy")
# def _band_profile_binned_doppler_fixed_width(
#         wn_grid: npt.NDArray[np.float64],
#         n_i: npt.NDArray[np.float64],
#         n_f: npt.NDArray[np.float64],
#         a_fi: npt.NDArray[np.float64],
#         g_f: npt.NDArray[np.float64],
#         g_i: npt.NDArray[np.float64],
#         energy_fi: npt.NDArray[np.float64],
#         temperature: float,
#         species_mass: float,
#         half_bin_width: float,
# ) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
#     assert n_i.shape[0] == n_f.shape[0] == a_fi.shape[0] == g_f.shape[0] == g_i.shape[0] == energy_fi.shape[0]
#
#     twice_bin_width = 4.0 * half_bin_width
#     sqrtln2 = math.sqrt(math.log(2))
#     num_grid = wn_grid.shape[0]
#     _abs_xsec = np.zeros(num_grid, dtype=np.float64)
#     _emi_xsec = np.zeros(num_grid, dtype=np.float64)
#     num_trans = energy_fi.shape[0]
#     cutoff = 25 + half_bin_width
#
#     abs_coef = a_fi * (n_i * g_f / g_i) / (const_8_pi_c * twice_bin_width * energy_fi * energy_fi)
#     emi_coef = n_f * a_fi * energy_fi * const_h_c_on_4_pi / twice_bin_width
#     sqrtln2_on_alpha = sqrtln2 / (energy_fi * const_sqrt_2_NA_kB_log2_on_c * np.sqrt(temperature / species_mass))
#
#     for i in numba.prange(num_trans):
#         energy_fi_i = energy_fi[i]
#         sqrtln2_on_alpha_i = sqrtln2_on_alpha[i]
#         abs_coef_i = abs_coef[i]
#         emi_coef_i = emi_coef[i]
#
#         for j in range(num_grid):
#             wn_shift = wn_grid[j] - energy_fi_i
#             if np.abs(wn_shift) <= cutoff:
#                 bin_term = math.erf(sqrtln2_on_alpha_i * (wn_shift + half_bin_width)) - math.erf(
#                     sqrtln2_on_alpha_i * (wn_shift - half_bin_width)
#                 )
#                 _abs_xsec[j] += abs_coef_i * bin_term
#                 _emi_xsec[j] += emi_coef_i * bin_term
#     return _abs_xsec, _emi_xsec


# @numba.njit(parallel=True, cache=True, error_model="numpy")
# def _band_profile_binned_voigt_variable_width(
#         wn_grid: npt.NDArray[np.float64],
#         n_i: npt.NDArray[np.float64],
#         n_f: npt.NDArray[np.float64],
#         a_fi: npt.NDArray[np.float64],
#         g_f: npt.NDArray[np.float64],
#         g_i: npt.NDArray[np.float64],
#         energy_fi: npt.NDArray[np.float64],
#         lifetimes: npt.NDArray[np.float64],
#         temperature: float,
#         pressure: float,
#         broad_n: npt.NDArray[np.float64],
#         broad_gamma: npt.NDArray[np.float64],
#         species_mass: float,
#         gh_roots: npt.NDArray[np.float64],
#         gh_weights: npt.NDArray[np.float64],
#         t_ref: float = 296,
#         pressure_ref: float = 1,
#         numba_num_threads: int = _DEFAULT_NUM_THREADS,
# ) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
#     assert n_i.shape[0] == n_f.shape[0] == a_fi.shape[0] == g_f.shape[0] == g_i.shape[0] == energy_fi.shape[0] == \
#            lifetimes.shape[0]
#     assert broad_n.shape[0] == broad_gamma.shape[0]
#     assert gh_roots.shape[0] == gh_weights.shape[0]
#
#     num_grid = wn_grid.shape[0]
#     num_trans = energy_fi.shape[0]
#     cutoff = 25
#
#     bin_edges = np.empty(num_grid + 1, dtype=np.float64)
#     bin_edges[0] = wn_grid[0] - (wn_grid[1] - wn_grid[0]) * 0.5
#     for j in range(1, num_grid):
#         bin_edges[j] = (wn_grid[j - 1] + wn_grid[j]) * 0.5
#     bin_edges[-1] = wn_grid[-1] + (wn_grid[-1] - wn_grid[-2]) * 0.5
#
#     bin_widths_lower = wn_grid - bin_edges[:-1]
#     bin_widths_upper = bin_edges[1:] - wn_grid
#     inv_bin_widths = 1.0 / (bin_widths_lower + bin_widths_upper)
#
#     # Stimulated emission removed!
#     abs_coef = a_fi * (n_i * g_f / g_i) / (const_8_pi_five_halves_c * energy_fi * energy_fi)
#     ste_coef = a_fi * n_f / (const_8_pi_five_halves_c * energy_fi * energy_fi)
#     spe_coef = n_f * a_fi * energy_fi * const_h_c_on_4_pi_five_halves
#     sigma = energy_fi * const_sqrt_NA_kB_on_c * np.sqrt(temperature / species_mass)
#     gamma_pressure = np.sum(broad_gamma * pressure * (t_ref / temperature) ** broad_n / pressure_ref)
#     gamma_lifetime = 1 / (const_4_pi_c * lifetimes)
#     gamma_total = gamma_lifetime + gamma_pressure
#     gamma_inv = 1 / gamma_total
#
#     grid_min = wn_grid[0]
#     grid_max = wn_grid[-1]
#
#     abs_xsec_buffer = np.zeros((numba_num_threads, num_grid), dtype=np.float64)
#     ste_xsec_buffer = np.zeros((numba_num_threads, num_grid), dtype=np.float64)
#     spe_xsec_buffer = np.zeros((numba_num_threads, num_grid), dtype=np.float64)
#
#     for i in numba.prange(num_trans):
#         thread_id = numba.get_thread_id()
#
#         energy_fi_i = energy_fi[i]
#         abs_coef_i = abs_coef[i]
#         ste_coef_i = ste_coef[i]
#         spe_coef_i = spe_coef[i]
#         gamma_inv_i = gamma_inv[i]
#         sigma_i = sigma[i]
#
#         gh_roots_sigma = gh_roots * sigma_i
#         start_sigma = grid_min - gh_roots_sigma
#         end_sigma = grid_max - gh_roots_sigma
#         b_corr = np.pi / (
#                 np.arctan((end_sigma - energy_fi_i) * gamma_inv_i)
#                 - np.arctan((start_sigma - energy_fi_i) * gamma_inv_i)
#         )
#         gh_weights_b_corr = gh_weights * b_corr
#
#         transition_min = energy_fi_i - cutoff
#         transition_max = energy_fi_i + cutoff
#
#         j_start = _binary_search_right(bin_edges, transition_min) - 1
#         j_start = max(0, j_start)
#         j_end = _binary_search_left(bin_edges, transition_max, start=j_start)
#         j_end = min(num_grid, j_end)
#
#         for j in range(j_start, j_end):
#             upper_width_j = bin_widths_upper[j]
#             lower_width_j = bin_widths_lower[j]
#             inv_bin_width_j = inv_bin_widths[j]
#             wn_j = wn_grid[j]
#
#             shift_sigma = wn_j - energy_fi_i - gh_roots_sigma
#             bin_term = np.sum(
#                 gh_weights_b_corr
#                 * (
#                         np.arctan((shift_sigma + upper_width_j) * gamma_inv_i)
#                         - np.arctan((shift_sigma - lower_width_j) * gamma_inv_i)
#                 )
#             ) * inv_bin_width_j
#
#             abs_xsec_buffer[thread_id, j] += abs_coef_i * bin_term
#             ste_xsec_buffer[thread_id, j] += ste_coef_i * bin_term
#             spe_xsec_buffer[thread_id, j] += spe_coef_i * bin_term
#     # Accumulate buffers.
#     _abs_xsec = np.zeros(num_grid, dtype=np.float64)
#     _ste_xsec = np.zeros(num_grid, dtype=np.float64)
#     _spe_xsec = np.zeros(num_grid, dtype=np.float64)
#     for k in numba.prange(num_grid):
#         abs_xsec_point = 0.0
#         ste_xsec_point = 0.0
#         spe_xsec_point = 0.0
#         for t in range(numba_num_threads):
#             abs_xsec_point += abs_xsec_buffer[t, k]
#             ste_xsec_point += ste_xsec_buffer[t, k]
#             spe_xsec_point += spe_xsec_buffer[t, k]
#         _abs_xsec[k] = abs_xsec_point
#         _ste_xsec[k] = ste_xsec_point
#         _spe_xsec[k] = spe_xsec_point
#
#     return _abs_xsec, _ste_xsec, _spe_xsec


# @numba.njit(parallel=True, cache=True, error_model="numpy")
# def _band_profile_binned_voigt_fixed_width(
#         wn_grid: npt.NDArray[np.float64],
#         n_i: npt.NDArray[np.float64],
#         n_f: npt.NDArray[np.float64],
#         a_fi: npt.NDArray[np.float64],
#         g_f: npt.NDArray[np.float64],
#         g_i: npt.NDArray[np.float64],
#         energy_fi: npt.NDArray[np.float64],
#         lifetimes: npt.NDArray[np.float64],
#         temperature: float,
#         pressure: float,
#         broad_n: npt.NDArray[np.float64],
#         broad_gamma: npt.NDArray[np.float64],
#         species_mass: float,
#         half_bin_width: float,
#         gh_roots: npt.NDArray[np.float64],
#         gh_weights: npt.NDArray[np.float64],
#         t_ref: float = 296,
#         pressure_ref: float = 1,
#         numba_num_threads: int = _DEFAULT_NUM_THREADS,
# ) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
#     assert n_i.shape[0] == n_f.shape[0] == a_fi.shape[0] == g_f.shape[0] == g_i.shape[0] == energy_fi.shape[0] == \
#            lifetimes.shape[0]
#     assert broad_n.shape[0] == broad_gamma.shape[0]
#     assert gh_roots.shape[0] == gh_weights.shape[0]
#
#     bin_width = 2.0 * half_bin_width
#     num_grid = wn_grid.shape[0]
#     num_trans = energy_fi.shape[0]
#     cutoff = 25 + half_bin_width
#
#     bin_edges = np.empty(num_grid + 1, dtype=np.float64)
#     bin_edges[0] = wn_grid[0] - (wn_grid[1] - wn_grid[0]) * 0.5
#     for j in range(1, num_grid):
#         bin_edges[j] = (wn_grid[j - 1] + wn_grid[j]) * 0.5
#     bin_edges[-1] = wn_grid[-1] + (wn_grid[-1] - wn_grid[-2]) * 0.5
#
#     abs_coef = a_fi * (n_i * g_f / g_i) / (const_8_pi_five_halves_c * bin_width * energy_fi * energy_fi)
#     ste_coef = a_fi * n_f / (const_8_pi_five_halves_c * bin_width * energy_fi * energy_fi)
#     spe_coef = n_f * a_fi * energy_fi * const_h_c_on_4_pi_five_halves / bin_width
#
#     sigma = energy_fi * const_sqrt_NA_kB_on_c * np.sqrt(temperature / species_mass)
#     gamma_pressure = np.sum(broad_gamma * pressure * (t_ref / temperature) ** broad_n / pressure_ref)
#     gamma_lifetime = 1 / (const_4_pi_c * lifetimes)
#     gamma_total = gamma_lifetime + gamma_pressure
#     gamma_inv = 1 / gamma_total
#
#     grid_min = wn_grid[0]
#     grid_max = wn_grid[-1]
#
#     abs_xsec_buffer = np.zeros((numba_num_threads, num_grid), dtype=np.float64)
#     ste_xsec_buffer = np.zeros((numba_num_threads, num_grid), dtype=np.float64)
#     spe_xsec_buffer = np.zeros((numba_num_threads, num_grid), dtype=np.float64)
#
#     for i in numba.prange(num_trans):
#         thread_id = numba.get_thread_id()
#
#         energy_fi_i = energy_fi[i]
#         abs_coef_i = abs_coef[i]
#         ste_coef_i = ste_coef[i]
#         spe_coef_i = spe_coef[i]
#         gamma_inv_i = gamma_inv[i]
#         sigma_i = sigma[i]
#
#         gh_roots_sigma = gh_roots * sigma_i
#         start_sigma = grid_min - gh_roots_sigma
#         end_sigma = grid_max - gh_roots_sigma
#         b_corr = np.pi / (
#                 np.arctan((end_sigma - energy_fi_i) * gamma_inv_i)
#                 - np.arctan((start_sigma - energy_fi_i) * gamma_inv_i)
#         )
#         gh_weights_b_corr = gh_weights * b_corr
#
#         transition_min = energy_fi_i - cutoff
#         transition_max = energy_fi_i + cutoff
#
#         j_start = _binary_search_right(bin_edges, transition_min) - 1
#         j_start = max(0, j_start)
#         j_end = _binary_search_left(bin_edges, transition_max, start=j_start)
#         j_end = min(num_grid, j_end)
#
#         for j in range(j_start, j_end):
#             wn_j = wn_grid[j]
#
#             shift_sigma = wn_j - energy_fi_i - gh_roots_sigma
#             bin_term = np.sum(
#                 gh_weights_b_corr
#                 * (
#                         np.arctan((shift_sigma + half_bin_width) * gamma_inv_i)
#                         - np.arctan((shift_sigma - half_bin_width) * gamma_inv_i)
#                 )
#             )
#             abs_xsec_buffer[thread_id, j] += abs_coef_i * bin_term
#             ste_xsec_buffer[thread_id, j] += ste_coef_i * bin_term
#             spe_xsec_buffer[thread_id, j] += spe_coef_i * bin_term
#     # Accumulate buffers.
#     _abs_xsec = np.zeros(num_grid, dtype=np.float64)
#     _ste_xsec = np.zeros(num_grid, dtype=np.float64)
#     _spe_xsec = np.zeros(num_grid, dtype=np.float64)
#     for k in numba.prange(num_grid):
#         abs_xsec_point = 0.0
#         ste_xsec_point = 0.0
#         spe_xsec_point = 0.0
#         for t in range(numba_num_threads):
#             abs_xsec_point += abs_xsec_buffer[t, k]
#             ste_xsec_point += ste_xsec_buffer[t, k]
#             spe_xsec_point += spe_xsec_buffer[t, k]
#         _abs_xsec[k] = abs_xsec_point
#         _ste_xsec[k] = ste_xsec_point
#         _spe_xsec[k] = spe_xsec_point
#     return _abs_xsec, _ste_xsec, _spe_xsec


# ------------------------------------- OLD, ABS/EMI XSEC PER LAYER CALCULATIONS -------------------------------------
# TODO: Refactor into multi-layer calculations as above in band profile calculations.

@numba.njit(parallel=True, cache=True, error_model="numpy")
def _abs_emi_binned_doppler_variable_width(
        wn_grid: npt.NDArray[np.float64],
        n_i: npt.NDArray[np.float64],
        n_f: npt.NDArray[np.float64],
        a_fi: npt.NDArray[np.float64],
        g_f: npt.NDArray[np.float64],
        g_i: npt.NDArray[np.float64],
        energy_fi: npt.NDArray[np.float64],
        temperature: float,
        species_mass: float,
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    assert n_i.shape[0] == n_f.shape[0] == a_fi.shape[0] == g_f.shape[0] == g_i.shape[0] == energy_fi.shape[0]

    sqrtln2 = math.sqrt(math.log(2))
    num_grid = wn_grid.shape[0]
    _abs_xsec = np.zeros(num_grid, dtype=np.float64)
    _emi_xsec = np.zeros(num_grid, dtype=np.float64)
    num_trans = energy_fi.shape[0]
    cutoff = 25

    bin_widths = np.zeros(num_grid + 1, dtype=np.float64)
    bin_widths[1:-1] = (wn_grid[:-1] + wn_grid[1:]) / 2.0 - wn_grid[:-1]

    abs_coef = a_fi * ((n_i * g_f / g_i) - n_f) / (const_16_pi_c * energy_fi * energy_fi)
    emi_coef = n_f * a_fi * energy_fi * const_h_c_on_8_pi
    sqrtln2_on_alpha = sqrtln2 / (energy_fi * const_sqrt_2_NA_kB_log2_on_c * np.sqrt(temperature / species_mass))

    for i in numba.prange(num_trans):
        energy_fi_i = energy_fi[i]
        sqrtln2_on_alpha_i = sqrtln2_on_alpha[i]
        abs_coef_i = abs_coef[i]
        emi_coef_i = emi_coef[i]

        for j in range(num_grid):
            wn_shift = wn_grid[j] - energy_fi_i
            upper_width = bin_widths[j + 1]
            lower_width = bin_widths[j]
            if -upper_width - cutoff <= wn_shift <= lower_width + cutoff:
                bin_term = (
                                   math.erf(sqrtln2_on_alpha_i * (wn_shift + upper_width))
                                   - math.erf(sqrtln2_on_alpha_i * (wn_shift - lower_width))
                           ) / (upper_width + lower_width)
                _abs_xsec[j] += abs_coef_i * bin_term
                _emi_xsec[j] += emi_coef_i * bin_term
    return _abs_xsec, _emi_xsec


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _abs_emi_binned_doppler_fixed_width(
        wn_grid: npt.NDArray[np.float64],
        n_i: npt.NDArray[np.float64],
        n_f: npt.NDArray[np.float64],
        a_fi: npt.NDArray[np.float64],
        g_f: npt.NDArray[np.float64],
        g_i: npt.NDArray[np.float64],
        energy_fi: npt.NDArray[np.float64],
        temperature: float,
        species_mass: float,
        half_bin_width: float,
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    assert n_i.shape[0] == n_f.shape[0] == a_fi.shape[0] == g_f.shape[0] == g_i.shape[0] == energy_fi.shape[0]

    twice_bin_width = 4.0 * half_bin_width
    sqrtln2 = math.sqrt(math.log(2))
    num_grid = wn_grid.shape[0]
    _abs_xsec = np.zeros(num_grid, dtype=np.float64)
    _emi_xsec = np.zeros(num_grid, dtype=np.float64)
    num_trans = energy_fi.shape[0]
    cutoff = 25 + half_bin_width

    abs_coef = a_fi * ((n_i * g_f / g_i) - n_f) / (const_8_pi_c * twice_bin_width * energy_fi * energy_fi)
    emi_coef = n_f * a_fi * energy_fi * const_h_c_on_4_pi / twice_bin_width
    sqrtln2_on_alpha = sqrtln2 / (energy_fi * const_sqrt_2_NA_kB_log2_on_c * np.sqrt(temperature / species_mass))

    for i in numba.prange(num_trans):
        energy_fi_i = energy_fi[i]
        sqrtln2_on_alpha_i = sqrtln2_on_alpha[i]
        abs_coef_i = abs_coef[i]
        emi_coef_i = emi_coef[i]

        for j in range(num_grid):
            wn_shift = wn_grid[j] - energy_fi_i
            if np.abs(wn_shift) <= cutoff:
                bin_term = math.erf(sqrtln2_on_alpha_i * (wn_shift + half_bin_width)) - math.erf(
                    sqrtln2_on_alpha_i * (wn_shift - half_bin_width)
                )
                _abs_xsec[j] += abs_coef_i * bin_term
                _emi_xsec[j] += emi_coef_i * bin_term
    return _abs_xsec, _emi_xsec


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _abs_emi_binned_voigt_variable_width(
        wn_grid: npt.NDArray[np.float64],
        n_i: npt.NDArray[np.float64],
        n_f: npt.NDArray[np.float64],
        a_fi: npt.NDArray[np.float64],
        g_f: npt.NDArray[np.float64],
        g_i: npt.NDArray[np.float64],
        energy_fi: npt.NDArray[np.float64],
        lifetimes: npt.NDArray[np.float64],
        temperature: float,
        pressure: float,
        broad_n: npt.NDArray[np.float64],
        broad_gamma: npt.NDArray[np.float64],
        species_mass: float,
        gh_roots: npt.NDArray[np.float64],
        gh_weights: npt.NDArray[np.float64],
        t_ref: float = 296,
        pressure_ref: float = 1,
        numba_num_threads: int = _DEFAULT_NUM_THREADS,
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    assert n_i.shape[0] == n_f.shape[0] == a_fi.shape[0] == g_f.shape[0] == g_i.shape[0] == energy_fi.shape[0] == \
           lifetimes.shape[0]
    assert broad_n.shape[0] == broad_gamma.shape[0]
    assert gh_roots.shape[0] == gh_weights.shape[0]

    num_grid = wn_grid.shape[0]
    num_trans = energy_fi.shape[0]
    cutoff = 25

    bin_edges = np.empty(num_grid + 1, dtype=np.float64)
    bin_edges[0] = wn_grid[0] - (wn_grid[1] - wn_grid[0]) * 0.5
    for j in range(1, num_grid):
        bin_edges[j] = (wn_grid[j - 1] + wn_grid[j]) * 0.5
    bin_edges[-1] = wn_grid[-1] + (wn_grid[-1] - wn_grid[-2]) * 0.5

    bin_widths_lower = wn_grid - bin_edges[:-1]
    bin_widths_upper = bin_edges[1:] - wn_grid
    inv_bin_widths = 1.0 / (bin_widths_lower + bin_widths_upper)

    abs_coef = a_fi * ((n_i * g_f / g_i) - n_f) / (const_8_pi_five_halves_c * energy_fi * energy_fi)
    emi_coef = n_f * a_fi * energy_fi * const_h_c_on_4_pi_five_halves
    sigma = energy_fi * const_sqrt_NA_kB_on_c * np.sqrt(temperature / species_mass)
    gamma_pressure = np.sum(broad_gamma * pressure * (t_ref / temperature) ** broad_n / pressure_ref)
    gamma_lifetime = 1 / (const_4_pi_c * lifetimes)
    gamma_total = gamma_lifetime + gamma_pressure
    gamma_inv = 1 / gamma_total

    grid_min = wn_grid[0]
    grid_max = wn_grid[-1]

    abs_xsec_buffer = np.zeros((numba_num_threads, num_grid), dtype=np.float64)
    emi_xsec_buffer = np.zeros((numba_num_threads, num_grid), dtype=np.float64)

    for i in numba.prange(num_trans):
        thread_id = numba.get_thread_id()

        energy_fi_i = energy_fi[i]
        abs_coef_i = abs_coef[i]
        emi_coef_i = emi_coef[i]
        gamma_inv_i = gamma_inv[i]
        sigma_i = sigma[i]

        gh_roots_sigma = gh_roots * sigma_i
        start_sigma = grid_min - gh_roots_sigma
        end_sigma = grid_max - gh_roots_sigma
        b_corr = np.pi / (
                np.arctan((end_sigma - energy_fi_i) * gamma_inv_i)
                - np.arctan((start_sigma - energy_fi_i) * gamma_inv_i)
        )
        gh_weights_b_corr = gh_weights * b_corr

        transition_min = energy_fi_i - cutoff
        transition_max = energy_fi_i + cutoff

        j_start = _binary_search_right(bin_edges, transition_min) - 1
        j_start = max(0, j_start)
        j_end = _binary_search_left(bin_edges, transition_max, start=j_start)
        j_end = min(num_grid, j_end)

        for j in range(j_start, j_end):
            upper_width_j = bin_widths_upper[j]
            lower_width_j = bin_widths_lower[j]
            inv_bin_width_j = inv_bin_widths[j]
            wn_j = wn_grid[j]

            shift_sigma = wn_j - energy_fi_i - gh_roots_sigma
            bin_term = np.sum(
                gh_weights_b_corr
                * (
                        np.arctan((shift_sigma + upper_width_j) * gamma_inv_i)
                        - np.arctan((shift_sigma - lower_width_j) * gamma_inv_i)
                )
            ) * inv_bin_width_j

            abs_xsec_buffer[thread_id, j] += abs_coef_i * bin_term
            emi_xsec_buffer[thread_id, j] += emi_coef_i * bin_term
    # Accumulate buffers.
    _abs_xsec = np.zeros(num_grid, dtype=np.float64)
    _emi_xsec = np.zeros(num_grid, dtype=np.float64)
    for k in numba.prange(num_grid):
        abs_xsec_point = 0.0
        emi_xsec_point = 0.0
        for t in range(numba_num_threads):
            abs_xsec_point += abs_xsec_buffer[t, k]
            emi_xsec_point += emi_xsec_buffer[t, k]
        _abs_xsec[k] = abs_xsec_point
        _emi_xsec[k] = emi_xsec_point

    return _abs_xsec, _emi_xsec


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _abs_emi_binned_voigt_fixed_width(
        wn_grid: npt.NDArray[np.float64],
        n_i: npt.NDArray[np.float64],
        n_f: npt.NDArray[np.float64],
        a_fi: npt.NDArray[np.float64],
        g_f: npt.NDArray[np.float64],
        g_i: npt.NDArray[np.float64],
        energy_fi: npt.NDArray[np.float64],
        lifetimes: npt.NDArray[np.float64],
        temperature: float,
        pressure: float,
        broad_n: npt.NDArray[np.float64],
        broad_gamma: npt.NDArray[np.float64],
        species_mass: float,
        half_bin_width: float,
        gh_roots: npt.NDArray[np.float64],
        gh_weights: npt.NDArray[np.float64],
        t_ref: float = 296,
        pressure_ref: float = 1,
        numba_num_threads: int = _DEFAULT_NUM_THREADS,
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    assert n_i.shape[0] == n_f.shape[0] == a_fi.shape[0] == g_f.shape[0] == g_i.shape[0] == energy_fi.shape[0] == \
           lifetimes.shape[0]
    assert broad_n.shape[0] == broad_gamma.shape[0]
    assert gh_roots.shape[0] == gh_weights.shape[0]

    bin_width = 2.0 * half_bin_width
    num_grid = wn_grid.shape[0]
    num_trans = energy_fi.shape[0]
    cutoff = 25 + half_bin_width

    bin_edges = np.empty(num_grid + 1, dtype=np.float64)
    bin_edges[0] = wn_grid[0] - (wn_grid[1] - wn_grid[0]) * 0.5
    for j in range(1, num_grid):
        bin_edges[j] = (wn_grid[j - 1] + wn_grid[j]) * 0.5
    bin_edges[-1] = wn_grid[-1] + (wn_grid[-1] - wn_grid[-2]) * 0.5

    abs_coef = a_fi * ((n_i * g_f / g_i) - n_f) / (const_8_pi_five_halves_c * bin_width * energy_fi * energy_fi)
    emi_coef = n_f * a_fi * energy_fi * const_h_c_on_4_pi_five_halves / bin_width
    sigma = energy_fi * const_sqrt_NA_kB_on_c * np.sqrt(temperature / species_mass)
    gamma_pressure = np.sum(broad_gamma * pressure * (t_ref / temperature) ** broad_n / pressure_ref)
    gamma_lifetime = 1 / (const_4_pi_c * lifetimes)
    gamma_total = gamma_lifetime + gamma_pressure
    gamma_inv = 1 / gamma_total

    grid_min = wn_grid[0]
    grid_max = wn_grid[-1]

    abs_xsec_buffer = np.zeros((numba_num_threads, num_grid), dtype=np.float64)
    emi_xsec_buffer = np.zeros((numba_num_threads, num_grid), dtype=np.float64)

    for i in numba.prange(num_trans):
        thread_id = numba.get_thread_id()

        energy_fi_i = energy_fi[i]
        abs_coef_i = abs_coef[i]
        emi_coef_i = emi_coef[i]
        gamma_inv_i = gamma_inv[i]
        sigma_i = sigma[i]

        gh_roots_sigma = gh_roots * sigma_i
        start_sigma = grid_min - gh_roots_sigma
        end_sigma = grid_max - gh_roots_sigma
        b_corr = np.pi / (
                np.arctan((end_sigma - energy_fi_i) * gamma_inv_i)
                - np.arctan((start_sigma - energy_fi_i) * gamma_inv_i)
        )
        gh_weights_b_corr = gh_weights * b_corr

        transition_min = energy_fi_i - cutoff
        transition_max = energy_fi_i + cutoff

        j_start = _binary_search_right(bin_edges, transition_min) - 1
        j_start = max(0, j_start)
        j_end = _binary_search_left(bin_edges, transition_max, start=j_start)
        j_end = min(num_grid, j_end)

        for j in range(j_start, j_end):
            wn_j = wn_grid[j]

            shift_sigma = wn_j - energy_fi_i - gh_roots_sigma
            bin_term = np.sum(
                gh_weights_b_corr
                * (
                        np.arctan((shift_sigma + half_bin_width) * gamma_inv_i)
                        - np.arctan((shift_sigma - half_bin_width) * gamma_inv_i)
                )
            )

            abs_xsec_buffer[thread_id, j] += abs_coef_i * bin_term
            emi_xsec_buffer[thread_id, j] += emi_coef_i * bin_term
    # Accumulate buffers.
    _abs_xsec = np.zeros(num_grid, dtype=np.float64)
    _emi_xsec = np.zeros(num_grid, dtype=np.float64)
    for k in numba.prange(num_grid):
        abs_xsec_point = 0.0
        emi_xsec_point = 0.0
        for t in range(numba_num_threads):
            abs_xsec_point += abs_xsec_buffer[t, k]
            emi_xsec_point += emi_xsec_buffer[t, k]
        _abs_xsec[k] = abs_xsec_point
        _emi_xsec[k] = emi_xsec_point
    return _abs_xsec, _emi_xsec


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _continuum_binned_gauss_variable_width(
        wn_grid: npt.NDArray[np.float64],
        n_f: npt.NDArray[np.float64],
        n_i: npt.NDArray[np.float64],
        a_fi: npt.NDArray[np.float64],
        g_f: npt.NDArray[np.float64],
        g_i: npt.NDArray[np.float64],
        energy_fi: npt.NDArray[np.float64],
        v_f: npt.NDArray[np.float64],
        temperature: float,
        species_mass: float,
        box_length: float,
        erf_lut: npt.NDArray[np.float64] = global_erf_lut,
        erf_arg_max: float = global_erf_arg_max,
        numba_num_threads: int = _DEFAULT_NUM_THREADS,
) -> npt.NDArray[np.float64]:
    """
    Note that the area of a Gaussian within 5 * HWHM is equal to erf(5*sqrt(ln(2))). The reciprocal of this can be used
    to recover the total intensity, though this value is 99.99999960685046%. Hence, this should only affect the
    9th decimal place, and Einstein A coefficients are only provided to 5 significant figures.

    Parameters
    ----------
    wn_grid
    n_f
    n_i
    a_fi
    g_f
    g_i
    energy_fi
    v_f
    temperature
    species_mass
    box_length
    erf_lut
    erf_arg_max
    numba_num_threads

    Returns
    -------

    """
    assert n_i.shape[0] == n_f.shape[0] == a_fi.shape[0] == g_f.shape[0] == g_i.shape[0] == energy_fi.shape[0] == \
           v_f.shape[0]

    sqrtln2 = math.sqrt(math.log(2))
    num_grid = wn_grid.shape[0]
    num_trans = energy_fi.shape[0]

    min_cutoff = 25.0
    max_cutoff = 3000.0
    cutoff_fwhm_multiple = 5.0

    # bin_widths = np.zeros(num_grid + 1, dtype=np.float64)
    # bin_widths[1:-1] = (wn_grid[:-1] + wn_grid[1:]) / 2.0 - wn_grid[:-1]
    bin_edges = np.empty(num_grid + 1, dtype=np.float64)
    bin_edges[0] = wn_grid[0] - (wn_grid[1] - wn_grid[0]) * 0.5
    for j in range(1, num_grid):
        bin_edges[j] = (wn_grid[j - 1] + wn_grid[j]) * 0.5
    bin_edges[-1] = wn_grid[-1] + (wn_grid[-1] - wn_grid[-2]) * 0.5

    bin_widths_lower = wn_grid - bin_edges[:-1]
    bin_widths_upper = bin_edges[1:] - wn_grid
    inv_bin_widths = 1.0 / (bin_widths_lower + bin_widths_upper)

    abs_coef = a_fi * ((n_i * g_f / g_i) - n_f) / (const_16_pi_c * energy_fi * energy_fi)

    alpha_doppler = energy_fi * const_sqrt_2_NA_kB_log2_on_c * np.sqrt(temperature / species_mass)
    alpha_box = const_h_on_8_c * (2 * v_f + 1) / (species_mass * const_amu * box_length * box_length)
    alpha_total = alpha_box + alpha_doppler
    sqrtln2_on_alpha = sqrtln2 / alpha_total
    cutoff = np.clip(alpha_total * cutoff_fwhm_multiple, a_min=min_cutoff, a_max=max_cutoff)

    xsec_buffer = np.zeros((numba_num_threads, num_grid), dtype=np.float64)

    for i in numba.prange(num_trans):
        thread_id = numba.get_thread_id()

        energy_fi_i = energy_fi[i]
        abs_coef_i = abs_coef[i]
        sqrtln2_on_alpha_i = sqrtln2_on_alpha[i]
        cutoff_i = cutoff[i]

        transition_min = energy_fi_i - cutoff_i
        transition_max = energy_fi_i + cutoff_i

        j_start = _binary_search_right(bin_edges, transition_min) - 1
        j_start = max(0, j_start)
        j_end = _binary_search_left(bin_edges, transition_max, start=j_start)
        j_end = min(num_grid, j_end)

        for j in range(j_start, j_end):
            wn_j = wn_grid[j]
            upper_width_j = bin_widths_upper[j]
            lower_width_j = bin_widths_lower[j]
            inv_bin_width_j = inv_bin_widths[j]

            wn_shift = wn_j - energy_fi_i

            erf_plus_arg = sqrtln2_on_alpha_i * (wn_shift + upper_width_j)
            erf_minus_arg = sqrtln2_on_alpha_i * (wn_shift - lower_width_j)

            erf_plus = get_erf_lut(erf_plus_arg, erf_lut, erf_arg_max)
            erf_minus = get_erf_lut(erf_minus_arg, erf_lut, erf_arg_max)

            xsec_buffer[thread_id, j] += abs_coef_i * (erf_plus - erf_minus) * inv_bin_width_j
    # Accumulate buffers.
    _abs_xsec = np.zeros(num_grid, dtype=np.float64)
    for k in numba.prange(num_grid):
        xsec_point = 0.0
        for t in range(numba_num_threads):
            xsec_point += xsec_buffer[t, k]
        _abs_xsec[k] = xsec_point
    return _abs_xsec


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _continuum_binned_gauss_fixed_width(
        wn_grid: npt.NDArray[np.float64],
        n_f: npt.NDArray[np.float64],
        n_i: npt.NDArray[np.float64],
        a_fi: npt.NDArray[np.float64],
        g_f: npt.NDArray[np.float64],
        g_i: npt.NDArray[np.float64],
        energy_fi: npt.NDArray[np.float64],
        v_f: npt.NDArray[np.float64],
        temperature: float,
        species_mass: float,
        box_length: float,
        half_bin_width: np.float64,
        erf_lut: npt.NDArray[np.float64] = global_erf_lut,
        erf_arg_max: float = global_erf_arg_max,
        numba_num_threads: int = _DEFAULT_NUM_THREADS,
) -> npt.NDArray[np.float64]:
    assert n_i.shape[0] == n_f.shape[0] == a_fi.shape[0] == g_f.shape[0] == g_i.shape[0] == energy_fi.shape[0] == \
           v_f.shape[0]

    twice_bin_width = 4.0 * half_bin_width
    sqrtln2 = math.sqrt(math.log(2))

    num_grid = wn_grid.shape[0]
    num_trans = energy_fi.shape[0]

    min_cutoff = 25.0
    max_cutoff = 3000.0
    cutoff_fwhm_multiple = 5.0

    bin_edges = np.empty(num_grid + 1, dtype=np.float64)
    bin_edges[0] = wn_grid[0] - (wn_grid[1] - wn_grid[0]) * 0.5
    for j in range(1, num_grid):
        bin_edges[j] = (wn_grid[j - 1] + wn_grid[j]) * 0.5
    bin_edges[-1] = wn_grid[-1] + (wn_grid[-1] - wn_grid[-2]) * 0.5

    abs_coef = a_fi * ((n_i * g_f / g_i) - n_f) / (const_8_pi_c * twice_bin_width * energy_fi * energy_fi)

    alpha_doppler = energy_fi * const_sqrt_2_NA_kB_log2_on_c * np.sqrt(temperature / species_mass)
    alpha_box = const_h_on_8_c * (2 * v_f + 1) / (species_mass * const_amu * box_length * box_length)
    alpha_total = alpha_box + alpha_doppler
    sqrtln2_on_alpha = sqrtln2 / alpha_total
    cutoff = np.clip(alpha_total * cutoff_fwhm_multiple, a_min=min_cutoff, a_max=max_cutoff)

    xsec_buffer = np.zeros((numba_num_threads, num_grid), dtype=np.float64)

    for i in numba.prange(num_trans):
        thread_id = numba.get_thread_id()

        energy_fi_i = energy_fi[i]
        abs_coef_i = abs_coef[i]
        sqrtln2_on_alpha_i = sqrtln2_on_alpha[i]
        cutoff_i = cutoff[i]

        transition_min = energy_fi_i - cutoff_i
        transition_max = energy_fi_i + cutoff_i

        j_start = _binary_search_right(bin_edges, transition_min) - 1
        j_start = max(0, j_start)
        j_end = _binary_search_left(bin_edges, transition_max, start=j_start)
        j_end = min(num_grid, j_end)

        for j in range(j_start, j_end):
            wn_j = wn_grid[j]

            wn_shift = wn_j - energy_fi_i

            erf_plus_arg = sqrtln2_on_alpha_i * (wn_shift + half_bin_width)
            erf_minus_arg = sqrtln2_on_alpha_i * (wn_shift - half_bin_width)

            erf_plus = get_erf_lut(erf_plus_arg, erf_lut, erf_arg_max)
            erf_minus = get_erf_lut(erf_minus_arg, erf_lut, erf_arg_max)

            xsec_buffer[thread_id, j] += abs_coef_i * (erf_plus - erf_minus)
    # Accumulate buffers.
    _abs_xsec = np.zeros(num_grid, dtype=np.float64)
    for k in numba.prange(num_grid):
        xsec_point = 0.0
        for t in range(numba_num_threads):
            xsec_point += xsec_buffer[t, k]
        _abs_xsec[k] = xsec_point
    return _abs_xsec
