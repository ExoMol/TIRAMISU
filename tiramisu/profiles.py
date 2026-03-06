import io
import time
import logging
import math
import pathlib
import typing as t
from functools import partial, lru_cache

import numba
import numpy as np
import polars as pl
from astropy import constants as ac, units as u
from dask import dataframe as dd
from numpy import typing as npt
from scipy.special import roots_hermite, erf

from tiramisu.config import _DEFAULT_NUM_THREADS, _INTENSITY_CUTOFF, _N_GH_QUAD_POINTS
from tiramisu.nlte import calc_einstein_b_fi, calc_einstein_b_if
from tiramisu.numerics import simpson_normalise_2d

log = logging.getLogger(__name__)

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


def create_erf_lut(
        n_points: int = 20000, arg_max: float = 6.0
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Create lookup table for erf() for samlpling by :func:`tiramisu.nlte.get_erf_lut`
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
    n_points = erf_lut.shape[0]

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
    wn_grid : (n_grid,)
        The full wavenumber grid.
    temperature_profile : (n_nlte_layers,)
        Kinetic temperatures for NLTE layers, float values in Kelvin.
    pressure_profile : (n_nlte_layers,)
        Pressure for NLTE layers, float values in bars.
    species_mass : float
        Mass of the species for broadening.
    broad_n : (n_broadeners,)
        Exponential temperature factor for pressure broadening.
    broad_gamma : (n_broadeners, n_nlte_layers)
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
    wn_grid : (n_grid,)
        The full wavenumber grid.
    temperature_profile : (n_nlte_layers,)
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


# ------------------------------------- PROCESS BATCH HELPERS -------------------------------------

def _process_trans_batch_layered(
        trans_batch: pl.DataFrame,
        states: pl.DataFrame,
        broadening_params: t.Optional[t.Any],
        species_mass: float,
        wn_grid: npt.NDArray[np.float64],
        n_lte_layers: int,
        n_layers: int,
        temperature_profile: npt.NDArray[np.float64],  # (n_nlte_layers,) plain float values
        pressure_profile: npt.NDArray[np.float64],  # (n_nlte_layers,) plain float values
) -> t.Tuple[t.Optional[pl.DataFrame], t.Optional[pl.DataFrame]]:
    """
    Layer-aware replacement for _process_trans_batch.

    Instead of being called once per layer by a ProcessPoolExecutor, this is
    called once per batch and processes all NLTE layers simultaneously. The
    join, filter, and energy computation are performed once on layer-invariant
    columns, and the per-layer n_frac populations are assembled into
    (n_layers, n_trans) arrays before being passed to calc_band_profile_layered.

    Parameters
    ----------
    trans_batch : polars.DataFrame
        Polars DataFrame of raw transitions (id_f, id_i, A_fi).
    states : polars.DataFrame
        Polars DataFrame of state data with all per-layer population columns (n_L{l}, n_agg_L{l} for all l).
    broadening_params : optional tuple
        (broad_gamma, broad_n) where broad_gamma has shape (n_broadeners, n_total_layers).
    species_mass : float
    wn_grid : (n_grid,)
    n_lte_layers : int
    n_layers : int
        Total number of layers (LTE + NLTE).
    temperature_profile : (n_nlte_layers,)
        Kinetic temperatures for NLTE layers, plain float values in Kelvin.
    pressure_profile : (n_nlte_layers,)
        Pressures for NLTE layers, plain float values in Pascals.

    Returns
    -------
    band_profile_data : pl.DataFrame or None
        DataFrame with columns [id_agg_f, id_agg_i, abs_profile, ste_profile,
        spe_profile] where each profile column holds a (n_nlte_layers, n_grid)
        array per band.
    agg_batch : pl.DataFrame or None
        Aggregated Einstein rates; computed once and returned (equivalent to
        the nlte_layer_idx == 0 branch in the original).
    """
    n_nlte_layers = n_layers - n_lte_layers

    n_frac_cols = [f"n_frac_nL{nlte_idx}" for nlte_idx in range(n_nlte_layers)]

    states_frac = states.with_columns(
        (
                pl.col(f"n_L{n_lte_layers + nlte_idx}") / pl.col(f"n_agg_L{n_lte_layers + nlte_idx}")
        ).alias(f"n_frac_nL{nlte_idx}")
        for nlte_idx in range(n_nlte_layers)
    )

    invariant_cols_i = ["id", "energy", "g", "id_agg"] + n_frac_cols
    invariant_cols_f = ["id", "energy", "g", "id_agg", "tau"] + n_frac_cols

    states_i = (
        states_frac
        .select(invariant_cols_i)
        .rename({col: f"{col}_i" for col in invariant_cols_i})
    )
    states_f = (
        states_frac
        .select(invariant_cols_f)
        .rename({col: f"{col}_f" for col in invariant_cols_f})
    )

    trans_batch = trans_batch.join(states_i, on="id_i", how="inner")
    trans_batch = trans_batch.join(states_f, on="id_f", how="inner")
    trans_batch = trans_batch.with_columns(
        (pl.col("energy_f") - pl.col("energy_i")).alias("energy_fi")
    )

    wn_min = wn_grid[0]
    wn_max = wn_grid[-1]
    trans_batch = trans_batch.filter(
        (pl.col("energy_f") >= wn_min)
        & (pl.col("energy_i") <= wn_max)
        & (pl.col("energy_fi") >= wn_min)
        & (pl.col("energy_fi") <= wn_max)
    )
    n_trans = trans_batch.height

    if n_trans == 0:
        return None, None

    trans_batch = trans_batch.fill_null(strategy="zero")

    # Broadening parameters: broad_gamma needs shape (n_broadeners, n_nlte_layers).
    # In the original, broadening_params[0] has shape (n_broadeners, n_total_layers) so we slice to the NLTE columns.
    # Default to zeros if absent.
    broad_n = broadening_params[1] if broadening_params is not None else np.zeros(1, dtype=np.float64)
    broad_gamma = (
        broadening_params[0][:, n_lte_layers:]  # (n_broadeners, n_nlte_layers)
        if broadening_params is not None
        else np.zeros((1, n_nlte_layers), dtype=np.float64)
    )

    # Compute band profiles for all bands and all layers at once.
    start_time = time.perf_counter()
    band_profile_partial = partial(
        calc_band_profile_layered,
        wn_grid,
        temperature_profile,
        pressure_profile,
        species_mass,
        broad_n,
        broad_gamma,
        n_nlte_layers,
    )
    band_profile_data = trans_batch.group_by(*["id_agg_f", "id_agg_i"]).map_groups(band_profile_partial)
    log.info(f"Profile duration (all layers) = {time.perf_counter() - start_time:.2f}s.")

    # Compute all rates.
    trans_batch_rates = trans_batch.filter(pl.col("id_agg_f") != pl.col("id_agg_i"))
    b_fi_vals = calc_einstein_b_fi(
        a_fi=trans_batch_rates["A_fi"].to_numpy(),
        energy_fi=(trans_batch_rates["energy_fi"].to_numpy() << 1 / u.cm)
        .to(u.Hz, equivalencies=u.spectral())
        .value,
    )
    b_if_vals = calc_einstein_b_if(
        b_fi=b_fi_vals,
        g_f=trans_batch_rates["g_f"].to_numpy(),
        g_i=trans_batch_rates["g_i"].to_numpy(),
    )
    agg_batch = (
        trans_batch_rates
        .with_columns([
            pl.Series("B_fi", b_fi_vals),
            pl.Series("B_if", b_if_vals),
        ])
        .group_by(["id_agg_f", "id_agg_i"])
        .agg([
            pl.col("A_fi").sum().alias("A_fi"),
            pl.col("B_fi").sum().alias("B_fi"),
            pl.col("B_if").sum().alias("B_if"),
        ])
    )

    return band_profile_data, agg_batch


def _process_continuum_trans_batch_layered(
        trans_batch: pl.DataFrame,
        states: pl.DataFrame,
        species_mass: float,
        wn_grid: npt.NDArray[np.float64],
        n_lte_layers: int,
        n_layers: int,
        cont_box_length: float,
        temperature_profile: npt.NDArray[np.float64],
) -> t.Tuple[t.Optional[pl.DataFrame], t.Optional[pl.DataFrame]]:
    """
    Instead of being called once per layer by a ProcessPoolExecutor, this is
    called once per batch and processes all NLTE layers simultaneously. The
    join, filter, and energy computation are performed once on layer-invariant
    columns, and the per-layer n_frac populations are assembled into
    (n_layers, n_trans) arrays before being passed to calc_band_profile_layered.

    Parameters
    ----------
    trans_batch : polars.DataFrame
        Polars DataFrame of raw transitions (id_f, id_i, A_fi).
    states : polars.DataFrame
        Polars DataFrame of state data with all per-layer population columns (n_L{l}, n_agg_L{l} for all l).
    species_mass : float
        Mass of the species for broadening.
    wn_grid : (n_grid,)
        The full wavenumber grid.
    n_lte_layers : int
        Number of LTE layers.
    n_layers : int
        Total number of layers (LTE + NLTE).
    cont_box_length : float
        Box length off to use in continuum box broadening.
    temperature_profile : (n_nlte_layers,)
        Kinetic temperatures for NLTE layers, plain float values in Kelvin.

    Returns
    -------
    band_profile_data : pl.DataFrame or None
        DataFrame with columns [id_agg_f, id_agg_i, abs_profile, ste_profile,
        spe_profile] where each profile column holds a (n_nlte_layers, n_grid)
        array per band.
    agg_batch : pl.DataFrame or None
        Aggregated Einstein rates; computed once and returned (equivalent to
        the nlte_layer_idx == 0 branch in the original).
    """
    n_nlte_layers = n_layers - n_lte_layers

    n_frac_cols = [f"n_frac_nL{nlte_idx}" for nlte_idx in range(n_nlte_layers)]

    states_frac = states.with_columns(
        (
                pl.col(f"n_L{n_lte_layers + nlte_idx}") / pl.col(f"n_agg_L{n_lte_layers + nlte_idx}")
        ).alias(f"n_frac_nL{nlte_idx}")
        for nlte_idx in range(n_nlte_layers)
    )

    invariant_cols_i = ["id", "energy", "g", "id_agg"] + n_frac_cols
    invariant_cols_f = ["id", "energy", "g", "id_agg", "v"]

    states_i = (
        states_frac
        .select(invariant_cols_i)
        .rename({col: f"{col}_i" for col in invariant_cols_i})
    )
    states_f = (
        states_frac
        .select(invariant_cols_f)
        .rename({col: f"{col}_f" for col in invariant_cols_f})
    )

    trans_batch = trans_batch.join(states_i, on="id_i", how="inner")
    trans_batch = trans_batch.join(states_f, on="id_f", how="inner")
    trans_batch = trans_batch.with_columns(
        (pl.col("energy_f") - pl.col("energy_i")).alias("energy_fi")
    )

    wn_min = wn_grid[0]
    wn_max = wn_grid[-1]
    trans_batch = trans_batch.filter(
        (pl.col("energy_f") >= wn_min)
        & (pl.col("energy_i") <= wn_max)
        & (pl.col("energy_fi") >= wn_min)
        & (pl.col("energy_fi") <= wn_max)
    )

    n_trans = trans_batch.height
    if n_trans == 0:
        return None, None

    trans_batch = trans_batch.fill_null(strategy="zero")
    # Compute band profiles for all bands and all layers at once.
    start_time = time.perf_counter()
    band_profile_partial = partial(
        calc_continuum_band_profile_layered,
        wn_grid,
        temperature_profile,
        species_mass,
        cont_box_length,
        n_nlte_layers,
    )
    band_profile_data = trans_batch.group_by("id_agg_i").map_groups(band_profile_partial)
    log.info(f"Profile duration (all layers) = {time.perf_counter() - start_time:.2f}s.")

    # Compute all rates.
    trans_batch_rates = trans_batch.filter(pl.col("id_agg_f") != pl.col("id_agg_i"))
    b_fi_vals = calc_einstein_b_fi(
        a_fi=trans_batch_rates["A_fi"].to_numpy(),
        energy_fi=(trans_batch_rates["energy_fi"].to_numpy() << 1 / u.cm)
        .to(u.Hz, equivalencies=u.spectral())
        .value,
    )
    b_if_vals = calc_einstein_b_if(
        b_fi=b_fi_vals,
        g_f=trans_batch_rates["g_f"].to_numpy(),
        g_i=trans_batch_rates["g_i"].to_numpy(),
    )
    agg_batch = (
        trans_batch_rates
        .with_columns([
            pl.Series("B_fi", b_fi_vals),
            pl.Series("B_if", b_if_vals),
        ])
        .group_by(["id_agg_f", "id_agg_i"])
        .agg([
            pl.col("A_fi").sum().alias("A_fi"),
            pl.col("B_fi").sum().alias("B_fi"),
            pl.col("B_if").sum().alias("B_if"),
        ])
    )

    return band_profile_data, agg_batch


# ------------------------------------- COMPACT PROFILE & STORE CLASSES -------------------------------------


def _sum_profiles(group: pl.DataFrame) -> pl.DataFrame:
    profile_label = group.columns[-1]
    summed_profiles = np.stack(group[profile_label].to_numpy(), axis=0).sum(axis=0)
    return pl.DataFrame({
        "id_agg_f": [group["id_agg_f"][0]],
        "id_agg_i": [group["id_agg_i"][0]],
        profile_label: [summed_profiles],
    })


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
        1-dimensional array storing populations of state ID corresponding to arry index.
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
    __slots__ = ["temporary_profiles", "profiles", "offsets", "start_idxs", "key_idx_map", "key_lookup"]

    def __init__(self):
        self.temporary_profiles: pl.DataFrame | None = None
        self.profiles: npt.NDArray[np.float64] | None = None
        self.offsets: npt.NDArray[np.int64] | None = None
        self.start_idxs: npt.NDArray[np.int64] | None = None
        self.key_idx_map: npt.NDArray[np.int64] | None = None
        self.key_lookup: t.Dict[t.Tuple[int, int] | int, int] = {}

    def add_batch(self, batch: pl.DataFrame) -> None:
        """
        batch must contain id_agg_f and id_agg_i columns plus a third column containing the relevant profile.

        :param batch:
        :return:
        """
        if self.temporary_profiles is None:
            self.temporary_profiles = batch
        else:
            self.temporary_profiles = pl.concat([self.temporary_profiles, batch])
            self.temporary_profiles = self.temporary_profiles.group_by("id_agg_f", "id_agg_i").map_groups(_sum_profiles)

    def finalise(self) -> None:
        """
        Trimmed profiles are stored contiguously in self.profiles.

        The starting indices within self.profiles of each individual profile is contained within self.offsets. An extra
        terminator is stored at the end of self.offsets equal to the total length of self.profiles; this is so the start
        and end indices of a given profile can always be obtained by looking at the current and next offset.

        The starting position of each profile on the main wavenumber grid is stored at the corresponding index in
        self.start_idxs.

        The upper and lower state IDs are stored in self.key_idx_map; the index of theh first dimension of this array
        matches the corresponding index in self.offsets and self.start_idxs. This is used for fast cross-section
        reconstruction in :func:`tiramisu.nlte.CompactProfile.build_xsec`. A fast dictionary lookup for accessing
        individual bands is stored in self.key_lookup, used by :func:`tiramisu.nlte.CompactProfile.get_profile`.

        Returns
        -------

        """
        num_profiles = self.temporary_profiles.height
        if num_profiles == 0:
            log.warning("CompactProfile finalising with 0 inputs!")
            return

        n_cols = len(self.temporary_profiles.columns)
        if n_cols not in (2, 3):
            raise ValueError("Unsupported number of columns in CompactProfile input.")

        if n_cols == 3:
            # Key is a tuple of the first two elements: (row[0], row[1])
            key_selector = lambda row_data: (row_data[0], row_data[1])
        else:  # n_cols == 2
            # Key is the first element: row[0]
            key_selector = lambda row_data: (-1, row_data[0])

        profile_idx = n_cols - 1

        estimate_profile_len = len(self.temporary_profiles.row(0)[profile_idx])
        self.profiles = np.zeros(num_profiles * estimate_profile_len, dtype=np.float64)
        # Offsets contains a terminator at the end of the array.
        self.offsets = np.zeros(num_profiles + 1, dtype=int)
        self.start_idxs = np.zeros(num_profiles, dtype=int)
        self.key_idx_map = np.empty((num_profiles, 2), dtype=int)

        current_offset = 0
        store_idx = 0
        for idx, row in enumerate(self.temporary_profiles.iter_rows(named=False)):
            key = key_selector(row)
            profile = np.array(row[profile_idx])

            above = profile >= _INTENSITY_CUTOFF
            if not np.any(above):
                # Nothing above threshold: skip or store empty?
                continue
            start_idx = np.argmax(above)
            end_idx = len(profile) - np.argmax(above[::-1])
            trimmed = profile[start_idx:end_idx]

            length = len(trimmed)
            if current_offset + length > len(self.profiles):
                remaining = num_profiles - idx
                self.profiles.resize(len(self.profiles) + remaining * length, refcheck=False)

            self.profiles[current_offset: current_offset + length] = trimmed
            self.offsets[store_idx] = current_offset
            self.start_idxs[store_idx] = start_idx
            self.key_idx_map[store_idx, :] = key
            self.key_lookup[key] = store_idx

            current_offset += length
            store_idx += 1

        # num_final_profiles = len(self.key_idx_map)
        num_final_profiles = store_idx + 1
        if num_final_profiles < num_profiles:
            # Some profiles where skipped (below cut-off); resize arrays.
            self.offsets = self.offsets[:num_final_profiles + 1]
            self.start_idxs = self.start_idxs[:num_final_profiles]
        # Add terminator and trim profiles to this length, in case it was overestimated or extended.
        self.offsets[-1] = current_offset
        self.profiles = self.profiles[:current_offset]
        log.info(f"Finalised CompactProfile with {current_offset} points for {len(self.start_idxs)} bands.")

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


class ProfileStore:
    """
    Store :class:`tiramisu.nlte.CompactProfile` objects representing the species' absorption, stimulated emission and
    spontaneous emission profiles. Used for bound-bound transitions (including quasi-bound).
    """
    __slots__ = ["n_layers", "n_grid", "abs_profiles", "ste_profiles", "spe_profiles"]

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self.n_grid: int | None = None  # This is set on the first add_batch() call.
        # Storage for final profiles.
        self.abs_profiles = [CompactProfile() for _ in range(n_layers)]  # Absorption
        self.ste_profiles = [CompactProfile() for _ in range(n_layers)]  # Stimulated Emission
        self.spe_profiles = [CompactProfile() for _ in range(n_layers)]  # Spontaneous Emission

    # def add_layer_batch(self, batch: pl.DataFrame, layer_idx: int) -> None:
    #     """
    #     The batch is a polars DataFrame containing id_agg_f and id_agg_i keys plus arrays representing each profile.
    #
    #     Parameters
    #     ----------
    #     batch
    #     layer_idx
    #
    #     Returns
    #     -------
    #
    #     """
    #     self.abs_profiles[layer_idx].add_batch(
    #         batch.select(pl.col("id_agg_f"), pl.col("id_agg_i"), pl.col("abs_profile"))
    #     )
    #     self.ste_profiles[layer_idx].add_batch(
    #         batch.select(pl.col("id_agg_f"), pl.col("id_agg_i"), pl.col("ste_profile"))
    #     )
    #     self.spe_profiles[layer_idx].add_batch(
    #         batch.select(pl.col("id_agg_f"), pl.col("id_agg_i"), pl.col("spe_profile"))
    #     )

    def add_batch(self, batch: pl.DataFrame) -> None:
        """
        Layered interface. Accepts a batch whose profile columns contain ravelled (n_nlte_layers * n_grid,) arrays
        alongside n_layers and n_grid metadata columns, produced by calc_band_profile_layered.

        Each band row is unravelled and the per-layer 1D slices are routed to the corresponding CompactProfile,
        preserving the existing behaviour that CompactProfile.add_batch always receives 1D profiles.

        Consistency checks
        ------------------
        - batch n_layers must equal self.n_layers on every call.
        - n_grid must be consistent across all calls; stored on first call.

        Parameters
        ----------
        batch: polars.DataFrame
            Output from :func:`~profiles.ProfileStore.calc_band_profile_layered`.

        Returns
        -------

        """
        if batch.height == 0:
            return

        batch_n_layers = batch["n_layers"][0]
        if batch_n_layers != self.n_layers:
            raise ValueError(f"ProfileStore has n_layers={self.n_layers} but batch reports n_layers={batch_n_layers}.")

        batch_n_grid = batch["n_grid"][0]
        if self.n_grid is None:
            self.n_grid = batch_n_grid
        elif batch_n_grid != self.n_grid:
            raise ValueError(f"ProfileStore has n_grid={self.n_grid} but batch reports n_grid={batch_n_grid}.")

        id_agg_f = batch["id_agg_f"]
        id_agg_i = batch["id_agg_i"]
        n_bands = batch.height

        abs_ravelled = np.stack(batch["abs_profile"].to_list()).reshape(n_bands, self.n_layers, self.n_grid)
        ste_ravelled = np.stack(batch["ste_profile"].to_list()).reshape(n_bands, self.n_layers, self.n_grid)
        spe_ravelled = np.stack(batch["spe_profile"].to_list()).reshape(n_bands, self.n_layers, self.n_grid)

        for layer_idx in range(self.n_layers):
            # Slice this layer's profiles: (n_bands, n_grid) -> list of 1D arrays.
            self.abs_profiles[layer_idx].add_batch(pl.DataFrame({
                "id_agg_f": id_agg_f,
                "id_agg_i": id_agg_i,
                "abs_profile": list(abs_ravelled[:, layer_idx, :]),
            }))
            self.ste_profiles[layer_idx].add_batch(pl.DataFrame({
                "id_agg_f": id_agg_f,
                "id_agg_i": id_agg_i,
                "ste_profile": list(ste_ravelled[:, layer_idx, :]),
            }))
            self.spe_profiles[layer_idx].add_batch(pl.DataFrame({
                "id_agg_f": id_agg_f,
                "id_agg_i": id_agg_i,
                "spe_profile": list(spe_ravelled[:, layer_idx, :]),
            }))

    def finalise(self) -> None:
        for layer_idx in range(self.n_layers):
            self.abs_profiles[layer_idx].finalise()
            self.ste_profiles[layer_idx].finalise()
            self.spe_profiles[layer_idx].finalise()

    def get_profiles(
            self, layer_idx: int, key: t.Tuple[int, int]
    ) -> t.Tuple[t.Tuple[npt.NDArray, int], t.Tuple[npt.NDArray, int], t.Tuple[npt.NDArray, int]]:
        return (
            self.abs_profiles[layer_idx].get_profile(key),
            self.ste_profiles[layer_idx].get_profile(key),
            self.spe_profiles[layer_idx].get_profile(key),
        )

    @lru_cache(maxsize=1000)
    def get_profile(self, layer_idx: int, key: t.Tuple[int, int], profile_type: str) -> t.Tuple[npt.NDArray, int]:
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

    def precompute_downward_emission_profiles(
            self, layer_idx: int, id_agg_cutoff: int, num_grid: int
    ) -> npt.NDArray[np.float64]:
        """
        Precompute spontaneous emission profiles for all upper states.

        Each profile represents the total downward emission from a given upper state summed over all lower states.

        Parameters
        ----------
        layer_idx : int
        id_agg_cutoff : int
        num_grid : int

        Returns
        -------
        all_profiles : np.ndarray, shape (n_agg_states, num_grid)
        """
        # all_profiles = np.zeros((n_agg_states, num_grid), dtype=np.float64)
        #
        # for o_idx in range(n_agg_states):
        #     all_profiles[o_idx] = self.spe_profiles[layer_idx].get_emission_from_upper(
        #         upper_state_id=o_idx, num_grid=num_grid,
        #     )
        #
        # return all_profiles
        return self.spe_profiles[layer_idx].get_all_emission_from_upper(
            id_agg_cutoff=id_agg_cutoff,
            num_grid=num_grid,
        )

    def precompute_normalised_downward_emission_profiles(
            self,
            layer_idx: int,
            id_agg_cutoff: int,
            wn_grid: u.Quantity,
    ) -> npt.NDArray[np.float64]:
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
        return simpson_normalise_2d(y_data=all_profiles, x_data=wn_grid.value) << 1 / wn_grid.unit


class ContinuumProfileStore:
    __slots__ = ["n_layers","n_grid",  "abs_profiles"]

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self.n_grid: int | None = None  # This is set on the first add_batch() call.
        # Storage for final profiles.
        self.abs_profiles = [CompactProfile() for _ in range(n_layers)]  # Absorption
        # self.ste_profiles = [CompactProfile() for _ in range(n_layers)]  # Stimulated Emission
        # self.spe_profiles = [CompactProfile() for _ in range(n_layers)]  # Spontaneous Emission

    def add_layer_batch(self, batch: pl.DataFrame, layer_idx: int) -> None:
        """
        The batch is a polars DataFrame containing id_agg_i keys plus arrays representing each profile.
        Continuum profiles are currently agnostic of any upper state aggregated identifier, as individual continuum
        states are not currently treated distinctly.


        :param batch:
        :param layer_idx:
        :return:
        """
        self.abs_profiles[layer_idx].add_batch(
            batch.select(pl.col("id_agg_i"), pl.col("abs_profile"))
        )
        # self.ste_profiles[layer_idx].add_layer_batch(
        #     batch.select(pl.col("id_agg_i"), pl.col("ste_profile"))
        # )
        # self.spe_profiles[layer_idx].add_layer_batch(
        #     batch.select(pl.col("id_agg_i"), pl.col("spe_profile"))
        # )

    def add_batch(self, batch: pl.DataFrame) -> None:
        """
        Layered interface. Accepts a batch whose profile columns contain ravelled (n_nlte_layers * n_grid,) arrays
        alongside n_layers and n_grid metadata columns, produced by calc_continuum_band_profile_layered.

        Each band row is unravelled and the per-layer 1D slices are routed to the corresponding CompactProfile,
        preserving the existing behaviour that CompactProfile.add_batch always receives 1D profiles.

        Consistency checks
        ------------------
        - batch n_layers must equal self.n_layers on every call.
        - n_grid must be consistent across all calls; stored on first call.

        Parameters
        ----------
        batch: polars.DataFrame
            Output from :func:`~profiles.ProfileStore.calc_band_profile_layered`.

        Returns
        -------

        """
        if batch.height == 0:
            return

        batch_n_layers = batch["n_layers"][0]
        if batch_n_layers != self.n_layers:
            raise ValueError(
                f"ContinuumProfileStore has n_layers={self.n_layers} but batch reports n_layers={batch_n_layers}."
            )

        batch_n_grid = batch["n_grid"][0]
        if self.n_grid is None:
            self.n_grid = batch_n_grid
        elif batch_n_grid != self.n_grid:
            raise ValueError(f"ContinuumProfileStore has n_grid={self.n_grid} but batch reports n_grid={batch_n_grid}.")

        id_agg_f = batch["id_agg_f"]
        id_agg_i = batch["id_agg_i"]
        n_bands = batch.height

        abs_ravelled = np.stack(batch["abs_profile"].to_list()).reshape(n_bands, self.n_layers, self.n_grid)
        # ste_ravelled = np.stack(batch["ste_profile"].to_list()).reshape(n_bands, self.n_layers, self.n_grid)
        # spe_ravelled = np.stack(batch["spe_profile"].to_list()).reshape(n_bands, self.n_layers, self.n_grid)

        for layer_idx in range(self.n_layers):
            # Slice this layer's profiles: (n_bands, n_grid) -> list of 1D arrays.
            self.abs_profiles[layer_idx].add_batch(pl.DataFrame({
                "id_agg_f": id_agg_f,
                "id_agg_i": id_agg_i,
                "abs_profile": list(abs_ravelled[:, layer_idx, :]),
            }))
            # self.ste_profiles[layer_idx].add_batch(pl.DataFrame({
            #     "id_agg_f": id_agg_f,
            #     "id_agg_i": id_agg_i,
            #     "ste_profile": list(ste_ravelled[:, layer_idx, :]),
            # }))
            # self.spe_profiles[layer_idx].add_batch(pl.DataFrame({
            #     "id_agg_f": id_agg_f,
            #     "id_agg_i": id_agg_i,
            #     "spe_profile": list(spe_ravelled[:, layer_idx, :]),
            # }))

    def finalise(self) -> None:
        for layer_idx in range(self.n_layers):
            self.abs_profiles[layer_idx].finalise()
            # self.ste_profiles[layer_idx].finalise()
            # self.spe_profiles[layer_idx].finalise()

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


# def calc_cont_band_profile(
#         wn_grid: npt.NDArray[np.float64],
#         temperature: float,
#         species_mass: float,
#         box_length: float,
#         group: pl.DataFrame,
# ) -> pl.DataFrame:
#     a_fi = np.ascontiguousarray(group["A_fi"].to_numpy())
#     g_i = np.ascontiguousarray(group["g_i"].to_numpy())
#     g_f = np.ascontiguousarray(group["g_f"].to_numpy())
#     energy_fi = np.ascontiguousarray(group["energy_fi"].to_numpy())
#     n_frac_i = np.ascontiguousarray(group["n_frac_i"].to_numpy())
#     n_frac_f = np.ascontiguousarray(group["n_frac_f"].to_numpy())
#     v_f = np.ascontiguousarray(group["v_f"].to_numpy())
#
#     assert a_fi.shape[0] == g_i.shape[0] == g_f.shape[0] == energy_fi.shape[0] == n_frac_i.shape[0] == n_frac_f.shape[0]
#
#     if energy_fi.min() > wn_grid.max() or energy_fi.max() < wn_grid.min():
#         raise RuntimeError("Computing band profile for band with no transitions on grid: check workflow logic.")
#
#     is_fixed_width = np.all(np.isclose(np.diff(wn_grid), np.abs(wn_grid[1] - wn_grid[0]), atol=0))
#
#     if is_fixed_width:
#         _abs_xsec = _continuum_binned_gauss_fixed_width(
#             wn_grid=wn_grid,
#             n_f=n_frac_f,
#             n_i=n_frac_i,
#             a_fi=a_fi,
#             g_f=g_f,
#             g_i=g_i,
#             energy_fi=energy_fi,
#             v_f=v_f,
#             temperature=temperature,
#             species_mass=species_mass,
#             box_length=box_length,
#             half_bin_width=np.abs(wn_grid[1] - wn_grid[0]) / 2.0,
#         )
#     else:
#         _abs_xsec = _continuum_binned_gauss_variable_width(
#             wn_grid=wn_grid,
#             n_f=n_frac_f,
#             n_i=n_frac_i,
#             a_fi=a_fi,
#             g_f=g_f,
#             g_i=g_i,
#             energy_fi=energy_fi,
#             v_f=v_f,
#             temperature=temperature,
#             species_mass=species_mass,
#             box_length=box_length,
#         )
#     return pl.DataFrame({
#         "id_agg_i": [group["id_agg_i"][0]],
#         "abs_profile": [_abs_xsec],
#     })
#
#
# def calc_band_profile(
#         wn_grid: npt.NDArray[np.float64],
#         temperature: float,
#         pressure: float,
#         species_mass: float,
#         broad_n: npt.NDArray[np.float64],
#         broad_gamma: npt.NDArray[np.float64],
#         group: pl.DataFrame,
# ) -> pl.DataFrame:
#     a_fi = np.ascontiguousarray(group["A_fi"].to_numpy())
#     g_i = np.ascontiguousarray(group["g_i"].to_numpy())
#     g_f = np.ascontiguousarray(group["g_f"].to_numpy())
#     energy_fi = np.ascontiguousarray(group["energy_fi"].to_numpy())
#     n_frac_i = np.ascontiguousarray(group["n_frac_i"].to_numpy())
#     n_frac_f = np.ascontiguousarray(group["n_frac_f"].to_numpy())
#     tau_f = np.ascontiguousarray(group["tau_f"].to_numpy())
#
#     assert a_fi.shape[0] == g_i.shape[0] == g_f.shape[0] == energy_fi.shape[0] == n_frac_i.shape[0] == n_frac_f.shape[0]
#
#     if energy_fi.min() > wn_grid.max() or energy_fi.max() < wn_grid.min():
#         raise RuntimeError("Computing band profile for band with no transitions on grid: check workflow logic.")
#
#     if (broad_n is None) ^ (broad_gamma is None):
#         raise RuntimeError(f"Either broadening n or gamma missing: n={broad_n}, gamma={broad_gamma}.")
#
#     is_fixed_width = np.all(np.isclose(np.diff(wn_grid), np.abs(wn_grid[1] - wn_grid[0]), atol=0))
#
#     gh_roots, gh_weights = roots_hermite(_N_GH_QUAD_POINTS)
#
#     if is_fixed_width:
#         _abs_xsec, _ste_xsec, _spe_xsec = _band_profile_binned_voigt_fixed_width(
#             wn_grid=wn_grid,
#             n_i=n_frac_i,
#             n_f=n_frac_f,
#             a_fi=a_fi,
#             g_f=g_f,
#             g_i=g_i,
#             energy_fi=energy_fi,
#             lifetimes=tau_f,
#             temperature=temperature,
#             pressure=pressure,
#             broad_n=broad_n,
#             broad_gamma=broad_gamma,
#             species_mass=species_mass,
#             half_bin_width=np.abs(wn_grid[1] - wn_grid[0]) / 2.0,
#             gh_roots=gh_roots,
#             gh_weights=gh_weights,
#         )
#     else:
#         _abs_xsec, _ste_xsec, _spe_xsec = _band_profile_binned_voigt_variable_width(
#             wn_grid=wn_grid,
#             n_i=n_frac_i,
#             n_f=n_frac_f,
#             a_fi=a_fi,
#             g_f=g_f,
#             g_i=g_i,
#             energy_fi=energy_fi,
#             lifetimes=tau_f,
#             temperature=temperature,
#             pressure=pressure,
#             broad_n=broad_n,
#             broad_gamma=broad_gamma,
#             species_mass=species_mass,
#             gh_roots=gh_roots,
#             gh_weights=gh_weights,
#         )
#     return pl.DataFrame({
#         "id_agg_f": [group["id_agg_f"][0]],
#         "id_agg_i": [group["id_agg_i"][0]],
#         "abs_profile": [_abs_xsec],
#         "ste_profile": [_ste_xsec],
#         "spe_profile": [_spe_xsec],
#     })


# ------------------------------------- OLD XSEC CALCULATIONS -------------------------------------

def abs_emi_xsec(
        states: pl.DataFrame,
        trans_files: t.List[pathlib.Path],
        layer_idx: int,
        temperature: u.Quantity,
        pressure: u.Quantity,
        species_mass: float,
        wn_grid: npt.NDArray[np.float64],
        broad_n: npt.NDArray[np.float64] = None,
        broad_gamma: npt.NDArray[np.float64] = None,
        n_gh_quad_points: int = _N_GH_QUAD_POINTS,
):
    half_bin_width = abs(wn_grid[1] - wn_grid[0]) / 2.0
    # TODO: Check is_fixed_width rigour!
    is_fixed_width = np.all(np.isclose(np.diff(wn_grid), np.abs(wn_grid[1] - wn_grid[0]), atol=0))
    num_grid = wn_grid.shape[0]
    abs_xsec = np.zeros(num_grid, dtype=np.float64)
    emi_xsec = np.zeros(num_grid, dtype=np.float64)

    wn_min = wn_grid[0]
    wn_max = wn_grid[-1]

    n_nlte_col = f"n_nlte_L{layer_idx}"

    pl_states_i = states.select(
        pl.col("id"),
        pl.col("energy").alias("energy_i"),
        pl.col(n_nlte_col).alias("n_nlte_i"),
        pl.col("g").alias("g_i"),
    )
    pl_states_f = states.select(
        pl.col("id"),
        pl.col("energy").alias("energy_f"),
        pl.col(n_nlte_col).alias("n_nlte_f"),
        pl.col("g").alias("g_f"),
        pl.col("tau").alias("tau_f"),
    )

    dask_dtypes = {"id_f": "int64", "id_i": "int64", "A_fi": "float64", }
    dask_blocksize = "256MB"

    for trans_file in trans_files:
        ddf = dd.read_csv(
            trans_file,
            sep=r"\s+",
            engine="python",
            header=None,
            names=["id_f", "id_i", "A_fi"],
            usecols=[0, 1, 2],
            dtype=dask_dtypes,
            blocksize=dask_blocksize,
        )
        delayed_batches = ddf.to_delayed()
        for delayed_batch in delayed_batches:
            trans_batch = pl.from_pandas(delayed_batch.compute())
            trans_batch = trans_batch.join(pl_states_i, left_on="id_i", right_on="id", how="left")
            trans_batch = trans_batch.join(pl_states_f, left_on="id_f", right_on="id", how="left")
            trans_batch = trans_batch.with_columns(
                (pl.col("energy_f") - pl.col("energy_i")).alias("energy_fi")
            )
            trans_batch = trans_batch.filter(
                (pl.col("energy_f") >= wn_min)
                & (pl.col("energy_i") <= wn_max)
                & (pl.col("energy_fi") >= wn_min)
                & (pl.col("energy_fi") <= wn_max)
            )
            final_cols = ["n_nlte_i", "n_nlte_f", "A_fi", "g_f", "g_i", "energy_fi", "tau_f"]
            trans_batch = trans_batch.select(final_cols)
            trans_chunk_np = trans_batch.to_numpy()

            # Matches final_cols ordering.
            n_i = np.ascontiguousarray(trans_chunk_np[:, 0])
            n_f = np.ascontiguousarray(trans_chunk_np[:, 1])
            a_fi = np.ascontiguousarray(trans_chunk_np[:, 2])
            g_f = np.ascontiguousarray(trans_chunk_np[:, 3])
            g_i = np.ascontiguousarray(trans_chunk_np[:, 4])
            energy_fi = np.ascontiguousarray(trans_chunk_np[:, 5])
            lifetimes = np.ascontiguousarray(trans_chunk_np[:, 6])

            if broad_n is None or broad_gamma is None:
                if is_fixed_width:
                    _abs_xsec, _emi_xsec = _abs_emi_binned_doppler_fixed_width(
                        wn_grid=wn_grid,
                        n_i=n_i,
                        n_f=n_f,
                        a_fi=a_fi,
                        g_f=g_f,
                        g_i=g_i,
                        energy_fi=energy_fi,
                        temperature=temperature.value,
                        species_mass=species_mass,
                        half_bin_width=half_bin_width,
                    )
                else:
                    _abs_xsec, _emi_xsec = _abs_emi_binned_doppler_variable_width(
                        wn_grid=wn_grid,
                        n_i=n_i,
                        n_f=n_f,
                        a_fi=a_fi,
                        g_f=g_f,
                        g_i=g_i,
                        energy_fi=energy_fi,
                        temperature=temperature.value,
                        species_mass=species_mass,
                    )
            else:
                gh_roots, gh_weights = roots_hermite(n_gh_quad_points)
                if is_fixed_width:
                    _abs_xsec, _emi_xsec = _abs_emi_binned_voigt_fixed_width(
                        wn_grid=wn_grid,
                        n_i=n_i,
                        n_f=n_f,
                        a_fi=a_fi,
                        g_f=g_f,
                        g_i=g_i,
                        energy_fi=energy_fi,
                        lifetimes=lifetimes,
                        temperature=temperature.value,
                        pressure=pressure.value,
                        broad_n=broad_n,
                        broad_gamma=broad_gamma,
                        species_mass=species_mass,
                        half_bin_width=half_bin_width,
                        gh_roots=gh_roots,
                        gh_weights=gh_weights,
                    )
                else:
                    _abs_xsec, _emi_xsec = _abs_emi_binned_voigt_variable_width(
                        wn_grid=wn_grid,
                        n_i=n_i,
                        n_f=n_f,
                        a_fi=a_fi,
                        g_f=g_f,
                        g_i=g_i,
                        energy_fi=energy_fi,
                        lifetimes=lifetimes,
                        temperature=temperature.value,
                        pressure=pressure.value,
                        broad_n=broad_n,
                        broad_gamma=broad_gamma,
                        species_mass=species_mass,
                        gh_roots=gh_roots,
                        gh_weights=gh_weights,
                    )
            abs_xsec += _abs_xsec
            emi_xsec += _emi_xsec
    return abs_xsec, emi_xsec


def continuum_xsec(
        continuum_states: pl.DataFrame,
        continuum_trans_files: t.List[pathlib.Path],
        layer_idx: int,
        wn_grid: npt.NDArray[np.float64],
        temperature: float,
        species_mass: float,
        cont_box_length: float,
) -> npt.NDArray[np.float64]:
    is_fixed_width = np.all(np.isclose(np.diff(wn_grid), np.abs(wn_grid[1] - wn_grid[0]), atol=0))
    cont_xsec = np.zeros(wn_grid.shape, dtype=np.float64)

    wn_min = wn_grid[0]
    wn_max = wn_grid[-1]

    n_nlte_col = f"n_nlte_L{layer_idx}"

    pl_states_i = continuum_states.select(
        pl.col("id"),
        pl.col("energy").alias("energy_i"),
        pl.col(n_nlte_col).alias("n_nlte_i"),
        pl.col("g").alias("g_i"),
    )
    pl_states_f = continuum_states.select(
        pl.col("id"),
        pl.col("energy").alias("energy_f"),
        pl.col(n_nlte_col).alias("n_nlte_f"),
        pl.col("g").alias("g_f"),
        pl.col("v").alias("v_f"),
    )

    dask_dtypes = {"id_f": "int64", "id_i": "int64", "A_fi": "float64"}
    dask_blocksize = "256MB"

    for trans_file in continuum_trans_files:
        ddf = dd.read_csv(
            trans_file,
            sep=r"\s+",
            engine="python",
            header=None,
            names=["id_f", "id_i", "A_fi"],
            usecols=[0, 1, 2],
            dtype=dask_dtypes,
            blocksize=dask_blocksize,
        )
        delayed_batches = ddf.to_delayed()
        for delayed_batch in delayed_batches:
            trans_batch = pl.from_pandas(delayed_batch.compute())
            trans_batch = trans_batch.join(pl_states_i, left_on="id_i", right_on="id", how="left")
            trans_batch = trans_batch.join(pl_states_f, left_on="id_f", right_on="id", how="left")
            trans_batch = trans_batch.with_columns(
                (pl.col("energy_f") - pl.col("energy_i")).alias("energy_fi")
            )
            trans_batch = trans_batch.filter(
                (pl.col("energy_f") >= wn_min)
                & (pl.col("energy_i") <= wn_max)
                & (pl.col("energy_fi") >= wn_min)
                & (pl.col("energy_fi") <= wn_max)
            )
            final_cols = ["n_nlte_f", "n_nlte_i", "A_fi", "g_f", "g_i", "energy_fi", "v_f"]
            trans_batch = trans_batch.select(final_cols)
            trans_chunk_np = trans_batch.to_numpy()

            # Matches final_cols ordering.
            n_f = np.ascontiguousarray(trans_chunk_np[:, 0])
            n_i = np.ascontiguousarray(trans_chunk_np[:, 1])
            a_fi = np.ascontiguousarray(trans_chunk_np[:, 2])
            g_f = np.ascontiguousarray(trans_chunk_np[:, 3])
            g_i = np.ascontiguousarray(trans_chunk_np[:, 4])
            energy_fi = np.ascontiguousarray(trans_chunk_np[:, 5])
            v_f = np.ascontiguousarray(trans_chunk_np[:, 6])

            if is_fixed_width:
                half_bin_width = abs(wn_grid[1] - wn_grid[0]) / 2.0
                cont_xsec += _continuum_binned_gauss_fixed_width(
                    wn_grid=wn_grid,
                    n_f=n_f,
                    n_i=n_i,
                    a_fi=a_fi,
                    g_f=g_f,
                    g_i=g_i,
                    energy_fi=energy_fi,
                    v_f=v_f,
                    temperature=temperature,
                    species_mass=species_mass,
                    box_length=cont_box_length,
                    half_bin_width=half_bin_width,
                )
            else:
                cont_xsec += _continuum_binned_gauss_variable_width(
                    wn_grid=wn_grid,
                    n_f=n_f,
                    n_i=n_i,
                    a_fi=a_fi,
                    g_f=g_f,
                    g_i=g_i,
                    energy_fi=energy_fi,
                    v_f=v_f,
                    temperature=temperature,
                    species_mass=species_mass,
                    box_length=cont_box_length,
                )
    return cont_xsec


# ------------------------------------- NUMBA XSEC CALCULATIONS -------------------------------------


@numba.njit(cache=True, error_model="numpy", inline="always")
def binary_search_left(arr: npt.NDArray[np.float64], value: float, start: int = 0) -> int:
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
def binary_search_right(arr: npt.NDArray[np.float64], value: float, start: int = 0) -> int:
    """Fast binary search for right insertion point with optional start hint."""
    left, right = start, len(arr)
    while left < right:
        mid = (left + right) >> 1
        if arr[mid] <= value:
            left = mid + 1
        else:
            right = mid
    return left


# ------------------------------------- ALL LAYER CALCULATION -------------------------------------

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

    Computes absorption, stimulated emission, and spontaneous emission profiles
    for all layers simultaneously. The bin_term inner loop — the dominant cost —
    is computed once per transition per grid point and reused across all layers,
    with only the per-layer coefficients (which depend on populations,
    temperature, and pressure) applied on top.

    Parameters
    ----------
    wn_grid : (n_grid,)
    n_i : (n_layers, n_trans)   LTE population fractions of lower states, per layer.
    n_f : (n_layers, n_trans)   LTE population fractions of upper states, per layer.
    a_fi : (n_trans,)
    g_f : (n_trans,)
    g_i : (n_trans,)
    energy_fi : (n_trans,)
    lifetimes : (n_trans,)
    temperatures : (n_layers,)
    pressures : (n_layers,)
    broad_n : (n_broadeners,)
    broad_gamma : (n_broadeners, n_layers)  Note: layers on axis-1, matching broadening_params[0].
    species_mass : float
    gh_roots : (n_gh,)
    gh_weights : (n_gh,)

    Returns
    -------
    abs_xsec : (n_layers, n_grid)
    ste_xsec : (n_layers, n_grid)
    spe_xsec : (n_layers, n_grid)
    """
    n_layers = temperatures.shape[0]
    num_grid = wn_grid.shape[0]
    num_trans = energy_fi.shape[0]
    num_broad = broad_n.shape[0]
    cutoff = 25

    # Bin geometry — identical to the 1-D version, computed once.
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

        abs_coef[l] = a_fi * (n_i[l] * g_f / g_i) / (const_8_pi_five_halves_c * energy_fi * energy_fi)
        ste_coef[l] = a_fi * n_f[l] / (const_8_pi_five_halves_c * energy_fi * energy_fi)
        spe_coef[l] = n_f[l] * a_fi * energy_fi * const_h_c_on_4_pi_five_halves

    gamma_inv = 1.0 / gamma_total  # (n_layers, n_trans)

    # Output buffers: (numba_num_threads, n_layers, n_grid).
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

        # Grid range is the same for all layers — compute once.
        j_start = binary_search_right(bin_edges, transition_min) - 1
        j_start = max(0, j_start)
        j_end = binary_search_left(bin_edges, transition_max, start=j_start)
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
    wn_grid : (n_grid,)
    n_i : (n_layers, n_trans)
    a_fi : (n_trans,)
    g_f : (n_trans,)
    g_i : (n_trans,)
    energy_fi : (n_trans,)
    v_f : (n_trans,)
    temperatures : (n_layers,)
    species_mass : float
    box_length : float

    Returns
    -------
    abs_xsec : (n_layers, n_grid)
    """
    n_layers = temperatures.shape[0]
    num_grid = wn_grid.shape[0]
    num_trans = energy_fi.shape[0]

    sqrtln2 = math.sqrt(math.log(2))

    min_cutoff = 25.0
    max_cutoff = 3000.0
    cutoff_fwhm_multiple = 5.0

    # Bin geometry — identical to the 1-D version, computed once.
    bin_edges = np.empty(num_grid + 1, dtype=np.float64)
    bin_edges[0] = wn_grid[0] - (wn_grid[1] - wn_grid[0]) * 0.5
    for j in range(1, num_grid):
        bin_edges[j] = (wn_grid[j - 1] + wn_grid[j]) * 0.5
    bin_edges[-1] = wn_grid[-1] + (wn_grid[-1] - wn_grid[-2]) * 0.5

    bin_widths_lower = wn_grid - bin_edges[:-1]
    bin_widths_upper = bin_edges[1:] - wn_grid
    inv_bin_widths = 1.0 / (bin_widths_lower + bin_widths_upper)

    # Per-layer, per-transition profile coefficients (n_f/n_i/T vary per layer).
    alpha_doppler = np.empty((n_layers, num_trans), dtype=np.float64)
    abs_coef = np.empty((n_layers, num_trans), dtype=np.float64)

    for l in range(n_layers):
        temp_l = temperatures[l]
        alpha_doppler[l] = energy_fi * const_sqrt_2_NA_kB_log2_on_c * np.sqrt(temp_l / species_mass)
        abs_coef[l] = a_fi * (n_i[l] * g_f / g_i) / (const_8_pi_five_halves_c * energy_fi * energy_fi)

    alpha_box = const_h_on_8_c * (2 * v_f + 1) / (species_mass * const_amu * box_length * box_length)
    alpha_total = alpha_box + alpha_doppler
    sqrtln2_on_alpha = sqrtln2 / alpha_total

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

    # Output buffers: (numba_num_threads, n_layers, n_grid).
    abs_xsec_buffer = np.zeros((numba_num_threads, n_layers, num_grid), dtype=np.float64)

    for i in numba.prange(num_trans):
        thread_id = numba.get_thread_id()

        energy_fi_i = energy_fi[i]
        transition_min = energy_fi_i - cutoff
        transition_max = energy_fi_i + cutoff

        # Grid range is the same for all layers — compute once.
        j_start = binary_search_right(bin_edges, transition_min) - 1
        j_start = max(0, j_start)
        j_end = binary_search_left(bin_edges, transition_max, start=j_start)
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


# ------------------------------------- PER LAYER CALCULATIONS -------------------------------------


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _band_profile_binned_doppler_variable_width(
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

    abs_coef = a_fi * (n_i * g_f / g_i) / (const_16_pi_c * energy_fi * energy_fi)
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
def _band_profile_binned_doppler_fixed_width(
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

    abs_coef = a_fi * (n_i * g_f / g_i) / (const_8_pi_c * twice_bin_width * energy_fi * energy_fi)
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
def _band_profile_binned_voigt_variable_width(
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
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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

    # Stimulated emission removed!
    abs_coef = a_fi * (n_i * g_f / g_i) / (const_8_pi_five_halves_c * energy_fi * energy_fi)
    ste_coef = a_fi * n_f / (const_8_pi_five_halves_c * energy_fi * energy_fi)
    spe_coef = n_f * a_fi * energy_fi * const_h_c_on_4_pi_five_halves
    sigma = energy_fi * const_sqrt_NA_kB_on_c * np.sqrt(temperature / species_mass)
    gamma_pressure = np.sum(broad_gamma * pressure * (t_ref / temperature) ** broad_n / pressure_ref)
    gamma_lifetime = 1 / (const_4_pi_c * lifetimes)
    gamma_total = gamma_lifetime + gamma_pressure
    gamma_inv = 1 / gamma_total

    grid_min = wn_grid[0]
    grid_max = wn_grid[-1]

    abs_xsec_buffer = np.zeros((numba_num_threads, num_grid), dtype=np.float64)
    ste_xsec_buffer = np.zeros((numba_num_threads, num_grid), dtype=np.float64)
    spe_xsec_buffer = np.zeros((numba_num_threads, num_grid), dtype=np.float64)

    for i in numba.prange(num_trans):
        thread_id = numba.get_thread_id()

        energy_fi_i = energy_fi[i]
        abs_coef_i = abs_coef[i]
        ste_coef_i = ste_coef[i]
        spe_coef_i = spe_coef[i]
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

        j_start = binary_search_right(bin_edges, transition_min) - 1
        j_start = max(0, j_start)
        j_end = binary_search_left(bin_edges, transition_max, start=j_start)
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
            ste_xsec_buffer[thread_id, j] += ste_coef_i * bin_term
            spe_xsec_buffer[thread_id, j] += spe_coef_i * bin_term
    # Accumulate buffers.
    _abs_xsec = np.zeros(num_grid, dtype=np.float64)
    _ste_xsec = np.zeros(num_grid, dtype=np.float64)
    _spe_xsec = np.zeros(num_grid, dtype=np.float64)
    for k in numba.prange(num_grid):
        abs_xsec_point = 0.0
        ste_xsec_point = 0.0
        spe_xsec_point = 0.0
        for t in range(numba_num_threads):
            abs_xsec_point += abs_xsec_buffer[t, k]
            ste_xsec_point += ste_xsec_buffer[t, k]
            spe_xsec_point += spe_xsec_buffer[t, k]
        _abs_xsec[k] = abs_xsec_point
        _ste_xsec[k] = ste_xsec_point
        _spe_xsec[k] = spe_xsec_point

    return _abs_xsec, _ste_xsec, _spe_xsec


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _band_profile_binned_voigt_fixed_width(
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
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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

    abs_coef = a_fi * (n_i * g_f / g_i) / (const_8_pi_five_halves_c * bin_width * energy_fi * energy_fi)
    ste_coef = a_fi * n_f / (const_8_pi_five_halves_c * bin_width * energy_fi * energy_fi)
    spe_coef = n_f * a_fi * energy_fi * const_h_c_on_4_pi_five_halves / bin_width

    sigma = energy_fi * const_sqrt_NA_kB_on_c * np.sqrt(temperature / species_mass)
    gamma_pressure = np.sum(broad_gamma * pressure * (t_ref / temperature) ** broad_n / pressure_ref)
    gamma_lifetime = 1 / (const_4_pi_c * lifetimes)
    gamma_total = gamma_lifetime + gamma_pressure
    gamma_inv = 1 / gamma_total

    grid_min = wn_grid[0]
    grid_max = wn_grid[-1]

    abs_xsec_buffer = np.zeros((numba_num_threads, num_grid), dtype=np.float64)
    ste_xsec_buffer = np.zeros((numba_num_threads, num_grid), dtype=np.float64)
    spe_xsec_buffer = np.zeros((numba_num_threads, num_grid), dtype=np.float64)

    for i in numba.prange(num_trans):
        thread_id = numba.get_thread_id()

        energy_fi_i = energy_fi[i]
        abs_coef_i = abs_coef[i]
        ste_coef_i = ste_coef[i]
        spe_coef_i = spe_coef[i]
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

        j_start = binary_search_right(bin_edges, transition_min) - 1
        j_start = max(0, j_start)
        j_end = binary_search_left(bin_edges, transition_max, start=j_start)
        j_end = min(num_grid, j_end)

        for j in range(j_start, j_end):
            wn_j = wn_grid[j]

            shift_sigma = wn_grid[j] - energy_fi_i - gh_roots_sigma
            bin_term = np.sum(
                gh_weights_b_corr
                * (
                        np.arctan((shift_sigma + half_bin_width) * gamma_inv_i)
                        - np.arctan((shift_sigma - half_bin_width) * gamma_inv_i)
                )
            )
            abs_xsec_buffer[thread_id, j] += abs_coef_i * bin_term
            ste_xsec_buffer[thread_id, j] += ste_coef_i * bin_term
            spe_xsec_buffer[thread_id, j] += spe_coef_i * bin_term
    # Accumulate buffers.
    _abs_xsec = np.zeros(num_grid, dtype=np.float64)
    _ste_xsec = np.zeros(num_grid, dtype=np.float64)
    _spe_xsec = np.zeros(num_grid, dtype=np.float64)
    for k in numba.prange(num_grid):
        abs_xsec_point = 0.0
        ste_xsec_point = 0.0
        spe_xsec_point = 0.0
        for t in range(numba_num_threads):
            abs_xsec_point += abs_xsec_buffer[t, k]
            ste_xsec_point += ste_xsec_buffer[t, k]
            spe_xsec_point += spe_xsec_buffer[t, k]
        _abs_xsec[k] = abs_xsec_point
        _ste_xsec[k] = ste_xsec_point
        _spe_xsec[k] = spe_xsec_point
    return _abs_xsec, _ste_xsec, _spe_xsec


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

        j_start = binary_search_right(bin_edges, transition_min) - 1
        j_start = max(0, j_start)
        j_end = binary_search_left(bin_edges, transition_max, start=j_start)
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

        j_start = binary_search_right(bin_edges, transition_min) - 1
        j_start = max(0, j_start)
        j_end = binary_search_left(bin_edges, transition_max, start=j_start)
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

        j_start = binary_search_right(bin_edges, transition_min) - 1
        j_start = max(0, j_start)
        j_end = binary_search_left(bin_edges, transition_max, start=j_start)
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

        j_start = binary_search_right(bin_edges, transition_min) - 1
        j_start = max(0, j_start)
        j_end = binary_search_left(bin_edges, transition_max, start=j_start)
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
