import logging
import abc
import pathlib
import pickle
import time
import typing as t

import numpy as np
import pandas as pd
import polars as pl
import numba
from numpy import typing as npt

from phoenix4all import get_spectrum

from astropy import units as u, constants as ac

from scipy.integrate import cumulative_simpson
from scipy.optimize import least_squares

from .atomic_nuclear_data import get_reduced_mass
from .chemistry import SpeciesFormula, ChemicalProfile
from .colchem import CollisionalRatesDatabase, RateTransition
from .config import output_dir, _LOG_FLOAT_FMT, _LOG_ARRAY_FMT, _PARQUET_BATCH_SIZE, _LOG_VERBOSE_1, _LOG_VERBOSE_2, \
    _LOG_VERBOSE_3
from .numerics import (loglinear_normalise_1d_nonnegative, loglinear_normalise_quantity_2d_nonnegative,
                       loglinear_normalise_quantity_1d_nonnegative, loglinear_integral_quantity_1d,
                       loglinear_integral_quantity_1d_nonnegative, loglinear_integral_quantity_2d_nonnegative,
                       loglinear_integral_1d, loglinear_integral_1d_nonnegative)
from .profiles import (ProfileStore, ContinuumProfileStore, abs_emi_xsec, continuum_xsec, const_8_pi_c,
                       _accumulate_superline_band_batch, _broaden_superline_band_buffer,
                       calc_einstein_bs, _accumulate_continuum_superline_band_batch,
                       _broaden_continuum_superline_band_buffer, _iter_trans_batches, _band_profile_sampled_voigt,
                       _continuum_band_profile_sampled_gauss_layered)

log = logging.getLogger(__name__)

# Units:
einstein_a_unit = 1 / u.s
einstein_b_unit = (u.m ** 2) / (u.J * u.s)

# Constants with units:

ac_h_c_on_kB = ac.h * ac.c.cgs / ac.k_B
ac_2_hc = 2 * ac.h * ac.c.cgs

ac_2_h_on_c_sq = 2 * ac.h / ac.c ** 2
ac_h_on_kB = ac.h / ac.k_B

# Dimensionless version for numba
const_h_c_on_kB = ac_h_c_on_kB.value
const_2_hc = ac_2_hc.value
const_2_h_on_c_sq = ac_2_h_on_c_sq.value
const_h_on_kB = ac_h_on_kB.value
const_2_pi_h_c_sq_on_sigma_sba = (
    (2 * np.pi * ac.h * ac.c.cgs ** 2 / ac.sigma_sb).to(u.K ** 4 * u.cm ** 4, equivalencies=u.spectral()).value
)
const_2_pi_c_kB = (2 * np.pi * ac.c.cgs * ac.k_B.cgs).value


# TODO: For NANs in state lifetimes; treat as inf? They imply inf but often they exist because of transition energy
#  cutoffs during computation and not because the state has no deexcitation pathways.


# ----------------------------------------- Rates & Profiles Parser Functions -----------------------------------------
def _update_band_registry(
        trans_batch: pl.DataFrame,
        band_registry: pl.DataFrame,
        band_registry_cols: t.List[str],
        profile_buffer: npt.NDArray[np.float64],
        n_bands_used: int,
        n_bands_max: int,
) -> t.Tuple[pl.DataFrame, npt.NDArray[np.float64], int, int, npt.NDArray[np.int64]]:
    """
    Updates the band registry with any new (id_agg_f, id_agg_i) pairs found in trans_batch, grows profile_buffer if
    needed, and returns per-transition band indices.

    Called by :func:`~xsec.NLTEProcessor.compute_rates_profiles`.

    Parameters
    ----------
    trans_batch : pl.DataFrame
    band_registry : pl.DataFrame
    band_registry_cols : List
    profile_buffer : ndarray, shape (x, n_bands, n_layers, n_grid)
        Buffer for writing profiles to. "x" here varies depending on whether bound-bound or bound-free transitions are
        being processed.
    n_bands_used : int
    n_bands_max : int

    Returns
    -------
    band_registry : pl.DataFrame
    profile_buffer : ndarray, shape (x, n_bands, n_layers, n_grid)
    n_bands_used : int
    n_bands_max : int
    band_indices : ndarray, shape (n_trans)

    """
    batch_bands = trans_batch.select(["id_agg_f", "id_agg_i"])
    new_keys = (
        batch_bands.unique() if band_registry.height == 0
        else batch_bands.unique().join(band_registry, on=["id_agg_f", "id_agg_i"], how="anti")
    )
    if new_keys.height > 0:
        new_keys = (
            new_keys
            .with_row_index(name="band_idx", offset=n_bands_used)
            .with_columns(pl.col("band_idx").cast(pl.Int32))
            .select(band_registry_cols)
        )
        # Update global counter.
        n_bands_used += new_keys.height
        if n_bands_used > n_bands_max:
            needed_extra = max(n_bands_used - n_bands_max, n_bands_max)
            extra = np.zeros(
                (profile_buffer.shape[0], needed_extra, profile_buffer.shape[2], profile_buffer.shape[3]),
                dtype=np.float64
            )
            profile_buffer = np.concatenate([profile_buffer, extra], axis=1)
            n_bands_max += needed_extra
        band_registry = pl.concat([band_registry, new_keys])

    band_indices = np.ascontiguousarray(
        batch_bands.join(band_registry, on=["id_agg_f", "id_agg_i"], how="left")
        ["band_idx"].to_numpy().copy()
    )
    return band_registry, profile_buffer, n_bands_used, n_bands_max, band_indices


def _compute_agg_rates(
        trans_batch: pl.DataFrame,
        g_lookup: npt.NDArray[np.float64],
        inv_g_lookup: npt.NDArray[np.float64],
) -> t.Optional[pl.DataFrame]:
    """
    Computes the radiative (Einstein) rate coefficients for the batch of transitions.

    Called by :func:`~xsec.NLTEProcessor.compute_rates_profiles`.

    Parameters
    ----------
    trans_batch : pl.DataFrame
    g_lookup : ndarray, shape (n_states + 1, )
    inv_g_lookup : ndarray, shape (n_states + 1, )

    Returns
    -------
    trans_batch_rates : pl.DataFrame

    """
    trans_batch_rates = trans_batch.filter(pl.col("id_agg_f") != pl.col("id_agg_i"))
    if trans_batch_rates.height == 0:
        return None
    b_fi, b_if = calc_einstein_bs(
        id_i=np.ascontiguousarray(trans_batch_rates["id_i"].to_numpy()),
        id_f=np.ascontiguousarray(trans_batch_rates["id_f"].to_numpy()),
        a_fi=np.ascontiguousarray(trans_batch_rates["A_fi"].to_numpy()),
        energy_fi=np.ascontiguousarray(
            (trans_batch_rates["energy_fi"].to_numpy() << 1 / u.cm)
            .to(u.Hz, equivalencies=u.spectral()).value
        ),
        g_lookup=g_lookup,
        inv_g_lookup=inv_g_lookup,
    )
    return (
        trans_batch_rates
        .with_columns(
            [pl.Series("B_fi", b_fi), pl.Series("B_if", b_if)]
        )
        .group_by(["id_agg_f", "id_agg_i"])
        .agg(
            [pl.col("A_fi").sum(), pl.col("B_fi").sum(), pl.col("B_if").sum()]
        )
    )


# -------------------------------------------------- Y matrix kernels --------------------------------------------------

@numba.njit(cache=True, error_model="numpy")
def _build_y_matrix_core(
        id_agg_f: npt.NDArray[np.int32],
        id_agg_i: npt.NDArray[np.int32],
        rates_grid_arr: npt.NDArray[np.float64],
        id_agg_cutoff: int,
        n_lookup: npt.NDArray[np.float64],
        chem_scale_factor: float,
        lambda_layer_grid: npt.NDArray[np.float64],
        global_chi: npt.NDArray[np.float64],
        i_prec: npt.NDArray[np.float64],
        wn_dx: npt.NDArray[np.float64],
        abs_profiles: npt.NDArray[np.float64],
        abs_offsets: npt.NDArray[np.int64],
        abs_start_idxs: npt.NDArray[np.int64],
        abs_profile_idx: npt.NDArray[np.int64],
        ste_profiles: npt.NDArray[np.float64],
        ste_offsets: npt.NDArray[np.int64],
        ste_start_idxs: npt.NDArray[np.int64],
        ste_profile_idx: npt.NDArray[np.int64],
        spe_profiles: npt.NDArray[np.float64],
        spe_offsets: npt.NDArray[np.int64],
        spe_start_idxs: npt.NDArray[np.int64],
        spe_profile_idx: npt.NDArray[np.int64],
        full_prec: bool,
        psi_approx_cross: npt.NDArray[np.float64],
        # Outputs
        y_matrix: npt.NDArray[np.float64],
) -> None:
    """

    Parameters
    ----------
    id_agg_f : ndarray, shape (n_agg,)
    id_agg_i : ndarray, shape (n_agg,)
    rates_grid_arr : ndarray, shape (n_agg, three)
        Each row contains the triple (A_fi, B_fi, B_if).
    id_agg_cutoff : int
    n_lookup : ndarray, shape (n_agg,)
    chem_scale_factor : float
        Dimensionless fractional abundance of the species at the current layer.
    lambda_layer_grid : ndarray, shape (n_grid,)
        Dimensionless Lambda operator at each grid point in the current layer.
    global_chi : ndarray, shape (n_grid,)
        Global opacity (from all sources) in the current layer [cm^2].
    i_prec : ndarray, shape (n_grid,)
        Preconditioned intensity in the current layer [J/(m^2)].
    wn_dx : ndarray, shape (n_grid - 1,)
        Wavenumber grid steps, used for integrals which do not require knowledge of grid point values.
    abs_profiles
    abs_offsets
    abs_start_idxs
    abs_profile_idx
    ste_profiles
    ste_offsets
    ste_start_idxs
    ste_profile_idx
    spe_profiles
    spe_offsets
    spe_start_idxs
    spe_profile_idx
    full_prec : bool
        Determines whether the full preconditioning strategy should be used, or whether rates are preconditioned only
        within their single transition/have no other profile overlap.
    psi_approx_cross : ndarray, shape (n_agg, n_grid,)
        Cross coupling terms for each upper state [cm^-1.s^-1]. Empty when full_prec is False.
    y_matrix : ndarray, shape (n_agg, n_agg)
        Output Y matrix that is modified inplace.

    """
    n_trans = rates_grid_arr.shape[0]
    n_grid = lambda_layer_grid.shape[0]
    n_cross = psi_approx_cross.shape[0]

    for t in range(n_trans):
        id_agg_f_t = id_agg_f[t]
        id_agg_i_t = id_agg_i[t]
        if id_agg_f_t > id_agg_cutoff or id_agg_i_t > id_agg_cutoff:
            continue

        a_fi = rates_grid_arr[t, 0]  # [s^-1]
        b_fi = rates_grid_arr[t, 1]  # [m^2/(J.s)]
        b_if = rates_grid_arr[t, 2]  # [m^2/(J.s)]

        # --- Slice out this transition's profile windows from the flat buffers. ---
        abs_pidx = abs_profile_idx[t]
        abs_offset_start_t = abs_offsets[abs_pidx]
        abs_offset_end_t = abs_offsets[abs_pidx + 1]
        abs_grid_start_t = abs_start_idxs[abs_pidx]
        n_abs = abs_offset_end_t - abs_offset_start_t
        abs_grid_end_t = abs_grid_start_t + n_abs

        ste_pidx = ste_profile_idx[t]
        ste_offset_start_t = ste_offsets[ste_pidx]
        ste_offset_end_t = ste_offsets[ste_pidx + 1]
        ste_grid_start_t = ste_start_idxs[ste_pidx]
        n_ste = ste_offset_end_t - ste_offset_start_t
        ste_grid_end_t = ste_grid_start_t + n_ste

        # Profile units [cm^2]
        abs_profile = abs_profiles[abs_offset_start_t:abs_offset_end_t]
        ste_profile = ste_profiles[ste_offset_start_t:ste_offset_end_t]

        # Normalised profiles [cm]
        abs_profile_norm = loglinear_normalise_1d_nonnegative(
            y_data=abs_profile, dx=wn_dx[abs_grid_start_t:abs_grid_end_t - 1]
        )
        ste_profile_norm = loglinear_normalise_1d_nonnegative(
            y_data=ste_profile, dx=wn_dx[ste_grid_start_t:ste_grid_end_t - 1]
        )

        # U_fi [s^-1]
        u_fi = float(a_fi)

        n_f = n_lookup[id_agg_f_t]
        n_i = n_lookup[id_agg_i_t]

        # Band opacity [cm^2]
        chi_if = np.zeros(n_grid, dtype=np.float64)
        for i in range(n_abs):
            chi_if[abs_grid_start_t + i] += n_i * abs_profile[i]
        for i in range(n_ste):
            chi_if[ste_grid_start_t + i] -= n_f * ste_profile[i]
        chi_if *= chem_scale_factor
        for i in range(n_grid):
            if chi_if[i] < 0.0:
                chi_if[i] = 0.0

        if full_prec:
            # psi_approx_cross_if = |chi_if[None, :] * psi_approx_cross|.
            # Overwrite array for integrating each row loop.
            psi_approx_cross_chi = np.empty(n_grid, dtype=np.float64)
            for row in range(n_cross):
                for i in range(n_grid):
                    # Val is units [cm.s^-1]
                    val = chi_if[i] * psi_approx_cross[row, i]
                    # psi_approx_cross[row, i] = val if val >= 0.0 else -val
                    psi_approx_cross_chi[i] = val if val >= 0.0 else -val
                # Integrated Psi is a rate [s^-1]
                psi_integral = loglinear_integral_1d_nonnegative(y_data=psi_approx_cross_chi, dx=wn_dx)
                if psi_integral != 0:
                    y_matrix[id_agg_i_t, row] -= psi_integral
                    y_matrix[id_agg_f_t, row] += psi_integral
        else:
            spe_pidx = spe_profile_idx[t]
            spe_offset_start_t = spe_offsets[spe_pidx]
            spe_offset_end_t = spe_offsets[spe_pidx + 1]
            spe_grid_start_t = spe_start_idxs[spe_pidx]
            n_spe = spe_offset_end_t - spe_offset_start_t
            spe_grid_end_t = spe_grid_start_t + n_spe

            # Emission is stored in [erg.cm/(s.sr)]
            spe_profile = spe_profiles[spe_offset_start_t:spe_offset_end_t]
            spe_profile_norm = loglinear_normalise_1d_nonnegative(
                y_data=spe_profile, dx=wn_dx[spe_grid_start_t:spe_grid_end_t - 1]
            )

            # self_prec_full = np.zeros(n_grid)
            # for i in range(n_grid):
            #     if chi_mask[i]:
            #         self_prec_full[i] = lambda_layer_grid[i] * chi_if[i] / global_chi[i]

            self_prec_windowed = np.empty(n_spe)
            for i in range(n_spe):
                # Windowed retains units of spe_profile_norm [cm]
                if global_chi[spe_grid_start_t + i] == 0:
                    self_prec_windowed[i] = 0
                else:
                    self_prec_windowed[i] = (
                            lambda_layer_grid[spe_grid_start_t + i]
                            * chi_if[spe_grid_start_t + i]
                            * spe_profile_norm[i]
                            / global_chi[spe_grid_start_t + i]
                    )
            # The self prec term is hence dimensionless.
            self_prec = loglinear_integral_1d(y_data=self_prec_windowed, dx=wn_dx[spe_grid_start_t:spe_grid_end_t - 1])
            u_fi *= (1.0 - self_prec)

        # Integrands are [cm.J/m^2]
        ste_integrand = np.empty(n_ste)
        for i in range(n_ste):
            ste_integrand[i] = ste_profile_norm[i] * i_prec[ste_grid_start_t + i]
        # Integral is [J/m^2], product is hence [s^-1].
        v_fi_prec = loglinear_integral_1d(y_data=ste_integrand, dx=wn_dx[ste_grid_start_t:ste_grid_end_t - 1]) * b_fi

        abs_integrand = np.empty(n_abs)
        for i in range(n_abs):
            abs_integrand[i] = abs_profile_norm[i] * i_prec[abs_grid_start_t + i]
        v_if_prec = loglinear_integral_1d(y_data=abs_integrand, dx=wn_dx[abs_grid_start_t:abs_grid_end_t - 1]) * b_if

        # Update Y matrix inplace.
        y_matrix[id_agg_f_t, id_agg_i_t] += v_if_prec
        y_matrix[id_agg_i_t, id_agg_f_t] += u_fi + v_fi_prec
        y_matrix[id_agg_f_t, id_agg_f_t] -= (u_fi + v_fi_prec)
        y_matrix[id_agg_i_t, id_agg_i_t] -= v_if_prec


@numba.njit(cache=True, error_model="numpy")
def _build_y_matrix_cont(
        id_agg_i: npt.NDArray[np.int32],
        rates_grid_arr: npt.NDArray[np.float64],
        id_agg_cutoff: int,
        n_lookup: npt.NDArray[np.float64],
        chem_scale_factor: float,
        lambda_layer_grid: npt.NDArray[np.float64],
        i_prec: npt.NDArray[np.float64],
        wn_dx: npt.NDArray[np.float64],
        abs_profiles: npt.NDArray[np.float64],
        abs_offsets: npt.NDArray[np.int64],
        abs_start_idxs: npt.NDArray[np.int64],
        abs_profile_idx: npt.NDArray[np.int64],
        limiting_species_num_dens: float,
        full_prec: bool,
        psi_approx_cross: npt.NDArray[np.float64],
        # Outputs
        y_matrix: npt.NDArray[np.float64],
) -> None:
    """

    Parameters
    ----------
    id_agg_i : ndarray, shape (n_agg,)
    rates_grid_arr : ndarray, shape (n_agg, three)
        Each row contains the triple (A_fi, B_fi, B_if).
    id_agg_cutoff : int
    n_lookup : ndarray, shape (n_agg,)
    chem_scale_factor : float
        Dimensionless fractional abundance of the species at the current layer.
    lambda_layer_grid : ndarray, shape (n_grid,)
        Dimensionless Lambda operator at each grid point in the current layer.
    i_prec : ndarray, shape (n_grid,)
        Preconditioned intensity in the current layer [J/(m^2)].
    wn_dx : ndarray, shape (n_grid - 1,)
        Wavenumber grid steps, used for integrals which do not require knowledge of grid point values.
    abs_profiles
    abs_offsets
    abs_start_idxs
    abs_profile_idx
    limiting_species_num_dens : float
        Limiting number density of the species involved in photoassociative process to produce the current species.
    full_prec : bool
        Determines whether the full preconditioning strategy should be used, or whether rates are preconditioned only
        within their single transition/have no other profile overlap.
    psi_approx_cross : ndarray, shape (n_agg, n_grid,)
        Cross coupling terms for each upper state [cm^-1.s^-1]. Empty when full_prec is False.
    y_matrix : ndarray, shape (n_agg, n_agg)
        Output Y matrix that is modified inplace.

    """
    n_trans = rates_grid_arr.shape[0]
    n_grid = lambda_layer_grid.shape[0]
    n_cross = psi_approx_cross.shape[0]

    for t in range(n_trans):
        id_agg_i_t = id_agg_i[t]
        if id_agg_i_t > id_agg_cutoff:
            continue

        # a_ci = rates_grid_arr[t, 0]  # [s^-1]
        # b_ci = rates_grid_arr[t, 1]  # [m^2/(J.s)]
        b_ic = rates_grid_arr[t, 2]  # [m^2/(J.s)]

        # --- Slice out this transition's profile windows from the flat buffers. ---
        abs_pidx = abs_profile_idx[t]
        abs_offset_start_t = abs_offsets[abs_pidx]
        abs_offset_end_t = abs_offsets[abs_pidx + 1]
        abs_grid_start_t = abs_start_idxs[abs_pidx]
        n_abs = abs_offset_end_t - abs_offset_start_t
        abs_grid_end_t = abs_grid_start_t + n_abs

        # Profile units [cm^2]
        abs_profile = abs_profiles[abs_offset_start_t:abs_offset_end_t]

        # Normalised profiles [cm]
        abs_profile_norm = loglinear_normalise_1d_nonnegative(
            y_data=abs_profile, dx=wn_dx[abs_grid_start_t:abs_grid_end_t - 1]
        )

        # U_ci [s^-1]
        # u_ci = float(a_ci)

        n_i = n_lookup[id_agg_i_t]
        n_i_scaled = n_i * chem_scale_factor

        # Band opacity [cm^2]
        chi_ic = np.zeros(n_grid, dtype=np.float64)
        for i in range(n_abs):
            chi_ic[abs_grid_start_t + i] += n_i_scaled * abs_profile[i]

        for i in range(n_grid):
            if chi_ic[i] < 0.0:
                chi_ic[i] = 0.0

        if full_prec:
            # psi_approx_cross_if = |chi_if[None, :] * psi_approx_cross|.
            # Overwrite array for integrating each row loop.
            psi_approx_cross_chi = np.empty(n_grid, dtype=np.float64)
            for row in range(n_cross):
                for i in range(n_grid):
                    # Val is units [cm.s^-1]
                    val = chi_ic[i] * psi_approx_cross[row, i]
                    # psi_approx_cross[row, i] = val if val >= 0.0 else -val
                    psi_approx_cross_chi[i] = val if val >= 0.0 else -val
                # Integrated Psi is a rate [s^-1]
                psi_integral = loglinear_integral_1d_nonnegative(y_data=psi_approx_cross_chi, dx=wn_dx)
                if psi_integral != 0:
                    y_matrix[id_agg_i_t, row] -= psi_integral
                    # y_matrix[id_agg_f_t, row] += psi_integral
        # Integrands are [cm.J/m^2]
        abs_integrand = np.empty(n_abs)
        for i in range(n_abs):
            abs_integrand[i] = abs_profile_norm[i] * i_prec[abs_grid_start_t + i]
        v_ic_prec = loglinear_integral_1d(y_data=abs_integrand, dx=wn_dx[abs_grid_start_t:abs_grid_end_t - 1]) * b_ic

        # limiting_scale_factor = 0.0
        # if limiting_species_num_dens != 0:
        #     limiting_scale_factor = chem_scale_factor * n_i / limiting_species_num_dens

        # Update Y matrix inplace.
        # This assumes an 100% dissociation efficiency.
        # Hence, we don't keep track of the continuum state population; id_agg_c_t is always 0.
        # y_matrix[id_agg_c_t, id_agg_i_t] += v_if_prec
        # y_matrix[id_agg_i_t, id_agg_c_t] += u_ci + v_ci_prec
        # y_matrix[id_agg_c_t, id_agg_c_t] -= (u_ci + v_ci_prec)
        y_matrix[id_agg_i_t, id_agg_i_t] -= v_ic_prec
        # However, we can fix the effective "c" population as a scaled function of the limiting, photoassociating
        # species' number density.
        # y_matrix[id_agg_i_t, id_agg_i_t] += u_ci * limiting_scale_factor
        # y_matrix[id_agg_i_t, id_agg_i_t] += v_ci_prec * limiting_scale_factor


# -------------------------------------------- Bezier Coefficients & Setup --------------------------------------------

def effective_source_tau_mu(
        global_source_func_matrix: u.Quantity,
        global_chi_matrix: u.Quantity,
        global_eta_matrix: u.Quantity,
        density_profile: u.Quantity,
        dz_profile: u.Quantity,
        mu_values: npt.NDArray[np.float64],
        negative_absorption_factor: float = 0.1,
) -> t.Tuple[u.Quantity, npt.NDArray[np.float64]]:
    """
    Compute an effective Source function and optical depth, tau, for use in calculation of Bezier interpolants. These
    are computed based on an effective opacity calculated using Eq. (39) from https://doi.org/10.48550/arXiv.2508.12873.

    Parameters
    ----------
    global_source_func_matrix
    global_chi_matrix
    global_eta_matrix
    density_profile
    dz_profile
    mu_values
    negative_absorption_factor : float
        Factor used to calculate the positive upper bound on the effective opacity to use in cases where the opacity is
        negative.

    Returns
    -------
    """
    # Set effective Chi.
    chi_prime: u.Quantity = negative_absorption_factor * np.max(global_chi_matrix, axis=1)

    effective_chi = global_chi_matrix.copy()
    row_mask, col_mask = np.nonzero(global_chi_matrix < 0)
    effective_chi[row_mask, col_mask] = chi_prime[row_mask] * np.exp(
        -np.abs(global_chi_matrix[row_mask, col_mask].value)
    )

    effective_source_func_matrix = global_source_func_matrix.copy()
    neg_mask = effective_source_func_matrix < 0
    # Update negative source functions.
    # Zero entries where effective_chi is 0 - avoid division by 0.
    zero_chi_mask = neg_mask & (effective_chi == 0)
    effective_source_func_matrix[zero_chi_mask] = 0 * global_source_func_matrix.unit
    # Set entries with non-zero effective_chi.
    pos_chi_mask = neg_mask & ~zero_chi_mask
    effective_source_func_matrix[pos_chi_mask] = (
            global_eta_matrix[pos_chi_mask] / (ac.c.cgs * effective_chi[pos_chi_mask])
    ).to(global_source_func_matrix.unit, equivalencies=u.spectral())

    # Compute optical depths using effective chi.
    res = effective_chi * density_profile[:, None] * dz_profile[:, None]
    dtau = res.decompose().value
    tau = dtau[::-1].cumsum(axis=0)[::-1]
    tau_mu = tau[:, None, :] / mu_values[None, :, None]
    return effective_source_func_matrix, tau_mu


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _compute_coefficients_core(
        delta_tau: npt.NDArray[np.float64],
        coefficients: npt.NDArray[np.float64]
) -> None:
    """
    Compute alpha, beta, gamma coefficients and store exp(-delta_tau) in coefficients[:, 0].
    This modifies coefficients in-place.
    """
    n_layers, _, n_angles, n_wavelengths = coefficients.shape
    delta_tau_limit = 1.4e-1

    for k in numba.prange(n_wavelengths):
        for i in range(n_layers):
            for j in range(n_angles):
                dt = delta_tau[i, j, k]

                if dt == 0.0:
                    coefficients[i, 0, j, k] = 1.0
                    # alpha, beta, gamma tend to 0 as dt -> 0.
                    coefficients[i, 1, j, k] = 0.0
                    coefficients[i, 2, j, k] = 0.0
                    coefficients[i, 3, j, k] = 0.0
                elif dt < delta_tau_limit:
                    coefficients[i, 0, j, k] = np.exp(-dt)
                    # Taylor expansion (Horner form)
                    coefficients[i, 1, j, k] = dt * (dt * (dt * (dt * (
                            dt * (dt * ((10 - dt) * dt - 90) + 720) - 5040) + 30240) - 151200) + 604800) / 1814400
                    coefficients[i, 2, j, k] = dt * (dt * (dt * (dt * (dt * (
                            dt * ((140 - 18 * dt) * dt - 945) + 5400) - 25200) + 90720) - 226800) + 302400) / 907200
                    coefficients[i, 3, j, k] = dt * (dt * (dt * (dt * (dt * (
                            dt * ((35 - 4 * dt) * dt - 270) + 1800) - 10080) + 45360) - 151200) + 302400) / 907200
                else:
                    # Exact formula
                    exp_neg_tau = np.exp(-dt)
                    dt_sq = dt * dt

                    coefficients[i, 0, j, k] = exp_neg_tau
                    coefficients[i, 1, j, k] = (2 + dt_sq - 2 * dt - 2 * exp_neg_tau) / dt_sq
                    coefficients[i, 2, j, k] = (2 - (2 + 2 * dt + dt_sq) * exp_neg_tau) / dt_sq
                    coefficients[i, 3, j, k] = (2 * dt - 4 + (2 * dt + 4) * exp_neg_tau) / dt_sq


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _compute_control_points_outward(
        tau_mu_matrix: npt.NDArray[np.float64],
        source_func_mu: npt.NDArray[np.float64],
        control_points: npt.NDArray[np.float64],
        coefficients: npt.NDArray[np.float64],
        start: int = 0,
        end: int | None = None,
) -> None:
    """Compute outward control points (index 0)."""
    n_layers, n_angles, n_wavelengths = tau_mu_matrix.shape
    if end is None or end > n_layers - 1:
        end = n_layers - 1

    # derivative indices required: i in [start-2, end).
    deriv_start = max(start - 2, 1)
    deriv_end = min(end, n_layers - 1)

    for k in numba.prange(n_wavelengths):
        for j in range(n_angles):
            # Compute derivatives
            d_s_d_tau_out = np.zeros(n_layers, dtype=np.float64)

            for i in range(deriv_start, deriv_end):
                tau_diff = tau_mu_matrix[i, j, k] - tau_mu_matrix[i + 1, j, k]
                if tau_diff != 0:
                    d_diff = (source_func_mu[i, j, k] - source_func_mu[i + 1, j, k]) / tau_diff

                    if i > 0:
                        tau_diff_prev = tau_mu_matrix[i - 1, j, k] - tau_mu_matrix[i, j, k]
                        if tau_diff_prev != 0:
                            d_diff_prev = (source_func_mu[i - 1, j, k] - source_func_mu[i, j, k]) / tau_diff_prev
                        else:
                            d_diff_prev = 0

                        zeta_denom = tau_mu_matrix[i - 1, j, k] - tau_mu_matrix[i + 1, j, k]
                        if zeta_denom != 0:
                            zeta = (1 + (tau_mu_matrix[i - 1, j, k] - tau_mu_matrix[i, j, k]) / zeta_denom) / 3
                        else:
                            zeta = 1.0 / 3.0

                        numerator = d_diff * d_diff_prev
                        denominator = zeta * d_diff_prev + (1 - zeta) * d_diff

                        if numerator >= 0 and denominator != 0:
                            d_s_d_tau_out[i] = numerator / denominator

            # Compute control points with clamping
            for i in range(max(start, 1), min(end + 1, n_layers)):
                tau_diff = tau_mu_matrix[i - 1, j, k] - tau_mu_matrix[i, j, k]

                control_0 = source_func_mu[i, j, k] + 0.5 * tau_diff * d_s_d_tau_out[i]
                control_1 = source_func_mu[i - 1, j, k] - 0.5 * tau_diff * d_s_d_tau_out[i - 1]

                min_source = min(source_func_mu[i - 1, j, k], source_func_mu[i, j, k])
                max_source = max(source_func_mu[i - 1, j, k], source_func_mu[i, j, k])

                control_0 = max(min(control_0, max_source), min_source)
                control_1 = max(min(control_1, max_source), min_source)

                if i == 1:
                    control_points[i, 0, j, k] = control_1
                else:
                    control_points[i, 0, j, k] = 0.5 * (control_0 + control_1)

                # Clamp to non-negative if gamma > 0; sign of gamma*C must be +.
                if coefficients[i, 3, j, k] > 0 > control_points[i, 0, j, k]:
                    # TODO: Check gamma index offset.
                    control_points[i, 0, j, k] = 0.0
    # End.


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _compute_control_points_inward(
        tau_mu_matrix: npt.NDArray[np.float64],
        source_func_mu: npt.NDArray[np.float64],
        control_points: npt.NDArray[np.float64],
        coefficients: npt.NDArray[np.float64],
        start: int = 0,
        end: int | None = None,
) -> None:
    """Compute inward control points (index 1)."""
    n_layers, n_angles, n_wavelengths = tau_mu_matrix.shape

    if end is None or end > n_layers - 1:
        end = n_layers - 1

    # derivative indices required: i in [start, end+2).
    deriv_start = start
    deriv_end = min(end + 2, n_layers - 1)

    for k in numba.prange(n_wavelengths):
        for j in range(n_angles):
            # Compute derivatives
            d_s_d_tau_in = np.zeros(n_layers, dtype=np.float64)

            for i in range(deriv_start, deriv_end):
                tau_diff = tau_mu_matrix[i + 1, j, k] - tau_mu_matrix[i, j, k]
                if tau_diff != 0:
                    d_diff = (source_func_mu[i + 1, j, k] - source_func_mu[i, j, k]) / tau_diff

                    if i < n_layers - 2:
                        tau_diff_next = tau_mu_matrix[i + 2, j, k] - tau_mu_matrix[i + 1, j, k]
                        if tau_diff_next != 0:
                            d_diff_next = (source_func_mu[i + 2, j, k] - source_func_mu[i + 1, j, k]) / tau_diff_next
                        else:
                            d_diff_next = 0

                        zeta_denom = tau_mu_matrix[i + 2, j, k] - tau_mu_matrix[i, j, k]
                        if zeta_denom != 0:
                            zeta = (1 + (tau_mu_matrix[i + 2, j, k] - tau_mu_matrix[i + 1, j, k]) / zeta_denom) / 3
                        else:
                            zeta = 1.0 / 3.0

                        numerator = d_diff * d_diff_next
                        denominator = zeta * d_diff_next + (1 - zeta) * d_diff

                        if numerator >= 0 and denominator != 0:
                            d_s_d_tau_in[i + 1] = numerator / denominator

            # Compute control points with clamping
            for i in range(max(start, 0), min(end + 1, n_layers - 1)):
                tau_diff = tau_mu_matrix[i + 1, j, k] - tau_mu_matrix[i, j, k]

                control_0 = source_func_mu[i, j, k] + 0.5 * tau_diff * d_s_d_tau_in[i]
                control_1 = source_func_mu[i + 1, j, k] - 0.5 * tau_diff * d_s_d_tau_in[i + 1]

                min_source = min(source_func_mu[i, j, k], source_func_mu[i + 1, j, k])
                max_source = max(source_func_mu[i, j, k], source_func_mu[i + 1, j, k])

                control_0 = max(min(control_0, max_source), min_source)
                control_1 = max(min(control_1, max_source), min_source)

                if i == n_layers - 2:
                    control_points[i, 1, j, k] = control_1
                else:
                    control_points[i, 1, j, k] = 0.5 * (control_0 + control_1)

                # Clamp to non-negative if gamma > 0; sign of gamma*C must be +.
                if i > 0 > control_points[i, 1, j, k] and coefficients[i, 3, j, k] > 0:
                    # TODO: Check gamma index offset.
                    control_points[i, 1, j, k] = 0
    # End.


# @numba.njit(parallel=True, cache=True, error_model="numpy")
def bezier_coefficients(
        tau_mu_matrix: npt.NDArray[np.float64],
        source_function_matrix: u.Quantity,
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Computes the Bézier coefficients and control points used for interpolation.

    Parameters
    ----------
    tau_mu_matrix : ndarray
        Optical depth matrix [n_layers, n_angles, n_wavelengths].
    source_function_matrix : ndarray
        Source function [n_layers, n_wavelengths].

    Returns
    -------
    tuple of (coefficients, control_points)
        coefficients : ndarray
            [n_layers+1, 4, n_angles, n_wavelengths];
            [:,0,:,:] is :math:`\\exp(-\\Delta\\tau)`, [:,1,:,:] is :math:`\\alpha`, [:,2,:,:] is :math:`\\beta`, [:,3,:,:] is :math:`\\gamma`.
        control_points : ndarray
            [n_layers, 2, n_angles, n_wavelengths]
    """
    n_layers, n_angles, n_wavelengths = tau_mu_matrix.shape

    tau_mu_matrix = np.ascontiguousarray(tau_mu_matrix)

    # Initialize arrays
    coefficients = np.zeros((n_layers + 1, 4, n_angles, n_wavelengths), dtype=np.float64)
    control_points = np.zeros((n_layers, 2, n_angles, n_wavelengths), dtype=np.float64)

    # Compute delta_tau (difference between layers)
    # coefficients[1:, 0, :, :] = tau_mu_matrix
    # coefficients[:-1, 0, :, :] -= tau_mu_matrix
    # # Store delta_tau temporarily for coefficient computation; contiguous copy is faster.
    # delta_tau = np.ascontiguousarray(coefficients[:, 0, :, :])

    # Delta tau was previously stored in coefficients but _compute_coefficients_core no longer accesses these and we
    # do not need to keep them for output.
    delta_tau = np.zeros((n_layers + 1, n_angles, n_wavelengths), dtype=np.float64)
    delta_tau[1:, :, :] = tau_mu_matrix
    delta_tau[:-1, :, :] -= tau_mu_matrix

    # Compute alpha, beta, gamma and overwrite coefficients[:, 0] with exp(-delta_tau)
    _compute_coefficients_core(delta_tau, coefficients)

    # Expand source function to all angles
    # Division through by mu was removed; following the maths through the control points are angle independent.
    # It also braks the clamping checks which ensure monotonicity.
    # source_func_mu = np.empty((n_layers, n_angles, n_wavelengths))
    # for i in range(n_layers):
    #     for j in range(n_angles):
    #         source_func_mu[i, j, :] = source_function_matrix[i, :]
    source_func_mu = np.ascontiguousarray(
        np.broadcast_to(source_function_matrix.value[:, None, :], (n_layers, n_angles, n_wavelengths))
    )

    # Compute control points
    _compute_control_points_outward(
        tau_mu_matrix=tau_mu_matrix,
        source_func_mu=source_func_mu,
        control_points=control_points,
        coefficients=coefficients
    )
    _compute_control_points_inward(
        tau_mu_matrix=tau_mu_matrix,
        source_func_mu=source_func_mu,
        control_points=control_points,
        coefficients=coefficients
    )

    return coefficients, control_points


@numba.njit(parallel=True, cache=True, error_model="numpy")
def update_layer_coefficients(
        layer_idx: int,
        tau_mu_matrix: npt.NDArray[np.float64],
        source_function_matrix: npt.NDArray[np.float64],
        coefficients: npt.NDArray[np.float64],
        control_points: npt.NDArray[np.float64]
) -> None:
    n_layers, n_angles, n_wavelengths = tau_mu_matrix.shape
    delta_tau_limit = 1.4e-1

    update_idxs = []
    if layer_idx > 0:
        update_idxs.append(layer_idx)
    if layer_idx < n_layers - 1:
        update_idxs.append(layer_idx + 1)

    for k in numba.prange(n_wavelengths):
        for j in range(n_angles):
            for i in update_idxs:
                dt = tau_mu_matrix[i - 1, j, k] - tau_mu_matrix[i, j, k]

                if dt == 0.0:
                    coefficients[i, 0, j, k] = 1.0
                    # Alpha, beta, gamma tend to 0 as dt -> 0.
                    coefficients[i, 1, j, k] = 0.0
                    coefficients[i, 2, j, k] = 0.0
                    coefficients[i, 3, j, k] = 0.0
                elif dt < delta_tau_limit:
                    # Taylor expansion (Horner form)
                    coefficients[i, 0, j, k] = np.exp(-dt)
                    coefficients[i, 1, j, k] = dt * (dt * (dt * (dt * (
                            dt * (dt * ((10 - dt) * dt - 90) + 720) - 5040) + 30240) - 151200) + 604800) / 1814400
                    coefficients[i, 2, j, k] = dt * (dt * (dt * (dt * (dt * (
                            dt * ((140 - 18 * dt) * dt - 945) + 5400) - 25200) + 90720) - 226800) + 302400) / 907200
                    coefficients[i, 3, j, k] = dt * (dt * (dt * (dt * (dt * (
                            dt * ((35 - 4 * dt) * dt - 270) + 1800) - 10080) + 45360) - 151200) + 302400) / 907200
                else:
                    dt_sq = dt * dt
                    exp_neg_tau = np.exp(-dt)
                    # Exact formula
                    coefficients[i, 0, j, k] = exp_neg_tau
                    coefficients[i, 1, j, k] = (2 + dt_sq - 2 * dt - 2 * exp_neg_tau) / dt_sq
                    coefficients[i, 2, j, k] = (2 - (2 + 2 * dt + dt_sq) * exp_neg_tau) / dt_sq
                    coefficients[i, 3, j, k] = (2 * dt - 4 + (2 * dt + 4) * exp_neg_tau) / dt_sq

    source_func_mu = np.empty((n_layers, n_angles, n_wavelengths), dtype=np.float64)
    for i in range(n_layers):
        for j in range(n_angles):
            source_func_mu[i, j, :] = source_function_matrix[i, :]

    # Recompute control_points[i, 0] and control_points[i, 1] for i in range [layer_idx-2 ... layer_idx+2]
    min_update_idx = min(update_idxs)
    max_update_idx = max(update_idxs)
    # i_low and i_high are the inclusive indices for updating; looping must account for range bounds.
    i_low = max(min_update_idx - 1, 0)
    i_high = min(max_update_idx + 1, n_layers - 1)

    _compute_control_points_outward(
        tau_mu_matrix=tau_mu_matrix, source_func_mu=source_func_mu, control_points=control_points,
        coefficients=coefficients, start=i_low, end=i_high,
    )
    _compute_control_points_inward(
        tau_mu_matrix=tau_mu_matrix, source_func_mu=source_func_mu, control_points=control_points,
        coefficients=coefficients, start=i_low, end=i_high,
    )
    # Done.


############# END NEW
# DEPRECATED BELOW:
# @numba.njit(parallel=True, cache=True, error_model="numpy")
# def bezier_coefficients_old(
#         tau_mu_matrix: npt.NDArray[np.float64],
#         source_function_matrix: npt.NDArray[np.float64],
# ) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
#     """
#     Computes the Bezier coefficients delta, alpha, beta and gamma and the control points used for interpolation. Inward
#     and outward directed coefficients are stored in the same array, with the first index (corresponding to the
#     atmospheric layer) offset by 1. Control points are stored in the same array with the second index corresponding to
#     the outward and inward components respectively.
#
#     The Horner rule is used to expand the computation of alpha, beta and gamma when delta is small.
#
#
#     See Eqs. (13-20) in https://doi.org/10.48550/arXiv.2508.12873 for full details.
#
#     :param tau_mu_matrix:
#     :param source_function_matrix:
#
#     :return: A tuple containing two arrays: an array containing the Bezier coefficients delta, alpha, beta and gamma and
#         an array containing the control points.
#     """
#     # New.
#     n_layers, n_angles, n_wavelengths = tau_mu_matrix.shape
#     coefficients = np.zeros((n_layers + 1, 4, n_angles, n_wavelengths), dtype=np.float64)
#     control_points = np.zeros((n_layers, 2, n_angles, n_wavelengths), dtype=np.float64)
#     d_s_d_tau_in = np.zeros_like(tau_mu_matrix, dtype=np.float64)
#     d_s_d_tau_out = np.zeros_like(tau_mu_matrix, dtype=np.float64)
#
#     # coefficients[1:-1, 0, :, :] = tau_matrix[:-1]
#     # coefficients[1:-1, 0, :, :] -= tau_matrix[1:]
#     # Below needed to get coefficients at the boundary layers.
#     coefficients[1:, 0, :, :] = tau_mu_matrix
#     coefficients[:-1, 0, :, :] -= tau_mu_matrix
#     # tau_plus is delta_tau_matrix[1:], tau_minus is delta_tau_matrix[:-1]
#
#     delta_tau_limit = 1.4e-1
#     delta_tau_limit_mask = coefficients[:, 0, :, :] < delta_tau_limit
#
#     delta_tau_sq = coefficients[:, 0, :, :] ** 2
#     # delta_tau_cube = coefficients[:, 0, :, :] ** 3
#     exp_neg_tau = np.exp(-coefficients[:, 0, :, :])
#
#     denom_delta_tau_sq = np.where(delta_tau_sq == 0, 1, delta_tau_sq)
#
#     # Change indices on delta_tau_matrix based on direction! - Old comment.
#     coefficients[:, 1, :, :] = np.where(
#         delta_tau_limit_mask,
#         # (coefficients[:, 0, :, :] / 3) - (delta_tau_sq / 12) + (delta_tau_cube / 60),  # Explicit Taylor
#         (
#                 coefficients[:, 0, :, :]
#                 * (
#                         coefficients[:, 0, :, :]
#                         * (
#                                 coefficients[:, 0, :, :]
#                                 * (
#                                         coefficients[:, 0, :, :]
#                                         * (
#                                                 coefficients[:, 0, :, :]
#                                                 * (
#                                                         coefficients[:, 0, :, :]
#                                                         * ((10 - coefficients[:, 0, :, :]) * coefficients[
#                                                     :, 0, :, :] - 90)
#                                                         + 720
#                                                 )
#                                                 - 5040
#                                         )
#                                         + 30240
#                                 )
#                                 - 151200
#                         )
#                         + 604800
#                 )
#         )
#         / 1814400,  # Horner Taylor
#         (2 + delta_tau_sq - 2 * coefficients[:, 0, :, :] - 2 * exp_neg_tau) / denom_delta_tau_sq,
#     )
#     coefficients[:, 2, :, :] = np.where(
#         delta_tau_limit_mask,
#         # (coefficients[:, 0, :, :] / 3) - (delta_tau_sq / 4) + (delta_tau_cube / 10),  # Explicit Taylor
#         (
#                 coefficients[:, 0, :, :]
#                 * (
#                         coefficients[:, 0, :, :]
#                         * (
#                                 coefficients[:, 0, :, :]
#                                 * (
#                                         coefficients[:, 0, :, :]
#                                         * (
#                                                 coefficients[:, 0, :, :]
#                                                 * (
#                                                         coefficients[:, 0, :, :]
#                                                         * ((140 - 18 * coefficients[:, 0, :, :]) * coefficients[
#                                                     :, 0, :, :] - 945)
#                                                         + 5400
#                                                 )
#                                                 - 25200
#                                         )
#                                         + 90720
#                                 )
#                                 - 226800
#                         )
#                         + 302400
#                 )
#         )
#         / 907200,  # Horner Taylor
#         (2 - (2 + 2 * coefficients[:, 0, :, :] + delta_tau_sq) * exp_neg_tau) / denom_delta_tau_sq,
#     )
#     coefficients[:, 3, :, :] = np.where(
#         delta_tau_limit_mask,
#         # (coefficients[:, 0, :, :] / 3) - (delta_tau_sq / 6) + (delta_tau_cube / 20),  # Explicit Taylor
#         (
#                 coefficients[:, 0, :, :]
#                 * (
#                         coefficients[:, 0, :, :]
#                         * (
#                                 coefficients[:, 0, :, :]
#                                 * (
#                                         coefficients[:, 0, :, :]
#                                         * (
#                                                 coefficients[:, 0, :, :]
#                                                 * (
#                                                         coefficients[:, 0, :, :]
#                                                         * ((35 - 4 * coefficients[:, 0, :, :]) * coefficients[
#                                                     :, 0, :, :] - 270)
#                                                         + 1800
#                                                 )
#                                                 - 10080
#                                         )
#                                         + 45360
#                                 )
#                                 - 151200
#                         )
#                         + 302400
#                 )
#         )
#         / 907200,  # Horner Taylor
#         (2 * coefficients[:, 0, :, :] - 4 + (2 * coefficients[:, 0, :, :] + 4) * exp_neg_tau) / denom_delta_tau_sq,
#     )
#     # NEW:
#     coefficients[:, 0, :, :] = exp_neg_tau
#
#     # source_func_mu = source_function_matrix.reshape(n_layers, 1, n_wavelengths) / mu_values.reshape(1, n_angles, 1)
#     # NEW! Note: Control points are mu independent, so dividing by mu breaks the clamping checks. Still need to reshape
#     # for division through by tau/mu.
#     source_func_mu = source_function_matrix.reshape(n_layers, 1, n_wavelengths) * np.ones((1, n_angles, 1))
#     min_source_mu = np.fmin(source_func_mu[:-1], source_func_mu[1:])
#     max_source_mu = np.fmax(source_func_mu[:-1], source_func_mu[1:])
#
#     # if np.any(min_source_mu < 0):
#     #     print(f"WARN: Min source below 0!")
#     # if np.any(max_source_mu < 0):
#     #     print(f"WARN: Max source below 0!")
#
#     tau_matrix_out_1_diff = tau_mu_matrix[:-1] - tau_mu_matrix[1:]
#     d_diff_out = np.where(
#         tau_matrix_out_1_diff == 0,
#         0,
#         (source_func_mu[:-1] - source_func_mu[1:]) / tau_matrix_out_1_diff,
#     )
#     zeta_out_denominator = tau_mu_matrix[:-2] - tau_mu_matrix[2:]
#     zeta_out = np.where(
#         zeta_out_denominator == 0,
#         1 / 3,
#         (1 + (tau_mu_matrix[:-2] - tau_mu_matrix[1:-1]) / zeta_out_denominator) / 3,
#     )
#     d_s_d_tau_out_numerator = d_diff_out[1:] * d_diff_out[:-1]
#     d_s_d_tau_out_denominator = (zeta_out * d_diff_out[:-1]) + ((1 - zeta_out) * d_diff_out[1:])
#     d_s_d_tau_out[1:-1] = np.where(
#         (d_s_d_tau_out_numerator < 0) | (d_s_d_tau_out_denominator == 0),
#         0,
#         d_s_d_tau_out_numerator / d_s_d_tau_out_denominator,
#     )
#
#     control_0_out = source_func_mu[1:] + 0.5 * tau_matrix_out_1_diff * d_s_d_tau_out[1:]
#     control_1_out = source_func_mu[:-1] - 0.5 * tau_matrix_out_1_diff * d_s_d_tau_out[:-1]
#
#     control_0_out = np.fmax(control_0_out, min_source_mu)
#     control_0_out = np.fmin(control_0_out, max_source_mu)
#     control_1_out = np.fmax(control_1_out, min_source_mu)
#     control_1_out = np.fmin(control_1_out, max_source_mu)
#
#     control_points[2:, 0, :, :] = 0.5 * (control_0_out[1:] + control_1_out[1:])
#     control_points[1, 0, :, :] = control_1_out[0]
#
#     # control_points[1:, 0, :, :] = np.fmax(control_points[1:, 0, :, :], 0)
#     control_points[1:, 0, :, :] = np.where(
#         (coefficients[1:-1, 3, :, :] > 0) & (control_points[1:, 0, :, :] < 0),
#         0,
#         control_points[1:, 0, :, :],
#     )
#
#     tau_matrix_in_1_diff = tau_mu_matrix[1:] - tau_mu_matrix[:-1]
#     d_diff_in = np.where(
#         tau_matrix_in_1_diff == 0,
#         0,
#         (source_func_mu[1:] - source_func_mu[:-1]) / tau_matrix_in_1_diff,
#     )
#     zeta_in_denominator = tau_mu_matrix[2:] - tau_mu_matrix[:-2]
#     zeta_in = np.where(
#         zeta_in_denominator == 0,
#         1 / 3,
#         (1 + (tau_mu_matrix[2:] - tau_mu_matrix[1:-1]) / zeta_in_denominator) / 3,
#     )
#     d_s_d_tau_in_numerator = d_diff_in[:-1] * d_diff_in[1:]
#     d_s_d_tau_in_denominator = (zeta_in * d_diff_in[1:]) + ((1 - zeta_in) * d_diff_in[:-1])
#     d_s_d_tau_in[1:-1] = np.where(
#         (d_s_d_tau_in_numerator < 0) | (d_s_d_tau_in_denominator == 0),
#         0,
#         d_s_d_tau_in_numerator / d_s_d_tau_in_denominator,
#     )
#
#     control_0_in = source_func_mu[:-1] + 0.5 * tau_matrix_in_1_diff * d_s_d_tau_in[:-1]
#     control_1_in = source_func_mu[1:] - 0.5 * tau_matrix_in_1_diff * d_s_d_tau_in[1:]
#
#     control_0_in = np.fmax(control_0_in, min_source_mu)
#     control_0_in = np.fmin(control_0_in, max_source_mu)
#     control_1_in = np.fmax(control_1_in, min_source_mu)
#     control_1_in = np.fmin(control_1_in, max_source_mu)
#
#     control_points[:-2, 1, :, :] = 0.5 * (control_0_in[:-1] + control_1_in[:-1])
#     control_points[-2, 1, :, :] = control_1_in[-1]
#
#     # control_points[:-1, 1, :, :] = np.fmax(control_points[:-1, 1, :, :], 0)
#     control_points[:-1, 1, :, :] = np.where(
#         (coefficients[1:-1, 3, :, :] > 0) & (control_points[:-1, 1, :, :] < 0),
#         0,
#         control_points[:-1, 1, :, :],
#     )
#
#     return coefficients, control_points


def blackbody(spectral_grid: u.Quantity, temperature: u.Quantity) -> u.Quantity:
    freq_grid = spectral_grid.to(u.Hz, equivalencies=u.spectral())
    temperature = np.atleast_1d(temperature)[:, None]
    return (ac_2_h_on_c_sq * freq_grid ** 3) / (np.exp(ac_h_on_kB * freq_grid / temperature) - 1) / u.sr


# def incident_stellar_radiation(
#         wn_grid: u.Quantity, star_temperature: u.Quantity, orbital_radius: u.Quantity, planet_radius: u.Quantity
# ) -> u.Quantity:
#     """
#     Assume the angular size of the planet relative to the star and orbital distance is small, allowing to assume that
#     the surface of the planet with incident radiation is approximately a circle.
#
#     :param wn_grid:
#     :param star_temperature:
#     :param orbital_radius:
#     :param planet_radius:
#     :return:
#     """
#     star_bb = blackbody(spectral_grid=wn_grid, temperature=star_temperature)[0]
#     incident_radiation = star_bb * (planet_radius / orbital_radius) ** 2
#     return incident_radiation.to(star_bb.unit, equivalencies=u.spectral())


def incident_srf(
        star_temperature: float, star_logg: float, star_feh: float, wn_grid: u.Quantity, orbital_radius: u.Quantity,
        star_radius: u.Quantity, star_alpha: float = 0.0
) -> u.Quantity:
    """
    Returns the substellar flux at the planet's surface.

    Parameters
    ----------
    star_temperature
    star_logg
    star_feh
    wn_grid
    orbital_radius
    star_radius
    star_alpha

    Returns
    -------
        Substellar flux [W /(m^2 cm^-1)].

    """
    srf_wavelength, srf_flux = get_spectrum(
        teff=star_temperature,
        logg=star_logg,
        feh=star_feh,
        alpha=star_alpha,
        # source="synphot",
        source="svo",
        model_name="bt-settl-cifist",
    )
    # srf_flux has units of erg / (Angstrom s cm^2)
    srf_wn = srf_wavelength.to(1 / u.cm, equivalencies=u.spectral())

    # srf_flux_wn = srf_flux / (srf_wn**2)
    # srf_flux_wn = srf_flux_wn.to(u.J / (u.s * u.m**2 * (1/u.cm)))

    srf_flux_nu = srf_flux.to(
        u.J / (u.s * u.m ** 2 * u.Hz),
        equivalencies=u.spectral_density(srf_wavelength)
    )

    sort_idx = np.argsort(srf_wn)
    srf_wn = srf_wn[sort_idx]
    srf_flux_nu = srf_flux_nu[sort_idx]

    srf_flux_interp = np.interp(wn_grid, srf_wn, srf_flux_nu, left=0, right=0) << srf_flux_nu.unit
    # This is F_nu [W/(Hz*m^2)]
    srf_flux_orbit = srf_flux_interp * (star_radius / orbital_radius) ** 2
    theta = np.arcsin((star_radius / orbital_radius).decompose().value)
    omega_star = 2 * np.pi * (1 - np.cos(theta)) * u.sr
    srf_specific_intensity = (srf_flux_orbit / omega_star).to(u.W / (u.Hz * u.m ** 2 * u.sr))
    return srf_specific_intensity


def boltzmann_population(states: pl.DataFrame, temperature: u.Quantity) -> pl.DataFrame:
    # TODO: This should be updated use implementations as elsewhere.
    if isinstance(states, pd.DataFrame):
        states = pl.from_pandas(states)

    g_np = states["g"].to_numpy()
    energy_np = states["energy"].to_numpy() << 1 / u.cm

    q_lev_np = g_np * np.exp(-ac_h_c_on_kB * energy_np / temperature).value
    n_np = q_lev_np / np.sum(q_lev_np)

    states = states.with_columns(
        pl.Series("q_lev", q_lev_np),
        pl.Series("n", n_np)
    )

    temp_pop_df = pl.DataFrame({
        "id_agg": states["id_agg"],
        "n": n_np,
    })
    states_agg_n = temp_pop_df.group_by("id_agg").agg(
        pl.col("n").sum().alias("n_agg")
    )
    states = states.join(states_agg_n, on="id_agg", how="left")
    # states["q_lev"] = states["g"] * np.exp(-ac_h_c_on_kB * (states["energy"] << 1 / u.cm) / temperature)
    # states["n"] = states["q_lev"] / states["q_lev"].sum()
    # states_agg_n = states.groupby(by=["id_agg"], as_index=False).agg(n_agg=("n", "sum"))
    # states = states.merge(states_agg_n, on=["id_agg"], how="left")
    return states


# def boltzmann_population(states: pd.DataFrame, temperature: u.Quantity) -> pd.DataFrame:
#     states["q_lev"] = states["g"] * np.exp(-ac_h_c_on_kB * (states["energy"] << 1 / u.cm) / temperature)
#     states["n"] = states["q_lev"] / states["q_lev"].sum()
#     states_agg_n = states.groupby(by=["id_agg"], as_index=False).agg(n_agg=("n", "sum"))
#     states = states.merge(states_agg_n, on=["id_agg"], how="left")
#     return states


@numba.njit(cache=True, error_model="numpy")
def calc_ev_grid(wn_grid: npt.NDArray[np.float64], temperature: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return (const_2_pi_h_c_sq_on_sigma_sba * wn_grid ** 3) / (
            temperature ** 4 * (np.exp(const_h_c_on_kB * wn_grid / temperature) - 1)
    )


@numba.njit(cache=True, error_model="numpy", inline="always")
def _sample_indices(
        ev_cdf: npt.NDArray[np.float64],
        temp_wn_grid: npt.NDArray[np.float64],
        num_points: int,
        max_step: float
):
    """
    Finds the indices in the energy density cumulative distribution function (CDF) such that the energy density is uniform.
    This is done by determining a uniform step size through the CDF based on the number of steps and ensuring the
    difference in the cumulative CDF between each point does not exceed this step size. An additional constraint is
    imposed such that the step size on the wavenumber grid also cannot exceed a maximum value. As such the number of
    points requested by the function is the lower bound on theh number of points returned.

    Parameters
    ----------
    ev_cdf: ndarray
        Energy density cumulative distribution function.
    temp_wn_grid: ndarray
        Temporary high-resolution wavenumber grid (with linear spacing).
    num_points: int
        Lower bound on the number of indices to return to sample the CDF.
    max_step: float
        Maximum step size along the wavenumber grid between each successive index.

    Returns
    -------
        Integer indices corresponding to the uniform sampling points in the CDF.

    """
    num_cdf_points = len(ev_cdf)
    sample_idxs = [0]
    current_idx = 0
    step_size = 1.0 / num_points

    while current_idx < num_cdf_points - 1:
        next_step = ev_cdf[current_idx] + step_size
        next_step_idx = np.searchsorted(ev_cdf, next_step) - 1
        next_step_idx = min(max(next_step_idx, current_idx + 1), num_cdf_points - 1)

        current_wn_val = temp_wn_grid[current_idx]
        next_wn_val = temp_wn_grid[next_step_idx]

        if max_step > 0.0 and (next_wn_val - current_wn_val) > max_step:
            seek_wn_val = current_wn_val + max_step
            seek_idx = np.searchsorted(temp_wn_grid, seek_wn_val) - 1
            current_idx = min(seek_idx, num_cdf_points - 1)
        elif next_step_idx >= num_cdf_points - 1:
            current_idx = num_cdf_points - 1
        else:
            current_idx = next_step_idx
        sample_idxs.append(current_idx)
    return np.array(sample_idxs, dtype=np.int64)


def cdf_opacity_sampling(
        wn_start: float,
        wn_end: float,
        temperature_profile: npt.NDArray[np.float64],
        num_points: int,
        max_step: float,
        num_cdf_points: int = 1000000,
) -> u.Quantity:
    temp_wn_grid = np.linspace(wn_start, wn_end, num_cdf_points, dtype=np.float64)
    ev_grid = calc_ev_grid(wn_grid=temp_wn_grid, temperature=np.atleast_1d(temperature_profile)[:, None]).sum(axis=0)
    # ev_norm = ev_grid / simpson(ev_grid, x=temp_wn_grid)
    temp_dx = np.diff(temp_wn_grid)
    # ev_norm = simpson_normalise_1d(y_data=ev_grid, x_data=temp_wn_grid)
    ev_norm = loglinear_normalise_1d_nonnegative(y_data=ev_grid, dx=temp_dx)

    ev_cdf = cumulative_simpson(ev_norm, x=temp_wn_grid, initial=0)

    sample_idxs = _sample_indices(ev_cdf, temp_wn_grid, num_points, max_step)

    return temp_wn_grid[sample_idxs] / u.cm


def formal_solve_general(
        dtau: u.Quantity,
        source_function: u.Quantity,
        mu_values: npt.NDArray[np.float64],
        mu_weights: npt.NDArray[np.float64],
        incident_radiation_field: u.Quantity = None,
        surface_albedo: float = 0
) -> t.Tuple[u.Quantity, u.Quantity]:
    """
    Solve the 1D plane–parallel radiative-transfer equation for a discretized atmosphere using the *formal solution* for
    each direction cosine :math:`\\mu`.

    This routine computes **upward** and **downward** specific intensities at every layer interface, then integrates
    over angle to obtain the hemispheric fluxes.

    ----------------------------------------------------------------------
    RADIATIVE-TRANSFER EQUATION
    ----------------------------------------------------------------------

    For a ray of direction cosine :math:`\\mu`, the monochromatic radiative-transfer equation in optical depth
    :math:`\\tau` is

    .. math::
        \\mu \\frac{\\mathrm{d} I(\\tau,\\mu)}{\\mathrm{d}\\tau} = I(\\tau,\\mu) - S(\\tau),

    where :math:`S(\\tau)` is the source function.

    The *formal solution* between two optical-depth points :math:`\\tau_{k}` and :math:`\\tau_{k+1}` is:

    .. math::
        I(\\tau_k,\\mu)
        = I(\\tau_{k+1},\\mu) \\, e^{-\\Delta\\tau/\\lvert\\mu\\rvert}
        + S_{k} \\,\\left(1 - e^{-\\Delta\\tau/\\lvert\\mu\\rvert}\\right),

    where :math:`\\Delta\\tau = \\tau_{k+1} - \\tau_{k}`.

    This expression is used for **downward** (TOA to BOA) rays with :math:`\\mu > 0` and **upward** (BOA to TOA) rays
    with :math:`\\mu < 0`.

    ----------------------------------------------------------------------
    NUMERICAL DISCRETIZATION
    ----------------------------------------------------------------------

    The atmosphere is divided into :math:`n_{\\mathrm{layers}}` layers. For each wavenumber :math:`\\tilde{\\nu}` the
    inputs have shapes:

    * :math:`\\Delta\\tau`: ``(n_layers, n_wn)`` optical-depth increment per layer.
    * ``source_function``: ``(n_layers, n_wn)`` source function at each point.
    * ``mu_values``: ``(n_mu,)`` direction cosines.
    * ``mu_weights``: ``(n_mu,)`` quadrature weights.

    Intensities are stored at the **interfaces**, so the output arrays have dimension ``n_layers + 1``.

    ----------------------------------------------------------------------
    BOUNDARY CONDITIONS
    ----------------------------------------------------------------------

    * At the top of atmosphere (TOA):

      .. math::
         I^{-}_{n_{\\mathrm{layers}}}(\\mu>0) = I_{\\mathrm{incident}} \\text{(if given)}.

    * At the bottom of the atmosphere (BOA):

      If no surface reflection is treated explicitly, the upward intensity is set to the source function of the lowest
      layer:

      .. math::
         I^{+}_{0}(\\mu<0) = S_{0}.

    ----------------------------------------------------------------------
    ANGULAR INTEGRATION
    ----------------------------------------------------------------------

    After computing intensities for each :math:`\\mu`, hemispheric fluxes are computed as:

    .. math::
        F^{\\pm}(\\tilde{\\nu})
        = 2\\pi \\sum_{i=1}^{n_{\\mu}} I^{\\pm}_i(\\tilde{\\nu}) \\, w_{i},

    where :math:`w_{i}` are the angular quadrature weights.

    Parameters
    ----------
    dtau : Quantity, shape (n_layers, n_wn)
        Optical-depth increment :math:`\\Delta\\tau_{j}(\\tilde{\\nu})` for each layer :math:`j` and wavenumber
        :math:`\\tilde{\\nu}`.
    source_function : Quantity, shape (n_layers, n_wn)
        Source function :math:`S_{j}(\\tilde{\\nu})` per layer.
    mu_values : ndarray, shape (n_mu,)
        Direction cosines :math:`\\mu_{i}`.
    mu_weights : ndarray, shape (n_mu,)
        Angular quadrature weights :math:`w_{i}` corresponding to ``mu_values``.
    incident_radiation_field : Quantity, shape (n_mu, n_wn), optional
        Downward incident intensity at TOA, :math:`I^{-}(\\tau_{\\mathrm{top}})`; defaults to zero.
    surface_albedo : float, optional
        Surface albedo :math:`A \\in [0,1]`. If nonzero, reflection modifies the BOA upward intensity. (Current
        implementation uses a simplified placeholder.)

    Returns
    -------
    i_up : Quantity, shape (n_layers + 1, n_wn)
        Hemispherically integrated *upward* flux:

        .. math::
            F^{+}(\\tilde{\\nu}) = 2\\pi \\sum_{i} I^{+}_{i}(\\tilde{\\nu}) w_{i}.

    i_down : Quantity, shape (n_layers + 1, n_wn)
        Hemispherically integrated *downward* flux:

        .. math::
            F^{-}(\\tilde{\\nu}) = 2\\pi \\sum_{i} I^{-}_{i}(\\tilde{\\nu}) w_{i}.
    """
    if surface_albedo < 0 or surface_albedo > 1:
        log.warning(f"Surface albedo {surface_albedo} is outside of [0, 1], clipping.")
        surface_albedo = np.clip(surface_albedo, 0, 1)

    n_layers, n_wavelengths = dtau.shape

    # Compute intensity at interfaces.
    i_up = np.zeros((len(mu_values), n_layers + 1, n_wavelengths)) * source_function.unit
    i_down = np.zeros((len(mu_values), n_layers + 1, n_wavelengths)) * source_function.unit

    # Upper boundary condition at the top (level n_layers) is zero, unless incident radiation field!
    if incident_radiation_field is not None:
        i_down[:, n_layers, :] = incident_radiation_field
    else:
        i_down[:, n_layers, :] = 0.0 * source_function.unit

    # Integrate from TOA (k=n_layers-1) down to BOA (k=0)
    for k in range(n_layers - 1, -1, -1):
        delta_tau_mu = dtau[k, :] / np.abs(mu_values[:, None])
        exp_term = np.exp(-delta_tau_mu)
        source_contribution = source_function[k, :] * (1 - exp_term)

        # Intensity at the top interface of each layer
        i_down[:, k, :] = i_down[:, k + 1, :] * exp_term + source_contribution

    # Include an albedo for terrestrial planets?
    # downward_flux = 2 * np.pi * (i_down[:, 0, :] * mu_values[:, None] * mu_weights[mu_weights > 0, None]).sum(axis=0)
    # Reflected intensity is diffuse (same in all directions)
    # reflected_intensity = surface_albedo * downward_flux / np.pi

    # bb = blackbody(...)
    # thermal_emission = bb(dtau.shape[1] * u.nm) # Wavelength version?
    # thermal_emission = source_function[-1] * surface_emissivity # Placeholder for scaling?
    # surface_emission = thermal_emission + reflected_intensity
    surface_emission = source_function[0, :]  # USE THIS IN PROD!

    # Lower boundary source function (black body) is surface upwards emission.
    i_up[:, 0, :] = surface_emission

    # Integrate from BOA (k=0) to TOA (k=n_layers-1)
    for k in range(n_layers):
        delta_tau_mu = dtau[k, :] / mu_values[:, None]
        exp_term = np.exp(-delta_tau_mu)
        source_contribution = source_function[None, k, :] * (1 - exp_term)

        # Intensity at the top of the layer (level k+1)
        i_up[:, k + 1, :] = i_up[:, k, :] * exp_term + source_contribution

    i_up = 2 * np.pi * u.sr * np.sum(i_up * mu_weights[:, None, None], axis=0)
    i_down = 2 * np.pi * u.sr * np.sum(i_down * mu_weights[:, None, None], axis=0)

    return i_up, i_down


class NLTEWorkflow(abc.ABC):
    # In theory NLTEProcessor can implement a generic NLTEWorkflow field that calls the workflow method. The current 
    # implementation hard-cords the Gauss-Seidel workflow, but it may be useful to be able to switch between that and
    # MALI without having to restructure the whole main workflow.
    @abc.abstractmethod
    def workflow(self) -> t.Any:
        pass


class GaussSeidelWorkflow(NLTEWorkflow):

    def workflow(self):
        # Implement current workflow from compute_opacities_profile().
        pass


class MALIWorkflow(NLTEWorkflow):

    def workflow(self):
        # Implement MALI layer step through, intensity and Lambda calculations.
        pass


class NLTEProcessor:
    """Handles all NLTE-specific functionality."""

    __slots__ = [
        # Public:
        "species", "states_file", "trans_files", "agg_col_nums", "agg_col_names", "species_mass", "reduced_mass",
        "broadening_params", "cont_states_file", "cont_trans_files", "cont_box_length", "cont_broad_col_num",
        "dissociation_products", "do_super_lines", "cont_do_super_lines", "approximate_t_ex", "debug",
        "debug_pop_matrix", "save_rates_profiles", "rates_pickle", "profile_pickle", "cont_rates_pickle",
        "cont_profile_pickle",
        # Private:
        "_states", "_n_agg_states", "_agg_states", "_id_agg_cutoff", "_rates_grid", "_profile_store", "_pop_matrix",
        "_mol_chi_matrix", "_mol_eta_matrix", "_cont_states", "_cont_rates", "_cont_profile_store", "_nlte_pop_frac",
        "_agg_lookup_cache", "_a_ox_vals", "_col_chem_c_matrix", "_col_chem_rhs_c", "_n_layers", "_n_lte_layers",
        "_y_matrix", "_y_reduced_idx_map", "_rhs_matrix",
    ]

    def __init__(
            self,
            species: str | SpeciesFormula,
            states_file: pathlib.Path,
            trans_files: pathlib.Path | t.List[pathlib.Path],
            agg_col_nums: t.List[int],
            broadening_params: t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None = None,
            cont_states_file: pathlib.Path | None = None,
            cont_trans_files: pathlib.Path | t.List[pathlib.Path] | None = None,
            cont_box_length: float | None = None,
            cont_broad_col_num: int | None = None,
            dissociation_products: t.Tuple[str] | None = None,
            do_super_lines: bool = False,
            cont_do_super_lines: bool = True,
            approximate_t_ex: bool = True,
            debug: bool = False,
            debug_pop_matrix: npt.NDArray[np.float64] | None = None,
            save_rates_profiles: bool = False,
            rates_pickle: pathlib.Path | None = None,
            profile_pickle: pathlib.Path | None = None,
            cont_rates_pickle: pathlib.Path | None = None,
            cont_profile_pickle: pathlib.Path | None = None,
    ):
        self.species = SpeciesFormula(species)

        if isinstance(states_file, str):
            states_file = pathlib.Path(states_file)
        self.states_file: pathlib.Path = states_file
        if not self.states_file.is_file():
            raise RuntimeError(f"{self.species} states file not found at {self.states_file}.")

        if isinstance(trans_files, str):
            trans_files = [pathlib.Path(trans_files)]
        elif not isinstance(trans_files, list):
            trans_files = [trans_files]
        self.trans_files: t.List[pathlib.Path] = trans_files
        for trans_file in self.trans_files:
            if not trans_file.is_file():
                raise RuntimeError(f"{self.species} trans file not found at {trans_file}.")

        self.agg_col_nums: t.List[int] = agg_col_nums
        self.agg_col_names: t.List[str] = ["agg" + str(idx + 1) for idx in range(0, len(self.agg_col_nums))]

        # self.species_mass: float = get_molecular_mass(species)
        self.species_mass: float = self.species.mass
        self.reduced_mass: float = get_reduced_mass(species)
        self.broadening_params = broadening_params

        # Required, set during runtime/setup.
        self._n_layers: int | None = None
        self._n_lte_layers: int | None = None
        self._states: pl.DataFrame | None = None
        self._n_agg_states: int | None = None
        self._agg_states: pl.DataFrame | None = None
        self._id_agg_cutoff: int | None = None
        self._rates_grid: pl.DataFrame | None = None
        self._profile_store: ProfileStore | None = None
        self._mol_chi_matrix: u.Quantity | None = None
        self._mol_eta_matrix: u.Quantity | None = None
        self._pop_matrix: npt.NDArray[np.float64] | None = None
        self._y_matrix: npt.NDArray[np.float64] | None = None
        self._y_reduced_idx_map: npt.NDArray[int] | None = None
        self._rhs_matrix: npt.NDArray[np.float64] | None = None

        # Continuum properties.
        if isinstance(cont_states_file, str):
            cont_states_file = pathlib.Path(cont_states_file)
        self.cont_states_file: pathlib.Path = cont_states_file
        if self.cont_states_file and not self.cont_states_file.is_file():
            raise RuntimeError(f"{self.species} continuum states file not found at {self.cont_states_file}.")
        if isinstance(cont_trans_files, str):
            cont_trans_files = [pathlib.Path(cont_trans_files)]
        elif not isinstance(cont_trans_files, list) and cont_trans_files is not None:
            cont_trans_files = [cont_trans_files]
        self.cont_trans_files: t.List[pathlib.Path] = cont_trans_files
        if self.cont_trans_files:
            for cont_trans_file in self.cont_trans_files:
                if not cont_trans_file.is_file():
                    raise RuntimeError(f"{self.species} continuum trans file not found at {cont_trans_file}.")
        self.cont_box_length: float | None = cont_box_length
        self.cont_broad_col_num: int | None = cont_broad_col_num
        self._cont_states: pl.DataFrame | None = None
        self._cont_rates: pl.DataFrame | None = None
        self._cont_profile_store: ContinuumProfileStore | None = None
        check_cont_args = [self.cont_states_file, self.cont_trans_files, self.cont_box_length, self.cont_broad_col_num]
        if not (all(arg is None for arg in check_cont_args) or all(arg is not None for arg in check_cont_args)):
            raise RuntimeError(
                "Continuum states and trans files must both be provided with a box length for broadening and "
                "column index for box broadening n."
            )
        self.dissociation_products: t.Tuple = dissociation_products

        self.do_super_lines = do_super_lines
        self.cont_do_super_lines = cont_do_super_lines
        self.approximate_t_ex = approximate_t_ex
        self.debug: bool = debug
        self.debug_pop_matrix: npt.NDArray[np.float64] | None = debug_pop_matrix
        self.save_rates_profiles = save_rates_profiles
        self._agg_lookup_cache = None
        self._nlte_pop_frac = None
        self._a_ox_vals = None
        self._col_chem_c_matrix = None
        self._col_chem_rhs_c = None

    # Properties and setters for required fields that aren't set at initialisation.
    @property
    def n_layers(self) -> int:
        if self._n_layers is None:
            raise RuntimeError(f"{self.species} NLTEProcessor field 'n_layers' not initialised.")
        return self._n_layers

    @n_layers.setter
    def n_layers(self, value: int) -> None:
        self._n_layers = value

    @property
    def n_lte_layers(self) -> int:
        if self._n_lte_layers is None:
            raise RuntimeError(f"{self.species} NLTEProcessor field 'n_lte_layers' not initialised.")
        return self._n_lte_layers

    @n_lte_layers.setter
    def n_lte_layers(self, value: int) -> None:
        self._n_lte_layers = value

    @property
    def n_agg_states(self) -> int:
        if self._n_agg_states is None:
            raise RuntimeError(f"{self.species} NLTEProcessor field 'n_agg_states' not initialised.")
        return self._n_agg_states

    @n_agg_states.setter
    def n_agg_states(self, value: int) -> None:
        self._n_agg_states = value

    @property
    def id_agg_cutoff(self) -> int:
        if self._id_agg_cutoff is None:
            raise RuntimeError(f"{self.species} NLTEProcessor field 'id_agg_cutoff' not initialised.")
        return self._id_agg_cutoff

    @id_agg_cutoff.setter
    def id_agg_cutoff(self, value: int) -> None:
        self._id_agg_cutoff = value

    @property
    def states(self) -> pl.DataFrame:
        if self._states is None:
            raise RuntimeError(f"{self.species} NLTEProcessor field 'states' not initialised.")
        return self._states

    @states.setter
    def states(self, value: pl.DataFrame) -> None:
        self._states = value

    @property
    def agg_states(self) -> pl.DataFrame:
        if self._agg_states is None:
            raise RuntimeError(f"{self.species} NLTEProcessor field 'agg_states' not initialised.")
        return self._agg_states

    @agg_states.setter
    def agg_states(self, value: pl.DataFrame) -> None:
        self._agg_states = value

    @property
    def rates_grid(self) -> pl.DataFrame:
        if self._rates_grid is None:
            raise RuntimeError(f"{self.species} NLTEProcessor field 'rates_grid' not initialised.")
        return self._rates_grid

    @rates_grid.setter
    def rates_grid(self, value: pl.DataFrame) -> None:
        self._rates_grid = value

    @property
    def profile_store(self) -> ProfileStore:
        if self._profile_store is None:
            raise RuntimeError(f"{self.species} NLTEProcessor field 'profile_store' not initialised.")
        return self._profile_store

    @profile_store.setter
    def profile_store(self, value: ProfileStore) -> None:
        self._profile_store = value

    @property
    def pop_matrix(self) -> npt.NDArray[np.float64]:
        if self._pop_matrix is None:
            raise RuntimeError(f"{self.species} NLTEProcessor field 'pop_matrix' not initialised.")
        return self._pop_matrix

    @pop_matrix.setter
    def pop_matrix(self, value: npt.NDArray[np.float64]) -> None:
        self._pop_matrix = value

    @property
    def mol_chi_matrix(self) -> u.Quantity:
        if self._mol_chi_matrix is None:
            raise RuntimeError(f"{self.species} NLTEProcessor field 'mol_chi_matrix' not initialised.")
        return self._mol_chi_matrix

    @mol_chi_matrix.setter
    def mol_chi_matrix(self, value: u.Quantity) -> None:
        self._mol_chi_matrix = value

    @property
    def mol_eta_matrix(self) -> u.Quantity:
        if self._mol_eta_matrix is None:
            raise RuntimeError(f"{self.species} NLTEProcessor field 'mol_eta_matrix' not initialised.")
        return self._mol_eta_matrix

    @mol_eta_matrix.setter
    def mol_eta_matrix(self, value: u.Quantity) -> None:
        self._mol_eta_matrix = value

    @property
    def y_matrix(self) -> u.Quantity:
        if self._y_matrix is None:
            raise RuntimeError(f"{self.species} NLTEProcessor field 'y_matrix' not initialised.")
        return self._y_matrix

    @y_matrix.setter
    def y_matrix(self, value: u.Quantity) -> None:
        self._y_matrix = value

    @property
    def y_reduced_idx_map(self) -> u.Quantity:
        if self._y_reduced_idx_map is None:
            raise RuntimeError(f"{self.species} NLTEProcessor field 'y_reduced_idx_map' not initialised.")
        return self._y_reduced_idx_map

    @y_reduced_idx_map.setter
    def y_reduced_idx_map(self, value: u.Quantity) -> None:
        self._y_reduced_idx_map = value

    @property
    def rhs_matrix(self) -> u.Quantity:
        if self._rhs_matrix is None:
            raise RuntimeError(f"{self.species} NLTEProcessor field 'rhs_matrix' not initialised.")
        return self._rhs_matrix

    @rhs_matrix.setter
    def rhs_matrix(self, value: u.Quantity) -> None:
        self._rhs_matrix = value

    # Methods.

    def get_latest_pop_grid(self, layer_idx: int = None) -> npt.NDArray[np.float64]:
        if self.pop_matrix is not None:
            if layer_idx is None:
                return self.pop_matrix[-1]
            else:
                return self.pop_matrix[-1, layer_idx]
        else:
            raise RuntimeError(f"No population matrix available for species {self.species}.")

    def _build_agg_state_lookup(self) -> None:
        """
        Pre-compute lookup table for aggregate state IDs and energies. Call once and cached.
        """
        if self._agg_lookup_cache is not None:
            return

        lookup = {}

        if len(self.agg_col_names) == 1:
            # Single aggregation (e.g., vibrational quantum number only)
            for row in self.agg_states.iter_rows(named=True):
                lookup[row['agg1']] = (row['id_agg'], row['energy_agg'])
        else:
            # Multiple aggregations (e.g., electronic state + vibrational)
            for row in self.agg_states.iter_rows(named=True):
                # lookup[(row['agg1'], row['agg2'])] = (row['id_agg'], row['energy_agg'])
                lookup[tuple(row[agg_col_name] for agg_col_name in self.agg_col_names)] = (row['id_agg'],
                                                                                           row['energy_agg'])

        self._agg_lookup_cache = lookup

    def aggregate_states(self, temperature_profile: u.Quantity, energy_cutoff: float = None):
        """
        Sets self.states with a polars DataFrame containing the ID, energy, degeneracy and lifetime columns of the
        states file, the columns on which state aggregation is performed and the corresponding aggregated state ID.

        Parameters
        ----------
        temperature_profile: astropy.units.Quantity
             The temperature of each layer in Kelvin.
        energy_cutoff: float
            Energy cutoff value above which to fit state populations to LTE; set with the maximum value of the
            wavenumber grid.

        Returns
        -------

        """
        if self.agg_col_nums is None:
            # Assuming diatomic by default.
            self.agg_col_nums = [9, 10]

        read_col_indices = [0, 1, 2, 5] + self.agg_col_nums
        read_col_names = ["id", "energy", "g", "tau"] + self.agg_col_names
        read_col_indices, read_col_names = (list(x) for x in zip(*sorted(zip(read_col_indices, read_col_names))))
        fixed_dtypes = {
            "id": "Int32",
            "energy": np.float64,
            "g": np.float64,
            "tau": np.float64,
        }
        self.states = pl.from_pandas(pd.read_csv(
            self.states_file,
            sep=r"\s+",
            names=read_col_names,
            usecols=read_col_indices,
            dtype=fixed_dtypes
        ))
        # Replace any nan or inf lifetimes with 0 to avoid numerical issues.
        self.states = self.states.with_columns(
            pl.when(pl.col("tau").is_finite())
            .then(pl.col("tau"))
            .otherwise(0)
            .alias("tau")
        )
        # Sanitise nulls in agg columns. If any aggregate columns are null then it is not a valid grouping and should
        # all be nulled.
        null_mask = pl.any_horizontal([pl.col(c).is_null() for c in self.agg_col_names])
        null_check_col = "null_check"
        self.states = self.states.with_columns(
            null_mask.alias(null_check_col),
            *[pl.when(null_mask).then(None).otherwise(pl.col(c)).alias(c)
              for c in self.agg_col_names]
        )
        # Drop states above grid cutoff? No; transitions between highly excited states can still lie on grid.
        # if energy_cutoff is not None:
        #     pl_states = self.states.filter(pl.col("energy") <= energy_cutoff)
        group_cols = self.agg_col_names + [null_check_col]
        self.agg_states = self.states.group_by(*group_cols).agg(
            pl.col("energy").min().alias("energy_agg")
        )
        valid_mask = (pl.col(null_check_col).not_()) & (pl.col("energy_agg") <= energy_cutoff)
        self.agg_states = self.agg_states.filter(valid_mask).sort("energy_agg", descending=False).with_columns(
            pl.int_range(0, pl.len(), dtype=pl.Int32).alias("id_agg")
        )
        self.id_agg_cutoff = self.agg_states.height - 1 if not self.agg_states.is_empty() else -1
        # All states above the cutoff or with any null values in agg columns are treated as the same overflow agg state.
        overflow_id = self.id_agg_cutoff + 1

        self.agg_states = self.agg_states.drop(null_check_col)
        self.n_agg_states = self.agg_states.height + 1  # Should this be reduced by 1?

        # self.n_agg_states = len(agg_temp)
        # self.agg_states = self.agg_states.sort([null_check_col, "energy_agg"], descending=False).with_columns(
        #     pl.int_range(0, self.n_agg_states, dtype=pl.Int64).alias("id_agg")
        # )
        log.log(_LOG_VERBOSE_1, f"{self.species} aggregated states =\n {self.agg_states}")
        # self.id_agg_cutoff = self.agg_states.select(
        #     pl.col("id_agg").filter(pl.col("energy_agg") <= energy_cutoff).max()
        # ).item()

        self.states = self.states.join(
            self.agg_states.select(
                ["id_agg"] + self.agg_col_names
            ),
            on=self.agg_col_names,
            how="left",
        )
        # Assign null group id_agg, if it exists, on _states.
        # null_group_data = self.agg_states.filter(pl.col(null_check_col)).select("id_agg")
        # if not null_group_data.is_empty():
        #     null_group_id = null_group_data.item()
        #     self.states = self.states.with_columns(
        #         pl.when(pl.col(null_check_col))
        #         .then(null_group_id)
        #         .otherwise(pl.col("id_agg"))
        #         .alias("id_agg")
        #     )
        self.states = self.states.with_columns(
            pl.col("id_agg").fill_null(overflow_id)
        ).drop(null_check_col)
        # Clean up null check column.
        # self.states = self.states.drop(null_check_col)
        # self.agg_states = self.agg_states.drop(null_check_col)

        # Vectorised compute for LTE populations.
        g_np = self.states["g"].to_numpy()
        energy_np = self.states["energy"].to_numpy() << 1 / u.cm
        id_agg_np = self.states["id_agg"].to_numpy()
        q_all = g_np[None, :] * np.exp(-ac_h_c_on_kB * energy_np[None, :] / temperature_profile[:, None])
        n_all = q_all / q_all.sum(axis=1, keepdims=True)
        n_agg_all = np.zeros((self.n_layers, self.n_agg_states))
        # Transposes sum into each state, for each layer.
        np.add.at(n_agg_all.T, id_agg_np, n_all.T)

        self.pop_matrix = np.zeros((1, self.n_layers, self.n_agg_states))
        self.pop_matrix[0] = n_agg_all

        below_cutoff_mask = id_agg_np <= self.id_agg_cutoff
        self._nlte_pop_frac = n_all[:, below_cutoff_mask].sum(axis=1)

        lte_col_exprs = []
        for layer_idx in range(self.n_layers):
            n_lte_col = f"n_L{layer_idx}"
            n_agg_lte_col = f"n_agg_L{layer_idx}"
            lte_col_exprs.extend([
                pl.Series(n_lte_col, n_all[layer_idx]),  # per-state
                pl.Series(n_agg_lte_col, n_agg_all[layer_idx][id_agg_np]),  # per-state via lookup
            ])
        self.states = self.states.with_columns(lte_col_exprs)

        log_columns = read_col_names + ["id_agg"]
        log.log(_LOG_VERBOSE_1, f"{self.species} States = {self.states.select(log_columns)}")
        self._build_agg_state_lookup()
        # Done.

    def compute_rates_profiles(
            self,
            temperature_profile: u.Quantity,
            pressure_profile: u.Quantity,
            wn_grid: u.Quantity,
    ) -> None:
        """
        Processes each transition file batch once across all NLTE layers
        simultaneously, removing the ProcessPoolExecutor. Parallelism is handled
        entirely within the Numba function via numba.prange over transitions.

        The profile_store.add_layer_batch() call uses the updated signature that accepts
        all-layer results in one call.
        """
        assert temperature_profile.shape[0] == pressure_profile.shape[0] == self.n_layers

        n_nlte_layers = self.n_layers - self.n_lte_layers

        if self.profile_pickle is not None and self.rates_pickle is not None:
            self.profile_store = pickle.load(open(self.profile_pickle, "rb"))
            self.rates_grid = pickle.load(open(self.rates_pickle, "rb"))

            assert type(self.rates_grid) == pl.DataFrame
            assert self.profile_store.n_layers == n_nlte_layers
            assert len(self.profile_store.abs_profiles) == n_nlte_layers
            assert len(self.profile_store.ste_profiles) == n_nlte_layers
            assert len(self.profile_store.spe_profiles) == n_nlte_layers
            # Any additional debugging metrics here?
            return

        rates_list = []
        self.profile_store = ProfileStore(n_layers=n_nlte_layers)

        trans_columns = ["id_f", "id_i", "A_fi"]
        dask_dtypes = {"id_f": "int32", "id_i": "int32", "A_fi": "float64"}

        # Plain float arrays — no astropy units — passed directly to Numba.
        temperature_slice = temperature_profile[self.n_lte_layers:].to_value(u.K)  # (n_nlte_layers,)
        pressure_slice = pressure_profile[self.n_lte_layers:].to_value(u.bar)  # (n_nlte_layers,)

        # n_frac_cols = [f"n_frac_nL{nlte_idx}" for nlte_idx in range(n_nlte_layers)]
        states_frac = self.states.with_columns(
            (
                    pl.col(f"n_L{self.n_lte_layers + nlte_idx}") / pl.col(f"n_agg_L{self.n_lte_layers + nlte_idx}")
            ).alias(f"n_frac_nL{nlte_idx}")
            for nlte_idx in range(n_nlte_layers)
        )
        invariant_cols = ["id", "energy", "id_agg"]
        states_i = (
            states_frac
            .select(invariant_cols)
            .rename({col: f"{col}_i" for col in invariant_cols})
        )
        states_f = (
            states_frac
            .select(invariant_cols)
            .rename({col: f"{col}_f" for col in invariant_cols})
        )

        # Grid parameters and buffers.
        wn_min = wn_grid[0]
        wn_max = wn_grid[-1]
        n_grid = wn_grid.shape[0]
        # Bin edges for finding trans bins.
        wn_arr = wn_grid.value
        bin_edges = np.empty(n_grid + 1, dtype=np.float64)
        bin_edges[0] = wn_arr[0] - (wn_arr[1] - wn_arr[0]) * 0.5
        for j in range(1, n_grid):
            bin_edges[j] = (wn_arr[j - 1] + wn_arr[j]) * 0.5
        bin_edges[-1] = wn_arr[-1] + (wn_arr[-1] - wn_arr[-2]) * 0.5

        # States lookup - much more memory efficient than duplicating it all with polars joins!
        state_ids = states_frac["id"].to_numpy()
        max_state_id = int(state_ids.max())
        # State ID lookup offset by 1 as IDs are 1-indexed!
        n_frac_lookup = np.zeros(
            (max_state_id + 1, n_nlte_layers),
            dtype=np.float64,
        )
        for l in range(n_nlte_layers):
            n_frac_lookup[state_ids, l] = states_frac[f"n_frac_nL{l}"].to_numpy()
        n_frac_lookup = np.ascontiguousarray(n_frac_lookup)

        g_lookup = np.zeros(max_state_id + 1, dtype=np.float64)
        g_lookup[state_ids] = states_frac["g"].to_numpy()
        g_lookup = np.ascontiguousarray(g_lookup)
        inv_g_lookup = np.zeros_like(g_lookup)
        zero_mask = g_lookup == 0.0
        inv_g_lookup[~zero_mask] = 1.0 / g_lookup[~zero_mask]
        inv_g_lookup = np.ascontiguousarray(inv_g_lookup)

        tau_lookup = np.zeros(max_state_id + 1, dtype=np.float64)
        tau_lookup[state_ids] = states_frac["tau"].to_numpy()
        tau_lookup = np.ascontiguousarray(tau_lookup)

        # Expand out broadening parameters.
        broad_n = np.zeros(1, dtype=np.float64)
        broad_gamma = np.zeros((1, n_nlte_layers), dtype=np.float64)
        if self.broadening_params is not None:
            broad_n = self.broadening_params[1]
            broad_gamma = self.broadening_params[0][:, self.n_lte_layers:]

        # Before the batch loop, pre-allocate once.
        n_bands_max = min(100, self.n_agg_states ** 2)

        # profile_buffer[0] = abs, profile_buffer[1] = ste, profile_buffer[2] = spe.
        profile_buffer = np.zeros((3, n_bands_max, n_nlte_layers, n_grid), dtype=np.float64)
        profile_buffer = np.ascontiguousarray(profile_buffer)
        # band_key_to_idx: dict[tuple[int, int], int] = {}
        n_bands_used = 0
        band_registry = pl.DataFrame(
            schema={"id_agg_f": pl.Int32, "id_agg_i": pl.Int32, "band_idx": pl.Int32}
        )
        band_registry_cols = band_registry.columns

        process_time = time.perf_counter()
        # New
        for trans_file in self.trans_files:
            log.info(f"Processing file {trans_file}.")
            for trans_batch in _iter_trans_batches(
                    trans_file=trans_file,
                    trans_columns=trans_columns,
                    states_i=states_i,
                    states_f=states_f,
                    wn_min=wn_min,
                    wn_max=wn_max,
                    parquet_batch_size=_PARQUET_BATCH_SIZE,
                    dask_dtypes=dask_dtypes,
                    do_super_lines=self.do_super_lines,
            ):
                if trans_batch.height == 0:
                    log.log(_LOG_VERBOSE_1, "No valid trans in batch.")
                    continue
                # Assign new band indices to any new bands; extend profile buffer if needed.
                band_registry, profile_buffer, n_bands_used, n_bands_max, band_indices = _update_band_registry(
                    trans_batch=trans_batch,
                    band_registry=band_registry,
                    band_registry_cols=band_registry_cols,
                    profile_buffer=profile_buffer,
                    n_bands_used=n_bands_used,
                    n_bands_max=n_bands_max,
                )
                if self.do_super_lines:
                    _accumulate_superline_band_batch(
                        profile_buffer=profile_buffer,
                        band_indices=band_indices,
                        bin_edges=bin_edges,
                        n_frac_lookup=n_frac_lookup,
                        g_lookup=g_lookup,
                        inv_g_lookup=inv_g_lookup,
                        id_f=np.ascontiguousarray(trans_batch["id_f"].to_numpy()),
                        id_i=np.ascontiguousarray(trans_batch["id_i"].to_numpy()),
                        a_fi=np.ascontiguousarray(trans_batch["A_fi"].to_numpy()),
                        energy_fi=np.ascontiguousarray(trans_batch["energy_fi"].to_numpy()),
                    )
                else:
                    _band_profile_sampled_voigt(
                        profile_buffer=profile_buffer,
                        wn_grid=wn_arr,
                        id_f=np.ascontiguousarray(trans_batch["id_f"].to_numpy()),
                        id_i=np.ascontiguousarray(trans_batch["id_i"].to_numpy()),
                        id_agg_f=np.ascontiguousarray(trans_batch["id_agg_f"].to_numpy()),
                        id_agg_i=np.ascontiguousarray(trans_batch["id_agg_i"].to_numpy()),
                        band_indices=band_indices,
                        n_lookup=n_frac_lookup,
                        g_lookup=g_lookup,
                        inv_g_lookup=inv_g_lookup,
                        tau_lookup=tau_lookup,
                        a_fi=np.ascontiguousarray(trans_batch["A_fi"].to_numpy()),
                        energy_fi=np.ascontiguousarray(trans_batch["energy_fi"].to_numpy()),
                        temperatures=temperature_slice,
                        pressures=pressure_slice,
                        broad_n=broad_n,
                        broad_gamma=broad_gamma,
                        species_mass=self.species_mass,
                    )
                # Do rates for batch - same regardless of line processing strategy.
                agg_batch = _compute_agg_rates(trans_batch=trans_batch, g_lookup=g_lookup, inv_g_lookup=inv_g_lookup)
                if agg_batch is not None:
                    rates_list.append(agg_batch)
        # Contract the buffer based on n_bands_used, drop superfluous rows.
        profile_buffer = profile_buffer[:, :n_bands_used, :, :]
        if self.do_super_lines:
            profile_buffer = _broaden_superline_band_buffer(
                profile_buffer=profile_buffer,
                wn_grid=wn_arr,
                temperatures=temperature_slice,
                pressures=pressure_slice,
                broad_n=broad_n,
                broad_gamma=broad_gamma,
                species_mass=self.species_mass,
            )
        # Finalise profile store from, profile_buffer accumulator.
        # self.profile_store.finalise(save=False, species=self.species)
        # Create band-key lookup.
        band_keys = (
            band_registry.sort("band_idx")
            .select(["id_agg_f", "id_agg_i"])
            .to_numpy()
        )  # (n_bands_used, 2)
        self.profile_store.finalise_from_buffer(
            profile_buffer=profile_buffer,
            band_keys=band_keys,
            save=True,
            species=self.species,
        )
        # Done, finalise list of rates chunks.
        log.log(_LOG_VERBOSE_2, f"{self.species} rates/profiles duration = {time.perf_counter() - process_time:.3f}.")
        # DEBUG!
        # log.debug("Doing debug xsecs.")
        # debug_layer_idx = 66 - self.n_lte_layers
        # debug_abs, debug_emi = self.profile_store.build_abs_emi(
        #     layer_idx=debug_layer_idx,
        #     pop_matrix=self.pop_matrix[-1, debug_layer_idx],
        #     wn_grid=wn_grid,
        # )
        # np.save(
        #     fr"/mnt/c/PhD/programs/TIRAMISU/tests/outputs/h2o_super_abs.npy",
        #     debug_abs
        # )
        # np.save(
        #     fr"/mnt/c/PhD/programs/TIRAMISU/tests/outputs/h2o_super_emi.npy",
        #     debug_emi
        # )
        # log.debug("Debug xsecs done.")
        ################################################################
        self.rates_grid = (
            pl.concat(rates_list)
            .group_by("id_agg_f", "id_agg_i")
            .agg([
                pl.col("A_fi").sum().alias("A_fi"),
                pl.col("B_fi").sum().alias("B_fi"),
                pl.col("B_if").sum().alias("B_if"),
            ])
            .sort(["id_agg_i", "id_agg_f"])
        )
        with pl.Config(tbl_rows=1000):
            log.log(_LOG_VERBOSE_1, f"{self.species} rates = \n{self.rates_grid}")
        # Save bands to disk.
        if self.save_rates_profiles:
            band_file = f"{self.species}_wn{int(wn_min)}-{int(wn_max)}_G{n_grid}_B{n_bands_used}_L{n_nlte_layers}.profilestore.pickle"
            with open((output_dir / band_file).resolve(), "wb") as band_pickle:
                pickle.dump(self.profile_store, band_pickle, protocol=pickle.HIGHEST_PROTOCOL)
            rates_file = f"{self.species}_wn{int(wn_min)}-{int(wn_max)}_G{n_grid}_B{n_bands_used}_L{n_nlte_layers}.ratesgrid.pickle"
            with open((output_dir / rates_file).resolve(), "wb") as rates_pickle:
                pickle.dump(self.profile_store, rates_pickle, protocol=pickle.HIGHEST_PROTOCOL)

    def compute_continuum_rates_profiles(
            self,
            temperature_profile: u.Quantity,
            wn_grid: u.Quantity,
    ):
        """
        Reads in continuum states and trans files to compute aggregated rates and band profiles. Equivalent to
        :func:`~xsec.NLTEProcessor.compute_rates_profiles`.

        Parameters
        ----------
        temperature_profile: astropy.units.Quantity
        wn_grid: astropy.units.Quantity

        Returns
        -------

        """
        assert temperature_profile.shape[0] == self.n_layers
        assert self.cont_broad_col_num is not None
        assert self.cont_states_file is not None
        assert self.cont_box_length is not None

        n_nlte_layers = self.n_layers - self.n_lte_layers

        if self.cont_profile_pickle is not None and self.cont_rates_pickle is not None:
            self._cont_profile_store = pickle.load(open(self.cont_profile_pickle, "rb"))
            self._cont_rates = pickle.load(open(self.cont_rates_pickle, "rb"))

            assert type(self._cont_rates) == pl.DataFrame
            assert self._cont_profile_store.n_layers == n_nlte_layers
            assert len(self._cont_profile_store.abs_profiles) == n_nlte_layers
            return

        log.log(_LOG_VERBOSE_2, f"[I0] {self.species} loading continuum absorption rates and profiles.")

        read_col_map = {num: "v" if num == self.cont_broad_col_num else name for num, name in
                        zip(self.agg_col_nums, self.agg_col_names)}
        if self.cont_broad_col_num not in read_col_map:
            read_col_map[self.cont_broad_col_num] = "v"

        extra_col_indices, extra_col_names = (list(x) for x in zip(*sorted(read_col_map.items())))

        read_col_names = ["id", "energy", "g"] + extra_col_names
        read_col_indices = [0, 1, 2] + extra_col_indices
        fixed_dtypes = {"id": "Int64", "energy": np.float64, "g": np.float64, "v": "Int64"}
        agg_dtypes = {name: "string" for name in extra_col_names if name != "v"}
        read_dtypes = fixed_dtypes | agg_dtypes

        self._cont_states = pl.from_pandas(pd.read_csv(
            self.cont_states_file,
            sep=r"\s+",
            names=read_col_names,
            usecols=read_col_indices,
            dtype=read_dtypes,
        ))

        merge_cols = ["id", "id_agg"]
        self._cont_states = self._cont_states.join(self.states.select(merge_cols), on="id", how="left")
        self._cont_states = self._cont_states.with_columns(pl.col("id_agg").fill_null(-1))

        # self.cont_states = self.cont_states.merge(self.states[merge_cols], on="id", how="left")
        # self.cont_states["id_agg"] = self.cont_states["id_agg"].astype("Int64")
        # # NB: Left join converts ints to float as some may be nan, does not occur for inner join but left needed here to
        # # preserve energy/degeneracy info of upper states with no id_agg map.

        states_frac = self._cont_states.clone()

        for nlte_layer_idx, layer_temp in enumerate(temperature_profile[self.n_lte_layers:]):
            layer_idx = nlte_layer_idx + self.n_lte_layers
            # Precompute boltzmann populations for each layer.
            temp_cont_states = boltzmann_population(self._cont_states.clone(), layer_temp)
            states_frac = states_frac.join(
                temp_cont_states.select([
                    pl.col("id"),
                    pl.col("n").alias(f"n_L{layer_idx}"),
                    pl.col("n_agg").alias(f"n_agg_L{layer_idx}")
                ]),
                on="id",
                how="left"
            )
            states_frac = states_frac.with_columns(
                (pl.col(f"n_L{layer_idx}") / pl.col(f"n_agg_L{layer_idx}")).alias(f"n_frac_nL{nlte_layer_idx}")
            )

        cont_rates_list = []
        self._cont_profile_store = ContinuumProfileStore(n_layers=n_nlte_layers)

        trans_columns = ["id_f", "id_i", "A_fi"]
        dask_dtypes = {"id_f": "int64", "id_i": "int64", "A_fi": "float64"}

        # Plain float arrays — no astropy units — passed directly to Numba.
        temperature_slice = temperature_profile[self.n_lte_layers:].to_value(u.K)  # (n_nlte_layers,)

        # n_frac_cols = [f"n_frac_nL{nlte_idx}" for nlte_idx in range(n_nlte_layers)]
        # invariant_cols_i = ["id", "energy", "g", "id_agg"] + n_frac_cols
        # invariant_cols_f = ["id", "energy", "g", "id_agg", "v"]
        invariant_cols = ["id", "energy", "id_agg"]
        states_i = (
            states_frac
            .select(invariant_cols)
            .rename({col: f"{col}_i" for col in invariant_cols})
        )
        states_f = (
            states_frac
            .select(invariant_cols)
            .rename({col: f"{col}_f" for col in invariant_cols})
        )

        # Grid parameters and buffers.
        wn_arr = wn_grid.value
        wn_min = wn_arr[0]
        wn_max = wn_arr[-1]
        n_grid = wn_grid.shape[0]
        # Bin edges for finding trans bins.
        bin_edges = np.empty(n_grid + 1, dtype=np.float64)
        bin_edges[0] = wn_arr[0] - (wn_arr[1] - wn_arr[0]) * 0.5
        for j in range(1, n_grid):
            bin_edges[j] = (wn_arr[j - 1] + wn_arr[j]) * 0.5
        bin_edges[-1] = wn_arr[-1] + (wn_arr[-1] - wn_arr[-2]) * 0.5

        # States lookup - much more memory efficient than duplicating it all with polars joins!
        state_ids = states_frac["id"].to_numpy()
        max_state_id = int(state_ids.max())
        # State ID lookup offset by 1 as IDs are 1-indexed!
        n_frac_lookup = np.zeros(
            (max_state_id + 1, n_nlte_layers),
            dtype=np.float64,
        )
        for l in range(n_nlte_layers):
            n_frac_lookup[state_ids, l] = states_frac[f"n_frac_nL{l}"].to_numpy()
        n_frac_lookup = np.ascontiguousarray(n_frac_lookup)

        g_lookup = np.zeros(max_state_id + 1, dtype=np.float64)
        g_lookup[state_ids] = states_frac["g"].to_numpy()
        g_lookup = np.ascontiguousarray(g_lookup)
        inv_g_lookup = np.zeros_like(g_lookup)
        zero_mask = g_lookup == 0.0
        inv_g_lookup[~zero_mask] = 1.0 / g_lookup[~zero_mask]
        inv_g_lookup = np.ascontiguousarray(inv_g_lookup)

        v_lookup = np.zeros(max_state_id + 1, dtype=np.float64)
        v_lookup[state_ids] = states_frac["v"].to_numpy()
        v_lookup = np.ascontiguousarray(v_lookup)

        # Before the batch loop, pre-allocate once.
        n_bands_max = min(100, self.n_agg_states ** 2)

        # profile_buffer[0] = abs, profile_buffer[1] = broadening.
        if self.cont_do_super_lines:
            # Use extra axis to accumulate broadening.
            profile_buffer = np.zeros((2, n_bands_max, n_nlte_layers, n_grid), dtype=np.float64)
        else:
            profile_buffer = np.zeros((1, n_bands_max, n_nlte_layers, n_grid), dtype=np.float64)
        profile_buffer = np.ascontiguousarray(profile_buffer)
        n_bands_used = 0
        band_registry = pl.DataFrame(
            schema={"id_agg_f": pl.Int32, "id_agg_i": pl.Int32, "band_idx": pl.Int32}
        )
        band_registry_cols = band_registry.columns

        process_time = time.perf_counter()

        for cont_trans_file in self.cont_trans_files:
            log.info(f"Processing file {cont_trans_file}.")
            for trans_batch in _iter_trans_batches(
                    trans_file=cont_trans_file,
                    trans_columns=trans_columns,
                    states_i=states_i,
                    states_f=states_f,
                    wn_min=wn_min,
                    wn_max=wn_max,
                    parquet_batch_size=_PARQUET_BATCH_SIZE,
                    dask_dtypes=dask_dtypes,
                    do_super_lines=self.cont_do_super_lines,
            ):
                if trans_batch.height == 0:
                    log.log(_LOG_VERBOSE_1, "No valid trans in batch.")
                    continue
                # Assign new band indices to any new bands; extend profile buffer if needed.
                band_registry, profile_buffer, n_bands_used, n_bands_max, band_indices = _update_band_registry(
                    trans_batch=trans_batch,
                    band_registry=band_registry,
                    band_registry_cols=band_registry_cols,
                    profile_buffer=profile_buffer,
                    n_bands_used=n_bands_used,
                    n_bands_max=n_bands_max,
                )
                if self.cont_do_super_lines:
                    _accumulate_continuum_superline_band_batch(
                        profile_buffer=profile_buffer,
                        band_indices=band_indices,
                        bin_edges=bin_edges,
                        n_frac_lookup=n_frac_lookup,
                        g_lookup=g_lookup,
                        inv_g_lookup=inv_g_lookup,
                        v_lookup=v_lookup,
                        id_f=np.ascontiguousarray(trans_batch["id_f"].to_numpy()),
                        id_i=np.ascontiguousarray(trans_batch["id_i"].to_numpy()),
                        a_fi=np.ascontiguousarray(trans_batch["A_fi"].to_numpy()),
                        energy_fi=np.ascontiguousarray(trans_batch["energy_fi"].to_numpy()),
                        reduced_mass=self.reduced_mass,
                        box_length=self.cont_box_length,
                    )
                else:
                    _continuum_band_profile_sampled_gauss_layered(
                        profile_buffer=profile_buffer,
                        wn_grid=wn_arr,
                        id_f=np.ascontiguousarray(trans_batch["id_f"].to_numpy()),
                        id_i=np.ascontiguousarray(trans_batch["id_i"].to_numpy()),
                        id_agg_f=np.ascontiguousarray(trans_batch["id_agg_f"].to_numpy()),
                        id_agg_i=np.ascontiguousarray(trans_batch["id_agg_i"].to_numpy()),
                        band_indices=band_indices,
                        n_lookup=n_frac_lookup,
                        g_lookup=g_lookup,
                        inv_g_lookup=inv_g_lookup,
                        v_lookup=v_lookup,
                        a_fi=np.ascontiguousarray(trans_batch["A_fi"].to_numpy()),
                        energy_fi=np.ascontiguousarray(trans_batch["energy_fi"].to_numpy()),
                        temperatures=temperature_slice,
                        species_mass=self.species_mass,
                        reduced_mass=self.reduced_mass,
                        box_length=self.cont_box_length,
                    )
                # Do rates for batch - same regardless of line processing strategy.
                agg_batch = _compute_agg_rates(trans_batch=trans_batch, g_lookup=g_lookup, inv_g_lookup=inv_g_lookup)
                if agg_batch is not None:
                    cont_rates_list.append(agg_batch)
        # Contract the buffer based on n_bands_used, drop superfluous rows.
        profile_buffer = profile_buffer[:, :n_bands_used, :, :]
        if self.do_super_lines:
            profile_buffer = _broaden_continuum_superline_band_buffer(
                profile_buffer=profile_buffer,
                wn_grid=wn_arr,
                temperatures=temperature_slice,
                species_mass=self.species_mass,
            )
        # Finalise profile store from, profile_buffer accumulator.
        # self._cont_profile_store.finalise()
        # Create band-key lookup.
        band_keys = (
            band_registry.sort("band_idx")
            .select(["id_agg_f", "id_agg_i"])
            .to_numpy()
        )  # (n_bands_used, 2)
        self._cont_profile_store.finalise_from_buffer(
            profile_buffer=profile_buffer,
            band_keys=band_keys,
            save=False,
            species=self.species,
        )
        log.log(_LOG_VERBOSE_2,
                f"{self.species} cont. rates/profiles duration = {time.perf_counter() - process_time:.3f}.")

        if len(cont_rates_list) > 0:
            self._cont_rates = pl.concat(cont_rates_list)
            self._cont_rates = self._cont_rates.group_by("id_agg_i").agg([
                pl.col("A_fi").sum().alias("A_fi"),
                pl.col("B_fi").sum().alias("B_fi"),
                pl.col("B_if").sum().alias("B_if"),
            ])
            with pl.Config(tbl_rows=1000):
                log.log(_LOG_VERBOSE_1, f"{self.species} continuum rates = \n{self._cont_rates}")
        else:
            self._cont_rates = None
            log.log(_LOG_VERBOSE_1, f"{self.species} No continuum rates computed on spectral grid.")
        # Done.

    def _build_a_ox_vals_cache(self) -> None:
        """
        Precompute summed Einstein A coefficients per upper aggregated state below the id_agg_cutoff.

        Iterates over rates_grid once and accumulates A_fi into the corresponding upper state index. Call once after
        rates_grid is finalised inside setup routine.

        Stores
        ------
        self._a_ox_vals : astropy.units.Quantity, shape (id_agg_cutoff + 1,)
            Sum of A_fi values for all transitions from each upper state.
        """
        a_ox_vals = np.zeros(self.id_agg_cutoff + 1, dtype=np.float64)
        for trans in self.rates_grid.iter_rows(named=True):
            o_idx = trans["id_agg_f"]
            if o_idx <= self.id_agg_cutoff:
                a_ox_vals[o_idx] += trans["A_fi"]
        self._a_ox_vals = a_ox_vals << einstein_a_unit

    def _build_col_chem_cache(
            self,
            chem_profile: ChemicalProfile,
            density_profile: u.Quantity,
            temperature_profile: u.Quantity,
    ) -> None:
        """
        Precompute and cache the collisional/chemical rate matrices for all layers.

        Since c_fi and c_if depend only on fixed quantities (rate coefficients, number densities, energy differences,
        and per-layer temperatures), the contribution to y_matrix and rhs_matrix from collisional/chemical rates can be
        fully precomputed once per run.

        The cached arrays are stored as:
            self._col_chem_c_matrix  : np.ndarray, shape (n_layers, n_states, n_states)
            self._col_chem_rhs_c     : np.ndarray, shape (n_layers, n_states)

        Parameters
        ----------
        chem_profile : ChemicalProfile
            Chemical abundance profile.
        density_profile : astropy.units.Quantity
            Total number density profile, shape (n_layers,).
        temperature_profile : astropy.units.Quantity
            Temperature at each layer, shape (n_layers,).
        """
        if self.species.atoms > 3:
            log.warning(f"Col./Chem. rates only implemented for species of 3 or fewer atoms, ({self.species} passed).")
            self._col_chem_c_matrix = None
            self._col_chem_rhs_c = None
            return

        n_layers = len(temperature_profile)
        species_str = str(self.species)
        is_temp_dependent = CollisionalRatesDatabase.is_temperature_dependent(species_str)

        # For temperature-independent species, fetch the rates table once.
        # For temperature-dependent species, fetch per layer inside the loop.
        rates_table_fixed = None
        if not is_temp_dependent:
            rates_table_fixed = CollisionalRatesDatabase.get_rates(species=species_str, layer_temp=None)
            if not rates_table_fixed:
                log.warning(f"No collisional/chemical rates configured for {self.species}.")
                self._col_chem_c_matrix = None
                self._col_chem_rhs_c = None
                return

        n_dim = self.id_agg_cutoff + 1

        c_matrix_cache = np.zeros((n_layers, n_dim, n_dim), dtype=np.float64)
        rhs_c_cache = np.zeros((n_layers, n_dim), dtype=np.float64)

        for layer_idx in range(n_layers):
            layer_temp_val = temperature_profile[layer_idx].value

            rates_table = (
                CollisionalRatesDatabase.get_rates(species=species_str, layer_temp=layer_temp_val)
                if is_temp_dependent
                else rates_table_fixed
            )
            if not rates_table:
                continue

            # Group by collision partner to avoid redundant number density lookups
            rates_by_partner: t.Dict[str, t.List[RateTransition]] = {}
            for rate in rates_table:
                rates_by_partner.setdefault(rate.mol_depend, []).append(rate)

            for partner, partner_rates in rates_by_partner.items():
                if partner not in chem_profile.species:
                    continue

                depend_num_dens = (
                        chem_profile[SpeciesFormula(partner)][layer_idx] * density_profile[layer_idx]
                ).to_value(u.cm ** -3)

                for rate in partner_rates:
                    try:
                        upper_id, upper_energy = self._agg_lookup_cache[rate.upper_key]
                        lower_id, lower_energy = self._agg_lookup_cache[rate.lower_key]
                    except KeyError:
                        continue

                    if upper_id > self.id_agg_cutoff or lower_id > self.id_agg_cutoff:
                        # Short circuit for bands above cutoff; adding to RHS biases to fixed distribution above cutoff.
                        continue

                    c_fi = rate.rate * depend_num_dens
                    energy_diff = (upper_energy - lower_energy) * const_h_c_on_kB
                    c_if = c_fi * np.exp(-energy_diff / layer_temp_val)

                    if lower_id == upper_id:
                        rhs_c_cache[layer_idx, upper_id] -= c_fi
                        if c_fi > 0:
                            # Chemical formation; independent of species population.
                            rhs_c_cache[layer_idx, upper_id] -= c_fi
                        else:
                            # Chemical destruction; depends on species population.
                            c_matrix_cache[layer_idx, upper_id, upper_id] += c_fi
                    else:
                        c_matrix_cache[layer_idx, upper_id, lower_id] += c_if
                        c_matrix_cache[layer_idx, lower_id, upper_id] += c_fi
                        c_matrix_cache[layer_idx, upper_id, upper_id] -= c_fi
                        c_matrix_cache[layer_idx, lower_id, lower_id] -= c_if

        self._col_chem_c_matrix = c_matrix_cache
        self._col_chem_rhs_c = rhs_c_cache

    def setup(
            self,
            chem_profile: ChemicalProfile,
            density_profile: u.Quantity,
            temperature_profile: u.Quantity,
            pressure_profile: u.Quantity,
            wn_grid: u.Quantity,
            initial_chi_matrix: u.Quantity
    ) -> None:
        """Setup NLTE calculations."""
        assert self.n_layers is not None
        assert self.n_lte_layers is not None
        assert self.n_layers == temperature_profile.shape[0] == pressure_profile.shape[0] == density_profile.shape[0]

        if self.dissociation_products is not None and any(
                [mol not in chem_profile.species for mol in self.dissociation_products]
        ):
            log.warning(
                f"Specified dissociation products {self.dissociation_products} not present in"
                f" chemical profile {chem_profile.species}."
            )

        self.aggregate_states(
            temperature_profile=temperature_profile,
            energy_cutoff=wn_grid[-1].value
        )
        self.compute_rates_profiles(
            temperature_profile=temperature_profile,
            pressure_profile=pressure_profile,
            wn_grid=wn_grid,
        )
        if self.cont_states_file is not None and self.cont_trans_files is not None:
            self.compute_continuum_rates_profiles(temperature_profile=temperature_profile, wn_grid=wn_grid)

        self.mol_chi_matrix = initial_chi_matrix
        lte_source_func_matrix = blackbody(
            spectral_grid=wn_grid, temperature=temperature_profile
        )
        self.mol_eta_matrix = lte_source_func_matrix * self.mol_chi_matrix * ac.c
        # TODO: Implement debug_pop_matrix here!
        self._build_a_ox_vals_cache()
        self._build_col_chem_cache(
            chem_profile=chem_profile,
            density_profile=density_profile,
            temperature_profile=temperature_profile,
        )

    def compute_approximate_t_ex(
            self,
            i_mean: u.Quantity,
            chem_profile: ChemicalProfile,
            density_profile: u.Quantity,
            temperature_profile: u.Quantity,
            wn_grid: u.Quantity,
            wn_dx: u.Quantity,
    ) -> None:
        n_layers = temperature_profile.shape[0]

        rates_filter = (pl.col("id_agg_f") <= self.id_agg_cutoff) & (pl.col("id_agg_i") <= self.id_agg_cutoff)

        a_fi_approx = self.rates_grid.select(
            pl.col("A_fi").filter(rates_filter).sum()
        ).item() * einstein_a_unit

        b_fi_approx = self.rates_grid.select(
            pl.col("B_fi").filter(rates_filter).sum()
        ).item() * einstein_b_unit

        b_if_approx = self.rates_grid.select(
            pl.col("B_if").filter(rates_filter).sum()
        ).item() * einstein_b_unit

        if self._cont_rates is not None:
            b_if_approx += self._cont_rates.select(
                pl.col("B_if").filter(pl.col("id_agg_i") <= self.id_agg_cutoff).sum()
            ).item() * einstein_b_unit
        # mol_chi_norm = simpson_normalise_quantity_2d(y_data=self.mol_chi_matrix, x_data=wn_grid)
        mol_chi_norm = loglinear_normalise_quantity_2d_nonnegative(y_data=self.mol_chi_matrix, dx=wn_dx)
        v_fi_approx = mol_chi_norm * b_fi_approx * i_mean
        # v_fi_rate = simpson_quantity_2d(y_data=v_fi_approx, x_data=wn_grid)
        v_fi_rate = loglinear_integral_quantity_2d_nonnegative(y_data=v_fi_approx, dx=wn_dx)
        v_if_approx = mol_chi_norm * b_if_approx * i_mean
        # v_if_rate = simpson_quantity_2d(y_data=v_if_approx, x_data=wn_grid)
        v_if_rate = loglinear_integral_quantity_2d_nonnegative(y_data=v_if_approx, dx=wn_dx)

        # energy_dif_approx = np.mean(np.diff(
        #     self.agg_states.filter(pl.col("energy_agg") <= wn_grid.value.max()).get_column("energy_agg").sort()
        #     .to_numpy()
        # )) / u.cm
        # c_fi_approx = density_profile * 1e-15 * u.m ** 3 / u.s
        # c_if_approx = c_fi_approx * np.exp(-(ac_h_c_on_kB * energy_dif_approx) / temperature_profile)
        # n_ratio_old = (c_if_approx + v_if_rate) / (c_fi_approx + a_fi_approx + v_fi_rate)
        # log.info(f"OLD: C_fi/C_if/N_ratio = {np.stack([c_fi_approx.value, c_if_approx.value, n_ratio_old]).T}")
        c_fi_approx, c_if_approx, mean_energy_dif = CollisionalRatesDatabase.compute_total_collisional_rates_profile(
            species=str(self.species),
            temperature_profile=temperature_profile,
            chem_profile=chem_profile,
            density_profile=density_profile,
            agg_lookup_cache=self._agg_lookup_cache,
            id_agg_cutoff=self.id_agg_cutoff,
        )
        # log.info(f"T_ex DEBUG: C_if = {c_if_approx}, C_fi = {c_fi_approx}, V_if = {v_if_rate}, V_fi = {v_fi_rate},"
        #          f" A_fi = {a_fi_approx}.")
        energy_dif_wmean_global = (mol_chi_norm @ wn_grid) / mol_chi_norm.sum(axis=1)
        n_ratio_old = (c_if_approx + v_if_rate) / (c_fi_approx + a_fi_approx + v_fi_rate)
        t_ex_profile_old = (ac_h_c_on_kB * energy_dif_wmean_global / np.log(1 / n_ratio_old)).to(u.K)
        log.log(
            _LOG_VERBOSE_3,
            f"{self.species} T_ex profile global fit = {np.array2string(t_ex_profile_old, formatter=_LOG_ARRAY_FMT)}"
        )
        # Get pairs of valid band keys: choosing adjacent IDs do not always exist, i.e.: in species with distinct
        # isomers whose states are close in energy.
        agg_energies = self.agg_states.sort("id_agg")["energy_agg"].to_numpy()
        max_id = self.id_agg_cutoff + 1
        num_pairs = max_id if max_id <= 10 else max(11, (max_id + 1) // 2)
        id_pairs = self.profile_store.get_sorted_band_keys(
            num_max=num_pairs, id_cutoff=self.id_agg_cutoff, agg_energies=agg_energies,
        )
        log.log(_LOG_VERBOSE_3, f"{self.species} num pairs = {num_pairs}, {len(id_pairs)} returned.")
        t_ex_profiles = np.zeros((len(id_pairs), n_layers), dtype=np.float64) << u.K
        log.log(_LOG_VERBOSE_3, f"{self.species} ID pairs for T_ex = {id_pairs}")
        for pair_idx, (id_agg_f, id_agg_i) in enumerate(id_pairs):
            id_filter = (pl.col("id_agg_f") == id_agg_f) & (pl.col("id_agg_i") == id_agg_i)
            a_fi = self.rates_grid.select(
                pl.col("A_fi").filter(id_filter).sum()
            ).item() * einstein_a_unit
            b_fi = self.rates_grid.select(
                pl.col("B_fi").filter(id_filter).sum()
            ).item() * einstein_b_unit
            b_if = self.rates_grid.select(
                pl.col("B_if").filter(id_filter).sum()
            ).item() * einstein_b_unit

            v_fi = np.zeros(n_layers) << 1 / u.s
            v_if = np.zeros(n_layers) << 1 / u.s
            energy_dif_wmean = np.zeros(n_layers) << wn_grid.unit
            for layer_idx in range(self.n_lte_layers, temperature_profile.shape[0]):
                nlte_layer_idx = layer_idx - self.n_lte_layers
                abs_profile, abs_start_idx = self.profile_store.get_profile(
                    layer_idx=nlte_layer_idx, key=(id_agg_f, id_agg_i), profile_type="abs"
                )
                abs_profile: u.Quantity = abs_profile << u.cm ** 2
                ste_profile, ste_start_idx = self.profile_store.get_profile(
                    layer_idx=nlte_layer_idx, key=(id_agg_f, id_agg_i), profile_type="ste"
                )
                ste_profile: u.Quantity = ste_profile << u.cm ** 2
                abs_end_idx = abs_start_idx + len(abs_profile)
                ste_end_idx = ste_start_idx + len(ste_profile)
                # abs_profile_norm = simpson_normalise_quantity_1d(
                #     y_data=abs_profile, x_data=wn_grid[abs_start_idx:abs_end_idx]
                # )
                # ste_profile_norm = simpson_normalise_quantity_1d(
                #     y_data=ste_profile, x_data=wn_grid[ste_start_idx:ste_end_idx]
                # )
                # v_fi[layer_idx] = simpson_quantity(
                #     y_data=ste_profile_norm * b_fi * i_mean[layer_idx, ste_start_idx:ste_end_idx],
                #     x_data=wn_grid[ste_start_idx:ste_end_idx]
                # )
                # v_if[layer_idx] = simpson_quantity(
                #     y_data=abs_profile_norm * b_if * i_mean[layer_idx, abs_start_idx:abs_end_idx],
                #     x_data=wn_grid[abs_start_idx:abs_end_idx]
                # )
                abs_profile_norm = loglinear_normalise_quantity_1d_nonnegative(
                    y_data=abs_profile, dx=wn_dx[abs_start_idx:abs_end_idx - 1]
                )
                ste_profile_norm = loglinear_normalise_quantity_1d_nonnegative(
                    y_data=ste_profile, dx=wn_dx[ste_start_idx:ste_end_idx - 1]
                )
                v_fi[layer_idx] = loglinear_integral_quantity_1d_nonnegative(
                    y_data=ste_profile_norm * i_mean[layer_idx, ste_start_idx:ste_end_idx],
                    dx=wn_dx[ste_start_idx:ste_end_idx - 1]
                ) * b_fi
                v_if[layer_idx] = loglinear_integral_quantity_1d_nonnegative(
                    y_data=abs_profile_norm * i_mean[layer_idx, abs_start_idx:abs_end_idx],
                    dx=wn_dx[abs_start_idx:abs_end_idx - 1]
                ) * b_if
                # log.info(f"[L{layer_idx}] {self.species} V_fi = {v_fi[layer_idx]:{_LOG_FLOAT_FMT}}"
                #          f" V_if = {v_if[layer_idx]:{_LOG_FLOAT_FMT}}.")
                energy_dif_wmean[layer_idx] = np.average(
                    wn_grid[abs_start_idx:abs_end_idx].value,
                    weights=abs_profile_norm.value,
                ) * wn_grid.unit
            c_fi = self._col_chem_c_matrix[:, id_agg_i, id_agg_f] << 1 / u.s
            c_if = self._col_chem_c_matrix[:, id_agg_f, id_agg_i] << 1 / u.s
            n_ratio = (c_if + v_if) / (c_fi + a_fi + v_fi)
            # energy_agg_f = self.agg_states.filter(pl.col("id_agg") == id_agg_f).select(pl.col("energy_agg")).item()
            # energy_agg_i = self.agg_states.filter(pl.col("id_agg") == id_agg_i).select(pl.col("energy_agg")).item()
            # energy_dif = (energy_agg_f - energy_agg_i) / u.cm
            # log.info(f"DEBUG: energy_dif = {energy_dif}, WMean = {energy_dif_wmean}.")
            t_ex_profile = (ac_h_c_on_kB * energy_dif_wmean / np.log(1 / n_ratio)).to(u.K)
            if np.any(t_ex_profile < 0):
                log.warning(f"{self.species} T_ex for ({id_agg_f}-{id_agg_i}) band contains negatives!")
            t_ex_profile = np.where(np.isnan(t_ex_profile), temperature_profile, t_ex_profile)
            t_ex_profile = np.clip(abs(t_ex_profile), a_min=0.0 * u.K, a_max=temperature_profile * 3.0)  # Failsafe
            log.log(
                _LOG_VERBOSE_2,
                f"{id_agg_f}-{id_agg_i} T_ex = {np.array2string(t_ex_profile, formatter=_LOG_ARRAY_FMT)}"
            )
            t_ex_profiles[pair_idx] = t_ex_profile
        # t_ex_profiles = np.stack(t_ex_profiles)
        t_ex_profile = t_ex_profiles.mean(axis=0)
        log.log(_LOG_VERBOSE_1, f"{self.species} T_ex = {t_ex_profile}")

        g_np = self.states["g"].to_numpy()
        energy_np = self.states["energy"].to_numpy() << 1 / u.cm
        id_agg_np = self.states["id_agg"].to_numpy()
        hcE_on_kB = ac_h_c_on_kB * energy_np

        cutoff_mask = id_agg_np <= self.id_agg_cutoff

        nlte_layer_slice = slice(self.n_lte_layers, None)
        t_ex_vals = t_ex_profile[nlte_layer_slice]
        t_k_vals = temperature_profile[nlte_layer_slice]
        n_nlte_layers = self.n_layers - self.n_lte_layers
        nlte_layers = range(self.n_lte_layers, self.n_layers)

        q_lte = g_np[None, :] * np.exp(-hcE_on_kB[None, :] / t_k_vals[:, None])
        n_lte = q_lte / q_lte.sum(axis=1, keepdims=True)
        n_agg_lte = np.zeros((n_nlte_layers, self.n_agg_states))
        np.add.at(n_agg_lte.T, id_agg_np, n_lte.T)

        t_eff = np.where(cutoff_mask, t_ex_vals[:, None], t_k_vals[:, None])
        q_eff = g_np[None, :] * np.exp(-hcE_on_kB[None, :] / t_eff)
        n_eff = q_eff / q_eff.sum(axis=1, keepdims=True)
        # This gives the aggregated populations using T_ex where needed, T_kin otherwise.
        n_agg_eff = np.zeros((n_nlte_layers, self.n_agg_states))
        # Witchcraft (adds each n_all to corresponding id_agg indices in n_agg_al, in place).
        np.add.at(n_agg_eff.T, id_agg_np, n_eff.T)
        # Equivalent to:
        # n_agg_all = np.stack([
        #     np.bincount(id_agg_np, weights=row, minlength=self.n_agg_states)
        #     for row in n_all
        # ])
        # Why the fuck is it done like this?
        # n_lte = self.states.select(
        #     [f"n_L{layer_idx}" for layer_idx in nlte_layers]
        # ).to_numpy().T
        # n_agg_lte = np.stack([
        #     (
        #         self.states[f"n_agg_L{layer_idx}"].to_numpy()[:self.n_agg_states]
        #         if f"n_agg_L{layer_idx}" in self.states.columns
        #         else self.pop_matrix[0, layer_idx]
        #     )[id_agg_np]
        #     for layer_idx in nlte_layers
        # ])
        # n_agg_nlte = n_agg_all[:, id_agg_np]  # n_agg for every state at every layer.
        # Map agg states back on to each state.
        n_agg_eff_map = n_agg_eff[:, id_agg_np]
        n_agg_lte_map = n_agg_lte[:, id_agg_np]

        scale_factor = np.divide(
            n_agg_eff_map,
            n_agg_lte_map,
            out=np.zeros_like(n_agg_eff_map),
            where=n_agg_lte_map != 0
        )
        # n_scaled = n_lte.copy()
        # n_scaled[:, cutoff_mask] *= scale_factor[:, cutoff_mask]
        n_scaled = n_lte * scale_factor
        # The LTE pops have been rescaled here, though the relative strengths in these bands remains the same.
        n_scaled /= n_scaled.sum(axis=1, keepdims=True)

        n_agg_final = np.zeros((n_nlte_layers, self.n_agg_states))
        np.add.at(n_agg_final.T, id_agg_np, n_scaled.T)

        # TODO: This is broken when there are null states in the grouping!
        # self._nlte_pop_frac[nlte_layer_slice] = n_scaled[:, cutoff_mask].sum(axis=1)
        self._nlte_pop_frac[nlte_layer_slice] = n_agg_final[:, :self.id_agg_cutoff + 1].sum(axis=1)
        log.log(_LOG_VERBOSE_2, f"{self.species} NLTE pop frac = {self._nlte_pop_frac}.")

        t_ex_pop_grid = np.zeros((1, self.pop_matrix.shape[1], self.pop_matrix.shape[2]))
        t_ex_pop_grid[0, :self.n_lte_layers] = self.pop_matrix[0, :self.n_lte_layers]
        t_ex_pop_grid[0, nlte_layer_slice] = n_agg_final
        log.log(_LOG_VERBOSE_2, f"{self.species} new pop grid sums = {t_ex_pop_grid[0].sum(axis=1)}")

        nlte_col_exprs = []
        for idx, layer_idx in enumerate(nlte_layers):
            nlte_col_exprs.extend([
                # pl.Series(f"n_nlte_L{layer_idx}", n_scaled[idx]),
                # pl.Series(f"n_agg_nlte_L{layer_idx}", n_agg_final[idx, id_agg_np]),
                # TODO: TEST THIS! Set as new LTE.
                pl.Series(f"n_L{layer_idx}", n_scaled[idx]),
                pl.Series(f"n_agg_L{layer_idx}", n_agg_final[idx, id_agg_np]),
            ])

        self.states = self.states.with_columns(nlte_col_exprs)
        self.pop_matrix = np.vstack((self.pop_matrix, t_ex_pop_grid))
        # return t_ex_profile

    def precompute_all_cross_terms(
            self,
            nlte_layer_idx: int,
            wn_grid: u.Quantity,
    ) -> u.Quantity:
        """
        Pre-compute a_ox_cross for all o_idx values. Called once per build_y_matrix call.

        NB: This could be cached for every layer but at high resolution and for polyatomics this could be on the order
        of 1Tb of RAM!

        Parameters
        ----------
        nlte_layer_idx : int
            Index into profile_store for the emission profiles at this layer.
        wn_grid : astropy.units.Quantity, shape (num_grid,)

        Returns
        -------
        a_ox_cross_cache : astropy.units.Quantity, shape (n_agg_states, num_grid)
        """

        spe_profiles_norm = self.profile_store.precompute_normalised_downward_emission_profiles(
            layer_idx=nlte_layer_idx,
            id_agg_cutoff=self.id_agg_cutoff,
            wn_grid=wn_grid,
        )
        # Normalised profiles have units 1/wn_grid.unit, i.e: u.cm.
        a_ox_cross_cache = spe_profiles_norm * self._a_ox_vals[:, None]
        # np.save(r"/mnt/c/PhD/NLTE/theory/cross_coupling/cross_terms_absolute.npy", a_ox_cross_cache)
        # np.save(r"/mnt/c/PhD/NLTE/theory/cross_coupling/cross_terms_normalised.npy", a_ox_cross_cache.value)
        return a_ox_cross_cache

    def build_y_matrix(
            self,
            layer_idx: int,
            nlte_layer_idx: int,
            i_layer_grid: u.Quantity,
            lambda_layer_grid: npt.NDArray[np.float64],
            chem_profile: ChemicalProfile,
            global_chi_matrix: u.Quantity,  # u.cm**2
            global_source_func_matrix: u.Quantity,
            wn_grid: u.Quantity,
            wn_dx: u.Quantity,
            full_prec: bool,
    ) -> None:
        """
        Build statistical equilibrium matrix.
        """
        num_grid = wn_grid.shape[0]

        species_eta: u.Quantity = chem_profile[self.species][layer_idx] * self.mol_eta_matrix[layer_idx] / ac.c
        # species_eta = self.mol_eta_matrix[layer_idx] / ac.c
        global_chi: u.Quantity = global_chi_matrix[layer_idx]  # / density_profile[layer_idx]
        chi_mask = global_chi != 0
        psi_approx_eta = np.zeros(num_grid) << global_source_func_matrix.unit
        psi_approx_eta[chi_mask] = (
                lambda_layer_grid[chi_mask] * species_eta[chi_mask] / global_chi[chi_mask]
        )
        psi_approx_eta = np.clip(abs(psi_approx_eta), 0, i_layer_grid)
        i_prec: u.Quantity = (i_layer_grid - psi_approx_eta) * 4 * np.pi * u.sr

        n_dim = self.id_agg_cutoff + 1
        self.y_matrix = np.zeros((n_dim, n_dim), dtype=np.float64) << (1 / u.s)
        self.rhs_matrix = np.zeros(n_dim, dtype=np.float64) << (1 / u.s)
        rates_start = time.perf_counter()

        psi_approx_cross = np.empty([])
        if full_prec:
            a_ox_cross_cache = self.precompute_all_cross_terms(
                nlte_layer_idx=nlte_layer_idx,
                wn_grid=wn_grid,
            )
            psi_approx_cross = np.zeros((n_dim, num_grid), dtype=np.float64) << a_ox_cross_cache.unit / global_chi.unit
            shielded_lambda = np.clip(lambda_layer_grid, 0, 1)

            psi_approx_cross[:, chi_mask] = (
                    shielded_lambda[chi_mask] * a_ox_cross_cache[:, chi_mask] / global_chi[chi_mask]
            )

        if self.debug:
            for trans_row in self.rates_grid.iter_rows(named=False):
                # 0 = id_agg_f, 1 = id_agg_i, 2 = A_fi, 3 = B_fi, 4 = B_if.
                if trans_row[0] > self.id_agg_cutoff or trans_row[1] > self.id_agg_cutoff:
                    # Short-circuit for bands involving states above cutoff; these pops are fixed and including them on RHS
                    # biases towards fixed distribution above cutoff.
                    continue
                a_fi = trans_row[2] * einstein_a_unit
                b_fi = trans_row[3] * einstein_b_unit
                b_if = trans_row[4] * einstein_b_unit
                log.debug(f"[L{layer_idx}] Trans: {trans_row}.")

                # These are pop-normalised within the band, but redundant due to normalisation.
                abs_profile, abs_start_idx = self.profile_store.get_profile(
                    layer_idx=nlte_layer_idx, key=(trans_row[0], trans_row[1]), profile_type="abs"
                )
                # abs_profile = abs_profile * self.pop_matrix[-1, layer_idx, trans_row[1]] << u.cm ** 2
                abs_profile: u.Quantity = abs_profile << u.cm ** 2
                ste_profile, ste_start_idx = self.profile_store.get_profile(
                    layer_idx=nlte_layer_idx, key=(trans_row[0], trans_row[1]), profile_type="ste"
                )
                # ste_profile = ste_profile * self.pop_matrix[-1, layer_idx, trans_row[0]] << u.cm ** 2
                ste_profile: u.Quantity = ste_profile << u.cm ** 2

                abs_end_idx = abs_start_idx + len(abs_profile)
                ste_end_idx = ste_start_idx + len(ste_profile)

                # Normalised profiles with units [cm].
                abs_profile_norm = loglinear_normalise_quantity_1d_nonnegative(
                    y_data=abs_profile, dx=wn_dx[abs_start_idx:abs_end_idx - 1]
                )
                ste_profile_norm = loglinear_normalise_quantity_1d_nonnegative(
                    y_data=ste_profile, dx=wn_dx[ste_start_idx:ste_end_idx - 1]
                )

                # U_fi is the integral of A_fi*phi_fi; phi_fi is integral normalised, so we can skip this.
                u_fi = a_fi
                u_fi = u_fi.decompose()

                chi_if: u.Quantity = np.zeros(num_grid) << abs_profile.unit
                chi_if[abs_start_idx: abs_end_idx] += (
                        self.pop_matrix[-1, layer_idx, trans_row[1]]
                        * abs_profile
                )
                chi_if[ste_start_idx: ste_end_idx] -= (
                        self.pop_matrix[-1, layer_idx, trans_row[0]]
                        * ste_profile
                )
                chi_if *= chem_profile[self.species][layer_idx]
                # chi_if = np.where(chi_if < 0, 0, chi_if)
                chi_if = np.clip(chi_if, a_min=0, a_max=None) << chi_if.unit
                if full_prec:
                    psi_approx_cross_if = np.abs(chi_if[None, :] * psi_approx_cross)
                    psi_integrals = loglinear_integral_quantity_2d_nonnegative(y_data=psi_approx_cross_if, dx=wn_dx)

                    nonzero_integral_mask = psi_integrals.value != 0

                    self.y_matrix[trans_row[1], nonzero_integral_mask] -= psi_integrals[nonzero_integral_mask]
                    self.y_matrix[trans_row[0], nonzero_integral_mask] += psi_integrals[nonzero_integral_mask]
                else:
                    # Here we compute (1 - Chi_if*Psi^{*})*Ufi, where Psi^{*} = Lambda^{*}[1/Chi_nu]. Expanding out the
                    # brackets, the first term has no wavenumber dependence, so we can skip the integral. The Chi_if*Psi^{*}
                    # term has a wavenumber dependence however, so we compute Lambda^{*}[Chi_if/Chi_nu]*phi_fi. The Lambda
                    # operator is dimensionless here and the normalised spontaneous emission profile has units of [cm];
                    # taking the yields the desired dimensionless factor to reduce A_fi by.
                    spe_profile, spe_start_idx = self.profile_store.get_profile(
                        layer_idx=nlte_layer_idx, key=(trans_row[0], trans_row[1]), profile_type="ste"
                    )
                    spe_profile = spe_profile << u.erg * u.cm / (u.s * u.sr)
                    spe_end_idx = spe_start_idx + len(spe_profile)
                    spe_profile_norm = loglinear_normalise_1d_nonnegative(
                        y_data=spe_profile, dx=wn_dx[spe_start_idx:spe_end_idx - 1]
                    )
                    self_prec = np.zeros(num_grid)
                    self_prec[chi_mask] = (
                            lambda_layer_grid[chi_mask]
                            * chi_if[chi_mask]
                            # * ac.h
                            / global_chi[chi_mask]
                    )
                    self_prec = self_prec[spe_start_idx:spe_end_idx] * spe_profile_norm
                    self_prec = loglinear_integral_quantity_1d(
                        y_data=self_prec, dx=wn_dx[spe_start_idx:spe_end_idx - 1]
                    )
                    u_fi *= 1 - self_prec
                # End cross.
                # DEBUG:
                # integrand = ste_profile_norm * i_prec[ste_start_idx: ste_end_idx]
                # plt.plot(wn_grid[ste_start_idx: ste_end_idx], integrand)
                # plt.xlim(left=wn_grid[ste_start_idx].value, right=wn_grid[ste_end_idx-1].value)
                # plt.title(f"V_{trans_row[0], trans_row[1]}_prec")
                # plt.xlabel(r"Wavenumbers (cm$^{-1}$)")
                # plt.ylabel(f"Integrand ({integrand.unit:latex})")
                # if np.all(integrand >= 0):
                #     plt.yscale("log")
                # plt.show()
                # plt.close()
                # log.info(f"DEBUG: ste/abs start/end idxs = {ste_start_idx, ste_end_idx, abs_start_idx, abs_end_idx}.")
                ##############################

                log.debug(f"[L{layer_idx}] U_{trans_row[0], trans_row[1]} = {u_fi}")
                v_fi_prec = loglinear_integral_quantity_1d(
                    y_data=ste_profile_norm * i_prec[ste_start_idx: ste_end_idx],
                    dx=wn_dx[ste_start_idx: ste_end_idx - 1]
                ) * b_fi
                v_fi_prec = v_fi_prec.decompose()
                log.debug(f"[L{layer_idx}] V_{trans_row[0], trans_row[1]}_prec = {v_fi_prec:{_LOG_FLOAT_FMT}}")

                v_if_prec = loglinear_integral_quantity_1d(
                    y_data=abs_profile_norm * i_prec[abs_start_idx: abs_end_idx],
                    dx=wn_dx[abs_start_idx: abs_end_idx - 1]
                ) * b_if
                v_if_prec = v_if_prec.decompose()
                log.debug(f"[L{layer_idx}] V_{trans_row[1], trans_row[0]}_prec = {v_if_prec:{_LOG_FLOAT_FMT}}")

                self.y_matrix[trans_row[0], trans_row[1]] += v_if_prec
                self.y_matrix[trans_row[1], trans_row[0]] += u_fi + v_fi_prec
                self.y_matrix[trans_row[0], trans_row[0]] -= u_fi + v_fi_prec
                self.y_matrix[trans_row[1], trans_row[1]] -= v_if_prec

            if self._cont_rates is not None:
                # log.debug(f"Cont rates = {self.cont_rates}")
                # log.debug((
                #     f"Cont profile store keys = "
                #     f"{self.cont_profile_store.get_keys(layer_idx=nlte_layer_idx, profile_type="abs")}"
                # ))
                for cont_trans_row in self._cont_rates.iter_rows(named=False):
                    if cont_trans_row[0] > self.id_agg_cutoff:
                        # Short-circuit for bands involving states above cutoff; these pops are fixed and including them on
                        # RHS biases towards fixed distribution above cutoff.
                        continue

                    a_ci = cont_trans_row[1] * einstein_a_unit
                    # b_ci = cont_trans_row[2] * einstein_b_unit
                    b_ic = cont_trans_row[3] * einstein_b_unit

                    log.debug(f"{self.species}: Cont. profile for state {cont_trans_row[0]}.")
                    cont_abs_profile, cont_abs_start_idx = self._cont_profile_store.get_profile(
                        layer_idx=nlte_layer_idx, key=cont_trans_row[0], profile_type="abs"
                    )
                    cont_abs_profile: u.Quantity = cont_abs_profile << u.cm ** 2
                    cont_abs_end_idx = cont_abs_start_idx + len(cont_abs_profile)

                    cont_abs_profile_norm = loglinear_normalise_quantity_1d_nonnegative(
                        y_data=cont_abs_profile, dx=wn_dx[cont_abs_start_idx:cont_abs_end_idx - 1]
                    )

                    # Cross terms:
                    chi_ic: u.Quantity = np.zeros(num_grid) << cont_abs_profile.unit
                    chi_ic[cont_abs_start_idx: cont_abs_end_idx] += (
                            self.pop_matrix[-1, layer_idx, cont_trans_row[0]]
                            * cont_abs_profile
                            * chem_profile[self.species][layer_idx]
                    )
                    chi_ic = np.clip(chi_ic, a_min=0, a_max=None) << chi_ic.unit
                    if full_prec:
                        psi_approx_cross_ic = np.abs(chi_ic[None, :] * psi_approx_cross)
                        psi_integrals = loglinear_integral_quantity_2d_nonnegative(
                            y_data=psi_approx_cross_ic, dx=wn_dx
                        )

                        # log.debug(f"[L{layer_idx}] chi_psi_{cont_trans_row[0]}c = {psi_integrals}")
                        nonzero_integral_mask = psi_integrals.value != 0

                        # This may be the wrong way round as [0] is i; confirm against numba kernel.
                        self.y_matrix[cont_trans_row[0], nonzero_integral_mask] += psi_integrals[nonzero_integral_mask]
                    # End cross.
                    v_ic_prec = loglinear_integral_quantity_1d(
                        y_data=cont_abs_profile_norm * i_prec[cont_abs_start_idx: cont_abs_end_idx],
                        dx=wn_dx[cont_abs_start_idx: cont_abs_end_idx - 1]
                    ) * b_ic
                    v_ic_prec = v_ic_prec.decompose()
                    log.debug(f"[L{layer_idx}] V_{cont_trans_row[0]}c_prec = {v_ic_prec:{_LOG_FLOAT_FMT}}")

                    limiting_species_num_dens = min(
                        (
                            chem_profile[self.dissociation_products[0]][layer_idx]
                            if self.dissociation_products[0] in chem_profile.species
                            else 0
                        ),
                        (
                            chem_profile[self.dissociation_products[1]][layer_idx]
                            if self.dissociation_products[1] in chem_profile.species
                            else 0
                        ),
                    )
                    if limiting_species_num_dens == 0:
                        limiting_scale_factor = 0
                    else:
                        mol_num_dens = chem_profile[self.species][layer_idx]
                        i_pop = self.pop_matrix[-1, layer_idx, cont_trans_row[0]]
                        limiting_scale_factor = i_pop * mol_num_dens / limiting_species_num_dens

                    self.y_matrix[cont_trans_row[0], cont_trans_row[0]] -= v_ic_prec
                    # self.y_matrix[cont_trans_row[0], cont_trans_row[0]] += a_ci * z_ci * limiting_scale_factor
                    self.y_matrix[cont_trans_row[0], cont_trans_row[0]] += a_ci * limiting_scale_factor
                    # self.y_matrix[cont_trans_row[0], cont_trans_row[0]] += v_ci_prec * limiting_scale_factor
        else:
            id_agg_f = self.rates_grid["id_agg_f"].to_numpy().astype(np.int32)
            id_agg_i = self.rates_grid["id_agg_i"].to_numpy().astype(np.int32)
            rates_grid_arr = self.rates_grid.select(["A_fi", "B_fi", "B_if"]).to_numpy().astype(np.float64)

            n_rates = id_agg_f.shape[0]

            abs_store = self.profile_store.abs_profiles[nlte_layer_idx]
            ste_store = self.profile_store.ste_profiles[nlte_layer_idx]
            spe_store = self.profile_store.spe_profiles[nlte_layer_idx]

            abs_profile_idx = np.zeros(n_rates, dtype=np.int64)
            ste_profile_idx = np.zeros(n_rates, dtype=np.int64)
            spe_profile_idx = np.zeros(n_rates, dtype=np.int64)

            for r in range(n_rates):
                if id_agg_f[r] > self.id_agg_cutoff or id_agg_i[r] > self.id_agg_cutoff:
                    continue
                key = (int(id_agg_f[r]), int(id_agg_i[r]))
                abs_profile_idx[r] = abs_store.key_lookup[key]
                ste_profile_idx[r] = ste_store.key_lookup[key]
                spe_profile_idx[r] = spe_store.key_lookup[key]
            n_lookup = np.asarray(self.pop_matrix[-1, layer_idx, :], dtype=np.float64)
            y_matrix_val = self.y_matrix.value  # in-place mutation target
            chem_scale_factor = chem_profile[self.species][layer_idx]

            if full_prec:
                psi_approx_cross_val = psi_approx_cross.value
            else:
                psi_approx_cross_val = np.empty((0, 0), dtype=np.float64)

            _build_y_matrix_core(
                id_agg_f=id_agg_f,
                id_agg_i=id_agg_i,
                rates_grid_arr=rates_grid_arr,
                id_agg_cutoff=self.id_agg_cutoff,
                n_lookup=n_lookup,
                chem_scale_factor=chem_scale_factor,
                lambda_layer_grid=lambda_layer_grid,
                global_chi=global_chi.value,
                i_prec=i_prec.value,
                wn_dx=wn_dx.value,
                abs_profiles=abs_store.profiles,
                abs_offsets=abs_store.offsets,
                abs_start_idxs=abs_store.start_idxs,
                abs_profile_idx=abs_profile_idx,
                ste_profiles=ste_store.profiles,
                ste_offsets=ste_store.offsets,
                ste_start_idxs=ste_store.start_idxs,
                ste_profile_idx=ste_profile_idx,
                spe_profiles=spe_store.profiles,
                spe_offsets=spe_store.offsets,
                spe_start_idxs=spe_store.start_idxs,
                spe_profile_idx=spe_profile_idx,
                full_prec=full_prec,
                psi_approx_cross=psi_approx_cross_val,
                y_matrix=y_matrix_val,
            )
            if self._cont_rates is not None:
                # id_agg_c = self._cont_rates["id_agg_f"].to_numpy().astype(np.int32)
                id_agg_i = self._cont_rates["id_agg_i"].to_numpy().astype(np.int32)
                rates_grid_arr = self._cont_rates.select(["A_fi", "B_fi", "B_if"]).to_numpy().astype(np.float64)

                n_rates = id_agg_i.shape[0]

                abs_store = self._cont_profile_store.abs_profiles[nlte_layer_idx]

                abs_profile_idx = np.zeros(n_rates, dtype=np.int64)

                for r in range(n_rates):
                    if id_agg_i[r] > self.id_agg_cutoff:
                        continue
                    # TODO: Check -1 key carried forward.
                    key = (-1, int(id_agg_i[r]))
                    abs_profile_idx[r] = abs_store.key_lookup[key]

                limiting_species_num_dens = min([
                    chem_profile[dis_prod][layer_idx] if dis_prod in chem_profile.species else 0
                    for dis_prod in self.dissociation_products
                ])

                _build_y_matrix_cont(
                    id_agg_i=id_agg_i,
                    rates_grid_arr=rates_grid_arr,
                    id_agg_cutoff=self.id_agg_cutoff,
                    n_lookup=n_lookup,
                    chem_scale_factor=chem_scale_factor,
                    lambda_layer_grid=lambda_layer_grid,
                    i_prec=i_prec.value,
                    wn_dx=wn_dx.value,
                    abs_profiles=abs_store.profiles,
                    abs_offsets=abs_store.offsets,
                    abs_start_idxs=abs_store.start_idxs,
                    abs_profile_idx=abs_profile_idx,
                    limiting_species_num_dens=limiting_species_num_dens,
                    full_prec=full_prec,
                    psi_approx_cross=psi_approx_cross,
                    y_matrix=y_matrix_val,
                )
        # Add collisional and chemical rates.
        self.add_col_chem_rates(layer_idx=layer_idx)
        log.log(
            _LOG_VERBOSE_2,
            f"[L{layer_idx}] {self.species} Y matrix construction duration = {time.perf_counter() - rates_start:.5f}"
        )

    def add_col_chem_rates(
            self,
            layer_idx: int,
    ) -> None:
        """
        Apply precomputed collisional/chemical rate contributions for a single layer.

        Must call _build_col_chem_cache() once before iterating over layers.

        Parameters
        ----------
        layer_idx : int
            Index of the atmospheric layer being processed.
        """
        if self._col_chem_c_matrix is not None:
            self.y_matrix += self._col_chem_c_matrix[layer_idx] * self.y_matrix.unit
            self.rhs_matrix += self._col_chem_rhs_c[layer_idx] * self.rhs_matrix.unit

    def solve_pops(
            self,
            layer_idx: int,
            n_iter: int,
    ) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        self.y_reduced_idx_map = np.where(np.abs(self.y_matrix).sum(axis=1) != 0)[0]
        y_matrix_reduced = self.y_matrix[np.ix_(self.y_reduced_idx_map, self.y_reduced_idx_map)]
        log.log(
            _LOG_VERBOSE_2,
            (
                f"[L{layer_idx}] {self.species} Y matrix (before row-normalisation) =\n"
                f"{np.array2string(y_matrix_reduced.value, formatter=_LOG_ARRAY_FMT)}\n"
                f"[L{layer_idx}] {self.species} Y matrix cond. (before row-normalisation) ="
                f" {np.linalg.cond(y_matrix_reduced.value):{_LOG_FLOAT_FMT}}"
            )
        )
        norm_factors = abs(y_matrix_reduced).sum(axis=1)[:, None]
        y_matrix_reduced /= norm_factors
        check_rows = np.array([
            np.all(y_matrix_reduced[idx, :] > 0) or np.all(y_matrix_reduced[idx, :] < 0)
            for idx in range(y_matrix_reduced.shape[0])
        ])
        if np.any(check_rows):
            log.error(
                f"[I{n_iter}][L{layer_idx}] Y matrix all same sign in rows "
                f"{np.nonzero(check_rows)[0]}; investigate unphysical rates."
            )

        y_rect = np.vstack([y_matrix_reduced.copy(), np.ones(y_matrix_reduced.shape[1])])
        rhs_rect = self.rhs_matrix[self.y_reduced_idx_map] / norm_factors[:, 0]
        rhs_rect = np.append(rhs_rect, 1)

        log.log(_LOG_VERBOSE_2, f"{self.species} Y matrix =\n{np.array2string(y_rect, precision=3)}")
        nppinv_pops = np.linalg.pinv(y_rect) @ rhs_rect
        nppinv_pops /= nppinv_pops.sum()

        if np.any(nppinv_pops < 0):
            log.error((
                f"[L{layer_idx}] Numpy Pseudo Inverse pops. contain negatives; falling back to least squares.\n"
                f"Negatives = {np.array2string(nppinv_pops, formatter=_LOG_ARRAY_FMT)}"
            ))
            lsq_res = least_squares(
                lambda x: np.dot(y_rect, x) - rhs_rect,
                # lambda x: np.log(np.dot(y_rect, x)) - np.log(rhs_rect),
                np.zeros(y_rect.shape[1]),
                bounds=(0.0, 1.0),
                method="trf",
                ftol=1e-15,
                gtol=1e-15,
                xtol=1e-15,
            )
            least_squares_pops = lsq_res.x
            least_squares_pops /= least_squares_pops.sum()
            log.log(
                _LOG_VERBOSE_2,
                (
                    f"[L{layer_idx}] Least Squares res = {lsq_res}\n"
                    f"Least Squares Pops. = {np.array2string(least_squares_pops, formatter=_LOG_ARRAY_FMT)}"
                )
            )
            if any(least_squares_pops < 0):
                raise RuntimeError(
                    f"[L{layer_idx}] Least squares population bounds failed; negative pops."
                )
            else:
                pop_matrix = least_squares_pops
        else:
            pop_matrix = nppinv_pops

        pop_old = self.pop_matrix[-1, layer_idx, self.y_reduced_idx_map].copy()
        # Important: If there are states above the cutoff (i.e.: self._nlte_pop_frac < 1) then these will be on a
        # different scale if not normalised.
        pop_old /= pop_old.sum()

        return pop_matrix, pop_old

    def update_pops(
            self,
            layer_idx: int,
            pop_updated: npt.NDArray[np.float64],
    ):
        # Normalise to required NLTE population fraction.
        pop_scaled = self._nlte_pop_frac[layer_idx] * pop_updated / pop_updated.sum()

        log.log(_LOG_VERBOSE_2, f"[L{layer_idx}] {self.species} New pops.:")
        for idx, y_idx in enumerate(self.y_reduced_idx_map):
            log.log(
                _LOG_VERBOSE_2,
                (
                    f"[L{layer_idx}]"
                    f" n{self.agg_states.filter(pl.col("id_agg") == y_idx).select(self.agg_col_names).row(0)}"
                    f" = {pop_scaled[idx]:{_LOG_FLOAT_FMT}}"
                )
            )
        # Store all updated pops, any zeros that were excldued from the Y-matrix calculation and pops above cutoff.
        pop_updated_full = np.zeros(self.n_agg_states)
        pop_updated_full[self.y_reduced_idx_map] = pop_scaled
        pop_updated_full[self.id_agg_cutoff + 1:] = self.pop_matrix[-1, layer_idx, self.id_agg_cutoff + 1:]  # TEST!

        n_agg_lte_col = f"n_agg_L{layer_idx}"
        n_lte_col = f"n_L{layer_idx}"
        n_agg_nlte_col = f"n_agg_nlte_L{layer_idx}"
        n_nlte_col = f"n_nlte_L{layer_idx}"
        self.states = self.states.with_columns(
            pl.Series(pop_updated_full).gather(self.states["id_agg"]).alias(n_agg_nlte_col)
        )
        # TODO: In the case where T_ex has been approximated, the LTE pops above the cutoff have been rescaled!
        #  Use those somehow - if the degree of NLTE decreases from the first T_ex run then states above cutoff are
        #  froxen into a more NLTE distribution.
        #  In non-T_ex runs, n_nlte_col doesn't exist the first iteration to pull from - rebalance in
        #  the n_lte_col directly?
        self.states = self.states.with_columns(
            pl.when(pl.col(n_agg_lte_col) == 0)
            .then(0)
            # If above cutoff, leave the states in LTE.
            .when(pl.col("id_agg") > self.id_agg_cutoff)
            .then(pl.col(n_lte_col))
            .otherwise(pl.col(n_lte_col) * pl.col(n_agg_nlte_col) / pl.col(n_agg_lte_col))
            .alias(n_nlte_col)
        )
        log_col_names = ["id", "energy", "g", "tau", n_lte_col, n_agg_lte_col, n_agg_nlte_col, n_nlte_col]
        log.log(
            _LOG_VERBOSE_2,
            (
                f"[L{layer_idx}] {self.species} NLTE States = \n{self.states.select(log_col_names)}\n"
                f"[L{layer_idx}] Sum of LTE populations = {self.states[n_lte_col].sum()}.\n"
                f"[L{layer_idx}] Sum of non-LTE populations = {self.states[n_nlte_col].sum()}."
            )
        )
        return pop_updated_full

    def update_layer_global_chi_eta(
            self,
            wn_grid: u.Quantity,
            layer_vmr: float,
            layer_global_chi_matrix: u.Quantity,
            layer_global_eta_matrix: u.Quantity,
            layer_idx: int,
            nlte_layer_idx: int,
            layer_pop_grid: npt.NDArray[np.float64] = None,
    ) -> None:
        if layer_pop_grid is None:
            # Use for T_ex approximation when all new popualtions stored internally.
            layer_pop_grid = self.pop_matrix[-1, layer_idx]

        start_time = time.perf_counter()
        abs_xsec, emi_xsec = self.profile_store.build_abs_emi(
            layer_idx=nlte_layer_idx, pop_matrix=layer_pop_grid, wn_grid=wn_grid.value,
        )
        log.log(_LOG_VERBOSE_2, f"[L{layer_idx}] {self.species} CBP duration = {time.perf_counter() - start_time:.5f}")

        if np.any(abs_xsec < 0):
            log.warning(
                f"[L{layer_idx}] {self.species} Negative contribution in absorption (stimulated emission dominates)"
            )

        if self._cont_states is not None and self.cont_trans_files is not None:
            cont_abs_xsec = self._cont_profile_store.build_abs(
                layer_idx=nlte_layer_idx, pop_matrix=layer_pop_grid, wn_grid=wn_grid.value,
            )
            abs_xsec += cont_abs_xsec

        # Update layer chi globally and then for species.
        abs_xsec = abs_xsec << u.cm ** 2
        layer_global_chi_matrix += (
                (abs_xsec - self.mol_chi_matrix[layer_idx]) * layer_vmr  # * layer_density
        )
        self.mol_chi_matrix[layer_idx] = abs_xsec
        # Update layer eta globally and then for species.
        emi_xsec = emi_xsec << u.erg * u.cm / (u.s * u.sr)
        layer_global_eta_matrix += (
                (emi_xsec - self.mol_eta_matrix[layer_idx]) * layer_vmr
        )
        self.mol_eta_matrix[layer_idx] = emi_xsec

    def commit_pops(
            self,
            pop_update_grid: npt.NDArray[np.float64],
    ) -> None:
        """
        Stores the updated populations from all layers for this iteration.

        Parameters
        ----------
        pop_update_grid : ndarray
            Updated populations across all layer from the current complete iteration.
        """
        self.pop_matrix = np.vstack((
            self.pop_matrix, pop_update_grid.reshape((1, self.pop_matrix.shape[1], self.pop_matrix.shape[2]))
        ))
        # TEMP!
        # if self.debug:
        with open((output_dir / f"{self.species}_pop_matrix.pickle").resolve(), "wb") as pickle_file:
            pickle.dump(self.pop_matrix, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    def finalise(self, temperature_profile: u.Quantity, pressure_profile: u.Quantity, wn_grid: u.Quantity) -> None:
        """
        Saves the population matrix, absorption and emission cross-sections profile to disk. Recomputes the
        cross-sections on the input grid, allowing for computation on a high-resolution grid after iteration.

        Parameters
        ----------
        temperature_profile
        pressure_profile
        wn_grid

        Returns
        -------

        """
        with open((output_dir / f"{self.species}_pop_matrix.pickle").resolve(), "wb") as pickle_file:
            pickle.dump(self.pop_matrix, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        # self.mol_chi_matrix = super().opacity(temperature, pressure, spectral_grid)
        # self.mol_source_func_matrix = blackbody(spectral_grid=spectral_grid, temperature=temperature)
        # self.mol_eta_matrix = self.mol_source_func_matrix * self.mol_chi_matrix * ac.c
        # Clean up large, no longer needed properties?
        self.profile_store = None
        self._cont_profile_store = None
        # Layered calculations.
        n_nlte_layers = self.n_layers - self.n_lte_layers
        abs_xsec, emi_xsec = abs_emi_xsec(
            states=self.states,
            trans_files=self.trans_files,
            n_lte_layers=self.n_lte_layers,
            n_nlte_layers=n_nlte_layers,
            temperature_profile=temperature_profile,
            pressure_profile=pressure_profile,
            wn_grid=wn_grid,
            species_mass=self.species_mass,
            do_super_lines=self.do_super_lines,
            broadening_params=self.broadening_params,
        )
        if self._cont_states is not None:
            cont_xsec = continuum_xsec(
                states=self.states,
                cont_states=self._cont_states,
                cont_trans_files=self.cont_trans_files,
                n_lte_layers=self.n_lte_layers,
                n_nlte_layers=n_nlte_layers,
                temperature_profile=temperature_profile,
                wn_grid=wn_grid.value,
                species_mass=self.species_mass,
                reduced_mass=self.reduced_mass,
                cont_box_length=self.cont_box_length,
                do_super_lines=self.cont_do_super_lines,
            )
            abs_xsec += cont_xsec
        abs_xsec = abs_xsec << u.cm ** 2
        self.mol_chi_matrix[self.n_lte_layers:] = abs_xsec
        emi_xsec = emi_xsec << u.erg * u.cm / (u.s * u.sr)
        self.mol_eta_matrix[self.n_lte_layers:] = emi_xsec

        # for layer_idx in range(self.n_lte_layers, len(temperature_profile)):
        #     layer_temp = temperature_profile[layer_idx]
        #     layer_pres = pressure_profile[layer_idx]
        #
        #     n_agg_lte_col = f"n_agg_L{layer_idx}"
        #     n_lte_col = f"n_L{layer_idx}"
        #     n_agg_nlte_col = f"n_agg_nlte_L{layer_idx}"
        #     n_nlte_col = f"n_nlte_L{layer_idx}"
        #     self.states = self.states.with_columns(
        #         pl.Series(self.pop_matrix[-1, layer_idx]).gather(self.states["id_agg"]).alias(n_agg_nlte_col)
        #     )
        #     self.states = self.states.with_columns(
        #         pl.when(pl.col(n_agg_lte_col) == 0)
        #         .then(0)
        #         .otherwise(pl.col(n_lte_col) * pl.col(n_agg_nlte_col) / pl.col(n_agg_lte_col))
        #         .alias(n_nlte_col)
        #     )
        #
        #     abs_xsec, emi_xsec = abs_emi_xsec(
        #         states=self.states,
        #         trans_files=self.trans_files,
        #         temperature_profile=temperature_profile,
        #         pressure_profile=pressure_profile,
        #         species_mass=self.species_mass,
        #         wn_grid=wn_grid.value,
        #         broadening_params=self.broadening_params,
        #     )
        #     if self._cont_states is not None:
        #         nlte_select_cols = [pl.col("id"), pl.col(n_nlte_col)]
        #         nlte_cont_states = self._cont_states.join(self.states.select(nlte_select_cols), on="id", how="left")
        #         cont_xsec = continuum_xsec(
        #             continuum_states=nlte_cont_states,
        #             continuum_trans_files=self.cont_trans_files,
        #             layer_idx=layer_idx,
        #             wn_grid=wn_grid.value,
        #             temperature=layer_temp,
        #             species_mass=self.species_mass,
        #             cont_box_length=self.cont_box_length
        #         )
        #         abs_xsec += cont_xsec
        #         # np.savetxt(
        #         #     (
        #         #             output_dir
        #         #             / f"nLTE_cxsec_L{layer_idx}_T{int(layer_temp.value)}_P{layer_pres.value:.4e}.txt"
        #         #     ).resolve(),
        #         #     np.array([wn_grid.value, cont_xsec]).T,
        #         #     fmt="%17.8E",
        #         # )
        #         # # LTE Comparison
        #         # lte_select_cols = [pl.col("id"), pl.col(n_lte_col)]
        #         # lte_cont_states = self._cont_states.join(self.states.select(lte_select_cols), on="id", how="left")
        #         # # lte_cont_states = self.cont_states.merge(nlte_states[["id", "n"]], on="id", how="left")
        #         # # Fudge for how continuum_xsec() picks column.
        #         # lte_cont_states = lte_cont_states.with_columns(pl.col(n_lte_col).alias(n_nlte_col))
        #         # lte_cont_xsec = continuum_xsec(
        #         #     continuum_states=lte_cont_states,
        #         #     continuum_trans_files=self.cont_trans_files,
        #         #     layer_idx=layer_idx,
        #         #     wn_grid=wn_grid.value,
        #         #     temperature=layer_temp,
        #         #     species_mass=self.species_mass,
        #         #     cont_box_length=self.cont_box_length
        #         # )
        #         # np.savetxt(
        #         #     (
        #         #             output_dir
        #         #             / f"LTE_cxsec_L{layer_idx}_T{int(layer_temp.value)}_P{layer_pres.value:.4e}.txt"
        #         #     ).resolve(),
        #         #     np.array([wn_grid.value, lte_cont_xsec]).T,
        #         #     fmt="%17.8E",
        #         # )
        #     abs_xsec = abs_xsec << u.cm ** 2
        #     self.mol_chi_matrix[layer_idx] = abs_xsec
        #     emi_xsec = emi_xsec << u.erg * u.cm / (u.s * u.sr)
        #     self.mol_eta_matrix[layer_idx] = emi_xsec
        with open((output_dir / f"{self.species}_abs_xsec.pickle").resolve(), "wb") as abs_pickle_file:
            pickle.dump(self.mol_chi_matrix, abs_pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        with open((output_dir / f"{self.species}_emi_xsec.pickle").resolve(), "wb") as emi_pickle_file:
            pickle.dump(self.mol_eta_matrix, emi_pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
