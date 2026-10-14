import logging
import pathlib
import pickle
import time
import typing as t

import numpy as np
import pandas as pd
import polars as pl
import numba
from numpy import typing as npt

from astropy import units as u, constants as ac

from scipy.optimize import least_squares

from .chemistry import SpeciesFormula, ChemicalProfile
from .colchem import CollisionalRatesDatabase, RateTransition
from .constants import ac_h_c_on_kB, einstein_a_unit, einstein_b_unit
from .config import output_dir, _LOG_FLOAT_FMT, _LOG_ARRAY_FMT, _LOG_VERBOSE_1, _LOG_VERBOSE_2, \
    _LOG_VERBOSE_3, _VMR_THRESHOLD, _DEFAULT_NUM_THREADS
from .numerics import (loglinear_normalise_1d_nonnegative,
                       loglinear_normalise_quantity_1d_nonnegative, loglinear_integral_quantity_1d,
                       loglinear_integral_quantity_1d_nonnegative, loglinear_integral_quantity_2d_nonnegative,
                       loglinear_integral_1d, loglinear_integral_1d_nonnegative)
from .profiles import ProfileStore, ContinuumProfileStore, abs_emi_xsec, continuum_xsec
from .profiles.bands import assign_band_indices, band_rate_sums, group_by_band
from .profiles.lineshape import grid_bin_edges, layer_broadening_terms, gamma_lifetime_lookup_from_tau
from .profiles.read import iter_trans_batches, read_ahead
from .profiles.sampled import band_profile_sampled_voigt, continuum_band_profile_sampled_gauss_layered
from .profiles.superline import (accumulate_superline_band_batch, accumulate_continuum_superline_band_batch,
                                 broaden_superline_band_buffer, broaden_continuum_superline_band_buffer)
from .rt import blackbody, shield_masing_chi
from . import config

log = logging.getLogger(__name__)


# TODO: For NANs in state lifetimes; treat as inf? They imply inf but often they exist because of transition energy
#  cutoffs during computation and not because the state has no deexcitation pathways.


# ----------------------------------------- Rates & Profiles Parser Functions -----------------------------------------
def _init_band_registry(
        id_agg: npt.NDArray[np.int32],
) -> t.Tuple[npt.NDArray[np.int32], int, npt.NDArray[np.int32]]:
    """
    Creates an empty band registry covering every aggregate state ID in id_agg.

    Parameters
    ----------
    id_agg : ndarray
        Aggregate state IDs of all states that transitions can connect. May include negative IDs (e.g. -1 for
        unaggregated continuum states).

    Returns
    -------
    band_lookup : ndarray, shape (n_agg_ids, n_agg_ids)
        Dense (id_agg_f, id_agg_i) -> band index table, offset by id_agg_offset and filled with -1 (unregistered).
    id_agg_offset : int
    band_keys : ndarray, shape (0, 2)
        (id_agg_f, id_agg_i) of each registered band, indexed by band index.
    """
    id_agg_offset = -int(id_agg.min())
    n_agg_ids = int(id_agg.max()) + id_agg_offset + 1
    band_lookup = np.full((n_agg_ids, n_agg_ids), -1, dtype=np.int32)
    band_keys = np.empty((0, 2), dtype=np.int32)
    return band_lookup, id_agg_offset, band_keys


def _update_band_registry(
        id_f: npt.NDArray[np.int32],
        id_i: npt.NDArray[np.int32],
        id_agg_lookup: npt.NDArray[np.int32],
        band_lookup: npt.NDArray[np.int32],
        id_agg_offset: int,
        band_keys: npt.NDArray[np.int32],
) -> t.Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """
    Registers any new (id_agg_f, id_agg_i) bands in a transition batch and returns per-transition band indices.

    Called by :func:`~xsec.NLTEProcessor.compute_rates_profiles`. The caller grows its profile buffer to
    band_keys.shape[0] bands afterwards (:func:`_grow_band_buffer` on the host).

    Parameters
    ----------
    id_f : ndarray, shape (n_trans,)
    id_i : ndarray, shape (n_trans,)
    id_agg_lookup : ndarray, shape (n_states + 1,)
        Aggregate state ID by state ID.
    band_lookup : ndarray, shape (n_agg_ids, n_agg_ids)
        From :func:`_init_band_registry`; updated in place.
    id_agg_offset : int
    band_keys : ndarray, shape (n_bands_used, 2)

    Returns
    -------
    band_keys : ndarray, shape (n_bands_used, 2)
    band_indices : ndarray, shape (n_trans)

    """
    n_bands_prev = band_keys.shape[0]
    band_indices, n_bands_used = assign_band_indices(
        id_f=id_f, id_i=id_i, id_agg_lookup=id_agg_lookup, band_lookup=band_lookup, id_agg_offset=id_agg_offset,
        n_bands_used=n_bands_prev,
    )
    if n_bands_used > n_bands_prev:
        new_f, new_i = np.nonzero(band_lookup >= n_bands_prev)
        new_rows = band_lookup[new_f, new_i] - n_bands_prev
        new_keys = np.empty((n_bands_used - n_bands_prev, 2), dtype=np.int32)
        new_keys[new_rows, 0] = new_f - id_agg_offset
        new_keys[new_rows, 1] = new_i - id_agg_offset
        band_keys = np.concatenate([band_keys, new_keys])
    return band_keys, band_indices


def _grow_band_buffer(profile_buffer: npt.NDArray[np.float64], n_bands: int) -> npt.NDArray[np.float64]:
    """
    Returns profile_buffer, shape (x, capacity, n_layers, n_grid), grown along the band axis (at least doubling) if it
    cannot hold n_bands bands.
    """
    capacity = profile_buffer.shape[1]
    if n_bands <= capacity:
        return profile_buffer
    new_capacity = capacity + max(n_bands - capacity, capacity)
    grown = np.zeros(profile_buffer.shape[:1] + (new_capacity,) + profile_buffer.shape[2:])
    grown[:, :capacity] = profile_buffer  # Faster than np.concatenate, which also copies the zeros.
    return grown


# -------------------------------------------------- Y matrix kernels --------------------------------------------------
# Photon-counting weights. A band's radiative rate is B_band * int(phi J dnu): the band sum B_band = sum_lines f B
# (profiles.bands.band_rate_sums) times J averaged over the band with each line weighted by its share f B of that sum.
# The stored profiles are cross-sections (energy removed from, or added to, the beam): a line's integrated cross-section
# is (h nu / 4 pi) B, so a normalised cross-section profile weights each line by f B nu, not f B. Rates count
# transitions (absorbed power / h nu), so every profile normalised as a rate weight here (absorption, stimulated and
# spontaneous emission, continuum) is first divided by the wavenumber. Within one line nu hardly changes; across an
# aggregate band it does (without the division, radiation-only balance at J = B_nu(T) missed LTE by 0.2-2.7% for OH
# bands, more for wide overtone and continuum bands). Opacity and emissivity (chi, eta) keep the cross-sections
# themselves.

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
        j_prec: npt.NDArray[np.float64],
        wn_grid: npt.NDArray[np.float64],
        wn_dx: npt.NDArray[np.float64],
        abs_profiles: npt.NDArray[np.float64],
        abs_offsets: npt.NDArray[np.int64],
        abs_start_idxs: npt.NDArray[np.int32],
        abs_profile_idx: npt.NDArray[np.int32],
        ste_profiles: npt.NDArray[np.float64],
        ste_offsets: npt.NDArray[np.int64],
        ste_start_idxs: npt.NDArray[np.int32],
        ste_profile_idx: npt.NDArray[np.int32],
        spe_profiles: npt.NDArray[np.float64],
        spe_offsets: npt.NDArray[np.int64],
        spe_start_idxs: npt.NDArray[np.int32],
        spe_profile_idx: npt.NDArray[np.int32],
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
        Each row contains the triple (A_fi, B_fi, B_if) for this layer: population-weighted rates per molecule in the
        source aggregate (see :func:`tiramisu.profiles.bands.band_rate_sums`).
    id_agg_cutoff : int
    n_lookup : ndarray, shape (n_agg,)
    chem_scale_factor : float
        Dimensionless fractional abundance of the species at the current layer.
    lambda_layer_grid : ndarray, shape (n_grid,)
        Dimensionless Lambda operator at each grid point in the current layer.
    global_chi : ndarray, shape (n_grid,)
        Global opacity (from all sources) in the current layer [cm^2].
    j_prec : ndarray, shape (n_grid,)
        Preconditioned mean intensity J - Lambda*[eta_species / chi] in the current layer [J/(m^2 sr)] (per Hz), i.e.
        J with this species' approximate local contribution removed. Rates are B j_prec, B per unit mean intensity.
    wn_grid : ndarray, shape (n_grid,)
        Wavenumbers, to turn cross-section profiles into photon-counting weights (see the note above).
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

        # Normalised photon-counting weights [cm] (cross-sections / nu, see "Photon-counting weights" above).
        abs_profile_norm = loglinear_normalise_1d_nonnegative(
            y_data=abs_profile / wn_grid[abs_grid_start_t:abs_grid_end_t],
            dx=wn_dx[abs_grid_start_t:abs_grid_end_t - 1],
        )
        ste_profile_norm = loglinear_normalise_1d_nonnegative(
            y_data=ste_profile / wn_grid[ste_grid_start_t:ste_grid_end_t],
            dx=wn_dx[ste_grid_start_t:ste_grid_end_t - 1],
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

            # Emission is stored in [erg.cm/(s.sr)]; normalised as photon-counting weights (see the note above).
            spe_profile = spe_profiles[spe_offset_start_t:spe_offset_end_t]
            spe_profile_norm = loglinear_normalise_1d_nonnegative(
                y_data=spe_profile / wn_grid[spe_grid_start_t:spe_grid_end_t],
                dx=wn_dx[spe_grid_start_t:spe_grid_end_t - 1],
            )

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

        # Integrands are [cm.J/(m^2 sr)]
        ste_integrand = np.empty(n_ste)
        for i in range(n_ste):
            ste_integrand[i] = ste_profile_norm[i] * j_prec[ste_grid_start_t + i]
        # Integral is [J/(m^2 sr)], product with B [m^2 sr/(J s)] is hence [s^-1].
        v_fi_prec = loglinear_integral_1d(y_data=ste_integrand, dx=wn_dx[ste_grid_start_t:ste_grid_end_t - 1]) * b_fi

        abs_integrand = np.empty(n_abs)
        for i in range(n_abs):
            abs_integrand[i] = abs_profile_norm[i] * j_prec[abs_grid_start_t + i]
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
        j_prec: npt.NDArray[np.float64],
        wn_grid: npt.NDArray[np.float64],
        wn_dx: npt.NDArray[np.float64],
        abs_profiles: npt.NDArray[np.float64],
        abs_offsets: npt.NDArray[np.int64],
        abs_start_idxs: npt.NDArray[np.int32],
        abs_profile_idx: npt.NDArray[np.int32],
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
        Each row contains the triple (A_fi, B_fi, B_if) for this layer: population-weighted rates per molecule in the
        source aggregate (see :func:`tiramisu.profiles.bands.band_rate_sums`).
    id_agg_cutoff : int
    n_lookup : ndarray, shape (n_agg,)
    chem_scale_factor : float
        Dimensionless fractional abundance of the species at the current layer.
    lambda_layer_grid : ndarray, shape (n_grid,)
        Dimensionless Lambda operator at each grid point in the current layer.
    j_prec : ndarray, shape (n_grid,)
        Preconditioned mean intensity J - Lambda*[eta_species / chi] in the current layer [J/(m^2 sr)] (per Hz), i.e.
        J with this species' approximate local contribution removed. Rates are B j_prec, B per unit mean intensity.
    wn_grid : ndarray, shape (n_grid,)
        Wavenumbers, to turn cross-section profiles into photon-counting weights (see the note above).
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

        # Normalised photon-counting weights [cm] (cross-sections / nu, see "Photon-counting weights" above).
        abs_profile_norm = loglinear_normalise_1d_nonnegative(
            y_data=abs_profile / wn_grid[abs_grid_start_t:abs_grid_end_t],
            dx=wn_dx[abs_grid_start_t:abs_grid_end_t - 1],
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
        # Integrands are [cm.J/(m^2 sr)]
        abs_integrand = np.empty(n_abs)
        for i in range(n_abs):
            abs_integrand[i] = abs_profile_norm[i] * j_prec[abs_grid_start_t + i]
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


def boltzmann_population(states: pl.DataFrame, temperature: u.Quantity) -> pl.DataFrame:
    # TODO: This should be updated use implementations as elsewhere.
    g_np = states["g"].to_numpy()
    energy_np = states["energy"].to_numpy() << 1 / u.cm

    q_lev_np = g_np * np.exp(-ac_h_c_on_kB * energy_np / temperature).value
    n_np = q_lev_np / np.sum(q_lev_np)

    # Per-aggregate-state sums with .over(): a group_by(...).agg(sum) sums in a thread-dependent order, so its results
    # differ between runs in the last bits.
    return states.with_columns(pl.Series("q_lev", q_lev_np), pl.Series("n", n_np)).with_columns(
        pl.col("n").sum().over("id_agg").alias("n_agg")
    )


_T = t.TypeVar("_T")


def _rates_log_table(columns: t.Dict[str, npt.NDArray[np.int32]], rates: npt.NDArray[np.float64]) -> str:
    """Rates as a table for the log: the key ``columns`` per row, and A_fi, B_fi, B_if each as an array over NLTE
    layers (``rates`` is (n_nlte_layers, n_rows, 3))."""
    table = pl.DataFrame({**columns, **{name: rates[:, :, k].T for k, name in enumerate(("A_fi", "B_fi", "B_if"))}})
    with pl.Config(tbl_rows=1000):
        return str(table)


class _Required(t.Generic[_T]):
    """
    A :class:`NLTEProcessor` field set during setup rather than at initialisation, stored in the slot ``_<name>``:
    reading it before it is set raises a RuntimeError naming the field instead of returning None.
    """

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name
        self.slot = "_" + name

    def __get__(self, obj: t.Any, objtype: type | None = None) -> _T:
        if obj is None:
            return self
        value = getattr(obj, self.slot)
        if value is None:
            raise RuntimeError(f"{obj.species} NLTEProcessor field '{self.name}' not initialised.")
        return value

    def __set__(self, obj: t.Any, value: _T) -> None:
        setattr(obj, self.slot, value)


def _as_path_list(files: str | pathlib.Path | t.List[str | pathlib.Path]) -> t.List[pathlib.Path]:
    """One path or a list of paths, as a list of pathlib.Path."""
    return [pathlib.Path(f) for f in (files if isinstance(files, list) else [files])]


class NLTEProcessor:
    """Handles all NLTE-specific functionality."""

    __slots__ = [
        # Public:
        "species", "states_file", "trans_files", "agg_col_nums", "agg_col_names", "species_mass", "reduced_mass",
        "broadening_params", "cont_states_file", "cont_trans_files", "cont_box_length", "cont_broad_col_num",
        "dissociation_products", "do_super_lines", "cont_do_super_lines", "do_initial_step", "debug",
        "debug_pop_matrix", "save_rates_profiles", "rates_pickle", "profile_pickle", "cont_rates_pickle",
        "cont_profile_pickle",
        # Private:
        "_states", "_n_agg_states", "_agg_states", "_id_agg_cutoff", "_rate_keys", "_rates", "_profile_store",
        "_pop_matrix", "_mol_chi_matrix", "_mol_eta_matrix", "_cont_states", "_cont_rate_ids", "_cont_rates",
        "_cont_profile_store", "_nlte_pop_frac",
        "_agg_lookup_cache", "_a_ox_vals", "_col_chem_c_matrix", "_col_chem_rhs_c", "_n_layers", "_n_lte_layers",
        "_y_matrix", "_y_reduced_idx_map", "_rhs_matrix",
        "_profile_idx_cache", "_cont_profile_idx_cache",
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
            do_initial_step: bool = True,
            debug: bool = False,
            debug_pop_matrix: npt.NDArray[np.float64] | None = None,
            save_rates_profiles: bool = False,
            rates_pickle: pathlib.Path | None = None,
            profile_pickle: pathlib.Path | None = None,
            cont_rates_pickle: pathlib.Path | None = None,
            cont_profile_pickle: pathlib.Path | None = None,
    ):
        self.species = SpeciesFormula(species)

        self.states_file: pathlib.Path = pathlib.Path(states_file)
        if not self.states_file.is_file():
            raise RuntimeError(f"{self.species} states file not found at {self.states_file}.")

        self.trans_files: t.List[pathlib.Path] = _as_path_list(trans_files)
        for trans_file in self.trans_files:
            if not trans_file.is_file():
                raise RuntimeError(f"{self.species} trans file not found at {trans_file}.")

        self.agg_col_nums: t.List[int] = agg_col_nums
        self.agg_col_names: t.List[str] = ["agg" + str(idx + 1) for idx in range(0, len(self.agg_col_nums))]

        # Isotopologue masses (most abundant isotopes unless the formula names them, e.g. "[13C]O"), matching the
        # line list, rather than molmass's natural-abundance average.
        self.species_mass: float = self.species.isotope.mass
        # Only meaningful for a diatomic, as is the continuum box broadening that uses it (see
        # SpeciesFormula.reduced_mass).
        self.reduced_mass: float = self.species.reduced_mass
        self.broadening_params = broadening_params

        # Required, set during runtime/setup.
        self._n_layers: int | None = None
        self._n_lte_layers: int | None = None
        self._states: pl.DataFrame | None = None
        self._n_agg_states: int | None = None
        self._agg_states: pl.DataFrame | None = None
        self._id_agg_cutoff: int | None = None
        self._rate_keys: npt.NDArray[np.int32] | None = None
        self._rates: npt.NDArray[np.float64] | None = None
        self._profile_store: ProfileStore | None = None
        self._mol_chi_matrix: u.Quantity | None = None
        self._mol_eta_matrix: u.Quantity | None = None
        self._pop_matrix: npt.NDArray[np.float64] | None = None
        self._y_matrix: npt.NDArray[np.float64] | None = None
        self._y_reduced_idx_map: npt.NDArray[np.int32] | None = None
        self._rhs_matrix: npt.NDArray[np.float64] | None = None

        # Continuum properties.
        self.cont_states_file: pathlib.Path | None = None if cont_states_file is None else pathlib.Path(
            cont_states_file)
        if self.cont_states_file and not self.cont_states_file.is_file():
            raise RuntimeError(f"{self.species} continuum states file not found at {self.cont_states_file}.")
        self.cont_trans_files: t.List[pathlib.Path] | None = (
            None if cont_trans_files is None else _as_path_list(cont_trans_files)
        )
        if self.cont_trans_files:
            for cont_trans_file in self.cont_trans_files:
                if not cont_trans_file.is_file():
                    raise RuntimeError(f"{self.species} continuum trans file not found at {cont_trans_file}.")
        self.cont_box_length: float | None = cont_box_length
        self.cont_broad_col_num: int | None = cont_broad_col_num
        self._cont_states: pl.DataFrame | None = None
        # Continuum (photo-absorption) rates per lower aggregate, None when no continuum transition is on the grid.
        self._cont_rate_ids: npt.NDArray[np.int32] | None = None
        self._cont_rates: npt.NDArray[np.float64] | None = None
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
        self.do_initial_step = do_initial_step
        self.debug: bool = debug
        self.debug_pop_matrix: npt.NDArray[np.float64] | None = debug_pop_matrix
        self.save_rates_profiles = save_rates_profiles
        self.profile_pickle = profile_pickle
        self.rates_pickle = rates_pickle
        self.cont_profile_pickle = cont_profile_pickle
        self.cont_rates_pickle = cont_rates_pickle
        self._agg_lookup_cache = None
        self._nlte_pop_frac = None
        self._a_ox_vals = None
        self._col_chem_c_matrix = None
        self._col_chem_rhs_c = None
        # Layer/iteration-invariant profile indices used by build_y_matrix, populated lazily on first use since they
        # depend only on profile_store's per-layer key_lookup, which does not change across NLTE iterations.
        self._profile_idx_cache = {}
        self._cont_profile_idx_cache = {}

    # Fields set during setup rather than at initialisation: reading one before it is set raises.
    n_layers = _Required[int]()
    n_lte_layers = _Required[int]()
    n_agg_states = _Required[int]()
    id_agg_cutoff = _Required[int]()
    states = _Required[pl.DataFrame]()
    agg_states = _Required[pl.DataFrame]()
    # Radiative rates between aggregates: band keys (n_rates, 2) as (id_agg_f, id_agg_i), sorted by id_agg_i then
    # id_agg_f, and per NLTE layer the population-weighted A_fi, B_fi, B_if (n_nlte_layers, n_rates, 3), see
    # profiles.bands.band_rate_sums. Layer-major, so each layer's slice is contiguous for _build_y_matrix_core.
    rate_keys = _Required[npt.NDArray[np.int32]]()
    rates = _Required[npt.NDArray[np.float64]]()
    profile_store = _Required[ProfileStore]()
    pop_matrix = _Required[npt.NDArray[np.float64]]()
    mol_chi_matrix = _Required[u.Quantity]()
    mol_eta_matrix = _Required[u.Quantity]()
    y_matrix = _Required[u.Quantity]()
    y_reduced_idx_map = _Required[npt.NDArray[np.int32]]()
    rhs_matrix = _Required[u.Quantity]()

    # Methods.

    def get_latest_pop_grid(self) -> npt.NDArray[np.float64]:
        return self.pop_matrix[-1]

    def set_pop_matrix_to_debug(self) -> None:
        if self.debug_pop_matrix is None:
            raise RuntimeError(f"Attempting to set debug pop matrix for {self.species} but argument not set.")
        if len(self.debug_pop_matrix.shape) == 2:
            self._pop_matrix = self.debug_pop_matrix[None, :, :]
        elif len(self.debug_pop_matrix.shape) == 3:
            self._pop_matrix = self.debug_pop_matrix
        else:
            raise RuntimeError(f"Invalid debug pop matrix dimensions {self.debug_pop_matrix.shape}.")

    def _set_nlte_state_columns(self) -> None:
        """
        Per-state NLTE populations ``n_nlte_L{layer}`` of the NLTE layers from the latest aggregate populations, for the
        final cross-sections (:func:`abs_emi_xsec`, :func:`continuum_xsec`): each state's LTE population scaled by its
        aggregate's NLTE/LTE ratio. States of an empty LTE aggregate are 0; states above the aggregation cutoff stay in
        LTE (their aggregates are fixed at LTE).
        """
        id_agg = self.states["id_agg"].to_numpy()
        above_cutoff = id_agg > self.id_agg_cutoff
        columns = []
        for layer_idx in range(self.n_lte_layers, self.n_layers):
            n_lte = self.states[f"n_L{layer_idx}"].to_numpy()
            n_agg_lte = self.states[f"n_agg_L{layer_idx}"].to_numpy()  # LTE even with a debug pop matrix.
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = n_lte * self.pop_matrix[-1, layer_idx, id_agg] / n_agg_lte
            columns.append(pl.Series(
                f"n_nlte_L{layer_idx}", np.where(n_agg_lte == 0, 0.0, np.where(above_cutoff, n_lte, ratio))
            ))
        self.states = self.states.with_columns(columns)

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

        # )
        if log.isEnabledFor(_LOG_VERBOSE_1):
            log.log(_LOG_VERBOSE_1, f"{self.species} aggregated states =\n {self.agg_states}")

        self.states = self.states.join(
            self.agg_states.select(
                ["id_agg"] + self.agg_col_names
            ),
            on=self.agg_col_names,
            how="left",
        )
        self.states = self.states.with_columns(
            pl.col("id_agg").fill_null(overflow_id)
        ).drop(null_check_col)

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
        if log.isEnabledFor(_LOG_VERBOSE_1):
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
            rates = pickle.load(open(self.rates_pickle, "rb"))
            assert isinstance(rates, dict) and rates["rates"].shape[0] == n_nlte_layers, (
                f"{self.rates_pickle} predates the per-layer population-weighted rate arrays (2026-10-06); "
                f"regenerate it."
            )
            self.rate_keys, self.rates = rates["keys"], rates["rates"]
            assert self.profile_store.n_layers == n_nlte_layers
            assert len(self.profile_store.abs_profiles) == n_nlte_layers
            assert len(self.profile_store.ste_profiles) == n_nlte_layers
            assert len(self.profile_store.spe_profiles) == n_nlte_layers
            log.info(f"ProfileStore and rates grid loaded from disk for {self.species}.")
            # Any additional debugging metrics here?
            return

        self.profile_store = ProfileStore(n_layers=n_nlte_layers)

        # Plain float arrays — no astropy units — passed directly to Numba.
        temperature_slice = temperature_profile[self.n_lte_layers:].to_value(u.K)  # (n_nlte_layers,)
        pressure_slice = pressure_profile[self.n_lte_layers:].to_value(u.bar)  # (n_nlte_layers,)

        states_frac = self.states.with_columns(
            (
                    pl.col(f"n_L{self.n_lte_layers + nlte_idx}") / pl.col(f"n_agg_L{self.n_lte_layers + nlte_idx}")
            ).alias(f"n_frac_nL{nlte_idx}")
            for nlte_idx in range(n_nlte_layers)
        )

        # Grid parameters and buffers.
        n_grid = wn_grid.shape[0]
        # Bin edges for finding trans bins.
        wn_arr = wn_grid.value
        wn_min = wn_arr[0]
        wn_max = wn_arr[-1]
        bin_edges = grid_bin_edges(wn_arr)

        # States lookup - much more memory efficient than duplicating it all with polars joins!
        state_ids = states_frac["id"].to_numpy()
        max_state_id = int(state_ids.max())
        # State ID lookup offset by 1 as IDs are 1-indexed! NaN energy marks IDs missing from the states table.
        energy_lookup = np.full(max_state_id + 1, np.nan, dtype=np.float64)
        energy_lookup[state_ids] = states_frac["energy"].to_numpy()
        id_agg_lookup = np.zeros(max_state_id + 1, dtype=np.int32)
        id_agg_lookup[state_ids] = states_frac["id_agg"].to_numpy()
        # State-major, so a state's populations in every layer are contiguous.
        n_frac_lookup = np.zeros((max_state_id + 1, n_nlte_layers), dtype=np.float64)
        n_frac_lookup[state_ids] = states_frac.select(
            f"n_frac_nL{nlte_idx}" for nlte_idx in range(n_nlte_layers)
        ).to_numpy()

        g_lookup = np.zeros(max_state_id + 1, dtype=np.float64)
        g_lookup[state_ids] = states_frac["g"].to_numpy()
        inv_g_lookup = np.zeros_like(g_lookup)
        zero_mask = g_lookup == 0.0
        inv_g_lookup[~zero_mask] = 1.0 / g_lookup[~zero_mask]

        tau_lookup = np.zeros(max_state_id + 1, dtype=np.float64)
        tau_lookup[state_ids] = states_frac["tau"].to_numpy()
        gamma_lifetime_lookup = gamma_lifetime_lookup_from_tau(tau_lookup)

        # Expand out broadening parameters.
        broad_n = np.zeros(1, dtype=np.float64)
        broad_gamma = np.zeros((1, n_nlte_layers), dtype=np.float64)
        if self.broadening_params is not None:
            broad_n = self.broadening_params[1]
            # Contiguous copy of the NLTE-layer slice: the GPU kernels copy it to the device as a single buffer.
            broad_gamma = np.ascontiguousarray(self.broadening_params[0][:, self.n_lte_layers:])

        # Before the batch loop, pre-allocate once.
        n_bands_max = min(100, self.n_agg_states ** 2)

        # profile_buffer[0] = abs, profile_buffer[1] = ste, profile_buffer[2] = spe. Super-lines also accumulate the
        # absorption-weighted lifetime HWHM in profile_buffer[3].
        n_channels = 4 if self.do_super_lines else 3
        band_lookup, id_agg_offset, band_keys = _init_band_registry(states_frac["id_agg"].to_numpy())
        inv_doppler, gamma_pressure = layer_broadening_terms(
            temperatures=temperature_slice, pressures=pressure_slice, broad_n=broad_n, broad_gamma=broad_gamma,
            species_mass=self.species_mass, t_ref=296.0, pressure_ref=1.0,
        )

        # The GPU keeps the profile buffer on the device for the whole batch loop (profiles.gpu.DeviceProfileBuffer), so
        # each batch only uploads its transitions. Its fixed costs (context, lookup upload, one buffer copy back) only pay
        # off for large line lists (profiles.gpu.gpu_pays_off).
        device_buffer = None
        profile_buffer = None
        if config.USE_GPU and (self.do_super_lines or config.GPU_DIRECT_PROFILES):
            from .profiles import gpu
            # Upper bound on the buffer copied back at the end: every pair of aggregate levels forming a band.
            if gpu.gpu_pays_off(
                    trans_files=self.trans_files, n_layers=n_nlte_layers,
                    buffer_bytes=n_channels * self.n_agg_states ** 2 * n_nlte_layers * n_grid * 8,
                    label=str(self.species),
            ):
                try:
                    device_buffer = gpu.DeviceProfileBuffer(
                        n_channels=n_channels, n_bands=n_bands_max, wn_grid=wn_arr, bin_edges=bin_edges,
                        n_frac_lookup=n_frac_lookup, g_lookup=g_lookup, inv_g_lookup=inv_g_lookup,
                        gamma_lifetime_lookup=gamma_lifetime_lookup, inv_doppler=inv_doppler,
                        gamma_pressure=gamma_pressure,
                    )
                except MemoryError as err:
                    log.warning(f"{self.species}: {err} Using the CPU.")
        if device_buffer is None:
            profile_buffer = np.zeros((n_channels, n_bands_max, n_nlte_layers, n_grid), dtype=np.float64)
        # Per-band, per-layer population-weighted sums of A_fi, B_fi and B_if (see band_rate_sums).
        rate_sums = np.zeros((0, n_nlte_layers, 3), dtype=np.float64)

        process_time = time.perf_counter()
        backend = "GPU" if device_buffer is not None else "CPU"
        accumulate_time = 0.0
        # New
        for trans_file in self.trans_files:
            log.info(f"Processing file {trans_file}.")
            for id_f, id_i, a_fi, energy_fi in read_ahead(batches=iter_trans_batches(
                    trans_file=trans_file,
                    energy_lookup=energy_lookup,
                    wn_min=wn_min,
                    wn_max=wn_max,
            )):
                # Assign new band indices to any new bands; extend the profile buffer if needed.
                n_bands_prev = band_keys.shape[0]
                band_keys, band_indices = _update_band_registry(
                    id_f=id_f,
                    id_i=id_i,
                    id_agg_lookup=id_agg_lookup,
                    band_lookup=band_lookup,
                    id_agg_offset=id_agg_offset,
                    band_keys=band_keys,
                )
                n_bands_used = band_keys.shape[0]
                # Do rates for batch - same regardless of line processing strategy. Before the direct-profile path
                # regroups the batch arrays (band_indices keeps the batch order).
                rate_sums = np.pad(
                    rate_sums, ((0, n_bands_used - rate_sums.shape[0]), (0, 0), (0, 0))
                ) + band_rate_sums(
                    band_indices=band_indices, n_bands=n_bands_used, id_f=id_f, id_i=id_i, a_fi=a_fi,
                    energy_fi=energy_fi, g_lookup=g_lookup, inv_g_lookup=inv_g_lookup, n_frac_lookup=n_frac_lookup,
                )
                if device_buffer is not None and not device_buffer.ensure_capacity(n_bands_used):
                    log.warning(
                        f"{self.species}: profile buffer for {n_bands_used} bands no longer fits in GPU memory; "
                        f"continuing on the CPU."
                    )
                    profile_buffer = device_buffer.to_host(n_bands_prev)
                    device_buffer.free()
                    device_buffer = None
                if device_buffer is None:
                    profile_buffer = _grow_band_buffer(profile_buffer=profile_buffer, n_bands=n_bands_used)

                t_accumulate = time.perf_counter()
                if self.do_super_lines:
                    if device_buffer is not None:
                        device_buffer.accumulate_superlines(
                            band_indices=band_indices, id_f=id_f, id_i=id_i, a_fi=a_fi, energy_fi=energy_fi,
                        )
                    else:
                        accumulate_superline_band_batch(
                            profile_buffer=profile_buffer,
                            band_indices=band_indices,
                            bin_edges=bin_edges,
                            n_frac_lookup=n_frac_lookup,
                            g_lookup=g_lookup,
                            inv_g_lookup=inv_g_lookup,
                            gamma_lifetime_lookup=gamma_lifetime_lookup,
                            id_f=id_f,
                            id_i=id_i,
                            a_fi=a_fi,
                            energy_fi=energy_fi,
                            n_workers=_DEFAULT_NUM_THREADS,
                        )
                else:
                    # Make each band's transitions contiguous for the per-(band, layer) kernel.
                    id_f, id_i, a_fi, energy_fi, group_bands, group_starts, group_ends = group_by_band(
                        band_indices=band_indices, n_bands=n_bands_used, id_f=id_f, id_i=id_i, a_fi=a_fi,
                        energy_fi=energy_fi,
                    )
                    if device_buffer is not None:
                        device_buffer.sampled_voigt(
                            id_f=id_f, id_i=id_i, group_bands=group_bands, group_starts=group_starts,
                            group_ends=group_ends, a_fi=a_fi, energy_fi=energy_fi,
                        )
                    else:
                        band_profile_sampled_voigt(
                            profile_buffer=profile_buffer,
                            wn_grid=wn_arr,
                            id_f=id_f,
                            id_i=id_i,
                            group_bands=group_bands,
                            group_starts=group_starts,
                            group_ends=group_ends,
                            n_lookup=n_frac_lookup,
                            g_lookup=g_lookup,
                            inv_g_lookup=inv_g_lookup,
                            gamma_lifetime_lookup=gamma_lifetime_lookup,
                            a_fi=a_fi,
                            energy_fi=energy_fi,
                            temperatures=temperature_slice,
                            pressures=pressure_slice,
                            broad_n=broad_n,
                            broad_gamma=broad_gamma,
                            species_mass=self.species_mass,
                            n_workers=_DEFAULT_NUM_THREADS,
                        )
                accumulate_time += time.perf_counter() - t_accumulate
        log.info(f"[{backend}] compute_rates_profiles: cumulative accumulate time = {accumulate_time:.3f}s.")
        # Contract the buffer based on n_bands_used, drop superfluous rows.
        n_bands_used = band_keys.shape[0]
        broadened = False
        if device_buffer is not None:
            t_broaden = time.perf_counter()
            if self.do_super_lines and config.GPU_BROADEN_SUPERLINES:
                profile_buffer = device_buffer.broaden_superlines(n_bands_used)
                broadened = True
            else:
                profile_buffer = device_buffer.to_host(n_bands_used)
            device_buffer.free()
            log.info(
                f"[GPU] compute_rates_profiles: {'broaden on device and copy' if broadened else 'copy buffer to host'} "
                f"{time.perf_counter() - t_broaden:.3f}s."
            )
        else:
            profile_buffer = profile_buffer[:, :n_bands_used, :, :]
        if self.do_super_lines and not broadened:
            t_broaden = time.perf_counter()
            profile_buffer = broaden_superline_band_buffer(
                profile_buffer=profile_buffer,
                wn_grid=wn_arr,
                temperatures=temperature_slice,
                pressures=pressure_slice,
                broad_n=broad_n,
                broad_gamma=broad_gamma,
                species_mass=self.species_mass,
            )
            log.info(f"[CPU] compute_rates_profiles: broaden time = {time.perf_counter() - t_broaden:.3f}s.")
        # Finalise profile store from, profile_buffer accumulator.
        self.profile_store.finalise_from_buffer(
            profile_buffer=profile_buffer,
            band_keys=band_keys,
            save=self.debug,
            species=self.species,
        )
        # Done, finalise list of rates chunks.
        if log.isEnabledFor(_LOG_VERBOSE_2):
            log.log(_LOG_VERBOSE_2,
                    f"{self.species} rates/profiles duration = {time.perf_counter() - process_time:.3f}.")
        # Radiative rates between different aggregate states only, sorted by id_agg_i then id_agg_f; per NLTE layer,
        # rates per molecule in the source aggregate (see band_rate_sums).
        inter_band = band_keys[:, 0] != band_keys[:, 1]
        keys = band_keys[inter_band].astype(np.int32)
        order = np.lexsort((keys[:, 0], keys[:, 1]))
        self.rate_keys = keys[order]
        self.rates = np.ascontiguousarray(rate_sums[inter_band][order].transpose(1, 0, 2))
        if log.isEnabledFor(_LOG_VERBOSE_1):
            table = _rates_log_table(
                columns={"id_agg_f": self.rate_keys[:, 0], "id_agg_i": self.rate_keys[:, 1]}, rates=self.rates,
            )
            log.log(_LOG_VERBOSE_1, f"{self.species} rates = \n{table}")
        # Save bands to disk.
        if self.save_rates_profiles:
            band_file = f"{self.species}_wn{int(wn_min)}-{int(wn_max)}_G{n_grid}_B{n_bands_used}_L{n_nlte_layers}.profilestore.pickle"
            with open((output_dir / band_file).resolve(), "wb") as band_pickle:
                pickle.dump(self.profile_store, band_pickle, protocol=pickle.HIGHEST_PROTOCOL)
            rates_file = f"{self.species}_wn{int(wn_min)}-{int(wn_max)}_G{n_grid}_B{n_bands_used}_L{n_nlte_layers}.ratesgrid.pickle"
            with open((output_dir / rates_file).resolve(), "wb") as rates_pickle:
                pickle.dump(
                    {"keys": self.rate_keys, "rates": self.rates}, rates_pickle, protocol=pickle.HIGHEST_PROTOCOL,
                )

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

        n_nlte_layers = self.n_layers - self.n_lte_layers

        if self.cont_profile_pickle is not None and self.cont_rates_pickle is not None:
            self._cont_profile_store = pickle.load(open(self.cont_profile_pickle, "rb"))
            cont_rates = pickle.load(open(self.cont_rates_pickle, "rb"))
            # None when no continuum transitions fall on the grid.
            assert cont_rates is None or (
                    isinstance(cont_rates, dict) and cont_rates["rates"].shape[0] == n_nlte_layers
            ), (
                f"{self.cont_rates_pickle} predates the per-layer population-weighted rate arrays (2026-10-06); "
                f"regenerate it."
            )
            if cont_rates is not None:
                self._cont_rate_ids, self._cont_rates = cont_rates["ids"], cont_rates["rates"]
            assert self._cont_profile_store.n_layers == n_nlte_layers
            assert len(self._cont_profile_store.abs_profiles) == n_nlte_layers
            return

        if log.isEnabledFor(_LOG_VERBOSE_2):
            log.log(_LOG_VERBOSE_2, f"{self.species} loading continuum absorption rates and profiles.")

        read_col_map = {num: "v" if num == self.cont_broad_col_num else name for num, name in
                        zip(self.agg_col_nums, self.agg_col_names)}
        if self.cont_broad_col_num not in read_col_map:
            read_col_map[self.cont_broad_col_num] = "v"

        extra_col_indices, extra_col_names = (list(x) for x in zip(*sorted(read_col_map.items())))

        read_col_names = ["id", "energy", "g"] + extra_col_names
        read_col_indices = [0, 1, 2] + extra_col_indices
        fixed_dtypes = {"id": "Int32", "energy": np.float64, "g": np.float64, "v": "Int32"}
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

        # # NB: Left join converts ints to float as some may be nan, does not occur for inner join but left needed here to
        # # preserve energy/degeneracy info of upper states with no id_agg map.

        states_frac = self._cont_states.clone()

        for nlte_layer_idx, layer_temp in enumerate(temperature_profile[self.n_lte_layers:]):
            layer_idx = nlte_layer_idx + self.n_lte_layers
            # Precompute boltzmann populations for each layer.
            temp_cont_states = boltzmann_population(self._cont_states, layer_temp)
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

        self._cont_profile_store = ContinuumProfileStore(n_layers=n_nlte_layers)

        # Plain float arrays — no astropy units — passed directly to Numba.
        temperature_slice = temperature_profile[self.n_lte_layers:].to_value(u.K)  # (n_nlte_layers,)

        # Grid parameters and buffers.
        wn_arr = wn_grid.value
        wn_min = wn_arr[0]
        wn_max = wn_arr[-1]
        n_grid = wn_grid.shape[0]
        # Bin edges for finding trans bins.
        bin_edges = grid_bin_edges(wn_arr)

        # States lookup - much more memory efficient than duplicating it all with polars joins!
        state_ids = states_frac["id"].to_numpy()
        max_state_id = int(state_ids.max())
        # State ID lookup offset by 1 as IDs are 1-indexed! NaN energy marks IDs missing from the states table.
        energy_lookup = np.full(max_state_id + 1, np.nan, dtype=np.float64)
        energy_lookup[state_ids] = states_frac["energy"].to_numpy()
        id_agg_lookup = np.zeros(max_state_id + 1, dtype=np.int32)
        id_agg_lookup[state_ids] = states_frac["id_agg"].to_numpy()
        # State-major, so a state's populations in every layer are contiguous.
        n_frac_lookup = np.zeros((max_state_id + 1, n_nlte_layers), dtype=np.float64)
        n_frac_lookup[state_ids] = states_frac.select(
            f"n_frac_nL{nlte_idx}" for nlte_idx in range(n_nlte_layers)
        ).to_numpy()

        g_lookup = np.zeros(max_state_id + 1, dtype=np.float64)
        g_lookup[state_ids] = states_frac["g"].to_numpy()
        inv_g_lookup = np.zeros_like(g_lookup)
        zero_mask = g_lookup == 0.0
        inv_g_lookup[~zero_mask] = 1.0 / g_lookup[~zero_mask]

        v_lookup = np.zeros(max_state_id + 1, dtype=np.float64)
        v_lookup[state_ids] = states_frac["v"].to_numpy()

        # Before the batch loop, pre-allocate once.
        n_bands_max = min(100, self.n_agg_states ** 2)

        # profile_buffer[0] = abs; super-lines also accumulate the broadening in profile_buffer[1].
        n_channels = 2 if self.cont_do_super_lines else 1
        profile_buffer = np.zeros((n_channels, n_bands_max, n_nlte_layers, n_grid), dtype=np.float64)
        band_lookup, id_agg_offset, band_keys = _init_band_registry(states_frac["id_agg"].to_numpy())
        # Per-band, per-layer population-weighted sums of A_fi, B_fi and B_if (see band_rate_sums).
        rate_sums = np.zeros((0, n_nlte_layers, 3), dtype=np.float64)

        process_time = time.perf_counter()

        for cont_trans_file in self.cont_trans_files:
            log.info(f"Processing file {cont_trans_file}.")
            for id_f, id_i, a_fi, energy_fi in read_ahead(batches=iter_trans_batches(
                    trans_file=cont_trans_file,
                    energy_lookup=energy_lookup,
                    wn_min=wn_min,
                    wn_max=wn_max,
            )):
                # Assign new band indices to any new bands; extend profile buffer if needed.
                band_keys, band_indices = _update_band_registry(
                    id_f=id_f,
                    id_i=id_i,
                    id_agg_lookup=id_agg_lookup,
                    band_lookup=band_lookup,
                    id_agg_offset=id_agg_offset,
                    band_keys=band_keys,
                )
                n_bands_used = band_keys.shape[0]
                # Do rates for batch - same regardless of line processing strategy. Before the direct-profile path
                # regroups the batch arrays (band_indices keeps the batch order).
                rate_sums = np.pad(
                    rate_sums, ((0, n_bands_used - rate_sums.shape[0]), (0, 0), (0, 0))
                ) + band_rate_sums(
                    band_indices=band_indices, n_bands=n_bands_used, id_f=id_f, id_i=id_i, a_fi=a_fi,
                    energy_fi=energy_fi, g_lookup=g_lookup, inv_g_lookup=inv_g_lookup, n_frac_lookup=n_frac_lookup,
                )
                profile_buffer = _grow_band_buffer(profile_buffer=profile_buffer, n_bands=n_bands_used)
                if self.cont_do_super_lines:
                    accumulate_continuum_superline_band_batch(
                        profile_buffer=profile_buffer,
                        band_indices=band_indices,
                        bin_edges=bin_edges,
                        n_frac_lookup=n_frac_lookup,
                        g_lookup=g_lookup,
                        inv_g_lookup=inv_g_lookup,
                        v_lookup=v_lookup,
                        id_f=id_f,
                        id_i=id_i,
                        a_fi=a_fi,
                        energy_fi=energy_fi,
                        reduced_mass=self.reduced_mass,
                        box_length=self.cont_box_length,
                        n_workers=_DEFAULT_NUM_THREADS,
                    )
                else:
                    # Make each band's transitions contiguous for the per-(band, layer) kernel.
                    id_f, id_i, a_fi, energy_fi, group_bands, group_starts, group_ends = group_by_band(
                        band_indices=band_indices, n_bands=n_bands_used, id_f=id_f, id_i=id_i, a_fi=a_fi,
                        energy_fi=energy_fi,
                    )
                    continuum_band_profile_sampled_gauss_layered(
                        profile_buffer=profile_buffer,
                        wn_grid=wn_arr,
                        id_f=id_f,
                        id_i=id_i,
                        group_bands=group_bands,
                        group_starts=group_starts,
                        group_ends=group_ends,
                        n_lookup=n_frac_lookup,
                        g_lookup=g_lookup,
                        inv_g_lookup=inv_g_lookup,
                        v_lookup=v_lookup,
                        a_fi=a_fi,
                        energy_fi=energy_fi,
                        temperatures=temperature_slice,
                        species_mass=self.species_mass,
                        reduced_mass=self.reduced_mass,
                        box_length=self.cont_box_length,
                        n_workers=_DEFAULT_NUM_THREADS,
                    )
        # Contract the buffer based on n_bands_used, drop superfluous rows.
        n_bands_used = band_keys.shape[0]
        profile_buffer = profile_buffer[:, :n_bands_used, :, :]
        if self.cont_do_super_lines:
            profile_buffer = broaden_continuum_superline_band_buffer(
                profile_buffer=profile_buffer,
                wn_grid=wn_arr,
                temperatures=temperature_slice,
                species_mass=self.species_mass,
                n_workers=_DEFAULT_NUM_THREADS,
            )
        # Finalise profile store from, profile_buffer accumulator.
        self._cont_profile_store.finalise_from_buffer(profile_buffer=profile_buffer, band_keys=band_keys)
        if log.isEnabledFor(_LOG_VERBOSE_2):
            log.log(_LOG_VERBOSE_2,
                    f"{self.species} cont. rates/profiles duration = {time.perf_counter() - process_time:.3f}.")

        # Radiative rates between different aggregate states only. Each lower aggregate absorbs into many continuum
        # "bands" (one per upper aggregate); only its total photo-absorption enters the rate matrix, so the bands are
        # summed per id_agg_i. Layer-major (n_nlte_layers, n_ids, 3), as for rates.
        inter_band = band_keys[:, 0] != band_keys[:, 1]
        if inter_band.any():
            id_agg_i, band_to_id = np.unique(band_keys[inter_band, 1], return_inverse=True)
            cont_sums = np.zeros((id_agg_i.shape[0], n_nlte_layers, 3), dtype=np.float64)
            np.add.at(cont_sums, band_to_id, rate_sums[inter_band])
            self._cont_rate_ids = id_agg_i.astype(np.int32)
            self._cont_rates = np.ascontiguousarray(cont_sums.transpose(1, 0, 2))
            if log.isEnabledFor(_LOG_VERBOSE_1):
                table = _rates_log_table(columns={"id_agg_i": self._cont_rate_ids}, rates=self._cont_rates)
                log.log(_LOG_VERBOSE_1, f"{self.species} continuum rates = \n{table}")
        else:
            self._cont_rate_ids = self._cont_rates = None
            if log.isEnabledFor(_LOG_VERBOSE_1):
                log.log(_LOG_VERBOSE_1, f"{self.species} No continuum rates computed on spectral grid.")
        if self.save_rates_profiles:
            # Read back by passing these files as cont_profile_pickle / cont_rates_pickle.
            cont_file = f"{self.species}_wn{int(wn_min)}-{int(wn_max)}_G{n_grid}_B{n_bands_used}_L{n_nlte_layers}"
            with open((output_dir / f"{cont_file}.cont.profilestore.pickle").resolve(), "wb") as band_pickle:
                pickle.dump(self._cont_profile_store, band_pickle, protocol=pickle.HIGHEST_PROTOCOL)
            with open((output_dir / f"{cont_file}.cont.ratesgrid.pickle").resolve(), "wb") as rates_pickle:
                cont_rates = (
                    None if self._cont_rates is None else {"ids": self._cont_rate_ids, "rates": self._cont_rates}
                )
                pickle.dump(cont_rates, rates_pickle, protocol=pickle.HIGHEST_PROTOCOL)
        # Done.

    def _build_a_ox_vals_cache(self) -> None:
        """
        Precompute summed Einstein A coefficients per upper aggregated state below the id_agg_cutoff.

        Accumulates each band's A_fi into its upper state index, per NLTE layer. Call once after the rates are
        finalised inside the setup routine.

        Stores
        ------
        self._a_ox_vals : astropy.units.Quantity, shape (n_nlte_layers, id_agg_cutoff + 1)
            Sum of A_fi values for all transitions from each upper state, per NLTE layer.
        """
        id_agg_f = self.rate_keys[:, 0]
        below_cutoff = id_agg_f <= self.id_agg_cutoff
        a_fi = self.rates[:, :, 0].T  # (n_rates, n_nlte_layers)
        a_ox_vals = np.zeros((self.id_agg_cutoff + 1, a_fi.shape[1]), dtype=np.float64)
        np.add.at(a_ox_vals, id_agg_f[below_cutoff], a_fi[below_cutoff])
        self._a_ox_vals = a_ox_vals.T << einstein_a_unit

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

            # Formation rates are per unit volume; divided by this species' number density they are per molecule, as the
            # fractional populations need.
            own_num_dens = (chem_profile[self.species][layer_idx] * density_profile[layer_idx]).to_value(u.cm ** -3)

            # Group by the species the rates depend on to avoid redundant number density lookups
            rates_by_species: t.Dict[t.Tuple[str, ...], t.List[RateTransition]] = {}
            for rate in rates_table:
                rates_by_species.setdefault(rate.species_depend, []).append(rate)

            for species_depend, species_rates in rates_by_species.items():
                if any(species not in chem_profile.species for species in species_depend):
                    continue

                depend_num_dens = 1.0
                for species in species_depend:
                    depend_num_dens *= (
                            chem_profile[SpeciesFormula(species)][layer_idx] * density_profile[layer_idx]
                    ).to_value(u.cm ** -3)

                for rate in species_rates:
                    try:
                        upper_id, _ = self._agg_lookup_cache[rate.upper_key]
                        lower_id, _ = self._agg_lookup_cache[rate.lower_key]
                    except KeyError:
                        continue

                    if upper_id > self.id_agg_cutoff or lower_id > self.id_agg_cutoff:
                        # Short circuit for bands above cutoff; adding to RHS biases to fixed distribution above cutoff.
                        continue

                    c_fi = rate.rate * depend_num_dens
                    # Detailed balance between aggregates: excitation over de-excitation is the ratio of their LTE
                    # populations (Boltzmann sums over their states, pop_matrix[0]), so collisions alone return LTE.
                    # exp(-dE/kT) between their lowest states would ignore each manifold's degeneracy and rotational
                    # partition function (for OH X, b_v ~ 1 - 0.039 v in collision-dominated layers).
                    c_if = c_fi * self.pop_matrix[0, layer_idx, upper_id] / self.pop_matrix[0, layer_idx, lower_id]

                    if lower_id == upper_id:
                        if c_fi > 0:
                            # Chemical formation; independent of species population.
                            if own_num_dens > 0:
                                rhs_c_cache[layer_idx, upper_id] -= c_fi / own_num_dens
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

        vmr_mask = chem_profile[self.species] != 0
        self.mol_chi_matrix = np.zeros_like(initial_chi_matrix)
        self.mol_chi_matrix[vmr_mask] = initial_chi_matrix[vmr_mask] / chem_profile[self.species][vmr_mask, None]
        lte_source_func_matrix = blackbody(
            spectral_grid=wn_grid, temperature=temperature_profile
        )
        self.mol_eta_matrix = lte_source_func_matrix * self.mol_chi_matrix * ac.c
        self._build_a_ox_vals_cache()
        self._build_col_chem_cache(
            chem_profile=chem_profile,
            density_profile=density_profile,
            temperature_profile=temperature_profile,
        )

    def compute_initial_step(
            self,
            j_mean: u.Quantity,
            chem_profile: ChemicalProfile,
            density_profile: u.Quantity,
            temperature_profile: u.Quantity,
            wn_grid: u.Quantity,
            wn_dx: u.Quantity,
    ) -> None:
        """
        Build an initial population guess by solving a cheap statistical-equilibrium matrix per layer - i.e. one
        un-accelerated Lambda-iteration using the input ``j_mean`` (mean intensity J at layer centres, the same
        convention as the ``j_prec`` used by :meth:`build_y_matrix`) as a frozen radiation field - reusing
        :meth:`solve_pops` for the linear algebra, rather than reconstructing populations from a single
        per-layer excitation temperature.

        Unlike a single scalar T_ex, this does not force a Boltzmann shape onto the aggregated manifold: bands are
        assembled into one reduced rate matrix (same row=target/col=source convention as :meth:`build_y_matrix`)
        and solved simultaneously, so overlapping bands are reconciled consistently and population inversions fall
        out naturally (no negative-temperature/abs/clip handling needed - non-negativity is instead guaranteed by
        the same pseudo-inverse/least-squares machinery :meth:`solve_pops` already uses for the real iteration).
        """
        n_dim = self.id_agg_cutoff + 1

        agg_energies = self.agg_states.sort("id_agg")["energy_agg"].to_numpy()
        agg_energies = np.concatenate((agg_energies, [wn_grid[-1].value]), axis=0)
        id_pairs = self.profile_store.get_sorted_band_keys(
            agg_energies=agg_energies, id_cutoff=self.id_agg_cutoff,
        )
        if log.isEnabledFor(_LOG_VERBOSE_3):
            log.log(_LOG_VERBOSE_3, f"{self.species} ID pairs for initial step = {id_pairs}")

        g_np = self.states["g"].to_numpy()
        energy_np = self.states["energy"].to_numpy() << 1 / u.cm
        id_agg_np = self.states["id_agg"].to_numpy()
        hcE_on_kB = ac_h_c_on_kB * energy_np

        q_lte = g_np[None, :] * np.exp(-hcE_on_kB[None, :] / temperature_profile[:, None])  # (n_layers, n_states)
        n_lte = q_lte / q_lte.sum(axis=1, keepdims=True)

        n_agg_lte = np.zeros((self.n_layers, self.n_agg_states), dtype=np.float64)
        np.add.at(n_agg_lte.T, id_agg_np, n_lte.T)

        self._nlte_pop_frac = n_agg_lte[:, :n_dim].sum(axis=1)

        valid_map = np.array([
            l >= self.n_lte_layers and chem_profile[self.species][l] > _VMR_THRESHOLD for l in range(self.n_layers)
        ])

        # Fallback shape for layers we don't solve below (below n_lte_layers or VMR-thresholded): the renormalised
        # LTE distribution within the cutoff manifold.
        nlte_pop_frac_safe = np.where(self._nlte_pop_frac > 0, self._nlte_pop_frac, 1.0)
        n_agg_shape = n_agg_lte[:, :n_dim] / nlte_pop_frac_safe[:, None]

        # Row of each (id_agg_f, id_agg_i) band in rate_keys/rates.
        rate_row = {(int(id_agg_f), int(id_agg_i)): row for row, (id_agg_f, id_agg_i) in enumerate(self.rate_keys)}

        for layer_idx in range(self.n_lte_layers, self.n_layers):
            if not valid_map[layer_idx]:
                continue
            nlte_layer_idx = layer_idx - self.n_lte_layers

            self.y_matrix = np.zeros((n_dim, n_dim), dtype=np.float64) << (1 / u.s)
            self.rhs_matrix = np.zeros(n_dim, dtype=np.float64) << (1 / u.s)

            for id_agg_f, id_agg_i in id_pairs:
                if id_agg_f > self.id_agg_cutoff or id_agg_i > self.id_agg_cutoff:
                    # Bands touching states above the cutoff are not modelled here, matching build_y_matrix's
                    # treatment of those states as a fixed (LTE) reservoir.
                    continue

                a_fi, b_fi, b_if = self.rates[nlte_layer_idx, rate_row[(id_agg_f, id_agg_i)]]
                a_fi, b_fi, b_if = a_fi * einstein_a_unit, b_fi * einstein_b_unit, b_if * einstein_b_unit

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

                # Photon-counting weights (cross-sections / nu, see "Photon-counting weights" at the Y matrix kernels).
                abs_profile_norm = loglinear_normalise_quantity_1d_nonnegative(
                    y_data=abs_profile / wn_grid[abs_start_idx:abs_end_idx], dx=wn_dx[abs_start_idx:abs_end_idx - 1]
                )
                ste_profile_norm = loglinear_normalise_quantity_1d_nonnegative(
                    y_data=ste_profile / wn_grid[ste_start_idx:ste_end_idx], dx=wn_dx[ste_start_idx:ste_end_idx - 1]
                )

                v_fi = loglinear_integral_quantity_1d_nonnegative(
                    y_data=ste_profile_norm * j_mean[layer_idx, ste_start_idx:ste_end_idx],
                    dx=wn_dx[ste_start_idx:ste_end_idx - 1]
                ) * b_fi
                v_if = loglinear_integral_quantity_1d_nonnegative(
                    y_data=abs_profile_norm * j_mean[layer_idx, abs_start_idx:abs_end_idx],
                    dx=wn_dx[abs_start_idx:abs_end_idx - 1]
                ) * b_if

                # Same sign convention as build_y_matrix: row = target state, column = source state.
                self.y_matrix[id_agg_f, id_agg_i] += v_if
                self.y_matrix[id_agg_i, id_agg_f] += a_fi + v_fi
                self.y_matrix[id_agg_f, id_agg_f] -= (a_fi + v_fi)
                self.y_matrix[id_agg_i, id_agg_i] -= v_if

            self.add_col_chem_rates(layer_idx=layer_idx)

            pop_updated, pop_old = self.solve_pops(layer_idx=layer_idx, n_iter=0)
            n_agg_shape[layer_idx] = 0.0
            n_agg_shape[layer_idx, self.y_reduced_idx_map] = pop_updated
            if log.isEnabledFor(_LOG_VERBOSE_2):
                log.log(
                    _LOG_VERBOSE_2,
                    (
                        f"[L{layer_idx}] {self.species} initial-step pops (was LTE {pop_old}) = "
                        f"{np.array2string(pop_updated, formatter=_LOG_ARRAY_FMT)}"
                    )
                )

        if log.isEnabledFor(_LOG_VERBOSE_1):
            log.log(
                _LOG_VERBOSE_1,
                f"{self.species} initial-step aggregated NLTE shape =\n"
                f"{np.array2string(n_agg_shape, formatter=_LOG_ARRAY_FMT)}"
            )

        n_agg_nlte = np.zeros((self.n_layers, self.n_agg_states), dtype=np.float64)
        n_agg_nlte[:, :n_dim] = n_agg_shape
        n_agg_nlte *= self._nlte_pop_frac[:, None]

        n_agg_final = n_agg_lte.copy()
        n_agg_final[:, :n_dim] = n_agg_nlte[:, :n_dim]

        if log.isEnabledFor(_LOG_VERBOSE_2):
            log.log(_LOG_VERBOSE_2, f"{self.species} NLTE pop frac = {self._nlte_pop_frac}.")
        if log.isEnabledFor(_LOG_VERBOSE_2):
            log.log(_LOG_VERBOSE_2, f"{self.species} new pop grid sums = {n_agg_final.sum(axis=1)}")

        # n_agg_final[None, :] has shape (1, n_layers, n_agg_states) as required.
        self.pop_matrix = np.vstack((self.pop_matrix, n_agg_final[None, :]))

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
        a_ox_cross_cache = spe_profiles_norm * self._a_ox_vals[nlte_layer_idx][:, None]
        return a_ox_cross_cache

    def build_y_matrix(
            self,
            layer_idx: int,
            nlte_layer_idx: int,
            j_layer_grid: u.Quantity,
            lambda_layer_grid: npt.NDArray[np.float64],
            chem_profile: ChemicalProfile,
            effective_chi_matrix: u.Quantity,  # u.cm**2 - shielded (always non-negative), see effective_source_tau_mu.
            wn_grid: u.Quantity,
            wn_dx: u.Quantity,
            full_prec: bool,
    ) -> None:
        """
        Build statistical equilibrium matrix.

        ``effective_chi_matrix`` must be the shielded effective opacity returned by the most recent call to
        :func:`effective_source_tau_mu` for the current global chi/eta state (i.e. reflecting this layer's own
        opacity as of just before it is solved this iteration), not the raw ``XSecCollection._global_chi_matrix``:
        dividing by the raw matrix here would reintroduce the masing (negative/near-zero opacity) blow-ups that
        ``effective_source_tau_mu`` already shields the radiative-transfer solve against - see the caller in
        ``XSecCollection.compute_opacities_profile``.
        """
        num_grid = wn_grid.shape[0]

        species_eta: u.Quantity = chem_profile[self.species][layer_idx] * self.mol_eta_matrix[layer_idx] / ac.c
        global_chi: u.Quantity = effective_chi_matrix[layer_idx]  # / density_profile[layer_idx]
        chi_mask = global_chi != 0
        # The Bezier Lambda* (exact diagonal of the discretised Lambda operator) can be negative even with a strictly
        # positive tau: the control points' dependence on S_l carries negative weights where S varies sharply between
        # optically thin layers. Clip it once here, so that the removed local emission (psi_approx_eta, old
        # populations) and the one added back (psi_approx_cross / self_prec, new populations) use the same Lambda*
        # and cancel at convergence: the clip then only changes the preconditioning, not the converged populations.
        lambda_layer_grid = np.clip(lambda_layer_grid, 0, 1)
        psi_approx_eta = np.zeros(num_grid) << j_layer_grid.unit
        psi_approx_eta[chi_mask] = (
                lambda_layer_grid[chi_mask] * species_eta[chi_mask] / global_chi[chi_mask]
        )
        # Preconditioned mean intensity (Rybicki & Hummer 1991): J with this species' approximate local contribution,
        # Lambda*[eta_old / chi], removed. Rates are B J_prec, with B per unit mean intensity (see einstein_b_unit).
        j_prec: u.Quantity = j_layer_grid - psi_approx_eta
        j_prec_negative_count = np.count_nonzero(j_prec.value < 0)
        # Saturate, don't rescale: unlike effective_chi's masing shielding (where the magnitude of a negative value
        # is physically meaningful - population inversion strength - and remapping to a smaller positive analog
        # preserves that shape), a negative j_prec here just means this species' own local-emission estimate
        # (psi_approx_eta) overshot the total mean intensity J - there's no meaningful shape to preserve by
        # magnitude, so we saturate to 0 outright rather than rescaling.
        j_prec = np.clip(j_prec, 0, None)
        if j_prec_negative_count:
            if log.isEnabledFor(_LOG_VERBOSE_1):
                log.log(
                    _LOG_VERBOSE_1,
                    f"[L{layer_idx}] {self.species}: {j_prec_negative_count}/{num_grid} J_prec entries were "
                    f"negative before clamping."
                )

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
            psi_approx_cross[:, chi_mask] = (
                    lambda_layer_grid[chi_mask] * a_ox_cross_cache[:, chi_mask] / global_chi[chi_mask]
            )

        if self.debug:
            layer_rates = self.rates[nlte_layer_idx].T.tolist()
            for trans_row in zip(self.rate_keys[:, 0].tolist(), self.rate_keys[:, 1].tolist(), *layer_rates):
                # 0 = id_agg_f, 1 = id_agg_i, 2 = A_fi, 3 = B_fi, 4 = B_if (this layer's).
                if trans_row[0] > self.id_agg_cutoff or trans_row[1] > self.id_agg_cutoff:
                    # Short-circuit for bands involving states above cutoff; these pops are fixed and including them on
                    # RHS biases towards fixed distribution above cutoff.
                    continue
                a_fi = trans_row[2] * einstein_a_unit
                b_fi = trans_row[3] * einstein_b_unit
                b_if = trans_row[4] * einstein_b_unit
                if log.isEnabledFor(_LOG_VERBOSE_2):
                    log.log(
                        _LOG_VERBOSE_2,
                        f"[L{layer_idx}] Trans: {trans_row[0:2]}: {[f'{val:6.3E}' for val in trans_row[2:]]}."
                    )

                # These are pop-normalised within the band, but redundant due to normalisation.
                abs_profile, abs_start_idx = self.profile_store.get_profile(
                    layer_idx=nlte_layer_idx, key=(trans_row[0], trans_row[1]), profile_type="abs"
                )
                abs_profile: u.Quantity = abs_profile << u.cm ** 2
                ste_profile, ste_start_idx = self.profile_store.get_profile(
                    layer_idx=nlte_layer_idx, key=(trans_row[0], trans_row[1]), profile_type="ste"
                )
                ste_profile: u.Quantity = ste_profile << u.cm ** 2

                abs_end_idx = abs_start_idx + len(abs_profile)
                ste_end_idx = ste_start_idx + len(ste_profile)

                # Normalised photon-counting weights with units [cm] (cross-sections / nu, see "Photon-counting
                # weights" at the Y matrix kernels).
                abs_profile_norm = loglinear_normalise_quantity_1d_nonnegative(
                    y_data=abs_profile / wn_grid[abs_start_idx:abs_end_idx], dx=wn_dx[abs_start_idx:abs_end_idx - 1]
                )
                ste_profile_norm = loglinear_normalise_quantity_1d_nonnegative(
                    y_data=ste_profile / wn_grid[ste_start_idx:ste_end_idx], dx=wn_dx[ste_start_idx:ste_end_idx - 1]
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
                        layer_idx=nlte_layer_idx, key=(trans_row[0], trans_row[1]), profile_type="spe"
                    )
                    spe_profile = spe_profile << u.erg * u.cm / (u.s * u.sr)
                    spe_end_idx = spe_start_idx + len(spe_profile)
                    spe_profile_norm = loglinear_normalise_1d_nonnegative(
                        y_data=(spe_profile / wn_grid[spe_start_idx:spe_end_idx]).value,
                        dx=wn_dx[spe_start_idx:spe_end_idx - 1].value,
                    )
                    self_prec = np.zeros(num_grid)
                    self_prec[chi_mask] = (
                            lambda_layer_grid[chi_mask]
                            * chi_if[chi_mask]
                            # * ac.h
                            / global_chi[chi_mask]
                    )
                    self_prec = self_prec[spe_start_idx:spe_end_idx] * spe_profile_norm
                    self_prec = loglinear_integral_1d(
                        y_data=self_prec, dx=wn_dx[spe_start_idx:spe_end_idx - 1].value
                    )
                    u_fi *= 1 - self_prec
                # End cross.
                if log.isEnabledFor(_LOG_VERBOSE_2):
                    log.log(
                        _LOG_VERBOSE_2, f"[L{layer_idx}] U_{trans_row[0], trans_row[1]} = {u_fi:{_LOG_FLOAT_FMT}}"
                    )
                v_fi_prec = loglinear_integral_quantity_1d(
                    y_data=ste_profile_norm * j_prec[ste_start_idx: ste_end_idx],
                    dx=wn_dx[ste_start_idx: ste_end_idx - 1]
                ) * b_fi
                v_fi_prec = v_fi_prec.decompose()
                if log.isEnabledFor(_LOG_VERBOSE_2):
                    log.log(
                        _LOG_VERBOSE_2,
                        f"[L{layer_idx}] V_{trans_row[0], trans_row[1]}_prec = {v_fi_prec:{_LOG_FLOAT_FMT}}"
                    )

                v_if_prec = loglinear_integral_quantity_1d(
                    y_data=abs_profile_norm * j_prec[abs_start_idx: abs_end_idx],
                    dx=wn_dx[abs_start_idx: abs_end_idx - 1]
                ) * b_if
                v_if_prec = v_if_prec.decompose()
                if log.isEnabledFor(_LOG_VERBOSE_2):
                    log.log(
                        _LOG_VERBOSE_2,
                        f"[L{layer_idx}] V_{trans_row[1], trans_row[0]}_prec = {v_if_prec:{_LOG_FLOAT_FMT}}"
                    )

                self.y_matrix[trans_row[0], trans_row[1]] += v_if_prec
                self.y_matrix[trans_row[1], trans_row[0]] += u_fi + v_fi_prec
                self.y_matrix[trans_row[0], trans_row[0]] -= u_fi + v_fi_prec
                self.y_matrix[trans_row[1], trans_row[1]] -= v_if_prec

            if self._cont_rates is not None:
                # 0 = id_agg_i, 1 = A_fi, 2 = B_fi, 3 = B_if (this layer's).
                layer_cont_rates = self._cont_rates[nlte_layer_idx].T.tolist()
                for cont_trans_row in zip(self._cont_rate_ids.tolist(), *layer_cont_rates):
                    if cont_trans_row[0] > self.id_agg_cutoff:
                        # Short-circuit for bands involving states above cutoff; these pops are fixed and including them
                        # on RHS biases towards fixed distribution above cutoff.
                        continue

                    a_ci = cont_trans_row[1] * einstein_a_unit
                    b_ic = cont_trans_row[3] * einstein_b_unit

                    log.debug(f"{self.species}: Cont. profile for state {cont_trans_row[0]}.")
                    cont_abs_profile, cont_abs_start_idx = self._cont_profile_store.get_profile(
                        layer_idx=nlte_layer_idx, key=cont_trans_row[0], profile_type="abs"
                    )
                    cont_abs_profile: u.Quantity = cont_abs_profile << u.cm ** 2
                    cont_abs_end_idx = cont_abs_start_idx + len(cont_abs_profile)

                    # Photon-counting weights (cross-sections / nu, see "Photon-counting weights" at the Y matrix
                    # kernels).
                    cont_abs_profile_norm = loglinear_normalise_quantity_1d_nonnegative(
                        y_data=cont_abs_profile / wn_grid[cont_abs_start_idx:cont_abs_end_idx],
                        dx=wn_dx[cont_abs_start_idx:cont_abs_end_idx - 1],
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

                        nonzero_integral_mask = psi_integrals.value != 0

                        # Photo-absorption out of level i of locally emitted photons: a loss from i proportional to
                        # each emitting level's population, as in _build_y_matrix_cont (the bound-bound cross terms
                        # likewise subtract from the lower level).
                        self.y_matrix[cont_trans_row[0], nonzero_integral_mask] -= psi_integrals[nonzero_integral_mask]
                    # End cross.
                    v_ic_prec = loglinear_integral_quantity_1d(
                        y_data=cont_abs_profile_norm * j_prec[cont_abs_start_idx: cont_abs_end_idx],
                        dx=wn_dx[cont_abs_start_idx: cont_abs_end_idx - 1]
                    ) * b_ic
                    v_ic_prec = v_ic_prec.decompose()
                    if log.isEnabledFor(_LOG_VERBOSE_2):
                        log.log(
                            _LOG_VERBOSE_2,
                            f"[L{layer_idx}] V_{cont_trans_row[0]}c_prec = {v_ic_prec:{_LOG_FLOAT_FMT}}"
                        )

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
            id_agg_f, id_agg_i = self.rate_keys[:, 0], self.rate_keys[:, 1]
            rates_grid_arr = self.rates[nlte_layer_idx]

            n_rates = id_agg_f.shape[0]

            abs_store = self.profile_store.abs_profiles[nlte_layer_idx]
            ste_store = self.profile_store.ste_profiles[nlte_layer_idx]
            spe_store = self.profile_store.spe_profiles[nlte_layer_idx]

            if nlte_layer_idx not in self._profile_idx_cache:
                # abs_store/ste_store/spe_store.key_lookup are fixed per layer once profiles are built, so this
                # index lookup is identical across all NLTE iterations for a given layer - cache per layer.
                abs_profile_idx = np.zeros(n_rates, dtype=np.int32)
                ste_profile_idx = np.zeros(n_rates, dtype=np.int32)
                spe_profile_idx = np.zeros(n_rates, dtype=np.int32)

                for r in range(n_rates):
                    if id_agg_f[r] > self.id_agg_cutoff or id_agg_i[r] > self.id_agg_cutoff:
                        continue
                    key = (int(id_agg_f[r]), int(id_agg_i[r]))
                    abs_profile_idx[r] = abs_store.key_lookup[key]
                    ste_profile_idx[r] = ste_store.key_lookup[key]
                    spe_profile_idx[r] = spe_store.key_lookup[key]
                self._profile_idx_cache[nlte_layer_idx] = (abs_profile_idx, ste_profile_idx, spe_profile_idx)

            abs_profile_idx, ste_profile_idx, spe_profile_idx = self._profile_idx_cache[nlte_layer_idx]
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
                j_prec=j_prec.to_value(u.J / (u.m ** 2 * u.sr)),
                wn_grid=wn_grid.value,
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
                id_agg_i, rates_grid_arr = self._cont_rate_ids, self._cont_rates[nlte_layer_idx]

                n_rates = id_agg_i.shape[0]

                abs_store = self._cont_profile_store.abs_profiles[nlte_layer_idx]

                if nlte_layer_idx not in self._cont_profile_idx_cache:
                    abs_profile_idx = np.zeros(n_rates, dtype=np.int32)

                    for r in range(n_rates):
                        if id_agg_i[r] > self.id_agg_cutoff:
                            continue
                        # TODO: Check -1 key carried forward.
                        key = (-1, int(id_agg_i[r]))
                        abs_profile_idx[r] = abs_store.key_lookup[key]
                    self._cont_profile_idx_cache[nlte_layer_idx] = abs_profile_idx

                abs_profile_idx = self._cont_profile_idx_cache[nlte_layer_idx]

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
                    j_prec=j_prec.to_value(u.J / (u.m ** 2 * u.sr)),
                    wn_grid=wn_grid.value,
                    wn_dx=wn_dx.value,
                    abs_profiles=abs_store.profiles,
                    abs_offsets=abs_store.offsets,
                    abs_start_idxs=abs_store.start_idxs,
                    abs_profile_idx=abs_profile_idx,
                    limiting_species_num_dens=limiting_species_num_dens,
                    full_prec=full_prec,
                    psi_approx_cross=psi_approx_cross_val,
                    y_matrix=y_matrix_val,
                )
        # Add collisional and chemical rates.
        self.add_col_chem_rates(layer_idx=layer_idx)
        if log.isEnabledFor(_LOG_VERBOSE_2):
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
        self.y_reduced_idx_map = np.nonzero(np.abs(self.y_matrix).sum(axis=1) != 0)[0].astype(np.int32)
        y_matrix_reduced = self.y_matrix[np.ix_(self.y_reduced_idx_map, self.y_reduced_idx_map)].value
        if log.isEnabledFor(_LOG_VERBOSE_3):
            log.log(
                _LOG_VERBOSE_3,
                (
                    f"[L{layer_idx}] {self.species} Y matrix (before row-normalisation) =\n"
                    f"{np.array2string(y_matrix_reduced, formatter=_LOG_ARRAY_FMT)}\n"
                    f"[L{layer_idx}] {self.species} Y matrix cond. (before row-normalisation) ="
                    f" {np.linalg.cond(y_matrix_reduced):{_LOG_FLOAT_FMT}}"
                )
            )
        norm_factors = abs(y_matrix_reduced).sum(axis=1)[:, None]
        y_matrix_reduced /= norm_factors
        check_rows = np.all(y_matrix_reduced > 0, axis=1) | np.all(y_matrix_reduced < 0, axis=1)
        if np.any(check_rows):
            log.error(
                f"[I{n_iter}][L{layer_idx}] Y matrix all same sign in rows "
                f"{np.nonzero(check_rows)[0]}; investigate unphysical rates."
            )

        y_rect = np.vstack([y_matrix_reduced.copy(), np.ones(y_matrix_reduced.shape[1])])
        rhs_rect = self.rhs_matrix[self.y_reduced_idx_map].value / norm_factors[:, 0]
        rhs_rect = np.append(rhs_rect, 1)

        if log.isEnabledFor(_LOG_VERBOSE_2):
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
                np.zeros(y_rect.shape[1]),
                bounds=(0.0, 1.0),
                method="trf",
                ftol=1e-15,
                gtol=1e-15,
                xtol=1e-15,
            )
            least_squares_pops = lsq_res.x
            least_squares_pops /= least_squares_pops.sum()
            if log.isEnabledFor(_LOG_VERBOSE_2):
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

        # Store all updated pops, any zeros that were excldued from the Y-matrix calculation and pops above cutoff.
        pop_updated_full = np.zeros(self.n_agg_states)
        pop_updated_full[self.y_reduced_idx_map] = pop_scaled
        pop_updated_full[self.id_agg_cutoff + 1:] = self.pop_matrix[-1, layer_idx, self.id_agg_cutoff + 1:]  # TEST!

        # Per-state populations are only needed by the final cross-sections, built in finalise.
        n_agg_lte, n_agg_old = self.pop_matrix[0, layer_idx], self.pop_matrix[-1, layer_idx]
        if log.isEnabledFor(_LOG_VERBOSE_2):
            rows = "\n".join(
                f"{id_agg:>6d}" + "".join(f"  {format(n, _LOG_FLOAT_FMT):>14s}" for n in pops)
                for id_agg, pops in enumerate(zip(n_agg_lte, n_agg_old, pop_updated_full))
            )
            log.log(
                _LOG_VERBOSE_2,
                f"[L{layer_idx}] {self.species} aggregate populations (old: previous iteration):\n"
                f"{'id_agg':>6s}  {'n_agg_lte':>14s}  {'n_agg_nlte old':>14s}  {'n_agg_nlte new':>14s}\n{rows}"
            )
        if log.isEnabledFor(_LOG_VERBOSE_3):
            log.log(
                _LOG_VERBOSE_3,
                (
                    f"[L{layer_idx}] Sum of LTE populations = {n_agg_lte.sum()}.\n"
                    f"[L{layer_idx}] Sum of non-LTE populations = {pop_updated_full.sum()}."
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
            # Use for T_ex approximation when all new populations stored internally.
            layer_pop_grid = self.pop_matrix[-1, layer_idx]

        start_time = time.perf_counter()
        abs_xsec, emi_xsec = self.profile_store.build_abs_emi(
            layer_idx=nlte_layer_idx, pop_matrix=layer_pop_grid, wn_grid=wn_grid.value,
        )
        if log.isEnabledFor(_LOG_VERBOSE_2):
            log.log(_LOG_VERBOSE_2,
                    f"[L{layer_idx}] {self.species} CBP duration = {time.perf_counter() - start_time:.5f}")

        if np.any(abs_xsec < 0):
            log.warning(
                f"[L{layer_idx}] {self.species} Negative contribution in absorption."
            )

        if self._cont_states is not None and self.cont_trans_files is not None:
            cont_abs_xsec = self._cont_profile_store.build_abs(
                layer_idx=nlte_layer_idx, pop_matrix=layer_pop_grid, wn_grid=wn_grid.value,
            )
            abs_xsec += cont_abs_xsec

        # Update layer chi globally and then for species, shielded before it is summed (see MASER_SHIELD_FACTOR);
        # mol_chi_matrix keeps the shielded value, so the next update removes exactly what was added.
        abs_xsec = shield_masing_chi(abs_xsec) << u.cm ** 2
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
        if self.debug:
            # The population history so far, rewritten every iteration.
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
        self._set_nlte_state_columns()
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
                wn_grid=wn_grid,
                species_mass=self.species_mass,
                reduced_mass=self.reduced_mass,
                cont_box_length=self.cont_box_length,
                do_super_lines=self.cont_do_super_lines,
            )
            abs_xsec += cont_xsec
        # Shielded as during the iteration (see MASER_SHIELD_FACTOR), so the spectra use the same treatment.
        abs_xsec = shield_masing_chi(abs_xsec) << u.cm ** 2
        self.mol_chi_matrix[self.n_lte_layers:] = abs_xsec
        emi_xsec = emi_xsec << u.erg * u.cm / (u.s * u.sr)
        self.mol_eta_matrix[self.n_lte_layers:] = emi_xsec

        with open((output_dir / f"{self.species}_abs_xsec.pickle").resolve(), "wb") as abs_pickle_file:
            pickle.dump(self.mol_chi_matrix, abs_pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        with open((output_dir / f"{self.species}_emi_xsec.pickle").resolve(), "wb") as emi_pickle_file:
            pickle.dump(self.mol_eta_matrix, emi_pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
