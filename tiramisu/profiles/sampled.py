"""Direct (per-line) sampled Voigt and Gaussian profiles on the wavenumber grid, without super-lines: per band
for NLTE, or straight into cross-sections."""
import math

import numba
import numpy as np
from numpy import typing as npt

from ..config import _DEFAULT_NUM_THREADS
from ..constants import const_h_on_8_c_amu, const_8_pi_c, const_h_c_on_4_pi, const_sqrt_2_NA_kB_log2_on_c
from .lineshape import layer_broadening_terms, binary_search_left, binary_search_right, voigt_humlicek_w


@numba.njit(parallel=True, nogil=True, cache=True, error_model="numpy")
def band_profile_sampled_voigt(
        profile_buffer: npt.NDArray[np.float64],
        wn_grid: npt.NDArray[np.float64],
        id_f: npt.NDArray[np.int32],
        id_i: npt.NDArray[np.int32],
        group_bands: npt.NDArray[np.int32],
        group_starts: npt.NDArray[np.int32],
        group_ends: npt.NDArray[np.int32],
        n_lookup: npt.NDArray[np.float64],
        g_lookup: npt.NDArray[np.float64],
        inv_g_lookup: npt.NDArray[np.float64],
        gamma_lifetime_lookup: npt.NDArray[np.float64],
        a_fi: npt.NDArray[np.float64],
        energy_fi: npt.NDArray[np.float64],
        temperatures: npt.NDArray[np.float64],
        pressures: npt.NDArray[np.float64],
        broad_n: npt.NDArray[np.float64],
        broad_gamma: npt.NDArray[np.float64],
        species_mass: float,
        t_ref: float = 296.0,
        pressure_ref: float = 1.0,
        n_workers: int = _DEFAULT_NUM_THREADS,
) -> None:
    """
    Sampled multi-layer Voigt cross-section using the Humlíček 4-region approximation for the Faddeeva function.

    Evaluates the Voigt profile at each wn_grid point directly rather than integrating over bins. This is faster per
    grid point than the binned version (no quadrature loop) but less accurate for coarse grids where the bin width is
    comparable to the line width - in that regime the binned version is preferred. For high-resolution grids (resolving
    power, R >> line width / grid spacing) the two should converge.

    Can be used for wn_grid input with variable or fixed grid spacing, as profiles are compute at grid points and are
    agnostic of distance to adjacent points (no integral limits).

    The Lorentzian HWHM is the pressure HWHM plus the lifetime HWHM of both the upper and lower states.

    Parameters
    ----------
    profile_buffer : ndarray, shape (3, n_bands, n_layers, n_grid)
    wn_grid : ndarray, shape (n_grid,)
    id_f : ndarray, shape (n_trans,)
    id_i : ndarray, shape (n_trans,)
    group_bands : ndarray, shape (n_bands_in_batch,)
        Global band index of each band in the batch, from :func:`group_by_band`.
    group_starts : ndarray, shape (n_bands_in_batch,)
    group_ends : ndarray, shape (n_bands_in_batch,)
        Each band's transitions are [group_starts[b], group_ends[b]) in the transition arrays.
    n_lookup : ndarray, shape (n_states + 1, n_layers)
        Lookup table for n_frac, avoiding extremely memory intensive joins.
    g_lookup : ndarray, shape (n_states + 1, )
        Lookup table for g, avoiding extremely memory intensive joins.
    inv_g_lookup : ndarray, shape (n_states + 1, )
    gamma_lifetime_lookup : ndarray, shape (n_states + 1, )
        Per-state lifetime HWHM contributions [cm^-1], from :func:`gamma_lifetime_lookup_from_tau`.
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
    n_bands_in_batch = group_starts.shape[0]

    cutoff = 25.0
    inv_sqrt_pi = 1 / np.sqrt(np.pi)

    # Per-layer broadening terms: 1 / (sigma_D * sqrt(2)) = inv_doppler[l] / energy_fi, and the pressure HWHM. The
    # per-transition terms are separable from these, so no (n_layers, n_trans) arrays are needed.
    inv_doppler, gamma_pressure = layer_broadening_terms(
        temperatures=temperatures, pressures=pressures, broad_n=broad_n, broad_gamma=broad_gamma,
        species_mass=species_mass, t_ref=t_ref, pressure_ref=pressure_ref,
    )

    # Each (band, layer) pair owns its output row, so pairs run in parallel without write conflicts. The pairs are
    # sorted by cost (lines in the band) and dealt round-robin to n_workers workers, heaviest first: a band holding most
    # of the lines is then spread over many threads, instead of all its layers landing in one thread's contiguous block
    # of a prange over pairs.
    n_pairs = n_bands_in_batch * n_layers
    cost = np.empty(n_pairs, dtype=np.int64)
    for bl in range(n_pairs):
        b_local = bl // n_layers
        cost[bl] = group_ends[b_local] - group_starts[b_local]
    order = np.argsort(-cost, kind="mergesort")

    for w in numba.prange(n_workers):
        for k in range(w, n_pairs, n_workers):
            bl = order[k]
            b_local = bl // n_layers
            l = bl % n_layers

            b_global = group_bands[b_local]
            gs = group_starts[b_local]
            ge = group_ends[b_local]

            abs_bl = profile_buffer[0, b_global, l]
            ste_bl = profile_buffer[1, b_global, l]
            spe_bl = profile_buffer[2, b_global, l]
            inv_doppler_l = inv_doppler[l]
            gamma_pressure_l = gamma_pressure[l]

            for t in range(gs, ge):
                energy_fi_t = energy_fi[t]
                transition_min = energy_fi_t - cutoff
                transition_max = energy_fi_t + cutoff
                j_start = max(0, binary_search_left(wn_grid, transition_min))
                j_end = min(n_grid, binary_search_right(wn_grid, transition_max) + 1)

                if j_start >= j_end:
                    continue

                id_f_t = id_f[t]
                id_i_t = id_i[t]
                a_fi_t = a_fi[t]
                # Layer-independent coefficient prefactors.
                ste_prefactor_t = a_fi_t / (const_8_pi_c * energy_fi_t * energy_fi_t)
                abs_prefactor_t = ste_prefactor_t * g_lookup[id_f_t] * inv_g_lookup[id_i_t]
                spe_prefactor_t = a_fi_t * energy_fi_t * const_h_c_on_4_pi

                gamma_total_tl = gamma_lifetime_lookup[id_f_t] + gamma_lifetime_lookup[id_i_t] + gamma_pressure_l
                inv_sigma_sqrt2_tl = inv_doppler_l / energy_fi_t
                # Voigt y-parameter: gamma_L / (sigma_D * sqrt(2)).
                y_tl = gamma_total_tl * inv_sigma_sqrt2_tl

                n_f_tl = n_lookup[id_f_t, l]
                abs_tl = n_lookup[id_i_t, l] * abs_prefactor_t
                ste_tl = n_f_tl * ste_prefactor_t
                spe_tl = n_f_tl * spe_prefactor_t

                # Integral of Re[w(x,y)] dx = sqrt(pi), so the normalised Voigt is Re[w(z)] / (sigma * sqrt(2*pi)).
                norm = inv_sigma_sqrt2_tl * inv_sqrt_pi

                for j in range(j_start, j_end):
                    # x = (nu - nu0) / (sigma * sqrt(2))
                    x_tj = (wn_grid[j] - energy_fi_t) * inv_sigma_sqrt2_tl
                    voigt_val = voigt_humlicek_w(x_tj, y_tl) * norm

                    abs_bl[j] += abs_tl * voigt_val
                    ste_bl[j] += ste_tl * voigt_val
                    spe_bl[j] += spe_tl * voigt_val


@numba.njit(parallel=True, nogil=True, cache=True, error_model="numpy")
def abs_emi_sampled_voigt(
        profile_buffer: npt.NDArray[np.float64],
        wn_grid: npt.NDArray[np.float64],
        id_f: npt.NDArray[np.int32],
        id_i: npt.NDArray[np.int32],
        n_lookup: npt.NDArray[np.float64],
        g_lookup: npt.NDArray[np.float64],
        inv_g_lookup: npt.NDArray[np.float64],
        gamma_lifetime_lookup: npt.NDArray[np.float64],
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
    profile_buffer : ndarray, shape (2, n_layers, n_grid)
    wn_grid : ndarray, shape (n_grid,)
    id_f : ndarray, shape (n_trans,)
    id_i : ndarray, shape (n_trans,)
    n_lookup : ndarray, shape (n_states + 1, n_layers)
        Lookup table for n_frac, avoiding extremely memory intensive joins.
    g_lookup : ndarray, shape (n_states + 1, )
        Lookup table for g, avoiding extremely memory intensive joins.
    inv_g_lookup : ndarray, shape (n_states + 1, )
    gamma_lifetime_lookup : ndarray, shape (n_states + 1, )
        Per-state lifetime HWHM contributions [cm^-1], from :func:`gamma_lifetime_lookup_from_tau`. The Lorentzian
        HWHM is the pressure HWHM plus the lifetime HWHM of both the upper and lower states.
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

    cutoff = 25.0
    inv_sqrt_pi = 1 / np.sqrt(np.pi)

    # Per-layer broadening terms: 1 / (sigma_D * sqrt(2)) = inv_doppler[l] / energy_fi, and the pressure HWHM; the
    # per-transition lifetime HWHM is added inside the layer loop, so no (n_layers, n_trans) arrays are needed.
    inv_doppler, gamma_pressure = layer_broadening_terms(
        temperatures=temperatures, pressures=pressures, broad_n=broad_n, broad_gamma=broad_gamma,
        species_mass=species_mass, t_ref=t_ref, pressure_ref=pressure_ref,
    )

    abs_prefactor = np.empty(n_trans, dtype=np.float64)
    emi_prefactor = np.empty(n_trans, dtype=np.float64)
    j_starts = np.empty(n_trans, dtype=np.int32)
    j_ends = np.empty(n_trans, dtype=np.int32)
    for t in numba.prange(n_trans):
        energy_fi_t = energy_fi[t]
        j_starts[t] = max(0, binary_search_left(wn_grid, energy_fi_t - cutoff))
        j_ends[t] = min(n_grid, binary_search_right(wn_grid, energy_fi_t + cutoff) + 1)
        a_fi_t = a_fi[t]
        abs_prefactor[t] = a_fi_t / (const_8_pi_c * energy_fi_t * energy_fi_t)
        emi_prefactor[t] = a_fi_t * energy_fi_t * const_h_c_on_4_pi

    for l in numba.prange(n_layers):
        abs_l = profile_buffer[0, l]
        emi_l = profile_buffer[1, l]

        for t in range(n_trans):
            j_start = j_starts[t]
            j_end = j_ends[t]
            if j_start >= j_end:
                continue

            energy_fi_t = energy_fi[t]
            id_f_t = id_f[t]
            id_i_t = id_i[t]
            g_f_t = g_lookup[id_f_t]
            inv_g_i_t = inv_g_lookup[id_i_t]
            g_f_on_g_i_t = g_f_t * inv_g_i_t
            # Layer-independent coeficient prefactors.
            abs_prefactor_t = abs_prefactor[t]
            emi_prefactor_t = emi_prefactor[t]

            gamma_total_tl = gamma_lifetime_lookup[id_f_t] + gamma_lifetime_lookup[id_i_t] + gamma_pressure[l]
            inv_sigma_sqrt2_tl = inv_doppler[l] / energy_fi_t
            y_tl = gamma_total_tl * inv_sigma_sqrt2_tl
            # Integral of Re[w(x,y)] dx = sqrt(pi), so the normalised Voigt is Re[w(z)] / (sigma * sqrt(2*pi)).
            norm = inv_sigma_sqrt2_tl * inv_sqrt_pi

            n_f_l = n_lookup[id_f_t, l]
            n_i_l = n_lookup[id_i_t, l]
            abs_tl = abs_prefactor_t * ((n_i_l * g_f_on_g_i_t) - n_f_l)
            emi_tl = emi_prefactor_t * n_f_l

            for j in range(j_start, j_end):
                # x = (nu - nu0) / (sigma * sqrt(2))
                wn_j = wn_grid[j]
                x_tj = (wn_j - energy_fi_t) * inv_sigma_sqrt2_tl
                voigt_val = voigt_humlicek_w(x_tj, y_tl) * norm

                abs_l[j] += abs_tl * voigt_val
                emi_l[j] += emi_tl * voigt_val


@numba.njit(parallel=True, nogil=True, cache=True, error_model="numpy")
def continuum_band_profile_sampled_gauss_layered(
        profile_buffer: npt.NDArray[np.float64],
        wn_grid: npt.NDArray[np.float64],
        id_f: npt.NDArray[np.int32],
        id_i: npt.NDArray[np.int32],
        group_bands: npt.NDArray[np.int32],
        group_starts: npt.NDArray[np.int32],
        group_ends: npt.NDArray[np.int32],
        n_lookup: npt.NDArray[np.float64],
        g_lookup: npt.NDArray[np.float64],
        inv_g_lookup: npt.NDArray[np.float64],
        v_lookup: npt.NDArray[np.float64],
        a_fi: npt.NDArray[np.float64],
        energy_fi: npt.NDArray[np.float64],
        temperatures: npt.NDArray[np.float64],
        species_mass: float,
        reduced_mass: float,
        box_length: float,
        n_workers: int = _DEFAULT_NUM_THREADS,
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
    group_bands : ndarray, shape (n_bands_in_batch,)
        Global band index of each band in the batch, from :func:`group_by_band`.
    group_starts : ndarray, shape (n_bands_in_batch,)
    group_ends : ndarray, shape (n_bands_in_batch,)
        Each band's transitions are [group_starts[b], group_ends[b]) in the transition arrays.
    n_lookup : ndarray, shape (n_states + 1, n_layers)
        Lookup table for n_frac, avoiding extremely memory intensive joins.
    g_lookup : ndarray, shape (n_states + 1, )
        Lookup table for g, avoiding extremely memory intensive joins.
    inv_g_lookup : ndarray, shape (n_states + 1, )
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
    max_cutoff = 5000.0
    cutoff_fwhm_multiple = 5.0

    inv_mass = 1 / species_mass

    alpha_box_lookup = np.empty(n_states, dtype=np.float64)
    alpha_box_prefactor = const_h_on_8_c_amu / (box_length * box_length * reduced_mass)
    for s in numba.prange(n_states):
        alpha_box_lookup[s] = alpha_box_prefactor * (2.0 * v_lookup[s] + 1.0)

    doppler_prefactor = np.empty((n_layers,), dtype=np.float64)

    doppler_coef = const_sqrt_2_NA_kB_log2_on_c * math.sqrt(inv_mass)
    for l in numba.prange(n_layers):
        doppler_prefactor[l] = doppler_coef * math.sqrt(temperatures[l])

    # alpha_doppler_max per transition (layer-independent bound).
    abs_prefactor = a_fi / (const_8_pi_c * energy_fi * energy_fi)

    # Each (band, layer) pair owns its output row, so pairs run in parallel without write conflicts. The pairs are
    # sorted by cost (lines in the band) and dealt round-robin to n_workers workers, heaviest first: a band holding most
    # of the lines is then spread over many threads, instead of all its layers landing in one thread's contiguous block
    # of a prange over pairs.
    n_pairs = n_bands_in_batch * n_layers
    cost = np.empty(n_pairs, dtype=np.int64)
    for bl in range(n_pairs):
        b_local = bl // n_layers
        cost[bl] = group_ends[b_local] - group_starts[b_local]
    order = np.argsort(-cost, kind="mergesort")

    for w in numba.prange(n_workers):
        for k in range(w, n_pairs, n_workers):
            bl = order[k]
            b_local = bl // n_layers
            l = bl % n_layers

            b_global = group_bands[b_local]
            gs = group_starts[b_local]
            ge = group_ends[b_local]

            abs_bl = profile_buffer[0, b_global, l]
            doppler_prefactor_l = doppler_prefactor[l]

            for t in range(gs, ge):
                energy_fi_t = energy_fi[t]
                id_f_t = id_f[t]
                # Broadening.
                alpha_box_t = alpha_box_lookup[id_f_t]
                alpha_doppler_tl = energy_fi_t * doppler_prefactor_l
                alpha_total_tl = alpha_box_t + alpha_doppler_tl
                cutoff_tl = cutoff_fwhm_multiple * alpha_total_tl
                if cutoff_tl < min_cutoff:
                    cutoff_tl = min_cutoff
                elif cutoff_tl > max_cutoff:
                    cutoff_tl = max_cutoff

                transition_min = energy_fi_t - cutoff_tl
                transition_max = energy_fi_t + cutoff_tl

                j_start = max(0, binary_search_left(wn_grid, transition_min))
                j_end = min(n_grid, binary_search_right(wn_grid, transition_max) + 1)

                if j_start >= j_end:
                    continue

                id_i_t = id_i[t]
                g_f_t = g_lookup[id_f_t]
                inv_g_i_t = inv_g_lookup[id_i_t]
                g_f_on_g_i_t = g_f_t * inv_g_i_t
                # Layer-independent coeficient prefactors.
                abs_prefactor_t = abs_prefactor[t]
                n_i_tl = n_lookup[id_i_t, l]
                abs_tl = abs_prefactor_t * (n_i_tl * g_f_on_g_i_t)

                sqrtln2_on_alpha_tl = sqrtln2 / alpha_total_tl
                # Normalised Gaussian: G(nu) = (sqrt(ln2) / (alpha * sqrt(pi))) * exp(-ln2 * ((nu - nu0) / alpha)^2)
                # Rewritten with sqrtln2_on_alpha precomputed:
                #   exponent arg  = sqrtln2_on_alpha * (nu - nu0)
                #   prefactor     = sqrtln2_on_alpha / sqrt(pi)
                norm = sqrtln2_on_alpha_tl * inv_sqrt_pi

                for j in range(j_start, j_end):
                    arg = sqrtln2_on_alpha_tl * (wn_grid[j] - energy_fi_t)
                    gauss_val = norm * math.exp(-(arg * arg))

                    abs_bl[j] += abs_tl * gauss_val


@numba.njit(parallel=True, nogil=True, cache=True, error_model="numpy")
def continuum_sampled_gauss(
        profile_buffer: npt.NDArray[np.float64],
        wn_grid: npt.NDArray[np.float64],
        id_f: npt.NDArray[np.int32],
        id_i: npt.NDArray[np.int32],
        n_lookup: npt.NDArray[np.float64],
        g_lookup: npt.NDArray[np.float64],
        inv_g_lookup: npt.NDArray[np.float64],
        v_lookup: npt.NDArray[np.float64],
        a_fi: npt.NDArray[np.float64],
        energy_fi: npt.NDArray[np.float64],
        temperatures: npt.NDArray[np.float64],
        species_mass: float,
        reduced_mass: float,
        box_length: float,
) -> None:
    """

    Parameters
    ----------
    profile_buffer : ndarray, shape (2, n_layers, n_grid)
    wn_grid : ndarray, shape (n_grid,)
    id_f : ndarray, shape (n_trans,)
    id_i : ndarray, shape (n_trans,)
    n_lookup : ndarray, shape (n_states + 1, n_layers)
        Lookup table for n.
    g_lookup : ndarray, shape (n_states + 1, )
        Lookup table for g.
    inv_g_lookup : ndarray, shape (n_states + 1, )
    v_lookup : ndarray, shape (n_states + 1, )
        Lookup table for v, for box broadening.
    a_fi : ndarray, shape (n_trans,)
    energy_fi : ndarray, shape (n_trans,)
    temperatures : ndarray, shape (n_layers,)
    species_mass : float
    reduced_mass : float
    box_length : float
    """
    n_layers = temperatures.shape[0]
    n_grid = wn_grid.shape[0]
    n_trans = energy_fi.shape[0]
    n_states = n_lookup.shape[0]

    sqrtln2 = np.sqrt(np.log(2.0))
    inv_sqrt_pi = 1 / np.sqrt(np.pi)

    min_cutoff = 25.0
    max_cutoff = 5000.0
    cutoff_fwhm_multiple = 5.0

    inv_mass = 1 / species_mass

    alpha_box_lookup = np.empty(n_states, dtype=np.float64)
    alpha_box_prefactor = const_h_on_8_c_amu / (box_length * box_length * reduced_mass)
    for s in numba.prange(n_states):
        alpha_box_lookup[s] = alpha_box_prefactor * (2.0 * v_lookup[s] + 1.0)

    doppler_prefactor = np.empty((n_layers,), dtype=np.float64)

    doppler_coef = const_sqrt_2_NA_kB_log2_on_c * math.sqrt(inv_mass)
    for l in numba.prange(n_layers):
        doppler_prefactor[l] = doppler_coef * math.sqrt(temperatures[l])

    # alpha_doppler_max per transition (layer-independent bound).
    abs_prefactor = a_fi / (const_8_pi_c * energy_fi * energy_fi)

    for l in numba.prange(n_layers):
        abs_l = profile_buffer[0, l]
        doppler_prefactor_l = doppler_prefactor[l]

        for t in range(n_trans):
            energy_fi_t = energy_fi[t]
            id_f_t = id_f[t]
            # Broadening.
            alpha_box_t = alpha_box_lookup[id_f_t]
            alpha_doppler_tl = energy_fi_t * doppler_prefactor_l
            alpha_total_tl = alpha_box_t + alpha_doppler_tl
            cutoff_tl = cutoff_fwhm_multiple * alpha_total_tl
            if cutoff_tl < min_cutoff:
                cutoff_tl = min_cutoff
            elif cutoff_tl > max_cutoff:
                cutoff_tl = max_cutoff

            transition_min = energy_fi_t - cutoff_tl
            transition_max = energy_fi_t + cutoff_tl

            j_start = max(0, binary_search_left(wn_grid, transition_min))
            j_end = min(n_grid, binary_search_right(wn_grid, transition_max) + 1)

            if j_start >= j_end:
                continue

            id_i_t = id_i[t]
            g_f_t = g_lookup[id_f_t]
            inv_g_i_t = inv_g_lookup[id_i_t]
            g_f_on_g_i_t = g_f_t * inv_g_i_t
            # Layer-independent coeficient prefactors.
            abs_prefactor_t = abs_prefactor[t]
            n_i_tl = n_lookup[id_i_t, l]
            abs_tl = abs_prefactor_t * (n_i_tl * g_f_on_g_i_t)

            sqrtln2_on_alpha_tl = sqrtln2 / alpha_total_tl
            norm = sqrtln2_on_alpha_tl * inv_sqrt_pi

            for j in range(j_start, j_end):
                arg = sqrtln2_on_alpha_tl * (wn_grid[j] - energy_fi_t)
                gauss_val = norm * math.exp(-(arg * arg))

                abs_l[j] += abs_tl * gauss_val
