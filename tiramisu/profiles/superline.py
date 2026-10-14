"""Super-line scheme: accumulate transitions into wavenumber bins (per band for NLTE, or per layer for
cross-sections), then broaden the binned buffer with Voigt (lines) or Gaussian (continuum) profiles."""
import math
import typing as t

import numba
import numpy as np
from numpy import typing as npt

from ..config import _DEFAULT_NUM_THREADS
from ..constants import const_h_on_8_c_amu, const_8_pi_c, const_h_c_on_4_pi, const_sqrt_2_NA_kB_log2_on_c
from .lineshape import (
    profile_windows, layer_broadening_terms, binary_search_left, binary_search_right, voigt_humlicek_w,
)
from .bands import band_layer_work_order


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _super_line_bins(bin_edges: npt.NDArray[np.float64], energy_fi: npt.NDArray[np.float64]) -> npt.NDArray[np.int32]:
    """Grid bin of each transition's super-line, clamped to the grid."""
    n_trans = energy_fi.shape[0]
    max_idx = bin_edges.shape[0] - 2
    bins = np.empty(n_trans, dtype=np.int32)
    for t in numba.prange(n_trans):
        bin_idx = binary_search_right(bin_edges, energy_fi[t]) - 1
        if bin_idx < 0:
            bin_idx = 0
        elif bin_idx > max_idx:
            bin_idx = max_idx
        bins[t] = bin_idx
    return bins


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _counting_sort_order(
        keys: npt.NDArray[np.int32], n_keys: int,
) -> t.Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """
    Stable parallel counting sort of keys in [0, n_keys), returned as a permutation plus the groups of equal keys.

    Returns
    -------
    order : ndarray, shape (n,)
        keys[order] is sorted; equal keys keep their input order.
    group_keys, group_starts, group_ends : ndarray, shape (n_groups,)
        Each present key and the [start, end) range of its entries in order.
    """
    n = keys.shape[0]
    n_chunks = max(1, min(_DEFAULT_NUM_THREADS, n))
    chunk_len = (n + n_chunks - 1) // n_chunks
    counts = np.zeros((n_chunks, n_keys), dtype=np.int32)
    for c in numba.prange(n_chunks):
        for p in range(c * chunk_len, min(n, (c + 1) * chunk_len)):
            counts[c, keys[p]] += 1
    chunk_offsets = np.empty((n_chunks, n_keys), dtype=np.int32)
    key_starts = np.empty(n_keys + 1, dtype=np.int32)
    running = 0
    for k in range(n_keys):
        key_starts[k] = running
        for c in range(n_chunks):
            chunk_offsets[c, k] = running
            running += counts[c, k]
    key_starts[n_keys] = running
    order = np.empty(n, dtype=np.int32)
    for c in numba.prange(n_chunks):
        positions = chunk_offsets[c].copy()
        for p in range(c * chunk_len, min(n, (c + 1) * chunk_len)):
            k = keys[p]
            order[positions[k]] = p
            positions[k] += 1
    n_groups = 0
    for k in range(n_keys):
        if key_starts[k + 1] > key_starts[k]:
            n_groups += 1
    group_keys = np.empty(n_groups, dtype=np.int32)
    group_starts = np.empty(n_groups, dtype=np.int32)
    group_ends = np.empty(n_groups, dtype=np.int32)
    g = 0
    for k in range(n_keys):
        if key_starts[k + 1] > key_starts[k]:
            group_keys[g] = k
            group_starts[g] = key_starts[k]
            group_ends[g] = key_starts[k + 1]
            g += 1
    return order, group_keys, group_starts, group_ends


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _group_by_bin_and_band(
        bin_indices: npt.NDArray[np.int32], band_indices: npt.NDArray[np.int32], n_grid: int, n_bands: int,
) -> t.Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.int32],
npt.NDArray[np.int32]]:
    """
    Stable permutation grouping a batch's transitions by (band, bin) super-line: a counting sort by bin, then a stable
    counting sort by band within each bin. Keeps the count tables small (n_grid, then n_bands per bin) where a single
    sort on the combined key would need (n_chunks, n_bands * n_grid) tables.

    Returns
    -------
    order : ndarray, shape (n_trans,)
    group_bands, group_bins, group_starts, group_ends : ndarray, shape (n_groups,)
        Each occupied (band, bin) and the [start, end) range of its transitions in order.
    """
    order_bin, bin_keys, bin_starts, bin_ends = _counting_sort_order(keys=bin_indices, n_keys=n_grid)
    n_bin_groups = bin_keys.shape[0]
    order = np.empty_like(order_bin)
    n_sub = np.zeros(n_bin_groups, dtype=np.int32)
    for g in numba.prange(n_bin_groups):
        counts = np.zeros(n_bands, dtype=np.int32)
        for p in range(bin_starts[g], bin_ends[g]):
            counts[band_indices[order_bin[p]]] += 1
        positions = np.empty(n_bands, dtype=np.int32)
        running = bin_starts[g]
        n_present = 0
        for b in range(n_bands):
            positions[b] = running
            running += counts[b]
            if counts[b] > 0:
                n_present += 1
        for p in range(bin_starts[g], bin_ends[g]):
            t_ = order_bin[p]
            b = band_indices[t_]
            order[positions[b]] = t_
            positions[b] += 1
        n_sub[g] = n_present
    sub_offsets = np.zeros(n_bin_groups + 1, dtype=np.int32)
    for g in range(n_bin_groups):
        sub_offsets[g + 1] = sub_offsets[g] + n_sub[g]
    n_groups = sub_offsets[n_bin_groups]
    group_bands = np.empty(n_groups, dtype=np.int32)
    group_bins = np.empty(n_groups, dtype=np.int32)
    group_starts = np.empty(n_groups, dtype=np.int32)
    group_ends = np.empty(n_groups, dtype=np.int32)
    for g in numba.prange(n_bin_groups):
        q = sub_offsets[g]
        p = bin_starts[g]
        while p < bin_ends[g]:
            b = band_indices[order[p]]
            start = p
            while p < bin_ends[g] and band_indices[order[p]] == b:
                p += 1
            group_bands[q] = b
            group_bins[q] = bin_keys[g]
            group_starts[q] = start
            group_ends[q] = p
            q += 1
    return order, group_bands, group_bins, group_starts, group_ends


@numba.njit(parallel=True, nogil=True, cache=True, error_model="numpy")
def accumulate_superline_batch(
        profile_buffer: npt.NDArray[np.float64],
        bin_edges: npt.NDArray[np.float64],
        n_lookup: npt.NDArray[np.float64],
        g_lookup: npt.NDArray[np.float64],
        inv_g_lookup: npt.NDArray[np.float64],
        gamma_lifetime_lookup: npt.NDArray[np.float64],
        id_f: npt.NDArray[np.int32],
        id_i: npt.NDArray[np.int32],
        a_fi: npt.NDArray[np.float64],
        energy_fi: npt.NDArray[np.float64],
        n_workers: int = _DEFAULT_NUM_THREADS,
) -> None:
    """
    Bins a batch of transitions into super-lines on the grid for every layer (no band axis): net absorption
    (absorption minus stimulated emission) in channel 0 and spontaneous emission in channel 1. Channels 2 and 3
    accumulate sum(abs * gamma_lifetime) and sum(abs) over the (gross) absorption; their ratio is the
    absorption-weighted mean lifetime HWHM of each super-line used in :func:`broaden_superline_buffer`, as for the
    per-band super-lines of :func:`accumulate_superline_band_batch`.

    Each occupied bin is owned by one task: the batch is counting-sorted by bin, and a task sums all of its bin's
    transitions for every layer in local accumulators before writing that bin once. This replaces a loop over layers
    that, for every (layer, transition), gathered two scattered populations and scatter-added into the buffer. Tasks
    are dealt round-robin to n_workers workers, heaviest first. Per-bin sums keep the transitions' input order.

    Parameters
    ----------
    profile_buffer : ndarray, shape (4, n_layers, n_grid)
    bin_edges : ndarray, shape (n_grid + 1,)
    n_lookup : ndarray, shape (n_states + 1, n_layers)
        State-major, so a state's populations in every layer are contiguous.
    g_lookup : ndarray, shape (n_states + 1)
    inv_g_lookup : ndarray, shape (n_states + 1)
    gamma_lifetime_lookup : ndarray, shape (n_states + 1,)
        Per-state lifetime HWHM contributions [cm^-1], from :func:`gamma_lifetime_lookup_from_tau`.
    id_f : ndarray, shape (n_trans,)
    id_i : ndarray, shape (n_trans,)
    a_fi : ndarray, shape (n_trans,)
    energy_fi : ndarray (n_trans,)
    n_workers : int
    """
    n_layers = n_lookup.shape[1]
    n_grid = bin_edges.shape[0] - 1
    order, group_bins, group_starts, group_ends = _counting_sort_order(
        keys=_super_line_bins(bin_edges=bin_edges, energy_fi=energy_fi), n_keys=n_grid,
    )
    n_groups = group_bins.shape[0]
    task_order = np.argsort(-(group_ends - group_starts), kind="mergesort")

    for w in numba.prange(n_workers):
        acc = np.empty((4, n_layers), dtype=np.float64)
        for k in range(w, n_groups, n_workers):
            gk = task_order[k]
            bin_t = group_bins[gk]
            acc[:] = 0.0
            for p in range(group_starts[gk], group_ends[gk]):
                t_ = order[p]
                a_fi_t = a_fi[t_]
                energy_fi_t = energy_fi[t_]
                id_f_t = id_f[t_]
                id_i_t = id_i[t_]
                ste_prefactor_t = a_fi_t / (const_8_pi_c * energy_fi_t * energy_fi_t)
                abs_prefactor_t = ste_prefactor_t * g_lookup[id_f_t] * inv_g_lookup[id_i_t]
                emi_prefactor_t = a_fi_t * energy_fi_t * const_h_c_on_4_pi
                gamma_lifetime_t = gamma_lifetime_lookup[id_f_t] + gamma_lifetime_lookup[id_i_t]
                n_i = n_lookup[id_i_t]
                n_f = n_lookup[id_f_t]
                for l in range(n_layers):
                    abs_tl = abs_prefactor_t * n_i[l]
                    acc[0, l] += abs_tl - ste_prefactor_t * n_f[l]
                    acc[1, l] += emi_prefactor_t * n_f[l]
                    acc[2, l] += abs_tl * gamma_lifetime_t
                    acc[3, l] += abs_tl
            for l in range(n_layers):
                profile_buffer[0, l, bin_t] += acc[0, l]
                profile_buffer[1, l, bin_t] += acc[1, l]
                profile_buffer[2, l, bin_t] += acc[2, l]
                profile_buffer[3, l, bin_t] += acc[3, l]


@numba.njit(parallel=True, nogil=True, cache=True, error_model="numpy")
def accumulate_continuum_superline_batch(
        profile_buffer: npt.NDArray[np.float64],
        bin_edges: npt.NDArray[np.float64],
        n_lookup: npt.NDArray[np.float64],
        g_lookup: npt.NDArray[np.float64],
        inv_g_lookup: npt.NDArray[np.float64],
        v_lookup: npt.NDArray[np.float64],
        id_f: npt.NDArray[np.int32],
        id_i: npt.NDArray[np.int32],
        a_fi: npt.NDArray[np.float64],
        energy_fi: npt.NDArray[np.float64],
        reduced_mass: float,
        box_length: float,
        n_workers: int = _DEFAULT_NUM_THREADS,
) -> None:
    """
    Bins a batch of bound-free transitions into super-lines on the grid for every layer (no band axis): absorption in
    channel 0 and the absorption-weighted box broadening in channel 1. Can be modified to add in photoassociation
    cross-section if desired. Bin-owned tasks as in :func:`accumulate_superline_batch`.

    Parameters
    ----------
    profile_buffer : ndarray, shape (2, n_layers, n_grid)
    bin_edges : ndarray, shape (n_grid + 1,)
    n_lookup : ndarray, shape (n_states + 1, n_layers)
        State-major, so a state's populations in every layer are contiguous.
    g_lookup : ndarray, shape (n_states + 1, )
    inv_g_lookup : ndarray, shape (n_states + 1, )
    v_lookup : ndarray, shape (n_states + 1, )
    id_f : ndarray, shape (n_trans,)
    id_i : ndarray, shape (n_trans,)
    a_fi : ndarray, shape (n_trans,)
    energy_fi : ndarray (n_trans,)
    reduced_mass : float
    box_length : float
    n_workers : int
    """
    n_layers = n_lookup.shape[1]
    n_grid = bin_edges.shape[0] - 1
    n_states = v_lookup.shape[0]
    alpha_box_lookup = np.empty(n_states, dtype=np.float64)
    alpha_box_prefactor = const_h_on_8_c_amu / (box_length * box_length * reduced_mass)
    for s in numba.prange(n_states):
        alpha_box_lookup[s] = alpha_box_prefactor * (2.0 * v_lookup[s] + 1.0)

    order, group_bins, group_starts, group_ends = _counting_sort_order(
        keys=_super_line_bins(bin_edges=bin_edges, energy_fi=energy_fi), n_keys=n_grid,
    )
    n_groups = group_bins.shape[0]
    task_order = np.argsort(-(group_ends - group_starts), kind="mergesort")

    for w in numba.prange(n_workers):
        acc = np.empty((2, n_layers), dtype=np.float64)
        for k in range(w, n_groups, n_workers):
            gk = task_order[k]
            bin_t = group_bins[gk]
            acc[:] = 0.0
            for p in range(group_starts[gk], group_ends[gk]):
                t_ = order[p]
                energy_fi_t = energy_fi[t_]
                abs_prefactor_t = a_fi[t_] / (const_8_pi_c * energy_fi_t * energy_fi_t)
                abs_scale_t = abs_prefactor_t * g_lookup[id_f[t_]] * inv_g_lookup[id_i[t_]]
                alpha_box_t = alpha_box_lookup[id_f[t_]]
                n_i = n_lookup[id_i[t_]]
                for l in range(n_layers):
                    abs_tl = abs_scale_t * n_i[l]
                    acc[0, l] += abs_tl
                    acc[1, l] += abs_tl * alpha_box_t
            for l in range(n_layers):
                profile_buffer[0, l, bin_t] += acc[0, l]
                profile_buffer[1, l, bin_t] += acc[1, l]


@numba.njit(parallel=True, nogil=True, cache=True, error_model="numpy")
def accumulate_superline_band_batch(
        profile_buffer: npt.NDArray[np.float64],
        band_indices: npt.NDArray[np.int32],
        bin_edges: npt.NDArray[np.float64],
        n_frac_lookup: npt.NDArray[np.float64],
        g_lookup: npt.NDArray[np.float64],
        inv_g_lookup: npt.NDArray[np.float64],
        gamma_lifetime_lookup: npt.NDArray[np.float64],
        id_f: npt.NDArray[np.int32],
        id_i: npt.NDArray[np.int32],
        a_fi: npt.NDArray[np.float64],
        energy_fi: npt.NDArray[np.float64],
        n_workers: int = _DEFAULT_NUM_THREADS,
) -> None:
    """
    Bins a batch of transitions into per-band super-lines on the grid, for every layer.

    Channels 0-2 of profile_buffer accumulate the absorption, stimulated emission and spontaneous emission
    coefficients. Channel 3 accumulates the absorption-weighted lifetime broadening, sum(abs * gamma_lifetime); dividing
    it by channel 0 gives the intensity-weighted mean lifetime HWHM of each super-line, used in
    :func:`_broaden_superline_band_buffer`. This mirrors the box-broadening channel of
    :func:`accumulate_continuum_superline_band_batch`.

    Each occupied (band, bin) super-line is owned by one task: the batch is grouped by (band, bin)
    (:func:`_group_by_bin_and_band`), and a task sums its transitions for every layer in local accumulators before
    writing its cells once. Tasks are dealt round-robin to n_workers workers, heaviest first.

    Parameters
    ----------
    profile_buffer : ndarray, shape (4, n_bands, n_layers, n_grid)
    band_indices : ndarray, shape (n_trans,)
    bin_edges : ndarray, shape (n_grid + 1,)
    n_frac_lookup : ndarray, shape (n_states + 1, n_layers)
        State-major, so a state's populations in every layer are contiguous.
    g_lookup : ndarray, shape (n_states + 1,)
    inv_g_lookup : ndarray, shape (n_states + 1,)
    gamma_lifetime_lookup : ndarray, shape (n_states + 1,)
        Per-state lifetime HWHM contributions [cm^-1], from :func:`gamma_lifetime_lookup_from_tau`.
    id_f : ndarray, shape (n_trans,)
    id_i : ndarray, shape (n_trans,)
    a_fi : ndarray, shape (n_trans,)
    energy_fi : ndarray (n_trans,)
    n_workers : int
    """
    n_layers = n_frac_lookup.shape[1]
    n_grid = bin_edges.shape[0] - 1
    order, group_bands, group_bins, group_starts, group_ends = _group_by_bin_and_band(
        bin_indices=_super_line_bins(bin_edges=bin_edges, energy_fi=energy_fi), band_indices=band_indices,
        n_grid=n_grid, n_bands=profile_buffer.shape[1],
    )
    n_groups = group_bins.shape[0]
    task_order = np.argsort(-(group_ends - group_starts), kind="mergesort")

    for w in numba.prange(n_workers):
        acc = np.empty((4, n_layers), dtype=np.float64)
        for k in range(w, n_groups, n_workers):
            gk = task_order[k]
            band_t = group_bands[gk]
            bin_t = group_bins[gk]
            acc[:] = 0.0
            for p in range(group_starts[gk], group_ends[gk]):
                t_ = order[p]
                a_fi_t = a_fi[t_]
                energy_fi_t = energy_fi[t_]
                id_f_t = id_f[t_]
                id_i_t = id_i[t_]
                ste_prefactor_t = a_fi_t / (const_8_pi_c * energy_fi_t * energy_fi_t)
                abs_prefactor_t = ste_prefactor_t * g_lookup[id_f_t] * inv_g_lookup[id_i_t]
                spe_prefactor_t = a_fi_t * energy_fi_t * const_h_c_on_4_pi
                gamma_lifetime_t = gamma_lifetime_lookup[id_f_t] + gamma_lifetime_lookup[id_i_t]
                n_i = n_frac_lookup[id_i_t]
                n_f = n_frac_lookup[id_f_t]
                for l in range(n_layers):
                    abs_tl = abs_prefactor_t * n_i[l]
                    acc[0, l] += abs_tl
                    acc[1, l] += ste_prefactor_t * n_f[l]
                    acc[2, l] += spe_prefactor_t * n_f[l]
                    acc[3, l] += abs_tl * gamma_lifetime_t
            for l in range(n_layers):
                profile_buffer[0, band_t, l, bin_t] += acc[0, l]
                profile_buffer[1, band_t, l, bin_t] += acc[1, l]
                profile_buffer[2, band_t, l, bin_t] += acc[2, l]
                profile_buffer[3, band_t, l, bin_t] += acc[3, l]


@numba.njit(parallel=True, nogil=True, cache=True, error_model="numpy")
def accumulate_continuum_superline_band_batch(
        profile_buffer: npt.NDArray[np.float64],
        band_indices: npt.NDArray[np.int32],
        bin_edges: npt.NDArray[np.float64],
        n_frac_lookup: npt.NDArray[np.float64],
        g_lookup: npt.NDArray[np.float64],
        inv_g_lookup: npt.NDArray[np.float64],
        v_lookup: npt.NDArray[np.float64],
        id_f: npt.NDArray[np.int32],
        id_i: npt.NDArray[np.int32],
        a_fi: npt.NDArray[np.float64],
        energy_fi: npt.NDArray[np.float64],
        reduced_mass: float,
        box_length: float,
        n_workers: int = _DEFAULT_NUM_THREADS,
) -> None:
    """
    Bins a batch of bound-free transitions into per-band super-lines: absorption in channel 0 and the
    absorption-weighted box broadening in channel 1. Can be modified to add in photoassociation cross-section if
    desired. (band, bin)-owned tasks as in :func:`accumulate_superline_band_batch`.

    Parameters
    ----------
    profile_buffer : ndarray, shape (2, n_bands, n_layers, n_grid)
    band_indices : ndarray, shape (n_trans,)
    bin_edges : ndarray, shape (n_grid + 1,)
    n_frac_lookup : ndarray, shape (n_states + 1, n_layers)
        State-major, so a state's populations in every layer are contiguous.
    g_lookup : ndarray, shape (n_states + 1, )
    inv_g_lookup : ndarray, shape (n_states + 1,)
    v_lookup : ndarray, shape (n_states + 1, )
    id_f : ndarray, shape (n_trans,)
    id_i : ndarray, shape (n_trans,)
    a_fi : ndarray, shape (n_trans,)
    energy_fi : ndarray (n_trans,)
    reduced_mass : float
    box_length : float
    n_workers : int
    """
    n_layers = n_frac_lookup.shape[1]
    n_grid = bin_edges.shape[0] - 1
    n_states = v_lookup.shape[0]
    alpha_box_lookup = np.empty(n_states, dtype=np.float64)
    alpha_box_prefactor = const_h_on_8_c_amu / (box_length * box_length * reduced_mass)
    for s in numba.prange(n_states):
        alpha_box_lookup[s] = alpha_box_prefactor * (2.0 * v_lookup[s] + 1.0)

    order, group_bands, group_bins, group_starts, group_ends = _group_by_bin_and_band(
        bin_indices=_super_line_bins(bin_edges=bin_edges, energy_fi=energy_fi), band_indices=band_indices,
        n_grid=n_grid, n_bands=profile_buffer.shape[1],
    )
    n_groups = group_bins.shape[0]
    task_order = np.argsort(-(group_ends - group_starts), kind="mergesort")

    for w in numba.prange(n_workers):
        acc = np.empty((2, n_layers), dtype=np.float64)
        for k in range(w, n_groups, n_workers):
            gk = task_order[k]
            band_t = group_bands[gk]
            bin_t = group_bins[gk]
            acc[:] = 0.0
            for p in range(group_starts[gk], group_ends[gk]):
                t_ = order[p]
                energy_fi_t = energy_fi[t_]
                abs_prefactor_t = a_fi[t_] / (const_8_pi_c * energy_fi_t * energy_fi_t)
                abs_scale_t = abs_prefactor_t * g_lookup[id_f[t_]] * inv_g_lookup[id_i[t_]]
                alpha_box_t = alpha_box_lookup[id_f[t_]]
                n_i = n_frac_lookup[id_i[t_]]
                for l in range(n_layers):
                    abs_tl = abs_scale_t * n_i[l]
                    acc[0, l] += abs_tl
                    acc[1, l] += abs_tl * alpha_box_t
            for l in range(n_layers):
                profile_buffer[0, band_t, l, bin_t] += acc[0, l]
                profile_buffer[1, band_t, l, bin_t] += acc[1, l]


@numba.njit(parallel=True, cache=True, error_model="numpy")
def broaden_superline_buffer(
        profile_buffer: npt.NDArray[np.float64],
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
    Broadens super-lines with a Voigt profile combining Doppler, pressure and lifetime broadening, the lifetime HWHM
    of each (layer, bin) being channel 2 / channel 3 of profile_buffer (see :func:`accumulate_superline_batch`).

    Parameters
    ----------
    profile_buffer : (4, n_layers, n_grid)
        Accumulated super-line coefficients per layer per grid bin, from :func:`accumulate_superline_batch`. Net
        absorption is stored in profile_buffer[0] and emission in profile_buffer[1].
    wn_grid : ndarray, shape (n_grid,)
    temperatures : ndarray, shape (n_layers,)
    pressures : ndarray, shape (n_layers,)
    broad_n : ndarray, shape (n_broadeners,)
    broad_gamma : ndarray, shape (n_broadeners, n_layers)
        Note: layers on axis-1, matching broadening_params[0].
    species_mass : float

    Returns
    -------
    out_abs : ndarray, shape (n_layers, n_grid)
    out_emi : ndarray, shape (n_layers, n_grid)
    """
    n_layers = temperatures.shape[0]
    n_grid = wn_grid.shape[0]
    cutoff = 25.0
    out_abs = np.zeros((n_layers, n_grid), dtype=np.float64)
    out_emi = np.zeros((n_layers, n_grid), dtype=np.float64)
    inv_sqrt_pi = 1 / np.sqrt(np.pi)

    # Per-layer broadening terms: 1 / (sigma_D * sqrt(2)) = inv_doppler[l] / energy, and the pressure HWHM.
    inv_doppler, gamma_pressure = layer_broadening_terms(
        temperatures=temperatures, pressures=pressures, broad_n=broad_n, broad_gamma=broad_gamma,
        species_mass=species_mass, t_ref=t_ref, pressure_ref=pressure_ref,
    )
    j_starts, j_ends = profile_windows(wn_grid=wn_grid, cutoff=cutoff)

    # Parallelised over layers, not grid points: adjacent grid points' +/-cutoff windows
    # overlap, so a thread-per-grid-point scheme would race on out_abs/out_emi writes from
    # different threads. Each thread here owns an exclusive output row (out_abs/out_emi[l, :]).
    for l in numba.prange(n_layers):
        for i in range(n_grid):
            abs_li = profile_buffer[0, l, i]
            emi_li = profile_buffer[1, l, i]
            # Skip empty bins.
            if abs_li == 0.0 and emi_li == 0.0:
                continue
            # Super-line centre is the bin centre.
            energy_i = wn_grid[i]
            inv_sigma_sqrt2_il = inv_doppler[l] / energy_i
            gamma_lifetime_li = 0.0
            if profile_buffer[3, l, i] != 0.0:
                gamma_lifetime_li = profile_buffer[2, l, i] / profile_buffer[3, l, i]
            # Voigt y-parameter: gamma_L / (sigma_D * sqrt(2)).
            y_il = (gamma_pressure[l] + gamma_lifetime_li) * inv_sigma_sqrt2_il
            # Integral of Re[w(x,y)] dx = sqrt(pi), so the normalised Voigt is Re[w(z)] / (sigma * sqrt(2*pi)).
            norm = inv_sigma_sqrt2_il * inv_sqrt_pi

            for j in range(j_starts[i], j_ends[i]):
                x_ij = (wn_grid[j] - energy_i) * inv_sigma_sqrt2_il
                voigt_val = voigt_humlicek_w(x_ij, y_il) * norm

                out_abs[l, j] += abs_li * voigt_val
                out_emi[l, j] += emi_li * voigt_val
    return out_abs, out_emi


@numba.njit(parallel=True, cache=True, error_model="numpy")
def broaden_continuum_superline_buffer(
        profile_buffer: npt.NDArray[np.float64],
        wn_grid: npt.NDArray[np.float64],
        temperatures: npt.NDArray[np.float64],
        species_mass: float,
) -> npt.NDArray[np.float64]:
    """
    Parameters
    ----------
    profile_buffer : ndarray, shape (2, n_layers, n_trans)
        Accumulated super-line coefficients per layer per grid bin. Absorption is stored in profile_buffer[0] and
        the product of the absorption coefficient and the line-by-line box broadening in profile_buffer[1]; this must be
        divided by profile_buffer[0] to obtain the intensity weighted mean box broadening.
    wn_grid : ndarray, shape (n_grid,)
    temperatures : ndarray, shape (n_layers,)
    species_mass : float

    Returns
    -------
    out_abs : ndarray, shape (n_layers, n_grid)
    """
    n_layers = temperatures.shape[0]
    n_grid = wn_grid.shape[0]
    min_cutoff = 25.0
    max_cutoff = 5000.0
    cutoff_fwhm_multiple = 5.0

    out_abs = np.zeros((n_layers, n_grid), dtype=np.float64)

    sqrtln2 = np.sqrt(np.log(2.0))
    inv_sqrt_pi = 1 / np.sqrt(np.pi)

    inv_mass = 1 / species_mass
    doppler_coef = const_sqrt_2_NA_kB_log2_on_c * math.sqrt(inv_mass)

    # Parallelised over layers, not grid points: the Doppler+box cutoff window varies per
    # (l, i) and adjacent grid points' windows overlap, so a thread-per-grid-point scheme
    # would race on out_abs writes from different threads. Each thread here owns an
    # exclusive output row (out_abs[l, :]).
    for l in numba.prange(n_layers):
        temp_l = temperatures[l]
        doppler_prefactor_l = doppler_coef * math.sqrt(temp_l)

        for i in range(n_grid):
            abs_li = profile_buffer[0, l, i]
            # Skip empty bins.
            if abs_li == 0.0:
                continue
            # Super-line centre is the bin centre.
            energy_i = wn_grid[i]

            weighted_broad = profile_buffer[1, l, i]
            alpha_box_li = weighted_broad / abs_li
            alpha_doppler_li = doppler_prefactor_l * energy_i
            alpha_total_li = alpha_box_li + alpha_doppler_li
            cutoff_li = cutoff_fwhm_multiple * alpha_total_li
            if cutoff_li < min_cutoff:
                cutoff_li = min_cutoff
            elif cutoff_li > max_cutoff:
                cutoff_li = max_cutoff

            transition_min = energy_i - cutoff_li
            transition_max = energy_i + cutoff_li
            j_start = max(0, binary_search_left(wn_grid, transition_min))
            j_end = min(n_grid, binary_search_right(wn_grid, transition_max) + 1)

            if j_start >= j_end:
                continue

            sqrtln2_on_alpha_li = sqrtln2 / alpha_total_li
            norm = sqrtln2_on_alpha_li * inv_sqrt_pi

            for j in range(j_start, j_end):
                wn_j = wn_grid[j]
                arg = sqrtln2_on_alpha_li * (wn_j - energy_i)
                gauss_val = norm * math.exp(-(arg * arg))

                out_abs[l, j] += abs_li * gauss_val

    return out_abs


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _broaden_superline_band_buffer(
        profile_buffer: npt.NDArray[np.float64],
        wn_grid: npt.NDArray[np.float64],
        temperatures: npt.NDArray[np.float64],
        pressures: npt.NDArray[np.float64],
        broad_n: npt.NDArray[np.float64],
        broad_gamma: npt.NDArray[np.float64],
        species_mass: float,
        t_ref: float = 296.0,
        pressure_ref: float = 1.0,
        n_workers: int = _DEFAULT_NUM_THREADS,
) -> npt.NDArray[np.float64]:
    """
    Broadens per-band super-lines with a Voigt profile combining Doppler, pressure and lifetime broadening.

    The Doppler width depends on the (layer, bin), the pressure width on the layer, and the lifetime width is the
    intensity-weighted mean lifetime HWHM of the lines binned into each (band, layer, bin): channel 3 / channel 0 of
    profile_buffer. The Voigt profile is evaluated exactly for every occupied (band, layer, bin).

    Parameters
    ----------
    profile_buffer : ndarray, shape (4, n_bands, n_layers, n_grid)
        Accumulated super-line absorption, stimulated emission and spontaneous emission coefficients, plus the
        absorption-weighted lifetime HWHM, from :func:`accumulate_superline_band_batch`.
    wn_grid : ndarray, shape (n_grid,)
    temperatures : ndarray, shape (n_layers,)
    pressures : ndarray, shape (n_layers,)
    broad_n : ndarray, shape (n_broadeners,)
    broad_gamma : ndarray, shape (n_broadeners, n_layers)
        Note: layers on axis-1, matching broadening_params[0].
    species_mass : float
    t_ref : float
    pressure_ref : float
    n_workers : int
        Number of parallel workers the (band, layer) pairs are dealt between; normally numba's thread count.

    Returns
    -------
    out_buffer : ndarray, shape (3, n_bands, n_layers, n_grid)
    """
    n_layers = temperatures.shape[0]
    n_grid = wn_grid.shape[0]
    n_bands = profile_buffer.shape[1]
    cutoff = 25.0

    buffer_out = np.zeros((3, n_bands, n_layers, n_grid), dtype=np.float64)

    inv_sqrt_pi = 1 / np.sqrt(np.pi)

    # Per-layer broadening terms: 1 / (sigma_D * sqrt(2)) = inv_doppler[l] / energy, and the pressure HWHM.
    inv_doppler, gamma_pressure = layer_broadening_terms(
        temperatures=temperatures, pressures=pressures, broad_n=broad_n, broad_gamma=broad_gamma,
        species_mass=species_mass, t_ref=t_ref, pressure_ref=pressure_ref,
    )

    j_starts, j_ends = profile_windows(wn_grid=wn_grid, cutoff=cutoff)

    # Each (band, layer) pair writes only its own output row buffer_out[:, band, l, :], so pairs can run in parallel
    # without two threads ever writing the same output point. Pairs differ enormously in cost (a band may occupy a few
    # bins or the whole grid), and a prange over the pairs would give each thread an equal-length contiguous block of
    # them, leaving the thread that gets the busiest bands with several times the average work. Instead, the pairs are
    # sorted by estimated cost and dealt round-robin to n_workers workers, heaviest first (worker w takes sorted pairs
    # w, w + n_workers, w + 2 * n_workers, ...), which keeps every worker's total within a few percent of the mean.
    order, cost = band_layer_work_order(profile_buffer=profile_buffer, j_starts=j_starts, j_ends=j_ends)
    n_pairs = order.shape[0]

    for w in numba.prange(n_workers):
        for k in range(w, n_pairs, n_workers):
            bl = order[k]
            if cost[bl] == 0:
                # Sorted by decreasing cost, so every remaining pair for this worker is empty too.
                break
            band = bl // n_layers
            l = bl % n_layers
            inv_doppler_l = inv_doppler[l]
            gamma_pressure_l = gamma_pressure[l]
            abs_out = buffer_out[0, band, l]
            ste_out = buffer_out[1, band, l]
            spe_out = buffer_out[2, band, l]

            # Super-line centre is the bin centre.
            for i in range(n_grid):
                abs_bli = profile_buffer[0, band, l, i]
                ste_bli = profile_buffer[1, band, l, i]
                spe_bli = profile_buffer[2, band, l, i]
                # Skip empty bins.
                if abs_bli == 0.0 and ste_bli == 0.0 and spe_bli == 0.0:
                    continue
                energy_i = wn_grid[i]
                inv_sigma_sqrt2_il = inv_doppler_l / energy_i
                # Integral of Re[w(x,y)] dx = sqrt(pi), so the normalised Voigt is Re[w(z)] / (sigma * sqrt(2*pi)).
                norm = inv_sigma_sqrt2_il * inv_sqrt_pi
                gamma_lifetime_bli = 0.0
                if abs_bli != 0.0:
                    gamma_lifetime_bli = profile_buffer[3, band, l, i] / abs_bli
                # Voigt y-parameter: gamma_L / (sigma_D * sqrt(2)).
                y_bli = (gamma_pressure_l + gamma_lifetime_bli) * inv_sigma_sqrt2_il

                for j in range(j_starts[i], j_ends[i]):
                    x_ij = (wn_grid[j] - energy_i) * inv_sigma_sqrt2_il
                    voigt_val = voigt_humlicek_w(x_ij, y_bli) * norm
                    abs_out[j] += abs_bli * voigt_val
                    ste_out[j] += ste_bli * voigt_val
                    spe_out[j] += spe_bli * voigt_val

    return buffer_out


def broaden_superline_band_buffer(
        *,
        profile_buffer: npt.NDArray[np.float64],
        wn_grid: npt.NDArray[np.float64],
        temperatures: npt.NDArray[np.float64],
        pressures: npt.NDArray[np.float64],
        broad_n: npt.NDArray[np.float64],
        broad_gamma: npt.NDArray[np.float64],
        species_mass: float,
        t_ref: float = 296.0,
        pressure_ref: float = 1.0,
) -> npt.NDArray[np.float64]:
    """CPU entry point for :func:`_broaden_superline_band_buffer`, dealing work between _DEFAULT_NUM_THREADS workers.
    The GPU equivalent is :meth:`profiles.gpu.DeviceProfileBuffer.broaden_superlines`."""
    return _broaden_superline_band_buffer(
        profile_buffer=profile_buffer, wn_grid=wn_grid, temperatures=temperatures, pressures=pressures,
        broad_n=broad_n, broad_gamma=broad_gamma, species_mass=species_mass, t_ref=t_ref, pressure_ref=pressure_ref,
        n_workers=_DEFAULT_NUM_THREADS,
    )


@numba.njit(cache=True, error_model="numpy", inline="always")
def _continuum_superline_window(
        wn_grid: npt.NDArray[np.float64], energy_i: float, alpha_total: float,
) -> t.Tuple[int, int]:
    """Grid window [j_start, j_end) of a continuum super-line: 5 Gaussian HWHM either side, clamped to 25-5000 cm^-1."""
    cutoff = 5.0 * alpha_total
    if cutoff < 25.0:
        cutoff = 25.0
    elif cutoff > 5000.0:
        cutoff = 5000.0
    j_start = max(0, binary_search_left(wn_grid, energy_i - cutoff))
    j_end = min(wn_grid.shape[0], binary_search_right(wn_grid, energy_i + cutoff) + 1)
    return j_start, j_end


@numba.njit(parallel=True, cache=True, error_model="numpy")
def broaden_continuum_superline_band_buffer(
        profile_buffer: npt.NDArray[np.float64],
        wn_grid: npt.NDArray[np.float64],
        temperatures: npt.NDArray[np.float64],
        species_mass: float,
        n_workers: int = _DEFAULT_NUM_THREADS,
) -> npt.NDArray[np.float64]:
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

    buffer_out = np.zeros((1, n_bands, n_layers, n_grid), dtype=np.float64)

    sqrtln2 = np.sqrt(np.log(2.0))
    inv_sqrt_pi = 1 / np.sqrt(np.pi)

    doppler_coef = const_sqrt_2_NA_kB_log2_on_c * math.sqrt(1 / species_mass)
    doppler_prefactor = np.empty(n_layers, dtype=np.float64)
    for l in range(n_layers):
        doppler_prefactor[l] = doppler_coef * math.sqrt(temperatures[l])

    # Each (band, layer) pair owns its output row, so pairs run in parallel without write conflicts. Window widths vary
    # by orders of magnitude between bands (the box width sets them), so each pair's cost is its summed window length;
    # pairs are then dealt round-robin to n_workers workers, heaviest first.
    n_pairs = n_bands * n_layers
    cost = np.zeros(n_pairs, dtype=np.int64)
    for bl in numba.prange(n_pairs):
        band = bl // n_layers
        l = bl % n_layers
        c = 0
        for i in range(n_grid):
            abs_li = profile_buffer[0, band, l, i]
            if abs_li == 0.0:
                continue
            alpha_total_li = profile_buffer[1, band, l, i] / abs_li + doppler_prefactor[l] * wn_grid[i]
            j_start, j_end = _continuum_superline_window(
                wn_grid=wn_grid, energy_i=wn_grid[i], alpha_total=alpha_total_li,
            )
            c += max(j_end - j_start, 0)
        cost[bl] = c
    order = np.argsort(-cost, kind="mergesort")

    for w in numba.prange(n_workers):
        for k in range(w, n_pairs, n_workers):
            bl = order[k]
            if cost[bl] == 0:
                # Sorted by decreasing cost, so every remaining pair for this worker is empty too.
                break
            band = bl // n_layers
            l = bl % n_layers
            out_bl = buffer_out[0, band, l]
            doppler_prefactor_l = doppler_prefactor[l]
            # Super-line centre is the bin centre.
            for i in range(n_grid):
                abs_li = profile_buffer[0, band, l, i]
                # Skip empty bins.
                if abs_li == 0.0:
                    continue
                energy_i = wn_grid[i]
                # Intensity-weighted mean box width plus the Doppler width.
                alpha_box_li = profile_buffer[1, band, l, i] / abs_li
                alpha_total_li = alpha_box_li + doppler_prefactor_l * energy_i
                j_start, j_end = _continuum_superline_window(
                    wn_grid=wn_grid, energy_i=energy_i, alpha_total=alpha_total_li,
                )
                if j_start >= j_end:
                    continue
                sqrtln2_on_alpha_li = sqrtln2 / alpha_total_li
                norm = sqrtln2_on_alpha_li * inv_sqrt_pi
                for j in range(j_start, j_end):
                    arg = sqrtln2_on_alpha_li * (wn_grid[j] - energy_i)
                    out_bl[j] += abs_li * norm * math.exp(-(arg * arg))

    return buffer_out
