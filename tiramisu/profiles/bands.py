"""Band bookkeeping for NLTE rates: band indices of transitions, grouping by band, per-band Einstein-coefficient
sums and the broadening work order over (band, layer)."""
import typing as t

import numba
import numpy as np
from numpy import typing as npt

from ..config import _DEFAULT_NUM_THREADS
from ..constants import const_c_sq_on_2h, const_c_cgs


@numba.njit(parallel=True, nogil=True, cache=True, error_model="numpy")
def assign_band_indices(
        id_f: npt.NDArray[np.int32],
        id_i: npt.NDArray[np.int32],
        id_agg_lookup: npt.NDArray[np.int32],
        band_lookup: npt.NDArray[np.int32],
        id_agg_offset: int,
        n_bands_used: int,
) -> t.Tuple[npt.NDArray[np.int32], int]:
    """
    Maps each transition's (id_agg_f, id_agg_i) pair to a global band index through a dense lookup table, registering
    any pairs not seen in earlier batches.

    Called by :func:`~nlte._update_band_registry`.

    Parameters
    ----------
    id_f : ndarray, shape (n_trans,)
    id_i : ndarray, shape (n_trans,)
    id_agg_lookup : ndarray, shape (n_states + 1,)
        Aggregate state ID by state ID.
    band_lookup : ndarray, shape (n_agg_ids, n_agg_ids)
        Entry [id_agg_f + id_agg_offset, id_agg_i + id_agg_offset] holds the band index of that pair, or -1 if it has
        not been registered yet. Updated in place with any new bands.
    id_agg_offset : int
        Offset applied to aggregate IDs so the smallest one (e.g. -1 for unaggregated continuum states) maps to 0.
    n_bands_used : int
        Number of bands registered before this batch.

    Returns
    -------
    band_indices : ndarray, shape (n_trans,)
    n_bands_used : int
        Number of bands registered after this batch.
    """
    n_trans = id_f.shape[0]
    band_indices = np.empty(n_trans, dtype=np.int32)
    n_missing = 0
    for t in numba.prange(n_trans):
        band_t = band_lookup[id_agg_lookup[id_f[t]] + id_agg_offset, id_agg_lookup[id_i[t]] + id_agg_offset]
        band_indices[t] = band_t
        if band_t < 0:
            n_missing += 1

    if n_missing > 0:
        # New bands only appear in the first few batches; registering them serially keeps band indices deterministic.
        for t in range(n_trans):
            if band_indices[t] >= 0:
                continue
            f = id_agg_lookup[id_f[t]] + id_agg_offset
            i = id_agg_lookup[id_i[t]] + id_agg_offset
            band_t = band_lookup[f, i]
            if band_t < 0:
                band_t = n_bands_used
                band_lookup[f, i] = band_t
                n_bands_used += 1
            band_indices[t] = band_t

    return band_indices, n_bands_used


@numba.njit(parallel=True, nogil=True, cache=True, error_model="numpy")
def group_by_band(
        band_indices: npt.NDArray[np.int32],
        n_bands: int,
        id_f: npt.NDArray[np.int32],
        id_i: npt.NDArray[np.int32],
        a_fi: npt.NDArray[np.float64],
        energy_fi: npt.NDArray[np.float64],
) -> t.Tuple[
    npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.float64], npt.NDArray[np.float64],
    npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.int32]
]:
    """
    Stable, parallel counting sort of a transition batch by band index, so that each band's transitions are contiguous.

    Replaces sorting the batch on (id_agg_f, id_agg_i): O(n_trans + n_bands) rather than O(n_trans log n_trans).

    Parameters
    ----------
    band_indices : ndarray, shape (n_trans,)
    n_bands : int
        Total number of registered bands; all band_indices are < n_bands.
    id_f, id_i, a_fi, energy_fi : ndarray, shape (n_trans,)

    Returns
    -------
    id_f, id_i, a_fi, energy_fi : ndarray, shape (n_trans,)
        Inputs reordered so each band's transitions are contiguous.
    group_bands : ndarray, shape (n_bands_in_batch,)
        Global band index of each band present in the batch.
    group_starts : ndarray, shape (n_bands_in_batch,)
        Start index in the sorted arrays of each band's transitions.
    group_ends : ndarray, shape (n_bands_in_batch,)
        End index (exclusive) in the sorted arrays of each band's transitions.
    """
    n_trans = band_indices.shape[0]
    n_chunks = max(1, min(_DEFAULT_NUM_THREADS, n_trans))
    chunk_len = (n_trans + n_chunks - 1) // n_chunks

    # Per-chunk band histograms.
    counts = np.zeros((n_chunks, n_bands), dtype=np.int32)
    for c in numba.prange(n_chunks):
        for t in range(c * chunk_len, min(n_trans, (c + 1) * chunk_len)):
            counts[c, band_indices[t]] += 1

    # Exclusive prefix sum in (band, chunk) order: each chunk writes its share of a band after earlier chunks, so the
    # sort is stable.
    chunk_offsets = np.empty((n_chunks, n_bands), dtype=np.int32)
    band_starts = np.empty(n_bands + 1, dtype=np.int32)
    running = 0
    for b in range(n_bands):
        band_starts[b] = running
        for c in range(n_chunks):
            chunk_offsets[c, b] = running
            running += counts[c, b]
    band_starts[n_bands] = running

    id_f_out = np.empty_like(id_f)
    id_i_out = np.empty_like(id_i)
    a_fi_out = np.empty_like(a_fi)
    energy_fi_out = np.empty_like(energy_fi)
    for c in numba.prange(n_chunks):
        positions = chunk_offsets[c].copy()
        for t in range(c * chunk_len, min(n_trans, (c + 1) * chunk_len)):
            b = band_indices[t]
            p = positions[b]
            id_f_out[p] = id_f[t]
            id_i_out[p] = id_i[t]
            a_fi_out[p] = a_fi[t]
            energy_fi_out[p] = energy_fi[t]
            positions[b] = p + 1

    n_groups = 0
    for b in range(n_bands):
        if band_starts[b + 1] > band_starts[b]:
            n_groups += 1
    group_bands = np.empty(n_groups, dtype=np.int32)
    group_starts = np.empty(n_groups, dtype=np.int32)
    group_ends = np.empty(n_groups, dtype=np.int32)
    g = 0
    for b in range(n_bands):
        if band_starts[b + 1] > band_starts[b]:
            group_bands[g] = b
            group_starts[g] = band_starts[b]
            group_ends[g] = band_starts[b + 1]
            g += 1

    return id_f_out, id_i_out, a_fi_out, energy_fi_out, group_bands, group_starts, group_ends


@numba.njit(parallel=True, cache=True, error_model="numpy")
def band_layer_work_order(
        profile_buffer: npt.NDArray[np.float64],
        j_starts: npt.NDArray[np.int32],
        j_ends: npt.NDArray[np.int32],
) -> t.Tuple[npt.NDArray[np.int32], npt.NDArray[np.int64]]:
    """
    Estimated broadening cost of every (band, layer) pair of a super-line buffer, and the pairs sorted heaviest first.

    The cost of a pair is the number of Voigt evaluations it needs: the summed profile window length of its occupied
    bins. Pair bl stands for (band, layer) = (bl // n_layers, bl % n_layers).

    Parameters
    ----------
    profile_buffer : ndarray, shape (x, n_bands, n_layers, n_grid)
        Channels 0-2 hold the super-line coefficients; a bin is occupied if any of them is non-zero.
    j_starts, j_ends : ndarray, shape (n_grid,)
        Profile windows from :func:`profile_windows`.

    Returns
    -------
    order : ndarray, shape (n_bands * n_layers,)
        Pair indices sorted by decreasing cost.
    cost : ndarray, shape (n_bands * n_layers,)
        int64: a pair's summed window lengths can exceed 2^31 on fine grids at low wavenumber.
    """
    n_bands, n_layers, n_grid = profile_buffer.shape[1], profile_buffer.shape[2], profile_buffer.shape[3]
    n_pairs = n_bands * n_layers
    cost = np.zeros(n_pairs, dtype=np.int64)
    for bl in numba.prange(n_pairs):
        band = bl // n_layers
        l = bl % n_layers
        c = 0
        for i in range(n_grid):
            if profile_buffer[0, band, l, i] != 0.0 or profile_buffer[1, band, l, i] != 0.0 \
                    or profile_buffer[2, band, l, i] != 0.0:
                c += j_ends[i] - j_starts[i]
        cost[bl] = c
    order = np.argsort(-cost, kind="mergesort").astype(np.int32)
    return order, cost


@numba.njit(parallel=True, nogil=True, cache=True, error_model="numpy")
def band_rate_sums(
        band_indices: npt.NDArray[np.int32],
        n_bands: int,
        id_f: npt.NDArray[np.int32],
        id_i: npt.NDArray[np.int32],
        a_fi: npt.NDArray[np.float64],
        energy_fi: npt.NDArray[np.float64],
        g_lookup: npt.NDArray[np.float64],
        inv_g_lookup: npt.NDArray[np.float64],
        n_frac_lookup: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Per-band, per-layer population-weighted sums of the Einstein coefficients A_fi, B_fi and B_if over a batch of
    transitions, with B in the frequency domain [m^2 / (J s)].

    Each line is weighted by its source state's fraction of the aggregate population (``n_frac_lookup``, the same
    within-aggregate distribution the band profiles are built with): the upper state's for A_fi and B_fi, the lower
    state's for B_if. A band's sum is then a rate per molecule in the source aggregate, to be multiplied by that
    aggregate's population in the rate matrix. Unweighted sums would count the aggregate's population once per state
    (e.g. 8246 s^-1 instead of ~18 s^-1 for OH X v=1->0), and break detailed balance: with J = B_nu(T) and no
    collisions, the aggregate equations only return LTE when the weights are the within-aggregate Boltzmann fractions.

    Parameters
    ----------
    n_frac_lookup : ndarray, shape (max_state_id + 1, n_layers)
        Fraction of its aggregate's population in each state, per layer (state-major).

    Returns
    -------
    rate_sums : ndarray, shape (n_bands, n_layers, 3)
    """
    n_trans = a_fi.shape[0]
    n_layers = n_frac_lookup.shape[1]
    n_chunks = max(1, min(_DEFAULT_NUM_THREADS, n_trans))
    chunk_len = (n_trans + n_chunks - 1) // n_chunks
    partial = np.zeros((n_chunks, n_bands, n_layers, 3), dtype=np.float64)
    for c in numba.prange(n_chunks):
        for t in range(c * chunk_len, min(n_trans, (c + 1) * chunk_len)):
            nu_t = energy_fi[t] * const_c_cgs
            b_fi_t = a_fi[t] * const_c_sq_on_2h / (nu_t * nu_t * nu_t)
            b_if_t = b_fi_t * g_lookup[id_f[t]] * inv_g_lookup[id_i[t]]
            band_t = band_indices[t]
            for layer in range(n_layers):
                frac_f = n_frac_lookup[id_f[t], layer]
                partial[c, band_t, layer, 0] += a_fi[t] * frac_f
                partial[c, band_t, layer, 1] += b_fi_t * frac_f
                partial[c, band_t, layer, 2] += b_if_t * n_frac_lookup[id_i[t], layer]
    return partial.sum(axis=0)
