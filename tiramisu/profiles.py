import time
import logging
import math
import pathlib
import typing as t
from concurrent.futures import ThreadPoolExecutor

import numba
import numpy as np
import polars as pl
from astropy import constants as ac, units as u

from pyarrow import parquet as pq
from dask import dataframe as dd
from numpy import typing as npt

from . import config
from .config import _DEFAULT_NUM_THREADS, _INTENSITY_CUTOFF, _PARQUET_BATCH_SIZE, _DASK_BLOCK_SIZE, _LOG_VERBOSE_2
from .numerics import loglinear_normalise_2d_nonnegative, loglinear_integral_1d

log = logging.getLogger(__name__)

# Constants.
ac_h_on_8_c = ac.h.cgs / (8 * ac.c.cgs)
ac_4_pi_c = 4 * np.pi * ac.c.cgs
ac_8_pi_c = 2 * ac_4_pi_c
ac_h_c_on_4_pi = ac.h.cgs * ac.c.cgs / (4 * np.pi)
ac_sqrt_NA_kB_on_c = (np.sqrt(ac.N_A * ac.k_B.cgs) / ac.c.cgs).to(
    u.g ** 0.5 / (u.K ** 0.5 * u.mol ** 0.5), equivalencies=u.spectral()
)
ac_sqrt_2_NA_kB_log2_on_c = ac_sqrt_NA_kB_on_c * np.sqrt(2 * np.log(2))
ac_c_sq_on_2h = (ac.c ** 2) / (2 * ac.h)
# Constant values.
const_amu = ac.u.cgs.value
const_h_on_8_c = ac_h_on_8_c.value
const_h_on_8_c_amu = const_h_on_8_c / const_amu
const_4_pi_c = ac_4_pi_c.value
const_8_pi_c = ac_8_pi_c.value
const_h_c_on_4_pi = ac_h_c_on_4_pi.value
const_sqrt_NA_kB_on_c = ac_sqrt_NA_kB_on_c.value
const_sqrt_2_NA_kB_log2_on_c = ac_sqrt_2_NA_kB_log2_on_c.value
const_c_sq_on_2h = ac_c_sq_on_2h.value
const_c_cgs = ac.c.cgs.value


@numba.njit(parallel=True, nogil=True, cache=True, error_model="numpy")
def _filter_trans_batch(
        id_f: npt.NDArray,
        id_i: npt.NDArray,
        a_fi: npt.NDArray[np.float64],
        energy_lookup: npt.NDArray[np.float64],
        wn_min: float,
        wn_max: float,
) -> t.Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Keeps, in input order, the transitions between known states (energy_lookup not NaN) whose wavenumber
    energy_f - energy_i lies in [wn_min, wn_max]. Replaces joining each batch with the states table.

    Each of n_chunks contiguous chunks counts its kept transitions, then writes them at its offset.
    """
    n = id_f.shape[0]
    n_states = energy_lookup.shape[0]
    n_chunks = max(1, min(_DEFAULT_NUM_THREADS, n))
    chunk_len = (n + n_chunks - 1) // n_chunks
    offsets = np.zeros(n_chunks + 1, dtype=np.int64)
    for c in numba.prange(n_chunks):
        kept = 0
        for p in range(c * chunk_len, min(n, (c + 1) * chunk_len)):
            f = id_f[p]
            i = id_i[p]
            # NaN (absent state) fails the range test.
            if 0 <= f < n_states and 0 <= i < n_states and wn_min <= energy_lookup[f] - energy_lookup[i] <= wn_max:
                kept += 1
        offsets[c + 1] = kept
    for c in range(n_chunks):
        offsets[c + 1] += offsets[c]

    n_kept = offsets[n_chunks]
    id_f_out = np.empty(n_kept, dtype=np.int32)
    id_i_out = np.empty(n_kept, dtype=np.int32)
    a_fi_out = np.empty(n_kept, dtype=np.float64)
    energy_fi_out = np.empty(n_kept, dtype=np.float64)
    for c in numba.prange(n_chunks):
        q = offsets[c]
        for p in range(c * chunk_len, min(n, (c + 1) * chunk_len)):
            f = id_f[p]
            i = id_i[p]
            if 0 <= f < n_states and 0 <= i < n_states:
                energy_fi_p = energy_lookup[f] - energy_lookup[i]
                if wn_min <= energy_fi_p <= wn_max:
                    id_f_out[q] = f
                    id_i_out[q] = i
                    a_fi_out[q] = a_fi[p]
                    energy_fi_out[q] = energy_fi_p
                    q += 1
    return id_f_out, id_i_out, a_fi_out, energy_fi_out


def _iter_trans_batches(
        trans_file: pathlib.Path,
        energy_lookup: npt.NDArray[np.float64],
        wn_min: float,
        wn_max: float,
) -> t.Iterator[t.Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.float64],
npt.NDArray[np.float64]]]:
    """
    Yields the (id_f, id_i, A_fi, energy_fi) arrays of the transitions in a line list (.trans text or parquet) that
    fall on the grid, batch by batch (see :func:`_filter_trans_batch`), skipping batches with none.

    Called by :func:`~xsec.NLTEProcessor.compute_rates_profiles`.

    Parameters
    ----------
    trans_file
    energy_lookup : ndarray, shape (n_states + 1,)
        State energy [cm^-1] by state ID, NaN for states not in the states table.
    wn_min
    wn_max
    """
    columns = ["id_f", "id_i", "A_fi"]
    if str(trans_file).endswith(".parquet"):
        with pq.ParquetFile(trans_file) as pq_file:
            for arrow_batch in pq_file.iter_batches(batch_size=_PARQUET_BATCH_SIZE, columns=columns, use_threads=True):
                id_f, id_i, a_fi = (arrow_batch.column(c).to_numpy() for c in columns)
                batch = _filter_trans_batch(
                    id_f=id_f, id_i=id_i, a_fi=a_fi, energy_lookup=energy_lookup, wn_min=wn_min, wn_max=wn_max,
                )
                if batch[0].shape[0] > 0:
                    yield batch
    else:
        ddf = dd.read_csv(
            trans_file,
            sep=r"\s+",
            # pandas' C engine has a fast native whitespace tokenizer for sep=r"\s+" - engine="python" forces the
            # much slower pure-Python regex parser for no benefit here (benchmarked ~4.8x faster with engine="c"
            # on a real 97MB/1.68M-line trans file, identical output), while keeping the same dask
            # blocksize-chunked/parallel read structure.
            engine="c",
            header=None,
            names=columns,
            usecols=[0, 1, 2],
            dtype={"id_f": "int32", "id_i": "int32", "A_fi": "float64"},
            blocksize=_DASK_BLOCK_SIZE,
        )
        for delayed_batch in ddf.to_delayed():
            frame = delayed_batch.compute()
            batch = _filter_trans_batch(
                id_f=frame["id_f"].to_numpy(), id_i=frame["id_i"].to_numpy(), a_fi=frame["A_fi"].to_numpy(),
                energy_lookup=energy_lookup, wn_min=wn_min, wn_max=wn_max,
            )
            if batch[0].shape[0] > 0:
                yield batch


def _read_ahead(batches: t.Iterator[t.Tuple]) -> t.Iterator[t.Tuple]:
    """
    Yields the items of batches (from :func:`_iter_trans_batches`), reading the next one in a background thread while
    the caller works on the current one. The batch loops' numba kernels are nogil=True so that both threads run at
    once (1.1-1.6x on the full H2O line list, CPU and GPU, tests/prefetch_nogil_bench.py); the caller does all
    accumulation, in batch order, so results are bit-identical. Needs a threadsafe numba threading layer (config.py
    sets TBB).
    """
    # set_num_threads is thread-local: the reader thread would otherwise run at NUMBA_NUM_THREADS.
    with ThreadPoolExecutor(max_workers=1, initializer=numba.set_num_threads, initargs=(_DEFAULT_NUM_THREADS,)) as pool:
        future = pool.submit(next, batches, None)
        while (batch := future.result()) is not None:
            future = pool.submit(next, batches, None)
            yield batch


@numba.njit(parallel=True, nogil=True, cache=True, error_model="numpy")
def _assign_band_indices(
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
def _group_by_band(
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


def _bin_edges(wn_arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Edges of the bins centred on the grid points: the midpoints, plus half a step beyond each end."""
    return np.concatenate([[wn_arr[0] - (wn_arr[1] - wn_arr[0]) * 0.5], (wn_arr[:-1] + wn_arr[1:]) * 0.5,
                           [wn_arr[-1] + (wn_arr[-1] - wn_arr[-2]) * 0.5]])


def rebin_spectrum(
        x_in: npt.NDArray[np.float64],
        y_in: npt.NDArray[np.float64],
        new_centers: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Re-bin a high-resolution spectrum onto bins centered at `new_centers` (edges from :func:`_bin_edges`), as the bin
    average of its piecewise log-linear interpolant. `x_in` must be ascending.
    """

    edges = _bin_edges(new_centers)
    # Exact edge values via interpolation.
    y_edges = np.interp(edges, x_in, y_in)
    # Interior points of bin i are x_in[lo[i]:hi[i]], i.e. left < x < right (a slice, as x_in is sorted).
    lo = np.searchsorted(x_in, edges[:-1], side="right")
    hi = np.searchsorted(x_in, edges[1:], side="left")

    y_out = np.zeros_like(new_centers)

    for i in range(len(new_centers)):
        x_bin = np.concatenate(([edges[i]], x_in[lo[i]:hi[i]], [edges[i + 1]]))
        y_bin = np.concatenate(([y_edges[i]], y_in[lo[i]:hi[i]], [y_edges[i + 1]]))
        # Bin average
        bin_integral = loglinear_integral_1d(y_data=y_bin, dx=np.diff(x_bin))
        y_out[i] = bin_integral / (edges[i + 1] - edges[i])

    return y_out


@numba.njit(parallel=True, cache=True, error_model="numpy")
def gamma_lifetime_lookup_from_tau(tau_lookup: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Lifetime (natural) broadening contribution of each state to the Lorentzian HWHM, 1 / (4 pi c tau) [cm^-1].

    The HWHM of a transition is the sum of its upper and lower state contributions. States without a finite, positive
    lifetime (tau is sanitised to 0 where it is missing or infinite, e.g. the ground state) contribute nothing.

    Parameters
    ----------
    tau_lookup : ndarray, shape (n_states + 1,)
        State lifetimes [s].

    Returns
    -------
    gamma_lifetime_lookup : ndarray, shape (n_states + 1,)
    """
    n_states = tau_lookup.shape[0]
    gamma_lifetime_lookup = np.zeros(n_states, dtype=np.float64)
    for s in numba.prange(n_states):
        tau_s = tau_lookup[s]
        if tau_s > 0.0 and np.isfinite(tau_s):
            gamma_lifetime_lookup[s] = 1.0 / (const_4_pi_c * tau_s)
    return gamma_lifetime_lookup


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _profile_windows(
        wn_grid: npt.NDArray[np.float64], cutoff: float
) -> t.Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """
    Grid index window [j_start, j_end) over which a line centred on each grid point is evaluated, for a fixed
    +/- cutoff [cm^-1]. Both bounds are non-decreasing in the grid index.
    """
    n_grid = wn_grid.shape[0]
    j_starts = np.empty(n_grid, dtype=np.int32)
    j_ends = np.empty(n_grid, dtype=np.int32)
    for i in numba.prange(n_grid):
        j_starts[i] = max(0, _binary_search_left(wn_grid, wn_grid[i] - cutoff))
        j_ends[i] = min(n_grid, _binary_search_right(wn_grid, wn_grid[i] + cutoff) + 1)
    return j_starts, j_ends


@numba.njit(cache=True, error_model="numpy")
def _layer_broadening_terms(
        temperatures: npt.NDArray[np.float64],
        pressures: npt.NDArray[np.float64],
        broad_n: npt.NDArray[np.float64],
        broad_gamma: npt.NDArray[np.float64],
        species_mass: float,
        t_ref: float,
        pressure_ref: float,
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Per-layer Voigt broadening terms shared by the CPU and GPU profile kernels.

    Returns
    -------
    inv_doppler : ndarray, shape (n_layers,)
        1 / (sigma_D * sqrt(2)) = inv_doppler[l] / energy for a line or bin at wavenumber energy.
    gamma_pressure : ndarray, shape (n_layers,)
        Pressure-broadening HWHM [cm^-1].
    """
    n_layers = temperatures.shape[0]
    sqrt2_NA_kB_on_c = np.sqrt(2.0) * const_sqrt_NA_kB_on_c
    inv_doppler = np.empty(n_layers, dtype=np.float64)
    gamma_pressure = np.zeros(n_layers, dtype=np.float64)
    for l in range(n_layers):
        temp_l = temperatures[l]
        inv_doppler[l] = 1.0 / (sqrt2_NA_kB_on_c * np.sqrt(temp_l / species_mass))
        for b in range(broad_n.shape[0]):
            gamma_pressure[l] += broad_gamma[b, l] * pressures[l] * (t_ref / temp_l) ** broad_n[b] / pressure_ref
    return inv_doppler, gamma_pressure


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _band_layer_work_order(
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
        Profile windows from :func:`_profile_windows`.

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


# ------------------------------------- EINSTEIN COEFFICIENTS -------------------------------------
@numba.njit(parallel=True, nogil=True, cache=True, error_model="numpy")
def _band_rate_sums(
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


# ------------------------------------- PROCESS BATCH HELPERS -------------------------------------
@numba.njit(parallel=True, cache=True, error_model="numpy")
def _super_line_bins(bin_edges: npt.NDArray[np.float64], energy_fi: npt.NDArray[np.float64]) -> npt.NDArray[np.int32]:
    """Grid bin of each transition's super-line, clamped to the grid."""
    n_trans = energy_fi.shape[0]
    max_idx = bin_edges.shape[0] - 2
    bins = np.empty(n_trans, dtype=np.int32)
    for t in numba.prange(n_trans):
        bin_idx = _binary_search_right(bin_edges, energy_fi[t]) - 1
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
def _accumulate_superline_batch(
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
    absorption-weighted mean lifetime HWHM of each super-line used in :func:`_broaden_superline_buffer`, as for the
    per-band super-lines of :func:`_accumulate_superline_band_batch`.

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
def _accumulate_continuum_superline_batch(
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
    cross-section if desired. Bin-owned tasks as in :func:`_accumulate_superline_batch`.

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
def _accumulate_superline_band_batch(
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
    :func:`_accumulate_continuum_superline_band_batch`.

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
def _accumulate_continuum_superline_band_batch(
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
    desired. (band, bin)-owned tasks as in :func:`_accumulate_superline_band_batch`.

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


# ------------------------------------- COMPACT PROFILE & STORE CLASSES -------------------------------------

@numba.njit(parallel=True, cache=True, error_model="numpy")
def _analyse_profiles(
        profile_matrix_2d: npt.NDArray[np.float64], cutoff: float
) -> t.Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.bool_]]:
    # profile_matrix_2d: (n_profiles, n_grid) - caller reshapes
    n_profiles, n_grid = profile_matrix_2d.shape
    start_idxs = np.empty(n_profiles, dtype=np.int32)
    end_idxs = np.empty(n_profiles, dtype=np.int32)
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
        start_idxs: npt.NDArray[np.int32],
        end_idxs: npt.NDArray[np.int32],
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
        start_idxs: npt.NDArray[np.int32],
        key_idx_array: npt.NDArray[np.int32],
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

    # Profiles are dealt round-robin to numba_num_threads workers, each accumulating into its own buffer row. Not a
    # buffer per thread (numba.get_thread_id()): TBB hands out prange chunks dynamically, so which profiles a thread's
    # row summed changed from call to call, and with it the floating-point rounding of the result (up to ~1e-16
    # relative per call, amplified by the NLTE iteration). The round-robin split makes it reproducible, at the same
    # speed (tests/check_build_xsec_determinism.py).
    for w in numba.prange(numba_num_threads):
        for i in range(w, n_profiles, numba_num_threads):
            # Absorption depends on ID_l population; Emission depends on ID_u population.
            if is_abs:
                pop_val = pop_matrix[key_idx_array[i, 1]]
            else:
                pop_val = pop_matrix[key_idx_array[i, 0]]

            offset_start = offsets[i]
            offset_end = offsets[i + 1]
            profile_len = offset_end - offset_start
            start_idx = start_idxs[i]

            # Element-wise accumulation into the worker's buffer row.
            for j in range(profile_len):
                buffers[w, start_idx + j] += pop_val * profiles[offset_start + j]

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
        key_idx_map: npt.NDArray[np.int32],
        profiles: npt.NDArray[np.float64],
        offsets: npt.NDArray[np.int64],
        start_idxs: npt.NDArray[np.int32],
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


class CompactProfile:
    __slots__ = ["profiles", "offsets", "start_idxs", "key_idx_map", "key_lookup"]

    def __init__(self):
        self.profiles: npt.NDArray[np.float64] | None = None
        # int64: offsets index the concatenated profiles, whose total length (bands x grid points) can exceed 2^31.
        self.offsets: npt.NDArray[np.int64] | None = None
        self.start_idxs: npt.NDArray[np.int32] | None = None
        self.key_idx_map: npt.NDArray[np.int32] | None = None
        self.key_lookup: t.Dict[t.Tuple[int, int], int] = {}

    def finalise_from_buffer(self, profile_matrix: npt.NDArray[np.float64], keys: npt.NDArray[np.int32]):
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
        n_profiles, n_grid = profile_matrix.shape

        if keys.shape[0] != n_profiles:
            raise ValueError(
                f"Number of keys ({keys.shape[0]}) does not match number of profiles ({n_profiles})."
            )

        start_idxs, end_idxs, valid = _analyse_profiles(
            profile_matrix_2d=profile_matrix,
            cutoff=_INTENSITY_CUTOFF,
        )
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
        self.start_idxs = np.empty(n_valid, dtype=np.int32)
        self.key_idx_map = np.empty((n_valid, 2), dtype=np.int32)
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
        if log.isEnabledFor(_LOG_VERBOSE_2):
            log.log(_LOG_VERBOSE_2, f"Finalised CompactProfile with {total_len} points for {n_valid} bands.")

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
        # finalise_from_buffer builds every array contiguous with the kernel's dtype.
        return _build_xsec(
            profiles=self.profiles, offsets=self.offsets, start_idxs=self.start_idxs, key_idx_array=self.key_idx_map,
            pop_matrix=pop_matrix, wn_grid_len=wn_grid.shape[0], is_abs=is_abs,
        )


class ProfileStore:
    """
    Store :class:`tiramisu.nlte.CompactProfile` objects representing the species' absorption, stimulated emission and
    spontaneous emission profiles. Used for bound-bound transitions (including quasi-bound).
    """
    __slots__ = ["n_layers", "abs_profiles", "ste_profiles", "spe_profiles", "_profile_cache"]

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        # Storage for final profiles.
        self.abs_profiles = [CompactProfile() for _ in range(n_layers)]  # Absorption
        self.ste_profiles = [CompactProfile() for _ in range(n_layers)]  # Stimulated Emission
        self.spe_profiles = [CompactProfile() for _ in range(n_layers)]  # Spontaneous Emission
        # Instance-level (not class-level lru_cache!) cache for get_profile: an lru_cache on an instance method
        # binds self into the cache key, which both pins every ProfileStore instance in memory forever and
        # shares one global-size cache across all instances/species/iterations.
        self._profile_cache: t.Dict[t.Tuple[int, t.Tuple[int, int], str], t.Tuple[npt.NDArray[np.float64], int]] = {}

    def finalise_from_buffer(
            self,
            profile_buffer: npt.NDArray[np.float64],
            band_keys: npt.NDArray[np.int32],
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
        if save:
            export_dir.mkdir(parents=True, exist_ok=True)
        #####
        for l in range(self.n_layers):
            for channel, (label, stores) in enumerate(
                    (("abs", self.abs_profiles), ("ste", self.ste_profiles), ("spe", self.spe_profiles))
            ):
                store = stores[l]
                store.finalise_from_buffer(profile_matrix=profile_buffer[channel, :, l, :], keys=band_keys)
                if save:
                    np.savez_compressed(
                        export_dir / f"{species}_L{l:03d}_{label}.npz",
                        profiles=store.profiles,  # flat 1D array of all profile data
                        offsets=store.offsets,  # ragged array boundaries
                        start_idxs=store.start_idxs,  # position of each profile on wn_grid
                        key_idx_map=store.key_idx_map,
                    )

    def get_profile(
            self, layer_idx: int, key: t.Tuple[int, int], profile_type: str
    ) -> t.Tuple[npt.NDArray[np.float64], int]:
        cache_key = (layer_idx, key, profile_type)
        if cache_key in self._profile_cache:
            return self._profile_cache[cache_key]

        if profile_type == "abs":
            result = self.abs_profiles[layer_idx].get_profile(key)
        elif profile_type == "ste":
            result = self.ste_profiles[layer_idx].get_profile(key)
        elif profile_type == "spe":
            result = self.spe_profiles[layer_idx].get_profile(key)
        else:
            raise RuntimeError(f"ProfileStore profile type {profile_type} not implemented.")

        self._profile_cache[cache_key] = result
        return result

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
        spe = self.spe_profiles[layer_idx]
        # Summed over lower states in one pass: row o_idx is the emission from upper state o_idx.
        all_profiles = _rebuild_all_ox_profiles(
            id_agg_cutoff=id_agg_cutoff, key_idx_map=spe.key_idx_map, profiles=spe.profiles, offsets=spe.offsets,
            start_idxs=spe.start_idxs, num_grid=wn_grid.shape[0],
        )
        wn_dx = np.diff(wn_grid.value)
        # Normalised as photon-counting weights: emission profiles are energy rates (h nu per photon), while the cross
        # terms they feed count transitions, so divide by nu first (see "Photon-counting weights" in nlte.py).
        photon_profiles = all_profiles / wn_grid.value[None, :]
        return loglinear_normalise_2d_nonnegative(y_data=photon_profiles, dx=wn_dx) << 1 / wn_grid.unit

    def get_sorted_band_keys(
            self, agg_energies: npt.NDArray[np.float64], id_cutoff: int = 0,
    ) -> t.List[t.Tuple[int, int]]:
        """
        Returns the absorption band keys (id_agg_f, id_agg_i) present in every layer, sorted by energy gap
        (id_agg_f - id_agg_i) ascending, then by id_agg_i, then by id_agg_f.

        Parameters
        ----------
        agg_energies : ndarray
            Array containing the energies for each aggregate state ordered on id_agg (ascending).
        id_cutoff : int
            ID above which states are fixed, so key pairs containing IDs above this should be excluded. If not provided,
            returned keys include band keys involving the cutoff state.

        Returns
        -------
        List of (id_agg_f, id_agg_i) tuples in the above order.
        """
        key_sets = [set(compact.key_lookup.keys()) for compact in self.abs_profiles]
        common_keys = key_sets[0].intersection(*key_sets[1:])
        if id_cutoff > 0:
            common_keys = [k for k in common_keys if id_cutoff >= k[0] > k[1] >= 0]
        else:
            common_keys = [k for k in common_keys if k[0] > k[1] >= 0]

        if not common_keys:
            return []

        # Vectorized equivalent of sorting by (e_i, nu) with a per-comparison Python closure: gather id_f/id_i via
        # fancy indexing and let np.lexsort do the ordering (primary key last: e_i, then nu) instead.
        common_keys_arr = np.array(common_keys, dtype=np.int32)
        id_f_arr = common_keys_arr[:, 0]
        id_i_arr = common_keys_arr[:, 1]
        e_i_arr = agg_energies[id_i_arr]  # lower state energy
        e_f_arr = agg_energies[id_f_arr]  # upper state energy
        nu_arr = e_f_arr - e_i_arr  # band centre frequency

        order = np.lexsort((nu_arr, e_i_arr))
        return [(int(id_f_arr[i]), int(id_i_arr[i])) for i in order]


class ContinuumProfileStore:
    __slots__ = [
        "n_layers", "abs_profiles",
        # "ste_profiles", "spe_profiles"
        "_profile_cache",
    ]

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        # Storage for final profiles.
        self.abs_profiles = [CompactProfile() for _ in range(n_layers)]  # Absorption
        # Instance-level cache (see ProfileStore.get_profile for why not a class-level lru_cache).
        self._profile_cache: t.Dict[t.Tuple[int, int, str], t.Tuple[npt.NDArray, int]] = {}

    def finalise_from_buffer(self, profile_buffer: npt.NDArray[np.float64], band_keys: npt.NDArray[np.int32]) -> None:
        """

        Parameters
        ----------
        profile_buffer : ndarray, shape (1, n_bands, n_layers, n_grid)
        band_keys : ndarray, shape (n_bands, 2)
            Contains the (id_f_agg, id_i_agg) pairs for each band.
        """
        for l in range(self.n_layers):
            self.abs_profiles[l].finalise_from_buffer(profile_matrix=profile_buffer[0, :, l, :], keys=band_keys)

    def get_profile(self, layer_idx: int, key: int, profile_type: str) -> t.Tuple[npt.NDArray, int]:
        cache_key = (layer_idx, key, profile_type)
        if cache_key in self._profile_cache:
            return self._profile_cache[cache_key]

        if profile_type == "abs":
            # Continuum profiles are stored with an arbitrary index for the upper continuum state (-1).
            result = self.abs_profiles[layer_idx].get_profile((-1, key))
            self._profile_cache[cache_key] = result
            return result
        else:
            raise RuntimeError(f"ContinuumProfileStore profile type {profile_type} not implemented.")

    def build_abs(
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
        temperature_profile: u.Quantity,
        pressure_profile: u.Quantity,
        wn_grid: u.Quantity,
        species_mass: float,
        do_super_lines: bool,
        broadening_params: t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] = None,
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
    do_super_lines
    broadening_params

    Returns
    -------
    abs_xsec : ndarray, shape (n_grid,)
    emi_xsec : ndarray, shape (n_grid,)
    """
    # Plain float arrays - no astropy units - passed directly to Numba.
    temperature_slice = temperature_profile[n_lte_layers:].value  # (n_nlte_layers,)
    pressure_slice = pressure_profile[n_lte_layers:].value  # (n_nlte_layers,)

    wn_arr = wn_grid.value
    wn_min = wn_arr[0]
    wn_max = wn_arr[-1]
    # Bin edges for finding trans bins (super-lines).
    n_grid = wn_grid.shape[0]
    bin_edges = _bin_edges(wn_arr)

    # States lookup - much more memory efficient than duplicating it all with polars joins!
    state_ids = states["id"].to_numpy()
    max_state_id = int(state_ids.max())
    # State ID lookup offset by 1 as IDs are 1-indexed! NaN energy marks IDs missing from the states table.
    energy_lookup = np.full(max_state_id + 1, np.nan, dtype=np.float64)
    energy_lookup[state_ids] = states["energy"].to_numpy()
    n_lookup = np.zeros(
        (max_state_id + 1, n_nlte_layers),
        dtype=np.float64,
    )
    # We need to check for the existence of the nLTE column here because n_nlte_Lx doesn't exist in layers that are
    # marked as skipped due to negligible VMR; these layers are unchanged so we use the LTE values. Resolve the
    # per-layer column choice first, then pull all layers out of polars in one bulk select/to_numpy call.
    cols_to_use = [
        f"n_nlte_L{n_lte_layers + l}" if f"n_nlte_L{n_lte_layers + l}" in states.columns else f"n_L{n_lte_layers + l}"
        for l in range(n_nlte_layers)
    ]
    n_lookup[state_ids, :] = states.select(cols_to_use).to_numpy()

    g_lookup = np.zeros(max_state_id + 1, dtype=np.float64)
    g_lookup[state_ids] = states["g"].to_numpy()
    inv_g_lookup = np.zeros_like(g_lookup)
    inv_g_lookup[1:] = 1.0 / g_lookup[1:]

    tau_lookup = np.zeros(max_state_id + 1, dtype=np.float64)
    tau_lookup[state_ids] = states["tau"].to_numpy()
    gamma_lifetime_lookup = gamma_lifetime_lookup_from_tau(tau_lookup)

    broad_n = np.zeros(1, dtype=np.float64)
    broad_gamma = np.zeros((1, n_nlte_layers), dtype=np.float64)
    if broadening_params is not None:
        broad_n = broadening_params[1]
        broad_gamma = broadening_params[0][:, n_lte_layers:]

    # Super-lines accumulate net absorption and emission, plus sum(abs * gamma_lifetime) and sum(abs) in channels 2 and
    # 3; direct profiles only the first two.
    profile_buffer = np.zeros((4 if do_super_lines else 2, n_nlte_layers, n_grid), dtype=np.float64)
    # Super-lines are accumulated on the GPU when that pays off (profiles_gpu.gpu_pays_off): the buffer stays on the
    # device for the whole batch loop and is copied back once.
    d_buffer = None
    if do_super_lines and config.USE_GPU:
        from . import profiles_gpu
        if profiles_gpu.gpu_pays_off(trans_files=trans_files, n_layers=n_nlte_layers,
                                     buffer_bytes=profile_buffer.nbytes, label="abs_emi_xsec"):
            d_lookups = profiles_gpu.to_device(bin_edges=bin_edges, n_frac=n_lookup, g=g_lookup, inv_g=inv_g_lookup,
                                               gamma_lifetime=gamma_lifetime_lookup)
            d_buffer = profiles_gpu._device_zeros(profile_buffer.shape)

    process_time = time.perf_counter()

    for trans_file in trans_files:
        log.info(f"Processing file {trans_file}.")
        for id_f, id_i, a_fi, energy_fi in _read_ahead(batches=_iter_trans_batches(
                trans_file=trans_file,
                energy_lookup=energy_lookup,
                wn_min=wn_min,
                wn_max=wn_max,
        )):
            if d_buffer is not None:
                profiles_gpu.accumulate_superlines(profile_buffer=d_buffer, lookups=d_lookups, id_f=id_f, id_i=id_i,
                                                   a_fi=a_fi, energy_fi=energy_fi)
            elif do_super_lines:
                _accumulate_superline_batch(
                    profile_buffer=profile_buffer,
                    bin_edges=bin_edges,
                    n_lookup=n_lookup,
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
                _abs_emi_sampled_voigt(
                    profile_buffer=profile_buffer,
                    wn_grid=wn_arr,
                    id_i=id_i,
                    id_f=id_f,
                    n_lookup=n_lookup,
                    g_lookup=g_lookup,
                    inv_g_lookup=inv_g_lookup,
                    gamma_lifetime_lookup=gamma_lifetime_lookup,
                    a_fi=a_fi,
                    energy_fi=energy_fi,
                    temperatures=temperature_slice,
                    pressures=pressure_slice,
                    broad_n=broad_n,
                    broad_gamma=broad_gamma,
                    species_mass=species_mass,
                )
    # Finalise.
    if d_buffer is not None:
        profile_buffer = d_buffer.copy_to_host()
    if do_super_lines:
        abs_xsec, emi_xsec = _broaden_superline_buffer(
            profile_buffer=profile_buffer,
            wn_grid=wn_arr,
            temperatures=temperature_slice,
            pressures=pressure_slice,
            broad_n=broad_n,
            broad_gamma=broad_gamma,
            species_mass=species_mass,
        )
    else:
        abs_xsec = profile_buffer[0]
        emi_xsec = profile_buffer[1]
    if log.isEnabledFor(_LOG_VERBOSE_2):
        log.log(_LOG_VERBOSE_2, f"New rates/profiles duration = {time.perf_counter() - process_time:.3f}.")
    return abs_xsec, emi_xsec


def continuum_xsec(
        states: pl.DataFrame,
        cont_states: pl.DataFrame,
        cont_trans_files: t.List[pathlib.Path],
        n_lte_layers: int,
        n_nlte_layers: int,
        temperature_profile: u.Quantity,
        wn_grid: u.Quantity,
        species_mass: float,
        reduced_mass: float,
        cont_box_length: float,
        do_super_lines: bool,
) -> npt.NDArray[np.float64]:
    """

    Parameters
    ----------
    states
    cont_states
    cont_trans_files
    n_lte_layers
    n_nlte_layers
    temperature_profile
    wn_grid
    species_mass
    reduced_mass
    cont_box_length
    do_super_lines

    Returns
    -------
    abs_xsec : ndarray, shape (n_grid,)
    """
    # Plain float arrays - no astropy units - passed directly to Numba.
    temperature_slice = temperature_profile[n_lte_layers:].value  # (n_nlte_layers,)

    # We need to check for the existence of the nLTE column here because n_nlte_Lx doesn't exist in layers that are
    # marked as skipped due to negligible VMR; these layers are unchanged so we use the LTE values. Resolve the
    # per-layer column choice first, then pull all layers out of polars in one bulk select/to_numpy call.
    cols_to_use = [
        f"n_nlte_L{n_lte_layers + l}" if f"n_nlte_L{n_lte_layers + l}" in states.columns else f"n_L{n_lte_layers + l}"
        for l in range(n_nlte_layers)
    ]
    # g and v are stored on cont_states.
    cont_states = cont_states.join(states.select(["id"] + cols_to_use), on="id", how="left")

    wn_arr = wn_grid.value
    wn_min = wn_arr[0]
    wn_max = wn_arr[-1]
    # Bin edges for finding trans bins (super-lines).
    n_grid = wn_grid.shape[0]
    bin_edges = _bin_edges(wn_arr)

    # States lookup - much more memory efficient than duplicating it all with polars joins!
    state_ids = cont_states["id"].to_numpy()
    max_state_id = int(state_ids.max())
    # State ID lookup offset by 1 as IDs are 1-indexed! NaN energy marks IDs missing from the states table.
    energy_lookup = np.full(max_state_id + 1, np.nan, dtype=np.float64)
    energy_lookup[state_ids] = cont_states["energy"].to_numpy()
    n_lookup = np.zeros(
        (max_state_id + 1, n_nlte_layers),
        dtype=np.float64,
    )
    n_lookup[state_ids, :] = cont_states.select(cols_to_use).to_numpy()

    g_lookup = np.zeros(max_state_id + 1, dtype=np.float64)
    g_lookup[state_ids] = cont_states["g"].to_numpy()
    zero_g_map = g_lookup == 0
    inv_g_lookup = np.zeros_like(g_lookup)
    inv_g_lookup[~zero_g_map] = 1.0 / g_lookup[~zero_g_map]

    v_lookup = np.zeros(max_state_id + 1, dtype=np.float64)
    v_lookup[state_ids] = cont_states["v"].to_numpy()

    # Super-lines accumulator.
    profile_buffer = np.zeros((2, n_nlte_layers, n_grid), dtype=np.float64)

    process_time = time.perf_counter()

    for trans_file in cont_trans_files:
        log.info(f"Processing file {trans_file}.")
        for id_f, id_i, a_fi, energy_fi in _read_ahead(batches=_iter_trans_batches(
                trans_file=trans_file,
                energy_lookup=energy_lookup,
                wn_min=wn_min,
                wn_max=wn_max,
        )):
            if do_super_lines:
                _accumulate_continuum_superline_batch(
                    profile_buffer=profile_buffer,
                    bin_edges=bin_edges,
                    n_lookup=n_lookup,
                    g_lookup=g_lookup,
                    inv_g_lookup=inv_g_lookup,
                    v_lookup=v_lookup,
                    id_f=id_f,
                    id_i=id_i,
                    a_fi=a_fi,
                    energy_fi=energy_fi,
                    reduced_mass=reduced_mass,
                    box_length=cont_box_length,
                    n_workers=_DEFAULT_NUM_THREADS,
                )
            else:
                _continuum_sampled_gauss(
                    profile_buffer=profile_buffer,
                    wn_grid=wn_arr,
                    id_i=id_i,
                    id_f=id_f,
                    n_lookup=n_lookup,
                    g_lookup=g_lookup,
                    inv_g_lookup=inv_g_lookup,
                    v_lookup=v_lookup,
                    a_fi=a_fi,
                    energy_fi=energy_fi,
                    temperatures=temperature_slice,
                    species_mass=species_mass,
                    reduced_mass=reduced_mass,
                    box_length=cont_box_length,
                )
    # Finalise.
    if do_super_lines:
        abs_xsec = _broaden_continuum_superline_buffer(
            profile_buffer=profile_buffer,
            wn_grid=wn_arr,
            temperatures=temperature_slice,
            species_mass=species_mass,
        )
    else:
        abs_xsec = profile_buffer[0]
    if log.isEnabledFor(_LOG_VERBOSE_2):
        log.log(_LOG_VERBOSE_2, f"New rates/profiles duration = {time.perf_counter() - process_time:.3f}.")
    return abs_xsec


# ------------------------------------- NUMBA XSEC CALCULATIONS -------------------------------------


@numba.njit(cache=True, error_model="numpy", inline="always")
def _binary_search_left(arr: npt.NDArray[np.float64], value: np.float64, start: np.int32 = 0) -> np.int32:
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
def _binary_search_right(arr: npt.NDArray[np.float64], value: np.float64, start: np.int32 = 0) -> np.int32:
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

    Relative accuracy ~1e-4 (measured against scipy.special.wofz). x and y are the real and imaginary parts of
    z = (nu - nu0) / (sigma_D * sqrt(2)), where y > 0 always (it is the ratio of Lorentzian to Gaussian halfwidths).

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
    of each (layer, bin) being channel 2 / channel 3 of profile_buffer (see :func:`_accumulate_superline_batch`).

    Parameters
    ----------
    profile_buffer : (4, n_layers, n_grid)
        Accumulated super-line coefficients per layer per grid bin, from :func:`_accumulate_superline_batch`. Net
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
    inv_doppler, gamma_pressure = _layer_broadening_terms(
        temperatures=temperatures, pressures=pressures, broad_n=broad_n, broad_gamma=broad_gamma,
        species_mass=species_mass, t_ref=t_ref, pressure_ref=pressure_ref,
    )
    j_starts, j_ends = _profile_windows(wn_grid=wn_grid, cutoff=cutoff)

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
                voigt_val = _voigt_humlicek_w(x_ij, y_il) * norm

                out_abs[l, j] += abs_li * voigt_val
                out_emi[l, j] += emi_li * voigt_val
    return out_abs, out_emi


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _broaden_continuum_superline_buffer(
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
        absorption-weighted lifetime HWHM, from :func:`_accumulate_superline_band_batch`.
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
    inv_doppler, gamma_pressure = _layer_broadening_terms(
        temperatures=temperatures, pressures=pressures, broad_n=broad_n, broad_gamma=broad_gamma,
        species_mass=species_mass, t_ref=t_ref, pressure_ref=pressure_ref,
    )

    j_starts, j_ends = _profile_windows(wn_grid=wn_grid, cutoff=cutoff)

    # Each (band, layer) pair writes only its own output row buffer_out[:, band, l, :], so pairs can run in parallel
    # without two threads ever writing the same output point. Pairs differ enormously in cost (a band may occupy a few
    # bins or the whole grid), and a prange over the pairs would give each thread an equal-length contiguous block of
    # them, leaving the thread that gets the busiest bands with several times the average work. Instead, the pairs are
    # sorted by estimated cost and dealt round-robin to n_workers workers, heaviest first (worker w takes sorted pairs
    # w, w + n_workers, w + 2 * n_workers, ...), which keeps every worker's total within a few percent of the mean.
    order, cost = _band_layer_work_order(profile_buffer=profile_buffer, j_starts=j_starts, j_ends=j_ends)
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
                    voigt_val = _voigt_humlicek_w(x_ij, y_bli) * norm
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
    The GPU equivalent is :meth:`profiles_gpu.DeviceProfileBuffer.broaden_superlines`."""
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
    j_start = max(0, _binary_search_left(wn_grid, energy_i - cutoff))
    j_end = min(wn_grid.shape[0], _binary_search_right(wn_grid, energy_i + cutoff) + 1)
    return j_start, j_end


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _broaden_continuum_superline_band_buffer(
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


# ------------------------------------- ALL LAYER, SAMPLED VOIGT CALCULATIONS -------------------------------------
@numba.njit(parallel=True, nogil=True, cache=True, error_model="numpy")
def _band_profile_sampled_voigt(
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
        Global band index of each band in the batch, from :func:`_group_by_band`.
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
    inv_doppler, gamma_pressure = _layer_broadening_terms(
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
                j_start = max(0, _binary_search_left(wn_grid, transition_min))
                j_end = min(n_grid, _binary_search_right(wn_grid, transition_max) + 1)

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
                    voigt_val = _voigt_humlicek_w(x_tj, y_tl) * norm

                    abs_bl[j] += abs_tl * voigt_val
                    ste_bl[j] += ste_tl * voigt_val
                    spe_bl[j] += spe_tl * voigt_val


@numba.njit(parallel=True, nogil=True, cache=True, error_model="numpy")
def _abs_emi_sampled_voigt(
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
    inv_doppler, gamma_pressure = _layer_broadening_terms(
        temperatures=temperatures, pressures=pressures, broad_n=broad_n, broad_gamma=broad_gamma,
        species_mass=species_mass, t_ref=t_ref, pressure_ref=pressure_ref,
    )

    abs_prefactor = np.empty(n_trans, dtype=np.float64)
    emi_prefactor = np.empty(n_trans, dtype=np.float64)
    j_starts = np.empty(n_trans, dtype=np.int32)
    j_ends = np.empty(n_trans, dtype=np.int32)
    for t in numba.prange(n_trans):
        energy_fi_t = energy_fi[t]
        j_starts[t] = max(0, _binary_search_left(wn_grid, energy_fi_t - cutoff))
        j_ends[t] = min(n_grid, _binary_search_right(wn_grid, energy_fi_t + cutoff) + 1)
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
                voigt_val = _voigt_humlicek_w(x_tj, y_tl) * norm

                abs_l[j] += abs_tl * voigt_val
                emi_l[j] += emi_tl * voigt_val


# ------------------------------------- ALL LAYER, SAMPLED GAUSSIAN CALCULATIONS -------------------------------------
@numba.njit(parallel=True, nogil=True, cache=True, error_model="numpy")
def _continuum_band_profile_sampled_gauss_layered(
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
        Global band index of each band in the batch, from :func:`_group_by_band`.
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

                j_start = max(0, _binary_search_left(wn_grid, transition_min))
                j_end = min(n_grid, _binary_search_right(wn_grid, transition_max) + 1)

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
def _continuum_sampled_gauss(
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

            j_start = max(0, _binary_search_left(wn_grid, transition_min))
            j_end = min(n_grid, _binary_search_right(wn_grid, transition_max) + 1)

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
