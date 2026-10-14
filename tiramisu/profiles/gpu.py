"""
GPU (numba.cuda) implementation of the band-profile construction in
:meth:`~tiramisu.nlte.NLTEProcessor.compute_rates_profiles`: super-line accumulation and broadening, and direct (per-line)
sampled Voigt profiles.

:class:`DeviceProfileBuffer` keeps the band profile buffer and the per-state lookup tables on the device for the whole
transition-batch loop, so each batch only uploads its own transition arrays. At the end the buffer is either broadened
on the device (:meth:`DeviceProfileBuffer.broaden_superlines`) or copied back to the host once
(:meth:`DeviceProfileBuffer.to_host`). This module is imported lazily (from nlte.py and profiles/cross_sections.py),
so CPU-only runs have no import cost or CUDA-toolkit dependency.

:func:`accumulate_superlines` adds a transition batch to a device super-line buffer of either path: the band buffer of
compute_rates_profiles or the band-less buffer of profiles.abs_emi_xsec. :func:`gpu_pays_off` is the cost model both
paths use to choose the GPU (see config._GPU_LINE_LAYER_SAVING_S).

Kernel design:

- Super-line accumulation is a histogram: many transitions land in the same (band, layer, bin) cell. Adding each line
  with an atomic is slower than the CPU (the atomics serialise), so the batch is first counting-sorted by cell key and
  cut into segments of at most _SEGMENT_MAX_LINES lines of one key; a thread per (segment, layer) sums its segment's
  lines. Threads of a segment differ only in layer, so their population reads are contiguous. The sort is stable (lines
  of a key keep their input order) and no float atomics are used: a key's only segment adds to its cell directly, and
  the segments of a multi-segment key write partials that a second kernel adds in segment order. Runs are therefore
  reproducible bit for bit; they differ from the CPU kernels at rounding level, which group the sums differently.
- Direct sampled Voigt profiles parallelise over (band, layer) pairs, each owning its output row, so no atomics.
- Super-line broadening gives each thread one (band, layer) output row, like the CPU kernel. Threads are ordered by
  estimated cost so the threads of a warp have similar work.

Kernel array parameters are annotated with the numpy element type of the arrays passed, which are device arrays; numba
ignores the annotations and compiles for the types it is given.
"""
import logging
import math
import os
import pathlib
import time

import numba
import numpy as np
from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numpy import typing as npt
from pyarrow import parquet as pq

from .. import config
from ..config import _GPU_THREADS_PER_BLOCK, _LOG_VERBOSE_2, _PARQUET_BATCH_SIZE
from ..constants import const_8_pi_c, const_h_c_on_4_pi
from .lineshape import profile_windows

log = logging.getLogger(__name__)

# Device memory kept free when sizing allocations (bytes).
_DEVICE_MEMORY_MARGIN = 256 * 1024 ** 2
# Device memory per transition while accumulate_superlines runs on a batch (inputs, band index, key, position,
# key-sorted copies), reserved next to the band buffer so a full-size batch always fits. The multi-segment partial sums
# add at most n_layers / 4 bytes per transition (rows <= n_trans / 128, 32 bytes per layer each).
_DEVICE_BYTES_PER_TRANSITION = 80
# Most lines one thread sums per segment (64-256 measured equally fast on the H2O line list).
_SEGMENT_MAX_LINES = 256


# ------------------------------------- DEVICE HELPERS -------------------------------------

@cuda.jit(device=True, inline=True, cache=True)
def _cexp_gpu(u: complex) -> complex:
    """exp(u) for complex u — numba's CUDA target does not support np.exp on complex128."""
    r = math.exp(u.real)
    return r * math.cos(u.imag) + 1j * r * math.sin(u.imag)


@cuda.jit(device=True, inline=True, cache=True)
def _voigt_humlicek_w_gpu(x: float, y: float) -> float:
    """Device-side port of lineshape.voigt_humlicek_w — see that function's docstring."""
    t_ = y - 1j * x
    s = abs(x) + y

    if s >= 15.0:
        w = t_ * 0.5641896 / (0.5 + t_ * t_)
    elif s >= 5.5:
        u = t_ * t_
        w = t_ * (1.410474 + u * 0.5641896) / (0.75 + u * (3.0 + u))
    elif y >= 0.195 * abs(x) - 0.176:
        w = (16.4955 + t_ * (
                20.20933 + t_ * (11.96482 + t_ * (3.778987 + t_ * 0.5642236)))) / (
                    16.4955 + t_ * (38.82363 + t_ * (
                    39.27121 + t_ * (21.69274 + t_ * (6.699398 + t_))))
            )
    else:
        u = t_ * t_
        w = _cexp_gpu(u) - t_ * (
                36183.31 - u * (3321.99 - u * (1540.787 - u * (
                219.031 - u * (35.7668 - u * (1.320522 - u * 0.56419)))))
        ) / (
                    32066.6 - u * (24322.84 - u * (
                    9022.228 - u * (2186.181 - u * (
                    364.2191 - u * (61.57037 - u * (1.841439 - u))))))
            )

    return w.real


@cuda.jit(device=True, inline=True, cache=True)
def _binary_search_left_gpu(arr: npt.NDArray[np.float64], value: float) -> int:
    left = 0
    right = arr.shape[0]
    while left < right:
        mid = (left + right) >> 1
        if arr[mid] < value:
            left = mid + 1
        else:
            right = mid
    return left


@cuda.jit(device=True, inline=True, cache=True)
def _binary_search_right_gpu(arr: npt.NDArray[np.float64], value: float) -> int:
    left = 0
    right = arr.shape[0]
    while left < right:
        mid = (left + right) >> 1
        if arr[mid] <= value:
            left = mid + 1
        else:
            right = mid
    return left


def _launch_dims(total_threads: int) -> tuple[int, int]:
    threads_per_block = _GPU_THREADS_PER_BLOCK
    blocks = max(1, (total_threads + threads_per_block - 1) // threads_per_block)
    return blocks, threads_per_block


# ------------------------------------- ACCUMULATE (SORTED SEGMENTS) -------------------------------------

@cuda.jit(cache=True)
def _key_count_kernel(
        bin_edges: npt.NDArray[np.float64],
        band_indices: npt.NDArray[np.int32],
        energy_fi: npt.NDArray[np.float64],
        n_grid: int,
        keys: npt.NDArray[np.int32],
        counts: npt.NDArray[np.int32],
) -> None:
    """Cell key of each transition (band * n_grid + super-line bin, or the bin alone if band_indices is empty, as
    superline._super_line_bins) and the number of transitions per key."""
    max_idx = bin_edges.shape[0] - 2
    with_bands = band_indices.shape[0] > 0
    for t in range(cuda.grid(1), energy_fi.shape[0], cuda.gridsize(1)):
        key = min(max(_binary_search_right_gpu(bin_edges, energy_fi[t]) - 1, 0), max_idx)
        if with_bands:
            key += band_indices[t] * n_grid
        keys[t] = key
        cuda.atomic.add(counts, key, 1)


@numba.njit(nogil=True, cache=True)
def _stable_positions(keys: npt.NDArray[np.int32], offsets: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
    """Position of each transition in key order, a key's transitions kept in input order (a stable counting sort), so
    that every segment sums its lines in the same order on every run."""
    cursor = offsets.copy()
    positions = np.empty(keys.shape[0], dtype=np.int32)
    for t in range(keys.shape[0]):
        key = keys[t]
        positions[t] = cursor[key]
        cursor[key] += 1
    return positions


@cuda.jit(cache=True)
def _scatter_kernel(
        positions: npt.NDArray[np.int32],
        g_lookup: npt.NDArray[np.float64],
        inv_g_lookup: npt.NDArray[np.float64],
        gamma_lifetime_lookup: npt.NDArray[np.float64],
        id_f: npt.NDArray[np.int32],
        id_i: npt.NDArray[np.int32],
        a_fi: npt.NDArray[np.float64],
        energy_fi: npt.NDArray[np.float64],
        s_id_f: npt.NDArray[np.int32],
        s_id_i: npt.NDArray[np.int32],
        s_abs: npt.NDArray[np.float64],
        s_ste: npt.NDArray[np.float64],
        s_spe: npt.NDArray[np.float64],
        s_gamma: npt.NDArray[np.float64],
) -> None:
    """Writes each transition, with its layer-independent prefactors, at its position in key order."""
    for t in range(cuda.grid(1), positions.shape[0], cuda.gridsize(1)):
        p = positions[t]
        id_f_t = id_f[t]
        id_i_t = id_i[t]
        energy_fi_t = energy_fi[t]
        ste_prefactor_t = a_fi[t] / (const_8_pi_c * energy_fi_t * energy_fi_t)
        s_id_f[p] = id_f_t
        s_id_i[p] = id_i_t
        s_abs[p] = ste_prefactor_t * g_lookup[id_f_t] * inv_g_lookup[id_i_t]
        s_ste[p] = ste_prefactor_t
        s_spe[p] = a_fi[t] * energy_fi_t * const_h_c_on_4_pi
        s_gamma[p] = gamma_lifetime_lookup[id_f_t] + gamma_lifetime_lookup[id_i_t]


@cuda.jit(cache=True)
def _accumulate_segments_band_kernel(
        profile_buffer: npt.NDArray[np.float64],
        partials: npt.NDArray[np.float64],
        n_frac_lookup: npt.NDArray[np.float64],
        seg_start: npt.NDArray[np.int32],
        seg_end: npt.NDArray[np.int32],
        seg_key: npt.NDArray[np.int32],
        seg_partial: npt.NDArray[np.int32],
        s_id_f: npt.NDArray[np.int32],
        s_id_i: npt.NDArray[np.int32],
        s_abs: npt.NDArray[np.float64],
        s_ste: npt.NDArray[np.float64],
        s_spe: npt.NDArray[np.float64],
        s_gamma: npt.NDArray[np.float64],
) -> None:
    """Band buffer (4, n_bands, n_layers, n_grid): absorption, stimulated and spontaneous emission, and the
    absorption-weighted lifetime HWHM, as superline.accumulate_superline_band_batch. The only segment of a key adds to
    the buffer; a segment of a multi-segment key writes its partial row (seg_partial), for _reduce_partials_kernel."""
    n_layers, n_grid = n_frac_lookup.shape[1], profile_buffer.shape[3]
    for flat in range(cuda.grid(1), seg_start.shape[0] * n_layers, cuda.gridsize(1)):
        s = flat // n_layers
        l = flat - s * n_layers
        abs_sum = ste_sum = spe_sum = gamma_sum = 0.0
        for p in range(seg_start[s], seg_end[s]):
            abs_tl = s_abs[p] * n_frac_lookup[s_id_i[p], l]
            n_f_tl = n_frac_lookup[s_id_f[p], l]
            abs_sum += abs_tl
            ste_sum += s_ste[p] * n_f_tl
            spe_sum += s_spe[p] * n_f_tl
            gamma_sum += abs_tl * s_gamma[p]
        q = seg_partial[s]
        if q < 0:
            band = seg_key[s] // n_grid
            bin_idx = seg_key[s] - band * n_grid
            profile_buffer[0, band, l, bin_idx] += abs_sum
            profile_buffer[1, band, l, bin_idx] += ste_sum
            profile_buffer[2, band, l, bin_idx] += spe_sum
            profile_buffer[3, band, l, bin_idx] += gamma_sum
        else:
            partials[q, l, 0] = abs_sum
            partials[q, l, 1] = ste_sum
            partials[q, l, 2] = spe_sum
            partials[q, l, 3] = gamma_sum


@cuda.jit(cache=True)
def _accumulate_segments_xsec_kernel(
        profile_buffer: npt.NDArray[np.float64],
        partials: npt.NDArray[np.float64],
        n_lookup: npt.NDArray[np.float64],
        seg_start: npt.NDArray[np.int32],
        seg_end: npt.NDArray[np.int32],
        seg_key: npt.NDArray[np.int32],
        seg_partial: npt.NDArray[np.int32],
        s_id_f: npt.NDArray[np.int32],
        s_id_i: npt.NDArray[np.int32],
        s_abs: npt.NDArray[np.float64],
        s_ste: npt.NDArray[np.float64],
        s_spe: npt.NDArray[np.float64],
        s_gamma: npt.NDArray[np.float64],
) -> None:
    """Band-less buffer (4, n_layers, n_grid): net absorption, spontaneous emission, sum(abs * gamma_lifetime) and
    sum(abs), as superline.accumulate_superline_batch; segments and partials as _accumulate_segments_band_kernel."""
    n_layers = n_lookup.shape[1]
    for flat in range(cuda.grid(1), seg_start.shape[0] * n_layers, cuda.gridsize(1)):
        s = flat // n_layers
        l = flat - s * n_layers
        net_sum = spe_sum = gamma_sum = abs_sum = 0.0
        for p in range(seg_start[s], seg_end[s]):
            abs_tl = s_abs[p] * n_lookup[s_id_i[p], l]
            n_f_tl = n_lookup[s_id_f[p], l]
            net_sum += abs_tl - s_ste[p] * n_f_tl
            spe_sum += s_spe[p] * n_f_tl
            gamma_sum += abs_tl * s_gamma[p]
            abs_sum += abs_tl
        q = seg_partial[s]
        if q < 0:
            bin_idx = seg_key[s]
            profile_buffer[0, l, bin_idx] += net_sum
            profile_buffer[1, l, bin_idx] += spe_sum
            profile_buffer[2, l, bin_idx] += gamma_sum
            profile_buffer[3, l, bin_idx] += abs_sum
        else:
            partials[q, l, 0] = net_sum
            partials[q, l, 1] = spe_sum
            partials[q, l, 2] = gamma_sum
            partials[q, l, 3] = abs_sum


@cuda.jit(cache=True)
def _reduce_partials_kernel(
        profile_buffer: npt.NDArray[np.float64],
        partials: npt.NDArray[np.float64],
        multi_first: npt.NDArray[np.int32],
        multi_count: npt.NDArray[np.int32],
        multi_key: npt.NDArray[np.int32],
) -> None:
    """Adds each multi-segment key's partial rows, in segment order, to its cell: one thread per (key, layer). The
    buffer is (4, n_bands, n_layers, n_grid), a band-less buffer viewed with n_bands = 1."""
    n_layers, n_grid = partials.shape[1], profile_buffer.shape[3]
    for flat in range(cuda.grid(1), multi_key.shape[0] * n_layers, cuda.gridsize(1)):
        m = flat // n_layers
        l = flat - m * n_layers
        band = multi_key[m] // n_grid
        bin_idx = multi_key[m] - band * n_grid
        for c in range(4):
            total = 0.0
            for q in range(multi_first[m], multi_first[m] + multi_count[m]):
                total += partials[q, l, c]
            profile_buffer[c, band, l, bin_idx] += total


@numba.njit(nogil=True, cache=True)
def _segments(
        counts: npt.NDArray[np.int32],
        max_lines: int,
) -> tuple[npt.NDArray[np.int32], ...]:
    """
    Offset of each key's transitions in key order, segments of at most max_lines transitions of one key, and which
    segments share a key.

    Returns
    -------
    offsets : ndarray, shape (n_keys,)
    seg_start, seg_end : ndarray, shape (n_segments,)
        [seg_start, seg_end) positions of each segment in key order.
    seg_key : ndarray, shape (n_segments,)
    seg_partial : ndarray, shape (n_segments,)
        Partial-sum row of a segment of a multi-segment key (rows of one key consecutive, in segment order), or -1 for
        a key's only segment.
    multi_first, multi_count, multi_key : ndarray, shape (n_multi_keys,)
        First partial row, number of rows and key of each multi-segment key.
    """
    n_keys = counts.shape[0]
    offsets = np.empty(n_keys, dtype=np.int32)
    total = n_segments = n_multi = 0
    for k in range(n_keys):
        offsets[k] = total
        total += counts[k]
        n_seg = (counts[k] + max_lines - 1) // max_lines
        n_segments += n_seg
        if n_seg > 1:
            n_multi += 1
    seg_start = np.empty(n_segments, dtype=np.int32)
    seg_end = np.empty(n_segments, dtype=np.int32)
    seg_key = np.empty(n_segments, dtype=np.int32)
    seg_partial = np.empty(n_segments, dtype=np.int32)
    multi_first = np.empty(n_multi, dtype=np.int32)
    multi_count = np.empty(n_multi, dtype=np.int32)
    multi_key = np.empty(n_multi, dtype=np.int32)
    s = q = m = 0
    for k in range(n_keys):
        n_seg = (counts[k] + max_lines - 1) // max_lines
        if n_seg > 1:
            multi_first[m] = q
            multi_count[m] = n_seg
            multi_key[m] = k
            m += 1
        end = offsets[k] + counts[k]
        for j in range(n_seg):
            seg_start[s] = offsets[k] + j * max_lines
            seg_end[s] = min(seg_start[s] + max_lines, end)
            seg_key[s] = k
            if n_seg > 1:
                seg_partial[s] = q
                q += 1
            else:
                seg_partial[s] = -1
            s += 1
    return offsets, seg_start, seg_end, seg_key, seg_partial, multi_first, multi_count, multi_key


def accumulate_superlines(
        profile_buffer: DeviceNDArray,
        lookups: dict[str, DeviceNDArray],
        id_f: npt.NDArray[np.int32],
        id_i: npt.NDArray[np.int32],
        a_fi: npt.NDArray[np.float64],
        energy_fi: npt.NDArray[np.float64],
        band_indices: npt.NDArray[np.int32] | None = None,
) -> None:
    """
    Adds a batch of transitions (host arrays) to a device super-line buffer, reproducibly: the same inputs give the
    same buffer bit for bit (no float atomics, and lines summed in input order within each segment).

    Parameters
    ----------
    profile_buffer : device array, shape (4, n_bands, n_layers, n_grid) or (4, n_layers, n_grid)
        Band buffer (superline.accumulate_superline_band_batch channels), with band_indices; or band-less buffer
        (superline.accumulate_superline_batch channels), without.
    lookups : dict
        Device arrays "bin_edges", "n_frac" (populations, state-major), "g", "inv_g" and "gamma_lifetime", from
        :func:`to_device`.
    id_f, id_i, a_fi, energy_fi : ndarray, shape (n_trans,)
    band_indices : ndarray, shape (n_trans,), optional
    """
    n_trans = energy_fi.shape[0]
    n_grid = profile_buffer.shape[-1]
    n_layers = lookups["n_frac"].shape[1]
    d_batch = [cuda.to_device(a) for a in (id_f, id_i, a_fi, energy_fi)]
    if band_indices is None:
        d_bands = cuda.device_array(0, dtype=np.int32)
        n_keys = n_grid
    else:
        d_bands = cuda.to_device(band_indices)
        n_keys = (int(band_indices.max()) + 1) * n_grid
    if max(n_keys, n_trans) >= 2 ** 31:
        raise ValueError(f"{n_keys} super-line cells or {n_trans} transitions overflow the int32 keys/positions.")
    keys = cuda.device_array(n_trans, dtype=np.int32)
    counts = cuda.to_device(np.zeros(n_keys, dtype=np.int32))
    blocks, threads = _launch_dims(n_trans)
    _key_count_kernel[blocks, threads](lookups["bin_edges"], d_bands, d_batch[3], n_grid, keys, counts)
    offsets, seg_start, seg_end, seg_key, seg_partial, multi_first, multi_count, multi_key = _segments(
        counts=counts.copy_to_host(), max_lines=_SEGMENT_MAX_LINES)
    positions = cuda.to_device(_stable_positions(keys=keys.copy_to_host(), offsets=offsets))
    sorted_batch = [cuda.device_array(n_trans, dtype=np.int32) for _ in range(2)] + [
        cuda.device_array(n_trans, dtype=np.float64) for _ in range(4)]
    _scatter_kernel[blocks, threads](positions, lookups["g"], lookups["inv_g"], lookups["gamma_lifetime"], *d_batch,
                                     *sorted_batch)
    partials = cuda.device_array((max(int(multi_count.sum()), 1), n_layers, 4), dtype=np.float64)
    kernel = _accumulate_segments_xsec_kernel if band_indices is None else _accumulate_segments_band_kernel
    blocks, threads = _launch_dims(seg_start.shape[0] * n_layers)
    kernel[blocks, threads](profile_buffer, partials, lookups["n_frac"], cuda.to_device(seg_start),
                            cuda.to_device(seg_end), cuda.to_device(seg_key), cuda.to_device(seg_partial),
                            *sorted_batch)
    if multi_key.shape[0]:
        buffer_4d = profile_buffer if band_indices is not None else profile_buffer.reshape(4, 1, n_layers, n_grid)
        blocks, threads = _launch_dims(multi_key.shape[0] * n_layers)
        _reduce_partials_kernel[blocks, threads](buffer_4d, partials, cuda.to_device(multi_first),
                                                 cuda.to_device(multi_count), cuda.to_device(multi_key))


# ------------------------------------- COST MODEL -------------------------------------

def _estimate_n_transitions(trans_files: list[pathlib.Path]) -> int:
    """
    Cheap estimate of the number of transitions in a set of line list files: exact for parquet (file metadata),
    and from the file size and the mean length of the first lines for text files.
    """
    total = 0
    for path in trans_files:
        if str(path).endswith(".parquet"):
            total += pq.ParquetFile(path).metadata.num_rows
        else:
            with open(path, "rb") as fh:
                head = fh.read(1 << 20)
            n_lines = max(head.count(b"\n"), 1)
            total += int(os.path.getsize(path) * n_lines / max(len(head), 1))
    return total


def gpu_pays_off(trans_files: list[pathlib.Path], n_layers: int, buffer_bytes: int, label: str) -> bool:
    """
    Whether accumulating the super-lines of trans_files over n_layers layers on the GPU is expected to be faster than
    on the CPU, given the fixed GPU costs and the copy of a buffer_bytes profile buffer back to the host
    (config._GPU_LINE_LAYER_SAVING_S, _GPU_FIXED_COST_S, _GPU_TRANSFER_BYTES_PER_S). Logs the estimate when not.
    """
    n_line_layers = _estimate_n_transitions(trans_files) * n_layers
    saving = n_line_layers * config._GPU_LINE_LAYER_SAVING_S
    cost = config._GPU_FIXED_COST_S + buffer_bytes / config._GPU_TRANSFER_BYTES_PER_S
    if saving < cost:
        log.info(f"{label}: GPU would save ~{saving:.1f}s for ~{n_line_layers:.2e} line-layers but costs ~{cost:.1f}s; "
                 f"using the CPU.")
    return saving >= cost


# ------------------------------------- SAMPLED VOIGT (NON-SUPER-LINE) -------------------------------------

@cuda.jit(cache=True)
def _band_profile_sampled_voigt_kernel(
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
        inv_doppler: npt.NDArray[np.float64],
        gamma_pressure: npt.NDArray[np.float64],
) -> None:
    n_layers = inv_doppler.shape[0]
    n_grid = wn_grid.shape[0]
    n_bands_in_batch = group_bands.shape[0]
    cutoff = 25.0
    inv_sqrt_pi = 1.0 / math.sqrt(math.pi)

    idx = cuda.grid(1)
    stride = cuda.gridsize(1)
    total = n_bands_in_batch * n_layers
    for bl in range(idx, total, stride):
        b_local = bl // n_layers
        l = bl % n_layers

        b_global = group_bands[b_local]
        gs = group_starts[b_local]
        ge = group_ends[b_local]
        inv_doppler_l = inv_doppler[l]
        gamma_pressure_l = gamma_pressure[l]

        for t in range(gs, ge):
            energy_fi_t = energy_fi[t]
            j_start = max(0, _binary_search_left_gpu(wn_grid, energy_fi_t - cutoff))
            j_end = min(n_grid, _binary_search_right_gpu(wn_grid, energy_fi_t + cutoff) + 1)
            if j_start >= j_end:
                continue

            id_f_t = id_f[t]
            id_i_t = id_i[t]
            a_fi_t = a_fi[t]

            ste_prefactor_t = a_fi_t / (const_8_pi_c * energy_fi_t * energy_fi_t)
            abs_prefactor_t = ste_prefactor_t * g_lookup[id_f_t] * inv_g_lookup[id_i_t]
            spe_prefactor_t = a_fi_t * energy_fi_t * const_h_c_on_4_pi

            gamma_total_tl = gamma_lifetime_lookup[id_f_t] + gamma_lifetime_lookup[id_i_t] + gamma_pressure_l
            inv_sigma_sqrt2_tl = inv_doppler_l / energy_fi_t
            y_tl = gamma_total_tl * inv_sigma_sqrt2_tl

            n_f_tl = n_lookup[id_f_t, l]
            abs_tl = n_lookup[id_i_t, l] * abs_prefactor_t
            ste_tl = n_f_tl * ste_prefactor_t
            spe_tl = n_f_tl * spe_prefactor_t

            norm = inv_sigma_sqrt2_tl * inv_sqrt_pi

            for j in range(j_start, j_end):
                x_tj = (wn_grid[j] - energy_fi_t) * inv_sigma_sqrt2_tl
                voigt_val = _voigt_humlicek_w_gpu(x_tj, y_tl) * norm

                profile_buffer[0, b_global, l, j] += abs_tl * voigt_val
                profile_buffer[1, b_global, l, j] += ste_tl * voigt_val
                profile_buffer[2, b_global, l, j] += spe_tl * voigt_val


# ------------------------------------- BROADEN (SUPER-LINE BANDS) -------------------------------------

@cuda.jit(cache=True)
def _band_layer_cost_kernel(
        profile_buffer: npt.NDArray[np.float64],
        j_starts: npt.NDArray[np.int32],
        j_ends: npt.NDArray[np.int32],
        cost: npt.NDArray[np.int64],
) -> None:
    """GPU counterpart of bands.band_layer_work_order's cost estimate: Voigt evaluations per (band, layer)."""
    n_bands, n_layers, n_grid = profile_buffer.shape[1], profile_buffer.shape[2], profile_buffer.shape[3]
    for bl in range(cuda.grid(1), n_bands * n_layers, cuda.gridsize(1)):
        band = bl // n_layers
        l = bl % n_layers
        c = 0
        for i in range(n_grid):
            if profile_buffer[0, band, l, i] != 0.0 or profile_buffer[1, band, l, i] != 0.0 \
                    or profile_buffer[2, band, l, i] != 0.0:
                c += j_ends[i] - j_starts[i]
        cost[bl] = c


@cuda.jit(cache=True)
def _broaden_superline_band_buffer_kernel(
        profile_buffer: npt.NDArray[np.float64],
        buffer_out: npt.NDArray[np.float64],
        wn_grid: npt.NDArray[np.float64],
        inv_doppler: npt.NDArray[np.float64],
        gamma_pressure: npt.NDArray[np.float64],
        j_starts: npt.NDArray[np.int32],
        j_ends: npt.NDArray[np.int32],
        order: npt.NDArray[np.int32],
        n_pairs: int,
) -> None:
    """One thread per (band, layer) output row, taking pairs order[k]; see superline._broaden_superline_band_buffer."""
    n_layers, n_grid = profile_buffer.shape[2], profile_buffer.shape[3]
    inv_sqrt_pi = 1.0 / math.sqrt(math.pi)
    for k in range(cuda.grid(1), n_pairs, cuda.gridsize(1)):
        bl = order[k]
        band = bl // n_layers
        l = bl % n_layers
        inv_doppler_l = inv_doppler[l]
        gamma_pressure_l = gamma_pressure[l]
        for i in range(n_grid):
            abs_bli = profile_buffer[0, band, l, i]
            ste_bli = profile_buffer[1, band, l, i]
            spe_bli = profile_buffer[2, band, l, i]
            if abs_bli == 0.0 and ste_bli == 0.0 and spe_bli == 0.0:
                continue
            energy_i = wn_grid[i]
            inv_sigma_sqrt2_il = inv_doppler_l / energy_i
            norm = inv_sigma_sqrt2_il * inv_sqrt_pi
            gamma_lifetime_bli = 0.0
            if abs_bli != 0.0:
                gamma_lifetime_bli = profile_buffer[3, band, l, i] / abs_bli
            y_bli = (gamma_pressure_l + gamma_lifetime_bli) * inv_sigma_sqrt2_il
            for j in range(j_starts[i], j_ends[i]):
                x_ij = (wn_grid[j] - energy_i) * inv_sigma_sqrt2_il
                voigt_val = _voigt_humlicek_w_gpu(x_ij, y_bli) * norm
                buffer_out[0, band, l, j] += abs_bli * voigt_val
                buffer_out[1, band, l, j] += ste_bli * voigt_val
                buffer_out[2, band, l, j] += spe_bli * voigt_val


# ------------------------------------- DEVICE-RESIDENT PROFILE BUFFER -------------------------------------

def _free_device_bytes() -> int:
    free, _ = cuda.current_context().get_memory_info()
    return int(free)


def device_zeros(shape: tuple[int, ...]) -> DeviceNDArray:
    d = cuda.device_array(shape, dtype=np.float64)
    cuda.driver.device_memset(d, 0, d.nbytes)
    return d


def to_device(**arrays: npt.NDArray) -> dict[str, DeviceNDArray]:
    """Contiguous device copies of arrays, by name."""
    return {name: cuda.to_device(np.ascontiguousarray(a)) for name, a in arrays.items()}


class DeviceProfileBuffer:
    """
    Band profile buffer, shape (n_channels, capacity, n_layers, n_grid), kept on the GPU for a whole transition-batch
    loop, together with the per-state lookup tables and per-layer broadening terms.

    Channels are those of the matching CPU buffer: abs, ste, spe (and the absorption-weighted lifetime HWHM for
    super-lines). The capacity grows with the number of bands registered (:meth:`ensure_capacity`); when the device
    cannot hold the buffer, :meth:`ensure_capacity` returns False and the caller continues on the CPU with
    :meth:`to_host`.

    Parameters
    ----------
    n_channels : int
        4 for super-lines, 3 for direct profiles.
    n_bands : int
        Initial band capacity.
    wn_grid : ndarray, shape (n_grid,)
    bin_edges : ndarray, shape (n_grid + 1,)
    n_frac_lookup : ndarray, shape (n_states + 1, n_layers)
    g_lookup, inv_g_lookup, gamma_lifetime_lookup : ndarray, shape (n_states + 1,)
    inv_doppler, gamma_pressure : ndarray, shape (n_layers,)
        From :func:`lineshape.layer_broadening_terms`.
    """

    def __init__(
            self,
            n_channels: int,
            n_bands: int,
            wn_grid: npt.NDArray[np.float64],
            bin_edges: npt.NDArray[np.float64],
            n_frac_lookup: npt.NDArray[np.float64],
            g_lookup: npt.NDArray[np.float64],
            inv_g_lookup: npt.NDArray[np.float64],
            gamma_lifetime_lookup: npt.NDArray[np.float64],
            inv_doppler: npt.NDArray[np.float64],
            gamma_pressure: npt.NDArray[np.float64],
    ) -> None:
        self.n_channels = n_channels
        self.n_layers = n_frac_lookup.shape[1]
        self.n_grid = wn_grid.shape[0]
        self.wn_grid = np.ascontiguousarray(wn_grid)
        self._d = to_device(
            wn_grid=wn_grid, bin_edges=bin_edges, n_frac=n_frac_lookup, g=g_lookup, inv_g=inv_g_lookup,
            gamma_lifetime=gamma_lifetime_lookup, inv_doppler=inv_doppler, gamma_pressure=gamma_pressure,
        )
        self.capacity = 0
        self.buffer = None
        if not self.ensure_capacity(n_bands):
            raise MemoryError(f"Profile buffer for {n_bands} bands does not fit in device memory.")

    def _band_bytes(self, n_channels: int | None = None) -> int:
        return (self.n_channels if n_channels is None else n_channels) * self.n_layers * self.n_grid * 8

    def ensure_capacity(self, n_bands: int) -> bool:
        """
        Grow the buffer to hold at least n_bands bands, keeping its contents. Grows on the device when old and new
        buffers fit together, otherwise through the host. Returns False (leaving the buffer unchanged) if the new
        buffer cannot fit on the device at all. Room for a full batch's working memory is kept free.
        """
        if n_bands <= self.capacity:
            return True
        new_capacity = max(n_bands, 2 * self.capacity)
        old_bytes = 0 if self.buffer is None else self.buffer.nbytes
        free = (_free_device_bytes() - _DEVICE_MEMORY_MARGIN
                - _PARQUET_BATCH_SIZE * (_DEVICE_BYTES_PER_TRANSITION + (self.n_layers + 3) // 4))
        # Fall back to exactly n_bands if doubling does not fit.
        if new_capacity * self._band_bytes() > free + old_bytes:
            new_capacity = n_bands
        new_bytes = new_capacity * self._band_bytes()
        if new_bytes > free + old_bytes:
            return False
        shape = (self.n_channels, new_capacity, self.n_layers, self.n_grid)
        if self.buffer is None:
            self.buffer = device_zeros(shape)
        elif old_bytes + new_bytes <= free:
            new = device_zeros(shape)
            for c in range(self.n_channels):
                new[c, :self.capacity].copy_to_device(self.buffer[c])
            self.buffer = new
        else:
            host = self.to_host(self.capacity)
            self.free_buffer()
            self.buffer = device_zeros(shape)
            for c in range(self.n_channels):
                self.buffer[c, :host.shape[1]].copy_to_device(host[c])
        self.capacity = new_capacity
        return True

    def accumulate_superlines(
            self,
            band_indices: npt.NDArray[np.int32],
            id_f: npt.NDArray[np.int32],
            id_i: npt.NDArray[np.int32],
            a_fi: npt.NDArray[np.float64],
            energy_fi: npt.NDArray[np.float64],
    ) -> None:
        """Add one batch of transitions to the super-line buffer (see superline.accumulate_superline_band_batch)."""
        accumulate_superlines(profile_buffer=self.buffer, lookups=self._d, id_f=id_f, id_i=id_i, a_fi=a_fi,
                              energy_fi=energy_fi, band_indices=band_indices)

    def sampled_voigt(
            self,
            id_f: npt.NDArray[np.int32],
            id_i: npt.NDArray[np.int32],
            group_bands: npt.NDArray[np.int32],
            group_starts: npt.NDArray[np.int32],
            group_ends: npt.NDArray[np.int32],
            a_fi: npt.NDArray[np.float64],
            energy_fi: npt.NDArray[np.float64],
    ) -> None:
        """Add one batch of transitions as direct sampled Voigt profiles (see sampled.band_profile_sampled_voigt)."""
        d_batch = [cuda.to_device(a) for a in (id_f, id_i, group_bands, group_starts, group_ends, a_fi, energy_fi)]
        d = self._d
        blocks, threads = _launch_dims(group_bands.shape[0] * self.n_layers)
        _band_profile_sampled_voigt_kernel[blocks, threads](
            self.buffer, d["wn_grid"], d_batch[0], d_batch[1], d_batch[2], d_batch[3], d_batch[4], d["n_frac"], d["g"],
            d["inv_g"], d["gamma_lifetime"], d_batch[5], d_batch[6], d["inv_doppler"], d["gamma_pressure"],
        )

    def to_host(self, n_bands: int) -> npt.NDArray[np.float64]:
        """Copy the first n_bands bands back to the host, shape (n_channels, n_bands, n_layers, n_grid)."""
        out = np.empty((self.n_channels, n_bands, self.n_layers, self.n_grid), dtype=np.float64)
        for c in range(self.n_channels):
            self.buffer[c, :n_bands].copy_to_host(out[c])
        return out

    def broaden_superlines(self, n_bands: int) -> npt.NDArray[np.float64]:
        """
        Broaden the first n_bands super-line bands on the device and return the result on the host, shape
        (3, n_bands, n_layers, n_grid) (see superline._broaden_superline_band_buffer). The output is produced in band
        chunks sized to the free device memory.
        """
        t0 = time.perf_counter()
        n_layers, n_grid = self.n_layers, self.n_grid
        j_starts, j_ends = profile_windows(wn_grid=self.wn_grid, cutoff=25.0)
        d_js, d_je = cuda.to_device(j_starts), cuda.to_device(j_ends)
        d_view = self.buffer[:, :n_bands]
        d_cost = cuda.device_array(n_bands * n_layers, dtype=np.int64)
        blocks, threads = _launch_dims(n_bands * n_layers)
        _band_layer_cost_kernel[blocks, threads](d_view, d_js, d_je, d_cost)
        cost = d_cost.copy_to_host()
        order = np.argsort(-cost, kind="stable").astype(np.int32)
        band_of = order // n_layers

        out = np.empty((3, n_bands, n_layers, n_grid), dtype=np.float64)
        chunk = max(1, min(n_bands, (_free_device_bytes() - _DEVICE_MEMORY_MARGIN) // self._band_bytes(3)))
        for b0 in range(0, n_bands, chunk):
            b1 = min(n_bands, b0 + chunk)
            sel = order[(band_of >= b0) & (band_of < b1) & (cost[order] > 0)] - b0 * n_layers
            d_out = device_zeros((3, b1 - b0, n_layers, n_grid))
            if sel.size:
                blocks, threads = _launch_dims(sel.size)
                _broaden_superline_band_buffer_kernel[blocks, threads](
                    self.buffer[:, b0:b1], d_out, self._d["wn_grid"], self._d["inv_doppler"],
                    self._d["gamma_pressure"], d_js, d_je, cuda.to_device(sel), sel.size,
                )
            for c in range(3):
                d_out[c].copy_to_host(out[c, b0:b1])
            del d_out
        if log.isEnabledFor(_LOG_VERBOSE_2):
            log.log(_LOG_VERBOSE_2, f"[GPU] broaden_superlines: {time.perf_counter() - t0:.3f}s for {n_bands} bands.")
        return out

    def free_buffer(self) -> None:
        self.buffer = None
        self.capacity = 0
        cuda.current_context().deallocations.clear()

    def free(self) -> None:
        """Release all device memory held by this buffer."""
        self.free_buffer()
        self._d = {}
        cuda.current_context().deallocations.clear()
