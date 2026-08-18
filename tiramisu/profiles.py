import time
import logging
import math
import pathlib
import typing as t
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numba
import numpy as np
import polars as pl
from astropy import constants as ac, units as u

from pyarrow import parquet as pq
from dask import dataframe as dd
from numpy import typing as npt

from .config import _DEFAULT_NUM_THREADS, _INTENSITY_CUTOFF, _PARQUET_BATCH_SIZE, _DASK_BLOCK_SIZE, _LOG_VERBOSE_1, \
    _LOG_VERBOSE_2
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
const_h_on_8_c_amu = const_h_on_8_c / const_amu
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


def _iter_trans_batches(
        trans_file: pathlib.Path,
        trans_columns: list[str],
        states_i: pl.DataFrame,
        states_f: pl.DataFrame,
        wn_min: float,
        wn_max: float,
        parquet_batch_size: int,
        dask_dtypes: dict,
        do_super_lines: bool,
) -> t.Iterator[pl.DataFrame]:
    """
    Yields filtered, joined trans batches regardless of file format.

    Called by :func:`~xsec.NLTEProcessor.compute_rates_profiles`.

    Parameters
    ----------
    trans_file
    trans_columns
    states_i
    states_f
    wn_min
    wn_max
    dask_dtypes
    do_super_lines

    Returns
    -------

    """

    def _process(raw: pl.DataFrame) -> pl.DataFrame:
        out = (
            raw
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
        )
        if not do_super_lines:
            return out.sort(["id_agg_f", "id_agg_i"])
        return out

    if str(trans_file).endswith(".parquet"):
        with pq.ParquetFile(trans_file) as pq_file:
            for arrow_batch in pq_file.iter_batches(
                    batch_size=parquet_batch_size,
                    columns=trans_columns,
                    use_threads=True,
            ):
                yield _process(pl.from_arrow(arrow_batch))
    else:
        ddf = dd.read_csv(
            trans_file,
            sep=r"\s+",
            engine="python",
            header=None,
            names=trans_columns,
            usecols=[0, 1, 2],
            dtype=dask_dtypes,
            blocksize=_DASK_BLOCK_SIZE,
        )
        for delayed_batch in ddf.to_delayed():
            yield _process(pl.from_pandas(delayed_batch.compute()))


@numba.njit(parallel=False, cache=True, error_model="numpy", inline="always")
def _find_groups_from_ids(
        id_agg_f: npt.NDArray[np.int32],
        id_agg_i: npt.NDArray[np.int32],
        band_indices: npt.NDArray[np.int32],
) -> t.Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """
    Parameters
    ----------
    id_agg_f : ndarray, shape (n_trans,)
    id_agg_i : ndarray, shape (n_trans,)
    band_indices : ndarray, shape (n_trans,)

    Returns
    -------
    band_group_indices : ndarray, shape (n_bands_in_batch,)
        Maps the bands in the current batch to the band index stored in the external band_indices map.
    group_starts : ndarray, shape (n_bands_in_batch,)
        Start index in the transition arrays for each band.
    group_ends : ndarray, shape (n_bands_in_batch,)
        End index (exclusive) in the transition arrays for each band.
    """
    n_keys = id_agg_f.shape[0]
    n_groups = 1

    starts = np.empty(n_keys, dtype=np.int32)
    starts[0] = 0

    for i in range(1, n_keys):
        if id_agg_f[i] != id_agg_f[i - 1] or id_agg_i[i] != id_agg_i[i - 1]:
            starts[n_groups] = i
            n_groups += 1
    starts = starts[:n_groups]

    ends = np.empty(n_groups, dtype=np.int32)
    for g in range(n_groups - 1):
        ends[g] = starts[g + 1]
    ends[n_groups - 1] = n_keys

    band_group_indices = np.empty(n_groups, dtype=band_indices.dtype, )

    for g in range(n_groups):
        band_group_indices[g] = band_indices[starts[g]]

    return band_group_indices, starts, ends


# ------------------------------------- EINSTEIN COEFFICIENTS -------------------------------------
@numba.njit(parallel=True, cache=True, error_model="numpy")
def calc_einstein_bs(
        id_i: npt.NDArray[np.int32],
        id_f: npt.NDArray[np.int32],
        a_fi: npt.NDArray[np.float64],
        energy_fi: npt.NDArray[np.float64],
        g_lookup: npt.NDArray[np.float64],
        inv_g_lookup: npt.NDArray[np.float64],
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    n_trans = a_fi.shape[0]
    b_fi = np.zeros(n_trans, dtype=np.float64)
    b_if = np.zeros(n_trans, dtype=np.float64)
    inv_energy_fi_cubed = 1.0 / (energy_fi ** 3)

    for t in numba.prange(n_trans):
        a_fi_t = a_fi[t]
        g_f_t = g_lookup[id_f[t]]
        inv_g_i_t = inv_g_lookup[id_i[t]]
        inv_energy_fi_cubed_t = inv_energy_fi_cubed[t]
        # b_fi_t = a_fi_t * inv_energy_fi_cubed_t / (2 * ac.h * ac.c)  # Wavenumber domain.
        # b_fi_t = a_fi_t / (inv_energy_fi_cubed_t * 2* ac.h * ac.c)  # Wavelength domain.
        b_fi_t = a_fi_t * const_c_sq_on_2h * inv_energy_fi_cubed_t  # Frequency domain.
        b_fi[t] = b_fi_t
        b_if[t] = b_fi_t * g_f_t * inv_g_i_t

    return b_fi, b_if


# ------------------------------------- PROCESS BATCH HELPERS -------------------------------------
@numba.njit(parallel=True, cache=True, error_model="numpy")
def _accumulate_superline_batch(
        profile_buffer: npt.NDArray[np.float64],
        bin_edges: npt.NDArray[np.float64],
        n_lookup: npt.NDArray[np.float64],
        g_lookup: npt.NDArray[np.float64],
        inv_g_lookup: npt.NDArray[np.float64],
        id_f: npt.NDArray[np.int32],
        id_i: npt.NDArray[np.int32],
        a_fi: npt.NDArray[np.float64],
        energy_fi: npt.NDArray[np.float64],
) -> None:
    """

    Parameters
    ----------
    profile_buffer : ndarray, shape (2, n_layers, n_grid)
    bin_edges : ndarray, shape (n_grid + 1,)
    n_lookup : ndarray, shape (n_states + 1, n_layers)
    g_lookup : ndarray, shape (n_states + 1)
    inv_g_lookup : ndarray, shape (n_states + 1)
    id_f : ndarray, shape (n_trans,)
    id_i : ndarray, shape (n_trans,)
    a_fi : ndarray, shape (n_trans,)
    energy_fi : ndarray (n_trans,)

    Returns
    -------

    """
    n_trans = energy_fi.shape[0]
    n_layers = n_lookup.shape[1]
    max_idx = bin_edges.shape[0] - 2

    bin_indices = np.empty(n_trans, dtype=np.int32)
    abs_prefactor = np.empty(n_trans, np.float64)
    emi_prefactor = np.empty(n_trans, np.float64)

    for t in numba.prange(n_trans):
        a_fi_t = a_fi[t]
        energy_fi_t = energy_fi[t]
        abs_prefactor[t] = a_fi_t / (const_8_pi_c * energy_fi_t * energy_fi_t)
        emi_prefactor[t] = a_fi_t * energy_fi_t * const_h_c_on_4_pi

        bin_idx = _binary_search_right(bin_edges, energy_fi_t) - 1
        if bin_idx < 0:
            bin_idx = 0
        elif bin_idx > max_idx:
            bin_idx = max_idx

        bin_indices[t] = bin_idx

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
            profile_buffer[0, l, bin_t] += (
                    abs_prefactor_t * ((n_il * g_f_t * inv_g_i_t) - n_fl)
            )
            profile_buffer[1, l, bin_t] += (
                    emi_prefactor_t * n_fl
            )


@numba.njit(parallel=True, cache=True, error_model="numpy")
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
) -> None:
    """
    Can be modified to add in photoassociation cross-section if desired.

    Parameters
    ----------
    profile_buffer : ndarray, shape (2, n_layers, n_grid)
    bin_edges : ndarray, shape (n_grid + 1,)
    n_lookup : ndarray, shape (n_states + 1, n_layers)
    g_lookup : ndarray, shape (n_states + 1, )
    inv_g_lookup : ndarray, shape (n_states + 1, )
    v_lookup : ndarray, shape (n_states + 1, )
    id_f : ndarray, shape (n_trans,)
    id_i : ndarray, shape (n_trans,)
    a_fi : ndarray, shape (n_trans,)
    energy_fi : ndarray (n_trans,)
    reduced_mass : float
    box_length : floatawdasdasda

    """
    n_trans = energy_fi.shape[0]
    n_layers = n_lookup.shape[1]
    n_states = v_lookup.shape[0]
    max_idx = bin_edges.shape[0] - 2

    bin_indices = np.empty(n_trans, dtype=np.int32)
    abs_prefactor = np.empty(n_trans, np.float64)

    for t in numba.prange(n_trans):
        a_fi_t = a_fi[t]
        energy_fi_t = energy_fi[t]
        abs_prefactor[t] = a_fi_t / (const_8_pi_c * energy_fi_t * energy_fi_t)

        bin_idx = _binary_search_right(bin_edges, energy_fi_t) - 1
        if bin_idx < 0:
            bin_idx = 0
        elif bin_idx > max_idx:
            bin_idx = max_idx

        bin_indices[t] = bin_idx

    alpha_box_lookup = np.empty(n_states, dtype=np.float64)
    alpha_box_prefactor = const_h_on_8_c_amu / (box_length * box_length * reduced_mass)
    for s in numba.prange(n_states):
        alpha_box_lookup[s] = alpha_box_prefactor * (2.0 * v_lookup[s] + 1.0)

    for l in numba.prange(n_layers):
        for t in range(n_trans):
            bin_t = bin_indices[t]
            abs_tl = abs_prefactor[t] * g_lookup[id_f[t]] * inv_g_lookup[id_i[t]] * n_lookup[id_i[t], l]
            # Accumulate into buffers.
            profile_buffer[0, l, bin_t] += abs_tl
            profile_buffer[1, l, bin_t] += abs_tl * alpha_box_lookup[id_f[t]]


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _accumulate_superline_band_batch(
        profile_buffer: npt.NDArray[np.float64],
        band_indices: npt.NDArray[np.int32],
        bin_edges: npt.NDArray[np.float64],
        n_frac_lookup: npt.NDArray[np.float64],
        g_lookup: npt.NDArray[np.float64],
        inv_g_lookup: npt.NDArray[np.float64],
        id_f: npt.NDArray[np.int32],
        id_i: npt.NDArray[np.int32],
        a_fi: npt.NDArray[np.float64],
        energy_fi: npt.NDArray[np.float64],
) -> None:
    """

    Parameters
    ----------
    profile_buffer : ndarray, shape (3, n_bands, n_layers, n_grid)
    band_indices : ndarray, shape (n_trans,)
    bin_edges : ndarray, shape (n_grid + 1,)
    n_frac_lookup : ndarray, shape (n_states + 1, n_layers)
    g_lookup : ndarray, shape (n_states + 1,)
    inv_g_lookup : ndarray, shape (n_states + 1,)
    id_f : ndarray, shape (n_trans,)
    id_i : ndarray, shape (n_trans,)
    a_fi : ndarray, shape (n_trans,)
    energy_fi : ndarray (n_trans,)
    """
    n_trans = energy_fi.shape[0]
    n_layers = n_frac_lookup.shape[1]
    max_idx = bin_edges.shape[0] - 2

    bin_indices = np.empty(n_trans, dtype=np.int32)
    abs_ste_prefactor = np.empty(n_trans, np.float64)
    spe_prefactor = np.empty(n_trans, np.float64)

    for t in numba.prange(n_trans):
        a_fi_t = a_fi[t]
        energy_fi_t = energy_fi[t]
        abs_ste_prefactor[t] = a_fi_t / (const_8_pi_c * energy_fi_t * energy_fi_t)
        spe_prefactor[t] = a_fi_t * energy_fi_t * const_h_c_on_4_pi

        bin_idx = _binary_search_right(bin_edges, energy_fi_t) - 1
        if bin_idx < 0:
            bin_idx = 0
        elif bin_idx > max_idx:
            bin_idx = max_idx

        bin_indices[t] = bin_idx

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
) -> None:
    """
    Can be modified to add in photoassociation cross-section if desired.

    Parameters
    ----------
    profile_buffer : ndarray, shape (2, n_bands, n_layers, n_grid)
    band_indices : ndarray, shape (n_trans,)
    bin_edges : ndarray, shape (n_grid + 1,)
    n_frac_lookup : ndarray, shape (n_states + 1, n_layers)
    g_lookup : ndarray, shape (n_states + 1, )
    inv_g_lookup : ndarray, shape (n_states + 1,)
    v_lookup : ndarray, shape (n_states + 1, )
    id_f : ndarray, shape (n_trans,)
    id_i : ndarray, shape (n_trans,)
    a_fi : ndarray, shape (n_trans,)
    energy_fi : ndarray (n_trans,)
    reduced_mass : float
    box_length : float
    """
    n_trans = energy_fi.shape[0]
    n_layers = n_frac_lookup.shape[1]
    n_states = v_lookup.shape[0]
    max_idx = bin_edges.shape[0] - 2

    bin_indices = np.empty(n_trans, dtype=np.int32)
    abs_prefactor = np.empty(n_trans, np.float64)

    for t in numba.prange(n_trans):
        a_fi_t = a_fi[t]
        energy_fi_t = energy_fi[t]
        abs_prefactor[t] = a_fi_t / (const_8_pi_c * energy_fi_t * energy_fi_t)

        bin_idx = _binary_search_right(bin_edges, energy_fi_t) - 1
        if bin_idx < 0:
            bin_idx = 0
        elif bin_idx > max_idx:
            bin_idx = max_idx

        bin_indices[t] = bin_idx

    alpha_box_lookup = np.empty(n_states, dtype=np.float64)
    alpha_box_prefactor = const_h_on_8_c_amu / (box_length * box_length * reduced_mass)
    for s in numba.prange(n_states):
        alpha_box_lookup[s] = alpha_box_prefactor * (2.0 * v_lookup[s] + 1.0)

    for l in numba.prange(n_layers):
        for t in range(n_trans):
            band_t = band_indices[t]
            bin_t = bin_indices[t]
            abs_tl = abs_prefactor[t] * g_lookup[id_f[t]] * inv_g_lookup[id_i[t]] * n_frac_lookup[id_i[t], l]
            # Accumulate into buffers.
            profile_buffer[0, band_t, l, bin_t] += abs_tl
            profile_buffer[1, band_t, l, bin_t] += abs_tl * alpha_box_lookup[id_f[t]]


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


# @numba.njit(parallel=True, cache=True, error_model="numpy")
# def _compute_all_cross_terms_vectorized(
#         emission_profiles: npt.NDArray[np.float64],
#         chem_factor: float,
# ) -> npt.NDArray[np.float64]:
#     """
#
#     Parameters
#     ----------
#     emission_profiles
#     chem_factor
#
#     Returns
#     -------
#
#     """
#     n_agg_states, num_grid = emission_profiles.shape
#     result = np.zeros((n_agg_states, num_grid), dtype=np.float64)
#
#     for o_idx in numba.prange(n_agg_states):
#         for wn_idx in range(num_grid):
#             result[o_idx, wn_idx] = (
#                     emission_profiles[o_idx, wn_idx] * chem_factor
#             )
#
#     return result


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

    # def get_profiles(
    #         self, layer_idx: int, key: t.Tuple[int, int]
    # ) -> t.Tuple[
    #     t.Tuple[npt.NDArray[np.float64], int],
    #     t.Tuple[npt.NDArray[np.float64], int],
    #     t.Tuple[npt.NDArray[np.float64], int]
    # ]:
    #     return (
    #         self.abs_profiles[layer_idx].get_profile(key),
    #         self.ste_profiles[layer_idx].get_profile(key),
    #         self.spe_profiles[layer_idx].get_profile(key),
    #     )

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
            self, agg_energies: npt.NDArray[np.float64], profile_type: str = "abs",  id_cutoff: int = 0,
            num_max: int = -1
    ) -> t.List[t.Tuple[int, int]]:
        """
        Returns band keys (id_agg_f, id_agg_i) from the CompactProfile for the given layer, sorted by energy gap
        (id_agg_f - id_agg_i) ascending, then by id_agg_i, then by id_agg_f. Returns at most num_max keys.

        Parameters
        ----------
        agg_energies : ndarray
            Array containing the energies for each aggregate state ordered on id_agg (ascending).
        profile_type : str
            One of "abs", "ste", "spe" - selects which CompactProfile to inspect.
        id_cutoff : int
            ID above which states are fixed, so key pairs containing IDs above this should be excluded. If not provided,
            returned keys include band keys involving the cutoff state.
        num_max : int
            Maximum number of keys to return. If not set to a psoitive value, return all band keys.

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
        if id_cutoff > 0:
            common_keys = [k for k in common_keys if id_cutoff >= k[0] > k[1] >= 0]
        else:
            common_keys = [k for k in common_keys if k[0] > k[1] >= 0]

        def sort_key(k):
            id_f, id_i = k
            e_i = agg_energies[id_i]  # lower state energy
            e_f = agg_energies[id_f]  # upper state energy
            nu = e_f - e_i  # band centre frequency
            return e_i, nu

        common_keys.sort(key=sort_key)
        if num_max > 0:
            return common_keys[:num_max]
        else:
            return common_keys


    def get_cutoff_band_keys(
            self, id_cutoff: int, profile_type: str = "abs",
    ) -> t.List[t.Tuple[int, int]]:
        """

        Parameters
        ----------
        id_cutoff : int
            ID above which states are fixed, so key pairs containing IDs above this should be excluded.
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
        common_keys = [k for k in common_keys if k[0] == id_cutoff or k[1] == id_cutoff]

        return common_keys


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

    # def get_keys(self, layer_idx: int, profile_type: str):
    #     if profile_type == "abs":
    #         return self.abs_profiles[layer_idx].key_lookup.keys()
    #     else:
    #         raise RuntimeError(f"ContinuumProfileStore profile type {profile_type} not implemented.")

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
    trans_columns = ["id_f", "id_i", "A_fi"]
    dask_dtypes = {"id_f": "int64", "id_i": "int64", "A_fi": "float64", }

    # Plain float arrays - no astropy units - passed directly to Numba.
    temperature_slice = temperature_profile[n_lte_layers:].value  # (n_nlte_layers,)
    pressure_slice = pressure_profile[n_lte_layers:].value  # (n_nlte_layers,)

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

    wn_arr = wn_grid.value
    wn_min = wn_arr[0]
    wn_max = wn_arr[-1]
    # Bin edges for finding trans bins (super-lines).
    n_grid = wn_grid.shape[0]
    bin_edges = np.empty(n_grid + 1, dtype=np.float64)
    bin_edges[0] = wn_arr[0] - (wn_arr[1] - wn_arr[0]) * 0.5
    for j in range(1, n_grid):
        bin_edges[j] = (wn_arr[j - 1] + wn_arr[j]) * 0.5
    bin_edges[-1] = wn_arr[-1] + (wn_arr[-1] - wn_arr[-2]) * 0.5

    # States lookup - much more memory efficient than duplicating it all with polars joins!
    state_ids = states["id"].to_numpy()
    max_state_id = int(state_ids.max())
    # State ID lookup offset by 1 as IDs are 1-indexed!
    n_lookup = np.zeros(
        (max_state_id + 1, n_nlte_layers),
        dtype=np.float64,
    )
    for l in range(n_nlte_layers):
        nlte_col = f"n_nlte_L{n_lte_layers + l}"
        lte_col = f"n_L{n_lte_layers + l}"
        # We need to check for the existence of the nLTE column here because n_nlte_Lx doesn't exist in layers that are
        # marked as skipped due to negligible VMR; these layers are unchanged so we use the LTE values.
        if nlte_col in states.columns:
            n_lookup[state_ids, l] = states[nlte_col].to_numpy()
        else:
            n_lookup[state_ids, l] = states[lte_col].to_numpy()

    n_lookup = np.ascontiguousarray(n_lookup)

    g_lookup = np.zeros(max_state_id + 1, dtype=np.float64)
    g_lookup[state_ids] = states["g"].to_numpy()
    g_lookup = np.ascontiguousarray(g_lookup)
    inv_g_lookup = np.zeros_like(g_lookup)
    inv_g_lookup[1:] = 1.0 / g_lookup[1:]
    inv_g_lookup = np.ascontiguousarray(inv_g_lookup)

    tau_lookup = np.zeros(max_state_id + 1, dtype=np.float64)
    tau_lookup[state_ids] = states["tau"].to_numpy()
    tau_lookup = np.ascontiguousarray(tau_lookup)

    broad_n = np.zeros(1, dtype=np.float64)
    broad_gamma = np.zeros((1, n_nlte_layers), dtype=np.float64)
    if broadening_params is not None:
        broad_n = broadening_params[1]
        broad_gamma = broadening_params[0][:, n_lte_layers:]

    # Super-lines accumulator.
    profile_buffer = np.zeros((2, n_nlte_layers, n_grid), dtype=np.float64)
    profile_buffer = np.ascontiguousarray(profile_buffer)

    process_time = time.perf_counter()

    for trans_file in trans_files:
        log.info(f"Processing file {trans_file}.")
        for trans_batch in _iter_trans_batches(
                trans_file=trans_file,
                trans_columns=trans_columns,
                states_i=states_i,
                states_f=states_f,
                wn_min=wn_min,
                wn_max=wn_max,
                parquet_batch_size=_PARQUET_BATCH_SIZE // 3,
                dask_dtypes=dask_dtypes,
                do_super_lines=do_super_lines,
        ):
            if trans_batch.height == 0:
                log.log(_LOG_VERBOSE_1, "No valid trans in batch.")
                continue

            if do_super_lines:
                _accumulate_superline_batch(
                    profile_buffer=profile_buffer,
                    bin_edges=bin_edges,
                    n_lookup=n_lookup,
                    g_lookup=g_lookup,
                    inv_g_lookup=inv_g_lookup,
                    id_f=np.ascontiguousarray(trans_batch["id_f"].to_numpy()),
                    id_i=np.ascontiguousarray(trans_batch["id_i"].to_numpy()),
                    a_fi=np.ascontiguousarray(trans_batch["A_fi"].to_numpy()),
                    energy_fi=np.ascontiguousarray(trans_batch["energy_fi"].to_numpy()),
                )
            else:
                _abs_emi_sampled_voigt(
                    profile_buffer=profile_buffer,
                    wn_grid=wn_arr,
                    id_i=np.ascontiguousarray(trans_batch["id_i"].to_numpy()),
                    id_f=np.ascontiguousarray(trans_batch["id_f"].to_numpy()),
                    n_lookup=n_lookup,
                    g_lookup=g_lookup,
                    inv_g_lookup=inv_g_lookup,
                    tau_lookup=tau_lookup,
                    a_fi=np.ascontiguousarray(trans_batch["A_fi"].to_numpy()),
                    energy_fi=np.ascontiguousarray(trans_batch["energy_fi"].to_numpy()),
                    temperatures=temperature_slice,
                    pressures=pressure_slice,
                    broad_n=broad_n,
                    broad_gamma=broad_gamma,
                    species_mass=species_mass,
                )
    # Finalise.
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
    trans_columns = ["id_f", "id_i", "A_fi"]
    dask_dtypes = {"id_f": "int64", "id_i": "int64", "A_fi": "float64"}

    # Plain float arrays - no astropy units - passed directly to Numba.
    temperature_slice = temperature_profile[n_lte_layers:].value  # (n_nlte_layers,)

    n_cols = [f"n_nlte_L{n_lte_layers + nlte_idx}" for nlte_idx in range(n_nlte_layers)]
    # select_cols = ["id", "g", "v"] + n_cols
    select_cols = ["id"] + n_cols  # g and v are stored on cont_states.
    cont_states = cont_states.join(states.select(select_cols), on="id", how="left")

    invariant_cols = ["id", "energy", "id_agg"]
    states_i = (
        cont_states
        .select(invariant_cols)
        .rename({col: f"{col}_i" for col in invariant_cols})
    )
    states_f = (
        cont_states
        .select(invariant_cols)
        .rename({col: f"{col}_f" for col in invariant_cols})
    )

    wn_arr = wn_grid.value
    wn_min = wn_arr[0]
    wn_max = wn_arr[-1]
    # Bin edges for finding trans bins (super-lines).
    n_grid = wn_grid.shape[0]
    bin_edges = np.empty(n_grid + 1, dtype=np.float64)
    bin_edges[0] = wn_arr[0] - (wn_arr[1] - wn_arr[0]) * 0.5
    for j in range(1, n_grid):
        bin_edges[j] = (wn_arr[j - 1] + wn_arr[j]) * 0.5
    bin_edges[-1] = wn_arr[-1] + (wn_arr[-1] - wn_arr[-2]) * 0.5

    # States lookup - much more memory efficient than duplicating it all with polars joins!
    state_ids = cont_states["id"].to_numpy()
    max_state_id = int(state_ids.max())
    # State ID lookup offset by 1 as IDs are 1-indexed!
    n_lookup = np.zeros(
        (max_state_id + 1, n_nlte_layers),
        dtype=np.float64,
    )
    for l in range(n_nlte_layers):
        nlte_col = f"n_nlte_L{n_lte_layers + l}"
        lte_col = f"n_L{n_lte_layers + l}"
        # We need to check for the existence of the nLTE column here because n_nlte_Lx doesn't exist in layers that are
        # marked as skipped due to negligible VMR; these layers are unchanged so we use the LTE values.
        if nlte_col in states.columns:
            n_lookup[state_ids, l] = cont_states[nlte_col].to_numpy()
        else:
            n_lookup[state_ids, l] = cont_states[lte_col].to_numpy()
    n_lookup = np.ascontiguousarray(n_lookup)

    g_lookup = np.zeros(max_state_id + 1, dtype=np.float64)
    g_lookup[state_ids] = cont_states["g"].to_numpy()
    g_lookup = np.ascontiguousarray(g_lookup)
    zero_g_map = g_lookup == 0
    inv_g_lookup = np.zeros_like(g_lookup)
    inv_g_lookup[~zero_g_map] = 1.0 / g_lookup[~zero_g_map]
    inv_g_lookup = np.ascontiguousarray(inv_g_lookup)

    v_lookup = np.zeros(max_state_id + 1, dtype=np.float64)
    v_lookup[state_ids] = cont_states["v"].to_numpy()
    v_lookup = np.ascontiguousarray(v_lookup)

    # Super-lines accumulator.
    profile_buffer = np.zeros((2, n_nlte_layers, n_grid), dtype=np.float64)
    profile_buffer = np.ascontiguousarray(profile_buffer)

    process_time = time.perf_counter()

    for trans_file in cont_trans_files:
        log.info(f"Processing file {trans_file}.")
        for trans_batch in _iter_trans_batches(
                trans_file=trans_file,
                trans_columns=trans_columns,
                states_i=states_i,
                states_f=states_f,
                wn_min=wn_min,
                wn_max=wn_max,
                parquet_batch_size=_PARQUET_BATCH_SIZE // 3,
                dask_dtypes=dask_dtypes,
                do_super_lines=do_super_lines,
        ):
            if trans_batch.height == 0:
                log.log(_LOG_VERBOSE_1, "No valid trans in batch.")
                continue

            if do_super_lines:
                _accumulate_continuum_superline_batch(
                    profile_buffer=profile_buffer,
                    bin_edges=bin_edges,
                    n_lookup=n_lookup,
                    g_lookup=g_lookup,
                    inv_g_lookup=inv_g_lookup,
                    v_lookup=v_lookup,
                    id_f=np.ascontiguousarray(trans_batch["id_f"].to_numpy()),
                    id_i=np.ascontiguousarray(trans_batch["id_i"].to_numpy()),
                    a_fi=np.ascontiguousarray(trans_batch["A_fi"].to_numpy()),
                    energy_fi=np.ascontiguousarray(trans_batch["energy_fi"].to_numpy()),
                    reduced_mass=reduced_mass,
                    box_length=cont_box_length,
                )
            else:
                _continuum_sampled_gauss(
                    profile_buffer=profile_buffer,
                    wn_grid=wn_arr,
                    id_i=np.ascontiguousarray(trans_batch["id_i"].to_numpy()),
                    id_f=np.ascontiguousarray(trans_batch["id_f"].to_numpy()),
                    n_lookup=n_lookup,
                    g_lookup=g_lookup,
                    inv_g_lookup=inv_g_lookup,
                    v_lookup=v_lookup,
                    a_fi=np.ascontiguousarray(trans_batch["A_fi"].to_numpy()),
                    energy_fi=np.ascontiguousarray(trans_batch["energy_fi"].to_numpy()),
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
    Parameters
    ----------
    profile_buffer : (2, n_layers, n_grid)
        Accumulated super-line coefficients per layer per grid bin. Absorption is stored in profile_buffer[0] and
        emission in profile_buffer[1].
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
    y_voigt = np.empty((n_grid, n_layers), dtype=np.float64)
    # gamma_lifetime = 1.0 / (const_4_pi_c * lifetimes)  # (n_trans,)

    for l in numba.prange(n_layers):
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

    # Precompute occupation mask: True if any layer has nonzero coefficients.
    occupied = np.zeros(n_grid, dtype=numba.boolean)
    for i in numba.prange(n_grid):
        for l in range(n_layers):
            # Loop and l/i index ordering could maybe be changed.
            if profile_buffer[0, l, i] != 0.0 or profile_buffer[1, l, i] != 0.0:
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
            abs_li = profile_buffer[0, l, i]
            emi_li = profile_buffer[1, l, i]
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

    doppler_prefactor = np.empty((n_layers,), dtype=np.float64)

    inv_mass = 1 / species_mass
    doppler_coef = const_sqrt_2_NA_kB_log2_on_c * math.sqrt(inv_mass)
    temp_max = temperatures[0]
    for l in numba.prange(n_layers):
        temp_l = temperatures[l]
        doppler_prefactor[l] = doppler_coef * math.sqrt(temp_l)
        if temp_l > temp_max:
            temp_max = temp_l

    # Precompute occupation mask: True if any layer has nonzero coefficients.
    occupied = np.zeros(n_grid, dtype=numba.boolean)
    for i in numba.prange(n_grid):
        for l in range(n_layers):
            # Loop and l/i index ordering could maybe be changed.
            if profile_buffer[0, l, i] != 0.0:
                occupied[i] = True
                break

    # As in other implementations, "i" tracks transitions.
    for i in numba.prange(n_grid):
        # Super-line centre is the bin centre.
        if not occupied[i]:
            continue
        energy_i = wn_grid[i]

        for l in range(n_layers):
            abs_li = profile_buffer[0, l, i]
            # Skip empty bins.
            if abs_li == 0.0:
                continue

            weighted_broad = profile_buffer[1, l, i]
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

    for l in numba.prange(n_layers):
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
    min_cutoff = 25.0
    max_cutoff = 5000.0
    cutoff_fwhm_multiple = 5.0

    buffer_out = np.zeros_like((1, n_bands, n_layers, n_grid), dtype=np.float64)

    sqrtln2 = np.sqrt(np.log(2.0))
    inv_sqrt_pi = 1 / np.sqrt(np.pi)

    doppler_prefactor = np.empty((n_layers,), dtype=np.float64)

    inv_mass = 1 / species_mass
    doppler_coef = const_sqrt_2_NA_kB_log2_on_c * math.sqrt(inv_mass)
    temp_max = temperatures[0]
    for l in numba.prange(n_layers):
        temp_l = temperatures[l]
        doppler_prefactor[l] = doppler_coef * math.sqrt(temp_l)
        if temp_l > temp_max:
            temp_max = temp_l

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


# ------------------------------------- ALL LAYER, SAMPLED VOIGT CALCULATIONS -------------------------------------
@numba.njit(parallel=True, cache=True, error_model="numpy")
def _band_profile_sampled_voigt(
        profile_buffer: npt.NDArray[np.float64],
        wn_grid: npt.NDArray[np.float64],
        id_f: npt.NDArray[np.int32],
        id_i: npt.NDArray[np.int32],
        id_agg_f: npt.NDArray[np.int32],
        id_agg_i: npt.NDArray[np.int32],
        band_indices: npt.NDArray[np.int32],
        n_lookup: npt.NDArray[np.float64],
        g_lookup: npt.NDArray[np.float64],
        inv_g_lookup: npt.NDArray[np.float64],
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
    id_agg_f : ndarray, shape (n_trans,)
    id_agg_i : ndarray, shape (n_trans,)
    band_indices : ndarray, shape (n_trans,)
    n_lookup : ndarray, shape (n_states + 1, n_layers)
        Lookup table for n_frac, avoiding extremely memory intensive joins.
    g_lookup : ndarray, shape (n_states + 1, )
        Lookup table for g, avoiding extremely memory intensive joins.
    inv_g_lookup : ndarray, shape (n_states + 1, )
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

    band_group_indices, group_starts, group_ends = _find_groups_from_ids(
        id_agg_f=id_agg_f,
        id_agg_i=id_agg_i,
        band_indices=band_indices,
    )
    n_bands_in_batch = group_starts.shape[0]

    cutoff = 25.0
    sqrt2 = np.sqrt(2.0)
    sqrt2_NA_kB_on_c = sqrt2 * const_sqrt_NA_kB_on_c
    inv_sqrt_pi = 1 / np.sqrt(np.pi)

    # Per-layer per-transition: sigma_D, gamma_total, and profile coefficients
    inv_sigma_sqrt2 = np.empty((n_layers, n_trans), dtype=np.float64)  # Doppler sigma (std dev)
    gamma_total_lookup = np.empty((n_layers, n_states), dtype=np.float64)

    # gamma_lifetime = 1.0 / (const_4_pi_c * lifetimes)  # (n_trans,)
    gamma_lifetime_lookup = 1.0 / (const_4_pi_c * tau_lookup)

    for l in numba.prange(n_layers):
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
def _abs_emi_sampled_voigt_threadlocal(
        profile_buffer: npt.NDArray[np.float64],
        wn_grid: npt.NDArray[np.float64],
        id_f: npt.NDArray[np.float64],
        id_i: npt.NDArray[np.float64],
        n_lookup: npt.NDArray[np.float64],
        g_lookup: npt.NDArray[np.float64],
        inv_g_lookup: npt.NDArray[np.float64],
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
) -> None:
    """
    Uses thread-local buffers for accumulation while still parallelising over n_trans. More memory intensive but faster
    when n_trans is much larger. Should be used for larger diatomics like TiO which might be better off with
    super-lines. For small line lists like CO, OH this is slower due to the final accumualtion step.


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

    cutoff = 25.0
    sqrt2 = np.sqrt(2.0)
    sqrt2_NA_kB_on_c = sqrt2 * const_sqrt_NA_kB_on_c
    inv_sqrt_pi = 1 / np.sqrt(np.pi)

    # Per-layer per-transition: sigma_D, gamma_total, and profile coefficients
    inv_sigma_sqrt2 = np.empty((n_layers, n_trans), dtype=np.float64)  # Doppler sigma (std dev)
    gamma_total_lookup = np.empty((n_layers, n_states), dtype=np.float64)

    # gamma_lifetime = 1.0 / (const_4_pi_c * lifetimes)  # (n_trans,)
    gamma_lifetime_lookup = 1.0 / (const_4_pi_c * tau_lookup)

    for l in numba.prange(n_layers):
        temp_l = temperatures[l]
        pres_l = pressures[l]
        inv_sigma_sqrt2[l] = 1 / (energy_fi * sqrt2_NA_kB_on_c * np.sqrt(temp_l / species_mass))

        gamma_pressure_l = 0.0
        for b in range(n_broad):
            gamma_pressure_l += broad_gamma[b, l] * pres_l * (t_ref / temp_l) ** broad_n[b] / pressure_ref
        # gamma_total[l] = gamma_lifetime + gamma_pressure_l
        gamma_total_lookup[l] = gamma_lifetime_lookup + gamma_pressure_l

    # Voigt y-parameter: gamma_L / (sigma_D * sqrt(2)).
    # y_voigt = gamma_total * inv_sigma_sqrt2  # (n_layers, n_trans)

    abs_buffer = np.zeros((numba_num_threads, n_layers, n_grid), dtype=np.float64)
    emi_buffer = np.zeros((numba_num_threads, n_layers, n_grid), dtype=np.float64)

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
            emi_tl = emi_prefactor_t * n_f_l

            abs_l = abs_buffer[thread_id, l]
            emi_l = emi_buffer[thread_id, l]

            # Integral of Re[w(x,y)] dx = sqrt(pi), so the normalised Voigt is Re[w(z)] / (sigma * sqrt(2*pi)).
            norm = inv_sigma_sqrt2_tl * inv_sqrt_pi

            for j in range(j_start, j_end):
                # x = (nu - nu0) / (sigma * sqrt(2))
                wn_j = wn_grid[j]
                x_tj = (wn_j - energy_fi_t) * inv_sigma_sqrt2_tl
                voigt_val = _voigt_humlicek_w(x_tj, y_tl) * norm

                abs_l[j] += abs_tl * voigt_val
                emi_l[j] += emi_tl * voigt_val
    # Accumulate into profile_buffer.
    for l in numba.prange(n_layers):
        for k in range(n_grid):
            for t in range(numba_num_threads):
                profile_buffer[0, l, k] += abs_buffer[t, l, k]
                profile_buffer[1, l, k] += emi_buffer[t, l, k]


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _abs_emi_sampled_voigt(
        profile_buffer: npt.NDArray[np.float64],
        wn_grid: npt.NDArray[np.float64],
        id_f: npt.NDArray[np.float64],
        id_i: npt.NDArray[np.float64],
        n_lookup: npt.NDArray[np.float64],
        g_lookup: npt.NDArray[np.float64],
        inv_g_lookup: npt.NDArray[np.float64],
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
    profile_buffer : ndarray, shape (2, n_layers, n_grid)
    wn_grid : ndarray, shape (n_grid,)
    id_f : ndarray, shape (n_trans,)
    id_i : ndarray, shape (n_trans,)
    n_lookup : ndarray, shape (n_states + 1, n_layers)
        Lookup table for n_frac, avoiding extremely memory intensive joins.
    g_lookup : ndarray, shape (n_states + 1, )
        Lookup table for g, avoiding extremely memory intensive joins.
    inv_g_lookup : ndarray, shape (n_states + 1, )
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

    cutoff = 25.0
    sqrt2 = np.sqrt(2.0)
    sqrt2_NA_kB_on_c = sqrt2 * const_sqrt_NA_kB_on_c
    inv_sqrt_pi = 1 / np.sqrt(np.pi)

    inv_sigma_sqrt2 = np.empty((n_layers, n_trans), dtype=np.float64)  # Doppler sigma (std dev)
    gamma_total_lookup = np.empty((n_layers, n_states), dtype=np.float64)

    gamma_lifetime_lookup = 1.0 / (const_4_pi_c * tau_lookup)

    for l in numba.prange(n_layers):
        temp_l = temperatures[l]
        pres_l = pressures[l]
        inv_sigma_sqrt2[l] = 1 / (energy_fi * sqrt2_NA_kB_on_c * np.sqrt(temp_l / species_mass))

        gamma_pressure_l = 0.0
        for b in range(n_broad):
            gamma_pressure_l += broad_gamma[b, l] * pres_l * (t_ref / temp_l) ** broad_n[b] / pressure_ref
        gamma_total_lookup[l] = gamma_lifetime_lookup + gamma_pressure_l

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
        temp_l = temperatures[l]
        inv_sqrt_temp_on_m = 1.0 / np.sqrt(temp_l / species_mass)

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

            gamma_total_tl = gamma_total_lookup[l, id_f_t]
            inv_sigma_sqrt2_tl = inv_sqrt_temp_on_m / (energy_fi_t * sqrt2_NA_kB_on_c)
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
@numba.njit(parallel=True, cache=True, error_model="numpy")
def _continuum_band_profile_sampled_gauss_layered(
        profile_buffer: npt.NDArray[np.float64],
        wn_grid: npt.NDArray[np.float64],
        id_f: npt.NDArray[np.float32],
        id_i: npt.NDArray[np.float32],
        id_agg_f: npt.NDArray[np.int32],
        id_agg_i: npt.NDArray[np.int32],
        band_indices: npt.NDArray[np.int32],
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
    id_agg_f : ndarray, shape (n_trans,)
    id_agg_i : ndarray, shape (n_trans,)
    band_indices : ndarray, shape (n_trans,)
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

    band_group_indices, group_starts, group_ends = _find_groups_from_ids(
        id_agg_f=id_agg_f,
        id_agg_i=id_agg_i,
        band_indices=band_indices,
    )
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


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _continuum_sampled_gauss(
        profile_buffer: npt.NDArray[np.float64],
        wn_grid: npt.NDArray[np.float64],
        id_f: npt.NDArray[np.float64],
        id_i: npt.NDArray[np.float64],
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
