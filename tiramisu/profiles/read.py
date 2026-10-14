"""Reading ExoMol transitions in batches: parquet/dask readers, the numba batch filter and a read-ahead thread."""
import pathlib
import typing as t
from concurrent.futures import ThreadPoolExecutor

import numba
import numpy as np
from pyarrow import parquet as pq
from dask import dataframe as dd
from numpy import typing as npt

from ..config import _DEFAULT_NUM_THREADS, _PARQUET_BATCH_SIZE, _DASK_BLOCK_SIZE


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


def iter_trans_batches(
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


def read_ahead(batches: t.Iterator[t.Tuple]) -> t.Iterator[t.Tuple]:
    """
    Yields the items of batches (from :func:`iter_trans_batches`), reading the next one in a background thread while
    the caller works on the current one. The batch loops' numba kernels are nogil=True so that both threads run at
    once (1.1-1.6x on the full H2O line list, CPU and GPU, tests/perf/prefetch_nogil_bench.py); the caller does all
    accumulation, in batch order, so results are bit-identical. Needs a threadsafe numba threading layer (config.py
    sets TBB).
    """
    # set_num_threads is thread-local: the reader thread would otherwise run at NUMBA_NUM_THREADS.
    with ThreadPoolExecutor(max_workers=1, initializer=numba.set_num_threads, initargs=(_DEFAULT_NUM_THREADS,)) as pool:
        future = pool.submit(next, batches, None)
        while (batch := future.result()) is not None:
            future = pool.submit(next, batches, None)
            yield batch
