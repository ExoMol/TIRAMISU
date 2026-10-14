import logging
import os
import pathlib
import numba

# Workflow configurations.
_DEFAULT_NUM_THREADS = 20
# Rows per parquet read batch in profiles._iter_trans_batches, for both the NLTE rates/profiles (compute_rates_profiles)
# and the final cross-sections (abs_emi_xsec, continuum_xsec). Benchmarked by tests/h2o_batch_size_bench.py on the full
# H2O POKAZATEL line list (5 parquet files in 20M-row groups, 2.2e9 lines on the grid, 43 NLTE layers, CPU super-lines):
# - compute_rates_profiles (20453-point grid, 36 bands): ~115 s for 20M-100M rows per batch and ~133 s at 200M, the
#   extra time all in reading (the reader joins row groups into each batch); accumulation took 67-75 s whatever the
#   number of batches (15-113). Peak memory grows by ~65 bytes per batch row: 5.4 GB at 20M, 15.8 GB at 200M.
# - abs_emi_xsec (R = 15000 grid, 46705 points): 104 s at 20M, 106 s at 66.7M, 109 s at 6.7M and 114 s at 200M, with
#   peaks of 4.0, 7.2, 3.3 and 14.7 GB. Its super-line buffer has no band axis and is small next to a batch, so it needs
#   no smaller batch size.
# One row group per batch is the fastest and leanest for these files; re-check for line lists with other row groups.
_PARQUET_BATCH_SIZE = 20_000_000
_DASK_BLOCK_SIZE = "512MB"
_INTENSITY_CUTOFF = 1e-100
_VMR_THRESHOLD = 1e-20

# GPU configuration.
# Number of threads per CUDA block for the profiles_gpu.py kernels.
_GPU_THREADS_PER_BLOCK = 256
# Cost model deciding whether super-lines are accumulated on the GPU (profiles_gpu.gpu_pays_off), for both
# compute_rates_profiles and abs_emi_xsec: the GPU is used when
#     (transitions x NLTE layers) x _GPU_LINE_LAYER_SAVING_S >= _GPU_FIXED_COST_S + buffer bytes / _GPU_TRANSFER_BYTES_PER_S
# i.e. when its speed-up outweighs the fixed costs of a run (CUDA context, lookup upload, first kernel calls, with the
# kernels cached) plus copying the profile buffer back. Measured by tests/gpu_xsec/calibrate_gpu_cost.py on an RTX 4060
# Laptop GPU with a 20-thread CPU, on 93.5M H2O POKAZATEL lines, with the sorted-segment GPU kernels: at 43 NLTE layers
# the GPU (batch uploads included) is 2.6-3.3x faster than the bin-owned CPU kernels, saving 0.47-0.59 ns per
# line-layer (0.94-1.19 ns at 10 layers, where the CPU's per-line costs weigh more, so the 43-layer value is the
# conservative one); the CUDA context takes 0.7-1.8 s and uploads plus first calls ~0.3 s; a 1 GB band buffer copies
# back at 2.6 GB/s. The GPU then pays off from ~5e9 line-layers (~120M lines at 43 layers). Re-measure on other hardware.
_GPU_LINE_LAYER_SAVING_S = 0.47e-9
_GPU_FIXED_COST_S = 2.0
_GPU_TRANSFER_BYTES_PER_S = 2.6e9


def _detect_gpu_available() -> bool:
    try:
        from numba import cuda
        return bool(cuda.is_available())
    except Exception:
        return False


_GPU_AVAILABLE = _detect_gpu_available()

# Runtime toggle for the CPU/GPU dispatch in profiles.py. Defaults to whatever's detected;
# override with TIRAMISU_USE_GPU=0/1 (e.g. to force a CPU-only run for benchmarking/debugging
# on a machine that does have a GPU), or set tiramisu.config.USE_GPU directly at runtime.
_env_use_gpu = os.environ.get("TIRAMISU_USE_GPU")
if _env_use_gpu is not None:
    USE_GPU = _env_use_gpu.strip().lower() not in ("0", "false", "no", "")
else:
    USE_GPU = _GPU_AVAILABLE
# Whether super-line broadening also runs on the GPU when the GPU built the profile buffer. Broadening evaluates the
# Voigt profile in double precision, which consumer GPUs run at a small fraction of their single-precision rate; on
# those it is faster to copy the accumulated buffer back once and broaden on the CPU. Measured on the full H2O line list
# (tests/gpu_xsec/bench_broaden.py): the per-(band, layer) kernel runs at 0.09x the CPU, and even one thread per output
# point only reaches 0.3x (bands) to 0.9x (no bands); the broadening takes ~1.5 s per call on the CPU.
GPU_BROADEN_SUPERLINES = False
# Whether direct (non-super-line) sampled Voigt profiles use the GPU. Off by default: on the hardware above the GPU was
# slower than the CPU in every case measured (double-precision Voigt evaluation, and one thread per (band, layer), which
# leaves the GPU mostly idle when a few bands hold most of the lines).
GPU_DIRECT_PROFILES = False

# Logging configurations
_LOG_FLOAT_FMT = "6.3E"
_LOG_ARRAY_FMT = {'float': lambda x: format(x, _LOG_FLOAT_FMT)}
_LOG_VERBOSE_1 = logging.INFO - 1
_LOG_VERBOSE_2 = logging.INFO - 2
_LOG_VERBOSE_3 = logging.INFO - 3

logging.addLevelName(_LOG_VERBOSE_1, "VERB1")
logging.addLevelName(_LOG_VERBOSE_2, "VERB2")
logging.addLevelName(_LOG_VERBOSE_3, "VERB3")

os.environ["RUST_BACKTRACE"] = "1"

(pathlib.Path(os.getcwd()) / "./outputs").resolve().mkdir(exist_ok=True)
output_dir = (pathlib.Path(os.getcwd()) / "./outputs").resolve()

# Threading layer: TBB, unless NUMBA_THREADING_LAYER names another. The transition batch loops call parallel kernels
# from two threads at once (profiles._read_ahead), which aborts the process under numba's fallback workqueue layer;
# with TBB pinned, a missing TBB raises here instead. OpenMP (omp) is also threadsafe and measured the same speed
# (tests/prefetch_nogil_bench.py, tests/outputs/layer_compare/). Set before set_num_threads, which starts the threads.
numba.config.THREADING_LAYER = os.environ.get("NUMBA_THREADING_LAYER", "tbb")
numba.set_num_threads(_DEFAULT_NUM_THREADS)


def setup_logging_main(logfile: str = "nlte.log", verbose: int = 0):
    """Log to output_dir / logfile at INFO (verbose=0), or down to the VERB1-VERB3 levels (verbose=1-3)."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO - verbose)

    file_handler = logging.FileHandler(
        filename=(output_dir / logfile).resolve(),
        encoding="utf-8",
        mode="a"
    )
    file_formatter = logging.Formatter("%(asctime)s [%(processName)s] [%(levelname)-5.5s] %(message)s")
    file_handler.setFormatter(file_formatter)
    root.addHandler(file_handler)

    logging.info(f"Writing outputs to {output_dir}.")
    logging.info(f"Numba num threads set to {numba.get_num_threads()} ({numba.threading_layer()} threading layer).")
    logging.info(f"GPU acceleration: {'enabled' if USE_GPU else 'disabled'} (device available: {_GPU_AVAILABLE}).")
