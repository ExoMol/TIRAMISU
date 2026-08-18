import logging
import multiprocessing as mp
import os
import pathlib
import numba

from logging.handlers import QueueHandler, QueueListener

# Workflow configurations.
_DEFAULT_NUM_THREADS = 20
_DEFAULT_CHUNK_SIZE = 10_000_000
_PARQUET_BATCH_SIZE = 200_000_000
_DASK_BLOCK_SIZE = "512MB"
_N_GH_QUAD_POINTS = 30
_INTENSITY_CUTOFF = 1e-100
_VMR_THRESHOLD = 1e-20

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

if numba.get_num_threads() != _DEFAULT_NUM_THREADS:
    # logging.info(f"Numba defaulting to {numba.get_num_threads()} threads: setting to {_DEFAULT_NUM_THREADS}.")
    numba.set_num_threads(_DEFAULT_NUM_THREADS)

# Configure spawn context for logging in ProcessPools.
ctx = mp.get_context("spawn")
log_queue = ctx.Queue()


def setup_logging_main(logfile: str = "nlte.log", level: int = logging.INFO, verbose: int | None = None):
    root = logging.getLogger()
    root.handlers.clear()
    if verbose is not None:
        root.setLevel(logging.INFO - verbose)
    else:
        root.setLevel(level)

    # Queue handler (thread/process safe).
    # queue_handler = QueueHandler(log_queue)
    # root.addHandler(queue_handler)

    # File handler (runs in separate thread).
    file_handler = logging.FileHandler(
        filename=(output_dir / logfile).resolve(),
        encoding="utf-8",
        mode="a"
    )
    # file_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s] %(message)s")
    file_formatter = logging.Formatter("%(asctime)s [%(processName)s] [%(levelname)-5.5s] %(message)s")
    file_handler.setFormatter(file_formatter)

    # Start queue listener (handles actual writes).
    listener = QueueListener(log_queue, file_handler)
    listener.start()

    # Store listener for clean-up.
    root.addHandler(QueueHandler(log_queue))
    root._queue_listener = listener

    logging.info(f"Writing outputs to {output_dir}.")
    logging.info(f"Numba num threads set to {numba.get_num_threads()}.")


def worker_logging_init(queue):
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    root.addHandler(logging.handlers.QueueHandler(queue))
