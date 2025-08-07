import logging
import os
import pathlib
import numba

(pathlib.Path(os.getcwd()) / "./outputs").resolve().mkdir(exist_ok=True)
output_dir = (pathlib.Path(os.getcwd()) / "./outputs").resolve()

log = logging.getLogger()
log.setLevel(logging.INFO)

# stream_handler = logging.StreamHandler()
# stream_formatter = logging.Formatter("%(asctime)s [%(levelname)s]  %(message)s")
# stream_handler.setFormatter(stream_formatter)
# log.addHandler(stream_handler)

file_handler = logging.FileHandler(
    filename=(output_dir / "nlte.log").resolve(), encoding="utf-8", mode="a"
)
file_formatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
file_handler.setFormatter(file_formatter)
log.addHandler(file_handler)

log.info(f"Writing outputs to {output_dir}.")

_DEFAULT_NUM_THREADS = 20
if numba.get_num_threads() != _DEFAULT_NUM_THREADS:
    log.info(f"Numba defaulting to {numba.get_num_threads()} threads: setting to {_DEFAULT_NUM_THREADS}.")
    numba.set_num_threads(_DEFAULT_NUM_THREADS)

_DEFAULT_CHUNK_SIZE = 10000000
_N_GH_QUAD_POINTS = 30
_INTENSITY_CUTOFF = 1e-100
