import os
import time
import subprocess
import pathlib
import typing as t
import numpy as np
import numpy.typing as npt

import pandas as pd
from astropy import units as u
from pathlib import Path
from .format import output_data

_EXOCROSS_PATH = r"/mnt/c/PhD/exocross/xcross.x"  # TODO: Store elsewhere.


def build_exocross_input(
    working_name: str,
    trans_files: t.List[str],
    temperature: u.Quantity,
    wn_grid: u.Quantity,
    pop_col_num: int,
    pressure: u.Quantity,
    spec_type: str,
    intensity_threshold: float,
    species_mass: float,
    profile: str,
    linear_grid: bool = True,
) -> Path:
    # working_name = f"{states_file.stem}__{temperature.value:.1f}K_{pressure.value:.2E}P_{spec_type}"
    input_file = working_name + f"_{spec_type}.inp"
    if spec_type[:3] not in ["abs", "emi"]:
        raise RuntimeError(
            f"Spectrum type not implemented: only 'abs' or 'emi' available. Value passed was {spec_type}"
        )
    with open(input_file, "w") as f:
        f.write(f"Temperature {temperature.value:.1f}\n")
        wn_start = wn_grid[0].value
        wn_end = wn_grid[-1].value
        f.write(f"Range {wn_start} {wn_end}\n")
        if linear_grid:
            f.write(f"Npoints {len(wn_grid)}\n")
        else:
            f.write(create_grid(wn_start=wn_start, wn_end=wn_end))
        f.write(f"NON-LTE\ndensity {pop_col_num}\nend\n")
        f.write(f"{'Absorption' if spec_type == 'abs' else 'Emission'}\n")
        f.write(f"{profile}\n")
        # f.write("Doppler\n")
        f.write(f"Threshold {intensity_threshold:.2E}\n")
        f.write(f"Mass {species_mass:.10f}\n")
        f.write(f"Pressure {pressure.value:.2E}\n")
        # Include pressure broadening parameters?
        f.write(f"Output {working_name + f"_{spec_type}"}\n")
        f.write(f"States {working_name + ".states"}\n")
        f.write(f"Transitions\n")
        for trans_file in trans_files:
            f.write(f"{trans_file}\n")
        f.write("end\n")
    input_file = (Path(os.getcwd()) / input_file).resolve()
    return input_file


def create_grid(wn_start: np.float64, wn_end: np.float64, resolution: np.float64 | None = 15000) -> str:
    split_points = np.array((0, 1000, 20000, 50000, 100000))
    split_steps = [100, 1000, 2000, 3000, 5000]
    out_string = "grid\n"
    range_start = wn_start
    range_counter = 0
    while range_counter <= 100:
        range_counter += 1
        range_step = split_steps[np.searchsorted(split_points, range_start) - 1]
        range_end = range_start + (range_step - range_start % range_step)
        if range_end > wn_end:
            range_end = wn_end
        range_npoints = int(np.ceil((range_end - range_start) / (range_start / resolution)))
        out_string += f"Range {range_start:.1f} {range_end:.1f} Npoints {range_npoints}\n"
        if range_end >= wn_end:
            break
        range_start = range_end
    out_string += "end\n"
    return out_string


def run_exocross(
    exocross_path: str | Path,
    input_file: Path,
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    working_file_root = ".".join(input_file.name.split(".")[:-1])
    out_file = (input_file.parent / (working_file_root + ".txt")).resolve()
    exocross_process = subprocess.Popen(
        f"{exocross_path} < {input_file} > {out_file}",
        env=dict(os.environ),
        shell=True,
    )
    exocross_process.communicate()
    xsec_file = (input_file.parent / (working_file_root + ".xsec")).resolve()
    if xsec_file.is_file():
        result = pd.read_csv(xsec_file, sep=r"\s+", names=["wn", "xsec"])
        Path.unlink(out_file)
        Path.unlink(xsec_file)
        return result["wn"].to_numpy(), result["xsec"].to_numpy()
    else:
        raise FileNotFoundError(f"Expected ExoCross Xsec output at {xsec_file}; file not found.")


def compute_xsec(
    nlte_states: pd.DataFrame,
    states_file: pathlib.Path,
    agg_col_nums: t.List[int],
    trans_files: t.List[str],
    temperature: u.Quantity,
    wn_grid: u.Quantity,
    pop_col_num: int,
    pressure: u.Quantity,
    intensity_threshold: float,
    species_mass: float,
    profile: str = "Voigt",
    timing: bool = False,
    linear_grid: bool = True,
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if profile.lower() not in ("voigt", "doppler"):
        raise RuntimeError(f"Invalid profile provided: {profile} - only Voigt and Doppler have been tested.")
    working_name = f"{states_file.stem}__{temperature.value:.1f}K_{pressure.value:.2E}P"
    states_path = pathlib.Path(working_name + ".states").resolve()
    output_data(
        nlte_states,
        states_path,
        fortran_format_list=["i12", "f12.6", "i6", "f7.1"] + ["a10"] * len(agg_col_nums) + ["E13.6"],
    )
    # NB: ExoCross requires a number in column 4, even if you've not loaded the standard ExoMol format J column into
    # memory. You cna just use id_agg in its place and it does not impact the calculation.

    abs_xsec = None
    emi_xsec = None
    for spec in ["abs", "emi"]:
        exocross_input = build_exocross_input(
            working_name=working_name,
            trans_files=trans_files,
            temperature=temperature,
            wn_grid=wn_grid,
            pop_col_num=pop_col_num,
            pressure=pressure,
            spec_type=spec,
            intensity_threshold=intensity_threshold,
            species_mass=species_mass,
            profile=profile,
            linear_grid=linear_grid,
        )
        exocross_start_time = time.perf_counter()
        xcross_wn_grid, xcross_xsec = run_exocross(
            exocross_path=_EXOCROSS_PATH,
            input_file=exocross_input,
        )
        Path.unlink(exocross_input)
        if timing:
            print(f"ExoCross {spec} duration = {time.perf_counter() - exocross_start_time}")
        if spec == "abs":
            abs_xsec = xcross_xsec
        else:
            emi_xsec = xcross_xsec

    Path.unlink(states_path)

    return xcross_wn_grid, abs_xsec, emi_xsec


# print(create_grid(wn_start=2000.0, wn_end=20000.0))
