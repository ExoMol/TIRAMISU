import functools
import typing as t
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

# TODO: Deprecated - REMOVE.

def output_data(
    data: pd.DataFrame,
    filename: t.Union[str, Path],
    fortran_format_list: t.List[str],
    n_workers: int = 8,
    append: bool = False,
) -> None:
    data_out = data.copy()

    worker = functools.partial(format_row, fortran_format_list)
    if append:
        file_mode = "a"
    else:
        file_mode = "w+"
    with open(filename, file_mode, newline="\n") as f, ThreadPoolExecutor(
        max_workers=n_workers
    ) as e:
        for out_row in e.map(worker, data_out.itertuples(index=False)):
            f.write(out_row + "\n")


def format_row(fortran_format_list: t.List, data_row: t.Tuple) -> str:
    out_row = ""
    for i in range(0, len(data_row)):
        if i >= 1:
            out_row += " "
        out_row += fortran_format(val=data_row[i], fmt=fortran_format_list[i])
    return out_row


def fortran_format(val: str, fmt: str) -> str:
    fmt_letter = fmt[0]
    fmt = fmt[1:]
    if fmt_letter in ["a", "A"] or (fmt_letter in ["i", "I"] and pd.isna(val)):
        if len(fmt) == 0:
            return val
        else:
            return "{val:>{fmt}}".format(val=val, fmt=fmt)
    if fmt_letter in ["e", "E", "f"]:
        val = float(val)
        fmt_parts = fmt.split(".")
        lhs_length = int(fmt_parts[0]) - int(fmt_parts[1])
        if val > 0 and (log_val := np.log10(val) + 1) > lhs_length:
            rhs_dif = int(np.ceil(log_val) - lhs_length)
            fmt = f"{int(fmt_parts[0])}.{int(fmt_parts[1]) - rhs_dif}"
        return "{val:{fmt}{fmt_letter}}".format(val=val, fmt=fmt, fmt_letter=fmt_letter)
    elif fmt_letter in ["g", "G"]:
        val = float(val)
        return "{val:#{fmt}{fmt_letter}}".format(
            val=val, fmt=fmt, fmt_letter=fmt_letter
        )
    elif fmt_letter in ["i", "I"]:
        val = int(val)
        if "." in fmt:
            fmt_w = int(fmt.split(".")[0])
            fmt_m = int(fmt.split(".")[1])
            return "{val:>{fmt_w}}".format(val=str(val).zfill(fmt_m), fmt_w=fmt_w)
        else:
            return "{val:{pad}d}".format(val=val, pad=fmt)
