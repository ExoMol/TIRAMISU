"""Cross-section drivers: absorption/emission (abs_emi_xsec) and continuum (continuum_xsec) cross-sections
computed straight from the transitions, with or without super-lines, on the CPU or the GPU."""
import logging
import time
import pathlib
import typing as t

import numpy as np
import polars as pl
from astropy import units as u
from numpy import typing as npt

from .. import config
from ..config import _DEFAULT_NUM_THREADS, _LOG_VERBOSE_2
from .read import iter_trans_batches, read_ahead
from .lineshape import grid_bin_edges, gamma_lifetime_lookup_from_tau
from .superline import (
    accumulate_superline_batch, accumulate_continuum_superline_batch, broaden_superline_buffer,
    broaden_continuum_superline_buffer,
)
from .sampled import abs_emi_sampled_voigt, continuum_sampled_gauss

log = logging.getLogger(__name__)


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
    bin_edges = grid_bin_edges(wn_arr)

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
    # Super-lines are accumulated on the GPU when that pays off (gpu.gpu_pays_off): the buffer stays on the
    # device for the whole batch loop and is copied back once.
    d_buffer = None
    if do_super_lines and config.USE_GPU:
        from . import gpu
        if gpu.gpu_pays_off(trans_files=trans_files, n_layers=n_nlte_layers,
                            buffer_bytes=profile_buffer.nbytes, label="abs_emi_xsec"):
            d_lookups = gpu.to_device(bin_edges=bin_edges, n_frac=n_lookup, g=g_lookup, inv_g=inv_g_lookup,
                                      gamma_lifetime=gamma_lifetime_lookup)
            d_buffer = gpu.device_zeros(profile_buffer.shape)

    process_time = time.perf_counter()

    for trans_file in trans_files:
        log.info(f"Processing file {trans_file}.")
        for id_f, id_i, a_fi, energy_fi in read_ahead(batches=iter_trans_batches(
                trans_file=trans_file,
                energy_lookup=energy_lookup,
                wn_min=wn_min,
                wn_max=wn_max,
        )):
            if d_buffer is not None:
                gpu.accumulate_superlines(profile_buffer=d_buffer, lookups=d_lookups, id_f=id_f, id_i=id_i,
                                          a_fi=a_fi, energy_fi=energy_fi)
            elif do_super_lines:
                accumulate_superline_batch(
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
                abs_emi_sampled_voigt(
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
        abs_xsec, emi_xsec = broaden_superline_buffer(
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
    bin_edges = grid_bin_edges(wn_arr)

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
        for id_f, id_i, a_fi, energy_fi in read_ahead(batches=iter_trans_batches(
                trans_file=trans_file,
                energy_lookup=energy_lookup,
                wn_min=wn_min,
                wn_max=wn_max,
        )):
            if do_super_lines:
                accumulate_continuum_superline_batch(
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
                continuum_sampled_gauss(
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
        abs_xsec = broaden_continuum_superline_buffer(
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
