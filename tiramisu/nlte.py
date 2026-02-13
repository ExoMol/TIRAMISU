import abc
import pathlib
import typing as t
import numpy.typing as npt
import numpy as np
import pandas as pd
import dask.dataframe as dd
import polars as pl
import numba
import math

from phoenix4all import get_spectrum

from astropy import units as u
from astropy import constants as ac

from scipy.integrate import simpson, cumulative_simpson
from scipy.special import roots_hermite, erf

from .config import log, _DEFAULT_CHUNK_SIZE, _N_GH_QUAD_POINTS, _INTENSITY_CUTOFF, _DEFAULT_NUM_THREADS

# Constants with units:
ac_h_on_8_c = ac.h.cgs / (8 * ac.c.cgs)

ac_h_c_on_kB = ac.h * ac.c.cgs / ac.k_B
ac_2_hc = 2 * ac.h * ac.c.cgs

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

ac_2_h_on_c_sq = 2 * ac.h / ac.c ** 2
ac_h_on_kB = ac.h / ac.k_B

# Dimensionless version for numba
const_amu = ac.u.cgs.value
const_h_on_8_c = ac_h_on_8_c.value
const_h_c_on_kB = ac_h_c_on_kB.value
const_2_hc = ac_2_hc.value
const_4_pi_c = ac_4_pi_c.value
const_8_pi_c = ac_8_pi_c.value
const_16_pi_c = ac_16_pi_c.value
const_8_pi_five_halves_c = ac_8_pi_five_halves_c.value
const_h_c_on_4_pi = ac_h_c_on_4_pi.value
const_h_c_on_4_pi_five_halves = ac_h_c_on_4_pi_five_halves.value
const_h_c_on_8_pi = ac_h_c_on_8_pi.value
const_sqrt_NA_kB_on_c = ac_sqrt_NA_kB_on_c.value
const_sqrt_2_NA_kB_log2_on_c = ac_sqrt_2_NA_kB_log2_on_c.value
const_2_h_on_c_sq = ac_2_h_on_c_sq.value
const_h_on_kB = ac_h_on_kB.value
const_2_pi_h_c_sq_on_sigma_sba = (
    (2 * np.pi * ac.h * ac.c.cgs ** 2 / ac.sigma_sb).to(u.K ** 4 * u.cm ** 4, equivalencies=u.spectral()).value
)
const_2_pi_c_kB = (2 * np.pi * ac.c.cgs * ac.k_B.cgs).value


# TODO: For NANs in state lifetimes; treat as inf? They imply inf but often they exist because of transition energy
#  cutoffs during computation and not because the state has no deexcitation pathways.


def create_erf_lut(
        n_points: int = 20000, arg_max: float = 6.0
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Create lookup table for erf() for samlpling by :func:`tiramisu.nlte.get_erf_lut`
    Only stores [0, arg_max] due to symmetry.

    Parameters
    ----------
    n_points: int
        Number of linearly spaced points up to arg_max.
    arg_max: float
        Maximum argument value for erf().
    Returns
    -------
        Values of erf() at n_points linearly spaced points up to arg_max.
    """
    args = np.linspace(0, arg_max, n_points, dtype=np.float64)
    erf_values = erf(args)
    return args, erf_values


# --------------- ERF LOOKUP TABLE ---------------
global_erf_arg_max = 6.0
_, global_erf_lut = create_erf_lut(n_points=20000, arg_max=global_erf_arg_max)


# ------------------------------------------------


@numba.njit(cache=True, error_model="numpy", inline="always")
def get_erf_lut(erf_arg: float, erf_lut: npt.NDArray[np.float64], erf_arg_max: float) -> float:
    """
    Fast erf lookup with linear interpolation between points. Uses symmetry; erf(-x) = -erf(x).

    Parameters
    ----------
    erf_arg: float
        Argument to look up erf() result for.
    erf_lut: ndarray
        Lookup table to search.
    erf_arg_max: float
        Maximum erf() argument value associated with the lookup table

    Returns
    -------
        Value of erf() at given argument, linearly interpolated between lookup table points.
    """
    n_points = erf_lut.shape[0]

    sign = 1.0 if erf_arg >= 0 else -1.0
    abs_arg = abs(erf_arg)

    # Saturation for large arguments.
    if abs_arg >= erf_arg_max:
        return sign * 1.0

    # Map to index; linear interpolation.
    float_idx = (abs_arg / erf_arg_max) * (n_points - 1)
    idx = int(float_idx)
    frac = float_idx - idx

    # Bounds check for safety.
    if idx >= n_points - 1:
        return sign * erf_lut[n_points - 1]

    # Linear interpolation.
    val = erf_lut[idx] * (1.0 - frac) + erf_lut[idx + 1] * frac
    return sign * val


def _sum_profiles(group: pl.DataFrame) -> pl.DataFrame:
    profile_label = group.columns[-1]
    summed_profiles = np.stack(group[profile_label].to_numpy(), axis=0).sum(axis=0)
    return pl.DataFrame({
        "id_agg_f": [group["id_agg_f"][0]],
        "id_agg_i": [group["id_agg_i"][0]],
        profile_label: [summed_profiles],
    })


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _build_xsec(
        profiles: npt.NDArray[np.float64],
        offsets: npt.NDArray[int],
        start_idxs: npt.NDArray[int],
        key_idx_array: npt.NDArray[int],
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
        1-dimensional array storing populations of state ID corresponding to arry index.
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


class CompactProfile:
    __slots__ = ["temporary_profiles", "profiles", "offsets", "start_idxs", "key_idx_map", "key_lookup"]

    def __init__(self):
        self.temporary_profiles: pl.DataFrame | None = None
        self.profiles: npt.NDArray[np.float64] | None = None
        self.offsets: npt.NDArray[int] | None = None
        self.start_idxs: npt.NDArray[int] | None = None
        self.key_idx_map: npt.NDArray[int] | None = None
        self.key_lookup: t.Dict[t.Tuple[int, int] | int, int] = {}

    def add_batch(self, batch: pl.DataFrame) -> None:
        """
        batch must contain id_agg_f and id_agg_i columns plus a third column containing the relevant profile.

        :param batch:
        :return:
        """
        if self.temporary_profiles is None:
            self.temporary_profiles = batch
        else:
            self.temporary_profiles = pl.concat([self.temporary_profiles, batch])
            self.temporary_profiles = self.temporary_profiles.group_by("id_agg_f", "id_agg_i").map_groups(_sum_profiles)

    def finalise(self) -> None:
        """
        Trimmed profiles are stored contiguously in self.profiles.

        The starting indices within self.profiles of each individual profile is contained within self.offsets. An extra
        terminator is stored at the end of self.offsets equal to the total length of self.profiles; this is so the start
        and end indices of a given profile can always be obtained by looking at the current and next offset.

        The starting position of each profile on the main wavenumber grid is stored at the corresponding index in
        self.start_idxs.

        The upper and lower state IDs are stored in self.key_idx_map; the index of theh first dimension of this array
        matches the corresponding index in self.offsets and self.start_idxs. This is used for fast cross-section
        reconstruction in :func:`tiramisu.nlte.CompactProfile.build_xsec`. A fast dictionary lookup for accessing
        individual bands is stored in self.key_lookup, used by :func:`tiramisu.nlte.CompactProfile.get_profile`.

        Returns
        -------

        """
        num_profiles = self.temporary_profiles.height
        if num_profiles == 0:
            log.warn("CompactProfile finalising with 0 inputs!")
            return

        n_cols = len(self.temporary_profiles.columns)
        if n_cols not in (2, 3):
            raise ValueError("Unsupported number of columns in CompactProfile input.")

        if n_cols == 3:
            # Key is a tuple of the first two elements: (row[0], row[1])
            key_selector = lambda row_data: (row_data[0], row_data[1])
        else:  # n_cols == 2
            # Key is the first element: row[0]
            key_selector = lambda row_data: (-1, row_data[0])

        profile_idx = n_cols - 1

        estimate_profile_len = len(self.temporary_profiles.row(0)[profile_idx])
        self.profiles = np.zeros(num_profiles * estimate_profile_len, dtype=np.float64)
        # Offsets contains a terminator at the end of the array.
        self.offsets = np.zeros(num_profiles + 1, dtype=int)
        self.start_idxs = np.zeros(num_profiles, dtype=int)
        self.key_idx_map = np.empty((num_profiles, 2), dtype=int)

        current_offset = 0
        store_idx = 0
        for idx, row in enumerate(self.temporary_profiles.iter_rows(named=False)):
            key = key_selector(row)
            profile = np.array(row[profile_idx])

            above = profile >= _INTENSITY_CUTOFF
            if not np.any(above):
                # Nothing above threshold: skip or store empty?
                continue
            start_idx = np.argmax(above)
            end_idx = len(profile) - np.argmax(above[::-1])
            trimmed = profile[start_idx:end_idx]

            length = len(trimmed)
            if current_offset + length > len(self.profiles):
                remaining = num_profiles - idx
                self.profiles.resize(len(self.profiles) + remaining * length, refcheck=False)

            self.profiles[current_offset: current_offset + length] = trimmed
            self.offsets[store_idx] = current_offset
            self.start_idxs[store_idx] = start_idx
            self.key_idx_map[store_idx, :] = key
            self.key_lookup[key] = store_idx

            current_offset += length
            store_idx += 1

        # num_final_profiles = len(self.key_idx_map)
        num_final_profiles = store_idx + 1
        if num_final_profiles < num_profiles:
            # Some profiles where skipped (below cut-off); resize arrays.
            self.offsets = self.offsets[:num_final_profiles + 1]
            self.start_idxs = self.start_idxs[:num_final_profiles]
        # Add terminator and trim profiles to this length, in case it was overestimated or extended.
        self.offsets[-1] = current_offset
        self.profiles = self.profiles[:current_offset]
        log.info(f"Finalised CompactProfile with {current_offset} points for {len(self.start_idxs)} bands.")

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

    def add_batch(self, batch: pl.DataFrame, layer_idx: int) -> None:
        """
        The batch is a polars DataFrame containing id_agg_f and id_agg_i keys plus arrays representing each profile.

        :param batch:
        :param layer_idx:
        :return:
        """
        self.abs_profiles[layer_idx].add_batch(
            batch.select(pl.col("id_agg_f"), pl.col("id_agg_i"), pl.col("abs_profile"))
        )
        self.ste_profiles[layer_idx].add_batch(
            batch.select(pl.col("id_agg_f"), pl.col("id_agg_i"), pl.col("ste_profile"))
        )
        self.spe_profiles[layer_idx].add_batch(
            batch.select(pl.col("id_agg_f"), pl.col("id_agg_i"), pl.col("spe_profile"))
        )

    def finalise(self) -> None:
        for layer_idx in range(self.n_layers):
            self.abs_profiles[layer_idx].finalise()
            self.ste_profiles[layer_idx].finalise()
            self.spe_profiles[layer_idx].finalise()

    def get_profiles(
            self, layer_idx: int, key: t.Tuple[int, int]
    ) -> t.Tuple[t.Tuple[npt.NDArray, int], t.Tuple[npt.NDArray, int], t.Tuple[npt.NDArray, int]]:
        return (
            self.abs_profiles[layer_idx].get_profile(key),
            self.ste_profiles[layer_idx].get_profile(key),
            self.spe_profiles[layer_idx].get_profile(key),
        )

    def get_profile(self, layer_idx: int, key: t.Tuple[int, int], profile_type: str) -> t.Tuple[npt.NDArray, int]:
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


class ContinuumProfileStore:
    __slots__ = ["n_layers", "abs_profiles"]

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        # Storage for final profiles.
        self.abs_profiles = [CompactProfile() for _ in range(n_layers)]  # Absorption
        # self.ste_profiles = [CompactProfile() for _ in range(n_layers)]  # Stimulated Emission
        # self.spe_profiles = [CompactProfile() for _ in range(n_layers)]  # Spontaneous Emission

    def add_batch(self, batch: pl.DataFrame, layer_idx: int) -> None:
        """
        The batch is a polars DataFrame containing id_agg_i keys plus arrays representing each profile.
        Continuum profiles are currently agnostic of any upper state aggregated identifier, as individual continuum
        states are not currently treated distinctly.


        :param batch:
        :param layer_idx:
        :return:
        """
        self.abs_profiles[layer_idx].add_batch(
            batch.select(pl.col("id_agg_i"), pl.col("abs_profile"))
        )
        # self.ste_profiles[layer_idx].add_batch(
        #     batch.select(pl.col("id_agg_i"), pl.col("ste_profile"))
        # )
        # self.spe_profiles[layer_idx].add_batch(
        #     batch.select(pl.col("id_agg_i"), pl.col("spe_profile"))
        # )

    def finalise(self) -> None:
        for layer_idx in range(self.n_layers):
            self.abs_profiles[layer_idx].finalise()
            # self.ste_profiles[layer_idx].finalise()
            # self.spe_profiles[layer_idx].finalise()

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

    def get_keys(self, layer_idx: int, profile_type: str):
        if profile_type == "abs":
            return self.abs_profiles[layer_idx].key_lookup.keys()
        else:
            raise RuntimeError(f"ContinuumProfileStore profile type {profile_type} not implemented.")

    def build_profiles(
            self, layer_idx: int, pop_matrix: npt.NDArray[np.float64], wn_grid: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        abs_profile = self.abs_profiles[layer_idx].build_xsec(pop_matrix=pop_matrix, wn_grid=wn_grid, is_abs=True)
        return abs_profile


def calc_cont_band_profile(
        wn_grid: npt.NDArray[np.float64],
        temperature: float,
        species_mass: float,
        box_length: float,
        group: pl.DataFrame,
) -> pl.DataFrame:
    a_fi = np.ascontiguousarray(group["A_fi"].to_numpy())
    g_i = np.ascontiguousarray(group["g_i"].to_numpy())
    g_f = np.ascontiguousarray(group["g_f"].to_numpy())
    energy_fi = np.ascontiguousarray(group["energy_fi"].to_numpy())
    n_frac_i = np.ascontiguousarray(group["n_frac_i"].to_numpy())
    n_frac_f = np.ascontiguousarray(group["n_frac_f"].to_numpy())
    v_f = np.ascontiguousarray(group["v_f"].to_numpy())

    assert a_fi.shape[0] == g_i.shape[0] == g_f.shape[0] == energy_fi.shape[0] == n_frac_i.shape[0] == n_frac_f.shape[0]

    if energy_fi.min() > wn_grid.max() or energy_fi.max() < wn_grid.min():
        raise RuntimeError("Computing band profile for band with no transitions on grid: check workflow logic.")

    is_fixed_width = np.all(np.isclose(np.diff(wn_grid), np.abs(wn_grid[1] - wn_grid[0]), atol=0))

    if is_fixed_width:
        _abs_xsec = _continuum_binned_gauss_fixed_width(
            wn_grid=wn_grid,
            n_f=n_frac_f,
            n_i=n_frac_i,
            a_fi=a_fi,
            g_f=g_f,
            g_i=g_i,
            energy_fi=energy_fi,
            v_f=v_f,
            temperature=temperature,
            species_mass=species_mass,
            box_length=box_length,
            half_bin_width=np.abs(wn_grid[1] - wn_grid[0]) / 2.0,
        )
    else:
        _abs_xsec = _continuum_binned_gauss_variable_width(
            wn_grid=wn_grid,
            n_f=n_frac_f,
            n_i=n_frac_i,
            a_fi=a_fi,
            g_f=g_f,
            g_i=g_i,
            energy_fi=energy_fi,
            v_f=v_f,
            temperature=temperature,
            species_mass=species_mass,
            box_length=box_length,
        )
    return pl.DataFrame({
        "id_agg_i": [group["id_agg_i"][0]],
        "abs_profile": [_abs_xsec],
    })


def calc_band_profile(
        wn_grid: npt.NDArray[np.float64],
        temperature: float,
        pressure: float,
        species_mass: float,
        broad_n: npt.NDArray[np.float64],
        broad_gamma: npt.NDArray[np.float64],
        group: pl.DataFrame,
) -> pl.DataFrame:
    a_fi = np.ascontiguousarray(group["A_fi"].to_numpy())
    g_i = np.ascontiguousarray(group["g_i"].to_numpy())
    g_f = np.ascontiguousarray(group["g_f"].to_numpy())
    energy_fi = np.ascontiguousarray(group["energy_fi"].to_numpy())
    n_frac_i = np.ascontiguousarray(group["n_frac_i"].to_numpy())
    n_frac_f = np.ascontiguousarray(group["n_frac_f"].to_numpy())
    tau_f = np.ascontiguousarray(group["tau_f"].to_numpy())

    assert a_fi.shape[0] == g_i.shape[0] == g_f.shape[0] == energy_fi.shape[0] == n_frac_i.shape[0] == n_frac_f.shape[0]

    if energy_fi.min() > wn_grid.max() or energy_fi.max() < wn_grid.min():
        raise RuntimeError("Computing band profile for band with no transitions on grid: check workflow logic.")

    if (broad_n is None) ^ (broad_gamma is None):
        raise RuntimeError(f"Either broadening n or gamma missing: n={broad_n}, gamma={broad_gamma}.")

    is_fixed_width = np.all(np.isclose(np.diff(wn_grid), np.abs(wn_grid[1] - wn_grid[0]), atol=0))

    gh_roots, gh_weights = roots_hermite(_N_GH_QUAD_POINTS)

    if is_fixed_width:
        _abs_xsec, _ste_xsec, _spe_xsec = _band_profile_binned_voigt_fixed_width(
            wn_grid=wn_grid,
            n_i=n_frac_i,
            n_f=n_frac_f,
            a_fi=a_fi,
            g_f=g_f,
            g_i=g_i,
            energy_fi=energy_fi,
            lifetimes=tau_f,
            temperature=temperature,
            pressure=pressure,
            broad_n=broad_n,
            broad_gamma=broad_gamma,
            species_mass=species_mass,
            half_bin_width=np.abs(wn_grid[1] - wn_grid[0]) / 2.0,
            gh_roots=gh_roots,
            gh_weights=gh_weights,
        )
    else:
        _abs_xsec, _ste_xsec, _spe_xsec = _band_profile_binned_voigt_variable_width(
            wn_grid=wn_grid,
            n_i=n_frac_i,
            n_f=n_frac_f,
            a_fi=a_fi,
            g_f=g_f,
            g_i=g_i,
            energy_fi=energy_fi,
            lifetimes=tau_f,
            temperature=temperature,
            pressure=pressure,
            broad_n=broad_n,
            broad_gamma=broad_gamma,
            species_mass=species_mass,
            gh_roots=gh_roots,
            gh_weights=gh_weights,
        )
    return pl.DataFrame({
        "id_agg_f": [group["id_agg_f"][0]],
        "id_agg_i": [group["id_agg_i"][0]],
        "abs_profile": [_abs_xsec],
        "ste_profile": [_ste_xsec],
        "spe_profile": [_spe_xsec],
    })


def abs_emi_xsec(
        states: pl.DataFrame,
        trans_files: t.List[pathlib.Path],
        temperature: u.Quantity,
        pressure: u.Quantity,
        species_mass: float,
        wn_grid: npt.NDArray[np.float64],
        broad_n: npt.NDArray[np.float64] = None,
        broad_gamma: npt.NDArray[np.float64] = None,
        n_gh_quad_points: int = _N_GH_QUAD_POINTS,
):
    half_bin_width = abs(wn_grid[1] - wn_grid[0]) / 2.0
    # TODO: Check is_fixed_width rigour!
    is_fixed_width = np.all(np.isclose(np.diff(wn_grid), np.abs(wn_grid[1] - wn_grid[0]), atol=0))
    num_grid = wn_grid.shape[0]
    abs_xsec = np.zeros(num_grid, dtype=np.float64)
    emi_xsec = np.zeros(num_grid, dtype=np.float64)

    wn_min = wn_grid[0]
    wn_max = wn_grid[-1]

    pl_states_i = states.select(
        pl.col("id"),
        pl.col("energy").alias("energy_i"),
        pl.col("n_nlte").alias("n_nlte_i"),
        pl.col("g").alias("g_i"),
    )
    pl_states_f = states.select(
        pl.col("id"),
        pl.col("energy").alias("energy_f"),
        pl.col("n_nlte").alias("n_nlte_f"),
        pl.col("g").alias("g_f"),
        pl.col("tau").alias("tau_f"),
    )

    dask_dtypes = {"id_f": "int64", "id_i": "int64", "A_fi": "float64", }
    dask_blocksize = "256MB"

    for trans_file in trans_files:
        ddf = dd.read_csv(
            trans_file,
            sep=r"\s+",
            engine="python",
            header=None,
            names=["id_f", "id_i", "A_fi"],
            usecols=[0, 1, 2],
            dtype=dask_dtypes,
            blocksize=dask_blocksize,
        )
        delayed_batches = ddf.to_delayed()
        for delayed_batch in delayed_batches:
            trans_batch = pl.from_pandas(delayed_batch.compute())
            trans_batch = trans_batch.join(pl_states_i, left_on="id_i", right_on="id", how="left")
            trans_batch = trans_batch.join(pl_states_f, left_on="id_f", right_on="id", how="left")
            trans_batch = trans_batch.with_columns(
                (pl.col("energy_f") - pl.col("energy_i")).alias("energy_fi")
            )
            trans_batch = trans_batch.filter(
                (pl.col("energy_f") >= wn_min)
                & (pl.col("energy_i") <= wn_max)
                & (pl.col("energy_fi") >= wn_min)
                & (pl.col("energy_fi") <= wn_max)
            )
            final_cols = ["n_nlte_i", "n_nlte_f", "A_fi", "g_f", "g_i", "energy_fi", "tau_f"]
            trans_batch = trans_batch.select(final_cols)
            trans_chunk_np = trans_batch.to_numpy()

            # Matches final_cols ordering.
            n_i = np.ascontiguousarray(trans_chunk_np[:, 0])
            n_f = np.ascontiguousarray(trans_chunk_np[:, 1])
            a_fi = np.ascontiguousarray(trans_chunk_np[:, 2])
            g_f = np.ascontiguousarray(trans_chunk_np[:, 3])
            g_i = np.ascontiguousarray(trans_chunk_np[:, 4])
            energy_fi = np.ascontiguousarray(trans_chunk_np[:, 5])
            lifetimes = np.ascontiguousarray(trans_chunk_np[:, 6])

            if broad_n is None or broad_gamma is None:
                if is_fixed_width:
                    _abs_xsec, _emi_xsec = _abs_emi_binned_doppler_fixed_width(
                        wn_grid=wn_grid,
                        n_i=n_i,
                        n_f=n_f,
                        a_fi=a_fi,
                        g_f=g_f,
                        g_i=g_i,
                        energy_fi=energy_fi,
                        temperature=temperature.value,
                        species_mass=species_mass,
                        half_bin_width=half_bin_width,
                    )
                else:
                    _abs_xsec, _emi_xsec = _abs_emi_binned_doppler_variable_width(
                        wn_grid=wn_grid,
                        n_i=n_i,
                        n_f=n_f,
                        a_fi=a_fi,
                        g_f=g_f,
                        g_i=g_i,
                        energy_fi=energy_fi,
                        temperature=temperature.value,
                        species_mass=species_mass,
                    )
            else:
                gh_roots, gh_weights = roots_hermite(n_gh_quad_points)
                if is_fixed_width:
                    _abs_xsec, _emi_xsec = _abs_emi_binned_voigt_fixed_width(
                        wn_grid=wn_grid,
                        n_i=n_i,
                        n_f=n_f,
                        a_fi=a_fi,
                        g_f=g_f,
                        g_i=g_i,
                        energy_fi=energy_fi,
                        lifetimes=lifetimes,
                        temperature=temperature.value,
                        pressure=pressure.value,
                        broad_n=broad_n,
                        broad_gamma=broad_gamma,
                        species_mass=species_mass,
                        half_bin_width=half_bin_width,
                        gh_roots=gh_roots,
                        gh_weights=gh_weights,
                    )
                else:
                    _abs_xsec, _emi_xsec = _abs_emi_binned_voigt_variable_width(
                        wn_grid=wn_grid,
                        n_i=n_i,
                        n_f=n_f,
                        a_fi=a_fi,
                        g_f=g_f,
                        g_i=g_i,
                        energy_fi=energy_fi,
                        lifetimes=lifetimes,
                        temperature=temperature.value,
                        pressure=pressure.value,
                        broad_n=broad_n,
                        broad_gamma=broad_gamma,
                        species_mass=species_mass,
                        gh_roots=gh_roots,
                        gh_weights=gh_weights,
                    )
            abs_xsec += _abs_xsec
            emi_xsec += _emi_xsec
    return abs_xsec, emi_xsec


def continuum_xsec(
        continuum_states: pl.DataFrame,
        continuum_trans_files: t.List[pathlib.Path],
        layer_idx: int,
        wn_grid: npt.NDArray[np.float64],
        temperature: float,
        species_mass: float,
        cont_box_length: float,
) -> npt.NDArray[np.float64]:
    is_fixed_width = np.all(np.isclose(np.diff(wn_grid), np.abs(wn_grid[1] - wn_grid[0]), atol=0))
    cont_xsec = np.zeros(wn_grid.shape, dtype=np.float64)

    wn_min = wn_grid[0]
    wn_max = wn_grid[-1]

    n_col = f"n_nlte_L{layer_idx}"

    pl_states_i = continuum_states.select(
        pl.col("id"),
        pl.col("energy").alias("energy_i"),
        pl.col(n_col).alias("n_nlte_i"),
        pl.col("g").alias("g_i"),
    )
    pl_states_f = continuum_states.select(
        pl.col("id"),
        pl.col("energy").alias("energy_f"),
        pl.col(n_col).alias("n_nlte_f"),
        pl.col("g").alias("g_f"),
        pl.col("v").alias("v_f"),
    )

    dask_dtypes = {"id_f": "int64", "id_i": "int64", "A_fi": "float64"}
    dask_blocksize = "256MB"

    for trans_file in continuum_trans_files:
        ddf = dd.read_csv(
            trans_file,
            sep=r"\s+",
            engine="python",
            header=None,
            names=["id_f", "id_i", "A_fi"],
            usecols=[0, 1, 2],
            dtype=dask_dtypes,
            blocksize=dask_blocksize,
        )
        delayed_batches = ddf.to_delayed()
        for delayed_batch in delayed_batches:
            trans_batch = pl.from_pandas(delayed_batch.compute())
            trans_batch = trans_batch.join(pl_states_i, left_on="id_i", right_on="id", how="left")
            trans_batch = trans_batch.join(pl_states_f, left_on="id_f", right_on="id", how="left")
            trans_batch = trans_batch.with_columns(
                (pl.col("energy_f") - pl.col("energy_i")).alias("energy_fi")
            )
            trans_batch = trans_batch.filter(
                (pl.col("energy_f") >= wn_min)
                & (pl.col("energy_i") <= wn_max)
                & (pl.col("energy_fi") >= wn_min)
                & (pl.col("energy_fi") <= wn_max)
            )
            final_cols = ["n_nlte_f", "n_nlte_i", "A_fi", "g_f", "g_i", "energy_fi", "v_f"]
            trans_batch = trans_batch.select(final_cols)
            trans_chunk_np = trans_batch.to_numpy()

            # Matches final_cols ordering.
            n_f = np.ascontiguousarray(trans_chunk_np[:, 0])
            n_i = np.ascontiguousarray(trans_chunk_np[:, 1])
            a_fi = np.ascontiguousarray(trans_chunk_np[:, 2])
            g_f = np.ascontiguousarray(trans_chunk_np[:, 3])
            g_i = np.ascontiguousarray(trans_chunk_np[:, 4])
            energy_fi = np.ascontiguousarray(trans_chunk_np[:, 5])
            v_f = np.ascontiguousarray(trans_chunk_np[:, 6])

            if is_fixed_width:
                half_bin_width = abs(wn_grid[1] - wn_grid[0]) / 2.0
                cont_xsec += _continuum_binned_gauss_fixed_width(
                    wn_grid=wn_grid,
                    n_f=n_f,
                    n_i=n_i,
                    a_fi=a_fi,
                    g_f=g_f,
                    g_i=g_i,
                    energy_fi=energy_fi,
                    v_f=v_f,
                    temperature=temperature,
                    species_mass=species_mass,
                    box_length=cont_box_length,
                    half_bin_width=half_bin_width,
                )
            else:
                cont_xsec += _continuum_binned_gauss_variable_width(
                    wn_grid=wn_grid,
                    n_f=n_f,
                    n_i=n_i,
                    a_fi=a_fi,
                    g_f=g_f,
                    g_i=g_i,
                    energy_fi=energy_fi,
                    v_f=v_f,
                    temperature=temperature,
                    species_mass=species_mass,
                    box_length=cont_box_length,
                )
    return cont_xsec


@numba.njit(cache=True)
def binary_search_left(arr: npt.NDArray[np.float64], value: float, start: int = 0) -> int:
    """Fast binary search for left insertion point with optional start hint."""
    left, right = start, len(arr)
    while left < right:
        mid = (left + right) >> 1
        if arr[mid] < value:
            left = mid + 1
        else:
            right = mid
    return left


@numba.njit(cache=True)
def binary_search_right(arr: npt.NDArray[np.float64], value: float, start: int = 0) -> int:
    """Fast binary search for right insertion point with optional start hint."""
    left, right = start, len(arr)
    while left < right:
        mid = (left + right) >> 1
        if arr[mid] <= value:
            left = mid + 1
        else:
            right = mid
    return left


# BAND PROFILE CALCULATIONS

@numba.njit(parallel=True, cache=True, error_model="numpy")
def _band_profile_binned_doppler_variable_width(
        wn_grid: npt.NDArray[np.float64],
        n_i: npt.NDArray[np.float64],
        n_f: npt.NDArray[np.float64],
        a_fi: npt.NDArray[np.float64],
        g_f: npt.NDArray[np.float64],
        g_i: npt.NDArray[np.float64],
        energy_fi: npt.NDArray[np.float64],
        temperature: float,
        species_mass: float,
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    assert n_i.shape[0] == n_f.shape[0] == a_fi.shape[0] == g_f.shape[0] == g_i.shape[0] == energy_fi.shape[0]

    sqrtln2 = math.sqrt(math.log(2))
    num_grid = wn_grid.shape[0]
    _abs_xsec = np.zeros(num_grid, dtype=np.float64)
    _emi_xsec = np.zeros(num_grid, dtype=np.float64)
    num_trans = energy_fi.shape[0]
    cutoff = 25

    bin_widths = np.zeros(num_grid + 1, dtype=np.float64)
    bin_widths[1:-1] = (wn_grid[:-1] + wn_grid[1:]) / 2.0 - wn_grid[:-1]

    abs_coef = a_fi * (n_i * g_f / g_i) / (const_16_pi_c * energy_fi * energy_fi)
    emi_coef = n_f * a_fi * energy_fi * const_h_c_on_8_pi
    sqrtln2_on_alpha = sqrtln2 / (energy_fi * const_sqrt_2_NA_kB_log2_on_c * np.sqrt(temperature / species_mass))

    for i in numba.prange(num_trans):
        energy_fi_i = energy_fi[i]
        sqrtln2_on_alpha_i = sqrtln2_on_alpha[i]
        abs_coef_i = abs_coef[i]
        emi_coef_i = emi_coef[i]

        for j in range(num_grid):
            wn_shift = wn_grid[j] - energy_fi_i
            upper_width = bin_widths[j + 1]
            lower_width = bin_widths[j]
            if -upper_width - cutoff <= wn_shift <= lower_width + cutoff:
                bin_term = (
                                   math.erf(sqrtln2_on_alpha_i * (wn_shift + upper_width))
                                   - math.erf(sqrtln2_on_alpha_i * (wn_shift - lower_width))
                           ) / (upper_width + lower_width)
                _abs_xsec[j] += abs_coef_i * bin_term
                _emi_xsec[j] += emi_coef_i * bin_term
    return _abs_xsec, _emi_xsec


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _band_profile_binned_doppler_fixed_width(
        wn_grid: npt.NDArray[np.float64],
        n_i: npt.NDArray[np.float64],
        n_f: npt.NDArray[np.float64],
        a_fi: npt.NDArray[np.float64],
        g_f: npt.NDArray[np.float64],
        g_i: npt.NDArray[np.float64],
        energy_fi: npt.NDArray[np.float64],
        temperature: float,
        species_mass: float,
        half_bin_width: float,
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    assert n_i.shape[0] == n_f.shape[0] == a_fi.shape[0] == g_f.shape[0] == g_i.shape[0] == energy_fi.shape[0]

    twice_bin_width = 4.0 * half_bin_width
    sqrtln2 = math.sqrt(math.log(2))
    num_grid = wn_grid.shape[0]
    _abs_xsec = np.zeros(num_grid, dtype=np.float64)
    _emi_xsec = np.zeros(num_grid, dtype=np.float64)
    num_trans = energy_fi.shape[0]
    cutoff = 25 + half_bin_width

    abs_coef = a_fi * (n_i * g_f / g_i) / (const_8_pi_c * twice_bin_width * energy_fi * energy_fi)
    emi_coef = n_f * a_fi * energy_fi * const_h_c_on_4_pi / twice_bin_width
    sqrtln2_on_alpha = sqrtln2 / (energy_fi * const_sqrt_2_NA_kB_log2_on_c * np.sqrt(temperature / species_mass))

    for i in numba.prange(num_trans):
        energy_fi_i = energy_fi[i]
        sqrtln2_on_alpha_i = sqrtln2_on_alpha[i]
        abs_coef_i = abs_coef[i]
        emi_coef_i = emi_coef[i]

        for j in range(num_grid):
            wn_shift = wn_grid[j] - energy_fi_i
            if np.abs(wn_shift) <= cutoff:
                bin_term = math.erf(sqrtln2_on_alpha_i * (wn_shift + half_bin_width)) - math.erf(
                    sqrtln2_on_alpha_i * (wn_shift - half_bin_width)
                )
                _abs_xsec[j] += abs_coef_i * bin_term
                _emi_xsec[j] += emi_coef_i * bin_term
    return _abs_xsec, _emi_xsec


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _band_profile_binned_voigt_variable_width(
        wn_grid: npt.NDArray[np.float64],
        n_i: npt.NDArray[np.float64],
        n_f: npt.NDArray[np.float64],
        a_fi: npt.NDArray[np.float64],
        g_f: npt.NDArray[np.float64],
        g_i: npt.NDArray[np.float64],
        energy_fi: npt.NDArray[np.float64],
        lifetimes: npt.NDArray[np.float64],
        temperature: float,
        pressure: float,
        broad_n: npt.NDArray[np.float64],
        broad_gamma: npt.NDArray[np.float64],
        species_mass: float,
        gh_roots: npt.NDArray[np.float64],
        gh_weights: npt.NDArray[np.float64],
        t_ref: float = 296,
        pressure_ref: float = 1,
        numba_num_threads: int = _DEFAULT_NUM_THREADS,
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    assert n_i.shape[0] == n_f.shape[0] == a_fi.shape[0] == g_f.shape[0] == g_i.shape[0] == energy_fi.shape[0] == \
           lifetimes.shape[0]
    assert broad_n.shape[0] == broad_gamma.shape[0]
    assert gh_roots.shape[0] == gh_weights.shape[0]

    num_grid = wn_grid.shape[0]
    num_trans = energy_fi.shape[0]
    cutoff = 25

    bin_edges = np.empty(num_grid + 1, dtype=np.float64)
    bin_edges[0] = wn_grid[0] - (wn_grid[1] - wn_grid[0]) * 0.5
    for j in range(1, num_grid):
        bin_edges[j] = (wn_grid[j - 1] + wn_grid[j]) * 0.5
    bin_edges[-1] = wn_grid[-1] + (wn_grid[-1] - wn_grid[-2]) * 0.5

    bin_widths_lower = wn_grid - bin_edges[:-1]
    bin_widths_upper = bin_edges[1:] - wn_grid
    inv_bin_widths = 1.0 / (bin_widths_lower + bin_widths_upper)

    # Stimulated emission removed!
    abs_coef = a_fi * (n_i * g_f / g_i) / (const_8_pi_five_halves_c * energy_fi * energy_fi)
    ste_coef = a_fi * n_f / (const_8_pi_five_halves_c * energy_fi * energy_fi)
    spe_coef = n_f * a_fi * energy_fi * const_h_c_on_4_pi_five_halves
    sigma = energy_fi * const_sqrt_NA_kB_on_c * np.sqrt(temperature / species_mass)
    gamma_pressure = np.sum(broad_gamma * pressure * (t_ref / temperature) ** broad_n / pressure_ref)
    gamma_lifetime = 1 / (const_4_pi_c * lifetimes)
    gamma_total = gamma_lifetime + gamma_pressure
    gamma_inv = 1 / gamma_total

    grid_min = wn_grid[0]
    grid_max = wn_grid[-1]

    abs_xsec_buffer = np.zeros((numba_num_threads, num_grid), dtype=np.float64)
    ste_xsec_buffer = np.zeros((numba_num_threads, num_grid), dtype=np.float64)
    spe_xsec_buffer = np.zeros((numba_num_threads, num_grid), dtype=np.float64)

    for i in numba.prange(num_trans):
        thread_id = numba.get_thread_id()

        energy_fi_i = energy_fi[i]
        abs_coef_i = abs_coef[i]
        ste_coef_i = ste_coef[i]
        spe_coef_i = spe_coef[i]
        gamma_inv_i = gamma_inv[i]
        sigma_i = sigma[i]

        gh_roots_sigma = gh_roots * sigma_i
        start_sigma = grid_min - gh_roots_sigma
        end_sigma = grid_max - gh_roots_sigma
        b_corr = np.pi / (
                np.arctan((end_sigma - energy_fi_i) * gamma_inv_i)
                - np.arctan((start_sigma - energy_fi_i) * gamma_inv_i)
        )
        gh_weights_b_corr = gh_weights * b_corr

        transition_min = energy_fi_i - cutoff
        transition_max = energy_fi_i + cutoff

        j_start = binary_search_right(bin_edges, transition_min) - 1
        j_start = max(0, j_start)
        j_end = binary_search_left(bin_edges, transition_max, start=j_start)
        j_end = min(num_grid, j_end)

        for j in range(j_start, j_end):
            upper_width_j = bin_widths_upper[j]
            lower_width_j = bin_widths_lower[j]
            inv_bin_width_j = inv_bin_widths[j]
            wn_j = wn_grid[j]

            shift_sigma = wn_j - energy_fi_i - gh_roots_sigma
            bin_term = np.sum(
                gh_weights_b_corr
                * (
                        np.arctan((shift_sigma + upper_width_j) * gamma_inv_i)
                        - np.arctan((shift_sigma - lower_width_j) * gamma_inv_i)
                )
            ) * inv_bin_width_j

            abs_xsec_buffer[thread_id, j] += abs_coef_i * bin_term
            ste_xsec_buffer[thread_id, j] += ste_coef_i * bin_term
            spe_xsec_buffer[thread_id, j] += spe_coef_i * bin_term
    # Accumulate buffers.
    _abs_xsec = np.zeros(num_grid, dtype=np.float64)
    _ste_xsec = np.zeros(num_grid, dtype=np.float64)
    _spe_xsec = np.zeros(num_grid, dtype=np.float64)
    for k in numba.prange(num_grid):
        abs_xsec_point = 0.0
        ste_xsec_point = 0.0
        spe_xsec_point = 0.0
        for t in range(numba_num_threads):
            abs_xsec_point += abs_xsec_buffer[t, k]
            ste_xsec_point += ste_xsec_buffer[t, k]
            spe_xsec_point += spe_xsec_buffer[t, k]
        _abs_xsec[k] = abs_xsec_point
        _ste_xsec[k] = ste_xsec_point
        _spe_xsec[k] = spe_xsec_point

    return _abs_xsec, _ste_xsec, _spe_xsec


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _band_profile_binned_voigt_fixed_width(
        wn_grid: npt.NDArray[np.float64],
        n_i: npt.NDArray[np.float64],
        n_f: npt.NDArray[np.float64],
        a_fi: npt.NDArray[np.float64],
        g_f: npt.NDArray[np.float64],
        g_i: npt.NDArray[np.float64],
        energy_fi: npt.NDArray[np.float64],
        lifetimes: npt.NDArray[np.float64],
        temperature: float,
        pressure: float,
        broad_n: npt.NDArray[np.float64],
        broad_gamma: npt.NDArray[np.float64],
        species_mass: float,
        half_bin_width: float,
        gh_roots: npt.NDArray[np.float64],
        gh_weights: npt.NDArray[np.float64],
        t_ref: float = 296,
        pressure_ref: float = 1,
        numba_num_threads: int = _DEFAULT_NUM_THREADS,
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    assert n_i.shape[0] == n_f.shape[0] == a_fi.shape[0] == g_f.shape[0] == g_i.shape[0] == energy_fi.shape[0] == \
           lifetimes.shape[0]
    assert broad_n.shape[0] == broad_gamma.shape[0]
    assert gh_roots.shape[0] == gh_weights.shape[0]

    bin_width = 2.0 * half_bin_width
    num_grid = wn_grid.shape[0]
    num_trans = energy_fi.shape[0]
    cutoff = 25 + half_bin_width

    bin_edges = np.empty(num_grid + 1, dtype=np.float64)
    bin_edges[0] = wn_grid[0] - (wn_grid[1] - wn_grid[0]) * 0.5
    for j in range(1, num_grid):
        bin_edges[j] = (wn_grid[j - 1] + wn_grid[j]) * 0.5
    bin_edges[-1] = wn_grid[-1] + (wn_grid[-1] - wn_grid[-2]) * 0.5

    abs_coef = a_fi * (n_i * g_f / g_i) / (const_8_pi_five_halves_c * bin_width * energy_fi * energy_fi)
    ste_coef = a_fi * n_f / (const_8_pi_five_halves_c * bin_width * energy_fi * energy_fi)
    spe_coef = n_f * a_fi * energy_fi * const_h_c_on_4_pi_five_halves / bin_width

    sigma = energy_fi * const_sqrt_NA_kB_on_c * np.sqrt(temperature / species_mass)
    gamma_pressure = np.sum(broad_gamma * pressure * (t_ref / temperature) ** broad_n / pressure_ref)
    gamma_lifetime = 1 / (const_4_pi_c * lifetimes)
    gamma_total = gamma_lifetime + gamma_pressure
    gamma_inv = 1 / gamma_total

    grid_min = wn_grid[0]
    grid_max = wn_grid[-1]

    abs_xsec_buffer = np.zeros((numba_num_threads, num_grid), dtype=np.float64)
    ste_xsec_buffer = np.zeros((numba_num_threads, num_grid), dtype=np.float64)
    spe_xsec_buffer = np.zeros((numba_num_threads, num_grid), dtype=np.float64)

    for i in numba.prange(num_trans):
        thread_id = numba.get_thread_id()

        energy_fi_i = energy_fi[i]
        abs_coef_i = abs_coef[i]
        ste_coef_i = ste_coef[i]
        spe_coef_i = spe_coef[i]
        gamma_inv_i = gamma_inv[i]
        sigma_i = sigma[i]

        gh_roots_sigma = gh_roots * sigma_i
        start_sigma = grid_min - gh_roots_sigma
        end_sigma = grid_max - gh_roots_sigma
        b_corr = np.pi / (
                np.arctan((end_sigma - energy_fi_i) * gamma_inv_i)
                - np.arctan((start_sigma - energy_fi_i) * gamma_inv_i)
        )
        gh_weights_b_corr = gh_weights * b_corr

        transition_min = energy_fi_i - cutoff
        transition_max = energy_fi_i + cutoff

        j_start = binary_search_right(bin_edges, transition_min) - 1
        j_start = max(0, j_start)
        j_end = binary_search_left(bin_edges, transition_max, start=j_start)
        j_end = min(num_grid, j_end)

        for j in range(j_start, j_end):
            wn_j = wn_grid[j]

            shift_sigma = wn_grid[j] - energy_fi_i - gh_roots_sigma
            bin_term = np.sum(
                gh_weights_b_corr
                * (
                        np.arctan((shift_sigma + half_bin_width) * gamma_inv_i)
                        - np.arctan((shift_sigma - half_bin_width) * gamma_inv_i)
                )
            )
            abs_xsec_buffer[thread_id, j] += abs_coef_i * bin_term
            ste_xsec_buffer[thread_id, j] += ste_coef_i * bin_term
            spe_xsec_buffer[thread_id, j] += spe_coef_i * bin_term
    # Accumulate buffers.
    _abs_xsec = np.zeros(num_grid, dtype=np.float64)
    _ste_xsec = np.zeros(num_grid, dtype=np.float64)
    _spe_xsec = np.zeros(num_grid, dtype=np.float64)
    for k in numba.prange(num_grid):
        abs_xsec_point = 0.0
        ste_xsec_point = 0.0
        spe_xsec_point = 0.0
        for t in range(numba_num_threads):
            abs_xsec_point += abs_xsec_buffer[t, k]
            ste_xsec_point += ste_xsec_buffer[t, k]
            spe_xsec_point += spe_xsec_buffer[t, k]
        _abs_xsec[k] = abs_xsec_point
        _ste_xsec[k] = ste_xsec_point
        _spe_xsec[k] = spe_xsec_point
    return _abs_xsec, _ste_xsec, _spe_xsec


# ABSORPTION/EMISSION CROSS SECTION CALCULATIONS:
@numba.njit(parallel=True, cache=True, error_model="numpy")
def _abs_emi_binned_doppler_variable_width(
        wn_grid: npt.NDArray[np.float64],
        n_i: npt.NDArray[np.float64],
        n_f: npt.NDArray[np.float64],
        a_fi: npt.NDArray[np.float64],
        g_f: npt.NDArray[np.float64],
        g_i: npt.NDArray[np.float64],
        energy_fi: npt.NDArray[np.float64],
        temperature: float,
        species_mass: float,
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    assert n_i.shape[0] == n_f.shape[0] == a_fi.shape[0] == g_f.shape[0] == g_i.shape[0] == energy_fi.shape[0]

    sqrtln2 = math.sqrt(math.log(2))
    num_grid = wn_grid.shape[0]
    _abs_xsec = np.zeros(num_grid, dtype=np.float64)
    _emi_xsec = np.zeros(num_grid, dtype=np.float64)
    num_trans = energy_fi.shape[0]
    cutoff = 25

    bin_widths = np.zeros(num_grid + 1, dtype=np.float64)
    bin_widths[1:-1] = (wn_grid[:-1] + wn_grid[1:]) / 2.0 - wn_grid[:-1]

    abs_coef = a_fi * ((n_i * g_f / g_i) - n_f) / (const_16_pi_c * energy_fi * energy_fi)
    emi_coef = n_f * a_fi * energy_fi * const_h_c_on_8_pi
    sqrtln2_on_alpha = sqrtln2 / (energy_fi * const_sqrt_2_NA_kB_log2_on_c * np.sqrt(temperature / species_mass))

    for i in numba.prange(num_trans):
        energy_fi_i = energy_fi[i]
        sqrtln2_on_alpha_i = sqrtln2_on_alpha[i]
        abs_coef_i = abs_coef[i]
        emi_coef_i = emi_coef[i]

        for j in range(num_grid):
            wn_shift = wn_grid[j] - energy_fi_i
            upper_width = bin_widths[j + 1]
            lower_width = bin_widths[j]
            if -upper_width - cutoff <= wn_shift <= lower_width + cutoff:
                bin_term = (
                                   math.erf(sqrtln2_on_alpha_i * (wn_shift + upper_width))
                                   - math.erf(sqrtln2_on_alpha_i * (wn_shift - lower_width))
                           ) / (upper_width + lower_width)
                _abs_xsec[j] += abs_coef_i * bin_term
                _emi_xsec[j] += emi_coef_i * bin_term
    return _abs_xsec, _emi_xsec


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _abs_emi_binned_doppler_fixed_width(
        wn_grid: npt.NDArray[np.float64],
        n_i: npt.NDArray[np.float64],
        n_f: npt.NDArray[np.float64],
        a_fi: npt.NDArray[np.float64],
        g_f: npt.NDArray[np.float64],
        g_i: npt.NDArray[np.float64],
        energy_fi: npt.NDArray[np.float64],
        temperature: float,
        species_mass: float,
        half_bin_width: float,
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    assert n_i.shape[0] == n_f.shape[0] == a_fi.shape[0] == g_f.shape[0] == g_i.shape[0] == energy_fi.shape[0]

    twice_bin_width = 4.0 * half_bin_width
    sqrtln2 = math.sqrt(math.log(2))
    num_grid = wn_grid.shape[0]
    _abs_xsec = np.zeros(num_grid, dtype=np.float64)
    _emi_xsec = np.zeros(num_grid, dtype=np.float64)
    num_trans = energy_fi.shape[0]
    cutoff = 25 + half_bin_width

    abs_coef = a_fi * ((n_i * g_f / g_i) - n_f) / (const_8_pi_c * twice_bin_width * energy_fi * energy_fi)
    emi_coef = n_f * a_fi * energy_fi * const_h_c_on_4_pi / twice_bin_width
    sqrtln2_on_alpha = sqrtln2 / (energy_fi * const_sqrt_2_NA_kB_log2_on_c * np.sqrt(temperature / species_mass))

    for i in numba.prange(num_trans):
        energy_fi_i = energy_fi[i]
        sqrtln2_on_alpha_i = sqrtln2_on_alpha[i]
        abs_coef_i = abs_coef[i]
        emi_coef_i = emi_coef[i]

        for j in range(num_grid):
            wn_shift = wn_grid[j] - energy_fi_i
            if np.abs(wn_shift) <= cutoff:
                bin_term = math.erf(sqrtln2_on_alpha_i * (wn_shift + half_bin_width)) - math.erf(
                    sqrtln2_on_alpha_i * (wn_shift - half_bin_width)
                )
                _abs_xsec[j] += abs_coef_i * bin_term
                _emi_xsec[j] += emi_coef_i * bin_term
    return _abs_xsec, _emi_xsec


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _abs_emi_binned_voigt_variable_width(
        wn_grid: npt.NDArray[np.float64],
        n_i: npt.NDArray[np.float64],
        n_f: npt.NDArray[np.float64],
        a_fi: npt.NDArray[np.float64],
        g_f: npt.NDArray[np.float64],
        g_i: npt.NDArray[np.float64],
        energy_fi: npt.NDArray[np.float64],
        lifetimes: npt.NDArray[np.float64],
        temperature: float,
        pressure: float,
        broad_n: npt.NDArray[np.float64],
        broad_gamma: npt.NDArray[np.float64],
        species_mass: float,
        gh_roots: npt.NDArray[np.float64],
        gh_weights: npt.NDArray[np.float64],
        t_ref: float = 296,
        pressure_ref: float = 1,
        numba_num_threads: int = _DEFAULT_NUM_THREADS,
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    assert n_i.shape[0] == n_f.shape[0] == a_fi.shape[0] == g_f.shape[0] == g_i.shape[0] == energy_fi.shape[0] == \
           lifetimes.shape[0]
    assert broad_n.shape[0] == broad_gamma.shape[0]
    assert gh_roots.shape[0] == gh_weights.shape[0]

    num_grid = wn_grid.shape[0]
    num_trans = energy_fi.shape[0]
    cutoff = 25

    bin_edges = np.empty(num_grid + 1, dtype=np.float64)
    bin_edges[0] = wn_grid[0] - (wn_grid[1] - wn_grid[0]) * 0.5
    for j in range(1, num_grid):
        bin_edges[j] = (wn_grid[j - 1] + wn_grid[j]) * 0.5
    bin_edges[-1] = wn_grid[-1] + (wn_grid[-1] - wn_grid[-2]) * 0.5

    bin_widths_lower = wn_grid - bin_edges[:-1]
    bin_widths_upper = bin_edges[1:] - wn_grid
    inv_bin_widths = 1.0 / (bin_widths_lower + bin_widths_upper)

    abs_coef = a_fi * ((n_i * g_f / g_i) - n_f) / (const_8_pi_five_halves_c * energy_fi * energy_fi)
    emi_coef = n_f * a_fi * energy_fi * const_h_c_on_4_pi_five_halves
    sigma = energy_fi * const_sqrt_NA_kB_on_c * np.sqrt(temperature / species_mass)
    gamma_pressure = np.sum(broad_gamma * pressure * (t_ref / temperature) ** broad_n / pressure_ref)
    gamma_lifetime = 1 / (const_4_pi_c * lifetimes)
    gamma_total = gamma_lifetime + gamma_pressure
    gamma_inv = 1 / gamma_total

    grid_min = wn_grid[0]
    grid_max = wn_grid[-1]

    abs_xsec_buffer = np.zeros((numba_num_threads, num_grid), dtype=np.float64)
    emi_xsec_buffer = np.zeros((numba_num_threads, num_grid), dtype=np.float64)

    for i in numba.prange(num_trans):
        thread_id = numba.get_thread_id()

        energy_fi_i = energy_fi[i]
        abs_coef_i = abs_coef[i]
        emi_coef_i = emi_coef[i]
        gamma_inv_i = gamma_inv[i]
        sigma_i = sigma[i]

        gh_roots_sigma = gh_roots * sigma_i
        start_sigma = grid_min - gh_roots_sigma
        end_sigma = grid_max - gh_roots_sigma
        b_corr = np.pi / (
                np.arctan((end_sigma - energy_fi_i) * gamma_inv_i)
                - np.arctan((start_sigma - energy_fi_i) * gamma_inv_i)
        )
        gh_weights_b_corr = gh_weights * b_corr

        transition_min = energy_fi_i - cutoff
        transition_max = energy_fi_i + cutoff

        j_start = binary_search_right(bin_edges, transition_min) - 1
        j_start = max(0, j_start)
        j_end = binary_search_left(bin_edges, transition_max, start=j_start)
        j_end = min(num_grid, j_end)

        for j in range(j_start, j_end):
            upper_width_j = bin_widths_upper[j]
            lower_width_j = bin_widths_lower[j]
            inv_bin_width_j = inv_bin_widths[j]
            wn_j = wn_grid[j]

            shift_sigma = wn_j - energy_fi_i - gh_roots_sigma
            bin_term = np.sum(
                gh_weights_b_corr
                * (
                        np.arctan((shift_sigma + upper_width_j) * gamma_inv_i)
                        - np.arctan((shift_sigma - lower_width_j) * gamma_inv_i)
                )
            ) * inv_bin_width_j

            abs_xsec_buffer[thread_id, j] += abs_coef_i * bin_term
            emi_xsec_buffer[thread_id, j] += emi_coef_i * bin_term
    # Accumulate buffers.
    _abs_xsec = np.zeros(num_grid, dtype=np.float64)
    _emi_xsec = np.zeros(num_grid, dtype=np.float64)
    for k in numba.prange(num_grid):
        abs_xsec_point = 0.0
        emi_xsec_point = 0.0
        for t in range(numba_num_threads):
            abs_xsec_point += abs_xsec_buffer[t, k]
            emi_xsec_point += emi_xsec_buffer[t, k]
        _abs_xsec[k] = abs_xsec_point
        _emi_xsec[k] = emi_xsec_point

    return _abs_xsec, _emi_xsec


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _abs_emi_binned_voigt_fixed_width(
        wn_grid: npt.NDArray[np.float64],
        n_i: npt.NDArray[np.float64],
        n_f: npt.NDArray[np.float64],
        a_fi: npt.NDArray[np.float64],
        g_f: npt.NDArray[np.float64],
        g_i: npt.NDArray[np.float64],
        energy_fi: npt.NDArray[np.float64],
        lifetimes: npt.NDArray[np.float64],
        temperature: float,
        pressure: float,
        broad_n: npt.NDArray[np.float64],
        broad_gamma: npt.NDArray[np.float64],
        species_mass: float,
        half_bin_width: float,
        gh_roots: npt.NDArray[np.float64],
        gh_weights: npt.NDArray[np.float64],
        t_ref: float = 296,
        pressure_ref: float = 1,
        numba_num_threads: int = _DEFAULT_NUM_THREADS,
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    assert n_i.shape[0] == n_f.shape[0] == a_fi.shape[0] == g_f.shape[0] == g_i.shape[0] == energy_fi.shape[0] == \
           lifetimes.shape[0]
    assert broad_n.shape[0] == broad_gamma.shape[0]
    assert gh_roots.shape[0] == gh_weights.shape[0]

    bin_width = 2.0 * half_bin_width
    num_grid = wn_grid.shape[0]
    num_trans = energy_fi.shape[0]
    cutoff = 25 + half_bin_width

    bin_edges = np.empty(num_grid + 1, dtype=np.float64)
    bin_edges[0] = wn_grid[0] - (wn_grid[1] - wn_grid[0]) * 0.5
    for j in range(1, num_grid):
        bin_edges[j] = (wn_grid[j - 1] + wn_grid[j]) * 0.5
    bin_edges[-1] = wn_grid[-1] + (wn_grid[-1] - wn_grid[-2]) * 0.5

    abs_coef = a_fi * ((n_i * g_f / g_i) - n_f) / (const_8_pi_five_halves_c * bin_width * energy_fi * energy_fi)
    emi_coef = n_f * a_fi * energy_fi * const_h_c_on_4_pi_five_halves / bin_width
    sigma = energy_fi * const_sqrt_NA_kB_on_c * np.sqrt(temperature / species_mass)
    gamma_pressure = np.sum(broad_gamma * pressure * (t_ref / temperature) ** broad_n / pressure_ref)
    gamma_lifetime = 1 / (const_4_pi_c * lifetimes)
    gamma_total = gamma_lifetime + gamma_pressure
    gamma_inv = 1 / gamma_total

    grid_min = wn_grid[0]
    grid_max = wn_grid[-1]

    abs_xsec_buffer = np.zeros((numba_num_threads, num_grid), dtype=np.float64)
    emi_xsec_buffer = np.zeros((numba_num_threads, num_grid), dtype=np.float64)

    for i in numba.prange(num_trans):
        thread_id = numba.get_thread_id()

        energy_fi_i = energy_fi[i]
        abs_coef_i = abs_coef[i]
        emi_coef_i = emi_coef[i]
        gamma_inv_i = gamma_inv[i]
        sigma_i = sigma[i]

        gh_roots_sigma = gh_roots * sigma_i
        start_sigma = grid_min - gh_roots_sigma
        end_sigma = grid_max - gh_roots_sigma
        b_corr = np.pi / (
                np.arctan((end_sigma - energy_fi_i) * gamma_inv_i)
                - np.arctan((start_sigma - energy_fi_i) * gamma_inv_i)
        )
        gh_weights_b_corr = gh_weights * b_corr

        transition_min = energy_fi_i - cutoff
        transition_max = energy_fi_i + cutoff

        j_start = binary_search_right(bin_edges, transition_min) - 1
        j_start = max(0, j_start)
        j_end = binary_search_left(bin_edges, transition_max, start=j_start)
        j_end = min(num_grid, j_end)

        for j in range(j_start, j_end):
            wn_j = wn_grid[j]

            shift_sigma = wn_j - energy_fi_i - gh_roots_sigma
            bin_term = np.sum(
                gh_weights_b_corr
                * (
                        np.arctan((shift_sigma + half_bin_width) * gamma_inv_i)
                        - np.arctan((shift_sigma - half_bin_width) * gamma_inv_i)
                )
            )

            abs_xsec_buffer[thread_id, j] += abs_coef_i * bin_term
            emi_xsec_buffer[thread_id, j] += emi_coef_i * bin_term
    # Accumulate buffers.
    _abs_xsec = np.zeros(num_grid, dtype=np.float64)
    _emi_xsec = np.zeros(num_grid, dtype=np.float64)
    for k in numba.prange(num_grid):
        abs_xsec_point = 0.0
        emi_xsec_point = 0.0
        for t in range(numba_num_threads):
            abs_xsec_point += abs_xsec_buffer[t, k]
            emi_xsec_point += emi_xsec_buffer[t, k]
        _abs_xsec[k] = abs_xsec_point
        _emi_xsec[k] = emi_xsec_point
    return _abs_xsec, _emi_xsec


# CONTINUUM CROSS SECTIONS:
@numba.njit(parallel=True, cache=True, error_model="numpy")
def _continuum_binned_gauss_variable_width(
        wn_grid: npt.NDArray[np.float64],
        n_f: npt.NDArray[np.float64],
        n_i: npt.NDArray[np.float64],
        a_fi: npt.NDArray[np.float64],
        g_f: npt.NDArray[np.float64],
        g_i: npt.NDArray[np.float64],
        energy_fi: npt.NDArray[np.float64],
        v_f: npt.NDArray[np.float64],
        temperature: float,
        species_mass: float,
        box_length: float,
        erf_lut: npt.NDArray[np.float64] = global_erf_lut,
        erf_arg_max: float = global_erf_arg_max,
        numba_num_threads: int = _DEFAULT_NUM_THREADS,
) -> npt.NDArray[np.float64]:
    """
    Note that the area of a Gaussian within 5 * HWHM is equal to erf(5*sqrt(ln(2))). The reciprocal of this can be used
    to recover the total intensity, though this value is 99.99999960685046% percent. Hence, this should only affect the
    9th decimal place, and Einstein A coefficients are only provided to 5 significant figures.

    Parameters
    ----------
    wn_grid
    n_f
    n_i
    a_fi
    g_f
    g_i
    energy_fi
    v_f
    temperature
    species_mass
    box_length
    erf_lut
    erf_arg_max
    numba_num_threads

    Returns
    -------

    """
    assert n_i.shape[0] == n_f.shape[0] == a_fi.shape[0] == g_f.shape[0] == g_i.shape[0] == energy_fi.shape[0] == \
           v_f.shape[0]

    sqrtln2 = math.sqrt(math.log(2))
    num_grid = wn_grid.shape[0]
    num_trans = energy_fi.shape[0]

    min_cutoff = 25.0
    max_cutoff = 3000.0
    cutoff_fwhm_multiple = 5.0

    # bin_widths = np.zeros(num_grid + 1, dtype=np.float64)
    # bin_widths[1:-1] = (wn_grid[:-1] + wn_grid[1:]) / 2.0 - wn_grid[:-1]
    bin_edges = np.empty(num_grid + 1, dtype=np.float64)
    bin_edges[0] = wn_grid[0] - (wn_grid[1] - wn_grid[0]) * 0.5
    for j in range(1, num_grid):
        bin_edges[j] = (wn_grid[j - 1] + wn_grid[j]) * 0.5
    bin_edges[-1] = wn_grid[-1] + (wn_grid[-1] - wn_grid[-2]) * 0.5

    bin_widths_lower = wn_grid - bin_edges[:-1]
    bin_widths_upper = bin_edges[1:] - wn_grid
    inv_bin_widths = 1.0 / (bin_widths_lower + bin_widths_upper)

    abs_coef = a_fi * ((n_i * g_f / g_i) - n_f) / (const_16_pi_c * energy_fi * energy_fi)

    alpha_doppler = energy_fi * const_sqrt_2_NA_kB_log2_on_c * np.sqrt(temperature / species_mass)
    alpha_box = const_h_on_8_c * (2 * v_f + 1) / (species_mass * const_amu * box_length * box_length)
    alpha_total = alpha_box + alpha_doppler
    sqrtln2_on_alpha = sqrtln2 / alpha_total
    cutoff = np.clip(alpha_total * cutoff_fwhm_multiple, a_min=min_cutoff, a_max=max_cutoff)

    xsec_buffer = np.zeros((numba_num_threads, num_grid), dtype=np.float64)

    for i in numba.prange(num_trans):
        thread_id = numba.get_thread_id()

        energy_fi_i = energy_fi[i]
        abs_coef_i = abs_coef[i]
        sqrtln2_on_alpha_i = sqrtln2_on_alpha[i]
        cutoff_i = cutoff[i]

        transition_min = energy_fi_i - cutoff_i
        transition_max = energy_fi_i + cutoff_i

        j_start = binary_search_right(bin_edges, transition_min) - 1
        j_start = max(0, j_start)
        j_end = binary_search_left(bin_edges, transition_max, start=j_start)
        j_end = min(num_grid, j_end)

        for j in range(j_start, j_end):
            wn_j = wn_grid[j]
            upper_width_j = bin_widths_upper[j]
            lower_width_j = bin_widths_lower[j]
            inv_bin_width_j = inv_bin_widths[j]

            wn_shift = wn_j - energy_fi_i

            erf_plus_arg = sqrtln2_on_alpha_i * (wn_shift + upper_width_j)
            erf_minus_arg = sqrtln2_on_alpha_i * (wn_shift - lower_width_j)

            erf_plus = get_erf_lut(erf_plus_arg, erf_lut, erf_arg_max)
            erf_minus = get_erf_lut(erf_minus_arg, erf_lut, erf_arg_max)

            xsec_buffer[thread_id, j] += abs_coef_i * (erf_plus - erf_minus) * inv_bin_width_j
    # Accumulate buffers.
    _abs_xsec = np.zeros(num_grid, dtype=np.float64)
    for k in numba.prange(num_grid):
        xsec_point = 0.0
        for t in range(numba_num_threads):
            xsec_point += xsec_buffer[t, k]
        _abs_xsec[k] = xsec_point
    return _abs_xsec


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _continuum_binned_gauss_fixed_width(
        wn_grid: npt.NDArray[np.float64],
        n_f: npt.NDArray[np.float64],
        n_i: npt.NDArray[np.float64],
        a_fi: npt.NDArray[np.float64],
        g_f: npt.NDArray[np.float64],
        g_i: npt.NDArray[np.float64],
        energy_fi: npt.NDArray[np.float64],
        v_f: npt.NDArray[np.float64],
        temperature: float,
        species_mass: float,
        box_length: float,
        half_bin_width: np.float64,
        erf_lut: npt.NDArray[np.float64] = global_erf_lut,
        erf_arg_max: float = global_erf_arg_max,
        numba_num_threads: int = _DEFAULT_NUM_THREADS,
) -> npt.NDArray[np.float64]:
    assert n_i.shape[0] == n_f.shape[0] == a_fi.shape[0] == g_f.shape[0] == g_i.shape[0] == energy_fi.shape[0] == \
           v_f.shape[0]

    twice_bin_width = 4.0 * half_bin_width
    sqrtln2 = math.sqrt(math.log(2))

    num_grid = wn_grid.shape[0]
    num_trans = energy_fi.shape[0]

    min_cutoff = 25.0
    max_cutoff = 3000.0
    cutoff_fwhm_multiple = 5.0

    bin_edges = np.empty(num_grid + 1, dtype=np.float64)
    bin_edges[0] = wn_grid[0] - (wn_grid[1] - wn_grid[0]) * 0.5
    for j in range(1, num_grid):
        bin_edges[j] = (wn_grid[j - 1] + wn_grid[j]) * 0.5
    bin_edges[-1] = wn_grid[-1] + (wn_grid[-1] - wn_grid[-2]) * 0.5

    abs_coef = a_fi * ((n_i * g_f / g_i) - n_f) / (const_8_pi_c * twice_bin_width * energy_fi * energy_fi)

    alpha_doppler = energy_fi * const_sqrt_2_NA_kB_log2_on_c * np.sqrt(temperature / species_mass)
    alpha_box = const_h_on_8_c * (2 * v_f + 1) / (species_mass * const_amu * box_length * box_length)
    alpha_total = alpha_box + alpha_doppler
    sqrtln2_on_alpha = sqrtln2 / alpha_total
    cutoff = np.clip(alpha_total * cutoff_fwhm_multiple, a_min=min_cutoff, a_max=max_cutoff)

    xsec_buffer = np.zeros((numba_num_threads, num_grid), dtype=np.float64)

    for i in numba.prange(num_trans):
        thread_id = numba.get_thread_id()

        energy_fi_i = energy_fi[i]
        abs_coef_i = abs_coef[i]
        sqrtln2_on_alpha_i = sqrtln2_on_alpha[i]
        cutoff_i = cutoff[i]

        transition_min = energy_fi_i - cutoff_i
        transition_max = energy_fi_i + cutoff_i

        j_start = binary_search_right(bin_edges, transition_min) - 1
        j_start = max(0, j_start)
        j_end = binary_search_left(bin_edges, transition_max, start=j_start)
        j_end = min(num_grid, j_end)

        for j in range(j_start, j_end):
            wn_j = wn_grid[j]

            wn_shift = wn_j - energy_fi_i

            erf_plus_arg = sqrtln2_on_alpha_i * (wn_shift + half_bin_width)
            erf_minus_arg = sqrtln2_on_alpha_i * (wn_shift - half_bin_width)

            erf_plus = get_erf_lut(erf_plus_arg, erf_lut, erf_arg_max)
            erf_minus = get_erf_lut(erf_minus_arg, erf_lut, erf_arg_max)

            xsec_buffer[thread_id, j] += abs_coef_i * (erf_plus - erf_minus)
    # Accumulate buffers.
    _abs_xsec = np.zeros(num_grid, dtype=np.float64)
    for k in numba.prange(num_grid):
        xsec_point = 0.0
        for t in range(numba_num_threads):
            xsec_point += xsec_buffer[t, k]
        _abs_xsec[k] = xsec_point
    return _abs_xsec


def effective_source_tau_mu(
        global_source_func_matrix: u.Quantity,
        global_chi_matrix: u.Quantity,
        global_eta_matrix: u.Quantity,
        density_profile: u.Quantity,
        dz_profile: u.Quantity,
        mu_values: npt.NDArray[np.float64],
        negative_absorption_factor: float = 0.1,
) -> t.Tuple[u.Quantity, npt.NDArray[np.float64]]:
    """
    Compute an effective Source function and optical depth, tau, for use in calculation of Bezier interpolants. These
    are computed based on an effective opacity calculated using Eq. (39) from https://doi.org/10.48550/arXiv.2508.12873.

    Parameters
    ----------
    global_source_func_matrix
    global_chi_matrix
    global_eta_matrix
    density_profile
    dz_profile
    mu_values
    negative_absorption_factor : float
        Factor used to calculate the positive upper bound on the effective opacity to use in cases where the opacity is
        negative.

    Returns
    -------
    """
    # Set effective Chi.
    chi_prime: u.Quantity = negative_absorption_factor * np.max(global_chi_matrix, axis=1)

    effective_chi = global_chi_matrix.copy()
    row_mask, col_mask = np.nonzero(global_chi_matrix < 0)
    effective_chi[row_mask, col_mask] = chi_prime[row_mask] * np.exp(
        -np.abs(global_chi_matrix[row_mask, col_mask].value)
    )

    effective_source_func_matrix = global_source_func_matrix.copy()
    neg_mask = effective_source_func_matrix < 0
    # Update negative source functions.
    # Zero entries where effective_chi is 0 - avoid division by 0.
    zero_chi_mask = neg_mask & (effective_chi == 0)
    effective_source_func_matrix[zero_chi_mask] = 0 * global_source_func_matrix.unit
    # Set entries with non-zero effective_chi.
    pos_chi_mask = neg_mask & ~zero_chi_mask
    effective_source_func_matrix[pos_chi_mask] = (
            global_eta_matrix[pos_chi_mask] / (ac.c.cgs * effective_chi[pos_chi_mask])
    ).to(global_source_func_matrix.unit, equivalencies=u.spectral())

    # Compute optical depths using effective chi.
    res = effective_chi * density_profile[:, None] * dz_profile[:, None]
    dtau = res.decompose().value
    tau = dtau[::-1].cumsum(axis=0)[::-1]
    tau_mu = tau[:, None, :] / mu_values[None, :, None]
    return effective_source_func_matrix, tau_mu


############ NEW
@numba.njit(parallel=True, cache=True, error_model="numpy")
def _compute_coefficients_core(
        delta_tau: npt.NDArray[np.float64],
        coefficients: npt.NDArray[np.float64]
) -> None:
    """
    Compute alpha, beta, gamma coefficients and store exp(-delta_tau) in coefficients[:, 0].
    This modifies coefficients in-place.
    """
    n_layers, _, n_angles, n_wavelengths = coefficients.shape
    delta_tau_limit = 1.4e-1

    for k in numba.prange(n_wavelengths):
        for i in range(n_layers):
            for j in range(n_angles):
                dt = delta_tau[i, j, k]

                if dt == 0.0:
                    coefficients[i, 0, j, k] = 1.0
                    # alpha, beta, gamma tend to 0 as dt -> 0.
                    coefficients[i, 1, j, k] = 0.0
                    coefficients[i, 2, j, k] = 0.0
                    coefficients[i, 3, j, k] = 0.0
                elif dt < delta_tau_limit:
                    coefficients[i, 0, j, k] = np.exp(-dt)
                    # Taylor expansion (Horner form)
                    coefficients[i, 1, j, k] = dt * (dt * (dt * (dt * (
                            dt * (dt * ((10 - dt) * dt - 90) + 720) - 5040) + 30240) - 151200) + 604800) / 1814400
                    coefficients[i, 2, j, k] = dt * (dt * (dt * (dt * (dt * (
                            dt * ((140 - 18 * dt) * dt - 945) + 5400) - 25200) + 90720) - 226800) + 302400) / 907200
                    coefficients[i, 3, j, k] = dt * (dt * (dt * (dt * (dt * (
                            dt * ((35 - 4 * dt) * dt - 270) + 1800) - 10080) + 45360) - 151200) + 302400) / 907200
                else:
                    # Exact formula
                    exp_neg_tau = np.exp(-dt)
                    dt_sq = dt * dt

                    coefficients[i, 0, j, k] = exp_neg_tau
                    coefficients[i, 1, j, k] = (2 + dt_sq - 2 * dt - 2 * exp_neg_tau) / dt_sq
                    coefficients[i, 2, j, k] = (2 - (2 + 2 * dt + dt_sq) * exp_neg_tau) / dt_sq
                    coefficients[i, 3, j, k] = (2 * dt - 4 + (2 * dt + 4) * exp_neg_tau) / dt_sq


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _compute_control_points_outward(
        tau_mu_matrix: npt.NDArray[np.float64],
        source_func_mu: npt.NDArray[np.float64],
        control_points: npt.NDArray[np.float64],
        coefficients: npt.NDArray[np.float64],
        start: int = 0,
        end: int | None = None,
) -> None:
    """Compute outward control points (index 0)."""
    n_layers, n_angles, n_wavelengths = tau_mu_matrix.shape
    if end is None or end > n_layers - 1:
        end = n_layers - 1

    # derivative indices required: i in [start-2, end).
    deriv_start = max(start - 2, 1)
    deriv_end = min(end, n_layers - 1)

    for k in numba.prange(n_wavelengths):
        for j in range(n_angles):
            # Compute derivatives
            d_s_d_tau_out = np.zeros(n_layers, dtype=np.float64)

            for i in range(deriv_start, deriv_end):
                tau_diff = tau_mu_matrix[i, j, k] - tau_mu_matrix[i + 1, j, k]
                if tau_diff != 0:
                    d_diff = (source_func_mu[i, j, k] - source_func_mu[i + 1, j, k]) / tau_diff

                    if i > 0:
                        tau_diff_prev = tau_mu_matrix[i - 1, j, k] - tau_mu_matrix[i, j, k]
                        if tau_diff_prev != 0:
                            d_diff_prev = (source_func_mu[i - 1, j, k] - source_func_mu[i, j, k]) / tau_diff_prev
                        else:
                            d_diff_prev = 0

                        zeta_denom = tau_mu_matrix[i - 1, j, k] - tau_mu_matrix[i + 1, j, k]
                        if zeta_denom != 0:
                            zeta = (1 + (tau_mu_matrix[i - 1, j, k] - tau_mu_matrix[i, j, k]) / zeta_denom) / 3
                        else:
                            zeta = 1.0 / 3.0

                        numerator = d_diff * d_diff_prev
                        denominator = zeta * d_diff_prev + (1 - zeta) * d_diff

                        if numerator >= 0 and denominator != 0:
                            d_s_d_tau_out[i] = numerator / denominator

            # Compute control points with clamping
            for i in range(max(start, 1), min(end + 1, n_layers)):
                tau_diff = tau_mu_matrix[i - 1, j, k] - tau_mu_matrix[i, j, k]

                control_0 = source_func_mu[i, j, k] + 0.5 * tau_diff * d_s_d_tau_out[i]
                control_1 = source_func_mu[i - 1, j, k] - 0.5 * tau_diff * d_s_d_tau_out[i - 1]

                min_source = min(source_func_mu[i - 1, j, k], source_func_mu[i, j, k])
                max_source = max(source_func_mu[i - 1, j, k], source_func_mu[i, j, k])

                control_0 = max(min(control_0, max_source), min_source)
                control_1 = max(min(control_1, max_source), min_source)

                if i == 1:
                    control_points[i, 0, j, k] = control_1
                else:
                    control_points[i, 0, j, k] = 0.5 * (control_0 + control_1)

                # Clamp to non-negative if gamma > 0; sign of gamma*C must be +.
                if coefficients[i, 3, j, k] > 0 > control_points[i, 0, j, k]:
                    # TODO: Check gamma index offset.
                    control_points[i, 0, j, k] = 0.0
    # End.


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _compute_control_points_inward(
        tau_mu_matrix: npt.NDArray[np.float64],
        source_func_mu: npt.NDArray[np.float64],
        control_points: npt.NDArray[np.float64],
        coefficients: npt.NDArray[np.float64],
        start: int = 0,
        end: int | None = None,
) -> None:
    """Compute inward control points (index 1)."""
    n_layers, n_angles, n_wavelengths = tau_mu_matrix.shape

    if end is None or end > n_layers - 1:
        end = n_layers - 1

    # derivative indices required: i in [start, end+2).
    deriv_start = start
    deriv_end = min(end + 2, n_layers - 1)

    for k in numba.prange(n_wavelengths):
        for j in range(n_angles):
            # Compute derivatives
            d_s_d_tau_in = np.zeros(n_layers, dtype=np.float64)

            for i in range(deriv_start, deriv_end):
                tau_diff = tau_mu_matrix[i + 1, j, k] - tau_mu_matrix[i, j, k]
                if tau_diff != 0:
                    d_diff = (source_func_mu[i + 1, j, k] - source_func_mu[i, j, k]) / tau_diff

                    if i < n_layers - 2:
                        tau_diff_next = tau_mu_matrix[i + 2, j, k] - tau_mu_matrix[i + 1, j, k]
                        if tau_diff_next != 0:
                            d_diff_next = (source_func_mu[i + 2, j, k] - source_func_mu[i + 1, j, k]) / tau_diff_next
                        else:
                            d_diff_next = 0

                        zeta_denom = tau_mu_matrix[i + 2, j, k] - tau_mu_matrix[i, j, k]
                        if zeta_denom != 0:
                            zeta = (1 + (tau_mu_matrix[i + 2, j, k] - tau_mu_matrix[i + 1, j, k]) / zeta_denom) / 3
                        else:
                            zeta = 1.0 / 3.0

                        numerator = d_diff * d_diff_next
                        denominator = zeta * d_diff_next + (1 - zeta) * d_diff

                        if numerator >= 0 and denominator != 0:
                            d_s_d_tau_in[i + 1] = numerator / denominator

            # Compute control points with clamping
            for i in range(max(start, 0), min(end + 1, n_layers - 1)):
                tau_diff = tau_mu_matrix[i + 1, j, k] - tau_mu_matrix[i, j, k]

                control_0 = source_func_mu[i, j, k] + 0.5 * tau_diff * d_s_d_tau_in[i]
                control_1 = source_func_mu[i + 1, j, k] - 0.5 * tau_diff * d_s_d_tau_in[i + 1]

                min_source = min(source_func_mu[i, j, k], source_func_mu[i + 1, j, k])
                max_source = max(source_func_mu[i, j, k], source_func_mu[i + 1, j, k])

                control_0 = max(min(control_0, max_source), min_source)
                control_1 = max(min(control_1, max_source), min_source)

                if i == n_layers - 2:
                    control_points[i, 1, j, k] = control_1
                else:
                    control_points[i, 1, j, k] = 0.5 * (control_0 + control_1)

                # Clamp to non-negative if gamma > 0; sign of gamma*C must be +.
                if i > 0 > control_points[i, 1, j, k] and coefficients[i, 3, j, k] > 0:
                    # TODO: Check gamma index offset.
                    control_points[i, 1, j, k] = 0
    # End.


# @numba.njit(parallel=True, cache=True, error_model="numpy")
def bezier_coefficients_new(
        tau_mu_matrix: npt.NDArray[np.float64],
        source_function_matrix: u.Quantity,
) -> t.Tuple[npt.NDArray[np.float64], u.Quantity]:
    """
    Computes the Bézier coefficients and control points used for interpolation.

    Parameters
    ----------
    tau_mu_matrix : ndarray
        Optical depth matrix [n_layers, n_angles, n_wavelengths].
    source_function_matrix : ndarray
        Source function [n_layers, n_wavelengths].

    Returns
    -------
    tuple of (coefficients, control_points)
        coefficients : ndarray
            [n_layers+1, 4, n_angles, n_wavelengths];
            [:,0,:,:] is :math:`\\exp(-\\Delta\\tau)`, [:,1,:,:] is :math:`\\alpha`, [:,2,:,:] is :math:`\\beta`, [:,3,:,:] is :math:`\\gamma`.
        control_points : ndarray
            [n_layers, 2, n_angles, n_wavelengths]
    """
    n_layers, n_angles, n_wavelengths = tau_mu_matrix.shape

    tau_mu_matrix = np.ascontiguousarray(tau_mu_matrix)

    # Initialize arrays
    coefficients = np.zeros((n_layers + 1, 4, n_angles, n_wavelengths), dtype=np.float64)
    control_points = np.zeros((n_layers, 2, n_angles, n_wavelengths), dtype=np.float64)

    # Compute delta_tau (difference between layers)
    # coefficients[1:, 0, :, :] = tau_mu_matrix
    # coefficients[:-1, 0, :, :] -= tau_mu_matrix
    # # Store delta_tau temporarily for coefficient computation; contiguous copy is faster.
    # delta_tau = np.ascontiguousarray(coefficients[:, 0, :, :])

    # Delta tau was previously stored in coefficients but _compute_coefficients_core no longer accesses these and we
    # do not need to keep them for output.
    delta_tau = np.zeros((n_layers + 1, n_angles, n_wavelengths), dtype=np.float64)
    delta_tau[1:, :, :] = tau_mu_matrix
    delta_tau[:-1, :, :] -= tau_mu_matrix

    # Compute alpha, beta, gamma and overwrite coefficients[:, 0] with exp(-delta_tau)
    _compute_coefficients_core(delta_tau, coefficients)

    # Expand source function to all angles
    # Division through by mu was removed; following the maths through the control points are angle independent.
    # It also braks the clamping checks which ensure monotonicity.
    # source_func_mu = np.empty((n_layers, n_angles, n_wavelengths))
    # for i in range(n_layers):
    #     for j in range(n_angles):
    #         source_func_mu[i, j, :] = source_function_matrix[i, :]
    source_func_mu = np.ascontiguousarray(
        np.broadcast_to(source_function_matrix.value[:, None, :], (n_layers, n_angles, n_wavelengths))
    )

    # Compute control points
    _compute_control_points_outward(tau_mu_matrix, source_func_mu, control_points, coefficients)
    _compute_control_points_inward(tau_mu_matrix, source_func_mu, control_points, coefficients)

    return coefficients, control_points * source_function_matrix.unit


@numba.njit(parallel=True, cache=True, error_model="numpy")
def update_layer_coefficients(
        layer_idx: int,
        tau_mu_matrix: npt.NDArray[np.float64],
        source_function_matrix: npt.NDArray[np.float64],
        coefficients: npt.NDArray[np.float64],
        control_points: npt.NDArray[np.float64]
) -> None:
    n_layers, n_angles, n_wavelengths = tau_mu_matrix.shape
    delta_tau_limit = 1.4e-1

    update_idxs = []
    if layer_idx > 0:
        update_idxs.append(layer_idx)
    if layer_idx < n_layers - 1:
        update_idxs.append(layer_idx + 1)

    for k in numba.prange(n_wavelengths):
        for j in range(n_angles):
            for i in update_idxs:
                dt = tau_mu_matrix[i - 1, j, k] - tau_mu_matrix[i, j, k]

                if dt == 0.0:
                    coefficients[i, 0, j, k] = 1.0
                    # Alpha, beta, gamma tend to 0 as dt -> 0.
                    coefficients[i, 1, j, k] = 0.0
                    coefficients[i, 2, j, k] = 0.0
                    coefficients[i, 3, j, k] = 0.0
                elif dt < delta_tau_limit:
                    # Taylor expansion (Horner form)
                    coefficients[i, 0, j, k] = np.exp(-dt)
                    coefficients[i, 1, j, k] = dt * (dt * (dt * (dt * (
                            dt * (dt * ((10 - dt) * dt - 90) + 720) - 5040) + 30240) - 151200) + 604800) / 1814400
                    coefficients[i, 2, j, k] = dt * (dt * (dt * (dt * (dt * (
                            dt * ((140 - 18 * dt) * dt - 945) + 5400) - 25200) + 90720) - 226800) + 302400) / 907200
                    coefficients[i, 3, j, k] = dt * (dt * (dt * (dt * (dt * (
                            dt * ((35 - 4 * dt) * dt - 270) + 1800) - 10080) + 45360) - 151200) + 302400) / 907200
                else:
                    dt_sq = dt * dt
                    exp_neg_tau = np.exp(-dt)
                    # Exact formula
                    coefficients[i, 0, j, k] = exp_neg_tau
                    coefficients[i, 1, j, k] = (2 + dt_sq - 2 * dt - 2 * exp_neg_tau) / dt_sq
                    coefficients[i, 2, j, k] = (2 - (2 + 2 * dt + dt_sq) * exp_neg_tau) / dt_sq
                    coefficients[i, 3, j, k] = (2 * dt - 4 + (2 * dt + 4) * exp_neg_tau) / dt_sq

    source_func_mu = np.empty((n_layers, n_angles, n_wavelengths), dtype=np.float64)
    for i in range(n_layers):
        for j in range(n_angles):
            source_func_mu[i, j, :] = source_function_matrix[i, :]

    # Recompute control_points[i, 0] and control_points[i, 1] for i in range [layer_idx-2 ... layer_idx+2]
    min_update_idx = min(update_idxs)
    max_update_idx = max(update_idxs)
    # i_low and i_high are the inclusive indices for updating; looping must account for range bounds.
    i_low = max(min_update_idx - 1, 0)
    i_high = min(max_update_idx + 1, n_layers - 1)

    _compute_control_points_outward(
        tau_mu_matrix, source_func_mu, coefficients, control_points,
        start=i_low, end=i_high
    )
    _compute_control_points_inward(
        tau_mu_matrix, source_func_mu, coefficients, control_points,
        start=i_low, end=i_high
    )
    # Done.


############# END NEW
# DEPRECATED BELOW:
@numba.njit(parallel=True, cache=True, error_model="numpy")
def bezier_coefficients(
        tau_mu_matrix: npt.NDArray[np.float64],
        source_function_matrix: npt.NDArray[np.float64],
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Computes the Bezier coefficients delta, alpha, beta and gamma and the control points used for interpolation. Inward
    and outward directed coefficients are stored in the same array, with the first index (corresponding to the
    atmospheric layer) offset by 1. Control points are stored in the same array with the second index corresponding to
    the outward and inward components respectively.

    The Horner rule is used to expand the computation of alpha, beta and gamma when delta is small.


    See Eqs. (13-20) in https://doi.org/10.48550/arXiv.2508.12873 for full details.

    :param tau_mu_matrix:
    :param source_function_matrix:

    :return: A tuple containing two arrays: an array containing the Bezier coefficients delta, alpha, beta and gamma and
        an array containing the control points.
    """
    # New.
    n_layers, n_angles, n_wavelengths = tau_mu_matrix.shape
    coefficients = np.zeros((n_layers + 1, 4, n_angles, n_wavelengths), dtype=np.float64)
    control_points = np.zeros((n_layers, 2, n_angles, n_wavelengths), dtype=np.float64)
    d_s_d_tau_in = np.zeros_like(tau_mu_matrix, dtype=np.float64)
    d_s_d_tau_out = np.zeros_like(tau_mu_matrix, dtype=np.float64)

    # coefficients[1:-1, 0, :, :] = tau_matrix[:-1]
    # coefficients[1:-1, 0, :, :] -= tau_matrix[1:]
    # Below needed to get coefficients at the boundary layers.
    coefficients[1:, 0, :, :] = tau_mu_matrix
    coefficients[:-1, 0, :, :] -= tau_mu_matrix
    # tau_plus is delta_tau_matrix[1:], tau_minus is delta_tau_matrix[:-1]

    delta_tau_limit = 1.4e-1
    delta_tau_limit_mask = coefficients[:, 0, :, :] < delta_tau_limit

    delta_tau_sq = coefficients[:, 0, :, :] ** 2
    # delta_tau_cube = coefficients[:, 0, :, :] ** 3
    exp_neg_tau = np.exp(-coefficients[:, 0, :, :])

    denom_delta_tau_sq = np.where(delta_tau_sq == 0, 1, delta_tau_sq)

    # Change indices on delta_tau_matrix based on direction! - Old comment.
    coefficients[:, 1, :, :] = np.where(
        delta_tau_limit_mask,
        # (coefficients[:, 0, :, :] / 3) - (delta_tau_sq / 12) + (delta_tau_cube / 60),  # Explicit Taylor
        (
                coefficients[:, 0, :, :]
                * (
                        coefficients[:, 0, :, :]
                        * (
                                coefficients[:, 0, :, :]
                                * (
                                        coefficients[:, 0, :, :]
                                        * (
                                                coefficients[:, 0, :, :]
                                                * (
                                                        coefficients[:, 0, :, :]
                                                        * ((10 - coefficients[:, 0, :, :]) * coefficients[
                                                    :, 0, :, :] - 90)
                                                        + 720
                                                )
                                                - 5040
                                        )
                                        + 30240
                                )
                                - 151200
                        )
                        + 604800
                )
        )
        / 1814400,  # Horner Taylor
        (2 + delta_tau_sq - 2 * coefficients[:, 0, :, :] - 2 * exp_neg_tau) / denom_delta_tau_sq,
    )
    coefficients[:, 2, :, :] = np.where(
        delta_tau_limit_mask,
        # (coefficients[:, 0, :, :] / 3) - (delta_tau_sq / 4) + (delta_tau_cube / 10),  # Explicit Taylor
        (
                coefficients[:, 0, :, :]
                * (
                        coefficients[:, 0, :, :]
                        * (
                                coefficients[:, 0, :, :]
                                * (
                                        coefficients[:, 0, :, :]
                                        * (
                                                coefficients[:, 0, :, :]
                                                * (
                                                        coefficients[:, 0, :, :]
                                                        * ((140 - 18 * coefficients[:, 0, :, :]) * coefficients[
                                                    :, 0, :, :] - 945)
                                                        + 5400
                                                )
                                                - 25200
                                        )
                                        + 90720
                                )
                                - 226800
                        )
                        + 302400
                )
        )
        / 907200,  # Horner Taylor
        (2 - (2 + 2 * coefficients[:, 0, :, :] + delta_tau_sq) * exp_neg_tau) / denom_delta_tau_sq,
    )
    coefficients[:, 3, :, :] = np.where(
        delta_tau_limit_mask,
        # (coefficients[:, 0, :, :] / 3) - (delta_tau_sq / 6) + (delta_tau_cube / 20),  # Explicit Taylor
        (
                coefficients[:, 0, :, :]
                * (
                        coefficients[:, 0, :, :]
                        * (
                                coefficients[:, 0, :, :]
                                * (
                                        coefficients[:, 0, :, :]
                                        * (
                                                coefficients[:, 0, :, :]
                                                * (
                                                        coefficients[:, 0, :, :]
                                                        * ((35 - 4 * coefficients[:, 0, :, :]) * coefficients[
                                                    :, 0, :, :] - 270)
                                                        + 1800
                                                )
                                                - 10080
                                        )
                                        + 45360
                                )
                                - 151200
                        )
                        + 302400
                )
        )
        / 907200,  # Horner Taylor
        (2 * coefficients[:, 0, :, :] - 4 + (2 * coefficients[:, 0, :, :] + 4) * exp_neg_tau) / denom_delta_tau_sq,
    )
    # NEW:
    coefficients[:, 0, :, :] = exp_neg_tau

    # source_func_mu = source_function_matrix.reshape(n_layers, 1, n_wavelengths) / mu_values.reshape(1, n_angles, 1)
    # NEW! Note: Control points are mu independent, so dividing by mu breaks the clamping checks. Still need to reshape
    # for division through by tau/mu.
    source_func_mu = source_function_matrix.reshape(n_layers, 1, n_wavelengths) * np.ones((1, n_angles, 1))
    min_source_mu = np.fmin(source_func_mu[:-1], source_func_mu[1:])
    max_source_mu = np.fmax(source_func_mu[:-1], source_func_mu[1:])

    # if np.any(min_source_mu < 0):
    #     print(f"WARN: Min source below 0!")
    # if np.any(max_source_mu < 0):
    #     print(f"WARN: Max source below 0!")

    tau_matrix_out_1_diff = tau_mu_matrix[:-1] - tau_mu_matrix[1:]
    d_diff_out = np.where(
        tau_matrix_out_1_diff == 0,
        0,
        (source_func_mu[:-1] - source_func_mu[1:]) / tau_matrix_out_1_diff,
    )
    zeta_out_denominator = tau_mu_matrix[:-2] - tau_mu_matrix[2:]
    zeta_out = np.where(
        zeta_out_denominator == 0,
        1 / 3,
        (1 + (tau_mu_matrix[:-2] - tau_mu_matrix[1:-1]) / zeta_out_denominator) / 3,
    )
    d_s_d_tau_out_numerator = d_diff_out[1:] * d_diff_out[:-1]
    d_s_d_tau_out_denominator = (zeta_out * d_diff_out[:-1]) + ((1 - zeta_out) * d_diff_out[1:])
    d_s_d_tau_out[1:-1] = np.where(
        (d_s_d_tau_out_numerator < 0) | (d_s_d_tau_out_denominator == 0),
        0,
        d_s_d_tau_out_numerator / d_s_d_tau_out_denominator,
    )

    control_0_out = source_func_mu[1:] + 0.5 * tau_matrix_out_1_diff * d_s_d_tau_out[1:]
    control_1_out = source_func_mu[:-1] - 0.5 * tau_matrix_out_1_diff * d_s_d_tau_out[:-1]

    control_0_out = np.fmax(control_0_out, min_source_mu)
    control_0_out = np.fmin(control_0_out, max_source_mu)
    control_1_out = np.fmax(control_1_out, min_source_mu)
    control_1_out = np.fmin(control_1_out, max_source_mu)

    control_points[2:, 0, :, :] = 0.5 * (control_0_out[1:] + control_1_out[1:])
    control_points[1, 0, :, :] = control_1_out[0]

    # control_points[1:, 0, :, :] = np.fmax(control_points[1:, 0, :, :], 0)
    control_points[1:, 0, :, :] = np.where(
        (coefficients[1:-1, 3, :, :] > 0) & (control_points[1:, 0, :, :] < 0),
        0,
        control_points[1:, 0, :, :],
    )

    tau_matrix_in_1_diff = tau_mu_matrix[1:] - tau_mu_matrix[:-1]
    d_diff_in = np.where(
        tau_matrix_in_1_diff == 0,
        0,
        (source_func_mu[1:] - source_func_mu[:-1]) / tau_matrix_in_1_diff,
    )
    zeta_in_denominator = tau_mu_matrix[2:] - tau_mu_matrix[:-2]
    zeta_in = np.where(
        zeta_in_denominator == 0,
        1 / 3,
        (1 + (tau_mu_matrix[2:] - tau_mu_matrix[1:-1]) / zeta_in_denominator) / 3,
    )
    d_s_d_tau_in_numerator = d_diff_in[:-1] * d_diff_in[1:]
    d_s_d_tau_in_denominator = (zeta_in * d_diff_in[1:]) + ((1 - zeta_in) * d_diff_in[:-1])
    d_s_d_tau_in[1:-1] = np.where(
        (d_s_d_tau_in_numerator < 0) | (d_s_d_tau_in_denominator == 0),
        0,
        d_s_d_tau_in_numerator / d_s_d_tau_in_denominator,
    )

    control_0_in = source_func_mu[:-1] + 0.5 * tau_matrix_in_1_diff * d_s_d_tau_in[:-1]
    control_1_in = source_func_mu[1:] - 0.5 * tau_matrix_in_1_diff * d_s_d_tau_in[1:]

    control_0_in = np.fmax(control_0_in, min_source_mu)
    control_0_in = np.fmin(control_0_in, max_source_mu)
    control_1_in = np.fmax(control_1_in, min_source_mu)
    control_1_in = np.fmin(control_1_in, max_source_mu)

    control_points[:-2, 1, :, :] = 0.5 * (control_0_in[:-1] + control_1_in[:-1])
    control_points[-2, 1, :, :] = control_1_in[-1]

    # control_points[:-1, 1, :, :] = np.fmax(control_points[:-1, 1, :, :], 0)
    control_points[:-1, 1, :, :] = np.where(
        (coefficients[1:-1, 3, :, :] > 0) & (control_points[:-1, 1, :, :] < 0),
        0,
        control_points[:-1, 1, :, :],
    )

    return coefficients, control_points


def blackbody(spectral_grid: u.Quantity, temperature: u.Quantity) -> u.Quantity:
    freq_grid = spectral_grid.to(u.Hz, equivalencies=u.spectral())
    temperature = np.atleast_1d(temperature)[:, None]
    return (ac_2_h_on_c_sq * freq_grid ** 3) / (np.exp(ac_h_on_kB * freq_grid / temperature) - 1) / u.sr


def incident_stellar_radiation(
        wn_grid: u.Quantity, star_temperature: u.Quantity, orbital_radius: u.Quantity, planet_radius: u.Quantity
) -> u.Quantity:
    """
    Assume the angular size of the planet relative to the star and orbital distance is small, allowing to assume that
    the surface of the planet with incident radiation is approximately a circle.

    :param wn_grid:
    :param star_temperature:
    :param orbital_radius:
    :param planet_radius:
    :return:
    """
    star_bb = blackbody(spectral_grid=wn_grid, temperature=star_temperature)[0]
    incident_radiation = star_bb * (planet_radius / orbital_radius) ** 2
    return incident_radiation.to(star_bb.unit, equivalencies=u.spectral())


def incident_srf(
        star_temperature: float, star_logg: float, star_feh: float, wn_grid: u.Quantity, orbital_radius: u.Quantity,
        star_radius: u.Quantity, star_alpha: float = 0.0
) -> u.Quantity:
    """
    Returns the substellar flux at the planet's surface.

    Parameters
    ----------
    star_temperature
    star_logg
    star_feh
    wn_grid
    orbital_radius
    star_radius
    star_alpha

    Returns
    -------
        Substellar flux [W /(m^2 cm^-1)].

    """
    srf_wavelength, srf_flux = get_spectrum(
        teff=star_temperature,
        logg=star_logg,
        feh=star_feh,
        alpha=star_alpha,
        source="synphot",
    )
    # srf_flux has units of erg / (Angstrom s cm^2)
    srf_wn = srf_wavelength.to(1 / u.cm, equivalencies=u.spectral())

    # srf_flux_wn = srf_flux / (srf_wn**2)
    # srf_flux_wn = srf_flux_wn.to(u.J / (u.s * u.m**2 * (1/u.cm)))

    srf_flux_nu = srf_flux.to(
        u.J / (u.s * u.m ** 2 * u.Hz),
        equivalencies=u.spectral_density(srf_wavelength)
    )

    sort_idx = np.argsort(srf_wn)
    srf_wn = srf_wn[sort_idx]
    srf_flux_nu = srf_flux_nu[sort_idx]

    srf_flux_interp = np.interp(wn_grid, srf_wn, srf_flux_nu, left=0, right=0) << srf_flux_nu.unit
    # This is F_nu [W/(Hz*m^2)]
    srf_flux_orbit = srf_flux_interp * (star_radius / orbital_radius) ** 2
    theta = np.arcsin((star_radius / orbital_radius).decompose().value)
    omega_star = 2 * np.pi * (1 - np.cos(theta)) * u.sr
    srf_specific_intensity = (srf_flux_orbit / omega_star).to(u.W / (u.Hz * u.m ** 2 * u.sr))
    return srf_specific_intensity


@numba.njit()
def calc_einstein_b_fi(a_fi: npt.NDArray[np.float64], energy_fi: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Here the Einstein B coefficient is given in :math:`\text{m}^2 / (\text{J·s})`.

    .. math::
        B_{fi}=\\frac{A_{fi}}{2hc\\tilde{\\nu}^{3}_{fi}}

    :param a_fi:
    :param energy_fi:
    :return:
    """
    # return a_fi / (2 * ac.h * ac.c * (energy_fi ** 3))  # WAVENUMBERS
    return (a_fi * (ac.c ** 2)) / (2 * ac.h * (energy_fi ** 3))  # FREQUENCY
    # return a_fi * (energy_fi ** 3) / (2 * ac.h * ac.c)  # LAMBDA


@numba.njit()
def calc_einstein_b_if(
        b_fi: npt.NDArray[np.float64],
        g_f: npt.NDArray[np.float64],
        g_i: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return b_fi * g_f / g_i


def boltzmann_population(states: pl.DataFrame, temperature: u.Quantity) -> pl.DataFrame:
    # TODO: Update so this is not needed!
    if isinstance(states, pd.DataFrame):
        states = pl.from_pandas(states)

    g_np = states["g"].to_numpy()
    energy_np = states["energy"].to_numpy() << 1 / u.cm

    q_lev_np = g_np * np.exp(-ac_h_c_on_kB * energy_np / temperature).value
    n_np = q_lev_np / np.sum(q_lev_np)

    states = states.with_columns(
        pl.Series("q_lev", q_lev_np),
        pl.Series("n", n_np)
    )

    temp_pop_df = pl.DataFrame({
        "id_agg": states["id_agg"],
        "n": n_np,
    })
    states_agg_n = temp_pop_df.group_by("id_agg").agg(
        pl.col("n").sum().alias("n_agg")
    )
    states = states.join(states_agg_n, on="id_agg", how="left")
    # states["q_lev"] = states["g"] * np.exp(-ac_h_c_on_kB * (states["energy"] << 1 / u.cm) / temperature)
    # states["n"] = states["q_lev"] / states["q_lev"].sum()
    # states_agg_n = states.groupby(by=["id_agg"], as_index=False).agg(n_agg=("n", "sum"))
    # states = states.merge(states_agg_n, on=["id_agg"], how="left")
    return states


# def boltzmann_population(states: pd.DataFrame, temperature: u.Quantity) -> pd.DataFrame:
#     states["q_lev"] = states["g"] * np.exp(-ac_h_c_on_kB * (states["energy"] << 1 / u.cm) / temperature)
#     states["n"] = states["q_lev"] / states["q_lev"].sum()
#     states_agg_n = states.groupby(by=["id_agg"], as_index=False).agg(n_agg=("n", "sum"))
#     states = states.merge(states_agg_n, on=["id_agg"], how="left")
#     return states


@numba.njit(cache=True, error_model="numpy")
def calc_ev_grid(wn_grid: npt.NDArray[np.float64], temperature: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return (const_2_pi_h_c_sq_on_sigma_sba * wn_grid ** 3) / (
            temperature ** 4 * (np.exp(const_h_c_on_kB * wn_grid / temperature) - 1)
    )


@numba.njit(cache=True, error_model="numpy", inline="always")
def _sample_indices(
        ev_cdf: npt.NDArray[np.float64],
        temp_wn_grid: npt.NDArray[np.float64],
        num_points: int,
        max_step: float
):
    """
    Finds the indices in the energy density cumulative distribution function (CDF) such that the energy density is uniform.
    This is done by determining a uniform step size through the CDF based on the number of steps and ensuring the
    difference in the cumulative CDF between each point does not exceed this step size. An additional constraint is
    imposed such that the step size on the wavenumber grid also cannot exceed a maximum value. As such the number of
    points requested by the function is the lower bound on theh number of points returned.

    Parameters
    ----------
    ev_cdf: ndarray
        Energy density cumulative distribution function.
    temp_wn_grid: ndarray
        Temporary high-resolution wavenumber grid (with linear spacing).
    num_points: int
        Lower bound on the number of indices to return to sample the CDF.
    max_step: float
        Maximum step size along the wavenumber grid between each successive index.

    Returns
    -------
        Integer indices corresponding to the uniform samplign points in the CDF.

    """
    num_cdf_points = len(ev_cdf)
    sample_idxs = [0]
    current_idx = 0
    step_size = 1.0 / num_points

    while current_idx < num_cdf_points - 1:
        next_step = ev_cdf[current_idx] + step_size
        next_step_idx = np.searchsorted(ev_cdf, next_step) - 1
        next_step_idx = min(max(next_step_idx, current_idx + 1), num_cdf_points - 1)

        current_wn_val = temp_wn_grid[current_idx]
        next_wn_val = temp_wn_grid[next_step_idx]

        if max_step > 0.0 and (next_wn_val - current_wn_val) > max_step:
            seek_wn_val = current_wn_val + max_step
            seek_idx = np.searchsorted(temp_wn_grid, seek_wn_val) - 1
            current_idx = min(seek_idx, num_cdf_points - 1)
        elif next_step_idx >= num_cdf_points - 1:
            current_idx = num_cdf_points - 1
        else:
            current_idx = next_step_idx
        sample_idxs.append(current_idx)
    return np.array(sample_idxs, dtype=np.int64)


def cdf_opacity_sampling(
        wn_start: float,
        wn_end: float,
        temperature_profile: npt.NDArray[np.float64],
        num_points: int,
        max_step: float = None,
        num_cdf_points: int = 1000000,
) -> u.Quantity:
    temp_wn_grid = np.linspace(wn_start, wn_end, num_cdf_points, dtype=np.float64)
    ev_grid = calc_ev_grid(wn_grid=temp_wn_grid, temperature=np.atleast_1d(temperature_profile)[:, None]).sum(axis=0)
    ev_norm = ev_grid / simpson(ev_grid, x=temp_wn_grid)

    ev_cdf = cumulative_simpson(ev_norm, x=temp_wn_grid, initial=0)

    sample_idxs = _sample_indices(ev_cdf, temp_wn_grid, num_points, max_step)

    return temp_wn_grid[sample_idxs] / u.cm


def formal_solve_general(
        dtau: u.Quantity,
        source_function: u.Quantity,
        mu_values: npt.NDArray[np.float64],
        mu_weights: npt.NDArray[np.float64],
        incident_radiation_field: u.Quantity = None,
        surface_albedo: float = 0
) -> t.Tuple[u.Quantity, u.Quantity]:
    """
    Solve the 1D plane–parallel radiative-transfer equation for a discretized atmosphere using the *formal solution* for
    each direction cosine :math:`\\mu`.

    This routine computes **upward** and **downward** specific intensities at every layer interface, then integrates
    over angle to obtain the hemispheric fluxes.

    ----------------------------------------------------------------------
    RADIATIVE-TRANSFER EQUATION
    ----------------------------------------------------------------------

    For a ray of direction cosine :math:`\\mu`, the monochromatic radiative-transfer equation in optical depth
    :math:`\\tau` is

    .. math::
        \\mu \\frac{\\mathrm{d} I(\\tau,\\mu)}{\\mathrm{d}\\tau} = I(\\tau,\\mu) - S(\\tau),

    where :math:`S(\\tau)` is the source function.

    The *formal solution* between two optical-depth points :math:`\\tau_{k}` and :math:`\\tau_{k+1}` is:

    .. math::
        I(\\tau_k,\\mu)
        = I(\\tau_{k+1},\\mu) \\, e^{-\\Delta\\tau/\\lvert\\mu\\rvert}
        + S_{k} \\,\\left(1 - e^{-\\Delta\\tau/\\lvert\\mu\\rvert}\\right),

    where :math:`\\Delta\\tau = \\tau_{k+1} - \\tau_{k}`.

    This expression is used for **downward** (TOA to BOA) rays with :math:`\\mu > 0` and **upward** (BOA to TOA) rays
    with :math:`\\mu < 0`.

    ----------------------------------------------------------------------
    NUMERICAL DISCRETIZATION
    ----------------------------------------------------------------------

    The atmosphere is divided into :math:`n_{\\mathrm{layers}}` layers. For each wavenumber :math:`\\tilde{\\nu}` the
    inputs have shapes:

    * :math:`\\Delta\\tau`: ``(n_layers, n_wn)`` optical-depth increment per layer.
    * ``source_function``: ``(n_layers, n_wn)`` source function at each point.
    * ``mu_values``: ``(n_mu,)`` direction cosines.
    * ``mu_weights``: ``(n_mu,)`` quadrature weights.

    Intensities are stored at the **interfaces**, so the output arrays have dimension ``n_layers + 1``.

    ----------------------------------------------------------------------
    BOUNDARY CONDITIONS
    ----------------------------------------------------------------------

    * At the top of atmosphere (TOA):

      .. math::
         I^{-}_{n_{\\mathrm{layers}}}(\\mu>0) = I_{\\mathrm{incident}} \\text{(if given)}.

    * At the bottom of the atmosphere (BOA):

      If no surface reflection is treated explicitly, the upward intensity is set to the source function of the lowest
      layer:

      .. math::
         I^{+}_{0}(\\mu<0) = S_{0}.

    ----------------------------------------------------------------------
    ANGULAR INTEGRATION
    ----------------------------------------------------------------------

    After computing intensities for each :math:`\\mu`, hemispheric fluxes are computed as:

    .. math::
        F^{\\pm}(\\tilde{\\nu})
        = 2\\pi \\sum_{i=1}^{n_{\\mu}} I^{\\pm}_i(\\tilde{\\nu}) \\, w_{i},

    where :math:`w_{i}` are the angular quadrature weights.

    Parameters
    ----------
    dtau : Quantity, shape (n_layers, n_wn)
        Optical-depth increment :math:`\\Delta\\tau_{j}(\\tilde{\\nu})` for each layer :math:`j` and wavenumber
        :math:`\\tilde{\\nu}`.
    source_function : Quantity, shape (n_layers, n_wn)
        Source function :math:`S_{j}(\\tilde{\\nu})` per layer.
    mu_values : ndarray, shape (n_mu,)
        Direction cosines :math:`\\mu_{i}`.
    mu_weights : ndarray, shape (n_mu,)
        Angular quadrature weights :math:`w_{i}` corresponding to ``mu_values``.
    incident_radiation_field : Quantity, shape (n_mu, n_wn), optional
        Downward incident intensity at TOA, :math:`I^{-}(\\tau_{\\mathrm{top}})`; defaults to zero.
    surface_albedo : float, optional
        Surface albedo :math:`A \\in [0,1]`. If nonzero, reflection modifies the BOA upward intensity. (Current
        implementation uses a simplified placeholder.)

    Returns
    -------
    i_up : Quantity, shape (n_layers + 1, n_wn)
        Hemispherically integrated *upward* flux:

        .. math::
            F^{+}(\\tilde{\\nu}) = 2\\pi \\sum_{i} I^{+}_{i}(\\tilde{\\nu}) w_{i}.

    i_down : Quantity, shape (n_layers + 1, n_wn)
        Hemispherically integrated *downward* flux:

        .. math::
            F^{-}(\\tilde{\\nu}) = 2\\pi \\sum_{i} I^{-}_{i}(\\tilde{\\nu}) w_{i}.
    """
    if surface_albedo < 0 or surface_albedo > 1:
        log.warning(f"Surface albedo {surface_albedo} is outside of [0, 1], clipping.")
        surface_albedo = np.clip(surface_albedo, 0, 1)

    n_layers, n_wavelengths = dtau.shape

    # Compute intensity at interfaces.
    i_up = np.zeros((len(mu_values), n_layers + 1, n_wavelengths)) * source_function.unit
    i_down = np.zeros((len(mu_values), n_layers + 1, n_wavelengths)) * source_function.unit

    # Upper boundary condition at the top (level n_layers) is zero, unless incident radiation field!
    if incident_radiation_field is not None:
        i_down[:, n_layers, :] = incident_radiation_field
    else:
        i_down[:, n_layers, :] = 0.0 * source_function.unit

    # Integrate from TOA (k=n_layers-1) down to BOA (k=0)
    for k in range(n_layers - 1, -1, -1):
        delta_tau_mu = dtau[k, :] / np.abs(mu_values[:, None])
        exp_term = np.exp(-delta_tau_mu)
        source_contribution = source_function[k, :] * (1 - exp_term)

        # Intensity at the top interface of each layer
        i_down[:, k, :] = i_down[:, k + 1, :] * exp_term + source_contribution

    # Include an albedo for terrestrial planets?
    # downward_flux = 2 * np.pi * (i_down[:, 0, :] * mu_values[:, None] * mu_weights[mu_weights > 0, None]).sum(axis=0)
    # Reflected intensity is diffuse (same in all directions)
    # reflected_intensity = surface_albedo * downward_flux / np.pi

    # bb = blackbody(...)
    # thermal_emission = bb(dtau.shape[1] * u.nm) # Wavelength version?
    # thermal_emission = source_function[-1] * surface_emissivity # Placeholder for scaling?
    # surface_emission = thermal_emission + reflected_intensity
    surface_emission = source_function[0, :]  # USE THIS IN PROD!

    # Lower boundary source function (black body) is surface upwards emission.
    i_up[:, 0, :] = surface_emission

    # Integrate from BOA (k=0) to TOA (k=n_layers-1)
    for k in range(n_layers):
        delta_tau_mu = dtau[k, :] / mu_values[:, None]
        exp_term = np.exp(-delta_tau_mu)
        source_contribution = source_function[None, k, :] * (1 - exp_term)

        # Intensity at the top of the layer (level k+1)
        i_up[:, k + 1, :] = i_up[:, k, :] * exp_term + source_contribution

    i_up = 2 * np.pi * u.sr * np.sum(i_up * mu_weights[:, None, None], axis=0)
    i_down = 2 * np.pi * u.sr * np.sum(i_down * mu_weights[:, None, None], axis=0)

    return i_up, i_down


class NLTEWorkflow(abc.ABC):
    # In theory NLTEProcessor can implement a generic NLTEWorkflow field that calls the workflow method. The current 
    # implementation hard-cords the Gauss-Seidel workflow, but it may be useful to be able to switch between that and
    # MALI without having to restructure the whole main workflow.
    @abc.abstractmethod
    def workflow(self) -> t.Any:
        pass


class GaussSeidelWorkflow(NLTEWorkflow):

    def workflow(self):
        # Implement current workflow from compute_opacities_profile().
        pass


class MALIWorkflow(NLTEWorkflow):

    def workflow(self):
        # Implement MALI layer step through, intensity and Lambda calculations.
        pass
