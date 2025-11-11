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
from scipy.special import roots_hermite

from .config import log, _DEFAULT_CHUNK_SIZE, _N_GH_QUAD_POINTS, _INTENSITY_CUTOFF

# Constants with units:
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


def _sum_profiles(group: pl.DataFrame) -> pl.DataFrame:
    summed_profiles = np.stack(group["profile"].to_numpy(), axis=0).sum(axis=0)
    return pl.DataFrame({
        "id_agg_f": [group["id_agg_f"][0]],
        "id_agg_i": [group["id_agg_i"][0]],
        "profile": [summed_profiles],
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
):
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

    n_threads = numba.get_num_threads()
    # Allocate per-thread accumulation buffer
    # Note: This allocation is performed inside the njit function; it is ok and reused only for the scope of this call.
    buffers = np.zeros((n_threads, wn_grid_len), dtype=np.float64)

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
        for t in range(n_threads):
            xsec_point += buffers[t, k]
        xsec_out[k] = xsec_point

    return xsec_out


class CompactProfile:
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
        if profile_type == "abs":
            return self.abs_profiles[layer_idx].get_profile(key)
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


# class BandProfile:
#     __slots__ = ("start_idx", "profile", "integral")
#
#     def __init__(self, profile: npt.NDArray[np.float64], start_idx: int = None, trim: bool = True) -> None:
#         """
#
#         :param start_idx: Index on the spectroscopic grid where the trimmed band profile begins.
#         :param profile:   The band profile of the transition.
#         integral:         The integral of the absorption profile pre-normalisation.
#         """
#         self.integral = 0.0
#         if not trim:
#             self.start_idx = 0
#             self.profile = profile
#         elif start_idx is None:
#             if np.all(profile < _INTENSITY_CUTOFF):
#                 log.warning(f"All of something is below the cutoff! len={len(self.profile)}")
#                 self.start_idx = 0
#                 self.profile = np.empty(0)
#             else:
#                 self.start_idx = np.argmax(profile >= _INTENSITY_CUTOFF)
#                 end_idx = len(profile) - np.argmax(profile[::-1] >= _INTENSITY_CUTOFF)
#                 self.profile = profile[self.start_idx: end_idx]
#         else:
#             self.start_idx = start_idx
#             self.profile = profile
#
#     def __repr__(self):
#         return f"BandProfile([{self.start_idx}, {self.profile}, {self.integral}])"
#
#     def __str__(self):
#         return (
#             f"BandProfile(start_idx: {self.start_idx}, profile: "
#             f"{self.profile}"
#             f"({self.profile.size} points),"
#             f" integral: {self.integral},"
#             ")"
#         )
#
#     def merge_band_profiles(
#             self, band_profiles: t.List["BandProfile"], normalise: bool = False, spectral_grid: u.Quantity = None
#     ) -> None:
#         # TODO: Handle case where some band_profiles are empty?
#         if len(band_profiles) > 0:
#             start_idxs = np.concatenate(([self.start_idx], [band_profile.start_idx for band_profile in band_profiles]))
#             profiles = [self.profile] + [band_profile.profile for band_profile in band_profiles]
#
#             min_start_idx = min(start_idxs)
#             primary_idx = np.argmax(start_idxs == min_start_idx)
#             max_end_idx = max(
#                 np.concatenate(
#                     (
#                         [self.start_idx + len(self.profile)],
#                         [band_profile.start_idx + len(band_profile.profile) for band_profile in band_profiles],
#                     )
#                 )
#             )
#             offset = max_end_idx - min_start_idx - len(profiles[primary_idx])
#
#             merged_profile = np.pad(profiles[primary_idx], (0, offset), "constant")
#
#             for profile_idx in range(len(start_idxs)):
#                 if profile_idx != primary_idx:
#                     profile_offset = start_idxs[profile_idx] - min_start_idx
#                     merged_profile[profile_offset: profile_offset + len(profiles[profile_idx])] += profiles[
#                         profile_idx
#                     ]
#
#             self.start_idx = min_start_idx
#             self.profile = merged_profile
#         if normalise:
#             if spectral_grid is None:
#                 raise RuntimeError("Normalisation specified but no wn_grid provided for integration.")
#             self.normalise_band_profile(spectral_grid=spectral_grid)
#
#     def normalise_band_profile(self, spectral_grid: u.Quantity) -> None:
#         if self.profile.size == 0:
#             pass
#         else:
#             if len(self.profile) == 1 and sum(self.profile) != 0:
#                 self.integral = self.profile.sum()
#             else:
#                 self.integral = simpson(
#                     self.profile,
#                     x=spectral_grid[self.start_idx: self.start_idx + len(self.profile)].value,
#                 )
#             if self.integral == 0:
#                 raise RuntimeError("Abs factor is 0 - Why?")
#             self.profile /= self.integral
#
#
# class BandProfileCollection(dict):
#     def __init__(self, band_profiles: npt.NDArray[BandProfile] | t.List[BandProfile] | pd.Series):
#         if type(band_profiles) is pd.Series:
#             for row_key, row in band_profiles.items():
#                 if row.profile.size > 0:
#                     if row_key in self:
#                         self[row_key].merge_band_profiles(row)
#                     else:
#                         self[row_key] = row
#                 else:
#                     log.info(f"BandProfile for key={row_key} is empty.")
#         # elif type(band_profiles) in (npt.NDArray[BandProfile], t.List[BandProfile]):
#         #     keys = [(band_profile.id_u, band_profile.id_l) for band_profile in band_profiles]
#         #     unique_keys = set(keys)
#         #     for unique_key in unique_keys:
#         #         key_idxs = [key_idx for key_idx, key in enumerate(keys) if key == unique_key]
#         #         key_profiles = band_profiles[key_idxs]
#         #         self[unique_key] = key_profiles[0]
#         #         if len(key_profiles) > 1:
#         #             self[unique_key].merge_band_profiles(band_profiles=key_profiles[1:])
#         else:
#             raise RuntimeError(
#                 "BandProfileCollection construction only implemented for list, np.array or pd.Series."
#                 f"Received {type(band_profiles)}."
#             )
#         super().__init__()
#
#     def __getitem__(self, key: t.Tuple[int, int] | int) -> BandProfile:
#         return super().__getitem__(key)
#
#     def get(self, key: t.Tuple[int, int] | int, default: t.Optional[t.Any] = None) -> BandProfile:
#         return super().get(key)
#
#     def __setitem__(self, key: t.Tuple[int, int] | int, value: BandProfile) -> None:
#         return super().__setitem__(key, value)
#
#     def __contains__(self, key: t.Tuple[int, int] | int) -> bool:
#         return super().__contains__(key)
#
#     def __delitem__(self, key: t.Tuple[int, int] | int) -> None:
#         return super().__delitem__(key)
#
#     def merge_collections(
#             self,
#             band_profile_collections: t.List["BandProfileCollection"] | npt.NDArray["BandProfileCollection"],
#             normalise: bool = False,
#             spectral_grid: u.Quantity = None,
#     ) -> None:
#         keys = [band_profile_collection.keys() for band_profile_collection in band_profile_collections]
#         keys = [key for sublist in keys for key in sublist]
#         unique_keys = set(keys)
#         for unique_key in unique_keys:
#             key_profiles = [
#                 band_profile_collection.get(unique_key) for band_profile_collection in band_profile_collections
#             ]
#             key_profiles = [profile for profile in key_profiles if profile is not None]
#             if unique_key in self:
#                 self[unique_key].merge_band_profiles(
#                     band_profiles=key_profiles, normalise=normalise, spectral_grid=spectral_grid
#                 )
#             else:
#                 self[unique_key] = key_profiles[0]
#                 if len(key_profiles) > 1:
#                     self[unique_key].merge_band_profiles(
#                         band_profiles=key_profiles[1:], normalise=normalise, spectral_grid=spectral_grid
#                     )
#                 else:
#                     self[unique_key].normalise_band_profile(spectral_grid=spectral_grid)
#
#     def normalise(self, spectral_grid: u.Quantity, sanitise: bool = True) -> None:
#         for key in list(self.keys()):
#             self[key].normalise_band_profile(spectral_grid=spectral_grid)
#             if sanitise and self[key].profile.size == 0:
#                 del self[key]
#
#     def add_no_trim(self, key: t.Tuple[int, int] | int, profile: npt.NDArray[np.float64]):
#         self[key] = BandProfile(profile=profile, trim=False)


def calc_cont_band_profile(
        wn_grid: npt.NDArray[np.float64],
        group: pl.DataFrame,
) -> pl.DataFrame:
    a_ci = np.ascontiguousarray(group["A_ci"].to_numpy())
    g_i = np.ascontiguousarray(group["g_i"].to_numpy())
    g_c = np.ascontiguousarray(group["g_c"].to_numpy())
    energy_ci = np.ascontiguousarray(group["energy_ci"].to_numpy())
    n_frac_i = np.ascontiguousarray(group["n_frac_i"].to_numpy())
    n_frac_c = np.ascontiguousarray(group["n_frac_c"].to_numpy())
    cont_broad = np.ascontiguousarray(group["broad"].to_numpy())

    assert a_ci.shape[0] == g_i.shape[0] == g_c.shape[0] == energy_ci.shape[0] == n_frac_i.shape[0] == n_frac_c.shape[0]

    if energy_ci.min() > wn_grid.max() or energy_ci.max() < wn_grid.min():
        raise RuntimeError("Computing band profile for band with no transitions on grid: check workflow logic.")

    is_fixed_width = np.all(np.isclose(np.diff(wn_grid), np.abs(wn_grid[1] - wn_grid[0]), atol=0))

    if is_fixed_width:
        _abs_xsec = _continuum_band_profile_fixed_width(
            wn_grid=wn_grid,
            n_frac_f=n_frac_c,
            n_frac_i=n_frac_i,
            a_fi=a_ci,
            g_f=g_c,
            g_i=g_i,
            energy_fi=energy_ci,
            cont_broad=cont_broad,
            half_bin_width=np.abs(wn_grid[1] - wn_grid[0]) / 2.0,
        )
    else:
        _abs_xsec = _continuum_band_profile_variable_width(
            wn_grid=wn_grid,
            n_frac_f=n_frac_c,
            n_frac_i=n_frac_i,
            a_fi=a_ci,
            g_f=g_c,
            g_i=g_i,
            energy_fi=energy_ci,
            cont_broad=cont_broad,
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
        _abs_xsec, _spe_xsec = _band_profile_binned_voigt_fixed_width(
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


# def calc_band_profile_old(
#         wn_grid: npt.NDArray[np.float64],
#         n_frac_i: npt.NDArray[np.float64],
#         a_fi: npt.NDArray[np.float64],
#         g_f: npt.NDArray[np.float64],
#         g_i: npt.NDArray[np.float64],
#         energy_fi: npt.NDArray[np.float64],
#         temperature: float,
#         species_mass: float,
#         n_frac_f: npt.NDArray[np.float64] = None,
#         lifetimes: npt.NDArray[np.float64] = None,
#         pressure: float = None,
#         broad_n: npt.NDArray[np.float64] = None,
#         broad_gamma: npt.NDArray[np.float64] = None,
#         cont_broad: npt.NDArray[np.float64] = None,
#         n_gh_quad_points: int = _N_GH_QUAD_POINTS,
# ) -> BandProfile | pd.Series:
#     """
#     Compute the integrated BandProfiles for a set of transitions between the same upper and lower aggregated states.
#     The absorption profiles are only for the f<-i upward transitions, and the emission profiles are for downward f->i
#     spontaneous emission. Stimulated Emission profiles can be reconstructed from the spontaneous emission profiles.
#
#     :param wn_grid:          Spectral grid in wavenumbers.
#     :param n_frac_i:         Populations of the lower states as a fraction of the total aggregated lower states.
#     :param a_fi:             Einstein A coefficients.
#     :param g_f:              Upper state degeneracies.
#     :param g_i:              Lower state degeneracies.
#     :param energy_fi:        Transition energies in wavenumbers.
#     :param temperature:      Temperature of the atmospheric layer.
#     :param species_mass:     Mass of the species in Daltons.
#     :param n_frac_f:         Populations of the upper states as a fraction of the total aggregated upper states.
#     :param lifetimes:        Lower state lifetimes.
#     :param pressure:         Pressure of atmospheric layer.
#     :param broad_n:          Pressure broadening exponent.
#     :param broad_gamma:      Pressure broadening Lorentzian half-width half-maximum, gamma.
#     :param cont_broad:       Continuum broadening half-width half-maximum.
#     :param n_gh_quad_points: Number of Gauss-Hermite quadrature points.
#
#     :return: For continuum transitions, a BandProfile object is returned. For bound-bound transitions, returns a
#         pandas.Series object containing two BandProfile objects for the absorption and emission profiles respectively.
#     """
#     if energy_fi.min() > wn_grid.max() or energy_fi.max() < wn_grid.min():
#         raise RuntimeError("Computing band profile for band with no transitions on grid: check workflow logic.")
#     else:
#         # Handle transitions outside wn range by calculating their corresponding coefficients to scale normalisation.
#         # trans_outside_logic = (energy_fi < wn_grid.min()) | (energy_fi > wn_grid.max())
#         # trans_inside_logic = (energy_fi >= wn_grid.min()) & (energy_fi <= wn_grid.max())
#
#         is_fixed_width = np.all(np.isclose(np.diff(wn_grid), np.abs(wn_grid[1] - wn_grid[0]), atol=0))
#         if cont_broad is None:
#             if broad_n is None or broad_gamma is None:
#                 if is_fixed_width:
#                     # _abs_xsec, _emi_xsec = _abs_emi_binned_doppler_fixed_width(
#                     #     wn_grid=wn_grid,
#                     #     n_i=n_frac_i,
#                     #     n_f=n_frac_f,
#                     #     a_fi=a_fi,
#                     #     g_f=g_f,
#                     #     g_i=g_i,
#                     #     energy_fi=energy_fi,
#                     #     temperature=temperature,
#                     #     species_mass=species_mass,
#                     #     half_bin_width=np.abs(wn_grid[1] - wn_grid[0]) / 2.0,
#                     # )
#                     _abs_xsec, _emi_xsec = _band_profile_binned_doppler_fixed_width(
#                         wn_grid=wn_grid,
#                         n_i=n_frac_i,
#                         n_f=n_frac_f,
#                         a_fi=a_fi,
#                         g_f=g_f,
#                         g_i=g_i,
#                         energy_fi=energy_fi,
#                         temperature=temperature,
#                         species_mass=species_mass,
#                         half_bin_width=np.abs(wn_grid[1] - wn_grid[0]) / 2.0,
#                     )
#                 else:
#                     # _abs_xsec, _emi_xsec = _abs_emi_binned_doppler_variable_width(
#                     #     wn_grid=wn_grid,
#                     #     n_i=n_frac_i,
#                     #     n_f=n_frac_f,
#                     #     a_fi=a_fi,
#                     #     g_f=g_f,
#                     #     g_i=g_i,
#                     #     energy_fi=energy_fi,
#                     #     temperature=temperature,
#                     #     species_mass=species_mass,
#                     # )
#                     _abs_xsec, _emi_xsec = _band_profile_binned_doppler_variable_width(
#                         wn_grid=wn_grid,
#                         n_i=n_frac_i,
#                         n_f=n_frac_f,
#                         a_fi=a_fi,
#                         g_f=g_f,
#                         g_i=g_i,
#                         energy_fi=energy_fi,
#                         temperature=temperature,
#                         species_mass=species_mass,
#                     )
#             else:
#                 if n_frac_f is None or lifetimes is None or pressure is None or broad_n is None or broad_gamma is None:
#                     raise RuntimeError("Missing inputs for calc_band_profile when computing Voigt profiles.")
#                 gh_roots, gh_weights = roots_hermite(n_gh_quad_points)
#                 if is_fixed_width:
#                     # _abs_xsec, _emi_xsec = _abs_emi_binned_voigt_fixed_width(
#                     #     wn_grid=wn_grid,
#                     #     n_i=n_frac_i,
#                     #     n_f=n_frac_f,
#                     #     a_fi=a_fi,
#                     #     g_f=g_f,
#                     #     g_i=g_i,
#                     #     energy_fi=energy_fi,
#                     #     lifetimes=lifetimes,
#                     #     temperature=temperature,
#                     #     pressure=pressure,
#                     #     broad_n=broad_n,
#                     #     broad_gamma=broad_gamma,
#                     #     species_mass=species_mass,
#                     #     half_bin_width=np.abs(wn_grid[1] - wn_grid[0]) / 2.0,
#                     #     gh_roots=gh_roots,
#                     #     gh_weights=gh_weights,
#                     # )
#                     _abs_xsec, _emi_xsec = _band_profile_binned_voigt_fixed_width(
#                         wn_grid=wn_grid,
#                         n_i=n_frac_i,
#                         n_f=n_frac_f,
#                         a_fi=a_fi,
#                         g_f=g_f,
#                         g_i=g_i,
#                         energy_fi=energy_fi,
#                         lifetimes=lifetimes,
#                         temperature=temperature,
#                         pressure=pressure,
#                         broad_n=broad_n,
#                         broad_gamma=broad_gamma,
#                         species_mass=species_mass,
#                         half_bin_width=np.abs(wn_grid[1] - wn_grid[0]) / 2.0,
#                         gh_roots=gh_roots,
#                         gh_weights=gh_weights,
#                     )
#                 else:
#                     # _abs_xsec, _emi_xsec = _abs_emi_binned_voigt_variable_width(
#                     #     wn_grid=wn_grid,
#                     #     n_i=n_frac_i,
#                     #     n_f=n_frac_f,
#                     #     a_fi=a_fi,
#                     #     g_f=g_f,
#                     #     g_i=g_i,
#                     #     energy_fi=energy_fi,
#                     #     lifetimes=lifetimes,
#                     #     temperature=temperature,
#                     #     pressure=pressure,
#                     #     broad_n=broad_n,
#                     #     broad_gamma=broad_gamma,
#                     #     species_mass=species_mass,
#                     #     gh_roots=gh_roots,
#                     #     gh_weights=gh_weights,
#                     # )
#                     _abs_xsec, _emi_xsec = _band_profile_binned_voigt_variable_width(
#                         wn_grid=wn_grid,
#                         n_i=n_frac_i,
#                         n_f=n_frac_f,
#                         a_fi=a_fi,
#                         g_f=g_f,
#                         g_i=g_i,
#                         energy_fi=energy_fi,
#                         lifetimes=lifetimes,
#                         temperature=temperature,
#                         pressure=pressure,
#                         broad_n=broad_n,
#                         broad_gamma=broad_gamma,
#                         species_mass=species_mass,
#                         gh_roots=gh_roots,
#                         gh_weights=gh_weights,
#                     )
#             return pd.Series([BandProfile(profile=_abs_xsec), BandProfile(profile=_emi_xsec)], index=["abs", "emi"])
#         else:
#             # abs_coef_outside = _sum_abs_coefs(
#             #     n_i=n_frac_i[trans_outside_logic],
#             #     a_fi=a_fi[trans_outside_logic],
#             #     g_f=g_f[trans_outside_logic],
#             #     g_i=g_i[trans_outside_logic],
#             #     energy_fi=energy_fi[trans_outside_logic],
#             #     temperature=temperature,
#             # )
#             if is_fixed_width:
#                 _abs_xsec = _continuum_band_profile_fixed_width(
#                     wn_grid=wn_grid,
#                     n_frac_f=n_frac_f,
#                     n_frac_i=n_frac_i,
#                     a_fi=a_fi,
#                     g_f=g_f,
#                     g_i=g_i,
#                     energy_fi=energy_fi,
#                     temperature=temperature,
#                     cont_broad=cont_broad,
#                     half_bin_width=np.abs(wn_grid[1] - wn_grid[0]) / 2.0,
#                 )
#             else:
#                 _abs_xsec = _continuum_band_profile_variable_width(
#                     wn_grid=wn_grid,
#                     n_frac_f=n_frac_f,
#                     n_frac_i=n_frac_i,
#                     a_fi=a_fi,
#                     g_f=g_f,
#                     g_i=g_i,
#                     energy_fi=energy_fi,
#                     temperature=temperature,
#                     cont_broad=cont_broad,
#                 )
#             return BandProfile(profile=_abs_xsec)


@numba.njit(parallel=True)
def _continuum_band_profile_variable_width(
        wn_grid: npt.NDArray[np.float64],
        n_frac_f: npt.NDArray[np.float64],
        n_frac_i: npt.NDArray[np.float64],
        a_fi: npt.NDArray[np.float64],
        g_f: npt.NDArray[np.float64],
        g_i: npt.NDArray[np.float64],
        energy_fi: npt.NDArray[np.float64],
        cont_broad: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Note that the area of a Gaussian within 5 * HWHM is equal to erf(5*sqrt(ln(2))). The reciprocal of this can be used
    to recover the total intensity, though this value is 99.99999960685046% percent. Hence, this should only affect the
    9th decimal place, and Einstein A coefficients are only provided to 5 significant figures.

    Parameters
    ----------
    wn_grid
    n_frac_f
    n_frac_i
    a_fi
    g_f
    g_i
    energy_fi
    cont_broad

    Returns
    -------

    """
    assert n_frac_f.shape[0] == n_frac_i.shape[0] == a_fi.shape[0] == g_f.shape[0] == g_i.shape[0] == energy_fi.shape[
        0] == \
           cont_broad.shape[0]
    sqrtln2 = math.sqrt(math.log(2))
    num_grid = wn_grid.shape[0]
    _abs_xsec = np.zeros(num_grid, dtype=np.float64)
    num_trans = energy_fi.shape[0]
    # cutoff = 1500
    min_cutoff = 25.0
    max_cutoff = 3000.0
    cutoff_fwhm_multiple = 5.0

    # bin_widths = np.zeros(num_grid + 1)
    # bin_widths[1:-1] = (wn_grid[:-1] + wn_grid[1:]) / 2.0 - wn_grid[:-1]
    bin_edges = np.empty(num_grid + 1, dtype=np.float64)
    bin_edges[0] = wn_grid[0] - (wn_grid[1] - wn_grid[0]) * 0.5
    for j in range(1, num_grid):
        bin_edges[j] = (wn_grid[j - 1] + wn_grid[j]) * 0.5
    bin_edges[-1] = wn_grid[-1] + (wn_grid[-1] - wn_grid[-2]) * 0.5

    bin_widths_lower = wn_grid - bin_edges[:-1]
    bin_widths_upper = bin_edges[1:] - wn_grid
    inv_bin_widths = 1.0 / (bin_widths_lower + bin_widths_upper)

    abs_coef = a_fi * ((n_frac_i * g_f / g_i) - n_frac_f) / (const_16_pi_c * energy_fi * energy_fi)
    sqrtln2_on_alpha = sqrtln2 / cont_broad

    for i in numba.prange(num_trans):
        energy_fi_i = energy_fi[i]
        abs_coef_i = abs_coef[i]
        sqrtln2_on_alpha_i = sqrtln2_on_alpha[i]
        cont_broad_i = cont_broad[i]
        cutoff_i = cont_broad_i * cutoff_fwhm_multiple
        cutoff_i = max(min_cutoff, min(cutoff_i, max_cutoff))

        transition_min = energy_fi_i - cutoff_i
        transition_max = energy_fi_i + cutoff_i

        j_start = binary_search_right(bin_edges, transition_min) - 1
        j_start = max(0, j_start)
        j_end = binary_search_left(bin_edges, transition_max, start=j_start)
        j_end = min(num_grid, j_end)

        for j in range(j_start, j_end):
            wn_shift = wn_grid[j] - energy_fi_i
            # upper_width = bin_widths[j + 1]
            # lower_width = bin_widths[j]
            upper_width = bin_widths_upper[j]
            lower_width = bin_widths_lower[j]
            # if min(abs(wn_shift - lower_width), abs(wn_shift + upper_width)) <= cutoff:
            _abs_xsec[j] += (
                    abs_coef_i
                    * (
                            math.erf(sqrtln2_on_alpha_i * (wn_shift + upper_width))
                            - math.erf(sqrtln2_on_alpha_i * (wn_shift - lower_width))
                    ) * inv_bin_widths[j]
            )
    return _abs_xsec


@numba.njit(parallel=True)
def _continuum_band_profile_fixed_width(
        wn_grid: npt.NDArray[np.float64],
        n_frac_f: npt.NDArray[np.float64],
        n_frac_i: npt.NDArray[np.float64],
        a_fi: npt.NDArray[np.float64],
        g_f: npt.NDArray[np.float64],
        g_i: npt.NDArray[np.float64],
        energy_fi: npt.NDArray[np.float64],
        cont_broad: npt.NDArray[np.float64],
        half_bin_width: float,
) -> npt.NDArray[np.float64]:
    assert n_frac_f.shape[0] == n_frac_i.shape[0] == a_fi.shape[0] == g_f.shape[0] == g_i.shape[0] == energy_fi.shape[
        0] == \
           cont_broad.shape[0]
    twice_bin_width = 4.0 * half_bin_width
    sqrtln2 = math.sqrt(math.log(2))
    num_grid = wn_grid.shape[0]
    _abs_xsec = np.zeros(num_grid, dtype=np.float64)
    num_trans = energy_fi.shape[0]
    # cutoff = 1500 + half_bin_width
    min_cutoff = 25.0
    max_cutoff = 3000.0
    cutoff_fwhm_multiple = 5.0

    bin_edges = np.empty(num_grid + 1, dtype=np.float64)
    bin_edges[0] = wn_grid[0] - (wn_grid[1] - wn_grid[0]) * 0.5
    for j in range(1, num_grid):
        bin_edges[j] = (wn_grid[j - 1] + wn_grid[j]) * 0.5
    bin_edges[-1] = wn_grid[-1] + (wn_grid[-1] - wn_grid[-2]) * 0.5

    abs_coef = a_fi * ((n_frac_i * g_f / g_i) - n_frac_f) / (const_8_pi_c * twice_bin_width * energy_fi * energy_fi)
    sqrtln2_on_alpha = sqrtln2 / cont_broad

    for i in numba.prange(num_trans):
        energy_fi_i = energy_fi[i]
        abs_coef_i = abs_coef[i]
        sqrtln2_on_alpha_i = sqrtln2_on_alpha[i]
        cont_broad_i = cont_broad[i]
        cutoff_i = cont_broad_i * cutoff_fwhm_multiple
        cutoff_i = max(min_cutoff, min(cutoff_i, max_cutoff))

        transition_min = energy_fi_i - cutoff_i
        transition_max = energy_fi_i + cutoff_i

        j_start = binary_search_right(bin_edges, transition_min) - 1
        j_start = max(0, j_start)
        j_end = binary_search_left(bin_edges, transition_max, start=j_start)
        j_end = min(num_grid, j_end)

        for j in range(j_start, j_end):
            wn_shift = wn_grid[j] - energy_fi_i
            # if np.abs(wn_shift) <= cutoff:
            _abs_xsec[j] += abs_coef_i * (
                    math.erf(sqrtln2_on_alpha_i * (wn_shift + half_bin_width))
                    - math.erf(sqrtln2_on_alpha_i * (wn_shift - half_bin_width))
            )
    return _abs_xsec


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
    pl_states_c = continuum_states.select(
        pl.col("id"),
        pl.col("energy").alias("energy_c"),
        pl.col(n_col).alias("n_nlte_c"),
        pl.col("g").alias("g_c"),
    )

    dask_dtypes = {"id_c": "int64", "id_i": "int64", "A_ci": "float64", "broad": "float64"}
    dask_blocksize = "256MB"

    for trans_file in continuum_trans_files:
        ddf = dd.read_csv(
            trans_file,
            sep=r"\s+",
            engine="python",
            header=None,
            names=["id_c", "id_i", "A_ci", "broad"],
            usecols=[0, 1, 2, 3],
            dtype=dask_dtypes,
            blocksize=dask_blocksize,
        )
        delayed_batches = ddf.to_delayed()
        for delayed_batch in delayed_batches:
            trans_batch = pl.from_pandas(delayed_batch.compute())
            trans_batch = trans_batch.join(pl_states_i, left_on="id_i", right_on="id", how="left")
            trans_batch = trans_batch.join(pl_states_c, left_on="id_c", right_on="id", how="left")
            trans_batch = trans_batch.with_columns(
                (pl.col("energy_c") - pl.col("energy_i")).alias("energy_ci")
            )
            trans_batch = trans_batch.filter(
                (pl.col("energy_c") >= wn_min)
                & (pl.col("energy_i") <= wn_max)
                & (pl.col("energy_ci") >= wn_min)
                & (pl.col("energy_ci") <= wn_max)
            )
            final_cols = ["n_nlte_c", "n_nlte_i", "A_ci", "g_c", "g_i", "energy_ci", "broad"]
            trans_batch = trans_batch.select(final_cols)
            trans_chunk_np = trans_batch.to_numpy()

            # Matches final_cols ordering.
            n_c = np.ascontiguousarray(trans_chunk_np[:, 0])
            n_i = np.ascontiguousarray(trans_chunk_np[:, 1])
            a_ci = np.ascontiguousarray(trans_chunk_np[:, 2])
            g_c = np.ascontiguousarray(trans_chunk_np[:, 3])
            g_i = np.ascontiguousarray(trans_chunk_np[:, 4])
            energy_ci = np.ascontiguousarray(trans_chunk_np[:, 5])
            broad = np.ascontiguousarray(trans_chunk_np[:, 6])

            if is_fixed_width:
                half_bin_width = abs(wn_grid[1] - wn_grid[0]) / 2.0
                cont_xsec += _continuum_binned_gauss_fixed_width(
                    wn_grid=wn_grid,
                    n_f=n_c,
                    n_i=n_i,
                    a_fi=a_ci,
                    g_f=g_c,
                    g_i=g_i,
                    energy_fi=energy_ci,
                    cont_broad=broad,
                    half_bin_width=half_bin_width,
                )
            else:
                cont_xsec += _continuum_binned_gauss_variable_width(
                    wn_grid=wn_grid,
                    n_f=n_c,
                    n_i=n_i,
                    a_fi=a_ci,
                    g_f=g_c,
                    g_i=g_i,
                    energy_fi=energy_ci,
                    cont_broad=broad,
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
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    assert n_i.shape[0] == n_f.shape[0] == a_fi.shape[0] == g_f.shape[0] == g_i.shape[0] == energy_fi.shape[0] == \
           lifetimes.shape[0]
    assert broad_n.shape[0] == broad_gamma.shape[0]
    assert gh_roots.shape[0] == gh_weights.shape[0]

    num_grid = wn_grid.shape[0]
    _abs_xsec = np.zeros(num_grid, dtype=np.float64)
    _ste_xsec = np.zeros(num_grid, dtype=np.float64)
    _spe_xsec = np.zeros(num_grid, dtype=np.float64)
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

    for i in numba.prange(num_trans):
        energy_fi_i = energy_fi[i]
        abs_coef_i = abs_coef[i]
        ste_coef_i = ste_coef[i]
        spe_coef_i = spe_coef[i]
        gamma_inv_i = gamma_inv[i]

        gh_roots_sigma = gh_roots * sigma[i]
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
            # wn_shift = wn_grid[j] - energy_fi[i]
            upper_width = bin_widths_upper[j]
            lower_width = bin_widths_lower[j]
            # if -upper_width - cutoff <= wn_shift <= lower_width + cutoff:
            shift_sigma = wn_grid[j] - energy_fi[i] - gh_roots_sigma
            bin_term = np.sum(
                gh_weights_b_corr
                * (
                        np.arctan((shift_sigma + upper_width) * gamma_inv_i)
                        - np.arctan((shift_sigma - lower_width) * gamma_inv_i)
                )
            ) * inv_bin_widths[j]
            _abs_xsec[j] += abs_coef_i * bin_term
            _ste_xsec[j] += ste_coef_i * bin_term
            _spe_xsec[j] += spe_coef_i * bin_term
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
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    assert n_i.shape[0] == n_f.shape[0] == a_fi.shape[0] == g_f.shape[0] == g_i.shape[0] == energy_fi.shape[0] == \
           lifetimes.shape[0]
    assert broad_n.shape[0] == broad_gamma.shape[0]
    assert gh_roots.shape[0] == gh_weights.shape[0]

    bin_width = 2.0 * half_bin_width
    num_grid = wn_grid.shape[0]
    _abs_xsec = np.zeros(num_grid, dtype=np.float64)
    _emi_xsec = np.zeros(num_grid, dtype=np.float64)
    num_trans = energy_fi.shape[0]
    cutoff = 25 + half_bin_width

    bin_edges = np.empty(num_grid + 1, dtype=np.float64)
    bin_edges[0] = wn_grid[0] - (wn_grid[1] - wn_grid[0]) * 0.5
    for j in range(1, num_grid):
        bin_edges[j] = (wn_grid[j - 1] + wn_grid[j]) * 0.5
    bin_edges[-1] = wn_grid[-1] + (wn_grid[-1] - wn_grid[-2]) * 0.5

    abs_coef = a_fi * (n_i * g_f / g_i) / (const_8_pi_five_halves_c * bin_width * energy_fi * energy_fi)
    emi_coef = n_f * a_fi * energy_fi * const_h_c_on_4_pi_five_halves / bin_width
    sigma = energy_fi * const_sqrt_NA_kB_on_c * np.sqrt(temperature / species_mass)
    gamma_pressure = np.sum(broad_gamma * pressure * (t_ref / temperature) ** broad_n / pressure_ref)
    gamma_lifetime = 1 / (const_4_pi_c * lifetimes)
    gamma_total = gamma_lifetime + gamma_pressure
    gamma_inv = 1 / gamma_total

    grid_min = wn_grid[0]
    grid_max = wn_grid[-1]

    for i in numba.prange(num_trans):
        energy_fi_i = energy_fi[i]
        abs_coef_i = abs_coef[i]
        emi_coef_i = emi_coef[i]
        gamma_inv_i = gamma_inv[i]

        gh_roots_sigma = gh_roots * sigma[i]
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
            # wn_shift = wn_grid[j] - energy_fi[i]
            # if np.abs(wn_shift) <= cutoff:
            shift_sigma = wn_grid[j] - energy_fi_i - gh_roots_sigma
            bin_term = np.sum(
                gh_weights_b_corr
                * (
                        np.arctan((shift_sigma + half_bin_width) * gamma_inv_i)
                        - np.arctan((shift_sigma - half_bin_width) * gamma_inv_i)
                )
            )
            _abs_xsec[j] += abs_coef_i * bin_term
            _emi_xsec[j] += emi_coef_i * bin_term
    return _abs_xsec, _emi_xsec


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
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    assert n_i.shape[0] == n_f.shape[0] == a_fi.shape[0] == g_f.shape[0] == g_i.shape[0] == energy_fi.shape[0] == \
           lifetimes.shape[0]
    assert broad_n.shape[0] == broad_gamma.shape[0]
    assert gh_roots.shape[0] == gh_weights.shape[0]

    num_grid = wn_grid.shape[0]
    _abs_xsec = np.zeros(num_grid, dtype=np.float64)
    _emi_xsec = np.zeros(num_grid, dtype=np.float64)
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

    for i in numba.prange(num_trans):
        energy_fi_i = energy_fi[i]
        abs_coef_i = abs_coef[i]
        emi_coef_i = emi_coef[i]
        gamma_inv_i = gamma_inv[i]

        gh_roots_sigma = gh_roots * sigma[i]
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
            # wn_shift = wn_grid[j] - energy_fi[i]
            upper_width = bin_widths_upper[j]
            lower_width = bin_widths_lower[j]
            # if -upper_width - cutoff <= wn_shift <= lower_width + cutoff:
            shift_sigma = wn_grid[j] - energy_fi[i] - gh_roots_sigma
            bin_term = np.sum(
                gh_weights_b_corr
                * (
                        np.arctan((shift_sigma + upper_width) * gamma_inv_i)
                        - np.arctan((shift_sigma - lower_width) * gamma_inv_i)
                )
            ) * inv_bin_widths[j]
            _abs_xsec[j] += abs_coef_i * bin_term
            _emi_xsec[j] += emi_coef_i * bin_term
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
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    assert n_i.shape[0] == n_f.shape[0] == a_fi.shape[0] == g_f.shape[0] == g_i.shape[0] == energy_fi.shape[0] == \
           lifetimes.shape[0]
    assert broad_n.shape[0] == broad_gamma.shape[0]
    assert gh_roots.shape[0] == gh_weights.shape[0]

    bin_width = 2.0 * half_bin_width

    num_grid = wn_grid.shape[0]

    _abs_xsec = np.zeros(num_grid, dtype=np.float64)
    _emi_xsec = np.zeros(num_grid, dtype=np.float64)
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

    for i in numba.prange(num_trans):
        energy_fi_i = energy_fi[i]
        abs_coef_i = abs_coef[i]
        emi_coef_i = emi_coef[i]
        gamma_inv_i = gamma_inv[i]

        gh_roots_sigma = gh_roots * sigma[i]
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
            # wn_shift = wn_grid[j] - energy_fi_i
            # if np.abs(wn_shift) <= cutoff:
            shift_sigma = wn_grid[j] - energy_fi_i - gh_roots_sigma
            bin_term = np.sum(
                gh_weights_b_corr
                * (
                        np.arctan((shift_sigma + half_bin_width) * gamma_inv_i)
                        - np.arctan((shift_sigma - half_bin_width) * gamma_inv_i)
                )
            )
            _abs_xsec[j] += abs_coef_i * bin_term
            _emi_xsec[j] += emi_coef_i * bin_term
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
        cont_broad: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    assert n_i.shape[0] == n_f.shape[0] == a_fi.shape[0] == g_f.shape[0] == g_i.shape[0] == energy_fi.shape[0] == \
           cont_broad.shape[0]

    sqrtln2 = math.sqrt(math.log(2))
    num_grid = wn_grid.shape[0]
    _abs_xsec = np.zeros(num_grid, dtype=np.float64)
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
    sqrtln2_on_alpha = sqrtln2 / cont_broad

    for i in numba.prange(num_trans):
        energy_fi_i = energy_fi[i]
        abs_coef_i = abs_coef[i]
        sqrtln2_on_alpha_i = sqrtln2_on_alpha[i]
        cont_broad_i = cont_broad[i]
        cutoff_i = cont_broad_i * cutoff_fwhm_multiple
        cutoff_i = max(min_cutoff, min(cutoff_i, max_cutoff))

        transition_min = energy_fi_i - cutoff_i
        transition_max = energy_fi_i + cutoff_i

        j_start = binary_search_right(bin_edges, transition_min) - 1
        j_start = max(0, j_start)
        j_end = binary_search_left(bin_edges, transition_max, start=j_start)
        j_end = min(num_grid, j_end)

        for j in range(j_start, j_end):
            wn_shift = wn_grid[j] - energy_fi_i

            upper_width = bin_widths_upper[j]
            lower_width = bin_widths_lower[j]
            # if -upper_width - cutoff <= wn_shift <= lower_width + cutoff:
            _abs_xsec[j] += (
                    abs_coef_i
                    * (
                            math.erf(sqrtln2_on_alpha_i * (wn_shift + upper_width))
                            - math.erf(sqrtln2_on_alpha_i * (wn_shift - lower_width))
                    ) * inv_bin_widths[j]
            )
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
        cont_broad: npt.NDArray[np.float64],
        half_bin_width: np.float64,
) -> npt.NDArray[np.float64]:
    assert n_i.shape[0] == n_f.shape[0] == a_fi.shape[0] == g_f.shape[0] == g_i.shape[0] == energy_fi.shape[0] == \
           cont_broad.shape[0]

    twice_bin_width = 4.0 * half_bin_width
    sqrtln2 = math.sqrt(math.log(2))
    num_grid = wn_grid.shape[0]
    _abs_xsec = np.zeros(wn_grid.shape, dtype=np.float64)
    num_trans = energy_fi.shape[0]
    # cutoff = 1500 + half_bin_width
    min_cutoff = 25.0
    max_cutoff = 3000.0
    cutoff_fwhm_multiple = 5.0

    bin_edges = np.empty(num_grid + 1, dtype=np.float64)
    bin_edges[0] = wn_grid[0] - (wn_grid[1] - wn_grid[0]) * 0.5
    for j in range(1, num_grid):
        bin_edges[j] = (wn_grid[j - 1] + wn_grid[j]) * 0.5
    bin_edges[-1] = wn_grid[-1] + (wn_grid[-1] - wn_grid[-2]) * 0.5

    abs_coef = a_fi * ((n_i * g_f / g_i) - n_f) / (const_8_pi_c * twice_bin_width * energy_fi * energy_fi)
    sqrtln2_on_alpha = sqrtln2 / cont_broad

    for i in numba.prange(num_trans):
        energy_fi_i = energy_fi[i]
        abs_coef_i = abs_coef[i]
        sqrtln2_on_alpha_i = sqrtln2_on_alpha[i]
        cont_broad_i = cont_broad[i]
        cutoff_i = cont_broad_i * cutoff_fwhm_multiple
        cutoff_i = max(min_cutoff, min(cutoff_i, max_cutoff))

        transition_min = energy_fi_i - cutoff_i
        transition_max = energy_fi_i + cutoff_i

        j_start = binary_search_right(bin_edges, transition_min) - 1
        j_start = max(0, j_start)
        j_end = binary_search_left(bin_edges, transition_max, start=j_start)
        j_end = min(num_grid, j_end)

        for j in range(j_start, j_end):
            wn_shift = wn_grid[j] - energy_fi_i
            # if np.abs(wn_shift) <= cutoff:
            _abs_xsec[j] += abs_coef_i * (
                    math.erf(sqrtln2_on_alpha_i * (wn_shift + half_bin_width))
                    - math.erf(sqrtln2_on_alpha_i * (wn_shift - half_bin_width))
            )
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
                if coefficients[i, 3, j, k] > 0 > control_points[i, 0, j, k]:  # TODO: Check gamma index offset.
                    control_points[i, 0, j, k] = 0.0


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
                if i > 0 > control_points[i, 1, j, k] and coefficients[
                    i, 3, j, k] > 0:  # TODO: Check gamma index offset.
                    control_points[i, 1, j, k] = 0


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

    # TODO: Change indices on delta_tau_matrix based on direction!
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
        star_temperature: u.Quantity, star_logg: float, star_feh: float, wn_grid: u.Quantity,
        planet_radius: u.Quantity, orbital_radius: u.Quantity, star_alpha: float = 0.0
) -> u.Quantity:
    srf_wavelength, srf_flux = get_spectrum(
        teff=star_temperature,
        logg=star_logg,
        feh=star_feh,
        alpha=star_alpha,
        source="synphot",
    )
    srf_wn = srf_wavelength.to(1 / u.cm, equivalencies=u.spectral())
    srf_flux = srf_flux / (ac.c * srf_wn * srf_wn)
    srf_flux = srf_flux.to(u.J / u.m ** 2, equivalencies=u.spectral())

    sort_idx = np.argsort(srf_wn)
    srf_wn = srf_wn[sort_idx]
    srf_flux = srf_flux[sort_idx]

    srf_flux_interp = np.interp(srf_wn, wn_grid, srf_flux) << srf_flux.unit
    return srf_flux_interp * (planet_radius / orbital_radius) ** 2


@numba.njit()
def calc_einstein_b_fi(a_fi: npt.NDArray[np.float64], energy_fi: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Here the Einstein B coefficient is given in :math:`\text{cm}^3 / (\text{J·s·m})`.

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


# def cdf_opacity_sampling(
#         wn_start: float,
#         wn_end: float,
#         temperature_profile: npt.NDArray[np.float64],
#         num_points: int,
#         max_step: float = None,
#         num_cdf_points: int = 1000000,
# ) -> u.Quantity:
#     temp_wn_grid = np.linspace(wn_start, wn_end, num_cdf_points)
#     ev_grid = calc_ev_grid(wn_grid=temp_wn_grid, temperature=np.atleast_1d(temperature_profile)[:, None]).sum(axis=0)
#     ev_norm = ev_grid / simpson(ev_grid, x=temp_wn_grid)
#
#     ev_cdf = cumulative_simpson(ev_norm, x=temp_wn_grid, initial=0)
#
#     # sample_idxs = np.searchsorted(ev_cdf, np.linspace(0, 1, num_points))
#     sample_idxs = np.array(
#         [
#             np.argmin(abs(ev_cdf - point)) if point <= ev_cdf[-1] else len(ev_cdf) - 1
#             for point in np.linspace(0, 1, num_points)
#         ]
#     )
#     sample_idxs = np.unique(sample_idxs)
#     # Both methods allow for the same wn point to be the closest to multiple values on the CDF, so remove duplicates.
#     # The argmin approach should be better when using a small number of points but is marginally slower.
#
#     if sample_idxs[-1] == len(ev_cdf):
#         sample_idxs[-1] -= 1
#     if max_step:
#         max_idx_step = int(np.ceil(max_step * num_cdf_points / (wn_end - wn_start))) - 1
#         idx_diffs = np.diff(sample_idxs)
#         idxs_diffs_over_max = np.nonzero(idx_diffs > max_idx_step)[0]
#         idxs_diffs_over_max_chunks = np.split(idxs_diffs_over_max, np.where(np.diff(idxs_diffs_over_max) != 1)[0] + 1)
#         chunk_idx = 0
#         while chunk_idx < len(idxs_diffs_over_max_chunks):
#             chunk = idxs_diffs_over_max_chunks[chunk_idx]
#             end_idx = chunk[-1] + 1 if chunk[-1] + 1 < len(sample_idxs) else chunk[-1]
#             n_new_points = int(np.ceil((sample_idxs[end_idx] - sample_idxs[chunk[0]]) / max_idx_step)) + 1
#             insert_vals = np.linspace(sample_idxs[chunk[0]], sample_idxs[end_idx], n_new_points, dtype=int)
#             sample_idxs = np.concatenate((sample_idxs[: chunk[0]], insert_vals, sample_idxs[chunk[-1] + 2:]))
#             idxs_diffs_over_max_chunks = list(
#                 idx_chunk + n_new_points - len(chunk) - 1 for idx_chunk in idxs_diffs_over_max_chunks
#             )
#             chunk_idx += 1
#
#     sampled_grid = temp_wn_grid[sample_idxs]
#     return sampled_grid << u.k


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
    temp_wn_grid = np.linspace(wn_start, wn_end, num_cdf_points)
    ev_grid = calc_ev_grid(wn_grid=temp_wn_grid, temperature=np.atleast_1d(temperature_profile)[:, None]).sum(axis=0)
    ev_norm = ev_grid / simpson(ev_grid, x=temp_wn_grid)

    ev_cdf = cumulative_simpson(ev_norm, x=temp_wn_grid, initial=0)

    sample_idxs = _sample_indices(ev_cdf, temp_wn_grid, num_points, max_step)

    return temp_wn_grid[sample_idxs] / u.cm


class NLTEWorkflow(abc.ABC):
    @abc.abstractmethod
    def workflow(self) -> t.Any:
        pass


class GaussSeidelWorkflow(NLTEWorkflow):

    def workflow(self):
        pass
