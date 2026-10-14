"""Compact storage of per-layer absorption/emission profiles (CompactProfile, ProfileStore,
ContinuumProfileStore) and the kernels that rebuild cross-sections from them."""
import logging
import pathlib
import typing as t

import numba
import numpy as np
from astropy import units as u
from numpy import typing as npt

from ..config import _DEFAULT_NUM_THREADS, _INTENSITY_CUTOFF, _LOG_VERBOSE_2
from ..numerics import loglinear_normalise_2d_nonnegative

log = logging.getLogger(__name__)


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _analyse_profiles(
        profile_matrix_2d: npt.NDArray[np.float64], cutoff: float
) -> t.Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.bool_]]:
    # profile_matrix_2d: (n_profiles, n_grid) - caller reshapes
    n_profiles, n_grid = profile_matrix_2d.shape
    start_idxs = np.empty(n_profiles, dtype=np.int32)
    end_idxs = np.empty(n_profiles, dtype=np.int32)
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
        start_idxs: npt.NDArray[np.int32],
        end_idxs: npt.NDArray[np.int32],
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
        start_idxs: npt.NDArray[np.int32],
        key_idx_array: npt.NDArray[np.int32],
        pop_matrix: npt.NDArray[np.float64],
        wn_grid_len: int,
        is_abs: bool,
        numba_num_threads: int = _DEFAULT_NUM_THREADS,
) -> npt.NDArray[np.float64]:
    """
    For use only by :class:`tiramisu.profiles.store.CompactProfile` instances.

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

    # Profiles are dealt round-robin to numba_num_threads workers, each accumulating into its own buffer row. Not a
    # buffer per thread (numba.get_thread_id()): TBB hands out prange chunks dynamically, so which profiles a thread's
    # row summed changed from call to call, and with it the floating-point rounding of the result (up to ~1e-16
    # relative per call, amplified by the NLTE iteration). The round-robin split makes it reproducible, at the same
    # speed (tests/opacity/check_build_xsec_determinism.py).
    for w in numba.prange(numba_num_threads):
        for i in range(w, n_profiles, numba_num_threads):
            # Absorption depends on ID_l population; Emission depends on ID_u population.
            if is_abs:
                pop_val = pop_matrix[key_idx_array[i, 1]]
            else:
                pop_val = pop_matrix[key_idx_array[i, 0]]

            offset_start = offsets[i]
            offset_end = offsets[i + 1]
            profile_len = offset_end - offset_start
            start_idx = start_idxs[i]

            # Element-wise accumulation into the worker's buffer row.
            for j in range(profile_len):
                buffers[w, start_idx + j] += pop_val * profiles[offset_start + j]

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
        key_idx_map: npt.NDArray[np.int32],
        profiles: npt.NDArray[np.float64],
        offsets: npt.NDArray[np.int64],
        start_idxs: npt.NDArray[np.int32],
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


class CompactProfile:
    __slots__ = ["profiles", "offsets", "start_idxs", "key_idx_map", "key_lookup"]

    def __init__(self):
        self.profiles: npt.NDArray[np.float64] | None = None
        # int64: offsets index the concatenated profiles, whose total length (bands x grid points) can exceed 2^31.
        self.offsets: npt.NDArray[np.int64] | None = None
        self.start_idxs: npt.NDArray[np.int32] | None = None
        self.key_idx_map: npt.NDArray[np.int32] | None = None
        self.key_lookup: t.Dict[t.Tuple[int, int], int] = {}

    def finalise_from_buffer(self, profile_matrix: npt.NDArray[np.float64], keys: npt.NDArray[np.int32]):
        """
        Finalise a set of band profiles stored in the compact super-line representation.

        The starting indices within self.profiles of each individual profile is contained within self.offsets. An extra
        terminator is stored at the end of self.offsets equal to the total length of self.profiles; this is so the start
        and end indices of a given profile can always be obtained by looking at the current and next offset.

        The starting position of each profile on the main wavenumber grid is stored at the corresponding index in
        self.start_idxs.

        The upper and lower state IDs are stored in self.key_idx_map; the index of the first dimension of this array
        matches the corresponding index in self.offsets and self.start_idxs. This is used for fast cross-section
        reconstruction in :func:`tiramisu.profiles.store.CompactProfile.build_xsec`. A fast dictionary lookup for
        accessing individual bands is stored in self.key_lookup, used by
        :func:`tiramisu.profiles.store.CompactProfile.get_profile`.

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
        self.start_idxs = np.empty(n_valid, dtype=np.int32)
        self.key_idx_map = np.empty((n_valid, 2), dtype=np.int32)
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
        if log.isEnabledFor(_LOG_VERBOSE_2):
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
        # finalise_from_buffer builds every array contiguous with the kernel's dtype.
        return _build_xsec(
            profiles=self.profiles, offsets=self.offsets, start_idxs=self.start_idxs, key_idx_array=self.key_idx_map,
            pop_matrix=pop_matrix, wn_grid_len=wn_grid.shape[0], is_abs=is_abs,
        )


class ProfileStore:
    """
    Store :class:`tiramisu.profiles.store.CompactProfile` objects representing the species' absorption, stimulated
    emission and spontaneous emission profiles. Used for bound-bound transitions (including quasi-bound).
    """
    __slots__ = ["n_layers", "abs_profiles", "ste_profiles", "spe_profiles", "_profile_cache"]

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        # Storage for final profiles.
        self.abs_profiles = [CompactProfile() for _ in range(n_layers)]  # Absorption
        self.ste_profiles = [CompactProfile() for _ in range(n_layers)]  # Stimulated Emission
        self.spe_profiles = [CompactProfile() for _ in range(n_layers)]  # Spontaneous Emission
        # Instance-level (not class-level lru_cache!) cache for get_profile: an lru_cache on an instance method
        # binds self into the cache key, which both pins every ProfileStore instance in memory forever and
        # shares one global-size cache across all instances/species/iterations.
        self._profile_cache: t.Dict[t.Tuple[int, t.Tuple[int, int], str], t.Tuple[npt.NDArray[np.float64], int]] = {}

    def finalise_from_buffer(
            self,
            profile_buffer: npt.NDArray[np.float64],
            band_keys: npt.NDArray[np.int32],
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
        if save:
            export_dir.mkdir(parents=True, exist_ok=True)
        #####
        for l in range(self.n_layers):
            for channel, (label, stores) in enumerate(
                    (("abs", self.abs_profiles), ("ste", self.ste_profiles), ("spe", self.spe_profiles))
            ):
                store = stores[l]
                store.finalise_from_buffer(profile_matrix=profile_buffer[channel, :, l, :], keys=band_keys)
                if save:
                    np.savez_compressed(
                        export_dir / f"{species}_L{l:03d}_{label}.npz",
                        profiles=store.profiles,  # flat 1D array of all profile data
                        offsets=store.offsets,  # ragged array boundaries
                        start_idxs=store.start_idxs,  # position of each profile on wn_grid
                        key_idx_map=store.key_idx_map,
                    )

    def get_profile(
            self, layer_idx: int, key: t.Tuple[int, int], profile_type: str
    ) -> t.Tuple[npt.NDArray[np.float64], int]:
        cache_key = (layer_idx, key, profile_type)
        if cache_key in self._profile_cache:
            return self._profile_cache[cache_key]

        if profile_type == "abs":
            result = self.abs_profiles[layer_idx].get_profile(key)
        elif profile_type == "ste":
            result = self.ste_profiles[layer_idx].get_profile(key)
        elif profile_type == "spe":
            result = self.spe_profiles[layer_idx].get_profile(key)
        else:
            raise RuntimeError(f"ProfileStore profile type {profile_type} not implemented.")

        self._profile_cache[cache_key] = result
        return result

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
        spe = self.spe_profiles[layer_idx]
        # Summed over lower states in one pass: row o_idx is the emission from upper state o_idx.
        all_profiles = _rebuild_all_ox_profiles(
            id_agg_cutoff=id_agg_cutoff, key_idx_map=spe.key_idx_map, profiles=spe.profiles, offsets=spe.offsets,
            start_idxs=spe.start_idxs, num_grid=wn_grid.shape[0],
        )
        wn_dx = np.diff(wn_grid.value)
        # Normalised as photon-counting weights: emission profiles are energy rates (h nu per photon), while the cross
        # terms they feed count transitions, so divide by nu first (see "Photon-counting weights" in nlte.py).
        photon_profiles = all_profiles / wn_grid.value[None, :]
        return loglinear_normalise_2d_nonnegative(y_data=photon_profiles, dx=wn_dx) << 1 / wn_grid.unit

    def get_sorted_band_keys(
            self, agg_energies: npt.NDArray[np.float64], id_cutoff: int = 0,
    ) -> t.List[t.Tuple[int, int]]:
        """
        Returns the absorption band keys (id_agg_f, id_agg_i) present in every layer, sorted by energy gap
        (id_agg_f - id_agg_i) ascending, then by id_agg_i, then by id_agg_f.

        Parameters
        ----------
        agg_energies : ndarray
            Array containing the energies for each aggregate state ordered on id_agg (ascending).
        id_cutoff : int
            ID above which states are fixed, so key pairs containing IDs above this should be excluded. If not provided,
            returned keys include band keys involving the cutoff state.

        Returns
        -------
        List of (id_agg_f, id_agg_i) tuples in the above order.
        """
        key_sets = [set(compact.key_lookup.keys()) for compact in self.abs_profiles]
        common_keys = key_sets[0].intersection(*key_sets[1:])
        if id_cutoff > 0:
            common_keys = [k for k in common_keys if id_cutoff >= k[0] > k[1] >= 0]
        else:
            common_keys = [k for k in common_keys if k[0] > k[1] >= 0]

        if not common_keys:
            return []

        # Vectorized equivalent of sorting by (e_i, nu) with a per-comparison Python closure: gather id_f/id_i via
        # fancy indexing and let np.lexsort do the ordering (primary key last: e_i, then nu) instead.
        common_keys_arr = np.array(common_keys, dtype=np.int32)
        id_f_arr = common_keys_arr[:, 0]
        id_i_arr = common_keys_arr[:, 1]
        e_i_arr = agg_energies[id_i_arr]  # lower state energy
        e_f_arr = agg_energies[id_f_arr]  # upper state energy
        nu_arr = e_f_arr - e_i_arr  # band centre frequency

        order = np.lexsort((nu_arr, e_i_arr))
        return [(int(id_f_arr[i]), int(id_i_arr[i])) for i in order]


class ContinuumProfileStore:
    __slots__ = [
        "n_layers", "abs_profiles",
        # "ste_profiles", "spe_profiles"
        "_profile_cache",
    ]

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        # Storage for final profiles.
        self.abs_profiles = [CompactProfile() for _ in range(n_layers)]  # Absorption
        # Instance-level cache (see ProfileStore.get_profile for why not a class-level lru_cache).
        self._profile_cache: t.Dict[t.Tuple[int, int, str], t.Tuple[npt.NDArray, int]] = {}

    def finalise_from_buffer(self, profile_buffer: npt.NDArray[np.float64], band_keys: npt.NDArray[np.int32]) -> None:
        """

        Parameters
        ----------
        profile_buffer : ndarray, shape (1, n_bands, n_layers, n_grid)
        band_keys : ndarray, shape (n_bands, 2)
            Contains the (id_f_agg, id_i_agg) pairs for each band.
        """
        for l in range(self.n_layers):
            self.abs_profiles[l].finalise_from_buffer(profile_matrix=profile_buffer[0, :, l, :], keys=band_keys)

    def get_profile(self, layer_idx: int, key: int, profile_type: str) -> t.Tuple[npt.NDArray, int]:
        cache_key = (layer_idx, key, profile_type)
        if cache_key in self._profile_cache:
            return self._profile_cache[cache_key]

        if profile_type == "abs":
            # Continuum profiles are stored with an arbitrary index for the upper continuum state (-1).
            result = self.abs_profiles[layer_idx].get_profile((-1, key))
            self._profile_cache[cache_key] = result
            return result
        else:
            raise RuntimeError(f"ContinuumProfileStore profile type {profile_type} not implemented.")

    def build_abs(
            self, layer_idx: int, pop_matrix: npt.NDArray[np.float64], wn_grid: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        abs_profile = self.abs_profiles[layer_idx].build_xsec(pop_matrix=pop_matrix, wn_grid=wn_grid, is_abs=True)
        return abs_profile
