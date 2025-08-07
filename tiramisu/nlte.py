import pathlib
import typing as t
import numpy.typing as npt
import numpy as np
import pandas as pd
import numba
import math

from astropy import units as u
from astropy import constants as ac

from scipy.integrate import simpson, cumulative_simpson
from scipy.special import roots_hermite

from .config import log, _DEFAULT_NUM_THREADS, _DEFAULT_CHUNK_SIZE, _N_GH_QUAD_POINTS, _INTENSITY_CUTOFF

# Constants with units:
ac_h_c_on_kB = ac.h * ac.c.cgs / ac.k_B
ac_2_hc = 2 * ac.h * ac.c.cgs

ac_4_pi_c = 4 * np.pi * ac.c.cgs
ac_8_pi_c = 2 * ac_4_pi_c
ac_8_pi_five_halves_c = ac_8_pi_c * (np.pi**1.5)
ac_16_pi_c = 2 * ac_8_pi_c

ac_h_c_on_4_pi = ac.h.cgs * ac.c.cgs / (4 * np.pi)
ac_h_c_on_4_pi_five_halves = ac_h_c_on_4_pi / (np.pi**1.5)
ac_h_c_on_8_pi = ac_h_c_on_4_pi / 2.0

ac_sqrt_NA_kB_on_c = (np.sqrt(ac.N_A * ac.k_B.cgs) / ac.c.cgs).to(
    u.kg**0.5 / (u.K**0.5 * u.mol**0.5), equivalencies=u.spectral()
)
ac_sqrt_2_NA_kB_log2_on_c = ac_sqrt_NA_kB_on_c * np.sqrt(2 * np.log(2))

ac_2_h_on_c_sq = 2 * ac.h / ac.c**2
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
    (2 * np.pi * ac.h * ac.c.cgs**2 / ac.sigma_sb).to(u.K**4 * u.cm**4, equivalencies=u.spectral()).value
)
const_2_pi_c_kB = (2 * np.pi * ac.c.cgs * ac.k_B.cgs).value


# TODO: Handles NANs in state lifetimes; treat as inf?


class BandProfile:
    __slots__ = ("start_idx", "profile", "integral")

    def __init__(self, profile: npt.NDArray[np.float64], start_idx: int = None, trim: bool = True) -> None:
        """

        :param start_idx: Index on the spectroscopic grid where the trimmed band profile begins.
        :param profile:   The band profile of the transition.
        integral:         The integral of the absorption profile pre-normalisation.
        """
        self.integral = 0.0
        if not trim:
            self.start_idx = 0
            self.profile = profile
        elif start_idx is None:
            if np.all(profile < _INTENSITY_CUTOFF):
                log.warning(f"All of something is below the cutoff! len={len(self.profile)}")
                self.start_idx = 0
                self.profile = np.empty(0)
            else:
                self.start_idx = np.argmax(profile >= _INTENSITY_CUTOFF)
                end_idx = len(profile) - np.argmax(profile[::-1] >= _INTENSITY_CUTOFF)
                self.profile = profile[self.start_idx : end_idx]
        else:
            self.start_idx = start_idx
            self.profile = profile

    def __repr__(self):
        return f"BandProfile([{self.start_idx}, {self.profile}, {self.integral}])"

    def __str__(self):
        return (
            f"BandProfile(start_idx: {self.start_idx}, profile: "
            f"{self.profile}"
            f"({self.profile.size} points),"
            f" integral: {self.integral},"
            ")"
        )

    def merge_band_profiles(
        self, band_profiles: t.List["BandProfile"], normalise: bool = False, spectral_grid: u.Quantity = None
    ) -> None:
        # TODO: Handle case where some band_profiles are empty?
        if len(band_profiles) > 0:
            start_idxs = np.concatenate(([self.start_idx], [band_profile.start_idx for band_profile in band_profiles]))
            profiles = [self.profile] + [band_profile.profile for band_profile in band_profiles]

            min_start_idx = min(start_idxs)
            primary_idx = np.argmax(start_idxs == min_start_idx)
            max_end_idx = max(
                np.concatenate(
                    (
                        [self.start_idx + len(self.profile)],
                        [band_profile.start_idx + len(band_profile.profile) for band_profile in band_profiles],
                    )
                )
            )
            offset = max_end_idx - min_start_idx - len(profiles[primary_idx])

            merged_profile = np.pad(profiles[primary_idx], (0, offset), "constant")

            for profile_idx in range(len(start_idxs)):
                if profile_idx != primary_idx:
                    profile_offset = start_idxs[profile_idx] - min_start_idx
                    merged_profile[profile_offset : profile_offset + len(profiles[profile_idx])] += profiles[
                        profile_idx
                    ]

            self.start_idx = min_start_idx
            self.profile = merged_profile
        if normalise:
            if spectral_grid is None:
                raise RuntimeError("Normalisation specified but no wn_grid provided for integration.")
            self.normalise_band_profile(spectral_grid=spectral_grid)

    def normalise_band_profile(self, spectral_grid: u.Quantity) -> None:
        if self.profile.size == 0:
            pass
        else:
            if len(self.profile) == 1 and sum(self.profile) != 0:
                self.integral = self.profile.sum()
            else:
                self.integral = simpson(
                    self.profile,
                    x=spectral_grid[self.start_idx : self.start_idx + len(self.profile)].value,
                )
            if self.integral == 0:
                raise RuntimeError("Abs factor is 0 - Why?")
            self.profile /= self.integral


class BandProfileCollection(dict):
    def __init__(self, band_profiles: npt.NDArray[BandProfile] | t.List[BandProfile] | pd.Series):
        if type(band_profiles) is pd.Series:
            for row_key, row in band_profiles.items():
                if row.profile.size > 0:
                    if row_key in self:
                        self[row_key].merge_band_profiles(row)
                    else:
                        self[row_key] = row
                else:
                    log.info(f"BandProfile for key={row_key} is empty.")
        # elif type(band_profiles) in (npt.NDArray[BandProfile], t.List[BandProfile]):
        #     keys = [(band_profile.id_u, band_profile.id_l) for band_profile in band_profiles]
        #     unique_keys = set(keys)
        #     for unique_key in unique_keys:
        #         key_idxs = [key_idx for key_idx, key in enumerate(keys) if key == unique_key]
        #         key_profiles = band_profiles[key_idxs]
        #         self[unique_key] = key_profiles[0]
        #         if len(key_profiles) > 1:
        #             self[unique_key].merge_band_profiles(band_profiles=key_profiles[1:])
        else:
            raise RuntimeError(
                "BandProfileCollection construction only implemented for list, np.array or pd.Series."
                f"Received {type(band_profiles)}."
            )
        super().__init__()

    def __getitem__(self, key: t.Tuple[int, int] | int) -> BandProfile:
        return super().__getitem__(key)

    def get(self, key: t.Tuple[int, int] | int, default: t.Optional[t.Any] = None) -> BandProfile:
        return super().get(key)

    def __setitem__(self, key: t.Tuple[int, int] | int, value: BandProfile) -> None:
        return super().__setitem__(key, value)

    def __contains__(self, key: t.Tuple[int, int] | int) -> bool:
        return super().__contains__(key)

    def __delitem__(self, key: t.Tuple[int, int] | int) -> None:
        return super().__delitem__(key)

    def merge_collections(
        self,
        band_profile_collections: t.List["BandProfileCollection"] | npt.NDArray["BandProfileCollection"],
        normalise: bool = False,
        spectral_grid: u.Quantity = None,
    ) -> None:
        keys = [band_profile_collection.keys() for band_profile_collection in band_profile_collections]
        keys = [key for sublist in keys for key in sublist]
        unique_keys = set(keys)
        for unique_key in unique_keys:
            key_profiles = [
                band_profile_collection.get(unique_key) for band_profile_collection in band_profile_collections
            ]
            key_profiles = [profile for profile in key_profiles if profile is not None]
            if unique_key in self:
                self[unique_key].merge_band_profiles(
                    band_profiles=key_profiles, normalise=normalise, spectral_grid=spectral_grid
                )
            else:
                self[unique_key] = key_profiles[0]
                if len(key_profiles) > 1:
                    self[unique_key].merge_band_profiles(
                        band_profiles=key_profiles[1:], normalise=normalise, spectral_grid=spectral_grid
                    )
                else:
                    self[unique_key].normalise_band_profile(spectral_grid=spectral_grid)

    def normalise(self, spectral_grid: u.Quantity, sanitise: bool = True) -> None:
        for key in list(self.keys()):
            self[key].normalise_band_profile(spectral_grid=spectral_grid)
            if sanitise and self[key].profile.size == 0:
                del self[key]

    def add_no_trim(self, key: t.Tuple[int, int] | int, profile: npt.NDArray[np.float64]):
        self[key] = BandProfile(profile=profile, trim=False)


def calc_band_profile(
    wn_grid: npt.NDArray[np.float64],
    n_frac_i: npt.NDArray[np.float64],
    a_fi: npt.NDArray[np.float64],
    g_f: npt.NDArray[np.float64],
    g_i: npt.NDArray[np.float64],
    energy_fi: npt.NDArray[np.float64],
    temperature: float,
    species_mass: float,
    n_frac_f: npt.NDArray[np.float64] = None,
    lifetimes: npt.NDArray[np.float64] = None,
    pressure: float = None,
    broad_n: npt.NDArray[np.float64] = None,
    broad_gamma: npt.NDArray[np.float64] = None,
    cont_broad: npt.NDArray[np.float64] = None,
    n_gh_quad_points: int = _N_GH_QUAD_POINTS,
) -> BandProfile | pd.Series:
    if energy_fi.min() > wn_grid.max() or energy_fi.max() < wn_grid.min():
        raise RuntimeError("Computing band profile for band with no transitions on grid: check workflow logic.")
    else:
        # Handle transitions outside wn range by calculating their corresponding coefficients to scale normalisation.
        # trans_outside_logic = (energy_fi < wn_grid.min()) | (energy_fi > wn_grid.max())
        # trans_inside_logic = (energy_fi >= wn_grid.min()) & (energy_fi <= wn_grid.max())

        is_fixed_width = np.all(np.isclose(np.diff(wn_grid), np.abs(wn_grid[1] - wn_grid[0]), atol=0))
        if cont_broad is None:
            # abs_coef_outside, emi_coef_outside = _sum_abs_emi_coefs(
            #     n_i=n_frac_i[trans_outside_logic],
            #     n_f=n_frac_f[trans_outside_logic],
            #     a_fi=a_fi[trans_outside_logic],
            #     g_f=g_f[trans_outside_logic],
            #     g_i=g_i[trans_outside_logic],
            #     energy_fi=energy_fi[trans_outside_logic],
            #     temperature=temperature,
            # )
            if broad_n is None or broad_gamma is None:
                if is_fixed_width:
                    _abs_xsec, _emi_xsec = _abs_emi_binned_doppler_fixed_width(
                        wn_grid=wn_grid,
                        n_i=n_frac_i,
                        n_f=n_frac_f,
                        a_fi=a_fi,
                        g_f=g_f,
                        g_i=g_i,
                        energy_fi=energy_fi,
                        temperature=temperature,
                        species_mass=species_mass,
                        half_bin_width=np.abs(wn_grid[1] - wn_grid[0]) / 2.0,
                    )
                else:
                    _abs_xsec, _emi_xsec = _abs_emi_binned_doppler_variable_width(
                        wn_grid=wn_grid,
                        n_i=n_frac_i,
                        n_f=n_frac_f,
                        a_fi=a_fi,
                        g_f=g_f,
                        g_i=g_i,
                        energy_fi=energy_fi,
                        temperature=temperature,
                        species_mass=species_mass,
                    )
            else:
                if n_frac_f is None or lifetimes is None or pressure is None or broad_n is None or broad_gamma is None:
                    raise RuntimeError("Missing inputs for calc_band_profile when computing Voigt profiles.")
                gh_roots, gh_weights = roots_hermite(n_gh_quad_points)
                if is_fixed_width:
                    _abs_xsec, _emi_xsec = _abs_emi_binned_voigt_fixed_width(
                        wn_grid=wn_grid,
                        n_i=n_frac_i,
                        n_f=n_frac_f,
                        a_fi=a_fi,
                        g_f=g_f,
                        g_i=g_i,
                        energy_fi=energy_fi,
                        lifetimes=lifetimes,
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
                    # _abs_xsec, _emi_xsec = _abs_emi_binned_voigt_variable_width(
                    #     wn_grid=wn_grid,
                    #     n_i=n_frac_i,
                    #     n_f=n_frac_f,
                    #     a_fi=a_fi,
                    #     g_f=g_f,
                    #     g_i=g_i,
                    #     energy_fi=energy_fi,
                    #     lifetimes=lifetimes,
                    #     temperature=temperature,
                    #     pressure=pressure,
                    #     broad_n=broad_n,
                    #     broad_gamma=broad_gamma,
                    #     species_mass=species_mass,
                    #     gh_roots=gh_roots,
                    #     gh_weights=gh_weights,
                    # )
                    # TODO: UPDATE THE REST MUST DOOOO!
                    _abs_xsec, _emi_xsec = _band_profile_binned_voigt_variable_width(
                        wn_grid=wn_grid,
                        n_i=n_frac_i,
                        n_f=n_frac_f,
                        a_fi=a_fi,
                        g_f=g_f,
                        g_i=g_i,
                        energy_fi=energy_fi,
                        lifetimes=lifetimes,
                        temperature=temperature,
                        pressure=pressure,
                        broad_n=broad_n,
                        broad_gamma=broad_gamma,
                        species_mass=species_mass,
                        gh_roots=gh_roots,
                        gh_weights=gh_weights,
                    )
            return pd.Series([BandProfile(profile=_abs_xsec), BandProfile(profile=_emi_xsec)], index=["abs", "emi"])
        else:
            # abs_coef_outside = _sum_abs_coefs(
            #     n_i=n_frac_i[trans_outside_logic],
            #     a_fi=a_fi[trans_outside_logic],
            #     g_f=g_f[trans_outside_logic],
            #     g_i=g_i[trans_outside_logic],
            #     energy_fi=energy_fi[trans_outside_logic],
            #     temperature=temperature,
            # )
            if is_fixed_width:
                _abs_xsec = _continuum_band_profile_fixed_width(
                    wn_grid=wn_grid,
                    n_frac_f=n_frac_f,
                    n_frac_i=n_frac_i,
                    a_fi=a_fi,
                    g_f=g_f,
                    g_i=g_i,
                    energy_fi=energy_fi,
                    temperature=temperature,
                    cont_broad=cont_broad,
                    half_bin_width=np.abs(wn_grid[1] - wn_grid[0]) / 2.0,
                )
            else:
                _abs_xsec = _continuum_band_profile_variable_width(
                    wn_grid=wn_grid,
                    n_frac_f=n_frac_f,
                    n_frac_i=n_frac_i,
                    a_fi=a_fi,
                    g_f=g_f,
                    g_i=g_i,
                    energy_fi=energy_fi,
                    temperature=temperature,
                    cont_broad=cont_broad,
                )
            return BandProfile(profile=_abs_xsec)



def calc_lambda_approx(
    band_profile: BandProfile,
    lambda_grid: npt.NDArray[np.float64],
    wn_grid: u.Quantity,
    # contribution_function: npt.NDArray[np.float64],
) -> float:
    """

    :param band_profile:
    :param lambda_grid:
    :param wn_grid:
    # :param contribution_function: Fractional contribution by the band to the total opacity.
    :return:
    """
    if len(band_profile.profile) == 0:
        return 0
    else:
        start_idx = band_profile.start_idx
        end_idx = start_idx + len(band_profile.profile)

        if len(band_profile.profile) == 1 and sum(band_profile.profile) != 0:
            return sum(lambda_grid[start_idx:end_idx])
        else:
            return simpson(
                band_profile.profile * lambda_grid[start_idx:end_idx],  # * contribution_function[start_idx:end_idx],
                x=wn_grid[start_idx:end_idx],
            )


def calc_imi(band_profile: BandProfile, i_grid: u.Quantity, wn_grid: u.Quantity) -> u.Quantity:
    """
    :param band_profile:
    :param i_grid:
    :param wn_grid:
    :return:
    """
    if len(band_profile.profile) == 0:
        return 0 << i_grid.unit
    else:
        start_idx = band_profile.start_idx
        end_idx = start_idx + len(band_profile.profile)

        if len(band_profile.profile) == 1 and sum(band_profile.profile) != 0:
            return sum(band_profile.profile * i_grid[start_idx:end_idx]) << i_grid.unit
        else:
            return (
                simpson(band_profile.profile * i_grid[start_idx:end_idx], x=wn_grid[start_idx:end_idx]) << i_grid.unit
            )


def pad_or_trim_profile(
    base_profile: npt.NDArray[np.float64], base_start: int, target_start: int, target_end: int
) -> npt.NDArray[np.float64]:
    base_end = base_start + len(base_profile)
    start_offset = target_start - base_start
    end_offset = base_end - target_end
    slice_start = max(0, start_offset)
    slice_end = len(base_profile) - max(0, end_offset)
    pad_left = max(-start_offset, 0)
    pad_right = max(0, -end_offset)
    return np.pad(base_profile[slice_start:slice_end], (pad_left, pad_right), constant_values=0)


def calc_lambda_approx_source(
    abs_band_profile: BandProfile,
    emi_band_profile: BandProfile,
    lambda_grid: npt.NDArray[np.float64],
    wn_grid: u.Quantity,
    a_fi: u.Quantity,
    b_fi: u.Quantity,
    b_if: u.Quantity,
    n_u: float,
    n_l: float,
    contribution_function: npt.NDArray[np.float64],
    is_emi: bool = True,
) -> u.Quantity:
    if is_emi:
        main_profile = emi_band_profile
    else:
        main_profile = abs_band_profile
    start_idx = main_profile.start_idx
    end_idx = main_profile.start_idx + len(main_profile.profile)

    if is_emi:
        emi_profile = emi_band_profile.profile
        abs_profile = pad_or_trim_profile(abs_band_profile.profile, abs_band_profile.start_idx, start_idx, end_idx)
    else:
        abs_profile = abs_band_profile.profile
        emi_profile = pad_or_trim_profile(emi_band_profile.profile, emi_band_profile.start_idx, start_idx, end_idx)

    source_profile_denom = n_l * b_if * abs_profile - n_u * b_fi * emi_profile
    source_profile = np.zeros(source_profile_denom.shape[0]) << a_fi.unit / source_profile_denom.unit / u.sr
    source_profile[source_profile_denom != 0] = (
        (a_fi * n_u * emi_profile[source_profile_denom != 0]) / source_profile_denom[source_profile_denom != 0] / u.sr
    )
    if np.any(np.isinf(source_profile)) or np.any(np.isnan(source_profile)):
        log.warning("NaNs/Infs. in Source profile = {source_profile}")
        log.warning(f"Abs. profile = {abs_profile}")
        log.warning(f"Emi. profile = {emi_profile}")

    if len(source_profile) == 0:
        return 0 << source_profile.unit

    product = (
        main_profile.profile
        * lambda_grid[start_idx:end_idx]
        * source_profile
        * contribution_function[start_idx:end_idx]
    )

    if len(main_profile.profile) == 1 and main_profile.profile.sum() != 0:
        return np.sum(product) << source_profile.unit
    else:
        return simpson(product, x=wn_grid[start_idx:end_idx]) << source_profile.unit


@numba.njit(parallel=True)
def _continuum_band_profile_variable_width(
    wn_grid: npt.NDArray[np.float64],
    n_frac_f: npt.NDArray[np.float64],
    n_frac_i: npt.NDArray[np.float64],
    a_fi: npt.NDArray[np.float64],
    g_f: npt.NDArray[np.float64],
    g_i: npt.NDArray[np.float64],
    energy_fi: npt.NDArray[np.float64],
    temperature: float,
    cont_broad: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    sqrtln2 = math.sqrt(math.log(2))
    _abs_xsec = np.zeros(wn_grid.shape)
    num_trans = energy_fi.shape[0]
    num_grid = wn_grid.shape[0]

    bin_widths = np.zeros(wn_grid.shape[0] + 1)
    bin_widths[1:-1] = (wn_grid[:-1] + wn_grid[1:]) / 2.0 - wn_grid[:-1]

    # abs_coef = (
    #     g_f
    #     * n_frac_i
    #     * a_fi
    #     * (1 - np.exp(-const_h_c_on_kB * energy_fi / temperature))
    #     / (const_16_pi_c * g_i * energy_fi**2)
    # )
    abs_coef = a_fi * ((n_frac_i * g_f / g_i) - n_frac_f) / (const_16_pi_c * energy_fi**2)
    sqrtln2_on_alpha = sqrtln2 / cont_broad

    for i in numba.prange(num_trans):
        # sqrtln2_on_alpha = sqrtln2 / cont_broad[i]
        for j in range(num_grid):
            wn_shift = wn_grid[j] - energy_fi[i]
            upper_width = bin_widths[j + 1]
            lower_width = bin_widths[j]
            # if np.abs(wn_shift) <= 25:
            if min(abs(wn_shift - lower_width), abs(wn_shift + upper_width)) <= 1500:
                _abs_xsec[j] += (
                    abs_coef[i]
                    * (
                        math.erf(sqrtln2_on_alpha[i] * (wn_shift + upper_width))
                        - math.erf(sqrtln2_on_alpha[i] * (wn_shift - lower_width))
                    )
                    / (upper_width + lower_width)
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
    temperature: float,
    cont_broad: npt.NDArray[np.float64],
    half_bin_width: float,
) -> npt.NDArray[np.float64]:
    twice_bin_width = 4.0 * half_bin_width
    sqrtln2 = math.sqrt(math.log(2))
    _abs_xsec = np.zeros(wn_grid.shape)
    num_trans = energy_fi.shape[0]
    num_grid = wn_grid.shape[0]
    cutoff = 1500 + half_bin_width

    # abs_coef = (
    #     g_f
    #     * n_frac_i
    #     * a_fi
    #     * (1 - np.exp(-const_h_c_on_kB * energy_fi / temperature))
    #     / (const_8_pi_c * twice_bin_width * g_i * energy_fi**2)
    # )
    abs_coef = a_fi * ((n_frac_i * g_f / g_i) - n_frac_f) / (const_8_pi_c * twice_bin_width * energy_fi**2)
    sqrtln2_on_alpha = sqrtln2 / cont_broad

    for i in numba.prange(num_trans):
        # sqrtln2_on_alpha = sqrtln2 / cont_broad[i]
        for j in range(num_grid):
            wn_shift = wn_grid[j] - energy_fi[i]
            if np.abs(wn_shift) <= cutoff:
                _abs_xsec[j] += abs_coef[i] * (
                    math.erf(sqrtln2_on_alpha[i] * (wn_shift + half_bin_width))
                    - math.erf(sqrtln2_on_alpha[i] * (wn_shift - half_bin_width))
                )
    return _abs_xsec


def abs_emi_xsec(
    states: pd.DataFrame,
    trans_files: t.List[pathlib.Path],
    temperature: u.Quantity,
    pressure: u.Quantity,
    species_mass: float,
    wn_grid: npt.NDArray[np.float64],
    broad_n: npt.NDArray[np.float64] = None,
    broad_gamma: npt.NDArray[np.float64] = None,
    n_gh_quad_points: int = _N_GH_QUAD_POINTS,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
):
    half_bin_width = abs(wn_grid[1] - wn_grid[0]) / 2.0
    # TODO: Check is_fixed_width rigour!
    is_fixed_width = np.all(np.isclose(np.diff(wn_grid), np.abs(wn_grid[1] - wn_grid[0]), atol=0))
    abs_xsec = np.zeros(wn_grid.shape)
    emi_xsec = np.zeros(wn_grid.shape)
    for trans_file in trans_files:
        trans_chunks = pd.read_csv(
            trans_file, sep=r"\s+", names=["id_f", "id_i", "A_fi"], usecols=[0, 1, 2], chunksize=chunk_size
        )
        for trans_chunk in trans_chunks:
            trans_chunk = trans_chunk.merge(
                states[["id", "energy", "n_nlte", "g"]], left_on="id_i", right_on="id", how="left"
            )
            # n_i being 0 doesn't matter for emission/n_f being 0 doesn't matter for absorption.
            # trans_chunk = trans_chunk.loc[trans_chunk["n_nlte"] > 0.0]
            trans_chunk = trans_chunk.drop(columns=["id"])
            trans_chunk = trans_chunk.rename(columns={"energy": "energy_i", "n_nlte": "n_nlte_i", "g": "g_i"})

            trans_chunk = trans_chunk.merge(
                states[["id", "energy", "n_nlte", "g", "tau"]], left_on="id_f", right_on="id", how="left"
            )
            trans_chunk = trans_chunk.drop(columns=["id"])
            trans_chunk = trans_chunk.rename(
                columns={"energy": "energy_f", "n_nlte": "n_nlte_f", "g": "g_f", "tau": "tau_f"}
            )

            trans_chunk["energy_fi"] = trans_chunk["energy_f"] - trans_chunk["energy_i"]
            # trans_chunk = trans_chunk.loc[(trans_chunk["energy_fi"] >= wn_grid[0])
            #                               & (trans_chunk["energy_fi"] <= wn_grid[-1])]
            trans_chunk = trans_chunk.loc[
                (trans_chunk["energy_f"] >= wn_grid[0])
                & (trans_chunk["energy_i"] <= wn_grid[-1])
                & (trans_chunk["energy_fi"] >= wn_grid[0])
                & (trans_chunk["energy_fi"] <= wn_grid[-1])
            ]

            if broad_n is None or broad_gamma is None:
                if is_fixed_width:
                    _abs_xsec, _emi_xsec = _abs_emi_binned_doppler_fixed_width(
                        wn_grid=wn_grid,
                        n_i=trans_chunk["n_nlte_i"].to_numpy(),
                        n_f=trans_chunk["n_nlte_f"].to_numpy(),
                        a_fi=trans_chunk["A_fi"].to_numpy(),
                        g_f=trans_chunk["g_f"].to_numpy(),
                        g_i=trans_chunk["g_i"].to_numpy(),
                        energy_fi=trans_chunk["energy_fi"].to_numpy(),
                        temperature=temperature.value,
                        species_mass=species_mass,
                        half_bin_width=half_bin_width,
                    )
                else:
                    _abs_xsec, _emi_xsec = _abs_emi_binned_doppler_variable_width(
                        wn_grid=wn_grid,
                        n_i=trans_chunk["n_nlte_i"].to_numpy(),
                        n_f=trans_chunk["n_nlte_f"].to_numpy(),
                        a_fi=trans_chunk["A_fi"].to_numpy(),
                        g_f=trans_chunk["g_f"].to_numpy(),
                        g_i=trans_chunk["g_i"].to_numpy(),
                        energy_fi=trans_chunk["energy_fi"].to_numpy(),
                        temperature=temperature.value,
                        species_mass=species_mass,
                    )
            else:
                gh_roots, gh_weights = roots_hermite(n_gh_quad_points)
                if is_fixed_width:
                    _abs_xsec, _emi_xsec = _abs_emi_binned_voigt_fixed_width(
                        wn_grid=wn_grid,
                        n_i=trans_chunk["n_nlte_i"].to_numpy(),
                        n_f=trans_chunk["n_nlte_f"].to_numpy(),
                        a_fi=trans_chunk["A_fi"].to_numpy(),
                        g_f=trans_chunk["g_f"].to_numpy(),
                        g_i=trans_chunk["g_i"].to_numpy(),
                        energy_fi=trans_chunk["energy_fi"].to_numpy(),
                        lifetimes=trans_chunk["tau_f"].to_numpy(),
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
                        n_i=trans_chunk["n_nlte_i"].to_numpy(),
                        n_f=trans_chunk["n_nlte_f"].to_numpy(),
                        a_fi=trans_chunk["A_fi"].to_numpy(),
                        g_f=trans_chunk["g_f"].to_numpy(),
                        g_i=trans_chunk["g_i"].to_numpy(),
                        energy_fi=trans_chunk["energy_fi"].to_numpy(),
                        lifetimes=trans_chunk["tau_f"].to_numpy(),
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
    continuum_states: pd.DataFrame,
    continuum_trans_files: t.List[pathlib.Path],
    temperature: u.Quantity,
    wn_grid: npt.NDArray[np.float64],
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
):
    is_fixed_width = np.all(np.isclose(np.diff(wn_grid), np.abs(wn_grid[1] - wn_grid[0]), atol=0))
    cont_xsec = np.zeros(wn_grid.shape)
    for trans_file in continuum_trans_files:
        trans_chunks = pd.read_csv(
            trans_file, sep=r"\s+", names=["id_c", "id_i", "A_ci", "broad"], usecols=[0, 1, 2, 3], chunksize=chunk_size
        )
        for trans_chunk in trans_chunks:
            trans_chunk = trans_chunk.merge(
                continuum_states[["id", "energy", "n_nlte", "g"]], left_on="id_i", right_on="id", how="left"
            )
            trans_chunk = trans_chunk.loc[trans_chunk["n_nlte"] > 0.0]  # Only absorption.
            trans_chunk = trans_chunk.drop(columns=["id"])
            trans_chunk = trans_chunk.rename(columns={"energy": "energy_i", "n_nlte": "n_nlte_i", "g": "g_i"})

            trans_chunk = trans_chunk.merge(
                continuum_states[["id", "energy", "n_nlte", "g"]], left_on="id_c", right_on="id", how="left"
            )
            trans_chunk = trans_chunk.drop(columns=["id"])
            trans_chunk = trans_chunk.rename(columns={"energy": "energy_c", "n_nlte": "n_nlte_c", "g": "g_c"})

            trans_chunk["energy_ci"] = trans_chunk["energy_c"] - trans_chunk["energy_i"]
            # trans_chunk = trans_chunk.loc[(trans_chunk["energy_ci"] >= wn_grid[0])
            #                               & (trans_chunk["energy_ci"] <= wn_grid[-1])]
            trans_chunk["n_nlte_c"] = trans_chunk["n_nlte_c"].fillna(0)
            trans_chunk = trans_chunk.loc[
                (trans_chunk["energy_c"] >= wn_grid[0])
                & (trans_chunk["energy_i"] <= wn_grid[-1])
                & (trans_chunk["energy_ci"] >= wn_grid[0])
                & (trans_chunk["energy_ci"] <= wn_grid[-1])
            ]
            if is_fixed_width:
                half_bin_width = abs(wn_grid[1] - wn_grid[0]) / 2.0
                cont_xsec += _continuum_binned_gauss_fixed_width(
                    wn_grid=wn_grid,
                    n_f=trans_chunk["n_nlte_c"].to_numpy(),
                    n_i=trans_chunk["n_nlte_i"].to_numpy(),
                    a_fi=trans_chunk["A_ci"].to_numpy(),
                    g_f=trans_chunk["g_c"].to_numpy(),
                    g_i=trans_chunk["g_i"].to_numpy(),
                    energy_fi=trans_chunk["energy_ci"].to_numpy(),
                    cont_broad=trans_chunk["broad"].to_numpy(),
                    temperature=temperature.value,
                    half_bin_width=half_bin_width,
                )
            else:
                cont_xsec += _continuum_binned_gauss_variable_width(
                    wn_grid=wn_grid,
                    n_f=trans_chunk["n_nlte_c"].to_numpy(),
                    n_i=trans_chunk["n_nlte_i"].to_numpy(),
                    a_fi=trans_chunk["A_ci"].to_numpy(),
                    g_f=trans_chunk["g_c"].to_numpy(),
                    g_i=trans_chunk["g_i"].to_numpy(),
                    energy_fi=trans_chunk["energy_ci"].to_numpy(),
                    cont_broad=trans_chunk["broad"].to_numpy(),
                    temperature=temperature.value,
                )
    return cont_xsec


@numba.njit(parallel=True)
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
    sqrtln2 = math.sqrt(math.log(2))
    _abs_xsec = np.zeros(wn_grid.shape)
    _emi_xsec = np.zeros(wn_grid.shape)
    num_trans = energy_fi.shape[0]
    num_grid = wn_grid.shape[0]
    cutoff = 25

    bin_widths = np.zeros(wn_grid.shape[0] + 1)
    bin_widths[1:-1] = (wn_grid[:-1] + wn_grid[1:]) / 2.0 - wn_grid[:-1]

    # abs_coef = (
    #     g_f
    #     * n_i
    #     * a_fi
    #     * (1 - np.exp(-const_h_c_on_kB * energy_fi / temperature))
    #     / (const_16_pi_c * g_i * energy_fi**2)
    # )
    abs_coef = a_fi * ((n_i * g_f / g_i) - n_f) / (const_16_pi_c * energy_fi**2)
    emi_coef = n_f * a_fi * energy_fi * const_h_c_on_8_pi
    sqrtln2_on_alpha = sqrtln2 / (energy_fi * const_sqrt_2_NA_kB_log2_on_c * np.sqrt(temperature / species_mass))
    for i in numba.prange(num_trans):
        # sqrtln2_on_alpha = sqrtln2 / alpha[i]
        for j in range(num_grid):
            wn_shift = wn_grid[j] - energy_fi[i]
            upper_width = bin_widths[j + 1]
            lower_width = bin_widths[j]
            # if np.abs(wn_shift) <= 25:
            # if min(abs(wn_shift - lower_width), abs(wn_shift + upper_width)) <= 25:  # WRONG!
            if -upper_width - cutoff <= wn_shift <= lower_width + cutoff:
                # if wn_grid[j] - lower_width - cutoff <= energy_fi[i] <= wn_grid[j] + upper_width + cutoff:
                bin_term = (
                    math.erf(sqrtln2_on_alpha[i] * (wn_shift + upper_width))
                    - math.erf(sqrtln2_on_alpha[i] * (wn_shift - lower_width))
                ) / (upper_width + lower_width)
                _abs_xsec[j] += abs_coef[i] * bin_term
                _emi_xsec[j] += emi_coef[i] * bin_term
    return _abs_xsec, _emi_xsec


@numba.njit(parallel=True)
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
    twice_bin_width = 4.0 * half_bin_width
    sqrtln2 = math.sqrt(math.log(2))
    _abs_xsec = np.zeros(wn_grid.shape)
    _emi_xsec = np.zeros(wn_grid.shape)
    num_trans = energy_fi.shape[0]
    num_grid = wn_grid.shape[0]
    cutoff = 25 + half_bin_width

    # abs_coef = (
    #     g_f
    #     * n_i
    #     * a_fi
    #     * (1 - np.exp(-const_h_c_on_kB * energy_fi / temperature))
    #     / (const_8_pi_c * twice_bin_width * g_i * energy_fi**2)
    # )
    abs_coef = a_fi * ((n_i * g_f / g_i) - n_f) / (const_8_pi_c * twice_bin_width * energy_fi**2)
    emi_coef = n_f * a_fi * energy_fi * const_h_c_on_4_pi / twice_bin_width
    sqrtln2_on_alpha = sqrtln2 / (energy_fi * const_sqrt_2_NA_kB_log2_on_c * np.sqrt(temperature / species_mass))
    for i in numba.prange(num_trans):
        # sqrtln2_on_alpha = sqrtln2 / alpha[i]
        for j in range(num_grid):
            wn_shift = wn_grid[j] - energy_fi[i]
            if np.abs(wn_shift) <= cutoff:
                bin_term = math.erf(sqrtln2_on_alpha[i] * (wn_shift + half_bin_width)) - math.erf(
                    sqrtln2_on_alpha[i] * (wn_shift - half_bin_width)
                )
                _abs_xsec[j] += abs_coef[i] * bin_term
                _emi_xsec[j] += emi_coef[i] * bin_term
    return _abs_xsec, _emi_xsec


@numba.njit(parallel=True)
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
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    _abs_xsec = np.zeros(wn_grid.shape)
    _emi_xsec = np.zeros(wn_grid.shape)
    num_trans = energy_fi.shape[0]
    num_grid = wn_grid.shape[0]
    cutoff = 25

    bin_widths = np.zeros(wn_grid.shape[0] + 1)
    bin_widths[1:-1] = (wn_grid[:-1] + wn_grid[1:]) / 2.0 - wn_grid[:-1]

    # Stimulated emission removed!
    abs_coef = a_fi * (n_i * g_f / g_i) / (const_8_pi_five_halves_c * energy_fi**2)
    emi_coef = n_f * a_fi * energy_fi * const_h_c_on_4_pi_five_halves
    sigma = energy_fi * const_sqrt_NA_kB_on_c * np.sqrt(temperature / species_mass)
    gamma_pressure = np.sum(broad_gamma * pressure * (t_ref / temperature) ** broad_n / pressure_ref)
    gamma_lifetime = 1 / (const_4_pi_c * lifetimes)
    gamma_total = gamma_lifetime + gamma_pressure

    for i in numba.prange(num_trans):
        gh_roots_sigma = gh_roots * sigma[i]
        start_sigma = wn_grid[0] - gh_roots_sigma
        end_sigma = wn_grid[-1] - gh_roots_sigma
        b_corr = np.pi / (
            np.arctan((end_sigma - energy_fi[i]) / gamma_total[i])
            - np.arctan((start_sigma - energy_fi[i]) / gamma_total[i])
        )
        for j in range(num_grid):
            wn_shift = wn_grid[j] - energy_fi[i]
            upper_width = bin_widths[j + 1]
            lower_width = bin_widths[j]
            # if min(abs(wn_shift - lower_width), abs(wn_shift + upper_width)) <= 25:  # WRONG!
            if -upper_width - cutoff <= wn_shift <= lower_width + cutoff:
                # if wn_grid[j] - lower_width - cutoff <= energy_fi[i] <= wn_grid[j] + upper_width + cutoff:
                shift_sigma = wn_shift - gh_roots_sigma
                bin_term = np.sum(
                    gh_weights
                    * b_corr
                    * (
                        np.arctan((shift_sigma + upper_width) / gamma_total[i])
                        - np.arctan((shift_sigma - lower_width) / gamma_total[i])
                    )
                ) / (upper_width + lower_width)
                _abs_xsec[j] += abs_coef[i] * bin_term
                _emi_xsec[j] += emi_coef[i] * bin_term
    return _abs_xsec, _emi_xsec


@numba.njit(parallel=True)
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
    _abs_xsec = np.zeros(wn_grid.shape)
    _emi_xsec = np.zeros(wn_grid.shape)
    num_trans = energy_fi.shape[0]
    num_grid = wn_grid.shape[0]
    cutoff = 25

    bin_widths = np.zeros(wn_grid.shape[0] + 1)
    bin_widths[1:-1] = (wn_grid[:-1] + wn_grid[1:]) / 2.0 - wn_grid[:-1]

    # abs_coef = (
    #     g_f
    #     * n_i
    #     * a_fi
    #     * (1 - np.exp(-const_h_c_on_kB * energy_fi / temperature))
    #     / (const_8_pi_five_halves_c * g_i * energy_fi**2)
    # )
    abs_coef = a_fi * ((n_i * g_f / g_i) - n_f) / (const_8_pi_five_halves_c * energy_fi**2)
    emi_coef = n_f * a_fi * energy_fi * const_h_c_on_4_pi_five_halves
    sigma = energy_fi * const_sqrt_NA_kB_on_c * np.sqrt(temperature / species_mass)
    gamma_pressure = np.sum(broad_gamma * pressure * (t_ref / temperature) ** broad_n / pressure_ref)
    gamma_lifetime = 1 / (const_4_pi_c * lifetimes)
    gamma_total = gamma_lifetime + gamma_pressure

    for i in numba.prange(num_trans):
        gh_roots_sigma = gh_roots * sigma[i]
        start_sigma = wn_grid[0] - gh_roots_sigma
        end_sigma = wn_grid[-1] - gh_roots_sigma
        b_corr = np.pi / (
            np.arctan((end_sigma - energy_fi[i]) / gamma_total[i])
            - np.arctan((start_sigma - energy_fi[i]) / gamma_total[i])
        )
        for j in range(num_grid):
            wn_shift = wn_grid[j] - energy_fi[i]
            upper_width = bin_widths[j + 1]
            lower_width = bin_widths[j]
            # if min(abs(wn_shift - lower_width), abs(wn_shift + upper_width)) <= 25:  # WRONG!
            if -upper_width - cutoff <= wn_shift <= lower_width + cutoff:
                # if wn_grid[j] - lower_width - cutoff <= energy_fi[i] <= wn_grid[j] + upper_width + cutoff:
                shift_sigma = wn_shift - gh_roots_sigma
                bin_term = np.sum(
                    gh_weights
                    * b_corr
                    * (
                        np.arctan((shift_sigma + upper_width) / gamma_total[i])
                        - np.arctan((shift_sigma - lower_width) / gamma_total[i])
                    )
                ) / (upper_width + lower_width)
                _abs_xsec[j] += abs_coef[i] * bin_term
                _emi_xsec[j] += emi_coef[i] * bin_term
    return _abs_xsec, _emi_xsec


@numba.njit(parallel=True)
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
    bin_width = 2.0 * half_bin_width
    _abs_xsec = np.zeros(wn_grid.shape)
    _emi_xsec = np.zeros(wn_grid.shape)
    num_trans = energy_fi.shape[0]
    num_grid = wn_grid.shape[0]
    cutoff = 25 + half_bin_width

    # abs_coef = (
    #     g_f
    #     * n_i
    #     * a_fi
    #     * (1 - np.exp(-const_h_c_on_kB * energy_fi / temperature))
    #     / (const_8_pi_five_halves_c * bin_width * g_i * energy_fi**2)
    # )
    abs_coef = a_fi * ((n_i * g_f / g_i) - n_f) / (const_8_pi_five_halves_c * bin_width * energy_fi**2)
    emi_coef = n_f * a_fi * energy_fi * const_h_c_on_4_pi_five_halves / bin_width
    sigma = energy_fi * const_sqrt_NA_kB_on_c * np.sqrt(temperature / species_mass)
    gamma_pressure = np.sum(broad_gamma * pressure * (t_ref / temperature) ** broad_n / pressure_ref)
    gamma_lifetime = 1 / (const_4_pi_c * lifetimes)
    gamma_total = gamma_lifetime + gamma_pressure

    for i in numba.prange(num_trans):
        gh_roots_sigma = gh_roots * sigma[i]
        start_sigma = wn_grid[0] - gh_roots_sigma
        end_sigma = wn_grid[-1] - gh_roots_sigma
        b_corr = np.pi / (
            np.arctan((end_sigma - energy_fi[i]) / gamma_total[i])
            - np.arctan((start_sigma - energy_fi[i]) / gamma_total[i])
        )
        for j in range(num_grid):
            wn_shift = wn_grid[j] - energy_fi[i]
            if np.abs(wn_shift) <= cutoff:
                shift_sigma = wn_shift - gh_roots_sigma
                bin_term = np.sum(
                    gh_weights
                    * b_corr
                    * (
                        np.arctan((shift_sigma + half_bin_width) / gamma_total[i])
                        - np.arctan((shift_sigma - half_bin_width) / gamma_total[i])
                    )
                )
                _abs_xsec[j] += abs_coef[i] * bin_term
                _emi_xsec[j] += emi_coef[i] * bin_term
    return _abs_xsec, _emi_xsec


@numba.njit(parallel=True)
def _continuum_binned_gauss_variable_width(
    wn_grid: npt.NDArray[np.float64],
    n_f: npt.NDArray[np.float64],
    n_i: npt.NDArray[np.float64],
    a_fi: npt.NDArray[np.float64],
    g_f: npt.NDArray[np.float64],
    g_i: npt.NDArray[np.float64],
    energy_fi: npt.NDArray[np.float64],
    cont_broad: npt.NDArray[np.float64],
    temperature: np.float64,
) -> npt.NDArray[np.float64]:
    sqrtln2 = math.sqrt(math.log(2))
    _abs_xsec = np.zeros(wn_grid.shape)
    num_trans = energy_fi.shape[0]
    num_grid = wn_grid.shape[0]
    cutoff = 1500

    bin_widths = np.zeros(wn_grid.shape[0] + 1)
    bin_widths[1:-1] = (wn_grid[:-1] + wn_grid[1:]) / 2.0 - wn_grid[:-1]

    # abs_coef = (
    #     g_f
    #     * n_i
    #     * a_fi
    #     * (1 - np.exp(-const_h_c_on_kB * energy_fi / temperature))
    #     / (const_16_pi_c * g_i * energy_fi**2)
    # )
    abs_coef = a_fi * ((n_i * g_f / g_i) - n_f) / (const_16_pi_c * energy_fi**2)
    sqrtln2_on_alpha = sqrtln2 / cont_broad
    for i in numba.prange(num_trans):
        # sqrtln2_on_alpha = sqrtln2 / cont_broad[i]
        for j in range(num_grid):
            wn_shift = wn_grid[j] - energy_fi[i]
            upper_width = bin_widths[j + 1]
            lower_width = bin_widths[j]
            # if min(abs(wn_shift - lower_width), abs(wn_shift + upper_width)) <= 1500:  # WRONG!
            if -upper_width - cutoff <= wn_shift <= lower_width + cutoff:
                # if wn_grid[j] - lower_width - cutoff <= energy_fi[i] <= wn_grid[j] + upper_width + cutoff:
                _abs_xsec[j] += (
                    abs_coef[i]
                    * (
                        math.erf(sqrtln2_on_alpha[i] * (wn_shift + upper_width))
                        - math.erf(sqrtln2_on_alpha[i] * (wn_shift - lower_width))
                    )
                    / (upper_width + lower_width)
                )
    return _abs_xsec


@numba.njit(parallel=True)
def _continuum_binned_gauss_fixed_width(
    wn_grid: npt.NDArray[np.float64],
    n_f: npt.NDArray[np.float64],
    n_i: npt.NDArray[np.float64],
    a_fi: npt.NDArray[np.float64],
    g_f: npt.NDArray[np.float64],
    g_i: npt.NDArray[np.float64],
    energy_fi: npt.NDArray[np.float64],
    cont_broad: npt.NDArray[np.float64],
    temperature: np.float64,
    half_bin_width: np.float64,
) -> npt.NDArray[np.float64]:
    twice_bin_width = 4.0 * half_bin_width
    sqrtln2 = math.sqrt(math.log(2))
    _abs_xsec = np.zeros(wn_grid.shape)
    num_trans = energy_fi.shape[0]
    num_grid = wn_grid.shape[0]
    cutoff = 1500 + half_bin_width

    # abs_coef = (
    #     g_f
    #     * n_i
    #     * a_fi
    #     * (1 - np.exp(-const_h_c_on_kB * energy_fi / temperature))
    #     / (const_8_pi_c * twice_bin_width * g_i * energy_fi**2)
    # )
    abs_coef = a_fi * ((n_i * g_f / g_i) - n_f) / (const_8_pi_c * twice_bin_width * energy_fi**2)
    sqrtln2_on_alpha = sqrtln2 / cont_broad
    for i in numba.prange(num_trans):
        # sqrtln2_on_alpha = sqrtln2 / cont_broad[i]
        for j in range(num_grid):
            wn_shift = wn_grid[j] - energy_fi[i]
            if np.abs(wn_shift) <= cutoff:
                _abs_xsec[j] += abs_coef[i] * (
                    math.erf(sqrtln2_on_alpha[i] * (wn_shift + half_bin_width))
                    - math.erf(sqrtln2_on_alpha[i] * (wn_shift - half_bin_width))
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
    chi_prime = negative_absorption_factor * np.max(global_chi_matrix, axis=1)
    effective_chi = np.where(
        global_chi_matrix < 0, chi_prime[:, None] * np.exp(-abs(global_chi_matrix.value)), global_chi_matrix
    )
    effective_source_func_matrix = np.where(
        global_source_func_matrix < 0,
        (global_eta_matrix * density_profile[:, None] / (ac.c.cgs * effective_chi)).to(
            u.J / (u.sr * u.m**2), equivalencies=u.spectral()
        ),
        global_source_func_matrix,
    )
    res = effective_chi * dz_profile[:, None]
    dtau = res.decompose().value
    tau = dtau[::-1].cumsum(axis=0)[::-1]
    tau_mu = tau[:, None, :] / mu_values[None, :, None]
    return effective_source_func_matrix, tau_mu


@numba.njit(parallel=True)
def bezier_coefficients(
    tau_mu_matrix: npt.NDArray[np.float64],
    source_function_matrix: npt.NDArray[np.float64],
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    # New.
    n_layers, n_angles, n_wavelengths = tau_mu_matrix.shape
    coefficients = np.zeros((n_layers + 1, 4, n_angles, n_wavelengths))
    control_points = np.zeros((n_layers, 2, n_angles, n_wavelengths))
    d_s_d_tau_in = np.zeros_like(tau_mu_matrix)
    d_s_d_tau_out = np.zeros_like(tau_mu_matrix)

    # coefficients[1:-1, 0, :, :] = tau_matrix[:-1]
    # coefficients[1:-1, 0, :, :] -= tau_matrix[1:]
    # Below needed to get coefficients at the boundary layers.
    coefficients[1:, 0, :, :] = tau_mu_matrix
    coefficients[:-1, 0, :, :] -= tau_mu_matrix
    # tau_plus is delta_tau_matrix[1:], tau_minus is delta_tau_matrix[:-1]

    delta_tau_limit = 1.4e-1
    delta_tau_limit_mask = coefficients[:, 0, :, :] < delta_tau_limit

    delta_tau_sq = coefficients[:, 0, :, :] ** 2
    delta_tau_cube = coefficients[:, 0, :, :] ** 3
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
                                * ((10 - coefficients[:, 0, :, :]) * coefficients[:, 0, :, :] - 90)
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
                                * ((140 - 18 * coefficients[:, 0, :, :]) * coefficients[:, 0, :, :] - 945)
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
                                * ((35 - 4 * coefficients[:, 0, :, :]) * coefficients[:, 0, :, :] - 270)
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
    return (ac_2_h_on_c_sq * freq_grid**3) / (np.exp(ac_h_on_kB * freq_grid / temperature) - 1) / u.sr


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


@numba.njit()
def calc_einstein_b_fi(a_fi: npt.NDArray[np.float64], energy_fi: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Here the Einstein B coefficient is given in cm^3 / (J * s * m).
    .. math:
        B_{fi}=\\frac{A_{fi}}{2hc\\tilde{\\nu}^{3}_{fi}

    :param a_fi:
    :param energy_fi:
    :return:
    """
    # return a_fi / (2 * ac.h * ac.c * (energy_fi ** 3))  # WAVENUMBERS
    return (a_fi * (ac.c**2)) / (2 * ac.h * (energy_fi**3))  # FREQUENCY
    # return a_fi * (energy_fi ** 3) / (2 * ac.h * ac.c)  # LAMBDA


@numba.njit()
def calc_einstein_b_if(
    b_fi: npt.NDArray[np.float64],
    g_f: npt.NDArray[np.float64],
    g_i: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return b_fi * g_f / g_i


def boltzmann_population(states: pd.DataFrame, temperature: u.Quantity) -> pd.DataFrame:
    states["q_lev"] = states["g"] * np.exp(-ac_h_c_on_kB * (states["energy"] << 1 / u.cm) / temperature)
    states["n"] = states["q_lev"] / states["q_lev"].sum()
    states_agg_n = states.groupby(by=["id_agg"], as_index=False).agg(n_agg=("n", "sum"))
    states = states.merge(states_agg_n, on=["id_agg"], how="left")
    return states


@numba.njit()
def sampling_point(wn_val: float, temperature: float) -> float:
    return (const_2_pi_h_c_sq_on_sigma_sba * wn_val**3) / (
        temperature**4 * (np.exp(const_h_c_on_kB * wn_val / temperature) - 1)
    )


def sample_grid_new(
    wn_start: float, wn_end: float, temperature: float, const: float, max_step: float = np.inf
) -> npt.NDArray[np.float64]:
    wn_grid = [wn_start]
    ev_grid = []
    while wn_grid[-1] <= wn_end:
        ev_dv = sampling_point(wn_grid[-1], temperature)
        ev_grid.append(ev_dv)
        step_size = min(const / ev_dv, max_step)
        next_wn = wn_grid[-1] + step_size

        if next_wn > wn_end or np.isnan(next_wn) or next_wn == np.inf:
            break
        wn_grid.append(next_wn)
    return np.stack((wn_grid, ev_grid))


# def unified_opacity_sampling_grid(
#     wn_start: float, wn_end: float, temperature_profile: u.Quantity, const: float = None, max_step: float = 1000
# ) -> u.Quantity:
#     if const is None:
#         # Needs testing for very low wn_start - density of points might get too high so impose conservative max_step?
#         const = 10 ** (-(6 - np.log10(wn_start)))
#     layer_samples = [
#         sample_grid_new(wn_start=wn_start, wn_end=wn_end, temperature=temperature, const=const, max_step=max_step)
#         for temperature in temperature_profile.value
#     ]
#
#     temperature_sorted_idxs = np.argsort(temperature_profile.value)
#
#     merged_wn = [wn_start]
#     current_step = 0
#
#     while current_step < len(temperature_sorted_idxs) - 1:
#         current_next_idx = np.searchsorted(layer_samples[temperature_sorted_idxs[current_step]][0], merged_wn[-1]) + 1
#         current_next_point = layer_samples[temperature_sorted_idxs[current_step]][:, current_next_idx]
#         next_idx = np.searchsorted(layer_samples[temperature_sorted_idxs[current_step + 1]][0], merged_wn[-1])
#         next_point = layer_samples[temperature_sorted_idxs[current_step + 1]][:, next_idx]
#         if next_point[0] <= current_next_point[0] and next_point[1] > current_next_point[1]:
#             merged_wn.append(next_point[0])
#             current_step += 1
#         else:
#             merged_wn.append(current_next_point[0])
#     final_next_idx = np.searchsorted(layer_samples[temperature_sorted_idxs[current_step]][0], merged_wn[-1]) + 1
#     final_next_points = layer_samples[temperature_sorted_idxs[current_step]][:, final_next_idx:]
#     # merged_wn.append(final_next_points[0])
#     return np.append(np.array(merged_wn), final_next_points[0]) << u.k


@numba.njit()
def calc_ev_grid(wn_grid: npt.NDArray[np.float64], temperature: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return (const_2_pi_h_c_sq_on_sigma_sba * wn_grid**3) / (
        temperature**4 * (np.exp(const_h_c_on_kB * wn_grid / temperature) - 1)
    )


# TODO: optimise this!
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

    # sample_idxs = np.searchsorted(ev_cdf, np.linspace(0, 1, num_points))
    sample_idxs = np.array(
        [
            np.argmin(abs(ev_cdf - point)) if point <= ev_cdf[-1] else len(ev_cdf) - 1
            for point in np.linspace(0, 1, num_points)
        ]
    )
    sample_idxs = np.unique(sample_idxs)
    # Both methods allow for the same wn point to be the closest to multiple values on the CDF, so remove duplicates.
    # The argmin approach should be better when using a small number of points but is marginally slower.

    if sample_idxs[-1] == len(ev_cdf):
        sample_idxs[-1] -= 1
    if max_step:
        max_idx_step = int(np.ceil(max_step * num_cdf_points / (wn_end - wn_start))) - 1
        idx_diffs = np.diff(sample_idxs)
        idxs_diffs_over_max = np.nonzero(idx_diffs > max_idx_step)[0]
        idxs_diffs_over_max_chunks = np.split(idxs_diffs_over_max, np.where(np.diff(idxs_diffs_over_max) != 1)[0] + 1)
        chunk_idx = 0
        while chunk_idx < len(idxs_diffs_over_max_chunks):
            chunk = idxs_diffs_over_max_chunks[chunk_idx]
            end_idx = chunk[-1] + 1 if chunk[-1] + 1 < len(sample_idxs) else chunk[-1]
            n_new_points = int(np.ceil((sample_idxs[end_idx] - sample_idxs[chunk[0]]) / max_idx_step)) + 1
            insert_vals = np.linspace(sample_idxs[chunk[0]], sample_idxs[end_idx], n_new_points, dtype=int)
            sample_idxs = np.concatenate((sample_idxs[: chunk[0]], insert_vals, sample_idxs[chunk[-1] + 2 :]))
            idxs_diffs_over_max_chunks = list(
                idx_chunk + n_new_points - len(chunk) - 1 for idx_chunk in idxs_diffs_over_max_chunks
            )
            chunk_idx += 1

    sampled_grid = temp_wn_grid[sample_idxs]
    return sampled_grid << u.k
