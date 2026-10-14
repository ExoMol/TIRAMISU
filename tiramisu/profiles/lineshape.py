"""Line-shape primitives (Humlicek Voigt, broadening terms, profile windows, binary searches) and spectral-grid
helpers shared by the profile kernels."""
import typing as t

import numba
import numpy as np
from numpy import typing as npt

from ..constants import const_4_pi_c, const_sqrt_NA_kB_on_c
from ..numerics import loglinear_integral_1d


def grid_bin_edges(wn_arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Edges of the bins centred on the grid points: the midpoints, plus half a step beyond each end."""
    return np.concatenate([[wn_arr[0] - (wn_arr[1] - wn_arr[0]) * 0.5], (wn_arr[:-1] + wn_arr[1:]) * 0.5,
                           [wn_arr[-1] + (wn_arr[-1] - wn_arr[-2]) * 0.5]])


def rebin_spectrum(
        x_in: npt.NDArray[np.float64],
        y_in: npt.NDArray[np.float64],
        new_centers: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Re-bin a high-resolution spectrum onto bins centered at `new_centers` (edges from :func:`grid_bin_edges`), as the bin
    average of its piecewise log-linear interpolant. `x_in` must be ascending.
    """

    edges = grid_bin_edges(new_centers)
    # Exact edge values via interpolation.
    y_edges = np.interp(edges, x_in, y_in)
    # Interior points of bin i are x_in[lo[i]:hi[i]], i.e. left < x < right (a slice, as x_in is sorted).
    lo = np.searchsorted(x_in, edges[:-1], side="right")
    hi = np.searchsorted(x_in, edges[1:], side="left")

    y_out = np.zeros_like(new_centers)

    for i in range(len(new_centers)):
        x_bin = np.concatenate(([edges[i]], x_in[lo[i]:hi[i]], [edges[i + 1]]))
        y_bin = np.concatenate(([y_edges[i]], y_in[lo[i]:hi[i]], [y_edges[i + 1]]))
        # Bin average
        bin_integral = loglinear_integral_1d(y_data=y_bin, dx=np.diff(x_bin))
        y_out[i] = bin_integral / (edges[i + 1] - edges[i])

    return y_out


@numba.njit(parallel=True, cache=True, error_model="numpy")
def gamma_lifetime_lookup_from_tau(tau_lookup: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Lifetime (natural) broadening contribution of each state to the Lorentzian HWHM, 1 / (4 pi c tau) [cm^-1].

    The HWHM of a transition is the sum of its upper and lower state contributions. States without a finite, positive
    lifetime (tau is sanitised to 0 where it is missing or infinite, e.g. the ground state) contribute nothing.

    Parameters
    ----------
    tau_lookup : ndarray, shape (n_states + 1,)
        State lifetimes [s].

    Returns
    -------
    gamma_lifetime_lookup : ndarray, shape (n_states + 1,)
    """
    n_states = tau_lookup.shape[0]
    gamma_lifetime_lookup = np.zeros(n_states, dtype=np.float64)
    for s in numba.prange(n_states):
        tau_s = tau_lookup[s]
        if tau_s > 0.0 and np.isfinite(tau_s):
            gamma_lifetime_lookup[s] = 1.0 / (const_4_pi_c * tau_s)
    return gamma_lifetime_lookup


@numba.njit(parallel=True, cache=True, error_model="numpy")
def profile_windows(
        wn_grid: npt.NDArray[np.float64], cutoff: float
) -> t.Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """
    Grid index window [j_start, j_end) over which a line centred on each grid point is evaluated, for a fixed
    +/- cutoff [cm^-1]. Both bounds are non-decreasing in the grid index.
    """
    n_grid = wn_grid.shape[0]
    j_starts = np.empty(n_grid, dtype=np.int32)
    j_ends = np.empty(n_grid, dtype=np.int32)
    for i in numba.prange(n_grid):
        j_starts[i] = max(0, binary_search_left(wn_grid, wn_grid[i] - cutoff))
        j_ends[i] = min(n_grid, binary_search_right(wn_grid, wn_grid[i] + cutoff) + 1)
    return j_starts, j_ends


@numba.njit(cache=True, error_model="numpy")
def layer_broadening_terms(
        temperatures: npt.NDArray[np.float64],
        pressures: npt.NDArray[np.float64],
        broad_n: npt.NDArray[np.float64],
        broad_gamma: npt.NDArray[np.float64],
        species_mass: float,
        t_ref: float,
        pressure_ref: float,
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Per-layer Voigt broadening terms shared by the CPU and GPU profile kernels.

    Returns
    -------
    inv_doppler : ndarray, shape (n_layers,)
        1 / (sigma_D * sqrt(2)) = inv_doppler[l] / energy for a line or bin at wavenumber energy.
    gamma_pressure : ndarray, shape (n_layers,)
        Pressure-broadening HWHM [cm^-1].
    """
    n_layers = temperatures.shape[0]
    sqrt2_NA_kB_on_c = np.sqrt(2.0) * const_sqrt_NA_kB_on_c
    inv_doppler = np.empty(n_layers, dtype=np.float64)
    gamma_pressure = np.zeros(n_layers, dtype=np.float64)
    for l in range(n_layers):
        temp_l = temperatures[l]
        inv_doppler[l] = 1.0 / (sqrt2_NA_kB_on_c * np.sqrt(temp_l / species_mass))
        for b in range(broad_n.shape[0]):
            gamma_pressure[l] += broad_gamma[b, l] * pressures[l] * (t_ref / temp_l) ** broad_n[b] / pressure_ref
    return inv_doppler, gamma_pressure


@numba.njit(cache=True, error_model="numpy", inline="always")
def binary_search_left(arr: npt.NDArray[np.float64], value: np.float64, start: np.int32 = 0) -> np.int32:
    """Fast binary search for left insertion point with optional start hint."""
    left, right = start, len(arr)
    while left < right:
        mid = (left + right) >> 1
        if arr[mid] < value:
            left = mid + 1
        else:
            right = mid
    return left


@numba.njit(cache=True, error_model="numpy", inline="always")
def binary_search_right(arr: npt.NDArray[np.float64], value: np.float64, start: np.int32 = 0) -> np.int32:
    """Fast binary search for right insertion point with optional start hint."""
    left, right = start, len(arr)
    while left < right:
        mid = (left + right) >> 1
        if arr[mid] <= value:
            left = mid + 1
        else:
            right = mid
    return left


@numba.njit(cache=True, error_model="numpy", inline="always")
def voigt_humlicek_w(x: float, y: float) -> float:
    """
    Real part of the Faddeeva function w(z) = exp(-z²) * erfc(-iz) via the Humlíček (1979) 4-region rational
    approximation.

    Relative accuracy ~1e-4 (measured against scipy.special.wofz). x and y are the real and imaginary parts of
    z = (nu - nu0) / (sigma_D * sqrt(2)), where y > 0 always (it is the ratio of Lorentzian to Gaussian halfwidths).

    Parameters
    ----------
    x : float
       (nu - nu0) / (sigma_D * sqrt(2)).
    y : float
       gamma_L / (sigma_D * sqrt(2)). Always positive.

    Returns
    -------
    float  Re[w(z)] - proportional to the Voigt profile at this point.
    """
    humlicek_t = y - 1j * x  # Numba supports complex arithmetic in njit

    # Region selection on |x| + y following Humlíček (1979).
    humlicek_s = abs(x) + y

    if humlicek_s >= 15.0:
        # Region 1: far wings, single-term approximation.
        humlicek_w = humlicek_t * 0.5641896 / (0.5 + humlicek_t * humlicek_t)
    elif humlicek_s >= 5.5:
        # Region 2.
        humlicek_u = humlicek_t * humlicek_t
        humlicek_w = humlicek_t * (1.410474 + humlicek_u * 0.5641896) / (0.75 + humlicek_u * (3.0 + humlicek_u))
    elif y >= 0.195 * abs(x) - 0.176:
        # Region 3: near line centre.
        humlicek_w = (16.4955 + humlicek_t * (
                20.20933 + humlicek_t * (11.96482 + humlicek_t * (3.778987 + humlicek_t * 0.5642236)))) / (
                             16.4955 + humlicek_t * (38.82363 + humlicek_t * (
                             39.27121 + humlicek_t * (21.69274 + humlicek_t * (6.699398 + humlicek_t))))
                     )
    else:
        # Region 4: intermediate, exponential term dominates.
        humlicek_u = humlicek_t * humlicek_t
        humlicek_w = np.exp(humlicek_u) - humlicek_t * (
                36183.31 - humlicek_u * (3321.99 - humlicek_u * (1540.787 - humlicek_u * (
                219.031 - humlicek_u * (35.7668 - humlicek_u * (1.320522 - humlicek_u * 0.56419)))))
        ) / (
                             32066.6 - humlicek_u * (24322.84 - humlicek_u * (
                             9022.228 - humlicek_u * (2186.181 - humlicek_u * (
                             364.2191 - humlicek_u * (61.57037 - humlicek_u * (1.841439 - humlicek_u))))))
                     )

    return humlicek_w.real
