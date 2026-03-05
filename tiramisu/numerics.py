import numba
import numpy as np
from astropy import units as u
from numpy import typing as npt


# ----------------------------- NUMBA SIMPSON INTEGRALS AND NORMALISATION -----------------------------

@numba.njit(cache=True, error_model="numpy", inline="always")
def simpson_integral_numba(y_data: npt.NDArray[np.float64], x_data: npt.NDArray[np.float64]) -> float:
    """
    Fast Simpson's rule integration using numba.
    Assumes evenly or unevenly spaced x values.
    """
    n_points = len(y_data)
    if n_points < 2:
        return 0.0

    if n_points == 2:
        # Trapezoidal for 2 points
        return 0.5 * (y_data[0] + y_data[1]) * (x_data[1] - x_data[0])

    # Simpson's 1/3 rule
    h = x_data[1:] - x_data[:-1]
    result = 0.0

    for i in range(0, n_points - 2, 2):
        # Simpson's rule for each pair of intervals
        h0 = h[i]
        h1 = h[i + 1]

        if abs(h0 - h1) < 1e-10:  # Uniform spacing
            result += (h0 / 3.0) * (y_data[i] + 4 * y_data[i + 1] + y_data[i + 2])
        else:  # Non-uniform spacing
            alpha = (2 * h0 ** 2 + 2 * h0 * h1 - h1 ** 2) / (6 * h0)
            beta = (h0 ** 2 + h0 * h1) / (3 * h1)
            result += alpha * y_data[i] + beta * y_data[i + 1] + (h0 + h1 - alpha - beta) * y_data[i + 2]

    # Handle last interval if odd number of points (use trapezoidal)
    if n_points % 2 == 0:
        result += 0.5 * (y_data[n_points - 2] + y_data[n_points - 1]) * h[n_points - 2]

    return result


@numba.njit(parallel=True, cache=True, error_model="numpy")
def simpson_integral_2d(
        y_data: npt.NDArray[np.float64],
        x_data: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Vectorized Simpson integration over axis 1.
    Each row is integrated independently.
    """
    num_rows, num_grid = y_data.shape
    result = np.zeros(num_rows, dtype=np.float64)

    if num_grid < 2:
        return result

    for row in numba.prange(num_rows):
        result[row] = simpson_integral_numba(y_data=y_data[row], x_data=x_data)

    return result


def simpson_quantity(y_data: u.Quantity, x_data: u.Quantity) -> u.Quantity:
    """Simpson integration preserving units."""
    result = simpson_integral_numba(y_data.value, x_data.value)
    return result * (y_data.unit * x_data.unit)


def simpson_quantity_2d(
        y_data: u.Quantity,
        x_data: u.Quantity,
) -> u.Quantity:
    """Vectorized Simpson integration preserving units."""
    result = simpson_integral_2d(y_data=y_data.value, x_data=x_data.value)
    return result << (y_data.unit * x_data.unit)


@numba.njit(cache=True, error_model="numpy")
def simpson_normalise_1d(
        y_data: npt.NDArray[np.float64],
        x_data: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Normalise y_data by its Simpson integral over x_data.

    Parameters
    ----------
    y_data : np.ndarray, shape (n_grid,)
    x_data : np.ndarray, shape (n_grid,)

    Returns
    -------
    normalised : np.ndarray, shape (n_grid,)
        y_data / integral(y_data, x_data). Units of 1/x_data.unit.
        Returned unchanged if the integral is zero.
    """
    integral = simpson_integral_numba(y_data=y_data, x_data=x_data)
    if integral == 0.0:
        return y_data.copy()
    return y_data / integral


def simpson_normalise_quantity_1d(
        y_data: u.Quantity,
        x_data: u.Quantity,
) -> u.Quantity:
    """
    Normalise y_data by its Simpson integral, preserving units.

    Parameters
    ----------
    y_data : astropy.units.Quantity, shape (n_grid,)
    x_data : astropy.units.Quantity, shape (n_grid,)

    Returns
    -------
    astropy.units.Quantity, shape (n_grid,)
        Units are 1 / x_data.unit.
    """
    result = simpson_normalise_1d(y_data=y_data.value, x_data=x_data.value)
    return result << (1 / x_data.unit)


@numba.njit(parallel=True, cache=True, error_model="numpy", inline="always")
def simpson_normalise_2d(
        y_data: npt.NDArray[np.float64],
        x_data: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Normalise each row of y_data by its Simpson integral over x_data.

    Each row of the output is y_data[row] / integral(y_data[row], x_data).
    Rows whose integral is zero are left as-is (no division).

    Parameters
    ----------
    y_data : np.ndarray, shape (n_rows, n_grid)
        2D array where each row is an independent distribution to normalise.
    x_data : np.ndarray, shape (n_grid,)
        Shared x-axis for all rows.

    Returns
    -------
    normalised : np.ndarray, shape (n_rows, n_grid)
        Row-normalised y_data. Units of 1/x_data.unit (normalisation removes y units).
    """
    num_rows, num_grid = y_data.shape
    result = np.empty((num_rows, num_grid), dtype=np.float64)

    if num_grid < 2:
        for row in numba.prange(num_rows):
            for col in range(num_grid):
                result[row, col] = y_data[row, col]
        return result

    for row in numba.prange(num_rows):
        integral = simpson_integral_numba(y_data=y_data[row], x_data=x_data)
        if integral != 0.0:
            inv = 1.0 / integral
            for col in range(num_grid):
                result[row, col] = y_data[row, col] * inv
        else:
            for col in range(num_grid):
                result[row, col] = y_data[row, col]

    return result


def simpson_normalise_quantity_2d(
        y_data: u.Quantity,
        x_data: u.Quantity,
) -> u.Quantity:
    """
    Normalise each row of y_data by its Simpson integral, preserving units.

    Parameters
    ----------
    y_data : astropy.units.Quantity, shape (n_rows, n_grid)
    x_data : astropy.units.Quantity, shape (n_grid,)

    Returns
    -------
    astropy.units.Quantity, shape (n_rows, n_grid)
        Each row divided by its integral; units are 1 / x_data.unit.
    """
    result = simpson_normalise_2d(y_data=y_data.value, x_data=x_data.value)
    return result << (1 / x_data.unit)
