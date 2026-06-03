import numba
import numpy as np
from astropy import units as u
from numpy import typing as npt


# ----------------------------- LOG-LINEAR QUADRATURE -----------------------------
# Exact for piecewise-exponential integrands (Voigt line wings, exponential cross-section variation). Falls back to
# trapezoidal for flat/near-flat panels.

@numba.njit(cache=True, error_model="numpy", inline="always")
def _loglinear_panel(y0: float, y1: float, dx: float) -> float:
    """
    Exact integral of the piecewise log-linear interpolant on one panel.

    Falls back to trapezoidal (l'Hopital limit) when y1 is roughly y0.
    Falls back to trapezoidal when either endpoint is non-positive (continuum or zero-padded edges).

    Parameters
    ----------
    y0 : float
        y data value at start of bin.
    y1 : float
        y data value at end of bin.
    dx : float
        x-axis increment.

    Returns
    ----------
    float
    """
    if y0 <= 0.0 or y1 <= 0.0:
        return 0.5 * dx * (y0 + y1)
    log_ratio = np.log(y1 / y0)
    if abs(log_ratio) < 1e-8:
        return 0.5 * dx * (y0 + y1)
    return dx * (y1 - y0) / log_ratio


@numba.njit(cache=True, error_model="numpy", inline="always")
def loglinear_integral_1d(
        y_data: npt.NDArray[np.float64],
        dx: npt.NDArray[np.float64],
) -> float:
    """
    Piecewise log-linear quadrature over a 1D array.

    Exact for functions that vary exponentially between grid points. Primarily important near line centres.

    Parameters
    ----------
    y_data : np.ndarray, shape (n_grid,)
        Cross-section values. Non-negative; zeros trigger trapezoidal fallback.
    dx : np.ndarray, shape (n_grid - 1,)
        Wavenumber grid steps.

    Returns
    -------
    float
    """
    n = len(dx)
    result = 0.0
    for i in range(n):
        result += _loglinear_panel(y_data[i], y_data[i + 1], dx[i])
    return result


@numba.njit(cache=True, error_model="numpy", inline="always")
def loglinear_integral_1d_nonnegative(
        y_data: npt.NDArray[np.float64],
        dx: npt.NDArray[np.float64],
) -> float:
    """
    Piecewise log-linear quadrature over a 1D array for y_data >= 0.

    Exact for functions that vary exponentially between grid points. Primarily important near line centres.

    Parameters
    ----------
    y_data : np.ndarray, shape (n_grid,)
        Cross-section values. Non-negative; zeros trigger trapezoidal fallback.
    dx : np.ndarray, shape (n_grid - 1,)
        Wavenumber grid steps.

    Returns
    -------
    float
    """
    n = len(dx)
    result = 0.0
    for i in range(n):
        dk = dx[i]
        y0 = y_data[i]
        y1 = y_data[i + 1]

        # Sparse branch in real spectra
        if y0 == 0.0 or y1 == 0.0:
            result += 0.5 * dk * (y0 + y1)
            continue

        lr = np.log(y1 / y0)
        if abs(lr) < 1e-12:
            # result += 0.5 * dk * (y0 + y1)
            # y0 roughly equals y1, i.e.: 0.5 * (y0 + y1) ~ y0
            result += dk * y0
        else:
            # result += dk * (y1 - y0) / lr
            # If y1 = y0 * exp(lr), then (y1 - y0)/lr = y0 * (exp(lr) - 1) / lr; the latter is more stable for small lr.
            result += dk * y0 * np.expm1(lr) / lr
    return result


@numba.njit(parallel=True, cache=True, error_model="numpy")
def loglinear_integral_2d(
        y_data: npt.NDArray[np.float64],
        dx: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Vectorised log-linear quadrature over rows of y_data.

    Parameters
    ----------
    y_data : np.ndarray, shape (n_rows, n_grid)
    dx : np.ndarray, shape (n_grid - 1,)

    Returns
    -------
    np.ndarray, shape (n_rows,)
    """
    n_rows = y_data.shape[0]
    result = np.zeros(n_rows, dtype=np.float64)
    for row in numba.prange(n_rows):
        result[row] = loglinear_integral_1d(y_data[row], dx)
    return result


@numba.njit(parallel=True, cache=True, error_model="numpy")
def loglinear_integral_2d_nonnegative(
        y_data: npt.NDArray[np.float64],
        dx: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Vectorised log-linear quadrature over rows of y_data. All y_data must be non-negative.

    Parameters
    ----------
    y_data : np.ndarray, shape (n_rows, n_grid)
    dx : np.ndarray, shape (n_grid - 1,)

    Returns
    -------
    np.ndarray, shape (n_rows,)
    """
    n_rows = y_data.shape[0]
    result = np.zeros(n_rows, dtype=np.float64)
    for row in numba.prange(n_rows):
        result[row] = loglinear_integral_1d_nonnegative(y_data[row], dx)
    return result


@numba.njit(cache=True, error_model="numpy", inline="always")
def loglinear_normalise_1d(
        y_data: npt.NDArray[np.float64],
        dx: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Normalise y_data by its log-linear integral.

    Uses the same quadrature as loglinear_integral_1d.

    Parameters
    ----------
    y_data : np.ndarray, shape (n_grid,)
        Cross-section values. Non-negative; zeros trigger trapezoidal fallback.
    dx : np.ndarray, shape (n_grid - 1,)
        Wavenumber grid steps.

    Returns
    -------
    float
    """
    integral = loglinear_integral_1d(y_data=y_data, dx=dx)
    if integral == 0.0:
        return np.zeros_like(y_data)
    return y_data / integral


@numba.njit(cache=True, error_model="numpy", inline="always")
def loglinear_normalise_1d_nonnegative(
        y_data: npt.NDArray[np.float64],
        dx: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Normalise y_data by its log-linear integral. All y_data must be non-negative.

    Uses the same quadrature as loglinear_integral_1d.

    Parameters
    ----------
    y_data : np.ndarray, shape (n_grid,)
        Cross-section values. Non-negative; zeros trigger trapezoidal fallback.
    dx : np.ndarray, shape (n_grid - 1,)
        Wavenumber grid steps.

    Returns
    -------
    float
    """
    integral = loglinear_integral_1d_nonnegative(y_data=y_data, dx=dx)
    if integral == 0.0:
        return np.zeros_like(y_data)
    return y_data / integral


@numba.njit(parallel=True, cache=True, error_model="numpy")
def loglinear_normalise_2d(
        y_data: npt.NDArray[np.float64],
        dx: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Normalise each row of y_data by its log-linear integral.

    Parameters
    ----------
    y_data : np.ndarray, shape (n_rows, n_grid)
    dx : np.ndarray, shape (n_grid - 1,)

    Returns
    -------
    np.ndarray, shape (n_rows, n_grid)
        Each row divided by its log-linear integral.
    """
    n_rows, n_cols = y_data.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)
    for row in numba.prange(n_rows):
        integral = loglinear_integral_1d(y_data[row], dx)
        inv = 1.0 / integral if integral != 0.0 else 0.0
        for col in range(n_cols):
            result[row, col] = y_data[row, col] * inv
    return result


@numba.njit(parallel=True, cache=True, error_model="numpy")
def loglinear_normalise_2d_nonnegative(
        y_data: npt.NDArray[np.float64],
        dx: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Normalise each row of y_data by its log-linear integral. All y_data must be non-negative.

    Parameters
    ----------
    y_data : np.ndarray, shape (n_rows, n_grid)
    dx : np.ndarray, shape (n_grid - 1,)

    Returns
    -------
    np.ndarray, shape (n_rows, n_grid)
        Each row divided by its log-linear integral.
    """
    n_rows, n_cols = y_data.shape
    result = np.empty((n_rows, n_cols), dtype=np.float64)
    for row in numba.prange(n_rows):
        integral = loglinear_integral_1d_nonnegative(y_data[row], dx)
        inv = 1.0 / integral if integral != 0.0 else 0.0
        for col in range(n_cols):
            result[row, col] = y_data[row, col] * inv
    return result


# ----------------------------- ASTROPY QUANTITY WRAPPERS -----------------------------

def loglinear_integral_quantity_1d(
        y_data: u.Quantity,
        dx: u.Quantity,
) -> u.Quantity:
    """

    Parameters
    ----------
    y_data : astropy.units.Quantity, shape (n_grid,)
    dx : astropy.units.Quantity, shape (n_grid - 1,)

    Returns
    -------
    integral : astropy.units.Quantity
        Scalar integral of y_data with respect to x_data. Has units of y_data.unit * x_data.unit.
    """
    result = loglinear_integral_1d(y_data.value, dx.value)
    return result * (y_data.unit * dx.unit)


def loglinear_integral_quantity_1d_nonnegative(
        y_data: u.Quantity,
        dx: u.Quantity,
) -> u.Quantity:
    """
    All y_data must be non-negative.

    Parameters
    ----------
    y_data : astropy.units.Quantity, shape (n_grid,)
    dx : astropy.units.Quantity, shape (n_grid - 1,)

    Returns
    -------
    integral : astropy.units.Quantity
        Scalar integral of y_data with respect to x_data. Has units of y_data.unit * x_data.unit.
    """
    result = loglinear_integral_1d_nonnegative(y_data.value, dx.value)
    return result * (y_data.unit * dx.unit)


def loglinear_integral_quantity_2d(
        y_data: u.Quantity,
        dx: u.Quantity,
) -> u.Quantity:
    """

    Parameters
    ----------
    y_data : astropy.units.Quantity, shape (n_rows, n_grid)
    dx : astropy.units.Quantity, shape (n_grid - 1,)

    Returns
    -------
    integral : astropy.units.Quantity, shape (n_rows,)
        Integral of each row of y_data with respect to x_data. Has units of y_data.unit * x_data.unit.
    """
    result = loglinear_integral_2d(y_data.value, dx.value)
    return result << (y_data.unit * dx.unit)


def loglinear_integral_quantity_2d_nonnegative(
        y_data: u.Quantity,
        dx: u.Quantity,
) -> u.Quantity:
    """
    All y_data must be non-negative.

    Parameters
    ----------
    y_data : astropy.units.Quantity, shape (n_rows, n_grid)
    dx : astropy.units.Quantity, shape (n_grid - 1,)

    Returns
    -------
    integral : astropy.units.Quantity, shape (n_rows,)
        Integral of each row of y_data with respect to x_data. Has units of y_data.unit * x_data.unit.
    """
    result = loglinear_integral_2d_nonnegative(y_data.value, dx.value)
    return result << (y_data.unit * dx.unit)


def loglinear_normalise_quantity_1d(
        y_data: u.Quantity,
        dx: u.Quantity,
) -> u.Quantity:
    """

    Parameters
    ----------
    y_data : astropy.units.Quantity, shape (n_grid,)
    dx : astropy.units.Quantity, shape (n_grid - 1,)

    Returns
    -------
    normalised : astropy.units.Quantity, shape (n_grid,)
        y_data / integral(y_data, x_data). Units of 1/x_data.unit.
        Returned unchanged if the integral is zero.
    """
    result = loglinear_normalise_1d(y_data.value, dx.value)
    return result << (1 / dx.unit)


def loglinear_normalise_quantity_1d_nonnegative(
        y_data: u.Quantity,
        dx: u.Quantity,
) -> u.Quantity:
    """
    All y_data must be non-negative.

    Parameters
    ----------
    y_data : astropy.units.Quantity, shape (n_grid,)
    dx : astropy.units.Quantity, shape (n_grid - 1,)

    Returns
    -------
    normalised : astropy.units.Quantity, shape (n_grid,)
        y_data / integral(y_data, x_data). Units of 1/x_data.unit.
        Returned unchanged if the integral is zero.
    """
    result = loglinear_normalise_1d(y_data.value, dx.value)
    return result << (1 / dx.unit)


def loglinear_normalise_quantity_2d(
        y_data: u.Quantity,
        dx: u.Quantity,
) -> u.Quantity:
    """

    Parameters
    ----------
    y_data : astropy.units.Quantity, shape (n_rows, n_grid)
    dx : astropy.units.Quantity, shape (n_grid - 1,)

    Returns
    -------
    normalised : astropy.units.Quantity, shape (n_rows, n_grid)
        Each row divided by its integral; units are 1 / x_data.unit.
    """
    result = loglinear_normalise_2d(y_data.value, dx.value)
    return result << (1 / dx.unit)


def loglinear_normalise_quantity_2d_nonnegative(
        y_data: u.Quantity,
        dx: u.Quantity,
) -> u.Quantity:
    """
    All y_data must be non-negative.

    Parameters
    ----------
    y_data : astropy.units.Quantity, shape (n_rows, n_grid)
    dx : astropy.units.Quantity, shape (n_grid - 1,)

    Returns
    -------
    normalised : astropy.units.Quantity, shape (n_rows, n_grid)
        Each row divided by its integral; units are 1 / x_data.unit.
    """
    result = loglinear_normalise_2d_nonnegative(y_data.value, dx.value)
    return result << (1 / dx.unit)


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
