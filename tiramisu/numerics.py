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
            # y0 roughly equals y1, i.e.: 0.5 * (y0 + y1) ~ y0
            result += dk * y0
        else:
            # If y1 = y0 * exp(lr), then (y1 - y0)/lr = y0 * (exp(lr) - 1) / lr; the latter is more stable for small lr.
            result += dk * y0 * np.expm1(lr) / lr
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
    result = loglinear_normalise_1d_nonnegative(y_data=y_data.value, dx=dx.value)
    return result << (1 / dx.unit)


# ----------------------------- LOG-LINEAR / LINEAR INTERPOLATION -----------------------------
# Interpolation counterparts to the log-linear quadrature above: log(y) varies linearly in x between nodes, i.e. the
# interpolant is piecewise exponential. Falls back to linear interpolation when either node is non-positive.

@numba.njit(cache=True, error_model="numpy", inline="always")
def _loglinear_interp_point(y0: float, y1: float, w: float) -> float:
    """
    Piecewise log-linear interpolation between two nodes.

    Evaluates y0 * (y1 / y0)**w, the exponential passing through (0, y0) and (1, y1). Falls back to linear
    interpolation when either node is non-positive (zero-padded or masked cross-sections).

    Parameters
    ----------
    y0 : float
        y data value at the lower node.
    y1 : float
        y data value at the upper node.
    w : float
        Fractional distance from the lower node, in [0, 1].

    Returns
    ----------
    float
    """
    if y0 <= 0.0 or y1 <= 0.0:
        return y0 + w * (y1 - y0)
    return y0 * np.exp(w * np.log(y1 / y0))


@numba.njit(parallel=True, cache=True, error_model="numpy")
def linear_loglinear_interp_2d(
        data: npt.NDArray[np.float64],
        idx_x0: npt.NDArray[np.int32],
        w_x: npt.NDArray[np.float64],
        idx_y0: npt.NDArray[np.int32],
        w_y: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Interpolate a (n_x, n_y, n_grid) table linearly along axis 0 and log-linearly along axis 1.

    For each output point the table is first interpolated log-linearly in y on the two bracketing x rows, and those
    results are then combined linearly in x. Bracketing indices and weights are precomputed by the caller, which
    handles clamping/extrapolation.

    Parameters
    ----------
    data : np.ndarray, shape (n_x, n_y, n_grid)
        Tabulated values (e.g. cross-sections over pressure, temperature, wavenumber).
    idx_x0 : np.ndarray, shape (n_points,)
        Lower bracketing index along axis 0; idx_x0 + 1 must be valid.
    w_x : np.ndarray, shape (n_points,)
        Fractional distance from idx_x0 towards idx_x0 + 1.
    idx_y0 : np.ndarray, shape (n_points,)
        Lower bracketing index along axis 1; idx_y0 + 1 must be valid.
    w_y : np.ndarray, shape (n_points,)
        Fractional distance from idx_y0 towards idx_y0 + 1.

    Returns
    -------
    np.ndarray, shape (n_points, n_grid)
    """
    n_points = idx_x0.shape[0]
    n_grid = data.shape[2]
    result = np.empty((n_points, n_grid))
    for i in numba.prange(n_points):
        ix0 = idx_x0[i]
        iy0 = idx_y0[i]
        wx = w_x[i]
        wy = w_y[i]
        for k in range(n_grid):
            lower = _loglinear_interp_point(data[ix0, iy0, k], data[ix0, iy0 + 1, k], wy)
            upper = _loglinear_interp_point(data[ix0 + 1, iy0, k], data[ix0 + 1, iy0 + 1, k], wy)
            result[i, k] = lower + wx * (upper - lower)
    return result
