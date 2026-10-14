"""Bezier short-characteristics terms and the control points along one ray, with their derivatives with
respect to the source function for the exact Lambda* (notes/lambda_star.tex). A plane-parallel column is
the vertical ray of this machinery; spherical rays cross the shells."""
import typing as t

import numpy as np
import numba
from numpy import typing as npt


@numba.njit(cache=True, error_model="numpy", inline="always")
def bezier_terms(dt: float) -> t.Tuple[float, float, float, float]:
    """
    Bezier short-characteristics terms for one layer boundary along one ray: exp(-dt), alpha, beta, gamma, where
    dt = delta_tau / mu. Taylor expansion (Horner form) below dt = 0.14, exact formulas above.
    """
    if dt == 0.0:
        # alpha, beta, gamma tend to 0 as dt -> 0.
        return 1.0, 0.0, 0.0, 0.0
    if dt < 1.4e-1:
        alpha = dt * (dt * (
                dt * (dt * (dt * (dt * ((10 - dt) * dt - 90) + 720) - 5040) + 30240) - 151200) + 604800) / 1814400
        beta = dt * (dt * (dt * (
                dt * (dt * (dt * ((140 - 18 * dt) * dt - 945) + 5400) - 25200) + 90720) - 226800) + 302400) / 907200
        gamma = dt * (dt * (dt * (
                dt * (dt * (dt * ((35 - 4 * dt) * dt - 270) + 1800) - 10080) + 45360) - 151200) + 302400) / 907200
        return np.exp(-dt), alpha, beta, gamma
    exp_neg_tau = np.exp(-dt)
    dt_sq = dt * dt
    return (exp_neg_tau, (2 + dt_sq - 2 * dt - 2 * exp_neg_tau) / dt_sq,
            (2 - (2 + 2 * dt + dt_sq) * exp_neg_tau) / dt_sq,
            (2 * dt - 4 + (2 * dt + 4) * exp_neg_tau) / dt_sq)


@numba.njit(cache=True, error_model="numpy", inline="always")
def ray_dtau(
        dtau: npt.NDArray[np.float64], seg_a: npt.NDArray[np.float64], seg_b: npt.NDArray[np.float64],
        point: int, ray: int, k: int,
) -> float:
    """Optical depth along ``ray`` between the layer centres ``point - 1`` and ``point``."""
    return seg_a[point, ray] * dtau[point - 1, k] + seg_b[point, ray] * dtau[point, k]


@numba.njit(cache=True, error_model="numpy", inline="always")
def _ray_source_derivative(
        source_function_matrix: npt.NDArray[np.float64], dtau: npt.NDArray[np.float64],
        seg_a: npt.NDArray[np.float64], seg_b: npt.NDArray[np.float64], tangent: int,
        point: int, ray: int, k: int, inward: bool, seed: int,
) -> t.Tuple[float, float, float]:
    """
    Source-function derivative at ``point`` along ``ray``, as in :func:`_compute_control_points_outward` (``inward``
    False) and :func:`_compute_control_points_inward` (True), with the ray's own optical depths. Zero at the ray's ends
    and at its tangent point, where the ray is symmetric about its closest approach.

    Also returns its derivatives with respect to ``source_function_matrix[seed, k]`` for S[seed] increasing and
    decreasing (forward-mode, through the same branches), for the exact Lambda*. They differ only at a kink of the
    limiter, where one slope is exactly zero (next to a flat stretch of S, e.g. isothermal LTE layers).
    """
    n_layers = source_function_matrix.shape[0]
    if point < 1 or point > n_layers - 2 or point <= tangent:
        return 0.0, 0.0, 0.0
    dtau_lo = ray_dtau(dtau, seg_a, seg_b, point, ray, k)
    dtau_hi = ray_dtau(dtau, seg_a, seg_b, point + 1, ray, k)
    if (dtau_lo if inward else dtau_hi) == 0:
        return 0.0, 0.0, 0.0
    s_lo, s_here, s_hi = (source_function_matrix[point - 1, k], source_function_matrix[point, k],
                          source_function_matrix[point + 1, k])
    slope_lo = (s_lo - s_here) / dtau_lo if dtau_lo != 0 else 0.0
    slope_hi = (s_here - s_hi) / dtau_hi if dtau_hi != 0 else 0.0
    total = dtau_lo + dtau_hi
    if total != 0:
        zeta = (1 + (dtau_hi if inward else dtau_lo) / total) / 3
    else:
        zeta = 1.0 / 3.0
    numerator = slope_lo * slope_hi
    denominator = zeta * slope_hi + (1 - zeta) * slope_lo if inward else zeta * slope_lo + (1 - zeta) * slope_hi
    if numerator < 0 or denominator == 0:
        return 0.0, 0.0, 0.0
    value = numerator / denominator
    if seed < point - 1 or seed > point + 1:  # S[seed] is not in the stencil (seed = -1: the value only)
        return value, 0.0, 0.0
    # d/dS[seed] of s_lo, s_here, s_hi.
    ds_lo = 1.0 if seed == point - 1 else 0.0
    ds_here = 1.0 if seed == point else 0.0
    ds_hi = 1.0 if seed == point + 1 else 0.0
    d_slope_lo = (ds_lo - ds_here) / dtau_lo if dtau_lo != 0 else 0.0
    d_slope_hi = (ds_here - ds_hi) / dtau_hi if dtau_hi != 0 else 0.0
    d_numerator = d_slope_lo * slope_hi + slope_lo * d_slope_hi
    if inward:
        d_denominator = zeta * d_slope_hi + (1 - zeta) * d_slope_lo
    else:
        d_denominator = zeta * d_slope_lo + (1 - zeta) * d_slope_hi
    d_value = (d_numerator - value * d_denominator) / denominator
    if numerator > 0:
        return value, d_value, d_value
    # A kink: the formula holds on the side where the numerator turns positive, S' = 0 on the other.
    return value, d_value if d_numerator > 0 else 0.0, d_value if d_numerator < 0 else 0.0


@numba.njit(cache=True, error_model="numpy", inline="always")
def _above(x: float, dx: float, y: float, dy: float, step: float) -> bool:
    """Is x above y after a small change of S[seed] of sign ``step``, x and y changing at rates dx and dy?"""
    return x > y or (x == y and step * (dx - dy) > 0)


@numba.njit(cache=True, error_model="numpy", inline="always")
def _clip_rate(value: float, d_value: float, s_a: float, ds_a: float, s_b: float, ds_b: float, step: float) -> float:
    """One-sided derivative of ``max(min(value, max(s_a, s_b)), min(s_a, s_b))`` for a change of S[seed] of sign
    ``step``: a clipped value is an end point's S, and ties go the way the values move."""
    lo, d_lo, hi, d_hi = (s_b, ds_b, s_a, ds_a) if _above(s_a, ds_a, s_b, ds_b, step) else (s_a, ds_a, s_b, ds_b)
    if _above(value, d_value, hi, d_hi, step):
        return d_hi
    if _above(lo, d_lo, value, d_value, step):
        return d_lo
    return d_value


@numba.njit(cache=True, error_model="numpy", inline="always")
def _control_point_rates(
        control: float, d_up: float, d_down: float, dtau_next: float,
) -> t.Tuple[float, float]:
    """The non-negativity clamp of a control point (where ``dtau_next > 0``), with its one-sided derivatives; returns
    the control point and the mean of those, the Lambda* derivative (a central difference, at a kink)."""
    if dtau_next > 0:
        if _above(0.0, 0.0, control, d_up, 1.0):
            d_up = 0.0
        if _above(0.0, 0.0, control, d_down, -1.0):
            d_down = 0.0
        if control < 0:
            control = 0.0
    return control, 0.5 * (d_up + d_down)


@numba.njit(cache=True, error_model="numpy", inline="always")
def ray_control_point_inward(
        source_function_matrix: npt.NDArray[np.float64], dtau: npt.NDArray[np.float64],
        seg_a: npt.NDArray[np.float64], seg_b: npt.NDArray[np.float64], tangent: int,
        layer_idx: int, ray: int, k: int, seed: int = -1,
) -> t.Tuple[float, float]:
    """
    Inward control point of the segment from centre ``layer_idx + 1`` to ``layer_idx`` (see
    :func:`_compute_control_points_inward`), and its derivative with respect to ``source_function_matrix[seed, k]``:
    the mean of the one-sided ones, which differ only at a kink (a tie of S, a clip or the clamp).
    """
    n_layers = source_function_matrix.shape[0]
    dtau_seg = ray_dtau(dtau, seg_a, seg_b, layer_idx + 1, ray, k)
    s_here, s_hi = source_function_matrix[layer_idx, k], source_function_matrix[layer_idx + 1, k]
    ds_here, ds_hi = (1.0 if seed == layer_idx else 0.0), (1.0 if seed == layer_idx + 1 else 0.0)
    deriv_0, d_up_0, d_down_0 = _ray_source_derivative(
        source_function_matrix, dtau, seg_a, seg_b, tangent, layer_idx, ray, k, True, seed)
    deriv_1, d_up_1, d_down_1 = _ray_source_derivative(
        source_function_matrix, dtau, seg_a, seg_b, tangent, layer_idx + 1, ray, k, True, seed)
    control_0 = s_here - 0.5 * dtau_seg * deriv_0
    control_1 = s_hi + 0.5 * dtau_seg * deriv_1
    s_min, s_max = min(s_here, s_hi), max(s_here, s_hi)
    if seed < layer_idx - 1 or seed > layer_idx + 2:  # S[seed] is not in the stencil: the value only
        control_0 = max(min(control_0, s_max), s_min)
        control_1 = max(min(control_1, s_max), s_min)
        control = control_1 if layer_idx == n_layers - 2 else 0.5 * (control_0 + control_1)
        return _control_point_rates(control, 0.0, 0.0, _inward_clamp_dtau(dtau, seg_a, seg_b, tangent, layer_idx,
                                                                          ray, k, dtau_seg))
    up_0 = _clip_rate(control_0, ds_here - 0.5 * dtau_seg * d_up_0, s_here, ds_here, s_hi, ds_hi, 1.0)
    up_1 = _clip_rate(control_1, ds_hi + 0.5 * dtau_seg * d_up_1, s_here, ds_here, s_hi, ds_hi, 1.0)
    down_0 = _clip_rate(control_0, ds_here - 0.5 * dtau_seg * d_down_0, s_here, ds_here, s_hi, ds_hi, -1.0)
    down_1 = _clip_rate(control_1, ds_hi + 0.5 * dtau_seg * d_down_1, s_here, ds_here, s_hi, ds_hi, -1.0)
    control_0 = max(min(control_0, s_max), s_min)
    control_1 = max(min(control_1, s_max), s_min)
    if layer_idx == n_layers - 2:
        control, d_up, d_down = control_1, up_1, down_1
    else:
        control, d_up, d_down = 0.5 * (control_0 + control_1), 0.5 * (up_0 + up_1), 0.5 * (down_0 + down_1)
    return _control_point_rates(control, d_up, d_down, _inward_clamp_dtau(dtau, seg_a, seg_b, tangent, layer_idx, ray,
                                                                          k, dtau_seg))


@numba.njit(cache=True, error_model="numpy", inline="always")
def _inward_clamp_dtau(
        dtau: npt.NDArray[np.float64], seg_a: npt.NDArray[np.float64], seg_b: npt.NDArray[np.float64], tangent: int,
        layer_idx: int, ray: int, k: int, dtau_seg: float,
) -> float:
    """The step the inward non-negativity clamp tests, as in the plane-parallel kernel: the ray's next segment
    (mirrored at a tangent point)."""
    if layer_idx == tangent:
        return dtau_seg
    if layer_idx > 0:
        return ray_dtau(dtau, seg_a, seg_b, layer_idx, ray, k)
    return 0.0


@numba.njit(cache=True, error_model="numpy", inline="always")
def ray_control_point_outward(
        source_function_matrix: npt.NDArray[np.float64], dtau: npt.NDArray[np.float64],
        seg_a: npt.NDArray[np.float64], seg_b: npt.NDArray[np.float64], tangent: int,
        layer_idx: int, ray: int, k: int, seed: int = -1,
) -> t.Tuple[float, float]:
    """Outward control point of the segment from centre ``layer_idx - 1`` to ``layer_idx`` (see
    :func:`_compute_control_points_outward`), and its derivative; see :func:`ray_control_point_inward`."""
    dtau_seg = ray_dtau(dtau, seg_a, seg_b, layer_idx, ray, k)
    s_lo, s_here = source_function_matrix[layer_idx - 1, k], source_function_matrix[layer_idx, k]
    ds_lo, ds_here = (1.0 if seed == layer_idx - 1 else 0.0), (1.0 if seed == layer_idx else 0.0)
    deriv_0, d_up_0, d_down_0 = _ray_source_derivative(
        source_function_matrix, dtau, seg_a, seg_b, tangent, layer_idx, ray, k, False, seed)
    deriv_1, d_up_1, d_down_1 = _ray_source_derivative(
        source_function_matrix, dtau, seg_a, seg_b, tangent, layer_idx - 1, ray, k, False, seed)
    control_0 = s_here + 0.5 * dtau_seg * deriv_0
    control_1 = s_lo - 0.5 * dtau_seg * deriv_1
    s_min, s_max = min(s_lo, s_here), max(s_lo, s_here)
    if seed < layer_idx - 2 or seed > layer_idx + 1:  # S[seed] is not in the stencil: the value only
        control_0 = max(min(control_0, s_max), s_min)
        control_1 = max(min(control_1, s_max), s_min)
        control = control_1 if layer_idx == 1 and tangent < 0 else 0.5 * (control_0 + control_1)
        return _control_point_rates(control, 0.0, 0.0, dtau_seg)
    up_0 = _clip_rate(control_0, ds_here + 0.5 * dtau_seg * d_up_0, s_lo, ds_lo, s_here, ds_here, 1.0)
    up_1 = _clip_rate(control_1, ds_lo - 0.5 * dtau_seg * d_up_1, s_lo, ds_lo, s_here, ds_here, 1.0)
    down_0 = _clip_rate(control_0, ds_here + 0.5 * dtau_seg * d_down_0, s_lo, ds_lo, s_here, ds_here, -1.0)
    down_1 = _clip_rate(control_1, ds_lo - 0.5 * dtau_seg * d_down_1, s_lo, ds_lo, s_here, ds_here, -1.0)
    control_0 = max(min(control_0, s_max), s_min)
    control_1 = max(min(control_1, s_max), s_min)
    # The first segment above the BOA has no derivative below it; a tangent point's (zero) derivative is known.
    if layer_idx == 1 and tangent < 0:
        control, d_up, d_down = control_1, up_1, down_1
    else:
        control, d_up, d_down = 0.5 * (control_0 + control_1), 0.5 * (up_0 + up_1), 0.5 * (down_0 + down_1)
    return _control_point_rates(control, d_up, d_down, dtau_seg)
