r"""
Bézier short characteristics along one ray: the step terms, the optical path between layer centres and the control
points, with the control points' derivatives with respect to the source function for the exact :math:`\Lambda^*`
(notes/spherical_2d_radiative_transfer.tex, Sec. 2; notes/lambda_star.tex, Sec. 5; :doc:`/theory/radiative_transfer`).

Layers :math:`l = 0, \dots, N-1` are counted from the bottom and :math:`S_l` is the source function at the centre of
layer :math:`l`. A segment of a ray from a departure point :math:`d` to an arrival point :math:`a`, an optical path
:math:`\delta` apart, advances the intensity by

.. math::

    I_a = I_d \, e^{-\delta} + \alpha(\delta) \, S_a + \beta(\delta) \, S_d + \gamma(\delta) \, C,

with :math:`C` the segment's control point. A ray is given by its segment weights ``seg_a`` and ``seg_b``
(:func:`ray_dtau`): spherical rays come from :func:`tiramisu.rt.spherical.spherical_rays`, and a plane-parallel column
is the vertical ray :math:`a_l = 0`, :math:`b_l = 1` with the vertical steps :math:`\tau_{l-1} - \tau_l` in place of
``dtau``. A ``tangent`` layer :math:`t \ge 0` marks a ray that turns there; core and plane-parallel rays have :math:`t =
-1`.
"""
import typing as t

import numpy as np
import numba
from numpy import typing as npt


@numba.njit(cache=True, error_model="numpy", inline="always")
def bezier_terms(dt: float) -> t.Tuple[float, float, float, float]:
    r"""
    The Bézier terms :math:`e^{-\delta}`, :math:`\alpha(\delta)`, :math:`\beta(\delta)` and :math:`\gamma(\delta)` of
    one segment of optical path :math:`\delta` (``dt``; :math:`\Delta\tau/\mu` in plane-parallel geometry), de la Cruz
    Rodríguez & Piskunov (2013, doi:10.1088/0004-637X/764/1/33):

    .. math::

        \alpha(\delta) = \frac{2 + \delta^2 - 2\delta - 2e^{-\delta}}{\delta^2}, \qquad
        \beta(\delta) = \frac{2 - (2 + 2\delta + \delta^2) \, e^{-\delta}}{\delta^2}, \qquad
        \gamma(\delta) = \frac{2\delta - 4 + (2\delta + 4) \, e^{-\delta}}{\delta^2}.

    They multiply the arrival point's :math:`S_a`, the departure point's :math:`S_d` and the control point. Below
    :math:`\delta = 0.14` the closed forms lose precision to cancellation and their Taylor series (Horner form) are used
    instead; at :math:`\delta = 0` all three are zero.
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
    r"""
    Optical path along ``ray`` (impact parameter :math:`p`) between the centres of layers :math:`l - 1` and :math:`l`
    (:math:`l` = ``point``), from the layers' optical thicknesses :math:`\Delta\tau` (``dtau[:, k]``):

    .. math::

        \Delta\tau_l(p) = a_l(p) \, \Delta\tau_{l-1} + b_l(p) \, \Delta\tau_l,

    with :math:`a_l` = ``seg_a[l, ray]`` and :math:`b_l` = ``seg_b[l, ray]`` the ray's paths through the upper half of
    layer :math:`l - 1` and the lower half of layer :math:`l` over those layers' thicknesses
    (:func:`tiramisu.rt.spherical.spherical_rays`).
    """
    return seg_a[point, ray] * dtau[point - 1, k] + seg_b[point, ray] * dtau[point, k]


@numba.njit(cache=True, error_model="numpy", inline="always")
def _ray_source_derivative(
        source_function_matrix: npt.NDArray[np.float64], dtau: npt.NDArray[np.float64],
        seg_a: npt.NDArray[np.float64], seg_b: npt.NDArray[np.float64], tangent: int,
        point: int, ray: int, k: int, inward: bool, seed: int,
) -> t.Tuple[float, float, float]:
    """
    Source-function derivative at ``point`` along ``ray``, as in ``plane_parallel._compute_control_points_outward``
    (``inward`` False) and ``_compute_control_points_inward`` (True), with the ray's own optical depths: the monotone
    harmonic mean of :func:`ray_control_point_inward`, always as dS/dtau with tau increasing inwards (``inward`` only
    chooses which neighbour is upwind in zeta), so outward control points enter it with the opposite sign. Zero at the
    ray's ends and at its tangent point, where the ray is symmetric about its closest approach.

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
    r"""
    Control point :math:`C^+_l` of the inward segment from layer centre :math:`l + 1` (departure :math:`d`) to :math:`l`
    (arrival :math:`a`, :math:`l` = ``layer_idx``) along ``ray``, at wavenumber index ``k``, and its derivative
    :math:`\partial C^+_l / \partial S_x` with respect to the source function of layer :math:`x` = ``seed`` (0 for
    ``seed = -1``, the value alone).

    With :math:`\delta = \Delta\tau_{l+1}(p)` the segment's optical path (:func:`ray_dtau`),

    .. math::

        C = \tfrac12 \left( C_a + C_d \right), \qquad C_a = S_a - \frac{\delta}{2} S'_a, \qquad
        C_d = S_d + \frac{\delta}{2} S'_d,

    each of :math:`C_a` and :math:`C_d` clipped to :math:`[\min(S_a, S_d), \max(S_a, S_d)]`. :math:`S' =
    \mathrm{d}S/\mathrm{d}t` is taken along the direction of propagation; at a point :math:`c` with upwind neighbour
    :math:`u` and downwind neighbour :math:`d` (here a point of the ray, not the segment's departure point), separated
    from them by the ray's optical paths :math:`\delta_u` and :math:`\delta_d`, it is the monotone harmonic mean

    .. math::

        S'_c = \begin{cases}
            \dfrac{s_u \, s_d}{\zeta s_u + (1 - \zeta) s_d} & s_u s_d > 0, \\[6pt]
            0 & \text{otherwise,}
        \end{cases}
        \qquad s_u = \frac{S_c - S_u}{\delta_u}, \quad s_d = \frac{S_d - S_c}{\delta_d}, \quad
        \zeta = \frac13 \left( 1 + \frac{\delta_u}{\delta_u + \delta_d} \right),

    and :math:`S' = 0` at the ray's end points and at its tangent point. The first segment below the top
    (:math:`l = N - 2`) uses :math:`C = C_d`, and :math:`C < 0` is set to zero where the next segment down has positive
    optical path.

    The derivative is taken in forward mode through the same branches:

    .. math::

        \frac{\partial C_a}{\partial S_x} = [a = x] - \frac{\delta}{2} \frac{\partial S'_a}{\partial S_x}, \qquad
        \frac{\partial C_d}{\partial S_x} = [d = x] + \frac{\delta}{2} \frac{\partial S'_d}{\partial S_x}

    where unclipped (a clipped value has the derivative of the end point it equals, 1 or 0), with

    .. math::

        \frac{\partial S'_c}{\partial S_u} = -\frac{(1 - \zeta) s_d^2}{Z^2 \delta_u}, \qquad
        \frac{\partial S'_c}{\partial S_c} = \frac{(1 - \zeta) s_d^2}{Z^2 \delta_u} - \frac{\zeta s_u^2}{Z^2 \delta_d},
        \qquad \frac{\partial S'_c}{\partial S_d} = \frac{\zeta s_u^2}{Z^2 \delta_d}, \qquad
        Z = \zeta s_u + (1 - \zeta) s_d,

    where :math:`s_u s_d > 0` and zero otherwise. At a kink (a tie of :math:`S`, a clip or the non-negativity clamp)
    the one-sided derivatives differ; their mean is returned, as a central difference would give.

    Returns
    -------
    control, d_control : float
        :math:`C^+_l` and :math:`\partial C^+_l / \partial S_x`.
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
    """The optical path the inward non-negativity clamp tests, as in the plane-parallel kernel: the ray's next segment
    down (mirrored at a tangent point; none below the bottom layer)."""
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
    r"""
    Control point :math:`C^-_l` of the outward segment from layer centre :math:`l - 1` (departure :math:`d`) to
    :math:`l` (arrival :math:`a`, :math:`l` = ``layer_idx``) along ``ray``, and its derivative
    :math:`\partial C^-_l / \partial S_x` with respect to the source function of layer :math:`x` = ``seed``.

    The formulas are those of :func:`ray_control_point_inward` with :math:`\delta = \Delta\tau_l(p)` and :math:`S'`
    along the outward direction. The first segment above the bottom of a core ray (:math:`l = 1`, ``tangent < 0``)
    uses :math:`C = C_d`; a tangent ray's first outward segment starts at its turning point, where :math:`S' = 0`. The
    non-negativity clamp tests this segment's own optical path.

    Returns
    -------
    control, d_control : float
        :math:`C^-_l` and :math:`\partial C^-_l / \partial S_x`.
    """
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
