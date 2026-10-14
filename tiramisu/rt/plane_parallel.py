"""Plane-parallel radiative transfer: the same directions at every layer. Bezier control points for
the whole column, the Gauss-Seidel layer updates, the constant-S formal solution of 4 pi J and the
emergent flux."""
import logging
import typing as t

import numpy as np
import numba
from numpy import typing as npt
from astropy import units as u

from .bezier import bezier_terms, ray_control_point_inward, ray_control_point_outward

log = logging.getLogger(__name__)


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _compute_control_points_outward(
        tau_matrix: npt.NDArray[np.float64],
        source_function_matrix: npt.NDArray[np.float64],
        control_points: npt.NDArray[np.float64],
        start: int = 0,
        end: int | None = None,
) -> None:
    """
    Compute outward control points (index 0) of control_points, shape (n_layers, 2, n_wavelengths).

    Control points do not depend on the ray angle: along a ray with direction cosine mu every optical depth difference
    is dtau / mu, so the source-function derivative estimate scales as mu and each control point
    S + 0.5 * (dtau / mu) * dS/dtau_mu is independent of mu; the clamps only compare source function values and test
    the sign of dtau. They are therefore computed from the vertical optical depth and shared by all angles.
    """
    n_layers, n_wavelengths = tau_matrix.shape
    if end is None or end > n_layers - 1:
        end = n_layers - 1

    # derivative indices required: i in [start-2, end).
    deriv_start = max(start - 2, 1)
    deriv_end = min(end, n_layers - 1)

    for k in numba.prange(n_wavelengths):
        # Compute derivatives
        d_s_d_tau_out = np.zeros(n_layers, dtype=np.float64)

        for i in range(deriv_start, deriv_end):
            tau_diff = tau_matrix[i, k] - tau_matrix[i + 1, k]
            if tau_diff != 0:
                d_diff = (source_function_matrix[i, k] - source_function_matrix[i + 1, k]) / tau_diff

                if i > 0:
                    tau_diff_prev = tau_matrix[i - 1, k] - tau_matrix[i, k]
                    if tau_diff_prev != 0:
                        d_diff_prev = (source_function_matrix[i - 1, k] - source_function_matrix[i, k]) / tau_diff_prev
                    else:
                        d_diff_prev = 0

                    zeta_denom = tau_matrix[i - 1, k] - tau_matrix[i + 1, k]
                    if zeta_denom != 0:
                        zeta = (1 + (tau_matrix[i - 1, k] - tau_matrix[i, k]) / zeta_denom) / 3
                    else:
                        zeta = 1.0 / 3.0

                    numerator = d_diff * d_diff_prev
                    denominator = zeta * d_diff_prev + (1 - zeta) * d_diff

                    if numerator >= 0 and denominator != 0:
                        d_s_d_tau_out[i] = numerator / denominator

        # Compute control points with clamping
        for i in range(max(start, 1), min(end + 1, n_layers)):
            tau_diff = tau_matrix[i - 1, k] - tau_matrix[i, k]

            control_0 = source_function_matrix[i, k] + 0.5 * tau_diff * d_s_d_tau_out[i]
            control_1 = source_function_matrix[i - 1, k] - 0.5 * tau_diff * d_s_d_tau_out[i - 1]

            min_source = min(source_function_matrix[i - 1, k], source_function_matrix[i, k])
            max_source = max(source_function_matrix[i - 1, k], source_function_matrix[i, k])

            control_0 = max(min(control_0, max_source), min_source)
            control_1 = max(min(control_1, max_source), min_source)

            if i == 1:
                control_points[i, 0, k] = control_1
            else:
                control_points[i, 0, k] = 0.5 * (control_0 + control_1)

            # Clamp to non-negative if gamma > 0; sign of gamma*C must be +. gamma of boundary i (coefficients[i, 3])
            # is positive at every angle exactly when that boundary's dtau = tau[i - 1] - tau[i] is.
            if tau_diff > 0 > control_points[i, 0, k]:
                # TODO: Check gamma index offset.
                control_points[i, 0, k] = 0.0
    # End.


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _compute_control_points_inward(
        tau_matrix: npt.NDArray[np.float64],
        source_function_matrix: npt.NDArray[np.float64],
        control_points: npt.NDArray[np.float64],
        start: int = 0,
        end: int | None = None,
) -> None:
    """Compute inward control points (index 1). Angle-independent; see :func:`_compute_control_points_outward`."""
    n_layers, n_wavelengths = tau_matrix.shape

    if end is None or end > n_layers - 1:
        end = n_layers - 1

    # derivative indices required: i in [start, end+2).
    deriv_start = start
    deriv_end = min(end + 2, n_layers - 1)

    for k in numba.prange(n_wavelengths):
        # Compute derivatives
        d_s_d_tau_in = np.zeros(n_layers, dtype=np.float64)

        for i in range(deriv_start, deriv_end):
            tau_diff = tau_matrix[i + 1, k] - tau_matrix[i, k]
            if tau_diff != 0:
                d_diff = (source_function_matrix[i + 1, k] - source_function_matrix[i, k]) / tau_diff

                if i < n_layers - 2:
                    tau_diff_next = tau_matrix[i + 2, k] - tau_matrix[i + 1, k]
                    if tau_diff_next != 0:
                        d_diff_next = (
                                (source_function_matrix[i + 2, k] - source_function_matrix[i + 1, k]) / tau_diff_next
                        )
                    else:
                        d_diff_next = 0

                    zeta_denom = tau_matrix[i + 2, k] - tau_matrix[i, k]
                    if zeta_denom != 0:
                        zeta = (1 + (tau_matrix[i + 2, k] - tau_matrix[i + 1, k]) / zeta_denom) / 3
                    else:
                        zeta = 1.0 / 3.0

                    numerator = d_diff * d_diff_next
                    denominator = zeta * d_diff_next + (1 - zeta) * d_diff

                    if numerator >= 0 and denominator != 0:
                        d_s_d_tau_in[i + 1] = numerator / denominator

        # Compute control points with clamping
        for i in range(max(start, 0), min(end + 1, n_layers - 1)):
            tau_diff = tau_matrix[i + 1, k] - tau_matrix[i, k]

            control_0 = source_function_matrix[i, k] + 0.5 * tau_diff * d_s_d_tau_in[i]
            control_1 = source_function_matrix[i + 1, k] - 0.5 * tau_diff * d_s_d_tau_in[i + 1]

            min_source = min(source_function_matrix[i, k], source_function_matrix[i + 1, k])
            max_source = max(source_function_matrix[i, k], source_function_matrix[i + 1, k])

            control_0 = max(min(control_0, max_source), min_source)
            control_1 = max(min(control_1, max_source), min_source)

            if i == n_layers - 2:
                control_points[i, 1, k] = control_1
            else:
                control_points[i, 1, k] = 0.5 * (control_0 + control_1)

            # Clamp to non-negative if gamma > 0; sign of gamma*C must be +. gamma of boundary i (coefficients[i, 3])
            # is positive at every angle exactly when that boundary's dtau = tau[i - 1] - tau[i] is.
            if i > 0 > control_points[i, 1, k] and tau_matrix[i - 1, k] - tau_matrix[i, k] > 0:
                # TODO: Check gamma index offset.
                control_points[i, 1, k] = 0
    # End.


def bezier_delta_tau_and_control_points(
        tau_matrix: npt.NDArray[np.float64],
        source_function_matrix: u.Quantity,
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Layer-boundary optical depths and control points for the Bezier formal solution.

    The Bezier coefficients exp(-dtau/mu), alpha, beta and gamma are not stored: each is a closed-form function of
    dtau / mu (:func:`bezier_terms`), computed by :func:`update_inward_layer`/:func:`update_outward_layer` where
    they are used. Storing them for every (boundary, angle, wavenumber) took 4 + 2 derived arrays of shape
    (n_layers + 1, n_angles, n_wavelengths) (~11 GB at 100 layers, 50 angles, R = 15k) and was slower overall than
    recomputing them in the passes.

    Parameters
    ----------
    tau_matrix : ndarray
        Optical depth [n_layers, n_wavelengths].
    source_function_matrix : Quantity
        Source function [n_layers, n_wavelengths].

    Returns
    -------
    delta_tau : ndarray, shape (n_layers + 1, n_wavelengths)
        delta_tau[i] = tau[i - 1] - tau[i] for 1 <= i < n_layers; delta_tau[0] = -tau[0], delta_tau[n_layers] =
        tau[n_layers - 1] (boundary rows).
    control_points : ndarray, shape (n_layers, 2, n_wavelengths)
        The same for every angle (see :func:`_compute_control_points_outward`).
    """
    n_layers, n_wavelengths = tau_matrix.shape
    tau_matrix = np.ascontiguousarray(tau_matrix)
    source_func_val = np.ascontiguousarray(source_function_matrix.value)

    delta_tau = np.zeros((n_layers + 1, n_wavelengths), dtype=np.float64)
    delta_tau[1:, :] = tau_matrix
    delta_tau[:-1, :] -= tau_matrix

    control_points = np.zeros((n_layers, 2, n_wavelengths), dtype=np.float64)
    _compute_control_points_outward(
        tau_matrix=tau_matrix, source_function_matrix=source_func_val, control_points=control_points,
    )
    _compute_control_points_inward(
        tau_matrix=tau_matrix, source_function_matrix=source_func_val, control_points=control_points,
    )
    return delta_tau, control_points


@numba.njit(parallel=True, cache=True, error_model="numpy")
def update_layer_coefficients(
        layer_idx: int,
        tau_matrix: npt.NDArray[np.float64],
        source_function_matrix: npt.NDArray[np.float64],
        delta_tau: npt.NDArray[np.float64],
        control_points: npt.NDArray[np.float64]
) -> None:
    """
    Refresh delta_tau and the control points for the one or two layer boundaries touched by ``layer_idx``, after that
    layer's opacity or source function changed.
    """
    n_layers, n_wavelengths = tau_matrix.shape

    update_idxs = []
    if layer_idx > 0:
        update_idxs.append(layer_idx)
    if layer_idx < n_layers - 1:
        update_idxs.append(layer_idx + 1)

    for k in numba.prange(n_wavelengths):
        for i in update_idxs:
            delta_tau[i, k] = tau_matrix[i - 1, k] - tau_matrix[i, k]

    # Recompute control_points[i, 0] and control_points[i, 1] for i in range [layer_idx-2 ... layer_idx+2]
    min_update_idx = min(update_idxs)
    max_update_idx = max(update_idxs)
    # i_low and i_high are the inclusive indices for updating; looping must account for range bounds.
    i_low = max(min_update_idx - 1, 0)
    i_high = min(max_update_idx + 1, n_layers - 1)

    _compute_control_points_outward(
        tau_matrix=tau_matrix, source_function_matrix=source_function_matrix, control_points=control_points,
        start=i_low, end=i_high,
    )
    _compute_control_points_inward(
        tau_matrix=tau_matrix, source_function_matrix=source_function_matrix, control_points=control_points,
        start=i_low, end=i_high,
    )
    # Done.


@numba.njit(parallel=True, cache=True, error_model="numpy")
def update_inward_layer(
        layer_idx: int,
        i_in_matrix: npt.NDArray[np.float64],
        lambda_in_layer: npt.NDArray[np.float64],
        delta_tau: npt.NDArray[np.float64],
        mu_values: npt.NDArray[np.float64],
        control_points: npt.NDArray[np.float64],
        source_function_matrix: npt.NDArray[np.float64],
        with_lambda: bool = True,
) -> None:
    """
    Update ``i_in_matrix[layer_idx]`` in place during a Gauss-Seidel inward pass:

        I_in[l] = I_in[l + 1] * exp(-dt) + alpha * S[l] + beta * S[l + 1] + gamma * C_in[l]

    with the Bezier terms of boundary l + 1 computed here from delta_tau and mu (:func:`bezier_terms`), and write this
    layer's inward Lambda* into ``lambda_in_layer`` (n_angles, n_grid). ``i_in_matrix`` keeps every layer, as the
    outward pass needs the inward intensity above each layer it reaches; Lambda* is only ever used at the current layer.

    Lambda* is the exact dI_in[l] / dS[l] (see :func:`_ray_inward_step`): the control points and their derivatives do
    not depend on the angle, so the derivatives are taken once per wavenumber, along a vertical ray whose steps are
    ``delta_tau`` itself. ``with_lambda`` False leaves ``lambda_in_layer`` alone, for calls whose Lambda* is not used
    (the inward pass above the first layer solved, LTE layers).
    """
    n_layers = source_function_matrix.shape[0]
    n_angles, n_grid = i_in_matrix.shape[1], i_in_matrix.shape[2]
    vertical_a, vertical_b = np.zeros((n_layers, 1)), np.ones((n_layers, 1))

    for k in numba.prange(n_grid):
        sf_here = source_function_matrix[layer_idx, k]
        sf_next = source_function_matrix[layer_idx + 1, k]
        control_in = control_points[layer_idx, 1, k]
        dtau_next = delta_tau[layer_idx + 1, k]
        if with_lambda:
            d_control_in = ray_control_point_inward(
                source_function_matrix, delta_tau, vertical_a, vertical_b, -1, layer_idx, 0, k, layer_idx)[1]
            has_prev = layer_idx + 2 < n_layers
            d_control_prev = ray_control_point_inward(
                source_function_matrix, delta_tau, vertical_a, vertical_b, -1, layer_idx + 1, 0, k, layer_idx
            )[1] if has_prev else 0.0

        for j in range(n_angles):
            exp_next, alpha_next, beta_next, gamma_next = bezier_terms(dtau_next / mu_values[j])
            i_in_matrix[layer_idx, j, k] = (
                    i_in_matrix[layer_idx + 1, j, k] * exp_next
                    + alpha_next * sf_here
                    + beta_next * sf_next
                    + gamma_next * control_in
            )
            if with_lambda:
                gamma_prev = bezier_terms(delta_tau[layer_idx + 2, k] / mu_values[j])[3] if has_prev else 0.0
                lambda_in_layer[j, k] = (alpha_next + gamma_next * d_control_in
                                         + exp_next * gamma_prev * d_control_prev)


@numba.njit(parallel=True, cache=True, error_model="numpy")
def update_outward_layer(
        layer_idx: int,
        i_out_below: npt.NDArray[np.float64],
        i_out_layer: npt.NDArray[np.float64],
        lambda_out_layer: npt.NDArray[np.float64],
        delta_tau: npt.NDArray[np.float64],
        mu_values: npt.NDArray[np.float64],
        control_points: npt.NDArray[np.float64],
        source_function_matrix: npt.NDArray[np.float64],
        with_lambda: bool = True,
) -> None:
    """
    Outward intensity and Lambda* at ``layer_idx`` during a Gauss-Seidel outward pass, written into ``i_out_layer`` and
    ``lambda_out_layer`` (n_angles, n_grid) from the layer below's outward intensity ``i_out_below``: the outward pass
    only ever needs these two layers.

    See :func:`update_inward_layer`; this is the mirror-image formula for the outward direction. Assumes
    ``layer_idx > 0`` - the ``layer_idx == 0`` boundary condition (blackbody, zero Lambda*) is the caller's
    responsibility, same as before. ``with_lambda`` as for :func:`update_inward_layer`.
    """
    n_layers = source_function_matrix.shape[0]
    n_angles, n_grid = i_out_layer.shape
    vertical_a, vertical_b = np.zeros((n_layers, 1)), np.ones((n_layers, 1))

    for k in numba.prange(n_grid):
        sf_here = source_function_matrix[layer_idx, k]
        sf_prev = source_function_matrix[layer_idx - 1, k]
        control_out = control_points[layer_idx, 0, k]
        dtau_here = delta_tau[layer_idx, k]
        if with_lambda:
            d_control_out = ray_control_point_outward(
                source_function_matrix, delta_tau, vertical_a, vertical_b, -1, layer_idx, 0, k, layer_idx)[1]
            has_prev = layer_idx >= 2
            d_control_prev = ray_control_point_outward(
                source_function_matrix, delta_tau, vertical_a, vertical_b, -1, layer_idx - 1, 0, k, layer_idx
            )[1] if has_prev else 0.0

        for j in range(n_angles):
            exp_here, alpha_here, beta_here, gamma_here = bezier_terms(dtau_here / mu_values[j])
            i_out_layer[j, k] = (
                    i_out_below[j, k] * exp_here
                    + alpha_here * sf_here
                    + beta_here * sf_prev
                    + gamma_here * control_out
            )
            if with_lambda:
                gamma_prev = bezier_terms(delta_tau[layer_idx - 1, k] / mu_values[j])[3] if has_prev else 0.0
                lambda_out_layer[j, k] = (alpha_here + gamma_here * d_control_out
                                          + exp_here * gamma_prev * d_control_prev)


def _constant_s_step(
        i_start: u.Quantity, dtau_layer: npt.NDArray[np.float64], source_layer: u.Quantity,
        mu_values: npt.NDArray[np.float64],
) -> u.Quantity:
    """One layer of the constant-S formal solution along every direction: I e^(-dtau/mu) + S (1 - e^(-dtau/mu))."""
    exp_term = np.exp(-dtau_layer / mu_values[:, None])
    return i_start * exp_term + source_layer * (1 - exp_term)


def four_pi_j_hemispheres(
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
    each hemisphere over solid angle to obtain its contribution to the angle-integrated mean intensity
    :math:`4\\pi J`. These are **not** fluxes (no :math:`\\mu` weighting).

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

    Each layer's source function :math:`S_{k}` is its layer-centre value, held across the layer's full
    :math:`\\Delta\\tau_{k}`. Intensities are stored at the **interfaces**, so the output arrays have dimension
    ``n_layers + 1``; values at layer centres are the mean of the two bounding interfaces.

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

    After computing intensities for each :math:`\\mu`, each hemisphere is integrated over solid angle:

    .. math::
        (4\\pi J)^{\\pm}(\\tilde{\\nu})
        = 2\\pi \\int_{0}^{1} I^{\\pm}(\\mu) \\, \\mathrm{d}\\mu
        \\approx 2\\pi \\sum_{i=1}^{n_{\\mu}} I^{\\pm}_i(\\tilde{\\nu}) \\, w_{i},

    where :math:`w_{i}` are the angular quadrature weights on :math:`[0, 1]` (summing to 1), so that
    :math:`(4\\pi J)^{+} + (4\\pi J)^{-} = 4\\pi J`. The hemispheric *fluxes* would instead need the extra cosine
    factor, :math:`F^{\\pm} = 2\\pi \\sum_i I^{\\pm}_i \\mu_i w_i`, which this routine does not compute.

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
    incident_radiation_field : Quantity, shape (n_wn,) or (n_mu, n_wn), optional
        Downward incident *specific intensity* at TOA, :math:`I^{-}(\\tau = 0, \\mu)`, applied to every downward
        direction when given per wavenumber only; defaults to zero.
    surface_albedo : float, optional
        Surface albedo :math:`A \\in [0,1]`. If nonzero, reflection modifies the BOA upward intensity. (Current
        implementation uses a simplified placeholder.)

    Returns
    -------
    four_pi_j_up : Quantity, shape (n_layers + 1, n_wn)
        Upward hemisphere's contribution to :math:`4\\pi J` at each interface:

        .. math::
            (4\\pi J)^{+}(\\tilde{\\nu}) = 2\\pi \\sum_{i} I^{+}_{i}(\\tilde{\\nu}) w_{i}.

    four_pi_j_down : Quantity, shape (n_layers + 1, n_wn)
        Downward hemisphere's contribution to :math:`4\\pi J` at each interface:

        .. math::
            (4\\pi J)^{-}(\\tilde{\\nu}) = 2\\pi \\sum_{i} I^{-}_{i}(\\tilde{\\nu}) w_{i}.
    """
    if surface_albedo < 0 or surface_albedo > 1:
        log.warning(f"Surface albedo {surface_albedo} is outside of [0, 1], clipping.")
        surface_albedo = np.clip(surface_albedo, 0, 1)

    n_layers, n_wavelengths = dtau.shape

    # Compute intensity at interfaces.
    i_up = np.zeros((len(mu_values), n_layers + 1, n_wavelengths)) << source_function.unit
    i_down = np.zeros((len(mu_values), n_layers + 1, n_wavelengths)) << source_function.unit

    # Upper boundary condition at the top (level n_layers) is zero, unless incident radiation field!
    if incident_radiation_field is not None:
        i_down[:, n_layers, :] = incident_radiation_field
    else:
        i_down[:, n_layers, :] = 0.0 * source_function.unit

    # Integrate from TOA (k=n_layers-1) down to BOA (k=0)
    for k in range(n_layers - 1, -1, -1):
        # Intensity at the bottom interface of each layer
        i_down[:, k, :] = _constant_s_step(
            i_start=i_down[:, k + 1, :], dtau_layer=dtau[k, :], source_layer=source_function[k, :], mu_values=mu_values
        )

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
        # Intensity at the top of the layer (level k+1)
        i_up[:, k + 1, :] = _constant_s_step(
            i_start=i_up[:, k, :], dtau_layer=dtau[k, :], source_layer=source_function[k, :], mu_values=mu_values
        )

    four_pi_j_up = 2 * np.pi * u.sr * np.sum(i_up * mu_weights[:, None, None], axis=0)
    four_pi_j_down = 2 * np.pi * u.sr * np.sum(i_down * mu_weights[:, None, None], axis=0)

    return four_pi_j_up, four_pi_j_down


def emergent_flux(
        dtau: npt.NDArray[np.float64],
        source_function: u.Quantity,
        mu_values: npt.NDArray[np.float64],
        mu_weights: npt.NDArray[np.float64],
) -> u.Quantity:
    """
    Emergent (upward) flux at the top of the atmosphere, the observable for an emission spectrum:

    .. math::
        F^{+}(\\tilde{\\nu}) = 2\\pi \\int_{0}^{1} I^{+}(\\mu) \\, \\mu \\, \\mathrm{d}\\mu
        \\approx 2\\pi \\sum_{i} w_{i} \\, \\mu_{i} \\, I^{+}_{i}(\\tilde{\\nu}).

    Uses the same constant-S upward solution, lower boundary (:math:`I^{+} = S_{0}`) and angular grid as
    :func:`four_pi_j_hemispheres`, but keeps only the running intensity, so memory is ``O(n_mu * n_wn)``. For a
    horizontally uniform planet, the flux received at distance :math:`d` is :math:`(R_p / d)^{2} F^{+}`: the integral
    over directions at one point equals the integral of the emergent intensity over the visible disk.

    Parameters
    ----------
    dtau : ndarray, shape (n_layers, n_wn)
        Optical-depth increment per layer, bottom layer first.
    source_function : Quantity, shape (n_layers, n_wn)
        Layer-centre source function per layer.
    mu_values : ndarray, shape (n_mu,)
        Direction cosines on (0, 1].
    mu_weights : ndarray, shape (n_mu,)
        Quadrature weights on [0, 1] (summing to 1).

    Returns
    -------
    flux_up : Quantity, shape (n_wn,)
        Emergent flux :math:`F^{+}` at the top of the atmosphere [W / (m^2 Hz)].
    """
    i_up = np.ones((len(mu_values), 1)) * source_function[0]
    for k in range(dtau.shape[0]):
        i_up = _constant_s_step(i_start=i_up, dtau_layer=dtau[k], source_layer=source_function[k], mu_values=mu_values)
    return (2 * np.pi * u.sr * np.sum((mu_weights * mu_values)[:, None] * i_up, axis=0)).to(u.W / (u.m ** 2 * u.Hz))
