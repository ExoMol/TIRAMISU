import logging
import abc
import typing as t
import numpy.typing as npt
import numpy as np
import pandas as pd
import polars as pl
import numba

from phoenix4all import get_spectrum

from astropy import units as u
from astropy import constants as ac

from scipy.integrate import simpson, cumulative_simpson

log = logging.getLogger(__name__)

# Constants with units:

ac_h_c_on_kB = ac.h * ac.c.cgs / ac.k_B
ac_2_hc = 2 * ac.h * ac.c.cgs

ac_2_h_on_c_sq = 2 * ac.h / ac.c ** 2
ac_h_on_kB = ac.h / ac.k_B

# Dimensionless version for numba
const_h_c_on_kB = ac_h_c_on_kB.value
const_2_hc = ac_2_hc.value
const_2_h_on_c_sq = ac_2_h_on_c_sq.value
const_h_on_kB = ac_h_on_kB.value
const_2_pi_h_c_sq_on_sigma_sba = (
    (2 * np.pi * ac.h * ac.c.cgs ** 2 / ac.sigma_sb).to(u.K ** 4 * u.cm ** 4, equivalencies=u.spectral()).value
)
const_2_pi_c_kB = (2 * np.pi * ac.c.cgs * ac.k_B.cgs).value


# TODO: For NANs in state lifetimes; treat as inf? They imply inf but often they exist because of transition energy
#  cutoffs during computation and not because the state has no deexcitation pathways.


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
                if coefficients[i, 3, j, k] > 0 > control_points[i, 0, j, k]:
                    # TODO: Check gamma index offset.
                    control_points[i, 0, j, k] = 0.0
    # End.


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
                if i > 0 > control_points[i, 1, j, k] and coefficients[i, 3, j, k] > 0:
                    # TODO: Check gamma index offset.
                    control_points[i, 1, j, k] = 0
    # End.


# @numba.njit(parallel=True, cache=True, error_model="numpy")
def bezier_coefficients(
        tau_mu_matrix: npt.NDArray[np.float64],
        source_function_matrix: u.Quantity,
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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
    _compute_control_points_outward(
        tau_mu_matrix=tau_mu_matrix,
        source_func_mu=source_func_mu,
        control_points=control_points,
        coefficients=coefficients
    )
    _compute_control_points_inward(
        tau_mu_matrix=tau_mu_matrix,
        source_func_mu=source_func_mu,
        control_points=control_points,
        coefficients=coefficients
    )

    return coefficients, control_points


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
        tau_mu_matrix=tau_mu_matrix, source_func_mu=source_func_mu, control_points=control_points,
        coefficients=coefficients, start=i_low, end=i_high,
    )
    _compute_control_points_inward(
        tau_mu_matrix=tau_mu_matrix, source_func_mu=source_func_mu, control_points=control_points,
        coefficients=coefficients, start=i_low, end=i_high,
    )
    # Done.


############# END NEW
# DEPRECATED BELOW:
@numba.njit(parallel=True, cache=True, error_model="numpy")
def bezier_coefficients_old(
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

    # Change indices on delta_tau_matrix based on direction! - Old comment.
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
        star_temperature: float, star_logg: float, star_feh: float, wn_grid: u.Quantity, orbital_radius: u.Quantity,
        star_radius: u.Quantity, star_alpha: float = 0.0
) -> u.Quantity:
    """
    Returns the substellar flux at the planet's surface.

    Parameters
    ----------
    star_temperature
    star_logg
    star_feh
    wn_grid
    orbital_radius
    star_radius
    star_alpha

    Returns
    -------
        Substellar flux [W /(m^2 cm^-1)].

    """
    srf_wavelength, srf_flux = get_spectrum(
        teff=star_temperature,
        logg=star_logg,
        feh=star_feh,
        alpha=star_alpha,
        source="synphot",
    )
    # srf_flux has units of erg / (Angstrom s cm^2)
    srf_wn = srf_wavelength.to(1 / u.cm, equivalencies=u.spectral())

    # srf_flux_wn = srf_flux / (srf_wn**2)
    # srf_flux_wn = srf_flux_wn.to(u.J / (u.s * u.m**2 * (1/u.cm)))

    srf_flux_nu = srf_flux.to(
        u.J / (u.s * u.m ** 2 * u.Hz),
        equivalencies=u.spectral_density(srf_wavelength)
    )

    sort_idx = np.argsort(srf_wn)
    srf_wn = srf_wn[sort_idx]
    srf_flux_nu = srf_flux_nu[sort_idx]

    srf_flux_interp = np.interp(wn_grid, srf_wn, srf_flux_nu, left=0, right=0) << srf_flux_nu.unit
    # This is F_nu [W/(Hz*m^2)]
    srf_flux_orbit = srf_flux_interp * (star_radius / orbital_radius) ** 2
    theta = np.arcsin((star_radius / orbital_radius).decompose().value)
    omega_star = 2 * np.pi * (1 - np.cos(theta)) * u.sr
    srf_specific_intensity = (srf_flux_orbit / omega_star).to(u.W / (u.Hz * u.m ** 2 * u.sr))
    return srf_specific_intensity


@numba.njit()
def calc_einstein_b_fi(a_fi: npt.NDArray[np.float64], energy_fi: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Here the Einstein B coefficient is given in :math:`\text{m}^2 / (\text{J·s})`.

    .. math::
        B_{fi}=\\frac{A_{fi}}{2hc\\tilde{\\nu}^{3}_{fi}}

    Parameters
    ----------
    a_fi
    energy_fi

    Returns
    -------

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
        Integer indices corresponding to the uniform sampling points in the CDF.

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
    temp_wn_grid = np.linspace(wn_start, wn_end, num_cdf_points, dtype=np.float64)
    ev_grid = calc_ev_grid(wn_grid=temp_wn_grid, temperature=np.atleast_1d(temperature_profile)[:, None]).sum(axis=0)
    ev_norm = ev_grid / simpson(ev_grid, x=temp_wn_grid)

    ev_cdf = cumulative_simpson(ev_norm, x=temp_wn_grid, initial=0)

    sample_idxs = _sample_indices(ev_cdf, temp_wn_grid, num_points, max_step)

    return temp_wn_grid[sample_idxs] / u.cm


def formal_solve_general(
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
    over angle to obtain the hemispheric fluxes.

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

    Intensities are stored at the **interfaces**, so the output arrays have dimension ``n_layers + 1``.

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

    After computing intensities for each :math:`\\mu`, hemispheric fluxes are computed as:

    .. math::
        F^{\\pm}(\\tilde{\\nu})
        = 2\\pi \\sum_{i=1}^{n_{\\mu}} I^{\\pm}_i(\\tilde{\\nu}) \\, w_{i},

    where :math:`w_{i}` are the angular quadrature weights.

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
    incident_radiation_field : Quantity, shape (n_mu, n_wn), optional
        Downward incident intensity at TOA, :math:`I^{-}(\\tau_{\\mathrm{top}})`; defaults to zero.
    surface_albedo : float, optional
        Surface albedo :math:`A \\in [0,1]`. If nonzero, reflection modifies the BOA upward intensity. (Current
        implementation uses a simplified placeholder.)

    Returns
    -------
    i_up : Quantity, shape (n_layers + 1, n_wn)
        Hemispherically integrated *upward* flux:

        .. math::
            F^{+}(\\tilde{\\nu}) = 2\\pi \\sum_{i} I^{+}_{i}(\\tilde{\\nu}) w_{i}.

    i_down : Quantity, shape (n_layers + 1, n_wn)
        Hemispherically integrated *downward* flux:

        .. math::
            F^{-}(\\tilde{\\nu}) = 2\\pi \\sum_{i} I^{-}_{i}(\\tilde{\\nu}) w_{i}.
    """
    if surface_albedo < 0 or surface_albedo > 1:
        log.warning(f"Surface albedo {surface_albedo} is outside of [0, 1], clipping.")
        surface_albedo = np.clip(surface_albedo, 0, 1)

    n_layers, n_wavelengths = dtau.shape

    # Compute intensity at interfaces.
    i_up = np.zeros((len(mu_values), n_layers + 1, n_wavelengths)) * source_function.unit
    i_down = np.zeros((len(mu_values), n_layers + 1, n_wavelengths)) * source_function.unit

    # Upper boundary condition at the top (level n_layers) is zero, unless incident radiation field!
    if incident_radiation_field is not None:
        i_down[:, n_layers, :] = incident_radiation_field
    else:
        i_down[:, n_layers, :] = 0.0 * source_function.unit

    # Integrate from TOA (k=n_layers-1) down to BOA (k=0)
    for k in range(n_layers - 1, -1, -1):
        delta_tau_mu = dtau[k, :] / np.abs(mu_values[:, None])
        exp_term = np.exp(-delta_tau_mu)
        source_contribution = source_function[k, :] * (1 - exp_term)

        # Intensity at the top interface of each layer
        i_down[:, k, :] = i_down[:, k + 1, :] * exp_term + source_contribution

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
        delta_tau_mu = dtau[k, :] / mu_values[:, None]
        exp_term = np.exp(-delta_tau_mu)
        source_contribution = source_function[None, k, :] * (1 - exp_term)

        # Intensity at the top of the layer (level k+1)
        i_up[:, k + 1, :] = i_up[:, k, :] * exp_term + source_contribution

    i_up = 2 * np.pi * u.sr * np.sum(i_up * mu_weights[:, None, None], axis=0)
    i_down = 2 * np.pi * u.sr * np.sum(i_down * mu_weights[:, None, None], axis=0)

    return i_up, i_down


class NLTEWorkflow(abc.ABC):
    # In theory NLTEProcessor can implement a generic NLTEWorkflow field that calls the workflow method. The current 
    # implementation hard-cords the Gauss-Seidel workflow, but it may be useful to be able to switch between that and
    # MALI without having to restructure the whole main workflow.
    @abc.abstractmethod
    def workflow(self) -> t.Any:
        pass


class GaussSeidelWorkflow(NLTEWorkflow):

    def workflow(self):
        # Implement current workflow from compute_opacities_profile().
        pass


class MALIWorkflow(NLTEWorkflow):

    def workflow(self):
        # Implement MALI layer step through, intensity and Lambda calculations.
        pass
