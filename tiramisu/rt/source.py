"""Quantities shared by both geometries: the effective source function and layer optical depths from the
opacity (with maser shielding), the Planck function and the angular quadrature."""
import typing as t

import numpy as np
import numba
from numpy import typing as npt
from astropy import units as u, constants as ac

from ..constants import ac_2_h_on_c_sq, ac_h_on_kB


# Maser shielding: a negative (masing) opacity chi is replaced by MASER_SHIELD_FACTOR * |chi| (see
# _effective_chi_source_dtau_core). NLTE species are shielded individually (shield_masing_chi) before their opacity
# enters the global chi, so masing in one species cannot cancel the others' opacity toward zero, where S = eta / chi
# blows up.
MASER_SHIELD_FACTOR = 0.1


@numba.njit(cache=True, inline="always")
def _shield_chi_value(chi: float) -> float:
    """The one place the shielding is defined: ``MASER_SHIELD_FACTOR * |chi|`` for a negative (masing) chi."""
    return -MASER_SHIELD_FACTOR * chi if chi < 0.0 else chi


@numba.njit(cache=True)
def shield_masing_chi(chi: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Copy of a (C-contiguous) ``chi`` with its negative (masing) entries shielded (:func:`_shield_chi_value`)."""
    out = np.empty_like(chi)
    src, dst = chi.reshape(-1), out.reshape(-1)
    for i in range(src.shape[0]):
        dst[i] = _shield_chi_value(src[i])
    return out


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _effective_chi_source_dtau_core(
        global_chi: npt.NDArray[np.float64],
        global_source_func: npt.NDArray[np.float64],
        global_eta: npt.NDArray[np.float64],
        density: npt.NDArray[np.float64],
        dz: npt.NDArray[np.float64],
        c_cgs: float,
        eta_to_source_func_scale: float,
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Fused, allocation-light core for :func:`effective_source_tau_mu`.

    Numpy's version of this (row max, boolean masks for negative entries, ``exp``, masked division) allocates
    several full ``(n_layers, n_grid)`` temporaries (the ``.copy()``, the ``nonzero`` index arrays, two boolean
    masks) on every call; this does the same row-local derivation in one pass per layer with no intermediates,
    which matters here since it's called once per NLTE layer per iteration on the same full-size arrays.

    All arguments are plain arrays/floats in a consistent cgs-like unit system (chi in cm^2, eta in
    erg*cm/(s*sr), density in 1/cm^3, dz in cm) - unit handling/conversion happens once in the
    :func:`effective_source_tau_mu` wrapper, not per element here.

    Returns
    -------
    effective_chi, effective_source_func, dtau : ndarray, shape (n_layers, n_grid)
        Shielded (always non-negative) effective opacity, the correspondingly corrected source function, and the
        per-layer optical depth increment computed from ``effective_chi``.
    """
    n_layers, n_grid = global_chi.shape
    effective_chi = np.empty((n_layers, n_grid), dtype=np.float64)
    effective_source_func = np.empty((n_layers, n_grid), dtype=np.float64)
    dtau = np.empty((n_layers, n_grid), dtype=np.float64)

    # Maser shielding: a negative (masing) chi is replaced by MASER_SHIELD_FACTOR * abs(chi) - a fixed fraction of that
    # element's own magnitude, not the row max/min. This is the standard treatment used in published coupled
    # SE+ALI maser codes (e.g. Nesterenok 2016, MNRAS 455, 3978, CH3OH masers). Unlike the row-relative
    # min/max-based remaps tried earlier (which made stronger masing map to a *smaller* effective_chi, so it
    # could collapse toward 0 and blow up anything dividing by it - e.g. psi_approx_eta/V-matrix terms in
    # build_y_matrix), this keeps effective_chi directly proportional to abs(chi), so it can never collapse
    # regardless of how extreme or how deep the accumulated masing is. It does not try to preserve the relative
    # shape of the masing profile - saturation (the real physical mechanism bounding maser gain) is left to the
    # statistical equilibrium equations, not to this shielding.

    for i in numba.prange(n_layers):
        for j in range(n_grid):
            chi_val = _shield_chi_value(global_chi[i, j])
            effective_chi[i, j] = chi_val

            sf_val = global_source_func[i, j]
            if sf_val < 0.0:
                if chi_val == 0.0:
                    sf_val = 0.0
                else:
                    sf_val = global_eta[i, j] / (c_cgs * chi_val) * eta_to_source_func_scale
            effective_source_func[i, j] = sf_val

            dtau[i, j] = chi_val * density[i] * dz[i]

    return effective_chi, effective_source_func, dtau


@numba.njit(parallel=True, cache=True, error_model="numpy")
def _layer_centre_tau(dtau: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Optical depth from the TOA to each layer's centre, where its source function and populations live: the sum of
    dtau from the top down to the layer, minus half its own (summed in the order of a cumulative sum from the top)."""
    n_layers, n_grid = dtau.shape
    tau = np.empty_like(dtau)
    for k in numba.prange(n_grid):
        above = 0.0
        for i in range(n_layers - 1, -1, -1):
            above += dtau[i, k]
            tau[i, k] = above - 0.5 * dtau[i, k]
    return tau


def effective_source_tau_mu(
        global_source_func_matrix: u.Quantity,
        global_chi_matrix: u.Quantity,
        global_eta_matrix: u.Quantity,
        density_profile: u.Quantity,
        dz_profile: u.Quantity,
) -> t.Tuple[u.Quantity, npt.NDArray[np.float64], u.Quantity, npt.NDArray[np.float64]]:
    """
    Compute an effective Source function and optical depth, tau, for use in calculation of Bezier interpolants. These
    are computed based on an effective opacity calculated using Eq. (39) from https://doi.org/10.48550/arXiv.2508.12873.

    Negative (masing/population-inversion) entries in ``global_chi_matrix`` are shielded by mapping ``chi`` to
    ``MASER_SHIELD_FACTOR * abs(chi)``, the standard treatment used in published coupled statistical-equilibrium
    + ALI maser codes (e.g. Nesterenok 2016, MNRAS 455, 3978). This keeps the shielded value directly proportional
    to the element's own magnitude - never collapsing toward zero regardless of how extreme the masing - unlike
    row-relative min/max-based remaps tried earlier, which made stronger masing map to a *smaller* effective_chi
    and could blow up anything dividing by it (e.g. :func:`NLTEProcessor.build_y_matrix`'s Lambda*-operator
    terms). It does not attempt to preserve the shape of the negative-opacity distribution; saturation (the real
    physical mechanism bounding maser gain) is left to the statistical equilibrium equations, not this shielding.
    NLTE species' opacities are already shielded individually (:func:`shield_masing_chi`) before they are summed, so
    with them ``global_chi_matrix`` has no negative entries and this only guards other inputs.

    Parameters
    ----------
    global_source_func_matrix
    global_chi_matrix
    global_eta_matrix
    density_profile
    dz_profile

    Returns
    -------
    effective_source_func_matrix : Quantity
    tau : ndarray, shape (n_layers, n_grid)
        Optical depth from the TOA to each layer centre, with no angle axis. This used to be broadcast over mu here
        (``tau_mu``, ``(n_layers, n_angles, n_grid)``) before being returned, materialising an array ~50x larger than
        ``tau`` itself for no reason: :func:`bezier_delta_tau_and_control_points`/:func:`update_layer_coefficients` (the
        only consumers) only ever use ``tau_mu[i, j, k]`` as ``tau[i, k] / mu_values[j]`` at a point they're already
        visiting in a loop, so the division now happens there instead of via a standalone broadcast. Callers must pass
        ``mu_values`` to those functions directly.
    effective_chi : Quantity, shape (n_layers, n_grid)
        The shielded (always non-negative) effective opacity used to compute ``tau``. Callers should reuse this -
        rather than re-deriving something from the raw ``global_chi_matrix`` - anywhere else in the iteration that
        needs a numerically safe stand-in for opacity near masing (population-inversion, negative-opacity) regions,
        e.g. :func:`NLTEProcessor.build_y_matrix`'s Lambda*-operator terms.
    dtau : ndarray, shape (n_layers, n_grid)
        Optical thickness of each layer from ``effective_chi``, the input of the spherical-ray kernels.
    """
    chi_unit = u.cm ** 2
    eta_unit = u.erg * u.cm / (u.s * u.sr)

    c_cgs = ac.c.cgs.value
    # Unit.to() (unlike Quantity.to()) already returns the bare conversion factor, not a Quantity.
    eta_to_source_func_scale = (
            eta_unit / (u.cm / u.s * chi_unit)
    ).to(global_source_func_matrix.unit, equivalencies=u.spectral())

    effective_chi_val, effective_source_func_val, dtau = _effective_chi_source_dtau_core(
        global_chi=global_chi_matrix.to_value(chi_unit),
        global_source_func=global_source_func_matrix.value,
        global_eta=global_eta_matrix.to_value(eta_unit),
        density=density_profile.to_value(1 / u.cm ** 3),
        dz=dz_profile.to_value(u.cm),
        c_cgs=c_cgs,
        eta_to_source_func_scale=eta_to_source_func_scale,
    )

    effective_chi = effective_chi_val << chi_unit
    effective_source_func_matrix = effective_source_func_val << global_source_func_matrix.unit

    tau = _layer_centre_tau(dtau)
    return effective_source_func_matrix, tau, effective_chi, dtau


def blackbody(spectral_grid: u.Quantity, temperature: u.Quantity) -> u.Quantity:
    freq_grid = spectral_grid.to(u.Hz, equivalencies=u.spectral())
    temperature = np.atleast_1d(temperature)[:, None]
    return (ac_2_h_on_c_sq * freq_grid ** 3) / (np.exp(ac_h_on_kB * freq_grid / temperature) - 1) / u.sr


def mu_quadrature(n_mu: int = 50) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Direction cosines and weights for every angular integral over one hemisphere: Gauss-Legendre nodes on (0, 1) with
    weights summing to 1, so that J = 1/2 sum_i w_i (I_i^+ + I_i^-). Shared by the radiative-transfer solvers and
    :func:`incident_intensity`, whose per-direction boundary must be built on the same nodes.
    """
    mu_values, mu_weights = np.polynomial.legendre.leggauss(n_mu)
    return (mu_values + 1) * 0.5, mu_weights / 2
