"""Spherical-shell radiative transfer (the p-z method): rays through concentric shells, their Gauss-Seidel
layer updates, the formal solution of 4 pi J and the emergent flux normalised to the planet radius."""
import typing as t

import numpy as np
import numba
from numpy import typing as npt
from astropy import units as u

from .source import mu_quadrature
from .bezier import bezier_terms, ray_dtau, ray_control_point_inward, ray_control_point_outward


class SphericalRays(t.NamedTuple):
    """
    Rays through concentric spherical shells (the impact-parameter or "p-z" method), for radiative transfer in a
    spherically symmetric atmosphere. See :func:`spherical_rays`.

    Arrays indexed ``[layer, ray]`` hold zeros where a ray does not reach that layer's centre.
    """
    impact_parameter: npt.NDArray[np.float64]  # (n_rays,) [m]
    tangent_layer: npt.NDArray[np.int64]  # (n_rays,) layer whose centre the ray grazes; -1 for rays hitting the BOA
    seg_a: npt.NDArray[np.float64]  # (n_layers, n_rays): optical depth between centres l-1 and l along the ray is
    seg_b: npt.NDArray[np.float64]  # seg_a[l] * dtau[l-1] + seg_b[l] * dtau[l]; row 0 unused
    mu: npt.NDArray[np.float64]  # (n_layers, n_rays) direction cosine of the ray at each layer centre
    weights: npt.NDArray[np.float64]  # (n_layers, n_rays) trapezoidal weights in mu at each layer, summing to 1


def spherical_rays(interface_radii: u.Quantity, n_core: int = 12, tangent_stride: int = 1) -> SphericalRays:
    """
    Build the rays for :func:`update_inward_layer_spherical`/:func:`update_outward_layer_spherical`.

    Each ray is a straight line at impact parameter ``p`` from the planet's centre. At a layer centre of radius
    ``r >= p`` it makes an angle with the local vertical of cosine ``mu = sqrt(1 - p^2 / r^2)``, so one ray serves every
    layer it crosses, at a different angle in each: this is what makes the geometry spherical rather than
    plane-parallel, where every layer sees the same directions. The rays are

    * the central ray ``p = 0`` and ``n_core`` "core" rays, whose directions at the bottom layer centre are the
      :func:`mu_quadrature` nodes. They reach the bottom of the atmosphere, where the planet below sets their
      outward intensity.
    * one ray tangent to every ``tangent_stride``-th layer centre counting down from the top, ``p = r_t``. It enters
      from the top, reaches layer ``t`` travelling horizontally (``mu = 0``) and leaves outwards along the same line.
      These carry the directions near the local horizon, including, high up, those that pass through the limb instead
      of hitting the planet.

    The optical depth between two neighbouring layer centres along a ray is the ray's path through the upper half of
    the lower layer and the lower half of the upper layer, each weighted by that layer's opacity per unit length
    (``dtau / dz``, constant within a layer as in the radial optical depth and :func:`tiramisu.eclipse.slant_tau`).
    Written as ``seg_a[l] * dtau[l - 1] + seg_b[l] * dtau[l]``; in the plane-parallel limit
    ``seg_a = seg_b = 1 / (2 mu)``.

    Mean intensities use trapezoidal weights in ``mu`` over the rays reaching each layer, from the central ray
    (``mu = 1``) down to the most horizontal one; where that is not the layer's own tangent ray (``tangent_stride >
    1``), its weight also covers the remaining interval down to ``mu = 0``.

    Parameters
    ----------
    interface_radii : Quantity, shape (n_layers + 1,)
        Radii of the layer interfaces, bottom first. Layer centres are taken halfway between.
    n_core : int
        Number of core rays besides the central one. The tangent rays already resolve the near-horizontal directions,
        so few are needed: for KELT-20b (80-100 layers), 12 give J to 0.2% (95th percentile) and 0.4% (max) of a
        converged solution, 8 to 0.4% / 0.9%. Atmospheres much thinner relative to the planet's radius spread the
        tangent rays over a narrower range of mu, and may need more.
    tangent_stride : int
        Keep a tangent ray at every ``tangent_stride``-th layer (always including the top layer). Every layer's own
        tangent ray matters: a stride of 2 already raises the error in J to ~0.5% (95th) / 2% (max).
    """
    radii_int = interface_radii.to_value(u.m)
    radii = 0.5 * (radii_int[:-1] + radii_int[1:])
    dz = np.diff(radii_int)
    n_layers = radii.size
    mu_core, _ = mu_quadrature(n_core)
    tangent_layers = np.arange(n_layers - 1, -1, -tangent_stride)[::-1]
    p = np.concatenate(([0.0], radii[0] * np.sqrt(1 - mu_core ** 2), radii[tangent_layers]))
    tangent_layer = np.concatenate((np.full(n_core + 1, -1), tangent_layers)).astype(np.int64)

    present = radii[:, None] >= p[None, :]
    # Path coordinate along the ray from its closest approach to the centre; (r - p)(r + p) avoids cancellation.
    z = np.sqrt(np.clip((radii[:, None] - p) * (radii[:, None] + p), 0, None)) * present
    z_int = np.sqrt(np.clip((radii_int[:, None] - p) * (radii_int[:, None] + p), 0, None))
    # Path from centre l-1 up to interface l, and from interface l up to centre l, each as (R^2 - r^2) / (z_R + z_r).
    seg_a = np.zeros((n_layers, p.size))
    seg_b = np.zeros((n_layers, p.size))
    reaches = present[:-1]
    r_lo, r_int, r_hi = radii[:-1, None], radii_int[1:-1, None], radii[1:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        seg_a[1:] = np.where(reaches, (r_int - r_lo) * (r_int + r_lo) / (z_int[1:-1] + z[:-1]) / dz[:-1, None], 0.0)
        seg_b[1:] = np.where(reaches, (r_hi - r_int) * (r_hi + r_int) / (z[1:] + z_int[1:-1]) / dz[1:, None], 0.0)

    mu = z / radii[:, None]
    weights = np.zeros_like(mu)
    for layer_idx in range(n_layers):
        idx = np.flatnonzero(present[layer_idx])
        order = idx[np.argsort(mu[layer_idx, idx])]
        d_mu = np.diff(mu[layer_idx, order])
        weights[layer_idx, order[:-1]] += 0.5 * d_mu
        weights[layer_idx, order[1:]] += 0.5 * d_mu
        weights[layer_idx, order[0]] += mu[layer_idx, order[0]]  # flat down to mu = 0 without an own tangent ray
    return SphericalRays(impact_parameter=p, tangent_layer=tangent_layer, seg_a=seg_a, seg_b=seg_b, mu=mu,
                         weights=weights)


@numba.njit(cache=True, error_model="numpy", inline="always")
def _ray_inward_step(
        i_in_matrix: npt.NDArray[np.float64], dtau: npt.NDArray[np.float64],
        seg_a: npt.NDArray[np.float64], seg_b: npt.NDArray[np.float64],
        source_function_matrix: npt.NDArray[np.float64], tangent: int, layer_idx: int, ray: int, k: int,
        with_lambda: bool,
) -> t.Tuple[float, float]:
    """
    Inward Bezier step along ``ray`` from centre ``layer_idx + 1`` to ``layer_idx``: the intensity arriving at
    ``layer_idx`` and its exact derivative with respect to ``S[layer_idx]`` (notes/lambda_star.tex, Sec. 5). Two
    steps depend on that S: this one (``alpha S_l + gamma C_l``) and the one into ``layer_idx + 1``, whose control
    point uses S_l through the source-function derivative at ``layer_idx + 1``, attenuated by this step (skipped where
    that attenuation is exactly zero). Without ``with_lambda`` the derivative is 0.
    """
    n_layers = source_function_matrix.shape[0]
    exp_next, alpha_next, beta_next, gamma_next = bezier_terms(ray_dtau(dtau, seg_a, seg_b, layer_idx + 1, ray, k))
    control_in, d_control_in = ray_control_point_inward(
        source_function_matrix, dtau, seg_a, seg_b, tangent, layer_idx, ray, k, layer_idx if with_lambda else -1)
    intensity = (
            i_in_matrix[layer_idx + 1, ray, k] * exp_next
            + alpha_next * source_function_matrix[layer_idx, k]
            + beta_next * source_function_matrix[layer_idx + 1, k]
            + gamma_next * control_in
    )
    if not with_lambda:
        return intensity, 0.0
    d_intensity = alpha_next + gamma_next * d_control_in
    if layer_idx + 2 < n_layers and exp_next > 0:
        gamma_prev = bezier_terms(ray_dtau(dtau, seg_a, seg_b, layer_idx + 2, ray, k))[3]
        d_intensity += exp_next * gamma_prev * ray_control_point_inward(
            source_function_matrix, dtau, seg_a, seg_b, tangent, layer_idx + 1, ray, k, layer_idx)[1]
    return intensity, d_intensity


@numba.njit(parallel=True, cache=True, error_model="numpy")
def update_inward_layer_spherical(
        layer_idx: int,
        i_in_matrix: npt.NDArray[np.float64],
        lambda_in_layer: npt.NDArray[np.float64],
        dtau: npt.NDArray[np.float64],
        seg_a: npt.NDArray[np.float64],
        seg_b: npt.NDArray[np.float64],
        tangent_layer: npt.NDArray[np.int64],
        source_function_matrix: npt.NDArray[np.float64],
        with_lambda: bool = True,
) -> None:
    """
    :func:`update_inward_layer` along spherical rays (:func:`spherical_rays`): the same Bezier short-characteristics
    step from centre ``layer_idx + 1`` to ``layer_idx``, with the ray's own optical depth and control points in place
    of the plane-parallel ``delta_tau / mu``. ``i_in_matrix`` has shape (n_layers, n_rays, n_grid) and
    ``lambda_in_layer`` (n_rays, n_grid); rays that do not reach ``layer_idx`` are left untouched (zero).
    """
    n_rays, n_grid = i_in_matrix.shape[1], i_in_matrix.shape[2]
    for k in numba.prange(n_grid):
        for j in range(n_rays):
            tangent = tangent_layer[j]
            if tangent > layer_idx:
                continue
            intensity, d_intensity = _ray_inward_step(
                i_in_matrix, dtau, seg_a, seg_b, source_function_matrix, tangent, layer_idx, j, k, with_lambda)
            i_in_matrix[layer_idx, j, k] = intensity
            if with_lambda:
                lambda_in_layer[j, k] = d_intensity


@numba.njit(parallel=True, cache=True, error_model="numpy")
def update_outward_layer_spherical(
        layer_idx: int,
        i_out_below: npt.NDArray[np.float64],
        i_out_layer: npt.NDArray[np.float64],
        lambda_out_layer: npt.NDArray[np.float64],
        sigma_below: npt.NDArray[np.float64],
        sigma_layer: npt.NDArray[np.float64],
        lambda_in_layer: npt.NDArray[np.float64],
        i_in_matrix: npt.NDArray[np.float64],
        dtau: npt.NDArray[np.float64],
        seg_a: npt.NDArray[np.float64],
        seg_b: npt.NDArray[np.float64],
        tangent_layer: npt.NDArray[np.int64],
        source_function_matrix: npt.NDArray[np.float64],
        with_lambda: bool = True,
) -> None:
    """
    :func:`update_outward_layer` along spherical rays: the Bezier step from centre ``layer_idx - 1`` (``i_out_below``)
    to ``layer_idx`` (``i_out_layer``, ``lambda_out_layer``; all (n_rays, n_grid)). A ray tangent to ``layer_idx``
    turns there: its outward intensity is the inward step arriving at the tangent point, recomputed from the current
    source function and ``i_in_matrix``. Rays hitting the BOA (``tangent_layer < 0``) are skipped at ``layer_idx == 0``,
    where the caller sets their boundary intensity.

    ``lambda_out_layer`` is the exact dI_out / dS[layer_idx] per ray (notes/lambda_star.tex, Sec. 5). A tangent ray
    crosses the layer on its way in as well, so S[layer_idx] also reaches the outward crossing through the inward steps
    near the inward crossing, attenuated by the optical path down to the turning point and back. This needs
    ``lambda_in_layer``, this layer's inward Lambda* from :func:`update_inward_layer_spherical` with the same source
    function (zero at the top layer), and ``sigma_below``, each ray's optical path from its turning point up to
    ``layer_idx - 1``; the path up to ``layer_idx`` is written to ``sigma_layer``, which the caller swaps with
    ``sigma_below`` along with ``i_out_below``/``i_out_layer``, and must be kept up to date in every call; without
    ``with_lambda`` (Lambda* not used), ``lambda_in_layer`` may hold anything and ``lambda_out_layer`` is left alone.
    """
    n_layers = source_function_matrix.shape[0]
    n_rays, n_grid = i_out_layer.shape
    for k in numba.prange(n_grid):
        for j in range(n_rays):
            tangent = tangent_layer[j]
            if tangent > layer_idx:
                continue
            if tangent == layer_idx:
                sigma_layer[j, k] = 0.0
                if layer_idx == n_layers - 1:
                    # Grazes the top point only: the incident intensity, with no local contribution.
                    i_out_layer[j, k] = i_in_matrix[layer_idx, j, k]
                    lambda_out_layer[j, k] = 0.0
                else:
                    intensity, d_intensity = _ray_inward_step(
                        i_in_matrix, dtau, seg_a, seg_b, source_function_matrix, tangent, layer_idx, j, k, with_lambda)
                    i_out_layer[j, k] = intensity
                    if with_lambda:
                        lambda_out_layer[j, k] = d_intensity
                continue
            if layer_idx == 0:
                continue
            dtau_here = ray_dtau(dtau, seg_a, seg_b, layer_idx, j, k)
            exp_here, alpha_here, beta_here, gamma_here = bezier_terms(dtau_here)
            control_out, d_control_out = ray_control_point_outward(
                source_function_matrix, dtau, seg_a, seg_b, tangent, layer_idx, j, k, layer_idx if with_lambda else -1)
            i_out_layer[j, k] = (
                    i_out_below[j, k] * exp_here
                    + alpha_here * source_function_matrix[layer_idx, k]
                    + beta_here * source_function_matrix[layer_idx - 1, k]
                    + gamma_here * control_out
            )
            if tangent >= 0:
                sigma_prev = sigma_below[j, k]
                sigma_here = sigma_prev + dtau_here
                sigma_layer[j, k] = sigma_here
            if not with_lambda:
                continue
            # Terms attenuated to exactly zero are skipped (their derivatives are finite, so this changes nothing).
            d_intensity = alpha_here + gamma_here * d_control_out
            # The outward step into layer_idx - 1, if the ray's outward leg (from the BOA or the turning point) has one.
            has_prev = layer_idx - 2 >= max(tangent, 0)
            if has_prev:
                dtau_prev = ray_dtau(dtau, seg_a, seg_b, layer_idx - 1, j, k)
                gamma_prev = bezier_terms(dtau_prev)[3]
                if exp_here > 0:
                    d_intensity += exp_here * gamma_prev * ray_control_point_outward(
                        source_function_matrix, dtau, seg_a, seg_b, tangent, layer_idx - 1, j, k, layer_idx)[1]
            if tangent >= 0:
                # S[layer_idx] also acts on the way in, through the inward steps arriving at layer_idx + 1 ... layer_idx -
                # 2, each attenuated down to the turning point and back up, exp(-(sigma(m) + sigma(layer_idx))). Those
                # arriving at layer_idx + 1 and layer_idx make up the inward derivative at layer_idx, lambda_in_layer.
                transmission = np.exp(-(sigma_prev + sigma_here))
                if transmission > 0:
                    d_q_prev = beta_here + gamma_here * ray_control_point_inward(
                        source_function_matrix, dtau, seg_a, seg_b, tangent, layer_idx - 1, j, k, layer_idx)[1]
                    d_intensity += transmission * (exp_here * lambda_in_layer[j, k] + d_q_prev)
                if has_prev:  # the inward step from layer_idx - 1 to layer_idx - 2
                    sigma_prev_2 = sigma_prev - dtau_prev if layer_idx - 2 > tangent else 0.0
                    transmission = np.exp(-(sigma_prev_2 + sigma_here))
                    if transmission > 0:
                        d_intensity += transmission * gamma_prev * ray_control_point_inward(
                            source_function_matrix, dtau, seg_a, seg_b, tangent, layer_idx - 2, j, k, layer_idx)[1]
            lambda_out_layer[j, k] = d_intensity


def four_pi_j_spherical(
        dtau: npt.NDArray[np.float64],
        source_function: u.Quantity,
        rays: SphericalRays,
        incident_radiation_field: t.Optional[u.Quantity] = None,
        bottom_intensity: t.Optional[u.Quantity] = None,
) -> t.Tuple[u.Quantity, u.Quantity]:
    """
    Outward and inward hemispheres' contributions to 4*pi*J at each layer centre, for a fixed source function, along
    spherical rays (one inward then one outward Bezier sweep, as in the NLTE Gauss-Seidel iteration without population
    updates). The spherical counterpart of :func:`four_pi_j_hemispheres`, whose values are at interfaces instead.

    Parameters
    ----------
    dtau : ndarray, shape (n_layers, n_wn)
        Optical thickness of each layer, bottom first.
    source_function : Quantity, shape (n_layers, n_wn)
    rays : SphericalRays
    incident_radiation_field : Quantity, shape (n_wn,) or (n_rays, n_wn), optional
        Downward specific intensity entering at the top along each ray; zero if omitted.
    bottom_intensity : Quantity, shape (n_wn,), optional
        Outward intensity at the bottom layer centre for rays hitting the BOA; defaults to the bottom source function.
    """
    unit = source_function.unit
    source = np.ascontiguousarray(source_function.to_value(unit))
    n_layers, n_grid = source.shape
    layer_shape = (rays.impact_parameter.size, n_grid)
    i_in = np.zeros((n_layers, *layer_shape))
    lam_in, lam_out, i_out_below, i_out_layer, sigma_below, sigma_layer = (np.zeros(layer_shape) for _ in range(6))
    if incident_radiation_field is not None:
        i_in[-1] = incident_radiation_field.to_value(unit)
    for layer_idx in range(n_layers - 2, -1, -1):
        update_inward_layer_spherical(
            layer_idx=layer_idx, i_in_matrix=i_in, lambda_in_layer=lam_in, dtau=dtau, seg_a=rays.seg_a,
            seg_b=rays.seg_b, tangent_layer=rays.tangent_layer, source_function_matrix=source, with_lambda=False,
        )
    four_pi_j_out = np.zeros((n_layers, n_grid))
    i_out_layer[rays.tangent_layer < 0] = source[0] if bottom_intensity is None else bottom_intensity.to_value(unit)
    for layer_idx in range(n_layers):
        if layer_idx > 0:
            i_out_below, i_out_layer = i_out_layer, i_out_below
            sigma_below, sigma_layer = sigma_layer, sigma_below
        update_outward_layer_spherical(
            layer_idx=layer_idx, i_out_below=i_out_below, i_out_layer=i_out_layer, lambda_out_layer=lam_out,
            sigma_below=sigma_below, sigma_layer=sigma_layer, lambda_in_layer=lam_in, i_in_matrix=i_in, dtau=dtau,
            seg_a=rays.seg_a, seg_b=rays.seg_b, tangent_layer=rays.tangent_layer, source_function_matrix=source,
            with_lambda=False,
        )
        four_pi_j_out[layer_idx] = rays.weights[layer_idx] @ i_out_layer
    four_pi = 2 * np.pi * u.sr * unit
    return four_pi_j_out * four_pi, np.einsum("lj,ljk->lk", rays.weights, i_in) * four_pi


def emergent_flux_spherical(
        dtau: npt.NDArray[np.float64],
        source_function: u.Quantity,
        interface_radii: u.Quantity,
        planet_radius: u.Quantity,
        n_core: int = 50,
) -> u.Quantity:
    """
    Emergent flux of a spherically symmetric atmosphere, normalised to ``planet_radius`` (R_p) so that it replaces
    :func:`emergent_flux`: the flux received at distance d is (R_p / d)^2 times the returned value.

    The disk seen by the observer is integrated over impact parameter p, F = (1 / R_p^2) int_0^R_top I(p) 2 pi p dp,
    with ``I(p)`` the intensity leaving along the line of sight at ``p``. With R_0 the bottom radius
    (``interface_radii[0]``), rays with ``p < R_0`` (Gauss-Legendre in ``p^2``) start at the planet (``I = S_0``, as in
    :func:`emergent_flux`); rays with ``p > R_0``, one per shell at its ``p^2`` midpoint (the impact parameters of
    :func:`tiramisu.eclipse.compute_transmission`), cross the far side of the limb from empty space and then the near
    side, so the extended atmosphere's limb emission is included. Along each
    ray, every shell is crossed with its opacity per unit length (``dtau / dz``) and constant source function, the
    constant-S scheme of :func:`emergent_flux`. With ``R_p = R_0`` and ``R_0 -> infinity`` this reduces to
    :func:`emergent_flux`.

    Returns
    -------
    Quantity, shape (n_wn,)
        Emergent flux [W / (m^2 Hz)], normalised to ``planet_radius``.
    """
    radii = interface_radii.to_value(u.m)
    dz = np.diff(radii)
    x, wx = np.polynomial.legendre.leggauss(n_core)
    p_sq = np.concatenate((0.5 * (x + 1) * radii[0] ** 2, 0.5 * (radii[:-1] ** 2 + radii[1:] ** 2)))
    d_p_sq = np.concatenate((0.5 * wx * radii[0] ** 2, np.diff(radii ** 2)))
    core = np.arange(p_sq.size) < n_core
    # Half-chord of each ray through each shell, (n_layers, n_rays).
    half_chord = (np.sqrt(np.clip(radii[1:, None] ** 2 - p_sq, 0, None))
                  - np.sqrt(np.clip(np.maximum(radii[:-1, None] ** 2, p_sq) - p_sq, 0, None)))
    path_per_dz = half_chord / dz[:, None]

    unit = source_function.unit
    intensity = np.zeros((p_sq.size, dtau.shape[1])) << unit
    for layer_idx in range(dtau.shape[0] - 1, -1, -1):  # far side of the limb, inwards
        exp_term = np.exp(-dtau[layer_idx] * path_per_dz[layer_idx][:, None])
        intensity = intensity * exp_term + source_function[layer_idx] * (1 - exp_term)
    intensity[core] = source_function[0]
    for layer_idx in range(dtau.shape[0]):  # near side, outwards
        exp_term = np.exp(-dtau[layer_idx] * path_per_dz[layer_idx][:, None])
        intensity = intensity * exp_term + source_function[layer_idx] * (1 - exp_term)
    return (np.pi * u.sr * np.sum(d_p_sq[:, None] * intensity, axis=0) / planet_radius.to_value(u.m) ** 2).to(
        u.W / (u.m ** 2 * u.Hz))
