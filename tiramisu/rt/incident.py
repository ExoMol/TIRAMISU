"""The star's irradiation of the top of the atmosphere: the stellar surface intensity (PHOENIX via
phoenix4all) and its distribution over the solvers' directions."""
import dataclasses
import typing as t

import numpy as np
from numpy import typing as npt
from phoenix4all import get_spectrum
from astropy import units as u

from ..profiles.lineshape import rebin_spectrum


def incident_srf(
        star_temperature: float, star_logg: float, star_feh: float, wn_grid: u.Quantity, star_alpha: float = 0.0
) -> u.Quantity:
    """
    Returns the disk-averaged specific intensity of the stellar surface (no limb darkening), I = F_s / pi, binned onto
    wn_grid.

    Specific intensity is conserved along a ray in vacuum, so this is also the intensity arriving at the planet from
    within the stellar disk: it is a property of the star alone, independent of the orbital distance. The distance
    only sets how much of the planet's sky the star covers, which is applied by :func:`incident_intensity`.

    Parameters
    ----------
    star_temperature
    star_logg
    star_feh
    wn_grid
    star_alpha

    Returns
    -------
        Stellar specific intensity [W / (m^2 Hz sr)].

    """
    srf_wavelength, srf_flux = get_spectrum(
        teff=star_temperature,
        logg=star_logg,
        feh=star_feh,
        alpha=star_alpha,
        # source="synphot",
        source="svo",
        model_name="bt-settl-agss",
        bounds_error=True,
    )
    # srf_flux is the stellar surface flux F_s, in erg / (Angstrom s cm^2)
    srf_wn = srf_wavelength.to(1 / u.cm, equivalencies=u.spectral())

    srf_flux_nu = srf_flux.to(
        u.J / (u.s * u.m ** 2 * u.Hz),
        equivalencies=u.spectral_density(srf_wavelength)
    )

    sort_idx = np.argsort(srf_wn)
    srf_flux_binned = rebin_spectrum(
        x_in=srf_wn[sort_idx].value,
        y_in=srf_flux_nu[sort_idx].value,
        new_centers=wn_grid.to_value(1 / u.cm),
    ) << srf_flux_nu.unit
    return (srf_flux_binned / (np.pi * u.sr)).to(u.W / (u.Hz * u.m ** 2 * u.sr))


def _stellar_ring_coverage(
        mu: npt.NDArray[np.float64], mu0: npt.NDArray[np.float64], star_angular_radius: float
) -> npt.NDArray[np.float64]:
    """
    Fraction of the ring of directions with zenith cosine ``mu`` (all azimuths) that lies inside the stellar disk, for
    a star centred at zenith cosine ``mu0`` with angular radius ``star_angular_radius`` [rad]. Broadcasts ``mu``
    against ``mu0``.

    A direction at zenith angle psi and azimuth phi (measured from the star's azimuth) is inside the disk when its
    angle to the disk centre is at most the angular radius theta: cos(psi) cos(psi0) + sin(psi) sin(psi0) cos(phi) >=
    cos(theta). The covered azimuths are |phi| <= arccos(c), with c = (cos(theta) - cos(psi) cos(psi0)) /
    (sin(psi) sin(psi0)), so the covered fraction of the ring is arccos(clip(c, -1, 1)) / pi. When either direction
    is vertical the test reduces to |psi - psi0| <= theta.
    """
    sin_prod = np.sqrt(np.clip(1 - mu ** 2, 0, None) * np.clip(1 - mu0 ** 2, 0, None))
    numerator = np.cos(star_angular_radius) - mu * mu0
    vertical = sin_prod == 0
    c = np.where(vertical, np.where(numerator <= 0, -1.0, 1.0), numerator / np.where(vertical, 1.0, sin_prod))
    return np.arccos(np.clip(c, -1, 1)) / np.pi


@dataclasses.dataclass(frozen=True, eq=False)
class IncidentIntensity:
    """
    Stellar irradiation of the top of the atmosphere, built by :func:`incident_intensity` (see there for the physics).
    Calling it on a solver's directions, ``field(mu_values=..., mu_weights=...)``, returns the downward specific
    intensity on each, shape (n_mu, n_wn).
    """
    star_intensity: u.Quantity
    star_radius: u.Quantity
    orbital_radius: u.Quantity
    location: t.Union[str, float] = "dayside"

    def __call__(self, mu_values: npt.NDArray[np.float64], mu_weights: npt.NDArray[np.float64]) -> u.Quantity:
        theta = float(np.arcsin((self.star_radius / self.orbital_radius).decompose().value))
        location = self.location
        if isinstance(location, str):
            mu0_range = {"dayside": (0.0, 1.0), "planet_average": (-1.0, 1.0)}[location]
            # Resolve the band of mu0 over which a ring is covered (width ~ theta) with ~20+ samples.
            n_mu0 = int(max(2000, (mu0_range[1] - mu0_range[0]) * 20 / theta))
            mu0 = mu0_range[0] + (np.arange(n_mu0) + 0.5) * (mu0_range[1] - mu0_range[0]) / n_mu0
        else:
            mu0 = np.array([float(location)])

        # Ring coverage on the solver's nodes, averaged over the locations the column represents.
        coverage = _stellar_ring_coverage(
            mu=mu_values[:, None], mu0=mu0[None, :], star_angular_radius=theta
        ).mean(axis=1)

        # Exact target: integral of the coverage over mu in [0, 1], on the part of [0, 1] the disk can reach.
        psi0 = np.arccos(np.clip(mu0, -1, 1))
        mu_lo = np.clip(np.cos(np.minimum(psi0 + theta, np.pi)), 0, 1)
        mu_hi = np.clip(np.cos(np.maximum(psi0 - theta, 0)), 0, 1)
        x, wx = np.polynomial.legendre.leggauss(256)
        mu_fine = mu_lo[None, :] + (x[:, None] + 1) * 0.5 * (mu_hi - mu_lo)[None, :]
        visible = 0.5 * (mu_hi - mu_lo) * np.sum(
            wx[:, None] * _stellar_ring_coverage(mu=mu_fine, mu0=mu0[None, :], star_angular_radius=theta), axis=0
        )
        target = visible.mean()

        quadrature = np.sum(mu_weights * coverage)
        if quadrature > 0:
            coverage *= target / quadrature
        elif target > 0:
            nearest = np.argmin(np.abs(mu_values - np.clip(mu0.mean(), 0, 1)))
            coverage = np.zeros_like(mu_values)
            coverage[nearest] = target / mu_weights[nearest]
        return coverage[:, None] * self.star_intensity[None, :]


def incident_intensity(
        star_intensity: u.Quantity,
        star_radius: u.Quantity,
        orbital_radius: u.Quantity,
        location: t.Union[str, float] = "dayside",
) -> IncidentIntensity:
    """
    The star's irradiation of the top of the atmosphere, as the ``incident_radiation_field`` of
    :class:`tiramisu.xsec.XSecCollection` and :meth:`tiramisu.eclipse.ExoplanetEmission.compute_emission`. The solvers
    evaluate it on their own directions (plane-parallel ``mu`` nodes or spherical rays), so the same object serves
    either geometry; calling it, ``field(mu_values=..., mu_weights=...)``, gives the downward specific intensity on each
    direction of that grid.

    **What it represents.** The solvers are 1D and azimuthally symmetric: each ``mu`` node stands for the whole ring
    of directions at that zenith cosine, seen from one point at the top of the column. The star is a disk of uniform
    specific intensity ``star_intensity`` (:func:`incident_srf`) and angular radius theta = arcsin(R_star / a), so
    the boundary value on a node is ``star_intensity`` times the fraction of that ring the disk covers
    (:func:`_stellar_ring_coverage`); the part of the disk below the horizon is hidden by the planet.

    **Why the distance enters here.** Specific intensity does not fall off with distance; the stellar radiation
    reaching the planet does, because the disk covers less of the sky. That dilution is exactly the covered fraction
    of the rings, so there is no separate (R_star / a)^2 factor: the mean intensity it produces at the top,
    J = 1/2 sum_i w_i I_i, equals star_intensity * Omega_visible / (4 pi), which falls as (R_star / a)^2 through
    theta.

    **Locations.** The column stands for one of:

    * ``"dayside"``: the area-average over the dayside. Area on a sphere is uniform in the stellar zenith cosine mu0,
      so this is the ring coverage averaged over mu0 in (0, 1). For a small disk it is isotropic, with
      I = (1 - cos theta) * star_intensity on every direction (J = Omega star_intensity / (4 pi), downward flux F_a / 2
      with F_a = pi sin^2(theta) star_intensity the flux at the planet), the usual dayside redistribution f = 1/2.
    * ``"planet_average"``: the same over the whole sphere, mu0 in (-1, 1); half the dayside value (f = 1/4). Every
      disk direction is above the horizon on exactly one of any two antipodal points, so J is exactly
      Omega star_intensity / (8 pi).
    * a float mu0 in [-1, 1]: a single location with the disk centre at stellar zenith cosine mu0 (1 substellar, 0
      terminator, < -sin(theta) no star). The attenuation below the top is plane-parallel, e^(-tau / mu), which
      overestimates the path of grazing starlight: in a spherical atmosphere it is capped near
      sqrt(pi R_p / (2 H)), so results for small mu0 (roughly below sqrt(2 H / R_p)) should not be trusted.

    **Quadrature.** A narrow disk can fall between ``mu`` nodes, so the node values are rescaled so that
    sum_i w_i I_i reproduces the exact J (the ring coverage integrated finely over mu). If no node lies inside the
    disk (a distant star seen at one location), the whole J is put on the node closest to the disk centre.

    Parameters
    ----------
    star_intensity : Quantity, shape (n_wn,)
        Stellar surface specific intensity [W / (m^2 Hz sr)], from :func:`incident_srf`.
    star_radius, orbital_radius : Quantity
        Stellar radius and star-planet distance.
    location : {"dayside", "planet_average"} or float
        Region of the planet the column represents (see above).

    Returns
    -------
    IncidentIntensity
        Callable with ``(mu_values, mu_weights)`` (shape (n_mu,) each) returning a Quantity of shape (n_mu, n_wn), the
        downward specific intensity on each direction at the top, in the units of ``star_intensity``.
    """
    return IncidentIntensity(
        star_intensity=star_intensity, star_radius=star_radius, orbital_radius=orbital_radius, location=location,
    )
