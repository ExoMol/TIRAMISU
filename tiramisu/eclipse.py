import logging
import typing as t
import numpy.typing as npt
from astropy import units as u
import numpy as np

from .chemistry import ChemicalProfile, SpeciesFormula
from .atmos import build_pressure_profiles, density_profile, hydrostatic_radii
from .xsec import XSecCollection
from .rt import (effective_source_tau_mu, four_pi_j_hemispheres, four_pi_j_spherical, emergent_flux,
                 emergent_flux_spherical, mu_quadrature, spherical_rays)
from .config import output_dir

log = logging.getLogger(__name__)


class ExoplanetEmission:
    __slots__ = [
        "planet_mass", "planet_radius", "chemistry_profile", "temperature_profile", "central_pressure",
        "pressure_levels", "dz", "density", "geometry", "interface_radii",
    ]

    def __init__(
            self,
            planet_mass: u.Quantity,
            planet_radius: u.Quantity,
            temperature_profile: u.Quantity,
            boa_pressure: u.Quantity,
            toa_pressure: u.Quantity,
            nlayers: int,
            chemistry_profile: ChemicalProfile,
            central_pressure: u.Quantity = None,
            pressure_levels: u.Quantity = None,
            geometry: str = "plane_parallel",
            reference_pressure: u.Quantity = None,
    ):
        """
        ``planet_radius`` is the radius at ``reference_pressure``, which must lie within the pressure grid; by default
        the bottom interface (``pressure_levels[0]``, or ``boa_pressure`` when the levels are built here from
        ``boa_pressure``, ``toa_pressure`` and ``nlayers``). The layers are in hydrostatic equilibrium with
        g(r) = G M / r^2 (:func:`tiramisu.atmos.hydrostatic_radii`). For a measured transit radius, use the pressure
        it probes (the optical transit photosphere is ~1-20 mbar). The emission spectrum is normalised to
        ``planet_radius``: the flux received at distance d is (planet_radius / d)^2 times it.

        ``geometry`` sets the radiative transfer for the NLTE solution and the emission spectrum: ``"plane_parallel"``
        (the same directions at every layer) or ``"spherical"`` (rays through the concentric shells, see
        :func:`tiramisu.rt.spherical.spherical_rays`). Transmission chords are always spherical.
        """
        if geometry not in ("plane_parallel", "spherical"):
            raise ValueError(f"geometry must be 'plane_parallel' or 'spherical', not {geometry!r}.")
        if central_pressure is None or pressure_levels is None:
            pressure_levels, central_pressure = build_pressure_profiles(
                boa_pressure=boa_pressure, toa_pressure=toa_pressure, nlayers=nlayers
            )
        self.interface_radii = hydrostatic_radii(
            temperature=temperature_profile,
            pressure_levels=pressure_levels,
            mu_profile=chemistry_profile.mean_molecular_weight,
            planet_mass=planet_mass,
            planet_radius=planet_radius,
            reference_pressure=pressure_levels[0] if reference_pressure is None else reference_pressure,
        )
        self.dz = np.diff(self.interface_radii)
        self.geometry = geometry
        self.planet_mass = planet_mass
        self.planet_radius = planet_radius
        self.chemistry_profile = chemistry_profile
        self.temperature_profile = temperature_profile
        self.pressure_levels = pressure_levels
        self.central_pressure = central_pressure
        self.density = density_profile(temperature=temperature_profile, central_pressure=central_pressure)

    @property
    def _nlte_interface_radii(self) -> t.Optional[u.Quantity]:
        return self.interface_radii if self.geometry == "spherical" else None

    def compute_emission(
            self,
            xsecs: XSecCollection,
            spectral_grid: t.Optional[u.Quantity] = None,
            output_intensity: bool = False,
            incident_radiation_field: t.Union[u.Quantity, t.Callable[..., u.Quantity], None] = None,
    ) -> t.Tuple[u.Quantity, u.Quantity, npt.NDArray[np.float64], t.Dict[SpeciesFormula, u.Quantity]]:
        """
        Returns the spectral grid, the emergent flux [W / (m^2 Hz)] (:func:`tiramisu.rt.plane_parallel.emergent_flux`,
        or :func:`tiramisu.rt.spherical.emergent_flux_spherical` normalised to ``planet_radius``), the per-layer optical
        depth and the per-species opacities. With ``output_intensity``, also writes the emergent flux and the
        hemispheric 4*pi*J profiles (at interfaces, :func:`tiramisu.rt.plane_parallel.four_pi_j_hemispheres`, or at
        layer centres, :func:`tiramisu.rt.spherical.four_pi_j_spherical`) to ``output_dir``.
        ``incident_radiation_field`` is as for :class:`tiramisu.xsec.XSecCollection`.
        """
        if spectral_grid is None:
            spectral_grid = xsecs.unified_grid

        opacities = xsecs.compute_opacities_profile(
            chem_profile=self.chemistry_profile,
            density_profile=self.density,
            dz_profile=self.dz,
            temperature=self.temperature_profile,
            pressure=self.central_pressure,
            spectral_grid=spectral_grid,
            interface_radii=self._nlte_interface_radii,
        )
        source_func, global_chi = self.source_function(xsecs=xsecs)

        res = global_chi * self.density[:, None] * self.dz[:, None]
        dtau = res.decompose().value
        spherical = self.geometry == "spherical"
        if spherical:
            flux_up = emergent_flux_spherical(dtau=dtau, source_function=source_func,
                                              interface_radii=self.interface_radii, planet_radius=self.planet_radius)
        else:
            mu_values, mu_weights = mu_quadrature()
            flux_up = emergent_flux(dtau=dtau, source_function=source_func, mu_values=mu_values,
                                    mu_weights=mu_weights)

        if output_intensity:
            if spherical:
                rays = spherical_rays(interface_radii=self.interface_radii)
                incident = incident_radiation_field
                if callable(incident):
                    incident = incident(mu_values=rays.mu[-1], mu_weights=rays.weights[-1])
                four_pi_j_up, four_pi_j_down = four_pi_j_spherical(
                    dtau=dtau, source_function=source_func, rays=rays, incident_radiation_field=incident,
                )
            else:
                incident = incident_radiation_field
                if callable(incident):
                    incident = incident(mu_values=mu_values, mu_weights=mu_weights)
                four_pi_j_up, four_pi_j_down = four_pi_j_hemispheres(
                    dtau=dtau,
                    source_function=source_func,
                    mu_values=mu_values,
                    mu_weights=mu_weights,
                    incident_radiation_field=incident,
                )
            out_name = "KELT-20b_nLTE_" if xsecs._is_converged else "KELT-20b_LTE_"
            # emergent_flux [W/(m^2 Hz)]; four_pi_j_up/down [J/m^2], bottom first, at every interface (plane-parallel)
            # or layer centre (spherical).
            for name, quantity in (
                    ("emergent_flux", flux_up), ("four_pi_j_up", four_pi_j_up), ("four_pi_j_down", four_pi_j_down)
            ):
                np.save((output_dir / f"{out_name}{name}.npy").resolve(), quantity.value)

        return spectral_grid, flux_up, dtau, opacities

    def source_function(self, xsecs: XSecCollection) -> t.Tuple[u.Quantity, u.Quantity]:
        """Source function and opacity of the last ``xsecs.compute_opacities_profile``, with negative (masing) opacity
        shielded as in the NLTE iteration (:func:`tiramisu.rt.source.effective_source_tau_mu`)."""
        effective_source_func, _, effective_chi, _ = effective_source_tau_mu(
            global_source_func_matrix=xsecs._global_source_func_matrix,
            global_chi_matrix=xsecs._global_chi_matrix,
            global_eta_matrix=xsecs._global_eta_matrix,
            density_profile=self.density,
            dz_profile=self.dz,
        )
        return effective_source_func, effective_chi

    def compute_transmission(
            self,
            xsecs: XSecCollection,
            star_radius: u.Quantity,
            spectral_grid: t.Optional[u.Quantity] = None,
            return_ratio: bool = False,
    ) -> t.Tuple[u.Quantity, u.Quantity, u.Quantity]:
        """
        Compute the transmission spectrum of the atmosphere using the slant-geometry integral over wavenumber grid
        :math:`\\tilde{\\nu}`.

        The transmission spectrum quantifies the wavenumber-dependent loss of stellar flux during transit:

        .. math::
            \\delta(\\tilde{\\nu}) = \\left( \\frac{R_{\\mathrm{p,eff}}(\\tilde{\\nu})}{R_{star}} \\right)^{2},

        where :math:`R_{\\mathrm{p,eff}}(\\tilde{\\nu})` is the effective radius of an opaque disk that would block the
        same stellar flux as the semi-transparent atmosphere.

        The effective radius follows from the optical-depth integral over impact parameters :math:`b`:

        .. math::
            R_{\\mathrm{p,eff}}^{2}(\\tilde{\\nu})
            = b_{0}^{2} + \\sum_{k=0}^{n_{\\mathrm{layers}}-1} \\left[ 1 - e^{-\\tau(b_{k},\\tilde{\\nu})} \\right]
                \\left( b_{k+1}^{2} - b_{k}^{2} \\right),

        where :math:`b_{0}` is the reference planetary radius, :math:`b_{k}` are the impact-parameter (annulus) edges
        and :math:`\\tau(b_{k},\\tilde{\\nu})` is the slant optical depth computed from

        .. math::
            \\tau(b, \\tilde{\\nu})
             = \\sum_{j=1}^{n_{\\mathrm{layers}}} \\alpha_{j}(\\tilde{\\nu}) \\, \\Delta s_{j}(b),

        where :math:`\\Delta s_{j}(b)` is the geometric path length through shell :math:`j` at impact parameter
        :math:`b`.

        Parameters
        ----------
        xsecs : XSecCollection
            Cross-section database used to compute extinction coefficients.
        star_radius : Quantity
            Stellar radius :math:`R_{star}` [m].
        spectral_grid : Quantity, optional
            Wavenumber grid :math:`\\tilde{\\nu}` with shape ``(n_wn,)``. If omitted, defaults to
            :func:``tiramisu.xsec.XSecCollection.unified_grid``.
        return_ratio : bool, optional
            If True, return :math:`R_{\\mathrm{p,eff}}/R_{star}` instead of the transit depth :math:`\\delta`.

        Returns
        -------
        spectral_grid : Quantity, shape (n_wn, )
            Wavenumber grid :math:`\\tilde{\\nu}`.
        transit_depth : Quantity, shape (n_wn,)
            Either :math:`\\delta(\\tilde{\\nu})` or :math:`R_{\\mathrm{p,eff}}/R_{star}` depending on ``return_ratio``.
        Rp_eff : Quantity, shape (n_wn, )
            Effective planetary radius :math:`R_{\\mathrm{p,eff}}(\\tilde{\\nu})` [m].
        """
        if spectral_grid is None:
            spectral_grid = xsecs.unified_grid

        _ = xsecs.compute_opacities_profile(
            chem_profile=self.chemistry_profile,
            density_profile=self.density,
            dz_profile=self.dz,
            temperature=self.temperature_profile,
            pressure=self.central_pressure,
            spectral_grid=spectral_grid,
            interface_radii=self._nlte_interface_radii,
        )

        # global_chi: (n_layers, n_wn) is mixing ratio weighted sum of absorption cross-sections [cm^2].
        _, global_chi = self.source_function(xsecs=xsecs)

        # Kappa is the mass extinction coefficient [m^2 / kg] and rho is mass density [kg/m^3].
        # alpha_nu_r is kappa*rho = (sigma/mu)*rho, but our self.density is number density i.e. n=rho/mu.
        # Avoid bringing mu back in and just get alpha_nu_r = sigma*n.
        # Volume extinction alpha: (n_wn, n_layers) [1/m]
        alpha_nu_r = (global_chi * self.density[:, None]).T.to(1 / u.m)

        # Shell edges and impact-parameter annulus edges are both the layer interfaces; one chord per shell midpoint.
        b_edges = self.interface_radii.to(u.m)  # (n_layers+1,)
        b_mid = 0.5 * (b_edges[:-1] + b_edges[1:])  # (n_layers,)
        tau_b_nu = slant_tau(alpha_nu_r=alpha_nu_r, r_edges=b_edges, b_mid=b_mid)  # (n_layers, n_wn)

        planet_radius_sq = effective_radius_squared(tau_b_nu, b_edges=b_edges)  # (n_wn,)
        planet_radius_eff = np.sqrt(planet_radius_sq).to(u.m)

        if return_ratio:
            delta = (planet_radius_eff / star_radius).decompose()
        else:
            delta = ((planet_radius_eff / star_radius) ** 2).decompose()

        return spectral_grid, delta, planet_radius_eff


def slant_tau(alpha_nu_r: u.Quantity, r_edges: u.Quantity, b_mid: u.Quantity) -> npt.NDArray[np.float64]:
    """
    Compute the *slant optical depth* :math:`\\tau(b, \\tilde{\\nu})` for a spherically symmetric atmosphere
    discretized into shells.

    For each wavenumber :math:`\\tilde{\\nu}` and impact parameter :math:`b`, the optical depth along the ray is:

    .. math::

        \\tau(b, \\tilde{\\nu}) = \\sum_{j=1}^{n_{\\mathrm{layers}}} \\alpha_j(\\tilde{\\nu}) \\, \\Delta s_j(b),

    where:

    * :math:`\\alpha_j(\\tilde{\\nu})`  
      is the volume extinction coefficient in layer :math:`j` [1/m].

    * :math:`\\Delta s_j(b)`  
      is the geometric path length through shell :math:`j` at impact
      parameter :math:`b`.

    Parameters
    ----------
    alpha_nu_r : Quantity, shape (n_wn, n_layers)
        Volume extinction coefficients :math:`\\alpha(\\tilde{\\nu}, j)` [1/m] for each wavenumber and layer.
        First dimension indexes wavenumber points :math:`n_{\\mathrm{wn}}`, second dimension indexes atmospheric layers.
    r_edges : Quantity, shape (n_layers + 1, )
        Radii of shell boundaries :math:`r_k` [m].
    b_mid : Quantity, shape (n_layers,)
        Impact parameters :math:`b_k` [m], typically midpoints between successive shell radii.

    Returns
    -------
    tau : ndarray, shape (n_layers, n_wn)
        Slant optical depth :math:`\\tau(b_k, \\tilde{\\nu}_i)`: first dimension indexes impact parameters :math:`b_k`,
        second dimension indexes wavenumber grid points :math:`\\tilde{\\nu}_i`. These values describe the total
        extinction along each chord (dimensionless).
    """
    r_edges_m = r_edges.to(u.m)
    b_mid_m = b_mid.to(u.m)
    alpha_nu_r = alpha_nu_r.to(1 / u.m)

    # Vectorized version of chord_lengths_through_shells over all impact parameters at once (broadcasting
    # (n_impacts, 1) against (1, n_layers)), rather than looping per-b and calling it once per row. The clip
    # before each sqrt keeps entries outside impact_mask from generating NaN/RuntimeWarning before being zeroed
    # out below - impact_mask guarantees r_out^2-b^2 >= 0 and rin_eff^2-b^2 >= 0 wherever it's True, matching the
    # unclipped per-b version exactly for those entries.
    r_in = r_edges_m[:-1].value
    r_out = r_edges_m[1:].value
    b = b_mid_m.value

    impact_mask = b[:, None] < r_out[None, :]
    r_in_eff = np.maximum(r_in[None, :], b[:, None])
    r_out_term = np.clip(r_out[None, :] ** 2 - b[:, None] ** 2, 0, None)
    r_in_term = np.clip(r_in_eff ** 2 - b[:, None] ** 2, 0, None)
    ds_mat_val = np.where(impact_mask, 2.0 * (np.sqrt(r_out_term) - np.sqrt(r_in_term)), 0.0)

    # alpha: (n_wn, n_layers), ds_mat: (n_impacts, n_layers) -> tau: (n_impacts,n_wn) [m * 1/m -> dimensionless]
    tau = np.einsum('kn,ln->kl', ds_mat_val, alpha_nu_r.value)
    return tau


def effective_radius_squared(tau_b_nu: npt.NDArray[np.float64], b_edges: u.Quantity) -> u.Quantity:
    """
    Compute the *effective planetary radius squared* :math:`R_{\\mathrm{p,eff}}^2(\\tilde{\\nu})` from the slant optical
    depth :math:`\\tau(b, \\tilde{\\nu})`.

    The fraction of stellar light removed by one annulus at impact parameter :math:`b` is:

    .. math::

       1 - e^{-\\tau(b, \\tilde{\\nu})}.

    The total occulted area of the atmosphere is then:

    .. math::

       A_{\\mathrm{atm}}(\\tilde{\\nu})
       = 2 \\int_{b_0}^{b_{\\mathrm{top}}} \\left[1 - e^{-\\tau(b, \\tilde{\\nu})}\\right] b \\, db.

    Discretizing using annulus edges :math:`b_k` gives:

    .. math::

       R_{\\mathrm{p,eff}}^2(\\tilde{\\nu})
       = b_0^2
         + \\sum_{k=0}^{n_{\\mathrm{layers}} - 1}
             \\left[
               1 - e^{-\\tau(b_k, \\tilde{\\nu})}
             \\right]
             \\left( b_{k+1}^2 - b_k^2 \\right),

    where :math:`b_0` is the lower boundary reference radius, below which the atmosphere is taken to be opaque.

    Parameters
    ----------
    tau_b_nu : ndarray, shape (n_layers, n_wn)
       Slant optical depths :math:`\\tau(b_k, \\tilde{\\nu}_i)` for each impact parameter :math:`b_k` and wavenumber
       :math:`\\tilde{\\nu}_i`.
    b_edges : Quantity, shape (n_layers + 1, )
       Radii of annulus edges :math:`b_k` [m].

    Returns
    -------
    planet_radius_sq : Quantity, shape (n_wn,)
       Effective planetary radius squared :math:`R_{\\mathrm{p,eff}}^2(\\tilde{\\nu})` [m^2] at each wavenumber. Transit
       depth is then computed via:

       .. math::

           \\delta(\\tilde{\\nu}) = \\left( \\frac{R_{\\mathrm{p,eff}}(\\tilde{\\nu})}{R_\\star} \\right)^2.
    """
    b_sq = (b_edges ** 2).to(u.m ** 2)
    dA = b_sq[1:] - b_sq[:-1]  # (n_layers, )
    occultation = 1.0 - np.exp(-tau_b_nu)  # (n_layers, n_wn)
    # Einstein summation is dimensionless by b_sq is m^2 so dimension is carried through.
    planet_radius_sq = b_sq[0] + np.einsum('k,kl->l', dA, occultation)  # (n_wn, )
    return planet_radius_sq.to(u.m ** 2)
