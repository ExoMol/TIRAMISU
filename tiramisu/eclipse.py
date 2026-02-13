import typing as t
import numpy.typing as npt
from astropy import units as u
from astropy import constants as ac
import numpy as np

from .chemistry import ChemicalProfile, SpeciesFormula
from .atmos import (
    build_pressure_profiles,
    planet_gravity,
    density_profile,
    solve_scaleheight,
)
from .xsec import XSecCollection
from .nlte import blackbody, formal_solve_general
from .config import log, output_dir


def emission_quadratures(
        ngauss: int,
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute the emission quadratures.

    Args:
        ngauss: Number of gauss points

    Returns:
        t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: abscissa and weights
    """
    mu, weight = np.polynomial.legendre.leggauss(ngauss)
    return (mu + 1) * 0.5, weight / 2


def emission_1d(
        dtau: u.Quantity,
        mu_tau: npt.NDArray[np.float64],
        source_function: u.Quantity,
) -> t.Tuple[u.Quantity, u.Quantity]:
    """1D emission.

    Args:
        dtau: Optical depth for each layer
        mu_tau: Cosine of the angle between the ray and the normal
        source_function: Source function

    Returns:
        t.Tuple[u.Quantity, u.Quantity]: Emission at each quadrature and optical depth

    """
    # dtau goes from boa to toa so we need to reverse it
    dtau_toa = dtau[::-1]

    inv_mu = 1 / mu_tau

    tau = dtau_toa.cumsum(axis=0)[::-1]

    layer_tau = tau - dtau

    emission_tau = np.exp(-layer_tau.value) - np.exp(-tau.value)

    intensity = (
                        np.exp(-inv_mu[:, None, None] * layer_tau[None, ...].value)
                        - np.exp(-inv_mu[:, None, None] * tau[None, ...].value)
                ) * source_function[None, ...]

    surface = np.exp(-inv_mu[:, None] * tau[None, 0, ...].value) * source_function[None, 0]

    return intensity.sum(axis=1) + surface, emission_tau


def boa_toa_optical_depth(xsec_opacity: t.List[u.Quantity], dz: u.Quantity) -> u.Quantity:
    """Compute the boa to toa optical depth.

    Args:
        xsec_opacity: Cross-section opacity
        dz: Layer thickness

    Returns:
        npt.NDArray[np.float64]: Optical depth for the layer
    """
    sum_opacity = sum(xsec_opacity)
    res = sum_opacity * dz[:, None]
    return res.decompose()


def integrate_emission_quadrature(
        emission_mu: u.Quantity,
        mu_tau: npt.NDArray[np.float64],
        mu_weights: npt.NDArray[np.float64],
) -> u.Quantity:
    """
    Deprecated. See :func:`formal_solve_general` for the current implementation.

    :param emission_mu:
    :param mu_tau:
    :param mu_weights:
    :return:
    """
    emission_mu = emission_mu

    return 2 * np.pi * (emission_mu * mu_tau[:, None] * mu_weights[:, None]).sum(axis=0) * u.sr


class ExoplanetEmission:

    def __init__(
            self,
            planet_mass: u.Quantity,
            planet_radius: u.Quantity,
            temperature_profile: u.Quantity,
            boa_pressure: u.Quantity,
            toa_pressure: u.Quantity,
            nlayers: int,
            chemistry_profile: ChemicalProfile,
            ngauss: int = 4,
            central_pressure: u.Quantity = None,
            pressure_levels: u.Quantity = None,
    ):

        self.planet_mass = planet_mass
        self.planet_radius = planet_radius
        self.boa_pressure = boa_pressure
        self.toa_pressure = toa_pressure
        self.nlayers = nlayers
        self.chemistry_profile = chemistry_profile
        self.temperature_profile = temperature_profile
        self.ngauss = ngauss
        self.central_pressure = central_pressure
        self.pressure_levels = pressure_levels
        self.build_atmosphere()

    def build_atmosphere(self):
        self.planet_gravity = planet_gravity(self.planet_mass, self.planet_radius)

        if self.central_pressure is None or self.pressure_levels is None:
            self.pressure_levels, self.central_pressure = build_pressure_profiles(
                self.boa_pressure, self.toa_pressure, self.nlayers
            )

        self.altitude, self.scale_height, self.gravity, self.dz = solve_scaleheight(
            self.temperature_profile,
            self.pressure_levels,
            self.chemistry_profile.mean_molecular_weight,
            self.planet_gravity,
        )

        self.density = density_profile(self.temperature_profile, self.central_pressure)

    def compute_emission(
            self,
            xsecs: XSecCollection,
            spectral_grid: t.Optional[u.Quantity] = None,
            output_intensity: bool = False,
            incident_radiation_field: u.Quantity = None,
            approximate_t_ex: bool = False,
    ) -> t.Tuple[u.Quantity, u.Quantity, u.Quantity, npt.NDArray[np.float64], t.Dict[SpeciesFormula, u.Quantity]]:

        if spectral_grid is None:
            spectral_grid = xsecs.unified_grid

        spectral_grid = spectral_grid

        self.mu_tau, self.mu_weights = emission_quadratures(4)
        opacities = xsecs.compute_opacities_profile(
            chemical_profile=self.chemistry_profile,
            density_profile=self.density,
            dz_profile=self.dz,
            temperature=self.temperature_profile,
            pressure=self.central_pressure,
            spectral_grid=spectral_grid,
            approximate_t_ex=approximate_t_ex,
        )
        source_func, global_chi, global_eta = self.source_function(spectral_grid, xsecs)
        np.save(
            fr"/mnt/c/PhD/NLTE/Models/KELT-20b/approximation/ohx1e0_source.npy",
            source_func.value
        )

        mu_values, mu_weights = np.polynomial.legendre.leggauss(50)
        mu_values, mu_weights = (mu_values + 1) * 0.5, mu_weights / 2
        res = global_chi * self.density[:, None] * self.dz[:, None]
        dtau = res.decompose().value
        # tau = dtau[::-1].cumsum(axis=0)[::-1]
        i_up, i_down = formal_solve_general(
            dtau=dtau,
            source_function=source_func,
            mu_values=mu_values,
            mu_weights=mu_weights,
            incident_radiation_field=incident_radiation_field,
        )

        if output_intensity:
            if xsecs.is_converged:
                out_name = "KELT-20b_nLTE_intensity_"
            else:
                out_name = "KELT-20b_LTE_intensity_"
            up_name = out_name + "up.txt"
            np.savetxt(
                (output_dir / up_name).resolve(),
                i_up.value,
                fmt="%17.8E",
            )
            down_name = out_name + "down.txt"
            np.savetxt(
                (output_dir / down_name).resolve(),
                i_down.value,
                fmt="%17.8E",
            )

        # return spectral_grid, emission, emission_tau, opacities
        return spectral_grid, i_up, i_down, dtau, opacities

    def source_function(
            self, spectral_grid: u.Quantity, xsecs: XSecCollection,
    ) -> t.Tuple[u.Quantity, u.Quantity, u.Quantity]:
        lte_source_func = blackbody(spectral_grid, self.temperature_profile)

        negative_chi_cap = - 1e-6 * xsecs.global_chi_matrix.max(axis=1)[:, None]
        effective_chi = np.clip(xsecs.global_chi_matrix, min=negative_chi_cap)
        negative_source_func_cap = - lte_source_func.max(axis=1)[:, None]

        effective_source_func = (xsecs.global_eta_matrix / (ac.c * effective_chi)).to(u.J / (u.sr * u.m ** 2),
                                                                                      equivalencies=u.spectral())
        # Limit the effects of stimulated emission to avoid exponential overflows.
        # combined_source_func[combined_source_func < negative_source_func_cap] = negative_source_func_cap
        effective_source_func = np.clip(effective_source_func, min=negative_source_func_cap)

        return effective_source_func, effective_chi, xsecs.global_eta_matrix

    def compute_transmission(
            self,
            xsecs: XSecCollection,
            star_radius: u.Quantity,
            spectral_grid: t.Optional[u.Quantity] = None,
            return_ratio: bool = False,
            approximate_t_ex: bool = False,
    ) -> t.Tuple[u.Quantity, u.Quantity, u.Quantity]:
        """
        Compute the transmission spectrum of the atmosphere using the slant-geometry integral over wavenumber grid
        :math:`\\tilde{\\nu}`.

        The transmission spectrum quantifies the wavenumber-dependent loss of stellar flux during transit:

        .. math::
            \\delta(\\tilde{\\nu}) = \\left( \\frac{R_{\\mathrm{p,eff}}(\\tilde{\\nu})}{R_{star}} \\right)^{2,

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
            chemical_profile=self.chemistry_profile,
            density_profile=self.density,
            dz_profile=self.dz,
            temperature=self.temperature_profile,
            pressure=self.central_pressure,
            spectral_grid=spectral_grid,
            approximate_t_ex=approximate_t_ex,
        )

        # global_chi: (n_layers, n_wn) is mixing ratio weighted sum of absorption cross-sections [cm^2].
        _, global_chi, _ = self.source_function(spectral_grid, xsecs)
        
        # Kappa is the mass extinction coefficient [m^2 / kg] and rho is mass density [kg/m^3].
        # alpha_nu_r is kappa*rho = (sigma/mu)*rho, but our self.density is number density i.e. n=rho/mu.
        # Avoid bringing mu back in and just get alpha_nu_r = sigma*n.
        # Volume extinction alpha: (n_wn, n_layers) [1/m]
        alpha_nu_r = (global_chi * self.density[:, None]).T.to(1 / u.m)

        # Geometry: build shell edges
        # self.altitude is ALWAYS stored at layer centers (length n_layers),
        # so reconstruct layer-edge radii from dz:
        dz = self.dz.to(u.m)  # (n_layers,)
        cumulative = np.concatenate(([0.0], np.cumsum(dz.value))) * u.m
        r_edges = (self.planet_radius + cumulative).to(u.m)  # (n_layers+1,)

        # Impact parameter grid (same as shell edges)
        b_edges = r_edges
        b_mid = 0.5 * (b_edges[:-1] + b_edges[1:])  # (n_layers,)
        tau_b_nu = slant_tau(alpha_nu_r, r_edges, b_mid)  # (n_layers, n_wn)

        planet_radius_sq = effective_radius_squared(tau_b_nu, b_edges=b_edges)  # (n_wn,)
        planet_radius_eff = np.sqrt(planet_radius_sq).to(u.m)

        if return_ratio:
            delta = (planet_radius_eff / star_radius).decompose()
        else:
            delta = ((planet_radius_eff / star_radius) ** 2).decompose()

        return spectral_grid, delta, planet_radius_eff


def chord_lengths_through_shells(b: float, r_edges: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Compute the geometric path length of a slant/tangent ray through each atmospheric layer, assuming spherically
    symmetric shells, for a given impact parameter :math:`b`.

    A straight ray at impact parameter :math:`b` intersects each shell bounded by radii :math:`r_{\\mathrm{in}}` and
    :math:`r_{\\mathrm{out}}` over a path length:

    .. math::
        \\Delta{}s_{j}(b) = 2 \\left[
            \\sqrt{r_{\\mathrm{out}}^{2} - b^{2}}
            - \\sqrt{max(r_{\\mathrm{in}}, b)^{2} - b^{2}}
            \\right],

    uch that only shells with :math:`r_{\\mathrm{out}} \\lt b` contribute (others are not intersected).

    Parameters
    ----------
    b : float
        Impact parameter (distance of the ray’s closest approach to the planet’s center) [m]. Corresponds to the tangent
        altitude of a slant path through the atmosphere.
    r_edges : ndarray, shape (n_layers + 1, )
        Radii of the layer boundaries :math:`r_0, r_1, ..., r_{n}` defining :math:`n_{\\mathrm{layers}}` spherical
        shells [m].

    Returns
    -------
    ds : ndarray, shape (n_layers,)
        Geometric path length through each shell [m]. This array represents :math:`\\Delta{}s_{j}(b)` for each layer
        :math:`j` and is used in the slant optical depth integral:

        .. math::
            \\tau(b, \\tilde{\\nu}) = \\sum_{j=1}^{n_{\\mathrm{layers}}} \\alpha_j(\\tilde{\\nu}) \\, \\Delta s_j(b),

        where :math:`\\alpha_j(\\tilde{\\nu})` is the volume extinction coefficient in layer :math:`j`.
    """
    r_in = r_edges[:-1]
    r_out = r_edges[1:]
    ds = np.zeros_like(r_in)
    # Only shells with outer radius > b are intersected.
    impact_mask = b < r_out
    if not np.any(impact_mask):
        return ds
    rin_eff = np.maximum(r_in[impact_mask], b)
    ds[impact_mask] = 2.0 * (np.sqrt(r_out[impact_mask] ** 2 - b ** 2) - np.sqrt(rin_eff ** 2 - b ** 2))
    return ds


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
        extinction along each chord [1/m].
    """
    n_wn, n_layers = alpha_nu_r.shape
    n_impacts = len(b_mid)
    r_edges_m = r_edges.to(u.m)
    b_mid_m = b_mid.to(u.m)
    alpha_nu_r = alpha_nu_r.to(1 / u.m)

    ds_mat = np.zeros((n_impacts, n_layers)) << r_edges_m.unit
    for k, b in enumerate(b_mid_m):
        ds_mat[k, :] = chord_lengths_through_shells(b, r_edges_m)


    # alpha: (n_wn, n_layers), ds_mat: (n_impacts, n_layers) -> tau: (n_impacts,n_wn) [m * 1/m -> dimensionless]
    tau = np.einsum('kn,ln->kl', ds_mat.value, alpha_nu_r.value)
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
