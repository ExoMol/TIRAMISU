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
from .xsec import XSecCollection, ExomolNLTEXsec
from .nlte import blackbody
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
            incident_radiation_field: u.Quantity = None
    ) -> t.Tuple[u.Quantity, u.Quantity, u.Quantity, t.Dict[SpeciesFormula, u.Quantity]]:

        if spectral_grid is None:
            spectral_grid = xsecs.unified_grid

        spectral_grid = spectral_grid

        self.mu_tau, self.mu_weights = emission_quadratures(4)
        opacities = xsecs.compute_opacities_profile(
            self.chemistry_profile,
            self.temperature_profile,
            self.central_pressure,
            spectral_grid,
        )
        opac_dens = [x * self.density[:, None] for x in opacities.values()]
        dtau = boa_toa_optical_depth(opac_dens, self.dz)

        source_func, global_chi, global_eta = self.source_function(spectral_grid, xsecs, opacities)


        # These two lines are now defunct and their results are not used.
        emission_mu, emission_tau = emission_1d(dtau, self.mu_tau, source_func)
        emission = integrate_emission_quadrature(emission_mu, self.mu_tau, self.mu_weights)

        for species in xsecs:
            if type(xsecs[species]) is ExomolNLTEXsec:
                xsecs[species].density_profile = self.density
                xsecs[species].dz_profile = self.dz
                xsecs[species].tau_matrix = emission_tau
                xsecs[species].global_chi_matrix = sum(opac_dens)
                xsecs[species].global_eta_matrix = global_eta
                xsecs[species].global_source_func_matrix = source_func  # Now redundant?

        if output_intensity:
            mu_values, mu_weights = np.polynomial.legendre.leggauss(50)
            mu_values, mu_weights = (mu_values + 1) * 0.5, mu_weights / 2
            res = global_chi * self.density[:, None] * self.dz[:, None]
            dtau = res.decompose().value
            i_up, i_down = formal_solve_general(
                dtau=dtau,
                source_function=source_func,
                mu_values=mu_values,
                mu_weights=mu_weights,
                incident_radiation_field=incident_radiation_field,
            )

            if xsecs.is_converged():
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

        return spectral_grid, emission, emission_tau, opacities

    def source_function(
            self, spectral_grid: u.Quantity, xsecs: XSecCollection, opacities: t.Dict[SpeciesFormula, u.Quantity]
    ) -> t.Tuple[u.Quantity, u.Quantity, u.Quantity]:
        lte_fraction = np.ones(len(self.temperature_profile))
        # # nlte_source_func = np.zeros((len(self.temperature_profile), len(spectral_grid))) << u.J / (u.s * u.sr * u.m ** 3)  # LAMBDA
        # nlte_source_func = np.zeros((len(self.temperature_profile), len(spectral_grid))) << u.J / (
        #     u.sr * u.m**2
        # )  # FREQUENCY
        # # nlte_source_func = np.zeros((len(self.temperature_profile), len(spectral_grid))) << u.J / u.m ** 2  # WAVENUMBER
        chi = np.zeros((len(self.temperature_profile), len(spectral_grid))) << u.cm ** 2
        eta = np.zeros((len(self.temperature_profile), len(spectral_grid))) << u.erg * u.cm / (u.s * u.sr)

        lte_source_func = blackbody(spectral_grid, self.temperature_profile)

        # Chi and Eta are both in terms of per molecule. In each layer the number density iis the same so we can just
        # multiply by the VMR (species_profile) to obtain the contributions from each).
        for species in xsecs:
            if species in self.chemistry_profile.species:
                species_profile = self.chemistry_profile[species]
                if type(xsecs[species]) is ExomolNLTEXsec and xsecs[species].mol_chi_matrix is not None:
                    # Remove the NLTE fractional abundances to scale the LTE blackbody contributions.
                    lte_fraction -= species_profile
                    # Multiply species source function by fractional abundance
                    # nlte_source_func += xsecs[species].mol_source_func_matrix * species_profile[:, None]
                    chi += xsecs[species].mol_chi_matrix * species_profile[:, None]
                    eta += xsecs[species].mol_eta_matrix * species_profile[:, None]
                else:
                    # Chi is known for the LTE species, Eta = Blackbody*chi
                    chi += opacities[species] * species_profile[:, None]
                    eta += opacities[species] * species_profile[:, None] * lte_source_func * ac.c

        negative_chi_cap = - 1e-5 * chi.max(axis=1)[:, None]
        chi = np.clip(chi, min=negative_chi_cap)
        negative_source_func_cap = - lte_source_func.max(axis=1)[:, None]
        # if sum(lte_fraction) == len(self.temperature_profile):
        #     combined_source_func = blackbody(spectral_grid, self.temperature_profile)
        # elif sum(lte_fraction) != 0:
        #     lte_source_func *= lte_fraction[:, None]
        #     combined_source_func = lte_source_func + nlte_source_func
        # else:
        #     combined_source_func = nlte_source_func

        combined_source_func = (eta / (ac.c * chi)).to(u.J / (u.sr * u.m ** 2), equivalencies=u.spectral())
        # Limit the effects of stimulated emission to avoid exponential overflows.
        # combined_source_func[combined_source_func < negative_source_func_cap] = negative_source_func_cap
        combined_source_func = np.clip(combined_source_func, min=negative_source_func_cap)

        # for species in xsecs:
        #     if type(xsecs[species]) is ExomolNLTEXsec:
        #         xsecs[species].global_source_func_matrix = combined_source_func

        return combined_source_func, chi, eta


def formal_solve_general(
        dtau: u.Quantity,
        source_function: u.Quantity,
        mu_values: npt.NDArray[np.float64],
        mu_weights: npt.NDArray[np.float64],
        incident_radiation_field: u.Quantity = None,
        surface_albedo: float = 0
) -> t.Tuple[u.Quantity, u.Quantity]:
    """
    Calculates the upward and downward intensity at the interface between each layer.
    Index 0 is the Bottom of the Atmosphere (BOA).

    :param dtau:                     Optical depth of each layer.
    :param source_function:          Source function in each layer.
    :param mu_values:                Array of mu cosines.
    :param mu_weights:               Array of angular integration weights.
    :param incident_radiation_field: Radiation field incident on the top of the atmosphere (J. m^2).
    :param surface_albedo:           Albedo of the planet's surface, in the range [0, 1] [Not tested].

    :return: Upwards and Downwards directed intensity at the interface of each layer.
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
    # thermal_emission = bb(dtau.shape[1] * u.nm) # Example wavelength
    # thermal_emission = source_function[-1] * surface_emissivity # Placeholder
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
