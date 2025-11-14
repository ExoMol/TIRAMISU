from astropy import units as u
from astropy import constants as const
import typing as t
import numpy as np


def planet_gravity(mass: u.Quantity, radius: u.Quantity) -> t.Callable[[u.Quantity], u.Quantity]:
    """
    Construct a function :math:`g(h)` that returns the gravitational acceleration at altitude :math:`h` above a planet
    of mass :math:`M_{\\mathrm{p}}` and radius :math:`R_{\\mathrm{p}}`.

    The gravitational acceleration follows:

    .. math::
        g(h) = \\frac{G M_{\\mathrm{p}}}{(R_{\\mathrm{p}} + h)^{2}}.

    Parameters
    ----------
    mass : Quantity
        Planetary mass :math:`M_{\\mathrm{p}}` [kg].
    radius : Quantity
        Reference radius :math:`R_{\\mathrm{p}}` [m].

    Returns
    -------
    gravity_at_height : Callable
        Function mapping height :math:`h` to :math:`g(h)` [m/s^2].
    """

    def gravity_at_height(height: u.Quantity, mass: u.Quantity = mass, radius: u.Quantity = radius) -> u.Quantity:
        """Calculates gravity at a given height."""
        return const.G * mass / (radius + height) ** 2

    return gravity_at_height


def density_profile(
        temperature: u.Quantity,
        central_pressure: u.Quantity,
) -> u.Quantity:
    """
    Compute number density from the ideal gas law at layer centers:

    .. math::
        \\rho = \\frac{P}{k_{\\mathrm{B}} T}.

    Parameters
    ----------
    temperature : Quantity, shape (n_layers,)
        Temperature :math:`T_{j}` of each layer [K].
    central_pressure : Quantity, shape (n_layers,)
        Pressure :math:`P_{j}` at layer centers [Pa].

    Returns
    -------
    density : Quantity, shape (n_layers,)
        Number density :math:`\\rho_{j}` [1/m^3].
    """
    return central_pressure / (const.k_B * temperature)


def build_pressure_profiles(
        boa_pressure: u.Quantity, toa_pressure: u.Quantity, nlayers: int
) -> t.Tuple[u.Quantity, u.Quantity]:
    """
    Construct logarithmically spaced pressure levels between the bottom of atmosphere (BOA) and the top of atmosphere
    (TOA).

    Taking the BOA pressure :math:`P_{0}` and TOA pressure :math:`P_{n}` over :math:`n = n_{\\mathrm{layers}}`, the
    pressure at each layer boundary satisfies:

    .. math::
        P_k = 10^{\\log_{10} P_{0} + k \\Delta}, \\qquad
        \\Delta = \\frac{\\log_{10}(P_{n}) - \\log_{10}(P_{0})}{n}.

    Layer-center pressures are then defined geometrically:

    .. math::
        P_{j,\\mathrm{center}} = \\sqrt{ P_{j} P_{j+1} }.

    Parameters
    ----------
    boa_pressure : Quantity
        Pressure at BOA :math:`P_{0}` [Pa].
    toa_pressure : Quantity
        Pressure at TOA :math:`P_{n}` [Pa].
    nlayers : int
        Number of layers :math:`n`.

    Returns
    -------
    pressure_levels : Quantity, shape (n_layers + 1,)
        Pressure at layer boundaries :math:`P_{0}, P_{1}, ..., P_{n}`.
    central_pressure : Quantity, shape (n_layers,)
        Layer-center pressures :math:`P_{j,\\mathrm{center}}`.
    """
    pressure_levels = (
            np.logspace(np.log10(boa_pressure.value), np.log10(toa_pressure.value), nlayers + 1) << boa_pressure.unit
    )

    central_pressure = pressure_levels[:-1] * np.sqrt(pressure_levels[1:] / pressure_levels[:-1])

    return pressure_levels, central_pressure


def scaleheight(
        height: u.Quantity,
        temperature: u.Quantity,
        mu: u.Quantity,
        gravity_function: t.Callable[[u.Quantity], u.Quantity],
) -> u.Quantity:
    """
    Compute the atmospheric scale height at altitude :math:`h`:

    .. math::
        H(h) = \\frac{ k_{\\mathrm{B}} T(h) }{ \\mu(h) \\, g(h) },

    where :math:`T(h)` is temperature, :math:`\\mu(h)` is the mean molecular mass, and :math:`g(h)` is the gravitational
    acceleration.

    Parameters
    ----------
    height : Quantity
        Altitude :math:`h` [m].
    temperature : Quantity
        Temperature :math:`T(h)` [K].
    mu : Quantity
        Mean molecular mass :math:`\\mu(h)` [kg].
    gravity_function : Callable
        Function returning :math:`g(h)`.

    Returns
    -------
    H : Quantity
        Scale height :math:`H(h)` [m].
    """
    return const.k_B * temperature / (mu * gravity_function(height))


def solve_scaleheight(
        temperature: u.Quantity,
        pressure_levels: u.Quantity,
        mu_profile: u.Quantity,
        gravity_function: t.Callable[[u.Quantity], u.Quantity],
) -> t.Tuple[u.Quantity, u.Quantity, u.Quantity, u.Quantity]:
    """
    Solve for altitude, scale height, gravity, and layer thickness across a discretized atmosphere.

    Pressure levels :math:`P_{k}` (k = 0..n) are provided, and altitude is reconstructed hydrostatically via:

    .. math::
        \\Delta z_{k} = - H_{k-1} \\, \\ln\\left( \\frac{P_{k}}{P_{k-1}} \\right),

    with cumulative altitude:

    .. math::
        z_{k} = z_{k-1} + \\Delta z_{k}.

    Scale height and gravity at each level follow:

    .. math::
        H_{k} = \\frac{k_{\\mathrm{B}} T_{k}}{\\mu_{k} g(z_{k})},
        \\qquad g_{k} = g(z_{k}).

    Returned arrays correspond to layer **centers**.

    Parameters
    ----------
    temperature : Quantity, shape (n_layers,)
        Layer-center temperatures :math:`T_{j}` [K].
    pressure_levels : Quantity, shape (n_layers + 1,)
        Pressure at layer boundaries :math:`P_{k}` [Pa].
    mu_profile : Quantity, shape (n_layers,)
        Mean molecular mass per layer :math:`\\mu_{j}` [kg].
    gravity_function : Callable
        Function :math:`g(h)`.

    Returns
    -------
    altitude : Quantity, shape (n_layers,)
        Altitude of layer centers :math:`z_{j}` [m].
    scale_height : Quantity, shape (n_layers + 1,)
        Scale height evaluated at pressure levels.
    gravity : Quantity, shape (n_layers,)
        Gravitational acceleration at layer centers.
    dz : Quantity, shape (n_layers,)
        Layer thickness :math:`\\Delta z_{j}` [m].
    """

    temperature = np.atleast_1d(temperature)
    pressure_levels = np.atleast_1d(pressure_levels)
    mu_profile = np.atleast_1d(mu_profile)

    surface_scale_height = scaleheight(0 * u.m, temperature[0], mu_profile[0], gravity_function)
    surface_gravity = gravity_function(0 * u.m)
    nlevels = pressure_levels.shape[0]

    altitude = np.zeros_like(pressure_levels.value) << surface_scale_height.unit

    scale_height = np.zeros_like(pressure_levels.value) << surface_scale_height.unit

    dz = np.zeros_like(pressure_levels.value) << surface_scale_height.unit

    gravity = np.zeros_like(pressure_levels.value) << surface_gravity.unit
    scale_height[0] = surface_scale_height
    gravity[0] = surface_gravity

    pressure_diff = np.log(pressure_levels[1:] / pressure_levels[:-1])

    for i in range(1, nlevels):
        dz[i] = -scale_height[i - 1] * pressure_diff[i - 1]
        altitude[i] = altitude[i - 1] + dz[i]
        gravity[i] = gravity_function(altitude[i])
        if i < nlevels - 1:
            scale_height[i] = scaleheight(altitude[i], temperature[i], mu_profile[i], gravity_function)

    return altitude[:-1], scale_height, gravity[:-1], dz[:-1]
