from astropy import units as u
from astropy import constants as const
import typing as t
import numpy as np


def planet_gravity(mass: u.Quantity, radius: u.Quantity) -> t.Callable[[u.Quantity], u.Quantity]:
    """Builds gravity function for given planet."""

    def gravity_at_height(height: u.Quantity, mass: u.Quantity = mass, radius: u.Quantity = radius) -> u.Quantity:
        """Calculates gravity at a given height."""
        return const.G * mass / (radius + height) ** 2

    return gravity_at_height


def density_profile(
    temperature: u.Quantity,
    central_pressure: u.Quantity,
) -> u.Quantity:
    """Calculates the density profile of an atmosphere."""
    return central_pressure / (const.k_B * temperature)


def build_pressure_profiles(
    boa_pressure: u.Quantity, toa_pressure: u.Quantity, nlayers: int
) -> t.Tuple[u.Quantity, u.Quantity]:
    """Builds a pressure profile from the boundary and top of atmosphere pressures.

    Args:
        boa_pressure: The boundary of atmosphere pressure.
        toa_pressure: The top of atmosphere pressure.
        nlayers: The number of layers in the profile.
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
    """Calculates scaleheight at a given height."""
    return const.k_B * temperature / (mu * gravity_function(height))


def solve_scaleheight(
    temperature: u.Quantity,
    pressure_levels: u.Quantity,
    mu_profile: u.Quantity,
    gravity_function: t.Callable[[u.Quantity], u.Quantity],
) -> t.Tuple[u.Quantity, u.Quantity, u.Quantity, u.Quantity]:
    """Solves scaleheight for an atmosphere."""

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
