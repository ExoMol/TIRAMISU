from astropy import units as u
from astropy import constants as const
import typing as t
import numpy as np


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


def hydrostatic_radii(
        temperature: u.Quantity,
        pressure_levels: u.Quantity,
        mu_profile: u.Quantity,
        planet_mass: u.Quantity,
        planet_radius: u.Quantity,
        reference_pressure: u.Quantity,
) -> u.Quantity:
    """
    Radii of the layer interfaces of an atmosphere in hydrostatic equilibrium, placed so that the radius at
    ``reference_pressure`` is ``planet_radius``.

    With gravity :math:`g = G M_{\\mathrm{p}} / r^{2}` and an ideal gas, hydrostatic equilibrium
    :math:`\\mathrm{d}P/\\mathrm{d}r = -P \\mu g / (k_{\\mathrm{B}} T)` is linear in :math:`1/r`:

    .. math::
        \\mathrm{d}\\left(\\frac{1}{r}\\right) = \\frac{k_{\\mathrm{B}} T}{\\mu G M_{\\mathrm{p}}} \\, \\mathrm{d}\\ln P,

    so a layer of constant :math:`T_j` and :math:`\\mu_j` between interfaces :math:`k = j` and :math:`k = j + 1`
    integrates exactly:

    .. math::
        \\frac{1}{r_{j+1}} = \\frac{1}{r_{j}} + \\frac{k_{\\mathrm{B}} T_{j}}{\\mu_{j} G M_{\\mathrm{p}}}
            \\ln\\frac{P_{j+1}}{P_{j}}.

    :math:`1/r` is linear in :math:`\\ln P` within each layer, so the reference level may lie anywhere in the grid and
    is interpolated in :math:`\\ln P`.

    Parameters
    ----------
    temperature : Quantity, shape (n_layers,)
        Layer-centre temperatures :math:`T_j` [K].
    pressure_levels : Quantity, shape (n_layers + 1,)
        Pressure at the layer interfaces :math:`P_k`, bottom first.
    mu_profile : Quantity, shape (n_layers,)
        Mean molecular mass per layer :math:`\\mu_j`.
    planet_mass : Quantity
        Planetary mass :math:`M_{\\mathrm{p}}`.
    planet_radius : Quantity
        Radius at ``reference_pressure``.
    reference_pressure : Quantity
        Pressure at which the radius is ``planet_radius``, between ``pressure_levels[-1]`` and ``pressure_levels[0]``.

    Returns
    -------
    interface_radii : Quantity, shape (n_layers + 1,)
        Radii of the layer interfaces, bottom first [m].
    """
    log_p = np.log(pressure_levels.to_value(u.Pa))
    log_p_ref = np.log(reference_pressure.to_value(u.Pa))
    if not log_p[-1] <= log_p_ref <= log_p[0]:
        raise ValueError(f"reference_pressure {reference_pressure} is outside the pressure grid "
                         f"({pressure_levels[-1]} to {pressure_levels[0]}).")
    inv_r_step = (const.k_B * temperature / (mu_profile * const.G * planet_mass)).to_value(1 / u.m) * np.diff(log_p)
    inv_r = np.concatenate(([0.0], np.cumsum(inv_r_step)))
    # np.interp needs increasing abscissae: -ln P increases upwards.
    inv_r += 1 / planet_radius.to_value(u.m) - np.interp(-log_p_ref, -log_p, inv_r)
    if inv_r[-1] <= 0:
        raise ValueError("The atmosphere is not bound: 1/r reaches zero below the top of the pressure grid, so this "
                         "temperature and mean molecular mass are too high/low for the planet's gravity.")
    return (1 / inv_r) << u.m
