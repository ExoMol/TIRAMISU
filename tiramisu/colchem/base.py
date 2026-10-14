r"""
Shared pieces of the collisional/chemical rate database: the rate record, per-species metadata, temperature
interpolation of tabulated rates, and the gas-kinetic ceiling used to cap estimated rates. The conventions here hold
for every species module; update this docstring whenever they change (see ``colchem.py``).

.. rubric:: Physical constants

Physical constants come from ``tiramisu.constants`` (``astropy.constants``, CODATA 2018), never literals: the
Boltzmann constant and atomic mass unit here, and :math:`h c / k_B` and the atmosphere in the species modules
(since 2026-10-14).

.. rubric:: How rates enter the statistical equilibrium

A ``RateTransition`` holds one rate coefficient :math:`k` [:math:`\mathrm{cm^3\,s^{-1}}`] from level ``upper_key``
(:math:`U`) to ``lower_key`` (:math:`L`), to be multiplied by the number density :math:`n_s` of every species
:math:`s` in ``species_depend``. ``NLTEProcessor._build_col_chem_cache`` adds them, per layer, to the rate matrix
:math:`Y` and source vector :math:`R` of :math:`Y n = R` (:math:`n` the level populations):

- Transition (:math:`U \ne L`, :math:`k > 0`): :math:`c_{UL} = k \prod_s n_s` per molecule in :math:`U`
  [:math:`\mathrm{s^{-1}}`], and the reverse by detailed balance, :math:`c_{LU} = c_{UL}\, n^*_U / n^*_L`, with
  :math:`n^*` the LTE population of each aggregate (the Boltzmann sum over its states). Collisions alone therefore
  return LTE. Aggregate populations, rather than :math:`e^{-\Delta E / k_B T}` between the aggregates' lowest states,
  keep each manifold's degeneracy and rotational partition function. Species modules list downward channels only.
- Destruction (:math:`U = L`, :math:`k < 0`): a loss :math:`|k| \prod_s n_s` per molecule on the diagonal of
  :math:`Y`.
- Formation (:math:`U = L`, :math:`k > 0`): :math:`k \prod_s n_s` is formation per unit volume
  [:math:`\mathrm{cm^{-3}\,s^{-1}}`]; divided by the formed species' own number density, it goes into :math:`R`.
- A rate is skipped when either key matches no aggregate, when either aggregate lies above the aggregation cutoff, or
  when a species in ``species_depend`` is missing from the chemical profile.

.. rubric:: Temperature interpolation

Tabulated rates are interpolated log-linearly in :math:`T` (piecewise power laws: the rates span orders of magnitude
and are roughly power laws in :math:`T`), linearly where a grid value is zero, and held at the end values outside
the grid; nothing is extrapolated (``_interp_rate``).

.. rubric:: Gas-kinetic ceiling

.. math::

    k_{\mathrm{gk}} = \sigma \langle v_{\mathrm{rel}} \rangle, \qquad
    \langle v_{\mathrm{rel}} \rangle = \left(\frac{8 k_B T}{\pi \mu}\right)^{1/2},

with :math:`\mu` the reduced mass of the target (default OH, 17.007 amu) and the collider (``_COLLIDER_MASS``), and
:math:`\sigma` the cross section (``_gas_kinetic_rate``).

- :math:`\sigma` = 30 :math:`\mathrm{\mathring{A}^2}` by default, 1.2-1.4 times the hard-sphere values for H2O with He
  and with itself (22.6 and 24.6 :math:`\mathrm{\mathring{A}^2}` at Zittel & Masturzo's 1989 diameters).
- 100 :math:`\mathrm{\mathring{A}^2}` for H2O as the collider (``_COLLIDER_CROSS_SECTION``): dipole-dipole attraction
  makes these collisions larger; OH(A) quenching by H2O alone has 85 :math:`\mathrm{\mathring{A}^2}` at 300 K (Tamura
  et al. 1998), and impact parameters up to 8 angstrom contribute to H2O self-relaxation (Braunstein & Conforti 2015).
- Species modules use it to bound a level's total loss to one collider where rates are estimated (``h2o.py``: every
  level; ``oh.py``: the extended set). Each module's docstring says where it binds.
"""
import typing as t
from dataclasses import dataclass

import numba
import numpy as np
import numpy.typing as npt

from ..constants import const_amu, const_k_B


@dataclass
class RateTransition:
    """
    Single collisional or chemical rate transition. ``rate`` times the number density of every species in
    ``species_depend`` is the rate per molecule of the upper level [s^-1]; for formation (upper_key == lower_key,
    rate > 0) it is formation per unit volume [cm^-3 s^-1], which NLTEProcessor divides by the formed species' own
    number density.
    """
    upper_key: int | t.Tuple[str, int] | t.Tuple[int, int, int, str] | t.Tuple[int, int, int]
    lower_key: int | t.Tuple[str, int] | t.Tuple[int, int, int, str] | t.Tuple[int, int, int]
    rate: float  # cm^3/s for collisions and bimolecular formation (cm^6/s for termolecular)
    species_depend: t.Tuple[str, ...]  # Species (molecules, atoms, electrons) whose number densities multiply the rate


@dataclass
class SpeciesRateInfo:
    """
    Metadata about collisional rates for a species.

    Attributes
    ----------
    temperature_dependent : bool
        Whether rates require temperature interpolation
    collision_partners : set of str
        Species for which collision rates are configured (e.g., {'O2', 'O', 'He', 'H'})
    """
    temperature_dependent: bool
    collision_partners: t.Set[str]


@numba.njit(cache=True, error_model="numpy")
def _interp_rate(temp: float, temp_grid: npt.NDArray[np.float64], rate_grid: npt.NDArray[np.float64]) -> float:
    """
    Temperature interpolation of collisional rate coefficients.

    Uses log-linear (piece-wise power-law) interpolation, which is physically appropriate because collisional rates
    vary over many orders of magnitude and are approximately power-law in temperature.  A linear fallback is used
    for any zero-valued grid points so that near-zero rates at low temperature are handled gracefully rather than
    producing -inf in log space.

    Module-level (rather than a class staticmethod) so :func:`_interp_rate_batch` below can call it directly by
    name from within its own jitted body - numba resolves plain module globals reliably for nopython-to-nopython
    calls, whereas calling through a class attribute expression from inside jitted code is not a supported pattern.
    """
    # Find bracketing indices via np.interp on a dummy linear scale
    num_t = len(temp_grid)
    # Clamp to grid range
    if temp <= temp_grid[0]:
        return rate_grid[0]
    if temp >= temp_grid[num_t - 1]:
        return rate_grid[num_t - 1]

    # Binary-search for the left bracket index
    lo = 0
    hi = num_t - 1
    while hi - lo > 1:
        mid: int = (lo + hi) // 2
        if temp_grid[mid] <= temp:
            lo = mid
        else:
            hi = mid

    rate_lo = rate_grid[lo]
    rate_hi = rate_grid[hi]
    t_lo = temp_grid[lo]
    t_hi = temp_grid[hi]

    # Log-linear interpolation; fall back to linear if either endpoint is zero
    if rate_lo > 0.0 and rate_hi > 0.0:
        log_r = np.log(rate_lo) + (np.log(rate_hi) - np.log(rate_lo)) * (temp - t_lo) / (t_hi - t_lo)
        return np.exp(log_r)
    else:
        return rate_lo + (rate_hi - rate_lo) * (temp - t_lo) / (t_hi - t_lo)


@numba.njit(cache=True, error_model="numpy")
def _interp_rate_batch(
        temp_profile: npt.NDArray[np.float64],
        temp_grid: npt.NDArray[np.float64],
        rate_matrix: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Interpolate every transition in ``rate_matrix`` at every temperature in ``temp_profile`` in one jitted call.

    Replaces calling :func:`_interp_rate` once per (layer, transition) pair from Python - each such call pays a
    small but non-negligible Python-to-numba dispatch cost, which adds up when done for every layer of every
    NLTE species setup. Not worth parallelising (``parallel=True``/``prange``): a typical call here covers at
    most a few hundred layers times a few dozen transitions, and thread-launch overhead would dwarf the actual
    work (a handful of comparisons and a log/exp per element).

    Parameters
    ----------
    temp_profile : ndarray, shape (n_layers,)
    temp_grid : ndarray, shape (n_temp_grid,) - shared by every row of rate_matrix.
    rate_matrix : ndarray, shape (n_transitions, n_temp_grid)

    Returns
    -------
    ndarray, shape (n_layers, n_transitions)
    """
    n_layers = temp_profile.shape[0]
    n_transitions = rate_matrix.shape[0]
    out = np.empty((n_layers, n_transitions), dtype=np.float64)
    for i in range(n_layers):
        for j in range(n_transitions):
            out[i, j] = _interp_rate(temp_profile[i], temp_grid, rate_matrix[j])
    return out


def _gas_kinetic_rate(
        collider: str, temperature: float, target_mass: float = 17.007, cross_section: float | None = None,
) -> float:
    """
    Ceiling for a level's total loss to ``collider`` [cm^3/s]: a collision cross section [A^2] (``cross_section``;
    by default 30 A^2, or the collider's _COLLIDER_CROSS_SECTION) times the mean relative speed of the collider and the
    molecule of mass ``target_mass`` [amu] (default OH).
    """
    mass = _COLLIDER_MASS[collider]
    mu = target_mass * mass / (target_mass + mass) * const_amu
    if cross_section is None:
        cross_section = _COLLIDER_CROSS_SECTION.get(collider, 30.0)
    return cross_section * 1e-16 * np.sqrt(8 * const_k_B * temperature / (np.pi * mu))


_COLLIDER_MASS = {"H": 1.008, "H2": 2.016, "He": 4.0026, "H2O": 18.015}
# H2O: dipole-dipole attraction makes OH + H2O collisions larger than hard-sphere; its measured OH(A) quenching cross
# section alone is 85 A^2 at 300 K (Tamura et al. 1998).
_COLLIDER_CROSS_SECTION = {"H2O": 100.0}
