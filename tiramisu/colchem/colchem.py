"""
The collisional/chemical rate database: per-species metadata, the configuration flags, and get_rates, which hands each
species to its data module (``co.py``, ``h2o.py``, ``oh.py``). Shared conventions (how rates enter the statistical
equilibrium, interpolation, the gas-kinetic ceiling) are in ``base.py``.

.. rubric:: Adding a species

Write a module with its rate tables and a ``get_rates(layer_temp, ...)`` returning a list of ``RateTransition``
(downward channels only), add it to ``SPECIES_INFO``, and dispatch to it in ``get_rates``. Give the module the
documentation below.

.. rubric:: Documentation rule

Each species module opens with a docstring that records every choice behind its rates: sources (with references),
extrapolation, the assignment of final levels, capping and when it binds, what is left out, and a dated change log.
Update it, and ``base.py``'s for shared conventions, in the same change as the code. The docstrings are
reStructuredText using only docutils constructs (``rubric``, ``list-table``, ``math``; no Sphinx roles), so Sphinx
renders them and docutils converts them to LaTeX for notes or a paper, e.g.::

    python -c "import docutils.core, tiramisu.colchem.h2o as m; \\
        print(docutils.core.publish_parts(m.__doc__, writer_name='latex')['body'])"
"""
import typing as t

from . import co, h2o, oh
from .base import RateTransition, SpeciesRateInfo


class CollisionalRatesDatabase:
    """
    Database of collisional and chemical rates for various species.
    Completely independent of any specific molecule instance.

    Attributes
    ----------
    SPECIES_INFO : Dict[str, SpeciesRateInfo]
        Metadata for each species with configured rates.
        If a species is in this dict, it has rates configured.
    """
    SPECIES_INFO: t.Dict[str, SpeciesRateInfo] = {
        "OH": SpeciesRateInfo(
            temperature_dependent=True,  # OH(A) quenching (Tamura et al. 1998); the X-state rates are 300 K values.
            collision_partners={"O2", "O", "O3", "He", "H", "H2", "H2O"}
        ),
        "CO": SpeciesRateInfo(
            temperature_dependent=True,
            collision_partners={"H", "He", "H2", "H2O"}
        ),
        "H2O": SpeciesRateInfo(
            temperature_dependent=True,
            collision_partners={"H2", "He", "H", "H2O"}
        ),
    }

    # Whether H2O collisional rates/state keys distinguish ortho/para at all. H2O+H2 is the only collision
    # partner with isomer-resolved rate data (Faure & Josselin 2008); every other partner (H2O, He, H) below is
    # only ever measured/estimated as a single isomer-unresolved rate, applied identically to both ortho and
    # para as a stand-in. When False, state keys drop the isomer label entirely (``(v1, v2, v3)`` instead of
    # ``(v1, v2, v3, 'o'/'p')``), matching an H2O NLTEProcessor configured with an isomer-unresolved
    # ``agg_col_nums`` (3 columns instead of 4): the H2O+H2 rates are combined down via nuclear-spin
    # statistical-weight averaging (see h2o.get_rates) and every other partner's rate is emitted once instead
    # of duplicated per isomer. This must match whatever aggregation scheme the H2O NLTEProcessor is built
    # with - the lookup in NLTEProcessor._build_agg_state_lookup keys purely on the aggregation columns, so a
    # mismatched isomer-vs-no-isomer key shape here will silently fail to match any state. False: the H2O models
    # aggregate on (v1, v2, v3) only (agg_col_nums=[8, 9, 10]).
    H2O_RESOLVE_ORTHO_PARA: bool = False

    # Which OH rate set to use. "literature": measured or calculated rates only - X-state relaxation as before, plus
    # OH(A) electronic quenching (split over X v'' by the assumed product distributions, reactive part as
    # destruction) and the measured A v'=1->0 relaxation. "extended": the literature set plus estimates where it has
    # no rate (see oh._get_oh_extended_rates): X-state relaxation up to v=16 and A-state relaxation for every collider
    # and v'. It replaces no literature rate.
    OH_RATE_SET: str = "literature"
    # Where OH(X, v >= 3) + H relaxation goes; each level's total is unchanged. "single": all to v-1. "statistical":
    # oh._OH_H_STATISTICAL_FRACTION by the H2O-complex prior (oh._oh_h_prior), the rest to v-1. "flat": evenly over
    # all v'' < v, like O + OH (Caridade et al. 2013). v=1, 2 keep Atahan & Alexander's state-to-state rates.
    OH_H_BRANCHING: str = "single"

    @classmethod
    def is_temperature_dependent(cls, species: str) -> bool:
        """
        Check if rates require temperature interpolation.

        Parameters
        ----------
        species : str
            Chemical formula

        Returns
        -------
        bool
            True if rates vary with temperature

        Raises
        ------
        KeyError
            If species has no configured rates
        """
        if species in cls.SPECIES_INFO:
            return cls.SPECIES_INFO[species].temperature_dependent
        else:
            raise RuntimeWarning(f"Collisional rates not configured for {species}.")

    @staticmethod
    def get_rates(species: str, layer_temp: float | None = None) -> list[RateTransition]:
        """
        Get collisional rates for a species.

        Parameters
        ----------
        species : str
            Chemical formula (e.g., 'OH', 'CO')
        layer_temp : float, optional
            Temperature in K (required for temperature-dependent rates)

        Returns
        -------
        list of RateTransition
            List of rate transitions for the species

        Raises
        ------
        ValueError
            If temperature-dependent species called without layer_temp.
        """
        if species == "OH":
            if layer_temp is None:
                raise ValueError("OH rates require layer_temp parameter")
            return oh.get_rates(
                layer_temp=layer_temp, rate_set=CollisionalRatesDatabase.OH_RATE_SET,
                h_branching=CollisionalRatesDatabase.OH_H_BRANCHING,
            )
        elif species == "CO":
            if layer_temp is None:
                raise ValueError("CO rates require layer_temp parameter")
            return co.get_rates(layer_temp=layer_temp)
        elif species == "H2O":
            if layer_temp is None:
                raise ValueError("H2O rates require layer_temp parameter")
            return h2o.get_rates(
                layer_temp=layer_temp, resolve_isomer=CollisionalRatesDatabase.H2O_RESOLVE_ORTHO_PARA,
            )
        else:
            return []
