r"""
H2O vibrational energy transfer in collisions with H2, He, H and H2O.

This docstring records every choice behind these rates: data sources, extrapolation, the assignment of final levels,
the gas-kinetic cap and what is left out. Update it whenever this module changes (see ``colchem.py``).

.. rubric:: Notation

- :math:`(v_1 v_2 v_3)`: H2O normal-mode quanta: :math:`v_1` symmetric stretch, :math:`v_2` bend, :math:`v_3`
  asymmetric stretch. Band origins :math:`\nu_1 = 3657.05`, :math:`\nu_2 = 1594.75` and
  :math:`\nu_3 = 3755.93\ \mathrm{cm^{-1}}` (POKAZATEL; ``_H2O_BAND_ORIGINS``). A *level* is every state with the
  same :math:`(v_1 v_2 v_3)`. The *stretches* are (100) and (001).
- :math:`U`, :math:`L`: upper and lower level of a transition; :math:`u`, :math:`l`: rovibrational states in them.
- :math:`M`: the collider (H2, He, H or H2O), with number density :math:`n_M` [:math:`\mathrm{cm^{-3}}`].
- :math:`T`: layer temperature [K]; :math:`k_B`: Boltzmann constant.
- :math:`k_M(U \to L)`: rate coefficient [:math:`\mathrm{cm^3\,s^{-1}}`] of :math:`U \to L` in collisions with
  :math:`M`. :math:`k_M(U \to L)\, n_M` is the rate per molecule in :math:`U` [:math:`\mathrm{s^{-1}}`]. Rate
  coefficients below are in :math:`\mathrm{cm^3\,s^{-1}}`, energies in :math:`\mathrm{cm^{-1}}`.
- :math:`k_1`, :math:`k_2`, :math:`k_3`: H2O self-relaxation *total removal* rate coefficients of (010), (020) and the
  *stretch reservoir*, (100) and (001) treated as one equilibrated pair (Zittel & Masturzo's notation).
- :math:`k_{\mathrm{gk},M}`: the gas-kinetic ceiling for collider :math:`M` (see "Gas-kinetic cap").

.. rubric:: How the rates enter the statistical equilibrium

Only downward channels are listed: :math:`U` lies above :math:`L` in harmonic energy
:math:`v_1 \nu_1 + v_2 \nu_2 + v_3 \nu_3`. ``NLTEProcessor._build_col_chem_cache`` adds each reverse rate by detailed
balance between the two aggregates,

.. math::

    k_M(L \to U) = k_M(U \to L)\, \frac{n^*_U}{n^*_L},

with :math:`n^*_U` the LTE population of aggregate :math:`U` (the Boltzmann sum over its states), so collisions alone
return LTE. The H2O processor aggregates on :math:`(v_1 v_2 v_3)` (``agg_col_nums=[8, 9, 10]``). A channel is skipped
when its keys match no aggregate, when either aggregate lies above the processor's aggregation cutoff, or when the
collider is not in the chemical profile.

.. rubric:: Rate model

All four colliders share one scheme (``get_rates``):

.. math::

    k_M(U \to L) = \min\left(1, \frac{k_{\mathrm{gk},M}}{\sum_{L'} m_{UL'}\, b_M(B_{UL'})}\right) m_{UL}\, b_M(B_{UL}).

- :math:`B_{UL}`: the *base channel* the transition is built on. The ten base channels connect the five levels below
  5000 :math:`\mathrm{cm^{-1}}`, (000), (010), (020), (100) and (001) (``_H2O_TABLE_KEYS``). For those levels
  :math:`B_{UL}` is :math:`U \to L` itself; above them it is a fundamental's channel (see "Higher levels").
- :math:`m_{UL}`: the number of quanta that can make the transition (1 for the five base levels).
- :math:`b_M(B)`: the base rate of channel :math:`B` for collider :math:`M`: Faure & Josselin's rate for H2, the same
  times a collider factor :math:`c_M` for He and H, and the self-relaxation scheme for H2O.
- The first factor caps the level's total loss to :math:`M`. The sum runs over every channel :math:`U \to L'` out of
  :math:`U`, including the stretch exchange.

.. rubric:: H2: Faure & Josselin (2008)

Faure & Josselin (2008, CDS J/A+A/492/257) give state-to-state rate coefficients for ortho- and para-H2O + H2 among
the rovibrational states of the five base levels, 200-5000 K, largely from quasi-classical trajectories.
``colrate_pipeline/faure_josselin_h2o.py`` reduces them to level-to-level rates by Boltzmann-averaging over the initial
states:

.. math::

    k(U \to L) = \sum_{u \in U} w_u \sum_{l \in L} k(u \to l), \qquad
    w_u = \frac{g_u\, e^{-E_u / k_B T}}{\sum_{u' \in U} g_{u'}\, e^{-E_{u'} / k_B T}},

with :math:`g_u` and :math:`E_u` the degeneracy and energy of state :math:`u`. The files list each state pair once,
downward in total energy. A pair with :math:`u` below :math:`l` (vibrationally down, rotationally up) gets its
:math:`u \to l` rate from detailed balance.

- ``_H2O_H2_ORTHO_RATES`` and ``_H2O_H2_PARA_RATES`` hold the result at 11 temperatures (5 significant figures;
  ``tests/check_h2o_rates.py`` compares them with the CDS files). They are interpolated log-linearly in :math:`T`
  (``base._interp_rate``) and held at the end values outside 200-5000 K.
- Final levels are as computed. A stretch fundamental relaxes 80% to (020) and 10% each to (010) and (000) at every
  tabulated temperature. Its (001) :math:`\to` (100) exchange is 44 times faster than its relaxation at 200 K and 3
  times faster at 5000 K.
- Accuracy: about 20% at 300 K, where two experiments and close-coupling theory agree; a factor ~3 above 1000 K
  (trajectories only). At 295 K the rates match Zittel & Masturzo (1991) within 10%: 1.41e-12 against 1.3e-12 for
  (010), 3.1e-12 against 2.9e-12 for (020), 8.7e-13 against 8.0e-13 for the stretches.

.. rubric:: He: scaled from H2

There are no usable tables. Every H2 base rate is multiplied by

.. math::

    c_{\mathrm{He}}(T) = \frac{k^{\mathrm{He}}_{\mathrm{bend}}(T)}{b_{\mathrm{H2}}((010) \to (000))}, \qquad
    k^{\mathrm{He}}_{\mathrm{bend}}(T) = 5 \times 10^{-13} \left(\frac{T}{300\ \mathrm{K}}\right)

(``_H2O_HE_BEND_300``), except the (001) :math:`\to` (100) exchange, which keeps H2's rate.

- :math:`k^{\mathrm{He}}_{\mathrm{bend}}` is 5.0e-13 at 300 K, 1.7e-12 at 1000 K, 4.2e-12 at 2500 K and 5.8e-12 at
  3500 K: inside Kung & Center's (1975) shock-tube range at 2500 K (1.5e-12 to 6e-12, depending on how their
  relaxation times are converted).
- The resulting He stretch relaxation (3.0e-13 at 295 K, 9.6e-13 at 924 K) lies 20-30% inside Zittel & Masturzo's
  (1989) He upper bounds (< 4e-13 at 295 K to < 1.2e-12 at 924 K, their Table I). They report no He bend rates.
- The stretch channels take the same factor: Kung & Center saw equal relaxation times in the bend
  (:math:`6.3\ \mu\mathrm{m}`) and stretch (:math:`2.7\ \mu\mathrm{m}`) emission, so He keeps H2's pattern of final
  levels.
- The exchange is near-resonant (:math:`\nu_3 - \nu_1 = 99\ \mathrm{cm^{-1}}`), so the bend's efficiency ratio does
  not apply to it. It only has to be fast compared with relaxation, which H2's value is.
- Uncertain by a factor 2-3.

.. rubric:: H: equal to H2

:math:`c_{\mathrm{H}} = 1` (``_H2O_H_OVER_H2``) for every channel. There are no thermal measurements for the low
levels. Close-coupling theory finds H much more efficient than He for the bend (Cabrera-Gonzalez et al. 2022), so H is
put with H2 rather than He. This is a guess: sensitivity runs at 0.3 and 3 bracket it.

.. rubric:: H2O self-relaxation

*Totals.* Zittel & Masturzo (1989), Tables I-III, laser-induced fluorescence (``_H2O_SELF_*``):

.. list-table:: Self-relaxation total removal rate coefficients [1e-11 cm^3 s^-1]
   :header-rows: 1

   * - Level
     - Symbol
     - :math:`T` [K]
     - Rate coefficient
   * - stretch reservoir
     - :math:`k_3`
     - 295, 410, 518, 648, 772, 922, 1020
     - 3.0, 2.4, 2.4, 2.4, 2.5, 2.4, 2.1
   * - (020)
     - :math:`k_2`
     - 295, 514, 730, 947
     - 12.1, 10.9, 10.9, 11.6
   * - (010)
     - :math:`k_1`
     - 295, 514, 730, 947
     - 5.4, 5.0, 5.4, 6.0

- Each is a *total removal* rate coefficient: the sum over every channel out of the level, not a rate to (000).
  Zittel & Masturzo's kinetic model (after Finzi et al. 1977) relaxes stretch :math:`\to` (020) :math:`\to` (010)
  :math:`\to` (000), and :math:`k_3` is the Boltzmann-weighted mean removal of (100) and (001), assumed equilibrated.
- Huestis (2006) re-fits (010) only, 5.1e-11 at 300 K (within 6% of Zittel & Masturzo), and states that the
  (010)/(000) yield of (020) relaxation is not established. It adds no rate here.
- Interpolated log-linearly in :math:`T` and held at the end values outside the measured range, so above ~1000 K the
  self rates are the ~1000 K values.

*Final levels.* Braunstein & Conforti (2015) ran quasi-classical trajectories for (001), (100), (020) and (010) at
295-2500 K on their own potential surface. Their totals are about twice Zittel & Masturzo's, so only their fractions
are used, applied to the measured totals (``_h2o_self_base``):

.. list-table:: Self-relaxation final levels (fractions of the level's total)
   :header-rows: 1

   * - Channel
     - Fraction
     - Source
   * - stretch :math:`\to` (020)
     - :math:`f_{32}(T)` = 0.31, 0.39, 0.58, 0.74 at 295, 500, 1000, 2500 K; linear in :math:`\ln T`, held outside
     - their Table 3 (:math:`k_{32}/k_3`)
   * - stretch :math:`\to` (010)
     - :math:`1 - f_{32} - 0.02`: 0.67 at 295 K, 0.24 at 2500 K
     - remainder
   * - stretch :math:`\to` (000)
     - 0.02
     - their Table 2
   * - (020) :math:`\to` (010)
     - 0.99
     - their Table 2
   * - (020) :math:`\to` (000)
     - 0.01
     - their Table 2
   * - (010) :math:`\to` (000)
     - 1
     - only channel

- Assignment of final levels: Braunstein & Conforti's Table 2 gives the joint final levels of both molecules. Only the
  net change in populations matters, so an outcome where the excited molecule ends in (000) and its partner in
  :math:`L` counts as :math:`U \to L` (the quantum was handed over), and an outcome that leaves or swaps the initial
  levels counts as nothing. When both end excited, e.g. (020) + (000) :math:`\to` (010) + (010), the excited molecule
  gets its own final level and the partner's excitation is dropped (see "Not included"). At 295 K this gives (001):
  0.24 to (020), 0.74 to (010), 0.02 to (000); (100): 0.33, 0.66, 0.01. At 2500 K: (001) 0.79, 0.18, 0.03; (100)
  0.80, 0.20, 0.00.
- Both stretches use Table 3's :math:`k_{32}/k_3`, their Boltzmann-weighted reservoir value. This matches the measured
  :math:`k_3`, which cannot separate the two (Braunstein & Conforti find (100) relaxing faster than (001)).
- Consistency: Braunstein & Conforti's ratios agree with Zittel & Masturzo's bounds (:math:`k_{32}/k_3 \ge`
  0.32-0.35 at 295-947 K) except at 295 K, where 0.31 is slightly low. Zittel & Masturzo's tighter 295 K bound,
  :math:`k_{32}/k_3 > 0.42`, assumes stretch :math:`\to` (000) is no faster than stretch :math:`\to` (020);
  Braunstein & Conforti satisfy that assumption and still give 0.31. The room-temperature fraction to (020) is
  therefore uncertain between about 0.3 and 0.6. Braunstein & Conforti are used because they are the only source of
  its temperature dependence. Both papers make single-quantum (020) :math:`\to` (010) the main (020) channel.

*Stretch exchange.* (001) :math:`\to` (100) is the gas-kinetic rate for a hard-sphere diameter of 2.8 angstrom (Zittel
& Masturzo's H2O value), :math:`\sigma = \pi (2.8\ \mathrm{\mathring{A}})^2 = 24.6\ \mathrm{\mathring{A}^2}`
(``_H2O_SELF_EXCHANGE_CROSS_SECTION``): 2.05e-10 at 295 K, rising as :math:`T^{1/2}` to 5.97e-10 at 2500 K.

- It is a measured lower bound. Finzi et al. (1977) concluded that the stretches equilibrate at least at the
  gas-kinetic rate and much faster than they relax; Zittel & Masturzo saw nothing contrary at any temperature
  (equilibration more than 10 times faster than :math:`k_3`).
- Braunstein & Conforti's trajectories give 1.1e-9 at 295 K and 3.2e-9 at 2500 K (joint probability 0.65 times
  :math:`\pi b_{\max}^2 \langle v_{\mathrm{rel}} \rangle`), about 5 times the bound and above the 100
  :math:`\mathrm{\mathring{A}^2}` ceiling, so the bound is used.
- The bound is 7 times the stretch relaxation :math:`k_3` at 295 K and 28 times at 2500 K, so (100) and (001) stay
  close to equilibrium wherever H2O collisions dominate radiative decay.

.. rubric:: Higher levels: harmonic ladder

Every level :math:`(v_1 v_2 v_3)` above the five base levels with harmonic energy up to 50,000
:math:`\mathrm{cm^{-1}}` (``_H2O_LADDER_MAX_ENERGY``) gets these downward channels for every collider
(``_h2o_ladder``):

.. list-table:: Ladder channels
   :header-rows: 1

   * - Channel
     - Final level :math:`L`
     - :math:`m_{UL}`
     - Base channel :math:`B_{UL}`
   * - bend
     - :math:`(v_1, v_2 - 1, v_3)`
     - :math:`v_2`
     - (010) :math:`\to` (000)
   * - symmetric stretch
     - :math:`(v_1 - 1, v_2 + n, v_3)`, :math:`n` = 2, 1, 0
     - :math:`v_1`
     - (100) :math:`\to` :math:`(0 n 0)`
   * - asymmetric stretch
     - :math:`(v_1, v_2 + n, v_3 - 1)`, :math:`n` = 2, 1, 0
     - :math:`v_3`
     - (001) :math:`\to` :math:`(0 n 0)`
   * - exchange
     - :math:`(v_1 + 1, v_2, v_3 - 1)`
     - 1
     - (001) :math:`\to` (100)

- Harmonic scaling, :math:`k(v \to v - 1) = v\, k(1 \to 0)` in each mode, as in Earth-atmosphere H2O models
  (Feofilov et al. 2009, Table 1). Evidence: the (020)/(010) ratio is 2.2 for H2 (Zittel & Masturzo 1991) and for
  self-relaxation (:math:`k_2/k_1` at 295 K), close to the harmonic 2; Ar stretch :math:`\to` bend scales as
  :math:`v_1 + v_3` within 20% for up to four stretch quanta (Butkovskaya & Setser 2022).
- Final levels: one mode loses one quantum per collision. A bend quantum goes to translation and rotation; a stretch
  quantum goes to :math:`n` bend quanta and the rest to translation and rotation, split over :math:`n` as the
  collider's stretch fundamental is (80% to :math:`n = 2` for H2, He and H; :math:`f_{32}(T)` for H2O). Larger jumps,
  such as :math:`\Delta v_2 = -2` from (030), are left out; (020) :math:`\to` (000) is a base channel and is not
  scaled.
- The exchange is not scaled (a harmonic estimate would be :math:`v_3 (v_1 + 1)` times the fundamental's). It only has
  to be fast compared with relaxation, and a rate growing with the quanta would reach the cap first and, since the
  cap scales all of a level's channels together, slow the relaxation.
- POKAZATEL labels :math:`(v_1 v_2 v_3)` only up to 20,112 :math:`\mathrm{cm^{-1}}` (134 levels, all covered;
  ``tests/check_h2o_rates.py``). Unlabelled states cannot be aggregated by quanta, so keys above that find no
  aggregate and are skipped. The 50,000 :math:`\mathrm{cm^{-1}}` limit is there for a fuller labelling.
- Uncertain by a factor ~10 above polyad :math:`2 v_1 + v_2 + 2 v_3 = 5` (no data).
- Cost: about 32,000 rates per layer, about 4 s per run in ``_build_col_chem_cache``.

.. rubric:: Gas-kinetic cap

:math:`k_{\mathrm{gk},M} = \sigma_M \langle v_{\mathrm{rel}} \rangle`, with mean relative speed
:math:`\langle v_{\mathrm{rel}} \rangle = (8 k_B T / \pi \mu)^{1/2}`, :math:`\mu` the H2O-:math:`M` reduced mass
(``base._gas_kinetic_rate``, H2O mass 18.011 amu) and cross section :math:`\sigma_M` = 30
:math:`\mathrm{\mathring{A}^2}` for H2, He and H, 100 :math:`\mathrm{\mathring{A}^2}` for H2O
(``base._COLLIDER_CROSS_SECTION``: dipole-dipole attraction; Braunstein & Conforti find impact parameters up to
8 angstrom contribute). It bounds each level's total loss to each collider, scaling all the level's channels by the
same factor so that the branching is kept. It is there for the ladder, whose linear growth in the quanta would
otherwise pass the collision rate.

.. list-table:: Where the cap binds: number of capped levels, and the lowest one (harmonic energy)
   :header-rows: 1

   * - :math:`T` [K]
     - H2
     - He
     - H
     - H2O
   * - 300
     - none
     - none
     - none
     - 477, from (0 12 1) at 22,893
   * - 1000
     - none
     - none
     - none
     - 84, from (0 19 1) at 34,056
   * - 2000
     - none
     - none
     - none
     - 2, from (0 27 1) at 46,814
   * - 3500
     - 4, from (0 27 1) at 46,814
     - none
     - none
     - none

It never binds for a POKAZATEL-labelled level, so it does not affect current models. The self rates bind most at low
:math:`T`, where they are largest against a ceiling that grows as :math:`T^{1/2}`.

.. rubric:: Ortho and para

``CollisionalRatesDatabase.H2O_RESOLVE_ORTHO_PARA`` (default False) sets the key shape, which must match the
processor's aggregation columns or no rate finds an aggregate.

- False: keys :math:`(v_1, v_2, v_3)`. The H2 tables are combined as :math:`(3 k_{\mathrm{ortho}} +
  k_{\mathrm{para}})/4` (nuclear-spin weights) before interpolation.
- True: keys :math:`(v_1, v_2, v_3, \mathrm{o/p})`. Each isomer gets its own H2 rates; the other colliders, with no
  isomer-resolved data, get the same rates on both.
- Collisions never convert ortho to para.

.. rubric:: Not included

- Partner co-excitation in self-collisions. In Braunstein & Conforti's trajectories the partner also ends excited in
  about 10% of stretch relaxations at 295 K and 11-15% of (020) relaxations, e.g. (020) + (000) :math:`\to` (010) +
  (010). Only the excited molecule's step is kept. The partner's step is an upward collision (000) :math:`\to` (010);
  its effect would come through detailed balance with the reverse bend pooling, (010) + (010) :math:`\to` (020) +
  (000). Both rates depend on the product of two H2O level populations, which the linear system :math:`Y n = R`
  (rate matrix :math:`Y`, level populations :math:`n`, source vector :math:`R`, fixed collider densities) cannot
  hold.
- Braunstein & Conforti's high-temperature excitation channels. At 2500 K they find collisions that excite rather
  than relax (mostly the partner going to (010)), some larger than the relaxation, though they note detailed balance
  suggests they should be smaller. Here upward rates come only from detailed balance of the downward channels.
- H2O + OH with H2O changing state: no data found (``notes/h2o_collision_data.md``).
- Reactions, e.g. H + H2O :math:`\to` OH + H2: the chemistry is fixed.

.. rubric:: Change log

- 2026-10-12: H2 tables re-derived as Boltzmann averages, downward only. Before, they were unweighted sums over all
  state pairs (100-120 times too fast, above gas-kinetic at high :math:`T`) and carried vibrationally upward pairs as
  extra keys, so earlier H2O NLTE results are not comparable.
- 2026-10-12: He and H scaled from H2. This replaced, for He, half of Zittel & Masturzo's (1989) values, whose bend
  entries were their Ar data (held above 947 K, 4 times below Kung & Center at 2500 K). For H it replaced a 1e-12
  placeholder and Barnes et al.'s (1999) overtone removal rates, which include reaction (0.34 of the removal of the
  local-mode ``|04>-`` level, Barnes et al. 2001) and lay above 10,000 :math:`\mathrm{cm^{-1}}`.
- 2026-10-12: harmonic ladder to 50,000 :math:`\mathrm{cm^{-1}}` with the per-level cap.
- 2026-10-12: self-relaxation as Zittel & Masturzo's totals with Braunstein & Conforti's final levels and the
  gas-kinetic exchange. This replaced the measured totals (Zittel & Masturzo; Huestis) used as rates straight to
  (000), with a conservative ladder above them.
- 2026-10-12: moved from ``colchem.py`` to ``colchem/h2o.py``.

.. rubric:: References

- Braunstein, M. & Conforti, P. F. 2015, J. Phys. Chem. A 119, 3311, doi:10.1021/jp5115527.
- Barnes, P. W., Sharkey, P., Sims, I. R. & Smith, I. W. M. 1999, Faraday Discuss. 113, 167, doi:10.1039/A902348H.
- Barnes, P. W., Sims, I. R., Smith, I. W. M., Lendvay, G. & Schatz, G. C. 2001, J. Chem. Phys. 115, 4586,
  doi:10.1063/1.1389304.
- Butkovskaya, N. I. & Setser, D. W. 2022, J. Chem. Phys. 157, 224305, doi:10.1063/5.0130747.
- Cabrera-Gonzalez, L. D. et al. 2022, MNRAS 514, 4426, doi:10.1093/mnras/stac1643.
- Faure, A. & Josselin, E. 2008, A&A 492, 257, doi:10.1051/0004-6361:200810717.
- Feofilov, A. G. et al. 2009, Atmos. Chem. Phys. 9, 8139, doi:10.5194/acp-9-8139-2009.
- Finzi, J., Hovis, F. E., Panfilov, V. N., Hess, P. & Moore, C. B. 1977, J. Chem. Phys. 67, 4053.
- Huestis, D. L. 2006, J. Phys. Chem. A 110, 6638, doi:10.1021/jp054889n.
- Kung, R. T. V. & Center, R. E. 1975, J. Chem. Phys. 62, 2187.
- Polyansky, O. L. et al. 2018, MNRAS 480, 2597 (POKAZATEL).
- Zittel, P. F. & Masturzo, D. E. 1989, J. Chem. Phys. 90, 977, doi:10.1063/1.456122.
- Zittel, P. F. & Masturzo, D. E. 1991, J. Chem. Phys. 95, 8005.
"""
import typing as t

import numpy as np
import numpy.typing as npt

from .base import RateTransition, _gas_kinetic_rate, _interp_rate_batch


# H2O collisional rate tables, module-level because get_rates runs once per atmospheric layer during NLTE setup. The
# module docstring gives the sources and choices.
# --------------------------------------------------------------------------------------------------------------

# A. Faure & E. Josselin (2008), A&A 492, 257, doi:10.1051/0004-6361:200810717 (CDS J/A+A/492/257): vibrational-level
# rates k(U -> L) [cm^3/s] for the five levels below 5000 cm^-1, Boltzmann-averaged over the initial rotational states
# of each state's rate summed over the final states (colrate_pipeline/faure_josselin_h2o.py regenerates them). Downward
# (U above L) only; upward rates come from detailed balance. Uncertain by ~20% at 300 K, where two experiments and
# close-coupling theory agree, and by a factor ~3 above 1000 K (quasi-classical trajectories only).
# Temperature grid shared by both isomers (K).
_H2O_H2_T_GRID = np.array([200.0, 400.0, 800.0, 1200.0, 1600.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0, 5000.0])

# --- ortho-H2O + H2 ---
_H2O_H2_ORTHO_RATES: t.Dict[t.Tuple[int, int, int, int, int, int], npt.NDArray[np.float64]] = {
    # (v1u, v2u, v3u, v1l, v2l, v3l)
    (0, 1, 0, 0, 0, 0): np.array([1.2937e-12, 1.5485e-12, 3.5631e-12, 7.1160e-12, 1.2315e-11, 1.9221e-11,
                                  3.0486e-11, 4.4477e-11, 6.1671e-11, 8.0816e-11, 1.2822e-10]),
    (0, 2, 0, 0, 0, 0): np.array([5.9411e-13, 7.0553e-13, 1.6536e-12, 3.3466e-12, 5.8425e-12, 9.1684e-12,
                                  1.4592e-11, 2.1381e-11, 2.9639e-11, 3.9140e-11, 6.2044e-11]),
    (0, 2, 0, 0, 1, 0): np.array([2.2949e-12, 2.7418e-12, 6.3446e-12, 1.2840e-11, 2.2463e-11, 3.5171e-11,
                                  5.6063e-11, 8.2079e-11, 1.1352e-10, 1.4987e-10, 2.3910e-10]),
    (1, 0, 0, 0, 0, 0): np.array([7.9859e-14, 9.5150e-14, 2.2365e-13, 4.5201e-13, 7.9669e-13, 1.2555e-12,
                                  2.0054e-12, 2.9483e-12, 4.0708e-12, 5.3878e-12, 8.5574e-12]),
    (1, 0, 0, 0, 1, 0): np.array([7.9859e-14, 9.5150e-14, 2.2365e-13, 4.5201e-13, 7.9665e-13, 1.2554e-12,
                                  2.0051e-12, 2.9478e-12, 4.0699e-12, 5.3866e-12, 8.5550e-12]),
    (1, 0, 0, 0, 2, 0): np.array([6.3430e-13, 7.5339e-13, 1.7603e-12, 3.5669e-12, 6.2599e-12, 9.8188e-12,
                                  1.5640e-11, 2.3023e-11, 3.1775e-11, 4.1869e-11, 6.6527e-11]),
    (0, 0, 1, 0, 0, 0): np.array([7.9549e-14, 9.5775e-14, 2.2978e-13, 4.7082e-13, 8.4548e-13, 1.3405e-12,
                                  2.1502e-12, 3.1568e-12, 4.3768e-12, 5.7766e-12, 9.2458e-12]),
    (0, 0, 1, 0, 1, 0): np.array([7.9549e-14, 9.5775e-14, 2.2978e-13, 4.7081e-13, 8.4524e-13, 1.3399e-12,
                                  2.1488e-12, 3.1540e-12, 4.3723e-12, 5.7698e-12, 9.2325e-12]),
    (0, 0, 1, 0, 2, 0): np.array([6.3523e-13, 7.6745e-13, 1.8235e-12, 3.7148e-12, 6.5599e-12, 1.0310e-11,
                                  1.6399e-11, 2.3934e-11, 3.3171e-11, 4.3934e-11, 6.9629e-11]),
    (0, 0, 1, 1, 0, 0): np.array([3.4900e-11, 5.3408e-11, 9.1908e-11, 1.2607e-10, 1.5629e-10, 1.8334e-10,
                                  2.0412e-10, 2.2346e-10, 2.4134e-10, 2.5780e-10, 2.8795e-10]),
}

# --- para-H2O + H2 ---
_H2O_H2_PARA_RATES: t.Dict[t.Tuple[int, int, int, int, int, int], npt.NDArray[np.float64]] = {
    # (v1u, v2u, v3u, v1l, v2l, v3l)
    (0, 1, 0, 0, 0, 0): np.array([1.2957e-12, 1.5437e-12, 3.5574e-12, 7.0788e-12, 1.2250e-11, 1.9199e-11,
                                  3.0473e-11, 4.4610e-11, 6.1434e-11, 8.1145e-11, 1.2851e-10]),
    (0, 2, 0, 0, 0, 0): np.array([5.9628e-13, 7.1168e-13, 1.6583e-12, 3.3573e-12, 5.8346e-12, 9.2798e-12,
                                  1.4737e-11, 2.1600e-11, 2.9903e-11, 3.9321e-11, 6.3084e-11]),
    (0, 2, 0, 0, 1, 0): np.array([2.2819e-12, 2.7363e-12, 6.3781e-12, 1.2815e-11, 2.2423e-11, 3.5556e-11,
                                  5.6309e-11, 8.2943e-11, 1.1457e-10, 1.5015e-10, 2.4200e-10]),
    (1, 0, 0, 0, 0, 0): np.array([7.9551e-14, 9.5533e-14, 2.2354e-13, 4.5673e-13, 7.9354e-13, 1.2741e-12,
                                  2.0049e-12, 2.9891e-12, 4.1308e-12, 5.4160e-12, 8.7179e-12]),
    (1, 0, 0, 0, 1, 0): np.array([7.9551e-14, 9.5533e-14, 2.2354e-13, 4.5673e-13, 7.9349e-13, 1.2740e-12,
                                  2.0047e-12, 2.9886e-12, 4.1300e-12, 5.4149e-12, 8.7156e-12]),
    (1, 0, 0, 0, 2, 0): np.array([6.3704e-13, 7.5607e-13, 1.7694e-12, 3.5934e-12, 6.2492e-12, 9.9814e-12,
                                  1.5687e-11, 2.3342e-11, 3.2052e-11, 4.2220e-11, 6.7968e-11]),
    (0, 0, 1, 0, 0, 0): np.array([7.9607e-14, 9.5973e-14, 2.3060e-13, 4.7244e-13, 8.4840e-13, 1.3459e-12,
                                  2.1474e-12, 3.1704e-12, 4.4039e-12, 5.8541e-12, 9.3156e-12]),
    (0, 0, 1, 0, 1, 0): np.array([7.9607e-14, 9.5973e-14, 2.3059e-13, 4.7244e-13, 8.4815e-13, 1.3453e-12,
                                  2.1457e-12, 3.1677e-12, 4.3994e-12, 5.8469e-12, 9.3015e-12]),
    (0, 0, 1, 0, 2, 0): np.array([6.3735e-13, 7.6572e-13, 1.8135e-12, 3.7138e-12, 6.5698e-12, 1.0346e-11,
                                  1.6488e-11, 2.4065e-11, 3.3392e-11, 4.4130e-11, 7.0095e-11]),
    (0, 0, 1, 1, 0, 0): np.array([3.6709e-11, 5.5851e-11, 9.4401e-11, 1.2941e-10, 1.5974e-10, 1.8723e-10,
                                  2.0909e-10, 2.2874e-10, 2.4701e-10, 2.6407e-10, 2.9499e-10]),
}

# H2O self-relaxation (module docstring, "H2O self-relaxation"). Total removal rate constants [cm^3/s] of the stretch
# reservoir (k3), (020) (k2) and (010) (k1), Zittel & Masturzo (1989) Tables I-III; held at the end values outside the
# measured range (_interp_rate clamps).
_H2O_SELF_STRETCH_T_GRID = np.array([295.0, 410.0, 518.0, 648.0, 772.0, 922.0, 1020.0])
_H2O_SELF_STRETCH = np.array([3.0e-11, 2.4e-11, 2.4e-11, 2.4e-11, 2.5e-11, 2.4e-11, 2.1e-11])
_H2O_SELF_BEND_T_GRID = np.array([295.0, 514.0, 730.0, 947.0])
_H2O_SELF_020 = np.array([12.1e-11, 10.9e-11, 10.9e-11, 11.6e-11])
_H2O_SELF_010 = np.array([5.4e-11, 5.0e-11, 5.4e-11, 6.0e-11])
# Final-level split (Braunstein & Conforti 2015): stretch -> (020) fraction (their Table 3, k32/k3), interpolated in
# ln T and held outside 295-2500 K; stretch -> (000) and (020) -> (000) fractions from their Table 2.
_H2O_SELF_STRETCH_TO_020_T = np.array([295.0, 500.0, 1000.0, 2500.0])
_H2O_SELF_STRETCH_TO_020 = np.array([0.31, 0.39, 0.58, 0.74])
_H2O_SELF_STRETCH_TO_000 = 0.02
_H2O_SELF_020_TO_000 = 0.01
# nu3 -> nu1 exchange: the gas-kinetic rate for a hard-sphere diameter of 2.8 A (Zittel & Masturzo), the measured lower
# bound (Finzi et al. 1977).
_H2O_SELF_EXCHANGE_CROSS_SECTION = np.pi * 2.8 ** 2  # [A^2]

# He and H (no usable tables; the earlier Zittel & Masturzo upper bounds and the Barnes et al. 1999 removal rates,
# which include reaction, are replaced): every H2 channel times a collider factor (_h2o_collider_factor).
# He: k_bend = 5e-13 (T / 300 K) cm^3/s for (010) -> (000), inside Kung & Center's (1975) 1.5-6e-12 at 2500 K; the
# other He channels follow H2's pattern (Kung & Center see equal bend and stretch relaxation times), inside Zittel &
# Masturzo's He stretch bounds, except the nu1/nu3 exchange, kept at H2's value. Factor 2-3 uncertain.
_H2O_HE_BEND_300 = 5.0e-13
# H: taken equal to H2 (no thermal data; close-coupling theory finds H much more efficient than He for the bend,
# Cabrera-Gonzalez et al. 2022). A guess: worth sensitivity runs at x0.3 and x3.
_H2O_H_OVER_H2 = 1.0

# Higher H2O levels (_h2o_ladder): every (v1, v2, v3) with harmonic energy v1 nu1 + v2 nu2 + v3 nu3 up to
# _H2O_LADDER_MAX_ENERGY. POKAZATEL labels vibrations only up to 20,112 cm^-1 (134 levels); unlabelled states cannot be
# aggregated by (v1, v2, v3), so keys above that are skipped.
_H2O_BAND_ORIGINS = (3657.05, 1594.75, 3755.93)  # nu1, nu2, nu3 [cm^-1] (POKAZATEL)
_H2O_LADDER_MAX_ENERGY = 50000.0
_H2O_MASS = 18.011


def _h2o_ladder() -> t.List[t.Tuple[t.Tuple[int, int, int], t.Tuple[int, int, int], t.Tuple[int, ...], int]]:
    """
    Downward collisional channels of the H2O levels above the five tabulated ones, as (upper, lower, base key,
    multiplier): the rate is multiplier x the tabulated base rate (the H2 table key ``base key``). Harmonic scaling
    in the number of quanta (Zittel & Masturzo 1991: k(020)/k(010) = 2.2; Butkovskaya & Setser 2022: stretch -> bend
    scales as v1 + v3 for up to 4 quanta), as in Earth-atmosphere H2O models (Feofilov et al. 2009, Table 1):

    - bend: (v1, v2, v3) -> (v1, v2 - 1, v3) at v2 k[(010) -> (000)];
    - stretch: (v1, v2, v3) -> (v1 - 1, v2 + n, v3) at v1 k[(100) -> (0n0)], and (v1, v2 + n, v3 - 1) at
      v3 k[(001) -> (0n0)], n = 2, 1, 0 (split over n as the collider's stretch fundamental);
    - nu3 -> nu1 exchange: (v1, v2, v3) -> (v1 + 1, v2, v3 - 1) at k[(001) -> (100)], unscaled, keeping the two
      stretches close to equilibrium with each other.

    Uncertain by a factor ~10 above polyad 2 v1 + 2 v3 + v2 = 5 (no data).
    """
    nu1, nu2, nu3 = _H2O_BAND_ORIGINS
    levels = [(v1, v2, v3) for v1 in range(int(_H2O_LADDER_MAX_ENERGY // nu1) + 1)
              for v2 in range(int(_H2O_LADDER_MAX_ENERGY // nu2) + 1)
              for v3 in range(int(_H2O_LADDER_MAX_ENERGY // nu3) + 1)
              if v1 * nu1 + v2 * nu2 + v3 * nu3 <= _H2O_LADDER_MAX_ENERGY
              and (v1, v2, v3) not in ((0, 0, 0), (0, 1, 0), (0, 2, 0), (1, 0, 0), (0, 0, 1))]
    ladder = []
    for v1, v2, v3 in levels:
        if v2:
            ladder.append(((v1, v2, v3), (v1, v2 - 1, v3), (0, 1, 0, 0, 0, 0), v2))
        for n in (2, 1, 0):
            if v1:
                ladder.append(((v1, v2, v3), (v1 - 1, v2 + n, v3), (1, 0, 0, 0, n, 0), v1))
            if v3:
                ladder.append(((v1, v2, v3), (v1, v2 + n, v3 - 1), (0, 0, 1, 0, n, 0), v3))
        if v3:
            ladder.append(((v1, v2, v3), (v1 + 1, v2, v3 - 1), (0, 0, 1, 1, 0, 0), 1))
    return ladder


# Every H2, He and H channel: the ten tabulated ones, then the ladder. Fixed, so built once: upper-level index, base
# (tabulated) channel index and multiplier of each channel.
_H2O_TABLE_KEYS = tuple(_H2O_H2_ORTHO_RATES)
_H2O_CHANNELS = [(key[:3], key[3:], key, 1) for key in _H2O_TABLE_KEYS] + _h2o_ladder()
_H2O_UPPER_INDEX = {upper: n for n, upper in enumerate(dict.fromkeys(upper for upper, *_ in _H2O_CHANNELS))}
_H2O_CHANNEL_UPPER = np.array([_H2O_UPPER_INDEX[upper] for upper, *_ in _H2O_CHANNELS])
_H2O_CHANNEL_BASE = np.array([_H2O_TABLE_KEYS.index(base_key) for _, _, base_key, _ in _H2O_CHANNELS])
_H2O_CHANNEL_MULTIPLIER = np.array([multiplier for *_, multiplier in _H2O_CHANNELS], dtype=np.float64)


def _h2o_collider_factor(collider: str, base_key: t.Tuple[int, ...], k_bend_h2: float, layer_temp: float) -> float:
    """Rate of H2O + ``collider`` relative to H2O + H2 for the channel with H2 table key ``base_key``."""
    if collider == "H":
        return _H2O_H_OVER_H2
    if collider == "He" and base_key != (0, 0, 1, 1, 0, 0):
        return _H2O_HE_BEND_300 * layer_temp / 300.0 / k_bend_h2
    return 1.0


def _h2o_self_base(layer_temp: float) -> npt.NDArray[np.float64]:
    """
    H2O self-relaxation rate [cm^3/s] of each tabulated channel (_H2O_TABLE_KEYS order) at ``layer_temp``: Zittel &
    Masturzo's total for the channel's upper level times Braunstein & Conforti's fraction for its lower level, and the
    gas-kinetic lower bound for nu3 -> nu1. The ladder builds the higher levels on these, as for H2.
    """
    temp = np.array([layer_temp])
    k3 = _interp_rate_batch(temp_profile=temp, temp_grid=_H2O_SELF_STRETCH_T_GRID,
                            rate_matrix=_H2O_SELF_STRETCH[None, :])[0, 0]
    k2, k1 = _interp_rate_batch(temp_profile=temp, temp_grid=_H2O_SELF_BEND_T_GRID,
                                rate_matrix=np.stack([_H2O_SELF_020, _H2O_SELF_010]))[0]
    to_020 = np.interp(np.log(layer_temp), np.log(_H2O_SELF_STRETCH_TO_020_T), _H2O_SELF_STRETCH_TO_020)
    stretch_split = {2: to_020, 1: 1.0 - to_020 - _H2O_SELF_STRETCH_TO_000, 0: _H2O_SELF_STRETCH_TO_000}
    base = {
        (0, 1, 0, 0, 0, 0): k1,
        (0, 2, 0, 0, 1, 0): (1.0 - _H2O_SELF_020_TO_000) * k2,
        (0, 2, 0, 0, 0, 0): _H2O_SELF_020_TO_000 * k2,
        (0, 0, 1, 1, 0, 0): _gas_kinetic_rate(collider="H2O", temperature=layer_temp, target_mass=_H2O_MASS,
                                              cross_section=_H2O_SELF_EXCHANGE_CROSS_SECTION),
        **{(*stretch, 0, n, 0): fraction * k3 for stretch in ((1, 0, 0), (0, 0, 1))
           for n, fraction in stretch_split.items()},
    }
    return np.array([base[key] for key in _H2O_TABLE_KEYS])


def get_rates(layer_temp: float, resolve_isomer: bool = False) -> t.List[RateTransition]:
    """
    H2O collisional rate coefficients [cm^3/s] with H2, He, H and H2O at ``layer_temp`` [K], downward channels only
    (module docstring, "Rate model"): each collider's base rates on the ten channels of the five lowest levels
    (Faure & Josselin's H2 rates, times _h2o_collider_factor for He and H; _h2o_self_base for H2O), scaled up the
    harmonic ladder (_h2o_ladder), with each level's total loss to each collider capped at the gas-kinetic rate
    (_gas_kinetic_rate).

    State keys depend on ``resolve_isomer`` (``CollisionalRatesDatabase.H2O_RESOLVE_ORTHO_PARA``): ``False`` gives
    (v1, v2, v3), with the ortho and para H2 rates combined 3:1 (nuclear-spin weights) before interpolation;
    ``True`` gives (v1, v2, v3, "o"/"p") with each isomer's own H2 rates and the other colliders' rates duplicated
    onto both.
    v1, v2 and v3 are the symmetric-stretch, bend and asymmetric-stretch quanta.
    """
    layer_temp_arr = np.array([layer_temp])

    def _make_keys(upper, lower, isomer_label):
        if resolve_isomer:
            return (*upper, isomer_label), (*lower, isomer_label)
        return upper, lower

    if resolve_isomer:
        h2_tables = (("o", _H2O_H2_ORTHO_RATES), ("p", _H2O_H2_PARA_RATES))
    else:
        h2_tables = ((None, {key: (3.0 * _H2O_H2_ORTHO_RATES[key] + _H2O_H2_PARA_RATES[key]) / 4.0
                             for key in _H2O_TABLE_KEYS}),)

    self_base = _h2o_self_base(layer_temp)
    rates = []
    for isomer_label, table in h2_tables:
        base = _interp_rate_batch(
            temp_profile=layer_temp_arr, temp_grid=_H2O_H2_T_GRID,
            rate_matrix=np.stack([table[key] for key in _H2O_TABLE_KEYS]),
        )[0]
        k_bend = base[_H2O_TABLE_KEYS.index((0, 1, 0, 0, 0, 0))]
        for collider in ("H2", "He", "H", "H2O"):
            if collider == "H2O":
                collider_base = self_base
            else:
                collider_base = base * np.array([
                    _h2o_collider_factor(collider=collider, base_key=key, k_bend_h2=k_bend, layer_temp=layer_temp)
                    for key in _H2O_TABLE_KEYS
                ])
            k = _H2O_CHANNEL_MULTIPLIER * collider_base[_H2O_CHANNEL_BASE]
            cap = _gas_kinetic_rate(collider=collider, temperature=layer_temp, target_mass=_H2O_MASS)
            k *= np.minimum(1.0, cap / np.bincount(_H2O_CHANNEL_UPPER, weights=k))[_H2O_CHANNEL_UPPER]
            for (upper, lower, _, _), rate in zip(_H2O_CHANNELS, k):
                upper_key, lower_key = _make_keys(upper, lower, isomer_label)
                rates.append(RateTransition(
                    upper_key=upper_key, lower_key=lower_key, rate=float(rate), species_depend=(collider,),
                ))

    return rates
