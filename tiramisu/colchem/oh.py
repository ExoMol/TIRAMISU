r"""
OH collisional and chemical rates: OH(X) vibrational relaxation by O2, O, He, H, H2 and H2O, formation by H + O3,
destruction by O, OH(A) electronic quenching and vibrational relaxation, and the estimates of the "extended" set.

This docstring records every choice behind these rates: data sources, extrapolation, the assignment of final levels,
the gas-kinetic cap and what is left out. Update it whenever this module changes (see ``colchem.py``).

.. rubric:: Notation

- X :math:`v` and A :math:`v'`: OH(:math:`X^2\Pi`, :math:`v`), :math:`v` = 0-16, and OH(:math:`A^2\Sigma^+`,
  :math:`v'`), :math:`v'` = 0-10, with keys ``("X2Pi", v)`` and ``("A2Sigma+", v')``. Each is an aggregate of all
  rotational states of that vibrational level.
- :math:`E_X(v)`, :math:`E_A(v')`: level energies [:math:`\mathrm{cm^{-1}}`], the lowest MYTHOS state of each level
  (``_OH_X_ENERGIES``, ``_OH_A_ENERGIES``; from ``16O-1H__MYTHOS.noB4.vfix.states``, :math:`v` relabelled by
  ``tests/fix_states_v_labels.py``). They set which product levels are open and the energy gaps of the gap laws.
- :math:`\Delta E`: the energy gap of a single-quantum step, :math:`E(v) - E(v - 1)`.
- :math:`M`: the collider, with number density :math:`n_M`; :math:`T`: layer temperature [K]; :math:`k_B`: Boltzmann
  constant.
- :math:`k_M(U \to L)`: rate coefficient [:math:`\mathrm{cm^3\,s^{-1}}`] of level :math:`U \to L` in collisions with
  :math:`M`. Rate coefficients below are in :math:`\mathrm{cm^3\,s^{-1}}`, energies in :math:`\mathrm{cm^{-1}}`.
- :math:`k_{\mathrm{gk},M}`: the gas-kinetic ceiling (see "Gas-kinetic cap").

.. rubric:: How the rates enter the statistical equilibrium

- Relaxation and quenching channels are downward only; ``NLTEProcessor._build_col_chem_cache`` adds the reverse by
  detailed balance between the two aggregates' LTE populations (see ``base.py``).
- Destruction (O + OH and reactive OH(A) quenching) is a ``RateTransition`` with ``upper_key == lower_key`` and a
  negative rate: a loss :math:`|k|\, n_M` per molecule on the diagonal of the rate matrix.
- Formation (H + O3) has ``upper_key == lower_key`` and a positive rate: :math:`k\, n_{\mathrm{H}}\, n_{\mathrm{O_3}}`
  per unit volume, divided by the OH number density and put on the right-hand side.
- A-state keys only apply when the OH processor aggregates A levels below its cutoff; otherwise they are skipped.
- Temperature: the A-state quenching and the H2O X-state rates depend on :math:`T`; every other X-state rate is a
  room-temperature value used at all :math:`T`.

.. rubric:: Rate sets and flags

- ``CollisionalRatesDatabase.OH_RATE_SET``. ``"literature"`` (default): measured or calculated rates only.
  ``"extended"``: the literature set plus the estimates of "Extended set" below, which replace no literature rate.
- ``CollisionalRatesDatabase.OH_H_BRANCHING``: where X :math:`v \ge 3` + H relaxation goes; each level's total is
  unchanged. ``"single"`` (default): all to :math:`v - 1`. ``"statistical"``: a fraction
  :math:`f_{\mathrm{stat}} = 0.84` spread by the phase-space prior of an H2O complex,
  :math:`P(v'') \propto (E_X(v) - E_X(v'') + 2 k_B T)^{3/2}` over :math:`v'' < v` (``_oh_h_prior``; :math:`2 k_B T`
  is the mean collision energy), the rest to :math:`v - 1`. :math:`f_{\mathrm{stat}}` is Atahan & Alexander's
  :math:`v = 2 \to 0` branching at 300 K (0.61) over the prior's (``_OH_H_STATISTICAL_FRACTION``). ``"flat"``: evenly
  over all :math:`v'' < v`, as O + OH does (Caridade et al. 2013). :math:`v` = 1, 2 keep Atahan & Alexander's
  state-to-state rates. The evidence favours multi-quantum relaxation (``notes/oh_collisional_rates.tex``); the
  default is still ``"single"``.

.. rubric:: Literature set: X-state relaxation, formation and destruction

.. list-table:: X-state rates (literature set)
   :header-rows: 1

   * - Collider
     - Channels
     - :math:`T`
     - Source and notes
   * - O2
     - :math:`v \to v - \Delta v`, all :math:`\Delta v`, :math:`v \le 9`
     - nominal
     - Adler-Golden (1997): :math:`k = C P_v^{\Delta v}`, :math:`C` = 4.4e-12, :math:`P_v` = 0.043-0.95 rising with
       :math:`v`.
   * - O
     - :math:`v' \to v''`, all :math:`v'' < v'`, :math:`v' \le 9`
     - 300 K
     - Caridade et al. (2013), Table 1, R5 (multi-quantum, roughly flat in :math:`v''`).
   * - O
     - destruction of :math:`v` = 0-10 (O + OH :math:`\to` O2 + H)
     - 300 K
     - Caridade et al. (2013), Table 1, R4, :math:`v` = 1-9; :math:`v` = 0 and 10 extrapolated (2.6e-11, 4.6e-11).
   * - H + O3
     - formation of :math:`v` = 0-10
     - 300 K
     - :math:`1.4 \times 10^{-10} e^{-470/T}` at :math:`T` = 300 K (DeMore et al. 1990, via Dodd et al. 1994), split
       by GRANADA's nascent distribution (Funke et al. 2012, Table 13; see below).
   * - He
     - :math:`v \to v - 1`, :math:`v` = 1-10
     - 298 K
     - Kohno et al. (2013).
   * - H
     - 1 :math:`\to` 0, 2 :math:`\to` 1, 2 :math:`\to` 0
     - 300 K
     - Atahan & Alexander (2006), quantum scattering (1.60e-10, 0.654e-10, 1.043e-10).
   * - H
     - :math:`v \to v - 1`, :math:`v` = 3-10
     - 300 K
     - Extrapolated, 1.8e-10 rising to 2.4e-10 (near gas-kinetic); a rate set fitted to LTE at 1 bar is kept
       commented out.
   * - H2
     - :math:`v \to v - 1`, :math:`v` = 1-10
     - 300 K
     - Streit & Johnston (1976), Fig. 5, extrapolated to higher and lower :math:`v`.
   * - H2O
     - 1 :math:`\to` 0, 2 :math:`\to` 1
     - :math:`T`
     - McCabe et al. (2006): :math:`k(1 \to 0) = 2.4 \times 10^{-12} e^{460/T}`, measured at 263-390 K, used at all
       :math:`T` (2.7e-12 by 3500 K, a factor ~10 uncertain there). :math:`k(2 \to 1) = 2 k(1 \to 0)`, inside the
       measured :math:`k(2)/k(1)` = 1.75-1.94 (Silvente et al. 1997; Raiche et al. 1990), single-quantum since at
       least 90% of :math:`v = 2` goes to :math:`v = 1` (Plastinina et al. 2025).

Final levels are as published, except where the table says otherwise.

*H + O3 nascent distribution.* GRANADA (Funke et al. 2012, Table 13) give :math:`F(v)` = 4, 0.5, 0.5, 1, 1, 2, 4,
19, 28, 38, 2% for :math:`v` = 0-10 (``ozone_formation_distribution``), citing Dodd et al. (1994) and Adler-Golden
(1997), and say their constants are very similar to Dodd et al.'s without saying how they were derived. Dodd et al.
take :math:`v` = 6-9 from Klenerman & Smith (1987) and add low-:math:`v` population needed to fit CIRRIS-1A limb
spectra (their "CP model", shown only in their Fig. 19). Adler-Golden (1997, Table 1) instead give
:math:`k_2(v)/k_2` = 0.01, 0.03, 0.15, 0.34, 0.47 for :math:`v` = 5-9 and none below (Charters et al. 1971,
extrapolated by Steinfeld et al. 1987). Charters et al. call :math:`v = 9` the highest accessible level at room
temperature; GRANADA's 2% in :math:`v = 10` is kept as given.

.. rubric:: Literature set: OH(A)

*Electronic quenching of A* :math:`v' = 0` (Tamura et al. 1998, Table 2, fitted at 300-2500 K and used beyond, where
the cross section levels off at :math:`\sigma_\infty`):

.. math::

    \sigma(T) = \sigma_\infty\, e^{\varepsilon / k_B T}, \qquad k = C\, \sigma(T)\, T^{1/2} \times 10^{-13},

with :math:`\sigma` in :math:`\mathrm{\mathring{A}^2}`; :math:`(\sigma_\infty, \varepsilon/k_B, C)` = (14.5, 84 K,
15.0) for H, (4.5, 224 K, 10.88) for H2 and (20.0, 434 K, 4.92) for H2O (``_OH_A_QUENCH_TAMURA``).

- Checks: H gives 5.0e-10 at 300 K, matching Becker et al. (1974); H2 is within 12% of Heard & Henderson (2000) up to
  their ~1200 K data; H2O gives :math:`\sigma` = 85 :math:`\mathrm{\mathring{A}^2}` and 7.2e-10 at 300 K, within 10%
  of Martini et al. (2017) (6.56e-10) and 0.83-1.13 times Heard & Henderson's fit up to 3500 K.
- He: 4e-14 at all :math:`T` (Martini et al. 2017, citing Dilecce et al. 2015), negligible.
- :math:`v' = 1`: :math:`v' = 0`'s rate times ``_OH_A_QUENCH_V1_FACTOR``: H2 (53.5 - 23.05)/20.8 = 1.46 and H2O
  (60.6 - 4.81)/65.6 = 0.85 from Martini et al. (2017) at 300 K, their total removal of :math:`v' = 1` less its
  1 :math:`\to` 0 relaxation; H and He 1 (no data).
- Reactive fraction (``_OH_A_REACTIVE_FRACTION``), entered as destruction of A :math:`v'`: H2 0.8 (to H2O + H; about
  80% in Brouard et al. 2017 and over 80% in Malbon et al. 2020, both citing Dempsey et al. 2007, who find under 15%
  non-reactive, and Lehman et al. 2010); H 0.2 (to
  O(:math:`^1`\ D) + H2; Schatz et al. 2001, trajectory surface hopping); He 0; H2O 0 (unmeasured; Heard & Henderson
  2000 find no reactive channel).

*Assignment of final levels.* The non-reactive part goes to every X level below A :math:`v'`, weighted by an assumed
function of the product's vibrational energy :math:`G = E_X(v'') - E_X(0)` (``_oh_product_weights``):

- H2 and He: :math:`e^{-G / 3714}`, so :math:`P(2)/P(1) = 0.4`. Cleary et al. (2007) find :math:`v'' = 1` populated
  markedly more than :math:`v'' = 2` for H2 (:math:`v'' = 0` not measured, put on the same trend). He has no data and
  takes H2's shape.
- H: a Gaussian in :math:`G` centred on 15,000 with standard deviation 4,000 (:math:`v''` ~ 4-5): Schatz et al.
  (2001) give a mean product vibrational energy of 42-45 kcal/mol for both non-reactive channels, but no distribution.
- H2O: flat over :math:`v'' \le 12`. Only "high :math:`v''`" is known (:math:`v''` = 1, 2 and 4 seen; Crosley &
  Copeland 1987).

*Vibrational relaxation* A 1 :math:`\to` 0 (``_OH_A_VET_10``, about 300 K): H2 2.305e-10 and H2O 4.81e-11 (Martini et
al. 2017), He 1.3e-12 (quoted by Cleveland & Wiesenfeld 1988). H has no data (extended set only).

.. rubric:: Extended set

Estimates where the literature has no rate (``_get_oh_extended_rates``). They add X :math:`v \to v - 1` for
:math:`v` = 11-16 (3-16 for H2O), A :math:`v' \to v' - 1` for :math:`v' \ge 2` (H2, He, H2O) and :math:`v' \ge 1`
(H), and electronic quenching of A :math:`v' \ge 2`.

*Energy-gap laws.* For H2 and He a single-quantum rate falls exponentially with the energy given to translation,

.. math::

    k = C_s\, e^{-\Delta E / E_0},

with one :math:`E_0` per collider and one constant :math:`C_s` per electronic state :math:`s`.

- :math:`E_0` and :math:`C_X` are fitted to the collider's X-state data: H2 Streit & Johnston :math:`v` = 1-7 (the
  measured-trend part), He Kohno et al. :math:`v` = 1-10. Both laws are steep: :math:`E_0` = 226 (H2) and 168 (He).
- :math:`C_A` is fixed by the measured A 1 :math:`\to` 0 rate. :math:`C_A / C_X` is about 1e3 for both (H2 1074,
  He 784).
- He's law matches its own measured :math:`v = 12` removal (3.6e-12, Sappey & Copeland 1990) to 1.5 times and is used
  as is.
- H2's law overshoots the measured :math:`v = 12` removal by H2 (3.0e-12, Sappey & Copeland 1990;
  ``_OH_X_H2_V12``) 30 times, so H2's X rates are anchored there instead: :math:`v = 11` is the geometric mean of the
  tabulated :math:`v = 10` rate and the anchor, and :math:`v \ge 12` follows He's gap dependence scaled to the anchor
  (H2, He and Ar removal agree at :math:`v = 12`).

*H* shows no gap law: its X rates rise with :math:`\Delta E` and sit near gas-kinetic. Its X rates for :math:`v \ge 11`
are the last tabulated value (2.4e-10), and its A rates that times the mean :math:`C_A / C_X` of H2 and He (929), which
the cap always cuts (see below).

*H2O*: X :math:`v \to v - 1` at :math:`v` times McCabe et al.'s :math:`v = 1` rate (harmonic, as the measured
:math:`v = 2` ratio suggests); A :math:`v' \to v' - 1` at the measured 1 :math:`\to` 0 rate for every :math:`v'`.

*A* :math:`v' \ge 2` *electronic quenching*: :math:`v' = 1`'s total for every collider (no data), with the same
reactive fraction and product weights over the X levels below each :math:`v'`.

.. rubric:: Gas-kinetic cap

Extended-set estimates only; literature rates are never capped. :math:`k_{\mathrm{gk},M} = \sigma_M
\langle v_{\mathrm{rel}} \rangle` (``base._gas_kinetic_rate``, OH mass 17.007 amu), with cross section
:math:`\sigma_M` = 30 :math:`\mathrm{\mathring{A}^2}` for H, H2 and He and 100 :math:`\mathrm{\mathring{A}^2}` for H2O
(dipole-dipole attraction; OH(A) quenching by H2O alone has 85 :math:`\mathrm{\mathring{A}^2}` at 300 K).

- X levels: each estimate is at most :math:`k_{\mathrm{gk},M}`. For :math:`v \ge 11` it is the level's only channel to
  that collider, so this bounds the level's total loss.
- A levels: the relaxation estimate is at most :math:`k_{\mathrm{gk},M}` minus the level's total electronic quenching
  (reactive part included, never below 0), so the level's total loss stays below the ceiling.

.. list-table:: Where the cap binds (estimates cut to the ceiling)
   :header-rows: 1

   * - Collider
     - 300 K
     - 1000 K
     - 2000 K
     - 3500 K
   * - H
     - A :math:`v'` = 1-10
     - A 1-10
     - A 1-10
     - A 1-10
   * - H2
     - X 16; A 2-10
     - X 16; A 3-10
     - X 16; A 3-10
     - X 16; A 4-10
   * - He
     - X 15-16; A 6-10
     - X 16; A 6-10
     - X 16; A 7-10
     - X 16; A 7-10
   * - H2O
     - none
     - none
     - none
     - none

So the H A-state relaxation always takes what electronic quenching leaves of the ceiling, and the gap laws reach the
ceiling only near dissociation (X) and for the high A levels.

.. rubric:: Not included and open items

- Predissociation of A :math:`v' \ge 2` (MYTHOS lifetimes: :math:`v' = 2` 0.1 :math:`\mu`\ s to 0.2 ns, :math:`v' = 3`
  0.2 ns to 14 ps, :math:`v' \ge 4` a few ps) is not yet a loss in the rate equations; it is planned in Phase 0
  (``notes/phase0_handover_predissociation.md``, ``notes/predissociation_fractions.tex``).
- Destruction without re-formation: O + OH and reactive OH(A) quenching remove OH while the chemistry is fixed. The
  reactive OH(A) term alone changes deep-layer OH(A) and high-X populations by up to 25%. Pending the agreed square
  solve with re-formation (``notes/oh_collisional_rates.tex``, ``notes/chemistry_coupling_plan.md``).
- Room-temperature X-state rates used at 2000-3500 K (all colliders but H2O).
- H + O3 formation evaluates the total rate at 300 K rather than the layer temperature.
- O2 vibronic quenching of OH(A) (Paul 1995) is kept commented out in ``get_rates``.
- H + OH :math:`\to` O + H2 and the identity exchange OH + H2O :math:`\to` H2O + OH are left out.
- He :math:`v` = 1-4 in the table against Kohno et al.'s measured values (up to 25% apart).

.. rubric:: Change log

- 2026-10-07: ``species_depend`` tuples; formation per unit volume divided by the OH density.
- 2026-10-07: ``OH_RATE_SET`` (extended set with gap laws and the H2 :math:`v = 12` anchor) and ``OH_H_BRANCHING``.
- 2026-10-09: H2O as a collider (X 1 :math:`\to` 0 and 2 :math:`\to` 1, A quenching and relaxation); the extended
  set extended to X :math:`v` = 16 and A :math:`v'` = 10 (all bound levels).
- 2026-10-12: moved from ``colchem.py`` to ``colchem/oh.py``.
- 2026-10-14: H + O3 formation includes GRANADA's :math:`v = 10` (2%), previously dropped.

.. rubric:: References

- Adler-Golden, S. 1997, J. Geophys. Res. 102, 19969, doi:10.1029/97JA01622.
- Atahan, S. & Alexander, M. H. 2006, J. Phys. Chem. A 110, 5436, doi:10.1021/jp055860m.
- Becker, K. H., Haaks, D. & Tatarczyk, T. 1974, Chem. Phys. Lett. 25, doi:10.1016/0009-2614(74)85368-6.
- Brouard, M. et al. 2017, J. Chem. Phys. 146, 244313, doi:10.1063/1.4989567.
- Caridade, P. J. S. B., Horta, J. & Varandas, A. J. C. 2013, Atmos. Chem. Phys. 13, 1, doi:10.5194/acp-13-1-2013.
- Charters, P. E., Macdonald, R. G. & Polanyi, J. C. 1971, Appl. Opt. 10, 1747, doi:10.1364/AO.10.001747.
- Cleary, P. A. et al. 2007, J. Chem. Phys. 126, 204316, doi:10.1063/1.2730505.
- Cleveland, C. B. & Wiesenfeld, J. R. 1988, Chem. Phys. Lett. 144, 479, doi:10.1016/0009-2614(88)87300-7.
- Crosley, D. R. & Copeland, R. A. 1987, Proc. SPIE 742, 6, doi:10.1117/12.966897.
- DeMore, W. B. et al. 1990, Chemical kinetics and photochemical data for use in stratospheric modeling, JPL Publ.
  90-1.
- Dempsey, L. P., Murray, C. & Lester, M. I. 2007, J. Chem. Phys. 127, 151101, doi:10.1063/1.2800316.
- Dilecce, G., Martini, L. M., Tosi, P., Scotoni, M. & De Benedictis, S. 2015, Plasma Sources Sci. Technol. 24,
  034007, doi:10.1088/0963-0252/24/3/034007.
- Dodd, J. A. et al. 1994, J. Geophys. Res. 99, 3559, doi:10.1029/93JD03338.
- Funke, B., Lopez-Puertas, M., Garcia-Comas, M., Kaufmann, M., Hopfner, M. & Stiller, G. P. 2012, J. Quant.
  Spectrosc. Radiat. Transf. 113, 1771, doi:10.1016/j.jqsrt.2012.05.001 (GRANADA).
- Heard, D. E. & Henderson, D. A. 2000, Phys. Chem. Chem. Phys. 2, 67, doi:10.1039/a908221b.
- Klenerman, D. & Smith, I. W. M. 1987, J. Chem. Soc. Faraday Trans. 2 83, 229, doi:10.1039/F29878300229.
- Kohno, N. et al. 2013, J. Phys. Chem. A 117, 3253, doi:10.1021/jp3114072.
- Lehman et al. 2010, J. Chem. Phys., doi:10.1063/1.3487734.
- Malbon, C. L., Zhao, B., Guo, H. & Yarkony, D. R. 2020, Phys. Chem. Chem. Phys., doi:10.1039/D0CP01754J.
- Martini, L. M. et al. 2017, J. Phys. D 50, 114003, doi:10.1088/1361-6463/aa5b28.
- McCabe, D. C. et al. 2006, Phys. Chem. Chem. Phys. 8, 4563, doi:10.1039/b609330b.
- Paul, P. H. 1995, J. Phys. Chem. 99, 8472, doi:10.1021/j100021a004.
- Plastinina, D. M. et al. 2025, Molecules 30, 540, doi:10.3390/molecules30030540.
- Raiche, G. A. et al. 1990, J. Chem. Phys. 92, 7258, doi:10.1063/1.458211.
- Sappey, A. D. & Copeland, R. A. 1990, J. Chem. Phys. 93, 5741, doi:10.1063/1.459567.
- Schatz, G. C. et al. 2001, J. Phys. Chem. A 105, 2515, doi:10.1021/jp003092n.
- Silvente, E., Richter, R. C. & Hynes, A. J. 1997, J. Chem. Soc. Faraday Trans. 93, 2821, doi:10.1039/a702575k.
- Steinfeld, J. I., Adler-Golden, S. M. & Gallagher, J. W. 1987, J. Phys. Chem. Ref. Data 16, 911,
  doi:10.1063/1.555796.
- Streit, G. E. & Johnston, H. S. 1976, J. Chem. Phys. 64, 95, doi:10.1063/1.431917.
- Tamura, M. et al. 1998, Combust. Flame 114, 502, doi:10.1016/S0010-2180(97)00324-6.
"""
import typing as t

import numpy as np
import numpy.typing as npt

from .base import RateTransition, _gas_kinetic_rate


# OH(A) collisional data and the OH energy-gap fits (see _get_oh_a_rates and
# _get_oh_extended_rates).
# --------------------------------------------------------------------------------------------------------------

# Level energies [cm^-1]: the lowest MYTHOS state of each (electronic state, v) in 16O-1H__MYTHOS.noB4.vfix.states
# (v relabelled by tests/fix_states_v_labels.py). Index = v. X runs to v = 16, just below dissociation.
_OH_X_ENERGIES = np.array([
    0.0, 3568.47, 6971.35, 10210.54, 13287.16, 16201.31, 18951.86, 21536.20, 23949.82, 26185.79, 28234.07, 30080.59,
    31706.40, 33086.40, 34194.94, 34993.54, 35443.98,
])
# A v'=0-10, all bound A levels. v'>=2 are predissociated (MYTHOS lifetimes: v'=2 0.1 us-0.2 ns, v'=3 0.2 ns-14 ps,
# v'>=4 a few ps); predissociation is not yet a loss in the rate equations (notes/chemistry_coupling_plan.md).
_OH_A_ENERGIES = np.array([
    32440.58, 35429.18, 38222.24, 40815.77, 43201.41, 45363.74, 47279.41, 48904.19, 50177.97, 51015.60, 51369.59,
])

# Electronic quenching of OH(A, v'=0), Tamura et al. (1998), Combust. Flame 114, 502, Table 2: cross section
# sigma(T) = sigma_inf exp(eps / k T) [A^2], rate k = C sigma(T) T^0.5 * 1e-13 cm^3/s; (sigma_inf, eps/k [K], C).
# Fitted 300-2500 K (used beyond that too: sigma -> sigma_inf smoothly). H: k(300 K) = 5.0e-10, matching Becker et al.
# (1974); H2: within ~12% of Heard & Henderson (2000) to their ~1200 K data. H2O: sigma(300 K) = 85 A^2, k(300 K) =
# 7.2e-10, within 10% of Martini et al. (2017) (6.56e-10) and 0.83-1.13x Heard & Henderson (2000) to 3500 K.
_OH_A_QUENCH_TAMURA = {"H": (14.5, 84.0, 15.0), "H2": (4.5, 224.0, 10.88), "H2O": (20.0, 434.0, 4.92)}
# He: negligible, 4e-14 cm^3/s (Martini et al. 2017, J. Phys. D 50, 114003, citing Dilecce et al. 2015).
_OH_A_QUENCH_HE = 4.0e-14
# v'=1 relative to v'=0. H2: Martini et al. (2017) electronic quenching, (kQ1 - k_1->0) / kQ0 = (53.5 - 23.05) / 20.8 at
# 300 K (their kQ1 includes the 1->0 relaxation, Q_A1 = A1 + Q^E_A1 + R_1->0); H2O likewise (60.6 - 4.81) / 65.6.
# H, He: no data, taken equal.
_OH_A_QUENCH_V1_FACTOR = {"H": 1.0, "H2": (53.5 - 23.05) / 20.8, "He": 1.0, "H2O": (60.6 - 4.81) / 65.6}
# Fraction of quenching that is reactive (removes OH; entered as destruction): H2 ~80% to H2O + H (Dempsey et al.
# 2007, Lehman et al. 2010, as quoted by Brouard et al. 2017 and Malbon et al. 2020); H ~20% to O(1D) + H2 (Schatz et
# al. 2001, trajectory surface hopping); He none; H2O unmeasured, taken as none (Heard & Henderson 2000 find no
# reactive channel).
_OH_A_REACTIVE_FRACTION = {"H": 0.2, "H2": 0.8, "He": 0.0, "H2O": 0.0}
# OH(A, v'=1 -> 0) vibrational relaxation [cm^3/s] at ~300 K: H2 23.05e-11, H2O 4.81e-11 (Martini et al. 2017); He
# 1.3e-12 (quoted by Cleveland & Wiesenfeld 1988). H: no data (only in the "extended" rate set).
_OH_A_VET_10 = {"H2": 2.305e-10, "He": 1.3e-12, "H2O": 4.81e-11}


def _oh_x_h2o_10(layer_temp: float) -> float:
    """
    OH(X, v=1 -> 0) + H2O [cm^3/s], McCabe et al. (2006), Phys. Chem. Chem. Phys. 8, 4563: 2.4e-12 exp(460 / T) over
    263-390 K (an H-bonded complex; falls with T). Used at all T: levels off at 2.7e-12 by 3500 K (uncertain by a
    factor ~10 there; their power-law fit would give 3e-13).
    """
    return 2.4e-12 * np.exp(460.0 / layer_temp)


# Total removal of OH(X, v=12) by H2 at room temperature [cm^3/s], Sappey & Copeland (1990), J. Chem. Phys. 93, 5741
# (He 3.6e-12, Ar 2.6e-12): the H2 anchor above the tabulated v <= 10 (see _get_oh_extended_rates).
_OH_X_H2_V12 = 3.0e-12
# OH(X, v=2) + H at 300 K, Atahan & Alexander (2006): fraction of relaxation going 2 -> 0 (1.043e-10 of 1.697e-10).
_OH_H_AA_V2_TO_0 = 1.043 / (1.043 + 0.654)


def _oh_h_prior(v: int, layer_temp: float) -> npt.NDArray[np.float64]:
    """
    Statistical (phase-space prior) distribution of OH(X, v) + H relaxation over v'' < v: atom-diatom break-up of a
    long-lived H2O complex, P(v'') ~ (E_v - E_v'' + 2 k T)^(3/2), with 2 k T the mean collision energy.
    """
    p = (_OH_X_ENERGIES[v] - _OH_X_ENERGIES[:v] + 2 * 0.6950348 * layer_temp) ** 1.5
    return p / p.sum()


# Fraction of OH(X) + H relaxation through the H2O complex (the rest is direct, v -> v-1): Atahan & Alexander's v=2
# branching at 300 K over the prior's (~0.84).
_OH_H_STATISTICAL_FRACTION = _OH_H_AA_V2_TO_0 / _oh_h_prior(v=2, layer_temp=300.0)[0]


def _oh_product_weights(collider: str, g_vib: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Relative OH(X, v'') yield of non-reactive OH(A) quenching by ``collider``, as a function of the product's
    vibrational energy above X v=0, ``g_vib`` [cm^-1]. Assumed shapes built on the only product-state information:

    - H2 (and He): Cleary et al. (2007), J. Chem. Phys. 126, 204316, OH(A, v'=0) + H2: v''=1 is populated markedly
      more than v''=2 (v''=0 not measured). Taken as exp(-G / 3714 cm^-1), i.e. P(2)/P(1) = 0.4, with v''=0 on the
      same trend.
    - H: Schatz et al. (2001), J. Phys. Chem. A 105, 2515: mean product vibrational energy ~42-45 kcal/mol (~15,000
      cm^-1, v'' ~ 4-5) for both non-reactive channels; no distribution given. Taken as a Gaussian in G centred on
      15,000 cm^-1 with a 4,000 cm^-1 standard deviation.
    - H2O: only qualitative, "high v''" (v'' = 1, 2, 4 seen; Crosley & Copeland 1987). Taken as flat over v'' <= 12.

    """
    if collider == "H":
        return np.exp(-0.5 * ((g_vib - 15000.0) / 4000.0) ** 2)
    if collider == "H2O":
        return np.where(g_vib <= _OH_X_ENERGIES[12], 1.0, 0.0)
    return np.exp(-g_vib / 3714.0)



def get_rates(layer_temp: float, rate_set: str = "literature", h_branching: str = "single") -> t.List[RateTransition]:
    """OH collisional and chemical rates at ``layer_temp`` [K] (module docstring): the X-state rates (room-temperature
    values, except H2O) and the OH(A) rates, plus the estimates of the extended set when ``rate_set`` is "extended",
    with X v >= 3 + H relaxation redistributed by ``h_branching`` (CollisionalRatesDatabase.OH_RATE_SET and
    OH_H_BRANCHING)."""
    # Adler-Golden O2 vibrational quenching; doi:10.1029/97JA01622.
    p_v_list = [0.043, 0.083, 0.15, 0.23, 0.36, 0.50, 0.72, 0.75, 0.95]
    c_val = 4.4e-12
    o2_rates = [
        (v_val, v_val - dv_val, c_val * p_v_list[v_val - 1] ** dv_val)
        for v_val in range(10) for dv_val in range(1, v_val + 1)
    ]

    # P. H. Paul O2 vibronic quenching; doi:10.1021/j100021a004.
    # NB: OH(A, v''=0, 1) electronic quenching is not specified as to which lower state: is total quenching.
    # OH(A, v'') + O_2 -> OH(X, v'') + O_2 @ 1900 K.
    # o2_vibronic_quenching_rates = [
    #     (0, 0, 13.4e-11),  # 15.6 @ 2300 K
    #     (1, 0, 15.1e-11),  #  16.8 @ 2300 K
    # ]
    # for v_u, v_l, rate in o2_vibronic_quenching_rates:
    #     rates.append(RateTransition(
    #         upper_key=("A2Sigma+", v_u),
    #         lower_key=("X2Pi", v_l),
    #         rate=rate,
    #         species_depend=("O2",),
    #     ))
    # # OH(A, v'') + O_2 -> OH(A, v'') + O_2 @ 1900 K.
    # o2_vibrational_quenching_rates = [
    #     (1, 0, 1.68e-11)  # 1.74 @ 2300 K
    # ]
    # for v_u, v_l, rate in o2_vibrational_quenching_rates:
    #     rates.append(RateTransition(
    #         upper_key=("A2Sigma+", v_u),
    #         lower_key=("A2Sigma+", v_l),
    #         rate=rate,
    #         species_depend=("O2",),
    #     ))

    # Caridade et al. (2013) O destruction (diagonal); doi:10.5194/acp-13-1-2013, Table 1, R4 rates.
    # O + OH -> O_2 + H (diagonal terms)
    o_destruction_rates = [
        (0, -26.0e-12),  # Extrapolated
        (1, -21.1e-12),  # @ 300K
        (2, -23.9e-12),  # @ 300K
        (3, -28.4e-12),  # @ 300K
        (4, -28.8e-12),  # @ 300K
        (5, -31.7e-12),  # @ 300K
        (6, -29.7e-12),  # @ 300K
        (7, -34.9e-12),  # @ 300K
        (8, -39.3e-12),  # @ 300K
        (9, -43.4e-12),  # @ 300K
        (10, -46.0e-12),  # Extrapolated
    ]
    # Caridade et al. (2013) O quenching (off-diagonal); doi:10.5194/acp-13-1-2013, Table 1, R5 rates.
    # O + OH(X, v') -> OH(X, v'') + O (off-diagonal terms) @ 300K.
    o_quenching_rates = [
        (1, 0, 19.2e-12), (2, 0, 14.2e-12), (2, 1, 10.5e-12),
        (3, 0, 9.4e-12), (3, 1, 9.6e-12), (3, 2, 8.1e-12),
        (4, 0, 6.4e-12), (4, 1, 7.8e-12), (4, 2, 6.9e-12), (4, 3, 4.8e-12),
        (5, 0, 6.3e-12), (5, 1, 4.7e-12), (5, 2, 6.0e-12), (5, 3, 3.8e-12), (5, 4, 3.8e-12),
        (6, 0, 4.6e-12), (6, 1, 4.4e-12), (6, 2, 5.0e-12), (6, 3, 4.7e-12), (6, 4, 4.1e-12), (6, 5, 4.5e-12),
        (7, 0, 3.4e-12), (7, 1, 3.1e-12), (7, 2, 3.6e-12), (7, 3, 3.3e-12), (7, 4, 3.5e-12), (7, 5, 3.1e-12),
        (7, 6, 4.0e-12),
        (8, 0, 2.4e-12), (8, 1, 2.3e-12), (8, 2, 2.4e-12), (8, 3, 2.4e-12), (8, 4, 2.1e-12), (8, 5, 2.7e-12),
        (8, 6, 3.0e-12), (8, 7, 4.2e-12),
        (9, 0, 1.2e-12), (9, 1, 1.3e-12), (9, 2, 2.1e-12), (9, 3, 1.8e-12), (9, 4, 2.0e-12), (9, 5, 1.7e-12),
        (9, 6, 1.8e-12), (9, 7, 2.1e-12), (9, 8, 3.3e-12),
    ]
    # O3 formation with vibrational distribution (GRANADA, Funke et al. 2012, Table 13; module docstring): H + O3 ->
    # OH(X, v) + O2, formation per unit volume k n_H n_O3
    # (divided by n_OH when the rates are applied). Produces OH(X, v) with temperature-dependent rate
    ozone_formation_distribution = np.array([4, 0.5, 0.5, 1, 1, 2, 4, 19, 28, 38, 2])
    # Note: total_rate = 1.4e-10 * exp(-470/T), using 300K nominal.
    total_rate = 1.4e-10 * np.exp(-470 / 300.0)
    o3_rates = [(v_val, v_val, total_rate * ozone_formation_distribution[v_val] / 100) for v_val in range(11)]

    # Kohno et al. (2013) He single-quantum vibrational quenching @ 298K; doi:10.1021/jp3114072.
    he_rates = [
        (1, 0, 3.2e-17), (2, 1, 1.4e-16), (3, 2, 4.4e-16), (4, 3, 1.2e-15),
        (5, 4, 3.2e-15), (6, 5, 8.2e-15), (7, 6, 2.1e-14), (8, 7, 5.1e-14),
        (9, 8, 1.3e-13), (10, 9, 3.4e-13),
        # (11, 9, 9.5e-13), (12, 11, 2.9e-12),
    ]
    # Atahan & Alexander (2006) H multi-quantum quenching @ 300K; doi:10.1021/jp055860m.
    h_rates_direct = [
        (1, 0, 1.600e-10),
        (2, 1, 0.654e-10),
        (2, 0, 1.043e-10),
    ]
    # H single-quantum extrapolation (conservative) @ 300K
    h_rates_extrap = [
        # (1, 0, 1.6e-10), (2, 1, 1.7e-10),
        (3, 2, 1.8e-10), (4, 3, 1.8e-10), (5, 4, 1.9e-10), (6, 5, 2.0e-10),
        (7, 6, 2.1e-10), (8, 7, 2.2e-10), (9, 8, 2.3e-10), (10, 9, 2.4e-10),
        # (11, 10, 2.6e-10),
    ]
    # Fit to LTE @ 1bar.
    # h_rates_extrap = [
    #     (3, 2, 5.8e-10), (4, 3, 1.5e-09), (5, 4, 4.0e-09), (6, 5, 9.8e-09),
    #     (7, 6, 2.4e-08), (8, 7, 7.6e-08), (9, 8, 1.8e-07), (10, 9, 5.1e-07),
    # ]
    # Streit & Johnston (1976) H2 vibrational quenching @ 300K; doi:10.1063/1.431917, Fig. 5, extrapolated to higher
    # & lower v.
    h2_rates = [
        (1, 0, 1.0e-14), (2, 1, 4.0e-14), (3, 2, 9.0e-14), (4, 3, 1.8e-13),
        (5, 4, 3.9e-13), (6, 5, 6.8e-13), (7, 6, 8.0e-13), (8, 7, 7.6e-13),
        (9, 8, 5.8e-13), (10, 9, 4.4e-13),
    ]
    # H2O: v=1 from McCabe et al. (2006) (_oh_x_h2o_10); v=2 as 2x v=1, inside the measured k(2)/k(1) = 1.75-1.94 at
    # room temperature (Silvente et al. 1997, Raiche et al. 1990, as tabulated by McCabe et al.). Single-quantum:
    # >= 90% of v=2 goes to v=1 (Plastinina et al. 2025).
    h2o_rates = [(v_val, v_val - 1, v_val * _oh_x_h2o_10(layer_temp)) for v_val in (1, 2)]

    # (upper v, lower v, rate [cm^3/s]) by the species the rates depend on, in the order they are emitted.
    rate_groups = (
        (("O2",), o2_rates),
        (("O",), [(v_val, v_val, rate) for v_val, rate in o_destruction_rates]),
        (("O",), o_quenching_rates),
        (("H", "O3"), o3_rates),
        (("He",), he_rates),
        (("H",), h_rates_direct),
        (("H",), h_rates_extrap),
        (("H2",), h2_rates),
        (("H2O",), h2o_rates),
    )
    rates = [
        RateTransition(upper_key=("X2Pi", v_u), lower_key=("X2Pi", v_l), rate=rate, species_depend=species)
        for species, species_rates in rate_groups for v_u, v_l, rate in species_rates
    ]
    rates += _get_oh_a_rates(layer_temp)
    if rate_set == "extended":
        rates += _get_oh_extended_rates(
            layer_temp=layer_temp, x_rates={"H2": h2_rates, "He": he_rates}, h_x_rate=h_rates_extrap[-1][2],
        )
    elif rate_set != "literature":
        raise ValueError(f"Unknown OH_RATE_SET {rate_set!r}.")

    branching = h_branching
    if branching == "single":
        return rates
    if branching not in ("statistical", "flat"):
        raise ValueError(f"Unknown OH_H_BRANCHING {branching!r}.")
    branched = []
    for r in rates:
        v = r.upper_key[1]
        if not (r.species_depend == ("H",) and r.upper_key[0] == r.lower_key[0] == "X2Pi" and v >= 3
                and r.lower_key[1] == v - 1):
            branched.append(r)
            continue
        if branching == "flat":
            p = np.full(v, 1.0 / v)
        else:
            p = _OH_H_STATISTICAL_FRACTION * _oh_h_prior(v=v, layer_temp=layer_temp)
            p[-1] += 1.0 - _OH_H_STATISTICAL_FRACTION
        branched += [
            RateTransition(
                upper_key=("X2Pi", v), lower_key=("X2Pi", v_low), rate=r.rate * p_low, species_depend=("H",),
            )
            for v_low, p_low in enumerate(p)
        ]
    return branched

def _oh_a_quench_total(collider: str, v_a: int, layer_temp: float) -> float:
    """Total electronic quenching rate [cm^3/s] of OH(A, v_a) by ``collider`` (see _OH_A_QUENCH_TAMURA)."""
    if collider == "He":
        k_v0 = _OH_A_QUENCH_HE
    else:
        sigma_inf, eps_on_k, c_fac = _OH_A_QUENCH_TAMURA[collider]
        k_v0 = c_fac * sigma_inf * np.exp(eps_on_k / layer_temp) * np.sqrt(layer_temp) * 1e-13
    return k_v0 * (_OH_A_QUENCH_V1_FACTOR[collider] if v_a >= 1 else 1.0)

def _oh_a_quench_rates(collider: str, v_a: int, layer_temp: float) -> t.List[RateTransition]:
    """
    Electronic quenching of OH(A, v_a) by ``collider``: the reactive fraction as destruction of A(v_a) (same upper
    and lower key, negative rate), the rest to OH(X, v'') for every X level below A(v_a), split by
    _oh_product_weights.
    """
    k_total = _oh_a_quench_total(collider=collider, v_a=v_a, layer_temp=layer_temp)
    reactive = _OH_A_REACTIVE_FRACTION[collider]
    open_v = np.nonzero(_OH_X_ENERGIES < _OH_A_ENERGIES[v_a])[0]
    weights = _oh_product_weights(collider=collider, g_vib=_OH_X_ENERGIES[open_v] - _OH_X_ENERGIES[0])
    rates = [
        RateTransition(
            upper_key=("A2Sigma+", v_a), lower_key=("X2Pi", int(v_x)), rate=rate, species_depend=(collider,),
        )
        for v_x, rate in zip(open_v, (1.0 - reactive) * k_total * weights / weights.sum())
    ]
    if reactive > 0:
        rates.append(RateTransition(
            upper_key=("A2Sigma+", v_a), lower_key=("A2Sigma+", v_a), rate=-reactive * k_total,
            species_depend=(collider,),
        ))
    return rates

def _get_oh_a_rates(layer_temp: float) -> t.List[RateTransition]:
    """
    Literature OH(A) rates: electronic quenching of v'=0 and 1 by H, H2, He and H2O (_oh_a_quench_rates), and the
    measured v'=1->0 relaxation (H2, He, H2O). They only apply when the NLTE processor aggregates OH(A) levels
    below its cutoff; otherwise the keys find no aggregate and are skipped.
    """
    rates = [
        rate for collider in ("H", "H2", "He", "H2O") for v_a in (0, 1)
        for rate in _oh_a_quench_rates(collider=collider, v_a=v_a, layer_temp=layer_temp)
    ]
    rates += [
        RateTransition(upper_key=("A2Sigma+", 1), lower_key=("A2Sigma+", 0), rate=rate, species_depend=(collider,))
        for collider, rate in _OH_A_VET_10.items()
    ]
    return rates

def _get_oh_extended_rates(
        layer_temp: float, x_rates: t.Dict[str, t.List[t.Tuple[int, int, float]]], h_x_rate: float,
) -> t.List[RateTransition]:
    """
    Estimates for OH rates missing from the literature set, mostly from energy-gap laws. For each collider with a
    gap law, a single-quantum rate falls exponentially with the energy given to translation,
    k = C_s exp(-dE / E0), with one E0 per collider and a constant C_s per electronic state s:

    - E0 and C_X from the collider's X-state data (``x_rates``; H2: Streit & Johnston v=1-7, the measured-trend
      part; He: Kohno et al. v=1-10). Both give steep laws (E0 ~ 230 and 170 cm^-1).
    - C_A from the measured A v'=1->0 rate. The A/X ratio C_A / C_X is ~1e3 for both (H2 1070, He 780).

    The H2 law overshoots the measured v=12 removal (_OH_X_H2_V12) ~30x, so H2's X rates are anchored there
    instead: v=11 log-linear between the tabulated v=10 and v=12, and above v=12 He's gap dependence (H2, He and Ar
    agree at v=12) scaled to it. The He law matches its own v=12 removal (3.6e-12) to 1.5x.
    H shows no gap law (its X rates rise with dE and sit near gas-kinetic), so its X rates are extended at the
    last tabulated value (``h_x_rate``) and its A rates taken as that times the mean A/X ratio of H2 and He. Every
    estimate is capped so that the level's total loss to the collider stays below a gas-kinetic rate (30 A^2,
    _gas_kinetic_rate): the exponential laws reach it near dissociation, and the H A-state estimates get what
    electronic quenching leaves.

    H2O (no data beyond X v=2, A v'=1): X v -> v-1 = v times McCabe et al.'s v=1 rate (harmonic scaling, as the
    measured v=2 ratio suggests), and A v' -> v'-1 = its measured 1 -> 0 rate.

    Adds: X v -> v-1 for v = 11-16 (v = 3-16 for H2O); A v' -> v'-1 for v' >= 2 (H2, He, H2O) and v' >= 1 (H); and
    electronic quenching of A v' >= 2 with v'=1's total (no data).
    """
    rates = []
    ratios = []
    laws = {}
    for collider, data in x_rates.items():
        single = [(v_u, rate) for v_u, v_l, rate in data if v_u - v_l == 1 and (collider != "H2" or v_u <= 7)]
        d_e = np.array([_OH_X_ENERGIES[v] - _OH_X_ENERGIES[v - 1] for v, _ in single])
        slope, ln_c_x = np.polyfit(d_e, np.log([rate for _, rate in single]), 1)
        e0 = -1.0 / slope
        d_e_a10 = _OH_A_ENERGIES[1] - _OH_A_ENERGIES[0]
        ln_c_a = np.log(_OH_A_VET_10[collider]) + d_e_a10 / e0
        laws[collider] = (e0, ln_c_x, ln_c_a)
        ratios.append(np.exp(ln_c_a - ln_c_x))
    a_over_x = float(np.mean(ratios))
    e0_he, ln_c_x_he, _ = laws["He"]
    he_x = np.exp(ln_c_x_he - np.diff(_OH_X_ENERGIES, prepend=np.nan) / e0_he)  # He law by upper v
    h2_x_10 = next(rate for v_u, v_l, rate in x_rates["H2"] if (v_u, v_l) == (10, 9))
    h2_x = {11: np.sqrt(h2_x_10 * _OH_X_H2_V12), **{v: _OH_X_H2_V12 * he_x[v] / he_x[12] for v in range(12, 17)}}
    for collider in ("H", "H2", "He", "H2O"):
        cap = _gas_kinetic_rate(collider=collider, temperature=layer_temp)

        def gap(state: str, d_e: float) -> float:
            if collider == "H2O":
                return min(_OH_A_VET_10["H2O"], cap)  # A only
            if collider == "H":
                return min(h_x_rate * (a_over_x if state == "A" else 1.0), cap)
            e0, ln_c_x, ln_c_a = laws[collider]
            return min(np.exp((ln_c_a if state == "A" else ln_c_x) - d_e / e0), cap)

        for v in range(3 if collider == "H2O" else 11, len(_OH_X_ENERGIES)):
            d_e = _OH_X_ENERGIES[v] - _OH_X_ENERGIES[v - 1]
            if collider == "H2":
                rate = min(h2_x[v], cap)
            elif collider == "H2O":
                rate = min(v * _oh_x_h2o_10(layer_temp), cap)
            else:
                rate = gap("X", d_e)
            rates.append(RateTransition(
                upper_key=("X2Pi", v), lower_key=("X2Pi", v - 1), rate=rate, species_depend=(collider,),
            ))
        for v_a in range(2 if collider != "H" else 1, len(_OH_A_ENERGIES)):
            d_e = _OH_A_ENERGIES[v_a] - _OH_A_ENERGIES[v_a - 1]
            # The cap bounds the level's total loss to this collider: the estimate gets what electronic quenching
            # (reactive part included) leaves.
            quench = _oh_a_quench_total(collider=collider, v_a=v_a, layer_temp=layer_temp)
            rates.append(RateTransition(
                upper_key=("A2Sigma+", v_a), lower_key=("A2Sigma+", v_a - 1),
                rate=min(gap("A", d_e), max(cap - quench, 0.0)), species_depend=(collider,),
            ))
        # A v' >= 2 electronic quenching, with v'=1's total (no data) split over the X levels below each.
        for v_a in range(2, len(_OH_A_ENERGIES)):
            rates += _oh_a_quench_rates(collider=collider, v_a=v_a, layer_temp=layer_temp)
    return rates
