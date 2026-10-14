r"""
CO vibrational relaxation in collisions with H, He, H2 and H2O: published tables interpolated in temperature, and a
measured relaxation time for H2O.

This docstring records every choice behind these rates. Update it whenever this module changes (see ``colchem.py``).

.. rubric:: Notation

- :math:`v`: CO(:math:`X^1\Sigma^+`) vibrational quantum number; keys are the integer :math:`v` (the CO processor
  aggregates on :math:`v`).
- :math:`k_M(v \to v')`: rate coefficient [:math:`\mathrm{cm^3\,s^{-1}}`] of :math:`v \to v'` in collisions with
  collider :math:`M`; :math:`T`: layer temperature [K].

.. rubric:: Data

.. list-table:: CO rate tables
   :header-rows: 1

   * - Collider
     - Channels
     - :math:`T` grid [K]
     - Source
   * - H
     - all :math:`v \to v'` with :math:`v \le 5`
     - 10-3000
     - Song et al. (2015), Table 4: quantum scattering, infinite-order sudden (see "H").
   * - He
     - :math:`v \to v - 1` and :math:`v - 2`, :math:`v` = 1-27
     - 40-5000
     - Measured :math:`1 \to 0` (Wilson et al. 1993; Wickham-Jones et al. 1987) and Cecchi-Pestellini et al.'s
       (2002) :math:`1 \to 0` scaled to the shock-tube measurements (Millikan 1964), times :math:`v`, plus
       :math:`\Delta v = -2` at their 500-1000 K ratios (see "He").
   * - H2
     - :math:`v \to v - 1` for :math:`v` = 1-5, and 2 :math:`\to` 0
     - 10-5000
     - Castro et al. (2017): five-dimensional coupled states, frozen H2 plus H2-rotation channels, Boltzmann-averaged
       here (see "H2").
   * - H2O
     - :math:`v \to v - 1`, :math:`v` = 1-27
     - formula
     - Stephenson & Mosburg (1974), von Rosenberg, Bray & Pratt (1971), Wang, Gu & Kong (1999) (see "H2O").

- Downward only; the reverse rates come from detailed balance (see ``base.py``). Final levels are as published.
- Interpolated log-linearly in :math:`T` (``base._interp_rate_batch``) and held at the end values outside each grid:
  H above 3000 K takes its 3000 K values.
- Not capped: every rate stays at most a tenth of gas-kinetic (the largest ratio is H2O :math:`27 \to 26` at 5000 K,
  3.1e-10 against 3.1e-9 for 100 :math:`\mathrm{\mathring{A}^2}`).
- No extrapolation in :math:`v` for H and H2: levels above :math:`v` = 5 have no rates for those colliders. He and H2O
  have single-quantum ladders up to :math:`v = 27`.

.. rubric:: H: Song et al. (2015)

Infinite-order sudden (IOS) vibrational rate coefficients for every :math:`v \to v' < v` from :math:`v \le 5`, 10-3000
K, on the Song, van der Avoird & Groenenboom (2013) surface, with a :math:`v = 0`-14 basis. IOS rates are already
averaged over the CO rotational states; Song et al. find them within 45% of Boltzmann-averaged close coupling for
:math:`1 \to 0` and :math:`2 \to 1`, and their :math:`1 \to 0` agrees with the shock-tube data of Glass & Kironde
(1982, 840-2680 K, as shown by Song et al.).

- Multi-quantum relaxation is large and falls smoothly with :math:`\Delta v` (at 2000 K, :math:`2 \to 0` is 0.32 of
  :math:`2 \to 1`), as in Balakrishnan, Yan & Dalgarno (2002) on another surface, consistent with strong coupling
  through the HCO well. Thermal :math:`\Delta v \le -2` rates have not been checked by close coupling or measured,
  so they rest on IOS alone.
- Against Balakrishnan et al. (2002), used until 2026-10-14: within a factor 1.5 above 1000 K, and 0.27-1.4 times at
  300 K.


.. rubric:: He: measured 1 -> 0, a single-quantum ladder and two-quantum steps

.. math::

    k_\mathrm{He}(v \to v - 1) = v\, k_{10}(T), \qquad
    k_\mathrm{He}(v \to v - 2) = r_2(v, T)\, v\, k_{10}(T), \qquad v = 1\text{-}27,

with :math:`k_{10}(T)` the :math:`1 \to 0` rate, interpolated log-linearly on ``_CO_HE_T_GRID`` and held outside
40-5000 K (``_CO_HE_10``):

- 40-150 K: Wilson et al. (1993), Table 4 (laser-induced fluorescence; the mean where a temperature has several
  values). 200-290 K: Wickham-Jones, Williams & Simpson (1987), read from the vector data of Semenov, Ivanov & Babikov
  (2013), Fig. 4 (11-21% above Wilson et al. at 100 and 150 K, where both exist).
- 500-5000 K: the :math:`1 \to 0` row of Cecchi-Pestellini et al. (2002), Table 2 (BASECOL collision 38), times 3.6
  (``_CO_HE_SCALE``): the geometric mean of the shock-tube measurements of Millikan (1964) over it at 940-2320 K,
  which lie 3.2-4.3 times above with no clear trend in :math:`T`. Peterson & McBane's (2005) full-quantum rates
  (590-1510 K) are 2.0-2.5 times above it, so the scaled rate is uncertain by about a factor 1.6. Between 290 and 500 K
  the rate is interpolated; at 580-590 K it lies between the two other values Semenov et al. plot.
- Higher levels: the harmonic ladder :math:`v\, k_{10}`, single-quantum only, up to :math:`v = 27` (harmonic energy
  below 50,000 :math:`\mathrm{cm^{-1}}`, as for H2O). Cecchi-Pestellini et al.'s own single-quantum rates at 1000 K
  rise a little faster (2.2, 3.9, 5.9, 8.2 times :math:`1 \to 0` for :math:`v` = 2-5), as do Castro et al.'s for H2
  (1.0-1.08 :math:`v`), so this is if anything an underestimate for high :math:`v`.
- The ladder is the single-quantum channel; with only small two-quantum steps it is also nearly each level's total
  loss. Against Cecchi-Pestellini et al.'s own single-quantum rates on the same 3.6 scale it is 0.41-0.89 times at
  500-1000 K (theirs rise faster than :math:`v`, 1.2-2.4 :math:`v` at 500 K), 0.92-1.07 times for :math:`v \le 4` at
  2000 K, and larger above, where theirs collapse (see below). An anharmonic factor from their low-temperature rates
  would raise the high-:math:`v` rates below 2000 K by up to 2.4 times (open item).
- :math:`r_2(v, T)` (``_CO_HE_R2``): Cecchi-Pestellini et al.'s :math:`k(v \to v - 2) / k(v \to v - 1)` for
  :math:`v` = 2-6 at 500-1000 K, held outside that range, with :math:`v > 6` taking :math:`v = 6`'s. There it is
  regular: it rises smoothly with :math:`v` and :math:`T`, from 1.2e-4 to 1.2e-3 at 500 K and 1.0e-3 to 3.1e-3 at
  1000 K, close to Castro et al.'s H2 :math:`2 \to 0 / 2 \to 1` (6.2e-4 at 500 K, 2.0e-3 at 1000 K). Above 1000 K their
  :math:`2 \to 0` departs from the trend (9.4e-2 at 2000 K, 2.0 at 5000 K) and their high-:math:`v` single-quantum rates
  start to collapse, so the ratio is held; H2's rises to 6.1e-3 by 3000 K, so holding is if anything low. These steps
  are 0.01-0.3% of a level's loss.
- :math:`\Delta v \le -3` is left out. At 500 K Cecchi-Pestellini et al.'s :math:`\Delta v = -3` is under 5e-6 of the
  single-quantum rate; from 1000 K it exceeds their :math:`\Delta v = -2`, and :math:`\Delta v \le -4` grows with
  :math:`\Delta v` even at 500 K, both against the energy-gap law.

*Why not Cecchi-Pestellini et al. (BASECOL) as published.* Their Table 2 (infinite-order sudden calculations, all
:math:`v \to v'` from :math:`v \le 6`, 500-5000 K) was used until 2026-10-14. Three problems:

- Their :math:`1 \to 0` is 3.2-4.3 times below the shock-tube measurements and 2.0-2.5 times below full-quantum
  rates (above). The authors note a factor 2-2.5 against Wickham-Jones et al.
- The multi-quantum entries contradict the paper's own statement that "rate coefficients for the transfer of more
  than one vibrational quantum are very small" and the energy-gap law. For :math:`v' = 0` the rates grow with the
  energy released instead of falling: at 500 K, :math:`6 \to 0` (12,478 :math:`\mathrm{cm^{-1}}`) is 4.6e-16 against
  4.3e-20 for :math:`6 \to 2` and 8.0e-16 for :math:`6 \to 5`. At high :math:`T` they dominate: the single-quantum
  share of a level's He relaxation is 0.47 for :math:`v = 4` and 0.015 for :math:`v = 6` at 2000 K, and 0.22 and
  0.007 at 3000 K, with :math:`6 \to 0` at 5000 K 380 times :math:`6 \to 5`.
- As those entries grow, the single-quantum rates of the same levels collapse: at 1000 K they scale roughly as
  :math:`v`, but :math:`6 \to 5` is 0.31 times :math:`1 \to 0` at 3000 K and 0.08 times at 5000 K. The table thus
  moves flux from :math:`\Delta v = -1` into :math:`v \to 0` at high :math:`v` and :math:`T`, which no physical
  mechanism here explains. We take it to be a numerical artefact of the large-:math:`\Delta v` sudden-approximation
  calculation, which no measurement or other calculation tests (all others are for :math:`v = 1`). Their table is
  used only for the :math:`1 \to 0` temperature dependence and the 500-1000 K two-quantum ratios.

*Other He data, all for* :math:`v = 1`: close coupling below about 150 K (Balakrishnan, Dalgarno & Forrey 2000,
within 1.3 times of the measurements of Wilson et al.) and Semenov et al.'s mixed quantum/classical rates (within 10%
of the measurements at 400-2500 K, not tabulated).

*H2 is not He scaled by the collision speed*, :math:`(\mu_\mathrm{He}/\mu_\mathrm{H2})^{1/2} = 1.37` (the rule McKee
et al. 1982 use for rotation, after Green & Thaddeus 1976; :math:`\mu` the reduced mass with CO). It holds roughly
for pure rotation (Castro et al.'s H2 over Cecchi-Pestellini et al.'s He is 1.3-1.6 for :math:`\Delta J = -1` at
100-500 K), but not for vibration: :math:`1 \to 0` by H2 over He is 145-540 at 40-100 K (Wilson et al. 1993) and,
from these tables, 18 at 300 K, 4.8 at 1000 K, 2.2 at 2000 K and 1.2 at 3000 K.

.. rubric:: H2: Castro et al. (2017)

The "frozen-H2" database of Castro et al. (2017) (five-dimensional coupled states; H2 held in :math:`j` = 0 for para,
1 for ortho) gives state-to-state rate coefficients :math:`k(v J \to v' J')` from 10 to 9900 K, made with R. C.
Forrey's routine, from his cross sections at collision energies of 1-10,000 :math:`\mathrm{cm^{-1}}`. The data
cover initial :math:`v \le 5` with :math:`J \le 30` for :math:`v \le 1` (para-H2 only; ortho-H2 stops at 20) and
:math:`J \le 20` above (the paper's Table 1), final :math:`J' \le 40`, and only :math:`\Delta v = -1` channels apart
from :math:`2 \to 0`. ``colrate_pipeline/castro_co_h2.py`` integrates the cross sections as Forrey's ``rate.f`` does
(it reproduces his rate files to their 5 printed figures) and reduces them to level-to-level rates:

.. math::

    k(v \to v') = \sum_J w_J \sum_{J'} k(v J \to v' J'), \qquad
    w_J = \frac{(2J + 1)\, e^{-E_{vJ} / k_B T}}{\sum_{J''} (2J'' + 1)\, e^{-E_{vJ''} / k_B T}},

with :math:`E_{vJ}` the energy of state :math:`(v, J)` [:math:`\mathrm{cm^{-1}}`] and the sums over every :math:`J`
of :math:`v` in the ExoMol Li2015 states file; ortho- and para-H2 are combined 3:1.

- Above each H2 isomer's last computed :math:`J` (30 or 20) the rate is held at that :math:`J`'s value. The summed
  rates rise slowly with :math:`J` (1.2-1.5 times from :math:`J` = 0 to 20 at 1000-3000 K), but :math:`J > 20` holds
  30% of the population at 1000 K and 67% at 3000 K, so this is a lower estimate. Continuing the :math:`J` = 10-20
  trend linearly in rotational energy would raise the rates by 12% at 1000 K, 24% at 2000 K and 33% at 3000 K.
- Every channel is computed from its initial state, including vibrationally down but rotationally up pairs, so none
  needs detailed balance.
- Check: :math:`1 \to 0` at 300 K is 3.0e-16, against 2.7e-16 from Thi et al.'s (2013) fit to measurements.
- Accuracy (Castro et al.): the coupled-states results agree with six-dimensional close coupling above 10
  :math:`\mathrm{cm^{-1}}` collision energy, but are not fully converged above 1000 K (final :math:`J'` stops at 40,
  where the rates are still large), and frozen H2 leaves out channels that excite H2 rotation. Both make the rates
  low at high :math:`T`. Forrey's thermal average integrates cross sections up to 10,000 :math:`\mathrm{cm^{-1}}`;
  its Maxwell-Boltzmann normalisation falls to 0.93 at 4000 K and 0.87 at 5000 K, so the rates there are low by up to
  13% more.
- H2-rotation channels: frozen H2 cannot change rotation, so each frozen state rate is multiplied by
  :math:`1 + k_\mathrm{rot} / k_\mathrm{same}` from the "flexible-H2" database (H2 may rotate; CO initial
  :math:`J \le 10`; ortho-H2 only for :math:`v \le 2`; the CO basis cut to :math:`J \le 20` in the initial level),
  with :math:`k_\mathrm{rot}` and :math:`k_\mathrm{same}` its rates for the same state and :math:`\Delta v` with H2
  rotationally excited and unchanged (``rotation_factor``). Above :math:`J = 10` the :math:`J = 10` factor is used, and
  ortho-H2 at :math:`v \ge 3` takes the ortho :math:`v = 2` factor. The flexible set has no :math:`2 \to 0`, so that
  channel is unchanged. This raises the level rates by 27-28% at 300 K, 12-13% at 1000 K and 20-22% at 3000 K; at
  10-50 K the ortho factor reaches 2.5-3.
- The flexible rates themselves are not used. ``thermal_rate`` reproduces their rate files too. For the transitions
  both databases share (H2 unchanged) the cross sections agree below 1000 :math:`\mathrm{cm^{-1}}` (median ratio
  0.98) and are typically 2-2.6 times larger above. At 300 K the per-state rates agree within 6% for initial
  :math:`J \le 4` but diverge to 3.1 times at :math:`J = 10`, consistent with the flexible basis cut at :math:`J = 20`.
  Boltzmann-averaged over :math:`J \le 10` in both, the shared transitions are 1.6-1.7 times the frozen ones at 300 K
  and 2.3-3.5 times at 1000-3000 K. The frozen basis is the larger, so it is kept for the H2-unchanged part.

.. rubric:: H2O

No calculation exists, and the measurements give total relaxation, so the rate is built from them:

.. math::

    k(1 \to 0) = \frac{k_B T}{p\tau\, (1 - e^{-\theta / T})},

the Landau-Teller relation :math:`1/\tau = n_\mathrm{H2O}\, k(1 \to 0)\, (1 - e^{-\theta / T})` with
:math:`n_\mathrm{H2O} = p / k_B T`, :math:`p\tau` the relaxation time of CO by H2O scaled to :math:`p` = 1 atm, and
:math:`\theta = h c \nu / k_B = 3084` K (:math:`\nu = 2143.27\ \mathrm{cm^{-1}}`, CO's band origin).

- :math:`p\tau` = 0.17 atm :math:`\mu`\ s at 1230-2600 K with virtually no temperature dependence (von Rosenberg,
  Bray & Pratt 1971: shock tube, CO emission with H2O measured in situ, assuming relaxation through H2O rotation or
  translation; they find a V-V reading does not change it significantly). That is 1.1e-12 at 1230 K, 2.0e-12 at 2000
  K and 3.0e-12 at 2600 K.
- 0.205 atm :math:`\mu`\ s at 297 K from Stephenson & Mosburg (1974; laser-induced fluorescence, 6400 :math:`\pm` 1600
  :math:`\mathrm{s^{-1}\,torr^{-1}}`, i.e. 1.97e-13); Wang, Gu & Kong (1999) give 1.7e-13 (0.24 atm :math:`\mu`\ s).
- ``_co_h2o_10`` interpolates :math:`p\tau` linearly between 297 and 1230 K and holds it outside, so above 2600 K it
  continues the measured temperature independence (6.0e-12 at 4000 K).
- Higher levels: :math:`v \to v - 1` at Wang et al.'s 300 K ratios to :math:`v = 1` (1.7, 3.4, 6.2, 8.0, 9.0, 12, 16,
  18e-13 for :math:`v` = 1-8, uncertain by up to 55% at :math:`v = 8`), applied at every :math:`T`, and in proportion
  to :math:`v` above :math:`v = 8`, up to :math:`v = 27` (harmonic energy below 50,000 :math:`\mathrm{cm^{-1}}`).
- H2O is treated as an ordinary collider (its own vibration unchanged). The H/D isotope effect (D2O about 10 times
  slower; Stephenson & Mosburg) and von Rosenberg et al.'s analysis point to CO vibration going to H2O rotation and
  translation; Wang et al. suggest the energy may go to the H2O bend in a hydrogen-bonded complex. No rate exists for
  CO(1) + H2O(000) :math:`\to` CO(0) + H2O(010) in either direction.
- Per collision H2O is about 650 times faster than H2 at 300 K, 30 times at 1000 K and 8 times at 2000 K, so at the
  1-10% H2O of cooler planets it dominates CO relaxation in cool layers.

.. rubric:: Not included

CO + CO collisions (V-V exchange depends on two CO populations); reactions (the chemistry is fixed).

.. rubric:: Change log

- 2026-10-12: moved from ``colchem.py`` to ``colchem/co.py``.
- 2026-10-12: H2 table corrected. It was the plain sum over the initial :math:`J` = 0-20 (``tests/co_collisions.py``),
  19-24 times too fast; CO NLTE results with H2 from before then are not comparable.
- 2026-10-14: H from Song et al. (2015), replacing Balakrishnan, Yan & Dalgarno (2002); H2O added as a collider.
- 2026-10-14: He from measured :math:`1 \to 0` rates, a single-quantum ladder and two-quantum steps, replacing
  Cecchi-Pestellini et al.'s (2002) table as published (see "He").
- 2026-10-14: H2 rates include the H2-rotation channels in the flexible-H2 proportion (up 12-28%).
- 2026-10-14: H2 rates computed from Forrey's cross sections, with para-H2 to :math:`J = 30` for :math:`v \le 1`
  (:math:`1 \to 0` up 3% above 1000 K).

.. rubric:: References

- Balakrishnan, N., Dalgarno, A. & Forrey, R. C. 2000, J. Chem. Phys. 113, 621, doi:10.1063/1.481838.
- Balakrishnan, N., Yan, M. & Dalgarno, A. 2002, ApJ 568, 443, doi:10.1086/321434.
- Castro, C., Doan, K., Klemka, M., Forrey, R. C., Yang, B., Stancil, P. C. & Balakrishnan, N. 2017, Mol.
  Astrophys. 6, 47, doi:10.1016/j.molap.2017.01.003.
- Cecchi-Pestellini, C., Bodo, E., Balakrishnan, N. & Dalgarno, A. 2002, ApJ 571, 1015, doi:10.1086/340020.
- Glass, G. P. & Kironde, S. 1982, J. Phys. Chem. 86, 908, doi:10.1021/j100395a014 (as shown by Song et al.).
- McKee, C. F., Storey, J. W. V., Watson, D. M. & Green, S. 1982, ApJ 259, 647, doi:10.1086/160200.
- Millikan, R. C. 1964, J. Chem. Phys. 40, 2594, doi:10.1063/1.1725568.
- Peterson, K. A. & McBane, G. C. 2005, J. Chem. Phys. 123, 084314, doi:10.1063/1.1947194.
- Semenov, A., Ivanov, M. & Babikov, D. 2013, J. Chem. Phys. 139, 074306, doi:10.1063/1.4818488.
- Song, L., Balakrishnan, N., Walker, K. M., Stancil, P. C., Thi, W.-F., Kamp, I., van der Avoird, A. &
  Groenenboom, G. C. 2015, ApJ 813, 96, doi:10.1088/0004-637X/813/2/96.
- Stephenson, J. C. & Mosburg, E. R. 1974, J. Chem. Phys. 60, 3562, doi:10.1063/1.1681574.
- Thi, W.-F. et al. 2013, A&A 551, A49, arXiv:1210.7654.
- von Rosenberg, C. W., Bray, K. N. C. & Pratt, N. H. 1971, Proc. Combust. Inst. 13, 89,
  doi:10.1016/S0082-0784(71)80013-9.
- Wang, B., Gu, Y. & Kong, F. 1999, J. Phys. Chem. A 103, 7395, doi:10.1021/jp984616a.
- Wickham-Jones, C. T., Williams, H. T. & Simpson, C. J. S. M. 1987, J. Chem. Phys. 87, 5294,
  doi:10.1063/1.453646.
- Wilson, G. J., Turnidge, M. L., Solodukhin, A. S. & Simpson, C. J. S. M. 1993, Chem. Phys. Lett. 207, 521,
  doi:10.1016/0009-2614(93)89041-F.

Green & Thaddeus (1976), Krems and Song et al. (2013) are cited as quoted in the papers above.
"""
import typing as t

import numpy as np
import numpy.typing as npt

from .base import RateTransition, _interp_rate_batch


# --------------------------------------------------------------------------------------------------------------
# CO collisional rate tables. Module-level (built once at import) rather than re-literalised inside get_rates on
# every call - it is invoked once per atmospheric layer during NLTE setup.
# --------------------------------------------------------------------------------------------------------------

# Song et al. (2015), Table 4 - CO + H: infinite-order sudden vibrational rates (module docstring).
_CO_H_T_GRID = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 200.0, 300.0, 500.0, 700.0,
                         1000.0, 1500.0, 2000.0, 2500.0, 3000.0])
_CO_H_RATES: t.Dict[t.Tuple[int, int], npt.NDArray[np.float64]] = {
    (1, 0): np.array([8.076e-17, 7.485e-17, 8.113e-17, 9.464e-17, 1.172e-16, 1.550e-16, 2.205e-16, 3.335e-16,
                      5.233e-16, 8.297e-16, 2.929e-14, 2.022e-13, 1.372e-12, 3.594e-12, 8.206e-12, 1.768e-11,
                      2.833e-11, 3.970e-11, 5.148e-11]),
    (2, 1): np.array([1.914e-16, 1.808e-16, 1.995e-16, 2.355e-16, 2.906e-16, 3.728e-16, 4.960e-16, 6.817e-16,
                      9.628e-16, 1.388e-15, 4.115e-14, 2.911e-13, 1.981e-12, 5.144e-12, 1.145e-11, 2.339e-11,
                      3.554e-11, 4.746e-11, 5.899e-11]),
    (2, 0): np.array([1.164e-17, 1.107e-17, 1.284e-17, 1.629e-17, 2.189e-17, 3.076e-17, 4.478e-17, 6.708e-17,
                      1.028e-16, 1.601e-16, 7.337e-15, 5.767e-14, 4.257e-13, 1.175e-12, 2.858e-12, 6.689e-12,
                      1.141e-11, 1.674e-11, 2.243e-11]),
    (3, 2): np.array([3.312e-16, 3.008e-16, 3.201e-16, 3.648e-16, 4.352e-16, 5.387e-16, 6.902e-16, 9.164e-16,
                      1.265e-15, 1.819e-15, 7.871e-14, 5.598e-13, 3.131e-12, 7.056e-12, 1.398e-11, 2.613e-11,
                      3.811e-11, 4.963e-11, 6.053e-11]),
    (3, 1): np.array([3.662e-17, 3.625e-17, 4.124e-17, 5.054e-17, 6.567e-17, 8.971e-17, 1.279e-16, 1.894e-16,
                      2.891e-16, 4.508e-16, 1.987e-14, 1.476e-13, 1.016e-12, 2.676e-12, 6.112e-12, 1.291e-11,
                      2.005e-11, 2.717e-11, 3.406e-11]),
    (3, 0): np.array([5.680e-18, 5.697e-18, 6.488e-18, 7.930e-18, 1.021e-17, 1.372e-17, 1.918e-17, 2.772e-17,
                      4.115e-17, 6.197e-17, 1.786e-15, 1.396e-14, 1.326e-13, 4.246e-13, 1.164e-12, 3.023e-12,
                      5.476e-12, 8.346e-12, 1.144e-11]),
    (4, 3): np.array([4.033e-16, 3.707e-16, 4.219e-16, 4.811e-16, 5.567e-16, 6.781e-16, 8.705e-16, 1.144e-15,
                      1.536e-15, 2.103e-15, 5.222e-14, 3.610e-13, 2.252e-12, 5.502e-12, 1.169e-11, 2.323e-11,
                      3.498e-11, 4.630e-11, 5.674e-11]),
    (4, 2): np.array([7.546e-17, 7.361e-17, 9.554e-17, 1.142e-16, 1.364e-16, 1.774e-16, 2.514e-16, 3.682e-16,
                      5.525e-16, 8.418e-16, 3.153e-14, 2.262e-13, 1.486e-12, 3.758e-12, 8.163e-12, 1.624e-11,
                      2.421e-11, 3.179e-11, 3.879e-11]),
    (4, 1): np.array([2.871e-17, 2.853e-17, 3.229e-17, 3.836e-17, 4.831e-17, 6.451e-17, 9.013e-17, 1.309e-16,
                      1.956e-16, 2.977e-16, 1.101e-14, 7.980e-14, 5.566e-13, 1.491e-12, 3.481e-12, 7.586e-12,
                      1.209e-11, 1.666e-11, 2.107e-11]),
    (4, 0): np.array([3.995e-18, 4.021e-18, 4.456e-18, 5.319e-18, 6.833e-18, 8.506e-18, 1.161e-17, 1.648e-17,
                      2.416e-17, 3.621e-17, 1.315e-15, 9.893e-15, 7.532e-14, 2.202e-13, 5.838e-13, 1.543e-12,
                      2.877e-12, 4.477e-12, 6.200e-12]),
    (5, 4): np.array([5.545e-16, 5.101e-16, 5.345e-16, 5.980e-16, 7.035e-16, 8.219e-16, 1.011e-15, 1.278e-15,
                      1.663e-15, 2.230e-15, 5.837e-14, 3.916e-13, 2.265e-12, 5.339e-12, 1.116e-11, 2.220e-11,
                      3.355e-11, 4.428e-11, 5.376e-11]),
    (5, 3): np.array([1.282e-16, 1.278e-16, 1.420e-16, 1.690e-16, 2.149e-16, 2.637e-16, 3.532e-16, 4.925e-16,
                      7.125e-16, 1.063e-15, 4.172e-14, 2.906e-13, 1.746e-12, 4.176e-12, 8.704e-12, 1.692e-11,
                      2.502e-11, 3.254e-11, 3.912e-11]),
    (5, 2): np.array([7.653e-17, 7.476e-17, 8.187e-17, 9.513e-17, 1.153e-16, 1.496e-16, 2.000e-16, 2.790e-16,
                      4.050e-16, 6.076e-16, 2.524e-14, 1.789e-13, 1.100e-12, 2.686e-12, 5.718e-12, 1.129e-11,
                      1.677e-11, 2.185e-11, 2.632e-11]),
    (5, 1): np.array([2.262e-17, 2.159e-17, 2.333e-17, 2.676e-17, 3.186e-17, 4.143e-17, 5.466e-17, 7.517e-17,
                      1.077e-16, 1.603e-16, 7.189e-15, 5.332e-14, 3.471e-13, 8.958e-13, 2.070e-12, 4.606e-12,
                      7.511e-12, 1.049e-11, 1.328e-11]),
    (5, 0): np.array([2.629e-18, 2.455e-18, 2.621e-18, 2.974e-18, 3.497e-18, 4.469e-18, 5.769e-18, 7.731e-18,
                      1.079e-17, 1.568e-17, 7.253e-16, 5.670e-15, 4.002e-14, 1.127e-13, 2.974e-13, 8.075e-13,
                      1.548e-12, 2.445e-12, 3.394e-12]),
}

# CO + He (module docstring, "He"): k(1 -> 0) [cm^3/s] on _CO_HE_T_GRID [K], and v -> v-1 = v k(1 -> 0) for
# v = 1-_CO_V_MAX. Up to 290 K measurements: Wilson et al. (1993), Table 4 (mean of each temperature's values, 40-150 K)
# and Wickham-Jones, Williams & Simpson (1987) (read from Semenov et al. 2013, Fig. 4, 200-290 K). From 500 K: the
# 1 -> 0 row of Cecchi-Pestellini et al. (2002), Table 2 (BASECOL), times _CO_HE_SCALE, the geometric mean of
# Millikan's (1964) shock-tube measurements over it at 940-2320 K (3.2-4.3).
_CO_V_MAX = 27  # highest CO v given He and H2O ladders: harmonic energy below 50,000 cm^-1, as for H2O
_CO_HE_SCALE = 3.6
_CO_HE_MEASURED_T = np.array([40.0, 50.0, 60.0, 70.0, 75.0, 85.0, 100.0, 150.0, 200.0, 250.0, 290.0])
_CO_HE_MEASURED = np.array([3.6e-20, 5.03e-20, 7.25e-20, 9.1e-20, 9.5e-20, 1.61e-19, 2.53e-19, 1.0e-18, 3.96e-18,
                            1.07e-17, 1.9e-17])
_CO_HE_CP_T = np.array([500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0, 1300.0, 1500.0, 2000.0, 2500.0, 3000.0,
                        3500.0, 4000.0, 4500.0, 5000.0])
_CO_HE_CP_10 = np.array([5.5e-17, 1.4e-16, 2.9e-16, 5.6e-16, 1.0e-15, 1.7e-15, 2.6e-15, 5.8e-15, 1.1e-14, 4.0e-14,
                         9.7e-14, 1.8e-13, 2.9e-13, 4.1e-13, 5.6e-13, 7.3e-13])
_CO_HE_T_GRID = np.concatenate([_CO_HE_MEASURED_T, _CO_HE_CP_T])
_CO_HE_10 = np.concatenate([_CO_HE_MEASURED, _CO_HE_SCALE * _CO_HE_CP_10])
# He Delta v = -2: k(v -> v-2) = r2 k(v -> v-1), r2 = Cecchi-Pestellini et al.'s k(v -> v-2) / k(v -> v-1) (rows
# v = 2-6) at 500-1000 K, where their table is still regular; held outside, and v > 6 takes v = 6's.
_CO_HE_R2_T = np.array([500.0, 600.0, 700.0, 800.0, 900.0, 1000.0])
_CO_HE_R2 = np.array([
    [1.23e-4, 1.91e-4, 2.84e-4, 4.00e-4, 6.09e-4, 9.74e-4],
    [2.87e-4, 4.46e-4, 6.42e-4, 8.70e-4, 1.20e-3, 1.64e-3],
    [5.14e-4, 7.89e-4, 1.16e-3, 1.58e-3, 2.03e-3, 2.60e-3],
    [8.21e-4, 1.21e-3, 1.75e-3, 2.26e-3, 2.67e-3, 3.07e-3],
    [1.15e-3, 1.56e-3, 1.83e-3, 2.03e-3, 2.27e-3, 2.42e-3],
])

# Castro et al. (2017) - CO + H2: Boltzmann averages over the initial J of the frozen-H2 state-to-state rates plus
# the H2-rotation channels, H2 3:1 ortho:para (module docstring; colrate_pipeline/castro_co_h2.py regenerates them).
_CO_H2_T_GRID = np.array([10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
                          1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000,
                          2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000, 5000], dtype=np.float64)
_CO_H2_RATES: t.Dict[t.Tuple[int, int], npt.NDArray[np.float64]] = {
    (1, 0): np.array([1.6668e-17, 1.5687e-17, 2.8933e-17, 1.2397e-16, 3.8611e-16, 9.3678e-16, 2.0502e-15, 4.0753e-15,
                      7.3979e-15, 1.2433e-14, 1.9610e-14, 2.9354e-14, 4.2057e-14, 5.8050e-14, 7.7581e-14, 1.0079e-13,
                      1.2772e-13, 1.5829e-13, 1.9234e-13, 2.2961e-13, 2.6979e-13, 3.1254e-13, 4.0418e-13, 5.0143e-13,
                      6.0137e-13, 7.0146e-13, 7.9964e-13, 8.9432e-13, 9.8432e-13, 1.0689e-12, 1.1475e-12, 1.2199e-12,
                      1.4916e-12]),
    (2, 0): np.array([1.3751e-20, 1.5473e-20, 1.7519e-20, 4.9425e-20, 2.1209e-19, 8.2785e-19, 2.7134e-18, 7.3859e-18,
                      1.7284e-17, 3.5956e-17, 6.8140e-17, 1.1966e-16, 1.9709e-16, 3.0725e-16, 4.5657e-16, 6.5056e-16,
                      8.9336e-16, 1.1875e-15, 1.5340e-15, 1.9321e-15, 2.3798e-15, 2.8739e-15, 3.9847e-15, 5.2263e-15,
                      6.5574e-15, 7.9388e-15, 9.3356e-15, 1.0719e-14, 1.2066e-14, 1.3359e-14, 1.4586e-14, 1.5739e-14,
                      2.0298e-14]),
    (2, 1): np.array([4.0789e-17, 3.8322e-17, 6.8181e-17, 2.8013e-16, 8.5555e-16, 2.0386e-15, 4.3864e-15, 8.5941e-15,
                      1.5411e-14, 2.5629e-14, 4.0061e-14, 5.9502e-14, 8.4682e-14, 1.1621e-13, 1.5454e-13, 1.9991e-13,
                      2.5237e-13, 3.1176e-13, 3.7774e-13, 4.4981e-13, 5.2737e-13, 6.0972e-13, 7.8587e-13, 9.7231e-13,
                      1.1635e-12, 1.3545e-12, 1.5416e-12, 1.7217e-12, 1.8926e-12, 2.0529e-12, 2.2017e-12, 2.3387e-12,
                      2.8500e-12]),
    (3, 2): np.array([8.8853e-17, 7.4059e-17, 1.2221e-16, 4.6085e-16, 1.3738e-15, 3.2201e-15, 6.8289e-15, 1.3219e-14,
                      2.3461e-14, 3.8666e-14, 5.9945e-14, 8.8369e-14, 1.2490e-13, 1.7031e-13, 2.2515e-13, 2.8971e-13,
                      3.6395e-13, 4.4762e-13, 5.4017e-13, 6.4091e-13, 7.4895e-13, 8.6333e-13, 1.1070e-12, 1.3638e-12,
                      1.6261e-12, 1.8875e-12, 2.1427e-12, 2.3878e-12, 2.6198e-12, 2.8370e-12, 3.0383e-12, 3.2230e-12,
                      3.9092e-12]),
    (4, 3): np.array([1.3944e-16, 1.2060e-16, 1.9037e-16, 6.8187e-16, 1.9972e-15, 4.6204e-15, 9.6840e-15, 1.8562e-14,
                      3.2672e-14, 5.3457e-14, 8.2340e-14, 1.2066e-13, 1.6961e-13, 2.3013e-13, 3.0284e-13, 3.8803e-13,
                      4.8562e-13, 5.9518e-13, 7.1599e-13, 8.4708e-13, 9.8732e-13, 1.1354e-12, 1.4501e-12, 1.7804e-12,
                      2.1170e-12, 2.4514e-12, 2.7772e-12, 3.0895e-12, 3.3846e-12, 3.6605e-12, 3.9156e-12, 4.1495e-12,
                      5.0142e-12]),
    (5, 4): np.array([1.7671e-16, 1.5832e-16, 2.5530e-16, 9.3109e-16, 2.7143e-15, 6.2182e-15, 1.2897e-14, 2.4494e-14,
                      4.2777e-14, 6.9511e-14, 1.0641e-13, 1.5507e-13, 2.1686e-13, 2.9286e-13, 3.8375e-13, 4.8978e-13,
                      6.1078e-13, 7.4617e-13, 8.9501e-13, 1.0561e-12, 1.2280e-12, 1.4092e-12, 1.7930e-12, 2.1947e-12,
                      2.6028e-12, 3.0074e-12, 3.4009e-12, 3.7773e-12, 4.1325e-12, 4.4639e-12, 4.7700e-12, 5.0503e-12,
                      6.0820e-12]),
}

# CO + H2O (module docstring, "H2O"): Landau-Teller p tau [atm us] of CO relaxed by H2O, 0.205 at 297 K (Stephenson &
# Mosburg 1974) and 0.17 at 1230-2600 K (von Rosenberg, Bray & Pratt 1971), linear in T between and held outside;
# CO's vibrational temperature theta = h c nu / k_B [K]; v -> v-1 relative to 1 -> 0 from Wang et al. (1999) at 300 K
# for v = 1-8, continued in proportion to v up to _CO_V_MAX.
_K_B = 1.380649e-16  # Boltzmann constant [erg/K]
_ATM = 1.01325e6  # 1 standard atmosphere [dyn/cm^2]
_US = 1e-6  # 1 microsecond [s]
_HC_ON_KB = 1.4387769  # second radiation constant h c / k_B [cm K]
_CO_BAND_ORIGIN = 2143.271  # CO v = 1 -> 0 band origin [cm^-1] (ExoMol Li2015)
_CO_THETA = _HC_ON_KB * _CO_BAND_ORIGIN  # CO's vibrational temperature [K], 3084 K
_CO_H2O_P_TAU_T = np.array([297.0, 1230.0])  # [K]
_CO_H2O_P_TAU = np.array([0.205, 0.17])  # [atm us]
_CO_H2O_WANG = np.array([1.7, 3.4, 6.2, 8.0, 9.0, 12.0, 16.0, 18.0])  # 1e-13 cm^3/s, v = 1-8
_CO_H2O_V_RATIO = (np.concatenate([_CO_H2O_WANG, _CO_H2O_WANG[-1] * np.arange(9, _CO_V_MAX + 1) / 8])
                   / _CO_H2O_WANG[0])


def _co_h2o_10(layer_temp: float) -> float:
    """
    CO(v=1 -> 0) + H2O [cm^3/s] from the Landau-Teller relaxation time at 1 atm of H2O: 1/tau = n k (1 - exp(-theta/T))
    with n = p / (k_B T), so k = k_B T / (p tau (1 - exp(-theta / T))).
    """
    p_tau = np.interp(layer_temp, _CO_H2O_P_TAU_T, _CO_H2O_P_TAU) * _ATM * _US  # [atm us] -> [dyn cm^-2 s]
    return _K_B * layer_temp / (p_tau * (1.0 - np.exp(-_CO_THETA / layer_temp)))


_CO_RATE_GROUPS: t.List[t.Tuple[npt.NDArray[np.float64], t.Dict[t.Tuple[int, int], npt.NDArray[np.float64]], str]] = [
    (_CO_H_T_GRID, _CO_H_RATES, "H"),
    (_CO_H2_T_GRID, _CO_H2_RATES, "H2"),
]


def get_rates(layer_temp: float) -> list[RateTransition]:
    """CO collisional rates [cm^3/s] at ``layer_temp`` [K] (module docstring): the H and H2 tables interpolated in
    temperature, and single-quantum ladders on k(1 -> 0) for He (interpolated, with Delta v = -2) and H2O (from its
    relaxation time)."""
    rates = []
    layer_temp_arr = np.array([layer_temp])

    for t_grid, rate_dict, partner in _CO_RATE_GROUPS:
        transitions = list(rate_dict.keys())
        rate_matrix = np.stack(list(rate_dict.values()))
        interpolated = _interp_rate_batch(temp_profile=layer_temp_arr, temp_grid=t_grid, rate_matrix=rate_matrix)[0]
        for (v_u, v_l), rate in zip(transitions, interpolated):
            rates.append(RateTransition(
                upper_key=v_u,
                lower_key=v_l,
                rate=float(rate),
                species_depend=(partner,),
            ))
    k_he = _interp_rate_batch(temp_profile=layer_temp_arr, temp_grid=_CO_HE_T_GRID,
                              rate_matrix=_CO_HE_10[None, :])[0, 0]
    for collider, k_10, v_ratio in (("He", k_he, np.arange(1, _CO_V_MAX + 1)),
                                    ("H2O", _co_h2o_10(layer_temp), _CO_H2O_V_RATIO)):
        rates += [RateTransition(upper_key=v, lower_key=v - 1, rate=float(ratio * k_10), species_depend=(collider,))
                  for v, ratio in enumerate(v_ratio, start=1)]
    r2 = _interp_rate_batch(temp_profile=layer_temp_arr, temp_grid=_CO_HE_R2_T, rate_matrix=_CO_HE_R2)[0]
    rates += [RateTransition(upper_key=v, lower_key=v - 2, rate=float(r2[min(v, 6) - 2] * v * k_he),
                             species_depend=("He",)) for v in range(2, _CO_V_MAX + 1)]
    return rates
