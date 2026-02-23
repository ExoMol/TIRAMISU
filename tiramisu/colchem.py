import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
import typing as t
import numba


@dataclass
class RateTransition:
    """Single collisional rate transition."""
    upper_key: int | t.Tuple[str, int]
    lower_key: int | t.Tuple[str, int]
    rate: float  # cm^3/s
    mol_depend: str  # Collision partner species


class CollisionalRatesDatabase:
    """
    Database of collisional and chemical rates for various species.
    Completely independent of any specific molecule instance.
    """

    @staticmethod
    def get_rates(species: str, layer_temp: float | None = None) -> list[RateTransition]:
        """
        Get collisional rates for a species.

        Parameters:
            species: Chemical formula (e.g., 'OH', 'CO')
            layer_temp: Temperature in K (required for temperature-dependent rates)

        Returns:
            List of RateTransition objects
        """
        if species == "OH":
            return CollisionalRatesDatabase._get_oh_rates()
        elif species == "CO":
            if layer_temp is None:
                raise ValueError("CO rates require layer_temp parameter")
            return CollisionalRatesDatabase._get_co_rates(layer_temp)
        else:
            return []

    @staticmethod
    def _get_oh_rates() -> t.List[RateTransition]:
        """OH collisional and chemical rates (300K nominal)."""
        rates = []

        # Adler-Golden O2 vibrational quenching; doi:10.1029/97JA01622.
        p_v_list = [0.043, 0.083, 0.15, 0.23, 0.36, 0.50, 0.72, 0.75, 0.95]
        c_val = 4.4e-12
        for v_val in range(10):
            for dv_val in range(1, v_val + 1):
                rates.append(RateTransition(
                    upper_key=("X2Pi", v_val),
                    lower_key=("X2Pi", v_val - dv_val),
                    rate=c_val * p_v_list[v_val - 1] ** dv_val,
                    mol_depend="O2",
                ))

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
        #         mol_depend="O2",
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
        #         mol_depend="O2",
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
        for v_val, rate in o_destruction_rates:
            rates.append(RateTransition(
                upper_key=("X2Pi", v_val),
                lower_key=("X2Pi", v_val),
                rate=rate,
                mol_depend="O",
            ))

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
        for v_u, v_l, rate in o_quenching_rates:
            rates.append(RateTransition(
                upper_key=("X2Pi", v_u),
                lower_key=("X2Pi", v_l),
                rate=rate,
                mol_depend="O",
            ))

        # O3 formation with vibrational distribution
        # Produces OH(X, v) with temperature-dependent rate
        ozone_formation_distribution = np.array([4, 0.5, 0.5, 1, 1, 2, 4, 19, 28, 38, 2])
        # Note: total_rate = 1.4e-10 * exp(-470/T), using 300K nominal.
        total_rate = 1.4e-10 * np.exp(-470 / 300.0)
        for v_val in range(10):
            v_rate = total_rate * ozone_formation_distribution[v_val] / 100
            rates.append(RateTransition(
                upper_key=("X2Pi", v_val),
                lower_key=("X2Pi", v_val),
                rate=v_rate,
                mol_depend="O3",
            ))

        # Kohno et al. (2013) He single-quantum vibrational quenching @ 298K; doi:10.1021/jp3114072.
        he_rates = [
            (1, 0, 3.2e-17), (2, 1, 1.4e-16), (3, 2, 4.4e-16), (4, 3, 1.2e-15),
            (5, 4, 3.2e-15), (6, 5, 8.2e-15), (7, 6, 2.1e-14), (8, 7, 5.1e-14),
            (9, 8, 1.3e-13), (10, 9, 3.4e-13),
            # (11, 9, 9.5e-13), (12, 11, 2.9e-12),
        ]
        for v_u, v_l, rate in he_rates:
            rates.append(RateTransition(
                upper_key=("X2Pi", v_u),
                lower_key=("X2Pi", v_l),
                rate=rate,
                mol_depend="H",
            ))

        # Atahan & Alexander (2006) H multi-quantum quenching @ 300K; doi:10.1021/jp055860m.
        h_rates_direct = [
            (1, 0, 1.600e-10),
            (2, 1, 0.654e-10),
            (2, 0, 1.043e-10),
        ]
        for v_u, v_l, rate in h_rates_direct:
            rates.append(RateTransition(
                upper_key=("X2Pi", v_u),
                lower_key=("X2Pi", v_l),
                rate=rate,
                mol_depend="H",
            ))

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
        for v_u, v_l, rate in h_rates_extrap:
            rates.append(RateTransition(
                upper_key=("X2Pi", v_u),
                lower_key=("X2Pi", v_l),
                rate=rate,
                mol_depend="H",
            ))

        # Streit & Johnston (1976) H2 vibrational quenching @ 300K; doi:10.1063/1.431917, Fig. 5, extrapolated to higher & lower v.
        h2_rates = [
            (1, 0, 1.0e-14), (2, 1, 4.0e-14), (3, 2, 9.0e-14), (4, 3, 1.8e-13),
            (5, 4, 3.9e-13), (6, 5, 6.8e-13), (7, 6, 8.0e-13), (8, 7, 7.6e-13),
            (9, 8, 5.8e-13), (10, 9, 4.4e-13),
        ]
        for v_u, v_l, rate in h2_rates:
            rates.append(RateTransition(
                upper_key=("X2Pi", v_u),
                lower_key=("X2Pi", v_l),
                rate=rate,
                mol_depend="H2",
            ))

        return rates

    @staticmethod
    @numba.njit(cache=True, error_model="numpy")
    def _interp_rate(temp: float, temp_grid: npt.NDArray[np.float64], rate_grid: npt.NDArray[np.float64]) -> float:
        """Fast temperature interpolation."""
        return np.interp(temp, temp_grid, rate_grid)

    @staticmethod
    def _get_co_rates(layer_temp: float) -> list[RateTransition]:
        """CO collisional rates with temperature interpolation."""
        rates = []

        # BASECOL: Balakrishnan et al (2002) - CO + H
        co_h_t_list = np.array([100.0, 200.0, 300.0, 500.0, 700.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0])
        co_h_rates = {
            (1, 0): np.array([2.2000e-15, 6.6600e-14, 3.7900e-13, 2.1300e-12, 5.1600e-12,
                              1.1000e-11, 2.2300e-11, 3.4000e-11, 4.5500e-11, 5.6100e-11]),
            (2, 0): np.array([4.7000e-16, 1.6000e-14, 1.0500e-13, 6.8900e-13, 1.8100e-12,
                              4.2000e-12, 9.2000e-12, 1.4700e-11, 1.9900e-11, 2.4600e-11]),
            (2, 1): np.array([2.6900e-15, 6.6700e-14, 3.9200e-13, 2.3800e-12, 5.9600e-12,
                              1.2900e-11, 2.5700e-11, 3.7900e-11, 4.8700e-11, 5.7500e-11]),
            (3, 0): np.array([3.0200e-16, 8.7100e-15, 5.1500e-14, 3.1500e-13, 8.2400e-13,
                              1.9400e-12, 4.3900e-12, 7.0900e-12, 9.5300e-12, 1.1500e-11]),
            (3, 1): np.array([1.3900e-15, 3.8700e-14, 2.3300e-13, 1.3900e-12, 3.4700e-12,
                              7.5500e-12, 1.5100e-11, 2.2000e-11, 2.7600e-11, 3.1700e-11]),
            (3, 2): np.array([3.4700e-15, 7.0700e-14, 4.0400e-13, 2.3300e-12, 5.6500e-12,
                              1.2000e-11, 2.3600e-11, 3.4400e-11, 4.3300e-11, 4.9800e-11]),
            (4, 0): np.array([1.0900e-16, 3.5600e-15, 2.2300e-14, 1.4500e-13, 3.9200e-13,
                              9.5500e-13, 2.1700e-12, 3.3500e-12, 4.2400e-12, 4.8400e-12]),
            (4, 1): np.array([7.1700e-16, 2.4000e-14, 1.4400e-13, 8.6300e-13, 2.1600e-12,
                              4.7200e-12, 9.3600e-12, 1.3200e-11, 1.5800e-11, 1.7300e-11]),
            (4, 2): np.array([1.5000e-15, 5.1000e-14, 3.0600e-13, 1.7400e-12, 4.1700e-12,
                              8.6800e-12, 1.6300e-11, 2.2300e-11, 2.6200e-11, 2.8400e-11]),
            (4, 3): np.array([2.5100e-15, 6.6100e-14, 3.8800e-13, 2.1700e-12, 5.0000e-12,
                              1.0800e-11, 2.0900e-11, 2.9300e-11, 3.5200e-11, 3.8700e-11]),
        }

        for (v_u, v_l), rate_array in co_h_rates.items():
            interpolated_rate = CollisionalRatesDatabase._interp_rate(layer_temp, co_h_t_list, rate_array)
            rates.append(RateTransition(
                upper_key=("X1Sigma+", v_u),
                lower_key=("X1Sigma+", v_l),
                rate=interpolated_rate,
                mol_depend="H",
            ))

        # BASECOL: Cecchi-Pestellini et al (2002) - CO + He.
        co_he_t_list = np.array([500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0, 1300.0,
                                 1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0, 4500.0, 5000.0])
        co_he_rates = {
            (1, 0): np.array([5.5000e-17, 1.4000e-16, 2.9000e-16, 5.6000e-16, 1.0000e-15, 1.7000e-15, 2.6000e-15,
                              5.8000e-15, 1.1000e-14, 4.0000e-14, 9.7000e-14, 1.8000e-13, 2.9000e-13, 4.1000e-13,
                              5.6000e-13, 7.3000e-13]),
            (2, 0): np.array([1.6000e-20, 6.1000e-20, 1.9000e-19, 5.2000e-19, 1.4000e-18, 3.7000e-18, 1.1000e-17,
                              8.3000e-17, 4.5000e-16, 8.0000e-15, 4.8000e-14, 1.6000e-13, 3.9000e-13, 7.7000e-13,
                              1.3000e-12, 2.0000e-12]),
            (2, 1): np.array([1.3000e-16, 3.2000e-16, 6.7000e-16, 1.3000e-15, 2.3000e-15, 3.8000e-15, 6.0000e-15,
                              1.3000e-14, 2.5000e-14, 8.5000e-14, 1.9000e-13, 3.3000e-13, 5.0000e-13, 6.7000e-13,
                              8.5000e-13, 1.0000e-12]),
            (3, 0): np.array([5.1000e-23, 1.6000e-21, 4.6000e-20, 6.1000e-19, 4.6000e-18, 2.3000e-17, 8.8000e-17,
                              6.8000e-16, 3.0000e-15, 3.5000e-14, 1.6000e-13, 4.3000e-13, 9.1000e-13, 1.6000e-12,
                              2.6000e-12, 3.7000e-12]),
            (3, 1): np.array([6.6000e-20, 2.5000e-19, 7.7000e-19, 2.0000e-18, 4.9000e-18, 1.1000e-17, 2.1000e-17,
                              7.0000e-17, 1.9000e-16, 1.3000e-15, 5.7000e-15, 1.8000e-14, 4.4000e-14, 9.2000e-14,
                              1.7000e-13, 2.7000e-13]),
            (3, 2): np.array([2.3000e-16, 5.6000e-16, 1.2000e-15, 2.3000e-15, 4.1000e-15, 6.7000e-15, 1.1000e-14,
                              2.3000e-14, 4.2000e-14, 1.3000e-13, 2.5000e-13, 4.0000e-13, 5.5000e-13, 6.8000e-13,
                              8.0000e-13, 9.1000e-13]),
            (4, 0): np.array([3.9000e-21, 1.9000e-19, 3.1000e-18, 2.5000e-17, 1.3000e-16, 4.6000e-16, 1.3000e-15,
                              6.6000e-15, 2.2000e-14, 1.6000e-13, 5.2000e-13, 1.2000e-12, 2.1000e-12, 3.4000e-12,
                              4.9000e-12, 6.7000e-12]),
            (4, 1): np.array([3.2000e-22, 3.6000e-21, 4.1000e-20, 3.1000e-19, 1.6000e-18, 5.8000e-18, 1.7000e-17,
                              9.5000e-17, 3.4000e-16, 3.2000e-15, 1.4000e-14, 4.1000e-14, 9.5000e-14, 1.8000e-13,
                              3.1000e-13, 4.8000e-13]),
            (4, 2): np.array([1.9000e-19, 7.1000e-19, 2.2000e-18, 5.7000e-18, 1.3000e-17, 2.6000e-17, 4.8000e-17,
                              1.3000e-16, 2.7000e-16, 9.8000e-16, 2.8000e-15, 7.3000e-15, 1.7000e-14, 3.6000e-14,
                              6.5000e-14, 1.1000e-13]),
            (4, 3): np.array([3.7000e-16, 9.0000e-16, 1.9000e-15, 3.6000e-15, 6.4000e-15, 1.0000e-14, 1.6000e-14,
                              3.3000e-14, 5.7000e-14, 1.5000e-13, 2.5000e-13, 3.5000e-13, 4.4000e-13, 5.2000e-13,
                              5.8000e-13, 6.4000e-13]),
            (5, 0): np.array([1.4000e-18, 2.6000e-17, 2.1000e-16, 9.8000e-16, 3.3000e-15, 8.6000e-15, 1.9000e-14,
                              6.4000e-14, 1.6000e-13, 6.8000e-13, 1.7000e-12, 3.1000e-12, 4.9000e-12, 7.1000e-12,
                              9.5000e-12, 1.2000e-11]),
            (5, 1): np.array([1.5000e-20, 2.8000e-19, 2.3000e-18, 1.1000e-17, 3.9000e-17, 1.1000e-16, 2.4000e-16,
                              9.0000e-16, 2.4000e-15, 1.4000e-14, 4.5000e-14, 1.1000e-13, 2.2000e-13, 3.8000e-13,
                              5.9000e-13, 8.6000e-13]),
            (5, 2): np.array([1.4000e-21, 9.1000e-21, 4.6000e-20, 1.8000e-19, 5.9000e-19, 1.6000e-18, 3.9000e-18,
                              1.7000e-17, 5.8000e-17, 6.7000e-16, 3.8000e-15, 1.3000e-14, 3.3000e-14, 6.7000e-14,
                              1.2000e-13, 1.8000e-13]),
            (5, 3): np.array([4.6000e-19, 1.7000e-18, 4.9000e-18, 1.2000e-17, 2.4000e-17, 4.3000e-17, 7.0000e-17,
                              1.5000e-16, 2.5000e-16, 6.9000e-16, 2.0000e-15, 6.0000e-15, 1.6000e-14, 3.4000e-14,
                              6.3000e-14, 1.0000e-13]),
            (5, 4): np.array([5.6000e-16, 1.4000e-15, 2.8000e-15, 5.3000e-15, 9.0000e-15, 1.4000e-14, 2.0000e-14,
                              3.7000e-14, 5.8000e-14, 1.1000e-13, 1.7000e-13, 2.0000e-13, 2.3000e-13, 2.5000e-13,
                              2.7000e-13, 2.8000e-13]),
            (6, 0): np.array([4.6000e-16, 3.2000e-15, 1.3000e-14, 3.7000e-14, 8.2000e-14, 1.6000e-13, 2.6000e-13,
                              5.9000e-13, 1.1000e-12, 2.9000e-12, 5.3000e-12, 8.2000e-12, 1.1000e-11, 1.5000e-11,
                              1.8000e-11, 2.2000e-11]),
            (6, 1): np.array([4.9000e-18, 3.5000e-17, 1.5000e-16, 4.3000e-16, 9.8000e-16, 1.9000e-15, 3.4000e-15,
                              8.4000e-15, 1.7000e-14, 5.9000e-14, 1.4000e-13, 2.9000e-13, 5.0000e-13, 7.9000e-13,
                              1.1000e-12, 1.5000e-12]),
            (6, 2): np.array([4.3000e-20, 3.3000e-19, 1.5000e-18, 4.5000e-18, 1.1000e-17, 2.4000e-17, 4.7000e-17,
                              1.5000e-16, 3.9000e-16, 2.8000e-15, 1.2000e-14, 3.5000e-14, 7.6000e-14, 1.4000e-13,
                              2.2000e-13, 3.3000e-13]),
            (6, 3): np.array([3.9000e-21, 1.5000e-20, 4.3000e-20, 1.0000e-19, 2.3000e-19, 5.1000e-19, 1.1000e-18,
                              6.0000e-18, 2.8000e-17, 5.3000e-16, 3.5000e-15, 1.3000e-14, 3.3000e-14, 6.7000e-14,
                              1.2000e-13, 1.8000e-13]),
            (6, 4): np.array([9.2000e-19, 2.8000e-18, 6.4000e-18, 1.2000e-17, 2.0000e-17, 2.9000e-17, 4.0000e-17,
                              6.3000e-17, 8.9000e-17, 2.3000e-16, 9.5000e-16, 3.4000e-15, 8.8000e-15, 1.8000e-14,
                              3.3000e-14, 5.2000e-14]),
            (6, 5): np.array([8.0000e-16, 1.8000e-15, 3.5000e-15, 5.9000e-15, 8.8000e-15, 1.2000e-14, 1.6000e-14,
                              2.3000e-14, 3.1000e-14, 4.5000e-14, 5.2000e-14, 5.6000e-14, 5.7000e-14, 5.7000e-14,
                              5.8000e-14, 5.8000e-14]),
        }

        for (v_u, v_l), rate_array in co_he_rates.items():
            interpolated_rate = CollisionalRatesDatabase._interp_rate(layer_temp, co_he_t_list, rate_array)
            rates.append(RateTransition(
                upper_key=("X1Sigma+", v_u),
                lower_key=("X1Sigma+", v_l),
                rate=interpolated_rate,
                mol_depend="He",
            ))

        # CASTRO et al. (2017) - CO + H2
        co_h2_t_list = np.array([10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
                                 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000,
                                 2200, 2400, 2600, 2800, 3000, 3200, 3400, 3600, 3800, 4000, 5000])
        co_h2_rates = {
            (1, 0): np.array([1.61805512e-16, 2.54663993e-16, 4.91120250e-16, 2.25936160e-15, 7.15516662e-15,
                              1.71298374e-14, 3.65476572e-14, 7.08395793e-14, 1.25822947e-13, 2.07643653e-13,
                              3.22658876e-13, 4.77172168e-13, 6.77051638e-13, 9.27299160e-13, 1.23165471e-12,
                              1.59232793e-12, 2.00985980e-12, 2.48316600e-12, 3.00968734e-12, 3.58562312e-12,
                              4.20621176e-12, 4.86602155e-12, 6.27980892e-12, 7.77945746e-12, 9.32008865e-12,
                              1.08627181e-11, 1.23754879e-11, 1.38338976e-11, 1.52201235e-11, 1.65220367e-11,
                              1.77322109e-11, 1.88469949e-11, 2.30241971e-11]),
            (2, 0): np.array([3.10674276e-19, 3.75234806e-19, 4.55142395e-19, 1.37684350e-18, 5.16487400e-18,
                              1.78816739e-17, 5.46113392e-17, 1.42628275e-16, 3.25752720e-16, 6.68406562e-16,
                              1.25761225e-15, 2.20132827e-15, 3.62223004e-15, 5.64773288e-15, 8.39838841e-15,
                              1.19771772e-14, 1.64615319e-14, 2.18988868e-14, 2.83056191e-14, 3.56686418e-14,
                              4.39492880e-14, 5.30877028e-14, 7.36231359e-14, 9.65595314e-14, 1.21129009e-13,
                              1.46600502e-13, 1.72330984e-13, 1.97786959e-13, 2.22547334e-13, 2.46296390e-13,
                              2.68808310e-13, 2.89935422e-13, 3.73228722e-13]),
            (2, 1): np.array([4.00437508e-16, 6.16908160e-16, 1.14492060e-15, 5.04690161e-15, 1.57581608e-14,
                              3.73310329e-14, 7.88842090e-14, 1.51645161e-13, 2.67430623e-13, 4.38509895e-13,
                              6.77372151e-13, 9.96212567e-13, 1.40619519e-12, 1.91665608e-12, 2.53437012e-12,
                              3.26306670e-12, 4.10323222e-12, 5.05219403e-12, 6.10446269e-12, 7.25220186e-12,
                              8.48577427e-12, 9.79433767e-12, 1.25898077e-11, 1.55451269e-11, 1.85726923e-11,
                              2.15968546e-11, 2.45563601e-11, 2.74041907e-11, 3.01064584e-11, 3.26403549e-11,
                              3.49922388e-11, 3.71555221e-11, 4.52285372e-11]),
            (3, 2): np.array([7.71143449e-16, 1.12172710e-15, 1.98106950e-15, 8.23536580e-15, 2.51970002e-14,
                              5.87905414e-14, 1.22499025e-13, 2.32687476e-13, 4.06155203e-13, 6.59926334e-13,
                              1.01095307e-12, 1.47543863e-12, 2.06789133e-12, 2.80009830e-12, 3.68023360e-12,
                              4.71227976e-12, 5.89583282e-12, 7.22629425e-12, 8.69537529e-12, 1.02917214e-11,
                              1.20017292e-11, 1.38102353e-11, 1.76586336e-11, 2.17091836e-11, 2.58434844e-11,
                              2.99602065e-11, 3.39777646e-11, 3.78343216e-11, 4.14855681e-11, 4.49022300e-11,
                              4.80671655e-11, 5.09728127e-11, 6.17570014e-11]),
            (4, 3): np.array([1.26646837e-15, 1.75638799e-15, 3.01500388e-15, 1.20929833e-14, 3.64674729e-14,
                              8.40910133e-14, 1.73277267e-13, 3.26019122e-13, 5.64446425e-13, 9.10541662e-13,
                              1.38580129e-12, 2.01040681e-12, 2.80208293e-12, 3.77486958e-12, 4.93808078e-12,
                              6.29566971e-12, 7.84601623e-12, 9.58231788e-12, 1.14931470e-11, 1.35633804e-11,
                              1.57751717e-11, 1.81088447e-11, 2.30594396e-11, 2.82519276e-11, 3.35363573e-11,
                              3.87851718e-11, 4.38964870e-11, 4.87933564e-11, 5.34213130e-11, 5.77447586e-11,
                              6.17433306e-11, 6.54087072e-11, 7.89527187e-11]),
            (5, 4): np.array([1.67197351e-15, 2.33784474e-15, 4.06587681e-15, 1.64417997e-14, 4.93519645e-14,
                              1.12805135e-13, 2.30175315e-13, 4.29268805e-13, 7.37532972e-13, 1.18171161e-12,
                              1.78748895e-12, 2.57856928e-12, 3.57537168e-12, 4.79364960e-12, 6.24333641e-12,
                              7.92790486e-12, 9.84420031e-12, 1.19828887e-11, 1.43293483e-11, 1.68645770e-11,
                              1.95665242e-11, 2.24111624e-11, 2.84283259e-11, 3.47191472e-11, 4.11040611e-11,
                              4.74312710e-11, 5.35802029e-11, 5.94605090e-11, 6.50087267e-11, 7.01837860e-11,
                              7.49630870e-11, 7.93377274e-11, 9.54353945e-11]),
        }

        for (v_u, v_l), rate_array in co_h2_rates.items():
            interpolated_rate = CollisionalRatesDatabase._interp_rate(layer_temp, co_h2_t_list, rate_array)
            rates.append(RateTransition(
                upper_key=("X1Sigma+", v_u),
                lower_key=("X1Sigma+", v_l),
                rate=interpolated_rate,
                mol_depend="H2"
            ))

        return rates
