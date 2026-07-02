import logging
import numpy as np
import numpy.typing as npt
import typing as t
import numba

from astropy import units as u, constants as ac
from dataclasses import dataclass

from .chemistry import ChemicalProfile, SpeciesFormula

log = logging.getLogger(__name__)

rate_unit = 1 / u.s

ac_h_c_on_kB = ac.h * ac.c.cgs / ac.k_B
const_h_c_on_kB = ac_h_c_on_kB.value


@dataclass
class RateTransition:
    """Single collisional rate transition."""
    upper_key: int | t.Tuple[str, int] | t.Tuple[int, int, int, str]
    lower_key: int | t.Tuple[str, int] | t.Tuple[int, int, int, str]
    rate: float  # cm^3/s
    mol_depend: str  # Collision partner species


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
            temperature_dependent=False,
            collision_partners={"O2", "O", "O3", "He", "H", "H2"}
        ),
        "CO": SpeciesRateInfo(
            temperature_dependent=True,
            collision_partners={"H", "He", "H2"}
        ),
        "H2O": SpeciesRateInfo(
            temperature_dependent=True,
            collision_partners={"H2"}
        ),
    }

    @classmethod
    def has_rates(cls, species: str) -> bool:
        """
        Check if rates are configured for a species.

        Parameters
        ----------
        species : str
            Chemical formula (e.g., 'OH', 'CO')

        Returns
        -------
        bool
            True if rates are available
        """
        return species in cls.SPECIES_INFO

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

    @classmethod
    def get_collision_partners(cls, species: str) -> t.Set[str]:
        """
        Get the collision partners for which rates are configured.

        Parameters
        ----------
        species : str
            Chemical formula

        Returns
        -------
        set of str
            Collision partner species (e.g., {'O2', 'O', 'He'})

        Raises
        ------
        KeyError
            If species has no configured rates
        """
        return cls.SPECIES_INFO[species].collision_partners

    @classmethod
    def get_species_info(cls, species: str) -> SpeciesRateInfo | None:
        """
        Get metadata for a species.

        Parameters
        ----------
        species : str
            Chemical formula

        Returns
        -------
        SpeciesRateInfo or None
            Rate metadata, or None if species not configured
        """
        return cls.SPECIES_INFO.get(species)

    @classmethod
    def get_configured_species(cls) -> t.Set[str]:
        """
        Get all species with configured rates.

        Returns
        -------
        set of str
            Species with collisional rates configured
        """
        return set(cls.SPECIES_INFO.keys())

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
            return CollisionalRatesDatabase._get_oh_rates()
        elif species == "CO":
            if layer_temp is None:
                raise ValueError("CO rates require layer_temp parameter")
            return CollisionalRatesDatabase._get_co_rates(layer_temp)
        elif species == "H2O":
            if layer_temp is None:
                raise ValueError("H2O rates require layer_temp parameter")
            return CollisionalRatesDatabase._get_h2o_rates(layer_temp)
        else:
            return []

    @staticmethod
    def _sum_collisional_rates(
            rates_table: t.List[RateTransition],
            temperature_profile: u.Quantity,
            chem_profile: ChemicalProfile,
            density_profile: u.Quantity,
            agg_lookup_cache: t.Dict,
            id_agg_cutoff: int,
    ) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        """
        Vectorized computation of collisional rates for all layers at once.

        Used for species with temperature-independent rates (e.g., OH). Fully vectorized over both layers and
        transitions for maximum performance.

        Parameters
        ----------
        rates_table : list of RateTransition
            Pre-computed rate transitions.
        temperature_profile : astropy.units.Quantity
            Temperature at each layer, shape (n_layers,).
        chem_profile : ChemicalProfile
            Chemical abundance profile.
        density_profile : astropy.units.Quantity
            Total number density profile, shape (n_layers,).
        agg_lookup_cache : dict
            Mapping from state keys to (state_id, energy) tuples.
        id_agg_cutoff : int
            Maximum state ID to include in rate summation.

        Returns
        -------
        c_fi_profile : np.ndarray
            Sum of downward rates, shape (n_layers,), units: 1/s.
        c_if_profile : np.ndarray
            Sum of upward rates, shape (n_layers,), units: 1/s.
        mean_energy_dif : float
            Mean energy gap for collisional rates, units: 1/cm.
        """
        n_layers = len(temperature_profile)
        temp_vals = temperature_profile.value  # [n_layers]

        # Initialize output arrays
        c_fi_profile = np.zeros(n_layers, dtype=np.float64)
        c_if_profile = np.zeros(n_layers, dtype=np.float64)

        # Pre-filter rates and extract metadata
        transitions_by_partner = {}
        energy_diffs = []
        weights = []

        for rate in rates_table:
            # Check if collision partner exists
            if rate.mol_depend not in chem_profile.species:
                continue

            try:
                upper_id, upper_energy = agg_lookup_cache[rate.upper_key]
                lower_id, lower_energy = agg_lookup_cache[rate.lower_key]
            except KeyError:
                continue

            # Only include if both states within cutoff and not diagonal
            if upper_id > id_agg_cutoff or lower_id > id_agg_cutoff or upper_id == lower_id:
                continue

            # Add to partner group
            partner = rate.mol_depend
            if partner not in transitions_by_partner:
                transitions_by_partner[partner] = []

            energy_diff = upper_energy - lower_energy
            energy_diffs.append(energy_diff)
            weights.append(rate.rate)

            transitions_by_partner[partner].append({
                'upper_id': upper_id,
                'lower_id': lower_id,
                'energy_diff': energy_diff,
                'rate': rate.rate,  # cm^3/s
            })

        if len(transitions_by_partner) == 0:
            return c_fi_profile, c_if_profile, 0

        # TODO: This crashes if weights is 0.
        energy_diffs = np.array(energy_diffs)
        weights = np.array(weights)
        mean_energy_dif = (energy_diffs * weights).sum() / weights.sum()
        # mean_energy_dif = np.mean(mean_energy_dif)

        # Process each collision partner
        for partner, partner_transitions in transitions_by_partner.items():
            # Get number density profile for this partner [n_layers] in 1/cm^3
            num_dens_profile = (
                    chem_profile[SpeciesFormula(partner)] * density_profile
            ).to_value(u.cm ** -3)  # [n_layers]

            # Extract arrays for this partner's transitions
            energy_diffs = np.array([t['energy_diff'] for t in partner_transitions], dtype=np.float64)
            base_rates = np.array([t['rate'] for t in partner_transitions], dtype=np.float64)  # cm^3/s

            # Vectorized computation over all layers and transitions
            # Shape: [n_layers, n_transitions]
            # Units: cm^3/s * 1/cm^3 = 1/s (no conversion needed!)
            c_fi_matrix = base_rates[None, :] * num_dens_profile[:, None]

            # Vectorized detailed balance
            # Shape: [n_layers, n_transitions]
            exp_factors = np.exp(-energy_diffs[None, :] * const_h_c_on_kB / temp_vals[:, None])
            c_if_matrix = c_fi_matrix * exp_factors

            # Sum over transitions for each layer
            c_fi_profile += np.sum(c_fi_matrix, axis=1)
            c_if_profile += np.sum(c_if_matrix, axis=1)

        return c_fi_profile, c_if_profile, mean_energy_dif

    @staticmethod
    def _sum_collisional_rates_t_dependent(
            species: str,
            temperature_profile: u.Quantity,
            chem_profile: ChemicalProfile,
            density_profile: u.Quantity,
            agg_lookup_cache: t.Dict,
            id_agg_cutoff: int,
    ) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        """
        Compute collisional rates for species with temperature-dependent rates.

        Used for species like CO where rates must be interpolated at each temperature. Processes layers sequentially
        with temperature-specific rate interpolation.

        Parameters
        ----------
        species : str
            Chemical formula of the species (e.g. 'CO').
        temperature_profile : astropy.units.Quantity
            Temperature at each layer, shape (n_layers,).
        chem_profile : ChemicalProfile
            Chemical abundance profile.
        density_profile : astropy.units.Quantity
            Total number density profile, shape (n_layers,).
        agg_lookup_cache : dict
            Mapping from state keys to (state_id, energy) tuples.
        id_agg_cutoff : int
            Maximum state ID to include in rate summation.

        Returns
        -------
        c_fi_profile : np.ndarray
            Sum of downward rates, shape (n_layers,), units: 1/s.
        c_if_profile : np.ndarray
            Sum of upward rates, shape (n_layers,), units: 1/s.
        mean_energy_dif : float
            Mean energy gap for collisional rates, units: 1/cm.
        """
        n_layers = len(temperature_profile)

        # Initialize output arrays
        c_fi_profile = np.zeros(n_layers, dtype=np.float64)
        c_if_profile = np.zeros(n_layers, dtype=np.float64)
        energy_diffs = []
        weights = []

        # Process each layer with its own temperature-interpolated rates
        for layer_idx in range(n_layers):
            layer_temp = temperature_profile[layer_idx].value

            # Get rates interpolated for this temperature
            rates_table = CollisionalRatesDatabase.get_rates(
                species=species,
                layer_temp=layer_temp
            )

            if not rates_table:
                continue

            # Filter and compute for this layer
            for rate in rates_table:
                if rate.mol_depend not in chem_profile.species:
                    continue

                try:
                    upper_id, upper_energy = agg_lookup_cache[rate.upper_key]
                    lower_id, lower_energy = agg_lookup_cache[rate.lower_key]
                except KeyError:
                    continue

                # Only include if both states within cutoff and not diagonal
                if upper_id > id_agg_cutoff or lower_id > id_agg_cutoff or upper_id == lower_id:
                    continue

                # Number density for this partner at this layer (1/cm^3)
                depend_num_dens = (
                        chem_profile[SpeciesFormula(rate.mol_depend)][layer_idx] * density_profile[layer_idx]
                ).to_value(u.cm ** -3)

                # Compute downward rate: cm^3/s * 1/cm^3 = 1/s
                c_fi = rate.rate * depend_num_dens

                # Detailed balance for upward rate
                energy_diff = upper_energy - lower_energy
                energy_diffs.append(energy_diff)
                weights.append(rate.rate)
                c_if = c_fi * np.exp(-energy_diff * const_h_c_on_kB / layer_temp)

                # Accumulate
                c_fi_profile[layer_idx] += c_fi
                c_if_profile[layer_idx] += c_if

        energy_diffs = np.array(energy_diffs)
        weights = np.array(weights)
        mean_energy_dif = (energy_diffs * weights).sum() / weights.sum()
        # mean_energy_dif = np.mean(mean_energy_dif)

        return c_fi_profile, c_if_profile, mean_energy_dif

    @staticmethod
    def compute_total_collisional_rates_profile(
            species: str,
            temperature_profile: u.Quantity,
            chem_profile: ChemicalProfile,
            density_profile: u.Quantity,
            agg_lookup_cache: t.Dict,
            id_agg_cutoff: int,
    ) -> t.Tuple[u.Quantity, u.Quantity, u.Quantity]:
        """
        Compute total collisional rates for all layers.

        Computes sum of all c_fi (downward) and c_if (upward) collisional rates for transitions where both upper and
        lower states are below the cutoff. Used for approximate excitation temperature calculations.

        Parameters
        ----------
        species : str
            Chemical formula of the species (e.g. 'OH', 'CO').
        temperature_profile : astropy.units.Quantity
            Temperature at each layer, shape (n_layers,).
        chem_profile : ChemicalProfile
            Chemical abundance profile.
        density_profile : astropy.units.Quantity
            Total number density profile, shape (n_layers,).
        agg_lookup_cache : dict
            Mapping from state keys to (state_id, energy) tuples.
        id_agg_cutoff : int
            Maximum state ID to include in rate summation.

        Returns
        -------
        c_fi_profile : astropy.units.Quantity
            Sum of downward collision rates at each layer, shape (n_layers,), units: 1/s.
        c_if_profile : astropy.units.Quantity
            Sum of upward collision rates at each layer, shape (n_layers,), units: 1/s.
        mean_energy_dif : astropy.units.Quantity
            Mean energy gap for collisional rates, units: 1/cm.

        Notes
        -----
        - Diagonal (formation/destruction) rates are excluded.
        - For temperature-dependent species (e.g., CO), rates are interpolated per layer.
        - For temperature-independent species (e.g., OH), rates are vectorized over all layers.
        """
        if not CollisionalRatesDatabase.has_rates(species):
            log.warning(f"No collisional rates configured for {species}")
            n_layers = len(temperature_profile)
            return np.zeros(n_layers) << rate_unit, np.zeros(n_layers) << rate_unit, 0 << 1 / u.cm

        # Choose computation method based on temperature dependence
        if CollisionalRatesDatabase.is_temperature_dependent(species):
            c_fi_profile, c_if_profile, mean_energy_dif = CollisionalRatesDatabase._sum_collisional_rates_t_dependent(
                species=species,
                temperature_profile=temperature_profile,
                chem_profile=chem_profile,
                density_profile=density_profile,
                agg_lookup_cache=agg_lookup_cache,
                id_agg_cutoff=id_agg_cutoff,
            )
        else:
            # Get rates once (temperature-independent)
            rates_table = CollisionalRatesDatabase.get_rates(
                species=species,
                layer_temp=None
            )

            c_fi_profile, c_if_profile, mean_energy_dif = CollisionalRatesDatabase._sum_collisional_rates(
                rates_table=rates_table,
                temperature_profile=temperature_profile,
                chem_profile=chem_profile,
                density_profile=density_profile,
                agg_lookup_cache=agg_lookup_cache,
                id_agg_cutoff=id_agg_cutoff,
            )

        return c_fi_profile << rate_unit, c_if_profile << rate_unit, mean_energy_dif * 1 / u.cm

    @staticmethod
    @numba.njit(cache=True, error_model="numpy")
    def _interp_rate(temp: float, temp_grid: npt.NDArray[np.float64], rate_grid: npt.NDArray[np.float64]) -> float:
        """
        Temperature interpolation of collisional rate coefficients.

        Uses log-linear (piece-wise power-law) interpolation, which is physically appropriate because collisional rates
        vary over many orders of magnitude and are approximately power-law in temperature.  A linear fallback is used
        for any zero-valued grid points so that near-zero rates at low temperature are handled gracefully rather than
        producing -inf in log space.
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

    # ----------------------------------------- BEGIN SPECIES SPECIFIC METHODS -----------------------------------------

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
                mol_depend="He",
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
        # h_rates_extrap = [
        #     # (1, 0, 1.6e-10), (2, 1, 1.7e-10),
        #     (3, 2, 1.8e-10), (4, 3, 1.8e-10), (5, 4, 1.9e-10), (6, 5, 2.0e-10),
        #     (7, 6, 2.1e-10), (8, 7, 2.2e-10), (9, 8, 2.3e-10), (10, 9, 2.4e-10),
        #     # (11, 10, 2.6e-10),
        # ]
        # Fit to LTE @ 1bar.
        h_rates_extrap = [
            (3, 2, 5.8e-10), (4, 3, 1.5e-09), (5, 4, 4.0e-09), (6, 5, 9.8e-09),
            (7, 6, 2.4e-08), (8, 7, 7.6e-08), (9, 8, 1.8e-07), (10, 9, 5.1e-07),
        ]
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
                upper_key=v_u,
                lower_key=v_l,
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
                upper_key=v_u,
                lower_key=v_l,
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
                upper_key=v_u,
                lower_key=v_l,
                rate=interpolated_rate,
                mol_depend="H2"
            ))

        return rates

    @staticmethod
    def _get_h2o_rates(layer_temp: float) -> t.List[RateTransition]:
        """
        H2O + H2 vibrational collisional rate coefficients.

        Data source: aggregated rates from quantum scattering calculations, summed over all rotational sub-states
        (J, Ka, Kc) for ortho- and para-H2O separately.  Reference temperatures: 200–5000 K.

        State keys follow the convention ``(v1, v2, v3, isomer)`` where isomer is ``"o"`` (ortho) or ``"p"`` (para), and
        (v1, v2, v3) are the symmetric-stretch, bend, and asymmetric-stretch vibrational quantum numbers.

        Rates are interpolated log-linearly via ``_interp_rate``.

        Parameters
        ----------
        layer_temp : float
            Gas temperature in K.

        Returns
        -------
        list of RateTransition
            One entry per (upper, lower, isomer) combination.
        """
        # Temperature grid shared by both isomers (K)
        h2o_h2_t_list = np.array([200.0, 400.0, 800.0, 1200.0, 1600.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0, 5000.0])

        # Data from A. Faure & E. Josselin (2008), doi:10.1051/0004-6361:200810717
        # Limited up to 5000 cm-1.

        # --- ortho-H2O + H2 ---
        h2o_h2_ortho_rates: t.Dict[t.Tuple, npt.NDArray[np.float64]] = {
            # (v1u, v2u, v3u, v1l, v2l, v3l)
            (0, 1, 0, 0, 0, 0): np.array(
                [1.5622153387014665e-10, 1.6629794504010064e-10, 3.6344339971400234e-10, 7.301591647700061e-10,
                 1.2655186941999865e-09, 1.989634796000006e-09, 3.173826854000071e-09, 4.654350342000007e-09,
                 6.472358715999993e-09, 8.499157076000069e-09, 1.3518406776000066e-08]),
            (0, 0, 0, 0, 1, 0): np.array(
                [4.8979988707999944e-20, 1.9034239682499943e-17, 4.69876311099999e-15, 2.883695443000001e-14,
                 4.27127634000004e-13, 9.77352200000002e-13, 1.921478420000017e-12, 3.2501459900000035e-12,
                 4.937935899999976e-12, 7.04288522e-12, 1.2405820299999861e-11]),
            (0, 2, 0, 0, 0, 0): np.array(
                [3.7797296675052783e-11, 4.027489782620771e-11, 8.855432065511038e-11, 1.7746792687800082e-10,
                 3.086062631e-10, 4.844856923000002e-10, 7.72382297700001e-10, 1.1334269600000027e-09,
                 1.5716231340000183e-09, 2.0770306450000324e-09, 3.295745127000018e-09]),
            (0, 2, 0, 0, 1, 0): np.array(
                [1.4492065079599606e-10, 1.5627036524499853e-10, 3.4016137879999847e-10, 6.80783073000005e-10,
                 1.1863064500000012e-09, 1.857221089999994e-09, 2.964020990000001e-09, 4.345640840000001e-09,
                 6.014446579999991e-09, 7.944905440000015e-09, 1.2673659289999999e-08]),
            (0, 0, 0, 0, 2, 0): np.array(
                [1.7055881100000033e-28, 1.803590449999996e-23, 3.3806011999999897e-19, 5.303468999999994e-18,
                 5.630117000000002e-16, 1.6970150000000061e-15, 4.176577000000013e-15, 8.217648000000037e-15,
                 1.3867290000000007e-14, 2.1447060000000005e-14, 4.236987000000001e-14]),
            (0, 1, 0, 0, 2, 0): np.array(
                [7.697758099999985e-20, 2.6129254300000027e-17, 5.118917599999986e-15, 3.0735128e-14,
                 3.7924610000000136e-13, 8.595574000000007e-13, 1.6783030000000058e-12, 2.8610580000000004e-12,
                 4.306020999999994e-12, 6.132936999999998e-12, 1.0787735999999981e-11]),
            (1, 0, 0, 0, 0, 0): np.array(
                [4.127903971991521e-12, 4.468501280534446e-12, 9.78647417390997e-12, 1.9366108302059786e-11,
                 3.380269334199972e-11, 5.3005908939999766e-11, 8.443156778999969e-11, 1.2389448358999997e-10,
                 1.7085273279999967e-10, 2.2581231099999844e-10, 3.5826533200000125e-10]),
            (1, 0, 0, 0, 1, 0): np.array(
                [4.127903971972611e-12, 4.468501243942981e-12, 9.786445650000015e-12, 1.936586269999984e-11,
                 3.3796897999999704e-11, 5.299094509999976e-11, 8.439893699999985e-11, 1.2383502100000013e-10,
                 1.7075796999999947e-10, 2.256716229999983e-10, 3.580042780000006e-10]),
            (1, 0, 0, 0, 2, 0): np.array(
                [3.27367942699999e-11, 3.489724999999995e-11, 7.483290639999995e-11, 1.4690103499999968e-10,
                 2.5262277999999977e-10, 3.914227000000002e-10, 6.187725700000021e-10, 9.066502500000014e-10,
                 1.2478981000000017e-09, 1.6400102200000024e-09, 2.5971227800000097e-09]),
            (0, 0, 0, 1, 0, 0): np.array(
                [6.530498000000017e-32, 4.226331000000001e-26, 2.9791010000000063e-21, 6.283320999999988e-20,
                 1.1587209999999999e-17, 3.844980000000007e-17, 1.0166790000000008e-16, 2.105042000000003e-16,
                 3.676989999999997e-16, 5.837780000000003e-16, 1.1926760000000016e-15]),
            (0, 2, 0, 1, 0, 0): np.array(
                [4.7909164140000054e-14, 3.0497889200000046e-13, 1.7007043500000011e-12, 4.780235100000006e-12,
                 1.0310782299999997e-11, 1.823641899999998e-11, 3.080462500000001e-11, 4.7148253999999903e-11,
                 6.634884499999984e-11, 8.970679999999996e-11, 1.465443399999997e-10]),
            (0, 1, 0, 1, 0, 0): np.array(
                [7.665670000000027e-24, 1.592945599999999e-20, 1.174941e-17, 9.489264000000019e-17,
                 2.0153060000000023e-15, 4.991109999999999e-15, 1.0496639999999989e-14, 1.868871999999998e-14,
                 2.92953e-14, 4.290469999999991e-14, 7.80461e-14]),
            (0, 0, 1, 0, 0, 0): np.array(
                [2.5505088427845632e-12, 3.6567797241442275e-12, 9.728247824379896e-12, 2.065888273169998e-11,
                 3.785872035e-11, 6.073217329999949e-11, 9.815966649999907e-11, 1.446446876999979e-10,
                 2.0166723099999834e-10, 2.6652190499999683e-10, 4.2722355400000106e-10]),
            (0, 0, 1, 0, 1, 0): np.array(
                [2.5505088427162053e-12, 3.656779543049989e-12, 9.728094519999988e-12, 2.065768370000001e-11,
                 3.7829099999999836e-11, 6.06615399999997e-11, 9.800221999999922e-11, 1.4435551999999856e-10,
                 2.0119838999999775e-10, 2.6582619999999755e-10, 4.259092000000018e-10]),
            (0, 0, 1, 0, 2, 0): np.array(
                [2.0307917999999918e-11, 2.9133118000000056e-11, 7.540581999999989e-11, 1.577677199999986e-10,
                 2.7744809999999816e-10, 4.381906999999979e-10, 6.988935999999985e-10, 1.0199680000000003e-09,
                 1.4159699999999994e-09, 1.8739030000000033e-09, 2.965786000000001e-09]),
            (0, 0, 1, 1, 0, 0): np.array(
                [9.125234999999999e-10, 1.5582662000000017e-09, 2.7863320000000025e-09, 3.892722000000002e-09,
                 4.739302000000005e-09, 5.605757999999995e-09, 6.25706e-09, 6.848989999999995e-09,
                 7.425160000000004e-09, 7.933149999999982e-09, 8.872589999999991e-09]),
            (0, 0, 0, 0, 0, 1): np.array(
                [2.497341499999994e-31, 2.1417294999999965e-25, 1.6064630000000005e-20, 3.026504000000002e-19,
                 5.716840000000009e-17, 1.7420750000000025e-16, 4.694503000000003e-16, 9.798420000000008e-16,
                 1.7364379999999977e-15, 2.7770449999999986e-15, 5.751330000000011e-15]),
            (0, 1, 0, 0, 0, 1): np.array(
                [2.7977633e-23, 7.717920999999993e-20, 5.982509999999996e-17, 4.3699609999999987e-16,
                 9.550280000000007e-15, 2.179857e-14, 4.677545000000001e-14, 8.350440000000001e-14,
                 1.3307080000000001e-13, 1.9513459999999998e-13, 3.5909059999999993e-13]),
            (1, 0, 0, 0, 0, 1): np.array(
                [3.222485899999999e-10, 5.273701299999999e-10, 1.0142572e-09, 1.4113003999999996e-09,
                 1.8889540000000002e-09, 2.219718999999998e-09, 2.4825570000000033e-09, 2.7197810000000014e-09,
                 2.939584999999999e-09, 3.138943000000003e-09, 3.5102560000000014e-09]),
            (0, 2, 0, 0, 0, 1): np.array(
                [1.3623583000000011e-14, 1.3951204000000001e-13, 1.4289573000000013e-12, 4.0773659999999985e-12,
                 1.2572110000000003e-11, 2.1730249999999956e-11, 3.735388e-11, 5.730298e-11, 8.181739999999995e-11,
                 1.1191309999999998e-10, 1.8374539999999963e-10]),
        }

        # --- para-H2O + H2 ---
        h2o_h2_para_rates: t.Dict[t.Tuple, npt.NDArray[np.float64]] = {
            (0, 1, 0, 0, 0, 0): np.array(
                [1.4468296578463645e-10, 1.6033103477657013e-10, 3.6159723919134315e-10, 7.228202533186276e-10,
                 1.2721070833720107e-09, 1.9974421300334366e-09, 3.202128296862668e-09, 4.704551969605054e-09,
                 6.502267351655913e-09, 8.61063667056062e-09, 1.368852915700209e-08]),
            (0, 0, 0, 0, 1, 0): np.array(
                [4.4888545709199725e-20, 1.805084316529996e-17, 4.629957907199948e-15, 2.859649695200014e-14,
                 4.2252799700000386e-13, 9.655259329999988e-13, 1.889920002000012e-12, 3.217998019000007e-12,
                 4.926248599999958e-12, 6.948109870000021e-12, 1.2436170640000069e-11]),
            (0, 2, 0, 0, 0, 0): np.array(
                [3.584069104887183e-11, 3.9785856164190916e-11, 8.967029909219985e-11, 1.804841873094161e-10,
                 3.1378366854561945e-10, 4.984751407562586e-10, 7.936033681704248e-10, 1.1647496831758667e-09,
                 1.61350970512124e-09, 2.124203307787731e-09, 3.4097606625169126e-09]),
            (0, 2, 0, 0, 1, 0): np.array(
                [1.376997174126559e-10, 1.5311897376190105e-10, 3.4534426357626944e-10, 6.883919084785353e-10,
                 1.2042999117213367e-09, 1.908955403844294e-09, 3.029185195659678e-09, 4.465831991360924e-09,
                 6.17584444208081e-09, 8.101705576705582e-09, 1.3059281591677956e-08]),
            (0, 1, 0, 0, 2, 0): np.array(
                [6.996567169999985e-20, 2.4420630799999968e-17, 5.0156578999999774e-15, 3.039253600000007e-14,
                 3.7495435000000164e-13, 8.491366999999995e-13, 1.657872400000004e-12, 2.8032953999999997e-12,
                 4.284286899999997e-12, 6.044941999999973e-12, 1.0767311000000006e-11]),
            (0, 0, 0, 0, 2, 0): np.array(
                [1.590841410000002e-28, 1.7326932400000033e-23, 3.384045600000003e-19, 5.332883100000009e-18,
                 5.665615999999994e-16, 1.7081220000000091e-15, 4.183869000000008e-15, 8.272825999999973e-15,
                 1.405345599999998e-14, 2.155363800000001e-14, 4.3024239999999996e-14]),
            (1, 0, 0, 0, 0, 0): np.array(
                [3.842192882159499e-12, 4.313505462494841e-12, 9.69352873455596e-12, 1.9464480413146993e-11,
                 3.356908786274685e-11, 5.3640249776831696e-11, 8.427901511039161e-11, 1.251950222801015e-10,
                 1.729523170787838e-10, 2.266626401087101e-10, 3.639188195562587e-10]),
            (1, 0, 0, 0, 1, 0): np.array(
                [3.842192882142132e-12, 4.313505428124958e-12, 9.693501054572022e-12, 1.9464238474656977e-11,
                 3.356340250374681e-11, 5.362553704683172e-11, 8.424713772039183e-11, 1.2513676609010173e-10,
                 1.7285885047878411e-10, 2.2652523900870976e-10, 3.6365986815626007e-10]),
            (1, 0, 0, 0, 2, 0): np.array(
                [3.06805806868e-11, 3.363277478445996e-11, 7.421368864461994e-11, 1.466608515552699e-10,
                 2.502447442659711e-10, 3.945134005567185e-10, 6.156203627212124e-10, 9.116663386474009e-10,
                 1.2482153818127712e-09, 1.6394719856294662e-09, 2.6269314151802926e-09]),
            (0, 0, 0, 1, 0, 0): np.array(
                [6.042884899999998e-32, 4.055709e-26, 2.9772980000000064e-21, 6.327502000000007e-20,
                 1.1648310000000023e-17, 3.8636090000000005e-17, 1.0212940000000007e-16, 2.1141939999999993e-16,
                 3.7223660000000126e-16, 5.833774000000009e-16, 1.2113369999999986e-15]),
            (0, 2, 0, 1, 0, 0): np.array(
                [5.254903602999998e-14, 3.3948072679999954e-13, 1.9120410899999977e-12, 5.367241200000003e-12,
                 1.1373419200000001e-11, 2.039508e-11, 3.424075799999999e-11, 5.192821999999997e-11,
                 7.335710399999994e-11, 9.897713800000015e-11, 1.6299107599999971e-10]),
            (0, 1, 0, 1, 0, 0): np.array(
                [7.019660700000023e-24, 1.516416399999998e-20, 1.1572914000000019e-17, 9.477238000000002e-17,
                 2.0110130000000037e-15, 5.000871999999999e-15, 1.046155799999998e-14, 1.8605519999999995e-14,
                 2.935264000000003e-14, 4.2609380000000024e-14, 7.853110000000015e-14]),
            (0, 0, 1, 0, 0, 0): np.array(
                [2.4093511995989676e-12, 3.451812083955521e-12, 9.180327723089869e-12, 1.9509567907499877e-11,
                 3.574847200999984e-11, 5.727119029999973e-11, 9.211334819999915e-11, 1.367332380999988e-10,
                 1.9057942699999872e-10, 2.539799589999989e-10, 4.0565207700000167e-10]),
            (0, 0, 1, 0, 1, 0): np.array(
                [2.409351199531324e-12, 3.451811906999985e-12, 9.180177709999946e-12, 1.9508398899999928e-11,
                 3.571958299999986e-11, 5.720211999999975e-11, 9.194908999999922e-11, 1.3645064999999866e-10,
                 1.901216799999992e-10, 2.532809099999987e-10, 4.0430900000000084e-10]),
            (0, 0, 1, 0, 2, 0): np.array(
                [1.9245701999999973e-11, 2.7399760000000027e-11, 7.053812999999972e-11, 1.4834757999999985e-10,
                 2.6070419999999957e-10, 4.1194529999999965e-10, 6.574005000000009e-10, 9.604827000000004e-10,
                 1.3317156000000007e-09, 1.7628413999999941e-09, 2.8011379999999966e-09]),
            (0, 0, 1, 1, 0, 0): np.array(
                [8.974479700000008e-10, 1.5250990000000012e-09, 2.6584542999999985e-09, 3.7218516000000087e-09,
                 4.488720999999991e-09, 5.2841950000000005e-09, 5.922463999999985e-09, 6.501564000000009e-09,
                 7.029143000000002e-09, 7.526886999999997e-09, 8.431253999999996e-09]),
            (0, 2, 0, 0, 0, 1): np.array(
                [9.757318999999988e-15, 1.174561900000001e-13, 1.3122470000000015e-12, 3.876731000000003e-12,
                 1.2179139999999992e-11, 2.1264560000000002e-11, 3.672619999999995e-11, 5.670959999999993e-11,
                 8.216580000000006e-11, 1.1014870000000003e-10, 1.824211999999996e-10]),
            (1, 0, 0, 0, 0, 1): np.array(
                [3.3200839999999957e-10, 5.397683999999996e-10, 1.0267880000000007e-09, 1.4227640000000004e-09,
                 1.9032529999999977e-09, 2.2387049999999984e-09, 2.5047349999999978e-09, 2.7438580000000015e-09,
                 2.965919000000001e-09, 3.172129999999999e-09, 3.54446e-09]),
            (0, 0, 0, 0, 0, 1): np.array(
                [1.8258510000000016e-31, 1.8431939999999987e-25, 1.502688999999997e-20, 2.9231279999999964e-19,
                 5.627849999999997e-17, 1.729667000000001e-16, 4.694110000000008e-16, 9.83987000000002e-16,
                 1.7447680000000066e-15, 2.7962400000000077e-15, 5.773420000000005e-15]),
            (0, 1, 0, 0, 0, 1): np.array(
                [1.999976000000002e-23, 6.506961999999998e-20, 5.5399810000000174e-17, 4.135886000000001e-16,
                 9.237149999999993e-15, 2.129670000000001e-14, 4.5936100000000044e-14, 8.234329999999998e-14,
                 1.3123240000000009e-13, 1.9334020000000025e-13, 3.573410000000009e-13]),
        }

        rates = []
        h2o_h2_isomer_datasets: t.List[t.Tuple[str, t.Dict]] = [
            ("o", h2o_h2_ortho_rates),
            ("p", h2o_h2_para_rates),
        ]

        for isomer_label, rate_dict in h2o_h2_isomer_datasets:
            for (v1u, v2u, v3u, v1l, v2l, v3l), rate_array in rate_dict.items():
                interpolated_rate = CollisionalRatesDatabase._interp_rate(
                    layer_temp, h2o_h2_t_list, rate_array
                )
                rates.append(RateTransition(
                    upper_key=(v1u, v2u, v3u, isomer_label),
                    lower_key=(v1l, v2l, v3l, isomer_label),
                    rate=interpolated_rate,
                    mol_depend="H2",
                ))

        # Data from P. F. Zittel & D. E. Masturzo (1998), doi:10.1063/1.456122
        # Temperature grid shared by both isomers (K)
        h2o_h2o_stretch_t_list = np.array([295.0, 410.0, 518.0, 648.0, 772.0, 922.0, 1020.0])

        # --- H2O(nu_1/nu_3) + H2O ---
        # Table 1; assumed same data for nu_1 and nu_3, stated to relax together due to fast equilibrium.
        h2o_h2o_stretch_t_depend_rates: t.Dict[t.Tuple, npt.NDArray[np.float64]] = {
            (1, 0, 0, 0, 0, 0): np.array([3.0e-11, 2.4e-11, 2.4e-11, 2.4e-11, 2.5e-11, 2.4e-11, 2.1e-11]),
            (0, 0, 1, 0, 0, 0): np.array([3.0e-11, 2.4e-11, 2.4e-11, 2.4e-11, 2.5e-11, 2.4e-11, 2.1e-11]),
        }

        h2o_h2o_bend_t_list = np.array([295.0, 514.0, 730.0, 947.0])

        # --- H2O(nu_2) + H2O ---
        # D. L. Huestis (2006), 10.1021/jp054889n, also quotes the specific reaction (i.e.: state specific for collider)
        # (0,1,0)+(0,0,0)->(0,0,0)+(0,0,0) as k(T=300K)= 5.1e-11 and k(200K)=5.0e-11.
        h2o_h2o_bend_t_depend_rates: t.Dict[t.Tuple, npt.NDArray[np.float64]] = {
            (0, 1, 0, 0, 0, 0): np.array([5.4e-11, 5.0e-11, 5.4e-11, 6.0e-11]),  # Table 3
            (0, 2, 0, 0, 0, 0): np.array([12.1e-11, 10.9e-11, 10.9e-11, 11.6e-11]),  # Table 2
        }
        # nu2 relaxation probability is known to increase by factor ~2 at 2500 K, see Zittel & Masturzo discussion.

        # --- H2O(nu_1/nu_3) + H2O ---
        # Table 1; assumed same data for nu_1 and nu_3, stated to relax together due to fast equilibrium.
        h2o_he_stretch_t_list = np.array([295.0, 410.0, 518.0, 648.0, 770.0, 924.0])

        # Rates are given as upper bounds - scale by some factor?
        h2o_he_scale_factor = 0.5
        h2o_he_stretch_t_depend_rates: t.Dict[t.Tuple, npt.NDArray[np.float64]] = {
            (1, 0, 0, 0, 0, 0): np.array([0.4e-12, 0.6e-12, 0.7e-12, 0.8e-12, 1.0e-12, 1.2e-12]) * h2o_he_scale_factor,
            (0, 0, 1, 0, 0, 0): np.array([0.4e-12, 0.6e-12, 0.7e-12, 0.8e-12, 1.0e-12, 1.2e-12]) * h2o_he_scale_factor,
        }
        h2o_he_bend_t_list = np.array([295.0, 514.0, 730.0, 947.0])
        # Assumed to be roughly equal to Ar collisions (same for stretch data in Table 1).
        h2o_he_bend_t_depend_rates: t.Dict[t.Tuple, npt.NDArray[np.float64]] = {
            (0, 1, 0, 0, 0, 0): np.array([1.2e-12, 0.9e-12, 1.1e-12, 2.9e-12]) * h2o_he_scale_factor,  # Table 3
            (0, 2, 0, 0, 0, 0): np.array([0.6e-12, 1.3e-12, 3.4e-12, 5.1e-12]),  # Table 2
        }

        # Data from P. W. Barnes et al. (1999), doi:10.1039/A902348H
        # All taken at 295K.
        # H2O + H2O, Table 1 column 3.
        # Mixed, assume can't distinguish |12> as (1,0,2) or (2,0,1) so use both.
        # h2o_h2o_rates: t.Dict[t.Tuple, float] = {
        #     (2, 2, 0, 0, 0, 0): 1.5e-10,
        #     (0, 2, 2, 0, 0, 0): 1.5e-10,
        #     (3, 0, 0, 0, 0, 0): 3.7e-10,
        #     (0, 0, 3, 0, 0, 0): 3.7e-10,
        #     (4, 0, 0, 0, 0, 0): 2.0e-10,
        #     (0, 0, 4, 0, 0, 0): 2.0e-10,
        #     (1, 0, 2, 0, 0, 0): 1.7e-10,
        #     (2, 0, 1, 0, 0, 0): 1.7e-10,
        #     (1, 0, 3, 0, 0, 0): 1.7e-10,
        #     (3, 0, 1, 0, 0, 0): 1.7e-10,
        # }
        # H2O + H, Table 1 column 2.
        lower_bound_rate = 1e-12
        h2o_h_rates: t.Dict[t.Tuple, float] = {
            (2, 2, 0, 0, 0, 0): 0.8e-10,
            (0, 2, 2, 0, 0, 0): 0.8e-10,
            (3, 0, 0, 0, 0, 0): 3.0e-10,
            (0, 0, 3, 0, 0, 0): 3.0e-10,
            (4, 0, 0, 0, 0, 0): 4.3e-10,
            (0, 0, 4, 0, 0, 0): 4.3e-10,
            (1, 0, 2, 0, 0, 0): 0.48e-10,
            (2, 0, 1, 0, 0, 0): 0.48e-10,
            (1, 0, 3, 0, 0, 0): 3.0e-10,
            (3, 0, 1, 0, 0, 0): 3.0e-10,
            # Spoof, lower-bound estimates for lower missing states.
            (0, 1, 0, 0, 0, 0): lower_bound_rate,
            (0, 2, 0, 0, 0, 0): lower_bound_rate,
            (1, 0, 0, 0, 0, 0): lower_bound_rate,
            (0, 0, 1, 0, 0, 0): lower_bound_rate,
            (0, 3, 0, 0, 0, 0): lower_bound_rate,
            (1, 1, 0, 0, 0, 0): lower_bound_rate,
            (0, 1, 1, 0, 0, 0): lower_bound_rate,
            (0, 4, 0, 0, 0, 0): lower_bound_rate,
            (1, 2, 0, 0, 0, 0): lower_bound_rate,
            (0, 2, 1, 0, 0, 0): lower_bound_rate,
            (2, 0, 0, 0, 0, 0): lower_bound_rate,
            (0, 0, 2, 0, 0, 0): lower_bound_rate,
            (0, 5, 0, 0, 0, 0): lower_bound_rate,
            (1, 3, 0, 0, 0, 0): lower_bound_rate,
            (0, 3, 1, 0, 0, 0): lower_bound_rate,
            (2, 1, 0, 0, 0, 0): lower_bound_rate,
            (0, 1, 2, 0, 0, 0): lower_bound_rate,
            (0, 6, 0, 0, 0, 0): lower_bound_rate,
            (1, 4, 0, 0, 0, 0): lower_bound_rate,
            (0, 4, 1, 0, 0, 0): lower_bound_rate,
            (0, 7, 0, 0, 0, 0): lower_bound_rate,
            # (2,2,0),(0,2,2) is next, get into actual data.
        }

        for isomer_label in ("o", "p"):
            for (v1u, v2u, v3u, v1l, v2l, v3l), rate_array in h2o_h2o_stretch_t_depend_rates.items():
                interpolated_rate = CollisionalRatesDatabase._interp_rate(
                    layer_temp, h2o_h2o_stretch_t_list, rate_array
                )
                rates.append(RateTransition(
                    upper_key=(v1u, v2u, v3u, isomer_label),
                    lower_key=(v1l, v2l, v3l, isomer_label),
                    rate=interpolated_rate,
                    mol_depend="H2O",
                ))

            for (v1u, v2u, v3u, v1l, v2l, v3l), rate_array in h2o_h2o_bend_t_depend_rates.items():
                interpolated_rate = CollisionalRatesDatabase._interp_rate(
                    layer_temp, h2o_h2o_bend_t_list, rate_array
                )
                rates.append(RateTransition(
                    upper_key=(v1u, v2u, v3u, isomer_label),
                    lower_key=(v1l, v2l, v3l, isomer_label),
                    rate=interpolated_rate,
                    mol_depend="H2O",
                ))

            for (v1u, v2u, v3u, v1l, v2l, v3l), rate_array in h2o_he_stretch_t_depend_rates.items():
                interpolated_rate = CollisionalRatesDatabase._interp_rate(
                    layer_temp, h2o_he_stretch_t_list, rate_array
                )
                rates.append(RateTransition(
                    upper_key=(v1u, v2u, v3u, isomer_label),
                    lower_key=(v1l, v2l, v3l, isomer_label),
                    rate=interpolated_rate,
                    mol_depend="He",
                ))

            for (v1u, v2u, v3u, v1l, v2l, v3l), rate_array in h2o_he_bend_t_depend_rates.items():
                interpolated_rate = CollisionalRatesDatabase._interp_rate(
                    layer_temp, h2o_he_bend_t_list, rate_array
                )
                rates.append(RateTransition(
                    upper_key=(v1u, v2u, v3u, isomer_label),
                    lower_key=(v1l, v2l, v3l, isomer_label),
                    rate=interpolated_rate,
                    mol_depend="He",
                ))

            for (v1u, v2u, v3u, v1l, v2l, v3l), rate in h2o_h_rates.items():
                rates.append(RateTransition(
                    upper_key=(v1u, v2u, v3u, isomer_label),
                    lower_key=(v1l, v2l, v3l, isomer_label),
                    rate=rate,
                    mol_depend="H",
                ))

        return rates
