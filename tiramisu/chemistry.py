from molmass import Formula
import typing as t
import numpy.typing as npt
import numpy as np
from dataclasses import dataclass
from astropy import units as u

SpeciesIdentType = t.Union[str, "SpeciesFormula"]


class SpeciesFormula(Formula):
    """Represents a particular species."""

    def __init__(self, formula: SpeciesIdentType, *args, **kwargs):
        if isinstance(formula, SpeciesFormula):
            formula = formula.formula

        super().__init__(formula, *args, **kwargs)
        # Composition is fixed once constructed (formula string never changes after __init__), so cache it here -
        # __hash__/__eq__ are called on every dict/set operation involving a species (ChemicalProfile lookups,
        # XSecCollection keys, etc.), and recomputing composition().asdict() from scratch each time was showing up
        # as real overhead on those hot paths.
        self._composition_frozenset = frozenset(self.composition().asdict().values())
        self._cached_hash = hash(self._composition_frozenset)

    @property
    def reduced_mass(self) -> float:
        """
        Product of the atomic masses over their sum [Da]: the reduced mass for a diatomic. Each element is taken as its
        most abundant isotope unless the formula names one (molmass notation, e.g. "[13C]O").

        Not a reduced mass for polyatomics: it is only used for the continuum (dissociation) box broadening, which is
        not implemented for polyatomics yet.
        """
        mass_sum, mass_prod = 0.0, 1.0
        for symbol, (count, _, _) in self.composition().asdict().items():
            # Composition keys are element symbols, or mass-number-prefixed isotopes ("13C") where the formula names one.
            mass = Formula(f"[{symbol}]" if symbol[0].isdigit() else symbol).isotope.mass
            mass_sum += mass * count
            mass_prod *= mass ** count
        return mass_prod / mass_sum

    def __hash__(self) -> int:
        """Hash function. Necessary for sets and dicts."""
        return self._cached_hash

    def __eq__(self, other):
        """Equality check. Necessary for sets and dicts."""
        if isinstance(other, str):
            other = SpeciesFormula(other)

        comp_other = getattr(other, "_composition_frozenset", None)
        if comp_other is None:
            comp_other = frozenset(other.composition().asdict().values())

        return self._composition_frozenset == comp_other


def build_chemical_profile(
    species_definition: t.Dict[SpeciesIdentType, float],
    fill_species: t.List[t.Union[SpeciesFormula, str]],
    fill_ratios: t.List[float],
    nlayers: int,
) -> t.Tuple[t.List[SpeciesFormula], npt.NDArray[np.float64]]:
    """Builds a chemical profile from a species definition.

    Args:
        species_definition: A dictionary of species and their abundances.
        fill_species: A list of species to fill gaps in the atmosphere.
        fill_ratios: A list of ratios to fill gaps in the atmosphere with.
        nlayers: The number of layers in the profile.
    """
    species_definition = {SpeciesFormula(k): v for k, v in species_definition.items()}

    fill_species = [SpeciesFormula(k) for k in fill_species]

    species = list(species_definition.keys())
    abundances = list(species_definition.values())

    num_species = len(species)

    vmr = np.empty((num_species, nlayers))

    # Fill in the abundances
    for i, abund in enumerate(abundances):
        vmr[i, ...] = abund

    total_vmr_layer = np.sum(vmr, axis=0)
    leftovers = 1 - total_vmr_layer

    normed_ratios = np.array(fill_ratios) / np.sum(fill_ratios)
    fill_vmr = np.outer(normed_ratios, leftovers)

    total_vmr = np.concatenate((fill_vmr, vmr), axis=0)

    return fill_species + species, total_vmr


@dataclass
class ChemicalProfile:
    species: t.List[SpeciesFormula]
    vmr: npt.NDArray[np.float64]

    @classmethod
    def from_species_definition(
        cls,
        species_definition: t.Dict[SpeciesIdentType, float | npt.NDArray[np.float64]],
        fill_species: t.List[SpeciesFormula | str],
        fill_ratios: t.List[float],
        nlayers: int,
    ) -> "ChemicalProfile":
        species, vmr = build_chemical_profile(species_definition, fill_species, fill_ratios, nlayers)

        return cls(species, vmr)

    @property
    def masses(self):
        """Calculates the masses of each species."""
        return np.array([species.mass for species in self.species]) << u.u

    @property
    def mean_molecular_weight(self) -> u.Quantity:
        """Calculates the mean molecular weight of the profile."""
        return (self.masses[:, None] * self.vmr).sum(axis=0)

    def __getitem__(self, key: SpeciesFormula):
        """Returns the volume mixing ratio of a species."""
        if isinstance(key, str):
            key = SpeciesFormula(key)

        if not isinstance(key, SpeciesFormula):
            raise TypeError("Key must be a SpeciesFormula or string.")

        idx = self.species.index(key)

        return self.vmr[idx, ...]
