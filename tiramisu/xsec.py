import multiprocessing
import time
import abc
import pathlib
import pickle
import typing as t
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import permutations

# from scipy.linalg import solve, lu_factor, lu_solve, pinv
from scipy.optimize import lsq_linear, minimize, least_squares
from scipy.integrate import simpson

# from sklearn.linear_model import Ridge, RidgeCV

import matplotlib.pyplot as plt
import numpy.typing as npt
import numpy as np

import astropy.units as u
import astropy.constants as ac
import pandas as pd

from .chemistry import SpeciesFormula, SpeciesIdentType, ChemicalProfile
from .nlte import (
    boltzmann_population,
    calc_einstein_b_fi,
    calc_einstein_b_if,
    abs_emi_xsec,
    calc_band_profile,
    BandProfileCollection,
    pad_or_trim_profile,
    calc_imi,
    calc_lambda_approx,
    calc_lambda_approx_source,
    blackbody,
    _DEFAULT_CHUNK_SIZE,
    _DEFAULT_NUM_THREADS,
    bezier_coefficients,
    continuum_xsec,
    effective_source_tau_mu,
)
from .config import log, output_dir


# def blackbody(spectral_grid: u.Quantity, temperature: u.Quantity) -> u.Quantity:
#     """Blackbody source function.
#
#     Args:
#         spectral_grid: Wavenumber grid
#         temperature: Temperature
#
#     Returns:
#         u.Quantity: Blackbody source function, with units u.J / u.m**2.
#
#     """
#     temperature = np.atleast_1d(temperature)[:, None]
#
#     # wavelength = spectral_grid.to(u.m, equivalencies=u.spectral())[None, :]
#     # result = (
#     #         ((2 * ac.h * ac.c ** 2) / (wavelength ** 5))
#     #         * (1 / (np.exp((ac.h * ac.c) / (wavelength * ac.k_B * temperature)) - 1))
#     #         / u.sr
#     # )
#     freq = spectral_grid.to(u.Hz, equivalencies=u.spectral())[None, :]
#     result = (
#             ((2 * ac.h * freq ** 3) / (ac.c ** 2))
#             * (1 / (np.exp((ac.h * freq) / (ac.k_B * temperature)) - 1))
#             / u.sr
#     )
#     # result = (
#     #         (2 * ac.h * ac.c * spectral_grid ** 3)
#     #         * (1 / (np.exp((ac.h * ac.c * spectral_grid) / (ac.k_B * temperature)) - 1))
#     # )  # J * m / cm**3
#
#     return result


def bilinear_interpolate(
    data: npt.NDArray[np.float64],
    x: t.Union[npt.NDArray[np.float64], float],
    y: t.Union[npt.NDArray[np.float64], float],
    x_coord: npt.NDArray[np.float64],
    y_coord: npt.NDArray[np.float64],
    axes: t.Tuple[int, int] = (0, 1),
    mode: t.Literal["zero", "hold"] = "hold",
) -> npt.NDArray[np.float64]:
    """Bilinear interpolation.

    Compatible with any numpy-like array

    Args:
        x: x values to interpolate
        y: y values to interpolate
        x_coord: x coordinates of data
        y_coord: y coordinates of data
        data: data to interpolate
        axes: axes to interpolate over
        mode: mode to use for extrapolation

    Returns:
        npt.NDArray[np.float64]: interpolated data

    Raises:
        ValueError: If data is not at least 2D

    """
    if data.ndim < 2:
        raise ValueError("Data must be at least 2D")

    min_x, max_x = x_coord.min(), x_coord.max()
    min_y, max_y = y_coord.min(), y_coord.max()

    x_ravel = x.ravel()
    y_ravel = y.ravel()

    idx_x1 = x_coord.searchsorted(x_ravel, side="right")
    idx_y1 = y_coord.searchsorted(y_ravel, side="right")
    idx_x1 = idx_x1.clip(1, len(x_coord) - 1)
    idx_y1 = idx_y1.clip(1, len(y_coord) - 1)
    idx_x0 = idx_x1 - 1
    idx_y0 = idx_y1 - 1

    x_ravel = x_ravel.clip(min_x, max_x)
    y_ravel = y_ravel.clip(min_y, max_y)

    # ia = data.take(idx_x0, axis=axes[0]).take(idx_y0, axis=axes[1])
    # ib = data.take(idx_x1, axis=axes[0]).take(idx_y0, axis=axes[1])
    # ic = data.take(idx_x0, axis=axes[0]).take(idx_y1, axis=axes[1])
    # id = data.take(idx_x1, axis=axes[0]).take(idx_y1, axis=axes[1])

    if axes[0] != 0:
        data = data.swapaxes(axes[0], 0)
    if axes[1] != 1:
        data = data.swapaxes(axes[1], 1)

    # print(data.shape, idx_x0, idx_y0)
    ia = data[idx_x0, idx_y0]
    ib = data[idx_x1, idx_y0]
    ic = data[idx_x0, idx_y1]
    id = data[idx_x1, idx_y1]

    x1 = x_coord[idx_x1]
    x0 = x_coord[idx_x0]
    y1 = y_coord[idx_y1]
    y0 = y_coord[idx_y0]

    factor = (x1 - x0) * (y1 - y0)

    wa = (x1 - x_ravel) * (y1 - y_ravel)
    wb = (x_ravel - x0) * (y1 - y_ravel)
    wc = (x1 - x_ravel) * (y_ravel - y0)
    wd = (x_ravel - x0) * (y_ravel - y0)
    diff = 0

    if wa.ndim != ia.ndim:
        # Add appropriate dimensions to end
        diff = ia.ndim - wa.ndim
        wa = wa.reshape(*wa.shape, *[1] * diff)
        wb = wb.reshape(*wb.shape, *[1] * diff)
        wc = wc.reshape(*wc.shape, *[1] * diff)
        wd = wd.reshape(*wd.shape, *[1] * diff)

        factor = factor.reshape(*factor.shape, *[1] * diff)

    result = (wa * ia + wb * ib + wc * ic + wd * id) / factor

    return result.reshape(*x.shape, *data.shape[2:])


def weight_broadening_parameters(
    broadening_dict: t.Dict, chemistry_profile
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    broad_n = []
    broad_gamma = []
    for species_idx, species in enumerate(broadening_dict.keys()):
        if species in chemistry_profile.species:
            species_vmr = chemistry_profile[species]
            species_broad = broadening_dict[species]
            broad_gamma.append(species_broad[0] * species_vmr)
            broad_n.append(species_broad[1])
        else:
            log.warning(f"Broadening parameters for {species} provided but not present in chemistry_profile.")
    return np.array(broad_gamma), np.array(broad_n)


class XSecData(abc.ABC):

    @abc.abstractmethod
    def opacity(
        self,
        temperature: u.Quantity,
        pressure: u.Quantity,
    ) -> u.Quantity:
        """Calculates the opacity at a given temperature and pressure."""
        pass


class InterpolatingXSecData(XSecData):

    def __init__(
        self,
        species: t.Union[str, SpeciesFormula],
        spectral_grid: u.Quantity,
        xsec_grid: u.Quantity,
        temperature_grid: u.Quantity,
        pressure_grid: u.Quantity,
        axes: t.Tuple[int, int],
    ) -> None:
        self.species = SpeciesFormula(species)
        self.spectral_grid = spectral_grid
        self.xsec_grid = xsec_grid
        self.temperature_grid = temperature_grid
        self.pressure_grid = pressure_grid
        self.axes = axes

    def _interpolate_tp(self, temperature: u.Quantity, pressure: u.Quantity) -> u.Quantity:
        """Interpolates the cross section data to a given temperature and pressure."""
        # return bilinear_interpolate(
        #     self.xsec_grid,
        #     temperature,
        #     pressure,
        #     self.temperature_grid,
        #     self.pressure_grid,
        #     axes=self.axes,
        # )  # ORIGINAL
        return bilinear_interpolate(
            self.xsec_grid,
            pressure,
            temperature,
            self.pressure_grid,
            self.temperature_grid,
            axes=self.axes,
        )

    def opacity(
        self,
        temperature: u.Quantity,
        pressure: u.Quantity,
        spectral_grid: t.Optional[u.Quantity] = None,
    ) -> u.Quantity:
        """Calculates the opacity at a given temperature and pressure."""
        from scipy.interpolate import interp1d

        interped_spectra = self._interpolate_tp(temperature, pressure)
        if spectral_grid is not None:
            if np.array_equal(spectral_grid.value, self.spectral_grid.value):
                return interped_spectra
            spectral_grid = spectral_grid.to(self.spectral_grid.unit, equivalencies=u.spectral())
            spl = interp1d(
                self.spectral_grid.value,
                interped_spectra.value,
                axis=-1,
                copy=False,
                bounds_error=False,
                fill_value=0.0,
                assume_sorted=True,
            )
            interped_spectra = spl(spectral_grid.value)

            if hasattr(self.xsec_grid, "unit"):
                interped_spectra = interped_spectra << self.xsec_grid.unit
        return interped_spectra


class ExomolHDF5Xsec(InterpolatingXSecData):

    @classmethod
    def discover_all(
        cls, directory: pathlib.Path, load_in_memory: t.Optional[bool] = False
    ) -> t.List["ExomolHDF5Xsec"]:
        """Discover all HDF5 files in a directory."""
        files = (
            p.resolve()
            for p in pathlib.Path(directory).glob("**/*")
            if p.suffix.lower() in {".hdf5", ".h5"} and p.is_file()
        )

        return [cls(f, load_in_memory=load_in_memory) for f in files]

    def __init__(self, filepath: pathlib.Path, load_in_memory: t.Optional[bool] = True) -> None:
        """Use H5 format

        Args:
            filepath: Path to HDF5 file
            load_in_memory: Whether opacities are loaded on the spot or in memory

        """
        import h5py

        self.load_in_memory = load_in_memory

        filepath = pathlib.Path(filepath)
        self.filepath = filepath
        with h5py.File(filepath, "r") as f:
            species = f["mol_name"][0].decode("utf-8")
            formula = SpeciesFormula(species)
            spectral_grid = f["bin_edges"][()] << u.k
            temp_grid = f["t"][()] << u.K
            pressure_grid = f["p"][()] << u.bar
            xsec_grid = None
            if self.load_in_memory:
                xsec_grid = f["xsecarr"][()] << u.cm**2

        super().__init__(formula, spectral_grid, xsec_grid, temp_grid, pressure_grid, (1, 0))

    def opacity(
        self,
        temperature: u.Quantity,
        pressure: u.Quantity,
        spectral_grid: t.Optional[u.Quantity] = None,
    ) -> u.Quantity:
        import h5py

        if self.load_in_memory:
            return super().opacity(temperature, pressure, spectral_grid)
        else:
            with h5py.File(self.filepath, "r") as f:
                self.xsec_grid = f["xsecarr"]
                return super().opacity(temperature, pressure, spectral_grid) << u.cm**2


class ExomolNLTEXsec(ExomolHDF5Xsec):

    def __init__(
        self,
        species: str | SpeciesFormula,
        species_mass: float,
        states_file: pathlib.Path,
        trans_files: pathlib.Path | t.List[pathlib.Path],
        agg_col_nums: t.List[int],
        planet_radius: u.Quantity,
        # vmr: npt.NDArray[np.float64],
        chem_profile: ChemicalProfile,
        broadening_params: t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] = None,
        intensity_threshold: np.float64 = 1e-35,
        n_lte_layers: int = 0,
        # source_func_threshold: float = None,
        lte_grid_file: pathlib.Path = None,
        cont_states_file: pathlib.Path = None,
        cont_trans_files: pathlib.Path | t.List[pathlib.Path] = None,
        dissociation_products: t.Tuple = None,
        incident_radiation_field: u.Quantity = None,
        sor: bool = True,
        load_in_memory: bool = True,
        debug: bool = False,
        debug_pop_matrix: npt.NDArray[np.float64] = None,
    ) -> None:
        self.species = SpeciesFormula(species)
        self.species_mass = species_mass
        if type(states_file) is str:
            states_file = pathlib.Path(states_file)
        self.states_file = states_file
        if type(trans_files) is str:
            trans_files = [pathlib.Path(trans_files)]
        elif type(trans_files) is not list:
            trans_files = [trans_files]
        self.trans_files = trans_files
        self.agg_col_nums = agg_col_nums
        self.agg_col_names = ["agg" + str(idx + 1) for idx in range(0, len(self.agg_col_nums))]
        self.planet_radius = planet_radius
        self.chem_profile = chem_profile
        self.broadening_params = broadening_params
        self.intensity_threshold = intensity_threshold
        # self.source_func_threshold = source_func_threshold
        self.n_agg_states = None
        self.states = None
        self.agg_states = None
        self.density_profile = None
        self.dz_profile = None
        self.tau_matrix = None
        self.rates_grid = None
        self.abs_profile_grid = None
        self.emi_profile_grid = None
        self.mol_source_func_matrix = None
        self.global_source_func_matrix = None
        self.mol_chi_matrix = None
        self.mol_eta_matrix = None
        self.global_chi_matrix = None  # density and VMR weighted sum of all cross sections
        self.global_eta_matrix = None
        self.intensity_matrix = None
        self.pop_matrix = None
        self.is_converged = False
        self.n_lte_layers = n_lte_layers
        self.lte_grid_file = lte_grid_file
        self.cont_rates = None
        if type(cont_states_file) is str:
            cont_states_file = pathlib.Path(cont_states_file)
        self.cont_states_file = cont_states_file
        self.cont_states = None
        if type(cont_trans_files) is str:
            cont_trans_files = [pathlib.Path(cont_trans_files)]
        elif type(cont_trans_files) is not list:
            cont_trans_files = [cont_trans_files]
        self.cont_trans_files = cont_trans_files
        self.cont_profile_grid = None
        self.dissociation_products = dissociation_products
        self.incident_radiation_field = incident_radiation_field
        self.sor = sor
        self.sor_enabled = False
        self.full_prec = True
        self.damping_enabled = False
        self.load_in_memory = load_in_memory
        self.n_iter = 0
        self.debug = debug
        self.negative_source_func_cap = None
        self.negative_absorption_factor = 0.1
        self.debug_pop_matrix = debug_pop_matrix

        super().__init__(self.lte_grid_file, self.load_in_memory)

    def opacity(
        self,
        temperature: u.Quantity,
        pressure: u.Quantity,
        spectral_grid: u.Quantity | None = None,
    ) -> u.Quantity:
        """

        :param temperature:   The temperature profile of the model.
        :param pressure:      The pressure profile of the model.
        :param spectral_grid: The wavenumber grid.
        :return:
        """
        if self.n_lte_layers > 0 and self.lte_grid_file is None:
            raise RuntimeError(f"No LTE grid file specified while using {self.n_lte_layers} LTE layers.")
        if self.n_lte_layers > len(temperature):
            raise RuntimeError(
                f"Number of LTE layers specified ({self.n_lte_layers}) greater than number of layers in"
                f" atmosphere ({len(temperature)}).)"
            )
        if self.debug_pop_matrix is not None:
            # Use pop_matrix pickle from previous run for debug purposes, testing final intensity/emission calculations.
            log.info("Loading pre-computed pop_grid.")
            self.is_converged = True
            self.aggregate_states(temperature_profile=temperature, energy_cutoff=spectral_grid[-1].value)
            self.mol_chi_matrix = super().opacity(temperature, pressure, spectral_grid)
            self.mol_source_func_matrix = blackbody(spectral_grid=spectral_grid, temperature=temperature)
            self.mol_eta_matrix = self.mol_source_func_matrix * self.mol_chi_matrix * ac.c
            for layer_idx in range(self.n_lte_layers, len(temperature)):
                nlte_states = boltzmann_population(self.states.copy(), temperature[layer_idx])
                nlte_states["n_agg_nlte"] = self.debug_pop_matrix[-1, layer_idx, nlte_states["id_agg"]]
                nlte_states["n_nlte"] = np.where(
                    nlte_states["n_agg"] == 0,
                    0,
                    nlte_states["n"] * nlte_states["n_agg_nlte"] / nlte_states["n_agg"],
                )
                abs_xsec, emi_xsec = abs_emi_xsec(
                    states=nlte_states,
                    trans_files=self.trans_files,
                    temperature=temperature[layer_idx],
                    pressure=pressure[layer_idx],
                    species_mass=self.species_mass,
                    wn_grid=spectral_grid.value,
                    broad_n=(self.broadening_params[1] if self.broadening_params is not None else None),
                    broad_gamma=(
                        self.broadening_params[0][:, layer_idx] if self.broadening_params is not None else None
                    ),
                )
                if self.cont_states is not None:
                    nlte_cont_states = self.cont_states.merge(nlte_states[["id", "n_nlte"]], on="id", how="left")
                    cont_xsec = continuum_xsec(
                        continuum_states=nlte_cont_states,
                        continuum_trans_files=self.cont_trans_files,
                        temperature=temperature[layer_idx],
                        wn_grid=spectral_grid.value,
                    )
                    abs_xsec += cont_xsec
                source_func_logic = (emi_xsec == 0) | (abs_xsec == 0)
                nlte_source_func = np.zeros_like(abs_xsec)
                nlte_source_func[~source_func_logic] = emi_xsec[~source_func_logic] / abs_xsec[~source_func_logic]
                nlte_source_func = nlte_source_func << u.erg / (u.s * u.sr * u.cm)
                nlte_source_func = (nlte_source_func / ac.c).to(u.J / (u.sr * u.m**2), equivalencies=u.spectral())
                self.mol_source_func_matrix[layer_idx] = nlte_source_func
                abs_xsec = abs_xsec << u.cm**2
                self.mol_chi_matrix[layer_idx] = abs_xsec
                emi_xsec = emi_xsec << u.erg * u.cm / (u.s * u.sr)
                self.mol_eta_matrix[layer_idx] = emi_xsec
            return self.mol_chi_matrix

        if self.mol_source_func_matrix is None:
            self.mol_source_func_matrix = blackbody(
                spectral_grid=spectral_grid, temperature=temperature
            )  # / (spectral_grid ** 2)
            self.negative_source_func_cap = -self.mol_source_func_matrix.max()

        if self.states is None:
            # First pass: start with all layers in LTE.
            log.info(f"[I{self.n_iter}] Initial LTE set up for {self.species}.")

            if any([mol not in self.chem_profile.species for mol in self.dissociation_products]):
                warn_string = (
                    f"Specified dissociation products {self.dissociation_products} not present in"
                    f" chemical profile {self.chem_profile.species}."
                )
                log.warning(warn_string)
                # raise RuntimeError(warn_string)

            self.aggregate_states(temperature_profile=temperature, energy_cutoff=spectral_grid[-1].value)
            self.compute_rates_profiles(
                temperature_profile=temperature,
                pressure_profile=pressure,
                wn_grid=spectral_grid,
            )
            if self.cont_states_file is not None and self.cont_trans_files is not None:
                self.load_continuum_rates(temperature_profile=temperature, wn_grid=spectral_grid)

            self.mol_chi_matrix = super().opacity(temperature, pressure, spectral_grid)
            self.mol_eta_matrix = self.mol_source_func_matrix * self.mol_chi_matrix * ac.c
            return self.mol_chi_matrix
        elif not self.is_converged:
            self.n_iter += 1
            log.info(f"[I{self.n_iter}] Solving Non-LTE statistical equilibrium for {self.species}...")

            pop_grid = self.compute_pops_xsecs(
                temperature_profile=temperature,
                pressure_profile=pressure,
                wn_grid=spectral_grid,
            )
            self.pop_matrix = np.vstack(
                (self.pop_matrix, pop_grid.reshape((1, self.pop_matrix.shape[1], self.pop_matrix.shape[2])))
            )
            # TEMP!
            with open(
                (output_dir / f"KELT-20b_cont_boundaryL{self.n_lte_layers}_pop_matrix.pickle").resolve(), "wb"
            ) as pickle_file:
                pickle.dump(self.pop_matrix, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

            n_layers = pop_grid.shape[0]
            max_pop_changes = np.empty(n_layers - self.n_lte_layers)
            # importance = self.chem_profile[self.species] / self.chem_profile[self.species].max()

            for nlte_layer_idx in range(n_layers - self.n_lte_layers):
                # layer_old_pops = self.pop_grid[self.n_lte_layers + nlte_layer_idx]
                # layer_new_pops = pop_grid[self.n_lte_layers + nlte_layer_idx]
                layer_old_pops = self.pop_matrix[self.n_iter - 1, self.n_lte_layers + nlte_layer_idx]
                layer_new_pops = self.pop_matrix[self.n_iter, self.n_lte_layers + nlte_layer_idx]

                non_zero_idx_map = (layer_old_pops != 0) & (layer_new_pops != 0)
                layer_delta_pops = layer_new_pops[non_zero_idx_map] - layer_old_pops[non_zero_idx_map]
                layer_changes = np.abs(layer_delta_pops / layer_old_pops[non_zero_idx_map])
                max_pop_changes[nlte_layer_idx] = (
                    layer_changes.max()
                )  # * importance[self.n_lte_layers + nlte_layer_idx]

            # if self.n_iter >= 3:
            #     # Convergence rate checks
            #     convergence_rate = (self.pop_matrix[self.n_iter] - self.pop_matrix[self.n_iter - 1]) / (
            #         self.pop_matrix[self.n_iter - 1] - self.pop_matrix[self.n_iter - 2]
            #     )

            # Check for oscillations
            check_iter = 6
            if self.n_iter > check_iter:
                check_changes = np.where(
                    (self.pop_matrix[-check_iter:, self.n_lte_layers :, :] == 0)
                    & (self.pop_matrix[-check_iter - 1 : -1, self.n_lte_layers :, :] == 0),
                    0,
                    abs(
                        self.pop_matrix[-check_iter:, self.n_lte_layers :, :]
                        - self.pop_matrix[-check_iter - 1 : -1, self.n_lte_layers :, :]
                    )
                    / self.pop_matrix[-check_iter - 1 : -1, self.n_lte_layers :, :],
                ).max(axis=(1, 2))
                change_dif = np.diff(check_changes)
                oscillating = change_dif[:-1] * change_dif[1:]
                oscillating = oscillating < 0
                log.info(f"[I{self.n_iter}] Oscillating? {oscillating}")
                if sum(oscillating) >= 3:
                    log.info(
                        f"[I{self.n_iter}] Oscillations detected in {sum(oscillating)}/{check_iter} previous iterations - damping enabled."
                    )
                    self.damping_enabled = True

            # TODO: Flag when levels oscillate between 0 and extremely small values, blocking convergence, to fix them
            #  to 0?
            log.info(
                (
                    f"[I{self.n_iter}] Maximum population changes per layer = {max_pop_changes}"
                    f" (Max. = {max_pop_changes.max()})"
                )
            )
            if self.sor and not self.sor_enabled and self.n_iter >= 4 and max_pop_changes.max() <= 0.01:
                log.info(f"[I{self.n_iter}] SOR threshold reached - SOR enabled.")
                self.sor_enabled = True

            if self.damping_enabled:
                self.is_converged = max_pop_changes.max() < 0.005
            else:
                self.is_converged = max_pop_changes.max() < 0.001  # All less than 0.1% change # TEMP 1.0%!!!!

            # self.pop_grid = pop_grid

            if self.is_converged:
                log.info(f"[I{self.n_iter}] Convergence achieved!")
                log.info(
                    (
                        f"[I{self.n_iter}] Converged non-LTE populations from [L{self.n_lte_layers}] and above =\n"
                        f"{pop_grid[self.n_lte_layers:]}"
                    )
                )
                with open(
                    (output_dir / f"KELT-20b_cont_boundaryL{self.n_lte_layers}_pop_matrix.pickle").resolve(), "wb"
                ) as pickle_file:
                    pickle.dump(self.pop_matrix, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

            return self.mol_chi_matrix
        else:
            # Is converged - compute final run at higher resolution.
            self.mol_chi_matrix = super().opacity(temperature, pressure, spectral_grid)
            self.mol_source_func_matrix = blackbody(spectral_grid=spectral_grid, temperature=temperature)
            self.mol_eta_matrix = self.mol_source_func_matrix * self.mol_chi_matrix * ac.c
            log.info("Computing final Source Function.")
            for layer_idx in range(self.n_lte_layers, len(temperature)):
                nlte_states = boltzmann_population(self.states.copy(), temperature[layer_idx])
                nlte_states["n_agg_nlte"] = self.pop_matrix[-1, layer_idx, nlte_states["id_agg"]]
                nlte_states["n_nlte"] = np.where(
                    nlte_states["n_agg"] == 0,
                    0,
                    nlte_states["n"] * nlte_states["n_agg_nlte"] / nlte_states["n_agg"],
                )
                abs_xsec, emi_xsec = abs_emi_xsec(
                    states=nlte_states,
                    trans_files=self.trans_files,
                    temperature=temperature[layer_idx],
                    pressure=pressure[layer_idx],
                    species_mass=self.species_mass,
                    wn_grid=spectral_grid.value,
                    broad_n=(self.broadening_params[1] if self.broadening_params is not None else None),
                    broad_gamma=(
                        self.broadening_params[0][:, layer_idx] if self.broadening_params is not None else None
                    ),
                )
                if self.cont_states is not None:
                    nlte_cont_states = self.cont_states.merge(nlte_states[["id", "n_nlte"]], on="id", how="left")
                    cont_xsec = continuum_xsec(
                        continuum_states=nlte_cont_states,
                        continuum_trans_files=self.cont_trans_files,
                        temperature=temperature[layer_idx],
                        wn_grid=spectral_grid.value,
                    )
                    abs_xsec += cont_xsec
                    np.savetxt(
                        (
                            output_dir
                            / f"KELT-20b_nLTE_cont_L{layer_idx}_T{int(temperature[layer_idx].value)}_P{pressure[layer_idx].value:.4e}.txt"
                        ).resolve(),
                        np.array([spectral_grid.value, cont_xsec]).T,
                        fmt="%17.8E",
                    )
                    # LTE Comparison
                    lte_cont_states = self.cont_states.merge(nlte_states[["id", "n"]], on="id", how="left")
                    lte_cont_states["n_nlte"] = lte_cont_states["n"]
                    lte_cont_xsec = continuum_xsec(
                        continuum_states=lte_cont_states,
                        continuum_trans_files=self.cont_trans_files,
                        temperature=temperature[layer_idx],
                        wn_grid=spectral_grid.value,
                    )
                    np.savetxt(
                        (
                            output_dir
                            / f"KELT-20b_LTE_cont_L{layer_idx}_T{int(temperature[layer_idx].value)}_P{pressure[layer_idx].value:.4e}.txt"
                        ).resolve(),
                        np.array([spectral_grid.value, lte_cont_xsec]).T,
                        fmt="%17.8E",
                    )
                if layer_idx == len(temperature) - 1:
                    nlte_states["n_nlte"] = nlte_states["n"]
                    lte_abs_xsec, lte_emi_xsec = abs_emi_xsec(
                        states=nlte_states,
                        trans_files=self.trans_files,
                        temperature=temperature[layer_idx],
                        pressure=pressure[layer_idx],
                        species_mass=self.species_mass,
                        wn_grid=spectral_grid.value,
                        broad_n=(self.broadening_params[1] if self.broadening_params is not None else None),
                        broad_gamma=(
                            self.broadening_params[0][:, layer_idx] if self.broadening_params is not None else None
                        ),
                    )
                    np.savetxt(
                        (
                            output_dir
                            / f"KELT-20b_LTE_TOA_T{int(temperature[layer_idx].value)}_P{pressure[layer_idx].value:.4e}.txt"
                        ).resolve(),
                        np.array([spectral_grid.value, lte_abs_xsec]).T,
                        fmt="%17.8E",
                    )
                    np.savetxt(
                        (
                            output_dir
                            / f"KELT-20b_LTE_TOA_cont_T{int(temperature[layer_idx].value)}_P{pressure[layer_idx].value:.4e}.txt"
                        ).resolve(),
                        np.array([spectral_grid.value, lte_abs_xsec + lte_cont_xsec]).T,
                        fmt="%17.8E",
                    )
                    np.savetxt(
                        (
                            output_dir
                            / f"KELT-20b_nLTE_TOA_cont_T{int(temperature[layer_idx].value)}_P{pressure[layer_idx].value:.4e}.txt"
                        ).resolve(),
                        np.array([spectral_grid.value, abs_xsec]).T,
                        fmt="%17.8E",
                    )

                source_func_logic = (emi_xsec == 0) | (abs_xsec == 0)
                nlte_source_func = np.zeros_like(abs_xsec)
                nlte_source_func[~source_func_logic] = emi_xsec[~source_func_logic] / abs_xsec[~source_func_logic]
                # if self.source_func_threshold is not None:
                #     nlte_source_func[nlte_source_func < self.source_func_threshold] = 0.0
                nlte_source_func = nlte_source_func << u.erg / (u.s * u.sr * u.cm)
                nlte_source_func = (nlte_source_func / ac.c).to(u.J / (u.sr * u.m**2), equivalencies=u.spectral())

                # self.global_source_func_matrix[layer_idx] += (
                # self.global_source_func_matrix[layer_idx] += (
                #     nlte_source_func - self.mol_source_func_matrix[layer_idx]
                # ) * self.chem_profile[self.species][layer_idx]
                self.mol_source_func_matrix[layer_idx] = nlte_source_func

                abs_xsec = abs_xsec << u.cm**2
                # self.global_chi_matrix[layer_idx] += (
                #     (abs_xsec - self.mol_xsec_matrix[layer_idx])
                #     * self.chem_profile[self.species][layer_idx]  # Mol VMR
                #     * self.density_profile[layer_idx]
                # )
                self.mol_chi_matrix[layer_idx] = abs_xsec
                emi_xsec = emi_xsec << u.erg * u.cm / (u.s * u.sr)
                self.mol_eta_matrix[layer_idx] = emi_xsec

            with open(
                (output_dir / f"KELT-20b_cont_bL{self.n_lte_layers}_abs_xsec.pickle").resolve(), "wb"
            ) as abs_pickle_file:
                pickle.dump(self.mol_chi_matrix, abs_pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
            with open(
                (output_dir / f"KELT-20b_cont_bL{self.n_lte_layers}_emi_xsec.pickle").resolve(), "wb"
            ) as emi_pickle_file:
                pickle.dump(self.mol_eta_matrix, emi_pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

            return self.mol_chi_matrix

    def aggregate_states(self, temperature_profile: u.Quantity, energy_cutoff: float = None):
        """
        Sets self.states with a pandas DataFrame containing the ID, energy, degeneracy and lifetime columns of the
        statesfile, the columns on which state aggregation is performed and the corresponding aggregated state ID.

        :param temperature_profile: The temperature of each layer in Kelvin.
        """
        if self.agg_col_nums is None:
            # Assuming diatomic by default.
            self.agg_col_nums = [9, 10]

        self.states = pd.read_csv(
            self.states_file,
            sep=r"\s+",
            names=["id", "energy", "g", "tau"] + self.agg_col_names,
            usecols=[0, 1, 2, 5] + self.agg_col_nums,
        )
        # TODO: Drop states above grid cutoff?
        # if energy_cutoff is not None:
        #     self.states = self.states.loc[self.states["energy"] <= energy_cutoff]
        self.agg_states = self.states.groupby(by=self.agg_col_names, as_index=False).agg(energy_agg=("energy", "min"))
        self.n_agg_states = len(self.agg_states)

        self.agg_states = self.agg_states.sort_values(by="energy_agg")
        self.agg_states["id_agg"] = np.arange(0, self.n_agg_states, dtype=int)
        log.debug("Vibronically aggregated states (head): \n", self.agg_states.head(30))

        self.states = self.states.merge(
            self.agg_states[["id_agg", "energy_agg"] + self.agg_col_names],
            on=self.agg_col_names,
            how="left",
        )

        # self.pop_grid = np.zeros((temperature_profile.shape[0], self.n_agg_states))
        self.pop_matrix = np.zeros((1, temperature_profile.shape[0], self.n_agg_states))
        ac_h_c_on_kB = ac.h * ac.c.cgs / ac.k_B
        for layer_idx, layer_temperature in enumerate(temperature_profile):
            self.states["q_lev"] = self.states["g"] * np.exp(
                -ac_h_c_on_kB * (self.states["energy"] << 1 / u.cm) / layer_temperature
            )
            self.states["n"] = self.states["q_lev"] / self.states["q_lev"].sum()
            states_agg_n = self.states.groupby(by=["id_agg"], as_index=False).agg(n_agg=("n", "sum"))
            # self.pop_grid[layer_idx] = states_agg_n.sort_values(by="id_agg")["n_agg"]
            self.pop_matrix[0, layer_idx] = states_agg_n.sort_values(by="id_agg")["n_agg"]

        self.states = self.states[["id", "energy", "g", "tau", "id_agg"] + self.agg_col_names]

    def compute_rates_profiles(
        self,
        temperature_profile: u.Quantity,
        pressure_profile: u.Quantity,
        wn_grid: u.Quantity,
        num_threads: int = _DEFAULT_NUM_THREADS,
    ):
        """
        Sets self.rates_grid and self.abs_profile_grid based on the aggregated state IDs computed in
        :func:`~xsec.ExomolNLTEXsec.aggregate_states`.

        self.rates_grid is a pandas DataFrame containing the Einstein A and B rates between upper and lower aggregated
        state IDs.

        self.abs_profile_grid is a list of DataFrame objects for each non-LTE layer in the model, with each DataFrame
        containing a band profile between each upper and lower aggregated state ID. The band profile consists of a
        profile computed on part of the input wn_grid and a start_idx specifying where the band profile starts on that
        grid. This is done to save on memory as the total number of band profiles can be very large.

        :param temperature_profile:    The temperature for each layer of the model, in Kelvin.
        :param pressure_profile:       The pressure for each layer of the model, in Pascals.
        :param wn_grid:             The wavenumber grid over which the opacity is being calculated.
        :param num_threads:         The number of threads to use when parallelising the transition file processing.
        """

        def _aggregate_trans(
            chunk: pd.DataFrame,
            rates_lock: multiprocessing.Lock,
            rates_set_event: multiprocessing.Event,
            layer_t_p_tuple: t.Tuple[int, t.Tuple[u.Quantity, u.Quantity]],
        ):
            """
            Calculates the Einstein coefficient A_fi, B_fi, B_if and the vibronic band profiles of transitions in the
            chunk, to be used to construct the rates grid.

            :param chunk:
            :param layer_t_p_tuple:

            :return:
            """
            layer_idx, (temperature, pressure) = layer_t_p_tuple

            states_temp = boltzmann_population(self.states.copy(), temperature)
            states_temp["n_frac"] = states_temp["n"] / states_temp["n_agg"]

            chunk = chunk.merge(
                states_temp[["id", "id_agg", "energy", "g", "n_frac"]],
                left_on="id_i",
                right_on="id",
                how="inner",
            )
            # chunk = chunk.loc[chunk["n_frac"] > 0.0]  # Needed for rates
            chunk = chunk.rename(
                columns={
                    "id_agg": "id_i_agg",
                    "energy": "energy_i",
                    "g": "g_i",
                    "n_frac": "n_frac_i",
                }
            )
            chunk = chunk.drop(columns=["id_i", "id"])

            chunk = chunk.merge(
                states_temp[["id", "id_agg", "energy", "g", "tau", "n_frac"]],
                left_on="id_f",
                right_on="id",
                how="inner",
            )
            chunk = chunk.rename(
                columns={
                    "id_agg": "id_f_agg",
                    "energy": "energy_f",
                    "g": "g_f",
                    "tau": "tau_f",
                    "n_frac": "n_frac_f",
                }
            )
            chunk = chunk.drop(columns=["id_f", "id"])

            chunk["energy_fi"] = chunk["energy_f"] - chunk["energy_i"]
            chunk = chunk.loc[
                (chunk["energy_f"] >= wn_grid[0].value)
                & (chunk["energy_i"] <= wn_grid[-1].value)
                & (chunk["energy_f"] <= wn_grid[-1].value)  # TODO: TEST/REMOVE
                & (chunk["energy_fi"] >= wn_grid[0].value)
                & (chunk["energy_fi"] <= wn_grid[-1].value)
            ]
            # log.info(f"[L{layer_idx + self.n_lte_layers}] Number of trans in chunk = {len(chunk)}")
            # log.info(
            #     f"[L{layer_idx}] Number of trans with upper state above grid limit "
            #     f"= {len(chunk.loc[chunk["energy_f"] > wn_grid[-1].value])}"
            # )
            chunk = chunk.drop(columns=["energy_f", "energy_i"])

            start_time = time.perf_counter()
            # log.info(f"[L{layer_idx + self.n_lte_layers}] Starting profiles...")
            abs_emi_bands = chunk.groupby(by=["id_f_agg", "id_i_agg"]).apply(
                lambda x: calc_band_profile(
                    wn_grid=wn_grid.value,
                    n_frac_i=x["n_frac_i"].to_numpy(),
                    n_frac_f=x["n_frac_f"].to_numpy(),
                    a_fi=x["A_fi"].to_numpy(),
                    g_f=x["g_f"].to_numpy(),
                    g_i=x["g_i"].to_numpy(),
                    energy_fi=x["energy_fi"].to_numpy(),
                    lifetimes=x["tau_f"].to_numpy(),
                    temperature=temperature.value,
                    pressure=pressure.value,
                    broad_n=(self.broadening_params[1] if self.broadening_params is not None else None),
                    broad_gamma=(
                        self.broadening_params[0][:, layer_idx] if self.broadening_params is not None else None
                    ),
                    species_mass=self.species_mass,
                ),
                include_groups=False,
            )
            self.abs_profile_grid[layer_idx].append(BandProfileCollection(band_profiles=abs_emi_bands["abs"]))
            self.emi_profile_grid[layer_idx].append(BandProfileCollection(band_profiles=abs_emi_bands["emi"]))
            # self.abs_profile_grid[layer_idx].append(
            #     BandProfileCollection(
            #         band_profiles=chunk.groupby(by=["id_f_agg", "id_i_agg"]).apply(
            #             lambda x: calc_band_profile(
            #                 wn_grid=wn_grid.value,
            #                 n_frac_i=x["n_frac_i"].to_numpy(),
            #                 n_frac_f=x["n_frac_f"].to_numpy(),
            #                 a_fi=x["A_fi"].to_numpy(),
            #                 g_f=x["g_f"].to_numpy(),
            #                 g_i=x["g_i"].to_numpy(),
            #                 energy_fi=x["energy_fi"].to_numpy(),
            #                 lifetimes=x["tau_f"].to_numpy(),
            #                 temperature=temperature.value,
            #                 pressure=pressure.value,
            #                 broad_n=(self.broadening_params[1] if self.broadening_params is not None else None),
            #                 broad_gamma=(
            #                     self.broadening_params[0][:, layer_idx] if self.broadening_params is not None else None
            #                 ),
            #                 species_mass=self.species_mass,
            #             ),
            #             include_groups=False,
            #         )
            #     )
            # )
            log.info(f"[L{layer_idx + self.n_lte_layers}] Profile duration = {time.perf_counter() - start_time:.2f}.")
            rates_lock.acquire(blocking=True)
            try:
                if not rates_set_event.is_set():
                    rates_set_event.set()
                    # Remove rates between the same aggregate state.
                    chunk = chunk.loc[
                        (chunk["id_f_agg"] != chunk["id_i_agg"])
                        # & (chunk["energy_fi"] >= wn_grid[0].value)
                        # & (chunk["energy_fi"] <= wn_grid[-1].value)
                    ]
                    # chunk = chunk.loc[chunk["id_f_agg"] != chunk["id_i_agg"]]

                    chunk["B_fi"] = calc_einstein_b_fi(
                        a_fi=chunk["A_fi"].to_numpy() << 1 / u.s,
                        energy_fi=(chunk["energy_fi"].to_numpy() << 1 / u.cm).to(u.Hz, equivalencies=u.spectral()),
                    )
                    chunk["B_if"] = calc_einstein_b_if(
                        b_fi=chunk["B_fi"].to_numpy(),
                        g_f=chunk["g_f"].to_numpy(),
                        g_i=chunk["g_i"].to_numpy(),
                    )
                    self.rates_grid.append(
                        chunk.groupby(by=["id_f_agg", "id_i_agg"], as_index=False).agg(
                            A_fi=("A_fi", "sum"), B_fi=("B_fi", "sum"), B_if=("B_if", "sum")
                        )
                    )
            finally:
                rates_lock.release()

        rates_manager = multiprocessing.Manager()
        n_nlte_layers = temperature_profile.shape[0] - self.n_lte_layers
        self.rates_grid = []
        self.abs_profile_grid = [[] for _ in range(n_nlte_layers)]
        self.emi_profile_grid = [[] for _ in range(n_nlte_layers)]
        for trans_file in self.trans_files:
            trans_reader = pd.read_csv(
                trans_file,
                sep=r"\s+",
                names=["id_f", "id_i", "A_fi"],
                usecols=[0, 1, 2],
                chunksize=_DEFAULT_CHUNK_SIZE,
            )
            for trans_chunk in trans_reader:
                chunk_rates_lock = rates_manager.Lock()
                chunk_rates_set_event = rates_manager.Event()
                agg_partial = partial(_aggregate_trans, trans_chunk, chunk_rates_lock, chunk_rates_set_event)
                with ThreadPoolExecutor(max_workers=num_threads) as ex:
                    ex.map(
                        agg_partial,
                        list(
                            enumerate(
                                zip(
                                    temperature_profile[self.n_lte_layers :],
                                    pressure_profile[self.n_lte_layers :],
                                )
                            )
                        ),
                    )

        # Each layer_idx is a list of 1 or more BandProfileCollection objects to be merged.
        for layer_idx in range(len(self.abs_profile_grid)):
            if len(self.abs_profile_grid[layer_idx]) > 1:
                self.abs_profile_grid[layer_idx][0].merge_collections(
                    self.abs_profile_grid[layer_idx][1:],
                    normalise=True,
                    spectral_grid=wn_grid,
                )
                self.abs_profile_grid[layer_idx] = self.abs_profile_grid[layer_idx][0]
            else:
                self.abs_profile_grid[layer_idx] = self.abs_profile_grid[layer_idx][0]
                self.abs_profile_grid[layer_idx].normalise(spectral_grid=wn_grid)

            if len(self.emi_profile_grid[layer_idx]) > 1:
                self.emi_profile_grid[layer_idx][0].merge_collections(
                    self.emi_profile_grid[layer_idx][1:],
                    normalise=True,
                    spectral_grid=wn_grid,
                )
                self.emi_profile_grid[layer_idx] = self.emi_profile_grid[layer_idx][0]
            else:
                self.emi_profile_grid[layer_idx] = self.emi_profile_grid[layer_idx][0]
                self.emi_profile_grid[layer_idx].normalise(spectral_grid=wn_grid)

        self.rates_grid = pd.concat(self.rates_grid)
        self.rates_grid = self.rates_grid.groupby(by=["id_f_agg", "id_i_agg"], as_index=False).agg(
            A_fi=("A_fi", "sum"), B_fi=("B_fi", "sum"), B_if=("B_if", "sum")
        )

    def add_col_chem_rates(
        self, y_matrix: npt.NDArray[np.float64], layer_idx: int, layer_temp: u.Quantity
    ) -> npt.NDArray[np.float64]:
        # TODO: Move formation destruction rates to RHS in rate equation as they do not depend on species n!
        def add_col_chem_rate(
            estate_u: str,
            v_u: int,
            estate_l: str,
            v_l: int,
            rate: float,
            mol_depend: str,
        ):
            if mol_depend in self.chem_profile.species:
                upper_id = self.agg_states.loc[
                    (self.agg_states["agg1"] == estate_u) & (self.agg_states["agg2"] == v_u),
                    "id_agg",
                ].values[0]
                lower_id = self.agg_states.loc[
                    (self.agg_states["agg1"] == estate_l) & (self.agg_states["agg2"] == v_l),
                    "id_agg",
                ].values[0]
                # TODO: Add check that state is within energy bounds!
                # log.debbug(f"Upper = {upper_id}, type ={type(upper_id)}")
                # log.debug(f"Lower = {lower_id}, type ={type(lower_id)}")
                depend_num_dens = (
                    self.chem_profile[SpeciesFormula(mol_depend)][layer_idx] * self.density_profile[layer_idx]
                ).to(u.cm**-3)

                c_fi = (rate * u.cm**3 / u.s) * depend_num_dens
                upper_energy = self.agg_states.loc[
                    (self.agg_states["agg1"] == estate_u) & (self.agg_states["agg2"] == v_u),
                    "energy_agg",
                ].values[0]
                lower_energy = self.agg_states.loc[
                    (self.agg_states["agg1"] == estate_l) & (self.agg_states["agg2"] == v_l),
                    "energy_agg",
                ].values[0]
                if self.pop_matrix[-1, layer_idx, lower_id] == 0:
                    c_if = 0 * c_fi
                    log.warning(
                        (
                            f"[L{layer_idx}] upwards Col/Chem. rate for IDs {upper_id}-{lower_id} 0 from balance due "
                            f"to 0 lower state population."
                        )
                    )
                else:
                    # c_if = c_fi * (self.pop_matrix[-1, layer_idx, upper_id] / self.pop_matrix[-1, layer_idx, lower_id])
                    c_if = c_fi * np.exp(-(ac.h * (upper_energy - lower_energy) * u.k * ac.c) / (ac.k_B * layer_temp))
                # log.debug(f"C_fi = {c_fi}, C_if = {c_if}.")
                if self.debug:
                    log.info(f"[L{layer_idx}] C_fi({mol_depend}; {estate_u, v_u}->{estate_l, v_l}) = {c_fi}")
                    log.info(f"[L{layer_idx}] C_if({mol_depend}; {estate_l, v_l}->{estate_u, v_u}) = {c_if}")

                if lower_id == upper_id:
                    # Formation/destruction:
                    y_matrix[upper_id, upper_id] += c_fi
                else:
                    y_matrix[upper_id, lower_id] += c_if
                    y_matrix[lower_id, upper_id] += c_fi
                    y_matrix[upper_id, upper_id] -= c_fi
                    y_matrix[lower_id, lower_id] -= c_if

        # log.debug("Initial Y matrix = ", y_matrix)
        if self.species == "OH":
            # P. H. Paul (10.1021/j100021a004)
            # OH(A, v'') + O_2 -> OH(X, v'') + O_2
            # add_col_chem_rate("A2Sigma+", 0, "X2Pi", 0, 13.4e-11, "O2")  # @ 1900 K, 15.6 @ 2300 K
            # add_col_chem_rate("A2Sigma+", 1, "X2Pi", 0, 15.1e-11, "O2")  # @ 1900 K, 16.8 @ 2300 K
            # add_col_chem_rate("A2Sigma+", 1, "A2Sigma+", 0, 1.68e-11, "O2")  # @ 1900 K, 1.74 @ 2300 K
            # NB: OH(A, v''=0, 1) electronic quenching is not specified as to which lower state: is total quenching.

            # Adler-Golden (10.1029/97JA01622)
            # OH(X, v'') + O_2 -> OH(X, v'') + O_2
            p_v_list = [0.043, 0.083, 0.15, 0.23, 0.36, 0.50, 0.72, 0.75, 0.95]
            c_val = 4.4e-12
            for v_val in range(10):
                for dv_val in range(1, v_val + 1):
                    add_col_chem_rate(
                        "X2Pi",
                        v_val,
                        "X2Pi",
                        v_val - dv_val,
                        c_val * p_v_list[v_val - 1] ** dv_val,
                        "O2",
                    )

            # Varandas (0.1016/j.cplett.2004.08.023)
            # Destruction by O + OH -> O_2 + H
            # Table 3, k9 rates from method 2.
            # add_col_chem_rate("X2Pi", 0, "X2Pi", 0, -26.45e-12, "O")  # @ 255 K, 33.62 @ 210 K
            # add_col_chem_rate("X2Pi", 1, "X2Pi", 1, -23.59e-12, "O")  # @ 255 K, 27.20 @ 210 K
            # add_col_chem_rate("X2Pi", 2, "X2Pi", 2, -24.36e-12, "O")  # @ 255 K, 29.09 @ 210 K
            # add_col_chem_rate("X2Pi", 3, "X2Pi", 3, -29.31e-12, "O")  # @ 255 K, 32.16 @ 210 K
            # add_col_chem_rate("X2Pi", 4, "X2Pi", 4, -34.49e-12, "O")  # @ 255 K, 33.83 @ 210 K
            # add_col_chem_rate("X2Pi", 5, "X2Pi", 5, -30.95e-12, "O")  # @ 255 K, 32.99 @ 210 K
            # add_col_chem_rate("X2Pi", 6, "X2Pi", 6, -32.81e-12, "O")  # @ 255 K, 35.09 @ 210 K
            # add_col_chem_rate("X2Pi", 7, "X2Pi", 7, -38.76e-12, "O")  # @ 255 K, 42.18 @ 210 K
            # add_col_chem_rate("X2Pi", 8, "X2Pi", 8, -41.85e-12, "O")  # @ 255 K, 42.71 @ 210 K
            # add_col_chem_rate("X2Pi", 9, "X2Pi", 9, -47.78e-12, "O")  # @ 255 K, 50.69 @ 210 K

            # P. J. S. B. Caridade et al. (10.5194/acp-13-1-2013)
            # Destruction by O + OH -> O_2 + H
            # Table 1, R4 rates.
            add_col_chem_rate("X2Pi", 0, "X2Pi", 0, -26.0e-12, "O")  # Extrapolated
            add_col_chem_rate("X2Pi", 1, "X2Pi", 1, -21.1e-12, "O")  # @ 300K
            add_col_chem_rate("X2Pi", 2, "X2Pi", 2, -23.9e-12, "O")  # @ 300K
            add_col_chem_rate("X2Pi", 3, "X2Pi", 3, -28.4e-12, "O")  # @ 300K
            add_col_chem_rate("X2Pi", 4, "X2Pi", 4, -28.8e-12, "O")  # @ 300K
            add_col_chem_rate("X2Pi", 5, "X2Pi", 5, -31.7e-12, "O")  # @ 300K
            add_col_chem_rate("X2Pi", 6, "X2Pi", 6, -29.7e-12, "O")  # @ 300K
            add_col_chem_rate("X2Pi", 7, "X2Pi", 7, -34.9e-12, "O")  # @ 300K
            add_col_chem_rate("X2Pi", 8, "X2Pi", 8, -39.3e-12, "O")  # @ 300K
            add_col_chem_rate("X2Pi", 9, "X2Pi", 9, -43.4e-12, "O")  # @ 300K
            add_col_chem_rate("X2Pi", 10, "X2Pi", 10, -46.0e-12, "O")  # @ Extrapolated
            # add_col_chem_rate("X2Pi", 11, "X2Pi", 11, -50.0e-12, "O")  # @ Extrapolated

            # P. J. S. B. Caridade et al. (10.5194/acp-13-1-2013)
            # O + OH(v') -> OH(v'') + O ALL @ 300 K.
            # Table 1, R5 rates.
            add_col_chem_rate("X2Pi", 1, "X2Pi", 0, 19.2e-12, "O")
            add_col_chem_rate("X2Pi", 2, "X2Pi", 0, 14.2e-12, "O")
            add_col_chem_rate("X2Pi", 2, "X2Pi", 1, 10.5e-12, "O")
            add_col_chem_rate("X2Pi", 3, "X2Pi", 0, 9.4e-12, "O")
            add_col_chem_rate("X2Pi", 3, "X2Pi", 1, 9.6e-12, "O")
            add_col_chem_rate("X2Pi", 3, "X2Pi", 2, 8.1e-12, "O")
            add_col_chem_rate("X2Pi", 4, "X2Pi", 0, 6.4e-12, "O")
            add_col_chem_rate("X2Pi", 4, "X2Pi", 1, 7.8e-12, "O")
            add_col_chem_rate("X2Pi", 4, "X2Pi", 2, 6.9e-12, "O")
            add_col_chem_rate("X2Pi", 4, "X2Pi", 3, 4.8e-12, "O")
            add_col_chem_rate("X2Pi", 5, "X2Pi", 0, 6.3e-12, "O")
            add_col_chem_rate("X2Pi", 5, "X2Pi", 1, 4.7e-12, "O")
            add_col_chem_rate("X2Pi", 5, "X2Pi", 2, 6.0e-12, "O")
            add_col_chem_rate("X2Pi", 5, "X2Pi", 3, 3.8e-12, "O")
            add_col_chem_rate("X2Pi", 5, "X2Pi", 4, 3.8e-12, "O")
            add_col_chem_rate("X2Pi", 6, "X2Pi", 0, 4.6e-12, "O")
            add_col_chem_rate("X2Pi", 6, "X2Pi", 1, 4.4e-12, "O")
            add_col_chem_rate("X2Pi", 6, "X2Pi", 2, 5.0e-12, "O")
            add_col_chem_rate("X2Pi", 6, "X2Pi", 3, 4.7e-12, "O")
            add_col_chem_rate("X2Pi", 6, "X2Pi", 4, 4.1e-12, "O")
            add_col_chem_rate("X2Pi", 6, "X2Pi", 5, 4.5e-12, "O")
            add_col_chem_rate("X2Pi", 7, "X2Pi", 0, 3.4e-12, "O")
            add_col_chem_rate("X2Pi", 7, "X2Pi", 1, 3.1e-12, "O")
            add_col_chem_rate("X2Pi", 7, "X2Pi", 2, 3.6e-12, "O")
            add_col_chem_rate("X2Pi", 7, "X2Pi", 3, 3.3e-12, "O")
            add_col_chem_rate("X2Pi", 7, "X2Pi", 4, 3.5e-12, "O")
            add_col_chem_rate("X2Pi", 7, "X2Pi", 5, 3.1e-12, "O")
            add_col_chem_rate("X2Pi", 7, "X2Pi", 6, 4.0e-12, "O")
            add_col_chem_rate("X2Pi", 8, "X2Pi", 0, 2.4e-12, "O")
            add_col_chem_rate("X2Pi", 8, "X2Pi", 1, 2.3e-12, "O")
            add_col_chem_rate("X2Pi", 8, "X2Pi", 2, 2.4e-12, "O")
            add_col_chem_rate("X2Pi", 8, "X2Pi", 3, 2.4e-12, "O")
            add_col_chem_rate("X2Pi", 8, "X2Pi", 4, 2.1e-12, "O")
            add_col_chem_rate("X2Pi", 8, "X2Pi", 5, 2.7e-12, "O")
            add_col_chem_rate("X2Pi", 8, "X2Pi", 6, 3.0e-12, "O")
            add_col_chem_rate("X2Pi", 8, "X2Pi", 7, 4.2e-12, "O")
            add_col_chem_rate("X2Pi", 9, "X2Pi", 0, 1.2e-12, "O")
            add_col_chem_rate("X2Pi", 9, "X2Pi", 1, 1.3e-12, "O")
            add_col_chem_rate("X2Pi", 9, "X2Pi", 2, 2.1e-12, "O")
            add_col_chem_rate("X2Pi", 9, "X2Pi", 3, 1.8e-12, "O")
            add_col_chem_rate("X2Pi", 9, "X2Pi", 4, 2.0e-12, "O")
            add_col_chem_rate("X2Pi", 9, "X2Pi", 5, 1.7e-12, "O")
            add_col_chem_rate("X2Pi", 9, "X2Pi", 6, 1.8e-12, "O")
            add_col_chem_rate("X2Pi", 9, "X2Pi", 7, 2.1e-12, "O")
            add_col_chem_rate("X2Pi", 9, "X2Pi", 8, 3.3e-12, "O")

            ozone_formation_distribution = np.array([4, 0.5, 0.5, 1, 1, 2, 4, 19, 28, 38, 2])
            # percentage distribution of total production going to each v=idx.
            total_rate = 1.4e-10 * np.exp(-470 / layer_temp.value)
            # for v_val in range(len(ozone_formation_distribution)):
            for v_val in range(0, 10):  # TODO: Implementation to limit this!
                v_rate = total_rate * ozone_formation_distribution[v_val] / 100
                add_col_chem_rate("X2Pi", v_val, "X2Pi", v_val, v_rate, "O3")

            # Single-quantum vibrational quenching by He
            # N. Kohno et al. (2013) (10.1021/jp3114072)
            add_col_chem_rate("X2Pi", 1, "X2Pi", 0, 3.2e-17, "He")  # @ 298 K
            add_col_chem_rate("X2Pi", 2, "X2Pi", 1, 1.4e-16, "He")  # @ 298 K
            add_col_chem_rate("X2Pi", 3, "X2Pi", 2, 4.4e-16, "He")  # @ 298 K
            add_col_chem_rate("X2Pi", 4, "X2Pi", 3, 1.2e-15, "He")  # @ 298 K
            add_col_chem_rate("X2Pi", 5, "X2Pi", 4, 3.2e-15, "He")  # @ 298 K
            add_col_chem_rate("X2Pi", 6, "X2Pi", 5, 8.2e-15, "He")  # @ 298 K
            add_col_chem_rate("X2Pi", 7, "X2Pi", 6, 2.1e-14, "He")  # @ 298 K
            add_col_chem_rate("X2Pi", 8, "X2Pi", 7, 5.1e-14, "He")  # @ 298 K
            add_col_chem_rate("X2Pi", 9, "X2Pi", 8, 1.3e-13, "He")  # @ 298 K
            add_col_chem_rate("X2Pi", 10, "X2Pi", 9, 3.4e-13, "He")  # @ 298 K
            # add_col_chem_rate("X2Pi", 11, "X2Pi", 10, 9.5e-13, "He")  # @ 298 K
            # add_col_chem_rate("X2Pi", 12, "X2Pi", 11, 2.9e-12, "He")  # @ 298 K

            # Multi-quantum vibrational quenching by H
            # Atahan & Alexander (10.1021/jp055860m)
            add_col_chem_rate("X2Pi", 1, "X2Pi", 0, 1.600e-10, "H")  # @ 300 K
            add_col_chem_rate("X2Pi", 2, "X2Pi", 1, 0.654e-10, "H")  # @ 300 K
            add_col_chem_rate("X2Pi", 2, "X2Pi", 0, 1.043e-10, "H")  # @ 300 K
            # Fit to data (single-quantum assumption)
            # OLD
            conservative_h_data = True
            if conservative_h_data:
                # # add_col_chem_rate("X2Pi", 1, "X2Pi", 0, 1.6e-10, "H")  # @ 300 K single-quantum extrapolation
                # # add_col_chem_rate("X2Pi", 2, "X2Pi", 1, 1.7e-10, "H")  # @ 300 K single-quantum extrapolation
                add_col_chem_rate("X2Pi", 3, "X2Pi", 2, 1.8e-10, "H")  # @ 300 K single-quantum extrapolation
                add_col_chem_rate("X2Pi", 4, "X2Pi", 3, 1.8e-10, "H")  # @ 300 K single-quantum extrapolation
                add_col_chem_rate("X2Pi", 5, "X2Pi", 4, 1.9e-10, "H")  # @ 300 K single-quantum extrapolation
                add_col_chem_rate("X2Pi", 6, "X2Pi", 5, 2.0e-10, "H")  # @ 300 K single-quantum extrapolation
                add_col_chem_rate("X2Pi", 7, "X2Pi", 6, 2.1e-10, "H")  # @ 300 K single-quantum extrapolation
                add_col_chem_rate("X2Pi", 8, "X2Pi", 7, 2.2e-10, "H")  # @ 300 K single-quantum extrapolation
                add_col_chem_rate("X2Pi", 9, "X2Pi", 8, 2.3e-10, "H")  # @ 300 K single-quantum extrapolation
                add_col_chem_rate("X2Pi", 10, "X2Pi", 9, 2.4e-10, "H")  # @ 300 K single-quantum extrapolation
                # add_col_chem_rate("X2Pi", 11, "X2Pi", 10, 2.6e-10, "H")  # @ 300 K single-quantum extrapolation
            else:
                # NEW
                add_col_chem_rate("X2Pi", 3, "X2Pi", 2, 5.8e-10, "H")  # @ 300 K single-quantum extrapolation
                add_col_chem_rate("X2Pi", 4, "X2Pi", 3, 1.5e-09, "H")  # @ 300 K single-quantum extrapolation
                add_col_chem_rate("X2Pi", 5, "X2Pi", 4, 4.0e-09, "H")  # @ 300 K single-quantum extrapolation
                add_col_chem_rate("X2Pi", 6, "X2Pi", 5, 9.8e-09, "H")  # @ 300 K single-quantum extrapolation
                add_col_chem_rate("X2Pi", 7, "X2Pi", 6, 2.4e-08, "H")  # @ 300 K single-quantum extrapolation
                add_col_chem_rate("X2Pi", 8, "X2Pi", 7, 7.6e-08, "H")  # @ 300 K single-quantum extrapolation
                add_col_chem_rate("X2Pi", 9, "X2Pi", 8, 1.8e-07, "H")  # @ 300 K single-quantum extrapolation
                add_col_chem_rate("X2Pi", 10, "X2Pi", 9, 5.1e-07, "H")  # @ 300 K single-quantum extrapolation
            #####
            # Streit G. E., Johnston H. S., (1976), doi: http://dx.doi.org/10.1063/1.431917
            # Taken from Fig. 5, extrapolated to higher, lower v.
            add_col_chem_rate("X2Pi", 1, "X2Pi", 0, 1.0e-14, "H2")  # @ 300 K single-quantum extrapolation
            add_col_chem_rate("X2Pi", 2, "X2Pi", 1, 4.0e-14, "H2")  # @ 300 K single-quantum extrapolation
            add_col_chem_rate("X2Pi", 3, "X2Pi", 2, 9.0e-14, "H2")  # @ 300 K single-quantum extrapolation
            add_col_chem_rate("X2Pi", 4, "X2Pi", 3, 1.8e-13, "H2")  # @ 300 K single-quantum extrapolation
            add_col_chem_rate("X2Pi", 5, "X2Pi", 4, 3.9e-13, "H2")  # @ 300 K single-quantum extrapolation
            add_col_chem_rate("X2Pi", 6, "X2Pi", 5, 6.8e-13, "H2")  # @ 300 K single-quantum extrapolation
            add_col_chem_rate("X2Pi", 7, "X2Pi", 6, 8.0e-13, "H2")  # @ 300 K single-quantum extrapolation
            add_col_chem_rate("X2Pi", 8, "X2Pi", 7, 7.6e-13, "H2")  # @ 300 K single-quantum extrapolation
            add_col_chem_rate("X2Pi", 9, "X2Pi", 8, 5.8e-13, "H2")  # @ 300 K single-quantum extrapolation
            add_col_chem_rate("X2Pi", 10, "X2Pi", 9, 4.4e-13, "H2")  # @ 300 K single-quantum extrapolation
        return y_matrix

    def load_continuum_rates(self, temperature_profile: u.Quantity, wn_grid: u.Quantity):
        log.info(f"[I{self.n_iter}] Loading continuum absorption rates and profiles.")

        self.cont_states = pd.read_csv(
            self.cont_states_file,
            names=["id", "energy", "g"] + self.agg_col_names,
            usecols=[0, 1, 2] + self.agg_col_nums,
            sep=r"\s+",
        )
        merge_cols = ["id", "id_agg"]
        self.cont_states = self.cont_states.merge(self.states[merge_cols], on="id", how="left")
        self.cont_states["id_agg"] = self.cont_states["id_agg"].astype("Int64")
        # NB: Left join converts ints to float as some may be nan, does not occur for inner join but left needed here to
        # preserve energy/degeneracy info of upper states with no id_agg map.
        self.cont_rates = []

        def _aggregate_continuum_trans(
            chunk: pd.DataFrame,
            rates_lock: multiprocessing.Lock(),
            rates_set_event: multiprocessing.Event,
            layer_temperature_tuple: t.Tuple[int, u.Quantity],
        ):
            layer_idx, layer_temp = layer_temperature_tuple

            temp_states = boltzmann_population(states=self.cont_states.copy(), temperature=layer_temp)
            temp_states["n_frac"] = temp_states["n"] / temp_states["n_agg"]
            # trans_chunk = trans_chunk.loc[trans_chunk["id_i"] <= main_states_id_max]
            chunk = chunk.merge(
                temp_states[["id", "g", "energy", "id_agg", "n_frac"]],
                left_on="id_i",
                right_on="id",
                how="inner",
            )
            # chunk = chunk.loc[chunk["n_frac"] > 0.0]  # Need to keep these for rates!
            chunk = chunk.rename(
                columns={
                    "g": "g_i",
                    "energy": "energy_i",
                    "id_agg": "id_i_agg",
                    "n_frac": "n_frac_i",
                }
            )
            chunk = chunk.drop(columns=["id_i", "id"])

            chunk = chunk.merge(
                temp_states[["id", "g", "energy", "n_frac"]],
                left_on="id_c",
                right_on="id",
                how="inner",
            )
            chunk = chunk.rename(
                columns={
                    "g": "g_c",
                    "energy": "energy_c",
                    "n_frac": "n_frac_c",
                }
            )
            chunk = chunk.drop(columns=["id_c", "id"])

            chunk["energy_ci"] = chunk["energy_c"] - chunk["energy_i"]
            chunk = chunk.loc[
                (chunk["energy_c"] >= wn_grid[0].value)
                & (chunk["energy_i"] <= wn_grid[-1].value)  # TODO: IS this condition valid?
                & (chunk["energy_ci"] >= wn_grid[0].value)
                & (chunk["energy_ci"] <= wn_grid[-1].value)
            ]
            # TODO: Handle proper continuum state pops for cases without assumed 100% dissociation efficiency.
            chunk["n_frac_c"] = chunk["n_frac_c"].fillna(0)
            # Group only on lower agg. state as upper in continuum and not included in equilibrium.
            start_time = time.perf_counter()
            self.cont_profile_grid[layer_idx].append(
                BandProfileCollection(
                    band_profiles=chunk.groupby(by=["id_i_agg"]).apply(
                        lambda x: calc_band_profile(
                            wn_grid=wn_grid.value,
                            n_frac_f=x["n_frac_c"].to_numpy(),
                            n_frac_i=x["n_frac_i"].to_numpy(),
                            a_fi=x["A_ci"].to_numpy(),
                            g_f=x["g_c"].to_numpy(),
                            g_i=x["g_i"].to_numpy(),
                            energy_fi=x["energy_ci"].to_numpy(),
                            temperature=layer_temp.value,
                            species_mass=self.species_mass,
                            cont_broad=x["broad"].to_numpy(),
                        ),
                        include_groups=False,
                    )
                )
            )
            log.info(
                (
                    f"[L{layer_idx + self.n_lte_layers}] Continuum band profile duration ="
                    f" {time.perf_counter() - start_time:.2f}."
                )
            )
            rates_lock.acquire(blocking=True)
            try:
                if not rates_set_event.is_set():
                    rates_set_event.set()
                    # chunk = chunk.loc[(chunk["energy_ci"] >= wn_grid[0].value) & (chunk["energy_ci"] <= wn_grid[-1].value)]
                    chunk["B_ci"] = calc_einstein_b_fi(
                        a_fi=chunk["A_ci"].to_numpy() << 1 / u.s,
                        energy_fi=(chunk["energy_ci"].to_numpy() << 1 / u.cm).to(u.Hz, equivalencies=u.spectral()),
                    )
                    chunk["B_ic"] = calc_einstein_b_if(
                        b_fi=chunk["B_ci"].to_numpy(),
                        g_f=chunk["g_c"].to_numpy(),
                        g_i=chunk["g_i"].to_numpy(),
                    )
                    self.cont_rates.append(
                        chunk.groupby(by=["id_i_agg"], as_index=False).agg(
                            A_ci=("A_ci", "sum"), B_ci=("B_ci", "sum"), B_ic=("B_ic", "sum")
                        )
                    )
            finally:
                rates_lock.release()

        # NB: setting values on self variables if using ProcessPoolExecutor might not work, as different process
        # instances might not be able to access the member variables.
        cont_rates_manager = multiprocessing.Manager()
        n_nlte_layers = temperature_profile.shape[0] - self.n_lte_layers  # len(temperature_profile[self.n_lte_layers:])
        self.cont_profile_grid = [[] for _ in range(n_nlte_layers)]
        for cont_trans_file in self.cont_trans_files:
            log.info(
                f"[I{self.n_iter}] Processing file {cont_trans_file}."
                # f" Begin at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}."
            )
            cont_trans_reader = pd.read_csv(
                cont_trans_file,
                names=["id_c", "id_i", "A_ci", "broad"],
                sep=r"\s+",
                chunksize=_DEFAULT_CHUNK_SIZE,
            )
            for cont_trans_chunk in cont_trans_reader:
                chunk_rates_lock = cont_rates_manager.Lock()
                chunk_rates_set_event = cont_rates_manager.Event()
                agg_partial = partial(
                    _aggregate_continuum_trans, cont_trans_chunk, chunk_rates_lock, chunk_rates_set_event
                )
                with ThreadPoolExecutor(max_workers=30) as ex:
                    ex.map(
                        agg_partial,
                        list(enumerate(temperature_profile[self.n_lte_layers :])),
                    )

        # Each layer_idx is a list of 1 or more BandProfileCollection objects to be merged.
        for layer_idx in range(len(self.cont_profile_grid)):
            if len(self.cont_profile_grid[layer_idx]) > 1:
                self.cont_profile_grid[layer_idx][0].merge_collections(
                    self.cont_profile_grid[layer_idx][1:],
                    normalise=True,
                    spectral_grid=wn_grid,
                )
                self.cont_profile_grid[layer_idx] = self.cont_profile_grid[layer_idx][0]
            else:
                self.cont_profile_grid[layer_idx] = self.cont_profile_grid[layer_idx][0]
                self.cont_profile_grid[layer_idx].normalise(spectral_grid=wn_grid)

        self.cont_rates = pd.concat(self.cont_rates)
        self.cont_rates = self.cont_rates.groupby(by=["id_i_agg"], as_index=False).agg(
            A_ci=("A_ci", "sum"), B_ci=("B_ci", "sum"), B_ic=("B_ic", "sum")
        )
        log.info(f"Continuum rates = \n{self.cont_rates}")
        # log.info(f"Number of band profiles computed = {[len(bp) for bp in self.continuum_band_profile_grid]}.")

    def compute_pops_xsecs(
        self,
        temperature_profile: u.Quantity,
        pressure_profile: u.Quantity,
        wn_grid: u.Quantity,
    ) -> npt.NDArray[np.float64]:
        """
        Electronic state and vibrational quantum number(s) default column numbers are based on the ExoMol standard for
        diatomic molecules. Triatomic and polyatomic molecules will likely have multiple vibrational modes and different
        symmetries than may need to be aggregated on instead.

        :return:
        """
        if self.incident_radiation_field is not None and self.incident_radiation_field.shape != wn_grid.shape:
            raise ValueError(
                f"Mismatch between the size of the incident radiation field for upper boundary layer with shape "
                f"{self.incident_radiation_field.shape}, does not align with wavenumber grid {wn_grid.shape}."
            )
        if self.n_lte_layers < 0:
            raise ValueError(f"Number of LTE layers ({self.n_lte_layers}) set to unphysical negative value,")
        if self.n_lte_layers > temperature_profile.shape[0]:
            raise ValueError(
                f"Number of LTE layers ({self.n_lte_layers}) greater than the number of layers in temperature"
                f" profile (temperature_profile.shape[0])"
            )
        if (self.cont_rates is None) ^ (self.cont_profile_grid is None):
            raise RuntimeError(
                f"Either continuum rates or band profile grid is none.\n"
                f"Rates =\n{self.cont_rates}\n"
                f"Band profile grid =\n{self.cont_profile_grid}"
            )
        if self.cont_rates is not None and self.dissociation_products is None:
            raise RuntimeError(f"Continuum rates provided but no dissociation products given.")

        n_layers = temperature_profile.shape[0]

        n_angular_points = 50
        mu_values, mu_weights = np.polynomial.legendre.leggauss(n_angular_points)
        mu_values, mu_weights = (mu_values + 1) * 0.5, mu_weights / 2

        start_time = time.perf_counter()

        res = self.global_chi_matrix * self.dz_profile[:, None]
        dtau = res.decompose().value
        tau = dtau[::-1].cumsum(axis=0)[::-1]
        tau_mu = tau[:, None, :] / mu_values[None, :, None]
        effective_source_func_matrix, effective_tau_mu = effective_source_tau_mu(
            global_source_func_matrix=self.global_source_func_matrix,
            global_chi_matrix=self.global_chi_matrix,
            global_eta_matrix=self.global_eta_matrix,
            density_profile=self.density_profile,
            dz_profile=self.dz_profile,
            mu_values=mu_values,
            negative_absorption_factor=self.negative_absorption_factor,
        )
        bezier_coefs, control_points = bezier_coefficients(
            tau_mu_matrix=effective_tau_mu,
            source_function_matrix=effective_source_func_matrix.value,
            # tau_mu_matrix=tau_mu,
            # source_function_matrix=self.global_source_func_matrix.value,
        )
        control_points = control_points << self.global_source_func_matrix.unit
        log.info(f"Coefficient duration = {time.perf_counter() - start_time}")

        i_in_matrix = np.zeros_like(tau_mu) << self.global_source_func_matrix.unit
        lambda_in_matrix = np.zeros_like(tau_mu)
        ################
        # new_pop_grid = self.pop_grid.copy()
        new_pop_grid = self.pop_matrix[-1, :, :].copy()
        do_tridiag = True
        # Inward pass
        for layer_idx in range(n_layers)[::-1]:
            if layer_idx == n_layers - 1:
                if self.incident_radiation_field is not None:
                    i_in_matrix[layer_idx] = self.incident_radiation_field
            # elif layer_idx == 0 or layer_idx == n_layers - 2:
            #     # Parabolic:
            #     i_in_matrix[layer_idx] = (
            #             i_in_matrix[layer_idx + 1] * np.exp(-tau_plus_matrix[layer_idx])
            #             + np.sum(plus_coefficients[layer_idx, :2]
            #                      * self.global_source_func_matrix[layer_idx:layer_idx + 2][::-1, None, :], axis=0)
            #     )
            #     # log.info("I in (parabolic) = ", i_in_matrix[layer_idx], "All positive? ",
            #     #       np.all(i_in_matrix[layer_idx] >= 0))
            #     if not np.all(i_in_matrix[layer_idx] >= 0):
            #         log.warning("IN PARABOLIC BAD")
            #     # Diagonal: (gamma_plus is 0 in FOSC).
            #     # lambda_in_matrix[layer_idx] = plus_coefficients[layer_idx, 1]
            #     # Tridiagonal:
            #     # lambda_in_matrix[layer_idx] = (
            #     #         plus_coefficients[layer_idx, 0] +
            #     #         plus_coefficients[layer_idx, 1] * (1 + np.exp(-tau_plus_matrix[layer_idx]))
            #     # )
            #     lambda_in_matrix[layer_idx] = (
            #             plus_coefficients[layer_idx - 1, 0] +
            #             plus_coefficients[layer_idx, 1] * (1 + np.exp(-tau_plus_matrix[layer_idx - 1]))
            #     )
            #     # Switch to digonal if negative! If doesn't work, try FOSC.
            #     # lambda_in_matrix[layer_idx] = np.where(
            #     #     lambda_in_matrix[layer_idx] < 0,
            #     #     plus_coefficients[layer_idx, 1],
            #     #     lambda_in_matrix[layer_idx]
            #     # )
            #     # Bezier:
            #     i_in_matrix[layer_idx] = (
            #             i_in_matrix[layer_idx + 1] * np.exp(-bezier_coefs[layer_idx + 1, 0])
            #             + np.sum(bezier_coefs[layer_idx + 1, 1:3]
            #                      * self.global_source_func_matrix[layer_idx:layer_idx + 2][:, None, :], axis=0)
            #             + bezier_coefs[layer_idx + 1, 3] * control_points[layer_idx, 1]
            #     )
            #     # log.info("I in (Bezier) = ", i_in_matrix[layer_idx], "All positive? ", np.all(i_in_matrix[layer_idx] >= 0))
            #     if not np.all(i_in_matrix[layer_idx] >= 0):
            #         log.warning("IN BEZIER BAD :(")
            elif layer_idx == n_layers - 2:
                i_in_matrix[layer_idx] = (
                    i_in_matrix[layer_idx + 1] * np.exp(-bezier_coefs[layer_idx + 1, 0])
                    + np.sum(
                        bezier_coefs[layer_idx + 1, 1:3]
                        * effective_source_func_matrix[layer_idx : layer_idx + 2][:, None, :],
                        axis=0,
                    )
                    + bezier_coefs[layer_idx + 1, 3] * control_points[layer_idx, 1]
                )
                if do_tridiag:
                    lambda_in_matrix[layer_idx] = (
                        (bezier_coefs[layer_idx + 1, 1] + bezier_coefs[layer_idx + 1, 3])
                        * (1 + np.exp(-bezier_coefs[layer_idx, 0]))
                        + bezier_coefs[layer_idx, 2]
                        # + bezier_coefs[layer_idx, 3]
                    )
                else:
                    lambda_in_matrix[layer_idx] = bezier_coefs[layer_idx + 1, 1] + bezier_coefs[layer_idx + 1, 3]
                if not np.all(i_in_matrix[layer_idx] >= 0) or not np.all(lambda_in_matrix[layer_idx] >= 0):
                    bezier_debug(
                        layer_idx,
                        i_in_matrix,
                        lambda_in_matrix,
                        "in",
                        bezier_coefs,
                        control_points,
                    )
            elif layer_idx == 0:
                i_in_matrix[layer_idx] = (
                    i_in_matrix[layer_idx + 1] * np.exp(-bezier_coefs[layer_idx + 1, 0])
                    + np.sum(
                        bezier_coefs[layer_idx + 1, 1:3]
                        * effective_source_func_matrix[layer_idx : layer_idx + 2][:, None, :],
                        axis=0,
                    )
                    + bezier_coefs[layer_idx + 1, 3] * control_points[layer_idx, 1]
                )
                lambda_in_matrix[layer_idx] = bezier_coefs[layer_idx + 1, 1] + bezier_coefs[layer_idx + 1, 3]
                if not np.all(i_in_matrix[layer_idx] >= 0) or not np.all(lambda_in_matrix[layer_idx] >= 0):
                    bezier_debug(
                        layer_idx,
                        i_in_matrix,
                        lambda_in_matrix,
                        "in",
                        bezier_coefs,
                        control_points,
                    )
            else:
                # # Parabolic:
                # i_in_matrix[layer_idx] = (
                #         i_in_matrix[layer_idx + 1] * np.exp(-tau_plus_matrix[layer_idx])
                #         + np.sum(plus_coefficients[layer_idx]
                #                  * self.global_source_func_matrix[layer_idx - 1:layer_idx + 2][::-1, None, :], axis=0)
                # )
                #
                # # Diagonal only: (exp term was alpha previously)
                # # lambda_in_matrix[layer_idx] = (plus_coefficients[layer_idx, 1]
                # #                                + plus_coefficients[layer_idx, 2] * np.exp(-tau_plus_matrix[layer_idx]))
                # # Tridiagonal:
                # # lambda_in_matrix[layer_idx] = (
                # #         plus_coefficients[layer_idx, 0] +
                # #         plus_coefficients[layer_idx, 1] * (1 + np.exp(-tau_plus_matrix[layer_idx])) +
                # #         plus_coefficients[layer_idx, 2] * (
                # #                 1 + np.exp(-tau_plus_matrix[layer_idx]) + np.exp(-2 * tau_plus_matrix[layer_idx]))
                # # )
                # lambda_in_matrix[layer_idx] = (
                #         plus_coefficients[layer_idx - 1, 0] +
                #         plus_coefficients[layer_idx, 1] * (1 + np.exp(-tau_plus_matrix[layer_idx - 1])) +
                #         plus_coefficients[layer_idx + 1, 2] * (
                #                 1 + np.exp(-tau_plus_matrix[layer_idx])
                #                 + np.exp(-tau_plus_matrix[layer_idx] - tau_plus_matrix[layer_idx - 1]))
                # )
                # # Bezier:
                i_in_matrix[layer_idx] = (
                    i_in_matrix[layer_idx + 1] * np.exp(-bezier_coefs[layer_idx + 1, 0])
                    + np.sum(
                        bezier_coefs[layer_idx + 1, 1:3]
                        * effective_source_func_matrix[layer_idx : layer_idx + 2][:, None, :],
                        axis=0,
                    )
                    + bezier_coefs[layer_idx + 1, 3] * control_points[layer_idx, 1]
                )
                if do_tridiag:
                    lambda_in_matrix[layer_idx] = (
                        (bezier_coefs[layer_idx + 1, 1] + bezier_coefs[layer_idx + 1, 3])
                        * (1 + np.exp(-bezier_coefs[layer_idx, 0]))
                        + bezier_coefs[layer_idx, 2]
                        # + bezier_coefs[layer_idx, 3]
                    )
                else:
                    lambda_in_matrix[layer_idx] = bezier_coefs[layer_idx + 1, 1] + bezier_coefs[layer_idx + 1, 3]
                if not np.all(i_in_matrix[layer_idx] >= 0) or not np.all(lambda_in_matrix[layer_idx] >= 0):
                    bezier_debug(
                        layer_idx,
                        i_in_matrix,
                        lambda_in_matrix,
                        "in",
                        bezier_coefs,
                        control_points,
                    )
        i_out_matrix = np.zeros_like(tau_mu) << self.global_source_func_matrix.unit
        lambda_out_matrix = np.zeros_like(tau_mu)
        # Outward pass
        for layer_idx in range(n_layers):
            if layer_idx == 0:
                i_out_matrix[layer_idx] = blackbody(spectral_grid=wn_grid, temperature=temperature_profile[0])[0]
            elif layer_idx == 1:
                i_out_matrix[layer_idx] = (
                    i_out_matrix[layer_idx - 1] * np.exp(-bezier_coefs[layer_idx, 0])
                    + np.sum(
                        bezier_coefs[layer_idx, 1:3]
                        * effective_source_func_matrix[layer_idx - 1 : layer_idx + 1][::-1, None, :],
                        axis=0,
                    )
                    + bezier_coefs[layer_idx, 3] * control_points[layer_idx, 0]
                )
                if do_tridiag:
                    lambda_out_matrix[layer_idx] = (
                        (bezier_coefs[layer_idx, 1] + bezier_coefs[layer_idx, 3])
                        * (1 + np.exp(-bezier_coefs[layer_idx + 1, 0]))
                        + bezier_coefs[layer_idx + 1, 2]
                        # + bezier_coefs[layer_idx + 1, 3]
                    )
                else:
                    lambda_out_matrix[layer_idx] = bezier_coefs[layer_idx, 1] + bezier_coefs[layer_idx, 3]
                if not np.all(i_out_matrix[layer_idx] >= 0) or not np.all(lambda_out_matrix[layer_idx] >= 0):
                    bezier_debug(
                        layer_idx,
                        i_out_matrix,
                        lambda_out_matrix,
                        "out",
                        bezier_coefs,
                        control_points,
                    )

                i_in_matrix[layer_idx] = (
                    i_in_matrix[layer_idx + 1] * np.exp(-bezier_coefs[layer_idx + 1, 0])
                    + np.sum(
                        bezier_coefs[layer_idx + 1, 1:3]
                        * effective_source_func_matrix[layer_idx : layer_idx + 2][:, None, :],
                        axis=0,
                    )
                    + bezier_coefs[layer_idx + 1, 3] * control_points[layer_idx, 1]
                )
                if do_tridiag:
                    lambda_in_matrix[layer_idx] = (
                        (bezier_coefs[layer_idx + 1, 1] + bezier_coefs[layer_idx + 1, 3])
                        * (1 + np.exp(-bezier_coefs[layer_idx, 0]))
                        + bezier_coefs[layer_idx, 2]
                        # + bezier_coefs[layer_idx, 3]
                    )
                else:
                    lambda_in_matrix[layer_idx] = bezier_coefs[layer_idx + 1, 1] + bezier_coefs[layer_idx + 1, 3]
                if not np.all(i_in_matrix[layer_idx] >= 0) or not np.all(lambda_in_matrix[layer_idx] >= 0):
                    bezier_debug(
                        layer_idx,
                        i_in_matrix,
                        lambda_in_matrix,
                        "in",
                        bezier_coefs,
                        control_points,
                    )
            elif layer_idx < n_layers - 1:
                # # Parabolic:
                # i_out_matrix[layer_idx] = (
                #         i_out_matrix[layer_idx - 1] * np.exp(-tau_minus_matrix[layer_idx])
                #         + np.sum(minus_coefficients[layer_idx]
                #                  * self.global_source_func_matrix[layer_idx - 1:layer_idx + 2, None, :], axis=0)
                # )
                # if not np.all(i_out_matrix[layer_idx] >= 0):
                #     log.warning("OUT PARABOLIC BAD")
                #
                # # Diagonal only: (exp term was alpha previously)
                # # lambda_out_matrix[layer_idx] = (
                # #         minus_coefficients[layer_idx, 1]
                # #         + minus_coefficients[layer_idx, 2] * np.exp(-tau_minus_matrix[layer_idx])
                # # )
                # # Tridiagonal:
                # # lambda_out_matrix[layer_idx] = (
                # #         minus_coefficients[layer_idx, 0] +
                # #         minus_coefficients[layer_idx, 1] * (1 + np.exp(-tau_minus_matrix[layer_idx])) +
                # #         minus_coefficients[layer_idx, 2] * (
                # #                 1 + np.exp(-tau_minus_matrix[layer_idx]) + np.exp(-2 * tau_minus_matrix[layer_idx]))
                # # )
                # lambda_out_matrix[layer_idx] = (
                #         minus_coefficients[layer_idx + 1, 0] +
                #         minus_coefficients[layer_idx, 1] * (1 + np.exp(-tau_minus_matrix[layer_idx])) +
                #         minus_coefficients[layer_idx - 1, 2] * (
                #                 1 + np.exp(-tau_minus_matrix[layer_idx - 1])
                #                 + np.exp(-tau_minus_matrix[layer_idx - 1] - tau_minus_matrix[layer_idx]))
                # )
                #
                # # Repeated, but changes the value due to updated global_source_func_matrix[layer_idx - 1]
                # i_in_matrix[layer_idx] = (
                #         i_in_matrix[layer_idx + 1] * np.exp(-tau_plus_matrix[layer_idx])
                #         + np.sum(plus_coefficients[layer_idx]
                #                  * self.global_source_func_matrix[layer_idx - 1:layer_idx + 2][::-1, None, :], axis=0)
                # )
                # # Diagonal only: (exp term was alpha previously)
                # # lambda_in_matrix[layer_idx] = (plus_coefficients[layer_idx, 1]
                # #                                + plus_coefficients[layer_idx, 2] * np.exp(-tau_plus_matrix[layer_idx]))
                # # Tridiagonal:
                # # lambda_in_matrix[layer_idx] = (
                # #         plus_coefficients[layer_idx, 0] +
                # #         plus_coefficients[layer_idx, 1] * (1 + np.exp(-tau_plus_matrix[layer_idx])) +
                # #         plus_coefficients[layer_idx, 2] * (
                # #                 1 + np.exp(-tau_plus_matrix[layer_idx]) + np.exp(-2 * tau_plus_matrix[layer_idx]))
                # # )
                # lambda_in_matrix[layer_idx] = (
                #         plus_coefficients[layer_idx - 1, 0] +
                #         plus_coefficients[layer_idx, 1] * (1 + np.exp(-tau_plus_matrix[layer_idx - 1])) +
                #         plus_coefficients[layer_idx + 1, 2] * (
                #                 1 + np.exp(-tau_plus_matrix[layer_idx])
                #                 + np.exp(-tau_plus_matrix[layer_idx] - tau_plus_matrix[layer_idx - 1]))
                # )
                # Bezier:
                i_out_matrix[layer_idx] = (
                    i_out_matrix[layer_idx - 1] * np.exp(-bezier_coefs[layer_idx, 0])
                    + np.sum(
                        bezier_coefs[layer_idx, 1:3]
                        * effective_source_func_matrix[layer_idx - 1 : layer_idx + 1][::-1, None, :],
                        axis=0,
                    )
                    + bezier_coefs[layer_idx, 3] * control_points[layer_idx, 0]
                )
                if do_tridiag:
                    lambda_out_matrix[layer_idx] = (
                        (bezier_coefs[layer_idx, 1] + bezier_coefs[layer_idx, 3])
                        * (1 + np.exp(-bezier_coefs[layer_idx + 1, 0]))
                        + bezier_coefs[layer_idx + 1, 2]
                        # + bezier_coefs[layer_idx + 1, 3]
                    )
                else:
                    lambda_out_matrix[layer_idx] = bezier_coefs[layer_idx, 1] + bezier_coefs[layer_idx, 3]
                if not np.all(i_out_matrix[layer_idx] >= 0) or not np.all(lambda_out_matrix[layer_idx] >= 0):
                    bezier_debug(
                        layer_idx,
                        i_out_matrix,
                        lambda_out_matrix,
                        "out",
                        bezier_coefs,
                        control_points,
                    )

                i_in_matrix[layer_idx] = (
                    i_in_matrix[layer_idx + 1] * np.exp(-bezier_coefs[layer_idx + 1, 0])
                    + np.sum(
                        bezier_coefs[layer_idx + 1, 1:3]
                        * effective_source_func_matrix[layer_idx : layer_idx + 2][:, None, :],
                        axis=0,
                    )
                    + bezier_coefs[layer_idx + 1, 3] * control_points[layer_idx, 1]
                )
                # lambda_in_matrix[layer_idx] = (
                #         (bezier_coefs[layer_idx + 1, 1] + 0.5 * bezier_coefs[layer_idx + 1, 3])
                #         * (1 + np.exp(-bezier_coefs[layer_idx, 0]))
                #         + bezier_coefs[layer_idx, 2] + 0.5 * bezier_coefs[layer_idx, 3]
                # )
                if do_tridiag:
                    lambda_in_matrix[layer_idx] = (
                        (bezier_coefs[layer_idx + 1, 1] + bezier_coefs[layer_idx + 1, 3])
                        * (1 + np.exp(-bezier_coefs[layer_idx, 0]))
                        + bezier_coefs[layer_idx, 2]
                        # + bezier_coefs[layer_idx, 3]
                    )
                else:
                    lambda_in_matrix[layer_idx] = bezier_coefs[layer_idx + 1, 1] + bezier_coefs[layer_idx + 1, 3]
                if not np.all(i_in_matrix[layer_idx] >= 0) or not np.all(lambda_in_matrix[layer_idx] >= 0):
                    bezier_debug(
                        layer_idx,
                        i_in_matrix,
                        lambda_in_matrix,
                        "in",
                        bezier_coefs,
                        control_points,
                    )
            else:
                # At the upper boundary
                # # Parabolic
                # i_out_matrix[layer_idx] = (
                #         i_out_matrix[layer_idx - 1] * np.exp(-tau_minus_matrix[layer_idx])
                #         + np.sum(minus_coefficients[layer_idx, :2]
                #                  * self.global_source_func_matrix[layer_idx - 1:layer_idx + 1, None, :], axis=0)
                # )
                # if not np.all(i_out_matrix[layer_idx] >= 0):
                #     log.warning("OUT PARABOLIC BAD")
                # # Diagonal only: (gamma_plus is 0 in FOSC).
                # # lambda_out_matrix[layer_idx] = minus_coefficients[layer_idx, 1]
                # # Tridiagonal:
                # # lambda_out_matrix[layer_idx] = (
                # #         minus_coefficients[layer_idx, 0] +
                # #         minus_coefficients[layer_idx, 1] * (1 + np.exp(-tau_minus_matrix[layer_idx]))
                # # )
                # lambda_out_matrix[layer_idx] = (
                #         minus_coefficients[layer_idx, 1] * (1 + np.exp(-tau_minus_matrix[layer_idx]))
                # )
                # Bezier:
                i_out_matrix[layer_idx] = (
                    i_out_matrix[layer_idx - 1] * np.exp(-bezier_coefs[layer_idx, 0])
                    + np.sum(
                        bezier_coefs[layer_idx, 1:3]
                        * effective_source_func_matrix[layer_idx - 1 : layer_idx + 1][::-1, None, :],
                        axis=0,
                    )
                    + bezier_coefs[layer_idx, 3] * control_points[layer_idx, 0]
                )
                lambda_out_matrix[layer_idx] = bezier_coefs[layer_idx, 1] + bezier_coefs[layer_idx, 3]
                if not np.all(i_out_matrix[layer_idx] >= 0) or not np.all(lambda_out_matrix[layer_idx] >= 0):
                    bezier_debug(
                        layer_idx,
                        i_out_matrix,
                        lambda_out_matrix,
                        "out",
                        bezier_coefs,
                        control_points,
                    )
            # Solve equilibrium for non-LTE layers.
            if layer_idx >= self.n_lte_layers:
                nlte_layer_idx = layer_idx - self.n_lte_layers
                layer_temp = temperature_profile[layer_idx]
                layer_pressure = pressure_profile[layer_idx]

                y_matrix = np.zeros((self.n_agg_states, self.n_agg_states)) << (1 / u.s)

                # Integrate over all angles. This can be done independent of the transitions.
                i_layer_grid = 0.5 * np.sum(
                    (i_in_matrix[layer_idx] + i_out_matrix[layer_idx]) * mu_weights[:, None],
                    axis=0,
                )
                lambda_layer_grid = 0.5 * np.sum(
                    (lambda_in_matrix[layer_idx] + lambda_out_matrix[layer_idx]) * mu_weights[:, None],
                    axis=0,
                )

                # species_eta = (
                #     self.mol_source_func_matrix[layer_idx]
                #     * self.mol_chi_matrix[layer_idx]
                #     * self.chem_profile[self.species][layer_idx]
                # )
                # species_num_dens = (
                #     self.chem_profile[self.species][layer_idx] * self.density_profile[layer_idx]
                # ).decompose()
                # log.info(f"Species number density = {species_num_dens}")
                species_eta = self.chem_profile[self.species][layer_idx] * self.mol_eta_matrix[layer_idx] / ac.c
                global_chi = self.global_chi_matrix[layer_idx] / self.density_profile[layer_idx]
                # global_chi = np.where(global_chi < 0, self.negative_absorption_factor * abs(global_chi), global_chi)
                psi_approx_eta = np.zeros(lambda_layer_grid.shape[0]) << self.global_source_func_matrix.unit
                psi_approx_eta[global_chi != 0] = (
                    lambda_layer_grid[global_chi != 0] * species_eta[global_chi != 0] / global_chi[global_chi != 0]
                )
                psi_approx_eta = np.clip(abs(psi_approx_eta), 0, i_layer_grid)

                log.info(f"[L{layer_idx}] Average I = {i_layer_grid.mean()}")
                i_prec = (i_layer_grid - psi_approx_eta) * 4 * np.pi * u.sr
                log.info(f"[L{layer_idx}] average I_prec = {i_prec.mean()}")

                for trans_row in self.rates_grid.itertuples(index=False):
                    # 0 = id_f_agg, 1 = id_i_agg, 2 = A_fi, 3 = B_fi, 4 = B_if.
                    a_fi = trans_row[2] / u.s
                    b_fi = trans_row[3] * (u.m**2) / (u.J * u.s)
                    b_if = trans_row[4] * (u.m**2) / (u.J * u.s)
                    log.info(f"[L{layer_idx}] Trans: {trans_row}.")

                    if (trans_row[0], trans_row[1]) not in self.abs_profile_grid[nlte_layer_idx]:
                        log.warning(
                            (
                                f"[L{layer_idx}] Trans {(trans_row[0], trans_row[1])} in rates but not in absorption "
                                f"band profiles."
                            )
                        )
                        continue
                    if (trans_row[0], trans_row[1]) not in self.emi_profile_grid[nlte_layer_idx]:
                        log.warning(
                            (
                                f"[L{layer_idx}] Trans {(trans_row[0], trans_row[1])} in rates but not in emission "
                                f"band profiles."
                            )
                        )
                        continue

                    abs_profile = self.abs_profile_grid[nlte_layer_idx].get((trans_row[0], trans_row[1]))
                    emi_profile = self.emi_profile_grid[nlte_layer_idx].get((trans_row[0], trans_row[1]))
                    abs_end_idx = abs_profile.start_idx + len(abs_profile.profile)
                    emi_end_idx = emi_profile.start_idx + len(emi_profile.profile)

                    # u_fi = (
                    #     emi_profile.profile
                    #     * emi_profile.integral
                    #     * u.erg
                    #     * u.cm
                    #     * 4
                    #     * np.pi
                    #     / (u.s * ac.h.cgs * ac.c.cgs * wn_grid[emi_profile.start_idx : emi_end_idx])
                    # )
                    # u_fi = simpson(u_fi, x=wn_grid[emi_profile.start_idx : emi_end_idx]) << u_fi.unit * wn_grid.unit
                    u_fi = a_fi
                    u_fi = u_fi.decompose()
                    log.debug(f"[L{layer_idx}] U_{trans_row[0], trans_row[1]} = {u_fi}")

                    # stim_emi_profile = (
                    #     emi_profile.profile
                    #     * emi_profile.integral
                    #     * u.erg
                    #     * u.cm
                    #     / (2 * u.s * ac.h.cgs**2 * ac.c.cgs**2 * wn_grid[emi_profile.start_idx : emi_end_idx] ** 4)
                    # )
                    # v_fi_prec = stim_emi_profile * i_prec[emi_profile.start_idx : emi_end_idx]
                    # v_fi_prec = (
                    #     simpson(v_fi_prec, x=wn_grid[emi_profile.start_idx : emi_end_idx])
                    #     << v_fi_prec.unit * wn_grid.unit
                    # )
                    stim_emi_profile = (
                        emi_profile.profile
                        * emi_profile.integral
                        * u.erg
                        * u.cm
                        / (2 * u.s * ac.h.cgs * ac.c.cgs**2 * wn_grid[emi_profile.start_idx : emi_end_idx] ** 3)
                    )
                    stim_emi_profile = stim_emi_profile.value / simpson(
                        stim_emi_profile, x=wn_grid[emi_profile.start_idx : emi_end_idx]
                    )

                    # Cross terms:
                    chi_if = np.zeros(lambda_layer_grid.shape[0]) << (u.m**2) / (u.J * u.s)
                    chi_if[abs_profile.start_idx : abs_profile.start_idx + len(abs_profile.profile)] += (
                        self.pop_matrix[-1, layer_idx, trans_row[1]]
                        * abs_profile.profile
                        * trans_row[4]
                        * u.m**2
                        / (u.J * u.s)
                    )
                    chi_if[emi_profile.start_idx : emi_end_idx] -= (
                        self.pop_matrix[-1, layer_idx, trans_row[0]]
                        * stim_emi_profile
                        * trans_row[3]
                        * u.m**2
                        / (u.J * u.s)
                    )
                    chi_if *= self.chem_profile[self.species][layer_idx]
                    chi_if = np.where(chi_if < 0, 0, chi_if)
                    if self.full_prec:
                        for o_idx in np.arange(0, self.n_agg_states):
                            a_ox_cross = np.zeros(lambda_layer_grid.shape[0]) << u.erg / u.sr
                            for ox_trans in self.rates_grid.loc[self.rates_grid["id_f_agg"] == o_idx].itertuples(
                                index=False
                            ):
                                ox_emi_profile = self.emi_profile_grid[nlte_layer_idx].get((ox_trans[0], ox_trans[1]))
                                ox_emi_end_idx = ox_emi_profile.start_idx + len(ox_emi_profile.profile)
                                a_ox_cross[ox_emi_profile.start_idx : ox_emi_end_idx] += (
                                    ox_emi_profile.profile
                                    * ox_emi_profile.integral
                                    * u.erg
                                    * u.cm
                                    * self.chem_profile[self.species][layer_idx]
                                    / (u.s * u.sr * ac.c.cgs)
                                )
                            if np.all(a_ox_cross == 0):
                                continue

                            psi_approx_cross = np.zeros(lambda_layer_grid.shape[0]) << a_ox_cross.unit / global_chi.unit
                            shielded_lambda = np.clip(lambda_layer_grid, 0, 1)
                            psi_approx_cross[global_chi != 0] = (
                                shielded_lambda[global_chi != 0]
                                * a_ox_cross[global_chi != 0]
                                / global_chi[global_chi != 0]
                            ).to(u.J / (u.m**2 * u.sr), equivalencies=u.spectral())

                            # if np.any(abs(psi_approx_cross) > i_layer_grid):
                            #     log.error(f"[L{layer_idx}] Psi*cross > Intensity!?")
                            # psi_approx_cross = np.clip(abs(psi_approx_cross), 0, i_layer_grid) * u.sr
                            psi_approx_cross = abs(chi_if * psi_approx_cross) * 4 * np.pi * u.sr

                            psi_approx_cross = (
                                simpson(psi_approx_cross, x=wn_grid) << psi_approx_cross.unit
                            ).decompose()
                            log.debug(
                                (
                                    f"[L{layer_idx}] Chi_{trans_row[0], trans_row[1]}"
                                    f"Psi*[Sum_(j<{o_idx}) A_({o_idx}, j)] = {-psi_approx_cross}"
                                )
                            )
                            y_matrix[trans_row[1], o_idx] -= psi_approx_cross
                            log.debug(
                                (
                                    f"[L{layer_idx}] Chi_{trans_row[1], trans_row[0]}"
                                    f"Psi*[Sum_(j<{o_idx}) A_({o_idx}, j)] = {psi_approx_cross}"
                                )
                            )
                            y_matrix[trans_row[0], o_idx] += psi_approx_cross
                    else:
                        self_prec = np.zeros(lambda_layer_grid.shape[0])
                        self_prec[global_chi != 0] = (
                            lambda_layer_grid[global_chi != 0]
                            * chi_if[global_chi != 0]
                            * self.chem_profile[self.species][layer_idx]
                            * ac.h
                            / global_chi[global_chi != 0].to(u.m**2, equivalencies=u.spectral())
                        )
                        self_prec = simpson(self_prec, x=wn_grid)
                        u_fi *= 1 - self_prec
                    # End cross.

                    v_fi_prec = stim_emi_profile * i_prec[emi_profile.start_idx : emi_end_idx]
                    v_fi_prec = (
                        simpson(v_fi_prec, x=wn_grid[emi_profile.start_idx : emi_end_idx]) << v_fi_prec.unit
                    ) * b_fi
                    v_fi_prec = v_fi_prec.decompose()
                    log.debug(f"[L{layer_idx}] V_{trans_row[0], trans_row[1]}_prec = {v_fi_prec}")

                    # v_if_prec = (
                    #     abs_profile.profile
                    #     * abs_profile.integral
                    #     * 4
                    #     * np.pi
                    #     * u.cm**2
                    #     * i_prec[abs_profile.start_idx : abs_end_idx]
                    #     / (wn_grid[abs_profile.start_idx : abs_end_idx] * ac.h)
                    # )
                    # v_if_prec = (
                    #     simpson(v_if_prec, x=wn_grid[abs_profile.start_idx : abs_end_idx])
                    #     << v_if_prec.unit * wn_grid.unit
                    # )
                    v_if_prec = abs_profile.profile * i_prec[abs_profile.start_idx : abs_end_idx]
                    v_if_prec = (
                        simpson(v_if_prec, x=wn_grid[abs_profile.start_idx : abs_end_idx]) << v_if_prec.unit
                    ) * b_if
                    v_if_prec = v_if_prec.decompose()
                    log.debug(f"[L{layer_idx}] V_{trans_row[1], trans_row[0]}_prec = {v_if_prec}")

                    imi_if = calc_imi(
                        band_profile=abs_profile,
                        i_grid=i_prec,
                        wn_grid=wn_grid,
                    )
                    # log.debug(f"[L{layer_idx}] IMI_{trans_row[1], trans_row[0]} = {imi_if}")
                    # log.debug(f"[L{layer_idx}] BX = {imi_if * b_if}")
                    # lambda_approx_source_if = calc_imi(
                    #     band_profile=emi_profile,
                    #     # band_profile=abs_profile,
                    #     i_grid=psi_approx_eta,
                    #     wn_grid=wn_grid,
                    # )

                    # if lambda_approx_source_if > imi_if:
                    #     log.warning(
                    #         (
                    #             f"[L{layer_idx}] Trans {(trans_row[0], trans_row[1])} Lambda*Source > IMI "
                    #             f"(Abs./if) leading to X_if<0; zeroing."
                    #         )
                    #     )
                    # x_if = 4.0 * np.pi * u.sr * max(imi_if.value - lambda_approx_source_if.value, 0) * imi_if.unit

                    # imi_fi = calc_imi(
                    #     band_profile=emi_profile,
                    #     i_grid=i_layer_grid,
                    #     wn_grid=wn_grid,
                    # )

                    # lambda_approx_source_fi = calc_imi(
                    #     band_profile=emi_profile,
                    #     i_grid=psi_approx_eta,
                    #     wn_grid=wn_grid,
                    # )
                    # if lambda_approx_source_fi > imi_fi:
                    #     log.warning(
                    #         (
                    #             f"[L{layer_idx}] Trans {(trans_row[0], trans_row[1])} Lambda*Source > IMI "
                    #             f"(Emi./fi) leading to X_fi<0; zeroing."
                    #         )
                    #     )
                    # x_fi = 4.0 * np.pi * u.sr * max(imi_fi.value - lambda_approx_source_fi.value, 0) * imi_fi.unit

                    # if imi_if.value < 0 or imi_if.value >= 1:
                    #     log.warning(
                    #         (
                    #             f"[L{layer_idx}] IMI_if outside interval [0,+1) for trans "
                    #             f"{(trans_row[0], trans_row[1])} within IMI_if = {imi_if}."
                    #         )
                    #     )
                    #     imi_if = max(min(imi_if.value, 1.0), 0.0) << imi_if.unit
                    # if imi_fi.value < 0 or imi_fi.value >= 1:
                    #     log.warning(
                    #         (
                    #             f"[L{layer_idx}] IMI_fi outside interval [0,+1) for trans "
                    #             f"{(trans_row[0], trans_row[1])} within IMI_fi = {imi_fi}."
                    #         )
                    #     )
                    #     imi_fi = max(min(imi_fi.value, 1.0), 0.0) << imi_fi.unit

                    # if self.debug:
                    #     log.info(f"[L{layer_idx}] Rate properties for trans {trans_row[0], trans_row[1]}")
                    #     log.info(f"[L{layer_idx}] (emi.) A_fi = {a_fi}")
                    #     log.info(f"[L{layer_idx}] (emi.) B_fi = {b_fi}")
                    #     log.info(f"[L{layer_idx}] (abs.) B_if = {b_if}")
                    #     log.info(f"[L{layer_idx}] (abs.) IMI_if = {imi_if}")
                    #     log.info(f"[L{layer_idx}] (abs.) lambda*Source_if = {lambda_approx_source_if}")
                    #     log.info(f"[L{layer_idx}] (abs.) X_if = {x_if}")
                    #     log.info(f"[L{layer_idx}] (abs.) B_if * X_if = {b_if * x_if}")
                    #     log.info(f"[L{layer_idx}] (emi.) IMI_fi = {imi_fi}")
                    #     # log.info(f"[L{layer_idx}] (emi.) Lambda*_fi = {lambda_approx_fi}")
                    #     log.info(f"[L{layer_idx}] (emi.) lambda*Source_fi = {lambda_approx_source_fi}")
                    #     log.info(f"[L{layer_idx}] (emi.) X_fi = {x_fi}")
                    #     # log.info(f"[L{layer_idx}] (emi.) Z_fi = {z_fi}")
                    #     # log.info(f"[L{layer_idx}] (emi.) A_fi * Z_fi = {a_fi * z_fi}")
                    #     log.info(f"[L{layer_idx}] (emi.) B_fi * X_fi = {b_fi * x_fi}")

                    # Original:
                    # y_matrix[trans_row[0], trans_row[1]] += b_if * x_if  # + C_if
                    # y_matrix[trans_row[1], trans_row[0]] += (a_fi * z_fi) + (b_fi * x_fi)  # + C_fi
                    # y_matrix[trans_row[0], trans_row[0]] -= (a_fi * z_fi) + (b_fi * x_fi)  # + C_fi
                    # y_matrix[trans_row[1], trans_row[1]] -= b_if * x_if  # + C_if
                    # Full overlap preconditioning:
                    # Old:
                    # y_matrix[trans_row[0], trans_row[1]] += b_if * x_if  # + C_if
                    # y_matrix[trans_row[1], trans_row[0]] += a_fi + (b_fi * x_fi)  # + C_fi
                    # y_matrix[trans_row[0], trans_row[0]] -= a_fi + (b_fi * x_fi)  # + C_fi
                    # y_matrix[trans_row[1], trans_row[1]] -= b_if * x_if  # + C_if
                    # New:
                    y_matrix[trans_row[0], trans_row[1]] += v_if_prec
                    y_matrix[trans_row[1], trans_row[0]] += u_fi + v_fi_prec
                    y_matrix[trans_row[0], trans_row[0]] -= u_fi + v_fi_prec
                    y_matrix[trans_row[1], trans_row[1]] -= v_if_prec

                if self.cont_rates is not None:
                    for cont_trans_row in self.cont_rates.itertuples(index=False):
                        a_ci = cont_trans_row[1] / u.s
                        # b_ci = cont_trans_row[2] * (u.m ** 2) / (u.J * u.s)
                        b_ic = cont_trans_row[3] * u.m**2 / (u.J * u.s)

                        if cont_trans_row[0] in self.cont_profile_grid[nlte_layer_idx]:
                            cont_abs_profile = self.cont_profile_grid[nlte_layer_idx].get(cont_trans_row[0])
                            cont_abs_end_idx = cont_abs_profile.start_idx + len(cont_abs_profile.profile)

                            # Cross terms:
                            chi_ci = np.zeros(lambda_layer_grid.shape[0]) << u.m**2 / (u.J * u.s)
                            chi_ci[cont_abs_profile.start_idx : cont_abs_end_idx] += (
                                self.pop_matrix[-1, layer_idx, cont_trans_row[0]]
                                * cont_abs_profile.profile
                                * cont_trans_row[3]
                                * self.chem_profile[self.species][layer_idx]
                                * (u.m**2)
                                / (u.J * u.s)
                            )
                            chi_ci = np.where(chi_ci < 0, 0, chi_ci)
                            if self.full_prec:
                                for o_idx in np.arange(0, self.n_agg_states):
                                    a_ox_cross = np.zeros(lambda_layer_grid.shape[0]) << u.erg / u.sr
                                    for ox_trans in self.rates_grid.loc[
                                        self.rates_grid["id_f_agg"] == o_idx
                                    ].itertuples(index=False):
                                        ox_emi_profile = self.emi_profile_grid[nlte_layer_idx].get(
                                            (ox_trans[0], ox_trans[1])
                                        )
                                        ox_emi_end_idx = ox_emi_profile.start_idx + len(ox_emi_profile.profile)

                                        a_ox_cross[ox_emi_profile.start_idx : ox_emi_end_idx] += (
                                            ox_emi_profile.profile
                                            * ox_emi_profile.integral
                                            * u.erg
                                            * u.cm
                                            * self.chem_profile[self.species][layer_idx]
                                            / (u.s * u.sr * ac.c.cgs)
                                        )
                                    if np.all(a_ox_cross == 0):
                                        continue
                                    psi_approx_cross = (
                                        np.zeros(lambda_layer_grid.shape[0]) << a_ox_cross.unit / global_chi.unit
                                    )
                                    shielded_lambda = np.clip(lambda_layer_grid, 0, 1)
                                    psi_approx_cross[global_chi != 0] = (
                                        shielded_lambda[global_chi != 0]
                                        * a_ox_cross[global_chi != 0]
                                        / global_chi[global_chi != 0]
                                    ).to(u.J / (u.m**2 * u.sr), equivalencies=u.spectral())
                                    # if np.any(abs(psi_approx_cross) > i_layer_grid):
                                    #     log.error(f"[L{layer_idx}] Psi*cross > Intensity!?")
                                    # psi_approx_cross = np.clip(abs(psi_approx_cross), 0, i_layer_grid) * u.sr
                                    psi_approx_cross = abs(chi_ci * psi_approx_cross) * 4 * np.pi * u.sr

                                    psi_approx_cross = (
                                        simpson(psi_approx_cross, x=wn_grid) << psi_approx_cross.unit
                                    ).decompose()
                                    log.debug(
                                        (
                                            f"[L{layer_idx}] Adding "
                                            f"Chi_({cont_trans_row[0]}, c)Psi*[Sum_(j<{o_idx}) A_({o_idx}, j)] ="
                                            f" {-psi_approx_cross}"
                                        )
                                    )
                                    y_matrix[cont_trans_row[0], o_idx] -= psi_approx_cross
                            # End cross.

                            # v_ic_prec = (
                            #     cont_band_profile.profile
                            #     * cont_band_profile.integral
                            #     * 4
                            #     * np.pi
                            #     * u.cm**2
                            #     * i_prec[cont_band_profile.start_idx : cont_end_idx]
                            #     / (wn_grid[cont_band_profile.start_idx : cont_end_idx] * ac.h)
                            # )
                            # v_ic_prec = (
                            #     simpson(v_ic_prec, x=wn_grid[cont_band_profile.start_idx : cont_end_idx])
                            #     << v_ic_prec.unit * wn_grid.unit
                            # )

                            v_ic_prec = cont_abs_profile.profile * i_prec[cont_abs_profile.start_idx : cont_abs_end_idx]
                            v_ic_prec = (
                                simpson(v_ic_prec, x=wn_grid[cont_abs_profile.start_idx : cont_abs_end_idx])
                                << v_ic_prec.unit
                            ) * b_ic
                            v_ic_prec = v_ic_prec.decompose()
                            log.debug(f"[L{layer_idx}] V_ic_prec = {v_ic_prec}")

                            # imi_ic = calc_imi(
                            #     band_profile=cont_band_profile,
                            #     i_grid=i_layer_grid,
                            #     wn_grid=wn_grid,
                            # )
                            # lambda_approx_source_ic = calc_imi(
                            #     band_profile=cont_band_profile,
                            #     i_grid=psi_approx_eta,
                            #     wn_grid=wn_grid,
                            # )
                            # if lambda_approx_source_ic > imi_ic:
                            #     log.warning(
                            #         (
                            #             f"[L{layer_idx}] Trans ({cont_trans_row[0]}, c) Lambda*Source > IMI "
                            #             f"(Abs./ic) leading to X_ic<0; zeroing"
                            #         )
                            #     )

                            limiting_species_num_dens = min(
                                (
                                    self.chem_profile[self.dissociation_products[0]][layer_idx]
                                    if self.dissociation_products[0] in self.chem_profile.species
                                    else 0
                                ),
                                (
                                    self.chem_profile[self.dissociation_products[1]][layer_idx]
                                    if self.dissociation_products[1] in self.chem_profile.species
                                    else 0
                                ),
                            )
                            if limiting_species_num_dens == 0:
                                limiting_scale_factor = 0
                            else:
                                mol_num_dens = self.chem_profile[self.species][layer_idx]
                                # i_pop = self.pop_grid[layer_idx, cont_trans_row[0]]
                                i_pop = self.pop_matrix[-1, layer_idx, cont_trans_row[0]]
                                limiting_scale_factor = i_pop * mol_num_dens / limiting_species_num_dens

                            # x_ic = (
                            #     4.0 * np.pi * u.sr * max(imi_ic.value - lambda_approx_source_ic.value, 0) * imi_ic.unit
                            # )
                        else:
                            log.warning(f"[L{layer_idx}] Warn: Continuum profile ({cont_trans_row[0]}, c) unavailable.")
                            # imi_ic = 0.0 * self.mol_source_func_matrix.unit
                            # lambda_approx_source_ic = 0.0 * self.mol_source_func_matrix.unit
                            # x_ic = 0.0 * u.sr << imi_ic.unit
                            limiting_scale_factor = 0.0

                        # if imi_ic.value < 0 or imi_ic.value >= 1:
                        #     log.warning(
                        #         (
                        #             f"[L{layer_idx}] IMI_ic outside interval [0,+1) for trans "
                        #             f"({cont_trans_row[0]}, c) within IMI_ic = {imi_ic}."
                        #         )
                        #     )
                        #     imi_ic = max(min(imi_ic.value, 1.0), 0.0) << imi_ic.unit

                        # if self.debug:
                        #     log.info(f"[L{layer_idx}] Rate properties for trans ({cont_trans_row[0]}, c)")
                        #     log.info(f"[L{layer_idx}] (cont. abs.) B_ic = {b_ic}")
                        #     log.info(f"[L{layer_idx}] (cont. abs.) IMI_ic = {imi_ic}")
                        #     log.info(f"[L{layer_idx}] (cont. abs.) lambda*Source_ic = {lambda_approx_source_ic}")
                        #     log.info(f"[L{layer_idx}] (cont. abs.) X_ic = {x_ic}")
                        #     log.info(f"[L{layer_idx}] (cont. abs.) B_ic * X_ic = {b_ic * x_ic}")
                        #     log.info(f"[L{layer_idx}] limiting_scale_factor = {limiting_scale_factor}")
                        # Old:
                        # y_matrix[cont_trans_row[0], cont_trans_row[0]] -= b_ic * x_ic
                        # # y_matrix[cont_trans_row[0], cont_trans_row[0]] += a_ci * z_ci * limiting_scale_factor
                        # y_matrix[cont_trans_row[0], cont_trans_row[0]] += a_ci * limiting_scale_factor
                        # y_matrix[cont_trans_row[0], cont_trans_row[0]] += b_ic * x_ic * limiting_scale_factor
                        # New:
                        y_matrix[cont_trans_row[0], cont_trans_row[0]] -= v_ic_prec
                        # y_matrix[cont_trans_row[0], cont_trans_row[0]] += a_ci * z_ci * limiting_scale_factor
                        y_matrix[cont_trans_row[0], cont_trans_row[0]] += a_ci * limiting_scale_factor
                        y_matrix[cont_trans_row[0], cont_trans_row[0]] += v_ic_prec * limiting_scale_factor

                y_matrix = self.add_col_chem_rates(y_matrix=y_matrix, layer_idx=layer_idx, layer_temp=layer_temp)

                y_matrix = y_matrix.value
                y_reduced_idx_map = [idx for idx in range(0, len(y_matrix)) if sum(abs(y_matrix[idx])) != 0]
                y_matrix_reduced = y_matrix[np.ix_(y_reduced_idx_map, y_reduced_idx_map)]
                log.debug(f"[L{layer_idx}] Y matrix (before row-normalisation) =\n{y_matrix_reduced}")
                log.debug(
                    f"[L{layer_idx}] Y matrix cond. (before row-normalisation) = {np.linalg.cond(y_matrix_reduced)}"
                )
                y_matrix_reduced /= abs(y_matrix_reduced).sum(axis=1)[:, None]

                check_rows = np.array(
                    [
                        np.all(y_matrix_reduced[idx, :] > 0) or np.all(y_matrix_reduced[idx, :] < 0)
                        for idx in range(y_matrix_reduced.shape[0])
                    ]
                )
                if np.any(check_rows):
                    major = (
                        f"[I{self.n_iter}][L{layer_idx}] Y matrix all same sign in rows {np.nonzero(check_rows)[0]};"
                        f" investigate unphysical rates."
                    )
                    log.error(major)
                    # raise RuntimeError(major)

                test_y_rect = np.vstack([y_matrix_reduced.copy(), np.ones(y_matrix_reduced.shape[1])])
                log.info(f"[L{layer_idx}] Y matrix (reduced) =\n{y_matrix_reduced}")
                log.info(f"[L{layer_idx}] Reduced Y matrix cond. = {np.linalg.cond(y_matrix_reduced)}")
                y_matrix_reduced[-1, :] = 1.0
                log.info(
                    f"[L{layer_idx}] Reduced Y matrix + conservation eq. cond. = {np.linalg.cond(y_matrix_reduced)}"
                )

                log.info(f"[L{layer_idx}] Rect. Y matrix cond. = {np.linalg.cond(test_y_rect)}")

                test_rhs_rect = np.zeros(test_y_rect.shape[0])
                test_rhs_rect[-1] = 1

                nppinv_pops = np.linalg.pinv(test_y_rect) @ test_rhs_rect
                nppinv_pops /= nppinv_pops.sum()

                if np.any(nppinv_pops < 0):
                    log.error(
                        f"[L{layer_idx}] Numpy Pseudo Inverse pops. contain negatives. Falling back to least squares..."
                    )
                    log.error(f"Negatives = {nppinv_pops}")
                    # log.error("DEBUG EXITING.")
                    # exit()
                    lsq_res = least_squares(
                        lambda x: np.dot(test_y_rect, x) - test_rhs_rect,
                        np.zeros(test_y_rect.shape[1]),
                        bounds=(0.0, 1.0),
                        method="trf",
                        ftol=1e-15,
                        gtol=1e-15,
                        xtol=1e-15,
                    )
                    log.debug(lsq_res)
                    least_squares_pops = lsq_res.x
                    log.debug(f"[L{layer_idx}] Least Squares pops. = {least_squares_pops}")
                    # least_squares_rms = np.sqrt(np.mean((test_full_y @ least_squares_pops) ** 2))
                    if any(least_squares_pops < 0):
                        raise RuntimeError(f"[L{layer_idx}] Least squares population bounds failed; negative pops.")
                    else:
                        pop_matrix = least_squares_pops
                else:
                    pop_matrix = nppinv_pops

                if self.damping_enabled:
                    #     new_frac = 0.1
                    #     old_frac = (1 - new_frac) / 2
                    #     pop_matrix = (
                    #         old_frac * self.pop_matrix[-2, layer_idx, y_reduced_idx_map]
                    #         + old_frac * self.pop_matrix[-1, layer_idx, y_reduced_idx_map]
                    #         + new_frac * pop_matrix
                    #     )
                    damping_factor = 0.5
                    pop_matrix = (
                        damping_factor * pop_matrix
                        + (1 - damping_factor) * self.pop_matrix[-1, layer_idx, y_reduced_idx_map]
                    )
                    log.info(f"[L{layer_idx}] New pops. (damped):")
                elif self.sor_enabled:
                    # old_pops = self.pop_matrix[-1, layer_idx, y_reduced_idx_map].copy()
                    pop_delta = pop_matrix - self.pop_matrix[-1, layer_idx, y_reduced_idx_map]
                    # sor_delta = max(abs(pop_delta)) / max(
                    #     abs(
                    #         self.pop_matrix[-1, layer_idx, y_reduced_idx_map]
                    #         - self.pop_matrix[-2, layer_idx, y_reduced_idx_map]
                    #     )
                    # )
                    with np.errstate(divide="ignore"):
                        current_max_change = max(abs(pop_delta) / self.pop_matrix[-1, layer_idx, y_reduced_idx_map])
                        previous_max_change = max(
                            abs(
                                self.pop_matrix[-1, layer_idx, y_reduced_idx_map]
                                - self.pop_matrix[-2, layer_idx, y_reduced_idx_map]
                            )
                            / self.pop_matrix[-2, layer_idx, y_reduced_idx_map]
                        )
                        # older_max_change = max(
                        #     abs(
                        #         self.pop_matrix[-2, layer_idx, y_reduced_idx_map]
                        #         - self.pop_matrix[-3, layer_idx, y_reduced_idx_map]
                        #     )
                        #     / self.pop_matrix[-3, layer_idx, y_reduced_idx_map]
                        # )
                        # current_delta = current_max_change / previous_max_change
                        # previous_delta = previous_max_change / older_max_change
                        # sor_delta = 0.5 * (current_delta + previous_delta)
                        sor_delta = current_max_change / previous_max_change
                    if sor_delta > 1:
                        log.warning(f"[L{layer_idx}] SOR delta greater than 1 ({sor_delta}) - limiting to 1.")

                    sor_delta = min(sor_delta, 1.0)
                    log.info(f"[L{layer_idx}] SOR delta = {sor_delta}")
                    # log.info(f"[L{layer_idx}] Maximum Lambda (iteration) matrix element = {max(lambda_layer_grid)}")
                    sor_omega = 2 / (1 + np.sqrt(1 - sor_delta))
                    sor_omega = min(max(sor_omega, 0.8), 1.4)
                    # # Adaptive SOR.
                    # if self.n_iter >= 3:
                    #     convergence_rate = np.linalg.norm(
                    #         self.pop_matrix[-1, :, :] - self.pop_matrix[-2, :, :], ord=2, axis=(1, 2)
                    #     ) / np.linalg.norm(
                    #         self.pop_matrix[-2, :, :] - self.pop_matrix[-3, :, :], ord=2, axis=(1, 2)
                    #     )
                    #     if convergence_rate >= 1:
                    #         # Diverging
                    #         sor_omega = 1.0
                    #     elif convergence_rate < 0.8:
                    #         # Converging
                    #         sor_omega = min(sor_omega + 0.1, 2)

                    pop_matrix = self.pop_matrix[-1, layer_idx, y_reduced_idx_map] + sor_omega * pop_delta
                    if np.any(pop_matrix < 0):
                        log.warning(f"[L{layer_idx}] SOR step lead to negative population(s) - scaling back SOR omega.")
                        # min_sor_omega = -old_pops[pop_delta != 0] / pop_delta[pop_delta != 0]
                        # sor_omega = min_sor_omega[(0 <= min_sor_omega) & (min_sor_omega <= 2)].min()
                        # Zero negative pops and normalise.
                        pop_matrix[pop_matrix < 0] = 0

                    pop_matrix /= pop_matrix.sum()
                    log.info(f"[L{layer_idx}] New pops. (SOR factor = {sor_omega})")
                else:
                    log.info(f"[L{layer_idx}] New pops.:")

                for idx, y_idx in enumerate(y_reduced_idx_map):
                    log.info(
                        (
                            f"[L{layer_idx}]"
                            f" n{self.agg_states.loc[self.agg_states["id_agg"] == y_idx, self.agg_col_names].values[0]}"
                            f" = {pop_matrix[idx]}"
                        )
                    )

                full_pops = np.zeros(self.n_agg_states)
                full_pops[y_reduced_idx_map] = pop_matrix
                new_pop_grid[layer_idx] = full_pops

                # Compute new Xsecs and Source function
                nlte_states = boltzmann_population(self.states.copy(), temperature_profile[layer_idx])
                nlte_states["n_agg_nlte"] = full_pops[nlte_states["id_agg"]]
                nlte_states["n_nlte"] = np.where(
                    nlte_states["n_agg"] == 0,
                    0,
                    nlte_states["n"] * nlte_states["n_agg_nlte"] / nlte_states["n_agg"],
                )
                # TODO: Above has some failure cases: where n_agg_nlte is non-zero, n_nlte should be non-zero even if
                #  n_agg or n are zero. This matters at low temperatures where some of the high energy states will have
                #  effectively 0 population. Does it? they will be small but never 0.
                log.info(
                    (
                        f"[L{layer_idx}] NLTE States = \n{nlte_states}\n"
                        f"[L{layer_idx}] Sum of LTE populations = {nlte_states['n'].sum()}.\n"
                        f"[L{layer_idx}] Sum of non-LTE populations = {nlte_states['n_nlte'].sum()}."
                    )
                )

                # _temp_xsec = np.zeros(len(wn_grid))
                # for key in self.abs_profile_grid[nlte_layer_idx].keys():
                #     n_i = full_pops[key[1]]
                #     band_profile = self.abs_profile_grid[nlte_layer_idx][key]
                #     _temp_xsec[band_profile.start_idx : band_profile.start_idx + len(band_profile.profile)] += (
                #         band_profile.profile * band_profile.integral * n_i
                #     )
                # TEST - ExoCross implementation.
                # log.info(nlte_states[["id", "energy", "g"] + self.agg_col_names + ["n", "n_nlte"]],)
                # xcross_wn, xcross_abs, xcross_emi = compute_xsec(
                #     nlte_states=nlte_states[["id", "energy", "g", "id_agg"] + self.agg_col_names + ["n_nlte"]].copy(),
                #     states_file=self.states_file, agg_col_nums=self.agg_col_nums, trans_files=self.trans_files,
                #     temperature=layer_temp, wn_grid=wn_grid, pop_col_num=5 + len(self.agg_col_nums),
                #     pressure=layer_pressure, intensity_threshold=1e-35, species_mass=self.species_mass,
                #     profile="Doppler", timing=True, linear_grid=True
                # )
                # xcross_wn2, xcross_abs2, xcross_emi2 = compute_xsec(
                #     nlte_states=nlte_states[["id", "energy", "g", "id_agg"] + self.agg_col_names + ["n"]].copy(),
                #     states_file=self.states_file, agg_col_nums=self.agg_col_nums, trans_files=self.trans_files,
                #     temperature=layer_temp, wn_grid=wn_grid, pop_col_num=5 + len(self.agg_col_nums),
                #     pressure=layer_pressure, intensity_threshold=1e-35, species_mass=self.species_mass,
                #     profile="Doppler", timing=True, linear_grid=True
                # )
                # END TEST
                start_time = time.perf_counter()
                abs_xsec, emi_xsec = abs_emi_xsec(
                    states=nlte_states,
                    trans_files=self.trans_files,
                    temperature=layer_temp,
                    pressure=layer_pressure,
                    species_mass=self.species_mass,
                    wn_grid=wn_grid.value,
                    broad_n=(self.broadening_params[1] if self.broadening_params is not None else None),
                    broad_gamma=(
                        self.broadening_params[0][:, layer_idx] if self.broadening_params is not None else None
                    ),
                )

                log.info(f"[L{layer_idx}] Numba Voigt duration = {time.perf_counter() - start_time}")
                if np.any(abs_xsec < 0):
                    log.warn(f"[L{layer_idx}] Negative contribution in absorption (stimulated emission dominates)")

                if self.cont_states is not None and self.cont_trans_files is not None:
                    # start_time = time.perf_counter()
                    # nlte_cont_states = self.continuum_states.merge(nlte_states[["id", "n_nlte"]], on="id", how="left")
                    # cont_xsec = continuum_xsec(
                    #     continuum_states=nlte_cont_states, continuum_trans_files=self.continuum_trans_files,
                    #     temperature=layer_temp, wn_grid=wn_grid.value
                    # )
                    # abs_xsec += cont_xsec
                    # log.info(f"[L{layer_idx}] Numba continuum Gauss duration = {time.perf_counter() - start_time}")
                    # TEST
                    # _temp_cont_xsec = np.zeros(len(wn_grid))
                    for key in self.cont_profile_grid[nlte_layer_idx].keys():
                        n_i = full_pops[key]
                        cont_abs_profile = self.cont_profile_grid[nlte_layer_idx][key]
                        abs_xsec[
                            cont_abs_profile.start_idx : cont_abs_profile.start_idx + len(cont_abs_profile.profile)
                        ] += (cont_abs_profile.profile * cont_abs_profile.integral * n_i)
                    # plt.plot(wn_grid.value, cont_xsec)
                    # plt.plot(wn_grid.value, _temp_cont_xsec)
                    # plt.show()
                    # END TEST
                # Abs. = cm**2 . mol**-1; Emi. = erg . s**-1 . sr**-1 . cm (i.e. per wavenumber)
                source_func_logic = (emi_xsec == 0) | (abs_xsec == 0)
                nlte_source_func = np.zeros_like(abs_xsec)
                nlte_source_func[~source_func_logic] = emi_xsec[~source_func_logic] / abs_xsec[~source_func_logic]
                # if self.source_func_threshold is not None:
                #     nlte_source_func[nlte_source_func < self.source_func_threshold] = 0.0
                nlte_source_func = nlte_source_func << u.erg / (u.s * u.sr * u.cm)
                nlte_source_func = (nlte_source_func / ac.c).to(u.J / (u.sr * u.m**2), equivalencies=u.spectral())

                # self.global_source_func_matrix[layer_idx] += (
                #     nlte_source_func - self.mol_source_func_matrix[layer_idx]
                # ) * self.chem_profile[self.species][layer_idx]
                # self.global_source_func_matrix[self.global_source_func_matrix < self.negative_source_func_cap] = (
                #     self.negative_source_func_cap
                # )
                self.mol_source_func_matrix[layer_idx] = nlte_source_func

                abs_xsec = abs_xsec << u.cm**2
                self.global_chi_matrix[layer_idx] += (
                    (abs_xsec - self.mol_chi_matrix[layer_idx])
                    * self.chem_profile[self.species][layer_idx]  # Mol VMR
                    * self.density_profile[layer_idx]
                )
                self.mol_chi_matrix[layer_idx] = abs_xsec
                emi_xsec = emi_xsec << u.erg * u.cm / (u.s * u.sr)
                self.global_eta_matrix[layer_idx] += (emi_xsec - self.mol_eta_matrix[layer_idx]) * self.chem_profile[
                    self.species
                ][layer_idx]
                self.mol_eta_matrix[layer_idx] = emi_xsec

                self.global_source_func_matrix[layer_idx] = (
                    self.global_eta_matrix[layer_idx]
                    * self.density_profile[layer_idx]
                    / (ac.c * self.global_chi_matrix[layer_idx])
                ).to(u.J / (u.sr * u.m**2), equivalencies=u.spectral())

                start_time = time.perf_counter()
                res = self.global_chi_matrix * self.dz_profile[:, None]
                dtau = res.decompose().value
                tau = dtau[::-1].cumsum(axis=0)[::-1]
                tau_mu = tau[:, None, :] / mu_values[None, :, None]

                effective_source_func_matrix, effective_tau_mu = effective_source_tau_mu(
                    global_source_func_matrix=self.global_source_func_matrix,
                    global_chi_matrix=self.global_chi_matrix,
                    global_eta_matrix=self.global_eta_matrix,
                    density_profile=self.density_profile,
                    dz_profile=self.dz_profile,
                    mu_values=mu_values,
                    negative_absorption_factor=self.negative_absorption_factor,
                )
                bezier_coefs, control_points = bezier_coefficients(
                    tau_mu_matrix=effective_tau_mu,
                    source_function_matrix=effective_source_func_matrix.value,
                    # tau_mu_matrix=tau_mu,
                    # source_function_matrix=self.global_source_func_matrix.value,
                )
                control_points = control_points << self.global_source_func_matrix.unit
                log.info(f"[L{layer_idx}] Coefficient (post) duration = {time.perf_counter() - start_time}")

                if layer_idx == 1:
                    i_out_matrix[layer_idx] = (
                        i_out_matrix[layer_idx - 1] * np.exp(-bezier_coefs[layer_idx, 0])
                        + np.sum(
                            bezier_coefs[layer_idx, 1:3]
                            * effective_source_func_matrix[layer_idx - 1 : layer_idx + 1][::-1, None, :],
                            axis=0,
                        )
                        + bezier_coefs[layer_idx, 3] * control_points[layer_idx, 0]
                    )
                    lambda_out_matrix[layer_idx] = (
                        (bezier_coefs[layer_idx, 1] + bezier_coefs[layer_idx, 3])
                        * (1 + np.exp(-bezier_coefs[layer_idx + 1, 0]))
                        + bezier_coefs[layer_idx + 1, 2]
                        # + bezier_coefs[layer_idx + 1, 3]
                    )
                    if not np.all(i_out_matrix[layer_idx] >= 0) or not np.all(lambda_out_matrix[layer_idx] >= 0):
                        bezier_debug(
                            layer_idx,
                            i_out_matrix,
                            lambda_out_matrix,
                            "out",
                            bezier_coefs,
                            control_points,
                        )
                if 1 < layer_idx < self.n_lte_layers - 1:
                    # Parabolic:
                    # i_out_matrix[layer_idx] = (
                    #         i_out_matrix[layer_idx - 1] * np.exp(-tau_minus_matrix[layer_idx])
                    #         + np.sum(minus_coefficients[layer_idx]
                    #                  * self.global_source_func_matrix[layer_idx - 1:layer_idx + 2, None, :], axis=0)
                    # )
                    #
                    # lambda_out_matrix[layer_idx] = (
                    #         minus_coefficients[layer_idx, 1]
                    #         + minus_coefficients[layer_idx, 0] * np.exp(-tau_minus_matrix[layer_idx])
                    # )
                    # Bezier:
                    i_out_matrix[layer_idx] = (
                        i_out_matrix[layer_idx - 1] * np.exp(-bezier_coefs[layer_idx, 0])
                        + np.sum(
                            bezier_coefs[layer_idx, 1:3]
                            * effective_source_func_matrix[layer_idx - 1 : layer_idx + 1][::-1, None, :],
                            axis=0,
                        )
                        + bezier_coefs[layer_idx, 3] * control_points[layer_idx, 0]
                    )
                    lambda_out_matrix[layer_idx] = (
                        (bezier_coefs[layer_idx, 1] + bezier_coefs[layer_idx, 3])
                        * (1 + np.exp(-bezier_coefs[layer_idx + 1, 0]))
                        + bezier_coefs[layer_idx + 1, 2]
                        # + bezier_coefs[layer_idx + 1, 3]
                    )
                    if not np.all(i_out_matrix[layer_idx] >= 0) or not np.all(lambda_out_matrix[layer_idx] >= 0):
                        bezier_debug(
                            layer_idx,
                            i_out_matrix,
                            lambda_out_matrix,
                            "out",
                            bezier_coefs,
                            control_points,
                        )
                if layer_idx == self.n_lte_layers - 1:
                    if np.any(i_out_matrix < 0) or np.any(i_in_matrix < 0):
                        with open((output_dir / f"i_out_I{self.n_iter}.pickle").resolve(), "wb") as pickle_file:
                            pickle.dump(i_out_matrix, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
                        with open((output_dir / f"i_in_I{self.n_iter}.pickle").resolve(), "wb") as pickle_file:
                            pickle.dump(i_in_matrix, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        return new_pop_grid


def bezier_debug(
    layer_idx: int,
    i_matrix: u.Quantity,
    lambda_matrix: npt.NDArray[np.float64],
    direction: str,
    bezier_coefs: npt.NDArray[np.float64],
    control_points: npt.NDArray[np.float64],
) -> None:
    if not np.all(i_matrix[layer_idx] >= 0) or not np.all(lambda_matrix[layer_idx] >= 0):
        coefs_check_idx = layer_idx if direction == "out" else layer_idx + 1
        control_point_idx = 0 if direction == "out" else 1
        also_check_idx = layer_idx - 1 if direction == "out" else layer_idx + 1
        if not np.all(i_matrix[layer_idx] >= 0):
            log.warning(f"[L{layer_idx}] Warn: {direction} INTENSITY BEZIER BAD :(")
            log.warning(f"Negative intensities in previous layer? {np.any(i_matrix[also_check_idx] < 0)}")


class XSecCollection(dict):

    @property
    def available_species(self) -> t.List[SpeciesFormula]:
        return list(self.keys())

    def __getitem__(self, key: SpeciesIdentType) -> XSecData:
        if isinstance(key, str):
            key = SpeciesFormula(key)
        return super().__getitem__(key)

    def get(self, key: SpeciesIdentType, default: t.Optional[t.Any] = None) -> XSecData:
        if isinstance(key, str):
            key = SpeciesFormula(key)
        return super().get(key, default=default)

    def __setitem__(self, key: SpeciesIdentType, value: XSecData) -> None:
        if isinstance(key, str):
            key = SpeciesFormula(key)
        return super().__setitem__(key, value)

    def __contains__(self, key: SpeciesIdentType) -> bool:
        if isinstance(key, str):
            key = SpeciesFormula(key)
        return super().__contains__(key)

    def __delitem__(self, key: SpeciesIdentType) -> None:
        if isinstance(key, str):
            key = SpeciesFormula(key)
        return super().__delitem__(key)

    def active_absorbers(self, species_list: t.List[SpeciesFormula]) -> t.Set[SpeciesFormula]:
        return set(species_list) & set(self.available_species)

    def add_replace_xsec_data(self, xsec_data: XSecData) -> None:
        self[xsec_data.species] = xsec_data

    def compute_opacities_profile(
        self,
        chemical_profile: ChemicalProfile,
        temperature: u.Quantity,
        pressure: u.Quantity,
        spectral_grid: u.Quantity,
    ) -> t.Dict[SpeciesFormula, u.Quantity]:
        active_species = self.active_absorbers(chemical_profile.species)
        # TODO: Move non-LTE process here when refactoring for multiple non-LTE species.

        return {
            species: self[species].opacity(temperature, pressure, spectral_grid) * chemical_profile[species][:, None]
            for species in active_species
        }

    @property
    def unified_grid(self) -> u.Quantity:
        res = [x.spectral_grid for x in self.values()]
        base_unit = res[0].unit
        res = np.concatenate([x.to(base_unit, equivalencies=u.spectral()).value for x in res])
        res = np.sort(np.unique(res)) << base_unit

        return res

    def is_converged(self) -> bool:
        # is_converged = True
        # for species in self:
        #     if type(self[species]) is ExomolNLTEXsec:
        #         is_converged = is_converged & self[species].is_converged
        is_converged = np.all([self[species].is_converged for species in self if type(self[species]) is ExomolNLTEXsec])
        return is_converged


def create_r_wn_grid(low: float, high: float, resolving_power: float) -> u.Quantity:
    resolving_f = np.log((resolving_power + 1) / resolving_power)
    n_points = round((np.log(high) - np.log(low)) / resolving_f) + 1
    return np.exp(np.arange(n_points) * resolving_f + np.log(low)) << u.k
