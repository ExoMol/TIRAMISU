import logging
import time
import abc
import pathlib
import typing as t

import numpy.typing as npt
import numpy as np

import astropy.units as u
import astropy.constants as ac

from .accelerator import HybridAccelerator
from .chemistry import SpeciesFormula, SpeciesIdentType, ChemicalProfile
from .nlte import (
    blackbody,
    effective_source_tau_mu,
    bezier_coefficients,
    update_layer_coefficients,
    formal_solve_general,
    NLTEProcessor,
)
from .numerics import loglinear_integral_1d

log = logging.getLogger(__name__)

# Units
source_func_unit = u.J / (u.sr * u.m ** 2)


def rebin_spectrum(
        x_in: npt.NDArray[np.float64],
        y_in: npt.NDArray[np.float64],
        new_centers: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Re-bin a high-resolution spectrum onto bins centered at `new_centers` using piecewise log-linear integration  for
    scientifically accurate averaging.
    """

    # Compute bin edges
    edges = np.zeros(len(new_centers) + 1)
    edges[1:-1] = 0.5 * (new_centers[:-1] + new_centers[1:])
    edges[0] = new_centers[0] - 0.5 * (new_centers[1] - new_centers[0])
    edges[-1] = new_centers[-1] + 0.5 * (new_centers[-1] - new_centers[-2])

    y_out = np.zeros_like(new_centers)

    for i in range(len(new_centers)):
        # Mask high-res points in this bin
        left = edges[i]
        right = edges[i + 1]

        # Interior points only
        mask = (x_in > left) & (x_in < right)

        # Add exact edge values via interpolation
        x_bin = np.concatenate((
            [left],
            x_in[mask],
            [right]
        ))
        y_bin = np.concatenate((
            [np.interp(left, x_in, y_in)],
            y_in[mask],
            [np.interp(right, x_in, y_in)]
        ))
        # Bin average
        dx_bin = np.diff(x_bin)
        bin_integral = loglinear_integral_1d(y_data=y_bin, dx=dx_bin)
        y_out[i] = bin_integral / (right - left)

    return y_out


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
                xsec_grid = f["xsecarr"][()] << u.cm ** 2

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
                return super().opacity(temperature, pressure, spectral_grid) << u.cm ** 2


class ExomolBinnedHDF5Xsec(XSecData):

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
        self.filepath = pathlib.Path(filepath)
        with h5py.File(self.filepath, "r") as f:
            species = f["mol_name"][0].decode("utf-8")
            self.species = SpeciesFormula(species)
            self.spectral_grid = f["bin_edges"][()] << u.k
            self.temperature_grid = f["t"][()] << u.K
            self.pressure_grid = f["p"][()] << u.bar
            self.xsec_grid = None
            if self.load_in_memory:
                self.xsec_grid = f["xsecarr"][()] << u.cm ** 2
        self.axes = (1, 0)

    def opacity(
            self,
            temperature: u.Quantity,
            pressure: u.Quantity,
            spectral_grid: t.Optional[u.Quantity] = None,
    ) -> u.Quantity:
        interped_spectra = bilinear_interpolate(
            self.xsec_grid,
            pressure,
            temperature,
            self.pressure_grid,
            self.temperature_grid,
            axes=self.axes,
        )
        if spectral_grid is not None:
            if np.array_equal(spectral_grid.value, self.spectral_grid.value):
                return interped_spectra
            spectral_grid = spectral_grid.to(self.spectral_grid.unit, equivalencies=u.spectral())

            binned_spectra = np.empty((interped_spectra.shape[0], spectral_grid.shape[0]))
            for idx in range(interped_spectra.shape[0]):
                binned_spectra[idx] = rebin_spectrum(
                    x_in=self.spectral_grid.value,
                    y_in=interped_spectra[idx].value,
                    new_centers=spectral_grid.value
                )
            interped_spectra = binned_spectra

            if hasattr(self.xsec_grid, "unit"):
                interped_spectra = interped_spectra << self.xsec_grid.unit
        return interped_spectra


class ExomolNLTEXsec(ExomolHDF5Xsec):

    def __init__(
            self,
            species: str | SpeciesFormula,
            states_file: pathlib.Path,
            trans_files: pathlib.Path | t.List[pathlib.Path],
            agg_col_nums: t.List[int],
            broadening_params: t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] = None,
            lte_grid_file: pathlib.Path = None,
            cont_states_file: pathlib.Path = None,
            cont_trans_files: pathlib.Path | t.List[pathlib.Path] = None,
            cont_box_length: float = None,
            cont_broad_col_num: int = None,
            dissociation_products: t.Tuple = None,
            load_in_memory: bool = True,
            do_super_lines: bool = False,
            approximate_t_ex: bool = True,
            debug: bool = False,
            debug_pop_matrix: npt.NDArray[np.float64] = None,
    ) -> None:
        self.species = SpeciesFormula(species)
        self.load_in_memory = load_in_memory

        super().__init__(lte_grid_file, self.load_in_memory)

        self.nlte_processor = NLTEProcessor(
            species=species,
            states_file=states_file,
            trans_files=trans_files,
            agg_col_nums=agg_col_nums,
            broadening_params=broadening_params,
            cont_states_file=cont_states_file,
            cont_trans_files=cont_trans_files,
            cont_box_length=cont_box_length,
            cont_broad_col_num=cont_broad_col_num,
            dissociation_products=dissociation_products,
            do_super_lines=do_super_lines,
            approximate_t_ex=approximate_t_ex,
            debug=debug,
            debug_pop_matrix=debug_pop_matrix,
        )

    def get_nlte_processor(self) -> NLTEProcessor:
        return getattr(self, 'nlte_processor', None)

    def configure_layers(self, n_layers: int, n_lte_layers: int = 0) -> None:
        """
        Configures the number of atmospheric layers, and the number fixed to LTE. Instantiates relevant properties on
        the :class:`tiramisu.xsec.NLTEProcessor` instance that depend on the number of layers to iterate over.

        Called when added to a :class:`tiramisu.xsec.XSecCollection` instance.

        Parameters
        ----------
        n_layers : int
            The total number of atmospehric layers.
        n_lte_layers : int
            The number of LTE layers.

        Returns
        -------

        """
        processor = self.get_nlte_processor()
        if processor is None:
            raise RuntimeError(f"NLTEProcessor instance not configured for {self.species} ExomolNLTEXsec instance.")
        processor.n_layers = n_layers
        processor.n_lte_layers = n_lte_layers
        processor.accelerator = HybridAccelerator(n_layers=n_layers - n_lte_layers)

    def opacity(
            self,
            temperature: u.Quantity,
            pressure: u.Quantity,
            spectral_grid: u.Quantity = None,
    ) -> u.Quantity:
        self.nlte_processor.mol_chi_matrix = super().opacity(temperature, pressure, spectral_grid)
        lte_source_fun_matrix = blackbody(spectral_grid=spectral_grid, temperature=temperature)
        self.nlte_processor.mol_eta_matrix = lte_source_fun_matrix * self.nlte_processor.mol_chi_matrix * ac.c

        return self.nlte_processor.mol_chi_matrix


def is_nlte_xsec(xsec_data: XSecData) -> t.TypeGuard['ExomolNLTEXsec']:
    return hasattr(xsec_data, 'get_nlte_processor') and callable(getattr(xsec_data, 'get_nlte_processor'))


class HMinusIon(XSecData):

    def __init__(self):
        super().__init__()

        # --- Bound-free coefficients (Table 2) ---
        self.coef_bf = np.array([152.519, 49.534, -118.858, 92.536, -34.194, 4.982])

        self.lambda_0 = 1.6419  # microns
        self.alpha = 1.439 * 10 ** 8

        # --- Free-free coefficients ---
        # Table 3 (λ > 0.3645 μm)
        self.coef_ff_long = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2483.3460, 285.8270, -2054.2910, 2827.7760, -1341.5370, 208.9520],
            [-3449.8890, -1158.3820, 8746.5230, -11485.6320, 5303.6090, -812.9390],
            [2200.0400, 2427.7190, -13651.1050, 16755.5240, -7510.4940, 1132.7380],
            [-696.2710, -1841.4000, 8642.9700, -10051.5300, 4400.0670, -655.0200],
            [88.2830, 444.5170, -1863.8640, 2095.2880, -901.7880, 132.9850],
        ])

        # Table 4 (0.1823 < λ < 0.3645 μm)
        self.coef_ff_short = np.array([
            [518.1021, -734.8666, 1021.1775, -479.0721, 93.1373, -6.4285],
            [473.2636, 1443.4137, -1977.3395, 922.3575, -178.9275, 12.3600],
            [-482.2089, -737.1616, 1096.8827, -521.1341, 101.7963, -7.0571],
            [115.5291, 169.6374, -245.6490, 114.2430, -21.9972, 1.5097],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])

    def _k_bound_free(self, lam: npt.NDArray[np.float64], temperature: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        John (1988), "Continuous absorption by the negative hydrogen ion reconsidered", via ADS
        https://ui.adsabs.harvard.edu/abs/1988A%26A...193..189J/abstract

        Uses coefficients from Table 2 and Equations 3-5.

        Parameters
        ----------
        lam : ndarray, shape (n_grid, )
            Wavelength values [microns].
        temperature : ndarray, shape (n_layers, )
            Temperature values at each atmospheric layer [Kelvin]

        Returns
        -------
            k_bf : ndarray, shape (n_layers, n_grid)
                Bound-free absorption coefficient [cm^4/dyne].
        """
        sigma = np.zeros_like(lam)
        mask = lam < self.lambda_0

        dif_inv_lambda = (1.0 / lam) - (1.0 / self.lambda_0)

        dif_inv_lambda = np.clip(dif_inv_lambda, 0.0, None)

        n_vals = np.arange(self.coef_bf.shape[0])[:, None]
        john_f = np.sum(
            self.coef_bf[:, None] * dif_inv_lambda ** (n_vals / 2.0), axis=0
        )

        sigma_val = 1e-18 * (lam ** 3) * (dif_inv_lambda ** (3 / 2)) * john_f  # cm^2

        sigma[mask] = sigma_val

        k_bf = (
                0.750
                * temperature ** (-5 / 2)
                * np.exp(self.alpha / (self.lambda_0 * temperature))
                # * (1 - np.exp(-self.alpha / (lam * temperature)))  # Moved outside
                * sigma
        )

        return k_bf

    def _k_free_free(self, lam: npt.NDArray[np.float64], temperature: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        John (1988), "Continuous absorption by the negative hydrogen ion reconsidered", via ADS
        https://ui.adsabs.harvard.edu/abs/1988A%26A...193..189J/abstract

        Uses coefficients from Table 3a and 3b and Equation 6.

        Parameters
        ----------
        lam : ndarray, shape (n_grid, )
            Wavelength values [microns].
        temperature : ndarray, shape (n_layers, )
            Temperature values at each atmospheric layer [Kelvin]

        Returns
        -------
            k_ff : ndarray, shape (n_layers, n_grid)
                Free-free absorption coefficient [cm^4/dyne].
        """
        # Masks for regimes
        mask_long = lam > 0.3645
        mask_short = (lam > 0.1823) & (lam <= 0.3645)
        idx_long = np.where(mask_long[0])[0]
        idx_short = np.where(mask_short[0])[0]

        def eval_region(coefs, mask):
            if not np.any(mask):
                return

            n_vals = np.arange(coefs.shape[0])[:, None] + 2

            a_coef = coefs[:, 0]
            b_coef = coefs[:, 1]
            c_coef = coefs[:, 2]
            d_coef = coefs[:, 3]
            e_coef = coefs[:, 4]
            f_coef = coefs[:, 5]

            k = np.sum(
                ((5040 / temperature) ** (n_vals / 2))
                * (
                        (lam ** 2) * a_coef[:, None]
                        + b_coef[:, None]
                        + c_coef[:, None] / lam
                        + d_coef[:, None] / (lam ** 2)
                        + e_coef[:, None] / (lam ** 3)
                        + f_coef[:, None] / (lam ** 4)
                ),
                axis=0,
            )

            return 10**-29 * k

        k_ff = np.zeros((temperature.shape[0], lam.shape[0]))

        if np.any(mask_long):
            k_ff[:, idx_long] = eval_region(self.coef_ff_long, mask_long)

        if np.any(mask_short):
            k_ff[:, idx_short] = eval_region(self.coef_ff_short, mask_short)

        return k_ff

    def opacity(
            self,
            temperature: u.Quantity,
            pressure: u.Quantity,
            spectral_grid: u.Quantity = None,
            electron_vmr: npt.NDArray[np.float64] = None,
            hydrogen_vmr: npt.NDArray[np.float64] = None,
    ):
        temperature = temperature[:, None].value
        electron_vmr = electron_vmr[:, None]
        hydrogen_vmr = hydrogen_vmr[:, None]
        # Convert grid
        lam = spectral_grid.to(u.um, equivalencies=u.spectral())[None, ...]

        k_bf = self._k_bound_free(lam=lam, temperature=temperature)
        k_ff = self._k_free_free(lam=lam, temperature=temperature)

        stim = 1 - np.exp(-self.alpha / (lam * temperature))

        xsec = (k_bf + k_ff) * pressure.to(u.dyne) * electron_vmr * hydrogen_vmr * stim

        return xsec


class XSecCollection(dict):
    __slots__ = [
        # Inherited:
        "__dict__",
        # Public:
        "n_layers", "intensity_threshold", "n_lte_layers", "incident_radiation_field", "debug",
        # Private:
        "_global_source_func_matrix", "_global_chi_matrix", "_global_eta_matrix", "_intensity_matrix",
        "_is_converged", "_full_prec", "_do_tridiag", "_damping_enabled", "_n_iter",
        "_negative_source_func_cap", "_negative_absorption_factor",
    ]

    def __init__(
            self,
            n_layers: int,
            intensity_threshold: np.float64 = 1e-35,
            n_lte_layers: int = 0,
            incident_radiation_field: u.Quantity = None,
            debug: bool = False,
    ) -> None:
        self.n_layers = n_layers
        self.intensity_threshold = intensity_threshold
        self.n_lte_layers = n_lte_layers
        self.incident_radiation_field = incident_radiation_field
        self.debug = debug
        self._global_source_func_matrix = None
        self._global_chi_matrix = None
        self._global_eta_matrix = None
        self._intensity_matrix = None
        self._is_converged = False
        self._full_prec = True  # Internal debug testing
        self._do_tridiag = True  # Internal debug testing
        self._damping_enabled = False
        self._n_iter = 0
        self._negative_source_func_cap = None
        self._negative_absorption_factor = 0.1

        super().__init__()

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
        if is_nlte_xsec(xsec_data):
            xsec_data.configure_layers(n_layers=self.n_layers, n_lte_layers=self.n_lte_layers)
        self[xsec_data.species] = xsec_data

    def compute_opacities_profile(
            self,
            chem_profile: ChemicalProfile,
            density_profile: u.Quantity,
            dz_profile: u.Quantity,
            temperature: u.Quantity,
            pressure: u.Quantity,
            spectral_grid: u.Quantity,
    ) -> t.Dict[SpeciesFormula, u.Quantity]:
        active_species = self.active_absorbers(chem_profile.species)

        spectral_grid = spectral_grid.to(1 / u.cm, equivalencies=u.spectral())
        wn_dx = np.diff(spectral_grid)

        any_approximate_t_ex = False
        nlte_processors = {}
        for species in active_species:
            xsec_data = self[species]
            if is_nlte_xsec(xsec_data):
                processor = xsec_data.get_nlte_processor()
                nlte_processors[species] = processor
                any_approximate_t_ex |= processor.approximate_t_ex

        output_opacities = {
            species: self[species].opacity(
                temperature, pressure, spectral_grid
            ) * chem_profile[species][:, None]
            for species in active_species
        }

        n_layers = temperature.shape[0]

        # -------------------------- GLOBAL PROPERTY CONFIGURATION --------------------------
        self._global_chi_matrix: u.Quantity = np.zeros((n_layers, spectral_grid.shape[0])) << u.cm ** 2
        self._global_eta_matrix: u.Quantity = np.zeros(self._global_chi_matrix.shape) << u.erg * u.cm / (u.s * u.sr)
        lte_source_func = blackbody(spectral_grid, temperature)

        for species in active_species:
            xsec_data = self[species]
            if is_nlte_xsec(xsec_data):
                log.info(f"[I{self._n_iter}] Initial LTE set up for {species}.")
                processor = xsec_data.get_nlte_processor()
                processor.setup(
                    chem_profile=chem_profile,
                    density_profile=density_profile,
                    temperature_profile=temperature,
                    pressure_profile=pressure,
                    wn_grid=spectral_grid,
                    initial_chi_matrix=output_opacities[species],
                )
                self._global_chi_matrix += processor.mol_chi_matrix
                self._global_eta_matrix += processor.mol_eta_matrix
            else:
                self._global_chi_matrix += output_opacities[species]
                self._global_eta_matrix += output_opacities[species] * lte_source_func * ac.c
                if str(species) == "OH":
                    np.save(
                        fr"/mnt/c/PhD/NLTE/Models/KELT-20b/approximation/ohx1e0_OH_eta.npy",
                        (output_opacities[species] * lte_source_func * ac.c / chem_profile[species][:, None]).value
                    )
                    np.save(
                        fr"/mnt/c/PhD/NLTE/Models/KELT-20b/approximation/ohx1e0_OH_chi.npy",
                        (output_opacities[species] / chem_profile[species][:, None]).value
                    )
        # Units for Emission/Absorption require extra 1/c factor for conversion (ExoCross convention).
        zero_chi_mask = self._global_chi_matrix == 0
        self._global_source_func_matrix: u.Quantity = np.zeros(self._global_eta_matrix.shape) * source_func_unit
        self._global_source_func_matrix[~zero_chi_mask] = (
                self._global_eta_matrix[~zero_chi_mask] / (ac.c * self._global_chi_matrix[~zero_chi_mask])
        )

        if len(nlte_processors) == 0:
            # Early exit when no NLTE species.
            return output_opacities

        n_angular_points = 50
        mu_values, mu_weights = np.polynomial.legendre.leggauss(n_angular_points)
        mu_values, mu_weights = (mu_values + 1) * 0.5, mu_weights / 2

        if any_approximate_t_ex:
            res = self._global_chi_matrix * dz_profile[:, None]
            dtau = res.decompose().value
            i_up, i_down = formal_solve_general(
                dtau=dtau,
                source_function=self._global_source_func_matrix,
                mu_values=mu_values,
                mu_weights=mu_weights,
                incident_radiation_field=self.incident_radiation_field,
            )
            # np.save(r"/mnt/c/PhD/NLTE/theory/opacity/LTE_tau.npy", dtau)
            i_mean_interfaces = (i_up + i_down)
            i_mean = 0.5 * (i_mean_interfaces[:-1] + i_mean_interfaces[1:])
            log.info(f"Any i_mean < 0 = {np.any(i_mean < 0)}; i_mean == 0 = {np.any(i_mean == 0)}.")
            for species in nlte_processors.keys():
                processor = nlte_processors[species]
                log.info(f"[I{self._n_iter}] Approximating T_ex for {species}.")

                if not processor.approximate_t_ex:
                    continue

                if processor.debug_pop_matrix is not None:
                    continue

                processor.compute_approximate_t_ex(
                    i_mean=i_mean,
                    chem_profile=chem_profile,
                    density_profile=density_profile,
                    temperature_profile=temperature,
                    wn_grid=spectral_grid,
                    wn_dx=wn_dx,
                )
                for layer_idx in range(self.n_lte_layers, n_layers):
                    self._global_chi_matrix[layer_idx], self._global_eta_matrix[layer_idx] = (
                        processor.update_layer_global_chi_eta(
                            wn_grid=spectral_grid,
                            layer_vmr=chem_profile[species][layer_idx],
                            layer_global_chi_matrix=self._global_chi_matrix[layer_idx],
                            layer_global_eta_matrix=self._global_eta_matrix[layer_idx],
                            layer_idx=layer_idx,
                            nlte_layer_idx=layer_idx - self.n_lte_layers,
                        )
                    )
            zero_chi_mask = self._global_chi_matrix == 0
            self._global_source_func_matrix[~zero_chi_mask] = (
                    self._global_eta_matrix[~zero_chi_mask] / (ac.c * self._global_chi_matrix[~zero_chi_mask])
            )

        # -------------------------- Iterative solution --------------------------
        while not self._is_converged:
            self._n_iter += 1
            effective_source_func_matrix, effective_tau_mu = effective_source_tau_mu(
                global_source_func_matrix=self._global_source_func_matrix,
                global_chi_matrix=self._global_chi_matrix,
                global_eta_matrix=self._global_eta_matrix,
                density_profile=density_profile,
                dz_profile=dz_profile,
                mu_values=mu_values,
                negative_absorption_factor=self._negative_absorption_factor,
            )
            # np.save(fr"/mnt/c/PhD/NLTE/theory/opacity/nLTE_tau_I{self.n_iter}.npy", effective_tau_mu)
            start_time = time.perf_counter()
            bezier_coefs, control_points = bezier_coefficients(
                tau_mu_matrix=effective_tau_mu,
                source_function_matrix=effective_source_func_matrix,
            )
            log.info(f"Coefficient duration = {time.perf_counter() - start_time:.3f}")
            # log.info(
            #     f"Coefs equal? {np.all(bezier_coefs_old == bezier_coefs)} {np.allclose(bezier_coefs_old, bezier_coefs, atol=1e-7)}"
            # )
            # log.info(
            #     f"Control equal? {np.all(control_points_old == control_points)} {np.allclose(control_points_old, control_points, atol=1e-7)}"
            # )

            # USEFUL BEZIER IDENTITIES
            alpha_plus_gamma = bezier_coefs[:, 1] + bezier_coefs[:, 3]
            one_plus_exp_neg_delta_tau = 1 + bezier_coefs[:, 0]

            i_in_matrix: u.Quantity = np.zeros_like(effective_tau_mu) << self._global_source_func_matrix.unit
            lambda_in_matrix = np.zeros_like(effective_tau_mu)
            ################
            pop_grid_updates = {}
            for species in nlte_processors.keys():
                # xsec_data = self[species]
                # if is_nlte_xsec(xsec_data):
                #     processor = xsec_data.get_nlte_processor()
                processor = nlte_processors[species]
                pop_grid_updates[species] = processor.get_latest_pop_grid()

            # -------------------------- GAUSS-SEIDEL PASSES --------------------------
            # ------------------------------ INWARD PASS ------------------------------
            # Upper boundary condition if incident radiation field present.
            if self.incident_radiation_field is not None:
                i_in_matrix[-1] = self.incident_radiation_field

            for layer_idx in range(n_layers - 1)[::-1]:
                # Inward intensity interpolation.
                i_in_matrix[layer_idx] = (
                        i_in_matrix[layer_idx + 1] * bezier_coefs[layer_idx + 1, 0] +
                        bezier_coefs[layer_idx + 1, 1] * effective_source_func_matrix[layer_idx, None, :] +
                        bezier_coefs[layer_idx + 1, 2] * effective_source_func_matrix[layer_idx + 1, None, :] +
                        bezier_coefs[layer_idx + 1, 3] * control_points[layer_idx, 1] * source_func_unit
                )

                # Inwards Lambda operator calculation.
                if self._do_tridiag and layer_idx > 0:
                    lambda_in_matrix[layer_idx] = (
                            alpha_plus_gamma[layer_idx + 1] * one_plus_exp_neg_delta_tau[layer_idx] +
                            bezier_coefs[layer_idx, 2]
                    )
                else:
                    lambda_in_matrix[layer_idx] = alpha_plus_gamma[layer_idx + 1]
            # ------------------------------ OUTWARD PASS ------------------------------
            i_out_matrix: u.Quantity = np.zeros_like(effective_tau_mu) << source_func_unit
            lambda_out_matrix = np.zeros_like(effective_tau_mu)

            for layer_idx in range(n_layers):
                if layer_idx == 0:
                    i_out_matrix[layer_idx] = blackbody(spectral_grid=spectral_grid, temperature=temperature[0])[0]
                else:
                    # Outward intensity interpolation.
                    i_out_matrix[layer_idx] = (
                            i_out_matrix[layer_idx - 1] * bezier_coefs[layer_idx, 0] +
                            bezier_coefs[layer_idx, 1] * effective_source_func_matrix[layer_idx, None, :] +
                            bezier_coefs[layer_idx, 2] * effective_source_func_matrix[layer_idx - 1, None, :] +
                            bezier_coefs[layer_idx, 3] * control_points[layer_idx, 0] * source_func_unit
                    )
                    # Outward Lambda operator calculation.
                    if self._do_tridiag and layer_idx < n_layers - 1:
                        lambda_out_matrix[layer_idx] = (
                                alpha_plus_gamma[layer_idx] * one_plus_exp_neg_delta_tau[layer_idx + 1] +
                                bezier_coefs[layer_idx + 1, 2]
                        )
                    else:
                        lambda_out_matrix[layer_idx] = alpha_plus_gamma[layer_idx]

                    # INWARD UPDATES (during outward pass)
                    if layer_idx < n_layers - 1:
                        i_in_matrix[layer_idx] = (
                                i_in_matrix[layer_idx + 1] * bezier_coefs[layer_idx + 1, 0] +
                                bezier_coefs[layer_idx + 1, 1] * effective_source_func_matrix[layer_idx, None, :] +
                                bezier_coefs[layer_idx + 1, 2] * effective_source_func_matrix[layer_idx + 1, None, :] +
                                bezier_coefs[layer_idx + 1, 3] * control_points[layer_idx, 1] * source_func_unit
                        )

                        if self._do_tridiag and layer_idx > 0:
                            lambda_in_matrix[layer_idx] = (
                                    alpha_plus_gamma[layer_idx + 1] * one_plus_exp_neg_delta_tau[layer_idx] +
                                    bezier_coefs[layer_idx, 2]
                            )
                        else:
                            lambda_in_matrix[layer_idx] = alpha_plus_gamma[layer_idx + 1]
                # Solve equilibrium for non-LTE layers.
                if layer_idx >= self.n_lte_layers:
                    nlte_layer_idx = layer_idx - self.n_lte_layers
                    # layer_temp = temperature[layer_idx]
                    # layer_pressure = pressure[layer_idx]

                    # Integrate over all angles. This can be done independent of the transitions.
                    i_layer_grid = 0.5 * np.sum(
                        (i_in_matrix[layer_idx] + i_out_matrix[layer_idx]) * mu_weights[:, None],
                        axis=0,
                    )
                    lambda_layer_grid = 0.5 * np.sum(
                        (lambda_in_matrix[layer_idx] + lambda_out_matrix[layer_idx]) * mu_weights[:, None],
                        axis=0,
                    )
                    y_mats = {}
                    rhs_mats = {}
                    # TODO: skip if species abundance is 0?
                    for species in nlte_processors.keys():
                        # xsec_data = self[species]
                        # if is_nlte_xsec(xsec_data):
                        #     processor = xsec_data.get_nlte_processor()
                        processor = nlte_processors[species]
                        y_mats[species], rhs_mats[species] = processor.build_y_matrix(
                            layer_idx=layer_idx,
                            nlte_layer_idx=nlte_layer_idx,
                            i_layer_grid=i_layer_grid,
                            lambda_layer_grid=lambda_layer_grid,
                            chem_profile=chem_profile,
                            global_chi_matrix=self._global_chi_matrix,
                            global_source_func_matrix=self._global_source_func_matrix,
                            wn_grid=spectral_grid,
                            wn_dx=wn_dx,
                            full_prec=self._full_prec,
                        )
                    # Solve statistical equilibrium for all species and update layer opacities, etc.
                    # These are solved in another loop so that all Y matrices are constructed using the same set of
                    # global parameters, rather than biasing the solution each iteration based on update order.
                    for species in nlte_processors.keys():
                        # xsec_data = self[species]
                        # if is_nlte_xsec(xsec_data):
                        #     processor = xsec_data.get_nlte_processor()
                        processor = nlte_processors[species]
                        pop_grid_update = processor.solve_pops(
                            y_matrix=y_mats[species],
                            rhs_matrix=rhs_mats[species],
                            pop_grid_update=pop_grid_updates[species],
                            layer_idx=layer_idx,
                            n_iter=self._n_iter,
                        )
                        pop_grid_updates[species] = pop_grid_update

                        self._global_chi_matrix[layer_idx], self._global_eta_matrix[layer_idx] = (
                            processor.update_layer_global_chi_eta(
                                wn_grid=spectral_grid,
                                layer_vmr=chem_profile[species][layer_idx],
                                layer_global_chi_matrix=self._global_chi_matrix[layer_idx],
                                layer_global_eta_matrix=self._global_eta_matrix[layer_idx],
                                layer_idx=layer_idx,
                                nlte_layer_idx=nlte_layer_idx,
                                layer_pop_grid=pop_grid_update[layer_idx],
                            )
                        )
                    #########
                    # Update all physical properties now all Non-LTE species' opacities have been updated.
                    self._global_source_func_matrix[layer_idx] = (
                            self._global_eta_matrix[layer_idx]
                            # * density_profile[layer_idx]  # No longer baking density into global Chi!
                            / (ac.c * self._global_chi_matrix[layer_idx])
                    ).to(source_func_unit, equivalencies=u.spectral())

                    effective_source_func_matrix, effective_tau_mu = effective_source_tau_mu(
                        global_source_func_matrix=self._global_source_func_matrix,
                        global_chi_matrix=self._global_chi_matrix,
                        global_eta_matrix=self._global_eta_matrix,
                        density_profile=density_profile,
                        dz_profile=dz_profile,
                        mu_values=mu_values,
                        negative_absorption_factor=self._negative_absorption_factor,
                    )
                    # np.save(fr"/mnt/c/PhD/NLTE/theory/opacity/nLTE_tau_I{self.n_iter}_L{nlte_layer_idx}.npy",
                    #         effective_tau_mu)
                    # start_time = time.perf_counter()
                    # bezier_coefs_old, control_points_old = bezier_coefficients_old(
                    #     tau_mu_matrix=effective_tau_mu,
                    #     source_function_matrix=effective_source_func_matrix.value,
                    # )
                    # control_points_old = control_points_old << source_func_unit
                    # log.info(f"[L{layer_idx}] (OLD) Coefficient (post) duration = {time.perf_counter() - start_time}")
                    # log.info(f"Test control points before = {control_points_old[layer_idx, 0]}")
                    # check_old_control_points = control_points[layer_idx - 1: layer_idx + 2, 0].copy()
                    # log.info(f"Test control points before = {check_old_control_points}")
                    start_time = time.perf_counter()
                    update_layer_coefficients(
                        layer_idx=layer_idx,
                        tau_mu_matrix=effective_tau_mu,
                        source_function_matrix=effective_source_func_matrix.value,
                        coefficients=bezier_coefs,
                        control_points=control_points
                    )
                    # log.info(f"TEST control points after = {control_points[layer_idx - 1: layer_idx + 2, 0]}")
                    # log.info(f"TEST Still the same? {np.all(control_points[layer_idx - 1: layer_idx + 2, 0] == check_old_control_points)}")
                    # control_points = control_points << source_func_unit
                    if layer_idx > 0:
                        one_plus_exp_neg_delta_tau[layer_idx] = 1 + bezier_coefs[layer_idx, 0]
                        alpha_plus_gamma[layer_idx] = bezier_coefs[layer_idx, 1] + bezier_coefs[layer_idx, 3]
                    if layer_idx < n_layers - 1:
                        one_plus_exp_neg_delta_tau[layer_idx + 1] = 1 + bezier_coefs[layer_idx + 1, 0]
                        alpha_plus_gamma[layer_idx + 1] = bezier_coefs[layer_idx + 1, 1] + bezier_coefs[
                            layer_idx + 1, 3]

                    log.info(f"[L{layer_idx}] Coefficient update duration = {time.perf_counter() - start_time:.3f}")

                    if layer_idx > 0:
                        i_out_matrix[layer_idx] = (
                                i_out_matrix[layer_idx - 1] * bezier_coefs[layer_idx, 0] +
                                bezier_coefs[layer_idx, 1] * effective_source_func_matrix[layer_idx, None, :] +
                                bezier_coefs[layer_idx, 2] * effective_source_func_matrix[layer_idx - 1, None, :] +
                                bezier_coefs[layer_idx, 3] * control_points[layer_idx, 0] * source_func_unit
                        )
                        if self._do_tridiag and layer_idx < n_layers - 1:
                            lambda_out_matrix[layer_idx] = (
                                    alpha_plus_gamma[layer_idx] * one_plus_exp_neg_delta_tau[layer_idx + 1] +
                                    bezier_coefs[layer_idx + 1, 2]
                            )
                        else:
                            lambda_out_matrix[layer_idx] = alpha_plus_gamma[layer_idx]
            # ---------------------------- PASSES COMPLETE ----------------------------
            # Commit pop_grid_updates to each species.
            all_converged = True
            for species in nlte_processors.keys():
                # xsec_data = self[species]
                # if is_nlte_xsec(xsec_data):
                #     processor = xsec_data.get_nlte_processor()
                processor = nlte_processors[species]
                converged = processor.update_pops(
                    pop_grid_updated=pop_grid_updates[species],
                    n_iter=self._n_iter
                )
                all_converged &= converged

            self._is_converged = all_converged
        log.info(f"[I{self._n_iter}] Convergence achieved!")

        # TODO: Default resolving power grid?
        high_res_grid = create_r_wn_grid(low=spectral_grid[0].value, high=spectral_grid[-1].value,
                                         resolving_power=15000)
        for species in active_species:
            xsec_data = self[species]
            if is_nlte_xsec(xsec_data):
                processor = xsec_data.get_nlte_processor()
                xsec_data.opacity(temperature, pressure, high_res_grid)
                processor.finalise(temperature_profile=temperature, pressure_profile=pressure, wn_grid=high_res_grid)
                output_opacities[species] = processor.mol_chi_matrix
            else:
                output_opacities[species] = xsec_data.opacity(temperature, pressure, high_res_grid)

        return output_opacities

    @property
    def unified_grid(self) -> u.Quantity:
        res = [x.spectral_grid for x in self.values()]
        base_unit = res[0].unit
        res = np.concatenate([x.to(base_unit, equivalencies=u.spectral()).value for x in res])
        res = np.sort(np.unique(res)) << base_unit

        return res


def create_r_wn_grid(low: float, high: float, resolving_power: float) -> u.Quantity:
    resolving_f = np.log((resolving_power + 1) / resolving_power)
    n_points = round((np.log(high) - np.log(low)) / resolving_f) + 1
    return np.exp(np.arange(n_points) * resolving_f + np.log(low)) << u.k
