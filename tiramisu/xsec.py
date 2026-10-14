import logging
import time
import abc
import pathlib
import typing as t

import numpy.typing as npt
import numpy as np

import astropy.units as u
import astropy.constants as ac

from .accelerator import MultiSpeciesAccelerator
from .chemistry import SpeciesFormula, SpeciesIdentType, ChemicalProfile
from .nlte import (
    blackbody,
    effective_source_tau_mu,
    bezier_delta_tau_and_control_points,
    update_layer_coefficients,
    update_inward_layer,
    update_outward_layer,
    four_pi_j_hemispheres,
    four_pi_j_spherical,
    mu_quadrature,
    spherical_rays,
    update_inward_layer_spherical,
    update_outward_layer_spherical,
    NLTEProcessor,
)
from .config import _VMR_THRESHOLD, _LOG_FLOAT_FMT, _LOG_ARRAY_FMT, _LOG_VERBOSE_2
from .numerics import linear_loglinear_interp_2d

log = logging.getLogger(__name__)

# Units
source_func_unit = u.J / (u.sr * u.m ** 2)


def _bracket(
        values: npt.NDArray[np.float64], coord: npt.NDArray[np.float64]
) -> t.Tuple[npt.NDArray[np.int32], npt.NDArray[np.float64]]:
    """Lower bracketing index and fractional weight of values on an ascending coord grid, held at the edges."""
    idx1 = coord.searchsorted(values, side="right").clip(1, len(coord) - 1)
    idx0 = idx1 - 1
    values = values.clip(coord[0], coord[-1])
    weight = (values - coord[idx0]) / (coord[idx1] - coord[idx0])
    return idx0.astype(np.int32), weight


def interpolate_pt_xsec(
        data: t.Union[u.Quantity, npt.NDArray[np.float64], t.Any],
        pressure: u.Quantity,
        temperature: u.Quantity,
        pressure_grid: u.Quantity,
        temperature_grid: u.Quantity,
        axes: t.Tuple[int, int] = (0, 1),
) -> t.Union[u.Quantity, npt.NDArray[np.float64]]:
    """Interpolate a tabulated cross-section grid to a set of (pressure, temperature) points.

    Interpolation is linear in pressure and piecewise log-linear in temperature (log σ linear in T, falling back to
    linear where a node is non-positive); see :func:`numerics.linear_loglinear_interp_2d`. Points outside the grid are
    held at the nearest edge.

    Only the sub-block of ``data`` bracketing the requested points is read, so ``data`` may be a lazily-loaded array
    such as an ``h5py.Dataset``.

    Args:
        data: Cross-section table with pressure, temperature and spectral axes.
        pressure: Pressures to interpolate to.
        temperature: Temperatures to interpolate to, same shape as ``pressure``.
        pressure_grid: Ascending pressure coordinates of ``data``.
        temperature_grid: Ascending temperature coordinates of ``data``.
        axes: Axes of ``data`` holding (pressure, temperature); the remaining axis is spectral.

    Returns:
        Interpolated cross-sections, shape ``(*pressure.shape, n_spectral)``, with the units of ``data`` if any.

    Raises:
        ValueError: If data is not 3D or pressure and temperature shapes differ.

    """
    if data.ndim != 3:
        raise ValueError("Data must be 3D (pressure, temperature, spectral).")
    pressure = u.Quantity(pressure, pressure_grid.unit)
    temperature = u.Quantity(temperature, temperature_grid.unit)
    if pressure.shape != temperature.shape:
        raise ValueError(f"Pressure shape {pressure.shape} does not match temperature shape {temperature.shape}.")

    p_coord = np.asarray(pressure_grid.value, dtype=np.float64)
    t_coord = np.asarray(temperature_grid.value, dtype=np.float64)
    idx_p0, w_p = _bracket(np.atleast_1d(pressure.value).ravel().astype(np.float64), p_coord)
    idx_t0, w_t = _bracket(np.atleast_1d(temperature.value).ravel().astype(np.float64), t_coord)

    # In-memory tables are used as-is (a view); lazy datasets (h5py) only read the bracketing sub-block.
    p_lo, t_lo = 0, 0
    if isinstance(data, np.ndarray):
        block = data
    else:
        p_lo, p_hi = idx_p0.min(), idx_p0.max() + 2
        t_lo, t_hi = idx_t0.min(), idx_t0.max() + 2
        block_slice = [slice(None)] * 3
        block_slice[axes[0]] = slice(p_lo, p_hi)
        block_slice[axes[1]] = slice(t_lo, t_hi)
        block = data[tuple(block_slice)]
    unit = getattr(block, "unit", None)
    block = np.asarray(getattr(block, "value", block), dtype=np.float64)
    # Kernel expects (pressure, temperature, spectral); moveaxis returns a strided view, which numba accepts.
    spectral_axis = 3 - axes[0] - axes[1]
    block = np.moveaxis(block, (axes[0], axes[1], spectral_axis), (0, 1, 2))

    result = linear_loglinear_interp_2d(block, idx_p0 - p_lo, w_p, idx_t0 - t_lo, w_t)
    result = result.reshape(*pressure.shape, block.shape[2])
    if unit is not None:
        result = result << unit
    return result


def weight_broadening_parameters(
        broadening_dict: t.Dict, chemistry_profile
) -> t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    broad_n = []
    broad_gamma = []
    for species in broadening_dict.keys():
        if species in chemistry_profile.species:
            species_vmr = chemistry_profile[species]
            species_broad = broadening_dict[species]
            broad_gamma.append(species_broad[0] * species_vmr)
            broad_n.append(species_broad[1])
        else:
            log.warning(f"Broadening parameters for {species} provided but not present in chemistry_profile.")
    return np.array(broad_gamma), np.array(broad_n)


class XSecData(abc.ABC):
    # Whether opacity() is already weighted by the mixing ratios it depends on (per particle), rather than per molecule
    # of the species; XSecCollection multiplies only unweighted opacities by the species' mixing ratio.
    abundance_weighted: bool = False
    # Native spectral grid, if the opacity has one.
    spectral_grid: u.Quantity | None = None

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
        return interpolate_pt_xsec(
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
                np.asarray(interped_spectra),  # a Quantity, or a plain array from a lazy h5py table
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

        super().__init__(formula, spectral_grid, xsec_grid, temp_grid, pressure_grid, (0, 1))

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
            do_super_lines: bool = True,
            do_initial_step: bool = True,
            debug: bool = False,
            debug_pop_matrix: npt.NDArray[np.float64] = None,
            save_rates_profiles: bool = False,
            rates_pickle: pathlib.Path | None = None,
            profile_pickle: pathlib.Path | None = None,
            cont_rates_pickle: pathlib.Path | None = None,
            cont_profile_pickle: pathlib.Path | None = None,
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
            do_initial_step=do_initial_step,
            debug=debug,
            debug_pop_matrix=debug_pop_matrix,
            save_rates_profiles=save_rates_profiles,
            rates_pickle=rates_pickle,
            profile_pickle=profile_pickle,
            cont_rates_pickle=cont_rates_pickle,
            cont_profile_pickle=cont_profile_pickle,
        )

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
        self.nlte_processor.n_layers = n_layers
        self.nlte_processor.n_lte_layers = n_lte_layers

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


class HMinusIon(XSecData):
    """
    H- continuum opacity from John (1988), "Continuous absorption by the negative hydrogen ion reconsidered",
    https://ui.adsabs.harvard.edu/abs/1988A%26A...193..189J/abstract, weighted by explicit H-, e- and H mixing ratios
    as in Goyal et al. (2020), MNRAS 498, 4680 (appendix A):

    .. math::
        \\chi = \\sigma_{bf}(\\lambda) \\left(1 - e^{-\\alpha / \\lambda T}\\right) x_{H^-}
            + k_{ff}(\\lambda, T) \\, P \\, x_{e^-} x_{H},

    per particle [cm^2], for bound-free photodetachment (H- + hv -> H + e-) and free-free absorption
    (e- + H + hv -> e- + H), with :math:`\\alpha = hc/k`. John's free-free fit already includes stimulated emission.

    The opacity is already weighted by every mixing ratio it depends on (``abundance_weighted``), so
    :class:`XSecCollection` does not multiply it by the H- mixing ratio again. It is keyed on H-, which must be in the
    chemical profile for it to be used.

    Parameters
    ----------
    h_minus_vmr, electron_vmr, hydrogen_vmr : ndarray, shape (n_layers,)
        Mixing ratios of H-, free electrons and neutral atomic H in each layer.
    spectral_grid : Quantity, optional
        Default grid for :meth:`opacity`, and this continuum's contribution to
        :meth:`XSecCollection.unified_grid` (an analytic continuum has no native grid, so None contributes nothing).
    """
    abundance_weighted = True
    # hc/k [um K] and the photodetachment threshold [um] (John 1988).
    alpha = 1.439e4
    lambda_0 = 1.6419
    # Bound-free coefficients C_n, n = 1..6 (John 1988, Table 2).
    coef_bf = np.array([152.519, 49.534, -118.858, 92.536, -34.194, 4.982])
    # Free-free coefficients (A_n, B_n, C_n, D_n, E_n, F_n), n = 1..6 (John 1988, Table 3a: lambda > 0.3645 um). C_5 is
    # 8624.970: the 8642.970 this table was copied with (as in TauREx 3.3.2) makes Tables 3a and 3b disagree by 1.1-11.6x
    # at 0.3645 um over 1000-10080 K; with 8624.970 they agree to 0.4% (5.9% at 1000 K) - tests/check_hminus.py.
    coef_ff_long = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [2483.3460, 285.8270, -2054.2910, 2827.7760, -1341.5370, 208.9520],
        [-3449.8890, -1158.3820, 8746.5230, -11485.6320, 5303.6090, -812.9390],
        [2200.0400, 2427.7190, -13651.1050, 16755.5240, -7510.4940, 1132.7380],
        [-696.2710, -1841.4000, 8624.9700, -10051.5300, 4400.0670, -655.0200],
        [88.2830, 444.5170, -1863.8640, 2095.2880, -901.7880, 132.9850],
    ])
    # Table 3b: 0.1823 < lambda < 0.3645 um.
    coef_ff_short = np.array([
        [518.1021, -734.8666, 1021.1775, -479.0721, 93.1373, -6.4285],
        [473.2636, 1443.4137, -1977.3395, 922.3575, -178.9275, 12.3600],
        [-482.2089, -737.1616, 1096.8827, -521.1341, 101.7963, -7.0571],
        [115.5291, 169.6374, -245.6490, 114.2430, -21.9972, 1.5097],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])

    def __init__(
            self,
            h_minus_vmr: npt.NDArray[np.float64],
            electron_vmr: npt.NDArray[np.float64],
            hydrogen_vmr: npt.NDArray[np.float64],
            spectral_grid: u.Quantity | None = None,
    ):
        super().__init__()
        self.species = SpeciesFormula("H-")
        self.spectral_grid = spectral_grid
        # (n_layers, 1), to broadcast against the (n_layers, n_grid) coefficients.
        self.h_minus_vmr = np.asarray(h_minus_vmr, dtype=np.float64).reshape(-1, 1)
        self.electron_vmr = np.asarray(electron_vmr, dtype=np.float64).reshape(-1, 1)
        self.hydrogen_vmr = np.asarray(hydrogen_vmr, dtype=np.float64).reshape(-1, 1)

    def _sigma_bound_free(self, lam: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Photodetachment cross-section (John 1988, equations 4-5), zero beyond the threshold lambda_0.

        Parameters
        ----------
        lam : ndarray, shape (n_grid,)
            Wavelength [microns].

        Returns
        -------
        sigma_bf : ndarray, shape (n_grid,)
            Bound-free cross-section [cm^2 per H- ion].
        """
        # Zero beyond the threshold, which zeroes sigma_bf there through the (1/lambda - 1/lambda_0)^(3/2) factor.
        inv_lambda_dif = np.clip(1.0 / lam - 1.0 / self.lambda_0, 0.0, None)
        n_vals = np.arange(1, self.coef_bf.shape[0] + 1)[:, None]
        john_f = np.sum(self.coef_bf[:, None] * inv_lambda_dif[None, :] ** ((n_vals - 1) / 2.0), axis=0)
        return 1e-18 * lam ** 3 * inv_lambda_dif ** 1.5 * john_f

    def _k_free_free(self, lam: npt.NDArray[np.float64], temperature: npt.NDArray[np.float64]) -> npt.NDArray[
        np.float64]:
        """
        Free-free absorption coefficient (John 1988, equation 6 with Tables 3a and 3b), zero below 0.1823 microns.
        Includes stimulated emission.

        Parameters
        ----------
        lam : ndarray, shape (n_grid,)
            Wavelength [microns].
        temperature : ndarray, shape (n_layers,)
            Temperature of each layer [Kelvin].

        Returns
        -------
        k_ff : ndarray, shape (n_layers, n_grid)
            Free-free absorption coefficient [cm^4/dyne per H atom per unit electron pressure].
        """
        k_ff = np.zeros((temperature.shape[0], lam.shape[0]))
        # (5040 / T)^((n + 1) / 2) for n = 1..6, shape (6, n_layers).
        n_vals = np.arange(1, self.coef_ff_long.shape[0] + 1)[:, None]
        theta_terms = (5040.0 / temperature[None, :]) ** ((n_vals + 1) / 2.0)
        for coefs, region in (
                (self.coef_ff_long, lam > 0.3645),
                (self.coef_ff_short, (lam > 0.1823) & (lam <= 0.3645)),
        ):
            lam_r = lam[region][None, :]
            # lambda^2 A_n + B_n + C_n / lambda + D_n / lambda^2 + E_n / lambda^3 + F_n / lambda^4, shape (6, n_region).
            lam_terms = (
                    coefs[:, 0:1] * lam_r ** 2 + coefs[:, 1:2] + coefs[:, 2:3] / lam_r + coefs[:, 3:4] / lam_r ** 2
                    + coefs[:, 4:5] / lam_r ** 3 + coefs[:, 5:6] / lam_r ** 4
            )
            k_ff[:, region] = 1e-29 * (theta_terms.T @ lam_terms)
        return k_ff

    def opacity(
            self,
            temperature: u.Quantity,
            pressure: u.Quantity,
            spectral_grid: u.Quantity = None,
    ) -> u.Quantity:
        """H- opacity per particle, shape (n_layers, n_grid) [cm^2], weighted by the H-, e- and H mixing ratios."""
        spectral_grid = self.spectral_grid if spectral_grid is None else spectral_grid
        if spectral_grid is None:
            raise ValueError("HMinusIon.opacity needs a spectral_grid (none given here or at construction).")
        lam = spectral_grid.to_value(u.um, equivalencies=u.spectral())
        temp = temperature.to_value(u.K)
        stim = 1.0 - np.exp(-self.alpha / (temp[:, None] * lam[None, :]))
        chi_bf = self._sigma_bound_free(lam=lam)[None, :] * stim * self.h_minus_vmr
        # [cm^4/dyne] x [dyne/cm^2] = [cm^2].
        electron_pressure = pressure.to_value(u.dyne / u.cm ** 2)[:, None] * self.electron_vmr
        chi_ff = self._k_free_free(lam=lam, temperature=temp) * electron_pressure * self.hydrogen_vmr
        return (chi_bf + chi_ff) << u.cm ** 2


class XSecCollection(dict):
    __slots__ = [
        # Inherited:
        "__dict__",
        # Public:
        "n_layers", "n_lte_layers", "incident_radiation_field", "debug",
        # Private:
        "accelerator", "_global_source_func_matrix", "_global_chi_matrix", "_global_eta_matrix",
        "_is_converged", "_full_prec", "_n_iter", "_effective_chi_matrix",
    ]

    def __init__(
            self,
            n_layers: int,
            n_lte_layers: int = 0,
            incident_radiation_field: t.Union[u.Quantity, t.Callable[..., u.Quantity], None] = None,
            debug: bool = False,
    ) -> None:
        """
        ``incident_radiation_field`` is the stellar irradiation from :func:`tiramisu.nlte.incident_intensity`, which is
        evaluated on the solver's own directions in either geometry. An array of the downward specific intensity on
        the plane-parallel grid (:func:`tiramisu.nlte.mu_quadrature`) is also accepted, for plane-parallel runs.
        """
        self.n_layers = n_layers
        self.n_lte_layers = n_lte_layers
        self.incident_radiation_field = incident_radiation_field
        self.debug = debug
        self.accelerator: MultiSpeciesAccelerator | None = None
        self._global_source_func_matrix = None
        self._global_chi_matrix = None
        self._global_eta_matrix = None
        self._is_converged = False
        self._full_prec = True  # Internal debug testing
        self._n_iter = 0
        # Shielded (always non-negative) counterpart of _global_chi_matrix, refreshed by every
        # effective_source_tau_mu() call in compute_opacities_profile - see build_y_matrix's docstring for why this,
        # and not _global_chi_matrix directly, must be used for its Lambda*-operator terms.
        self._effective_chi_matrix = None

        super().__init__()

    def __getitem__(self, key: SpeciesIdentType) -> XSecData:
        if isinstance(key, str):
            key = SpeciesFormula(key)
        return super().__getitem__(key)

    def __setitem__(self, key: SpeciesIdentType, value: XSecData) -> None:
        if isinstance(key, str):
            key = SpeciesFormula(key)
        return super().__setitem__(key, value)

    def active_absorbers(self, species_list: t.List[SpeciesFormula]) -> t.Set[SpeciesFormula]:
        return set(species_list) & self.keys()

    def add_replace_xsec_data(self, xsec_data: XSecData) -> None:
        if isinstance(xsec_data, ExomolNLTEXsec):
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
            interface_radii: t.Optional[u.Quantity] = None,
    ) -> t.Dict[SpeciesFormula, u.Quantity]:
        """
        Opacities per species, with any NLTE species' populations solved first. The radiative transfer for the NLTE
        solution is plane-parallel, or through spherical shells (:func:`tiramisu.nlte.spherical_rays`) when
        ``interface_radii`` (layer interface radii, bottom first) is given.
        """
        active_species = self.active_absorbers(chem_profile.species)

        spectral_grid = spectral_grid.to(1 / u.cm, equivalencies=u.spectral())
        wn_dx = np.diff(spectral_grid)

        any_initial_step = False
        nlte_processors = {}
        for species in active_species:
            xsec_data = self[species]
            if isinstance(xsec_data, ExomolNLTEXsec):
                processor = xsec_data.nlte_processor
                nlte_processors[species] = processor
                any_initial_step |= processor.do_initial_step

        output_opacities = {}
        for species in active_species:
            species_opacity = self[species].opacity(temperature, pressure, spectral_grid)
            if not self[species].abundance_weighted:
                species_opacity = species_opacity * chem_profile[species][:, None]
            output_opacities[species] = species_opacity

        n_layers = temperature.shape[0]

        # -------------------------- GLOBAL PROPERTY CONFIGURATION --------------------------
        self._global_chi_matrix: u.Quantity = np.zeros((n_layers, spectral_grid.shape[0])) << u.cm ** 2
        self._global_eta_matrix: u.Quantity = np.zeros(self._global_chi_matrix.shape) << u.erg * u.cm / (u.s * u.sr)
        lte_source_func = blackbody(spectral_grid, temperature)

        for species in active_species:
            if species in nlte_processors:
                log.info(f"Initial LTE set up for {species}.")
                processor = nlte_processors[species]
                processor.setup(
                    chem_profile=chem_profile,
                    density_profile=density_profile,
                    temperature_profile=temperature,
                    pressure_profile=pressure,
                    wn_grid=spectral_grid,
                    initial_chi_matrix=output_opacities[species],
                )
                # Setup removes the VMR baked into output_opacities, add back here.
                self._global_chi_matrix += processor.mol_chi_matrix * chem_profile[species][:, None]
                self._global_eta_matrix += processor.mol_eta_matrix * chem_profile[species][:, None]
            else:
                self._global_chi_matrix += output_opacities[species]
                self._global_eta_matrix += output_opacities[species] * lte_source_func * ac.c
        # Units for Emission/Absorption require extra 1/c factor for conversion (ExoCross convention).
        zero_chi_mask = self._global_chi_matrix == 0
        self._global_source_func_matrix: u.Quantity = np.zeros(self._global_eta_matrix.shape) * source_func_unit
        self._global_source_func_matrix[~zero_chi_mask] = (
                self._global_eta_matrix[~zero_chi_mask] / (ac.c * self._global_chi_matrix[~zero_chi_mask])
        )

        if len(nlte_processors) == 0:
            # Early exit when no NLTE species.
            return output_opacities

        # ----------------------------------------------- NLTE WORKFLOW -----------------------------------------------
        # Configure species accelerator.
        n_nlte_layers = self.n_layers - self.n_lte_layers
        self.accelerator = MultiSpeciesAccelerator(
            species_layers={species: n_nlte_layers for species in nlte_processors.keys()}
        )
        all_debug_pop_matrix = []
        for species, processor in nlte_processors.items():
            all_debug_pop_matrix.append(processor.debug_pop_matrix is not None)
            for layer_idx in range(self.n_lte_layers, n_layers):
                nlte_layer_idx = layer_idx - self.n_lte_layers
                if chem_profile[species][layer_idx] <= _VMR_THRESHOLD:
                    self.accelerator.mark_skipped(species=species, layer_idx=nlte_layer_idx)

        if all(all_debug_pop_matrix):
            log.info(f"All species provided with debugging population matrices; skipping to finalise.")
            # All species have debugging population matrices; skip iteration and set converged.
            for species, processor in nlte_processors.items():
                processor.set_pop_matrix_to_debug()
            self._is_converged = True

        # NLTE layers' opacity and emissivity from each processor's profile store at its starting populations (LTE, or
        # the debug matrix), replacing the tabulated cross-sections from setup there. The iteration's rates and Lambda*
        # terms are built from the store, while the tabulated grid is interpolated from high-resolution data (missing
        # line centres) over its own wavelength range, so it differs from the store bin by bin. Starting from it let a
        # band's chi exceed the global chi it is divided by, so the preconditioning removed more than the radiative
        # rate and the first solves went negative (KELT-20b, OH x100, without the initial step).
        for species, processor in nlte_processors.items():
            for layer_idx in range(self.n_lte_layers, n_layers):
                processor.update_layer_global_chi_eta(
                    wn_grid=spectral_grid,
                    layer_vmr=chem_profile[species][layer_idx],
                    layer_global_chi_matrix=self._global_chi_matrix[layer_idx],
                    layer_global_eta_matrix=self._global_eta_matrix[layer_idx],
                    layer_idx=layer_idx,
                    nlte_layer_idx=layer_idx - self.n_lte_layers,
                )
        zero_chi_mask = self._global_chi_matrix == 0
        self._global_source_func_matrix[~zero_chi_mask] = (
                self._global_eta_matrix[~zero_chi_mask] / (ac.c * self._global_chi_matrix[~zero_chi_mask])
        )

        # Directions: plane-parallel, every layer sees the same mu nodes and weights; spherical, rays through the shells
        # reach each layer at their own angles, with per-layer weights (zero for rays not reaching it).
        mu_values, mu_weights = mu_quadrature()
        if interface_radii is None:
            rays = None
            n_angular_points = len(mu_values)
            ray_weights = np.broadcast_to(mu_weights, (n_layers, n_angular_points))
            top_mu, top_weights = mu_values, mu_weights
        else:
            rays = spherical_rays(interface_radii=interface_radii)
            n_angular_points = rays.impact_parameter.size
            ray_weights = rays.weights
            top_mu, top_weights = rays.mu[-1], rays.weights[-1]
        incident = self.incident_radiation_field
        if callable(incident):
            incident = incident(mu_values=top_mu, mu_weights=top_weights)
        elif rays is not None and incident is not None and incident.ndim == 2 and incident.shape[0] != n_angular_points:
            raise ValueError(
                "Spherical geometry evaluates the incident field on its own rays: pass the result of "
                "tiramisu.nlte.incident_intensity() rather than an array."
            )

        if any_initial_step:
            res = self._global_chi_matrix * density_profile[:, None] * dz_profile[:, None]
            dtau = res.decompose().value
            if rays is None:
                four_pi_j_up, four_pi_j_down = four_pi_j_hemispheres(
                    dtau=dtau,
                    source_function=self._global_source_func_matrix,
                    mu_values=mu_values,
                    mu_weights=mu_weights,
                    incident_radiation_field=incident,
                )
                four_pi_j_interfaces = (four_pi_j_up + four_pi_j_down)
                four_pi_j_mean = 0.5 * (four_pi_j_interfaces[:-1] + four_pi_j_interfaces[1:])
            else:
                four_pi_j_out, four_pi_j_in = four_pi_j_spherical(
                    dtau=dtau,
                    source_function=self._global_source_func_matrix,
                    rays=rays,
                    incident_radiation_field=incident,
                )
                four_pi_j_mean = four_pi_j_out + four_pi_j_in
            for species in nlte_processors.keys():
                processor = nlte_processors[species]
                if not processor.do_initial_step or processor.debug_pop_matrix is not None:
                    continue

                log.info(f"Computing initial population step for {species}.")

                processor.compute_initial_step(
                    j_mean=four_pi_j_mean / (4 * np.pi * u.sr),
                    chem_profile=chem_profile,
                    density_profile=density_profile,
                    temperature_profile=temperature,
                    wn_grid=spectral_grid,
                    wn_dx=wn_dx,
                )
                for layer_idx in range(self.n_lte_layers, n_layers):
                    processor.update_layer_global_chi_eta(
                        wn_grid=spectral_grid,
                        layer_vmr=chem_profile[species][layer_idx],
                        layer_global_chi_matrix=self._global_chi_matrix[layer_idx],
                        layer_global_eta_matrix=self._global_eta_matrix[layer_idx],
                        layer_idx=layer_idx,
                        nlte_layer_idx=layer_idx - self.n_lte_layers,
                    )
            zero_chi_mask = self._global_chi_matrix == 0
            self._global_source_func_matrix[~zero_chi_mask] = (
                    self._global_eta_matrix[~zero_chi_mask] / (ac.c * self._global_chi_matrix[~zero_chi_mask])
            )

        # tau_mu is no longer materialised (effective_source_tau_mu now returns 2D tau; see its docstring), so the
        # intensity/Lambda matrices' shape comes from the global matrices + n_angular_points (mu nodes or spherical
        # rays) directly.
        layer_shape = (n_angular_points, self._global_source_func_matrix.shape[1])
        # i_in_matrix/i_out_* are specific intensities I(mu) per direction (inward/outward hemispheres) at layer
        # centres, NOT mean intensities: J is formed from them per layer as j_layer_grid below.
        # Only the inward intensity is kept for every layer (the outward pass needs it above each layer it reaches);
        # the outward intensity is only needed at the current layer and the one below, and Lambda* at the current
        # layer, so those are per-layer buffers.
        # They are kept as plain arrays (not Quantity) through the whole Gauss-Seidel sweep -
        # update_inward_layer/update_outward_layer are numba kernels and can't operate on Quantity objects, and
        # per-layer unit rewrapping on every update would cost more than the arithmetic itself. Units are
        # reattached only where something outside this sweep needs them (j_layer_grid, below).
        # Allocated once for all iterations: each sweep rewrites every entry it reads, except those of directions that
        # do not reach a layer, which carry zero weight there.
        i_in_matrix = np.zeros((n_layers, *layer_shape))
        lambda_in_layer = np.zeros(layer_shape)
        i_out_below, i_out_layer, lambda_out_layer = (np.zeros(layer_shape) for _ in range(3))
        # Each spherical ray's optical path up from its turning point, for the exact outward Lambda*.
        sigma_below, sigma_layer = (np.zeros(layer_shape) for _ in range(2)) if rays is not None else (None, None)

        # -------------------------- Iterative solution --------------------------
        while not self._is_converged:
            self._n_iter += 1

            (effective_source_func_matrix, effective_tau, self._effective_chi_matrix,
             effective_dtau) = effective_source_tau_mu(
                global_source_func_matrix=self._global_source_func_matrix,
                global_chi_matrix=self._global_chi_matrix,
                global_eta_matrix=self._global_eta_matrix,
                density_profile=density_profile,
                dz_profile=dz_profile,
            )
            if rays is None:
                # Spherical rays compute their (ray-dependent) control points inside the kernels instead.
                start_time = time.perf_counter()
                delta_tau, control_points = bezier_delta_tau_and_control_points(
                    tau_matrix=effective_tau,
                    source_function_matrix=effective_source_func_matrix,
                )
                log.log(
                    _LOG_VERBOSE_2,
                    f"Gauss-Seidel/Bezier coefficient duration = {time.perf_counter() - start_time:.3f}"
                )

            effective_source_func_value = effective_source_func_matrix.value
            ################
            # A copy, so this iteration's accepted populations don't overwrite the previous iteration's row of
            # pop_matrix in place: the solve only reads each layer's committed row before that layer is updated.
            pop_update_dict = {}
            for species in nlte_processors.keys():
                processor = nlte_processors[species]
                pop_update_dict[species] = processor.get_latest_pop_grid().copy()

            # One Bezier step at a layer for every direction, in the chosen geometry. These read the current
            # effective_source_func_value/effective_dtau/delta_tau/control_points, which are refreshed as layers update.
            # with_lambda: whether this call's Lambda* is used (once per layer and direction, in NLTE layers only).
            def sweep_inward(layer_idx: int, with_lambda: bool) -> None:
                if rays is None:
                    update_inward_layer(
                        layer_idx=layer_idx,
                        i_in_matrix=i_in_matrix,
                        lambda_in_layer=lambda_in_layer,
                        delta_tau=delta_tau,
                        mu_values=mu_values,
                        control_points=control_points,
                        source_function_matrix=effective_source_func_value,
                        with_lambda=with_lambda,
                    )
                else:
                    update_inward_layer_spherical(
                        layer_idx=layer_idx,
                        i_in_matrix=i_in_matrix,
                        lambda_in_layer=lambda_in_layer,
                        dtau=effective_dtau,
                        seg_a=rays.seg_a,
                        seg_b=rays.seg_b,
                        tangent_layer=rays.tangent_layer,
                        source_function_matrix=effective_source_func_value,
                        with_lambda=with_lambda,
                    )

            def sweep_outward(layer_idx: int, with_lambda: bool) -> None:
                if rays is None:
                    update_outward_layer(
                        layer_idx=layer_idx,
                        i_out_below=i_out_below,
                        i_out_layer=i_out_layer,
                        lambda_out_layer=lambda_out_layer,
                        delta_tau=delta_tau,
                        mu_values=mu_values,
                        control_points=control_points,
                        source_function_matrix=effective_source_func_value,
                        with_lambda=with_lambda,
                    )
                else:
                    update_outward_layer_spherical(
                        layer_idx=layer_idx,
                        i_out_below=i_out_below,
                        i_out_layer=i_out_layer,
                        lambda_out_layer=lambda_out_layer,
                        sigma_below=sigma_below,
                        sigma_layer=sigma_layer,
                        lambda_in_layer=lambda_in_layer,
                        i_in_matrix=i_in_matrix,
                        dtau=effective_dtau,
                        seg_a=rays.seg_a,
                        seg_b=rays.seg_b,
                        tangent_layer=rays.tangent_layer,
                        source_function_matrix=effective_source_func_value,
                        with_lambda=with_lambda,
                    )

            # -------------------------- GAUSS-SEIDEL PASSES --------------------------
            # ------------------------------ INWARD PASS ------------------------------
            # Upper boundary condition if incident radiation field present: the downward specific intensity on each
            # direction. The top point is the top layer's centre, dtau_top / 2 below the TOA; that half layer is not
            # attenuated here (negligible for a thin top layer).
            if incident is not None:
                i_in_matrix[-1] = incident.to_value(source_func_unit)

            for layer_idx in range(n_layers - 1)[::-1]:
                # Inward intensity; Lambda* only where it is used before the outward pass recomputes it (layer 0).
                sweep_inward(layer_idx, with_lambda=layer_idx == 0 and layer_idx >= self.n_lte_layers)
            # ------------------------------ OUTWARD PASS ------------------------------
            for layer_idx in range(n_layers):
                if layer_idx > 0:
                    # The last layer's outward intensity becomes the one below; its old buffer is overwritten here.
                    i_out_below, i_out_layer = i_out_layer, i_out_below
                    sigma_below, sigma_layer = sigma_layer, sigma_below
                if layer_idx == 0:
                    bottom_intensity = (
                        blackbody(spectral_grid=spectral_grid, temperature=temperature[0])[0]
                        .to_value(source_func_unit)
                    )
                    lambda_out_layer[:] = 0.0
                    if rays is None:
                        i_out_layer[:] = bottom_intensity
                    else:
                        # Rays hitting the BOA start there; the ray tangent to the bottom layer turns there.
                        i_out_layer[rays.tangent_layer < 0] = bottom_intensity
                        sweep_outward(layer_idx, with_lambda=layer_idx >= self.n_lte_layers)
                    # lambda_in_layer still holds layer 0's inward Lambda* from the inward pass, which ends there.
                else:
                    # INWARD UPDATES (during outward pass), first: the spherical outward Lambda* uses this layer's inward
                    # Lambda*.
                    if layer_idx < n_layers - 1:
                        # Below the first NLTE layer nothing it depends on has changed since the inward pass (the
                        # source function there is LTE), so the recomputation would reproduce it exactly.
                        if layer_idx >= self.n_lte_layers:
                            sweep_inward(layer_idx, with_lambda=True)
                    else:
                        # The top point's inward intensity is the incident boundary: no local contribution.
                        lambda_in_layer[:] = 0.0

                    # Outward intensity + Lambda operator update.
                    sweep_outward(layer_idx, with_lambda=layer_idx >= self.n_lte_layers)
                # Solve equilibrium for non-LTE layers.
                if layer_idx >= self.n_lte_layers:
                    nlte_layer_idx = layer_idx - self.n_lte_layers

                    # Mean intensity J = 1/2 sum_mu w (I_in + I_out), and the matching angle-averaged Lambda*.
                    # This can be done independent of the transitions.
                    j_layer_grid = (0.5 * np.sum(
                        (i_in_matrix[layer_idx] + i_out_layer) * ray_weights[layer_idx][:, None],
                        axis=0,
                    )) << source_func_unit
                    lambda_layer_grid = 0.5 * np.sum(
                        (lambda_in_layer + lambda_out_layer) * ray_weights[layer_idx][:, None],
                        axis=0,
                    )
                    # If species VMR is very small or 0 we should not bother solving for it as the contribution to
                    # overall opacity, emission will be negligable.
                    layer_active_species = {
                        species: chem_profile[species][layer_idx] > _VMR_THRESHOLD
                        for species in nlte_processors.keys()
                    }
                    # Store raw new/old populations (normalised), so the accelerator can register per-species change
                    # and attempt one merged cross-species extrapolation for this layer. Pops are processed after
                    # acceleration in the update_pops call.
                    pop_new_dict = {}
                    pop_old_dict = {}
                    # TODO: skip if species abundance is 0 on this layer?
                    for species in nlte_processors.keys():
                        if not layer_active_species[species]:
                            continue
                        processor = nlte_processors[species]
                        processor.build_y_matrix(
                            layer_idx=layer_idx,
                            nlte_layer_idx=nlte_layer_idx,
                            j_layer_grid=j_layer_grid,
                            lambda_layer_grid=lambda_layer_grid,
                            chem_profile=chem_profile,
                            effective_chi_matrix=self._effective_chi_matrix,
                            wn_grid=spectral_grid,
                            wn_dx=wn_dx,
                            full_prec=self._full_prec,
                        )
                        # We can solve for the pops here because we don't modify any of the properties used to construct
                        # the Y matrices for the next species until the next loop.
                        pop_new, pop_old = processor.solve_pops(
                            layer_idx=layer_idx,
                            n_iter=self._n_iter,
                        )
                        # We store these because pop_old is sliced according to the y_idx_map.
                        pop_new_dict[species] = pop_new
                        pop_old_dict[species] = pop_old

                    # One call handles every active species on this layer together: shared-omega damping and
                    # cross-species Anderson extrapolation (fit jointly across species, since they're radiatively
                    # coupled on this layer) both live in MultiSpeciesAccelerator now.
                    pop_updated_dict = self.accelerator.apply(
                        layer_idx=nlte_layer_idx,
                        pop_new=pop_new_dict,
                        pop_old=pop_old_dict,
                        iteration=self._n_iter,
                    )

                    # Global properties are then updated per species.
                    for species in nlte_processors.keys():
                        if not layer_active_species[species]:
                            continue
                        processor = nlte_processors[species]
                        pop_updated_full = processor.update_pops(
                            layer_idx=layer_idx,
                            pop_updated=pop_updated_dict[species],
                        )
                        pop_update_dict[species][layer_idx] = pop_updated_full

                        # Global matrices updated inplace.
                        processor.update_layer_global_chi_eta(
                            wn_grid=spectral_grid,
                            layer_vmr=chem_profile[species][layer_idx],
                            layer_global_chi_matrix=self._global_chi_matrix[layer_idx],
                            layer_global_eta_matrix=self._global_eta_matrix[layer_idx],
                            layer_idx=layer_idx,
                            nlte_layer_idx=nlte_layer_idx,
                            layer_pop_grid=pop_updated_full,
                        )
                    #########
                    # Update all physical properties now all Non-LTE species' opacities have been updated.
                    self._global_source_func_matrix[layer_idx] = (
                            self._global_eta_matrix[layer_idx]
                            / (ac.c * self._global_chi_matrix[layer_idx])
                    ).to(source_func_unit, equivalencies=u.spectral())

                    (effective_source_func_matrix, effective_tau, self._effective_chi_matrix,
                     effective_dtau) = effective_source_tau_mu(
                        global_source_func_matrix=self._global_source_func_matrix,
                        global_chi_matrix=self._global_chi_matrix,
                        global_eta_matrix=self._global_eta_matrix,
                        density_profile=density_profile,
                        dz_profile=dz_profile,
                    )
                    effective_source_func_value = effective_source_func_matrix.value
                    # np.save(fr"/mnt/c/PhD/NLTE/theory/opacity/nLTE_tau_I{self.n_iter}_L{nlte_layer_idx}.npy",
                    #         effective_tau)
                    if rays is None:
                        start_time = time.perf_counter()
                        update_layer_coefficients(
                            layer_idx=layer_idx,
                            tau_matrix=effective_tau,
                            source_function_matrix=effective_source_func_value,
                            delta_tau=delta_tau,
                            control_points=control_points,
                        )

                        log.log(
                            _LOG_VERBOSE_2,
                            f"[L{layer_idx}] Coefficient update duration = {time.perf_counter() - start_time:.3f}"
                        )

                    if layer_idx > 0 or rays is not None:
                        # The outward intensity with the new source function; its Lambda* is not used.
                        sweep_outward(layer_idx, with_lambda=False)
            # ---------------------------- PASSES COMPLETE ----------------------------
            # Commit pop_grid_updates to each species.
            for species in nlte_processors.keys():
                processor = nlte_processors[species]
                processor.commit_pops(pop_update_grid=pop_update_dict[species])
                converged = self.accelerator.converged(species=species)
                max_changes = self.accelerator.get_max_changes(species=species)
                log.info(f"[I{self._n_iter}] {species} Max. pop. change = {max_changes.max():{_LOG_FLOAT_FMT}}"
                         f"{' CONVERGED' if converged else ''}\n"
                         f"Per layer chnages = {np.array2string(max_changes, formatter=_LOG_ARRAY_FMT)}")

            self._is_converged = self.accelerator.converged()
        log.info(f"[I{self._n_iter}] Convergence achieved!")

        # TODO: Default resolving power grid?
        high_res_grid = create_r_wn_grid(low=spectral_grid[0].value, high=spectral_grid[-1].value,
                                         resolving_power=15000)
        for species in active_species:
            xsec_data = self[species]
            if species in nlte_processors:
                processor = nlte_processors[species]
                xsec_data.opacity(temperature, pressure, high_res_grid)
                processor.finalise(temperature_profile=temperature, pressure_profile=pressure, wn_grid=high_res_grid)
                output_opacities[species] = processor.mol_chi_matrix
            else:
                output_opacities[species] = xsec_data.opacity(temperature, pressure, high_res_grid)

        return output_opacities

    @property
    def unified_grid(self) -> u.Quantity:
        # Opacities without a native grid (HMinusIon without one) contribute nothing.
        res = [x.spectral_grid for x in self.values() if x.spectral_grid is not None]
        if not res:
            raise ValueError("No opacity in the collection has a spectral grid: pass spectral_grid explicitly.")
        base_unit = res[0].unit
        res = np.concatenate([x.to_value(base_unit, equivalencies=u.spectral()) for x in res])
        res = np.unique(res) << base_unit

        return res


def create_r_wn_grid(low: float, high: float, resolving_power: float) -> u.Quantity:
    resolving_f = np.log((resolving_power + 1) / resolving_power)
    n_points = round((np.log(high) - np.log(low)) / resolving_f) + 1
    return np.exp(np.arange(n_points) * resolving_f + np.log(low)) << u.k
