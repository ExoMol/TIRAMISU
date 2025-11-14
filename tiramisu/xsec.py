import io
import multiprocessing
import time
import abc
import pathlib
import pickle
import typing as t
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from multiprocessing.synchronize import Lock, Event

from scipy.optimize import least_squares
from scipy.integrate import simpson
import numpy.typing as npt
import numpy as np

import astropy.units as u
import astropy.constants as ac
import polars as pl
import pandas as pd
import dask.dataframe as dd

from .chemistry import SpeciesFormula, SpeciesIdentType, ChemicalProfile
from .nlte import (
    boltzmann_population,
    calc_einstein_b_fi,
    calc_einstein_b_if,
    abs_emi_xsec,
    calc_band_profile,
    blackbody,
    bezier_coefficients,
    continuum_xsec,
    effective_source_tau_mu,
    ProfileStore, calc_cont_band_profile, ContinuumProfileStore, bezier_coefficients_new, update_layer_coefficients,
)
from .config import log, output_dir, _DEFAULT_NUM_THREADS


def _process_trans_batch(
        serialised_batch: bytes,
        layer_t_p_tuple: t.Tuple[int, t.Tuple[u.Quantity, u.Quantity]],
        serialised_states: bytes,
        broadening_params: t.Optional[t.Any],  # TODO: Fix typing
        species_mass: float,
        wn_grid: npt.NDArray[np.float64],
        n_lte_layers: int,
) -> t.Tuple[int, t.Optional[pl.DataFrame], t.Optional[pl.DataFrame]]:
    """
    Calculates the Einstein coefficient A_fi, B_fi, B_if and the vibronic band profiles of transitions in the
    chunk.

    :param serialised_batch: Serialized form of Polars DataFrame chunk read from the large file.
    :param layer_t_p_tuple: (layer_idx, (temperature, pressure)) for the current layer.
    :param serialised_states: Serialized form of the pre-calculated state data with populations for all layers.
    :param broadening_params: Broadening parameters from the main class instance.
    :param species_mass: Species mass from the main class instance.
    :param wn_grid: The full wavenumber grid.
    :param n_lte_layers: Number of LTE layers.
    :return: (layer_idx, band_profile_data, agg_batch_rates)
    """
    nlte_layer_idx, (temperature, pressure) = layer_t_p_tuple
    layer_idx = n_lte_layers + nlte_layer_idx

    # trans_batch = pl.from_pandas(trans_batch)
    trans_batch = pl.DataFrame.deserialize(io.BytesIO(serialised_batch))
    layers_states = pl.DataFrame.deserialize(io.BytesIO(serialised_states))

    layer_states_cols = ["id", "energy", "g", "tau", "id_agg", f"n_L{layer_idx}", f"n_agg_L{layer_idx}", ]
    layer_states = layers_states.select(layer_states_cols)
    layer_states = layer_states.with_columns(
        (pl.col(f"n_L{layer_idx}") / pl.col(f"n_agg_L{layer_idx}")).alias("n_frac")
    )

    select_cols_i = ["id", "energy", "g", "id_agg", "n_frac"]
    layer_states_i = layer_states.select(select_cols_i).rename({
        col: f"{col}_i" for col in select_cols_i
    })

    select_cols_f = ["id", "energy", "g", "id_agg", "tau", "n_frac"]
    layer_states_f = layer_states.select(select_cols_f).rename({
        col: f"{col}_f" for col in select_cols_f
    })

    trans_batch = trans_batch.join(layer_states_i, on="id_i", how="inner")
    trans_batch = trans_batch.join(layer_states_f, on="id_f", how="inner")

    trans_batch = trans_batch.with_columns(
        (pl.col("energy_f") - pl.col("energy_i")).alias("energy_fi")
    )

    wn_min = wn_grid[0]
    wn_max = wn_grid[-1]
    trans_batch = trans_batch.filter(
        (pl.col("energy_f") >= wn_min)
        & (pl.col("energy_i") <= wn_max)
        & (pl.col("energy_f") <= wn_max)
        & (pl.col("energy_fi") >= wn_min)
        & (pl.col("energy_fi") <= wn_max)
    )
    if trans_batch.height == 0:
        return nlte_layer_idx, None, None

    start_time = time.perf_counter()
    band_profile_partial = partial(
        calc_band_profile,
        wn_grid,
        temperature.value,
        pressure.value,
        species_mass,
        (broadening_params[1] if broadening_params is not None else None),
        (broadening_params[0][:, layer_idx] if broadening_params is not None else None),
    )
    # Calculate band profiles (required for every layer)
    band_profile_data = trans_batch.group_by(*["id_agg_f", "id_agg_i"]).map_groups(band_profile_partial)
    log.info(f"[L{layer_idx}] Profile duration = {time.perf_counter() - start_time:.2f}.")

    agg_batch = None
    if nlte_layer_idx == 0:
        trans_batch_rates = trans_batch.filter(pl.col("id_agg_f") != pl.col("id_agg_i"))

        b_fi_vals = calc_einstein_b_fi(
            a_fi=trans_batch_rates["A_fi"].to_numpy(),
            energy_fi=(trans_batch_rates["energy_fi"].to_numpy() << 1 / u.cm)
            .to(u.Hz, equivalencies=u.spectral())
            .value,
        )
        b_if_vals = calc_einstein_b_if(
            b_fi=b_fi_vals,
            g_f=trans_batch_rates["g_f"].to_numpy(),
            g_i=trans_batch_rates["g_i"].to_numpy(),
        )
        agg_batch = (
            trans_batch_rates
            .with_columns([
                pl.Series("B_fi", b_fi_vals),
                pl.Series("B_if", b_if_vals),
            ])
            .group_by(["id_agg_f", "id_agg_i"])
            .agg([
                pl.col("A_fi").sum().alias("A_fi"),
                pl.col("B_fi").sum().alias("B_fi"),
                pl.col("B_if").sum().alias("B_if"),
            ])
        )

    return nlte_layer_idx, band_profile_data, agg_batch


def _process_cont_batch(
        serialised_batch: bytes,
        layer_temperature_tuple: t.Tuple[int, u.Quantity],
        serialised_states: bytes,
        wn_grid: npt.NDArray[np.float64],
        species_mass: float,
        cont_box_length: float,
        n_lte_layers: int,
) -> t.Tuple[int, t.Optional[pl.DataFrame], t.Optional[pl.DataFrame]]:
    """
    Calculates the Einstein coefficient A_fi, B_fi, B_if and the vibronic band profiles of transitions in the chunk.

    Parameters
    ----------
    serialised_batch: bytes
        Serialised form of Polars DataFrame chunk read from the large file.
    layer_temperature_tuple: tuple
        (layer_idx, temperature) for the current layer.
    serialised_states: bytes
        Serialised form of the pre-calculated state data with populations for all layers.
    wn_grid: ndarray
        The full wavenumber grid.
    species_mass: float
        Mass of the species for broadening.
    cont_box_length: float
        Box length off to use in continuum box broadening.
    n_lte_layers: int
        Number of LTE layers.
    Returns
    -------
        (layer_idx, band_profile_data, agg_batch_rates)
    """
    nlte_layer_idx, layer_temp = layer_temperature_tuple
    layer_idx = n_lte_layers + nlte_layer_idx

    trans_batch = pl.DataFrame.deserialize(io.BytesIO(serialised_batch))
    layers_cont_states = pl.DataFrame.deserialize(io.BytesIO(serialised_states))

    layer_states_cols = ["id", "g", "energy", "id_agg", f"n_L{layer_idx}", f"n_agg_L{layer_idx}", ]
    layer_cont_states = layers_cont_states.select(layer_states_cols)
    layer_cont_states = layer_cont_states.with_columns(
        (pl.col(f"n_L{layer_idx}") / pl.col(f"n_agg_L{layer_idx}")).alias("n_frac")
    )

    select_cols_i = ["id", "g", "energy", "id_agg", "n_frac", ]
    layer_cont_states_i = layer_cont_states.select(select_cols_i).rename({
        col: f"{col}_i" for col in select_cols_i
    })

    select_cols_f = ["id", "g", "energy", "n_frac", "v_f"]
    layer_cont_states_f = layer_cont_states.select(select_cols_f).rename({
        col: f"{col}_f" for col in select_cols_f
    })

    trans_batch = trans_batch.join(layer_cont_states_i, on="id_i", how="inner")
    trans_batch = trans_batch.join(layer_cont_states_f, on="id_f", how="inner")

    trans_batch = trans_batch.with_columns(
        (pl.col("energy_f") - pl.col("energy_i")).alias("energy_fi")
    )
    wn_min = wn_grid[0]
    wn_max = wn_grid[-1]
    trans_batch = trans_batch.filter(
        (pl.col("energy_f") >= wn_min)
        & (pl.col("energy_i") <= wn_max)
        & (pl.col("energy_fi") >= wn_min)
        & (pl.col("energy_fi") <= wn_max)
    )
    if trans_batch.height == 0:
        return nlte_layer_idx, None, None

    # TODO: Handle proper continuum state pops for cases without assumed 100% dissociation efficiency.
    trans_batch = trans_batch.fill_null(strategy="zero")

    # Group only on lower agg. state as upper in continuum and not included in equilibrium.
    start_time = time.perf_counter()
    band_profile_partial = partial(
        calc_cont_band_profile,
        wn_grid,
        layer_temp,
        species_mass,
        cont_box_length,
    )
    band_profile_data = trans_batch.group_by("id_agg_i").map_groups(band_profile_partial)
    log.info(f"[L{layer_idx}] Continuum profile duration = {time.perf_counter() - start_time:.2f}.")

    agg_batch = None
    if nlte_layer_idx == 0:
        # chunk = chunk.loc[(chunk["energy_ci"] >= wn_grid[0].value) & (chunk["energy_ci"] <= wn_grid[-1].value)]
        b_ci_vals = calc_einstein_b_fi(
            a_fi=trans_batch["A_fi"].to_numpy(),
            energy_fi=(trans_batch["energy_fi"].to_numpy() << 1 / u.cm)
            .to(u.Hz, equivalencies=u.spectral())
            .value,
        )
        b_ic_vals = calc_einstein_b_if(
            b_fi=b_ci_vals,
            g_f=trans_batch["g_f"].to_numpy(),
            g_i=trans_batch["g_i"].to_numpy(),
        )
        agg_batch = (
            trans_batch
            .with_columns([
                pl.Series("B_fi", b_ci_vals),
                pl.Series("B_if", b_ic_vals),
            ])
            .group_by("id_agg_i")
            .agg([
                pl.col("A_fi").sum().alias("A_i"),
                pl.col("B_fi").sum().alias("B_fi"),
                pl.col("B_if").sum().alias("B_if"),
            ])
        )

    return nlte_layer_idx, band_profile_data, agg_batch


class NLTEProcessor:
    """Handles all NLTE-specific functionality."""

    def __init__(
            self,
            species: str | SpeciesFormula,
            states_file: pathlib.Path,
            trans_files: pathlib.Path | t.List[pathlib.Path],
            agg_col_nums: t.List[int],
            species_mass: float,
            broadening_params: t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] = None,
            n_lte_layers: int = 0,
            cont_states_file: pathlib.Path = None,
            cont_trans_files: pathlib.Path | t.List[pathlib.Path] = None,
            cont_box_length: float = None,
            cont_broad_col_num: int = None,
            dissociation_products: t.Tuple = None,
            debug: bool = False,
            debug_pop_matrix: npt.NDArray[np.float64] = None,
    ):
        self.species = SpeciesFormula(species)
        if type(states_file) is str:
            states_file = pathlib.Path(states_file)
        self.states_file: pathlib.Path = states_file
        if type(trans_files) is str:
            trans_files = [pathlib.Path(trans_files)]
        elif type(trans_files) is not list:
            trans_files = [trans_files]
        self.trans_files: t.List[pathlib.Path] = trans_files
        self.agg_col_nums: t.List[int] = agg_col_nums
        self.agg_col_names: t.List[str] = ["agg" + str(idx + 1) for idx in range(0, len(self.agg_col_nums))]
        self.species_mass: float = species_mass
        self.broadening_params = broadening_params
        self.n_agg_states: int | None = None
        self.states: pl.DataFrame | None = None
        self.agg_states: pl.DataFrame | None = None
        self.rates_grid: pl.DataFrame | None = None
        self.profile_store: ProfileStore | None = None
        # self.abs_profile_grid: BandProfileCollection | None = None  # Deprecated
        # self.emi_profile_grid: BandProfileCollection | None = None  # Deprecated
        self.mol_chi_matrix: u.Quantity | None = None
        self.mol_eta_matrix: u.Quantity | None = None
        self.pop_matrix: npt.NDArray[np.float64] | None = None
        self.n_lte_layers: int = n_lte_layers
        self.cont_rates: pl.DataFrame | None = None
        if type(cont_states_file) is str:
            cont_states_file = pathlib.Path(cont_states_file)
        self.cont_states_file: pathlib.Path = cont_states_file
        self.cont_states: pl.DataFrame | None = None
        if type(cont_trans_files) is str:
            cont_trans_files = [pathlib.Path(cont_trans_files)]
        elif type(cont_trans_files) is not list:
            cont_trans_files = [cont_trans_files]
        self.cont_trans_files: t.List[pathlib.Path] = cont_trans_files
        self.cont_profile_store: ContinuumProfileStore | None = None
        self.cont_box_length: float | None = cont_box_length
        self.cont_broad_col_num: int | None = cont_broad_col_num
        if (self.cont_states is None) ^ (self.cont_trans_files is None) ^ (self.cont_box_length is None) ^ (
                self.cont_broad_col_num is None):
            raise RuntimeError(
                "Continuum states and trans files must both be provided with a box length for broadening and "
                "column index for box broadening n."
            )
        self.dissociation_products: t.Tuple = dissociation_products
        self.debug: bool = debug
        self.debug_pop_matrix: npt.NDArray[np.float64] | None = debug_pop_matrix

    def aggregate_states(self, temperature_profile: u.Quantity, energy_cutoff: float = None):
        """
        Sets self.states with a pandas DataFrame containing the ID, energy, degeneracy and lifetime columns of the
        statesfile, the columns on which state aggregation is performed and the corresponding aggregated state ID.

        :param temperature_profile: The temperature of each layer in Kelvin.
        """
        if self.agg_col_nums is None:
            # Assuming diatomic by default.
            self.agg_col_nums = [9, 10]

        read_col_indices = [0, 1, 2, 5] + self.agg_col_nums
        read_col_names = ["id", "energy", "g", "tau"] + self.agg_col_names
        fixed_dtypes = {
            "id": "Int64",
            "energy": np.float64,
            "g": np.float64,
            "tau": np.float64,
        }
        self.states = pl.from_pandas(pd.read_csv(
            self.states_file,
            sep=r"\s+",
            names=read_col_names,
            usecols=read_col_indices,
            dtype=fixed_dtypes
        ))
        # TODO: Drop states above grid cutoff?
        # if energy_cutoff is not None:
        #     pl_states = self.states.filter(pl.col("energy") <= energy_cutoff)
        self.agg_states = self.states.group_by(*self.agg_col_names).agg(
            pl.col("energy").min().alias("energy_agg")
        )
        self.n_agg_states = len(self.agg_states)

        self.agg_states = self.agg_states.sort("energy_agg")
        self.agg_states = self.agg_states.with_columns(
            pl.int_range(0, self.n_agg_states, dtype=pl.Int64).alias("id_agg")
        )
        log.debug(f"Vibronically aggregated states (head): \n {self.agg_states.head(30)}")

        self.states = self.states.join(
            self.agg_states.select(
                ["id_agg"] + self.agg_col_names
            ),
            on=self.agg_col_names,
            how="left",
        )
        log.debug(f"Working states = {self.states}")
        self.pop_matrix = np.zeros((1, temperature_profile.shape[0], self.n_agg_states))

        g_np = self.states["g"].to_numpy()
        energy_np = self.states["energy"].to_numpy() << 1 / u.cm

        ac_h_c_on_kB = ac.h * ac.c.cgs / ac.k_B

        for layer_idx, layer_temperature in enumerate(temperature_profile):
            q_lev_np = g_np * np.exp(
                -ac_h_c_on_kB * energy_np / layer_temperature
            ).value
            n_np = q_lev_np / np.sum(q_lev_np)
            n_col = f"n_L{layer_idx}"
            n_agg_col = f"n_agg_L{layer_idx}"

            # Join computed populations back onto states.
            self.states = self.states.with_columns(pl.Series(n_col, n_np))

            states_agg_n = (
                self.states
                .select(["id_agg", n_col])
                .group_by("id_agg")
                .agg(pl.col(n_col).sum().alias(n_agg_col))
            )
            # Join aggregated populations back onto states.
            self.states = self.states.join(states_agg_n, on="id_agg", how="left")
            self.pop_matrix[0, layer_idx] = states_agg_n.sort("id_agg")[n_agg_col].to_numpy()
        log.info(f"[I0] States = {self.states}")

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

        n_nlte_layers = temperature_profile.shape[0] - self.n_lte_layers
        self.rates_grid = []
        self.profile_store = ProfileStore(n_layers=n_nlte_layers)

        dask_dtypes = {"id_f": "int64", "id_i": "int64", "A_fi": "float64"}
        dask_blocksize = "256MB"

        serialised_states = self.states.serialize()

        for trans_file in self.trans_files:
            log.info(f"[I0] Processing file {trans_file}.")
            ddf = dd.read_csv(
                trans_file,
                sep=r"\s+",
                engine="python",
                header=None,
                names=["id_f", "id_i", "A_fi"],
                usecols=[0, 1, 2],
                dtype=dask_dtypes,
                blocksize=dask_blocksize,
            )
            delayed_batches = ddf.to_delayed()

            for delayed_batch in delayed_batches:
                trans_batch_pd = delayed_batch.compute()  # Dask -> Pandas.
                trans_batch = pl.from_pandas(trans_batch_pd)  # Pandas -> Polars
                serialised_batch = trans_batch.serialize()  # Polars -> Arrow IPC binary stream

                # if trans_batch.height == 0:
                #     continue

                agg_partial = partial(
                    _process_trans_batch,
                    serialised_batch,
                    serialised_states=serialised_states,
                    broadening_params=self.broadening_params,
                    species_mass=self.species_mass,
                    wn_grid=wn_grid.value,
                    n_lte_layers=self.n_lte_layers
                )
                with ProcessPoolExecutor(max_workers=num_threads,
                                         mp_context=multiprocessing.get_context("spawn")) as ex:
                    results = ex.map(
                        agg_partial,
                        list(enumerate(zip(
                            temperature_profile[self.n_lte_layers:],
                            pressure_profile[self.n_lte_layers:],
                        ))),
                    )
                    rates_appended = False
                    for nlte_layer_idx, band_profile_data, agg_batch in results:
                        if band_profile_data is not None:
                            self.profile_store.add_batch(batch=band_profile_data, layer_idx=nlte_layer_idx)

                        if not rates_appended and agg_batch is not None:
                            self.rates_grid.append(agg_batch)
                            rates_appended = True

        self.profile_store.finalise()

        self.rates_grid: pl.DataFrame = pl.concat(self.rates_grid)
        self.rates_grid = self.rates_grid.group_by(*["id_agg_f", "id_agg_i"]).agg([
            pl.col("A_fi").sum().alias("A_fi"),
            pl.col("B_fi").sum().alias("B_fi"),
            pl.col("B_if").sum().alias("B_if"),
        ])
        log.info(f"[I0] Rates = \n{self.rates_grid}")
        # Done.

    def load_continuum_rates(
            self,
            temperature_profile: u.Quantity,
            wn_grid: u.Quantity,
            num_threads: int = _DEFAULT_NUM_THREADS,
    ):
        log.info(f"Loading continuum absorption rates and profiles.")

        read_col_map = {num: "v" if num == self.cont_broad_col_num else name for num, name in
                        zip(self.agg_col_nums, self.agg_col_names)}
        if self.cont_broad_col_num not in read_col_map:
            read_col_map[self.cont_broad_col_num] = "v"

        # extra_col_indices = [k for k, _ in sorted(read_col_map.items())]
        # extra_col_names = [v for _, v in sorted(read_col_map.items())]
        extra_col_indices, extra_col_names = (list(x) for x in zip(*sorted(read_col_map.items())))

        read_col_names = ["id", "energy", "g"] + extra_col_names
        read_col_indices = [0, 1, 2] + extra_col_indices
        fixed_dtypes = {"id": "Int64", "energy": np.float64, "g": np.float64, "v": "Int64"}

        self.cont_states = pl.from_pandas(pd.read_csv(
            self.cont_states_file,
            sep=r"\s+",
            names=read_col_names,
            usecols=read_col_indices,
            dtype=fixed_dtypes,
        ))

        merge_cols = ["id", "id_agg"]
        self.cont_states = self.cont_states.join(self.states.select(merge_cols), on="id", how="left")
        # Above is new! Below has not yet been updated to polars.

        # self.cont_states = self.cont_states.merge(self.states[merge_cols], on="id", how="left")
        # self.cont_states["id_agg"] = self.cont_states["id_agg"].astype("Int64")
        # # NB: Left join converts ints to float as some may be nan, does not occur for inner join but left needed here to
        # # preserve energy/degeneracy info of upper states with no id_agg map.

        layers_cont_states = self.cont_states.clone()

        for nlte_layer_idx, layer_temp in enumerate(temperature_profile[self.n_lte_layers:]):
            layer_idx = nlte_layer_idx + self.n_lte_layers
            # Precompute boltzmann populations for each layer.
            temp_cont_states = boltzmann_population(self.cont_states.clone(), layer_temp)
            layers_cont_states = layers_cont_states.join(
                temp_cont_states.select([
                    pl.col("id"),
                    pl.col("n").alias(f"n_L{layer_idx}"),
                    pl.col("n_agg").alias(f"n_agg_L{layer_idx}")
                ]),
                on="id",
                how="left"
            )

        n_nlte_layers = temperature_profile.shape[0] - self.n_lte_layers
        self.cont_rates = []
        self.cont_profile_store = ContinuumProfileStore(n_layers=n_nlte_layers)

        dask_dtypes = {"id_f": "int64", "id_i": "int64", "A_fi": "float64"}
        dask_blocksize = "256MB"

        serialised_states = layers_cont_states.serialize()

        for cont_trans_file in self.cont_trans_files:
            log.info(f"[I0] Processing file {cont_trans_file}.")

            ddf = dd.read_csv(
                cont_trans_file,
                sep=r"\s+",
                engine="python",
                header=None,
                names=["id_f", "id_i", "A_fi"],
                usecols=[0, 1, 2],
                dtype=dask_dtypes,
                blocksize=dask_blocksize,
            )
            delayed_batches = ddf.to_delayed()

            for delayed_batch in delayed_batches:
                trans_batch_pd = delayed_batch.compute()  # Dask -> Pandas.
                trans_batch = pl.from_pandas(trans_batch_pd)  # Pandas -> Polars
                serialised_batch = trans_batch.serialize()  # Polars -> Arrow IPC binary stream

                # if trans_batch.height == 0:
                #     continue

                agg_partial = partial(
                    _process_cont_batch,
                    serialised_batch,
                    serialised_states=serialised_states,
                    wn_grid=wn_grid.value,
                    species_mass=self.species_mass,
                    cont_box_length=self.cont_box_length,
                    n_lte_layers=self.n_lte_layers
                )
                log.info(trans_batch)
                with ProcessPoolExecutor(max_workers=num_threads,
                                         mp_context=multiprocessing.get_context("spawn")) as ex:
                    results = ex.map(
                        agg_partial,
                        list(enumerate(temperature_profile[self.n_lte_layers:])),
                    )
                    rates_appended = False
                    for layer_idx, band_profile_data, agg_batch in results:
                        if band_profile_data is not None:
                            self.cont_profile_store.add_batch(batch=band_profile_data, layer_idx=layer_idx)

                        if not rates_appended and agg_batch is not None:
                            self.cont_rates.append(agg_batch)
                            rates_appended = True

        self.cont_profile_store.finalise()

        self.cont_rates: pl.DataFrame = pl.concat(self.cont_rates)
        self.cont_rates = self.cont_rates.group_by("id_agg_i").agg([
            pl.col("A_fi").sum().alias("A_fi"),
            pl.col("B_fi").sum().alias("B_fi"),
            pl.col("B_if").sum().alias("B_if"),
        ])
        log.info(f"[I0] Continuum rates = \n{self.cont_rates}")
        # Done.

    def setup(self, chem_profile: ChemicalProfile, temperature_profile: u.Quantity,
              pressure_profile: u.Quantity, wn_grid: u.Quantity, initial_chi_matrix: u.Quantity) -> None:
        """Setup NLTE calculations."""
        if any([mol not in chem_profile.species for mol in self.dissociation_products]):
            warn_string = (
                f"Specified dissociation products {self.dissociation_products} not present in"
                f" chemical profile {chem_profile.species}."
            )
            log.warning(warn_string)
            # raise RuntimeError(warn_string)

        self.aggregate_states(temperature_profile=temperature_profile, energy_cutoff=wn_grid[-1].value)
        self.compute_rates_profiles(
            temperature_profile=temperature_profile,
            pressure_profile=pressure_profile,
            wn_grid=wn_grid,
        )
        if self.cont_states_file is not None and self.cont_trans_files is not None:
            self.load_continuum_rates(temperature_profile=temperature_profile, wn_grid=wn_grid)

        self.mol_chi_matrix = initial_chi_matrix
        lte_source_func_matrix = blackbody(
            spectral_grid=wn_grid, temperature=temperature_profile
        )
        self.mol_eta_matrix = lte_source_func_matrix * self.mol_chi_matrix * ac.c

    def build_y_matrix(
            self,
            layer_idx: int,
            nlte_layer_idx: int,
            layer_temperature: u.Quantity,
            i_layer_grid: u.Quantity,
            lambda_layer_grid: npt.NDArray[np.float64],
            chem_profile: ChemicalProfile,
            density_profile: u.Quantity,
            global_chi_matrix: u.Quantity,
            global_source_func_matrix: u.Quantity,
            wn_grid: u.Quantity,
            full_prec: bool,
    ) -> npt.NDArray[np.float64]:
        """
        Build statistical equilibrium matrix.
        """
        species_eta = chem_profile[self.species][layer_idx] * self.mol_eta_matrix[layer_idx] / ac.c
        global_chi = global_chi_matrix[layer_idx]  # / density_profile[layer_idx]
        psi_approx_eta = np.zeros(lambda_layer_grid.shape[0]) << global_source_func_matrix.unit
        psi_approx_eta[global_chi != 0] = (
                lambda_layer_grid[global_chi != 0] * species_eta[global_chi != 0] / global_chi[global_chi != 0]
        )
        psi_approx_eta = np.clip(abs(psi_approx_eta), 0, i_layer_grid)
        i_prec: u.Quantity = (i_layer_grid - psi_approx_eta) * 4 * np.pi * u.sr

        y_matrix = np.zeros((self.n_agg_states, self.n_agg_states)) << (1 / u.s)
        for trans_row in self.rates_grid.iter_rows(named=False):
            # 0 = id_agg_f, 1 = id_agg_i, 2 = A_fi, 3 = B_fi, 4 = B_if.
            a_fi = trans_row[2] / u.s
            b_fi = trans_row[3] * (u.m ** 2) / (u.J * u.s)
            b_if = trans_row[4] * (u.m ** 2) / (u.J * u.s)
            log.info(f"[L{layer_idx}] Trans: {trans_row}.")

            # if (trans_row[0], trans_row[1]) not in self.abs_profile_grid[nlte_layer_idx]:
            #     log.warning(
            #         (
            #             f"[L{layer_idx}] Trans {(trans_row[0], trans_row[1])} in rates but not in absorption "
            #             f"band profiles."
            #         )
            #     )
            #     continue
            # if (trans_row[0], trans_row[1]) not in self.emi_profile_grid[nlte_layer_idx]:
            #     log.warning(
            #         (
            #             f"[L{layer_idx}] Trans {(trans_row[0], trans_row[1])} in rates but not in emission "
            #             f"band profiles."
            #         )
            #     )
            #     continue

            abs_profile, abs_start_idx = self.profile_store.get_profile(
                layer_idx=layer_idx, key=(trans_row[0], trans_row[1]), profile_type="abs"
            )
            ste_profile, ste_start_idx = self.profile_store.get_profile(
                layer_idx=layer_idx, key=(trans_row[0], trans_row[1]), profile_type="ste"
            )

            # abs_profile = self.abs_profile_grid[nlte_layer_idx].get((trans_row[0], trans_row[1]))
            # emi_profile = self.emi_profile_grid[nlte_layer_idx].get((trans_row[0], trans_row[1]))
            # abs_end_idx = abs_profile.start_idx + len(abs_profile.profile)
            # emi_end_idx = emi_profile.start_idx + len(emi_profile.profile)
            abs_end_idx = abs_start_idx + len(abs_profile)
            ste_end_idx = ste_start_idx + len(ste_profile)

            abs_profile_norm = abs_profile / simpson(abs_profile, x=wn_grid[abs_start_idx:abs_end_idx])
            ste_profile_norm = ste_profile / simpson(ste_profile, x=wn_grid[ste_start_idx:ste_end_idx])

            u_fi = a_fi
            u_fi = u_fi.decompose()
            log.debug(f"[L{layer_idx}] U_{trans_row[0], trans_row[1]} = {u_fi}")

            # stim_emi_profile = (
            #         emi_profile.profile
            #         * emi_profile.integral
            #         * u.erg
            #         * u.cm
            #         / (2 * u.s * ac.h.cgs * ac.c.cgs ** 2 * wn_grid[emi_profile.start_idx: emi_end_idx] ** 3)
            # )
            # stim_emi_profile = stim_emi_profile.value / simpson(
            #     stim_emi_profile, x=wn_grid[emi_profile.start_idx: emi_end_idx]
            # )

            # Cross terms:
            chi_if = np.zeros(lambda_layer_grid.shape[0]) << (u.m ** 2) / (u.J * u.s)
            chi_if[abs_start_idx: abs_end_idx] += (
                    self.pop_matrix[-1, layer_idx, trans_row[1]]
                    * abs_profile_norm
                    * trans_row[4]
                    * u.m ** 2
                    / (u.J * u.s)
            )
            chi_if[ste_start_idx: ste_end_idx] -= (
                    self.pop_matrix[-1, layer_idx, trans_row[0]]
                    * ste_profile_norm
                    * trans_row[3]
                    * u.m ** 2
                    / (u.J * u.s)
            )
            chi_if *= chem_profile[self.species][layer_idx]
            chi_if = np.where(chi_if < 0, 0, chi_if)
            if full_prec:
                for o_idx in np.arange(0, self.n_agg_states):
                    a_ox_cross = np.zeros(lambda_layer_grid.shape[0]) << u.erg / u.sr
                    for ox_trans in self.rates_grid.filter(pl.col("id_agg_f") == o_idx).iter_rows(named=False):
                        # ox_emi_profile = self.emi_profile_grid[nlte_layer_idx].get((ox_trans[0], ox_trans[1]))
                        ox_emi_profile, ox_emi_start_idx = self.profile_store.get_profile(
                            layer_idx=nlte_layer_idx, key=(ox_trans[0], ox_trans[1]), profile_type="spe"
                        )

                        ox_emi_end_idx = ox_emi_start_idx + len(ox_emi_profile)
                        a_ox_cross[ox_emi_start_idx: ox_emi_end_idx] += (
                                ox_emi_profile  # Absolute value, not normalised.
                                * u.erg
                                * u.cm
                                * chem_profile[self.species][layer_idx]
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
                    ).to(u.J / (u.m ** 2 * u.sr), equivalencies=u.spectral())

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
                        * chem_profile[self.species][layer_idx]
                        * ac.h
                        / global_chi[global_chi != 0].to(u.m ** 2, equivalencies=u.spectral())
                )
                self_prec = simpson(self_prec, x=wn_grid)
                u_fi *= 1 - self_prec
            # End cross.

            v_fi_prec = ste_profile_norm * i_prec[ste_start_idx: ste_end_idx]
            v_fi_prec = (
                                simpson(v_fi_prec, x=wn_grid[ste_start_idx: ste_end_idx]) << v_fi_prec.unit
                        ) * b_fi
            v_fi_prec = v_fi_prec.decompose()
            log.debug(f"[L{layer_idx}] V_{trans_row[0], trans_row[1]}_prec = {v_fi_prec}")

            v_if_prec = abs_profile_norm * i_prec[abs_start_idx: abs_end_idx]
            v_if_prec = (
                                simpson(v_if_prec, x=wn_grid[abs_start_idx: abs_end_idx]) << v_if_prec.unit
                        ) * b_if
            v_if_prec = v_if_prec.decompose()
            log.debug(f"[L{layer_idx}] V_{trans_row[1], trans_row[0]}_prec = {v_if_prec}")

            y_matrix[trans_row[0], trans_row[1]] += v_if_prec
            y_matrix[trans_row[1], trans_row[0]] += u_fi + v_fi_prec
            y_matrix[trans_row[0], trans_row[0]] -= u_fi + v_fi_prec
            y_matrix[trans_row[1], trans_row[1]] -= v_if_prec
        if self.cont_rates is not None:
            for cont_trans_row in self.cont_rates.iter_rows(named=False):
                a_ci = cont_trans_row[1] / u.s
                # b_ci = cont_trans_row[2] * (u.m ** 2) / (u.J * u.s)
                b_ic = cont_trans_row[3] * u.m ** 2 / (u.J * u.s)

                # if cont_trans_row[0] in self.cont_profile_grid[nlte_layer_idx]:
                cont_abs_profile, cont_abs_start_idx = self.cont_profile_store.get_profile(
                    layer_idx=nlte_layer_idx, key=cont_trans_row[0], profile_type="abs"
                )
                # cont_abs_profile = self.cont_profile_grid[nlte_layer_idx].get(cont_trans_row[0])
                # cont_abs_end_idx = cont_abs_profile.start_idx + len(cont_abs_profile.profile)
                cont_abs_end_idx = cont_abs_start_idx + len(cont_abs_profile)

                cont_abs_profile_norm = cont_abs_profile / simpson(
                    cont_abs_profile, x=wn_grid[cont_abs_start_idx:cont_abs_end_idx]
                )

                # Cross terms:
                chi_ci = np.zeros(lambda_layer_grid.shape[0]) << u.m ** 2 / (u.J * u.s)
                chi_ci[cont_abs_start_idx: cont_abs_end_idx] += (
                        self.pop_matrix[-1, layer_idx, cont_trans_row[0]]
                        * cont_abs_profile_norm
                        * cont_trans_row[3]
                        * chem_profile[self.species][layer_idx]
                        * (u.m ** 2)
                        / (u.J * u.s)
                )
                chi_ci = np.where(chi_ci < 0, 0, chi_ci)
                if full_prec:
                    for o_idx in np.arange(0, self.n_agg_states):
                        a_ox_cross = np.zeros(lambda_layer_grid.shape[0]) << u.erg / u.sr
                        for ox_trans in self.rates_grid.filter(pl.col("id_agg_f") == o_idx).iter_rows(named=False):
                            ox_emi_profile, ox_emi_start_idx = self.profile_store.get_profile(
                                layer_idx=nlte_layer_idx, key=(ox_trans[0], ox_trans[1]), profile_type="spe"
                            )
                            # ox_emi_profile = self.emi_profile_grid[nlte_layer_idx].get(
                            #     (ox_trans[0], ox_trans[1])
                            # )
                            ox_emi_end_idx = ox_emi_start_idx + len(ox_emi_profile)

                            a_ox_cross[ox_emi_start_idx: ox_emi_end_idx] += (
                                    ox_emi_profile  # Absolute value, not normalised.
                                    * u.erg
                                    * u.cm
                                    * chem_profile[self.species][layer_idx]
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
                        ).to(u.J / (u.m ** 2 * u.sr), equivalencies=u.spectral())
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
                v_ic_prec = cont_abs_profile_norm * i_prec[cont_abs_start_idx: cont_abs_end_idx]
                v_ic_prec = (
                                    simpson(v_ic_prec, x=wn_grid[cont_abs_start_idx: cont_abs_end_idx])
                                    << v_ic_prec.unit
                            ) * b_ic
                v_ic_prec = v_ic_prec.decompose()
                log.debug(f"[L{layer_idx}] V_ic_prec = {v_ic_prec}")

                limiting_species_num_dens = min(
                    (
                        chem_profile[self.dissociation_products[0]][layer_idx]
                        if self.dissociation_products[0] in chem_profile.species
                        else 0
                    ),
                    (
                        chem_profile[self.dissociation_products[1]][layer_idx]
                        if self.dissociation_products[1] in chem_profile.species
                        else 0
                    ),
                )
                if limiting_species_num_dens == 0:
                    limiting_scale_factor = 0
                else:
                    mol_num_dens = chem_profile[self.species][layer_idx]
                    i_pop = self.pop_matrix[-1, layer_idx, cont_trans_row[0]]
                    limiting_scale_factor = i_pop * mol_num_dens / limiting_species_num_dens

                y_matrix[cont_trans_row[0], cont_trans_row[0]] -= v_ic_prec
                # y_matrix[cont_trans_row[0], cont_trans_row[0]] += a_ci * z_ci * limiting_scale_factor
                y_matrix[cont_trans_row[0], cont_trans_row[0]] += a_ci * limiting_scale_factor
                y_matrix[cont_trans_row[0], cont_trans_row[0]] += v_ic_prec * limiting_scale_factor
        # Add collisional and chemical rates.
        y_matrix = self.add_col_chem_rates(
            y_matrix=y_matrix,
            layer_idx=layer_idx,
            layer_temp=layer_temperature,
            chem_profile=chem_profile,
            density_profile=density_profile,
        )

        return y_matrix.value

    def add_col_chem_rates(
            self,
            y_matrix: u.Quantity,
            layer_idx: int,
            layer_temp: u.Quantity,
            chem_profile: ChemicalProfile,
            density_profile: u.Quantity,
    ) -> u.Quantity:
        # TODO: Move formation destruction rates to RHS in rate equation as they do not depend on species n!
        def add_col_chem_rate(
                estate_u: str,
                v_u: int,
                estate_l: str,
                v_l: int,
                rate: float,
                mol_depend: str,
        ):
            if mol_depend in chem_profile.species:
                upper_id, upper_energy = (
                    self.agg_states
                    .filter((pl.col("agg1") == estate_u) & (pl.col("agg2") == v_u))
                    .select(["id_agg", "energy_agg"])
                    .row(0)
                )
                lower_id, lower_energy = (
                    self.agg_states
                    .filter((pl.col("agg1") == estate_l) & (pl.col("agg2") == v_l))
                    .select(["id_agg", "energy_agg"])
                    .row(0)  # returns tuple of first row
                )
                # TODO: Add check that state is within energy bounds!?
                # log.debbug(f"Upper = {upper_id}, type ={type(upper_id)}")
                # log.debug(f"Lower = {lower_id}, type ={type(lower_id)}")
                depend_num_dens = (
                        chem_profile[SpeciesFormula(mol_depend)][layer_idx] * density_profile[layer_idx]
                ).to(u.cm ** -3)

                c_fi = (rate * u.cm ** 3 / u.s) * depend_num_dens

                if self.pop_matrix[-1, layer_idx, lower_id] == 0:
                    c_if = 0 * c_fi
                    log.warning((
                        f"[L{layer_idx}] upwards Col/Chem. rate for IDs {upper_id}-{lower_id} 0 from balance due "
                        f"to 0 lower state population."
                    ))
                else:
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
            for v_val in range(0, 10):
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

    def solve_pops(
            self, y_matrix: npt.NDArray[np.float64], pop_grid_update: npt.NDArray[np.float64],
            layer_idx: int, sor_enabled: bool, damping_enabled: bool, species: str, n_iter: int,
    ) -> npt.NDArray[np.float64]:
        y_reduced_idx_map = [idx for idx in range(0, len(y_matrix)) if sum(abs(y_matrix[idx])) != 0]
        y_matrix_reduced = y_matrix[np.ix_(y_reduced_idx_map, y_reduced_idx_map)]
        log.debug((
            f"[L{layer_idx}] {species} Y matrix (before row-normalisation) =\n{y_matrix_reduced}"
            f"[L{layer_idx}] {species} Y matrix cond. "
            f"(before row-normalisation) = {np.linalg.cond(y_matrix_reduced)}"
        ))
        y_matrix_reduced /= abs(y_matrix_reduced).sum(axis=1)[:, None]
        check_rows = np.array([
            np.all(y_matrix_reduced[idx, :] > 0) or np.all(y_matrix_reduced[idx, :] < 0)
            for idx in range(y_matrix_reduced.shape[0])
        ])
        if np.any(check_rows):
            major = (
                f"[I{n_iter}][L{layer_idx}] Y matrix all same sign in rows "
                f"{np.nonzero(check_rows)[0]}; investigate unphysical rates."
            )
            log.error(major)
        y_rect = np.vstack([y_matrix_reduced.copy(), np.ones(y_matrix_reduced.shape[1])])
        rhs_rect = np.zeros(y_rect.shape[0])
        rhs_rect[-1] = 1

        nppinv_pops = np.linalg.pinv(y_rect) @ rhs_rect
        nppinv_pops /= nppinv_pops.sum()

        if np.any(nppinv_pops < 0):
            log.error((
                f"[L{layer_idx}] Numpy Pseudo Inverse pops. contain negatives. "
                f"Falling back to least squares...\nNegatives = {nppinv_pops}"
            ))
            lsq_res = least_squares(
                lambda x: np.dot(y_rect, x) - rhs_rect,
                np.zeros(y_rect.shape[1]),
                bounds=(0.0, 1.0),
                method="trf",
                ftol=1e-15,
                gtol=1e-15,
                xtol=1e-15,
            )
            least_squares_pops = lsq_res.x
            log.debug((
                f"[L{layer_idx}] Least Squares res = {lsq_res}\n"
                f"Least Squares Pops. = {least_squares_pops}"
            ))
            if any(least_squares_pops < 0):
                raise RuntimeError(
                    f"[L{layer_idx}] Least squares population bounds failed; negative pops."
                )
            else:
                pop_matrix = least_squares_pops
        else:
            pop_matrix = nppinv_pops

        if damping_enabled:
            damping_factor = 0.5
            pop_matrix = (
                    damping_factor * pop_matrix
                    + (1 - damping_factor) * self.pop_matrix[-1, layer_idx, y_reduced_idx_map]
            )
            log.info(f"[L{layer_idx}] New pops. (damped):")
        elif sor_enabled:
            pop_delta = pop_matrix - self.pop_matrix[-1, layer_idx, y_reduced_idx_map]
            with np.errstate(divide="ignore"):
                current_max_change = max(
                    abs(pop_delta) / self.pop_matrix[-1, layer_idx, y_reduced_idx_map])
                previous_max_change = max(
                    abs(
                        self.pop_matrix[-1, layer_idx, y_reduced_idx_map]
                        - self.pop_matrix[-2, layer_idx, y_reduced_idx_map]
                    )
                    / self.pop_matrix[-2, layer_idx, y_reduced_idx_map]
                )
                sor_delta = current_max_change / previous_max_change
            if sor_delta > 1:
                log.warning(
                    f"[L{layer_idx}] SOR delta greater than 1 ({sor_delta}) - limiting to 1."
                )

            sor_delta = min(sor_delta, 1.0)
            log.info(f"[L{layer_idx}] SOR delta = {sor_delta}")
            sor_omega = 2 / (1 + np.sqrt(1 - sor_delta))
            sor_omega = min(max(sor_omega, 0.8), 1.4)

            pop_matrix = self.pop_matrix[-1, layer_idx, y_reduced_idx_map] + sor_omega * pop_delta
            if np.any(pop_matrix < 0):
                log.warning(
                    f"[L{layer_idx}] SOR step lead to negative population(s) - scaling back SOR omega."
                )
                pop_matrix[pop_matrix < 0] = 0

            pop_matrix /= pop_matrix.sum()
            log.info(f"[L{layer_idx}] New pops. (SOR factor = {sor_omega})")
        else:
            log.info(f"[L{layer_idx}] New pops.:")
        for idx, y_idx in enumerate(y_reduced_idx_map):
            log.info((
                f"[L{layer_idx}]"
                f" n{self.agg_states.filter(pl.col("id_agg") == y_idx).select(self.agg_col_names).row(0)}"
                f" = {pop_matrix[idx]}"
            ))
        full_pops = np.zeros(self.n_agg_states)
        full_pops[y_reduced_idx_map] = pop_matrix
        pop_grid_update[layer_idx] = full_pops

        n_agg_lte_col = f"n_agg_L{layer_idx}"
        n_lte_col = f"n_L{layer_idx}"
        n_agg_nlte_col = f"n_agg_nlte_L{layer_idx}"
        n_nlte_col = f"n_nlte_L{layer_idx}"
        self.states = self.states.with_columns(
            pl.Series(full_pops).gather(self.states["id_agg"]).alias(n_agg_nlte_col)
        )
        self.states = self.states.with_columns(
            pl.when(pl.col(n_agg_lte_col) == 0)
            .then(0)
            .otherwise(pl.col(n_lte_col) * pl.col(n_agg_nlte_col) / pl.col(n_agg_lte_col))
            .alias(n_nlte_col)
        )
        log_col_names = ["id", "energy", "g", "tau"] + self.agg_col_names + [n_agg_nlte_col, n_nlte_col]
        log.info((
            f"[L{layer_idx}] NLTE States = \n{self.states.select(log_col_names)}\n"
            f"[L{layer_idx}] Sum of LTE populations = {self.states[n_lte_col].sum()}.\n"
            f"[L{layer_idx}] Sum of non-LTE populations = {self.states[n_nlte_col].sum()}."
        ))
        return pop_grid_update

    def update_layer_global_chi_eta(
            self,
            wn_grid: u.Quantity,
            layer_temp: u.Quantity,
            layer_pressure: u.Quantity,
            layer_pop_grid: npt.NDArray[np.float64],
            layer_vmr: float,
            # layer_density: u.Quantity,
            layer_global_chi_matrix: u.Quantity,
            layer_global_eta_matrix: u.Quantity,
            layer_idx: int,
            nlte_layer_idx: int,
    ) -> t.Tuple[u.Quantity, u.Quantity]:
        nlte_states = self.states.select(
            pl.col("id"),
            pl.col("energy"),
            pl.col("g"),
            pl.col("tau"),
            pl.col(f"n_nlte_L{layer_idx}").alias("n_nlte")
        )
        start_time = time.perf_counter()
        # TODO: Replace with looping through band profiles?
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
            # for key in self.cont_profile_grid[nlte_layer_idx].keys():
            for key in self.cont_profile_store.get_keys(layer_idx=nlte_layer_idx, profile_type="abs"):
                n_i = layer_pop_grid[key]
                cont_abs_profile, cont_abs_start_idx = self.cont_profile_store.get_profile(
                    layer_idx=nlte_layer_idx, key=key, profile_type="abs",
                )
                cont_abs_end_idx = cont_abs_start_idx + len(cont_abs_profile)
                abs_xsec[cont_abs_start_idx: cont_abs_end_idx] += cont_abs_profile * n_i

        # Update layer chi globally and then for species.
        abs_xsec = abs_xsec << u.cm ** 2
        layer_global_chi_matrix += (
                (abs_xsec - self.mol_chi_matrix[layer_idx]) * layer_vmr  # * layer_density
        )
        self.mol_chi_matrix[layer_idx] = abs_xsec
        # Update layer eta globally and then for species.
        emi_xsec = emi_xsec << u.erg * u.cm / (u.s * u.sr)
        layer_global_eta_matrix += (
                (emi_xsec - self.mol_eta_matrix[layer_idx]) * layer_vmr
        )
        self.mol_eta_matrix[layer_idx] = emi_xsec

        return layer_global_chi_matrix, layer_global_eta_matrix

    def update_pops(
            self, pop_grid_updated: npt.NDArray[np.float64], n_lte_layers: int, do_sor: bool, sor_enabled: bool,
            n_iter: int
    ) -> t.Tuple[float, bool, bool]:
        self.pop_matrix = np.vstack(
            (self.pop_matrix, pop_grid_updated.reshape((1, self.pop_matrix.shape[1], self.pop_matrix.shape[2])))
        )
        # TEMP!
        with open(
                (output_dir / f"{self.species}_pop_matrix.pickle").resolve(), "wb"
        ) as pickle_file:
            pickle.dump(self.pop_matrix, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

        n_layers = pop_grid_updated.shape[0]
        max_pop_changes = np.empty(n_layers - n_lte_layers)
        # importance = self.chem_profile[self.species] / self.chem_profile[self.species].max()

        for nlte_layer_idx in range(n_layers - n_lte_layers):
            # layer_old_pops = self.pop_grid[self.n_lte_layers + nlte_layer_idx]
            # layer_new_pops = pop_grid[self.n_lte_layers + nlte_layer_idx]
            layer_old_pops = self.pop_matrix[-2, n_lte_layers + nlte_layer_idx]
            layer_new_pops = self.pop_matrix[-1, n_lte_layers + nlte_layer_idx]

            non_zero_idx_map = (layer_old_pops != 0) & (layer_new_pops != 0)
            layer_delta_pops = layer_new_pops[non_zero_idx_map] - layer_old_pops[non_zero_idx_map]
            layer_changes = np.abs(layer_delta_pops / layer_old_pops[non_zero_idx_map])
            max_pop_changes[nlte_layer_idx] = (
                layer_changes.max()
            )  # * importance[self.n_lte_layers + nlte_layer_idx]

        # Check for oscillations
        check_iter = 6
        damping_enabled = False
        if n_iter > check_iter:
            check_changes = np.where(
                (self.pop_matrix[-check_iter:, n_lte_layers:, :] == 0)
                & (self.pop_matrix[-check_iter - 1: -1, n_lte_layers:, :] == 0),
                0,
                abs(
                    self.pop_matrix[-check_iter:, n_lte_layers:, :]
                    - self.pop_matrix[-check_iter - 1: -1, n_lte_layers:, :]
                )
                / self.pop_matrix[-check_iter - 1: -1, n_lte_layers:, :],
            ).max(axis=(1, 2))
            change_dif = np.diff(check_changes)
            oscillating = change_dif[:-1] * change_dif[1:]
            oscillating = oscillating < 0
            log.info(f"[I{n_iter}] Oscillating? {oscillating}")
            if sum(oscillating) >= 3:
                log.info((
                    f"[I{n_iter}] Oscillations detected in {sum(oscillating)}/{check_iter}"
                    f" previous iterations - damping enabled."
                ))
                damping_enabled = True

        max_pop_change = max_pop_changes.max()
        # TODO: Flag when levels oscillate between 0 and extremely small values, blocking convergence, to fix them
        #  to 0?
        log.info((
            f"[I{n_iter}] Maximum population changes per layer = {max_pop_changes}"
            f" (Max. = {max_pop_change})"
        ))
        if do_sor and not sor_enabled and n_iter >= 4 and max_pop_change <= 0.01:
            log.info(f"[I{n_iter}] SOR threshold reached - SOR enabled.")
            sor_enabled = True

        return max_pop_change, damping_enabled, sor_enabled

    def finalise(self, temperature_profile: u.Quantity, pressure_profile: u.Quantity, wn_grid: u.Quantity) -> None:
        """
        Saves the population matrix, absorption and emission cross-sections profile to disk. Recomputes the
        cross-sections on the input grid, allowing for computation on a high-resolution grid after iteration.

        :param temperature_profile:
        :param pressure_profile:
        :param wn_grid:
        :return:
        """
        with open((output_dir / f"{self.species}_pop_matrix.pickle").resolve(), "wb") as pickle_file:
            pickle.dump(self.pop_matrix, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        # self.mol_chi_matrix = super().opacity(temperature, pressure, spectral_grid)
        # self.mol_source_func_matrix = blackbody(spectral_grid=spectral_grid, temperature=temperature)
        # self.mol_eta_matrix = self.mol_source_func_matrix * self.mol_chi_matrix * ac.c
        for layer_idx in range(self.n_lte_layers, len(temperature_profile)):
            layer_temp = temperature_profile[layer_idx]
            layer_pres = pressure_profile[layer_idx]

            n_agg_lte_col = f"n_agg_L{layer_idx}"
            n_lte_col = f"n_L{layer_idx}"
            n_agg_nlte_col = f"n_agg_nlte_L{layer_idx}"
            n_nlte_col = f"n_nlte_L{layer_idx}"
            self.states = self.states.with_columns(
                pl.Series(self.pop_matrix[-1, layer_idx]).gather(self.states["id_agg"]).alias(n_agg_nlte_col)
            )
            self.states = self.states.with_columns(
                pl.when(pl.col(n_agg_lte_col) == 0)
                .then(0)
                .otherwise(pl.col(n_lte_col) * pl.col(n_agg_nlte_col) / pl.col(n_agg_lte_col))
                .alias(n_nlte_col)
            )

            abs_xsec, emi_xsec = abs_emi_xsec(
                states=self.states,
                trans_files=self.trans_files,
                temperature=layer_temp,
                pressure=layer_pres,
                species_mass=self.species_mass,
                wn_grid=wn_grid.value,
                broad_n=(self.broadening_params[1] if self.broadening_params is not None else None),
                broad_gamma=(
                    self.broadening_params[0][:, layer_idx] if self.broadening_params is not None else None
                ),
            )
            if self.cont_states is not None:
                nlte_select_cols = [pl.col("id"), pl.col(n_nlte_col)]
                nlte_cont_states = self.cont_states.join(self.states.select(nlte_select_cols), on="id", how="left")
                cont_xsec = continuum_xsec(
                    continuum_states=nlte_cont_states,
                    continuum_trans_files=self.cont_trans_files,
                    layer_idx=layer_idx,
                    wn_grid=wn_grid.value,
                    temperature=layer_temp,
                    species_mass=self.species_mass,
                    cont_box_length=self.cont_box_length
                )
                abs_xsec += cont_xsec
                np.savetxt(
                    (
                            output_dir
                            / f"nLTE_cxsec_L{layer_idx}_T{int(layer_temp.value)}_P{layer_pres.value:.4e}.txt"
                    ).resolve(),
                    np.array([wn_grid.value, cont_xsec]).T,
                    fmt="%17.8E",
                )
                # LTE Comparison
                lte_select_cols = [pl.col("id"), pl.col(n_lte_col)]
                lte_cont_states = self.cont_states.join(self.states.select(lte_select_cols), on="id", how="left")
                # lte_cont_states = self.cont_states.merge(nlte_states[["id", "n"]], on="id", how="left")
                lte_cont_states["n_nlte"] = lte_cont_states["n"]
                lte_cont_xsec = continuum_xsec(
                    continuum_states=lte_cont_states,
                    continuum_trans_files=self.cont_trans_files,
                    layer_idx=layer_idx,
                    wn_grid=wn_grid.value,
                    temperature=layer_temp,
                    species_mass=self.species_mass,
                    cont_box_length=self.cont_box_length
                )
                np.savetxt(
                    (
                            output_dir
                            / f"LTE_cxsec_L{layer_idx}_T{int(layer_temp.value)}_P{layer_pres.value:.4e}.txt"
                    ).resolve(),
                    np.array([wn_grid.value, lte_cont_xsec]).T,
                    fmt="%17.8E",
                )
            abs_xsec = abs_xsec << u.cm ** 2
            self.mol_chi_matrix[layer_idx] = abs_xsec
            emi_xsec = emi_xsec << u.erg * u.cm / (u.s * u.sr)
            self.mol_eta_matrix[layer_idx] = emi_xsec
        with open((output_dir / f"{self.species}_abs_xsec.pickle").resolve(), "wb") as abs_pickle_file:
            pickle.dump(self.mol_chi_matrix, abs_pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        with open((output_dir / f"{self.species}_emi_xsec.pickle").resolve(), "wb") as emi_pickle_file:
            pickle.dump(self.mol_eta_matrix, emi_pickle_file, protocol=pickle.HIGHEST_PROTOCOL)


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


class ExomolNLTEXsec(ExomolHDF5Xsec):

    def __init__(
            self,
            species: str | SpeciesFormula,
            species_mass: float,
            states_file: pathlib.Path,
            trans_files: pathlib.Path | t.List[pathlib.Path],
            agg_col_nums: t.List[int],
            planet_radius: u.Quantity,
            chem_profile: ChemicalProfile,
            broadening_params: t.Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] = None,
            intensity_threshold: np.float64 = 1e-35,
            n_lte_layers: int = 0,
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
        # self.species_mass = species_mass
        # if type(states_file) is str:
        #     states_file = pathlib.Path(states_file)
        # self.states_file = states_file
        # if type(trans_files) is str:
        #     trans_files = [pathlib.Path(trans_files)]
        # elif type(trans_files) is not list:
        #     trans_files = [trans_files]
        # self.trans_files = trans_files
        # self.agg_col_nums = agg_col_nums
        # self.agg_col_names = ["agg" + str(idx + 1) for idx in range(0, len(self.agg_col_nums))]
        # self.planet_radius = planet_radius
        # self.chem_profile = chem_profile
        # self.broadening_params = broadening_params
        # self.intensity_threshold = intensity_threshold
        # self.n_agg_states = None
        # self.states = None
        # self.agg_states = None
        # self.density_profile = None
        # self.dz_profile = None
        # self.tau_matrix = None
        # self.rates_grid = None
        # self.abs_profile_grid = None
        # self.emi_profile_grid = None
        # self.mol_source_func_matrix = None
        # self.global_source_func_matrix = None
        # self.mol_chi_matrix = None
        # self.mol_eta_matrix = None
        # self.global_chi_matrix = None  # density and VMR weighted sum of all cross sections
        # self.global_eta_matrix = None
        # self.intensity_matrix = None
        # self.pop_matrix = None
        # self.is_converged = False
        # self.n_lte_layers = n_lte_layers
        # self.lte_grid_file = lte_grid_file
        # self.cont_rates = None
        # if type(cont_states_file) is str:
        #     cont_states_file = pathlib.Path(cont_states_file)
        # self.cont_states_file = cont_states_file
        # self.cont_states = None
        # if type(cont_trans_files) is str:
        #     cont_trans_files = [pathlib.Path(cont_trans_files)]
        # elif type(cont_trans_files) is not list:
        #     cont_trans_files = [cont_trans_files]
        # self.cont_trans_files = cont_trans_files
        # self.cont_profile_grid = None
        # self.dissociation_products = dissociation_products
        # self.incident_radiation_field = incident_radiation_field
        # self.sor = sor
        # self.sor_enabled = False
        # self.full_prec = True
        # self.damping_enabled = False
        self.load_in_memory = load_in_memory
        # self.n_iter = 0
        # self.debug = debug
        # self.negative_source_func_cap = None
        # self.negative_absorption_factor = 0.1
        # self.debug_pop_matrix = debug_pop_matrix

        super().__init__(lte_grid_file, self.load_in_memory)

        self.nlte_processor = NLTEProcessor(
            species=self.species,
            states_file=states_file,
            trans_files=trans_files,
            agg_col_nums=agg_col_nums,
            species_mass=species_mass,
            broadening_params=broadening_params,
            n_lte_layers=n_lte_layers,
            cont_states_file=cont_states_file,
            cont_trans_files=cont_trans_files,
            dissociation_products=dissociation_products,
            debug=debug,
            debug_pop_matrix=debug_pop_matrix,
        )

    def get_nlte_processor(self) -> NLTEProcessor:
        return getattr(self, 'nlte_processor', None)

    # def setup(
    #         self,
    #         chem_profile: ChemicalProfile,
    #         temperature_profile: u.Quantity,
    #         pressure_profile: u.Quantity,
    #         wn_grid: u.Quantity
    # ) -> None:
    #     if any([mol not in chem_profile.species for mol in self.dissociation_products]):
    #         warn_string = (
    #             f"Specified dissociation products {self.dissociation_products} not present in"
    #             f" chemical profile {chem_profile.species}."
    #         )
    #         log.warning(warn_string)
    #         # raise RuntimeError(warn_string)
    #
    #     self.aggregate_states(temperature_profile=temperature_profile, energy_cutoff=wn_grid[-1].value)
    #     self.compute_rates_profiles(
    #         temperature_profile=temperature_profile,
    #         pressure_profile=pressure_profile,
    #         wn_grid=wn_grid,
    #     )
    #     if self.cont_states_file is not None and self.cont_trans_files is not None:
    #         self.load_continuum_rates(temperature_profile=temperature_profile, wn_grid=wn_grid)
    #
    #     self.mol_chi_matrix = super().opacity(temperature_profile, pressure_profile, wn_grid)
    #     lte_source_func_matrix = blackbody(
    #         spectral_grid=wn_grid, temperature=temperature_profile
    #     )
    #     self.mol_eta_matrix = lte_source_func_matrix * self.mol_chi_matrix * ac.c

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


def bezier_debug(
        layer_idx: int,
        i_matrix: u.Quantity,
        lambda_matrix: npt.NDArray[np.float64],
        direction: str,
) -> None:
    if not np.all(i_matrix[layer_idx] >= 0) or not np.all(lambda_matrix[layer_idx] >= 0):
        # coefs_check_idx = layer_idx if direction == "out" else layer_idx + 1
        # control_point_idx = 0 if direction == "out" else 1
        also_check_idx = layer_idx - 1 if direction == "out" else layer_idx + 1
        if not np.all(i_matrix[layer_idx] >= 0):
            log.warning(f"[L{layer_idx}] Warn: {direction} INTENSITY BEZIER BAD :(")
            log.warning(f"Negative intensities in previous layer? {np.any(i_matrix[also_check_idx] < 0)}")


class XSecCollection(dict):

    def __init__(
            self,
            intensity_threshold: np.float64 = 1e-35,
            n_lte_layers: int = 0,
            incident_radiation_field: u.Quantity = None,
            sor: bool = True,
            debug: bool = False,
    ) -> None:
        self.intensity_threshold = intensity_threshold
        self.global_source_func_matrix = None
        self.global_chi_matrix = None  # density and VMR weighted sum of all cross-sections
        self.global_eta_matrix = None
        self.intensity_matrix = None
        self.is_converged = False
        self.n_lte_layers = n_lte_layers
        self.incident_radiation_field = incident_radiation_field
        self.sor = sor
        self.sor_enabled = False  # Used to turn on SOR if available after conditions met
        self.full_prec = True  # Internal debug testing
        self.do_tridiag = True  # Internal debug testing
        self.damping_enabled = False
        self.n_iter = 0
        self.debug = debug
        self.negative_source_func_cap = None
        self.negative_absorption_factor = 0.1

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
        self[xsec_data.species] = xsec_data

    def compute_opacities_profile(
            self,
            chemical_profile: ChemicalProfile,
            density_profile: u.Quantity,
            dz_profile: u.Quantity,
            temperature: u.Quantity,
            pressure: u.Quantity,
            spectral_grid: u.Quantity,
    ) -> t.Dict[SpeciesFormula, u.Quantity]:
        active_species = self.active_absorbers(chemical_profile.species)
        nlte_species = np.array([
            species for species in active_species
            if is_nlte_xsec(self[species])
        ])

        output_opacities = {
            species: self[species].opacity(
                temperature, pressure, spectral_grid
            ) * chemical_profile[species][:, None]
            for species in active_species
        }

        # TODO: Add check to fit T_ex for each layer in first combined inward+outward pass, use for boltzmann (not T_k).

        n_layers = temperature.shape[0]

        # -------------------------- GLOBAL PROPERTY CONFIGURATION --------------------------
        self.global_chi_matrix: u.Quantity = np.zeros((n_layers, spectral_grid.shape[0])) << u.cm ** 2
        self.global_eta_matrix: u.Quantity = np.zeros(self.global_chi_matrix.shape) << u.erg * u.cm / (u.s * u.sr)
        lte_source_func = blackbody(spectral_grid, temperature)

        for species in active_species:
            xsec_data = self[species]
            if is_nlte_xsec(xsec_data):
                log.info(f"[I{self.n_iter}] Initial LTE set up for {species}.")
                processor = xsec_data.get_nlte_processor()
                processor.setup(
                    chem_profile=chemical_profile,
                    temperature_profile=temperature,
                    pressure_profile=pressure,
                    wn_grid=spectral_grid,
                    initial_chi_matrix=output_opacities[species],
                )
                self.global_chi_matrix += processor.mol_chi_matrix
                self.global_eta_matrix += processor.mol_eta_matrix
            else:
                self.global_chi_matrix += output_opacities[species]
                self.global_eta_matrix += output_opacities[species] * lte_source_func * ac.c * \
                                          chemical_profile[species][:, None]
        # Units for Emission/Absorption requrie extra 1/c factor for conversion.
        source_func_units = u.J / (u.sr * u.m ** 2)
        zero_chi_mask = self.global_chi_matrix == 0
        self.global_source_func_matrix: u.Quantity = np.zeros(self.global_eta_matrix.shape) * source_func_units
        self.global_source_func_matrix[~zero_chi_mask] = (
                self.global_eta_matrix[~zero_chi_mask] / (ac.c * self.global_chi_matrix[~zero_chi_mask])
        )

        if nlte_species.size == 0:
            # Early exit when no NLTE species.
            return output_opacities

        # Perform Gauss-Seidel passes.
        n_angular_points = 50
        mu_values, mu_weights = np.polynomial.legendre.leggauss(n_angular_points)
        mu_values, mu_weights = (mu_values + 1) * 0.5, mu_weights / 2

        # Begin iterative solution.

        while not self.is_converged:
            # res = self.global_chi_matrix * dz_profile[:, None]
            # dtau = res.decompose().value
            # tau = dtau[::-1].cumsum(axis=0)[::-1]
            # tau_mu = tau[:, None, :] / mu_values[None, :, None]
            effective_source_func_matrix, effective_tau_mu = effective_source_tau_mu(
                global_source_func_matrix=self.global_source_func_matrix,
                global_chi_matrix=self.global_chi_matrix,
                global_eta_matrix=self.global_eta_matrix,
                density_profile=density_profile,
                dz_profile=dz_profile,
                mu_values=mu_values,
                negative_absorption_factor=self.negative_absorption_factor,
            )
            start_time = time.perf_counter()
            bezier_coefs, control_points = bezier_coefficients(
                tau_mu_matrix=effective_tau_mu,
                source_function_matrix=effective_source_func_matrix.value,
                # tau_mu_matrix=tau_mu,
                # source_function_matrix=self.global_source_func_matrix.value,
            )
            control_points: u.Quantity = control_points << self.global_source_func_matrix.unit
            log.info(f"Coefficient duration (old) = {time.perf_counter() - start_time}")
            start_time = time.perf_counter()
            bezier_coefs_new, control_points_new = bezier_coefficients_new(
                tau_mu_matrix=effective_tau_mu,
                source_function_matrix=effective_source_func_matrix.value,
            )
            log.info(f"Coefficient duration (new) = {time.perf_counter() - start_time}")
            log.info(
                f"Coefs equal? {np.all(bezier_coefs == bezier_coefs_new)} {np.allclose(bezier_coefs, bezier_coefs_new, atol=1e-7)}")
            log.info(
                f"Control equal? {np.all(control_points == control_points_new)} {np.allclose(control_points, control_points_new, atol=1e-7)}")

            # USEFUL BEZIER IDENTITIES
            alpha_plus_gamma = bezier_coefs[:, 1] + bezier_coefs[:, 3]
            one_plus_exp_neg_delta_tau = 1 + bezier_coefs[:, 0]

            i_in_matrix: u.Quantity = np.zeros_like(effective_tau_mu) << self.global_source_func_matrix.unit
            lambda_in_matrix = np.zeros_like(effective_tau_mu)
            ################
            pop_grid_updates = {}
            for species in nlte_species:
                xsec_data = self[species]
                if is_nlte_xsec(xsec_data):
                    processor = xsec_data.get_nlte_processor()
                    pop_grid_updates[species] = processor.pop_matrix[-1, :, :].copy()
            # INWARD PASS
            # Upper boundary condition if incident radiation field present.
            if self.incident_radiation_field is not None:
                i_in_matrix[-1] = self.incident_radiation_field

            for layer_idx in range(n_layers - 1)[::-1]:
                # Inward intensity interpolation.
                i_in_matrix[layer_idx] = (
                        i_in_matrix[layer_idx + 1] * bezier_coefs[layer_idx + 1, 0] +
                        bezier_coefs[layer_idx + 1, 1] * effective_source_func_matrix[layer_idx, None, :] +
                        bezier_coefs[layer_idx + 1, 2] * effective_source_func_matrix[layer_idx + 1, None, :] +
                        bezier_coefs[layer_idx + 1, 3] * control_points[layer_idx, 1]
                )

                # Inwards Lambda operator calculation.
                if self.do_tridiag and layer_idx > 0:
                    lambda_in_matrix[layer_idx] = (
                            alpha_plus_gamma[layer_idx + 1] * one_plus_exp_neg_delta_tau[layer_idx] +
                            bezier_coefs[layer_idx, 2]
                    )
                else:
                    lambda_in_matrix[layer_idx] = alpha_plus_gamma[layer_idx + 1]

                # if layer_idx == n_layers - 2:
                #     i_in_matrix[layer_idx] = (
                #             i_in_matrix[layer_idx + 1] * np.exp(-bezier_coefs[layer_idx + 1, 0])
                #             + np.sum(
                #                 bezier_coefs[layer_idx + 1, 1:3]
                #                 * effective_source_func_matrix[layer_idx: layer_idx + 2][:, None, :],
                #                 axis=0,
                #             )
                #             + bezier_coefs[layer_idx + 1, 3] * control_points[layer_idx, 1]
                #     )
                #     if self.do_tridiag:
                #         lambda_in_matrix[layer_idx] = (
                #                 (bezier_coefs[layer_idx + 1, 1] + bezier_coefs[layer_idx + 1, 3])
                #                 * (1 + np.exp(-bezier_coefs[layer_idx, 0]))
                #                 + bezier_coefs[layer_idx, 2]
                #             # + bezier_coefs[layer_idx, 3]
                #         )
                #     else:
                #         lambda_in_matrix[layer_idx] = bezier_coefs[layer_idx + 1, 1] + bezier_coefs[layer_idx + 1, 3]
                #     if not np.all(i_in_matrix[layer_idx] >= 0) or not np.all(lambda_in_matrix[layer_idx] >= 0):
                #         bezier_debug(
                #             layer_idx,
                #             i_in_matrix,
                #             lambda_in_matrix,
                #             "in",
                #         )
                # elif layer_idx == 0:
                #     i_in_matrix[layer_idx] = (
                #             i_in_matrix[layer_idx + 1] * np.exp(-bezier_coefs[layer_idx + 1, 0])
                #             + np.sum(
                #                 bezier_coefs[layer_idx + 1, 1:3]
                #                 * effective_source_func_matrix[layer_idx: layer_idx + 2][:, None, :],
                #                 axis=0,
                #             )
                #             + bezier_coefs[layer_idx + 1, 3] * control_points[layer_idx, 1]
                #     )
                #     lambda_in_matrix[layer_idx] = bezier_coefs[layer_idx + 1, 1] + bezier_coefs[layer_idx + 1, 3]
                #     if not np.all(i_in_matrix[layer_idx] >= 0) or not np.all(lambda_in_matrix[layer_idx] >= 0):
                #         bezier_debug(
                #             layer_idx,
                #             i_in_matrix,
                #             lambda_in_matrix,
                #             "in",
                #         )
                # else:
                #     i_in_matrix[layer_idx] = (
                #             i_in_matrix[layer_idx + 1] * np.exp(-bezier_coefs[layer_idx + 1, 0])
                #             + np.sum(
                #                 bezier_coefs[layer_idx + 1, 1:3]
                #                 * effective_source_func_matrix[layer_idx: layer_idx + 2][:, None, :],
                #                 axis=0,
                #             )
                #             + bezier_coefs[layer_idx + 1, 3] * control_points[layer_idx, 1]
                #     )
                #     if self.do_tridiag:
                #         lambda_in_matrix[layer_idx] = (
                #                 (bezier_coefs[layer_idx + 1, 1] + bezier_coefs[layer_idx + 1, 3])
                #                 * (1 + np.exp(-bezier_coefs[layer_idx, 0]))
                #                 + bezier_coefs[layer_idx, 2]
                #             # + bezier_coefs[layer_idx, 3]
                #         )
                #     else:
                #         lambda_in_matrix[layer_idx] = bezier_coefs[layer_idx + 1, 1] + bezier_coefs[layer_idx + 1, 3]
                #     if not np.all(i_in_matrix[layer_idx] >= 0) or not np.all(lambda_in_matrix[layer_idx] >= 0):
                #         bezier_debug(
                #             layer_idx,
                #             i_in_matrix,
                #             lambda_in_matrix,
                #             "in",
                #         )
            # OUTWARD PASS
            i_out_matrix: u.Quantity = np.zeros_like(effective_tau_mu) << self.global_source_func_matrix.unit
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
                            bezier_coefs[layer_idx, 3] * control_points[layer_idx, 0]
                    )
                    # Outward Lambda operator calculation.
                    if self.do_tridiag and layer_idx < n_layers - 1:
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
                                bezier_coefs[layer_idx + 1, 3] * control_points[layer_idx, 1]
                        )

                        if self.do_tridiag and layer_idx > 0:
                            lambda_in_matrix[layer_idx] = (
                                    alpha_plus_gamma[layer_idx + 1] * one_plus_exp_neg_delta_tau[layer_idx] +
                                    bezier_coefs[layer_idx, 2]
                            )
                        else:
                            lambda_in_matrix[layer_idx] = alpha_plus_gamma[layer_idx + 1]
                # elif layer_idx == 1:
                #     i_out_matrix[layer_idx] = (
                #             i_out_matrix[layer_idx - 1] * np.exp(-bezier_coefs[layer_idx, 0])
                #             + np.sum(
                #                 bezier_coefs[layer_idx, 1:3]
                #                 * effective_source_func_matrix[layer_idx - 1: layer_idx + 1][::-1, None, :],
                #                 axis=0,
                #             )
                #             + bezier_coefs[layer_idx, 3] * control_points[layer_idx, 0]
                #     )
                #     if self.do_tridiag:
                #         lambda_out_matrix[layer_idx] = (
                #                 (bezier_coefs[layer_idx, 1] + bezier_coefs[layer_idx, 3])
                #                 * (1 + np.exp(-bezier_coefs[layer_idx + 1, 0]))
                #                 + bezier_coefs[layer_idx + 1, 2]
                #             # + bezier_coefs[layer_idx + 1, 3]
                #         )
                #     else:
                #         lambda_out_matrix[layer_idx] = bezier_coefs[layer_idx, 1] + bezier_coefs[layer_idx, 3]
                #     if not np.all(i_out_matrix[layer_idx] >= 0) or not np.all(lambda_out_matrix[layer_idx] >= 0):
                #         bezier_debug(
                #             layer_idx,
                #             i_out_matrix,
                #             lambda_out_matrix,
                #             "out",
                #         )
                #
                #     i_in_matrix[layer_idx] = (
                #             i_in_matrix[layer_idx + 1] * np.exp(-bezier_coefs[layer_idx + 1, 0])
                #             + np.sum(
                #                 bezier_coefs[layer_idx + 1, 1:3]
                #                 * effective_source_func_matrix[layer_idx: layer_idx + 2][:, None, :],
                #                 axis=0,
                #             )
                #             + bezier_coefs[layer_idx + 1, 3] * control_points[layer_idx, 1]
                #     )
                #     if self.do_tridiag:
                #         lambda_in_matrix[layer_idx] = (
                #                 (bezier_coefs[layer_idx + 1, 1] + bezier_coefs[layer_idx + 1, 3])
                #                 * (1 + np.exp(-bezier_coefs[layer_idx, 0]))
                #                 + bezier_coefs[layer_idx, 2]
                #             # + bezier_coefs[layer_idx, 3]
                #         )
                #     else:
                #         lambda_in_matrix[layer_idx] = bezier_coefs[layer_idx + 1, 1] + bezier_coefs[layer_idx + 1, 3]
                #     if not np.all(i_in_matrix[layer_idx] >= 0) or not np.all(lambda_in_matrix[layer_idx] >= 0):
                #         bezier_debug(
                #             layer_idx,
                #             i_in_matrix,
                #             lambda_in_matrix,
                #             "in",
                #         )
                # elif layer_idx < n_layers - 1:
                #     i_out_matrix[layer_idx] = (
                #             i_out_matrix[layer_idx - 1] * np.exp(-bezier_coefs[layer_idx, 0])
                #             + np.sum(
                #                 bezier_coefs[layer_idx, 1:3]
                #                 * effective_source_func_matrix[layer_idx - 1: layer_idx + 1][::-1, None, :],
                #                 axis=0,
                #             )
                #             + bezier_coefs[layer_idx, 3] * control_points[layer_idx, 0]
                #     )
                #     if self.do_tridiag:
                #         lambda_out_matrix[layer_idx] = (
                #                 (bezier_coefs[layer_idx, 1] + bezier_coefs[layer_idx, 3])
                #                 * (1 + np.exp(-bezier_coefs[layer_idx + 1, 0]))
                #                 + bezier_coefs[layer_idx + 1, 2]
                #             # + bezier_coefs[layer_idx + 1, 3]
                #         )
                #     else:
                #         lambda_out_matrix[layer_idx] = bezier_coefs[layer_idx, 1] + bezier_coefs[layer_idx, 3]
                #     if not np.all(i_out_matrix[layer_idx] >= 0) or not np.all(lambda_out_matrix[layer_idx] >= 0):
                #         bezier_debug(
                #             layer_idx,
                #             i_out_matrix,
                #             lambda_out_matrix,
                #             "out",
                #         )
                #
                #     i_in_matrix[layer_idx] = (
                #             i_in_matrix[layer_idx + 1] * np.exp(-bezier_coefs[layer_idx + 1, 0])
                #             + np.sum(
                #                 bezier_coefs[layer_idx + 1, 1:3]
                #                 * effective_source_func_matrix[layer_idx: layer_idx + 2][:, None, :],
                #                 axis=0,
                #             )
                #             + bezier_coefs[layer_idx + 1, 3] * control_points[layer_idx, 1]
                #     )
                #     if self.do_tridiag:
                #         lambda_in_matrix[layer_idx] = (
                #                 (bezier_coefs[layer_idx + 1, 1] + bezier_coefs[layer_idx + 1, 3])
                #                 * (1 + np.exp(-bezier_coefs[layer_idx, 0]))
                #                 + bezier_coefs[layer_idx, 2]
                #             # + bezier_coefs[layer_idx, 3]
                #         )
                #     else:
                #         lambda_in_matrix[layer_idx] = bezier_coefs[layer_idx + 1, 1] + bezier_coefs[layer_idx + 1, 3]
                #     if not np.all(i_in_matrix[layer_idx] >= 0) or not np.all(lambda_in_matrix[layer_idx] >= 0):
                #         bezier_debug(
                #             layer_idx,
                #             i_in_matrix,
                #             lambda_in_matrix,
                #             "in",
                #         )
                # else:
                #     # At the upper boundary.
                #     i_out_matrix[layer_idx] = (
                #             i_out_matrix[layer_idx - 1] * np.exp(-bezier_coefs[layer_idx, 0])
                #             + np.sum(
                #                 bezier_coefs[layer_idx, 1:3]
                #                 * effective_source_func_matrix[layer_idx - 1: layer_idx + 1][::-1, None, :],
                #                 axis=0,
                #             )
                #             + bezier_coefs[layer_idx, 3] * control_points[layer_idx, 0]
                #     )
                #     lambda_out_matrix[layer_idx] = bezier_coefs[layer_idx, 1] + bezier_coefs[layer_idx, 3]
                #     if not np.all(i_out_matrix[layer_idx] >= 0) or not np.all(lambda_out_matrix[layer_idx] >= 0):
                #         bezier_debug(
                #             layer_idx,
                #             i_out_matrix,
                #             lambda_out_matrix,
                #             "out",
                #         )
                # Solve equilibrium for non-LTE layers.
                if layer_idx >= self.n_lte_layers:
                    nlte_layer_idx = layer_idx - self.n_lte_layers
                    layer_temp = temperature[layer_idx]
                    layer_pressure = pressure[layer_idx]

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
                    for species in nlte_species:
                        xsec_data = self[species]
                        if is_nlte_xsec(xsec_data):
                            processor = xsec_data.get_nlte_processor()
                            y_mats[species] = processor.build_y_matrix(
                                layer_idx=layer_idx,
                                nlte_layer_idx=nlte_layer_idx,
                                layer_temperature=layer_temp,
                                i_layer_grid=i_layer_grid,
                                lambda_layer_grid=lambda_layer_grid,
                                chem_profile=chemical_profile,
                                density_profile=density_profile,
                                global_chi_matrix=self.global_chi_matrix,
                                global_source_func_matrix=self.global_source_func_matrix,
                                wn_grid=spectral_grid,
                                full_prec=self.full_prec,
                            )
                    # Solve statistical equilibrium for all species and update layer opacities, etc.
                    # These are solved in another loop so that all Y matrices are constructed using the same set of
                    # global parameters, rather than biasing the solution each iteration based on update order.
                    for species in y_mats.keys():
                        xsec_data = self[species]
                        if is_nlte_xsec(xsec_data):
                            processor = xsec_data.get_nlte_processor()
                            pop_grid_update = processor.solve_pops(
                                y_matrix=y_mats[species],
                                pop_grid_update=pop_grid_updates[species],
                                layer_idx=layer_idx,
                                sor_enabled=self.sor_enabled,
                                damping_enabled=self.damping_enabled,
                                species=species,
                                n_iter=self.n_iter,
                            )
                            pop_grid_updates[species] = pop_grid_update

                            self.global_chi_matrix[layer_idx], self.global_eta_matrix[layer_idx] = (
                                processor.update_layer_global_chi_eta(
                                    wn_grid=spectral_grid,
                                    layer_temp=layer_temp,
                                    layer_pressure=layer_pressure,
                                    layer_pop_grid=pop_grid_update[layer_idx],
                                    layer_vmr=chemical_profile[species][layer_idx],
                                    # layer_density=density_profile[layer_idx],
                                    layer_global_chi_matrix=self.global_chi_matrix[layer_idx],
                                    layer_global_eta_matrix=self.global_eta_matrix[layer_idx],
                                    layer_idx=layer_idx,
                                    nlte_layer_idx=nlte_layer_idx,
                                )
                            )
                    #########
                    # Update all physical properties now all Non-LTE species' opacities have been updated.
                    self.global_source_func_matrix[layer_idx] = (
                            self.global_eta_matrix[layer_idx]
                            # * density_profile[layer_idx]  # No longer baking density into global Chi!
                            / (ac.c * self.global_chi_matrix[layer_idx])
                    ).to(u.J / (u.sr * u.m ** 2), equivalencies=u.spectral())

                    effective_source_func_matrix, effective_tau_mu = effective_source_tau_mu(
                        global_source_func_matrix=self.global_source_func_matrix,
                        global_chi_matrix=self.global_chi_matrix,
                        global_eta_matrix=self.global_eta_matrix,
                        density_profile=density_profile,
                        dz_profile=dz_profile,
                        mu_values=mu_values,
                        negative_absorption_factor=self.negative_absorption_factor,
                    )
                    start_time = time.perf_counter()
                    bezier_coefs, control_points = bezier_coefficients(
                        tau_mu_matrix=effective_tau_mu,
                        source_function_matrix=effective_source_func_matrix.value,
                    )
                    control_points = control_points << self.global_source_func_matrix.unit
                    log.info(f"[L{layer_idx}] (OLD) Coefficient (post) duration = {time.perf_counter() - start_time}")

                    log.info(f"Test control points before = {control_points[layer_idx]}")
                    start_time = time.perf_counter()
                    update_layer_coefficients(
                        layer_idx=layer_idx,
                        tau_mu_matrix=effective_tau_mu,
                        source_function_matrix=effective_source_func_matrix.value,
                        coefficients=bezier_coefs,
                        control_points=control_points.value
                    )
                    # TODO: Check that in-place updates work for Quantities; control_points changed?
                    log.info(f"Test control points after = {control_points[layer_idx]}")
                    control_points = control_points << self.global_source_func_matrix.unit
                    if layer_idx > 0:
                        one_plus_exp_neg_delta_tau[layer_idx] = 1 + bezier_coefs[layer_idx, 0]
                        alpha_plus_gamma[layer_idx] = bezier_coefs[layer_idx, 1] + bezier_coefs[layer_idx, 3]
                    if layer_idx < n_layers - 1:
                        one_plus_exp_neg_delta_tau[layer_idx + 1] = 1 + bezier_coefs[layer_idx + 1, 0]
                        alpha_plus_gamma[layer_idx + 1] = bezier_coefs[layer_idx + 1, 1] + bezier_coefs[
                            layer_idx + 1, 3]

                    log.info(f"[L{layer_idx}] (NEW) Coefficient (post) duration = {time.perf_counter() - start_time}")

                    if layer_idx > 0:
                        i_out_matrix[layer_idx] = (
                                i_out_matrix[layer_idx - 1] * bezier_coefs[layer_idx: 0] +
                                bezier_coefs[layer_idx, 1] * effective_source_func_matrix[layer_idx, None, :] +
                                bezier_coefs[layer_idx, 2] * effective_source_func_matrix[layer_idx - 1, None, :] +
                                bezier_coefs[layer_idx, 3] * control_points[layer_idx, 0]
                        )
                        if self.do_tridiag and layer_idx < n_layers - 1:
                            lambda_out_matrix[layer_idx] = (
                                    alpha_plus_gamma[layer_idx] * one_plus_exp_neg_delta_tau[layer_idx + 1] +
                                    bezier_coefs[layer_idx + 1, 2]
                            )
                        else:
                            lambda_out_matrix[layer_idx] = alpha_plus_gamma[layer_idx]
                    # if layer_idx == 1:
                    #     i_out_matrix[layer_idx] = (
                    #             i_out_matrix[layer_idx - 1] * np.exp(-bezier_coefs[layer_idx, 0])
                    #             + np.sum(
                    #                 bezier_coefs[layer_idx, 1:3]
                    #                 * effective_source_func_matrix[layer_idx - 1: layer_idx + 1][::-1, None, :],
                    #                 axis=0,
                    #             )
                    #             + bezier_coefs[layer_idx, 3] * control_points[layer_idx, 0]
                    #     )
                    #     lambda_out_matrix[layer_idx] = (
                    #             (bezier_coefs[layer_idx, 1] + bezier_coefs[layer_idx, 3])
                    #             * (1 + np.exp(-bezier_coefs[layer_idx + 1, 0]))
                    #             + bezier_coefs[layer_idx + 1, 2]
                    #         # + bezier_coefs[layer_idx + 1, 3]
                    #     )
                    #     if not np.all(i_out_matrix[layer_idx] >= 0) or not np.all(lambda_out_matrix[layer_idx] >= 0):
                    #         bezier_debug(
                    #             layer_idx,
                    #             i_out_matrix,
                    #             lambda_out_matrix,
                    #             "out",
                    #         )
                    # elif 1 < layer_idx < self.n_lte_layers - 1:
                    #     i_out_matrix[layer_idx] = (
                    #             i_out_matrix[layer_idx - 1] * np.exp(-bezier_coefs[layer_idx, 0])
                    #             + np.sum(
                    #                 bezier_coefs[layer_idx, 1:3]
                    #                 * effective_source_func_matrix[layer_idx - 1: layer_idx + 1][::-1, None, :],
                    #                 axis=0,
                    #             )
                    #             + bezier_coefs[layer_idx, 3] * control_points[layer_idx, 0]
                    #     )
                    #     lambda_out_matrix[layer_idx] = (
                    #             (bezier_coefs[layer_idx, 1] + bezier_coefs[layer_idx, 3])
                    #             * (1 + np.exp(-bezier_coefs[layer_idx + 1, 0]))
                    #             + bezier_coefs[layer_idx + 1, 2]
                    #         # + bezier_coefs[layer_idx + 1, 3]
                    #     )
                    #     if not np.all(i_out_matrix[layer_idx] >= 0) or not np.all(lambda_out_matrix[layer_idx] >= 0):
                    #         bezier_debug(
                    #             layer_idx,
                    #             i_out_matrix,
                    #             lambda_out_matrix,
                    #             "out",
                    #         )
            # PASSES COMPLETE.
            # Commit pop_grid_updates to each species.
            max_pop_change_list = []
            damping_enabled_list = []
            sor_enabled_list = []
            for species in nlte_species:
                xsec_data = self[species]
                if is_nlte_xsec(xsec_data):
                    processor = xsec_data.get_nlte_processor()
                    max_pop_change, damping_enabled, sor_enabled = processor.update_pops(
                        pop_grid_updated=pop_grid_updates[species],
                        n_lte_layers=self.n_lte_layers,
                        do_sor=self.sor,
                        sor_enabled=self.sor_enabled,
                        n_iter=self.n_iter
                    )
                    max_pop_change_list.append(max_pop_change)
                    damping_enabled_list.append(damping_enabled)
                    sor_enabled_list.append(sor_enabled)
            self.damping_enabled = np.all(damping_enabled_list)
            self.sor_enabled = np.all(sor_enabled_list)
            if self.damping_enabled:
                self.is_converged = max(max_pop_change_list) < 0.005
            else:
                self.is_converged = max(max_pop_change_list) < 0.001
        log.info(f"[I{self.n_iter}] Convergence achieved!")

        high_res_grid = ...
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

    # def is_converged(self) -> bool:
    #     # is_converged = True
    #     # for species in self:
    #     #     if type(self[species]) is ExomolNLTEXsec:
    #     #         is_converged = is_converged & self[species].is_converged
    #     is_converged = np.all([self[species].is_converged for species in self if type(self[species]) is ExomolNLTEXsec])
    #     return is_converged


def create_r_wn_grid(low: float, high: float, resolving_power: float) -> u.Quantity:
    resolving_f = np.log((resolving_power + 1) / resolving_power)
    n_points = round((np.log(high) - np.log(low)) / resolving_f) + 1
    return np.exp(np.arange(n_points) * resolving_f + np.log(low)) << u.k
