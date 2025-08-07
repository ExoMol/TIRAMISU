import os
import pathlib
import typing as t

import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from itertools import islice, cycle

from tiramisu import xsec
from tiramisu.chemistry import ChemicalProfile, SpeciesFormula
from tiramisu.eclipse import ExoplanetEmission
from tiramisu.config import log
from tiramisu.nlte import incident_stellar_radiation, cdf_opacity_sampling
from tiramisu.xsec import ExomolNLTEXsec, create_r_wn_grid
from pathlib import Path

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)


# def plot_emission(
#     plot_grid: u.Quantity,
#     plot_flux: u.Quantity,
#     plot_tau: u.Quantity,
#     plot_xsecs: t.Dict[SpeciesFormula, u.Quantity],
#     label_fontsize: int = 22,
#     tick_fontsize: int = 18,
#     save_file: str | Path | None = None,
# ) -> None:
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
#     x_axis_left = plot_grid[0].value
#     x_axis_right = plot_grid[-1].value
#     # ax1.plot(plot_grid, plot_flux.to(u.W / u.m ** 2 / u.um))
#     ax1.plot(plot_grid, plot_flux.to(u.J / u.m**2, equivalencies=u.spectral()))
#     ax1.set_xlabel(r"Spectral Grid (cm$^\mathbf{-1}$)", fontsize=label_fontsize)
#     ax1.set_xlim(left=x_axis_left, right=x_axis_right)
#     ax1.set_ylim(bottom=1e-11)
#     ax1.set_ylabel(
#         f"Flux ({plot_flux.to(u.J / u.m ** 2, equivalencies=u.spectral()).unit:latex})", fontsize=label_fontsize
#     )
#     ax1.set_yscale("log")
#     ax1.set_title("Emission Spectrum", fontsize=label_fontsize)
#     ax1.tick_params(labelsize=tick_fontsize)
#     ax2.imshow(
#         plot_tau,
#         aspect="auto",
#         origin="lower",
#         # extent=[x_axis_left, x_axis_right, 0, plot_tau.shape[0]],
#         interpolation="none",
#         cmap="plasma",
#     )
#     ax2.set_title("Optical Depth", fontsize=label_fontsize)
#     ax2.set_xlabel(r"Spectral Grid (cm$^\mathbf{-1}$)", fontsize=label_fontsize)
#     ax2.set_ylabel("Altitude (km)", fontsize=label_fontsize)
#     # X ticks wrong if using linear grid.
#     if not np.all(np.isclose(np.diff(plot_grid.value), np.abs(plot_grid.value[1] - plot_grid.value[0]), atol=0)):
#         x_tick_locs = np.linspace(0, len(plot_grid) - 1, 9, dtype=int)
#         x_ticks = [int(plot_grid.value[x_tick_loc]) for x_tick_loc in x_tick_locs]
#         ax2.set_xticks(x_tick_locs, x_ticks)
#     ax2.tick_params(labelsize=tick_fontsize)
#
#     # ax2.axis("off")
#     # toa_offset = 5
#     def get_vibrant_colors(n_colors: int, ordered: bool = False) -> t.List[str]:
#         if ordered:
#             vibrant_color_list = [
#                 "#0077BB",
#                 "#33BBEE",
#                 "#44EE66",
#                 "#229933",
#                 "#FFBB00",  # FFCC11
#                 "#EE7733",
#                 "#CC3311",
#                 "#EE3377",
#                 "#BB33BB",
#                 "#8833EE",
#             ]
#         else:
#             vibrant_color_list = [
#                 "#EE7733",
#                 "#0077BB",
#                 "#CC3311",
#                 "#33BBEE",
#                 "#229933",
#                 "#EE3377",
#                 "#44EE66",
#                 "#BB33BB",
#                 "#FFBB00",  # FFCC11
#                 "#8833EE",
#             ]
#         if n_colors > len(vibrant_color_list):
#             return list(islice(cycle(vibrant_color_list), n_colors))
#         else:
#             colors_list_idx = np.linspace(0, len(vibrant_color_list) - 1, n_colors, dtype=int).tolist()
#             return list([vibrant_color_list[idx] for idx in colors_list_idx])
#
#     plt.tight_layout()
#     if save_file is not None:
#         plt.savefig(save_file, bbox_inches="tight", dpi=600)
#     # plt.show()
#     plt.close()
#
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
#     plot_colours = get_vibrant_colors(n_colors=len(plot_xsecs))
#     for species_idx, xsec_key in enumerate(plot_xsecs.keys()):
#         if xsec_key == "HO":
#             for layer_idx in np.concatenate((np.arange(0, plot_tau.shape[0], 10), [plot_tau.shape[0] - 1])):
#                 ax1.plot(
#                     plot_grid,
#                     plot_xsecs.get(xsec_key)[layer_idx] / oh_vmr[layer_idx],
#                     label=f"{xsec_key} [L{layer_idx}]",
#                     color=plot_colours[species_idx],
#                     alpha=0.5 + 0.5 * layer_idx / plot_tau.shape[0],
#                     zorder=layer_idx,
#                 )
#     ax1.set_xlabel(r"Spectral Grid (cm$^\mathbf{-1}$)", fontsize=label_fontsize)
#     ax1.set_xlim(left=x_axis_left, right=x_axis_right)
#     ax1.set_ylabel(r"Cross Section (cm$^\mathbf{2}$ molecule$^\mathbf{-1}$)", fontsize=label_fontsize)
#     ax1.set_yscale("log")
#     ax1.set_ylim(bottom=1e-35, top=1e-16)
#     ax1.set_title(f"Non-LTE Molecular Opacities", fontsize=label_fontsize)
#     ax1.tick_params(labelsize=tick_fontsize)
#     ax1.legend(
#         loc="best",
#         prop={"size": tick_fontsize},
#         ncol=int(np.ceil(len(plot_xsecs) / 2)),
#     )
#     plot_colours = get_vibrant_colors(n_colors=len(plot_xsecs))
#     for idx, xsec_key in enumerate(plot_xsecs.keys()):
#         ax2.plot(
#             plot_grid,
#             plot_xsecs.get(xsec_key)[-1],
#             label=xsec_key,
#             color=plot_colours[idx],
#         )
#     ax2.set_xlabel(r"Spectral Grid (cm$^\mathbf{-1}$)", fontsize=label_fontsize)
#     ax2.set_xlim(left=x_axis_left, right=x_axis_right)
#     ax2.set_ylabel(r"VMR x Cross Section (cm$^\mathbf{2}$ molecule$^\mathbf{-1}$)", fontsize=label_fontsize)
#     ax2.set_yscale("log")
#     ax2.set_ylim(bottom=1e-40, top=1e-20)
#     ax2.set_title(f"Molecular Opacities at TOA", fontsize=label_fontsize)
#     ax2.tick_params(labelsize=tick_fontsize)
#     ax2.legend(
#         loc="best",
#         prop={"size": tick_fontsize},
#         ncol=int(np.ceil(len(plot_xsecs) / 2)),
#     )
#     plt.tight_layout()
#     if save_file is not None:
#         plt.savefig(save_file, bbox_inches="tight", dpi=600)
#     # plt.show()
#     plt.close()


# KELT-20 b profiles
profiles = pd.read_csv(
    r"/mnt/c/PhD/programs/charlesrex/tests/inputs/KELT20_PT_abunds_med.dat",
    header=0,
    names=["logP", "T", "logCO", "logH2O", "logOH", "logFe"],
)
central_pressure = 10 ** profiles["logP"].to_numpy()[::-1] << u.bar
pressure_levels = (
    np.loadtxt(r"/mnt/c/PhD/programs/charlesrex/tests/inputs/KELT20_PT_abunds_maxl_fit_pressure_levels.txt") << u.bar
)
co_mmr = 10 ** profiles["logCO"].to_numpy()[::-1]
h2o_mmr = 10 ** profiles["logH2O"].to_numpy()[::-1]
oh_mmr = 10 ** profiles["logOH"].to_numpy()[::-1]
oh_scale_factor = 1
# oh_mmr = np.repeat(0.3, len(profiles))
oh_mmr *= oh_scale_factor
fe_mmr = 10 ** profiles["logFe"].to_numpy()[::-1]

# h_mmr = 10 ** (-0.1)  # Max-likelihood
h_mmr = 10 ** (-0.2)  # Median
h_mmr = np.repeat(h_mmr, len(profiles))

h_total_mmr = 10 ** (-0.2) # Median
h2_mmr = np.repeat(h_total_mmr, len(profiles))
h2_dissociation_pressure = 10 ** +0.2 << u.bar
h2_dissociation_logic = central_pressure < h2_dissociation_pressure
h2_mmr[h2_dissociation_logic] = h2_mmr[h2_dissociation_logic] * (central_pressure[h2_dissociation_logic] / h2_dissociation_pressure) ** 4
h_mmr = np.repeat(h_total_mmr, len(profiles)) - h2_mmr

he_mmr = np.ones_like(h_mmr) - h2_mmr - h_mmr - co_mmr - h2o_mmr - oh_mmr - fe_mmr

mass_h = 1.00782503223
mass_h2 = 2 * mass_h
mass_he = 4.00260325413
mass_o = 15.99491461957
mass_co = 12 + mass_o
mass_h2o = 2 * mass_h + mass_o
mass_oh = mass_o + mass_h
mass_fe = 55.93493632600

h2_rmmr = h2_mmr / mass_h2
h_rmmr = h_mmr / mass_h
he_rmmr = he_mmr / mass_he
co_rmmr = co_mmr / mass_co
h2o_rmmr = h2o_mmr / mass_h2o
oh_rmmr = oh_mmr / mass_oh
fe_rmmr = fe_mmr / mass_fe

sum_rmmr = sum([h2_rmmr, h_rmmr, he_rmmr, co_rmmr, h2o_rmmr, oh_rmmr, fe_rmmr])
h2_vmr = h2_rmmr / sum_rmmr
h_vmr = h_rmmr / sum_rmmr
he_vmr = he_rmmr / sum_rmmr
co_vmr = co_rmmr / sum_rmmr
h2o_vmr = h2o_rmmr / sum_rmmr
oh_vmr = oh_rmmr / sum_rmmr
fe_vmr = fe_rmmr / sum_rmmr

nlayers = len(profiles)

temperature_profile = profiles["T"].to_numpy()[::-1] << u.K
log.info(temperature_profile[20])

spectral_grid = cdf_opacity_sampling(
    wn_start=10, wn_end=30000, temperature_profile=temperature_profile, num_points=1000, max_step=50
)

# # HD 209458 b
# planet_mass = 0.682 << u.Mjup
# planet_radius = 1.37 << u.Rjup

# WASP-39b
# planet_mass = 0.28 << u.Mjup
# planet_radius = 1.332 << u.Rjup
# star_temperature = 5485 << u.K
# orbital_radius = 0.0486 << u.AU

# KELT-20 b
planet_mass = 3.372 << u.Mjup
planet_radius = 1.83 << u.Rjup
star_temperature = 8730 << u.K
orbital_radius = 0.057 << u.AU
incident_radiation_field = incident_stellar_radiation(
    wn_grid=spectral_grid, star_temperature=star_temperature, orbital_radius=orbital_radius, planet_radius=planet_radius
)

chemistry_profile = ChemicalProfile.from_species_definition(
    species_definition={
        "H": h_vmr,
        "He": he_vmr,
        "CO": co_vmr,
        "H2O": h2o_vmr,
        "OH": oh_vmr,
        "Fe": fe_vmr,
        # # "O": o_vmr,
    },
    # fill_species=["H", "He", "O", "O2", "O3"],
    # fill_ratios=[hydrogen_vmr, 1 - hydrogen_vmr, 1e-7, 1e-5, 1e-6],
    fill_species=[
        # "H",
        # "He",
    ],
    fill_ratios=[
        # hydrogen_vmr,
        # 1.0,  # 1 - hydrogen_vmr,
    ],
    nlayers=nlayers,
)

emission = ExoplanetEmission(
    planet_mass=planet_mass,
    planet_radius=planet_radius,
    temperature_profile=temperature_profile,
    boa_pressure=1e2 << u.bar,
    toa_pressure=1e-6 << u.bar,
    nlayers=nlayers,
    chemistry_profile=chemistry_profile,
    ngauss=4,
    central_pressure=central_pressure,
    pressure_levels=pressure_levels,
)
xsecs = xsec.XSecCollection()

hdf5_xsecs = xsec.ExomolHDF5Xsec.discover_all(
    pathlib.Path(r"/mnt/c/PhD/programs/charlesrex/tests/inputs"),
    load_in_memory=True,
)

for x in hdf5_xsecs:
    xsecs.add_replace_xsec_data(x)

# NLTE Cross sections:
# mass_oh = 15.99491461957 + 1.00782503223  # O + H
broadening_params = xsec.weight_broadening_parameters(
    broadening_dict={"H": (0.089, 0.5), "He": (0.015, 0.5)}, chemistry_profile=chemistry_profile
)
n_lte_layers = 20
nlte_xsec = xsec.ExomolNLTEXsec(
    species="OH",
    species_mass=mass_oh,
    states_file=Path(r"/mnt/c/PhD/OH/ExoMol/16O-1H__MYTHOS.noB4.states"),
    trans_files=Path(r"/mnt/c/PhD/OH/ExoMol/16O-1H__MYTHOS.trunc.trans"),
    agg_col_nums=[9, 10],
    planet_radius=planet_radius,
    chem_profile=chemistry_profile,
    broadening_params=broadening_params,  # (gamma, n) for H, He
    # source_func_threshold=1e-50,
    n_lte_layers=n_lte_layers,
    # lte_grid_file=Path(r"/mnt/c/PhD/OH/Opacities/16O-1H__MYTHOS.R15000_0.125-100000mu.xsec.TauREx.h5"),
    lte_grid_file=Path(r"/mnt/c/PhD/OH/Opacities/16O-1H__MYTHOS_cont.R15000_0.125-100000mu.xsec.TauREx.h5"),
    cont_states_file=Path(r"/mnt/c/PhD/OH/ExoMol/XABC11_Unbound_States_Trans/16O-1H__MYTHOS.states.cont"),
    cont_trans_files=Path(r"/mnt/c/PhD/OH/ExoMol/XABC11_Unbound_States_Trans/16O-1H__MYTHOS.trans.cont"),
    dissociation_products=("O", "H"),
    incident_radiation_field=incident_radiation_field,
    debug=True,
    sor=True,
)
# mass_tio = 47.94794197900 + 15.99491461957  # Ti + O
# # broadening_params = xsec.weight_broadening_parameters(broadening_dict={"H": (0.089, 0.5), "He": (0.015, 0.5)},
# #                                                       chemistry_profile=chemistry_profile)
# nlte_xsec = xsec.ExomolNLTEXsec(
#     species="TiO",
#     species_mass=mass_tio,
#     states_file=Path(r"/mnt/c/PhD/TiO/ExoMol/48Ti-16O__Toto.states"),
#     trans_files=Path(r"/mnt/c/PhD/TiO/ExoMol/48Ti-16O__Toto.trans"),
#     agg_col_nums=[9, 10],
#     chem_profile=chemistry_profile,
#     # broadening_params=broadening_params,
#     n_lte_layers=95,
#     lte_grid_file=Path(r"/mnt/c/PhD/TiO/Opacities/48Ti-16O__Toto.R15000_0.125-100000mu.xsec.TauREx.h5"),
#     incident_radiation_field=incident_radiation_field,
# )
xsecs.add_replace_xsec_data(nlte_xsec)

start = time.perf_counter()

# spectral_grid = create_r_wn_grid(low=spectral_grid[0].value, high=spectral_grid[-1].value, resolving_power=15000)
spectral_grid, emission_flux, emission_tau, emission_xsecs = emission.compute_emission(
    xsecs, spectral_grid=spectral_grid, output_intensity=True
)
# plot_emission(plot_grid=spectral_grid, plot_flux=emission_flux, plot_tau=emission_tau, plot_xsecs=emission_xsecs)
# np.savetxt(
#     (output_dir / f"KELT-20b_LTE_cont_{oh_scale_factor}xOH_boundaryL{n_lte_layers}.txt").resolve(),
#     np.array([spectral_grid.value, emission_flux.to(u.J / u.m**2, equivalencies=u.spectral()).value]).T,
#     fmt="%17.8E",
# )

tau_old = []
tau_residual = None
is_converged = False
iter_num = 1
while not is_converged:
    tau_old = emission_tau

    spectral_grid, emission_flux, emission_tau, emission_xsecs = emission.compute_emission(
        xsecs, spectral_grid=spectral_grid, output_intensity=False
    )

    tau_residual = abs(emission_tau - tau_old).sum()
    log.info(f"Tau residual after iteration, combined across all layers = {tau_residual}")
    convergence_list = []
    for species in xsecs:
        if type(xsecs[species]) is ExomolNLTEXsec:
            convergence_list.append(xsecs[species].is_converged)
    if all(convergence_list):
        is_converged = True
    iter_num += 1

# Result gives the spectral grid requested, the emission flux, the optical depth and the cross-sections used.
high_res_grid = create_r_wn_grid(low=spectral_grid[0].value, high=spectral_grid[-1].value, resolving_power=15000)
spectral_grid, emission_flux, emission_tau, emission_xsecs = emission.compute_emission(
    xsecs, spectral_grid=high_res_grid, output_intensity=True
)

log.info(f"Time taken: {time.perf_counter() - start:.2f} s")
