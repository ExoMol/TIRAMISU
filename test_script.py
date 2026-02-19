import pathlib

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd

from tiramisu import xsec
from tiramisu.chemistry import ChemicalProfile
from tiramisu.eclipse import ExoplanetEmission
from tiramisu.config import log, output_dir
from tiramisu.nlte import incident_stellar_radiation, cdf_opacity_sampling
from tiramisu.xsec import create_r_wn_grid
from pathlib import Path


if __name__ == "__main__":
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

    spectral_grid = cdf_opacity_sampling(
        wn_start=100, wn_end=30000, temperature_profile=temperature_profile, num_points=10000, max_step=50
    )
    # spectral_grid = np.linspace(100, 30000, 2991) << 1 / u.cm

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
    n_lte_layers = 40
    xsecs = xsec.XSecCollection(
        n_lte_layers=n_lte_layers,
        incident_radiation_field=incident_radiation_field,
        debug=True,
        sor=True,
    )

    hdf5_xsecs = xsec.ExomolHDF5Xsec.discover_all(
        pathlib.Path(r"/mnt/c/PhD/programs/TIRAMISU/tests/inputs"),
        load_in_memory=True,
    )

    for x in hdf5_xsecs:
        if x.species in chemistry_profile.species:
            xsecs.add_replace_xsec_data(x)

    # NLTE Cross-sections:
    # mass_oh = 15.99491461957 + 1.00782503223  # O + H
    broadening_params = xsec.weight_broadening_parameters(
        broadening_dict={"H": (0.089, 0.5), "He": (0.015, 0.5)}, chemistry_profile=chemistry_profile
    )
    nlte_xsec = xsec.ExomolNLTEXsec(
        species="OH",
        states_file=Path(r"/mnt/c/PhD/OH/ExoMol/16O-1H__MYTHOS.noB4.states"),
        trans_files=Path(r"/mnt/c/PhD/OH/ExoMol/16O-1H__MYTHOS.trunc.trans"),
        agg_col_nums=[9, 10],
        broadening_params=broadening_params,  # (gamma, n) for H, He
        n_lte_layers=n_lte_layers,
        lte_grid_file=Path(r"/mnt/c/PhD/OH/Opacities/16O-1H__MYTHOS_cont.R15000_0.125-100000mu.xsec.TauREx.h5"),
        cont_states_file=Path(r"/mnt/c/PhD/OH/ExoMol/XABC11_Unbound_States_Trans/16O-1H__MYTHOS.states.cont"),
        cont_trans_files=Path(r"/mnt/c/PhD/OH/ExoMol/XABC11_Unbound_States_Trans/16O-1H__MYTHOS.trans.cont"),
        cont_box_length=6.5E-8,
        cont_broad_col_num=10,
        dissociation_products=("O", "H"),
        debug=True,
    )
    xsecs.add_replace_xsec_data(nlte_xsec)

    start = time.perf_counter()

    # spectral_grid = create_r_wn_grid(low=spectral_grid[0].value, high=spectral_grid[-1].value, resolving_power=15000)
    # incident_radiation_field = incident_stellar_radiation(
    #     wn_grid=spectral_grid, star_temperature=star_temperature, orbital_radius=orbital_radius, planet_radius=planet_radius
    # )
    # This now contains the loop!
    spectral_grid, i_up, i_down, dtau, opacities = emission.compute_emission(
        xsecs,
        spectral_grid=spectral_grid,
        output_intensity=True,
        incident_radiation_field=incident_radiation_field,
        approximate_t_ex=True,
    )
    import matplotlib.colors as colors

    plt.figure(figsize=(8, 4), dpi=300)
    plt.imshow(
        xsecs.global_chi_matrix.value, interpolation=None, origin="lower", aspect=62.5,
        norm=colors.LogNorm(
            vmin=xsecs.global_chi_matrix.value[xsecs.global_chi_matrix.value > 0].min(),
            vmax=xsecs.global_chi_matrix.value.max()
        ),
    )
    plt.colorbar(label=f"Chi ({xsecs.global_chi_matrix.unit:latex})")
    plt.show()

    plt.figure(figsize=(8, 4), dpi=300)
    plt.imshow(
        xsecs.global_eta_matrix.value, interpolation=None, origin="lower", aspect=62.5,
        norm=colors.LogNorm(
            vmin=xsecs.global_eta_matrix.value[xsecs.global_eta_matrix.value > 0].min(),
            vmax=xsecs.global_eta_matrix.value.max()
        ),
    )
    plt.colorbar(label=f"Eta ({xsecs.global_eta_matrix.unit:latex})")
    plt.show()

    plt.figure(figsize=(8, 4), dpi=300)
    plt.imshow(
        dtau, interpolation=None, origin="lower", aspect=62.5,
        norm=colors.LogNorm(vmin=dtau[dtau > 0].min(), vmax=1),
    )
    plt.colorbar(label="Tau")
    plt.show()

    np.save(
        fr"/mnt/c/PhD/NLTE/Models/KELT-20b/approximation/ohx1e{int(np.log10(oh_scale_factor))}_OH_vmr.npy",
        oh_vmr
    )
    np.save(
        fr"/mnt/c/PhD/NLTE/Models/KELT-20b/approximation/ohx1e{int(np.log10(oh_scale_factor))}_chi.npy",
        xsecs.global_chi_matrix.value
    )
    np.save(
        fr"/mnt/c/PhD/NLTE/Models/KELT-20b/approximation/ohx1e{int(np.log10(oh_scale_factor))}_eta.npy",
        xsecs.global_eta_matrix.value
    )
    np.save(
        fr"/mnt/c/PhD/NLTE/Models/KELT-20b/approximation/ohx1e{int(np.log10(oh_scale_factor))}_dtau.npy",
        dtau
    )
    np.save(
        fr"/mnt/c/PhD/NLTE/Models/KELT-20b/approximation/ohx1e{int(np.log10(oh_scale_factor))}_wn_grid.npy",
        spectral_grid.value
    )
    np.save(
        fr"/mnt/c/PhD/NLTE/Models/KELT-20b/approximation/ohx1e{int(np.log10(oh_scale_factor))}_nd.npy",
        emission.density.to(1 / u.m**3).value
    )
    np.save(
        fr"/mnt/c/PhD/NLTE/Models/KELT-20b/approximation/ohx1e{int(np.log10(oh_scale_factor))}_dz.npy",
        emission.dz.to(u.m).value
    )
    print(incident_radiation_field.unit)
    np.save(
        fr"/mnt/c/PhD/NLTE/Models/KELT-20b/approximation/ohx1e{int(np.log10(oh_scale_factor))}_isrf.npy",
        incident_radiation_field.value
    )
    exit()

    # TODO: Update below to ensure re-computing output on high-res grid works fine.
    # Result gives the spectral grid requested, the emission flux, the optical depth and the cross-sections used.
    high_res_grid = create_r_wn_grid(low=spectral_grid[0].value, high=spectral_grid[-1].value, resolving_power=15000)
    incident_radiation_field = incident_stellar_radiation(
        wn_grid=high_res_grid, star_temperature=star_temperature, orbital_radius=orbital_radius, planet_radius=planet_radius
    )
    spectral_grid, emission_flux, emission_tau, emission_xsecs = emission.compute_emission(
        xsecs, spectral_grid=high_res_grid, output_intensity=True, incident_radiation_field=incident_radiation_field
    )
    np.savetxt(
        (output_dir / f"nLTE_spectral_grid.txt").resolve(),
        high_res_grid.value,
        fmt="%17.8E",
    )

    log.info(f"Time taken: {time.perf_counter() - start:.2f} s")

    # TODO: Update to new workflow.
    #  Check difference between setting number of LTE layers or just having the bottom boundary layer be higher? Should they
    #  not be the same?
