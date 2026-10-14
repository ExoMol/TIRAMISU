"""Physical constants and units for the whole package, all derived from ``astropy`` (constants: CODATA 2018 in
astropy 7) so that every module uses the same values. ``ac_*`` are Quantities; ``const_*`` are their float values
(cgs unless the name says otherwise) for numba kernels, which cannot take Quantities; ``*_unit`` are the units
attached to arrays computed without them. Add new constants here, derived from astropy, rather than as literals in
the modules that use them."""
import numpy as np
from astropy import constants as ac, units as u

# Units.
einstein_a_unit = 1 / u.s
# B = A c^2 / (2 h nu^3) is per unit mean intensity J [J / (m^2 sr)] (per Hz), so B J is a rate: the steradian is part
# of B, not cancelled by multiplying J by 4 pi sr (B 4 pi J would be 4 pi too large).
einstein_b_unit = (u.m ** 2 * u.sr) / (u.J * u.s)
source_func_unit = u.J / (u.sr * u.m ** 2)

# Quantities.
ac_h_c_on_kB = ac.h * ac.c.cgs / ac.k_B  # Second radiation constant [cm K].
ac_h_on_kB = ac.h / ac.k_B
ac_2_h_on_c_sq = 2 * ac.h / ac.c ** 2
ac_h_on_8_c = ac.h.cgs / (8 * ac.c.cgs)
ac_4_pi_c = 4 * np.pi * ac.c.cgs
ac_8_pi_c = 2 * ac_4_pi_c
ac_h_c_on_4_pi = ac.h.cgs * ac.c.cgs / (4 * np.pi)
ac_sqrt_NA_kB_on_c = (np.sqrt(ac.N_A * ac.k_B.cgs) / ac.c.cgs).to(
    u.g ** 0.5 / (u.K ** 0.5 * u.mol ** 0.5), equivalencies=u.spectral()
)
ac_sqrt_2_NA_kB_log2_on_c = ac_sqrt_NA_kB_on_c * np.sqrt(2 * np.log(2))
ac_c_sq_on_2h = (ac.c ** 2) / (2 * ac.h)

# Float values, for numba.
const_amu = ac.u.cgs.value  # [g]
const_k_B = ac.k_B.cgs.value  # [erg / K]
const_atm = ac.atm.cgs.value  # 1 standard atmosphere [dyn / cm^2].
const_c_cgs = ac.c.cgs.value
const_h_c_on_kB = ac_h_c_on_kB.value  # [cm K]
const_h_on_8_c = ac_h_on_8_c.value
const_h_on_8_c_amu = const_h_on_8_c / const_amu
const_4_pi_c = ac_4_pi_c.value
const_8_pi_c = ac_8_pi_c.value
const_h_c_on_4_pi = ac_h_c_on_4_pi.value
const_sqrt_NA_kB_on_c = ac_sqrt_NA_kB_on_c.value
const_sqrt_2_NA_kB_log2_on_c = ac_sqrt_2_NA_kB_log2_on_c.value
const_c_sq_on_2h = ac_c_sq_on_2h.value
const_2_pi_h_c_sq_on_sigma_sba = (
    (2 * np.pi * ac.h * ac.c.cgs ** 2 / ac.sigma_sb).to_value(u.K ** 4 * u.cm ** 4, equivalencies=u.spectral())
)
