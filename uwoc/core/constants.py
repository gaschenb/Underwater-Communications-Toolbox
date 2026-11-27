"""Physical constants (SI)."""
import numpy as np
H = 6.62607015e-34          # Planck, J s
C0 = 299_792_458.0          # vacuum light speed, m/s
Q = 1.602176634e-19         # electron charge, C
KB = 1.380649e-23           # Boltzmann, J/K
N_WATER = 1.34              # refractive index of seawater (visible)
V_WATER = C0 / N_WATER      # group velocity used for delays, m/s


def photon_energy(lam_nm):
    return H * C0 / (lam_nm * 1e-9)


trapz = getattr(np, "trapezoid", None) or np.trapz      # numpy 2 renamed trapz
