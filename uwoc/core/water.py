"""Seawater inherent optical properties (IOPs) and scattering phase functions.

Water is described either by
  * a Jerlov type (I, IA, IB, II, III open ocean; 1C ... 9C coastal),
  * one of the four classic Petzold-measured waters (pure sea, clear ocean, coastal, turbid harbour),
  * a bio-optical mix (chlorophyll, CDOM, non-algal particles, minerals), or
  * user absorption a and scattering b coefficients at a reference wavelength.

Bio-optical model (all coefficients in 1/m, wavelength in nm):
  a(λ)  = a_w(λ) + 0.06 C^0.65 â_ph(λ) + a_g440 e^{-0.014(λ-440)} + a_nap440 e^{-0.011(λ-440)}
  b(λ)  = b_w(λ) + 0.30 C^0.62 (550/λ) + b_min550 (550/λ)^0.5
  b_w   = 0.00288 (λ/500)^-4.32           (Morel 1974, seawater)
  b_b   = b_w/2 + B_p (b - b_w)             B_p = 0.0183 (Petzold) unless overridden
  K_d  ≈ (a + b_b)/0.85                     (diffuse attenuation, sun near zenith)
with a_w from Pope & Fry (1997). The Jerlov constituents below were chosen so that K_d(λ)
follows Jerlov's spectral curves (minimum in the blue for oceanic types, moving to the green
and yellow for coastal types).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .constants import trapz

# Pope & Fry (1997) pure-water absorption, 1/m
_AW_LAM = np.arange(380, 710, 10.0)
_AW = np.array([0.01137, 0.00851, 0.00663, 0.00473, 0.00454, 0.00495, 0.00635, 0.00922, 0.00979,
                0.0106, 0.0127, 0.0150, 0.0204, 0.0325, 0.0409, 0.0434, 0.0474, 0.0565, 0.0619,
                0.0695, 0.0896, 0.1351, 0.2224, 0.2644, 0.2755, 0.2916, 0.3108, 0.340, 0.410,
                0.439, 0.465, 0.516, 0.624])
# normalised chlorophyll-specific phytoplankton absorption shape (= 1 at 440 nm)
_APH_LAM = np.array([380, 400, 420, 440, 460, 480, 500, 520, 550, 580, 600, 620, 650, 675, 700, 710.0])
_APH = np.array([0.50, 0.687, 0.92, 1.00, 0.90, 0.80, 0.63, 0.44, 0.24, 0.16, 0.17, 0.20, 0.27, 0.66,
                 0.10, 0.05])

B_PETZOLD = 0.0183


def a_water(lam):
    return np.interp(lam, _AW_LAM, _AW)


def b_water(lam):
    return 0.00288 * (np.asarray(lam, float) / 500.0) ** -4.32


def aph_shape(lam):
    return np.interp(lam, _APH_LAM, _APH)


# --------------------------------------------------------------------------- phase functions
def _theta_grid(n=1600):
    """Scattering-angle grid, log-spaced near the forward peak."""
    th = np.concatenate([[0.0], np.geomspace(1e-5, 0.2, n // 2), np.linspace(0.2, np.pi, n // 2)[1:]])
    return th


class PhaseFunction:
    """Tabulated phase function on θ ∈ [0, π]. `pdf_mu(μ)` integrates to 1 over μ = cos θ ∈ [-1, 1];
    per steradian it is pdf_mu / 2π. Sampling inverts the tabulated CDF."""

    def __init__(self, name, beta_fn, n=1600):
        self.name = name
        th = _theta_grid(n)
        beta = np.maximum(beta_fn(th), 0.0)                  # ∝ per-steradian β̃(θ)
        w = 2 * np.pi * beta * np.sin(th)                    # pdf in θ (unnormalised)
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(th))])
        tot = cdf[-1]
        self.theta, self.cdf = th, cdf / tot
        self._beta = beta / tot                               # normalised: ∫ β̃ dΩ = 1
        mu = np.cos(th)
        self.g = float(trapz(mu * w / tot, th))
        k = th >= np.pi / 2
        self.B = float(1 - np.interp(np.pi / 2, th, self.cdf))   # backscatter fraction
        del k, mu

    def beta(self, theta):
        """Normalised volume scattering function per steradian (1/sr)."""
        return np.interp(theta, self.theta, self._beta)

    def pdf_sr(self, mu):
        return self.beta(np.arccos(np.clip(mu, -1, 1)))

    def sample(self, rng, n):
        return np.interp(rng.random(n), self.cdf, self.theta)


def hg_beta(g):
    return lambda th: (1 - g * g) / (4 * np.pi * (1 + g * g - 2 * g * np.cos(th)) ** 1.5)


def rayleigh_water_beta(th):
    return 0.06225 * (1 + 0.835 * np.cos(th) ** 2)


def ff_beta(B=B_PETZOLD, n=1.10):
    """Fournier–Forand phase function with refractive index n; the slope μ is solved so that the
    backscatter fraction equals B (Mobley et al. 2002)."""
    from scipy.optimize import brentq

    def bfrac(mu):
        nu = (3 - mu) / 2
        d90 = 4 / (3 * (n - 1) ** 2) * np.sin(np.pi / 4) ** 2
        return 1 - (1 - d90 ** (nu + 1) - 0.5 * (1 - d90 ** nu)) / ((1 - d90) * d90 ** nu)

    mu = brentq(lambda m: bfrac(m) - B, 3.0001, 4.9999)
    nu = (3 - mu) / 2
    d180 = 4 / (3 * (n - 1) ** 2)

    def beta(th):
        th = np.maximum(th, 1e-7)
        s2 = np.sin(th / 2) ** 2
        d = d180 * s2
        dn = d ** nu
        t1 = (nu * (1 - d) - (1 - dn) + (d * (1 - dn) - nu * (1 - d)) / s2) / (4 * np.pi * (1 - d) ** 2 * dn)
        t2 = (1 - d180 ** nu) / (16 * np.pi * (d180 - 1) * d180 ** nu) * (3 * np.cos(th) ** 2 - 1)
        return t1 + t2
    return beta


def tthg_beta(B=B_PETZOLD, g1=0.9809, g2=0.6):
    """Two-term Henyey–Greenstein, forward lobe g1 and backward lobe −g2, mixed to match B."""
    from scipy.integrate import quad
    def back(g):
        f = hg_beta(g)
        return quad(lambda t: 2 * np.pi * f(t) * np.sin(t), np.pi / 2, np.pi)[0]
    bf, bb = back(g1), back(-g2)
    alpha = float(np.clip((bb - B) / (bb - bf), 0, 1))
    f1, f2 = hg_beta(g1), hg_beta(-g2)
    return lambda th: alpha * f1(th) + (1 - alpha) * f2(th)


PHASE_FUNCTIONS = ("Fournier-Forand", "Henyey-Greenstein", "Two-term HG")
_PF_CACHE = {}


def phase_function(kind="Fournier-Forand", B=B_PETZOLD, g=0.924, water_frac=0.0):
    """Particle phase function mixed with molecular (water) scattering in proportion water_frac."""
    key = (kind, round(B, 6), round(g, 6), round(water_frac, 4))
    if key in _PF_CACHE:
        return _PF_CACHE[key]
    if kind == "Henyey-Greenstein":
        part = hg_beta(g)
    elif kind == "Two-term HG":
        part = tthg_beta(B)
    else:
        part = ff_beta(B)
    wf = float(np.clip(water_frac, 0, 1))
    pf = PhaseFunction(kind, lambda th: (1 - wf) * part(th) + wf * rayleigh_water_beta(th))
    if len(_PF_CACHE) > 64:
        _PF_CACHE.clear()
    _PF_CACHE[key] = pf
    return pf


# --------------------------------------------------------------------------- water bodies
@dataclass
class Water:
    """A water body. kind: 'bio' (Jerlov types and user mixes), 'petzold' or 'custom'."""
    name: str = "Jerlov II"
    kind: str = "bio"
    chl: float = 0.3                 # mg/m^3
    ag440: float = 0.03              # CDOM absorption at 440 nm, 1/m
    anap440: float = 0.0             # non-algal particle absorption at 440 nm, 1/m
    bmin550: float = 0.0             # mineral scattering at 550 nm, 1/m
    a_ref: float = 0.069             # petzold / custom: absorption at lam_ref
    b_ref: float = 0.08              # petzold / custom: scattering at lam_ref
    lam_ref: float = 520.0
    phase: str = "Fournier-Forand"
    bb_ratio: float = B_PETZOLD      # particle backscatter fraction
    g: float = 0.924                 # HG asymmetry
    description: str = ""

    def copy(self, **kw):
        return replace(self, **kw)

    # --- IOPs -------------------------------------------------------------------------------
    def a(self, lam):
        lam = np.asarray(lam, float)
        if self.kind == "bio":
            return (a_water(lam) + 0.06 * self.chl ** 0.65 * aph_shape(lam)
                    + self.ag440 * np.exp(-0.014 * (lam - 440)) + self.anap440 * np.exp(-0.011 * (lam - 440)))
        extra = max(self.a_ref - float(a_water(self.lam_ref)), 0.0)
        base = a_water(lam) if self.a_ref >= float(a_water(self.lam_ref)) else \
            a_water(lam) * self.a_ref / float(a_water(self.lam_ref))
        return base + extra * np.exp(-0.006 * (lam - self.lam_ref))

    def b(self, lam):
        lam = np.asarray(lam, float)
        if self.kind == "bio":
            return b_water(lam) + 0.30 * self.chl ** 0.62 * (550 / lam) + self.bmin550 * (550 / lam) ** 0.5
        return self.b_ref * (self.lam_ref / lam)

    def b_water_part(self, lam):
        return np.minimum(b_water(lam), self.b(lam))

    def c(self, lam):
        return self.a(lam) + self.b(lam)

    def bb(self, lam):
        bw = self.b_water_part(lam)
        return 0.5 * bw + self.bb_ratio * (self.b(lam) - bw)

    def Kd(self, lam):
        return (self.a(lam) + self.bb(lam)) / 0.85

    def albedo(self, lam):
        return self.b(lam) / self.c(lam)

    def best_wavelength(self, lo=400, hi=650):
        lam = np.arange(lo, hi + 1, 1.0)
        return float(lam[np.argmin(self.c(lam))])

    def phase_function(self, lam):
        wf = float(self.b_water_part(lam) / max(self.b(lam), 1e-12))
        return phase_function(self.phase, self.bb_ratio, self.g, wf)

    def iops(self, lam):
        return dict(a=float(self.a(lam)), b=float(self.b(lam)), c=float(self.c(lam)), bb=float(self.bb(lam)),
                    Kd=float(self.Kd(lam)), albedo=float(self.albedo(lam)))


def _jerlov(name, chl, ag, nap=0.0, bmin=0.0, desc=""):
    return Water(name=name, kind="bio", chl=chl, ag440=ag, anap440=nap, bmin550=bmin, description=desc)


def _petzold(name, a, b, desc):
    return Water(name=name, kind="petzold", a_ref=a, b_ref=b, lam_ref=514.0, description=desc)


WATER_TYPES = {
    "Jerlov I": _jerlov("Jerlov I", 0.01, 0.002, desc="clearest open ocean (Sargasso, subtropical gyres)"),
    "Jerlov IA": _jerlov("Jerlov IA", 0.05, 0.005, desc="very clear oceanic"),
    "Jerlov IB": _jerlov("Jerlov IB", 0.1, 0.01, desc="clear oceanic"),
    "Jerlov II": _jerlov("Jerlov II", 0.3, 0.03, desc="oceanic, moderate productivity"),
    "Jerlov III": _jerlov("Jerlov III", 1.0, 0.06, desc="productive oceanic / upwelling"),
    "Jerlov 1C": _jerlov("Jerlov 1C", 1.5, 0.15, 0.02, 0.2, "clearest coastal"),
    "Jerlov 3C": _jerlov("Jerlov 3C", 3.0, 0.3, 0.08, 0.6, "coastal"),
    "Jerlov 5C": _jerlov("Jerlov 5C", 5.0, 0.6, 0.2, 1.2, "turbid coastal"),
    "Jerlov 7C": _jerlov("Jerlov 7C", 8.0, 1.0, 0.45, 2.2, "very turbid coastal"),
    "Jerlov 9C": _jerlov("Jerlov 9C", 12.0, 2.0, 1.0, 4.0, "most turbid coastal / estuarine"),
    "Pure sea water": _petzold("Pure sea water", 0.0405, 0.0025, "Petzold-style: pure seawater"),
    "Clear ocean": _petzold("Clear ocean", 0.114, 0.037, "Petzold clear ocean (Bahamas)"),
    "Coastal ocean": _petzold("Coastal ocean", 0.179, 0.219, "Petzold coastal (San Pedro channel)"),
    "Turbid harbour": _petzold("Turbid harbour", 0.366, 1.824, "Petzold turbid harbour (San Diego)"),
}


def custom_water(a, b, lam_ref=520.0, phase="Fournier-Forand", bb_ratio=B_PETZOLD, g=0.924):
    return Water(name=f"custom a={a:g}, b={b:g}", kind="custom", a_ref=a, b_ref=b, lam_ref=lam_ref,
                 phase=phase, bb_ratio=bb_ratio, g=g, description="user absorption and scattering")


def water_from_name(name):
    if name in WATER_TYPES:
        return WATER_TYPES[name].copy()
    hits = [k for k in WATER_TYPES if k.lower() == name.lower() or k.lower().endswith(" " + name.lower())]
    if len(hits) == 1:
        return WATER_TYPES[hits[0]].copy()
    raise KeyError(f"unknown water type {name!r}; choose from {', '.join(WATER_TYPES)}")
