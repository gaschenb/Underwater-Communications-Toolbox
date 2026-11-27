"""Optical channel: geometry, attenuation and scattering models, impulse response, turbulence,
pointing error and ambient light.

Geometric capture of a Gaussian beam (1/e² radius w(L) = sqrt(w0² + (L tan θ)²)) by a disk of
radius R displaced by r from the beam axis (Farid & Hranilovic 2007):
    η_geo(r) ≈ A0 exp(-2 r² / w_eq²),   A0 = erf(v)²,   v = sqrt(π/2) R / w,
    w_eq² = w² sqrt(π) erf(v) / (2 v e^{-v²})
Attenuation models
    'beer'      e^{-c L}: only unscattered light (narrow beam, narrow FOV)
    'effective' e^{-c L} + captured scattered light: e^{-(a + (1 - f) b) L} - e^{-c L} weighted by
                the fraction of the small-angle scattered halo inside the aperture and FOV, with
                f = fraction of the phase function inside the forward lobe
    'mc'        Monte Carlo photon transport (montecarlo.py)
Small-angle multiple scattering: scattering into a forward lobe θ < θc (fraction P of b, rms
single-scattering angle θ₁) broadens the beam, <θ²>(z) = P b z θ₁², and lengthens the path by
P b θ₁² L²/4 on average, so the mean excess delay is τ̄ = P b θ₁² L² / (4 v); scattering outside
the lobe is counted as loss. θc = 10° for the captured power and 20° for the delay reproduce the
Monte Carlo results within about a factor 2 from clear ocean to turbid harbour.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.special import erf

from .constants import V_WATER, trapz
from .water import Water


# =========================================================================== geometry
def beam_radius(L, divergence_rad, waist_m):
    return np.sqrt(waist_m ** 2 + (np.asarray(L, float) * np.tan(divergence_rad)) ** 2)


def gaussian_capture(L, divergence_rad, waist_m, aperture_radius, offset=0.0):
    w = beam_radius(L, divergence_rad, waist_m)
    v = np.sqrt(np.pi / 2) * aperture_radius / w
    A0 = erf(v) ** 2
    weq2 = w ** 2 * np.sqrt(np.pi) * erf(v) / (2 * v * np.exp(-v ** 2))
    return A0 * np.exp(-2 * np.asarray(offset, float) ** 2 / weq2), np.sqrt(weq2)


# =========================================================================== turbulence
def rytov_variance(Cn2, lam_nm, L, wave="spherical"):
    """σ_R² = 1.23 Cn² k^{7/6} L^{11/6} (plane wave); 0.5 × that for a spherical wave."""
    k = 2 * np.pi / (lam_nm * 1e-9)
    s = 1.23 * Cn2 * k ** (7 / 6) * np.asarray(L, float) ** (11 / 6)
    return s * (0.4 if wave == "spherical" else 1.0)


def scintillation_index(sigR2):
    """Andrews–Phillips strong-fluctuation plane-wave scintillation index."""
    s = np.asarray(sigR2, float)
    s125 = s ** (6 / 5)
    return np.exp(0.49 * s / (1 + 1.11 * s125) ** (7 / 6) + 0.51 * s / (1 + 0.69 * s125) ** (5 / 6)) - 1


def gamma_gamma_params(sigR2):
    s = np.asarray(sigR2, float)
    s125 = s ** (6 / 5)
    alpha = 1 / (np.exp(0.49 * s / (1 + 1.11 * s125) ** (7 / 6)) - 1)
    beta = 1 / (np.exp(0.51 * s / (1 + 0.69 * s125) ** (5 / 6)) - 1)
    return alpha, beta


def aperture_averaging(lam_nm, L, D):
    """Plane-wave aperture-averaging factor A = [1 + 1.062 k D² / (4L)]^{-7/6}."""
    k = 2 * np.pi / (lam_nm * 1e-9)
    return (1 + 1.062 * k * D ** 2 / (4 * np.asarray(L, float))) ** (-7 / 6)


@dataclass
class Turbulence:
    """Intensity fading from refractive-index fluctuations (temperature / salinity microstructure,
    bubbles). model: 'none', 'lognormal', 'gamma-gamma', 'exponential' (saturated).
    Strength is either a scintillation index σ_I² ('si') or Cn² (1/m^{2/3}, 'cn2')."""
    model: str = "none"
    strength_mode: str = "si"
    si: float = 0.1
    cn2: float = 1e-13
    coherence_s: float = 3e-3           # temporal correlation of the fading
    aperture_avg: bool = True

    def sigma_I2(self, lam_nm, L, D):
        if self.model == "none":
            return 0.0
        if self.strength_mode == "cn2":
            s = float(scintillation_index(rytov_variance(self.cn2, lam_nm, L)))
        else:
            s = self.si
        if self.aperture_avg:
            s *= float(aperture_averaging(lam_nm, L, D))
        return max(s, 1e-8)

    def pdf(self, h, si):
        """Fading pdf with unit mean."""
        h = np.asarray(h, float)
        if self.model == "lognormal":
            s2 = np.log1p(si)
            return np.exp(-(np.log(np.maximum(h, 1e-300)) + s2 / 2) ** 2 / (2 * s2)) / (h * np.sqrt(2 * np.pi * s2))
        if self.model == "gamma-gamma":
            from scipy.special import gammaln, kve
            al, be = _gg_from_si(si)
            x = 2 * np.sqrt(al * be * np.maximum(h, 1e-300))
            lg = (np.log(2) + (al + be) / 2 * np.log(al * be) - gammaln(al) - gammaln(be)
                  + ((al + be) / 2 - 1) * np.log(np.maximum(h, 1e-300)) + np.log(np.maximum(kve(al - be, x), 1e-300)) - x)
            return np.exp(lg)
        if self.model == "exponential":
            return np.exp(-h)
        return np.zeros_like(h)

    def _process(self, n, dt, rng):
        return _ar1(n, dt, self.coherence_s, rng)

    def sample(self, n, dt, si, rng):
        """Correlated unit-mean fading process (Gaussian copula, AR(1) correlation)."""
        if self.model == "none" or n == 0:
            return np.ones(n)
        u = _norm_cdf(self._process(n, dt, rng))
        if self.model == "gamma-gamma":
            from scipy.special import gammaincinv
            al, be = _gg_from_si(si)
            v = np.clip(_norm_cdf(self._process(n, dt, rng)), 1e-12, 1 - 1e-12)
            return gammaincinv(al, np.clip(u, 1e-12, 1 - 1e-12)) / al * gammaincinv(be, v) / be
        return self.quantile(u, si)

    def draw(self, n, si, rng=None):
        """Stratified fading samples (equal-probability quantiles) for averaging the BER over the
        fading pdf; gamma-gamma uses a product grid of its two gamma factors."""
        if self.model == "none":
            return np.ones(n)
        if self.model == "gamma-gamma":
            from scipy.special import gammaincinv
            al, be = _gg_from_si(si)
            m = max(int(np.sqrt(n)), 2)
            u = (np.arange(m) + 0.5) / m
            x, y = gammaincinv(al, u) / al, gammaincinv(be, u) / be
            return (x[:, None] * y[None, :]).ravel()
        return self.quantile((np.arange(n) + 0.5) / n, si)

    def quantile(self, u, si):
        u = np.clip(u, 1e-12, 1 - 1e-12)
        if self.model == "lognormal":
            from scipy.special import ndtri
            s2 = np.log1p(si)
            return np.exp(-s2 / 2 + np.sqrt(s2) * ndtri(u))
        if self.model == "exponential":
            return -np.log1p(-u)
        return np.ones_like(u)

    def outage(self, margin_db, si):
        """P(h < 10^{-margin/10})."""
        thr = 10 ** (-np.asarray(margin_db, float) / 10)
        if self.model == "none":
            return np.where(thr > 1, 1.0, 0.0)
        hh = np.geomspace(1e-6, 1, 4000)
        p = self.pdf(hh, si)
        cdf = np.concatenate([[0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(hh))])
        return np.interp(thr, hh, cdf)


def _ar1(n, dt, tau, rng):
    """Unit-variance Gaussian AR(1) process sampled every dt with correlation time tau. Long
    records are generated on a tau/8 grid and interpolated; otherwise sampled exactly."""
    if n == 0:
        return np.zeros(0)
    tau = max(tau, 1e-12)
    m = int(np.ceil(n * dt / tau * 8)) + 2
    exact = m >= n
    m = n if exact else m
    tc = np.arange(n) * dt if exact else np.linspace(0, n * dt, m)
    rc = float(np.exp(-(tc[1] - tc[0]) / tau)) if m > 1 else 0.0
    e = rng.standard_normal(m)
    if rc < 1e-12:
        g = e
    else:
        from scipy.signal import lfilter
        g = np.empty(m); g[0] = e[0]
        g[1:] = lfilter([np.sqrt(1 - rc * rc)], [1, -rc], e[1:], zi=[rc * e[0]])[0]
    return g if exact else np.interp(np.arange(n) * dt, tc, g)


def _norm_cdf(x):
    from scipy.special import ndtr
    return ndtr(x)


def _gg_from_si(si):
    """Gamma-gamma α = β chosen so that σ_I² = 1/α + 1/β + 1/(αβ)."""
    a = (1 + np.sqrt(1 + si)) / si
    return float(a), float(a)


# =========================================================================== pointing
@dataclass
class Pointing:
    """Static boresight offset (m) plus Gaussian jitter (σ, m) of the beam centre at the receiver."""
    offset_m: float = 0.0
    jitter_m: float = 0.0
    coherence_s: float = 50e-3

    def sample(self, n, dt, rng):
        if self.jitter_m <= 0 or n == 0:
            return np.full(n, self.offset_m)
        x = self.offset_m + self.jitter_m * _ar1(n, dt, self.coherence_s, rng)
        y = self.jitter_m * _ar1(n, dt, self.coherence_s, rng)
        return np.hypot(x, y)


# =========================================================================== ambient light
SOLAR = {"night": 0.0, "moonlight": 2e-6, "overcast": 0.25, "clear sky": 1.0}


@dataclass
class Ambient:
    """Ambient (background) light. Downwelling irradiance E_d(z) = E_s e^{-K_d z} with
    E_s ≈ 1.4 W m⁻² nm⁻¹ at noon times a sky factor; radiance seen by the receiver
    L = f_dir E_d / π with f_dir = 1 looking up, 0.1 horizontal, 0.02 looking down.
    Background power: P_b = L · A · π sin²(FOV) · Δλ · T_filter, plus an artificial lamp term."""
    sky: str = "overcast"
    depth_m: float = 20.0
    look: str = "horizontal"
    lamp_w_m2_nm: float = 0.0          # extra irradiance at the receiver (lamps, bioluminescence)

    def radiance(self, water: Water, lam_nm):
        Es = 1.4 * SOLAR.get(self.sky, 0.0)
        Ed = Es * float(np.exp(-water.Kd(lam_nm) * self.depth_m))
        f = {"up": 1.0, "horizontal": 0.1, "down": 0.02}.get(self.look, 0.1)
        return (f * Ed + self.lamp_w_m2_nm) / np.pi          # W m⁻² sr⁻¹ nm⁻¹

    def power(self, water, lam_nm, area_m2, fov_rad, filter_nm, filter_T):
        return self.radiance(water, lam_nm) * area_m2 * np.pi * np.sin(fov_rad) ** 2 * filter_nm * filter_T


# =========================================================================== link geometry + channel
@dataclass
class Geometry:
    range_m: float = 20.0
    wavelength_nm: float = 470.0
    divergence_deg: float = 5.0       # 1/e² half-angle
    waist_mm: float = 2.0
    aperture_mm: float = 50.0          # receiver lens diameter
    fov_deg: float = 10.0              # receiver half-angle FOV

    @property
    def aperture_radius(self):
        return self.aperture_mm / 2000

    @property
    def area(self):
        return np.pi * self.aperture_radius ** 2


@dataclass
class ChannelResult:
    L: float
    gain_ballistic: float
    gain_scattered: float
    t: np.ndarray                     # excess delay grid of the scattered impulse response
    h: np.ndarray                     # scattered impulse response, 1/s (∫ h dt = gain_scattered)
    model: str
    mean_delay: float
    rms_delay: float
    geom: float
    mc: object = None
    gamma_k: float = 0.0              # analytic scattered response: gamma(k, mean tau)
    gamma_tau: float = 0.0

    @property
    def gain(self):
        return self.gain_ballistic + self.gain_scattered

    def frequency_response(self, f):
        """H(f) normalised to H(0) = 1."""
        f = np.asarray(f, float)
        if self.gamma_k > 0:
            th = self.gamma_tau / self.gamma_k
            H = self.gain_ballistic + self.gain_scattered * (1 + 2j * np.pi * f * th) ** (-self.gamma_k)
            return H / max(self.gain, 1e-300)
        dt = self.t[1] - self.t[0] if self.t.size > 1 else 0.0
        # histogram bins are treated as uniform over their width (sinc), so a narrow pulse
        # inside the first bin does not alias back to |H| = 1 at high frequency
        H = self.gain_ballistic + (self.h[None, :] * np.exp(-2j * np.pi * f[:, None] * self.t[None, :])).sum(1) \
            * dt * np.sinc(f * dt)
        return H / max(self.gain, 1e-300)

    def bandwidth_3db(self):
        if self.gain_scattered <= 1e-6 * self.gain or self.rms_delay <= 0:
            return np.inf
        f = np.geomspace(1e5, 1e11, 600 if self.gamma_k > 0 else 200)
        H = np.abs(self.frequency_response(f))
        below = np.nonzero(H < 1 / np.sqrt(2))[0]
        return float(f[below[0]]) if below.size else np.inf

    def kernel(self, dt):
        """Discrete impulse response on a sample grid dt (sum = total gain), first tap = ballistic."""
        if self.gain_scattered <= 0 or self.t.size < 2:
            return np.array([self.gain])
        n = int(np.ceil(self.t[-1] / dt)) + 1
        n = min(max(n, 1), 4096)
        edges = np.arange(n + 1) * dt
        cdf = np.concatenate([[0], np.cumsum(self.h) * (self.t[1] - self.t[0])])
        tt = np.concatenate([[0], self.t + (self.t[1] - self.t[0]) / 2])
        k = np.diff(np.interp(edges, tt, cdf))
        k[0] += self.gain_ballistic
        nz = np.nonzero(k > 1e-6 * k.max())[0]
        return k[: nz[-1] + 1]


def forward_lobe(pf, theta_c_deg=20.0):
    """Split a phase function at θc: the fraction P scattered inside the forward lobe and the
    rms angle of a single forward scattering inside it."""
    cache = pf.__dict__.setdefault("_lobes", {})
    if theta_c_deg in cache:
        return cache[theta_c_deg]
    tc = np.radians(theta_c_deg)
    k = pf.theta <= tc
    P = float(np.interp(tc, pf.theta, pf.cdf))
    pdf = np.gradient(pf.cdf, pf.theta)
    m2 = float(trapz((pf.theta ** 2 * pdf)[k], pf.theta[k]) / max(P, 1e-12))
    cache[theta_c_deg] = (P, np.sqrt(m2))
    return P, np.sqrt(m2)


def _small_angle_scattered(water, lam, geo: Geometry, offset=0.0):
    """Analytic approximation of the captured scattered power and its delay distribution.
    Scattering inside a 15° forward lobe (fraction P, rms angle θ₁) keeps light near the beam;
    wider-angle scattering is treated as loss. The angular spread at the receiver grows as
    θ₁ (P b L)^0.75 (multiple forward scattering of a peaked phase function: a narrow core with a
    wide tail, so slower than the √(bL) of Gaussian diffusion), and the footprint and FOV
    acceptance use fitted core widths. The three constants were fitted to this package's Monte
    Carlo over Jerlov IB / 3C / harbour waters, cL = 3–15, divergence 1–20°, FOV 5–40°:
    rms error of the captured scattered power is a factor 1.44, worst case about 2."""
    a, b, c = float(water.a(lam)), float(water.b(lam)), float(water.c(lam))
    L = geo.range_m
    pf = water.phase_function(lam)
    P, th1 = forward_lobe(pf, 15.0)
    fwd = max(np.exp(-(a + (1 - P) * b) * L) - np.exp(-c * L), 0.0)
    th_s = th1 * max(P * b * L, 1e-9) ** 0.75
    th_beam = np.radians(geo.divergence_deg)
    w_sc = np.sqrt(beam_radius(L, th_beam, geo.waist_mm / 1e3) ** 2 + 0.0224 * (L * th_s) ** 2)
    R = geo.aperture_radius
    cap = (1 - np.exp(-2 * R ** 2 / w_sc ** 2)) * np.exp(-2 * offset ** 2 / max(w_sc ** 2, 1e-12))
    fov = np.radians(geo.fov_deg)
    acc = 1 - np.exp(-fov ** 2 / max(0.389 * th_s ** 2, 1e-12))   # arrivals inside the FOV
    gain = fwd * cap * acc
    # mean excess delay of the captured scattered light, fitted to the same Monte Carlo set
    # (rms error a factor 1.2): grows as (bL)^1.17, opens up with FOV and beam divergence, and
    # scales with the squared forward-scattering angle relative to Petzold-like water
    from .water import phase_function
    th2 = forward_lobe(pf, 20.0)[1]
    th2_ref = forward_lobe(phase_function("Fournier-Forand"), 20.0)[1]
    fov_c = np.radians(min(max(geo.fov_deg, 0.5), 60.0))
    div_c = np.radians(min(max(geo.divergence_deg, 0.1), 45.0))
    tau = (L / V_WATER) * np.exp(-6.19) * (b * L) ** 1.17 * fov_c ** 0.40 * div_c ** 0.25 * (th2 / th2_ref) ** 2
    return gain, tau + 1e-13


def channel(water: Water, geo: Geometry, model="effective", offset=0.0, mc_photons=20000, seed=0,
            progress=None, n_paths=60):
    lam, L = geo.wavelength_nm, geo.range_m
    c, b = float(water.c(lam)), float(water.b(lam))
    th = np.radians(geo.divergence_deg)
    eta, _ = gaussian_capture(L, th, geo.waist_mm / 1e3, geo.aperture_radius, offset)
    # unscattered light arrives from the transmitter direction: rejected if outside the FOV
    eta = float(eta) * float(np.clip((np.radians(geo.fov_deg) - np.arctan2(abs(offset), L)) / 1e-3 + 0.5, 0, 1))
    if model == "mc":
        from . import montecarlo
        r = montecarlo.simulate(c, b, water.phase_function(lam), L, th, geo.waist_mm / 1e3, geo.aperture_radius,
                                np.radians(geo.fov_deg), offset=(offset, 0.0), n_photons=mc_photons, seed=seed,
                                progress=progress, n_paths=n_paths)
        dt = r.t_edges[1] - r.t_edges[0]
        tc = 0.5 * (r.t_edges[1:] + r.t_edges[:-1])
        m, s = r.delay_stats()
        return ChannelResult(L, r.ballistic, r.scattered, tc, r.h_scat / dt, "mc", m, s, r.geom, mc=r)
    gb = eta * np.exp(-c * L)
    if model == "beer":
        return ChannelResult(L, gb, 0.0, np.zeros(1), np.zeros(1), "beer", 0.0, 0.0, eta)
    gs, tau = _small_angle_scattered(water, lam, geo, offset)
    k = 1.5                                            # gamma-distributed excess delay, mean tau
    t = np.linspace(0, 12 * tau, 400)[1:]
    from scipy.special import gammaln
    th = tau / k
    h = gs * np.exp((k - 1) * np.log(t / th) - t / th - gammaln(k)) / th
    tot = gb + gs
    mean = gs * tau / tot if tot > 0 else 0.0
    var = (gs * (tau ** 2 / k + tau ** 2) / tot - mean ** 2) if tot > 0 else 0.0
    return ChannelResult(L, gb, gs, t, h, "effective", mean, float(np.sqrt(max(var, 0))), eta,
                         gamma_k=k, gamma_tau=tau)


def path_loss_db(water, geo, ranges, model="effective", offset=0.0):
    out = []
    for L in np.atleast_1d(ranges):
        g = replace(geo, range_m=float(L))
        out.append(channel(water, g, model, offset).gain)
    return -10 * np.log10(np.maximum(np.array(out), 1e-300))
