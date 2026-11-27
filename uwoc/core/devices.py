"""Transmitters, photodetectors and cameras.

Photodetection (all detectors share one model, per sample of length dt):
    n̄ = η P dt / (hν) + r_d dt                    mean primary photoelectrons
    N ~ Poisson(n̄)                                 shot noise (exact, any light level)
    Q = M N + sqrt(N M² (F - 1)) z                 gain with excess noise factor F
    i = q Q / dt + i_n sqrt(f_s / 2) z'            front-end (thermal / amplifier) noise
followed by the receiver's 2nd-order Butterworth bandwidth.
    PIN   M = 1, F = 1, dark current shot noise, TIA noise i_n
    APD   F = k M + (1 - k)(2 - 1/M)  (McIntyre)
    SiPM  microcell saturation n̄_eff = n̄ / (1 + n̄ τ_rec / (N_cells dt)), optical crosstalk p_ct
          (F ≈ 1 + p_ct), high gain so electronics noise is negligible: photon counting
    PMT   gain ≈ 10⁶, F ≈ 1.2, dark counts
Analytic electrical noise variance in bandwidth B (used for link budgets and closed-form BER):
    σ² = 2 q M² F (R P + I_d) B + i_n² B
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .constants import Q, photon_energy


# =========================================================================== transmitter
@dataclass
class Transmitter:
    kind: str = "LED"                     # 'LED' or 'Laser diode'
    wavelength_nm: float = 470.0
    power_w: float = 1.0                  # optical power when the drive is fully on (x = 1)
    bandwidth_hz: float = 20e6            # modulation 3 dB bandwidth (1st-order response)
    divergence_deg: float = 10.0          # 1/e² half-angle
    waist_mm: float = 2.0
    extinction: float = 0.0               # P(x=0)/P(x=1)
    source_mm: float = 30.0               # emitting diameter seen by a camera
    array: int = 1                        # LED array side for spatial multiplexing (camera)
    pitch_mm: float = 40.0                # LED array pitch
    rgb_nm: tuple = (625.0, 525.0, 460.0) # CSK colours

    def copy(self, **kw):
        return replace(self, **kw)

    def drive(self, x):
        """Normalised drive x ∈ [0, 1] -> optical power (W), with extinction-ratio leakage."""
        x = np.clip(x, 0.0, 1.0)
        return self.power_w * (self.extinction + (1 - self.extinction) * x)


TRANSMITTERS = {
    "Blue LED, 1 W": Transmitter("LED", 470, 1.0, 20e6, 15, 2.0, 0.0, 30),
    "Blue LED array, 10 W": Transmitter("LED", 470, 10.0, 8e6, 30, 20.0, 0.0, 120),
    "Green LED, 1 W": Transmitter("LED", 525, 1.0, 15e6, 15, 2.0, 0.0, 30),
    "Blue laser diode, 0.5 W": Transmitter("Laser diode", 450, 0.5, 1e9, 1.0, 1.5, 0.02, 3),
    "Green laser (532 nm), 1 W": Transmitter("Laser diode", 532, 1.0, 500e6, 0.5, 1.0, 0.02, 3),
    "Micro-LED, 10 mW": Transmitter("LED", 450, 0.01, 400e6, 20, 0.2, 0.0, 1),
    "RGB LED, 3 W": Transmitter("LED", 525, 3.0, 10e6, 20, 2.0, 0.0, 40),
}


# =========================================================================== receiver optics
@dataclass
class ReceiverOptics:
    aperture_mm: float = 50.0
    fov_deg: float = 10.0
    filter_nm: float = 30.0
    filter_T: float = 0.8
    optics_T: float = 0.9


# =========================================================================== photodetectors
@dataclass
class Detector:
    kind: str = "PIN"
    qe: float = 0.6                       # quantum efficiency / PDE
    gain: float = 1.0
    k_ion: float = 0.02                   # APD ionisation ratio
    excess: float = 1.0                   # PMT / SiPM gain excess noise factor
    dark: float = 1e-9                    # PIN/APD: dark current (A); SiPM/PMT: dark count rate (1/s)
    bandwidth_hz: float = 50e6
    noise_density: float = 2e-12          # input-referred amplifier noise (A/√Hz)
    n_cells: int = 0                      # SiPM microcells
    recovery_s: float = 0.0               # SiPM microcell recovery time
    crosstalk: float = 0.0
    name: str = ""

    def copy(self, **kw):
        return replace(self, **kw)

    @property
    def counting(self):
        return self.kind in ("SiPM", "PMT")

    def responsivity(self, lam_nm):
        return self.qe * Q / photon_energy(lam_nm)

    def F(self):
        if self.kind == "APD":
            M, k = self.gain, self.k_ion
            return k * M + (1 - k) * (2 - 1 / M)
        if self.kind == "SiPM":
            return self.excess * (1 + self.crosstalk)
        return self.excess if self.kind == "PMT" else 1.0

    def dark_rate(self):
        """Primary dark electrons per second."""
        return self.dark if self.counting else self.dark / Q

    def saturation(self, rate):
        """SiPM: detected fraction of a photoelectron rate (non-paralysable microcells)."""
        if self.kind != "SiPM" or self.n_cells <= 0:
            return np.ones_like(np.asarray(rate, float))
        return 1.0 / (1.0 + np.asarray(rate, float) * self.recovery_s / self.n_cells)

    def mean_current(self, P, lam_nm):
        rate = self.qe * np.asarray(P, float) / photon_energy(lam_nm) + self.dark_rate()
        return Q * self.gain * (1 + self.crosstalk) * rate * self.saturation(rate)

    def noise_var(self, P, lam_nm, B=None):
        """Electrical noise variance (A²) at optical power P (W) in bandwidth B."""
        B = self.bandwidth_hz if B is None else B
        rate = self.qe * np.asarray(P, float) / photon_energy(lam_nm) + self.dark_rate()
        rate = rate * self.saturation(rate)
        M = self.gain * (1 + self.crosstalk)
        return 2 * Q ** 2 * M ** 2 * self.F() * rate * B + self.noise_density ** 2 * B

    def detect(self, P, dt, lam_nm, rng):
        """Sampled photocurrent (A) for optical power samples P (W) of length dt (s)."""
        P = np.maximum(np.asarray(P, float), 0.0)
        rate = self.qe * P / photon_energy(lam_nm) + self.dark_rate()
        rate = rate * self.saturation(rate)
        nbar = rate * dt
        N = rng.poisson(np.minimum(nbar, 1e15)).astype(float) if np.all(nbar < 1e15) else \
            nbar + np.sqrt(nbar) * rng.standard_normal(nbar.shape)
        if self.crosstalk > 0:
            N = N + rng.poisson(N * self.crosstalk)
        M, F = self.gain, (self.excess if self.counting else self.F())
        Qn = M * N
        if F > 1:
            Qn = Qn + np.sqrt(N * M * M * (F - 1)) * rng.standard_normal(N.shape)
        i = Q * np.maximum(Qn, 0.0) / dt
        i = i + self.noise_density * np.sqrt(0.5 / dt) * rng.standard_normal(i.shape)
        return i


DETECTORS = {
    "PIN photodiode": Detector("PIN", 0.65, 1.0, 0, 1.0, 1e-9, 50e6, 2e-12, name="Si PIN, 1 mm², TIA"),
    "APD (M = 50)": Detector("APD", 0.75, 50.0, 0.02, 1.0, 1e-9, 100e6, 2e-12, name="Si APD, 1 mm²"),
    "SiPM (photon counting)": Detector("SiPM", 0.35, 1e6, 0, 1.05, 3e5, 100e6, 1e-12, n_cells=18980,
                                       recovery_s=50e-9, crosstalk=0.05, name="6 mm SiPM"),
    "PMT (photon counting)": Detector("PMT", 0.25, 1e6, 0, 1.25, 1e3, 200e6, 1e-12, name="bialkali PMT"),
}


def lowpass(x, fc, fs, order=2):
    """Zero-delay-free causal Butterworth low-pass (no filtering when fc >= 0.45 fs)."""
    if fc is None or fc <= 0 or fc >= 0.45 * fs:
        return x
    from scipy.signal import butter, lfilter
    b, a = butter(order, fc / (fs / 2))
    return lfilter(b, a, x, axis=0)


def led_response(x, fc, fs):
    """First-order LED modulation response."""
    if fc is None or fc <= 0 or fc >= 0.45 * fs:
        return x
    from scipy.signal import lfilter
    al = 1 - np.exp(-2 * np.pi * fc / fs)
    zi = np.atleast_1d(x[0] * (1 - al)) if np.ndim(x) == 1 else (x[0] * (1 - al))[None, :]
    y, _ = lfilter([al], [1, -(1 - al)], x, axis=0, zi=zi)
    return y


# =========================================================================== cameras
@dataclass
class Camera:
    """CMOS camera. Rolling shutter: row r starts exposing r·t_row after row 0; the frame period
    includes `vblank` idle rows. Global shutter: all rows expose together."""
    shutter: str = "rolling"
    width: int = 192
    height: int = 320
    pixel_um: float = 3.0
    focal_mm: float = 12.0
    f_number: float = 2.0
    fps: float = 30.0
    exposure_s: float = 1 / 4000
    vblank: int = 20                     # idle rows per frame (rolling shutter)
    qe: float = 0.6
    full_well: float = 10000.0
    read_noise: float = 3.0              # e- rms
    dark_e_s: float = 20.0               # dark current e-/s
    bits: int = 10
    color: bool = False
    color_xt: float = 0.12               # Bayer/dye crosstalk between neighbouring channels
    band_nm: float = 100.0               # optical band seen per channel (no filter)
    name: str = ""

    def copy(self, **kw):
        return replace(self, **kw)

    @property
    def aperture_m(self):
        return self.focal_mm / self.f_number / 1e3

    @property
    def row_time(self):
        """Row-to-row readout delay (rolling shutter)."""
        return 1.0 / (self.fps * (self.height + self.vblank))

    @property
    def ifov(self):
        return self.pixel_um * 1e-6 / (self.focal_mm * 1e-3)     # rad per pixel

    @property
    def fov_deg(self):
        return np.degrees(2 * np.arctan(0.5 * max(self.width, self.height) * self.ifov))

    @property
    def dn_per_e(self):
        return (2 ** self.bits - 1) / self.full_well

    def color_matrix(self):
        x = self.color_xt
        return np.array([[1 - x, x, 0.0], [x / 2, 1 - x, x / 2], [0.0, x, 1 - x]])

    def row_rate(self):
        return self.fps * (self.height + self.vblank)


CAMERAS = {
    "Rolling-shutter CMOS (mono)": Camera("rolling", 192, 320, 3.0, 12, 2.0, 30, 1 / 8000, 20, name="rolling mono"),
    "Rolling-shutter CMOS (colour)": Camera("rolling", 192, 320, 3.0, 12, 2.0, 30, 1 / 8000, 20, color=True,
                                            name="rolling colour"),
    "Global-shutter machine vision": Camera("global", 160, 120, 5.0, 16, 1.8, 120, 1 / 2000, 0, qe=0.65,
                                            full_well=20000, read_noise=6, name="global mono"),
    "Global-shutter high-speed": Camera("global", 128, 96, 6.0, 16, 1.4, 1000, 1 / 5000, 0, qe=0.55,
                                        full_well=15000, read_noise=10, name="high-speed"),
    "Global-shutter colour": Camera("global", 160, 120, 4.0, 16, 1.8, 60, 1 / 2000, 0, color=True,
                                    name="global colour"),
}
