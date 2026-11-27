"""Photodiode links: link budget, chip statistics, closed-form BER and full waveform simulation.

Waveform simulation (one frame):
  chips (preamble m-sequence | training | payload) -> rectangular pulses at `sps` samples per chip
  -> LED first-order response -> optical power -> channel impulse response (ballistic tap +
  scattered tail) -> turbulence and pointing fading -> + background light -> photodetector
  (Poisson shot noise, gain and excess noise, amplifier noise) -> receiver Butterworth filter
  -> preamble correlation for timing -> integrate-and-dump per chip -> level calibration on the
  training chips -> scheme decision -> BER.
Closed form: per-chip Gaussian statistics of the two drive levels including an intersymbol-
interference term from the composite LED / channel / receiver bandwidth,
  Δμ = κ (μ1 - μ0),  κ = 1 - (1 - e^{-2π B Tc}) / (2π B Tc),  1/B² = 1/B_led² + 1/B_ch² + 1/B_rx²
averaged over the turbulence and pointing fading distributions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import schemes as S
from .channel import gaussian_capture
from .devices import led_response, lowpass


def mseq(n=5, taps=(5, 3)):
    reg = [1] * n
    out = []
    for _ in range(2 ** n - 1):
        out.append(reg[-1])
        fb = reg[taps[0] - 1] ^ reg[taps[1] - 1]
        reg = [fb] + reg[:-1]
    return np.array(out, float)


PREAMBLE = np.concatenate([mseq(), mseq()])          # 62 chips, timing


# =========================================================================== powers + noise
def _colours(sc, scheme):
    return list(sc.tx.rgb_nm) if scheme.channels == 3 else [sc.wavelength]


def received(sc, range_m=None, wavelength=None, ch=None):
    """Peak received optical power (x = 1), background power, and the channel."""
    lam = sc.wavelength if wavelength is None else wavelength
    ch = sc.channel(range_m, lam) if ch is None else ch
    T = sc.optics.optics_T * sc.optics.filter_T
    P1 = sc.tx.power_w * ch.gain * T
    Pb = sc.ambient.power(sc.water, lam, np.pi * (sc.optics.aperture_mm / 2000) ** 2,
                          np.radians(sc.optics.fov_deg), sc.optics.filter_nm, sc.optics.filter_T)
    return dict(P1=P1, P0=P1 * sc.tx.extinction, Pb=Pb, ch=ch, T=T, lam=lam)


def isi_factor(B, Tc):
    a = 2 * np.pi * B * Tc
    if not np.isfinite(a) or a > 1e4:
        return 1.0
    return float(1 - (1 - np.exp(-a)) / a)


def composite_bandwidth(sc, ch):
    inv = 1 / sc.tx.bandwidth_hz ** 2 + 1 / sc.detector.bandwidth_hz ** 2
    bch = ch.bandwidth_3db()
    if np.isfinite(bch):
        inv += 1 / bch ** 2
    return 1 / np.sqrt(inv)


def chip_stats(sc, scheme, bit_rate=None, range_m=None, fade=1.0, rx=None, B=None):
    """Integrate-and-dump chip statistics (mean photocurrent over a chip, A) for x = 0 and 1."""
    bit_rate = sc.bit_rate if bit_rate is None else bit_rate
    Tc = 1.0 / scheme.chip_rate(bit_rate)
    rx = received(sc, range_m) if rx is None else rx
    det, lam = sc.detector, rx["lam"]
    P1, P0, Pb = rx["P1"] * fade, rx["P0"] * fade, rx["Pb"]
    Bn = 1 / (2 * Tc)                                   # integrate-and-dump noise bandwidth
    i1, i0 = det.mean_current(P1 + Pb, lam), det.mean_current(P0 + Pb, lam)
    v1, v0 = det.noise_var(P1 + Pb, lam, Bn), det.noise_var(P0 + Pb, lam, Bn)
    B = composite_bandwidth(sc, rx["ch"]) if B is None else B
    d = scheme.duty
    if scheme.family == "ofdm":
        # cyclic prefix + zero-forcing equaliser: no ISI, but noise enhancement <1/|H(f)|^2> over 0..Rc/2
        k = 1.0 / np.sqrt(1.0 + (0.5 / (Tc * B)) ** 2 / 3.0)
        isi = 0.0
    else:
        k = isi_factor(B, Tc)
        isi = ((1 - k) * (i1 - i0)) ** 2 * d * (1 - d)
    from .constants import photon_energy
    r1 = det.qe * (P1 + Pb) / photon_energy(lam) + det.dark_rate()
    r0 = det.qe * (P0 + Pb) / photon_energy(lam) + det.dark_rate()
    n1 = float(r1 * det.saturation(r1) * Tc) * k + float(r0 * det.saturation(r0) * Tc) * (1 - k)
    n0 = float(r0 * det.saturation(r0) * Tc)
    mid = 0.5 * (i0 + i1)
    m0, m1 = mid - k * (mid - i0), mid + k * (i1 - mid)
    colours = None
    if scheme.channels == 3:                       # one filtered detector per LED colour
        from types import SimpleNamespace
        mono = SimpleNamespace(family=scheme.family, duty=scheme.duty, channels=1, chip_rate=scheme.chip_rate)
        colours = []
        for lam_c in sc.tx.rgb_nm:
            stc = chip_stats(sc, mono, bit_rate, range_m, fade, rx=received(sc, range_m, wavelength=lam_c))
            colours.append((stc["m0"], stc["s0"], stc["m1"], stc["s1"]))
    return dict(colours=colours, m0=float(m0), m1=float(m1), s0=float(np.sqrt(v0 + isi)), s1=float(np.sqrt(v1 + isi)),
                n0=n0, n1=n1, Tc=Tc, kappa=k, B=B, P1=P1, Pb=Pb, i_bg=float(det.mean_current(Pb, lam)),
                shot1=float(det.noise_var(P1 + Pb, lam, Bn) - det.noise_density ** 2 * Bn),
                thermal=float(det.noise_density ** 2 * Bn),
                bg_shot=float(det.noise_var(Pb, lam, Bn) - det.noise_var(0.0, lam, Bn)),
                dark_shot=float(det.noise_var(0.0, lam, Bn) - det.noise_density ** 2 * Bn), isi=float(isi))


def chip_ber(sc, scheme, st):
    """BER from chip statistics: exact Poisson counting for SiPM / PMT when the scheme supports it,
    otherwise the Gaussian closed form."""
    if sc.detector.counting and st["n1"] < 3000:
        v = scheme.theory_counts(st["n0"], st["n1"])
        if v is not None:
            return float(v)
    if st.get("colours") is not None and hasattr(scheme, "theory_colour"):
        return scheme.theory_colour(st["colours"])
    v = scheme.theory(st["m0"], st["s0"], st["m1"], st["s1"])
    return float(v) if v is not None else np.nan


def fading_samples(sc, n=64, range_m=None, seed=3):
    """Relative gain samples combining turbulence (stratified quantiles) and pointing jitter."""
    rng = np.random.default_rng(seed)
    L = sc.range_m if range_m is None else range_m
    si = sc.turbulence.sigma_I2(sc.wavelength, L, sc.optics.aperture_mm / 1e3)
    h = sc.turbulence.draw(n, si) if sc.turbulence.model != "none" else np.ones(n)
    if sc.pointing.jitter_m > 0:
        m = h.size
        r = np.hypot(sc.pointing.offset_m + sc.pointing.jitter_m * rng.standard_normal(m),
                     sc.pointing.jitter_m * rng.standard_normal(m))
        h = h * pointing_ratio(sc, r, L)
    return h


def pointing_ratio(sc, r, L=None):
    L = sc.range_m if L is None else L
    th = np.radians(sc.tx.divergence_deg)
    g, _ = gaussian_capture(L, th, sc.tx.waist_mm / 1e3, sc.optics.aperture_mm / 2000, r)
    g0, _ = gaussian_capture(L, th, sc.tx.waist_mm / 1e3, sc.optics.aperture_mm / 2000, sc.pointing.offset_m)
    return g / max(float(g0), 1e-300)


def ber_theory(sc, scheme, bit_rate=None, range_m=None, rx=None, faded=True):
    """Closed-form BER (averaged over fading when turbulence / jitter are on)."""
    rx = received(sc, range_m) if rx is None else rx
    B = composite_bandwidth(sc, rx["ch"])
    if faded and (sc.turbulence.model != "none" or sc.pointing.jitter_m > 0):
        hs = fading_samples(sc, 64, range_m)
        vals = []
        for h in hs:
            st = chip_stats(sc, scheme, bit_rate, range_m, fade=h, rx=rx, B=B)
            vals.append(chip_ber(sc, scheme, st))
        return float(np.mean(vals))
    return chip_ber(sc, scheme, chip_stats(sc, scheme, bit_rate, range_m, rx=rx, B=B))


def link_budget(sc, scheme=None, bit_rate=None, target_ber=1e-6):
    scheme = S.make(sc.scheme) if scheme is None else scheme
    rx = received(sc)
    ch = rx["ch"]
    th = np.radians(sc.tx.divergence_deg)
    geo, _ = gaussian_capture(sc.range_m, th, sc.tx.waist_mm / 1e3, sc.optics.aperture_mm / 2000,
                              sc.pointing.offset_m)
    lam = sc.wavelength
    c = float(sc.water.c(lam))
    db = lambda x: 10 * np.log10(max(float(x), 1e-300))
    Pt = sc.tx.power_w
    st = chip_stats(sc, scheme, bit_rate, rx=rx)
    q = (st["m1"] - st["m0"]) / (st["s0"] + st["s1"])
    si = sc.turbulence.sigma_I2(lam, sc.range_m, sc.optics.aperture_mm / 1e3)
    terms = [
        ("Tx peak power", db(Pt * 1e3)),
        ("Geometric loss", db(geo)),
        ("Beam attenuation e^{-cL}", db(np.exp(-c * sc.range_m))),
        ("Scattered light recovered", db(ch.gain / max(ch.gain_ballistic, 1e-300)) if ch.gain_ballistic > 0 else 0.0),
        ("Receiver optics + filter", db(rx["T"])),
    ]
    Pr = rx["P1"]
    out = dict(terms=terms, Pr_dBm=db(Pr * 1e3), Pavg_dBm=db(Pr * scheme.duty * 1e3), Pb_dBm=db(rx["Pb"] * 1e3),
               Q=q, snr_db=20 * np.log10(max(q, 1e-12)), ber=ber_theory(sc, scheme, bit_rate, rx=rx),
               ber_nofade=ber_theory(sc, scheme, bit_rate, rx=rx, faded=False), stats=st, si=si,
               gain=ch.gain, ballistic=ch.gain_ballistic, scattered=ch.gain_scattered, c=c,
               attenuation_lengths=c * sc.range_m, rms_delay=ch.rms_delay, B_channel=ch.bandwidth_3db(),
               kappa=st["kappa"], chip_rate=scheme.chip_rate(sc.bit_rate if bit_rate is None else bit_rate))
    return out


def required_power_dbm(sc, scheme, target=1e-6, bit_rate=None):
    """Peak received power needed for the target BER (bisection on a power scale)."""
    rx0 = received(sc)
    lo, hi = -40.0, 40.0             # dB relative to current
    B = composite_bandwidth(sc, rx0["ch"])

    def ber_at(dbr):
        rx = dict(rx0, P1=rx0["P1"] * 10 ** (dbr / 10), P0=rx0["P0"] * 10 ** (dbr / 10))
        return chip_ber(sc, scheme, chip_stats(sc, scheme, bit_rate, rx=rx, B=B))
    if ber_at(hi) > target:
        return np.inf
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if ber_at(mid) > target:
            lo = mid
        else:
            hi = mid
    return 10 * np.log10(rx0["P1"] * 1e3) + hi


# =========================================================================== waveform simulation
@dataclass
class PDRun:
    scheme: object
    fs: float
    sps: int
    chip_rate: float
    t: np.ndarray                     # sample times (s)
    drive: np.ndarray                 # normalised drive (after LED response), (n,) or (n, 3)
    p_rx: np.ndarray                  # noiseless received optical power incl. background (W)
    i_rx: np.ndarray                  # detector output current (A), noisy, before receiver filter
    i_f: np.ndarray                   # after receiver filter (A)
    fade: np.ndarray
    offset: int                       # timing offset (samples)
    n_pre: int
    n_train: int
    y: np.ndarray                     # chip statistics (payload)
    y_train: np.ndarray
    train_labels: np.ndarray
    cal: dict
    tx_bits: np.ndarray
    rx_bits: np.ndarray
    tx_syms: np.ndarray
    rx_syms: np.ndarray
    errors: float
    n_bits: int
    extra: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    @property
    def ber(self):
        return self.errors / max(self.n_bits, 1)

    def eye(self, span_chips=2, max_traces=300):
        """Folded traces of the filtered waveform, span_chips chips long (payload region)."""
        x = self.i_f if self.i_f.ndim == 1 else self.i_f.sum(1)
        L = span_chips * self.sps
        start = self.offset + (self.n_pre + self.n_train) * self.sps
        n = min((x.size - start - L) // self.sps, max_traces)
        if n <= 0:
            return np.zeros((0, L + 1)), np.arange(L + 1) / self.sps
        idx = start + np.arange(n)[:, None] * self.sps + np.arange(L + 1)[None, :] - self.sps // 2
        idx = np.clip(idx, 0, x.size - 1)
        return x[idx], (np.arange(L + 1) - self.sps // 2) / self.sps

    def q_factor(self):
        c = self.cal
        if "m1" in c and "s0" in c:
            return (c["m1"] - c["m0"]) / (c["s0"] + c["s1"])
        return np.nan


def simulate(sc, scheme=None, n_bits=2000, sps=8, seed=0, bit_rate=None, ch=None):
    if scheme is None:
        scheme = S.make(sc.scheme)
    elif isinstance(scheme, str):
        scheme = S.make(scheme)
    rng = np.random.default_rng(seed)
    bit_rate = sc.bit_rate if bit_rate is None else bit_rate
    Rc = scheme.chip_rate(bit_rate)
    fs = sps * Rc
    dt = 1 / fs
    bits = rng.integers(0, 2, scheme.n_bits(n_bits)).astype(np.int8)
    chips, syms = scheme.encode(bits)
    tr, labels = scheme.training()
    nch = scheme.channels
    # preamble: white flashes for CSK (1/3 per colour keeps the total power of a CSK symbol)
    pre = PREAMBLE if nch == 1 else np.repeat(PREAMBLE[:, None], 3, 1) / 3
    guard = np.zeros((16,) + ((3,) if nch == 3 else ()))
    seq = np.concatenate([guard, pre, tr, chips, guard])
    x = np.repeat(seq, sps, axis=0)
    x = led_response(x, sc.tx.bandwidth_hz, fs)
    lams = _colours(sc, scheme)
    n = x.shape[0]
    # fading (shared by colours)
    si = sc.turbulence.sigma_I2(sc.wavelength, sc.range_m, sc.optics.aperture_mm / 1e3)
    fade = sc.turbulence.sample(n, dt, si, rng) if sc.turbulence.model != "none" else np.ones(n)
    if sc.pointing.jitter_m > 0:
        fade = fade * pointing_ratio(sc, sc.pointing.sample(n, dt, rng))
    from scipy.signal import fftconvolve
    cols_p, cols_i, cols_f = [], [], []
    info = {}
    for j, lam in enumerate(lams):
        chj = sc.channel(wavelength=lam) if (ch is None or nch == 3) else ch
        rx = received(sc, wavelength=lam, ch=chj)
        xj = x if nch == 1 else x[:, j]
        P = sc.tx.drive(xj) * rx["T"]
        k = chj.kernel(dt)
        Pr = fftconvolve(P, k)[:n] if k.size > 1 else P * k[0]
        Pr = Pr * fade + rx["Pb"]
        i = sc.detector.detect(Pr, dt, lam, rng)
        f = lowpass(i, sc.detector.bandwidth_hz, fs)
        cols_p.append(Pr); cols_i.append(i); cols_f.append(f)
        if j == 0:
            info.update(gain=chj.gain, ballistic=chj.gain_ballistic, scattered=chj.gain_scattered,
                        kernel_len=k.size, Pb=rx["Pb"], P1=sc.tx.power_w * chj.gain * rx["T"],
                        rms_delay=chj.rms_delay)
    stack = (lambda L: L[0]) if nch == 1 else (lambda L: np.stack(L, 1))
    p_rx, i_rx, i_f = stack(cols_p), stack(cols_i), stack(cols_f)
    # timing: correlate the (colour-summed) filtered signal with the preamble
    sig = i_f if nch == 1 else i_f.sum(1)
    tmpl = np.repeat(PREAMBLE - PREAMBLE.mean(), sps)
    g0 = guard.shape[0] * sps
    search = np.arange(0, min(8 * sps + 4 * max(info.get("kernel_len", 1), 1), n - g0 - tmpl.size - 1))
    seg = np.array([np.dot(sig[g0 + s: g0 + s + tmpl.size], tmpl) for s in search])
    off = int(search[np.argmax(seg)]) if search.size else 0
    # integrate and dump (central 3/4 of each chip)
    a, b = sps // 8, sps - sps // 8
    def chipify(start_chip, count):
        st = g0 + off + start_chip * sps
        idx = st + np.arange(count)[:, None] * sps + np.arange(a, b)[None, :]
        idx = np.clip(idx, 0, n - 1)
        return i_f[idx].mean(1)
    npre, ntr = pre.shape[0], tr.shape[0]
    y_tr = chipify(npre, ntr)
    y = chipify(npre + ntr, chips.shape[0])
    cal = scheme.calibrate(y_tr, labels)
    rs, rb, extra = scheme.decode(y, cal, n_sym=len(syms))
    e, nb = scheme.ber(syms, bits, rs, rb)
    t = np.arange(n) * dt
    info.update(si=si, bit_rate=bit_rate, avg_power_tx=float(np.mean(sc.tx.drive(chips))) if nch == 1 else
                sc.tx.power_w / 3, duty=scheme.duty)
    return PDRun(scheme, fs, sps, Rc, t, x, p_rx, i_rx, i_f, fade, off + g0, npre, ntr, y, y_tr, labels, cal,
                 bits, rb, syms, rs, e, nb, extra, info)


def monte_carlo_ber(sc, scheme, ranges=None, powers_db=None, n_bits=4000, sps=4, seed=0, progress=None,
                    ctx=None, bit_rate=None):
    """Waveform Monte Carlo BER at several ranges (or Tx power offsets in dB)."""
    out = []
    xs = ranges if ranges is not None else powers_db
    for i, v in enumerate(xs):
        if ctx is not None:
            ctx.check()
        s2 = sc.copy(range_m=float(v)) if ranges is not None else sc.copy()
        if powers_db is not None:
            s2.tx.power_w = sc.tx.power_w * 10 ** (v / 10)
        r = simulate(s2, scheme, n_bits=n_bits, sps=sps, seed=seed + i, bit_rate=bit_rate)
        out.append(r.ber)
        if progress is not None:
            progress(i + 1, len(xs))
    return np.array(out)


def ber_vs_range(sc, scheme, ranges, bit_rate=None, faded=True):
    out = []
    for L in ranges:
        rx = received(sc, L)
        out.append(ber_theory(sc, scheme, bit_rate, L, rx=rx, faded=faded))
    return np.array(out)


def max_rate_vs_range(sc, scheme, ranges, target=1e-6, rates=None):
    """Largest bit rate (on a log grid) meeting the target BER at each range (closed form).

    Chip rates above 10x the composite 3 dB bandwidth are not considered: the first-order roll-off
    model of the LED/laser and detector is not trustworthy that far out."""
    rates = np.geomspace(1e3, 1e10, 57) if rates is None else rates
    out = []
    for L in ranges:
        rx = received(sc, L)
        B = composite_bandwidth(sc, rx["ch"])
        best = 0.0
        for R in rates:
            if scheme.chip_rate(R) > 10 * B:
                break
            if chip_ber(sc, scheme, chip_stats(sc, scheme, R, L, rx=rx, B=B)) <= target:
                best = R
            elif best > 0:
                break
        out.append(best)
    return np.array(out)


def max_range(sc, scheme, target=1e-6, bit_rate=None, lo=0.5, hi=500.0):
    f = lambda L: ber_theory(sc, scheme, bit_rate, L, faded=False) - target
    if f(lo) > 0:
        return 0.0
    if f(hi) < 0:
        return hi
    for _ in range(40):
        m = np.sqrt(lo * hi)
        if f(m) > 0:
            hi = m
        else:
            lo = m
    return float(lo)
