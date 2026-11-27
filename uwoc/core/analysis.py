"""Analytics shared by the GUI, the CLI and the README gallery."""
from __future__ import annotations

import numpy as np

from . import camera as C
from . import link
from . import schemes as S


def iop_spectra(water, lam=None):
    lam = np.arange(400, 701, 2.0) if lam is None else lam
    return dict(lam=lam, a=water.a(lam), b=water.b(lam), c=water.c(lam), bb=water.bb(lam), Kd=water.Kd(lam))


def received_vs_range(sc, ranges, models=("beer", "effective")):
    """Peak received optical power (dBm) against range for several channel models."""
    out = {}
    for m in models:
        vals = []
        for L in ranges:
            ch = sc.channel(L, model=m)
            vals.append(10 * np.log10(max(sc.tx.power_w * ch.gain * sc.optics.optics_T * sc.optics.filter_T, 1e-300) * 1e3))
        out[m] = np.array(vals)
    return out


def ber_curves(sc, names, ranges, bit_rate=None, faded=True):
    out = {}
    for n in names:
        sch = S.make(n)
        out[n] = link.ber_vs_range(sc, sch, ranges, bit_rate, faded)
    return out


def ber_vs_power(sc, names, pr_dbm, bit_rate=None):
    """Closed-form BER against peak received power (dBm) with the scenario's background and receiver."""
    rx0 = link.received(sc)
    out = {}
    B = link.composite_bandwidth(sc, rx0["ch"])
    for n in names:
        sch = S.make(n)
        vals = []
        for p in pr_dbm:
            P1 = 10 ** (p / 10) / 1e3
            rx = dict(rx0, P1=P1, P0=P1 * sc.tx.extinction)
            vals.append(link.chip_ber(sc, sch, link.chip_stats(sc, sch, bit_rate, rx=rx, B=B)))
        out[n] = np.array(vals)
    return out


def mc_vs_power(sc, name, pr_dbm, n_bits=4000, sps=4, bit_rate=None, ctx=None, seed=0):
    """Waveform Monte Carlo BER against peak received power, by scaling the Tx power."""
    rx0 = link.received(sc)
    p0 = 10 * np.log10(rx0["P1"] * 1e3)
    out = []
    for i, p in enumerate(pr_dbm):
        if ctx is not None:
            ctx.progress(i, len(pr_dbm))
        s2 = sc.copy()
        s2.tx.power_w = sc.tx.power_w * 10 ** ((p - p0) / 10)
        r = link.simulate(s2, name, n_bits=n_bits, sps=sps, seed=seed + i, bit_rate=bit_rate)
        out.append(r.ber)
    return np.array(out)


def throughput_curves(sc, names, ranges, target=1e-6):
    return {n: link.max_rate_vs_range(sc, S.make(n), ranges, target) for n in names}


def efficiency_map(names=None, target=1e-6):
    """Power and bandwidth efficiency of each scheme in thermal-noise-limited IM/DD at the same
    bit rate: bandwidth efficiency = bits per chip; power requirement = average optical power
    for the target BER, in dB relative to OOK-NRZ (noise variance ∝ chip rate)."""
    names = list(S.SCHEMES) if names is None else names
    def req(sch):
        sig = np.sqrt(1 / sch.bits_per_chip)
        lo, hi = -40.0, 60.0
        f = lambda d: sch.theory(0.0, sig, 10 ** (d / 20), sig)
        if sch.channels == 3:
            f = lambda d: sch.theory(0.0, sig, 10 ** (d / 20), sig)
        if f(hi) > target:
            return np.nan
        for _ in range(60):
            m = 0.5 * (lo + hi)
            if f(m) > target:
                lo = m
            else:
                hi = m
        return 20 * np.log10(sch.duty) + hi if sch.channels == 1 else 20 * np.log10(1 / 3) + hi
    ref = req(S.make("OOK-NRZ"))
    out = []
    for n in names:
        sch = S.make(n)
        r = req(sch)
        out.append((n, sch.bits_per_chip, (r - ref) / 2 if np.isfinite(r) else np.nan, sch.duty))
    return out


def camera_ber_vs_range(sc, names, ranges):
    out = {}
    for n in names:
        if not C.applicable(n, sc.camera):
            continue
        vals = []
        for L in ranges:
            vals.append(C.ber_theory(sc.copy(range_m=float(L)), n))
        out[n] = np.array(vals)
    return out


def camera_mc_vs_range(sc, name, ranges, n_frames=16, ctx=None):
    out, obs = [], []
    for i, L in enumerate(ranges):
        if ctx is not None:
            ctx.progress(i, len(ranges))
        r = C.simulate(sc.copy(range_m=float(L)), name, n_frames=n_frames, seed=i)
        out.append(r.ber if r.n_bits else np.nan)
        obs.append(r.goodput)
    return np.array(out), np.array(obs)


def psd(x, fs, nper=512):
    from scipy.signal import welch
    x = np.asarray(x, float)
    if x.ndim > 1:
        x = x.sum(1)
    f, p = welch(x - x.mean(), fs=fs, nperseg=min(nper, max(16, x.size // 4)))
    return f, p


def link_summary(sc):
    sch = S.make(sc.scheme)
    b = link.link_budget(sc, sch)
    return dict(scheme=sch.name, chip_rate=b["chip_rate"], Pr_dBm=b["Pr_dBm"], Pavg_dBm=b["Pavg_dBm"],
                Pb_dBm=b["Pb_dBm"], Q=b["Q"], ber=b["ber"], gain_dB=10 * np.log10(max(b["gain"], 1e-300)),
                attenuation_lengths=b["attenuation_lengths"], rms_delay_ns=b["rms_delay"] * 1e9,
                B_channel_MHz=b["B_channel"] / 1e6, scintillation_index=b["si"],
                max_range_1e6=link.max_range(sc, sch, 1e-6), max_range_fec=link.max_range(sc, sch, 3.8e-3))
