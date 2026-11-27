"""Regenerate the README figures:  python -m uwoc.examples.gallery docs/figures"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..core import analysis, camera as C, link, montecarlo, schemes as S  # noqa: E402
from ..core.scenarios import SCENARIOS  # noqa: E402
from ..core.water import PHASE_FUNCTIONS, WATER_TYPES, phase_function  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 110,
                     "axes.spines.top": False, "axes.spines.right": False})
COLS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#76b7b2", "#edc948", "#9c755f", "#ff9da7", "#bab0ac"]


def fig_water(path):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    lam = np.arange(400, 701, 2.0)
    for i, (k, w) in enumerate((k, w) for k, w in WATER_TYPES.items() if k.startswith("Jerlov")):
        ax[0].semilogy(lam, w.Kd(lam), color=COLS[i % 10], label=k.replace("Jerlov ", ""))
    ax[0].set(title="Jerlov water types: diffuse attenuation K_d(λ)", xlabel="wavelength (nm)", ylabel="1/m")
    ax[0].legend(ncol=2, fontsize=7)
    for i, k in enumerate(["Pure sea water", "Clear ocean", "Coastal ocean", "Turbid harbour"]):
        w = WATER_TYPES[k]
        ax[1].semilogy(lam, w.c(lam), color=COLS[i], label=f"{k}: c")
        ax[1].semilogy(lam, w.a(lam), color=COLS[i], ls="--", lw=1)
    ax[1].set(title="Classic waters: beam attenuation c (solid) and absorption a (dashed)",
              xlabel="wavelength (nm)", ylabel="1/m")
    ax[1].legend(fontsize=7)
    th = np.geomspace(0.01, 180, 400)
    for i, k in enumerate(PHASE_FUNCTIONS):
        pf = phase_function(k)
        ax[2].loglog(th, pf.beta(np.radians(th)), color=COLS[i], label=f"{k} (g = {pf.g:.3f}, B = {pf.B:.4f})")
    ax[2].set(title="Particle phase functions", xlabel="scattering angle (°)", ylabel="β̃ (1/sr)")
    ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_channel(path):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    base = SCENARIOS["Coastal, 10 m, LED + PIN"].copy()
    for i, (wn, R) in enumerate((("Jerlov IB", 120), ("Jerlov II", 80), ("Jerlov 3C", 25), ("Turbid harbour", 8))):
        sc = base.copy(); sc.water = WATER_TYPES[wn].copy(); sc.tx.wavelength_nm = 470 if "Jerlov I" in wn else 520
        Rs = np.geomspace(0.5, R, 40)
        pr = analysis.received_vs_range(sc, Rs)
        ax[0].semilogx(Rs, pr["effective"], color=COLS[i], label=wn)
        ax[0].semilogx(Rs, pr["beer"], color=COLS[i], ls="--", lw=1)
        Rm = np.geomspace(1, R, 5)
        mc = [10 * np.log10(sc.tx.power_w * sc.channel(L, model="mc").gain * 0.72 * 1e3) for L in Rm]
        ax[0].semilogx(Rm, mc, "o", color=COLS[i], ms=5)
    ax[0].set(title="Received power: effective model (solid), ballistic (dashed), Monte Carlo (dots)",
              xlabel="range (m)", ylabel="dBm"); ax[0].legend(fontsize=7)
    ax[0].title.set_fontsize(8)
    sc = base.copy(); sc.water = WATER_TYPES["Jerlov 3C"].copy(); sc.tx.wavelength_nm = 520
    for i, L in enumerate((3, 6, 10)):
        m = sc.channel(L, model="mc"); e = sc.channel(L, model="effective")
        ax[1].semilogy(m.t * 1e9, np.maximum(m.h / m.gain / 1e9, 1e-9), color=COLS[i], label=f"{L} m Monte Carlo")
        ax[1].semilogy(e.t * 1e9, np.maximum(e.h / e.gain / 1e9, 1e-9), color=COLS[i], ls="--", lw=1)
    ax[1].set(title="Scattered impulse response, Jerlov 3C (dashed: analytic)", xlabel="excess delay (ns)",
              ylabel="h(t)/G (1/ns)", ylim=(1e-5, 10), xlim=(0, 20)); ax[1].legend(fontsize=7)
    w = WATER_TYPES["Jerlov 3C"]
    r = montecarlo.simulate(float(w.c(520)), float(w.b(520)), w.phase_function(520), 10, np.radians(10), 0.002,
                            0.025, np.radians(20), n_photons=3000, n_paths=120)
    for i, tr in enumerate(r.paths):
        ax[2].plot(tr[:, 2], tr[:, 0], lw=0.6, color=COLS[i % 10], alpha=0.7)
    ax[2].plot([10, 10], [-0.025, 0.025], color="k", lw=4)
    ax[2].set(title="Monte Carlo photon paths, Jerlov 3C, 10 m", xlabel="z (m)", ylabel="x (m)", xlim=(-0.5, 10.5),
              ylim=(-3, 3))
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_schemes(path):
    names = ["OOK-NRZ", "OOK-RZ", "Manchester", "4-PPM", "MPPM (8,2)", "8-DPPM", "8-DPIM", "8-DH-PIM", "VPPM (25 %)",
             "4-PWM", "4-PAM", "SIM-QPSK", "DCO-OFDM", "ACO-OFDM", "Flip-OFDM", "4-CSK"]
    fig, axs = plt.subplots(len(names), 1, figsize=(13, 1.0 * len(names)), sharex=False)
    rng = np.random.default_rng(3)
    for ax, n in zip(axs, names):
        sch = S.make(n)
        c, _ = sch.encode(rng.integers(0, 2, sch.n_bits(400)).astype(np.int8))
        c = c[:96]
        if c.ndim == 1:
            ax.step(np.arange(c.size + 1), np.append(c, c[-1]), where="post", color=COLS[0], lw=1.2)
        else:
            for j, col in enumerate(("#e15759", "#59a14f", "#4e79a7")):
                ax.step(np.arange(c.shape[0] + 1), np.append(c[:, j], c[-1, j]), where="post", color=col, lw=1.2)
        ax.set_ylabel(n, rotation=0, ha="right", va="center", fontsize=8)
        ax.set_yticks([]); ax.set_xticks([]); ax.set_ylim(-0.1, 1.1); ax.set_xlim(0, 96)
        ax.text(96.5, 0.5, f"{sch.bits_per_chip:.2f} b/chip, duty {sch.duty:.2f}", fontsize=7, va="center")
    axs[0].set_title("Drive waveforms (96 chips) of a sample of the schemes")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_photodiode(sc, scheme, path, n_show=48):
    sch = S.make(scheme)
    r = link.simulate(sc, sch, n_bits=4000)
    b = link.link_budget(sc, sch)
    fig = plt.figure(figsize=(14, 6.5))
    gs = fig.add_gridspec(3, 3)
    a0 = r.offset + (r.n_pre + r.n_train) * r.sps
    sl = slice(a0, a0 + n_show * r.sps)
    t = (r.t[sl] - r.t[a0]) * 1e6
    ax = fig.add_subplot(gs[0, :2])
    d = r.drive[sl]
    if d.ndim == 1:
        ax.plot(t, d, color="#555")
    else:
        for j, col in enumerate(("#e15759", "#59a14f", "#4e79a7")):
            ax.plot(t, d[:, j], color=col)
    ax.set(title=f"{sch.name} at {r.info['bit_rate'] / 1e6:.3g} Mbit/s over {sc.range_m:g} m of {sc.water.name}: drive",
           ylabel="x")
    ax = fig.add_subplot(gs[1, :2])
    pr = r.p_rx[sl] if r.p_rx.ndim == 1 else r.p_rx[sl].sum(1)
    ax.plot(t, pr * 1e6, color=COLS[2]); ax.set(ylabel="received (µW)")
    ax = fig.add_subplot(gs[2, :2])
    i = r.i_f[sl] if r.i_f.ndim == 1 else r.i_f[sl].sum(1)
    ir = r.i_rx[sl] if r.i_rx.ndim == 1 else r.i_rx[sl].sum(1)
    ax.plot(t, ir * 1e6, color=COLS[0], alpha=0.35, lw=0.7)
    ax.plot(t, i * 1e6, color=COLS[1])
    y = r.y[:n_show] if r.y.ndim == 1 else r.y[:n_show].sum(1)
    ax.plot((np.arange(y.size) + 0.5) / r.chip_rate * 1e6, y * 1e6, "k.", ms=4)
    ax.set(xlabel="time (µs)", ylabel=f"{sc.detector.kind} current (µA)")
    ax = fig.add_subplot(gs[:2, 2])
    if sch.family in ("ofdm", "subcarrier"):
        z = r.extra["points"]
        ax.plot(z.real, z.imag, ".", ms=2, alpha=0.5)
        ax.set(title="equalised symbols", aspect="equal")
    elif sch.channels == 3:
        z = S.chromaticity(np.maximum(r.y - r.y_train.min(0), 0))
        ax.plot(z[:, 0], z[:, 1], ".", ms=2); ax.set(title="colour chromaticity", aspect="equal")
    else:
        tr, x = r.eye(2, 300)
        for row in tr:
            ax.plot(x, row * 1e6, color=COLS[0], alpha=0.05, lw=0.8)
        ax.set(title="eye diagram", xlabel="chips", ylabel="µA")
    ax = fig.add_subplot(gs[2, 2]); ax.axis("off")
    txt = (f"peak Rx power {b['Pr_dBm']:.1f} dBm\nbackground {b['Pb_dBm']:.1f} dBm\nQ = {b['Q']:.2f}\n"
           f"BER Monte Carlo {r.ber:.2e} ({int(r.errors)}/{r.n_bits})\nBER closed form {b['ber']:.2e}")
    ax.text(0, 0.95, txt, va="top", family="monospace", fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_camera(sc, scheme, path, n=8):
    r = C.simulate(sc, scheme, n_frames=max(n, sc.n_frames))
    fig = plt.figure(figsize=(14, 6.4))
    gs = fig.add_gridspec(2, n)
    vmax = max(np.percentile(r.frames, 99.9), 1)
    for i in range(n):
        ax = fig.add_subplot(gs[0, i])
        f = r.frames[i].astype(float)
        ax.imshow(np.clip(f / vmax, 0, 1) if f.ndim == 3 else f, cmap="gray", vmin=0, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"frame {i}", fontsize=8)
    ax = fig.add_subplot(gs[1, : n // 2])
    if r.row_profile is not None:
        ax.imshow(r.row_profile, aspect="auto", cmap="viridis")
        ax.set(title="ROI row profile, every frame (rolling-shutter stripes)", xlabel="row", ylabel="frame")
    else:
        v = r.samples
        for j in range(min(v.shape[1], 16)):
            ax.plot(v[:, j] / max(np.abs(v).max(), 1e-12) + 1.1 * j, ".-", color=COLS[j % 10], ms=3)
        ax.set(title="ROI value per frame", xlabel="frame", yticks=[])
    ax = fig.add_subplot(gs[1, n // 2:])
    k = r.t < r.frame_t[min(2, len(r.frame_t) - 1)] + 1 / sc.camera.fps
    ax.plot(r.t[k] * 1e3, r.drive[k, 0, 0], color="#555", lw=0.8)
    for f in r.frame_t[:3]:
        if r.rolling:
            rows = np.nonzero(r.valid[: sc.camera.height])[0]
            if rows.size:
                ax.axvspan((f + rows[0] * r.row_time) * 1e3, (f + rows[-1] * r.row_time + r.exposure) * 1e3,
                           color=COLS[1], alpha=0.25)
        else:
            ax.axvspan(f * 1e3, (f + r.exposure) * 1e3, color=COLS[1], alpha=0.4)
    ax.set(title=f"{scheme}: drive and exposure ({sc.camera.shutter} shutter); BER {r.ber:.2g} "
                 f"({int(r.errors)}/{r.n_bits}), goodput {r.goodput:.4g} bit/s", xlabel="time (ms)")
    ax.title.set_fontsize(8)
    fig.suptitle(f"{sc.name}: {sc.description}", fontsize=9)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_cameras(path):
    names = [n for n, s in SCENARIOS.items() if s.receiver == "camera"]
    fig, axs = plt.subplots(2, len(names), figsize=(3.0 * len(names), 5.6))
    for i, n in enumerate(names):
        sc = SCENARIOS[n]
        r = C.simulate(sc, n_frames=max(sc.n_frames, 6))
        for j in range(2):
            f = r.frames[j + 2].astype(float)
            vmax = max(np.percentile(r.frames, 99.9), 1)
            axs[j, i].imshow(np.clip(f / vmax, 0, 1) if f.ndim == 3 else f, cmap="gray", vmin=0, vmax=vmax)
            axs[j, i].set_xticks([]); axs[j, i].set_yticks([])
        axs[0, i].set_title(f"{n}\n{sc.camera_scheme}, {sc.camera.shutter} shutter", fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_ber(path):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
    sc = SCENARIOS["Coastal, 10 m, LED + PIN"].copy()
    sc.bit_rate = 1e6
    names = ["OOK-NRZ", "Manchester", "4-PPM", "16-PPM", "8-DPIM", "4-PAM", "SIM-BPSK", "DCO-OFDM", "ACO-OFDM"]
    Rs = np.linspace(7, 16, 46)
    th = analysis.ber_curves(sc, names, Rs)
    Rm = np.linspace(10, 14, 5)
    for i, n in enumerate(names):
        ax[0].semilogy(Rs, np.maximum(th[n], 1e-10), color=COLS[i], label=n)
        if n in ("OOK-NRZ", "16-PPM", "DCO-OFDM", "8-DPIM"):
            mc = link.monte_carlo_ber(sc, n, ranges=Rm, n_bits=20000)
            ok = mc > 0
            ax[0].semilogy(Rm[ok], mc[ok], "o", color=COLS[i], ms=5)
    ax[0].axhline(3.8e-3, color="k", ls=":", lw=0.8)
    ax[0].set(title="BER vs range, Jerlov 3C, 1 W LED, PIN, 1 Mbit/s (dots: waveform Monte Carlo)",
              xlabel="range (m)", ylabel="BER", ylim=(1e-10, 1)); ax[0].legend(fontsize=7, ncol=2)
    ax[0].title.set_fontsize(8)
    sc2 = SCENARIOS["Clear ocean, 50 m, blue laser + APD"].copy()
    Rt = np.geomspace(10, 150, 24)
    thr = analysis.throughput_curves(sc2, ["OOK-NRZ", "4-PPM", "16-PPM", "4-PAM", "DCO-OFDM"], Rt, 1e-6)
    for i, (n, v) in enumerate(thr.items()):
        ok = v > 0
        ax[1].loglog(Rt[ok], v[ok], color=COLS[i], label=n)
    ax[1].set(title="Largest bit rate at BER 1e-6, Jerlov IB, 0.5 W laser + APD", xlabel="range (m)",
              ylabel="bit/s"); ax[1].legend(fontsize=7)
    ax[1].title.set_fontsize(8)
    for i, (n, bpc, dp, duty) in enumerate(analysis.efficiency_map()):
        if np.isfinite(dp):
            ax[2].plot(bpc, dp, "o", color=COLS[i % 10])
            ax[2].annotate(n, (bpc, dp), fontsize=6, xytext=(3, 2), textcoords="offset points")
    ax[2].set_xscale("log", base=2); ax[2].invert_yaxis()
    ax[2].set(title="Power vs bandwidth efficiency (thermal noise, BER 1e-6)", xlabel="bits per chip",
              ylabel="average optical power vs OOK (dB)")
    ax[2].title.set_fontsize(8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def gui_shots(out):
    """Screenshots of every workspace (needs PyQt5 + pyqtgraph; runs offscreen)."""
    import time
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5 import QtWidgets
    from ..gui.app import MainWindow
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = MainWindow(); w.resize(1560, 940); w.show()

    def settle(p):
        t = time.time()
        while time.time() - t < 120:
            app.processEvents()
            if not p.runner.running() and not p.runner.threads and not p._debounce.isActive():
                break
            time.sleep(0.01)
        for _ in range(8):
            app.processEvents()

    shots = [("High-rate OFDM, 15 m", 0, 2, "gui_channel"), ("Coastal, 10 m, LED + PIN", 1, 0, "gui_photodiode"),
             ("Coastal, 10 m, LED + PIN", 1, 1, "gui_eye"), ("High-rate OFDM, 15 m", 1, 2, "gui_ofdm"),
             ("Diver to diver, 4 m, rolling shutter", 2, 0, "gui_camera_frames"),
             ("Diver to diver, 4 m, rolling shutter", 2, 1, "gui_camera_timing"),
             ("Colour CSK camera, 6 m", 2, 0, "gui_camera_csk"), ("LED panel MIMO, 5 m", 2, 0, "gui_camera_mimo"),
             ("Clear ocean, 50 m, blue laser + APD", 3, 2, "gui_budget_map"),
             ("Coastal, 10 m, LED + PIN", 4, 0, "gui_ber"), ("Turbulent thermocline, 30 m", 1, 4, "gui_fading")]
    for sc, page, tab, name in shots:
        w.cb_sc.setCurrentText(sc)
        w.nav.setCurrentRow(page); p = w.page(); settle(p)
        p.tabs.setCurrentIndex(tab); app.processEvents(); p._do_refresh(); settle(p)
        w.grab().save(os.path.join(out, f"{name}.png"))
    w.toggle_theme()
    w.cb_sc.setCurrentText("Beacon, 20 m, UFSOOK"); w.nav.setCurrentRow(2); p = w.page(); settle(p)
    p.tabs.setCurrentIndex(1); p._do_refresh(); settle(p)
    w.grab().save(os.path.join(out, "gui_dark.png"))
    w.close()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    out = argv[0] if argv else "docs/figures"
    os.makedirs(out, exist_ok=True)
    fig_water(os.path.join(out, "water.png"))
    fig_channel(os.path.join(out, "channel.png"))
    fig_schemes(os.path.join(out, "schemes.png"))
    fig_photodiode(SCENARIOS["Coastal, 10 m, LED + PIN"].copy(range_m=12.3), "4-PPM",
                   os.path.join(out, "photodiode_ppm.png"))
    fig_photodiode(SCENARIOS["High-rate OFDM, 15 m"].copy(range_m=26.0), "DCO-OFDM",
                   os.path.join(out, "photodiode_ofdm.png"))
    fig_camera(SCENARIOS["Diver to diver, 4 m, rolling shutter"].copy(), "Manchester",
               os.path.join(out, "camera_rolling.png"))
    fig_camera(SCENARIOS["Beacon, 20 m, UFSOOK"].copy(), "UFSOOK", os.path.join(out, "camera_ufsook.png"))
    fig_cameras(os.path.join(out, "cameras.png"))
    fig_ber(os.path.join(out, "ber.png"))
    try:
        import PyQt5  # noqa: F401
        import pyqtgraph  # noqa: F401
    except ImportError:
        return 0
    if "--no-gui" not in argv:
        gui_shots(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
