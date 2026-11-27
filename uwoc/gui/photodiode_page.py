"""Photodiode link workspace: one frame end to end with intensity-versus-time and receiver analytics."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

from ..core import analysis, link, schemes as S
from ..core.channel import SOLAR
from ..core.devices import DETECTORS, TRANSMITTERS
from . import theme
from .base import Page
from .widgets import Empty, Slider, bar_ticks, combo, log_x, new_plot, pen, spin

EQ = [r"\bar n=\frac{\eta P\,\Delta t}{h\nu}+r_d\Delta t,\quad N\sim\mathrm{Poisson}(\bar n)",
      r"\sigma^2=2qM^2F\,(RP+I_d)B+i_n^2B,\quad F=kM+(1-k)\left(2-\frac{1}{M}\right)",
      r"Q=\frac{\mu_1-\mu_0}{\sigma_0+\sigma_1},\quad P_b^{OOK}\approx Q\!\left(Q\right),\quad P_s^{L\!-\!PPM}=1-\int\phi(z)\,\Phi^{L-1}\,dz",
      r"\kappa=1-\frac{1-e^{-2\pi BT_c}}{2\pi BT_c},\quad B^{-2}=B_{led}^{-2}+B_{ch}^{-2}+B_{rx}^{-2}"]


def scheme_combo():
    c = QtWidgets.QComboBox()
    for fam, names in S.FAMILIES.items():
        c.addItem(f"— {fam} —"); c.model().item(c.count() - 1).setEnabled(False)
        for n in names:
            c.addItem(n)
    return c


class PhotodiodePage(Page):
    title = "Photodiode link"

    def build(self):
        f, _ = self.add_card("Modulation")
        self.cb_scheme = scheme_combo(); f.addRow("Scheme", self.cb_scheme)
        self.sl_rate = Slider(1e3, 2e9, 1e6, log=True, decimals=0, suffix=" b/s"); f.addRow("Bit rate", self.sl_rate)
        self.sp_bits = spin(200, 200000, 3000, 500); f.addRow("Bits per frame", self.sp_bits)
        self.sp_sps = spin(4, 32, 8); f.addRow("Samples per chip", self.sp_sps)
        self.sp_seed = spin(0, 9999, 0); f.addRow("Seed", self.sp_seed)
        self.sp_show = spin(8, 2000, 64, 8); f.addRow("Chips shown", self.sp_show)
        f, _ = self.add_card("Transmitter and range")
        self.cb_tx = combo(["(scenario)"] + list(TRANSMITTERS)); f.addRow("Preset", self.cb_tx)
        self.sl_R = Slider(0.5, 400, 20, log=True, decimals=1, suffix=" m"); f.addRow("Range", self.sl_R)
        self.sl_P = Slider(1e-3, 100, 1, log=True, decimals=3, suffix=" W"); f.addRow("Peak optical power", self.sl_P)
        self.sl_led = Slider(1e5, 5e9, 20e6, log=True, decimals=0, suffix=" Hz"); f.addRow("Modulation bandwidth", self.sl_led)
        self.sl_er = Slider(0, 0.5, 0, decimals=3); f.addRow("Extinction (P0/P1)", self.sl_er)
        self.sl_div = Slider(0.05, 60, 5, log=True, decimals=2, suffix=" °"); f.addRow("Beam divergence", self.sl_div)
        f, _ = self.add_card("Receiver")
        self.cb_det = combo(list(DETECTORS)); f.addRow("Detector", self.cb_det)
        self.sl_gain = Slider(1, 1e7, 1, log=True, decimals=0); f.addRow("Gain", self.sl_gain)
        self.sl_bw = Slider(1e5, 5e9, 50e6, log=True, decimals=0, suffix=" Hz"); f.addRow("Bandwidth", self.sl_bw)
        self.sl_in = Slider(0.05, 50, 2, log=True, decimals=2, suffix=" pA/√Hz"); f.addRow("Amplifier noise", self.sl_in)
        self.sl_ap = Slider(1, 300, 50, log=True, decimals=1, suffix=" mm"); f.addRow("Aperture", self.sl_ap)
        self.sl_fov = Slider(0.5, 80, 10, log=True, decimals=1, suffix=" °"); f.addRow("Half FOV", self.sl_fov)
        self.sl_filt = Slider(1, 300, 30, log=True, decimals=1, suffix=" nm"); f.addRow("Optical filter", self.sl_filt)
        f, _ = self.add_card("Impairments")
        self.cb_turb = combo(["none", "lognormal", "gamma-gamma", "exponential"]); f.addRow("Turbulence", self.cb_turb)
        self.sl_si = Slider(0.001, 3, 0.1, log=True, decimals=3); f.addRow("Scintillation σ_I²", self.sl_si)
        self.sl_tc = Slider(1e-5, 1, 3e-3, log=True, decimals=5, suffix=" s"); f.addRow("Fade coherence", self.sl_tc)
        self.sl_jit = Slider(0, 2, 0, decimals=3, suffix=" m"); f.addRow("Pointing jitter σ", self.sl_jit)
        self.cb_sky = combo(list(SOLAR)); f.addRow("Ambient light", self.cb_sky)
        self.sl_depth = Slider(0, 1000, 20, decimals=0, suffix=" m"); f.addRow("Receiver depth", self.sl_depth)
        self.cb_look = combo(["up", "horizontal", "down"], "horizontal"); f.addRow("Looking", self.cb_look)
        self.eq = self.add_equations("Receiver model")
        self.eq.set_equations(EQ, "Integrate-and-dump per chip, levels calibrated on a training sequence, "
                                  "timing from a 62-chip m-sequence preamble.")
        self.cb_tx.currentTextChanged.connect(self._tx_preset)
        self.cb_det.currentTextChanged.connect(self._det_preset)
        self.watch(self.cb_scheme, self.sl_rate, self.sp_bits, self.sp_sps, self.sp_seed, self.sl_R, self.sl_P,
                   self.sl_led, self.sl_er, self.sl_div, self.cb_det, self.sl_gain, self.sl_bw, self.sl_in,
                   self.sl_ap, self.sl_fov, self.sl_filt, self.cb_turb, self.sl_si, self.sl_tc, self.sl_jit,
                   self.cb_sky, self.sl_depth, self.cb_look)
        self.sp_show.valueChanged.connect(lambda _: self.invalidate())
        self.add_view("Intensity vs time", self.draw_time)
        self.add_view("Eye diagram", self.draw_eye)
        self.add_view("Levels / constellation", self.draw_const)
        self.add_view("Spectrum", self.draw_spec)
        self.add_view("Fading", self.draw_fade)
        self.add_view("Noise breakdown", self.draw_noise)
        self.add_table(["quantity", "value", "unit"])

    # ---------------------------------------------------------------- presets / scenario
    def _tx_preset(self, name):
        if name not in TRANSMITTERS or self._loading:
            return
        t = TRANSMITTERS[name]
        self._loading = True
        self.sl_P.setValue(t.power_w); self.sl_led.setValue(t.bandwidth_hz); self.sl_er.setValue(t.extinction)
        self.sl_div.setValue(t.divergence_deg)
        self.state.scenario.tx = t.copy()
        self._loading = False
        self.schedule()

    def _det_preset(self, name):
        if self._loading:
            return
        d = DETECTORS[name]
        self._loading = True
        self.sl_gain.setValue(d.gain); self.sl_bw.setValue(d.bandwidth_hz); self.sl_in.setValue(d.noise_density * 1e12)
        self._loading = False

    def load_scenario(self, sc):
        self._loading = True
        i = self.cb_scheme.findText(sc.scheme); self.cb_scheme.setCurrentIndex(max(i, 1))
        self.sl_rate.setValue(sc.bit_rate)
        self.cb_tx.setCurrentIndex(0)
        self.sl_R.setValue(sc.range_m); self.sl_P.setValue(sc.tx.power_w); self.sl_led.setValue(sc.tx.bandwidth_hz)
        self.sl_er.setValue(sc.tx.extinction); self.sl_div.setValue(sc.tx.divergence_deg)
        d = sc.detector
        name = next((k for k, v in DETECTORS.items() if v.kind == d.kind), list(DETECTORS)[0])
        self.cb_det.setCurrentText(name)
        self.sl_gain.setValue(d.gain); self.sl_bw.setValue(d.bandwidth_hz); self.sl_in.setValue(d.noise_density * 1e12)
        self.sl_ap.setValue(sc.optics.aperture_mm); self.sl_fov.setValue(sc.optics.fov_deg)
        self.sl_filt.setValue(sc.optics.filter_nm)
        self.cb_turb.setCurrentText(sc.turbulence.model); self.sl_si.setValue(sc.turbulence.si)
        self.sl_tc.setValue(sc.turbulence.coherence_s); self.sl_jit.setValue(sc.pointing.jitter_m)
        self.cb_sky.setCurrentText(sc.ambient.sky); self.sl_depth.setValue(sc.ambient.depth_m)
        self.cb_look.setCurrentText(sc.ambient.look)
        self._loading = False
        self.schedule()

    def write(self, sc):
        if self.cb_scheme.currentText() in S.SCHEMES:
            sc.scheme = self.cb_scheme.currentText()
        sc.bit_rate = self.sl_rate.value(); sc.range_m = self.sl_R.value()
        sc.tx.power_w = self.sl_P.value(); sc.tx.bandwidth_hz = self.sl_led.value()
        sc.tx.extinction = self.sl_er.value(); sc.tx.divergence_deg = self.sl_div.value()
        d = DETECTORS[self.cb_det.currentText()].copy(gain=self.sl_gain.value(), bandwidth_hz=self.sl_bw.value(),
                                                      noise_density=self.sl_in.value() * 1e-12)
        if d.kind == "PIN":
            d.gain = 1.0
        sc.detector = d
        sc.optics.aperture_mm = self.sl_ap.value(); sc.optics.fov_deg = self.sl_fov.value()
        sc.optics.filter_nm = self.sl_filt.value()
        sc.turbulence.model = self.cb_turb.currentText(); sc.turbulence.si = self.sl_si.value()
        sc.turbulence.strength_mode = "si"; sc.turbulence.coherence_s = self.sl_tc.value()
        sc.pointing.jitter_m = self.sl_jit.value()
        sc.ambient.sky = self.cb_sky.currentText(); sc.ambient.depth_m = self.sl_depth.value()
        sc.ambient.look = self.cb_look.currentText()

    def job(self, heavy=False):
        sc = self.state.scenario
        self.write(sc)
        self.publish()
        sc = sc.copy()
        nb = self.sp_bits.value() * (5 if heavy else 1)
        sps, seed = self.sp_sps.value(), self.sp_seed.value()

        def run(ctx):
            sch = S.make(sc.scheme)
            nsamp = nb / sch.bits_per_chip * sps
            if nsamp > 4e6:
                raise ValueError(f"{nsamp:.3g} samples: lower the bits per frame or samples per chip")
            r = link.simulate(sc, sch, n_bits=nb, sps=sps, seed=seed)
            ctx.check()
            b = link.link_budget(sc, sch)
            f1, p1 = analysis.psd(r.drive, r.fs)
            f2, p2 = analysis.psd(r.i_f, r.fs)
            return dict(run=r, budget=b, sc=sc, psd=(f1, p1, f2, p2),
                        status=f"{sch.name}: {int(r.errors)} errors / {r.n_bits} bits, BER {r.ber:.3g} "
                               f"(closed form {b['ber']:.3g})")
        return run

    def after(self):
        r, b = self.result["run"], self.result["budget"]
        sch = r.scheme
        db = lambda x: 10 * np.log10(max(x, 1e-300))
        rows = [["scheme", sch.name, ""], ["bit rate", r.info["bit_rate"] / 1e6, "Mbit/s"],
                ["chip rate", r.chip_rate / 1e6, "Mchip/s"], ["bits per chip", sch.bits_per_chip, ""],
                ["duty (average / peak power)", sch.duty, ""],
                ["peak received power", b["Pr_dBm"], "dBm"], ["average received power", b["Pavg_dBm"], "dBm"],
                ["background power", b["Pb_dBm"], "dBm"], ["channel gain", db(b["gain"]), "dB"],
                ["Q factor (closed form)", b["Q"], ""],
                ["Q factor (training chips)", r.q_factor(), ""],
                ["BER Monte Carlo", r.ber, f"{int(r.errors)} / {r.n_bits}"],
                ["BER closed form", b["ber"], "with fading" if b["si"] > 0 or self.state.scenario.pointing.jitter_m > 0 else ""],
                ["ISI eye factor κ", b["kappa"], ""], ["rms delay spread", b["rms_delay"] * 1e9, "ns"],
                ["scintillation index", b["si"], ""]]
        flags = [{} for _ in rows]
        flags[11] = {1: "good" if r.ber < 1e-3 else ("warn" if r.ber < 3.8e-3 else "bad")}
        self.table.set_rows(rows, flags=flags)
        self.readout.setText(f"frame of {r.n_bits} bits · {r.t.size} samples at {r.fs / 1e6:.4g} MS/s · timing "
                             f"offset {r.offset} samples · scattered/ballistic "
                             f"{r.info['scattered'] / max(r.info['ballistic'], 1e-300):.3g}")

    def _need(self):
        if self.result is None:
            raise Empty("computing…")
        return self.result

    # ---------------------------------------------------------------- views
    def draw_time(self, glw):
        R = self._need(); r = R["run"]
        nshow = self.sp_show.value()
        start_chip = r.n_pre + r.n_train
        a = r.offset + start_chip * r.sps - 2 * r.sps
        b = min(a + (nshow + 4) * r.sps, r.t.size)
        a = max(a, 0)
        t = (r.t[a:b] - r.t[r.offset + start_chip * r.sps]) * 1e6
        sch = r.scheme
        p1 = new_plot(glw, 0, 0, f"Transmitted drive ({sch.name})", None, "drive x", legend=True, xvalues=False)
        if r.drive.ndim == 1:
            p1.plot(t, r.drive[a:b], pen=pen("tx", 1.8), name="LED drive (after LED response)")
        else:
            for j, c in enumerate("RGB"):
                p1.plot(t, r.drive[a:b, j], pen=pen(c, 1.8), name=f"{c} channel")
        p2 = new_plot(glw, 1, 0, "Received optical power (noiseless, incl. background and fading)", None, "µW",
                      xvalues=False)
        pr = r.p_rx[a:b] if r.p_rx.ndim == 1 else r.p_rx[a:b].sum(1)
        p2.plot(t, pr * 1e6, pen=pen("clean", 1.8))
        p3 = new_plot(glw, 2, 0, "Detector output: raw and after the receiver filter; chip decisions",
                      "time (µs)", "µA")
        i_raw = r.i_rx[a:b] if r.i_rx.ndim == 1 else r.i_rx[a:b].sum(1)
        i_f = r.i_f[a:b] if r.i_f.ndim == 1 else r.i_f[a:b].sum(1)
        p3.plot(t, i_raw * 1e6, pen=pen("rx", 0.8, alpha=110))
        p3.plot(t, i_f * 1e6, pen=pen("filt", 1.8))
        Tc = 1e6 / r.chip_rate
        n = min(nshow, r.y.shape[0])
        tc = (np.arange(n) + 0.5) * Tc
        y = r.y[:n] if r.y.ndim == 1 else r.y[:n].sum(1)
        p3.plot(tc, y * 1e6, pen=None, symbol="o", symbolSize=5, symbolBrush=theme.qcolor("c", 200))
        if "thr" in r.cal and r.y.ndim == 1 and sch.family in ("pulse",):
            p3.addItem(pg.InfiniteLine(r.cal["thr"] * 1e6, angle=0, pen=pen("thr", 1.2, Qt.DashLine)))
        for q in (p1, p2, p3):
            q.setXLink(p3)
            for k in range(0, n + 1, max(1, n // 64)):
                q.addItem(pg.InfiniteLine(k * Tc, angle=90, pen=pen("tx", 0.5, alpha=60)))
        # symbol errors (fixed-length schemes)
        if sch.fixed and r.rx_bits.size:
            k = sch.k
            nsym = min(r.tx_syms.size, r.rx_syms.size)
            bad = np.nonzero(r.tx_syms[:nsym] != r.rx_syms[:nsym])[0]
            bad = bad[bad * sch.n < n]
            if bad.size:
                p3.plot((bad * sch.n + sch.n / 2) * Tc, np.full(bad.size, float(np.max(i_f) * 1e6)), pen=None,
                        symbol="x", symbolSize=11, symbolPen=pen("err", 2))
        glw.ci.layout.setRowStretchFactor(3, 2)

    def draw_eye(self, glw):
        R = self._need(); r = R["run"]
        if r.scheme.family == "ofdm":
            raise Empty("OFDM has no eye: see Levels / constellation for the subcarrier symbols")
        tr, x = r.eye(2, 400)
        if tr.shape[0] == 0:
            raise Empty("not enough samples for an eye diagram")
        p = new_plot(glw, 0, 0, f"Eye diagram over two chips ({tr.shape[0]} traces)", "time (chips)", "µA")
        col = theme.qcolor("rx", 40)
        xs = np.tile(np.append(x, np.nan), tr.shape[0])
        ys = np.concatenate([np.append(row * 1e6, np.nan) for row in tr])
        p.plot(xs, ys, pen=pg.mkPen(col, width=1), connect="finite")
        if "thr" in r.cal and r.scheme.family == "pulse":
            p.addItem(pg.InfiniteLine(r.cal["thr"] * 1e6, angle=0, pen=pen("thr", 1.4, Qt.DashLine)))
        for k in (0, 1):
            p.addItem(pg.InfiniteLine(k + 0.5, angle=90, pen=pen("tx", 1, Qt.DotLine)))
        q = new_plot(glw, 0, 1, "Chip statistics", "count", None)
        y = r.y if r.y.ndim == 1 else r.y.sum(1)
        h, e = np.histogram(y * 1e6, bins=80)
        q.addItem(pg.BarGraphItem(x0=np.zeros(h.size), y=0.5 * (e[1:] + e[:-1]), height=np.diff(e), width=h,
                                  brush=theme.qcolor("filt", 140), pen=pg.mkPen(None)))
        q.setYLink(p)
        glw.ci.layout.setColumnStretchFactor(0, 4)
        glw.ci.layout.setColumnStretchFactor(1, 1)

    def draw_const(self, glw):
        R = self._need(); r = R["run"]; sch = r.scheme
        if sch.family in ("ofdm", "subcarrier") and "points" in r.extra:
            z = r.extra["points"]
            p = new_plot(glw, 0, 0, f"{sch.name}: equalised symbols ({z.size})", "in-phase", "quadrature")
            p.plot(z.real, z.imag, pen=None, symbol="o", symbolSize=3, symbolPen=None,
                   symbolBrush=theme.qcolor("rx", 120))
            ref = sch.pts if sch.family == "ofdm" else sch._points(np.arange(sch.order))
            p.plot(ref.real, ref.imag, pen=None, symbol="+", symbolSize=14, symbolPen=pen("err", 2))
            p.setAspectLocked(True)
            if sch.family == "ofdm" and "per_carrier" in r.extra:
                Y = r.extra["per_carrier"]
                X = sch.pts[r.tx_syms[: Y.size].reshape(Y.shape)] if r.tx_syms.size >= Y.size else None
                if X is not None:
                    snr = -10 * np.log10(np.maximum(np.mean(np.abs(Y - X) ** 2, 0), 1e-12))
                    q = new_plot(glw, 0, 1, "SNR per subcarrier", "subcarrier index", "dB")
                    q.plot(sch.data, snr, pen=pen("filt", 2), symbol="o", symbolSize=4)
            return
        if sch.channels == 3:
            p = new_plot(glw, 0, 0, f"{sch.name}: received colour chromaticity", "x (triangle)", "y (triangle)")
            V = S.chromaticity(np.eye(3))
            p.plot(np.append(V[:, 0], V[0, 0]), np.append(V[:, 1], V[0, 1]), pen=pen("tx", 1.4))
            z = S.chromaticity(np.maximum(r.y - np.min(r.y_train, 0), 0))
            p.plot(z[:, 0], z[:, 1], pen=None, symbol="o", symbolSize=4, symbolPen=None,
                   symbolBrush=theme.qcolor("rx", 110))
            m = S.chromaticity(np.maximum(r.cal["means"] - np.min(r.y_train, 0), 0))
            p.plot(m[:, 0], m[:, 1], pen=None, symbol="+", symbolSize=16, symbolPen=pen("err", 2))
            ref = S.chromaticity(sch.P)
            p.plot(ref[:, 0], ref[:, 1], pen=None, symbol="s", symbolSize=7, symbolBrush=None, symbolPen=pen("c", 1.4))
            p.setAspectLocked(True)
            return
        # pulse / level schemes: chip statistics by transmitted level
        p = new_plot(glw, 0, 0, "Chip statistics by transmitted level (training + payload)", "µA", "count",
                     legend=True)
        chips_tx, _ = sch.encode(r.tx_bits)
        y = r.y[: chips_tx.size]
        lv = np.unique(np.round(chips_tx, 6))
        for i, l in enumerate(lv):
            v = y[np.isclose(chips_tx[: y.size], l)] * 1e6
            if v.size < 2:
                continue
            h, e = np.histogram(v, bins=60)
            p.plot(e, h, stepMode="center", fillLevel=0, brush=theme.qcolor(f"c{i % 10}", 80),
                   pen=pen(f"c{i % 10}", 1.4), name=f"x = {l:.3g}")
        cal = r.cal
        for t in np.atleast_1d(cal.get("thresholds", cal.get("thr"))):
            p.addItem(pg.InfiniteLine(float(t) * 1e6, angle=90, pen=pen("thr", 1.4, Qt.DashLine)))

    def draw_spec(self, glw):
        R = self._need(); r = R["run"]
        f1, p1, f2, p2 = R["psd"]
        p = new_plot(glw, 0, 0, "Power spectral density (normalised)", "frequency (Hz)", "dB", legend=True)
        k1, k2 = f1 > 0, f2 > 0
        lx = log_x(f1[k1], p)
        p.plot(lx, 10 * np.log10(np.maximum(p1[k1] / p1[k1].max(), 1e-12)), pen=pen("tx", 1.8), name="transmitted drive")
        p.plot(np.log10(f2[k2]), 10 * np.log10(np.maximum(p2[k2] / p2[k2].max(), 1e-12)), pen=pen("filt", 1.8),
               name="received current")
        p.addItem(pg.InfiniteLine(np.log10(r.chip_rate), angle=90, pen=pen("thr", 1, Qt.DashLine)))
        p.setYRange(-60, 3)

    def draw_fade(self, glw):
        R = self._need(); r = R["run"]; sc = R["sc"]
        if np.allclose(r.fade, 1):
            raise Empty("no fading: choose a turbulence model or pointing jitter on the left")
        p = new_plot(glw, 0, 0, "Fading gain over the frame (turbulence × pointing)", "time (µs)", "h")
        step = max(1, r.fade.size // 4000)
        p.plot(r.t[::step] * 1e6, r.fade[::step], pen=pen("fade", 1.6))
        q = new_plot(glw, 1, 0, "Fading distribution", "h", "pdf", legend=True)
        h, e = np.histogram(r.fade, bins=60, density=True)
        q.plot(e, h, stepMode="center", fillLevel=0, brush=theme.qcolor("fade", 80), pen=pen("fade", 1.2),
               name="this frame")
        if sc.turbulence.model != "none":
            hh = np.linspace(1e-3, max(3, e[-1]), 400)
            si = r.info["si"]
            q.plot(hh, sc.turbulence.pdf(hh, si), pen=pen("c", 2), name=f"{sc.turbulence.model}, σ_I² = {si:.3g}")

    def draw_noise(self, glw):
        R = self._need(); st = R["budget"]["stats"]
        names = ["signal shot", "background shot", "dark", "amplifier", "ISI"]
        vals = [max(st["shot1"] - st["bg_shot"] - st["dark_shot"], 0), st["bg_shot"], st["dark_shot"], st["thermal"],
                st["isi"]]
        keys = ["shot", "background", "dark", "thermal", "isi"]
        p = new_plot(glw, 0, 0, "Noise variance per chip at the '1' level", None, "dB (A²)")
        y = 10 * np.log10(np.maximum(vals, 1e-60))
        lo = min(y) - 5
        for i, (v, k) in enumerate(zip(y, keys)):
            p.addItem(pg.BarGraphItem(x=[i], height=[v - lo], y0=[lo], width=0.6, brush=theme.qcolor(k, 200)))
        bar_ticks(p, names)
        sig = 20 * np.log10(max(st["m1"] - st["m0"], 1e-30))
        p.addItem(pg.InfiniteLine(sig - 20 * np.log10(2), angle=0, pen=pen("c", 1.4, Qt.DashLine),
                                  label="(Δμ/2)²", labelOpts=dict(position=0.95, color=theme.current["text"])))
