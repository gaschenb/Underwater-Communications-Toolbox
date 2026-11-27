"""BER & throughput workspace: many schemes side by side, closed form and Monte Carlo."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt

from ..core import analysis, camera as C, link, schemes as S
from . import theme
from .base import Page
from .widgets import Empty, MethodList, Slider, combo, log_x, new_plot, pen, spin

SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _decades(p, lo, hi, axis="left"):
    p.getAxis(axis).setTicks([[(k, f"10{str(k).translate(SUP)}") for k in range(lo, hi + 1)]])


def _key(i):
    return f"c{i % 10}"


class BerPage(Page):
    title = "BER & throughput"

    def build(self):
        self.mc = {}
        f, v = self.add_card("Schemes")
        self.cb_rx = combo(["photodiode", "camera"]); f.addRow("Receiver", self.cb_rx)
        self.lst_pd = MethodList({n: S.make(n).describe() for n in S.SCHEMES})
        self.lst_cam = MethodList({n: n for n in C.CAMERA_SCHEMES})
        for i, l in enumerate((self.lst_pd, self.lst_cam)):
            for k in range(l.count()):
                l.item(k).setIcon(theme.swatch(_key(k)))
            l.fit_height()
            v.addWidget(l)
        self.lst_pd.set_checked(["OOK-NRZ", "Manchester", "4-PPM", "16-PPM", "4-PAM", "DCO-OFDM"])
        self.lst_cam.set_checked(["OOK-NRZ", "Manchester", "4-PPM", "4-PAM"])
        f, _ = self.add_card("Axes")
        self.cb_x = combo(["range", "received power"]); f.addRow("x axis", self.cb_x)
        self.sl_lo = Slider(0.3, 500, 2, log=True, decimals=1, suffix=" m"); f.addRow("Range from", self.sl_lo)
        self.sl_hi = Slider(0.3, 800, 80, log=True, decimals=1, suffix=" m"); f.addRow("Range to", self.sl_hi)
        self.sl_tgt = Slider(1e-12, 1e-2, 1e-6, log=True, decimals=12); f.addRow("Target BER", self.sl_tgt)
        self.ck_fade = QtWidgets.QCheckBox("Average over turbulence / jitter"); self.ck_fade.setChecked(True)
        f.addRow(self.ck_fade)
        f, _ = self.add_card("Monte Carlo (Ctrl+R)")
        self.sp_mcbits = spin(500, 500000, 6000, 1000); f.addRow("Bits per point", self.sp_mcbits)
        self.sp_mcpts = spin(3, 20, 7); f.addRow("Points", self.sp_mcpts)
        self.sp_frames = spin(4, 200, 24, 4); f.addRow("Camera frames", self.sp_frames)
        self.watch(self.cb_rx, self.lst_pd, self.lst_cam, self.cb_x, self.sl_lo, self.sl_hi, self.sl_tgt, self.ck_fade)
        self.cb_rx.currentIndexChanged.connect(self._vis)
        self.add_view("BER curves", self.draw_ber)
        self.add_view("Throughput", self.draw_thr)
        self.add_view("Power / bandwidth efficiency", self.draw_eff)
        self.add_view("Fading outage", self.draw_outage)
        self.add_table(["scheme", "bits/chip", "duty", "BER here", "max range (m)", "max rate here (b/s)"])
        self._vis()

    def _vis(self):
        cam = self.cb_rx.currentIndex() == 1
        self.lst_pd.setVisible(not cam); self.lst_cam.setVisible(cam)

    def load_scenario(self, sc):
        self._loading = True
        self.cb_rx.setCurrentText(sc.receiver)
        self.sl_lo.setValue(max(sc.range_m / 10, 0.3)); self.sl_hi.setValue(sc.range_m * 4)
        self.mc = {}
        self._loading = False
        self._vis()
        self.schedule()

    def names(self):
        if self.cb_rx.currentIndex() == 1:
            return [n for n in self.lst_cam.checked() if C.applicable(n, self.state.scenario.camera)]
        return self.lst_pd.checked()

    def job(self, heavy=False):
        sc = self.state.scenario.copy()
        names = self.names()
        cam = self.cb_rx.currentIndex() == 1
        lo, hi = sorted((self.sl_lo.value(), self.sl_hi.value()))
        tgt, faded = self.sl_tgt.value(), self.ck_fade.isChecked()
        xmode = self.cb_x.currentText()
        mcbits, mcpts, nfr = self.sp_mcbits.value(), self.sp_mcpts.value(), self.sp_frames.value()
        if not heavy:
            keep = {k: v for k, v in self.mc.items() if v.get("key") == (cam, xmode, lo, hi)}
            self.mc = keep

        def run(ctx):
            Rs = np.geomspace(lo, hi, 60)
            out = dict(names=names, cam=cam, Rs=Rs, xmode=xmode, tgt=tgt, sc=sc, key=(cam, xmode, lo, hi))
            if cam:
                out["ber"] = analysis.camera_ber_vs_range(sc, names, Rs)
                out["thr"] = {n: np.array([C.throughput(sc.copy(range_m=float(L)), n) for L in Rs[::4]]) for n in names}
                out["Rt"] = Rs[::4]
            else:
                if xmode == "range":
                    out["ber"] = {}
                    for i, n in enumerate(names):
                        ctx.progress(i, len(names))
                        out["ber"][n] = link.ber_vs_range(sc, S.make(n), Rs, faded=faded)
                else:
                    p0 = 10 * np.log10(max(link.received(sc)["P1"], 1e-300) * 1e3)
                    out["P"] = np.linspace(p0 - 25, p0 + 15, 60)
                    out["ber"] = analysis.ber_vs_power(sc, names, out["P"])
                Rt = np.geomspace(lo, hi, 24)
                out["Rt"] = Rt
                out["thr"] = {n: link.max_rate_vs_range(sc, S.make(n), Rt, tgt) for n in names}
                out["eff"] = analysis.efficiency_map(target=tgt)
            si = sc.turbulence.sigma_I2(sc.wavelength, sc.range_m, sc.optics.aperture_mm / 1e3) \
                if sc.turbulence.model != "none" else 0.0
            out["si"] = si
            rows = []
            for n in names:
                if cam:
                    rows.append([n, np.nan, np.nan, C.ber_theory(sc, n), np.nan, C.throughput(sc, n)])
                else:
                    s = S.make(n)
                    rows.append([n, s.bits_per_chip, s.duty, link.ber_theory(sc, s, faded=faded),
                                 link.max_range(sc, s, tgt), link.max_rate_vs_range(sc, s, [sc.range_m], tgt)[0]])
            out["rows"] = rows
            if heavy:
                mc = {}
                xs = np.geomspace(lo, hi, mcpts) if (cam or xmode == "range") else np.linspace(out["P"][0], out["P"][-1], mcpts)
                for i, n in enumerate(names):
                    ctx.progress(i, len(names))
                    if cam:
                        b, g = analysis.camera_mc_vs_range(sc, n, xs, n_frames=nfr, ctx=ctx)
                        mc[n] = dict(x=xs, ber=b, key=out["key"])
                    elif xmode == "range":
                        mc[n] = dict(x=xs, ber=link.monte_carlo_ber(sc, n, ranges=xs, n_bits=mcbits, ctx=ctx),
                                     key=out["key"])
                    else:
                        mc[n] = dict(x=xs, ber=analysis.mc_vs_power(sc, n, xs, n_bits=mcbits, ctx=ctx), key=out["key"])
                out["mc"] = mc
            out["status"] = f"{len(names)} schemes" + (" with Monte Carlo" if heavy else " (Ctrl+R adds Monte Carlo)")
            return out
        return run

    def after(self):
        R = self.result
        if "mc" in R:
            self.mc.update(R["mc"])
        keys = [_key(self._index(n)) for n in R["names"]]
        self.table.set_rows(R["rows"])
        for i, k in enumerate(keys):
            it = self.table.item(i, 0)
            if it is not None:
                it.setIcon(theme.swatch(k))
        sc = R["sc"]
        self.readout.setText(f"{sc.water.name} · {sc.tx.wavelength_nm:.0f} nm · "
                             f"{'camera ' + sc.camera.name if R['cam'] else sc.detector.kind} · "
                             f"{sc.bit_rate / 1e6:.4g} Mbit/s · σ_I² = {R['si']:.3g}")

    def _index(self, n):
        lst = self.lst_cam if self.result and self.result["cam"] else self.lst_pd
        for k in range(lst.count()):
            if lst.item(k).data(Qt.UserRole) == n:
                return k
        return 0

    def _need(self):
        if self.result is None:
            raise Empty("computing…")
        if not self.result["names"]:
            raise Empty("tick one or more schemes on the left")
        return self.result

    def draw_ber(self, glw):
        R = self._need()
        xr = R["xmode"] == "range" or R["cam"]
        p = new_plot(glw, 0, 0, "Bit error rate" + (" versus range" if xr else " versus peak received power"),
                     "range (m)" if xr else "peak received power (dBm)", "BER", legend=True)
        x = log_x(R["Rs"], p) if xr else R["P"]
        for n in R["names"]:
            if n not in R["ber"]:
                continue
            k = _key(self._index(n))
            y = np.log10(np.clip(R["ber"][n], 1e-15, 0.5))
            p.plot(x, y, pen=pen(k, 2.2), name=n)
            if n in self.mc:
                m = self.mc[n]
                ok = np.isfinite(m["ber"]) & (m["ber"] > 0)
                mx = np.log10(m["x"]) if xr else m["x"]
                p.plot(mx[ok], np.log10(m["ber"][ok]), pen=None, symbol="o", symbolSize=8, symbolBrush=theme.qcolor(k),
                       symbolPen=pen("c", 0.8))
                z = np.isfinite(m["ber"]) & (m["ber"] == 0)
                if z.any():
                    p.plot(mx[z], np.full(z.sum(), -6.5), pen=None, symbol="t", symbolSize=8,
                           symbolBrush=theme.qcolor(k))
        p.addItem(pg.InfiniteLine(np.log10(R["tgt"]), angle=0, pen=pen("thr", 1.2, Qt.DashLine)))
        p.addItem(pg.InfiniteLine(np.log10(3.8e-3), angle=0, pen=pen("tx", 1, Qt.DotLine), label="HD-FEC 3.8e-3",
                                  labelOpts=dict(position=0.08, color=theme.current["text"])))
        p.setYRange(-12, 0)
        _decades(p, -12, 0)
        if xr:
            p.addItem(pg.InfiniteLine(np.log10(R["sc"].range_m), angle=90, pen=pen("tx", 1, Qt.DotLine)))

    def draw_thr(self, glw):
        R = self._need()
        title = (f"Largest bit rate meeting BER {R['tgt']:.0e} (closed form)" if not R["cam"]
                 else "Camera link raw throughput (erasures removed)")
        p = new_plot(glw, 0, 0, title, "range (m)", "bit/s", legend=True)
        lx = log_x(R["Rt"], p)
        lo, hi = 20, -20
        for n in R["names"]:
            v = R["thr"].get(n)
            if v is None:
                continue
            ok = v > 0
            if ok.any():
                y = np.log10(v[ok]); lo, hi = min(lo, int(np.floor(y.min()))), max(hi, int(np.ceil(y.max())))
                p.plot(lx[ok], y, pen=pen(_key(self._index(n)), 2.2), name=n)
        if hi >= lo:
            _decades(p, lo, hi)

    def draw_eff(self, glw):
        R = self._need()
        if R["cam"]:
            raise Empty("power / bandwidth efficiency is shown for photodiode schemes")
        p = new_plot(glw, 0, 0, f"Power versus bandwidth efficiency (thermal-noise limited, BER {R['tgt']:.0e})",
                     "bandwidth efficiency (bit per chip)", "average optical power vs OOK (dB)")
        for i, (n, bpc, dp, duty) in enumerate(R["eff"]):
            if not np.isfinite(dp):
                continue
            k = _key(i)
            sel = n in R["names"]
            p.plot([np.log2(bpc)], [dp], pen=None, symbol="o", symbolSize=11 if sel else 7,
                   symbolBrush=theme.qcolor(k, 255 if sel else 120), symbolPen=pen("c", 1.0 if sel else 0.3))
            t = pg.TextItem(n, color=theme.current["plot_fg"] if sel else theme.current["muted"], anchor=(0, 0.5))
            t.setPos(np.log2(bpc) + 0.05, dp); p.addItem(t)
        p.getAxis("bottom").setTicks([[(k, f"{2.0 ** k:g}") for k in range(-4, 3)]])
        p.invertY(True)

    def draw_outage(self, glw):
        R = self._need(); sc = R["sc"]
        from ..core.channel import Turbulence
        p = new_plot(glw, 0, 0, "Fade outage probability versus link margin", "margin (dB)", "P(outage)", legend=True)
        m = np.linspace(0, 25, 120)
        si = R["si"] if R["si"] > 0 else 0.3
        for i, mod in enumerate(("lognormal", "gamma-gamma", "exponential")):
            t = Turbulence(mod, si=si)
            v = t.outage(m, si)
            p.plot(m, np.log10(np.clip(v, 1e-12, 1)), pen=pen(f"c{i}", 2.2), name=f"{mod}, σ_I² = {si:.3g}")
        _decades(p, -12, 0)
        p.setYRange(-8, 0)
