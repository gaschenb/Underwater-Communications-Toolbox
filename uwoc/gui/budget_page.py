"""Link budget workspace: waterfall, power versus range, BER map, detector comparison."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt

from ..core import link, schemes as S
from ..core.devices import DETECTORS
from . import theme
from .base import Page
from .photodiode_page import scheme_combo
from .widgets import Empty, Slider, bar_ticks, combo, image_cmap, log_x, new_plot, pen

EQ = [r"P_r\,[\mathrm{dBm}]=P_t+10\log_{10}\eta_{geo}-4.34\,cL+G_{sc}+10\log_{10}T_{opt}",
      r"\mathrm{margin}=P_r-P_{req}(\mathrm{BER}^*),\quad P_{req}:\ \mathrm{BER}(P_{req})=\mathrm{BER}^*"]
SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


class BudgetPage(Page):
    title = "Link budget"

    def build(self):
        f, _ = self.add_card("Link")
        self.cb_scheme = scheme_combo(); f.addRow("Scheme", self.cb_scheme)
        self.sl_R = Slider(0.5, 400, 20, log=True, decimals=1, suffix=" m"); f.addRow("Range", self.sl_R)
        self.sl_rate = Slider(1e3, 2e9, 1e6, log=True, decimals=0, suffix=" b/s"); f.addRow("Bit rate", self.sl_rate)
        self.cb_det = combo(list(DETECTORS)); f.addRow("Detector", self.cb_det)
        self.sl_P = Slider(1e-3, 100, 1, log=True, decimals=3, suffix=" W"); f.addRow("Peak optical power", self.sl_P)
        self.sl_tgt = Slider(1e-12, 1e-2, 1e-6, log=True, decimals=12); f.addRow("Target BER", self.sl_tgt)
        self.eq = self.add_equations("Budget")
        self.eq.set_equations(EQ, "Every term uses the scenario's water, geometry and receiver; the map and "
                                  "detector curves use the closed-form BER.")
        self.watch(self.cb_scheme, self.sl_R, self.sl_rate, self.cb_det, self.sl_P, self.sl_tgt)
        self.add_view("Budget", self.draw_budget)
        self.add_view("Power vs range", self.draw_range)
        self.add_view("BER map", self.draw_map)
        self.add_view("Detectors", self.draw_det)
        self.add_table(["term", "dB / value", "note"])

    def load_scenario(self, sc):
        self._loading = True
        i = self.cb_scheme.findText(sc.scheme); self.cb_scheme.setCurrentIndex(max(i, 1))
        self.sl_R.setValue(sc.range_m); self.sl_rate.setValue(sc.bit_rate); self.sl_P.setValue(sc.tx.power_w)
        name = next((k for k, v in DETECTORS.items() if v.kind == sc.detector.kind), list(DETECTORS)[0])
        self._det = sc.detector
        self.cb_det.setCurrentText(name)
        self._loading = False
        self.schedule()

    def write(self, sc):
        if self.cb_scheme.currentText() in S.SCHEMES:
            sc.scheme = self.cb_scheme.currentText()
        sc.range_m = self.sl_R.value(); sc.bit_rate = self.sl_rate.value(); sc.tx.power_w = self.sl_P.value()
        d = DETECTORS[self.cb_det.currentText()]
        if getattr(self, "_det", None) is None or self._det.kind != d.kind:
            self._det = d.copy()
        sc.detector = self._det.copy()

    def job(self, heavy=False):
        sc = self.state.scenario
        self.write(sc)
        self.publish()
        sc = sc.copy()
        tgt = self.sl_tgt.value()
        n = 28 if heavy else 18

        def run(ctx):
            sch = S.make(sc.scheme)
            b = link.link_budget(sc, sch)
            req = link.required_power_dbm(sc, sch, tgt)
            Rs = np.geomspace(max(sc.range_m / 20, 0.5), sc.range_m * 5, 50)
            pr = np.array([10 * np.log10(max(link.received(sc, L)["P1"], 1e-300) * 1e3) for L in Rs])
            pr_b = np.array([10 * np.log10(max(link.received(sc, L, ch=sc.channel(L, model="beer"))["P1"], 1e-300) * 1e3)
                             for L in Rs])
            ctx.check()
            # BER map over range x bit rate
            rr = np.geomspace(Rs[0], Rs[-1], n)
            rates = np.geomspace(1e3, 2e9, n)
            M = np.zeros((n, n))
            for i, L in enumerate(rr):
                ctx.progress(i, n)
                rx = link.received(sc, L)
                B = link.composite_bandwidth(sc, rx["ch"])
                for j, R in enumerate(rates):
                    M[i, j] = link.chip_ber(sc, sch, link.chip_stats(sc, sch, R, L, rx=rx, B=B))
            # detector comparison: required power vs bit rate, max range vs bit rate
            dr = np.geomspace(1e3, 2e9, 30)
            det = {}
            for name, d in DETECTORS.items():
                s2 = sc.copy(); s2.detector = d.copy()
                reqs = [link.required_power_dbm(s2, sch, tgt, R) for R in dr]
                det[name] = np.array(reqs)
            return dict(b=b, req=req, Rs=Rs, pr=pr, pr_b=pr_b, rr=rr, rates=rates, M=M, dr=dr, det=det, sc=sc,
                        tgt=tgt, sch=sch.name,
                        status=f"margin {b['Pr_dBm'] - req:+.1f} dB at BER {tgt:.0e}")
        return run

    def after(self):
        R = self.result; b = R["b"]
        rows = [[k, v, ""] for k, v in b["terms"]]
        rows += [["peak received power", b["Pr_dBm"], "dBm"], ["average received power", b["Pavg_dBm"], "dBm"],
                 ["background power", b["Pb_dBm"], "dBm"],
                 ["required peak power", R["req"], f"dBm for BER {R['tgt']:.0e}"],
                 ["link margin", b["Pr_dBm"] - R["req"], "dB"], ["BER at this range", b["ber"], ""],
                 ["attenuation lengths cL", b["attenuation_lengths"], ""],
                 ["channel 3 dB bandwidth", b["B_channel"] / 1e6 if np.isfinite(b["B_channel"]) else float("inf"), "MHz"]]
        flags = [{} for _ in rows]
        m = b["Pr_dBm"] - R["req"]
        flags[len(b["terms"]) + 4] = {1: "good" if m > 3 else ("warn" if m > 0 else "bad")}
        self.table.set_rows(rows, flags=flags)
        self.readout.setText(f"{R['sch']} at {R['sc'].bit_rate / 1e6:.4g} Mbit/s over {R['sc'].range_m:.1f} m "
                             f"of {R['sc'].water.name}")

    def _need(self):
        if self.result is None:
            raise Empty("computing…")
        return self.result

    def draw_budget(self, glw):
        R = self._need(); b = R["b"]
        p = new_plot(glw, 0, 0, "Link budget (cumulative, dBm)", None, "dBm")
        level = 0.0
        names = []
        for i, (k, v) in enumerate(b["terms"]):
            start = level if i else 0.0
            level = v if i == 0 else level + v
            lo, hi = (0, level) if i == 0 else (min(start, level), max(start, level))
            col = "c0" if i == 0 else ("c3" if v < 0 else "c2")
            p.addItem(pg.BarGraphItem(x=[i], y0=[lo], height=[hi - lo], width=0.6, brush=theme.qcolor(col, 200)))
            names.append({"Tx peak power": "Tx power", "Geometric loss": "geometry",
                          "Beam attenuation e^{-cL}": "e^(-cL)", "Scattered light recovered": "scatter gain",
                          "Receiver optics + filter": "optics"}.get(k, k))
        n = len(b["terms"])
        p.addItem(pg.BarGraphItem(x=[n], y0=[min(0, b["Pr_dBm"])], height=[abs(b["Pr_dBm"])], width=0.6,
                                  brush=theme.qcolor("c1", 200)))
        names.append("received")
        p.addItem(pg.InfiniteLine(R["req"], angle=0, pen=pen("thr", 1.6, Qt.DashLine),
                                  label=f"required for BER {R['tgt']:.0e}",
                                  labelOpts=dict(position=0.85, color=theme.current["text"])))
        if b["Pb_dBm"] > -200:
            p.addItem(pg.InfiniteLine(b["Pb_dBm"], angle=0, pen=pen("bg", 1.2, Qt.DotLine), label="background",
                                      labelOpts=dict(position=0.6, color=theme.current["text"])))
        bar_ticks(p, names)

    def draw_range(self, glw):
        R = self._need()
        p = new_plot(glw, 0, 0, "Peak received power versus range", "range (m)", "dBm", legend=True)
        lx = log_x(R["Rs"], p)
        p.plot(lx, R["pr"], pen=pen("effective", 2.4), name=f"{R['sc'].channel_model} channel")
        p.plot(lx, R["pr_b"], pen=pen("beer", 1.8, Qt.DashLine), name="ballistic only")
        p.addItem(pg.InfiniteLine(R["req"], angle=0, pen=pen("thr", 1.6, Qt.DashLine)))
        p.plot([lx[0], lx[-1]], [R["req"], R["req"]], pen=pen("thr", 1.6, Qt.DashLine),
               name=f"sensitivity, BER {R['tgt']:.0e}")
        p.addItem(pg.InfiniteLine(np.log10(R["sc"].range_m), angle=90, pen=pen("tx", 1, Qt.DotLine)))

    def draw_map(self, glw):
        R = self._need()
        p = new_plot(glw, 0, 0, f"log₁₀ BER over range and bit rate ({R['sch']})", "range (m)", "bit rate (b/s)")
        Z = np.log10(np.clip(R["M"], 1e-15, 0.5))
        img = pg.ImageItem(Z.T)
        lr, lt = np.log10(R["rr"]), np.log10(R["rates"])
        from PyQt5.QtCore import QRectF
        img.setRect(QRectF(lr[0], lt[0], lr[-1] - lr[0], lt[-1] - lt[0]))
        img.setLookupTable(image_cmap("viridis", reverse=True).getLookupTable(0, 1, 256))
        img.setLevels((-15, np.log10(0.5)))
        p.addItem(img)
        iso = pg.IsocurveItem(data=Z.T, level=np.log10(R["tgt"]), pen=pen("err", 2))
        iso.setParentItem(img)
        for ax, lo, hi in (("bottom", lr[0], lr[-1]), ("left", lt[0], lt[-1])):
            ticks = [(k, f"10{str(k).translate(SUP)}") for k in range(int(np.floor(lo)), int(np.ceil(hi)) + 1)]
            p.getAxis(ax).setTicks([ticks])
        sc = R["sc"]
        p.plot([np.log10(sc.range_m)], [np.log10(sc.bit_rate)], pen=None, symbol="+", symbolSize=16,
               symbolPen=pen("c", 2))
        cb = pg.ColorBarItem(values=(-15, np.log10(0.5)), colorMap=image_cmap("viridis", reverse=True),
                             interactive=False, width=12, label="log₁₀ BER")
        cb.setImageItem(img)
        glw.addItem(cb, row=1, col=1)

    def draw_det(self, glw):
        R = self._need()
        p = new_plot(glw, 0, 0, f"Receiver sensitivity (peak power for BER {R['tgt']:.0e}, {R['sch']})",
                     "bit rate (b/s)", "dBm", legend=True)
        lx = log_x(R["dr"], p)
        for name, v in R["det"].items():
            k = DETECTORS[name].kind
            ok = np.isfinite(v)
            p.plot(lx[ok], v[ok], pen=pen(k, 2.2), name=name)
        p.addItem(pg.InfiniteLine(R["b"]["Pr_dBm"], angle=0, pen=pen("c", 1.2, Qt.DotLine),
                                  label="received", labelOpts=dict(position=0.1, color=theme.current["text"])))
