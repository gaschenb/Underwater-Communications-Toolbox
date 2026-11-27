"""Water & channel workspace: water IOPs, phase functions, attenuation, impulse response, Monte Carlo."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt

from ..core import montecarlo, water as W
from ..core.channel import Geometry, channel
from . import theme
from .base import Page
from .widgets import Empty, Slider, combo, log_ticks, log_x, new_plot, pen, spin

EQ = [r"c(\lambda)=a(\lambda)+b(\lambda),\quad \omega_0=b/c",
      r"P_r=P_t\,\eta_{geo}\,e^{-cL}+P_t\,\eta_{sc}\left[e^{-(a+(1-f)b)L}-e^{-cL}\right]",
      r"\eta_{geo}=\mathrm{erf}^2\!\left(\sqrt{\pi/2}\,R/w\right),\ \ w=\sqrt{w_0^2+(L\tan\theta)^2}",
      r"\bar\tau=\frac{P\,b\,\theta_1^2L^2}{4v},\quad \Delta w=w\,\omega_0\,\tilde\beta(\theta_s)\frac{A\cos\theta_r}{r^2}e^{-cr}"]
WATER_MODES = ("Water type", "Custom a, b", "Bio-optical mix")
SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def water_controls(page, fl):
    """Water selection widgets shared with other pages; returns a reader/loader pair."""
    page.cb_wmode = combo(WATER_MODES); fl.addRow("Water", page.cb_wmode)
    page.cb_water = combo(list(W.WATER_TYPES), "Jerlov II"); fl.addRow("Type", page.cb_water)
    page.sl_a = Slider(0.01, 5, 0.1, log=True, decimals=3, suffix=" 1/m"); fl.addRow("Absorption a", page.sl_a)
    page.sl_b = Slider(0.001, 10, 0.2, log=True, decimals=3, suffix=" 1/m"); fl.addRow("Scattering b", page.sl_b)
    page.sl_chl = Slider(0.01, 30, 0.3, log=True, decimals=2, suffix=" mg/m³"); fl.addRow("Chlorophyll", page.sl_chl)
    page.sl_cdom = Slider(0.001, 3, 0.03, log=True, decimals=3, suffix=" 1/m"); fl.addRow("CDOM a(440)", page.sl_cdom)
    page.sl_min = Slider(0.0, 5, 0.0, decimals=2, suffix=" 1/m"); fl.addRow("Minerals b(550)", page.sl_min)
    page.cb_phase = combo(W.PHASE_FUNCTIONS); fl.addRow("Phase function", page.cb_phase)
    page.sl_B = Slider(0.002, 0.05, W.B_PETZOLD, log=True, decimals=4); fl.addRow("Backscatter fraction", page.sl_B)
    page.sl_g = Slider(0.5, 0.99, 0.924, decimals=3); fl.addRow("HG asymmetry g", page.sl_g)

    def vis():
        m = page.cb_wmode.currentIndex()
        for w, on in ((page.cb_water, m == 0), (page.sl_a, m == 1), (page.sl_b, m == 1), (page.sl_chl, m == 2),
                      (page.sl_cdom, m == 2), (page.sl_min, m == 2)):
            w.setVisible(on); lab = fl.labelForField(w)
            if lab is not None:
                lab.setVisible(on)
        hg = page.cb_phase.currentText() == "Henyey-Greenstein"
        for w, on in ((page.sl_B, not hg), (page.sl_g, hg)):
            w.setVisible(on); lab = fl.labelForField(w)
            if lab is not None:
                lab.setVisible(on)
    page.cb_wmode.currentIndexChanged.connect(vis)
    page.cb_phase.currentIndexChanged.connect(vis)
    vis()
    page.watch(page.cb_wmode, page.cb_water, page.sl_a, page.sl_b, page.sl_chl, page.sl_cdom, page.sl_min,
               page.cb_phase, page.sl_B, page.sl_g)

    def read(lam):
        m = page.cb_wmode.currentIndex()
        if m == 0:
            w = W.water_from_name(page.cb_water.currentText())
        elif m == 1:
            w = W.custom_water(page.sl_a.value(), page.sl_b.value(), lam_ref=lam)
        else:
            w = W.Water(name="bio-optical mix", kind="bio", chl=page.sl_chl.value(), ag440=page.sl_cdom.value(),
                        bmin550=page.sl_min.value(), description="user bio-optical mix")
        return w.copy(phase=page.cb_phase.currentText(), bb_ratio=page.sl_B.value(), g=page.sl_g.value())

    def load(w, lam):
        if w.kind == "custom":
            page.cb_wmode.setCurrentIndex(1)
            page.sl_a.setValue(float(w.a(lam))); page.sl_b.setValue(float(w.b(lam)))
        elif w.name in W.WATER_TYPES:
            page.cb_wmode.setCurrentIndex(0); page.cb_water.setCurrentText(w.name)
        else:
            page.cb_wmode.setCurrentIndex(2)
            page.sl_chl.setValue(w.chl); page.sl_cdom.setValue(w.ag440); page.sl_min.setValue(w.bmin550)
        page.cb_phase.setCurrentText(w.phase); page.sl_B.setValue(w.bb_ratio); page.sl_g.setValue(w.g)
        vis()
    return read, load


class ChannelPage(Page):
    title = "Water & channel"

    def build(self):
        f, _ = self.add_card("Water")
        self.read_water, self.load_water = water_controls(self, f)
        f, _ = self.add_card("Geometry")
        self.sl_R = Slider(0.5, 300, 20, log=True, decimals=1, suffix=" m"); f.addRow("Range", self.sl_R)
        self.sl_lam = Slider(400, 650, 470, decimals=0, suffix=" nm"); f.addRow("Wavelength", self.sl_lam)
        self.sl_div = Slider(0.05, 60, 5, log=True, decimals=2, suffix=" °"); f.addRow("Beam divergence", self.sl_div)
        self.sl_ap = Slider(1, 300, 50, log=True, decimals=1, suffix=" mm"); f.addRow("Rx aperture", self.sl_ap)
        self.sl_fov = Slider(0.5, 80, 10, log=True, decimals=1, suffix=" °"); f.addRow("Rx half FOV", self.sl_fov)
        self.sl_off = Slider(0, 5, 0, decimals=2, suffix=" m"); f.addRow("Pointing offset", self.sl_off)
        f, _ = self.add_card("Channel model")
        self.cb_model = combo(["effective", "beer", "mc"], "effective"); f.addRow("Model", self.cb_model)
        self.cb_model.setItemData(0, "Beer–Lambert + small-angle forward-scatter recovery", Qt.ToolTipRole)
        self.cb_model.setItemData(1, "unscattered light only, e^{-cL}", Qt.ToolTipRole)
        self.cb_model.setItemData(2, "Monte Carlo photon transport", Qt.ToolTipRole)
        self.sp_ph = spin(2000, 400000, 20000, 2000); f.addRow("MC photons", self.sp_ph)
        self.sp_seed = spin(0, 9999, 0); f.addRow("Seed", self.sp_seed)
        self.eq = self.add_equations("Model")
        self.eq.set_equations(EQ, "Ballistic light follows Beer–Lambert; forward-scattered light inside a "
                                  "10° lobe is partly recovered, broadening the beam and the pulse. The Monte "
                                  "Carlo uses forced collisions and semi-analytic local estimates.")
        self.watch(self.sl_R, self.sl_lam, self.sl_div, self.sl_ap, self.sl_fov, self.sl_off, self.cb_model,
                   self.sp_ph, self.sp_seed)
        self.add_view("IOP spectra", self.draw_iop)
        self.add_view("Attenuation vs range", self.draw_range)
        self.add_view("Impulse response", self.draw_cir)
        self.add_view("Photon paths", self.draw_paths)
        self.add_view("Phase function", self.draw_phase)
        self.add_table(["quantity", "value", "unit"])

    def load_scenario(self, sc):
        self._loading = True
        self.load_water(sc.water, sc.wavelength)
        self.sl_R.setValue(sc.range_m); self.sl_lam.setValue(sc.wavelength)
        self.sl_div.setValue(sc.tx.divergence_deg); self.sl_ap.setValue(sc.optics.aperture_mm)
        self.sl_fov.setValue(sc.optics.fov_deg); self.sl_off.setValue(sc.pointing.offset_m)
        self.cb_model.setCurrentText(sc.channel_model); self.sp_ph.setValue(sc.mc_photons)
        self._loading = False
        self.schedule()

    def write(self, sc):
        lam = self.sl_lam.value()
        sc.water = self.read_water(lam)
        sc.range_m = self.sl_R.value(); sc.tx.wavelength_nm = lam
        sc.tx.divergence_deg = self.sl_div.value(); sc.optics.aperture_mm = self.sl_ap.value()
        sc.optics.fov_deg = self.sl_fov.value(); sc.pointing.offset_m = self.sl_off.value()
        sc.channel_model = self.cb_model.currentText(); sc.mc_photons = self.sp_ph.value()

    def job(self, heavy=False):
        sc = self.state.scenario
        self.write(sc)
        self.publish()
        sc = sc.copy()
        seed = self.sp_seed.value()

        def run(ctx):
            lam, L = sc.wavelength, sc.range_m
            w = sc.water
            out = dict(lam=lam, L=L, water=w.name, iop=w.iops(lam))
            ll = np.arange(400, 701, 2.0)
            out["spec"] = dict(lam=ll, a=w.a(ll), b=w.b(ll), c=w.c(ll), Kd=w.Kd(ll))
            out["jerlov"] = {k: v.Kd(ll) for k, v in W.WATER_TYPES.items() if k.startswith("Jerlov")}
            geo = sc.geometry()
            model = sc.channel_model
            ch = channel(w, geo, model, sc.pointing.offset_m, mc_photons=sc.mc_photons * (4 if heavy else 1),
                         seed=seed, progress=lambda i, n: ctx.progress(i, n))
            out["ch"] = ch
            ctx.check()
            # attenuation vs range (analytic models on a dense grid, MC on a few points)
            Rs = np.geomspace(max(0.5, L / 20), max(L * 4, 5), 40)
            T = sc.optics.optics_T * sc.optics.filter_T
            dbm = lambda g: 10 * np.log10(np.maximum(np.asarray(g) * sc.tx.power_w * T, 1e-300) * 1e3)
            out["Rs"] = Rs
            for m in ("beer", "effective"):
                out[m] = dbm([channel(w, Geometry(**{**geo.__dict__, "range_m": r}), m, sc.pointing.offset_m).gain
                              for r in Rs])
            if model == "mc" or heavy:
                Rm = np.geomspace(Rs[0], Rs[-1], 7)
                g = []
                for i, r in enumerate(Rm):
                    ctx.check()
                    g.append(channel(w, Geometry(**{**geo.__dict__, "range_m": r}), "mc", sc.pointing.offset_m,
                                     mc_photons=max(sc.mc_photons // 4, 3000), seed=seed + i).gain)
                out["Rm"], out["mc"] = Rm, dbm(g)
            # sample photon paths (small MC if the main model is analytic)
            mc = ch.mc if ch.mc is not None else montecarlo.simulate(
                float(w.c(lam)), float(w.b(lam)), w.phase_function(lam), L, np.radians(geo.divergence_deg),
                geo.waist_mm / 1e3, geo.aperture_radius, np.radians(geo.fov_deg), (sc.pointing.offset_m, 0),
                n_photons=3000, seed=seed, n_paths=80)
            out["paths"] = mc.paths
            out["angles"], out["angle_w"] = mc.angles, mc.angle_w
            out["pf"] = {k: W.phase_function(k, w.bb_ratio, w.g) for k in W.PHASE_FUNCTIONS}
            out["pf_cur"] = w.phase_function(lam)
            out["B3"] = ch.bandwidth_3db()
            f = np.geomspace(1e5, 1e10, 300)
            out["f"], out["H"] = f, ch.frequency_response(f)
            out["status"] = f"{w.name}, c = {out['iop']['c']:.3f} 1/m, {model} gain {10 * np.log10(max(ch.gain, 1e-300)):.1f} dB"
            return out
        return run

    def after(self):
        r = self.result
        ch, io = r["ch"], r["iop"]
        rows = [["absorption a", io["a"], "1/m"], ["scattering b", io["b"], "1/m"],
                ["beam attenuation c", io["c"], "1/m"], ["backscattering b_b", io["bb"], "1/m"],
                ["diffuse attenuation K_d", io["Kd"], "1/m"], ["single-scattering albedo", io["albedo"], ""],
                ["attenuation lengths cL", io["c"] * r["L"], ""],
                ["ballistic gain", 10 * np.log10(max(ch.gain_ballistic, 1e-300)), "dB"],
                ["scattered gain", 10 * np.log10(max(ch.gain_scattered, 1e-300)), "dB"],
                ["total gain", 10 * np.log10(max(ch.gain, 1e-300)), "dB"],
                ["mean excess delay", ch.mean_delay * 1e9, "ns"], ["rms delay spread", ch.rms_delay * 1e9, "ns"],
                ["3 dB channel bandwidth", r["B3"] / 1e6 if np.isfinite(r["B3"]) else float("inf"), "MHz"]]
        self.table.set_rows(rows)
        self.readout.setText(f"{r['water']} at {r['lam']:.0f} nm, {r['L']:.1f} m · phase function g = "
                             f"{r['pf_cur'].g:.3f}, B = {r['pf_cur'].B:.4f} · best wavelength in this water "
                             f"{self.state.scenario.water.best_wavelength():.0f} nm")

    def _need(self):
        if self.result is None:
            raise Empty("computing…")
        return self.result

    # ---------------------------------------------------------------- views
    def draw_iop(self, glw):
        r = self._need(); s = r["spec"]
        p = new_plot(glw, 0, 0, f"Inherent optical properties: {r['water']}", "wavelength (nm)", "1/m", legend=True)
        allv = []
        for k, nm in (("a", "absorption a"), ("b", "scattering b"), ("c", "attenuation c"), ("Kd", "K_d")):
            v = np.log10(np.maximum(s[k], 1e-6)); allv.append(v)
            p.plot(s["lam"], v, pen=pen(k, 2.2), name=nm)
        allv = np.concatenate(allv)
        p.getAxis("left").setTicks(log_ticks(allv.min(), allv.max()))
        p.addItem(pg.InfiniteLine(r["lam"], angle=90, pen=pen("tx", 1.2, Qt.DashLine)))
        q = new_plot(glw, 1, 0, "Jerlov diffuse attenuation K_d(λ)", "wavelength (nm)", "1/m")
        q.addLegend(offset=(10, 10), labelTextSize="8pt", colCount=2)
        allv = []
        for i, (k, v) in enumerate(r["jerlov"].items()):
            lv = np.log10(v); allv.append(lv)
            q.plot(s["lam"], lv, pen=pen(f"c{i % 10}", 1.6), name=k.replace("Jerlov ", ""))
        allv = np.concatenate(allv)
        q.getAxis("left").setTicks(log_ticks(allv.min(), allv.max()))
        q.addItem(pg.InfiniteLine(r["lam"], angle=90, pen=pen("tx", 1.2, Qt.DashLine)))

    def draw_range(self, glw):
        r = self._need()
        p = new_plot(glw, 0, 0, "Peak received optical power versus range", "range (m)", "dBm", legend=True)
        lx = log_x(r["Rs"], p)
        p.plot(lx, r["beer"], pen=pen("beer", 2.2), name="Beer–Lambert (ballistic)")
        p.plot(lx, r["effective"], pen=pen("effective", 2.2), name="with forward-scatter recovery")
        if "mc" in r:
            p.plot(np.log10(r["Rm"]), r["mc"], pen=None, symbol="o", symbolSize=8, symbolBrush=theme.qcolor("mc"),
                   name="Monte Carlo")
        p.addItem(pg.InfiniteLine(np.log10(r["L"]), angle=90, pen=pen("tx", 1.2, Qt.DashLine)))

    def draw_cir(self, glw):
        r = self._need(); ch = r["ch"]
        p = new_plot(glw, 0, 0, f"Channel impulse response ({ch.model}), ballistic tap at τ = 0",
                     "excess delay (ns)", "h(t)  (1/ns, normalised)", legend=True)
        if ch.gain_scattered > 0 and ch.t.size > 1:
            p.plot(ch.t * 1e9, ch.h / 1e9 / max(ch.gain, 1e-300), pen=pen("scattered", 2.0), name="scattered light")
        p.plot([0, 0], [0, ch.gain_ballistic / max(ch.gain, 1e-300)], pen=pen("ballistic", 3.0), name="ballistic fraction")
        q = new_plot(glw, 1, 0, "Frequency response", "frequency (Hz)", "|H(f)| (dB)")
        lx = log_x(r["f"], q)
        q.plot(lx, 20 * np.log10(np.maximum(np.abs(r["H"]), 1e-6)), pen=pen("effective", 2.2))
        q.addItem(pg.InfiniteLine(-3, angle=0, pen=pen("tx", 1, Qt.DashLine)))
        q.setYRange(-30, 2)

    def draw_paths(self, glw):
        r = self._need()
        p = new_plot(glw, 0, 0, "Sample photon trajectories (side view)", "z (m)", "x (m)")
        for i, tr in enumerate(r["paths"][:80]):
            if tr.shape[0] > 1:
                p.plot(tr[:, 2], tr[:, 0], pen=pen(f"c{i % 10}", 1.0, alpha=150))
        L = r["L"]; R = self.state.scenario.optics.aperture_mm / 2000
        off = self.state.scenario.pointing.offset_m
        p.plot([L, L], [off - R, off + R], pen=pen("err", 4.0))
        p.plot([0], [0], pen=None, symbol="t1", symbolSize=12, symbolBrush=theme.qcolor("tx"))
        w = max(0.25 * L, 4 * R, abs(off) + 2 * R)
        p.setXRange(-0.05 * L, 1.05 * L); p.setYRange(-w, w)
        q = new_plot(glw, 1, 0, "Arrival angles of scattered light at the receiver", "angle (°)", "fraction")
        if r["angles"].size:
            h, e = np.histogram(np.degrees(r["angles"]), bins=40, weights=r["angle_w"])
            h = h / max(h.sum(), 1e-300)
            q.plot(e, h, stepMode="center", fillLevel=0, brush=theme.qcolor("scattered", 90), pen=pen("scattered", 1.4))

    def draw_phase(self, glw):
        r = self._need()
        p = new_plot(glw, 0, 0, "Volume scattering phase function β̃(θ)", "scattering angle (°)", "1/sr", legend=True)
        th = np.geomspace(0.01, 180, 400)
        for i, (k, pf) in enumerate(r["pf"].items()):
            p.plot(np.log10(th), np.log10(np.maximum(pf.beta(np.radians(th)), 1e-12)), pen=pen(f"c{i}", 2.0),
                   name=f"{k} (g = {pf.g:.3f})")
        cur = r["pf_cur"]
        p.plot(np.log10(th), np.log10(np.maximum(cur.beta(np.radians(th)), 1e-12)), pen=pen("c", 2.4, Qt.DashLine),
               name="in use (with water molecular scattering)")
        p.getAxis("bottom").setTicks([[(k, f"10{str(k).translate(SUP)}") for k in range(-2, 3)]])
        p.getAxis("left").setTicks([[(k, f"10{str(k).translate(SUP)}") for k in range(-4, 6)]])
