"""Camera link workspace: frame sequences of the transmitter and camera-side analytics."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt

from ..core import camera as C, schemes as S
from ..core.channel import SOLAR
from ..core.devices import CAMERAS
from . import theme
from .base import Page
from .widgets import Empty, Slider, combo, image_cmap, new_plot, pen, spin

EQ = [r"t_{f,r}=\frac{f+\varphi}{f_{fps}}+r\,t_{row},\quad t_{row}=\frac{1}{f_{fps}(H+V)}",
      r"e_{px}=\sum_{LED}\left[N_bT_b+N_sT_s\right]\frac{1}{T_e}\int_{t_{f,r}}^{t_{f,r}+T_e}x(t)\,dt+e_{bg}",
      r"r_{img}=\frac{fD}{2L\,p},\quad \theta_h=\frac{\theta_{rms}}{\sqrt{3}},\quad \theta_{rms}^2=P\,b\,L\,\theta_1^2",
      r"\mathrm{UFSOOK:}\ f_1=(n+\frac{1}{2})f_{fps},\ f_0=n f_{fps}"]


class CameraPage(Page):
    title = "Camera link"

    def build(self):
        f, _ = self.add_card("Scheme")
        self.cb_scheme = combo(C.CAMERA_SCHEMES, "Manchester"); f.addRow("Scheme", self.cb_scheme)
        self.sl_rate = Slider(5, 20000, 2000, log=True, decimals=1, suffix=" chip/s"); f.addRow("Chip rate", self.sl_rate)
        self.sl_car = Slider(30, 20000, 240, log=True, decimals=0, suffix=" Hz"); f.addRow("Blink / tone", self.sl_car)
        self.sp_frames = spin(2, 240, 8); f.addRow("Frames", self.sp_frames)
        self.sl_phase = Slider(0, 0.99, 0.0, decimals=3); f.addRow("Frame phase", self.sl_phase)
        self.sp_seed = spin(0, 9999, 0); f.addRow("Seed", self.sp_seed)
        f, _ = self.add_card("Camera")
        self.cb_cam = combo(["(scenario)"] + list(CAMERAS)); f.addRow("Preset", self.cb_cam)
        self.cb_shutter = combo(["rolling", "global"]); f.addRow("Shutter", self.cb_shutter)
        self.sl_fps = Slider(1, 2000, 30, log=True, decimals=1, suffix=" fps"); f.addRow("Frame rate", self.sl_fps)
        self.sl_exp = Slider(1e-6, 0.05, 1.25e-4, log=True, decimals=6, suffix=" s"); f.addRow("Exposure", self.sl_exp)
        self.sp_w = spin(32, 1280, 192, 16); f.addRow("Width (px)", self.sp_w)
        self.sp_h = spin(32, 1080, 320, 16); f.addRow("Height (rows)", self.sp_h)
        self.sl_f = Slider(2, 100, 12, log=True, decimals=1, suffix=" mm"); f.addRow("Focal length", self.sl_f)
        self.sl_N = Slider(1, 16, 2, log=True, decimals=1); f.addRow("f-number", self.sl_N)
        self.sl_pix = Slider(1, 10, 3, decimals=2, suffix=" µm"); f.addRow("Pixel pitch", self.sl_pix)
        self.sl_rn = Slider(0.5, 30, 3, log=True, decimals=1, suffix=" e⁻"); f.addRow("Read noise", self.sl_rn)
        self.ck_col = QtWidgets.QCheckBox("Colour (Bayer RGB)"); f.addRow(self.ck_col)
        f, _ = self.add_card("Transmitter and range")
        self.sl_R = Slider(0.3, 100, 5, log=True, decimals=2, suffix=" m"); f.addRow("Range", self.sl_R)
        self.sl_P = Slider(1e-3, 100, 1, log=True, decimals=3, suffix=" W"); f.addRow("Optical power", self.sl_P)
        self.sl_src = Slider(1, 500, 30, log=True, decimals=1, suffix=" mm"); f.addRow("Source diameter", self.sl_src)
        self.sp_arr = spin(1, 8, 1); f.addRow("LED array side", self.sp_arr)
        self.sl_pitch = Slider(5, 300, 40, log=True, decimals=1, suffix=" mm"); f.addRow("Array pitch", self.sl_pitch)
        self.sl_div = Slider(0.5, 80, 15, log=True, decimals=1, suffix=" °"); f.addRow("Beam divergence", self.sl_div)
        self.cb_sky = combo(list(SOLAR)); f.addRow("Ambient light", self.cb_sky)
        self.sl_depth = Slider(0, 1000, 10, decimals=0, suffix=" m"); f.addRow("Depth", self.sl_depth)
        self.eq = self.add_equations("Image formation")
        self.eq.set_equations(EQ, "Ballistic light forms the source image, scattered light a halo; rows of a "
                                  "rolling shutter sample the waveform at different times (stripes).")
        self.cb_cam.currentTextChanged.connect(self._cam_preset)
        self.cb_scheme.currentTextChanged.connect(self._vis)
        self.watch(self.cb_scheme, self.sl_rate, self.sl_car, self.sp_frames, self.sl_phase, self.sp_seed,
                   self.cb_shutter, self.sl_fps, self.sl_exp, self.sp_w, self.sp_h, self.sl_f, self.sl_N, self.sl_pix,
                   self.sl_rn, self.ck_col, self.sl_R, self.sl_P, self.sl_src, self.sp_arr, self.sl_pitch,
                   self.sl_div, self.cb_sky, self.sl_depth)
        self.add_view("Frames", self.draw_frames)
        self.add_view("Exposure timing", self.draw_timing)
        self.add_view("Samples vs time", self.draw_samples)
        self.add_view("Row profile", self.draw_rows)
        self.add_view("Constellation / levels", self.draw_const)
        self.add_view("Pixel statistics", self.draw_pixels)
        self.add_table(["quantity", "value", "unit"])
        self._vis()

    def _vis(self):
        sch = self.cb_scheme.currentText()
        for w, on in ((self.sl_rate, sch in C.CHIP_SCHEMES), (self.sl_car, sch in ("UFSOOK", "UPSOOK", "RS-FSK"))):
            w.setEnabled(on)

    def _cam_preset(self, name):
        if name not in CAMERAS or self._loading:
            return
        self._loading = True
        self._set_cam(CAMERAS[name])
        self._loading = False
        self.schedule()

    def _set_cam(self, c):
        self.cb_shutter.setCurrentText(c.shutter); self.sl_fps.setValue(c.fps); self.sl_exp.setValue(c.exposure_s)
        self.sp_w.setValue(c.width); self.sp_h.setValue(c.height); self.sl_f.setValue(c.focal_mm)
        self.sl_N.setValue(c.f_number); self.sl_pix.setValue(c.pixel_um); self.sl_rn.setValue(c.read_noise)
        self.ck_col.setChecked(c.color)
        self._cam_base = c

    def load_scenario(self, sc):
        self._loading = True
        self.cb_scheme.setCurrentText(sc.camera_scheme)
        self.sl_rate.setValue(sc.camera_chip_rate); self.sl_car.setValue(sc.camera_carrier_hz)
        self.sp_frames.setValue(sc.n_frames); self.sl_phase.setValue(sc.frame_phase)
        self.cb_cam.setCurrentIndex(0)
        self._set_cam(sc.camera)
        self.sl_R.setValue(sc.range_m); self.sl_P.setValue(sc.tx.power_w); self.sl_src.setValue(sc.tx.source_mm)
        self.sp_arr.setValue(sc.tx.array); self.sl_pitch.setValue(sc.tx.pitch_mm); self.sl_div.setValue(sc.tx.divergence_deg)
        self.cb_sky.setCurrentText(sc.ambient.sky); self.sl_depth.setValue(sc.ambient.depth_m)
        self._loading = False
        self._vis()
        self.schedule()

    def write(self, sc):
        sc.camera_scheme = self.cb_scheme.currentText()
        sc.camera_chip_rate = self.sl_rate.value(); sc.camera_carrier_hz = self.sl_car.value()
        sc.n_frames = self.sp_frames.value(); sc.frame_phase = self.sl_phase.value()
        base = getattr(self, "_cam_base", sc.camera)
        sc.camera = base.copy(shutter=self.cb_shutter.currentText(), fps=self.sl_fps.value(),
                              exposure_s=self.sl_exp.value(), width=self.sp_w.value(), height=self.sp_h.value(),
                              focal_mm=self.sl_f.value(), f_number=self.sl_N.value(), pixel_um=self.sl_pix.value(),
                              read_noise=self.sl_rn.value(), color=self.ck_col.isChecked(),
                              vblank=base.vblank if base.vblank else 20)
        sc.range_m = self.sl_R.value(); sc.tx.power_w = self.sl_P.value(); sc.tx.source_mm = self.sl_src.value()
        sc.tx.array = self.sp_arr.value(); sc.tx.pitch_mm = self.sl_pitch.value(); sc.tx.divergence_deg = self.sl_div.value()
        sc.ambient.sky = self.cb_sky.currentText(); sc.ambient.depth_m = self.sl_depth.value()

    def job(self, heavy=False):
        sc = self.state.scenario
        self.write(sc)
        self.publish()
        sc = sc.copy()
        sch = sc.camera_scheme
        if not C.applicable(sch, sc.camera):
            raise ValueError(f"{sch} needs a {'colour' if sch.endswith('CSK') else 'rolling-shutter'} camera")
        nf = sc.n_frames * (4 if heavy else 1)
        seed = self.sp_seed.value()
        exp_rows = sc.camera.exposure_s * sc.camera.fps
        if exp_rows > 0.98:
            raise ValueError("exposure longer than the frame period")

        def run(ctx):
            r = C.simulate(sc, sch, n_frames=nf, seed=seed)
            ctx.check()
            th = C.ber_theory(sc, sch)
            st = C.sample_stats(sc, sch)
            return dict(run=r, theory=th, stats=st, sc=sc, thr=C.throughput(sc, sch),
                        status=f"{sch}: {int(r.errors)} errors / {r.n_bits} bits, goodput {r.goodput:.4g} bit/s")
        return run

    def after(self):
        R = self.result; r, st = R["run"], R["stats"]
        sc = R["sc"]
        sat = r.info["peak_dn"] >= 2 ** sc.camera.bits - 1
        rows = [["scheme", r.scheme, ""], ["shutter", "rolling" if r.rolling else "global", ""],
                ["frames", len(r.frame_t), f"{sc.camera.fps:g} fps"],
                ["BER (decoded)", r.ber, f"{int(r.errors)} / {r.n_bits}"], ["BER closed form", R["theory"], ""],
                ["goodput", r.goodput, "bit/s"], ["raw throughput", R["thr"], "bit/s"],
                ["chips observed", r.info.get("observed_fraction", 1.0), "fraction"],
                ["source image radius", r.info["r_img"], "px"], ["scatter halo σ", r.info["halo_px"], "px"],
                ["ballistic e⁻ / exposure", r.info["Nb"], "per LED"], ["scattered e⁻ / exposure", r.info["Ns"], "per LED"],
                ["background e⁻ / px", r.info["bg_e"], "per exposure"], ["peak pixel", r.info["peak_dn"],
                                                                         "DN (saturated)" if sat else "DN"],
                ["sample SNR", (st["m1"] - st["m0"]) / max(st["s0"] + st["s1"], 1e-12), "Q per sample"]]
        if r.rolling:
            rows.append(["row time", r.row_time * 1e6, "µs"])
            rows.append(["source rows", st["n_rows"], "rows"])
        flags = [{} for _ in rows]
        flags[3] = {1: "good" if r.ber < 1e-3 else ("warn" if r.ber < 3.8e-3 else "bad")}
        if sat:
            flags[13] = {1: "warn"}
        self.table.set_rows(rows, flags=flags)
        self.readout.setText(f"camera FOV {sc.camera.fov_deg:.1f}° · IFOV {sc.camera.ifov * 1e3:.3f} mrad · "
                             f"exposure {sc.camera.exposure_s * 1e6:.0f} µs · {r.info.get('pilot', 0)} pilot symbols")

    def _need(self):
        if self.result is None:
            raise Empty("computing…")
        return self.result

    # ---------------------------------------------------------------- views
    def draw_frames(self, glw):
        R = self._need(); r = R["run"]
        F = min(r.frames.shape[0], 12)
        cols = 4 if F > 4 else F
        int(np.ceil(F / cols))
        vmax = max(float(np.percentile(r.frames, 99.9)), 1.0)
        cy, cx, rr = r.roi
        for i in range(F):
            vb = glw.addViewBox(row=i // cols, col=i % cols)
            vb.setAspectLocked(True); vb.invertY(True); vb.setMouseEnabled(True, True)
            fr = r.frames[i].astype(float)
            if fr.ndim == 3:
                img = pg.ImageItem(np.clip(fr / vmax, 0, 1), levels=(0, 1))
            else:
                img = pg.ImageItem(fr, levels=(0, vmax))
                img.setLookupTable(image_cmap("gray").getLookupTable(0, 1, 256))
            vb.addItem(img)
            ell = QtWidgets.QGraphicsEllipseItem(cx - rr, cy - rr, 2 * rr, 2 * rr)
            ell.setPen(pen("exposure", 1.0, Qt.DashLine)); vb.addItem(ell)
            t = pg.TextItem(f"frame {i}  t = {r.frame_t[i] * 1e3:.1f} ms", color=theme.current["plot_fg"], anchor=(0, 1))
            t.setPos(0, 0); vb.addItem(t)
            vb.autoRange(padding=0.02)

    def draw_timing(self, glw):
        R = self._need(); r = R["run"]; cam = R["sc"].camera
        Tf = 1 / cam.fps
        p = new_plot(glw, 0, 0, "Transmitted drive over the first frames; shaded: when the source is being exposed",
                     "time (ms)", "drive", legend=True)
        nf = min(3, len(r.frame_t))
        k = r.t < r.frame_t[nf - 1] + Tf
        d = r.drive[k]
        if d.shape[2] == 3:
            for j, c in enumerate("RGB"):
                p.plot(r.t[k] * 1e3, d[:, 0, j], pen=pen(c, 1.2), name=f"{c} drive")
        else:
            p.plot(r.t[k] * 1e3, d[:, 0, 0], pen=pen("tx", 1.2), name="LED drive")
        rows = np.nonzero(r.valid[: cam.height])[0] if r.rolling else np.array([0])
        col = theme.qcolor("exposure", 60)
        for f in range(nf):
            if r.rolling and rows.size:
                a = r.frame_t[f] + rows[0] * r.row_time
                b = r.frame_t[f] + rows[-1] * r.row_time + r.exposure
            else:
                a, b = r.frame_t[f], r.frame_t[f] + r.exposure
            reg = pg.LinearRegionItem(values=(a * 1e3, b * 1e3), movable=False, brush=pg.mkBrush(col),
                                      pen=pg.mkPen(None))
            reg.setZValue(-5); p.addItem(reg)
        # zoom: individual exposure windows against the waveform
        q = new_plot(glw, 1, 0, "Zoom: exposure window of every 4th source row (rolling) / each frame (global)",
                     "time (ms)", "row" if r.rolling else "frame")
        if r.rolling and rows.size:
            mid = rows[rows.size // 2]
            t0 = r.frame_t[0] + mid * r.row_time
            span = max(12 / max(r.chip_rate, 1e-9) if r.chip_rate else 0, 40 * r.row_time, 3 * r.exposure)
            lo, hi = t0 - span / 2, t0 + span / 2
            kk = (r.t >= lo) & (r.t <= hi)
            rr = rows[(r.frame_t[0] + rows * r.row_time >= lo) & (r.frame_t[0] + rows * r.row_time + r.exposure <= hi)]
            top = rr.max() + 4 if rr.size else 10
            bot = rr.min() - 4 if rr.size else 0
            q.plot(r.t[kk] * 1e3, bot + (top - bot) * r.drive[kk, 0, 0] / max(r.drive.max(), 1e-12),
                   pen=pen("tx", 1.4))
            prof = r.row_profile[0] if r.row_profile is not None else None
            vmax = np.abs(prof[rr]).max() if prof is not None and rr.size else 1
            for row in rr[::4]:
                a = r.frame_t[0] + row * r.row_time
                c = float(np.clip(prof[row] / max(vmax, 1e-12), 0, 1)) if prof is not None else 0.5
                cc = theme.qcolor("exposure", int(80 + 175 * c))
                q.plot([a * 1e3, (a + r.exposure) * 1e3], [row, row], pen=pg.mkPen(cc, width=4))
            q.invertY(True)
        else:
            span = max(4 * r.exposure, 6 / max(r.chip_rate, 1e-9) if r.chip_rate else 0, 2e-3)
            t0 = r.frame_t[min(1, len(r.frame_t) - 1)]
            lo, hi = t0 - span / 2, t0 + span / 2
            kk = (r.t >= lo) & (r.t <= hi)
            q.plot(r.t[kk] * 1e3, r.drive[kk, 0, 0], pen=pen("tx", 1.4))
            for f, tf in enumerate(r.frame_t):
                if lo <= tf <= hi:
                    reg = pg.LinearRegionItem(values=(tf * 1e3, (tf + r.exposure) * 1e3), movable=False,
                                              brush=pg.mkBrush(col), pen=pg.mkPen(None))
                    q.addItem(reg)

    def draw_samples(self, glw):
        R = self._need(); r = R["run"]
        p = new_plot(glw, 0, 0, "Region-of-interest samples (normalised) with the transmitted drive",
                     "time (ms)", "value", legend=True)
        v = r.samples[:, 0] if r.samples.ndim == 2 else r.samples
        ok = r.valid
        x = r.samples_t * 1e3
        tk = r.t < r.samples_t.max() + 1e-3
        dmax = r.drive[tk].max() if r.drive[tk].size else 1
        sc = max(np.nanmax(np.abs(v[ok])) if ok.any() else 1, 1e-12)
        p.plot(r.t[tk] * 1e3, r.drive[tk, 0, 0] / max(dmax, 1e-12), pen=pen("tx", 1.0, alpha=150),
               name="Tx drive (LED 0)")
        if r.samples.ndim == 2 and r.samples.shape[1] > 1 and r.scheme != "Spatial OOK":
            for j, c in enumerate("RGB"[: r.samples.shape[1]]):
                p.plot(x[ok], r.samples[ok, j] / sc, pen=None, symbol="o", symbolSize=3, symbolPen=None,
                       symbolBrush=theme.qcolor(c, 160), name=f"{c} samples")
        else:
            inside = ok.copy()
            if r.chip_rate > 0:
                Tc = 1 / r.chip_rate
                j0 = np.floor((r.samples_t - r.exposure / 2) / Tc + 1e-9)
                j1 = np.floor((r.samples_t + r.exposure / 2) / Tc - 1e-9)
                inside &= j0 == j1
            p.plot(x[inside], v[inside] / sc, pen=None, symbol="o", symbolSize=4, symbolPen=None,
                   symbolBrush=theme.qcolor("rx", 200), name="samples inside one chip")
            st = ok & ~inside
            if st.any():
                p.plot(x[st], v[st] / sc, pen=None, symbol="o", symbolSize=3, symbolPen=None,
                       symbolBrush=theme.qcolor("erased", 200), name="samples straddling chips")
        if r.chips_rx is not None and r.chip_rate > 0:
            seen = r.chip_seen
            Tc = 1 / r.chip_rate
            tc = (np.arange(seen.size) + 0.5) * Tc * 1e3
            m = tc <= x.max() + Tc * 1e3
            ers = m & ~seen
            p.plot(tc[ers], np.full(ers.sum(), -0.15), pen=None, symbol="s", symbolSize=5, symbolPen=None,
                   symbolBrush=theme.qcolor("erased", 200), name="erased chips")
        p.setYRange(-0.3, 1.25)
        if r.rolling and ok.any():
            t1 = x[ok][0]
            p.setXRange(t1 - 2, t1 + 1.2e3 / R["sc"].camera.fps)

    def draw_rows(self, glw):
        R = self._need(); r = R["run"]
        if r.row_profile is None:
            p = new_plot(glw, 0, 0, "Per-frame ROI value (global shutter)", "frame", "value")
            v = r.samples
            for j in range(min(v.shape[1], 16)):
                p.plot(np.arange(v.shape[0]), v[:, j] / max(np.abs(v).max(), 1e-12) + 1.1 * j, pen=pen(f"c{j % 10}", 1.4),
                       symbol="o", symbolSize=4, symbolBrush=theme.qcolor(f"c{j % 10}"))
            return
        prof = r.row_profile
        p = new_plot(glw, 0, 0, "Row profile through the source, every frame (rolling-shutter stripes)", "row",
                     "frame")
        img = pg.ImageItem(prof)
        img.setLookupTable(image_cmap("viridis").getLookupTable(0, 1, 256))
        img.setRect(QtCore.QRectF(0, 0, prof.shape[1], prof.shape[0]))
        p.addItem(img)
        q = new_plot(glw, 1, 0, "Frame 0 row signal against row time", "time (ms)", "ROI row sum (DN)")
        tt = (r.frame_t[0] + np.arange(prof.shape[1]) * r.row_time + r.exposure / 2) * 1e3
        q.plot(tt, prof[0], pen=pen("rx", 1.6))
        if r.scheme == "RS-FSK":
            tones = r.info.get("tones", [])
            q.setTitle(f"Frame 0 row signal; tones {', '.join(f'{t:.0f}' for t in tones)} Hz", size="9pt")

    def draw_const(self, glw):
        R = self._need(); r = R["run"]
        if r.scheme.endswith("CSK"):
            p = new_plot(glw, 0, 0, f"{r.scheme}: chromaticity of received samples", "x (triangle)", "y (triangle)")
            V = S.chromaticity(np.eye(3))
            p.plot(np.append(V[:, 0], V[0, 0]), np.append(V[:, 1], V[0, 1]), pen=pen("tx", 1.4))
            s = r.samples[r.valid]
            z = S.chromaticity(np.maximum(s, 0))
            p.plot(z[:, 0], z[:, 1], pen=None, symbol="o", symbolSize=5, symbolPen=None, symbolBrush=theme.qcolor("rx", 160))
            p.setAspectLocked(True)
            return
        p = new_plot(glw, 0, 0, "Histogram of sample values", "value", "count")
        v = r.samples[r.valid].ravel()
        h, e = np.histogram(v, bins=80)
        p.plot(e, h, stepMode="center", fillLevel=0, brush=theme.qcolor("rx", 100), pen=pen("rx", 1.2))

    def draw_pixels(self, glw):
        R = self._need(); r = R["run"]
        fr = r.frames.astype(float)
        if fr.ndim == 4:
            fr = fr.sum(-1)
        cy, cx, rr = r.roi
        H, W = fr.shape[1:]
        yy, xx = np.mgrid[0:H, 0:W]
        roi = ((yy + 0.5 - cy) ** 2 + (xx + 0.5 - cx) ** 2) <= rr ** 2
        p = new_plot(glw, 0, 0, "Pixel values inside and outside the source region", "DN", "count (log)", legend=True)
        lo, hi = fr.min(), fr.max() + 1
        bins = np.linspace(lo, hi, 80)
        for k, m, nm in (("rx", roi, "source region"), ("bg", ~roi, "background")):
            h, e = np.histogram(fr[:, m], bins=bins)
            p.plot(e, np.log10(h + 1), stepMode="center", fillLevel=0, brush=theme.qcolor(k, 90), pen=pen(k, 1.2), name=nm)
