"""Underwater Optical Comms Toolbox desktop app (PyQt5 + pyqtgraph).

    uwoc-gui      or      python -m uwoc.gui

Workspaces in the sidebar share one scenario (water, geometry, transmitter, receivers, impairments):
  Water & channel    Jerlov / classic / custom / bio-optical water, IOP spectra, phase functions,
                     attenuation versus range, impulse and frequency response, Monte Carlo photon paths
  Photodiode link    one frame end to end for any scheme and detector: intensity versus time, eye
                     diagram, levels / constellations, spectra, fading, noise breakdown
  Camera link        global- or rolling-shutter frame sequences of the transmitter, exposure timing,
                     sample streams, row profiles, colour constellations, pixel statistics
  Link budget        budget waterfall, power versus range with sensitivities, BER map over range and
                     bit rate, detector comparison
  BER & throughput   closed-form and Monte Carlo BER for many schemes, throughput versus range,
                     power / bandwidth efficiency, fading outage

Controls recompute live in a worker thread (newer requests supersede older ones); plots are
pyqtgraph (drag to pan, wheel to zoom, right-click for export).
"""
from __future__ import annotations

import os
import sys

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt

from ..core.scenarios import SCENARIOS
from . import theme

APP_NAME = "Underwater Optical Comms Toolbox"
PAGES = ("Water & channel", "Photodiode link", "Camera link", "Link budget", "BER & throughput")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, theme_name="light"):
        super().__init__()
        app = QtWidgets.QApplication.instance()
        theme.apply(app, theme_name)
        from .base import SharedState
        from .ber_page import BerPage
        from .budget_page import BudgetPage
        from .camera_page import CameraPage
        from .channel_page import ChannelPage
        from .photodiode_page import PhotodiodePage
        self.setWindowTitle(APP_NAME)
        self.resize(1560, 960)
        first = next(iter(SCENARIOS.values()))
        self.state = SharedState(first.copy())

        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        h = QtWidgets.QHBoxLayout(central); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(0)
        side = QtWidgets.QWidget(); side.setFixedWidth(200)
        sv = QtWidgets.QVBoxLayout(side); sv.setContentsMargins(0, 0, 0, 0); sv.setSpacing(0)
        b = QtWidgets.QLabel(APP_NAME); b.setObjectName("brand"); b.setWordWrap(True)
        sv.addWidget(b)
        s = QtWidgets.QLabel("photodiode · camera · water optics"); s.setObjectName("brandSub")
        sv.addWidget(s)
        self.nav = QtWidgets.QListWidget(); self.nav.setObjectName("nav")
        self.nav.setFocusPolicy(Qt.NoFocus)
        for name in PAGES:
            self.nav.addItem(name)
        sv.addWidget(self.nav, 1)
        h.addWidget(side)
        self.stack = QtWidgets.QStackedWidget()
        h.addWidget(self.stack, 1)
        self.chan = ChannelPage(self.state)
        self.pd = PhotodiodePage(self.state)
        self.cam = CameraPage(self.state)
        self.budget = BudgetPage(self.state)
        self.ber = BerPage(self.state)
        self.pages = [self.chan, self.pd, self.cam, self.budget, self.ber]
        for p in self.pages:
            self.stack.addWidget(p)
            p.status.connect(self.show_status)
            p.runner.busy.connect(self._busy)
        self.state.changed.connect(self._state_changed)
        self.nav.currentRowChanged.connect(self._page_changed)

        tb = QtWidgets.QToolBar(); tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.addToolBar(tb)
        tb.addWidget(QtWidgets.QLabel("  Scenario  "))
        self.cb_sc = QtWidgets.QComboBox(); self.cb_sc.setMinimumWidth(300)
        for name, sc in SCENARIOS.items():
            self.cb_sc.addItem(name)
            self.cb_sc.setItemData(self.cb_sc.count() - 1, sc.description, Qt.ToolTipRole)
        self.cb_sc.currentTextChanged.connect(self.load_scenario)
        tb.addWidget(self.cb_sc)
        tb.addSeparator()
        self.a_run = tb.addAction("▶  Run", self.run); self.a_run.setShortcut("Ctrl+R")
        self.a_cancel = tb.addAction("■  Cancel", self.cancel); self.a_cancel.setShortcut("Esc")
        self.a_live = tb.addAction("Live updates"); self.a_live.setCheckable(True)
        self.a_live.setChecked(True); self.a_live.setShortcut("Ctrl+L")
        self.a_live.toggled.connect(self._live)
        tb.addSeparator()
        tb.addAction("Export plot…", self.export_png).setShortcut("Ctrl+E")
        tb.addAction("Export table…", self.export_csv).setShortcut("Ctrl+Shift+E")
        spacer = QtWidgets.QWidget(); spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                                          QtWidgets.QSizePolicy.Preferred)
        tb.addWidget(spacer)
        self.a_theme = tb.addAction("Dark theme", self.toggle_theme); self.a_theme.setShortcut("Ctrl+T")
        tb.addAction("Help", self.about).setShortcut("F1")
        self.busy = QtWidgets.QProgressBar(); self.busy.setRange(0, 0); self.busy.setFixedWidth(120)
        self.busy.setVisible(False)
        self.statusBar().addPermanentWidget(self.busy)
        self.sc_label = QtWidgets.QLabel(""); self.sc_label.setObjectName("muted")
        self.statusBar().addWidget(self.sc_label, 1)
        self.nav.setCurrentRow(0)
        self.load_scenario(self.cb_sc.currentText())

    def load_scenario(self, name):
        sc = SCENARIOS[name].copy()
        self.state.scenario = sc
        self.sc_label.setText(sc.description)
        for p in self.pages:
            p.result = None if p is not self.page() else p.result
            p.stale = p is not self.page()
        self.page().load_scenario(sc)
        if sc.receiver == "camera" and self.page() is self.pd:
            self.nav.setCurrentRow(2)
        elif sc.receiver == "photodiode" and self.page() is self.cam:
            self.nav.setCurrentRow(1)

    def _state_changed(self, origin):
        for p in self.pages:
            if p.title.lower() != origin:
                p.stale = True

    def _busy(self, _):
        self.busy.setVisible(any(p.runner.running() for p in self.pages))

    def page(self):
        return self.stack.currentWidget()

    def _page_changed(self, i):
        self.stack.setCurrentIndex(i)
        p = self.page()
        if getattr(p, "stale", False) or p.result is None:
            p.stale = False
            p.load_scenario(self.state.scenario)
        p.refresh()

    def show_status(self, msg):
        self.statusBar().showMessage(msg, 8000)

    def run(self):
        self.page().run_all()

    def cancel(self):
        self.page().cancel()

    def _live(self, on):
        for p in self.pages:
            p.set_live(on)

    def export_png(self):
        v = self.page().current_view()
        name = getattr(v, "title", "plot").lower().replace(" ", "_").replace("&", "and")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export plot", f"{name}.png",
                                                        "PNG image (*.png)")
        if path:
            self.page().export_png(path); self.show_status(f"saved {path}")

    def export_csv(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export table", "results.csv", "CSV (*.csv)")
        if path:
            self.page().export_csv(path); self.show_status(f"saved {path}")

    def toggle_theme(self):
        new = "dark" if theme.current["name"] == "light" else "light"
        theme.apply(QtWidgets.QApplication.instance(), new)
        self.a_theme.setText("Light theme" if new == "dark" else "Dark theme")
        for p in self.pages:
            if p.result is not None:
                p.after()
            p.invalidate()
            for eq in p.findChildren(QtWidgets.QWidget):
                eq.update()

    def about(self):
        QtWidgets.QMessageBox.information(self, APP_NAME, (
            f"{APP_NAME}\n\n"
            "Pick a scenario in the toolbar; every workspace shares its water, geometry, transmitter, "
            "receivers and impairments. Edits on one page carry over to the others.\n\n"
            "Ctrl+R  run (Monte Carlo BER points, more frames, more photons)\n"
            "Esc     cancel the running job\n"
            "Ctrl+L  toggle live updates\n"
            "Ctrl+E  export the current plot, Ctrl+Shift+E the table\n"
            "Ctrl+T  light / dark theme\n\n"
            "Plots: drag to pan, wheel to zoom, right-click for options, click a legend entry to "
            "hide a curve."))

    def closeEvent(self, e):
        for p in self.pages:
            p.runner.cancel()
            for t in list(p.runner.threads):
                t.wait(3000)
        super().closeEvent(e)


def main(argv=None):
    argv = sys.argv if argv is None else argv
    QtCore.QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QtCore.QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    app.setApplicationName(APP_NAME)
    f = app.font(); f.setPointSize(9)
    if sys.platform.startswith("win"):
        f.setFamily("Segoe UI")
    app.setFont(f)
    w = MainWindow("dark" if "--dark" in argv or os.environ.get("UWOC_THEME") == "dark" else "light")
    w.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
