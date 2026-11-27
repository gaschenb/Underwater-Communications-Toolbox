"""Page scaffold shared by every workspace: control column, tabbed plot views, results table,
debounced live recomputation in a worker thread."""
from __future__ import annotations

import pyqtgraph as pg
import pyqtgraph.exporters  # noqa: F401
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt

from .widgets import EquationView, PlotView, ResultsTable, Runner, card, form


class SharedState(QtCore.QObject):
    """The scenario being edited (water, geometry, transmitter, receivers, impairments); every page
    writes its controls into it before computing and reads it back when it becomes visible."""
    changed = QtCore.pyqtSignal(str)        # origin page name

    def __init__(self, scenario):
        super().__init__()
        self.scenario = scenario

    def set(self, scenario, origin=""):
        self.scenario = scenario
        self.changed.emit(origin)


class Page(QtWidgets.QWidget):
    status = QtCore.pyqtSignal(str)
    title = "page"
    DEBOUNCE_MS = 90

    def __init__(self, state: SharedState, parent=None):
        super().__init__(parent)
        self.setObjectName("page")
        self.state = state
        self.result = None
        self.live = True
        self.views = []
        self.runner = Runner(self)
        self.runner.done.connect(self._done)
        self.runner.failed.connect(self._failed)
        self.runner.progressed.connect(self._progress)
        self._debounce = QtCore.QTimer(self); self._debounce.setSingleShot(True)
        self._debounce.setInterval(self.DEBOUNCE_MS)
        self._debounce.timeout.connect(self._launch)
        self._loading = False

        h = QtWidgets.QHBoxLayout(self); h.setContentsMargins(10, 10, 10, 10); h.setSpacing(10)
        scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFixedWidth(344); scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        inner = QtWidgets.QWidget()
        self.side = QtWidgets.QVBoxLayout(inner); self.side.setContentsMargins(0, 0, 8, 0)
        self.side.setSpacing(10)
        scroll.setWidget(inner)
        h.addWidget(scroll)
        split = QtWidgets.QSplitter(Qt.Vertical)
        self.tabs = QtWidgets.QTabWidget(); self.tabs.setDocumentMode(True)
        self.tabs.currentChanged.connect(lambda _: self.refresh())
        split.addWidget(self.tabs)
        self.table = None
        bottom = QtWidgets.QWidget()
        bv = QtWidgets.QVBoxLayout(bottom); bv.setContentsMargins(0, 0, 0, 0); bv.setSpacing(4)
        self.readout = QtWidgets.QLabel(""); self.readout.setObjectName("readout")
        self.readout.setWordWrap(True)
        bv.addWidget(self.readout)
        self._bottom = bv
        split.addWidget(bottom)
        split.setStretchFactor(0, 3); split.setStretchFactor(1, 1)
        split.setSizes([640, 230])
        self.split = split
        h.addWidget(split, 1)
        self.build()
        self.side.addStretch(1)

    # ------------------------------------------------------------------ construction helpers
    def add_card(self, title, subtitle=None):
        f, v = card(title, subtitle)
        self.side.addWidget(f)
        fl = form()
        v.addLayout(fl)
        return fl, v

    def add_equations(self, title="Model"):
        f, v = card(title)
        eq = EquationView()
        v.addWidget(eq)
        self.side.addWidget(f)
        return eq

    def add_view(self, title, draw, reusable=False):
        v = PlotView(title, draw, reusable=reusable)
        self.views.append(v)
        self.tabs.addTab(v, title)
        return v

    def add_table(self, headers):
        self.table = ResultsTable(headers)
        self.table.setMinimumHeight(90)
        self._bottom.addWidget(self.table, 1)
        return self.table

    def watch(self, *widgets):
        """Connect value-changed signals of controls to the debounced recompute."""
        for w in widgets:
            for sig in ("changed", "valueChanged", "currentIndexChanged", "toggled"):
                s = getattr(w, sig, None)
                if s is not None:
                    try:
                        s.connect(self.schedule)
                    except TypeError:
                        pass
                    break

    # ------------------------------------------------------------------ to implement
    def build(self):
        pass

    def job(self, heavy=False):
        """Return a function ctx -> result run in the worker thread."""
        raise NotImplementedError

    def after(self):
        pass

    def load_scenario(self, sc):
        pass

    def publish(self):
        """Tell the other pages that shared scenario fields changed."""
        self.state.changed.emit(self.title.lower())

    # ------------------------------------------------------------------ lifecycle
    def schedule(self, *_):
        if self._loading:
            return
        if self.live:
            self._debounce.start()
        else:
            self.status.emit("inputs changed: press Run (Ctrl+R)")

    def _launch(self, heavy=False):
        try:
            fn = self.job(heavy)
        except Exception as e:  # invalid inputs
            self.status.emit(f"{type(e).__name__}: {e}")
            return
        if fn is None:
            return
        self.runner.submit(fn)
        self.status.emit(f"{self.title}: computing…")

    def run_all(self):
        self._debounce.stop()
        self._launch(heavy=True)

    def cancel(self):
        self.runner.cancel()
        self.status.emit("cancelled")

    def set_live(self, on):
        self.live = on

    def _done(self, gen, res):
        if not self.runner.is_current(gen):
            return
        self.result = res
        self.after()
        for v in self.views:
            v.dirty = True
        self.refresh()
        msg = res.get("status") if isinstance(res, dict) else None
        self.status.emit(msg or f"{self.title}: done")

    def _failed(self, gen, msg):
        if self.runner.is_current(gen):
            self.status.emit(msg.splitlines()[0])
            print(msg)

    def _progress(self, gen, i, n):
        if self.runner.is_current(gen):
            self.status.emit(f"{self.title}: {i}/{n}")

    def current_view(self):
        return self.tabs.currentWidget()

    def refresh(self):
        v = self.current_view()
        if v is not None and v.dirty:
            v.redraw()

    def _do_refresh(self):
        v = self.current_view()
        if v is not None:
            v.redraw()

    def invalidate(self):
        for v in self.views:
            v.dirty = True
            v._key = v._plots = None
        self.refresh()

    def export_png(self, path):
        v = self.current_view()
        ex = pg.exporters.ImageExporter(v.glw.scene())
        ex.parameters()["width"] = max(int(v.glw.width() * 2), 800)
        ex.export(path)

    def export_csv(self, path):
        if self.table is not None:
            self.table.to_csv(path)

    def showEvent(self, e):
        super().showEvent(e)
        if self.result is None and not self.runner.running():
            self.schedule()
