"""Shared GUI building blocks: background runner, parameter sliders, cards, tables, plots."""
from __future__ import annotations

import csv
import math
import traceback

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

from . import theme


# =========================================================================== background jobs
class Cancelled(Exception):
    pass


class JobContext:
    """Handed to a job function; emitting from a cancelled job raises Cancelled."""

    def __init__(self, runner, gen):
        self.gen, self.cancelled, self._r = gen, False, runner

    def check(self):
        if self.cancelled:
            raise Cancelled

    def emit(self, key, obj):
        self.check()
        self._r.partial.emit(self.gen, key, obj)

    def progress(self, i, n):
        self.check()
        self._r.progressed.emit(self.gen, int(i), int(n))

    def log(self, msg):
        self._r.logged.emit(self.gen, msg)


class _JobThread(QtCore.QThread):
    def __init__(self, fn, ctx, runner):
        super().__init__()
        self.fn, self.ctx, self.runner = fn, ctx, runner

    def run(self):
        try:
            out = self.fn(self.ctx)
            if not self.ctx.cancelled:
                self.runner.done.emit(self.ctx.gen, out)
        except Cancelled:
            pass
        except Exception as e:  # surfaced in the log
            if not self.ctx.cancelled:
                self.runner.failed.emit(self.ctx.gen, f"{type(e).__name__}: {e}\n"
                                                      f"{traceback.format_exc()}")


class Runner(QtCore.QObject):
    """Runs one job at a time in a QThread. Submitting a new job cancels the previous one
    (cooperatively, at its next emit/progress/check) and results carry a generation number,
    so stale results are never applied."""
    partial = QtCore.pyqtSignal(int, str, object)
    done = QtCore.pyqtSignal(int, object)
    failed = QtCore.pyqtSignal(int, str)
    progressed = QtCore.pyqtSignal(int, int, int)
    logged = QtCore.pyqtSignal(int, str)
    busy = QtCore.pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.gen = 0
        self.ctx = None
        self.threads = set()

    def submit(self, fn):
        if self.ctx is not None:
            self.ctx.cancelled = True
        self.gen += 1
        self.ctx = JobContext(self, self.gen)
        t = _JobThread(fn, self.ctx, self)
        self.threads.add(t)
        t.finished.connect(self._finished)
        t.start()
        self.busy.emit(True)
        return self.gen

    def cancel(self):
        if self.ctx is not None:
            self.ctx.cancelled = True

    def _finished(self):
        t = self.sender()
        self.threads.discard(t)
        if t is not None:
            t.deleteLater()
        if not self.running():
            self.busy.emit(False)

    def running(self):
        return any(t.isRunning() for t in self.threads)

    def is_current(self, gen):
        return gen == self.gen

    def wait_idle(self, ms=600_000):
        app = QtWidgets.QApplication.instance()
        timer = QtCore.QElapsedTimer(); timer.start()
        while (self.running() or self.threads) and timer.elapsed() < ms:
            app.processEvents()
            QtCore.QThread.msleep(5)
        app.processEvents()


# =========================================================================== small widgets
def card(title=None, subtitle=None):
    """A rounded panel with an optional title; returns (frame, inner QVBoxLayout)."""
    f = QtWidgets.QFrame(); f.setObjectName("card")
    v = QtWidgets.QVBoxLayout(f); v.setContentsMargins(12, 10, 12, 12); v.setSpacing(6)
    if title:
        lab = QtWidgets.QLabel(title); lab.setObjectName("cardTitle"); v.addWidget(lab)
    if subtitle:
        s = QtWidgets.QLabel(subtitle); s.setObjectName("muted"); s.setWordWrap(True); v.addWidget(s)
    return f, v


def form():
    fl = QtWidgets.QFormLayout()
    fl.setLabelAlignment(Qt.AlignLeft)
    fl.setHorizontalSpacing(10); fl.setVerticalSpacing(5)
    fl.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
    return fl


def spin(lo, hi, val, step=1, suffix=""):
    s = QtWidgets.QSpinBox(); s.setRange(lo, hi); s.setValue(val); s.setSingleStep(step)
    s.setSuffix(suffix); s.setKeyboardTracking(False); s.setAccelerated(True)
    return s


def muted(text):
    lab = QtWidgets.QLabel(text); lab.setObjectName("muted"); lab.setWordWrap(True)
    return lab


class SciSpin(QtWidgets.QDoubleSpinBox):
    """Double spin box that shows 5 significant digits and accepts 1e-3 notation;
    the step is 10 % of the value (multiplicative), which suits scale parameters."""

    def __init__(self, lo, hi, val, parent=None):
        super().__init__(parent)
        self.setDecimals(12); self.setRange(lo, hi); self.setValue(val)
        self.setKeyboardTracking(False); self.setAccelerated(True)

    def textFromValue(self, v):
        return f"{v:.4g}"

    def valueFromText(self, text):
        try:
            return float(text)
        except ValueError:
            return self.value()

    def validate(self, text, pos):
        try:
            float(text)
            return QtGui.QValidator.Acceptable, text, pos
        except ValueError:
            return QtGui.QValidator.Intermediate, text, pos

    def stepBy(self, n):
        v = self.value()
        step = 0.1 * abs(v) if v != 0 else 0.01 * (self.maximum() - self.minimum())
        self.setValue(v + n * step)


class ParamEditor(QtWidgets.QWidget):
    """One row per parameter: name, slider (log scale for positive ranges spanning > 50×),
    and an editable value. Emits `changed` continuously while dragging."""
    changed = QtCore.pyqtSignal()
    N = 1000

    def __init__(self, parent=None):
        super().__init__(parent)
        self.grid = QtWidgets.QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0); self.grid.setHorizontalSpacing(8)
        self.grid.setVerticalSpacing(4)
        self.rows = []

    def set_params(self, names, values, lo, hi, defaults=None):
        while self.grid.count():
            w = self.grid.takeAt(0).widget()
            if w is not None:
                w.hide(); w.setParent(None); w.deleteLater()
        self.rows = []
        self.defaults = np.array(values if defaults is None else defaults, float)
        for i, (n, v, a, b) in enumerate(zip(names, values, lo, hi)):
            logs = a > 0 and b / a > 50
            lab = QtWidgets.QLabel(n)
            lab.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
            sl = QtWidgets.QSlider(Qt.Horizontal); sl.setRange(0, self.N); sl.setMinimumWidth(70)
            sp = SciSpin(float(a), float(b), float(v))
            sp.setFixedWidth(96)
            row = dict(lo=float(a), hi=float(b), log=logs, slider=sl, spin=sp)
            self.rows.append(row)
            self._sync_slider(row)
            sl.valueChanged.connect(lambda pos, r=row: self._from_slider(r, pos))
            sp.valueChanged.connect(lambda val, r=row: self._from_spin(r))
            self.grid.addWidget(lab, i, 0); self.grid.addWidget(sl, i, 1); self.grid.addWidget(sp, i, 2)
        self.grid.setColumnStretch(1, 1)

    def _to_pos(self, r, v):
        if r["log"]:
            u = (math.log(v) - math.log(r["lo"])) / (math.log(r["hi"]) - math.log(r["lo"]))
        else:
            u = (v - r["lo"]) / (r["hi"] - r["lo"])
        return int(round(min(max(u, 0.0), 1.0) * self.N))

    def _from_pos(self, r, pos):
        u = pos / self.N
        if r["log"]:
            return math.exp(math.log(r["lo"]) + u * (math.log(r["hi"]) - math.log(r["lo"])))
        return r["lo"] + u * (r["hi"] - r["lo"])

    def _sync_slider(self, r):
        r["slider"].blockSignals(True)
        r["slider"].setValue(self._to_pos(r, r["spin"].value()))
        r["slider"].blockSignals(False)

    def _from_slider(self, r, pos):
        v = float(f"{self._from_pos(r, pos):.4g}")
        r["spin"].blockSignals(True); r["spin"].setValue(v); r["spin"].blockSignals(False)
        self.changed.emit()

    def _from_spin(self, r):
        self._sync_slider(r)
        self.changed.emit()

    def values(self):
        return np.array([r["spin"].value() for r in self.rows])

    def set_values(self, vals, emit=True):
        for r, v in zip(self.rows, vals):
            r["spin"].blockSignals(True); r["spin"].setValue(float(v)); r["spin"].blockSignals(False)
            self._sync_slider(r)
        if emit:
            self.changed.emit()

    def reset(self):
        self.set_values(self.defaults)


class MethodList(QtWidgets.QListWidget):
    """Checkable list with colour swatches; disabled rows for inapplicable methods."""
    changed = QtCore.pyqtSignal()

    def __init__(self, methods, parent=None):
        super().__init__(parent)
        for k, desc in methods.items():
            it = QtWidgets.QListWidgetItem(theme.swatch(k), k)
            it.setToolTip(desc)
            it.setData(Qt.UserRole, k)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Unchecked)
            self.addItem(it)
        self.itemChanged.connect(lambda _: self.changed.emit())
        self.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.itemClicked.connect(self._toggle)

    def _toggle(self, it):
        pass

    def configure(self, enabled, checked):
        self.blockSignals(True)
        for i in range(self.count()):
            it = self.item(i)
            k = it.data(Qt.UserRole)
            ok = k in enabled
            it.setFlags((it.flags() | Qt.ItemIsEnabled) if ok else (it.flags() & ~Qt.ItemIsEnabled))
            it.setCheckState(Qt.Checked if ok and k in checked else Qt.Unchecked)
        self.blockSignals(False)
        self.changed.emit()

    def checked(self):
        return [self.item(i).data(Qt.UserRole) for i in range(self.count())
                if self.item(i).checkState() == Qt.Checked and self.item(i).flags() & Qt.ItemIsEnabled]

    def set_checked(self, names):
        self.blockSignals(True)
        for i in range(self.count()):
            it = self.item(i)
            if it.flags() & Qt.ItemIsEnabled:
                it.setCheckState(Qt.Checked if it.data(Qt.UserRole) in names else Qt.Unchecked)
        self.blockSignals(False)
        self.changed.emit()

    def fit_height(self):
        h = sum(self.sizeHintForRow(i) for i in range(self.count())) + 2 * self.frameWidth() + 4
        self.setFixedHeight(min(h, 340))


class ResultsTable(QtWidgets.QTableWidget):
    def __init__(self, headers, parent=None):
        super().__init__(0, len(headers), parent)
        self.headers = headers
        self.setHorizontalHeaderLabels(headers)
        hh = self.horizontalHeader()
        hh.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        hh.setStretchLastSection(True)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(22)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setShowGrid(False)
        self.setAlternatingRowColors(False)

    def set_rows(self, rows, keys=None, flags=None):
        """rows: list of lists; keys: method name per row for the colour swatch;
        flags: per-row dict(col -> 'good'|'bad'|'stale')."""
        self.setSortingEnabled(False)
        self.setRowCount(len(rows))
        muted = QtGui.QColor(theme.current["muted"])
        for i, row in enumerate(rows):
            fl = (flags or [{}] * len(rows))[i] or {}
            for j, v in enumerate(row):
                if isinstance(v, (float, np.floating)):
                    txt = "—" if not np.isfinite(v) else f"{v:.4g}"
                    it = QtWidgets.QTableWidgetItem(txt)
                    it.setData(Qt.UserRole, float(v))
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                else:
                    it = QtWidgets.QTableWidgetItem(str(v))
                if j == 0 and keys is not None and keys[i]:
                    it.setIcon(theme.swatch(keys[i]))
                tag = fl.get(j) or fl.get("row")
                if tag == "good":
                    it.setForeground(QtGui.QColor("#16a34a"))
                elif tag == "bad":
                    it.setForeground(QtGui.QColor("#dc2626"))
                elif tag == "warn":
                    it.setForeground(QtGui.QColor("#d97706"))
                elif tag == "stale":
                    it.setForeground(muted)
                self.setItem(i, j, it)

    def to_csv(self, path):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(self.headers)
            for i in range(self.rowCount()):
                w.writerow([self.item(i, j).text() if self.item(i, j) else ""
                            for j in range(self.columnCount())])


# =========================================================================== plotting helpers
def pen(name_or_color, width=1.6, style=Qt.SolidLine, alpha=255):
    c = QtGui.QColor(theme.color(name_or_color) if not str(name_or_color).startswith("#")
                     else name_or_color)
    c.setAlpha(alpha)
    p = pg.mkPen(c, width=width)
    p.setStyle(style)
    p.setCosmetic(True)
    return p


def truth_pen(width=2.0):
    return pg.mkPen(theme.current["truth"], width=width)


def method_style(name):
    return Qt.DashLine if "+" in name else Qt.SolidLine


def new_plot(glw, row, col=0, title=None, xlabel=None, ylabel=None, legend=False, xvalues=True):
    """PlotItem at (row + 1, col): row 0 of every layout is reserved for a legend strip,
    so legends never cover data. legend=True attaches the strip to this plot."""
    p = glw.addPlot(row=row + 1, col=col)
    p.showGrid(x=True, y=True, alpha=0.18)
    for ax in ("left", "bottom"):
        p.getAxis(ax).enableAutoSIPrefix(False)
    if title:
        p.setTitle(title, size="9pt", color=theme.current["plot_fg"])
    if xlabel:
        p.setLabel("bottom", xlabel)
    if ylabel:
        p.setLabel("left", ylabel)
    if not xvalues:
        p.getAxis("bottom").setStyle(showValues=False)
        p.getAxis("bottom").setHeight(8)
    p.getAxis("left").setWidth(56)
    glw.ci.layout.setRowStretchFactor(row + 1, 1)
    if legend:
        lg = pg.LegendItem(colCount=12, labelTextSize="8pt", horSpacing=14, verSpacing=-6)
        glw.addItem(lg, row=0, col=0, colspan=2)
        lg.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        lg.setMaximumHeight(46)
        glw.ci.layout.setRowStretchFactor(0, 0)
        p.legend = lg
    return p


_SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def log_x(x, *plots):
    """Plot against log10(x) with decade labels 10ⁿ (pyqtgraph's own log axis labels every
    minor tick, which is unreadable over several decades). Returns the transformed x."""
    lx = np.log10(np.asarray(x, float))
    lo, hi = int(np.floor(lx.min())), int(np.ceil(lx.max()))
    major = [(k, f"10{str(k).translate(_SUP)}") for k in range(lo, hi + 1)]
    minor = [(k + np.log10(m), "") for k in range(lo, hi) for m in range(2, 10)]
    for q in plots:
        q.getAxis("bottom").setTicks([major, minor])
    return lx


def log_ticks(lo, hi):
    """Tick list for a log10 axis: decades labelled 10ⁿ, plus 2 and 5 labelled when the span is
    short."""
    lo, hi = int(np.floor(lo)), int(np.ceil(hi))
    short = hi - lo <= 2
    major, minor = [], []
    for k in range(lo, hi + 1):
        major.append((k, f"10{str(k).translate(_SUP)}" if not short else f"{10.0 ** k:g}"))
        for m in range(2, 10):
            v = k + np.log10(m)
            if v <= hi:
                (major if short and m in (2, 5) else minor).append((v, f"{m * 10.0 ** k:g}" if short else ""))
    return [major, minor]


def log_y(y, *plots):
    """Plot against log10(y) with readable ticks; returns the transformed y."""
    ly = np.log10(np.maximum(np.asarray(y, float), 1e-300))
    fin = ly[np.isfinite(ly)]
    if fin.size:
        for q in plots:
            q.getAxis("left").setTicks(log_ticks(fin.min(), fin.max()))
    return ly


def no_si(plot):
    for ax in ("left", "bottom"):
        plot.getAxis(ax).enableAutoSIPrefix(False)


def band(plot, x, lo, hi, name, alpha=38):
    c1 = pg.PlotDataItem(x, lo, pen=None)
    c2 = pg.PlotDataItem(x, hi, pen=None)
    f = pg.FillBetweenItem(c1, c2, brush=pg.mkBrush(theme.qcolor(name, alpha)))
    plot.addItem(f)
    return f


def hband(plot, lo, hi, alpha=40):
    c = QtGui.QColor(theme.current["band"]); c.setAlpha(alpha)
    r = pg.LinearRegionItem(values=(lo, hi), orientation="horizontal", movable=False,
                            brush=pg.mkBrush(c), pen=pg.mkPen(None))
    r.setZValue(-10)
    plot.addItem(r)
    return r


def message(glw, text):
    glw.clear()
    lab = glw.addLabel(text, color=theme.current["muted"], size="10pt")
    return lab


def ellipse(m, P, nsig=2.0, n=72):
    w, V = np.linalg.eigh(0.5 * (P + P.T))
    w = np.sqrt(np.maximum(w, 0.0))
    t = np.linspace(0, 2 * np.pi, n)
    pts = (V * (nsig * w)) @ np.vstack([np.cos(t), np.sin(t)])
    return m[0] + pts[0], m[1] + pts[1]


def bar_ticks(plot, names):
    plot.getAxis("bottom").setTicks([[(i, n) for i, n in enumerate(names)]])


class PlotView(QtWidgets.QWidget):
    """A view = optional header row + one GraphicsLayoutWidget, redrawn lazily.

    Reusable views keep their PlotItems between redraws while `begin(key)` sees the same
    layout key: only the data items are replaced, which is ~3× faster than rebuilding the
    plots and keeps the user's zoom while estimates update live."""

    def __init__(self, title, draw, parent=None, reusable=False):
        super().__init__(parent)
        self.title, self.draw_fn, self.reusable = title, draw, reusable
        self.dirty = True
        self._key, self._plots = None, None
        lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(6, 6, 6, 6); lay.setSpacing(4)
        self.header = QtWidgets.QHBoxLayout(); self.header.setSpacing(8)
        lay.addLayout(self.header)
        self.glw = pg.GraphicsLayoutWidget()
        lay.addWidget(self.glw, 1)

    def begin(self, key):
        """Return the cached plots (cleared of data) if the layout key is unchanged, else
        clear the canvas and return None (the caller builds plots and calls keep())."""
        if self._plots is not None and key == self._key:
            for p in self._plots:
                p.clear()
            return self._plots
        self.glw.clear()
        self._key, self._plots = key, None
        return None

    def keep(self, plots):
        self._plots = list(plots)

    def redraw(self):
        self.glw.setBackground(theme.current["plot_bg"])
        if not self.reusable:
            self.glw.clear()
        try:
            self.draw_fn(self.glw)
            limit_zoom(self.glw)
        except _Empty as e:
            self._key = self._plots = None
            message(self.glw, str(e))
        except Exception as e:  # keep the GUI alive, show why
            self._key = self._plots = None
            message(self.glw, f"cannot draw: {type(e).__name__}: {e}")
            traceback.print_exc()
        self.dirty = False


def limit_zoom(glw, pad=0.02):
    """Stop zooming or panning out past the data (or image) extent in every plot of a canvas."""
    for it in glw.scene().items():
        if not isinstance(it, pg.ViewBox):
            continue
        try:
            (x0, x1), (y0, y1) = it.childrenBounds()
        except (TypeError, ValueError):
            continue
        if None in (x0, x1, y0, y1) or not np.all(np.isfinite([x0, x1, y0, y1])):
            continue
        dx, dy = max(x1 - x0, 1e-12) * pad, max(y1 - y0, 1e-12) * pad
        it.setLimits(xMin=x0 - dx, xMax=x1 + dx, yMin=y0 - dy, yMax=y1 + dy,
                     maxXRange=x1 - x0 + 2 * dx, maxYRange=y1 - y0 + 2 * dy)


class _Empty(Exception):
    """Raised by a draw function when there is nothing to show yet (message shown instead)."""


Empty = _Empty


# =========================================================================== equations
def _tex_fix(tex):
    import re
    return re.sub(r"\\(mathbf|mathrm|mathit|boldsymbol)\s+([A-Za-z])", r"\\\1{\2}", tex.strip().strip("$"))


_PATH_CACHE = {}


def tex_path(tex):
    """Vector outline of a mathtext string (y flipped to Qt's y-down), or None if it fails."""
    if tex in _PATH_CACHE:
        return _PATH_CACHE[tex]
    import matplotlib
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    try:
        with matplotlib.rc_context({"mathtext.fontset": "cm"}):
            tp = TextPath((0, 0), f"${_tex_fix(tex)}$", size=20, prop=FontProperties(family="DejaVu Sans"))
            verts, codes = tp.vertices, tp.codes
    except Exception:
        _PATH_CACHE[tex] = None
        return None
    path = QtGui.QPainterPath()
    i = 0
    pt = lambda k: QtCore.QPointF(verts[k, 0], -verts[k, 1])
    while i < len(codes):
        c = codes[i]
        if c == 1:
            path.moveTo(pt(i)); i += 1
        elif c == 2:
            path.lineTo(pt(i)); i += 1
        elif c == 3:
            path.quadTo(pt(i), pt(i + 1)); i += 2
        elif c == 4:
            path.cubicTo(pt(i), pt(i + 1), pt(i + 2)); i += 3
        else:
            path.closeSubpath(); i += 1
    path.setFillRule(Qt.WindingFill)
    _PATH_CACHE[tex] = path
    return path


class EquationView(QtWidgets.QWidget):
    """Equations drawn in the program as vector glyph outlines (matplotlib mathtext -> QPainterPath):
    sharp at any DPI, coloured from the theme, no images."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lines = []
        self._note = ""
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    def set_equations(self, lines, note=""):
        if isinstance(lines, str):
            lines = [lines]
        self._lines = [(tex_path(t), t) for t in lines if t and t.strip()]
        self._note = note
        self.setMinimumHeight(self.heightForWidth(max(self.width(), 240)))
        self.updateGeometry()
        self.update()

    def hasHeightForWidth(self):
        return True

    def _layout(self, width):
        fm = QtGui.QFontMetricsF(self.font())
        W = max(width - 12, 40)
        paths = [(p, t, p.boundingRect() if p is not None else QtCore.QRectF()) for p, t in self._lines]
        widest = max([b.width() for p, _, b in paths if p is not None and not p.isEmpty()] or [1.0])
        k = min(fm.height() * 1.25 / 20.0, W / widest)
        gap = fm.height() * 0.45
        rows, y = [], 6.0
        for p, t, b in paths:
            h = b.height() * k if p is not None and not p.isEmpty() else fm.height()
            rows.append((p, t, b, y, h))
            y += h + gap
        note_h = 0.0
        if self._note:
            note_h = fm.boundingRect(QtCore.QRectF(0, 0, W, 1e4), Qt.TextWordWrap, self._note).height()
        return rows, k, y, note_h, W

    def heightForWidth(self, width):
        _, _, y, note_h, _ = self._layout(width)
        return int(y + note_h + 6)

    def sizeHint(self):
        return QtCore.QSize(240, self.heightForWidth(max(self.width(), 240)))

    def minimumSizeHint(self):
        return QtCore.QSize(80, self.heightForWidth(max(self.width(), 80)))

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.setMinimumHeight(self.heightForWidth(self.width()))

    def paintEvent(self, _):
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing)
        ink = QtGui.QColor(theme.current["text"])
        rows, k, y, _, W = self._layout(self.width())
        for path, tex, br, y0, h in rows:
            if path is None or path.isEmpty():
                qp.setPen(ink)
                qp.drawText(QtCore.QRectF(6, y0, W, h), Qt.AlignVCenter | Qt.AlignLeft, tex)
                continue
            qp.save()
            qp.translate(6 - br.left() * k, y0 - br.top() * k)
            qp.scale(k, k)
            qp.fillPath(path, ink)
            qp.restore()
        if self._note:
            col = QtGui.QColor(theme.current["muted"])
            qp.setPen(col)
            qp.drawText(QtCore.QRectF(6, y, W, 1e4), Qt.AlignTop | Qt.AlignLeft | Qt.TextWordWrap, self._note)
        qp.end()


# =========================================================================== form helpers
def dspin(lo, hi, val, step=0.1, decimals=2, suffix=""):
    s = QtWidgets.QDoubleSpinBox(); s.setRange(lo, hi); s.setDecimals(decimals)
    s.setValue(val); s.setSingleStep(step); s.setSuffix(suffix)
    s.setKeyboardTracking(False); s.setAccelerated(True)
    return s


def combo(items, current=None):
    c = QtWidgets.QComboBox()
    for it in items:
        c.addItem(str(it))
    if current is not None:
        c.setCurrentText(str(current))
    return c


class Slider(QtWidgets.QWidget):
    """Labelled slider + spin box for one float, linear or log; emits `changed` while dragging."""
    changed = QtCore.pyqtSignal(float)
    N = 1000

    def __init__(self, lo, hi, val, log=False, decimals=2, suffix="", parent=None):
        super().__init__(parent)
        self.lo, self.hi, self.log = float(lo), float(hi), bool(log)
        h = QtWidgets.QHBoxLayout(self); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(6)
        self.sl = QtWidgets.QSlider(Qt.Horizontal); self.sl.setRange(0, self.N)
        self.sp = dspin(lo, hi, val, (hi - lo) / 100 if not log else val / 10, decimals, suffix)
        self.sp.setFixedWidth(92)
        h.addWidget(self.sl, 1); h.addWidget(self.sp)
        self._sync()
        self.sl.valueChanged.connect(self._from_slider)
        self.sp.valueChanged.connect(self._from_spin)

    def _to_pos(self, v):
        if self.log:
            u = (math.log(max(v, self.lo)) - math.log(self.lo)) / (math.log(self.hi) - math.log(self.lo))
        else:
            u = (v - self.lo) / (self.hi - self.lo)
        return int(round(min(max(u, 0.0), 1.0) * self.N))

    def _from_pos(self, pos):
        u = pos / self.N
        if self.log:
            return math.exp(math.log(self.lo) + u * (math.log(self.hi) - math.log(self.lo)))
        return self.lo + u * (self.hi - self.lo)

    def _sync(self):
        self.sl.blockSignals(True); self.sl.setValue(self._to_pos(self.sp.value())); self.sl.blockSignals(False)

    def _from_slider(self, pos):
        self.sp.blockSignals(True); self.sp.setValue(self._from_pos(pos)); self.sp.blockSignals(False)
        self.changed.emit(self.value())

    def _from_spin(self, _):
        self._sync()
        self.changed.emit(self.value())

    def value(self):
        return float(self.sp.value())

    def setValue(self, v, emit=False):
        self.sp.blockSignals(True); self.sp.setValue(float(v)); self.sp.blockSignals(False)
        self._sync()
        if emit:
            self.changed.emit(self.value())


def colorbar(glw, image, row, col, label=""):
    """Attach a pyqtgraph colour bar to an ImageItem."""
    cb = pg.ColorBarItem(values=image.getLevels() if image.image is not None else (0, 1),
                         colorMap=image_cmap(), interactive=False, width=12, label=label)
    cb.setImageItem(image)
    glw.addItem(cb, row=row, col=col)
    return cb


def image_cmap(name="viridis", reverse=False):
    try:
        cm = pg.colormap.get(name)
    except Exception:
        cm = pg.colormap.getFromMatplotlib(name)
    if reverse:
        cm.reverse()
    return cm
