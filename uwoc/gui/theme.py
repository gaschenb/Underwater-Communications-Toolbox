"""Light / dark themes: Qt palette + style sheet + pyqtgraph defaults."""
from __future__ import annotations

import pyqtgraph as pg
from PyQt5 import QtGui, QtWidgets

THEMES = {
    "light": dict(
        bg="#f3f4f6", panel="#ffffff", border="#e2e5ea", text="#1f2937", muted="#6b7280",
        accent="#2563eb", accent_hover="#1d4ed8", accent_text="#ffffff", input="#ffffff",
        sel="#dbeafe", nav="#111827", nav_text="#cbd5e1", nav_sel="#1e293b",
        plot_bg="#ffffff", plot_fg="#374151", truth="#111111", band="#9ca3af",
        heat=((255, 255, 255), (191, 219, 254), (37, 99, 235), (30, 27, 75))),
    "dark": dict(
        bg="#15171c", panel="#1d2026", border="#2d313a", text="#e5e7eb", muted="#9ca3af",
        accent="#3b82f6", accent_hover="#60a5fa", accent_text="#ffffff", input="#23262d",
        sel="#1e3a8a", nav="#0b0d11", nav_text="#9ca3af", nav_sel="#1f2937",
        plot_bg="#1a1d23", plot_fg="#d1d5db", truth="#f9fafb", band="#6b7280",
        heat=((26, 29, 35), (30, 58, 138), (59, 130, 246), (219, 234, 254))),
}

# series colours (Tableau-like, readable on both backgrounds)
PALETTE = {
    "a": "#e15759", "b": "#4e79a7", "c": "#111111", "bb": "#b07aa1", "Kd": "#59a14f",
    "beer": "#e15759", "effective": "#4e79a7", "mc": "#f28e2b", "ballistic": "#e15759", "scattered": "#4e79a7",
    "tx": "#8c8c8c", "rx": "#4e79a7", "filt": "#f28e2b", "clean": "#59a14f", "bg": "#b07aa1",
    "thr": "#e15759", "err": "#e15759", "ok": "#59a14f", "fade": "#9c755f", "theory": "#8c8c8c",
    "sim": "#4e79a7", "pilot": "#edc948", "erased": "#bab0ac", "exposure": "#f28e2b",
    "shot": "#4e79a7", "thermal": "#f28e2b", "background": "#b07aa1", "dark": "#76b7b2", "isi": "#e15759",
    "PIN": "#4e79a7", "APD": "#f28e2b", "SiPM": "#59a14f", "PMT": "#e15759",
    "R": "#e15759", "G": "#59a14f", "B": "#4e79a7",
    "c0": "#4e79a7", "c1": "#f28e2b", "c2": "#59a14f", "c3": "#e15759", "c4": "#b07aa1",
    "c5": "#76b7b2", "c6": "#edc948", "c7": "#9c755f", "c8": "#ff9da7", "c9": "#bab0ac",
}
SERIES = ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"]

current = dict(THEMES["light"], name="light")


def color(name):
    if name == "total" or name == "true":
        return current.get("truth", "#111111")
    return PALETTE.get(name, PALETTE.get(name.split("+")[0].strip(), "#7f7f7f"))


def qcolor(name, alpha=255):
    c = QtGui.QColor(color(name))
    c.setAlpha(alpha)
    return c


def heat_lut():
    stops = current["heat"]
    cmap = pg.ColorMap([0.0, 0.25, 0.65, 1.0], [tuple(c) for c in stops])
    return cmap.getLookupTable(0.0, 1.0, 256)


def swatch(name, size=10):
    pm = QtGui.QPixmap(size, size)
    pm.fill(qcolor(name))
    return QtGui.QIcon(pm)


def style_sheet(t):
    return f"""
    QWidget {{ color: {t['text']}; font-size: 9pt; }}
    QMainWindow, QStackedWidget, #page {{ background: {t['bg']}; }}
    QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; border: none; }}
    #card {{ background: {t['panel']}; border: 1px solid {t['border']}; border-radius: 8px; }}
    #cardTitle {{ font-weight: 600; font-size: 9pt; color: {t['text']}; }}
    #muted {{ color: {t['muted']}; }}
    #readout {{ color: {t['muted']}; font-family: Consolas, 'DejaVu Sans Mono', monospace; }}
    #nav {{ background: {t['nav']}; border: none; outline: none; padding-top: 8px; }}
    #nav::item {{ color: {t['nav_text']}; padding: 10px 14px; border-left: 3px solid transparent; }}
    #nav::item:selected {{ background: {t['nav_sel']}; color: #ffffff;
                           border-left: 3px solid {t['accent']}; }}
    #nav::item:hover:!selected {{ background: {t['nav_sel']}; }}
    #brand {{ background: {t['nav']}; color: #ffffff; font-weight: 700; font-size: 11pt;
              padding: 14px 14px 4px 14px; }}
    #brandSub {{ background: {t['nav']}; color: {t['nav_text']}; font-size: 8pt;
                 padding: 0 14px 10px 14px; }}
    QPushButton {{ background: {t['panel']}; border: 1px solid {t['border']}; border-radius: 6px;
                   padding: 5px 12px; }}
    QPushButton:hover {{ border-color: {t['accent']}; }}
    QPushButton:disabled {{ color: {t['muted']}; }}
    QPushButton#primary {{ background: {t['accent']}; color: {t['accent_text']}; border: none;
                           font-weight: 600; padding: 7px 14px; }}
    QPushButton#primary:hover {{ background: {t['accent_hover']}; }}
    QToolBar {{ background: {t['panel']}; border: none; border-bottom: 1px solid {t['border']};
                spacing: 4px; padding: 4px 8px; }}
    QToolButton {{ border: 1px solid transparent; border-radius: 6px; padding: 4px 9px; }}
    QToolButton:hover {{ background: {t['bg']}; border-color: {t['border']}; }}
    QToolButton:checked {{ background: {t['sel']}; border-color: {t['accent']}; }}
    QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{ background: {t['input']};
        border: 1px solid {t['border']}; border-radius: 5px; padding: 3px 6px; min-height: 20px; }}
    QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{
        border-color: {t['accent']}; }}
    QComboBox QAbstractItemView {{ background: {t['panel']}; selection-background-color: {t['sel']};
        selection-color: {t['text']}; border: 1px solid {t['border']}; }}
    QListWidget {{ background: {t['input']}; border: 1px solid {t['border']}; border-radius: 5px; }}
    QListWidget::item {{ padding: 2px; }}
    QListWidget::item:selected {{ background: {t['sel']}; color: {t['text']}; }}
    QTabWidget::pane {{ border: 1px solid {t['border']}; border-radius: 6px; background: {t['panel']};
                        top: -1px; }}
    QTabBar::tab {{ background: transparent; padding: 6px 14px; border: none;
                    border-bottom: 2px solid transparent; color: {t['muted']}; }}
    QTabBar::tab:selected {{ color: {t['text']}; border-bottom: 2px solid {t['accent']}; }}
    QTabBar::tab:hover:!selected {{ color: {t['text']}; }}
    QTabBar::tab:disabled {{ color: {t['border']}; }}
    QTableWidget {{ background: {t['panel']}; gridline-color: {t['border']};
                    border: 1px solid {t['border']}; border-radius: 6px;
                    selection-background-color: {t['sel']}; selection-color: {t['text']}; }}
    QHeaderView::section {{ background: {t['bg']}; border: none; border-bottom: 1px solid {t['border']};
                            padding: 4px 6px; font-weight: 600; }}
    QPlainTextEdit {{ background: {t['panel']}; border: 1px solid {t['border']}; border-radius: 6px;
                      font-family: Consolas, 'DejaVu Sans Mono', monospace; font-size: 8pt; }}
    QSlider::groove:horizontal {{ height: 4px; background: {t['border']}; border-radius: 2px; }}
    QSlider::sub-page:horizontal {{ background: {t['accent']}; border-radius: 2px; }}
    QSlider::handle:horizontal {{ background: {t['panel']}; border: 2px solid {t['accent']};
                                  width: 12px; margin: -6px 0; border-radius: 7px; }}
    QProgressBar {{ border: 1px solid {t['border']}; border-radius: 4px; background: {t['bg']};
                    max-height: 8px; text-align: center; font-size: 1px; }}
    QProgressBar::chunk {{ background: {t['accent']}; border-radius: 3px; }}
    QStatusBar {{ background: {t['panel']}; border-top: 1px solid {t['border']}; color: {t['muted']}; }}
    QSplitter::handle {{ background: {t['bg']}; }}
    QCheckBox {{ spacing: 6px; }}
    QToolTip {{ background: {t['panel']}; color: {t['text']}; border: 1px solid {t['border']}; }}
    """


def apply(app, name="light"):
    t = THEMES[name]
    current.clear(); current.update(t); current["name"] = name
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    for role, key in [(QtGui.QPalette.Window, "bg"), (QtGui.QPalette.Base, "input"),
                      (QtGui.QPalette.AlternateBase, "bg"), (QtGui.QPalette.Button, "panel"),
                      (QtGui.QPalette.Text, "text"), (QtGui.QPalette.WindowText, "text"),
                      (QtGui.QPalette.ButtonText, "text"), (QtGui.QPalette.Highlight, "accent"),
                      (QtGui.QPalette.ToolTipBase, "panel"), (QtGui.QPalette.ToolTipText, "text")]:
        pal.setColor(role, QtGui.QColor(t[key]))
    pal.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#ffffff"))
    app.setPalette(pal)
    app.setStyleSheet(style_sheet(t))
    pg.setConfigOptions(antialias=True, background=t["plot_bg"], foreground=t["plot_fg"],
                        imageAxisOrder="row-major")


def font_ui():
    f = QtWidgets.QApplication.font()
    return f
