from __future__ import annotations

import os


THEME_COLORS = {
    "special": "#C65A66",
    "danger": "#8E2F3D",
    "warning": "#73E6CB",
    "bg": "#0A3C30",
    "title_bar": "#041E17",
    "panel": "#00674F",
    "panel_alt": "#0D5745",
    "card": "#0E4B3D",
    "accent": "#3EBB9E",
    "accent_hover": "#00674F",
    "accent_pressed": "#00674F",
    "success": "#73E6CB",
    "graph_bg": "#06281F",
    "graph_grid": "#165A49",
    "graph_axis": "#73E6CB",
    "graph_zero": "#3EBB9E",
    "border": "#3EBB9E",
    "text": "#F2FFFB",
    "muted": "#B6F3E6",
    "disabled": "#7DB8AB",
    "input_bg": "#0D4A3B",
    "selection": "#00674F",
}


def apply_dark_title_bar(window):
    """Request a dark native title bar on Windows where supported."""
    try:
        import ctypes
        import sys
        from ctypes import wintypes
    except Exception:
        return False

    if sys.platform != "win32":
        return False

    try:
        hwnd = int(window.winId())
    except Exception:
        return False

    def _hex_to_colorref(hex_color):
        value = (hex_color or "").strip().lstrip("#")
        if len(value) != 6:
            return None
        try:
            r = int(value[0:2], 16)
            g = int(value[2:4], 16)
            b = int(value[4:6], 16)
        except ValueError:
            return None
        return (b << 16) | (g << 8) | r

    use_dark = ctypes.c_int(1)
    use_dark_size = ctypes.sizeof(use_dark)
    attrs = (20, 19)
    enabled_dark = False
    for attr in attrs:
        try:
            result = ctypes.windll.dwmapi.DwmSetWindowAttribute(
                wintypes.HWND(hwnd),
                wintypes.DWORD(attr),
                ctypes.byref(use_dark),
                wintypes.DWORD(use_dark_size),
            )
            if result == 0:
                enabled_dark = True
                break
        except Exception:
            continue

    caption_color = _hex_to_colorref(THEME_COLORS["title_bar"])
    text_color = _hex_to_colorref(THEME_COLORS["text"])
    if caption_color is not None:
        try:
            caption_val = ctypes.c_int(caption_color)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                wintypes.HWND(hwnd),
                wintypes.DWORD(35),
                ctypes.byref(caption_val),
                wintypes.DWORD(ctypes.sizeof(caption_val)),
            )
        except Exception:
            pass
    if text_color is not None:
        try:
            text_val = ctypes.c_int(text_color)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                wintypes.HWND(hwnd),
                wintypes.DWORD(36),
                ctypes.byref(text_val),
                wintypes.DWORD(ctypes.sizeof(text_val)),
            )
        except Exception:
            pass
    return enabled_dark


def _default_font_family() -> str:
    if os.name == "nt":
        return "Segoe UI"
    return "Sans Serif"


def app_stylesheet(font_size: int = 16) -> str:
    c = THEME_COLORS
    default_font = _default_font_family()
    return f"""
QWidget {{
    background-color: {c['bg']};
    color: {c['text']};
    font-family: "{default_font}";
    font-size: {int(font_size)}px;
}}
QMainWindow, QDialog {{
    background-color: {c['bg']};
}}
QLabel {{
    color: {c['text']};
    background: transparent;
}}
QGroupBox {{
    background-color: {c['panel']};
    border: 1px solid {c['border']};
    border-radius: 12px;
    margin-top: 14px;
    padding: 14px 14px 12px 14px;
    font-weight: 700;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 6px;
    color: {c['text']};
}}
QFrame[card="true"] {{
    background-color: {c['panel']};
    border: 1px solid {c['border']};
    border-radius: 8px;
}}
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QListWidget {{
    background-color: {c['input_bg']};
    color: {c['text']};
    border: 1px solid {c['border']};
    border-radius: 8px;
    padding: 5px 8px;
    selection-background-color: {c['selection']};
}}
QComboBox {{
    background-color: {c['input_bg']};
    color: {c['text']};
    border: 1px solid {c['border']};
    border-radius: 8px;
    padding: 3px 34px 3px 8px;
    min-height: 32px;
    selection-background-color: {c['selection']};
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QListWidget:focus {{
    border: 1px solid {c['accent_hover']};
}}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 28px;
    border: none;
    border-left: 1px solid {c['border']};
    background-color: {c['panel_alt']};
    border-top-right-radius: 8px;
    border-bottom-right-radius: 8px;
}}
QComboBox QLineEdit {{
    background: transparent;
    border: none;
    padding: 0px 4px 0px 0px;
    margin: 0px;
    color: {c['text']};
}}
QComboBox QAbstractItemView, QListWidget {{
    background-color: {c['input_bg']};
    color: {c['text']};
    selection-background-color: {c['selection']};
    border: 1px solid {c['border']};
}}
QPushButton {{
    background-color: {c['accent']};
    color: {c['text']};
    border: 1px solid {c['accent']};
    border-radius: 8px;
    padding: 3px 12px;
    min-height: 28px;
    font-weight: 700;
}}
QPushButton:hover:!disabled {{
    background-color: {c['accent_hover']};
    color: {c['text']};
    border-color: {c['accent']};
}}
QPushButton:pressed:!disabled {{
    background-color: {c['panel_alt']};
    border-color: {c['special']};
    color: {c['text']};
}}
QPushButton:checked:!disabled {{
    background-color: {c['accent_pressed']};
    border-color: {c['accent']};
}}
QPushButton:disabled {{
    background-color: {c['panel_alt']};
    color: {c['disabled']};
    border-color: {c['accent']};
}}
QCheckBox, QRadioButton {{
    spacing: 8px;
    background: transparent;
}}
QHeaderView::section {{
    background-color: {c['panel_alt']};
    color: {c['text']};
    border: 1px solid {c['border']};
    padding: 6px;
    font-weight: 700;
}}
QTableWidget {{
    background-color: {c['input_bg']};
    color: {c['text']};
    gridline-color: {c['border']};
    border: 1px solid {c['border']};
}}
QProgressBar {{
    border: 1px solid {c['border']};
    border-radius: 7px;
    background-color: {c['panel_alt']};
    color: {c['text']};
    text-align: center;
}}
QProgressBar::chunk {{
    background-color: {c['success']};
    border-radius: 6px;
}}
QScrollArea {{
    border: 1px solid {c['border']};
    border-radius: 10px;
    background-color: {c['panel']};
}}
QScrollBar:vertical {{
    border: none;
    background: {c['bg']};
    width: 8px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {c['accent']};
    min-height: 28px;
    border-radius: 4px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
    background: transparent;
    border: none;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: {c['bg']};
}}
QScrollBar:horizontal {{
    border: none;
    background: {c['bg']};
    height: 8px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {c['accent']};
    min-width: 28px;
    border-radius: 4px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
    background: transparent;
    border: none;
}}
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: {c['bg']};
}}
QToolTip {{
    background-color: {c['panel_alt']};
    color: {c['text']};
    border: 1px solid {c['border']};
    padding: 6px 8px;
}}
"""


def apply_dark_theme(app, font_size: int = 16) -> None:
    font = app.font()
    font.setFamily(_default_font_family())
    font.setPointSize(max(10, int(round(font_size * 0.62))))
    app.setFont(font)
    app.setStyleSheet(app_stylesheet(font_size=font_size))


def themed_button_style(kind: str = "accent") -> str:
    c = THEME_COLORS
    if kind == "danger":
        bg = c["danger"]
        hover_bg = "#A53B4B"
        hover_border = c["special"]
        pressed_bg = "#6F2430"
        pressed_border = c["special"]
        border = c["danger"]
    elif kind == "success":
        bg = c["success"]
        hover_bg = c["accent_hover"]
        hover_border = c["accent"]
        pressed_bg = c["panel_alt"]
        pressed_border = c["special"]
        border = c["accent"]
    elif kind == "muted":
        bg = c["panel_alt"]
        hover_bg = c["accent_hover"]
        hover_border = c["accent"]
        pressed_bg = c["panel_alt"]
        pressed_border = c["special"]
        border = c["border"]
    else:
        bg = c["accent"]
        hover_bg = c["accent_hover"]
        hover_border = c["accent"]
        pressed_bg = c["panel_alt"]
        pressed_border = c["special"]
        border = c["accent"]
    return (
        f"QPushButton {{"
        f" background-color: {bg}; color: {c['text']}; font-weight: 700;"
        f" border: 1px solid {border}; border-radius: 8px; padding: 3px 12px; min-height: 28px;"
        f"}}"
        f" QPushButton:hover:!disabled {{ background-color: {hover_bg}; border-color: {hover_border}; color: {c['text']}; }}"
        f" QPushButton:pressed:!disabled {{ background-color: {pressed_bg}; border-color: {pressed_border}; color: {c['text']}; }}"
        f" QPushButton:checked:!disabled {{ background-color: {c['accent_pressed']}; border-color: {border}; }}"
        f" QPushButton:disabled {{"
        f" background-color: {c['panel_alt']}; color: {c['disabled']}; border-color: {c['accent']};"
        f"}}"
    )


def themed_label_style(kind: str = "muted") -> str:
    c = THEME_COLORS
    if kind == "danger":
        color = c["danger"]
    elif kind == "success":
        color = c["success"]
    elif kind == "warning":
        color = c["warning"]
    else:
        color = c["muted"]
    return f"color: {color}; font-weight: 700;"


def themed_status_color(color_hint=None):
    hint = (color_hint or "").strip().lower()
    if hint in {"error", "danger", "bad"}:
        return THEME_COLORS["danger"]
    if hint in {"warning", "warn"}:
        return THEME_COLORS["warning"]
    if hint in {"success", "good", "ok"}:
        return THEME_COLORS["success"]
    return THEME_COLORS["muted"]
