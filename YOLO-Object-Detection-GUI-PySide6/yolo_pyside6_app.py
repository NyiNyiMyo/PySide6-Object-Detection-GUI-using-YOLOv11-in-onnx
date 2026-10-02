"""
📟 Makers - YOLO Object Detection  (PySide6 GUI, cross-platform, ONNX Runtime)

Same detection logic/config as the wxPython version. UI concept (new):

  - Animated splash screen (rounded, soft shadow, gradient, animated progress
    and mode-coloured pulse dots) while the default model loads in a thread,
    then a smooth fade-out.
  - Left sidebar: brand, vertical mode navigation (Image / Video / Webcam /
    Folder, each with its own colour) and a model card pinned to the bottom
    (current model + Change Model / Reload).
  - Workspace: a slim rainbow accent strip, an action toolbar, and a card
    with the display. Header of the display card carries the selected file.
  - Image / Folder pages: display on the left, collapsible detection settings
    + analysis on the right in a draggable splitter (2 : 1 by default).
  - Video / Webcam pages: no analysis panel; the two detection sliders sit
    side by side in a card underneath the view.
  - Status bar with a coloured status dot.

Install:
    pip install PySide6 opencv-python-headless numpy onnxruntime
    (for GPU: pip install onnxruntime-gpu)

    NOTE: prefer opencv-python-headless. The regular opencv-python wheel ships
    its own Qt plugins on Linux, which can clash with PySide6 ("xcb" errors).
    This script also removes the env var OpenCV sets for that plugin path.

Packaging with PyInstaller (--onefile):
    The model is looked up via resource_base_dir() (sys._MEIPASS when frozen)
    and runtime output (captures/) goes next to the executable via
    writable_base_dir(). Bundle the model with e.g.:
        pyinstaller --onefile --windowed --add-data "models;models" yolo_pyside6_app.py   (Windows)
        pyinstaller --onefile --windowed --add-data "models:models" yolo_pyside6_app.py   (macOS/Linux)
"""

import ast
import math
import os
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np
import cv2

try:
    from PySide6.QtCore import (Qt, QObject, Signal, Slot, QTimer, QRectF, QPointF,
                                QPropertyAnimation, QEasingCurve)
    from PySide6.QtGui import (QColor, QFont, QImage, QPainter, QPainterPath, QPen,
                               QLinearGradient, QPalette, QGuiApplication, QFontMetrics)
    from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QFrame, QLabel,
                                   QPushButton, QSlider, QSplitter, QScrollArea,
                                   QVBoxLayout, QHBoxLayout, QStackedWidget,
                                   QButtonGroup, QToolButton, QProgressBar, QStatusBar,
                                   QFileDialog, QMessageBox, QSizePolicy)
except ImportError:
    print("ERROR: PySide6 not installed!")
    print("Install with: pip install PySide6")
    raise SystemExit(1)

# OpenCV (non-headless) can leave a Qt plugin path behind that breaks PySide6
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime not installed!")
    print("Install with: pip install onnxruntime")
    print("(For GPU acceleration: pip install onnxruntime-gpu)")
    raise SystemExit(1)

# Colorful theme: deep indigo sidebar, soft light workspace, one colour per mode.
COLORS = {
    # surfaces
    'bg': '#f0f3fb',
    'bg_light': '#ffffff',
    'bg_lighter': '#f6f8fd',
    'surface': '#ffffff',
    'canvas': '#e7ecf8',
    # text
    'fg': '#1e293b',
    'fg_dim': '#64748b',
    # brand / status
    'accent': '#4f46e5',
    'accent_hover': '#4338ca',
    'accent_soft': '#eceeff',
    'success': '#15803d',
    'warning': '#c2410c',
    'error': '#dc2626',
    'border': '#d7deef',
    'danger': '#e11d48',
    'danger_hover': '#be123c',
    'green': '#16a34a',
    'green_hover': '#15803d',
    # sidebar (deep indigo -> violet, light text)
    'header': '#131a4a',
    'header_b': '#2d1a6b',
    'h_fg': '#ffffff',
    'h_dim': '#aeb9ee',
    'h_accent': '#7dd3fc',
    'h_ok': '#86efac',
    'h_err': '#fca5a5',
    'h_warn': '#fde68a',
    'h_btn': '#4a5ce6',
    # one colour per mode
    'mode_image': '#3b82f6',
    'mode_video': '#8b5cf6',
    'mode_webcam': '#14b8a6',
    'mode_folder': '#f97316',
    # disabled buttons
    'dis_bg': '#dde3f0',
    'dis_fg': '#93a0b8',
    # splash gradient
    'splash_a': '#1e3a8a',
    'splash_b': '#6d28d9',
}

# A fixed color palette (BGR) used to draw boxes per class id
BOX_PALETTE = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255),
    (255, 255, 0), (0, 128, 255), (255, 128, 0), (128, 0, 255), (0, 255, 128),
    (128, 255, 0), (255, 0, 128), (0, 128, 128), (128, 128, 0), (128, 0, 128),
    (192, 192, 192), (64, 64, 255), (64, 255, 64), (255, 64, 64), (200, 200, 0),
]

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}
PLACEHOLDER = "📟 Makers - YOLO Object Detection\n\nClick 'Image', 'Video', or 'Webcam' to start"

# Grid settings for folder inference (each cell is rendered at this size)
GRID_COLS, GRID_ROWS = 2, 2
GRID_COUNT = GRID_COLS * GRID_ROWS
CELL_W, CELL_H = 640, 480


# ============================================================
# PyInstaller-safe resource paths
# ============================================================
def resource_base_dir():
    """Base directory for bundled *read-only* resources (models/, etc).

    In a PyInstaller --onefile build, sys._MEIPASS points at the temp folder
    the exe unpacked itself into; use that so a bundled models/ folder is
    found. Running from source, use the folder this script lives in."""
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


def writable_base_dir():
    """Base directory for files this app *creates* at runtime (captures/).

    The --onefile temp extraction folder is wiped after the process exits,
    so runtime output must NOT go there. sys.executable points at the real
    .exe on disk when frozen, so write next to it instead."""
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


MODELS_DIR = resource_base_dir() / 'models'
CAPTURES_DIR = writable_base_dir() / 'captures'


# ============================================================
# ONNX inference (unchanged logic)
# ============================================================
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize + pad image to new_shape while keeping aspect ratio.
    Returns the padded image, the resize ratio, and (dw, dh) half-padding."""
    shape = im.shape[:2]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


class SimpleBox:
    """Mimics the small subset of ultralytics' Boxes API the UI code relies on."""

    def __init__(self, cls_id, conf, xyxy):
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxy = [np.array(xyxy, dtype=float)]


class SimpleResult:
    """Mimics the small subset of ultralytics' Results API the UI code relies on."""

    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class ONNXYOLO:
    """Thin ONNX Runtime wrapper that reproduces the predict() -> result interface
    the rest of the app expects, so the GUI code barely has to change."""

    def __init__(self, model_path):
        providers = []
        available = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')

        self.session = ort.InferenceSession(model_path, providers=providers)

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = inp.shape
        self.input_h = shape[2] if isinstance(shape[2], int) else 640
        self.input_w = shape[3] if isinstance(shape[3], int) else 640

        self.names = self._load_names()

    def _load_names(self):
        names = {}
        try:
            meta = self.session.get_modelmeta()
            custom = meta.custom_metadata_map
            if custom and 'names' in custom:
                parsed = ast.literal_eval(custom['names'])
                if isinstance(parsed, dict):
                    names = {int(k): v for k, v in parsed.items()}
                elif isinstance(parsed, (list, tuple)):
                    names = {i: v for i, v in enumerate(parsed)}
        except Exception:
            names = {}

        if not names:
            names = {i: f"class{i}" for i in range(1000)}
        return names

    def _postprocess(self, pred, conf_thres, iou_thres, ratio, dw, dh, orig_shape):
        orig_h, orig_w = orig_shape[:2]

        pred = pred[0]
        # Normalize to shape (num_anchors, 4 + num_classes)
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T

        boxes_cxcywh = pred[:, :4]
        class_scores = pred[:, 4:]

        if class_scores.shape[1] == 0:
            return [], [], []

        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(len(class_ids)), class_ids]

        mask = scores > conf_thres
        boxes_cxcywh = boxes_cxcywh[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if len(scores) == 0:
            return [], [], []

        cx, cy, w, h = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # Undo letterbox padding/scale to map back to original image coordinates
        x1 = (x1 - dw) / ratio
        y1 = (y1 - dh) / ratio
        x2 = (x2 - dw) / ratio
        y2 = (y2 - dh) / ratio

        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        nms_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        nms_scores = scores.tolist()

        indices = cv2.dnn.NMSBoxes(nms_boxes, nms_scores, conf_thres, iou_thres)
        if indices is None or len(indices) == 0:
            return [], [], []
        indices = np.array(indices).flatten()

        final_boxes = [[float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])] for i in indices]
        final_scores = [float(scores[i]) for i in indices]
        final_class_ids = [int(class_ids[i]) for i in indices]

        return final_boxes, final_scores, final_class_ids

    def _draw(self, img_bgr, boxes):
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls_id = box.cls[0]
            conf = box.conf[0]
            name = self.names.get(cls_id, f"class{cls_id}")
            color = BOX_PALETTE[cls_id % len(BOX_PALETTE)]

            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
            label = f"{name} {conf * 100:.1f}%"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty1 = max(0, y1 - th - baseline - 4)
            cv2.rectangle(img_bgr, (x1, ty1), (x1 + tw + 4, ty1 + th + baseline + 4), color, -1)
            cv2.putText(img_bgr, label, (x1 + 2, ty1 + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return img_bgr

    def infer(self, image_bgr, conf=0.25, iou=0.45):
        """Run detection on a BGR numpy image. Returns (SimpleResult, annotated_bgr)."""
        letter_img, ratio, (dw, dh) = letterbox(image_bgr, (self.input_h, self.input_w))
        img_rgb = cv2.cvtColor(letter_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_chw = np.transpose(img_rgb, (2, 0, 1))
        input_tensor = np.expand_dims(img_chw, 0).astype(np.float32)

        outputs = self.session.run(None, {self.input_name: input_tensor})
        pred = outputs[0]

        boxes, scores, class_ids = self._postprocess(pred, conf, iou, ratio, dw, dh, image_bgr.shape)
        simple_boxes = [SimpleBox(class_ids[i], scores[i], boxes[i]) for i in range(len(boxes))]
        result = SimpleResult(simple_boxes, self.names)

        annotated = self._draw(image_bgr.copy(), simple_boxes)
        return result, annotated


def fit_into_cell(img_bgr, w, h, bg=(45, 45, 45)):
    """Fit an image into a w x h cell (keep aspect ratio, centered)."""
    ih, iw = img_bgr.shape[:2]
    r = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * r)), max(1, int(ih * r))
    # INTER_AREA is great for shrinking but blocky/soft when enlarging
    interp = cv2.INTER_AREA if r < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=interp)
    cell = np.full((h, w, 3), bg, dtype=np.uint8)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    cell[y0:y0 + nh, x0:x0 + nw] = resized
    return cell


def build_grid(annotated_list, names_list):
    """Compose annotated BGR images into a GRID_COLS x GRID_ROWS grid (BGR)."""
    cells = []
    for img, name in zip(annotated_list, names_list):
        cell = fit_into_cell(img, CELL_W, CELL_H)
        label = name if len(name) <= 40 else name[:37] + "..."
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(cell, (0, CELL_H - th - bl - 10), (tw + 12, CELL_H), (30, 30, 30), -1)
        cv2.putText(cell, label, (6, CELL_H - bl - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(cell, (0, 0), (CELL_W - 1, CELL_H - 1), (64, 64, 64), 2)
        cells.append(cell)
    while len(cells) < GRID_COUNT:
        cells.append(np.full((CELL_H, CELL_W, 3), (45, 45, 45), dtype=np.uint8))
    rows = [np.hstack(cells[r * GRID_COLS:(r + 1) * GRID_COLS]) for r in range(GRID_ROWS)]
    return np.vstack(rows)


# ============================================================
# Qt helpers
# ============================================================
def C(name):
    """Colour hex string by COLORS key, or a raw colour string."""
    return COLORS.get(name, name)


def qc(name, alpha=None):
    col = QColor(C(name))
    if alpha is not None:
        col.setAlpha(alpha)
    return col


def make_font(px=14, bold=False):
    f = QFont(QApplication.font())
    f.setPixelSize(px)
    f.setBold(bold)
    return f


def recolor(label, color):
    label.setStyleSheet(f"color:{C(color)}; background:transparent;")


def static_text(parent, text, size=14, bold=False, color='fg', cls=QLabel):
    lbl = cls(text, parent) if parent is not None else cls(text)
    lbl.setFont(make_font(size, bold))
    recolor(lbl, color)
    return lbl


class ElidedLabel(QLabel):
    """QLabel that elides its text in the middle instead of growing."""

    def __init__(self, text="", parent=None):
        super().__init__(parent)
        self._full = ""
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setMinimumWidth(40)
        self.setText(text)

    def setText(self, text):
        self._full = text
        self._apply()

    def fullText(self):
        return self._full

    def _apply(self):
        fm = QFontMetrics(self.font())
        super().setText(fm.elidedText(self._full, Qt.ElideMiddle, max(10, self.width() - 2)))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._apply()


def _btn_qss(bg, fg, size, radius, pad):
    base = QColor(bg)
    hover = base.lighter(112).name()
    pressed = base.darker(118).name()
    return f"""
    QPushButton {{ background:{bg}; color:{fg}; border:none; border-radius:{radius}px;
                   padding:0 {pad}px; font-size:{size}px; font-weight:700; }}
    QPushButton:hover {{ background:{hover}; }}
    QPushButton:pressed {{ background:{pressed}; }}
    QPushButton:disabled {{ background:{C('dis_bg')}; color:{C('dis_fg')}; }}
    """


def make_button(parent, label, handler, bg='accent', fg='#ffffff', enabled=True,
                size=14, height=44, pad_x=20, radius=12):
    btn = QPushButton(label, parent)
    btn.setFixedHeight(height)
    btn.setCursor(Qt.PointingHandCursor)
    btn.setStyleSheet(_btn_qss(C(bg), C(fg), size, radius, pad_x))
    if handler:
        btn.clicked.connect(lambda _=False, h=handler: h(None))
    btn.setEnabled(enabled)
    return btn


def make_gauge(parent, fraction, color, width=120):
    g = QProgressBar(parent)
    g.setRange(0, 1000)
    g.setValue(int(max(0.0, min(1.0, fraction)) * 1000))
    g.setTextVisible(False)
    g.setFixedSize(width, 10)
    g.setStyleSheet(
        f"QProgressBar {{ background:#e2e8f4; border:none; border-radius:5px; }}"
        f"QProgressBar::chunk {{ background:{C(color)}; border-radius:5px; }}")
    return g


def build_qss():
    return f"""
    QMainWindow, QWidget#root {{ background:{C('bg')}; }}
    QToolTip {{ background:#1e293b; color:#ffffff; border:none; padding:6px 8px; }}

    QFrame#card  {{ background:{C('surface')}; border:1px solid {C('border')}; border-radius:16px; }}
    QFrame#rcard {{ background:{C('surface')}; border:1px solid {C('border')}; border-radius:12px; }}

    QScrollArea {{ background:transparent; border:none; }}
    QScrollBar:vertical {{ background:transparent; width:10px; margin:2px; }}
    QScrollBar::handle:vertical {{ background:#c3cde6; border-radius:4px; min-height:30px; }}
    QScrollBar::handle:vertical:hover {{ background:#a5b3d9; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background:none; }}

    QSlider::groove:horizontal {{ height:6px; background:#dbe3f3; border-radius:3px; }}
    QSlider::sub-page:horizontal {{ background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
                                    stop:0 #6366f1, stop:1 #ec4899); border-radius:3px; }}
    QSlider::handle:horizontal {{ background:#ffffff; border:3px solid {C('accent')};
                                  width:14px; height:14px; margin:-7px 0; border-radius:10px; }}
    QSlider::handle:horizontal:hover {{ border-color:#ec4899; }}

    QSplitter::handle {{ background:transparent; }}
    QSplitter::handle:horizontal {{ width:14px; }}
    QSplitter::handle:hover {{ background:rgba(79,70,229,14%); border-radius:5px; }}

    QStatusBar {{ background:{C('surface')}; border-top:1px solid {C('border')}; }}
    QStatusBar::item {{ border:none; }}

    QToolButton#collapse {{ border:none; background:transparent; color:{C('accent')};
                            font-size:15px; font-weight:700; text-align:left; padding:10px 14px; }}
    QToolButton#collapse:hover {{ background:{C('accent_soft')}; border-radius:12px; }}
    """


# ============================================================
# Thread -> UI bridge
# ============================================================
class UiBridge(QObject):
    """Lets worker threads schedule a callable on the UI thread."""
    call = Signal(object)

    def __init__(self):
        super().__init__()
        self.call.connect(self._run)

    @Slot(object)
    def _run(self, fn):
        fn()


# ============================================================
# Small custom widgets
# ============================================================
class BusyBar(QWidget):
    """Slim indeterminate progress bar (a gradient segment sweeping across)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(6)
        self._active = False
        self._phase = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)

    def start(self):
        self._active = True
        self._timer.start()
        self.update()

    def stop(self):
        self._active = False
        self._timer.stop()
        self.update()

    def _tick(self):
        self._phase = (self._phase + 0.012) % 1.0
        self.update()

    def paintEvent(self, e):
        if not self._active:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, w, h), 3, 3)
        p.setClipPath(path)
        p.fillPath(path, qc('accent', 40))
        seg = max(90, int(w * 0.25))
        x = -seg + (w + seg) * self._phase
        grad = QLinearGradient(x, 0, x + seg, 0)
        grad.setColorAt(0.0, QColor('#6366f1'))
        grad.setColorAt(1.0, QColor('#ec4899'))
        p.setPen(Qt.NoPen)
        p.setBrush(grad)
        p.drawRoundedRect(QRectF(x, 0, seg, h), 3, 3)


class DisplayPanel(QWidget):
    """Image / placeholder area (replaces the canvas). Aspect-fit, centred."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._rgb = None      # source frame (HxWx3 uint8, RGB)
        self._fill = False
        self._cache = None    # frame pre-scaled to the exact physical pixel size
        self._fps = None
        self.placeholder_visible = True

    def show_placeholder(self):
        self.placeholder_visible = True
        self._rgb = None
        self._cache = None
        self._fps = None
        self.update()

    def set_image(self, rgb_array, fill=False):
        """rgb_array: HxWx3 uint8 numpy array."""
        self.placeholder_visible = False
        self._rgb = np.ascontiguousarray(rgb_array)
        self._fill = fill
        self._cache = None
        self.update()

    def set_fps(self, fps):
        """Show a live FPS badge at the top-left (None hides it)."""
        self._fps = fps
        self.update()

    def resizeEvent(self, e):
        self._cache = None
        super().resizeEvent(e)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        cw, ch = self.width(), self.height()
        if cw <= 1 or ch <= 1:
            return

        frame = QPainterPath()
        frame.addRoundedRect(QRectF(0.5, 0.5, cw - 1, ch - 1), 12, 12)
        p.fillPath(frame, qc('canvas'))
        p.setClipPath(frame)

        if self.placeholder_visible or self._rgb is None:
            lines = PLACEHOLDER.split('\n')
            fonts = [make_font(22, True)] + [make_font(15)] * (len(lines) - 1)
            heights = [QFontMetrics(f).height() for f in fonts]
            total_h = sum(heights) + 4 * (len(lines) - 1)
            y = (ch - total_h) // 2
            for i, (line, f, lh) in enumerate(zip(lines, fonts, heights)):
                p.setFont(f)
                p.setPen(qc('accent' if i == 0 else 'fg_dim'))
                p.drawText(QRectF(0, y, cw, lh), Qt.AlignCenter, line)
                y += lh + 4
        else:
            ih, iw = self._rgb.shape[:2]
            if iw > 0 and ih > 0:
                if self._fill:
                    r = min((cw - 20) / iw, (ch - 20) / ih)
                else:
                    r = min((cw - 20) / iw, (ch - 20) / ih, 1.0)
                nw, nh = max(1, int(iw * r)), max(1, int(ih * r))

                # Scale to the *physical* pixel size (HiDPI aware) with OpenCV:
                # INTER_AREA when shrinking keeps detail crisp, cubic when enlarging.
                dpr = self.devicePixelRatioF()
                pw, ph = max(1, round(nw * dpr)), max(1, round(nh * dpr))
                if (self._cache is None or self._cache.width() != pw
                        or self._cache.height() != ph):
                    interp = cv2.INTER_AREA if (pw < iw or ph < ih) else cv2.INTER_CUBIC
                    scaled = np.ascontiguousarray(
                        cv2.resize(self._rgb, (pw, ph), interpolation=interp))
                    qimg = QImage(scaled.data, pw, ph, 3 * pw, QImage.Format_RGB888).copy()
                    qimg.setDevicePixelRatio(dpr)
                    self._cache = qimg
                p.drawImage(QPointF((cw - nw) / 2, (ch - nh) / 2), self._cache)

        # FPS badge (video / webcam), top-left of the display
        if self._fps is not None:
            font = make_font(22, True)
            text = f"FPS: {self._fps:.1f}"
            fm = QFontMetrics(font)
            bw, bh = fm.horizontalAdvance(text) + 28, fm.height() + 12
            badge = QRectF(16, 16, bw, bh)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(15, 23, 42, 190))
            p.drawRoundedRect(badge, 12, 12)
            p.setFont(font)
            p.setPen(QColor('#4ade80'))
            p.drawText(badge, Qt.AlignCenter, text)

        p.setClipping(False)
        p.setPen(QPen(qc('border'), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(QRectF(0.5, 0.5, cw - 1, ch - 1), 12, 12)


class RatioSplitter(QSplitter):
    """Horizontal splitter whose left pane takes `left_ratio` of the width,
    both initially and while the window is resized. The sash stays draggable."""

    def __init__(self, left_ratio=0.67, min_pane=320):
        super().__init__(Qt.Horizontal)
        self._ratio = left_ratio
        self._min_pane = min_pane
        self._placed = False
        self.setChildrenCollapsible(False)
        self.setHandleWidth(14)

    def split(self, left, right):
        left.setMinimumWidth(self._min_pane)
        right.setMinimumWidth(self._min_pane)
        self.addWidget(left)
        self.addWidget(right)
        self.setStretchFactor(0, 2)
        self.setStretchFactor(1, 1)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if not self._placed and self.width() > 400:
            self._placed = True
            w = self.width() - self.handleWidth()
            left = int(w * self._ratio)
            self.setSizes([left, w - left])


class CollapsibleCard(QFrame):
    """Card with a clickable header that expands / collapses its body (animated)."""

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 6)
        lay.setSpacing(0)

        self.toggle = QToolButton(self)
        self.toggle.setObjectName("collapse")
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(True)
        self.toggle.setArrowType(Qt.DownArrow)
        self.toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.toggle.setCursor(Qt.PointingHandCursor)
        lay.addWidget(self.toggle)

        self.body = QWidget(self)
        self.body.setMinimumHeight(0)
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(10, 4, 10, 10)
        self.body_layout.setSpacing(14)
        lay.addWidget(self.body)

        self._anim = QPropertyAnimation(self.body, b"maximumHeight", self)
        self._anim.setDuration(180)
        self._anim.setEasingCurve(QEasingCurve.InOutCubic)
        self._anim.finished.connect(self._anim_done)
        self.toggle.toggled.connect(self._on_toggled)

    def add_widget(self, w):
        self.body_layout.addWidget(w)

    def _on_toggled(self, on):
        self.toggle.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
        self._anim.stop()
        self._anim.setStartValue(self.body.height())
        self._anim.setEndValue(self.body.sizeHint().height() if on else 0)
        self._anim.start()

    def _anim_done(self):
        if self.toggle.isChecked():
            self.body.setMaximumHeight(16777215)


# ============================================================
# Splash / loading screen
# ============================================================
class SplashScreen(QWidget):
    """Frameless animated loading screen: rounded gradient card with a soft
    shadow, the app title, mode-coloured pulse dots, a status line and an
    indeterminate progress bar. Fades out when closed."""

    W, H, M = 780, 420, 26
    TITLE = "📟 Makers - YOLO Object Detection"

    def __init__(self):
        super().__init__(None, Qt.SplashScreen | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(self.W + 2 * self.M, self.H + 2 * self.M)
        geo = QGuiApplication.primaryScreen().availableGeometry()
        self.move(geo.center() - self.rect().center())
        self._status = "Loading..."
        self._phase = 0.0
        self._closing = False
        self._anim = None
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def set_status(self, text):
        self._status = text
        self.update()

    def close_fade(self):
        if self._closing:
            return
        self._closing = True
        self._anim = QPropertyAnimation(self, b"windowOpacity", self)
        self._anim.setDuration(350)
        self._anim.setStartValue(1.0)
        self._anim.setEndValue(0.0)
        self._anim.finished.connect(self._finish)
        self._anim.start()

    def _finish(self):
        self._timer.stop()
        self.close()

    def _tick(self):
        self._phase = (self._phase + 0.012) % 1.0
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)
        M, W, H = self.M, self.W, self.H
        rect = QRectF(M, M, W, H)
        radius = 26

        # soft drop shadow
        p.setPen(Qt.NoPen)
        for i in range(M):
            a = int(10 * (1 - i / M) ** 2)
            p.setBrush(QColor(10, 12, 40, a))
            p.drawRoundedRect(rect.adjusted(-i, -i + 8, i, i + 8), radius + i, radius + i)

        card = QPainterPath()
        card.addRoundedRect(rect, radius, radius)
        p.setClipPath(card)

        grad = QLinearGradient(rect.topLeft(), rect.bottomRight())
        grad.setColorAt(0.0, QColor(C('splash_a')))
        grad.setColorAt(1.0, QColor(C('splash_b')))
        p.fillPath(card, grad)

        # translucent decor circles
        p.setPen(Qt.NoPen)
        for cx, cy, r, alpha, col in (
                (M + W * 0.10, M + H * 0.16, 130, 40, '#22d3ee'),
                (M + W * 0.94, M + H * 0.88, 180, 36, '#f472b6'),
                (M + W * 0.84, M + H * 0.08, 70, 46, '#facc15')):
            c = QColor(col)
            c.setAlpha(alpha)
            p.setBrush(c)
            p.drawEllipse(QPointF(cx, cy), r, r)

        # title (shrinks to fit)
        size = 36
        while True:
            font = make_font(size, True)
            tw = QFontMetrics(font).horizontalAdvance(self.TITLE)
            th = QFontMetrics(font).height()
            if tw <= W - 80 or size <= 18:
                break
            size -= 1
        p.setFont(font)
        p.setPen(QColor('#ffffff'))
        title_y = M + int(H * 0.28)
        p.drawText(QRectF(M, title_y, W, th), Qt.AlignHCenter | Qt.AlignVCenter, self.TITLE)

        # accent underline
        bar_w = 96
        p.setPen(Qt.NoPen)
        p.setBrush(QColor('#22d3ee'))
        p.drawRoundedRect(QRectF(M + (W - bar_w) / 2, title_y + th + 14, bar_w, 5), 2.5, 2.5)

        # mode-coloured pulse dots
        dots = ('mode_image', 'mode_video', 'mode_webcam', 'mode_folder')
        dy = M + H * 0.55
        for i, name in enumerate(dots):
            pulse = 0.5 + 0.5 * math.sin(2 * math.pi * (self._phase * 2 - i * 0.16))
            r = 5 + 3.5 * pulse
            col = QColor(C(name))
            col.setAlpha(int(140 + 115 * pulse))
            p.setBrush(col)
            p.drawEllipse(QPointF(M + W / 2 + (i - 1.5) * 30, dy), r, r)

        # indeterminate progress bar
        track_x, track_w, track_h = M + 130, W - 260, 8
        track_y = M + H - 110
        track = QPainterPath()
        track.addRoundedRect(QRectF(track_x, track_y, track_w, track_h), 4, 4)
        p.fillPath(track, QColor('#5468d4'))
        seg_w = 150
        seg_x = track_x - seg_w + (track_w + seg_w) * self._phase
        p.save()
        p.setClipPath(track)
        sg = QLinearGradient(seg_x, 0, seg_x + seg_w, 0)
        sg.setColorAt(0.0, QColor('#22d3ee'))
        sg.setColorAt(1.0, QColor('#f472b6'))
        p.setBrush(sg)
        p.drawRoundedRect(QRectF(seg_x, track_y, seg_w, track_h), 4, 4)
        p.restore()

        # status line
        p.setFont(make_font(16))
        p.setPen(QColor('#dbe4ff'))
        p.drawText(QRectF(M, track_y + 22, W, 28), Qt.AlignHCenter | Qt.AlignVCenter, self._status)

        # thin outline
        p.setClipping(False)
        p.setPen(QPen(QColor(255, 255, 255, 60), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), radius, radius)


# ============================================================
# Page - holds the widgets/state that belong to one mode
# ============================================================
class Page:
    def __init__(self, mode):
        self.mode = mode
        self.panel = None
        self.display = None
        self.file_label = None
        self.select_btn = None
        self.stop_btn = None
        self.clear_btn = None
        self.save_btn = None
        self.swap_btn = None
        self.shuffle_btn = None
        self.pause_btn = None
        self.progress = None
        self.results_scroll = None   # only Image / Folder pages have a results panel
        self.results_host = None
        self.results_layout = None
        self.last_result = None


MODES = [
    ('image', "🖼 Image"),
    ('video', "🎬 Video"),
    ('webcam', "📹 Webcam"),
    ('folder', "📂 Folder"),
]
MODE_COLOR = {
    'image': 'mode_image',
    'video': 'mode_video',
    'webcam': 'mode_webcam',
    'folder': 'mode_folder',
}


# ============================================================
# Main window
# ============================================================
class YOLODetectorFrame(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Object Detection (ONNX)")
        self.resize(1400, 850)
        self.setMinimumSize(1100, 700)

        # Variables
        self.model = None
        self.model_path = None
        self.current_file = None
        self.folder_path = None
        self.folder_images = []
        self.video_thread = None
        self.stop_video = False
        self.is_processing = False
        self.active_page = None
        self._ui_busy = False
        self.paused = False
        self.conf_value = 0.30
        self.iou_value = 0.30

        # Camera state
        self.available_cameras = [0]
        self.current_cam_index = 0
        self._swapping = False

        self.pages = {}
        self.nav_buttons = {}
        self._conf_ctrls = []   # every copy of the confidence slider (+ value label)
        self._iou_ctrls = []    # every copy of the IoU slider (+ value label)
        self._splash = None
        self._startup_t0 = time.time()
        self._closing = False

        self._bridge = UiBridge()

        self._build_status_bar()
        self._build_ui()

        # Detect cameras in the background so we don't block startup
        threading.Thread(target=self._detect_cameras_bg, daemon=True).start()

    # ------------------------------------------------------------------
    # Startup (splash screen + default model loaded in a worker thread)
    # ------------------------------------------------------------------
    def begin_startup(self, splash):
        self._splash = splash
        self._startup_t0 = time.time()
        splash.set_status("Loading model...")
        self.update_status("Loading default model from models/yolo11n.onnx...", COLORS['warning'])
        threading.Thread(target=self._startup_worker, daemon=True).start()

    def _startup_worker(self):
        model_path = MODELS_DIR / 'yolo11n.onnx'
        model, err = None, None
        if model_path.exists():
            try:
                model = ONNXYOLO(str(model_path))
            except Exception as e:
                err = str(e)
        self.ui(lambda: self._startup_loaded(model_path, model, err))

    def _startup_loaded(self, model_path, model, err):
        # Keep the splash up long enough to be seen, even if the model loads fast
        remaining_ms = max(0, int((1.8 - (time.time() - self._startup_t0)) * 1000))
        QTimer.singleShot(remaining_ms, lambda: self._finish_startup(model_path, model, err))

    def _finish_startup(self, model_path, model, err):
        if model is not None:
            self._apply_model(model, str(model_path))
        elif err is not None:
            self.model_label.setText("❌ Failed to load")
            recolor(self.model_label, 'h_err')
            self.update_status("Error loading model", COLORS['error'])
        else:
            self.update_status("No model found in 'models' folder. Please select a model.", COLORS['error'])
            self.model_label.setText("No model loaded")
            recolor(self.model_label, 'h_err')

        self.showMaximized()
        self.raise_()
        self.activateWindow()
        if self._splash is not None:
            self._splash.close_fade()
            self._splash = None

        # Small delay so the dialog isn't hidden behind the fading splash
        if err is not None:
            QTimer.singleShot(400, lambda: QMessageBox.critical(
                self, "Error", f"Failed to load model:\n{err}"))
        elif model is None:
            QTimer.singleShot(400, self._ask_for_model)

    def _ask_for_model(self):
        answer = QMessageBox.question(
            self, "Model Not Found",
            "Model file 'models/yolo11n.onnx' not found.\n\n"
            "Would you like to select an ONNX model file now?",
            QMessageBox.Yes | QMessageBox.No)
        if answer == QMessageBox.Yes:
            self.on_load_model(None)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_status_bar(self):
        sb = QStatusBar(self)
        sb.setSizeGripEnabled(False)
        self.setStatusBar(sb)
        holder = QWidget()
        hl = QHBoxLayout(holder)
        hl.setContentsMargins(14, 4, 14, 4)
        hl.setSpacing(10)
        self.status_dot = QLabel()
        self.status_dot.setFixedSize(10, 10)
        self.status_text = static_text(None, "Ready", 13, False, 'fg')
        hl.addWidget(self.status_dot)
        hl.addWidget(self.status_text, 1)
        sb.addWidget(holder, 1)
        self.update_status("Ready", COLORS['fg_dim'])

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        row = QHBoxLayout(root)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        row.addWidget(self._build_sidebar())

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(0)

        # Thin multi-colour strip: one colour per mode
        strip = QFrame()
        strip.setFixedHeight(4)
        strip.setStyleSheet(
            "background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 {C('mode_image')}, stop:0.35 {C('mode_video')},"
            f"stop:0.68 {C('mode_webcam')}, stop:1 {C('mode_folder')});")
        rv.addWidget(strip)

        self.workspace_book = QStackedWidget()
        for mode, title in MODES:
            page = Page(mode)
            self.pages[mode] = page
            self.workspace_book.addWidget(self._build_page(page, title))
        rv.addWidget(self.workspace_book, 1)
        row.addWidget(right, 1)

        self.active_page = self.pages['image']
        self._switch_to_mode('image')

    def _build_sidebar(self):
        side = QFrame()
        side.setObjectName("sidebar")
        side.setFixedWidth(272)
        side.setStyleSheet(
            "QFrame#sidebar { background:qlineargradient(x1:0,y1:0,x2:0.6,y2:1,"
            f"stop:0 {C('header')}, stop:1 {C('header_b')}); border:none; }}")
        v = QVBoxLayout(side)
        v.setContentsMargins(20, 26, 20, 20)
        v.setSpacing(10)

        # Brand
        v.addWidget(static_text(side, "📟 Makers", 30, True, 'h_fg'))
        v.addWidget(static_text(side, "YOLO Object Detection", 14, False, 'h_dim'))
        v.addSpacing(14)
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background:rgba(255,255,255,16%); border:none;")
        v.addWidget(sep)
        v.addSpacing(10)

        # Mode navigation (radio group)
        v.addWidget(static_text(side, "📁 Select Option", 15, True, 'h_accent'))
        v.addSpacing(2)
        group = QButtonGroup(self)
        group.setExclusive(True)
        for mode, title in MODES:
            col = C(MODE_COLOR[mode])
            btn = QPushButton(title, side)
            btn.setCheckable(True)
            btn.setFixedHeight(54)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(f"""
                QPushButton {{ text-align:left; padding:0 20px; border:none; border-radius:14px;
                               color:#dbe4ff; background:rgba(255,255,255,8%);
                               font-size:16px; font-weight:600; }}
                QPushButton:hover {{ background:rgba(255,255,255,16%); color:#ffffff; }}
                QPushButton:checked {{ color:#ffffff; background:qlineargradient(
                    x1:0,y1:0,x2:1,y2:0, stop:0 {col}, stop:1 {QColor(col).lighter(125).name()}); }}
            """)
            btn.clicked.connect(lambda _=False, m=mode: self._switch_to_mode(m))
            group.addButton(btn)
            self.nav_buttons[mode] = btn
            v.addWidget(btn)

        v.addStretch(1)

        # Model card
        card = QFrame(side)
        card.setObjectName("modelcard")
        card.setStyleSheet(
            "QFrame#modelcard { background:rgba(255,255,255,9%);"
            " border:1px solid rgba(255,255,255,18%); border-radius:16px; }")
        cv = QVBoxLayout(card)
        cv.setContentsMargins(16, 14, 16, 16)
        cv.setSpacing(6)
        cv.addWidget(static_text(card, "🤖 ONNX Model", 15, True, 'h_accent'))
        cv.addWidget(static_text(card, "Current Model:", 13, False, 'h_dim'))
        self.model_label = static_text(card, "Loading...", 16, True, 'h_warn', cls=ElidedLabel)
        cv.addWidget(self.model_label)
        cv.addSpacing(6)
        cv.addWidget(make_button(card, "📂 Change Model", self.on_load_model, bg='h_btn',
                                 size=14, height=42, pad_x=18))
        cv.addWidget(make_button(card, "🔄 Reload", self.on_reload_model, bg='h_btn',
                                 size=14, height=42, pad_x=18))
        v.addWidget(card)
        return side

    def _switch_to_mode(self, mode):
        idx = [m for m, _ in MODES].index(mode)
        # Switching pages while a stream is running: stop it first
        if self.is_processing and self.active_page and self.active_page.mode != mode:
            self.stop_video = True
        self.workspace_book.setCurrentIndex(idx)
        self.active_page = self.pages[mode]
        for m, btn in self.nav_buttons.items():
            btn.setChecked(m == mode)

    # -- pages ---------------------------------------------------------
    def _build_page(self, page, title):
        panel = QWidget()
        page.panel = panel
        v = QVBoxLayout(panel)
        v.setContentsMargins(24, 20, 24, 16)
        v.setSpacing(16)

        v.addLayout(self._build_actions(panel, page, title))

        if page.mode in ('image', 'folder'):
            # Display on the left, settings + analysis on the right (2 : 1)
            splitter = RatioSplitter(left_ratio=0.67, min_pane=320)
            splitter.split(self._build_display_card(page), self._build_side_panel(page))
            v.addWidget(splitter, 1)
        else:
            # Video / Webcam: view with the two sliders side by side underneath
            v.addWidget(self._build_display_card(page), 1)
            v.addWidget(self._build_settings_strip(panel), 0, Qt.AlignHCenter)
        return panel

    def _make_threshold(self, parent, caption, kind, width=None):
        """One threshold control (caption + value + slider). Every copy is
        registered so all sliders in the app stay in sync."""
        box = QWidget(parent)
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(10)
        row = QHBoxLayout()
        row.addWidget(static_text(box, caption, 14))
        row.addStretch(1)
        value = static_text(box, "0.30", 15, True, 'accent')
        value.setStyleSheet(f"color:{C('accent')}; background:{C('accent_soft')};"
                            " border-radius:9px; padding:2px 12px;")
        row.addWidget(value)
        v.addLayout(row)

        slider = QSlider(Qt.Horizontal, box)
        slider.setRange(10, 90)
        slider.setValue(30)
        if width:
            slider.setMinimumWidth(width)
        if kind == 'conf':
            slider.valueChanged.connect(self.on_conf_slider)
            self._conf_ctrls.append((slider, value))
        else:
            slider.valueChanged.connect(self.on_iou_slider)
            self._iou_ctrls.append((slider, value))
        v.addWidget(slider)
        return box

    def _build_actions(self, panel, page, title):
        """Action toolbar shown above every page."""
        mode = page.mode
        actions = QHBoxLayout()
        actions.setSpacing(10)

        select_cmd = {
            'image': self.upload_image,
            'video': self.upload_video,
            'webcam': self.use_webcam,
            'folder': self.upload_folder,
        }[mode]
        page.select_btn = make_button(panel, title, select_cmd, bg=MODE_COLOR[mode])
        actions.addWidget(page.select_btn)

        if mode == 'folder':
            page.shuffle_btn = make_button(panel, "🎲 Shuffle 4 Images", self.on_shuffle_folder,
                                           enabled=False)
            actions.addWidget(page.shuffle_btn)

        if mode == 'webcam':
            page.swap_btn = make_button(panel, "🔀 Swap Cam", self.on_swap_camera, enabled=False)
            actions.addWidget(page.swap_btn)

        if mode == 'video':
            page.pause_btn = make_button(panel, "⏸ Pause", self.on_toggle_pause, enabled=False)
            actions.addWidget(page.pause_btn)

        if mode in ('video', 'webcam'):
            page.stop_btn = make_button(panel, "⏹ Stop", self.on_stop_detection,
                                        bg='danger', enabled=False)
            actions.addWidget(page.stop_btn)

        page.clear_btn = make_button(panel, "🗑 Clear", self.on_clear_display, bg='danger')
        actions.addWidget(page.clear_btn)

        if mode == 'webcam':
            page.save_btn = make_button(panel, "📸 Capture", self.on_capture_frame,
                                        bg='green', enabled=False)
            actions.addWidget(page.save_btn)
        elif mode != 'video':
            page.save_btn = make_button(panel, "💾 Save Result", self.on_save_result,
                                        bg='green', enabled=False)
            actions.addWidget(page.save_btn)

        actions.addStretch(1)
        return actions

    def _build_display_card(self, page):
        """Card with a header (Display + selected file), the view and a busy bar."""
        card = QFrame()
        card.setObjectName("card")
        v = QVBoxLayout(card)
        v.setContentsMargins(16, 12, 16, 12)
        v.setSpacing(10)

        head = QHBoxLayout()
        head.setSpacing(8)
        head.addWidget(static_text(card, "📺 Display", 16, True, 'accent'))
        head.addStretch(1)
        head.addWidget(static_text(card, "📄 Selected File:", 14, False, 'fg_dim'))
        page.file_label = static_text(card, "No file selected", 15, True, 'error')
        head.addWidget(page.file_label)
        v.addLayout(head)

        page.display = DisplayPanel(card)
        v.addWidget(page.display, 1)

        page.progress = BusyBar(card)
        v.addWidget(page.progress)
        return card

    def _build_settings_strip(self, parent):
        """Video / Webcam: detection settings under the view, sliders side by side."""
        card = QFrame(parent)
        card.setObjectName("card")
        v = QVBoxLayout(card)
        v.setContentsMargins(20, 14, 20, 16)
        v.setSpacing(6)
        v.addWidget(static_text(card, "⚙️ Detection Settings", 16, True, 'accent'), 0,
                    Qt.AlignHCenter)
        row = QHBoxLayout()
        row.setSpacing(36)
        row.addWidget(self._make_threshold(card, "Confidence Threshold:", 'conf', width=380))
        row.addWidget(self._make_threshold(card, "IoU Threshold:", 'iou', width=380))
        v.addLayout(row)
        return card

    def _build_side_panel(self, page):
        """Right-hand pane of the Image / Folder pages: collapsible detection
        settings on top, detection analysis underneath."""
        panel = QWidget()
        v = QVBoxLayout(panel)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(14)

        coll = CollapsibleCard("⚙️ Detection Settings", panel)
        coll.add_widget(self._make_threshold(coll, "Confidence Threshold:", 'conf'))
        coll.add_widget(self._make_threshold(coll, "IoU Threshold:", 'iou'))
        v.addWidget(coll)

        card = QFrame(panel)
        card.setObjectName("card")
        cv = QVBoxLayout(card)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(0)

        banner = static_text(card, "📊 Detection Results", 16, True, '#ffffff')
        banner.setStyleSheet(
            "color:#ffffff; padding:13px 18px; border-top-left-radius:15px;"
            " border-top-right-radius:15px; background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 {C('accent')}, stop:1 #a855f7);")
        cv.addWidget(banner)

        page.results_scroll = QScrollArea(card)
        page.results_scroll.setWidgetResizable(True)
        page.results_scroll.setFrameShape(QFrame.NoFrame)
        page.results_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        page.results_scroll.viewport().setAutoFillBackground(False)
        page.results_host = QWidget()
        page.results_host.setAutoFillBackground(False)
        page.results_layout = QVBoxLayout(page.results_host)
        page.results_layout.setContentsMargins(10, 10, 10, 10)
        page.results_layout.setSpacing(8)
        page.results_scroll.setWidget(page.results_host)
        cv.addWidget(page.results_scroll, 1)

        v.addWidget(card, 1)
        self.show_no_results(page)
        return panel

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def ui(self, fn):
        """Schedule fn on the UI thread (safe to call from worker threads)."""
        try:
            self._bridge.call.emit(fn)
        except Exception:
            pass

    def on_conf_slider(self, v):
        self.conf_value = v / 100.0
        for slider, label in self._conf_ctrls:
            if slider.value() != v:
                slider.blockSignals(True)
                slider.setValue(v)
                slider.blockSignals(False)
            label.setText(f"{self.conf_value:.2f}")

    def on_iou_slider(self, v):
        self.iou_value = v / 100.0
        for slider, label in self._iou_ctrls:
            if slider.value() != v:
                slider.blockSignals(True)
                slider.setValue(v)
                slider.blockSignals(False)
            label.setText(f"{self.iou_value:.2f}")

    def update_status(self, message, color=None):
        self.status_text.setText(message)
        self.status_dot.setStyleSheet(
            f"background:{color or COLORS['fg_dim']}; border-radius:5px;")

    def _set_file_label(self, page, text, color):
        page.file_label.setText(text)
        recolor(page.file_label, color)

    def _require_model(self):
        if not self.model:
            QMessageBox.warning(self, "Warning", "Please load a model first!")
            return False
        return True

    def _begin_job(self, page, show_progress):
        self.is_processing = True
        self.stop_video = False
        page.select_btn.setEnabled(False)
        if page.shuffle_btn:
            page.shuffle_btn.setEnabled(False)
        if page.stop_btn:
            page.stop_btn.setEnabled(True)
        if page.pause_btn:
            self.paused = False
            page.pause_btn.setEnabled(True)
            page.pause_btn.setText("⏸ Pause")
        if page.mode == 'webcam':
            page.save_btn.setEnabled(True)
        if show_progress:
            page.progress.start()

    def _end_job(self, page):
        page.progress.stop()
        page.select_btn.setEnabled(True)
        if page.shuffle_btn and self.folder_path:
            page.shuffle_btn.setEnabled(True)
        if page.stop_btn:
            page.stop_btn.setEnabled(False)
        if page.pause_btn:
            self.paused = False
            page.pause_btn.setEnabled(False)
            page.pause_btn.setText("⏸ Pause")
        self.is_processing = False

    # ------------------------------------------------------------------
    # Camera detection / swapping
    # ------------------------------------------------------------------
    def _detect_cameras_bg(self):
        """Probe a few camera indices in a background thread."""
        found = []
        for i in range(3):
            try:
                cap = cv2.VideoCapture(i)
                if cap is not None and cap.isOpened():
                    found.append(i)
                cap.release()
            except Exception:
                pass
        if not found:
            found = [0]
        self.ui(lambda: self._on_cameras_detected(found))

    def _on_cameras_detected(self, cams):
        self.available_cameras = cams
        self.current_cam_index = cams[0]
        self.pages['webcam'].swap_btn.setEnabled(len(cams) > 1)

    def on_swap_camera(self, evt):
        self.swap_camera()

    def swap_camera(self):
        """Cycle to the next detected camera index.

        Clean stop -> wait for the capture to fully release -> restart, so the
        old and new camera handles never overlap."""
        if len(self.available_cameras) < 2:
            return
        if self._swapping:
            return

        page = self.pages['webcam']
        try:
            idx = self.available_cameras.index(self.current_cam_index)
        except ValueError:
            idx = -1
        idx = (idx + 1) % len(self.available_cameras)
        new_index = self.available_cameras[idx]
        self.current_cam_index = new_index

        if not isinstance(self.current_file, int):
            self.update_status(
                f"Camera set to index {new_index} (used next time you select Webcam)",
                COLORS['accent'])
            return

        self._swapping = True
        page.swap_btn.setEnabled(False)

        if self.is_processing:
            self.update_status(f"Switching to camera {new_index}...", COLORS['warning'])
            self.stop_video = True
            old_thread = self.video_thread
            threading.Thread(target=self._wait_and_restart_camera,
                             args=(old_thread, new_index), daemon=True).start()
        else:
            self.current_file = new_index
            self._set_file_label(page, f"📹 Webcam {new_index}", 'success')
            self.update_status(f"Switched to camera index {new_index}", COLORS['accent'])
            self._swapping = False
            page.swap_btn.setEnabled(True)

    def _wait_and_restart_camera(self, old_thread, new_index):
        if old_thread is not None:
            old_thread.join(timeout=5)
        self.ui(lambda: self._finish_camera_swap(new_index))

    def _finish_camera_swap(self, new_index):
        page = self.pages['webcam']
        self.current_file = new_index
        self._set_file_label(page, f"📹 Webcam {new_index}", 'success')
        self._swapping = False
        page.swap_btn.setEnabled(len(self.available_cameras) > 1)
        self.update_status(f"Switched to camera index {new_index}", COLORS['accent'])
        # Seamlessly resume detection on the new camera
        self.run_video(page, self.current_file)

    # ------------------------------------------------------------------
    # Model loading (ONNX)
    # ------------------------------------------------------------------
    def _apply_model(self, model, file_path):
        self.model = model
        self.model_path = file_path
        model_name = Path(file_path).name
        self.model_label.setText(f"✓ {model_name}")
        recolor(self.model_label, 'h_ok')
        self.model_label.setToolTip(str(file_path))
        self.update_status(f"Model loaded successfully: {model_name}", COLORS['success'])

    def on_load_model(self, evt):
        """Load YOLO ONNX model from file dialog"""
        initial_dir = str(MODELS_DIR) if MODELS_DIR.exists() else "."
        path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO ONNX Model File", initial_dir,
            "ONNX Models (*.onnx);;All files (*.*)")
        if path:
            self.load_model_file(path)

    def load_model_file(self, file_path):
        """Load ONNX model from given path"""
        try:
            self.update_status("Loading model...", COLORS['warning'])
            QApplication.processEvents()
            model = ONNXYOLO(file_path)
            self._apply_model(model, file_path)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model:\n{str(e)}")
            self.model_label.setText("❌ Failed to load")
            recolor(self.model_label, 'h_err')
            self.update_status("Error loading model", COLORS['error'])

    def on_reload_model(self, evt):
        """Reload the current model"""
        if self.model_path:
            self.load_model_file(self.model_path)
        else:
            QMessageBox.information(self, "Info", "No model to reload. Please load a model first.")

    # ------------------------------------------------------------------
    # Selection actions - each one starts detection immediately
    # ------------------------------------------------------------------
    def upload_image(self, evt):
        """Select an image and run detection immediately"""
        if not self._require_model():
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Image files (*.jpg *.jpeg *.png *.bmp);;All files (*.*)")
        if not file_path:
            return

        page = self.pages['image']
        self.current_file = file_path
        self._set_file_label(page, Path(file_path).name, 'success')
        self.update_status(f"Image loaded: {Path(file_path).name}", COLORS['success'])
        self.run_image(page, file_path)

    def upload_video(self, evt):
        """Select a video and run detection immediately"""
        if not self._require_model():
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "",
            "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*.*)")
        if not file_path:
            return

        page = self.pages['video']
        self.current_file = file_path
        self._set_file_label(page, Path(file_path).name, 'success')
        self.update_status(f"Video loaded: {Path(file_path).name}", COLORS['success'])
        self.run_video(page, file_path)

    def use_webcam(self, evt):
        """Use webcam for detection - starts immediately"""
        if not self._require_model():
            return
        page = self.pages['webcam']
        self.current_file = self.current_cam_index
        self._set_file_label(page, f"📹 Webcam {self.current_cam_index}", 'success')
        self.update_status(f"Webcam {self.current_cam_index} selected", COLORS['success'])
        self.run_video(page, self.current_file)

    def upload_folder(self, evt):
        """Select a folder, randomly pick 4 images and run batch detection"""
        if not self._require_model():
            return
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder:
            return

        images = [p for p in Path(folder).iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if not images:
            QMessageBox.warning(self, "Warning", "No images found in the selected folder!")
            return

        page = self.pages['folder']
        self.folder_path = folder
        self.folder_images = images
        self._set_file_label(page, f"{Path(folder).name}  ({len(images)} images)", 'success')
        self.update_status(f"Folder loaded: {Path(folder).name} ({len(images)} images)",
                           COLORS['success'])
        self.shuffle_folder()

    def on_shuffle_folder(self, evt):
        self.shuffle_folder()

    def shuffle_folder(self):
        """Pick a new random set of up to 4 images and run detection"""
        if not self._require_model() or not self.folder_images:
            return
        page = self.pages['folder']
        picks = random.sample(self.folder_images, min(GRID_COUNT, len(self.folder_images)))
        self.run_folder(page, picks)

    # ------------------------------------------------------------------
    # Image inference
    # ------------------------------------------------------------------
    def run_image(self, page, path):
        if self.is_processing:
            return
        self._begin_job(page, show_progress=True)
        self.update_status("Processing image...", COLORS['warning'])
        conf, iou = self.conf_value, self.iou_value
        threading.Thread(target=self._image_worker, args=(page, path, conf, iou),
                         daemon=True).start()

    def _image_worker(self, page, path, conf, iou):
        try:
            img_bgr = cv2.imread(str(path))
            if img_bgr is None:
                raise ValueError("Could not read the selected image file")
            result, annotated_bgr = self.model.infer(img_bgr, conf=conf, iou=iou)
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            self.ui(lambda: self._image_done(page, result, annotated_rgb))
        except Exception as e:
            import traceback
            traceback.print_exc()
            msg = str(e)
            self.ui(lambda: self._job_failed(page, "Detection failed", msg))

    def _image_done(self, page, result, annotated_rgb):
        page.display.set_image(annotated_rgb, fill=False)
        self.display_results(page, result)
        page.last_result = annotated_rgb
        page.save_btn.setEnabled(True)
        self.update_status(f"✓ Detection complete: {len(result.boxes)} objects found",
                           COLORS['success'])
        self._end_job(page)

    def _job_failed(self, page, status, detail):
        QMessageBox.critical(self, "Error", f"{status}:\n{detail}")
        self.update_status(status, COLORS['error'])
        self._end_job(page)

    # ------------------------------------------------------------------
    # Folder (batch) inference -> 2x2 grid
    # ------------------------------------------------------------------
    def run_folder(self, page, paths):
        if self.is_processing:
            return
        self._begin_job(page, show_progress=True)
        self.update_status(f"Processing {len(paths)} images...", COLORS['warning'])
        conf, iou = self.conf_value, self.iou_value
        threading.Thread(target=self._folder_worker, args=(page, paths, conf, iou),
                         daemon=True).start()

    def _folder_worker(self, page, paths, conf, iou):
        try:
            entries, annotated_list, names_list = [], [], []
            for p in paths:
                img_bgr = cv2.imread(str(p))
                if img_bgr is None:
                    continue
                result, annotated_bgr = self.model.infer(img_bgr, conf=conf, iou=iou)
                entries.append((p.name, result))
                annotated_list.append(annotated_bgr)
                names_list.append(p.name)
            if not entries:
                raise ValueError("None of the selected images could be read")

            grid_bgr = build_grid(annotated_list, names_list)
            grid_rgb = cv2.cvtColor(grid_bgr, cv2.COLOR_BGR2RGB)
            self.ui(lambda: self._folder_done(page, entries, grid_rgb))
        except Exception as e:
            import traceback
            traceback.print_exc()
            msg = str(e)
            self.ui(lambda: self._job_failed(page, "Batch detection failed", msg))

    def _folder_done(self, page, entries, grid_rgb):
        page.display.set_image(grid_rgb, fill=False)
        self.display_folder_results(page, entries)
        page.last_result = grid_rgb
        page.save_btn.setEnabled(True)
        total = sum(len(r.boxes) for _, r in entries)
        self.update_status(
            f"✓ Detection complete: {total} objects found in {len(entries)} images",
            COLORS['success'])
        self._end_job(page)

    # ------------------------------------------------------------------
    # Video / webcam inference
    # ------------------------------------------------------------------
    def run_video(self, page, source):
        if self.is_processing:
            return
        self._begin_job(page, show_progress=False)
        self.video_thread = threading.Thread(target=self._video_worker,
                                             args=(page, source), daemon=True)
        self.video_thread.start()

    def _push_frame(self, page, rgb, text, fps=None):
        """Drop frames if the UI hasn't finished drawing the previous one."""
        if self._ui_busy:
            return
        self._ui_busy = True
        self.ui(lambda: self._show_frame(page, rgb, text, fps))

    def _show_frame(self, page, rgb, text, fps=None):
        try:
            page.display.set_image(rgb, fill=True)
            page.display.set_fps(fps)
            page.last_result = rgb
            self.update_status(text, COLORS['warning'])
        finally:
            self._ui_busy = False

    def _video_worker(self, page, source):
        frame_count = 0
        total_detections = 0
        error = None
        try:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                self.ui(lambda: QMessageBox.critical(self, "Error", "Failed to open video source!"))
                cap.release()
                self.ui(lambda: self._video_done(page, 0, 0, True, None))
                return

            fps = None                      # smoothed frames per second
            t_prev = time.perf_counter()
            while cap.isOpened() and not self.stop_video:
                if self.paused:
                    time.sleep(0.05)
                    t_prev = time.perf_counter()   # don't count paused time
                    continue
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                result, annotated_bgr = self.model.infer(
                    frame, conf=self.conf_value, iou=self.iou_value)
                total_detections += len(result.boxes)

                now = time.perf_counter()
                inst = 1.0 / max(now - t_prev, 1e-6)
                t_prev = now
                fps = inst if fps is None else 0.9 * fps + 0.1 * inst

                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                self._push_frame(
                    page, annotated_rgb,
                    f"Frame {frame_count} - {len(result.boxes)} objects detected", fps)

                time.sleep(0.01)

            cap.release()
        except Exception as e:
            error = str(e)

        stopped = self.stop_video
        self.ui(lambda: self._video_done(page, frame_count, total_detections, stopped, error))

    def _video_done(self, page, frame_count, total_detections, stopped, error):
        self._ui_busy = False
        page.display.set_fps(None)
        self._end_job(page)
        if page.mode == 'webcam' and page.save_btn:
            page.save_btn.setEnabled(False)

        if error:
            QMessageBox.critical(self, "Error", f"Video processing failed:\n{error}")
            self.update_status("Video processing failed", COLORS['error'])
        elif self._swapping:
            pass  # camera swap in progress; status handled there
        elif not stopped:
            self.update_status(
                f"✓ Video complete: {frame_count} frames, {total_detections} total detections",
                COLORS['success'])
            QMessageBox.information(self, "Complete",
                                    f"Video processing complete!\n"
                                    f"Frames: {frame_count}\n"
                                    f"Total detections: {total_detections}")
        else:
            self.update_status("Video processing stopped", COLORS['warning'])

    def on_toggle_pause(self, evt):
        """Play/pause toggle for video playback"""
        page = self.pages['video']
        if not self.is_processing:
            return
        self.paused = not self.paused
        if self.paused:
            page.pause_btn.setText("▶ Play")
            self.update_status("Video paused", COLORS['warning'])
        else:
            page.pause_btn.setText("⏸ Pause")
            self.update_status("Video playing", COLORS['accent'])

    def on_capture_frame(self, evt):
        """Save the current annotated webcam frame instantly (no dialog)"""
        page = self.pages['webcam']
        if page.last_result is None:
            QMessageBox.warning(self, "Warning", "No frame to capture yet!")
            return
        try:
            CAPTURES_DIR.mkdir(exist_ok=True)
            file_path = CAPTURES_DIR / f"capture_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}.jpg"
            frame = page.last_result.copy()
            cv2.imwrite(str(file_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            self.update_status(f"📸 Captured: {file_path}", COLORS['success'])
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to capture frame:\n{str(e)}")

    def on_stop_detection(self, evt):
        """Stop video processing"""
        self.paused = False
        self.stop_video = True
        self.update_status("Stopping...", COLORS['warning'])

    # ------------------------------------------------------------------
    # Results panel (Image / Folder pages only)
    # ------------------------------------------------------------------
    def _clear_results(self, page):
        lay = page.results_layout
        if lay is None:
            return
        while lay.count():
            item = lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    def _refresh_results(self, page):
        if page.results_layout is None:
            return
        page.results_layout.addStretch(1)
        page.results_scroll.verticalScrollBar().setValue(0)

    def show_no_results(self, page):
        if page.results_layout is None:
            return
        self._clear_results(page)
        hint = {
            'image': "No detections yet\n\nSelect an image\nto run detection",
            'video': "No detections yet\n\nSelect a video\nto run detection",
            'webcam': "No detections yet\n\nStart the webcam\nto run detection",
            'folder': "No detections yet\n\nSelect a folder\nto run batch detection",
        }[page.mode]
        st = static_text(page.results_host, hint, 15, False, 'fg_dim')
        st.setAlignment(Qt.AlignCenter)
        st.setContentsMargins(0, 50, 0, 50)
        page.results_layout.addWidget(st)
        self._refresh_results(page)

    def _make_card(self, page, accent='accent', horizontal=False):
        """White card with a coloured stripe on its left edge. Returns
        (card, inner_layout); add content to inner_layout, then add the card
        to page.results_layout."""
        card = QFrame(page.results_host)
        card.setObjectName("rcard")
        outer = QHBoxLayout(card)
        outer.setContentsMargins(1, 1, 1, 1)
        outer.setSpacing(0)
        strip = QFrame(card)
        strip.setFixedWidth(6)
        strip.setStyleSheet(f"background:{C(accent)}; border:none;"
                            " border-top-left-radius:11px; border-bottom-left-radius:11px;")
        outer.addWidget(strip)
        inner = QHBoxLayout() if horizontal else QVBoxLayout()
        inner.setContentsMargins(14, 10, 14, 10)
        inner.setSpacing(4)
        outer.addLayout(inner, 1)
        return card, inner

    def _badge(self, parent, count, color):
        badge = static_text(parent, f" {count} ", 14, True, '#ffffff')
        badge.setStyleSheet(f"color:#ffffff; background:{C(color)};"
                            " border-radius:11px; padding:2px 8px;")
        return badge

    def _total_card(self, page, count, subtitle="Objects Detected"):
        card, lay = self._make_card(page, 'accent')
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(0)
        lay.addWidget(static_text(card, f"{count}", 46, True, 'accent'), 0, Qt.AlignHCenter)
        lay.addWidget(static_text(card, subtitle, 13, False, 'fg_dim'), 0, Qt.AlignHCenter)
        page.results_layout.addWidget(card)

    def _section_header(self, page, text):
        st = static_text(page.results_host, text, 16, True, 'accent')
        st.setContentsMargins(4, 8, 0, 0)
        page.results_layout.addWidget(st)

    def _divider(self, page):
        line = QFrame(page.results_host)
        line.setFixedHeight(1)
        line.setStyleSheet(f"background:{C('border')}; border:none;")
        page.results_layout.addWidget(line)

    def _no_objects_label(self, page):
        lbl = static_text(page.results_host, "No objects detected", 13, False, 'fg_dim')
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setContentsMargins(0, 20, 0, 20)
        page.results_layout.addWidget(lbl)

    def display_results(self, page, result, frame_num=None):
        """Display detection results in structured format"""
        self._clear_results(page)
        boxes = result.boxes

        if frame_num:
            card, lay = self._make_card(page, 'mode_video')
            lay.addWidget(static_text(card, f"FRAME {frame_num}", 14, True, 'accent'), 0,
                          Qt.AlignHCenter)
            page.results_layout.addWidget(card)

        self._total_card(page, len(boxes))

        if len(boxes) == 0:
            self._no_objects_label(page)
            self._refresh_results(page)
            return

        self._section_header(page, "📈 Summary by Class")
        class_counts = {}
        for box in boxes:
            class_name = result.names[int(box.cls[0])]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            self.create_summary_card(page, class_name, count)

        self._divider(page)

        self._section_header(page, "🔍 Detailed Detections")
        for i, box in enumerate(boxes, 1):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = result.names[cls_id]
            x1, y1, x2, y2 = box.xyxy[0]
            self.create_detection_card(page, i, class_name, conf, x1, y1, x2, y2)

        self._refresh_results(page)

    def display_folder_results(self, page, entries):
        """Analysis for the 2x2 grid: overall totals, class summary across all
        images, then one card per image (count, avg/max confidence, classes)."""
        self._clear_results(page)

        total = sum(len(r.boxes) for _, r in entries)
        all_confs = [float(b.conf[0]) for _, r in entries for b in r.boxes]
        avg_conf = (sum(all_confs) / len(all_confs)) if all_confs else 0.0

        self._total_card(page, total, f"Objects Detected in {len(entries)} Images")

        if total == 0:
            self._no_objects_label(page)
        else:
            card, lay = self._make_card(page, 'success', horizontal=True)
            for label, value in (("Avg Confidence", f"{avg_conf * 100:.1f}%"),
                                 ("Max Confidence", f"{max(all_confs) * 100:.1f}%"),
                                 ("Avg / Image", f"{total / len(entries):.1f}")):
                col = QVBoxLayout()
                col.setSpacing(0)
                col.addWidget(static_text(card, value, 17, True, 'success'), 0, Qt.AlignHCenter)
                col.addWidget(static_text(card, label, 12, False, 'fg_dim'), 0, Qt.AlignHCenter)
                lay.addLayout(col, 1)
            page.results_layout.addWidget(card)

            self._section_header(page, "📈 Summary by Class")
            class_counts = {}
            for _, r in entries:
                for b in r.boxes:
                    n = r.names[int(b.cls[0])]
                    class_counts[n] = class_counts.get(n, 0) + 1
            for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                self.create_summary_card(page, class_name, count)

        self._divider(page)

        self._section_header(page, "🔍 Detailed Detections")
        for i, (name, r) in enumerate(entries, 1):
            self.create_image_card(page, i, name, r)

        self._refresh_results(page)

    def create_summary_card(self, page, class_name, count):
        """Create a summary card for each class"""
        card, lay = self._make_card(page, 'mode_video', horizontal=True)
        lay.addWidget(static_text(card, class_name.capitalize(), 15, True), 1)
        lay.addWidget(self._badge(card, count, 'mode_video'))
        page.results_layout.addWidget(card)

    def create_image_card(self, page, index, name, result):
        """Per-image analysis card used by folder mode"""
        boxes = result.boxes
        card, lay = self._make_card(page, 'mode_folder')

        head = QHBoxLayout()
        head.setSpacing(8)
        head.addWidget(static_text(card, f"#{index}", 14, True, 'mode_folder'))
        short = name if len(name) <= 26 else name[:23] + "..."
        head.addWidget(static_text(card, short, 15, True), 1)
        head.addWidget(self._badge(card, len(boxes), 'mode_folder'))
        lay.addLayout(head)

        if not boxes:
            lay.addWidget(static_text(card, "No objects detected", 13, False, 'fg_dim'))
            page.results_layout.addWidget(card)
            return

        confs = [float(b.conf[0]) for b in boxes]
        avg = sum(confs) / len(confs)
        counts = {}
        for b in boxes:
            n = result.names[int(b.cls[0])]
            counts[n] = counts.get(n, 0) + 1
        cls_text = ", ".join(f"{n.capitalize()} ×{c}" for n, c in
                             sorted(counts.items(), key=lambda x: x[1], reverse=True))

        conf_color = ('success' if avg > 0.7 else 'warning' if avg > 0.4 else 'error')
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(static_text(card, "Confidence:", 13, False, 'fg_dim'))
        row.addWidget(make_gauge(card, avg, conf_color, 100))
        row.addWidget(static_text(card, f"avg {avg * 100:.1f}% · max {max(confs) * 100:.1f}%",
                                  13, True, conf_color))
        row.addStretch(1)
        lay.addLayout(row)

        cls_st = static_text(card, cls_text, 13, False, 'fg_dim')
        cls_st.setWordWrap(True)
        lay.addWidget(cls_st)

        page.results_layout.addWidget(card)

    def create_detection_card(self, page, index, class_name, conf, x1, y1, x2, y2):
        """Create a detection card for individual detection"""
        card, lay = self._make_card(page, 'mode_webcam')

        header = QHBoxLayout()
        header.setSpacing(8)
        header.addWidget(static_text(card, f"#{index}", 14, True, 'mode_webcam'))
        header.addWidget(static_text(card, class_name.capitalize(), 15, True), 1)
        lay.addLayout(header)

        conf_pct = max(0.0, min(1.0, conf))
        conf_color = ('success' if conf_pct > 0.7 else 'warning' if conf_pct > 0.4 else 'error')

        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(static_text(card, "Confidence:", 13, False, 'fg_dim'))
        row.addWidget(make_gauge(card, conf_pct, conf_color, 120))
        row.addWidget(static_text(card, f"{conf_pct * 100:.1f}%", 13, True, conf_color))
        row.addStretch(1)
        lay.addLayout(row)

        box_st = static_text(card, f"Box: ({x1:.0f}, {y1:.0f}) → ({x2:.0f}, {y2:.0f})",
                             13, False, 'fg_dim')
        lay.addWidget(box_st)

        page.results_layout.addWidget(card)

    # ------------------------------------------------------------------
    # Save / Clear
    # ------------------------------------------------------------------
    def on_save_result(self, evt):
        self.save_result()

    def save_result(self):
        """Save detection result (image, last video frame, or the 2x2 grid)"""
        page = self.active_page
        if page is None or page.last_result is None:
            QMessageBox.warning(self, "Warning", "No result to save!")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Result", "result.jpg",
            "JPEG (*.jpg);;PNG (*.png);;All files (*.*)")
        if not file_path:
            return
        if not Path(file_path).suffix:
            file_path += ".jpg"

        try:
            cv2.imwrite(file_path, cv2.cvtColor(page.last_result, cv2.COLOR_RGB2BGR))
            QMessageBox.information(self, "Success", f"Result saved to:\n{file_path}")
            self.update_status(f"✓ Result saved: {Path(file_path).name}", COLORS['success'])
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save result:\n{str(e)}")

    def on_clear_display(self, evt):
        self.clear_display()

    def clear_display(self):
        """Clear display and results of the active page"""
        page = self.active_page
        if self.is_processing:
            self.paused = False
            self.stop_video = True

        page.display.show_placeholder()
        self.show_no_results(page)

        page.last_result = None
        if page.mode == 'folder':
            self.folder_path = None
            self.folder_images = []
            page.shuffle_btn.setEnabled(False)
        else:
            self.current_file = None
        self._set_file_label(page, "No file selected", 'error')
        if page.save_btn:
            page.save_btn.setEnabled(False)
        self.update_status("Display cleared", COLORS['fg_dim'])

    # ------------------------------------------------------------------
    def closeEvent(self, event):
        self._closing = True
        self.stop_video = True
        self.paused = False
        for page in self.pages.values():
            page.progress.stop()
        if self._splash is not None:
            self._splash.close_fade()
        t = self.video_thread
        if t is not None and t.is_alive():
            t.join(timeout=2)
        event.accept()


# ============================================================
# Application setup
# ============================================================
def apply_theme(app):
    app.setStyle("Fusion")   # same rendering engine on Windows / macOS / Linux

    font = QFont()
    font.setFamilies(["Segoe UI", "SF Pro Text", "Helvetica Neue", "Inter",
                      "Noto Sans", "Ubuntu", "Arial"])
    font.setPixelSize(14)
    app.setFont(font)

    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(C('bg')))
    pal.setColor(QPalette.WindowText, QColor(C('fg')))
    pal.setColor(QPalette.Base, QColor(C('surface')))
    pal.setColor(QPalette.AlternateBase, QColor(C('bg_lighter')))
    pal.setColor(QPalette.Text, QColor(C('fg')))
    pal.setColor(QPalette.Button, QColor(C('surface')))
    pal.setColor(QPalette.ButtonText, QColor(C('fg')))
    pal.setColor(QPalette.ToolTipBase, QColor('#1e293b'))
    pal.setColor(QPalette.ToolTipText, QColor('#ffffff'))
    pal.setColor(QPalette.Highlight, QColor(C('accent')))
    pal.setColor(QPalette.HighlightedText, QColor('#ffffff'))
    pal.setColor(QPalette.PlaceholderText, QColor(C('fg_dim')))
    app.setPalette(pal)

    app.setStyleSheet(build_qss())


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("YOLO Object Detection (ONNX)")
    apply_theme(app)

    splash = SplashScreen()
    splash.show()
    app.processEvents()

    holder = {}

    def create_main():
        # Built (hidden) while the splash animates; shown once the model is ready
        frame = YOLODetectorFrame()
        holder['frame'] = frame
        frame.begin_startup(splash)

    QTimer.singleShot(80, create_main)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()