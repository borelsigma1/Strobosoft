"""Bildanzeige mit Zoom und Mauseingaben in Bildkoordinaten."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView


def bgr_to_qimage(img: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(img[..., ::-1])
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class ImageView(QGraphicsView):
    """Zeigt ein Bild an. Modi:

    * ``"none"``  – nur ansehen (Mausrad zoomt, Ziehen verschiebt)
    * ``"point"`` – Klick liefert ``pointClicked(x, y)``
    * ``"rect"``  – Rahmen ziehen liefert ``rectDrawn(x, y, w, h)``
    """

    pointClicked = Signal(float, float)
    rectDrawn = Signal(int, int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self._pix = QGraphicsPixmapItem()
        self._pix.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.scene().addItem(self._pix)
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setBackgroundBrush(QColor(40, 40, 40))
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self._mode = "none"
        self._rect_start: QPointF | None = None
        self._rect_item = None
        self._has_image = False
        self._auto_fit = True

    # --- Bild -------------------------------------------------------------
    def set_image(self, img: np.ndarray | None) -> None:
        if img is None:
            self._pix.setPixmap(QPixmap())
            self._has_image = False
            return
        pixmap = QPixmap.fromImage(bgr_to_qimage(img))
        first = not self._has_image or self._pix.pixmap().size() != pixmap.size()
        self._pix.setPixmap(pixmap)
        self.scene().setSceneRect(QRectF(self._pix.pixmap().rect()))
        self._has_image = True
        if first:
            self.fit()

    def fit(self) -> None:
        if self._has_image:
            self.fitInView(self._pix, Qt.AspectRatioMode.KeepAspectRatio)
            self._auto_fit = True

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._auto_fit:
            self.fit()

    def wheelEvent(self, event):
        if not self._has_image:
            return
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)
        self._auto_fit = False

    # --- Modus ------------------------------------------------------------
    def set_mode(self, mode: str) -> None:
        self._mode = mode
        if mode == "none":
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)

    @property
    def mode(self) -> str:
        return self._mode

    def mousePressEvent(self, event):
        if self._mode == "none" or event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        p = self.mapToScene(event.position().toPoint())
        if self._mode == "point":
            self.pointClicked.emit(p.x(), p.y())
        elif self._mode == "rect":
            self._rect_start = p
            pen = QPen(QColor(0, 200, 255), 0)
            pen.setCosmetic(True)
            pen.setWidth(2)
            self._rect_item = self.scene().addRect(QRectF(p, p), pen)

    def mouseMoveEvent(self, event):
        if self._mode == "rect" and self._rect_start is not None and self._rect_item is not None:
            p = self.mapToScene(event.position().toPoint())
            self._rect_item.setRect(QRectF(self._rect_start, p).normalized())
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._mode == "rect" and self._rect_start is not None:
            r = self._rect_item.rect() if self._rect_item else QRectF()
            if self._rect_item is not None:
                self.scene().removeItem(self._rect_item)
            self._rect_item = None
            self._rect_start = None
            if r.width() >= 4 and r.height() >= 4:
                self.rectDrawn.emit(int(r.x()), int(r.y()), int(r.width()), int(r.height()))
            return
        super().mouseReleaseEvent(event)
