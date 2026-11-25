"""Manuelle Objektwahl + Verfolgung mit einem OpenCV-Tracker (CSRT).

Fallback, wenn die automatische Erkennung scheitert (z. B. mehrere bewegte
Objekte, Kamera wackelt leicht, Objekt kontrastarm). Die Lehrkraft zieht
im ersten Bild einen Rahmen um das Objekt; der Tracker folgt ihm.
"""

from __future__ import annotations

import cv2
import numpy as np

from .detection import DetectionParams, foreground_mask, make_sample, missing_sample
from .models import BBox, ObjectSample


def create_tracker():
    for factory in (
        getattr(cv2, "TrackerCSRT_create", None),
        getattr(getattr(cv2, "TrackerCSRT", None), "create", None),
        getattr(getattr(cv2, "legacy", None), "TrackerCSRT_create", None),
        getattr(cv2, "TrackerKCF_create", None),
    ):
        if factory is not None:
            try:
                return factory()
            except cv2.error:
                continue
    raise RuntimeError(
        "Kein OpenCV-Tracker verfügbar. Bitte 'opencv-contrib-python' installieren."
    )


class BoxTracker:
    def __init__(self, frame: np.ndarray, bbox: BBox):
        self._tracker = create_tracker()
        self._tracker.init(frame, (int(bbox.x), int(bbox.y), int(bbox.w), int(bbox.h)))
        self.bbox = bbox

    def update(self, frame: np.ndarray) -> BBox | None:
        ok, box = self._tracker.update(frame)
        if not ok:
            return None
        x, y, w, h = (int(round(v)) for v in box)
        self.bbox = BBox(x, y, w, h)
        return self.bbox


def sample_from_box(
    frame: np.ndarray,
    background: np.ndarray | None,
    bbox: BBox,
    frame_index: int,
    time: float,
    params: DetectionParams,
) -> ObjectSample:
    """Erzeugt eine Probe aus einem Tracker-Rahmen.

    Wenn ein Hintergrund vorliegt, wird innerhalb des (etwas vergrößerten)
    Rahmens die Differenzmaske verwendet, sodass nur das Objekt und nicht
    das ganze Rechteck eingefügt wird. Sonst wird das Rechteck übernommen.
    """
    h, w = frame.shape[:2]
    box = bbox.clip(w, h)
    if box.is_empty:
        return missing_sample(frame_index, time)
    mask = np.zeros((h, w), np.uint8)
    if background is not None:
        area = box.expand(max(4, box.w // 6)).clip(w, h)
        sl = (slice(area.y, area.y + area.h), slice(area.x, area.x + area.w))
        local = foreground_mask(frame[sl], background[sl], params)
        mask[sl] = local
        if np.count_nonzero(mask) < max(params.min_area, 0.1 * box.w * box.h):
            mask[:] = 0
    if not mask.any():
        mask[box.y : box.y + box.h, box.x : box.x + box.w] = 255
        position = box.center
    else:
        ys, xs = np.nonzero(mask)
        position = (float(xs.mean()), float(ys.mean()))
    sample = make_sample(frame, mask, frame_index, time, params.feather, position)
    return sample or missing_sample(frame_index, time)
