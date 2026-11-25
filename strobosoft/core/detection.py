"""Objekterkennung per Hintergrundsubtraktion.

Für jedes Bild wird die Differenz zum Median-Hintergrund berechnet. Pixel
mit großer Differenz gehören zum bewegten Objekt. Aus der Maske wird ein
zusammenhängender Bereich (Blob) ausgewählt.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .models import BBox, ObjectSample


@dataclass(frozen=True)
class DetectionParams:
    threshold: int | None = None  # None = automatisch (Otsu)
    min_area: int = 30  # kleinste Blobfläche in Pixeln
    blur: int = 5  # Glättung vor dem Schwellwert (ungerade)
    morph: int = 5  # Größe für Öffnen/Schließen der Maske
    feather: int = 3  # weiche Kante beim Einfügen (Pixel)
    max_jump: float | None = None  # max. Sprung zwischen zwei Proben (px)


def difference_image(frame: np.ndarray, background: np.ndarray, blur: int = 5) -> np.ndarray:
    """Maximale Farbkanal-Differenz zum Hintergrund (uint8)."""
    diff = cv2.absdiff(frame, background)
    if diff.ndim == 3:
        diff = diff.max(axis=2)
    if blur and blur > 1:
        k = blur | 1
        diff = cv2.GaussianBlur(diff, (k, k), 0)
    return diff


def auto_threshold(diff: np.ndarray, floor: int = 20) -> int:
    t, _ = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(max(floor, t))


def foreground_mask(
    frame: np.ndarray, background: np.ndarray, params: DetectionParams
) -> np.ndarray:
    diff = difference_image(frame, background, params.blur)
    t = params.threshold if params.threshold is not None else auto_threshold(diff)
    _, mask = cv2.threshold(diff, t, 255, cv2.THRESH_BINARY)
    if params.morph and params.morph > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params.morph, params.morph))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


@dataclass(frozen=True)
class Blob:
    area: float
    centroid: tuple[float, float]
    bbox: BBox
    contour: np.ndarray


def find_blobs(mask: np.ndarray, min_area: int) -> list[Blob]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in contours:
        m = cv2.moments(c)
        area = m["m00"]
        if area < min_area:
            continue
        cx, cy = m["m10"] / area, m["m01"] / area
        x, y, w, h = cv2.boundingRect(c)
        blobs.append(Blob(area, (cx, cy), BBox(x, y, w, h), c))
    return blobs


def choose_blob(
    blobs: list[Blob],
    previous: tuple[float, float] | None,
    max_jump: float | None,
) -> Blob | None:
    """Größter Blob; liegt eine vorherige Position vor, wird Nähe bevorzugt."""
    if not blobs:
        return None
    if previous is None:
        return max(blobs, key=lambda b: b.area)
    px, py = previous

    def dist(b: Blob) -> float:
        return float(np.hypot(b.centroid[0] - px, b.centroid[1] - py))

    candidates = [b for b in blobs if max_jump is None or dist(b) <= max_jump]
    if not candidates:
        return None
    biggest = max(b.area for b in candidates)
    # Nur ernsthafte Kandidaten (mind. 30 % der größten Fläche), davon der nächste.
    serious = [b for b in candidates if b.area >= 0.3 * biggest]
    return min(serious, key=dist)


def make_sample(
    frame: np.ndarray,
    mask: np.ndarray,
    frame_index: int,
    time: float,
    feather: int = 3,
    position: tuple[float, float] | None = None,
) -> ObjectSample | None:
    """Schneidet das Objekt (Maske ``mask``, volle Bildgröße) aus ``frame`` aus."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    h, w = mask.shape[:2]
    pad = max(2, 2 * feather)
    bbox = BBox(int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    box = bbox.expand(pad).clip(w, h)
    sl = (slice(box.y, box.y + box.h), slice(box.x, box.x + box.w))
    hard = (mask[sl] > 0).astype(np.float32)
    alpha = hard
    if feather > 0:
        k = 2 * feather + 1
        soft = cv2.GaussianBlur(hard, (k, k), 0)
        # Kern des Objekts voll deckend lassen, nur den Rand weich machen.
        core = cv2.erode(hard, np.ones((k, k), np.uint8))
        alpha = np.maximum(soft * 1.5, core)
    if position is None:
        position = (float(xs.mean()), float(ys.mean()))
    return ObjectSample(
        frame_index=frame_index,
        time=time,
        position=position,
        bbox=bbox,
        patch=frame[sl].copy(),
        alpha=np.clip(alpha, 0.0, 1.0)[..., None],
        patch_origin=(box.x, box.y),
        found=True,
    )


def missing_sample(frame_index: int, time: float) -> ObjectSample:
    empty = np.zeros((0, 0, 3), np.uint8)
    return ObjectSample(
        frame_index, time, (float("nan"), float("nan")), BBox(0, 0, 0, 0),
        empty, np.zeros((0, 0, 1), np.float32), (0, 0), found=False,
    )


def detect_in_frame(
    frame: np.ndarray,
    background: np.ndarray,
    frame_index: int,
    time: float,
    params: DetectionParams,
    previous: tuple[float, float] | None = None,
    roi: BBox | None = None,
) -> ObjectSample:
    mask = foreground_mask(frame, background, params)
    if roi is not None:
        r = roi.clip(mask.shape[1], mask.shape[0])
        limited = np.zeros_like(mask)
        limited[r.y : r.y + r.h, r.x : r.x + r.w] = mask[r.y : r.y + r.h, r.x : r.x + r.w]
        mask = limited
    blob = choose_blob(find_blobs(mask, params.min_area), previous, params.max_jump)
    if blob is None:
        return missing_sample(frame_index, time)
    blob_mask = np.zeros_like(mask)
    cv2.drawContours(blob_mask, [blob.contour], -1, 255, thickness=cv2.FILLED)
    sample = make_sample(frame, blob_mask, frame_index, time, params.feather, blob.centroid)
    return sample or missing_sample(frame_index, time)
