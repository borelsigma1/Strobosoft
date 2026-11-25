"""Stroboskopbild aus Hintergrund und Objekt-Ausschnitten zusammensetzen."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .models import ObjectSample


def compose_stroboscope(
    background: np.ndarray,
    samples: Iterable[ObjectSample],
    opacity: float = 1.0,
) -> np.ndarray:
    """Legt alle gefundenen Objekt-Ausschnitte auf den Hintergrund.

    Spätere Zeitpunkte liegen oben. ``opacity`` < 1 lässt Überlappungen
    durchscheinen (hilfreich bei langsamen Bewegungen).
    """
    out = background.astype(np.float32)
    for s in sorted(samples, key=lambda s: s.frame_index):
        if not s.found or s.patch.size == 0:
            continue
        h, w = s.patch.shape[:2]
        x0, y0 = s.patch_origin
        region = out[y0 : y0 + h, x0 : x0 + w]
        a = s.alpha[: region.shape[0], : region.shape[1]] * float(opacity)
        p = s.patch[: region.shape[0], : region.shape[1]].astype(np.float32)
        region[:] = region * (1.0 - a) + p * a
    return np.clip(out, 0, 255).astype(np.uint8)

