"""Datenmodelle des Verarbeitungskerns.

Bewusst schlichte Datenklassen ohne Logik-Abhängigkeiten, damit sie sich
1:1 in Rust-Structs übertragen lassen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


@dataclass(frozen=True)
class BBox:
    """Achsenparalleles Rechteck in Pixeln (x, y = linke obere Ecke)."""

    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    def clip(self, width: int, height: int) -> "BBox":
        x0 = max(0, min(self.x, width))
        y0 = max(0, min(self.y, height))
        x1 = max(x0, min(self.x + self.w, width))
        y1 = max(y0, min(self.y + self.h, height))
        return BBox(x0, y0, x1 - x0, y1 - y0)

    def expand(self, margin: int) -> "BBox":
        return BBox(self.x - margin, self.y - margin, self.w + 2 * margin, self.h + 2 * margin)

    @property
    def is_empty(self) -> bool:
        return self.w <= 0 or self.h <= 0


@dataclass
class ObjectSample:
    """Das Objekt zu einem Zeitpunkt (ein Stroboskop-"Blitz").

    ``bbox`` umschließt das Objekt. ``patch`` und ``alpha`` sind etwas größere
    Ausschnitte (mit weichem Rand) mit der linken oberen Ecke bei
    ``patch_origin``; ``alpha`` (float32, 0..1) bestimmt, wie stark das Objekt beim
    Komponieren den Hintergrund überdeckt.
    """

    frame_index: int
    time: float
    position: tuple[float, float]  # Schwerpunkt in Pixeln
    bbox: BBox
    patch: np.ndarray = field(repr=False)
    alpha: np.ndarray = field(repr=False)
    patch_origin: tuple[int, int] = (0, 0)
    found: bool = True


@dataclass
class Track:
    samples: list[ObjectSample] = field(default_factory=list)

    @property
    def found_samples(self) -> list[ObjectSample]:
        return [s for s in self.samples if s.found]


@dataclass(frozen=True)
class Calibration:
    """Maßstab und Koordinatensystem.

    ``pixels_per_meter``: Pixel pro Meter.
    ``origin``: Ursprung in Pixelkoordinaten.
    ``angle_deg``: Richtung der positiven x-Achse im Bild, gemessen gegen
    den Uhrzeigersinn (wie in der Mathematik, also mit y nach oben).
    Die y-Achse steht senkrecht dazu und zeigt (im Bild betrachtet) nach
    "oben" relativ zur x-Achse.
    """

    pixels_per_meter: float = 1.0
    origin: tuple[float, float] = (0.0, 0.0)
    angle_deg: float = 0.0
    is_set: bool = False


@dataclass(frozen=True)
class KinematicPoint:
    frame_index: int
    time: float
    pixel: tuple[float, float]
    x: float  # in m (bzw. px ohne Kalibrierung)
    y: float
    vx: float | None  # in m/s
    vy: float | None

    @property
    def speed(self) -> float | None:
        if self.vx is None or self.vy is None:
            return None
        return float(np.hypot(self.vx, self.vy))
