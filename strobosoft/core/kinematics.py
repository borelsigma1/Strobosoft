"""Maßstab, Koordinatensystem und Geschwindigkeitsabschätzung."""

from __future__ import annotations

import math
from collections.abc import Sequence

from .models import Calibration, KinematicPoint, ObjectSample

Point = tuple[float, float]


def scale_from_points(p1: Point, p2: Point, length_m: float) -> float:
    """Pixel pro Meter aus zwei Bildpunkten mit bekanntem Abstand."""
    if length_m <= 0:
        raise ValueError("Die Länge muss positiv sein")
    d = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    if d < 1e-9:
        raise ValueError("Die beiden Punkte fallen zusammen")
    return d / length_m


def angle_from_points(origin: Point, on_x_axis: Point) -> float:
    """Winkel der x-Achse (Grad, gegen den Uhrzeigersinn) aus Ursprung und
    einem Punkt auf der positiven x-Achse."""
    dx = on_x_axis[0] - origin[0]
    dy = -(on_x_axis[1] - origin[1])  # Bild-y zeigt nach unten
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        raise ValueError("Achsenpunkt fällt mit dem Ursprung zusammen")
    return math.degrees(math.atan2(dy, dx))


def pixel_to_world(cal: Calibration, p: Point) -> Point:
    dx = p[0] - cal.origin[0]
    dy = -(p[1] - cal.origin[1])
    a = math.radians(cal.angle_deg)
    x = (dx * math.cos(a) + dy * math.sin(a)) / cal.pixels_per_meter
    y = (-dx * math.sin(a) + dy * math.cos(a)) / cal.pixels_per_meter
    return (x, y)


def world_vector_to_pixel(cal: Calibration, v: Point) -> Point:
    """Richtungsvektor (Weltkoordinaten) -> Bildvektor in Pixeln pro Einheit."""
    a = math.radians(cal.angle_deg)
    mx = v[0] * math.cos(a) - v[1] * math.sin(a)
    my = v[0] * math.sin(a) + v[1] * math.cos(a)
    return (mx * cal.pixels_per_meter, -my * cal.pixels_per_meter)


def world_to_pixel(cal: Calibration, p: Point) -> Point:
    dx, dy = world_vector_to_pixel(cal, p)
    return (cal.origin[0] + dx, cal.origin[1] + dy)


def compute_kinematics(
    samples: Sequence[ObjectSample],
    cal: Calibration,
    method: str = "central",
) -> list[KinematicPoint]:
    """Ortsvektoren und Geschwindigkeiten der gefundenen Proben.

    ``method``:
      * ``"central"`` – v_i = (r_{i+1} - r_{i-1}) / (t_{i+1} - t_{i-1}),
        am Rand einseitig. Beste Schätzung der Momentangeschwindigkeit.
      * ``"forward"`` – v_i = (r_{i+1} - r_i) / (t_{i+1} - t_i), also die
        mittlere Geschwindigkeit bis zum nächsten Bild (wie im Unterricht
        häufig per Hand ausgewertet). Der letzte Punkt erhält keinen Pfeil.
    """
    found = [s for s in samples if s.found]
    pos = [pixel_to_world(cal, s.position) for s in found]
    times = [s.time for s in found]
    n = len(found)
    out: list[KinematicPoint] = []
    for i, s in enumerate(found):
        vx = vy = None
        if method == "forward":
            j, k = i, i + 1
        else:
            j, k = max(0, i - 1), min(n - 1, i + 1)
        if 0 <= j < k < n and times[k] > times[j]:
            dt = times[k] - times[j]
            vx = (pos[k][0] - pos[j][0]) / dt
            vy = (pos[k][1] - pos[j][1]) / dt
        out.append(
            KinematicPoint(s.frame_index, s.time, s.position, pos[i][0], pos[i][1], vx, vy)
        )
    return out


def auto_arrow_scale(points: Sequence[KinematicPoint], fraction: float = 0.8) -> float:
    """Pfeilmaßstab (Pixel pro m/s) für gut lesbare Pfeile.

    Ein Pfeil ist dann etwa ``fraction`` mal so lang wie der typische Abstand
    zweier aufeinanderfolgender Stroboskop-Bilder, sodass sich Pfeile und
    Objekte kaum überdecken.
    """
    gaps = [
        math.hypot(b.pixel[0] - a.pixel[0], b.pixel[1] - a.pixel[1])
        for a, b in zip(points, points[1:])
    ]
    speeds = [p.speed for p in points if p.speed]
    if not gaps or not speeds:
        return 1.0
    gaps.sort()
    speeds.sort()
    gap = gaps[len(gaps) // 2]
    speed = speeds[len(speeds) // 2]
    return fraction * gap / speed if speed > 0 and gap > 0 else 1.0
