"""Einblendungen (Achsen, Punkte, Geschwindigkeitspfeile) als Datenliste.

Die Einblendungen werden zuerst als einfache Grafik-Primitive beschrieben
und danach entweder ins Rasterbild gezeichnet (PNG, Vorschau) oder als
Vektorgrafik ausgegeben (SVG). So sehen beide Exporte gleich aus.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np

from .kinematics import world_vector_to_pixel
from .models import Calibration, KinematicPoint

Color = tuple[int, int, int]  # RGB


@dataclass(frozen=True)
class Line:
    p1: tuple[float, float]
    p2: tuple[float, float]
    color: Color
    width: float
    arrow: bool = False  # Pfeilspitze bei p2


@dataclass(frozen=True)
class Circle:
    center: tuple[float, float]
    radius: float
    color: Color
    width: float  # 0 = gefüllt


@dataclass(frozen=True)
class Text:
    pos: tuple[float, float]  # linke Grundlinie
    text: str
    color: Color
    size: float  # Schrifthöhe in Pixeln


Primitive = Line | Circle | Text


@dataclass
class OverlayOptions:
    show_points: bool = True
    show_vectors: bool = True
    show_axes: bool = True
    show_times: bool = False
    show_speeds: bool = False
    show_legend: bool = True  # Referenzpfeil, z. B. "1 m/s"
    arrow_scale: float = 1.0  # Pixel pro (m/s)
    point_color: Color = (255, 220, 0)
    vector_color: Color = (230, 30, 30)
    axis_color: Color = (30, 90, 230)
    text_color: Color = (20, 20, 20)
    line_width: float | None = None  # None = abhängig von der Bildgröße
    decimal_comma: bool = True
    extra: dict = field(default_factory=dict)


def _fmt(v: float, decimals: int, comma: bool) -> str:
    s = f"{v:.{decimals}f}"
    return s.replace(".", ",") if comma else s


def nice_step(raw: float) -> float:
    """Nächstgrößerer "schöner" Wert (1, 2, 5 · 10^k)."""
    if raw <= 0:
        return 1.0
    exp = math.floor(math.log10(raw))
    base = raw / 10**exp
    for m in (1, 2, 5, 10):
        if base <= m:
            return m * 10**exp
    return 10 ** (exp + 1)


def nice_floor(raw: float) -> float:
    """Nächstkleinerer "schöner" Wert (1, 2, 5 · 10^k)."""
    if raw <= 0:
        return 1.0
    exp = math.floor(math.log10(raw))
    base = raw / 10**exp
    for m in (5, 2, 1):
        if base >= m:
            return m * 10**exp
    return 10**exp


def _axis_range(origin, u, width, height) -> tuple[float, float] | None:
    """Parameterbereich t, für den origin + t*u im Bild liegt."""
    lo, hi = -math.inf, math.inf
    for o, d, size in ((origin[0], u[0], width), (origin[1], u[1], height)):
        if abs(d) < 1e-12:
            if not 0 <= o <= size:
                return None
            continue
        t1, t2 = (0 - o) / d, (size - o) / d
        lo, hi = max(lo, min(t1, t2)), min(hi, max(t1, t2))
    if lo >= hi:
        return None
    return lo, hi


def build_overlay(
    size: tuple[int, int],
    cal: Calibration,
    points: Sequence[KinematicPoint],
    opt: OverlayOptions,
) -> list[Primitive]:
    width, height = size
    lw = opt.line_width or max(1.5, min(width, height) / 300)
    font = max(11.0, min(width, height) / 40)
    prims: list[Primitive] = []
    unit = "m" if cal.is_set else "px"

    if opt.show_axes and cal.is_set:
        tick_len = 2.5 * lw
        step_m = nice_step(70 / cal.pixels_per_meter)
        decimals = max(0, -int(math.floor(math.log10(step_m)))) if step_m < 1 else 0
        for axis, name in (((1.0, 0.0), "x"), ((0.0, 1.0), "y")):
            u = world_vector_to_pixel(cal, axis)  # Pixel pro Meter
            rng = _axis_range(cal.origin, u, width, height)
            if rng is None:
                continue
            lo, hi = rng
            p1 = (cal.origin[0] + lo * u[0], cal.origin[1] + lo * u[1])
            p2 = (cal.origin[0] + hi * u[0], cal.origin[1] + hi * u[1])
            prims.append(Line(p1, p2, opt.axis_color, lw, arrow=True))
            norm = math.hypot(*u)
            nx, ny = -u[1] / norm, u[0] / norm  # Normale für Teilstriche
            k0, k1 = math.ceil(lo / step_m), math.floor(hi / step_m)
            for k in range(k0, k1 + 1):
                if k == 0:
                    continue
                t = k * step_m
                c = (cal.origin[0] + t * u[0], cal.origin[1] + t * u[1])
                prims.append(Line(
                    (c[0] - nx * tick_len, c[1] - ny * tick_len),
                    (c[0] + nx * tick_len, c[1] + ny * tick_len),
                    opt.axis_color, lw,
                ))
                prims.append(Text(
                    (c[0] + nx * 3 * tick_len + 2, c[1] + ny * 3 * tick_len + font / 3),
                    _fmt(t, decimals, opt.decimal_comma), opt.axis_color, font * 0.8,
                ))
            ux, uy = u[0] / norm, u[1] / norm
            text = f"{name} in {unit}"
            tw = text_width(text, font)
            # neben der Pfeilspitze, auf der den Teilstrich-Beschriftungen abgewandten Seite
            cx = p2[0] - ux * (tw / 2 + font) - nx * 1.6 * font
            cy = p2[1] - uy * (tw / 2 + font) - ny * 1.6 * font
            prims.append(Text((cx - tw / 2, cy + font / 3), text, opt.axis_color, font))
        prims.append(Circle(cal.origin, 2 * lw, opt.axis_color, 0))

    for p in points:
        if opt.show_vectors and p.vx is not None and p.vy is not None:
            dpx = world_vector_to_pixel(cal, (p.vx, p.vy))
            ppm = cal.pixels_per_meter
            ex = p.pixel[0] + dpx[0] / ppm * opt.arrow_scale
            ey = p.pixel[1] + dpx[1] / ppm * opt.arrow_scale
            if math.hypot(ex - p.pixel[0], ey - p.pixel[1]) >= 1:
                prims.append(Line(p.pixel, (ex, ey), opt.vector_color, lw * 1.4, arrow=True))
        if opt.show_points:
            prims.append(Circle(p.pixel, 2.2 * lw, opt.point_color, 0))
            prims.append(Circle(p.pixel, 2.2 * lw, (0, 0, 0), max(1.0, lw / 2)))
        labels = []
        if opt.show_times:
            labels.append(f"t = {_fmt(p.time, 2, opt.decimal_comma)} s")
        if opt.show_speeds and p.speed is not None:
            labels.append(f"|v| = {_fmt(p.speed, 2, opt.decimal_comma)} {unit}/s")
        for i, text in enumerate(labels):
            prims.append(Text(
                (p.pixel[0] + 4 * lw, p.pixel[1] + 5 * lw + font + i * font * 1.15),
                text, opt.text_color, font * 0.85,
            ))
    if opt.show_vectors and opt.show_legend and opt.arrow_scale > 0 and any(p.vx is not None for p in points):
        # Referenzpfeil unten rechts: ein "schöner" Geschwindigkeitswert, ca. 12 % der Bildbreite
        v_ref = nice_floor(0.12 * width / opt.arrow_scale)
        length = v_ref * opt.arrow_scale
        decimals = 0 if v_ref >= 1 else max(1, -int(math.floor(math.log10(v_ref))))
        label = f"{_fmt(v_ref, decimals, opt.decimal_comma)} {unit}/s"
        x1 = width - 12 - length
        y = height - 14
        prims.append(Line((x1, y), (x1 + length, y), opt.vector_color, lw * 1.4, arrow=True))
        prims.append(Text((x1, y - 2 * lw - 4), label, opt.vector_color, font * 0.9))
    return [_clamp_text(p, width, height) if isinstance(p, Text) else p for p in prims]


def text_width(text: str, size: float) -> float:
    """Grobe Textbreite (Pixel) für die Platzierung."""
    return 0.58 * size * len(text)


def _clamp_text(t: Text, width: int, height: int) -> Text:
    tw = text_width(t.text, t.size)
    x = min(max(2.0, t.pos[0]), max(2.0, width - tw - 2))
    y = min(max(t.size + 2, t.pos[1]), height - 0.3 * t.size)
    return Text((x, y), t.text, t.color, t.size)


def _bgr(c: Color) -> tuple[int, int, int]:
    return (int(c[2]), int(c[1]), int(c[0]))


def render_raster(image: np.ndarray, prims: Sequence[Primitive]) -> np.ndarray:
    """Zeichnet die Primitive in eine Kopie des Bildes (BGR)."""
    out = image.copy()
    sub = 4  # Subpixel-Genauigkeit für cv2
    f = 1 << sub

    def P(p):
        return (int(round(p[0] * f)), int(round(p[1] * f)))

    for pr in prims:
        if isinstance(pr, Line):
            t = max(1, int(round(pr.width)))
            if pr.arrow:
                length = math.hypot(pr.p2[0] - pr.p1[0], pr.p2[1] - pr.p1[1])
                tip = min(0.35, (4 * pr.width + 6) / max(length, 1e-6))
                cv2.arrowedLine(out, P(pr.p1), P(pr.p2), _bgr(pr.color), t, cv2.LINE_AA, sub, tip)
            else:
                cv2.line(out, P(pr.p1), P(pr.p2), _bgr(pr.color), t, cv2.LINE_AA, sub)
        elif isinstance(pr, Circle):
            t = -1 if pr.width <= 0 else max(1, int(round(pr.width)))
            cv2.circle(out, P(pr.center), int(round(pr.radius * f)), _bgr(pr.color), t, cv2.LINE_AA, sub)
        elif isinstance(pr, Text):
            scale = pr.size / 22.0
            th = max(1, int(round(pr.size / 14)))
            org = (int(round(pr.pos[0])), int(round(pr.pos[1])))
            # heller Rand für Lesbarkeit auf jedem Hintergrund
            cv2.putText(out, pr.text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), th + 3, cv2.LINE_AA)
            cv2.putText(out, pr.text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, _bgr(pr.color), th, cv2.LINE_AA)
    return out
