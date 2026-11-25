"""Export: PNG (Rasterbild mit Einblendungen), SVG (Bild + Vektorgrafik), CSV."""

from __future__ import annotations

import base64
import csv
import io
import math
from collections.abc import Sequence
from pathlib import Path
from xml.sax.saxutils import escape

import cv2
import numpy as np

from .models import Calibration, KinematicPoint
from .overlay import Circle, Line, Primitive, Text, render_raster
from .video import write_image


def export_png(path: str | Path, image: np.ndarray, prims: Sequence[Primitive]) -> None:
    write_image(path, render_raster(image, prims))


def _rgb(c) -> str:
    return f"rgb({int(c[0])},{int(c[1])},{int(c[2])})"


def svg_document(image: np.ndarray, prims: Sequence[Primitive]) -> str:
    """SVG mit eingebettetem Stroboskopbild und Einblendungen als Vektoren.

    Die Vektorelemente lassen sich z. B. in Inkscape oder PowerPoint
    nachträglich bearbeiten (Farben, Strichstärken, Beschriftungen).
    """
    h, w = image.shape[:2]
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("PNG-Kodierung fehlgeschlagen")
    data = base64.b64encode(buf.tobytes()).decode("ascii")
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        f'<image id="stroboskopbild" x="0" y="0" width="{w}" height="{h}" '
        f'xlink:href="data:image/png;base64,{data}"/>',
    ]
    markers: dict[str, str] = {}
    body: list[str] = []
    for pr in prims:
        if isinstance(pr, Line):
            attrs = (
                f'x1="{pr.p1[0]:.2f}" y1="{pr.p1[1]:.2f}" x2="{pr.p2[0]:.2f}" y2="{pr.p2[1]:.2f}" '
                f'stroke="{_rgb(pr.color)}" stroke-width="{pr.width:.2f}" stroke-linecap="round"'
            )
            if pr.arrow:
                mid = "arrow_" + "_".join(str(int(c)) for c in pr.color)
                markers[mid] = (
                    f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" '
                    f'markerWidth="4" markerHeight="4" orient="auto-start-reverse">'
                    f'<path d="M0,0 L10,5 L0,10 z" fill="{_rgb(pr.color)}"/></marker>'
                )
                attrs += f' marker-end="url(#{mid})"'
            body.append(f"<line {attrs}/>")
        elif isinstance(pr, Circle):
            if pr.width <= 0:
                style = f'fill="{_rgb(pr.color)}"'
            else:
                style = f'fill="none" stroke="{_rgb(pr.color)}" stroke-width="{pr.width:.2f}"'
            body.append(
                f'<circle cx="{pr.center[0]:.2f}" cy="{pr.center[1]:.2f}" r="{pr.radius:.2f}" {style}/>'
            )
        elif isinstance(pr, Text):
            body.append(
                f'<text x="{pr.pos[0]:.2f}" y="{pr.pos[1]:.2f}" font-family="Arial, Helvetica, sans-serif" '
                f'font-size="{pr.size:.1f}" fill="{_rgb(pr.color)}" stroke="white" '
                f'stroke-width="{pr.size / 6:.2f}" paint-order="stroke">{escape(pr.text)}</text>'
            )
    if markers:
        parts.append("<defs>" + "".join(markers.values()) + "</defs>")
    parts.append('<g id="einblendungen">')
    parts.extend(body)
    parts.append("</g></svg>")
    return "\n".join(parts)


def export_svg(path: str | Path, image: np.ndarray, prims: Sequence[Primitive]) -> None:
    Path(path).write_text(svg_document(image, prims), encoding="utf-8")


def csv_text(
    points: Sequence[KinematicPoint],
    cal: Calibration,
    german: bool = True,
) -> str:
    """Messwerttabelle.

    ``german=True``: Semikolon als Trennzeichen und Dezimalkomma, damit die
    Datei in einem deutschsprachigen Excel/LibreOffice direkt passt.
    """
    unit = "m" if cal.is_set else "px"
    header = [
        "Bild", "t in s", "x_Pixel", "y_Pixel",
        f"x in {unit}", f"y in {unit}",
        f"vx in {unit}/s", f"vy in {unit}/s", f"|v| in {unit}/s",
    ]

    def num(v: float | None, digits: int = 5) -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return ""
        s = f"{v:.{digits}f}"
        return s.replace(".", ",") if german else s

    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=";" if german else ",", lineterminator="\n")
    writer.writerow(header)
    for p in points:
        writer.writerow([
            p.frame_index, num(p.time, 4), num(p.pixel[0], 1), num(p.pixel[1], 1),
            num(p.x), num(p.y), num(p.vx), num(p.vy), num(p.speed),
        ])
    return buf.getvalue()


def export_csv(path: str | Path, points: Sequence[KinematicPoint], cal: Calibration, german: bool = True) -> None:
    # utf-8-sig, damit Excel Umlaute in den Spaltenköpfen richtig anzeigt
    Path(path).write_text(csv_text(points, cal, german), encoding="utf-8-sig")
