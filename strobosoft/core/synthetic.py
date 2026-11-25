"""Erzeugt ein Beispielvideo (schiefer Wurf) mit bekannter Bahn.

Dient für Tests und zum Ausprobieren der Software ohne eigenes Video.
Im Hintergrund ist ein Maßstab von 1 m eingezeichnet.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class ThrowScene:
    width: int = 960
    height: int = 540
    fps: float = 30.0
    duration: float = 1.0
    pixels_per_meter: float = 200.0
    start_px: tuple[float, float] = (90.0, 330.0)
    v0: tuple[float, float] = (3.5, 4.2)  # m/s, y nach oben
    g: float = 9.81
    radius: int = 16

    @property
    def frame_count(self) -> int:
        return int(round(self.duration * self.fps))

    def position_px(self, t: float) -> tuple[float, float]:
        x = self.v0[0] * t
        y = self.v0[1] * t - 0.5 * self.g * t * t
        return (self.start_px[0] + x * self.pixels_per_meter, self.start_px[1] - y * self.pixels_per_meter)

    def velocity(self, t: float) -> tuple[float, float]:
        return (self.v0[0], self.v0[1] - self.g * t)

    @property
    def ruler(self) -> tuple[tuple[float, float], tuple[float, float]]:
        y = self.height - 30.0
        return ((60.0, y), (60.0 + self.pixels_per_meter, y))


def scene_background(scene: ThrowScene, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = scene.height, scene.width
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w, 3), np.float32)
    img[..., 0] = 150 + 40 * xx / w
    img[..., 1] = 170 + 30 * yy / h
    img[..., 2] = 175
    for _ in range(25):  # ein paar "Möbel"
        x, y = int(rng.integers(0, w - 60)), int(rng.integers(0, h - 60))
        cw, ch = int(rng.integers(20, 120)), int(rng.integers(20, 120))
        color = rng.integers(90, 230, 3).tolist()
        cv2.rectangle(img, (x, y), (x + cw, y + ch), color, -1)
    for gx in range(0, w, 40):
        cv2.line(img, (gx, 0), (gx, h), (200, 200, 200), 1)
    img += rng.normal(0, 3, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    (x1, y1), (x2, y2) = scene.ruler
    cv2.rectangle(img, (int(x1) - 6, int(y1) - 14), (int(x2) + 6, int(y1) + 10), (250, 250, 250), -1)
    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 2)
    for x in (x1, x2):
        cv2.line(img, (int(x), int(y1) - 10), (int(x), int(y1) + 6), (0, 0, 0), 2)
    cv2.putText(img, "1 m", (int((x1 + x2) / 2) - 18, int(y1) - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def render_frame(scene: ThrowScene, background: np.ndarray, t: float, rng: np.random.Generator) -> np.ndarray:
    frame = background.astype(np.int16)
    frame += rng.normal(0, 2, frame.shape).astype(np.int16)  # Sensorrauschen
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    cx, cy = scene.position_px(t)
    sub = 4
    center = (int(round(cx * (1 << sub))), int(round(cy * (1 << sub))))
    cv2.circle(frame, center, scene.radius << sub, (20, 60, 220), -1, cv2.LINE_AA, sub)
    cv2.circle(frame, center, (scene.radius // 3) << sub, (90, 140, 255), -1, cv2.LINE_AA, sub)
    return frame


def write_demo_video(path: str | Path, scene: ThrowScene = ThrowScene(), backend: str | None = None) -> Path:
    """Schreibt das Beispielvideo und gibt den tatsächlichen Pfad zurück.

    Bevorzugt PyAV (H.264/MPEG-4 als .mp4); ohne PyAV OpenCV, notfalls
    als .avi mit MJPG.
    """
    path = Path(path)
    bg = scene_background(scene)
    rng = np.random.default_rng(7)
    frames = (render_frame(scene, bg, i / scene.fps, rng) for i in range(scene.frame_count))
    if backend in (None, "pyav"):
        try:
            import av
        except ImportError:
            av = None
        if av is not None:
            out_path = path.with_suffix(".mp4")
            for codec in ("libx264", "h264", "mpeg4"):
                try:
                    with av.open(str(out_path), "w") as container:
                        stream = container.add_stream(codec, rate=int(round(scene.fps)))
                        stream.width, stream.height = scene.width, scene.height
                        stream.pix_fmt = "yuv420p"
                        if codec == "libx264":
                            stream.options = {"crf": "18"}
                        for img in frames:
                            frame = av.VideoFrame.from_ndarray(img, format="bgr24")
                            for packet in stream.encode(frame):
                                container.mux(packet)
                        for packet in stream.encode():
                            container.mux(packet)
                    return out_path
                except Exception:
                    rng = np.random.default_rng(7)
                    frames = (render_frame(scene, bg, i / scene.fps, rng) for i in range(scene.frame_count))
                    continue
    for suffix, fourcc in ((path.suffix or ".mp4", "mp4v"), (".avi", "MJPG")):
        out_path = path.with_suffix(suffix)
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*fourcc), scene.fps, (scene.width, scene.height))
        if writer.isOpened():
            break
        writer.release()
    else:
        raise RuntimeError("Kein Video-Codec zum Schreiben verfügbar")
    try:
        for img in frames:
            writer.write(img)
    finally:
        writer.release()
    return out_path
