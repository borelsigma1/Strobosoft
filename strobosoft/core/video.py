"""Video-Ein- und -Ausgabe über OpenCV."""

from __future__ import annotations

from collections.abc import Container, Iterator
from pathlib import Path

import cv2
import numpy as np

from .models import VideoInfo


class VideoError(RuntimeError):
    pass


class VideoSource:
    """Liest Einzelbilder aus einer Videodatei.

    ``max_width`` verkleinert sehr große Videos beim Lesen (z. B. 4K), damit
    Speicherbedarf und Rechenzeit im Rahmen bleiben. Alle Pixelkoordinaten
    beziehen sich dann auf die verkleinerte Auflösung.
    """

    def __init__(self, path: str | Path, max_width: int | None = 1920):
        self.path = str(path)
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise VideoError(f"Video konnte nicht geöffnet werden: {self.path}")
        fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0 or not np.isfinite(fps):
            fps = 30.0
        count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._scale = 1.0
        if max_width and w > max_width:
            self._scale = max_width / w
            w = int(round(w * self._scale))
            h = int(round(h * self._scale))
        if count <= 0:
            count = self._count_frames()
        self.info = VideoInfo(self.path, fps, count, w, h)
        self._next_index = 0

    def _count_frames(self) -> int:
        n = 0
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while self._cap.grab():
            n += 1
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return n

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        if self._scale != 1.0:
            frame = cv2.resize(
                frame, (self.info.width, self.info.height), interpolation=cv2.INTER_AREA
            )
        return frame

    def read(self, index: int) -> np.ndarray:
        """Liest das Bild mit dem Index ``index`` (0-basiert)."""
        if not 0 <= index < self.info.frame_count:
            raise IndexError(index)
        if index != self._next_index:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self._cap.read()
        if not ok:
            raise VideoError(f"Bild {index} konnte nicht gelesen werden")
        self._next_index = index + 1
        return self._prepare(frame)

    def iter_range(
        self, start: int, stop: int, wanted: Container[int] | None = None
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Liest die Bilder ``start`` .. ``stop - 1`` nacheinander.

        Mit ``wanted`` werden nur diese Bilder ausgegeben; die übrigen
        werden nur dekodiert (``grab``), was schneller ist. Sequentielles
        Lesen ist bei komprimierten Videos bildgenau – Springen mit
        ``CAP_PROP_POS_FRAMES`` ist das nicht bei jedem Codec.
        """
        stop = min(stop, self.info.frame_count)
        if start >= stop:
            return
        if start != self._next_index:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            self._next_index = start
        for i in range(start, stop):
            if wanted is not None and i not in wanted:
                ok = self._cap.grab()
                self._next_index = i + 1
                if not ok:
                    break
                continue
            ok, frame = self._cap.read()
            if not ok:
                break
            self._next_index = i + 1
            yield i, self._prepare(frame)

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def write_image(path: str | Path, image: np.ndarray) -> None:
    """Speichert ein Bild; funktioniert auch mit Umlauten im Pfad (Windows)."""
    path = Path(path)
    ext = path.suffix.lower() or ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise VideoError(f"Bild konnte nicht kodiert werden: {path}")
    path.write_bytes(buf.tobytes())
