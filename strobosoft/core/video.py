"""Video-Ein- und -Ausgabe.

Zum Dekodieren gibt es zwei Wege ("Backends"):

* **PyAV** (Paket ``av``, bringt ein vollständiges FFmpeg mit): bevorzugt,
  weil es praktisch alle Formate kann (H.264, HEVC/H.265 vom iPhone, .mov,
  …), auch unter Windows auf ARM-Rechnern, und mit Umlauten im Pfad
  zurechtkommt.
* **OpenCV** (``cv2.VideoCapture``): Ausweichlösung, falls PyAV fehlt oder
  die Datei nicht öffnen kann.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Container, Iterator
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from .models import VideoInfo

try:  # optional, aber empfohlen
    import av  # type: ignore
except ImportError:  # pragma: no cover - abhängig von der Installation
    av = None


class VideoError(RuntimeError):
    pass


def _rotate(frame: np.ndarray, degrees: int) -> np.ndarray:
    """Dreht ein Bild um ein Vielfaches von 90° gegen den Uhrzeigersinn."""
    k = (int(round(degrees / 90.0))) % 4
    return np.ascontiguousarray(np.rot90(frame, k)) if k else frame


# ---------------------------------------------------------------- Backends
class _PyAVReader:
    name = "PyAV/FFmpeg"

    def __init__(self, path: str):
        self._container = av.open(path)
        if not self._container.streams.video:
            self._container.close()
            raise VideoError("Die Datei enthält keine Videospur.")
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = "AUTO"
        s = self._stream
        rate = s.average_rate or s.guessed_rate or s.base_rate
        self.fps = float(rate) if rate else 0.0
        if not self.fps or not np.isfinite(self.fps):
            self.fps = 30.0
        self._tb = s.time_base or Fraction(1, 1000)
        self._start = s.start_time or 0
        count = int(s.frames or 0)
        if count <= 0 and s.duration:
            count = int(round(float(s.duration * self._tb) * self.fps))
        if count <= 0 and self._container.duration:
            count = int(round(self._container.duration / 1_000_000 * self.fps))
        self.frame_count = count
        self._decoder = self._container.decode(self._stream)
        self._pos = 0
        self._pending = None

    def _restart(self) -> None:
        self._container.seek(0, stream=self._stream)
        self._decoder = self._container.decode(self._stream)
        self._pos = 0
        self._pending = None

    def _index_of(self, frame) -> int | None:
        if frame.pts is None:
            return None
        return int(round(float((frame.pts - self._start) * self._tb) * self.fps))

    def _next_frame(self):
        if self._pending is not None:
            f, self._pending = self._pending, None
            return f
        try:
            return next(self._decoder)
        except (StopIteration, av.error.EOFError):
            return None
        except av.error.FFmpegError:
            return None

    def grab(self) -> bool:
        f = self._next_frame()
        if f is None:
            return False
        self._pos += 1
        return True

    def read(self) -> np.ndarray | None:
        f = self._next_frame()
        if f is None:
            return None
        self._pos += 1
        img = f.to_ndarray(format="bgr24")
        rot = getattr(f, "rotation", 0) or 0
        # PyAV liefert den Winkel der Display-Matrix (gegen den Uhrzeigersinn);
        # zum Aufrichten muss das Bild um genau diesen Winkel gedreht werden.
        return _rotate(img, rot)

    def seek(self, index: int) -> None:
        if index == self._pos:
            return
        ahead = index - self._pos
        if 0 < ahead <= max(30, int(2 * self.fps)):
            while self._pos < index and self.grab():
                pass
            return
        # Zum Schlüsselbild davor springen, dann bis zum Zielbild dekodieren.
        target = self._start + int(index / self.fps / self._tb)
        try:
            self._container.seek(target, stream=self._stream, backward=True, any_frame=False)
        except av.error.FFmpegError:
            self._restart()
            while self._pos < index and self.grab():
                pass
            return
        self._decoder = self._container.decode(self._stream)
        self._pending = None
        while True:
            try:
                f = next(self._decoder)
            except (StopIteration, av.error.FFmpegError):
                self._pos = index
                return
            i = self._index_of(f)
            if i is None or i >= index:
                self._pending = f
                self._pos = index
                return

    def close(self) -> None:
        self._container.close()


def _windows_short_path(path: str) -> str | None:
    """8.3-Kurzpfad ohne Umlaute (nur Windows), für OpenCV."""
    if os.name != "nt":
        return None
    try:
        import ctypes

        buf = ctypes.create_unicode_buffer(32768)
        n = ctypes.windll.kernel32.GetShortPathNameW(path, buf, len(buf))
        return buf.value if 0 < n < len(buf) else None
    except Exception:
        return None


class _OpenCVReader:
    name = "OpenCV"

    def __init__(self, path: str):
        candidates = [path]
        if not path.isascii():
            short = _windows_short_path(path)
            if short and short != path:
                candidates.insert(0, short)
        apis = [cv2.CAP_FFMPEG, cv2.CAP_ANY]
        if sys.platform == "win32":
            apis.insert(1, cv2.CAP_MSMF)
        self._cap = None
        for p in candidates:
            for api in apis:
                cap = cv2.VideoCapture(p, api)
                if cap.isOpened():
                    ok, _ = cap.read()
                    if ok:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self._cap = cap
                        break
                cap.release()
            if self._cap is not None:
                break
        if self._cap is None:
            raise VideoError("OpenCV kann die Datei nicht öffnen.")
        fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps = fps if fps > 0 and np.isfinite(fps) else 30.0
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._pos = 0

    def grab(self) -> bool:
        ok = self._cap.grab()
        self._pos += ok
        return bool(ok)

    def read(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        if not ok:
            return None
        self._pos += 1
        return frame

    def seek(self, index: int) -> None:
        if index != self._pos:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            self._pos = index

    def close(self) -> None:
        self._cap.release()


def _open_reader(path: str, backend: str | None = None):
    if not Path(path).is_file():
        raise VideoError(f"Die Datei wurde nicht gefunden:\n{path}")
    order = []
    if backend in (None, "pyav") and av is not None:
        order.append(_PyAVReader)
    if backend in (None, "opencv"):
        order.append(_OpenCVReader)
    errors = []
    for cls in order:
        try:
            return cls(path)
        except Exception as e:  # nächstes Backend versuchen
            errors.append(f"{cls.name}: {e}")
    tips = [
        "Mögliche Ursachen und Abhilfe:",
        "• Das Video ist in einem Format/Codec gespeichert, den der Decoder nicht kennt "
        "(häufig HEVC/H.265 vom iPhone).",
    ]
    if av is None:
        tips.append("• Das Paket „av“ (PyAV) fehlt. Bitte im Terminal ausführen:\n"
                    "    pip install av\n  und Strobosoft neu starten – damit lassen sich fast alle Videos öffnen.")
    tips.append("• Die Datei ist unvollständig oder beschädigt (z. B. noch nicht fertig aus der Cloud geladen).")
    tips.append("• Notfalls das Video in H.264/MP4 umwandeln (z. B. mit HandBrake oder VLC).")
    details = "\n".join(errors) if errors else "kein Decoder verfügbar"
    raise VideoError(
        f"Video konnte nicht geöffnet werden:\n{path}\n\n" + "\n".join(tips) + f"\n\nTechnische Details:\n{details}"
    )


# -------------------------------------------------------------- VideoSource
class VideoSource:
    """Liest Einzelbilder aus einer Videodatei.

    ``max_width`` verkleinert sehr große Videos beim Lesen (z. B. 4K), damit
    Speicherbedarf und Rechenzeit im Rahmen bleiben. Alle Pixelkoordinaten
    beziehen sich dann auf die verkleinerte Auflösung.
    ``backend``: ``None`` (automatisch), ``"pyav"`` oder ``"opencv"``.
    """

    def __init__(self, path: str | Path, max_width: int | None = 1920, backend: str | None = None):
        self.path = str(path)
        self._reader = _open_reader(self.path, backend)
        self.backend = self._reader.name
        first = self._reader.read()
        if first is None:
            self._reader.close()
            raise VideoError(f"Aus dem Video ließ sich kein Bild lesen:\n{self.path}")
        h, w = first.shape[:2]  # nach evtl. Drehung (Hochkant-Videos)
        self._scale = 1.0
        if max_width and w > max_width:
            self._scale = max_width / w
            w = int(round(w * self._scale))
            h = int(round(h * self._scale))
        count = self._reader.frame_count
        if count <= 0:
            count = 1
            while self._reader.grab():
                count += 1
        self._reader.seek(0)
        self.info = VideoInfo(self.path, self._reader.fps, count, w, h)

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[1] != self.info.width or frame.shape[0] != self.info.height:
            frame = cv2.resize(frame, (self.info.width, self.info.height), interpolation=cv2.INTER_AREA)
        return frame

    def read(self, index: int) -> np.ndarray:
        """Liest das Bild mit dem Index ``index`` (0-basiert)."""
        if not 0 <= index < self.info.frame_count:
            raise IndexError(index)
        self._reader.seek(index)
        frame = self._reader.read()
        if frame is None:
            raise VideoError(f"Bild {index} konnte nicht gelesen werden")
        return self._prepare(frame)

    def iter_range(
        self, start: int, stop: int, wanted: Container[int] | None = None
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Liest die Bilder ``start`` .. ``stop - 1`` nacheinander.

        Mit ``wanted`` werden nur diese Bilder ausgegeben; die übrigen
        werden nur dekodiert, was schneller ist. Sequentielles Lesen ist
        bildgenau – Springen ist das nicht bei jedem Codec.
        """
        stop = min(stop, self.info.frame_count)
        if start >= stop:
            return
        self._reader.seek(start)
        for i in range(start, stop):
            if wanted is not None and i not in wanted:
                if not self._reader.grab():
                    break
                continue
            frame = self._reader.read()
            if frame is None:
                break
            yield i, self._prepare(frame)

    def close(self) -> None:
        self._reader.close()

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
