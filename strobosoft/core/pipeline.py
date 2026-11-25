"""Ablauf: Bilder auswählen -> Objekt finden -> Proben sammeln.

Hier werden die Bausteine aus ``detection`` und ``tracking`` über einen
Zeitbereich des Videos angewendet.
"""

from __future__ import annotations

import numpy as np

from .background import ProgressFn
from .detection import DetectionParams, detect_in_frame, missing_sample
from .models import BBox, ObjectSample, Track
from .tracking import BoxTracker, sample_from_box
from .video import VideoSource


def strobe_indices(start: int, stop: int, step: int) -> list[int]:
    """Bildindizes der Stroboskop-"Blitze": start, start+n, start+2n, ..."""
    if step < 1:
        raise ValueError("Schrittweite muss mindestens 1 sein")
    return list(range(start, stop, step))


def detect_auto(
    source: VideoSource,
    background: np.ndarray,
    start: int,
    stop: int,
    step: int,
    params: DetectionParams = DetectionParams(),
    roi: BBox | None = None,
    progress: ProgressFn | None = None,
) -> Track:
    """Automatische Erkennung in jedem n-ten Bild."""
    fps = source.info.fps
    wanted = strobe_indices(start, stop, step)
    if not wanted:
        return Track()
    track = Track()
    previous: tuple[float, float] | None = None
    for k, (idx, frame) in enumerate(source.iter_range(start, stop, set(wanted))):
        sample = detect_in_frame(
            frame, background, idx, (idx - start) / fps, params, previous, roi
        )
        track.samples.append(sample)
        if sample.found:
            previous = sample.position
        if progress and not progress(k + 1, len(wanted)):
            raise InterruptedError
    return track


def track_manual(
    source: VideoSource,
    background: np.ndarray | None,
    init_box: BBox,
    start: int,
    stop: int,
    step: int,
    params: DetectionParams = DetectionParams(),
    progress: ProgressFn | None = None,
) -> Track:
    """Verfolgung eines im Startbild markierten Objekts über alle Bilder.

    Der Tracker sieht jedes Bild (das macht ihn robust); gespeichert werden
    nur die Stroboskop-Bilder.
    """
    fps = source.info.fps
    wanted = set(strobe_indices(start, stop, step))
    track = Track()
    tracker: BoxTracker | None = None
    lost = False
    total = max(1, min(stop, source.info.frame_count) - start)
    for i, frame in source.iter_range(start, stop):
        if tracker is None:
            tracker = BoxTracker(frame, init_box)
            box: BBox | None = init_box
        elif not lost:
            box = tracker.update(frame)
            lost = box is None
        else:
            box = None
        if i in wanted:
            t = (i - start) / fps
            if box is None:
                track.samples.append(missing_sample(i, t))
            else:
                track.samples.append(sample_from_box(frame, background, box, i, t, params))
        if progress and not progress(i - start + 1, total):
            raise InterruptedError
    return track


def remove_sample(track: Track, frame_index: int) -> None:
    """Probe manuell ausblenden (z. B. Fehlerkennung)."""
    for s in track.samples:
        if s.frame_index == frame_index:
            s.found = False


def nearest_sample(track: Track, point: tuple[float, float], radius: float) -> ObjectSample | None:
    best, best_d = None, radius
    for s in track.found_samples:
        d = float(np.hypot(s.position[0] - point[0], s.position[1] - point[1]))
        if d <= best_d:
            best, best_d = s, d
    return best
