"""Hintergrundschätzung: Median über viele Einzelbilder.

Bewegt sich das Objekt, ist es an jedem Pixel nur in wenigen Bildern zu
sehen. Der Median über alle Bilder liefert deshalb an jedem Pixel den
Hintergrund – das Objekt "verschwindet".
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .video import VideoSource

ProgressFn = Callable[[int, int], bool]  # (erledigt, gesamt) -> weitermachen?


def sample_indices(start: int, stop: int, count: int) -> list[int]:
    """Bis zu ``count`` gleichmäßig verteilte Bildindizes aus [start, stop)."""
    n = stop - start
    if n <= 0:
        return []
    if n <= count:
        return list(range(start, stop))
    return sorted({int(round(v)) for v in np.linspace(start, stop - 1, count)})


def median_of(frames: Sequence[np.ndarray]) -> np.ndarray:
    if not frames:
        raise ValueError("Keine Bilder für den Hintergrund")
    stack = np.stack(frames, axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


def median_background(
    source: VideoSource,
    start: int,
    stop: int,
    samples: int = 41,
    progress: ProgressFn | None = None,
) -> np.ndarray:
    indices = sample_indices(start, stop, samples)
    wanted = set(indices)
    frames: list[np.ndarray] = []
    for _, frame in source.iter_range(start, stop, wanted):
        frames.append(frame)
        if progress and not progress(len(frames), len(indices)):
            raise InterruptedError
    return median_of(frames)
