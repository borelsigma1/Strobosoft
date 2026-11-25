import subprocess

import numpy as np
import pytest

from strobosoft.core.synthetic import ThrowScene, write_demo_video
from strobosoft.core.video import VideoError, VideoSource, av

BACKENDS = ["opencv"] + (["pyav"] if av is not None else [])


@pytest.fixture(scope="module")
def umlaut_video(tmp_path_factory):
    folder = tmp_path_factory.mktemp("Videos für Physik")
    scene = ThrowScene(duration=3.0, v0=(1.0, 4.2))
    return write_demo_video(folder / "Wurf äöü ß.mp4", scene), scene


@pytest.mark.parametrize("backend", BACKENDS)
def test_umlaut_path_and_random_access(umlaut_video, backend):
    path, scene = umlaut_video
    with VideoSource(path, backend=backend) as src:
        assert src.info.frame_count == scene.frame_count
        assert (src.info.width, src.info.height) == (scene.width, scene.height)
        seq = dict(src.iter_range(0, src.info.frame_count))
        assert len(seq) == scene.frame_count
        for i in [0, 5, 44, 45, 46, 89, 3, 60, 1]:
            assert np.array_equal(src.read(i), seq[i]), f"Bild {i} weicht ab"


def test_missing_file_message(tmp_path):
    with pytest.raises(VideoError, match="nicht gefunden"):
        VideoSource(tmp_path / "gibt_es_nicht.mp4")


def test_broken_file_message(tmp_path):
    bad = tmp_path / "kaputt.mp4"
    bad.write_bytes(b"das ist kein video" * 100)
    with pytest.raises(VideoError, match="Mögliche Ursachen"):
        VideoSource(bad)


@pytest.mark.skipif(av is None, reason="PyAV nicht installiert")
def test_rotated_phone_video(umlaut_video, tmp_path):
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    path, scene = umlaut_video
    rotated = tmp_path / "hochkant.mp4"
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-display_rotation", "90",
         "-i", str(path), "-c", "copy", str(rotated)],
        check=True,
    )
    frames = {}
    for backend in BACKENDS:
        with VideoSource(rotated, backend=backend) as src:
            assert (src.info.width, src.info.height) == (scene.height, scene.width)
            frames[backend] = src.read(10)
    if len(frames) == 2:
        assert np.array_equal(frames["pyav"], frames["opencv"])
