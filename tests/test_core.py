import math

import numpy as np
import pytest

from strobosoft.core import kinematics as kin
from strobosoft.core.background import median_background, sample_indices
from strobosoft.core.compose import compose_stroboscope
from strobosoft.core.export import csv_text, svg_document
from strobosoft.core.models import BBox, Calibration
from strobosoft.core.overlay import OverlayOptions, build_overlay, nice_floor, nice_step, render_raster
from strobosoft.core.pipeline import detect_auto, strobe_indices, track_manual
from strobosoft.core.synthetic import scene_background
from strobosoft.core.video import VideoSource


def scene_calibration(scene):
    p1, p2 = scene.ruler
    return Calibration(kin.scale_from_points(p1, p2, 1.0), scene.start_px, 0.0, True)


def test_sample_indices():
    assert sample_indices(0, 5, 10) == [0, 1, 2, 3, 4]
    idx = sample_indices(0, 100, 11)
    assert idx[0] == 0 and idx[-1] == 99 and len(idx) == 11


def test_strobe_indices():
    assert strobe_indices(2, 12, 3) == [2, 5, 8, 11]
    with pytest.raises(ValueError):
        strobe_indices(0, 10, 0)


def test_video_info(demo_video, scene):
    with VideoSource(demo_video) as src:
        assert src.info.frame_count == scene.frame_count
        assert src.info.fps == pytest.approx(scene.fps)
        assert src.read(5).shape == (scene.height, scene.width, 3)


def test_background_removes_object(demo_video, scene):
    with VideoSource(demo_video) as src:
        bg = median_background(src, 0, src.info.frame_count)
    truth = scene_background(scene).astype(int)
    # An allen Stellen, an denen der Ball war, muss der Hintergrund zu sehen sein
    # (kleine Abweichungen durch Videokompression sind erlaubt).
    for i in range(scene.frame_count):
        x, y = (int(round(v)) for v in scene.position_px(i / scene.fps))
        r = scene.radius - 2
        area = (slice(y - r, y + r), slice(x - r, x + r))
        assert np.abs(bg[area].astype(int) - truth[area]).mean() < 10


def test_auto_detection_positions(demo_video, scene):
    with VideoSource(demo_video) as src:
        bg = median_background(src, 0, src.info.frame_count)
        track = detect_auto(src, bg, 0, src.info.frame_count, 3)
    assert len(track.samples) == 10
    assert all(s.found for s in track.samples)
    for s in track.samples:
        tx, ty = scene.position_px(s.frame_index / scene.fps)
        assert math.hypot(s.position[0] - tx, s.position[1] - ty) < 1.5


def test_manual_tracking(demo_video, scene):
    x, y = scene.start_px
    r = scene.radius + 4
    with VideoSource(demo_video) as src:
        bg = median_background(src, 0, src.info.frame_count)
        track = track_manual(src, bg, BBox(int(x - r), int(y - r), 2 * r, 2 * r), 0, src.info.frame_count, 3)
    assert len(track.found_samples) == 10
    for s in track.samples:
        tx, ty = scene.position_px(s.frame_index / scene.fps)
        assert math.hypot(s.position[0] - tx, s.position[1] - ty) < 2.0


def test_velocities(demo_video, scene):
    with VideoSource(demo_video) as src:
        bg = median_background(src, 0, src.info.frame_count)
        track = detect_auto(src, bg, 0, src.info.frame_count, 3)
    pts = kin.compute_kinematics(track.samples, scene_calibration(scene), "central")
    for p in pts[1:-1]:  # Randpunkte sind einseitig geschätzt
        vx, vy = scene.velocity(p.time)
        assert p.vx == pytest.approx(vx, abs=0.08)
        assert p.vy == pytest.approx(vy, abs=0.08)
    assert pts[0].x == pytest.approx(0, abs=0.01)
    fwd = kin.compute_kinematics(track.samples, scene_calibration(scene), "forward")
    assert fwd[-1].vx is None
    # Vorwärtsdifferenz = Geschwindigkeit in der Intervallmitte
    vx, vy = scene.velocity(fwd[2].time + 0.05)
    assert fwd[2].vy == pytest.approx(vy, abs=0.08)


def test_calibration_roundtrip():
    cal = Calibration(150.0, (100.0, 200.0), 30.0, True)
    for p in [(100.0, 200.0), (250.0, 120.0), (0.0, 0.0)]:
        w = kin.pixel_to_world(cal, p)
        back = kin.world_to_pixel(cal, w)
        assert back == pytest.approx(p)
    # Punkt auf der gedrehten x-Achse hat y = 0
    a = math.radians(30)
    on_axis = (100 + 150 * math.cos(a), 200 - 150 * math.sin(a))
    assert kin.pixel_to_world(cal, on_axis) == pytest.approx((1.0, 0.0))
    assert kin.angle_from_points((100, 200), on_axis) == pytest.approx(30)


def test_y_axis_points_up():
    cal = Calibration(100.0, (0.0, 100.0), 0.0, True)
    assert kin.pixel_to_world(cal, (0.0, 0.0)) == pytest.approx((0.0, 1.0))


def test_nice_numbers():
    assert nice_step(0.34) == 0.5
    assert nice_step(3) == 5
    assert nice_floor(7.7) == 5
    assert nice_floor(0.34) == pytest.approx(0.2)


def test_compose_and_exports(demo_video, scene):
    with VideoSource(demo_video) as src:
        bg = median_background(src, 0, src.info.frame_count)
        track = detect_auto(src, bg, 0, src.info.frame_count, 3)
    img = compose_stroboscope(bg, track.samples)
    assert img.shape == bg.shape
    # An jeder Objektposition ist der (rote) Ball zu sehen
    for s in track.found_samples:
        x, y = (int(round(v)) for v in s.position)
        b, g, r = img[y + scene.radius // 2 + 2, x].astype(int)
        assert r > 150 and b < 100
    cal = scene_calibration(scene)
    pts = kin.compute_kinematics(track.samples, cal)
    opt = OverlayOptions(arrow_scale=kin.auto_arrow_scale(pts), show_times=True, show_speeds=True)
    prims = build_overlay((img.shape[1], img.shape[0]), cal, pts, opt)
    out = render_raster(img, prims)
    assert out.shape == img.shape and not np.array_equal(out, img)
    svg = svg_document(img, prims)
    assert svg.startswith("<?xml") and svg.count("marker-end") >= 9 and "x in m" in svg
    text = csv_text(pts, cal, german=True)
    lines = text.strip().splitlines()
    assert len(lines) == 11 and lines[0].startswith("Bild;t in s")
    assert "," in lines[1].split(";")[1]
    text_en = csv_text(pts, cal, german=False)
    assert text_en.splitlines()[1].count(",") == 8
