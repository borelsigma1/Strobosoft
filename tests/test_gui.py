"""Rauchtest der Oberfläche (ohne Bildschirm, Qt-"offscreen")."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_full_workflow(app, demo_video, scene, tmp_path):
    from strobosoft.gui.main_window import VIEW_STROBO, MainWindow

    w = MainWindow()
    w.open_video(str(demo_video))
    assert w.end_spin.value() == scene.frame_count - 1
    w.compute()
    assert w.track is not None and len(w.track.found_samples) == 10
    assert w.view_combo.currentIndex() == VIEW_STROBO

    (x1, y1), (x2, y2) = scene.ruler
    w._expect("scale")
    w._on_point(x1, y1)
    w._on_point(x2, y2)
    assert w.ppm == pytest.approx(scene.pixels_per_meter)
    pts = w.kinematic_points()
    assert pts[3].vx == pytest.approx(3.5, abs=0.1)

    # Fehlerkennung entfernen
    s = w.track.samples[4]
    w._toggle_remove_mode(True)
    w._on_point(*s.position)
    assert len(w.track.found_samples) == 9
    w._toggle_remove_mode(False)

    from strobosoft.core.export import export_csv, export_png, export_svg

    export_png(tmp_path / "a.png", w.strobo, w.overlay_primitives(w.strobo))
    export_svg(tmp_path / "a.svg", w.strobo, w.overlay_primitives(w.strobo))
    export_csv(tmp_path / "a.csv", w.kinematic_points(), w.calibration())
    assert (tmp_path / "a.png").stat().st_size > 1000
    assert (tmp_path / "a.csv").read_text(encoding="utf-8-sig").count("\n") == 10
    w.close()
