"""Hauptfenster der Stroboskop-Software."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from strobosoft.core import kinematics as kin
from strobosoft.core.background import median_background
from strobosoft.core.compose import compose_stroboscope
from strobosoft.core.detection import DetectionParams
from strobosoft.core.export import export_csv, export_png, export_svg
from strobosoft.core.models import BBox, Calibration, Track
from strobosoft.core.overlay import OverlayOptions, build_overlay, render_raster
from strobosoft.core.pipeline import detect_auto, nearest_sample, remove_sample, track_manual
from strobosoft.core.synthetic import write_demo_video
from strobosoft.core.video import VideoError, VideoSource

VIDEO_FILTER = "Videos (*.mp4 *.mov *.m4v *.avi *.mkv *.wmv *.mts *.webm);;Alle Dateien (*)"

VIEW_FRAME, VIEW_BACKGROUND, VIEW_STROBO = 0, 1, 2


def fmt(v: float, decimals: int = 2) -> str:
    return f"{v:.{decimals}f}".replace(".", ",")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Strobosoft – Stroboskopbilder aus Videos")
        self.resize(1400, 860)
        self.settings = QSettings("Strobosoft", "Strobosoft")

        self.source: VideoSource | None = None
        self.current_frame: np.ndarray | None = None
        self.background: np.ndarray | None = None
        self._bg_key: tuple | None = None
        self.track: Track | None = None
        self.strobo: np.ndarray | None = None
        self.manual_box: BBox | None = None

        # Kalibrierung
        self.ppm: float | None = None
        self.scale_points: list[tuple[float, float]] = []
        self.origin: tuple[float, float] | None = None
        self.angle_deg: float = 0.0
        self._pending = ""  # welche Eingabe per Klick gerade erwartet wird

        from strobosoft.gui.image_view import ImageView

        self.view = ImageView()
        self.view.pointClicked.connect(self._on_point)
        self.view.rectDrawn.connect(self._on_rect)
        self.setCentralWidget(self.view)
        self._build_panel()
        self._build_menu()
        self.statusBar().showMessage("Bitte ein Video öffnen (Datei → Video öffnen …).")
        self._update_enabled()

    # ------------------------------------------------------------------ UI
    def _build_menu(self):
        m = self.menuBar().addMenu("&Datei")
        a = QAction("Video öffnen …", self, shortcut=QKeySequence.StandardKey.Open)
        a.triggered.connect(self.open_video_dialog)
        m.addAction(a)
        a = QAction("Beispielvideo erzeugen (schiefer Wurf)", self)
        a.triggered.connect(self.create_demo)
        m.addAction(a)
        m.addSeparator()
        for text, fn in (("Als PNG exportieren …", self.export_png),
                         ("Als SVG exportieren …", self.export_svg),
                         ("Messwerte als CSV exportieren …", self.export_csv)):
            a = QAction(text, self)
            a.triggered.connect(fn)
            m.addAction(a)
        m.addSeparator()
        a = QAction("Beenden", self, shortcut=QKeySequence.StandardKey.Quit)
        a.triggered.connect(self.close)
        m.addAction(a)
        v = self.menuBar().addMenu("&Ansicht")
        a = QAction("Ganzes Bild anzeigen", self, shortcut="F")
        a.triggered.connect(self.view.fit)
        v.addAction(a)
        h = self.menuBar().addMenu("&Hilfe")
        a = QAction("Kurzanleitung", self, shortcut=QKeySequence.StandardKey.HelpContents)
        a.triggered.connect(self.show_help)
        h.addAction(a)

    def _build_panel(self):
        panel = QWidget()
        lay = QVBoxLayout(panel)

        # 1 Video ---------------------------------------------------------
        g = QGroupBox("1  Video und Zeitbereich")
        f = QFormLayout(g)
        b = QPushButton("Video öffnen …")
        b.clicked.connect(self.open_video_dialog)
        f.addRow(b)
        self.info_label = QLabel("–")
        self.info_label.setWordWrap(True)
        f.addRow(self.info_label)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.valueChanged.connect(self._on_frame_changed)
        self.frame_label = QLabel("Bild 0")
        f.addRow(self.frame_label, self.frame_slider)
        row = QHBoxLayout()
        self.start_spin = QSpinBox()
        self.end_spin = QSpinBox()
        bs = QPushButton("← aktuelles")
        bs.setToolTip("Startbild = aktuell angezeigtes Bild")
        bs.clicked.connect(lambda: self.start_spin.setValue(self.frame_slider.value()))
        be = QPushButton("← aktuelles")
        be.setToolTip("Endbild = aktuell angezeigtes Bild")
        be.clicked.connect(lambda: self.end_spin.setValue(self.frame_slider.value()))
        row.addWidget(self.start_spin)
        row.addWidget(bs)
        f.addRow("Startbild", row)
        row = QHBoxLayout()
        row.addWidget(self.end_spin)
        row.addWidget(be)
        f.addRow("Endbild", row)
        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, 1000)
        self.step_spin.setValue(3)
        self.step_spin.setSuffix(". Bild")
        self.step_spin.setPrefix("jedes ")
        f.addRow("Blitz", self.step_spin)
        self.dt_label = QLabel("Δt = –")
        f.addRow(self.dt_label)
        for w in (self.start_spin, self.end_spin, self.step_spin):
            w.valueChanged.connect(self._update_dt)
        lay.addWidget(g)

        # 2 Objekt --------------------------------------------------------
        g = QGroupBox("2  Objekt finden")
        f = QFormLayout(g)
        self.mode_auto = QRadioButton("automatisch (bewegtes Objekt)")
        self.mode_manual = QRadioButton("Objekt markieren und verfolgen")
        self.mode_auto.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.mode_auto)
        grp.addButton(self.mode_manual)
        f.addRow(self.mode_auto)
        f.addRow(self.mode_manual)
        self.box_button = QPushButton("Rahmen um Objekt im Startbild ziehen")
        self.box_button.clicked.connect(self.start_box_selection)
        self.box_label = QLabel("")
        f.addRow(self.box_button)
        f.addRow(self.box_label)
        self.thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self.thresh_slider.setRange(0, 120)
        self.thresh_slider.setValue(0)
        self.thresh_label = QLabel("auto")
        self.thresh_slider.valueChanged.connect(
            lambda v: self.thresh_label.setText("auto" if v == 0 else str(v)))
        row = QHBoxLayout()
        row.addWidget(self.thresh_slider)
        row.addWidget(self.thresh_label)
        f.addRow("Empfindlichkeit", row)
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(1, 100000)
        self.min_area_spin.setValue(40)
        self.min_area_spin.setSuffix(" px²")
        f.addRow("Mindestgröße", self.min_area_spin)
        self.compute_button = QPushButton("Stroboskopbild berechnen")
        self.compute_button.setStyleSheet("font-weight: bold; padding: 6px;")
        self.compute_button.clicked.connect(self.compute)
        f.addRow(self.compute_button)
        self.remove_button = QPushButton("Fehlerkennung entfernen (anklicken)")
        self.remove_button.setCheckable(True)
        self.remove_button.toggled.connect(self._toggle_remove_mode)
        f.addRow(self.remove_button)
        self.view_combo = QComboBox()
        self.view_combo.addItems(["Videobild", "Hintergrund", "Stroboskopbild"])
        self.view_combo.currentIndexChanged.connect(self.refresh)
        f.addRow("Anzeige", self.view_combo)
        lay.addWidget(g)

        # 3 Kalibrierung --------------------------------------------------
        g = QGroupBox("3  Maßstab und Koordinatensystem")
        f = QFormLayout(g)
        self.length_spin = QDoubleSpinBox()
        self.length_spin.setDecimals(3)
        self.length_spin.setRange(0.001, 1000)
        self.length_spin.setValue(1.0)
        self.length_spin.setSuffix(" m")
        self.length_spin.valueChanged.connect(self._update_scale)
        f.addRow("Bekannte Länge", self.length_spin)
        b = QPushButton("Maßstab: Anfang und Ende anklicken")
        b.clicked.connect(lambda: self._expect("scale"))
        f.addRow(b)
        b = QPushButton("Ursprung anklicken")
        b.clicked.connect(lambda: self._expect("origin"))
        f.addRow(b)
        b = QPushButton("Richtung der x-Achse anklicken")
        b.clicked.connect(lambda: self._expect("axis"))
        f.addRow(b)
        b = QPushButton("x-Achse waagrecht")
        b.clicked.connect(self._reset_axis)
        f.addRow(b)
        self.cal_label = QLabel("Kein Maßstab – Angaben in Pixeln.")
        self.cal_label.setWordWrap(True)
        f.addRow(self.cal_label)
        lay.addWidget(g)

        # 4 Darstellung ---------------------------------------------------
        g = QGroupBox("4  Darstellung")
        f = QFormLayout(g)
        self.cb_points = QCheckBox("Schwerpunkte")
        self.cb_vectors = QCheckBox("Geschwindigkeitspfeile")
        self.cb_axes = QCheckBox("Koordinatenachsen")
        self.cb_legend = QCheckBox("Referenzpfeil")
        self.cb_times = QCheckBox("Zeiten")
        self.cb_speeds = QCheckBox("Beträge |v|")
        for cb, on in ((self.cb_points, True), (self.cb_vectors, True), (self.cb_axes, True),
                       (self.cb_legend, True), (self.cb_times, False), (self.cb_speeds, False)):
            cb.setChecked(on)
            cb.toggled.connect(self.refresh)
        grid = QHBoxLayout()
        col1, col2 = QVBoxLayout(), QVBoxLayout()
        for cb in (self.cb_points, self.cb_vectors, self.cb_axes):
            col1.addWidget(cb)
        for cb in (self.cb_legend, self.cb_times, self.cb_speeds):
            col2.addWidget(cb)
        grid.addLayout(col1)
        grid.addLayout(col2)
        f.addRow(grid)
        self.method_combo = QComboBox()
        self.method_combo.addItems(["zentral: (r₊ − r₋) / 2Δt", "vorwärts: (r₊ − r) / Δt"])
        self.method_combo.currentIndexChanged.connect(self.refresh)
        f.addRow("Geschwindigkeit", self.method_combo)
        self.arrow_slider = QSlider(Qt.Orientation.Horizontal)
        self.arrow_slider.setRange(10, 400)
        self.arrow_slider.setValue(100)
        self.arrow_slider.valueChanged.connect(self.refresh)
        f.addRow("Pfeillänge", self.arrow_slider)
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(20, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.valueChanged.connect(self._recompose)
        f.addRow("Deckkraft Objekt", self.opacity_slider)
        lay.addWidget(g)

        # 5 Export --------------------------------------------------------
        g = QGroupBox("5  Export")
        f = QFormLayout(g)
        row = QHBoxLayout()
        for text, fn in (("PNG", self.export_png), ("SVG", self.export_svg), ("CSV", self.export_csv)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        f.addRow(row)
        self.cb_german = QCheckBox("CSV für deutsches Excel (; und Komma)")
        self.cb_german.setChecked(True)
        f.addRow(self.cb_german)
        lay.addWidget(g)
        lay.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(360)
        dock = QDockWidget("Einstellungen", self)
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        dock.setWidget(scroll)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.mode_manual.toggled.connect(self._update_enabled)

    def _update_enabled(self):
        has_video = self.source is not None
        for w in (self.compute_button, self.frame_slider, self.start_spin, self.end_spin, self.step_spin):
            w.setEnabled(has_video)
        self.box_button.setEnabled(has_video and self.mode_manual.isChecked())
        self.remove_button.setEnabled(self.track is not None)

    # --------------------------------------------------------------- Video
    def open_video_dialog(self):
        start_dir = self.settings.value("last_dir", str(Path.home()))
        path, _ = QFileDialog.getOpenFileName(self, "Video öffnen", start_dir, VIDEO_FILTER)
        if path:
            self.settings.setValue("last_dir", str(Path(path).parent))
            self.open_video(path)

    def create_demo(self):
        start_dir = self.settings.value("last_dir", str(Path.home()))
        path, _ = QFileDialog.getSaveFileName(
            self, "Beispielvideo speichern", str(Path(start_dir) / "schiefer_wurf.mp4"), "Video (*.mp4)")
        if not path:
            return
        real = write_demo_video(path)
        self.open_video(str(real))
        QMessageBox.information(
            self, "Beispielvideo",
            "Beispielvideo erzeugt (schiefer Wurf, v₀ = (3,5 | 4,2) m/s, g = 9,81 m/s²).\n\n"
            "Der schwarze Balken unten links ist genau 1 m lang – damit lässt sich der Maßstab setzen.")

    def open_video(self, path: str):
        try:
            src = VideoSource(path)
        except VideoError as e:
            QMessageBox.critical(self, "Video öffnen", str(e))
            return
        except Exception as e:  # unerwartet, aber nicht abstürzen
            QMessageBox.critical(self, "Video öffnen", f"Video konnte nicht geöffnet werden:\n{path}\n\n{e}")
            return
        if self.source:
            self.source.close()
        self.source = src
        info = src.info
        self.background = None
        self._bg_key = None
        self.track = None
        self.strobo = None
        self.manual_box = None
        self.box_label.setText("")
        last = max(0, info.frame_count - 1)
        self.frame_slider.setRange(0, last)
        for sp in (self.start_spin, self.end_spin):
            sp.setRange(0, last)
        self.start_spin.setValue(0)
        self.end_spin.setValue(last)
        self.info_label.setText(
            f"{Path(path).name}\n{info.width} × {info.height} px, {fmt(info.fps)} Bilder/s, "
            f"{info.frame_count} Bilder ({fmt(info.duration)} s)\nDecoder: {src.backend}")
        self.view_combo.setCurrentIndex(VIEW_FRAME)
        self.frame_slider.setValue(0)
        self._on_frame_changed(0)
        self._update_dt()
        self._update_enabled()
        self.statusBar().showMessage("Zeitbereich und Schrittweite wählen, dann „Stroboskopbild berechnen“.")

    def _on_frame_changed(self, idx: int):
        if not self.source:
            return
        try:
            self.current_frame = self.source.read(idx)
        except (VideoError, IndexError):
            return
        t = idx / self.source.info.fps
        self.frame_label.setText(f"Bild {idx} ({fmt(t)} s)")
        if self.view_combo.currentIndex() != VIEW_FRAME:
            self.view_combo.setCurrentIndex(VIEW_FRAME)
        else:
            self.refresh()

    def _update_dt(self):
        if not self.source:
            return
        fps = self.source.info.fps
        n = self.step_spin.value()
        count = len(range(self.start_spin.value(), self.end_spin.value() + 1, n))
        self.dt_label.setText(f"Δt = {n} / {fmt(fps)} s⁻¹ = {fmt(n / fps, 4)} s  →  {count} Blitze")

    # ------------------------------------------------------------ Rechnen
    def _progress(self, title: str):
        dlg = QProgressDialog(title, "Abbrechen", 0, 100, self)
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(300)

        def cb(done: int, total: int) -> bool:
            dlg.setValue(int(100 * done / max(1, total)))
            QApplication.processEvents()
            return not dlg.wasCanceled()

        return dlg, cb

    def _ensure_background(self, start: int, stop: int) -> bool:
        key = (start, stop)
        if self.background is not None and self._bg_key == key:
            return True
        dlg, cb = self._progress("Hintergrund wird berechnet …")
        try:
            self.background = median_background(self.source, start, stop, progress=cb)
            self._bg_key = key
            return True
        except InterruptedError:
            return False
        finally:
            dlg.close()

    def _params(self) -> DetectionParams:
        t = self.thresh_slider.value()
        return DetectionParams(threshold=None if t == 0 else t, min_area=self.min_area_spin.value())

    def compute(self):
        if not self.source:
            return
        start, end, step = self.start_spin.value(), self.end_spin.value(), self.step_spin.value()
        if end <= start:
            QMessageBox.warning(self, "Zeitbereich", "Das Endbild muss nach dem Startbild liegen.")
            return
        stop = end + 1
        if self.mode_manual.isChecked() and self.manual_box is None:
            QMessageBox.information(self, "Objekt markieren",
                                    "Bitte zuerst einen Rahmen um das Objekt im Startbild ziehen.")
            return
        if not self._ensure_background(start, stop):
            return
        dlg, cb = self._progress("Objekt wird gesucht …")
        try:
            if self.mode_manual.isChecked():
                self.track = track_manual(self.source, self.background, self.manual_box,
                                          start, stop, step, self._params(), cb)
            else:
                self.track = detect_auto(self.source, self.background, start, stop, step,
                                         self._params(), progress=cb)
        except InterruptedError:
            return
        except RuntimeError as e:
            QMessageBox.critical(self, "Fehler", str(e))
            return
        finally:
            dlg.close()
        found = len(self.track.found_samples)
        total = len(self.track.samples)
        self._recompose()
        self.view_combo.setCurrentIndex(VIEW_STROBO)
        self._update_enabled()
        msg = f"Objekt in {found} von {total} Bildern gefunden."
        if found < total:
            msg += " Tipp: Empfindlichkeit oder Mindestgröße anpassen oder Objekt markieren."
        self.statusBar().showMessage(msg)

    def _recompose(self):
        if self.track is None or self.background is None:
            return
        self.strobo = compose_stroboscope(self.background, self.track.samples,
                                          self.opacity_slider.value() / 100)
        self.refresh()

    # -------------------------------------------------------- Kalibrierung
    def calibration(self) -> Calibration:
        origin = self.origin
        if origin is None:
            found = self.track.found_samples if self.track else []
            if found:
                origin = found[0].position
            elif self.source:
                origin = (0.0, float(self.source.info.height))
            else:
                origin = (0.0, 0.0)
        if self.ppm:
            return Calibration(self.ppm, origin, self.angle_deg, True)
        return Calibration(1.0, origin, self.angle_deg, False)

    def _expect(self, what: str):
        if not self.source:
            return
        self.remove_button.setChecked(False)
        self._pending = what
        if what == "scale":
            self.scale_points = []
            msg = "Maßstab: Anfang der bekannten Strecke anklicken."
        elif what == "origin":
            msg = "Ursprung des Koordinatensystems anklicken."
        else:
            msg = "Einen Punkt auf der positiven x-Achse anklicken."
        self.view.set_mode("point")
        self.statusBar().showMessage(msg)

    def _on_point(self, x: float, y: float):
        if self._pending == "scale":
            self.scale_points.append((x, y))
            if len(self.scale_points) == 1:
                self.statusBar().showMessage("Maßstab: Ende der bekannten Strecke anklicken.")
                self.refresh()
                return
            self._update_scale()
            if self.ppm:
                self.statusBar().showMessage(f"Maßstab gesetzt: {fmt(self.ppm, 1)} px/m.")
        elif self._pending == "origin":
            self.origin = (x, y)
            self.statusBar().showMessage("Ursprung gesetzt.")
        elif self._pending == "axis":
            o = self.calibration().origin
            try:
                self.angle_deg = kin.angle_from_points(o, (x, y))
            except ValueError:
                return
            self.statusBar().showMessage(f"x-Achse um {fmt(self.angle_deg, 1)}° gedreht.")
        elif self._pending == "remove" and self.track is not None:
            s = nearest_sample(self.track, (x, y), radius=40)
            if s is not None:
                remove_sample(self.track, s.frame_index)
                self.statusBar().showMessage(f"Bild {s.frame_index} entfernt.")
                self._recompose()
            return
        self._pending = ""
        self.view.set_mode("none")
        self._update_cal_label()
        self.refresh()

    def _update_scale(self):
        if len(self.scale_points) == 2:
            try:
                self.ppm = kin.scale_from_points(*self.scale_points, self.length_spin.value())
            except ValueError as e:
                self.statusBar().showMessage(str(e))
                return
            self._update_cal_label()
            self.refresh()

    def _reset_axis(self):
        self.angle_deg = 0.0
        self._update_cal_label()
        self.refresh()

    def _update_cal_label(self):
        if not self.ppm:
            self.cal_label.setText("Kein Maßstab – Angaben in Pixeln.")
            return
        mm = 1000 / self.ppm
        txt = f"Maßstab: {fmt(self.ppm, 1)} px/m (1 px ≙ {fmt(mm, 2)} mm)"
        txt += f"\nx-Achse: {fmt(self.angle_deg, 1)}°"
        txt += "\nUrsprung: " + ("gesetzt" if self.origin else "erster Objektpunkt")
        self.cal_label.setText(txt)

    def _toggle_remove_mode(self, on: bool):
        if on:
            self.view_combo.setCurrentIndex(VIEW_STROBO)
            self._pending = "remove"
            self.view.set_mode("point")
            self.statusBar().showMessage("Fehlerhaft erkannte Objekte anklicken. Erneut auf den Knopf klicken zum Beenden.")
        else:
            if self._pending == "remove":
                self._pending = ""
            self.view.set_mode("none")

    # ---------------------------------------------------- Rahmen (manuell)
    def start_box_selection(self):
        if not self.source:
            return
        self.frame_slider.setValue(self.start_spin.value())
        self._on_frame_changed(self.start_spin.value())
        self.view_combo.setCurrentIndex(VIEW_FRAME)
        self.view.set_mode("rect")
        self.statusBar().showMessage("Rahmen eng um das Objekt ziehen.")

    def _on_rect(self, x: int, y: int, w: int, h: int):
        self.manual_box = BBox(x, y, w, h)
        self.box_label.setText(f"Rahmen: {w} × {h} px bei ({x}, {y})")
        self.view.set_mode("none")
        self.statusBar().showMessage("Rahmen gesetzt. Jetzt „Stroboskopbild berechnen“.")
        self.refresh()

    # ----------------------------------------------------------- Anzeige
    def overlay_options(self, points) -> tuple[OverlayOptions, Calibration]:
        cal = self.calibration()
        base = kin.auto_arrow_scale(points)
        return OverlayOptions(
            show_points=self.cb_points.isChecked(),
            show_vectors=self.cb_vectors.isChecked(),
            show_axes=self.cb_axes.isChecked(),
            show_legend=self.cb_legend.isChecked(),
            show_times=self.cb_times.isChecked(),
            show_speeds=self.cb_speeds.isChecked(),
            arrow_scale=base * self.arrow_slider.value() / 100,
            decimal_comma=True,
        ), cal

    def kinematic_points(self):
        if self.track is None:
            return []
        method = "central" if self.method_combo.currentIndex() == 0 else "forward"
        return kin.compute_kinematics(self.track.samples, self.calibration(), method)

    def overlay_primitives(self, image: np.ndarray):
        points = self.kinematic_points()
        opt, cal = self.overlay_options(points)
        return build_overlay((image.shape[1], image.shape[0]), cal, points, opt)

    def refresh(self, *_):
        idx = self.view_combo.currentIndex()
        if idx == VIEW_STROBO and self.strobo is not None:
            img = render_raster(self.strobo, self.overlay_primitives(self.strobo))
        elif idx == VIEW_BACKGROUND and self.background is not None:
            img = self.background.copy()
        else:
            img = None if self.current_frame is None else self.current_frame.copy()
            if img is not None and self.manual_box is not None and self.mode_manual.isChecked():
                b = self.manual_box
                cv2.rectangle(img, (b.x, b.y), (b.x + b.w, b.y + b.h), (255, 200, 0), 2)
        if img is not None:
            self._draw_calibration(img)
        self.view.set_image(img)

    def _draw_calibration(self, img: np.ndarray):
        """Maßstabsstrecke und Ursprung als Hilfe einzeichnen."""
        col = (0, 140, 255)
        for p in self.scale_points:
            cv2.drawMarker(img, (int(p[0]), int(p[1])), col, cv2.MARKER_CROSS, 18, 2)
        if len(self.scale_points) == 2:
            p1, p2 = self.scale_points
            cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), col, 2, cv2.LINE_AA)
        if self.origin and self.view_combo.currentIndex() != VIEW_STROBO:
            o = (int(self.origin[0]), int(self.origin[1]))
            a = math.radians(self.angle_deg)
            cv2.arrowedLine(img, o, (int(o[0] + 60 * math.cos(a)), int(o[1] - 60 * math.sin(a))), (230, 90, 30), 2, cv2.LINE_AA)
            cv2.arrowedLine(img, o, (int(o[0] - 60 * math.sin(a)), int(o[1] - 60 * math.cos(a))), (230, 90, 30), 2, cv2.LINE_AA)

    # ------------------------------------------------------------ Export
    def _export_target(self, title: str, suffix: str, filt: str) -> str | None:
        if self.strobo is None:
            QMessageBox.information(self, "Export", "Bitte zuerst ein Stroboskopbild berechnen.")
            return None
        start_dir = self.settings.value("export_dir", self.settings.value("last_dir", str(Path.home())))
        name = Path(self.source.path).stem + "_stroboskop" + suffix if self.source else "stroboskop" + suffix
        path, _ = QFileDialog.getSaveFileName(self, title, str(Path(start_dir) / name), filt)
        if path:
            if not path.lower().endswith(suffix):
                path += suffix
            self.settings.setValue("export_dir", str(Path(path).parent))
        return path or None

    def export_png(self):
        path = self._export_target("Als PNG speichern", ".png", "PNG-Bild (*.png)")
        if path:
            export_png(path, self.strobo, self.overlay_primitives(self.strobo))
            self.statusBar().showMessage(f"Gespeichert: {path}")

    def export_svg(self):
        path = self._export_target("Als SVG speichern", ".svg", "SVG-Vektorgrafik (*.svg)")
        if path:
            export_svg(path, self.strobo, self.overlay_primitives(self.strobo))
            self.statusBar().showMessage(f"Gespeichert: {path}")

    def export_csv(self):
        path = self._export_target("Messwerte speichern", ".csv", "CSV-Tabelle (*.csv)")
        if path:
            export_csv(path, self.kinematic_points(), self.calibration(), self.cb_german.isChecked())
            self.statusBar().showMessage(f"Gespeichert: {path}")

    # ------------------------------------------------------------- Hilfe
    def show_help(self):
        QMessageBox.information(self, "Kurzanleitung", HELP_TEXT)

    def closeEvent(self, event):
        if self.source:
            self.source.close()
        super().closeEvent(event)


HELP_TEXT = """\
1. Video öffnen. Am besten mit Stativ filmen (ruhiger Hintergrund).
2. Mit dem Schieberegler Start- und Endbild suchen und übernehmen.
   „jedes n-te Bild“ legt den Abstand der Blitze fest: Δt = n / fps.
3. „Stroboskopbild berechnen“. Der Hintergrund wird aus dem Median
   vieler Bilder bestimmt; das bewegte Objekt wird automatisch gefunden.
   Klappt das nicht (mehrere bewegte Dinge), „Objekt markieren und
   verfolgen“ wählen und einen Rahmen um das Objekt ziehen.
4. Für Geschwindigkeiten in m/s: bekannte Länge eintragen und
   Anfang und Ende dieser Strecke im Bild anklicken. Optional Ursprung
   und Richtung der x-Achse setzen (sonst Ursprung = erster Objektpunkt).
5. Export als PNG (Bild), SVG (Pfeile bearbeitbar) oder CSV (Messwerte).

Mausrad: zoomen · Ziehen: verschieben · F: ganzes Bild
"""
