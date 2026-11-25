# Architektur

Strobosoft trennt strikt zwischen **Verarbeitungskern** (`strobosoft/core`) und
**Oberfläche** (`strobosoft/gui`). Der Kern kennt kein Qt; er arbeitet nur mit
numpy-Arrays und schlichten Datenklassen. Damit lässt er sich unabhängig testen
und später Modul für Modul nach Rust übertragen (z. B. für eine Tauri-App auf
dem iPad).

```
Video ─► VideoSource ─► median_background ─► Hintergrund
                    │
                    ├─► detect_auto  (Differenz zum Hintergrund, größter/naher Blob)
                    └─► track_manual (Rahmen + CSRT-Tracker, Maske im Rahmen)
                                 │
                                 ▼
                     Track = [ObjectSample …]  (Position, Ausschnitt, Alpha)
                                 │
           ┌─────────────────────┼──────────────────────┐
           ▼                     ▼                      ▼
 compose_stroboscope    compute_kinematics       (Fehlerkennung entfernen)
   (Bild)                (x, y, vx, vy)
           │                     │
           └──────► build_overlay ◄── Calibration, OverlayOptions
                        │  (Liste von Line / Circle / Text)
            ┌───────────┼─────────────┐
            ▼           ▼             ▼
      render_raster  svg_document   csv_text
        (PNG)          (SVG)          (CSV)
```

## Module im Kern

| Modul | Aufgabe |
|---|---|
| `models.py` | Datenklassen: `VideoInfo`, `BBox`, `ObjectSample`, `Track`, `Calibration`, `KinematicPoint` |
| `video.py` | Bilder lesen (sequentiell, bildgenau), große Videos auf max. 1920 px Breite verkleinern, Bild speichern |
| `background.py` | Median über bis zu 41 gleichmäßig verteilte Bilder |
| `detection.py` | Differenzbild, Schwellwert (Otsu oder fest), Morphologie, Blob-Auswahl, Ausschnitt mit weicher Kante |
| `tracking.py` | CSRT-Tracker (Fallback KCF), Maske innerhalb des Tracker-Rahmens |
| `pipeline.py` | wendet Erkennung/Tracking auf jedes n-te Bild an |
| `compose.py` | Alpha-Blending der Ausschnitte auf den Hintergrund |
| `kinematics.py` | Pixel ↔ Weltkoordinaten (Maßstab, Ursprung, Drehung), Differenzenquotienten, Pfeilmaßstab |
| `overlay.py` | Achsen, Punkte, Pfeile, Beschriftungen als **Grafik-Primitive**; Rasterisierung mit OpenCV |
| `export.py` | PNG, SVG (eingebettetes Bild + Vektoren), CSV |
| `synthetic.py` | Beispielvideo mit bekannter Bahn (für Tests und Demo) |

## Konventionen

- Bilder: `uint8`, BGR (OpenCV), Form `(Höhe, Breite, 3)`.
- Pixelkoordinaten: x nach rechts, y nach unten, Ursprung oben links.
- Weltkoordinaten: y nach oben, Winkel der x-Achse gegen den Uhrzeigersinn.
- Zeit eines Blitzes: `t = (Bildindex − Startbild) / fps`.
- Fortschritt: Funktionen nehmen optional `progress(erledigt, gesamt) -> bool`;
  `False` bricht ab (`InterruptedError`).

## Hinweise für eine Rust-Portierung

- Datenklassen → `struct`s mit `serde`; `ObjectSample.patch/alpha` → `ndarray`
  oder `image::RgbImage` / `GrayImage<f32>`.
- OpenCV-Aufrufe sind auf wenige Stellen begrenzt (absdiff, GaussianBlur,
  threshold/Otsu, morphologyEx, findContours, moments, Tracker, Video-Dekodierung).
  In Rust: `opencv`-Crate, oder reine Rust-Alternativen (`imageproc` für
  Morphologie/Konturen, eigener Median, Video über `ffmpeg-next` bzw. auf iOS
  AVFoundation).
- Die Grafik-Primitive aus `overlay.py` lassen sich unverändert als
  SVG/Canvas im Tauri-Frontend zeichnen; PNG-Export dann z. B. per `tiny-skia`.
- Die Tests mit dem synthetischen Video eignen sich als Referenz: gleiche
  Eingaben müssen in Rust (innerhalb kleiner Toleranzen) gleiche Werte liefern.
