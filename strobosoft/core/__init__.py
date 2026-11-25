"""Verarbeitungskern ohne GUI-Abhängigkeiten.

Alle Module arbeiten nur mit numpy-Arrays und einfachen Datenklassen aus
``models``. Bilder sind BGR-uint8-Arrays (OpenCV-Konvention), Masken sind
uint8-Arrays mit Werten 0 oder 255. Pixelkoordinaten: x nach rechts,
y nach unten, Ursprung oben links.
"""
