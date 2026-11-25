"""Programmstart: ``python -m strobosoft [video]``."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from strobosoft.gui.main_window import MainWindow


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    app = QApplication(argv)
    app.setApplicationName("Strobosoft")
    win = MainWindow()
    win.show()
    if len(argv) > 1:
        win.open_video(argv[1])
    return app.exec()
