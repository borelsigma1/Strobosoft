"""Start mit ``python -m strobosoft`` – oder direkt per ▶ in VS Code."""

import sys
from pathlib import Path

if not __package__:
    # Datei wurde direkt gestartet (python strobosoft/__main__.py):
    # Projektordner in den Suchpfad aufnehmen, damit "strobosoft" gefunden wird.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strobosoft.gui.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
