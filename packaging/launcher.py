"""Entry point for the frozen (PyInstaller) executable.

PyInstaller executes the entry script as a top level module, which breaks the
relative imports used inside the package.  This launcher therefore uses an
absolute import and starts the command line interface.
"""

from __future__ import annotations

import multiprocessing
import sys

from aivideoupscaler.ui.cli import _force_utf8_output, main


def run() -> None:
    """Start the command line interface."""
    _force_utf8_output()
    main()


if __name__ == "__main__":
    # Required so that a frozen build can spawn worker processes.
    multiprocessing.freeze_support()
    run()
