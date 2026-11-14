"""Allow ``python -m aivideoupscaler`` to run the command line interface."""

from __future__ import annotations

from .ui.cli import _force_utf8_output, main

if __name__ == "__main__":
    _force_utf8_output()
    main()
