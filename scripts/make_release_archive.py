"""Create the compressed release archive of the standalone build.

Usage::

    python scripts/make_release_archive.py dist/aivideoupscaler aivideoupscaler-windows-x64.zip
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path


def build(source: Path, target: Path) -> int:
    """Zip the ``source`` folder into ``target`` with deflate compression."""
    if not source.is_dir():
        print(f"The source folder does not exist: {source}", file=sys.stderr)
        return 1

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()

    files = sorted(p for p in source.rglob("*") if p.is_file())
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, Path(source.name) / path.relative_to(source))

    size = target.stat().st_size
    print(f"Wrote {target} ({size / 1048576:.2f} MiB, {len(files)} files)")
    return 0


def main(argv: list[str]) -> int:
    """Run the archive builder."""
    if len(argv) < 3:
        print(__doc__)
        return 2
    return build(Path(argv[1]), Path(argv[2]))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
