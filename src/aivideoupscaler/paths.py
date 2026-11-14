"""Filesystem locations used by the application.

The package works both from a source checkout and from a frozen (PyInstaller)
executable.  Every path can be overridden with an environment variable so the
tool stays scriptable in CI pipelines.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "aivideoupscaler"
ENV_HOME = "AIVU_HOME"
ENV_FFMPEG = "FFMPEG_BINARY"
ENV_FFPROBE = "FFPROBE_BINARY"

#: Repository root when running from a source checkout (``src/aivideoupscaler``).
SOURCE_ROOT = Path(__file__).resolve().parents[2]


def is_frozen() -> bool:
    """Return ``True`` when running inside a PyInstaller/Nuitka bundle."""
    return bool(getattr(sys, "frozen", False))


def app_root() -> Path:
    """Base directory of the application (source checkout or bundle folder)."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return SOURCE_ROOT


def home_dir() -> Path:
    """Writable root for models, binaries and caches."""
    override = os.environ.get(ENV_HOME)
    if override:
        return Path(override).expanduser().resolve()
    return app_root()


def models_dir() -> Path:
    """Directory holding downloaded model weights."""
    return home_dir() / "models"


def binaries_dir() -> Path:
    """Directory holding bundled helper executables (ffmpeg, ncnn-vulkan)."""
    return home_dir() / "bin"


def cache_dir() -> Path:
    """Directory for scratch files that may be safely deleted."""
    return home_dir() / ".cache"


def work_dir() -> Path:
    """Directory for temporary processing artefacts."""
    return cache_dir() / "work"


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_binary(name: str, env_var: str | None = None) -> Path | None:
    """Locate an executable taking bundled copies and environment overrides first."""
    override = os.environ.get(env_var) if env_var else None
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_file():
            return candidate.resolve()

    local = binaries_dir() / name
    if local.is_file():
        return local.resolve()

    local_exe = binaries_dir() / f"{name}.exe"
    if local_exe.is_file():
        return local_exe.resolve()

    return None
