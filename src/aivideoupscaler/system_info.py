"""Live hardware statistics for the terminal interface.

The information is best effort: every function degrades gracefully when the
hardware or the utilities are not available, because the user interface must
never fail because a metric cannot be read.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache

_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


@dataclass(frozen=True)
class GpuInfo:
    """A snapshot of the graphics adapter."""

    name: str = ""
    vendor: str = ""
    total_mb: int = 0
    used_mb: int = 0
    source: str = ""

    @property
    def available(self) -> bool:
        """Return ``True`` when a usable adapter was detected."""
        return bool(self.name) and self.total_mb > 0

    @property
    def free_mb(self) -> int:
        """Return the free video memory in megabytes."""
        return max(0, self.total_mb - self.used_mb)

    @property
    def used_percent(self) -> float:
        """Return the used video memory as a percentage."""
        if self.total_mb <= 0:
            return 0.0
        return self.used_mb * 100.0 / self.total_mb


@dataclass(frozen=True)
class MemoryInfo:
    """A snapshot of the system memory."""

    total_mb: int = 0
    used_mb: int = 0
    available_mb: int = 0

    @property
    def used_percent(self) -> float:
        """Return the used system memory as a percentage."""
        if self.total_mb <= 0:
            return 0.0
        return self.used_mb * 100.0 / self.total_mb


@lru_cache(maxsize=1)
def _nvidia_smi_path() -> str:
    """Return the path of ``nvidia-smi``, or an empty string."""
    return shutil.which("nvidia-smi") or ""


def _query_nvidia() -> GpuInfo:
    """Read the statistics of the first NVIDIA adapter."""
    binary = _nvidia_smi_path()
    if not binary:
        return GpuInfo()

    query = "name,memory.total,memory.used,driver_version"
    try:
        result = subprocess.run(
            [
                binary,
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            creationflags=_CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        return GpuInfo()

    if result.returncode != 0 or not result.stdout.strip():
        return GpuInfo()

    first = result.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    if len(parts) < 3:
        return GpuInfo()

    def _to_int(value: str) -> int:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0

    name = parts[0]
    # ``nounits`` makes nvidia-smi report the values in megabytes already.
    return GpuInfo(
        name=name,
        vendor="NVIDIA",
        total_mb=_to_int(parts[1]),
        used_mb=_to_int(parts[2]),
        source="nvidia-smi",
    )


def gpu_info() -> GpuInfo:
    """Return the statistics of the graphics adapter.

    NVIDIA adapters are read through ``nvidia-smi``.  When no adapter can be
    detected an empty snapshot is returned, which the interface renders as
    "no GPU detected".
    """
    info = _query_nvidia()
    if info.available:
        return info

    try:
        import torch
    except Exception:
        return GpuInfo()

    try:
        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(index)
            return GpuInfo(
                name=props.name,
                vendor="NVIDIA",
                total_mb=props.total_memory // (1024 * 1024),
                used_mb=0,
                source="torch",
            )
    except Exception:
        return GpuInfo()
    return GpuInfo()


def memory_info() -> MemoryInfo:
    """Return the system memory statistics."""
    try:
        import psutil
    except Exception:
        return MemoryInfo()

    try:
        virtual = psutil.virtual_memory()
    except Exception:
        return MemoryInfo()

    return MemoryInfo(
        total_mb=virtual.total // (1024 * 1024),
        used_mb=virtual.used // (1024 * 1024),
        available_mb=virtual.available // (1024 * 1024),
    )


def format_megabytes(value: int) -> str:
    """Format a megabyte value for the status line."""
    if value <= 0:
        return "-"
    if value >= 1024:
        return f"{value / 1024:.1f} GB"
    return f"{value} MB"


def describe_environment_short() -> str:
    """Return a one line summary of the environment."""
    gpu = gpu_info()
    if gpu.available:
        return f"GPU {gpu.name} ({format_megabytes(gpu.total_mb)})"
    return "no GPU detected (CPU mode)"
