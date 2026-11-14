"""Tests for the hardware statistics helpers."""

from __future__ import annotations

from aivideoupscaler.system_info import (
    MemoryInfo,
    format_megabytes,
    gpu_info,
    memory_info,
)


def test_the_gpu_snapshot_is_always_usable() -> None:
    """Reading the GPU never fails, even without an adapter."""
    info = gpu_info()
    assert info.total_mb >= 0
    assert info.used_mb >= 0
    assert info.free_mb >= 0
    assert 0.0 <= info.used_percent <= 100.0


def test_an_unknown_gpu_is_reported_as_unavailable() -> None:
    """A machine without a GPU reports no adapter."""
    info = gpu_info()
    if not info.available:
        assert info.name == ""
        assert info.total_mb == 0


def test_the_memory_snapshot_is_always_usable() -> None:
    """Reading the system memory never fails."""
    info = memory_info()
    assert info.total_mb >= 0
    assert info.used_mb >= 0
    assert 0.0 <= info.used_percent <= 100.0


def test_megabytes_are_formatted_for_humans() -> None:
    """Small values stay in megabytes, large ones switch to gigabytes."""
    assert format_megabytes(0) == "-"
    assert format_megabytes(512) == "512 MB"
    assert format_megabytes(2048) == "2.0 GB"


def test_the_memory_dataclass_defaults_are_empty() -> None:
    """An empty snapshot is still a valid object."""
    info = MemoryInfo()
    assert info.used_percent == 0.0
