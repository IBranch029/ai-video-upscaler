"""Tests for the filesystem layout helpers."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from aivideoupscaler import paths


def test_home_dir_honours_the_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``AIVU_HOME`` overrides the application root."""
    monkeypatch.setenv(paths.ENV_HOME, str(tmp_path))
    assert paths.home_dir() == tmp_path.resolve()


def test_home_dir_defaults_to_the_app_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the environment variable the application root is used."""
    monkeypatch.delenv(paths.ENV_HOME, raising=False)
    assert paths.home_dir() == paths.app_root()


def test_models_and_binaries_live_under_the_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The models and binaries folders are derived from the home directory."""
    monkeypatch.setenv(paths.ENV_HOME, str(tmp_path))
    assert paths.models_dir() == tmp_path.resolve() / "models"
    assert paths.binaries_dir() == tmp_path.resolve() / "bin"


def test_resolve_binary_prefers_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An explicit environment variable wins over the bundled copies."""
    target = tmp_path / "ffmpeg_custom"
    target.write_bytes(b"")
    monkeypatch.setenv(paths.ENV_FFMPEG, str(target))
    assert paths.resolve_binary("ffmpeg", paths.ENV_FFMPEG) == target.resolve()


def test_resolve_binary_finds_a_bundled_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A bundled executable is found in the application folders."""
    monkeypatch.delenv(paths.ENV_FFMPEG, raising=False)
    monkeypatch.setattr(paths, "app_root", lambda: tmp_path)
    bundle = tmp_path / "_internal" / "bin"
    bundle.mkdir(parents=True)
    executable = bundle / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    executable.write_bytes(b"")

    assert paths.resolve_binary("ffmpeg", paths.ENV_FFMPEG) == executable.resolve()


def test_resolve_binary_returns_none_when_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing executable is reported as ``None``."""
    monkeypatch.delenv(paths.ENV_FFMPEG, raising=False)
    monkeypatch.setattr(paths, "app_root", lambda: tmp_path)
    assert paths.resolve_binary("definitely-not-here", paths.ENV_FFMPEG) is None


def test_ensure_dir_creates_the_directory(tmp_path: Path) -> None:
    """``ensure_dir`` creates the folder and returns it."""
    target = tmp_path / "a" / "b"
    assert paths.ensure_dir(target) == target
    assert target.is_dir()
