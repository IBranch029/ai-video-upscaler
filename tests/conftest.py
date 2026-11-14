"""Shared pytest fixtures."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _ffmpeg() -> str | None:
    """Return the ffmpeg executable available in this environment."""
    from aivideoupscaler.ffmpeg_wrapper import find_ffmpeg

    return find_ffmpeg()


@pytest.fixture(scope="session")
def ffmpeg_bin() -> str:
    """Skip the test when FFmpeg is not installed."""
    path = _ffmpeg()
    if not path:
        pytest.skip("FFmpeg is not available in this environment")
    return path


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory: pytest.TempPathFactory, ffmpeg_bin: str) -> Path:
    """Create a small synthetic video with an audio track."""
    target = tmp_path_factory.mktemp("media") / "sample.mp4"
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=160x90:rate=12:duration=1",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=1",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(target),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f"Could not create the sample video: {result.stderr}")
    return target


@pytest.fixture(scope="session")
def silent_video(tmp_path_factory: pytest.TempPathFactory, ffmpeg_bin: str) -> Path:
    """Create a small synthetic video without an audio track."""
    target = tmp_path_factory.mktemp("media") / "silent.mp4"
    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=160x90:rate=12:duration=1",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        str(target),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f"Could not create the sample video: {result.stderr}")
    return target
