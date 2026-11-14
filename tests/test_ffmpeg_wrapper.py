"""Tests for the FFmpeg wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest

from aivideoupscaler.errors import CorruptMediaError, MediaProbeError
from aivideoupscaler.ffmpeg_wrapper import (
    build_decode_command,
    build_encode_command,
    build_mux_command,
    has_encoder,
    list_encoders,
    parse_rational,
    probe,
)


def test_parse_rational_handles_fractions() -> None:
    """FFmpeg's rational frame rates are converted to floats."""
    assert parse_rational("30000/1001") == pytest.approx(29.97, abs=0.01)
    assert parse_rational("25") == 25.0
    assert parse_rational("0/0") == 0.0
    assert parse_rational("") == 0.0


def test_probe_reads_the_video_spec(sample_video: Path) -> None:
    """The probe reports the resolution, frame rate and duration."""
    media = probe(sample_video)
    assert media.video.width == 160
    assert media.video.height == 90
    assert media.video.fps == pytest.approx(12.0, abs=0.5)
    assert media.video.duration == pytest.approx(1.0, abs=0.3)
    assert media.video.frame_count > 0


def test_probe_detects_the_audio_track(sample_video: Path) -> None:
    """The audio stream of the sample file is detected."""
    media = probe(sample_video)
    assert media.has_audio
    assert media.audios[0].codec == "aac"


def test_probe_reports_a_missing_file(tmp_path: Path) -> None:
    """A missing input raises a probe error instead of crashing."""
    with pytest.raises(MediaProbeError):
        probe(tmp_path / "nope.mp4")


def test_probe_reports_an_empty_file(tmp_path: Path) -> None:
    """An empty file is reported as corrupt."""
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(CorruptMediaError):
        probe(empty)


def test_probe_reports_a_directory(tmp_path: Path) -> None:
    """A directory is not a valid input."""
    with pytest.raises(MediaProbeError):
        probe(tmp_path)


def test_probe_reports_a_non_video_file(tmp_path: Path) -> None:
    """A text file is rejected because it has no video track."""
    junk = tmp_path / "notes.txt"
    junk.write_text("this is not a video", encoding="utf-8")
    with pytest.raises(MediaProbeError):
        probe(junk)


def test_decode_command_produces_rawvideo() -> None:
    """The decode command pipes raw BGR24 frames to stdout."""
    cmd = build_decode_command("ffmpeg", Path("in.mp4"), 640, 360, 30.0)
    assert "rawvideo" in cmd
    assert "bgr24" in cmd
    assert cmd[-1] == "-"


def test_encode_command_uses_the_requested_geometry() -> None:
    """The encode command carries the target resolution and frame rate."""
    cmd = build_encode_command("ffmpeg", Path("out.mkv"), 3840, 2160, 25.0)
    assert "3840x2160" in cmd
    assert "25.000000" in cmd
    assert "libx264" in cmd


def test_encode_command_can_use_hardware_encoders() -> None:
    """A hardware encoder replaces the software codec."""
    cmd = build_encode_command("ffmpeg", Path("out.mkv"), 1920, 1080, 30.0, hwaccel="cuda")
    assert "h264_nvenc" in cmd
    assert "libx264" not in cmd


def test_mux_command_copies_the_audio() -> None:
    """The mux command copies the video and the audio streams."""
    cmd = build_mux_command(
        "ffmpeg", Path("v.mkv"), Path("src.mp4"), Path("out.mkv"), audio_codec="copy"
    )
    assert "-c:a" in cmd
    assert cmd[cmd.index("-c:a") + 1] == "copy"
    assert "copy" in cmd  # the video stream is copied as well


def test_encoder_detection_works(ffmpeg_bin: str) -> None:
    """The encoder list of the local FFmpeg build is readable."""
    encoders = list_encoders(ffmpeg_bin)
    assert encoders
    assert has_encoder("libx264", ffmpeg_bin)
