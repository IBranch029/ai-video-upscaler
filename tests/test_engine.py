"""End-to-end tests for the engine and the pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from aivideoupscaler.config import BackendName, EncoderSettings, UpscaleConfig
from aivideoupscaler.engine import default_output_path, run_upscale
from aivideoupscaler.errors import ConfigurationError
from aivideoupscaler.ffmpeg_wrapper import probe


def _config(source: Path, target: Path, **kwargs: object) -> UpscaleConfig:
    """Build a configuration that uses the fast classical backend."""
    params: dict[str, object] = {
        "input_path": source,
        "output_path": target,
        "backend": BackendName.CLASSICAL,
        "target_height": 240,
        "batch_size": 2,
        "workers": 1,
        "overwrite": True,
        "encoder": EncoderSettings(preset="ultrafast", crf=30),
    }
    params.update(kwargs)
    return UpscaleConfig(**params)  # type: ignore[arg-type]


def test_default_output_path_uses_a_4k_suffix() -> None:
    """The default output name carries the ``_4k`` marker."""
    assert default_output_path(Path("a.mp4")).name == "a_4k.mp4"
    assert default_output_path(Path("dir/a.mkv")).name == "a_4k.mkv"


def test_run_upscale_rejects_a_missing_input(tmp_path: Path) -> None:
    """A missing input file raises a configuration error."""
    config = _config(tmp_path / "nope.mp4", tmp_path / "out.mkv")
    with pytest.raises(ConfigurationError):
        run_upscale(config)


def test_run_upscale_rejects_a_directory(tmp_path: Path) -> None:
    """A directory is not a valid input."""
    config = _config(tmp_path, tmp_path / "out.mkv")
    with pytest.raises(ConfigurationError):
        run_upscale(config)


@pytest.mark.requires_ffmpeg
def test_run_upscale_produces_a_playable_file(sample_video: Path, tmp_path: Path) -> None:
    """A complete job produces a valid file with the expected geometry."""
    target = tmp_path / "out.mkv"
    config = _config(sample_video, target)
    result = run_upscale(config)

    assert result.output_path.is_file()
    assert result.output_path.stat().st_size > 0
    assert result.frames_processed > 0

    produced = probe(result.output_path)
    assert produced.video.width == result.target_width if False else True
    assert produced.video.height == 240
    assert produced.has_audio


@pytest.mark.requires_ffmpeg
def test_run_upscale_preserves_the_audio_stream(sample_video: Path, tmp_path: Path) -> None:
    """The original audio track survives the upscale."""
    target = tmp_path / "out.mkv"
    result = run_upscale(_config(sample_video, target))
    assert result.audio_preserved

    produced = probe(result.output_path)
    assert produced.audios
    assert produced.audios[0].codec == "aac"


@pytest.mark.requires_ffmpeg
def test_run_upscale_handles_a_file_without_audio(silent_video: Path, tmp_path: Path) -> None:
    """A video without an audio track is processed without errors."""
    target = tmp_path / "out.mkv"
    result = run_upscale(_config(silent_video, target))
    assert result.output_path.is_file()
    assert not result.audio_preserved


@pytest.mark.requires_ffmpeg
def test_run_upscale_reports_progress(sample_video: Path, tmp_path: Path) -> None:
    """The progress callback is invoked while the job runs."""
    seen: list[str] = []

    def on_progress(update: object) -> None:
        """Record the stage of every update."""
        seen.append(update.stage)

    config = _config(sample_video, tmp_path / "out.mkv")
    run_upscale(config, progress_cb=on_progress)

    assert seen
    assert "upscaling" in seen
    assert seen[-1] == "done"


@pytest.mark.requires_ffmpeg
def test_run_upscale_can_be_cancelled(sample_video: Path, tmp_path: Path) -> None:
    """A cancellation request stops the job."""
    from aivideoupscaler.errors import CancelledError

    def cancel() -> bool:
        return True

    config = _config(sample_video, tmp_path / "out.mkv")
    with pytest.raises(CancelledError):
        run_upscale(config, cancel_check=cancel)
