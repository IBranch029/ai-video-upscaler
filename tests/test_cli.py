"""Tests for the command line interface."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from aivideoupscaler.ui.cli import app

runner = CliRunner()


def test_help_is_shown() -> None:
    """The top level help page renders."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "4K" in result.stdout


def test_version_is_reported() -> None:
    """The version flag prints the package version."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "aivu" in result.stdout


@pytest.mark.parametrize("command", ["upscale", "info", "doctor", "tui", "models"])
def test_commands_are_registered(command: str) -> None:
    """Every documented command appears in the help page."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert command in result.stdout


def test_the_help_page_mentions_the_interactive_interface() -> None:
    """The help page explains that running without a command opens the TUI."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "interactive" in result.stdout.lower()


def test_the_tui_command_is_registered() -> None:
    """The ``tui`` command has its own help page."""
    result = runner.invoke(app, ["tui", "--help"])
    assert result.exit_code == 0
    assert "interactive" in result.stdout.lower()


def test_unknown_model_is_rejected(tmp_path: Path) -> None:
    """An unknown model name produces a friendly error."""
    source = tmp_path / "in.mp4"
    source.write_bytes(b"not a real video")
    result = runner.invoke(app, ["upscale", str(source), "--model", "nope"])
    assert result.exit_code == 1


def test_unknown_backend_is_rejected(tmp_path: Path) -> None:
    """An unknown backend name produces a friendly error."""
    source = tmp_path / "in.mp4"
    source.write_bytes(b"not a real video")
    result = runner.invoke(app, ["upscale", str(source), "--backend", "nope"])
    assert result.exit_code == 1


def test_missing_input_is_reported(tmp_path: Path) -> None:
    """A missing input file is reported instead of raising."""
    result = runner.invoke(app, ["upscale", str(tmp_path / "missing.mp4")])
    assert result.exit_code == 1


def test_doctor_runs() -> None:
    """The doctor command reports the environment."""
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Backends" in result.stdout


def test_the_frozen_console_is_kept_open(monkeypatch: pytest.MonkeyPatch) -> None:
    """A frozen executable waits before the console window is closed."""
    from aivideoupscaler.ui import cli

    monkeypatch.setattr(cli.sys, "frozen", True, raising=False)
    monkeypatch.delenv("AIVU_NO_PAUSE", raising=False)
    assert cli._should_pause() is True


def test_the_pause_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """``AIVU_NO_PAUSE`` disables the pause for scripted usage."""
    from aivideoupscaler.ui import cli

    monkeypatch.setattr(cli.sys, "frozen", True, raising=False)
    monkeypatch.setenv("AIVU_NO_PAUSE", "1")
    assert cli._should_pause() is False


def test_a_normal_installation_never_pauses(monkeypatch: pytest.MonkeyPatch) -> None:
    """A script installation must never block on a key press."""
    from aivideoupscaler.ui import cli

    monkeypatch.setattr(cli.sys, "frozen", False, raising=False)
    assert cli._should_pause() is False


def test_models_list_runs() -> None:
    """The model list renders a table."""
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    assert "realesr-animevideov3" in result.stdout


@pytest.mark.requires_ffmpeg
def test_info_reports_the_media(sample_video: Path) -> None:
    """The info command describes a media file."""
    result = runner.invoke(app, ["info", str(sample_video)])
    assert result.exit_code == 0
    assert "160x90" in result.stdout


@pytest.mark.requires_ffmpeg
def test_dry_run_reports_the_plan(sample_video: Path, tmp_path: Path) -> None:
    """The dry run prints the planned upscale without doing the work."""
    target = tmp_path / "out.mkv"
    result = runner.invoke(app, ["upscale", str(sample_video), "-o", str(target), "--dry-run"])
    assert result.exit_code == 0
    assert "Dry run" in result.stdout
    assert not target.exists()
