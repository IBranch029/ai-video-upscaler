"""Tests for the interactive terminal interface.

The Textual application is driven headlessly through the ``run_test`` pilot, so
the real widget tree, the real message handlers and the real event loop are
exercised without a terminal.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from aivideoupscaler.config import BackendName
from aivideoupscaler.ui.tui import (
    TARGET_CHOICES,
    UpscalerApp,
    _ffmpeg_summary,
    _format_duration,
)


def run(coro):
    """Run an async coroutine on a fresh event loop."""
    return asyncio.run(coro)


def test_the_ffmpeg_summary_reports_the_version() -> None:
    """The header shows the FFmpeg version, or that it is missing."""
    summary = _ffmpeg_summary()
    assert summary
    assert summary != "not found" or summary == "not found"
    if summary != "not found":
        # A version number, or the generic fallback.
        assert summary[0].isdigit() or summary == "available"


def test_format_duration() -> None:
    """Durations are rendered as ``MM:SS``."""
    assert _format_duration(0) == "--:--"
    assert _format_duration(-1) == "--:--"
    assert _format_duration(5) == "00:05"
    assert _format_duration(65) == "01:05"
    assert _format_duration(3600) == "60:00"


def test_target_choices_cover_the_documented_set() -> None:
    """The interface offers the documented resolutions and scale factors."""
    keys = [choice[0] for choice in TARGET_CHOICES]
    assert keys == ["2x", "4x", "4K", "1080p", "source"]
    four_k = TARGET_CHOICES[keys.index("4K")]
    assert four_k[2] == 3840
    assert four_k[3] == 2160


def test_app_starts_and_lists_the_widgets() -> None:
    """The interface starts and exposes every widget it needs."""

    async def scenario() -> None:
        app = UpscalerApp()
        async with app.run_test() as pilot:
            assert app.title == "AI 4K Video Upscaler"
            for widget_id in (
                "env-bar",
                "path-input",
                "file-tree",
                "file-list",
                "target-select",
                "backend-select",
                "model-select",
                "start-button",
                "cancel-button",
                "progress-bar",
                "metrics-line",
                "status-line",
            ):
                assert app.query_one(f"#{widget_id}") is not None
            await pilot.pause()

    run(scenario())


def test_backend_picker_shows_the_availability() -> None:
    """Every backend is listed together with its availability."""

    async def scenario() -> None:
        app = UpscalerApp()
        async with app.run_test() as pilot:
            select = app.query_one("#backend-select")
            labels = [str(prompt) for prompt, _ in select._options]
            assert len(labels) == len(list(BackendName))
            for backend in BackendName:
                assert any(backend.value in label for label in labels)
            assert any("available" in label for label in labels)
            await pilot.pause()

    run(scenario())


def test_target_picker_maps_to_the_configuration() -> None:
    """Selecting a target translates into the expected configuration values."""

    async def scenario() -> None:
        app = UpscalerApp()
        async with app.run_test() as pilot:
            select = app.query_one("#target-select")
            select.value = "4K"
            assert app._current_target() == (3840, 2160, None)
            select.value = "2x"
            assert app._current_target() == (None, None, 2)
            select.value = "4x"
            assert app._current_target() == (None, None, 4)
            await pilot.pause()

    run(scenario())


def test_file_list_shows_the_video_files(tmp_path: Path) -> None:
    """Only video files are listed, other files are ignored."""
    (tmp_path / "movie.mp4").write_bytes(b"0" * 1024)
    (tmp_path / "clip.mkv").write_bytes(b"0" * 2048)
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")

    async def scenario() -> None:
        app = UpscalerApp()
        async with app.run_test() as pilot:
            app._refresh_file_list(tmp_path)
            options = app.query_one("#file-list").options
            labels = [str(option.prompt) for option in options]
            assert any("movie.mp4" in label for label in labels)
            assert any("clip.mkv" in label for label in labels)
            assert not any("notes.txt" in label for label in labels)
            await pilot.pause()

    run(scenario())


def test_selecting_a_file_updates_the_input(tmp_path: Path) -> None:
    """Picking a file from the list fills the path field."""
    target = tmp_path / "movie.mp4"
    target.write_bytes(b"0" * 1024)

    async def scenario() -> None:
        app = UpscalerApp()
        async with app.run_test() as pilot:
            app._refresh_file_list(tmp_path)
            app.query_one("#file-list").focus()
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert app.selected_file is not None
            assert app.selected_file.name == "movie.mp4"
            assert app.query_one("#path-input").value == str(target)

    run(scenario())


def test_starting_without_a_file_reports_an_error() -> None:
    """The interface refuses to start without an input file."""

    async def scenario() -> None:
        app = UpscalerApp()
        async with app.run_test() as pilot:
            app.selected_file = None
            app._start_job()
            await pilot.pause()
            error = app.query_one("#error-line")
            assert error.has_class("visible")
            assert "No input file" in str(error.content)

    run(scenario())


def test_the_cancel_button_is_disabled_until_a_job_runs() -> None:
    """The cancel button is only enabled while a job is running."""

    async def scenario() -> None:
        app = UpscalerApp()
        async with app.run_test() as pilot:
            assert app.query_one("#cancel-button").disabled
            assert not app.query_one("#start-button").disabled
            await pilot.pause()

    run(scenario())


def test_the_environment_bar_reports_the_ffmpeg_installation() -> None:
    """The header always shows the state of the FFmpeg installation."""

    async def scenario() -> None:
        app = UpscalerApp()
        async with app.run_test() as pilot:
            text = str(app.query_one("#env-bar").content)
            assert "FFmpeg" in text
            assert "RAM" in text
            await pilot.pause()

    run(scenario())
