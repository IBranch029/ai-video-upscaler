"""Interactive terminal user interface.

The interface is built with Textual and offers everything the command line
client does, but without remembering any flags:

* a file browser that lists the video files of the selected folder,
* pickers for the target resolution, the backend and the model,
* a live progress display with the throughput, the ETA, the video memory and
  the system memory,
* a cancel button (and ``Ctrl+C``) that stops the running job.

The application can be started with ``aivu tui`` or by running ``aivu``
without any argument.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, HorizontalGroup, Vertical
from textual.widgets import (
    Button,
    DirectoryTree,
    Footer,
    Header,
    Input,
    Label,
    OptionList,
    ProgressBar,
    Select,
    Static,
)
from textual.widgets.option_list import Option

from ..config import BackendName, Codec, Container, EncoderSettings, UpscaleConfig
from ..engine import default_output_path, run_upscale
from ..errors import UpscalerError
from ..models.registry import list_models
from ..pipeline import PipelineResult, Progress
from ..system_info import format_megabytes, gpu_info, memory_info

VIDEO_SUFFIXES = (
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".webm",
    ".m4v",
    ".wmv",
    ".flv",
    ".ts",
    ".m2ts",
    ".mpg",
    ".mpeg",
    ".vob",
    ".ogv",
)

#: The selectable output geometries: key, label, width, height.
TARGET_CHOICES: list[tuple[str, str, int | None, int | None]] = [
    ("2x", "2x (double the source resolution)", None, None),
    ("4x", "4x (quadruple the source resolution)", None, None),
    ("4K", "4K - 3840x2160", 3840, 2160),
    ("1080p", "Full HD - 1920x1080", 1920, 1080),
    ("source", "Keep the source resolution", None, None),
]


@dataclass
class BackendChoice:
    """A backend option together with its availability."""

    name: str
    label: str
    available: bool
    detail: str


def backend_options() -> tuple[list[tuple[str, str]], str | None]:
    """Return the backend picker options and the preferred backend.

    Each option shows the device the backend would use and whether it is
    available in the current environment, so the user can see at a glance
    which backend will actually run.
    """
    from ..backends.factory import list_backends

    options: list[tuple[str, str]] = []
    preferred: str | None = None
    for status in list_backends():
        mark = "available" if status.available else "unavailable"
        options.append((f"{status.name} - {status.device} ({mark})", status.name))
        if status.available and preferred is None:
            preferred = status.name
    if preferred is None and options:
        preferred = options[0][1]
    return options, preferred


class UpscalerApp(App):
    """The interactive upscaler interface."""

    CSS = """
    Screen {
        layers: base overlay;
    }

    #env-bar {
        height: auto;
        padding: 0 1;
        background: $boost;
        color: $text;
    }

    #source-panel, #settings-panel, #progress-panel {
        border: round $primary;
        padding: 0 1;
        margin-bottom: 1;
    }

    .panel-title {
        text-style: bold;
        color: $accent;
    }

    #path-row, #browse-row {
        height: auto;
    }

    #path-input {
        width: 1fr;
    }

    #file-tree {
        width: 40%;
        height: 12;
        border: round $surface;
    }

    #file-list {
        width: 1fr;
        height: 12;
        border: round $surface;
    }

    #target-select, #backend-select {
        width: 44;
    }

    #model-select {
        width: 44;
    }

    #action-row {
        height: auto;
        align-horizontal: center;
        margin-bottom: 1;
    }

    #start-button {
        margin-right: 2;
    }

    #status-line, #metrics-line {
        height: auto;
        color: $text-muted;
    }

    #progress-panel {
        display: none;
    }

    #progress-panel.visible {
        display: block;
    }

    #result-line {
        height: auto;
        display: none;
        color: $success;
    }

    #result-line.visible {
        display: block;
    }

    #error-line {
        height: auto;
        display: none;
        color: $error;
    }

    #error-line.visible {
        display: block;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "cancel_job", "Cancel", show=True),
        Binding("ctrl+r", "reload", "Reload"),
        Binding("ctrl+o", "open_path", "Open path"),
        Binding("f1", "focus('file-list')", "Files"),
        Binding("f2", "focus('path-input')", "Path"),
        Binding("f3", "focus('target-select')", "Target"),
        Binding("f4", "focus('backend-select')", "Backend"),
        Binding("f5", "focus('model-select')", "Model"),
        Binding("ctrl+q", "quit_app", "Quit"),
    ]

    TITLE = "AI 4K Video Upscaler"
    SUB_TITLE = "Real-ESRGAN super-resolution with lossless audio"

    def __init__(self, initial_path: Path | None = None) -> None:
        super().__init__()
        self.initial_path = initial_path
        self.selected_file: Path | None = initial_path
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._last_metrics = 0.0
        self._last_progress: Progress | None = None

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def compose(self) -> ComposeResult:
        """Build the widget tree."""
        yield Header(show_clock=True)
        yield Static(id="env-bar")
        with Vertical(id="body"):
            with Vertical(id="source-panel"):
                yield Label("Source", classes="panel-title")
                with Horizontal(id="path-row"):
                    yield Input(
                        placeholder="Type a path to a video file and press Enter",
                        id="path-input",
                    )
                with Horizontal(id="browse-row"):
                    yield Button("Open path", id="open-path", variant="primary")
                    yield Button("Use current folder", id="use-cwd")
                    yield Button("Refresh", id="refresh")
                with HorizontalGroup():
                    yield DirectoryTree(
                        str(self.initial_path.parent if self.initial_path else Path.cwd()),
                        id="file-tree",
                    )
                    yield OptionList(id="file-list")
            with Vertical(id="settings-panel"):
                yield Label("Settings", classes="panel-title")
                with HorizontalGroup():
                    yield Label("Target resolution")
                    yield Select(
                        [(label, key) for key, label, _, _ in TARGET_CHOICES],
                        id="target-select",
                        allow_blank=False,
                        value="4K",
                    )
                with HorizontalGroup():
                    yield Label("Backend")
                    backend_choices, preferred_backend = backend_options()
                    yield Select(
                        backend_choices,
                        id="backend-select",
                        allow_blank=False,
                        value=preferred_backend,
                    )
                with HorizontalGroup():
                    yield Label("Model")
                    yield Select(
                        [(model.display_name, model.key) for model in list_models()],
                        id="model-select",
                        allow_blank=False,
                        value=list_models()[0].key if list_models() else None,
                    )
            with Horizontal(id="action-row"):
                yield Button("Start upscaling", id="start-button", variant="success")
                yield Button("Cancel", id="cancel-button", variant="error", disabled=True)
            yield Static("Ready.", id="status-line")
            with Vertical(id="progress-panel"):
                yield ProgressBar(id="progress-bar", show_eta=False)
                yield Static("", id="metrics-line")
            yield Static("", id="result-line")
            yield Static("", id="error-line")
        yield Footer()

    def on_mount(self) -> None:
        """Populate the widgets once the interface is ready."""
        self._refresh_environment()
        self._populate_targets()
        self._populate_backends()

        if self.initial_path and self.initial_path.is_file():
            self._set_selected_file(self.initial_path)

        self._refresh_file_list(self.initial_path.parent if self.initial_path else Path.cwd())

    # ------------------------------------------------------------------
    # Populating
    # ------------------------------------------------------------------
    def _refresh_environment(self) -> None:
        """Update the environment summary in the header."""
        gpu = gpu_info()
        memory = memory_info()
        parts = [f"FFmpeg {_ffmpeg_summary()}"]
        if gpu.available:
            parts.append(f"GPU {gpu.name} - {format_megabytes(gpu.total_mb)} VRAM")
        else:
            parts.append("no GPU detected - CPU mode")
        parts.append(
            f"RAM {format_megabytes(memory.used_mb)} / {format_megabytes(memory.total_mb)}"
        )
        self.query_one("#env-bar", Static).update("  |  ".join(parts))

    def _populate_targets(self) -> None:
        """Fill the target resolution picker."""
        select = self.query_one("#target-select", Select)
        select.set_options([(label, key) for key, label, _, _ in TARGET_CHOICES])
        select.value = "4K"

    def _populate_backends(self) -> None:
        """Fill the backend picker with the availability of each backend."""
        select = self.query_one("#backend-select", Select)
        options, preferred = backend_options()
        select.set_options(options)
        if preferred is not None:
            select.value = preferred

    def _refresh_file_list(self, directory: Path) -> None:
        """List the video files of ``directory``."""
        option_list = self.query_one("#file-list", OptionList)
        option_list.clear_options()
        try:
            entries = sorted(directory.iterdir())
        except (OSError, PermissionError):
            self.query_one("#status-line", Static).update(f"Cannot read {directory}")
            return

        found = 0
        for entry in entries:
            if entry.is_dir() or entry.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            try:
                size = entry.stat().st_size
            except OSError:
                size = 0
            option_list.add_option(
                Option(
                    f"{entry.name}  ({size / (1024 * 1024):.1f} MB)",
                    id=str(entry),
                )
            )
            found += 1

        if found:
            # Highlight the first entry so that Enter selects it right away.
            option_list.highlighted = 0
        else:
            option_list.add_option(
                Option("no video files in this folder", id="__empty__", disabled=True)
            )
        self.query_one("#status-line", Static).update(
            f"{found} video file(s) in {directory} - press F1 for the file list"
        )

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------
    @on(DirectoryTree.DirectorySelected)
    def _directory_selected(self, event: DirectoryTree.DirectorySelected) -> None:
        """Show the video files of the folder the user picked."""
        event.stop()
        self._refresh_file_list(Path(event.path))

    @on(OptionList.OptionSelected)
    def _file_selected(self, event: OptionList.OptionSelected) -> None:
        """Remember the file the user picked."""
        event.stop()
        option_id = event.option.id
        if not option_id:
            return
        self._set_selected_file(Path(option_id))

    @on(Input.Submitted, "#path-input")
    def _path_submitted(self, event: Input.Submitted) -> None:
        """Accept a path that was typed into the input field."""
        event.stop()
        raw = event.value.strip().strip('"')
        if not raw:
            return
        path = Path(raw).expanduser()
        if path.is_dir():
            self._refresh_file_list(path)
            self.query_one("#status-line", Static).update(f"Listing {path}")
            return
        if path.is_file():
            self._set_selected_file(path)
            return
        error = self.query_one("#error-line", Static)
        error.update(f"Path not found: {path}")
        error.add_class("visible")

    @on(Button.Pressed, "#open-path")
    def _open_path(self) -> None:
        """Focus the path input so the user can type a location."""
        self.query_one("#path-input", Input).focus()

    @on(Button.Pressed, "#use-cwd")
    def _use_cwd(self) -> None:
        """List the video files of the current working directory."""
        self._refresh_file_list(Path.cwd())

    @on(Button.Pressed, "#refresh")
    def _refresh_clicked(self) -> None:
        """Reload the environment and the file list."""
        self.action_reload()

    @on(Button.Pressed, "#start-button")
    def _start_clicked(self) -> None:
        """Start the upscale job."""
        self._start_job()

    @on(Button.Pressed, "#cancel-button")
    def _cancel_clicked(self) -> None:
        """Cancel the running job."""
        self.action_cancel_job()

    def action_reload(self) -> None:
        """Reload the environment information."""
        self._refresh_environment()
        self._populate_backends()

    def action_cancel_job(self) -> None:
        """Ask the running job to stop."""
        if self._thread is not None and self._thread.is_alive():
            self._cancel.set()
            self.query_one("#status-line", Static).update("Cancelling...")

    def action_open_path(self) -> None:
        """Focus the path input."""
        self.query_one("#path-input", Input).focus()

    def action_quit_app(self) -> None:
        """Cancel any running job and leave the application."""
        self._cancel.set()
        self.exit()

    # ------------------------------------------------------------------
    # Job handling
    # ------------------------------------------------------------------
    def _set_selected_file(self, path: Path) -> None:
        """Store the file the user selected."""
        self.selected_file = path
        self.query_one("#path-input", Input).value = str(path)
        self.query_one("#status-line", Static).update(f"Selected {path.name}")

    def _current_target(self) -> tuple[int | None, int | None, int | None]:
        """Return the ``(width, height, scale)`` of the selected target."""
        key = self.query_one("#target-select", Select).value
        if key is None:
            return None, None, None
        for choice_key, _, width, height in TARGET_CHOICES:
            if choice_key != key:
                continue
            if key == "2x":
                return None, None, 2
            if key == "4x":
                return None, None, 4
            if key == "source":
                return None, None, 1
            return width, height, None
        return None, None, None

    def _current_backend(self) -> BackendName | None:
        """Return the selected backend."""
        value = self.query_one("#backend-select", Select).value
        if value is None:
            return None
        try:
            return BackendName(str(value))
        except ValueError:
            return None

    def _build_config(self) -> UpscaleConfig:
        """Build the configuration from the current state of the interface."""
        if self.selected_file is None:
            raise UpscalerError("No input file has been selected.")

        width, height, scale = self._current_target()
        backend = self._current_backend()
        model_select = self.query_one("#model-select", Select)
        model = model_select.value or "realesr-animevideov3"

        return UpscaleConfig(
            input_path=self.selected_file,
            output_path=default_output_path(self.selected_file),
            model=str(model),
            backend=backend,
            target_width=width,
            target_height=height,
            scale=scale if scale and scale > 1 else None,
            batch_size=4,
            workers=1,
            overwrite=True,
            encoder=EncoderSettings(
                codec=Codec.H264,
                container=Container.MKV,
                preset="medium",
                audio_codec="copy",
            ),
        )

    def _start_job(self) -> None:
        """Start the upscale job in a background thread."""
        if self._thread is not None and self._thread.is_alive():
            self.query_one("#status-line", Static).update("A job is already running.")
            return

        try:
            config = self._build_config()
        except UpscalerError as exc:
            error = self.query_one("#error-line", Static)
            error.update(str(exc))
            error.add_class("visible")
            return

        self._cancel.clear()
        self.query_one("#error-line", Static).remove_class("visible")
        self.query_one("#result-line", Static).remove_class("visible")
        self.query_one("#progress-panel").add_class("visible")
        self.query_one("#start-button").disabled = True
        self.query_one("#cancel-button").disabled = False
        self.query_one("#status-line", Static).update("Starting...")
        self.query_one("#progress-bar", ProgressBar).update(progress=0)

        self._thread = threading.Thread(
            target=self._run_job,
            args=(config,),
            name="upscale-job",
            daemon=True,
        )
        self._thread.start()

    def _run_job(self, config: UpscaleConfig) -> None:
        """Execute the job and report the result back to the interface."""
        try:
            result = run_upscale(
                config,
                progress_cb=self._on_progress,
                cancel_check=self._cancel.is_set,
            )
        except UpscalerError as exc:
            self.call_from_thread(self._job_failed, str(exc))
            return
        except Exception as exc:
            self.call_from_thread(self._job_failed, f"{type(exc).__name__}: {exc}")
            return
        self.call_from_thread(self._job_finished, result)

    def _on_progress(self, update: Progress) -> None:
        """Forward the pipeline progress to the interface thread."""
        self.call_from_thread(self._apply_progress, update)

    def _apply_progress(self, update: Progress) -> None:
        """Update the progress bar and the metrics."""
        self._last_progress = update
        total = max(1, update.frames_total)
        self.query_one("#progress-bar", ProgressBar).update(
            progress=update.frames_done / total * 100
        )

        now = time.monotonic()
        if now - self._last_metrics < 0.5 and update.stage != "done":
            return
        self._last_metrics = now

        gpu = gpu_info()
        memory = memory_info()
        parts = [
            f"{update.frames_done}/{update.frames_total} frames",
            f"{update.fps:.1f} fps",
            f"ETA {_format_duration(update.eta_seconds)}",
        ]
        if gpu.available:
            parts.append(f"VRAM {format_megabytes(gpu.used_mb)}/{format_megabytes(gpu.total_mb)}")
        parts.append(f"RAM {format_megabytes(memory.used_mb)}/{format_megabytes(memory.total_mb)}")
        self.query_one("#metrics-line", Static).update("  |  ".join(parts))
        self.query_one("#status-line", Static).update(update.stage.capitalize())

    def _job_finished(self, result: PipelineResult) -> None:
        """Report a finished job."""
        self.query_one("#progress-bar", ProgressBar).update(progress=100)
        self.query_one("#progress-panel").remove_class("visible")
        self.query_one("#start-button").disabled = False
        self.query_one("#cancel-button").disabled = True
        self.query_one("#status-line", Static).update("Finished.")
        result_line = self.query_one("#result-line", Static)
        result_line.update(
            f"Saved {result.output_path}  ({result.output_resolution}, "
            f"{result.frames_processed} frames, {result.average_fps:.1f} fps)"
        )
        result_line.add_class("visible")

    def _job_failed(self, message: str) -> None:
        """Report a failed job."""
        self.query_one("#progress-panel").remove_class("visible")
        self.query_one("#start-button").disabled = False
        self.query_one("#cancel-button").disabled = True
        self.query_one("#status-line", Static).update("Failed.")
        error = self.query_one("#error-line", Static)
        error.update(message)
        error.add_class("visible")


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as ``MM:SS``."""
    if seconds <= 0:
        return "--:--"
    total = int(seconds)
    return f"{total // 60:02d}:{total % 60:02d}"


def _ffmpeg_summary() -> str:
    """Return a short description of the FFmpeg installation."""
    from ..ffmpeg_wrapper import ffmpeg_version

    version = ffmpeg_version()
    if not version:
        return "not found"
    for part in version.split():
        if part.startswith("ffmpeg") and any(char.isdigit() for char in part):
            return part.replace("ffmpeg", "").replace("version", "").strip()
    return "available"


def run_tui(initial_path: Path | None = None) -> int:
    """Start the interactive interface and return the exit code."""
    app = UpscalerApp(initial_path=initial_path)
    app.run()
    return 0
