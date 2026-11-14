"""PySide6 graphical interface with drag-and-drop support.

The GUI is optional: PySide6 is only required when this module is imported.
It offers the same feature set as the command line client, driven by the same
:class:`~aivideoupscaler.config.UpscaleConfig` and engine.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

try:
    from PySide6.QtCore import QObject, Signal
    from PySide6.QtGui import QDragEnterEvent, QDropEvent
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QProgressBar,
        QPushButton,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - PySide6 is an optional dependency
    raise ImportError(
        "The graphical interface requires PySide6. Install it with "
        "'pip install aivideoupscaler[gui]'."
    ) from exc

from ..config import BackendName, Codec, Container, EncoderSettings, UpscaleConfig
from ..engine import default_output_path, run_upscale
from ..errors import UpscalerError
from ..models.registry import list_models
from ..pipeline import Progress


class _Worker(QObject):
    """Runs an upscale job in a background thread."""

    progress = Signal(int, int, str)
    finished = Signal(str, bool)
    failed = Signal(str)

    def __init__(self, config: UpscaleConfig) -> None:
        super().__init__()
        self.config = config
        self._cancelled = False

    def cancel(self) -> None:
        """Request that the running job stops."""
        self._cancelled = True

    def run(self) -> None:
        """Execute the job and emit the outcome."""
        try:
            result = run_upscale(
                self.config,
                progress_cb=self._on_progress,
                cancel_check=lambda: self._cancelled,
            )
        except UpscalerError as exc:
            self.failed.emit(str(exc))
            return
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.finished.emit(str(result.output_path), True)

    def _on_progress(self, update: Progress) -> None:
        """Forward the pipeline progress to the UI thread."""
        self.progress.emit(update.frames_done, update.frames_total, update.stage)


class DropArea(QLineEdit):
    """A line edit that accepts dragged-and-dropped video files."""

    fileDropped = Signal(str)  # noqa: N815

    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)
        self.setPlaceholderText("Drop a video file here or click Browse...")

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Accept the drag when it carries at least one URL."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        """Store the first dropped file path and notify the listener."""
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path:
                self.setText(path)
                self.fileDropped.emit(path)
                event.acceptProposedAction()
                return
        event.ignore()


class MainWindow(QWidget):
    """The main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AI 4K Video Upscaler")
        self.resize(720, 520)
        self._thread: threading.Thread | None = None
        self._worker: _Worker | None = None
        self._build_ui()

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        """Create and lay out the widgets."""
        layout = QVBoxLayout(self)

        input_box = QGroupBox("Input")
        input_layout = QFormLayout(input_box)
        self.input_edit = DropArea()
        self.input_edit.fileDropped.connect(self._on_file_dropped)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse_input)
        row = QHBoxLayout()
        row.addWidget(self.input_edit, 1)
        row.addWidget(browse)
        input_layout.addRow("Video file", row)
        layout.addWidget(input_box)

        output_box = QGroupBox("Output")
        output_layout = QFormLayout(output_box)
        self.output_edit = QLineEdit()
        output_layout.addRow("Output file", self.output_edit)
        output_button = QPushButton("Choose...")
        output_button.clicked.connect(self._browse_output)
        output_layout.addRow("", output_button)
        layout.addWidget(output_box)

        settings_box = QGroupBox("Settings")
        settings_layout = QFormLayout(settings_box)

        self.model_combo = QComboBox()
        for model in list_models():
            self.model_combo.addItem(model.display_name, model.key)
        settings_layout.addRow("Model", self.model_combo)

        self.backend_combo = QComboBox()
        self.backend_combo.addItem("Automatic", "")
        for backend in BackendName:
            self.backend_combo.addItem(backend.value, backend.value)
        settings_layout.addRow("Backend", self.backend_combo)

        self.codec_combo = QComboBox()
        for codec in Codec:
            self.codec_combo.addItem(codec.value.upper(), codec.value)
        settings_layout.addRow("Codec", self.codec_combo)

        self.container_combo = QComboBox()
        for container in Container:
            self.container_combo.addItem(container.value.upper(), container.value)
        settings_layout.addRow("Container", self.container_combo)

        self.height_spin = QSpinBox()
        self.height_spin.setRange(0, 16384)
        self.height_spin.setValue(2160)
        self.height_spin.setSpecialValueText("source")
        settings_layout.addRow("Target height", self.height_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(4)
        settings_layout.addRow("Batch size", self.batch_spin)

        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 16)
        self.workers_spin.setValue(1)
        settings_layout.addRow("Workers", self.workers_spin)

        self.tile_spin = QSpinBox()
        self.tile_spin.setRange(0, 4096)
        self.tile_spin.setSpecialValueText("auto")
        settings_layout.addRow("Tile size", self.tile_spin)

        self.audio_check = QCheckBox("Keep the original audio (bit-exact)")
        self.audio_check.setChecked(True)
        settings_layout.addRow("", self.audio_check)

        layout.addWidget(settings_box)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.status_label = QLabel("Ready.")
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)

        buttons = QHBoxLayout()
        self.start_button = QPushButton("Start upscaling")
        self.start_button.clicked.connect(self._start)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _on_file_dropped(self, path: str) -> None:
        """React to a file being dropped onto the input field."""
        self.input_edit.setText(path)
        self.output_edit.setText(str(default_output_path(Path(path))))
        self.status_label.setText(f"Loaded {Path(path).name}")

    def _browse_input(self) -> None:
        """Open a file dialog to pick the input video."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select a video",
            "",
            "Video files (*.mp4 *.mkv *.mov *.avi *.webm);;All files (*)",
        )
        if path:
            self._on_file_dropped(path)

    def _browse_output(self) -> None:
        """Open a file dialog to pick the output file."""
        path, _ = QFileDialog.getSaveFileName(self, "Save the result", self.output_edit.text())
        if path:
            self.output_edit.setText(path)

    def _build_config(self) -> UpscaleConfig:
        """Create the configuration from the current widget values."""
        source = Path(self.input_edit.text())
        output = Path(self.output_edit.text() or default_output_path(source))

        backend_value = self.backend_combo.currentData()
        backend = BackendName(backend_value) if backend_value else None

        return UpscaleConfig(
            input_path=source,
            output_path=output,
            model=self.model_combo.currentData() or "realesr-animevideov3",
            backend=backend,
            target_height=self.height_spin.value() or None,
            batch_size=self.batch_spin.value(),
            workers=self.workers_spin.value(),
            tile=self.tile_spin.value(),
            overwrite=True,
            encoder=EncoderSettings(
                codec=Codec(self.codec_combo.currentData()),
                container=Container(self.container_combo.currentData()),
                audio_codec="copy" if self.audio_check.isChecked() else "aac",
            ),
        )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _start(self) -> None:
        """Start the upscale job in a background thread."""
        source = self.input_edit.text()
        if not source or not Path(source).is_file():
            self.status_label.setText("Please choose a valid input file.")
            return

        try:
            config = self._build_config()
        except (UpscalerError, ValueError) as exc:
            self.status_label.setText(f"Invalid configuration: {exc}")
            return

        self._worker = _Worker(config)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)

        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting...")

        self._thread = threading.Thread(target=self._worker.run, daemon=True)
        self._thread.start()

    def _cancel(self) -> None:
        """Ask the running job to stop."""
        if self._worker is not None:
            self._worker.cancel()
            self.status_label.setText("Cancelling...")

    def _on_progress(self, done: int, total: int, stage: str) -> None:
        """Update the progress bar."""
        if total > 0:
            self.progress_bar.setValue(int(done * 100 / total))
        self.status_label.setText(f"{stage.capitalize()} - {done}/{total} frames")

    def _on_finished(self, path: str, _ok: bool) -> None:
        """Report a successful job."""
        self.progress_bar.setValue(100)
        self.status_label.setText(f"Done: {path}")
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)

    def _on_failed(self, message: str) -> None:
        """Report a failed job."""
        self.status_label.setText(f"Failed: {message}")
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)


def run_gui() -> int:
    """Start the Qt event loop."""
    app = QApplication(sys.argv)
    app.setApplicationName("AI 4K Video Upscaler")
    window = MainWindow()
    window.show()
    return app.exec()


def main() -> None:
    """Console script entry point for the GUI."""
    raise SystemExit(run_gui())


if __name__ == "__main__":
    main()
