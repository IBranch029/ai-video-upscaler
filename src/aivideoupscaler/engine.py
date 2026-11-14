"""High level engine API.

This module is the stable entry point for the library: it validates the
configuration, probes the input, selects a backend and runs the pipeline.
Both the CLI and the GUI build on top of it.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from .backends.base import BackendStatus
from .backends.factory import BackendOptions, auto_backend, build_backend
from .config import UpscaleConfig
from .errors import ConfigurationError, ModelError
from .ffmpeg_wrapper import probe
from .models.downloader import is_model_present, model_path
from .models.registry import get_model
from .pipeline import PipelineResult, Progress, UpscalePipeline

ProgressCallback = Callable[[Progress], None]


def default_output_path(input_path: Path, suffix: str = "_4k") -> Path:
    """Build a sensible output path for ``input_path``."""
    source = Path(input_path)
    return source.with_name(f"{source.stem}{suffix}{source.suffix}")


def resolve_backend(config: UpscaleConfig) -> tuple[object, list[BackendStatus]]:
    """Select the backend to use for ``config``.

    The result contains the backend instance and the status of every backend
    that was evaluated, which lets the UI explain the decision.
    """
    model = get_model(config.model)
    options = BackendOptions(
        tile=config.tile,
        device="auto",
        gpu_id=-1,
        threads=max(1, config.workers),
    )

    if config.backend is not None:
        backend = build_backend(config.backend, model, options)
        return backend, [backend.status()]

    return auto_backend(model, options)


def ensure_model_available(config: UpscaleConfig, backend: object) -> None:
    """Verify that the weights the backend needs are present locally."""
    status = backend.status()  # type: ignore[attr-defined]
    if not status.requires_model:
        return
    if backend.__class__.__name__ == "NcnnBackend":
        return
    if not is_model_present(config.model):
        raise ModelError(
            f"The weights for '{config.model}' are not installed. "
            "Run 'aivu models install' to download them."
        )


def run_upscale(
    config: UpscaleConfig,
    progress_cb: ProgressCallback | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> PipelineResult:
    """Run one complete upscale job.

    Raises a subclass of :class:`~aivideoupscaler.errors.UpscalerError` on
    every expected failure mode.
    """
    input_path = Path(config.input_path)
    if not input_path.exists():
        raise ConfigurationError(f"The input file does not exist: {input_path}")
    if input_path.is_dir():
        raise ConfigurationError(f"The input path is a directory, not a file: {input_path}")

    output_path = Path(config.output_path)
    if output_path.exists() and not config.overwrite:
        raise ConfigurationError(
            f"The output file already exists: {output_path}. Use --overwrite to replace it."
        )

    media = probe(input_path)
    if media.video.width <= 0 or media.video.height <= 0:
        raise ConfigurationError("The input video has an invalid resolution.")

    backend, _statuses = resolve_backend(config)
    ensure_model_available(config, backend)

    try:
        pipeline = UpscalePipeline(
            config,
            media,
            backend,
            progress_cb=progress_cb,
            cancel_check=cancel_check,
        )
        return pipeline.run()
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            close()


def describe_environment(model: str = "realesr-animevideov3") -> dict[str, object]:
    """Collect diagnostic information about the current environment."""
    from .backends.factory import list_backends
    from .ffmpeg_wrapper import ffmpeg_version, ffprobe_version, find_ffmpeg, find_ffprobe

    statuses = list_backends(model)
    return {
        "ffmpeg": ffmpeg_version(),
        "ffmpeg_path": find_ffmpeg() or "",
        "ffprobe": ffprobe_version(),
        "ffprobe_path": find_ffprobe() or "",
        "backends": statuses,
        "model": model,
        "model_present": is_model_present(model),
        "model_path": str(model_path(model)),
        "cpu_count": os.cpu_count() or 1,
    }
