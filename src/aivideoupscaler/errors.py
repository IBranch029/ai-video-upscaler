"""Exception hierarchy for the AI video upscaler.

Every error raised on purpose by this package derives from :class:`UpscalerError`,
which lets the CLI and the GUI present a single, predictable error surface.
"""

from __future__ import annotations


class UpscalerError(Exception):
    """Base class for every deliberate error raised by :mod:`aivideoupscaler`."""


class ConfigurationError(UpscalerError):
    """Invalid or contradictory configuration was supplied by the user."""


class FFmpegNotFoundError(UpscalerError):
    """FFmpeg or ffprobe could not be located on the host system."""

    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            message
            or "FFmpeg was not found. Install it system-wide, point FFMPEG_BINARY at the "
            "executable, or run 'aivu ffmpeg install' to download a static build."
        )


class FFmpegError(UpscalerError):
    """An ffmpeg/ffprobe invocation failed."""

    def __init__(self, message: str, *, returncode: int | None = None, stderr: str = "") -> None:
        self.returncode = returncode
        self.stderr = stderr
        detail = f"{message} (exit code {returncode})" if returncode is not None else message
        if stderr.strip():
            detail = f"{detail}\n{stderr.strip()}"
        super().__init__(detail)


class MediaProbeError(FFmpegError):
    """The input file could not be probed or is not a usable video."""


class CorruptMediaError(MediaProbeError):
    """The input file appears to be damaged, truncated or empty."""


class BackendError(UpscalerError):
    """The selected upscaling backend failed while processing frames."""


class BackendUnavailableError(BackendError):
    """The selected backend cannot run in this environment."""


class ModelError(UpscalerError):
    """Model weights are missing, corrupt or incompatible."""


class ModelDownloadError(ModelError):
    """Downloading model weights failed."""


class PipelineError(UpscalerError):
    """The frame processing pipeline failed."""


class CancelledError(UpscalerError):
    """The user cancelled the operation."""
