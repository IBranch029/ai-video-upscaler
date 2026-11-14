"""AI 4K video upscaler.

A production-ready pipeline that super-resolves low resolution video to 4K
using Real-ESRGAN, while preserving the original audio bit-exactly.

Typical use::

    from aivideoupscaler import UpscaleConfig, run_upscale

    config = UpscaleConfig(input_path="in.mp4", output_path="out.mkv")
    result = run_upscale(config)
    print(result.output_resolution)
"""

from __future__ import annotations

__version__ = "1.1.0"
__author__ = "IBranch029"
__license__ = "MIT"

from .config import (
    AudioSpec,
    BackendName,
    Codec,
    Container,
    EncoderSettings,
    MediaInfo,
    UpscaleConfig,
    VideoSpec,
)
from .engine import default_output_path, describe_environment, run_upscale
from .errors import (
    BackendError,
    BackendUnavailableError,
    CancelledError,
    ConfigurationError,
    CorruptMediaError,
    FFmpegError,
    FFmpegNotFoundError,
    MediaProbeError,
    ModelDownloadError,
    ModelError,
    PipelineError,
    UpscalerError,
)
from .models.registry import MODELS, ModelInfo, get_model, list_models
from .pipeline import PipelineResult, Progress

__all__ = [
    "MODELS",
    "AudioSpec",
    "BackendError",
    "BackendName",
    "BackendUnavailableError",
    "CancelledError",
    "Codec",
    "ConfigurationError",
    "Container",
    "CorruptMediaError",
    "EncoderSettings",
    "FFmpegError",
    "FFmpegNotFoundError",
    "MediaInfo",
    "MediaProbeError",
    "ModelDownloadError",
    "ModelError",
    "ModelInfo",
    "PipelineError",
    "PipelineResult",
    "Progress",
    "UpscaleConfig",
    "UpscalerError",
    "VideoSpec",
    "__version__",
    "default_output_path",
    "describe_environment",
    "get_model",
    "list_models",
    "run_upscale",
]
