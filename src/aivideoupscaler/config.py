"""Runtime configuration objects for the upscaling pipeline.

All tunables live in :class:`UpscaleConfig` so that the CLI, the GUI and the
Python API share exactly the same validation and defaults.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .errors import ConfigurationError


class BackendName(str, Enum):
    """Available super-resolution backends."""

    NCNN = "ncnn"
    """Real-ESRGAN compiled with ncnn/Vulkan (no PyTorch required)."""

    TORCH = "torch"
    """Real-ESRGAN running on PyTorch (CUDA / DirectML / MPS / CPU)."""

    CLASSICAL = "classical"
    """Dependency-free Lanczos + edge-aware sharpening fallback."""


class Codec(str, Enum):
    """Video codecs supported by the encoder stage."""

    H264 = "h264"
    H265 = "h265"
    AV1 = "av1"


class Container(str, Enum):
    """Output container formats."""

    MP4 = "mp4"
    MKV = "mkv"


@dataclass(frozen=True)
class VideoSpec:
    """Static description of a video track."""

    codec: str
    width: int
    height: int
    fps: float
    duration: float
    frame_count: int
    bitrate: int = 0
    pix_fmt: str = ""

    @property
    def aspect_ratio(self) -> float:
        if self.height == 0:
            return 0.0
        return self.width / self.height

    @property
    def resolution_label(self) -> str:
        return f"{self.width}x{self.height}"


@dataclass(frozen=True)
class AudioSpec:
    """Static description of an audio stream."""

    index: int
    codec: str
    channels: int
    sample_rate: int
    language: str = ""

    @property
    def layout(self) -> str:
        return {1: "mono", 2: "stereo"}.get(self.channels, f"{self.channels}ch")


@dataclass
class MediaInfo:
    """Everything the engine needs to know about an input file."""

    path: Path
    format_name: str
    duration: float
    size_bytes: int
    video: VideoSpec
    audios: list[AudioSpec] = field(default_factory=list)
    subtitles: int = 0
    bitrate: int = 0

    @property
    def has_audio(self) -> bool:
        return bool(self.audios)


@dataclass
class EncoderSettings:
    """Encoder (muxing) options."""

    codec: Codec = Codec.H264
    container: Container = Container.MKV
    crf: int = 18
    preset: str = "slow"
    pix_fmt: str = "yuv420p"
    audio_codec: str = "copy"
    """``copy`` keeps the original audio bit-exact, anything else re-encodes."""
    audio_bitrate: str = "192k"
    extra_args: list[str] = field(default_factory=list)
    hwaccel: str = ""
    """Optional hardware encoder prefix, e.g. ``cuda``, ``qsv`` or ``vaapi``."""


@dataclass
class UpscaleConfig:
    """Complete description of one upscale job."""

    input_path: Path
    output_path: Path
    model: str = "realesr-animevideov3"
    backend: BackendName | None = None
    """``None`` means auto-detect the best available backend."""
    target_width: int | None = None
    target_height: int | None = None
    """Target output geometry.  ``None`` keeps the model scale factor."""
    scale: int | None = None
    """Force a specific scale factor instead of the model's native one."""
    tile: int = 0
    """Tile size for memory-bounded inference.  ``0`` lets the backend decide."""
    batch_size: int = 4
    """Number of frames pushed through the backend at once (VRAM safety)."""
    workers: int = 1
    """Number of parallel inference worker processes."""
    encoder: EncoderSettings = field(default_factory=EncoderSettings)
    ffmpeg_path: str = ""
    """Explicit path to the ffmpeg binary; empty means auto-detect."""
    keep_temp: bool = False
    overwrite: bool = True
    progress_cb: object = None
    """Optional callable invoked with :class:`~aivideoupscaler.pipeline.Progress`."""
    cancel_check: object = None
    """Optional callable returning ``True`` to abort the job."""

    def __post_init__(self) -> None:
        self.input_path = Path(self.input_path)
        self.output_path = Path(self.output_path)
        if self.batch_size < 1:
            raise ConfigurationError("batch_size must be >= 1")
        if self.workers < 1:
            raise ConfigurationError("workers must be >= 1")
        if self.scale is not None and self.scale < 1:
            raise ConfigurationError("scale must be >= 1")
        if self.target_width is not None and self.target_width < 2:
            raise ConfigurationError("target_width must be >= 2")
        if self.target_height is not None and self.target_height < 2:
            raise ConfigurationError("target_height must be >= 2")

    def replace(self, **changes: object) -> UpscaleConfig:
        """Return a copy with ``changes`` applied."""
        return dataclasses.replace(self, **changes)  # type: ignore[arg-type]
