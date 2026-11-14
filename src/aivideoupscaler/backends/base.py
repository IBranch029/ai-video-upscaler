"""Common interface shared by every super-resolution backend.

A backend receives ``numpy`` frames in **BGR** order (the layout produced by
FFmpeg's ``rawvideo``/``bgr24`` decoder) and returns upscaled frames in the
same layout.  Keeping BGR everywhere avoids a needless colour-space round trip
between the decoder, the inference engine and the encoder.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..models.registry import ModelInfo


@dataclass
class BackendStatus:
    """Runtime information about a backend instance."""

    name: str
    available: bool
    device: str = "cpu"
    detail: str = ""
    scale: int = 1
    requires_model: bool = True
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """Human readable one-line description."""
        state = "available" if self.available else "unavailable"
        return f"{self.name} ({self.device}) - {state}: {self.detail}"


class UpscalerBackend(ABC):
    """Base class for all super-resolution backends."""

    #: Stable identifier used by the CLI and configuration files.
    name: str = "base"

    def __init__(self, model: ModelInfo, tile: int = 0, device: str = "auto") -> None:
        self.model = model
        self.tile = tile
        self.device = device
        self._scale = model.scale

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def scale(self) -> int:
        """Native scale factor of the loaded network."""
        return self._scale

    @abstractmethod
    def status(self) -> BackendStatus:
        """Return the current status of this backend."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` when the backend can run in this environment."""

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @abstractmethod
    def upscale(self, frame: np.ndarray) -> np.ndarray:
        """Upscale a single BGR frame and return the upscaled BGR frame."""

    def upscale_batch(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Upscale a batch of BGR frames.

        The default implementation simply loops; backends with native batch
        support override this to keep VRAM usage predictable.
        """
        return [self.upscale(frame) for frame in frames]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:  # noqa: B027
        """Release any resources held by the backend."""

    def __enter__(self) -> UpscalerBackend:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def to_bgr_float(frame: np.ndarray) -> np.ndarray:
    """Convert a ``uint8`` BGR frame to a contiguous ``float32`` array in [0, 1]."""
    return np.ascontiguousarray(frame, dtype=np.float32) / 255.0


def to_bgr_uint8(array: np.ndarray) -> np.ndarray:
    """Convert a ``float32`` array in [0, 1] back to a ``uint8`` BGR frame."""
    return np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
