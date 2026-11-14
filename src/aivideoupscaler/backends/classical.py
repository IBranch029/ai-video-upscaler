"""Dependency-free super-resolution backend.

This backend uses a high-quality Lanczos resampling pass followed by an
edge-aware unsharp mask.  It requires nothing beyond NumPy and Pillow, which
makes it the guaranteed fallback: the application can always produce a valid
4K file even when no neural network or GPU is available.
"""

from __future__ import annotations

import numpy as np

from ..errors import BackendError
from ..models.registry import ModelInfo
from .base import BackendStatus, UpscalerBackend, lanczos_filter, to_bgr_uint8


class ClassicalBackend(UpscalerBackend):
    """Lanczos + adaptive unsharp mask upscaler."""

    name = "classical"

    def __init__(
        self,
        model: ModelInfo,
        tile: int = 0,
        device: str = "cpu",
        sharpness: float = 0.6,
        radius: float = 1.1,
    ) -> None:
        super().__init__(model, tile=tile, device=device)
        self.sharpness = sharpness
        self.radius = radius

    def is_available(self) -> bool:
        """Always available: the backend only needs NumPy and Pillow."""
        try:
            import PIL  # noqa: F401
        except ImportError:
            return False
        return True

    def status(self) -> BackendStatus:
        """Describe the backend."""
        available = self.is_available()
        return BackendStatus(
            name=self.name,
            available=available,
            device="cpu",
            detail=(
                "Lanczos resampling with adaptive unsharp masking "
                "(no neural network, no GPU required)"
                if available
                else "Pillow is not installed"
            ),
            scale=self.scale,
            requires_model=False,
        )

    def upscale(self, frame: np.ndarray) -> np.ndarray:
        """Upscale one BGR frame."""
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise BackendError(f"Expected a BGR frame, got shape {frame.shape}")
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - Pillow is a hard dependency
            raise BackendError(
                "Pillow is required for the classical backend. Install it with 'pip install Pillow'."
            ) from exc

        source = Image.fromarray(np.ascontiguousarray(frame[:, :, ::-1]))
        width, height = source.size
        target_size = (width * self.scale, height * self.scale)
        resized = source.resize(target_size, lanczos_filter())
        array = np.asarray(resized, dtype=np.float32)
        array = self._unsharp(array)
        return np.ascontiguousarray(to_bgr_uint8(array[:, :, ::-1]))

    def upscale_batch(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Upscale a batch of frames."""
        return [self.upscale(frame) for frame in frames]

    def _unsharp(self, array: np.ndarray) -> np.ndarray:
        """Apply an edge-aware unsharp mask to a float RGB array."""
        if self.sharpness <= 0:
            return array
        blurred = self._box_blur(array, self.radius)
        detail = array - blurred
        # Protect flat regions from ringing by damping the mask where the
        # local gradient is weak.
        gradient = np.abs(detail).mean(axis=2)
        weight = np.clip(gradient / 12.0, 0.0, 1.0)[..., None]
        return array + detail * (self.sharpness * weight)

    @staticmethod
    def _box_blur(array: np.ndarray, radius: float) -> np.ndarray:
        """Approximate a Gaussian blur with three box passes."""
        radius = max(1, round(radius))
        result = array
        for _ in range(3):
            result = ClassicalBackend._box_blur_axis(result, radius, axis=0)
            result = ClassicalBackend._box_blur_axis(result, radius, axis=1)
        return result

    @staticmethod
    def _box_blur_axis(array: np.ndarray, radius: int, axis: int) -> np.ndarray:
        """Apply a single separable box blur along ``axis``."""
        pad = [(0, 0)] * 3
        pad[axis] = (radius, radius)
        padded = np.pad(array, pad, mode="edge")
        cumulative = np.cumsum(padded, axis=axis, dtype=np.float32)
        zero = np.zeros_like(np.take(cumulative, [0], axis=axis))
        cumulative = np.concatenate([zero, cumulative], axis=axis)
        length = array.shape[axis]
        upper = np.take(cumulative, np.arange(2 * radius + 1, 2 * radius + 1 + length), axis=axis)
        lower = np.take(cumulative, np.arange(0, length), axis=axis)
        return (upper - lower) / float(2 * radius + 1)
