"""Tests for the super-resolution backends."""

from __future__ import annotations

import numpy as np
import pytest

from aivideoupscaler.backends.base import to_bgr_float, to_bgr_uint8
from aivideoupscaler.backends.classical import ClassicalBackend
from aivideoupscaler.errors import BackendError
from aivideoupscaler.models.registry import get_model


@pytest.fixture()
def frame() -> np.ndarray:
    """Return a small deterministic BGR frame."""
    rng = np.random.default_rng(1234)
    return rng.integers(0, 256, size=(16, 24, 3), dtype=np.uint8)


def test_classical_backend_is_always_available() -> None:
    """The classical backend only needs NumPy and Pillow."""
    backend = ClassicalBackend(get_model("realesr-animevideov3"))
    assert backend.is_available()
    assert backend.status().available


def test_classical_backend_upscales_by_the_model_scale(frame: np.ndarray) -> None:
    """The output dimensions are the input multiplied by the model scale."""
    backend = ClassicalBackend(get_model("realesr-animevideov3"))
    result = backend.upscale(frame)
    assert result.shape == (frame.shape[0] * 4, frame.shape[1] * 4, 3)
    assert result.dtype == np.uint8


def test_classical_backend_preserves_the_frame_order(frame: np.ndarray) -> None:
    """Batch processing returns the frames in the order they were given."""
    backend = ClassicalBackend(get_model("realesr-animevideov3"))
    frames = [frame, np.zeros_like(frame), np.full_like(frame, 255)]
    results = backend.upscale_batch(frames)
    assert len(results) == 3
    assert results[1].max() <= 1
    assert results[2].min() >= 254


def test_classical_backend_rejects_bad_frames() -> None:
    """A frame with the wrong shape raises a backend error."""
    backend = ClassicalBackend(get_model("realesr-animevideov3"))
    with pytest.raises(BackendError):
        backend.upscale(np.zeros((10, 10), dtype=np.uint8))


def test_classical_backend_sharpens_flat_images() -> None:
    """The unsharp mask increases local contrast on a step edge."""
    backend = ClassicalBackend(get_model("realesr-animevideov3"))
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:, 8:] = 200
    result = backend.upscale(image)
    assert result.std() > image.std()


def test_float_conversion_round_trip(frame: np.ndarray) -> None:
    """Converting to float and back preserves the pixel values."""
    as_float = to_bgr_float(frame)
    assert as_float.dtype == np.float32
    assert as_float.min() >= 0.0 and as_float.max() <= 1.0
    restored = to_bgr_uint8(as_float)
    assert np.array_equal(restored, frame)
