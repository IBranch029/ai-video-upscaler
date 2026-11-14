"""Tests for the configuration objects and the validation rules."""

from __future__ import annotations

from pathlib import Path

import pytest

from aivideoupscaler.config import (
    BackendName,
    Codec,
    Container,
    UpscaleConfig,
)
from aivideoupscaler.errors import ConfigurationError


def test_defaults_are_sane() -> None:
    """The default configuration uses a video-friendly model and codec."""
    config = UpscaleConfig(input_path="a.mp4", output_path="b.mkv")
    assert config.model == "realesr-animevideov3"
    assert config.batch_size == 4
    assert config.workers == 1
    assert config.encoder.codec == Codec.H264
    assert config.encoder.container == Container.MKV


def test_paths_are_converted_to_path_objects() -> None:
    """String paths are normalised into :class:`pathlib.Path` instances."""
    config = UpscaleConfig(input_path="in.mp4", output_path="out.mkv")
    assert isinstance(config.input_path, Path)
    assert isinstance(config.output_path, Path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"workers": 0},
        {"scale": 0},
        {"target_width": 1},
        {"target_height": 1},
    ],
)
def test_invalid_values_are_rejected(kwargs: dict[str, object]) -> None:
    """Nonsensical tuning values raise :class:`ConfigurationError`."""
    with pytest.raises(ConfigurationError):
        UpscaleConfig(input_path="a.mp4", output_path="b.mkv", **kwargs)


def test_replace_returns_a_copy() -> None:
    """:meth:`UpscaleConfig.replace` does not mutate the original."""
    config = UpscaleConfig(input_path="a.mp4", output_path="b.mkv")
    other = config.replace(model="RealESRGAN_x4plus")
    assert other.model == "RealESRGAN_x4plus"
    assert config.model == "realesr-animevideov3"
    assert other.input_path == config.input_path


def test_backend_names_are_stable() -> None:
    """The backend enum values are part of the public API."""
    assert BackendName.NCNN.value == "ncnn"
    assert BackendName.TORCH.value == "torch"
    assert BackendName.CLASSICAL.value == "classical"
