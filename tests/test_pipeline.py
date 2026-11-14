"""Tests for the pipeline's geometry helpers."""

from __future__ import annotations

import pytest

from aivideoupscaler.pipeline import compute_target, even


@pytest.mark.parametrize(
    ("value", "expected"),
    [(3840, 3840), (3841, 3840), (1920, 1920), (3, 2), (1, 2)],
)
def test_even_rounds_down(value: int, expected: int) -> None:
    """Dimensions are rounded down to even numbers for yuv420p."""
    assert even(value) == expected


def test_default_target_is_4k_with_preserved_aspect() -> None:
    """A 16:9 source is scaled to a 3840x2160 target by default."""
    width, height = compute_target(1920, 1080)
    assert (width, height) == (3840, 2160)


def test_explicit_dimensions_win() -> None:
    """Explicit width and height are honoured exactly."""
    assert compute_target(1920, 1080, 1280, 720) == (1280, 720)


def test_explicit_height_preserves_aspect_ratio() -> None:
    """Setting only the height keeps the source aspect ratio."""
    width, height = compute_target(1000, 500, None, 1000)
    assert height == 1000
    assert width == 2000


def test_scale_overrides_the_default_height() -> None:
    """An explicit scale factor wins over the 4K default."""
    width, height = compute_target(320, 180, None, None, 2)
    assert (width, height) == (640, 360)


def test_odd_source_dimensions_produce_even_output() -> None:
    """The output dimensions are always even, even for odd sources."""
    width, height = compute_target(321, 181)
    assert width % 2 == 0
    assert height % 2 == 0
