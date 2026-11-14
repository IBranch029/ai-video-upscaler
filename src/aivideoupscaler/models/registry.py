"""Catalogue of downloadable Real-ESRGAN model weights.

Each entry describes both the PyTorch weights (``.pth``) and, where available,
the matching ncnn/Vulkan parameter+weight pair so the two backends can share a
single user-facing model name.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import ModelError


@dataclass(frozen=True)
class ModelInfo:
    """Metadata for a downloadable model."""

    key: str
    display_name: str
    scale: int
    description: str
    torch_url: str = ""
    torch_filename: str = ""
    torch_sha256: str = ""
    ncnn_name: str = ""
    """Name of the ``.param``/``.bin`` pair shipped inside the ncnn release."""
    ncnn_url: str = ""
    ncnn_filename: str = ""
    ncnn_sha256: str = ""
    preferred_for_video: bool = False

    def __post_init__(self) -> None:
        if not self.torch_url and not self.ncnn_url:
            raise ModelError(f"Model '{self.key}' has no download source")


_REALESRGAN_RELEASE = "https://github.com/xinntao/Real-ESRGAN/releases/download"
_NCNN_RELEASE = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0"

#: The ncnn/Vulkan bundle ships the compiled executable *and* the parameter
#: files for every supported network, so a single archive serves all models.
NCNN_BUNDLE_URL = f"{_NCNN_RELEASE}/realesrgan-ncnn-vulkan-20220424-windows.zip"
NCNN_BUNDLE_URL_LINUX = f"{_NCNN_RELEASE}/realesrgan-ncnn-vulkan-20220424-ubuntu.zip"
NCNN_BUNDLE_FILENAME = "realesrgan-ncnn-vulkan-20220424.zip"


MODELS: dict[str, ModelInfo] = {
    "realesr-animevideov3": ModelInfo(
        key="realesr-animevideov3",
        display_name="RealESRGAN AnimeVideo v3 (x4)",
        scale=4,
        description=(
            "Lightweight x4 network tuned for anime and video content. "
            "Fastest option and the recommended default for animated sources."
        ),
        torch_url=f"{_REALESRGAN_RELEASE}/v0.2.5.0/realesr-animevideov3.pth",
        torch_filename="realesr-animevideov3.pth",
        torch_sha256="",
        ncnn_name="realesr-animevideov3-x4",
        ncnn_url=f"{_NCNN_RELEASE}/realesrgan-ncnn-vulkan-20220424-windows.zip",
        ncnn_filename="realesrgan-ncnn-vulkan-20220424.zip",
        preferred_for_video=True,
    ),
    "RealESRGAN_x4plus": ModelInfo(
        key="RealESRGAN_x4plus",
        display_name="RealESRGAN x4plus (x4)",
        scale=4,
        description=(
            "The general purpose x4 network. Highest quality on photographic and "
            "live-action footage, noticeably slower than the anime video model."
        ),
        torch_url=f"{_REALESRGAN_RELEASE}/v0.1.0/RealESRGAN_x4plus.pth",
        torch_filename="RealESRGAN_x4plus.pth",
        torch_sha256="",
        ncnn_name="RealESRGAN_x4plus",
        ncnn_url=f"{_NCNN_RELEASE}/realesrgan-ncnn-vulkan-20220424-windows.zip",
        ncnn_filename="realesrgan-ncnn-vulkan-20220424.zip",
    ),
    "RealESRGAN_x4plus_anime_6B": ModelInfo(
        key="RealESRGAN_x4plus_anime_6B",
        display_name="RealESRGAN x4plus Anime 6B (x4)",
        scale=4,
        description="Compact x4 anime network with a shallower RRDB trunk.",
        torch_url=f"{_REALESRGAN_RELEASE}/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        torch_filename="RealESRGAN_x4plus_anime_6B.pth",
        torch_sha256="",
        ncnn_name="RealESRGAN_x4plus_anime_6B",
        ncnn_url=f"{_NCNN_RELEASE}/realesrgan-ncnn-vulkan-20220424-windows.zip",
        ncnn_filename="realesrgan-ncnn-vulkan-20220424.zip",
    ),
    "RealESRGAN_x2plus": ModelInfo(
        key="RealESRGAN_x2plus",
        display_name="RealESRGAN x2plus (x2)",
        scale=2,
        description="Milder x2 network, useful when 4x would overshoot the target.",
        torch_url=f"{_REALESRGAN_RELEASE}/v0.2.1/RealESRGAN_x2plus.pth",
        torch_filename="RealESRGAN_x2plus.pth",
        torch_sha256="",
        ncnn_name="RealESRGAN_x2plus",
        ncnn_url=f"{_NCNN_RELEASE}/realesrgan-ncnn-vulkan-20220424-windows.zip",
        ncnn_filename="realesrgan-ncnn-vulkan-20220424.zip",
    ),
    "realesr-general-x4v3": ModelInfo(
        key="realesr-general-x4v3",
        display_name="RealESRGAN General x4 v3 (x4)",
        scale=4,
        description="General purpose x4 network with improved natural-image fidelity.",
        torch_url=f"{_REALESRGAN_RELEASE}/v0.2.5.0/realesr-general-x4v3.pth",
        torch_filename="realesr-general-x4v3.pth",
        torch_sha256="",
        ncnn_name="realesr-general-x4v3",
        ncnn_url=f"{_NCNN_RELEASE}/realesrgan-ncnn-vulkan-20220424-windows.zip",
        ncnn_filename="realesrgan-ncnn-vulkan-20220424.zip",
    ),
}

DEFAULT_MODEL = "realesr-animevideov3"


def get_model(key: str) -> ModelInfo:
    """Return the :class:`ModelInfo` for ``key``."""
    try:
        return MODELS[key]
    except KeyError:
        available = ", ".join(sorted(MODELS))
        raise ModelError(f"Unknown model '{key}'. Available models: {available}") from None


def list_models() -> list[ModelInfo]:
    """Return all known models, video-preferred ones first."""
    return sorted(MODELS.values(), key=lambda m: (not m.preferred_for_video, m.key))
