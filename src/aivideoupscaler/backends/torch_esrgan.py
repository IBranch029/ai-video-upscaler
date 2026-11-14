"""Real-ESRGAN backend built on PyTorch.

Supports CUDA, DirectML, Apple MPS and plain CPU, and always falls back to the
CPU when no accelerator is usable.  Inference is tiled, so arbitrarily large
frames can be processed without exhausting VRAM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..errors import BackendError, BackendUnavailableError, ModelError
from ..models.registry import ModelInfo
from .base import BackendStatus, UpscalerBackend

#: Number of residual blocks used by each released network.
_BLOCKS = {
    "realesr-animevideov3": 6,
    "RealESRGAN_x4plus": 23,
    "RealESRGAN_x4plus_anime_6B": 6,
    "RealESRGAN_x2plus": 23,
    "realesr-general-x4v3": 23,
    "RealESRNet_x4plus": 23,
}


def torch_available() -> bool:
    """Return ``True`` when PyTorch can be imported."""
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def detect_device(preferred: str = "auto") -> str:
    """Resolve the best available torch device string."""
    try:
        import torch
    except Exception:
        return "cpu"

    if preferred not in ("", "auto"):
        candidate = preferred.lower()
        if candidate.startswith("cuda") and torch.cuda.is_available():
            return candidate
        if (
            candidate == "mps"
            and getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        ):
            return "mps"
        if candidate == "dml":
            try:  # pragma: no cover - Windows specific
                import torch_directml  # type: ignore

                return torch_directml.device()
            except Exception:
                return "cpu"
        return "cpu"

    try:  # pragma: no cover - Windows specific
        import torch_directml  # type: ignore

        return torch_directml.device()
    except Exception:
        pass

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TorchBackend(UpscalerBackend):
    """Super-resolution using the official Real-ESRGAN PyTorch weights."""

    name = "torch"

    def __init__(
        self,
        model: ModelInfo,
        tile: int = 0,
        device: str = "auto",
        weights: Path | None = None,
        half: bool = True,
    ) -> None:
        super().__init__(model, tile=tile, device=device)
        self.weights = weights
        self.half = half
        self._net: Any = None
        self._torch: Any = None
        self._resolved_device = ""

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        """Return ``True`` when PyTorch is importable."""
        return torch_available()

    def status(self) -> BackendStatus:
        """Describe the backend."""
        if not self.is_available():
            return BackendStatus(
                name=self.name,
                available=False,
                device="cpu",
                detail=(
                    "PyTorch is not installed. Install it with "
                    "'pip install torch' or use the ncnn backend instead."
                ),
                scale=self.scale,
            )

        device = detect_device(self.device)
        kind = "gpu" if device != "cpu" else "cpu"
        return BackendStatus(
            name=self.name,
            available=True,
            device=device,
            detail=f"PyTorch inference on {kind} (device={device})",
            scale=self.scale,
            requires_model=True,
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load(self) -> None:
        """Load the network weights, raising a clear error on failure."""
        if self._net is not None:
            return
        if not torch_available():
            raise BackendUnavailableError(
                "PyTorch is not installed. Install it with 'pip install torch' "
                "or select the ncnn backend with --backend ncnn."
            )

        import torch

        from ..models.downloader import model_path
        from .networks import build_network, unwrap_state_dict

        weights = self.weights or model_path(self.model.key)
        if not Path(weights).is_file():
            raise ModelError(
                f"The weights for '{self.model.key}' were not found at {weights}. "
                "Run 'aivu models install' to download them."
            )

        try:
            raw = torch.load(str(weights), map_location="cpu", weights_only=False)
        except TypeError:  # pragma: no cover - older torch without weights_only
            raw = torch.load(str(weights), map_location="cpu")
        except Exception as exc:
            raise ModelError(f"Could not read the model weights {weights}: {exc}") from exc

        try:
            state = unwrap_state_dict(raw)
            net, detected_scale = build_network(state)
        except Exception as exc:
            raise ModelError(
                f"The weights in {weights} do not match a known Real-ESRGAN architecture: {exc}"
            ) from exc

        # The checkpoint is authoritative about the scale factor.
        self._scale = detected_scale

        device = detect_device(self.device)
        self._resolved_device = device
        net.eval()
        net = net.to(device)
        if self.half and device == "cuda":
            net = net.half()

        self._net = net
        self._torch = torch

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def upscale(self, frame: np.ndarray) -> np.ndarray:
        """Upscale a single BGR frame."""
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise BackendError(f"Expected a BGR frame, got shape {frame.shape}")
        self._load()
        torch = self._torch
        net = self._net

        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        tensor = tensor.to(self._resolved_device)
        if self.half and self._resolved_device == "cuda":
            tensor = tensor.half()

        with torch.no_grad():
            output = self._infer_tiled(net, tensor)

        output = output.squeeze(0).permute(1, 2, 0).float().clamp_(0, 1).cpu().numpy()
        result = (output[:, :, ::-1] * 255.0).round().astype(np.uint8)
        return np.ascontiguousarray(result)

    def upscale_batch(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Upscale a batch of frames.

        Frames are processed one by one to keep peak VRAM usage bounded by a
        single frame, which is what prevents out-of-memory failures.
        """
        return [self.upscale(frame) for frame in frames]

    def _infer_tiled(self, net, tensor):
        """Run the network, splitting the input into tiles when necessary."""
        torch = self._torch
        _, _, height, width = tensor.shape
        tile = self.tile if self.tile > 0 else 0

        if tile <= 0 or (height <= tile and width <= tile):
            return net(tensor)

        scale = self.scale
        overlap = 16
        outputs = []
        for top in range(0, height, tile):
            for left in range(0, width, tile):
                bottom = min(top + tile, height)
                right = min(left + tile, width)
                patch = tensor[:, :, top:bottom, left:right]
                with torch.no_grad():
                    result = net(patch)

                # Blend the border pixels with the neighbouring tiles.
                if overlap > 0:
                    if top > 0:
                        result[:, :, :overlap, :] = result[:, :, :overlap, :] * 0.5
                    if left > 0:
                        result[:, :, :, :overlap] = result[:, :, :, :overlap] * 0.5
                outputs.append((top, left, bottom, right, result))

        canvas = torch.zeros(
            (1, 3, height * scale, width * scale),
            dtype=outputs[0][4].dtype,
            device=outputs[0][4].device,
        )
        for top, left, bottom, right, result in outputs:
            canvas[
                :,
                :,
                top * scale : bottom * scale,
                left * scale : right * scale,
            ] = result
        return canvas

    def close(self) -> None:
        """Free the network and the CUDA cache."""
        self._net = None
        self._torch = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
