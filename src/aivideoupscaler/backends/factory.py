"""Backend selection logic.

The factory decides which super-resolution backend to use for a job.  It
prefers the fastest *usable* option and always keeps the dependency-free
classical backend as the last resort, so a job never fails purely because an
optional accelerator stack is missing.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import BackendName
from ..errors import BackendUnavailableError
from ..models.registry import ModelInfo, get_model
from .base import BackendStatus, UpscalerBackend
from .classical import ClassicalBackend
from .ncnn import NcnnBackend
from .torch_esrgan import TorchBackend


@dataclass
class BackendOptions:
    """Tuning knobs shared by every backend implementation."""

    tile: int = 0
    device: str = "auto"
    gpu_id: int = -1
    threads: int = 2
    half: bool = True


def build_backend(
    name: BackendName | str,
    model: ModelInfo | str,
    options: BackendOptions | None = None,
) -> UpscalerBackend:
    """Instantiate the requested backend.

    Raises :class:`BackendUnavailableError` when the requested backend cannot
    run in the current environment.
    """
    opts = options or BackendOptions()
    info = get_model(model) if isinstance(model, str) else model

    if isinstance(name, str):
        try:
            name = BackendName(name.lower())
        except ValueError as exc:
            valid = ", ".join(item.value for item in BackendName)
            raise BackendUnavailableError(
                f"Unknown backend '{name}'. Valid backends: {valid}"
            ) from exc

    if name == BackendName.CLASSICAL:
        return ClassicalBackend(info, tile=opts.tile, device=opts.device)
    if name == BackendName.NCNN:
        backend = NcnnBackend(
            info,
            tile=opts.tile,
            device=opts.device,
            gpu_id=opts.gpu_id,
            threads=opts.threads,
        )
        if not backend.is_available():
            raise BackendUnavailableError(backend.status().detail)
        return backend
    if name == BackendName.TORCH:
        backend = TorchBackend(
            info,
            tile=opts.tile,
            device=opts.device,
            half=opts.half,
        )
        if not backend.is_available():
            raise BackendUnavailableError(backend.status().detail)
        return backend

    raise BackendUnavailableError(f"Unsupported backend '{name}'")


def auto_backend(
    model: ModelInfo | str,
    options: BackendOptions | None = None,
    prefer: BackendName | None = None,
) -> tuple[UpscalerBackend, list[BackendStatus]]:
    """Pick the best available backend and report every backend's status.

    The returned list always contains the status of all backends, so the UI can
    explain what was chosen and why the alternatives were rejected.
    """
    info = get_model(model) if isinstance(model, str) else model
    opts = options or BackendOptions()

    order: list[BackendName] = [BackendName.NCNN, BackendName.TORCH, BackendName.CLASSICAL]
    if prefer is not None and prefer in order:
        order.remove(prefer)
        order.insert(0, prefer)

    statuses: list[BackendStatus] = []
    for candidate in order:
        try:
            backend = build_backend(candidate, info, opts)
        except BackendUnavailableError as exc:
            status = BackendStatus(
                name=candidate.value,
                available=False,
                device="cpu",
                detail=str(exc),
                scale=info.scale,
            )
            statuses.append(status)
            continue
        status = backend.status()
        statuses.append(status)
        if status.available:
            backend.close()
            return build_backend(candidate, info, opts), statuses

    raise BackendUnavailableError(
        "No usable super-resolution backend was found. "
        + " | ".join(f"{s.name}: {s.detail}" for s in statuses)
    )


def list_backends(model: ModelInfo | str = "realesr-animevideov3") -> list[BackendStatus]:
    """Return the status of every backend for the given model."""
    info = get_model(model) if isinstance(model, str) else model
    statuses: list[BackendStatus] = []
    for candidate in BackendName:
        try:
            backend = build_backend(candidate, info)
        except BackendUnavailableError as exc:
            statuses.append(
                BackendStatus(
                    name=candidate.value,
                    available=False,
                    device="cpu",
                    detail=str(exc),
                    scale=info.scale,
                )
            )
            continue
        statuses.append(backend.status())
        backend.close()
    return statuses
