"""Real-ESRGAN backend built on the official ``realesrgan-ncnn-vulkan`` binary.

The ncnn build is a self-contained executable that runs the trained networks
through Vulkan, which covers NVIDIA, AMD and Intel GPUs without pulling in a
PyTorch installation.  The binary processes one image at a time, so the
backend drives it from a worker pool.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..errors import BackendError, BackendUnavailableError, ModelError
from ..models.registry import ModelInfo
from ..paths import binaries_dir, ensure_dir, models_dir
from .base import BackendStatus, UpscalerBackend

#: Name of the compiled executable inside the release archive.
EXECUTABLE_NAME = "realesrgan-ncnn-vulkan"
#: Environment variable that overrides the binary location.
ENV_NCNN = "AIVU_NCNN_BIN"

_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def _executable_candidates() -> list[Path]:
    """Return every location the ncnn executable may live in."""
    names = [EXECUTABLE_NAME]
    if sys.platform == "win32":
        names.insert(0, f"{EXECUTABLE_NAME}.exe")

    candidates: list[Path] = []
    override = os.environ.get(ENV_NCNN)
    if override:
        candidates.append(Path(override))
    candidates.append(binaries_dir() / names[0])
    for name in names:
        candidates.append(Path(name))

    local = models_dir() / "ncnn"
    for name in names:
        candidates.append(local / name)
    return candidates


def find_ncnn_executable() -> Path | None:
    """Locate the ``realesrgan-ncnn-vulkan`` executable."""
    for candidate in _executable_candidates():
        if candidate.is_file():
            return candidate.resolve()

    found = shutil.which(EXECUTABLE_NAME)
    if found:
        return Path(found).resolve()
    return None


def find_ncnn_models_dir() -> Path | None:
    """Locate the directory holding the ``.param``/``.bin`` network files."""
    candidates = [
        models_dir() / "ncnn",
        binaries_dir(),
        models_dir(),
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.param")):
            return candidate


class NcnnBackend(UpscalerBackend):
    """Super-resolution via the ncnn/Vulkan Real-ESRGAN executable."""

    name = "ncnn"

    def __init__(
        self,
        model: ModelInfo,
        tile: int = 0,
        device: str = "auto",
        gpu_id: int = -1,
        threads: int = 2,
    ) -> None:
        super().__init__(model, tile=tile, device=device)
        self.gpu_id = gpu_id
        self.threads = max(1, threads)
        self._executable: Path | None = None
        self._models_dir: Path | None = None

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        """Return ``True`` when the executable and model files are present."""
        return find_ncnn_executable() is not None and find_ncnn_models_dir() is not None

    def status(self) -> BackendStatus:
        """Describe the backend and explain what is missing when unusable."""
        executable = find_ncnn_executable()
        models = find_ncnn_models_dir()
        available = executable is not None and models is not None

        if available:
            detail = f"{executable.name} using {models}"
            device = f"gpu:{self.gpu_id}" if self.gpu_id >= 0 else "vulkan"
        elif executable is None and models is None:
            detail = (
                "ncnn/Vulkan binary and model files are missing. "
                "Run 'aivu models install --ncnn' to download them."
            )
            device = "vulkan"
        elif executable is None:
            detail = "The realesrgan-ncnn-vulkan executable is missing."
            device = "vulkan"
        else:
            detail = "The ncnn model files (*.param/*.bin) are missing."
            device = "vulkan"

        return BackendStatus(
            name=self.name,
            available=available,
            device=device,
            detail=detail,
            scale=self.scale,
            requires_model=True,
            extra={"executable": str(executable or ""), "models": str(models or "")},
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def upscale(self, frame: np.ndarray) -> np.ndarray:
        """Upscale a single BGR frame."""
        return self.upscale_batch([frame])[0]

    def upscale_batch(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Upscale a batch of frames using a small thread pool."""
        if not frames:
            return []
        executable = self._require_executable()
        models_dir_path = self._require_models_dir()

        with tempfile.TemporaryDirectory(prefix="aivup-ncnn-") as tmp:
            work = Path(tmp)
            inputs: list[Path] = []
            for index, frame in enumerate(frames):
                path = work / f"in_{index:06d}.png"
                self._write_png(path, frame)
                inputs.append(path)

            outputs: list[Path] = [work / f"out_{index:06d}.png" for index in range(len(frames))]
            with ThreadPoolExecutor(max_workers=self.threads) as pool:
                list(
                    pool.map(
                        lambda pair: self._run_binary(
                            executable, models_dir_path, pair[0], pair[1]
                        ),
                        zip(inputs, outputs, strict=False),
                    )
                )

            results: list[np.ndarray] = []
            for path in outputs:
                if not path.is_file():
                    raise BackendError(
                        "The ncnn/Vulkan backend did not produce an output image. "
                        "The GPU may be unsupported or the driver may be missing."
                    )
                results.append(self._read_png(path))
            return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _require_executable(self) -> Path:
        """Return the executable path or raise a helpful error."""
        if self._executable is None:
            executable = find_ncnn_executable()
            if executable is None:
                raise BackendUnavailableError(
                    "The realesrgan-ncnn-vulkan binary was not found. "
                    "Run 'aivu models install --ncnn' to download it, or select "
                    "another backend with --backend."
                )
            self._executable = executable
        return self._executable

    def _require_models_dir(self) -> Path:
        """Return the ncnn models directory or raise a helpful error."""
        if self._models_dir is None:
            models = find_ncnn_models_dir()
            if models is None:
                raise ModelError(
                    "The ncnn network files (*.param / *.bin) were not found. "
                    "Run 'aivu models install --ncnn' to download them."
                )
            self._models_dir = models
        return self._models_dir

    def _run_binary(self, executable: Path, models: Path, src: Path, dst: Path) -> None:
        """Invoke the ncnn executable for a single image."""
        model_name = self.model.ncnn_name or self.model.key
        cmd = [
            str(executable),
            "-i",
            str(src),
            "-o",
            str(dst),
            "-n",
            model_name,
            "-s",
            str(self.scale),
            "-f",
            "png",
        ]
        if self.tile > 0:
            cmd += ["-t", str(self.tile)]
        if self.gpu_id >= 0:
            cmd += ["-g", str(self.gpu_id)]
        cmd += ["-m", str(models)]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=_CREATE_NO_WINDOW,
            )
        except OSError as exc:
            raise BackendError(f"Could not run the ncnn/Vulkan backend: {exc}") from exc

        if result.returncode != 0:
            message = (result.stderr or result.stdout or "").strip()
            raise BackendError(
                f"The ncnn/Vulkan backend failed (exit code {result.returncode}). "
                f"This usually means the GPU driver is missing or Vulkan is unsupported.\n{message}"
            )

    @staticmethod
    def _write_png(path: Path, frame: np.ndarray) -> None:
        """Write a BGR frame to disk as a PNG image."""
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover
            raise BackendError("Pillow is required to bridge frames to the ncnn binary.") from exc
        Image.fromarray(np.ascontiguousarray(frame[:, :, ::-1])).save(path, format="PNG")

    @staticmethod
    def _read_png(path: Path) -> np.ndarray:
        """Read a PNG image back into a BGR frame."""
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover
            raise BackendError("Pillow is required to bridge frames from the ncnn binary.") from exc
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return np.ascontiguousarray(rgb[:, :, ::-1])


def install_ncnn_bundle(archive: Path, target: Path | None = None) -> Path:
    """Extract the ncnn release archive and return the executable path.

    The archive contains the executable plus the ``.param``/``.bin`` pairs for
    every supported network.  They are split into ``bin/`` and ``models/ncnn``
    so the application can find them independently.
    """
    if not archive.is_file():
        raise BackendUnavailableError(f"ncnn bundle not found: {archive}")

    bin_dir = ensure_dir(target or binaries_dir())
    models_target = ensure_dir(models_dir() / "ncnn")
    names = [EXECUTABLE_NAME]
    if sys.platform == "win32":
        names.insert(0, f"{EXECUTABLE_NAME}.exe")

    extracted: list[str] = []
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            name = Path(member).name
            if not name or name.endswith("/"):
                continue
            if name.endswith((".param", ".bin")):
                with bundle.open(member) as source, (models_target / name).open("wb") as dest:
                    dest.write(source.read())
            elif name in names or name.endswith(".exe"):
                with bundle.open(member) as source, (bin_dir / name).open("wb") as dest:
                    dest.write(source.read())
                extracted.append(name)
        if not extracted:
            raise BackendUnavailableError(
                f"The ncnn archive {archive.name} contains no executable."
            )

    for name in names:
        candidate = bin_dir / name
        if candidate.is_file():
            if sys.platform != "win32":
                candidate.chmod(0o755)
            return candidate
    raise BackendUnavailableError(
        f"The ncnn archive {archive.name} did not contain the expected executable."
    )
