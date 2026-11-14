"""Download and verify model weights.

The downloader is atomic, verifies SHA256 digests when they are known and
falls back to a size sanity check otherwise.  It never overwrites a file whose
digest does not match.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path

from ..errors import ModelDownloadError, ModelError
from ..paths import ensure_dir, models_dir
from .registry import MODELS, get_model

ProgressFn = Callable[[str, int, int], None]

_USER_AGENT = "aivideoupscaler/1.0 (+https://github.com/IBranch029/ai-video-upscaler)"
_CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    """Compute the SHA256 digest of ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path, progress: ProgressFn | None) -> Path:
    """Download ``url`` to ``destination`` atomically."""
    ensure_dir(destination.parent)
    tmp_dir = ensure_dir(Path(tempfile.gettempdir()) / "aivideoupscaler-dl")

    # Reserve a unique name, then release it immediately so that the file can
    # be written (and later removed) on every platform, including Windows.
    handle, tmp_name = tempfile.mkstemp(dir=tmp_dir, suffix=".part")
    os.close(handle)
    tmp_path = Path(tmp_name)
    tmp_path.unlink(missing_ok=True)

    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            with tmp_path.open("wb") as out:
                while True:
                    block = response.read(_CHUNK)
                    if not block:
                        break
                    out.write(block)
                    done += len(block)
                    if progress:
                        progress("download", done, total)
    except urllib.error.HTTPError as exc:
        tmp_path.unlink(missing_ok=True)
        raise ModelDownloadError(f"HTTP {exc.code} while downloading {url}") from exc
    except urllib.error.URLError as exc:
        tmp_path.unlink(missing_ok=True)
        raise ModelDownloadError(f"Network error while downloading {url}: {exc.reason}") from exc
    except OSError as exc:
        tmp_path.unlink(missing_ok=True)
        raise ModelDownloadError(f"Could not write {destination}: {exc}") from exc

    try:
        shutil.move(str(tmp_path), str(destination))
    except OSError as exc:
        tmp_path.unlink(missing_ok=True)
        raise ModelDownloadError(f"Could not move download into place: {exc}") from exc
    return destination


def verify(path: Path, expected_sha256: str = "", min_size: int = 1024) -> bool:
    """Return ``True`` when ``path`` looks like a valid, complete download."""
    if not path.is_file():
        return False
    if path.stat().st_size < min_size:
        return False
    if expected_sha256:
        return sha256_file(path).lower() == expected_sha256.lower()
    return True


def model_path(key: str, directory: Path | None = None) -> Path:
    """Return the expected on-disk location of a model's PyTorch weights."""
    info = get_model(key)
    base = ensure_dir(directory or models_dir())
    return base / info.torch_filename


def is_model_present(key: str, directory: Path | None = None) -> bool:
    """Return ``True`` when the weights for ``key`` are already on disk."""
    try:
        info = get_model(key)
    except ModelError:
        return False
    return verify(model_path(key, directory), info.torch_sha256)


def download_model(
    key: str,
    directory: Path | None = None,
    force: bool = False,
    progress: ProgressFn | None = None,
) -> Path:
    """Download the PyTorch weights for ``key`` and return their path."""
    info = get_model(key)
    if not info.torch_url:
        raise ModelDownloadError(f"Model '{key}' has no PyTorch weights available")
    base = ensure_dir(directory or models_dir())
    target = base / info.torch_filename

    if not force and verify(target, info.torch_sha256):
        return target

    if progress:
        progress(f"Downloading {info.display_name}", 0, 0)
    _download(info.torch_url, target, progress)

    if not verify(target, info.torch_sha256):
        target.unlink(missing_ok=True)
        raise ModelDownloadError(
            f"Checksum verification failed for {info.display_name}; the download was discarded."
        )
    return target


def download_models(
    keys: Iterable[str] | None = None,
    directory: Path | None = None,
    force: bool = False,
    progress: ProgressFn | None = None,
) -> dict[str, Path]:
    """Download several models and return a mapping of key to local path."""
    selected = list(keys) if keys else list(MODELS)
    result: dict[str, Path] = {}
    for key in selected:
        result[key] = download_model(key, directory=directory, force=force, progress=progress)
    return result


def download_ncnn_bundle(
    directory: Path | None = None,
    force: bool = False,
    progress: ProgressFn | None = None,
) -> Path:
    """Download the ncnn/Vulkan release bundle with the compiled binary.

    The archive contains ``realesrgan-ncnn-vulkan`` plus the ``.param``/``.bin``
    pairs for every supported network, so a single download enables the whole
    ncnn backend.
    """
    from .registry import NCNN_BUNDLE_FILENAME, NCNN_BUNDLE_URL

    base = ensure_dir(directory or models_dir())
    target = base / NCNN_BUNDLE_FILENAME
    if not force and target.is_file() and target.stat().st_size > 1_000_000:
        return target

    if progress:
        progress("Downloading Real-ESRGAN ncnn/Vulkan bundle", 0, 0)
    _download(NCNN_BUNDLE_URL, target, progress)
    if target.stat().st_size < 1_000_000:
        target.unlink(missing_ok=True)
        raise ModelDownloadError("The ncnn/Vulkan bundle download looks truncated.")
    return target
