"""Download the Real-ESRGAN weights used by the upscaler.

The script is a thin, dependency-free wrapper around
:mod:`aivideoupscaler.models.downloader` so the weights can be fetched
without installing the whole package::

    python models/download_models.py                    # every model
    python models/download_models.py realesr-animevideov3
    python models/download_models.py --ncnn             # plus the ncnn bundle
    python models/download_models.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aivideoupscaler.errors import UpscalerError  # noqa: E402
from aivideoupscaler.models.downloader import (  # noqa: E402
    download_model,
    download_ncnn_bundle,
    is_model_present,
)
from aivideoupscaler.models.registry import MODELS, list_models  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Download the requested model weights."""
    parser = argparse.ArgumentParser(
        description="Download the Real-ESRGAN weights used by the 4K video upscaler.",
    )
    parser.add_argument(
        "models",
        nargs="*",
        help="Model keys to download. Omit to download every model.",
    )
    parser.add_argument(
        "--ncnn",
        action="store_true",
        help="Also download the ncnn/Vulkan binary and its network files.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download existing files.")
    parser.add_argument("--list", action="store_true", help="List the models and exit.")
    args = parser.parse_args(argv)

    if args.list:
        for model in list_models():
            state = "installed" if is_model_present(model.key) else "not installed"
            print(f"{model.key:32} x{model.scale}  {state:14} {model.description}")
        return 0

    selected = args.models or list(MODELS)
    unknown = [key for key in selected if key not in MODELS]
    if unknown:
        parser.error(f"unknown model(s): {', '.join(unknown)}")

    failures = 0
    for key in selected:
        print(f"Downloading {MODELS[key].display_name} ...", flush=True)
        try:
            path = download_model(key, force=args.force)
        except UpscalerError as exc:
            print(f"  failed: {exc}", file=sys.stderr)
            failures += 1
            continue
        print(f"  saved to {path}")

    if args.ncnn:
        print("Downloading the ncnn/Vulkan bundle ...", flush=True)
        try:
            archive = download_ncnn_bundle(force=args.force)
            from aivideoupscaler.backends.ncnn import install_ncnn_bundle

            executable = install_ncnn_bundle(archive)
        except UpscalerError as exc:
            print(f"  failed: {exc}", file=sys.stderr)
            failures += 1
        else:
            print(f"  installed the ncnn/Vulkan backend at {executable}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
