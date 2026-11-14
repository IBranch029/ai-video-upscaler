"""Validate every local Real-ESRGAN checkpoint against the bundled networks.

The script loads each checkpoint, builds the matching architecture and runs a
real inference pass, so a mismatch between weights and code is caught
immediately.  It is used by the test suite and as a manual smoke test.

Usage::

    python scripts/validate_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aivideoupscaler.backends.networks import build_network, unwrap_state_dict
from aivideoupscaler.models.registry import MODELS


def main() -> int:
    """Validate every checkpoint found in the models directory."""
    import torch

    root = Path(__file__).resolve().parents[1]
    models_dir = root / "models"

    failures = 0
    for key in MODELS:
        path = models_dir / MODELS[key].torch_filename
        if not path.is_file():
            print(f"[SKIP] {key}: {path.name} not present")
            continue

        try:
            raw = torch.load(str(path), map_location="cpu", weights_only=False)
            state = unwrap_state_dict(raw)
            net, scale = build_network(state)
            net.eval()

            sample = torch.rand(1, 3, 16, 16)
            with torch.no_grad():
                output = net(sample)

            expected = 16 * scale
            if output.shape[-1] != expected or output.shape[-2] != expected:
                print(
                    f"[FAIL] {key}: expected {expected}x{expected}, got "
                    f"{output.shape[-2]}x{output.shape[-1]}"
                )
                failures += 1
                continue

            params = sum(p.numel() for p in net.parameters())
            print(
                f"[ OK ] {key}: {type(net).__name__} scale={scale} "
                f"params={params:,} out={tuple(output.shape)}"
            )
        except Exception as exc:
            print(f"[FAIL] {key}: {type(exc).__name__}: {exc}")
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
