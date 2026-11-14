"""Inspect Real-ESRGAN checkpoints and report their architecture layout.

Usage::

    python scripts/inspect_models.py models/*.pth
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import torch


def inner_state(obj: object) -> dict:
    """Return the raw parameter dict of a checkpoint."""
    if not isinstance(obj, dict):
        raise TypeError(type(obj))
    for key in ("params_ema", "params", "state_dict", "model"):
        if key in obj and isinstance(obj[key], dict):
            return obj[key]
    return obj


def main(argv: list[str]) -> int:
    """Print a structural summary of every checkpoint given on the command line."""
    paths: list[str] = []
    for arg in argv[1:]:
        paths.extend(glob.glob(arg) or [arg])

    for raw_path in paths:
        path = Path(raw_path)
        print(f"=== {path.name} ({path.stat().st_size} bytes)")
        try:
            state = inner_state(torch.load(str(path), map_location="cpu", weights_only=False))
        except Exception as exc:
            print(f"  UNREADABLE: {exc}")
            continue

        keys = list(state.keys())
        print(f"  keys={len(keys)}  trunk_conv={any('trunk' in k for k in keys)}")
        print(f"  has_conv_first={any(k.startswith('conv_first.') for k in keys)}")

        blocks = sorted({k.split(".")[1] for k in keys if k.startswith("body.")})
        if blocks:
            print(f"  body blocks: {len(blocks)} (0..{blocks[-1]})")
            sample = [k for k in keys if k.startswith("body.0.")]
            for key in sample[:8]:
                print(f"    {key} {tuple(state[key].shape)}")
        for key in keys:
            if not key.startswith("body."):
                print(f"    {key} {tuple(state[key].shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
