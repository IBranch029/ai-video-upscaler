"""Verify that every model URL in the registry resolves.

The script performs a HEAD request against each download URL and reports the
status code, so a stale upstream tag is caught before a user hits it.
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aivideoupscaler.models.registry import MODELS  # noqa: E402

_USER_AGENT = "aivideoupscaler/1.0"


def check(url: str) -> tuple[int, str]:
    """Return the HTTP status code and a short note for ``url``."""
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            return response.status, str(response.headers.get("Content-Length", "?"))
    except urllib.error.HTTPError as exc:
        return exc.code, exc.reason
    except Exception as exc:
        return 0, str(exc)


def main() -> int:
    """Check every registered download URL."""
    failures = 0
    seen: set[str] = set()

    for key, model in MODELS.items():
        for label, url in (("torch", model.torch_url), ("ncnn", model.ncnn_url)):
            if not url or url in seen:
                continue
            seen.add(url)
            status, note = check(url)
            marker = "OK " if status == 200 else "BAD"
            print(f"[{marker}] {key:32} {label:6} {status} {note} {url}")
            if status != 200:
                failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
