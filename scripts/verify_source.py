"""Verify the separate Relax checkout used by the validation harness."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path


def main() -> None:
    location = os.environ.get("FROZEN_CACHE_RELAX_SRC")
    if not location:
        raise SystemExit("Set FROZEN_CACHE_RELAX_SRC to the separate Relax checkout.")
    source = Path(location).expanduser().resolve()
    lock = json.loads((Path(__file__).resolve().parents[1] / "sources.lock.json").read_text())
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != lock["commit"]:
        raise SystemExit(f"Relax revision mismatch: expected {lock['commit']}, got {commit}")
    for name, expected in lock["runtime_sha256"].items():
        actual = hashlib.sha256((source / name).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"Relax runtime checksum mismatch: {name}")
    print(f"Verified Relax {commit} and {len(lock['runtime_sha256'])} runtime files.")


if __name__ == "__main__":
    main()
