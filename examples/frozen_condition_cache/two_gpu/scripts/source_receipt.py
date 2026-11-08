"""Record separate production and validation source identities for a new run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def snapshot(work: Path, kit: Path, *, lock_path: Path | None = None) -> dict:
    source = Path(os.environ["FROZEN_CACHE_RELAX_SRC"]).expanduser().resolve()
    lock_path = lock_path or kit.parents[2] / "sources.v2.lock.json"
    lock = json.loads(lock_path.read_text())
    if lock.get("status") != "READY" or not lock.get("commit"):
        raise ValueError("sources.v2.lock.json needs the reviewed production commit before a new GPU run")
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != lock["commit"]:
        raise ValueError("Selected Relax HEAD differs from sources.v2.lock.json")
    expected = lock["runtime_sha256"]
    actual = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in expected}
    if actual != expected:
        raise ValueError("Selected Relax runtime differs from sources.v2.lock.json")
    files = {}
    for label, root in (("relax", source / "relax"), ("launcher", source / "scripts"), ("validation", kit)):
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".sh"} and "__pycache__" not in path.parts:
                files[label + ":" + path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    example = kit.parent
    for path in (example / "runner/common.py", *(example / "assets").glob("*.py"), example / "assets/manifest.json"):
        if path.is_file():
            files["validation-shared:" + path.relative_to(example).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return {
        "schema": 2,
        "release": "public-two-gpu-validation-20261108",
        "source_id": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
        "recorded_historical_source_id": lock["recorded_source_id"],
        "runtime_matches_delivered": True,
        "runtime_sha256": actual,
        "production_commit": lock["commit"],
        "production_worktree_dirty": bool(
            subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip()
        ),
        "source_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = snapshot(Path(os.environ["WORK"]), Path(os.environ["KIT"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["source_id"])


if __name__ == "__main__":
    main()
