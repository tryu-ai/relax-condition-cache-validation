import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest
from two_gpu.scripts import source_receipt


def test_source_receipt_checks_actual_head_and_separates_roots(tmp_path, monkeypatch):
    source, kit = tmp_path / "source", tmp_path / "kit"
    (source / "relax").mkdir(parents=True)
    (source / "scripts").mkdir()
    kit.mkdir()
    for root in (source / "relax", kit):
        (root / "shared.py").write_text("value = 1\n")
    expected = hashlib.sha256((source / "relax/shared.py").read_bytes()).hexdigest()
    lock = tmp_path / "lock.json"
    lock.write_text(
        json.dumps(
            {
                "status": "READY",
                "commit": "expected",
                "recorded_source_id": "historical",
                "runtime_sha256": {"relax/shared.py": expected},
            }
        )
    )
    monkeypatch.setenv("FROZEN_CACHE_RELAX_SRC", str(source))
    monkeypatch.setattr(subprocess, "check_output", lambda command, **kwargs: "wrong\n")
    with pytest.raises(ValueError, match="HEAD differs"):
        source_receipt.snapshot(tmp_path, kit, lock_path=lock)
    monkeypatch.setattr(
        subprocess, "check_output", lambda command, **kwargs: "expected\n" if "rev-parse" in command else ""
    )
    result = source_receipt.snapshot(tmp_path, kit, lock_path=lock)
    assert result["production_commit"] == "expected"
    assert result["runtime_matches_delivered"]
    assert set(result["files"]) == {"relax:shared.py", "validation:shared.py"}
    (source / "relax/shared.py").write_text("value = 2\n")
    with pytest.raises(ValueError, match="runtime differs"):
        source_receipt.snapshot(tmp_path, kit, lock_path=lock)


def test_environment_canonicalizes_relative_paths_and_rejects_other_batch(tmp_path):
    env_script = Path(source_receipt.__file__).resolve().parents[1] / "env.sh"
    (tmp_path / "source").mkdir()
    env = {
        **os.environ,
        "FROZEN_CACHE_RELAX_SRC": "source",
        "WORK": "work",
        "ASSETS": "assets",
        "GPUS": "2",
        "GLOBAL_BATCH": "4",
        "MICRO_BATCH_SIZE": "1",
    }
    command = [
        "bash",
        "-c",
        'source "$1"; printf "%s\\n" "$FROZEN_CACHE_RELAX_SRC" "$WORK" "$ASSETS"',
        "_",
        str(env_script),
    ]
    result = subprocess.run(command, cwd=tmp_path, env=env, text=True, capture_output=True, check=True)
    assert result.stdout.splitlines() == [str(tmp_path / n) for n in ("source", "work", "assets")]
    env["GLOBAL_BATCH"] = "2"
    failed = subprocess.run(
        ["bash", "-c", 'source "$1"', "_", str(env_script)], cwd=tmp_path, env=env, text=True, capture_output=True
    )
    assert failed.returncode == 2
