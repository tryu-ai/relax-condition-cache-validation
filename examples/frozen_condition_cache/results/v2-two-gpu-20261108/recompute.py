"""Recompute all recorded two-GPU timing pairs using the unchanged measured parser."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path


def recompute(root: Path) -> dict:
    parser_path = root.parents[1] / "two_gpu/scripts/60_summarize.py"
    spec = importlib.util.spec_from_file_location("recorded_ray_summary", parser_path)
    summary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(summary)
    result = {}
    for host in ("host-a", "host-b"):
        # The measured summarizer writes output. Confine those writes to a
        # temporary copy; inspection never modifies the recorded evidence.
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            shutil.copytree(root / "performance" / host / "e2e", work / "e2e")
            previous = sys.argv
            try:
                sys.argv = [str(parser_path), "--work", str(work)]
                with contextlib.redirect_stdout(io.StringIO()):
                    summary.main()
            finally:
                sys.argv = previous
            report = json.loads((work / "summary.json").read_text())
            result[host] = {
                "jobs": report["jobs"],
                "pairs": report["pairs"],
                "formal_jobs": len(report["jobs"]),
                "optimizer_updates": sum(j["steps_logged"] for j in report["jobs"].values()),
            }
    return result


if __name__ == "__main__":
    print(json.dumps(recompute(Path(__file__).resolve().parent), indent=2))
