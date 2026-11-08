"""Verify public bytes, all16 timing records and four stored numerical reports."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path

from recompute import recompute


def verify(root: Path) -> dict:
    manifest = json.loads((root / "EVIDENCE-MANIFEST.json").read_text())
    for name, entry in manifest.get("generated_files", {}).items():
        raw = (root / name).read_bytes()
        if len(raw) != entry["bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError(f"Generated inspection member checksum mismatch: {name}")
    for entry in manifest["files"]:
        path = root / entry["path"]
        raw = path.read_bytes()
        if path.is_symlink() or len(raw) != entry["stored_bytes"]:
            raise ValueError(f"Public member size/type mismatch: {entry['path']}")
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError(f"Public member checksum mismatch: {entry['path']}")
        decoded = gzip.decompress(raw) if entry["encoding"] == "gzip" else raw
        if len(decoded) != entry["public_decoded_bytes"]:
            raise ValueError(f"Decoded member size mismatch: {entry['path']}")
        if hashlib.sha256(decoded).hexdigest() != entry["public_decoded_sha256"]:
            raise ValueError(f"Decoded member checksum mismatch: {entry['path']}")
        if re.search(
            r"/Users/|/root/|autodl-container-|\b(?:gh[pousr]_|github_pat_)[A-Za-z0-9_]{20,}\b|\bhf_[A-Za-z0-9]{20,}\b",
            decoded.decode(),
        ):
            raise ValueError(f"Private path/credential signature in {entry['path']}")
    numerical = {}
    for family in ("ray", "native"):
        for model in ("sensenova", "qwen_edit"):
            path = root / "correctness" / family / model / "compare.json.gz"
            report = json.loads(gzip.decompress(path.read_bytes()))
            required = ("passed", "strict_acceptance", "cache_payload_exact_with_checked_hits")
            if not all(report.get(key) is True for key in required):
                raise ValueError(f"Stored numerical report not strictly accepted: {family}/{model}")
            numerical[family + "/" + model] = {key: report[key] for key in required}
    timing = recompute(root)
    if sum(r["formal_jobs"] for r in timing.values()) != 16:
        raise ValueError("All16 formal jobs are required")
    if sum(r["optimizer_updates"] for r in timing.values()) != 3328:
        raise ValueError("Expected3328 actual optimizer updates")
    if not all(j["valid_timing"] for r in timing.values() for j in r["jobs"].values()):
        raise ValueError("An incomplete timing job was included")
    return {
        "verified_public_files": len(manifest["files"]),
        "formal_jobs": 16,
        "optimizer_updates": 3328,
        "stored_numerical_reports": numerical,
        "raw_tensor_payloads_public": False,
        "scope": "Checksums and stored reports verified; tensor arithmetic cannot be recomputed without private PT inputs",
    }


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parent), indent=2))
