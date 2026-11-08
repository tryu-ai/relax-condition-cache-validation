"""Offline, fail-closed baseline-first gate for actual Ray single updates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .evidence import (
    assess_strict_acceptance,
    atomic_json,
    compare_trajectories,
    numerical_components,
    payload_digest,
)


def file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def load(root: Path, phase: str) -> list[dict[str, Any]]:
    files = sorted((root / phase).glob("trajectory-rank*.pt"))
    if not files:
        raise FileNotFoundError(f"No complete trajectory files: {root / phase}")
    rows = [torch.load(path, map_location="cpu", weights_only=False) for path in files]
    if [row["rank"] for row in rows] != list(range(len(rows))) or any(row["world"] != len(rows) for row in rows):
        raise ValueError("Incomplete/global-world-mismatched rank evidence")
    for row in rows:
        validate_coverage(row)
        receipt = root / phase / f"receipt-rank{row['rank']:04d}.json"
        if json.loads(receipt.read_text())["status"] != "completed":
            raise ValueError(f"Incomplete diagnostic receipt: {receipt}")
    return rows


def validate_coverage(row: dict[str, Any]) -> None:
    """Equal omissions in A and B must not masquerade as full-batch parity."""
    batch, world, rank = row["contract"]["global_batch_size"], row["world"], row["rank"]
    if type(batch) is not int or type(world) is not int or world <= 0 or batch <= 0 or batch % world:
        raise ValueError("Invalid effective batch/world in diagnostic evidence")
    assigned = row["assigned_sample_ids"]
    if len(assigned) != batch or sorted(assigned) != sorted(row["sample_ids"]):
        raise ValueError("Diagnostic sample assignments omit the effective batch")
    expected = assigned[rank::world]
    trace = row["effective_input_trace"]
    if (
        row["expected_rank_sample_ids"] != expected
        or row["dataset_sample_ids"] != expected
        or len(trace) != batch // world
        or [entry["sample_id"] for entry in trace] != expected
        or [entry["sha256"] for entry in trace] != row["effective_input_sha256"]
    ):
        raise ValueError("Incomplete or inconsistent effective-batch microbatch input evidence")
    if row["gradient_capture_calls"] != 1:
        raise ValueError("Expected one complete accumulated-gradient capture")
    last_end = 0
    for entry in trace:
        start, end = entry["random_start"], entry["random_end"]
        if (
            type(start) is not int
            or type(end) is not int
            or start < last_end
            or not start < end <= len(row["random_outputs"])
        ):
            raise ValueError("Missing sampled noise/timestep evidence for an effective-batch microbatch")
        last_end = end
    if rank == 0 and (row["partition_reads"] != 1 or row["partition_sample_ids"] != row["sample_ids"]):
        raise ValueError("Missing complete rank-zero Ray/TQ partition evidence")


def control_failures(reference: list[dict], candidate: list[dict]) -> list[str]:
    if len(reference) != len(candidate):
        return ["rank_count"]
    failures = []
    for rank, (a, b) in enumerate(zip(reference, candidate)):
        for field in (
            "scheduler_initial_sha256",
            "partition_sample_ids",
            "partition_reads",
            "assigned_sample_ids",
            "dataset_sample_ids",
            "expected_rank_sample_ids",
            "effective_input_trace",
            "gradient_capture_calls",
            "effective_input_sha256",
            "diagnostic_identity",
            "contract",
        ):
            if a[field] != b[field]:
                failures.append(f"rank{rank}:{field}")
        if not a["random_outputs"] or payload_digest(a["random_outputs"]) != payload_digest(b["random_outputs"]):
            failures.append(f"rank{rank}:actual_noise_sigma_timestep_outputs")
    return failures


def inputs(root: Path, phases: tuple[str, ...]) -> dict[str, str]:
    files = [path for phase in phases for path in sorted((root / phase).glob("trajectory-rank*.pt"))]
    files += [path for phase in phases for path in sorted((root / phase).glob("receipt-rank*.json"))]
    return {path.relative_to(root).as_posix(): file_digest(path) for path in files}


def baseline_report(root: Path) -> dict[str, Any]:
    a1, a2 = load(root, "a1"), load(root, "a2")
    comparison = compare_trajectories(a1, a2)
    controls = control_failures(a1, a2)
    components = numerical_components(comparison)
    hard = comparison["passed"] and not controls
    strict = hard and bool(components) and all(row["bitwise_equal"] for row in components.values())
    return {
        "schema": 1,
        "passed": hard,
        "strict_acceptance": strict,
        "acceptance_status": "accepted" if strict else "needs_nondeterminism_review" if hard else "failed",
        "baseline": comparison,
        "control_failures": controls,
        "input_sha256": inputs(root, ("a1", "a2")),
        "policy": "fixed 1%/zero/finite/RNG; nonexact online baseline cannot automatically authorize cache acceptance",
    }


def require_baseline(root: Path) -> dict[str, Any]:
    report = json.loads((root / "baseline.json").read_text())
    if not report.get("passed") or not report.get("strict_acceptance"):
        raise ValueError("Online A1/A2 baseline gate has not strictly passed")
    if report["input_sha256"] != inputs(root, ("a1", "a2")):
        raise ValueError("Baseline evidence changed after its gate")
    return report


def full_report(root: Path) -> dict[str, Any]:
    baseline = require_baseline(root)
    runs = {phase: load(root, phase) for phase in ("a1", "bcold", "bwarm")}
    comparisons, policies, controls = {}, {}, {}
    for phase in ("bcold", "bwarm"):
        comparisons[phase] = compare_trajectories(runs["a1"], runs[phase])
        policies[phase] = assess_strict_acceptance(baseline["baseline"], comparisons[phase])
        controls[phase] = control_failures(runs["a1"], runs[phase])
    cache_exact = all(
        row["cache_audit"]["failed"] == 0 and row["cache_audit"]["checked_hits"] > 0 for row in runs["bwarm"]
    )
    passed = cache_exact and not any(controls.values()) and all(row["passed"] for row in policies.values())
    strict = passed and all(row["strict_acceptance"] for row in policies.values())
    return {
        "schema": 1,
        "passed": passed,
        "strict_acceptance": strict,
        "acceptance_status": "accepted" if strict else "failed",
        "scope": "real Ray/FSDPSFTTrainer single update; no 66-step trajectory or performance acceptance implied",
        "baseline": baseline,
        "comparisons": comparisons,
        "strict_comparisons": policies,
        "control_failures": controls,
        "cache_payload_exact_with_checked_hits": cache_exact,
        "input_sha256": inputs(root, ("a1", "a2", "bcold", "bwarm")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        report = baseline_report(root) if args.baseline_only else full_report(root)
    except (ValueError, KeyError, FileNotFoundError) as error:
        report = {"passed": False, "strict_acceptance": False, "acceptance_status": "failed", "error": str(error)}
    path = root / ("baseline.json" if args.baseline_only else "compare.json")
    atomic_json(path, report)
    print(json.dumps({key: report[key] for key in ("passed", "strict_acceptance", "acceptance_status")}))
    return 0 if report["strict_acceptance"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
