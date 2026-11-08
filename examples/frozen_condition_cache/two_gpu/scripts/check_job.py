#!/usr/bin/env python3
"""Reject failed/incomplete Ray logs and check repeated-sample smoke hits."""

import argparse
import json
import math
import re
from pathlib import Path


def nonnegative_integer(metrics: dict, key: str) -> int:
    value = metrics[key]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or value != int(value)
    ):
        raise ValueError(f"Invalid smoke metric {key}: {value}")
    return int(value)


def check_cache_exit(out: Path, cfg: dict) -> dict:
    """Inspect a worker-side lstat snapshot, never the reviewer's local path."""
    record = json.loads((out / "cache-exit.json").read_text())
    if record["root"].rstrip("/") != cfg["cache_root"].rstrip("/"):
        raise ValueError("Cache exit snapshot is for a different root")
    if type(record["exists"]) is not bool or not isinstance(record["entries"], list):
        raise ValueError("Malformed cache exit snapshot")
    if not record["exists"] and record["entries"]:
        raise ValueError("Missing cache root has recorded children")
    for entry in record["entries"]:
        if (
            entry.get("name") != ".lifecycle.lock"
            or entry.get("type") != "file"
            or type(entry.get("bytes")) is not int
            or entry["bytes"] != 0
        ):
            raise ValueError(f"Cache did not clean its run/entry files after exit: {entry}")
    if len(record["entries"]) > 1:
        raise ValueError("Duplicate lifecycle lock entries in cache exit snapshot")
    return record


def check_smoke(out: Path, cfg: dict, records: list[tuple[int, dict]]) -> dict:
    # Retain this early guard for a successfully exited process with no reuse.
    if cfg["arm"] != "cache" or not records or records[-1][1].get("cache/global_hits", 0) <= 0:
        raise ValueError("Repeated-sample smoke did not show collective cache hits")
    batch, size, world = int(cfg["global_batch"]), int(cfg["dataset_size"]), int(cfg["gpus"])
    if batch <= 0 or world <= 0 or batch % world or size <= 0 or size % batch:
        raise ValueError("Smoke needs complete global batches with equal rank microbatches")
    epoch = size // batch
    steps = int(cfg["steps"])
    if steps < 3 * epoch or steps % epoch:
        raise ValueError("Smoke needs a complete cold epoch and at least two complete hot epochs")
    if [step for step, _ in records] != list(range(1, steps + 1)):
        raise ValueError("Incomplete or duplicate per-step cache smoke metrics")
    errors = (
        "cache/corruptions",
        "cache/write_errors",
        "cache/write_dropped",
        "cache/evictions",
        "cache/oversized",
        "cache/bypasses",
    )
    calls = []
    previous_hits = previous_misses = 0
    for step, metrics in records:
        hits = nonnegative_integer(metrics, "cache/global_hits")
        misses = nonnegative_integer(metrics, "cache/global_misses")
        entries = nonnegative_integer(metrics, "cache/entries")
        if hits < previous_hits or misses < previous_misses:
            raise ValueError("Cache branch counters reset during a smoke run")
        lookups = hits + misses - previous_hits - previous_misses
        if lookups <= 0:
            raise ValueError("Smoke cache lookup coverage is empty")
        calls.append(lookups)
        for key in errors:
            if nonnegative_integer(metrics, key):
                raise ValueError(f"Smoke reports {key} at step {step}")
        # The shared store reports node-wide entries, not each rank's commits.
        if entries > size:
            raise ValueError("Cache entries exceed the fixed dataset; possible rank/device-dependent keys")
        if step > epoch and (hits - previous_hits != lookups or misses != previous_misses):
            raise ValueError(f"Hot epoch lacks complete collective cache hits at step {step}")
        # Async cold writes may be visible late. The final complete hot epoch
        # must have the stable, complete node-wide working set.
        if step > steps - epoch and entries != size:
            raise ValueError("Final hot epoch does not retain a stable complete cache working set")
        if nonnegative_integer(metrics, "cache/bytes") > int(cfg["cache_bytes"]):
            raise ValueError("Smoke exceeds its node-wide cache byte budget")
        previous_hits, previous_misses = hits, misses
    per_step = calls[0]
    micros = batch // world
    if any(value != per_step for value in calls) or per_step % micros:
        raise ValueError("Cache metric coverage does not match all accumulated microbatches")
    if nonnegative_integer(records[-1][1], "cache/pending_bytes"):
        raise ValueError("Smoke finishes with pending cache writes")
    return {
        "epochs": steps // epoch,
        "steps_per_epoch": epoch,
        "rank_microbatches_per_update": micros,
        "observed_collective_branch_events_per_update": per_step,
        "observed_counter_multiplier": per_step // micros,
        "counter_note": "Branch counts may be rank-local or aggregated; every observed hot event must hit. Entries are node-wide.",
        "hot_hit_rate": 1.0,
        "stable_entries": size,
        "cache_exit": check_cache_exit(out, cfg),
    }


def check(out: Path, smoke: bool = False) -> dict:
    cfg = json.loads((out / "config.json").read_text())
    result = json.loads((out / "result.json").read_text())
    if result["status"] != "SUCCEEDED":
        raise ValueError(result)
    log = (out / "job.log").read_text()
    rows = re.findall(r"i2i SFT step=(\d+).*?loss=([-\d.eE+naif]+).*?grad_norm=([-\d.eE+naif]+)", log)
    # The trainer logs step one as its first update.
    ids = [int(row[0]) for row in rows]
    expected = list(range(1, int(cfg["steps"]) + 1))
    if ids != expected:
        raise ValueError(f"Expected contiguous steps {expected[:3]}..{expected[-1:]}, got {ids[:5]}..{ids[-5:]}")
    if not all(math.isfinite(float(v)) for row in rows for v in row[1:]):
        raise ValueError("Nonfinite logged loss/grad_norm")
    records = [
        (int(step), json.loads(value)) for step, value in re.findall(r"condition cache step=(\d+) (\{.*\})", log)
    ]
    smoke_quality = check_smoke(out, cfg, records) if smoke else None
    report = {
        "passed": True,
        "steps": len(rows),
        "smoke_hits_checked": smoke,
        "last_cache": records[-1][1] if records else None,
        "smoke_quality": smoke_quality,
    }
    (out / ("smoke-check.json" if smoke else "completion-check.json")).write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(check(args.out, args.smoke))
