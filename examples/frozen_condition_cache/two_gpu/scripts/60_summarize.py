#!/usr/bin/env python3
"""Summarize real Ray SFT jobs (online vs cache) into summary.json / summary.md.

Per job (from the patched trainer's rank-0 log lines):
  i2i SFT step=N loss=.. grad_norm=.. ... train_seconds=.. data_wait_seconds=.. wall=<unix>
  i2i SFT condition cache step=N {"cache/global_hits": .., ...}
Step duration = wall[N] - wall[N-1] (includes TransferQueue read, data preparation, condition
loading, forward, backward, optimizer and logging: the complete training loop). Step 1 uses
its own train_seconds. Epoch 0 is the cold cache-building epoch; "steady" is every later step.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path


STEP = re.compile(
    r"i2i SFT step=(\d+) loss=([-\d.eE+naif]+) grad_norm=([-\d.eE+naif]+) .*?train_seconds=([\d.]+) "
    r"data_wait_seconds=([\d.]+) wall=([\d.]+)"
)
CACHE = re.compile(r"condition cache step=(\d+) (\{.*\})")
RECIPE_FIELDS = (
    "model",
    "global_batch",
    "micro_batch_size",
    "gradient_accumulation_steps",
    "gpus",
    "dataset_size",
    "steps",
    "metadata",
    "metadata_sha256",
    "model_identity",
    "deterministic",
    "image_prefetch_depth",
    "lr_warmup_iters",
    "save_interval",
    "max_pixels",
    "requested_attention_implementation",
    "seed",
    "lora_rank",
    "learning_rate",
    "monitor_gpu_ids",
    "hostname",
    "physical_gpu_uuids",
    "probe_torch_versions",
)


def recipe_identity(job: dict, cached: dict) -> str:
    """Keep repeat statistics separate when autonomous retries change the recipe."""
    values = {key: job.get(key) for key in RECIPE_FIELDS}
    values["cache_bytes"] = cached.get("cache_bytes")
    values["cache_device"] = cached.get("runtime_env", {}).get("env_vars", {}).get("RELAX_CONDITION_CACHE_DEVICE")
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def parse_job(directory: Path) -> dict | None:
    config_path, log_path = directory / "config.json", directory / "job.log"
    if not config_path.exists() or not log_path.exists():
        return None
    config = json.loads(config_path.read_text())
    result = json.loads((directory / "result.json").read_text()) if (directory / "result.json").exists() else {}
    text = log_path.read_text(errors="replace")
    steps, cache = {}, {}
    for match in STEP.finditer(text):
        n = int(match.group(1))
        steps[n] = dict(
            loss=match.group(2),
            grad_norm=match.group(3),
            train_seconds=float(match.group(4)),
            data_wait=float(match.group(5)),
            wall=float(match.group(6)),
        )
    for match in CACHE.finditer(text):
        cache[int(match.group(1))] = json.loads(match.group(2))
    peak = [int(v) for v in re.findall(r"peak_allocated=(\d+)", text)]
    source_path = directory / "source.json"
    source = json.loads(source_path.read_text()) if source_path.exists() else {}
    job = {
        "name": directory.name,
        **config,
        "status": result.get("status"),
        "steps_logged": len(steps),
        "source_id": source.get("source_id"),
        "valid_timing": False,
    }
    proof = config.get("gpu_preflight", {})
    job["hostname"] = config.get("hostname", proof.get("driver_hostname"))
    job["physical_gpu_uuids"] = sorted(row["uuid"] for row in proof.get("authorized_physical_gpus", []))
    job["probe_torch_versions"] = sorted({row["torch_version"] for row in proof.get("ray_actors", [])})
    if not steps:
        return job
    order = sorted(steps)
    if order != list(range(1, config["steps"] + 1)) or result.get("status") != "SUCCEEDED":
        job["timing_exclusion"] = "failed or incomplete job"
        return job
    if not all(
        math.isfinite(float(s[k])) for s in steps.values() for k in ("loss", "grad_norm", "train_seconds", "wall")
    ):
        job["timing_exclusion"] = "nonfinite measurement"
        return job
    gb = config["global_batch"]
    spe = math.ceil(config["dataset_size"] / gb)
    duration = {order[0]: steps[order[0]]["train_seconds"]}
    for prev, cur in zip(order, order[1:]):
        duration[cur] = steps[cur]["wall"] - steps[prev]["wall"]
    if any(t <= 0 for t in duration.values()):
        job["timing_exclusion"] = "nonpositive duration"
        return job
    job["valid_timing"] = bool(job["source_id"])
    cold = [duration[s] for s in order if s <= spe]
    steady = [duration[s] for s in order if s > spe]
    job.update(
        steps_per_epoch=spe,
        step_seconds=duration,
        cold_epoch_seconds=sum(cold),
        train_loop_seconds=sum(duration.values()),
        steady_steps=len(steady),
        steady_samples_per_second=(gb * len(steady) / sum(steady)) if steady else None,
        all_samples_per_second=gb * len(order) / sum(duration.values()),
        mean_data_wait=statistics.mean(steps[s]["data_wait"] for s in order),
        peak_allocated_bytes=max(peak) if peak else None,
        losses={s: steps[s]["loss"] for s in order},
        grad_norms={s: steps[s]["grad_norm"] for s in order},
    )
    if result.get("end_time_ms") and result.get("submit_wall"):
        job["job_wall_seconds"] = result["end_time_ms"] / 1000 - result["submit_wall"]
    if cache:
        last = cache[max(cache)]
        boundary = cache.get(spe, {})
        hits = last.get("cache/global_hits", 0) - boundary.get("cache/global_hits", 0)
        misses = last.get("cache/global_misses", 0) - boundary.get("cache/global_misses", 0)
        job["cache_final"] = last
        job["steady_hit_rate"] = hits / (hits + misses) if hits + misses else None
    monitor = directory / "monitor.csv"
    if monitor.exists():
        rows = list(csv.DictReader(monitor.open()))
        if rows:
            job["resources"] = {
                "max_mem_used_bytes": max(int(r["mem_used_bytes"]) for r in rows),
                "max_shm_used_bytes": max(int(r["shm_used_bytes"]) for r in rows),
                "max_cache_dir_bytes": max(int(r["cache_dir_bytes"]) for r in rows),
                "final_cache_dir_bytes": int(rows[-1]["cache_dir_bytes"]),
                "final_cache_run_dirs": int(rows[-1]["cache_run_dirs"]),
                "max_gpu_mem_used_mib": max(
                    (
                        int(r["gpu_mem_used_mib_max"])
                        for r in rows
                        if r.get("gpu_mem_used_mib_max", "").strip().isdigit()
                    ),
                    default=None,
                ),
                "gpu_samples_missing_max": sum(not r.get("gpu_mem_used_mib_max", "").strip().isdigit() for r in rows),
                "first_mem_used_bytes": int(rows[0]["mem_used_bytes"]),
                "final_mem_used_bytes": int(rows[-1]["mem_used_bytes"]),
            }
    return job


def payback_step(online: dict, cached: dict) -> int | None:
    """First step after which the cached run's cumulative time stays <= online's."""
    common = sorted(set(online["step_seconds"]) & set(cached["step_seconds"]), key=int)
    a = b = 0.0
    ahead = []
    for step in common:
        a += online["step_seconds"][step]
        b += cached["step_seconds"][step]
        ahead.append((int(step), b <= a))
    for index, (step, _) in enumerate(ahead):
        if all(flag for _, flag in ahead[index:]):
            return step
    return None


def stats(values: list[float]) -> dict:
    values = [v for v in values if v is not None]
    if not values:
        return {}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else None,
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    root = args.work / "e2e"
    jobs = {d.name: j for d in sorted(root.iterdir()) if d.is_dir() and (j := parse_job(d))} if root.exists() else {}
    summary = {
        "jobs": {
            k: {kk: vv for kk, vv in v.items() if kk not in ("step_seconds", "losses", "grad_norms")}
            for k, v in jobs.items()
        }
    }
    md = [
        "# No.17 two-GPU repair — Ray measurements grouped by actual source, recipe and attempt",
        "",
        "Throughput/time-ratio gains are not elapsed-time reductions. n=1 has no sample standard deviation.",
        "Host RAM/SHM are whole-machine readings. GPU readings cover MONITOR_GPU_IDS only.",
        "SUCCEEDED means process completion, not numerical acceptance. Diagnostic jobs use a separate protocol.",
        "",
    ]

    # Throughput pairs.
    pairs = {}
    for name, job in jobs.items():
        match = re.fullmatch(r"(sn|qw)-(high|low)(?:-p(\d))?-(a|b)-(.+)", name)
        if match and job.get("valid_timing"):
            key = (
                match.group(1),
                match.group(2),
                match.group(3) or "1",
                match.group(5),
                job["source_id"],
            )
            pairs.setdefault(key, {})[match.group(4)] = job
    groups = {}
    md += [
        "| workload | pair | online steady samples/s | cache steady samples/s | steady gain | train-loop gain | "
        "job gain | cold epoch online/cache s | payback step | steady hit rate |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (model, data, index, attempt, source_id), pair in sorted(pairs.items()):
        if "a" not in pair or "b" not in pair:
            continue
        a, b = pair["a"], pair["b"]
        for key in RECIPE_FIELDS:
            if a.get(key) != b.get(key):
                raise ValueError(f"Incompatible A/B configuration {key}: {a['name']} vs {b['name']}")
        row = {
            "steady_gain": (b["steady_samples_per_second"] / a["steady_samples_per_second"] - 1)
            if a.get("steady_samples_per_second") and b.get("steady_samples_per_second")
            else None,
            "all_steps_gain": b["all_samples_per_second"] / a["all_samples_per_second"] - 1,
            "train_loop_gain": a["train_loop_seconds"] / b["train_loop_seconds"] - 1,
            "job_gain": (a["job_wall_seconds"] / b["job_wall_seconds"] - 1)
            if a.get("job_wall_seconds") and b.get("job_wall_seconds")
            else None,
            "cold_epoch_seconds": [a["cold_epoch_seconds"], b["cold_epoch_seconds"]],
            "payback_step": payback_step(a, b),
            "steady_hit_rate": b.get("steady_hit_rate"),
            "train_time_reduction": 1 - b["train_loop_seconds"] / a["train_loop_seconds"],
            "job_time_reduction": 1 - b["job_wall_seconds"] / a["job_wall_seconds"]
            if a.get("job_wall_seconds") and b.get("job_wall_seconds")
            else None,
            "online": a["name"],
            "cached": b["name"],
            "source_id": source_id,
            "gpus": a["gpus"],
            "global_batch": a["global_batch"],
            "micro_batch_size": a.get("micro_batch_size"),
            "gradient_accumulation_steps": a.get("gradient_accumulation_steps"),
            "recipe_sha256": recipe_identity(a, b),
            "cache_bytes": b.get("cache_bytes"),
            "attempt": attempt,
        }
        group_key = f"{model}-{data}/{attempt}/{source_id[:12]}/{row['recipe_sha256'][:12]}"
        groups.setdefault(group_key, []).append(row)
        fmt = lambda v: "n/a" if v is None else f"{100 * v:+.2f}%"
        md.append(
            f"| {group_key} | {index} | {a.get('steady_samples_per_second') or a['all_samples_per_second']:.4f} | "
            f"{b.get('steady_samples_per_second') or b['all_samples_per_second']:.4f} | {fmt(row['steady_gain'])} | "
            f"{fmt(row['train_loop_gain'])} | {fmt(row['job_gain'])} | {a['cold_epoch_seconds']:.1f} / "
            f"{b['cold_epoch_seconds']:.1f} | {row['payback_step']} | {b.get('steady_hit_rate')} |"
        )
    summary["pairs"] = {
        key: {
            metric: stats([r[metric] for r in rows])
            for metric in (
                "steady_gain",
                "all_steps_gain",
                "train_loop_gain",
                "job_gain",
            )
        }
        | {"rows": rows}
        for key, rows in groups.items()
    }
    md += ["", "## Aggregates (mean ± sample std over pairs)", ""]
    for key, value in summary["pairs"].items():
        for metric in ("steady_gain", "all_steps_gain", "train_loop_gain", "job_gain"):
            s = value[metric]
            if s:
                spread = "n/a (n=1)" if s["std"] is None else f"{100 * s['std']:.2f} percentage points"
                md.append(
                    f"- {key} {metric}: {100 * s['mean']:+.2f}% ± {spread} "
                    f"(n={s['n']}, min {100 * s['min']:+.2f}%, max {100 * s['max']:+.2f}%)"
                )

    # Deterministic runs: per-step loss/grad_norm as logged (%.8f).
    md += ["", "## Real-job numerics (rank-0 logged loss / grad_norm per step)", ""]
    summary["numerics"] = {}
    for model in ("sn", "qw"):
        ref = jobs.get(f"{model}-det-a1")
        if not ref or "losses" not in ref:
            continue
        for other in (f"{model}-det-b", f"{model}-det-a2"):
            cand = jobs.get(other)
            if not cand or "losses" not in cand:
                continue
            common = sorted(set(ref["losses"]) & set(cand["losses"]))

            def rel(x, y):
                x, y = float(x), float(y)
                return 0.0 if x == y else abs(x - y) / abs(x) if x else math.inf

            loss_rel = max(rel(ref["losses"][s], cand["losses"][s]) for s in common)
            grad_rel = max(rel(ref["grad_norms"][s], cand["grad_norms"][s]) for s in common)
            exact = all(
                ref["losses"][s] == cand["losses"][s] and ref["grad_norms"][s] == cand["grad_norms"][s] for s in common
            )
            row = {
                "steps": len(common),
                "printed_exact": exact,
                "max_loss_rel": loss_rel,
                "max_grad_norm_rel": grad_rel,
                "candidate_hit_rate": cand.get("steady_hit_rate"),
            }
            summary["numerics"][f"{model}-det-a1 vs {other}"] = row
            md.append(
                f"- {model}-det-a1 vs {other}: steps={len(common)} printed-exact={exact} "
                f"max loss rel={loss_rel:.3e} max grad_norm rel={grad_rel:.3e} hit_rate={row['candidate_hit_rate']}"
            )
        lora = args.work / "e2e" / f"{model}-det-lora-compare.json"
        if lora.exists():
            report = json.loads(lora.read_text())
            summary["numerics"][f"{model}-lora"] = report
            for name, cand in report["candidates"].items():
                md.append(
                    f"- {model} FP32 LoRA θ/Δθ vs a1 — {name}: passed={cand['passed']} bitwise={cand['bitwise_equal']}"
                )

    # Native runner comparisons.
    native = args.work / "numerics"
    if native.exists():
        md += ["", "## Native FSDPSFTTrainer comparisons (validation runner)", ""]
        summary["native"] = {}
        for path in sorted(native.rglob("compare.json")):
            report = json.loads(path.read_text())
            keep = {
                k: report.get(k)
                for k in (
                    "passed",
                    "strict_acceptance",
                    "acceptance_status",
                    "cache_payload_exact_with_checked_hits",
                    "baseline_observed_exact",
                )
            }
            label = str(path.parent.relative_to(native))
            summary["native"][label] = keep
            md.append(f"- {label}: {keep}")

    diagnostics = args.work / "ray-single"
    if diagnostics.exists():
        md += ["", "## Controlled Ray single-update diagnostics", ""]
        summary["ray_single"] = {}
        for path in sorted(set(diagnostics.rglob("*compare*.json")) | set(diagnostics.rglob("baseline.json"))):
            report = json.loads(path.read_text())
            label = str(path.relative_to(diagnostics))
            summary["ray_single"][label] = report
            md.append(
                f"- {label}: passed={report.get('passed')}, strict={report.get('strict_acceptance')}; see complete report"
            )

    # Resources and lifecycle.
    md += [
        "",
        "## Resources and lifecycle",
        "",
        "| job | status | peak GPU alloc GiB | max host mem GiB | max shm GiB | max cache dir GiB | cache dir after job | final cache stats |",
        "|---|---|---|---|---|---|---|---|",
    ]
    gib = lambda v: "n/a" if v is None else f"{v / 1024**3:.2f}"
    for name, job in jobs.items():
        res = job.get("resources", {})
        final = job.get("cache_final", {})
        brief = {
            k.split("/")[-1]: final[k]
            for k in (
                "cache/bytes",
                "cache/entries",
                "cache/evictions",
                "cache/write_dropped",
                "cache/write_errors",
                "cache/corruptions",
            )
            if k in final
        }
        md.append(
            f"| {name} | {job.get('status')} | {gib(job.get('peak_allocated_bytes'))} | "
            f"{gib(res.get('max_mem_used_bytes'))} | {gib(res.get('max_shm_used_bytes'))} | "
            f"{gib(res.get('max_cache_dir_bytes'))} | {gib(res.get('final_cache_dir_bytes'))} "
            f"({res.get('final_cache_run_dirs')} dirs) | {brief} |"
        )
    (args.work / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (args.work / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
