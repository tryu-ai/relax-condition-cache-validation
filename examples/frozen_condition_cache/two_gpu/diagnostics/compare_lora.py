#!/usr/bin/env python3
"""Compare FP32 LoRA parameters and per-step updates between Relax SFT checkpoints.

Reads ``<save>/i2i_sft/iter_<k>/fsdp`` (DCP) for each requested step, converts it with
``dcp_to_torch_save``, and compares, per parameter tensor:
  * theta_k                         (parameters after step k)
  * delta_k = theta_k - theta_{k-1}  (the FP32 update of one optimizer step), when k-1 is present
Acceptance per task No.17: L2(actual - reference) <= 0.01 * L2(reference); zero reference requires zero;
all values finite; max absolute error reported. Bitwise equality is reported separately.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import torch


def flatten(tree, prefix=""):
    if isinstance(tree, torch.Tensor):
        yield prefix, tree
    elif isinstance(tree, dict):
        for key, value in tree.items():
            yield from flatten(value, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(tree, (list, tuple)):
        for index, value in enumerate(tree):
            yield from flatten(value, f"{prefix}.{index}")


def audit_lora_state(state) -> tuple[dict[str, torch.Tensor], dict]:
    """Assert original FP32 before widening arithmetic; never infer source precision."""
    original = {
        name: tensor.detach()
        for name, tensor in flatten(state)
        if "lora_" in name and not any(tag in name for tag in ("exp_avg", "optim", ".state.", "step"))
    }
    if not original:
        raise ValueError("No LoRA parameters found")
    invalid = {name: str(tensor.dtype) for name, tensor in original.items() if tensor.dtype != torch.float32}
    if invalid:
        raise ValueError(f"Original LoRA checkpoint parameters must be FP32: {list(invalid.items())[:5]}")
    audit = {
        "source_dtype": "torch.float32",
        "tensors": len(original),
        "shapes": {name: list(tensor.shape) for name, tensor in original.items()},
    }
    return {name: tensor.to(torch.float64) for name, tensor in original.items()}, audit


def load_lora(save: Path, step: int, audits: dict | None = None) -> dict[str, torch.Tensor]:
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    directory = save / "i2i_sft" / f"iter_{step:07d}" / "fsdp"
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "state.pt"
        dcp_to_torch_save(str(directory), str(target))
        state = torch.load(target, map_location="cpu", weights_only=False)
    params, audit = audit_lora_state(state)
    if audits is not None:
        audits[str(directory)] = audit
    return params


def compare(reference: dict, actual: dict) -> dict:
    if reference.keys() != actual.keys():
        missing = sorted(reference.keys() ^ actual.keys())[:5]
        return {"passed": False, "error": f"parameter sets differ: {missing}"}
    worst_rel, worst_abs, bitwise, failures = 0.0, 0.0, True, []
    for name, ref in reference.items():
        act = actual[name]
        if ref.shape != act.shape:
            return {"passed": False, "error": f"parameter shape differs: {name}"}
        finite = bool(torch.isfinite(ref).all() and torch.isfinite(act).all())
        diff = act - ref
        ref_norm, diff_norm = float(ref.norm()), float(diff.norm())
        max_abs = float(diff.abs().max()) if diff.numel() else 0.0
        bitwise &= bool(torch.equal(ref, act))
        if ref_norm == 0.0:
            ok, rel = diff_norm == 0.0, 0.0 if diff_norm == 0.0 else math.inf
        else:
            rel = diff_norm / ref_norm
            ok = rel <= 0.01
        ok &= finite
        worst_rel, worst_abs = max(worst_rel, rel), max(worst_abs, max_abs)
        if not ok:
            failures.append({"name": name, "rel_l2": rel, "max_abs": max_abs, "finite": finite})
    return {
        "passed": not failures,
        "bitwise_equal": bitwise,
        "tensors": len(reference),
        "max_rel_l2": worst_rel,
        "max_abs": worst_abs,
        "failures": failures[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    steps = sorted(set(args.steps))
    audits = {}
    reference = {step: load_lora(args.reference, step, audits) for step in steps}
    report = {"reference": str(args.reference), "steps": steps, "candidates": {}}
    for candidate in args.candidates:
        loaded = {step: load_lora(candidate, step, audits) for step in steps}
        rows = {}
        for step in steps:
            rows[f"theta_{step}"] = compare(reference[step], loaded[step])
            if step - 1 in reference:
                ref_delta = {k: reference[step][k] - reference[step - 1][k] for k in reference[step]}
                act_delta = {k: loaded[step][k] - loaded[step - 1][k] for k in loaded[step]}
                nonzero = any(float(v.abs().max()) > 0 for v in ref_delta.values())
                rows[f"delta_{step}"] = {**compare(ref_delta, act_delta), "reference_update_nonzero": nonzero}
        report["candidates"][candidate.name] = {
            "passed": all(row["passed"] for row in rows.values()),
            "bitwise_equal": all(row.get("bitwise_equal", False) for row in rows.values()),
            "comparisons": rows,
        }
    report["passed"] = all(c["passed"] for c in report["candidates"].values())
    report["original_checkpoint_dtype_audit"] = audits
    report["comparison_arithmetic"] = "FP64 differences/norms of asserted original FP32 checkpoint parameters"
    args.output.write_text(json.dumps(report, indent=2))
    for name, row in report["candidates"].items():
        print(name, "passed", row["passed"], "bitwise", row["bitwise_equal"])
        for key, value in row["comparisons"].items():
            print(
                f"  {key}: passed={value['passed']} bitwise={value.get('bitwise_equal')} "
                f"max_rel_l2={value.get('max_rel_l2')} max_abs={value.get('max_abs')}"
            )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
