"""Opt-in hooks around the *real* Ray actor's FSDPSFTTrainer update."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from .capture import (
    CacheAudit,
    RandomCapture,
    execution_state,
    failure,
    restore_state,
    sample_digest,
    save_tensor_file,
    snapshot_state,
    source_identity,
    trainable_parameters,
)
from .evidence import atomic_json, capture_rng, cpu_clone, payload_digest


def assigned_ids(trainer: Any) -> list[int]:
    if not trainer.args.i2i_sft_config.get("dp_tier_micro_balance", False):
        return list(trainer.sampler.batch(0))
    from relax.engine.sft.image import assign_image_batch

    return assign_image_batch(
        trainer.sampler.batch(0),
        trainer.args.i2i_sft_config,
        step=0,
        world=trainer.world,
    )


class ObservedDataset:
    """Transparent synchronous reads; no decoding or sample preparation is repeated."""

    def __init__(self, original, observe):
        self.original, self.observe = original, observe

    def __getitem__(self, index):
        value = self.original[index]
        self.observe(index)
        return value

    def __getattr__(self, name):
        return getattr(self.original, name)


class Probe:
    def __init__(self, trainer: Any) -> None:
        self.trainer = trainer
        self.phase = os.environ.get("DIAG_PHASE", "")
        if self.phase not in {"a1", "a2", "bcold", "bwarm"}:
            raise ValueError("DIAG_PHASE must be a1, a2, bcold or bwarm")
        self.root = Path(os.environ["DIAG_ROOT"]).expanduser().resolve()
        self.model_identity = os.environ.get("DIAG_MODEL_IDENTITY", "")
        if not re.fullmatch(r"[a-f0-9]{64}", self.model_identity):
            raise ValueError("DIAG_MODEL_IDENTITY must be a verified model manifest SHA256")
        source_receipt = os.environ.get("DIAG_SOURCE_ID", "")
        if not re.fullmatch(r"[a-f0-9]{64}", source_receipt):
            raise ValueError("DIAG_SOURCE_ID must be the current full source-receipt SHA256")
        self.rank, self.world = trainer.rank, trainer.world
        self.directory = self.root / self.phase
        self.directory.mkdir(parents=True, exist_ok=True)
        self.receipt = self.directory / f"receipt-rank{self.rank:04d}.json"
        if self.receipt.exists():
            raise FileExistsError("Use a new DIAG_ROOT for any retry or modified diagnostic")
        atomic_json(self.receipt, {"status": "initializing", "rank": self.rank, "phase": self.phase})
        self.control = None
        if dist.is_initialized() and self.world > 1:
            from relax.utils.distributed_utils import get_gloo_group

            self.control = get_gloo_group()
        self.used = False
        self.cache_audit = None
        self.input_hashes: list[str] = []
        self.input_trace: list[dict[str, Any]] = []
        self.partition_ids = None
        self.partition_reads = 0
        self.dataset_ids: list[int] = []
        self.identity = {
            "model_identity": self.model_identity,
            "source_receipt_sha256": source_receipt,
            "sources": source_identity(trainer),
            "execution_state": execution_state(trainer.actor.model),
        }

    def agree(self, action):
        value, error = None, None
        try:
            value = action()
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
        errors = [error]
        if self.control is not None:
            errors = [None] * self.world
            dist.all_gather_object(errors, error, group=self.control)
        if any(errors):
            raise RuntimeError(f"Diagnostic rank failure: {errors}")
        return value

    def prepare(self) -> None:
        self.initial = self.agree(self.prepare_initial)
        if self.phase == "bwarm":
            self.warm()
        self.agree(self.finish_prepare)

    def prepare_initial(self) -> dict[str, Any]:
        t = self.trainer
        if t.step != 0 or t.args.num_rollout != 1 or t.args.load:
            raise ValueError("Diagnostic requires a fresh Ray job, --num-rollout 1 and no --load")
        if t.args.i2i_sft_config.get("image_prefetch_depth", 0):
            raise ValueError("Set IMAGE_PREFETCH_DEPTH=0 in all single-update diagnostic jobs")
        batch = t.contract["global_batch_size"]
        if type(batch) is not int or batch <= 0 or batch % self.world:
            raise ValueError("Diagnostic effective batch must divide across all ranks")
        self.assigned_ids = assigned_ids(t)
        if len(self.assigned_ids) != batch or sorted(self.assigned_ids) != sorted(t.sampler.batch(0)):
            raise ValueError("Assigned sample IDs do not cover the complete effective batch")
        self.expected_rank_ids = self.assigned_ids[self.rank :: self.world]
        cached = getattr(t.actor.model, "condition_cache", None) is not None
        if cached != self.phase.startswith("b"):
            raise ValueError("A1/A2 must be online (cache bytes 0); Bcold/Bwarm require enabled cache")
        initial = self.root / "initial" / f"rank{self.rank:04d}.pt"
        if self.phase.startswith("b"):
            from .compare import require_baseline

            require_baseline(self.root)
        if self.phase == "a1":
            if t.actor.optimizer.state:
                raise ValueError("A1 optimizer must be pristine before its first update")
            state = snapshot_state(t)
            state["diagnostic_identity"] = self.identity
            save_tensor_file(initial, state, exclusive=True)
        else:
            state = torch.load(initial, map_location="cpu", weights_only=False)
            if state["diagnostic_identity"] != self.identity:
                raise ValueError("Source/model/attention/determinism state differs from A1")
            restore_state(t, state)
        self.parameters = trainable_parameters(t.actor.model)
        if cached:
            self.cache_audit = CacheAudit(t.actor.model.condition_cache)
        return state

    def finish_prepare(self) -> None:
        t, state = self.trainer, self.initial
        # Initial state must stay identical even after prefill; no model
        # load_state_dict is used here (that would invalidate the warm cache).
        restore_state(t, state)
        self.before = snapshot_state(t)
        atomic_json(
            self.receipt,
            {
                "status": "ready",
                "rank": self.rank,
                "world": self.world,
                "phase": self.phase,
                "identity": self.identity,
                "initial_parameters_sha256": payload_digest(self.before["parameters"]),
                "optimizer_initial_sha256": payload_digest(self.before["optimizer"]),
                "scheduler_initial_sha256": payload_digest(self.before["scheduler"]),
                "rng_before_sha256": payload_digest(self.before["rng"]),
                "contract": t.contract,
                "expected_rank_sample_ids": self.expected_rank_ids,
                "expected_microbatches": len(self.expected_rank_ids),
            },
        )

    def warm(self) -> None:
        t = self.trainer
        ids = assigned_ids(t)
        for sample_id in ids[self.rank :: self.world]:
            data = self.agree(lambda: t.dataset[sample_id])
            with torch.no_grad():
                t.actor.model(data)
        cache = t.actor.model.condition_cache
        if hasattr(cache, "flush"):
            self.agree(cache.flush)
        if self.control is not None:
            dist.barrier(group=self.control)

    def run(self, original, module, step: int) -> Any:
        if self.used or step != 0:
            raise ValueError("Only one optimizer update is allowed per diagnostic Ray job")
        self.used = True
        t = self.trainer
        gradients, metrics = {}, []
        clip_calls = 0
        original_clip, original_report = module.clip_sharded_gradients, module._report_sft_metrics
        original_partition = t.actor._read_train_partition
        original_dataset = t.dataset
        pending_ids = []
        had_partition = "_read_train_partition" in vars(t.actor)
        partition_attribute = vars(t.actor).get("_read_train_partition")

        def clip(parameters, maximum, group):
            nonlocal clip_calls
            clip_calls += 1
            if clip_calls != 1 or len(self.input_trace) != len(self.expected_rank_ids):
                raise ValueError("Gradient capture must occur once after every effective-batch microbatch")
            if any(row["random_end"] is None for row in self.input_trace):
                raise ValueError("A microbatch forward is incomplete at gradient capture")
            gradients.update(
                {name: None if p.grad is None else cpu_clone(p.grad) for name, p in self.parameters.items()}
            )
            return original_clip(parameters, maximum, group)

        def report(args, values):
            if args is t.args:
                metrics.append(dict(values))
            return original_report(args, values)

        def partition(index):
            value = original_partition(index)
            if value is not None:
                self.partition_reads += 1
                self.partition_ids = value["sample_id"].tolist()
            return value

        def inputs(_model, args):
            index = len(self.input_trace)
            if index >= len(self.expected_rank_ids):
                raise ValueError("Observed more microbatches than the effective batch")
            if len(pending_ids) != 1:
                raise ValueError("Each microbatch must consume exactly one observed dataset read")
            actual_id = pending_ids.pop()
            if actual_id != self.expected_rank_ids[index]:
                raise ValueError("Observed dataset sample order differs from the assigned effective batch")
            digest = sample_digest(args[0])
            self.input_hashes.append(digest)
            self.input_trace.append(
                {
                    "sample_id": actual_id,
                    "sha256": digest,
                    "random_start": len(randoms.records),
                    "random_end": None,
                }
            )

        def outputs(_model, _args, _output):
            self.input_trace[-1]["random_end"] = len(randoms.records)

        def observe_read(index):
            self.dataset_ids.append(index)
            pending_ids.append(index)

        module.clip_sharded_gradients, module._report_sft_metrics = clip, report
        t.actor._read_train_partition = partition
        t.dataset = ObservedDataset(original_dataset, observe_read)
        handle = t.actor.model.register_forward_pre_hook(inputs)
        output_handle = t.actor.model.register_forward_hook(outputs)
        randoms = RandomCapture()
        try:
            with randoms:
                value = original(t, step)
            if not gradients or not randoms.records:
                raise ValueError("Missing full gradient or observable noise/timestep/sigma evidence")
            if clip_calls != 1 or len(self.input_trace) != len(self.expected_rank_ids):
                raise ValueError("Missing effective-batch inputs or single complete gradient capture")
            if pending_ids or self.dataset_ids != self.expected_rank_ids:
                raise ValueError("Observed dataset reads omit, duplicate, or reorder effective-batch samples")
            if self.rank == 0 and (self.partition_reads != 1 or self.partition_ids != t.sampler.batch(0)):
                raise ValueError("Rank zero did not read exactly the complete effective batch from Ray/TQ")
            holder = [metrics if self.rank == 0 else None]
            if self.control is not None:
                dist.broadcast_object_list(holder, src=0, group=self.control)
            metrics = holder[0]
            if len(metrics) != 1 or "train/loss" not in metrics[0]:
                raise ValueError("Expected exactly one native full-batch loss metric")
            audit = self.cache_audit.result() if self.cache_audit else dict(writes=0, checked_hits=0, failed=0)
            if self.phase == "bwarm" and (audit["failed"] or not audit["checked_hits"]):
                raise ValueError("Warm diagnostic did not verify a real exact cache hit on this rank")
            row = {
                "rank": self.rank,
                "world": self.world,
                "loss": metrics[0]["train/loss"],
                "gradients": gradients,
                "delta": {name: cpu_clone(p) - self.before["parameters"][name] for name, p in self.parameters.items()},
                "initial_parameters_sha256": payload_digest(self.before["parameters"]),
                "optimizer_initial_sha256": payload_digest(self.before["optimizer"]),
                "scheduler_initial_sha256": payload_digest(self.before["scheduler"]),
                "rng_before_sha256": payload_digest(self.before["rng"]),
                "rng_after_sha256": payload_digest(capture_rng()),
                "sample_ids": t.sampler.batch(0),
                "partition_sample_ids": self.partition_ids,
                "partition_reads": self.partition_reads,
                "assigned_sample_ids": self.assigned_ids,
                "dataset_sample_ids": self.dataset_ids,
                "expected_rank_sample_ids": self.expected_rank_ids,
                "effective_input_trace": self.input_trace,
                "gradient_capture_calls": clip_calls,
                "effective_input_sha256": self.input_hashes,
                "parameter_global_shapes": self.before["global_shapes"],
                "cache_audit": audit,
                "random_outputs": randoms.records,
                "random_capture_scope": RandomCapture.__doc__,
                "diagnostic_identity": self.identity,
                "contract": t.contract,
                "additional_optimizer_updates": 0,
            }
            save_tensor_file(self.directory / f"trajectory-rank{self.rank:04d}.pt", row, exclusive=True)
            atomic_json(
                self.receipt,
                {
                    "status": "completed",
                    "phase": self.phase,
                    "rank": self.rank,
                    "world": self.world,
                    "cache_audit": audit,
                    "random_calls": len(randoms.records),
                    "random_bytes": randoms.bytes,
                    "effective_microbatches": len(self.input_trace),
                    "gradient_capture_calls": clip_calls,
                    "scope": "single-update correctness diagnostic, not performance timing",
                    "identity": self.identity,
                },
            )
            return value
        except BaseException as error:
            failure(self.receipt, error)
            raise
        finally:
            module.clip_sharded_gradients, module._report_sft_metrics = original_clip, original_report
            t.dataset = original_dataset
            if had_partition:
                t.actor._read_train_partition = partition_attribute
            else:
                del t.actor._read_train_partition
            handle.remove()
            output_handle.remove()
            if self.cache_audit:
                self.cache_audit.close()


def install(module=None) -> None:
    if module is None:
        from relax.backends.fsdp import sft as module
    trainer_class = module.FSDPSFTTrainer
    if getattr(trainer_class, "_single_update_probe", False):
        return
    original_init, original_train = trainer_class.__init__, trainer_class.train

    def initialize(self, actor):
        original_init(self, actor)
        if getattr(actor.adapter, "_relax_single_update_diagnostic", False) is not True:
            return
        probe = Probe(self)
        self._single_update_state = probe
        try:
            probe.prepare()
        except BaseException as error:
            failure(probe.receipt, error)
            if probe.cache_audit:
                probe.cache_audit.close()
            self.close()
            raise

    def train(self, step):
        probe = getattr(self, "_single_update_state", None)
        if probe is None:
            return original_train(self, step)
        try:
            return probe.run(original_train, module, step)
        except BaseException:
            self.close()
            raise

    trainer_class.__init__, trainer_class.train = initialize, train
    trainer_class._single_update_probe = True
