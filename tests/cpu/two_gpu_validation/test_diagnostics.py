"""CPU tests of real tensor capture/restore/FIFO state, no models or Ray/GPU."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch
from two_gpu.diagnostics.capture import RandomCapture, restore_state, snapshot_state
from two_gpu.diagnostics.compare import baseline_report, full_report, require_baseline, validate_coverage
from two_gpu.diagnostics.compare_lora import audit_lora_state, compare
from two_gpu.diagnostics.evidence import atomic_json, capture_rng, payload_digest, restore_rng
from two_gpu.diagnostics.ray_hook import install


class TinyCache:
    def __init__(self):
        self.data = {}

    def get(self, key, device=None):
        return self.data.get(key)

    def put(self, key, value):
        self.data[key] = value.clone()
        return True

    def flush(self):
        pass


class TinyModel(torch.nn.Module):
    def __init__(self, cached):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(2))
        self.condition_cache = TinyCache() if cached else None

    def forward(self, data):
        x = data["x"]
        if self.condition_cache is not None:
            conditions = self.condition_cache.get("x")
            if conditions is None:
                self.condition_cache.put("x", x)
                conditions = x
        else:
            conditions = x
        noise = torch.randn_like(x)
        sigma = torch.rand((1,))
        return ((self.weight * conditions + noise * sigma).sum() - 0.5).square()


class TinyAdapter:
    family = "toy"
    _relax_single_update_diagnostic = True


class TinyActor:
    def __init__(self, cached):
        self.args = SimpleNamespace(num_rollout=1, load=None, i2i_sft_config={}, save=None)
        self.model, self.adapter = TinyModel(cached), TinyAdapter()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)

    def _read_train_partition(self, step):
        return {"sample_id": torch.tensor([0])}


def module_factory():
    module = SimpleNamespace()

    class Trainer:
        def __init__(self, actor):
            self.actor, self.args = actor, actor.args
            self.rank, self.world, self.step = 0, 1, 0
            self.contract = {"global_batch_size": 1, "master_dtype": "fp32"}
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(actor.optimizer, lambda _: 1)
            self.sampler = SimpleNamespace(batch=lambda _: [0])
            self.dataset = [{"x": torch.tensor([0.3, -0.6])}]
            self.closed = False

        def close(self):
            self.closed = True

        def train(self, step):
            self.actor._read_train_partition(step)
            self.actor.optimizer.zero_grad(set_to_none=True)
            loss = self.actor.model(self.dataset[0])
            loss.backward()
            module.clip_sharded_gradients(list(self.actor.model.parameters()), 1, None)
            self.actor.optimizer.step()
            self.scheduler.step()
            self.actor.optimizer.zero_grad(set_to_none=True)
            self.step += 1
            module._report_sft_metrics(self.args, {"train/loss": float(loss.detach())})

    module.FSDPSFTTrainer = Trainer
    module.clip_sharded_gradients = lambda p, maximum, group: torch.nn.utils.clip_grad_norm_(p, maximum)
    module._report_sft_metrics = lambda args, values: None
    return module


def test_random_capture_preserves_values_rng_and_fails_at_bound():
    state = capture_rng()
    expected = (torch.randn(3), torch.rand(1), torch.randint(0, 4, (1,)))
    after = payload_digest(capture_rng())
    restore_rng(state)
    with RandomCapture() as capture:
        actual = (torch.randn(3), torch.rand(1), torch.randint(0, 4, (1,)))
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    assert payload_digest(capture_rng()) == after
    assert len(capture.records) == 3
    with pytest.raises(RuntimeError, match="bound exceeded"), RandomCapture(max_bytes=1):
        torch.randn(3)


def test_source_dtype_and_shape_are_checked_before_comparing():
    with pytest.raises(ValueError, match="Original.*FP32"):
        audit_lora_state({"model": {"lora_A.weight": torch.ones(2, dtype=torch.bfloat16)}})
    params, audit = audit_lora_state({"model": {"lora_A.weight": torch.ones(2)}})
    assert params["model.lora_A.weight"].dtype == torch.float64
    assert audit["source_dtype"] == "torch.float32"
    assert not compare({"p": torch.ones(2)}, {"p": torch.ones(1)})["passed"]


def test_actual_single_update_roundtrip_and_gate(monkeypatch, tmp_path):
    from two_gpu.diagnostics import ray_hook

    # Production delegates assignment to Relax's already-tested sampler. This
    # CPU toy has one fixed row; keep the actual prefill/restore implementation.
    monkeypatch.setattr(ray_hook, "assigned_ids", lambda trainer: [0])
    module = module_factory()
    install(module)
    original_clip = module.clip_sharded_gradients
    monkeypatch.setenv("DIAG_ROOT", str(tmp_path))
    monkeypatch.setenv("DIAG_MODEL_IDENTITY", "a" * 64)
    monkeypatch.setenv("DIAG_SOURCE_ID", "b" * 64)
    for phase in ("a1", "a2", "bcold", "bwarm"):
        if phase == "bcold":
            report = baseline_report(tmp_path)
            assert report["strict_acceptance"]
            atomic_json(tmp_path / "baseline.json", report)
        monkeypatch.setenv("DIAG_PHASE", phase)
        torch.manual_seed(42 if phase == "a1" else 999)
        trainer = module.FSDPSFTTrainer(TinyActor(phase.startswith("b")))
        trainer.train(0)
        assert module.clip_sharded_gradients is original_clip
        assert "_read_train_partition" not in vars(trainer.actor)
        with pytest.raises(ValueError, match="Only one"):
            trainer.train(1)
    report = full_report(tmp_path)
    assert report["strict_acceptance"]
    assert report["cache_payload_exact_with_checked_hits"]
    row = torch.load(tmp_path / "bwarm/trajectory-rank0000.pt", weights_only=False)
    assert row["random_outputs"] and all(t.dtype == torch.float32 for t in row["delta"].values())
    assert any(torch.count_nonzero(t) for t in row["delta"].values())
    row["random_outputs"][0]["tensor"].add_(1)
    torch.save(row, tmp_path / "bwarm/trajectory-rank0000.pt")
    assert not full_report(tmp_path)["passed"]
    with (tmp_path / "a2/receipt-rank0000.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="Baseline evidence changed"):
        require_baseline(tmp_path)


def test_restore_rejects_bf16_and_exception_restores_hooks(monkeypatch, tmp_path):
    module = module_factory()
    install(module)
    monkeypatch.setenv("DIAG_ROOT", str(tmp_path))
    monkeypatch.setenv("DIAG_PHASE", "a1")
    monkeypatch.setenv("DIAG_MODEL_IDENTITY", "a" * 64)
    monkeypatch.setenv("DIAG_SOURCE_ID", "b" * 64)
    trainer = module.FSDPSFTTrainer(TinyActor(False))
    state = snapshot_state(trainer)
    state["parameters"]["weight"] = state["parameters"]["weight"].bfloat16()
    with pytest.raises(ValueError, match="FP32 master"):
        restore_state(trainer, state)
    original_clip = module.clip_sharded_gradients
    original_report = module._report_sft_metrics
    trainer.dataset[0] = {"x": torch.ones(3)}
    with pytest.raises(RuntimeError):
        trainer.train(0)
    assert trainer.closed
    assert module.clip_sharded_gradients is original_clip
    assert module._report_sft_metrics is original_report
    assert not trainer.actor.model._forward_pre_hooks


def test_plain_trainer_after_diagnostic_ignores_removed_environment(monkeypatch, tmp_path):
    module = module_factory()
    install(module)
    monkeypatch.setenv("DIAG_ROOT", str(tmp_path))
    monkeypatch.setenv("DIAG_PHASE", "a1")
    monkeypatch.setenv("DIAG_MODEL_IDENTITY", "a" * 64)
    monkeypatch.setenv("DIAG_SOURCE_ID", "b" * 64)
    diagnostic = module.FSDPSFTTrainer(TinyActor(False))
    diagnostic.train(0)
    assert diagnostic._single_update_state.used
    for name in ("DIAG_PHASE", "DIAG_ROOT", "DIAG_MODEL_IDENTITY", "DIAG_SOURCE_ID"):
        monkeypatch.delenv(name)
    ordinary_actor = TinyActor(False)
    ordinary_actor.adapter = SimpleNamespace(family="ordinary")
    ordinary = module.FSDPSFTTrainer(ordinary_actor)
    assert not hasattr(ordinary, "_single_update_state")
    ordinary.train(0)
    assert ordinary.step == 1 and not ordinary.closed


def accumulated_module(*, omit_second=False, actual_ids=None):
    """A real CPU rank-local update with world2/B4 assignment and two micros.

    This tests the diagnostic hooks, not CUDA FSDP or inter-rank communication.
    The underlying production trainer supplies the actual all-reduced shards.
    """
    module = module_factory()
    base = module.FSDPSFTTrainer

    class Trainer(base):
        def __init__(self, actor):
            super().__init__(actor)
            self.rank, self.world = 0, 2
            self.contract["global_batch_size"] = 4
            self.sampler = SimpleNamespace(batch=lambda _: [0, 1, 2, 3])
            self.dataset = [{"x": torch.tensor([0.2 + i / 10, -0.4 - i / 10])} for i in range(4)]
            actor._read_train_partition = lambda step: {"sample_id": torch.tensor([0, 1, 2, 3])}

        def train(self, step):
            self.actor._read_train_partition(step)
            self.actor.optimizer.zero_grad(set_to_none=True)
            losses = []
            for sample_id in actual_ids if actual_ids is not None else [0] if omit_second else [0, 2]:
                loss = self.actor.model(self.dataset[sample_id]) / 2
                loss.backward()
                losses.append(loss.detach())
            module.clip_sharded_gradients(list(self.actor.model.parameters()), 1e9, None)
            self.actor.optimizer.step()
            self.scheduler.step()
            self.actor.optimizer.zero_grad(set_to_none=True)
            self.step += 1
            module._report_sft_metrics(self.args, {"train/loss": float(torch.stack(losses).sum())})

    module.FSDPSFTTrainer = Trainer
    return module


def multi_trainer(monkeypatch, tmp_path, *, omit_second=False, actual_ids=None):
    module = accumulated_module(omit_second=omit_second, actual_ids=actual_ids)
    install(module)
    monkeypatch.setenv("DIAG_ROOT", str(tmp_path))
    monkeypatch.setenv("DIAG_PHASE", "a1")
    monkeypatch.setenv("DIAG_MODEL_IDENTITY", "a" * 64)
    monkeypatch.setenv("DIAG_SOURCE_ID", "b" * 64)
    return module.FSDPSFTTrainer(TinyActor(False))


def test_two_micro_capture_matches_complete_accumulated_gradient_and_adam_delta(monkeypatch, tmp_path):
    trainer = multi_trainer(monkeypatch, tmp_path)
    reference = copy.deepcopy(trainer.actor.model)
    optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01)
    initial = reference.weight.detach().clone()
    before_rng = capture_rng()
    for sample_id in (0, 2):
        (reference(trainer.dataset[sample_id]) / 2).backward()
    gradient = reference.weight.grad.clone()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1e9)
    optimizer.step()
    expected_delta = reference.weight.detach() - initial
    after_rng = payload_digest(capture_rng())
    restore_rng(before_rng)
    trainer.train(0)
    row = torch.load(tmp_path / "a1/trajectory-rank0000.pt", weights_only=False)
    validate_coverage(row)
    assert torch.equal(row["gradients"]["weight"], gradient)
    assert torch.equal(row["delta"]["weight"], expected_delta)
    assert row["rng_after_sha256"] == after_rng
    assert [entry["sample_id"] for entry in row["effective_input_trace"]] == [0, 2]
    assert row["dataset_sample_ids"] == [0, 2]
    assert [entry["random_end"] - entry["random_start"] for entry in row["effective_input_trace"]] == [2, 2]
    assert row["gradient_capture_calls"] == 1
    # Equal A/B omissions must fail their own coverage validation.
    incomplete = copy.deepcopy(row)
    incomplete["effective_input_trace"].pop()
    incomplete["effective_input_sha256"].pop()
    with pytest.raises(ValueError, match="effective-batch microbatch"):
        validate_coverage(incomplete)
    incomplete = copy.deepcopy(row)
    incomplete["effective_input_trace"][1]["random_end"] = incomplete["effective_input_trace"][1]["random_start"]
    with pytest.raises(ValueError, match="noise/timestep"):
        validate_coverage(incomplete)


def test_omitted_second_micro_fails_before_optimizer_update(monkeypatch, tmp_path):
    trainer = multi_trainer(monkeypatch, tmp_path, omit_second=True)
    initial = trainer.actor.model.weight.detach().clone()
    with pytest.raises(ValueError, match="every effective-batch microbatch"):
        trainer.train(0)
    assert torch.equal(trainer.actor.model.weight.detach(), initial)
    assert trainer.closed and not trainer.actor.model._forward_pre_hooks
    assert not trainer.actor.model._forward_hooks
    assert json.loads((tmp_path / "a1/receipt-rank0000.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("actual_ids", [[0, 0], [2, 0]])
def test_same_count_duplicate_or_wrong_order_dataset_reads_fail(monkeypatch, tmp_path, actual_ids):
    trainer = multi_trainer(monkeypatch, tmp_path, actual_ids=actual_ids)
    original_dataset = trainer.dataset
    initial = trainer.actor.model.weight.detach().clone()
    with pytest.raises(ValueError, match="Observed dataset sample order"):
        trainer.train(0)
    assert trainer.dataset is original_dataset
    assert torch.equal(trainer.actor.model.weight.detach(), initial)
    assert trainer.closed and not trainer.actor.model._forward_pre_hooks
