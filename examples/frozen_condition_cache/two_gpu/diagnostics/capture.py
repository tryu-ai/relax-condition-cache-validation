"""Correctness-only Ray SFT capture. Never install in a performance job."""

from __future__ import annotations

import copy
import hashlib
import inspect
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from .evidence import atomic_json, capture_rng, cpu_clone, local_tensor, payload_digest, restore_rng


def save_tensor_file(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise FileExistsError(f"Refusing to replace diagnostic evidence: {path}")
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("xb" if exclusive else "wb") as stream:
        torch.save(value, stream)
    temporary.replace(path)


def tree_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return cpu_clone(value)
    if isinstance(value, dict):
        return {key: tree_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [tree_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(tree_cpu(item) for item in value)
    return copy.deepcopy(value)


def trainable_parameters(model: Any) -> dict[str, Any]:
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if not parameters or any(local_tensor(p).dtype != torch.float32 for p in parameters.values()):
        raise ValueError("Diagnostic requires nonempty, original FP32 trainable masters")
    return parameters


def snapshot_state(trainer: Any) -> dict[str, Any]:
    parameters = trainable_parameters(trainer.actor.model)
    return {
        "parameters": {name: cpu_clone(p) for name, p in parameters.items()},
        "optimizer": tree_cpu(trainer.actor.optimizer.state_dict()),
        "scheduler": copy.deepcopy(trainer.scheduler.state_dict()),
        "rng": capture_rng(),
        "contract": copy.deepcopy(trainer.contract),
        "completed_steps": trainer.step,
        "global_shapes": {name: list(p.shape) for name, p in parameters.items()},
    }


def restore_state(trainer: Any, state: dict[str, Any]) -> None:
    if state["contract"] != trainer.contract or state["completed_steps"] != 0 or trainer.step != 0:
        raise ValueError("Initial model/data/optimizer/world/precision contract differs or cursor is nonzero")
    parameters = trainable_parameters(trainer.actor.model)
    if set(parameters) != set(state["parameters"]):
        raise ValueError("Trainable parameter set differs")
    with torch.no_grad():
        for name, parameter in parameters.items():
            target, source = local_tensor(parameter), state["parameters"][name]
            if source.dtype != torch.float32 or target.shape != source.shape:
                raise ValueError(f"Original FP32 master/shard shape differs: {name}")
            target.copy_(source.to(target.device))
    trainer.actor.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
    trainer.scheduler.load_state_dict(copy.deepcopy(state["scheduler"]))
    restore_rng(state["rng"])


def sample_digest(value: Any) -> str:
    """Effective post-prepare pixels/tensors, not merely source filenames."""

    def normalize(item: Any) -> Any:
        if hasattr(item, "mode") and hasattr(item, "size") and hasattr(item, "tobytes"):
            return {"mode": item.mode, "size": list(item.size), "pixels": item.tobytes()}
        if isinstance(item, dict):
            return {key: normalize(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return type(item)(normalize(child) for child in item)
        return item

    return payload_digest(normalize(value))


class RandomCapture(TorchDispatchMode):
    """Copy already-generated RNG outputs without generating or replacing any.

    Scope includes sampling inside the existing update (noise/sigma/timestep)
    and any other aten rand/randn/randint calls, labelled by operator. It does
    not claim to trace randomness internal to fused CUDA kernels.
    """

    def __init__(self, *, max_bytes: int = 256 << 20, max_calls: int = 128) -> None:
        super().__init__()
        self.max_bytes, self.max_calls = max_bytes, max_calls
        self.bytes = 0
        self.records: list[dict[str, Any]] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        name = str(func)
        base = name.split(".")[1] if name.startswith("aten.") else ""
        if base in {"rand", "randn", "rand_like", "randn_like", "randint", "randint_like"}:
            if not isinstance(result, torch.Tensor):
                raise TypeError(f"Unexpected random operator output: {name}")
            size = result.numel() * result.element_size()
            if self.bytes + size > self.max_bytes or len(self.records) >= self.max_calls:
                raise RuntimeError("Diagnostic RNG capture bound exceeded; evidence is incomplete")
            self.records.append({"operator": name, "tensor": cpu_clone(result)})
            self.bytes += size
        return result


class CacheAudit:
    def __init__(self, cache: Any) -> None:
        self.cache = cache
        self.get, self.put = cache.get, cache.put
        self.expected: dict[Any, str] = {}
        self.checked_hits = self.failed = self.writes = 0

        def put(key: Any, value: Any) -> Any:
            # Retain the online reference even if another rank won this write.
            self.expected[key] = payload_digest(value)
            self.writes += 1
            return self.put(key, value)

        def get(key: Any, *args: Any, **kwargs: Any) -> Any:
            value = self.get(key, *args, **kwargs)
            if value is not None:
                if key not in self.expected or payload_digest(value) != self.expected[key]:
                    self.failed += 1
                    raise AssertionError("Cache read lacks an exact online reference on this rank")
                self.checked_hits += 1
            return value

        cache.get, cache.put = get, put

    def close(self) -> None:
        self.cache.get, self.cache.put = self.get, self.put

    def result(self) -> dict[str, int]:
        return dict(writes=self.writes, checked_hits=self.checked_hits, failed=self.failed)


def execution_state(model: Any) -> dict[str, Any]:
    attention = []
    for name, module in model.named_modules():
        if "attn" not in name.lower() and "attention" not in type(module).__name__.lower():
            continue
        if len(attention) >= 16384:
            raise ValueError("Attention metadata bound exceeded")
        config = getattr(module, "config", None)
        processor = getattr(module, "processor", None)
        attention.append(
            {
                "name": name,
                "class": type(module).__module__ + "." + type(module).__qualname__,
                "training": module.training,
                "config_attn_implementation": getattr(config, "_attn_implementation", None),
                "attention_backend": str(getattr(module, "_attention_backend", None)),
                "processor": None
                if processor is None
                else type(processor).__module__ + "." + type(processor).__qualname__,
            }
        )
    cuda = torch.backends.cuda
    return {
        "scope": "configured module/processor/backend state; not a profiler trace of actual CUDA kernel dispatch",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "sdp_enabled": {
            name: getattr(cuda, name)()
            for name in ("flash_sdp_enabled", "math_sdp_enabled", "mem_efficient_sdp_enabled", "cudnn_sdp_enabled")
            if hasattr(cuda, name)
        },
        "attention": attention,
    }


def source_identity(trainer: Any) -> dict[str, str]:
    modules = [trainer.__class__, trainer.actor.model.__class__, trainer.actor.adapter.__class__]
    result = {}
    for item in modules:
        path = Path(inspect.getfile(item)).resolve()
        result[item.__module__ + "." + item.__qualname__] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("capture.py", "ray_hook.py", "adapters.py", "evidence.py", "compare.py"):
        result["diagnostics/" + name] = hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
    # Bind the actual source, independently of the checkout's Git status.
    source_root = Path(inspect.getfile(trainer.__class__)).resolve().parents[2]
    for directory in (source_root / "backends/fsdp", source_root / "models/sensenova"):
        for path in sorted(directory.glob("*.py")):
            result["relax/" + path.relative_to(source_root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in (
        "models/condition_cache.py",
        "models/qwen_image/sft_adapter.py",
        "engine/sft/image.py",
        "utils/env.py",
    ):
        path = source_root / name
        if path.is_file():
            result["relax/" + name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def failure(path: Path, error: BaseException) -> None:
    atomic_json(path, {"status": "failed", "error_type": type(error).__name__, "error": str(error)})
