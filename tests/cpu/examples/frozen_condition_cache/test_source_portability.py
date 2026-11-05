import shutil

import pytest
from runner import campaign
from runner.common import relax_source_root


def test_production_source_is_explicit(monkeypatch, tmp_path):
    monkeypatch.delenv("FROZEN_CACHE_RELAX_SRC", raising=False)
    with pytest.raises(RuntimeError, match="FROZEN_CACHE_RELAX_SRC"):
        relax_source_root()
    monkeypatch.setenv("FROZEN_CACHE_RELAX_SRC", str(tmp_path))
    with pytest.raises(ValueError, match="not a Relax source checkout"):
        relax_source_root()


def test_source_identity_covers_both_roots_without_binding_absolute_paths(monkeypatch, tmp_path):
    benchmark = tmp_path / "validation"
    production = tmp_path / "separate-production"
    kit_relative = "examples/frozen_condition_cache"
    for root, relative in [
        (benchmark, f"{kit_relative}/experiments/evidence.py"),
        (benchmark, f"{kit_relative}/runner/campaign.py"),
        (production, "relax/__init__.py"),
        (production, "relax/backends/fsdp/runtime.py"),
        (production, "relax/models/sensenova/adapter.py"),
        (production, "relax/models/condition_cache.py"),
        (production, "relax/models/qwen_image/sft_adapter.py"),
    ]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# original\n")
    monkeypatch.setattr(campaign, "BENCHMARK_REPO", benchmark)
    monkeypatch.setattr(campaign, "KIT", benchmark / kit_relative)
    monkeypatch.setenv("FROZEN_CACHE_RELAX_SRC", str(production))
    original = campaign.source_identity()
    runtime = production / "relax/backends/fsdp/runtime.py"
    runtime.write_text("# changed production\n")
    assert campaign.source_identity() != original
    runtime.write_text("# original\n")
    evidence = benchmark / kit_relative / "experiments/evidence.py"
    evidence.write_text("# changed benchmark\n")
    assert campaign.source_identity() != original
    evidence.write_text("# original\n")
    moved_benchmark = tmp_path / "moved-validation"
    moved_production = tmp_path / "moved-production"
    shutil.copytree(benchmark, moved_benchmark)
    shutil.copytree(production, moved_production)
    monkeypatch.setattr(campaign, "BENCHMARK_REPO", moved_benchmark)
    monkeypatch.setattr(campaign, "KIT", moved_benchmark / kit_relative)
    monkeypatch.setenv("FROZEN_CACHE_RELAX_SRC", str(moved_production))
    assert campaign.source_identity() == original
