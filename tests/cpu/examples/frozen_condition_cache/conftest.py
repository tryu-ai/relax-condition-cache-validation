import importlib
import sys
from pathlib import Path


BENCHMARK_REPO = Path(__file__).resolve().parents[4]
KIT = BENCHMARK_REPO / "examples" / "frozen_condition_cache"
sys.path.insert(0, str(KIT))

sys.path.insert(0, str(importlib.import_module("runner.common").relax_source_root()))
