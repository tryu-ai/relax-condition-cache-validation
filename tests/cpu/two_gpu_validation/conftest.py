import sys
from pathlib import Path


EXAMPLE = Path(__file__).resolve().parents[3] / "examples/frozen_condition_cache"
sys.path.insert(0, str(EXAMPLE))
