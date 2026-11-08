# Source and changes

- `evidence.py` is copied unchanged from
  `tryu-ai/relax-condition-cache-validation`, commit
  `f48e36655a338fd1dd1167ecb382e607dcd0ca8d`, path
  `examples/frozen_condition_cache/experiments/evidence.py`.
  Its fixed 1%/zero/finite/RNG and strict-baseline policy is reused.
- `compare_lora.py` derives from the delivered
  `kit-four/scripts/70_compare_lora.py`. This version rejects non-FP32 original
  LoRA tensors, records their original dtype and shapes, and rejects mismatched
  shapes before FP64 comparison arithmetic.
- The remaining files add external, opt-in diagnostic adapter hooks and CPU
  tests. Production Relax source is not modified by these files.

Original project: <https://github.com/redai-studio/Relax>.
Validation project: <https://github.com/tryu-ai/relax-condition-cache-validation>.
Copyright (c) 2026 Relax Authors. All Rights Reserved.
Distributed under the included Apache License 2.0. Models and datasets retain
their respective licenses and are not included here.
