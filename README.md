# Relax frozen-condition cache validation

Standalone validation tools and recorded evidence for [Relax PR #557](https://github.com/redai-studio/Relax/pull/557), Task 17. The cache, model adapters, runtime lifecycle, and product regression tests are maintained in Relax. This repository holds the fixed experiment matrix, data preparation, numerical comparisons, timing and resource audits, and historical results.

The original directory layout is retained so the archived evidence and inspection commands remain usable. No model weights, dataset images, or private connection settings are included.

## Current shared-store validation

The [November 8 two-GPU results](examples/frozen_condition_cache/results/v2-two-gpu-20261108/README.md) contain 16 formal Ray SFT jobs, 3328 optimizer updates and four strict numerical reports. Mean steady full-loop throughput improves 38.30% for SenseNova and 13.01% for Qwen Edit. The single-pass low-reuse results, including regressions, remain visible. The models ran in separate hardware domains; each A/B pair used the same physical GPU pair.

```bash
python3 examples/frozen_condition_cache/results/v2-two-gpu-20261108/verify.py
python3 examples/frozen_condition_cache/results/v2-two-gpu-20261108/recompute.py
```

Use the [two-GPU harness](examples/frozen_condition_cache/two_gpu/README.md) and `sources.v2.lock.json` for this implementation at production commit `53a814afb7e9fc1b24e065c34cae451d2fbda316`. It exercises actual Ray/TQ SFT, so it needs a working Relax Ray environment. `FROZEN_CACHE_RELAX_SRC` selects a separate checkout; this repository does not vendor Relax or manage cloud resources.

With the selected checkout and validation dependencies, run all 61 current CPU tool regressions:

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest tests/cpu -q
```

The stored numerical reports passed the fixed 1%/strict-zero rules. Large original PT/DCP payloads are not public, so report inspection does not independently recompute tensor arithmetic. Public file manifests distinguish original and path-redacted hashes.

## Historical native-only results

These commands need only Python's standard library and do not launch training:

```bash
python3 examples/frozen_condition_cache/results/h200-20261103/verify.py
python3 examples/frozen_condition_cache/results/h200-20261103/recompute.py
```

- [H200 results](examples/frozen_condition_cache/results/h200-20261103/README.md): all 18 timing segments, two full numerical comparison reports, and 28 console logs and execution receipts.
- [RTX 3090 checks](examples/frozen_condition_cache/results/cuda-20261104/README.md): 14 small CUDA checks and their script and environment.
- [Benchmark report](docs/draft/task17-benchmark-en.md) · [中文报告](docs/draft/task17-benchmark-zh.md).

The recorded repeated-workload median throughput gains are +7.78% for SenseNova and +14.54% for Qwen Edit; low-reuse SenseNova regresses by 20.68%. The H200 runs used two concurrent world1 lanes sharing host bandwidth. The portable runner defaults to a serial, single-GPU matrix. Timing covers native SFT updates and excludes Ray/TQ orchestration, initialization, and checkpoint I/O.

The stored numerical reports cover loss, gradients, FP32 parameter updates, RNG state, and cached payloads. The public archive omits the large original tensor/checkpoint files, so inspecting the JSON reports does not recompute the tensor comparisons. Relocation does not create a new benchmark result.

## Select the historical native implementation

Use a separate checkout of the source revision recorded in `sources.lock.json`:

```bash
git clone https://github.com/tryu-ai/Relax.git Relax-source
git -C Relax-source checkout "$(python3 -c 'import json; print(json.load(open("sources.lock.json"))["commit"])')"
export FROZEN_CACHE_RELAX_SRC="$PWD/Relax-source"
python3 scripts/verify_source.py
```

The checkout contains the runtime implementation; this validation repository does not vendor or replace it. The source verifier checks the Git revision and hashes of the changed runtime files. New campaign receipts separately bind the validation and Relax source bytes, so resuming across a source change is rejected.

## Reproduce or test the historical native tools

Follow the [environment and execution instructions](examples/frozen_condition_cache/README.md). They install the pinned model-library environment used by the recorded recipe. Keep `FROZEN_CACHE_RELAX_SRC` set in that environment. Installing Relax's entire Ray/SGLang dependency stack is not required for this standalone harness.

From this repository root, after installing the validation dependencies:

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest tests/cpu/examples/frozen_condition_cache -q
```

To launch the explicit GPU comparison, use your own input/output directories and selected devices:

```bash
cd examples/frozen_condition_cache
python -m runner.campaign prepare --assets /path/to/assets
python -m runner.campaign validate --prepared /path/to/assets/prepared.json
python -m runner.campaign run \
  --prepared /path/to/assets/prepared.json \
  --work /path/to/results --devices 0 --world 1
python -m runner.campaign summarize --work /path/to/results
```

Model/data preparation downloads about 97 GB. The recipe defaults to a 64 GiB CPU cache and needs GPUs sized for the full models. No download or GPU run occurs merely by inspecting this repository.

## Provenance and scope

`MIGRATION.json` records the original public paths and hashes extracted from Relax commit `9e13cf0398df9b0c4ae80ac644d238d24be06a04`. The `results/` subtrees preserve every extracted byte. Portability changes select an explicit Relax checkout, separate source hash namespaces, and adapt the test bootstrap; they do not change model/data locks, numerical thresholds, or timing arithmetic.

See [LICENSE](LICENSE) and [NOTICE](NOTICE) for code attribution. Referenced models and datasets retain their own licenses.
