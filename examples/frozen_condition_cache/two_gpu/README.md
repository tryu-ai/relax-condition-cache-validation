# Two-GPU SFT validation

This is the validation harness used for the November 8 shared-cache results. It runs real Relax Ray SFT jobs, checks one-update numerical equivalence, and records full-loop timing and resources. It contains no cloud account, rental controller, remote connection, or production Relax source.

The earlier `runner.campaign` harness remains available for the historical native-only results. **This harness needs a working Relax Ray SFT environment**, including Ray, TransferQueue and the model dependencies. The native-only requirements in the parent directory are not a complete Ray environment.

## Source and environment

From this repository root, select the published production implementation:

```bash
git clone https://github.com/tryu-ai/Relax.git Relax-source
git -C Relax-source checkout 53a814afb7e9fc1b24e065c34cae451d2fbda316
export FROZEN_CACHE_RELAX_SRC="$PWD/Relax-source"
python3 scripts/verify_source.py --lock sources.v2.lock.json
```

The eight runtime files at this commit are byte-identical to the measured implementation. `sources.v2.lock.json` keeps that runtime identity separate from the historical combined benchmark source hash. A new source receipt checks the actual Git HEAD and runtime bytes, records both source trees, and reports a dirty worktree.

Recorded versions: Python 3.12, Torch 2.9.0+cu128, torchvision 0.24.0+cu128, transformers 5.10.1, diffusers 0.37.1, peft 0.20.0, Ray 2.58.0 and flash-attn 2.8.3. Use a compatible Relax environment; this repository does not install or replace CUDA or drivers. The two full models require GPUs with sufficient loading and training memory; the recorded runs used two 84 GB RTX6000D GPUs.

Set your own paths and devices from `examples/frozen_condition_cache/`:

```bash
cd examples/frozen_condition_cache
export WORK=/path/to/new-results
export ASSETS=/path/to/assets
export MONITOR_GPU_IDS=0,1
export CUDA_VISIBLE_DEVICES="$MONITOR_GPU_IDS"
source two_gpu/env.sh
```

Paths are canonicalized before jobs change directory. GPU indices or UUIDs are explicit. Connect to a single-node Ray cluster exposing exactly this pair, using your own addresses:

```bash
export RAY_ADDRESS=127.0.0.1:6379
export RAY_DASHBOARD_ADDRESS=http://127.0.0.1:8265
```

The preflight checks the two physical UUIDs against Ray workers and rejects an occupied cluster. The harness stops only its submitted job on interruption; it does not run global cluster cleanup.

## Prepare and check

```bash
bash "$KIT/scripts/20_prepare_assets.sh"
python -m runner.campaign validate --prepared "$ASSETS/prepared.json"
```

Preparation reuses the parent pinned asset manifest and MagicBrush export: debug8, high128 and low1024 are disjoint. Both models and source Parquet require about 97 GB before image export. Existing complete, verified files can be reused.

Run the four controlled Ray trajectories for each model. The baseline A1/A2 gate runs before either cached trajectory:

```bash
MODEL=sensenova ATTEMPT=sn-check DIAG_ROOT="$WORK/ray-single/sn-check" \
  bash "$KIT/scripts/35_ray_single_update.sh"
MODEL=qwen_edit ATTEMPT=qw-check DIAG_ROOT="$WORK/ray-single/qw-check" \
  bash "$KIT/scripts/35_ray_single_update.sh"
```

These use world2/globalB4/micro1/accumulation2 and one optimizer update. The adapter opt-in captures the actual full batch, accumulated/reduced gradient and FP32 parameter update. Loss and per-tensor gradient/update relative error must be at most 1%; zero references require exact zero and all quantities must be finite. Deterministic baseline components stay exact; an unresolved nondeterministic baseline does not become a strict pass. Cache payloads keep their dtype and exact bytes. Raw trajectory files remain in your result directory.

The separate native trainer check uses world2/**globalB8**, four microbatches per rank, and its own fixed seed/SDPA contract. It does not measure Ray/TQ performance:

```bash
ATTEMPT=native-check bash "$KIT/scripts/30_numerics_native.sh"
```

## Measure both workloads

After the matching model's strict numerical report passes, run stages serially. Each high stage runs three A/B pairs in AB/BA/AB order: six epochs of high128, 192 updates per job. Low runs one pass of low1024, 256 updates per job, one A/B pair.

```bash
STAGE=perf-sn ATTEMPT=sn-high PERF_CONTEXT="$WORK/ray-single/sn-check/compare.json" \
  bash "$KIT/scripts/41_e2e_matrix.sh"
STAGE=low-sn ATTEMPT=sn-low PERF_CONTEXT="$WORK/ray-single/sn-check/compare.json" \
  bash "$KIT/scripts/41_e2e_matrix.sh"
STAGE=perf-qw ATTEMPT=qw-high PERF_CONTEXT="$WORK/ray-single/qw-check/compare.json" \
  bash "$KIT/scripts/41_e2e_matrix.sh"
STAGE=low-qw ATTEMPT=qw-low PERF_CONTEXT="$WORK/ray-single/qw-check/compare.json" \
  bash "$KIT/scripts/41_e2e_matrix.sh"
python "$KIT/scripts/60_summarize.py" --work "$WORK"
```

Use a fresh `WORK` or attempt for another run; existing job output is never overwritten. `JOB_TIMEOUT_SECONDS` bounds a submitted job. Do not mix machines, software, resolution, data order or batch settings within a pair. Ray performance fixes B4/LoRA rank 32/LR 1e-4/seed 42/max_pixels 1048576/FA2/warmup 200. Existing SenseNova image prefetch remains 2; Qwen remains 0. Trainable masters and the compared updates are FP32.

The SHM store's 64 GiB limit covers **committed entry-file sizes**, including their format headers/alignment. Temporary writes, reader staging and filesystem/directory overhead are reported separately; the limit is not a hard bound on every byte of the directory or system SHM. Rank0 branch/eviction counters are labelled as such, rather than presented as a node-wide event total.

Recorded results and standard-library inspection commands: [November 8 evidence](../results/v2-two-gpu-20261108/README.md). No GPU is used by those inspection commands.
