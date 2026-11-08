# Two-GPU shared-cache results — November 8, 2026

**16 real Ray SFT performance jobs, 3328 optimizer updates, and four accepted numerical comparisons.** Both models use the same bounded shared-store implementation. SenseNova and Qwen Edit achieve 38.30% and 13.01% mean steady-state throughput gains over their online baselines.

| Model / hardware domain | High steady throughput gain | Full training-loop gain | Whole-job time-ratio gain | High pairs |
| --- | ---: | ---: | ---: | ---: |
| SenseNova / host-a | +38.30% ±0.29pp | +29.74% ±0.09pp | +26.25% ±0.08pp | 3 |
| Qwen Edit / host-b | +13.01% ±0.15pp | +10.58% ±0.09pp | +9.15% ±1.23pp | 3 |

These are means and sample standard deviations of `online_time/cache_time−1`. A throughput gain is not the same percentage reduction in elapsed time. The complete train-loop measurement includes data preparation, condition loading, forward, backward and optimizer updates. Whole-job time additionally includes initialization and teardown.

| Single-pass low-reuse workload | Full training-loop gain | Whole-job gain | Pairs |
| --- | ---: | ---: | ---: |
| SenseNova | −1.05546% | −1.28237% | 1 |
| Qwen Edit | +0.37291% | +0.35348% | 1 |

Low reuse has no hot epoch; n=1 has no sample standard deviation. Qwen's near-zero difference is not evidence of a stable speedup or cache payback: its recorded low workload has zero hits. Negative SenseNova results are retained.

High's first 32 updates form the cold epoch; updates 33–192 form the steady window. Sustained cumulative timing crossovers (the parser's `payback_step`) occur at updates 33/33/34 for SenseNova and 8/33/3 for Qwen. Qwen's updates 8 and 3 are still in the cold phase, so these crossings include run-to-run timing differences and do not establish cache savings before the first hit. Per-pair cold-epoch seconds and full-job overhead are retained in `recomputed-metrics.json` and the recomputation output. These are measured full-loop/job intervals, not isolated encoder kernel timings. A low-workload timing crossover with zero cache hits is not interpreted as proven caching payback.

## Inspect without a GPU

From the validation repository root:

```bash
python3 examples/frozen_condition_cache/results/v2-two-gpu-20261108/verify.py
python3 examples/frozen_condition_cache/results/v2-two-gpu-20261108/recompute.py
```

`verify.py` checks all 298 public source-evidence members and their decoded hashes, all 16 complete timing jobs, and the four stored strict reports. `recompute.py` runs the unchanged measured timing summarizer on temporary copies of the public logs. It never trains or modifies these recorded files.

## Recipe and provenance

Both hardware domains used two 84 GB RTX6000D GPUs and Torch 2.9.0+cu128. Each model's A/B pairs ran serially on the same physical pair. Host-a contains SenseNova performance and both models' Ray/native numerical checks; host-b contains Qwen performance. The full numerical checks were not rerun on host-b. Stable public host/GPU aliases preserve those boundaries.

Ray performance: world2/globalB4/micro1/accumulation2, LoRA rank 32, LR 1e-4, seed 42, max_pixels 1048576, FA2 requested, warmup 200, no checkpoint saving. SenseNova prefetch depth 2/workers 2 is retained; Qwen depth 0/workers 2 is retained. High128 runs six epochs, 192 updates/job, AB/BA/AB; low1024 runs once, 256 updates/job, A/B. Each model contributes 1664 formal updates.

Models are pinned to SenseNova-U1-8B-MoT `bfa9b436503cb8aed4f2bc60e3236710cc77468d` and initial Qwen-Image-Edit `ac7f9318f633fc4b5778c59367c8128225f1e3de`; MagicBrush revision `6c254560879df7851233ea0fd2ae2a62764d0ef2`. The public asset manifest retains the exact file hashes and disjoint debug8/high128/low1024 selection rule.

The recorded combined source identity is `b1b380451750b009dcf3833829e8c234b1ea9c95598b450e0384158c46f604d0`, a content digest rather than a Git commit. Its eight runtime files are byte-identical at production commit `53a814afb7e9fc1b24e065c34cae451d2fbda316`. See [provenance.json](provenance.json), [source lock](../../../../sources.v2.lock.json), and the [portable harness](../../two_gpu/README.md). Namespaces and path handling changed when extracting the harness, so new harness receipts have their own source identity.

## Numerical and resource scope

The two Ray reports use B4; the two supplemental native reports use B8. They cover A1/A2, cold and warm cache, complete accumulated/reduced gradients, FP32 updates, RNG/input identities, payload equality, finite values and the fixed 1%/strict-zero rules. A1/A2 establishes the repeat baseline. The original failed Qwen diagnostic is retained under `history/`; its bounded diagnostic metadata capacity was repaired before the accepted rerun.

The cache limit is 64 GiB of **committed entry files**, including headers/alignment. Temporary writes, reader staging, filesystem overhead and whole-system SHM are separate. Monitor CSV files retain host RAM/SHM/global GPU readings; logs retain allocated-memory and cache counters. SenseNova low's 305 evictions are the rank0-reported count, not a proven total across all ranks. Exit snapshots retain lifecycle cleanup evidence.

## Public files and omissions

- `performance/`:16 full small job evidence sets, including original step logs, configuration, source identity, completion, resource monitoring and cache-exit snapshots.
- `correctness/`: four full compare reports, Ray repeat baselines and rank receipts, native result/lifecycle receipts and phase logs. Large JSON records use deterministic gzip.
- `EVIDENCE-MANIFEST.json`: original file SHA, public decoded SHA and stored-file SHA. Only operational paths/hostnames/UUIDs/private node addresses were replaced with role placeholders; numeric data, times and original input SHA identities were preserved.
- `tensor-identities-not-published.json`: identities of 56 previously verified private PT files. **The 14.366 GB tensor payloads and 16 DCP files are not public.** Stored report checks cannot independently recompute tensor arithmetic from hashes. No complete training checkpoint is claimed here.

No model/data weights, rental controller, billing record, SSH settings, private environment script or account image is included. The98 historical result files remain unchanged in their separate directories; these results evaluate the newer shared-store implementation, not the earlier process-local LRU alone.
