#!/bin/bash
# Supplemental native one-update check, separately labelled from Ray/global-batch-4 evidence.
set -euo pipefail
: "${KIT:?}"; : "${WORK:?}"
: "${CUDA_VISIBLE_DEVICES:?Set only the authorized, currently free GPU indices}"
source "$KIT/env.sh"
require_two_gpu_ids "$MONITOR_GPU_IDS"
require_two_gpu_ids "$CUDA_VISIBLE_DEVICES"
IFS=, read -r first second <<< "$MONITOR_GPU_IDS"
[[ "$CUDA_VISIBLE_DEVICES" == "$first,$second" || "$CUDA_VISIBLE_DEVICES" == "$second,$first" ]] || {
  echo "CUDA_VISIBLE_DEVICES must select the same two physical IDs as MONITOR_GPU_IDS." >&2
  exit 2
}
CONFIGS="${CONFIGS:-sensenova:$GPUS:shm qwen_edit:$GPUS:shm}"
ATTEMPT="${ATTEMPT:-$(date -u +%Y%m%dT%H%M%S)-$$}"
OUT="$WORK/numerics/$ATTEMPT"
mkdir -p "$WORK/numerics"
mkdir "$OUT"
python3 "$KIT/scripts/source_receipt.py" --output "$OUT/source.json"
python3 - "$OUT/protocol.json" <<'PY'
import json,os,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps(dict(
    schema=1, protocol_scope='supplemental_native_world2_global_batch8',
    world=2, global_batch=8, micro_batch_size=1, gradient_accumulation_steps_per_rank=4,
    monitor_gpu_ids=os.environ['MONITOR_GPU_IDS'], cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
),indent=2)+'\n')
PY
cd "$FROZEN_CACHE_EXAMPLE_ROOT"
field() { python3 - "$ASSETS/prepared.json" "$@" <<'PY'
import json,sys
value=json.load(open(sys.argv[1]))
for key in sys.argv[2:]: value=value[key]
print(value)
PY
}
DATA_ROOT=$(field data root); METADATA=$(field data debug8)
for config in $CONFIGS; do
  IFS=: read -r model world storage <<<"$config"
  [[ "$model" == sensenova || "$model" == qwen_edit ]] || exit 2
  [[ "$world" == 2 ]] || { echo "This native protocol requires world=2 and global batch=8." >&2; exit 2; }
  [[ "$storage" == cpu || "$storage" == shm ]] || exit 2
  adapter=relax.models.sensenova.adapter.SenseNovaSFTAdapter
  [[ "$model" == qwen_edit ]] && adapter=relax.models.qwen_image.sft_adapter.QwenImageEditSFTAdapter
  model_path=$(field models "$model" path); identity=$(field models "$model" identity)
  base="$OUT/$model-w$world-$storage"; mkdir "$base"
  for trajectory in a1 a2 bcold bwarm; do
    bytes=0
    if [[ "$trajectory" == b* ]]; then
      divisor=$world; [[ "$storage" == shm ]] && divisor=1
      bytes=$((64 * 1024 * 1024 * 1024 / divisor))
    fi
    cache_root="${RELAX_CONDITION_CACHE_DIR}-native-$ATTEMPT-$model-$world-$trajectory"
    RELAX_CONDITION_CACHE_DIR="$cache_root" timeout "${NATIVE_TIMEOUT:-7200}" \
      python3 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$world" \
      -m two_gpu.experiments.native_runner --mode correctness --trajectory "$trajectory" \
      --adapter "$adapter" --model-path "$model_path" --metadata "$METADATA" --dataset-size 8 \
      --data-root "$DATA_ROOT" --run-dir "$base/$trajectory" --initial-state-dir "$base/initial" \
      --world "$world" --global-batch 8 --cache-identity "$identity" --cache-device "$storage" --cache-bytes "$bytes" \
      2>&1 | tee "$base/$trajectory.log"
    ls -la "$cache_root" > "$base/$trajectory.cache-after.txt" 2>&1 || true
  done
  python3 -m two_gpu.experiments.compare --a1 "$base/a1" --a2 "$base/a2" --bcold "$base/bcold" --bwarm "$base/bwarm" \
    --output "$base/compare.json" 2>&1 | tee "$base/compare.log"
  python3 - "$base/compare.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
if not (r.get('passed') and r.get('strict_acceptance') and r.get('cache_payload_exact_with_checked_hits')):
    raise SystemExit('Native comparison not strictly accepted; preserve and diagnose')
print('Native strict check passed:',sys.argv[1])
PY
done
