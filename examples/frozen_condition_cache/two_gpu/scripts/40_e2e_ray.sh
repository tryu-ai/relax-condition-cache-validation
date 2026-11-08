#!/bin/bash
# One real Ray job. New NAME for each attempt; no old result/checkpoint is overwritten.
set -euo pipefail
: "${KIT:?}"; : "${WORK:?}"; : "${NAME:?}"; : "${MODEL:?}"; : "${ARM:?}"; : "${DATA:?}"; : "${STEPS:?}"
: "${RAY_ADDRESS:?Set the task-owned cluster GCS address}"
: "${RAY_DASHBOARD_ADDRESS:?Set the same cluster dashboard address}"
source "$KIT/env.sh"
require_two_gpu_ids "$MONITOR_GPU_IDS"
DET="${DET:-0}"; SAVE_INTERVAL="${SAVE_INTERVAL:-0}"
CACHE_BYTES="${CACHE_BYTES:-68719476736}"
[[ "$NAME" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid NAME" >&2; exit 2; }
[[ "$MODEL" == sensenova || "$MODEL" == qwen_edit ]] || exit 2
[[ "$ARM" == online || "$ARM" == cache ]] || exit 2
[[ "$DATA" == debug8 || "$DATA" == high128 || "$DATA" == low1024 ]] || exit 2
[[ "$GPUS" =~ ^[1-9][0-9]*$ && "$STEPS" =~ ^[1-9][0-9]*$ ]] || exit 2
OUT="$WORK/e2e/$NAME"
mkdir -p "$WORK/e2e" "$WORK/logs"
mkdir "$OUT" || { echo "NAME already exists; preserve it and use a new attempt name." >&2; exit 2; }
python3 "$KIT/scripts/source_receipt.py" --output "$OUT/source.json"
export DIAG_SOURCE_ID
DIAG_SOURCE_ID=$(python3 - "$OUT/source.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['source_id'])
PY
)
field() { python3 - "$ASSETS/prepared.json" "$@" <<'PY'
import json,sys
value=json.load(open(sys.argv[1]))
for key in sys.argv[2:]: value=value[key]
print(value)
PY
}
MODEL_PATH=$(field models "$MODEL" path)
MODEL_IDENTITY=$(field models "$MODEL" identity)
DATA_ROOT=$(field data root)
METADATA_PATH=$(field data "$DATA")
DATASET_SIZE=$(python3 - "$METADATA_PATH" <<'PY'
import json,sys
print(len(json.load(open(sys.argv[1]))))
PY
)
EXTRA=(--global-batch-size "$GLOBAL_BATCH" --rollout-batch-size "$GLOBAL_BATCH" --micro-batch-size "$MICRO_BATCH_SIZE")
PREFETCH=2
if [[ "$MODEL" == qwen_edit ]]; then
  ADAPTER=relax.models.qwen_image.sft_adapter.QwenImageEditSFTAdapter
  PREFETCH=0
else
  ADAPTER=relax.models.sensenova.adapter.SenseNovaSFTAdapter
fi
if [[ -n "${DIAG_PHASE:-}" ]]; then
  : "${DIAG_ROOT:?Set an absolute directory shared by these four diagnostic phases}"
  [[ "$STEPS" == 1 && "$DATA" == debug8 ]] || { echo "Diagnostic jobs use one update and debug8" >&2; exit 2; }
  [[ "$DIAG_ROOT" == /* ]] || exit 2
  export DIAG_MODEL_IDENTITY="$MODEL_IDENTITY"
  if [[ "$MODEL" == qwen_edit ]]; then
    ADAPTER=two_gpu.diagnostics.adapters.QwenImageEditDiagnosticAdapter
  else
    ADAPTER=two_gpu.diagnostics.adapters.SenseNovaDiagnosticAdapter
  fi
  PREFETCH=0
  DIAG_WARMUP_ITERS="${DIAG_WARMUP_ITERS:-0}"
  [[ "$DIAG_WARMUP_ITERS" =~ ^[0-9]+$ ]] || exit 2
  export DIAG_WARMUP_ITERS
  EXTRA+=(--lr-warmup-iters "$DIAG_WARMUP_ITERS")
fi
EXTRA+=(--model-adapter-path "$ADAPTER")
BYTES=0; [[ "$ARM" == cache ]] && BYTES=$CACHE_BYTES
export RELAX_SKIP_CLUSTER_CLEANUP=1
export RAY_NO_WAIT=1
RUNTIME_ENV_JSON=$(python3 - "$BYTES" "$RELAX_CONDITION_CACHE_DIR" "$MODEL_IDENTITY" <<'PY'
import json,os,sys
base=json.loads(os.environ.get('BASE_RUNTIME_ENV_JSON') or '{}')
env=base.setdefault('env_vars', {})
env.update(RELAX_CONDITION_CACHE_BYTES=sys.argv[1], RELAX_CONDITION_CACHE_DEVICE='shm',
           RELAX_CONDITION_CACHE_DIR=sys.argv[2], RELAX_CONDITION_CACHE_IDENTITY=sys.argv[3],
           CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONPATH=os.environ['PYTHONPATH'])
for key in ('DIAG_PHASE','DIAG_ROOT','DIAG_MODEL_IDENTITY','DIAG_SOURCE_ID','DIAG_DETERMINISTIC'):
    if key in os.environ: env[key]=os.environ[key]
print(json.dumps(base))
PY
)
export MODEL_PATH MODEL_IDENTITY DATA_ROOT METADATA_PATH DATASET_SIZE RUNTIME_ENV_JSON
export DET SAVE_INTERVAL PREFETCH BYTES OUT
python3 - <<'PY'
import hashlib,importlib.util,json,os
from pathlib import Path
import ray
from ray.job_submission import JobSubmissionClient
spec=importlib.util.spec_from_file_location('no17_gpu_preflight',Path(os.environ['KIT'],'scripts/gpu_preflight.py'))
preflight=importlib.util.module_from_spec(spec); spec.loader.exec_module(preflight)
preflight.check_idle(JobSubmissionClient(os.environ['RAY_DASHBOARD_ADDRESS']))
ray.init(address=os.environ['RAY_ADDRESS'], log_to_driver=False)
try:
    gpu_proof=preflight.verify(ray,os.environ['MONITOR_GPU_IDS'])
finally:
    ray.shutdown()
Path(os.environ['OUT'],'gpu-preflight.json').write_text(json.dumps(gpu_proof,indent=2)+'\n')
config={k.lower():os.environ[k] for k in ('NAME','MODEL','ARM','DATA','MODEL_PATH','MODEL_IDENTITY','METADATA_PATH')}
config.update(name=os.environ['NAME'], dataset_size=int(os.environ['DATASET_SIZE']),steps=int(os.environ['STEPS']),
              gpus=int(os.environ['GPUS']),global_batch=int(os.environ['GLOBAL_BATCH']),
              rollout_batch=int(os.environ['GLOBAL_BATCH']),micro_batch_size=int(os.environ['MICRO_BATCH_SIZE']),
              gradient_accumulation_steps=int(os.environ['GLOBAL_BATCH'])//(int(os.environ['GPUS'])*int(os.environ['MICRO_BATCH_SIZE'])),
              protocol_scope='real_ray_world2_global_batch4',deterministic=int(os.environ['DET']),
              save_interval=int(os.environ['SAVE_INTERVAL']),image_prefetch_depth=int(os.environ['PREFETCH']),
              image_prefetch_workers=2,max_pixels=int(os.environ.get('MAX_PIXELS','1048576')),
              requested_attention_implementation=os.environ.get('ATTENTION_IMPLEMENTATION','flash_attention_2'),
              seed=42,lora_rank=int(os.environ.get('LORA_RANK','32')),learning_rate=float(os.environ.get('LR','1e-4')),
              cache_bytes=int(os.environ['BYTES']),cache_root=os.environ['RELAX_CONDITION_CACHE_DIR'],
              cache_metrics_scope='rank0_branch_node_store',
              gpu_preflight=gpu_proof,
              model_path=os.environ['MODEL_PATH'],metadata=os.environ['METADATA_PATH'],
              metadata_sha256=hashlib.sha256(Path(os.environ['METADATA_PATH']).read_bytes()).hexdigest(),
              runtime_env=json.loads(os.environ['RUNTIME_ENV_JSON']),monitor_gpu_ids=os.environ['MONITOR_GPU_IDS'],
              diagnostic_phase=os.environ.get('DIAG_PHASE'),
              lr_warmup_iters=int(os.environ.get('DIAG_WARMUP_ITERS','0')) if os.environ.get('DIAG_PHASE') else 200)
Path(os.environ['OUT'],'config.json').write_text(json.dumps(config,indent=2)+'\n')
PY
SAVE_DIR_FOR_JOB=""
if (( SAVE_INTERVAL > 0 )); then
  mkdir -p "$WORK/ckpt"
  mkdir "$WORK/ckpt/$NAME"
  SAVE_DIR_FOR_JOB="$WORK/ckpt/$NAME"
fi
mkdir -p "$RELAX_CONDITION_CACHE_DIR"
df -h "$RELAX_CONDITION_CACHE_DIR" > "$OUT/cache-filesystem.txt"
ls -la "$RELAX_CONDITION_CACHE_DIR" > "$OUT/shm-before.txt"
bash "$KIT/scripts/50_monitor.sh" "$OUT/monitor.csv" "$RELAX_CONDITION_CACHE_DIR" &
MONITOR=$!
JOB_ID=""; JOB_FINISHED=0
cleanup() {
  local status=$?
  kill "$MONITOR" 2>/dev/null || true
  if [[ -n "$JOB_ID" && "$JOB_FINISHED" != 1 ]]; then
    ray job stop --address "$RAY_DASHBOARD_ADDRESS" "$JOB_ID" >> "$OUT/owned-job-cleanup.log" 2>&1 || true
  fi
  return "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
SUBMIT=$(python3 -c 'import time; print(time.time())')
cd "$FROZEN_CACHE_RELAX_SRC"
env MODE=lora GPUS="$GPUS" STEPS="$STEPS" CONDITIONING_IMAGES=1 \
  ATTENTION_IMPLEMENTATION="${ATTENTION_IMPLEMENTATION:-flash_attention_2}" DETERMINISTIC_ATTENTION="$DET" \
  IMAGE_PREFETCH_DEPTH="$PREFETCH" IMAGE_PREFETCH_WORKERS=2 TEST_RUNTIME_ENV_RECEIPT="$OUT/submitted-runtime-env.json" \
  SAVE_DIR="$SAVE_DIR_FOR_JOB" SAVE_INTERVAL="$SAVE_INTERVAL" LOAD_DIR= \
  bash "$KIT/scripts/submit_with_env.sh" "${EXTRA[@]}" \
  2>&1 | tee "$OUT/submit.log"
JOB_ID=$(python3 - "$OUT/submit.log" <<'PY'
import re,sys
matches=re.findall(r'raysubmit_[A-Za-z0-9]+',open(sys.argv[1]).read())
if not matches: raise SystemExit('No job id found; inspect submit.log before another attempt')
print(matches[0])
PY
)
printf '%s\n' "$JOB_ID" > "$OUT/job-id.txt"
if timeout "${JOB_TIMEOUT_SECONDS:-5400}" ray job logs --address "$RAY_DASHBOARD_ADDRESS" -f "$JOB_ID" > "$OUT/job.log" 2>&1; then
  :
else
  tail_status=$?
  if [[ "$tail_status" == 124 ]]; then
    echo "Owned job exceeded JOB_TIMEOUT_SECONDS; stopping only $JOB_ID" >&2
    ray job stop --address "$RAY_DASHBOARD_ADDRESS" "$JOB_ID" >> "$OUT/owned-job-cleanup.log" 2>&1 || true
  fi
fi
python3 - "$RAY_DASHBOARD_ADDRESS" "$JOB_ID" "$SUBMIT" "$OUT/result.json" <<'PY'
import json,sys,time
from pathlib import Path
from ray.job_submission import JobSubmissionClient
addr,job,submit,out=sys.argv[1:]
client=JobSubmissionClient(addr)
for _ in range(120):
    info=client.get_job_info(job)
    status=str(info.status).split('.')[-1]
    if status in ('SUCCEEDED','FAILED','STOPPED'): break
    time.sleep(5)
result={'status':status,'job_id':job,'submit_wall':float(submit),
        'start_time_ms':info.start_time,'end_time_ms':info.end_time,'message':info.message}
Path(out).write_text(json.dumps(result,indent=2)+'\n')
print(result)
if status!='SUCCEEDED': raise SystemExit(1)
PY
JOB_FINISHED=1
sleep 5
kill "$MONITOR" 2>/dev/null || true
ls -laR "$RELAX_CONDITION_CACHE_DIR" > "$OUT/shm-after.txt" 2>&1 || true
du -sb "$RELAX_CONDITION_CACHE_DIR" >> "$OUT/shm-after.txt" 2>&1 || true
python3 - "$RELAX_CONDITION_CACHE_DIR" "$OUT/cache-exit.json" <<'PY'
import json,stat,sys
from pathlib import Path
root=Path(sys.argv[1]); entries=[]
if root.exists():
    for item in sorted(root.iterdir()):
        try: info=item.lstat()
        except FileNotFoundError: continue
        kind=('symlink' if stat.S_ISLNK(info.st_mode) else 'directory' if stat.S_ISDIR(info.st_mode)
              else 'file' if stat.S_ISREG(info.st_mode) else 'other')
        entries.append(dict(name=item.name,type=kind,bytes=info.st_size))
Path(sys.argv[2]).write_text(json.dumps(dict(root=str(root),exists=root.exists(),entries=entries),indent=2)+'\n')
PY
python3 "$KIT/scripts/check_job.py" "$OUT"
