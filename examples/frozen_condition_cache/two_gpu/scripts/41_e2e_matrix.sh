#!/bin/bash
# Select one bounded stage. No automatic 66-step numerical/full matrix rerun.
# STAGE=smoke-sn|smoke-qw|perf-sn|perf-qw|low-sn|low-qw ATTEMPT=<new label>
set -euo pipefail
: "${KIT:?}"; : "${WORK:?}"; : "${STAGE:?Choose a stage listed at the top of this script}"
source "$KIT/env.sh"
ATTEMPT="${ATTEMPT:-$(date -u +%Y%m%dT%H%M%S)-$$}"
[[ "$STAGE" =~ ^(smoke|perf|low)-(sn|qw)$ ]] || exit 2
suffix=${STAGE##*-}; model=sensenova
[[ "$suffix" == qw ]] && model=qwen_edit
mkdir -p "$WORK/logs"
run() {
  local label=$1 arm=$2 data=$3 steps=$4
  NAME="$label-$ATTEMPT" MODEL="$model" ARM="$arm" DATA="$data" STEPS="$steps" \
    bash "$KIT/scripts/40_e2e_ray.sh" 2>&1 | tee "$WORK/logs/$label-$ATTEMPT.log"
}
if [[ "$STAGE" == smoke-* ]]; then
  (( 8 % GLOBAL_BATCH == 0 )) || { echo "The global batch must divide debug8" >&2; exit 2; }
  run "smoke-$suffix-online" online debug8 "$((8/GLOBAL_BATCH))"
  run "smoke-$suffix-cache" cache debug8 "$((3*8/GLOBAL_BATCH))"
  python3 "$KIT/scripts/check_job.py" "$WORK/e2e/smoke-$suffix-cache-$ATTEMPT" --smoke
elif [[ "$STAGE" == perf-* ]]; then
  # Record the correctness evidence (or an explicit unresolved diagnostic context).
  : "${PERF_CONTEXT:?Set PERF_CONTEXT to the passing report path, or an explicit reason this is timing-only}"
  printf '%s\n' "$PERF_CONTEXT" > "$WORK/logs/$STAGE-$ATTEMPT.correctness-context.txt"
  (( 128 % GLOBAL_BATCH == 0 )) || exit 2
  steps=$((6*128/GLOBAL_BATCH))
  run "$suffix-high-p1-a" online high128 "$steps"
  run "$suffix-high-p1-b" cache high128 "$steps"
  run "$suffix-high-p2-b" cache high128 "$steps"
  run "$suffix-high-p2-a" online high128 "$steps"
  run "$suffix-high-p3-a" online high128 "$steps"
  run "$suffix-high-p3-b" cache high128 "$steps"
else
  : "${PERF_CONTEXT:?Set the correctness evidence or timing-only context}"
  printf '%s\n' "$PERF_CONTEXT" > "$WORK/logs/$STAGE-$ATTEMPT.correctness-context.txt"
  (( 1024 % GLOBAL_BATCH == 0 )) || exit 2
  run "$suffix-low-a" online low1024 "$((1024/GLOBAL_BATCH))"
  run "$suffix-low-b" cache low1024 "$((1024/GLOBAL_BATCH))"
fi
