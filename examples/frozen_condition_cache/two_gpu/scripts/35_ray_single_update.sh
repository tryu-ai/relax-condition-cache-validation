#!/bin/bash
# Four real Ray jobs, one optimizer update each. Baseline gate precedes cache phases.
set -euo pipefail
: "${KIT:?}"; : "${WORK:?}"
source "$KIT/env.sh"
MODEL="${MODEL:-qwen_edit}"
[[ "$MODEL" == qwen_edit || "$MODEL" == sensenova ]] || exit 2
ATTEMPT="${ATTEMPT:-$(date -u +%Y%m%dT%H%M%S)-$$}"
export DIAG_ROOT="${DIAG_ROOT:-$WORK/ray-single/$MODEL-$ATTEMPT}"
export DIAG_DETERMINISTIC="${DIAG_DETERMINISTIC:-1}"
mkdir -p "$(dirname "$DIAG_ROOT")"
mkdir "$DIAG_ROOT" || { echo "Use a new DIAG_ROOT for every attempted protocol" >&2; exit 2; }
python3 "$KIT/scripts/source_receipt.py" --output "$DIAG_ROOT/source.json"
for phase in a1 a2 bcold bwarm; do
  if [[ "$phase" == bcold ]]; then
    python3 -m two_gpu.diagnostics.compare --root "$DIAG_ROOT" --baseline-only \
      2>&1 | tee "$DIAG_ROOT/baseline-compare.log"
  fi
  arm=online; [[ "$phase" == b* ]] && arm=cache
  NAME="single-$MODEL-$ATTEMPT-$phase" MODEL="$MODEL" ARM="$arm" DATA=debug8 STEPS=1 \
    DET="${DET:-1}" DIAG_PHASE="$phase" SAVE_INTERVAL=0 \
    bash "$KIT/scripts/40_e2e_ray.sh" 2>&1 | tee "$DIAG_ROOT/$phase.launch.log"
done
python3 -m two_gpu.diagnostics.compare --root "$DIAG_ROOT" 2>&1 | tee "$DIAG_ROOT/compare.log"
