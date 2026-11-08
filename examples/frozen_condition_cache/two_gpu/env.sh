#!/bin/bash
# Explicit local paths; no remote, billing, deployment, or global cluster cleanup.
: "${FROZEN_CACHE_RELAX_SRC:?Set a separate Relax checkout}"
: "${WORK:?Set a fresh writable result directory}"
FROZEN_CACHE_RELAX_SRC="$(cd -- "$FROZEN_CACHE_RELAX_SRC" && pwd)" || return 2
WORK="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$WORK")" || return 2
export KIT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export FROZEN_CACHE_EXAMPLE_ROOT="$(dirname "$KIT")"
ASSETS="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "${ASSETS:-$WORK/assets}")" || return 2
export GPUS="${GPUS:-2}" GLOBAL_BATCH="${GLOBAL_BATCH:-4}" MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
if [[ "$GPUS" != 2 || "$GLOBAL_BATCH" != 4 || "$MICRO_BATCH_SIZE" != 1 ]]; then
  echo 'Ray protocol requires GPUS=2, GLOBAL_BATCH=4, MICRO_BATCH_SIZE=1.' >&2
  return 2
fi
export KIT WORK ASSETS FROZEN_CACHE_RELAX_SRC FROZEN_CACHE_EXAMPLE_ROOT
export MONITOR_GPU_IDS="${MONITOR_GPU_IDS:-}"
require_two_gpu_ids() {
  local -a ids
  IFS=, read -r -a ids <<< "${1:-}"
  if [[ ${#ids[@]} != 2 || "${ids[0]:-}" == "${ids[1]:-}" ]] ||
     [[ ! "${ids[0]:-}" =~ ^([0-9]+|GPU-[A-Za-z0-9-]+)$ || ! "${ids[1]:-}" =~ ^([0-9]+|GPU-[A-Za-z0-9-]+)$ ]]; then
    echo 'Set exactly two distinct physical GPU indices or UUIDs.' >&2
    return 2
  fi
}
export PYTHONPATH="$FROZEN_CACHE_RELAX_SRC:$FROZEN_CACHE_EXAMPLE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export RELAX_SKIP_CLUSTER_CLEANUP=1 RAY_NO_WAIT=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export RELAX_CONDITION_CACHE_DIR="${RELAX_CONDITION_CACHE_DIR:-/dev/shm/relax-validation-$(id -u)-$(basename "$WORK")}"
