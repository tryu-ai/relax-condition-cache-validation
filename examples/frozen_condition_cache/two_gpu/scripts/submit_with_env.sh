#!/bin/bash
# Preserve the kit's cache settings after the official environment setup.
set -eo pipefail
export RELAX_SKIP_CLUSTER_CLEANUP=1
export RAY_NO_WAIT=1
export TEST_REQUESTED_RUNTIME_ENV_JSON="${RUNTIME_ENV_JSON:-\{\}}"
unset _RAY_JOB_RUN_SCRIPT
unset RELAX_ENTRYPOINT_MODE
source scripts/entrypoint/ray-job.sh
RUNTIME_ENV_JSON=$(python3 - <<'PY'
import json, os
base = json.loads(os.environ['RUNTIME_ENV_JSON'])
requested = json.loads(os.environ['TEST_REQUESTED_RUNTIME_ENV_JSON'])
for key, value in requested.items():
    if key == 'env_vars':
        base.setdefault(key, {}).update(value)
    else:
        base[key] = value
if os.environ.get('TEST_RUNTIME_ENV_RECEIPT'):
    with open(os.environ['TEST_RUNTIME_ENV_RECEIPT'], 'w') as output:
        json.dump(base, output, indent=2)
print(json.dumps(base))
PY
)
export RUNTIME_ENV_JSON
exec bash scripts/training/diffusion/run-sensenova-i2i-sft-8xgpu.sh "$@"
