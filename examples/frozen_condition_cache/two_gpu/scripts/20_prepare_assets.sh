#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/../env.sh"
cd "$FROZEN_CACHE_EXAMPLE_ROOT"
python3 -m assets.prepare --output "$ASSETS" --workers "${DOWNLOAD_WORKERS:-4}"
