#!/bin/bash
# Reproduce selected-cycle lower bounds through the shared evaluator, without training.
# Usage: scripts/traffic_signal/cadence.sh <output-directory> [cycle CLI options]
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"
OUTPUT="${1:?Usage: $0 <output-directory> [cycle CLI options]}"
shift
uv run python -m tcirl.cli.traffic_cadence --output "$OUTPUT" "$@"
