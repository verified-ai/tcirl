#!/bin/bash
# The paper's cycle benchmarks: the exact returns of the best tested fixed action cycles and of the
# optimal policy, through the shared evaluator and without training.
# Usage: scripts/traffic_signal/cycles.sh <output-directory> [cycle CLI options]
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"
OUTPUT="${1:?Usage: $0 <output-directory> [cycle CLI options]}"
shift
uv run python -m tcirl.cli.traffic_cycles --output "$OUTPUT" "$@"
