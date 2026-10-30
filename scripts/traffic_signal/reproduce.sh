#!/bin/bash
# Traffic-signal case study: every arm the paper prints, trained and exported in one call.
#
# Nine arms in one W&B project (TCIRL, Q-learning with and without the cause, the DQN at a
# 4-frame and a 1-frame window, and the four TCIRL ablations of the appendix), 50 seeds each, vmapped inside one process per arm. The export
# then writes every per-seed curve of every arm -- the training curves, the inference curves and
# the evaluation trajectories -- into a timestamped snapshot under
# `tmp/results/traffic_signal/<project>/`, with a `latest` symlink beside it.
#
# `export_paper_results` turns that snapshot into the paper's `results/traffic_signal/` tree.
#
# Usage: scripts/traffic_signal/reproduce.sh <project>
# Env overrides: TCIRL_RUNS (default 50), TCIRL_JOBS, TCIRL_LIMIT, TCIRL_TRAIN_ARGS,
# TCIRL_EXPORT_SNAPSHOT.
set -euo pipefail

# shellcheck source=../lib.sh
source "$(cd "$(dirname "$0")/.." && pwd)/lib.sh"
tcirl_require_project "$@"

EXPERIMENTS=()
tcirl_paper_experiments traffic_signal EXPERIMENTS
declare -A EXPERIMENT_NOTES=()
tcirl_paper_notes EXPERIMENT_NOTES

run_reproduce_experiments \
    --root "$ROOT_DIR" \
    --family traffic_signal \
    --project "$PROJECT" \
    --runs "${TCIRL_RUNS:-50}" \
    --jobs "${TCIRL_JOBS:-}" \
    --limit "${TCIRL_LIMIT:-0}" \
    --experiments EXPERIMENTS \
    --exporter export_group_metrics
