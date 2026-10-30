#!/bin/bash
# Every result the paper prints, from training to its `results/` tree, in one call.
#
# Usage: scripts/reproduce_paper.sh <suffix>
#
# Trains and exports both domains into the W&B projects `tcirl-<domain>-<suffix>`, then projects
# the two snapshots onto the paper's `results/` tree under `tmp/paper_results/`.
#
# The project name carries the domain because the group names are the paper's arm names and
# are the same in both domains: one project for both would put two runs in every group, which
# every resolver refuses (`tcirl.export_runs.exactly_one_run`). `export_paper_results` therefore
# takes the domain as a `{domain}` placeholder in its `--project`.
#
# Env overrides: the TCIRL_* variables the per-domain launchers take (TCIRL_RUNS, TCIRL_JOBS,
# TCIRL_LIMIT, TCIRL_TRAIN_ARGS, TCIRL_EXPORT_SNAPSHOT), plus ALLOW_DIRTY=1, which skips the
# code-version gate for a trial run.
#
# TCIRL_PAPER_RESULTS_DIR names any directory, whose `results/` tree the projection writes.
# It defaults to empty, which keeps the output in `tmp/paper_results/results/`.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SUFFIX="${1:?Usage: $0 <suffix>}"

for domain in gene_editing traffic_signal; do
    echo "############ $domain ############"
    "$ROOT_DIR/scripts/$domain/reproduce.sh" "tcirl-${domain}-${SUFFIX}"
done

TRAFFIC_SNAPSHOT="$ROOT_DIR/tmp/results/traffic_signal/tcirl-traffic_signal-${SUFFIX}/latest"
bash "$ROOT_DIR/scripts/traffic_signal/cadence.sh" "$TRAFFIC_SNAPSHOT/cadence" --training-project "tcirl-traffic_signal-${SUFFIX}"

echo "############ the paper's results tree ############"
PROJECTION_ARGS=(--project "tcirl-{domain}-${SUFFIX}")
if [[ -n "${TCIRL_PAPER_RESULTS_DIR:-}" ]]; then
    PROJECTION_ARGS+=(--out_dir "$TCIRL_PAPER_RESULTS_DIR")
fi
uv run export_paper_results "${PROJECTION_ARGS[@]}"
