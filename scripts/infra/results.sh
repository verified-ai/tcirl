#!/bin/bash
# Results directory layout and W&B group export.
#
# Public API:
#   prepare_results_layout <repo_root> <family> <project> <dir_var>
#   run_export_for_groups <export_cmd> <project> <results_dir> <group> [<group>...]
#
# Layout convention:
#   <repo_root>/tmp/results/<family>/<project>/<snapshot>/
# Plus a sibling `latest` symlink that points to the newest snapshot.
#
# Env vars honored:
#   EXPORT_SNAPSHOT  Snapshot directory name. Defaults to a UTC timestamp.
#   EXPORT_SOURCE    Optional --source value (auto|local|wandb) forwarded to the export
#                    command. Unset means the command's own default, which is local-first.

prepare_results_layout() {
    local root_dir="$1"
    local family="$2"
    local project="$3"
    local -n dir_ref="$4"

    local snapshot="${EXPORT_SNAPSHOT:-$(date -u +%Y%m%dT%H%M%SZ)}"
    local results_root="$root_dir/tmp/results/$family/$project"
    dir_ref="$results_root/$snapshot"
    mkdir -p "$dir_ref"
    ln -sfn "$dir_ref" "$results_root/latest"
}

# Invoke `uv run <export_cmd>` for each group with the standard --project /
# --group / --output_dir flags. EXPORT_SOURCE, if set, is forwarded as --source.
#
# The export commands read runs from the local `wandb/` directory by default and
# only fall back to the W&B API, so this works for offline (`--upload False`)
# experiments with no network. The fallback also triggers for a group whose local
# runs have not finished (still training, killed, crashed) -- and the API copy is
# then checked for completeness too, so a partial group fails the export instead
# of being written out as final, whatever EXPORT_SOURCE says. Set
# EXPORT_SOURCE=local to skip the network attempt and fail on the disk copy alone.
#
# The caller passes the export console script as the first argument
# (`export_group_metrics`).
run_export_for_groups() {
    local export_cmd="$1"
    local project="$2"
    local results_dir="$3"
    shift 3
    local extra_args=()
    if [[ -n "${EXPORT_SOURCE:-}" ]]; then
        extra_args+=(--source "$EXPORT_SOURCE")
    fi
    local group
    for group in "$@"; do
        echo "=== Exporting group=$group ==="
        uv run "$export_cmd" \
            --project "$project" \
            --group "$group" \
            --output_dir "$results_dir" \
            "${extra_args[@]}"
    done
}
