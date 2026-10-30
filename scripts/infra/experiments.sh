#!/bin/bash
# Experiment-pool dispatcher shared by the reproduction scripts.
#
# Each experiment entry is a whitespace-separated record:
#   "<group> <command> <config_path> [<extra_arg>...]"
# Schedules one process per experiment over the already-initialized pool,
# accumulates the group names for export, and waits for completion.
#
# Public API:
#   run_experiment_pool <experiments_array_name> <project> <runs> <out_groups_array_name>
#
# Prerequisites (caller responsibility):
#   - pool_init + pool_install_trap must have been called.
#   - EXTRA_TRAIN_ARGS array must exist (use parse_extra_train_args to populate).
#   - Each child receives
#     --config_path, --project, --num_runs, --group, --name plus the entry's
#     extra fields and any EXTRA_TRAIN_ARGS.
#   - OPTIONAL: an EXPERIMENT_NOTES associative array keyed by group. A group with an
#     entry receives `--notes "<text>"` (the W&B run note + explanations/run panel).
#     Notes may contain spaces -- that is why they cannot ride in the entry string,
#     which is whitespace-split. The shipped launchers set one per arm.
# Returns non-zero if any child failed.

run_experiment_pool() {
    local -n _exp_ref="$1"
    local project="$2"
    local runs="$3"
    local -n _out_groups_ref="$4"
    local total="${#_exp_ref[@]}"
    local idx=0
    local entry slot slot_label

    for entry in "${_exp_ref[@]}"; do
        # shellcheck disable=SC2206
        local fields=( $entry )
        local group="${fields[0]}"
        local command="${fields[1]}"
        local config_path="${fields[2]}"
        local extra_fields=( "${fields[@]:3}" )
        local note_args=()
        # The declare -p guard first: under `set -u`, subscripting an UNDECLARED array errors,
        # and subscripting an INDEXED array with a group name evaluates it arithmetically and
        # errors too -- so require a declared ASSOCIATIVE array specifically.
        if declare -p EXPERIMENT_NOTES 2>/dev/null | grep -q 'declare -A' \
            && [[ -n "${EXPERIMENT_NOTES[$group]+x}" ]]; then
            note_args=( --notes "${EXPERIMENT_NOTES[$group]}" )
        fi
        _out_groups_ref+=("$group")
        idx=$((idx + 1))

        slot=""
        pool_acquire_slot slot
        slot_label="$(pool_slot_label "$slot")"

        echo "=== [${idx}/${total}] Training group=$group ($slot_label) ==="
        pool_spawn "$slot" uv run "$command" \
            --config_path "$config_path" \
            "${extra_fields[@]}" \
            "${EXTRA_TRAIN_ARGS[@]}" \
            "${note_args[@]}" \
            --project "$project" \
            --num_runs "$runs" \
            --group "$group" \
            --name "$group"
    done

    pool_wait_all

    if (( POOL_FAILED > 0 )); then
        echo "$POOL_FAILED/${total} experiment group(s) FAILED." >&2
        return 1
    fi
}
