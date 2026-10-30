#!/bin/bash
# GPU-aware rolling-window parallel pool primitives.
#
# This file only manages the pool. XLA flag handling, results layout, env-var
# parsing, and exporting live in sibling files under scripts/infra/.
#
# Public API:
#   pool_init <jobs> <gpus_array_name>
#   pool_install_trap
#   pool_acquire_slot <out_slot_var>
#   pool_spawn <slot> <cmd> [args...]
#   pool_wait_all
#   pool_slot_label <slot> -> stdout
#
# Globals populated by pool_init:
#   POOL_PID_SET, POOL_PID_TO_SLOT (assoc arrays)
#   POOL_SLOT_ENV_VALUES, POOL_SLOT_LABELS, POOL_FREE_SLOTS (indexed arrays)
#   POOL_FAILED (int counter)
#
# Children inherit CUDA_VISIBLE_DEVICES from the pool's per-slot GPU id when
# the GPU array passed to pool_init is non-empty.

pool_init() {
    local jobs="$1"
    local -n gpu_ref="$2"
    local slot

    declare -gA POOL_PID_SET=()
    declare -gA POOL_PID_TO_SLOT=()
    declare -ga POOL_SLOT_ENV_VALUES=()
    declare -ga POOL_SLOT_LABELS=()
    declare -ga POOL_FREE_SLOTS=()
    declare -g POOL_FAILED=0

    if (( ${#gpu_ref[@]} > 0 )); then
        local gpu_count=${#gpu_ref[@]}
        local gpu_id replica
        for (( slot=0; slot<jobs; slot++ )); do
            # Wrap, so `jobs` may exceed the visible GPUs and put several runs on one card, and every
            # slot gets a CUDA_VISIBLE_DEVICES of exactly one listed GPU.
            gpu_id="${gpu_ref[$(( slot % gpu_count ))]}"
            replica=$(( slot / gpu_count ))
            POOL_SLOT_ENV_VALUES+=("$gpu_id")
            if (( jobs > gpu_count )); then
                POOL_SLOT_LABELS+=("GPU $gpu_id #$replica")
            else
                POOL_SLOT_LABELS+=("GPU $gpu_id")
            fi
            POOL_FREE_SLOTS+=("$slot")
        done
    else
        for (( slot=0; slot<jobs; slot++ )); do
            POOL_SLOT_ENV_VALUES+=("")
            POOL_SLOT_LABELS+=("worker $slot")
            POOL_FREE_SLOTS+=("$slot")
        done
    fi
}

pool_cleanup() {
    echo ""
    echo "Caught signal, killing ${#POOL_PID_SET[@]} child process(es)..."
    pool_kill_children
    exit 1
}

pool_kill_children() {
    local skip_pid="${1:-}"
    local pid

    for pid in "${!POOL_PID_SET[@]}"; do
        if [[ -n "$skip_pid" && "$pid" == "$skip_pid" ]]; then
            continue
        fi
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true

    for pid in "${!POOL_PID_SET[@]}"; do
        unset "POOL_PID_SET[$pid]"
        unset "POOL_PID_TO_SLOT[$pid]"
    done
    POOL_FREE_SLOTS=()
}

pool_install_trap() {
    trap pool_cleanup INT TERM
}

pool_wait_for_one() {
    local done_pid=""
    local slot
    local wait_status=0

    if wait -n -p done_pid "${!POOL_PID_SET[@]}"; then
        :
    else
        wait_status=$?
        POOL_FAILED=$((POOL_FAILED + 1))
        if [[ -n "${done_pid-}" ]]; then
            slot="${POOL_PID_TO_SLOT[$done_pid]}"
            unset "POOL_PID_SET[$done_pid]"
            unset "POOL_PID_TO_SLOT[$done_pid]"
            echo "Error: child process failed on $(pool_slot_label "$slot") (exit status $wait_status). Stopping remaining jobs." >&2
        else
            echo "Error: child process failed (exit status $wait_status). Stopping remaining jobs." >&2
        fi
        pool_kill_children
        return "$wait_status"
    fi

    slot="${POOL_PID_TO_SLOT[$done_pid]}"
    unset "POOL_PID_SET[$done_pid]"
    unset "POOL_PID_TO_SLOT[$done_pid]"
    POOL_FREE_SLOTS+=("$slot")
}

pool_acquire_slot() {
    local -n slot_ref="$1"
    while (( ${#POOL_FREE_SLOTS[@]} == 0 )); do
        pool_wait_for_one || return $?
    done
    slot_ref="${POOL_FREE_SLOTS[0]}"
    POOL_FREE_SLOTS=("${POOL_FREE_SLOTS[@]:1}")
}

pool_spawn() {
    local slot="$1"
    shift
    local env_value="${POOL_SLOT_ENV_VALUES[$slot]}"
    local -a device_env=()
    if [[ -n "$env_value" ]]; then
        device_env=(CUDA_VISIBLE_DEVICES="$env_value")
    fi
    env "${device_env[@]}" "$@" &
    local pid=$!
    POOL_PID_SET[$pid]=1
    POOL_PID_TO_SLOT[$pid]="$slot"
}

pool_wait_all() {
    while (( ${#POOL_PID_SET[@]} > 0 )); do
        pool_wait_for_one || return $?
    done
}

pool_slot_label() {
    local slot="$1"
    printf '%s\n' "${POOL_SLOT_LABELS[$slot]}"
}
