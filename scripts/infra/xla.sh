#!/bin/bash
# JAX/XLA reproducibility flag helpers and bash version check.
#
# Public API:
#   require_bash_5_1
#   ensure_xla_repro_flags
#   export_xla_memory_on_demand
#   export_pythonhashseed <seed>

require_bash_5_1() {
    if (( BASH_VERSINFO[0] < 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1) )); then
        echo "Error: bash >= 5.1 required (have ${BASH_VERSION})." >&2
        exit 1
    fi
}

# Idempotently ensure XLA_FLAGS contains --xla_gpu_autotune_level=0 in the
# current shell. Inherited by every child process. Required for bitwise-stable
# JAX GPU runs across processes; see tcirl.jax_session.validate_jax_reproducibility.
ensure_xla_repro_flags() {
    local existing="${XLA_FLAGS:-}"
    if [[ "$existing" =~ --xla_gpu_autotune_level=([^[:space:]]+) ]]; then
        local current="${BASH_REMATCH[1]}"
        if [[ "$current" != "0" ]]; then
            existing="${existing//--xla_gpu_autotune_level=$current/--xla_gpu_autotune_level=0}"
            export XLA_FLAGS="$existing"
        fi
    else
        export XLA_FLAGS="${existing:+$existing }--xla_gpu_autotune_level=0"
    fi
}

# XLA otherwise reserves 75% of the card at start-up, which blocks other jobs on a shared GPU for the whole run.
# A value the caller set stays.
export_xla_memory_on_demand() {
    export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
}

# Export PYTHONHASHSEED to the given seed in the current shell.
export_pythonhashseed() {
    export PYTHONHASHSEED="$1"
}
