#!/bin/bash
# TRAIN_ARGS env var parsing for forwarding extra CLI flags to training calls.
#
# Public API:
#   parse_extra_train_args <out_array_name>
#
# Reads the shell-quoted TRAIN_ARGS env var and unpacks it into the named array.

parse_extra_train_args() {
    local -n out_ref="$1"
    out_ref=()
    if [[ -n "${TRAIN_ARGS:-}" ]]; then
        eval "out_ref=( ${TRAIN_ARGS} )"
    fi
}
