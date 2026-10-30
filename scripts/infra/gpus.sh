#!/bin/bash
# CUDA_VISIBLE_DEVICES parsing and parallel-job count resolution.
#
# Public API:
#   parse_visible_gpus <out_array_name>
#   resolve_pool_jobs <gpu_count> <configured_jobs> <cpu_fallback_jobs> -> stdout
#   pick_idle_gpu -> stdout (one PCI_BUS_ID index)

# The compute GPU with the lowest utilisation, then the lowest memory, in PCI_BUS_ID order; never a fixed default, because a GPU may be shared with other jobs.
# Every caller must also export CUDA_DEVICE_ORDER=PCI_BUS_ID, or this index disagrees with CUDA's own default order.
# Utilisation comes first, because a job under XLA_PYTHON_CLIENT_PREALLOCATE=false holds little memory.
# It reserves nothing, so two launches at the same moment can choose the same card; pass TCIRL_GPU explicitly when you start several at once.
pick_idle_gpu() {
    if ! command -v nvidia-smi > /dev/null 2>&1; then
        echo "Error: nvidia-smi not found, so no GPU can be chosen. Set TCIRL_GPU=<index> explicitly." >&2
        return 1
    fi
    local listing chosen
    listing="$(nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits)"
    # Emits "<util%> <used_mib> <index>" for the compute cards only, then takes the smallest pair.
    chosen="$(printf '%s\n' "$listing" | awk -F', *' '$2 !~ /Display/ { print $4, $3, $1 }' | sort -n -k1,1 -k2,2 | head -1 | awk '{print $3}')"
    if [[ -z "$chosen" ]]; then
        echo "Error: nvidia-smi listed no compute GPU. Set TCIRL_GPU=<index> explicitly." >&2
        return 1
    fi
    printf '%s\n' "$chosen"
}

parse_visible_gpus() {
    local -n out_ref="$1"
    out_ref=()
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        # An index list means nothing without an order. CUDA's default is FASTEST_FIRST, which
        # sorts a slow rendering GPU last instead of leaving it at its bus position, so the SAME
        # list selects different cards under the two orders and one of them can be a display GPU.
        if [[ -z "${CUDA_DEVICE_ORDER:-}" ]]; then
            echo "Error: CUDA_VISIBLE_DEVICES is set to '${CUDA_VISIBLE_DEVICES}' but" \
                 "CUDA_DEVICE_ORDER is not, so these indices name no definite cards." >&2
            echo "  Export CUDA_DEVICE_ORDER=PCI_BUS_ID and list the indices exactly as" \
                 "'CUDA_DEVICE_ORDER=PCI_BUS_ID nvidia-smi' prints them." >&2
            exit 1
        fi
        IFS=',' read -r -a out_ref <<< "$CUDA_VISIBLE_DEVICES"
        if (( ${#out_ref[@]} == 1 )) && [[ -z "${out_ref[0]}" ]]; then
            out_ref=()
        fi
    fi
}

resolve_pool_jobs() {
    local gpu_count="$1"
    local configured_jobs="$2"
    local cpu_fallback_jobs="$3"
    local jobs

    if [[ -z "$configured_jobs" ]]; then
        if (( gpu_count > 0 )); then
            jobs="$gpu_count"
        else
            jobs="$cpu_fallback_jobs"
        fi
    elif (( configured_jobs == 0 )); then
        if (( gpu_count > 0 )); then
            jobs="$gpu_count"
        else
            jobs="$(nproc)"
        fi
    else
        jobs="$configured_jobs"
    fi

    if (( jobs < 1 )); then
        echo "Error: parallel job count must be >= 1 (got $jobs)." >&2
        exit 1
    fi
    # More jobs than GPUs share the cards (`pool_init` wraps slots around the GPU list) and need enough spare memory.
    # It also needs XLA_PYTHON_CLIENT_PREALLOCATE=false, which the reproduce launcher exports (`export_xla_memory_on_demand`) unless the caller set another value.
    if (( gpu_count > 0 && jobs > gpu_count )); then
        if [[ -z "${XLA_PYTHON_CLIENT_PREALLOCATE:-}" || "${XLA_PYTHON_CLIENT_PREALLOCATE}" == "true" ]]; then
            echo "Error: $jobs job(s) on $gpu_count GPU(s) needs XLA_PYTHON_CLIENT_PREALLOCATE=false;" \
                 "without it each process reserves most of the card and the rest thrash in OOM backoff." >&2
            exit 1
        fi
        echo "Note: $jobs job(s) sharing $gpu_count GPU(s) (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})." >&2
    fi

    printf '%s\n' "$jobs"
}
