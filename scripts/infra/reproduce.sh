#!/bin/bash
# Single-call reproduce-pipeline driver.
#
# Wraps the standard sequence used by every scripts/<domain>/reproduce.sh:
#   parse_extra_train_args -> check_git_clean -> export_xla_memory_on_demand -> check_accelerator ->
#   ensure_xla_repro_flags -> export_pythonhashseed -> cd <root> ->
#   parse_visible_gpus -> resolve_pool_jobs -> pool_init -> pool_install_trap ->
#   apply LIMIT to experiments -> check_runs_absent -> prepare_results_layout ->
#   run_experiment_pool -> run_export_for_groups.
#
# Every gate runs before prepare_results_layout, because that call repoints the `latest` symlink
# and a refused pipeline must leave the last verified snapshot where `export_paper_results` reads
# it.
#
# Public API:
#   run_reproduce_experiments \
#       --root <repo_root> \
#       --family <name> \
#       --project <wandb_project> \
#       --runs <num_seeds> \
#       --experiments <bash_array_name> \
#       --exporter <console_script_name> \
#       [--hash-seed <int>] \
#       [--limit <n>] \
#       [--jobs <n>]
#
# Notes:
#   --experiments names a bash array; each entry is a whitespace-separated
#       "<group> <command> <config_path> [<extra_arg>...]" record.
#   --hash-seed defaults to 43.
#   --limit truncates the experiments array to its first N entries (0 = all).
#   --jobs overrides the auto-resolved pool size (default: visible GPU count, or 1).
#   EXPORT_SNAPSHOT, TRAIN_ARGS env vars are honored
#       by the underlying helpers.

run_reproduce_experiments() {
    local root family project runs experiments_name exporter
    local hash_seed=43
    local limit=0
    local jobs_override=""

    while (( $# > 0 )); do
        case "$1" in
            --root)        root="$2"; shift 2 ;;
            --family)      family="$2"; shift 2 ;;
            --project)     project="$2"; shift 2 ;;
            --runs)        runs="$2"; shift 2 ;;
            --experiments) experiments_name="$2"; shift 2 ;;
            --exporter)    exporter="$2"; shift 2 ;;
            --hash-seed)   hash_seed="$2"; shift 2 ;;
            --limit)       limit="$2"; shift 2 ;;
            --jobs)        jobs_override="$2"; shift 2 ;;
            *)
                echo "run_reproduce_experiments: unknown argument '$1'" >&2
                return 2
                ;;
        esac
    done

    : "${root:?run_reproduce_experiments: --root required}"
    : "${family:?run_reproduce_experiments: --family required}"
    : "${project:?run_reproduce_experiments: --project required}"
    : "${runs:?run_reproduce_experiments: --runs required}"
    : "${experiments_name:?run_reproduce_experiments: --experiments required}"
    : "${exporter:?run_reproduce_experiments: --exporter required}"

    EXTRA_TRAIN_ARGS=()
    parse_extra_train_args EXTRA_TRAIN_ARGS

    "$_INFRA_DIR/check_git_clean.sh" "$root"
    # Before the accelerator check, which starts JAX on every visible card.
    export_xla_memory_on_demand
    "$_INFRA_DIR/check_accelerator.sh"
    ensure_xla_repro_flags
    export_pythonhashseed "$hash_seed"
    cd "$root"

    local -a gpus=()
    parse_visible_gpus gpus
    local pool_jobs
    pool_jobs="$(resolve_pool_jobs "${#gpus[@]}" "$jobs_override" 1)"
    pool_init "$pool_jobs" gpus
    pool_install_trap

    local -n _exp_ref="$experiments_name"
    if (( limit > 0 && limit < ${#_exp_ref[@]} )); then
        _exp_ref=("${_exp_ref[@]:0:limit}")
    fi

    local gpu_label
    if (( ${#gpus[@]} > 0 )); then
        gpu_label="visible GPUs=${CUDA_VISIBLE_DEVICES}"
    else
        gpu_label="CPU-only"
    fi
    echo "=== Reproducing $family experiments (project=$project, runs=$runs, jobs=$pool_jobs, $gpu_label) ==="

    # One arm is one run, so a project that already holds a run of one of these groups would
    # end the pipeline at its own export gate -- after the training. Ask now instead.
    local -a preflight_groups=()
    local entry
    for entry in "${_exp_ref[@]}"; do
        # shellcheck disable=SC2206
        local fields=( $entry )
        preflight_groups+=("${fields[0]}")
    done
    uv run check_runs_absent --project "$project" --groups "${preflight_groups[@]}"

    # Only now: `prepare_results_layout` repoints the `latest` symlink at a fresh empty snapshot,
    # and a run refused by any gate above must leave the last verified snapshot reachable.
    local results_dir
    prepare_results_layout "$root" "$family" "$project" results_dir
    echo "=== $family results -> $results_dir ==="

    local -a export_groups=()
    run_experiment_pool "$experiments_name" "$project" "$runs" export_groups
    run_export_for_groups "$exporter" "$project" "$results_dir" "${export_groups[@]}"

    echo "=== All $family experiments completed ==="
}
