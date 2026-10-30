#!/bin/bash
# One-stop entry point for the reusable bash infrastructure.
#
# The launchers source this file:
#
#   source "$ROOT_DIR/scripts/infra/lib.sh"
#
# Calls require_bash_5_1 once at source time. After sourcing, the following
# helpers are available:
#
#   require_bash_5_1, ensure_xla_repro_flags, export_xla_memory_on_demand,
#   export_pythonhashseed                                            (xla.sh)
#   parse_visible_gpus, resolve_pool_jobs, pick_idle_gpu             (gpus.sh)
#   parse_extra_train_args                                           (train_args.sh)
#   prepare_results_layout, run_export_for_groups                    (results.sh)
#   pool_init, pool_install_trap, pool_acquire_slot, pool_spawn,
#   pool_wait_all, pool_slot_label,
#   plus POOL_PID_SET / POOL_FREE_SLOTS / POOL_FAILED globals        (pool.sh)
#   run_experiment_pool                                              (experiments.sh)
#   run_reproduce_experiments                                        (reproduce.sh)
#
# Use scripts/infra/check_git_clean.sh to require a clean tree whose HEAD is a commit of main.
#
# Each arm vmaps its seeds inside one process (`run.num_runs`), so one arm is one run and one W&B
# group, and `run.seed_offset` is the one seed convention that keeps contrasts across arms paired.

if [[ -z "${BASH_VERSION:-}" ]]; then
    echo "Error: infra/lib.sh requires bash." >&2
    exit 1
fi

_INFRA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=./xla.sh
source "$_INFRA_DIR/xla.sh"
# shellcheck source=./gpus.sh
source "$_INFRA_DIR/gpus.sh"
# shellcheck source=./train_args.sh
source "$_INFRA_DIR/train_args.sh"
# shellcheck source=./results.sh
source "$_INFRA_DIR/results.sh"
# shellcheck source=./pool.sh
source "$_INFRA_DIR/pool.sh"
# shellcheck source=./experiments.sh
source "$_INFRA_DIR/experiments.sh"
# shellcheck source=./reproduce.sh
source "$_INFRA_DIR/reproduce.sh"

require_bash_5_1
