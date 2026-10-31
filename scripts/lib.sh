#!/bin/bash
# Shared preamble for the tcirl launcher scripts (scripts/<domain>/*.sh).
#
# Sourcing this file resolves ROOT_DIR (the tcirl repo root), maps the TCIRL_*
# env overrides onto the generic infra names (TRAIN_ARGS, EXPORT_SNAPSHOT), and
# sources scripts/infra/lib.sh. Call `tcirl_require_project "$@"` afterwards to
# parse the single <project> argument into PROJECT.

TCIRL_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$TCIRL_LIB_DIR/.." && pwd)"

: "${TRAIN_ARGS:=${TCIRL_TRAIN_ARGS:-}}"
: "${EXPORT_SNAPSHOT:=${TCIRL_EXPORT_SNAPSHOT:-}}"
export TRAIN_ARGS EXPORT_SNAPSHOT

# shellcheck source=./infra/lib.sh
source "$ROOT_DIR/scripts/infra/lib.sh"

tcirl_require_project() {
    if (( $# != 1 )); then
        echo "Usage: $0 <project>" >&2
        exit 1
    fi
    PROJECT="$1"
}

# The five arms the paper prints, then the four arms of its ablations of value transfer and
# counterfactual updates. Each entry is
# "<group> <command> <config_path> [<extra_arg>...]"; the group name is also the W&B run name and
# the directory `results/<domain>/<group>/` that the export writes and the paper's data use.
tcirl_paper_experiments() {
    local domain="$1"
    local -n _out="$2"
    local ql tc dq
    case "$domain" in
        gene_editing)  ql=train_gene_editing_q_learning; tc=train_gene_editing_tcirl; dq=train_gene_editing_dqn ;;
        traffic_signal) ql=train_traffic_q_learning; tc=train_traffic_tcirl; dq=train_traffic_dqn ;;
        *) echo "tcirl_paper_experiments: unknown domain '$domain'" >&2; return 2 ;;
    esac
    _out=(
        "tcirl-sat $tc config/${domain}_tcirl.yaml"
        "q_learning-cause_known $ql config/${domain}_q_learning.yaml --cause_known"
        "q_learning-cause_unknown $ql config/${domain}_q_learning.yaml --nocause_known"
        "dqn-cause_unknown_4 $dq config/${domain}_dqn.yaml --nocause_known --history_len 4"
        "dqn-cause_unknown_1 $dq config/${domain}_dqn.yaml --nocause_known --history_len 1"
        "tcirl-sat-no_transfer $tc config/${domain}_tcirl.yaml --value_transfer none"
        "tcirl-sat-exact_transfer $tc config/${domain}_tcirl.yaml --value_transfer exact"
        "tcirl-sat-counterfactual $tc config/${domain}_tcirl.yaml --counterfactual"
        "q_learning-cause_known-counterfactual $ql config/${domain}_q_learning.yaml --cause_known --counterfactual"
    )
}

# Per-arm W&B notes (--notes; also the run's explanations/run panel). Keyed by group.
tcirl_paper_notes() {
    local -n _notes="$1"
    _notes=(
        [tcirl-sat]="TCIRL: infers the hidden cause DFA from counterexamples with the rejection marker x_r and the acceptance marker x_a, SAT synthesis. The paper's method. Compare against q_learning-cause_known (the full-information reference) and the three cause-blind arms."
        [q_learning-cause_known]="Q-learning on the true cause DFA, whose state it tracks from the labels: the full-information reference for a tabular agent."
        [q_learning-cause_unknown]="Q-learning on the one-state DFA, with no cause information at all: the lower reference. It cannot represent the phase, so it learns one policy for both phases."
        [dqn-cause_unknown_4]="DQN over a window of the last 4 observation frames, where one frame is the observable state plus the label of the transition into it."
        [dqn-cause_unknown_1]="The same DQN with a 1-frame window, which is the MEMORYLESS agent: it sees only the current frame, so no bounded memory stands in for the cause state. Compare against dqn-cause_unknown_4."
        [tcirl-sat-no_transfer]="Ablation: TCIRL that restarts its whole Q-table from q_init and its exploration schedule at every re-inference instead of transferring values. Compare against tcirl-sat."
        [tcirl-sat-exact_transfer]="Ablation: TCIRL whose every new hypothesis state takes the values of an old state that accepts the same label sequences, and otherwise of the old state that its shortest label sequence reaches, instead of the old state that shares the most prefixes of the causal sample with it. Compare against tcirl-sat."
        [tcirl-sat-counterfactual]="Ablation: TCIRL whose every transition also updates the entry of its action at every hypothesis state with the phase of the visited state. Compare against tcirl-sat, and against q_learning-cause_known-counterfactual."
        [q_learning-cause_known-counterfactual]="Ablation reference: Q-learning with the cause state and the same counterfactual updates over the states of the true cause DFA. Compare against q_learning-cause_known, which runs without them."
    )
}
