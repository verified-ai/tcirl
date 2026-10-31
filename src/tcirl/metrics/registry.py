"""tcirl's metric catalog: the shared base defaults plus the cause-phase evaluation metrics.

Every arm of both domains logs the same cause metrics at each evaluation point, because the paper's figure of
the episodes in Phase 2 and its phase claims compare exactly those numbers across arms: `eval_final/frac_in_accept` is the
share of evaluation steps spent in Phase 2, and `frac_x_a` / `frac_x_r` are the marker rates that
say whether the phase was observable at all. They are `EvalMetric`s, so they travel the one eval
channel (`logging.recorder.log_eval_metrics`) into `metric_curves/*.json` and out through the one
exporter, from which a script assembles the table.

The reducer is the mean across seeds.
"""

from __future__ import annotations

from tcirl.metrics.spec import (
    BASE_DQN_SPECS,
    BASE_TABULAR_SPECS,
    EvalMetric,
    Metric,
    MetricRegistry,
    SeedDisplay,
)

# The cause-phase and reward eval metrics every arm emits, from the greedy evaluation episodes.
_CAUSE_EVAL_METRICS: tuple[Metric, ...] = (
    # The UNDISCOUNTED per-episode return: the plain sum of an evaluation episode's rewards,
    # averaged over the seed's episodes. Named after the training metric that measures the same
    # quantity (`train/cumulative_reward`), so the two can be read against each other. Its ceiling
    # is the 100-step episode limit.
    #
    # `eval_final/return` beside it is the DISCOUNTED sum, which is what the agents maximise. A
    # 0.99 discount over a 100-step episode compresses the ceiling to 63.4, so the two numbers are
    # not comparable and each has to be read against its own ceiling.
    EvalMetric(
        key="eval_final/cumulative_reward",
        display=SeedDisplay(stats=("mean",), console="mean", fmt=".2f", short="ep_ret"),
        smoothable=True,
    ),
    # The mean reward per evaluation STEP: a rate, not a return, and no panel plots it. It divides
    # an episode's reward by that episode's length, so an agent whose episode ends early after
    # collecting almost nothing still scores well on it. Kept as a diagnostic, because a rate and a
    # return that disagree say the episode lengths moved: this number times
    # `eval_final/episode_length` is `eval_final/cumulative_reward`.
    EvalMetric(
        key="eval_final/mean_reward",
        display=SeedDisplay(stats=("mean",), console="mean", fmt=".4f", short="reward"),
        smoothable=True,
    ),
    # The share of evaluation EPISODES that reach Phase 2 at all. Acceptance is absorbing, so this
    # is equally "still in the accepting state at the episode's last step", which is what the
    # paper's figure "Episodes in Phase 2 (%)" reports.
    EvalMetric(
        key="eval_final/frac_episodes_in_accept",
        display=SeedDisplay(stats=("mean",), console="mean", fmt=".4f", short="ep_accept"),
        smoothable=False,
    ),
    # Whether this SEED ends most of its evaluation episodes in Phase 2, as a 0/1 indicator. Its
    # cross-seed mean is therefore the share of SEEDS, which the paper's section on cause activation
    # plots against training steps, and which the episode share above cannot show.
    # A pooled episode share of a fifth could come from a few seeds that fire the cause in most
    # episodes or from every seed firing it a fifth of the time.
    # Domain-neutral on purpose -- in
    # genetic therapy a seed above the line has solved the task and in traffic it has failed it,
    # and the caption says which.
    # Its interval is the Wilson interval (`tcirl.paper_results.PAPER_INTERVAL`).
    # The evaluation records it on each checkpoint's raw share, and the export recomputes it on the
    # share smoothed by the run's EMA (`tcirl.export.plotted_payloads`).
    EvalMetric(
        key="eval_final/mostly_in_accept",
        display=SeedDisplay(stats=("mean",), console="mean", fmt=".4f", short="seed_accept"),
        smoothable=False,
    ),
    # The share of evaluation STEPS spent in Phase 2, which is a different quantity and cannot be
    # substituted for the one above: an episode that fires the cause halfway through counts 1.0
    # there and 0.5 here. Its per-timestep resolution is the `trajectories/in_accept` curve.
    EvalMetric(
        key="eval_final/frac_steps_in_accept",
        display=SeedDisplay(stats=("mean",), console="mean", fmt=".4f", short="step_accept"),
        smoothable=False,
    ),
    # How many distinct cause-DFA states an evaluation episode visits: 1 means the policy never
    # advanced the cause at all, the full count means it drove it to acceptance.
    EvalMetric(
        key="eval_final/mean_distinct_cause_states",
        display=SeedDisplay(stats=("mean",), console=None, fmt=".2f", short="cause_q"),
        smoothable=False,
    ),
    # Marker rates. x_a (acceptance) can fire only in Phase 2 and x_r (rejection) only in Phase 1,
    # so the pair is the observable shadow of frac_in_accept: an arm with a high accept share and a
    # near-zero x_a rate spent Phase 2 somewhere the marker cannot fire, which the inference cannot
    # certify. TCIRL's re-inference count is only explicable next to these.
    EvalMetric(
        key="eval_final/frac_x_a",
        display=SeedDisplay(stats=("mean",), console=None, fmt=".4f", short="x_a"),
        smoothable=False,
    ),
    EvalMetric(
        key="eval_final/frac_x_r",
        display=SeedDisplay(stats=("mean",), console=None, fmt=".4f", short="x_r"),
        smoothable=False,
    ),
)

# Genetic therapy: the symptom and medicine-harm levels averaged over the evaluation episodes.
_GENE_EVAL_METRICS: tuple[Metric, ...] = (
    EvalMetric(
        key="eval_final/mean_disease",
        display=SeedDisplay(stats=("mean",), console="mean", fmt=".2f", short="disease"),
        smoothable=True,
    ),
    EvalMetric(
        key="eval_final/mean_harm",
        display=SeedDisplay(stats=("mean",), console=None, fmt=".2f", short="harm"),
        smoothable=True,
    ),
)

# Traffic signal: the queue load averaged over the evaluation episodes.
_TRAFFIC_EVAL_METRICS: tuple[Metric, ...] = (
    EvalMetric(
        key="eval_final/mean_total_queue",
        display=SeedDisplay(stats=("mean",), console="mean", fmt=".2f", short="queue"),
        smoothable=True,
    ),
)


def gene_editing_specs(*, deep: bool = False) -> MetricRegistry:
    """The genetic-therapy registry: base defaults + cause-phase and symptom eval metrics."""
    base = BASE_DQN_SPECS if deep else BASE_TABULAR_SPECS
    return base.extended((*_CAUSE_EVAL_METRICS, *_GENE_EVAL_METRICS))


def traffic_signal_specs(*, deep: bool = False) -> MetricRegistry:
    """The traffic-signal registry: base defaults + cause-phase and queue eval metrics."""
    base = BASE_DQN_SPECS if deep else BASE_TABULAR_SPECS
    return base.extended((*_CAUSE_EVAL_METRICS, *_TRAFFIC_EVAL_METRICS))
