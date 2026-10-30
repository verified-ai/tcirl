"""Analysis of the greedy evaluation episodes: the one path for both domains.

An arm's evaluation produces one tensor of tracked state indices per (seed, episode, timestep)
plus the per-step rewards and an active mask. This module turns that into two things and nothing
else:

- the per-seed `eval_final/*` values, handed to the one eval channel
  (`tcirl.logging.recorder.log_eval_metrics`), which validates them against the run's registry and
  persists them per seed. The paper's figure of evaluation trajectories that reach Phase 2 is
  `eval_final/frac_episodes_in_accept` times 100 per arm, and `tcirl.paper_results` assembles it
  from those persisted values.
- the per-seed per-timestep `trajectories/*` curves, handed to a recorder whose x axis is the
  timestep inside one episode. They travel the same JSON, the same reducer and the same exporter
  as the training curves; nothing here reduces across seeds for the export.

`ScheduledEvaluation` owns both, and a training loop hands it one rollout per point of the run's
`~tcirl.rl.common.eval_schedule.EvalSchedule`. The first bullet is produced at EVERY point, so the
`eval_final/*` metrics are a curve whenever the grid holds more than one step; the second is
produced at the LAST point only, because those curves describe the policy the run finishes with.
The shipped `num_eval_points: 501` therefore gives an `eval_final/*` curve at the training
curve's own resolution and one set of `trajectories/*` profiles; at 1 the grid holds only that last
step, the classic train-then-evaluate split.

The W&B trajectory images are previews drawn from the same per-seed arrays. The exported CSV is
the artifact.

Marker presence (x_r, x_a) is set membership on the label the rollout recorded for each
transition, so a marker sits on the same row as the reward and the phase of the state that
transition left (`~tcirl.rl.common.greedy_eval` fixes that frame).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from tcirl.logging.recorder import PerSeedMetricRecorder, log_eval_metrics
from tcirl.logging.wandb import WandbLogger
from tcirl.metrics.smoothing import Smoother, ema_alpha
from tcirl.metrics.stats import aggregate_stats, compute_mean
from tcirl.rl.common.eval_schedule import EvalSchedule
from tcirl.rl.common.greedy_eval import EvalRollout

# Gene-editing trajectory columns, in the order the CLI passes `trajectory_indices`.
_DISEASE_COL = 0
_HARM_COL = 1
_GE_CAUSE_COL = 2

# Traffic-signal trajectory columns.
_Q0_COL = 0
_Q1_COL = 1
_Q2_COL = 2
_TRAFFIC_CAUSE_COL = 3


@dataclass(frozen=True, slots=True, kw_only=True)
class TrajectorySeries:
    """One per-timestep evaluation curve: its registry-style key, its values, and its plot frame."""

    key: str
    title: str
    ylabel: str
    ylim: tuple[float, float]
    values: jax.Array  # (num_runs, eval_episodes, max_steps) float32


# ---------------------------------------------------------------------------
# Per-episode derived quantities
# ---------------------------------------------------------------------------


def _marker_arrays(
    obs_label: np.ndarray[tuple[int, int, int], np.dtype[np.int32]],
    active: np.ndarray[tuple[int, int, int], np.dtype[np.bool_]],
    x_r_obs_ids: frozenset[int],
    x_a_obs_ids: frozenset[int],
) -> tuple[
    np.ndarray[tuple[int, int, int], np.dtype[np.float32]],
    np.ndarray[tuple[int, int, int], np.dtype[np.float32]],
]:
    """Per-timestep x_a and x_r presence, as 0.0/1.0 arrays of the label tensor's shape.

    Vectorized over the whole tensor: membership becomes a lookup in a boolean table indexed by
    the label, so the two marker sets are read once instead of once per (seed, episode, step).
    Inactive steps are zeroed. The rollout freezes the env state at the episode's end and keeps
    stepping it to keep the key stream regular, so every trailing row carries a label that no real
    transition of that episode produced; the LAST real transition's label sits on the final ACTIVE
    row, so nothing is lost by the mask.
    """
    num_symbols = int(obs_label.max(initial=0)) + 1
    is_x_r = np.zeros(num_symbols, dtype=np.bool_)
    is_x_a = np.zeros(num_symbols, dtype=np.bool_)
    for label_id in x_r_obs_ids:
        if label_id < num_symbols:
            is_x_r[label_id] = True
    for label_id in x_a_obs_ids:
        if label_id < num_symbols:
            is_x_a[label_id] = True
    x_r = (is_x_r[obs_label] & active).astype(np.float32)
    x_a = (is_x_a[obs_label] & active).astype(np.float32)
    return x_a, x_r


def _distinct_cause_states(
    cause: np.ndarray[tuple[int, int, int], np.dtype[np.int32]],
    active: np.ndarray[tuple[int, int, int], np.dtype[np.bool_]],
) -> np.ndarray[tuple[int, int, int], np.dtype[np.float32]]:
    """The running count of distinct cause-DFA states each episode has visited up to each timestep.

    Vectorized: a state is counted at the first timestep it appears, so the running count is the
    cumulative sum of the first-occurrence indicator. Inactive steps hold 0, matching the mask the
    per-seed means weight by.
    """
    num_states = int(cause.max(initial=0)) + 1
    # (runs, eps, steps, states): the one-hot of the visited state at each active step.
    onehot = (cause[..., None] == np.arange(num_states)) & active[..., None]
    seen_before = np.cumsum(onehot, axis=2) - onehot
    first_time = onehot & ~seen_before.astype(np.bool_)
    return np.cumsum(first_time.sum(axis=3), axis=2).astype(np.float32) * active


def _cause_series(
    trajectories: jax.Array,
    active_mask: jax.Array,
    labels: jax.Array,
    *,
    cause_col: int,
    cause_accepting_state: int,
    x_r_obs_ids: frozenset[int],
    x_a_obs_ids: frozenset[int],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """`(distinct_cause, in_accept, x_a_present, x_r_present)`, each (runs, episodes, steps)."""
    cause = np.asarray(trajectories[:, :, :, cause_col], dtype=np.int32)
    active = np.asarray(active_mask, dtype=np.bool_)
    obs_label = np.asarray(labels, dtype=np.int32)
    x_a, x_r = _marker_arrays(obs_label, active, x_r_obs_ids, x_a_obs_ids)
    in_accept = ((cause == cause_accepting_state) & active).astype(np.float32)
    return (
        jnp.asarray(_distinct_cause_states(cause, active)),
        jnp.asarray(in_accept),
        jnp.asarray(x_a),
        jnp.asarray(x_r),
    )


def _per_seed_mean(vals: jax.Array, active_f: jax.Array, active_sum: jax.Array) -> jax.Array:
    """One value per seed: the mean of `vals` over that seed's active episode steps."""
    return jnp.sum(vals * active_f, axis=(1, 2)) / jnp.maximum(active_sum, 1.0)


def _per_seed_episode_fraction(vals: jax.Array, active_mask: jax.Array) -> jax.Array:
    """One value per seed: the share of its episodes in which `vals` is ever nonzero.

    For `in_accept` this is the paper's figure of "evaluation trajectories that reach Phase 2" --
    and it is NOT the share of STEPS in Phase 2, which an episode that fires the cause halfway
    through would report as 0.5. Acceptance is absorbing, so "ever" and "at the last step" agree;
    "ever" is computed here because it does not depend on finding the last active step.
    """
    ever = jnp.any((vals > 0.0) & active_mask, axis=2)  # (num_runs, eval_episodes)
    return jnp.mean(ever.astype(jnp.float32), axis=1)


def _per_seed_per_timestep(vals: jax.Array, active_f: jax.Array) -> jax.Array:
    """`(num_runs, max_steps)`: each seed's mean of `vals` over the episodes still active at t."""
    active_at_t = jnp.sum(active_f, axis=1)
    return jnp.sum(vals * active_f, axis=1) / jnp.maximum(active_at_t, 1.0)


# ---------------------------------------------------------------------------
# The one emission path
# ---------------------------------------------------------------------------


def _record_trajectories(
    logger: WandbLogger,
    traj_recorder: PerSeedMetricRecorder,
    series: Sequence[TrajectorySeries],
    active_f: jax.Array,
    step: int,
) -> dict[str, tuple[list[int], list[float], list[float], list[float]]]:
    """Persist each series per seed per timestep and draw its W&B preview image.

    A timestep at which no episode of any seed is still running ends the curve: past that point
    every value would be the frozen final state repeated, which is not data.
    """
    active_at_t = jnp.sum(active_f, axis=1)  # (num_runs, max_steps)
    seed_has_data = active_at_t > 0
    max_steps = active_f.shape[2]
    per_seed_ts = {s.key: _per_seed_per_timestep(s.values, active_f) for s in series}

    timesteps: list[int] = []
    preview: dict[str, tuple[list[float], list[float], list[float], list[float]]] = {
        s.key: ([], [], [], []) for s in series
    }
    bootstrap_key = jax.random.key(789)
    for t in range(max_steps):
        seed_mask_t = seed_has_data[:, t]
        if not bool(jnp.any(seed_mask_t)):
            break
        timesteps.append(t)
        row = {s.key: per_seed_ts[s.key][:, t] for s in series}
        traj_recorder.record(t, {k: jnp.where(seed_mask_t, v, jnp.nan) for k, v in row.items()})
        keys = jax.random.split(bootstrap_key, len(series) + 1)
        bootstrap_key = keys[len(series)]
        for i, s in enumerate(series):
            cs = aggregate_stats(row[s.key], mask=seed_mask_t, key=keys[i])
            mean_l, ci_lo_l, ci_hi_l, median_l = preview[s.key]
            mean_l.append(cs.mean)
            ci_lo_l.append(cs.mean_ci_lo)
            ci_hi_l.append(cs.mean_ci_hi)
            median_l.append(cs.median)

    result: dict[str, tuple[list[int], list[float], list[float], list[float]]] = {}
    for s in series:
        mean_l, ci_lo_l, ci_hi_l, median_l = preview[s.key]
        logger.log_trajectory_ci(
            s.key,
            timesteps,
            mean_l,
            ci_lo_l,
            ci_hi_l,
            median=median_l,
            title=s.title,
            ylabel=s.ylabel,
            ylim=s.ylim,
            step=step,
        )
        result[s.key] = (timesteps, mean_l, ci_lo_l, ci_hi_l)
    return result


def _log_gap_plot(
    logger: WandbLogger,
    preview: Mapping[str, tuple[list[int], list[float], list[float], list[float]]],
    step: int,
) -> None:
    """The x_a / in_accept / x_r image: how much of the phase the markers actually certify."""
    ts, xa_mean, xa_lo, xa_hi = preview["trajectories/x_a"]
    _, a_mean, a_lo, a_hi = preview["trajectories/in_accept"]
    _, xr_mean, xr_lo, xr_hi = preview["trajectories/x_r"]
    logger.log_trajectory_gap(
        "trajectories/gap",
        ts,
        [
            ("x_a", xa_mean, xa_lo, xa_hi, "tab:green"),
            ("in_accept", a_mean, a_lo, a_hi, "tab:blue"),
            ("x_r", xr_mean, xr_lo, xr_hi, "tab:orange"),
        ],
        title="x_a / Accept / x_r Gap",
        ylabel="Fraction",
        ylim=(0.0, 1.0),
        step=step,
    )


# ---------------------------------------------------------------------------
# One evaluation point, and the grid of them
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class EvalDomain:
    """What one domain contributes to an evaluation point.

    The cause, the marker and the reward quantities are computed once for both domains, because
    both make the same claim about the phase. A domain supplies only what is its own: which
    trajectory columns the rollout has to track, where the cause state sits among them, and how the
    domain's own state variables (the symptom level, the queue load) are read out of them.
    """

    trajectory_indices: tuple[int, ...]
    cause_col: int
    num_cause_states: int
    cause_accepting_state: int
    x_r_obs_ids: frozenset[int]
    x_a_obs_ids: frozenset[int]
    console_keys: tuple[str, ...]
    state_metrics: Callable[[jax.Array], dict[str, jax.Array]]
    state_series: Callable[[jax.Array], list[TrajectorySeries]]

    def __post_init__(self) -> None:
        # `cause_col` indexes the tracked columns the rollout collects, so a domain that asks for
        # fewer columns than its own cause column would read a neighbouring state component as the
        # cause DFA state and report a phase share computed from the wrong number. JAX clamps an
        # out-of-range index silently, so this has to be checked here.
        if not 0 <= self.cause_col < len(self.trajectory_indices):
            raise ValueError(
                f"cause_col {self.cause_col} is not a column of the {len(self.trajectory_indices)} "
                f"tracked indices {self.trajectory_indices}"
            )


class ScheduledEvaluation:
    """The one sink for every greedy evaluation a run performs.

    A run evaluates at each step of its `~tcirl.rl.common.eval_schedule.EvalSchedule`, and each
    point produces the same per-seed `eval_final/*` values through the same channel. At one point
    -- the grid's last, which is always `total_steps` -- the point ALSO produces the per-timestep
    `trajectories/*` curves and the W&B summary, because those describe the policy the run
    finishes with and the paper reads them as such.

    At `num_eval_points: 1` this is an end-of-run evaluation, and it reports the numbers that
    evaluating the pair `train()` returns would report -- with ONE exception, which is the TCIRL
    arm and only when a seed re-infers during the final batch. `train()` returns the pair the
    re-inference left; this evaluates the pair the batch trained (`agents.tcirl.train` says why).

    The final values do not depend on how many points the grid holds.
    """

    def __init__(
        self,
        *,
        logger: WandbLogger,
        recorder: PerSeedMetricRecorder,
        traj_recorder: PerSeedMetricRecorder,
        schedule: EvalSchedule,
        smoothing_steps: int | None,
        domain: EvalDomain,
        num_runs: int,
        eval_episodes: int,
    ) -> None:
        self._logger = logger
        self._recorder = recorder
        self._traj_recorder = traj_recorder
        self._domain = domain
        self._num_runs = num_runs
        self._eval_episodes = eval_episodes
        self.schedule = schedule
        self._recorded: list[int] = []
        # The live W&B eval curve carries the same EMA the exported one does, which is the invariant
        # `tcirl.metrics.smoothing` states: one smoothing setting, one result, live and offline.
        # The WEIGHT is not the training curve's, because the cadence is not: `smoothing_steps` is a
        # horizon in ENV STEPS, so a grid whose points sit `spacing` apart needs its own alpha to
        # smooth over the same 10000 steps the training panel does. A one-point grid smooths
        # nothing, there being one point to smooth.
        alpha = ema_alpha(smoothing_steps, schedule.spacing)
        smoothable = logger.specs.smoothable_keys
        self._smoother = (
            None if alpha is None else Smoother(alpha=alpha, smoothable_keys=smoothable)
        )

    @property
    def eval_episodes(self) -> int:
        """How many greedy episodes each seed plays at each point."""
        return self._eval_episodes

    @property
    def trajectory_indices(self) -> tuple[int, ...]:
        """The state components the rollout has to track, which the domain fixes."""
        return self._domain.trajectory_indices

    def record(self, step: int, rollout: EvalRollout) -> None:
        """Record one evaluation point: the per-seed values, and the curves if it is the last."""
        if step not in self.schedule.steps:
            raise ValueError(
                f"step {step} is not a point of the evaluation grid {self.schedule.steps}"
            )
        if step in self._recorded:
            raise ValueError(f"the evaluation point at step {step} was already recorded")
        self._recorded.append(step)
        domain = self._domain
        active_mask = rollout.active_mask
        active_f = active_mask.astype(jnp.float32)
        active_sum = jnp.sum(active_f, axis=(1, 2))
        distinct_cause, in_accept, x_a, x_r = _cause_series(
            rollout.trajectories,
            active_mask,
            rollout.labels,
            cause_col=domain.cause_col,
            cause_accepting_state=domain.cause_accepting_state,
            x_r_obs_ids=domain.x_r_obs_ids,
            x_a_obs_ids=domain.x_a_obs_ids,
        )
        # The reward the eval loop actually paid, zeroed where the episode is over. The env pays 0
        # on a terminal step, so the running total over an episode is that episode's return.
        reward_vals = jnp.where(active_mask, rollout.rewards, 0.0).astype(jnp.float32)

        per_seed: dict[str, jax.Array] = {
            key: _per_seed_mean(vals, active_f, active_sum)
            for key, vals in domain.state_metrics(rollout.trajectories).items()
        }
        # The three per-episode scores, each a mean over this seed's evaluation episodes.
        # `return` is DISCOUNTED (the quantity the agents maximise) and `cumulative_reward` is the
        # same sum undiscounted, named after the training metric that measures the same thing.
        per_seed["eval_final/return"] = jnp.mean(rollout.returns, axis=1)
        per_seed["eval_final/cumulative_reward"] = jnp.mean(rollout.total_returns, axis=1)
        per_seed["eval_final/episode_length"] = jnp.mean(
            rollout.lengths.astype(jnp.float32), axis=1
        )
        per_seed["eval_final/mean_reward"] = _per_seed_mean(reward_vals, active_f, active_sum)
        per_seed["eval_final/mean_distinct_cause_states"] = _per_seed_mean(
            distinct_cause, active_f, active_sum
        )
        frac_in_accept = _per_seed_episode_fraction(in_accept, active_mask)
        per_seed["eval_final/frac_episodes_in_accept"] = frac_in_accept
        # A per-seed INDICATOR, so its cross-seed mean is the share of SEEDS rather than of
        # episodes. The two answer different questions and one cannot be recovered from the other:
        # a 24% episode share is 50 seeds firing the cause not quite a quarter of the time, or 12
        # seeds firing it always and 38 never, and only this metric tells those apart. The threshold
        # is one half.
        per_seed["eval_final/mostly_in_accept"] = (frac_in_accept > 0.5).astype(jnp.float32)
        per_seed["eval_final/frac_steps_in_accept"] = _per_seed_mean(
            in_accept, active_f, active_sum
        )
        per_seed["eval_final/frac_x_a"] = _per_seed_mean(x_a, active_f, active_sum)
        per_seed["eval_final/frac_x_r"] = _per_seed_mean(x_r, active_f, active_sum)

        smoothed = None if self._smoother is None else self._smoother.update(per_seed)
        metrics = log_eval_metrics(
            self._logger, self._recorder, per_seed, step=step, per_seed_smoothed=smoothed
        )
        self._logger.flush_step(step)

        # The console line reports the mean across seeds, matching the registry.
        parts = [
            f"{key.rsplit('/', 1)[-1]}={compute_mean(per_seed[key]):.4f}"
            for key in domain.console_keys
        ]
        print(
            f"  Eval @ {step} ({self._num_runs} seeds x {self._eval_episodes} episodes): "
            + ", ".join(parts)
        )

        if self.schedule.is_last(step):
            for key, value in metrics.items():
                self._logger.log_summary(key, value)
            self._record_final_curves(
                step,
                rollout,
                reward_vals=reward_vals,
                distinct_cause=distinct_cause,
                in_accept=in_accept,
                x_a=x_a,
                x_r=x_r,
                active_f=active_f,
            )

    def _record_final_curves(
        self,
        step: int,
        rollout: EvalRollout,
        *,
        reward_vals: jax.Array,
        distinct_cause: jax.Array,
        in_accept: jax.Array,
        x_a: jax.Array,
        x_r: jax.Array,
        active_f: jax.Array,
    ) -> None:
        """The per-timestep `trajectories/*` curves of the policy the run finishes with."""
        domain = self._domain
        cumulative_reward = jnp.cumsum(reward_vals, axis=2)
        series: list[TrajectorySeries] = [
            *domain.state_series(rollout.trajectories),
            TrajectorySeries(
                key="trajectories/reward",
                title="Reward per Step",
                ylabel="Reward",
                ylim=(0.0, 1.0),
                values=reward_vals,
            ),
            TrajectorySeries(
                key="trajectories/cumulative_reward",
                title="Cumulative Reward",
                ylabel="Reward",
                ylim=(0.0, float(active_f.shape[2])),
                values=cumulative_reward,
            ),
            TrajectorySeries(
                key="trajectories/distinct_cause_states",
                title="Distinct Cause DFA States",
                ylabel="Count",
                ylim=(0.0, float(domain.num_cause_states)),
                values=distinct_cause,
            ),
            TrajectorySeries(
                key="trajectories/in_accept",
                title="Fraction in Accept",
                ylabel="Fraction",
                ylim=(0.0, 1.0),
                values=in_accept,
            ),
            TrajectorySeries(
                key="trajectories/x_a",
                title="x_a Marker",
                ylabel="Fraction",
                ylim=(0.0, 1.0),
                values=x_a,
            ),
            TrajectorySeries(
                key="trajectories/x_r",
                title="x_r Marker",
                ylabel="Fraction",
                ylim=(0.0, 1.0),
                values=x_r,
            ),
        ]
        preview = _record_trajectories(self._logger, self._traj_recorder, series, active_f, step)
        _log_gap_plot(self._logger, preview, step)
        self._traj_recorder.save()

    def finish(self) -> None:
        """Refuse a run that skipped a point of its own grid.

        The grid is the run's claim about which steps it measured, and a missing point turns a
        curve into a curve with a hole nothing downstream can see. This raises at the end of
        training, where the traceback still names the run.
        """
        missing = [step for step in self.schedule.steps if step not in self._recorded]
        if missing:
            raise ValueError(
                f"the evaluation grid {self.schedule.steps} has no recorded point at {missing}; "
                f"the training loop did not call `record` there"
            )


# ---------------------------------------------------------------------------
# Domain adapters: which column is what, and which curves the domain owns
# ---------------------------------------------------------------------------


def gene_editing_domain(
    *,
    disease_index: int,
    harm_index: int,
    disease_max: int,
    harm_max: int,
    cause_index: int,
    num_cause_states: int,
    cause_accepting_state: int,
    x_r_obs_ids: frozenset[int],
    x_a_obs_ids: frozenset[int],
) -> EvalDomain:
    """Gene editing: the symptom and medicine-harm levels beside the shared cause metrics."""

    def state_metrics(trajectories: jax.Array) -> dict[str, jax.Array]:
        return {
            "eval_final/mean_disease": trajectories[:, :, :, _DISEASE_COL].astype(jnp.float32),
            "eval_final/mean_harm": trajectories[:, :, :, _HARM_COL].astype(jnp.float32),
        }

    def state_series(trajectories: jax.Array) -> list[TrajectorySeries]:
        return [
            TrajectorySeries(
                key="trajectories/disease",
                title="Disease Level",
                ylabel="Disease",
                ylim=(0.0, float(disease_max)),
                values=trajectories[:, :, :, _DISEASE_COL].astype(jnp.float32),
            ),
            TrajectorySeries(
                key="trajectories/harm",
                title="Harm Level",
                ylabel="Harm",
                ylim=(0.0, float(harm_max)),
                values=trajectories[:, :, :, _HARM_COL].astype(jnp.float32),
            ),
        ]

    return EvalDomain(
        trajectory_indices=(disease_index, harm_index, cause_index),
        cause_col=_GE_CAUSE_COL,
        num_cause_states=num_cause_states,
        cause_accepting_state=cause_accepting_state,
        x_r_obs_ids=x_r_obs_ids,
        x_a_obs_ids=x_a_obs_ids,
        console_keys=(
            "eval_final/return",
            "eval_final/cumulative_reward",
            "eval_final/episode_length",
            "eval_final/mean_reward",
            "eval_final/mean_disease",
            "eval_final/mean_harm",
            "eval_final/mean_distinct_cause_states",
            "eval_final/frac_episodes_in_accept",
            "eval_final/mostly_in_accept",
            "eval_final/frac_steps_in_accept",
            "eval_final/frac_x_a",
            "eval_final/frac_x_r",
        ),
        state_metrics=state_metrics,
        state_series=state_series,
    )


def traffic_signal_domain(
    *,
    queue_indices: tuple[int, int, int],
    cause_index: int,
    q_max: int,
    num_cause_states: int,
    cause_accepting_state: int,
    x_r_obs_ids: frozenset[int],
    x_a_obs_ids: frozenset[int],
) -> EvalDomain:
    """Traffic signal: the three queues and their total beside the shared cause metrics."""
    queue_frame = (0.0, float(q_max))

    def _queues(trajectories: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        return (
            trajectories[:, :, :, _Q0_COL].astype(jnp.float32),
            trajectories[:, :, :, _Q1_COL].astype(jnp.float32),
            trajectories[:, :, :, _Q2_COL].astype(jnp.float32),
        )

    def state_metrics(trajectories: jax.Array) -> dict[str, jax.Array]:
        q0, q1, q2 = _queues(trajectories)
        return {"eval_final/mean_total_queue": q0 + q1 + q2}

    def state_series(trajectories: jax.Array) -> list[TrajectorySeries]:
        q0, q1, q2 = _queues(trajectories)
        return [
            TrajectorySeries(
                key="trajectories/total_queue",
                title="Total Queue Length",
                ylabel="Queue",
                ylim=(0.0, float(3 * q_max)),
                values=q0 + q1 + q2,
            ),
            TrajectorySeries(
                key="trajectories/q0",
                title="Queue 0",
                ylabel="Queue",
                ylim=queue_frame,
                values=q0,
            ),
            TrajectorySeries(
                key="trajectories/q1",
                title="Queue 1",
                ylabel="Queue",
                ylim=queue_frame,
                values=q1,
            ),
            TrajectorySeries(
                key="trajectories/q2",
                title="Queue 2",
                ylabel="Queue",
                ylim=queue_frame,
                values=q2,
            ),
        ]

    return EvalDomain(
        trajectory_indices=(*queue_indices, cause_index),
        cause_col=_TRAFFIC_CAUSE_COL,
        num_cause_states=num_cause_states,
        cause_accepting_state=cause_accepting_state,
        x_r_obs_ids=x_r_obs_ids,
        x_a_obs_ids=x_a_obs_ids,
        console_keys=(
            "eval_final/return",
            "eval_final/cumulative_reward",
            "eval_final/episode_length",
            "eval_final/mean_reward",
            "eval_final/mean_total_queue",
            "eval_final/mean_distinct_cause_states",
            "eval_final/frac_episodes_in_accept",
            "eval_final/mostly_in_accept",
            "eval_final/frac_steps_in_accept",
            "eval_final/frac_x_a",
            "eval_final/frac_x_r",
        ),
        state_metrics=state_metrics,
        state_series=state_series,
    )
