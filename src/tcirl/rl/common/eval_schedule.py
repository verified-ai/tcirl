"""The grid of env steps at which a run stops training and evaluates the policy it holds.

One object owns the grid, so the config, the three training loops and the recorder cannot disagree
about which steps carry an evaluation point.

The grid is fixed by ONE number, `num_eval_points`, and it always ends at `total_steps`:

- 1 point is the end of training alone, which is the classic train-then-evaluate split.
- 2 points are the start and the end.
- k points from 2 up are k equidistant steps, the first at 0 and the last at `total_steps`, so 3
  is (start, half-point, end) and 501 is one every 0.2% of the budget.

Two constraints make a multi-point grid land on steps a run can actually stop at, and both raise
here rather than silently rounding:

- `total_steps` must divide into `num_eval_points - 1` equal parts, or the grid is not equidistant.
- that spacing must be a whole number of `log_interval`s, because a training loop surfaces from its
  JAX batch only at a batch boundary and every batch boundary has to be a log point.

What the grid does NOT have to divide is `stop_interval`, and that is the whole reason this object
computes `batch_steps`. `stop_interval` is not a batch size: it is how often the outer loop does its
per-interval work, which for TCIRL is the counterexample check and the re-inference it triggers.
It is a parameter of the method, which the figure resolution must not move. The BATCH is the separate, purely mechanical question of how many steps
one `lax.scan` runs before Python sees the carry again, and the answer is the finest grid both
cadences land on: `gcd(stop_interval, spacing)`. The loop then runs `batches_per_stop` batches
between two rounds of per-interval work, so the work happens at exactly the steps it would at one
batch per interval, while the policy becomes readable at every point of the evaluation grid.

Splitting a scan this way cannot move a number: every per-step quantity lives in the scan carry, so
`batches_per_stop` scans of `batch_steps` thread the same carry through the same steps as one scan
of `stop_interval`.

At `num_eval_points: 1` the spacing does not exist, `batch_steps` IS `stop_interval`, and every loop
runs one batch per interval.

The two domains of this repo run the same `total_steps` and the same `log_interval`, so the same
`num_eval_points` gives them one shared x axis and a reader can put their two evaluation curves side
by side. They do NOT share `stop_interval`, and with `batch_steps` derived they need not.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from tcirl.rl.common.greedy_eval import EvalRollout, GreedyPolicy, build_greedy_eval
from tcirl.rl.common.tabular import TabularEnv


@dataclass(frozen=True, slots=True, kw_only=True)
class EvalSchedule:
    """Which env steps carry an evaluation point, and how the training batch is cut to reach them."""

    total_steps: int
    num_eval_points: int
    stop_interval: int
    log_interval: int

    def __post_init__(self) -> None:
        if self.total_steps <= 0:
            raise ValueError("total_steps must be > 0")
        if self.stop_interval <= 0:
            raise ValueError("stop_interval must be > 0")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be > 0")
        if self.stop_interval % self.log_interval:
            raise ValueError(
                f"stop_interval ({self.stop_interval}) must be a multiple of log_interval "
                f"({self.log_interval}); every batch boundary has to be a log point"
            )
        if self.num_eval_points < 1:
            raise ValueError(
                f"num_eval_points is {self.num_eval_points}; it must be >= 1, because a run always "
                f"evaluates the policy it finishes with"
            )
        # The grid's last point is `total_steps`, so the end of training has to be a batch boundary
        # for ANY grid size, the one-point grid included. Without this, a one-point grid is accepted
        # and then never fires, and the run only finds out at `ScheduledEvaluation.finish`.
        if self.total_steps % self.stop_interval:
            raise ValueError(
                f"total_steps ({self.total_steps}) must be a multiple of stop_interval "
                f"({self.stop_interval}); the grid ends at total_steps and a training loop can "
                f"read its policy only at a batch boundary"
            )
        if self.num_eval_points == 1:
            return
        spacing, remainder = divmod(self.total_steps, self.num_eval_points - 1)
        if remainder:
            raise ValueError(
                f"total_steps ({self.total_steps}) is not divisible by num_eval_points - 1 "
                f"({self.num_eval_points - 1}), so the {self.num_eval_points} points are not "
                f"equidistant"
            )
        if spacing % self.log_interval:
            raise ValueError(
                f"the evaluation spacing ({spacing} steps) must be a multiple of log_interval "
                f"({self.log_interval}); a batch ends at a grid point and every batch boundary has "
                f"to be a log point"
            )

    @property
    def spacing(self) -> int:
        """The env steps between two neighbouring grid points; `total_steps` for a one-point grid."""
        if self.num_eval_points == 1:
            return self.total_steps
        return self.total_steps // (self.num_eval_points - 1)

    @property
    def batch_steps(self) -> int:
        """How many env steps one `lax.scan` runs before the loop can read the policy again.

        The finest grid both cadences land on, so every grid point AND every `stop_interval`
        boundary is a batch boundary. A one-point grid leaves this at `stop_interval`, one batch per
        interval.
        """
        return math.gcd(self.stop_interval, self.spacing)

    @property
    def batches_per_stop(self) -> int:
        """How many batches the loop runs between two rounds of its per-`stop_interval` work."""
        return self.stop_interval // self.batch_steps

    @property
    def num_batches(self) -> int:
        """How many batches the whole run takes."""
        return self.total_steps // self.batch_steps

    @property
    def steps(self) -> tuple[int, ...]:
        """Every step of the grid, in order, always ending at `total_steps`."""
        if self.num_eval_points == 1:
            return (self.total_steps,)
        return tuple(i * self.spacing for i in range(self.num_eval_points))

    @property
    def evaluates_before_training(self) -> bool:
        """True when the grid holds step 0, the untrained policy."""
        return self.num_eval_points > 1

    def is_eval_batch(self, batch_idx: int) -> bool:
        """True when the batch with this 0-based index ENDS on a point of the grid."""
        return self.step_after_batch(batch_idx) in self.steps

    def is_stop_batch(self, batch_idx: int) -> bool:
        """True when the batch with this 0-based index ends a whole `stop_interval`.

        The outer loop's per-interval work runs here and nowhere else, so cutting the batch finer
        to reach a grid point cannot change how often that work happens.
        """
        return self.step_after_batch(batch_idx) % self.stop_interval == 0

    def step_after_batch(self, batch_idx: int) -> int:
        """The env step the batch with this 0-based index ends at."""
        return (batch_idx + 1) * self.batch_steps

    def is_last(self, step: int) -> bool:
        """True for the grid's final point, the only one that also records the trajectory curves."""
        return step == self.total_steps


class EvalSink(Protocol):
    """What a training loop needs from whatever collects its evaluation points.

    The loop knows WHEN to evaluate and WHAT policy it holds; it must not know what a `trajectories`
    column means or which W&B series a share ends up in. So it talks to this narrow interface, and
    `~tcirl.evaluation.ScheduledEvaluation` is the one implementation of it.
    """

    schedule: EvalSchedule

    @property
    def eval_episodes(self) -> int:
        """How many greedy episodes each seed plays at each point."""
        ...

    @property
    def trajectory_indices(self) -> tuple[int, ...]:
        """The state components the rollout has to track, which the domain fixes."""
        ...

    def record(self, step: int, rollout: EvalRollout) -> None:
        """Hand one evaluation point in, at a step the schedule holds."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class EvalDriver[ParamsT]:
    """Runs one arm's evaluation at the points of its schedule, and nothing anywhere else.

    A training loop asks this object twice -- once before it trains and once after every batch --
    and never asks whether an evaluation is configured, which step the point falls on, or how the
    policy is rolled out. With no sink both calls do nothing, so the loop has ONE code path whether
    or not the run measures its policy.

    `ParamsT` is the arm's per-seed evaluable parameters: a `TabularParams` pair of a Q-table and its
    DFA, or a network's `nnx.GraphState`. It is NOT a `GreedyPolicy`'s `P`, which is the state the policy
    carries between steps of one episode.
    """

    sink: EvalSink | None
    run: Callable[[ParamsT], EvalRollout] | None

    def __post_init__(self) -> None:
        # `build_eval_driver` sets both or neither, and the two methods below rely on that: a sink
        # with nothing to run would silently skip every point of a grid the run declared.
        if (self.sink is None) != (self.run is None):
            raise ValueError("a driver needs a sink and a compiled rollout together, or neither")

    def at_start(self, params: ParamsT) -> None:
        """Evaluate the untrained policy, if the grid holds step 0."""
        if self.sink is None:
            return
        assert self.run is not None
        if self.sink.schedule.evaluates_before_training:
            self.sink.record(0, self.run(params))

    def after_batch(self, batch_idx: int, params: ParamsT) -> None:
        """Evaluate the policy the batch just finished training, if the batch ends on the grid."""
        if self.sink is None:
            return
        assert self.run is not None
        if self.sink.schedule.is_eval_batch(batch_idx):
            self.sink.record(self.sink.schedule.step_after_batch(batch_idx), self.run(params))


def build_eval_driver[ParamsT, PolicyStateT](
    schedule: EvalSchedule,
    sink: EvalSink | None,
    env: TabularEnv,
    policy: GreedyPolicy[ParamsT, PolicyStateT],
    *,
    gamma: float,
    num_runs: int,
    max_episode_steps: int,
) -> EvalDriver[ParamsT]:
    """The driver for one arm: its policy compiled once, pointed at the run's sink.

    The compilation happens here rather than at each point, so a 501-point grid traces the rollout
    once and not 501 times. With no sink nothing is compiled either.

    `schedule` is the grid the TRAINING LOOP cut its batches to. The sink carries one too, and the
    two have to be the same object's worth of numbers: a loop batching for one grid while the sink
    records against another would drop points silently, and only `ScheduledEvaluation.finish` would
    ever notice.
    """
    if sink is None:
        return EvalDriver(sink=None, run=None)
    if sink.schedule != schedule:
        raise ValueError(
            f"the training loop batches for {schedule} and the sink records against "
            f"{sink.schedule}; both must come from one `tcirl.config.eval_schedule` call"
        )
    return EvalDriver(
        sink=sink,
        run=build_greedy_eval(
            env,
            policy,
            gamma=gamma,
            num_runs=num_runs,
            eval_episodes=sink.eval_episodes,
            max_episode_steps=max_episode_steps,
            trajectory_indices=sink.trajectory_indices,
        ),
    )
