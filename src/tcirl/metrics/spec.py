"""The declarative metric catalog: one typed object per metric, in one registry.

Each metric is declared exactly once as an immutable `Metric` -- a tagged union over the four
production channels, each carrying its own display config. The engines consume this single registry:
the windowed engine reads the `WindowedMetric`s, the recorder/console/W&B layers read the per-seed
display, and the scalar channel reads `ScalarMetric`s. Invalid (production x display) combinations
are unrepresentable by construction.

Each per-seed kind has exactly one emission channel, and the channel validates the kind:

- `WindowedMetric`  -- produced by the `WindowedMetrics` engine from per-step signals; a per-seed
  curve with a `TimeCategory` (step/episode mean, or cumulative max/count) and the signal names it
  reduces. Emitted at every log point. Carries a `SeedDisplay`.
- `BatchMetric`     -- a per-seed value handed in verbatim once per `stop_interval`, at the end of
  the batch that completes it (`inference/*` from the agent outer loop). It is logged at that batch
  boundary rather than replicated across the batch's sub-batch log points, so its cadence is
  `stop_interval`, like the re-inference it reports. Carries a `SeedDisplay`.
- `EvalMetric`      -- a per-seed value the greedy evaluation pass produces (`eval_final/*`), one
  point per seed at each step of the run's `~tcirl.rl.common.eval_schedule.EvalSchedule`. At
  `num_eval_points: 1` that grid is the single step `total_steps`, which is the end-of-run reading
  the prefix is named after; the shipped 501 makes the same metric a curve over env steps.
  Carries a `SeedDisplay`.
- `ScalarMetric`    -- a run-global value logged verbatim (`scalars/*`); no per-seed array.

`MetricRegistry` catalogs the declarations by key (the single control point attached to the one
`WandbLogger`): cross-seed reduction, W&B series naming, console rendering, and which metrics the
windowed engine must compute. An unregistered base key raises `UnknownMetricError`; logging a metric
through the wrong channel raises `MetricKindError`. `hidden=True` keeps a metric registered but drops
it from W&B and console.

W&B series names decorate the registry key so the reduction is legible from the name: a windowed
metric inserts its TimeCategory code (step step_mean, episode episode_mean, max cum_max, count
cum_count) as the second path segment, and every per-seed metric appends the stat as the final
segment, always (no suffix-dropping for a unique stat). So `train/reward` (step_mean, mean) ->
`train/step/reward/mean`, `train/num_episodes` (cum_count, median) -> `train/count/num_episodes/median`, and
`inference/hypothesis_states` (mean, max; no category) -> `inference/hypothesis_states/{mean,max}`.
Scalars (`scalars/epsilon`) carry no stat and are logged under their verbatim key.

Layout is enforced at construction (`ValueError`): every top-level prefix holds exactly one metric
kind (no mixing), `scalars/` is reserved for `ScalarMetric`, and `eval_final/` for `EvalMetric`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np

from tcirl.metrics.stats import ArrayLike, Stat, summarize

# A metric's TIME CATEGORY: how a seed's raw per-step signals become one value per log point. See
# `tcirl.metrics.windowed` for the reductions; the only window is the `log_interval` tile.
#   step_mean    -- mean over the tile's steps (reward, td_error, qmax).
#   episode_mean -- sum(numerator) / count(episode_over) over the tile; NaN when no episode completed.
#   cum_max      -- running max over all steps so far (NaN-ignoring): the monotone "ever reached".
#   cum_count    -- running sum of the signal over all steps so far (num_episodes).
TimeCategory = Literal["step_mean", "episode_mean", "cum_max", "cum_count"]

# The mean categories: a temporal EMA (see `tcirl.metrics.smoothing`) is meaningful only here.
_SMOOTHABLE: frozenset[TimeCategory] = frozenset({"step_mean", "episode_mean"})

# W&B name code for each windowed TimeCategory, inserted as the second path segment of the series
# name so a curve's reduction is legible from its name (`train/reward` step_mean ->
# `train/step/reward`): step = mean per env step over the tile, episode = mean per completed
# episode over the tile, max = running max since step 0, count = running sum since step 0.
_CATEGORY_CODE: dict[TimeCategory, str] = {
    "step_mean": "step",
    "episode_mean": "episode",
    "cum_max": "max",
    "cum_count": "count",
}


class UnknownMetricError(KeyError):
    """Raised when a metric is logged with no registered `Metric`."""


class MetricKindError(TypeError):
    """Raised when a metric is logged through a channel its kind does not allow."""


@dataclass(frozen=True, slots=True, kw_only=True)
class SeedDisplay:
    """How a per-seed array is reduced across seeds and shown (shared by the per-seed variants).

    `stats`    cross-seed statistics emitted to W&B; each one, a single stat included, suffixes the series name.
    `console`  which stat the CLI shows, or `None` to hide it from the console.
    `fmt`      console number format.
    `short`    console label (defaults to the metric's last path segment).
    `hidden`   drop from W&B and console while staying registered.
    """

    stats: tuple[Stat, ...] = ("mean",)
    console: Stat | None = None
    fmt: str = ".2f"
    short: str | None = None
    hidden: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class WindowedMetric:
    """A per-seed curve the `WindowedMetrics` engine reduces from per-step `signals`.

    For `episode_mean`, `signals` is `(numerator, episode_over)`; otherwise a single signal name.
    The engine reads `key`/`category`/`signals` only; `display` drives the downstream layers.
    """

    key: str
    category: TimeCategory
    signals: tuple[str, ...]
    display: SeedDisplay

    @property
    def smoothable(self) -> bool:
        """True iff this is a mean category, where a temporal EMA is meaningful."""
        return self.category in _SMOOTHABLE


@dataclass(frozen=True, slots=True, kw_only=True)
class BatchMetric:
    """A per-seed value handed in verbatim once per `stop_interval`, at the end of the batch that completes it.

    The agent's outer loop produces it after each counterexample check (the current hypothesis
    size, for example). It is logged at that batch boundary rather than replicated across the
    batch's sub-batch log points. No engine, no smoothing.
    """

    key: str
    display: SeedDisplay


@dataclass(frozen=True, slots=True, kw_only=True)
class EvalMetric:
    """A per-seed value the greedy evaluation produces at each point of the evaluation grid.

    One point per seed at each step of the run's `~tcirl.rl.common.eval_schedule.EvalSchedule`. A
    one-point grid, `(total_steps,)`, makes the metric the single end-of-run reading the
    `eval_final/` prefix is named after; a larger grid turns the same metric into a curve over env
    steps, which differs from a `WindowedMetric`'s curve in WHAT it measures: a windowed metric
    reduces the behaviour the exploring agent produced while training, and this one measures the
    greedy policy the agent holds at that step, on a fixed bank of episodes.

    No engine and no per-batch cadence. `smoothable` is DECLARED here rather than derived, because
    an eval metric has no `TimeCategory` to derive it from, and because the right answer differs
    between two metrics that look identical (both are per-seed means over the same 50 episodes).
    The three kinds are the ones `tcirl.paper_results.PAPER_STATISTIC` classifies, and they split
    two against two:

    - a SCORE (the per-episode return, the per-step reward, the episode length) is SMOOTHED. It is
      plotted as a curve against env steps, so it carries the same EMA horizon as the training
      panel beside it and the two are read the same way.
    - a RECORDED STATE VARIABLE averaged over a whole evaluation (`eval_final/mean_disease` and its
      kind) is SMOOTHED for the same reason: it is a curve over env steps and nothing reads its
      last row as a point. Its per-TIMESTEP form, `trajectories/*`, is never smoothed, because
      there the x axis is the timestep inside one episode and the profile is the measurement, not a
      noisy estimate of it.
    - a SHARE or a COUNT is NOT smoothed, because the paper's figure of evaluation trajectories that reach
      Phase 2 reads the last row of one of these as the run's final measurement, and a time average over the last few checkpoints is not that
      measurement.

    Each point already rests on 2500 episodes, so the point-to-point movement an EMA removes is
    the greedy policy changing between checkpoints. An analysis that wants that signal reads
    `ExportRun.per_seed_curves`, which is raw; `tcirl.export.plotted_curves` is the smoothed view
    the figures draw. The W&B run summary of a smoothed metric therefore carries its smoothed final
    value, which nothing in this repository reads back.
    """

    key: str
    display: SeedDisplay
    smoothable: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class ScalarMetric:
    """A run-global value logged verbatim: no per-seed array, no cross-seed reduction.

    `console` is a bool (show the raw value or not), unlike the per-seed variants where it selects a
    statistic.
    """

    key: str
    console: bool = True
    fmt: str = ".3f"
    short: str | None = None
    hidden: bool = False


Metric = WindowedMetric | BatchMetric | EvalMetric | ScalarMetric

# The per-seed variants: all carry a `SeedDisplay` and flow through the cross-seed reducers.
_PerSeedMetric = (WindowedMetric, BatchMetric, EvalMetric)


def _segment(key: str) -> str:
    return key.rpartition("/")[2]


def _decorated_base(spec: WindowedMetric | BatchMetric | EvalMetric) -> str:
    """The registry key with the windowed category code inserted after the top-level segment.

    Windowed metrics gain the step/episode/max/count code (`train/reward` -> `train/step/reward`);
    batch and eval metrics have no TimeCategory, so the key is returned unchanged.
    """
    if isinstance(spec, WindowedMetric):
        top, _, rest = spec.key.partition("/")
        return f"{top}/{_CATEGORY_CODE[spec.category]}/{rest}"
    return spec.key


@dataclass(frozen=True, slots=True, kw_only=True)
class MetricRegistry:
    """An ordered catalog of `Metric`s; the run's single control point.

    Tuple order is console render order. The key lives on each `Metric`; lookups are linear scans
    over `metrics` (registries are small and lookups run per log point, not per env step), and
    duplicate keys are rejected at construction.
    """

    metrics: tuple[Metric, ...]

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for m in self.metrics:
            if m.key in seen:
                raise ValueError(f"duplicate metric key {m.key!r} in registry")
            seen.add(m.key)
        self._validate_layout()

    def _validate_layout(self) -> None:
        """Enforce the naming layout: reserved prefixes and one metric kind per top-level segment.

        `scalars/` is reserved for `ScalarMetric` and `eval_final/` for `EvalMetric` (each kind lives
        only there, and that prefix holds only that kind); no top-level segment mixes two kinds.
        Raises `ValueError` on any violation, so a mis-prefixed or mixed registry fails at construction.
        """
        kind_by_top: dict[str, type[Metric]] = {}
        for m in self.metrics:
            top, sep, _ = m.key.partition("/")
            if not sep:
                raise ValueError(f"metric key {m.key!r} needs a top-level prefix, e.g. 'train/...'")
            if isinstance(m, ScalarMetric) != (top == "scalars"):
                raise ValueError(
                    f"{m.key!r}: 'scalars/' is reserved for ScalarMetric and every ScalarMetric "
                    f"must use it"
                )
            if isinstance(m, EvalMetric) != (top == "eval_final"):
                raise ValueError(
                    f"{m.key!r}: 'eval_final/' is reserved for EvalMetric and every EvalMetric "
                    f"must use it"
                )
            prev = kind_by_top.get(top)
            if prev is not None and prev is not type(m):
                raise ValueError(
                    f"top-level {top!r}/ mixes {prev.__name__} and {type(m).__name__}; each "
                    f"top-level prefix holds exactly one metric kind"
                )
            kind_by_top[top] = type(m)

    @property
    def smoothable_keys(self) -> frozenset[str]:
        """Every key whose per-seed curve carries the run's EMA, across all metric kinds.

        The recorder writes this into the JSON so the offline export applies the EMA exactly where
        the live run did. One place answers the question, so a windowed metric and an eval metric
        cannot end up under two different rules.
        """
        return frozenset(
            m.key
            for m in self.metrics
            if isinstance(m, (WindowedMetric, EvalMetric)) and m.smoothable
        )

    def get(self, key: str) -> Metric | None:
        """The registered `Metric` for a key, or `None` (a linear scan; registries are small)."""
        for m in self.metrics:
            if m.key == key:
                return m
        return None

    def keys(self) -> frozenset[str]:
        """All registered base keys."""
        return frozenset(m.key for m in self.metrics)

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None

    def spec_for(self, base: str) -> Metric:
        """Return the `Metric` for a base path, or raise `UnknownMetricError`."""
        spec = self.get(base)
        if spec is None:
            raise UnknownMetricError(
                f"No Metric registered for {base!r}. Register it in your project's "
                f"registry (and merge it into the run's), or set hidden=True to keep "
                f"it registered but silent."
            )
        return spec

    def windowed_metrics(self) -> list[WindowedMetric]:
        """The `WindowedMetric`s in registry order; the windowed engine computes exactly these."""
        return [m for m in self.metrics if isinstance(m, WindowedMetric)]

    def wandb_series(self, base: str) -> list[tuple[str, Stat]]:
        """W&B `(series_name, stat)` pairs for a per-seed metric.

        The series name decorates the registry key: a windowed metric inserts its TimeCategory code
        (step/episode/max/count) as the second path segment, and every per-seed metric appends the
        stat as the final segment -- always, even for a single stat. So `train/reward` (step_mean,
        mean) -> `train/step/reward/mean`, and `inference/hypothesis_states` (mean, max; no
        category) -> `inference/hypothesis_states/{mean,max}`. `[]` if hidden.
        """
        spec = self.spec_for(base)
        if not isinstance(spec, _PerSeedMetric):
            raise MetricKindError(
                f"{base!r} is a {type(spec).__name__}; log it through its own channel, "
                f"not as a per-seed array."
            )
        display = spec.display
        if display.hidden:
            return []
        name = _decorated_base(spec)
        return [(f"{name}/{stat}", stat) for stat in display.stats]

    def emit_scalar(self, base: str) -> bool:
        """Validate a scalar key's kind and report whether it should be emitted."""
        spec = self.spec_for(base)
        if not isinstance(spec, ScalarMetric):
            raise MetricKindError(
                f"{base!r} is registered as {type(spec).__name__}, not ScalarMetric; "
                f"it cannot be logged as a verbatim scalar."
            )
        return not spec.hidden

    def merged_with(self, override: MetricRegistry) -> MetricRegistry:
        """Compose two registries, `override` winning on key collisions."""
        return self.extended(override.metrics)

    def extended(self, extra: tuple[Metric, ...]) -> MetricRegistry:
        """Return a new registry with `extra` added/overridden.

        Existing keys keep their position; new keys append, so the console render order is this
        registry's order followed by the additions.
        """
        by_key: dict[str, Metric] = {m.key: m for m in self.metrics}
        order: list[str] = [m.key for m in self.metrics]
        for m in extra:
            if m.key not in by_key:
                order.append(m.key)
            by_key[m.key] = m  # override wins, original position preserved
        return MetricRegistry(metrics=tuple(by_key[k] for k in order))

    def format_console(
        self, per_seed: Mapping[str, ArrayLike], scalars: Mapping[str, float]
    ) -> str:
        """Build the CLI metric line (the part after `Step N/T |`) from the registry.

        Renders each console-visible metric in registry order, so every algorithm prints an
        identically-ordered line and differs only by which metrics it produced this batch (absent
        metrics are skipped).
        """
        parts: list[str] = []
        for m in self.metrics:
            if isinstance(m, ScalarMetric):
                if m.hidden or not m.console or m.key not in scalars:
                    continue
                label = m.short if m.short is not None else _segment(m.key)
                parts.append(f"{label}: {scalars[m.key]:{m.fmt}}")
                continue
            display = m.display
            if display.hidden or display.console is None or m.key not in per_seed:
                continue
            arr = np.asarray(per_seed[m.key], dtype=np.float64)
            finite = arr[~np.isnan(arr)]
            if finite.size == 0:
                continue  # no seed has data at this point
            value = summarize(finite, display.console)
            label = display.short if display.short is not None else _segment(m.key)
            parts.append(f"{label}: {value:{display.fmt}}")
        return " | ".join(parts)


# Universal metrics the shared algorithms (Q-learning) emit. Registries merge their own
# registry over this one; order here is console render order. The three per-seed groups differ only
# in their emission channel: train/* are windowed (BatchLogger engine, every log point), inference/*
# are BatchMetrics (agent outer loop, once per stop_interval), eval_final/* are EvalMetrics (one greedy
# evaluation pass per point of the run's evaluation grid, which the shipped configs set to 501
# points).
BASE_SPECS = MetricRegistry(
    metrics=(
        # Scalars: run-global verbatim values (no per-seed array, no cross-seed reduction).
        ScalarMetric(key="scalars/epsilon", console=True, fmt=".3f", short="eps"),
        ScalarMetric(key="scalars/total_sps", hidden=True, console=True, fmt=".0f", short="sps"),
        ScalarMetric(key="scalars/seed_sps", console=True, fmt=".0f", short="sps/seed"),
        # train/*: windowed per-seed curves (the engine reduces the per-step signals over the tile).
        # Every train/* curve, including the optimizer diagnostics (td_error, qmax, loss,
        # grad_norm), reduces across seeds by the mean.
        WindowedMetric(
            key="train/reward",
            category="step_mean",
            signals=("reward",),
            display=SeedDisplay(stats=("mean",), console="mean", fmt=".4f"),
        ),
        # NOT comparable between the tabular and the deep arms, and no paper figure reads it.
        # The tabular iteration reports the |TD| of the step's own visited transition; the deep one
        # reports the mean |TD| over its 64-row replay batch, and NaN before the learn gate opens
        # (which the step_mean tile drops). Both are the right diagnostic for their own optimizer.
        WindowedMetric(
            key="train/td_error",
            category="step_mean",
            signals=("td_abs",),
            display=SeedDisplay(stats=("mean",), console=None, fmt=".4f", short="td", hidden=True),
        ),
        WindowedMetric(
            key="train/qmax",
            category="step_mean",
            signals=("q_max",),
            display=SeedDisplay(stats=("mean",), console=None, hidden=True),
        ),
        WindowedMetric(
            key="train/num_episodes",
            category="cum_count",
            signals=("episode_over",),
            display=SeedDisplay(stats=("median",), console="median", fmt=".0f", hidden=True),
        ),
        WindowedMetric(
            key="train/episode_length",
            category="episode_mean",
            signals=("ep_length", "episode_over"),
            display=SeedDisplay(stats=("mean",), console="mean", fmt=".1f", short="ep_len"),
        ),
        # The two per-episode returns. Both read a signal the iteration emits ONCE per episode, at the
        # step that ends it (`ep_return`, `ep_disc_return`), so `sum(signal) / count(episode_over)` over a
        # window is the plain mean return of the episodes that FINISHED in that window, as `ep_length` is the
        # mean length of those episodes.
        #
        # `train/discounted_return` is the closest training-side estimate of the objective, and it is
        # NOT the objective itself. Both learners bootstrap THROUGH the 100-step truncation, so what
        # they maximise is the infinite-horizon discounted return; this metric stops the sum at the
        # truncation, and gamma ** 100 = 0.366 of the weight sits beyond it. Read it as the 100-step
        # truncated discounted return. `train/cumulative_reward` is the same sum without the
        # discount, and the paper's reward panels plot it, because it is the quantity a reader can
        # compare against the episode ceiling of 100.
        #
        # The per-STEP alternative, `train/reward`, is a rate rather than a return: it divides the same
        # rewards by the number of steps, so an agent that ends its episode early scores well on it while
        # collecting almost nothing.
        #
        # Both reduce across seeds by the mean.
        WindowedMetric(
            key="train/cumulative_reward",
            category="episode_mean",
            signals=("ep_return", "episode_over"),
            display=SeedDisplay(stats=("mean",), console="mean", fmt=".2f", short="ep_ret"),
        ),
        WindowedMetric(
            key="train/discounted_return",
            category="episode_mean",
            signals=("ep_disc_return", "episode_over"),
            display=SeedDisplay(stats=("mean",), console="mean", fmt=".2f", short="ep_disc"),
        ),
        # inference/*: per-seed BatchMetrics handed in verbatim once per stop_interval. Hypothesis
        # size shows whether the hypotheses stay small: the `mean` is average model complexity, and
        # the `/max` series and the figure's min-max envelope show the largest seed.
        # `reinferences` (re-solve cost) is mean-only.
        BatchMetric(
            key="inference/hypothesis_states",
            display=SeedDisplay(stats=("mean", "max"), console="mean", fmt=".1f", short="hyp"),
        ),
        # Throughput count: report the typical seed (median), like train/num_episodes.
        BatchMetric(
            key="inference/traces",
            display=SeedDisplay(stats=("median",), console="median", fmt=".1f"),
        ),
        BatchMetric(
            key="inference/reinferences",
            display=SeedDisplay(stats=("mean",), console="mean", fmt=".1f", short="reinf"),
        ),
        # The exploration rate each TCIRL seed uses: its schedule restarts at each re-inference that
        # restarts the Q-table, so no run-global `scalars/epsilon` exists for TCIRL.
        BatchMetric(
            key="inference/epsilon",
            display=SeedDisplay(stats=("mean",), console="mean", fmt=".3f", short="eps"),
        ),
        # The share of the update counts of each TCIRL seed's table before its last re-inference
        # that lie in the rows of the old states whose values some new state took; 0 before the
        # first re-inference and under the value transfer rule "none".
        BatchMetric(
            key="inference/kept_updates",
            display=SeedDisplay(stats=("mean",), console=None, fmt=".2f"),
        ),
        # Task-outcome metrics reduce across seeds by the mean, with its bootstrap interval
        # (`stats.py`).
        EvalMetric(
            key="eval_final/return",
            display=SeedDisplay(stats=("mean",), console=None, fmt=".4f"),
            smoothable=True,
        ),
        EvalMetric(
            key="eval_final/episode_length",
            display=SeedDisplay(stats=("mean",), console="mean", fmt=".1f", short="ep_len"),
            smoothable=True,
        ),
    )
)

# Additional metrics the tabular agents emit on top of the universal set. Each Q-table entry has its
# own step size, a function of its own update count (`rl.schedules.rescaled_linear_step_size`), so
# no run-global rate exists; `train/step_size` is the mean step size of the window's updates.
BASE_TABULAR_SPECS = BASE_SPECS.extended(
    (
        WindowedMetric(
            key="train/step_size",
            category="step_mean",
            signals=("step_size",),
            display=SeedDisplay(stats=("mean",), console="mean", fmt=".4f", short="alpha"),
        ),
    )
)

# The metric TCIRL emits on top of the tabular set: the mean number of Q-table entries one transition
# updates, 1 without counterfactual updates, and with them the number of live hypothesis states in
# the visited state's acceptance class.
TCIRL_TRAIN_METRICS = (
    WindowedMetric(
        key="train/updated_states",
        category="step_mean",
        signals=("updated_states",),
        display=SeedDisplay(stats=("mean",), console=None, fmt=".2f"),
    ),
)
BASE_TCIRL_SPECS = BASE_TABULAR_SPECS.extended(TCIRL_TRAIN_METRICS)

# Additional metrics the deep DQN baseline emits on top of the universal set. `train/loss` and
# `train/grad_norm` are per-step signals that are NaN on env steps with no gradient step, so the
# step_mean tile reduces over learn steps only (nanmean). `scalars/alpha` is the constant
# optimizer learning rate.
BASE_DQN_SPECS = BASE_SPECS.extended(
    (
        ScalarMetric(key="scalars/alpha", console=True, fmt=".2e", short="lr"),
        WindowedMetric(
            key="train/loss",
            category="step_mean",
            signals=("loss",),
            display=SeedDisplay(stats=("mean",), console="mean", fmt=".4f"),
        ),
        WindowedMetric(
            key="train/grad_norm",
            category="step_mean",
            signals=("grad_norm",),
            display=SeedDisplay(
                stats=("mean",), console=None, fmt=".2f", short="grad", hidden=True
            ),
        ),
    )
)
