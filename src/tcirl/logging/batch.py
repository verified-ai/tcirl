"""Shared windowed-metric logging for all RL training loops.

Windowed curves (`train/*`) depend only on `log_interval`, never on the JAX-batch
size (`~tcirl.rl.common.eval_schedule.EvalSchedule.batch_steps`), and are computed by the
`WindowedMetrics` engine (`windowed_metrics`) at every log point.
`BatchMetric`s (`inference/*`) are produced by the agent's outer loop once per `stop_interval`; they
are logged at the end of the batch that completes it, via the `batch_metrics` channel.
"""

from __future__ import annotations

import time
from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from tcirl.config import ScheduleConfig
from tcirl.logging.recorder import PerSeedMetricRecorder, log_live_metrics
from tcirl.logging.wandb import WandbLogger
from tcirl.metrics.smoothing import Smoother, ema_alpha
from tcirl.metrics.spec import BatchMetric, MetricKindError
from tcirl.metrics.windowed import WindowedMetrics
from tcirl.rl.schedules import linear_schedule_value


class BatchLogger:
    """Windowed metric computation + W&B logging across batches.

    The metrics to compute are the run registry's `WindowedMetric`s (`logger.specs`), so adding a
    `WindowedMetric` makes this logger compute it -- no separate registration here. The matching
    per-step signals are fed to `process_batch` (built by `qrm_logging.log_batch`).
    """

    def __init__(
        self,
        *,
        logger: WandbLogger,
        recorder: PerSeedMetricRecorder,
        num_runs: int,
        total_steps: int,
        log_interval: int,
        smoothing_steps: int | None,
        global_epsilon: ScheduleConfig | None,
        learning_rate: float | None,
        q_init: float,
    ) -> None:
        self._logger = logger
        self._recorder = recorder
        self._num_runs = num_runs
        self._total_steps = total_steps
        self._log_interval = log_interval
        # The one epsilon schedule every seed follows, logged as scalars/epsilon. None for TCIRL, whose
        # seeds can restart their schedules at their re-inferences and log them as inference/epsilon.
        self._global_epsilon = global_epsilon
        # The deep arm's constant optimizer rate, logged as scalars/alpha. None for the tabular
        # agents, whose step size belongs to each Q-table entry and is logged as train/step_size.
        self._learning_rate = learning_rate
        self._q_init = q_init

        windowed = logger.specs.windowed_metrics()
        self._engine = WindowedMetrics(num_runs=num_runs, specs=windowed, log_interval=log_interval)
        # Required per-step signal names, checked once against the first batch's signals.
        self._required_signals = self._engine.required_signals()
        self._validated = False

        # Per-seed temporal smoothing of the mean-category metrics, before cross-seed reduction.
        # The recorder is told about EVERY smoothable key of the registry, the eval channel's
        # included, because that flag travels into the JSON and tells the offline export where to
        # apply the EMA. This logger's own `Smoother` covers only what this logger produces: the
        # eval channel has its own cadence and therefore its own weight
        # (`~tcirl.evaluation.ScheduledEvaluation`).
        recorder.mark_smoothable(logger.specs.smoothable_keys)
        windowed_smoothable = frozenset(m.key for m in windowed if m.smoothable)
        alpha = ema_alpha(smoothing_steps, log_interval)
        self._smoother = (
            None if alpha is None else Smoother(alpha=alpha, smoothable_keys=windowed_smoothable)
        )

        # Last log point's per-seed metrics, for the once-per-batch console line (raw jax arrays,
        # or the smoothed numpy arrays when smoothing is on).
        self._last_per_seed: Mapping[str, jax.Array | np.ndarray] = {}

        # Wall-clock at the end of the previous batch cycle, for steps-per-second.
        # Set by log_initial just before the training loop starts.
        self._last_time: float | None = None

    def _schedule_scalars(self, step: int) -> dict[str, float]:
        scalars: dict[str, float] = {}
        if self._global_epsilon is not None:
            eps = self._global_epsilon
            scalars["scalars/epsilon"] = linear_schedule_value(
                step, eps.start, eps.end, eps.decay_steps
            )
        if self._learning_rate is not None:
            scalars["scalars/alpha"] = self._learning_rate
        return scalars

    def log_initial(
        self,
        *,
        batch_metrics: dict[str, jax.Array] | None = None,
    ) -> None:
        """Seed values at step 0, before any window has data.

        `batch_metrics` (the per-batch `inference/*` values) are emitted here too so their curves
        start at step 0, matching the windowed metrics.
        """
        per_seed: dict[str, jax.Array] = {
            "train/qmax": jnp.full(self._num_runs, self._q_init, dtype=jnp.float32),
        }
        # Every running sum starts at 0, so the step-0 row has an x value for the cumulative curves.
        for spec in self._logger.specs.windowed_metrics():
            if spec.category == "cum_count":
                per_seed[spec.key] = jnp.zeros(self._num_runs, dtype=jnp.float32)
        if batch_metrics is not None:
            per_seed.update(batch_metrics)
        smoothed = None if self._smoother is None else self._smoother.update(per_seed)
        log_live_metrics(
            self._logger,
            self._recorder,
            per_seed,
            self._schedule_scalars(0),
            step=0,
            per_seed_smoothed=smoothed,
        )
        self._logger.flush_step(0)
        print(f"Step       0/{self._total_steps} | (initial)")
        self._last_time = time.perf_counter()

    def process_batch(
        self,
        batch_idx: int,
        signals: dict[str, jax.Array],
        *,
        batch_metrics: dict[str, jax.Array] | None = None,
        scalars: dict[str, float] | None = None,
    ) -> None:
        """Window this batch's per-step `signals` and log at each contained log point.

        `signals` are per-step `(num_runs, batch_steps)` arrays (the engine windows them).
        `batch_metrics` (the `inference/*` values) are per-seed arrays merged at the batch-end log
        point (which exists because batch_steps is a multiple of log_interval) and not replicated
        across the batch's sub-batch log points. TCIRL passes them only on the batches that end a
        `stop_interval`, and `None` on the rest, so the curve keeps the cadence of its re-inference.
        Per-seed reduction and the console line are driven by the registry on `logger.specs`.
        """
        if not self._validated:
            missing = self._required_signals - signals.keys()
            if missing:
                raise ValueError(
                    f"WindowedMetrics need per-step signals {sorted(missing)} not supplied this "
                    f"batch (check the registry's WindowedMetric signal names against the core "
                    f"signals + the run's signal_provider); got {sorted(signals.keys())}"
                )
            extra = signals.keys() - self._required_signals
            if extra:
                raise ValueError(
                    f"per-step signals {sorted(extra)} have no registered WindowedMetric consumer "
                    f"(a signal_provider emitted them but no metric in this run's registry reads "
                    f"them, so they would be silently dropped); register the matching WindowedMetric "
                    f"or stop emitting the signal"
                )
            if batch_metrics is not None:
                for base in batch_metrics:
                    spec = self._logger.specs.spec_for(base)
                    if not isinstance(spec, BatchMetric):
                        raise MetricKindError(
                            f"{base!r} is a {type(spec).__name__}; the batch_metrics channel logs "
                            f"only BatchMetrics (windowed metrics come from signals; eval metrics "
                            f"from the eval pass)."
                        )
            self._validated = True
        batch_steps = next(iter(signals.values())).shape[1]
        batch_end = (batch_idx + 1) * batch_steps
        sps_scalars = self._sps_scalars(batch_steps)
        for step, windowed in self._engine.update(signals, batch_idx=batch_idx):
            per_seed = dict(windowed)
            # BatchMetrics are per-batch constants; merge them once, at the batch-end log point
            # (guaranteed to exist since batch_steps is a multiple of log_interval).
            if batch_metrics is not None and step == batch_end:
                per_seed.update(batch_metrics)
            smoothed = None if self._smoother is None else self._smoother.update(per_seed)
            point_scalars = {**self._schedule_scalars(step), **sps_scalars}
            if scalars is not None:
                point_scalars.update(scalars)
            log_live_metrics(
                self._logger,
                self._recorder,
                per_seed,
                point_scalars,
                step=step,
                per_seed_smoothed=smoothed,
            )
            self._logger.flush_step(step)
            # Console shows what is plotted (the smoothed, NaN-masked values when smoothing is on).
            self._last_per_seed = per_seed if smoothed is None else smoothed

        # Console print (once per batch), driven entirely by logger.specs (the MetricRegistry).
        metric_str = self._logger.specs.format_console(
            self._last_per_seed,
            {**self._schedule_scalars(batch_end), **sps_scalars},
        )
        print(f"Step {batch_end:>7d}/{self._total_steps} | {metric_str}")

    def restart_timer(self) -> None:
        """Start the throughput clock again, so what follows is not charged to the next batch.

        `scalars/seed_sps` measures wall time from the start of the previous `process_batch`, which
        is the training of one batch and nothing else -- until an evaluation runs between two
        batches. A run with more than one evaluation point does exactly that, and without this call
        every batch that follows a grid point reports a throughput depressed by the evaluation's
        own wall time. The training loop calls it after it evaluates.
        """
        self._last_time = time.perf_counter()

    def _sps_scalars(self, batch_steps: int) -> dict[str, float]:
        """Steps-per-second over the batch cycle just completed, stamped per call.

        A wall-clock throughput metric -- intentionally NOT batch_steps-invariant like the
        windowed curves; it measures real throughput of the chosen batching. The first batch
        includes JIT compilation, so its point reads low. `total_sps` counts env steps across
        all seeds; `seed_sps` is per-seed (`total_sps / num_runs`).
        """
        now = time.perf_counter()
        elapsed = None if self._last_time is None else now - self._last_time
        self._last_time = now
        if elapsed is None or elapsed <= 0:
            return {}
        seed_sps = batch_steps / elapsed
        return {"scalars/total_sps": seed_sps * self._num_runs, "scalars/seed_sps": seed_sps}
