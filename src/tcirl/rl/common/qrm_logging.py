"""The one adapter from an iteration's per-step outputs to the metric stack.

`log_batch` turns a `~tcirl.rl.common.step_outputs.TrainStepOutputs` into the per-step signal dict
the windowed engine reads, so the tabular and the deep iteration emit identical training metrics.
That is all this module does; the eval side has its own channel
(`~tcirl.logging.recorder.log_eval_metrics`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tcirl.logging.batch import BatchLogger
from tcirl.rl.common.step_outputs import TrainStepOutputs


def log_batch(
    bl: BatchLogger,
    batch_idx: int,
    outputs: TrainStepOutputs,
    *,
    batch_metrics: dict[str, jax.Array] | None = None,
    extra_scalars: dict[str, float] | None = None,
    extra_signals: dict[str, jax.Array] | None = None,
) -> None:
    """Build per-step signals from `outputs` and forward to `BatchLogger.process_batch`.

    `outputs` is any `~tcirl.rl.common.step_outputs.TrainStepOutputs`: the tabular iteration's
    `TabularOutputs` and the deep iteration's `StepOutputs` both satisfy it, which is why the two
    arms share one logging, reducing and export path.

    `batch_metrics` are the per-batch-constant per-seed metrics (`inference/*`), logged once per
    batch at the batch's end step; `extra_scalars` are run-global scalars; `extra_signals` are
    additional per-step `(num_runs, batch_steps)` arrays the caller already holds -- the deep
    arm's `loss` and `grad_norm`, whose producer is not the iteration's own output. Every key must
    have a matching `WindowedMetric` or `BatchMetric` in the run's registry.
    """
    signals: dict[str, jax.Array] = {
        "reward": outputs.reward,
        "td_abs": outputs.td_abs,
        "q_max": outputs.q_max,
        "episode_over": outputs.episode_over.astype(jnp.float32),
        "ep_length": outputs.ep_length,
        "ep_return": outputs.ep_return,
        "ep_disc_return": outputs.ep_disc_return,
    }
    if extra_signals is not None:
        signals.update(extra_signals)
    bl.process_batch(
        batch_idx,
        signals,
        batch_metrics=batch_metrics,
        scalars=extra_scalars,
    )
