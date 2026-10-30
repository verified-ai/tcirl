"""Spec-driven windowed-metric engine: per-step signals -> per-log-point per-seed values.

The single guarantee this module exists to enforce: a metric's value at a log point is a
function only of the window of steps ending at that log point (or, for `cum_count`, the
cumulative count) -- NEVER of how the step stream was chunked into JAX batches. So the
logged curves are invariant to `batch_steps` (the JAX-batch size); only `log_interval`
matters. `batch_steps` has no effect here, by construction.

The metrics to compute are the `WindowedMetric`s declared in the run's registry
(`tcirl.metrics.spec`). Each batch the caller feeds per-step signal arrays
`(num_runs, batch_steps)`; the engine
carries a per-signal rolling tail across batches and reduces the window ending at each log point.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tcirl.metrics.spec import TimeCategory, WindowedMetric

# Cumulative categories keep no rolling tail (a persistent scalar carries the running state).
_CUMULATIVE: frozenset[TimeCategory] = frozenset({"cum_max", "cum_count"})


def _tile(full: jax.Array, offset: int, li: int) -> jax.Array:
    """The `li`-step tile ending at `offset` steps into the batch.

    `full = concat(tail[li], batch)`; the tile is `full[:, offset : offset + li]`, the `li` steps
    ending at global step `batch_start + offset`. `offset >= 1`, so the slice never underflows.
    """
    return full[:, offset : offset + li]


def _finite_or_nan(x: jax.Array) -> jax.Array:
    """Turn a not-yet-initialized running max (-inf) into NaN so it is dropped, not plotted."""
    return jnp.where(jnp.isfinite(x), x, jnp.nan)


class WindowedMetrics:
    """Carries per-signal tails and emits per-log-point per-seed metric dicts.

    `num_runs` seeds; `specs` the `WindowedMetric`s to compute; `log_interval` the global-grid log
    cadence and the (only) tile width. NaN-prefilled tails make partial tiles at training start fall
    out of the NaN-skipping mean reductions; the producer-side NaN drop (a seed with no data this
    tile) and key omission (no seed has data) live in `update`. Values depend only on `log_interval`,
    never on the JAX-batch size.
    """

    def __init__(self, *, num_runs: int, specs: list[WindowedMetric], log_interval: int) -> None:
        self._specs = specs
        self._num_runs = num_runs
        self._log_interval = log_interval

        # Mean categories tile over `log_interval`, so each of their signals keeps a tail of that
        # length; cumulative categories keep no tail (a persistent scalar carries the running state).
        windowed_signals: set[str] = set()
        for spec in specs:
            if spec.category in _CUMULATIVE:
                continue
            windowed_signals.update(spec.signals)
        self._tails: dict[str, jax.Array] = {
            name: jnp.full((num_runs, log_interval), jnp.nan, dtype=jnp.float32)
            for name in windowed_signals
        }
        self._cum_count: dict[str, jax.Array] = {
            spec.key: jnp.zeros(num_runs, dtype=jnp.float32)
            for spec in specs
            if spec.category == "cum_count"
        }
        self._cum_max: dict[str, jax.Array] = {
            spec.key: jnp.full(num_runs, -jnp.inf, dtype=jnp.float32)
            for spec in specs
            if spec.category == "cum_max"
        }

    def required_signals(self) -> set[str]:
        """Every per-step signal name the specs reduce (for one-time validation in the log layer)."""
        return {name for spec in self._specs for name in spec.signals}

    def _reduce(
        self,
        spec: WindowedMetric,
        full: dict[str, jax.Array],
        cum_count_batch: dict[str, jax.Array],
        cum_max_batch: dict[str, jax.Array],
        offset: int,
    ) -> jax.Array:
        """Per-seed `(num_runs,)` value for one spec at a log point (may contain NaN)."""
        li = self._log_interval
        if spec.category == "cum_count":
            (sig,) = spec.signals
            return self._cum_count[spec.key] + cum_count_batch[sig][:, offset - 1]
        if spec.category == "cum_max":
            (sig,) = spec.signals
            return _finite_or_nan(
                jnp.maximum(self._cum_max[spec.key], cum_max_batch[sig][:, offset - 1])
            )
        if spec.category == "step_mean":
            (sig,) = spec.signals
            return jnp.nanmean(_tile(full[sig], offset, li), axis=1)
        # episode_mean
        numer_sig, over_sig = spec.signals
        numer = jnp.nansum(_tile(full[numer_sig], offset, li), axis=1)
        count = jnp.nansum(_tile(full[over_sig], offset, li), axis=1)
        return jnp.where(count > 0, numer / count, jnp.nan)

    def update(
        self, batch_signals: dict[str, jax.Array], *, batch_idx: int
    ) -> list[tuple[int, dict[str, jax.Array]]]:
        """Fold one batch of per-step signals; return `(step, per_seed)` for each log point.

        Each `per_seed` dict holds the full `(num_runs,)` array for every spec, NaN where a seed has
        no datum this tile (e.g. an episode metric whose seed completed no episode). Seed identity is
        preserved across log points so the per-seed smoother can carry state; dropping NaN seeds and
        omitting all-NaN metrics is a display concern handled in the log layer.
        """
        li = self._log_interval
        batch_steps = next(iter(batch_signals.values())).shape[1]
        batch_start = batch_idx * batch_steps
        batch_end = batch_start + batch_steps

        full = {
            name: jnp.concatenate([self._tails[name], batch_signals[name]], axis=1)
            for name in self._tails
        }
        cum_count_batch = {
            spec.signals[0]: jnp.cumsum(batch_signals[spec.signals[0]], axis=1)
            for spec in self._specs
            if spec.category == "cum_count"
        }
        cum_max_batch = {
            spec.signals[0]: jax.lax.cummax(
                jnp.where(jnp.isnan(s := batch_signals[spec.signals[0]]), -jnp.inf, s), axis=1
            )
            for spec in self._specs
            if spec.category == "cum_max"
        }

        out: list[tuple[int, dict[str, jax.Array]]] = []
        for k in range(batch_start // li + 1, batch_end // li + 1):
            step = k * li
            offset = step - batch_start  # steps into this batch (1-indexed)
            per_seed = {
                spec.key: self._reduce(spec, full, cum_count_batch, cum_max_batch, offset)
                for spec in self._specs
            }
            out.append((step, per_seed))

        for name in self._tails:
            self._tails[name] = full[name][:, -li:]
        for spec in self._specs:
            if spec.category == "cum_count":
                self._cum_count[spec.key] = (
                    self._cum_count[spec.key] + cum_count_batch[spec.signals[0]][:, -1]
                )
            elif spec.category == "cum_max":
                self._cum_max[spec.key] = jnp.maximum(
                    self._cum_max[spec.key], cum_max_batch[spec.signals[0]][:, -1]
                )
        return out
