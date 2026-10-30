"""Cross-seed aggregation: point statistics (mean/median/min/max) plus a bootstrap CI of the mean.

The mean is the centre the export and the paper plot, and it carries a 95% percentile bootstrap CI.
The median, min and max get no interval: a median CI is unreliable at the seed counts used here, and an order-statistic CI is not meaningful.

`compute_iqm` and `iqm_interval` compute the interquartile mean, and `iqm_interval` draws the same resamples as `aggregate_stats` for the same key.

Every interval here is a bootstrap. A metric whose per-seed value is 0 or 1 takes the Wilson
interval of `tcirl.metrics.intervals` instead; `tcirl.paper_results.PAPER_INTERVAL` names those
metrics, and the projection replaces this module's CI columns for them with the Wilson bounds.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

ArrayLike = Sequence[float] | np.ndarray | jax.Array

# The cross-seed statistics a per-seed metric can be reduced by. Keep in sync
# with the keys of STAT_REDUCERS below, the runtime source of truth; this alias
# is its static mirror.
Stat = Literal["mean", "max", "min", "median"]


def _as_1d(values: ArrayLike) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D array, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError("Cannot summarize an empty array")
    return arr


def compute_iqm(values: ArrayLike) -> float:
    """Interquartile mean of a 1D array-like collection, for analyses that report the interquartile mean."""
    arr = _as_1d(values)
    sorted_vals = np.sort(arr)
    lo = arr.size // 4
    hi = arr.size - lo
    return float(sorted_vals[lo:hi].mean())


def compute_mean(values: ArrayLike) -> float:
    """Arithmetic mean across seeds."""
    return float(_as_1d(values).mean())


def compute_max(values: ArrayLike) -> float:
    return float(_as_1d(values).max())


def compute_min(values: ArrayLike) -> float:
    return float(_as_1d(values).min())


def compute_median(values: ArrayLike) -> float:
    return float(np.median(_as_1d(values)))


# Cross-seed reducers, keyed by the statistic name used in `<prefix>/<stat>/<metric>`.
STAT_REDUCERS: dict[Stat, Callable[[ArrayLike], float]] = {
    "mean": compute_mean,
    "max": compute_max,
    "min": compute_min,
    "median": compute_median,
}


def summarize(values: ArrayLike, stat: Stat) -> float:
    """Reduce a per-seed array to a scalar by the named statistic."""
    try:
        reducer = STAT_REDUCERS[stat]
    except KeyError:
        raise ValueError(f"Unknown stat {stat!r}; known: {sorted(STAT_REDUCERS)}") from None
    return reducer(values)


def _resample(k: jax.Array, vals: jax.Array, p: jax.Array) -> jax.Array:
    """One bootstrap resample of the slots with nonzero probability, of the array's full length."""
    return jax.random.choice(k, vals, shape=(vals.shape[0],), replace=True, p=p)


@partial(jax.vmap, in_axes=(0, None, None))
def _boot_mean(k: jax.Array, vals: jax.Array, p: jax.Array) -> jax.Array:
    """The mean of one bootstrap resample."""
    return jnp.mean(_resample(k, vals, p))


@partial(jax.vmap, in_axes=(0, None, None))
def _boot_iqm(k: jax.Array, vals: jax.Array, p: jax.Array) -> jax.Array:
    """The interquartile mean of one bootstrap resample."""
    n = vals.shape[0]
    lo = n // 4
    return jnp.mean(jnp.sort(_resample(k, vals, p))[lo : n - lo])


def masked_iqm(vals: jax.Array, mask: jax.Array) -> jax.Array:
    """Interquartile mean of vals where mask is True, for `iqm_interval`."""
    safe = jnp.where(mask, vals, jnp.inf)
    sorted_vals = jnp.sort(safe)
    count = jnp.sum(mask)
    lo = count // 4
    hi = count - lo
    idx = jnp.arange(vals.shape[0])
    iqm_mask = (idx >= lo) & (idx < hi)
    iqm_count = jnp.maximum(hi - lo, 1)
    return jnp.sum(jnp.where(iqm_mask, sorted_vals, 0.0)) / iqm_count


def _masked_mean(vals: jax.Array, mask: jax.Array) -> jax.Array:
    """Mean of vals where mask is True; NaN when nothing is unmasked."""
    count = jnp.sum(mask)
    total = jnp.sum(jnp.where(mask, vals, 0.0))
    return jnp.where(count > 0, total / jnp.maximum(count, 1), jnp.nan)


def _masked_max(vals: jax.Array, mask: jax.Array) -> jax.Array:
    """Max of vals where mask is True; NaN when nothing is unmasked (never -inf)."""
    count = jnp.sum(mask)
    return jnp.where(count > 0, jnp.max(jnp.where(mask, vals, -jnp.inf)), jnp.nan)


def _masked_min(vals: jax.Array, mask: jax.Array) -> jax.Array:
    """Min of vals where mask is True; NaN when nothing is unmasked (never +inf)."""
    count = jnp.sum(mask)
    return jnp.where(count > 0, jnp.min(jnp.where(mask, vals, jnp.inf)), jnp.nan)


@dataclass(frozen=True, slots=True, kw_only=True)
class CurveStats:
    """Cross-seed point statistics at one step, plus the bootstrap CI of the mean.

    `n` is how many seeds contributed at this step, which is the seed count minus the masked ones.
    It is exported beside the statistics because an interval means nothing without it, and because the closed-form interval for a proportion (`wilson_interval`) is a function of the centre and `n` alone.
    """

    n: int
    mean: float
    median: float
    min: float
    max: float
    mean_ci_lo: float
    mean_ci_hi: float


def _percentile_ci(boots: jax.Array) -> tuple[float, float]:
    """The 2.5/97.5 percentiles of one statistic's bootstrap replicates."""
    return float(jnp.percentile(boots, 2.5)), float(jnp.percentile(boots, 97.5))


def _sorted_draw_inputs(
    values: jax.Array, mask: jax.Array, key: jax.Array, n_bootstrap: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """The replicate keys, and the values and probabilities co-sorted so that a draw depends on the multiset of seeds only."""
    # A fixed key applied to a permuted array resamples different elements, so an unordered input would make the CI depend on the order the caller pooled its seeds in.
    # Co-sorting the values with their probabilities removes that dependence exactly, and masked slots (p = 0) sort to the end, so their position cannot shift the draw either.
    count = jnp.sum(mask)
    p = jnp.where(mask, 1.0 / jnp.maximum(count, 1), 0.0)
    order = jnp.argsort(jnp.where(mask, values, jnp.inf))
    return jax.random.split(key, n_bootstrap), values[order], p[order]


def aggregate_stats(
    values: jax.Array,
    *,
    mask: jax.Array,
    key: jax.Array,
    n_bootstrap: int = 2000,
) -> CurveStats:
    """Cross-seed point stats plus the bootstrap CI of the mean, a function of the multiset of unmasked values, with static shapes throughout."""
    keys, sorted_values, sorted_p = _sorted_draw_inputs(values, mask, key, n_bootstrap)
    mean_ci_lo, mean_ci_hi = _percentile_ci(_boot_mean(keys, sorted_values, sorted_p))
    nan_vals = jnp.where(mask, values, jnp.nan)
    return CurveStats(
        n=int(jnp.sum(mask)),
        mean=float(_masked_mean(values, mask)),
        median=float(jnp.nanmedian(nan_vals)),
        min=float(_masked_min(values, mask)),
        max=float(_masked_max(values, mask)),
        mean_ci_lo=mean_ci_lo,
        mean_ci_hi=mean_ci_hi,
    )


def iqm_interval(
    values: jax.Array,
    *,
    mask: jax.Array,
    key: jax.Array,
    n_bootstrap: int = 2000,
) -> tuple[float, float, float]:
    """The interquartile mean with its 95% bootstrap CI, from the resamples `aggregate_stats` draws for the same key."""
    keys, sorted_values, sorted_p = _sorted_draw_inputs(values, mask, key, n_bootstrap)
    lo, hi = _percentile_ci(_boot_iqm(keys, sorted_values, sorted_p))
    return float(masked_iqm(values, mask)), lo, hi
