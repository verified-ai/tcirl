"""Config-driven temporal smoothing of per-seed metric curves.

Smoothing is a single source of truth (the run's `smoothing_steps`) applied identically to the
live wandb curves and the exported figures. It operates on the logged tile-means (per seed, at
`log_interval` resolution), NOT on raw per-step signals: for a timescale `tau >= log_interval` the
two agree up to `O(log_interval / tau)`, so no raw per-step data is needed.

It is an EMA with a per-log-point weight `alpha = 1 - exp(-log_interval / tau)`, chosen so the same
`tau` (in env steps) smooths the same physical horizon regardless of `log_interval`. It is applied
**per seed, before** the cross-seed reduction (each seed is the unit of replication), to the mean
categories (`step_mean`/`episode_mean`) and the eval metrics declared smoothable.
NaN tiles (a seed with no episode this tile) carry the previous smoothed value forward.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import jax
import numpy as np


def ema_alpha(smoothing_steps: int | None, log_interval: int) -> float | None:
    """Per-log-point EMA weight for a step-timescale `tau`, or None when smoothing is off."""
    if smoothing_steps is None:
        return None
    return 1.0 - math.exp(-log_interval / smoothing_steps)


def _ema_step(state: np.ndarray, value: np.ndarray, alpha: float) -> np.ndarray:
    """One EMA update, per seed, carrying the previous state through NaN inputs.

    `state`/`value` are `(num_runs,)`. A NaN `value` (seed had no datum this tile) carries `state`;
    the first finite `value` initializes the state; otherwise `alpha*value + (1-alpha)*state`.
    """
    init_or_step = np.where(np.isnan(state), value, alpha * value + (1.0 - alpha) * state)
    return np.where(np.isnan(value), state, init_or_step)


class Smoother:
    """Online per-seed EMA, carried across log points; identity on non-smoothable keys."""

    def __init__(self, *, alpha: float, smoothable_keys: Iterable[str]) -> None:
        self._alpha = alpha
        self._smoothable = frozenset(smoothable_keys)
        self._state: dict[str, np.ndarray] = {}

    def update(self, per_seed: Mapping[str, jax.Array]) -> dict[str, np.ndarray]:
        """Return the smoothed per-seed arrays for this log point (non-smoothable keys verbatim)."""
        out: dict[str, np.ndarray] = {}
        for key, values in per_seed.items():
            arr = np.asarray(jax.device_get(values), dtype=np.float64)
            if key not in self._smoothable:
                out[key] = arr
                continue
            prev = self._state.get(key)
            state = arr.copy() if prev is None else _ema_step(prev, arr, self._alpha)
            self._state[key] = state
            out[key] = state
        return out


def smooth_per_seed_curve(values: list[list[float]], alpha: float) -> list[list[float]]:
    """Offline EMA of a per-seed curve `[log_point][seed]`, matching the live `Smoother`.

    Requires rectangular rows (the full `(num_runs,)` arrays the engine emits). Used by the
    export so figures reproduce the live smoothing from the saved raw curves + the run's config.
    """
    if not values:
        return values
    mat = np.asarray(values, dtype=np.float64)  # (n_points, num_runs)
    state = mat[0].copy()
    out = [state.copy()]
    for row in mat[1:]:
        state = _ema_step(state, row, alpha)
        out.append(state.copy())
    return [row.tolist() for row in out]
