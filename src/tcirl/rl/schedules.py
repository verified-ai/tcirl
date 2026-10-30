"""Shared schedule and step-size helpers for RL code."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def linear_schedule_value(step: float, start: float, end: float, decay_steps: int) -> float:
    fraction = min(max(step / decay_steps, 0.0), 1.0)  # clip to [0, 1] to match the JAX variant
    return start + fraction * (end - start)


def linear_schedule_value_jax(
    step: jax.Array, start: float, end: float, decay_steps: int
) -> jax.Array:
    step_value = jnp.asarray(step)
    fraction = jnp.clip(step_value / decay_steps, 0.0, 1.0)
    start_value = jnp.asarray(start, dtype=fraction.dtype)
    end_value = jnp.asarray(end, dtype=fraction.dtype)
    return start_value + fraction * (end_value - start_value)


def rescaled_linear_step_size(update_count: jax.Array, scale: float) -> jax.Array:
    """The rescaled linear step size `c / (c + n - 1)` of an entry's `n`-th update, with the scale `c`.

    The first update replaces the initial value, and the rate then falls like `c / n`, which satisfies the Robbins-Monro conditions per entry.
    Wainwright (2019) uses `c = 1 / (1 - gamma)`, the smallest scale at which his bound removes any initial error at the rate `1 / n`.
    """
    n = jnp.asarray(update_count).astype(jnp.float32)
    return 1.0 / (1.0 + (n - 1.0) / jnp.float32(scale))
