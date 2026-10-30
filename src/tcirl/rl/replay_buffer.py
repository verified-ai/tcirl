"""Generic JAX-resident circular replay buffer.

Works with any pytree transition type and is fully JIT-compatible.
Uses uniform sampling without replacement (top-k over gumbel noise).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class ReplayBufferState[T](NamedTuple):
    """State of a circular replay buffer."""

    data: T  # pytree of arrays, each with leading dim = max_size
    write_idx: jax.Array  # int32 scalar
    size: jax.Array  # int32 scalar (current fill level, up to max_size)


def buffer_init[T](transition_template: T, max_size: int) -> ReplayBufferState[T]:
    """Initialize an empty buffer from a template transition.

    The template must have the correct shapes and dtypes for each field.
    The leading dimension is prepended with max_size.
    """

    def batched_zeros(x: jax.Array) -> jax.Array:
        return jnp.zeros((max_size, *x.shape), dtype=x.dtype)

    data: T = jax.tree.map(batched_zeros, transition_template)
    return ReplayBufferState(data=data, write_idx=jnp.int32(0), size=jnp.int32(0))


def buffer_add[T](
    state: ReplayBufferState[T], transition: T, max_size: int
) -> ReplayBufferState[T]:
    """Add a single transition to the buffer (overwrites oldest when full)."""

    def write(buf: jax.Array, item: jax.Array) -> jax.Array:
        return buf.at[state.write_idx].set(item)

    new_data: T = jax.tree.map(write, state.data, transition)
    new_write_idx = (state.write_idx + 1) % max_size
    new_size = jnp.minimum(state.size + 1, max_size)
    return ReplayBufferState(data=new_data, write_idx=new_write_idx, size=new_size)


def buffer_sample[T](state: ReplayBufferState[T], key: jax.Array, batch_size: int) -> T:
    """Sample a batch of transitions uniformly without replacement.

    Callers are expected to ensure state.size >= batch_size before sampling.
    """
    first_leaf = jax.tree.leaves(state.data)[0]
    max_size = first_leaf.shape[0]
    valid_mask = jnp.arange(max_size, dtype=jnp.int32) < state.size
    noise = jax.random.gumbel(key, (max_size,))
    scores = jnp.where(valid_mask, noise, -jnp.inf)
    _, indices = jax.lax.top_k(scores, batch_size)

    def gather(buf: jax.Array) -> jax.Array:
        return buf[indices]

    return jax.tree.map(gather, state.data)
