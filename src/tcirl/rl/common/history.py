"""What the deep baseline sees: a window of the last `k` observation FRAMES.

One frame is one observation: the observable index vector, then the label of the transition that
produced that state. The window holds the last `k` frames, oldest first, and its newest slot is
always the current observation, so `history_len=1` is exactly the memoryless agent. A frame is what
the paper's "observation history buffer of k frames" names.

The phase is already in one label, because both domains carry phase-exclusive markers:
`dec_recovery` fires only after the gene cause has fired and `inc` only before it, and in traffic
each marker certifies the phase that owns its bursty draw. What a stack of frames adds is the recent
history of LEVELS, which separates observations that a window of labels confounds, and a window of
frames contains a window of labels at the same `k`.

No window can hold either cause: the gene cause completes in 8 moves from the start
(`_ a _ b _ a _ d`), so its span is 7 labels, and the traffic cause needs 6 action labels; a window
of exactly that length would see the cause only at the instant it completes, because acceptance is
absorbing. And no window remembers that an absorbing acceptance already happened.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tcirl.rl.common.tabular import EnvState, TabularEnv


def frame_dim(state_shape: tuple[int, ...]) -> int:
    """How many index entries one frame holds: one per observable component, plus the label."""
    return len(state_shape) + 1


def frame_shape(state_shape: tuple[int, ...], num_labels: int) -> tuple[int, ...]:
    """The index shape of one frame: one entry per observable component, then the label."""
    if num_labels < 1:
        raise ValueError(f"num_labels must be >= 1, got {num_labels}")
    return (*state_shape, num_labels)


def observation_frame(env: TabularEnv, state: EnvState) -> jax.Array:
    """One frame, `(len(state_shape) + 1,)` int32: the observable indices, then the label.

    `EnvState.indices` puts the observable components first and hidden bookkeeping (the cause DFA
    state, the last observation label) last; `state_shape` has one axis per observable component,
    so its length is the prefix to take. That prefix is exactly the tuple the tabular agent's
    Q-table is indexed by, which is what makes the deep and tabular cause-blind arms see the same
    state. The label is the symbol the transition INTO this state emitted, which at an episode's
    first step is the reset label.
    """
    n_obs = frame_dim(env.state_shape) - 1
    parts = [state.indices[i] for i in range(n_obs)]
    parts.append(state.label)
    return jnp.stack(parts).astype(jnp.int32)


def init_window(frame: jax.Array, history_len: int) -> jax.Array:
    """A fresh window, `(history_len, frame_dim)` int32, with every slot holding `frame`.

    An episode's first step has no past, and repeating the current frame is the one filler that
    says "nothing has changed yet": every difference the window can form is zero.
    """
    if history_len < 1:
        raise ValueError(f"history_len must be >= 1, got {history_len}")
    return jnp.tile(frame[None, :], (history_len, 1)).astype(jnp.int32)


def push_window(window: jax.Array, frame: jax.Array) -> jax.Array:
    """The window after one transition: drop the oldest slot and append `frame` as the newest.

    The newest slot is the LAST one, so `window[-1]` is always the current observation and
    `window[0]` the oldest of the `history_len` frames kept.
    """
    return jnp.concatenate([window[1:], frame[None, :].astype(jnp.int32)], axis=0)


def encode_window(window: jax.Array, *, frame_shape: tuple[int, ...]) -> jax.Array:
    """The network input: one-hot per frame component, frame by frame, oldest first.

    `(history_len * sum(frame_shape),)` float32. The layout is fixed and positional, so a replay
    row stores the index window and the encoding is rebuilt from it -- storing the encoded vector
    instead costs 12.8 times the memory per row in the gene domain at k = 4 and 6.3 times in
    traffic.
    """
    per_component = [
        jax.nn.one_hot(window[:, i], size, dtype=jnp.float32) for i, size in enumerate(frame_shape)
    ]
    return jnp.concatenate(per_component, axis=1).reshape(-1)


def observation_dim(frame_shape: tuple[int, ...], history_len: int) -> int:
    """The width `encode_window` produces for this frame shape and window length."""
    if history_len < 1:
        raise ValueError(f"history_len must be >= 1, got {history_len}")
    return history_len * sum(frame_shape)
