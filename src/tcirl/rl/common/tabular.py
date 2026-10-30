"""Shared types for tabular environments."""

from __future__ import annotations

from typing import NamedTuple, Protocol

import jax


class EnvState(NamedTuple):
    """Universal state for tabular environments: Q-table indices + step counter + label."""

    indices: tuple[jax.Array, ...]
    step_count: jax.Array
    label: jax.Array


class RendersMap(Protocol):
    """Anything that can render its static map (the logger's env-render hook)."""

    def render_map(self) -> str: ...


class TabularEnv(Protocol):
    """Protocol for finite-state environments compatible with tabular Q-learning."""

    @property
    def name(self) -> str: ...

    @property
    def state_shape(self) -> tuple[int, ...]: ...

    @property
    def num_actions(self) -> int: ...

    @property
    def is_time_indexed(self) -> bool:
        """True iff the state carries a time index as its last base axis.

        Neither environment of this repository is time-indexed; wrappers delegate to their base.
        """
        ...

    def reset(self, key: jax.Array) -> EnvState: ...

    def step(
        self, state: EnvState, action: jax.Array, key: jax.Array
    ) -> tuple[EnvState, jax.Array, jax.Array, jax.Array, jax.Array]:
        """Returns (next_state, reward, terminated, aut_terminated, aut_crashed).

        `terminated` is the episode-boundary bit (base/horizon/reward-machine, OR'd). `aut_terminated` and `aut_crashed` are the automaton's termination flags; the environments of both case studies return `False` for both.
        """
        ...

    def action_mask(self, state: EnvState) -> jax.Array: ...

    def render(self, state: EnvState) -> str: ...

    def render_map(self) -> str:
        """Render the static environment (the 'map') with no live agent/state drawn."""
        ...

    def get_size_hint(self) -> int | None:
        """The ground-truth automaton's control-state count, if the env emits one.

        `None` means not emitted: the agent falls back to its configured size bound. When an int is returned, an agent that has a size bound must have been configured with exactly this value. Agents that read the true automaton directly, such as Q-learning on the product env, ignore it. Non-automaton envs return `None`.
        """
        ...
