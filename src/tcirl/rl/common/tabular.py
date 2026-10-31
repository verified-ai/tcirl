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

    def reset(self, key: jax.Array) -> EnvState: ...

    def step(
        self, state: EnvState, action: jax.Array, key: jax.Array
    ) -> tuple[EnvState, jax.Array, jax.Array]:
        """Returns (next_state, reward, terminated).

        `terminated` says that the transition ends the episode; the episode limit is a truncation, which the learner applies.
        """
        ...

    def action_mask(self, state: EnvState) -> jax.Array: ...

    def render(self, state: EnvState) -> str: ...

    def render_map(self) -> str:
        """Render the static environment (the 'map') with no live agent/state drawn."""
        ...
