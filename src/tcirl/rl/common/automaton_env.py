"""Product MDP: base environment x reward automaton."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from tcirl.rl.common.reward_automaton import RewardAutomaton, aut_step
from tcirl.rl.common.tabular import EnvState, TabularEnv


@dataclass(frozen=True, slots=True, kw_only=True)
class AutomatonEnv:
    """Wraps a base TabularEnv and a RewardAutomaton into a product MDP."""

    base: TabularEnv
    aut: RewardAutomaton
    # Optional hint: the true number of automaton control states the agent
    # should size its hypothesis bound to. `None` means not emitted: the agent falls
    # back to its configured bound. When set, an agent must have been configured
    # with exactly this value (otherwise its Q-table would carry permanently-
    # unused padding rows).
    size_hint: int | None

    @property
    def name(self) -> str:
        return self.base.name

    @property
    def state_shape(self) -> tuple[int, ...]:
        # The product-MDP state is the base state followed by the automaton's
        # configuration axes (one per component of `config_shape`).
        return (*self.base.state_shape, *self.aut.config_shape)

    @property
    def base_state_shape(self) -> tuple[int, ...]:
        return self.base.state_shape

    @property
    def num_actions(self) -> int:
        return self.base.num_actions

    @property
    def is_time_indexed(self) -> bool:
        # Config axes append after the base, so base_state_shape's time-indexing
        # (its last axis, when present) is exactly the base's.
        return self.base.is_time_indexed

    @property
    def _n_base(self) -> int:
        return len(self.base.state_shape)

    @property
    def _config_arity(self) -> int:
        return len(self.aut.config_shape)

    def get_aut(self) -> RewardAutomaton:
        return self.aut

    def get_size_hint(self) -> int | None:
        """Number of automaton control states the agent should target, if known.

        Returns `None` when the env does not emit a hint (the agent then uses
        its configured bound). See the `size_hint` field.
        """
        return self.size_hint

    def reset(self, key: jax.Array) -> EnvState:
        base_state = self.base.reset(key)
        zero_config = tuple(jnp.int32(0) for _ in range(self._config_arity))
        # The base's reset label is propagated as-is (the automaton consumes labels
        # only on steps).
        return EnvState(
            indices=(*base_state.indices, *zero_config),
            step_count=jnp.int32(0),
            label=base_state.label,
        )

    def step(
        self, state: EnvState, action: jax.Array, key: jax.Array
    ) -> tuple[EnvState, jax.Array, jax.Array, jax.Array, jax.Array]:
        n_base = self._n_base
        m = self._config_arity
        base_indices = state.indices[:n_base]
        config = jnp.stack(state.indices[n_base:])  # (m,) configuration

        base_state = EnvState(
            indices=base_indices,
            step_count=state.step_count,
            label=state.label,
        )
        new_base_state, _base_reward, base_terminated, _base_aut, _base_crash = self.base.step(
            base_state, action, key
        )

        label = new_base_state.label
        new_config, reward, aut_terminated, aut_crashed = aut_step(self.aut, config, label)

        new_state = EnvState(
            indices=(*new_base_state.indices, *(new_config[i] for i in range(m))),
            step_count=new_base_state.step_count,
            label=label,
        )
        # `aut_terminated` is the reward machine's own termination bit (accept,
        # reject-terminal, or crash) -- the dual of `reward`; `aut_crashed` is its
        # no-run subset. The third return is the episode-boundary bit.
        return new_state, reward, base_terminated | aut_terminated, aut_terminated, aut_crashed

    def action_mask(self, state: EnvState) -> jax.Array:
        base_state = EnvState(
            indices=state.indices[: self._n_base],
            step_count=state.step_count,
            label=state.label,
        )
        return self.base.action_mask(base_state)

    def render(self, state: EnvState) -> str:
        base_state = EnvState(
            indices=state.indices[: self._n_base],
            step_count=state.step_count,
            label=state.label,
        )
        config = tuple(int(c) for c in state.indices[self._n_base :])
        return f"{self.base.render(base_state)}\nautomaton configuration {config}"

    def render_map(self) -> str:
        return self.base.render_map()
