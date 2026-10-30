"""RewardAutomaton protocol and shared stepping logic.

A reward automaton is a deterministic transition system over labels that emits a reward at every transition and signals episode termination via a per-`(configuration, label)` flag. Termination is the absence of a continuing run: either the source automaton has no defined transition on this symbol, or the target is an accepting configuration from which no further transitions are taken.

The automaton state is a configuration: a fixed-arity tuple of indices over the product space `config_shape`. A DFA configuration is `(q,)` (arity 1). The transition arrays are indexed by the configuration's axes; `delta[(*config, label)]` returns the next configuration directly (an `(m,)` index vector), so the same machinery serves any arity.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax


@runtime_checkable
class RewardAutomaton(Protocol):
    """Structural protocol that `DFAReward` satisfies."""

    @property
    def config_shape(self) -> tuple[int, ...]: ...  # product-space axis sizes; arity m = len

    @property
    def delta(self) -> jax.Array: ...  # (*config_shape, num_labels, m) int32, next configuration

    @property
    def sigma(self) -> jax.Array: ...  # (*config_shape, num_labels) float32

    @property
    def terminal(self) -> jax.Array: ...  # (*config_shape, num_labels) bool, per-transition

    # (*config_shape, num_labels) bool: True iff the machine has NO RUN on this
    # transition (a structurally-undefined transition from a live configuration).
    # A strict subset of `terminal` (`crash => terminal`, and `crash => sigma == 0`):
    # a crash is a terminal with no landing configuration, distinct from a
    # reject-terminal, which HAS a defined transition into a dead configuration. A
    # `DFAReward` is all-False for the unrestricted alphabets the RL stack ships and
    # non-False only for a `valid_symbols`-restricted DFA (out-of-scope symbols at a
    # live state).
    @property
    def crash(self) -> jax.Array: ...

    @property
    def num_states(self) -> jax.Array: ...  # int32 scalar, live configuration count

    @property
    def num_labels(self) -> int: ...


def aut_step(
    aut: RewardAutomaton,
    config: jax.Array,
    label: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Step the reward automaton. Returns (next_config, reward, terminated, crashed).

    `config` is the `(m,)` configuration index vector; `next_config` is the `(m,)` next configuration read straight out of `delta`. `terminated` reads `aut.terminal[(*config, label)]` at the source configuration + label, capturing "this transition ends the episode" whether the cause is absence of a run or entry to an accepting configuration. `crashed` reads `aut.crash[(*config, label)]`: the run does not exist on this transition (a subset of `terminated`).
    """
    m = len(aut.config_shape)
    idx = (*(config[i] for i in range(m)), label)
    return aut.delta[idx], aut.sigma[idx], aut.terminal[idx], aut.crash[idx]
