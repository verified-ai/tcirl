"""Lightweight JAX-friendly DFA for hypothesis tracking.

Only contains delta (transition table) and accepting mask. No reward
function or terminal state tracking -- those are handled by the environment.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from tcirl.automata.dfa import DFA


@dataclass(frozen=True, slots=True, kw_only=True)
class PlainDFA:
    """Padded DFA represented as JAX arrays."""

    delta: jax.Array  # (max_states, num_labels) int32
    accepting: jax.Array  # (max_states,) bool
    num_states: jax.Array  # int32 scalar
    num_labels: int

    @classmethod
    def from_dfa(cls, dfa: DFA, max_states: int) -> PlainDFA:
        """Build a PlainDFA from a `tcirl.automata.dfa.DFA`, padding with self-loop filler states."""
        if dfa.num_states > max_states:
            raise ValueError(f"DFA has {dfa.num_states} states, exceeding max_states={max_states}")

        table = np.zeros((max_states, dfa.num_symbols), dtype=np.int32)
        for (state, symbol), next_state in dfa.transitions.items():
            table[state, symbol] = next_state
        # Filler states self-loop
        for s in range(dfa.num_states, max_states):
            table[s, :] = s
        delta = jnp.array(table, dtype=jnp.int32)

        accepting_np = np.array(
            [s in dfa.accepting_states for s in range(max_states)],
        )
        accepting = jnp.array(accepting_np, dtype=jnp.bool_)

        return cls(
            delta=delta,
            accepting=accepting,
            num_states=jnp.int32(dfa.num_states),
            num_labels=dfa.num_symbols,
        )


jax.tree_util.register_dataclass(
    PlainDFA,
    data_fields=["delta", "accepting", "num_states"],
    meta_fields=["num_labels"],
)
