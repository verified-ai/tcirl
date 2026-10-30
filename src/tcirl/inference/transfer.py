"""The rules that choose, at a re-inference, the state of the old hypothesis whose Q-values each state of the new hypothesis takes."""

from __future__ import annotations

from collections import deque
from typing import assert_never

import jax
import jax.numpy as jnp
import numpy as np

from tcirl.automata.dfa import DFA, compute_equivalent_states
from tcirl.config import ValueTransfer
from tcirl.inference.causal_sample import CausalSample


def transfer_sources(
    old: DFA, new: DFA, sample: CausalSample, rule: ValueTransfer
) -> dict[int, int]:
    """Map every state of `new` to the state of `old` whose row it takes, or to nothing under the rule `"none"`."""
    match rule:
        case "none":
            return {}
        case "exact":
            return exact_sources(old, new)
        case "counterexample":
            return counterexample_sources(old, new, sample)
        case _:
            assert_never(rule)


def transfer_rows(
    table: jax.Array, sources: dict[int, int], *, num_states: int, cause_index: int, fill: float
) -> jax.Array:
    """The table of the new hypothesis: on the hypothesis axis `cause_index`, row `q` is row `sources[q]` of `table` below `num_states` and `fill` from there on."""
    hypothesis_axis = (slice(None),) * cause_index
    old_rows = (*hypothesis_axis, jnp.asarray([sources[q] for q in range(num_states)], jnp.int32))
    new_rows = (*hypothesis_axis, jnp.arange(num_states, dtype=jnp.int32))
    return jnp.full_like(table, fill).at[new_rows].set(table[old_rows])


def kept_update_share(counts: jax.Array, sources: dict[int, int], *, cause_index: int) -> float:
    """The share of the update counts in `counts` that lie in the rows of the old states that some new state takes."""
    kept_rows = (
        *((slice(None),) * cause_index),
        jnp.asarray(sorted(set(sources.values())), jnp.int32),
    )
    return float(jnp.sum(counts[kept_rows]) / jnp.sum(counts))


def exact_sources(old: DFA, new: DFA) -> dict[int, int]:
    """Each new state takes an old state that accepts the same label sequences, and otherwise the old state that its shortest label sequence reaches."""
    sources = shortest_sequence_sources(old, new)
    sources.update(compute_equivalent_states(old, new))
    return sources


def shortest_sequence_sources(old: DFA, new: DFA) -> dict[int, int]:
    """Map each new state to the old state that the shortest label sequence leading to it reaches, with ties broken by the smaller symbol first."""
    sources = {0: 0}
    queue = deque([0])
    while queue:
        state = queue.popleft()
        for symbol in range(new.num_symbols):
            successor = new.step(state, symbol)
            if successor not in sources:
                sources[successor] = old.step(sources[state], symbol)
                queue.append(successor)
    if len(sources) != new.num_states:
        raise ValueError(
            f"{new.num_states - len(sources)} states of the new hypothesis are unreachable"
        )
    return sources


def counterexample_sources(old: DFA, new: DFA, sample: CausalSample) -> dict[int, int]:
    """Each new state takes the old state that the most prefixes of the causal sample lead to together with it, with ties broken by the smaller old state."""
    # The new hypothesis is a smallest DFA consistent with `sample`, so a prefix of the sample reaches every one of its states: turning each transition into a state that no prefix reaches into a self-loop, and keeping only the states reachable from the initial state in breadth-first order, gives a smaller DFA that meets every constraint of the SAT encoding.
    counts = shared_prefix_counts(old, new, sample)
    unreached = np.flatnonzero(counts.sum(axis=0) == 0)
    if unreached.size:
        raise ValueError(
            f"No prefix of the causal sample reaches the new states {unreached.tolist()}, "
            "so the new hypothesis is not a smallest DFA consistent with the sample"
        )
    return {state: int(np.argmax(counts[:, state])) for state in range(new.num_states)}


def shared_prefix_counts(old: DFA, new: DFA, sample: CausalSample) -> np.ndarray:
    """`counts[u, v]` is the number of prefixes of `sample`, the empty one included, that lead `old` to `u` and `new` to `v`."""
    counts = np.zeros((old.num_states, new.num_states), dtype=np.int64)
    stack = [(0, 0, 0)]
    while stack:
        node, old_state, new_state = stack.pop()
        counts[old_state, new_state] += 1
        for symbol, child in sample.children[node]:
            stack.append((child, old.step(old_state, symbol), new.step(new_state, symbol)))
    return counts
