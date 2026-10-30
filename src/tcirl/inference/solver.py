"""Passive learning loop for cause DFA inference: the smallest state count whose SAT instance is satisfiable wins."""

from __future__ import annotations

from tcirl.automata.dfa import DFA
from tcirl.inference.causal_sample import CausalSample
from tcirl.inference.cause_dfa_sat import cause_dfa_sat


def learn_cause_dfa(
    sample: CausalSample,
    *,
    max_states: int,
    initial_rejecting: bool = False,
    symmetry_breaking: bool = True,
    silent: bool = True,
) -> DFA | None:
    """Learn a minimal cause DFA from a causal sample by trying n = 1, ..., max_states; None if every instance is unsatisfiable."""
    if max_states < 1:
        raise ValueError(f"max_states must be >= 1, got {max_states}")
    for n in range(1, max_states + 1):
        if not silent:
            print(f"Trying {n} states...")
        result = cause_dfa_sat(
            n,
            sample,
            initial_rejecting=initial_rejecting,
            symmetry_breaking=symmetry_breaking,
            silent=silent,
        )
        if result is not None:
            if not silent:
                print(f"Found cause DFA with {n} states.")
            return result
    if not silent:
        print(f"No cause DFA found up to {max_states} states.")
    return None
