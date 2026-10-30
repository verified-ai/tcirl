"""Atomic marker-based counterexample detection for cause DFA inference.

Two phase-exclusive atomic markers x_r and x_a observed in the obs-alphabet:
  - x_r: a proposition that can only fire in the REJECT phase.
  - x_a: a proposition that can only fire in the ACCEPT phase.

A marker in the label at position i of a trace certifies the phase of the prefix of length i, the labels before it.
The certificates are read from the full trace, before any compression, and a compression moves each one to the compressed form of the prefix it certifies.
A trace is a counterexample if the hypothesis disagrees with a certificate.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass

from tcirl.automata.dfa import DFA


@dataclass(frozen=True, slots=True, kw_only=True)
class Certificates:
    """The prefix lengths of one trace that a marker certifies as rejecting (x_r) or as accepting (x_a)."""

    rejecting: frozenset[int]
    accepting: frozenset[int]


def certificates(
    obs_labels: Sequence[int], x_r_obs_ids: frozenset[int], x_a_obs_ids: frozenset[int]
) -> Certificates:
    """The certificates of a full trace: a marker at position i certifies the prefix of length i."""
    return Certificates(
        rejecting=frozenset(i for i, label in enumerate(obs_labels) if label in x_r_obs_ids),
        accepting=frozenset(i for i, label in enumerate(obs_labels) if label in x_a_obs_ids),
    )


def compress_certificates(certs: Certificates, keep: Sequence[int]) -> Certificates:
    """The certificates of a trace compressed to the positions `keep`, each moved to the compressed form of the prefix it certifies, which holds the kept positions before it."""
    # Sound for the compression of `tcirl.learning.trace.filter_symbol_from_trace`: a removed position carries a symbol that leaves the cause in its state, so a prefix and its compressed form share their phase.
    kept = sorted(keep)
    return Certificates(
        rejecting=frozenset(bisect_left(kept, i) for i in certs.rejecting),
        accepting=frozenset(bisect_left(kept, i) for i in certs.accepting),
    )


def is_counterexample(grid_labels: tuple[int, ...], certs: Certificates, hypothesis: DFA) -> bool:
    """Whether the hypothesis classifies a certified prefix of the trace against its certificate."""
    # False positive: the hypothesis accepts a prefix that an x_r certifies as rejecting.
    # False negative: it rejects a prefix that an x_a certifies as accepting.
    # A certificate may sit at every length from 0 to the full trace, which is where a marker after the last kept position lands.
    # Every hypothesis is successor-closed, so an acceptance before an x_r shows up as an acceptance at the x_r itself.
    q = 0  # DFA initial state
    for i in range(len(grid_labels) + 1):
        is_accepting = q in hypothesis.accepting_states
        if is_accepting and i in certs.rejecting:
            return True
        if not is_accepting and i in certs.accepting:
            return True
        if i < len(grid_labels):
            q = hypothesis.transitions[(q, grid_labels[i])]
    return False
