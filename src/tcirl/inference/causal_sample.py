"""Causal sample: the prefix tree of the stored counterexamples, annotated with the paper's sets P(X) and N(X).

A node is in positive_prefixes when an x_a marker certified that the cause is
active at that prefix, and in negative_prefixes when an x_r marker certified
that it is inactive. Only the prefix directly in front of a marker is
annotated, because the SAT encoding's successor-closed acceptance constraint
already propagates a certificate to the prefixes and extensions of the certified
word.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from tcirl.inference.markers import Certificates


class MutableCausalSample:
    """Mutable prefix tree with causal annotations for incremental building."""

    def __init__(self, num_symbols: int):
        self._num_symbols = num_symbols
        self._num_traces = 0

        # Initialize with epsilon (root node)
        self._prefix_to_id: dict[tuple[int, ...], int] = {(): 0}
        self._parents: list[int] = [-1]
        self._children: list[list[tuple[int, int]]] = [[]]
        self._lengths: list[int] = [0]

        # Causal annotations on prefix-tree node IDs
        self._positive_prefixes: set[int] = set()
        self._negative_prefixes: set[int] = set()

    @property
    def num_traces(self) -> int:
        return self._num_traces

    def _ensure_prefix(self, prefix: tuple[int, ...]) -> int:
        """Ensure prefix exists in tree, return its ID."""
        if prefix in self._prefix_to_id:
            return self._prefix_to_id[prefix]
        if not prefix:
            raise ValueError("Empty prefix not found in tree")

        parent_id = self._ensure_prefix(prefix[:-1]) if len(prefix) > 1 else 0

        new_id = len(self._prefix_to_id)
        self._prefix_to_id[prefix] = new_id
        self._parents.append(parent_id)
        self._children.append([])
        self._children[parent_id].append((prefix[-1], new_id))  # pyright: ignore[reportGeneralTypeIssues]
        self._lengths.append(len(prefix))
        return new_id

    def _annotate(self, prefix: tuple[int, ...], target: set[int], opposite: set[int]) -> None:
        """Add one prefix to P(X) or to N(X), rejecting a sample that certifies it both ways."""
        node_id = self._ensure_prefix(prefix)
        if node_id in opposite:
            raise ValueError(
                f"Conflict at prefix length {len(prefix)}: marked as both positive and negative."
            )
        target.add(node_id)

    def add_trace(self, grid_labels: tuple[int, ...], certs: Certificates) -> None:
        """Insert a trace into the prefix tree and annotate each prefix that a certificate names, into N(X) if rejecting and P(X) if accepting."""
        n = len(grid_labels)
        lengths = certs.rejecting | certs.accepting
        if lengths and max(lengths) > n:
            raise ValueError(f"A certificate names a prefix of length {max(lengths)} > {n}")

        # Marker soundness: acceptance is absorbing, so no rejecting certificate may follow an accepting one, and one prefix cannot be certified both ways.
        if certs.rejecting and certs.accepting and max(certs.rejecting) >= min(certs.accepting):
            raise ValueError(
                f"Unsound markers: the prefix of length {min(certs.accepting)} is certified as "
                f"accepting, but the prefix of length {max(certs.rejecting)} as rejecting."
            )

        self._num_traces += 1
        for i in sorted(certs.rejecting):
            self._annotate(grid_labels[:i], self._negative_prefixes, self._positive_prefixes)
        for i in sorted(certs.accepting):
            self._annotate(grid_labels[:i], self._positive_prefixes, self._negative_prefixes)

        # Also ensure the full-length prefix node exists in the tree.
        self._ensure_prefix(grid_labels)

    def to_causal_sample(self) -> CausalSample:
        """Convert to immutable CausalSample."""
        return CausalSample(
            num_symbols=self._num_symbols,
            num_prefixes=len(self._prefix_to_id),
            parent=tuple(self._parents),
            children=tuple(tuple(edges) for edges in self._children),
            length=tuple(self._lengths),
            positive_prefixes=frozenset(self._positive_prefixes),
            negative_prefixes=frozenset(self._negative_prefixes),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class CausalSample:
    """Immutable prefix tree with causal annotations for SAT solving."""

    num_symbols: int
    num_prefixes: int
    parent: tuple[int, ...]
    children: tuple[tuple[tuple[int, int], ...], ...]
    length: tuple[int, ...]
    positive_prefixes: frozenset[int]
    negative_prefixes: frozenset[int]

    @property
    def alphabet(self) -> range:
        return range(self.num_symbols)

    @property
    def prefixes(self) -> range:
        return range(self.num_prefixes)

    def nonempty(self) -> Iterator[tuple[int, int, int]]:
        """Iterate over non-empty prefixes as (parent, symbol, child) triples."""
        for u, edges in enumerate(self.children):
            for sym, v in edges:
                yield u, sym, v
