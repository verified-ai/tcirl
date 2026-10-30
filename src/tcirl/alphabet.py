"""Alphabet types for DFA symbol encoding.

Alphabet[L]: base protocol parametrized by label type L, with encode(L)->int and decode(int)->L.
BitmaskAlphabet: powerset encoding (2^n symbols) for formula progression and DFA construction. Satisfies Alphabet[Collection[P]].
OrderedAlphabet: compact sequential encoding (N symbols) for automata and JAX tables. Satisfies Alphabet[Collection[P]].
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Protocol

from tcirl.formulas.bool_interner import BoolInterner
from tcirl.formulas.bool_syntax import BoolSyntax
from tcirl.formulas.ltlf_syntax import LTLfSyntax, collect_appears


def format_label_set[P](label: Collection[P]) -> str:
    """Render an AP set for display: empty -> {}, singleton -> bare AP, else {a, b}. No quotes."""
    parts = sorted(str(p) for p in label)
    if not parts:
        return "{}"
    if len(parts) == 1:
        return parts[0]
    return "{" + ", ".join(parts) + "}"


class Alphabet[L](Protocol):
    """Base protocol for any DFA alphabet, parametrized by label type L."""

    @property
    def num_symbols(self) -> int: ...
    @property
    def valid_symbols(self) -> frozenset[int]: ...
    def encode(self, label: L) -> int: ...
    def decode(self, symbol: int) -> L: ...
    def format_label(self, label: L) -> str: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class BitmaskAlphabet[P]:
    """Bidirectional mapping between propositions and integer bitmasks.

    For n propositions, the alphabet has 2^n symbols (the powerset).
    Each label (set of true props) maps to an int via bitmask encoding:
    - Empty set encodes to 0
    - {prop_i} encodes to (1 << i) where i is the prop's sorted index

    Props are sorted by str() for deterministic ordering.

    If a label_filter is provided, only symbols satisfying the filter are valid.
    This enables partial DFAs where transitions are only defined for valid symbols.
    """

    props: tuple[P, ...]
    prop_to_bit: dict[P, int]
    valid_symbols: frozenset[int]

    @staticmethod
    def from_syntax[Q](
        formula: LTLfSyntax[Q],
        label_filter: BoolSyntax[Q] | None = None,
        extra_props: Collection[Q] | None = None,
    ) -> BitmaskAlphabet[Q]:
        """Create alphabet from propositions appearing in a formula."""
        all_props = collect_appears(formula)
        if extra_props is not None:
            all_props = all_props | frozenset(extra_props)
        props = tuple(sorted(all_props, key=str))
        if len(props) != len(set(props)):
            raise ValueError("Duplicate propositions")

        prop_to_bit = {p: 1 << i for i, p in enumerate(props)}
        num_symbols = 1 << len(props)

        if label_filter is not None:
            prop_to_index = {p: i for i, p in enumerate(props)}
            bool_interner = BoolInterner(prop_to_index)
            compiled_filter = bool_interner.compile(label_filter)
            valid_symbols = frozenset(
                s
                for s in range(num_symbols)
                if bool_interner.check(
                    compiled_filter,
                    frozenset(i for i in range(len(props)) if s & (1 << i)),
                )
            )
        else:
            valid_symbols = frozenset(range(num_symbols))

        return BitmaskAlphabet(props=props, prop_to_bit=prop_to_bit, valid_symbols=valid_symbols)

    @property
    def prop_to_index(self) -> dict[P, int]:
        """Map from prop to its bit index (0-based)."""
        return {p: i for i, p in enumerate(self.props)}

    @property
    def num_props(self) -> int:
        return len(self.props)

    @property
    def num_symbols(self) -> int:
        """Total alphabet size (2^num_props)."""
        return 1 << len(self.props)

    @property
    def num_valid_symbols(self) -> int:
        return len(self.valid_symbols)

    @property
    def is_filtered(self) -> bool:
        return len(self.valid_symbols) < self.num_symbols

    def encode(self, label: Collection[P]) -> int:
        """Encode a label (set of true props) to a bitmask. O(|label|)."""
        result = 0
        for prop in label:
            if prop in self.prop_to_bit:
                result |= self.prop_to_bit[prop]
        return result

    def encode_single(self, prop: P) -> int:
        """Encode a single proposition to its bitmask. O(1)."""
        return self.prop_to_bit[prop]

    def decode(self, symbol: int) -> frozenset[P]:
        """Decode a bitmask to a label (set of true props). O(num_props)."""
        return frozenset(p for i, p in enumerate(self.props) if symbol & (1 << i))

    def format_label(self, label: Collection[P]) -> str:
        return format_label_set(label)

    def decode_to_ints(self, symbol: int) -> frozenset[int]:
        """Decode to frozenset of prop indices. O(num_props)."""
        return frozenset(i for i in range(len(self.props)) if symbol & (1 << i))

    def is_valid_symbol(self, symbol: int) -> bool:
        """Check if symbol satisfies the label filter. O(1)."""
        return symbol in self.valid_symbols


@dataclass(frozen=True, slots=True, kw_only=True)
class OrderedAlphabet[P]:
    """Compact sequential encoding: symbol IDs 0..N-1 for N valid labels.

    Used for automata and JAX tables, where compact, contiguous symbol IDs are needed.
    """

    props: tuple[P, ...]
    labels: tuple[frozenset[P], ...]
    _label_to_id: dict[frozenset[P], int]

    def __post_init__(self) -> None:
        expected = {label: i for i, label in enumerate(self.labels)}
        if self._label_to_id != expected:
            raise ValueError("_label_to_id is inconsistent with labels; build via create()")

    @classmethod
    def create(
        cls, *, props: tuple[P, ...], labels: tuple[frozenset[P], ...]
    ) -> OrderedAlphabet[P]:
        """Build the alphabet with the id table derived from `labels` order."""
        return cls(
            props=props,
            labels=labels,
            _label_to_id={label: i for i, label in enumerate(labels)},
        )

    @classmethod
    def from_singleton_props(cls, props: tuple[P, ...]) -> OrderedAlphabet[P]:
        """The empty label at id 0, then one singleton label per prop in order."""
        labels = (frozenset[P](), *(frozenset({p}) for p in props))
        return cls.create(props=props, labels=labels)

    @property
    def num_symbols(self) -> int:
        return len(self.labels)

    @property
    def valid_symbols(self) -> frozenset[int]:
        return frozenset(range(len(self.labels)))

    def encode(self, label: Collection[P]) -> int:
        return self._label_to_id[frozenset(label)]

    def decode(self, symbol: int) -> frozenset[P]:
        return self.labels[symbol]

    def format_label(self, label: Collection[P]) -> str:
        return format_label_set(label)
