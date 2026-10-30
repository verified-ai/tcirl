"""DFA state encoding for neural network observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from tcirl.formulas.ltlf_interned import LTLfTag
from tcirl.formulas.ltlf_timers import TimerRegistry, find_timer_bounds

if TYPE_CHECKING:
    from collections.abc import Hashable

    from numpy.typing import NDArray

    from tcirl.formulas.ltlf_interned import LTLfInterned


def compute_signature(formula: LTLfInterned) -> tuple[Hashable, ...]:
    """Compute hashable signature for finite control (timer-independent structure).

    For bounded operators, uses the timer_name instead of bound value to create
    signatures that group states with the same structure but different countdowns.
    """
    match formula.tag:
        case LTLfTag.TOP | LTLfTag.BOTTOM | LTLfTag.ATOM:
            return (formula.key,)

        case LTLfTag.FINALLY_BOUND | LTLfTag.WEAK_FINALLY_BOUND | LTLfTag.GLOBALLY_BOUND:
            return ("TIMER", formula.timer_name, compute_signature(formula.inner))

        case LTLfTag.UNTIL_BOUND | LTLfTag.WEAK_UNTIL_BOUND:
            return (
                "TIMER",
                formula.timer_name,
                compute_signature(formula.left),
                compute_signature(formula.right),
            )

        case LTLfTag.NEXT | LTLfTag.NOT | LTLfTag.GLOBALLY | LTLfTag.FINALLY:
            return (formula.tag, compute_signature(formula.inner))

        case LTLfTag.AND | LTLfTag.OR:
            left_sig = compute_signature(formula.left)
            right_sig = compute_signature(formula.right)
            children = tuple(sorted([left_sig, right_sig], key=str))
            return (formula.tag, *children)

        case LTLfTag.IMPLIES | LTLfTag.UNTIL | LTLfTag.RELEASE:
            return (
                formula.tag,
                compute_signature(formula.left),
                compute_signature(formula.right),
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class TimerGroupSchema:
    """Schema for one timer group, derived from the TimerRegistry."""

    name: str
    original_lower: int
    original_upper: int

    @property
    def encoding_dim(self) -> int:
        """Number of encoding dimensions for this timer."""
        if self.original_upper == 0:
            return 0  # No countdown, entirely finite control
        if self.original_lower == 0:
            return 1  # Only upper bound dimension
        return 2  # Both lower and upper bound dimensions


@dataclass(frozen=True, slots=True, kw_only=True)
class EncodingSchema:
    """Schema describing the encoding layout."""

    timer_groups: tuple[TimerGroupSchema, ...]
    signature_to_index: dict[tuple[Hashable, ...], int]

    @property
    def timer_dim(self) -> int:
        """Total timer encoding dimension."""
        return sum(g.encoding_dim for g in self.timer_groups)

    @property
    def finite_control_dim(self) -> int:
        return len(self.signature_to_index)

    @property
    def total_dim(self) -> int:
        return self.timer_dim + self.finite_control_dim


class DFAStateEncoder:
    """Encodes DFA states (formulas) to float vectors for neural network observations.

    Encoding layout: [<timer_1 values> ... <timer_N values> <finite_control_one_hot>]

    Each timer has encoding_dim values: 0 if original_upper==0, 1 (upper/original_upper)
    if original_lower==0, or 2 (lower/original_upper, upper/original_upper) if
    original_lower>0. When a timer is inactive (not counting down), it encodes as 1.0.

    The finite control part is a one-hot vector over structural signatures.
    """

    def __init__(
        self,
        signature_to_index: dict[tuple[Hashable, ...], int],
        timer_groups: tuple[TimerGroupSchema, ...],
        encoding_table: NDArray[np.float32],
    ) -> None:
        self._signature_to_index = signature_to_index
        self._timer_groups = timer_groups
        self._encoding_table = encoding_table

    @staticmethod
    def build(
        state_to_formula: dict[int, LTLfInterned],
        timer_registry: TimerRegistry,
    ) -> DFAStateEncoder:
        """Build encoder from DFA state formulas and the specification's timer registry."""
        # Timer schema is a global property of the specification, not per-state
        timer_groups_list: list[TimerGroupSchema] = []
        for name in sorted(timer_registry.names):
            desc = timer_registry.get(name)
            assert desc is not None
            timer_groups_list.append(
                TimerGroupSchema(
                    name=desc.name,
                    original_lower=desc.original_lower,
                    original_upper=desc.original_upper,
                )
            )
        timer_groups = tuple(timer_groups_list)

        # Collect structural signatures from all DFA states
        signature_to_index: dict[tuple[Hashable, ...], int] = {}
        for formula in state_to_formula.values():
            sig = compute_signature(formula)
            if sig not in signature_to_index:
                signature_to_index[sig] = len(signature_to_index)

        # Build encoding table
        timer_dim = sum(g.encoding_dim for g in timer_groups)
        total_dim = timer_dim + len(signature_to_index)
        max_state = max(state_to_formula.keys())
        table = np.zeros((max_state + 1, total_dim), dtype=np.float32)

        # Precompute group offsets
        group_offsets: list[int] = []
        offset = 0
        for g in timer_groups:
            group_offsets.append(offset)
            offset += g.encoding_dim

        for state_id, formula in state_to_formula.items():
            encoding = table[state_id]

            # Per-state timer values: find_timer_bounds keeps the tightest (min upper)
            bounds = find_timer_bounds(formula)

            for gi, group in enumerate(timer_groups):
                base = group_offsets[gi]
                dim = group.encoding_dim
                if dim == 0:
                    continue
                bound = bounds.get(group.name)
                if bound is None:
                    # Timer not active in this state
                    for j in range(dim):
                        encoding[base + j] = 1.0
                elif dim == 1:
                    encoding[base] = bound[1] / group.original_upper
                else:
                    encoding[base] = bound[0] / group.original_upper
                    encoding[base + 1] = bound[1] / group.original_upper

            # Encode finite control (one-hot)
            sig = compute_signature(formula)
            encoding[timer_dim + signature_to_index[sig]] = 1.0

        return DFAStateEncoder(signature_to_index, timer_groups, table)

    @property
    def encoding_dim(self) -> int:
        return self.timer_dim + self.finite_control_dim

    @property
    def timer_dim(self) -> int:
        """Timer encoding dimension."""
        return sum(g.encoding_dim for g in self._timer_groups)

    @property
    def finite_control_dim(self) -> int:
        return len(self._signature_to_index)

    @property
    def schema(self) -> EncodingSchema:
        return EncodingSchema(
            timer_groups=self._timer_groups,
            signature_to_index=self._signature_to_index,
        )

    def encode(self, state_id: int) -> NDArray[np.float32]:
        """Encode a DFA state to a float vector (view into precomputed table)."""
        return self._encoding_table[state_id]

    def encode_batch(self, state_ids: NDArray[np.int64]) -> NDArray[np.float32]:
        """Batch encode multiple DFA states. Returns (len(state_ids), encoding_dim)."""
        return self._encoding_table[state_ids]

    def decode_timer_values(self, encoding: NDArray[np.float32]) -> dict[str, list[float]]:
        """Decode timer values from encoding.

        Returns a dict mapping timer group name to a list of values
        (1 value if encoding_dim==1, 2 values if ==2).
        """
        result: dict[str, list[float]] = {}
        offset = 0
        for group in self._timer_groups:
            dim = group.encoding_dim
            values = [float(encoding[offset + j]) for j in range(dim)]
            result[group.name] = values
            offset += dim
        return result

    def to_jax_array(self) -> jax.Array:
        """Return the full encoding table as a JAX array (num_states, encoding_dim)."""
        return jnp.asarray(self._encoding_table)
