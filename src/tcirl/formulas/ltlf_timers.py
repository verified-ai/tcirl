"""Timer tracking for bounded LTLf operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from tcirl.formulas.ltlf_interned import LTLfInterned, LTLfTag


@dataclass(frozen=True, slots=True, kw_only=True)
class TimerDescriptor:
    """Describes a named timer (bounded operator)."""

    name: str
    original_lower: int
    original_upper: int
    kind: Literal[
        "FinallyBound", "GloballyBound", "UntilBound", "WeakFinallyBound", "WeakUntilBound"
    ]


class TimerRegistry:
    """Tracks all named timers (bounded operators) in a formula.

    Each bounded operator must have a unique name. The registry stores the
    original bound for normalization purposes.
    """

    def __init__(self) -> None:
        self._by_name: dict[str, TimerDescriptor] = {}

    def register(
        self,
        name: str,
        lower: int,
        upper: int,
        kind: Literal[
            "FinallyBound",
            "GloballyBound",
            "UntilBound",
            "WeakFinallyBound",
            "WeakUntilBound",
        ],
    ) -> None:
        """Register a timer. Raises if name already registered."""
        if name in self._by_name:
            raise ValueError(f"Timer name '{name}' already registered")

        self._by_name[name] = TimerDescriptor(
            name=name,
            original_lower=lower,
            original_upper=upper,
            kind=kind,
        )

    def get(self, name: str) -> TimerDescriptor | None:
        return self._by_name.get(name)

    @property
    def num_timers(self) -> int:
        return len(self._by_name)

    @property
    def names(self) -> list[str]:
        return list(self._by_name)


def find_timer_bounds(formula: LTLfInterned) -> dict[str, tuple[int, int]]:
    """Extract current bounds for all named bounded operators in a formula.

    Returns a dict mapping timer_name -> (lower, upper). For timers that appear
    multiple times, keeps the minimum upper bound (tightest constraint).
    """
    result: dict[str, tuple[int, int]] = {}
    _collect_timer_bounds(formula, result)
    return result


def _collect_timer_bounds(
    formula: LTLfInterned,
    result: dict[str, tuple[int, int]],
) -> None:
    """Recursively collect bounds from bounded operators by name."""
    match formula.tag:
        case LTLfTag.FINALLY_BOUND | LTLfTag.GLOBALLY_BOUND | LTLfTag.WEAK_FINALLY_BOUND:
            name = formula.timer_name
            if name not in result or formula.upper < result[name][1]:
                result[name] = (formula.lower, formula.upper)
            _collect_timer_bounds(formula.inner, result)

        case LTLfTag.UNTIL_BOUND | LTLfTag.WEAK_UNTIL_BOUND:
            name = formula.timer_name
            if name not in result or formula.upper < result[name][1]:
                result[name] = (formula.lower, formula.upper)
            _collect_timer_bounds(formula.left, result)
            _collect_timer_bounds(formula.right, result)

        case LTLfTag.AND | LTLfTag.OR | LTLfTag.IMPLIES | LTLfTag.UNTIL | LTLfTag.RELEASE:
            _collect_timer_bounds(formula.left, result)
            _collect_timer_bounds(formula.right, result)

        case LTLfTag.NOT | LTLfTag.NEXT | LTLfTag.GLOBALLY | LTLfTag.FINALLY:
            _collect_timer_bounds(formula.inner, result)

        case LTLfTag.TOP | LTLfTag.BOTTOM | LTLfTag.ATOM:
            pass  # Terminal nodes, nothing to do
