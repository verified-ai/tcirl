"""Interned LTLf formulas using int proposition indices."""

from __future__ import annotations

from enum import IntEnum
from typing import override


class LTLfTag(IntEnum):
    """Tags for fast LTLf formula type dispatch."""

    TOP = 0
    BOTTOM = 1
    ATOM = 2
    NOT = 3
    AND = 4
    OR = 5
    IMPLIES = 6
    NEXT = 7
    GLOBALLY = 8
    FINALLY = 9
    GLOBALLY_BOUND = 10
    FINALLY_BOUND = 11
    UNTIL = 12
    RELEASE = 13
    UNTIL_BOUND = 14
    WEAK_FINALLY_BOUND = 15
    WEAK_UNTIL_BOUND = 16


type LTLfInternKey = tuple[LTLfTag, tuple[int | str | LTLfInternKey, ...]]


class LTLfInterned:
    """Interned LTLf formula with int-based propositions."""

    __slots__ = (
        "formula_id",
        "inner",
        "key",
        "left",
        "lower",
        "prop",
        "right",
        "tag",
        "timer_name",
        "upper",
    )

    formula_id: int
    tag: LTLfTag
    key: LTLfInternKey
    prop: int  # For ATOM
    inner: LTLfInterned  # For NOT, NEXT, GLOBALLY, FINALLY, GLOBALLY_BOUND, FINALLY_BOUND, WEAK_FINALLY_BOUND
    left: LTLfInterned  # For AND, OR, IMPLIES, UNTIL, RELEASE, UNTIL_BOUND, WEAK_UNTIL_BOUND
    right: LTLfInterned  # For AND, OR, IMPLIES, UNTIL, RELEASE, UNTIL_BOUND, WEAK_UNTIL_BOUND
    lower: int  # For *_BOUND - lower bound of time window
    upper: int  # For *_BOUND - upper bound of time window
    timer_name: str  # For *_BOUND - the timer name

    def __init__(
        self,
        formula_id: int,
        tag: LTLfTag,
        key: LTLfInternKey,
        *,
        prop: int = -1,
        inner: LTLfInterned | None = None,
        left: LTLfInterned | None = None,
        right: LTLfInterned | None = None,
        lower: int = -1,
        upper: int = -1,
        timer_name: str = "",
    ) -> None:
        self.formula_id = formula_id
        self.tag = tag
        self.key = key
        if prop != -1:
            self.prop = prop
        if inner is not None:
            self.inner = inner
        if left is not None:
            self.left = left
        if right is not None:
            self.right = right
        if lower != -1:
            self.lower = lower
        if upper != -1:
            self.upper = upper
        if timer_name:
            self.timer_name = timer_name

    @override
    def __hash__(self) -> int:
        return self.formula_id

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LTLfInterned):
            return NotImplemented
        return self.formula_id == other.formula_id

    @override
    def __repr__(self) -> str:
        match self.tag:
            case LTLfTag.TOP:
                return "T"
            case LTLfTag.BOTTOM:
                return "⊥"
            case LTLfTag.ATOM:
                return f"p{self.prop}"
            case LTLfTag.NOT:
                return f"¬({self.inner!r})"
            case LTLfTag.AND:
                return f"({self.left!r} ∧ {self.right!r})"
            case LTLfTag.OR:
                return f"({self.left!r} | {self.right!r})"
            case LTLfTag.IMPLIES:
                return f"({self.left!r} → {self.right!r})"
            case LTLfTag.NEXT:
                return f"X({self.inner!r})"
            case LTLfTag.GLOBALLY:
                return f"G({self.inner!r})"
            case LTLfTag.FINALLY:
                return f"F({self.inner!r})"
            case LTLfTag.GLOBALLY_BOUND:
                return f"G^[{self.lower},{self.upper}]({self.inner!r})"
            case LTLfTag.FINALLY_BOUND:
                return f"F^[{self.lower},{self.upper}]({self.inner!r})"
            case LTLfTag.UNTIL:
                return f"({self.left!r} U {self.right!r})"
            case LTLfTag.RELEASE:
                return f"({self.left!r} R {self.right!r})"
            case LTLfTag.UNTIL_BOUND:
                return f"({self.left!r} U^[{self.lower},{self.upper}] {self.right!r})"
            case LTLfTag.WEAK_FINALLY_BOUND:
                return f"WF^[{self.lower},{self.upper}]({self.inner!r})"
            case LTLfTag.WEAK_UNTIL_BOUND:
                return f"({self.left!r} WU^[{self.lower},{self.upper}] {self.right!r})"
