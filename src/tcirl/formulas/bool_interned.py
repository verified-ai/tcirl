"""Interned boolean formulas using int proposition indices."""

from __future__ import annotations

from enum import IntEnum
from typing import override


class BoolTag(IntEnum):
    """Tags for fast boolean formula type dispatch."""

    TOP = 0
    BOTTOM = 1
    ATOM = 2
    NOT = 3
    AND = 4
    OR = 5


type BoolInternKey = tuple[BoolTag, tuple[int | BoolInternKey, ...]]


class BoolInterned:
    """Interned boolean formula with int-based propositions."""

    __slots__ = ("formula_id", "inner", "key", "left", "prop", "right", "tag")

    formula_id: int
    tag: BoolTag
    prop: int  # For ATOM
    inner: BoolInterned  # For NOT
    left: BoolInterned  # For AND, OR
    right: BoolInterned  # For AND, OR
    key: BoolInternKey

    def __init__(
        self,
        formula_id: int,
        tag: BoolTag,
        key: BoolInternKey,
        *,
        prop: int = -1,
        inner: BoolInterned | None = None,
        left: BoolInterned | None = None,
        right: BoolInterned | None = None,
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

    @override
    def __hash__(self) -> int:
        return self.formula_id

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BoolInterned):
            return NotImplemented
        return self.formula_id == other.formula_id

    @override
    def __repr__(self) -> str:
        match self.tag:
            case BoolTag.TOP:
                return "T"
            case BoolTag.BOTTOM:
                return "⊥"
            case BoolTag.ATOM:
                return f"p{self.prop}"
            case BoolTag.NOT:
                return f"¬({self.inner!r})"
            case BoolTag.AND:
                return f"({self.left!r} ∧ {self.right!r})"
            case BoolTag.OR:
                return f"({self.left!r} V {self.right!r})"
