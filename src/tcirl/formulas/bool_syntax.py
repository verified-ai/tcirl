"""Boolean expression syntax - typed AST without interning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import override

type BoolSyntax[P] = BoolAtom[P] | BoolTop[P] | BoolBottom[P] | BoolNot[P] | BoolAnd[P] | BoolOr[P]


class BoolOps[P]:
    """Operator sugar and the derived `appears` property shared by all bool node classes.

    Carries no state and is never instantiated directly. Must NOT be added to the BoolSyntax union: exhaustive `match` relies on the alias listing exactly the concrete node classes. Bodies funnel `self` through `_as_syntax` because pyright requires an explicit `self` annotation to be a supertype of the defining class, which the union is not.
    """

    __slots__ = ()

    def __and__(self, other: BoolSyntax[P]) -> BoolAnd[P]:
        return BoolAnd(left=_as_syntax(self), right=other)

    def __or__(self, other: BoolSyntax[P]) -> BoolOr[P]:
        return BoolOr(left=_as_syntax(self), right=other)

    def __invert__(self) -> BoolNot[P]:
        return BoolNot(inner=_as_syntax(self))

    def __rshift__(self, other: BoolSyntax[P]) -> BoolOr[P]:
        return BoolOr(left=BoolNot(inner=_as_syntax(self)), right=other)

    @property
    def appears(self) -> frozenset[P]:
        return collect_appears(_as_syntax(self))


def _as_syntax[P](value: BoolOps[P]) -> BoolSyntax[P]:
    """Narrow the mixin type to the closed union; every subclass of BoolOps is a union member."""
    assert isinstance(value, (BoolAtom, BoolTop, BoolBottom, BoolNot, BoolAnd, BoolOr))
    return value


@dataclass(frozen=True, slots=True)
class BoolAtom[P](BoolOps[P]):
    """Atomic proposition."""

    prop: P

    @override
    def __str__(self) -> str:
        return str(self.prop)


@dataclass(frozen=True, slots=True, kw_only=True)
class BoolTop[P](BoolOps[P]):
    """Constant true (T)."""

    @override
    def __str__(self) -> str:
        return "T"


@dataclass(frozen=True, slots=True, kw_only=True)
class BoolBottom[P](BoolOps[P]):
    """Constant false."""

    @override
    def __str__(self) -> str:
        return "⊥"


@dataclass(frozen=True, slots=True, kw_only=True)
class BoolNot[P](BoolOps[P]):
    """Negation: not p."""

    inner: BoolSyntax[P]

    @override
    def __str__(self) -> str:
        return f"¬{self.inner}"


@dataclass(frozen=True, slots=True, kw_only=True)
class BoolAnd[P](BoolOps[P]):
    """Conjunction: p & q."""

    left: BoolSyntax[P]
    right: BoolSyntax[P]

    @override
    def __str__(self) -> str:
        return f"({self.left} ∧ {self.right})"


@dataclass(frozen=True, slots=True, kw_only=True)
class BoolOr[P](BoolOps[P]):
    """Disjunction: p | q."""

    left: BoolSyntax[P]
    right: BoolSyntax[P]

    @override
    def __str__(self) -> str:
        return f"({self.left} V {self.right})"


def collect_appears[P](formula: BoolSyntax[P]) -> frozenset[P]:
    """Collect all atomic propositions appearing in a formula."""
    match formula:
        case BoolAtom(prop=p):
            return frozenset({p})
        case BoolTop() | BoolBottom():
            return frozenset()
        case BoolNot(inner=inner):
            return collect_appears(inner)
        case BoolAnd(left=left, right=right) | BoolOr(left=left, right=right):
            return collect_appears(left) | collect_appears(right)
