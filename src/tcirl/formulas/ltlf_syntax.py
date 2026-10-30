"""LTLf formula syntax - typed AST without interning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import override

type LTLfSyntax[P] = (
    Atom[P]
    | Top[P]
    | Bottom[P]
    | Not[P]
    | And[P]
    | Or[P]
    | Implies[P]
    | Next[P]
    | Globally[P]
    | Finally[P]
    | Until[P]
    | Release[P]
    | GloballyBound[P]
    | FinallyBound[P]
    | UntilBound[P]
    | WeakFinallyBound[P]
    | WeakUntilBound[P]
)


class LTLfOps[P]:
    """Operator sugar and derived properties shared by all LTLf node classes.

    Carries no state and is never instantiated directly. Must NOT be added to the LTLfSyntax union: exhaustive `match` relies on the alias listing exactly the concrete node classes. Bodies funnel `self` through `_as_syntax` because pyright requires an explicit `self` annotation to be a supertype of the defining class, which the union is not.
    """

    __slots__ = ()

    def __and__(self, other: LTLfSyntax[P]) -> And[P]:
        return And(left=_as_syntax(self), right=other)

    def __or__(self, other: LTLfSyntax[P]) -> Or[P]:
        return Or(left=_as_syntax(self), right=other)

    def __invert__(self) -> Not[P]:
        return Not(inner=_as_syntax(self))

    def __rshift__(self, other: LTLfSyntax[P]) -> Implies[P]:
        return Implies(antecedent=_as_syntax(self), consequent=other)

    @property
    def appears(self) -> frozenset[P]:
        return collect_appears(_as_syntax(self))

    @property
    def horizon(self) -> int | None:
        return collect_horizon(_as_syntax(self))


def _as_syntax[P](value: LTLfOps[P]) -> LTLfSyntax[P]:
    """Narrow the mixin type to the closed union; every subclass of LTLfOps is a union member."""
    assert isinstance(
        value,
        (
            Atom,
            Top,
            Bottom,
            Not,
            And,
            Or,
            Implies,
            Next,
            Globally,
            Finally,
            Until,
            Release,
            GloballyBound,
            FinallyBound,
            UntilBound,
            WeakFinallyBound,
            WeakUntilBound,
        ),
    )
    return value


# === Atomic ===


@dataclass(frozen=True, slots=True)
class Atom[P](LTLfOps[P]):
    """Atomic proposition."""

    prop: P

    @override
    def __str__(self) -> str:
        return str(self.prop)


@dataclass(frozen=True, slots=True, kw_only=True)
class Top[P](LTLfOps[P]):
    """Constant true (T)."""

    @override
    def __str__(self) -> str:
        return "T"


@dataclass(frozen=True, slots=True, kw_only=True)
class Bottom[P](LTLfOps[P]):
    """Constant false."""

    @override
    def __str__(self) -> str:
        return "⊥"


# === Boolean Connectives ===


@dataclass(frozen=True, slots=True, kw_only=True)
class Not[P](LTLfOps[P]):
    """Negation: not p."""

    inner: LTLfSyntax[P]

    @override
    def __str__(self) -> str:
        return f"¬{self.inner}"


@dataclass(frozen=True, slots=True, kw_only=True)
class And[P](LTLfOps[P]):
    """Conjunction: p & q."""

    left: LTLfSyntax[P]
    right: LTLfSyntax[P]

    @override
    def __str__(self) -> str:
        return f"({self.left} ∧ {self.right})"


@dataclass(frozen=True, slots=True, kw_only=True)
class Or[P](LTLfOps[P]):
    """Disjunction: p | q."""

    left: LTLfSyntax[P]
    right: LTLfSyntax[P]

    @override
    def __str__(self) -> str:
        return f"({self.left} V {self.right})"


@dataclass(frozen=True, slots=True, kw_only=True)
class Implies[P](LTLfOps[P]):
    """Implication: p -> q."""

    antecedent: LTLfSyntax[P]
    consequent: LTLfSyntax[P]

    @override
    def __str__(self) -> str:
        return f"({self.antecedent} → {self.consequent})"


# === Temporal Operators ===


@dataclass(frozen=True, slots=True)
class Next[P](LTLfOps[P]):
    """Next: X p."""

    inner: LTLfSyntax[P]

    @override
    def __str__(self) -> str:
        return f"X({self.inner})"


@dataclass(frozen=True, slots=True)
class Globally[P](LTLfOps[P]):
    """Globally: G p."""

    inner: LTLfSyntax[P]

    @override
    def __str__(self) -> str:
        return f"G({self.inner})"


@dataclass(frozen=True, slots=True)
class Finally[P](LTLfOps[P]):
    """Finally: F p."""

    inner: LTLfSyntax[P]

    @override
    def __str__(self) -> str:
        return f"F({self.inner})"


@dataclass(frozen=True, slots=True, kw_only=True)
class GloballyBound[P](LTLfOps[P]):
    """Bounded Globally: G^[lower,upper] phi."""

    lower: int
    upper: int
    inner: LTLfSyntax[P]
    name: str

    def __post_init__(self) -> None:
        if self.lower < 0:
            raise ValueError(f"lower must be >= 0, got {self.lower}")
        if self.upper < self.lower:
            raise ValueError(f"upper must be >= lower, got upper={self.upper} < lower={self.lower}")

    @override
    def __str__(self) -> str:
        return f"G^[{self.lower},{self.upper}]({self.inner})"


@dataclass(frozen=True, slots=True, kw_only=True)
class FinallyBound[P](LTLfOps[P]):
    """Bounded Finally: F^[lower,upper] phi."""

    lower: int
    upper: int
    inner: LTLfSyntax[P]
    name: str

    def __post_init__(self) -> None:
        if self.lower < 0:
            raise ValueError(f"lower must be >= 0, got {self.lower}")
        if self.upper < self.lower:
            raise ValueError(f"upper must be >= lower, got upper={self.upper} < lower={self.lower}")

    @override
    def __str__(self) -> str:
        return f"F^[{self.lower},{self.upper}]({self.inner})"


@dataclass(frozen=True, slots=True, kw_only=True)
class Until[P](LTLfOps[P]):
    """Until: p U q."""

    left: LTLfSyntax[P]
    right: LTLfSyntax[P]

    @override
    def __str__(self) -> str:
        return f"({self.left} U {self.right})"


@dataclass(frozen=True, slots=True, kw_only=True)
class Release[P](LTLfOps[P]):
    """Release: p R q."""

    left: LTLfSyntax[P]
    right: LTLfSyntax[P]

    @override
    def __str__(self) -> str:
        return f"({self.left} R {self.right})"


@dataclass(frozen=True, slots=True, kw_only=True)
class UntilBound[P](LTLfOps[P]):
    """Bounded Until: phi U^[lower,upper] psi."""

    lower: int
    upper: int
    left: LTLfSyntax[P]
    right: LTLfSyntax[P]
    name: str

    def __post_init__(self) -> None:
        if self.lower < 0:
            raise ValueError(f"lower must be >= 0, got {self.lower}")
        if self.upper < self.lower:
            raise ValueError(f"upper must be >= lower, got upper={self.upper} < lower={self.lower}")

    @override
    def __str__(self) -> str:
        return f"({self.left} U^[{self.lower},{self.upper}] {self.right})"


@dataclass(frozen=True, slots=True, kw_only=True)
class WeakFinallyBound[P](LTLfOps[P]):
    """Weak Bounded Finally: WF^[lower,upper] phi. Accepts at end of trace."""

    lower: int
    upper: int
    inner: LTLfSyntax[P]
    name: str

    def __post_init__(self) -> None:
        if self.lower < 0:
            raise ValueError(f"lower must be >= 0, got {self.lower}")
        if self.upper < self.lower:
            raise ValueError(f"upper must be >= lower, got upper={self.upper} < lower={self.lower}")

    @override
    def __str__(self) -> str:
        return f"WF^[{self.lower},{self.upper}]({self.inner})"


@dataclass(frozen=True, slots=True, kw_only=True)
class WeakUntilBound[P](LTLfOps[P]):
    """Weak Bounded Until: phi WU^[lower,upper] psi. Accepts at end of trace."""

    lower: int
    upper: int
    left: LTLfSyntax[P]
    right: LTLfSyntax[P]
    name: str

    def __post_init__(self) -> None:
        if self.lower < 0:
            raise ValueError(f"lower must be >= 0, got {self.lower}")
        if self.upper < self.lower:
            raise ValueError(f"upper must be >= lower, got upper={self.upper} < lower={self.lower}")

    @override
    def __str__(self) -> str:
        return f"({self.left} WU^[{self.lower},{self.upper}] {self.right})"


def collect_appears[P](formula: LTLfSyntax[P]) -> frozenset[P]:
    """Collect all atomic propositions appearing in a formula."""
    match formula:
        case Atom(prop=p):
            return frozenset({p})
        case Top() | Bottom():
            return frozenset()
        case (
            Not(inner=inner)
            | Next(inner=inner)
            | Globally(inner=inner)
            | Finally(inner=inner)
            | GloballyBound(inner=inner)
            | FinallyBound(inner=inner)
            | WeakFinallyBound(inner=inner)
        ):
            return collect_appears(inner)
        case (
            And(left=left, right=right)
            | Or(left=left, right=right)
            | Until(left=left, right=right)
            | Release(left=left, right=right)
            | UntilBound(left=left, right=right)
            | WeakUntilBound(left=left, right=right)
        ):
            return collect_appears(left) | collect_appears(right)
        case Implies(antecedent=ant, consequent=cons):
            return collect_appears(ant) | collect_appears(cons)


def combine_horizons(h1: int | None, h2: int | None) -> int | None:
    """Combine two horizons, returning None if either is unbounded."""
    if h1 is None or h2 is None:
        return None
    return max(h1, h2)


def collect_horizon[P](formula: LTLfSyntax[P]) -> int | None:
    """Compute the horizon of a formula, or None if unbounded."""
    match formula:
        case Atom() | Top() | Bottom():
            return 0
        case Not(inner=inner):
            return collect_horizon(inner)
        case And(left=left, right=right) | Or(left=left, right=right):
            return combine_horizons(collect_horizon(left), collect_horizon(right))
        case Implies(antecedent=ant, consequent=cons):
            return combine_horizons(collect_horizon(ant), collect_horizon(cons))
        case Next(inner=inner):
            inner_h = collect_horizon(inner)
            return None if inner_h is None else 1 + inner_h
        case Globally() | Finally() | Until() | Release():
            return None
        case (
            GloballyBound(upper=upper, inner=inner)
            | FinallyBound(upper=upper, inner=inner)
            | WeakFinallyBound(upper=upper, inner=inner)
        ):
            inner_h = collect_horizon(inner)
            return None if inner_h is None else upper + inner_h
        case (
            UntilBound(upper=upper, left=left, right=right)
            | WeakUntilBound(upper=upper, left=left, right=right)
        ):
            combined = combine_horizons(collect_horizon(left), collect_horizon(right))
            return None if combined is None else upper + combined
