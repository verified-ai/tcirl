"""LTLf formula interner - compiles syntax to interned formulas with normalization.

This module provides the LTLfInterner class which:
1. Interns formulas (deduplicates semantically equivalent formulas)
2. Normalizes formulas to canonical forms (via LTLfNormalizationMixin)
3. Compiles LTLf syntax AST to interned representations
"""

from __future__ import annotations

from typing import override

from tcirl.formulas.ltlf_interned import (
    LTLfInterned,
    LTLfInternKey,
    LTLfTag,
)
from tcirl.formulas.ltlf_normalization import LTLfNormalizationMixin
from tcirl.formulas.ltlf_syntax import (
    And,
    Atom,
    Bottom,
    Finally,
    FinallyBound,
    Globally,
    GloballyBound,
    Implies,
    LTLfSyntax,
    Next,
    Not,
    Or,
    Release,
    Top,
    Until,
    UntilBound,
    WeakFinallyBound,
    WeakUntilBound,
)
from tcirl.formulas.ltlf_timers import TimerRegistry


class LTLfInterner[P](LTLfNormalizationMixin):
    """Interns LTLf formulas with normalization for canonical representations."""

    def __init__(self, prop_to_index: dict[P, int] | None = None) -> None:
        self._intern_table: dict[LTLfInternKey, LTLfInterned] = {}
        self._id_counter: int = 0
        self._prop_to_index: dict[P, int] = prop_to_index.copy() if prop_to_index else {}
        self._accepts_at_end_cache: dict[int, bool] = {}
        self.timers = TimerRegistry()

        # Singletons
        self._top: LTLfInterned | None = None
        self._bottom: LTLfInterned | None = None

    def _next_id(self) -> int:
        self._id_counter += 1
        return self._id_counter

    def _get_prop_index(self, prop: P) -> int:
        if prop not in self._prop_to_index:
            self._prop_to_index[prop] = len(self._prop_to_index)
        return self._prop_to_index[prop]

    # === Interning primitives ===

    @override
    def _intern(
        self,
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
    ) -> LTLfInterned:
        if key in self._intern_table:
            return self._intern_table[key]
        formula = LTLfInterned(
            self._next_id(),
            tag,
            key,
            prop=prop,
            inner=inner,
            left=left,
            right=right,
            lower=lower,
            upper=upper,
            timer_name=timer_name,
        )
        self._intern_table[key] = formula
        return formula

    @override
    def make_top(self) -> LTLfInterned:
        if self._top is None:
            key: LTLfInternKey = (LTLfTag.TOP, ())
            self._top = self._intern(LTLfTag.TOP, key)
        return self._top

    @override
    def make_bottom(self) -> LTLfInterned:
        if self._bottom is None:
            key: LTLfInternKey = (LTLfTag.BOTTOM, ())
            self._bottom = self._intern(LTLfTag.BOTTOM, key)
        return self._bottom

    def make_atom(self, prop_idx: int) -> LTLfInterned:
        key: LTLfInternKey = (LTLfTag.ATOM, (prop_idx,))
        return self._intern(LTLfTag.ATOM, key, prop=prop_idx)

    def make_not(self, inner: LTLfInterned) -> LTLfInterned:
        """Create negation with NNF transformation.

        Pushes negation inward using De Morgan's laws and temporal duality,
        producing Negation Normal Form where negation only appears on atoms.
        """
        match inner.tag:
            # Constants
            case LTLfTag.TOP:
                return self.make_bottom()
            case LTLfTag.BOTTOM:
                return self.make_top()
            # Double negation elimination
            case LTLfTag.NOT:
                return inner.inner
            # De Morgan's laws
            case LTLfTag.AND:
                return self.make_or(self.make_not(inner.left), self.make_not(inner.right))
            case LTLfTag.OR:
                return self.make_and(self.make_not(inner.left), self.make_not(inner.right))
            # Implication: ~(A -> B) = A & ~B
            case LTLfTag.IMPLIES:
                return self.make_and(inner.left, self.make_not(inner.right))
            # Temporal duality
            case LTLfTag.GLOBALLY:
                return self.make_finally(self.make_not(inner.inner))
            case LTLfTag.FINALLY:
                return self.make_globally(self.make_not(inner.inner))
            # Until/Release duality: ~(p U q) = ~q R ~p, ~(p R q) = ~q U ~p
            case LTLfTag.UNTIL:
                return self.make_release(self.make_not(inner.right), self.make_not(inner.left))
            case LTLfTag.RELEASE:
                return self.make_until(self.make_not(inner.right), self.make_not(inner.left))
            # Next: ~X(p) -> X(~p) (NNF rewrite). Under strong next the two differ
            # only at trace end, where ~X(p) holds but X(~p) does not -- the precise
            # dual is weak-next(~p), which this AST does not have.
            case LTLfTag.NEXT:
                return self.make_next(self.make_not(inner.inner))
            case _:
                pass

        key: LTLfInternKey = (LTLfTag.NOT, (inner.key,))
        return self._intern(LTLfTag.NOT, key, inner=inner)

    def make_and(self, left: LTLfInterned, right: LTLfInterned) -> LTLfInterned:
        # Quick paths
        if left.tag == LTLfTag.BOTTOM or right.tag == LTLfTag.BOTTOM:
            return self.make_bottom()
        if left.tag == LTLfTag.TOP:
            return right
        if right.tag == LTLfTag.TOP:
            return left
        if left is right:
            return left

        # Flatten and normalize (from mixin)
        conjuncts = self._flatten_and(left) + self._flatten_and(right)
        conjuncts = self._normalize_conjuncts(conjuncts)

        if not conjuncts:
            return self.make_top()
        if len(conjuncts) == 1:
            return conjuncts[0]

        return self._build_and_tree(conjuncts)

    def make_or(self, left: LTLfInterned, right: LTLfInterned) -> LTLfInterned:
        # Quick paths
        if left.tag == LTLfTag.TOP or right.tag == LTLfTag.TOP:
            return self.make_top()
        if left.tag == LTLfTag.BOTTOM:
            return right
        if right.tag == LTLfTag.BOTTOM:
            return left
        if left is right:
            return left

        # Flatten and normalize (from mixin)
        disjuncts = self._flatten_or(left) + self._flatten_or(right)
        disjuncts = self._normalize_disjuncts(disjuncts)

        if not disjuncts:
            return self.make_bottom()
        if len(disjuncts) == 1:
            return disjuncts[0]

        return self._build_or_tree(disjuncts)

    def make_implies(self, antecedent: LTLfInterned, consequent: LTLfInterned) -> LTLfInterned:
        """Create implication, normalized to disjunction: A -> B = ~A | B."""
        return self.make_or(self.make_not(antecedent), consequent)

    def make_next(self, inner: LTLfInterned) -> LTLfInterned:
        """Create next with constant propagation and distribution."""
        match inner.tag:
            case LTLfTag.BOTTOM:
                return self.make_bottom()
            case LTLfTag.TOP:
                return self.make_top()  # X(T) = T in weak next semantics
            # Distribution: X(p & q) = X(p) & X(q), X(p | q) = X(p) | X(q)
            case LTLfTag.AND:
                return self.make_and(self.make_next(inner.left), self.make_next(inner.right))
            case LTLfTag.OR:
                return self.make_or(self.make_next(inner.left), self.make_next(inner.right))
            case _:
                pass

        key: LTLfInternKey = (LTLfTag.NEXT, (inner.key,))
        return self._intern(LTLfTag.NEXT, key, inner=inner)

    def make_globally(self, inner: LTLfInterned) -> LTLfInterned:
        """Create globally with normalization rules."""
        match inner.tag:
            case LTLfTag.TOP:
                return self.make_top()
            case LTLfTag.BOTTOM:
                return self.make_bottom()
            case LTLfTag.GLOBALLY:
                return inner  # G(G(p)) = G(p)
            case LTLfTag.AND:
                # G(p & q) = G(p) & G(q)
                return self.make_and(
                    self.make_globally(inner.left),
                    self.make_globally(inner.right),
                )
            case _:
                pass

        key: LTLfInternKey = (LTLfTag.GLOBALLY, (inner.key,))
        return self._intern(LTLfTag.GLOBALLY, key, inner=inner)

    def make_finally(self, inner: LTLfInterned) -> LTLfInterned:
        """Create finally with normalization rules."""
        match inner.tag:
            case LTLfTag.TOP:
                return self.make_top()
            case LTLfTag.BOTTOM:
                return self.make_bottom()
            case LTLfTag.FINALLY:
                return inner  # F(F(p)) = F(p)
            case LTLfTag.OR:
                # F(p | q) = F(p) | F(q)
                return self.make_or(
                    self.make_finally(inner.left),
                    self.make_finally(inner.right),
                )
            case _:
                pass

        key: LTLfInternKey = (LTLfTag.FINALLY, (inner.key,))
        return self._intern(LTLfTag.FINALLY, key, inner=inner)

    @override
    def make_globally_bound(
        self, lower: int, upper: int, inner: LTLfInterned, name: str
    ) -> LTLfInterned:
        if lower < 0:
            raise ValueError(f"lower must be >= 0, got {lower}")
        if upper < lower:
            raise ValueError(f"upper must be >= lower, got upper={upper} < lower={lower}")
        match inner.tag:
            case LTLfTag.TOP:
                return self.make_top()
            case LTLfTag.BOTTOM:
                return self.make_bottom()
            case _:
                pass

        key: LTLfInternKey = (LTLfTag.GLOBALLY_BOUND, (lower, upper, inner.key, name))
        return self._intern(
            LTLfTag.GLOBALLY_BOUND, key, inner=inner, lower=lower, upper=upper, timer_name=name
        )

    @override
    def make_finally_bound(
        self, lower: int, upper: int, inner: LTLfInterned, name: str
    ) -> LTLfInterned:
        if lower < 0:
            raise ValueError(f"lower must be >= 0, got {lower}")
        if upper < lower:
            raise ValueError(f"upper must be >= lower, got upper={upper} < lower={lower}")
        match inner.tag:
            case LTLfTag.TOP:
                return self.make_top()
            case LTLfTag.BOTTOM:
                return self.make_bottom()
            case _:
                pass

        key: LTLfInternKey = (LTLfTag.FINALLY_BOUND, (lower, upper, inner.key, name))
        return self._intern(
            LTLfTag.FINALLY_BOUND, key, inner=inner, lower=lower, upper=upper, timer_name=name
        )

    def make_until(self, left: LTLfInterned, right: LTLfInterned) -> LTLfInterned:
        """Create until with normalization rules."""
        if right.tag == LTLfTag.TOP:
            return self.make_top()
        if right.tag == LTLfTag.BOTTOM:
            return self.make_bottom()
        if left.tag == LTLfTag.BOTTOM:
            return right
        if left.tag == LTLfTag.TOP:
            # T U p = F(p)
            return self.make_finally(right)

        key: LTLfInternKey = (LTLfTag.UNTIL, (left.key, right.key))
        return self._intern(LTLfTag.UNTIL, key, left=left, right=right)

    def make_release(self, left: LTLfInterned, right: LTLfInterned) -> LTLfInterned:
        """Create release with normalization rules."""
        if right.tag == LTLfTag.BOTTOM:
            return self.make_bottom()
        if right.tag == LTLfTag.TOP:
            return self.make_top()
        if left.tag == LTLfTag.TOP:
            return right
        if left.tag == LTLfTag.BOTTOM:
            # bot R p = G(p)
            return self.make_globally(right)

        key: LTLfInternKey = (LTLfTag.RELEASE, (left.key, right.key))
        return self._intern(LTLfTag.RELEASE, key, left=left, right=right)

    @override
    def make_until_bound(
        self, lower: int, upper: int, left: LTLfInterned, right: LTLfInterned, name: str
    ) -> LTLfInterned:
        if lower < 0:
            raise ValueError(f"lower must be >= 0, got {lower}")
        if upper < lower:
            raise ValueError(f"upper must be >= lower, got upper={upper} < lower={lower}")
        if right.tag == LTLfTag.TOP:
            return self.make_top()
        if right.tag == LTLfTag.BOTTOM:
            return self.make_bottom()
        if left.tag == LTLfTag.BOTTOM:
            if lower > 0:
                # Can't hold left during waiting phase
                return self.make_bottom()
            return self.make_finally_bound(0, upper, right, name)
        if left.tag == LTLfTag.TOP:
            return self.make_finally_bound(lower, upper, right, name)

        key: LTLfInternKey = (LTLfTag.UNTIL_BOUND, (lower, upper, left.key, right.key, name))
        return self._intern(
            LTLfTag.UNTIL_BOUND,
            key,
            left=left,
            right=right,
            lower=lower,
            upper=upper,
            timer_name=name,
        )

    @override
    def make_weak_finally_bound(
        self, lower: int, upper: int, inner: LTLfInterned, name: str
    ) -> LTLfInterned:
        if lower < 0:
            raise ValueError(f"lower must be >= 0, got {lower}")
        if upper < lower:
            raise ValueError(f"upper must be >= lower, got upper={upper} < lower={lower}")
        match inner.tag:
            case LTLfTag.TOP:
                return self.make_top()
            case LTLfTag.BOTTOM:
                return self.make_bottom()
            case _:
                pass

        key: LTLfInternKey = (LTLfTag.WEAK_FINALLY_BOUND, (lower, upper, inner.key, name))
        return self._intern(
            LTLfTag.WEAK_FINALLY_BOUND, key, inner=inner, lower=lower, upper=upper, timer_name=name
        )

    @override
    def make_weak_until_bound(
        self, lower: int, upper: int, left: LTLfInterned, right: LTLfInterned, name: str
    ) -> LTLfInterned:
        if lower < 0:
            raise ValueError(f"lower must be >= 0, got {lower}")
        if upper < lower:
            raise ValueError(f"upper must be >= lower, got upper={upper} < lower={lower}")
        if right.tag == LTLfTag.TOP:
            return self.make_top()
        if right.tag == LTLfTag.BOTTOM:
            return self.make_bottom()
        if left.tag == LTLfTag.BOTTOM:
            if lower > 0:
                return self.make_bottom()
            return self.make_weak_finally_bound(0, upper, right, name)
        if left.tag == LTLfTag.TOP:
            return self.make_weak_finally_bound(lower, upper, right, name)

        key: LTLfInternKey = (LTLfTag.WEAK_UNTIL_BOUND, (lower, upper, left.key, right.key, name))
        return self._intern(
            LTLfTag.WEAK_UNTIL_BOUND,
            key,
            left=left,
            right=right,
            lower=lower,
            upper=upper,
            timer_name=name,
        )

    # === Compilation ===
    def compile(self, syntax: LTLfSyntax[P]) -> LTLfInterned:
        """Compile syntax formula to interned formula, auto-registering named obligations."""
        match syntax:
            case Top():
                return self.make_top()
            case Bottom():
                return self.make_bottom()
            case Atom(prop=p):
                return self.make_atom(self._get_prop_index(p))
            case Not(inner=inner):
                return self.make_not(self.compile(inner))
            case And(left=left, right=right):
                return self.make_and(self.compile(left), self.compile(right))
            case Or(left=left, right=right):
                return self.make_or(self.compile(left), self.compile(right))
            case Implies(antecedent=ant, consequent=cons):
                return self.make_implies(self.compile(ant), self.compile(cons))
            case Next(inner=inner):
                return self.make_next(self.compile(inner))
            case Globally(inner=inner):
                return self.make_globally(self.compile(inner))
            case Finally(inner=inner):
                return self.make_finally(self.compile(inner))
            case GloballyBound(lower=lower, upper=upper, inner=inner, name=name):
                compiled_inner = self.compile(inner)
                self.timers.register(name, lower, upper, "GloballyBound")
                return self.make_globally_bound(lower, upper, compiled_inner, name)
            case FinallyBound(lower=lower, upper=upper, inner=inner, name=name):
                compiled_inner = self.compile(inner)
                self.timers.register(name, lower, upper, "FinallyBound")
                return self.make_finally_bound(lower, upper, compiled_inner, name)
            case Until(left=left, right=right):
                return self.make_until(self.compile(left), self.compile(right))
            case Release(left=left, right=right):
                return self.make_release(self.compile(left), self.compile(right))
            case UntilBound(lower=lower, upper=upper, left=left, right=right, name=name):
                compiled_left = self.compile(left)
                compiled_right = self.compile(right)
                self.timers.register(name, lower, upper, "UntilBound")
                return self.make_until_bound(lower, upper, compiled_left, compiled_right, name)
            case WeakFinallyBound(lower=lower, upper=upper, inner=inner, name=name):
                compiled_inner = self.compile(inner)
                self.timers.register(name, lower, upper, "WeakFinallyBound")
                return self.make_weak_finally_bound(lower, upper, compiled_inner, name)
            case WeakUntilBound(lower=lower, upper=upper, left=left, right=right, name=name):
                compiled_left = self.compile(left)
                compiled_right = self.compile(right)
                self.timers.register(name, lower, upper, "WeakUntilBound")
                return self.make_weak_until_bound(lower, upper, compiled_left, compiled_right, name)

    # === Residual Acceptance ===
    def accepts_at_end(self, formula: LTLfInterned) -> bool:
        """Determine if a formula is satisfied when the trace ends immediately.

        This is the residual acceptance function A(psi) from LTLf semantics.
        It returns True iff the formula is satisfied by the empty *suffix* -
        i.e., if no more symbols arrive, is the current obligation fulfilled?

        Note: This is NOT about empty input traces (which are disallowed in LTLf).
        Rather, it determines DFA accepting states: a state (formula) is accepting
        if the trace can end here and the specification is satisfied.

        Key cases:
        - Safety (Globally, GloballyBound, Release): True (vacuously satisfied)
        - Reachability (Finally, FinallyBound, Until, UntilBound): False (eventuality unfulfilled)
        - Weak bounded (WeakFinallyBound, WeakUntilBound): True (obligation discharged at trace end)
        - Next: False (strong next requires a successor state)
        """
        fid = formula.formula_id
        if fid in self._accepts_at_end_cache:
            return self._accepts_at_end_cache[fid]

        result = self._compute_accepts_at_end(formula)
        self._accepts_at_end_cache[fid] = result
        return result

    def _compute_accepts_at_end(self, formula: LTLfInterned) -> bool:
        match formula.tag:
            case LTLfTag.TOP:
                return True
            case LTLfTag.BOTTOM:
                return False
            case LTLfTag.ATOM:
                return False
            case LTLfTag.NOT:
                return not self.accepts_at_end(formula.inner)
            case LTLfTag.AND:
                return self.accepts_at_end(formula.left) and self.accepts_at_end(formula.right)
            case LTLfTag.OR:
                return self.accepts_at_end(formula.left) or self.accepts_at_end(formula.right)
            case LTLfTag.IMPLIES:
                return not self.accepts_at_end(formula.left) or self.accepts_at_end(formula.right)
            case LTLfTag.NEXT:
                return False  # Strong next fails on empty suffix
            case LTLfTag.GLOBALLY:
                return True  # Vacuously true: no steps to violate
            case LTLfTag.FINALLY:
                return False  # Eventuality unfulfilled
            case LTLfTag.GLOBALLY_BOUND:
                return True  # Safety: no remaining steps = no violation possible
            case LTLfTag.FINALLY_BOUND:
                return False  # Eventuality unfulfilled within bound
            case LTLfTag.UNTIL:
                return False  # Right side never happened
            case LTLfTag.RELEASE:
                return True  # Dual of Until: vacuously true
            case LTLfTag.UNTIL_BOUND:
                return False  # Right side never happened within bound
            case LTLfTag.WEAK_FINALLY_BOUND:
                return True  # Weak: obligation discharged at trace end
            case LTLfTag.WEAK_UNTIL_BOUND:
                return True  # Weak: obligation discharged at trace end
