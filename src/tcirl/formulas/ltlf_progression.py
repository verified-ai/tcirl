"""LTLf formula progression (Brzozowski derivative for temporal logic).

This module provides the Progressor class which computes formula progression
with memoization. It depends on LTLfInterner for formula construction.
"""

from __future__ import annotations

from collections.abc import Callable

from tcirl.formulas.ltlf_interned import LTLfInterned, LTLfTag
from tcirl.formulas.ltlf_interner import LTLfInterner


class Progressor[P]:
    """Computes LTLf formula progression with memoization.

    The Progressor is the Brzozowski derivative for LTLf: given a formula and
    a label (set of true propositions), it computes the successor formula.
    """

    def __init__(self, interner: LTLfInterner[P]) -> None:
        self._interner = interner
        self._cache: dict[tuple[int, frozenset[int]], LTLfInterned] = {}

    def progress(self, formula: LTLfInterned, label: frozenset[int]) -> LTLfInterned:
        """Compute formula progression after reading label (memoized)."""
        cache_key = (formula.formula_id, label)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        result = self._progress_impl(formula, label)
        self._cache[cache_key] = result
        return result

    def _progress_impl(self, formula: LTLfInterned, label: frozenset[int]) -> LTLfInterned:
        """Compute formula progression after reading label."""
        i = self._interner

        match formula.tag:
            case LTLfTag.ATOM:
                return i.make_top() if formula.prop in label else i.make_bottom()

            case LTLfTag.TOP:
                return i.make_top()

            case LTLfTag.BOTTOM:
                return i.make_bottom()

            case LTLfTag.NOT:
                return i.make_not(self.progress(formula.inner, label))

            case LTLfTag.AND:
                return i.make_and(
                    self.progress(formula.left, label),
                    self.progress(formula.right, label),
                )

            case LTLfTag.OR:
                return i.make_or(
                    self.progress(formula.left, label),
                    self.progress(formula.right, label),
                )

            case LTLfTag.IMPLIES:
                return i.make_implies(
                    self.progress(formula.left, label),
                    self.progress(formula.right, label),
                )

            case LTLfTag.NEXT:
                # X(phi) progresses to phi
                return formula.inner

            case LTLfTag.GLOBALLY:
                # G(phi) -> phi & G(phi)
                progressed = self.progress(formula.inner, label)
                return i.make_and(progressed, i.make_globally(formula.inner))

            case LTLfTag.FINALLY:
                # F(phi) -> phi | F(phi)
                progressed = self.progress(formula.inner, label)
                return i.make_or(progressed, i.make_finally(formula.inner))

            case LTLfTag.GLOBALLY_BOUND:
                # All-flavored: phi_now must hold in the active phase, so
                # combine with AND.
                return self._progress_unary_bound(formula, label, i.make_globally_bound, i.make_and)

            case LTLfTag.FINALLY_BOUND:
                # Exists-flavored: phi_now suffices in the active phase, so
                # combine with OR.
                return self._progress_unary_bound(formula, label, i.make_finally_bound, i.make_or)

            case LTLfTag.WEAK_FINALLY_BOUND:
                # Same progression as F^[lo,hi], but creates weak variants.
                return self._progress_unary_bound(
                    formula, label, i.make_weak_finally_bound, i.make_or
                )

            case LTLfTag.UNTIL:
                # phi U psi -> psi | (phi & (phi U psi))
                progressed_r = self.progress(formula.right, label)
                progressed_l = self.progress(formula.left, label)
                return i.make_or(
                    progressed_r,
                    i.make_and(progressed_l, i.make_until(formula.left, formula.right)),
                )

            case LTLfTag.RELEASE:
                # phi R psi -> psi & (phi | (phi R psi))
                progressed_r = self.progress(formula.right, label)
                progressed_l = self.progress(formula.left, label)
                return i.make_and(
                    progressed_r,
                    i.make_or(progressed_l, i.make_release(formula.left, formula.right)),
                )

            case LTLfTag.UNTIL_BOUND:
                return self._progress_until_bound(formula, label, i.make_until_bound)

            case LTLfTag.WEAK_UNTIL_BOUND:
                # Same progression as U^[lo,hi], but creates weak variants.
                return self._progress_until_bound(formula, label, i.make_weak_until_bound)

    def _progress_unary_bound(
        self,
        formula: LTLfInterned,
        label: frozenset[int],
        make: Callable[[int, int, LTLfInterned, str], LTLfInterned],
        combine: Callable[[LTLfInterned, LTLfInterned], LTLfInterned],
    ) -> LTLfInterned:
        """Progress a unary bounded operator (G^/F^/WF^, selected by `make`/`combine`).

        - lower > 0: waiting phase, just decrement both bounds
        - lower == 0, upper > 0: combine(phi_now, op^[0, upper-1](phi))
        - lower == 0, upper == 0: just phi_now
        """
        lower = formula.lower
        upper = formula.upper
        if lower > 0:
            return make(lower - 1, upper - 1, formula.inner, formula.timer_name)
        progressed = self.progress(formula.inner, label)
        if upper == 0:
            return progressed
        return combine(progressed, make(0, upper - 1, formula.inner, formula.timer_name))

    def _progress_until_bound(
        self,
        formula: LTLfInterned,
        label: frozenset[int],
        make: Callable[[int, int, LTLfInterned, LTLfInterned, str], LTLfInterned],
    ) -> LTLfInterned:
        """Progress a bounded until (U^/WU^, selected by `make`).

        - lower > 0: waiting phase, left must hold; decrement both bounds
        - lower == 0: active phase; right_now | (left_now & op^[0, upper-1](left, right)), with early exits when right_now or left_now is constant or upper == 0
        """
        i = self._interner
        lower = formula.lower
        upper = formula.upper
        left = formula.left
        right = formula.right

        if lower > 0:
            left_now = self.progress(left, label)
            if left_now.tag == LTLfTag.BOTTOM:
                return i.make_bottom()
            return i.make_and(
                left_now,
                make(lower - 1, upper - 1, left, right, formula.timer_name),
            )

        right_now = self.progress(right, label)

        if right_now.tag == LTLfTag.TOP:
            return i.make_top()

        if right_now.tag == LTLfTag.BOTTOM:
            left_now = self.progress(left, label)
            if left_now.tag == LTLfTag.BOTTOM:
                return i.make_bottom()
            if upper == 0:
                return i.make_bottom()
            return i.make_and(
                left_now,
                make(0, upper - 1, left, right, formula.timer_name),
            )

        left_now = self.progress(left, label)

        if upper == 0:
            return right_now

        if left_now.tag == LTLfTag.BOTTOM:
            return right_now

        continuation = make(0, upper - 1, left, right, formula.timer_name)
        return i.make_or(right_now, i.make_and(left_now, continuation))
