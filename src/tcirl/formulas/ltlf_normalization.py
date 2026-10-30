"""LTLf formula normalization rules.

This module implements formula normalization as a mixin class. Normalization
ensures that semantically equivalent formulas produce identical interned
representations by applying canonical forms.

Normalization rules applied:
- Double negation elimination: ~~p = p
- Constants propagation: p & T = p, p | F = p, etc.
- AND/OR flattening and associativity: (a & b) & c = a & (b & c)
- Commutativity via canonical ordering: a & b = b & a (sorted by formula_id)
- Deduplication: a & a = a
- Implication rewriting: A -> B = ~A | B
- De Morgan's laws: ~(A & B) = ~A | ~B, ~(A | B) = ~A & ~B
- Temporal duality: G(p) = ~F(~p), F(p) = ~G(~p)
- Until/Release duality: ~(p U q) = ~q R ~p
- Temporal idempotence: F(F(p)) = F(p), G(G(p)) = G(p)
- Complementation: A & ~A = F, A | ~A = T
- Absorption: A | (A & B) = A, A & (A | B) = A
- Next distribution: X(p & q) = X(p) & X(q)
- Bounded operator merging: F^n(p) & F^m(p) = F^min(n,m)(p)
- Temporal distribution: G(p & q) = G(p) & G(q), F(p | q) = F(p) | F(q)
- Temporal absorption: G(p) & p = G(p), F(p) | p = F(p)
- Until/Release canonical: T U p = F(p), F R p = G(p)

The result is formulas in a canonical form close to Negation Normal Form (NNF),
where negation only appears directly on atoms.
"""

from __future__ import annotations

from tcirl.formulas.ltlf_interned import LTLfInterned, LTLfInternKey, LTLfTag


def _intervals_nested(lo1: int, hi1: int, lo2: int, hi2: int) -> bool:
    """True iff one of `[lo1,hi1]`, `[lo2,hi2]` contains the other.

    Two same-name bounded operators combined by an intersection-type merge (a
    conjunction of "exists" operators, or a disjunction of "all" operators)
    collapse to one bounded operator only when nested -- the result is then the
    inner (tighter) window `[max(lo), min(hi)]`. For overlapping-but-not-nested or
    disjoint windows the conjunction/disjunction is genuinely two constraints, not
    one, so the caller keeps both separately.
    """
    return (lo1 <= lo2 and hi2 <= hi1) or (lo2 <= lo1 and hi1 <= hi2)


def _intervals_contiguous(lo1: int, hi1: int, lo2: int, hi2: int) -> bool:
    """True iff `[lo1,hi1] union [lo2,hi2]` is itself a single interval.

    Two same-name bounded operators combined by a union-type merge (a conjunction
    of "all" operators, or a disjunction of "exists" operators) collapse to one
    bounded operator `[min(lo), max(hi)]` only when the windows overlap or are
    adjacent (no integer gap between them). With a gap the union covers steps in
    neither window, so the caller keeps both separately.
    """
    return max(lo1, lo2) <= min(hi1, hi2) + 1


# Same-name merge table for one bounded-operator kind: timer name -> (lower, upper,
# operand formulas). The operands (inner, or left/right) come from the first
# occurrence; progression copies of one source operator share them.
type _BoundsTable = dict[str, tuple[int, int, tuple[LTLfInterned, ...]]]


def _merge_bounded(
    table: _BoundsTable,
    other: list[LTLfInterned],
    formula: LTLfInterned,
    operands: tuple[LTLfInterned, ...],
    *,
    intersect: bool,
) -> None:
    """Fold one bounded operator into its kind's same-name merge table.

    `intersect=True` (exists-flavored conjunction / all-flavored disjunction) merges
    only nested windows, to the inner `[max(lo), min(hi)]`; `intersect=False` (the
    union-type merge) merges only contiguous windows, to `[min(lo), max(hi)]`. A
    non-mergeable second copy is kept separately in `other`.
    """
    name = formula.timer_name
    if name not in table:
        table[name] = (formula.lower, formula.upper, operands)
        return
    old_lo, old_hi, ops = table[name]
    if intersect:
        if _intervals_nested(old_lo, old_hi, formula.lower, formula.upper):
            table[name] = (max(old_lo, formula.lower), min(old_hi, formula.upper), ops)
        else:
            other.append(formula)
    else:
        if _intervals_contiguous(old_lo, old_hi, formula.lower, formula.upper):
            table[name] = (min(old_lo, formula.lower), max(old_hi, formula.upper), ops)
        else:
            other.append(formula)


class LTLfNormalizationMixin:
    """Mixin providing normalization methods for LTLf formula interning."""

    # Stub methods to be overridden by concrete class
    def _intern(
        self,
        tag: LTLfTag,
        key: LTLfInternKey,
        *,
        left: LTLfInterned | None = None,
        right: LTLfInterned | None = None,
    ) -> LTLfInterned:
        raise NotImplementedError

    def make_top(self) -> LTLfInterned:
        raise NotImplementedError

    def make_bottom(self) -> LTLfInterned:
        raise NotImplementedError

    def make_globally_bound(
        self, lower: int, upper: int, inner: LTLfInterned, name: str
    ) -> LTLfInterned:
        raise NotImplementedError

    def make_finally_bound(
        self, lower: int, upper: int, inner: LTLfInterned, name: str
    ) -> LTLfInterned:
        raise NotImplementedError

    def make_until_bound(
        self, lower: int, upper: int, left: LTLfInterned, right: LTLfInterned, name: str
    ) -> LTLfInterned:
        raise NotImplementedError

    def make_weak_finally_bound(
        self, lower: int, upper: int, inner: LTLfInterned, name: str
    ) -> LTLfInterned:
        raise NotImplementedError

    def make_weak_until_bound(
        self, lower: int, upper: int, left: LTLfInterned, right: LTLfInterned, name: str
    ) -> LTLfInterned:
        raise NotImplementedError

    # === Flattening ===

    def _flatten_and(self, formula: LTLfInterned) -> list[LTLfInterned]:
        """Flatten nested AND into a list of conjuncts."""
        if formula.tag == LTLfTag.AND:
            return self._flatten_and(formula.left) + self._flatten_and(formula.right)
        return [formula]

    def _flatten_or(self, formula: LTLfInterned) -> list[LTLfInterned]:
        """Flatten nested OR into a list of disjuncts."""
        if formula.tag == LTLfTag.OR:
            return self._flatten_or(formula.left) + self._flatten_or(formula.right)
        return [formula]

    # === Conjunct normalization ===

    def _normalize_conjuncts(self, conjuncts: list[LTLfInterned]) -> list[LTLfInterned]:
        """Normalize conjuncts: dedupe, complement detection, absorption, merge bounded.

        Additional rules:
        - G(p) & p = G(p)  (globally absorbs its inner)
        - G^k(p) & p = G^k(p)  (bounded globally absorbs its inner)
        """
        # Check for bottom
        if any(c.tag == LTLfTag.BOTTOM for c in conjuncts):
            return [self.make_bottom()]

        # Collect positive and negated formula IDs for complement detection
        positive_ids: set[int] = set()
        negated_inner_ids: set[int] = set()
        for c in conjuncts:
            if c.tag == LTLfTag.NOT:
                negated_inner_ids.add(c.inner.formula_id)
            else:
                positive_ids.add(c.formula_id)

        # Complement detection: p & ~p = bottom
        if positive_ids & negated_inner_ids:
            return [self.make_bottom()]

        # Collect inner formula IDs of G and G^k operators for absorption.
        # G(p) & p = G(p), so if G(p) is present, p is redundant.
        # G^[lo,hi](p) & p = G^[lo,hi](p) only when lo == 0.
        globally_inner_ids: set[int] = set()
        for c in conjuncts:
            if c.tag == LTLfTag.GLOBALLY or (c.tag == LTLfTag.GLOBALLY_BOUND and c.lower == 0):
                globally_inner_ids.add(c.inner.formula_id)

        # Remove top, deduplicate, and apply G absorption
        seen: set[int] = set()
        filtered: list[LTLfInterned] = []
        for c in conjuncts:
            if c.tag == LTLfTag.TOP or c.formula_id in seen:
                continue
            # G(p) & p = G(p): skip p if G(p) is present
            if c.formula_id in globally_inner_ids:
                continue
            seen.add(c.formula_id)
            filtered.append(c)

        # Merge same-name bounded operators (a name identifies one source operator,
        # whose progression copies share it). A merge is sound only for special
        # interval positions; otherwise the two copies are kept separate.
        # For conjunction: F/U/weak (exists-flavored) intersect -> mergeable only
        #   when nested, to the inner window [max(lo),min(hi)]; G (all-flavored)
        #   unions -> mergeable only when contiguous, to [min(lo),max(hi)].
        finally_bounds: _BoundsTable = {}
        globally_bounds: _BoundsTable = {}
        until_bounds: _BoundsTable = {}
        weak_finally_bounds: _BoundsTable = {}
        weak_until_bounds: _BoundsTable = {}
        other: list[LTLfInterned] = []

        for c in filtered:
            match c.tag:
                case LTLfTag.FINALLY_BOUND:
                    _merge_bounded(finally_bounds, other, c, (c.inner,), intersect=True)
                case LTLfTag.GLOBALLY_BOUND:
                    _merge_bounded(globally_bounds, other, c, (c.inner,), intersect=False)
                case LTLfTag.UNTIL_BOUND:
                    _merge_bounded(until_bounds, other, c, (c.left, c.right), intersect=True)
                case LTLfTag.WEAK_FINALLY_BOUND:
                    _merge_bounded(weak_finally_bounds, other, c, (c.inner,), intersect=True)
                case LTLfTag.WEAK_UNTIL_BOUND:
                    _merge_bounded(weak_until_bounds, other, c, (c.left, c.right), intersect=True)
                case _:
                    other.append(c)

        # Rebuild normalized
        normalized: list[LTLfInterned] = []
        for name, (lower, upper, (inner,)) in finally_bounds.items():
            normalized.append(self.make_finally_bound(lower, upper, inner, name))
        for name, (lower, upper, (inner,)) in globally_bounds.items():
            normalized.append(self.make_globally_bound(lower, upper, inner, name))
        for name, (lower, upper, (left, right)) in until_bounds.items():
            normalized.append(self.make_until_bound(lower, upper, left, right, name))
        for name, (lower, upper, (inner,)) in weak_finally_bounds.items():
            normalized.append(self.make_weak_finally_bound(lower, upper, inner, name))
        for name, (lower, upper, (left, right)) in weak_until_bounds.items():
            normalized.append(self.make_weak_until_bound(lower, upper, left, right, name))
        normalized.extend(other)

        # Absorption: A & (A | B) -> A (remove disjunctions subsumed by conjuncts)
        normalized = self._apply_and_absorption(normalized)

        # Sort for canonical form
        normalized.sort(key=lambda f: f.formula_id)
        return normalized

    def _apply_and_absorption(self, conjuncts: list[LTLfInterned]) -> list[LTLfInterned]:
        """Apply absorption: if A is a conjunct and (A | B) is also present, remove (A | B)."""
        conjunct_ids = {c.formula_id for c in conjuncts}
        result: list[LTLfInterned] = []
        for c in conjuncts:
            if c.tag == LTLfTag.OR:
                # Check if any disjunct of c is also a top-level conjunct
                disjuncts = self._flatten_or(c)
                if any(d.formula_id in conjunct_ids and d is not c for d in disjuncts):
                    continue  # Absorbed: A & (A | B) -> A
            result.append(c)
        return result

    # === Disjunct normalization ===

    def _normalize_disjuncts(self, disjuncts: list[LTLfInterned]) -> list[LTLfInterned]:
        """Normalize disjuncts: dedupe, complement detection, absorption, merge bounded.

        Additional rules:
        - F(p) | p = F(p)  (finally absorbs its inner)
        - F^k(p) | p = F^k(p)  (bounded finally absorbs its inner)
        """
        # Check for top
        if any(d.tag == LTLfTag.TOP for d in disjuncts):
            return [self.make_top()]

        # Collect positive and negated formula IDs for complement detection
        positive_ids: set[int] = set()
        negated_inner_ids: set[int] = set()
        for d in disjuncts:
            if d.tag == LTLfTag.NOT:
                negated_inner_ids.add(d.inner.formula_id)
            else:
                positive_ids.add(d.formula_id)

        # Complement detection: p | ~p = top
        if positive_ids & negated_inner_ids:
            return [self.make_top()]

        # Collect inner formula IDs of F and F^k operators for absorption
        # F(p) | p = F(p), so if F(p) is present, p is redundant
        # F^[lo,hi](p) | p = F^[lo,hi](p) only when lo == 0
        # Same for weak variants
        finally_inner_ids: set[int] = set()
        for d in disjuncts:
            if d.tag == LTLfTag.FINALLY or (
                d.tag in (LTLfTag.FINALLY_BOUND, LTLfTag.WEAK_FINALLY_BOUND) and d.lower == 0
            ):
                finally_inner_ids.add(d.inner.formula_id)

        # Remove bottom, deduplicate, and apply F absorption
        seen: set[int] = set()
        filtered: list[LTLfInterned] = []
        for d in disjuncts:
            if d.tag == LTLfTag.BOTTOM or d.formula_id in seen:
                continue
            # F(p) | p = F(p): skip p if F(p) is present
            if d.formula_id in finally_inner_ids:
                continue
            seen.add(d.formula_id)
            filtered.append(d)

        # Merge same-name bounded operators (see _normalize_conjuncts). For
        # disjunction the flavors swap: F/weak (exists-flavored) union -> mergeable
        # only when contiguous, to [min(lo),max(hi)]; G (all-flavored) intersect ->
        # mergeable only when nested, to the inner window [max(lo),min(hi)].
        # Otherwise the two copies are kept separate.
        finally_bounds: _BoundsTable = {}
        globally_bounds: _BoundsTable = {}
        weak_finally_bounds: _BoundsTable = {}
        other: list[LTLfInterned] = []

        for d in filtered:
            match d.tag:
                case LTLfTag.FINALLY_BOUND:
                    _merge_bounded(finally_bounds, other, d, (d.inner,), intersect=False)
                case LTLfTag.GLOBALLY_BOUND:
                    _merge_bounded(globally_bounds, other, d, (d.inner,), intersect=True)
                case LTLfTag.WEAK_FINALLY_BOUND:
                    _merge_bounded(weak_finally_bounds, other, d, (d.inner,), intersect=False)
                case _:
                    other.append(d)

        # Rebuild
        normalized: list[LTLfInterned] = []
        for name, (lower, upper, (inner,)) in finally_bounds.items():
            normalized.append(self.make_finally_bound(lower, upper, inner, name))
        for name, (lower, upper, (inner,)) in globally_bounds.items():
            normalized.append(self.make_globally_bound(lower, upper, inner, name))
        for name, (lower, upper, (inner,)) in weak_finally_bounds.items():
            normalized.append(self.make_weak_finally_bound(lower, upper, inner, name))
        normalized.extend(other)

        # Absorption: A | (A & B) -> A (remove conjunctions subsumed by disjuncts)
        normalized = self._apply_or_absorption(normalized)

        normalized.sort(key=lambda f: f.formula_id)
        return normalized

    def _apply_or_absorption(self, disjuncts: list[LTLfInterned]) -> list[LTLfInterned]:
        """Apply absorption: if A is a disjunct and (A & B) is also present, remove (A & B)."""
        disjunct_ids = {d.formula_id for d in disjuncts}
        result: list[LTLfInterned] = []
        for d in disjuncts:
            if d.tag == LTLfTag.AND:
                # Check if any conjunct of d is also a top-level disjunct
                conjuncts = self._flatten_and(d)
                if any(c.formula_id in disjunct_ids and c is not d for c in conjuncts):
                    continue  # Absorbed: A | (A & B) -> A
            result.append(d)
        return result

    # === Tree building ===

    def _build_and_tree(self, conjuncts: list[LTLfInterned]) -> LTLfInterned:
        """Build right-associative AND tree from sorted conjuncts."""
        if len(conjuncts) == 1:
            return conjuncts[0]

        result = conjuncts[-1]
        for i in range(len(conjuncts) - 2, -1, -1):
            left = conjuncts[i]
            if left.formula_id > result.formula_id:
                left, result = result, left
            key: LTLfInternKey = (LTLfTag.AND, (left.key, result.key))
            result = self._intern(LTLfTag.AND, key, left=left, right=result)
        return result

    def _build_or_tree(self, disjuncts: list[LTLfInterned]) -> LTLfInterned:
        """Build right-associative OR tree from sorted disjuncts."""
        if len(disjuncts) == 1:
            return disjuncts[0]

        result = disjuncts[-1]
        for i in range(len(disjuncts) - 2, -1, -1):
            left = disjuncts[i]
            if left.formula_id > result.formula_id:
                left, result = result, left
            key: LTLfInternKey = (LTLfTag.OR, (left.key, result.key))
            result = self._intern(LTLfTag.OR, key, left=left, right=result)
        return result
