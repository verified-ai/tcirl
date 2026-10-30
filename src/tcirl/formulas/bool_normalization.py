"""Boolean formula normalization rules.

This mixin provides the flattening, deduplication, and canonical-ordering steps
of boolean normalization; together with `BoolInterner` (which handles negation,
including double negation elimination) the full pipeline ensures semantically
equivalent formulas produce identical interned representations.

Normalization rules applied across the pipeline:
- Double negation elimination: ~~p = p (in `BoolInterner.make_not`)
- Constants propagation: p & T = p, p | F = p, etc.
- AND/OR flattening and associativity
- Commutativity via canonical ordering (sorted by formula_id)
- Deduplication: a & a = a
"""

from __future__ import annotations

from tcirl.formulas.bool_interned import BoolInterned, BoolInternKey, BoolTag


class BoolNormalizationMixin:
    """Mixin providing normalization methods for boolean formula interning."""

    # These will be provided by the interner class
    def _intern(
        self,
        tag: BoolTag,
        key: BoolInternKey,
        *,
        prop: int = -1,
        inner: BoolInterned | None = None,
        left: BoolInterned | None = None,
        right: BoolInterned | None = None,
    ) -> BoolInterned: ...

    def make_top(self) -> BoolInterned: ...
    def make_bottom(self) -> BoolInterned: ...

    # === Flattening ===

    def _flatten_and(self, formula: BoolInterned) -> list[BoolInterned]:
        """Flatten nested AND into a list of conjuncts."""
        if formula.tag == BoolTag.AND:
            return self._flatten_and(formula.left) + self._flatten_and(formula.right)
        return [formula]

    def _flatten_or(self, formula: BoolInterned) -> list[BoolInterned]:
        """Flatten nested OR into a list of disjuncts."""
        if formula.tag == BoolTag.OR:
            return self._flatten_or(formula.left) + self._flatten_or(formula.right)
        return [formula]

    # === Deduplication and filtering ===

    def _dedupe_and_filter(
        self,
        formulas: list[BoolInterned],
        identity_tag: BoolTag,
        absorbing_tag: BoolTag,
    ) -> list[BoolInterned] | None:
        """Remove duplicates and identity elements, check for absorbing element.

        Returns None if an absorbing element is found.
        """
        seen: set[int] = set()
        result: list[BoolInterned] = []
        for f in formulas:
            if f.tag == absorbing_tag:
                return None
            if f.formula_id in seen:
                continue
            seen.add(f.formula_id)
            if f.tag == identity_tag:
                continue
            result.append(f)
        return result

    # === Tree building ===

    def _build_and_tree(self, conjuncts: list[BoolInterned]) -> BoolInterned:
        """Build right-associative AND tree from sorted conjuncts."""
        if len(conjuncts) == 1:
            return conjuncts[0]

        # Build right-associatively
        result = conjuncts[-1]
        for i in range(len(conjuncts) - 2, -1, -1):
            left = conjuncts[i]
            # Canonical ordering by formula_id
            if left.formula_id > result.formula_id:
                left, result = result, left
            key: BoolInternKey = (BoolTag.AND, (left.key, result.key))
            result = self._intern(BoolTag.AND, key, left=left, right=result)
        return result

    def _build_or_tree(self, disjuncts: list[BoolInterned]) -> BoolInterned:
        """Build right-associative OR tree from sorted disjuncts."""
        if len(disjuncts) == 1:
            return disjuncts[0]

        # Build right-associatively
        result = disjuncts[-1]
        for i in range(len(disjuncts) - 2, -1, -1):
            left = disjuncts[i]
            # Canonical ordering by formula_id
            if left.formula_id > result.formula_id:
                left, result = result, left
            key: BoolInternKey = (BoolTag.OR, (left.key, result.key))
            result = self._intern(BoolTag.OR, key, left=left, right=result)
        return result
