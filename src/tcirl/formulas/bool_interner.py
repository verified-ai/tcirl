"""Boolean formula interner - compiles syntax to interned formulas with normalization.

This module provides the BoolInterner class which:
1. Interns formulas (deduplicates semantically equivalent formulas)
2. Normalizes formulas to canonical forms (via BoolNormalizationMixin)
3. Compiles boolean syntax AST to interned representations
"""

from __future__ import annotations

from typing import override

from tcirl.formulas.bool_interned import (
    BoolInterned,
    BoolInternKey,
    BoolTag,
)
from tcirl.formulas.bool_normalization import BoolNormalizationMixin
from tcirl.formulas.bool_syntax import (
    BoolAnd,
    BoolAtom,
    BoolBottom,
    BoolNot,
    BoolOr,
    BoolSyntax,
    BoolTop,
)


class BoolInterner[P](BoolNormalizationMixin):
    """Interns boolean formulas with normalization for canonical representations."""

    def __init__(self, prop_to_index: dict[P, int] | None = None) -> None:
        self._intern_table: dict[BoolInternKey, BoolInterned] = {}
        self._id_counter: int = 0
        self._prop_to_index: dict[P, int] = prop_to_index if prop_to_index is not None else {}
        self._index_to_prop: list[P] = (
            sorted(prop_to_index.keys(), key=lambda p: prop_to_index[p]) if prop_to_index else []
        )
        # Cache singletons
        self._top: BoolInterned | None = None
        self._bottom: BoolInterned | None = None

    def _next_id(self) -> int:
        self._id_counter += 1
        return self._id_counter

    def _get_prop_index(self, prop: P) -> int:
        if prop not in self._prop_to_index:
            idx = len(self._prop_to_index)
            self._prop_to_index[prop] = idx
            self._index_to_prop.append(prop)
        return self._prop_to_index[prop]

    @property
    def num_props(self) -> int:
        return len(self._prop_to_index)

    @property
    def prop_to_index(self) -> dict[P, int]:
        return self._prop_to_index

    @property
    def index_to_prop(self) -> list[P]:
        return self._index_to_prop

    # === Interning primitives ===

    @override
    def _intern(
        self,
        tag: BoolTag,
        key: BoolInternKey,
        *,
        prop: int = -1,
        inner: BoolInterned | None = None,
        left: BoolInterned | None = None,
        right: BoolInterned | None = None,
    ) -> BoolInterned:
        if key in self._intern_table:
            return self._intern_table[key]
        formula = BoolInterned(
            self._next_id(), tag, key, prop=prop, inner=inner, left=left, right=right
        )
        self._intern_table[key] = formula
        return formula

    @override
    def make_top(self) -> BoolInterned:
        if self._top is None:
            key: BoolInternKey = (BoolTag.TOP, ())
            self._top = self._intern(BoolTag.TOP, key)
        return self._top

    @override
    def make_bottom(self) -> BoolInterned:
        if self._bottom is None:
            key: BoolInternKey = (BoolTag.BOTTOM, ())
            self._bottom = self._intern(BoolTag.BOTTOM, key)
        return self._bottom

    def make_atom(self, prop_idx: int) -> BoolInterned:
        key: BoolInternKey = (BoolTag.ATOM, (prop_idx,))
        return self._intern(BoolTag.ATOM, key, prop=prop_idx)

    def make_not(self, inner: BoolInterned) -> BoolInterned:
        # Normalization
        match inner.tag:
            case BoolTag.TOP:
                return self.make_bottom()
            case BoolTag.BOTTOM:
                return self.make_top()
            case BoolTag.NOT:
                return inner.inner  # Double negation elimination
            case BoolTag.ATOM | BoolTag.AND | BoolTag.OR:
                pass  # Proceed to create NOT

        key: BoolInternKey = (BoolTag.NOT, (inner.key,))
        return self._intern(BoolTag.NOT, key, inner=inner)

    def make_and(self, left: BoolInterned, right: BoolInterned) -> BoolInterned:
        # Quick normalization paths
        if left.tag == BoolTag.BOTTOM or right.tag == BoolTag.BOTTOM:
            return self.make_bottom()
        if left.tag == BoolTag.TOP:
            return right
        if right.tag == BoolTag.TOP:
            return left
        if left is right:
            return left

        # Flatten, deduplicate, sort for canonical form (from mixin)
        flat = self._flatten_and(left) + self._flatten_and(right)
        conjuncts = self._dedupe_and_filter(flat, BoolTag.TOP, BoolTag.BOTTOM)

        if conjuncts is None:
            return self.make_bottom()
        if not conjuncts:
            return self.make_top()

        # Sort by formula_id for canonical ordering
        conjuncts.sort(key=lambda f: f.formula_id)

        return self._build_and_tree(conjuncts)

    def make_or(self, left: BoolInterned, right: BoolInterned) -> BoolInterned:
        # Quick normalization paths
        if left.tag == BoolTag.TOP or right.tag == BoolTag.TOP:
            return self.make_top()
        if left.tag == BoolTag.BOTTOM:
            return right
        if right.tag == BoolTag.BOTTOM:
            return left
        if left is right:
            return left

        # Flatten, deduplicate, sort for canonical form (from mixin)
        flat = self._flatten_or(left) + self._flatten_or(right)
        disjuncts = self._dedupe_and_filter(flat, BoolTag.BOTTOM, BoolTag.TOP)

        if disjuncts is None:
            return self.make_top()
        if not disjuncts:
            return self.make_bottom()

        # Sort by formula_id for canonical ordering
        disjuncts.sort(key=lambda f: f.formula_id)

        return self._build_or_tree(disjuncts)

    # === Compilation ===

    def compile(self, syntax: BoolSyntax[P]) -> BoolInterned:
        """Compile syntax formula to interned formula."""
        match syntax:
            case BoolTop():
                return self.make_top()
            case BoolBottom():
                return self.make_bottom()
            case BoolAtom(prop=p):
                idx = self._get_prop_index(p)
                return self.make_atom(idx)
            case BoolNot(inner=inner):
                return self.make_not(self.compile(inner))
            case BoolAnd(left=left, right=right):
                return self.make_and(self.compile(left), self.compile(right))
            case BoolOr(left=left, right=right):
                return self.make_or(self.compile(left), self.compile(right))

    # === Evaluation ===

    def check(self, formula: BoolInterned, label: frozenset[int]) -> bool:
        """Evaluate formula against a label (set of true prop indices)."""
        match formula.tag:
            case BoolTag.TOP:
                return True
            case BoolTag.BOTTOM:
                return False
            case BoolTag.ATOM:
                return formula.prop in label
            case BoolTag.NOT:
                return not self.check(formula.inner, label)
            case BoolTag.AND:
                return self.check(formula.left, label) and self.check(formula.right, label)
            case BoolTag.OR:
                return self.check(formula.left, label) or self.check(formula.right, label)

    def check_with_props(self, formula: BoolInterned, label: frozenset[P]) -> bool:
        """Evaluate formula against a label with original prop types."""
        int_label = frozenset(self._prop_to_index[p] for p in label if p in self._prop_to_index)
        return self.check(formula, int_label)

    # === Utility ===

    @property
    def stats(self) -> dict[str, int]:
        return {
            "interned_formulas": len(self._intern_table),
            "propositions": len(self._prop_to_index),
            "next_id": self._id_counter,
        }
