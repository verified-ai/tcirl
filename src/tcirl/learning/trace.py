"""Trace preprocessing for TCIRL's cause inference."""

from __future__ import annotations

from collections.abc import Sequence


def filter_symbol_from_trace(
    labels: Sequence[int],
    symbol: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Remove ALL occurrences of *symbol* from a trace.

    Returns (new_labels, keep_indices) where keep_indices are the original
    positions retained, so callers can apply the same filtering to companion
    arrays (rewards, masks, etc.).
    """
    out_labels: list[int] = []
    keep: list[int] = []
    for i, lbl in enumerate(labels):
        if lbl != symbol:
            out_labels.append(lbl)
            keep.append(i)
    return tuple(out_labels), tuple(keep)
