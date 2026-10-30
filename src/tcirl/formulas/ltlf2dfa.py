"""LTLf formula to DFA conversion using formula progression."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import os
import pickle
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Any

from tcirl.alphabet import BitmaskAlphabet
from tcirl.automata.dfa import (
    DFA,
    minimize_dfa_with_mapping,
)
from tcirl.formulas.bool_syntax import (
    BoolAnd,
    BoolAtom,
    BoolBottom,
    BoolNot,
    BoolOr,
    BoolSyntax,
    BoolTop,
)
from tcirl.formulas.encoding import DFAStateEncoder
from tcirl.formulas.ltlf_interned import LTLfInterned
from tcirl.formulas.ltlf_interner import LTLfInterner
from tcirl.formulas.ltlf_progression import Progressor
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

# Bump this to force cache invalidation during development (e.g., for breaking changes
# to pickled object structure). Combined with the package version in the cache key.
_CACHE_VERSION = 6

_PKG_NAME = "tcirl"


def _get_pkg_version() -> str:
    """Return the installed package version, or 'dev' if not installed."""
    try:
        return importlib.metadata.version(_PKG_NAME)
    except importlib.metadata.PackageNotFoundError:
        return "dev"


def formula_structural_hash[P](formula: LTLfSyntax[P]) -> str:
    """Compute a structural hash of an LTLf formula.

    Uses explicit tree traversal instead of repr() for stability across
    code changes that don't affect semantics.
    """

    def visit(f: LTLfSyntax[P]) -> str:
        match f:
            case Atom(prop=p):
                return f"Atom({p!r})"
            case Top():
                return "Top"
            case Bottom():
                return "Bottom"
            case Not(inner=inner):
                return f"Not({visit(inner)})"
            case And(left=left, right=right):
                return f"And({visit(left)},{visit(right)})"
            case Or(left=left, right=right):
                return f"Or({visit(left)},{visit(right)})"
            case Implies(antecedent=ant, consequent=cons):
                return f"Implies({visit(ant)},{visit(cons)})"
            case Next(inner=inner):
                return f"Next({visit(inner)})"
            case Globally(inner=inner):
                return f"Globally({visit(inner)})"
            case Finally(inner=inner):
                return f"Finally({visit(inner)})"
            case Until(left=left, right=right):
                return f"Until({visit(left)},{visit(right)})"
            case Release(left=left, right=right):
                return f"Release({visit(left)},{visit(right)})"
            case GloballyBound(lower=lower, upper=upper, inner=inner, name=name):
                return f"GloballyBound({lower},{upper},{visit(inner)},{name!r})"
            case FinallyBound(lower=lower, upper=upper, inner=inner, name=name):
                return f"FinallyBound({lower},{upper},{visit(inner)},{name!r})"
            case UntilBound(lower=lower, upper=upper, left=left, right=right, name=name):
                return f"UntilBound({lower},{upper},{visit(left)},{visit(right)},{name!r})"
            case WeakFinallyBound(lower=lower, upper=upper, inner=inner, name=name):
                return f"WeakFinallyBound({lower},{upper},{visit(inner)},{name!r})"
            case WeakUntilBound(lower=lower, upper=upper, left=left, right=right, name=name):
                return f"WeakUntilBound({lower},{upper},{visit(left)},{visit(right)},{name!r})"

    return visit(formula)


def bool_structural_hash[P](formula: BoolSyntax[P] | None) -> str:
    """Compute a structural hash of a boolean filter formula."""
    if formula is None:
        return "None"

    def visit(f: BoolSyntax[P]) -> str:
        match f:
            case BoolAtom(prop=p):
                return f"BoolAtom({p!r})"
            case BoolTop():
                return "BoolTop"
            case BoolBottom():
                return "BoolBottom"
            case BoolNot(inner=inner):
                return f"BoolNot({visit(inner)})"
            case BoolAnd(left=left, right=right):
                return f"BoolAnd({visit(left)},{visit(right)})"
            case BoolOr(left=left, right=right):
                return f"BoolOr({visit(left)},{visit(right)})"

    return visit(formula)


def report_perf(perf: dict[str, float]) -> str:
    """Format performance metrics as a readable string."""
    if perf.get("cached"):
        lines = ["=== Performance (cached) ==="]
        lines.append(
            f"States: {int(perf['states_after_minimization'])} "
            f"(before minimization: {int(perf['states_before_minimization'])})"
        )
        return "\n".join(lines)

    lines = ["=== Performance ==="]
    lines.append(f"Compile:   {perf['compile_s'] * 1000:.2f}ms")
    lines.append(
        f"Build DFA: {perf['build_dfa_s'] * 1000:.2f}ms "
        f"({int(perf['states_before_minimization'])} states)"
    )
    lines.append(
        f"Minimize:  {perf['minimize_dfa_s'] * 1000:.2f}ms "
        f"({int(perf['states_after_minimization'])} states)"
    )
    lines.append(f"Encoder:   {perf['build_encoder_s'] * 1000:.2f}ms")
    lines.append(f"Total:     {perf['total_s'] * 1000:.2f}ms")
    return "\n".join(lines)


# === Core DFA Construction ===
def build_dfa_core[P](
    formula: LTLfInterned,
    alphabet: BitmaskAlphabet[P],
    interner: LTLfInterner[P],
) -> tuple[DFA, list[LTLfInterned]]:
    """Build DFA using interned formulas and progression."""
    is_partial = alphabet.is_filtered
    symbols_to_explore = alphabet.valid_symbols
    progressor = Progressor(interner)

    initial = formula
    state_to_id: dict[int, int] = {initial.formula_id: 0}
    id_to_formula: list[LTLfInterned] = [initial]
    transitions: dict[tuple[int, int], int] = {}

    queue: deque[LTLfInterned] = deque([initial])

    while queue:
        current = queue.popleft()
        current_state_id = state_to_id[current.formula_id]

        for symbol in symbols_to_explore:
            label = alphabet.decode_to_ints(symbol)
            next_formula = progressor.progress(current, label)

            if next_formula.formula_id not in state_to_id:
                next_state_id = len(id_to_formula)
                state_to_id[next_formula.formula_id] = next_state_id
                id_to_formula.append(next_formula)
                queue.append(next_formula)

            transitions[(current_state_id, symbol)] = state_to_id[next_formula.formula_id]

    # Accepting states are those where the formula is satisfied if the trace ends
    accepting_states = frozenset(
        state_id
        for state_id in state_to_id.values()
        if interner.accepts_at_end(id_to_formula[state_id])
    )

    dfa = DFA(
        num_states=len(id_to_formula),
        num_symbols=alphabet.num_symbols,
        accepting_states=accepting_states,
        transitions=transitions,
        valid_symbols=frozenset(symbols_to_explore) if is_partial else None,
    )

    return dfa, id_to_formula


# === Caching ===
def _compute_cache_key[P](
    formula: LTLfSyntax[P],
    label_filter: BoolSyntax[P] | None,
    minimize: bool,
    build_encoding: bool,
    extra_props: tuple[P, ...] | None = None,
) -> str:
    """Compute a hash key for caching based on formula structure and package version.

    Uses structural hashes (not repr) for stability. Cache is invalidated by
    package version changes or _CACHE_VERSION bumps.
    """
    pkg_version = _get_pkg_version()
    formula_hash = formula_structural_hash(formula)
    filter_hash = bool_structural_hash(label_filter)
    key_data = (
        _CACHE_VERSION,
        pkg_version,
        formula_hash,
        filter_hash,
        minimize,
        build_encoding,
        extra_props,
    )
    key_bytes = str(key_data).encode("utf-8")
    return hashlib.sha256(key_bytes).hexdigest()[:16]


class StaleCacheError(Exception):
    """Raised when cached data structure doesn't match current code."""


def _try_load_from_cache(
    cache_path: Path,
) -> tuple[DFA, BitmaskAlphabet[Any], DFAStateEncoder | None] | None:
    """Attempt to load cached DFA result, returning None if not found or invalid."""
    if not cache_path.exists():
        return None

    with cache_path.open("rb") as f:
        try:
            result: tuple[DFA, BitmaskAlphabet[Any], DFAStateEncoder | None] = pickle.load(f)  # noqa: S301
            _dfa, _alphabet, encoder = result
            # Quick sanity check - encoder should have _timer_groups attribute
            if encoder is not None and not hasattr(encoder, "_timer_groups"):
                raise StaleCacheError(
                    f"Stale cache detected: encoder missing _timer_groups. "
                    f"Delete cache at {cache_path.parent}"
                )
        except (AttributeError, ImportError, KeyError, TypeError, ValueError) as e:
            raise StaleCacheError(
                f"Stale cache detected: {e}. Delete cache at {cache_path.parent}"
            ) from e
        else:
            return result


def _save_to_cache(
    cache_path: Path,
    dfa: DFA,
    alphabet: BitmaskAlphabet[Any],
    encoder: DFAStateEncoder | None,
) -> None:
    """Save DFA result to cache using atomic write.

    Writes to a temp file first, then renames. Rename is atomic on POSIX,
    preventing corruption if multiple processes write simultaneously.
    """
    cache_dir = cache_path.parent
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Write to temp file in same directory (ensures same filesystem for atomic rename)
    fd, tmp_path = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump((dfa, alphabet, encoder), f)
        # Atomic rename (on POSIX; on Windows this may fail if target exists)
        os.replace(tmp_path, cache_path)
    except BaseException:
        # Clean up temp file on any error
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


# === Public API ===
def ltlf_to_dfa[P](
    formula: LTLfSyntax[P],
    label_filter: BoolSyntax[P] | None = None,
    minimize: bool = True,
    build_encoding: bool = False,
    recreate: bool = False,
    extra_props: tuple[P, ...] | None = None,
    cache_dir: Path | None = None,
) -> tuple[DFA, BitmaskAlphabet[P], DFAStateEncoder | None, dict[str, float]]:
    """Convert an LTLf formula to an explicit DFA with optional state encoder.

    Args:
        formula: The LTLf formula to convert.
        label_filter: Optional boolean filter restricting the alphabet.
        minimize: Whether to minimize the DFA via partition refinement.
        build_encoding: Whether to build a DFAStateEncoder for RL observations.
        recreate: If True and cache_dir is set, ignore existing cache and rebuild.
        extra_props: Additional propositions to include in the alphabet.
        cache_dir: Directory for caching results. If None (default), no caching
            is performed and no disk I/O occurs. When set, results are persisted
            as pickle files keyed by formula structure, options, and package version.

    Returns:
        A tuple of (DFA, BitmaskAlphabet, optional DFAStateEncoder, performance metrics dict).
    """
    # Resolve cache path up front (None when caching disabled)
    cache_path: Path | None = None
    if cache_dir is not None:
        dfa_cache_dir = cache_dir / "dfa"
        cache_key = _compute_cache_key(formula, label_filter, minimize, build_encoding, extra_props)
        cache_path = dfa_cache_dir / f"{cache_key}.pkl"

        # Try loading from cache (unless recreate=True)
        if not recreate:
            cached = _try_load_from_cache(cache_path)
            if cached is not None:
                cached_dfa, cached_alphabet, encoder = cached
                perf: dict[str, float] = {
                    "cached": 1.0,
                    "compile_s": 0.0,
                    "build_dfa_s": 0.0,
                    "minimize_dfa_s": 0.0,
                    "build_encoder_s": 0.0,
                    "total_s": 0.0,
                    "states_before_minimization": float(cached_dfa.num_states),
                    "states_after_minimization": float(cached_dfa.num_states),
                }
                return cached_dfa, cached_alphabet, encoder, perf

    # Build fresh
    perf = {}

    t0 = time.perf_counter()
    alphabet = BitmaskAlphabet.from_syntax(formula, label_filter, extra_props)
    interner = LTLfInterner(alphabet.prop_to_index)
    compiled = interner.compile(formula)
    t1 = time.perf_counter()
    perf["compile_s"] = t1 - t0

    t2 = time.perf_counter()
    dfa, id_to_formula = build_dfa_core(compiled, alphabet, interner)
    t3 = time.perf_counter()
    perf["build_dfa_s"] = t3 - t2
    perf["states_before_minimization"] = float(dfa.num_states)

    if minimize:
        t4 = time.perf_counter()
        dfa, old_to_new = minimize_dfa_with_mapping(dfa)
        t5 = time.perf_counter()
        perf["minimize_dfa_s"] = t5 - t4

        new_id_to_formula: dict[int, LTLfInterned] = {}
        for old_id, new_id in old_to_new.items():
            if new_id not in new_id_to_formula:
                new_id_to_formula[new_id] = id_to_formula[old_id]
        state_to_formula = new_id_to_formula
    else:
        perf["minimize_dfa_s"] = 0.0
        state_to_formula = dict(enumerate(id_to_formula))

    perf["states_after_minimization"] = float(dfa.num_states)

    encoder: DFAStateEncoder | None = None
    if build_encoding:
        t6 = time.perf_counter()
        encoder = DFAStateEncoder.build(state_to_formula, interner.timers)
        t7 = time.perf_counter()
        perf["build_encoder_s"] = t7 - t6
        perf["total_s"] = t7 - t0
    else:
        perf["build_encoder_s"] = 0.0
        perf["total_s"] = time.perf_counter() - t0

    # Cache the result (only when cache_dir was provided)
    if cache_path is not None:
        _save_to_cache(cache_path, dfa, alphabet, encoder)

    return dfa, alphabet, encoder, perf
