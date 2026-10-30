from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Self, override

import numpy as np
from numpy.typing import NDArray


class UndefinedTransitionError(Exception):
    """Raised when stepping on an undefined transition in a partial DFA."""

    def __init__(self, state: int, symbol: int) -> None:
        self.state = state
        self.symbol = symbol
        super().__init__(f"Undefined transition from state {state} on symbol {symbol}")


@dataclass(frozen=True, slots=True, kw_only=True)
class DFA:
    """Deterministic Finite Automaton with states `0..num_states-1`, input alphabet `0..num_symbols-1`, and initial state 0.

    Termination follows the "absence of a run" convention: a state is terminal
    iff it has no outgoing transitions in `transitions` (for every symbol in
    the DFA's valid alphabet). A total DFA has no terminal states; a partial
    DFA's terminal states are those with no outgoing transition.

    `valid_symbols` restricts which symbols count as in the input alphabet.
    Per-state coverage is all-or-none within scope; symbols outside scope are
    not in play.
    """

    num_states: int
    num_symbols: int
    accepting_states: frozenset[int]
    transitions: dict[tuple[int, int], int]  # (state, symbol) -> next_state
    # Alphabet subset in scope. `None` means the full alphabet `0..num_symbols-1`.
    # A state with no outgoing transition over the in-scope alphabet is a
    # rejecting terminal state; the `partial` property reports whether any exist.
    valid_symbols: frozenset[int] | None

    def __post_init__(self) -> None:
        if self.num_states < 1:
            raise ValueError("num_states must be >= 1")
        if self.num_symbols < 1:
            raise ValueError("num_symbols must be >= 1")
        if not self.accepting_states <= set(range(self.num_states)):
            raise ValueError("accepting_states contains invalid state")

        scope: frozenset[int] = (
            self.valid_symbols
            if self.valid_symbols is not None
            else frozenset(range(self.num_symbols))
        )
        # All transition keys must lie within (states x scope).
        for key in self.transitions:
            if not (0 <= key[0] < self.num_states):
                raise ValueError(f"transition key {key} references invalid state")
            if key[1] not in scope:
                raise ValueError(
                    f"transition key {key} uses symbol outside the DFA's scope {scope}"
                )

        # Per-state all-or-none over the in-scope alphabet: a state is either
        # fully defined or fully terminal (no outgoing). Mixed coverage matches
        # no automaton class here. Whether an environment DFA may use rejecting
        # termination is governed by the termination-convention
        # flag (see `assert_termination_convention`), not here.
        n_scope = len(scope)
        for s in range(self.num_states):
            covered = sum(1 for a in scope if (s, a) in self.transitions)
            if covered != 0 and covered != n_scope:
                raise ValueError(
                    f"state {s} has {covered}/{n_scope} outgoing transitions; "
                    f"a state must be either fully defined or fully terminal "
                    f"(no outgoing)"
                )

        if not all(0 <= t < self.num_states for t in self.transitions.values()):
            raise ValueError("transition targets must be valid states")

    @property
    def partial(self) -> bool:
        """True iff at least one state has no outgoing transition over the in-scope alphabet (a rejecting terminal state)."""
        scope: frozenset[int] | range = (
            self.valid_symbols if self.valid_symbols is not None else range(self.num_symbols)
        )
        n_scope = len(scope) if isinstance(scope, frozenset) else scope.stop - scope.start
        expected_total = self.num_states * n_scope
        return len(self.transitions) < expected_total

    def is_terminal(self, state: int) -> bool:
        """True iff `state` has no outgoing transition over any in-scope symbol (absence-of-run)."""
        if not 0 <= state < self.num_states:
            raise ValueError(f"state {state} out of range [0, {self.num_states})")
        scope: frozenset[int] | range = (
            self.valid_symbols if self.valid_symbols is not None else range(self.num_symbols)
        )
        return all((state, a) not in self.transitions for a in scope)

    @property
    def all_states(self) -> frozenset[int]:
        return frozenset(range(self.num_states))

    def reset(self) -> int:
        return 0

    def step(self, state: int, input: int) -> int:
        """Transition from `state` on `input`, raising `UndefinedTransitionError` on an absence-of-run (no transition defined)."""
        key = (state, input)
        if key not in self.transitions:
            raise UndefinedTransitionError(state, input)
        return self.transitions[key]

    def accepts(self, input_seq: Iterable[int], print_runs: bool = False) -> bool:
        state = 0
        if print_runs:
            print(f"DFA-INPUT: {input_seq!r}")
            print(f"  state={state}")
        for symbol in input_seq:
            state = self.step(state, symbol)
            if print_runs:
                print(f"  --[{symbol}]--> state={state}")
        accepted = state in self.accepting_states
        if print_runs:
            print(f"  accepted={accepted}")
        return accepted

    @classmethod
    def trivial_hypothesis(cls, num_symbols: int) -> Self:
        """Create a trivial single-state DFA that rejects all words."""
        transitions = {(0, a): 0 for a in range(num_symbols)}
        return cls(
            num_states=1,
            num_symbols=num_symbols,
            accepting_states=frozenset(),
            transitions=transitions,
            valid_symbols=None,
        )

    @override
    def __str__(self) -> str:
        lines = [
            "DFA(",
            f"    num_states={self.num_states},",
            f"    num_symbols={self.num_symbols},",
            f"    accepting_states={self.accepting_states!r},",
            f"    transitions={self.transitions!r},",
            f"    valid_symbols={self.valid_symbols!r},",
            ")",
        ]
        return "\n".join(lines)


def _scope_symbols(dfa: DFA) -> frozenset[int] | range:
    """Symbols in the DFA's input alphabet: `valid_symbols` if set, else `0..num_symbols-1`."""
    return dfa.valid_symbols if dfa.valid_symbols is not None else range(dfa.num_symbols)


def _find_reachable_states(dfa: DFA) -> set[int]:
    """BFS from the initial state to find all reachable states.

    Undefined transitions in partial DFAs are skipped: no run continues, so no
    successor.
    """
    reachable = {0}
    queue: deque[int] = deque([0])
    symbols = _scope_symbols(dfa)

    while queue:
        state = queue.popleft()
        for symbol in symbols:
            if (state, symbol) not in dfa.transitions:
                continue
            next_state = dfa.transitions[(state, symbol)]
            if next_state not in reachable:
                reachable.add(next_state)
                queue.append(next_state)

    return reachable


def _hopcroft_minimize(
    dfa: DFA,
    reachable: set[int],
) -> tuple[dict[int, int], int]:
    """Hopcroft's algorithm returning (state -> partition_id, num_partitions).

    Assumes `dfa` is total over `_scope_symbols(dfa)`; the caller must make
    partial DFAs total via an internal sink first (see
    `compute_equivalent_states`).
    """
    symbols: tuple[int, ...] = tuple(_scope_symbols(dfa))

    # Inverse transitions: (target, symbol) -> set of source states.
    inverse: dict[tuple[int, int], set[int]] = {}
    for state in reachable:
        for symbol in symbols:
            target = dfa.transitions[(state, symbol)]
            if target in reachable:
                key = (target, symbol)
                if key not in inverse:
                    inverse[key] = set()
                inverse[key].add(state)

    # Initial partition: {accepting, non-accepting} intersected with reachable.
    accepting = frozenset(dfa.accepting_states & reachable)
    non_accepting = frozenset(reachable - accepting)

    # partition_id -> set of states in that partition
    partitions: dict[int, set[int]] = {}
    # state -> partition_id
    state_to_partition: dict[int, int] = {}
    next_partition_id = 0

    if accepting:
        partitions[next_partition_id] = set(accepting)
        for s in accepting:
            state_to_partition[s] = next_partition_id
        next_partition_id += 1
    if non_accepting:
        partitions[next_partition_id] = set(non_accepting)
        for s in non_accepting:
            state_to_partition[s] = next_partition_id
        next_partition_id += 1

    if not partitions:
        return {}, 0

    # Worklist of (partition_id, symbol) splitters; `in_worklist` mirrors it
    # for O(1) membership.
    worklist: deque[tuple[int, int]] = deque()
    in_worklist: set[tuple[int, int]] = set()

    # Seed with the smaller initial partition for every symbol.
    if len(partitions) == 2:
        p0, p1 = list(partitions.keys())
        smaller = p0 if len(partitions[p0]) <= len(partitions[p1]) else p1
        for sym in symbols:
            worklist.append((smaller, sym))
            in_worklist.add((smaller, sym))
    elif len(partitions) == 1:
        p0 = next(iter(partitions.keys()))
        for sym in symbols:
            worklist.append((p0, sym))
            in_worklist.add((p0, sym))

    while worklist:
        splitter_id, symbol = worklist.popleft()
        in_worklist.discard((splitter_id, symbol))

        if splitter_id not in partitions:
            continue

        splitter = partitions[splitter_id]

        predecessors: set[int] = set()
        for target_state in splitter:
            predecessors |= inverse.get((target_state, symbol), set())

        if not predecessors:
            continue

        # Partitions straddling the predecessor set must split.
        partitions_to_split: list[int] = []
        for pid in list(partitions.keys()):
            part = partitions[pid]
            if part & predecessors and part - predecessors:
                partitions_to_split.append(pid)

        for pid in partitions_to_split:
            part = partitions[pid]
            intersection = part & predecessors
            difference = part - predecessors

            new_pid = next_partition_id
            next_partition_id += 1

            # Intersection stays in pid; difference moves to new_pid.
            partitions[pid] = intersection
            for s in intersection:
                state_to_partition[s] = pid

            partitions[new_pid] = difference
            for s in difference:
                state_to_partition[s] = new_pid

            # Hopcroft's rule: if (pid, sym) is already a pending splitter, both
            # halves must be too; otherwise enqueueing the smaller half suffices.
            for sym in symbols:
                if (pid, sym) in in_worklist:
                    if (new_pid, sym) not in in_worklist:
                        worklist.append((new_pid, sym))
                        in_worklist.add((new_pid, sym))
                else:
                    smaller_pid = pid if len(intersection) <= len(difference) else new_pid
                    if (smaller_pid, sym) not in in_worklist:
                        worklist.append((smaller_pid, sym))
                        in_worklist.add((smaller_pid, sym))

    # Renumber partitions contiguously 0..n-1.
    old_to_new = {pid: i for i, pid in enumerate(partitions.keys())}

    final_state_to_partition = {s: old_to_new[p] for s, p in state_to_partition.items()}
    return final_state_to_partition, len(partitions)


def _build_minimized_dfa(
    dfa: DFA,
    state_to_partition: dict[int, int],
    num_partitions: int,
) -> tuple[DFA, dict[int, int]]:
    """Build the minimized DFA, returning it with the old_state -> new_state mapping."""
    # Renumber partitions so the one containing state 0 becomes state 0.
    initial_partition = state_to_partition[0]
    partition_remap: dict[int, int] = {initial_partition: 0}
    next_id = 1
    for p in range(num_partitions):
        if p not in partition_remap:
            partition_remap[p] = next_id
            next_id += 1

    final_state_map = {s: partition_remap[p] for s, p in state_to_partition.items()}

    # One representative per equivalence class suffices for transitions.
    new_transitions: dict[tuple[int, int], int] = {}
    for (old_state, symbol), old_target in dfa.transitions.items():
        if old_state not in final_state_map:
            continue  # unreachable source
        if old_target not in final_state_map:
            continue  # unreachable target
        new_state = final_state_map[old_state]
        new_target = final_state_map[old_target]
        key = (new_state, symbol)
        if key not in new_transitions:
            new_transitions[key] = new_target

    new_accepting = frozenset(
        final_state_map[s] for s in dfa.accepting_states if s in final_state_map
    )

    minimized = DFA(
        num_states=num_partitions,
        num_symbols=dfa.num_symbols,
        accepting_states=new_accepting,
        transitions=new_transitions,
        valid_symbols=dfa.valid_symbols,
    )
    return minimized, final_state_map


def minimize_dfa_with_mapping(dfa: DFA) -> tuple[DFA, dict[int, int]]:
    """Minimize a DFA using Hopcroft's algorithm, returning it with the old_state -> new_state mapping (reachable states only)."""
    reachable = _find_reachable_states(dfa)

    if len(reachable) == 1:
        if dfa.num_states == 1:
            return dfa, {0: 0}
        # Rebuild over the single reachable state.
        symbols = _scope_symbols(dfa)
        new_transitions = {(0, s): 0 for s in symbols}
        new_accepting: frozenset[int] = frozenset({0}) if 0 in dfa.accepting_states else frozenset()
        minimized = DFA(
            num_states=1,
            num_symbols=dfa.num_symbols,
            accepting_states=new_accepting,
            transitions=new_transitions,
            valid_symbols=dfa.valid_symbols,
        )
        return minimized, dict.fromkeys(reachable, 0)

    state_to_partition, num_partitions = _hopcroft_minimize(dfa, reachable)

    # Already minimal: nothing to merge or prune.
    if num_partitions == len(reachable) == dfa.num_states:
        return dfa, {s: s for s in range(dfa.num_states)}

    return _build_minimized_dfa(dfa, state_to_partition, num_partitions)


def build_transition_table(dfa: DFA) -> NDArray[np.int64]:
    """Build a (num_states, num_symbols) numpy transition table; undefined (partial) transitions are -1."""
    table = np.full((dfa.num_states, dfa.num_symbols), -1, dtype=np.int64)
    for (s, a), t in dfa.transitions.items():
        table[s, a] = t
    return table


def compute_live_states(dfa: DFA) -> frozenset[int]:
    """Backward BFS: find all states from which an accepting state is reachable."""
    accepting = dfa.accepting_states
    live: set[int] = set(accepting)
    frontier = list(accepting)
    reverse: dict[int, set[int]] = {s: set() for s in range(dfa.num_states)}
    for (src, _sym), dst in dfa.transitions.items():
        reverse.setdefault(dst, set()).add(src)
    while frontier:
        state = frontier.pop()
        for pred in reverse.get(state, set()):
            if pred not in live:
                live.add(pred)
                frontier.append(pred)
    return frozenset(live)


def compute_absorbing_states(dfa: DFA) -> frozenset[int]:
    """Return all states whose outgoing transitions are all self-loops."""
    symbols: frozenset[int] | range = _scope_symbols(dfa)
    absorbing: set[int] = set()
    for s in range(dfa.num_states):
        if all(dfa.transitions[(s, a)] == s for a in symbols):
            absorbing.add(s)
    return frozenset(absorbing)


def compute_equivalent_states(
    dfa_a: DFA,
    dfa_b: DFA,
    *,
    restrict_to_reachable: bool = False,
) -> dict[int, int]:
    """Find equivalent states between two DFAs over the same alphabet via Hopcroft refinement on their disjoint union.

    Two states are equivalent iff they accept the same language. Partial DFAs
    are supported: undefined transitions (absence of a run) route to an
    algorithm-internal sink, so two states with matching "no run on symbol a"
    behaviour land in the same partition. The sink never appears in the result.

    With `restrict_to_reachable=False` (default), partitioning runs over all
    states (strict language equivalence). With `True`, only over states
    BFS-reachable from state 0 in each DFA, relaxing equivalence to "agrees on
    all reachable extensions".

    Returns a mapping dfa_b state -> dfa_a state for each dfa_b state with an
    equivalent in dfa_a; unmatched states are omitted.
    """
    if dfa_a.num_symbols != dfa_b.num_symbols:
        raise ValueError(f"Alphabet size mismatch: {dfa_a.num_symbols} vs {dfa_b.num_symbols}")

    num_symbols = dfa_a.num_symbols

    # Combined state space:
    #   [0, dfa_a.num_states)                       -> dfa_a real states
    #   [dfa_a.num_states, +dfa_b.num_states)       -> dfa_b real states
    #   total_states - 1                            -> shared sink
    # One shared sink absorbs every undefined transition from either DFA;
    # per-DFA sinks would Hopcroft-merge anyway (both non-accepting absorbing).
    offset = dfa_a.num_states
    sink = dfa_a.num_states + dfa_b.num_states
    total_states = sink + 1

    valid_syms_a: frozenset[int] | range = _scope_symbols(dfa_a)
    valid_syms_b: frozenset[int] | range = _scope_symbols(dfa_b)

    combined_transitions: dict[tuple[int, int], int] = {}
    for s in range(dfa_a.num_states):
        for a in range(num_symbols):
            if a in valid_syms_a and (s, a) in dfa_a.transitions:
                combined_transitions[(s, a)] = dfa_a.transitions[(s, a)]
            else:
                combined_transitions[(s, a)] = sink
    for s in range(dfa_b.num_states):
        for a in range(num_symbols):
            if a in valid_syms_b and (s, a) in dfa_b.transitions:
                combined_transitions[(s + offset, a)] = dfa_b.transitions[(s, a)] + offset
            else:
                combined_transitions[(s + offset, a)] = sink
    for a in range(num_symbols):
        combined_transitions[(sink, a)] = sink

    combined_accepting = dfa_a.accepting_states | frozenset(
        s + offset for s in dfa_b.accepting_states
    )

    combined = DFA(
        num_states=total_states,
        num_symbols=num_symbols,
        accepting_states=combined_accepting,
        transitions=combined_transitions,
        valid_symbols=None,
    )

    if restrict_to_reachable:
        reachable_a = _find_reachable_states_in_combined(dfa_a, valid_syms_a, sink, offset=0)
        reachable_b = _find_reachable_states_in_combined(dfa_b, valid_syms_b, sink, offset=offset)
        all_states = reachable_a | reachable_b
    else:
        all_states = set(range(total_states))
    state_to_partition, _num_partitions = _hopcroft_minimize(combined, all_states)

    # For each partition, take a representative dfa_a state, then map each dfa_b
    # state to it.
    partition_to_a: dict[int, int] = {}
    for s in range(dfa_a.num_states):
        pid = state_to_partition.get(s)
        if pid is None:
            continue
        if pid not in partition_to_a:
            partition_to_a[pid] = s

    result: dict[int, int] = {}
    for s in range(dfa_b.num_states):
        pid = state_to_partition.get(s + offset)
        if pid is None:
            continue
        if pid in partition_to_a:
            result[s] = partition_to_a[pid]

    return result


def _find_reachable_states_in_combined(
    dfa: DFA,
    valid_syms: frozenset[int] | range,
    sink: int,
    *,
    offset: int,
) -> set[int]:
    """BFS reachability in the combined-DFA coordinate system of `compute_equivalent_states`: root is state 0 (at `offset`), undefined transitions route to the shared sink."""
    start = offset
    reachable: set[int] = {start}
    queue: deque[int] = deque([start])
    while queue:
        s = queue.popleft()
        local = s - offset
        for a in range(dfa.num_symbols):
            if a in valid_syms and (local, a) in dfa.transitions:
                target = dfa.transitions[(local, a)] + offset
            else:
                target = sink
            if target not in reachable:
                reachable.add(target)
                queue.append(target)
    return reachable


def validate_self_loop_symbol(dfa: DFA, symbol: int) -> None:
    """Raise ValueError unless `symbol` self-loops at every non-terminal state: `delta(q, symbol) == q`.

    Self-loop trace compression (`filter_symbol_from_trace`) drops every occurrence of
    the symbol; this is sound only when reading it never changes the env automaton's
    configuration. Terminal states are exempt (no outgoing; `force_self_loops` skips
    them too). Checked at setup so an unsound compression symbol fails loudly here
    rather than as a mid-training replay error.
    """
    for q in range(dfa.num_states):
        if dfa.is_terminal(q):
            continue
        p = dfa.transitions[(q, symbol)]
        if p != q:
            raise ValueError(
                f"Symbol {symbol} is not a self-loop: delta({q}, {symbol}) = {p}, expected {q}"
            )


def validate_cause_dfa(dfa: DFA) -> None:
    """Refuse a cause DFA unless it has exactly one accepting state, which is absorbing and is not the initial state."""
    non_absorbing_accepting = dfa.accepting_states - compute_absorbing_states(dfa)
    if non_absorbing_accepting:
        raise ValueError(
            f"Cause DFA accepting states must be absorbing, "
            f"but these are not: {non_absorbing_accepting}"
        )
    if len(dfa.accepting_states) != 1:
        raise ValueError(
            f"Cause DFA must have exactly one accepting state, "
            f"got {len(dfa.accepting_states)}: {dfa.accepting_states}"
        )
    if 0 in dfa.accepting_states:
        raise ValueError("Cause DFA initial state (0) must not be accepting")


def bfs_canonical(dfa: DFA) -> DFA:
    """Renumber the states of a total DFA in BFS order from state 0 with symbols visited in increasing order; every state must be reachable."""
    order = [0]
    seen = {0}
    i = 0
    while i < len(order):
        p = order[i]
        i += 1
        for a in range(dfa.num_symbols):
            q = dfa.transitions[(p, a)]
            if q not in seen:
                seen.add(q)
                order.append(q)
    if len(order) != dfa.num_states:
        raise ValueError(f"DFA has {dfa.num_states - len(order)} unreachable states")
    rename = {old: new for new, old in enumerate(order)}
    return DFA(
        num_states=dfa.num_states,
        num_symbols=dfa.num_symbols,
        transitions={(rename[p], a): rename[q] for (p, a), q in dfa.transitions.items()},
        accepting_states=frozenset(rename[q] for q in dfa.accepting_states),
        valid_symbols=dfa.valid_symbols,
    )


def force_self_loops(dfa: DFA, symbol: int) -> DFA:
    """Return a new DFA with `delta(q, symbol) = q` for every non-terminal state.

    Terminal states stay terminal: a self-loop there would break the per-state
    all-or-none rule and the absence-of-run convention.
    """
    new_transitions = dict(dfa.transitions)
    for q in range(dfa.num_states):
        if dfa.is_terminal(q):
            continue
        new_transitions[(q, symbol)] = q
    return DFA(
        num_states=dfa.num_states,
        num_symbols=dfa.num_symbols,
        transitions=new_transitions,
        accepting_states=dfa.accepting_states,
        valid_symbols=dfa.valid_symbols,
    )
