"""SAT-based cause DFA learning from causal samples.

Learns a DFA consistent with the causal sample: positive prefixes (P, the cause is active) must end in an accepting state and negative prefixes (N, the cause is inactive) in a rejecting one.
The clause groups follow the paper: determinism and totality, execution consistency, successor-closed acceptance, and the P/N consistency constraints.
Two optional groups sit behind flags: `initial_rejecting` forces state 0 to reject, and `symmetry_breaking` adds the BFS-based predicates of Ulyantsev, Zakirzyanov, and Shalyto (LATA 2015), which admit only the BFS-canonical numbering of each DFA.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from pysat.formula import CNF, IDPool
from pysat.solvers import Solver

from tcirl.automata.dfa import DFA
from tcirl.inference.causal_sample import CausalSample
from tcirl.learning import helpers


def cause_dfa_sat(
    num_states: int,
    sample: CausalSample,
    *,
    initial_rejecting: bool = False,
    symmetry_breaking: bool = True,
    silent: bool = True,
) -> DFA | None:
    """Synthesize a cause DFA with `num_states` states consistent with the causal sample, or None if the instance is unsatisfiable."""
    vpool = IDPool()
    cnf = CNF()

    states = list(range(num_states))
    initial_state = 0
    u_eps = 0

    prefixes = sample.prefixes
    alphabet = list(sample.alphabet)

    if not silent:
        print(f"Generating cause DFA constraints ({num_states=}, prefixes={len(prefixes)})...")

    # ===== 1. Variables =====
    def var_delta(p: int, a: int, q: int) -> int:
        return helpers.pool_id(vpool, ("delta", p, a, q))

    def var_acc(q: int) -> int:
        return helpers.pool_id(vpool, ("acc", q))

    def var_x(u: int, q: int) -> int:
        return helpers.pool_id(vpool, ("x", u, q))

    def add_clause(*lits: int) -> None:
        helpers.add_clause(cnf, *lits)

    def imply(premises: list[int], conclusion: int) -> None:
        helpers.imply(cnf, premises, conclusion)

    def exactly_one(lits: list[int]) -> None:
        helpers.exactly_one_cardenc(cnf, vpool, lits)

    # ===== 2. Determinism and totality =====
    for p in states:
        for a in alphabet:
            exactly_one([var_delta(p, a, q) for q in states])

    # ===== 3. Successor-closed acceptance: acc(p) and delta(p, a, q) => acc(q) =====
    for p in states:
        for a in alphabet:
            for q in states:
                if q != p:
                    imply([var_acc(p), var_delta(p, a, q)], var_acc(q))

    if initial_rejecting:
        add_clause(-var_acc(initial_state))

    # ===== 4. Execution consistency =====
    add_clause(var_x(u_eps, initial_state))
    for q in states:
        if q != initial_state:
            add_clause(-var_x(u_eps, q))

    for u in prefixes:
        exactly_one([var_x(u, q) for q in states])

    for u, a, v in sample.nonempty():
        for p in states:
            for q in states:
                imply([var_x(u, p), var_delta(p, a, q)], var_x(v, q))

    # ===== 5. P/N consistency =====
    for u in sample.positive_prefixes:
        for q in states:
            imply([var_x(u, q)], var_acc(q))

    for u in sample.negative_prefixes:
        for q in states:
            imply([var_x(u, q)], -var_acc(q))

    # ===== 6. Symmetry breaking =====
    if symmetry_breaking:
        _add_bfs_symmetry_breaking(cnf, vpool, states, alphabet, var_delta)

    # ===== 7. Solve =====
    if not silent:
        print(f"Solving SAT with {len(cnf.clauses):,} clauses...")  # type: ignore[arg-type]

    with Solver(name="glucose4", bootstrap_with=cnf.clauses) as s:
        solved: bool = s.solve()  # type: ignore[assignment]
        if not solved:
            return None

        model: set[int] = set(s.get_model())  # type: ignore[arg-type]

        transitions: dict[tuple[int, int], int] = {}
        for p in states:
            for a in alphabet:
                for q in states:
                    if var_delta(p, a, q) in model:
                        transitions[(p, a)] = q
                        break

        accepting_states = frozenset(q for q in states if var_acc(q) in model)

        return DFA(
            num_states=num_states,
            num_symbols=sample.num_symbols,
            transitions=transitions,
            accepting_states=accepting_states,
            valid_symbols=None,
        )


def _add_bfs_symmetry_breaking(
    cnf: CNF,
    vpool: IDPool,
    states: Sequence[int],
    alphabet: Sequence[int],
    var_delta: Callable[[int, int, int], int],
) -> None:
    """Admit only the numbering in which states appear in BFS order from state 0, with the children of a state ordered by their smallest symbol."""

    def var_t(i: int, j: int) -> int:
        # Some transition leads from i to j (i < j).
        return helpers.pool_id(vpool, ("t", i, j))

    def var_p(j: int, i: int) -> int:
        # i is the smallest state with a transition to j, that is, its BFS parent (i < j).
        return helpers.pool_id(vpool, ("p", j, i))

    def var_m(i: int, a: int, j: int) -> int:
        # a is the smallest symbol on a transition from i to j (i < j).
        return helpers.pool_id(vpool, ("m", i, a, j))

    def add_clause(*lits: int) -> None:
        helpers.add_clause(cnf, *lits)

    num_states = len(states)
    for j in range(1, num_states):
        for i in range(j):
            add_clause(-var_t(i, j), *[var_delta(i, a, j) for a in alphabet])
            for a in alphabet:
                add_clause(-var_delta(i, a, j), var_t(i, j))
            add_clause(-var_p(j, i), var_t(i, j))
            for k in range(i):
                add_clause(-var_p(j, i), -var_t(k, j))
            add_clause(var_p(j, i), -var_t(i, j), *[var_t(k, j) for k in range(i)])
            for idx, a in enumerate(alphabet):
                smaller = alphabet[:idx]
                add_clause(-var_m(i, a, j), var_delta(i, a, j))
                for b in smaller:
                    add_clause(-var_m(i, a, j), -var_delta(i, b, j))
                add_clause(
                    var_m(i, a, j), -var_delta(i, a, j), *[var_delta(i, b, j) for b in smaller]
                )
        # Every non-initial state has a parent, so every state is reachable.
        add_clause(*[var_p(j, i) for i in range(j)])

    for j in range(1, num_states - 1):
        for i in range(j):
            # Parents are non-decreasing along the numbering.
            for k in range(i):
                add_clause(-var_p(j, i), -var_p(j + 1, k))
            # Children of the same parent are ordered by their smallest symbol.
            for idx, a in enumerate(alphabet):
                for b in alphabet[:idx]:
                    add_clause(-var_p(j, i), -var_p(j + 1, i), -var_m(i, a, j), -var_m(i, b, j + 1))
