from collections.abc import Sequence

from pysat.card import CardEnc
from pysat.formula import CNF, IDPool


def pool_id(vpool: IDPool, key: object) -> int:
    """Typed wrapper over pysat's untyped `IDPool.id`."""
    return vpool.id(key)  # pyright: ignore[reportUnknownVariableType]


def add_clause(cnf: CNF, *lits: int) -> None:
    cnf.append(list(lits))


def imply(cnf: CNF, premise: Sequence[int], consequence: int) -> None:
    add_clause(cnf, *([-p for p in premise] + [consequence]))


def none_true(cnf: CNF, lits: Sequence[int]) -> None:
    for lit in lits:
        cnf.append([-lit])


def exactly_one_cardenc(cnf: CNF, vpool: IDPool, lits: Sequence[int]) -> None:
    if lits:
        cnf.extend(CardEnc.equals(lits=list(lits), bound=1, vpool=vpool).clauses)


def at_most_one_cardenc(cnf: CNF, vpool: IDPool, lits: Sequence[int]) -> None:
    if lits:
        cnf.extend(CardEnc.atmost(lits=list(lits), bound=1, vpool=vpool).clauses)
