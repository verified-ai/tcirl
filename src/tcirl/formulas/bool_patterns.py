"""Reusable boolean formula patterns for alphabet filtering."""

from collections.abc import Iterable
from functools import reduce

from tcirl.formulas.bool_syntax import BoolAtom, BoolSyntax


def mutual_exclusion_filter[P](aps: Iterable[P]) -> BoolSyntax[P]:
    """At most one of the given APs can be true at any time step."""
    ap_list = list(aps)
    constraints: list[BoolSyntax[P]] = [
        BoolAtom(ap_list[i]) >> ~BoolAtom(ap_list[j])
        for i in range(len(ap_list))
        for j in range(i + 1, len(ap_list))
    ]
    return reduce(lambda a, b: a & b, constraints)


def exactly_one_filter[P](aps: Iterable[P]) -> BoolSyntax[P]:
    """Exactly one of the given APs must be true at any time step."""
    at_most_one = mutual_exclusion_filter(aps)
    at_least_one: BoolSyntax[P] = reduce(lambda a, b: a | b, [BoolAtom(ap) for ap in aps])
    return at_most_one & at_least_one
