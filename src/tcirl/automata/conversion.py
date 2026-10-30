"""Conversion between BitmaskAlphabet and OrderedAlphabet DFA encodings."""

from __future__ import annotations

from tcirl.alphabet import BitmaskAlphabet, OrderedAlphabet
from tcirl.automata.dfa import DFA


def bitmask_to_ordered[P](
    dfa: DFA,
    alphabet: BitmaskAlphabet[P],
) -> tuple[DFA, OrderedAlphabet[P]]:
    """Convert a bitmask-encoded DFA to use compact sequential symbol IDs.

    Only transitions for valid_symbols are preserved.
    The resulting DFA has num_symbols = len(valid_symbols).
    """
    valid_sorted = sorted(alphabet.valid_symbols)
    bitmask_to_seq = {bitmask: seq for seq, bitmask in enumerate(valid_sorted)}

    labels = tuple(alphabet.decode(bitmask) for bitmask in valid_sorted)
    ordered = OrderedAlphabet[P].create(props=alphabet.props, labels=labels)

    new_transitions: dict[tuple[int, int], int] = {}
    for (state, bitmask), next_state in dfa.transitions.items():
        if bitmask in bitmask_to_seq:
            new_transitions[(state, bitmask_to_seq[bitmask])] = next_state

    num_new_symbols = len(valid_sorted)
    new_dfa = DFA(
        num_states=dfa.num_states,
        num_symbols=num_new_symbols,
        accepting_states=dfa.accepting_states,
        transitions=new_transitions,
        valid_symbols=None,
    )

    return new_dfa, ordered


def ordered_to_bitmask[P](
    dfa: DFA,
    ordered: OrderedAlphabet[P],
    bitmask_template: BitmaskAlphabet[P],
) -> tuple[DFA, BitmaskAlphabet[P]]:
    """Convert a compact sequential DFA back to bitmask encoding.

    Uses bitmask_template for the proposition-to-bit mapping.
    The resulting DFA is partial (only valid symbols have transitions).
    """
    valid_sorted = sorted(bitmask_template.valid_symbols)
    if len(valid_sorted) != ordered.num_symbols:
        raise ValueError(
            f"OrderedAlphabet has {ordered.num_symbols} symbols but "
            f"BitmaskAlphabet has {len(valid_sorted)} valid symbols"
        )

    seq_to_bitmask = dict(enumerate(valid_sorted))
    new_transitions: dict[tuple[int, int], int] = {}
    for (state, seq_id), next_state in dfa.transitions.items():
        new_transitions[(state, seq_to_bitmask[seq_id])] = next_state

    is_partial = bitmask_template.is_filtered
    new_dfa = DFA(
        num_states=dfa.num_states,
        num_symbols=bitmask_template.num_symbols,
        accepting_states=dfa.accepting_states,
        transitions=new_transitions,
        valid_symbols=bitmask_template.valid_symbols if is_partial else None,
    )

    return new_dfa, bitmask_template
