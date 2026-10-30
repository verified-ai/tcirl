"""DFA rendering and encoding summaries for logging."""

import sys
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import Protocol, TextIO, overload, runtime_checkable

from graphviz import Digraph

from tcirl.alphabet import Alphabet, BitmaskAlphabet, format_label_set
from tcirl.automata.dfa import DFA
from tcirl.formulas.encoding import DFAStateEncoder, TimerGroupSchema
from tcirl.formulas.ltlf2dfa import report_perf


def init_automaton_digraph(
    name: str,
    num_states: int,
    accepting_states: Collection[int],
    *,
    graph_attr: dict[str, str] | None = None,
) -> Digraph:
    """LR digraph with state nodes (doublecircle = accepting) and the initial-state arrow drawn."""
    g = Digraph(name, graph_attr={"rankdir": "LR", **(graph_attr or {})})
    for s in range(num_states):
        shape = "doublecircle" if s in accepting_states else "circle"
        g.node(str(s), label=str(s), shape=shape)
    g.node("_start", shape="point", width="0")
    g.edge("_start", "0")
    return g


def add_grouped_edges(
    g: Digraph,
    pair_labels: Iterable[tuple[tuple[int, int], str]],
    **edge_attrs: str,
) -> None:
    """Group (src, dst) -> labels, emit one edge per pair with labels deduped and joined by ' | '."""
    grouped: dict[tuple[int, int], set[str]] = {}
    for (p, q), label in pair_labels:
        grouped.setdefault((p, q), set()).add(label)
    for (p, q), labels in grouped.items():
        joined = " | ".join(sorted(labels, key=lambda s: (len(s), s)))
        g.edge(str(p), str(q), label=joined, **edge_attrs)


@runtime_checkable
class AutomatonSummary(Protocol):
    """Protocol for any automaton summary that can be logged to W&B.

    Implementations provide both a plain-text dump (for `*.txt` upload) and a
    Graphviz digraph (for SVG render). `num_states` gates SVG rendering on
    automaton size.
    """

    @property
    def num_states(self) -> int: ...
    def format_text(self, *, max_states: int) -> str: ...
    def to_digraph(self) -> Digraph: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class DFASummaryInfo[L]:
    """DFA summary for logging: DFA + alphabet + optional metadata."""

    dfa: DFA
    alphabet: Alphabet[L]
    formula_str: str | None
    perf: dict[str, float] | None
    encoder: DFAStateEncoder | None
    live_states: frozenset[int] | None

    @property
    def num_states(self) -> int:
        return self.dfa.num_states

    def format_text(self, *, max_states: int) -> str:
        return format_dfa_file(self, max_states=max_states)

    def to_digraph(self) -> Digraph:
        return dfa_to_digraph(self.dfa, self.alphabet)


def _dfa_to_digraph_unlabeled(dfa: DFA) -> Digraph:
    """Convert a plain DFA to a Graphviz Digraph with integer symbol labels."""
    g = init_automaton_digraph("DFA", dfa.num_states, dfa.accepting_states)
    add_grouped_edges(g, (((p, q), str(a)) for (p, a), q in dfa.transitions.items()))
    return g


def _dfa_to_digraph_labeled[L](dfa: DFA, alphabet: Alphabet[L]) -> Digraph:
    """Convert a DFA with an alphabet to a Graphviz Digraph with decoded labels."""
    g = init_automaton_digraph("DFA", dfa.num_states, dfa.accepting_states)
    add_grouped_edges(
        g,
        (
            ((p, q), alphabet.format_label(alphabet.decode(a)))
            for (p, a), q in dfa.transitions.items()
        ),
    )
    return g


@overload
def dfa_to_digraph(dfa: DFA) -> Digraph: ...
@overload
def dfa_to_digraph[L](dfa: DFA, alphabet: Alphabet[L]) -> Digraph: ...


def dfa_to_digraph[L](dfa: DFA, alphabet: Alphabet[L] | None = None) -> Digraph:
    """Convert a DFA to a Graphviz Digraph.

    If an alphabet is provided, edge labels show decoded labels.
    Otherwise, edge labels show integer symbols.
    """
    if alphabet is not None:
        return _dfa_to_digraph_labeled(dfa, alphabet)
    return _dfa_to_digraph_unlabeled(dfa)


def format_encoding_summary_state(
    dfa: DFA,
    encoder: DFAStateEncoder,
    state_id: int,
    *,
    indent: str = "",
    include_header: bool = True,
    raw_encoding: bool = False,
) -> str:
    """Format encoding summary for a single DFA state as a string."""
    enc = encoder.encode(state_id)
    timer_values = encoder.decode_timer_values(enc)

    enc_list = enc.tolist()
    timer_dim = encoder.timer_dim
    control_part = enc_list[timer_dim:]
    control_id = control_part.index(1.0) if 1.0 in control_part else -1

    lines: list[str] = []
    if include_header:
        mark = " ✓" if state_id in dfa.accepting_states else ""
        lines.append(f"{indent}State {state_id}{mark}:")
        timer_indent = indent + "  "
    else:
        timer_indent = indent

    for name, values in timer_values.items():
        vals_str = "[" + ", ".join(f"{v:.2f}" for v in values) + "]"
        lines.append(f"{timer_indent}{name}: {vals_str}")
    lines.append(f"{indent}Control state = {control_id}")
    if raw_encoding:
        timer_part = enc_list[:timer_dim]
        timer_str = "[" + ", ".join(f"{v:.2f}" for v in timer_part) + "]"
        lines.append(f"{indent}Encoding: timers={timer_str}, control={control_id}")
    return "\n".join(lines)


def write_encoding_summary_state(
    dfa: DFA,
    encoder: DFAStateEncoder,
    state_id: int,
    file: TextIO | None = None,
    *,
    raw_encoding: bool = False,
) -> None:
    """Write encoding summary for a single DFA state to a file or stdout."""
    out = file if file is not None else sys.stdout
    out.write(
        format_encoding_summary_state(
            dfa, encoder, state_id, indent="    ", raw_encoding=raw_encoding
        )
        + "\n"
    )


def print_encoding_summary_state(dfa: DFA, encoder: DFAStateEncoder, state_id: int) -> None:
    """Print encoding summary for a single DFA state to stdout."""
    write_encoding_summary_state(dfa, encoder, state_id)


def format_encoding_summary_all_states(
    dfa: DFA,
    encoder: DFAStateEncoder,
    *,
    title: str | None = None,
    max_states: int | None = None,
) -> str:
    """Format encoding summary for all DFA states as a string."""
    lines: list[str] = []
    if title:
        lines.append(f"\n=== {title} ===")
    lines.append(f"Timer dim: {encoder.timer_dim}, Control dim: {encoder.finite_control_dim}")

    total_states = dfa.num_states
    limit = total_states if max_states is None else min(max_states, total_states)

    lines.extend(
        format_encoding_summary_state(dfa, encoder, state_id, indent="    ")
        for state_id in range(limit)
    )

    if max_states is not None and total_states > max_states:
        lines.append(f"    ... (showing {max_states} of {total_states} states)")

    return "\n".join(lines)


def print_encoding_summary_all_states(
    dfa: DFA,
    encoder: DFAStateEncoder,
    *,
    title: str | None = None,
) -> None:
    """Print encoding summary for all DFA states."""
    print(format_encoding_summary_all_states(dfa, encoder, title=title))


def print_run[P](
    dfa: DFA,
    alphabet: BitmaskAlphabet[P],
    labels: Iterable[Collection[P]],
    encoder: DFAStateEncoder | None = None,
    raw_encoding: bool = True,
) -> None:
    """Print a trace through the DFA showing state transitions.

    Displays the automaton's run on a label sequence, showing each transition
    with state IDs, acceptance marks, and optional encodings.
    """
    labels_list = list(labels)
    print(f"Input: {labels_list}\n")

    # Compute all states
    states: list[int] = [dfa.reset()]
    for label in labels_list:
        states.append(dfa.step(states[-1], alphabet.encode(label)))

    # Format labels for alignment
    label_strs = [format_label_set(label) for label in labels_list]
    max_label_len = max(len(s) for s in label_strs) if label_strs else 0

    def fmt_state(s: int) -> str:
        mark = "✓" if s in dfa.accepting_states else " "
        return f"({s}) {mark}"

    def print_encoder_info(state_id: int) -> None:
        if encoder:
            print(
                format_encoding_summary_state(
                    dfa,
                    encoder,
                    state_id,
                    indent="        ",
                    include_header=False,
                    raw_encoding=raw_encoding,
                )
            )

    # Print initial state
    print(f"{'':>{max_label_len}}    {fmt_state(states[0])}")
    print_encoder_info(states[0])

    # Print transitions
    for i, label_str in enumerate(label_strs):
        print(f"{label_str:>{max_label_len}}  → {fmt_state(states[i + 1])}")
        print_encoder_info(states[i + 1])

    print()
    print("ACCEPTED" if states[-1] in dfa.accepting_states else "REJECTED")


def _format_timer_groups(groups: tuple[TimerGroupSchema, ...]) -> str:
    """Format timer group schema as a readable string."""
    if not groups:
        return "  (none)"
    lines: list[str] = [
        f"  {g.name}: original_lower={g.original_lower}, original_upper={g.original_upper}, dim={g.encoding_dim}"
        for g in groups
    ]
    return "\n".join(lines)


def _format_transitions[L](
    dfa: DFA,
    alphabet: Alphabet[L],
    *,
    max_states: int = 250,
) -> str:
    """Format DFA transitions with decoded labels."""
    lines: list[str] = []
    lines.append("\n=== Transitions ===")
    limit = min(dfa.num_states, max_states)
    for state in range(limit):
        mark = " (accepting)" if state in dfa.accepting_states else ""
        lines.append(f"  State {state}{mark}:")
        # Group transitions by destination
        dest_labels: dict[int, list[str]] = {}
        for (s, sym), nxt in dfa.transitions.items():
            if s != state:
                continue
            decoded = alphabet.decode(sym)
            dest_labels.setdefault(nxt, []).append(alphabet.format_label(decoded))
        for dest in sorted(dest_labels):
            label_strs = sorted(dest_labels[dest], key=lambda x: (len(x), x))
            lines.append(f"    -> {dest}: {' | '.join(label_strs)}")
    if dfa.num_states > max_states:
        lines.append(f"  ... (showing {max_states} of {dfa.num_states} states)")
    return "\n".join(lines)


def format_dfa_file[L](
    info: DFASummaryInfo[L],
    *,
    max_states: int = 250,
) -> str:
    """Compose a complete text summary for a DFA."""
    dfa = info.dfa
    alphabet = info.alphabet

    sections: list[str] = []

    # Section 1: Formula (optional)
    if info.formula_str is not None:
        sections.append("=== Formula ===")
        sections.append(info.formula_str)

    # Section 2: DFA construction performance (optional)
    if info.perf is not None:
        sections.append("")
        sections.append(report_perf(info.perf))

    # Section 3: DFA meta-info
    sections.append("")
    sections.append("=== DFA Info ===")
    sections.append(f"States: {dfa.num_states}")
    sections.append(f"Accepting states: {sorted(dfa.accepting_states)}")
    if info.live_states is not None:
        sections.append(f"Live states: {len(info.live_states)} of {dfa.num_states}")
    sections.append(f"Symbols: {alphabet.num_symbols} ({len(alphabet.valid_symbols)} valid)")
    sections.append(f"Partial DFA: {dfa.partial}")

    # Section 4: Encoding schema (only when encoder is available)
    if info.encoder is not None:
        schema = info.encoder.schema
        sections.append("")
        sections.append("=== Encoding Schema ===")
        sections.append(f"Total encoding dim: {schema.total_dim}")
        sections.append(f"Timer dim: {schema.timer_dim}")
        sections.append(f"Finite control dim: {schema.finite_control_dim}")
        sections.append("Timer groups:")
        sections.append(_format_timer_groups(schema.timer_groups))

    # Section 5: State encodings or decoded transitions
    if info.encoder is not None:
        sections.append("")
        sections.append(
            format_encoding_summary_all_states(
                dfa, info.encoder, title="State Encodings", max_states=max_states
            )
        )
    else:
        sections.append(_format_transitions(dfa, alphabet, max_states=max_states))

    return "\n".join(sections)
