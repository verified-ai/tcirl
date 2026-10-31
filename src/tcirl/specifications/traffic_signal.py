"""Traffic signal control task specification: hidden cause DFA.

The environment models a traffic signal control system with 3 intersections
in a left-to-right pipeline.  The agent chooses which single intersection is
blocked (red); the other two are green.  Two actions -- hold (keep current
block) and advance (rotate the block to the next intersection) -- control
traffic flow.

The hidden temporal cause fires when the agent completes three hold-advance
pairs without interruption: h,a,h,a,h,a.  This is exactly the optimal
pattern for a cause-unaware agent that only respects the mandatory hold
(advance every other step).  Avoiding the cause requires inserting an extra
hold after every second advance to break the pattern.

Two alphabets:
  - Action alphabet: {advance} -> 2 symbols (advance or hold=empty).
    Used by the cause DFA, hypothesis DFA, and SAT solver.
  - Observation alphabet: action x {multi_arr, multi_dep} -> 6 symbols.
    Used by the environment label output for arrival/departure tracking.

Marker propositions (atomic, phase-exclusive):
  - x_r = "multi_dep": 2+ cars exited from q2. Only possible in REJECT phase
    (Geom flow). Impossible in ACCEPT (Bernoulli, max 1 exit).
  - x_a = "multi_arr": 2+ cars arrived at q0. Only possible in ACCEPT phase
    (Geom arrivals). Impossible in REJECT (Bernoulli, max 1).
  - Neither: ambiguous -- can occur in both phases.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tcirl.alphabet import OrderedAlphabet, projection_table
from tcirl.automata.conversion import bitmask_to_ordered
from tcirl.automata.dfa import DFA, validate_cause_dfa
from tcirl.formulas.bool_patterns import mutual_exclusion_filter
from tcirl.formulas.bool_syntax import BoolSyntax, BoolTop
from tcirl.formulas.ltlf2dfa import ltlf_to_dfa
from tcirl.formulas.ltlf_syntax import Atom, Finally, LTLfSyntax, Next
from tcirl.path_consts import CACHE_DIR

# The action proposition: an advance happened. Hold is the action alphabet's empty symbol.
_ADV = Atom("advance")


def build_word_formula(word: tuple[bool, ...]) -> LTLfSyntax[str]:
    """The cause that fires when `word` appears as a run of CONSECUTIVE actions.

    `True` is an advance and `False` a hold, and the chain of `Next` operators under one `Finally`
    is what makes the match consecutive: the cause is a factor of the action word, not a scattered
    subsequence. Built right to left.
    """
    if len(word) < 3:
        raise ValueError(f"a cause word needs at least 3 actions, got {len(word)}")
    inner: LTLfSyntax[str] = _ADV if word[-1] else ~_ADV
    for step in reversed(word[:-1]):
        atom: LTLfSyntax[str] = _ADV if step else ~_ADV
        inner = atom & Next(inner)
    return Finally(inner)


def validate_difficulty(difficulty: int) -> None:
    """Refuse an alternating cause shorter than three actions."""
    if difficulty < 3:
        raise ValueError(f"difficulty must be >= 3, got {difficulty}")


def alternating_word(difficulty: int) -> tuple[bool, ...]:
    """The cause's word: `difficulty` alternating hold-advance actions, hold first."""
    validate_difficulty(difficulty)
    return tuple(step % 2 == 1 for step in range(difficulty))


def build_cause_formula(difficulty: int) -> LTLfSyntax[str]:
    """Build the cause formula for a given difficulty (number of hold-advance steps)."""
    return build_word_formula(alternating_word(difficulty))


# Marker propositions (atomic, phase-exclusive).
# Two departures in one step need the bursty departure draw, which only Phase 1 has, so `multi_dep` proves the cause has NOT fired (x_r).
# Two arrivals need the bursty arrival draw, which only Phase 2 has, so `multi_arr` proves it HAS (x_a).
MULTI_DEP = "multi_dep"
MULTI_ARR = "multi_arr"


ACTION_PROPS: tuple[str, ...] = ("advance",)
EFFECT_PROPS: tuple[str, ...] = ("multi_arr", "multi_dep")

# Single action prop -> mutual exclusion is vacuously true.
ACTION_FILTER: BoolSyntax[str] = BoolTop()

# multi_arr and multi_dep are mutually exclusive (phase-exclusive by design).
OBS_FILTER: BoolSyntax[str] = mutual_exclusion_filter(("multi_arr", "multi_dep"))


@dataclass(frozen=True, slots=True, kw_only=True)
class TrafficSignalSpec:
    """All specification data for the traffic signal environment."""

    cause_dfa: DFA
    cause_formula: LTLfSyntax[str]
    action_alphabet: OrderedAlphabet[str]
    obs_alphabet: OrderedAlphabet[str]
    combined_to_action: dict[int, int]
    x_r_obs_ids: frozenset[int]
    x_a_obs_ids: frozenset[int]
    action_to_obs_base: dict[int, int]
    action_to_obs_multi_arr: dict[int, int]
    action_to_obs_multi_dep: dict[int, int]
    perf: dict[str, float]

    @property
    def label_projection(self) -> np.ndarray:
        """The action symbol of each observation label, the symbol the cause and every hypothesis read."""
        return projection_table(self.combined_to_action, self.obs_alphabet.num_symbols)


def create_traffic_signal_cause(difficulty: int) -> TrafficSignalSpec:
    """Build the cause DFA of the alternating cause, both alphabets, and all label mappings."""
    cause_formula = build_cause_formula(difficulty)

    # 1. Build cause DFA on the 2-symbol action alphabet.
    cause_bitmask_dfa, cause_bitmask_alph, _encoder, perf = ltlf_to_dfa(
        cause_formula,
        label_filter=ACTION_FILTER,
        cache_dir=CACHE_DIR,
        minimize=True,
        build_encoding=False,
    )
    cause_ordered_dfa, action_alphabet = bitmask_to_ordered(cause_bitmask_dfa, cause_bitmask_alph)

    validate_cause_dfa(cause_ordered_dfa)
    if frozenset() not in set(action_alphabet.labels):
        raise ValueError("Action alphabet must have an encoding for the empty-set label")

    # 2. Build the 6-symbol observation alphabet (2 action x 3 effect).
    _obs_extra = ("multi_arr", "multi_dep")
    obs_bitmask_dfa, obs_bitmask_alph, _, _ = ltlf_to_dfa(
        cause_formula,
        label_filter=OBS_FILTER,
        extra_props=_obs_extra,
        cache_dir=CACHE_DIR,
        minimize=True,
        build_encoding=False,
    )
    _obs_ordered_dfa, obs_alphabet = bitmask_to_ordered(obs_bitmask_dfa, obs_bitmask_alph)

    # 3. Compute marker obs IDs: which obs-alphabet symbols contain x_r / x_a.
    x_r_obs_ids = frozenset(
        obs_id
        for obs_id in range(obs_alphabet.num_symbols)
        if MULTI_DEP in obs_alphabet.decode(obs_id)
    )
    x_a_obs_ids = frozenset(
        obs_id
        for obs_id in range(obs_alphabet.num_symbols)
        if MULTI_ARR in obs_alphabet.decode(obs_id)
    )

    # 4. Build projection: obs symbol ID -> action symbol ID (strip effect props).
    combined_to_action: dict[int, int] = {}
    for obs_id in range(obs_alphabet.num_symbols):
        obs_label = obs_alphabet.decode(obs_id)
        action_label = obs_label - {"multi_arr", "multi_dep"}
        action_id = action_alphabet.encode(action_label)
        combined_to_action[obs_id] = action_id

    # 5. Build action -> obs mappings (for env step to construct combined labels).
    action_to_obs_base: dict[int, int] = {}
    action_to_obs_multi_arr: dict[int, int] = {}
    action_to_obs_multi_dep: dict[int, int] = {}
    for action_id in range(action_alphabet.num_symbols):
        action_label = action_alphabet.decode(action_id)
        action_to_obs_base[action_id] = obs_alphabet.encode(action_label)
        action_to_obs_multi_arr[action_id] = obs_alphabet.encode(action_label | {"multi_arr"})
        action_to_obs_multi_dep[action_id] = obs_alphabet.encode(action_label | {"multi_dep"})

    return TrafficSignalSpec(
        cause_dfa=cause_ordered_dfa,
        cause_formula=cause_formula,
        action_alphabet=action_alphabet,
        obs_alphabet=obs_alphabet,
        combined_to_action=combined_to_action,
        x_r_obs_ids=x_r_obs_ids,
        x_a_obs_ids=x_a_obs_ids,
        action_to_obs_base=action_to_obs_base,
        action_to_obs_multi_arr=action_to_obs_multi_arr,
        action_to_obs_multi_dep=action_to_obs_multi_dep,
        perf=perf,
    )
