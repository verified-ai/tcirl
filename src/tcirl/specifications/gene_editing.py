"""Gene-editing task specification: hidden cause DFA.

The environment models a gene-editing protocol with gene-modification sites
(a, b, d) on a gridworld plus a distractor gene c and a medicine cell e.
The hidden cause requires visiting a, b, a, d in strict order:
F(a & X(F(b & X(F(a & X(F(d))))))).
Gene c and the medicine cell e are distractors (on the grid but NOT in the cause formula).

Two alphabets:
  - Grid alphabet: {a, b, c, d, e} with mutual exclusion -> 6 symbols (including empty).
    Used by the cause DFA, hypothesis DFA, and SAT solver.
  - Observation alphabet: grid props x {inc, dec_medicine, dec_recovery} -> 30 symbols.
    inc is exclusive with both dec_medicine and dec_recovery, but
    dec_medicine and dec_recovery can coexist.
    Used by the environment label output for disease direction tracking.

Marker propositions (atomic, phase-exclusive):
  - x_r = "inc": disease increased. Only possible in REJECT phase (d_rej > 0).
    Impossible in ACCEPT phase (d_acc < 0, disease only decreases or stays flat).
  - x_a = "dec_recovery": disease ends below the level the medicine alone leaves (below the old
    level off the medicine cell). Only the ACCEPT phase lowers disease on its own (d_acc < 0).
  - "dec_medicine": agent stepped on the medicine cell. Can occur in both phases.
    dec_medicine and dec_recovery can fire simultaneously (on medicine in ACCEPT phase
    when disease falls below the medicine's level).
  Every proposition is a function of the transition: the cell entered and the disease before and after.
"""

from __future__ import annotations

from dataclasses import dataclass

from tcirl.alphabet import OrderedAlphabet
from tcirl.automata.conversion import bitmask_to_ordered
from tcirl.automata.dfa import DFA, validate_cause_dfa
from tcirl.formulas.bool_patterns import mutual_exclusion_filter
from tcirl.formulas.bool_syntax import BoolAtom, BoolSyntax
from tcirl.formulas.ltlf2dfa import ltlf_to_dfa
from tcirl.formulas.ltlf_syntax import Atom, Finally, Next
from tcirl.path_consts import CACHE_DIR

# Grid propositions: gene-modification sites + medicine cell
_A = Atom("a")  # Gene 1
_B = Atom("b")  # Gene 2
_C = Atom("c")  # Gene 3 (distractor -- NOT in cause formula)
_D = Atom("d")  # Gene 4
_E = Atom("e")  # Medicine cell (NOT in cause formula)

# The hidden cause: visit a, then b, then a again, then d in strict order.
# F(a & X(F(b & X(F(a & X(F(d)))))))
CAUSE_FORMULA = Finally(_A & Next(Finally(_B & Next(Finally(_A & Next(Finally(_D)))))))

# Marker propositions (atomic, phase-exclusive).
X_R_PROP = "inc"
X_A_PROP = "dec_recovery"


GRID_PROPS: tuple[str, ...] = ("a", "b", "c", "d", "e")
EFFECT_PROPS: tuple[str, ...] = ("inc", "dec_medicine", "dec_recovery")

# Grid cell props are mutually exclusive (agent occupies one cell at a time).
GRID_FILTER: BoolSyntax[str] = mutual_exclusion_filter(list(GRID_PROPS))

# Effect props: inc is exclusive with both dec_medicine and dec_recovery,
# but dec_medicine and dec_recovery can coexist.
_INC = BoolAtom("inc")
_MED = BoolAtom("dec_medicine")
_REC = BoolAtom("dec_recovery")
EFFECT_FILTER: BoolSyntax[str] = (~_INC | ~_MED) & (~_INC | ~_REC)

# Combined filter for the observation alphabet.
OBS_FILTER: BoolSyntax[str] = GRID_FILTER & EFFECT_FILTER


@dataclass(frozen=True, slots=True, kw_only=True)
class GeneEditingSpec:
    """All specification data for the gene-editing environment."""

    cause_dfa: DFA
    grid_alphabet: OrderedAlphabet[str]
    obs_alphabet: OrderedAlphabet[str]
    medicine_symbol_id: int
    combined_to_grid: dict[int, int]
    x_r_obs_ids: frozenset[int]
    x_a_obs_ids: frozenset[int]
    grid_to_obs_base: dict[int, int]
    grid_to_obs_inc: dict[int, int]
    grid_to_obs_dec_medicine: dict[int, int]
    grid_to_obs_dec_recovery: dict[int, int]
    grid_to_obs_dec_medicine_recovery: dict[int, int]
    perf: dict[str, float]


def create_gene_editing_cause() -> GeneEditingSpec:
    """Build the cause DFA, both alphabets, and all label mappings."""
    # 1. Build cause DFA on the 6-symbol grid alphabet.
    cause_bitmask_dfa, cause_bitmask_alph, _encoder, perf = ltlf_to_dfa(
        CAUSE_FORMULA,
        label_filter=GRID_FILTER,
        extra_props=("c", "e"),
        cache_dir=CACHE_DIR,
        minimize=True,
        build_encoding=False,
    )
    cause_ordered_dfa, grid_alphabet = bitmask_to_ordered(cause_bitmask_dfa, cause_bitmask_alph)

    validate_cause_dfa(cause_ordered_dfa)
    if frozenset() not in set(grid_alphabet.labels):
        raise ValueError("Grid alphabet must have an encoding for the empty-set label")

    medicine_symbol_id = grid_alphabet.encode(frozenset({"e"}))

    # 2. Build the 30-symbol observation alphabet (grid x effect flags).
    # 6 grid symbols x 5 effect combos (none, inc, dec_medicine, dec_recovery,
    # dec_medicine+dec_recovery) = 30.
    _obs_extra = ("c", "e", "inc", "dec_medicine", "dec_recovery")
    obs_bitmask_dfa, obs_bitmask_alph, _, _ = ltlf_to_dfa(
        CAUSE_FORMULA,
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
        if X_R_PROP in obs_alphabet.decode(obs_id)
    )
    # x_a = dec_recovery: directly tied to the recovery Bernoulli in ACCEPT phase, so it is
    # phase-exclusive by construction.
    x_a_obs_ids = frozenset(
        obs_id
        for obs_id in range(obs_alphabet.num_symbols)
        if X_A_PROP in obs_alphabet.decode(obs_id)
    )

    # 4. Build projection: obs symbol ID -> grid symbol ID (strip effect props).
    combined_to_grid: dict[int, int] = {}
    for obs_id in range(obs_alphabet.num_symbols):
        obs_label = obs_alphabet.decode(obs_id)
        grid_label = obs_label - {"inc", "dec_medicine", "dec_recovery"}
        grid_id = grid_alphabet.encode(grid_label)
        combined_to_grid[obs_id] = grid_id

    # 5. Build grid -> obs mappings (for env step to construct combined labels).
    grid_to_obs_base: dict[int, int] = {}
    grid_to_obs_inc: dict[int, int] = {}
    grid_to_obs_dec_medicine: dict[int, int] = {}
    grid_to_obs_dec_recovery: dict[int, int] = {}
    grid_to_obs_dec_medicine_recovery: dict[int, int] = {}
    for grid_id in range(grid_alphabet.num_symbols):
        grid_label = grid_alphabet.decode(grid_id)
        grid_to_obs_base[grid_id] = obs_alphabet.encode(grid_label)
        grid_to_obs_inc[grid_id] = obs_alphabet.encode(grid_label | {"inc"})
        grid_to_obs_dec_medicine[grid_id] = obs_alphabet.encode(grid_label | {"dec_medicine"})
        grid_to_obs_dec_recovery[grid_id] = obs_alphabet.encode(grid_label | {"dec_recovery"})
        grid_to_obs_dec_medicine_recovery[grid_id] = obs_alphabet.encode(
            grid_label | {"dec_medicine", "dec_recovery"}
        )

    return GeneEditingSpec(
        cause_dfa=cause_ordered_dfa,
        grid_alphabet=grid_alphabet,
        obs_alphabet=obs_alphabet,
        medicine_symbol_id=medicine_symbol_id,
        combined_to_grid=combined_to_grid,
        x_r_obs_ids=x_r_obs_ids,
        x_a_obs_ids=x_a_obs_ids,
        grid_to_obs_base=grid_to_obs_base,
        grid_to_obs_inc=grid_to_obs_inc,
        grid_to_obs_dec_medicine=grid_to_obs_dec_medicine,
        grid_to_obs_dec_recovery=grid_to_obs_dec_recovery,
        grid_to_obs_dec_medicine_recovery=grid_to_obs_dec_medicine_recovery,
        perf=perf,
    )
