"""Gene-editing environment: gridworld x hidden cause DFA x disease x harm.

The full MDP state is (row, col, disease, harm, cause_state, obs_label), with:
  G = gridworld cells (row, col)
  C = cause DFA states (hidden by default)
  D = disease level 0..disease_max (starts at disease_init)
  H = harm level 0..harm_max (starts at HARM_INIT = 0)
  obs_label = most recent combined observation label (hidden bookkeeping)

Observable dimensions always come before hidden ones in EnvState.indices.
Agents observe indices[:n_obs] where n_obs = len(state_shape):
  cause_known=True  -> state_shape = (H, W, disease_max + 1, harm_max + 1, |C|), n_obs=5
  cause_known=False -> state_shape = (H, W, disease_max + 1, harm_max + 1), n_obs=4

The hidden cause DFA reads gridworld labels and transitions synchronously.
It has a single absorbing accepting state. Two-phase disease dynamics are parameterized by
the env config fields `d_rej`, `p_rej`, `d_acc`, and `p_acc`:

  REJECT phase (cause DFA in rejecting state):
        Off medicine: disease +d_rej (with probability p_rej) or +0
        On medicine:  medicine (-cure_reduction) + phase contribution, additive

  ACCEPT phase (cause DFA in accepting state):
        Off medicine: disease d_acc (with probability p_acc) or +0
        On medicine:  medicine (-cure_reduction) + phase contribution, additive

Disease is clamped to [0, disease_max]. Disease overflow is a no-op, not terminal.
Medicine: instantly removes cure_reduction disease points (additive with the phase trend).

Harm: set to harm_max on medicine, -1 off medicine (deterministic, min 0).
Termination: applying medicine while harm > 0 terminates the episode with reward 0.
Dense reward on non-terminal steps: 1.0 - disease / disease_max. No terminal on disease overflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from tcirl.alphabet import OrderedAlphabet
from tcirl.automata.dfa import DFA, validate_cause_dfa
from tcirl.rl.common.tabular import EnvState
from tcirl.rl.envs.gridworld import GridWorld, gridworld_step, render_gridworld
from tcirl.specifications.gene_editing import GeneEditingSpec

if TYPE_CHECKING:
    # Only for the annotation: `tcirl.config` imports this module for its validator.
    from tcirl.config import GeneEditingConfig

# At the default reset no dose has been given, so the first dose of such an episode never terminates it.
HARM_INIT = 0


def validate_gene_editing_parameters(
    *,
    slip_prob: float,
    d_rej: int,
    p_rej: float,
    d_acc: int,
    p_acc: float,
    disease_max: int,
    disease_init: int,
    harm_max: int,
    cure_reduction: int,
) -> None:
    """Refuse a numerical parameter outside the range the two-phase dynamics need."""
    if not (0 <= slip_prob < 1):
        raise ValueError(f"slip_prob must be in [0, 1), got {slip_prob}")
    if d_rej <= 0:
        raise ValueError(f"d_rej must be > 0, got {d_rej}")
    if d_acc >= 0:
        raise ValueError(f"d_acc must be < 0, got {d_acc}")
    # A phase step that never happens would silence that phase's marker: x_r is a rise, x_a a recovery.
    if not (0 < p_rej <= 1):
        raise ValueError(
            f"p_rej must be in (0, 1], or the rejection marker never fires, got {p_rej}"
        )
    if not (0 < p_acc <= 1):
        raise ValueError(
            f"p_acc must be in (0, 1], or the acceptance marker never fires, got {p_acc}"
        )
    if disease_max < 1:
        raise ValueError(
            f"disease_max must be >= 1, because the reward divides by it, got {disease_max}"
        )
    if not (0 <= disease_init <= disease_max):
        raise ValueError(
            f"disease_init must be in [0, disease_max={disease_max}], got {disease_init}"
        )
    # A dose at positive harm is the only termination, so at harm_max = 0 the medicine would cost nothing.
    if harm_max < 1:
        raise ValueError(f"harm_max must be >= 1, so that a dose has a cost, got {harm_max}")
    if cure_reduction < 1:
        raise ValueError(
            f"cure_reduction must be >= 1, so that a dose lowers the disease level, got {cure_reduction}"
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class GeneEditingEnv:
    """Gridworld x hidden-cause-DFA x disease x harm with dense reward."""

    gw: GridWorld
    cause_dfa: DFA
    grid_alphabet: OrderedAlphabet[str]
    obs_alphabet: OrderedAlphabet[str]
    medicine_symbol_id: int
    cause_known: bool
    cause_dfa_size: int
    random_start: bool
    d_rej: int
    p_rej: float
    d_acc: int
    p_acc: float
    disease_max: int
    disease_init: int
    harm_max: int
    cure_reduction: int
    # Precomputed JAX arrays
    cause_delta: jax.Array  # (num_states, grid_num_symbols) int32
    cause_accepting: jax.Array  # (num_states,) bool
    unlabeled_cells: jax.Array  # (N, 2) int32 -- unlabeled walkable (row, col) pairs
    # Label mapping arrays for combined obs labels
    label_with_inc: jax.Array  # (grid_num_symbols,) int32 -> obs symbol with inc
    label_with_dec_medicine: jax.Array  # (grid_num_symbols,) int32 -> obs symbol with dec_medicine
    label_with_dec_recovery: jax.Array  # (grid_num_symbols,) int32 -> obs symbol with dec_recovery
    label_with_dec_medicine_recovery: jax.Array  # (grid_num_symbols,) int32 -> obs symbol with both
    label_base_obs: jax.Array  # (grid_num_symbols,) int32 -> obs symbol with neither
    combined_to_grid: jax.Array  # (obs_num_symbols,) int32 -> grid symbol

    @classmethod
    def create(
        cls,
        path: Path | str,
        *,
        cause_dfa: DFA,
        grid_alphabet: OrderedAlphabet[str],
        obs_alphabet: OrderedAlphabet[str],
        medicine_symbol_id: int,
        slip_prob: float,
        cause_known: bool,
        d_rej: int,
        p_rej: float,
        d_acc: int,
        p_acc: float,
        disease_max: int,
        disease_init: int,
        harm_max: int,
        cure_reduction: int,
        grid_to_obs_base: dict[int, int],
        grid_to_obs_inc: dict[int, int],
        grid_to_obs_dec_medicine: dict[int, int],
        grid_to_obs_dec_recovery: dict[int, int],
        grid_to_obs_dec_medicine_recovery: dict[int, int],
        combined_to_grid_map: dict[int, int],
        random_start: bool = False,
    ) -> GeneEditingEnv:
        """Build a GeneEditingEnv from grid file and a pre-built cause DFA."""
        validate_gene_editing_parameters(
            slip_prob=slip_prob,
            d_rej=d_rej,
            p_rej=p_rej,
            d_acc=d_acc,
            p_acc=p_acc,
            disease_max=disease_max,
            disease_init=disease_init,
            harm_max=harm_max,
            cure_reduction=cure_reduction,
        )

        validate_cause_dfa(cause_dfa)

        gw = GridWorld.create(path, alphabet=grid_alphabet, slip_prob_low=slip_prob)

        # Precompute unlabeled walkable cells for random start.
        walls_np = np.asarray(gw.walls)
        labels_np = np.asarray(gw.labels)
        unlabeled_mask = ~walls_np & (labels_np == gw.no_label)
        unlabeled_positions = np.argwhere(unlabeled_mask).astype(np.int32)
        if unlabeled_positions.shape[0] == 0:
            raise ValueError("No unlabeled walkable cells found in grid")
        unlabeled_cells_arr = jnp.array(unlabeled_positions, dtype=jnp.int32)

        # Precompute cause DFA transition table as JAX array.
        cause_table = np.zeros((cause_dfa.num_states, cause_dfa.num_symbols), dtype=np.int32)
        for (state, symbol), next_state in cause_dfa.transitions.items():
            cause_table[state, symbol] = next_state
        cause_delta_arr = jnp.array(cause_table, dtype=jnp.int32)

        cause_accepting_np = np.array(
            [s in cause_dfa.accepting_states for s in range(cause_dfa.num_states)]
        )
        cause_accepting_arr = jnp.array(cause_accepting_np, dtype=jnp.bool_)

        # Precompute label mapping arrays for combined obs labels.
        grid_n = grid_alphabet.num_symbols
        obs_n = obs_alphabet.num_symbols

        inc_arr = np.zeros(grid_n, dtype=np.int32)
        dec_medicine_arr = np.zeros(grid_n, dtype=np.int32)
        dec_recovery_arr = np.zeros(grid_n, dtype=np.int32)
        dec_medicine_recovery_arr = np.zeros(grid_n, dtype=np.int32)
        base_arr = np.zeros(grid_n, dtype=np.int32)
        for gid in range(grid_n):
            inc_arr[gid] = grid_to_obs_inc[gid]
            dec_medicine_arr[gid] = grid_to_obs_dec_medicine[gid]
            dec_recovery_arr[gid] = grid_to_obs_dec_recovery[gid]
            dec_medicine_recovery_arr[gid] = grid_to_obs_dec_medicine_recovery[gid]
            base_arr[gid] = grid_to_obs_base[gid]

        c2g_arr = np.zeros(obs_n, dtype=np.int32)
        for oid in range(obs_n):
            c2g_arr[oid] = combined_to_grid_map[oid]

        return cls(
            gw=gw,
            cause_dfa=cause_dfa,
            grid_alphabet=grid_alphabet,
            obs_alphabet=obs_alphabet,
            medicine_symbol_id=medicine_symbol_id,
            cause_known=cause_known,
            cause_dfa_size=cause_dfa.num_states,
            random_start=random_start,
            d_rej=d_rej,
            p_rej=p_rej,
            d_acc=d_acc,
            p_acc=p_acc,
            disease_max=disease_max,
            disease_init=disease_init,
            harm_max=harm_max,
            cure_reduction=cure_reduction,
            cause_delta=cause_delta_arr,
            cause_accepting=cause_accepting_arr,
            unlabeled_cells=unlabeled_cells_arr,
            label_with_inc=jnp.array(inc_arr, dtype=jnp.int32),
            label_with_dec_medicine=jnp.array(dec_medicine_arr, dtype=jnp.int32),
            label_with_dec_recovery=jnp.array(dec_recovery_arr, dtype=jnp.int32),
            label_with_dec_medicine_recovery=jnp.array(dec_medicine_recovery_arr, dtype=jnp.int32),
            label_base_obs=jnp.array(base_arr, dtype=jnp.int32),
            combined_to_grid=jnp.array(c2g_arr, dtype=jnp.int32),
        )

    @classmethod
    def from_config(
        cls, config: GeneEditingConfig, spec: GeneEditingSpec, *, cause_known: bool
    ) -> GeneEditingEnv:
        """Build the environment that a config's `env` section describes, over the cause and alphabets of `spec`."""
        return cls.create(
            config.grid_path,
            cause_dfa=spec.cause_dfa,
            grid_alphabet=spec.grid_alphabet,
            obs_alphabet=spec.obs_alphabet,
            medicine_symbol_id=spec.medicine_symbol_id,
            slip_prob=config.slip_prob,
            cause_known=cause_known,
            d_rej=config.d_rej,
            p_rej=config.p_rej,
            d_acc=config.d_acc,
            p_acc=config.p_acc,
            disease_max=config.disease_max,
            disease_init=config.disease_init,
            harm_max=config.harm_max,
            cure_reduction=config.cure_reduction,
            grid_to_obs_base=spec.grid_to_obs_base,
            grid_to_obs_inc=spec.grid_to_obs_inc,
            grid_to_obs_dec_medicine=spec.grid_to_obs_dec_medicine,
            grid_to_obs_dec_recovery=spec.grid_to_obs_dec_recovery,
            grid_to_obs_dec_medicine_recovery=spec.grid_to_obs_dec_medicine_recovery,
            combined_to_grid_map=spec.combined_to_grid,
        )

    @property
    def name(self) -> str:
        return "gene_editing"

    @property
    def cause_index(self) -> int:
        """Position of the cause DFA state in EnvState.indices."""
        return 4

    @property
    def disease_index(self) -> int:
        """Position of the disease level in EnvState.indices."""
        return 2

    @property
    def harm_index(self) -> int:
        """Position of the harm level in EnvState.indices."""
        return 3

    @property
    def obs_label_index(self) -> int:
        """Position of the observation label in EnvState.indices."""
        return 5

    @property
    def state_shape(self) -> tuple[int, ...]:
        """Observable state dimensions for Q-table indexing."""
        if self.cause_known:
            return (
                self.gw.height,
                self.gw.width,
                self.disease_max + 1,
                self.harm_max + 1,
                self.cause_dfa_size,
            )
        return (self.gw.height, self.gw.width, self.disease_max + 1, self.harm_max + 1)

    @property
    def num_actions(self) -> int:
        return self.gw.num_actions

    def reset(self, key: jax.Array) -> EnvState:
        if self.random_start:
            pos_key, d_key, h_key = jax.random.split(key, 3)
            idx = jax.random.randint(pos_key, (), 0, self.unlabeled_cells.shape[0])
            row = self.unlabeled_cells[idx, 0]
            col = self.unlabeled_cells[idx, 1]
            disease = jax.random.randint(d_key, (), 0, self.disease_max + 1).astype(jnp.int32)
            harm = jax.random.randint(h_key, (), 0, self.harm_max + 1).astype(jnp.int32)
        else:
            row = self.gw.starts[0, 0]
            col = self.gw.starts[0, 1]
            disease = jnp.int32(self.disease_init)
            harm = jnp.int32(HARM_INIT)
        cause = jnp.int32(0)
        initial_label = jnp.int32(self.obs_alphabet.encode([]))
        indices = (row, col, disease, harm, cause, initial_label)
        return EnvState(
            indices=indices,
            step_count=jnp.int32(0),
            label=initial_label,
        )

    def render_map(self) -> str:
        return self.render(self.reset(jax.random.key(0)))

    def step(
        self, state: EnvState, action: jax.Array, key: jax.Array
    ) -> tuple[EnvState, jax.Array, jax.Array]:
        row, col = state.indices[0], state.indices[1]
        cause = state.indices[self.cause_index]
        disease = state.indices[self.disease_index]
        harm = state.indices[self.harm_index]

        gw_key, disease_key = jax.random.split(key)

        # 1. Stochastic gridworld step.
        new_row, new_col, grid_label = gridworld_step(self.gw, row, col, action, gw_key)

        # 2. Step cause DFA on grid label.
        # Phase is determined by the CURRENT cause state (before DFA step).
        # Transition probabilities p_phase(q)(s, a) depend on q, not q'.
        in_accepting = self.cause_accepting[cause]
        new_cause = self.cause_delta[cause, grid_label]

        # 3. Determine if on medicine cell.
        on_medicine = grid_label == self.medicine_symbol_id

        # 4. Disease delta (two-phase, medicine is additive with phase trend).
        # Medicine: -cure_reduction (additive with phase contribution).
        phase_delta = jnp.asarray(
            jnp.where(in_accepting, jnp.int32(self.d_acc), jnp.int32(self.d_rej))
        )
        phase_prob = jnp.asarray(
            jnp.where(in_accepting, jnp.float32(self.p_acc), jnp.float32(self.p_rej))
        )
        stochastic_hit = jax.random.uniform(disease_key) < phase_prob
        phase_contribution = jnp.asarray(jnp.where(stochastic_hit, phase_delta, jnp.int32(0)))
        medicine_delta = jnp.asarray(
            jnp.where(on_medicine, jnp.int32(-self.cure_reduction), jnp.int32(0))
        )
        actual_disease_delta = phase_contribution + medicine_delta
        raw_disease = disease + actual_disease_delta

        # 5. Harm: set to harm_max on medicine, -1 off medicine (min 0).
        new_harm = jnp.asarray(
            jnp.where(on_medicine, jnp.int32(self.harm_max), jnp.maximum(harm - 1, jnp.int32(0)))
        )

        # 6. Termination: applying medicine while harm > 0.
        terminated = jnp.bool_(on_medicine & (harm > 0))

        # 7. Clamp disease to valid range. Overflow is a no-op (not terminal).
        new_disease = jnp.clip(raw_disease, 0, self.disease_max)

        # 8. Reward: 0 on terminal, dense otherwise.
        dense_reward = jnp.float32(1.0) - new_disease.astype(jnp.float32) / jnp.float32(
            self.disease_max
        )
        reward = jnp.asarray(jnp.where(terminated, jnp.float32(0.0), dense_reward))

        # Combined label: grid label + effect flags for observation alphabet, a function of the transition.
        # inc: net disease increased. dec_medicine: on medicine cell.
        # dec_recovery: disease ends below what the medicine alone leaves, which only the ACCEPT phase's fall can do.
        is_inc = new_disease > disease
        is_dec_medicine = on_medicine
        is_dec_recovery = new_disease < jnp.clip(disease + medicine_delta, 0, self.disease_max)
        combined_label = jnp.where(
            is_inc,
            self.label_with_inc[grid_label],
            jnp.where(
                is_dec_medicine & is_dec_recovery,
                self.label_with_dec_medicine_recovery[grid_label],
                jnp.where(
                    is_dec_medicine,
                    self.label_with_dec_medicine[grid_label],
                    jnp.where(
                        is_dec_recovery,
                        self.label_with_dec_recovery[grid_label],
                        self.label_base_obs[grid_label],
                    ),
                ),
            ),
        )

        # Observable dimensions first, hidden bookkeeping last.
        new_indices = (
            new_row,
            new_col,
            new_disease,
            new_harm,
            new_cause,
            combined_label,
        )

        new_state = EnvState(
            indices=new_indices,
            step_count=state.step_count + 1,
            label=combined_label,
        )
        return new_state, reward, terminated

    def action_mask(self, state: EnvState) -> jax.Array:
        del state
        return jnp.ones(self.num_actions, dtype=jnp.bool_)

    def render(self, state: EnvState) -> str:
        agent_row = int(state.indices[0])
        agent_col = int(state.indices[1])
        grid = render_gridworld(self.gw, agent_row, agent_col)
        disease = int(state.indices[self.disease_index])
        harm = int(state.indices[self.harm_index])
        cause = int(state.indices[self.cause_index])
        status = f"disease={disease}/{self.disease_max} harm={harm}/{self.harm_max} cause={cause}"
        return f"{grid}\n{status}"
