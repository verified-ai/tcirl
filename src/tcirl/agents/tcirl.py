"""TCIRL: Temporal-Causal Inference for Reinforcement Learning.

Standard Q-learning with a hypothesis cause DFA included in the state.
The hypothesis DFA is maintained per-seed and updated via causal inference
from observed effect trajectories and label data.

Q-table layout: (*obs_prefix, cause_hyp, *obs_suffix, action).
The Q-table shape is constant regardless of inference mode; hypothesis DFAs
are always padded to cause_dfa_size.

The agent is transparent to the inference mode: it receives initial hypothesis
DFAs from the orchestration layer.

Each seed maintains its own hypothesis cause DFA, and for sat mode, its own
persistent causal sample (prefix tree with positive/negative annotations).
Re-inference is counterexample-driven: we only re-solve when a completed
episode contradicts the current hypothesis.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tcirl.alphabet import OrderedAlphabet
from tcirl.automata.dfa import DFA, force_self_loops
from tcirl.automata.plain_dfa import PlainDFA
from tcirl.config import InferenceMode, ValueTransfer
from tcirl.inference.causal_sample import MutableCausalSample
from tcirl.inference.markers import certificates, compress_certificates, is_counterexample
from tcirl.inference.solver import learn_cause_dfa
from tcirl.inference.transfer import kept_update_share, transfer_rows, transfer_sources
from tcirl.learning.trace import filter_symbol_from_trace
from tcirl.logging.batch import BatchLogger
from tcirl.logging.recorder import PerSeedMetricRecorder
from tcirl.logging.wandb import WandbLogger
from tcirl.rl.common.eval_schedule import EvalSchedule, EvalSink, build_eval_driver
from tcirl.rl.common.greedy_eval import tabular_argmax
from tcirl.rl.common.tabular import EnvState, TabularEnv
from tcirl.rl.schedules import rescaled_linear_step_size
from tcirl.visualization import DFASummaryInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_padded_cause_arrays(dfa: DFA, max_states: int) -> tuple[jax.Array, jax.Array]:
    """Build padded (delta, accepting) JAX arrays from a cause DFA hypothesis."""
    plain = PlainDFA.from_dfa(dfa, max_states)
    return plain.delta, plain.accepting


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
# The scan carry and the per-step outputs of this agent's own iteration, named so the four
# signatures that thread them cannot drift apart. The carry holds the hypothesis's acceptance
# classes, the two running per-episode return accumulators, the update counts and the step at which
# the seed's exploration schedule starts; the outputs carry the accumulators' boundary samples, the same two signals
# `rl.common.qrm_core` emits, so this agent feeds the one logging adapter like every other arm.
_TrainCarry = tuple[
    jax.Array,  # q_table
    EnvState,
    jax.Array,  # hypothesis cause state
    jax.Array,  # global step
    jax.Array,  # PRNG key
    jax.Array,  # hypothesis delta
    jax.Array,  # hypothesis class of each padded state: 1 accepting, 0 rejecting, -1 filler
    jax.Array,  # ep_return, the running undiscounted sum of this episode's rewards
    jax.Array,  # ep_disc_return, the same sum discounted from the episode's own first step
    jax.Array,  # update_count, the number of TD updates of each Q-table entry
    jax.Array,  # the step at which this seed's exploration schedule starts
]
_TrainOutputs = tuple[
    jax.Array,  # reward
    jax.Array,  # |TD|
    jax.Array,  # q_max
    jax.Array,  # episode_over
    jax.Array,  # ep_length
    jax.Array,  # ep_return at the boundary, else 0
    jax.Array,  # ep_disc_return at the boundary, else 0
    jax.Array,  # label
    jax.Array,  # terminated
    jax.Array,  # effect
    jax.Array,  # the step size of this step's update
    jax.Array,  # the number of Q-table entries this step updated
]


class TrainedTcirl(NamedTuple):
    """What a TCIRL run ends with: the pair its last re-inference leaves, and the pair it measured.

    `q_tables`, `update_counts`, `exploration_starts` and `hypotheses` are the END state: on every
    seed whose hypothesis the final batch's re-inference replaced, the table and its counts as the
    value transfer left them with every count at 0, or reset to `q_init` with the exploration
    schedule restarted under the rule "none", which is what a continued run would start
    from. `evaluated_q_tables` and `evaluated_hypotheses`
    are the pair the LAST EVALUATION POINT measured, bound before that re-inference, which is what
    the run's final numbers describe and what `checkpoints/final_policy.npz` holds. They differ
    only on a seed that re-infers in the final batch.
    """

    q_tables: jax.Array
    update_counts: jax.Array
    exploration_starts: jax.Array
    hypotheses: list[DFA]
    evaluated_q_tables: jax.Array
    evaluated_hypotheses: list[DFA]


def train(
    env: TabularEnv,
    *,
    logger: WandbLogger,
    recorder: PerSeedMetricRecorder,
    grid_alphabet: OrderedAlphabet[str],
    cause_dfa_size: int,
    cause_index: int,
    effect_index: int,
    init_hypothesis_dfas: list[DFA],
    inference_mode: InferenceMode,
    combined_to_grid: jax.Array,
    x_r_obs_ids: frozenset[int],
    x_a_obs_ids: frozenset[int],
    self_loop_symbol: int | None,
    max_dfa_size: int,
    initial_rejecting: bool,
    symmetry_breaking: bool,
    value_transfer: ValueTransfer,
    counterfactual: bool,
    gamma: float,
    step_size_scale: float,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_steps: int,
    max_episode_steps: int,
    q_init: float,
    seed_offset: int,
    num_runs: int,
    schedule: EvalSchedule,
    smoothing_steps: int | None,
    evaluation: EvalSink | None = None,
) -> TrainedTcirl:
    """Train TCIRL agents vmapped across the seed bank `seed_offset + range(num_runs)`.

    `evaluation` is where each point of the run's evaluation grid goes; `None` trains and measures
    nothing about the resulting policy, which suits a caller that only trains.

    `schedule` fixes this loop's two cadences, and they are not the same number. One JAX batch runs
    `schedule.batch_steps` env steps, which is as fine as the evaluation grid needs; the
    counterexample check and the re-inference it triggers run once every `schedule.batches_per_stop`
    batches, which is every `stop_interval` steps, the latency bound of the counterexample check. At
    a one-point grid the two coincide.

    `value_transfer` names the rule by which, at each re-inference, every state of the new
    hypothesis takes the Q-values of a state of the old one
    (`tcirl.inference.transfer`), and every update count restarts at 0. Under the rule "none" the
    whole table restarts from `q_init`, and the seed's exploration schedule restarts with it, so
    that the restarted table is learned as the first one was; the two other rules keep learned
    values in every entry, so they keep the schedule running.

    `counterfactual` also updates, at each step, every live hypothesis state whose acceptance equals
    that of the visited state, each toward the observed reward and the value of its own successor.
    The hypothesis assigns all of them the visited state's phase, so the sampled transition is drawn
    from the kernel it assigns to each of them exactly when it is drawn from the kernel it assigns
    to the visited state; the set is fixed before the transition is sampled, so it selects no outcome.

    Returns both pairs a run ends with, see `TrainedTcirl`; the checkpoint needs the evaluated pair.
    """
    if len(init_hypothesis_dfas) != num_runs:
        raise ValueError(
            f"Expected {num_runs} initial hypothesis DFAs, got {len(init_hypothesis_dfas)}"
        )

    num_actions = env.num_actions
    obs_shape = env.state_shape
    dfa_dim = cause_dfa_size

    # combined_to_grid projects obs-alphabet labels to grid-alphabet labels.
    combined_to_grid_np = np.asarray(combined_to_grid, dtype=np.int32)
    num_grid_symbols = int(combined_to_grid_np.max()) + 1

    # Q-table: (*obs_prefix, dfa_dim, *obs_suffix, num_actions)
    q_shape = (*obs_shape[:cause_index], dfa_dim, *obs_shape[cause_index:], num_actions)
    n_obs_suffix = len(obs_shape) - cause_index

    # Per-seed hypothesis cause DFA.
    hypothesis_dfas: list[DFA] = list(init_hypothesis_dfas)

    # Per-seed pending episode data (obs-alphabet labels + disease levels).
    pending_labels: list[list[int]] = [[] for _ in range(num_runs)]
    pending_effects: list[list[int]] = [[] for _ in range(num_runs)]

    # Inference-mode-specific per-seed state.
    # SAT mode: persistent causal samples + counterexample-driven solving.
    causal_samples: list[MutableCausalSample] | None = None
    per_seed_reinferences: list[int] = [0] * num_runs
    # The share of the update counts of each seed's table before its last re-inference that lie in the rows of the old states whose values some new state took.
    per_seed_kept_updates: list[float] = [0.0] * num_runs

    if inference_mode == "sat":
        causal_samples = [MutableCausalSample(num_grid_symbols) for _ in range(num_runs)]

    # Stack hypothesis arrays: the deltas (num_runs, dfa_dim, num_grid_symbols) and the class of
    # each padded state (num_runs, dfa_dim), 1 accepting, 0 rejecting and -1 for a filler state,
    # which no run reaches and no counterfactual update writes.
    def _stack_hypotheses() -> tuple[jax.Array, jax.Array]:
        deltas: list[jax.Array] = []
        classes: list[jax.Array] = []
        for dfa in hypothesis_dfas:
            d, accepting = build_padded_cause_arrays(dfa, dfa_dim)
            deltas.append(d)
            live = jnp.arange(dfa_dim) < dfa.num_states
            classes.append(jnp.where(live, accepting.astype(jnp.int32), jnp.int32(-1)))
        return jnp.stack(deltas), jnp.stack(classes)

    all_hyp_delta, all_hyp_class = _stack_hypotheses()

    # Log seed 0's initial hypothesis DFA
    seed0_hyp_count = 0
    logger.log_automaton_summary(
        DFASummaryInfo(
            dfa=hypothesis_dfas[0],
            alphabet=grid_alphabet,
            formula_str=None,
            perf=None,
            encoder=None,
            live_states=None,
        ),
        subdir="data/hypotheses",
        filename=f"hyp_{seed0_hyp_count}.txt",
        render_if=20,
    )
    seed0_hyp_count += 1

    def epsilon_schedule(step: jax.Array) -> jax.Array:
        frac = jnp.clip(step / epsilon_decay_steps, 0.0, 1.0)
        return epsilon_start + (epsilon_end - epsilon_start) * frac

    # ------------------------------------------------------------------
    # Train step: standard Q-learning with hypothesis cause state in index.
    # ------------------------------------------------------------------
    def train_step(
        carry: _TrainCarry,
        _: None,
    ) -> tuple[
        _TrainCarry,
        _TrainOutputs,
    ]:
        (
            q_table,
            st,
            hyp_cause_u,
            step,
            key,
            hyp_delta,
            hyp_class,
            ep_return,
            ep_disc_return,
            update_count,
            exploration_start,
        ) = carry
        key, ak, ek, sk, tk, rk = jax.random.split(key, num=6)
        eps = epsilon_schedule(step - exploration_start)

        # State decomposition: split around hidden cause index.
        obs_prefix = st.indices[:cause_index]
        obs_suffix = st.indices[cause_index + 1 : cause_index + 1 + n_obs_suffix]
        mask = env.action_mask(st)
        has_actions = jnp.any(mask)

        # Action selection: Q[*obs_prefix, hyp_cause, *obs_suffix]
        q_vals = q_table[(*obs_prefix, hyp_cause_u, *obs_suffix)]
        greedy_action = tabular_argmax(q_vals, mask, tk)
        safe_logits = jnp.where(has_actions, jnp.where(mask, 0.0, -jnp.inf), 0.0)
        random_action = jax.random.categorical(ak, safe_logits)
        action = jnp.where(jax.random.uniform(ek) < eps, random_action, greedy_action)

        # Step environment
        new_st, reward, terminated, _aut_terminated, _aut_crashed = env.step(st, action, sk)

        new_obs_prefix = new_st.indices[:cause_index]
        new_obs_suffix = new_st.indices[cause_index + 1 : cause_index + 1 + n_obs_suffix]
        new_effect = new_st.indices[effect_index]
        label = new_st.label

        # Step hypothesis cause DFA on grid-projected label
        grid_label = combined_to_grid[label]
        new_hyp_cause_u = hyp_delta[hyp_cause_u, grid_label]

        next_mask = env.action_mask(new_st)
        next_has_actions = jnp.any(next_mask)
        next_q = q_table[(*new_obs_prefix, new_hyp_cause_u, *new_obs_suffix)]
        masked_next_q = jnp.where(next_mask, next_q, -jnp.inf)
        bootstrap = jnp.where(next_has_actions, jnp.max(masked_next_q), 0.0)
        terminated = terminated | ~has_actions
        target = reward + jnp.asarray(
            jnp.where(terminated | ~next_has_actions, jnp.float32(0.0), gamma * bootstrap)
        )

        truncated = new_st.step_count >= max_episode_steps
        episode_over = terminated | truncated

        # The entry's own update count sets its step size, and a re-inference restarts every count,
        # so every entry of a new hypothesis's table starts its schedule again.
        if counterfactual:
            # Every live state of the visited state's acceptance class learns from this transition,
            # each toward the observed reward and the value of its own successor on the same label;
            # the class is fixed before the step, so the set selects no outcome. Row `hyp_cause_u`
            # is the factual update, and every target is read before any row is written.
            hyp_states = jnp.arange(dfa_dim, dtype=jnp.int32)
            updated = (hyp_class == hyp_class[hyp_cause_u]) & has_actions
            next_q_all = q_table[(*new_obs_prefix, hyp_delta[:, grid_label], *new_obs_suffix)]
            bootstrap_all = jnp.where(
                next_has_actions,
                jnp.max(jnp.where(next_mask, next_q_all, -jnp.inf), axis=-1),
                0.0,
            )
            targets = reward + jnp.where(
                terminated | ~next_has_actions, jnp.float32(0.0), gamma * bootstrap_all
            )
            rows = (*obs_prefix, hyp_states, *obs_suffix, action)
            current = q_table[rows]
            counts = update_count[rows] + updated.astype(jnp.int32)
            # A row that this step skips may have the count 0, whose rate c / (c - 1) is infinite at the scale 1, so its unused rate is read at count 1.
            step_sizes = rescaled_linear_step_size(jnp.maximum(counts, 1), step_size_scale)
            tds = jnp.where(updated, targets - current, 0.0)
            q_table = q_table.at[rows].set(jnp.where(updated, current + step_sizes * tds, current))
            update_count = update_count.at[rows].set(counts)
            effective_td = tds[hyp_cause_u]
            step_size = step_sizes[hyp_cause_u]
            num_updated = jnp.sum(updated.astype(jnp.float32))
        else:
            entry = (*obs_prefix, hyp_cause_u, *obs_suffix, action)
            td_error = target - q_table[entry]
            effective_td = jnp.where(has_actions, td_error, 0.0)
            count = update_count[entry] + has_actions.astype(jnp.int32)
            # Without a legal action the count stays 0 on an unvisited entry, whose rate c / (c - 1) is infinite at the scale 1, so the unused rate is read at count 1.
            step_size = rescaled_linear_step_size(jnp.maximum(count, 1), step_size_scale)
            q_table = q_table.at[entry].add(step_size * effective_td)
            update_count = update_count.at[entry].set(count)
            num_updated = has_actions.astype(jnp.float32)

        # Reset on episode_over.
        reset_st = env.reset(rk)
        final_st_tree = jax.tree.map(
            lambda a, b: jnp.where(episode_over, a, b),  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
            reset_st,
            new_st,
        )
        assert isinstance(final_st_tree, EnvState)
        final_st: EnvState = final_st_tree

        final_hyp_cause_u = jnp.where(episode_over, jnp.int32(0), new_hyp_cause_u)

        # The episode's two returns, accumulated and sampled at the same boundary as `ep_length`.
        # `step_reward` is the value the `reward` signal reports, and `new_st.step_count` is the
        # pre-reset within-episode count, 1 on an episode's first transition, so that transition's
        # discount weight is gamma ** 0.
        step_reward = jnp.where(has_actions, reward, 0.0)
        discount_weight = jnp.asarray(gamma, jnp.float32) ** (
            new_st.step_count.astype(jnp.float32) - 1.0
        )
        running_return = ep_return + step_reward
        running_disc_return = ep_disc_return + discount_weight * step_reward

        new_carry = (
            q_table,
            final_st,
            final_hyp_cause_u,
            step + 1,
            key,
            hyp_delta,
            hyp_class,
            jnp.where(episode_over, jnp.float32(0.0), running_return),
            jnp.where(episode_over, jnp.float32(0.0), running_disc_return),
            update_count,
            exploration_start,
        )
        outputs = (
            step_reward,
            jnp.abs(effective_td),
            jnp.where(has_actions, jnp.max(jnp.where(mask, q_vals, -jnp.inf)), 0.0),
            episode_over,
            jnp.where(episode_over, new_st.step_count.astype(jnp.float32), 0.0),
            jnp.where(episode_over, running_return, jnp.float32(0.0)),
            jnp.where(episode_over, running_disc_return, jnp.float32(0.0)),
            label,
            terminated,
            new_effect,
            jnp.where(has_actions, step_size, jnp.nan),
            jnp.where(has_actions, num_updated, jnp.nan),
        )
        return new_carry, outputs

    def run_batch(
        q_table: jax.Array,
        state: EnvState,
        hyp_cause_u: jax.Array,
        global_step: jax.Array,
        key: jax.Array,
        hyp_delta: jax.Array,
        hyp_class: jax.Array,
        ep_return: jax.Array,
        ep_disc_return: jax.Array,
        update_count: jax.Array,
        exploration_start: jax.Array,
    ) -> tuple[
        _TrainCarry,
        _TrainOutputs,
    ]:
        # The two accumulators are per-EPISODE and an episode crosses a batch boundary, so they are
        # carried as arguments like the env state, and an episode that straddles a boundary keeps
        # its whole return.
        return jax.lax.scan(
            train_step,
            (
                q_table,
                state,
                hyp_cause_u,
                global_step,
                key,
                hyp_delta,
                hyp_class,
                ep_return,
                ep_disc_return,
                update_count,
                exploration_start,
            ),
            None,
            length=schedule.batch_steps,
        )

    _jitted_batch_raw = jax.jit(jax.vmap(run_batch))

    def _jitted_batch(
        q_table: jax.Array,
        state: EnvState,
        hyp_cause_u: jax.Array,
        global_step: jax.Array,
        key: jax.Array,
        hyp_delta: jax.Array,
        hyp_class: jax.Array,
        ep_return: jax.Array,
        ep_disc_return: jax.Array,
        update_count: jax.Array,
        exploration_start: jax.Array,
    ) -> tuple[
        _TrainCarry,
        _TrainOutputs,
    ]:
        return _jitted_batch_raw(  # type: ignore[reportReturnType]
            q_table,
            state,
            hyp_cause_u,
            global_step,
            key,
            hyp_delta,
            hyp_class,
            ep_return,
            ep_disc_return,
            update_count,
            exploration_start,
        )

    # Initialize
    all_q_tables = jnp.full((num_runs, *q_shape), fill_value=q_init, dtype=jnp.float32)
    reset_q_table = jnp.full(q_shape, fill_value=q_init, dtype=jnp.float32)
    all_update_counts = jnp.zeros((num_runs, *q_shape), dtype=jnp.int32)
    reset_update_count = jnp.zeros(q_shape, dtype=jnp.int32)

    def _make_init_state(seed: jax.Array) -> EnvState:
        return env.reset(jax.random.key(seed))

    # One bank, `seed_offset + range(num_runs)`, keys both the env reset and the step stream, so a
    # seed's initial state and its action stream belong together at every offset.
    seeds = seed_offset + jnp.arange(num_runs, dtype=jnp.int32)
    all_states = jax.vmap(_make_init_state)(seeds)
    all_hyp_cause_u = jnp.zeros(num_runs, dtype=jnp.int32)
    all_steps = jnp.zeros(num_runs, dtype=jnp.int32)
    # The step from which each seed's exploration schedule runs: 0, then the step of its last re-inference that restarted the table.
    all_exploration_starts = jnp.zeros(num_runs, dtype=jnp.int32)
    all_keys = jax.vmap(jax.random.key)(seeds)
    # The per-episode return accumulators, carried across batches beside the env state. A
    # re-inference replaces the Q-table and the hypothesis state and keeps the env state, so the
    # episode in progress continues and these continue with it.
    all_ep_returns = jnp.zeros(num_runs, dtype=jnp.float32)
    all_ep_disc_returns = jnp.zeros(num_runs, dtype=jnp.float32)

    bl = BatchLogger(
        logger=logger,
        recorder=recorder,
        num_runs=num_runs,
        total_steps=schedule.total_steps,
        log_interval=schedule.log_interval,
        smoothing_steps=smoothing_steps,
        # Under the value transfer rule "none" each seed's schedule restarts at its re-inferences, so `inference/epsilon` logs the rate a seed uses and no global one exists.
        global_epsilon=None,
        learning_rate=None,
        q_init=q_init,
    )

    def _compute_inference_metrics() -> dict[str, jax.Array]:
        if causal_samples is not None:
            per_seed_episodes = jnp.array(
                [cs.num_traces for cs in causal_samples], dtype=jnp.float32
            )
        else:
            per_seed_episodes = jnp.zeros(num_runs, dtype=jnp.float32)
        per_seed_hyp_states = jnp.array([d.num_states for d in hypothesis_dfas], dtype=jnp.float32)
        per_seed_reinf = jnp.array(per_seed_reinferences, dtype=jnp.float32)
        return {
            "inference/traces": per_seed_episodes,
            "inference/hypothesis_states": per_seed_hyp_states,
            "inference/reinferences": per_seed_reinf,
            "inference/kept_updates": jnp.array(per_seed_kept_updates, dtype=jnp.float32),
            "inference/epsilon": epsilon_schedule(all_steps - all_exploration_starts),
        }

    initial_inf = _compute_inference_metrics()
    bl.log_initial(batch_metrics=initial_inf)

    driver = build_eval_driver(
        schedule,
        evaluation,
        env,
        _TcirlPolicy(env, cause_index=cause_index, combined_to_grid=combined_to_grid),
        gamma=gamma,
        num_runs=num_runs,
        max_episode_steps=max_episode_steps,
    )
    # The pair the most recent evaluation point measured. The loop rebinds it before each batch's
    # re-inference (see the comment there), and it is what `train` returns: the policy the run's
    # final numbers describe, never the table the last re-inference left.
    trained_policy = TcirlSeedParams(q_table=all_q_tables, hyp_delta=all_hyp_delta)
    evaluated_hypotheses: list[DFA] = list(hypothesis_dfas)
    driver.at_start(trained_policy)

    # The seeds whose episodes have refuted the current hypothesis since the last re-inference.
    # It spans BATCHES, not one batch: a `stop_interval` can hold several of them, and the whole
    # point of `batches_per_stop` is that the re-inference still happens once per `stop_interval`.
    # `pending_labels` and the causal samples already spanned batches for the same reason.
    seeds_with_new_ces: set[int] = set()

    for batch_idx in range(schedule.num_batches):
        carries, outputs = _jitted_batch(
            all_q_tables,
            all_states,
            all_hyp_cause_u,
            all_steps,
            all_keys,
            all_hyp_delta,
            all_hyp_class,
            all_ep_returns,
            all_ep_disc_returns,
            all_update_counts,
            all_exploration_starts,
        )
        (
            all_q_tables,
            all_states,
            all_hyp_cause_u,
            all_steps,
            all_keys,
            _,
            _,
            all_ep_returns,
            all_ep_disc_returns,
            all_update_counts,
            all_exploration_starts,
        ) = carries
        assert isinstance(all_states, EnvState)
        assert isinstance(all_hyp_cause_u, jax.Array)
        assert isinstance(all_steps, jax.Array)
        assert isinstance(all_keys, jax.Array)
        assert isinstance(all_ep_returns, jax.Array)
        assert isinstance(all_ep_disc_returns, jax.Array)

        # The policy this batch finished training: the Q-table and the hypothesis it was learned
        # under, which are only meaningful as a PAIR (the table is indexed by the hypothesis's
        # state). The re-inference below replaces both on every seed whose hypothesis changed --
        # the table with transferred values, or with `q_init` under the value transfer rule "none" --
        # so the evaluable policy at this step has to be bound HERE. JAX arrays are immutable and the re-inference rebinds the names rather than writing
        # through them, so this costs nothing and keeps the trained pair alive.
        #
        # A re-inference consumes no env steps, so this step evaluates the trained pair and the
        # next points evaluate the pair the re-inference left.
        #
        # This is also the ONE place where evaluating inside training differs from evaluating after
        # `train` returns, which would measure the pair the re-inference LEAVES. The two differ only
        # on a seed that re-infers in the final batch.
        #
        # The `inference/*` metrics below sit on the OTHER side of the re-inference: they are read
        # after it, so at a step where a seed re-infers, `inference/hypothesis_states` describes the
        # hypothesis the run continues with and `eval_final/*` the one the evaluated policy used.
        # They are adjacent moments of the same batch boundary.
        trained_policy = TcirlSeedParams(q_table=all_q_tables, hyp_delta=all_hyp_delta)
        # `DFA` is frozen and the re-inference assigns into the list, so a copy of the list keeps
        # the hypotheses this table was learned under.
        evaluated_hypotheses = list(hypothesis_dfas)
        (
            rewards_flat,
            td_errors_abs,
            q_maxes,
            ep_overs_flat,
            ep_lens_flat,
            ep_returns_flat,
            ep_disc_returns_flat,
            labels_flat,
            terminated_flat,
            effects_flat,
            step_sizes_flat,
            updated_states_flat,
        ) = outputs

        # -- Episode collection and counterexample detection (Python-side) --
        labels_np = np.asarray(labels_flat)
        effects_np = np.asarray(effects_flat)
        ep_overs_np = np.asarray(ep_overs_flat)
        terminated_np = np.asarray(terminated_flat)

        for s in range(num_runs):
            for t in range(schedule.batch_steps):
                label_val = int(labels_np[s, t])
                effect_val = int(effects_np[s, t])
                terminated_val = bool(terminated_np[s, t])
                truncated_val = bool(ep_overs_np[s, t]) and not terminated_val

                pending_labels[s].append(label_val)
                pending_effects[s].append(effect_val)

                if not (terminated_val or truncated_val):
                    continue

                # Episode ended. The trace holds the labels of the sampled steps, the terminating one included; the self-loops of a terminal state are never sampled, so no label follows it.
                obs_labels = tuple(pending_labels[s])

                if inference_mode == "sat" and obs_labels:
                    assert causal_samples is not None

                    # The certificates come from the full trace, as the theory reads it: a marker certifies the prefix in front of it.
                    certs = certificates(obs_labels, x_r_obs_ids, x_a_obs_ids)
                    # Project to the grid alphabet for the prefix tree and the hypothesis check.
                    grid_labels = tuple(int(combined_to_grid_np[lbl]) for lbl in obs_labels)

                    # Remove the declared self-loop symbol, which leaves the cause in its state, and move each certificate to the compressed form of the prefix it certifies.
                    if self_loop_symbol is not None:
                        grid_labels, keep = filter_symbol_from_trace(grid_labels, self_loop_symbol)
                        certs = compress_certificates(certs, keep)

                    # Check if hypothesis disagrees with marker evidence.
                    if is_counterexample(grid_labels, certs, hypothesis_dfas[s]):
                        causal_samples[s].add_trace(grid_labels, certs)
                        seeds_with_new_ces.add(s)

                pending_labels[s] = []
                pending_effects[s] = []

        # -- SAT re-inference (counterexample-driven, like JIRP of Xu et al. 2020) --
        # Once per `stop_interval`, whatever the batch size. The hypothesis is therefore constant
        # across every batch of one interval, which is what makes the counterexample test above
        # answer the same question as if a batch were a whole interval.
        at_stop = schedule.is_stop_batch(batch_idx)
        if at_stop and seeds_with_new_ces:
            assert causal_samples is not None
            for s in sorted(seeds_with_new_ces):
                sample = causal_samples[s].to_causal_sample()
                result = learn_cause_dfa(
                    sample,
                    max_states=max_dfa_size,
                    initial_rejecting=initial_rejecting,
                    symmetry_breaking=symmetry_breaking,
                )
                if result is None:
                    raise ValueError(
                        f"Failed to learn a consistent cause DFA for seed {s} with "
                        f"max_dfa_size={max_dfa_size}."
                    )
                if self_loop_symbol is not None:
                    result = force_self_loops(result, self_loop_symbol)
                per_seed_reinferences[s] += 1
                sources = transfer_sources(hypothesis_dfas[s], result, sample, value_transfer)
                if sources:
                    # Every state of the new hypothesis starts from the Q-values of its source, and the filler rows beyond the new hypothesis from `q_init`.
                    # Every update count restarts at 0, because no update has used the new hypothesis yet: a copied count would claim the samples of another state, of every new state that shares the source, and keep a wrong copied value with a small step size.
                    per_seed_kept_updates[s] = kept_update_share(
                        all_update_counts[s], sources, cause_index=cause_index
                    )
                    all_q_tables = all_q_tables.at[s].set(
                        transfer_rows(
                            all_q_tables[s],
                            sources,
                            num_states=result.num_states,
                            cause_index=cause_index,
                            fill=q_init,
                        )
                    )
                    all_update_counts = all_update_counts.at[s].set(reset_update_count)
                else:
                    per_seed_kept_updates[s] = 0.0
                    # The whole table restarts, and the exploration schedule restarts with it, so the new table is learned under the same exploration as the first one.
                    all_q_tables = all_q_tables.at[s].set(reset_q_table)
                    all_update_counts = all_update_counts.at[s].set(reset_update_count)
                    all_exploration_starts = all_exploration_starts.at[s].set(all_steps[s])
                hypothesis_dfas[s] = result

                # Replay pending labels through new hypothesis to stay synced.
                state_val = 0
                for lbl in pending_labels[s]:
                    grid_lbl = int(combined_to_grid_np[lbl])
                    state_val = result.step(state_val, grid_lbl)
                all_hyp_cause_u = all_hyp_cause_u.at[s].set(state_val)

                # Log seed 0's updated hypothesis DFA
                if s == 0:
                    logger.log_automaton_summary(
                        DFASummaryInfo(
                            dfa=result,
                            alphabet=grid_alphabet,
                            formula_str=None,
                            perf=None,
                            encoder=None,
                            live_states=None,
                        ),
                        subdir="data/hypotheses",
                        filename=f"hyp_{seed0_hyp_count}.txt",
                        render_if=20,
                    )
                    seed0_hyp_count += 1

            all_hyp_delta, all_hyp_class = _stack_hypotheses()
        if at_stop:
            seeds_with_new_ces = set()

        # -- Logging --
        # `inference/*` describes the hypothesis, which only the re-inference moves, so it keeps the
        # `stop_interval` cadence its producer has rather than repeating itself on every batch.
        inference_metrics = _compute_inference_metrics() if at_stop else None

        bl.process_batch(
            batch_idx,
            {
                "reward": rewards_flat,
                "td_abs": td_errors_abs,
                "q_max": q_maxes,
                "episode_over": ep_overs_flat.astype(jnp.float32),
                "ep_length": ep_lens_flat,
                "ep_return": ep_returns_flat,
                "ep_disc_return": ep_disc_returns_flat,
                "step_size": step_sizes_flat,
                "updated_states": updated_states_flat,
            },
            batch_metrics=inference_metrics,
        )
        driver.after_batch(batch_idx, trained_policy)
        bl.restart_timer()  # the evaluation's wall time is not the next batch's throughput

    return TrainedTcirl(
        q_tables=all_q_tables,
        update_counts=all_update_counts,
        exploration_starts=all_exploration_starts,
        hypotheses=hypothesis_dfas,
        evaluated_q_tables=trained_policy.q_table,
        evaluated_hypotheses=evaluated_hypotheses,
    )


# ---------------------------------------------------------------------------
# Evaluation (the shared greedy path, with the hypothesis cause DFA in the policy)
# ---------------------------------------------------------------------------
class TcirlSeedParams(NamedTuple):
    """One seed's evaluation parameters: its Q-table and its final hypothesis delta."""

    q_table: jax.Array
    hyp_delta: jax.Array


class _TcirlPolicy:
    """The greedy TCIRL policy: the carried state is the hypothesis cause-DFA state.

    The Q-row is read at `(*observable, hypothesis state)` -- the hypothesis axis sits after every
    observable axis, which is the layout `train` builds -- and the state advances through the
    seed's own hypothesis delta on the grid label the transition emitted.
    """

    def __init__(
        self,
        env: TabularEnv,
        *,
        cause_index: int,
        combined_to_grid: jax.Array,
    ) -> None:
        # TCIRL indexes its Q-table by the HYPOTHESIS state in place of the env's cause state, so
        # the cause must be hidden: `cause_index == len(state_shape)` says the observable axes stop
        # right before it. On an env that also exposes the cause, the slice below would skip the
        # cause and read the next index (the bookkeeping label) into the hypothesis axis, which JAX
        # clamps silently. Both TCIRL CLIs build the env with `cause_known=False`; this refuses the
        # other case rather than producing a number from a mis-indexed table.
        if cause_index != len(env.state_shape):
            raise ValueError(
                f"TCIRL needs the cause hidden: cause_index is {cause_index} but the env exposes "
                f"{len(env.state_shape)} observable components. Build the env with "
                f"cause_known=False."
            )
        self._env = env
        self._cause_index = cause_index
        self._combined_to_grid = combined_to_grid
        self._n_obs_suffix = len(env.state_shape) - cause_index

    def _q_index(self, state: EnvState, hyp_state: jax.Array) -> tuple[jax.Array, ...]:
        prefix = state.indices[: self._cause_index]
        suffix = state.indices[self._cause_index + 1 : self._cause_index + 1 + self._n_obs_suffix]
        return (*prefix, hyp_state, *suffix)

    def init(self, params: TcirlSeedParams, env_state: EnvState) -> jax.Array:
        del params, env_state
        return jnp.int32(0)

    def select(
        self,
        params: TcirlSeedParams,
        policy_state: jax.Array,
        env_state: EnvState,
        key: jax.Array,
    ) -> jax.Array:
        return tabular_argmax(
            params.q_table[self._q_index(env_state, policy_state)],
            self._env.action_mask(env_state),
            key,
        )

    def advance(
        self, params: TcirlSeedParams, policy_state: jax.Array, new_env_state: EnvState
    ) -> jax.Array:
        return params.hyp_delta[policy_state, self._combined_to_grid[new_env_state.label]]
