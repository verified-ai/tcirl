"""Tabular Q-learning on the product of the environment with a DFA over its labels, the one learner of every tabular arm.

The agent tracks the DFA's state itself, from the label each transition emits, and its Q-table is
indexed by the observable environment state and that DFA state: `(*env.state_shape, automaton state,
action)`. The tabular arms differ only in the DFA:

- Q-learning with the cause state runs on the cause DFA, so the state it tracks is the cause state.
- Q-learning without the cause state runs on the one-state DFA, which tracks nothing.
- TCIRL starts every seed on the one-state rejecting DFA and replaces it at each re-inference
  (`tcirl.agents.tcirl.CauseInference`).

Each transition updates the visited entry, and with `counterfactual` also the entry of the same
action at every other automaton state of the visited state's phase, each toward the observed reward
and the value of its own successor on the observed label. The phase of an automaton state is its
acceptance, and the automaton state selects the transition kernel only through its phase, so the
observed transition is a sample from the kernel of each of these states exactly when it is one from
the kernel of the visited state. The set depends only on the visited state, so it is fixed before
the transition is sampled.

Each Q-table entry's step size is `rescaled_linear_step_size` of its own update count. The training
loop also drives the run's evaluation through the shared greedy evaluation, whose policy reads the
same product state (`TabularPolicy`).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp

from tcirl.automata.dfa import DFA
from tcirl.automata.plain_dfa import PlainDFA
from tcirl.config import ScheduleConfig
from tcirl.logging.batch import BatchLogger
from tcirl.logging.recorder import PerSeedMetricRecorder
from tcirl.logging.wandb import WandbLogger
from tcirl.rl.common.eval_schedule import EvalSchedule, EvalSink, build_eval_driver
from tcirl.rl.common.greedy_eval import tabular_argmax
from tcirl.rl.common.qrm_logging import log_batch
from tcirl.rl.common.tabular import EnvState, TabularEnv
from tcirl.rl.schedules import linear_schedule_value_jax, rescaled_linear_step_size


class Automata(NamedTuple):
    """Every seed's DFA as the iteration reads it, padded to one number of states.

    A filler state beyond a seed's own DFA loops on every symbol, no run reaches it, and its phase
    is -1, so no counterfactual update writes its rows.
    """

    delta: jax.Array  # (num_runs, size, num_symbols) int32
    phase: jax.Array  # (num_runs, size) int32: 1 accepting, 0 rejecting, -1 filler


def stack_automata(dfas: Sequence[DFA], size: int) -> Automata:
    """The per-seed arrays of `dfas`, each padded to `size` states."""
    deltas: list[jax.Array] = []
    phases: list[jax.Array] = []
    for dfa in dfas:
        plain = PlainDFA.from_dfa(dfa, size)
        deltas.append(plain.delta)
        live = jnp.arange(size) < dfa.num_states
        phases.append(jnp.where(live, plain.accepting.astype(jnp.int32), jnp.int32(-1)))
    return Automata(delta=jnp.stack(deltas), phase=jnp.stack(phases))


class TabularCarry(NamedTuple):
    """Per-seed state carried across steps and batches, seeds on the leading axis outside the iteration."""

    q_table: jax.Array  # (*state_shape, size, num_actions) float32
    update_count: jax.Array  # like `q_table`, int32: the number of updates each entry has received
    env_state: EnvState
    automaton_state: jax.Array  # int32
    step: jax.Array  # int32, the seed's environment steps so far
    exploration_start: jax.Array  # int32, the step from which the seed's exploration schedule runs
    key: jax.Array
    ep_return: jax.Array  # float32, the running undiscounted return of the episode in progress
    ep_disc_return: jax.Array  # float32, the same sum discounted from the episode's first step


class TabularOutputs(NamedTuple):
    """Per-step per-seed outputs, `(num_runs, batch_steps)` after vmap and scan.

    The first eight fields are `~tcirl.rl.common.step_outputs.TrainStepOutputs`.
    """

    reward: jax.Array
    td_abs: jax.Array  # |TD error| of the visited entry
    q_max: jax.Array
    episode_over: jax.Array
    ep_length: jax.Array
    ep_return: jax.Array
    ep_disc_return: jax.Array
    label: jax.Array
    terminated: jax.Array  # termination, not truncation
    step_size: jax.Array  # the step size of the visited entry's update; NaN where none ran
    updated_states: jax.Array  # the number of automaton states whose entry this step updated


type IterationFn = Callable[[TabularCarry, Automata], tuple[TabularCarry, TabularOutputs]]


def make_iteration(
    env: TabularEnv,
    *,
    label_projection: jax.Array,
    counterfactual: bool,
    batch_steps: int,
    gamma: float,
    step_size_scale: float,
    epsilon: ScheduleConfig,
    max_episode_steps: int,
) -> IterationFn:
    """`batch_steps` environment steps of every seed, JIT'd and vmapped over seeds.

    `label_projection` maps each label the environment emits to the DFA's symbol.
    """
    num_observed = len(env.state_shape)

    def train_step(carry: TabularCarry, automata: Automata) -> tuple[TabularCarry, TabularOutputs]:
        key, ak, ek, sk, tk, rk = jax.random.split(carry.key, num=6)
        eps = linear_schedule_value_jax(
            carry.step - carry.exploration_start, epsilon.start, epsilon.end, epsilon.decay_steps
        )
        st = carry.env_state
        state = carry.automaton_state
        q_table = carry.q_table
        update_count = carry.update_count

        observed = st.indices[:num_observed]
        mask = env.action_mask(st)
        has_actions = jnp.any(mask)
        q_vals = q_table[(*observed, state)]
        greedy_action = tabular_argmax(q_vals, mask, tk)
        safe_logits = jnp.where(has_actions, jnp.where(mask, 0.0, -jnp.inf), 0.0)
        random_action = jax.random.categorical(ak, safe_logits)
        action = jnp.where(jax.random.uniform(ek) < eps, random_action, greedy_action)

        new_st, reward, terminated = env.step(st, action, sk)
        terminated = terminated | ~has_actions
        new_observed = new_st.indices[:num_observed]
        symbol = label_projection[new_st.label]

        # The entries of `action` at every automaton state, each with the target of its own successor
        # on `symbol`. Row `state` is the visited entry, and every target is read before any row is
        # written.
        states = jnp.arange(automata.phase.shape[0], dtype=jnp.int32)
        same = (automata.phase == automata.phase[state]) if counterfactual else (states == state)
        updated = same & has_actions
        next_mask = env.action_mask(new_st)
        next_has_actions = jnp.any(next_mask)
        next_q = q_table[(*new_observed, automata.delta[:, symbol])]
        bootstrap = jnp.where(
            next_has_actions, jnp.max(jnp.where(next_mask, next_q, -jnp.inf), axis=-1), 0.0
        )
        targets = reward + jnp.where(
            terminated | ~next_has_actions, jnp.float32(0.0), gamma * bootstrap
        )
        rows = (*observed, states, action)
        counts = update_count[rows] + updated.astype(jnp.int32)
        # A row that this step skips may have the count 0, whose rate c / (c - 1) is infinite at the
        # scale 1, so its unused rate is read at count 1.
        step_sizes = rescaled_linear_step_size(jnp.maximum(counts, 1), step_size_scale)
        tds = jnp.where(updated, targets - q_table[rows], 0.0)
        q_table = q_table.at[rows].add(step_sizes * tds)
        update_count = update_count.at[rows].set(counts)

        truncated = new_st.step_count >= max_episode_steps
        episode_over = terminated | truncated
        reset_st = env.reset(rk)
        final_st_tree = jax.tree.map(
            lambda a, b: jnp.where(episode_over, a, b),  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
            reset_st,
            new_st,
        )
        assert isinstance(final_st_tree, EnvState)
        final_st: EnvState = final_st_tree
        new_state = automata.delta[state, symbol]

        # The episode's two returns, sampled at the same boundary as `ep_length`. `new_st.step_count`
        # is the pre-reset count within the episode, 1 on its first transition, whose discount weight
        # is therefore gamma ** 0.
        step_reward = jnp.where(has_actions, reward, 0.0)
        discount_weight = jnp.asarray(gamma, jnp.float32) ** (
            new_st.step_count.astype(jnp.float32) - 1.0
        )
        running_return = carry.ep_return + step_reward
        running_disc_return = carry.ep_disc_return + discount_weight * step_reward

        new_carry = TabularCarry(
            q_table=q_table,
            update_count=update_count,
            env_state=final_st,
            automaton_state=jnp.where(episode_over, jnp.int32(0), new_state),
            step=carry.step + 1,
            exploration_start=carry.exploration_start,
            key=key,
            ep_return=jnp.where(episode_over, jnp.float32(0.0), running_return),
            ep_disc_return=jnp.where(episode_over, jnp.float32(0.0), running_disc_return),
        )
        outputs = TabularOutputs(
            reward=step_reward,
            td_abs=jnp.abs(tds[state]),
            q_max=jnp.where(has_actions, jnp.max(jnp.where(mask, q_vals, -jnp.inf)), 0.0),
            episode_over=episode_over,
            ep_length=jnp.where(episode_over, new_st.step_count.astype(jnp.float32), 0.0),
            ep_return=jnp.where(episode_over, running_return, jnp.float32(0.0)),
            ep_disc_return=jnp.where(episode_over, running_disc_return, jnp.float32(0.0)),
            label=new_st.label,
            terminated=terminated,
            step_size=jnp.where(has_actions, step_sizes[state], jnp.nan),
            updated_states=jnp.where(has_actions, jnp.sum(updated.astype(jnp.float32)), jnp.nan),
        )
        return new_carry, outputs

    def run_batch(carry: TabularCarry, automata: Automata) -> tuple[TabularCarry, TabularOutputs]:
        def step(c: TabularCarry, _: None) -> tuple[TabularCarry, TabularOutputs]:
            return train_step(c, automata)

        return jax.lax.scan(step, carry, None, length=batch_steps)

    jitted = jax.jit(jax.vmap(run_batch))

    def iteration(carry: TabularCarry, automata: Automata) -> tuple[TabularCarry, TabularOutputs]:
        return jitted(carry, automata)  # pyright: ignore[reportUnknownVariableType]

    return iteration


class AutomatonInference(Protocol):
    """A procedure that revises each seed's DFA from the episodes the seed plays (TCIRL's cause inference)."""

    def observe(self, outputs: TabularOutputs, automata: Sequence[DFA]) -> None:
        """Read one batch's transitions, judged against the DFAs they were played under."""
        ...

    def revise(
        self, carry: TabularCarry, automata: list[DFA]
    ) -> tuple[TabularCarry, list[DFA]] | None:
        """At the end of a `stop_interval`: the carry and DFAs after the revisions the batches since the last call warrant, or None when no seed changes."""
        ...

    def metrics(self, automata: Sequence[DFA]) -> dict[str, jax.Array]:
        """The per-seed `inference/*` metrics after the latest revision."""
        ...


class TabularParams(NamedTuple):
    """One seed's greedy policy: its Q-table and the transition table of its DFA."""

    q_table: jax.Array
    delta: jax.Array


class TrainedTabular(NamedTuple):
    """The per-seed Q-tables and DFAs that the last evaluation point measured.

    A re-inference in the final batch replaces a seed's pair after that batch trained it, so these
    are the pair the final batch trained, which is what the run's final numbers describe.
    """

    q_tables: jax.Array
    automata: list[DFA]


def train(
    env: TabularEnv,
    *,
    automata: Sequence[DFA],
    automaton_size: int,
    label_projection: jax.Array,
    counterfactual: bool,
    inference: AutomatonInference | None,
    count_restart_steps: tuple[int, ...],
    logger: WandbLogger,
    recorder: PerSeedMetricRecorder,
    gamma: float,
    step_size_scale: float,
    epsilon: ScheduleConfig,
    max_episode_steps: int,
    q_init: float,
    seed_offset: int,
    num_runs: int,
    schedule: EvalSchedule,
    smoothing_steps: int | None,
    evaluation: EvalSink | None,
) -> TrainedTabular:
    """Train one tabular agent per seed of the bank `seed_offset + range(num_runs)`.

    `automata` are the seeds' DFAs at the start, all over the symbols of `label_projection`, and
    `automaton_size` is the padded size of the Q-table's automaton axis. Without `inference` they
    stay fixed; with it, `inference.revise` runs once every `stop_interval` steps
    (`schedule.batches_per_stop` batches), and the exploration schedule of a seed runs from its
    `exploration_start`, which only a revision moves.

    `count_restart_steps` are the steps at which every update count restarts at 0 while the
    Q-values stay; each must end a batch of the schedule before the end of the run.

    `evaluation` receives each point of the run's evaluation grid; `None` measures nothing.
    """
    if len(automata) != num_runs:
        raise ValueError(f"expected {num_runs} DFAs, one per seed, got {len(automata)}")
    restarts = set(count_restart_steps)
    if len(restarts) != len(count_restart_steps) or any(
        step <= 0 or step >= schedule.total_steps or step % schedule.batch_steps
        for step in count_restart_steps
    ):
        raise ValueError(
            f"count_restart_steps {count_restart_steps} must be distinct multiples of the batch size "
            f"{schedule.batch_steps} below total_steps {schedule.total_steps}"
        )

    current = list(automata)
    arrays = stack_automata(current, automaton_size)
    iteration = make_iteration(
        env,
        label_projection=label_projection,
        counterfactual=counterfactual,
        batch_steps=schedule.batch_steps,
        gamma=gamma,
        step_size_scale=step_size_scale,
        epsilon=epsilon,
        max_episode_steps=max_episode_steps,
    )

    # One bank keys both the environment reset and the step stream, so a seed's initial state and
    # its actions belong together at every offset.
    seeds = seed_offset + jnp.arange(num_runs, dtype=jnp.int32)
    q_shape = (*env.state_shape, automaton_size, env.num_actions)
    carry = TabularCarry(
        q_table=jnp.full((num_runs, *q_shape), fill_value=q_init, dtype=jnp.float32),
        update_count=jnp.zeros((num_runs, *q_shape), dtype=jnp.int32),
        env_state=jax.vmap(lambda seed: env.reset(jax.random.key(seed)))(seeds),  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
        automaton_state=jnp.zeros(num_runs, dtype=jnp.int32),
        step=jnp.zeros(num_runs, dtype=jnp.int32),
        exploration_start=jnp.zeros(num_runs, dtype=jnp.int32),
        key=jax.vmap(jax.random.key)(seeds),
        ep_return=jnp.zeros(num_runs, dtype=jnp.float32),
        ep_disc_return=jnp.zeros(num_runs, dtype=jnp.float32),
    )

    def inference_metrics() -> dict[str, jax.Array]:
        assert inference is not None
        return {
            **inference.metrics(current),
            "inference/epsilon": linear_schedule_value_jax(
                carry.step - carry.exploration_start,
                epsilon.start,
                epsilon.end,
                epsilon.decay_steps,
            ),
        }

    bl = BatchLogger(
        logger=logger,
        recorder=recorder,
        num_runs=num_runs,
        total_steps=schedule.total_steps,
        log_interval=schedule.log_interval,
        smoothing_steps=smoothing_steps,
        # A revision can restart a seed's exploration schedule, and then `inference/epsilon` logs
        # the rate each seed uses, since no run-global one exists.
        global_epsilon=epsilon if inference is None else None,
        learning_rate=None,
        q_init=q_init,
    )
    bl.log_initial(batch_metrics=inference_metrics() if inference is not None else None)

    driver = build_eval_driver(
        schedule,
        evaluation,
        env,
        TabularPolicy(env, label_projection=label_projection),
        gamma=gamma,
        num_runs=num_runs,
        max_episode_steps=max_episode_steps,
    )
    evaluated = TabularParams(q_table=carry.q_table, delta=arrays.delta)
    evaluated_automata = list(current)
    driver.at_start(evaluated)

    for batch_idx in range(schedule.num_batches):
        carry, outputs = iteration(carry, arrays)
        if (batch_idx + 1) * schedule.batch_steps in restarts:
            carry = carry._replace(update_count=jnp.zeros_like(carry.update_count))
        # The policy this batch trained, bound before a revision replaces a seed's table and DFA:
        # the table is indexed by its DFA's states, so the two are only meaningful as a pair. This
        # point evaluates the pair and the next points evaluate what the revision leaves.
        evaluated = TabularParams(q_table=carry.q_table, delta=arrays.delta)
        evaluated_automata = list(current)

        batch_metrics = None
        if inference is not None:
            inference.observe(outputs, current)
            # Once per `stop_interval`, whatever the batch size, so the DFAs stay constant across
            # every batch of one interval.
            if schedule.is_stop_batch(batch_idx):
                revised = inference.revise(carry, current)
                if revised is not None:
                    carry, current = revised
                    arrays = stack_automata(current, automaton_size)
                # `inference/*` describes the DFAs, which only a revision moves, so it keeps the
                # cadence of the revisions.
                batch_metrics = inference_metrics()

        log_batch(
            bl,
            batch_idx,
            outputs,
            batch_metrics=batch_metrics,
            extra_signals={
                "step_size": outputs.step_size,
                "updated_states": outputs.updated_states,
            },
        )
        driver.after_batch(batch_idx, evaluated)
        bl.restart_timer()  # the evaluation's wall time is not the next batch's throughput

    return TrainedTabular(q_tables=evaluated.q_table, automata=evaluated_automata)


class TabularPolicy:
    """The greedy policy on the product state: the carried state is the DFA state.

    The Q-row is read at `(*observable, DFA state)`, and the DFA state advances through the seed's
    own transition table on the symbol of the label the transition emitted. The environment's
    indices can carry hidden components after the observable ones, the cause state among them,
    which the policy never reads.
    """

    def __init__(self, env: TabularEnv, *, label_projection: jax.Array) -> None:
        self._env = env
        self._num_observed = len(env.state_shape)
        self._label_projection = label_projection

    def init(self, params: TabularParams, env_state: EnvState) -> jax.Array:
        del params, env_state
        return jnp.int32(0)

    def select(
        self,
        params: TabularParams,
        policy_state: jax.Array,
        env_state: EnvState,
        key: jax.Array,
    ) -> jax.Array:
        return tabular_argmax(
            params.q_table[(*env_state.indices[: self._num_observed], policy_state)],
            self._env.action_mask(env_state),
            key,
        )

    def advance(
        self, params: TabularParams, policy_state: jax.Array, new_env_state: EnvState
    ) -> jax.Array:
        return params.delta[policy_state, self._label_projection[new_env_state.label]]
