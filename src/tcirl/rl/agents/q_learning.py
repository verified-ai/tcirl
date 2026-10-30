"""Plain tabular Q-learning through the shared tabular iteration `qrm_core.make_iteration`.

Only the visited entry learns, from the true env reward, with the step size of its own update count. On an `AutomatonEnv`
the fixed hypothesis is the env's true automaton (`true_automaton_hyp`), so the
tracked configuration stays in lockstep with the env's product state and the
Q-table `(*base_state_shape, *config_shape, num_actions)` carries exactly the
information of a product-state table; on a plain env the trivial one-configuration
hypothesis (`trivial_hyp`) adds a size-1 configuration axis.

`train` also drives the run's EVALUATION: at each point of the `EvalSchedule` it hands the greedy
Q-table lookup to the shared eval (`~tcirl.rl.common.greedy_eval.build_greedy_eval`) and the
resulting rollout to the run's `EvalSink`. Every arm of this repo is scored by that one
implementation and differs only in the policy it supplies.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tcirl.config import ScheduleConfig
from tcirl.logging.batch import BatchLogger
from tcirl.logging.recorder import PerSeedMetricRecorder
from tcirl.logging.wandb import WandbLogger
from tcirl.rl.common.automaton_env import AutomatonEnv
from tcirl.rl.common.eval_schedule import EvalSchedule, EvalSink, build_eval_driver
from tcirl.rl.common.greedy_eval import tabular_argmax
from tcirl.rl.common.qrm_core import (
    make_initial_carry,
    make_iteration,
    trivial_hyp,
    true_automaton_hyp,
)
from tcirl.rl.common.qrm_logging import log_batch
from tcirl.rl.common.tabular import EnvState, TabularEnv


def train(
    env: TabularEnv,
    *,
    logger: WandbLogger,
    recorder: PerSeedMetricRecorder,
    gamma: float,
    step_size_scale: float,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_steps: int,
    max_episode_steps: int,
    q_init: float,
    num_runs: int,
    schedule: EvalSchedule,
    smoothing_steps: int | None,
    count_restart_steps: tuple[int, ...],
    seed_offset: int = 0,
    num_labels: int | None = None,
    evaluation: EvalSink | None = None,
) -> jax.Array:
    """Train tabular Q-learning agents vmapped across the seed bank `seed_offset + range(num_runs)`.

    Returns per-seed Q-tables of shape `(num_runs, *base_state_shape, *config_shape, num_actions)`.

    Args:
        num_labels: label-alphabet size of a plain (automaton-free) env, sizing the trivial
            hypothesis's label axis; must cover every label id the env emits. Required for
            plain envs and must be omitted for an `AutomatonEnv` (its automaton fixes it).
        schedule: the run's evaluation grid, which also fixes this loop's cadence: `batch_steps`
            per JAX batch and `log_interval` per logged point. This agent has no per-`stop_interval`
            work, so it runs one batch per iteration and nothing else depends on the cut.
        count_restart_steps: the steps at which every update count restarts at 0 while the Q-values
            stay, so that each entry learns at the step size 1 again, as after a TCIRL re-inference.
            Each must end a batch of the schedule before the end of the run.
        evaluation: where each point of the run's evaluation grid goes. `None` trains and measures
            nothing about the resulting policy, which suits a caller that only trains; every CLI passes a `~tcirl.evaluation.ScheduledEvaluation`.
    """

    if isinstance(env, AutomatonEnv):
        if num_labels is not None:
            raise ValueError("num_labels is fixed by the automaton for an AutomatonEnv; omit it")
        base_state_shape = env.base_state_shape
        config_shape = env.get_aut().config_shape
        hyp = true_automaton_hyp(env.get_aut(), num_runs)
    else:
        if num_labels is None:
            raise ValueError("a plain env does not expose its label count; pass num_labels")
        base_state_shape = env.state_shape
        config_shape = (1,)
        hyp = trivial_hyp(num_labels, num_runs)

    q_shape = (*base_state_shape, *config_shape, env.num_actions)
    restarts = set(count_restart_steps)
    if len(restarts) != len(count_restart_steps) or any(
        step <= 0 or step >= schedule.total_steps or step % schedule.batch_steps
        for step in count_restart_steps
    ):
        raise ValueError(
            f"count_restart_steps {count_restart_steps} must be distinct multiples of the batch size "
            f"{schedule.batch_steps} below total_steps {schedule.total_steps}"
        )

    iteration = make_iteration(
        env,
        base_state_shape=base_state_shape,
        config_shape=config_shape,
        batch_steps=schedule.batch_steps,
        gamma=gamma,
        step_size_scale=step_size_scale,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_steps=epsilon_decay_steps,
        max_episode_steps=max_episode_steps,
    )

    carry = make_initial_carry(
        env,
        num_runs=num_runs,
        q_shape=q_shape,
        q_init=q_init,
        config_arity=len(config_shape),
        seed_offset=seed_offset,
    )

    bl = BatchLogger(
        logger=logger,
        recorder=recorder,
        num_runs=num_runs,
        total_steps=schedule.total_steps,
        log_interval=schedule.log_interval,
        smoothing_steps=smoothing_steps,
        global_epsilon=ScheduleConfig(
            start=epsilon_start, end=epsilon_end, decay_steps=epsilon_decay_steps
        ),
        learning_rate=None,
        q_init=q_init,
    )
    bl.log_initial()

    driver = build_eval_driver(
        schedule,
        evaluation,
        env,
        _QLearningPolicy(env),
        gamma=gamma,
        num_runs=num_runs,
        max_episode_steps=max_episode_steps,
    )
    driver.at_start(carry.q_table)

    for batch_idx in range(schedule.num_batches):
        carry, outputs = iteration(carry, hyp)
        if (batch_idx + 1) * schedule.batch_steps in restarts:
            carry = carry._replace(update_count=jnp.zeros_like(carry.update_count))
        log_batch(bl, batch_idx, outputs, extra_signals={"step_size": outputs.step_size})
        driver.after_batch(batch_idx, carry.q_table)
        bl.restart_timer()  # the evaluation's wall time is not the next batch's throughput

    return carry.q_table


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
class _QLearningPolicy:
    """The greedy tabular policy: no carried state, the Q-row is read at the observable state.

    A plain env's `EnvState.indices` may carry hidden trailing components past the observable ones
    (both tcirl envs carry the cause state and the last label there), so the lookup slices the
    observable prefix instead of consuming `indices` whole; an `AutomatonEnv`'s indices are exactly
    `(*observable_base, *config)` and align 1:1 with the Q-table axes. Consuming `indices` whole
    would mis-index the Q-table on every env with hidden state.
    """

    def __init__(self, env: TabularEnv) -> None:
        self._env = env
        self._product = isinstance(env, AutomatonEnv)
        self._n_obs = len(env.state_shape)

    def _q_index(self, state: EnvState) -> tuple[jax.Array, ...]:
        if self._product:
            return state.indices
        return (*state.indices[: self._n_obs], jnp.int32(0))

    def init(self, params: jax.Array, env_state: EnvState) -> None:
        del params, env_state
        return

    def select(
        self, params: jax.Array, policy_state: None, env_state: EnvState, key: jax.Array
    ) -> jax.Array:
        del policy_state
        return tabular_argmax(
            params[self._q_index(env_state)], self._env.action_mask(env_state), key
        )

    def advance(self, params: jax.Array, policy_state: None, new_env_state: EnvState) -> None:
        del params, new_env_state
        return policy_state
