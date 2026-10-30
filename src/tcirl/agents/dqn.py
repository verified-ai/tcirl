# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportMissingTypeArgument=false
"""The deep cause-blind baseline: a DQN over a window of the last k observation frames.

This is the paper's DQN arm. One frame is what the tabular cause-unknown agent sees -- the
observable product state, with the cause DFA hidden -- plus the label the transition into it
emitted, and the window of the last `history_len` frames is the bounded-memory stand-in for the
cause DFA state the agent is denied (`tcirl.rl.common.history` states precisely what such a window
can and cannot represent, and why it stacks whole frames rather than labels alone).

Training and evaluation go through the shared paths: the iteration emits
`~tcirl.rl.common.step_outputs.StepOutputs`, which `qrm_logging.log_batch` and the `BatchLogger`
consume exactly as they consume the tabular iteration's output, and the greedy evaluation is
`~tcirl.rl.common.greedy_eval.build_greedy_eval` with a network policy in place of a Q-table
lookup. Nothing here reduces a metric or writes a file.
"""

from __future__ import annotations

from typing import NamedTuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from tcirl.config import ScheduleConfig
from tcirl.logging.batch import BatchLogger
from tcirl.logging.final_policy import dqn_arrays
from tcirl.logging.recorder import PerSeedMetricRecorder
from tcirl.logging.wandb import WandbLogger
from tcirl.rl.common.dqn_core import (
    DqnCarry,
    init_per_seed_params,
    make_dqn_iteration,
    make_initial_dqn_carry,
)
from tcirl.rl.common.dqn_model import DqnModel, make_optimizer
from tcirl.rl.common.eval_schedule import EvalSchedule, EvalSink, build_eval_driver
from tcirl.rl.common.history import (
    encode_window,
    frame_shape,
    init_window,
    observation_dim,
    observation_frame,
    push_window,
)
from tcirl.rl.common.qrm_logging import log_batch
from tcirl.rl.common.tabular import EnvState, TabularEnv


class TrainedDqn(NamedTuple):
    """What training produced: the shared graph definition and the per-seed parameters."""

    graphdef: nnx.GraphDef[DqnModel]
    params: nnx.GraphState


def train(
    env: TabularEnv,
    *,
    logger: WandbLogger,
    recorder: PerSeedMetricRecorder,
    num_labels: int,
    gamma: float,
    lr: float,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_steps: int,
    max_episode_steps: int,
    hidden_sizes: tuple[int, ...],
    history_len: int,
    buffer_size: int,
    batch_size: int,
    warmup_steps: int,
    target_update_steps: int,
    max_grad_norm: float | None,
    huber_delta: float,
    double_q: bool,
    huber: bool,
    num_runs: int,
    schedule: EvalSchedule,
    smoothing_steps: int | None,
    seed_offset: int = 0,
    evaluation: EvalSink | None = None,
) -> TrainedDqn:
    """Train the DQN arm, vmapped across the seed bank `seed_offset + range(num_runs)`.

    `evaluation` is where each point of the run's evaluation grid goes; `None` trains and measures
    nothing about the resulting policy, which suits a caller that only trains.

    `schedule` also fixes this loop's cadence: `batch_steps` per JAX batch, `log_interval` per logged
    point. The deep arm has no per-`stop_interval` work -- its target sync counts backward passes
    carried in the scan carry -- so cutting the batch finer to reach a grid point changes nothing.
    """
    shape = frame_shape(env.state_shape, num_labels)
    obs_dim = observation_dim(shape, history_len)
    print(
        f"DQN observation: {obs_dim} inputs "
        f"({history_len} frames x {sum(shape)} one-hots per frame, "
        f"{len(env.state_shape)} state components + the label)"
    )

    graphdef, stacked_params = init_per_seed_params(
        obs_dim=obs_dim,
        num_actions=env.num_actions,
        hidden_sizes=hidden_sizes,
        num_runs=num_runs,
        seed_offset=seed_offset,
    )
    tx = make_optimizer(lr, max_grad_norm)
    iteration = make_dqn_iteration(
        env,
        graphdef,
        tx,
        num_labels=num_labels,
        history_len=history_len,
        batch_steps=schedule.batch_steps,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_steps=epsilon_decay_steps,
        max_episode_steps=max_episode_steps,
        batch_size=batch_size,
        buffer_size=buffer_size,
        warmup_steps=warmup_steps,
        target_update_steps=target_update_steps,
        double_q=double_q,
        huber=huber,
        huber_delta=huber_delta,
    )
    carry: DqnCarry = make_initial_dqn_carry(
        env,
        stacked_params,
        tx,
        num_runs=num_runs,
        history_len=history_len,
        buffer_size=buffer_size,
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
        # The deep arm's step size is the optimizer's constant rate.
        learning_rate=lr,
        # The network's initial output is not a known constant the way an optimistic Q-table's is,
        # so the step-0 qmax row starts at 0 rather than claiming a value nothing predicts.
        q_init=0.0,
    )
    bl.log_initial()

    driver = build_eval_driver(
        schedule,
        evaluation,
        env,
        _DqnPolicy(env, graphdef, num_labels=num_labels, history_len=history_len),
        gamma=gamma,
        num_runs=num_runs,
        max_episode_steps=max_episode_steps,
    )
    driver.at_start(carry.params)

    for batch_idx in range(schedule.num_batches):
        carry, (outputs, extras) = iteration(carry)
        log_batch(
            bl,
            batch_idx,
            outputs,
            extra_signals={"loss": extras.loss, "grad_norm": extras.grad_norm},
        )
        driver.after_batch(batch_idx, carry.params)
        bl.restart_timer()  # the evaluation's wall time is not the next batch's throughput

    return TrainedDqn(graphdef=graphdef, params=carry.params)


def final_policy_arrays(trained: TrainedDqn) -> dict[str, np.ndarray]:
    """The per-seed network leaves of a finished run, as `tcirl.logging.final_policy` saves them."""
    return dqn_arrays(trained.params)


class _DqnPolicy:
    """The greedy network policy: the carried state is the frame window."""

    def __init__(
        self,
        env: TabularEnv,
        graphdef: nnx.GraphDef[DqnModel],
        *,
        num_labels: int,
        history_len: int,
    ) -> None:
        self._env = env
        self._graphdef = graphdef
        self._frame_shape = frame_shape(env.state_shape, num_labels)
        self._history_len = history_len

    def _encode(self, window: jax.Array) -> jax.Array:
        return encode_window(window, frame_shape=self._frame_shape)

    def init(self, params: nnx.GraphState, env_state: EnvState) -> jax.Array:
        del params
        return init_window(observation_frame(self._env, env_state), self._history_len)

    def select(
        self,
        params: nnx.GraphState,
        policy_state: jax.Array,
        env_state: EnvState,
        key: jax.Array,
    ) -> jax.Array:
        del key  # a network's Q-values do not sit on a plateau, so no tie-break is needed
        model = nnx.merge(self._graphdef, params)
        # The window's newest frame IS the current observation, so the state is only needed for
        # the action mask.
        q_values = model(self._encode(policy_state)[None])[0]
        mask = self._env.action_mask(env_state)
        return jnp.argmax(jnp.where(mask, q_values, -jnp.inf)).astype(jnp.int32)

    def advance(
        self, params: nnx.GraphState, policy_state: jax.Array, new_env_state: EnvState
    ) -> jax.Array:
        del params
        return push_window(policy_state, observation_frame(self._env, new_env_state))
