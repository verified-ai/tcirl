# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownLambdaType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false
"""The deep baseline's training iteration: a replay-buffer DQN over a `TabularEnv`.

`make_dqn_iteration` builds one JIT'd function, vmapped over seeds and scanned over
`batch_steps` steps: `iteration(carry) -> (carry, outputs)`. It is the deep mirror of
`agents.tabular.make_iteration` and emits the same per-step training quantities
(`~tcirl.rl.common.step_outputs.StepOutputs`), so the metric stack -- `qrm_logging.log_batch`, the
`BatchLogger`, the recorder, the registry and the exporter -- applies to the deep and the tabular
arms without a second code path.

The agent is cause-blind: its state is a window of the last `k` observation frames
(`tcirl.rl.common.history`), whose newest slot is the current observable state, and nothing
anywhere in this module knows that a cause DFA exists. That is the point of the arm.

A replay row stores the index window, not the encoded observation. The encoding is one-hot and
positional, so rebuilding it inside the loss costs a gather and saves an order of magnitude of
memory per row (12.8 times in the gene domain at k = 4) -- which is what keeps 50 seeds times a
100000-row buffer under a gigabyte.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

from tcirl.rl.common.dqn_model import DqnModel, global_norm, huber_or_squared, make_dqn_model
from tcirl.rl.common.history import (
    encode_window,
    frame_dim,
    frame_shape,
    init_window,
    observation_frame,
    push_window,
)
from tcirl.rl.common.step_outputs import StepOutputs
from tcirl.rl.common.tabular import EnvState, TabularEnv
from tcirl.rl.replay_buffer import ReplayBufferState, buffer_add, buffer_init, buffer_sample
from tcirl.rl.schedules import linear_schedule_value_jax


class DqnTransition(NamedTuple):
    """One replay row, stored as indices: the encoding is rebuilt from it inside the loss."""

    window: jax.Array  # (history_len, frame_dim) int32, the frames the action was taken from
    action: jax.Array  # () int32
    reward: jax.Array  # () float32, the env reward this transition paid
    terminated: jax.Array  # () bool, true env termination (never truncation)
    next_window: jax.Array  # (history_len, frame_dim) int32
    next_mask: jax.Array  # (num_actions,) bool, the legal actions at the successor state


class DqnCarry(NamedTuple):
    """Per-seed state carried across steps and batches."""

    params: nnx.GraphState
    target_params: nnx.GraphState
    opt_state: optax.OptState
    buffer: ReplayBufferState[DqnTransition]
    env_state: EnvState
    window: jax.Array  # (history_len, frame_dim) int32
    step: jax.Array  # () int32, agent steps taken so far
    backward_passes: jax.Array  # () int32
    ep_return: jax.Array  # () float32, the running UNDISCOUNTED sum of the current episode's
    # rewards, reset at each episode boundary; sampled at the boundary it is that episode's return.
    ep_disc_return: jax.Array  # () float32, the same sum with each reward weighted by
    # gamma ** (its index within the episode), so the boundary sample is the discounted return.
    key: jax.Array


class DqnStepExtras(NamedTuple):
    """Per-step per-seed DQN diagnostics, NaN on a step with no gradient step."""

    loss: jax.Array
    grad_norm: jax.Array


# The learn step's carry and its result, named once so the two `lax.cond` branches agree.
_LearnCarry = tuple[nnx.GraphState, optax.OptState]
_LearnResult = tuple[_LearnCarry, tuple[jax.Array, jax.Array, jax.Array]]


class DqnIterationFn(Protocol):
    def __call__(self, carry: DqnCarry) -> tuple[DqnCarry, tuple[StepOutputs, DqnStepExtras]]: ...


def make_dqn_iteration(
    env: TabularEnv,
    graphdef: nnx.GraphDef[DqnModel],
    tx: optax.GradientTransformation,
    *,
    num_labels: int,
    history_len: int,
    batch_steps: int,
    gamma: float,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_steps: int,
    max_episode_steps: int,
    batch_size: int,
    buffer_size: int,
    warmup_steps: int,
    target_update_steps: int,
    double_q: bool,
    huber: bool,
    huber_delta: float,
    update_q: bool = True,
) -> DqnIterationFn:
    """Build the iteration. `update_q=False` gives a greedy rollout: no buffer writes, no updates.

    `target_update_steps` counts BACKWARD PASSES, not env steps; with one gradient step per env
    step past `warmup_steps` the two agree up to the warm-up offset. `warmup_steps` counts env
    steps and is also the minimum buffer fill: no gradient step runs before the buffer holds
    `max(warmup_steps, batch_size)` rows, so the first batches are not drawn from a handful of rows.
    """
    if history_len < 1:
        raise ValueError(f"history_len must be >= 1, got {history_len}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if buffer_size < batch_size:
        raise ValueError(f"buffer_size ({buffer_size}) must be >= batch_size ({batch_size})")
    if warmup_steps > buffer_size:
        raise ValueError(
            f"warmup_steps ({warmup_steps}) must be <= buffer_size ({buffer_size}); the buffer "
            f"saturates there, so the learn gate could never open and no gradient step would run"
        )
    if target_update_steps < 1:
        raise ValueError(f"target_update_steps must be >= 1, got {target_update_steps}")
    shape = frame_shape(env.state_shape, num_labels)
    min_fill = max(warmup_steps, batch_size)

    def encode(window: jax.Array) -> jax.Array:
        return encode_window(window, frame_shape=shape)

    def epsilon_schedule(step: jax.Array) -> jax.Array:
        return linear_schedule_value_jax(step, epsilon_start, epsilon_end, epsilon_decay_steps)

    def dqn_loss(
        params: nnx.GraphState,
        target_params: nnx.GraphState,
        batch: DqnTransition,
    ) -> tuple[jax.Array, jax.Array]:
        """`(loss, mean |TD error|)` over the batch."""
        model = nnx.merge(graphdef, params)
        target_model = nnx.merge(graphdef, target_params)
        rows = jnp.arange(batch.action.shape[0])
        obs = jax.vmap(encode)(batch.window)
        next_obs = jax.vmap(encode)(batch.next_window)

        q_all = model(obs)
        q_sel = q_all[rows, batch.action]
        # The bootstrap maximizes over the LEGAL successor actions only, matching the behaviour
        # policy and the greedy evaluation. Both shipped envs allow every action; an env with a
        # real mask would otherwise bootstrap from an action the agent can never take.
        q_next_target = jnp.where(batch.next_mask, target_model(next_obs), -jnp.inf)
        if double_q:
            # The online net picks the successor action, the target net scores it: the standard
            # correction for the max operator's overestimation bias.
            q_next_online = jnp.where(batch.next_mask, model(next_obs), -jnp.inf)
            a_star = jnp.argmax(jax.lax.stop_gradient(q_next_online), axis=1)
        else:
            a_star = jnp.argmax(q_next_target, axis=1)
        v_next = jnp.where(batch.terminated, 0.0, q_next_target[rows, a_star])
        target = jax.lax.stop_gradient(batch.reward + gamma * v_next)
        per_row = huber_or_squared(q_sel, target, delta=huber_delta, huber=huber)
        return jnp.mean(per_row), jnp.mean(jnp.abs(q_sel - target))

    def train_step(carry: DqnCarry, _: None) -> tuple[DqnCarry, tuple[StepOutputs, DqnStepExtras]]:
        key, k_explore, k_action, k_env, k_reset, k_learn = jax.random.split(carry.key, num=6)
        eps = epsilon_schedule(carry.step)

        obs = encode(carry.window)
        model = nnx.merge(graphdef, carry.params)
        q_vals = model(obs[None])[0]
        mask = env.action_mask(carry.env_state)
        masked_q = jnp.where(mask, q_vals, -jnp.inf)
        q_max = jnp.max(masked_q)
        greedy_action = jnp.argmax(masked_q)
        # A uniform draw over the LEGAL actions, so exploration never proposes a masked one.
        random_action = jax.random.categorical(k_action, jnp.where(mask, 0.0, -jnp.inf))
        explore = jax.random.uniform(k_explore) < eps
        action = jnp.where(explore, random_action, greedy_action).astype(jnp.int32)

        new_env_state, reward, terminated = env.step(
            carry.env_state, action, k_env
        )
        label = new_env_state.label
        new_window = push_window(carry.window, observation_frame(env, new_env_state))

        truncated = new_env_state.step_count >= max_episode_steps
        episode_over = terminated | truncated

        buffer = carry.buffer
        if update_q:
            # `terminated` and not `episode_over`: a horizon truncation is not a terminal state, so
            # its bootstrap must NOT be zeroed. Zeroing it teaches the agent that the value of a
            # perfectly good state is its one-step reward, which on a dense-reward task with a
            # 100-step horizon is the difference between a policy and noise.
            buffer = buffer_add(
                buffer,
                DqnTransition(
                    window=carry.window,
                    action=action,
                    reward=reward,
                    terminated=terminated,
                    next_window=new_window,
                    next_mask=env.action_mask(new_env_state),
                ),
                buffer_size,
            )

        def _do_learn(args: _LearnCarry) -> _LearnResult:
            params, opt_state = args
            batch = buffer_sample(buffer, k_learn, batch_size)
            (loss, td_abs), grads = jax.value_and_grad(dqn_loss, has_aux=True)(
                params, carry.target_params, batch
            )
            grad_norm = global_norm(grads)
            updates, new_opt_state = tx.update(grads, opt_state, params)
            new_params: nnx.GraphState = optax.apply_updates(params, updates)  # pyright: ignore[reportAssignmentType]
            return (new_params, new_opt_state), (loss, grad_norm, td_abs)

        def _no_learn(args: _LearnCarry) -> _LearnResult:
            nan = jnp.float32(jnp.nan)
            return args, (nan, nan, nan)

        if update_q:
            should_learn = buffer.size >= min_fill
            (params, opt_state), (loss, grad_norm, td_abs) = jax.lax.cond(
                should_learn, _do_learn, _no_learn, (carry.params, carry.opt_state)
            )
            new_bp = carry.backward_passes + should_learn.astype(jnp.int32)
            # Sync when the pass counter crosses a multiple of the period, so the cadence is in
            # backward passes and a warm-up of any length cannot shift it.
            should_sync = (new_bp // target_update_steps) > (
                carry.backward_passes // target_update_steps
            )
            target_params = jax.tree.map(
                lambda p, tp: jnp.where(should_sync, p, tp), params, carry.target_params
            )
        else:
            params = carry.params
            opt_state = carry.opt_state
            target_params = carry.target_params
            new_bp = carry.backward_passes
            loss = jnp.float32(jnp.nan)
            grad_norm = jnp.float32(jnp.nan)
            td_abs = jnp.float32(jnp.nan)

        # The episode's two returns, accumulated and sampled at the same boundary as `ep_length`.
        # `new_env_state.step_count` is the pre-reset within-episode count, 1 on an episode's first
        # transition, so that transition's discount weight is gamma ** 0.
        discount_weight = jnp.asarray(gamma, jnp.float32) ** (
            new_env_state.step_count.astype(jnp.float32) - 1.0
        )
        running_return = carry.ep_return + reward
        running_disc_return = carry.ep_disc_return + discount_weight * reward
        new_ep_return = jnp.where(episode_over, jnp.float32(0.0), running_return)
        new_ep_disc_return = jnp.where(episode_over, jnp.float32(0.0), running_disc_return)

        reset_state = env.reset(k_reset)
        final_env_state_tree = jax.tree.map(
            lambda a, b: jnp.where(episode_over, a, b), reset_state, new_env_state
        )
        assert isinstance(final_env_state_tree, EnvState)
        final_env_state: EnvState = final_env_state_tree
        final_window = jnp.where(
            episode_over, init_window(observation_frame(env, reset_state), history_len), new_window
        )

        new_carry = DqnCarry(
            params=params,
            target_params=target_params,
            opt_state=opt_state,
            buffer=buffer,
            env_state=final_env_state,
            window=final_window,
            step=carry.step + 1,
            backward_passes=new_bp,
            ep_return=new_ep_return,
            ep_disc_return=new_ep_disc_return,
            key=key,
        )
        outputs = StepOutputs(
            reward=reward,
            td_abs=td_abs,
            q_max=q_max,
            episode_over=episode_over,
            ep_length=jnp.where(
                episode_over, new_env_state.step_count.astype(jnp.float32), jnp.float32(0.0)
            ),
            ep_return=jnp.where(episode_over, running_return, jnp.float32(0.0)),
            ep_disc_return=jnp.where(episode_over, running_disc_return, jnp.float32(0.0)),
            label=label,
        )
        return new_carry, (outputs, DqnStepExtras(loss=loss, grad_norm=grad_norm))

    def run_batch(carry: DqnCarry) -> tuple[DqnCarry, tuple[StepOutputs, DqnStepExtras]]:
        return jax.lax.scan(train_step, carry, None, length=batch_steps)

    jitted = jax.jit(jax.vmap(run_batch))

    def iteration(carry: DqnCarry) -> tuple[DqnCarry, tuple[StepOutputs, DqnStepExtras]]:
        return jitted(carry)

    return iteration


def seed_keys(seed_offset: int, index: int | jax.Array) -> tuple[jax.Array, jax.Array]:
    """`(init_key, step_key)` for one seed of the bank `seed_offset + range(num_runs)`.

    One root key per seed, split once, so the network initialization and the env/action stream are
    independent draws of one bank. Every arm of every domain keys its seeds the same way, which is
    what makes seed `i` comparable across arms.
    """
    root = jax.random.key(seed_offset + index)
    k_init, k_step = jax.random.split(root)
    return k_init, k_step


def init_per_seed_params(
    *,
    obs_dim: int,
    num_actions: int,
    hidden_sizes: tuple[int, ...],
    num_runs: int,
    seed_offset: int,
) -> tuple[nnx.GraphDef[DqnModel], nnx.GraphState]:
    """The graph definition plus per-seed initializations stacked on a leading seed axis.

    Each seed initializes from its own key, so the arm's reported spread includes initialization
    variance; one shared initialization would understate it. The loop runs on the host because an
    `nnx.Module` is built in Python and cannot be constructed inside `jax.vmap`; it builds three
    dense layers `num_runs` times, which is nothing beside one training batch.
    """
    graphdef: nnx.GraphDef[DqnModel] | None = None
    per_seed: list[nnx.GraphState] = []
    for index in range(num_runs):
        k_init, _ = seed_keys(seed_offset, index)
        graphdef, params = make_dqn_model(
            obs_dim=obs_dim,
            num_actions=num_actions,
            hidden_sizes=hidden_sizes,
            key=k_init,
        )
        per_seed.append(params)
    assert graphdef is not None  # num_runs >= 1 is enforced by RunConfig
    stacked = jax.tree.map(lambda *leaves: jnp.stack(leaves), *per_seed)
    return graphdef, stacked


def make_initial_dqn_carry(
    env: TabularEnv,
    stacked_params: nnx.GraphState,
    tx: optax.GradientTransformation,
    *,
    num_runs: int,
    history_len: int,
    buffer_size: int,
    seed_offset: int,
) -> DqnCarry:
    """One vmapped carry for `num_runs` seeds, each with its own network, buffer and env stream.

    `stacked_params` comes from `init_per_seed_params` and already carries the seed axis; the
    target network starts equal to the online one, as a DQN's does.
    """
    width = frame_dim(env.state_shape)
    indices = jnp.arange(num_runs, dtype=jnp.int32)

    template = DqnTransition(
        window=jnp.zeros((history_len, width), dtype=jnp.int32),
        action=jnp.int32(0),
        reward=jnp.float32(0.0),
        terminated=jnp.bool_(False),
        next_window=jnp.zeros((history_len, width), dtype=jnp.int32),
        next_mask=jnp.ones(env.num_actions, dtype=jnp.bool_),
    )

    def one_seed(index: jax.Array, params: nnx.GraphState) -> DqnCarry:
        # `seed_keys` is the ONE derivation; repeating it here by hand is how the network
        # initialization and the env stream would come to be drawn from different banks.
        _k_init, k_step = seed_keys(seed_offset, index)
        k_reset, k_carry = jax.random.split(k_step)
        env_state = env.reset(k_reset)
        return DqnCarry(
            params=params,
            target_params=params,
            opt_state=tx.init(params),
            buffer=buffer_init(template, buffer_size),
            env_state=env_state,
            window=init_window(observation_frame(env, env_state), history_len),
            step=jnp.int32(0),
            backward_passes=jnp.int32(0),
            ep_return=jnp.float32(0.0),
            ep_disc_return=jnp.float32(0.0),
            key=k_carry,
        )

    carry_tree = jax.vmap(one_seed)(indices, stacked_params)
    assert isinstance(carry_tree, DqnCarry)
    return carry_tree
