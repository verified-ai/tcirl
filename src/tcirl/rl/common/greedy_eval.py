"""The one greedy evaluation of this repo: `build_greedy_eval`.

Every arm is scored the same way, so there is one implementation and the arms differ only in the
policy handed to it. A policy is three closures over that seed's parameters -- what extra state the
agent carries between steps, how it picks an action, and how that state advances -- which is
exactly the difference between the three arms:

- Q-learning: no extra state; the action is the argmax of the Q-row at the observable state.
- TCIRL: the hypothesis cause-DFA state; the Q-row is indexed by it as well, and it advances
  through the hypothesis delta on the emitted label.
- DQN: the window of the last k observation frames; the action is the argmax of the network's
  output on the encoded window, and the window shifts by one.

Every training seed is evaluated on the SAME bank of episode keys (`jax.random.key(0)` split
`eval_episodes` ways), so two arms face identical environment stochasticity and a difference
between them is a difference in policy, not in luck. The bank is a constant of this module and is
never split from a training key, which is what lets a run evaluate MID-training without moving the
training it interrupts.

The carried env state FREEZES at the episode's end rather than resetting: the scan runs
`max_episode_steps` iterations to collect exactly one episode per key, so a second episode started
inside the same scan would be recorded and then masked away. Freezing makes every trailing
trajectory row a repeat of the episode's final state, which is what the masked reductions expect.
The frozen step still calls `env.step` and consumes its key, so the key stream does not depend on
when an episode happened to end.

ONE FRAME per row, which is the whole reason the rollout records the reward and the label itself
instead of leaving them to be read out of the state:

- `trajectories[t]` is the state the action was taken FROM, so row 0 is the reset state.
- `rewards[t]` is what the transition out of that state paid.
- `labels[t]` is the symbol that same transition emitted.

The markers are computed from `labels`, so `in_accept[t]` (the phase at state `t`) and the marker at row `t`
(emitted by the transition that phase governs) are the pair the marker metrics compare.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp

from tcirl.rl.common.tabular import EnvState, TabularEnv


class GreedyPolicy[S, P](Protocol):
    """One arm's greedy policy: `S` is its per-seed parameters, `P` its carried policy state."""

    def init(self, params: S, env_state: EnvState) -> P:
        """The policy state at an episode's first step."""
        ...

    def select(self, params: S, policy_state: P, env_state: EnvState, key: jax.Array) -> jax.Array:
        """The greedy action, `()` int32, over the env's legal actions."""
        ...

    def advance(self, params: S, policy_state: P, new_env_state: EnvState) -> P:
        """The policy state after one transition into `new_env_state`."""
        ...


class EvalRollout(NamedTuple):
    """What one evaluation pass produced, seeds on the leading axis.

    returns: (num_runs, eval_episodes) DISCOUNTED return, the quantity the agents maximise.
    total_returns: (num_runs, eval_episodes) UNDISCOUNTED return, the plain sum of the episode's
        rewards. Both are recorded at every evaluation point, because the discount is an
        algorithmic device and a reader compares the undiscounted sum against the episode ceiling.
    lengths: (num_runs, eval_episodes) episode length in env steps.
    trajectories: (num_runs, eval_episodes, max_steps, n_tracked) the state acted from, per step.
    active_mask: (num_runs, eval_episodes, max_steps) True while the episode is still running.
    rewards: (num_runs, eval_episodes, max_steps) what the transition out of that state paid, 0
        once the episode is over.
    labels: (num_runs, eval_episodes, max_steps) the symbol that same transition emitted.
    """

    returns: jax.Array
    total_returns: jax.Array
    lengths: jax.Array
    trajectories: jax.Array
    active_mask: jax.Array
    rewards: jax.Array
    labels: jax.Array


def build_greedy_eval[S, P](
    env: TabularEnv,
    policy: GreedyPolicy[S, P],
    *,
    gamma: float,
    num_runs: int,
    eval_episodes: int,
    max_episode_steps: int,
    trajectory_indices: tuple[int, ...] = (),
) -> Callable[[S], EvalRollout]:
    """Compile the evaluation pass ONCE and return it as a function of the per-seed parameters.

    A run evaluates at every point of its `~tcirl.rl.common.eval_schedule.EvalSchedule`. The
    parameters change value at every point and their shapes never do, so one compiled program
    serves the whole run. A fresh `jax.jit` inside the call would trace the 100-step scan once per
    point instead, which a one-point grid would not notice and a 51-point grid would.
    """
    max_steps = max_episode_steps
    traj_idx = trajectory_indices

    def eval_seed(
        params: S,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        def eval_step(
            carry: tuple[
                EnvState, P, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array
            ],
            _: None,
        ) -> tuple[
            tuple[EnvState, P, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
            tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        ]:
            (
                state,
                policy_state,
                k,
                ep_return,
                ep_total,
                discount,
                ep_length,
                episode_over,
            ) = carry
            k, sk, tk = jax.random.split(k, num=3)

            mask = env.action_mask(state)
            has_actions = jnp.any(mask)
            action = policy.select(params, policy_state, state, tk)

            new_state, reward, terminated, _aut_terminated, _aut_crashed = env.step(
                state, action, sk
            )
            terminated = terminated | ~has_actions

            active = ~episode_over & has_actions
            ep_return = ep_return + jnp.where(active, discount * reward, 0.0)
            ep_total = ep_total + jnp.where(active, reward, 0.0)
            discount = jnp.where(active, discount * gamma, discount)
            ep_length = ep_length + jnp.int32(active)

            truncated = new_state.step_count >= max_steps
            episode_over = episode_over | terminated | truncated

            # Freeze the carried env state and the policy state once the episode is over, so every
            # trailing row repeats the final state instead of recording a second episode.
            new_policy_state = policy.advance(params, policy_state, new_state)

            def hold[T](new: T, old: T) -> T:
                held: T = jax.tree.map(
                    lambda a, b: jnp.where(active, a, b),  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
                    new,
                    old,
                )
                return held

            held_state = hold(new_state, state)
            held_policy = hold(new_policy_state, policy_state)

            # One frame per row: the state acted from, and what the transition out of it produced.
            tracked = (
                jnp.stack([state.indices[i] for i in traj_idx])
                if traj_idx
                else jnp.empty(0, dtype=jnp.int32)
            )
            step_reward = jnp.where(active, reward, 0.0)
            step_label = new_state.label

            return (
                held_state,
                held_policy,
                k,
                ep_return,
                ep_total,
                discount,
                ep_length,
                episode_over,
            ), (tracked, active, step_reward, step_label)

        def run_episode(
            ep_key: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            rk, step_key = jax.random.split(ep_key)
            state = env.reset(rk)
            init_carry = (
                state,
                policy.init(params, state),
                step_key,
                jnp.float32(0.0),
                jnp.float32(0.0),
                jnp.float32(1.0),
                jnp.int32(0),
                jnp.bool_(False),
            )
            scanned = jax.lax.scan(eval_step, init_carry, None, length=max_steps)
            carry_out, (tracked, active, rewards, labels) = scanned
            _, _, _, ep_return, ep_total, _, ep_length, _ = carry_out
            return ep_return, ep_total, ep_length, tracked, active, rewards, labels

        # ONE shared bank of episode keys: the same `eval_episodes` keys for every seed, every arm
        # and every point of the evaluation grid. The bank is a constant of this function and is
        # split from `jax.random.key(0)`, never from the training key stream, so an evaluation
        # cannot move a training run.
        keys = jax.random.split(jax.random.key(0), eval_episodes)
        return jax.vmap(run_episode)(keys)

    jitted = jax.jit(jax.vmap(eval_seed))

    def run(per_seed_params: S) -> EvalRollout:
        returns, total_returns, lengths, trajectories, active_mask, rewards, labels = jitted(  # pyright: ignore[reportUnknownVariableType]
            per_seed_params
        )
        if returns.shape[0] != num_runs:
            raise ValueError(
                f"per_seed_params carries {returns.shape[0]} seeds but num_runs is {num_runs}"
            )
        return EvalRollout(
            returns=returns,
            total_returns=total_returns,
            lengths=lengths,
            trajectories=trajectories,
            active_mask=active_mask,
            rewards=rewards,
            labels=labels,
        )

    return run


def greedy_trajectory_eval[S, P](
    env: TabularEnv,
    policy: GreedyPolicy[S, P],
    per_seed_params: S,
    *,
    gamma: float,
    num_runs: int,
    eval_episodes: int,
    max_episode_steps: int,
    trajectory_indices: tuple[int, ...] = (),
) -> EvalRollout:
    """One evaluation pass: build the compiled pass and run it once.

    For a single pass, as an analysis makes. A training run evaluates at every point of its
    schedule and holds the compiled function from `build_greedy_eval` instead.
    """
    return build_greedy_eval(
        env,
        policy,
        gamma=gamma,
        num_runs=num_runs,
        eval_episodes=eval_episodes,
        max_episode_steps=max_episode_steps,
        trajectory_indices=trajectory_indices,
    )(per_seed_params)


def tabular_argmax(q_values: jax.Array, mask: jax.Array, key: jax.Array) -> jax.Array:
    """The greedy action over a tabular Q-row, with the plateau tie broken uniformly."""
    maximum = jnp.max(jnp.where(mask, q_values, -jnp.inf))
    maximizers = mask & (q_values == maximum)
    priorities = jax.random.uniform(key, q_values.shape)
    return jnp.argmax(jnp.where(maximizers, priorities, -jnp.inf)).astype(jnp.int32)
