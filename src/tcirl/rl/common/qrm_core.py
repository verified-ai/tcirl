"""The tabular Q-learning iteration, JIT'd, vmapped over seeds and scanned over `batch_steps`.

One iteration `iteration(carry, hyp) -> (new_carry, outputs)` performs `batch_steps` env steps per seed with:
  - Epsilon-greedy action selection on `Q[*base_indices, *aut_config]`.
  - True rewards and terminations from `env.step`.
  - One TD update of the visited entry, whose step size is the rescaled linear rate of its own update count (`rl.schedules.rescaled_linear_step_size`).
  - External tracking of the hypothesis configuration via `hyp.delta[(*aut_config, label)]`; a plain env runs under the one-configuration `trivial_hyp`.

The inner scan resets env and configuration on an episode boundary, so `env.step` is never called from a terminal state.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp

from tcirl.rl.common.greedy_eval import tabular_argmax
from tcirl.rl.common.reward_automaton import RewardAutomaton
from tcirl.rl.common.tabular import EnvState, TabularEnv
from tcirl.rl.schedules import linear_schedule_value_jax, rescaled_linear_step_size


class BatchCarry(NamedTuple):
    """Per-seed JAX state carried across batches (and across scan steps)."""

    q_table: jax.Array  # (*base_state_shape, *config_shape, num_actions)
    env_state: EnvState
    aut_config: jax.Array  # (m,) int32, externally tracked hypothesis configuration
    step: jax.Array  # int32 scalar, global env-step counter
    key: jax.Array  # PRNG key
    ep_config_max: jax.Array  # (env_aut_arity,) int32, running per-episode max of the env's
    # true configuration, reset at each episode boundary. qrm_core only maxes components and never
    # interprets them.
    hyp_ep_config_max: jax.Array  # (m,) int32, the hypothesis-config analogue of ep_config_max
    # (running per-episode max of `aut_config`, reset at each episode boundary).
    ep_return: jax.Array  # float32 scalar, the running UNDISCOUNTED sum of the current episode's
    # rewards, reset at each episode boundary. Sampled at the boundary it is that episode's return
    # (the per-episode analogue of step_count -> ep_length).
    ep_disc_return: jax.Array  # float32 scalar, the same running sum with each reward weighted by
    # gamma ** (its index within the episode), so the boundary sample is the episode's discounted
    # return -- the quantity this iteration's TD target is built to maximise.
    update_count: jax.Array  # int32, the number of TD updates each Q-table entry has received,
    # shaped like `q_table`. An entry's step size is a function of its own count, so the count
    # belongs to the entry: whatever resets an entry's value resets its count.


class BatchHyp(NamedTuple):
    """Per-seed hypothesis automaton arrays consumed by the iteration.

    Fixed across the run: in this repository only Q-learning runs this iteration, on one automaton for every seed. Arrays are indexed by the configuration's axes (one per component of `config_shape`). `terminal` is per `(config, label)`: True iff the transition ends the episode because the source automaton has no run on this symbol or because the target is accepting.
    """

    delta: jax.Array  # (*config_shape, num_labels, m) int32, next configuration
    sigma: jax.Array  # (*config_shape, num_labels) float32
    terminal: jax.Array  # (*config_shape, num_labels) bool, per (config, label)
    reachable_mask: jax.Array  # (*config_shape,) bool, which configs receive Q-updates


def broadcast_hyp(hyp: BatchHyp, num_runs: int) -> BatchHyp:
    """Replicate a single hypothesis across seeds by prepending a `(num_runs,)` axis.

    Q-learning shares one automaton across all seeds; TCIRL keeps a hypothesis per seed in its own loop (`tcirl.agents.tcirl`).
    """
    out = jax.tree.map(
        lambda x: jnp.broadcast_to(x, (num_runs, *x.shape)),  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
        hyp,
    )
    assert isinstance(out, BatchHyp)
    return out


def true_automaton_hyp(aut: RewardAutomaton, num_runs: int) -> BatchHyp:
    """The env's true reward automaton as a fixed hypothesis, broadcast across seeds.

    Arity-1 only: the flat-prefix reachable mask (the first `num_states` indices are the live configurations) holds for DFA automata, while a multi-axis configuration space does not order its live configurations first. Used by Q-learning on a product env.
    """
    config_shape = aut.config_shape
    assert len(config_shape) == 1, f"expected an arity-1 automaton, got config_shape={config_shape}"
    reachable_mask = jnp.arange(config_shape[0], dtype=jnp.int32) < aut.num_states
    return broadcast_hyp(
        BatchHyp(
            delta=aut.delta,
            sigma=aut.sigma,
            terminal=aut.terminal,
            reachable_mask=reachable_mask,
        ),
        num_runs,
    )


def trivial_hyp(num_labels: int, num_runs: int) -> BatchHyp:
    """The one-configuration hypothesis for automaton-free training, broadcast across seeds.

    Plain Q-learning on a plain env drives the iteration with it: `delta` maps every label back to configuration `(0,)`, `sigma` is 0 and `terminal` False everywhere, so rewards and episode boundaries come solely from the env. `num_labels` must cover every label id the env emits (label indexing into `delta` is not range-checked under JIT).
    """
    if num_labels <= 0:
        raise ValueError(f"num_labels must be > 0, got {num_labels}")
    return broadcast_hyp(
        BatchHyp(
            delta=jnp.zeros((1, num_labels, 1), dtype=jnp.int32),
            sigma=jnp.zeros((1, num_labels), dtype=jnp.float32),
            terminal=jnp.zeros((1, num_labels), dtype=jnp.bool_),
            reachable_mask=jnp.ones((1,), dtype=jnp.bool_),
        ),
        num_runs,
    )


class BatchOutputs(NamedTuple):
    """Per-step per-seed outputs collected by scan.

    Scalar fields are (num_runs, batch_steps) after vmap+scan; per-configuration fields gain a trailing axis: `env_aut_config` and `episode_config_max` are (num_runs, batch_steps, env_aut_arity), `hyp_aut_config` and `hyp_episode_config_max` are (num_runs, batch_steps, m).
    """

    reward: jax.Array  # float32, true env reward at this step
    td_abs: jax.Array  # float32, |TD error| for the actually-visited configuration
    q_max: jax.Array  # float32, max Q over masked actions at the current state
    episode_over: jax.Array  # bool, terminated | truncated | hyp_terminated
    ep_length: jax.Array  # float32, env step count when episode_over (else 0)
    ep_return: jax.Array  # float32, the episode's undiscounted return when episode_over (else 0)
    ep_disc_return: jax.Array  # float32, the episode's discounted return when episode_over (else 0)
    label: jax.Array  # int32, the alphabet symbol emitted by this transition
    terminated: jax.Array  # bool, true env termination (not truncation)
    env_aut_config: jax.Array  # (env_aut_arity,) int32, env's true configuration after this step
    # (pre-reset; reflects the actual configuration at the step boundary).
    # A DFA configuration is (q,); the caller reads its components directly, and
    # qrm_core never inspects it.
    hyp_terminated: jax.Array  # bool, hyp.terminal[(*aut_config, label)] fired this step
    termination_ce: jax.Array  # bool, hyp_terminated & ~env_terminated_raw
    # Termination counterexample: the hypothesis says the run terminates here,
    # but the env did not. Its run would end (by crash or accept) where the env
    # continues, so the only sound action is to end the episode and treat the
    # trace prefix as a counterexample. qrm_core triggers the reset; the field is
    # computed but not read in this repository.
    env_rm_terminated: jax.Array  # bool, the env reward machine's own terminal bit
    # (accept/reject-terminal/crash) on this transition, EXCLUDING horizon and base
    # termination -- the dual of `reward`. The env-side mirror of `hyp_terminated`.
    env_rm_crashed: jax.Array  # bool, the env reward machine's no-run subset of
    # `env_rm_terminated`: the run does not exist on this transition, distinct
    # from a reject-terminal. Always False for the DFA envs of this repository.
    termination_dual_ce: jax.Array  # bool, env_rm_terminated & ~hyp_terminated
    # Dual termination counterexample: the env reward machine ended the run here
    # but the hypothesis continues (the mirror of `termination_ce`). Computed but
    # not read in this repository, and never a TD target.
    episode_config_max: jax.Array  # (env_aut_arity,) int32, running per-episode max of
    # `env_aut_config` including this step. Read at `episode_over` steps it is the episode's
    # peak true configuration.
    hyp_aut_config: jax.Array  # (m,) int32, the hypothesis configuration after this step
    # (pre-reset, hyp arity), the hypothesis mirror of `env_aut_config`.
    hyp_episode_config_max: jax.Array  # (m,) int32, running per-episode max of `hyp_aut_config`
    # including this step (the per-episode analogue of `episode_config_max` for the hypothesis).
    # Read at `episode_over` steps it is the episode's peak inferred configuration.
    step_size: jax.Array  # float32, the step size of this step's update; NaN where none ran


class IterationFn(Protocol):
    """JIT'd, vmap'd function: (carry, hyp) -> (new_carry, outputs)."""

    def __call__(self, carry: BatchCarry, hyp: BatchHyp) -> tuple[BatchCarry, BatchOutputs]: ...


def make_iteration(
    env: TabularEnv,
    *,
    base_state_shape: tuple[int, ...],
    config_shape: tuple[int, ...],
    batch_steps: int,
    gamma: float,
    step_size_scale: float,
    epsilon_start: float,
    epsilon_end: float,
    epsilon_decay_steps: int,
    max_episode_steps: int,
) -> IterationFn:
    """Build the JIT'd, vmapped scan-over-batch_steps iteration.

    The returned function takes vmap'd (per-seed) BatchCarry and BatchHyp and returns vmap'd (BatchCarry, BatchOutputs). Outputs have an additional leading scan dimension of length `batch_steps`.

    `config_shape` is the automaton's configuration product space (arity `m = len(config_shape)`). Only the visited entry `(*base, *config, action)` is updated, toward the true env reward plus the discounted successor value, which is zeroed when the env or the hypothesis ends the run. `hyp.sigma` and `hyp.reachable_mask` go unused.

    Each entry's step size is `rescaled_linear_step_size` of its own update count with the scale `step_size_scale`, so the schedule needs no clock and no horizon, and an entry that a caller resets restarts it.
    """
    n_base = len(base_state_shape)
    m = len(config_shape)
    env_aut_arity = len(env.state_shape) - n_base  # arity of the env's true configuration tail

    def epsilon_schedule(step: jax.Array) -> jax.Array:
        return linear_schedule_value_jax(step, epsilon_start, epsilon_end, epsilon_decay_steps)

    def train_step(
        carry: BatchCarry,
        hyp: BatchHyp,
    ) -> tuple[BatchCarry, BatchOutputs]:
        (
            q_table,
            st,
            aut_config,
            step,
            key,
            ep_config_max,
            hyp_ep_config_max,
            ep_return,
            ep_disc_return,
            update_count,
        ) = carry
        key, ak, ek, sk, tk, rk = jax.random.split(key, num=6)
        eps = epsilon_schedule(step)

        mask = env.action_mask(st)
        has_actions = jnp.any(mask)

        base_indices = st.indices[:n_base]
        cfg = tuple(aut_config[i] for i in range(m))  # m static scalar indices
        q_vals = q_table[(*base_indices, *cfg)]
        greedy_action = tabular_argmax(q_vals, mask, tk)

        safe_logits = jnp.where(has_actions, jnp.where(mask, 0.0, -jnp.inf), 0.0)
        random_action = jax.random.categorical(ak, safe_logits)
        action = jnp.where(jax.random.uniform(ek) < eps, random_action, greedy_action)

        # True environment step: source of truth for reward and termination.
        # `env_aut_terminated` is the env reward machine's own termination bit
        # (accept/reject-terminal/crash), excluding horizon and base; `env_aut_crashed`
        # is its no-run subset. Both are recorded in the batch outputs and never feed the TD target.
        new_st, reward, env_terminated_raw, env_aut_terminated, env_aut_crashed = env.step(
            st, action, sk
        )
        terminated = env_terminated_raw | ~has_actions

        s_prime = new_st.indices[:n_base]
        label = new_st.label

        # Step the (externally tracked) hypothesis configuration: `delta` maps a
        # configuration + label to the next configuration directly.
        new_aut_config = hyp.delta[(*cfg, label)]  # (m,)
        hyp_terminated_now = hyp.terminal[(*cfg, label)]
        termination_ce = hyp_terminated_now & ~env_terminated_raw
        # Dual termination CE: the env reward machine ended the run here, but the
        # hypothesis continues. Recorded in the batch outputs; like `termination_ce` it
        # never touches the TD target.
        termination_dual_ce = env_aut_terminated & ~hyp_terminated_now

        # `delta` holds NO_TRANSITION sentinels where no run exists, which is a subset of the
        # hypothesis-terminal transitions; their bootstrap is zeroed below, so the gather reads
        # configuration 0 there to stay in bounds.
        next_cfg = tuple(
            jnp.where(hyp_terminated_now, jnp.int32(0), new_aut_config[j]) for j in range(m)
        )
        next_mask = env.action_mask(new_st)
        next_has_actions = jnp.any(next_mask)
        next_q = q_table[(*s_prime, *next_cfg)]
        bootstrap = jnp.where(
            next_has_actions, jnp.max(jnp.where(next_mask, next_q, -jnp.inf)), 0.0
        )
        # Bootstrap is zeroed when the env OR the hypothesis ends the run here (the latter resets
        # the episode, so its successor is unreachable).
        done = env_terminated_raw | hyp_terminated_now | ~next_has_actions
        target = reward + jnp.where(done, 0.0, gamma * bootstrap)

        entry = (*base_indices, *cfg, action)
        td_error = target - q_table[entry]
        count = update_count[entry] + has_actions.astype(jnp.int32)
        step_size = rescaled_linear_step_size(count, step_size_scale)
        q_table = q_table.at[entry].add(jnp.where(has_actions, step_size * td_error, 0.0))
        update_count = update_count.at[entry].set(count)

        truncated = new_st.step_count >= max_episode_steps
        # `hyp_terminated_now` is folded in so the episode always resets when the
        # hypothesis says the run is over; otherwise the configuration would get
        # stuck at a hyp-terminal cell while the env keeps emitting labels.
        episode_over = terminated | truncated | hyp_terminated_now

        # Reset env and configuration on episode boundary.
        reset_st = env.reset(rk)
        final_st_tree = jax.tree.map(
            lambda a, b: jnp.where(episode_over, a, b),  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
            reset_st,
            new_st,
        )
        assert isinstance(final_st_tree, EnvState)
        final_st: EnvState = final_st_tree
        final_aut_config = jnp.where(episode_over, jnp.zeros((m,), jnp.int32), new_aut_config)

        q_max = jnp.max(jnp.where(mask, q_vals, -jnp.inf))

        # The `env_aut_arity` axes of the product-MDP state after the base dims are the env
        # automaton's (truth) configuration. Read them directly (pre-reset); any further
        # trailing indices are env-private bookkeeping and are not part of the configuration.
        aut_tail = new_st.indices[n_base : n_base + env_aut_arity]
        env_aut_config = jnp.stack(aut_tail) if env_aut_arity else jnp.zeros((0,), jnp.int32)
        # Running per-episode max of the true configuration, sampled at episode end then
        # reset (the per-episode analogue of step_count -> ep_length).
        episode_config_max = jnp.maximum(ep_config_max, env_aut_config)
        new_ep_config_max = jnp.where(
            episode_over, jnp.zeros_like(ep_config_max), episode_config_max
        )
        # Same per-episode running max for the hypothesis configuration (pre-reset `new_aut_config`).
        hyp_episode_config_max = jnp.maximum(hyp_ep_config_max, new_aut_config)
        new_hyp_ep_config_max = jnp.where(
            episode_over, jnp.zeros_like(hyp_ep_config_max), hyp_episode_config_max
        )

        # The episode's two returns, accumulated the same way and sampled at the same boundary as
        # `ep_length`. `step_reward` is the value the `reward` signal reports, so the two agree by
        # construction. `new_st.step_count` is the pre-reset within-episode step count, 1 on an
        # episode's first transition, so that transition's discount weight is gamma ** 0.
        step_reward = jnp.where(has_actions, reward, 0.0)
        discount_weight = jnp.asarray(gamma, jnp.float32) ** (
            new_st.step_count.astype(jnp.float32) - 1.0
        )
        running_return = ep_return + step_reward
        running_disc_return = ep_disc_return + discount_weight * step_reward
        new_ep_return = jnp.where(episode_over, jnp.float32(0.0), running_return)
        new_ep_disc_return = jnp.where(episode_over, jnp.float32(0.0), running_disc_return)

        new_carry = BatchCarry(
            q_table=q_table,
            env_state=final_st,
            aut_config=final_aut_config,
            step=step + 1,
            key=key,
            ep_config_max=new_ep_config_max,
            hyp_ep_config_max=new_hyp_ep_config_max,
            ep_return=new_ep_return,
            ep_disc_return=new_ep_disc_return,
            update_count=update_count,
        )

        outputs = BatchOutputs(
            reward=step_reward,
            td_abs=jnp.where(has_actions, jnp.abs(td_error), 0.0),
            q_max=jnp.where(has_actions, q_max, 0.0),
            episode_over=episode_over,
            ep_length=jnp.where(episode_over, new_st.step_count.astype(jnp.float32), 0.0),
            ep_return=jnp.where(episode_over, running_return, jnp.float32(0.0)),
            ep_disc_return=jnp.where(episode_over, running_disc_return, jnp.float32(0.0)),
            label=label,
            terminated=terminated,
            env_aut_config=env_aut_config,
            hyp_terminated=hyp_terminated_now,
            termination_ce=termination_ce,
            env_rm_terminated=env_aut_terminated,
            env_rm_crashed=env_aut_crashed,
            termination_dual_ce=termination_dual_ce,
            episode_config_max=episode_config_max,
            hyp_aut_config=new_aut_config,
            hyp_episode_config_max=hyp_episode_config_max,
            step_size=jnp.where(has_actions, step_size, jnp.nan),
        )
        return new_carry, outputs

    def run_batch(carry: BatchCarry, hyp: BatchHyp) -> tuple[BatchCarry, BatchOutputs]:
        # scan iterates train_step `batch_steps` times. `hyp` is captured by
        # closure and stays constant across the scan (it only changes between
        # batches, in the outer Python loop).
        def step(c: BatchCarry, _: None) -> tuple[BatchCarry, BatchOutputs]:
            return train_step(c, hyp)

        return jax.lax.scan(step, carry, None, length=batch_steps)

    _jitted = jax.jit(jax.vmap(run_batch))

    def iteration(carry: BatchCarry, hyp: BatchHyp) -> tuple[BatchCarry, BatchOutputs]:
        return _jitted(carry, hyp)  # pyright: ignore[reportUnknownVariableType]

    return iteration


def make_initial_carry(
    env: TabularEnv,
    *,
    num_runs: int,
    q_shape: tuple[int, ...],
    q_init: float,
    config_arity: int,
    seed_offset: int = 0,
) -> BatchCarry:
    """Build a vmap'd BatchCarry for `num_runs` seeds, ready for the iteration.

    The Q-table is filled with `q_init` and every update count is 0. `config_arity` is `m`, the number of configuration components. `seed_offset` shifts the per-run seeds to `seed_offset + range(num_runs)`, giving a fresh seed bank without changing `num_runs`.
    """
    q_tables = jnp.full((num_runs, *q_shape), fill_value=q_init, dtype=jnp.float32)

    def _reset(seed: jax.Array) -> EnvState:
        return env.reset(jax.random.key(seed))

    # q_tables is (num_runs, *base_state_shape, *config_shape, num_actions); recover n_base to
    # size ep_config_max to the env's true-configuration arity (matches train_step's env_aut_config).
    n_base = len(q_tables.shape) - config_arity - 2
    env_aut_arity = len(env.state_shape) - n_base
    seeds = seed_offset + jnp.arange(num_runs, dtype=jnp.int32)
    return BatchCarry(
        q_table=q_tables,
        env_state=jax.vmap(_reset)(seeds),
        aut_config=jnp.zeros((num_runs, config_arity), dtype=jnp.int32),
        step=jnp.zeros(num_runs, dtype=jnp.int32),
        key=jax.vmap(jax.random.key)(seeds),
        ep_config_max=jnp.zeros((num_runs, env_aut_arity), dtype=jnp.int32),
        # hyp_ep_config_max is sized to the hypothesis arity (config_arity), like aut_config.
        hyp_ep_config_max=jnp.zeros((num_runs, config_arity), dtype=jnp.int32),
        # Both episode-return accumulators start at zero, like the running configuration maxima.
        ep_return=jnp.zeros(num_runs, dtype=jnp.float32),
        ep_disc_return=jnp.zeros(num_runs, dtype=jnp.float32),
        update_count=jnp.zeros(q_tables.shape, dtype=jnp.int32),
    )
