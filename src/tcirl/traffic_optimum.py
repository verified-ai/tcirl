"""Solve the discounted traffic MDP for a stationary policy and evaluate that fixed policy on truncated episodes."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tcirl.envs.traffic_signal import ADVANCE, BURSTY_DRAWS, HOLD, TrafficSignalEnv


def _law(p: float, *, bursty: bool, q_max: int) -> NDArray[np.float64]:
    """The distribution of one draw of cars: a Bernoulli count, or the count of consecutive successes capped at `q_max`."""
    if not bursty:
        return np.asarray([1.0 - p, p])
    law = np.asarray((1.0 - p) * p ** np.arange(q_max + 1), dtype=np.float64)
    law[-1] = p**q_max
    return law


def queue_vectors(env: TrafficSignalEnv) -> NDArray[np.int64]:
    """Every queue vector `(q0, q1, q2)`, in the row-major order of the queue axes of the state."""
    size = env.q_max + 1
    return np.indices((size, size, size)).reshape(3, -1).T


def transition_matrices(env: TrafficSignalEnv) -> NDArray[np.float64]:
    """`P[phase, blocked, x, y]`: the probability that queues `x` become `y` in one step, given the phase and the intersection blocked after the action."""
    queues = queue_vectors(env)
    count = len(queues)
    size = env.q_max + 1
    matrices = np.zeros((2, 3, count, count), dtype=np.float64)
    for phase, (depart, arrive, depart_name, arrive_name) in enumerate(
        (
            (env.p_rej_depart, env.p_rej_arrive, "p_rej_depart", "p_rej_arrive"),
            (env.p_acc_depart, env.p_acc_arrive, "p_acc_depart", "p_acc_arrive"),
        )
    ):
        departures = _law(depart, bursty=depart_name in BURSTY_DRAWS, q_max=env.q_max)
        arrivals = _law(arrive, bursty=arrive_name in BURSTY_DRAWS, q_max=env.q_max)
        for blocked in range(3):
            laws = [np.asarray([1.0]) if i == blocked else departures for i in range(3)]
            for f0, p0 in enumerate(laws[0]):
                for f1, p1 in enumerate(laws[1]):
                    for f2, p2 in enumerate(laws[2]):
                        # Downstream first: a car moves only out of its queue before the step and only into room.
                        out2 = np.minimum(queues[:, 2], f2)
                        out1 = np.minimum(
                            np.minimum(queues[:, 1], f1), env.q_max - (queues[:, 2] - out2)
                        )
                        out0 = np.minimum(
                            np.minimum(queues[:, 0], f0), env.q_max - (queues[:, 1] - out1)
                        )
                        moved = queues - np.stack([out0, out1, out2], axis=1)
                        moved[:, 1] += out0
                        moved[:, 2] += out1
                        for arrived, pa in enumerate(arrivals):
                            updated = moved.copy()
                            updated[:, 0] += np.minimum(arrived, env.q_max - moved[:, 0])
                            columns = np.ravel_multi_index(updated.T, (size, size, size))
                            matrices[phase, blocked, np.arange(count), columns] += p0 * p1 * p2 * pa
    if not np.allclose(matrices.sum(axis=-1), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("a row of the traffic transition model does not sum to one")
    return matrices


def _rewards(env: TrafficSignalEnv) -> NDArray[np.float64]:
    """The reward of a step into each queue vector, `1 - max_i q_i / q_max`."""
    return 1.0 - queue_vectors(env).max(axis=1) / env.q_max


def _cause_tables(env: TrafficSignalEnv) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    """`next_cause[u, a]` after action `a` in cause state `u`, and whether `u` is accepting (Phase 2)."""
    delta = np.asarray(env.cause_delta)[:, np.asarray(env.action_labels)]
    return delta.astype(np.int64), np.asarray(env.cause_accepting, dtype=np.bool_)


def _initial_queue_index(env: TrafficSignalEnv) -> int:
    size = env.q_max + 1
    return int(np.ravel_multi_index((env.queue_init,) * 3, (size, size, size)))


def action_values(
    env: TrafficSignalEnv,
    values: NDArray[np.float64],
    matrices: NDArray[np.float64],
    *,
    gamma: float,
) -> NDArray[np.float64]:
    """Apply the discounted one-step Bellman targets to all full states and actions."""
    rewards = _rewards(env)
    next_cause, accepting = _cause_tables(env)
    count = len(rewards)
    cooldowns = env.cooldown_max + 1
    causes = len(accepting)
    columns = np.ascontiguousarray(values.transpose(1, 0, 2, 3).reshape(3, count, -1))
    expected = (matrices @ columns[None]).reshape(2, 3, count, cooldowns, causes)
    immediate = matrices @ rewards
    targets = np.zeros((*values.shape, 2), dtype=np.float64)
    for blocked in range(3):
        for cooldown in range(cooldowns):
            for cause in range(causes):
                phase = int(accepting[cause])
                targets[:, blocked, cooldown, cause, HOLD] = (
                    immediate[phase, blocked]
                    + gamma
                    * expected[phase, blocked, :, max(cooldown - 1, 0), next_cause[cause, HOLD]]
                )
                if cooldown == 0:
                    moved = (blocked + 1) % 3
                    targets[:, blocked, cooldown, cause, ADVANCE] = (
                        immediate[phase, moved]
                        + gamma
                        * expected[phase, moved, :, env.cooldown_max, next_cause[cause, ADVANCE]]
                    )
    return targets


def solve_optimum(
    env: TrafficSignalEnv, matrices: NDArray[np.float64], *, gamma: float
) -> tuple[NDArray[np.float64], NDArray[np.int8]]:
    """Solve the infinite-horizon discounted Bellman equation and return values and a stationary greedy policy."""
    if not 0 < gamma < 1:
        raise ValueError(f"gamma must be between zero and one, got {gamma}")
    values = np.zeros(
        ((env.q_max + 1) ** 3, 3, env.cooldown_max + 1, env.cause_dfa_size), dtype=np.float64
    )
    for _ in range(10_000):
        targets = action_values(env, values, matrices, gamma=gamma)
        updated = targets.max(axis=-1)
        if float(np.max(np.abs(updated - values))) <= 1e-12:
            return values, targets.argmax(axis=-1).astype(np.int8)
        values = updated
    raise RuntimeError("discounted value iteration did not reach a Bellman residual of 1e-12")


def reachable_states(
    env: TrafficSignalEnv, policy: NDArray[np.int8], matrices: NDArray[np.float64]
) -> NDArray[np.bool_]:
    """The full states that the stationary policy reaches from reset with positive probability, at any step; an advance during the mandatory hold ends the episode."""
    next_cause, accepting = _cause_tables(env)
    reached = np.zeros(policy.shape, dtype=bool)
    reached[_initial_queue_index(env), 0, 0, 0] = True
    frontier = reached.copy()
    while frontier.any():
        found = np.zeros_like(reached)
        for blocked, cooldown, cause in zip(*np.nonzero(frontier.any(axis=0)), strict=True):
            phase = int(accepting[cause])
            rows = frontier[:, blocked, cooldown, cause]
            for action in (HOLD, ADVANCE):
                taken = rows & (policy[:, blocked, cooldown, cause] == action)
                if not taken.any() or (action == ADVANCE and cooldown > 0):
                    continue
                if action == ADVANCE:
                    moved, after = (blocked + 1) % 3, env.cooldown_max
                else:
                    moved, after = blocked, max(cooldown - 1, 0)
                successors = (matrices[phase, moved][taken] > 0).any(axis=0)
                found[:, moved, after, next_cause[cause, action]] |= successors
        frontier = found & ~reached
        reached |= found
    return reached


def policy_return(
    env: TrafficSignalEnv,
    policy: NDArray[np.int8],
    horizon: int,
    matrices: NDArray[np.float64],
) -> float:
    """Evaluate the undiscounted truncated return of a fixed stationary policy from reset."""
    shape = ((env.q_max + 1) ** 3, 3, env.cooldown_max + 1, env.cause_dfa_size)
    if policy.shape != shape or np.any((policy != HOLD) & (policy != ADVANCE)):
        raise ValueError(f"policy must be a stationary action table with shape {shape}")
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    values = np.zeros(shape, dtype=np.float64)
    for _ in range(horizon):
        targets = action_values(env, values, matrices, gamma=1.0)
        values = np.take_along_axis(targets, policy[..., None], axis=-1)[..., 0]
    return float(values[_initial_queue_index(env), 0, 0, 0])


def word_return(
    env: TrafficSignalEnv, word: str, horizon: int, matrices: NDArray[np.float64]
) -> float:
    """The exact expected sum of rewards of repeating the action word (`a` advance, `h` hold) from the reset state for `horizon` steps."""
    if not word or set(word) - {"a", "h"}:
        raise ValueError(f"a word holds only the letters a and h, got {word!r}")
    rewards = _rewards(env)
    next_cause, accepting = _cause_tables(env)
    distribution = np.zeros(len(rewards), dtype=np.float64)
    distribution[_initial_queue_index(env)] = 1.0
    blocked, cooldown, cause, total = 0, 0, 0, 0.0
    for step in range(horizon):
        action = ADVANCE if word[step % len(word)] == "a" else HOLD
        if action == ADVANCE and cooldown > 0:
            return total
        phase = int(accepting[cause])
        if action == ADVANCE:
            blocked, cooldown = (blocked + 1) % 3, env.cooldown_max
        else:
            cooldown = max(cooldown - 1, 0)
        distribution = distribution @ matrices[phase, blocked]
        total += float(distribution @ rewards)
        cause = int(next_cause[cause, action])
    return total
