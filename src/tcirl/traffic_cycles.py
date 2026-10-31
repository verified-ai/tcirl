"""Select traffic cycles in a finite screen and evaluate their executable feedback policies."""

from __future__ import annotations

import csv
import json
import os
import platform
import shlex
import shutil
import subprocess
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import version
from itertools import product
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tcirl.config import TrafficSignalQLConfig
from tcirl.envs.traffic_signal import ADVANCE, HOLD, NUM_INTERSECTIONS, TrafficSignalEnv
from tcirl.metrics.stats import compute_mean
from tcirl.path_consts import ROOT_DIR
from tcirl.rl.common.greedy_eval import build_greedy_eval
from tcirl.rl.common.tabular import EnvState
from tcirl.traffic_optimum import (
    action_values,
    policy_return,
    reachable_states,
    solve_optimum,
    transition_matrices,
    word_return,
)

type Json = bool | int | float | str | list[Json] | dict[str, Json] | None


def validate_comparison_settings(
    selected: TrafficSignalQLConfig, trained: TrafficSignalQLConfig
) -> None:
    """Reject a comparison with a different task, discount, episode limit or evaluation bank size."""
    if (
        selected.env != trained.env
        or selected.params.gamma != trained.params.gamma
        or selected.params.max_episode_steps != trained.params.max_episode_steps
        or selected.run.eval_episodes != trained.run.eval_episodes
    ):
        raise ValueError(
            "cycle settings differ from the trained traffic task; select its configuration with --config_path"
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class CycleSettings:
    """The finite selection screen and its independent final evaluation banks."""

    max_period: int
    horizon: int
    coarse_episodes: int
    fine_episodes: int
    nominees: int
    holdout_episodes: int
    comparison_episodes: int

    def __post_init__(self) -> None:
        if min(asdict(self).values()) < 1:
            raise ValueError("cycle settings must be positive")
        if (
            min(
                self.coarse_episodes,
                self.fine_episodes,
                self.holdout_episodes,
                self.comparison_episodes,
            )
            < 2
        ):
            raise ValueError("at least two episodes are needed for a standard error")
        if self.horizon < self.max_period:
            raise ValueError("horizon must cover the largest candidate period")


@dataclass(frozen=True, slots=True, kw_only=True)
class CycleCell:
    """One selected cycle and the two independent evaluation samples of its feedback policy."""

    frames: int
    with_cause: bool
    word: str
    candidates: int
    refined: int
    selection_mean: float
    holdout_returns: np.ndarray
    comparison_returns: np.ndarray
    rules: int


@dataclass(frozen=True, slots=True, kw_only=True)
class ControlFacts:
    """The deterministic control frames and cause states along each candidate cycle."""

    frames: np.ndarray
    causes: np.ndarray
    cause_free: np.ndarray


def enumerate_cycles(*, max_period: int, horizon: int, cooldown_max: int) -> np.ndarray:
    """Enumerate distinct legal action sequences in lexicographic order."""
    sequences: set[tuple[int, ...]] = set()
    for period in range(1, max_period + 1):
        for word in product((HOLD, ADVANCE), repeat=period):
            if all(action == HOLD for action in word) and period != 1:
                continue
            if all(
                word[(i - d) % period] == HOLD
                for i, action in enumerate(word)
                if action == ADVANCE
                for d in range(1, cooldown_max + 1)
            ):
                sequences.add(tuple((word * (horizon // period + 1))[:horizon]))
    return np.asarray(sorted(sequences), dtype=np.int32)


def control_facts(env: TrafficSignalEnv, sequences: np.ndarray) -> ControlFacts:
    """Build red-light and mandatory-hold frames before each action, with the reset frame at time zero."""
    count, horizon = sequences.shape
    frames = np.zeros((count, horizon), dtype=np.int64)
    causes = np.zeros((count, horizon), dtype=np.int64)
    blocked = np.zeros(count, dtype=np.int64)
    cooldown = np.zeros(count, dtype=np.int64)
    cause = np.zeros(count, dtype=np.int64)
    cause_free = np.ones(count, dtype=bool)
    delta = np.asarray(env.cause_delta)
    accepting = np.asarray(env.cause_accepting)
    action_labels = np.asarray(env.action_labels)
    for time in range(horizon):
        frames[:, time] = blocked * (env.cooldown_max + 1) + cooldown
        causes[:, time] = cause
        action = sequences[:, time]
        if np.any((action == ADVANCE) & (cooldown > 0)):
            raise ValueError("a candidate advances during the mandatory hold")
        blocked = (blocked + action) % NUM_INTERSECTIONS
        cooldown = np.where(action == ADVANCE, env.cooldown_max, np.maximum(cooldown - 1, 0))
        cause = delta[cause, action_labels[action]]
        cause_free &= ~accepting[cause]
    return ControlFacts(frames=frames, causes=causes, cause_free=cause_free)


def expressible_mask(
    sequences: np.ndarray,
    facts: ControlFacts,
    *,
    frames: int,
    frame_count: int,
    cause_states: int,
    with_cause: bool,
) -> np.ndarray:
    """Test exactly whether each reset-filled frame window assigns only one action to each input."""
    if frame_count**frames * cause_states * 2 > np.iinfo(np.int64).max:
        raise ValueError("the exact window encoding exceeds int64")
    codes = np.zeros_like(facts.frames)
    times = np.arange(int(sequences.shape[1]), dtype=np.int64)
    for age in range(frames - 1, -1, -1):
        codes = codes * frame_count + facts.frames[:, np.maximum(times - age, 0)]
    if with_cause:
        codes = codes * cause_states + facts.causes

    def unique_count(values: np.ndarray) -> np.ndarray:
        ordered = np.sort(values, axis=1)
        return 1 + np.count_nonzero(ordered[:, 1:] != ordered[:, :-1], axis=1)

    return unique_count(codes) == unique_count(codes * 2 + sequences)


def screen_samples(
    env: TrafficSignalEnv, sequences: np.ndarray, *, episodes: int, seed: int
) -> np.ndarray:
    """Measure the finite screen's cycles under common episode keys with its declared random stream."""

    def one(actions: jax.Array, episode_key: jax.Array) -> jax.Array:
        reset_key, step_key = jax.random.split(episode_key)

        def step(
            carry: tuple[EnvState, jax.Array, jax.Array], action: jax.Array
        ) -> tuple[tuple[EnvState, jax.Array, jax.Array], None]:
            state, key, total = carry
            key, env_key = jax.random.split(key)
            new_state, reward, terminated = env.step(state, action, env_key)
            # Every enumerated word respects the cooldown, the task's only termination condition.
            total = total + jnp.where(terminated, 0.0, reward)
            return (new_state, key, total), None

        (_, _, total), _ = jax.lax.scan(
            step, (env.reset(reset_key), step_key, jnp.float32(0)), actions
        )
        return total

    evaluate = jax.jit(jax.vmap(jax.vmap(one, in_axes=(None, 0)), in_axes=(0, None)))
    keys = jax.random.split(jax.random.key(seed), episodes)
    return np.asarray(evaluate(jnp.asarray(sequences), keys), dtype=np.float64)


def shortest_word(sequence: np.ndarray) -> str:
    """Find the shortest period that generates an action sequence."""
    for period in range(1, len(sequence) + 1):
        if all(sequence[i] == sequence[i % period] for i in range(len(sequence))):
            return "".join("a" if action == ADVANCE else "h" for action in sequence[:period])
    raise AssertionError("every finite sequence is its own period")


def feedback_table(
    env: TrafficSignalEnv, word: str, *, frames: int, with_cause: bool
) -> tuple[np.ndarray, int]:
    """Construct the feedback rules and prove their consistency until the complete deterministic state repeats."""
    if not word or set(word) - {"a", "h"}:
        raise ValueError("a cycle word must contain only a and h")
    base = NUM_INTERSECTIONS * (env.cooldown_max + 1)
    cause_states = env.cause_dfa_size if with_cause else 1
    table = np.full(base**frames * cause_states, -1, dtype=np.int32)
    blocked, cooldown, cause = 0, 0, 0
    history = [0] * frames
    seen: set[tuple[int, int, int, int, tuple[int, ...]]] = set()
    rules: set[int] = set()
    delta, accepting, labels = (
        np.asarray(env.cause_delta),
        np.asarray(env.cause_accepting),
        np.asarray(env.action_labels),
    )
    time = 0
    while (state := (time % len(word), blocked, cooldown, cause, tuple(history))) not in seen:
        seen.add(state)
        code = 0
        for frame in history:
            code = base * code + frame
        if with_cause:
            code = code * cause_states + cause
        action = ADVANCE if word[time % len(word)] == "a" else HOLD
        if table[code] >= 0 and table[code] != action:
            raise ValueError(
                f"{word} is not executable with {frames} frames and with_cause={with_cause}"
            )
        table[code] = action
        rules.add(code)
        if action == ADVANCE and cooldown > 0:
            raise ValueError(f"{word} advances during the mandatory hold")
        blocked = (blocked + action) % NUM_INTERSECTIONS
        cooldown = env.cooldown_max if action == ADVANCE else max(cooldown - 1, 0)
        cause = int(delta[cause, labels[action]])
        if accepting[cause]:
            raise ValueError(f"the selected cycle {word} activates the cause")
        history = [*history[1:], blocked * (env.cooldown_max + 1) + cooldown]
        time += 1
    return table, len(rules)


@dataclass(frozen=True, slots=True, kw_only=True)
class CyclePolicy:
    """Execute a cycle from red-light and mandatory-hold frames, with cause state only when declared."""

    env: TrafficSignalEnv
    frames: int
    with_cause: bool

    def frame(self, state: EnvState) -> jax.Array:
        # Cooldown equals cooldown_max exactly after an advance, so the action is redundant.
        return (
            state.indices[self.env.blocked_index] * (self.env.cooldown_max + 1)
            + state.indices[self.env.cooldown_index]
        )

    def init(self, params: jax.Array, env_state: EnvState) -> jax.Array:
        del params
        return jnp.full((self.frames,), self.frame(env_state), dtype=jnp.int32)

    def select(
        self, params: jax.Array, policy_state: jax.Array, env_state: EnvState, key: jax.Array
    ) -> jax.Array:
        del key
        code = jnp.int32(0)
        for time in range(self.frames):
            code = code * (NUM_INTERSECTIONS * (self.env.cooldown_max + 1)) + policy_state[time]
        if self.with_cause:
            code = code * self.env.cause_dfa_size + env_state.indices[self.env.cause_index]
        return params[code]

    def advance(
        self, params: jax.Array, policy_state: jax.Array, new_env_state: EnvState
    ) -> jax.Array:
        del params
        return jnp.concatenate((policy_state[1:], self.frame(new_env_state)[None]))


def evaluate_cycle(
    env: TrafficSignalEnv, word: str, *, frames: int, with_cause: bool, episodes: int, horizon: int
) -> tuple[np.ndarray, int]:
    """Measure a proved feedback policy through the same evaluator as the learned policies."""
    table, rules = feedback_table(env, word, frames=frames, with_cause=with_cause)
    evaluate = build_greedy_eval(
        env,
        CyclePolicy(env=env, frames=frames, with_cause=with_cause),
        gamma=1.0,
        num_runs=1,
        eval_episodes=episodes,
        max_episode_steps=horizon,
        trajectory_indices=(env.cause_index,),
    )
    rollout = evaluate(jnp.asarray(table)[None])
    expected = np.asarray([ADVANCE if word[t % len(word)] == "a" else HOLD for t in range(horizon)])
    actions = np.asarray(env.combined_to_action)[np.asarray(rollout.labels)[0]]
    if not np.all(actions == expected) or not np.all(np.asarray(rollout.lengths) == horizon):
        raise ValueError("the feedback policy did not execute the selected cycle")
    if np.any(np.asarray(env.cause_accepting)[np.asarray(rollout.trajectories)[..., 0]]):
        raise ValueError("the feedback policy activated the cause")
    return np.asarray(rollout.total_returns[0], dtype=np.float64), rules


def screen_cycles(env: TrafficSignalEnv, settings: CycleSettings) -> tuple[list[CycleCell], int]:
    """Select one cycle per memory class (two frames, four frames, the cause state) by a coarse and a fine
    screen, and evaluate each on episode banks independent of the selection."""
    sequences = enumerate_cycles(
        max_period=settings.max_period, horizon=settings.horizon, cooldown_max=env.cooldown_max
    )
    facts = control_facts(env, sequences)
    coarse = screen_samples(env, sequences, episodes=settings.coarse_episodes, seed=1).mean(axis=1)
    order = np.argsort(-coarse, kind="stable")
    subsets: dict[tuple[int | None, bool, bool], np.ndarray] = {}
    for frames in (1, 2, 3, 4, 5, 6, 8, 12):
        for with_cause in (False, True):
            mask = expressible_mask(
                sequences,
                facts,
                frames=frames,
                frame_count=NUM_INTERSECTIONS * (env.cooldown_max + 1),
                cause_states=env.cause_dfa_size,
                with_cause=with_cause,
            )
            subsets[(frames, with_cause, False)] = mask
            subsets[(frames, with_cause, True)] = mask & facts.cause_free
    subsets[(None, True, False)] = np.ones(len(sequences), dtype=bool)
    subsets[(None, True, True)] = facts.cause_free
    subsets[(None, False, False)] = ~facts.cause_free
    nominees = {
        key: [int(i) for i in order if mask[i]][: settings.nominees]
        for key, mask in subsets.items()
    }
    refined = sorted({index for chosen in nominees.values() for index in chosen})
    fine = np.full(len(sequences), np.nan)
    fine[refined] = screen_samples(
        env, sequences[np.asarray(refined)], episodes=settings.fine_episodes, seed=2
    ).mean(axis=1)
    cells: list[CycleCell] = []
    for frames, with_cause in ((2, False), (4, False), (1, True)):
        key = (frames, with_cause, False)
        chosen = max(nominees[key], key=lambda i: fine[i])
        word = shortest_word(sequences[chosen])
        held = screen_samples(
            env, sequences[chosen : chosen + 1], episodes=settings.holdout_episodes, seed=3
        )[0]
        comparison, rules = evaluate_cycle(
            env,
            word,
            frames=frames,
            with_cause=with_cause,
            episodes=settings.comparison_episodes,
            horizon=settings.horizon,
        )
        cells.append(
            CycleCell(
                frames=frames,
                with_cause=with_cause,
                word=word,
                candidates=int(subsets[key].sum()),
                refined=len(nominees[key]),
                selection_mean=float(fine[chosen]),
                holdout_returns=held,
                comparison_returns=comparison,
                rules=rules,
            )
        )
    return cells, len(sequences)


def write_cycle_results(output: Path, cells: list[CycleCell]) -> None:
    """Write full-precision means, standard errors, and all individual evaluation returns."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / "cycles.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "frames",
                "with_cause",
                "word",
                "candidates",
                "refined",
                "rules",
                "selection_mean",
                "holdout_episodes",
                "holdout_mean",
                "holdout_se",
                "comparison_episodes",
                "comparison_mean",
                "comparison_se",
            )
        )
        for cell in cells:
            held, compared = cell.holdout_returns, cell.comparison_returns
            writer.writerow(
                (
                    cell.frames,
                    int(cell.with_cause),
                    cell.word,
                    cell.candidates,
                    cell.refined,
                    cell.rules,
                    cell.selection_mean,
                    len(held),
                    compute_mean(held),
                    float(held.std(ddof=1) / np.sqrt(len(held))),
                    len(compared),
                    compute_mean(compared),
                    float(compared.std(ddof=1) / np.sqrt(len(compared))),
                )
            )
    with (output / "cycles_episodes.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("frames", "with_cause", "word", "bank", "episode", "return"))
        for cell in cells:
            for bank, values in (
                ("holdout", cell.holdout_returns),
                ("comparison", cell.comparison_returns),
            ):
                writer.writerows(
                    (cell.frames, int(cell.with_cause), cell.word, bank, episode, float(value))
                    for episode, value in enumerate(values)
                )


def write_cycle_optimum(
    output: Path, env: TrafficSignalEnv, cells: list[CycleCell], horizon: int, *, gamma: float
) -> tuple[float, list[float]]:
    """Write cycle scores and the truncated score of a discount-optimal stationary policy."""
    matrices = transition_matrices(env)
    values, policy = solve_optimum(env, matrices, gamma=gamma)
    optimum = policy_return(env, policy, horizon, matrices)
    exact = [word_return(env, cell.word, horizon, matrices) for cell in cells]
    targets = action_values(env, values, matrices, gamma=gamma)
    residual = float(np.max(np.abs(targets.max(axis=-1) - values)))
    # Some states that no optimal policy reaches tie both actions exactly; the return from reset depends only on the reached ones.
    reached = reachable_states(env, policy, matrices)
    gap = float(np.min(np.abs(targets[..., ADVANCE] - targets[..., HOLD])[reached]))
    value_error = residual / (1 - gamma)
    if gap <= 2 * gamma * value_error:
        raise ValueError("the Bellman error bound does not certify the stationary policy's actions")
    np.savez_compressed(
        output / "stationary_policy.npz",
        values=values,
        policy=policy,
        gamma=gamma,
        bellman_residual=residual,
        value_error_bound=value_error,
        minimum_action_gap=gap,
        reachable_states=reached,
    )
    with (output / "cycles_optimum.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("policy", "frames", "with_cause", "word", "exact_return"))
        for cell, value in zip(cells, exact, strict=True):
            writer.writerow(("cycle", cell.frames, int(cell.with_cause), cell.word, value))
        writer.writerow(("optimal", "", "", "", optimum))
    return optimum, exact


def write_cycle_source_records(
    output: Path,
    *,
    config: TrafficSignalQLConfig,
    settings: CycleSettings,
    candidates: int,
    command: list[str],
) -> None:
    """Retain the resolved settings, source patch, dependency lock, and numerical runtime beside the results."""

    def git(*args: str) -> str:
        return subprocess.run(  # noqa: S603 - fixed Git executable and argument vectors, without a shell.
            ["git", "-C", str(ROOT_DIR), *args], check=True, capture_output=True, text=True
        ).stdout

    patch = git(
        "diff",
        "--binary",
        "HEAD",
        "--",
        "src",
        "scripts",
        "config",
        "pyproject.toml",
        "uv.lock",
    )
    for name in git(
        "ls-files",
        "--others",
        "--exclude-standard",
        "--",
        "src",
        "scripts",
        "config",
        "pyproject.toml",
        "uv.lock",
    ).splitlines():
        difference = subprocess.run(  # noqa: S603 - names come from Git, without shell interpolation.
            ["git", "-C", str(ROOT_DIR), "diff", "--no-index", "--binary", "/dev/null", name],
            capture_output=True,
            text=True,
            check=False,
        )
        if difference.returncode != 1:
            raise ValueError(f"cannot record the untracked source {name}")
        patch += difference.stdout
    (output / "cycles_source.patch").write_text(patch)
    shutil.copyfile(ROOT_DIR / "uv.lock", output / "cycles_uv.lock")
    portable_command = [
        str(Path(value).resolve().relative_to(ROOT_DIR.resolve()))
        if Path(value).is_absolute()
        else value
        for value in command
    ]
    metadata: dict[str, Json] = {
        "kind": "selected_cycle_lower_bounds_and_stationary_discounted_reference",
        "candidate_count": candidates,
        "configuration": config.to_dict(),
        "settings": asdict(settings),
        "source_repository": git("config", "--get", "remote.origin.url").strip(),
        "source_commit": git("rev-parse", "HEAD").strip(),
        "source_files": {
            name: sha256((ROOT_DIR / name).read_bytes()).hexdigest()
            for name in git(
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "--",
                "src",
                "scripts",
                "config",
                "data",
                "pyproject.toml",
                "uv.lock",
            ).splitlines()
            if (ROOT_DIR / name).is_file()
        },
        "source_dirty": bool(git("status", "--porcelain")),
        "source_patch": "cycles_source.patch",
        "dependency_lock": "cycles_uv.lock",
        "command": shlex.join(portable_command),
        "command_output": "the output directory is relocatable",
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jaxlib": version("jaxlib"),
        "numpy": np.__version__,
        "backend": jax.default_backend(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "jax_enable_x64": jax.config.x64_enabled,
        "jax_default_prng_impl": str(jax.config.jax_default_prng_impl),
        "xla_flags": os.environ.get("XLA_FLAGS"),
        "selection": {
            "coarse_seed": 1,
            "fine_seed": 2,
            "step_key_split": 2,
            "ranking": "stable, common episode keys",
        },
        "evaluation": {
            "implementation": "tcirl.rl.common.greedy_eval.build_greedy_eval",
            "seed": 0,
            "step_key_split": 3,
            "policy": "feedback on reset-filled control frames, ignoring queues and effect markers",
            "statistic": "arithmetic mean of undiscounted episode returns",
            "uncertainty": "sample standard deviation divided by square root of episode count",
            "smoothing": None,
            "ceiling": settings.horizon,
            "comparison_bank": "jax.random.split(jax.random.key(0), comparison_episodes); distinct from selection",
            "holdout_bank": "jax.random.split(jax.random.key(3), holdout_episodes); step key split in two; distinct from selection",
            "holdout_implementation": "tcirl.traffic_cycles.screen_samples; current environment step, fixed selected actions",
        },
        "optimum": {
            "implementation": "tcirl.traffic_optimum.solve_optimum, policy_return and word_return",
            "model": "every Bernoulli and capped geometric draw of one step integrated exactly, float64",
            "objective": "infinite-horizon discounted expected return",
            "gamma": config.params.gamma,
            "policy": "stationary deterministic, with no episode-time input",
            "solver": "value iteration to a maximum Bellman residual of 1e-12",
            "certificate": "stationary_policy.npz stores the Bellman residual, value-error bound, the states the policy reaches from reset and the minimum action gap over them; the gap must exceed twice gamma times the value-error bound, so every optimal policy takes the same actions on those states and has the same return from reset",
            "evaluation": "undiscounted sum over the first horizon steps from reset under that fixed stationary policy",
            "evaluation_is_upper_bound": False,
            "state": "queues, blocked intersection, cooldown and cause DFA state",
        },
        "files": {
            name: sha256((output / name).read_bytes()).hexdigest()
            for name in (
                "cycles.csv",
                "cycles_episodes.csv",
                "cycles_optimum.csv",
                "stationary_policy.npz",
                "cycles_source.patch",
                "cycles_uv.lock",
            )
        },
    }
    (output / "cycles_source_records.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
