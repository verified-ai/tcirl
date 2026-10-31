"""Configuration dataclasses for TCIRL experiments and the shared CLI config parser."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from simple_parsing import ArgumentParser, NestedMode, Serializable

from tcirl.envs.gene_editing import validate_gene_editing_parameters
from tcirl.envs.traffic_signal import validate_traffic_signal_parameters
from tcirl.path_consts import CONFIG_DIR, ROOT_DIR
from tcirl.rl.common.eval_schedule import EvalSchedule
from tcirl.specifications.traffic_signal import validate_difficulty


def parse_config[T: Serializable](
    config_cls: type[T],
    default_yaml: Path,
    description: str,
    argv: Sequence[str] | None = None,
) -> T:
    """Parse a config from its default yaml, an optional --config_path, and CLI overrides."""
    parser = ArgumentParser(
        description=description,
        add_config_path_arg=True,
        nested_mode=NestedMode.WITHOUT_ROOT,
        config_path=default_yaml,
    )
    parser.add_arguments(config_cls, dest="config")
    args = parser.parse_args(argv)
    config = args.config
    assert isinstance(config, config_cls)
    return config


@dataclass(frozen=True, slots=True, kw_only=True)
class ScheduleConfig:
    """Linear decay schedule configuration."""

    start: float
    end: float
    decay_steps: int

    def __post_init__(self) -> None:
        if not (0 <= self.start <= 1):
            raise ValueError("start must be in [0, 1]")
        if not (0 <= self.end <= 1):
            raise ValueError("end must be in [0, 1]")
        if self.end > self.start:
            raise ValueError("end must be <= start")
        if self.decay_steps <= 0:
            raise ValueError("decay_steps must be > 0")


@dataclass(frozen=True, slots=True, kw_only=True)
class QLearningParams:
    """Hyperparameters of the tabular learner (`tcirl.agents.tabular`), which Q-learning and TCIRL share.

    Each Q-table entry's step size is `rl.schedules.rescaled_linear_step_size` of its own update count with the scale `step_size_scale`.
    """

    gamma: float
    epsilon: ScheduleConfig
    max_episode_steps: int
    q_init: float
    step_size_scale: float
    # Each transition also updates the entry of its action at every other state of the agent's DFA with the phase of the visited state.
    counterfactual: bool

    def __post_init__(self) -> None:
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be > 0")
        # Truncation does not zero the bootstrap, so the values are bounded only when gamma is below 1.
        if not 0 < self.gamma < 1:
            raise ValueError("gamma must be in (0, 1)")
        if self.step_size_scale <= 0:
            raise ValueError("step_size_scale must be > 0")


@dataclass(frozen=True, slots=True, kw_only=True)
class LoggingConfig:
    """W&B configuration."""

    # W&B metadata, not algorithm inputs, so these have defaults.
    #
    # `upload` picks the run's W&B mode: True is an online run, False an offline one
    # (`wandb/offline-run-<timestamp>-<id>/`, syncable later with `wandb sync`). Either
    # way the run records its metrics and artifacts, so `--noupload` costs you the
    # upload, never the results.
    upload: bool = True
    entity: str | None = None
    name: str | None = None
    project: str | None = None
    group: str | None = None
    notes: str | None = None
    tags: list[str] | None = None
    run_id: str | None = None
    job_type: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class LoggingCadenceConfig:
    """Training logging cadence for BatchLogger-style loops.

    `log_interval` is the render resolution / tile width (one logged point per `log_interval`
    steps); `stop_interval` is how often the outer loop does its per-interval work, which for
    TCIRL is the counterexample check and the re-inference it triggers. It is NOT the JAX-batch
    size: `~tcirl.rl.common.eval_schedule.EvalSchedule.batch_steps` derives that from this and
    the evaluation spacing, so the figure's resolution cannot move a method parameter.
    `smoothing_steps` is the EMA denoising timescale in env steps applied to mean-type metrics
    (the single source of truth for smoothing, used for both the wandb curves and the exports);
    `None` disables smoothing. It lives here, not in `LoggingConfig`, because `LoggingConfig` is
    excluded from the tracked W&B config while the export needs to recover the smoothing setting.
    """

    log_interval: int
    stop_interval: int
    smoothing_steps: int | None

    def __post_init__(self) -> None:
        if self.log_interval <= 0:
            raise ValueError("log_interval must be > 0")
        if self.stop_interval <= 0:
            raise ValueError("stop_interval must be > 0")
        if self.stop_interval % self.log_interval != 0:
            raise ValueError(
                f"stop_interval ({self.stop_interval}) must be a multiple of log_interval "
                f"({self.log_interval}) so every batch boundary falls on a log point, where "
                f"BatchMetrics (inference/*) are emitted."
            )
        if self.smoothing_steps is not None and self.smoothing_steps < self.log_interval:
            raise ValueError(
                f"smoothing_steps ({self.smoothing_steps}) must be >= log_interval "
                f"({self.log_interval}); you cannot denoise below the sampling resolution."
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class RunConfig:
    """Top-level run parameters."""

    # The process seed: `PYTHONHASHSEED` must equal it (`session.require_pythonhashseed`), and it
    # seeds Python's and numpy's global state. It is NOT the per-run bank -- that is `seed_offset`.
    seed: int
    num_runs: int
    total_steps: int
    # How many greedy evaluation episodes each seed plays at each point of the evaluation grid.
    # Every seed of every arm plays the SAME bank of episodes (`rl.common.greedy_eval`), so a
    # difference between two curves is a difference in policy and not in the episodes drawn.
    eval_episodes: int
    # How many points the evaluation grid holds. 1 is the end of training alone, the
    # classic train-then-evaluate split; 2 adds
    # the untrained policy at step 0; k is k equidistant points from 0 to total_steps, so 3 is
    # (start, half-point, end) and the shipped 501 is one every 1000 steps.
    # `rl.common.eval_schedule.EvalSchedule` turns it into the steps, derives the JAX batch from it,
    # and rejects a value below 1 or a spacing that is not a whole number of log_intervals.
    num_eval_points: int
    fully_reproducible: bool
    # Shifts the per-run seed bank to `seed_offset + range(num_runs)`, which keys EVERY per-seed
    # stream of EVERY agent: the env reset, the action/env key, and the deep arm's network
    # initialization. Seed `i` of two arms therefore shares one bank, so arms can be compared seed
    # by seed. Another offset draws a fresh, non-overlapping bank. 0 is the bank every shipped
    # config uses.
    seed_offset: int

    def __post_init__(self) -> None:
        if self.num_runs <= 0:
            raise ValueError("num_runs must be > 0")
        if self.seed_offset < 0:
            raise ValueError("seed_offset must be >= 0")


def eval_schedule(run: RunConfig, cadence: LoggingCadenceConfig) -> EvalSchedule:
    """The run's evaluation grid, built from the two config groups that between them fix it."""
    return EvalSchedule(
        total_steps=run.total_steps,
        num_eval_points=run.num_eval_points,
        stop_interval=cadence.stop_interval,
        log_interval=cadence.log_interval,
    )


def _check_cadence(run: RunConfig, cadence: LoggingCadenceConfig) -> None:
    if run.total_steps % cadence.stop_interval != 0:
        raise ValueError(
            f"total_steps ({run.total_steps}) must be divisible by "
            f"stop_interval ({cadence.stop_interval})"
        )
    # Constructing the schedule validates it, so an unreachable or unequal grid stops the run at
    # config time rather than at the first point the loop cannot stop on.
    eval_schedule(run, cadence)


# The rule by which, at a re-inference, every state of the new hypothesis takes the Q-values of a state of the old one (`tcirl.inference.transfer`), and every update count restarts at 0: "none" restarts the whole table, "exact" takes an equivalent old state and otherwise the old state of the shortest label sequence, and "counterexample" takes the old state that the most prefixes of the causal sample lead to together with it.
ValueTransfer = Literal["none", "exact", "counterexample"]


# ---------------------------------------------------------------------------
# Parameter groups
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True, kw_only=True)
class TcirlParams:
    """The hyperparameters of TCIRL's cause inference (`tcirl.agents.tcirl.CauseInference`)."""

    self_loop_symbol_enabled: bool
    initial_rejecting: bool
    symmetry_breaking: bool
    # The state of the old hypothesis whose Q-values each state of the new one takes at a re-inference.
    value_transfer: ValueTransfer
    # The largest hypothesis the SAT search tries, which also sizes the hypothesis axis of the Q-table.
    max_hypothesis_states: int


@dataclass(frozen=True, slots=True, kw_only=True)
class DqnParams:
    """The deep baseline's hyperparameters.

    `lr` is the optimizer's constant learning rate.

    `history_len` is the number of observation FRAMES the network sees, which is the paper's word:
    one frame is the observable state plus the label of the transition into it
    (`~tcirl.rl.common.history`).
    """

    gamma: float
    lr: float
    epsilon: ScheduleConfig
    max_episode_steps: int
    hidden_sizes: list[int]
    history_len: int
    buffer_size: int
    batch_size: int
    warmup_steps: int
    target_update_steps: int
    max_grad_norm: float | None
    huber_delta: float
    double_q: bool
    huber: bool

    def __post_init__(self) -> None:
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be > 0")
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        if not self.lr > 0:
            raise ValueError("lr must be > 0")
        if self.history_len < 1:
            raise ValueError("history_len must be >= 1")
        if not self.hidden_sizes:
            raise ValueError("hidden_sizes must name at least one hidden layer")
        if any(width < 1 for width in self.hidden_sizes):
            raise ValueError("every hidden layer must have at least one unit")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.buffer_size < self.batch_size:
            raise ValueError("buffer_size must be >= batch_size")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        # The buffer never holds more than buffer_size rows, so a warm-up above it is a gate that
        # can never open: the run would complete its full budget, take zero gradient steps, and
        # export an untrained network as the baseline with no error anywhere.
        if self.warmup_steps > self.buffer_size:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) must be <= buffer_size "
                f"({self.buffer_size}); the buffer can never reach the gate otherwise"
            )
        if self.target_update_steps < 1:
            raise ValueError("target_update_steps must be >= 1")
        if self.max_grad_norm is not None and not self.max_grad_norm > 0:
            raise ValueError("max_grad_norm must be > 0, or null for no clipping")
        if not self.huber_delta > 0:
            raise ValueError("huber_delta must be > 0")

    @property
    def hidden(self) -> tuple[int, ...]:
        """`hidden_sizes` as the tuple the agent wants (YAML gives a list)."""
        return tuple(self.hidden_sizes)


# ---------------------------------------------------------------------------
# Environment configs
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True, kw_only=True)
class GeneEditingConfig:
    """Genetic-therapy environment parameters."""

    grid_path: Path
    slip_prob: float
    d_rej: int
    p_rej: float
    d_acc: int
    p_acc: float
    disease_max: int
    disease_init: int
    harm_max: int
    cure_reduction: int

    def __post_init__(self) -> None:
        # One validator, in the environment that owns the dynamics, so the ranges cannot drift.
        validate_gene_editing_parameters(
            slip_prob=self.slip_prob,
            d_rej=self.d_rej,
            p_rej=self.p_rej,
            d_acc=self.d_acc,
            p_acc=self.p_acc,
            disease_max=self.disease_max,
            disease_init=self.disease_init,
            harm_max=self.harm_max,
            cure_reduction=self.cure_reduction,
        )


# ---------------------------------------------------------------------------
# Per-CLI top-level configs
# ---------------------------------------------------------------------------
@dataclass(slots=True, kw_only=True)
class GeneEditingQLConfig(Serializable):
    """Config for Q-learning on genetic-therapy gridworld."""

    algo: Literal["q_learning"]
    cause_known: bool
    # The steps at which every update count of the Q-table restarts at 0 while the Q-values stay,
    # as at a TCIRL re-inference. The shipped configurations restart none.
    count_restart_steps: list[int]
    run: RunConfig
    params: QLearningParams
    logging: LoggingConfig
    cadence: LoggingCadenceConfig
    env: GeneEditingConfig

    def __post_init__(self) -> None:
        _check_cadence(self.run, self.cadence)

    @property
    def all_tags(self) -> list[str]:
        cause_tag = "cause_known" if self.cause_known else "cause_unknown"
        return [*(self.logging.tags or ()), self.algo, cause_tag]


@dataclass(slots=True, kw_only=True)
class GeneEditingTcirlConfig(Serializable):
    """Config for TCIRL on genetic-therapy gridworld."""

    algo: Literal["tcirl"]
    run: RunConfig
    params: QLearningParams
    tcirl: TcirlParams
    logging: LoggingConfig
    cadence: LoggingCadenceConfig
    env: GeneEditingConfig

    def __post_init__(self) -> None:
        _check_cadence(self.run, self.cadence)

    @property
    def all_tags(self) -> list[str]:
        return [*(self.logging.tags or ()), self.algo]


@dataclass(slots=True, kw_only=True)
class GeneEditingDqnConfig(Serializable):
    """Config for the DQN baseline on the genetic-therapy gridworld."""

    algo: Literal["dqn"]
    cause_known: bool
    run: RunConfig
    params: DqnParams
    logging: LoggingConfig
    cadence: LoggingCadenceConfig
    env: GeneEditingConfig

    def __post_init__(self) -> None:
        _check_cadence(self.run, self.cadence)

    @property
    def all_tags(self) -> list[str]:
        cause_tag = "cause_known" if self.cause_known else "cause_unknown"
        return [*(self.logging.tags or ()), self.algo, cause_tag, f"h{self.params.history_len}"]


# ---------------------------------------------------------------------------
# Traffic signal configs
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True, kw_only=True)
class TrafficSignalConfig:
    """Traffic signal environment parameters."""

    difficulty: int
    p_rej_depart: float
    p_rej_arrive: float
    p_acc_depart: float
    p_acc_arrive: float
    q_max: int
    # Cars in every queue at reset (not q_init, which params uses).
    queue_init: int
    cooldown_max: int

    def __post_init__(self) -> None:
        validate_difficulty(self.difficulty)
        # One validator, in the environment that owns the dynamics, so the ranges cannot drift.
        validate_traffic_signal_parameters(
            p_rej_depart=self.p_rej_depart,
            p_rej_arrive=self.p_rej_arrive,
            p_acc_depart=self.p_acc_depart,
            p_acc_arrive=self.p_acc_arrive,
            q_max=self.q_max,
            cooldown_max=self.cooldown_max,
            queue_init=self.queue_init,
        )


@dataclass(slots=True, kw_only=True)
class TrafficSignalQLConfig(Serializable):
    """Config for Q-learning on traffic signal control."""

    algo: Literal["q_learning"]
    cause_known: bool
    # The steps at which every update count of the Q-table restarts at 0 while the Q-values stay,
    # as at a TCIRL re-inference. The shipped configurations restart none.
    count_restart_steps: list[int]
    run: RunConfig
    params: QLearningParams
    logging: LoggingConfig
    cadence: LoggingCadenceConfig
    env: TrafficSignalConfig

    def __post_init__(self) -> None:
        _check_cadence(self.run, self.cadence)

    @property
    def all_tags(self) -> list[str]:
        cause_tag = "cause_known" if self.cause_known else "cause_unknown"
        return [*(self.logging.tags or ()), self.algo, "traffic", cause_tag]


@dataclass(slots=True, kw_only=True)
class TrafficSignalTcirlConfig(Serializable):
    """Config for TCIRL on traffic signal control."""

    algo: Literal["tcirl"]
    run: RunConfig
    params: QLearningParams
    tcirl: TcirlParams
    logging: LoggingConfig
    cadence: LoggingCadenceConfig
    env: TrafficSignalConfig

    def __post_init__(self) -> None:
        _check_cadence(self.run, self.cadence)

    @property
    def all_tags(self) -> list[str]:
        return [*(self.logging.tags or ()), self.algo, "traffic"]


@dataclass(slots=True, kw_only=True)
class TrafficSignalDqnConfig(Serializable):
    """Config for the DQN baseline on traffic signal control."""

    algo: Literal["dqn"]
    cause_known: bool
    run: RunConfig
    params: DqnParams
    logging: LoggingConfig
    cadence: LoggingCadenceConfig
    env: TrafficSignalConfig

    def __post_init__(self) -> None:
        _check_cadence(self.run, self.cadence)

    @property
    def all_tags(self) -> list[str]:
        cause_tag = "cause_known" if self.cause_known else "cause_unknown"
        return [
            *(self.logging.tags or ()),
            self.algo,
            "traffic",
            cause_tag,
            f"h{self.params.history_len}",
        ]


type QLearningConfig = GeneEditingQLConfig | TrafficSignalQLConfig
type TcirlConfig = GeneEditingTcirlConfig | TrafficSignalTcirlConfig
type DqnConfig = GeneEditingDqnConfig | TrafficSignalDqnConfig
type AnyConfig = QLearningConfig | TcirlConfig | DqnConfig


# ---------------------------------------------------------------------------
# The paper's environments
# ---------------------------------------------------------------------------
def official_gene_editing_env_config() -> GeneEditingConfig:
    """The `env` section of the shipped genetic-therapy configs, with its grid path resolved against the repository root."""
    env = parse_config(
        GeneEditingTcirlConfig,
        CONFIG_DIR / "gene_editing_tcirl.yaml",
        "the paper's genetic-therapy environment",
        argv=[],
    ).env
    return replace(env, grid_path=ROOT_DIR / env.grid_path)


def official_traffic_signal_env_config() -> TrafficSignalConfig:
    """The `env` section of the shipped traffic configs, which the three arms share."""
    return parse_config(
        TrafficSignalTcirlConfig,
        CONFIG_DIR / "traffic_signal_tcirl.yaml",
        "the paper's traffic environment",
        argv=[],
    ).env
