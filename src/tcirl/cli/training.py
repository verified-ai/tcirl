"""What the six training entry points share: the case study, the run's records, and the three learners.

An entry point names its configuration and its case study and calls `train_q_learning`,
`train_tcirl` or `train_dqn`. Q-learning and TCIRL are the one tabular learner of
`tcirl.agents.tabular` on different DFAs: Q-learning on the cause DFA when the cause is known and
on the one-state DFA otherwise, TCIRL on hypotheses that its cause inference revises. The tabular
learner tracks the DFA state itself, so its environment never exposes the cause state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from tcirl.agents import dqn, tabular
from tcirl.agents.tcirl import CauseInference
from tcirl.alphabet import OrderedAlphabet
from tcirl.automata.dfa import DFA, validate_self_loop_symbol
from tcirl.config import (
    AnyConfig,
    DqnConfig,
    GeneEditingConfig,
    QLearningConfig,
    TcirlConfig,
    TrafficSignalConfig,
    eval_schedule,
)
from tcirl.envs.gene_editing import GeneEditingEnv
from tcirl.envs.traffic_signal import TrafficSignalEnv
from tcirl.evaluation import (
    EvalDomain,
    ScheduledEvaluation,
    gene_editing_domain,
    traffic_signal_domain,
)
from tcirl.jax_session import save_jax_reproducibility_metadata, setup_jax_run
from tcirl.logging.final_policy import save_final_policy, tabular_arrays
from tcirl.logging.recorder import PerSeedMetricRecorder, init_metric_recorder
from tcirl.logging.wandb import WandbLogger
from tcirl.metrics.registry import gene_editing_specs, traffic_signal_specs
from tcirl.metrics.spec import MetricRegistry
from tcirl.rl.common.tabular import TabularEnv
from tcirl.session import init_logger
from tcirl.specifications.gene_editing import CAUSE_FORMULA, create_gene_editing_cause
from tcirl.specifications.traffic_signal import create_traffic_signal_cause
from tcirl.visualization import DFASummaryInfo


@dataclass(frozen=True, slots=True, kw_only=True)
class CaseStudy:
    """A case study as the learners see it.

    `cause_alphabet` is the alphabet of the cause DFA, and so of every DFA a tabular learner runs
    on; `label_projection` maps each label the environment emits to its symbol there.
    """

    make_env: Callable[[bool], TabularEnv]  # cause_known -> the environment
    cause_dfa: DFA
    cause_formula: str
    cause_alphabet: OrderedAlphabet[str]
    num_labels: int
    label_projection: np.ndarray
    x_r_obs_ids: frozenset[int]
    x_a_obs_ids: frozenset[int]
    perf: dict[str, float]
    evaluation_domain: EvalDomain
    tabular_specs: MetricRegistry
    deep_specs: MetricRegistry


def gene_editing_case_study(config: GeneEditingConfig) -> CaseStudy:
    """Genetic therapy, whose cause reads the grid cell of each step."""
    spec = create_gene_editing_cause()

    def make_env(cause_known: bool) -> GeneEditingEnv:
        return GeneEditingEnv.from_config(config, spec, cause_known=cause_known)

    env = make_env(False)
    return CaseStudy(
        make_env=make_env,
        cause_dfa=spec.cause_dfa,
        cause_formula=str(CAUSE_FORMULA),
        cause_alphabet=spec.grid_alphabet,
        num_labels=spec.obs_alphabet.num_symbols,
        label_projection=spec.label_projection,
        x_r_obs_ids=spec.x_r_obs_ids,
        x_a_obs_ids=spec.x_a_obs_ids,
        perf=spec.perf,
        evaluation_domain=gene_editing_domain(
            disease_index=env.disease_index,
            harm_index=env.harm_index,
            disease_max=env.disease_max,
            harm_max=env.harm_max,
            cause_index=env.cause_index,
            num_cause_states=spec.cause_dfa.num_states,
            cause_accepting_state=next(iter(spec.cause_dfa.accepting_states)),
            x_r_obs_ids=spec.x_r_obs_ids,
            x_a_obs_ids=spec.x_a_obs_ids,
        ),
        tabular_specs=gene_editing_specs(),
        deep_specs=gene_editing_specs(deep=True),
    )


def traffic_signal_case_study(config: TrafficSignalConfig) -> CaseStudy:
    """Traffic signal control, whose cause reads the action of each step."""
    spec = create_traffic_signal_cause(config.difficulty)

    def make_env(cause_known: bool) -> TrafficSignalEnv:
        return TrafficSignalEnv.from_config(config, spec, cause_known=cause_known)

    env = make_env(False)
    return CaseStudy(
        make_env=make_env,
        cause_dfa=spec.cause_dfa,
        cause_formula=str(spec.cause_formula),
        cause_alphabet=spec.action_alphabet,
        num_labels=spec.obs_alphabet.num_symbols,
        label_projection=spec.label_projection,
        x_r_obs_ids=spec.x_r_obs_ids,
        x_a_obs_ids=spec.x_a_obs_ids,
        perf=spec.perf,
        evaluation_domain=traffic_signal_domain(
            queue_indices=(env.q0_index, env.q1_index, env.q2_index),
            cause_index=env.cause_index,
            q_max=env.q_max,
            num_cause_states=spec.cause_dfa.num_states,
            cause_accepting_state=next(iter(spec.cause_dfa.accepting_states)),
            x_r_obs_ids=spec.x_r_obs_ids,
            x_a_obs_ids=spec.x_a_obs_ids,
        ),
        tabular_specs=traffic_signal_specs(),
        deep_specs=traffic_signal_specs(deep=True),
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class _Run:
    """One open run: its environment and the records its training writes."""

    env: TabularEnv
    logger: WandbLogger
    recorder: PerSeedMetricRecorder
    evaluation: ScheduledEvaluation


def _open_run(
    config: AnyConfig,
    case: CaseStudy,
    *,
    runtime: dict[str, Any],
    cause_known: bool,
    specs: MetricRegistry,
) -> _Run:
    env = case.make_env(cause_known)
    logger = init_logger(config, env.name, specs)
    save_jax_reproducibility_metadata(logger.run_dir, logger, runtime)
    recorder = init_metric_recorder(logger, config.algo)
    traj_recorder = init_metric_recorder(logger, f"{config.algo}_trajectories", x_label="timestep")
    print(f"Environment: state_shape={env.state_shape}, num_actions={env.num_actions}")
    print(f"Cause DFA: {case.cause_dfa.num_states} states")
    logger.log_automaton_summary(
        DFASummaryInfo(
            dfa=case.cause_dfa,
            alphabet=case.cause_alphabet,
            formula_str=case.cause_formula,
            perf=case.perf,
            encoder=None,
            live_states=None,
        ),
        filename="cause_dfa.txt",
        subdir="data/dfas",
        render_if=20,
    )
    logger.save_text(f"cause: {case.cause_formula}\n", filename="formulas.txt", subdir="data")
    evaluation = ScheduledEvaluation(
        logger=logger,
        recorder=recorder,
        traj_recorder=traj_recorder,
        schedule=eval_schedule(config.run, config.cadence),
        smoothing_steps=config.cadence.smoothing_steps,
        domain=case.evaluation_domain,
        num_runs=config.run.num_runs,
        eval_episodes=config.run.eval_episodes,
    )
    print(
        f"\nTraining {config.algo} for {config.run.total_steps} steps x {config.run.num_runs} seeds..."
    )
    return _Run(env=env, logger=logger, recorder=recorder, evaluation=evaluation)


def _close_run(run: _Run, arrays: dict[str, np.ndarray], *, algo: str) -> None:
    run.evaluation.finish()
    run.recorder.save()
    # After the recorder, so a failure here cannot cost the run its curves.
    save_final_policy(run.logger.run_dir, run.logger, arrays, algo=algo)
    run.logger.finish_run()


def _train_tabular(
    config: QLearningConfig | TcirlConfig,
    case: CaseStudy,
    run: _Run,
    *,
    automata: list[DFA],
    automaton_size: int,
    inference: CauseInference | None,
    count_restart_steps: tuple[int, ...],
) -> None:
    p = config.params
    # Training drives the evaluation too, so the policies it returns are already measured.
    trained = tabular.train(
        run.env,
        automata=automata,
        automaton_size=automaton_size,
        label_projection=jnp.asarray(case.label_projection),
        counterfactual=p.counterfactual,
        inference=inference,
        count_restart_steps=count_restart_steps,
        logger=run.logger,
        recorder=run.recorder,
        gamma=p.gamma,
        step_size_scale=p.step_size_scale,
        epsilon=p.epsilon,
        max_episode_steps=p.max_episode_steps,
        q_init=p.q_init,
        seed_offset=config.run.seed_offset,
        num_runs=config.run.num_runs,
        schedule=run.evaluation.schedule,
        smoothing_steps=config.cadence.smoothing_steps,
        evaluation=run.evaluation,
    )
    _close_run(run, tabular_arrays(trained.q_tables, trained.automata), algo=config.algo)


def train_q_learning(config: QLearningConfig, case_study: Callable[[], CaseStudy]) -> None:
    """Q-learning on the cause DFA when `config.cause_known`, and on the one-state DFA otherwise."""
    runtime = setup_jax_run(seed=config.run.seed, fully_reproducible=config.run.fully_reproducible)
    case = case_study()
    automaton = (
        case.cause_dfa
        if config.cause_known
        else DFA.trivial_hypothesis(case.cause_alphabet.num_symbols)
    )
    run = _open_run(config, case, runtime=runtime, cause_known=False, specs=case.tabular_specs)
    _train_tabular(
        config,
        case,
        run,
        automata=[automaton] * config.run.num_runs,
        automaton_size=automaton.num_states,
        inference=None,
        count_restart_steps=tuple(config.count_restart_steps),
    )


def train_tcirl(config: TcirlConfig, case_study: Callable[[], CaseStudy]) -> None:
    """TCIRL: the tabular learner on hypotheses that its cause inference revises."""
    runtime = setup_jax_run(seed=config.run.seed, fully_reproducible=config.run.fully_reproducible)
    case = case_study()
    tc = config.tcirl
    # The declared self-loop symbol is the empty label. The compression is sound only when the cause
    # really never reads it, so the validator refuses a cause that does.
    self_loop_symbol = (
        case.cause_alphabet.encode(frozenset()) if tc.self_loop_symbol_enabled else None
    )
    if self_loop_symbol is not None:
        validate_self_loop_symbol(case.cause_dfa, self_loop_symbol)
    run = _open_run(config, case, runtime=runtime, cause_known=False, specs=case.tabular_specs)
    inference = CauseInference(
        num_runs=config.run.num_runs,
        alphabet=case.cause_alphabet,
        label_projection=case.label_projection,
        x_r_obs_ids=case.x_r_obs_ids,
        x_a_obs_ids=case.x_a_obs_ids,
        self_loop_symbol=self_loop_symbol,
        max_states=tc.max_hypothesis_states,
        initial_rejecting=tc.initial_rejecting,
        symmetry_breaking=tc.symmetry_breaking,
        value_transfer=tc.value_transfer,
        q_init=config.params.q_init,
        logger=run.logger,
    )
    _train_tabular(
        config,
        case,
        run,
        automata=inference.initial_hypotheses,
        automaton_size=tc.max_hypothesis_states,
        inference=inference,
        count_restart_steps=(),
    )


def train_dqn(config: DqnConfig, case_study: Callable[[], CaseStudy]) -> None:
    """The deep baseline, on a window of the last `history_len` observation frames."""
    runtime = setup_jax_run(seed=config.run.seed, fully_reproducible=config.run.fully_reproducible)
    case = case_study()
    run = _open_run(
        config, case, runtime=runtime, cause_known=config.cause_known, specs=case.deep_specs
    )
    p = config.params
    # Training drives the evaluation too, so the network it returns is already measured.
    trained = dqn.train(
        run.env,
        logger=run.logger,
        recorder=run.recorder,
        num_labels=case.num_labels,
        gamma=p.gamma,
        lr=p.lr,
        epsilon_start=p.epsilon.start,
        epsilon_end=p.epsilon.end,
        epsilon_decay_steps=p.epsilon.decay_steps,
        max_episode_steps=p.max_episode_steps,
        hidden_sizes=p.hidden,
        history_len=p.history_len,
        buffer_size=p.buffer_size,
        batch_size=p.batch_size,
        warmup_steps=p.warmup_steps,
        target_update_steps=p.target_update_steps,
        max_grad_norm=p.max_grad_norm,
        huber_delta=p.huber_delta,
        double_q=p.double_q,
        huber=p.huber,
        num_runs=config.run.num_runs,
        schedule=run.evaluation.schedule,
        smoothing_steps=config.cadence.smoothing_steps,
        seed_offset=config.run.seed_offset,
        evaluation=run.evaluation,
    )
    _close_run(run, dqn.final_policy_arrays(trained), algo=config.algo)
