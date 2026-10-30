from __future__ import annotations

from tcirl.config import (
    GeneEditingQLConfig,
    eval_schedule,
    parse_config,
)
from tcirl.envs.gene_editing import GeneEditingEnv
from tcirl.evaluation import ScheduledEvaluation, gene_editing_domain
from tcirl.jax_session import save_jax_reproducibility_metadata, setup_jax_run
from tcirl.logging.final_policy import q_learning_arrays, save_final_policy
from tcirl.logging.recorder import init_metric_recorder
from tcirl.metrics.registry import gene_editing_specs
from tcirl.path_consts import CONFIG_DIR
from tcirl.rl.agents import q_learning
from tcirl.session import init_logger
from tcirl.specifications.gene_editing import (
    CAUSE_FORMULA,
    create_gene_editing_cause,
)
from tcirl.visualization import DFASummaryInfo


def main() -> None:
    config = parse_config(
        GeneEditingQLConfig,
        CONFIG_DIR / "gene_editing_q_learning.yaml",
        "Q-learning on gene-editing gridworld",
    )
    runtime = setup_jax_run(seed=config.run.seed, fully_reproducible=config.run.fully_reproducible)
    rc = config.run
    p = config.params
    cadence = config.cadence

    spec = create_gene_editing_cause()
    env = GeneEditingEnv.from_config(config.env, spec, cause_known=config.cause_known)
    cause_dfa = spec.cause_dfa
    grid_alphabet = spec.grid_alphabet
    formula_str = str(CAUSE_FORMULA)
    logger = init_logger(config, env.name, gene_editing_specs())
    save_jax_reproducibility_metadata(logger.run_dir, logger, runtime)
    recorder = init_metric_recorder(logger, config.algo)
    traj_recorder = init_metric_recorder(
        logger,
        f"{config.algo}_trajectories",
        x_label="timestep",
    )
    print(f"Environment: state_shape={env.state_shape}, num_actions={env.num_actions}")
    print(f"Grid alphabet: {grid_alphabet.num_symbols} symbols")
    print(f"Cause DFA: {cause_dfa.num_states} states (cause_known={config.cause_known})")

    logger.log_automaton_summary(
        DFASummaryInfo(
            dfa=cause_dfa,
            alphabet=grid_alphabet,
            formula_str=formula_str,
            perf=spec.perf,
            encoder=None,
            live_states=None,
        ),
        filename="cause_dfa.txt",
        subdir="data/dfas",
        render_if=20,
    )
    logger.save_text(
        f"cause: {formula_str}\n",
        filename="formulas.txt",
        subdir="data",
    )

    schedule = eval_schedule(rc, cadence)
    evaluation = ScheduledEvaluation(
        logger=logger,
        recorder=recorder,
        traj_recorder=traj_recorder,
        schedule=schedule,
        smoothing_steps=cadence.smoothing_steps,
        domain=gene_editing_domain(
            disease_index=env.disease_index,
            harm_index=env.harm_index,
            disease_max=env.disease_max,
            harm_max=env.harm_max,
            cause_index=env.cause_index,
            num_cause_states=cause_dfa.num_states,
            cause_accepting_state=next(iter(cause_dfa.accepting_states)),
            x_r_obs_ids=spec.x_r_obs_ids,
            x_a_obs_ids=spec.x_a_obs_ids,
        ),
        num_runs=rc.num_runs,
        eval_episodes=rc.eval_episodes,
    )

    print(f"\nTraining q-learning for {rc.total_steps} steps x {rc.num_runs} seeds...")

    # Training drives the evaluation too, so the Q-tables it returns are already measured.
    q_table = q_learning.train(
        env,
        logger=logger,
        recorder=recorder,
        gamma=p.gamma,
        step_size_scale=p.step_size_scale,
        epsilon_start=p.epsilon.start,
        epsilon_end=p.epsilon.end,
        epsilon_decay_steps=p.epsilon.decay_steps,
        max_episode_steps=p.max_episode_steps,
        q_init=p.q_init,
        num_runs=rc.num_runs,
        schedule=schedule,
        smoothing_steps=cadence.smoothing_steps,
        count_restart_steps=tuple(config.count_restart_steps),
        seed_offset=rc.seed_offset,
        num_labels=spec.obs_alphabet.num_symbols,
        evaluation=evaluation,
    )

    evaluation.finish()

    recorder.save()
    # After the recorder, so a failure here cannot cost the run its curves.
    save_final_policy(logger.run_dir, logger, q_learning_arrays(q_table), algo=config.algo)
    logger.finish_run()


if __name__ == "__main__":
    main()
