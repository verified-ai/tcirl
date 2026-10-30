from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from tcirl.agents import tcirl
from tcirl.automata.dfa import DFA, validate_self_loop_symbol
from tcirl.automata.plain_dfa import trivial_rejecting_dfa
from tcirl.config import (
    GeneEditingTcirlConfig,
    InferenceMode,
    eval_schedule,
    parse_config,
)
from tcirl.envs.gene_editing import GeneEditingEnv
from tcirl.evaluation import ScheduledEvaluation, gene_editing_domain
from tcirl.jax_session import save_jax_reproducibility_metadata, setup_jax_run
from tcirl.logging.final_policy import save_final_policy, tcirl_arrays
from tcirl.logging.recorder import init_metric_recorder
from tcirl.metrics.registry import gene_editing_specs
from tcirl.metrics.spec import TCIRL_TRAIN_METRICS
from tcirl.path_consts import CONFIG_DIR
from tcirl.session import init_logger
from tcirl.specifications.gene_editing import (
    CAUSE_FORMULA,
    create_gene_editing_cause,
)
from tcirl.visualization import DFASummaryInfo


def _build_init_hypotheses(
    inference_mode: InferenceMode,
    cause_dfa: DFA,
    num_symbols: int,
    num_runs: int,
) -> list[DFA]:
    """Build per-seed initial hypothesis DFAs based on the inference mode."""
    match inference_mode:
        case "oracle":
            return [cause_dfa for _ in range(num_runs)]
        case "none" | "sat":
            trivial = trivial_rejecting_dfa(num_symbols)
            return [trivial for _ in range(num_runs)]


def main() -> None:
    config = parse_config(
        GeneEditingTcirlConfig,
        CONFIG_DIR / "gene_editing_tcirl.yaml",
        "TCIRL on gene-editing gridworld",
    )
    runtime = setup_jax_run(seed=config.run.seed, fully_reproducible=config.run.fully_reproducible)
    rc = config.run
    p = config.params
    cadence = config.cadence
    mode = config.tcirl.inference_mode

    spec = create_gene_editing_cause()
    env = GeneEditingEnv.from_config(config.env, spec, cause_known=False)
    cause_dfa = spec.cause_dfa
    grid_alphabet = spec.grid_alphabet
    obs_alphabet = spec.obs_alphabet
    formula_str = str(CAUSE_FORMULA)
    logger = init_logger(config, env.name, gene_editing_specs().extended(TCIRL_TRAIN_METRICS))
    save_jax_reproducibility_metadata(logger.run_dir, logger, runtime)
    recorder = init_metric_recorder(logger, config.algo)
    traj_recorder = init_metric_recorder(
        logger,
        f"{config.algo}_trajectories",
        x_label="timestep",
    )
    print(f"Environment: state_shape={env.state_shape}, num_actions={env.num_actions}")
    print(f"Grid alphabet: {grid_alphabet.num_symbols} symbols")
    print(f"Obs alphabet: {obs_alphabet.num_symbols} symbols")
    print(f"Cause DFA: {cause_dfa.num_states} states (inference_mode={mode})")

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

    # Hypothesis DFAs operate on the grid alphabet (6 symbols).
    # In sat mode the hypothesis axis is padded to the search cap, so a re-inference of a different size does not retrigger JIT.
    # `agents.tcirl.train` raises when no size up to the cap is consistent with the sample.
    cause_dfa_size = config.tcirl.max_hypothesis_states if mode == "sat" else cause_dfa.num_states

    init_hypotheses = _build_init_hypotheses(
        mode, cause_dfa, grid_alphabet.num_symbols, rc.num_runs
    )

    # The declared self-loop symbol: the empty label (no proposition).
    empty_label_id = grid_alphabet.encode(frozenset())
    tc = config.tcirl
    self_loop_symbol: int | None = empty_label_id if tc.self_loop_symbol_enabled else None
    # The compression is sound only when the symbol really has the property the config claims, so the validator refuses a cause that does not have it.
    if self_loop_symbol is not None:
        validate_self_loop_symbol(cause_dfa, self_loop_symbol)

    # Build combined_to_grid JAX array for the agent.
    c2g_np = np.zeros(obs_alphabet.num_symbols, dtype=np.int32)
    for obs_id, grid_id in spec.combined_to_grid.items():
        c2g_np[obs_id] = grid_id
    combined_to_grid_jax = jnp.array(c2g_np, dtype=jnp.int32)

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

    print(f"\nTraining tcirl for {rc.total_steps} steps x {rc.num_runs} seeds...")

    cause_index = env.cause_index
    effect_index = env.disease_index

    # Training drives the evaluation too, so the Q-tables and the hypotheses it returns are
    # already measured by the time it returns and nothing here reads them again.
    trained = tcirl.train(
        env,
        logger=logger,
        recorder=recorder,
        grid_alphabet=grid_alphabet,
        cause_dfa_size=cause_dfa_size,
        cause_index=cause_index,
        effect_index=effect_index,
        init_hypothesis_dfas=init_hypotheses,
        inference_mode=mode,
        combined_to_grid=combined_to_grid_jax,
        x_r_obs_ids=spec.x_r_obs_ids,
        x_a_obs_ids=spec.x_a_obs_ids,
        self_loop_symbol=self_loop_symbol,
        max_dfa_size=cause_dfa_size,
        initial_rejecting=tc.initial_rejecting,
        symmetry_breaking=tc.symmetry_breaking,
        value_transfer=tc.value_transfer,
        counterfactual=tc.counterfactual,
        gamma=p.gamma,
        step_size_scale=p.step_size_scale,
        epsilon_start=p.epsilon.start,
        epsilon_end=p.epsilon.end,
        epsilon_decay_steps=p.epsilon.decay_steps,
        max_episode_steps=p.max_episode_steps,
        q_init=p.q_init,
        seed_offset=rc.seed_offset,
        num_runs=rc.num_runs,
        schedule=schedule,
        smoothing_steps=cadence.smoothing_steps,
        evaluation=evaluation,
    )

    evaluation.finish()

    recorder.save()
    # After the recorder, so a failure here cannot cost the run its curves.
    save_final_policy(
        logger.run_dir,
        logger,
        tcirl_arrays(trained.evaluated_q_tables, trained.evaluated_hypotheses),
        algo=config.algo,
    )
    logger.finish_run()


if __name__ == "__main__":
    main()
