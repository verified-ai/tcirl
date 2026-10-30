"""Reproduce the selected traffic cycles from the environment model, without training."""

from __future__ import annotations

import argparse
from pathlib import Path

from tcirl.config import TrafficSignalQLConfig, parse_config
from tcirl.envs.traffic_signal import TrafficSignalEnv
from tcirl.export_runs import single_finished_run, survey_local_runs
from tcirl.jax_platform import pin_jax_to_cpu
from tcirl.path_consts import CONFIG_DIR, ROOT_DIR
from tcirl.specifications.traffic_signal import create_traffic_signal_cause
from tcirl.traffic_cadence import (
    CadenceSettings,
    screen_cadences,
    validate_comparison_settings,
    write_cadence_optimum,
    write_cadence_results,
    write_cadence_source_records,
)


def main() -> None:
    pin_jax_to_cpu()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config_path", type=Path, default=CONFIG_DIR / "traffic_signal_q_learning.yaml"
    )
    parser.add_argument("--max-period", type=int, default=12)
    parser.add_argument("--coarse-episodes", type=int, default=16)
    parser.add_argument("--fine-episodes", type=int, default=256)
    parser.add_argument("--nominees", type=int, default=200)
    parser.add_argument("--holdout-episodes", type=int, default=1024)
    parser.add_argument(
        "--training-project", help="check the finished local traffic run's settings"
    )
    args = parser.parse_args()
    config_path: Path = args.config_path
    if not config_path.resolve().is_relative_to(ROOT_DIR.resolve()):
        raise ValueError(
            "--config_path must name a configuration inside the repository so its source remains reproducible"
        )
    config = parse_config(
        TrafficSignalQLConfig,
        args.config_path,
        "Traffic environment of the cycle screen",
        argv=["--cause_known", "false"],
    )
    if args.training_project is not None:
        run = single_finished_run(
            survey_local_runs(ROOT_DIR / "wandb"), args.training_project, "q_learning-cause_known"
        )
        trained = parse_config(
            TrafficSignalQLConfig,
            run.directory / "files/run_config.yaml",
            "Recorded traffic configuration",
            argv=[],
        )
        validate_comparison_settings(config, trained)
    settings = CadenceSettings(
        max_period=args.max_period,
        horizon=config.params.max_episode_steps,
        coarse_episodes=args.coarse_episodes,
        fine_episodes=args.fine_episodes,
        nominees=args.nominees,
        holdout_episodes=args.holdout_episodes,
        comparison_episodes=config.run.eval_episodes,
    )
    spec = create_traffic_signal_cause(config.env.difficulty)
    env = TrafficSignalEnv.from_config(config.env, spec, cause_known=False)
    cells, candidates = screen_cadences(env, settings)
    write_cadence_results(args.output, cells)
    optimum, exact = write_cadence_optimum(
        args.output, env, cells, settings.horizon, gamma=config.params.gamma
    )
    write_cadence_source_records(
        args.output,
        config=config,
        settings=settings,
        candidates=candidates,
        command=[
            "uv",
            "run",
            "python",
            "-m",
            "tcirl.cli.traffic_cadence",
            "--output",
            "tmp/traffic_cadence",
            "--config_path",
            str(config_path.resolve().relative_to(ROOT_DIR.resolve())),
            "--max-period",
            str(settings.max_period),
            "--coarse-episodes",
            str(settings.coarse_episodes),
            "--fine-episodes",
            str(settings.fine_episodes),
            "--nominees",
            str(settings.nominees),
            "--holdout-episodes",
            str(settings.holdout_episodes),
            *(
                ["--training-project", args.training_project]
                if args.training_project is not None
                else []
            ),
        ],
    )
    for cell, value in zip(cells, exact, strict=True):
        print(
            f"{cell.frames} frames, cause={cell.with_cause}, {cell.word}: exact {value:.8f}; holdout {cell.holdout_returns.mean():.8f}; canonical comparison {cell.comparison_returns.mean():.8f}"
        )
    print(f"discount-optimal stationary policy: expected episode return {optimum:.8f}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
