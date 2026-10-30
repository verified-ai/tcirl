"""TCIRL training session: sigterm handling, W&B logger setup, seeding.

Run artifacts belong in the W&B run's own files directory -- `WandbLogger.run_dir`,
unique per run. `LoggingConfig.upload` chooses only whether that run is online or
offline: an offline run still records every metric and artifact locally, so turning
uploads off never costs you results. Writing a file there does NOT register it for
upload: every artifact must be handed to `WandbLogger.save_file` / `save_text`.
"""

from __future__ import annotations

import os
import random
import signal
from pathlib import Path
from typing import Any

import numpy as np
from simple_parsing import Serializable

from tcirl.config import (
    GeneEditingDqnConfig,
    GeneEditingQLConfig,
    GeneEditingTcirlConfig,
    LoggingConfig,
    TrafficSignalDqnConfig,
    TrafficSignalQLConfig,
    TrafficSignalTcirlConfig,
)
from tcirl.logging.explanations import log_run_explanations
from tcirl.logging.wandb import WandbLogger
from tcirl.metrics.spec import MetricRegistry
from tcirl.path_consts import ROOT_DIR

type _AnyConfig = (
    GeneEditingQLConfig
    | GeneEditingTcirlConfig
    | GeneEditingDqnConfig
    | TrafficSignalQLConfig
    | TrafficSignalTcirlConfig
    | TrafficSignalDqnConfig
)


def install_sigterm_handler() -> None:
    """Convert SIGTERM into SystemExit so finally blocks (W&B cleanup) run."""

    def _handler(_signum: int, _frame: object) -> None:
        raise SystemExit(1)

    signal.signal(signal.SIGTERM, _handler)


def init_wandb_logger(
    logging_cfg: LoggingConfig,
    *,
    specs: MetricRegistry,
    method: str,
    config_dict: dict[str, Any],
    tags: list[str],
    default_project: str,
    base_dir: Path,
    default_name: str | None = None,
    default_job_type: str | None = None,
) -> WandbLogger:
    """Create a WandbLogger and start a run, online when `logging_cfg.upload` is set.

    A run is always started: `upload=False` makes it an offline run, which records
    metrics and artifacts locally and stays syncable, rather than discarding them.
    `base_dir` is the parent under which W&B lays out `wandb/run-<timestamp>-<id>/`;
    the run's own files directory is then available as `WandbLogger.run_dir`.
    """
    logger = WandbLogger(specs)
    logger.init_run(
        project=logging_cfg.project or default_project,
        entity=logging_cfg.entity,
        method=method,
        config=config_dict,
        tags=tags,
        group=logging_cfg.group,
        job_type=logging_cfg.job_type or default_job_type or method,
        name=logging_cfg.name or default_name,
        notes=logging_cfg.notes,
        base_dir=base_dir,
        run_id=logging_cfg.run_id,
        online=logging_cfg.upload,
    )
    return logger


def save_run_config(cfg: Serializable, logger: WandbLogger) -> None:
    """Snapshot the run's config into the run dir as `run_config.yaml` and upload it.

    Not `config.yaml`: W&B writes its own `config.yaml` (the tracked config plus
    run metadata) into the same directory and would clobber the snapshot. This
    one is the verbatim training config, re-loadable with `--config_path`.
    """
    run_dir = logger.run_dir
    if run_dir is None:
        return
    config_path = run_dir / "run_config.yaml"
    cfg.save(config_path)
    logger.save_file(config_path, base_path=run_dir)


def require_pythonhashseed(seed: int) -> None:
    """Require PYTHONHASHSEED to be explicitly pinned to the run seed."""
    if "PYTHONHASHSEED" not in os.environ:
        raise ValueError(
            "PYTHONHASHSEED environment variable needs to be set explicitly (`PYTHONHASHSEED=<seed> uv run ...`)."
        )
    if os.environ["PYTHONHASHSEED"] != str(seed):
        raise ValueError(
            f"PYTHONHASHSEED is {os.environ['PYTHONHASHSEED']}, but should be {seed} for full determinism."
        )


def seed_python_process(seed: int) -> None:
    """Seed Python and NumPy process-global RNG state."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002


def _default_run_name(config: _AnyConfig) -> str:
    """The arm's name, which is also the directory its results land in.

    These names are the paper's result directories (`results/<domain>/<name>/`), so changing one
    changes where a figure looks for its CSV. The DQN's window length is part of its name because
    the paper reports two window lengths as two arms.
    """
    match config:
        case GeneEditingQLConfig() | TrafficSignalQLConfig():
            cause_tag = "known" if config.cause_known else "unknown"
            return f"q_learning-cause_{cause_tag}"
        case GeneEditingTcirlConfig() | TrafficSignalTcirlConfig():
            return f"tcirl-{config.tcirl.inference_mode}"
        case GeneEditingDqnConfig() | TrafficSignalDqnConfig():
            cause_tag = "known" if config.cause_known else "unknown"
            return f"dqn-cause_{cause_tag}_{config.params.history_len}"


def init_logger(
    config: _AnyConfig,
    env_name: str,
    specs: MetricRegistry,
) -> WandbLogger:
    """Init the W&B logger for a tcirl run and snapshot the config, online iff `logging.upload`.

    `specs` is the run's metric catalog (`tcirl.metrics.registry`): it decides which series exist,
    how each reduces across seeds, and what the run's `explanations/metrics` legend says. Every
    caller passes its domain's registry, so a metric no arm of that domain emits cannot be named.
    """
    lc = config.logging
    logger = init_wandb_logger(
        lc,
        specs=specs,
        method=config.algo,
        config_dict=config.to_dict(),  # pyright: ignore[reportUnknownArgumentType]
        tags=config.all_tags,
        default_project=f"tcirl-{env_name}",
        base_dir=ROOT_DIR,
        default_name=_default_run_name(config),
        default_job_type=config.algo,
    )
    save_run_config(config, logger)
    log_run_explanations(logger, note=lc.notes)
    return logger
