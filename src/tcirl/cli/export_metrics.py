#!/usr/bin/env python
"""CLI: export one run's per-seed metric curves to CSV + SVG, by run id."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tcirl.cli._export_cli import RunSourceArgs, parse_export_config
from tcirl.export import export_metrics
from tcirl.jax_platform import pin_jax_to_cpu
from tcirl.logging.wandb import resolve_entity


@dataclass(frozen=True, slots=True, kw_only=True)
class ExportMetricsConfig(RunSourceArgs):
    project: str
    run_id: str
    metrics: list[str] = field(default_factory=list[str])
    output_dir: str | None = None


def main() -> None:
    pin_jax_to_cpu()
    config, out = parse_export_config(
        ExportMetricsConfig,
        "Export one run's metrics from its per-seed curves or its scalar history",
    )
    export_metrics(
        resolve_entity(None) if config.source == "wandb" else None,
        config.project,
        config.run_id,
        config.metrics,
        out,
        source=config.source,
        local_root=config.local_root(Path.cwd() / "wandb"),
    )


if __name__ == "__main__":
    main()
