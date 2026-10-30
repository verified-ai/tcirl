#!/usr/bin/env python
"""CLI: export every per-seed curve of a group's runs, aggregated across seeds.

One arm is one run (the seeds are vmapped inside it), so a group is normally a single run and this
is the exporter the reproduce launchers call. With no `--metrics` it auto-discovers every curve the
runs recorded -- the training curves, the inference curves and the evaluation trajectories alike,
because all three travel one recorder.

Runs are read from the local `wandb/` directory by default -- an offline run (`--noupload`) records
everything an export needs on disk -- and from the W&B API only as a fallback or when
`--source wandb` asks for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tcirl.cli._export_cli import RunSourceArgs, parse_export_config
from tcirl.export import export_group_metrics
from tcirl.jax_platform import pin_jax_to_cpu


@dataclass(frozen=True, slots=True, kw_only=True)
class ExportGroupMetricsConfig(RunSourceArgs):
    project: str
    group: str
    metrics: list[str] = field(default_factory=list[str])
    output_dir: str | None = None


def main() -> None:
    pin_jax_to_cpu()
    config, out = parse_export_config(
        ExportGroupMetricsConfig,
        "Export every per-seed curve of a group's runs, reduced across seeds, from disk by default",
    )
    export_group_metrics(
        config.project,
        config.group,
        config.metrics,
        out,
        source=config.source,
        local_root=config.local_root(Path.cwd() / "wandb"),
    )


if __name__ == "__main__":
    main()
