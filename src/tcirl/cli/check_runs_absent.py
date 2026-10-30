#!/usr/bin/env python
"""CLI: refuse to start training when a project already holds a run of one of its groups.

An arm's seeds are vmapped inside one process, so one arm is one run and a second run of the same
`(project, group)` is a relaunch. The export refuses to reduce two runs of one arm
(`~tcirl.export_runs.exactly_one_run`), and this is the same question asked before the GPU hour
rather than after it: the launcher calls it, so a project name that was already used fails in
seconds instead of at the export at the end.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from simple_parsing import ArgumentParser

from tcirl.export_runs import survey_local_runs
from tcirl.jax_platform import pin_jax_to_cpu
from tcirl.path_consts import ROOT_DIR


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckRunsAbsentConfig:
    project: str
    groups: list[str] = field(default_factory=list[str])
    wandb_dir: str = ""


def main() -> None:
    pin_jax_to_cpu()
    parser = ArgumentParser(description="Fail when a project already holds runs of these groups")
    parser.add_arguments(CheckRunsAbsentConfig, dest="config")
    args = parser.parse_args()
    config = args.config
    assert isinstance(config, CheckRunsAbsentConfig)

    # `ROOT_DIR`, not the working directory: that is where `tcirl.session` makes W&B write, so a
    # default keyed to the caller's cwd would answer about an empty directory and pass in silence.
    root = Path(config.wandb_dir) if config.wandb_dir else ROOT_DIR / "wandb"
    if not root.exists():
        print(f"  no run directory at {root} yet")
        return
    survey = survey_local_runs(root)
    for path, reason in survey.unreadable:
        # It cannot be attributed to a group, so it cannot be refused -- but a damaged directory
        # may well be one of these groups, and then the pass is wrong.
        print(f"  warning: cannot identify run directory {path.name} ({reason})", file=sys.stderr)
    taken = {
        group: [run.label for run in survey.matching(config.project, group)]
        for group in config.groups
    }
    occupied = {group: labels for group, labels in taken.items() if labels}
    if not occupied:
        return

    print(
        f"Project {config.project!r} already holds runs of "
        f"{len(occupied)} of the {len(config.groups)} groups about to be trained, under {root}:",
        file=sys.stderr,
    )
    for group, labels in occupied.items():
        print(f"  {group}: {', '.join(labels)}", file=sys.stderr)
    print(
        "One arm is one run, so training these again would leave two runs per group and the "
        "export would refuse to reduce them.\n"
        "  Train into a fresh --project, or move the listed run directories out of "
        f"{root} first.",
        file=sys.stderr,
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
