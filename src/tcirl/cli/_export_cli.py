"""Shared argument-parsing scaffold for the export CLI entry points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from simple_parsing import ArgumentParser
from simple_parsing.utils import Dataclass

from tcirl.export_runs import RunSource
from tcirl.path_consts import TMP_DIR


@dataclass(frozen=True, slots=True, kw_only=True)
class RunSourceArgs:
    """Where a grouped export reads its runs from. Inherit this in an export CLI's config."""

    # Where the runs come from. `local` reads the run directories training wrote under
    # `--wandb_dir` and never touches the network (it fails naming that directory if the group
    # is not there, or if one of its runs has not finished); `wandb` always queries the W&B API,
    # which only sees uploaded runs; `auto` (the default) reads local data and falls back to the
    # API when no local run matches or a matching one is still training, killed, or crashed.
    # None of the three will export a group that has not finished: the API's run state is
    # checked just as the local exit record is.
    source: RunSource = "auto"
    # Directory holding the `run-*` / `offline-run-*` directories. Empty = the repository's own
    # `wandb/`, which is where training writes them.
    wandb_dir: str = ""

    def local_root(self, default: Path) -> Path:
        """The directory to scan for local runs: `--wandb_dir` when given, else `default`."""
        return Path(self.wandb_dir) if self.wandb_dir else default


class ExportCliConfig(Dataclass, Protocol):
    """Export CLI configs share an optional output directory (None = the default results dir)."""

    @property
    def output_dir(self) -> str | None: ...


def parse_export_config[T: ExportCliConfig](
    config_cls: type[T], description: str
) -> tuple[T, Path]:
    """Parse an export CLI's arguments into `(config, output_dir)`."""
    parser = ArgumentParser(description=description)
    parser.add_arguments(config_cls, dest="config")
    args = parser.parse_args()
    config = args.config
    assert isinstance(config, config_cls)
    out = TMP_DIR / "results" if config.output_dir is None else Path(config.output_dir)
    return config, out
