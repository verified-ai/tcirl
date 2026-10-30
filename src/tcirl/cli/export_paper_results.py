#!/usr/bin/env python
"""CLI: turn export snapshots into the paper's `results/` tree.

`--project` may carry a `{domain}` placeholder, so one call covers both domains:
`--project 'tcirl-{domain}-paper'` reads `tmp/results/gene_editing/tcirl-gene_editing-paper/latest`
and `tmp/results/traffic_signal/tcirl-traffic_signal-paper/latest`.

The last step of the pipeline. `scripts/<domain>/reproduce.sh` trains every arm and exports every
per-seed curve into a timestamped snapshot; this copies the subset the paper's LaTeX reads into the
names and columns it reads them under, and assembles the figure of evaluation trajectories that
reach Phase 2 across the five arms (`tcirl.paper_results`).

Point `--out_dir` at any directory, or leave it to land under `tmp/paper_results/`.

This entry point pins JAX to the CPU, as every entry point that needs no GPU does. It copies
columns and computes one closed form, the Wilson interval of
`tcirl.metrics.intervals`, in plain Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from simple_parsing import ArgumentParser

from tcirl.jax_platform import pin_jax_to_cpu
from tcirl.paper_results import ACCEPT_METRIC, project_snapshot
from tcirl.path_consts import TMP_DIR

# The domains the paper prints, each with the export family the launcher writes under.
DOMAINS: tuple[str, ...] = ("gene_editing", "traffic_signal")


@dataclass(frozen=True, slots=True, kw_only=True)
class ExportPaperResultsConfig:
    # The W&B project the reproduce launchers were pointed at; it names the snapshot directory.
    # `{domain}` in it is replaced by each domain's name, which is what a run of both domains
    # needs: the group names are the paper's arm names and are the same in both domains, so
    # one project holding both would put two runs in every group and every resolver refuses that
    # (`tcirl.export_runs.exactly_one_run`). Hence one project per domain, and one flag that
    # names them both.
    project: str
    # Which domains to project. Empty means both.
    domains: list[str] = field(default_factory=list[str])
    # Where the paper's `results/` tree goes. Empty = `tmp/paper_results`.
    out_dir: str = ""
    # Which snapshot of each domain to read. Empty = the `latest` symlink the launcher maintains.
    snapshot: str = ""


def main() -> None:
    pin_jax_to_cpu()
    parser = ArgumentParser(description="Project export snapshots onto the paper's results tree")
    parser.add_arguments(ExportPaperResultsConfig, dest="config")
    args = parser.parse_args()
    config = args.config
    assert isinstance(config, ExportPaperResultsConfig)

    out_root = Path(config.out_dir) if config.out_dir else TMP_DIR / "paper_results"
    domains = tuple(config.domains) if config.domains else DOMAINS
    unknown = [domain for domain in domains if domain not in DOMAINS]
    if unknown:
        raise ValueError(f"unknown domain(s) {unknown}; expected any of {list(DOMAINS)}")

    snapshot_name = config.snapshot or "latest"
    for domain in domains:
        project = config.project.replace("{domain}", domain)
        snapshot = TMP_DIR / "results" / domain / project / snapshot_name
        print(f"=== {domain}: reading {snapshot} ===")
        written = project_snapshot(snapshot, domain, out_root / "results")
        for entry in written:
            print(
                f"  {entry.target.relative_to(out_root)} <- {entry.source.name} "
                f"({entry.rows} rows, {entry.statistic})"
            )
        print(
            f"  results/{domain}/accept_state.csv <- each arm's {ACCEPT_METRIC}.csv (mean, 95% CI)"
        )
    print(f"\nDone. The paper's tree is under {out_root / 'results'}")


if __name__ == "__main__":
    main()
