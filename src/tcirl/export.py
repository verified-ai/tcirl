"""Export per-seed metric curves to CSV + SVG: the one reduction path of this repo.

Runs reach this module as `~tcirl.export_runs.ExportRun`s, which are read from the local
`wandb/` directory by default and from the W&B API on request or as a fallback. Everything here
-- pooling, smoothing, bootstrapping, rendering -- is backend-agnostic and shared by both.

Training curves and evaluation trajectories come through the same door: both are per-seed arrays
persisted by `~tcirl.logging.recorder.PerSeedMetricRecorder`, and each payload names its own x axis
(`x_label`: the env step for a training curve, the timestep inside one episode for a trajectory).
So there is exactly one reducer, one bootstrap and one CSV writer for every curve the paper prints.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt

from tcirl.export_runs import (
    ExportRun,
    RunSource,
    resolve_group_run,
    resolve_run,
    write_export_metadata,
)
from tcirl.logging.naming import safe_filename
from tcirl.metrics.smoothing import ema_alpha, smooth_per_seed_curve
from tcirl.metrics.stats import CurveStats, aggregate_stats
from tcirl.path_consts import (
    FINAL_INTERVALS_FILE,
    PER_SEED_SAMPLE_FILE,
    PER_SEED_SAMPLE_PREFIX,
    PRINTED_DRAWS,
)


def _run_smoothing_steps(run: ExportRun) -> int | None:
    """The run's EMA horizon in ENV STEPS from its tracked `cadence`, or None when smoothing is off.

    Smoothing is reproduced offline from the run's own config (the single source of truth), so the
    exported figures match the live wandb curves. `logging` is excluded from the W&B config, which
    is why `smoothing_steps` lives on `cadence`.
    """
    cadence: dict[str, Any] = run.config.get("cadence", {})
    smoothing_steps = cadence.get("smoothing_steps")
    return None if smoothing_steps is None else int(smoothing_steps)


def _curve_alpha(metric_payload: dict[str, Any], smoothing_steps: int | None) -> float | None:
    """The per-point EMA weight for ONE curve, or None when that curve carries no EMA.

    `smoothing_steps` is a horizon in env steps, and the weight that realises it depends on how far
    apart the curve's own points are: a training curve steps by `log_interval` and an evaluation
    curve by the grid's spacing. Deriving it from the curve rather than from one config field is
    what makes both panels smooth over the same 10000 steps, and it reads `log_interval` off a
    training curve exactly, so nothing about those curves moves.

    A curve whose x values are not evenly spaced has no single weight and raises rather than pick
    one. A curve of fewer than two points has nothing to smooth.
    """
    if smoothing_steps is None or not metric_payload.get("smoothable", False):
        return None
    steps: list[float] = metric_payload["steps"]
    if len(steps) < 2:
        return None
    spacings = {steps[i + 1] - steps[i] for i in range(len(steps) - 1)}
    if len(spacings) != 1:
        raise ValueError(
            f"a smoothable curve must have evenly spaced x values to carry a step-timescale EMA; "
            f"this one steps by {sorted(spacings)}"
        )
    spacing = spacings.pop()
    if spacing <= 0:
        raise ValueError(f"a curve's x values must increase; this one steps by {spacing}")
    return ema_alpha(smoothing_steps, int(spacing))


# A seed counts as mostly in Phase 2 when its share of evaluation episodes in Phase 2, smoothed by the run's EMA, exceeds one half.
# The evaluation records the indicator of each checkpoint's raw share; the export judges a seed on the smoothed share, over the same horizon as its smoothed return, and the result is still one 0 or 1 per seed, so the share of seeds keeps its Wilson interval.
MOSTLY_IN_ACCEPT = "eval_final/mostly_in_accept"
EPISODES_IN_ACCEPT = "eval_final/frac_episodes_in_accept"


def plotted_payloads(per_seed: Mapping[str, Any], smoothing_steps: int | None) -> dict[str, Any]:
    """Every per-seed curve as the CSV and the figures hold it: each smoothable curve smoothed, and the seed indicator `MOSTLY_IN_ACCEPT` taken on the smoothed episode share."""
    plotted: dict[str, Any] = {
        key: {**payload, "values": _smoothed_values(payload, smoothing_steps)}
        for key, payload in per_seed.items()
    }
    if MOSTLY_IN_ACCEPT not in per_seed:
        return plotted
    share: dict[str, Any] = per_seed[EPISODES_IN_ACCEPT]
    if share["steps"] != per_seed[MOSTLY_IN_ACCEPT]["steps"]:
        raise ValueError(f"{MOSTLY_IN_ACCEPT} and {EPISODES_IN_ACCEPT} lie on different grids")
    smoothed_share = _smoothed_values({**share, "smoothable": True}, smoothing_steps)
    plotted[MOSTLY_IN_ACCEPT] = {
        **per_seed[MOSTLY_IN_ACCEPT],
        "values": [[float(value > 0.5) for value in row] for row in smoothed_share],
    }
    return plotted


def plotted_curves(run: ExportRun) -> dict[str, Any]:
    """The run's per-seed payload as the CSV and the figures hold it, with the run's EMA applied by `plotted_payloads`.

    `ExportRun.per_seed_curves` returns the raw curves the recorder persisted.
    """
    curves = run.per_seed_curves()
    if curves is None:
        raise ValueError(f"{run.label} recorded no per-seed curves")
    return plotted_payloads(curves, _run_smoothing_steps(run))


def _smoothed_values(
    metric_payload: dict[str, Any], smoothing_steps: int | None
) -> list[list[float]]:
    """The metric's per-seed curves, EMA-smoothed over `smoothing_steps` env steps where declared."""
    values: list[list[float]] = metric_payload["values"]
    alpha = _curve_alpha(metric_payload, smoothing_steps)
    if alpha is None:
        return values
    return smooth_per_seed_curve(values, alpha)


def write_csv(
    path: Path,
    header: list[str],
    rows: Iterable[Iterable[str | float | int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def render_curve_svg(
    path: Path,
    *,
    steps: Sequence[float],
    curves: Sequence[CurveStats],
    xlabel: str = "Step",
    ylabel: str = "",
    title: str = "",
) -> None:
    """Curve figure: the faint min..max seed spread, the mean with its 95% bootstrap CI band, and the dashed median."""
    matplotlib.use("Agg")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(
        steps,
        [c.min for c in curves],
        [c.max for c in curves],
        alpha=0.12,
        color="tab:blue",
        label="min-max spread",
    )
    ax.fill_between(
        steps,
        [c.mean_ci_lo for c in curves],
        [c.mean_ci_hi for c in curves],
        alpha=0.25,
        color="tab:blue",
        label="95% CI",
    )
    ax.plot(steps, [c.mean for c in curves], color="tab:blue", linewidth=1.5, label="Mean")
    ax.plot(
        steps,
        [c.median for c in curves],
        color="tab:orange",
        linewidth=1.0,
        linestyle="--",
        label="Median",
    )
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg")
    plt.close(fig)


def render_scalar_svg(
    path: Path,
    *,
    steps: Sequence[float],
    values: Sequence[float],
    xlabel: str = "Step",
    ylabel: str = "",
    title: str = "",
) -> None:
    matplotlib.use("Agg")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, values, color="tab:blue", linewidth=1.5)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg")
    plt.close(fig)


# The one seed every bootstrap in this repo is drawn from; each point folds its step into it.
_BOOTSTRAP_SEED = 42


def compute_curve_stats(
    steps: Sequence[int],
    values: Sequence[Sequence[float]],
) -> tuple[list[float], list[CurveStats]]:
    """Per-step `CurveStats` (every point stat and the bootstrap CI of the mean) across the per-seed arrays.

    The bootstrap key is FOLDED FROM THE STEP, so a point's interval is a function of that point's
    own seeds and of nothing else. Splitting one key along the curve instead would make the interval
    at a given step depend on how many points the curve holds, so the same metric would report
    two different intervals for the same 50 numbers depending on the size of the evaluation grid.
    Keyed this way, two runs that agree on the values agree on the bounds,
    and a check that two grids agree can be exact instead of tolerant.
    """
    steps_out: list[float] = []
    curves: list[CurveStats] = []
    for step, row_values in zip(steps, values, strict=True):
        vals = jnp.asarray(row_values, dtype=jnp.float32)
        mask = ~jnp.isnan(vals)
        if not bool(jnp.any(mask)):
            continue  # no seed has data at this step
        curves.append(
            aggregate_stats(
                vals, mask=mask, key=jax.random.fold_in(jax.random.key(_BOOTSTRAP_SEED), int(step))
            )
        )
        steps_out.append(float(step))
    return steps_out, curves


def export_per_seed_metric(
    metric: str,
    metric_payload: dict[str, Any],
    output_dir: Path,
    *,
    title_suffix: str = "",
    smoothing_steps: int | None = None,
) -> None:
    """Write one metric's CSV and curve SVG: the mean with its bootstrap CI, the median, the seed range and the seed count per point.

    `smoothing_steps` (the run's EMA horizon in env steps) smooths the per-seed curves first when the metric is smoothable, at the weight this curve's own x spacing calls for, so the figure matches the live wandb curve.
    """
    steps = metric_payload["steps"]
    values = _smoothed_values(metric_payload, smoothing_steps)
    steps_out, curves = compute_curve_stats(steps, values)
    # The payload names its own x axis (`tcirl.logging.recorder`): env steps for a training curve,
    # the timestep inside one episode for an evaluation trajectory.
    x_label: str = metric_payload.get("x_label", "step")

    safe = safe_filename(metric)
    header = [
        x_label,
        "n",
        "mean",
        "mean_ci_lo",
        "mean_ci_hi",
        "median",
        "min",
        "max",
    ]
    rows = [
        [
            f"{s:.0f}",
            f"{c.n:d}",
            f"{c.mean:.6g}",
            f"{c.mean_ci_lo:.6g}",
            f"{c.mean_ci_hi:.6g}",
            f"{c.median:.6g}",
            f"{c.min:.6g}",
            f"{c.max:.6g}",
        ]
        for s, c in zip(steps_out, curves, strict=True)
    ]
    csv_path = output_dir / f"{safe}.csv"
    write_csv(csv_path, header, rows)
    print(f"    CSV -> {csv_path}")

    name_part = metric.rsplit("/", maxsplit=1)[-1]
    svg_path = output_dir / f"{safe}.svg"
    render_curve_svg(
        svg_path,
        steps=steps_out,
        curves=curves,
        xlabel=x_label.capitalize(),
        ylabel=name_part,
        title=f"{name_part}{title_suffix}",
    )
    print(f"    SVG -> {svg_path}")


def export_per_seed_sample(
    per_seed: Mapping[str, Any], output_dir: Path, *, smoothing_steps: int | None
) -> Path | None:
    """Write every `eval_final/*` metric's 50 per-seed values at the last evaluation point.

    This is the one file of an export that holds a SAMPLE rather than a reduction of one.
    Everything else an export writes is 50 seeds reduced to a centre and an interval, which cannot
    show that an arm's seeds fall into more than one group; this can, and the paper's seed-score
    panel reads it.

    One row per metric, named by the same stem its curve CSV carries, and one column per seed in
    SEED ORDER, so seed `i` of two arms is the same seed and a reader can pair them. The values are
    the ones the plotted curve ends at: smoothing is applied first, exactly as
    `export_per_seed_metric` applies it, or the sample would disagree with the endpoint every
    figure and every table prints.

    Returns the path written, or None when the run recorded no evaluation metric at all.
    """
    stems = sorted(key for key in per_seed if key.startswith(PER_SEED_SAMPLE_PREFIX))
    if not stems:
        return None
    rows: list[list[str]] = []
    seeds = 0
    for metric in stems:
        payload: dict[str, Any] = per_seed[metric]
        final = _smoothed_values(payload, smoothing_steps)[-1]
        if seeds and len(final) != seeds:
            raise ValueError(
                f"{metric} holds {len(final)} seeds where an earlier metric of the same run holds "
                f"{seeds}; one run is one seed bank"
            )
        seeds = len(final)
        rows.append([safe_filename(metric), *(f"{value:.6g}" for value in final)])
    header = ["metric", *(f"seed_{i}" for i in range(seeds))]
    path = output_dir / PER_SEED_SAMPLE_FILE
    write_csv(path, header, rows)
    print(f"    CSV -> {path}")
    return path


def printed_interval(values: Sequence[float], step: int) -> CurveStats:
    """The mean of one point's per-seed values with its 95% bootstrap interval from `PRINTED_DRAWS` resamples, keyed by the point's step as every point of a curve is."""
    array = jnp.asarray(values, dtype=jnp.float32)
    return aggregate_stats(
        array,
        mask=~jnp.isnan(array),
        key=jax.random.fold_in(jax.random.key(_BOOTSTRAP_SEED), step),
        n_bootstrap=PRINTED_DRAWS,
    )


def export_final_intervals(per_seed: Mapping[str, Any], output_dir: Path) -> Path | None:
    """Write every `eval_final/*` metric's mean and 95% bootstrap interval at the last evaluation point, from `PRINTED_DRAWS` resamples.

    The values are the ones `export_per_seed_sample` writes, so `per_seed` must already be the plotted payload; `printed_interval` reduces each.
    Returns the path written, or None when the run recorded no evaluation metric at all.
    """
    stems = sorted(key for key in per_seed if key.startswith(PER_SEED_SAMPLE_PREFIX))
    path = output_dir / FINAL_INTERVALS_FILE
    if not stems:
        # A file an earlier export left would pair another run's intervals with this run's curves.
        path.unlink(missing_ok=True)
        return None
    rows: list[list[str]] = []
    for metric in stems:
        payload: dict[str, Any] = per_seed[metric]
        step = int(payload["steps"][-1])
        stats = printed_interval(payload["values"][-1], step)
        rows.append(
            [
                safe_filename(metric),
                str(step),
                str(stats.n),
                f"{stats.mean:.6g}",
                f"{stats.mean_ci_lo:.6g}",
                f"{stats.mean_ci_hi:.6g}",
                str(PRINTED_DRAWS),
            ]
        )
    write_csv(path, ["metric", "step", "n", "mean", "mean_ci_lo", "mean_ci_hi", "draws"], rows)
    print(f"    CSV -> {path}")
    return path


def export_scalar_metric(run: ExportRun, metric: str, output_dir: Path) -> None:
    print(f"  fetching {metric} ...", flush=True)
    pairs = run.scalar_history(metric)

    if not pairs:
        print(f"    no data found for {metric}")
        return

    steps = [step for step, _ in pairs]
    values = [value for _, value in pairs]

    safe = safe_filename(metric)
    csv_path = output_dir / f"{safe}.csv"
    rows = [[f"{step:.0f}", f"{value:.6g}"] for step, value in zip(steps, values, strict=True)]
    write_csv(csv_path, ["step", "value"], rows)
    print(f"    CSV -> {csv_path}")

    name_part = metric.rsplit("/", maxsplit=1)[-1]
    svg_path = output_dir / f"{safe}.svg"
    render_scalar_svg(svg_path, steps=steps, values=values, ylabel=name_part, title=name_part)
    print(f"    SVG -> {svg_path}")


def export_run_metrics(run: ExportRun, metrics: list[str], output_dir: Path) -> None:
    """Export a run's curves: every per-seed curve it recorded, or just the ones named.

    The one export body. Naming a run by id and naming it by group differ only in how the run was
    resolved, so the two cannot write different files. Smoothing is reproduced from the run's own
    cadence and applied per seed by `plotted_payloads`, and `export_per_seed_metric` makes the single
    cross-seed reduction. `export_per_seed_sample` writes the one unreduced file beside
    them, and `export_final_intervals` the final intervals that the paper prints.
    """
    # The plotted payloads are already smoothed, so the two writers below are called with no horizon of their own.
    per_seed = plotted_payloads(run.per_seed_curves() or {}, _run_smoothing_steps(run))
    # From the FULL payload rather than from `metrics`: the sample describes the run, so naming one
    # metric on the command line must not leave a file behind that holds one row of five.
    export_per_seed_sample(per_seed, output_dir, smoothing_steps=None)
    export_final_intervals(per_seed, output_dir)
    if not metrics:
        if not per_seed:
            print("  no per-seed metric curves found in run files")
            return
        for metric in sorted(per_seed):
            print(f"  exporting {metric} ...", flush=True)
            export_per_seed_metric(metric, per_seed[metric], output_dir, smoothing_steps=None)
        return
    for metric in metrics:
        metric_payload = per_seed.get(metric)
        if isinstance(metric_payload, dict):
            export_per_seed_metric(
                metric,
                metric_payload,  # pyright: ignore[reportUnknownArgumentType]
                output_dir,
                smoothing_steps=None,
            )
            continue
        export_scalar_metric(run, metric, output_dir)


def export_metrics(
    entity: str | None,
    project: str,
    run_id: str,
    metrics: list[str],
    results_dir: Path,
    *,
    source: RunSource = "auto",
    local_root: Path | None = None,
) -> None:
    """Export one W&B run's metrics from its per-seed curves or scalar histories.

    With no `metrics` every per-seed curve the run recorded is exported, exactly as
    `export_group_metrics` does for a group: both call `export_run_metrics`, so naming a run by id
    and naming it by group cannot produce different files.

    Acquisition is local-first through `~tcirl.export_runs.resolve_run`, so an offline
    (`--upload False`) run exports from its run directory with no network. The run is held to
    the same completeness gate as a group export, for the same reason: a run named by id is no
    more finished than one found by group, and a CSV written from a crashed run reads as the
    final number either way.
    """
    resolved = resolve_run(
        project=project, run_id=run_id, entity=entity, source=source, local_root=local_root
    )
    run = resolved.run
    print(f"Run {run_id}: reading from {resolved.origin} ({resolved.location})")
    run_name = run.name
    output_dir = results_dir / safe_filename(run_name)
    write_export_metadata(
        output_dir,
        [run],
        project=project,
        origin=resolved.origin,
        location=resolved.location,
        entity=entity,
    )
    print(f"  run name: {run_name}")
    export_run_metrics(run, metrics, output_dir)
    print(f"Done. Output in {output_dir}")


def export_group_metrics(
    project: str,
    group: str,
    metrics: list[str],
    results_dir: Path,
    *,
    entity: str | None = None,
    source: RunSource = "auto",
    local_root: Path | None = None,
) -> None:
    """Resolve a group's one run and reduce its metrics into CSV + SVG.

    One arm is one run, so this differs from `export_metrics` only in naming that run by its group
    instead of by its id; the export itself is `export_run_metrics` either way, so the same run
    gives the same bytes whichever name and whichever source found it. A group holding two
    finished runs is refused rather than pooled (`~tcirl.export_runs.exactly_one_run`).

    Acquisition is local-first: `source="auto"` reads the run directories under `local_root`
    (default `./wandb`) and falls back to the W&B API only when no local run matches or the
    matching one never finished, `"local"` never touches the network, `"wandb"` always does.

    No source will write a CSV for a group that has not finished -- the W&B fallback is gated on
    run state exactly as the local read is gated on the exit record, so `UnfinishedRunError`
    means the group is not ready, never merely that the disk copy was not.
    """
    resolved = resolve_group_run(
        project=project, group=group, entity=entity, source=source, local_root=local_root
    )
    run = resolved.run
    output_dir = results_dir / safe_filename(group)
    write_export_metadata(
        output_dir,
        [run],
        project=project,
        origin=resolved.origin,
        location=resolved.location,
        entity=entity,
        group=group,
    )
    print(f"Group {group!r}: run {run.label} from {resolved.origin} ({resolved.location})")
    export_run_metrics(run, metrics, output_dir)
    print(f"Done. Output in {output_dir}")
