"""Self-documentation panels for a W&B run (`explanations/*`).

Two `wandb.Html` panels logged once at step 0 make the workspace self-explanatory, so a reader
never has to leave the W&B UI to decode a run or a metric name:

- `explanations/run` -- what this run IS: the operator-written note from `LoggingConfig.notes`
  (the same text also lands in the run's W&B notes field), which states what the arm varies and
  what to compare it against.
- `explanations/metrics` -- the metric-name legend, generated from the run's `MetricRegistry`:
  the name anatomy, the four time reductions, the cross-seed statistics, the top-level prefixes,
  and one line per visible series.

These keys are registered in no `MetricRegistry`: they are documentation, not metrics, and bypass
the registry exactly like the media channel of trajectory images.
"""

from __future__ import annotations

import html

from tcirl.logging.wandb import WandbLogger
from tcirl.metrics.spec import (
    BatchMetric,
    EvalMetric,
    MetricRegistry,
    ScalarMetric,
    WindowedMetric,
)

# How a seed's raw per-step signals become one value per log point (the second path segment of a
# windowed series name). Keys match `tcirl.metrics.spec._CATEGORY_CODE` values;
TIME_REDUCTIONS: dict[str, str] = {
    "step": "mean of the per-step signal over each log tile (log_interval env steps)",
    "episode": "mean over the episodes that COMPLETED in the tile; a gap means no episode ended",
    "max": "running maximum since step 0 (a monotone 'ever reached' extreme, not a typical value)",
    "count": "running sum of the per-step signal since step 0 (monotone)",
}

# How the per-seed values reduce to the plotted line (the last path segment of a series name).
CROSS_SEED_STATS: dict[str, str] = {
    "mean": "mean across seeds",
    "median": "median across seeds (the typical seed)",
    "min": "minimum across seeds",
    "max": "maximum across seeds",
}

# What each top-level metric prefix holds. Static domain knowledge, kept next to the generator so
# the legend and the registries change together.
PREFIXES: dict[str, str] = {
    "train": "training curves reduced from per-step signals",
    "inference": "cause-DFA inference diagnostics (TCIRL only), logged once per stop_interval",
    "scalars": "run-global values with no per-seed array (schedules, throughput)",
    "eval_final": "greedy eval at one point of the run's evaluation grid, one value per seed",
    "trajectories": "per-timestep evaluation curves over one episode (images + JSON, not series)",
    "explanations": "these text panels",
}

_STYLE = (
    "font-family: sans-serif; font-size: 13px; line-height: 1.45; max-width: 60em; color: inherit;"
)


def _dl(items: dict[str, str]) -> str:
    rows = "".join(
        f"<tr><td style='padding:1px 10px 1px 0; white-space:nowrap; vertical-align:top'>"
        f"<code>{html.escape(k)}</code></td>"
        f"<td style='padding:1px 0'>{html.escape(v)}</td></tr>"
        for k, v in items.items()
    )
    return f"<table style='border-collapse:collapse'>{rows}</table>"


def _series_line(spec: WindowedMetric | BatchMetric | EvalMetric, registry: MetricRegistry) -> str:
    series = registry.wandb_series(spec.key)
    names = ", ".join(name for name, _ in series)
    if isinstance(spec, WindowedMetric):
        code = registry.wandb_series(spec.key)[0][0].split("/")[1]
        cadence = TIME_REDUCTIONS[code]
    elif isinstance(spec, BatchMetric):
        cadence = "one value per seed per stop_interval, logged at the end of the batch that completes it"
    else:
        cadence = "one value per seed at each evaluation point"
    stats = "; ".join(CROSS_SEED_STATS[stat] for _, stat in series)
    return f"<code>{html.escape(names)}</code> — {html.escape(cadence)}. {html.escape(stats)}."


def metric_legend_html(registry: MetricRegistry) -> str:
    """The `explanations/metrics` panel: name anatomy, reductions, stats, prefixes, all series."""
    per_series: list[str] = []
    for m in registry.metrics:
        if isinstance(m, ScalarMetric):
            continue
        if m.display.hidden:
            continue
        per_series.append(f"<li style='margin:2px 0'>{_series_line(m, registry)}</li>")
    prefixes = {
        k: v
        for k, v in PREFIXES.items()
        if any(m.key.startswith(k + "/") for m in registry.metrics)
    }
    # Always-documented prefixes that carry no registry entry of their own.
    for extra in ("explanations",):
        prefixes[extra] = PREFIXES[extra]
    return (
        f"<div style='{_STYLE}'>"
        "<h3>How to read a metric name</h3>"
        "<p>Per-seed curves are named "
        "<code>&lt;prefix&gt;/&lt;time reduction&gt;/&lt;metric&gt;/&lt;cross-seed statistic&gt;</code>. "
        "The time reduction says how one seed's per-step signals become one value per log point; "
        "the cross-seed statistic says how the per-seed values become the plotted line. "
        "Batch/eval metrics skip the time-reduction segment; scalars are logged under their "
        "verbatim key.</p>"
        "<h4>Time reductions</h4>"
        + _dl(TIME_REDUCTIONS)
        + "<h4>Cross-seed statistics</h4>"
        + _dl(CROSS_SEED_STATS)
        + "<h4>Prefixes</h4>"
        + _dl(prefixes)
        + "<h4>Every visible series in this run</h4>"
        f"<ul style='padding-left:1.2em'>{''.join(per_series)}</ul>"
        "</div>"
    )


def run_note_html(note: str) -> str:
    """The `explanations/run` panel: the operator-written note, verbatim."""
    return f"<div style='{_STYLE}'><h3>What this run is</h3><p>{html.escape(note)}</p></div>"


def log_run_explanations(logger: WandbLogger, *, note: str | None) -> None:
    """Log the `explanations/*` panels once, right after the run opens (step 0)."""
    if note:
        logger.log_html("explanations/run", run_note_html(note), step=0)
    logger.log_html("explanations/metrics", metric_legend_html(logger.specs), step=0)
