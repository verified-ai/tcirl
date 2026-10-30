"""Shared helpers for live metric summaries with export-time reconstruction."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import jax
import numpy as np

from tcirl.logging.naming import safe_filename
from tcirl.logging.wandb import WandbLogger
from tcirl.metrics.spec import EvalMetric, MetricKindError
from tcirl.metrics.stats import summarize


def init_metric_recorder(
    logger: WandbLogger,
    file_stem: str,
    *,
    x_label: str = "step",
) -> PerSeedMetricRecorder:
    """A recorder rooted at the run's artifact directory (no persistence when no run is open).

    `x_label` names the axis the recorded points are indexed by, and travels in the JSON so the
    export labels the CSV's first column with it. Training curves are indexed by the env step
    (`"step"`); the evaluation trajectories by the timestep inside one episode (`"timestep"`).
    """
    return PerSeedMetricRecorder(logger.run_dir, logger, file_stem, x_label=x_label)


class PerSeedMetricRecorder:
    """Persist per-seed metric curves as uploaded W&B run files."""

    def __init__(
        self,
        run_dir: Path | None,
        logger: WandbLogger,
        file_stem: str,
        *,
        subdir: str = "metric_curves",
        x_label: str = "step",
    ):
        self._run_dir = run_dir
        self._logger = logger
        self._x_label = x_label
        self._path = (
            None if run_dir is None else run_dir / subdir / f"{safe_filename(file_stem)}.json"
        )
        self._steps_by_metric: dict[str, list[int]] = {}
        self._values_by_metric: dict[str, list[list[float]]] = {}
        # Which metrics are smoothable (mean categories); declared by BatchLogger and written into
        # the JSON so the offline export knows where applying the EMA is meaningful.
        self._smoothable_keys: frozenset[str] = frozenset()

    def mark_smoothable(self, keys: frozenset[str]) -> None:
        self._smoothable_keys = keys

    def record(self, step: int, per_seed: Mapping[str, jax.Array]) -> None:
        if self._path is None:
            return
        for path, values in per_seed.items():
            arr = np.asarray(jax.device_get(values), dtype=np.float32)
            if arr.ndim != 1:
                raise ValueError(
                    f"Expected per-seed metric {path!r} to be 1D, got shape {arr.shape}"
                )
            if path not in self._steps_by_metric:
                self._steps_by_metric[path] = []
                self._values_by_metric[path] = []
            steps = self._steps_by_metric[path]
            all_values = self._values_by_metric[path]
            steps.append(step)
            all_values.append(arr.astype(np.float64).tolist())

    def save(self) -> None:
        if self._path is None or self._run_dir is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metrics": {
                path: {
                    "steps": self._steps_by_metric[path],
                    "values": self._values_by_metric[path],
                    "smoothable": path in self._smoothable_keys,
                    "x_label": self._x_label,
                }
                for path in self._steps_by_metric
            },
        }
        self._path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        self._logger.save_file(self._path, base_path=self._run_dir)


def log_live_metrics(
    logger: WandbLogger,
    recorder: PerSeedMetricRecorder,
    per_seed: Mapping[str, jax.Array],
    scalars: Mapping[str, float],
    *,
    step: int,
    per_seed_smoothed: Mapping[str, np.ndarray] | None = None,
) -> dict[str, float]:
    """Persist the RAW per-seed arrays and emit the cross-seed W&B series.

    Naming and which series are emitted come from `logger.specs` (the run's
    `~tcirl.metrics.spec.MetricRegistry`): each per-seed `base` emits its `wandb_series` (the
    windowed category code and stat decorate the name, always suffixed; hidden metrics emit nothing).
    The **raw** per-seed arrays are persisted (the export's source of truth, which re-derives smoothing
    from the run config). The cross-seed statistic is computed from `per_seed_smoothed` (the per-seed
    EMA; defaults to the raw arrays when smoothing is off), NaN seeds masked out and a metric omitted
    at a point where every seed is NaN. Scalars are validated as `ScalarMetric` and logged verbatim;
    any unregistered key raises `~tcirl.metrics.spec.UnknownMetricError`.
    """
    specs = logger.specs
    recorder.record(step, per_seed)
    reduced_from = per_seed if per_seed_smoothed is None else per_seed_smoothed
    metrics: dict[str, float] = {}
    for base, values in reduced_from.items():
        arr = np.asarray(jax.device_get(values))
        finite = arr[~np.isnan(arr)]
        if finite.size == 0:
            continue  # no seed has data at this point -> omit the metric here
        for key, stat in specs.wandb_series(base):
            metrics[key] = summarize(finite, stat)
    metrics.update({base: value for base, value in scalars.items() if specs.emit_scalar(base)})
    logger.log(metrics)
    return metrics


def log_eval_metrics(
    logger: WandbLogger,
    recorder: PerSeedMetricRecorder,
    per_seed: Mapping[str, jax.Array],
    *,
    step: int,
    per_seed_smoothed: Mapping[str, np.ndarray] | None = None,
) -> dict[str, float]:
    """Emit the end-of-run eval pass's per-seed `EvalMetric`s (one point per seed at `step`).

    The eval channel for `EvalMetric`s, mirroring `BatchLogger.process_batch`'s batch channel for
    `BatchMetric`s: it validates that every key is a registered `EvalMetric` (so a windowed or batch
    metric cannot be mis-routed here) and delegates to `log_live_metrics`.
    """
    for base in per_seed:
        spec = logger.specs.spec_for(base)
        if not isinstance(spec, EvalMetric):
            raise MetricKindError(
                f"{base!r} is a {type(spec).__name__}; the eval channel logs only EvalMetrics."
            )
    return log_live_metrics(
        logger, recorder, per_seed, {}, step=step, per_seed_smoothed=per_seed_smoothed
    )
