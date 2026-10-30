"""W&B logging for RL experiments."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib
import matplotlib.pyplot as plt
import wandb

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tcirl.metrics.spec import MetricRegistry
    from tcirl.rl.common.tabular import RendersMap
    from tcirl.visualization import AutomatonSummary

_logger = logging.getLogger(__name__)


def _get_git_info() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except subprocess.CalledProcessError:
        commit = "unknown"
    try:
        status = subprocess.run(
            ["git", "status"], capture_output=True, text=True, check=True
        ).stdout
    except subprocess.CalledProcessError:
        status = "unknown"
    return f"{commit}\n{status}"


def _get_git_diff() -> str:
    try:
        diff = subprocess.run(["git", "diff"], capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError:
        diff = "unknown"
    try:
        staged = subprocess.run(
            ["git", "diff", "--staged"], capture_output=True, text=True, check=True
        ).stdout
    except subprocess.CalledProcessError:
        staged = "unknown"
    parts: list[str] = []
    if diff:
        parts.append(diff)
    if staged:
        parts.append(staged)
    return "\n".join(parts)


def resolve_entity(entity: str | None) -> str:
    """Return the W&B entity, falling back to the WANDB_ENTITY env var."""

    resolved = entity or os.environ.get("WANDB_ENTITY")
    if resolved is None:
        raise ValueError("W&B entity must be set via --entity or the WANDB_ENTITY env var")
    return resolved


class WandbLogger:
    """W&B logger with accumulate-and-flush pattern.

    Carries the run's `~tcirl.metrics.spec.MetricRegistry` (`specs`):
    the single control point that drives per-seed reduction, W&B series naming,
    and the console line for every metric this logger emits.

    Every method is a no-op until `init_run` opens a run. An offline run is a run:
    it logs, records artifacts, and produces a run directory exactly like an online
    one, it just never talks to the W&B servers. So the guards below ask "is a run
    open", never "are we uploading".
    """

    def __init__(self, specs: MetricRegistry) -> None:
        self.specs = specs
        self._active: bool = False
        self._pending: dict[str, float] = {}
        self._last_flushed_step: int | None = None

    def init_run(
        self,
        *,
        project: str,
        entity: str | None,
        method: str,
        config: dict[str, Any],
        tags: list[str],
        group: str | None = None,
        job_type: str | None = None,
        name: str | None = None,
        notes: str | None = None,
        base_dir: Path | None = None,
        run_id: str | None = None,
        online: bool = True,
    ) -> None:
        """Initialize a W&B run, online or offline.

        `base_dir` is the parent W&B lays `wandb/run-<timestamp>-<id>/` out under
        (the cwd when omitted); the run's files directory is then `self.run_dir`.

        `online=False` opens an offline run instead: metrics, summaries and run
        files are all recorded locally under `wandb/offline-run-<timestamp>-<id>/`
        and nothing is sent to the W&B servers. The run stays syncable afterwards
        with `wandb sync <run-dir>`. An entity is only required when online.
        """
        run = wandb.init(
            entity=resolve_entity(entity) if online else entity,
            project=project,
            id=run_id,
            job_type=job_type or method,
            group=group,
            name=name,
            notes=notes,
            config={k: v for k, v in config.items() if k != "logging"},
            tags=tags or None,
            dir=base_dir,
            mode="online" if online else "offline",
            reinit="finish_previous",
            settings=wandb.Settings(quiet=True),
        )
        assert run is not None
        self._active = True
        print(f"wandb run id: {run.id}" + ("" if online else " (offline)"))

        assert run.dir is not None
        wandb_run_dir = Path(str(run.dir))
        git_file = wandb_run_dir / "git_info.txt"
        git_file.write_text(_get_git_info() + "\n\n\n---\n\n\n" + _get_git_diff())
        wandb.save(str(git_file), base_path=str(wandb_run_dir), policy="now")

    def log(self, metrics: dict[str, float]) -> None:
        """Accumulate metrics for the next flush."""
        if not self._active:
            return
        self._pending.update(metrics)

    def log_html(self, key: str, content: str, *, step: int) -> None:
        """Log a one-off `wandb.Html` panel (the `explanations/*` text boxes).

        Bypasses the registry like the media channels: these are documentation, not metrics.
        """
        if not self._active or wandb.run is None:
            return
        wandb.log({key: wandb.Html(content)}, step=step)

    def flush_step(self, step: int) -> None:
        """Send all accumulated metrics to W&B as a single log call, then clear."""
        if not self._active:
            return
        payload: dict[str, Any] = {}
        if self._pending:
            payload.update(self._pending)
            self._pending.clear()
        if payload:
            wandb.log(payload, step=step)
        self._last_flushed_step = step

    def log_summary(self, key: str, value: int | float | str) -> None:
        """Write a value to the W&B run summary."""
        if not self._active or wandb.run is None:
            return
        wandb.run.summary[key] = value

    @property
    def is_active(self) -> bool:
        """Whether a run is open, online or offline.

        The predicate for "should this logging side-effect happen at all". Callers
        should ask this rather than reading the `wandb.run` global, so that one
        object owns the answer.
        """
        return self._active and wandb.run is not None

    @property
    def project(self) -> str | None:
        """The project this run resolved to, or `None` when no run is open.

        Callers that need to find sibling runs on disk must ask here rather than re-deriving the
        name from their config: the default is applied at `wandb.init` time, so the config's
        `project` is `None` on exactly the runs that took the default.
        """
        if not self._active or wandb.run is None:
            return None
        return str(wandb.run.project)

    @property
    def run_dir(self) -> Path | None:
        """This run's artifact directory: the active W&B run's files directory.

        The path is unique per run, so artifacts can never overwrite an earlier
        run's. Writing a file here does NOT register it for upload -- hand it to
        `save_file` or `save_text` for that (a no-op on the wire for an offline
        run, but it still marks the file for a later `wandb sync`). `None` only
        when no run is open at all, which means "persist nothing".
        """
        if not self._active or wandb.run is None:
            return None
        return Path(str(wandb.run.dir))

    def save_file(self, path: Path, *, base_path: Path) -> None:
        """Register a run-directory file with the run, uploading it when online."""
        if not self._active or wandb.run is None:
            return
        wandb.save(str(path), base_path=str(base_path), policy="now")

    def save_text(self, text: str, *, filename: str, subdir: str | None = None) -> None:
        """Write a text file into the run directory and upload to W&B."""
        if not self._active or wandb.run is None:
            return
        assert wandb.run.dir is not None
        run_dir = Path(str(wandb.run.dir))
        target_dir = run_dir / subdir if subdir is not None else run_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / filename
        target_file.write_text(text)
        wandb.save(str(target_file), base_path=str(run_dir), policy="now")

    def log_initial_env_render(self, env: RendersMap) -> None:
        """Render the static env map (no live agent) and save as env/map.txt."""
        text = env.render_map()
        print(text)
        self.save_text(text, filename="map.txt", subdir="env")

    def log_automaton_summary(
        self,
        summary: AutomatonSummary,
        *,
        filename: str,
        subdir: str | None = None,
        max_states: int = 250,
        render_if: int | None = None,
    ) -> None:
        """Save an automaton summary text file (and optional SVG) to the W&B run.

        Generic over any AutomatonSummary protocol implementer (e.g. a DFA).
        """
        if not filename.endswith(".txt"):
            raise ValueError(f"filename must end with .txt, got {filename!r}")
        if not self._active or wandb.run is None:
            return

        self.save_text(summary.format_text(max_states=max_states), filename=filename, subdir=subdir)

        if render_if is not None and summary.num_states <= render_if:
            assert wandb.run.dir is not None
            run_dir_path = Path(str(wandb.run.dir))
            target_dir = run_dir_path / subdir if subdir is not None else run_dir_path
            digraph = summary.to_digraph()
            svg_stem = str((target_dir / filename).with_suffix(""))
            digraph.render(svg_stem, format="svg", cleanup=True)
            wandb.save(svg_stem + ".svg", base_path=str(run_dir_path), policy="now")

    def log_trajectory_ci(
        self,
        key: str,
        timesteps: Sequence[int] | Sequence[float],
        mean: Sequence[float],
        ci_lo: Sequence[float],
        ci_hi: Sequence[float],
        *,
        median: Sequence[float] | None = None,
        title: str = "",
        ylabel: str = "",
        ylim: tuple[float, float] | None = None,
        step: int,
    ) -> None:
        """Draw one evaluation trajectory as a W&B preview image (mean line, 95% CI band).

        An image only. The per-seed values behind it are persisted by
        `~tcirl.logging.recorder.PerSeedMetricRecorder` and reduced once, at export; a second copy
        written here would be a second definition of the same curve.
        """
        if not self._active or wandb.run is None:
            return

        matplotlib.use("Agg")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.fill_between(timesteps, ci_lo, ci_hi, alpha=0.25, color="tab:blue", label="95% CI")
        ax.plot(timesteps, mean, color="tab:blue", linewidth=1.5, label="Mean")
        if median is not None:
            ax.plot(
                timesteps, median, color="tab:orange", linewidth=1.0, linestyle="--", label="Median"
            )
        ax.set_xlabel("Episode step")
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.legend()
        fig.tight_layout()
        wandb.log({key: wandb.Image(fig)}, step=step)
        plt.close(fig)

    def log_trajectory_gap(
        self,
        key: str,
        timesteps: Sequence[int] | Sequence[float],
        series: Sequence[tuple[str, Sequence[float], Sequence[float], Sequence[float], str]],
        *,
        title: str = "",
        ylabel: str = "",
        ylim: tuple[float, float] | None = None,
        step: int,
    ) -> None:
        """Draw a multi-series trajectory preview image (each with mean line + CI band).

        Each element of `series` is (name, mean, ci_lo, ci_hi, color). An image only, for the same
        reason as `log_trajectory_ci`: the per-seed values are persisted by the recorder.
        """
        if not self._active or wandb.run is None:
            return

        matplotlib.use("Agg")
        fig, ax = plt.subplots(figsize=(8, 4))
        for name, mean, ci_lo, ci_hi, color in series:
            ax.fill_between(timesteps, ci_lo, ci_hi, alpha=0.18, color=color)
            ax.plot(timesteps, mean, color=color, linewidth=1.5, label=name)
        ax.set_xlabel("Episode step")
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.legend()
        fig.tight_layout()
        wandb.log({key: wandb.Image(fig)}, step=step)
        plt.close(fig)

    def finish_run(self, exit_code: int = 0) -> None:
        """Finish the W&B run, flushing any remaining metrics."""
        if not self._active:
            return
        try:
            if self._last_flushed_step is not None:
                self.flush_step(self._last_flushed_step)
            wandb.finish(exit_code=exit_code)
        except Exception:
            _logger.warning("Failed to finish W&B run", exc_info=True)
