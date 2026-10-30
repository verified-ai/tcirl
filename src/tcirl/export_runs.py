"""Where an export's per-run payloads come from: the local run directories, or the W&B API.

`export.py` owns the reduction -- pooling per-seed curves across a group, bootstrapping, writing
CSV and SVG. This module owns the one thing that differs between an offline experiment and an
uploaded one: *acquisition*. Both backends expose the same `ExportRun`, so everything downstream
of `resolve_group_run` is shared verbatim and a local export and a W&B export of the same run
produce the same bytes. Nothing below this seam knows which backend it is reading.

Local is the default. Training with `upload=False` still writes a full run directory under
`<root>/wandb/`, so the data an export needs is on disk whether or not it ever reached the W&B
servers. W&B is the *backup*: the place to go when the local copy is gone or unfinished.

Completeness is checked on *both* sides. A run directory exists from the moment training starts, so
a group whose runs are still training, were killed, or crashed sits on disk in the same shape as a
finished one, just short, and the servers' copy of such a run is short in the same way.

So completeness is *one* question asked through the `ExportRun` seam -- `unfinished_reason()` --
which each backend answers from what it can see: the local one from the exit record its
transaction log ends with (`~tcirl.logging.wandb_local.read_local_exit_code`, no network
needed), the W&B one from the run state the API reports. No source exports a run that never
finished. `local` and `wandb` fail naming it; `auto` treats an unfinished local copy as a reason
to try the backup, then holds the backup to the very same standard.

Determinism note: both backends hand back runs sorted by `(name, run_id)`, so a re-export of the
same group prints and resolves the same way every time, and the two backends agree numerically
because the reduction is order-invariant: the bootstrap CI of `~tcirl.metrics.stats.aggregate_stats`
is a function of the pooled multiset alone.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import wandb
from wandb.errors import CommError

from tcirl.logging.wandb import resolve_entity
from tcirl.logging.wandb_local import (
    LocalRunInfo,
    LocalRunReadError,
    local_run_dirs,
    read_local_exit_code,
    read_local_history,
    read_local_run,
)

# Where an export reads runs from. `auto` prefers local data and falls back to the network when
# no local run matches or a matching one never finished; `local` never touches the network;
# `wandb` never touches the disk.
RunSource = Literal["auto", "local", "wandb"]

RUN_SOURCES: tuple[RunSource, ...] = ("auto", "local", "wandb")

_CURVES_SUBDIR = "metric_curves"


@runtime_checkable
class ExportRun(Protocol):
    """One run's export payloads, independent of where the run is stored."""

    @property
    def run_id(self) -> str: ...

    @property
    def name(self) -> str: ...

    @property
    def label(self) -> str:
        """How to name this run to a human: whatever identifies it where it is stored.

        The run directory locally, the display name and id on W&B -- so a message about a run
        names the thing the reader would go and look at.
        """
        ...

    @property
    def config(self) -> dict[str, Any]:
        """The run's tracked config (the exporters read `cadence` from it for smoothing)."""
        ...

    def unfinished_reason(self) -> str | None:
        """Why this run's data is not final, or None when the run finished cleanly.

        The one question both backends must answer the same way, because a truncated export is
        equally wrong whichever side served it. Each answers it from the evidence it has -- an
        exit record on disk, a run state on the server -- and a backend that cannot tell says so
        with a reason rather than returning None: "could not check" is not "checked and fine".
        """
        ...

    def per_seed_curves(self) -> dict[str, Any] | None:
        """The run's `metric_curves/*.json` payloads merged into one metric -> payload map.

        None when the run persisted no such files at all, which is
        what makes the group exporter fall back to the scalar history.
        """
        ...

    def scalar_history(self, metric: str) -> list[tuple[float, float]]:
        """The run's non-null `(step, value)` pairs for one scalar metric key."""
        ...


def sort_runs[R: ExportRun](runs: Sequence[R]) -> list[R]:
    """Runs in a stable order, so repeated exports of one group agree on tie-breaks and output.

    Deterministic per backend, but not comparable across them (unnamed runs sort by id locally
    and by the server's generated name on W&B). The numbers must not depend on this order.
    """
    return sorted(runs, key=lambda run: (run.name, run.run_id))


# --------------------------------------------------------------------------------------
# W&B backend
# --------------------------------------------------------------------------------------


def fetch_run(entity: str, project: str, run_id: str) -> Any:
    api = wandb.Api()
    return api.run(f"{entity}/{project}/{run_id}")  # pyright: ignore[reportUnknownVariableType]


def list_run_files(run: Any) -> list[Any]:
    """List a run's files, working around the W&B client's pagination-boundary bug.

    `run.files()` paginates at 50 files per page. When a run's fileCount is an exact
    multiple of that page size, the backend reports `hasNextPage=True` on the final full
    page and answers the request past the last cursor with `files: null` instead of an
    empty edge list, so the client paginator dies with a TypeError on `["edges"]`. The
    fallback fetches everything in one oversized page (`per_page = fileCount + 1`; the
    paginator's `len` reads the server-side fileCount without paginating): a single
    non-full page never ends on the boundary, so pagination terminates correctly.
    """
    try:
        return list(run.files())
    except TypeError:
        total = len(run.files())
        try:
            return list(run.files(per_page=total + 1))
        except TypeError as err:
            raise RuntimeError(
                f"W&B file listing failed for run {run.id}: default pagination died on "
                f"the fileCount={total} page boundary, and the single-page fallback "
                f"(per_page={total + 1}) also got a null file index from the backend."
            ) from err


# The one `run.state` the W&B API reports for a run that reached a clean teardown. Everything
# else -- "running", "crashed", "failed", "killed", or a value this code has never heard of --
# means the server's copy is as short as the local one would have been.
WANDB_FINISHED_STATE = "finished"


@dataclass(frozen=True, slots=True)
class WandbExportRun:
    """An `ExportRun` backed by a run on the W&B servers."""

    run: Any

    @property
    def run_id(self) -> str:
        return str(self.run.id)

    @property
    def name(self) -> str:
        return str(self.run.name)

    @property
    def label(self) -> str:
        return f"{self.name} ({self.run_id})"

    @property
    def config(self) -> dict[str, Any]:
        config: dict[str, Any] = self.run.config if hasattr(self.run, "config") else {}
        return config

    def unfinished_reason(self) -> str | None:
        """The remote twin of the local exit-record check, read from the API's `run.state`.

        `state` is the server's own record of how the run ended, so it catches the case the
        local check cannot help with: the run crashed, its directory is gone or was never
        written, and the uploaded copy is truncated at exactly the same step. An absent state
        counts as unfinished -- a backup that cannot say whether it is complete is not one this
        export may quietly treat as final.
        """
        state = getattr(self.run, "state", None)
        if state is None:
            return "the W&B API reports no state for it, so completeness cannot be confirmed"
        text = str(state)
        if text == WANDB_FINISHED_STATE:
            return None
        if text == "running":
            return "W&B reports it still running"
        return f"W&B reports its state as {text!r}, not {WANDB_FINISHED_STATE!r}"

    def per_seed_curves(self) -> dict[str, Any] | None:
        with tempfile.TemporaryDirectory() as tmp:
            merged: dict[str, Any] = {}
            found = False
            for remote_file in list_run_files(self.run):
                name = str(remote_file.name)
                normalized = name.removeprefix("data/")
                if not normalized.startswith(f"{_CURVES_SUBDIR}/") or not normalized.endswith(
                    ".json"
                ):
                    continue
                downloaded = remote_file.download(root=tmp, replace=True)
                metrics = json.loads(Path(downloaded.name).read_text()).get("metrics")
                if not isinstance(metrics, dict):
                    continue
                merged.update(metrics)  # pyright: ignore[reportUnknownArgumentType]
                found = True
            return merged if found else None

    def scalar_history(self, metric: str) -> list[tuple[float, float]]:
        history: list[dict[str, Any]] = self.run.history(
            keys=[metric], samples=self.run.lastHistoryStep + 1, pandas=False
        )
        return [
            (float(row["_step"]), float(row[metric]))
            for row in history
            if row.get(metric) is not None
        ]


# --------------------------------------------------------------------------------------
# Local backend
# --------------------------------------------------------------------------------------


class LocalExportRun:
    """An `ExportRun` backed by a run directory under `wandb/`, online or offline.

    History is read lazily and cached: one pass over the transaction log yields every series at
    once, so the group exporter's per-metric `scalar_history` calls share a single read.
    """

    def __init__(self, info: LocalRunInfo) -> None:
        self.info = info
        self._history: dict[str, list[tuple[float, float]]] | None = None
        self._reason: str | None = None
        self._reason_read = False

    @property
    def run_id(self) -> str:
        return self.info.run_id

    @property
    def name(self) -> str:
        return self.info.display_name

    @property
    def label(self) -> str:
        # The directory, not the display name: an offline run often has no display name at all,
        # and the directory is the thing the reader can go and inspect, wait for, or delete.
        return self.info.directory.name

    @property
    def config(self) -> dict[str, Any]:
        return self.info.config

    @property
    def directory(self) -> Path:
        return self.info.directory

    def unfinished_reason(self) -> str | None:
        """Why this run's local data is not final, or None when the run exited cleanly.

        Reading the exit code means scanning the log to its end, so it is done lazily and
        cached: only the runs a group actually matched pay for it, not every directory the
        survey walked. A log too damaged to scan is reported here rather than raised, because
        an unreadable local copy is precisely the case the W&B backup exists for.
        """
        if not self._reason_read:
            self._reason = self._read_unfinished_reason()
            self._reason_read = True
        return self._reason

    def _read_unfinished_reason(self) -> str | None:
        try:
            exit_code = read_local_exit_code(self.info.directory)
        except LocalRunReadError as err:
            return f"its transaction log could not be read to the end ({err})"
        if exit_code is None:
            return "its transaction log records no exit (still training, or killed)"
        if exit_code != 0:
            return f"it exited with code {exit_code} (the run crashed)"
        return None

    def curve_files(self) -> list[Path]:
        """The run's persisted per-seed curve files, ordered by their run-relative name.

        Both layouts the uploader can produce are accepted (`metric_curves/` and
        `data/metric_curves/`), matching what the W&B backend normalizes remote names to.
        """
        files_dir = self.info.files_dir
        candidates = [
            *files_dir.glob(f"{_CURVES_SUBDIR}/*.json"),
            *files_dir.glob(f"data/{_CURVES_SUBDIR}/*.json"),
        ]
        return sorted(candidates, key=lambda p: str(p.relative_to(files_dir)))

    def per_seed_curves(self) -> dict[str, Any] | None:
        merged: dict[str, Any] = {}
        found = False
        for path in self.curve_files():
            metrics = json.loads(path.read_text()).get("metrics")
            if not isinstance(metrics, dict):
                continue
            merged.update(metrics)  # pyright: ignore[reportUnknownArgumentType]
            found = True
        return merged if found else None

    def scalar_history(self, metric: str) -> list[tuple[float, float]]:
        if self._history is None:
            self._history = read_local_history(self.info.directory)
        return list(self._history.get(metric, ()))


@dataclass(frozen=True, slots=True)
class LocalRunSurvey:
    """What a scan of one `wandb/` directory turned up -- the input to the not-found message."""

    root: Path
    runs: list[LocalExportRun]
    unreadable: list[tuple[Path, str]]

    def matching(self, project: str, group: str) -> list[LocalExportRun]:
        return sort_runs(
            [r for r in self.runs if r.info.project == project and r.info.group == group]
        )

    def describe(self) -> str:
        """A human-readable inventory of the directory: what is there, keyed by project/group."""
        if not self.root.exists():
            return f"  searched: {self.root} (does not exist)"
        lines = [f"  searched: {self.root}"]
        if not self.runs and not self.unreadable:
            lines.append("  found: no W&B run directories at all")
            return "\n".join(lines)
        lines.append(f"  found: {len(self.runs)} readable run directories")
        counts: dict[tuple[str, str | None], int] = {}
        for run in self.runs:
            key = (run.info.project, run.info.group)
            counts[key] = counts.get(key, 0) + 1
        for project, group in sorted(counts, key=lambda kv: (kv[0], str(kv[1]))):
            shown = "<no group>" if group is None else repr(group)
            lines.append(f"    project={project!r} group={shown}: {counts[project, group]} run(s)")
        for path, reason in self.unreadable:
            lines.append(f"    unreadable: {path.name} ({reason})")
        return "\n".join(lines)


def survey_local_runs(root: Path) -> LocalRunSurvey:
    """Read the identity of every run directory under `root` (never touches the network)."""
    runs: list[LocalExportRun] = []
    unreadable: list[tuple[Path, str]] = []
    for directory in local_run_dirs(root):
        try:
            runs.append(LocalExportRun(read_local_run(directory)))
        except LocalRunReadError as err:
            unreadable.append((directory, str(err)))
    return LocalRunSurvey(root=root, runs=runs, unreadable=unreadable)


def default_local_root() -> Path:
    """The `wandb/` directory W&B lays run directories out in, relative to the working directory."""
    return Path.cwd() / "wandb"


# --------------------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------------------


class NoRunsFoundError(RuntimeError):
    """No runs matched the requested project and group in the requested source."""


class UnfinishedRunError(RuntimeError):
    """A run matched the group but never finished, so its curves are truncated.

    Raised for either backend: the W&B copy of a run that crashed is truncated at the same step
    the local copy is, so the network is no escape from this and must not read like one.
    """


def unfinished_runs[R: ExportRun](runs: Sequence[R]) -> list[tuple[R, str]]:
    """The runs whose data is not final, each with the reason it is not."""
    return [(run, reason) for run in runs if (reason := run.unfinished_reason()) is not None]


def _describe_unfinished(unfinished: Sequence[tuple[ExportRun, str]]) -> str:
    return "\n".join(f"  {run.label}: {reason}" for run, reason in unfinished)


def require_finished(runs: Sequence[ExportRun], *, where: str, hint: str, note: str = "") -> None:
    """Raise `UnfinishedRunError` unless every run finished; the one gate before any export.

    `where` names what was being exported, `hint` says what to do instead, and `note` carries
    any extra context the caller already has (for `auto`, the local runs it skipped on the way
    here -- otherwise the message would blame the servers for a problem that started on disk).
    """
    unfinished = unfinished_runs(runs)
    if not unfinished:
        return
    raise UnfinishedRunError(
        f"Runs for {where} have not finished:\n"
        f"{_describe_unfinished(unfinished)}\n"
        "  Exporting them would report a truncated group as the final result.\n"
        f"  hint: {hint}{note}"
    )


def exactly_one_run[R: ExportRun](runs: Sequence[R], *, where: str) -> R:
    """The one run of an arm, refusing a set that holds several.

    An arm's seeds are vmapped inside a single process, so one arm is one run, and a second run of
    one group is a relaunch nobody cleaned up. Reducing both together would pool the seeds of two
    attempts -- of two different commits, if that is what the relaunch was for -- into one arm, and
    every number downstream would be a blend of runs that were never comparable.

    The count is taken over every matched run, finished or not, and it is asked BEFORE the
    completeness question. Two runs is not a state any source can resolve: a killed attempt beside
    a finished one leaves the export choosing which attempt the arm is, and treating the pair as
    "the local copy is unusable" would send it to the W&B backup, where the killed attempt does not
    exist and the finished one is served in silence.
    """
    if len(runs) == 1:
        return runs[0]
    labels = ", ".join(r.label for r in runs)
    raise ValueError(
        f"{where} has {len(runs)} runs ({labels}); expected one, because an arm's seeds live in "
        "one process. Train the arm into a fresh --project, which is the only fix that also holds "
        "for the W&B copy; moving the stale run directories aside fixes the local copy alone."
    )


def single_finished_run(survey: LocalRunSurvey, project: str, group: str) -> LocalExportRun:
    """The one finished run of `(project, group)` in `survey`; raises when there is none or several."""
    runs = survey.matching(project, group)
    if not runs:
        raise NoRunsFoundError(f"no local run for project={project!r} group={group!r}")
    run = exactly_one_run(runs, where=f"{project}/{group}")
    for _, reason in unfinished_runs([run]):
        raise UnfinishedRunError(f"{run.label} ({project}/{group}) is not final: {reason}")
    return run


def _skipped_local_note(unfinished: Sequence[tuple[ExportRun, str]]) -> str:
    """The `auto` fallback's trail, appended to whatever the W&B attempt then failed with."""
    if not unfinished:
        return ""
    return f"\n  local runs skipped as unfinished:\n{_describe_unfinished(unfinished)}"


@dataclass(frozen=True, slots=True)
class ResolvedRun:
    """A resolved run: the run plus where it was read from, for the export's banner."""

    run: ExportRun
    origin: Literal["local", "wandb"]
    location: str


def resolve_group_run(
    *,
    project: str,
    group: str,
    entity: str | None = None,
    source: RunSource = "auto",
    local_root: Path | None = None,
) -> ResolvedRun:
    """The group's one run, read from disk by default and from W&B only when asked or as a fallback.

    A group is returned only when its run finished cleanly, and that holds
    whichever side served them: a run still training, killed, or crashed is not exported short
    from disk, and it is not exported short from the servers either. `source="local"` never
    opens a network connection and raises naming the directory -- `NoRunsFoundError` when the
    group is not on disk, `UnfinishedRunError` when it is but is not final. `source="auto"` (the
    default) treats both of those as "the local copy is unusable" and falls back to the W&B
    backup, after saying on stdout which run it skipped and why -- then applies the same
    completeness gate to what comes back, because the backup of a run that crashed is truncated
    at the same step the local copy is. `source="wandb"` skips the disk entirely and is gated
    just the same.

    So no `--source` exports an unfinished group.

    The local scan matches on project and group only -- a run directory belongs to whoever
    checked out the repository, so its recorded entity is not used to filter.
    """
    if source not in RUN_SOURCES:
        raise ValueError(f"Unknown run source {source!r}; expected one of {list(RUN_SOURCES)}")

    root = default_local_root() if local_root is None else local_root
    survey: LocalRunSurvey | None = None
    unfinished: list[tuple[LocalExportRun, str]] = []
    if source in ("auto", "local"):
        survey = survey_local_runs(root)
        # A directory too damaged to identify cannot be refused, because it cannot be attributed
        # to a group at all -- it may belong to another project entirely. But it must not pass in
        # silence: if it *was* one of this group's seeds, the export is one seed short and looks
        # entirely successful, which is the failure this whole gate exists to prevent.
        for path, reason in survey.unreadable:
            print(f"  warning: ignoring unidentifiable run directory {path.name} ({reason})")
        matches = survey.matching(project, group)
        # Counted before the completeness question and before any fallback: see `exactly_one_run`.
        if len(matches) > 1:
            exactly_one_run(matches, where=f"project={project!r} group={group!r}")
        unfinished = unfinished_runs(matches)
        if matches and not unfinished:
            return ResolvedRun(run=matches[0], origin="local", location=str(root))
        if source == "local":
            # Past the early return, `local` has exactly two ways to be here: the group is on
            # disk but not final, or it is not on disk at all. Both end the call.
            require_finished(
                matches,
                where=f"project={project!r} group={group!r} under {root}",
                hint="wait for the runs to finish and re-export. The W&B copy of an unfinished "
                "run is truncated at the same step, so --source wandb will refuse it too.",
            )
            raise NoRunsFoundError(
                f"No local runs for project={project!r} group={group!r}.\n"
                f"{survey.describe()}\n"
                "  hint: pass --source wandb to fetch this group from the W&B servers instead."
            )
        for run, reason in unfinished:
            print(f"  skipping local run {run.label}: {reason}")
        if unfinished:
            print(f"  falling back to W&B for group {group!r} (local data is not final)")

    resolved_entity = resolve_entity(entity)
    api = wandb.Api()
    runs = [
        WandbExportRun(run)
        for run in api.runs(f"{resolved_entity}/{project}", filters={"group": group})  # pyright: ignore[reportUnknownVariableType]
    ]
    if not runs:
        detail = "" if survey is None else f"\n{survey.describe()}"
        raise NoRunsFoundError(
            f"No runs found for group {group!r} in {resolved_entity}/{project}."
            f"{detail}{_skipped_local_note(unfinished)}"
        )
    require_finished(
        runs,
        where=f"group {group!r} in {resolved_entity}/{project}",
        hint="wait for the runs to finish and re-export; a run that crashed has to be re-run.",
        note=_skipped_local_note(unfinished),
    )
    return ResolvedRun(
        run=exactly_one_run(
            sort_runs(runs), where=f"group {group!r} in {resolved_entity}/{project}"
        ),
        origin="wandb",
        location=f"{resolved_entity}/{project}",
    )


def write_export_metadata(
    output_dir: Path,
    runs: Sequence[ExportRun],
    *,
    project: str,
    origin: str,
    location: str,
    entity: str | None = None,
    group: str | None = None,
) -> None:
    """Write `export_metadata.yaml` beside the exported files: which runs made these numbers.

    The exported CSVs carry no run identity, and recovering "which W&B run produced this
    figure" months later from names and timestamps is guesswork. This sidecar records the
    run ids, the W&B links, and where the data was read from, at the moment of export.
    The entity comes from the argument or the WANDB_ENTITY env var; without either, the
    links are omitted rather than failing an otherwise-offline export.
    """
    try:
        url_entity: str | None = resolve_entity(entity)
    except ValueError:
        url_entity = None

    def quote(value: str) -> str:
        return json.dumps(value, ensure_ascii=False)

    lines = [
        f"project: {quote(project)}",
        f"entity: {quote(url_entity) if url_entity is not None else 'null'}",
        f"source: {quote(origin)}",
        f"location: {quote('local run directories' if origin == 'local' else location)}",
    ]
    if group is not None:
        lines.append(f"group: {quote(group)}")
    lines.append("runs:")
    for run in sort_runs(list(runs)):
        lines.append(f"  - run_id: {quote(run.run_id)}")
        lines.append(f"    name: {quote(run.name)}")
        if url_entity is not None:
            url = f"https://wandb.ai/{url_entity}/{project}/runs/{run.run_id}"
            lines.append(f"    url: {quote(url)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "export_metadata.yaml").write_text("\n".join(lines) + "\n")


def resolve_run(
    *,
    project: str,
    run_id: str,
    entity: str | None = None,
    source: RunSource = "auto",
    local_root: Path | None = None,
) -> ResolvedRun:
    """One run by id, read from disk by default and from W&B on request or as a fallback.

    The by-id twin of `resolve_group_run`, holding the same line: a run named by id is no
    more finished than one found by group, so the same completeness gate applies whichever side
    serves it, and there is no source that exports an unfinished run. The local scan matches on
    project and run id; the recorded entity is not used to filter.
    """
    if source not in RUN_SOURCES:
        raise ValueError(f"Unknown run source {source!r}; expected one of {list(RUN_SOURCES)}")

    root = default_local_root() if local_root is None else local_root
    survey: LocalRunSurvey | None = None
    unfinished: list[tuple[LocalExportRun, str]] = []
    if source in ("auto", "local"):
        survey = survey_local_runs(root)
        for path, reason in survey.unreadable:
            print(f"  warning: ignoring unidentifiable run directory {path.name} ({reason})")
        matches = sort_runs(
            [r for r in survey.runs if r.info.project == project and r.run_id == run_id]
        )
        # A run id is not unique on disk: `LoggingConfig.run_id` is a CLI flag, so two launches can
        # both be told to use one id and both write a directory carrying it.
        if len(matches) > 1:
            exactly_one_run(matches, where=f"run {run_id!r} in project {project!r}")
        unfinished = unfinished_runs(matches)
        if matches and not unfinished:
            return ResolvedRun(run=matches[0], origin="local", location=str(root))
        if source == "local":
            require_finished(
                matches,
                where=f"run {run_id!r} in project {project!r} under {root}",
                hint="wait for the run to finish and re-export. The W&B copy of an unfinished "
                "run is truncated at the same step, so --source wandb will refuse it too.",
            )
            raise NoRunsFoundError(
                f"No local run with id {run_id!r} for project {project!r}.\n"
                f"{survey.describe()}\n"
                "  hint: pass --source wandb to fetch this run from the W&B servers instead."
            )
        for run, reason in unfinished:
            print(f"  skipping local run {run.label}: {reason}")
        if unfinished:
            print(f"  falling back to W&B for run {run_id!r} (local data is not final)")

    resolved_entity = resolve_entity(entity)
    try:
        # Constructing the API run loads it eagerly, so a missing run raises right here.
        wandb_run = WandbExportRun(fetch_run(resolved_entity, project, run_id))
    except (CommError, ValueError) as err:
        detail = "" if survey is None else f"\n{survey.describe()}"
        raise NoRunsFoundError(
            f"No run {run_id!r} found in {resolved_entity}/{project} "
            f"({err}).{detail}{_skipped_local_note(unfinished)}"
        ) from err
    require_finished(
        [wandb_run],
        where=f"run {run_id} in {resolved_entity}/{project}",
        hint="wait for the run to finish and re-export; a run that crashed has to be re-run.",
        note=_skipped_local_note(unfinished),
    )
    return ResolvedRun(run=wandb_run, origin="wandb", location=f"{resolved_entity}/{project}")
