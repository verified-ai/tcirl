"""Read back what `wandb.py` wrote: the run directories under `<root>/wandb/`.

An offline run (`upload=False`) records everything an online one does -- identity, tracked
config, history, run files -- into `wandb/offline-run-<timestamp>-<id>/`, it just never reaches
the W&B servers, so `wandb.Api()` cannot see it. This module reads those directories directly,
which is what makes an export work with no network at all.

Everything an export needs to trust a run -- identity (project / entity / group / display
name), the tracked config, the history, and whether the run actually finished -- lives **only**
in the run's `run-<id>.wandb` transaction log. The sidecar files are no substitute:
`files/config.yaml`, `files/wandb-summary.json` and `files/wandb-metadata.json` can be absent from a
cleanly finished *offline* run and present on a *crashed* online one, so their presence tracks
online-vs-offline, not completion, and none of them records the project or the group. The transaction log is a
leveldb-style record log of `wandb_internal_pb2.Record` protobufs whose first non-header record
is the `RunRecord` and whose last, for a run that reached teardown, is the `RunExitRecord`; and
`wandb.sdk.internal.datastore` is W&B's own reader for it. That reader is private API, but it is
the only reader there is (`wandb sync` goes through the same code), so it is used here behind
the narrow surface below: everything this module returns is plain Python.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore

# Run directories W&B lays out under `<root>/wandb/`. `latest-run` is a symlink to one of these
# and is deliberately not matched, so a run is never discovered twice.
_RUN_DIR_PREFIXES = ("run-", "offline-run-")


class LocalRunReadError(RuntimeError):
    """A local run directory exists but its transaction log could not be read."""


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalRunInfo:
    """One local W&B run directory: its identity, its tracked config, and where it lives.

    `group` is None for a run started without one (W&B stores that as an empty string).
    `config` mirrors the W&B API's `run.config`: the top-level tracked keys, `_wandb` excluded.
    """

    directory: Path
    run_id: str
    project: str
    entity: str
    group: str | None
    display_name: str
    job_type: str
    config: dict[str, Any]

    @property
    def files_dir(self) -> Path:
        """The run's files directory -- the local twin of the run's uploaded file set."""
        return self.directory / "files"


def transaction_log(directory: Path) -> Path | None:
    """The run directory's `*.wandb` transaction log, or None when it has none."""
    logs = sorted(directory.glob("*.wandb"))
    return logs[0] if logs else None


def is_run_dir(path: Path) -> bool:
    """Whether `path` is a W&B run directory (online or offline) with a transaction log."""
    return (
        path.is_dir()
        and path.name.startswith(_RUN_DIR_PREFIXES)
        and transaction_log(path) is not None
    )


def local_run_dirs(root: Path) -> list[Path]:
    """Every W&B run directory directly under `root`, in a stable (name-sorted) order."""
    if not root.is_dir():
        return []
    return sorted((p for p in root.iterdir() if is_run_dir(p)), key=lambda p: p.name)


@contextmanager
def _records(log_path: Path) -> Generator[Iterator[Any]]:
    """Stream the transaction log's `Record` protobufs, closing the file on the way out."""
    store = DataStore()
    store.open_for_scan(str(log_path))
    try:
        yield _scan(store, log_path)
    finally:
        store.close()


def _scan(store: DataStore, log_path: Path) -> Iterator[Any]:
    """Records up to the end of the log, stopping cleanly at a half-written trailing block.

    `wandb/` is scanned while other training jobs are running, so a log's final leveldb block is
    routinely mid-write, and a killed run leaves one behind permanently. The reader reports that
    with a bare `AssertionError` from inside W&B; `in_last_block()` distinguishes it from real
    corruption earlier in the file, exactly as W&B's own `wandb sync` does. A partial tail is
    end-of-data (warn, keep what was read); anything else is a `LocalRunReadError`.
    """
    while True:
        try:
            data = store.scan_data()
        except AssertionError as err:
            if store.in_last_block():
                warnings.warn(
                    f"{log_path} ends in a half-written record ({err}); reading the "
                    "records before it. Re-export once the run has finished.",
                    stacklevel=2,
                )
                return
            raise LocalRunReadError(f"{log_path} is corrupt: {err}") from err
        if data is None:
            return
        record: Any = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        yield record


def _flat_key(item: Any) -> str | None:
    """The item's key when it names a single top-level entry, else None.

    W&B stores a key either as `key` or as a `nested_key` path into a nested dict. Config and
    history entries this project writes are always top-level (history keys carry their own
    slashes, e.g. `eval_final/return`), so a genuinely nested path is not something either
    caller can consume and is skipped.
    """
    nested: list[str] = [str(part) for part in item.nested_key]
    if nested:
        return nested[0] if len(nested) == 1 else None
    key = str(item.key)
    return key or None


def _config_from(config_record: Any) -> dict[str, Any]:
    """The tracked config as the W&B API reports it: top-level keys, `_wandb` excluded."""
    config: dict[str, Any] = {}
    for item in config_record.update:
        key = _flat_key(item)
        if key is None or key == "_wandb":
            continue
        try:
            config[key] = json.loads(str(item.value_json))
        except json.JSONDecodeError:
            continue
    return config


def read_local_run(directory: Path) -> LocalRunInfo:
    """Read one run directory's identity and tracked config from its transaction log.

    Stops at the `RunRecord`, which is the log's first non-header record, so this stays cheap
    enough to call on every directory in `wandb/` just to filter by project and group.

    Every low-level failure becomes a `LocalRunReadError`. This is called on *every* directory
    in `wandb/`, including logs a concurrent training process is mid-write on, and the reader
    signals a malformed block with a bare `AssertionError` from deep inside W&B; one such
    directory must not take down the scan of all the others.
    """
    log_path = transaction_log(directory)
    if log_path is None:
        raise LocalRunReadError(f"{directory} has no *.wandb transaction log")
    try:
        with _records(log_path) as records:
            for record in records:
                if record.WhichOneof("record_type") != "run":
                    continue
                run = record.run
                return LocalRunInfo(
                    directory=directory,
                    run_id=str(run.run_id),
                    project=str(run.project),
                    entity=str(run.entity),
                    group=str(run.run_group) or None,
                    display_name=str(run.display_name),
                    job_type=str(run.job_type),
                    config=_config_from(run.config),
                )
    except LocalRunReadError:
        raise
    except Exception as err:
        raise LocalRunReadError(f"{log_path} is unreadable: {err}") from err
    raise LocalRunReadError(f"{log_path} contains no run record (the run never started)")


def read_local_exit_code(directory: Path) -> int | None:
    """The exit code the run recorded on teardown, or None when it never recorded one.

    This is the network-free twin of the W&B API's `run.state`, and the only trustworthy one:
    W&B writes a `RunExitRecord` as the **last** record of the transaction log when the run
    tears down -- explicitly via `run.finish()`, implicitly via the interpreter's exit hook, or
    on an uncaught exception (which records a non-zero code). A run that is still training, or
    that was killed before teardown, simply has no such record. So

        `read_local_exit_code(d) == 0`  <=>  the run finished and its data is complete.

    Verified against real run directories on wandb 0.25:
    an offline run SIGKILLed mid-training left 96 flushed history records, a readable identity
    and *no* exit record, while both a `run.finish()` run and one relying on the exit hook wrote
    `exit_code=0`, and an uncaught exception wrote `exit_code=1`. The file-presence signals one
    might reach for instead do not work: `files/config.yaml`, `files/wandb-summary.json` and
    `files/wandb-metadata.json` are written for *online* runs regardless of how they ended and
    for offline runs not at all, and `run-<id>.wandb.synced` only marks a `wandb sync` upload.

    A log whose tail is half-written yields None (`_scan` warns and stops there), which is the
    right answer: such a run is mid-flight or was killed. Real corruption earlier in the log
    still raises `LocalRunReadError`.
    """
    log_path = transaction_log(directory)
    if log_path is None:
        raise LocalRunReadError(f"{directory} has no *.wandb transaction log")
    exit_code: int | None = None
    with _records(log_path) as records:
        for record in records:
            if record.WhichOneof("record_type") == "exit":
                exit_code = int(record.exit.exit_code)
    return exit_code


def read_local_history(directory: Path) -> dict[str, list[tuple[float, float]]]:
    """Every scalar history series in the run's transaction log, as `(step, value)` pairs.

    The local twin of the W&B API's `run.history(keys=[...])`, used for the group exporter's
    scalar fallback (runs that persist no per-seed `metric_curves` files). Non-numeric entries
    (images, tables, ...) and W&B's own `_`-prefixed bookkeeping keys are dropped.

    Unlike `read_local_run`, which stops at the first record, this reads to the end of the log
    and so meets half-written tails: `_scan` warns and returns the history up to that point,
    which is the whole history for every run that is not still training.
    """
    log_path = transaction_log(directory)
    if log_path is None:
        raise LocalRunReadError(f"{directory} has no *.wandb transaction log")
    series: dict[str, list[tuple[float, float]]] = {}
    with _records(log_path) as records:
        for record in records:
            if record.WhichOneof("record_type") != "history":
                continue
            step = float(record.history.step.num)
            for item in record.history.item:
                key = _flat_key(item)
                if key is None or key.startswith("_"):
                    continue
                try:
                    value = json.loads(str(item.value_json))
                except json.JSONDecodeError:
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                series.setdefault(key, []).append((step, float(value)))
    return series
