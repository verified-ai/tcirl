"""Project one export snapshot onto the file names and columns the paper's LaTeX reads.

This module selects columns, renames them, and assembles one table across arms. It computes exactly
one thing, the Wilson interval named under THE INTERVAL below; every other number it writes is
copied from a CSV that `tcirl.export` produced. That is the point: the paper and the repository
must not be able to disagree about a number, so there is one reduction, the exporter's, and this is
a view of it.

What the paper reads, and where each file comes from:

| paper file                            | source                                                    |
|---------------------------------------|-----------------------------------------------------------|
| `<arm>/eval_<stat>_<metric>.csv`      | `eval_final_<metric>.csv`: <stat>, <stat>_ci_*             |
| `<arm>/inference_<stat>_<metric>.csv` | `inference_<metric>.csv`: <stat>, <stat>_ci_*              |
| `<arm>/trajectories_<metric>.csv`     | `trajectories_<metric>.csv`: <stat>, <stat>_ci_*           |
| `accept_state.csv`                    | the `eval_final_frac_episodes_in_accept` row of each arm's `final_intervals.csv` |
| `final_returns.csv` (traffic)         | the `eval_final_cumulative_reward` row of each arm's `final_intervals.csv` |
| `seed_scores.csv`                     | the `eval_final_cumulative_reward` row of each arm's `per_seed_final.csv` |
| `transfer_returns.csv`, `counterfactual_returns.csv` | the `eval_final_cumulative_reward` row of the `final_intervals.csv` of each arm of `ABLATION_TABLES` |

`PAPER_CURVES` names the curves of each domain that the paper's LaTeX reads, and the projection writes those and no other file, so the paper's tree holds exactly what its text reads.

`<stat>` is the metric's entry in `PAPER_STATISTIC`, so a file's name says which statistic its
centre column holds. `seed_scores.csv` is the one file that carries no statistic at all: it is the
50 per-seed numbers themselves.

THE STATISTIC. Every number the paper prints is one number per seed reduced across the 50 seeds,
and that reduction is the plain MEAN for every metric without exception. The mean
estimates the expected return over the seed distribution: the quantity reinforcement learning
optimises, and the one a reader gets by training once and deploying. `PAPER_STATISTIC`
records the choice per metric, so a new metric is classified before it is printed rather than
defaulting to a statistic nobody chose.

THE INTERVAL. Every interval is a 95% PERCENTILE BOOTSTRAP over the 50 seeds: resamples with
replacement, the 2.5th and 97.5th percentile of the resampled statistic. Every point of a curve
uses 2000 resamples, and every interval the paper prints as a number uses
`tcirl.path_consts.PRINTED_DRAWS` = 50000; the tables read the exporter's `final_intervals.csv`.

`PAPER_INTERVAL` declares one exception: `eval_final/mostly_in_accept` gives each seed a 0 or a 1,
and its interval is the Wilson interval for a proportion (`metrics.intervals.wilson_interval`).

The canonical CSV keeps the degenerate bootstrap columns for that metric; nothing reads them.

THREE KINDS OF QUANTITY. All three are reduced by the mean; they differ in what the number MEANS.

- A SCORE is a per-step reward or a per-episode return: the quantity the agent optimises.
- A RECORDED STATE VARIABLE is the symptom level or the queue length at a timestep of an evaluation
  episode, or its average over one. It is a behaviour profile and not a score.
- A SHARE or a COUNT pools what it is a share or a count OF, and the mean is what makes that
  pooling correct: every seed evaluates the same 50 episodes, so the mean of the 50 per-seed
  fractions is the fraction of all 2500 evaluation episodes, and a mean count stops rising only
  once every seed has stopped.

THE DISTRIBUTION behind the mean appears in three outputs. `eval_final/mostly_in_accept` is the
share of SEEDS that end most of their evaluation episodes in Phase 2; `seed_scores.csv` is every
seed's final return, which the paper's figure of per-seed returns plots as 50 dots per arm; and the
figure of evaluation trajectories that reach Phase 2 reports the share of EPISODES.

SMOOTHING: `tcirl.export.plotted_payloads` applies
an exponential moving average per seed, over `cadence.smoothing_steps` env steps (10000 in both
domains), BEFORE the cross-seed statistic is taken, and the weight comes from each curve's own x
spacing so the horizon is the same 10000 steps whatever the curve's resolution. It reaches the
`train_*` curves and the `eval_*` scores and state variables, which includes both reward panels and
the per-seed sample behind `seed_scores.csv`. It does NOT reach the `trajectories_*` profiles, the
`eval_*` shares and counts, or the inference counts, so the Phase 2 figure and the re-inference curve report a
measurement and never a time average. The one exception is the seed indicator
`eval_final/mostly_in_accept`, which `tcirl.export.plotted_payloads` takes on each seed's SMOOTHED
episode share, so that one checkpoint above one half does not count a seed; it stays a 0 or a 1.

The panels read the evaluation curve and not the training curve, because a training curve measures
the exploring behaviour policy.
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal

from tcirl.metrics.intervals import wilson_interval
from tcirl.path_consts import FINAL_INTERVALS_FILE, PER_SEED_SAMPLE_FILE, PRINTED_DRAWS

# The paper's five arms, in the order its legend lists them, with the column each occupies in
# `accept_state.csv`. The keys are the W&B group names, which are also the export's directory
# names and the paper's `results/<domain>/<arm>/` directories.
ARM_TABLE_COLUMNS: dict[str, str] = {
    "q_learning-cause_known": "oracle",
    "q_learning-cause_unknown": "unknown",
    "tcirl-sat": "tcirl",
    "dqn-cause_unknown_4": "dqn4",
    "dqn-cause_unknown_1": "dqn1",
}

# The four ablation arms of the paper's appendix: TCIRL with the value transfer rules
# "none" and "exact", TCIRL with counterfactual updates, and the true cause with counterfactual
# updates, whose reference is `q_learning-cause_known`.
ABLATION_ARMS: tuple[str, ...] = (
    "tcirl-sat-no_transfer",
    "tcirl-sat-exact_transfer",
    "tcirl-sat-counterfactual",
    "tcirl-oracle-counterfactual",
)

# Every arm the projection writes, the paper's five and then the ablations.
PROJECTED_ARMS: tuple[str, ...] = (*ARM_TABLE_COLUMNS, *ABLATION_ARMS)

# The two tables of final returns of the appendix, by file name, each with its arms and the column each
# occupies: the ablation of value transfer and that of the counterfactual updates.
ABLATION_TABLES: dict[str, dict[str, str]] = {
    "transfer_returns.csv": {
        "tcirl-sat": "tcirl",
        "tcirl-sat-exact_transfer": "exact_transfer",
        "tcirl-sat-no_transfer": "no_transfer",
        "q_learning-cause_known": "oracle",
    },
    "counterfactual_returns.csv": {
        "tcirl-sat": "tcirl",
        "tcirl-sat-counterfactual": "counterfactual",
        "q_learning-cause_known": "oracle",
        "tcirl-oracle-counterfactual": "oracle_counterfactual",
    },
}

Statistic = Literal["mean"]

# The cross-seed statistic each canonical CSV stem is read under, by the paper for the curves in `PAPER_CURVES` and by other analyses for the rest.
# See the module docstring for the three rules; `paper_statistic` refuses a stem that is not here, so a new metric is classified before anything reads it rather than defaulting to a statistic nobody chose.
PAPER_STATISTIC: dict[str, Statistic] = {
    # Scores, measured while training.
    "train_reward": "mean",
    "train_cumulative_reward": "mean",
    "train_discounted_return": "mean",
    "train_episode_length": "mean",
    # The same scores, measured on the greedy policy at each point of the evaluation grid.
    # `eval_cumulative_reward` is the undiscounted per-episode return the paper's two reward panels
    # plot and `eval_return` the discounted one, whose ceiling is 63.4 rather than 100.
    "eval_final_return": "mean",
    "eval_final_cumulative_reward": "mean",
    "eval_final_episode_length": "mean",
    "eval_final_mean_reward": "mean",
    # Optimizer diagnostics, which no panel plots.
    "train_td_error": "mean",
    "train_qmax": "mean",
    "train_loss": "mean",
    "train_grad_norm": "mean",
    # The tabular step size, the mean over the window's updates, and each TCIRL seed's exploration rate.
    "train_step_size": "mean",
    "inference_epsilon": "mean",
    # The Q-table entries one TCIRL transition updates, and the share of the old table's update counts in the rows whose values its last re-inference transferred.
    "train_updated_states": "mean",
    "inference_kept_updates": "mean",
    # Recorded state variables, per timestep of an evaluation episode and averaged over one.
    "trajectories_reward": "mean",
    "trajectories_cumulative_reward": "mean",
    "trajectories_disease": "mean",
    "trajectories_harm": "mean",
    "trajectories_q0": "mean",
    "trajectories_q1": "mean",
    "trajectories_q2": "mean",
    "trajectories_total_queue": "mean",
    "eval_final_mean_disease": "mean",
    "eval_final_mean_harm": "mean",
    "eval_final_mean_total_queue": "mean",
    # Shares. A share of EPISODES pools the 2500; a share of SEEDS pools the 50.
    "trajectories_in_accept": "mean",
    "trajectories_x_a": "mean",
    "trajectories_x_r": "mean",
    "eval_final_frac_episodes_in_accept": "mean",
    "eval_final_mostly_in_accept": "mean",
    "eval_final_frac_steps_in_accept": "mean",
    "eval_final_frac_x_a": "mean",
    "eval_final_frac_x_r": "mean",
    # Counts, whose plateau is every seed's plateau.
    "train_num_episodes": "mean",
    "trajectories_distinct_cause_states": "mean",
    "eval_final_mean_distinct_cause_states": "mean",
    "inference_hypothesis_states": "mean",
    "inference_traces": "mean",
    "inference_reinferences": "mean",
}

Interval = Literal["bootstrap", "wilson"]

# How the paper's interval around each centre is CONSTRUCTED: the 95% percentile bootstrap the
# exporter already computed, except for a metric whose per-seed value is 0 or 1, which takes the
# Wilson score interval for a proportion. `eval_final/mostly_in_accept` is the only such metric;
# every other share here is a per-seed fraction of 50 evaluation episodes.
PAPER_INTERVAL: dict[str, Interval] = {"eval_final_mostly_in_accept": "wilson"}

# The score whose 50 per-seed values the paper plots as a sample rather than as a centre: the same
# undiscounted per-episode return of the greedy policy that both reward panels end at, so a reader
# can put the panel's endpoint and the 50 dots behind it side by side.
SEED_SCORE_METRIC = "eval_final_cumulative_reward"

# The eval metric that the paper's figure of evaluation trajectories that reach Phase 2 reports: the share of evaluation EPISODES that reach Phase 2, as a
# percentage. An episode that fires the cause halfway through counts as one trajectory that reached
# Phase 2.
ACCEPT_METRIC = "eval_final_frac_episodes_in_accept"

# The arms in the order the paper's figures read them.
_EVERY_ARM: tuple[str, ...] = tuple(ARM_TABLE_COLUMNS)

# The canonical curves the paper's LaTeX reads in each domain, each with the arms whose figure reads it.
# The two reward panels read the evaluation return, the seed-share figure the share of seeds, the profile panels the domain's state variable, and the re-inference figure TCIRL's count alone.
PAPER_CURVES: dict[str, dict[str, tuple[str, ...]]] = {
    "gene_editing": {
        "eval_final_cumulative_reward": PROJECTED_ARMS,
        "eval_final_mostly_in_accept": _EVERY_ARM,
        "trajectories_disease": _EVERY_ARM,
        "inference_reinferences": ("tcirl-sat",),
    },
    "traffic_signal": {
        "eval_final_cumulative_reward": PROJECTED_ARMS,
        "eval_final_mostly_in_accept": _EVERY_ARM,
        "trajectories_total_queue": _EVERY_ARM,
        "inference_reinferences": ("tcirl-sat",),
    },
}

# Which canonical prefix becomes which paper prefix.
# The `eval_final_` and `inference_` curves carry the statistic in their name (`eval_mean_cumulative_reward`, `inference_mean_reinferences`), because the `.tex` reads the centre column under that name; the trajectories keep their name and carry the statistic in the column header alone.
# `{stat}` is filled from `PAPER_STATISTIC`.
_PREFIX_RENAMES: tuple[tuple[str, str], ...] = (
    ("eval_final_", "eval_{stat}_"),
    ("inference_", "inference_{stat}_"),
    ("trajectories_", "trajectories_"),
)


class MissingExportError(FileNotFoundError):
    """A file the paper reads has no counterpart in the snapshot."""


@dataclass(frozen=True, slots=True)
class ProjectedFile:
    """One written paper file and the canonical CSV it was copied from."""

    source: Path
    target: Path
    rows: int
    statistic: Statistic


def paper_statistic(stem: str) -> Statistic:
    """The statistic the paper prints for a canonical CSV stem; raises on a stem nobody classified."""
    try:
        return PAPER_STATISTIC[stem]
    except KeyError:
        raise ValueError(
            f"no paper statistic is declared for {stem!r}; add it to "
            f"tcirl.paper_results.PAPER_STATISTIC with the rule it falls under"
        ) from None


def paper_interval(stem: str) -> Interval:
    """How the paper's interval for a canonical CSV stem is built; the bootstrap unless declared."""
    return PAPER_INTERVAL.get(stem, "bootstrap")


_PROJECTED_PREFIXES: tuple[str, ...] = tuple(prefix for prefix, _ in _PREFIX_RENAMES)


def paper_file_name(stem: str, statistic: Statistic) -> str:
    """The paper's file name for a canonical CSV stem under one of the projected prefixes.

    Public because a check of the paper's tree has to derive the same name, and deriving
    it a second way is how a renamed prefix comes to be checked against a file nobody writes.
    """
    for prefix, replacement in _PREFIX_RENAMES:
        if stem.startswith(prefix):
            return replacement.format(stat=statistic) + stem.removeprefix(prefix)
    raise ValueError(f"{stem!r} carries none of the projected prefixes {_PROJECTED_PREFIXES}")


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise MissingExportError(f"{path} is empty")
    return rows[0], rows[1:]


# A centre printed to six significant digits times a seed count of 50 lands within this of the
# whole number of successes it came from. It is a guard and not the decision: `PAPER_INTERVAL` says
# which metric is a count of successes, and this only refuses to believe it when the file disagrees.
_COUNT_TOLERANCE = 1e-3


def _wilson_row(value: float, seeds: int, source: Path, x: str) -> tuple[float, float]:
    """The Wilson bounds for one row of a declared indicator metric, on the metric's own 0..1 scale."""
    successes = round(value * seeds)
    if abs(value * seeds - successes) > _COUNT_TOLERANCE:
        raise MissingExportError(
            f"{source} is declared an indicator metric in PAPER_INTERVAL, but at {x} its centre "
            f"{value} over {seeds} seeds is {value * seeds}, which is not a whole number of "
            f"successes; a per-seed value that is not 0 or 1 takes the bootstrap instead"
        )
    return wilson_interval(successes, seeds)


def project_curve(
    source: Path, target: Path, statistic: Statistic, interval: Interval
) -> ProjectedFile:
    """Copy one canonical curve CSV into the paper's shape: x, the statistic, its interval, the median.

    The x column keeps the name the exporter gave it (`step` for an evaluation or inference curve,
    `timestep` for an evaluation trajectory), because that is the name the `.tex` reads for each.
    The centre column is named after the statistic it holds, so a file says which one it carries.

    `interval` is the only thing here that is computed rather than copied, and only for a metric
    `PAPER_INTERVAL` declares an indicator: the Wilson bounds for that row's count of successes out
    of its seeds. The centre column is still the exporter's own number.
    """
    header, rows = _read_csv(source)
    wanted = (header[0], statistic, f"{statistic}_ci_lo", f"{statistic}_ci_hi", "median")
    if interval == "wilson":
        wanted = (header[0], "n", statistic, "median")
    missing = [name for name in wanted if name not in header]
    if missing:
        raise MissingExportError(f"{source} has no column(s) {missing}; found {header}")
    index = {name: header.index(name) for name in wanted}
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([header[0], statistic, "ci_lo", "ci_hi", "median"])
        for row in rows:
            x = row[index[header[0]]]
            if interval == "wilson":
                value = float(row[index[statistic]])
                lo, hi = _wilson_row(value, int(row[index["n"]]), source, x)
                writer.writerow(
                    [x, row[index[statistic]], f"{lo:.6g}", f"{hi:.6g}", row[index["median"]]]
                )
                continue
            writer.writerow([row[index[name]] for name in wanted])
    return ProjectedFile(source=source, target=target, rows=len(rows), statistic=statistic)


def paper_curves(domain: str, arm: str) -> list[str]:
    """The canonical stems of one arm's curves that the paper reads in `domain`."""
    try:
        curves = PAPER_CURVES[domain]
    except KeyError:
        raise ValueError(
            f"the paper reads no curve of domain {domain!r}; known: {sorted(PAPER_CURVES)}"
        ) from None
    return [stem for stem, arms in curves.items() if arm in arms]


def project_arm(arm_dir: Path, target_dir: Path, stems: list[str]) -> list[ProjectedFile]:
    """Project the curves `stems` of one arm's export directory under the paper's names.

    A CSV already in `target_dir` that this projection does not write is REMOVED: an arm directory of the paper's tree holds projected curves and nothing else, so a leftover file there is a curve under a name the paper does not read (a renamed statistic, a dropped metric), and the paper must not be able to read a stale number from it.
    """
    if not stems:
        raise ValueError(f"no curve to project for {arm_dir}")
    written: list[ProjectedFile] = []
    for stem in stems:
        source = arm_dir / f"{stem}.csv"
        if not source.exists():
            raise MissingExportError(f"{source} is missing, and the paper reads it")
        statistic = paper_statistic(stem)
        name = paper_file_name(stem, statistic)
        written.append(
            project_curve(source, target_dir / f"{name}.csv", statistic, paper_interval(stem))
        )
    kept = {entry.target for entry in written}
    for stale in sorted(target_dir.glob("*.csv")):
        if stale not in kept:
            stale.unlink()
    return written


@dataclass(frozen=True, slots=True)
class AcceptShare:
    """One arm's share of evaluation episodes that reach Phase 2, in percent, with its interval."""

    value: float
    lo: float
    hi: float


def final_interval(arm_dir: Path, metric_stem: str) -> tuple[float, float, float]:
    """One arm's final mean of `metric_stem` with the bounds of its 95% interval, as the exporter wrote them to `final_intervals.csv` from `PRINTED_DRAWS` resamples.

    The row must name the step, the seed count and the mean of the last row of the metric's curve CSV, as the exporter prints them, and the resample count `PRINTED_DRAWS`, so a stale file cannot pair one run's curve with another run's interval.
    """
    source = arm_dir / FINAL_INTERVALS_FILE
    if not source.exists():
        raise MissingExportError(
            f"{source} is missing: re-export this arm, because the paper prints its final "
            f"intervals from that file"
        )
    header, rows = _read_csv(source)
    matches = [row for row in rows if row[0] == metric_stem]
    if len(matches) != 1:
        raise MissingExportError(f"{source} holds {len(matches)} rows for {metric_stem!r}, not one")
    row = dict(zip(header, matches[0], strict=True))
    curve_header, curve_rows = _read_csv(arm_dir / f"{metric_stem}.csv")
    if not curve_rows:
        raise MissingExportError(f"{arm_dir / f'{metric_stem}.csv'} has no rows")
    last = dict(zip(curve_header, curve_rows[-1], strict=True))
    found = (int(float(row["step"])), row["n"], row["mean"], row["draws"])
    expected = (int(float(curve_rows[-1][0])), last["n"], last["mean"], str(PRINTED_DRAWS))
    if found != expected:
        raise MissingExportError(
            f"{source} gives {metric_stem} as (step, n, mean, draws) {found}, but its curve ends "
            f"at {expected[:3]} and the paper prints {PRINTED_DRAWS} resamples"
        )
    return float(row["mean"]), float(row["mean_ci_lo"]), float(row["mean_ci_hi"])


def accept_share(arm_dir: Path) -> AcceptShare:
    """The arm's share of evaluation episodes that reach Phase 2, as a percentage with its 95% interval.

    Read from the arm's `final_intervals.csv`, which the exporter wrote from the per-seed values the
    evaluation recorded at its last point. The statistic is the one `PAPER_STATISTIC` declares for
    the metric, and nothing here averages anything.
    """
    if paper_statistic(ACCEPT_METRIC) != "mean":
        raise ValueError(f"{ACCEPT_METRIC} must print the mean, which `final_intervals.csv` holds")
    value, lo, hi = (100.0 * number for number in final_interval(arm_dir, ACCEPT_METRIC))
    return AcceptShare(value=value, lo=lo, hi=hi)


def write_accept_table(snapshot: Path, target: Path) -> dict[str, AcceptShare]:
    """Write `accept_state.csv` for one domain: one row, and per arm its share and interval bounds.

    The columns are `<arm>`, `<arm>_lo` and `<arm>_hi` for each arm of `ARM_TABLE_COLUMNS`, so the
    `.tex` reads a value by name and its interval by the same name with a suffix. Every arm must be
    present. A table assembled from four of five arms would print a stale number in the fifth
    column, and the paper's own claim is a comparison across all five.
    """
    shares: dict[str, AcceptShare] = {}
    for arm, column in ARM_TABLE_COLUMNS.items():
        arm_dir = snapshot / arm
        if not arm_dir.is_dir():
            raise MissingExportError(
                f"{snapshot} has no directory for arm {arm!r}; the figure of evaluation trajectories that reach Phase 2 needs all "
                f"{len(ARM_TABLE_COLUMNS)} arms"
            )
        shares[column] = accept_share(arm_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    header: list[str] = []
    values: list[str] = []
    for column in ARM_TABLE_COLUMNS.values():
        share = shares[column]
        header.extend((column, f"{column}_lo", f"{column}_hi"))
        values.extend((f"{share.value:.4f}", f"{share.lo:.4f}", f"{share.hi:.4f}"))
    with target.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerow(values)
    return shares


def write_final_returns(snapshot: Path, target: Path, arms: dict[str, str]) -> None:
    """Write the final mean return of each of `arms` (arm -> column) and its interval bounds for one domain, in one row: `final_returns.csv`, which the paper's table of traffic cycles prints, and the two files of `ABLATION_TABLES`, which the appendix tables print."""
    if paper_statistic(SEED_SCORE_METRIC) != "mean":
        raise ValueError(
            f"{SEED_SCORE_METRIC} must print the mean, which `final_intervals.csv` holds"
        )
    header: list[str] = []
    values: list[str] = []
    for arm, column in arms.items():
        mean, lo, hi = final_interval(snapshot / arm, SEED_SCORE_METRIC)
        header.extend((column, f"{column}_lo", f"{column}_hi"))
        values.extend((f"{mean:.4f}", f"{lo:.4f}", f"{hi:.4f}"))
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerow(values)


def read_seed_sample(arm_dir: Path, metric_stem: str) -> list[float]:
    """One arm's per-seed values of `metric_stem` at the last evaluation point, in seed order."""
    source = arm_dir / PER_SEED_SAMPLE_FILE
    if not source.exists():
        raise MissingExportError(
            f"{source} is missing: re-export this arm, because the per-seed sample is what the "
            f"paper's seed-score panel plots and no reduction can stand in for it"
        )
    _, rows = _read_csv(source)
    for row in rows:
        if row[0] == metric_stem:
            return [float(value) for value in row[1:]]
    raise MissingExportError(f"{source} holds no row for {metric_stem!r}")


def write_seed_scores(snapshot: Path, target: Path, arms: dict[str, str]) -> int:
    """Write `seed_scores.csv` for one domain: the 50 per-seed final returns of `arms` (arm -> column).

    One row per seed and one column per arm, in SEED ORDER, so column `tcirl` row `i` and column
    `dqn4` row `i` are the same seed. This is the one file of the paper's tree that is a sample and
    not a summary: every other file reduces 50 seeds to a centre and an interval, and a centre
    cannot show that an arm's seeds fall into two groups. Returns the number of seeds.
    """
    columns: dict[str, list[float]] = {}
    for arm, column in arms.items():
        arm_dir = snapshot / arm
        if not arm_dir.is_dir():
            raise MissingExportError(f"{snapshot} has no directory for arm {arm!r}")
        columns[column] = read_seed_sample(arm_dir, SEED_SCORE_METRIC)
    sizes = {len(values) for values in columns.values()}
    if len(sizes) != 1:
        raise MissingExportError(
            f"the arms of {snapshot} hold {sorted(sizes)} seeds; one domain is one seed bank, and "
            f"a column of a different length would break the pairing this file exists to keep"
        )
    seeds = sizes.pop()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", *arms.values()])
        for index in range(seeds):
            writer.writerow([index, *(f"{columns[column][index]:.6g}" for column in arms.values())])
    return seeds


# The file in each projected domain that names the export snapshot the tree came from, as `<domain>/<project>/<snapshot>` below the code repository's `tmp/results/`.
# The paper's numbers trace back to their runs through that snapshot's `export_metadata.yaml`.
# The record holds repository-relative paths only.
SOURCE_FILE = "source.yaml"
# The domains whose final returns a table prints, as `final_returns.csv`: the paper's table of traffic cycles puts the learned traffic policies beside the cycles.
FINAL_RETURN_DOMAINS = ("traffic_signal",)
CADENCE_FILES = (
    "cadence.csv",
    "cadence_episodes.csv",
    "cadence_optimum.csv",
    "stationary_policy.npz",
    "cadence_source_records.json",
    "cadence_uv.lock",
    "cadence_source.patch",
)


def write_source(snapshot: Path, target: Path) -> None:
    """Record the export snapshot, with its `latest` link resolved, by its last three path components."""
    place = "/".join(snapshot.resolve().parts[-3:])
    target.write_text(f"snapshot: {json.dumps(place)}\n")


def cadence_files(snapshot: Path) -> tuple[Path, ...]:
    """Validate the complete optional cycle export before a projection writes any result."""
    source = snapshot / "cadence"
    if not source.exists():
        return ()
    missing = [name for name in CADENCE_FILES if not (source / name).is_file()]
    if missing:
        raise MissingExportError(
            f"{source} lacks cadence export files {missing}; nothing was written"
        )
    metadata = json.loads((source / "cadence_source_records.json").read_text())
    for name in CADENCE_FILES:
        if name == "cadence_source_records.json":
            continue
        if sha256((source / name).read_bytes()).hexdigest() != metadata["files"][name]:
            raise MissingExportError(
                f"{source / name} differs from its cadence source records; nothing was written"
            )
    return tuple(source / name for name in CADENCE_FILES)


def project_snapshot(snapshot: Path, domain: str, out_root: Path) -> list[ProjectedFile]:
    """Project one domain's export snapshot into `<out_root>/<domain>/`, arms and table.

    `snapshot` is an export directory holding one subdirectory per arm -- the `latest` symlink the
    reproduce launcher maintains under `tmp/results/<domain>/<project>/`.
    """
    if not snapshot.is_dir():
        raise MissingExportError(f"{snapshot} is not a directory")
    # Check every arm BEFORE writing any of them. A half-written tree is the worst outcome here:
    # the arms that succeeded carry the new run's numbers, the missing one keeps the previous
    # run's, and the paper then prints a mix of two runs with nothing but a traceback to say so.
    missing = [arm for arm in PROJECTED_ARMS if not (snapshot / arm).is_dir()]
    if missing:
        raise MissingExportError(
            f"{snapshot} has no directory for arm(s) {missing}; the paper reads all "
            f"{len(PROJECTED_ARMS)} arms of this domain, so nothing was written"
        )
    missing_accept = [
        arm for arm in ARM_TABLE_COLUMNS if not (snapshot / arm / f"{ACCEPT_METRIC}.csv").exists()
    ]
    if missing_accept:
        raise MissingExportError(
            f"arm(s) {missing_accept} recorded no {ACCEPT_METRIC}, so the figure of evaluation trajectories that reach Phase 2 cannot be "
            f"assembled; nothing was written"
        )
    # Check every curve the paper reads before writing any: a missing or unnamed curve in the fifth arm must not leave the first four written.
    missing_curves: list[str] = []
    for arm in PROJECTED_ARMS:
        for stem in paper_curves(domain, arm):
            paper_file_name(stem, paper_statistic(stem))
            if not (snapshot / arm / f"{stem}.csv").exists():
                missing_curves.append(f"{arm}/{stem}.csv")
    if missing_curves:
        raise MissingExportError(
            f"{snapshot} lacks curve(s) the paper reads: {missing_curves}; nothing was written"
        )
    # Read every final interval a table prints before writing any, so a missing or stale `final_intervals.csv` in one arm writes nothing.
    for arm in ARM_TABLE_COLUMNS:
        final_interval(snapshot / arm, ACCEPT_METRIC)
        if domain in FINAL_RETURN_DOMAINS:
            final_interval(snapshot / arm, SEED_SCORE_METRIC)
    for arm in {arm for arms in ABLATION_TABLES.values() for arm in arms}:
        final_interval(snapshot / arm, SEED_SCORE_METRIC)
    cycles = cadence_files(snapshot) if domain == "traffic_signal" else ()
    domain_out = out_root / domain
    # A projection that fails after its first write must not leave the previous snapshot's name beside the new numbers.
    (domain_out / SOURCE_FILE).unlink(missing_ok=True)
    written: list[ProjectedFile] = []
    for arm in PROJECTED_ARMS:
        written.extend(project_arm(snapshot / arm, domain_out / arm, paper_curves(domain, arm)))
    write_accept_table(snapshot, domain_out / "accept_state.csv")
    if domain in FINAL_RETURN_DOMAINS:
        write_final_returns(snapshot, domain_out / "final_returns.csv", ARM_TABLE_COLUMNS)
    else:
        (domain_out / "final_returns.csv").unlink(missing_ok=True)
    write_seed_scores(snapshot, domain_out / "seed_scores.csv", ARM_TABLE_COLUMNS)
    for name, arms in ABLATION_TABLES.items():
        write_final_returns(snapshot, domain_out / name, arms)
    for source in cycles:
        shutil.copyfile(source, domain_out / source.name)
    if domain == "traffic_signal" and not cycles:
        for name in CADENCE_FILES:
            (domain_out / name).unlink(missing_ok=True)
    write_source(snapshot, domain_out / SOURCE_FILE)
    _prune_stale_arms(domain_out)
    return written


def _prune_stale_arms(domain_out: Path) -> None:
    """Remove any arm directory the paper does not read.

    `project_arm` already removes a stale CSV inside an arm it writes, for the reason that a file
    the projection does not write holds a PREVIOUS run's numbers. A whole stale arm directory is
    the same hazard one level up: it carries the numbers of a run that no current arm reproduces.
    """
    for child in sorted(domain_out.iterdir()):
        if not child.is_dir() or child.name in PROJECTED_ARMS:
            continue
        for stale in sorted(child.glob("*.csv")):
            stale.unlink()
        if not any(child.iterdir()):
            child.rmdir()
