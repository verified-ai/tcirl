from pathlib import Path

ROOT_DIR: Path = Path(__file__).parent.parent.parent
CONFIG_DIR: Path = ROOT_DIR / "config"
DATA_DIR: Path = ROOT_DIR / "data"
TMP_DIR: Path = ROOT_DIR / "tmp"
CACHE_DIR: Path = TMP_DIR / "cache"

# The one file an export writes that holds a SAMPLE rather than a reduction of one: per row an
# evaluation metric, per column a seed, at the last point of the run's evaluation grid.
# `tcirl.export` writes it and `tcirl.paper_results` reads it, and the name lives here so the
# second does not have to import the first, which would pull jax and matplotlib into a CSV copier.
PER_SEED_SAMPLE_FILE = "per_seed_final.csv"
PER_SEED_SAMPLE_PREFIX = "eval_final/"
# The final interval of every evaluation metric, from as many resamples as an interval needs to be printed as a number (`tcirl.export.PRINTED_DRAWS`); the paper's tables read it through `tcirl.paper_results`.
FINAL_INTERVALS_FILE = "final_intervals.csv"
# The resamples of every interval that is printed as a number, the default of rliable (Agarwal et al. 2021), which `final_intervals.csv` records in its `draws` column.
# A curve keeps fewer resamples per point, because its bounds are drawn rather than printed.
PRINTED_DRAWS = 50_000
