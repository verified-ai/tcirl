"""Closed-form confidence intervals: the ones that are a formula rather than a resample.

`tcirl.metrics.stats` holds the bootstrap, which is where nearly every interval in this repository
comes from. This module holds the exception and nothing else, and it imports only the standard
library on purpose: `tcirl.paper_results` needs it, and that module copies CSV columns and must not
drag in jax, jaxlib and matplotlib to do so.
"""

from __future__ import annotations

import math
from statistics import NormalDist

# The two-sided 95% normal quantile, the same 95% the bootstrap intervals of this repository report.
Z_95 = NormalDist().inv_cdf(0.975)


def wilson_interval(successes: int, n: int, *, z: float = Z_95) -> tuple[float, float]:
    """The Wilson score interval for `successes` out of `n` independent trials, on the 0..1 scale.

    It is the standard interval for a proportion, and it keeps a positive width when every trial
    succeeds or every trial fails, where a bootstrap of the observed values has zero width.

    `successes` is a COUNT, so whoever recovers it from a mean rounds it and says what tolerance
    made that rounding safe; this function will not silently take a fractional trial.
    """
    if n <= 0:
        raise ValueError(f"a proportion needs at least one trial, got n={n}")
    if not 0 <= successes <= n:
        raise ValueError(f"{successes} successes is outside 0..{n}")
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    half_width = z * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n))
    return (centre - half_width) / denominator, (centre + half_width) / denominator
