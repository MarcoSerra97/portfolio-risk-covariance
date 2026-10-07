"""Paired stationary block bootstrap (Politis & Romano, 1994).

Implements ``docs/02-evaluation-design.md`` §10.

The stationary bootstrap resamples a time series in blocks of random,
geometrically distributed length with mean ``mean_block``, wrapping around the
end of the sample. Random block lengths make the resampled series stationary,
and blocks preserve short-range dependence (for example volatility clustering
across consecutive holding periods or days).

**Pairing.** Each replication draws one index sequence and applies it to every
column of the data at once. When columns are estimators, statistics of
differences between columns (e.g. QLIKE of QIS minus QLIKE of the sample
covariance) are computed on the same resampled periods, so their intervals
account for the noise the estimators share.

Intervals are percentile intervals.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


def stationary_bootstrap_indices(
    n: int, mean_block: float, reps: int, rng: np.random.Generator
) -> np.ndarray:
    """Index sequences of the stationary bootstrap, shape ``(reps, n)``.

    Each sequence starts at a uniform random position; at every step it either
    continues to the next index (probability ``1 - 1/mean_block``) or jumps to
    a new uniform random position. Indices wrap around modulo ``n``.

    Raises:
        ValueError: If ``n < 1``, ``reps < 1`` or ``mean_block < 1``.
    """
    if n < 1 or reps < 1:
        raise ValueError("n and reps must be >= 1")
    if mean_block < 1:
        raise ValueError(f"mean_block must be >= 1, got {mean_block}")
    p_new = 1.0 / mean_block
    idx = np.empty((reps, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, size=reps)
    jumps = rng.random((reps, n)) < p_new
    fresh = rng.integers(0, n, size=(reps, n))
    for t in range(1, n):
        idx[:, t] = np.where(jumps[:, t], fresh[:, t], (idx[:, t - 1] + 1) % n)
    return idx


@dataclass(frozen=True)
class BootstrapResult:
    """Point estimates, percentile intervals and replications of a statistic.

    Attributes:
        estimate: Statistic on the original data, one value per output.
        lower, upper: Percentile interval bounds.
        draws: Statistic on each replication, shape ``(reps, n_outputs)``.
        level: Confidence level of the interval.
    """

    estimate: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    draws: np.ndarray
    level: float


def paired_bootstrap(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], np.ndarray],
    mean_block: float,
    reps: int,
    seed: int,
    level: float = 0.95,
) -> BootstrapResult:
    """Bootstrap a statistic of the rows of ``data``, resampling all columns together.

    Args:
        data: Array of shape ``(n, m)``: ``n`` time points (rows), ``m`` series.
        statistic: Maps an ``(n, m)`` array to a 1-d array of outputs.
        mean_block: Mean block length in rows.
        reps: Number of replications.
        seed: Seed for reproducibility.
        level: Confidence level of the percentile interval.

    Returns:
        A :class:`BootstrapResult`.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data[:, None]
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(data.shape[0], mean_block, reps, rng)
    estimate = np.atleast_1d(np.asarray(statistic(data), dtype=float))
    draws = np.vstack([np.atleast_1d(statistic(data[i])) for i in idx])
    alpha = (1.0 - level) / 2.0
    lower, upper = np.quantile(draws, [alpha, 1.0 - alpha], axis=0)
    return BootstrapResult(estimate, lower, upper, draws, level)


def mean_differences(baseline: int) -> Callable[[np.ndarray], np.ndarray]:
    """Statistic: column means minus the mean of column ``baseline``."""

    def stat(x: np.ndarray) -> np.ndarray:
        m = x.mean(axis=0)
        out: np.ndarray = m - m[baseline]
        return out

    return stat


def volatility_ratios(
    baseline: int, periods_per_year: int = 252
) -> Callable[[np.ndarray], np.ndarray]:
    """Statistic: annualised volatility of each column divided by that of column ``baseline``."""

    def stat(x: np.ndarray) -> np.ndarray:
        sd = x.std(axis=0, ddof=1) * np.sqrt(periods_per_year)
        out: np.ndarray = sd / sd[baseline]
        return out

    return stat


__all__ = [
    "BootstrapResult",
    "mean_differences",
    "paired_bootstrap",
    "stationary_bootstrap_indices",
    "volatility_ratios",
]
