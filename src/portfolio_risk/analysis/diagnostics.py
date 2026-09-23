"""Diagnostics for the study data panel.

Helpers used by ``notebooks/01_data_diagnostics.ipynb`` to produce the
design-document §7 deliverables:

    - Coverage table (per-instrument availability)
    - Rolling coverage plot (# live instruments per date)
    - Return distribution statistics (moments + unit-root test)
    - Full-sample correlation heatmap
    - Rolling median pairwise correlation (regime tell-tale)

Each function is pure and takes an in-memory panel. The notebook is
responsible for loading the parquets and saving the outputs into
``memo/figures/`` and ``memo/tables/``.
"""

from __future__ import annotations

from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from statsmodels.tsa.stattools import adfuller

from portfolio_risk.data import universe

DEFAULT_TRADING_DAYS_PER_YEAR: int = 252
DEFAULT_ROLLING_CORR_WINDOW: int = 63


def compute_coverage_table(
    prices: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Per-instrument coverage statistics.

    Combines the universe metadata with observed data availability. For
    each instrument reports its first and last observation dates, the
    number of observations, the expected number of NYSE trading days
    over its active period, and the missing fraction.

    Args:
        prices: Cleaned adjusted-close panel; index is NYSE trading days.
        calendar: The canonical NYSE trading-day calendar.

    Returns:
        DataFrame indexed by ticker with columns: name, bucket, subgroup,
        inception, first_obs, last_obs, n_obs, expected_obs, pct_missing,
        notes. Rows are ordered by the canonical universe order.
    """
    meta = universe.metadata_frame()
    records = []
    for ticker in meta.index:
        if ticker not in prices.columns:
            records.append(
                {
                    "ticker": ticker,
                    "first_obs": pd.NaT,
                    "last_obs": pd.NaT,
                    "n_obs": 0,
                    "expected_obs": 0,
                    "pct_missing": np.nan,
                }
            )
            continue
        series = prices[ticker]
        non_na = series.dropna()
        if non_na.empty:
            records.append(
                {
                    "ticker": ticker,
                    "first_obs": pd.NaT,
                    "last_obs": pd.NaT,
                    "n_obs": 0,
                    "expected_obs": 0,
                    "pct_missing": np.nan,
                }
            )
            continue
        first = non_na.index[0]
        last = non_na.index[-1]
        n_obs = int(non_na.shape[0])
        active_calendar = calendar[(calendar >= first) & (calendar <= last)]
        expected_obs = len(active_calendar)
        pct_missing = 1.0 - n_obs / expected_obs if expected_obs > 0 else np.nan
        records.append(
            {
                "ticker": ticker,
                "first_obs": first,
                "last_obs": last,
                "n_obs": n_obs,
                "expected_obs": expected_obs,
                "pct_missing": pct_missing,
            }
        )

    coverage = pd.DataFrame(records).set_index("ticker")
    joined = meta.join(coverage, how="left")
    columns = [
        "name",
        "bucket",
        "subgroup",
        "inception",
        "first_obs",
        "last_obs",
        "n_obs",
        "expected_obs",
        "pct_missing",
        "notes",
    ]
    return joined[columns]


def compute_rolling_coverage(prices: pd.DataFrame) -> pd.Series:
    """Number of instruments with a non-NaN observation on each date."""
    result = prices.notna().sum(axis=1)
    result.name = "n_live"
    return result


def compute_return_stats(
    returns: pd.DataFrame,
    trading_days_per_year: int = DEFAULT_TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Per-instrument distribution statistics.

    Reports daily and annualised location and scale, higher moments,
    extremes, and the augmented Dickey-Fuller unit-root test statistic
    and p-value.

    Args:
        returns: Log-return panel.
        trading_days_per_year: Scaling factor for annualisation.

    Returns:
        DataFrame indexed by ticker with columns: mean_daily, mean_annual,
        std_daily, std_annual, skew, excess_kurtosis, min, max, adf_stat,
        adf_pvalue, n_obs.
    """
    records = []
    for ticker in returns.columns:
        r = returns[ticker].dropna()
        if r.empty:
            records.append(
                {
                    "ticker": ticker,
                    "mean_daily": np.nan,
                    "mean_annual": np.nan,
                    "std_daily": np.nan,
                    "std_annual": np.nan,
                    "skew": np.nan,
                    "excess_kurtosis": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                    "adf_stat": np.nan,
                    "adf_pvalue": np.nan,
                    "n_obs": 0,
                }
            )
            continue
        adf_stat, adf_pvalue = _safe_adf(r.to_numpy())
        r_np = r.to_numpy(dtype=float)
        records.append(
            {
                "ticker": ticker,
                "mean_daily": float(r_np.mean()),
                "mean_annual": float(r_np.mean() * trading_days_per_year),
                "std_daily": float(r_np.std(ddof=1)),
                "std_annual": float(r_np.std(ddof=1) * np.sqrt(trading_days_per_year)),
                "skew": float(cast(float, r.skew())),
                "excess_kurtosis": float(cast(float, r.kurt())),
                "min": float(r_np.min()),
                "max": float(r_np.max()),
                "adf_stat": adf_stat,
                "adf_pvalue": adf_pvalue,
                "n_obs": len(r),
            }
        )
    return pd.DataFrame(records).set_index("ticker")


def _safe_adf(values: np.ndarray) -> tuple[float, float]:
    """Run ADF and return (statistic, p-value) or (NaN, NaN) on failure."""
    if len(values) < 12:
        return (float("nan"), float("nan"))
    try:
        result = adfuller(values, autolag="AIC")
    except (ValueError, np.linalg.LinAlgError):
        return (float("nan"), float("nan"))
    return (float(result[0]), float(result[1]))


def compute_full_correlation(returns: pd.DataFrame) -> pd.DataFrame:
    """Full-sample Pearson correlation matrix, canonically ordered.

    Column and row order follows the universe's bucket → subgroup order,
    so bucket structure is visible as blocks along the diagonal.
    """
    meta = universe.metadata_frame()
    ordered = [t for t in meta.index if t in returns.columns]
    return returns[ordered].corr()


def compute_rolling_pairwise_correlation(
    returns: pd.DataFrame,
    window: int = DEFAULT_ROLLING_CORR_WINDOW,
) -> pd.Series:
    """Median off-diagonal pairwise correlation on a rolling window.

    For each date t with at least ``window`` trailing observations, this
    computes the correlation matrix of the trailing window and returns
    the median of its off-diagonal (upper-triangle) entries. Provides a
    single-number summary of how correlated the panel is at each date;
    spikes flag regime shifts (crises, recoveries).

    Args:
        returns: Log-return panel.
        window: Trailing window in trading days.

    Returns:
        Series indexed by ``returns.index``; entries in the first
        ``window - 1`` positions are NaN.
    """
    n_dates = len(returns)
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if window > n_dates:
        raise ValueError(f"window ({window}) larger than data ({n_dates})")

    n_assets = returns.shape[1]
    triu_i, triu_j = np.triu_indices(n_assets, k=1)
    results = np.full(n_dates, np.nan)

    for i in range(window - 1, n_dates):
        window_slice = returns.iloc[i - window + 1 : i + 1]
        corr = window_slice.corr().to_numpy()
        off_diag = corr[triu_i, triu_j]
        off_diag = off_diag[~np.isnan(off_diag)]
        if len(off_diag) > 0:
            results[i] = float(np.median(off_diag))

    return pd.Series(results, index=returns.index, name="median_pairwise_correlation")


def bucket_boundary_positions(ordered_tickers: list[str]) -> list[int]:
    """Positions in an ordered ticker list where the bucket changes.

    Used to draw dividers on the correlation heatmap between buckets.
    """
    meta = universe.metadata_frame()
    positions: list[int] = []
    prev_bucket: str | None = None
    for i, ticker in enumerate(ordered_tickers):
        bucket = str(meta.at[ticker, "bucket"])
        if prev_bucket is not None and bucket != prev_bucket:
            positions.append(i)
        prev_bucket = bucket
    return positions


def plot_rolling_coverage(coverage: pd.Series, target: int = 75) -> Figure:
    """Line plot of live-instrument count over time."""
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(coverage.index, coverage.to_numpy(), color="steelblue", linewidth=1.2)
    ax.axhline(
        target, color="black", linestyle="--", linewidth=0.7, label=f"Full universe ({target})"
    )
    ax.set_ylabel("Instruments live")
    ax.set_xlabel("")
    ax.set_title("Rolling coverage — number of live instruments by date")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(
    corr: pd.DataFrame,
    bucket_boundaries: list[int] | None = None,
    tick_fontsize: int = 6,
) -> Figure:
    """Heatmap of a correlation matrix with optional bucket dividers."""
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        corr,
        ax=ax,
        cmap="RdBu_r",
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        square=True,
        cbar_kws={"shrink": 0.55, "label": "Pearson correlation"},
        xticklabels=True,
        yticklabels=True,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=tick_fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=tick_fontsize)
    if bucket_boundaries:
        for boundary in bucket_boundaries:
            ax.axhline(boundary, color="black", linewidth=0.8)
            ax.axvline(boundary, color="black", linewidth=0.8)
    ax.set_title("Full-sample correlation matrix (log-returns)")
    fig.tight_layout()
    return fig


def plot_rolling_correlation(rolling_corr: pd.Series, window: int) -> Figure:
    """Line plot of rolling median pairwise correlation."""
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(rolling_corr.index, rolling_corr.to_numpy(), color="darkorange", linewidth=1.0)
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_ylabel(r"Median pairwise $\rho$")
    ax.set_xlabel("")
    ax.set_title(f"Rolling median pairwise correlation ({window}-day window)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


__all__ = [
    "DEFAULT_ROLLING_CORR_WINDOW",
    "DEFAULT_TRADING_DAYS_PER_YEAR",
    "bucket_boundary_positions",
    "compute_coverage_table",
    "compute_full_correlation",
    "compute_return_stats",
    "compute_rolling_coverage",
    "compute_rolling_pairwise_correlation",
    "plot_correlation_heatmap",
    "plot_rolling_correlation",
    "plot_rolling_coverage",
]
