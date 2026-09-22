"""Panel-wide cleaning: calendar alignment, universe-wide drops, outlier flagging.

Two independent operations, chained by the acquisition pipeline:

    raw prices ──clean_prices──▶ cleaned prices ──log_returns──▶ returns
                                                                    │
                                                                    ▼
                                                          flag_return_outliers
                                                                    │
                                                                    ▼
                                                              anomaly log

``clean_prices`` applies ``calendar.align_to_calendar`` to every column
of a raw price panel, then drops the union of multi-day-gap dates from
the panel universe-wide, per the design document §5.1.

``flag_return_outliers`` walks a returns panel with a rolling median-and-
MAD z-score (right-anchored, 63-day window, threshold 8) per §5.4.
Outliers are *flagged*, never removed — robust estimators should see the
same data any user would, and removing outliers would prejudge the
comparison the study is designed to make.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio_risk.data import calendar as cal

# 1 / Phi^{-1}(0.75). Precomputed to avoid a scipy import for a constant.
# Scales the MAD so that for Gaussian data, MAD * MAD_SCALING is a
# consistent estimator of the standard deviation.
MAD_SCALING: float = 1.4826022185056

DEFAULT_OUTLIER_WINDOW: int = 63
DEFAULT_OUTLIER_Z_THRESHOLD: float = 8.0


@dataclass(frozen=True, slots=True)
class CleaningResult:
    """Output of ``clean_prices``.

    Attributes:
        prices: Cleaned adjusted-close panel indexed by the NYSE
            calendar (minus any universe-wide dropped dates). Columns
            match the input.
        dropped_dates: Dates dropped universe-wide because at least one
            instrument had a multi-day gap on those dates.
        gap_reports: Per-ticker ``GapReport`` from ``align_to_calendar``.
    """

    prices: pd.DataFrame
    dropped_dates: pd.DatetimeIndex
    gap_reports: dict[str, cal.GapReport]


def clean_prices(
    raw_prices: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> CleaningResult:
    """Align a raw price panel to the NYSE calendar and drop universe-wide gaps.

    For each column, this calls ``calendar.align_to_calendar`` to reindex
    to the trading-day calendar and forward-fill isolated 1-day gaps. It
    then computes the union of all instruments' multi-day gap dates and
    drops those dates from the panel — those dates are, by policy, not
    usable for cross-sectional analysis.

    Args:
        raw_prices: Raw adjusted-close panel from the fetcher. May have
            differing per-column indices (pre-inception NaN allowed).
            Columns are ticker symbols.
        calendar: NYSE trading-day calendar covering the study window.

    Returns:
        A ``CleaningResult`` bundling the cleaned panel, the dropped
        dates, and per-instrument gap reports.
    """
    aligned_columns: dict[str, pd.Series] = {}
    gap_reports: dict[str, cal.GapReport] = {}

    for ticker in raw_prices.columns:
        aligned, report = cal.align_to_calendar(raw_prices[ticker], calendar)
        aligned_columns[ticker] = aligned
        gap_reports[ticker] = report

    panel = pd.DataFrame(aligned_columns, index=calendar)
    dropped = cal.universe_dropped_dates(gap_reports)
    if len(dropped) > 0:
        panel = panel.drop(index=dropped)

    return CleaningResult(prices=panel, dropped_dates=dropped, gap_reports=gap_reports)


def _mad(x: np.ndarray) -> float:
    """Median absolute deviation of a 1-D array."""
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def flag_return_outliers(
    returns: pd.DataFrame,
    window: int = DEFAULT_OUTLIER_WINDOW,
    z_threshold: float = DEFAULT_OUTLIER_Z_THRESHOLD,
) -> pd.DataFrame:
    """Flag return observations whose rolling-MAD z-score exceeds a threshold.

    Uses a right-anchored rolling window and the scaled median absolute
    deviation as a robust standard-deviation estimator. Observations in
    the first ``window - 1`` days of each series cannot be flagged
    (insufficient history — ``min_periods=window``).

    Args:
        returns: Log-return panel indexed by ``DatetimeIndex``, columns
            are ticker symbols. NaN observations are ignored.
        window: Rolling window length in trading days. Default 63
            (~3 months) per design document §5.4.
        z_threshold: Absolute z-score above which to flag. Default 8.

    Returns:
        Long-form ``DataFrame`` with one row per flagged observation.
        Columns: ``ticker``, ``date``, ``return``, ``median``, ``mad``,
        ``z_score``. Sorted by ``(date, ticker)``. Empty (but with the
        correct schema) if no outliers are flagged.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if z_threshold <= 0:
        raise ValueError(f"z_threshold must be positive, got {z_threshold}")

    rolling = returns.rolling(window=window, min_periods=window)
    median = rolling.median()
    mad = rolling.apply(_mad, raw=True)

    # Suppress the divide-by-zero warning that appears in flat sub-windows
    # (MAD = 0); z-score becomes ±inf there and is either flagged or not
    # by the same threshold logic.
    with np.errstate(divide="ignore", invalid="ignore"):
        z_score = (returns - median) / (MAD_SCALING * mad)

    flagged_mask = z_score.abs() > z_threshold
    columns = ["ticker", "date", "return", "median", "mad", "z_score"]

    if not bool(flagged_mask.to_numpy().any()):
        return pd.DataFrame(columns=columns)

    row_idx, col_idx = np.where(flagged_mask.to_numpy())
    dates = returns.index[row_idx]
    tickers = returns.columns[col_idx]
    values = returns.to_numpy()[row_idx, col_idx]
    medians = median.to_numpy()[row_idx, col_idx]
    mads = mad.to_numpy()[row_idx, col_idx]
    zs = z_score.to_numpy()[row_idx, col_idx]

    return (
        pd.DataFrame(
            {
                "ticker": tickers,
                "date": dates,
                "return": values,
                "median": medians,
                "mad": mads,
                "z_score": zs,
            }
        )
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
