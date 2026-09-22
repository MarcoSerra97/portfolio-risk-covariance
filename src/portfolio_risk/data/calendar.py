"""NYSE trading calendar utilities and gap classification.

Provides a canonical NYSE trading-day calendar for the study window and
helpers that align instrument price series to the calendar, classifying
missing observations as isolated (forward-fillable) or multi-day
(universe-wide drop candidates) per the design document
(docs/01-data-acquisition-design.md §4–§5).

Design decisions encoded here:
    - The reference calendar is NYSE; instruments trading on other
      calendars (e.g. FX) are reindexed to NYSE trading days.
    - Half-days are treated as full trading days (they are valid NYSE
      trading days for our purposes).
    - Pre-inception NaN and post-delisting NaN are not gaps — a gap
      exists only inside the instrument's active period.
    - Isolated 1-day gaps get forward-filled from the previous close,
      producing a zero log-return on the gap day and the true return
      on the following day.
    - Multi-day gaps (2+ consecutive missing trading days) stay as NaN
      and are collected across the panel so the caller can drop the
      offending dates universe-wide.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import cast

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

NYSE_CALENDAR_NAME: str = "NYSE"


def nyse_trading_days(
    start: str | date | pd.Timestamp,
    end: str | date | pd.Timestamp,
) -> pd.DatetimeIndex:
    """Return the NYSE trading-day calendar between two dates (inclusive).

    Both endpoints are inclusive. Half-days are included. The returned
    index is timezone-naive at midnight so it joins cleanly against
    timezone-naive price data from yfinance.

    Args:
        start: Inclusive start date. Any type accepted by
            ``pandas_market_calendars.MarketCalendar.valid_days``.
        end: Inclusive end date.

    Returns:
        Sorted-ascending, timezone-naive ``pd.DatetimeIndex`` of NYSE
        trading days.
    """
    nyse = mcal.get_calendar(NYSE_CALENDAR_NAME)
    days = nyse.valid_days(start_date=start, end_date=end)
    return pd.DatetimeIndex(days.tz_localize(None))


def is_nyse_trading_day(day: str | date | pd.Timestamp) -> bool:
    """Return True if ``day`` is a valid NYSE trading day."""
    ts = pd.Timestamp(day)
    calendar = nyse_trading_days(ts, ts)
    return len(calendar) == 1


@dataclass(frozen=True, slots=True)
class GapReport:
    """Classification of missing observations for one price series.

    Attributes:
        isolated: Dates on which the instrument was missing but both
            neighbouring NYSE trading days were present. Safe to
            forward-fill without introducing a spurious jump.
        multi_day: Dates that fell inside a run of two or more
            consecutive missing NYSE trading days within the instrument's
            active period. Candidates for universe-wide drop.
        active_start: First non-missing date in the reindexed series,
            or ``None`` if the series is empty after reindexing.
        active_end: Last non-missing date, or ``None`` if empty.
    """

    isolated: pd.DatetimeIndex
    multi_day: pd.DatetimeIndex
    active_start: pd.Timestamp | None
    active_end: pd.Timestamp | None


def classify_gaps(prices: pd.Series, calendar: pd.DatetimeIndex) -> GapReport:
    """Classify missing observations in a price series against a calendar.

    Only considers dates inside the instrument's active period (first to
    last non-NaN observation). Pre-inception and post-delisting NaNs are
    ignored — they are not "gaps".

    Args:
        prices: Price series indexed by a ``DatetimeIndex``. May contain
            NaNs; its index may be a subset or superset of ``calendar``.
        calendar: The canonical NYSE trading-day calendar to align against.

    Returns:
        A ``GapReport`` naming the isolated and multi-day missing dates.
    """
    aligned = prices.reindex(calendar)
    non_na = aligned.dropna()
    if non_na.empty:
        empty_idx = pd.DatetimeIndex([])
        return GapReport(
            isolated=empty_idx,
            multi_day=empty_idx,
            active_start=None,
            active_end=None,
        )

    active_start = cast(pd.Timestamp, non_na.index[0])
    active_end = cast(pd.Timestamp, non_na.index[-1])
    active = aligned.loc[active_start:active_end]

    is_na = active.isna().to_numpy()
    isolated_dates: list[pd.Timestamp] = []
    multi_day_dates: list[pd.Timestamp] = []

    i = 0
    n = len(is_na)
    while i < n:
        if is_na[i]:
            j = i
            while j < n and is_na[j]:
                j += 1
            run = active.index[i:j]
            if len(run) == 1:
                isolated_dates.extend(run)
            else:
                multi_day_dates.extend(run)
            i = j
        else:
            i += 1

    return GapReport(
        isolated=pd.DatetimeIndex(isolated_dates),
        multi_day=pd.DatetimeIndex(multi_day_dates),
        active_start=active_start,
        active_end=active_end,
    )


def align_to_calendar(
    prices: pd.Series,
    calendar: pd.DatetimeIndex,
) -> tuple[pd.Series, GapReport]:
    """Reindex to the NYSE calendar and forward-fill isolated 1-day gaps.

    Multi-day gaps within the active period stay as NaN so the caller can
    drop those dates universe-wide. Pre-inception and post-delisting NaNs
    are preserved unchanged.

    Args:
        prices: Price series indexed by a ``DatetimeIndex``.
        calendar: The canonical NYSE trading-day calendar.

    Returns:
        Tuple of ``(aligned_series, gap_report)``. The aligned series has
        ``calendar`` as its index; any remaining NaNs are either
        pre-inception, post-delisting, or inside a multi-day gap.
    """
    report = classify_gaps(prices, calendar)
    aligned = prices.reindex(calendar)

    if report.active_start is None:
        return aligned, report

    active_mask = (aligned.index >= report.active_start) & (aligned.index <= report.active_end)
    active_slice = aligned.loc[active_mask].ffill()
    if len(report.multi_day) > 0:
        active_slice.loc[report.multi_day] = np.nan
    aligned.loc[active_mask] = active_slice

    return aligned, report


def universe_dropped_dates(
    gap_reports: dict[str, GapReport],
) -> pd.DatetimeIndex:
    """Return the union of multi-day-gap dates across all instruments.

    These dates should be dropped from the panel universe-wide per the
    design document §5.1.

    Args:
        gap_reports: Mapping ``ticker -> GapReport`` from
            ``align_to_calendar``.

    Returns:
        Sorted, deduplicated ``DatetimeIndex`` of all multi-day-gap dates.
    """
    all_dates: set[pd.Timestamp] = set()
    for report in gap_reports.values():
        all_dates.update(report.multi_day)
    return pd.DatetimeIndex(sorted(all_dates))
