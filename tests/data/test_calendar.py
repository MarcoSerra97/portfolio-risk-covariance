"""Tests for the NYSE calendar utilities and gap classification."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.data import calendar as cal


class TestNyseTradingDays:
    def test_known_year_2019(self) -> None:
        """2019 had 252 NYSE trading days."""
        days = cal.nyse_trading_days("2019-01-01", "2019-12-31")
        assert len(days) == 252

    def test_endpoints_inclusive(self) -> None:
        days = cal.nyse_trading_days("2019-01-02", "2019-01-02")
        assert len(days) == 1
        assert days[0] == pd.Timestamp("2019-01-02")

    def test_new_years_day_excluded(self) -> None:
        days = cal.nyse_trading_days("2019-01-01", "2019-01-01")
        assert len(days) == 0

    def test_weekend_excluded(self) -> None:
        # 2019-03-16 is a Saturday, 2019-03-17 is Sunday
        days = cal.nyse_trading_days("2019-03-16", "2019-03-17")
        assert len(days) == 0

    def test_returns_timezone_naive(self) -> None:
        days = cal.nyse_trading_days("2019-01-01", "2019-12-31")
        assert days.tz is None

    def test_sorted_ascending(self) -> None:
        days = cal.nyse_trading_days("2019-01-01", "2019-12-31")
        assert days.is_monotonic_increasing


class TestIsNyseTradingDay:
    def test_ordinary_weekday(self) -> None:
        assert cal.is_nyse_trading_day("2019-03-14") is True

    def test_saturday(self) -> None:
        assert cal.is_nyse_trading_day("2019-03-16") is False

    def test_holiday(self) -> None:
        assert cal.is_nyse_trading_day("2019-01-01") is False


@pytest.fixture
def calendar_10() -> pd.DatetimeIndex:
    """A calendar of 10 consecutive NYSE trading days (Jan 2019)."""
    return cal.nyse_trading_days("2019-01-02", "2019-01-15")


class TestClassifyGaps:
    def test_no_gaps(self, calendar_10: pd.DatetimeIndex) -> None:
        prices = pd.Series(np.arange(len(calendar_10), dtype=float), index=calendar_10)
        report = cal.classify_gaps(prices, calendar_10)
        assert len(report.isolated) == 0
        assert len(report.multi_day) == 0
        assert report.active_start == calendar_10[0]
        assert report.active_end == calendar_10[-1]

    def test_isolated_gap(self, calendar_10: pd.DatetimeIndex) -> None:
        prices = pd.Series(np.arange(len(calendar_10), dtype=float), index=calendar_10)
        prices.iloc[3] = np.nan
        report = cal.classify_gaps(prices, calendar_10)
        assert len(report.isolated) == 1
        assert report.isolated[0] == calendar_10[3]
        assert len(report.multi_day) == 0

    def test_multi_day_gap(self, calendar_10: pd.DatetimeIndex) -> None:
        prices = pd.Series(np.arange(len(calendar_10), dtype=float), index=calendar_10)
        prices.iloc[3:6] = np.nan
        report = cal.classify_gaps(prices, calendar_10)
        assert len(report.isolated) == 0
        assert len(report.multi_day) == 3
        assert list(report.multi_day) == list(calendar_10[3:6])

    def test_two_day_gap_is_multi_day(self, calendar_10: pd.DatetimeIndex) -> None:
        """The isolated/multi-day boundary is at 1 vs 2."""
        prices = pd.Series(np.arange(len(calendar_10), dtype=float), index=calendar_10)
        prices.iloc[3:5] = np.nan
        report = cal.classify_gaps(prices, calendar_10)
        assert len(report.isolated) == 0
        assert len(report.multi_day) == 2

    def test_mixed_gaps(self, calendar_10: pd.DatetimeIndex) -> None:
        prices = pd.Series(np.arange(len(calendar_10), dtype=float), index=calendar_10)
        prices.iloc[2] = np.nan
        prices.iloc[5:7] = np.nan
        report = cal.classify_gaps(prices, calendar_10)
        assert list(report.isolated) == [calendar_10[2]]
        assert list(report.multi_day) == list(calendar_10[5:7])

    def test_pre_inception_ignored(self, calendar_10: pd.DatetimeIndex) -> None:
        """NaNs before the first non-NaN observation are not gaps."""
        prices = pd.Series(np.nan, index=calendar_10)
        prices.iloc[3:] = np.arange(len(calendar_10) - 3, dtype=float)
        report = cal.classify_gaps(prices, calendar_10)
        assert len(report.isolated) == 0
        assert len(report.multi_day) == 0
        assert report.active_start == calendar_10[3]
        assert report.active_end == calendar_10[-1]

    def test_post_delisting_ignored(self, calendar_10: pd.DatetimeIndex) -> None:
        """NaNs after the last non-NaN observation are not gaps."""
        prices = pd.Series(np.arange(len(calendar_10), dtype=float), index=calendar_10)
        prices.iloc[-3:] = np.nan
        report = cal.classify_gaps(prices, calendar_10)
        assert len(report.isolated) == 0
        assert len(report.multi_day) == 0
        assert report.active_end == calendar_10[-4]

    def test_empty_series(self, calendar_10: pd.DatetimeIndex) -> None:
        prices = pd.Series(np.nan, index=calendar_10)
        report = cal.classify_gaps(prices, calendar_10)
        assert len(report.isolated) == 0
        assert len(report.multi_day) == 0
        assert report.active_start is None
        assert report.active_end is None

    def test_multiple_isolated_gaps(self, calendar_10: pd.DatetimeIndex) -> None:
        prices = pd.Series(np.arange(len(calendar_10), dtype=float), index=calendar_10)
        prices.iloc[2] = np.nan
        prices.iloc[5] = np.nan
        prices.iloc[8] = np.nan
        report = cal.classify_gaps(prices, calendar_10)
        assert len(report.isolated) == 3
        assert len(report.multi_day) == 0

    def test_gap_at_active_boundary(self, calendar_10: pd.DatetimeIndex) -> None:
        """A gap right after inception should still classify."""
        prices = pd.Series(np.nan, index=calendar_10)
        prices.iloc[2] = 100.0
        # active_start = calendar[2]; isolated NaN at calendar[3]; then values
        prices.iloc[4:] = np.arange(len(calendar_10) - 4, dtype=float)
        report = cal.classify_gaps(prices, calendar_10)
        assert list(report.isolated) == [calendar_10[3]]


class TestAlignToCalendar:
    def test_isolated_gap_is_ffilled(self, calendar_10: pd.DatetimeIndex) -> None:
        prices = pd.Series(np.arange(len(calendar_10), dtype=float), index=calendar_10)
        prices.iloc[3] = np.nan
        aligned, _ = cal.align_to_calendar(prices, calendar_10)
        assert aligned.iloc[3] == prices.iloc[2]
        assert not aligned.isna().any()

    def test_multi_day_gap_stays_nan(self, calendar_10: pd.DatetimeIndex) -> None:
        prices = pd.Series(np.arange(len(calendar_10), dtype=float), index=calendar_10)
        prices.iloc[3:6] = np.nan
        aligned, report = cal.align_to_calendar(prices, calendar_10)
        assert aligned.iloc[3:6].isna().all()
        assert len(report.multi_day) == 3

    def test_pre_inception_stays_nan(self, calendar_10: pd.DatetimeIndex) -> None:
        prices = pd.Series(np.nan, index=calendar_10)
        prices.iloc[5:] = np.arange(len(calendar_10) - 5, dtype=float)
        aligned, _ = cal.align_to_calendar(prices, calendar_10)
        assert aligned.iloc[:5].isna().all()
        assert not aligned.iloc[5:].isna().any()

    def test_mixed(self, calendar_10: pd.DatetimeIndex) -> None:
        prices = pd.Series(np.arange(len(calendar_10), dtype=float), index=calendar_10)
        prices.iloc[2] = np.nan  # isolated -> ffill
        prices.iloc[5:7] = np.nan  # multi-day -> stays NaN
        aligned, report = cal.align_to_calendar(prices, calendar_10)
        assert not np.isnan(aligned.iloc[2])
        assert aligned.iloc[5:7].isna().all()
        assert len(report.isolated) == 1
        assert len(report.multi_day) == 2

    def test_no_change_when_complete(self, calendar_10: pd.DatetimeIndex) -> None:
        prices = pd.Series(np.arange(len(calendar_10), dtype=float), index=calendar_10)
        aligned, report = cal.align_to_calendar(prices, calendar_10)
        pd.testing.assert_series_equal(aligned, prices, check_names=False)
        assert len(report.isolated) == 0
        assert len(report.multi_day) == 0

    def test_index_matches_calendar(self, calendar_10: pd.DatetimeIndex) -> None:
        prices = pd.Series(np.arange(len(calendar_10), dtype=float), index=calendar_10)
        aligned, _ = cal.align_to_calendar(prices, calendar_10)
        pd.testing.assert_index_equal(aligned.index, calendar_10)


class TestUniverseDroppedDates:
    def test_union_across_reports(self, calendar_10: pd.DatetimeIndex) -> None:
        prices_a = pd.Series(np.arange(len(calendar_10), dtype=float), index=calendar_10)
        prices_a.iloc[3:5] = np.nan
        _, report_a = cal.align_to_calendar(prices_a, calendar_10)

        prices_b = pd.Series(np.arange(len(calendar_10), dtype=float), index=calendar_10)
        prices_b.iloc[6:8] = np.nan
        _, report_b = cal.align_to_calendar(prices_b, calendar_10)

        dropped = cal.universe_dropped_dates({"A": report_a, "B": report_b})
        assert len(dropped) == 4
        expected = sorted(list(calendar_10[3:5]) + list(calendar_10[6:8]))
        assert list(dropped) == expected

    def test_overlapping_gaps_deduplicated(self, calendar_10: pd.DatetimeIndex) -> None:
        prices = pd.Series(np.arange(len(calendar_10), dtype=float), index=calendar_10)
        prices.iloc[3:5] = np.nan
        _, report = cal.align_to_calendar(prices, calendar_10)

        # Two instruments with the SAME multi-day gap
        dropped = cal.universe_dropped_dates({"A": report, "B": report})
        assert len(dropped) == 2

    def test_no_gaps_returns_empty(self, calendar_10: pd.DatetimeIndex) -> None:
        prices = pd.Series(np.arange(len(calendar_10), dtype=float), index=calendar_10)
        _, report = cal.align_to_calendar(prices, calendar_10)
        dropped = cal.universe_dropped_dates({"X": report})
        assert len(dropped) == 0

    def test_empty_input(self) -> None:
        dropped = cal.universe_dropped_dates({})
        assert len(dropped) == 0

    def test_result_sorted(self, calendar_10: pd.DatetimeIndex) -> None:
        prices_a = pd.Series(np.arange(len(calendar_10), dtype=float), index=calendar_10)
        prices_a.iloc[7:9] = np.nan
        _, report_a = cal.align_to_calendar(prices_a, calendar_10)

        prices_b = pd.Series(np.arange(len(calendar_10), dtype=float), index=calendar_10)
        prices_b.iloc[3:5] = np.nan
        _, report_b = cal.align_to_calendar(prices_b, calendar_10)

        dropped = cal.universe_dropped_dates({"A": report_a, "B": report_b})
        assert dropped.is_monotonic_increasing
