"""Tests for the backtest calendar, pinned to docs/02-evaluation-design.md."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.data.calendar import nyse_trading_days
from portfolio_risk.evaluation.schedule import (
    UNCLASSIFIED,
    Regime,
    Schedule,
    assign_regime,
    build_schedule,
    eligible_assets,
)


@pytest.fixture(scope="module")
def study_index() -> pd.DatetimeIndex:
    """The study's trading days, 2010-01-04 to 2025-12-31."""
    return nyse_trading_days(date(2010, 1, 4), date(2025, 12, 31))


@pytest.fixture(scope="module")
def schedule(study_index: pd.DatetimeIndex) -> Schedule:
    return build_schedule(study_index)


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


class TestStudyCalendar:
    """The numbers recorded in the design document."""

    def test_index_length(self, study_index: pd.DatetimeIndex) -> None:
        assert len(study_index) == 4024

    def test_131_periods(self, schedule: Schedule) -> None:
        assert len(schedule) == 131

    def test_first_period(self, schedule: Schedule) -> None:
        p = schedule.periods[0]
        assert p.rebalance == _ts("2014-12-31")
        assert (p.start, p.end) == (_ts("2015-01-02"), _ts("2015-02-02"))

    def test_last_period_and_dropped_tail(
        self, schedule: Schedule, study_index: pd.DatetimeIndex
    ) -> None:
        last = schedule.periods[-1]
        assert last.end == _ts("2025-12-09")
        assert len(study_index) - 1 - last.end_pos == 15

    def test_tariff_crash_period(self, schedule: Schedule) -> None:
        p = schedule.periods[122]
        assert (p.start, p.end) == (_ts("2025-03-12"), _ts("2025-04-09"))
        assert p.regime == "Tariff shock"

    def test_regime_counts_and_ranges(self, schedule: Schedule) -> None:
        frame = schedule.to_frame()
        counts = frame["regime"].value_counts().to_dict()
        assert counts == {
            "Calm I": 60,
            "Calm II": 26,
            "Post-COVID": 19,
            "Rates shock": 18,
            "COVID shock": 4,
            "Tariff shock": 3,
            UNCLASSIFIED: 1,
        }
        ks = {r: list(frame.index[frame["regime"] == r]) for r in counts}
        assert ks["Calm I"] == list(range(0, 60))
        assert ks["COVID shock"] == [61, 62, 63, 64]
        assert ks["Tariff shock"] == [122, 123, 124]
        assert ks[UNCLASSIFIED] == [60]

    def test_window_starts_at_first_rebalance(self, schedule: Schedule) -> None:
        p0 = schedule.periods[0]
        s252, e252 = schedule.rolling_window(p0, 252)
        s504, _ = schedule.rolling_window(p0, 504)
        assert schedule.index[s252] == _ts("2014-01-02")
        assert schedule.index[s504] == _ts("2013-01-02")
        assert schedule.index[e252] == p0.rebalance

    def test_dcc_expanding_window(self, schedule: Schedule) -> None:
        p0 = schedule.periods[0]
        start, end = schedule.expanding_window(p0, date(2013, 4, 18))
        assert schedule.index[start] == _ts("2013-04-18")
        assert end - start + 1 == 431


class TestStructure:
    def test_periods_contiguous_and_full_length(self, schedule: Schedule) -> None:
        for prev, cur in zip(schedule.periods, schedule.periods[1:], strict=False):
            assert cur.start_pos == prev.end_pos + 1
            assert cur.rebalance_pos == prev.end_pos
        assert all(p.end_pos - p.start_pos + 1 == 21 for p in schedule.periods)

    def test_no_look_ahead(self, schedule: Schedule) -> None:
        for p in schedule.periods:
            _, end = schedule.rolling_window(p, 252)
            assert end == p.rebalance_pos < p.start_pos

    def test_rolling_windows_slide_by_holding(self, schedule: Schedule) -> None:
        a = schedule.rolling_window(schedule.periods[10], 252)
        b = schedule.rolling_window(schedule.periods[11], 252)
        assert (b[0] - a[0], b[1] - a[1]) == (21, 21)

    def test_first_rebalance_on_non_trading_day(self, study_index: pd.DatetimeIndex) -> None:
        """A weekend date resolves to the preceding trading day."""
        s = build_schedule(study_index, first_rebalance=date(2015, 1, 3))  # Saturday
        assert s.periods[0].rebalance == _ts("2015-01-02")

    def test_custom_holding(self, study_index: pd.DatetimeIndex) -> None:
        s = build_schedule(study_index, holding=63)
        assert all(p.end_pos - p.start_pos + 1 == 63 for p in s.periods)
        assert len(s) == 2766 // 63


class TestErrors:
    def test_insufficient_history(self, study_index: pd.DatetimeIndex) -> None:
        s = build_schedule(study_index[1000:])
        with pytest.raises(ValueError, match="history"):
            s.rolling_window(s.periods[0], 504)

    def test_expanding_start_after_rebalance(self, schedule: Schedule) -> None:
        with pytest.raises(ValueError, match="after the rebalance"):
            schedule.expanding_window(schedule.periods[0], date(2016, 1, 1))

    def test_bad_holding(self, study_index: pd.DatetimeIndex) -> None:
        with pytest.raises(ValueError, match="holding"):
            build_schedule(study_index, holding=0)

    def test_first_rebalance_before_index(self, study_index: pd.DatetimeIndex) -> None:
        with pytest.raises(ValueError, match="before the start"):
            build_schedule(study_index, first_rebalance=date(2009, 1, 1))

    def test_unsorted_index(self, study_index: pd.DatetimeIndex) -> None:
        with pytest.raises(ValueError, match="sorted"):
            build_schedule(study_index[::-1])


class TestAssignRegime:
    def test_majority_rule(self) -> None:
        regimes = [Regime("A", date(2020, 1, 1), date(2020, 1, 10))]
        days = pd.bdate_range("2020-01-06", periods=5)  # 6..10 Jan: all in A
        assert assign_regime(days, regimes) == "A"
        days = pd.bdate_range("2020-01-09", periods=5)  # 2 of 5 in A
        assert assign_regime(days, regimes) == UNCLASSIFIED

    def test_split_regime_counts_together(self) -> None:
        regimes = [
            Regime("A", date(2020, 1, 1), date(2020, 1, 7)),
            Regime("B", date(2020, 1, 8), date(2020, 1, 8)),
            Regime("A", date(2020, 1, 9), date(2020, 1, 31)),
        ]
        days = pd.bdate_range("2020-01-06", periods=5)  # 6,7 A; 8 B; 9,10 A
        assert assign_regime(days, regimes) == "A"


class TestEligibleAssets:
    def test_mtum_enters_504_run_at_period_4(self, schedule: Schedule) -> None:
        idx = schedule.index
        returns = pd.DataFrame(np.ones((len(idx), 2)), index=idx, columns=["SPY", "MTUM"])
        returns.loc[returns.index < _ts("2013-04-18"), "MTUM"] = np.nan
        first_full = next(
            p.k
            for p in schedule.periods
            if "MTUM" in eligible_assets(returns, *schedule.rolling_window(p, 504))
        )
        assert first_full == 4
        assert schedule.periods[4].rebalance == _ts("2015-05-04")
        for p in schedule.periods:
            assert "MTUM" in eligible_assets(returns, *schedule.rolling_window(p, 252))
