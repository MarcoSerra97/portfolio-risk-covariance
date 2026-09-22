"""Tests for the cleaning module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.data import calendar as cal
from portfolio_risk.data.cleaning import (
    DEFAULT_OUTLIER_WINDOW,
    DEFAULT_OUTLIER_Z_THRESHOLD,
    CleaningResult,
    clean_prices,
    flag_return_outliers,
)


@pytest.fixture
def calendar_10() -> pd.DatetimeIndex:
    return cal.nyse_trading_days("2019-01-02", "2019-01-15")


@pytest.fixture
def calendar_long() -> pd.DatetimeIndex:
    """~200 trading days: enough to exercise the 63-day outlier window."""
    return cal.nyse_trading_days("2019-01-02", "2019-10-15")


class TestCleanPrices:
    def test_returns_cleaning_result(self, calendar_10: pd.DatetimeIndex) -> None:
        raw = pd.DataFrame(
            {"A": np.arange(1, len(calendar_10) + 1, dtype=float)},
            index=calendar_10,
        )
        result = clean_prices(raw, calendar_10)
        assert isinstance(result, CleaningResult)
        assert set(result.gap_reports.keys()) == {"A"}

    def test_complete_data_unchanged(self, calendar_10: pd.DatetimeIndex) -> None:
        raw = pd.DataFrame(
            {"A": np.arange(1, len(calendar_10) + 1, dtype=float)},
            index=calendar_10,
        )
        result = clean_prices(raw, calendar_10)
        pd.testing.assert_frame_equal(result.prices, raw)
        assert len(result.dropped_dates) == 0

    def test_isolated_gap_ffilled(self, calendar_10: pd.DatetimeIndex) -> None:
        raw = pd.DataFrame(
            {"A": np.arange(1, len(calendar_10) + 1, dtype=float)},
            index=calendar_10,
        )
        raw.iloc[3, 0] = np.nan
        result = clean_prices(raw, calendar_10)
        assert result.prices.iloc[3, 0] == raw.iloc[2, 0]
        assert len(result.dropped_dates) == 0

    def test_multi_day_gap_drops_dates(self, calendar_10: pd.DatetimeIndex) -> None:
        raw = pd.DataFrame(
            {"A": np.arange(1, len(calendar_10) + 1, dtype=float)},
            index=calendar_10,
        )
        raw.iloc[3:6, 0] = np.nan
        result = clean_prices(raw, calendar_10)
        assert len(result.dropped_dates) == 3
        assert len(result.prices) == len(calendar_10) - 3
        # Dropped dates should not appear in the cleaned panel
        for dt in result.dropped_dates:
            assert dt not in result.prices.index

    def test_universe_wide_drop_across_instruments(self, calendar_10: pd.DatetimeIndex) -> None:
        """A multi-day gap in ONE instrument drops those dates for ALL."""
        raw = pd.DataFrame(
            {
                "A": np.arange(1, len(calendar_10) + 1, dtype=float),
                "B": np.arange(1, len(calendar_10) + 1, dtype=float) * 10,
            },
            index=calendar_10,
        )
        raw.iloc[3:5, 0] = np.nan  # A has a 2-day gap; B is complete
        result = clean_prices(raw, calendar_10)
        assert len(result.dropped_dates) == 2
        # Both columns should have those dates dropped
        assert len(result.prices) == len(calendar_10) - 2
        assert result.prices["B"].notna().all()

    def test_result_index_is_calendar_minus_drops(self, calendar_10: pd.DatetimeIndex) -> None:
        raw = pd.DataFrame(
            {"A": np.arange(1, len(calendar_10) + 1, dtype=float)},
            index=calendar_10,
        )
        raw.iloc[3:5, 0] = np.nan
        result = clean_prices(raw, calendar_10)
        expected = calendar_10.drop(calendar_10[3:5])
        pd.testing.assert_index_equal(result.prices.index, expected)

    def test_pre_inception_preserved(self, calendar_10: pd.DatetimeIndex) -> None:
        raw = pd.DataFrame({"A": [np.nan] * len(calendar_10)}, index=calendar_10)
        raw.iloc[5:, 0] = np.arange(1, len(calendar_10) - 4, dtype=float)
        result = clean_prices(raw, calendar_10)
        assert result.prices.iloc[:5, 0].isna().all()
        assert result.prices.iloc[5:, 0].notna().all()
        assert len(result.dropped_dates) == 0

    def test_columns_preserved(self, calendar_10: pd.DatetimeIndex) -> None:
        raw = pd.DataFrame(
            {
                "AAPL": np.arange(1, len(calendar_10) + 1, dtype=float),
                "MSFT": np.arange(1, len(calendar_10) + 1, dtype=float) * 2,
                "GOOG": np.arange(1, len(calendar_10) + 1, dtype=float) * 3,
            },
            index=calendar_10,
        )
        result = clean_prices(raw, calendar_10)
        assert list(result.prices.columns) == ["AAPL", "MSFT", "GOOG"]


class TestFlagReturnOutliers:
    @pytest.fixture
    def stable_returns(self, calendar_long: pd.DatetimeIndex) -> pd.DataFrame:
        """A well-behaved return panel: N(0, 0.01) with fixed seed."""
        rng = np.random.default_rng(42)
        return pd.DataFrame(
            {
                "A": rng.normal(0.0, 0.01, len(calendar_long)),
                "B": rng.normal(0.0, 0.01, len(calendar_long)),
            },
            index=calendar_long,
        )

    def test_returns_empty_df_when_no_outliers(self, stable_returns: pd.DataFrame) -> None:
        result = flag_return_outliers(stable_returns)
        assert result.empty
        assert list(result.columns) == [
            "ticker",
            "date",
            "return",
            "median",
            "mad",
            "z_score",
        ]

    def test_flags_extreme_positive_outlier(
        self, stable_returns: pd.DataFrame, calendar_long: pd.DatetimeIndex
    ) -> None:
        # Inject a clear outlier well past the burn-in window
        stable_returns.iloc[100, 0] = 0.5  # 50% return = ~50 sigma
        result = flag_return_outliers(stable_returns)
        assert len(result) == 1
        assert result.iloc[0]["ticker"] == "A"
        assert result.iloc[0]["date"] == calendar_long[100]
        assert result.iloc[0]["return"] == pytest.approx(0.5)
        assert result.iloc[0]["z_score"] > DEFAULT_OUTLIER_Z_THRESHOLD

    def test_flags_extreme_negative_outlier(
        self, stable_returns: pd.DataFrame, calendar_long: pd.DatetimeIndex
    ) -> None:
        stable_returns.iloc[100, 1] = -0.5
        result = flag_return_outliers(stable_returns)
        assert len(result) == 1
        assert result.iloc[0]["ticker"] == "B"
        assert result.iloc[0]["z_score"] < -DEFAULT_OUTLIER_Z_THRESHOLD

    def test_does_not_flag_sub_threshold(self, stable_returns: pd.DataFrame) -> None:
        # ~3 sigma move: well below threshold of 8
        stable_returns.iloc[100, 0] = 0.03
        result = flag_return_outliers(stable_returns)
        assert result.empty

    def test_burn_in_period_not_flagged(self, stable_returns: pd.DataFrame) -> None:
        """The first `window - 1` days have insufficient history to flag."""
        # Put a huge outlier in the burn-in period
        stable_returns.iloc[10, 0] = 1.0
        result = flag_return_outliers(stable_returns)
        # Should not be flagged because rolling window is 63
        assert result.empty

    def test_multiple_outliers_sorted_by_date(
        self, stable_returns: pd.DataFrame, calendar_long: pd.DatetimeIndex
    ) -> None:
        stable_returns.iloc[150, 0] = 0.5
        stable_returns.iloc[100, 1] = -0.5
        result = flag_return_outliers(stable_returns)
        assert len(result) == 2
        assert result.iloc[0]["date"] == calendar_long[100]
        assert result.iloc[1]["date"] == calendar_long[150]

    def test_column_schema(self, stable_returns: pd.DataFrame) -> None:
        stable_returns.iloc[100, 0] = 0.5
        result = flag_return_outliers(stable_returns)
        assert list(result.columns) == [
            "ticker",
            "date",
            "return",
            "median",
            "mad",
            "z_score",
        ]

    def test_custom_threshold(self, stable_returns: pd.DataFrame) -> None:
        # A 3-sigma move (~0.03 return) — below 8, above 2
        stable_returns.iloc[100, 0] = 0.05
        strict = flag_return_outliers(stable_returns, z_threshold=2.0)
        lenient = flag_return_outliers(stable_returns, z_threshold=8.0)
        assert not strict.empty
        assert lenient.empty

    def test_custom_window(self, calendar_long: pd.DatetimeIndex) -> None:
        rng = np.random.default_rng(0)
        returns = pd.DataFrame(
            {"A": rng.normal(0.0, 0.01, len(calendar_long))},
            index=calendar_long,
        )
        returns.iloc[30, 0] = 0.5  # inside the burn-in for window=63, outside for window=20
        result_63 = flag_return_outliers(returns, window=63)
        result_20 = flag_return_outliers(returns, window=20)
        assert result_63.empty
        assert not result_20.empty

    def test_raises_on_invalid_window(self, stable_returns: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="window"):
            flag_return_outliers(stable_returns, window=1)

    def test_raises_on_invalid_threshold(self, stable_returns: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="z_threshold"):
            flag_return_outliers(stable_returns, z_threshold=0)


class TestDefaults:
    def test_window_default_is_63(self) -> None:
        assert DEFAULT_OUTLIER_WINDOW == 63

    def test_z_threshold_default_is_8(self) -> None:
        assert DEFAULT_OUTLIER_Z_THRESHOLD == 8.0
