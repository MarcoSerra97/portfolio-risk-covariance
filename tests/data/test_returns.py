"""Tests for the returns module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.data import calendar as cal
from portfolio_risk.data.returns import log_returns


@pytest.fixture
def calendar_10() -> pd.DatetimeIndex:
    return cal.nyse_trading_days("2019-01-02", "2019-01-15")


class TestLogReturns:
    def test_basic_shape_preserved(self, calendar_10: pd.DatetimeIndex) -> None:
        prices = pd.DataFrame(
            {"A": np.arange(1, len(calendar_10) + 1, dtype=float)},
            index=calendar_10,
        )
        r = log_returns(prices)
        assert r.shape == prices.shape
        assert list(r.columns) == list(prices.columns)
        assert list(r.index) == list(prices.index)

    def test_first_row_all_nan(self, calendar_10: pd.DatetimeIndex) -> None:
        prices = pd.DataFrame(
            {"A": np.arange(1, len(calendar_10) + 1, dtype=float)},
            index=calendar_10,
        )
        r = log_returns(prices)
        assert r.iloc[0].isna().all()

    def test_known_value(self, calendar_10: pd.DatetimeIndex) -> None:
        prices = pd.DataFrame(
            {"A": [100.0] * len(calendar_10)},
            index=calendar_10,
        )
        prices.iloc[1, 0] = 110.0
        r = log_returns(prices)
        assert r.iloc[1, 0] == pytest.approx(np.log(1.10))

    def test_flat_series_gives_zero_returns(self, calendar_10: pd.DatetimeIndex) -> None:
        prices = pd.DataFrame({"A": [100.0] * len(calendar_10)}, index=calendar_10)
        r = log_returns(prices)
        assert r.iloc[1:, 0].tolist() == pytest.approx([0.0] * (len(calendar_10) - 1))

    def test_multiple_columns_independent(self, calendar_10: pd.DatetimeIndex) -> None:
        prices = pd.DataFrame(
            {
                "A": np.arange(1, len(calendar_10) + 1, dtype=float),
                "B": np.arange(2, len(calendar_10) + 2, dtype=float) * 10.0,
            },
            index=calendar_10,
        )
        r = log_returns(prices)
        # Each column's return only depends on its own history.
        assert r["A"].iloc[1] == pytest.approx(np.log(2 / 1))
        assert r["B"].iloc[1] == pytest.approx(np.log(30 / 20))

    def test_nan_gap_propagates_to_two_returns(self, calendar_10: pd.DatetimeIndex) -> None:
        """A NaN price at t makes r_t and r_{t+1} both NaN."""
        prices = pd.DataFrame(
            {"A": [100.0] * len(calendar_10)},
            index=calendar_10,
        )
        prices.iloc[3, 0] = np.nan
        r = log_returns(prices)
        assert np.isnan(r.iloc[3, 0])  # P_t is NaN
        assert np.isnan(r.iloc[4, 0])  # P_{t-1} is NaN
        assert not np.isnan(r.iloc[5, 0])  # both present again

    def test_pre_inception_nan_preserved(self, calendar_10: pd.DatetimeIndex) -> None:
        prices = pd.DataFrame({"A": [np.nan] * len(calendar_10)}, index=calendar_10)
        prices.iloc[3:, 0] = np.arange(1, len(calendar_10) - 2, dtype=float)
        r = log_returns(prices)
        assert r.iloc[:3, 0].isna().all()
        # First return after inception is NaN (P_{t-1} is NaN)
        assert np.isnan(r.iloc[3, 0])
        # Second return after inception is defined
        assert not np.isnan(r.iloc[4, 0])

    def test_raises_on_non_datetime_index(self) -> None:
        prices = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=[0, 1, 2])
        with pytest.raises(TypeError, match="DatetimeIndex"):
            log_returns(prices)

    def test_raises_on_zero_price(self, calendar_10: pd.DatetimeIndex) -> None:
        prices = pd.DataFrame({"A": [100.0] * len(calendar_10)}, index=calendar_10)
        prices.iloc[3, 0] = 0.0
        with pytest.raises(ValueError, match="Non-positive"):
            log_returns(prices)

    def test_raises_on_negative_price(self, calendar_10: pd.DatetimeIndex) -> None:
        prices = pd.DataFrame({"A": [100.0] * len(calendar_10)}, index=calendar_10)
        prices.iloc[3, 0] = -1.0
        with pytest.raises(ValueError, match="Non-positive"):
            log_returns(prices)

    def test_allows_nan_alongside_positive(self, calendar_10: pd.DatetimeIndex) -> None:
        """NaN by itself shouldn't trigger the non-positive check."""
        prices = pd.DataFrame({"A": [100.0] * len(calendar_10)}, index=calendar_10)
        prices.iloc[3, 0] = np.nan
        # Should not raise
        _ = log_returns(prices)
