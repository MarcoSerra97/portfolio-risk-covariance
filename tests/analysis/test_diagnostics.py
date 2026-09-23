"""Tests for the diagnostics helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless testing

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from portfolio_risk.analysis import diagnostics as diag
from portfolio_risk.data import calendar as cal
from portfolio_risk.data import universe


@pytest.fixture
def calendar_long() -> pd.DatetimeIndex:
    """~200 trading days: enough for rolling 63-day windows."""
    return cal.nyse_trading_days("2019-01-02", "2019-10-15")


@pytest.fixture
def synthetic_prices(calendar_long: pd.DatetimeIndex) -> pd.DataFrame:
    """A tiny 3-column price panel with known properties for testing."""
    rng = np.random.default_rng(0)
    n = len(calendar_long)
    return pd.DataFrame(
        {
            "AAPL": 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
            "MSFT": 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
            "JPM": 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
        },
        index=calendar_long,
    )


@pytest.fixture
def synthetic_returns(synthetic_prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(synthetic_prices / synthetic_prices.shift(1)).dropna(how="all")


class TestComputeCoverageTable:
    def test_metadata_columns_present(
        self, synthetic_prices: pd.DataFrame, calendar_long: pd.DatetimeIndex
    ) -> None:
        table = diag.compute_coverage_table(synthetic_prices, calendar_long)
        for col in ("name", "bucket", "subgroup", "inception", "notes"):
            assert col in table.columns

    def test_coverage_columns_present(
        self, synthetic_prices: pd.DataFrame, calendar_long: pd.DatetimeIndex
    ) -> None:
        table = diag.compute_coverage_table(synthetic_prices, calendar_long)
        for col in ("first_obs", "last_obs", "n_obs", "expected_obs", "pct_missing"):
            assert col in table.columns

    def test_missing_ticker_gets_nan(
        self, synthetic_prices: pd.DataFrame, calendar_long: pd.DatetimeIndex
    ) -> None:
        """A universe ticker that's not in prices gets a NaT/zero row."""
        table = diag.compute_coverage_table(synthetic_prices, calendar_long)
        # 'BIL' is in the universe but not in the tiny prices panel
        assert table.at["BIL", "n_obs"] == 0
        assert pd.isna(table.at["BIL", "first_obs"])

    def test_indexed_by_ticker_in_universe_order(
        self, synthetic_prices: pd.DataFrame, calendar_long: pd.DatetimeIndex
    ) -> None:
        table = diag.compute_coverage_table(synthetic_prices, calendar_long)
        assert list(table.index) == universe.all_tickers()

    def test_row_count_matches_universe(
        self, synthetic_prices: pd.DataFrame, calendar_long: pd.DatetimeIndex
    ) -> None:
        table = diag.compute_coverage_table(synthetic_prices, calendar_long)
        assert len(table) == 75

    def test_pct_missing_zero_when_full_coverage(
        self, synthetic_prices: pd.DataFrame, calendar_long: pd.DatetimeIndex
    ) -> None:
        table = diag.compute_coverage_table(synthetic_prices, calendar_long)
        assert table.at["AAPL", "pct_missing"] == pytest.approx(0.0)
        assert table.at["AAPL", "n_obs"] == len(calendar_long)


class TestComputeRollingCoverage:
    def test_all_live(self, synthetic_prices: pd.DataFrame) -> None:
        result = diag.compute_rolling_coverage(synthetic_prices)
        assert (result == 3).all()
        assert result.name == "n_live"

    def test_pre_inception_not_counted(self, synthetic_prices: pd.DataFrame) -> None:
        synthetic_prices.iloc[:10, 0] = np.nan  # AAPL missing first 10 days
        result = diag.compute_rolling_coverage(synthetic_prices)
        assert (result.iloc[:10] == 2).all()
        assert (result.iloc[10:] == 3).all()


class TestComputeReturnStats:
    def test_basic_shape(self, synthetic_returns: pd.DataFrame) -> None:
        stats = diag.compute_return_stats(synthetic_returns)
        assert list(stats.index) == list(synthetic_returns.columns)
        for col in (
            "mean_daily",
            "mean_annual",
            "std_daily",
            "std_annual",
            "skew",
            "excess_kurtosis",
            "min",
            "max",
            "adf_stat",
            "adf_pvalue",
            "n_obs",
        ):
            assert col in stats.columns

    def test_annualisation(self, synthetic_returns: pd.DataFrame) -> None:
        stats = diag.compute_return_stats(synthetic_returns, trading_days_per_year=252)
        assert stats["mean_annual"].iloc[0] == pytest.approx(stats["mean_daily"].iloc[0] * 252)
        assert stats["std_annual"].iloc[0] == pytest.approx(
            stats["std_daily"].iloc[0] * np.sqrt(252)
        )

    def test_adf_rejects_random_walk_returns(self, synthetic_returns: pd.DataFrame) -> None:
        """Log-returns of a Gaussian random walk should be stationary."""
        stats = diag.compute_return_stats(synthetic_returns)
        # Not asserting a specific threshold; just checking ADF produced numbers
        assert not np.isnan(stats["adf_stat"]).any()
        assert not np.isnan(stats["adf_pvalue"]).any()
        # And that all p-values indicate stationarity (< 0.05)
        assert (stats["adf_pvalue"] < 0.05).all()

    def test_empty_series_handled(self, calendar_long: pd.DatetimeIndex) -> None:
        empty = pd.DataFrame({"X": [np.nan] * len(calendar_long)}, index=calendar_long)
        stats = diag.compute_return_stats(empty)
        assert stats.at["X", "n_obs"] == 0
        assert np.isnan(stats.at["X", "mean_daily"])


class TestComputeFullCorrelation:
    def test_shape(self, synthetic_returns: pd.DataFrame) -> None:
        corr = diag.compute_full_correlation(synthetic_returns)
        assert corr.shape == (3, 3)

    def test_diagonal_is_one(self, synthetic_returns: pd.DataFrame) -> None:
        corr = diag.compute_full_correlation(synthetic_returns)
        assert np.allclose(np.diag(corr.to_numpy()), 1.0)

    def test_symmetric(self, synthetic_returns: pd.DataFrame) -> None:
        corr = diag.compute_full_correlation(synthetic_returns)
        assert np.allclose(corr.to_numpy(), corr.to_numpy().T)

    def test_ordering_follows_universe(self, synthetic_returns: pd.DataFrame) -> None:
        """AAPL and MSFT (both equity Tech) should appear before JPM (Financials)."""
        corr = diag.compute_full_correlation(synthetic_returns)
        cols = list(corr.columns)
        assert cols.index("AAPL") < cols.index("JPM")
        assert cols.index("MSFT") < cols.index("JPM")


class TestComputeRollingPairwiseCorrelation:
    def test_first_window_minus_one_are_nan(self, synthetic_returns: pd.DataFrame) -> None:
        rolling = diag.compute_rolling_pairwise_correlation(synthetic_returns, window=20)
        assert rolling.iloc[:19].isna().all()
        assert not rolling.iloc[19:].isna().any()

    def test_index_matches_input(self, synthetic_returns: pd.DataFrame) -> None:
        rolling = diag.compute_rolling_pairwise_correlation(synthetic_returns, window=20)
        pd.testing.assert_index_equal(rolling.index, synthetic_returns.index)

    def test_values_bounded_by_one(self, synthetic_returns: pd.DataFrame) -> None:
        rolling = diag.compute_rolling_pairwise_correlation(synthetic_returns, window=20)
        valid = rolling.dropna()
        assert (valid.abs() <= 1.0).all()

    def test_raises_on_bad_window(self, synthetic_returns: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="window"):
            diag.compute_rolling_pairwise_correlation(synthetic_returns, window=1)
        with pytest.raises(ValueError, match="larger than"):
            diag.compute_rolling_pairwise_correlation(synthetic_returns, window=10_000)


class TestBucketBoundaryPositions:
    def test_returns_boundaries_between_buckets(self) -> None:
        # Take first few equities + first srf_etf: should show one boundary
        tickers = ["AAPL", "MSFT", "XLK"]
        positions = diag.bucket_boundary_positions(tickers)
        assert positions == [2]

    def test_empty_input(self) -> None:
        assert diag.bucket_boundary_positions([]) == []

    def test_single_bucket_no_boundaries(self) -> None:
        assert diag.bucket_boundary_positions(["AAPL", "MSFT", "JPM"]) == []


class TestPlots:
    def test_rolling_coverage_returns_figure(self, synthetic_prices: pd.DataFrame) -> None:
        coverage = diag.compute_rolling_coverage(synthetic_prices)
        fig = diag.plot_rolling_coverage(coverage, target=3)
        assert isinstance(fig, Figure)

    def test_correlation_heatmap_returns_figure(self, synthetic_returns: pd.DataFrame) -> None:
        corr = diag.compute_full_correlation(synthetic_returns)
        fig = diag.plot_correlation_heatmap(corr)
        assert isinstance(fig, Figure)

    def test_correlation_heatmap_with_boundaries(self, synthetic_returns: pd.DataFrame) -> None:
        corr = diag.compute_full_correlation(synthetic_returns)
        fig = diag.plot_correlation_heatmap(corr, bucket_boundaries=[2])
        assert isinstance(fig, Figure)

    def test_rolling_correlation_returns_figure(self, synthetic_returns: pd.DataFrame) -> None:
        rolling = diag.compute_rolling_pairwise_correlation(synthetic_returns, window=20)
        fig = diag.plot_rolling_correlation(rolling, window=20)
        assert isinstance(fig, Figure)
