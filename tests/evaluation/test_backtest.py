"""Tests for the backtest engine."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.data.calendar import nyse_trading_days
from portfolio_risk.estimators.sample import SampleCovariance
from portfolio_risk.evaluation import backtest as bt
from portfolio_risk.evaluation.config import ESTIMATORS, EstimatorSpec, RunConfig
from portfolio_risk.evaluation.schedule import build_schedule

LATE_START = pd.Timestamp("2014-06-02")


def _returns(seed: int = 0) -> pd.DataFrame:
    """Synthetic panel 2012–2015: four factor-correlated assets, a cash-like BIL,
    and LATE, which only trades from 2014-06-02."""
    idx = nyse_trading_days(date(2012, 1, 3), date(2015, 12, 31))
    rng = np.random.default_rng(seed)
    n = len(idx)
    f = rng.standard_normal(n)
    data = {
        name: 0.01 * (0.6 * f + 0.8 * rng.standard_normal(n)) * scale
        for name, scale in [("A", 1.0), ("B", 1.5), ("C", 0.8), ("D", 2.0)]
    }
    data["BIL"] = 0.0002 * rng.standard_normal(n)
    data["LATE"] = 0.01 * (0.5 * f + rng.standard_normal(n))
    df = pd.DataFrame(data, index=idx)
    df.loc[df.index < LATE_START, "LATE"] = np.nan
    return df


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    return _returns()


@pytest.fixture(scope="module")
def config() -> RunConfig:
    return RunConfig(estimators=("sample", "lw_std", "qis_std"))


@pytest.fixture(scope="module")
def result(returns: pd.DataFrame, config: RunConfig) -> bt.BacktestResult:
    return bt.run_backtest(returns, config)


class _Spy:
    """Sample covariance that records the window it is fitted on."""

    calls: list[tuple[pd.Timestamp, pd.Timestamp, int, tuple[str, ...]]] = []

    def fit(self, returns: pd.DataFrame) -> Self:
        _Spy.calls.append(
            (returns.index[0], returns.index[-1], len(returns), tuple(returns.columns))
        )
        self.covariance_ = SampleCovariance().fit(returns).covariance_
        return self


class TestShapes:
    def test_forecast_rows(self, result: bt.BacktestResult, returns: pd.DataFrame) -> None:
        n_periods = len(build_schedule(returns.index))
        assert len(result.forecasts) == 3 * 4 * n_periods
        assert list(result.forecasts.columns) == bt.FORECAST_COLUMNS

    def test_daily_returns_cover_holding_periods(
        self, result: bt.BacktestResult, returns: pd.DataFrame
    ) -> None:
        s = build_schedule(returns.index)
        expected = returns.index[s.periods[0].start_pos : s.periods[-1].end_pos + 1]
        pd.testing.assert_index_equal(result.daily_returns.index, expected)
        assert result.daily_returns.shape[1] == 3 * 4
        assert result.period_of_day.index.equals(expected)

    def test_weights_sum_to_one(self, result: bt.BacktestResult) -> None:
        np.testing.assert_allclose(result.weights.sum(axis=1), 1.0, atol=1e-10)


class TestValues:
    def test_forecast_matches_direct_computation(
        self, result: bt.BacktestResult, returns: pd.DataFrame, config: RunConfig
    ) -> None:
        s = build_schedule(returns.index)
        p = s.periods[5]
        assets = bt.run_asset_set(returns, s, p, config.window_length)
        start, end = s.rolling_window(p, config.window_length)
        cov = SampleCovariance().fit(returns.iloc[start : end + 1][assets]).covariance_
        w = result.weights.loc[("sample", "min_variance_long_short", 5)][assets].to_numpy()
        row = result.forecasts.query(
            "estimator == 'sample' and portfolio == 'min_variance_long_short' and k == 5"
        ).iloc[0]
        assert row["forecast_var"] == pytest.approx(21 * w @ cov.to_numpy() @ w)

    def test_realised_and_daily_match_direct_computation(
        self, result: bt.BacktestResult, returns: pd.DataFrame
    ) -> None:
        s = build_schedule(returns.index)
        p = s.periods[3]
        w = result.weights.loc[("lw_std", "risk_parity", 3)]
        hold = returns.iloc[p.start_pos : p.end_pos + 1].fillna(0.0)
        daily = hold.to_numpy() @ w.reindex(hold.columns).to_numpy()
        np.testing.assert_allclose(
            result.daily_returns[("lw_std", "risk_parity")].loc[hold.index], daily
        )
        row = result.forecasts.query(
            "estimator == 'lw_std' and portfolio == 'risk_parity' and k == 3"
        ).iloc[0]
        assert row["realised_var"] == pytest.approx(float((daily**2).sum()))

    def test_equal_weight_identical_across_estimators(self, result: bt.BacktestResult) -> None:
        ew = result.weights.xs("equal_weight", level="portfolio")
        for k in (0, 7):
            rows = ew.xs(k, level="k")
            assert np.allclose(rows.to_numpy(), rows.iloc[0].to_numpy())

    def test_cash_excluded_from_optimised_portfolios(self, result: bt.BacktestResult) -> None:
        for port in ("risk_parity", "min_variance_long_only", "min_variance_long_short"):
            assert (result.weights.xs(port, level="portfolio")["BIL"] == 0).all()
        assert (result.weights.xs("equal_weight", level="portfolio")["BIL"] > 0).all()

    def test_late_asset_enters_when_window_is_complete(
        self, result: bt.BacktestResult, returns: pd.DataFrame
    ) -> None:
        s = build_schedule(returns.index)
        ew = result.weights.xs(("sample", "equal_weight"), level=("estimator", "portfolio"))
        for p in s.periods:
            start, _ = s.rolling_window(p, 252)
            complete = s.index[start] >= LATE_START
            assert (ew.loc[p.k, "LATE"] > 0) == complete
        assert (ew["LATE"] == 0).any() and (ew["LATE"] > 0).any()


class TestNoLookAhead:
    def test_holding_period_returns_do_not_affect_that_period(self, returns: pd.DataFrame) -> None:
        cfg = RunConfig(estimators=("sample",))
        base = bt.run_backtest(returns, cfg)
        s = build_schedule(returns.index)
        p = s.periods[3]
        shocked = returns.copy()
        shocked.iloc[p.start_pos : p.end_pos + 1] *= 5.0
        alt = bt.run_backtest(shocked, cfg)
        early = base.forecasts["k"] <= 3
        np.testing.assert_allclose(
            alt.forecasts.loc[early, "forecast_var"], base.forecasts.loc[early, "forecast_var"]
        )
        pd.testing.assert_frame_equal(
            alt.weights.xs(slice(0, 3), level="k", drop_level=False),
            base.weights.xs(slice(0, 3), level="k", drop_level=False),
        )
        k3 = base.forecasts["k"] == 3
        assert not np.allclose(
            alt.forecasts.loc[k3, "realised_var"], base.forecasts.loc[k3, "realised_var"]
        )


class TestWindows:
    def test_rolling_and_expanding_windows(
        self, returns: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(
            ESTIMATORS, "spy_roll", EstimatorSpec("spy_roll", lambda h: _Spy(), "rolling")
        )
        monkeypatch.setitem(
            ESTIMATORS, "spy_exp", EstimatorSpec("spy_exp", lambda h: _Spy(), "expanding")
        )
        s = build_schedule(returns.index)

        _Spy.calls = []
        bt.run_backtest(returns, RunConfig(estimators=("spy_roll",), portfolios=("equal_weight",)))
        assert all(n == 252 for _, _, n, _ in _Spy.calls)
        assert [end for _, end, _, _ in _Spy.calls] == [p.rebalance for p in s.periods]

        _Spy.calls = []
        bt.run_backtest(returns, RunConfig(estimators=("spy_exp",), portfolios=("equal_weight",)))
        for (start, end, _n, cols), p in zip(_Spy.calls, s.periods, strict=True):
            assert end == p.rebalance
            # Starts at dcc_start, or later once an instrument that trades only from
            # LATE_START has joined the asset set.
            expected = LATE_START if "LATE" in cols else pd.Timestamp("2013-04-18")
            assert start == expected
        before = [n for _, _, n, cols in _Spy.calls if "LATE" not in cols]
        assert all(b - a == 21 for a, b in zip(before, before[1:], strict=False))
        assert any("LATE" in cols for *_, cols in _Spy.calls)

    def test_same_asset_set_for_both_window_kinds(
        self, returns: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(
            ESTIMATORS, "spy_exp", EstimatorSpec("spy_exp", lambda h: _Spy(), "expanding")
        )
        _Spy.calls = []
        bt.run_backtest(returns, RunConfig(estimators=("spy_exp",), portfolios=("equal_weight",)))
        s = build_schedule(returns.index)
        for p, (_, _, _, cols) in zip(s.periods, _Spy.calls, strict=True):
            assert list(cols) == bt.run_asset_set(returns, s, p, 252)


class TestCache:
    def test_cache_reused_and_invalidated_by_data(
        self, returns: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = RunConfig(estimators=("sample",))
        first = bt.run_backtest(returns, cfg, cache_dir=tmp_path)
        assert len(list(tmp_path.glob("sample_w252_h21_*.npz"))) == 1

        def boom(self: SampleCovariance, r: pd.DataFrame) -> SampleCovariance:
            raise AssertionError("estimator should not be refitted")

        monkeypatch.setattr(SampleCovariance, "fit", boom)
        second = bt.run_backtest(returns, cfg, cache_dir=tmp_path)
        pd.testing.assert_frame_equal(first.forecasts, second.forecasts)

        changed = returns.copy()
        changed.iloc[-1, 0] += 1e-6
        with pytest.raises(AssertionError, match="should not be refitted"):
            bt.run_backtest(changed, cfg, cache_dir=tmp_path)

    def test_fingerprint_sensitivity(self, returns: pd.DataFrame) -> None:
        fp = bt.data_fingerprint(returns)
        assert fp == bt.data_fingerprint(returns.copy())
        changed = returns.copy()
        changed.iloc[100, 2] += 1e-12
        assert bt.data_fingerprint(changed) != fp


class TestSaveLoad:
    def test_roundtrip(self, result: bt.BacktestResult, config: RunConfig, tmp_path: Path) -> None:
        result.save(tmp_path)
        loaded = bt.BacktestResult.load(tmp_path, config)
        pd.testing.assert_frame_equal(loaded.forecasts, result.forecasts)
        pd.testing.assert_frame_equal(loaded.daily_returns, result.daily_returns)
        pd.testing.assert_frame_equal(loaded.weights, result.weights, check_names=False)
        pd.testing.assert_series_equal(
            loaded.period_of_day, result.period_of_day, check_names=False
        )

    def test_load_rejects_mismatched_config(
        self, result: bt.BacktestResult, tmp_path: Path
    ) -> None:
        result.save(tmp_path)
        with pytest.raises(ValueError, match="does not match"):
            bt.BacktestResult.load(tmp_path, RunConfig(window_length=504))


class TestDCCSmoke:
    def test_dcc_runs_end_to_end(self, returns: pd.DataFrame) -> None:
        res = bt.run_backtest(
            returns, RunConfig(estimators=("dcc",), portfolios=("equal_weight", "risk_parity"))
        )
        f = res.forecasts
        assert np.isfinite(f["forecast_var"]).all() and (f["forecast_var"] > 0).all()
        assert set(f["estimator"]) == {"dcc"}
