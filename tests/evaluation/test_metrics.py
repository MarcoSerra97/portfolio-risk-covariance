"""Tests for the Criterion A and B metrics."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.data.calendar import nyse_trading_days
from portfolio_risk.evaluation import metrics as m
from portfolio_risk.evaluation.backtest import run_backtest
from portfolio_risk.evaluation.config import RunConfig

# ── Criterion A ───────────────────────────────────────────────────────────────


class TestQlike:
    def test_zero_for_perfect_forecast(self) -> None:
        np.testing.assert_allclose(m.qlike(np.array([1.0, 2.0]), np.array([1.0, 2.0])), 0.0)

    def test_positive_and_asymmetric(self) -> None:
        under = m.qlike(np.array([2.0]), np.array([1.0]))[0]  # forecast half the truth
        over = m.qlike(np.array([0.5]), np.array([1.0]))[0]  # forecast double the truth
        assert under == pytest.approx(2 - np.log(2) - 1)  # 0.307
        assert over == pytest.approx(0.5 + np.log(2) - 1)  # 0.193
        assert under > over > 0

    def test_scale_free(self) -> None:
        a = m.qlike(np.array([3.0]), np.array([2.0]))
        b = m.qlike(np.array([3e-6]), np.array([2e-6]))
        np.testing.assert_allclose(a, b)

    @pytest.mark.parametrize(("v", "f"), [(1.0, 0.0), (0.0, 1.0), (1.0, -1.0)])
    def test_rejects_non_positive(self, v: float, f: float) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            m.qlike(np.array([v]), np.array([f]))


class TestNeweyWest:
    def test_coefficients_match_ols(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=200)
        y = 1.0 + 2.0 * x + rng.normal(size=200)
        beta, se, r2 = m.newey_west_ols(y, x, lags=3)
        slope, intercept = np.polyfit(x, y, 1)
        np.testing.assert_allclose(beta, [intercept, slope])
        assert (se > 0).all() and 0 < r2 < 1

    def test_lag_zero_is_white_hc0(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.normal(size=50)
        y = 0.5 * x + rng.normal(size=50) * (1 + np.abs(x))
        beta, se, _ = m.newey_west_ols(y, x, lags=0)
        design = np.column_stack([np.ones(50), x])
        resid = y - design @ beta
        bread = np.linalg.inv(design.T @ design)
        hc0 = bread @ (design.T * resid**2) @ design @ bread
        np.testing.assert_allclose(se, np.sqrt(np.diag(hc0)))

    def test_perfect_fit(self) -> None:
        x = np.linspace(1, 2, 20)
        beta, _, r2 = m.newey_west_ols(x, x)
        np.testing.assert_allclose(beta, [0.0, 1.0], atol=1e-12)
        assert r2 == pytest.approx(1.0)


class TestDieboldMariano:
    def test_identical_losses(self) -> None:
        loss = np.array([0.1, 0.3, 0.2, 0.5])
        assert m.diebold_mariano(loss, loss) == (0.0, 1.0)

    def test_detects_better_forecast(self) -> None:
        rng = np.random.default_rng(2)
        base = rng.exponential(size=300)
        stat, p = m.diebold_mariano(base, base + 0.2 + 0.05 * rng.normal(size=300))
        assert stat < 0 and p < 0.01

    def test_no_difference_not_rejected_too_often(self) -> None:
        rng = np.random.default_rng(3)
        rejections = sum(
            m.diebold_mariano(rng.normal(size=131), rng.normal(size=131))[1] < 0.05
            for _ in range(400)
        )
        assert rejections / 400 < 0.10


def _forecasts(ratio_by_est: dict[str, float], n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    rows = []
    for est, ratio in ratio_by_est.items():
        f = rng.uniform(1.0, 3.0, n)
        for k in range(n):
            rows.append(
                {"estimator": est, "portfolio": "p", "k": k, "regime": "r",
                 "forecast_var": f[k], "realised_var": ratio * f[k]}
            )  # fmt: skip
    return pd.DataFrame(rows)


class TestCriterionATable:
    def test_values(self) -> None:
        table = m.criterion_a_table(_forecasts({"perfect": 1.0, "low": 2.0}))
        perfect = table.loc[("perfect", "p")]
        assert perfect["qlike"] == pytest.approx(0.0, abs=1e-12)
        assert perfect["mae"] == pytest.approx(0.0, abs=1e-12)
        assert perfect["bias_ratio"] == pytest.approx(1.0)
        assert perfect["mz_b"] == pytest.approx(1.0) and perfect["mz_r2"] == pytest.approx(1.0)
        low = table.loc[("low", "p")]
        assert low["bias_ratio"] == pytest.approx(2.0)
        assert low["qlike"] == pytest.approx(2 - np.log(2) - 1)
        assert low["mz_b"] == pytest.approx(2.0)

    def test_group_by_regime(self) -> None:
        f = _forecasts({"a": 1.0})
        f["regime"] = np.where(f["k"] < 20, "calm", "shock")
        table = m.criterion_a_table(f, by=("estimator", "portfolio", "regime"))
        assert table.loc[("a", "p", "calm"), "n_periods"] == 20
        assert table.loc[("a", "p", "shock"), "n_periods"] == 30


# ── Criterion B ───────────────────────────────────────────────────────────────


def _two_asset_setup() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Two periods of 2 days. Asset X doubles over period 0; Y is flat; RF earns 0."""
    days = pd.bdate_range("2020-01-01", periods=4)
    log_r = pd.DataFrame(
        {"X": [np.log(2) / 2, np.log(2) / 2, 0.0, 0.0], "Y": [0.0] * 4, "RF": [0.0] * 4},
        index=days,
    )
    period = pd.Series([0, 0, 1, 1], index=days, name="k")
    idx = pd.MultiIndex.from_tuples(
        [("e", "p", 0), ("e", "p", 1)], names=["estimator", "portfolio", "k"]
    )
    weights = pd.DataFrame({"X": [0.5, 0.5], "Y": [0.5, 0.5], "RF": [0.0, 0.0]}, index=idx)
    return weights, log_r, period


class TestSimpleReturns:
    def test_exact_simple_returns(self) -> None:
        weights, log_r, period = _two_asset_setup()
        r = m.simple_portfolio_returns(weights, log_r, period)[("e", "p")]
        expected = 0.5 * (np.exp(np.log(2) / 2) - 1)
        np.testing.assert_allclose(r.to_numpy(), [expected, expected, 0.0, 0.0])


class TestTurnover:
    def test_worked_example(self) -> None:
        """X doubles: (0.5, 0.5) drifts to (2/3, 1/3); back to (0.5, 0.5) trades 1/3."""
        weights, log_r, period = _two_asset_setup()
        t = m.turnover_series(weights, log_r, period)
        assert list(t.index.get_level_values("k")) == [1]  # first rebalance excluded
        assert t.iloc[0] == pytest.approx(1 / 3)

    def test_no_drift_no_turnover(self) -> None:
        weights, log_r, period = _two_asset_setup()
        log_r["Y"] = log_r["X"]  # both assets move together: no drift
        assert m.turnover_series(weights, log_r, period).iloc[0] == pytest.approx(0.0)


class TestCriterionBTable:
    def test_volatility_sharpe_and_risk_free(self) -> None:
        days = pd.bdate_range("2020-01-01", periods=6)
        rng = np.random.default_rng(5)
        log_r = pd.DataFrame({"A": rng.normal(0, 0.01, 6), "RF": [0.0001] * 6}, index=days)
        period = pd.Series([0, 0, 0, 1, 1, 1], index=days)
        idx = pd.MultiIndex.from_tuples(
            [("e", "a_only", 0), ("e", "a_only", 1), ("e", "cash", 0), ("e", "cash", 1)],
            names=["estimator", "portfolio", "k"],
        )
        weights = pd.DataFrame({"A": [1, 1, 0, 0], "RF": [0, 0, 1, 1]}, index=idx, dtype=float)
        table = m.criterion_b_table(weights, log_r, period, risk_free="RF")
        excess = np.expm1(log_r["A"]) - np.expm1(log_r["RF"])
        vol = excess.std(ddof=1) * np.sqrt(252)
        assert table.loc[("e", "a_only"), "volatility"] == pytest.approx(vol)
        assert table.loc[("e", "a_only"), "sharpe"] == pytest.approx(excess.mean() * 252 / vol)
        assert table.loc[("e", "cash"), "volatility"] == pytest.approx(0.0, abs=1e-15)

    def test_concentration_and_exposure(self) -> None:
        weights, log_r, period = _two_asset_setup()
        weights.loc[("e", "p", 1)] = [1.5, -0.5, 0.0]
        row = m.criterion_b_table(weights, log_r, period, risk_free="RF").loc[("e", "p")]
        assert row["n_eff"] == pytest.approx((2.0 + 1 / 2.5) / 2)
        assert row["max_abs_weight"] == pytest.approx((0.5 + 1.5) / 2)
        assert row["gross_exposure"] == pytest.approx((1.0 + 2.0) / 2)

    def test_period_subset(self) -> None:
        weights, log_r, period = _two_asset_setup()
        row = m.criterion_b_table(weights, log_r, period, risk_free="RF", periods=[1])
        assert row.loc[("e", "p"), "n_days"] == 2


class TestOnBacktest:
    def test_tables_from_a_real_run(self) -> None:
        idx = nyse_trading_days(date(2013, 1, 2), date(2015, 12, 31))
        rng = np.random.default_rng(6)
        f = rng.standard_normal(len(idx))
        data = {n: 0.01 * (0.6 * f + 0.8 * rng.standard_normal(len(idx))) for n in "ABCD"}
        data["BIL"] = 0.0002 * rng.standard_normal(len(idx))
        returns = pd.DataFrame(data, index=idx)
        res = run_backtest(returns, RunConfig(estimators=("sample", "lw_std")))
        a = m.criterion_a_table(res.forecasts)
        b = m.criterion_b_table(res.weights, returns, res.period_of_day)
        assert len(a) == 2 * 4 and len(b) == 2 * 4
        assert np.isfinite(a[["mae", "qlike", "bias_ratio", "mz_b"]].to_numpy()).all()
        assert np.isfinite(b[["volatility", "turnover", "n_eff"]].to_numpy()).all()
        assert (b.xs("equal_weight", level="portfolio")["turnover"] < 0.2).all()
