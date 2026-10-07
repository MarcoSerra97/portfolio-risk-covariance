"""Tests for Criterion C, paired comparisons, block sensitivity and rank stability."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.data.calendar import nyse_trading_days
from portfolio_risk.evaluation import regimes as rg
from portfolio_risk.evaluation.backtest import BacktestResult, run_backtest
from portfolio_risk.evaluation.config import RunConfig
from portfolio_risk.evaluation.metrics import excess_simple_returns
from portfolio_risk.evaluation.schedule import UNCLASSIFIED

OPTIMISED = ("risk_parity", "min_variance_long_only", "min_variance_long_short")


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    """Synthetic panel 2013 to mid-2020, with volatility tripling from Feb 2020."""
    idx = nyse_trading_days(date(2013, 1, 2), date(2020, 7, 31))
    rng = np.random.default_rng(0)
    n = len(idx)
    scale = np.where(idx >= pd.Timestamp("2020-02-01"), 3.0, 1.0)
    f = rng.standard_normal(n)
    data = {
        name: 0.01 * s * (0.6 * f + 0.8 * rng.standard_normal(n)) * scale
        for name, s in [("A", 1.0), ("B", 1.5), ("C", 0.8), ("D", 2.0)]
    }
    data["BIL"] = 0.0002 * rng.standard_normal(n)
    return pd.DataFrame(data, index=idx)


@pytest.fixture(scope="module")
def config() -> RunConfig:
    return RunConfig(estimators=("sample", "lw_std", "qis_std"), bootstrap_reps=200)


@pytest.fixture(scope="module")
def result(returns: pd.DataFrame, config: RunConfig) -> BacktestResult:
    return run_backtest(returns, config)


class TestLabels:
    def test_period_regimes(self, result: BacktestResult) -> None:
        labels = rg.period_regimes(result.forecasts)
        assert labels.index.tolist() == sorted(result.forecasts["k"].unique())
        assert labels.loc[60] == UNCLASSIFIED
        assert labels.loc[[61, 62, 63, 64]].eq("COVID shock").all()

    def test_regime_names_exclude_unclassified(self, result: BacktestResult) -> None:
        names = rg.regime_names(result.forecasts)
        assert UNCLASSIFIED not in names
        assert names[:2] == ["Calm I", "COVID shock"]


class TestRegimeTables:
    def test_table_a_counts_and_exclusion(self, result: BacktestResult) -> None:
        t = rg.regime_table_a(result.forecasts)
        regimes = t.index.get_level_values("regime")
        assert UNCLASSIFIED not in set(regimes)
        assert t.loc[("sample", "equal_weight", "Calm I"), "n_periods"] == 60
        assert t.loc[("sample", "equal_weight", "COVID shock"), "n_periods"] == 4

    def test_table_b_ratios_and_days(self, result: BacktestResult, returns: pd.DataFrame) -> None:
        t = rg.regime_table_b(
            result.weights, returns, result.period_of_day, result.forecasts, OPTIMISED
        )
        assert (t.xs("sample", level="estimator")["vol_ratio"] == 1.0).all()
        assert t.loc[("risk_parity", "COVID shock", "sample"), "n_days"] == 4 * 21
        assert t.loc[("risk_parity", rg.ALL, "sample"), "n_days"] == len(result.period_of_day)
        calm = t.loc[("risk_parity", "Calm I", "sample"), "volatility"]
        covid = t.loc[("risk_parity", "COVID shock", "sample"), "volatility"]
        assert covid > 2 * calm  # volatility tripled in the synthetic shock


def _synthetic_forecasts(n: int = 120) -> pd.DataFrame:
    """'good' forecasts exactly; 'bad' forecasts half the truth; 'copy' equals 'sample'."""
    rng = np.random.default_rng(1)
    truth = rng.uniform(1.0, 2.0, n)
    realised = truth * rng.chisquare(21, n) / 21
    rows = []
    for est, f in [
        ("sample", truth * 0.8),
        ("copy", truth * 0.8),
        ("good", truth),
        ("bad", truth * 0.5),
    ]:
        for k in range(n):
            rows.append({"estimator": est, "portfolio": "p", "k": k, "regime": "R",
                         "forecast_var": f[k], "realised_var": realised[k]})  # fmt: skip
    return pd.DataFrame(rows)


class TestCompareA:
    def test_baseline_copy_good_bad(self) -> None:
        out = pd.DataFrame(rg.compare_a(_synthetic_forecasts(), "p", 3.0, 300, seed=2))
        out = out.set_index("estimator")
        assert list(out.index) == ["sample", "copy", "good", "bad"]  # baseline first
        assert (out.loc["sample", ["estimate", "lower", "upper"]] == 0).all()
        assert (out.loc["copy", ["estimate", "lower", "upper"]] == 0).all()
        assert out.loc["good", "upper"] < 0  # reliably better than the baseline
        assert out.loc["bad", "lower"] > 0  # reliably worse

    def test_mae_differences(self) -> None:
        out = pd.DataFrame(
            rg.compare_a(_synthetic_forecasts(), "p", 3.0, 300, seed=2, loss="abs_error")
        ).set_index("estimator")
        assert set(out["criterion"]) == {"A_mae_diff"}
        assert (out.loc["copy", ["estimate", "lower", "upper"]] == 0).all()
        f = _synthetic_forecasts()
        mae = (f["forecast_var"] - f["realised_var"]).abs().groupby(f["estimator"]).mean()
        assert out.loc["good", "estimate"] == pytest.approx(mae["good"] - mae["sample"])

    def test_unknown_loss_raises(self) -> None:
        with pytest.raises(ValueError, match="loss"):
            rg.compare_a(_synthetic_forecasts(), "p", 3.0, 10, seed=0, loss="mse")

    def test_missing_baseline_raises(self) -> None:
        f = _synthetic_forecasts()
        with pytest.raises(ValueError, match="baseline"):
            rg.compare_a(f, "p", 3.0, 50, seed=0, baseline="nope")


class TestCompareB:
    def test_scaled_returns_give_exact_ratio(self) -> None:
        days = pd.bdate_range("2020-01-01", periods=300)
        base = np.random.default_rng(3).normal(0, 0.01, 300)
        cols = pd.MultiIndex.from_tuples(
            [("sample", "p"), ("double", "p")], names=["estimator", "portfolio"]
        )
        excess = pd.DataFrame(np.column_stack([base, 2 * base]), index=days, columns=cols)
        out = pd.DataFrame(rg.compare_b(excess, "p", 21.0, 300, seed=4)).set_index("estimator")
        assert out.loc["sample", "estimate"] == 1.0
        np.testing.assert_allclose(out.loc["double", ["estimate", "lower", "upper"]], 2.0)


class TestComparisons:
    def test_scopes_and_rows(
        self, result: BacktestResult, returns: pd.DataFrame, config: RunConfig
    ) -> None:
        out = rg.comparisons(
            result.forecasts, result.weights, returns, result.period_of_day,
            ("equal_weight",), OPTIMISED, 3.0, 21.0, config.bootstrap_reps, config.seed,
        )  # fmt: skip
        assert list(out.columns) == rg.COMPARISON_COLUMNS
        scopes = list(dict.fromkeys(out["regime"]))
        assert scopes == [rg.ALL, "Calm I", "COVID shock", "Post-COVID"]
        per_scope = 3 * (2 + len(OPTIMISED))  # 3 estimators x (2 losses for 1 A portfolio + 3 B)
        assert len(out) == per_scope * len(scopes)
        base = out[out["estimator"] == "sample"]
        assert (base.loc[base["criterion"] == "A_qlike_diff", "estimate"] == 0).all()
        assert (base.loc[base["criterion"] == "B_vol_ratio", "estimate"] == 1).all()
        covid_b = out.query("regime == 'COVID shock' and criterion == 'B_vol_ratio'")
        assert (covid_b["n_rows"] == 84).all()

    def test_reproducible(
        self, result: BacktestResult, returns: pd.DataFrame, config: RunConfig
    ) -> None:
        args = (result.forecasts, result.weights, returns, result.period_of_day,
                ("equal_weight",), ("risk_parity",), 3.0, 21.0, 100, 7)  # fmt: skip
        pd.testing.assert_frame_equal(rg.comparisons(*args), rg.comparisons(*args))

    def test_block_sensitivity(
        self, result: BacktestResult, returns: pd.DataFrame, config: RunConfig
    ) -> None:
        out = rg.block_sensitivity(
            result.forecasts, result.weights, returns, result.period_of_day,
            ("equal_weight",), ("risk_parity",), (3.0, 6.0, 12.0), (63.0, 126.0, 252.0),
            100, config.seed,
        )  # fmt: skip
        assert set(out["regime"]) == {rg.ALL}
        assert sorted(out.loc[out["criterion"] == "A_qlike_diff", "mean_block"].unique()) == [
            3,
            6,
            12,
        ]
        assert sorted(out.loc[out["criterion"] == "B_vol_ratio", "mean_block"].unique()) == [
            63,
            126,
            252,
        ]
        with pytest.raises(ValueError, match="same length"):
            rg.block_sensitivity(
                result.forecasts, result.weights, returns, result.period_of_day,
                ("equal_weight",), ("risk_parity",), (3.0,), (63.0, 126.0), 10, 0,
            )  # fmt: skip

    def test_excess_returns_consistent_with_metrics(
        self, result: BacktestResult, returns: pd.DataFrame
    ) -> None:
        ex = excess_simple_returns(result.weights, returns, result.period_of_day)
        t = rg.regime_table_b(
            result.weights, returns, result.period_of_day, result.forecasts, ("risk_parity",)
        )
        vol = ex[("lw_std", "risk_parity")].std(ddof=1) * np.sqrt(252)
        assert t.loc[("risk_parity", rg.ALL, "lw_std"), "volatility"] == pytest.approx(vol)


class TestRankStability:
    def test_rank_table(self) -> None:
        values = pd.Series(
            {("R1", "a"): 0.1, ("R1", "b"): 0.3, ("R1", "c"): 0.2,
             ("R2", "a"): 0.5, ("R2", "b"): 0.4, ("R2", "c"): 0.4},
        )  # fmt: skip
        values.index = values.index.set_names(["regime", "estimator"])
        ranks = rg.rank_table(values)
        assert ranks["R1"].tolist() == [1.0, 3.0, 2.0]
        assert ranks["R2"].tolist() == [3.0, 1.5, 1.5]  # tie -> average rank

    def test_kendalls_w_known_values(self) -> None:
        same = pd.DataFrame({"r1": [1, 2, 3], "r2": [1, 2, 3]})
        reversed_ = pd.DataFrame({"r1": [1, 2, 3], "r2": [3, 2, 1]})
        textbook = pd.DataFrame({"r1": [1, 2, 3], "r2": [1, 3, 2]})  # S = 6, W = 0.75
        assert rg.kendalls_w(same) == pytest.approx(1.0)
        assert rg.kendalls_w(reversed_) == pytest.approx(0.0)
        assert rg.kendalls_w(textbook) == pytest.approx(0.75)

    def test_kendalls_w_ties_and_errors(self) -> None:
        all_tied = pd.DataFrame({"r1": [2.0, 2.0, 2.0], "r2": [2.0, 2.0, 2.0]})
        assert np.isnan(rg.kendalls_w(all_tied))
        with pytest.raises(ValueError):
            rg.kendalls_w(pd.DataFrame({"r1": [1, 2]}))
