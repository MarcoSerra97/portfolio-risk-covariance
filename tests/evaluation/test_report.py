"""Tests for the evaluation report and the run command."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.data.calendar import nyse_trading_days
from portfolio_risk.evaluation import report as rp
from portfolio_risk.evaluation import run as run_cli
from portfolio_risk.evaluation.backtest import BacktestResult, run_backtest
from portfolio_risk.evaluation.config import RunConfig
from portfolio_risk.evaluation.regimes import ALL

ESTIMATORS = ("sample", "lw_std", "qis_std")


def _returns() -> pd.DataFrame:
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
def returns() -> pd.DataFrame:
    return _returns()


@pytest.fixture(scope="module")
def result(returns: pd.DataFrame) -> BacktestResult:
    return run_backtest(returns, RunConfig(estimators=ESTIMATORS, bootstrap_reps=100))


@pytest.fixture(scope="module")
def tables(result: BacktestResult, returns: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return rp.evaluate(result, returns)


class TestEvaluate:
    def test_all_tables_present(self, tables: dict[str, pd.DataFrame]) -> None:
        assert set(tables) == {
            "calendar", "criterion_a", "criterion_a_regimes", "diebold_mariano",
            "criterion_b", "criterion_b_regimes", "comparisons", "block_sensitivity",
            "ranks", "kendall_w", "diagnostics_summary", "diagnostics",
        }  # fmt: skip

    def test_calendar(self, tables: dict[str, pd.DataFrame], result: BacktestResult) -> None:
        cal = tables["calendar"]
        assert len(cal) == result.forecasts["k"].nunique()
        assert (cal["days"] == 21).all()
        assert cal.loc[0, "start"] == pd.Timestamp("2015-01-02")
        assert cal.loc[60, "regime"] == "Unclassified"

    def test_bootstrap_settings_come_from_config(
        self, tables: dict[str, pd.DataFrame], result: BacktestResult
    ) -> None:
        c = tables["comparisons"]
        cfg = result.config
        assert set(c.loc[c["criterion"].str.startswith("A_"), "mean_block"]) == {
            cfg.block_length_periods
        }
        assert set(c.loc[c["criterion"] == "B_vol_ratio", "mean_block"]) == {cfg.block_length_days}
        s = tables["block_sensitivity"]
        assert sorted(s.loc[s["criterion"] == "B_vol_ratio", "mean_block"].unique()) == [
            21,
            63,
            126,
        ]

    def test_criterion_tables(self, tables: dict[str, pd.DataFrame]) -> None:
        a = tables["criterion_a"]
        assert len(a) == len(ESTIMATORS) * 4
        assert {"mzlog_b", "mz_b", "qlike", "bias_ratio"} <= set(a.columns)
        assert len(tables["criterion_b"]) == len(ESTIMATORS) * 4
        b_reg = tables["criterion_b_regimes"]
        assert set(b_reg.index.get_level_values("portfolio")) == {
            "risk_parity", "min_variance_long_only", "min_variance_long_short",
        }  # fmt: skip

    def test_diebold_mariano(self, tables: dict[str, pd.DataFrame]) -> None:
        dm = tables["diebold_mariano"]
        assert len(dm) == (len(ESTIMATORS) - 1) * 4
        assert dm["p_value"].between(0, 1).all()
        assert "sample" not in set(dm["estimator"])

    def test_rank_stability(self, tables: dict[str, pd.DataFrame]) -> None:
        ranks = tables["ranks"]
        assert ranks["rank"].between(1, len(ESTIMATORS)).all()
        assert ALL not in set(ranks["regime"])
        w = tables["kendall_w"]
        assert len(w) == 4 + 3  # A for every portfolio, B for the optimised ones
        assert w["kendall_w"].dropna().between(0, 1).all()

    def test_diagnostics_summary(self, tables: dict[str, pd.DataFrame]) -> None:
        d = tables["diagnostics_summary"]
        assert list(d.index) == list(ESTIMATORS)
        assert "corr_cond_median" in d.columns and "shrinkage_median" in d.columns

    def test_missing_baseline(self, result: BacktestResult, returns: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="baseline"):
            rp.evaluate(result, returns, baseline="dcc")

    def test_write_tables(self, tables: dict[str, pd.DataFrame], tmp_path: Path) -> None:
        paths = rp.write_tables(tables, tmp_path / "w252")
        assert {p.name for p in paths} == {f"{k}.csv" for k in tables}
        assert all(p.exists() and p.stat().st_size > 0 for p in paths)


class TestRunCommand:
    def test_end_to_end(
        self, returns: pd.DataFrame, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        data = tmp_path / "returns.parquet"
        returns.to_parquet(data)
        argv = [
            "--window", "252", "--estimators", "sample", "lw_std", "--reps", "50",
            "--returns", str(data), "--cache", str(tmp_path / "cache"),
            "--results", str(tmp_path / "results"), "--tables", str(tmp_path / "tables"),
        ]  # fmt: skip
        result = run_cli.main(argv)
        assert result.config.estimators == ("sample", "lw_std")
        assert result.config.bootstrap_reps == 50
        assert (tmp_path / "results" / "w252" / "forecasts.parquet").exists()
        assert (tmp_path / "tables" / "w252" / "comparisons.csv").exists()
        assert any((tmp_path / "cache").iterdir())
        out = capsys.readouterr().out
        assert "QLIKE minus sample" in out and "volatility / sample" in out

    def test_argument_defaults(self) -> None:
        args = run_cli.parse_args([])
        assert args.window == 252
        assert args.reps is None
        assert args.returns.name == "returns.parquet"
