"""Tests for the result figures."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from portfolio_risk.data import universe
from portfolio_risk.data.calendar import nyse_trading_days
from portfolio_risk.evaluation import figures as fg
from portfolio_risk.evaluation.backtest import BacktestResult, run_backtest
from portfolio_risk.evaluation.config import RunConfig
from portfolio_risk.evaluation.report import evaluate, write_tables

ESTIMATORS = ("sample", "lw_std", "qis_std", "dcc")
TICKERS = ["AAPL", "JPM", "XLK", "SHY", "UUP", "FXE", "BIL"]  # real tickers: metadata lookups work


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    idx = nyse_trading_days(date(2013, 1, 2), date(2020, 7, 31))
    rng = np.random.default_rng(0)
    n = len(idx)
    scale = np.where(idx >= pd.Timestamp("2020-02-01"), 3.0, 1.0)
    f = rng.standard_normal(n)
    vols = {"AAPL": 0.015, "JPM": 0.015, "XLK": 0.012, "SHY": 0.001, "UUP": 0.004,
            "FXE": 0.004, "BIL": 0.0002}  # fmt: skip
    data = {t: v * (0.6 * f + 0.8 * rng.standard_normal(n)) * scale for t, v in vols.items()}
    return pd.DataFrame(data, index=idx)[TICKERS]


@pytest.fixture(scope="module")
def result(returns: pd.DataFrame) -> BacktestResult:
    return run_backtest(returns, RunConfig(estimators=ESTIMATORS, bootstrap_reps=50))


@pytest.fixture(scope="module")
def tables(result: BacktestResult, returns: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return evaluate(result, returns)


@pytest.fixture(autouse=True)
def _close_figures() -> None:
    yield
    plt.close("all")


class TestHelpers:
    def test_periods_contiguous(self, result: BacktestResult) -> None:
        p = fg._period_dates(result)
        assert (p["until"].iloc[:-1].to_numpy() == p["start"].iloc[1:].to_numpy()).all()
        assert (p["until"] > p["end"]).all()

    def test_equity_weight(self, result: BacktestResult) -> None:
        eq = fg.equity_weight(result)
        w = result.weights.xs("risk_parity", level="portfolio")
        manual = w[["AAPL", "JPM", "XLK"]].sum(axis=1).groupby(level="estimator").mean()
        np.testing.assert_allclose(eq.loc[manual.index], manual)
        assert list(eq.index) == list(ESTIMATORS)


class TestFigures:
    @pytest.mark.parametrize("criterion", ["A_qlike_diff", "A_mae_diff", "B_vol_ratio"])
    def test_forest(self, tables: dict[str, pd.DataFrame], criterion: str) -> None:
        fig = fg.plot_forest({252: tables["comparisons"], 504: tables["comparisons"]}, criterion)
        n_panels = 4 if criterion.startswith("A_") else 3
        assert isinstance(fig, Figure) and len(fig.axes) == n_panels
        labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
        assert "sample" not in labels and "dcc" in labels  # baseline omitted

    def test_forest_end_caps(self, tables: dict[str, pd.DataFrame]) -> None:
        """Every interval has two vertical end caps at its lower and upper bounds."""
        fig = fg.plot_forest({252: tables["comparisons"]}, "A_mae_diff")
        ax = fig.axes[0]

        def is_cap(ln: plt.Line2D) -> bool:
            x, y = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
            return len(x) == 2 and x[0] == x[1] and np.isclose(y[1] - y[0], 2 * fg.CAP)

        caps = [ln for ln in ax.get_lines() if is_cap(ln)]  # excludes the dashed reference line
        c = tables["comparisons"]
        rows = c[(c.criterion == "A_mae_diff") & (c.portfolio == "equal_weight")
                 & (c.regime == "All") & (c.estimator != "sample")]  # fmt: skip
        assert len(caps) == 2 * len(rows)
        cap_x = sorted(float(ln.get_xdata()[0]) for ln in caps)
        np.testing.assert_allclose(cap_x, sorted([*rows["lower"], *rows["upper"]]))

    def test_forecast_vs_realised(self, result: BacktestResult) -> None:
        fig = fg.plot_forecast_vs_realised(result, estimators=("sample", "dcc"))
        assert len(fig.axes) == 3
        assert {line.get_label() for line in fig.axes[0].get_lines()} >= {"realised", "dcc"}

    def test_forecast_vs_realised_rejects_optimised(self, result: BacktestResult) -> None:
        with pytest.raises(ValueError, match="plot_forecast_vs_own_realised"):
            fg.plot_forecast_vs_realised(result, portfolio="min_variance_long_short")

    def test_forecast_vs_own_realised(self, result: BacktestResult) -> None:
        fig = fg.plot_forecast_vs_own_realised(result, "min_variance_long_short")
        visible = [a for a in fig.axes if a.get_visible()]
        assert len(visible) == len(ESTIMATORS)
        # Each panel's realised line is that estimator's own portfolio.
        f = result.forecasts.query("portfolio == 'min_variance_long_short'")
        scale = 252 / 21
        for ax, est in zip(visible, ESTIMATORS, strict=True):
            realised = [ln for ln in ax.get_lines() if ln.get_label() == "realised"][0]
            own = np.sqrt(f[f["estimator"] == est].sort_values("k")["realised_var"] * scale)
            np.testing.assert_allclose(np.asarray(realised.get_ydata(), dtype=float), own)

    def test_cumulative_qlike(self, result: BacktestResult) -> None:
        fig = fg.plot_cumulative_qlike(result)
        assert len(fig.axes) == 4
        # one line per non-baseline estimator, plus the zero line
        assert len(fig.axes[0].get_lines()) == len(ESTIMATORS) - 1 + 1

    def test_qlike_by_regime(self, tables: dict[str, pd.DataFrame]) -> None:
        fig = fg.plot_qlike_by_regime(tables["criterion_a_regimes"], ["Calm I", "COVID shock"])
        assert len(fig.axes) == 4 + 1  # four panels and the colour bar

    @pytest.mark.parametrize("by", ["bucket", "instrument"])
    def test_composition(self, result: BacktestResult, by: str) -> None:
        fig = fg.plot_composition(result, "min_variance_long_short", by=by, top=3)
        visible = [a for a in fig.axes if a.get_visible()]
        assert len(visible) == len(ESTIMATORS)
        labels = {line.get_label() for line in visible[0].get_lines()}
        assert ("other" in labels) == (by == "instrument")

    def test_composition_rejects_bad_mode(self, result: BacktestResult) -> None:
        with pytest.raises(ValueError, match="bucket"):
            fg.plot_composition(result, "risk_parity", by="sector")

    def test_risk_parity_mechanism(
        self, result: BacktestResult, tables: dict[str, pd.DataFrame]
    ) -> None:
        fig = fg.plot_risk_parity_mechanism({252: result}, {252: tables["comparisons"]})
        assert len(fig.axes) == 1

    def test_dcc_panel(self, result: BacktestResult) -> None:
        fig = fg.plot_dcc_panel(result)
        assert len(fig.axes) == 4 + 1  # four panels and the half-life axis


class TestMakeFigures:
    def test_writes_all_pngs(
        self, result: BacktestResult, tables: dict[str, pd.DataFrame], tmp_path: Path
    ) -> None:
        paths = fg.make_figures({252: result}, {252: tables}, 252, tmp_path)
        names = {p.stem for p in paths}
        assert "dcc_panel" in names and "forest_volatility" in names and len(names) == 13
        optimised = ("risk_parity", "min_variance_long_only", "min_variance_long_short")
        assert {f"forecast_vs_realised_{p}" for p in optimised} <= names
        assert all(p.stat().st_size > 10_000 for p in paths)

    def test_command_from_saved_files(
        self, result: BacktestResult, tables: dict[str, pd.DataFrame], tmp_path: Path
    ) -> None:
        result.save(tmp_path / "results" / "w252")
        write_tables(tables, tmp_path / "tables" / "w252")
        paths = fg.main([
            "--window", "252", "--results", str(tmp_path / "results"),
            "--tables", str(tmp_path / "tables"), "--figures", str(tmp_path / "figures"),
        ])  # fmt: skip
        assert len(paths) == 13
        assert (tmp_path / "figures" / "w252" / "qlike_by_regime.png").exists()

    def test_command_missing_window(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            fg.main(["--window", "504", "--results", str(tmp_path)])


def test_tickers_are_in_universe() -> None:
    assert set(TICKERS) <= set(universe.metadata_frame().index)
