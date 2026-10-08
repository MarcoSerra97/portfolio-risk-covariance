"""All evaluation tables for one backtest run.

:func:`evaluate` turns a :class:`~portfolio_risk.evaluation.backtest.BacktestResult`
into the tables of Criteria A, B and C, reading every setting (bootstrap
replications, block lengths, seed, portfolios, estimators) from the run's own
configuration, so the reported intervals always follow the design document.
:func:`write_tables` saves them as CSV files.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from portfolio_risk.evaluation.backtest import BacktestResult
from portfolio_risk.evaluation.metrics import (
    DEFAULT_RISK_FREE,
    add_losses,
    criterion_a_table,
    criterion_b_table,
    diebold_mariano,
)
from portfolio_risk.evaluation.regimes import (
    ALL,
    DEFAULT_BASELINE,
    block_sensitivity,
    comparisons,
    kendalls_w,
    rank_table,
    regime_table_a,
    regime_table_b,
)
from portfolio_risk.portfolios.constructions import CONSTRUCTIONS

#: Block lengths for the sensitivity analysis: Criterion A (periods), Criterion B (days).
SENSITIVITY_BLOCKS_A: tuple[float, ...] = (3.0, 6.0, 12.0)
SENSITIVITY_BLOCKS_B: tuple[float, ...] = (21.0, 63.0, 126.0)


def optimised_portfolios(result: BacktestResult) -> list[str]:
    """The run's portfolios whose weights depend on the covariance."""
    return [p for p in result.config.portfolios if CONSTRUCTIONS[p].uses_covariance]


def calendar_table(result: BacktestResult) -> pd.DataFrame:
    """One row per holding period: rebalance date, first and last day, regime."""
    days = result.period_of_day.reset_index()
    days.columns = ["date", "k"]
    span = days.groupby("k")["date"].agg(start="min", end="max", days="count")
    info = result.forecasts.drop_duplicates("k").set_index("k")[["rebalance", "regime"]]
    return info.join(span)[["rebalance", "start", "end", "days", "regime"]]


def diebold_mariano_table(result: BacktestResult, baseline: str = DEFAULT_BASELINE) -> pd.DataFrame:
    """Full-sample Diebold–Mariano tests of QLIKE against the baseline, per portfolio."""
    losses = add_losses(result.forecasts)
    rows = []
    for port in result.config.portfolios:
        table = losses[losses["portfolio"] == port].pivot(
            index="k", columns="estimator", values="qlike"
        )
        for est in result.config.estimators:
            if est == baseline:
                continue
            stat, p = diebold_mariano(table[est].to_numpy(), table[baseline].to_numpy())
            rows.append({"portfolio": port, "estimator": est, "dm_stat": stat, "p_value": p})
    return pd.DataFrame(rows)


def rank_stability(
    result: BacktestResult, log_returns: pd.DataFrame, baseline: str = DEFAULT_BASELINE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ranks of the estimators by regime, and Kendall's W of agreement across regimes.

    Criterion A ranks by mean QLIKE (every portfolio); Criterion B by
    volatility (optimised portfolios). Rank 1 is best.

    Returns:
        ``(ranks, kendall)``: long tables.
    """
    a_tab = regime_table_a(result.forecasts)
    b_tab = regime_table_b(
        result.weights, log_returns, result.period_of_day, result.forecasts,
        optimised_portfolios(result), baseline,
    )  # fmt: skip
    rank_rows, w_rows = [], []
    for port in result.config.portfolios:
        q = a_tab.xs(port, level="portfolio")["qlike"]
        r = rank_table(q.swaplevel())  # index (regime, estimator)
        rank_rows.append(
            r.reset_index().melt(id_vars="estimator").assign(criterion="A_qlike", portfolio=port)
        )
        w_rows.append({"criterion": "A_qlike", "portfolio": port, "kendall_w": kendalls_w(r)})
    for port in optimised_portfolios(result):
        vols = b_tab.xs(port, level="portfolio")["volatility"]
        vols = vols[vols.index.get_level_values("regime") != ALL]
        r = rank_table(vols)
        rank_rows.append(
            r.reset_index()
            .melt(id_vars="estimator")
            .assign(criterion="B_volatility", portfolio=port)
        )
        w_rows.append({"criterion": "B_volatility", "portfolio": port, "kendall_w": kendalls_w(r)})
    ranks = pd.concat(rank_rows, ignore_index=True).rename(columns={"value": "rank"})
    return ranks[["criterion", "portfolio", "regime", "estimator", "rank"]], pd.DataFrame(w_rows)


def diagnostics_summary(result: BacktestResult) -> pd.DataFrame:
    """Median, minimum and maximum of each diagnostic over the rebalances, per estimator."""
    d = result.diagnostics
    out = d.groupby(level="estimator", sort=False).agg(["median", "min", "max"])
    out.columns = [f"{c[0]}_{c[1]}" for c in out.columns]
    return out.dropna(axis=1, how="all")


def evaluate(
    result: BacktestResult,
    log_returns: pd.DataFrame,
    baseline: str = DEFAULT_BASELINE,
    risk_free: str | None = DEFAULT_RISK_FREE,
    blocks_a: Sequence[float] = SENSITIVITY_BLOCKS_A,
    blocks_b: Sequence[float] = SENSITIVITY_BLOCKS_B,
) -> dict[str, pd.DataFrame]:
    """Every evaluation table for one run.

    Bootstrap settings come from ``result.config``: ``bootstrap_reps``,
    ``block_length_periods`` (Criterion A), ``block_length_days`` (Criterion
    B) and ``seed``.

    Raises:
        ValueError: If ``baseline`` is not one of the run's estimators.
    """
    cfg = result.config
    if baseline not in cfg.estimators:
        raise ValueError(f"baseline {baseline!r} is not among the run's estimators")
    optimised = optimised_portfolios(result)
    ranks, kendall = rank_stability(result, log_returns, baseline)
    common = (result.forecasts, result.weights, log_returns, result.period_of_day)
    return {
        "calendar": calendar_table(result),
        "criterion_a": criterion_a_table(result.forecasts),
        "criterion_a_regimes": regime_table_a(result.forecasts),
        "diebold_mariano": diebold_mariano_table(result, baseline),
        "criterion_b": criterion_b_table(
            result.weights, log_returns, result.period_of_day, risk_free
        ),
        "criterion_b_regimes": regime_table_b(
            result.weights, log_returns, result.period_of_day, result.forecasts,
            optimised, baseline, risk_free,
        ),
        "comparisons": comparisons(
            *common, cfg.portfolios, optimised, cfg.block_length_periods,
            cfg.block_length_days, cfg.bootstrap_reps, cfg.seed, baseline, risk_free,
        ),
        "block_sensitivity": block_sensitivity(
            *common, cfg.portfolios, optimised, blocks_a, blocks_b,
            cfg.bootstrap_reps, cfg.seed, baseline, risk_free,
        ),
        "ranks": ranks,
        "kendall_w": kendall,
        "diagnostics_summary": diagnostics_summary(result),
        "diagnostics": result.diagnostics,
    }  # fmt: skip


def write_tables(tables: dict[str, pd.DataFrame], directory: Path) -> list[Path]:
    """Write each table to ``directory/<name>.csv``; returns the paths written."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, table in tables.items():
        path = directory / f"{name}.csv"
        table.to_csv(path)
        paths.append(path)
    return paths


__all__ = [
    "SENSITIVITY_BLOCKS_A",
    "SENSITIVITY_BLOCKS_B",
    "calendar_table",
    "diagnostics_summary",
    "diebold_mariano_table",
    "evaluate",
    "optimised_portfolios",
    "rank_stability",
    "write_tables",
]
