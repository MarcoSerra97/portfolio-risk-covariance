"""Criterion C (regime robustness), paired comparisons and block-length sensitivity.

Implements ``docs/02-evaluation-design.md`` §9 and §10 on top of
:mod:`portfolio_risk.evaluation.metrics` and
:mod:`portfolio_risk.evaluation.bootstrap`:

* regime tables: the Criterion A metrics and the Criterion B volatility (with
  its ratio to the baseline estimator) recomputed regime by regime; holding
  periods labelled ``"Unclassified"`` are excluded;
* paired comparisons against a baseline estimator, with stationary-bootstrap
  percentile intervals: the mean QLIKE difference (Criterion A, rows =
  holding periods) and the volatility ratio of daily excess returns
  (Criterion B, rows = trading days), for the full sample and each regime;
* block-length sensitivity of those intervals;
* rank stability of the estimators across regimes (Kendall's W).

Diebold–Mariano tests are deliberately not computed within regimes (too few
periods); they belong to the full-sample Criterion A analysis.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from portfolio_risk.evaluation.bootstrap import (
    mean_differences,
    paired_bootstrap,
    volatility_ratios,
)
from portfolio_risk.evaluation.metrics import (
    DEFAULT_NW_LAGS,
    DEFAULT_RISK_FREE,
    TRADING_DAYS,
    add_losses,
    criterion_a_table,
    excess_simple_returns,
)
from portfolio_risk.evaluation.schedule import UNCLASSIFIED

ALL: str = "All"
DEFAULT_BASELINE: str = "sample"
COMPARISON_COLUMNS: list[str] = [
    "criterion",
    "portfolio",
    "regime",
    "estimator",
    "mean_block",
    "n_rows",
    "estimate",
    "lower",
    "upper",
]


def period_regimes(forecasts: pd.DataFrame) -> pd.Series:
    """Regime label of each holding period ``k``, from a forecasts table."""
    labels = forecasts.drop_duplicates("k").set_index("k")["regime"].sort_index()
    labels.name = "regime"
    return labels


def regime_names(forecasts: pd.DataFrame) -> list[str]:
    """Classified regimes in order of first appearance."""
    names = list(dict.fromkeys(period_regimes(forecasts)))
    return [r for r in names if r != UNCLASSIFIED]


# ── regime tables ─────────────────────────────────────────────────────────────


def regime_table_a(forecasts: pd.DataFrame, lags: int = DEFAULT_NW_LAGS) -> pd.DataFrame:
    """Criterion A metrics by (estimator, portfolio, regime), unclassified periods excluded."""
    classified = forecasts[forecasts["regime"] != UNCLASSIFIED]
    return criterion_a_table(classified, by=("estimator", "portfolio", "regime"), lags=lags)


def regime_table_b(
    weights: pd.DataFrame,
    log_returns: pd.DataFrame,
    period_of_day: pd.Series,
    forecasts: pd.DataFrame,
    portfolios: Iterable[str],
    baseline: str = DEFAULT_BASELINE,
    risk_free: str | None = DEFAULT_RISK_FREE,
) -> pd.DataFrame:
    """Annualised excess-return volatility, and its ratio to ``baseline``, by regime.

    Returns:
        DataFrame indexed by ``(portfolio, regime, estimator)`` with columns
        ``n_days``, ``volatility`` and ``vol_ratio``. Includes the full sample
        as regime ``"All"``.
    """
    excess = excess_simple_returns(weights, log_returns, period_of_day, risk_free)
    day_regime = period_of_day.map(period_regimes(forecasts))
    rows = []
    for port in portfolios:
        ex = excess.xs(port, level="portfolio", axis=1)
        for regime in [ALL, *regime_names(forecasts)]:
            mask = np.ones(len(ex), bool) if regime == ALL else (day_regime == regime).to_numpy()
            vols = ex.loc[mask].std(ddof=1) * np.sqrt(TRADING_DAYS)
            for est in ex.columns:
                rows.append(
                    {
                        "portfolio": port,
                        "regime": regime,
                        "estimator": est,
                        "n_days": int(mask.sum()),
                        "volatility": float(vols[est]),
                        "vol_ratio": float(vols[est] / vols[baseline]),
                    }
                )
    return pd.DataFrame(rows).set_index(["portfolio", "regime", "estimator"])


# ── paired comparisons ────────────────────────────────────────────────────────


def _ordered(columns: Iterable[str], baseline: str) -> list[str]:
    cols = list(columns)
    if baseline not in cols:
        raise ValueError(f"baseline {baseline!r} not among estimators {cols}")
    return [baseline] + [c for c in cols if c != baseline]


def _rows(
    criterion: str,
    portfolio: str,
    regime: str,
    names: Sequence[str],
    mean_block: float,
    n_rows: int,
    est: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> list[dict[str, object]]:
    return [
        {
            "criterion": criterion,
            "portfolio": portfolio,
            "regime": regime,
            "estimator": name,
            "mean_block": mean_block,
            "n_rows": n_rows,
            "estimate": float(e),
            "lower": float(lo_),
            "upper": float(hi_),
        }
        for name, e, lo_, hi_ in zip(names, est, lo, hi, strict=True)
    ]


LOSSES: dict[str, str] = {"qlike": "A_qlike_diff", "abs_error": "A_mae_diff"}


def compare_a(
    forecasts: pd.DataFrame,
    portfolio: str,
    mean_block: float,
    reps: int,
    seed: int,
    baseline: str = DEFAULT_BASELINE,
    periods: Iterable[int] | None = None,
    regime: str = ALL,
    level: float = 0.95,
    loss: str = "qlike",
) -> list[dict[str, object]]:
    """Mean loss of each estimator minus the baseline's, with paired bootstrap intervals.

    ``loss`` is ``"qlike"`` or ``"abs_error"`` (whose mean is the MAE). Rows of
    the resampled table are holding periods; negative estimates favour the
    estimator over the baseline.
    """
    if loss not in LOSSES:
        raise ValueError(f"loss must be one of {sorted(LOSSES)}, got {loss!r}")
    data = add_losses(forecasts[forecasts["portfolio"] == portfolio])
    if periods is not None:
        data = data[data["k"].isin(set(periods))]
    order = list(dict.fromkeys(data["estimator"]))  # run order, not alphabetical
    table = data.pivot(index="k", columns="estimator", values=loss)[order]
    names = _ordered(table.columns, baseline)
    res = paired_bootstrap(
        table[names].to_numpy(), mean_differences(0), mean_block, reps, seed, level
    )
    return _rows(
        LOSSES[loss], portfolio, regime, names, mean_block, len(table),
        res.estimate, res.lower, res.upper,
    )  # fmt: skip


def compare_b(
    excess: pd.DataFrame,
    portfolio: str,
    mean_block: float,
    reps: int,
    seed: int,
    baseline: str = DEFAULT_BASELINE,
    day_mask: np.ndarray | None = None,
    regime: str = ALL,
    level: float = 0.95,
) -> list[dict[str, object]]:
    """Volatility of each estimator's daily excess returns divided by the baseline's.

    Rows of the resampled table are trading days; estimates below 1 favour
    the estimator over the baseline.
    """
    ex = excess.xs(portfolio, level="portfolio", axis=1)
    if day_mask is not None:
        ex = ex.loc[day_mask]
    names = _ordered(ex.columns, baseline)
    res = paired_bootstrap(
        ex[names].to_numpy(), volatility_ratios(0, TRADING_DAYS), mean_block, reps, seed, level
    )
    return _rows(
        "B_vol_ratio", portfolio, regime, names, mean_block, len(ex),
        res.estimate, res.lower, res.upper,
    )  # fmt: skip


def comparisons(
    forecasts: pd.DataFrame,
    weights: pd.DataFrame,
    log_returns: pd.DataFrame,
    period_of_day: pd.Series,
    portfolios_a: Iterable[str],
    portfolios_b: Iterable[str],
    mean_block_a: float,
    mean_block_b: float,
    reps: int,
    seed: int,
    baseline: str = DEFAULT_BASELINE,
    risk_free: str | None = DEFAULT_RISK_FREE,
    by_regime: bool = True,
    level: float = 0.95,
    losses: Sequence[str] = ("qlike", "abs_error"),
) -> pd.DataFrame:
    """All paired comparisons: full sample, and (optionally) each classified regime.

    Returns:
        Long table with columns :data:`COMPARISON_COLUMNS`.
    """
    regimes_of_k = period_regimes(forecasts)
    day_regime = period_of_day.map(regimes_of_k).to_numpy()
    excess = excess_simple_returns(weights, log_returns, period_of_day, risk_free)
    scopes = [ALL, *(regime_names(forecasts) if by_regime else [])]
    rows: list[dict[str, object]] = []
    for regime in scopes:
        ks = None if regime == ALL else regimes_of_k.index[regimes_of_k == regime]
        mask = None if regime == ALL else day_regime == regime
        for port in portfolios_a:
            for loss in losses:
                rows += compare_a(
                    forecasts, port, mean_block_a, reps, seed, baseline, ks, regime, level, loss
                )
        for port in portfolios_b:
            rows += compare_b(excess, port, mean_block_b, reps, seed, baseline, mask, regime, level)
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def block_sensitivity(
    forecasts: pd.DataFrame,
    weights: pd.DataFrame,
    log_returns: pd.DataFrame,
    period_of_day: pd.Series,
    portfolios_a: Iterable[str],
    portfolios_b: Iterable[str],
    blocks_a: Sequence[float],
    blocks_b: Sequence[float],
    reps: int,
    seed: int,
    baseline: str = DEFAULT_BASELINE,
    risk_free: str | None = DEFAULT_RISK_FREE,
) -> pd.DataFrame:
    """Full-sample comparisons repeated for each pair of mean block lengths."""
    if len(blocks_a) != len(blocks_b):
        raise ValueError("blocks_a and blocks_b must have the same length")
    frames = [
        comparisons(
            forecasts, weights, log_returns, period_of_day, portfolios_a, portfolios_b,
            ba, bb, reps, seed, baseline, risk_free, by_regime=False,
        )
        for ba, bb in zip(blocks_a, blocks_b, strict=True)
    ]  # fmt: skip
    return pd.concat(frames, ignore_index=True)


# ── rank stability ────────────────────────────────────────────────────────────


def rank_table(
    values: pd.Series, groups: str = "regime", items: str = "estimator", ascending: bool = True
) -> pd.DataFrame:
    """Ranks of ``items`` within each group (1 = best; ties get average ranks).

    Args:
        values: Series indexed by at least ``groups`` and ``items`` (e.g. QLIKE
            by regime and estimator, for one portfolio).
        ascending: ``True`` when lower values are better.

    Returns:
        DataFrame with items as rows and groups as columns.
    """
    wide = values.unstack(groups)
    return wide.rank(axis=0, ascending=ascending, method="average")


def kendalls_w(ranks: pd.DataFrame) -> float:
    """Kendall's coefficient of concordance, with the correction for ties.

    Args:
        ranks: Items as rows, raters (here: regimes) as columns, each column
            a ranking of the items.

    Returns:
        W in [0, 1]: 1 when every regime ranks the estimators identically, 0
        when the rankings show no agreement.
    """
    r = ranks.to_numpy(dtype=float)
    n, m = r.shape
    if n < 2 or m < 2:
        raise ValueError("need at least two items and two raters")
    s = float(((r.sum(axis=1) - m * (n + 1) / 2.0) ** 2).sum())
    ties = 0.0
    for j in range(m):
        _, counts = np.unique(r[:, j], return_counts=True)
        ties += float((counts**3 - counts).sum())
    denom = m**2 * (n**3 - n) - m * ties
    return 12.0 * s / denom if denom > 0 else float("nan")


__all__ = [
    "ALL",
    "COMPARISON_COLUMNS",
    "DEFAULT_BASELINE",
    "LOSSES",
    "block_sensitivity",
    "compare_a",
    "compare_b",
    "comparisons",
    "kendalls_w",
    "period_regimes",
    "rank_table",
    "regime_names",
    "regime_table_a",
    "regime_table_b",
]
