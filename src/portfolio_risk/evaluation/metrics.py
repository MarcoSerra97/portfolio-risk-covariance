"""Criterion A and Criterion B metrics.

Implements ``docs/02-evaluation-design.md`` §7 and §8.

Criterion A (risk-measurement accuracy) works on the per-period forecast and
realised variances recorded by the backtest, both built from log returns:

* MAE and QLIKE (co-primary) and the bias ratio mean(v / v̂) (level);
* the Mincer–Zarnowitz calibration regression in logs,
  log v = a + b log v̂, whose slope measures proportional tracking: a
  constant multiplicative bias (e.g. the optimism of optimised portfolios)
  moves only the intercept. The regression in levels, v = a + b v̂, is also
  reported for comparison with the literature; for optimised portfolios its
  slope is confounded by that optimism. Both use Newey–West standard errors;
* a Diebold–Mariano test for comparing two estimators' loss series.

Criterion B (optimisation quality) recomputes the daily out-of-sample returns
from the rebalance weights with simple returns, which are exact for weights
held constant within each period (Ledoit & Wolf, 2017):
``R_t^p = Σ_i w_i (exp(r_{i,t}) - 1)``, and reports

* realised volatility of returns in excess of the risk-free instrument
  (annualised with √252) and the Sharpe ratio;
* turnover against the previous weights drifted with prices over the holding
  period (DeMiguel, Garlappi & Uppal, 2009), excluding the first rebalance;
* concentration (effective number of assets, maximum absolute weight) and
  gross exposure.

Table functions group by arbitrary columns, so regime tables (Criterion C)
reuse them with ``by=("estimator", "portfolio", "regime")``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

TRADING_DAYS: int = 252
DEFAULT_RISK_FREE: str = "BIL"
DEFAULT_NW_LAGS: int = 3
#: Fewest periods for which the Mincer–Zarnowitz regression is reported.
MIN_PERIODS_MZ: int = 20


# ── Criterion A: losses ───────────────────────────────────────────────────────


def qlike(realised: np.ndarray | pd.Series, forecast: np.ndarray | pd.Series) -> np.ndarray:
    """QLIKE loss ``v/v̂ - log(v/v̂) - 1`` per observation (0 for a perfect forecast).

    Raises:
        ValueError: If any forecast or realised value is not strictly positive.
    """
    v = np.asarray(realised, dtype=float)
    f = np.asarray(forecast, dtype=float)
    if bool((f <= 0).any()) or bool((v <= 0).any()):
        raise ValueError("QLIKE requires strictly positive forecasts and realised variances")
    ratio = v / f
    loss: np.ndarray = ratio - np.log(ratio) - 1.0
    return loss


def add_losses(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Copy of a forecasts table with per-period ``abs_error``, ``qlike`` and ``ratio`` columns."""
    out = forecasts.copy()
    out["abs_error"] = (out["forecast_var"] - out["realised_var"]).abs()
    out["qlike"] = qlike(out["realised_var"], out["forecast_var"])
    out["ratio"] = out["realised_var"] / out["forecast_var"]
    return out


# ── regression with Newey–West standard errors ────────────────────────────────


def newey_west_ols(
    y: np.ndarray, x: np.ndarray, lags: int = DEFAULT_NW_LAGS
) -> tuple[np.ndarray, np.ndarray, float]:
    """OLS of ``y`` on ``[1, x]`` with Newey–West (Bartlett) standard errors.

    Returns:
        ``(coefficients, standard_errors, r_squared)`` with coefficients
        ordered ``[intercept, slope]``. With ``lags=0`` the standard errors
        are White's heteroskedasticity-robust (HC0) errors.
    """
    y = np.asarray(y, dtype=float)
    design = np.column_stack([np.ones(len(y)), np.asarray(x, dtype=float)])
    xtx_inv = np.linalg.inv(design.T @ design)
    beta = xtx_inv @ design.T @ y
    resid = y - design @ beta
    scores = design * resid[:, None]
    meat = scores.T @ scores
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma = scores[lag:].T @ scores[:-lag]
        meat += weight * (gamma + gamma.T)
    cov = xtx_inv @ meat @ xtx_inv
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid**2).sum()) / tss if tss > 0 else float("nan")
    return beta, np.sqrt(np.diag(cov)), r2


def diebold_mariano(
    loss_a: np.ndarray | pd.Series, loss_b: np.ndarray | pd.Series, lags: int = DEFAULT_NW_LAGS
) -> tuple[float, float]:
    """Diebold–Mariano test of equal expected loss (two-sided).

    The statistic is the mean loss differential ``a - b`` divided by its
    Newey–West standard error; negative values favour ``a``.

    Returns:
        ``(statistic, p_value)``; ``(0.0, 1.0)`` if the losses are identical.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = len(d)
    dc = d - d.mean()
    lrv = float(dc @ dc) / n
    for lag in range(1, lags + 1):
        lrv += 2.0 * (1.0 - lag / (lags + 1.0)) * float(dc[lag:] @ dc[:-lag]) / n
    if lrv <= 0:
        return 0.0, 1.0
    stat = float(d.mean() / np.sqrt(lrv / n))
    return stat, float(2.0 * norm.sf(abs(stat)))


def criterion_a_table(
    forecasts: pd.DataFrame,
    by: Sequence[str] = ("estimator", "portfolio"),
    lags: int = DEFAULT_NW_LAGS,
    min_periods_mz: int = MIN_PERIODS_MZ,
) -> pd.DataFrame:
    """Criterion A metrics for each group of a forecasts table.

    Columns: ``n_periods``, ``mae``, ``qlike``, ``bias_ratio``; the log-log
    calibration regression ``mzlog_a``, ``mzlog_b``, ``mzlog_b_se``
    (Newey–West), ``mzlog_r2`` (tracking measure); and the levels regression
    ``mz_a``, ``mz_b``, ``mz_b_se``, ``mz_r2`` (conventional, appendix). The
    regression columns are NaN for groups with fewer than ``min_periods_mz``
    periods (e.g. the shock regimes), where a two-parameter regression is not
    meaningful.
    """
    data = add_losses(forecasts)
    rows = []
    for key, g in data.groupby(list(by), sort=False):
        enough = len(g) >= max(min_periods_mz, 3)
        nan_fit = (np.full(2, np.nan), np.full(2, np.nan), float("nan"))
        v = g["realised_var"].to_numpy()
        f = g["forecast_var"].to_numpy()
        beta, se, r2 = newey_west_ols(v, f, lags) if enough else nan_fit
        lbeta, lse, lr2 = newey_west_ols(np.log(v), np.log(f), lags) if enough else nan_fit
        rows.append(
            dict(zip(by, key if isinstance(key, tuple) else (key,), strict=True))
            | {
                "n_periods": len(g),
                "mae": float(g["abs_error"].mean()),
                "qlike": float(g["qlike"].mean()),
                "bias_ratio": float(g["ratio"].mean()),
                "mzlog_a": float(lbeta[0]),
                "mzlog_b": float(lbeta[1]),
                "mzlog_b_se": float(lse[1]),
                "mzlog_r2": lr2,
                "mz_a": float(beta[0]),
                "mz_b": float(beta[1]),
                "mz_b_se": float(se[1]),
                "mz_r2": r2,
            }
        )
    return pd.DataFrame(rows).set_index(list(by))


# ── Criterion B ───────────────────────────────────────────────────────────────


def simple_portfolio_returns(
    weights: pd.DataFrame, log_returns: pd.DataFrame, period_of_day: pd.Series
) -> pd.DataFrame:
    """Daily out-of-sample simple returns of every (estimator, portfolio) pair.

    Args:
        weights: Rebalance weights, index ``(estimator, portfolio, k)``,
            columns instruments.
        log_returns: Daily log-return panel.
        period_of_day: Holding period ``k`` of each out-of-sample day.

    Returns:
        DataFrame indexed by the out-of-sample days, columns MultiIndex
        ``(estimator, portfolio)``.
    """
    panel = log_returns.loc[period_of_day.index, weights.columns]
    simple = pd.DataFrame(
        np.expm1(panel.to_numpy(dtype=float)), index=panel.index, columns=panel.columns
    ).fillna(0.0)
    days_k = period_of_day.to_numpy()
    out: dict[tuple[str, str], np.ndarray] = {}
    for (est, port), w in weights.groupby(level=["estimator", "portfolio"], sort=False):
        w_k = w.droplevel(["estimator", "portfolio"]).reindex(days_k).to_numpy()
        out[(str(est), str(port))] = np.einsum("ij,ij->i", simple.to_numpy(), w_k)
    frame = pd.DataFrame(out, index=period_of_day.index)
    frame.columns = pd.MultiIndex.from_tuples(list(out), names=["estimator", "portfolio"])
    return frame


def excess_simple_returns(
    weights: pd.DataFrame,
    log_returns: pd.DataFrame,
    period_of_day: pd.Series,
    risk_free: str | None = DEFAULT_RISK_FREE,
) -> pd.DataFrame:
    """Daily simple portfolio returns minus the risk-free instrument's simple return.

    Same layout as :func:`simple_portfolio_returns`. With ``risk_free=None`` the
    raw simple returns are returned.
    """
    returns = simple_portfolio_returns(weights, log_returns, period_of_day)
    if risk_free is None:
        return returns
    rf = np.expm1(log_returns.loc[returns.index, risk_free].to_numpy(dtype=float))
    return returns.sub(rf, axis=0)


def turnover_series(
    weights: pd.DataFrame, log_returns: pd.DataFrame, period_of_day: pd.Series
) -> pd.Series:
    """Turnover at each rebalance after the first, per (estimator, portfolio).

    Turnover at rebalance ``k`` is ``Σ_i |w_{i,k} - w̃_{i,k-1}|``, where ``w̃``
    are the weights of period ``k-1`` after drifting with prices:
    ``w̃_i = w_i (1 + R_i) / Σ_j w_j (1 + R_j)`` with ``R_i`` the compounded
    simple return of asset ``i`` over period ``k-1``.

    Returns:
        Series indexed by ``(estimator, portfolio, k)`` for ``k >= 1``.
    """
    panel = log_returns.loc[period_of_day.index, weights.columns].fillna(0.0)
    growth = (
        pd.DataFrame(np.exp(panel.to_numpy(dtype=float)), index=panel.index, columns=panel.columns)
        .groupby(period_of_day.to_numpy())
        .prod()
    )  # 1 + R_i over each period
    values: dict[tuple[str, str, int], float] = {}
    for (est, port), w in weights.groupby(level=["estimator", "portfolio"], sort=False):
        wk = w.droplevel(["estimator", "portfolio"]).sort_index()
        ks = wk.index.to_numpy()
        for prev, cur in zip(ks[:-1], ks[1:], strict=True):
            grown = wk.loc[prev].to_numpy() * growth.loc[prev].to_numpy()
            total = grown.sum()
            if total <= 0:
                raise ValueError(f"{est}/{port}: portfolio value non-positive in period {prev}")
            drifted = grown / total
            values[(str(est), str(port), int(cur))] = float(
                np.abs(wk.loc[cur].to_numpy() - drifted).sum()
            )
    s = pd.Series(values, name="turnover")
    s.index = s.index.set_names(["estimator", "portfolio", "k"])
    return s


def criterion_b_table(
    weights: pd.DataFrame,
    log_returns: pd.DataFrame,
    period_of_day: pd.Series,
    risk_free: str | None = DEFAULT_RISK_FREE,
    periods: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Criterion B metrics per (estimator, portfolio).

    Args:
        weights, log_returns, period_of_day: As produced by the backtest.
        risk_free: Instrument whose simple return is subtracted (``None``: raw returns).
        periods: Restrict to these holding periods (e.g. one regime); ``None`` uses all.

    Returns:
        DataFrame indexed by ``(estimator, portfolio)`` with ``volatility``
        (annualised, excess returns), ``sharpe``, ``turnover`` (mean over
        rebalances after the first), ``n_eff``, ``max_abs_weight`` and
        ``gross_exposure`` (means over rebalances), and ``n_days``.
    """
    excess = excess_simple_returns(weights, log_returns, period_of_day, risk_free)
    turnover = turnover_series(weights, log_returns, period_of_day)
    w = weights
    if periods is not None:
        keep = set(periods)
        excess = excess.loc[period_of_day.isin(keep).to_numpy()]
        turnover = turnover[turnover.index.get_level_values("k").isin(keep)]
        w = weights[weights.index.get_level_values("k").isin(keep)]

    sd = excess.std(ddof=1)
    rows = []
    for col in excess.columns:
        est, port = str(col[0]), str(col[1])
        wk = w.xs((est, port), level=("estimator", "portfolio")).to_numpy()
        t_mean = (
            float(turnover.xs((est, port), level=("estimator", "portfolio")).mean())
            if len(turnover)
            else float("nan")
        )
        vol = float(sd[(est, port)]) * np.sqrt(TRADING_DAYS)
        mean = float(excess[(est, port)].mean()) * TRADING_DAYS
        rows.append(
            {
                "estimator": est,
                "portfolio": port,
                "n_days": len(excess),
                "volatility": vol,
                "sharpe": mean / vol if vol > 0 else float("nan"),
                "turnover": t_mean,
                "n_eff": float(np.mean(1.0 / (wk**2).sum(axis=1))),
                "max_abs_weight": float(np.mean(np.abs(wk).max(axis=1))),
                "gross_exposure": float(np.mean(np.abs(wk).sum(axis=1))),
            }
        )
    return pd.DataFrame(rows).set_index(["estimator", "portfolio"])


__all__ = [
    "DEFAULT_NW_LAGS",
    "DEFAULT_RISK_FREE",
    "MIN_PERIODS_MZ",
    "TRADING_DAYS",
    "add_losses",
    "criterion_a_table",
    "criterion_b_table",
    "diebold_mariano",
    "excess_simple_returns",
    "newey_west_ols",
    "qlike",
    "simple_portfolio_returns",
    "turnover_series",
]
