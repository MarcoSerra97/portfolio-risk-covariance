"""Backtest loop: fit every estimator at every rebalance, form portfolios, record outcomes.

Implements ``docs/02-evaluation-design.md`` §4–§6. For one run (one rolling
window length), at each rebalance of the schedule:

1. The run's asset set is the instruments with complete data over the rolling
   estimation window (shared by every estimator in the run).
2. Each estimator is fitted on its window: the last ``window_length`` days, or
   for expanding-window estimators (DCC) all days from ``config.dcc_start``,
   or from the first day on which every instrument in the asset set trades,
   whichever is later. (On the study's data all 75 instruments trade from
   ``dcc_start``, so the two coincide.)
3. Each portfolio is built from the fitted covariance.
4. For each (estimator, portfolio) pair the engine records the forecast
   variance of the holding-period return, ``H · w' Σ̂ w``, the realised
   variance ``Σ_t (w' r_t)²`` over the holding period, the weights, and the
   daily out-of-sample portfolio returns ``w' r_t`` (weights held constant).

Each fit also records numerical diagnostics (condition numbers and extreme
eigenvalues of the covariance and of the implied correlation matrix, plus the
estimator's own fitted quantities: shrinkage intensity, number of MP signal
eigenvalues, QIS eigenvalue shrinkage, DCC parameters and GARCH fallbacks).

Fitted covariances and diagnostics are cached on disk, keyed by estimator,
window length, holding period, a fingerprint of the returns data and the cache
format version, so metrics and figures can be recomputed without refitting,
and re-fetched data is never matched with stale fits.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_risk.evaluation.config import ESTIMATORS, RunConfig, make_estimator
from portfolio_risk.evaluation.config import portfolio_builders as _portfolio_builders
from portfolio_risk.evaluation.schedule import HoldingPeriod, Schedule, build_schedule
from portfolio_risk.evaluation.schedule import eligible_assets as _eligible_assets

logger = logging.getLogger(__name__)

FORECAST_COLUMNS: list[str] = [
    "window_length",
    "estimator",
    "portfolio",
    "k",
    "rebalance",
    "regime",
    "n_assets",
    "forecast_var",
    "realised_var",
]


# ── data fingerprint and cache ────────────────────────────────────────────────


def data_fingerprint(returns: pd.DataFrame) -> str:
    """Short SHA-256 fingerprint of the returns panel (index, columns and values)."""
    h = hashlib.sha256()
    h.update(np.asarray(returns.index, dtype="datetime64[ns]").view(np.int64).tobytes())
    h.update("|".join(map(str, returns.columns)).encode())
    h.update(np.ascontiguousarray(returns.to_numpy(dtype=float)).tobytes())
    return h.hexdigest()[:12]


#: Bumped whenever the cache contents change, so older cache files are ignored.
CACHE_VERSION: int = 2

Diagnostics = dict[int, dict[str, float]]


def _cache_path(cache_dir: Path, estimator: str, config: RunConfig, fingerprint: str) -> Path:
    name = f"{estimator}_w{config.window_length}_h{config.holding}_{fingerprint}"
    return cache_dir / f"{name}_v{CACHE_VERSION}.npz"


def _save_cache(path: Path, covs: dict[int, pd.DataFrame], diags: Diagnostics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for k, cov in covs.items():
        arrays[f"cov_{k}"] = cov.to_numpy()
        arrays[f"assets_{k}"] = np.array([str(c) for c in cov.columns])
    arrays["periods"] = np.array(sorted(covs))
    arrays["diagnostics_json"] = np.array(json.dumps({str(k): v for k, v in diags.items()}))
    np.savez_compressed(path, **arrays)  # type: ignore[arg-type]


def _load_cache(path: Path) -> tuple[dict[int, pd.DataFrame], Diagnostics]:
    with np.load(path, allow_pickle=False) as data:
        covs: dict[int, pd.DataFrame] = {}
        for k in data["periods"].tolist():
            assets = [str(a) for a in data[f"assets_{k}"]]
            covs[int(k)] = pd.DataFrame(data[f"cov_{k}"], index=assets, columns=assets)
        raw = json.loads(str(data["diagnostics_json"]))
    return covs, {int(k): v for k, v in raw.items()}


# ── diagnostics ───────────────────────────────────────────────────────────────


def estimator_diagnostics(est: object, cov: pd.DataFrame, n_obs: int) -> dict[str, float]:
    """Numerical diagnostics of one fit.

    Always: ``n_obs``, ``n_assets``, extreme eigenvalues and condition number
    of the covariance and of the implied correlation matrix (the latter is
    free of volatility scale). When the estimator exposes them: ``shrinkage``
    (Ledoit-Wolf), ``n_signal`` and ``mp_edge`` (MP), ``sample_eig_min`` and
    ``shrunk_eig_min`` (QIS; MP gives the sample value), ``dcc_a``, ``dcc_b``,
    ``dcc_converged``, ``n_fallbacks`` and ``garch_persistence_mean`` (DCC).
    """
    m = cov.to_numpy(dtype=float)
    sd = np.sqrt(np.diag(m))
    corr = m / np.outer(sd, sd)
    ev = np.linalg.eigvalsh(m)
    evc = np.linalg.eigvalsh(corr)
    out: dict[str, float] = {
        "n_obs": float(n_obs),
        "n_assets": float(m.shape[0]),
        "cov_eig_min": float(ev[0]),
        "cov_eig_max": float(ev[-1]),
        "cov_cond": float(ev[-1] / ev[0]),
        "corr_eig_min": float(evc[0]),
        "corr_eig_max": float(evc[-1]),
        "corr_cond": float(evc[-1] / evc[0]),
    }
    simple = {"shrinkage_": "shrinkage", "n_signal_": "n_signal", "edge_": "mp_edge",
              "dcc_a_": "dcc_a", "dcc_b_": "dcc_b", "dcc_converged_": "dcc_converged"}  # fmt: skip
    for attr, key in simple.items():
        if hasattr(est, attr):
            out[key] = float(getattr(est, attr))
    if hasattr(est, "eigenvalues_"):
        out["sample_eig_min"] = float(np.min(est.eigenvalues_))
    if hasattr(est, "shrunk_eigenvalues_"):
        out["shrunk_eig_min"] = float(np.min(est.shrunk_eigenvalues_))
    if hasattr(est, "fallbacks_"):
        out["n_fallbacks"] = float(len(est.fallbacks_))
    if hasattr(est, "garch_params_"):
        out["garch_persistence_mean"] = float(est.garch_params_["persistence"].mean())
    return out


# ── fitting ───────────────────────────────────────────────────────────────────


def run_asset_set(
    returns: pd.DataFrame, schedule: Schedule, period: HoldingPeriod, window: int
) -> list[str]:
    """Instruments with complete data over the run's rolling window at ``period``."""
    return _eligible_assets(returns, *schedule.rolling_window(period, window))


def _first_complete_position(panel: pd.DataFrame, end: int) -> int:
    """First row position from which every column has data through ``end``."""
    missing = panel.iloc[: end + 1].isna().to_numpy()
    rows_with_gap = np.flatnonzero(missing.any(axis=1))
    return int(rows_with_gap[-1]) + 1 if rows_with_gap.size else 0


def fit_covariances(
    returns: pd.DataFrame,
    schedule: Schedule,
    config: RunConfig,
    estimator: str,
    cache_dir: Path | None = None,
) -> tuple[dict[int, pd.DataFrame], Diagnostics]:
    """Fitted covariance and diagnostics for every period of the schedule, for one estimator.

    Loads from ``cache_dir`` when a matching cache file exists; otherwise fits
    and (if ``cache_dir`` is given) saves.

    Raises:
        ValueError: If an estimation window contains missing values for the
            run's asset set.
    """
    path = None
    if cache_dir is not None:
        path = _cache_path(cache_dir, estimator, config, data_fingerprint(returns))
        if path.exists():
            logger.info("Loaded cached fits: %s", path.name)
            return _load_cache(path)

    spec = ESTIMATORS[estimator]
    covs: dict[int, pd.DataFrame] = {}
    diags: Diagnostics = {}
    for p in schedule.periods:
        assets = run_asset_set(returns, schedule, p, config.window_length)
        if spec.window == "rolling":
            start, end = schedule.rolling_window(p, config.window_length)
        else:
            start, end = schedule.expanding_window(p, config.dcc_start)
            start = max(start, _first_complete_position(returns[assets], end))
        window = returns.iloc[start : end + 1][assets]
        if bool(window.isna().to_numpy().any()):
            missing = list(window.columns[window.isna().any().to_numpy()])
            raise ValueError(f"{estimator}, period {p.k}: missing data for {missing} in the window")
        est = make_estimator(estimator, config).fit(window)
        covs[p.k] = est.covariance_
        diags[p.k] = estimator_diagnostics(est, est.covariance_, len(window))
        logger.debug(
            "%s: fitted period %d (%d obs, %d assets)", estimator, p.k, len(window), len(assets)
        )

    if path is not None:
        _save_cache(path, covs, diags)
        logger.info("Saved fits: %s", path.name)
    return covs, diags


# ── results ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BacktestResult:
    """Everything the criteria need from one run.

    Attributes:
        config: The run configuration.
        forecasts: One row per (estimator, portfolio, period), columns
            :data:`FORECAST_COLUMNS`.
        daily_returns: Out-of-sample daily portfolio log-returns; index is the
            trading days of all holding periods, columns a MultiIndex
            ``(estimator, portfolio)``.
        weights: Rebalance weights; index MultiIndex ``(estimator, portfolio, k)``,
            columns all instruments (zero where not held or not eligible).
        period_of_day: Holding-period number ``k`` for each out-of-sample day.
        diagnostics: Per-fit numerical diagnostics; index ``(estimator, k)``,
            one column per diagnostic (NaN where an estimator does not have it).
    """

    config: RunConfig
    forecasts: pd.DataFrame
    daily_returns: pd.DataFrame
    weights: pd.DataFrame
    period_of_day: pd.Series
    diagnostics: pd.DataFrame

    def save(self, directory: Path) -> None:
        """Write the result tables to ``directory`` (parquet) plus the window length (JSON)."""
        directory.mkdir(parents=True, exist_ok=True)
        self.forecasts.to_parquet(directory / "forecasts.parquet")
        flat = self.daily_returns.copy()
        flat.columns = [f"{c[0]}|{c[1]}" for c in flat.columns]
        flat.to_parquet(directory / "daily_returns.parquet")
        self.weights.to_parquet(directory / "weights.parquet")
        self.period_of_day.to_frame("k").to_parquet(directory / "period_of_day.parquet")
        self.diagnostics.to_parquet(directory / "diagnostics.parquet")
        meta = {"window_length": self.config.window_length, "holding": self.config.holding}
        (directory / "run.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: Path, config: RunConfig) -> BacktestResult:
        """Read a result written by :meth:`save`; ``config`` must match its run."""
        meta = json.loads((directory / "run.json").read_text())
        if meta != {"window_length": config.window_length, "holding": config.holding}:
            raise ValueError(f"saved run {meta} does not match the given config")
        daily = pd.read_parquet(directory / "daily_returns.parquet")
        daily.columns = pd.MultiIndex.from_tuples(
            [tuple(c.split("|", 1)) for c in daily.columns], names=["estimator", "portfolio"]
        )
        return cls(
            config=config,
            forecasts=pd.read_parquet(directory / "forecasts.parquet"),
            daily_returns=daily,
            weights=pd.read_parquet(directory / "weights.parquet"),
            period_of_day=pd.read_parquet(directory / "period_of_day.parquet")["k"],
            diagnostics=pd.read_parquet(directory / "diagnostics.parquet"),
        )


def run_backtest(
    returns: pd.DataFrame,
    config: RunConfig,
    cache_dir: Path | None = None,
) -> BacktestResult:
    """Run the full backtest for one configuration.

    Args:
        returns: Daily log-return panel (instruments in columns), NaN before
            each instrument's first observation.
        config: Run configuration.
        cache_dir: Directory for cached fits; ``None`` disables caching.

    Returns:
        A :class:`BacktestResult`.

    Raises:
        ValueError: If a holding period contains missing returns for a held asset.
        RuntimeError: Re-raised from a portfolio construction, with context.
    """
    schedule = build_schedule(
        pd.DatetimeIndex(returns.index), config.first_rebalance, config.holding
    )
    builders = _portfolio_builders(config)
    h = config.holding

    rows: list[dict[str, object]] = []
    daily_cols: dict[tuple[str, str], list[pd.Series]] = {}
    weight_rows: dict[tuple[str, str, int], pd.Series] = {}
    diag_rows: dict[tuple[str, int], dict[str, float]] = {}

    for est_name in config.estimators:
        logger.info("Running estimator %s (window %d)", est_name, config.window_length)
        covs, diags = fit_covariances(returns, schedule, config, est_name, cache_dir)
        for k, d in diags.items():
            diag_rows[(est_name, k)] = d
        for p in schedule.periods:
            cov = covs[p.k]
            assets = list(cov.columns)
            hold = returns.iloc[p.start_pos : p.end_pos + 1][assets]
            if bool(hold.isna().to_numpy().any()):
                raise ValueError(f"period {p.k}: missing returns in the holding period")
            for port_name, build in builders.items():
                try:
                    w = build(cov)
                except RuntimeError as err:
                    raise RuntimeError(f"{est_name} / {port_name}, period {p.k}: {err}") from err
                wv = w.reindex(assets).to_numpy(dtype=float)
                daily = pd.Series(hold.to_numpy() @ wv, index=hold.index)
                rows.append(
                    {
                        "window_length": config.window_length,
                        "estimator": est_name,
                        "portfolio": port_name,
                        "k": p.k,
                        "rebalance": p.rebalance,
                        "regime": p.regime,
                        "n_assets": len(assets),
                        "forecast_var": float(h * wv @ cov.to_numpy() @ wv),
                        "realised_var": float((daily**2).sum()),
                    }
                )
                daily_cols.setdefault((est_name, port_name), []).append(daily)
                weight_rows[(est_name, port_name, p.k)] = w.reindex(returns.columns, fill_value=0.0)

    forecasts = pd.DataFrame(rows, columns=FORECAST_COLUMNS)
    keys = list(daily_cols)
    daily_returns = pd.concat([pd.concat(daily_cols[key]) for key in keys], axis=1)
    daily_returns.columns = pd.MultiIndex.from_tuples(keys, names=["estimator", "portfolio"])
    weights = pd.DataFrame(weight_rows).T
    weights.index = weights.index.set_names(["estimator", "portfolio", "k"])
    period_of_day = pd.Series(
        np.concatenate([np.full(p.end_pos - p.start_pos + 1, p.k) for p in schedule.periods]),
        index=returns.index[schedule.periods[0].start_pos : schedule.periods[-1].end_pos + 1],
        name="k",
    )
    diagnostics = pd.DataFrame.from_dict(diag_rows, orient="index")
    diagnostics.index = pd.MultiIndex.from_tuples(list(diag_rows), names=["estimator", "k"])
    return BacktestResult(
        config=config,
        diagnostics=diagnostics,
        forecasts=forecasts,
        daily_returns=daily_returns,
        weights=weights,
        period_of_day=period_of_day,
    )


__all__ = [
    "FORECAST_COLUMNS",
    "CACHE_VERSION",
    "BacktestResult",
    "data_fingerprint",
    "estimator_diagnostics",
    "fit_covariances",
    "run_asset_set",
    "run_backtest",
]
