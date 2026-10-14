"""Figures of the evaluation results (first batch).

Every function reads a saved :class:`~portfolio_risk.evaluation.backtest.BacktestResult`
and/or the tables written by :func:`~portfolio_risk.evaluation.report.evaluate`,
returns a Matplotlib figure, and never refits anything. :func:`make_figures`
renders them all for one run and saves PNGs.

Usage::

    python -m portfolio_risk.evaluation.figures --window 252

Figures:

* ``forest_<criterion>``: each estimator's difference from the sample covariance
  with its 95% paired-bootstrap interval, one panel per portfolio, both windows;
* ``forecast_vs_realised``: forecast and realised volatility of equal weight
  over time, with close-ups of the COVID and tariff shocks;
* ``forecast_vs_realised_<portfolio>``: for each optimised portfolio, one panel
  per estimator, its forecast against the realised volatility of the portfolio
  that estimator itself built;
* ``cumulative_qlike``: cumulative QLIKE difference against the sample over time;
* ``qlike_by_regime``: mean QLIKE by regime and estimator (log colour scale);
* ``composition_<portfolio>``: net weight by asset group over time, per estimator;
* ``risk_parity_mechanism``: risk parity's equity weight against its volatility ratio;
* ``dcc_panel``: cDCC parameters, GARCH fallbacks and persistence across refits.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator  # noqa: E402

from portfolio_risk.data import universe  # noqa: E402
from portfolio_risk.data.paths import DATA_DIR, PROJECT_ROOT  # noqa: E402
from portfolio_risk.evaluation.backtest import BacktestResult  # noqa: E402
from portfolio_risk.evaluation.config import RunConfig  # noqa: E402
from portfolio_risk.evaluation.metrics import add_losses  # noqa: E402
from portfolio_risk.evaluation.regimes import ALL  # noqa: E402
from portfolio_risk.evaluation.schedule import UNCLASSIFIED  # noqa: E402
from portfolio_risk.portfolios.constructions import CONSTRUCTIONS  # noqa: E402

BASELINE = "sample"
CAP = 0.09  # half-height of interval end caps, in units of one estimator row
ANNUALISE = 252
ESTIMATOR_COLOURS: dict[str, str] = {
    "sample": "#444444",
    "lw_raw": "#e6a23c",
    "lw_std": "#c77c00",
    "mp": "#8e44ad",
    "qis_raw": "#5dade2",
    "qis_std": "#1f5f99",
    "dcc": "#c0392b",
}
REGIME_COLOURS: dict[str, str] = {
    "Calm I": "#e8f5e9",
    "Calm II": "#e8f5e9",
    "COVID shock": "#fdecea",
    "Tariff shock": "#fdecea",
    "Post-COVID": "#fff8e1",
    "Rates shock": "#fff3e0",
    "Unclassified": "#f0f0f0",
}
CRITERIA: dict[str, tuple[str, float]] = {
    "A_qlike_diff": ("QLIKE minus sample (negative = better forecasts)", 0.0),
    "A_mae_diff": ("MAE minus sample (negative = better forecasts)", 0.0),
    "B_vol_ratio": ("Volatility / sample (below 1 = less risky portfolio)", 1.0),
}
BUCKET_LABELS: dict[str, str] = {
    "equity": "Equities",
    "srf_etf": "Sector/regional/factor ETFs",
    "rate_etf": "Rates",
    "commodity_fx": "Commodities / FX",
    "vol_spread": "Vol / spread",
}


def _colour(est: str) -> str:
    return ESTIMATOR_COLOURS.get(est, "#7f8c8d")


def _period_dates(result: BacktestResult) -> pd.DataFrame:
    """Start, end, regime of each holding period, and ``until`` (next period's start).

    ``until`` makes consecutive periods contiguous on a time axis (no gaps over
    weekends between one period's last day and the next period's first).
    """
    days = result.period_of_day.reset_index()
    days.columns = ["date", "k"]
    span = days.groupby("k")["date"].agg(start="min", end="max")
    span["until"] = span["start"].shift(-1).fillna(span["end"].iloc[-1] + pd.Timedelta(days=1))
    regime = result.forecasts.drop_duplicates("k").set_index("k")["regime"]
    out: pd.DataFrame = span.join(regime)
    return out


def shade_regimes(ax: Axes, periods: pd.DataFrame) -> None:
    """Shade the background of a time axis by regime (contiguous across periods)."""
    for _, p in periods.iterrows():
        colour = REGIME_COLOURS.get(str(p["regime"]))
        if colour is not None:
            ax.axvspan(p["start"], p["until"], color=colour, lw=0, zorder=0)


def _percent_log_axis(ax: Axes) -> None:
    """Log-scale y axis labelled in percent."""
    ax.set_yscale("log")
    pct = FuncFormatter(lambda v, _: f"{100 * v:.2g}%")  # 0.8%, 1.5%, 2%, 12%
    ax.yaxis.set_major_formatter(pct)
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(2.0, 3.0, 5.0)))
    ax.yaxis.set_minor_formatter(pct)
    ax.tick_params(axis="y", which="minor", labelsize=8)


# ── 1. forest plots ───────────────────────────────────────────────────────────


def plot_forest(comparisons: dict[int, pd.DataFrame], criterion: str) -> Figure:
    """Point estimates and 95% intervals against the sample, per portfolio and window.

    Args:
        comparisons: Comparison tables keyed by window length (e.g. ``{252: df, 504: df}``).
        criterion: ``"A_qlike_diff"``, ``"A_mae_diff"`` or ``"B_vol_ratio"``.
    """
    label, ref = CRITERIA[criterion]
    first = next(iter(comparisons.values()))
    portfolios = list(dict.fromkeys(first.loc[first["criterion"] == criterion, "portfolio"]))
    fig, axes = plt.subplots(
        1, len(portfolios), figsize=(4.2 * len(portfolios), 4.2), squeeze=False
    )
    windows = sorted(comparisons)
    offsets = np.linspace(-0.15, 0.15, len(windows)) if len(windows) > 1 else [0.0]
    markers = ["o", "s", "^", "D"]
    for ax, port in zip(axes[0], portfolios, strict=True):
        for off, marker, w in zip(offsets, markers, windows, strict=False):
            c = comparisons[w]
            g = c[(c["criterion"] == criterion) & (c["portfolio"] == port) & (c["regime"] == ALL)]
            g = g[g["estimator"] != BASELINE]
            y = np.arange(len(g)) + off
            for yi, (_, r) in zip(y, g.iterrows(), strict=True):
                col = _colour(str(r["estimator"]))
                ax.plot([r["lower"], r["upper"]], [yi, yi], color=col, lw=1.6)
                for end in (r["lower"], r["upper"]):  # end caps: visible even for tiny intervals
                    ax.plot([end, end], [yi - CAP, yi + CAP], color=col, lw=1.6)
                ax.plot(
                    r["estimate"],
                    yi,
                    marker=marker,
                    color=col,
                    ms=6,
                    mfc="white" if w != windows[0] else col,
                )
            ax.set_yticks(np.arange(len(g)), list(g["estimator"]))
        ax.axvline(ref, color="black", lw=0.8, ls="--")
        ax.set_title(port.replace("_", " "))
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
    for marker, w in zip(markers, windows, strict=False):
        axes[0][-1].plot([], [], marker=marker, color="grey", ls="", label=f"{w}-day window",
                         mfc="grey" if w == windows[0] else "white")  # fmt: skip
    axes[0][-1].legend(loc="lower right", fontsize=8)
    fig.suptitle(label)
    fig.tight_layout()
    return fig


# ── 3. forecast vs realised volatility ────────────────────────────────────────


def plot_forecast_vs_realised(
    result: BacktestResult,
    portfolio: str = "equal_weight",
    estimators: Sequence[str] = ("sample", "qis_std", "dcc"),
) -> Figure:
    """Annualised forecast and realised volatility per holding period, with shock close-ups.

    Only for portfolios whose weights do not depend on the estimate (equal
    weight): every estimator then holds the same portfolio, so a single
    realised series is shared by all forecasts. For optimised portfolios use
    :func:`plot_forecast_vs_own_realised`.

    Raises:
        ValueError: If ``portfolio`` uses the covariance estimate.
    """
    if CONSTRUCTIONS[portfolio].uses_covariance:
        raise ValueError(
            f"{portfolio} differs across estimators; use plot_forecast_vs_own_realised"
        )
    f = result.forecasts[result.forecasts["portfolio"] == portfolio]
    periods = _period_dates(result)
    scale = ANNUALISE / result.config.holding
    fig = plt.figure(figsize=(13, 7.5))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.3, 1])
    axes = [fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    windows = [None, ("2019-12-01", "2020-09-30"), ("2025-01-15", "2025-07-31")]
    titles = [
        f"{portfolio.replace('_', ' ')}: forecast vs realised volatility",
        "COVID shock",
        "Tariff shock",
    ]
    realised = f.drop_duplicates("k").set_index("k")["realised_var"]
    series: dict[str, pd.Series] = {"realised": np.sqrt(realised * scale)}
    for est in estimators:
        fe = f[f["estimator"] == est].set_index("k")["forecast_var"]
        if len(fe):
            series[est] = np.sqrt(fe.reindex(periods.index) * scale)
    for ax, win, title in zip(axes, windows, titles, strict=True):
        shade_regimes(ax, periods)
        for name, vol in series.items():
            ax.step(periods["start"], vol, where="post",
                    color="black" if name == "realised" else _colour(name),
                    lw=1.4 if name == "realised" else 1.1, label=name)  # fmt: skip
        _percent_log_axis(ax)
        ax.set_title(title)
        ax.set_ylabel("annualised volatility")
        if win is not None:
            lo, hi = pd.Timestamp(win[0]), pd.Timestamp(win[1])
            ax.set_xlim(
                float(mdates.date2num(lo)),  # type: ignore[no-untyped-call]
                float(mdates.date2num(hi)),  # type: ignore[no-untyped-call]
            )
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))  # type: ignore[no-untyped-call]
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))  # type: ignore[no-untyped-call]
            in_win = (
                (periods["start"] >= lo - pd.Timedelta(days=40)) & (periods["start"] <= hi)
            ).to_numpy()
            vals = np.concatenate([v.to_numpy(dtype=float)[in_win] for v in series.values()])
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if vals.size:
                ax.set_ylim(vals.min() / 1.25, vals.max() * 1.25)
            _percent_log_axis(ax)
    axes[0].legend(ncol=len(estimators) + 1, fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def plot_forecast_vs_own_realised(result: BacktestResult, portfolio: str) -> Figure:
    """Each estimator's forecast against the realised volatility of its own portfolio.

    One panel per estimator, on a common log scale (annualised volatility), with
    regimes shaded. The gap between the realised (black) and forecast (coloured)
    lines is the estimator's forecast error for the portfolio it built:
    realised above forecast means the risk was underestimated (optimism).
    """
    f = result.forecasts[result.forecasts["portfolio"] == portfolio]
    periods = _period_dates(result)
    scale = ANNUALISE / result.config.holding
    ests = list(result.config.estimators)
    ncol = 4
    nrow = int(np.ceil(len(ests) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.0 * nrow), sharex=True,
                             sharey=True, squeeze=False)  # fmt: skip
    for ax, est in zip(axes.flat, ests, strict=False):
        fe = f[f["estimator"] == est].set_index("k").reindex(periods.index)
        shade_regimes(ax, periods)
        ax.step(periods["start"], np.sqrt(fe["realised_var"] * scale), where="post",
                color="black", lw=1.1, label="realised")  # fmt: skip
        ax.step(periods["start"], np.sqrt(fe["forecast_var"] * scale), where="post",
                color=_colour(est), lw=1.1, label="forecast")  # fmt: skip
        _percent_log_axis(ax)
        ax.set_title(est, fontsize=10)
    for ax in list(axes.flat)[len(ests) :]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(
        f"{portfolio.replace('_', ' ')}: forecast vs realised volatility of each estimator's "
        "own portfolio (annualised)"
    )
    fig.tight_layout()
    return fig


# ── 6. cumulative QLIKE difference ────────────────────────────────────────────


def plot_cumulative_qlike(result: BacktestResult) -> Figure:
    """Cumulative QLIKE of each estimator minus the sample's, per portfolio."""
    losses = add_losses(result.forecasts)
    periods = _period_dates(result)
    ports = list(result.config.portfolios)
    fig, axes = plt.subplots(len(ports), 1, figsize=(12, 2.6 * len(ports)), sharex=True)
    for ax, port in zip(np.atleast_1d(axes), ports, strict=True):
        t = losses[losses["portfolio"] == port].pivot(
            index="k", columns="estimator", values="qlike"
        )
        shade_regimes(ax, periods)
        for est in result.config.estimators:
            if est == BASELINE:
                continue
            ax.plot(periods.loc[t.index, "end"], (t[est] - t[BASELINE]).cumsum(),
                    color=_colour(est), lw=1.2, label=est)  # fmt: skip
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(port.replace("_", " "), fontsize=10)
        ax.set_ylabel("cumulative QLIKE diff.")
    np.atleast_1d(axes)[0].legend(ncol=6, fontsize=8, loc="lower left")
    fig.suptitle("Cumulative QLIKE minus sample (falling = better than the sample)")
    fig.tight_layout()
    return fig


# ── 7. QLIKE by regime ────────────────────────────────────────────────────────


def plot_qlike_by_regime(regime_table_a: pd.DataFrame, regime_order: Sequence[str]) -> Figure:
    """Mean QLIKE by regime (rows) and estimator (columns), one panel per portfolio."""
    t = regime_table_a.reset_index()
    ports = list(dict.fromkeys(t["portfolio"]))
    estimators = list(dict.fromkeys(t["estimator"]))
    fig, axes = plt.subplots(
        1, len(ports), figsize=(4.0 * len(ports) + 1.5, 4.2), squeeze=False, sharey=True
    )
    values = t["qlike"].to_numpy()
    norm = LogNorm(vmin=max(values[values > 0].min(), 1e-3), vmax=values.max())
    image = None
    for col, (ax, port) in enumerate(zip(axes[0], ports, strict=True)):
        m = t[t["portfolio"] == port].pivot(index="regime", columns="estimator", values="qlike")
        m = m.reindex(index=[r for r in regime_order if r in m.index], columns=estimators)
        image = ax.imshow(m.to_numpy(), norm=norm, cmap="magma_r", aspect="auto")
        ax.set_xticks(range(len(m.columns)), list(m.columns), rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(m.index)), list(m.index), fontsize=8)
        ax.tick_params(axis="y", labelleft=col == 0)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                v = m.iat[i, j]
                dark = float(norm(v)) > 0.55 if v > 0 else False
                ax.text(j, i, f"{v:.2g}", ha="center", va="center", fontsize=7,
                        color="white" if dark else "black")  # fmt: skip
        ax.set_title(port.replace("_", " "), fontsize=10)
    fig.suptitle("Mean QLIKE by regime (lower = better)")
    fig.tight_layout()
    if image is not None:
        fig.colorbar(image, ax=axes[0].tolist(), shrink=0.8, label="mean QLIKE (log scale)")
    return fig


# ── 9. composition ────────────────────────────────────────────────────────────


def plot_composition(
    result: BacktestResult, portfolio: str, by: str = "bucket", top: int = 6
) -> Figure:
    """Weights over time, one panel per estimator.

    Args:
        by: ``"bucket"`` for net weight by asset group, or ``"instrument"`` for
            the ``top`` instruments with the largest average absolute weight
            (across estimators) plus the net remainder as ``"other"``.
    """
    meta = universe.metadata_frame()
    w = result.weights.xs(portfolio, level="portfolio")
    periods = _period_dates(result)
    if by == "bucket":
        groups = meta.loc[w.columns, "bucket"]
        series_labels = BUCKET_LABELS
    elif by == "instrument":
        names = list(pd.Series(w.abs().mean()).sort_values(ascending=False).index[:top])
        groups = pd.Series([c if c in names else "other" for c in w.columns], index=w.columns)
        series_labels = {n: n for n in [*names, "other"]}
    else:
        raise ValueError(f"by must be 'bucket' or 'instrument', got {by!r}")
    ests = list(result.config.estimators)
    ncol = 4
    nrow = int(np.ceil(len(ests) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.0 * nrow), sharex=True, sharey=True,
                             squeeze=False)  # fmt: skip
    palette = plt.get_cmap("tab10")
    for ax, est in zip(axes.flat, ests, strict=False):
        net = w.loc[est].T.groupby(groups.to_numpy()).sum().T
        shade_regimes(ax, periods)
        for i, (key, lab) in enumerate(series_labels.items()):
            if key in net.columns:
                ax.step(periods.loc[net.index, "start"], net[key], where="post",
                        color="#999999" if key == "other" else palette(i), lw=1.1,
                        label=lab)  # fmt: skip
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(est, fontsize=10)
    for ax in list(axes.flat)[len(ests) :]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=7, loc="upper left")
    what = (
        "net weight by asset group" if by == "bucket" else f"weights of the {top} largest positions"
    )
    fig.suptitle(f"{portfolio.replace('_', ' ')}: {what}")
    fig.tight_layout()
    return fig


# ── 10. risk-parity mechanism ─────────────────────────────────────────────────


def equity_weight(result: BacktestResult, portfolio: str = "risk_parity") -> pd.Series:
    """Average weight in equities and equity ETFs, per estimator."""
    meta = universe.metadata_frame()
    w = pd.DataFrame(result.weights.xs(portfolio, level="portfolio"))
    eq = [t for t in w.columns if meta.loc[t, "bucket"] in ("equity", "srf_etf")]
    return w[eq].sum(axis=1).groupby(level="estimator", sort=False).mean()


def plot_risk_parity_mechanism(
    results: dict[int, BacktestResult], comparisons: dict[int, pd.DataFrame]
) -> Figure:
    """Risk parity: average equity weight against volatility relative to the sample."""
    fig, ax = plt.subplots(figsize=(6.5, 5))
    markers = ["o", "s"]
    for marker, w in zip(markers, sorted(results), strict=False):
        eq = equity_weight(results[w])
        c = comparisons[w]
        ratio = c[(c["criterion"] == "B_vol_ratio") & (c["portfolio"] == "risk_parity")
                  & (c["regime"] == ALL)].set_index("estimator")["estimate"]  # fmt: skip
        for est in eq.index:
            ax.scatter(eq[est], ratio[est], color=_colour(est), marker=marker, s=60,
                       facecolors=_colour(est) if marker == "o" else "white")  # fmt: skip
            if marker == "o":
                ax.annotate(est, (eq[est], ratio[est]), xytext=(5, 4), textcoords="offset points",
                            fontsize=8)  # fmt: skip
        ax.scatter([], [], marker=marker, color="grey", label=f"{w}-day window")
    ax.axhline(1, color="black", lw=0.8, ls="--")
    ax.set_xlabel("average weight in equities + equity ETFs")
    ax.set_ylabel("risk-parity volatility / sample")
    ax.set_title("Risk parity: more equity weight, more volatility")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ── 14. DCC panel ─────────────────────────────────────────────────────────────


def plot_dcc_panel(result: BacktestResult) -> Figure:
    """cDCC (a, b), persistence a + b, GARCH fallbacks and mean GARCH persistence over refits."""
    d = result.diagnostics.loc["dcc"]
    periods = _period_dates(result)
    x = periods.loc[d.index, "start"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5), sharex=True)
    ax = axes[0, 0]
    ax.plot(x, d["dcc_a"], label="a (reaction)")
    ax.plot(x, d["dcc_b"], label="b (memory)")
    ax.legend(fontsize=8)
    ax.set_title("DCC parameters")
    ax = axes[0, 1]
    persistence = d["dcc_a"] + d["dcc_b"]
    ax.plot(x, persistence, color="black")
    ax.set_title("correlation persistence a + b")
    half_life = np.log(0.5) / np.log(persistence.clip(upper=0.9999))
    twin = ax.twinx()
    twin.plot(x, half_life, color="grey", ls=":", lw=1)
    twin.set_ylabel("half-life (days)", color="grey")
    ax = axes[1, 0]
    ax.bar(x, d["n_fallbacks"], width=20, color="#c0392b")
    ax.set_title("GARCH fallbacks per refit (constant variance used)")
    ax = axes[1, 1]
    ax.plot(x, d["garch_persistence_mean"], color="#1f5f99")
    ax.set_title("mean GARCH persistence (alpha + beta, all assets)")
    for a in axes.flat:
        shade_regimes(a, periods)
        a.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ── rendering ─────────────────────────────────────────────────────────────────


def make_figures(
    results: dict[int, BacktestResult],
    tables: dict[int, dict[str, pd.DataFrame]],
    window: int,
    out_dir: Path,
) -> list[Path]:
    """Render the first batch of figures for ``window`` and save them as PNGs.

    Args:
        results: Backtest results keyed by window length (one or both windows).
        tables: For each window, the tables written by ``report.evaluate``
            (only ``comparisons`` and ``criterion_a_regimes`` are needed).
        window: The window whose single-run figures are drawn.
        out_dir: Output directory.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    result = results[window]
    comps = {w: t["comparisons"] for w, t in tables.items()}
    regime_order = [r for r in dict.fromkeys(_period_dates(result)["regime"]) if r != UNCLASSIFIED]
    figs: dict[str, Figure] = {
        "forest_qlike": plot_forest(comps, "A_qlike_diff"),
        "forest_mae": plot_forest(comps, "A_mae_diff"),
        "forest_volatility": plot_forest(comps, "B_vol_ratio"),
        "forecast_vs_realised": plot_forecast_vs_realised(
            result,
            estimators=[e for e in ("sample", "qis_std", "dcc") if e in result.config.estimators],
        ),
        "cumulative_qlike": plot_cumulative_qlike(result),
        "qlike_by_regime": plot_qlike_by_regime(
            tables[window]["criterion_a_regimes"], regime_order
        ),
        "composition_min_variance_long_short": plot_composition(
            result, "min_variance_long_short", by="instrument"
        ),
        "composition_risk_parity": plot_composition(result, "risk_parity"),
        "risk_parity_mechanism": plot_risk_parity_mechanism(results, comps),
    }
    for port in result.config.portfolios:
        if CONSTRUCTIONS[port].uses_covariance:
            figs[f"forecast_vs_realised_{port}"] = plot_forecast_vs_own_realised(result, port)
    if "dcc" in result.config.estimators:
        figs["dcc_panel"] = plot_dcc_panel(result)
    paths = []
    for name, fig in figs.items():
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def _load_window(
    w: int, results_dir: Path, tables_dir: Path
) -> tuple[BacktestResult, dict[str, pd.DataFrame]]:
    import json

    meta = json.loads((results_dir / f"w{w}" / "run.json").read_text())
    config = RunConfig(window_length=meta["window_length"], holding=meta["holding"])
    result = BacktestResult.load(results_dir / f"w{w}", config)
    estimators = tuple(dict.fromkeys(result.forecasts["estimator"]))
    result = BacktestResult(
        config=RunConfig(window_length=w, holding=meta["holding"], estimators=estimators),
        forecasts=result.forecasts, daily_returns=result.daily_returns, weights=result.weights,
        period_of_day=result.period_of_day, diagnostics=result.diagnostics,
    )  # fmt: skip
    tables = {
        "comparisons": pd.read_csv(tables_dir / f"w{w}" / "comparisons.csv", index_col=0),
        "criterion_a_regimes": pd.read_csv(
            tables_dir / f"w{w}" / "criterion_a_regimes.csv", index_col=[0, 1, 2]
        ),
    }
    return result, tables


def main(argv: Sequence[str] | None = None) -> list[Path]:
    parser = argparse.ArgumentParser(description="Render the result figures for one window.")
    parser.add_argument("--window", type=int, default=252)
    parser.add_argument("--results", type=Path, default=DATA_DIR / "results")
    parser.add_argument("--tables", type=Path, default=PROJECT_ROOT / "memo" / "tables")
    parser.add_argument("--figures", type=Path, default=PROJECT_ROOT / "memo" / "figures")
    args = parser.parse_args(argv)
    windows = [w for w in (252, 504) if (args.results / f"w{w}" / "run.json").exists()]
    if args.window not in windows:
        raise SystemExit(f"no saved results for window {args.window} in {args.results}")
    loaded = {w: _load_window(w, args.results, args.tables) for w in windows}
    paths = make_figures(
        {w: r for w, (r, _) in loaded.items()},
        {w: t for w, (_, t) in loaded.items()},
        args.window,
        args.figures / f"w{args.window}",
    )
    for p in paths:
        print(p)
    return paths


if __name__ == "__main__":
    main()


__all__ = [
    "equity_weight",
    "make_figures",
    "plot_composition",
    "plot_cumulative_qlike",
    "plot_dcc_panel",
    "plot_forecast_vs_own_realised",
    "plot_forecast_vs_realised",
    "plot_forest",
    "plot_qlike_by_regime",
    "plot_risk_parity_mechanism",
    "shade_regimes",
]
