"""Run the full evaluation for one estimation-window length.

Usage::

    python -m portfolio_risk.evaluation.run --window 252
    python -m portfolio_risk.evaluation.run --window 504
    python -m portfolio_risk.evaluation.run --window 252 --estimators sample lw_std --reps 200

Steps: load the returns panel, run the backtest (reusing cached fits), save the
backtest result to ``data/results/w<window>/``, compute every Criterion A, B
and C table with :func:`~portfolio_risk.evaluation.report.evaluate`, write them
as CSV files to ``memo/tables/w<window>/``, and print a short summary.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from portfolio_risk.data.paths import DATA_DIR, PROCESSED_DIR, PROJECT_ROOT
from portfolio_risk.evaluation.backtest import BacktestResult, run_backtest
from portfolio_risk.evaluation.config import ESTIMATORS, RunConfig
from portfolio_risk.evaluation.regimes import ALL
from portfolio_risk.evaluation.report import evaluate, write_tables

logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--window", type=int, default=252, help="rolling estimation window (days)")
    parser.add_argument(
        "--estimators", nargs="+", choices=list(ESTIMATORS), default=list(ESTIMATORS)
    )
    parser.add_argument(
        "--reps", type=int, default=None, help="bootstrap replications (default: config)"
    )
    parser.add_argument("--returns", type=Path, default=PROCESSED_DIR / "returns.parquet")
    parser.add_argument("--cache", type=Path, default=DATA_DIR / "cache")
    parser.add_argument("--results", type=Path, default=DATA_DIR / "results")
    parser.add_argument("--tables", type=Path, default=PROJECT_ROOT / "memo" / "tables")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def summary(tables: dict[str, pd.DataFrame]) -> str:
    """Headline numbers: full-sample paired comparisons against the baseline."""
    c = tables["comparisons"]
    c = c[c["regime"] == ALL]
    lines = []
    for crit, label in [("A_qlike_diff", "QLIKE minus sample (negative = better)"),
                        ("B_vol_ratio", "volatility / sample (below 1 = better)")]:  # fmt: skip
        part = c[c["criterion"] == crit]
        for port, g in part.groupby("portfolio", sort=False):
            lines.append(f"\n{label} -- {port}")
            for _, r in g.iterrows():
                interval = f"[{r['lower']:+.4f}, {r['upper']:+.4f}]"
                lines.append(f"  {r['estimator']:8s} {r['estimate']:+.4f}  {interval}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> BacktestResult:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg_kwargs: dict[str, object] = {
        "window_length": args.window,
        "estimators": tuple(args.estimators),
    }
    if args.reps is not None:
        cfg_kwargs["bootstrap_reps"] = args.reps
    config = RunConfig(**cfg_kwargs)  # type: ignore[arg-type]

    returns = pd.read_parquet(args.returns)
    logger.info("Returns: %d days x %d instruments", *returns.shape)
    result = run_backtest(returns, config, cache_dir=args.cache)
    run_dir = args.results / f"w{args.window}"
    result.save(run_dir)
    logger.info("Saved backtest result: %s", run_dir)

    tables = evaluate(result, returns)
    written = write_tables(tables, args.tables / f"w{args.window}")
    logger.info("Wrote %d tables to %s", len(written), args.tables / f"w{args.window}")
    print(summary(tables))
    return result


if __name__ == "__main__":
    main()
