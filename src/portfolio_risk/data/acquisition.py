"""yfinance fetcher and end-to-end data pipeline.

Fetches adjusted-close prices for the 75-instrument universe, cleans
against the NYSE calendar, computes log-returns, flags anomalies, and
writes the parquet cache plus the reproducibility manifest.

Usage from the command line::

    python -m portfolio_risk.data.acquisition
    python -m portfolio_risk.data.acquisition --start 2010-01-04 --end 2024-12-31
    python -m portfolio_risk.data.acquisition --verbose

Programmatic entry point::

    from portfolio_risk.data.acquisition import run_pipeline
    run_pipeline(start=date(2010, 1, 4), end=date(2024, 12, 31))

Fetch strategy: always fetches the full window. This deviates from the
design document's incremental refresh (§6) in favour of correctness:
yfinance retroactively adjusts historical prices when new dividends or
splits are announced, so simple append can produce inconsistent series.
A 15-year * 75-instrument full fetch takes ~30 seconds in practice, well
within the "weekly-refresh" cadence this project needs. Incremental
refresh with reconciliation is left as future work.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from typing import cast

import pandas as pd
import yfinance as yf

from portfolio_risk.data import calendar as cal
from portfolio_risk.data import manifest as mf
from portfolio_risk.data import paths, universe
from portfolio_risk.data.cleaning import clean_prices, flag_return_outliers
from portfolio_risk.data.returns import log_returns

logger = logging.getLogger(__name__)

DEFAULT_START: date = date(2010, 1, 4)
DEFAULT_END: date = date(2024, 12, 31)
DEFAULT_BATCH_SIZE: int = 25
DEFAULT_RETRIES: int = 3
DEFAULT_RETRY_BACKOFF: float = 1.0

# Type alias for the yfinance download function, to make monkeypatching in
# tests explicit and keep the module imports thin.
YfDownloader = Callable[[list[str], date, date], pd.DataFrame]


def _default_downloader(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """Thin wrapper around ``yfinance.download`` with our fixed options.

    yfinance's ``end`` argument is exclusive; we add one day so the caller
    can pass an inclusive end date.
    """
    return cast(
        pd.DataFrame,
        yf.download(
            tickers,
            start=start.isoformat(),
            end=(end + pd.Timedelta(days=1)).isoformat(),
            auto_adjust=True,
            group_by="column",
            progress=False,
            threads=True,
        ),
    )


def fetch_prices(
    tickers: list[str],
    start: date,
    end: date,
    batch_size: int = DEFAULT_BATCH_SIZE,
    retries: int = DEFAULT_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
    downloader: YfDownloader | None = None,
) -> pd.DataFrame:
    """Fetch adjusted-close prices from yfinance in batches.

    Args:
        tickers: Ticker symbols to fetch.
        start: Inclusive first date.
        end: Inclusive last date.
        batch_size: Number of tickers per yfinance call.
        retries: Total attempts per batch on failure (including the first).
        retry_backoff: Base seconds for exponential backoff between retries.
        downloader: Function with the ``(tickers, start, end) -> DataFrame``
            signature. Overridable for tests to avoid real network calls.

    Returns:
        Wide-form DataFrame indexed by date, columns are tickers in the
        input order, values are auto-adjusted close prices. Any ticker
        that could not be fetched (delisted, invalid, etc.) appears as
        an all-NaN column.
    """
    if not tickers:
        raise ValueError("tickers must be a non-empty list")
    if end < start:
        raise ValueError(f"end ({end}) is before start ({start})")

    download = downloader if downloader is not None else _default_downloader
    frames: list[pd.DataFrame] = []

    n_batches = (len(tickers) + batch_size - 1) // batch_size
    for batch_idx in range(n_batches):
        batch = tickers[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        logger.info("Fetching batch %d/%d (%d tickers)", batch_idx + 1, n_batches, len(batch))
        raw = _fetch_batch_with_retry(batch, start, end, download, retries, retry_backoff)
        frames.append(_extract_close(raw, batch))

    combined = pd.concat(frames, axis=1)
    combined = combined.reindex(columns=tickers)
    combined.index = pd.DatetimeIndex(combined.index).tz_localize(None)
    return combined


def _fetch_batch_with_retry(
    tickers: list[str],
    start: date,
    end: date,
    download: YfDownloader,
    retries: int,
    backoff: float,
) -> pd.DataFrame:
    """Download one batch with exponential-backoff retry."""
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return download(tickers, start, end)
        except Exception as exc:  # noqa: BLE001 -- yfinance raises many types
            last_exc = exc
            wait = backoff * (2 ** (attempt - 1))
            logger.warning(
                "Batch fetch attempt %d/%d failed (%s); retrying in %.1fs",
                attempt,
                retries,
                exc,
                wait,
            )
            time.sleep(wait)
    assert last_exc is not None
    raise RuntimeError(f"All {retries} attempts failed for batch {tickers}") from last_exc


def _extract_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Pull the 'Close' slice out of yfinance's MultiIndex or flat-column result.

    yfinance's shape depends on how many tickers you request:
    - Multi-ticker: columns are a MultiIndex like ``('Close', 'AAPL')``.
    - Single-ticker: columns are flat like ``'Close'``.
    """
    if raw.empty:
        return pd.DataFrame(index=raw.index, columns=tickers, dtype=float)

    close: pd.DataFrame
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise KeyError(f"'Close' column not found in fetched data (levels {raw.columns})")
        close = cast(pd.DataFrame, raw["Close"])
    else:
        if "Close" not in raw.columns:
            raise KeyError(f"'Close' column not found in fetched data ({list(raw.columns)})")
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})

    return close


def run_pipeline(
    start: date = DEFAULT_START,
    end: date | None = None,
    downloader: YfDownloader | None = None,
) -> dict[str, Path]:
    """Run the full data pipeline end to end.

    Steps:
        1. Fetch adjusted-close prices for the universe.
        2. Save raw panel to ``data/raw/prices.parquet``.
        3. Align to NYSE calendar; drop universe-wide multi-day gaps.
        4. Save cleaned panel to ``data/processed/prices.parquet``.
        5. Compute log-returns; save to ``data/processed/returns.parquet``.
        6. Flag anomalies; save to ``data/processed/anomalies.csv``.
        7. Save dropped-dates log to ``data/processed/dropped_dates.csv``.
        8. Write reproducibility manifest to ``data/manifest.yaml``.

    Args:
        start: Inclusive first date.
        end: Inclusive last date. Defaults to ``DEFAULT_END`` when None.
        downloader: Testing hook overriding the yfinance downloader.

    Returns:
        Mapping of artefact name to the path it was written to.
    """
    end = end or DEFAULT_END
    logger.info("Pipeline run: %s to %s (%d instruments)", start, end, len(universe.UNIVERSE))
    paths.ensure_data_dirs()

    tickers = universe.all_tickers()
    raw_prices = fetch_prices(tickers, start, end, downloader=downloader)
    raw_path = paths.RAW_DIR / "prices.parquet"
    raw_prices.to_parquet(raw_path)
    logger.info("Wrote raw prices: %s (%d rows, %d cols)", raw_path, *raw_prices.shape)

    calendar = cal.nyse_trading_days(start, end)
    cleaning_result = clean_prices(raw_prices, calendar)
    logger.info(
        "Cleaned panel: %d rows, %d dropped dates",
        len(cleaning_result.prices),
        len(cleaning_result.dropped_dates),
    )

    cleaned_prices_path = paths.PROCESSED_DIR / "prices.parquet"
    cleaning_result.prices.to_parquet(cleaned_prices_path)

    returns = log_returns(cleaning_result.prices.dropna(how="all"))
    returns_path = paths.PROCESSED_DIR / "returns.parquet"
    returns.to_parquet(returns_path)
    logger.info("Wrote returns: %s (%d rows, %d cols)", returns_path, *returns.shape)

    anomalies = flag_return_outliers(returns)
    anomalies_path = paths.PROCESSED_DIR / "anomalies.csv"
    anomalies.to_csv(anomalies_path, index=False)
    logger.info("Flagged %d anomalies -> %s", len(anomalies), anomalies_path)

    dropped_dates_path = paths.PROCESSED_DIR / "dropped_dates.csv"
    pd.DataFrame({"date": cleaning_result.dropped_dates}).to_csv(dropped_dates_path, index=False)

    manifest = mf.build_manifest(
        window_start=start,
        window_end=end,
        artefacts={
            "raw_prices": raw_path,
            "cleaned_prices": cleaned_prices_path,
            "returns": returns_path,
        },
        fetched_at=datetime.now().astimezone(),
    )
    mf.write_manifest(manifest, paths.MANIFEST_PATH)
    logger.info("Wrote manifest: %s", paths.MANIFEST_PATH)

    return {
        "raw_prices": raw_path,
        "cleaned_prices": cleaned_prices_path,
        "returns": returns_path,
        "anomalies": anomalies_path,
        "dropped_dates": dropped_dates_path,
        "manifest": paths.MANIFEST_PATH,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: parse args, configure logging, run the pipeline."""
    parser = argparse.ArgumentParser(
        prog="python -m portfolio_risk.data.acquisition",
        description="Fetch, clean, and cache the 75-instrument study universe.",
    )
    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        default=DEFAULT_START,
        help=f"Inclusive start date, YYYY-MM-DD (default: {DEFAULT_START})",
    )
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        default=DEFAULT_END,
        help=f"Inclusive end date, YYYY-MM-DD (default: {DEFAULT_END})",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        run_pipeline(start=args.start, end=args.end)
    except Exception:
        logger.exception("Pipeline failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_END",
    "DEFAULT_RETRIES",
    "DEFAULT_RETRY_BACKOFF",
    "DEFAULT_START",
    "YfDownloader",
    "fetch_prices",
    "main",
    "run_pipeline",
]
