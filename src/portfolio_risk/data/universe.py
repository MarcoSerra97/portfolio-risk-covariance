"""Universe of instruments for the portfolio-risk-covariance study.

Defines the 75-instrument panel across five buckets: 38 US equities, 15
sector / regional / factor ETFs, 8 rate ETFs, 8 commodities / FX, and 6
vol / spread / alternative-credit instruments.

The universe is fixed as of the design document (docs/01-data-acquisition-
design.md v1). It should not be modified without a corresponding revision
of that document.

Inception dates are set only for post-2010 launches, i.e. instruments that
are missing from the burn-in period (2010-2014) and must be handled with
care by rolling-window estimators. Pre-2010 launches have inception=None
and are assumed to have full coverage from the study start date. Declared
inception dates should be cross-checked against each series' first
observation in the fetched cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

import pandas as pd

Bucket = Literal["equity", "srf_etf", "rate_etf", "commodity_fx", "vol_spread"]

BUCKETS: tuple[Bucket, ...] = (
    "equity",
    "srf_etf",
    "rate_etf",
    "commodity_fx",
    "vol_spread",
)


@dataclass(frozen=True, slots=True)
class Instrument:
    """Metadata for a single instrument in the study universe.

    Attributes:
        ticker: yfinance ticker symbol, uppercase.
        name: Full instrument name.
        bucket: Top-level classification (see ``Bucket``).
        subgroup: Sub-classification within the bucket. For equities this
            is the GICS-style sector; for SRF ETFs it is one of
            "US Sector", "Regional", "Factor"; for other buckets it names
            the exposure type. Free-form but consistent within each bucket.
        inception: Instrument inception date. ``None`` when the instrument
            predates the study window (2010-01-04) and full coverage is
            assumed. Set only for post-2010 launches, which are absent
            from the burn-in period.
        notes: Optional free-form annotation (e.g. corporate-action history).
    """

    ticker: str
    name: str
    bucket: Bucket
    subgroup: str
    inception: date | None = None
    notes: str = ""


# ── Equities (38) ─────────────────────────────────────────────────────────

_TECH: tuple[Instrument, ...] = (
    Instrument("AAPL", "Apple Inc.", "equity", "Technology"),
    Instrument("MSFT", "Microsoft Corporation", "equity", "Technology"),
    Instrument("ORCL", "Oracle Corporation", "equity", "Technology"),
    Instrument("INTC", "Intel Corporation", "equity", "Technology"),
    Instrument("CSCO", "Cisco Systems, Inc.", "equity", "Technology"),
    Instrument(
        "IBM",
        "International Business Machines Corp.",
        "equity",
        "Technology",
        notes="Kyndryl spinoff Nov 2021 (adjusted close handles)",
    ),
    Instrument("ADBE", "Adobe Inc.", "equity", "Technology"),
    Instrument("QCOM", "QUALCOMM Incorporated", "equity", "Technology"),
)

_FINANCIALS: tuple[Instrument, ...] = (
    Instrument("JPM", "JPMorgan Chase & Co.", "equity", "Financials"),
    Instrument("BAC", "Bank of America Corporation", "equity", "Financials"),
    Instrument("WFC", "Wells Fargo & Company", "equity", "Financials"),
    Instrument("C", "Citigroup Inc.", "equity", "Financials"),
    Instrument("GS", "The Goldman Sachs Group, Inc.", "equity", "Financials"),
    Instrument("MS", "Morgan Stanley", "equity", "Financials"),
    Instrument("AXP", "American Express Company", "equity", "Financials"),
    Instrument("USB", "U.S. Bancorp", "equity", "Financials"),
)

_HEALTHCARE: tuple[Instrument, ...] = (
    Instrument(
        "JNJ",
        "Johnson & Johnson",
        "equity",
        "Healthcare",
        notes="Kenvue spinoff Aug 2023 (adjusted close handles)",
    ),
    Instrument(
        "PFE",
        "Pfizer Inc.",
        "equity",
        "Healthcare",
        notes="Viatris spinoff Nov 2020 (adjusted close handles)",
    ),
    Instrument(
        "MRK",
        "Merck & Co., Inc.",
        "equity",
        "Healthcare",
        notes="Organon spinoff Jun 2021 (adjusted close handles)",
    ),
    Instrument("LLY", "Eli Lilly and Company", "equity", "Healthcare"),
    Instrument("BMY", "Bristol-Myers Squibb Company", "equity", "Healthcare"),
    Instrument("AMGN", "Amgen Inc.", "equity", "Healthcare"),
    Instrument("GILD", "Gilead Sciences, Inc.", "equity", "Healthcare"),
    Instrument(
        "MDT",
        "Medtronic plc",
        "equity",
        "Healthcare",
        notes="Covidien inversion Jan 2015; reorganised to Ireland",
    ),
)

_INDUSTRIALS: tuple[Instrument, ...] = (
    Instrument("BA", "The Boeing Company", "equity", "Industrials"),
    Instrument("CAT", "Caterpillar Inc.", "equity", "Industrials"),
    Instrument(
        "MMM",
        "3M Company",
        "equity",
        "Industrials",
        notes="Solventum spinoff Apr 2024 (adjusted close handles)",
    ),
    Instrument(
        "HON",
        "Honeywell International Inc.",
        "equity",
        "Industrials",
        notes="Garrett + Resideo spinoffs Oct 2018",
    ),
    Instrument("UPS", "United Parcel Service, Inc.", "equity", "Industrials"),
    Instrument("LMT", "Lockheed Martin Corporation", "equity", "Industrials"),
    Instrument("UNP", "Union Pacific Corporation", "equity", "Industrials"),
)

_STAPLES: tuple[Instrument, ...] = (
    Instrument("PG", "The Procter & Gamble Company", "equity", "Consumer Staples"),
    Instrument("KO", "The Coca-Cola Company", "equity", "Consumer Staples"),
    Instrument("PEP", "PepsiCo, Inc.", "equity", "Consumer Staples"),
    Instrument("WMT", "Walmart Inc.", "equity", "Consumer Staples"),
    Instrument("MO", "Altria Group, Inc.", "equity", "Consumer Staples"),
    Instrument("CL", "Colgate-Palmolive Company", "equity", "Consumer Staples"),
    Instrument(
        "KMB",
        "Kimberly-Clark Corporation",
        "equity",
        "Consumer Staples",
        notes="Halyard Health spinoff Oct 2014",
    ),
)

EQUITIES: tuple[Instrument, ...] = (
    *_TECH,
    *_FINANCIALS,
    *_HEALTHCARE,
    *_INDUSTRIALS,
    *_STAPLES,
)


# ── Sector / regional / factor ETFs (15) ──────────────────────────────────

_SECTOR_ETFS: tuple[Instrument, ...] = (
    Instrument("XLK", "Technology Select Sector SPDR Fund", "srf_etf", "US Sector"),
    Instrument("XLF", "Financial Select Sector SPDR Fund", "srf_etf", "US Sector"),
    Instrument("XLV", "Health Care Select Sector SPDR Fund", "srf_etf", "US Sector"),
    Instrument("XLI", "Industrial Select Sector SPDR Fund", "srf_etf", "US Sector"),
    Instrument("XLP", "Consumer Staples Select Sector SPDR Fund", "srf_etf", "US Sector"),
    Instrument(
        "XLY",
        "Consumer Discretionary Select Sector SPDR Fund",
        "srf_etf",
        "US Sector",
    ),
    Instrument("XLE", "Energy Select Sector SPDR Fund", "srf_etf", "US Sector"),
    Instrument("XLB", "Materials Select Sector SPDR Fund", "srf_etf", "US Sector"),
    Instrument("XLU", "Utilities Select Sector SPDR Fund", "srf_etf", "US Sector"),
)

_REGIONAL_ETFS: tuple[Instrument, ...] = (
    Instrument("EFA", "iShares MSCI EAFE ETF", "srf_etf", "Regional"),
    Instrument("EEM", "iShares MSCI Emerging Markets ETF", "srf_etf", "Regional"),
    Instrument("VGK", "Vanguard FTSE Europe ETF", "srf_etf", "Regional"),
)

_FACTOR_ETFS: tuple[Instrument, ...] = (
    Instrument(
        "MTUM",
        "iShares MSCI USA Momentum Factor ETF",
        "srf_etf",
        "Factor",
        inception=date(2013, 4, 16),
        notes="Post-2010 launch; missing from burn-in period",
    ),
    Instrument("IWD", "iShares Russell 1000 Value ETF", "srf_etf", "Factor"),
    Instrument("IWF", "iShares Russell 1000 Growth ETF", "srf_etf", "Factor"),
)

SRF_ETFS: tuple[Instrument, ...] = (
    *_SECTOR_ETFS,
    *_REGIONAL_ETFS,
    *_FACTOR_ETFS,
)


# ── Rate ETFs (8) ─────────────────────────────────────────────────────────

RATE_ETFS: tuple[Instrument, ...] = (
    Instrument("SHY", "iShares 1-3 Year Treasury Bond ETF", "rate_etf", "Treasury Duration"),
    Instrument("IEF", "iShares 7-10 Year Treasury Bond ETF", "rate_etf", "Treasury Duration"),
    Instrument("TLT", "iShares 20+ Year Treasury Bond ETF", "rate_etf", "Treasury Duration"),
    Instrument("TIP", "iShares TIPS Bond ETF", "rate_etf", "Inflation"),
    Instrument(
        "LQD",
        "iShares iBoxx $ Investment Grade Corporate Bond ETF",
        "rate_etf",
        "IG Credit",
    ),
    Instrument(
        "HYG",
        "iShares iBoxx $ High Yield Corporate Bond ETF",
        "rate_etf",
        "HY Credit",
    ),
    Instrument("MBB", "iShares MBS ETF", "rate_etf", "Mortgage"),
    Instrument(
        "EMB",
        "iShares J.P. Morgan USD Emerging Markets Bond ETF",
        "rate_etf",
        "EM Sovereign",
    ),
)


# ── Commodities / FX (8) ──────────────────────────────────────────────────

COMMODITY_FX: tuple[Instrument, ...] = (
    Instrument("GLD", "SPDR Gold Shares", "commodity_fx", "Precious Metals"),
    Instrument("SLV", "iShares Silver Trust", "commodity_fx", "Precious Metals"),
    Instrument(
        "USO",
        "United States Oil Fund, LP",
        "commodity_fx",
        "Energy",
        notes="Roll methodology changed Apr 2020 after negative WTI event",
    ),
    Instrument(
        "DBC",
        "Invesco DB Commodity Index Tracking Fund",
        "commodity_fx",
        "Broad Commodity",
    ),
    Instrument("DBA", "Invesco DB Agriculture Fund", "commodity_fx", "Agriculture"),
    Instrument("UUP", "Invesco DB US Dollar Index Bullish Fund", "commodity_fx", "FX"),
    Instrument("FXE", "Invesco CurrencyShares Euro Trust", "commodity_fx", "FX"),
    Instrument("FXY", "Invesco CurrencyShares Japanese Yen Trust", "commodity_fx", "FX"),
)


# ── Vol / spread / alternative-credit (6) ─────────────────────────────────

VOL_SPREAD: tuple[Instrument, ...] = (
    Instrument(
        "VIXY",
        "ProShares VIX Short-Term Futures ETF",
        "vol_spread",
        "Vol Futures",
        inception=date(2011, 1, 3),
        notes="Post-2010 launch; missing from burn-in period",
    ),
    Instrument(
        "SPLV",
        "Invesco S&P 500 Low Volatility ETF",
        "vol_spread",
        "Low-Vol Equity",
        inception=date(2011, 5, 5),
        notes="Post-2010 launch; missing from burn-in period",
    ),
    Instrument(
        "BKLN",
        "Invesco Senior Loan ETF",
        "vol_spread",
        "Bank Loans",
        inception=date(2011, 3, 3),
        notes="Post-2010 launch; missing from burn-in period",
    ),
    Instrument(
        "CWB",
        "SPDR Bloomberg Convertible Securities ETF",
        "vol_spread",
        "Convertibles",
    ),
    Instrument(
        "PFF",
        "iShares Preferred and Income Securities ETF",
        "vol_spread",
        "Preferreds",
    ),
    Instrument("BIL", "SPDR Bloomberg 1-3 Month T-Bill ETF", "vol_spread", "Cash"),
)


# ── Full universe ─────────────────────────────────────────────────────────

UNIVERSE: tuple[Instrument, ...] = (
    *EQUITIES,
    *SRF_ETFS,
    *RATE_ETFS,
    *COMMODITY_FX,
    *VOL_SPREAD,
)

_BY_TICKER: dict[str, Instrument] = {inst.ticker: inst for inst in UNIVERSE}

# Import-time sanity checks: bucket counts and uniqueness.
assert len(EQUITIES) == 38, f"Expected 38 equities, got {len(EQUITIES)}"
assert len(SRF_ETFS) == 15, f"Expected 15 SRF ETFs, got {len(SRF_ETFS)}"
assert len(RATE_ETFS) == 8, f"Expected 8 rate ETFs, got {len(RATE_ETFS)}"
assert len(COMMODITY_FX) == 8, f"Expected 8 commodity/FX, got {len(COMMODITY_FX)}"
assert len(VOL_SPREAD) == 6, f"Expected 6 vol/spread, got {len(VOL_SPREAD)}"
assert len(UNIVERSE) == 75, f"Expected 75 instruments, got {len(UNIVERSE)}"
assert len(_BY_TICKER) == len(UNIVERSE), "Duplicate ticker in UNIVERSE"


# ── Accessors ─────────────────────────────────────────────────────────────


def all_tickers() -> list[str]:
    """Return all 75 tickers in canonical (bucket, subgroup) order."""
    return [inst.ticker for inst in UNIVERSE]


def get_instrument(ticker: str) -> Instrument:
    """Look up an instrument by ticker (case-insensitive).

    Raises:
        KeyError: If the ticker is not in the universe.
    """
    try:
        return _BY_TICKER[ticker.upper()]
    except KeyError as exc:
        raise KeyError(f"Ticker {ticker!r} is not in the universe") from exc


def by_bucket(bucket: Bucket) -> tuple[Instrument, ...]:
    """Return all instruments in a given bucket, in canonical order."""
    return tuple(inst for inst in UNIVERSE if inst.bucket == bucket)


def by_subgroup(subgroup: str) -> tuple[Instrument, ...]:
    """Return all instruments with a matching subgroup (case-sensitive)."""
    return tuple(inst for inst in UNIVERSE if inst.subgroup == subgroup)


def metadata_frame() -> pd.DataFrame:
    """Return the universe as a DataFrame indexed by ticker.

    Columns: ``name``, ``bucket``, ``subgroup``, ``inception``, ``notes``.
    Convenient for the coverage table and any downstream tabular reporting.
    """
    return pd.DataFrame(
        [
            {
                "ticker": inst.ticker,
                "name": inst.name,
                "bucket": inst.bucket,
                "subgroup": inst.subgroup,
                "inception": inst.inception,
                "notes": inst.notes,
            }
            for inst in UNIVERSE
        ]
    ).set_index("ticker")
