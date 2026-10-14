# Data Acquisition & Cleaning — Design Document

**Project:** portfolio-risk-covariance
**Status:** v1.1
**Author:** Marco Serra
**Last updated:** 2026-10-06

## 1. Purpose and scope

This document specifies the data-acquisition layer for the covariance-benchmark study. It fixes the instrument universe, source, date range, calendar conventions, cleaning rules, and cache layout. Code in `src/portfolio_risk/` treats this specification as authoritative; downstream deviations must be captured as edits to this file before the code is written. The evaluation layer (backtest calendar, estimation windows, criteria) is specified separately in `02-evaluation-design.md`.

## 2. Universe

**Total instruments: 75**, split across five buckets. All series are log-returns of daily adjusted close for tradable instruments (unit-homogeneous).

The equities bucket is fixed as of the Jan 2010 S&P 500 constituent list, filtered for continuity through 2025 (no major spinoffs, mergers, or corporate restructurings judged materially disruptive to the adjusted price series). This introduces a soft survivorship bias, addressed in §8.

### 2.1 US large-cap equities (38)

Balanced across five sectors.

**Technology (8)**

| Ticker | Name | Notes |
|--------|------|-------|
| AAPL | Apple | |
| MSFT | Microsoft | |
| ORCL | Oracle | |
| INTC | Intel | |
| CSCO | Cisco | |
| IBM | IBM | Kyndryl spinoff Nov 2021 — adjusted close handles |
| ADBE | Adobe | |
| QCOM | Qualcomm | |

**Financials (8)**

| Ticker | Name | Notes |
|--------|------|-------|
| JPM | JPMorgan Chase | |
| BAC | Bank of America | |
| WFC | Wells Fargo | |
| C | Citigroup | |
| GS | Goldman Sachs | |
| MS | Morgan Stanley | |
| AXP | American Express | |
| USB | US Bancorp | |

**Healthcare (8)**

| Ticker | Name | Notes |
|--------|------|-------|
| JNJ | Johnson & Johnson | Kenvue spinoff Aug 2023 — adjusted close handles |
| PFE | Pfizer | Viatris spinoff Nov 2020 — adjusted close handles |
| MRK | Merck | Organon spinoff Jun 2021 — adjusted close handles |
| LLY | Eli Lilly | |
| BMY | Bristol-Myers Squibb | |
| AMGN | Amgen | |
| GILD | Gilead | |
| MDT | Medtronic | Covidien inversion Jan 2015 — reorganized to Ireland; price series continuous |

**Industrials (7)**

| Ticker | Name | Notes |
|--------|------|-------|
| BA | Boeing | |
| CAT | Caterpillar | |
| MMM | 3M | Solventum spinoff Apr 2024 — adjusted close handles |
| HON | Honeywell | Garrett spinoff Oct 2018, Resideo spinoff Oct 2018 — adjusted close handles |
| UPS | United Parcel Service | |
| LMT | Lockheed Martin | |
| UNP | Union Pacific | |

**Consumer Staples (7)**

| Ticker | Name | Notes |
|--------|------|-------|
| PG | Procter & Gamble | |
| KO | Coca-Cola | |
| PEP | PepsiCo | |
| WMT | Walmart | |
| MO | Altria | |
| CL | Colgate-Palmolive | |
| KMB | Kimberly-Clark | Halyard Health spinoff Oct 2014 — minor magnitude |

**Deliberately excluded despite Jan 2010 S&P 500 membership** (spinoffs / mergers judged too disruptive for a comparative covariance study, even with adjusted close):

| Ticker | Reason |
|--------|--------|
| GE | 2024 three-way split (GE Aerospace / GE HealthCare / GE Vernova) |
| UTX → RTX | 2020 Raytheon merger + Otis/Carrier spinoffs |
| DD | 2019 tri-spin (Dow / DuPont / Corteva) |
| ABT | Jan 2013 AbbVie spinoff (large magnitude) |
| K → KLG | 2023 Kellanova split |
| BAX | 2015 Baxalta spinoff |

### 2.2 Sector / regional / factor ETFs (15)

**US sector — SPDR Select Sector (9):** XLK, XLF, XLV, XLI, XLP, XLY, XLE, XLB, XLU. All launched 1998. XLC (Communication Services, 2018) and XLRE (Real Estate, 2015) excluded to keep the sector set stable across the study window.

**Regional (3):** EFA (developed ex-US, 2001), EEM (emerging markets, 2003), VGK (Europe, 2005).

**Factor (3):** MTUM (US momentum, Apr 2013), IWD (Russell 1000 Value, 2000), IWF (Russell 1000 Growth, 2000).

MTUM's first observation in the fetched data is 2013-04-18, so it is absent from most of the burn-in period. Flagged in the coverage table; not dropped — handled at the estimation-window level per §5.3.

### 2.3 Rate ETFs (8)

| Ticker | Name | Bucket |
|--------|------|--------|
| SHY | iShares 1-3yr Treasury | Short duration |
| IEF | iShares 7-10yr Treasury | Intermediate duration |
| TLT | iShares 20+yr Treasury | Long duration |
| TIP | iShares TIPS | Inflation-linked |
| LQD | iShares IG Corporate Bond | IG credit |
| HYG | iShares HY Corporate Bond | HY credit |
| MBB | iShares MBS | Agency mortgage |
| EMB | iShares JPM USD EM Bond | EM sovereign USD |

### 2.4 Commodities and FX (8)

| Ticker | Name | Notes |
|--------|------|-------|
| GLD | SPDR Gold | |
| SLV | iShares Silver Trust | |
| USO | US Oil Fund | Roll methodology changed Apr 2020 following negative WTI event |
| DBC | Invesco DB Commodity Index | Broad commodity |
| DBA | Invesco DB Agriculture | |
| UUP | Invesco DB US Dollar Index Bullish | USD |
| FXE | Invesco CurrencyShares Euro | EUR |
| FXY | Invesco CurrencyShares Japanese Yen | JPY |

### 2.5 Vol / spread / alternative-credit (6)

The original spec called this bucket "vol / spread proxies as log-returns of tradable instruments" — a constraint that rules out ^VIX (level series, not tradable) and makes classical VIX-futures ETFs (VXX) problematic due to reverse-split discontinuities and a v1→v2 relaunch in Jan 2019. The composition below substitutes one clean vol-futures proxy plus five instruments that add non-redundant credit and low-vol equity exposures. The bucket label has been widened accordingly.

| Ticker | Name | Exposure | First observation |
|--------|------|----------|-------------------|
| VIXY | ProShares VIX Short-Term Futures | Vol futures | 2011-01-04 |
| SPLV | Invesco S&P 500 Low Volatility | Low-vol equity | 2011-05-05 |
| BKLN | Invesco Senior Loan | Floating-rate bank loans | 2011-03-03 |
| CWB | SPDR Bloomberg Convertible Securities | Convertibles | full window (launched Apr 2009) |
| PFF | iShares Preferred Stock | Subordinated credit / preferreds | full window (launched Mar 2007) |
| BIL | SPDR Bloomberg 1-3M T-Bill | Cash proxy | full window (launched May 2007) |

VIXY, SPLV, and BKLN launch in 2011 — treated like MTUM (missing in part of the burn-in) per §5.3. BIL is excluded from the optimised portfolios in the evaluation (see `02-evaluation-design.md`); it remains in the universe and in the risk-forecast tests.

## 3. Source

Primary: **yfinance** for all 75 instruments. Adjusted close is used throughout (splits, dividends, and spinoff consideration handled at source).

Fallback / cross-check: **FRED** for a handful of macro reference series used only in diagnostics (e.g. ^VIX level, TED spread), not in the estimation universe.

No proprietary vendor data. No manual price overrides.

## 4. Date range and calendar

- **Study window:** 2010-01-04 to 2025-12-31 (4,024 NYSE trading days).
- **Burn-in:** 2010–2014, providing history for the first estimation windows.
- **Backtest:** 131 non-overlapping 21-day holding periods from 2015-01-02 to 2025-12-09; the final 15 trading days of 2025 do not form a full period and are dropped. Rebalance dates, estimation windows and regimes are specified in `02-evaluation-design.md`.
- **Reference calendar:** NYSE trading days via `pandas_market_calendars`. Half-days included.
- **Non-NYSE instruments** (FX, some commodities): reindex to the NYSE calendar. Gaps handled per §5.1.

## 5. Data handling

### 5.1 Missing-observation policy

Evaluated per instrument, per date, against the NYSE calendar:

- **Isolated 1-day gap** (missing on day t, present on t-1 and t+1): forward-fill from t-1. Produces a zero log-return on t and the true return on t+1 computed against the ffill'd price — no artificial jump.
- **Multi-day gap** (≥2 consecutive missing days on valid trading days): flag date universe-wide, drop from the return panel. Logged to `data/processed/dropped_dates.csv`.
- **Pre-inception missing:** NaN in the panel.

**Result on the fetched data:** 0 dates dropped; the cleaned panel has all 4,024 trading days.

### 5.2 Return conventions

- **Log-returns:** `r_t = log(P_t / P_{t-1})` on adjusted close.
- **First observation per instrument dropped** (no lag-1 available).
- **Units:** daily. Weekly and monthly returns derived downstream by summing log-returns over the appropriate NYSE trading-day count.

### 5.3 Post-2010 inception instruments

MTUM, VIXY, SPLV, and BKLN start after 2010-01-04; their columns contain NaN before their first observation. (CWB, listed here in v1, has data for the full window and is not affected.)

Estimators require a complete panel, so an instrument enters an estimation window only if it has data on every day of that window. Within a given evaluation run, all estimators use the same asset set at each rebalance date. In practice: with 252-day windows all 75 instruments are eligible throughout the backtest; with 504-day windows MTUM enters at the rebalance of 2015-05-04. Details in `02-evaluation-design.md` §4. (This replaces the v1 working rule of a 60-observation minimum.)

### 5.4 Outlier flagging

- **Rule:** flag any observation with `|z_t| > 8`, where `z_t = (r_t − median) / (1.4826 × MAD)` is computed on a right-anchored rolling 63-day window (~3 months) that includes day t. The first 62 days of each series cannot be flagged.
- **Action:** flag only, do not remove. Logged to `data/processed/anomalies.csv` with ticker, date, return, rolling median, MAD and z-score.
- **Rationale:** the estimator comparison should see the same data any user would; robust estimators (nonlinear shrinkage) are expected to handle outliers better than naive sample covariance, and removing them would prejudge the comparison.

**Result on the fetched data (2010–2025): 1,118 flags.**

- **601 are artefacts of the rule on BIL.** BIL's price moves in one-cent steps (≈0.02%) and is unchanged on most days, particularly in 2010–2015, so in many 63-day windows more than half of its returns are exactly zero. The MAD is then zero and any one-cent move gets an infinite z-score. The underlying prices are correct; the rule is undefined for such windows. The flags are not used by any downstream computation, so the rule is left unchanged; reports of the anomaly count exclude rows with MAD = 0 and state why.
- **517 are genuine**, about 7 per instrument over 16 years, consistent with fat-tailed daily returns. They coincide with known events: the May 2010 flash crash, the August 2011 US downgrade, August 2015, December 2018, March 2020, August 2024, April 2025.
- **Distribution adjustments checked on BKLN and PFF** (the monthly-paying funds most exposed to adjustment errors): their flags cluster in crisis periods and are spread evenly across days of the month, with no concentration around ex-distribution dates.

## 6. Cache layout

```
data/
├── raw/
│   └── prices.parquet         ← 75-column panel, DatetimeIndex, adjusted close as fetched
├── processed/
│   ├── prices.parquet         ← cleaned panel aligned to the NYSE calendar
│   ├── returns.parquet        ← 75-column panel of log-returns
│   ├── anomalies.csv          ← outlier flag log
│   └── dropped_dates.csv      ← universe-wide dropped dates
└── manifest.yaml              ← universe version, date range, SHA-256 hashes (tracked)
```

`data/raw/` and `data/processed/` are git-ignored; `data/manifest.yaml` is tracked and serves as the reproducibility manifest. Instrument metadata (name, bucket, subgroup, inception) lives in `src/portfolio_risk/data/universe.py`, not in a separate metadata file as planned in v1.

**Refresh:** every run re-fetches the full window rather than appending a delta (deviation from v1). yfinance retroactively re-adjusts historical prices when new dividends or splits occur, so appending can produce inconsistent series; a full fetch takes about 30 seconds. The manifest's hashes record which version of the data a given set of results used.

## 7. Initial diagnostics 

Produced by `notebooks/01_data_diagnostics.ipynb` and saved to `memo/figures/` and `memo/tables/`:

1. **Coverage table** (`coverage.csv`): per-instrument first observation, last observation, total N obs, % missing on NYSE trading days within the instrument's live period, inception date, bucket.
2. **Rolling coverage plot** (`coverage_rolling.png`): count of live instruments per date.
3. **Return distribution stats** (`return_stats.csv`): per instrument — mean, std, skew, excess kurtosis, min, max, ADF test statistic and p-value.
4. **Full-sample correlation heatmap** (`correlation_full.png`): 75×75 Pearson correlation of log-returns, ordered by bucket, with bucket labels on the diagonal blocks.
5. **Rolling median pairwise correlation** (`correlation_rolling.png`): median off-diagonal pairwise correlation on a 63-day rolling window.




## 8. Known limitations

- **Survivorship bias:** the Jan-2010-membership-plus-survivorship filter excludes companies that went bankrupt, were acquired, or restructured out of existence during the window. Biases the equities panel toward higher-return, lower-tail-risk names than the "true" S&P 500 experience. Acceptable for a *comparative* estimator study (all estimators see the same biased panel); material for absolute performance claims.
- **US-centric universe:** regional and FX ETFs partially offset, but the study is not globally diversified in the sense a real institutional risk system would require.
- **Daily frequency only:** intraday effects (overnight vs. intraday returns, autocorrelation in HY credit) are pooled into a single daily observation.
- **yfinance data quality:** free retail data source; occasional bad ticks and split-adjustment quirks known. Outlier flagging in §5.4 partially mitigates; the checks on the 2010–2025 fetch found no data errors.
- **Early stress episodes outside the backtest:** the 2011 and 2013 stress episodes fall in the burn-in, before the backtest starts in 2015.