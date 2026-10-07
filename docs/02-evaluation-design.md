# Evaluation — Design Document

**Project:** portfolio-risk-covariance
**Scope:** `src/portfolio_risk/evaluation/` — backtest engine and Criteria A, B, C
**Status:** Draft v1.5
**Author:** Marco Serra
**Last updated:** 2026-10-05

## 1. Purpose and scope

This document specifies how every (estimator, portfolio) pair is evaluated: the backtest calendar, the information set at each rebalance, the estimation windows, the forecast and realised quantities, the metrics for Criteria A, B and C, the bootstrap, and the engine's structure. As with `01-data-acquisition-design.md`, the code treats this document as authoritative; deviations are recorded here before they are implemented.

## 2. Decisions this document depends on

| Decision | Choice | Status |
|---|---|---|
| Cash instrument (BIL) | Excluded from the optimised portfolios (risk parity, min-var long-only, min-var long-short); kept in equal weight and in Criterion A | Decided |
| Criterion A loss | MAE and QLIKE, co-primary | Decided |
| Ledoit-Wolf | Two equal variants, both evaluated in full: LW on raw returns (scaled-identity target) and LW on standardised returns (correlations shrunk, variances kept); constant-correlation target not implemented | Decided |
| DCC variant | cDCC (Aielli 2013), composite likelihood, exact Q-level forecasts | Decided |
| Estimator-independent portfolios | Equal weight only (cap weight dropped) | Decided (scope cut) |
| Window sensitivity | 252 days (baseline) and 504 days | Decided (scope cut) |
| Failure-mode tests | Dimensionality stress only | Decided (scope cut) |
| Data window | Extended to 2025-12-31 | Decided; action in §13 |
| Regimes | Five regimes plus a 2025 tariff-shock sub-period; assignment by majority of days | Decided |
| QIS | Two equal variants, both evaluated in full: QIS on raw returns (as published) and QIS on standardised returns | Decided |
| Return conventions | Criterion A: log returns (consistent with the log-return covariance being forecast). Criterion B: simple returns, exact for constant weights, in excess of BIL | Decided |
| Textbook conventions | Weights constant within each 21-day period (Ledoit & Wolf, 2017); turnover from drifted weights (DeMiguel, Garlappi & Uppal, 2009) | Decided |

## 3. Calendar

- **Data window:** 2010-01-04 to 2025-12-31.
- **Burn-in:** 2010–2014. No evaluation; estimation windows fill.
- **Holding period:** H = 21 trading days.
- **Rebalance dates:** τ₀ = last NYSE trading day of 2014; τₖ = τ₀ + 21k trading days. Holding period k covers trading days τₖ + 1 … τₖ + 21. Holding periods do not overlap. A final partial period is dropped. This gives about 131 holding periods.
- **Why 21-day blocks rather than calendar months:** the forecast horizon then equals the holding period exactly, so the forecast H·w′Σ̂w and the realised variance (§6) always refer to the same number of days.

## 4. Information set and estimation windows

At rebalance date τ, an estimator sees returns up to and including τ, and nothing after. Weights are set at τ and held through the holding period.

| Estimators | Estimation sample at τ |
|---|---|
| Sample, LW raw, LW standardised, MP, QIS raw, QIS standardised | Rolling window of the last n days; n = 252 (baseline) and n = 504 (sensitivity) |
| cDCC-GARCH | Expanding window from the common start date 2013-04-18 (first date on which all 75 instruments trade) up to τ; parameters re-estimated at every rebalance |

The window-length sensitivity therefore applies to the static estimators. DCC's memory is governed by its fitted parameters, not by a window, and its expanding sample avoids the unstable persistence estimates a one-year window would produce. The memo states this explicitly, so readers do not read DCC as being evaluated on different data. The first DCC fits (early 2015) use about 1.7 years of data; this grows to about 12.7 years by the end of the sample.

**Asset set at τ.** Within a given run (fixed window length), every estimator at τ uses the same asset set: all instruments with complete data over that run's static estimation window. With n = 252, all 75 instruments are eligible throughout the backtest. With n = 504, MTUM (launched April 2013) enters at the first rebalance with 504 days of history, around April 2015. The asset set is recorded for every τ.

## 5. Estimators and portfolios

**Estimators (7):** sample covariance; Ledoit-Wolf on raw returns (scaled-identity target); Ledoit-Wolf on standardised returns; MP clipping; QIS raw; QIS standardised; cDCC-GARCH (horizon = 21). Ledoit-Wolf and QIS each have two equal variants, on raw and on standardised returns: neither is designated primary, and the comparison within each pair is reported as a finding (for Ledoit-Wolf the raw variant distorts low-volatility instruments strongly; for QIS only mildly, which locates the problem in the linear target rather than in shrinkage generally). MP and DCC work in correlation space by construction.

**Portfolios (4):**

| Portfolio | Universe | Uses the covariance |
|---|---|---|
| Equal weight | All eligible instruments, including BIL | No |
| Risk parity | Eligible instruments excluding BIL | Yes |
| Min-variance long-only | Eligible instruments excluding BIL | Yes |
| Min-variance long-short | Eligible instruments excluding BIL | Yes |

The BIL exclusion is applied through `portfolios.constructions.with_excluded`. Equal weight is identical across estimators by construction; it provides the clean forecast test for Criterion A.

## 6. Forecast and realised variance

For each estimator, portfolio and rebalance date:

- **Forecast:** v̂_τ = H · w_τ′ Σ̂_τ w_τ, where Σ̂_τ is the estimator's `covariance_`. For DCC this is the average forecast covariance over the holding period, so the same formula applies to every estimator.
- **Realised:** v_τ = Σ_{t=τ+1}^{τ+H} (w_τ′ r_t)², the sum of squared daily portfolio log-returns (zero-mean approximation). It is unbiased for the expected sum of daily variances up to a relative bias μ²/σ² ≈ 0.2%, and noisy: relative error √((κ−1)/H), 31% for Gaussian returns, about 40–60% for the study's portfolios. Ignoring the mean has lower mean squared error than demeaning within the period. Derivations in `docs/notes/estimators_note.tex` §4.
- **Weights within the holding period** are held constant at w_τ (daily rebalancing to target), consistent with the forecast. Buy-and-hold drift is used only for turnover (§8).

## 7. Criterion A — risk-measurement accuracy (all four portfolios)

Per (estimator, portfolio), over all holding periods:

| Metric | Definition | Role |
|---|---|---|
| MAE | mean of \|v̂_τ − v_τ\| | Co-primary; reported per portfolio only (variance units) |
| QLIKE | mean of v_τ/v̂_τ − log(v_τ/v̂_τ) − 1 | Co-primary |
| Calibration | Mincer–Zarnowitz regression v_τ = a + b·v̂_τ + ε; report b with Newey–West (HAC) standard errors and R² (b = 1 ideal); only for groups of at least 20 periods (not within the shock regimes) | Secondary |
| Bias ratio | mean of v_τ / v̂_τ (1 = unbiased; > 1 = forecasts too low) | Secondary |
| Error heteroskedasticity | errors against forecast level | Secondary (figure) |

**Interpretation.** Equal weight is a clean forecast test. Risk parity and minimum variance are consistency tests: their weights depend on Σ̂_τ, so forecasts are optimistic by construction.

**Why QLIKE as well as MAE.** The realised variance is a noisy proxy (about 31% relative error with 21 daily returns). QLIKE ranks forecasts consistently under a noisy proxy; MAE does not, because it targets the median of a right-skewed quantity and so can favour forecasts that are too low (Patton, 2011). QLIKE is also scale-free, so it can be compared across portfolios and across calm and crisis months. Where the two rankings disagree, QLIKE governs statements about forecasting skill, and the disagreement is reported.

**Pairwise comparisons.** For each estimator against the sample covariance, the mean QLIKE difference with a paired block-bootstrap confidence interval (§10). A Diebold–Mariano test (Newey–West variance, 3 lags) is reported alongside as a cross-check, on the full sample only: the shock regimes have too few periods for its asymptotics.

**Guards.** QLIKE requires v̂_τ > 0 and v_τ > 0; non-positive values raise an error rather than being silently dropped.

## 8. Criterion B — optimisation quality (risk parity, min-var long-only, min-var long-short)

Criterion B tables also report equal weight as a reference row; the comparison itself concerns the three optimised portfolios.

Out-of-sample daily portfolio returns are computed with simple returns, which are exact for constant weights: R_t^p = Σᵢ w_{i,τ} (exp(r_{i,t}) − 1), and concatenated across all holding periods. Weights are held constant within each period, following Ledoit & Wolf (2017).

| Metric | Definition | Role |
|---|---|---|
| Realised volatility | √252 × standard deviation of the daily out-of-sample returns in excess of BIL | Primary |
| Turnover | Σᵢ \|w_{i,τ} − w̃_{i,τ}\| per rebalance, where w̃ are the previous weights after drifting with prices over the holding period (DeMiguel, Garlappi & Uppal, 2009), computed from simple returns; averaged over rebalances, the first (set-up from cash) excluded | Secondary |
| Concentration | effective number of assets N_eff = 1 / Σᵢ w_i² and maximum absolute weight; for long-short, N_eff is indicative only | Secondary |
| Gross exposure | Σᵢ \|w_i\| (long-short only) | Secondary |
| Realised Sharpe ratio | mean excess return over BIL, annualised, divided by realised volatility | Secondary |

Transaction costs are not modelled; turnover is reported so a reader can apply any cost assumption. The tension between low volatility and investability (concentration, turnover, gross exposure) is reported, not hidden.

## 9. Criterion C — regime robustness

Each holding period is assigned to the regime that contains the majority of its 21 trading days (no ties are possible). Periods whose majority falls outside every regime are reported as unclassified: excluded from regime tables, included in full-sample results.

| Regime | Dates | Holding periods |
|---|---|---|
| Calm I | 2015-01 to 2019-12 | 0–59 |
| COVID shock | 2020-02 to 2020-05 | 61–64 |
| Post-COVID | 2020-06 to 2021-12 | |
| Rates shock | 2022-01 to 2023-06 | |
| Calm II | 2023-07 to 2025-02, and 2025-06 to 2025-12 | |
| Tariff shock | 2025-03 to 2025-05 | 122–124 |

Period 60 (2020-01-06 to 2020-02-04) is unclassified.

**Why majority-of-days, and why the tariff shock starts in March.** With 21-day blocks anchored at the end of 2014, the crash days of 2–8 April 2025 fall inside period 122 (2025-03-12 to 2025-04-09), most of whose days are in March. A sub-period starting on 1 April would leave the crash in Calm II under either assignment rule. The episode itself began earlier: equities peaked in mid-February 2025 and fell about 10% into mid-March on tariff fears, before the early-April crash and the recovery after the 9 April pause, with volatility normalising by late May. Defining the sub-period as March–May captures the slide, the crash and the recovery. For COVID, the majority rule places the February–March crash in periods 61–62.

- **Primary:** the A metrics (MAE, QLIKE) and the B metric (realised volatility, as a ratio to the sample covariance) recomputed regime by regime, with paired bootstrap intervals for the MAE and QLIKE differences and the volatility ratio against the sample covariance.
- **Resampling within a regime** uses only that regime's periods (A) or days (B). Calm II is split by the tariff shock, so its two parts are resampled as one sequence; a block may join the end of February 2025 to June 2025. The effect is minor and noted in the memo.
- **Secondary:** rank stability of the estimators across regimes (Kendall's W, plus a rank table).
- **Caveat:** the COVID regime contains 4 holding periods and the tariff shock 3. Their intervals will be wide, and conclusions from them are stated as indicative.

## 10. Bootstrap

- **Criterion A:** stationary block bootstrap (Politis–Romano) on the sequence of holding periods, mean block length 3 periods, 2,000 replications, percentile intervals.
- **Criterion B:** stationary block bootstrap on the daily out-of-sample excess returns, mean block length 21 days, 2,000 replications.
- **Block-length sensitivity:** full-sample intervals recomputed with longer mean blocks (A: 6 and 12 periods; B: 63 and 126 days); a conclusion is called robust only if it holds at every block length.
- **Paired resampling:** the same resampled indices are used for every estimator within a replication, so intervals for differences between estimators (for example, QLIKE of QIS minus QLIKE of the sample covariance) account for their correlation.
- **Seeds:** fixed and recorded, so every table is reproducible.

## 11. Rigour additions computed by the engine

- **Window sensitivity:** full A/B/C results for n = 252 and n = 504; findings labelled window-robust or window-sensitive.
- **Numerical diagnostics over time:** condition number of each Σ̂_τ; largest and smallest eigenvalues; LW shrinkage intensity; MP number of signal eigenvalues; QIS shrunk vs sample eigenvalues; DCC (a, b), GARCH persistence and fallback count.
- **DCC diagnostic panel:** distribution of (ω, α, β) across assets; parameter stability across refits; Ljung–Box tests on squared standardised residuals; coverage of 1-day predicted volatility.
- **Dimensionality stress test:** for N ∈ {30, 45, 60, 75}, 20 random asset subsets per N (stratified by bucket, fixed seeds), Criteria A and B recomputed; shows how each estimator degrades as p/n grows.

## 12. Engine structure

```
src/portfolio_risk/evaluation/
├── config.py       ← run configuration (window length, estimators, portfolios, exclusions, seeds)
├── schedule.py     ← rebalance dates, holding periods, regime labels
├── backtest.py     ← fit estimators at each τ, form weights, compute forecasts and realised quantities
├── metrics.py      ← Criterion A and B metrics
├── bootstrap.py    ← stationary block bootstrap (paired)
└── regimes.py      ← Criterion C tables and rank stability
```

- **Caching:** fitted covariance matrices are written to `data/cache/` (git-ignored), keyed by estimator, window length and τ, so metrics and figures can be recomputed without refitting.
- **Determinism:** no randomness in the backtest itself; seeds only in the bootstrap and the stress-test subsets.
- **Runtime (estimate):** static estimators take milliseconds per fit. DCC takes about 2 seconds per fit; it does not depend on the window length, so about 131 refits ≈ 5 minutes once. Portfolio construction is under 35 ms per call. A full run, both window lengths, is a few minutes plus the bootstrap.
- **Outputs:** tables to `memo/tables/`, figures to `memo/figures/`, as for the diagnostics.

## 13. Actions before implementation

1. **Extend the data to 2025-12-31.** Change `DEFAULT_END` in `src/portfolio_risk/data/acquisition.py` to `date(2025, 12, 31)`, re-run `python -m portfolio_risk.data.acquisition --verbose`, check the coverage table, and commit the updated `data/manifest.yaml`. Add a change-log entry to `01-data-acquisition-design.md` (§4 window, §10 change log).

## 14. Open questions

- **Stress-test stratification:** whether to keep bucket proportions fixed across N or sample uniformly.

## 15. Change log

- **v1 (2026-10-05):** initial draft.
- **v1.1 (2026-10-05):** Ledoit-Wolf on raw and on standardised returns set as equal variants.
- **v1.2 (2026-10-05):** tariff-shock sub-period (2025-03 to 2025-05) added; regime assignment by majority of days; holding-period numbers listed for the shock regimes.
- **v1.3 (2026-10-05):** QIS on raw and on standardised returns set as equal variants.
- **v1.4 (2026-10-07):** return conventions (log for A, simple excess over BIL for B); textbook conventions verified (Ledoit & Wolf 2017 constant weights; DeMiguel et al. 2009 turnover, first rebalance excluded); MAE per portfolio; HAC standard errors for calibration; Diebold–Mariano on the full sample only; QLIKE guards; realised-variance properties summarised.
- **v1.5 (2026-10-07):** `regimes.py` (Criterion C): MAE and QLIKE differences bootstrapped; calibration regression only for groups of at least 20 periods; block-length sensitivity specified; resampling within split regimes noted; equal weight shown as a reference row in Criterion B tables.
