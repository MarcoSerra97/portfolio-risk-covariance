"""DCC at the two COVID rebalances: persistence, damping, and the weight of recent days."""

import numpy as np
import pandas as pd

from portfolio_risk.estimators.dcc_garch import DCCGarch

returns = pd.read_parquet("data/processed/returns.parquet")

for end in ("2020-03-05", "2020-04-03"):
    window = returns.loc["2013-04-18":end].dropna()
    est = DCCGarch(horizon=21).fit(window)
    g = est.garch_params_
    ok = g[~g["fallback"]]
    a, b = est.dcc_a_, est.dcc_b_

    # Equal weight: next-day forecast vs the 21-day average actually used
    w = np.full(window.shape[1], 1 / window.shape[1])
    one_day = w @ est.one_step_covariance_.to_numpy() @ w
    average = w @ est.covariance_.to_numpy() @ w

    # Weight share and actual contribution of the last 9 days to next-day variance
    weight_share = 1 - ok["beta"] ** 9
    last9 = window[ok.index].iloc[-9:][::-1]  # most recent first
    decay = ok["beta"].to_numpy() ** np.arange(9)[:, None]
    next_var = np.diag(est.one_step_covariance_.loc[ok.index, ok.index].to_numpy())
    contrib = ok["alpha"].to_numpy() * (decay * last9.to_numpy() ** 2).sum(axis=0) / next_var

    print(f"\n=== refit on {end} ({len(window)} days) ===")
    print(f"GARCH fits: {len(ok)} of {len(g)}; fallbacks: {est.fallbacks_}")
    print(f"median GARCH persistence: {ok['persistence'].median():.3f}")
    print(
        f"last 9 days, weight share 1 - beta^9: median {weight_share.median():.2f} "
        f"(quartiles {weight_share.quantile(.25):.2f}-{weight_share.quantile(.75):.2f})"
    )
    print(
        f"last 9 days, contribution to next-day variance: median {np.median(contrib):.2f} "
        f"(quartiles {np.quantile(contrib, .25):.2f}-{np.quantile(contrib, .75):.2f})"
    )
    print(
        f"DCC a = {a:.4f}, b = {b:.4f}, a + b = {a + b:.3f}; "
        f"last 9 days in correlation state: {a * (1 - b ** 9) / (1 - b):.3f}"
    )
    print(
        f"equal weight, annualised: next-day {np.sqrt(252 * one_day):.1%}, "
        f"21-day average {np.sqrt(252 * average):.1%}"
    )
