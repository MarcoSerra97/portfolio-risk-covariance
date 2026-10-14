"""Decompose DCC's equal-weight variance forecast at one rebalance.

For the forecast w' Sigma_H w (w = 1/p, Sigma_H = average of the H daily forecast
covariances), report:

1. the fitted parameters and the looking-back / looking-ahead factors;
2. the weight and contribution of the last m days in the starting state
   (next-day variances and correlation state);
3. the diagonal and off-diagonal parts of the forecast, next-day and H-day, and how
   much of each is due to the last m days, via a counterfactual in which those days
   were ordinary (squared returns and co-movements equal to their long-run values).

The script reproduces DCC's internal state from the fitted parameters and checks it
against the estimator's own one-step and H-day outputs before using it.

Usage:  python scripts/dcc_decomposition.py [rebalance date] [m]
        python scripts/dcc_decomposition.py 2020-03-05 9
"""

import sys

import numpy as np
import numpy.typing as npt
import pandas as pd

from portfolio_risk.estimators.dcc_garch import DCCGarch

END = sys.argv[1] if len(sys.argv) > 1 else "2020-03-05"
M = int(sys.argv[2]) if len(sys.argv) > 2 else 9
H = 21
PATH = sys.argv[3] if len(sys.argv) > 3 else "data/processed/returns.parquet"

window = pd.read_parquet(PATH).loc["2013-04-18":END].dropna()
est = DCCGarch(horizon=H).fit(window)
scale = est.scale
y = window.to_numpy(float) * scale
n, p = y.shape
w = np.full(p, 1 / p)
g = est.garch_params_
omega = g["omega"].to_numpy() * scale**2
alpha, beta, fb = g["alpha"].to_numpy(), g["beta"].to_numpy(), g["fallback"].to_numpy()
phi = alpha + beta
a, b = est.dcc_a_, est.dcc_b_

# ── reproduce the GARCH variance paths, starting from arch's exponentially weighted backcast
sig2 = np.empty((n + 1, p))
bw = 0.94 ** np.arange(min(75, n))
backcast = (bw[:, None] * y[: len(bw)] ** 2).sum(axis=0) / bw.sum()
sig2[0] = omega + (alpha + beta) * backcast
for t in range(n):
    sig2[t + 1] = omega + alpha * y[t] ** 2 + beta * sig2[t]
const = (y**2).mean(axis=0)
sig2[:, fb] = const[fb]  # fallback assets: constant variance
next_var = sig2[n]
long_run = np.where(fb, next_var, omega / np.where(fb, 1.0, 1.0 - phi))

# ── reproduce the cDCC state
z = y / np.sqrt(sig2[:n])
qd = np.empty((n, p))
qd[0] = 1.0
for t in range(1, n):
    qd[t] = (1 - a - b) + (a * z[t - 1] ** 2 + b) * qd[t - 1]
zs = z * np.sqrt(qd)
target = zs.T @ zs / n
dt = np.sqrt(np.diag(target))
target = target / np.outer(dt, dt)
q = target.copy()
q_hist = {}
for t in range(1, n + 1):  # q after step t is Q_t (state for day t)
    u = zs[t - 1] if t < n else z[n - 1] * np.sqrt(np.diag(q))
    q = (1 - a - b) * target + a * np.outer(u, u) + b * q
    q_hist[t] = q
q_next = q_hist[n]


def forecast(var1: np.ndarray, q1: np.ndarray, horizon: int) -> np.ndarray:
    """Average forecast covariance over `horizon` days, exactly as DCCGarch does."""
    acc = np.zeros((p, p))
    for h in range(1, horizon + 1):
        var_h = long_run + phi ** (h - 1) * (var1 - long_run)
        q_h = target + (a + b) ** (h - 1) * (q1 - target)
        d = np.sqrt(np.diag(q_h))
        sd = np.sqrt(var_h)
        acc += q_h / np.outer(d, d) * np.outer(sd, sd)
    return acc / horizon / scale**2


# ── validate the reproduction against the estimator
err_1 = np.abs(forecast(next_var, q_next, 1) - est.one_step_covariance_.to_numpy()).max()
err_h = np.abs(forecast(next_var, q_next, H) - est.covariance_.to_numpy()).max()
ref = np.abs(est.covariance_.to_numpy()).max()
print(
    f"=== rebalance {END}: window {window.index[0].date()} -> {window.index[-1].date()}, "
    f"{n} days, last {M} days = {M / n:.2%} of the estimation sample ==="
)
print(
    f"reproduction check: max relative error next-day {err_1 / ref:.1e}, {H}-day {err_h / ref:.1e}"
)

# ── 1. parameters, looking back and ahead
ok = ~fb


def psi(f: np.ndarray | float) -> np.ndarray | float:
    """Fraction of the excess kept by the H-day average forecast."""
    return (1 - f**H) / (H * (1 - f))


def half(f: np.ndarray | float) -> np.ndarray | float:
    """Half-life in days of a shock with persistence f."""
    return np.log(0.5) / np.log(f)


def qs(x: npt.ArrayLike) -> str:
    """Median and quartiles across assets."""
    lo, mid, hi = np.quantile(np.asarray(x, dtype=float), [0.25, 0.5, 0.75])
    return f"median {mid:.3f} (quartiles {lo:.3f}-{hi:.3f})"


print("\n1. PARAMETERS")
print(f"  GARCH ({ok.sum()} fits, fallbacks: {est.fallbacks_})")
print(f"    alpha {qs(alpha[ok])}\n    beta  {qs(beta[ok])}\n    alpha+beta {qs(phi[ok])}")
print(f"    look back: memory 1/(1-beta) {qs(1 / (1 - beta[ok]))} days")
print(f"    look ahead: half-life {qs(half(phi[ok]))} days; psi_{H} {qs(psi(phi[ok]))}")
print(f"  cDCC: a = {a:.4f}, b = {b:.4f}, a+b = {a + b:.4f}")
print(
    f"    look back: memory 1/(1-b) = {1 / (1 - b):.1f} days; "
    f"anchor share {(1 - a - b) / (1 - b):.3f}, dynamic share {a / (1 - b):.3f}"
)
print(f"    look ahead: half-life {half(a + b):.1f} days; psi_{H} = {psi(a + b):.3f}")

# ── 2. weight and contribution of the last M days in the starting state
lr_excess = next_var - long_run
cf_var = long_run + beta**M * (sig2[n - M] - long_run)  # last M days ordinary
cf_var[fb] = next_var[fb]
contrib_var = np.where(ok, (next_var - cf_var) / next_var, 0.0)
q_cf = target + b**M * (q_hist[n - M] - target)
print(f"\n2. LAST {M} DAYS IN THE STARTING STATE (day after the rebalance)")
print(f"  variances: weight share 1-beta^{M} {qs((1 - beta**M)[ok])}")
anchor_v = (1 - phi[ok]) / (1 - beta[ok])
print(f"             anchor share of the state (1-phi)/(1-beta) {qs(anchor_v)}")
for mm in sorted({9, 30, M}):
    share_v = alpha[ok] * (1 - beta[ok] ** mm) / (1 - beta[ok])
    print(
        f"             last {mm} days, share of the state alpha(1-beta^{mm})/(1-beta) {qs(share_v)}"
    )
print(f"             share of next-day variance due to them {qs(contrib_var[ok])}")
print(
    f"  correlation state: weight share of last {M} days a(1-b^{M})/(1-b) = "
    f"{a * (1 - b**M) / (1 - b):.3f} (within dynamic part {1 - b**M:.3f})"
)
for mm in sorted({9, 30} - {M}):
    print(
        f"  correlation state: last {mm} days a(1-b^{mm})/(1-b) = {a * (1 - b**mm) / (1 - b):.3f}"
    )
off = ~np.eye(p, dtype=bool)
r_act = q_next / np.outer(np.sqrt(np.diag(q_next)), np.sqrt(np.diag(q_next)))
r_cf = q_cf / np.outer(np.sqrt(np.diag(q_cf)), np.sqrt(np.diag(q_cf)))
print(
    f"  average pairwise correlation, day after rebalance: actual {r_act[off].mean():.3f}, "
    f"if last {M} days ordinary {r_cf[off].mean():.3f}, long-run target {target[off].mean():.3f}"
)


# ── 3. diagonal / off-diagonal parts, actual vs counterfactual
def parts(cov: np.ndarray) -> tuple[float, float, float]:
    total = float(w @ cov @ w)
    diag = float(np.sum(w**2 * np.diag(cov)))
    return total, diag, total - diag


scen = {
    "actual": (next_var, q_next),
    f"last {M} days ordinary (both layers)": (cf_var, q_cf),
    "... only variances ordinary": (cf_var, q_next),
    "... only correlations ordinary": (next_var, q_cf),
}
ann = 252
print(
    "\n3. EQUAL-WEIGHT FORECAST: DIAGONAL vs OFF-DIAGONAL (annualised variance, in %^2; vol in %)"
)
for horizon, label in ((1, "next day"), (H, f"{H}-day average")):
    print(f"  --- {label} ---")
    res = {k: parts(forecast(v, qq, horizon)) for k, (v, qq) in scen.items()}
    for k, (tot, dg, od) in res.items():
        print(
            f"  {k:38s} vol {np.sqrt(ann * tot):6.2%} "
            f"| diag {ann * dg * 1e4:7.2f} ({dg / tot:5.1%}) "
            f"| off-diag {ann * od * 1e4:7.2f} ({od / tot:5.1%})"
        )
    tot, dg, od = res["actual"]
    tot0, dg0, od0 = res[f"last {M} days ordinary (both layers)"]
    print(
        f"  due to the last {M} days: total {(tot - tot0) / tot:5.1%} of the forecast; "
        f"diag {(dg - dg0) / dg:5.1%} of the diagonal part; "
        f"off-diag {(od - od0) / od:5.1%} of the off-diagonal part"
    )

one, avg = parts(forecast(next_var, q_next, 1))[0], parts(forecast(next_var, q_next, H))[0]
print(f"\n  {H}-day average / next-day variance (overall damping): {avg / one:.3f}")
