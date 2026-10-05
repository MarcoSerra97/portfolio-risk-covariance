"""cDCC-GARCH: time-varying volatilities and correlations.

Two-stage dynamic conditional correlation model (Engle, 2002) in the corrected
form of Aielli (2013), with the correlation dynamics estimated by composite
likelihood (Engle, Shephard & Sheppard; Pakel et al., 2021).

Stage 1 — one GARCH(1,1) per asset (zero mean, Gaussian quasi-likelihood,
fitted with :mod:`arch` on returns scaled to percent):

.. math::

    \\sigma_{i,t}^2 = \\omega_i + \\alpha_i r_{i,t-1}^2 + \\beta_i \\sigma_{i,t-1}^2,
    \\qquad z_{i,t} = r_{i,t} / \\sigma_{i,t}.

Stage 2 — scalar cDCC recursion on the standardised residuals, with
correlation targeting:

.. math::

    Q_t = (1-a-b)\\,\\bar Q + a\\, z^*_{t-1} z^{*\\top}_{t-1} + b\\, Q_{t-1},
    \\qquad z^*_t = \\operatorname{diag}(Q_t)^{1/2} z_t,
    \\qquad R_t = \\operatorname{diag}(Q_t)^{-1/2} Q_t \\operatorname{diag}(Q_t)^{-1/2}.

The diagonal of ``Q_t`` follows its own univariate recursion (target 1), so
``z*`` and the target ``Q̄ = corr(z*)`` can be computed before the
off-diagonal recursion; this rescaling is what makes the correlation-targeting
estimator consistent (the "c" in cDCC). ``(a, b)`` maximise the sum of
bivariate Gaussian log-likelihoods over pairs of assets, which avoids the
attenuation of ``a`` that the full likelihood suffers in large dimensions.

Forecasts mean-revert: variances geometrically toward ``omega / (1 - alpha -
beta)`` at rate ``alpha + beta``. For correlations, cDCC makes ``Q_t`` the
conditional covariance of ``z*``, so its forecast is exact,
``E_t Q_{t+h} = Q̄ + (a + b)^{h-1} (Q_{t+1} - Q̄)``, and is normalised to a
correlation matrix. (Applying the recursion to ``R`` instead, the Engle-Sheppard
approximation designed for the original DCC, is markedly less accurate for
cDCC after extreme joint moves; checked by Monte Carlo.) ``covariance_`` is the
**average** forecast covariance over the next ``horizon`` days, so
``horizon * w @ covariance_ @ w`` is the forecast variance of the
``horizon``-day portfolio return, exactly as for the static estimators.

Assets whose GARCH fit fails or is degenerate (non-convergence, persistence at
or above ``max_persistence``, non-finite parameters) fall back to a constant
variance equal to their mean squared return; they are listed in
``fallbacks_``. Parameter estimation on an expanding window is the caller's
choice: the estimator fits whatever history it is given.
"""

from __future__ import annotations

from typing import Literal, Self

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.optimize import minimize

from portfolio_risk.estimators.base import validate_returns, wrap_covariance

_RHO_CLIP = 1.0 - 1e-10


def _pairs(p: int, kind: Literal["contiguous", "all"]) -> tuple[np.ndarray, np.ndarray]:
    """Index arrays (i, j) of the asset pairs used in the composite likelihood."""
    if kind == "contiguous":
        i = np.arange(p - 1)
        return i, i + 1
    iu, ju = np.triu_indices(p, k=1)
    return iu, ju


def _cdcc_diagonal(z: np.ndarray, a: float, b: float) -> np.ndarray:
    """Diagonal of Q_t: q_t = (1 - a - b) + (a z_{t-1}^2 + b) q_{t-1}, q_0 = 1."""
    n, p = z.shape
    q = np.empty((n, p))
    q[0] = 1.0
    c = 1.0 - a - b
    for t in range(1, n):
        q[t] = c + (a * z[t - 1] ** 2 + b) * q[t - 1]
    return q


def _offdiag_recursion(u: np.ndarray, target: np.ndarray, a: float, b: float) -> np.ndarray:
    """q_t = (1 - a - b) target + a u_{t-1} + b q_{t-1}, q_0 = target (vectorised over pairs)."""
    n = u.shape[0]
    q = np.empty_like(u)
    q[0] = target
    c = (1.0 - a - b) * target
    for t in range(1, n):
        q[t] = c + a * u[t - 1] + b * q[t - 1]
    return q


def _composite_negloglik(
    params: np.ndarray, z: np.ndarray, pi: np.ndarray, pj: np.ndarray
) -> float:
    """Negative composite (pairwise) log-likelihood of the cDCC correlation part."""
    s, frac = float(params[0]), float(params[1])
    a, b = s * frac, s * (1.0 - frac)
    qd = _cdcc_diagonal(z, a, b)
    zs = z * np.sqrt(qd)
    u = zs[:, pi] * zs[:, pj]
    s_ij = u.mean(axis=0) / np.sqrt((zs[:, pi] ** 2).mean(axis=0) * (zs[:, pj] ** 2).mean(axis=0))
    q_ij = _offdiag_recursion(u, s_ij, a, b)
    rho = np.clip(q_ij / np.sqrt(qd[:, pi] * qd[:, pj]), -_RHO_CLIP, _RHO_CLIP)
    zi, zj = z[:, pi], z[:, pj]
    one_m = 1.0 - rho**2
    ll = np.log(one_m) + (zi**2 - 2.0 * rho * zi * zj + zj**2) / one_m
    return float(0.5 * ll.sum())


class DCCGarch:
    """cDCC-GARCH(1,1) covariance estimator.

    Conforms to the :class:`~portfolio_risk.estimators.base.CovarianceEstimator`
    protocol; ``covariance_`` is the average forecast covariance over the next
    ``horizon`` days.

        >>> est = DCCGarch(horizon=21).fit(returns)  # doctest: +SKIP
        >>> est.dcc_a_, est.dcc_b_, est.fallbacks_    # doctest: +SKIP

    Attributes:
        horizon: Forecast horizon in days over which ``covariance_`` averages.
        pairs: Asset pairs in the composite likelihood: ``"contiguous"``
            (``p - 1`` neighbouring pairs, the standard choice) or ``"all"``.
        scale: Factor applied to returns before fitting (100 = percent), for
            the numerical conditioning of the GARCH optimiser. Results are
            reported in the original units.
        max_persistence: GARCH fits with ``alpha + beta`` at or above this
            value are treated as degenerate and replaced by a constant
            variance.
        covariance_: Average forecast covariance over ``horizon`` days.
        one_step_covariance_: One-day-ahead forecast covariance.
        garch_params_: Per-asset ``omega`` (original units), ``alpha``,
            ``beta``, ``persistence`` and ``fallback`` flag.
        fallbacks_: Assets whose GARCH fit was replaced by a constant variance.
        dcc_a_, dcc_b_: Fitted correlation reaction and persistence.
        dcc_converged_: Whether the composite-likelihood optimiser converged.
            If not, ``a = b = 0`` (constant correlation) is used.
        correlation_target_: The long-run correlation ``Q̄`` as a DataFrame.
        one_step_q_: ``Q_{T+1}``, the one-step-ahead cDCC state (diagnostic).
        correlation_path_: Forecast correlation matrices ``R_{T+h|T}`` for
            ``h = 1..horizon``, shape ``(horizon, p, p)``.
    """

    def __init__(
        self,
        horizon: int = 21,
        pairs: Literal["contiguous", "all"] = "contiguous",
        scale: float = 100.0,
        max_persistence: float = 0.999,
    ) -> None:
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if pairs not in ("contiguous", "all"):
            raise ValueError(f"pairs must be 'contiguous' or 'all', got {pairs!r}")
        self.horizon: int = horizon
        self.pairs: Literal["contiguous", "all"] = pairs
        self.scale: float = scale
        self.max_persistence: float = max_persistence

    # ── stage 1 ───────────────────────────────────────────────────────────────

    def _fit_garch(
        self, y: np.ndarray
    ) -> tuple[np.ndarray, float, tuple[float, float, float], bool]:
        """Fit one zero-mean GARCH(1,1); return (sigma2_t, sigma2_{T+1}, params, fallback)."""
        try:
            res = arch_model(
                y, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False
            ).fit(disp="off")
            omega = float(res.params["omega"])
            alpha = float(res.params["alpha[1]"])
            beta = float(res.params["beta[1]"])
            sigma2 = np.asarray(res.conditional_volatility, dtype=float) ** 2
            ok = (
                int(res.convergence_flag) == 0
                and np.isfinite([omega, alpha, beta]).all()
                and omega > 0
                and alpha >= 0
                and beta >= 0
                and alpha + beta < self.max_persistence
                and bool(np.all(sigma2 > 0))
            )
        except (ValueError, np.linalg.LinAlgError, RuntimeError):
            ok = False
        if ok:
            next_var = omega + alpha * y[-1] ** 2 + beta * sigma2[-1]
            return sigma2, float(next_var), (omega, alpha, beta), False
        const = float(np.mean(y**2))
        return np.full(y.shape[0], const), const, (const, 0.0, 0.0), True

    # ── stage 2 ───────────────────────────────────────────────────────────────

    def _fit_dcc(self, z: np.ndarray) -> tuple[float, float, bool]:
        pi, pj = _pairs(z.shape[1], self.pairs)
        res = minimize(
            _composite_negloglik,
            x0=np.array([0.95, 0.03]),
            args=(z, pi, pj),
            method="L-BFGS-B",
            bounds=[(1e-4, 0.999), (1e-4, 0.9999)],
        )
        if not res.success or not np.isfinite(res.fun):
            return 0.0, 0.0, False
        s, frac = float(res.x[0]), float(res.x[1])
        return s * frac, s * (1.0 - frac), True

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, returns: pd.DataFrame) -> Self:
        """Fit both stages on ``returns`` and form the forecasts.

        Args:
            returns: Clean return panel, observations in rows and instruments
                in columns. See
                :func:`~portfolio_risk.estimators.base.validate_returns`.

        Returns:
            The estimator itself, with all fitted attributes populated.
        """
        validate_returns(returns)
        y = returns.to_numpy(dtype=float) * self.scale
        n, p = y.shape

        sigma2 = np.empty((n, p))
        next_var = np.empty(p)
        params = np.empty((p, 3))
        fallback = np.zeros(p, dtype=bool)
        for k in range(p):
            sigma2[:, k], next_var[k], prm, fallback[k] = self._fit_garch(y[:, k])
            params[k] = prm
        z = y / np.sqrt(sigma2)

        a, b, converged = self._fit_dcc(z)

        # Full-matrix cDCC filter at the fitted (a, b).
        qd = _cdcc_diagonal(z, a, b)
        zs = z * np.sqrt(qd)
        target = zs.T @ zs / n
        d = np.sqrt(np.diag(target))
        target = target / np.outer(d, d)
        q = target.copy()
        c = 1.0 - a - b
        for t in range(1, n):
            q = c * target + a * np.outer(zs[t - 1], zs[t - 1]) + b * q
        zs_last = z[-1] * np.sqrt(np.diag(q))
        q_next = c * target + a * np.outer(zs_last, zs_last) + b * q
        dq = np.sqrt(np.diag(q_next))
        r_next = q_next / np.outer(dq, dq)

        # Multi-step forecasts with mean reversion.
        omega, alpha, beta = params[:, 0], params[:, 1], params[:, 2]
        persistence = alpha + beta
        long_run = np.where(fallback, next_var, omega / np.where(fallback, 1.0, 1.0 - persistence))
        cov_sum = np.zeros((p, p))
        corr_path = np.empty((self.horizon, p, p))
        for h in range(1, self.horizon + 1):
            var_h = long_run + persistence ** (h - 1) * (next_var - long_run)
            q_h = target + (a + b) ** (h - 1) * (q_next - target)  # exact for cDCC
            dh = np.sqrt(np.diag(q_h))
            r_h = q_h / np.outer(dh, dh)
            corr_path[h - 1] = r_h
            sd_h = np.sqrt(var_h)
            cov_sum += r_h * np.outer(sd_h, sd_h)
        scale2 = self.scale**2
        avg_cov = cov_sum / self.horizon / scale2
        sd1 = np.sqrt(next_var)
        one_step = r_next * np.outer(sd1, sd1) / scale2

        cols = returns.columns
        self.covariance_: pd.DataFrame = wrap_covariance((avg_cov + avg_cov.T) / 2, cols)
        self.one_step_covariance_: pd.DataFrame = wrap_covariance((one_step + one_step.T) / 2, cols)
        self.garch_params_: pd.DataFrame = pd.DataFrame(
            {
                "omega": omega / scale2,
                "alpha": alpha,
                "beta": beta,
                "persistence": persistence,
                "fallback": fallback,
            },
            index=cols,
        )
        self.fallbacks_: list[str] = [str(c_) for c_ in cols[fallback]]
        self.dcc_a_: float = a
        self.dcc_b_: float = b
        self.dcc_converged_: bool = converged
        self.correlation_target_: pd.DataFrame = wrap_covariance(target, cols)
        self.one_step_q_: np.ndarray = q_next
        self.correlation_path_: np.ndarray = corr_path
        return self


__all__ = ["DCCGarch"]
