"""Quadratic-Inverse Shrinkage (QIS): nonlinear shrinkage of the eigenvalues.

Implements the estimator of Ledoit & Wolf (2022), "Quadratic shrinkage for
large covariance matrices", *Bernoulli* 28(3). Like linear shrinkage it keeps
the sample eigenvectors and replaces the eigenvalues, but instead of moving
every eigenvalue toward the mean by the same proportion, it applies a smooth,
data-driven nonlinear map derived from random-matrix theory and optimal under
the Frobenius loss (and, up to scale, under the inverse-Stein and
minimum-variance losses). Small eigenvalues can be raised substantially while
large, well-estimated ones are left nearly untouched, which is exactly what a
single linear intensity cannot do.

The shrinkage formula follows the authors' reference implementation
(github.com/pald22/covShrinkage), rewritten in vectorised NumPy:

* effective sample size ``n_eff = n - 1`` after de-meaning, ``n`` otherwise;
  concentration ``c = p / n_eff``;
* bandwidth ``h = min(c^2, 1/c^2)^0.35 / p^0.35``;
* with ``t_i = 1 / lambda_i`` over the non-null sample eigenvalues,
  ``theta_j = mean_i t_i (t_i - t_j) / ((t_i - t_j)^2 + h^2 t_i^2)`` and
  ``H_j = mean_i h t_i^2 / ((t_i - t_j)^2 + h^2 t_i^2)``;
* shrunk eigenvalues
  ``d_j = 1 / (t_j [(1 - c)^2 + 2c(1 - c) theta_j + c^2 (theta_j^2 + H_j^2)])``
  when ``p <= n_eff``; when ``p > n_eff`` the non-null ones are
  ``1 / (t_j (theta_j^2 + H_j^2))`` and the ``p - n_eff`` null ones are set to
  ``1 / ((c - 1) mean(t))``;
* finally all shrunk eigenvalues are rescaled to preserve the trace.

The map is homogeneous of degree one in the eigenvalues, so applying it to the
divisor-``n`` sample covariance (this package's convention) gives exactly the
authors' estimate rescaled by ``n_eff / n``.

Two variants are available through ``standardize``, as for
:class:`~portfolio_risk.estimators.ledoit_wolf.LedoitWolfShrinkage`:

``standardize=False`` (default) — the paper's estimator on the raw covariance.
    Rotation-equivariant; trace preserved. Like identity-target linear
    shrinkage, it pulls the spectrum of the raw covariance together, which on
    a panel with very heterogeneous volatilities mainly distorts the
    low-volatility instruments.

``standardize=True`` — QIS on the correlation matrix.
    Columns are scaled to unit variance, QIS cleans the correlation spectrum,
    the result is renormalised to a unit diagonal and rescaled by the sample
    volatilities. Variances are preserved exactly; only correlations are
    shrunk. This is the construction used inside DCC-NL (Engle, Ledoit & Wolf,
    2019) and the recommended variant for this project's panel.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd

from portfolio_risk.estimators._spectral import rebuild_correlation, sorted_eigh, standardize
from portfolio_risk.estimators.base import validate_returns, wrap_covariance


def qis_shrink_eigenvalues(eigenvalues: np.ndarray, n_eff: int) -> np.ndarray:
    """Apply the QIS nonlinear map to a set of sample eigenvalues.

    Args:
        eigenvalues: Sample eigenvalues in descending order (length ``p``).
            Values below zero (rounding noise) are treated as zero.
        n_eff: Effective sample size used to compute the concentration ratio.

    Returns:
        Shrunk eigenvalues, in the same (descending) order as the input, with
        the same sum.

    Raises:
        ValueError: If a non-null eigenvalue is not strictly positive, i.e. the
            sample matrix is singular beyond what ``p > n_eff`` explains.
    """
    p = eigenvalues.shape[0]
    lam = np.clip(eigenvalues[::-1], 0.0, None)  # ascending, as in the reference
    c = p / n_eff
    h = min(c**2, 1.0 / c**2) ** 0.35 / p**0.35

    m = min(p, n_eff)  # number of non-null eigenvalues
    nonnull = lam[p - m :]
    # Same tolerance numpy.linalg.matrix_rank uses to call an eigenvalue zero.
    tol = lam.max() * p * np.finfo(float).eps
    if bool((nonnull <= tol).any()):
        raise ValueError(
            "sample matrix has zero eigenvalues beyond the p > n_eff null space; "
            "check for duplicated or perfectly collinear columns"
        )
    t = 1.0 / nonnull

    diff = t[:, None] - t[None, :]  # [i, j] = t_i - t_j
    denom = diff**2 + h**2 * t[:, None] ** 2
    theta = (t[:, None] * diff / denom).mean(axis=0)
    h_theta = (h * t[:, None] ** 2 / denom).mean(axis=0)
    a_theta2 = theta**2 + h_theta**2

    if p <= n_eff:
        delta = 1.0 / (t * ((1 - c) ** 2 + 2 * c * (1 - c) * theta + c**2 * a_theta2))
    else:
        delta_null = 1.0 / ((c - 1) * t.mean())
        delta = np.concatenate([np.full(p - m, delta_null), 1.0 / (t * a_theta2)])

    delta = delta * (lam.sum() / delta.sum())  # preserve the trace
    shrunk: np.ndarray = delta[::-1]  # back to descending order
    return shrunk


class QuadraticInverseShrinkage:
    """Ledoit-Wolf (2022) quadratic-inverse nonlinear shrinkage estimator.

    Conforms to the :class:`~portfolio_risk.estimators.base.CovarianceEstimator`
    protocol.

        >>> est = QuadraticInverseShrinkage(standardize=True).fit(returns)  # doctest: +SKIP

    Attributes:
        standardize: If ``True``, shrink the correlation spectrum and keep the
            sample variances. If ``False`` (default), shrink the spectrum of
            the raw covariance, as in the paper.
        assume_centered: If ``True``, the columns are assumed to already have
            zero mean; no de-meaning, and ``n_eff = n``. Default ``False``.
        covariance_: Fitted ``(p, p)`` covariance matrix as a labelled
            DataFrame. Populated by :meth:`fit`.
        eigenvalues_: Sample eigenvalues (of the covariance, or of the
            correlation matrix when ``standardize=True``), descending.
        shrunk_eigenvalues_: The corresponding QIS eigenvalues, descending.
            With ``standardize=True`` these are before the final
            renormalisation to a unit diagonal.
    """

    def __init__(self, standardize: bool = False, assume_centered: bool = False) -> None:
        self.standardize: bool = standardize
        self.assume_centered: bool = assume_centered

    def fit(self, returns: pd.DataFrame) -> Self:
        """Fit the estimator to ``returns`` and return ``self``.

        Args:
            returns: Clean return panel, observations in rows and instruments
                in columns. See
                :func:`~portfolio_risk.estimators.base.validate_returns`.

        Returns:
            The estimator itself, with all fitted attributes populated.

        Raises:
            ValueError: If some column has zero variance, or if the sample
                matrix is singular beyond what ``p > n_eff`` explains.
        """
        validate_returns(returns)
        z, scale, n_eff = standardize(returns, self.assume_centered)
        n = z.shape[0]

        if self.standardize:
            matrix = z.T @ z / n  # sample correlation matrix
        else:
            x = z * scale  # centred returns in original units
            matrix = x.T @ x / n  # divisor-n sample covariance

        eigenvalues, eigenvectors = sorted_eigh(matrix)
        shrunk = qis_shrink_eigenvalues(eigenvalues, n_eff)

        if self.standardize:
            cov = rebuild_correlation(shrunk, eigenvectors) * np.outer(scale, scale)
        else:
            cov = (eigenvectors * shrunk) @ eigenvectors.T
            cov = (cov + cov.T) / 2

        self.covariance_: pd.DataFrame = wrap_covariance(cov, returns.columns)
        self.eigenvalues_: np.ndarray = eigenvalues
        self.shrunk_eigenvalues_: np.ndarray = shrunk
        return self


__all__ = ["QuadraticInverseShrinkage", "qis_shrink_eigenvalues"]
