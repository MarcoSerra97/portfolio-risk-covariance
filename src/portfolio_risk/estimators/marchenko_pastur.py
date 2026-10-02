"""Marchenko-Pastur eigenvalue clipping on the correlation matrix.

Random-matrix filtering in the style of Laloux, Cizeau, Bouchaud and Potters
(1999). If returns were pure noise with unit variance, the eigenvalues of the
sample correlation matrix would follow the Marchenko-Pastur law, supported on
``[(1 - sqrt(q))^2, (1 + sqrt(q))^2]`` with ``q = p / n_eff``. Eigenvalues
above the upper edge carry signal (market, sector, rates, ... factors); those
below are indistinguishable from noise.

The procedure:

1. Standardise each column to unit variance; form the correlation matrix ``R``.
2. Eigendecompose ``R``. Keep eigenvalues above ``lambda_+ = (1 + sqrt(q))^2``;
   replace every eigenvalue below it by their average. This preserves the
   trace, so the result is still a valid correlation spectrum.
3. Rebuild the matrix and rescale it back to a unit diagonal.
4. Multiply by the sample volatilities.

Variances are therefore kept exactly at their sample values; only correlations
are cleaned. Working on the correlation matrix is not optional here: the MP
null model assumes a common variance across variables, which is badly violated
by a raw covariance matrix whose variances span orders of magnitude.

Caveats:

* Step 3 slightly perturbs the flattened noise eigenvalues, spreading them
  around their clipped value. See
  :func:`~portfolio_risk.estimators._spectral.rebuild_correlation`.
* The edge assumes Gaussian, independent observations and noise variance 1.
  Fat tails, volatility clustering and the variance absorbed by strong factors
  blur the boundary; the filter is a well-motivated heuristic, not a test with
  exact size.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd

from portfolio_risk.estimators._spectral import rebuild_correlation, sorted_eigh, standardize
from portfolio_risk.estimators.base import validate_returns, wrap_covariance


class MarchenkoPasturClipping:
    """Covariance estimator that clips sub-MP-edge correlation eigenvalues.

    Conforms to the :class:`~portfolio_risk.estimators.base.CovarianceEstimator`
    protocol.

        >>> est = MarchenkoPasturClipping().fit(returns)  # doctest: +SKIP
        >>> est.n_signal_                                  # doctest: +SKIP

    Attributes:
        assume_centered: If ``True``, the columns are assumed to already have
            zero mean; no de-meaning is applied and the effective sample size
            is ``n`` rather than ``n - 1``. Default ``False``.
        covariance_: Fitted ``(p, p)`` covariance matrix as a labelled
            DataFrame. Populated by :meth:`fit`.
        eigenvalues_: Eigenvalues of the sample correlation matrix, in
            descending order (before clipping). Useful as a diagnostic.
        edge_: The Marchenko-Pastur upper edge ``(1 + sqrt(p / n_eff))^2``.
        n_signal_: Number of eigenvalues above the edge, i.e. kept as signal.
    """

    def __init__(self, assume_centered: bool = False) -> None:
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
            ValueError: If some column has zero variance.
        """
        validate_returns(returns)
        z, scale, n_eff = standardize(returns, self.assume_centered)
        n, p = z.shape

        corr = z.T @ z / n
        eigenvalues, eigenvectors = sorted_eigh(corr)
        edge = (1.0 + np.sqrt(p / n_eff)) ** 2
        noise = eigenvalues < edge

        clipped = eigenvalues.copy()
        if bool(noise.any()):
            clipped[noise] = eigenvalues[noise].mean()

        cleaned = rebuild_correlation(clipped, eigenvectors)
        cov = cleaned * np.outer(scale, scale)

        self.covariance_: pd.DataFrame = wrap_covariance(cov, returns.columns)
        self.eigenvalues_: np.ndarray = eigenvalues
        self.edge_: float = float(edge)
        self.n_signal_: int = int((~noise).sum())
        return self


__all__ = ["MarchenkoPasturClipping"]
