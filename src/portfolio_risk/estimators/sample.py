"""Sample covariance estimator, backed by scikit-learn's EmpiricalCovariance.

Computes the classical maximum-likelihood sample covariance

.. math::

    \\hat{\\Sigma} = \\frac{1}{n} (X - \\bar{X})^{\\top} (X - \\bar{X}),

with divisor ``n`` (not ``n - 1``), consistent with every other estimator in
this package. See :mod:`portfolio_risk.estimators` for why the study uses the
ML convention throughout.

This is the baseline the shrinkage estimators improve upon: well-behaved when
``n >> p``, increasingly ill-conditioned as ``p / n`` grows (the
Marchenko-Pastur regime), where its extreme eigenvalues are systematically
too spread out.
"""

from __future__ import annotations

from typing import Self

import pandas as pd
from sklearn.covariance import EmpiricalCovariance

from portfolio_risk.estimators.base import validate_returns, wrap_covariance


class SampleCovariance:
    """Maximum-likelihood sample covariance (divisor ``n``).

    Thin wrapper around :class:`sklearn.covariance.EmpiricalCovariance` that
    validates the input, attaches ticker labels to both axes of the result,
    and conforms to the
    :class:`~portfolio_risk.estimators.base.CovarianceEstimator` protocol.

        >>> est = SampleCovariance().fit(returns)  # doctest: +SKIP
        >>> est.covariance_                         # doctest: +SKIP

    Attributes:
        assume_centered: If ``True``, the columns are assumed to already have
            zero mean and no de-meaning is applied. Default ``False``.
        covariance_: Fitted ``(p, p)`` covariance matrix as a labelled
            DataFrame. Populated by :meth:`fit`.
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
            The estimator itself, with ``covariance_`` populated.
        """
        validate_returns(returns)
        # store_precision=False skips computing the inverse, which we don't need.
        sk = EmpiricalCovariance(store_precision=False, assume_centered=self.assume_centered)
        sk.fit(returns.to_numpy())
        self.covariance_: pd.DataFrame = wrap_covariance(sk.covariance_, returns.columns)
        return self


__all__ = ["SampleCovariance"]
