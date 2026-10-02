"""Ledoit-Wolf linear shrinkage estimator, backed by scikit-learn.

Shrinks the sample covariance ``S`` toward a structured target ``F``:

.. math::

    \\hat{\\Sigma} = (1 - \\delta) S + \\delta F,

with the shrinkage intensity ``delta`` chosen analytically to minimise the
expected Frobenius loss (Ledoit & Wolf, 2004, "A well-conditioned estimator
for large-dimensional covariance matrices"). The computation is delegated to
:class:`sklearn.covariance.LedoitWolf`, which uses the same divisor-``n``
convention as the rest of this package.

Two targets are available through the ``standardize`` flag:

``standardize=False`` (default) — scaled identity, ``F = mu * I``
    The textbook LW 2004 estimator; ``mu`` is the average sample variance.
    Every diagonal entry is pulled toward the same common variance. This is
    exactly sklearn's output.

``standardize=True`` — correlation shrinkage
    Columns are scaled to unit variance, LW is applied to the standardised
    data, and the result is rescaled. Correlations are shrunk toward zero
    while every variance stays at its sample value; equivalently, the target
    is ``F = diag(S)``. Preferable when instruments have very different
    volatilities: on this project's panel the variance ratio between the most
    and least volatile instrument is about 65,000, and the scaled-identity
    target would inflate the variance of low-volatility instruments (such as
    T-bill ETFs) by orders of magnitude.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from portfolio_risk.estimators.base import validate_returns, wrap_covariance


class LedoitWolfShrinkage:
    """Ledoit-Wolf linear shrinkage covariance estimator.

    Conforms to the :class:`~portfolio_risk.estimators.base.CovarianceEstimator`
    protocol.

        >>> est = LedoitWolfShrinkage().fit(returns)  # doctest: +SKIP
        >>> est.covariance_, est.shrinkage_           # doctest: +SKIP

    Attributes:
        standardize: If ``True``, shrink correlations toward zero while
            preserving sample variances (target ``diag(S)``). If ``False``
            (default), shrink toward the scaled identity ``mu * I``.
        assume_centered: If ``True``, the columns are assumed to already have
            zero mean and no de-meaning is applied. Default ``False``.
        covariance_: Fitted ``(p, p)`` covariance matrix as a labelled
            DataFrame. Populated by :meth:`fit`.
        shrinkage_: Fitted shrinkage intensity ``delta`` in ``[0, 1]``:
            ``0`` means the sample covariance was kept unchanged, ``1`` means
            it was replaced entirely by the target. Populated by :meth:`fit`.
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
            The estimator itself, with ``covariance_`` and ``shrinkage_``
            populated.

        Raises:
            ValueError: If ``standardize=True`` and some column has zero
                variance (it cannot be scaled to unit variance).
        """
        validate_returns(returns)
        x = returns.to_numpy(dtype=float)

        if not self.standardize:
            sk = LedoitWolf(store_precision=False, assume_centered=self.assume_centered)
            sk.fit(x)
            cov = sk.covariance_
        else:
            centered = x if self.assume_centered else x - x.mean(axis=0)
            # Divisor-n scale, consistent with the package convention.
            scale = np.sqrt((centered**2).mean(axis=0))
            if bool((scale == 0).any()):
                zero_cols = list(returns.columns[scale == 0])
                raise ValueError(f"cannot standardize zero-variance column(s): {zero_cols}")
            sk = LedoitWolf(store_precision=False, assume_centered=True)
            sk.fit(centered / scale)
            # Rescale the shrunk correlation matrix back to covariance units.
            cov = sk.covariance_ * np.outer(scale, scale)

        self.covariance_: pd.DataFrame = wrap_covariance(cov, returns.columns)
        self.shrinkage_: float = float(sk.shrinkage_)
        return self


__all__ = ["LedoitWolfShrinkage"]
