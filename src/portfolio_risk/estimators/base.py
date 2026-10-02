"""Shared interface and helpers for covariance estimators.

The project compares several covariance estimators (sample, Ledoit-Wolf linear
shrinkage, nonlinear shrinkage, DCC-GARCH) on a common evaluation harness.
This module defines the interface they all satisfy and a small set of helpers
every implementation can lean on.

Design decisions
----------------

**Protocol over abstract base class.** The protocol-based interface (duck
typing with static checking via :mod:`mypy`) suits a project whose estimators
come from different origins — custom implementations, scikit-learn wrappers,
and ``arch``-package wrappers for DCC-GARCH. An abstract base class would
force every implementation into our inheritance tree, which is awkward when
wrapping an external estimator that already has its own class hierarchy.
Protocol conformance is checked at type-check time (and optionally at runtime
through :func:`isinstance`), without the composition acrobatics.

**scikit-learn convention.** The ``fit(returns) -> self`` + ``covariance_``
attribute shape is deliberately identical to :class:`sklearn.covariance.LedoitWolf`,
so wrapping sklearn estimators (and later swapping between them in evaluation
code) is a one-line adaptation.

**Caller handles NaN.** :func:`validate_returns` raises on any missing value;
NaN cleaning belongs in the data layer, not inside each estimator. The data
pipeline already enforces this invariant.
"""

from __future__ import annotations

from typing import Protocol, Self, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class CovarianceEstimator(Protocol):
    """Protocol every covariance estimator in this project satisfies.

    The contract mirrors scikit-learn's convention: :meth:`fit` returns the
    estimator itself (enabling ``est.fit(X).covariance_``), and the fitted
    matrix lives on the ``covariance_`` attribute as a labelled
    :class:`pandas.DataFrame`.

    Attributes:
        covariance_: Fitted covariance matrix. Square DataFrame with the
            input return columns on both axes. Populated after :meth:`fit`
            succeeds; accessing it before is a programming error (the
            attribute will not exist and :class:`AttributeError` is raised).
    """

    covariance_: pd.DataFrame

    def fit(self, returns: pd.DataFrame) -> Self:
        """Fit the estimator to ``returns`` and return ``self`` for chaining.

        Args:
            returns: Clean return panel — rows are observations (typically
                daily log-returns), columns are instruments. Must contain no
                NaN values; see :func:`validate_returns`.

        Returns:
            The estimator itself, with ``covariance_`` populated.
        """
        ...


def validate_returns(returns: pd.DataFrame) -> None:
    """Raise if ``returns`` is not a valid input for a covariance estimator.

    Centralised validation keeps every estimator's :meth:`fit` body short and
    guarantees a uniform error experience. Rejecting NaN at the door avoids
    pandas's default pairwise-deletion behaviour in :meth:`DataFrame.cov`,
    which silently produces an inconsistent matrix across pairs — a subtle
    bug that would otherwise propagate into every estimator.

    Args:
        returns: The input to validate.

    Raises:
        TypeError: If ``returns`` is not a :class:`pandas.DataFrame`.
        ValueError: If ``returns`` is empty, has fewer than two columns, or
            contains any NaN value.
    """
    if not isinstance(returns, pd.DataFrame):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"returns must be a pd.DataFrame, got {type(returns).__name__}")
    if returns.empty:
        raise ValueError("returns is empty")
    if returns.shape[1] < 2:
        raise ValueError(f"returns must have at least 2 columns (got {returns.shape[1]})")
    if bool(returns.isna().any().any()):
        raise ValueError(
            "returns contains NaN values; estimators require a clean panel. "
            "Drop or fill NaN in the data layer before fitting."
        )


def wrap_covariance(cov: np.ndarray, columns: pd.Index) -> pd.DataFrame:
    """Wrap a raw covariance matrix as a labelled :class:`pandas.DataFrame`.

    Convenience for estimators that compute the covariance as a plain NumPy
    array (typically scikit-learn wrappers) and need to attach ticker labels
    to both axes before storing it on ``self.covariance_``.

    Args:
        cov: Square ``(p, p)`` covariance matrix.
        columns: Length-``p`` index of ticker labels used on both axes.

    Returns:
        The matrix with ``columns`` as both ``index`` and ``columns``.

    Raises:
        ValueError: If ``cov`` is not square or does not match ``columns``.
    """
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"cov must be square, got shape {cov.shape}")
    if cov.shape[0] != len(columns):
        raise ValueError(f"cov shape {cov.shape} does not match columns length {len(columns)}")
    return pd.DataFrame(cov, index=columns, columns=columns)


__all__ = [
    "CovarianceEstimator",
    "validate_returns",
    "wrap_covariance",
]
