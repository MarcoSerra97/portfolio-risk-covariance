"""Shared helpers for estimators that clean the correlation matrix's spectrum.

The Marchenko-Pastur clipping and nonlinear-shrinkage estimators follow the
same recipe: standardise each column to unit variance, modify the eigenvalues
of the resulting correlation matrix, then rescale by the sample volatilities.
Variances are therefore always kept at their sample values, and only the
correlations are cleaned. This module holds the parts of that recipe the
estimators share.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def standardize(returns: pd.DataFrame, assume_centered: bool) -> tuple[np.ndarray, np.ndarray, int]:
    """Centre and scale a return panel to unit (divisor-``n``) variance.

    Args:
        returns: Validated return panel, observations in rows.
        assume_centered: If ``True``, skip de-meaning.

    Returns:
        ``(z, scale, n_eff)``: the standardised ``(n, p)`` array, the length-``p``
        vector of divisor-``n`` volatilities used to scale it, and the
        effective sample size, which is ``n - 1`` when the data were de-meaned
        (one degree of freedom spent on the means) and ``n`` otherwise.

    Raises:
        ValueError: If any column has zero variance.
    """
    x = returns.to_numpy(dtype=float)
    centered = x if assume_centered else x - x.mean(axis=0)
    scale = np.sqrt((centered**2).mean(axis=0))
    if bool((scale == 0).any()):
        zero_cols = list(returns.columns[scale == 0])
        raise ValueError(f"cannot standardize zero-variance column(s): {zero_cols}")
    n = x.shape[0]
    n_eff = n if assume_centered else n - 1
    return centered / scale, scale, n_eff


def sorted_eigh(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecomposition of a symmetric matrix, eigenvalues in descending order."""
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def rebuild_correlation(eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
    """Rebuild a correlation matrix from modified eigenvalues.

    Modifying eigenvalues disturbs the unit diagonal, so the rebuilt matrix is
    rescaled as ``D^{-1/2} M D^{-1/2}`` with ``D = diag(M)``. This restores the
    unit diagonal and keeps the matrix positive definite, at the cost of
    slightly perturbing the modified eigenvalues.
    """
    m = (eigenvectors * eigenvalues) @ eigenvectors.T
    m = (m + m.T) / 2  # remove floating-point asymmetry
    d = np.sqrt(np.diag(m))
    rebuilt: np.ndarray = m / np.outer(d, d)
    return rebuilt


__all__ = ["rebuild_correlation", "sorted_eigh", "standardize"]
