"""Portfolio constructions used in the study.

Each construction is a function ``covariance -> weights``: it takes a labelled
square covariance DataFrame and returns a labelled weight Series that sums to
one. Mean-variance optimisation is excluded by design (no expected-return
input).

==========================  ==================================  ===============
Construction                Rule                                Uses covariance
==========================  ==================================  ===============
``equal_weight``            ``w_i = 1/p``                       no
``risk_parity``             equal risk contributions, ``w>=0``  yes (``Σw``)
``min_variance_long_only``  ``min w'Σw``, ``1'w=1``, ``w>=0``   yes
``min_variance_long_short`` ``min w'Σw``, ``1'w=1``             yes (``Σ⁻¹``)
==========================  ==================================  ===============

Numerical notes. Covariances of daily returns have entries of order 1e-4 to
1e-8, which makes optimiser tolerances awkward; every solver therefore works on
the covariance divided by its mean variance. Weights are invariant to that
rescaling. Long-only minimum variance uses SLSQP with an analytic gradient,
starting from the long-short solution clipped to the simplex; if the long-short
solution is already non-negative it is returned directly (it is then exactly
optimal). Risk parity uses cyclical coordinate descent on the convex
formulation of Spinu (2013), following Griveau-Billion, Richard and Roncalli
(2013): each coordinate update solves a scalar quadratic in closed form.

:func:`with_excluded` restricts any construction to a sub-universe (for example
excluding a cash-like instrument from the optimisation) and returns zero weight
for the excluded assets.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

WeightFn = Callable[[pd.DataFrame], pd.Series]


# ── validation ────────────────────────────────────────────────────────────────


def validate_covariance(cov: pd.DataFrame) -> np.ndarray:
    """Check that ``cov`` is a usable covariance matrix and return it as an array.

    Raises:
        TypeError: If ``cov`` is not a DataFrame.
        ValueError: If it is empty, not square, has mismatched row and column
            labels, contains non-finite values, is not symmetric, or is not
            positive definite.
    """
    if not isinstance(cov, pd.DataFrame):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"cov must be a pd.DataFrame, got {type(cov).__name__}")
    if cov.empty or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"cov must be a non-empty square matrix, got shape {cov.shape}")
    if not cov.index.equals(cov.columns):
        raise ValueError("cov must have identical row and column labels in the same order")
    m = cov.to_numpy(dtype=float)
    if not np.isfinite(m).all():
        raise ValueError("cov contains non-finite values")
    scale = np.abs(m).max()
    if not np.allclose(m, m.T, rtol=0, atol=1e-10 * scale):
        raise ValueError("cov is not symmetric")
    try:
        np.linalg.cholesky((m + m.T) / 2)
    except np.linalg.LinAlgError as err:
        raise ValueError("cov is not positive definite") from err
    sym: np.ndarray = (m + m.T) / 2
    return sym


def _normalised(m: np.ndarray) -> np.ndarray:
    """Covariance divided by its mean variance (weights are scale-invariant)."""
    scaled: np.ndarray = m / np.mean(np.diag(m))
    return scaled


def _as_series(w: np.ndarray, cov: pd.DataFrame) -> pd.Series:
    return pd.Series(w / w.sum(), index=cov.index, name="weight")


# ── constructions ─────────────────────────────────────────────────────────────


def equal_weight(cov: pd.DataFrame) -> pd.Series:
    """Equal weight ``1/p``. Uses only the labels of ``cov``, never its values."""
    validate_covariance(cov)
    return _as_series(np.ones(cov.shape[0]), cov)


def min_variance_long_short(cov: pd.DataFrame) -> pd.Series:
    """Global minimum variance with the budget constraint only.

    Closed form ``w = Σ⁻¹1 / (1'Σ⁻¹1)``; weights may be negative.
    """
    m = _normalised(validate_covariance(cov))
    x = np.linalg.solve(m, np.ones(m.shape[0]))
    return _as_series(x, cov)


def min_variance_long_only(cov: pd.DataFrame, tol: float = 1e-14) -> pd.Series:
    """Global minimum variance with ``w >= 0`` and ``1'w = 1`` (quadratic programme).

    Raises:
        RuntimeError: If the optimiser does not converge.
    """
    m = _normalised(validate_covariance(cov))
    p = m.shape[0]
    x = np.linalg.solve(m, np.ones(p))
    w_ls = x / x.sum()
    if bool((w_ls >= 0).all()):
        return _as_series(w_ls, cov)

    w0 = np.clip(w_ls, 0.0, None)
    w0 = w0 / w0.sum() if w0.sum() > 0 else np.full(p, 1.0 / p)
    res = minimize(
        lambda w: float(w @ m @ w),
        w0,
        jac=lambda w: 2.0 * (m @ w),
        method="SLSQP",
        bounds=[(0.0, None)] * p,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(p)}],
        options={"maxiter": 2000, "ftol": tol},
    )
    if not res.success:
        raise RuntimeError(f"long-only minimum variance did not converge: {res.message}")
    w = np.clip(res.x, 0.0, None)
    w = w / w.sum()
    return _as_series(_polish_active_set(w, m), cov)


def _polish_active_set(w: np.ndarray, m: np.ndarray, threshold: float = 1e-8) -> np.ndarray:
    """Refine an approximate long-only solution to the exact optimum.

    On the set of held assets the long-only optimum is the closed-form
    minimum-variance portfolio of that sub-universe. The refined weights are
    accepted only if they satisfy the optimality (KKT) conditions: all held
    weights positive, and no excluded asset has a marginal variance below the
    portfolio variance. Otherwise the solver's weights are returned unchanged.
    """
    held = w > threshold
    x = np.linalg.solve(m[np.ix_(held, held)], np.ones(int(held.sum())))
    if not bool((x > 0).all()):
        return w
    refined = np.zeros_like(w)
    refined[held] = x / x.sum()
    grad = m @ refined
    lam = refined @ grad
    if bool((grad[~held] >= lam * (1 - 1e-10)).all()):
        return refined
    return w


def risk_parity(cov: pd.DataFrame, tol: float = 1e-12, max_sweeps: int = 10_000) -> pd.Series:
    """Equal risk contributions: ``w_i (Σw)_i = w'Σw / p`` for every ``i``, ``w > 0``.

    Solves ``min_y ½ y'Σy - (1/p) Σ log y_i`` by cyclical coordinate descent;
    the stationarity condition ``y_i (Σy)_i = 1/p`` is exactly equal risk
    contribution, and ``w = y / 1'y``.

    Raises:
        RuntimeError: If coordinate descent does not converge.
    """
    m = _normalised(validate_covariance(cov))
    p = m.shape[0]
    b = 1.0 / p
    diag = np.diag(m)
    y = 1.0 / np.sqrt(diag)
    y = y / np.sqrt(y @ m @ y)
    for _ in range(max_sweeps):
        y_old = y.copy()
        for i in range(p):
            c = m[i] @ y - diag[i] * y[i]  # Σ_{j≠i} Σ_ij y_j
            y[i] = (-c + np.sqrt(c * c + 4.0 * diag[i] * b)) / (2.0 * diag[i])
        if np.max(np.abs(y - y_old) / y) < tol:
            return _as_series(y, cov)
    raise RuntimeError("risk parity coordinate descent did not converge")


# ── helpers ───────────────────────────────────────────────────────────────────


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Fraction of portfolio variance contributed by each asset: ``w_i (Σw)_i / w'Σw``."""
    w = weights.reindex(cov.index).to_numpy(dtype=float)
    m = cov.to_numpy(dtype=float)
    mw = m @ w
    return pd.Series(w * mw / (w @ mw), index=cov.index, name="risk_contribution")


def with_excluded(build: WeightFn, exclude: Iterable[str]) -> WeightFn:
    """Wrap a construction so it ignores the assets in ``exclude``.

    The wrapped function builds weights on the remaining assets and returns a
    Series over the full universe, with zero weight on the excluded ones.
    Excluded names absent from the covariance are ignored.
    """
    excluded = set(exclude)

    def wrapped(cov: pd.DataFrame) -> pd.Series:
        keep = [c for c in cov.index if c not in excluded]
        if len(keep) == 0:
            raise ValueError("every asset is excluded")
        sub = cov.loc[keep, keep]
        return build(sub).reindex(cov.index, fill_value=0.0).rename("weight")

    return wrapped


@dataclass(frozen=True)
class Construction:
    """A named portfolio construction.

    Attributes:
        name: Identifier used in results tables.
        build: The weight function.
        uses_covariance: Whether the weights depend on the covariance values.
            Estimator-independent constructions give Criterion A its clean
            forecast test; estimator-dependent ones give a consistency test.
    """

    name: str
    build: WeightFn
    uses_covariance: bool


CONSTRUCTIONS: dict[str, Construction] = {
    c.name: c
    for c in (
        Construction("equal_weight", equal_weight, uses_covariance=False),
        Construction("risk_parity", risk_parity, uses_covariance=True),
        Construction("min_variance_long_only", min_variance_long_only, uses_covariance=True),
        Construction("min_variance_long_short", min_variance_long_short, uses_covariance=True),
    )
}


__all__ = [
    "CONSTRUCTIONS",
    "Construction",
    "WeightFn",
    "equal_weight",
    "min_variance_long_only",
    "min_variance_long_short",
    "risk_contributions",
    "risk_parity",
    "validate_covariance",
    "with_excluded",
]
