"""Tests for the Marchenko-Pastur clipping estimator and spectral helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.estimators import _spectral, base
from portfolio_risk.estimators.marchenko_pastur import MarchenkoPasturClipping
from portfolio_risk.estimators.sample import SampleCovariance

TRUE_RHO = 0.3


def _one_factor(n: int, p: int, seed: int = 0, vols: np.ndarray | None = None) -> pd.DataFrame:
    """Panel with uniform true correlation TRUE_RHO and optional column volatilities."""
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((n, 1))
    z = np.sqrt(TRUE_RHO) * f + np.sqrt(1 - TRUE_RHO) * rng.standard_normal((n, p))
    if vols is not None:
        z = z * vols
    return pd.DataFrame(z, columns=[f"x{i}" for i in range(p)])


def _true_corr(p: int) -> np.ndarray:
    return TRUE_RHO * np.ones((p, p)) + (1 - TRUE_RHO) * np.eye(p)


def _corr_of(cov: pd.DataFrame) -> np.ndarray:
    c = cov.to_numpy()
    d = np.sqrt(np.diag(c))
    return c / np.outer(d, d)


@pytest.fixture
def returns() -> pd.DataFrame:
    vols = np.geomspace(1e-4, 4e-2, 40)  # heterogeneous, like the real panel
    return _one_factor(n=500, p=40, vols=vols)


# ── spectral helpers ──────────────────────────────────────────────────────────


class TestStandardize:
    def test_unit_variance_and_zero_mean(self, returns: pd.DataFrame) -> None:
        z, _, _ = _spectral.standardize(returns, assume_centered=False)
        np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose((z**2).mean(axis=0), 1.0, atol=1e-12)

    def test_effective_sample_size(self, returns: pd.DataFrame) -> None:
        assert _spectral.standardize(returns, assume_centered=False)[2] == len(returns) - 1
        assert _spectral.standardize(returns, assume_centered=True)[2] == len(returns)

    def test_rejects_zero_variance(self) -> None:
        df = pd.DataFrame({"a": [0.1, -0.1, 0.2], "b": [1.0, 1.0, 1.0]})
        with pytest.raises(ValueError, match="zero-variance"):
            _spectral.standardize(df, assume_centered=False)


class TestSpectralHelpers:
    def test_sorted_eigh_descending_and_reconstructs(self) -> None:
        m = _true_corr(5)
        w, v = _spectral.sorted_eigh(m)
        assert np.all(np.diff(w) <= 0)
        np.testing.assert_allclose((v * w) @ v.T, m, atol=1e-12)

    def test_rebuild_has_unit_diagonal(self) -> None:
        w, v = _spectral.sorted_eigh(_true_corr(6))
        rebuilt = _spectral.rebuild_correlation(w * np.linspace(0.5, 1.5, 6), v)
        np.testing.assert_allclose(np.diag(rebuilt), 1.0, atol=1e-12)
        np.testing.assert_allclose(rebuilt, rebuilt.T, atol=1e-14)


# ── estimator ─────────────────────────────────────────────────────────────────


class TestFit:
    def test_fit_returns_self(self, returns: pd.DataFrame) -> None:
        est = MarchenkoPasturClipping()
        assert est.fit(returns) is est

    def test_labels_preserved(self, returns: pd.DataFrame) -> None:
        cov = MarchenkoPasturClipping().fit(returns).covariance_
        assert list(cov.index) == list(returns.columns)
        assert list(cov.columns) == list(returns.columns)

    def test_symmetric_positive_definite(self, returns: pd.DataFrame) -> None:
        cov = MarchenkoPasturClipping().fit(returns).covariance_.to_numpy()
        np.testing.assert_allclose(cov, cov.T, rtol=1e-12, atol=0)
        assert (
            np.linalg.eigvalsh(_corr_of(MarchenkoPasturClipping().fit(returns).covariance_)).min()
            > 0
        )

    def test_preserves_sample_variances(self, returns: pd.DataFrame) -> None:
        est = MarchenkoPasturClipping().fit(returns)
        s = SampleCovariance().fit(returns).covariance_.to_numpy()
        np.testing.assert_allclose(np.diag(est.covariance_.to_numpy()), np.diag(s), rtol=1e-10)

    def test_invariant_to_column_scaling(self, returns: pd.DataFrame) -> None:
        scaled = returns.copy()
        scaled["x0"] *= 1000.0
        a = MarchenkoPasturClipping().fit(returns)
        b = MarchenkoPasturClipping().fit(scaled)
        np.testing.assert_allclose(_corr_of(a.covariance_), _corr_of(b.covariance_), atol=1e-10)
        assert a.n_signal_ == b.n_signal_


class TestDiagnostics:
    def test_eigenvalues_sorted_and_sum_to_p(self, returns: pd.DataFrame) -> None:
        est = MarchenkoPasturClipping().fit(returns)
        assert len(est.eigenvalues_) == returns.shape[1]
        assert np.all(np.diff(est.eigenvalues_) <= 0)
        assert est.eigenvalues_.sum() == pytest.approx(returns.shape[1])

    def test_edge_formula_uses_effective_sample_size(self, returns: pd.DataFrame) -> None:
        n, p = returns.shape
        demeaned = MarchenkoPasturClipping().fit(returns)
        centered = MarchenkoPasturClipping(assume_centered=True).fit(returns)
        assert demeaned.edge_ == pytest.approx((1 + np.sqrt(p / (n - 1))) ** 2)
        assert centered.edge_ == pytest.approx((1 + np.sqrt(p / n)) ** 2)

    def test_n_signal_counts_eigenvalues_above_edge(self, returns: pd.DataFrame) -> None:
        est = MarchenkoPasturClipping().fit(returns)
        assert est.n_signal_ == int((est.eigenvalues_ > est.edge_).sum())


class TestFiltering:
    def test_pure_noise_gives_diagonal_covariance(self) -> None:
        """No signal: every eigenvalue is averaged to 1, so R_clean = I."""
        rng = np.random.default_rng(5)
        df = pd.DataFrame(rng.standard_normal((1000, 20)) * np.linspace(0.5, 3.0, 20))
        est = MarchenkoPasturClipping().fit(df)
        assert est.n_signal_ == 0
        np.testing.assert_allclose(_corr_of(est.covariance_), np.eye(20), atol=1e-10)

    def test_detects_single_factor(self, returns: pd.DataFrame) -> None:
        assert MarchenkoPasturClipping().fit(returns).n_signal_ == 1

    def test_closer_to_truth_than_sample(self, returns: pd.DataFrame) -> None:
        p = returns.shape[1]
        mp_err = np.linalg.norm(
            _corr_of(MarchenkoPasturClipping().fit(returns).covariance_) - _true_corr(p)
        )
        sample_err = np.linalg.norm(returns.corr().to_numpy() - _true_corr(p))
        assert mp_err < 0.6 * sample_err

    def test_better_conditioned_than_sample(self, returns: pd.DataFrame) -> None:
        mp = np.linalg.cond(_corr_of(MarchenkoPasturClipping().fit(returns).covariance_))
        sample = np.linalg.cond(returns.corr().to_numpy())
        assert mp < sample

    def test_noise_bulk_stays_near_clipped_value(self, returns: pd.DataFrame) -> None:
        """Renormalisation spreads the flattened bulk only slightly."""
        est = MarchenkoPasturClipping().fit(returns)
        w = np.sort(np.linalg.eigvalsh(_corr_of(est.covariance_)))[::-1]
        bulk = w[est.n_signal_ :]
        clipped_value = est.eigenvalues_[est.n_signal_ :].mean()
        assert np.abs(bulk - clipped_value).max() < 0.15
        assert bulk.mean() == pytest.approx(clipped_value, abs=0.02)

    def test_works_when_p_exceeds_n(self) -> None:
        """Sample correlation is singular; the clipped one must not be."""
        df = _one_factor(n=30, p=50, seed=7)
        est = MarchenkoPasturClipping().fit(df)
        assert np.linalg.eigvalsh(_corr_of(est.covariance_)).min() > 0


class TestValidationAndProtocol:
    def test_rejects_nan(self) -> None:
        df = pd.DataFrame({"a": [0.1, np.nan, 0.2], "b": [0.0, 0.1, 0.2]})
        with pytest.raises(ValueError, match="NaN"):
            MarchenkoPasturClipping().fit(df)

    def test_rejects_zero_variance_column(self) -> None:
        df = pd.DataFrame({"a": [0.1, -0.2, 0.3, 0.0], "b": [0.5, 0.5, 0.5, 0.5]})
        with pytest.raises(ValueError, match="zero-variance"):
            MarchenkoPasturClipping().fit(df)

    def test_satisfies_protocol_after_fit(self, returns: pd.DataFrame) -> None:
        assert isinstance(MarchenkoPasturClipping().fit(returns), base.CovarianceEstimator)

    def test_fails_protocol_before_fit(self) -> None:
        assert not isinstance(MarchenkoPasturClipping(), base.CovarianceEstimator)
