"""Tests for the QIS nonlinear shrinkage estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.estimators import base
from portfolio_risk.estimators.ledoit_wolf import LedoitWolfShrinkage
from portfolio_risk.estimators.marchenko_pastur import MarchenkoPasturClipping
from portfolio_risk.estimators.qis import QuadraticInverseShrinkage, qis_shrink_eigenvalues
from portfolio_risk.estimators.sample import SampleCovariance

TRUE_RHO = 0.3


def _one_factor(n: int, p: int, seed: int = 0, vols: np.ndarray | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((n, 1))
    z = np.sqrt(TRUE_RHO) * f + np.sqrt(1 - TRUE_RHO) * rng.standard_normal((n, p))
    if vols is not None:
        z = z * vols
    return pd.DataFrame(z, columns=[f"x{i}" for i in range(p)])


def _corr_of(cov: pd.DataFrame) -> np.ndarray:
    c = cov.to_numpy()
    d = np.sqrt(np.diag(c))
    return c / np.outer(d, d)


@pytest.fixture
def returns() -> pd.DataFrame:
    return _one_factor(n=500, p=40, vols=np.geomspace(1e-4, 4e-2, 40))


# ── the eigenvalue map ────────────────────────────────────────────────────────


class TestShrinkEigenvalues:
    """Regression values cross-checked against the authors' reference code
    (github.com/pald22/covShrinkage, QIS.py) in a development sandbox."""

    def test_regression_p_less_than_n(self) -> None:
        ev = np.array([9.0, 5.0, 3.0, 2.0, 1.5, 1.0, 0.7, 0.4])
        expected = np.array(
            [6.616963610068529, 4.239289043542687, 2.817309300563589, 2.156368567279641,
             2.094389143549868, 1.722206862138135, 1.677794717378791, 1.275678755478762]
        )  # fmt: skip
        np.testing.assert_allclose(qis_shrink_eigenvalues(ev, n_eff=20), expected, rtol=1e-12)

    def test_regression_p_greater_than_n(self) -> None:
        ev = np.array([6.0, 3.0, 1.0, 0.5, 0.0, 0.0])
        expected = np.array(
            [2.769260069259504, 2.287523832552858, 1.003044905548356,
             0.69329345182691, 1.873438870406186, 1.873438870406186]
        )  # fmt: skip
        np.testing.assert_allclose(qis_shrink_eigenvalues(ev, n_eff=4), expected, rtol=1e-12)

    def test_preserves_sum(self) -> None:
        ev = np.array([9.0, 5.0, 3.0, 2.0, 1.5, 1.0, 0.7, 0.4])
        assert qis_shrink_eigenvalues(ev, n_eff=20).sum() == pytest.approx(ev.sum())

    def test_pulls_spectrum_inward(self) -> None:
        ev = np.array([9.0, 5.0, 3.0, 2.0, 1.5, 1.0, 0.7, 0.4])
        shrunk = qis_shrink_eigenvalues(ev, n_eff=20)
        assert shrunk.max() < ev.max()
        assert shrunk.min() > ev.min()

    def test_homogeneous_of_degree_one(self) -> None:
        ev = np.array([4.0, 2.0, 1.0, 0.5])
        np.testing.assert_allclose(
            qis_shrink_eigenvalues(7.0 * ev, n_eff=12), 7.0 * qis_shrink_eigenvalues(ev, n_eff=12)
        )

    def test_rejects_unexplained_zero_eigenvalue(self) -> None:
        with pytest.raises(ValueError, match="zero eigenvalues"):
            qis_shrink_eigenvalues(np.array([3.0, 1.0, 0.0]), n_eff=10)


# ── raw-covariance variant ────────────────────────────────────────────────────


class TestRawVariant:
    def test_fit_returns_self_and_labels(self, returns: pd.DataFrame) -> None:
        est = QuadraticInverseShrinkage()
        assert est.fit(returns) is est
        assert list(est.covariance_.index) == list(returns.columns)
        assert list(est.covariance_.columns) == list(returns.columns)

    def test_symmetric_positive_definite(self, returns: pd.DataFrame) -> None:
        cov = QuadraticInverseShrinkage().fit(returns).covariance_.to_numpy()
        np.testing.assert_allclose(cov, cov.T, rtol=1e-10, atol=0)
        assert np.linalg.eigvalsh(cov).min() > 0

    def test_preserves_trace_of_sample(self, returns: pd.DataFrame) -> None:
        qis = QuadraticInverseShrinkage().fit(returns).covariance_.to_numpy()
        s = SampleCovariance().fit(returns).covariance_.to_numpy()
        assert np.trace(qis) == pytest.approx(np.trace(s))

    def test_keeps_sample_eigenvectors(self) -> None:
        df = _one_factor(n=200, p=15, seed=2)
        qis = QuadraticInverseShrinkage().fit(df).covariance_.to_numpy()
        s = SampleCovariance().fit(df).covariance_.to_numpy()
        _, u = np.linalg.eigh(s)
        rotated = u.T @ qis @ u
        off = ~np.eye(15, dtype=bool)
        assert np.abs(rotated[off]).max() < 1e-10 * np.abs(rotated).max()

    def test_scale_equivariance(self, returns: pd.DataFrame) -> None:
        a = QuadraticInverseShrinkage().fit(returns).covariance_.to_numpy()
        b = QuadraticInverseShrinkage().fit(returns * 3.0).covariance_.to_numpy()
        np.testing.assert_allclose(b, 9.0 * a, rtol=1e-10)

    def test_close_to_sample_when_data_plentiful(self) -> None:
        df = _one_factor(n=50_000, p=5, seed=3)
        qis = QuadraticInverseShrinkage().fit(df).covariance_.to_numpy()
        s = SampleCovariance().fit(df).covariance_.to_numpy()
        assert np.linalg.norm(qis - s) < 0.01 * np.linalg.norm(s)

    def test_beats_linear_shrinkage_on_clustered_spectrum(self) -> None:
        """Nonlinear shrinkage's raison d'etre: a multi-cluster population spectrum."""
        p, n = 100, 300
        pop = np.concatenate([np.full(20, 10.0), np.full(40, 3.0), np.full(40, 1.0)])
        rng = np.random.default_rng(1)
        q, _ = np.linalg.qr(rng.standard_normal((p, p)))
        sigma = (q * pop) @ q.T
        chol = np.linalg.cholesky(sigma)
        losses: dict[str, list[float]] = {"sample": [], "lw": [], "qis": []}
        for _ in range(5):
            df = pd.DataFrame(rng.standard_normal((n, p)) @ chol.T)
            for name, est in [
                ("sample", SampleCovariance()),
                ("lw", LedoitWolfShrinkage()),
                ("qis", QuadraticInverseShrinkage()),
            ]:
                losses[name].append(
                    float(np.linalg.norm(est.fit(df).covariance_.to_numpy() - sigma))
                )
        mean = {k: np.mean(v) for k, v in losses.items()}
        assert mean["qis"] < mean["lw"] < mean["sample"]

    def test_works_when_p_exceeds_n(self) -> None:
        df = _one_factor(n=30, p=50, seed=7)
        cov = QuadraticInverseShrinkage().fit(df).covariance_.to_numpy()
        assert np.isfinite(cov).all()
        assert np.linalg.eigvalsh(cov).min() > 0

    def test_rejects_duplicated_column(self) -> None:
        df = _one_factor(n=100, p=5, seed=4)
        df["dup"] = df["x0"]
        with pytest.raises(ValueError, match="zero eigenvalues"):
            QuadraticInverseShrinkage().fit(df)


# ── correlation variant ───────────────────────────────────────────────────────


class TestStandardizedVariant:
    def test_preserves_sample_variances(self, returns: pd.DataFrame) -> None:
        est = QuadraticInverseShrinkage(standardize=True).fit(returns)
        s = SampleCovariance().fit(returns).covariance_.to_numpy()
        np.testing.assert_allclose(np.diag(est.covariance_.to_numpy()), np.diag(s), rtol=1e-10)

    def test_positive_definite(self, returns: pd.DataFrame) -> None:
        cov = QuadraticInverseShrinkage(standardize=True).fit(returns).covariance_
        assert np.linalg.eigvalsh(_corr_of(cov)).min() > 0

    def test_invariant_to_column_scaling(self, returns: pd.DataFrame) -> None:
        scaled = returns.copy()
        scaled["x0"] *= 1000.0
        a = QuadraticInverseShrinkage(standardize=True).fit(returns)
        b = QuadraticInverseShrinkage(standardize=True).fit(scaled)
        np.testing.assert_allclose(_corr_of(a.covariance_), _corr_of(b.covariance_), atol=1e-10)

    def test_much_closer_to_truth_than_sample(self, returns: pd.DataFrame) -> None:
        p = returns.shape[1]
        true = TRUE_RHO * np.ones((p, p)) + (1 - TRUE_RHO) * np.eye(p)
        qis = _corr_of(QuadraticInverseShrinkage(standardize=True).fit(returns).covariance_)
        lw = _corr_of(LedoitWolfShrinkage(standardize=True).fit(returns).covariance_)
        assert np.linalg.norm(qis - true) < 0.7 * np.linalg.norm(returns.corr().to_numpy() - true)
        assert np.linalg.norm(qis - true) < np.linalg.norm(lw - true)

    def test_comparable_to_mp_on_one_factor_data(self, returns: pd.DataFrame) -> None:
        """MP's ideal setting; QIS should be in the same league, not necessarily better."""
        p = returns.shape[1]
        true = TRUE_RHO * np.ones((p, p)) + (1 - TRUE_RHO) * np.eye(p)
        qis = _corr_of(QuadraticInverseShrinkage(standardize=True).fit(returns).covariance_)
        mp = _corr_of(MarchenkoPasturClipping().fit(returns).covariance_)
        assert np.linalg.norm(qis - true) < 1.5 * np.linalg.norm(mp - true)

    def test_eigenvalue_attributes(self, returns: pd.DataFrame) -> None:
        est = QuadraticInverseShrinkage(standardize=True).fit(returns)
        p = returns.shape[1]
        assert est.eigenvalues_.shape == (p,)
        assert est.shrunk_eigenvalues_.shape == (p,)
        assert est.eigenvalues_.sum() == pytest.approx(p)
        assert est.shrunk_eigenvalues_.sum() == pytest.approx(p)


class TestValidationAndProtocol:
    def test_rejects_nan(self) -> None:
        df = pd.DataFrame({"a": [0.1, np.nan, 0.2], "b": [0.0, 0.1, 0.2]})
        with pytest.raises(ValueError, match="NaN"):
            QuadraticInverseShrinkage().fit(df)

    def test_rejects_zero_variance_column(self) -> None:
        df = pd.DataFrame({"a": [0.1, -0.2, 0.3, 0.0], "b": [0.5, 0.5, 0.5, 0.5]})
        with pytest.raises(ValueError, match="zero-variance"):
            QuadraticInverseShrinkage(standardize=True).fit(df)

    def test_satisfies_protocol_after_fit(self, returns: pd.DataFrame) -> None:
        assert isinstance(QuadraticInverseShrinkage().fit(returns), base.CovarianceEstimator)

    def test_fails_protocol_before_fit(self) -> None:
        assert not isinstance(QuadraticInverseShrinkage(), base.CovarianceEstimator)
