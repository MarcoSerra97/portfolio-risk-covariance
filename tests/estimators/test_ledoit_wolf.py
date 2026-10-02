"""Tests for the LedoitWolfShrinkage estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.covariance import LedoitWolf

from portfolio_risk.estimators import base
from portfolio_risk.estimators.ledoit_wolf import LedoitWolfShrinkage
from portfolio_risk.estimators.sample import SampleCovariance


def _panel(n: int, p: int, seed: int = 0) -> pd.DataFrame:
    """Correlated Gaussian panel with one common factor."""
    rng = np.random.default_rng(seed)
    factor = rng.standard_normal((n, 1))
    data = 0.6 * factor + rng.standard_normal((n, p))
    return pd.DataFrame(data, columns=[f"x{i}" for i in range(p)])


@pytest.fixture
def returns() -> pd.DataFrame:
    return _panel(n=120, p=10)


@pytest.fixture
def heterogeneous_returns() -> pd.DataFrame:
    """Panel whose column volatilities span four orders of magnitude."""
    base_panel = _panel(n=300, p=6, seed=1)
    scales = np.array([1e-4, 1e-3, 1e-2, 1e-2, 1e-1, 1.0])
    return base_panel * scales


class TestFit:
    def test_fit_returns_self(self, returns: pd.DataFrame) -> None:
        est = LedoitWolfShrinkage()
        assert est.fit(returns) is est

    def test_attributes_populated(self, returns: pd.DataFrame) -> None:
        est = LedoitWolfShrinkage()
        assert not hasattr(est, "covariance_")
        assert not hasattr(est, "shrinkage_")
        est.fit(returns)
        assert est.covariance_.shape == (10, 10)
        assert isinstance(est.shrinkage_, float)

    def test_labels_preserved(self, returns: pd.DataFrame) -> None:
        cov = LedoitWolfShrinkage().fit(returns).covariance_
        assert list(cov.index) == list(returns.columns)
        assert list(cov.columns) == list(returns.columns)

    @pytest.mark.parametrize("standardize", [False, True])
    def test_symmetric_positive_definite(self, returns: pd.DataFrame, standardize: bool) -> None:
        cov = LedoitWolfShrinkage(standardize=standardize).fit(returns).covariance_.to_numpy()
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)
        assert np.linalg.eigvalsh(cov).min() > 0

    @pytest.mark.parametrize("standardize", [False, True])
    def test_shrinkage_in_unit_interval(self, returns: pd.DataFrame, standardize: bool) -> None:
        delta = LedoitWolfShrinkage(standardize=standardize).fit(returns).shrinkage_
        assert 0.0 <= delta <= 1.0


class TestScaledIdentityTarget:
    """standardize=False: textbook LW 2004, identical to sklearn."""

    def test_matches_sklearn(self, returns: pd.DataFrame) -> None:
        sk = LedoitWolf().fit(returns.to_numpy())
        est = LedoitWolfShrinkage().fit(returns)
        np.testing.assert_allclose(est.covariance_.to_numpy(), sk.covariance_, atol=1e-12)
        assert est.shrinkage_ == pytest.approx(sk.shrinkage_)

    def test_is_convex_combination_of_sample_and_target(self, returns: pd.DataFrame) -> None:
        """Sigma = (1 - delta) S + delta mu I, with S from our SampleCovariance."""
        est = LedoitWolfShrinkage().fit(returns)
        s = SampleCovariance().fit(returns).covariance_.to_numpy()
        mu = np.trace(s) / s.shape[0]
        expected = (1 - est.shrinkage_) * s + est.shrinkage_ * mu * np.eye(s.shape[0])
        np.testing.assert_allclose(est.covariance_.to_numpy(), expected, atol=1e-12)

    def test_preserves_trace(self, returns: pd.DataFrame) -> None:
        est = LedoitWolfShrinkage().fit(returns)
        s = SampleCovariance().fit(returns).covariance_.to_numpy()
        assert np.trace(est.covariance_.to_numpy()) == pytest.approx(np.trace(s))

    def test_assume_centered_passed_through(self, returns: pd.DataFrame) -> None:
        sk = LedoitWolf(assume_centered=True).fit(returns.to_numpy())
        est = LedoitWolfShrinkage(assume_centered=True).fit(returns)
        np.testing.assert_allclose(est.covariance_.to_numpy(), sk.covariance_, atol=1e-12)

    def test_inflates_low_volatility_variances(self, heterogeneous_returns: pd.DataFrame) -> None:
        """Documents why standardize=True exists: the common target distorts small variances."""
        est = LedoitWolfShrinkage().fit(heterogeneous_returns)
        s = SampleCovariance().fit(heterogeneous_returns).covariance_.to_numpy()
        ratio = est.covariance_.to_numpy()[0, 0] / s[0, 0]
        assert ratio > 10


class TestCorrelationTarget:
    """standardize=True: shrink correlations, preserve variances."""

    def test_preserves_sample_variances(self, heterogeneous_returns: pd.DataFrame) -> None:
        est = LedoitWolfShrinkage(standardize=True).fit(heterogeneous_returns)
        s = SampleCovariance().fit(heterogeneous_returns).covariance_.to_numpy()
        np.testing.assert_allclose(np.diag(est.covariance_.to_numpy()), np.diag(s), rtol=1e-10)

    def test_correlations_shrunk_by_one_minus_delta(self, returns: pd.DataFrame) -> None:
        """R_shrunk = (1 - delta) R_sample off the diagonal."""
        est = LedoitWolfShrinkage(standardize=True).fit(returns)
        cov = est.covariance_.to_numpy()
        d = np.sqrt(np.diag(cov))
        r_lw = cov / np.outer(d, d)
        r_sample = returns.corr().to_numpy()
        off = ~np.eye(len(d), dtype=bool)
        np.testing.assert_allclose(r_lw[off], (1 - est.shrinkage_) * r_sample[off], atol=1e-10)

    def test_invariant_to_column_scaling(self, returns: pd.DataFrame) -> None:
        """Rescaling a column rescales its row/column but leaves delta unchanged."""
        scaled = returns.copy()
        scaled["x0"] *= 1000.0
        a = LedoitWolfShrinkage(standardize=True).fit(returns)
        b = LedoitWolfShrinkage(standardize=True).fit(scaled)
        assert a.shrinkage_ == pytest.approx(b.shrinkage_)
        assert b.covariance_.iloc[0, 1] == pytest.approx(1000.0 * a.covariance_.iloc[0, 1])

    def test_rejects_zero_variance_column(self) -> None:
        df = pd.DataFrame({"a": [0.1, -0.2, 0.3, 0.0], "b": [0.5, 0.5, 0.5, 0.5]})
        with pytest.raises(ValueError, match="zero-variance"):
            LedoitWolfShrinkage(standardize=True).fit(df)


class TestShrinkageBehaviour:
    def test_more_shrinkage_when_data_scarce(self) -> None:
        """Fewer observations per variable -> noisier S -> larger delta."""
        scarce = LedoitWolfShrinkage().fit(_panel(n=30, p=20, seed=3)).shrinkage_
        plentiful = LedoitWolfShrinkage().fit(_panel(n=3000, p=20, seed=3)).shrinkage_
        assert scarce > plentiful

    def test_better_conditioned_than_sample(self) -> None:
        df = _panel(n=40, p=30, seed=4)
        lw = np.linalg.cond(LedoitWolfShrinkage().fit(df).covariance_.to_numpy())
        sample = np.linalg.cond(SampleCovariance().fit(df).covariance_.to_numpy())
        assert lw < sample


class TestValidationAndProtocol:
    def test_rejects_nan(self) -> None:
        df = pd.DataFrame({"a": [0.1, np.nan, 0.2], "b": [0.0, 0.1, 0.2]})
        with pytest.raises(ValueError, match="NaN"):
            LedoitWolfShrinkage().fit(df)

    def test_rejects_non_dataframe(self) -> None:
        with pytest.raises(TypeError, match="pd.DataFrame"):
            LedoitWolfShrinkage().fit(np.ones((5, 2)))  # type: ignore[arg-type]

    def test_satisfies_protocol_after_fit(self, returns: pd.DataFrame) -> None:
        assert isinstance(LedoitWolfShrinkage().fit(returns), base.CovarianceEstimator)

    def test_fails_protocol_before_fit(self) -> None:
        assert not isinstance(LedoitWolfShrinkage(), base.CovarianceEstimator)
