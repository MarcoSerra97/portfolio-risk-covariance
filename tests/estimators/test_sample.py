"""Tests for the SampleCovariance estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.covariance import EmpiricalCovariance

from portfolio_risk.estimators import base
from portfolio_risk.estimators.sample import SampleCovariance


@pytest.fixture
def small_returns() -> pd.DataFrame:
    """Deterministic 4-row, 3-column return panel."""
    return pd.DataFrame(
        {
            "a": [0.01, -0.02, 0.03, -0.01],
            "b": [0.00, 0.02, -0.01, 0.01],
            "c": [-0.01, 0.01, 0.02, 0.00],
        }
    )


@pytest.fixture
def random_returns() -> pd.DataFrame:
    """Larger synthetic panel for numerical comparisons."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.standard_normal((200, 5)), columns=list("abcde"))


class TestFit:
    def test_fit_returns_self(self, small_returns: pd.DataFrame) -> None:
        est = SampleCovariance()
        assert est.fit(small_returns) is est

    def test_covariance_populated_after_fit(self, small_returns: pd.DataFrame) -> None:
        est = SampleCovariance()
        assert not hasattr(est, "covariance_")
        est.fit(small_returns)
        assert hasattr(est, "covariance_")

    def test_covariance_shape(self, small_returns: pd.DataFrame) -> None:
        assert SampleCovariance().fit(small_returns).covariance_.shape == (3, 3)

    def test_covariance_labels_preserved(self, small_returns: pd.DataFrame) -> None:
        cov = SampleCovariance().fit(small_returns).covariance_
        assert list(cov.index) == ["a", "b", "c"]
        assert list(cov.columns) == ["a", "b", "c"]

    def test_covariance_is_symmetric(self, random_returns: pd.DataFrame) -> None:
        cov = SampleCovariance().fit(random_returns).covariance_.to_numpy()
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)

    def test_covariance_is_positive_semidefinite(self, random_returns: pd.DataFrame) -> None:
        cov = SampleCovariance().fit(random_returns).covariance_.to_numpy()
        assert np.linalg.eigvalsh(cov).min() >= -1e-12

    def test_diagonal_matches_ml_variances(self, random_returns: pd.DataFrame) -> None:
        cov = SampleCovariance().fit(random_returns).covariance_
        np.testing.assert_allclose(
            np.diag(cov.to_numpy()), random_returns.var(ddof=0).to_numpy(), atol=1e-12
        )


class TestConvention:
    """The estimator uses the ML divisor n, matching sklearn."""

    def test_matches_sklearn(self, random_returns: pd.DataFrame) -> None:
        expected = EmpiricalCovariance().fit(random_returns.to_numpy()).covariance_
        cov = SampleCovariance().fit(random_returns).covariance_
        np.testing.assert_allclose(cov.to_numpy(), expected, atol=1e-12)

    def test_matches_numpy_ddof0(self, random_returns: pd.DataFrame) -> None:
        expected = np.cov(random_returns.to_numpy(), rowvar=False, ddof=0)
        cov = SampleCovariance().fit(random_returns).covariance_
        np.testing.assert_allclose(cov.to_numpy(), expected, atol=1e-12)

    def test_matches_pandas_ddof0(self, random_returns: pd.DataFrame) -> None:
        cov = SampleCovariance().fit(random_returns).covariance_
        pd.testing.assert_frame_equal(cov, random_returns.cov(ddof=0))

    def test_rescales_to_unbiased(self, random_returns: pd.DataFrame) -> None:
        """Multiplying by n/(n-1) recovers the Bessel-corrected estimator."""
        n = len(random_returns)
        cov = SampleCovariance().fit(random_returns).covariance_
        np.testing.assert_allclose(
            cov.to_numpy() * n / (n - 1), random_returns.cov(ddof=1).to_numpy(), atol=1e-12
        )


class TestAssumeCentered:
    def test_default_is_false(self) -> None:
        assert SampleCovariance().assume_centered is False

    def test_centered_matches_raw_second_moment(self, random_returns: pd.DataFrame) -> None:
        """With assume_centered=True the estimator returns X'X / n (no de-meaning)."""
        x = random_returns.to_numpy()
        cov = SampleCovariance(assume_centered=True).fit(random_returns).covariance_
        np.testing.assert_allclose(cov.to_numpy(), x.T @ x / len(x), atol=1e-12)

    def test_centered_equals_default_on_demeaned_data(self, random_returns: pd.DataFrame) -> None:
        demeaned = random_returns - random_returns.mean()
        default = SampleCovariance().fit(demeaned).covariance_
        centered = SampleCovariance(assume_centered=True).fit(demeaned).covariance_
        np.testing.assert_allclose(default.to_numpy(), centered.to_numpy(), atol=1e-12)


class TestValidation:
    def test_rejects_nan_returns(self) -> None:
        df = pd.DataFrame({"a": [0.1, np.nan], "b": [0.0, 0.1]})
        with pytest.raises(ValueError, match="NaN"):
            SampleCovariance().fit(df)

    def test_rejects_empty_returns(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            SampleCovariance().fit(pd.DataFrame())

    def test_rejects_single_column(self) -> None:
        with pytest.raises(ValueError, match="at least 2 columns"):
            SampleCovariance().fit(pd.DataFrame({"a": [0.1, 0.2]}))

    def test_rejects_non_dataframe(self) -> None:
        with pytest.raises(TypeError, match="pd.DataFrame"):
            SampleCovariance().fit(np.ones((3, 2)))  # type: ignore[arg-type]


class TestProtocolConformance:
    def test_satisfies_protocol_after_fit(self, small_returns: pd.DataFrame) -> None:
        assert isinstance(SampleCovariance().fit(small_returns), base.CovarianceEstimator)

    def test_fails_protocol_before_fit(self) -> None:
        assert not isinstance(SampleCovariance(), base.CovarianceEstimator)
