"""Tests for the shared estimator interface and helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.estimators import base


class TestValidateReturns:
    def test_accepts_valid_frame(self) -> None:
        df = pd.DataFrame({"a": [0.1, 0.2, 0.3], "b": [0.0, -0.1, 0.1]})
        # Should not raise.
        base.validate_returns(df)

    def test_rejects_non_dataframe(self) -> None:
        with pytest.raises(TypeError, match="pd.DataFrame"):
            base.validate_returns([[0.1, 0.2]])  # type: ignore[arg-type]

    def test_rejects_ndarray(self) -> None:
        with pytest.raises(TypeError, match="pd.DataFrame"):
            base.validate_returns(np.ones((3, 2)))  # type: ignore[arg-type]

    def test_rejects_empty_frame(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            base.validate_returns(pd.DataFrame())

    def test_rejects_single_column(self) -> None:
        with pytest.raises(ValueError, match="at least 2 columns"):
            base.validate_returns(pd.DataFrame({"a": [0.1, 0.2, 0.3]}))

    def test_rejects_frame_with_nan(self) -> None:
        df = pd.DataFrame({"a": [0.1, np.nan, 0.3], "b": [0.0, -0.1, 0.1]})
        with pytest.raises(ValueError, match="NaN"):
            base.validate_returns(df)

    def test_nan_in_any_column_rejected(self) -> None:
        df = pd.DataFrame({"a": [0.1, 0.2, 0.3], "b": [0.0, -0.1, np.nan]})
        with pytest.raises(ValueError, match="NaN"):
            base.validate_returns(df)


class TestWrapCovariance:
    def test_wraps_square_matrix(self) -> None:
        cov = np.array([[1.0, 0.5], [0.5, 2.0]])
        columns = pd.Index(["a", "b"])
        wrapped = base.wrap_covariance(cov, columns)
        assert isinstance(wrapped, pd.DataFrame)
        assert wrapped.shape == (2, 2)
        assert list(wrapped.index) == ["a", "b"]
        assert list(wrapped.columns) == ["a", "b"]
        np.testing.assert_array_equal(wrapped.to_numpy(), cov)

    def test_rejects_non_square(self) -> None:
        cov = np.array([[1.0, 0.5, 0.2], [0.5, 2.0, 0.3]])
        with pytest.raises(ValueError, match="square"):
            base.wrap_covariance(cov, pd.Index(["a", "b", "c"]))

    def test_rejects_mismatched_columns(self) -> None:
        cov = np.array([[1.0, 0.5], [0.5, 2.0]])
        with pytest.raises(ValueError, match="does not match"):
            base.wrap_covariance(cov, pd.Index(["a", "b", "c"]))

    def test_rejects_1d_array(self) -> None:
        cov = np.array([1.0, 0.5, 2.0])
        with pytest.raises(ValueError, match="square"):
            base.wrap_covariance(cov, pd.Index(["a", "b", "c"]))


class _MinimalEstimator:
    """Minimal class satisfying the CovarianceEstimator protocol for tests."""

    covariance_: pd.DataFrame

    def fit(self, returns: pd.DataFrame) -> _MinimalEstimator:
        self.covariance_ = returns.cov()
        return self


class _IncompleteEstimator:
    """Class missing the covariance_ attribute — should fail protocol check."""

    def fit(self, returns: pd.DataFrame) -> _IncompleteEstimator:
        return self


class TestCovarianceEstimatorProtocol:
    def test_minimal_estimator_satisfies_protocol(self) -> None:
        df = pd.DataFrame({"a": [0.1, 0.2, 0.3], "b": [0.0, -0.1, 0.1]})
        est = _MinimalEstimator().fit(df)
        # Runtime isinstance check (works because Protocol is @runtime_checkable)
        assert isinstance(est, base.CovarianceEstimator)

    def test_incomplete_estimator_fails_protocol_after_no_fit(self) -> None:
        est = _IncompleteEstimator()
        # Before fit, covariance_ doesn't exist, so protocol check fails
        assert not isinstance(est, base.CovarianceEstimator)
