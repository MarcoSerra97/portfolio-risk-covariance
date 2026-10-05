"""Tests for the cDCC-GARCH estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.estimators import base
from portfolio_risk.estimators.dcc_garch import DCCGarch

TRUE_A, TRUE_B = 0.04, 0.93
TRUE_ALPHA, TRUE_BETA, TRUE_OMEGA = 0.08, 0.90, 0.05  # GARCH in percent units


def _true_corr(p: int) -> np.ndarray:
    corr = 0.4 * np.ones((p, p)) + 0.6 * np.eye(p)
    corr[0, 1] = corr[1, 0] = 0.8
    corr[2, 4] = corr[4, 2] = -0.2
    return corr


def simulate_cdcc(
    n: int, corr: np.ndarray, a: float, b: float, seed: int = 0, scale: np.ndarray | None = None
) -> pd.DataFrame:
    """Simulate cDCC-GARCH(1,1) returns (decimal units) with known parameters."""
    rng = np.random.default_rng(seed)
    p = corr.shape[0]
    omega = np.full(p, TRUE_OMEGA)
    sig2 = omega / (1 - TRUE_ALPHA - TRUE_BETA)
    q = corr.copy()
    z_prev = np.zeros(p)
    y_prev = np.zeros(p)
    out = np.empty((n, p))
    for t in range(n):
        if t > 0:
            sig2 = omega + TRUE_ALPHA * y_prev**2 + TRUE_BETA * sig2
            zs = np.sqrt(np.diag(q)) * z_prev
            q = (1 - a - b) * corr + a * np.outer(zs, zs) + b * q
        d = np.sqrt(np.diag(q))
        z = np.linalg.cholesky(q / np.outer(d, d)) @ rng.standard_normal(p)
        y = np.sqrt(sig2) * z
        out[t] = y
        z_prev, y_prev = z, y
    data = out / 100.0
    if scale is not None:
        data = data * scale
    return pd.DataFrame(data, columns=[f"x{i}" for i in range(p)])


def _corr_of(cov: pd.DataFrame) -> np.ndarray:
    c = cov.to_numpy()
    d = np.sqrt(np.diag(c))
    return c / np.outer(d, d)


@pytest.fixture(scope="module")
def returns() -> pd.DataFrame:
    return simulate_cdcc(3000, _true_corr(5), TRUE_A, TRUE_B, seed=0)


@pytest.fixture(scope="module")
def fitted(returns: pd.DataFrame) -> DCCGarch:
    return DCCGarch().fit(returns)


class TestParameterRecovery:
    def test_dcc_parameters(self, fitted: DCCGarch) -> None:
        assert fitted.dcc_converged_
        assert fitted.dcc_a_ == pytest.approx(TRUE_A, abs=0.015)
        assert fitted.dcc_b_ == pytest.approx(TRUE_B, abs=0.03)
        assert fitted.dcc_a_ + fitted.dcc_b_ < 1

    def test_garch_parameters(self, fitted: DCCGarch) -> None:
        g = fitted.garch_params_
        assert g["alpha"].mean() == pytest.approx(TRUE_ALPHA, abs=0.02)
        assert g["beta"].mean() == pytest.approx(TRUE_BETA, abs=0.03)
        assert not g["fallback"].any()
        assert fitted.fallbacks_ == []

    def test_correlation_target(self, fitted: DCCGarch) -> None:
        target = fitted.correlation_target_.to_numpy()
        np.testing.assert_allclose(np.diag(target), 1.0, atol=1e-12)
        assert np.abs(target - _true_corr(5)).max() < 0.1

    def test_constant_correlation_data_gives_small_a(self) -> None:
        df = simulate_cdcc(3000, _true_corr(5), a=0.0, b=0.0, seed=3)
        assert DCCGarch().fit(df).dcc_a_ < 0.02


class TestMatrixProperties:
    @pytest.mark.parametrize("attr", ["covariance_", "one_step_covariance_"])
    def test_labels_symmetric_positive_definite(self, fitted: DCCGarch, attr: str) -> None:
        cov: pd.DataFrame = getattr(fitted, attr)
        assert list(cov.index) == [f"x{i}" for i in range(5)]
        assert list(cov.columns) == list(cov.index)
        m = cov.to_numpy()
        np.testing.assert_allclose(m, m.T, atol=0)
        assert np.linalg.eigvalsh(m).min() > 0

    def test_implied_correlations_valid(self, fitted: DCCGarch) -> None:
        r = _corr_of(fitted.covariance_)
        np.testing.assert_allclose(np.diag(r), 1.0, atol=1e-12)
        assert np.abs(r[~np.eye(5, dtype=bool)]).max() < 1

    def test_garch_params_frame(self, fitted: DCCGarch) -> None:
        g = fitted.garch_params_
        assert list(g.columns) == ["omega", "alpha", "beta", "persistence", "fallback"]
        np.testing.assert_allclose(g["persistence"], g["alpha"] + g["beta"])
        assert (g["omega"] > 0).all()


class TestForecastHorizon:
    def test_horizon_one_equals_one_step(self, returns: pd.DataFrame) -> None:
        est = DCCGarch(horizon=1).fit(returns)
        np.testing.assert_allclose(
            est.covariance_.to_numpy(), est.one_step_covariance_.to_numpy(), rtol=1e-12
        )

    def test_average_variance_between_one_step_and_long_run(self, fitted: DCCGarch) -> None:
        g = fitted.garch_params_
        long_run = (g["omega"] / (1 - g["persistence"])).to_numpy()
        one_step = np.diag(fitted.one_step_covariance_.to_numpy())
        avg = np.diag(fitted.covariance_.to_numpy())
        lo, hi = np.minimum(one_step, long_run), np.maximum(one_step, long_run)
        assert np.all(avg >= lo * (1 - 1e-12))
        assert np.all(avg <= hi * (1 + 1e-12))

    def test_shock_decays_over_horizon(self, returns: pd.DataFrame) -> None:
        """After a large final-day move, risk is forecast high now and decaying."""
        shocked = returns.copy()
        shocked.iloc[-1, 0] = 12 * returns["x0"].std()
        est = DCCGarch().fit(shocked)
        g = est.garch_params_.loc["x0"]
        long_run = g["omega"] / (1 - g["persistence"])
        one_step = est.one_step_covariance_.loc["x0", "x0"]
        avg = est.covariance_.loc["x0", "x0"]
        assert one_step > avg > long_run


class TestCorrelationForecast:
    """The cDCC correlation forecast must match a Monte Carlo of the model itself."""

    @staticmethod
    def _monte_carlo(est: DCCGarch, paths: int = 10_000, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        target = est.correlation_target_.to_numpy()
        a, b, h_max = est.dcc_a_, est.dcc_b_, est.horizon
        p = target.shape[0]
        q = np.broadcast_to(est.one_step_q_, (paths, p, p)).copy()
        mean_r = np.empty((h_max, p, p))
        for h in range(h_max):
            d = np.sqrt(np.einsum("kii->ki", q))
            r = q / (d[:, :, None] * d[:, None, :])
            mean_r[h] = r.mean(axis=0)
            z = np.einsum("kij,kj->ki", np.linalg.cholesky(r), rng.standard_normal((paths, p)))
            zs = d * z
            q = (1 - a - b) * target + a * zs[:, :, None] * zs[:, None, :] + b * q
        return mean_r

    def test_path_shape_and_first_step(self, fitted: DCCGarch) -> None:
        assert fitted.correlation_path_.shape == (fitted.horizon, 5, 5)
        np.testing.assert_allclose(
            fitted.correlation_path_[0], _corr_of(fitted.one_step_covariance_), atol=1e-12
        )

    def test_path_reverts_to_target(self, returns: pd.DataFrame) -> None:
        est = DCCGarch(horizon=2000).fit(returns)
        np.testing.assert_allclose(
            est.correlation_path_[-1], est.correlation_target_.to_numpy(), atol=1e-6
        )

    def test_matches_monte_carlo_after_extreme_joint_move(self, returns: pd.DataFrame) -> None:
        shocked = returns.copy()
        shocked.iloc[-1, :2] = [0.06, -0.06]  # two highly correlated assets move apart
        est = DCCGarch().fit(shocked)
        mc = self._monte_carlo(est)
        assert np.abs(est.correlation_path_ - mc).max() < 0.03


class TestScaleAndOptions:
    def test_scale_equivariance(self, returns: pd.DataFrame, fitted: DCCGarch) -> None:
        scaled = DCCGarch().fit(returns * 10.0)
        np.testing.assert_allclose(
            scaled.covariance_.to_numpy(), 100.0 * fitted.covariance_.to_numpy(), rtol=1e-2
        )

    def test_all_pairs_close_to_contiguous(self, returns: pd.DataFrame, fitted: DCCGarch) -> None:
        est = DCCGarch(pairs="all").fit(returns)
        assert est.dcc_a_ == pytest.approx(fitted.dcc_a_, abs=0.01)
        assert est.dcc_b_ == pytest.approx(fitted.dcc_b_, abs=0.02)


class TestFallback:
    def test_degenerate_fits_fall_back_to_constant_variance(self, returns: pd.DataFrame) -> None:
        """max_persistence=0 forces every asset onto the fallback path."""
        est = DCCGarch(max_persistence=0.0).fit(returns)
        assert est.fallbacks_ == list(returns.columns)
        assert (est.garch_params_["alpha"] == 0).all()
        assert (est.garch_params_["beta"] == 0).all()
        np.testing.assert_allclose(
            np.diag(est.covariance_.to_numpy()), (returns**2).mean().to_numpy(), rtol=1e-10
        )

    def test_fallback_still_positive_definite(self, returns: pd.DataFrame) -> None:
        est = DCCGarch(max_persistence=0.0).fit(returns)
        assert np.linalg.eigvalsh(est.covariance_.to_numpy()).min() > 0


class TestValidationAndProtocol:
    def test_rejects_bad_horizon(self) -> None:
        with pytest.raises(ValueError, match="horizon"):
            DCCGarch(horizon=0)

    def test_rejects_bad_pairs(self) -> None:
        with pytest.raises(ValueError, match="pairs"):
            DCCGarch(pairs="some")  # type: ignore[arg-type]

    def test_rejects_nan(self) -> None:
        df = pd.DataFrame({"a": [0.01, np.nan, 0.02], "b": [0.0, 0.01, 0.02]})
        with pytest.raises(ValueError, match="NaN"):
            DCCGarch().fit(df)

    def test_satisfies_protocol_after_fit(self, fitted: DCCGarch) -> None:
        assert isinstance(fitted, base.CovarianceEstimator)

    def test_fails_protocol_before_fit(self) -> None:
        assert not isinstance(DCCGarch(), base.CovarianceEstimator)
