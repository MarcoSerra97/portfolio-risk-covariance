"""Tests for the portfolio constructions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.portfolios.constructions import (
    CONSTRUCTIONS,
    equal_weight,
    min_variance_long_only,
    min_variance_long_short,
    risk_contributions,
    risk_parity,
    validate_covariance,
    with_excluded,
)


def _cov(p: int = 12, seed: int = 0) -> pd.DataFrame:
    """Heterogeneous-volatility covariance with one factor, some negative loadings."""
    rng = np.random.default_rng(seed)
    beta = rng.uniform(-0.3, 1.0, p)
    corr = np.outer(beta, beta) * 0.5
    np.fill_diagonal(corr, 1.0)
    corr = corr + 0.05 * np.eye(p)
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    vols = np.geomspace(2e-4, 3e-2, p)
    m = corr * np.outer(vols, vols)
    names = [f"a{i}" for i in range(p)]
    return pd.DataFrame(m, index=names, columns=names)


def _kkt(w: pd.Series, cov: pd.DataFrame) -> tuple[float, float]:
    """(max relative deviation of marginal variance among held assets, min for unheld)."""
    m = cov.to_numpy()
    x = w.to_numpy()
    g = m @ x
    lam = x @ g
    held = x > 0
    dev = float(np.abs(g[held] / lam - 1).max())
    slack = float((g[~held] / lam - 1).min()) if (~held).any() else np.inf
    return dev, slack


@pytest.fixture
def cov() -> pd.DataFrame:
    return _cov()


class TestCommonProperties:
    @pytest.mark.parametrize("name", list(CONSTRUCTIONS))
    def test_weights_sum_to_one_and_labelled(self, cov: pd.DataFrame, name: str) -> None:
        w = CONSTRUCTIONS[name].build(cov)
        assert list(w.index) == list(cov.index)
        assert w.sum() == pytest.approx(1.0, abs=1e-12)
        assert np.isfinite(w.to_numpy()).all()

    @pytest.mark.parametrize("name", list(CONSTRUCTIONS))
    def test_scale_invariance(self, cov: pd.DataFrame, name: str) -> None:
        build = CONSTRUCTIONS[name].build
        np.testing.assert_allclose(build(cov * 1e6).to_numpy(), build(cov).to_numpy(), atol=1e-9)

    @pytest.mark.parametrize("name", list(CONSTRUCTIONS))
    def test_permutation_equivariance(self, cov: pd.DataFrame, name: str) -> None:
        build = CONSTRUCTIONS[name].build
        order = list(reversed(cov.index))
        w = build(cov)
        w_perm = build(cov.loc[order, order])
        pd.testing.assert_series_equal(w_perm.reindex(cov.index), w, atol=1e-9)

    def test_registry_flags(self) -> None:
        assert CONSTRUCTIONS["equal_weight"].uses_covariance is False
        for name in ("risk_parity", "min_variance_long_only", "min_variance_long_short"):
            assert CONSTRUCTIONS[name].uses_covariance is True


class TestEqualWeight:
    def test_uniform(self, cov: pd.DataFrame) -> None:
        np.testing.assert_allclose(equal_weight(cov).to_numpy(), 1 / cov.shape[0])


class TestMinVarianceLongShort:
    def test_two_asset_closed_form(self) -> None:
        s1, s2, rho = 0.02, 0.01, 0.3
        c = rho * s1 * s2
        cov = pd.DataFrame([[s1**2, c], [c, s2**2]], index=["x", "y"], columns=["x", "y"])
        w1 = (s2**2 - c) / (s1**2 + s2**2 - 2 * c)
        np.testing.assert_allclose(min_variance_long_short(cov).to_numpy(), [w1, 1 - w1])

    def test_equal_marginal_variance(self, cov: pd.DataFrame) -> None:
        """First-order condition: Σw is proportional to the vector of ones."""
        w = min_variance_long_short(cov).to_numpy()
        g = cov.to_numpy() @ w
        np.testing.assert_allclose(g / g.mean(), 1.0, atol=1e-10)

    def test_beats_any_other_budget_portfolio(self, cov: pd.DataFrame) -> None:
        m = cov.to_numpy()
        w = min_variance_long_short(cov).to_numpy()
        rng = np.random.default_rng(1)
        for _ in range(200):
            x = rng.standard_normal(m.shape[0])
            x = x / x.sum()
            assert w @ m @ w <= x @ m @ x + 1e-18


class TestMinVarianceLongOnly:
    def test_equals_long_short_when_that_is_nonnegative(self) -> None:
        names = ["x", "y", "z"]
        cov = pd.DataFrame(np.diag([1e-4, 2e-4, 4e-4]), index=names, columns=names)
        np.testing.assert_allclose(
            min_variance_long_only(cov).to_numpy(), min_variance_long_short(cov).to_numpy()
        )

    def test_nonnegative_and_kkt(self, cov: pd.DataFrame) -> None:
        w = min_variance_long_only(cov)
        assert (w >= 0).all()
        assert (min_variance_long_short(cov) < 0).any(), "fixture should make the constraint bind"
        dev, slack = _kkt(w, cov)
        assert dev < 1e-9
        assert slack >= -1e-9

    def test_variance_ordering(self, cov: pd.DataFrame) -> None:
        """Long-short <= long-only <= equal weight, in variance."""
        m = cov.to_numpy()

        def var(w: pd.Series) -> float:
            x = w.to_numpy()
            return float(x @ m @ x)

        assert var(min_variance_long_short(cov)) <= var(min_variance_long_only(cov)) + 1e-18
        assert var(min_variance_long_only(cov)) <= var(equal_weight(cov)) + 1e-18


class TestRiskParity:
    def test_equal_risk_contributions(self, cov: pd.DataFrame) -> None:
        w = risk_parity(cov)
        assert (w > 0).all()
        rc = risk_contributions(w, cov)
        np.testing.assert_allclose(rc.to_numpy(), 1 / cov.shape[0], atol=1e-10)
        assert rc.sum() == pytest.approx(1.0)

    def test_diagonal_covariance_gives_inverse_volatility(self) -> None:
        vols = np.array([0.01, 0.02, 0.04, 0.005])
        names = list("abcd")
        cov = pd.DataFrame(np.diag(vols**2), index=names, columns=names)
        expected = (1 / vols) / (1 / vols).sum()
        np.testing.assert_allclose(risk_parity(cov).to_numpy(), expected, atol=1e-12)

    def test_two_assets_is_inverse_volatility_for_any_correlation(self) -> None:
        s1, s2 = 0.03, 0.01
        for rho in (-0.5, 0.0, 0.7):
            c = rho * s1 * s2
            cov = pd.DataFrame([[s1**2, c], [c, s2**2]], index=["x", "y"], columns=["x", "y"])
            np.testing.assert_allclose(risk_parity(cov).to_numpy(), [0.25, 0.75], atol=1e-12)


class TestWithExcluded:
    @pytest.mark.parametrize("name", list(CONSTRUCTIONS))
    def test_excluded_get_zero_and_rest_matches_submatrix(
        self, cov: pd.DataFrame, name: str
    ) -> None:
        build = CONSTRUCTIONS[name].build
        w = with_excluded(build, ["a0"])(cov)
        assert w["a0"] == 0.0
        assert w.sum() == pytest.approx(1.0)
        keep = list(cov.index[1:])
        pd.testing.assert_series_equal(w[keep], build(cov.loc[keep, keep]), atol=1e-12)

    def test_unknown_names_ignored(self, cov: pd.DataFrame) -> None:
        w = with_excluded(equal_weight, ["NOT_THERE"])(cov)
        np.testing.assert_allclose(w.to_numpy(), 1 / cov.shape[0])

    def test_excluding_everything_raises(self, cov: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="every asset"):
            with_excluded(equal_weight, list(cov.index))(cov)


class TestValidation:
    def test_rejects_non_dataframe(self) -> None:
        with pytest.raises(TypeError, match="DataFrame"):
            validate_covariance(np.eye(2))  # type: ignore[arg-type]

    def test_rejects_non_square(self) -> None:
        with pytest.raises(ValueError, match="square"):
            validate_covariance(pd.DataFrame(np.ones((2, 3))))

    def test_rejects_mismatched_labels(self) -> None:
        df = pd.DataFrame(np.eye(2), index=["a", "b"], columns=["b", "a"])
        with pytest.raises(ValueError, match="labels"):
            validate_covariance(df)

    def test_rejects_asymmetric(self) -> None:
        df = pd.DataFrame([[1.0, 0.5], [0.1, 1.0]], index=["a", "b"], columns=["a", "b"])
        with pytest.raises(ValueError, match="symmetric"):
            validate_covariance(df)

    def test_rejects_non_finite(self) -> None:
        df = pd.DataFrame([[1.0, np.nan], [np.nan, 1.0]], index=["a", "b"], columns=["a", "b"])
        with pytest.raises(ValueError, match="non-finite"):
            validate_covariance(df)

    def test_rejects_not_positive_definite(self) -> None:
        df = pd.DataFrame([[1.0, 2.0], [2.0, 1.0]], index=["a", "b"], columns=["a", "b"])
        with pytest.raises(ValueError, match="positive definite"):
            validate_covariance(df)
