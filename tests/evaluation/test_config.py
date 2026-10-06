"""Tests for the run configuration and registries."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.estimators.base import CovarianceEstimator
from portfolio_risk.estimators.dcc_garch import DCCGarch
from portfolio_risk.evaluation.config import (
    DCC_COMMON_START,
    ESTIMATORS,
    RunConfig,
    make_estimator,
    portfolio_builders,
)


def _cov_with_cash() -> pd.DataFrame:
    """Small covariance with a near-riskless instrument named BIL."""
    names = ["A", "B", "C", "BIL"]
    vols = np.array([0.02, 0.015, 0.01, 0.0002])
    corr = np.full((4, 4), 0.3)
    np.fill_diagonal(corr, 1.0)
    corr[3, :3] = corr[:3, 3] = 0.0
    return pd.DataFrame(corr * np.outer(vols, vols), index=names, columns=names)


class TestRegistry:
    def test_seven_estimators(self) -> None:
        assert list(ESTIMATORS) == [
            "sample", "lw_raw", "lw_std", "mp", "qis_raw", "qis_std", "dcc",
        ]  # fmt: skip

    def test_windows(self) -> None:
        assert ESTIMATORS["dcc"].window == "expanding"
        assert all(s.window == "rolling" for n, s in ESTIMATORS.items() if n != "dcc")

    @pytest.mark.parametrize("name", list(ESTIMATORS))
    def test_factories_build_unfitted_estimators(self, name: str) -> None:
        est = make_estimator(name, RunConfig())
        assert hasattr(est, "fit")
        assert not isinstance(est, CovarianceEstimator)  # covariance_ not set until fit

    def test_dcc_horizon_follows_holding(self) -> None:
        est = make_estimator("dcc", RunConfig(holding=10))
        assert isinstance(est, DCCGarch)
        assert est.horizon == 10

    def test_lw_and_qis_variants_differ(self) -> None:
        cfg = RunConfig()
        assert make_estimator("lw_raw", cfg).standardize is False  # type: ignore[attr-defined]
        assert make_estimator("lw_std", cfg).standardize is True  # type: ignore[attr-defined]
        assert make_estimator("qis_raw", cfg).standardize is False  # type: ignore[attr-defined]
        assert make_estimator("qis_std", cfg).standardize is True  # type: ignore[attr-defined]


class TestRunConfigDefaults:
    def test_design_doc_defaults(self) -> None:
        cfg = RunConfig()
        assert cfg.window_length == 252
        assert cfg.holding == 21
        assert cfg.first_rebalance == date(2014, 12, 31)
        assert cfg.dcc_start == DCC_COMMON_START == date(2013, 4, 18)
        assert cfg.exclude_from_optimised == ("BIL",)
        assert cfg.bootstrap_reps == 2000
        assert (cfg.block_length_periods, cfg.block_length_days) == (3.0, 21.0)
        assert set(cfg.portfolios) == {
            "equal_weight", "risk_parity", "min_variance_long_only", "min_variance_long_short",
        }  # fmt: skip

    def test_frozen(self) -> None:
        cfg = RunConfig()
        with pytest.raises(AttributeError):
            cfg.window_length = 504  # type: ignore[misc]

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"window_length": 1}, "window_length"),
            ({"holding": 0}, "holding"),
            ({"estimators": ("sample", "nope")}, "unknown estimators"),
            ({"portfolios": ("nope",)}, "unknown portfolios"),
            ({"bootstrap_reps": 0}, "bootstrap_reps"),
        ],
    )
    def test_validation(self, kwargs: dict[str, object], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            RunConfig(**kwargs)  # type: ignore[arg-type]


class TestPortfolioBuilders:
    def test_cash_excluded_only_from_optimised(self) -> None:
        cov = _cov_with_cash()
        builders = portfolio_builders(RunConfig())
        assert builders["equal_weight"](cov)["BIL"] == pytest.approx(0.25)
        for name in ("risk_parity", "min_variance_long_only", "min_variance_long_short"):
            w = builders[name](cov)
            assert w["BIL"] == 0.0
            assert w.sum() == pytest.approx(1.0)

    def test_no_exclusion_puts_weight_on_cash(self) -> None:
        cov = _cov_with_cash()
        w = portfolio_builders(RunConfig(exclude_from_optimised=()))["min_variance_long_short"](cov)
        assert w["BIL"] > 0.9

    def test_subset_of_portfolios(self) -> None:
        builders = portfolio_builders(RunConfig(portfolios=("equal_weight",)))
        assert list(builders) == ["equal_weight"]
