"""Run configuration for the backtest, with the study's decisions as defaults.

Implements the choices recorded in ``docs/02-evaluation-design.md``: the seven
estimators and how each is windowed, the four portfolios, the exclusion of the
cash instrument (BIL) from the optimised portfolios, the calendar parameters,
and the bootstrap settings.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Literal

from portfolio_risk.estimators.base import CovarianceEstimator
from portfolio_risk.estimators.dcc_garch import DCCGarch
from portfolio_risk.estimators.ledoit_wolf import LedoitWolfShrinkage
from portfolio_risk.estimators.marchenko_pastur import MarchenkoPasturClipping
from portfolio_risk.estimators.qis import QuadraticInverseShrinkage
from portfolio_risk.estimators.sample import SampleCovariance
from portfolio_risk.evaluation.schedule import DEFAULT_FIRST_REBALANCE, DEFAULT_HOLDING
from portfolio_risk.portfolios.constructions import CONSTRUCTIONS, WeightFn, with_excluded

WindowKind = Literal["rolling", "expanding"]

#: First date on which all 75 instruments trade (MTUM's first observation).
DCC_COMMON_START: date = date(2013, 4, 18)


@dataclass(frozen=True)
class EstimatorSpec:
    """How to build an estimator and which data window it is fitted on.

    Attributes:
        name: Identifier used in results tables.
        factory: Builds a fresh estimator given the holding period (used as
            the forecast horizon by time-varying estimators).
        window: ``"rolling"`` (last ``window_length`` days) or ``"expanding"``
            (from ``RunConfig.dcc_start`` to the rebalance date).
    """

    name: str
    factory: Callable[[int], CovarianceEstimator]
    window: WindowKind


ESTIMATORS: dict[str, EstimatorSpec] = {
    s.name: s
    for s in (
        EstimatorSpec("sample", lambda h: SampleCovariance(), "rolling"),
        EstimatorSpec("lw_raw", lambda h: LedoitWolfShrinkage(standardize=False), "rolling"),
        EstimatorSpec("lw_std", lambda h: LedoitWolfShrinkage(standardize=True), "rolling"),
        EstimatorSpec("mp", lambda h: MarchenkoPasturClipping(), "rolling"),
        EstimatorSpec("qis_raw", lambda h: QuadraticInverseShrinkage(standardize=False), "rolling"),
        EstimatorSpec("qis_std", lambda h: QuadraticInverseShrinkage(standardize=True), "rolling"),
        EstimatorSpec("dcc", lambda h: DCCGarch(horizon=h), "expanding"),
    )
}


@dataclass(frozen=True)
class RunConfig:
    """Settings for one backtest run (one estimation-window length).

    Attributes:
        window_length: Rolling estimation window in trading days (252 or 504).
        holding: Holding period and forecast horizon in trading days.
        first_rebalance: The first rebalance is the last trading day on or before this.
        dcc_start: Start of the expanding window for expanding-window estimators.
        estimators: Names from :data:`ESTIMATORS` to run.
        portfolios: Names from :data:`~portfolio_risk.portfolios.constructions.CONSTRUCTIONS`.
        exclude_from_optimised: Instruments excluded from covariance-based portfolios.
        bootstrap_reps: Bootstrap replications.
        block_length_periods: Mean block length for the Criterion A bootstrap.
        block_length_days: Mean block length for the Criterion B bootstrap.
        seed: Seed for the bootstrap and the stress-test subsets.
    """

    window_length: int = 252
    holding: int = DEFAULT_HOLDING
    first_rebalance: date = DEFAULT_FIRST_REBALANCE
    dcc_start: date = DCC_COMMON_START
    estimators: tuple[str, ...] = tuple(ESTIMATORS)
    portfolios: tuple[str, ...] = tuple(CONSTRUCTIONS)
    exclude_from_optimised: tuple[str, ...] = ("BIL",)
    bootstrap_reps: int = 2000
    block_length_periods: float = 3.0
    block_length_days: float = 21.0
    seed: int = 20261006

    def __post_init__(self) -> None:
        if self.window_length < 2:
            raise ValueError(f"window_length must be >= 2, got {self.window_length}")
        if self.holding < 1:
            raise ValueError(f"holding must be >= 1, got {self.holding}")
        unknown_e = set(self.estimators) - set(ESTIMATORS)
        if unknown_e:
            raise ValueError(f"unknown estimators: {sorted(unknown_e)}")
        unknown_p = set(self.portfolios) - set(CONSTRUCTIONS)
        if unknown_p:
            raise ValueError(f"unknown portfolios: {sorted(unknown_p)}")
        if self.bootstrap_reps < 1:
            raise ValueError(f"bootstrap_reps must be >= 1, got {self.bootstrap_reps}")


def make_estimator(name: str, config: RunConfig) -> CovarianceEstimator:
    """A fresh, unfitted estimator for ``name`` under ``config``."""
    return ESTIMATORS[name].factory(config.holding)


def portfolio_builders(config: RunConfig) -> dict[str, WeightFn]:
    """Weight functions for the configured portfolios.

    Covariance-based constructions exclude ``config.exclude_from_optimised``
    (with zero weight); estimator-independent ones (equal weight) keep every
    instrument.
    """
    builders: dict[str, WeightFn] = {}
    for name in config.portfolios:
        c = CONSTRUCTIONS[name]
        builders[name] = (
            with_excluded(c.build, config.exclude_from_optimised)
            if c.uses_covariance and config.exclude_from_optimised
            else c.build
        )
    return builders


__all__ = [
    "DCC_COMMON_START",
    "ESTIMATORS",
    "EstimatorSpec",
    "RunConfig",
    "WindowKind",
    "make_estimator",
    "portfolio_builders",
]
