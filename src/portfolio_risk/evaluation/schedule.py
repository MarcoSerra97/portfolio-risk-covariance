"""Backtest calendar: rebalance dates, holding periods, estimation windows, regimes.

Implements ``docs/02-evaluation-design.md`` §3, §4 and §9 on the trading-day
index of the returns panel itself, so the schedule always matches the data the
estimators see.

* The first rebalance is the last trading day on or before ``first_rebalance``
  (2014-12-31 by default). Rebalances then follow every ``holding`` trading days.
* Holding period ``k`` covers the ``holding`` trading days after rebalance ``k``.
  Periods are contiguous and do not overlap; a final partial period is dropped.
* At rebalance ``k`` an estimator may use returns up to and including the
  rebalance date: the estimation window ends exactly one day before the
  holding period starts (no look-ahead).
* Each period is assigned to the regime containing the majority of its days.

Positions are integer offsets into the index; ``end`` positions are inclusive.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd

DEFAULT_FIRST_REBALANCE: date = date(2014, 12, 31)
DEFAULT_HOLDING: int = 21
UNCLASSIFIED: str = "Unclassified"


@dataclass(frozen=True, slots=True)
class Regime:
    """A named calendar interval (inclusive). A regime may span several intervals."""

    name: str
    start: date
    end: date


REGIMES: tuple[Regime, ...] = (
    Regime("Calm I", date(2015, 1, 1), date(2019, 12, 31)),
    Regime("COVID shock", date(2020, 2, 1), date(2020, 5, 31)),
    Regime("Post-COVID", date(2020, 6, 1), date(2021, 12, 31)),
    Regime("Rates shock", date(2022, 1, 1), date(2023, 6, 30)),
    Regime("Calm II", date(2023, 7, 1), date(2025, 2, 28)),
    Regime("Tariff shock", date(2025, 3, 1), date(2025, 5, 31)),
    Regime("Calm II", date(2025, 6, 1), date(2025, 12, 31)),
)


@dataclass(frozen=True, slots=True)
class HoldingPeriod:
    """One rebalance and the holding period that follows it.

    Attributes:
        k: Period number, starting at 0.
        rebalance: Rebalance date; data up to and including it may be used.
        start: First day of the holding period.
        end: Last day of the holding period.
        rebalance_pos: Position of ``rebalance`` in the index.
        start_pos: Position of ``start`` (``rebalance_pos + 1``).
        end_pos: Position of ``end`` (inclusive).
        regime: Regime label (majority of days) or ``"Unclassified"``.
    """

    k: int
    rebalance: pd.Timestamp
    start: pd.Timestamp
    end: pd.Timestamp
    rebalance_pos: int
    start_pos: int
    end_pos: int
    regime: str


def assign_regime(days: pd.DatetimeIndex, regimes: Sequence[Regime] = REGIMES) -> str:
    """Regime containing more than half of ``days``, else ``"Unclassified"``."""
    counts: dict[str, int] = {}
    for r in regimes:
        inside = (days >= pd.Timestamp(r.start)) & (days <= pd.Timestamp(r.end))
        counts[r.name] = counts.get(r.name, 0) + int(inside.sum())
    if not counts:
        return UNCLASSIFIED
    best = max(counts, key=lambda name: counts[name])
    return best if counts[best] * 2 > len(days) else UNCLASSIFIED


@dataclass(frozen=True)
class Schedule:
    """The full backtest calendar built on a trading-day index."""

    index: pd.DatetimeIndex
    holding: int
    periods: tuple[HoldingPeriod, ...]

    def __len__(self) -> int:
        return len(self.periods)

    def rolling_window(self, period: HoldingPeriod, length: int) -> tuple[int, int]:
        """Positions ``(start, end_inclusive)`` of the ``length``-day window up to the rebalance.

        Raises:
            ValueError: If the index does not contain ``length`` days up to the rebalance.
        """
        if length < 2:
            raise ValueError(f"window length must be >= 2, got {length}")
        start = period.rebalance_pos - length + 1
        if start < 0:
            first = self.index[0].date()
            raise ValueError(f"period {period.k}: {length}-day window needs history before {first}")
        return start, period.rebalance_pos

    def expanding_window(self, period: HoldingPeriod, start_date: date) -> tuple[int, int]:
        """Positions ``(start, end_inclusive)`` from the first day on or after ``start_date``.

        Raises:
            ValueError: If ``start_date`` is after the rebalance date.
        """
        start = int(self.index.searchsorted(pd.Timestamp(start_date)))
        if start > period.rebalance_pos:
            raise ValueError(f"period {period.k}: expanding window starts after the rebalance")
        return start, period.rebalance_pos

    def to_frame(self) -> pd.DataFrame:
        """One row per holding period, for inspection and export."""
        return pd.DataFrame(
            [
                {
                    "k": p.k,
                    "rebalance": p.rebalance,
                    "start": p.start,
                    "end": p.end,
                    "days": p.end_pos - p.start_pos + 1,
                    "regime": p.regime,
                }
                for p in self.periods
            ]
        ).set_index("k")


def build_schedule(
    index: pd.DatetimeIndex,
    first_rebalance: date = DEFAULT_FIRST_REBALANCE,
    holding: int = DEFAULT_HOLDING,
    regimes: Sequence[Regime] = REGIMES,
) -> Schedule:
    """Build the backtest calendar on ``index``.

    Args:
        index: Sorted trading-day index (typically the returns panel's index).
        first_rebalance: The first rebalance is the last index date on or before this.
        holding: Holding period length in trading days.
        regimes: Regime intervals for the majority-of-days labels.

    Returns:
        A :class:`Schedule` with every full holding period.

    Raises:
        ValueError: On an unsorted index, ``holding < 1``, or a ``first_rebalance``
            before the start of the index.
    """
    if holding < 1:
        raise ValueError(f"holding must be >= 1, got {holding}")
    if not index.is_monotonic_increasing:
        raise ValueError("index must be sorted in increasing order")
    i0 = int(index.searchsorted(pd.Timestamp(first_rebalance), side="right")) - 1
    if i0 < 0:
        raise ValueError(f"first_rebalance {first_rebalance} is before the start of the index")

    periods: list[HoldingPeriod] = []
    k = 0
    while i0 + holding * (k + 1) <= len(index) - 1:
        reb = i0 + holding * k
        start, end = reb + 1, reb + holding
        periods.append(
            HoldingPeriod(
                k=k,
                rebalance=index[reb],
                start=index[start],
                end=index[end],
                rebalance_pos=reb,
                start_pos=start,
                end_pos=end,
                regime=assign_regime(index[start : end + 1], regimes),
            )
        )
        k += 1
    return Schedule(index=index, holding=holding, periods=tuple(periods))


def eligible_assets(returns: pd.DataFrame, start: int, end: int) -> list[str]:
    """Columns with no missing value in rows ``start..end`` (inclusive)."""
    window = returns.iloc[start : end + 1]
    return [str(c) for c in window.columns[window.notna().all().to_numpy()]]


__all__ = [
    "DEFAULT_FIRST_REBALANCE",
    "DEFAULT_HOLDING",
    "REGIMES",
    "UNCLASSIFIED",
    "HoldingPeriod",
    "Regime",
    "Schedule",
    "assign_regime",
    "build_schedule",
    "eligible_assets",
]
