"""Log-return computation from adjusted-close price panels.

Follows the design document §5.2:
    r_t = log(P_t / P_{t-1})   on adjusted close

The first observation of each series produces NaN (no lag-1 available).
NaN propagates through gaps: r_t is NaN whenever either P_t or P_{t-1}
is NaN. The caller is responsible for dropping or masking initial NaN
rows for downstream computation.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log-returns from an adjusted-close price panel.

    Args:
        prices: Adjusted-close panel indexed by a ``DatetimeIndex``,
            columns are ticker symbols. May contain NaN gaps (e.g. from
            pre-inception dates or unfilled multi-day gaps).

    Returns:
        Log-return panel with the same shape as ``prices``. The first row
        is always all-NaN. NaN in either ``P_t`` or ``P_{t-1}`` produces
        NaN in ``r_t``.

    Raises:
        TypeError: If ``prices.index`` is not a ``DatetimeIndex``.
        ValueError: If any non-NaN price is non-positive (log-returns
            undefined). NaN values are ignored by this check.
    """
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError(f"prices.index must be a DatetimeIndex, got {type(prices.index).__name__}")
    # NaN <= 0 evaluates to False, so this catches only real non-positives.
    if (prices <= 0).any().any():
        raise ValueError("Non-positive prices detected; log-returns undefined")

    return cast(pd.DataFrame, np.log(prices / prices.shift(1)))
