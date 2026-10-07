"""Tests for the paired stationary block bootstrap."""

from __future__ import annotations

import numpy as np
import pytest

from portfolio_risk.evaluation import bootstrap as b


def _mean_run_length(idx_row: np.ndarray, n: int) -> float:
    """Average length of runs of consecutive (wrapping) indices."""
    breaks = np.flatnonzero(np.diff(idx_row) % n != 1)
    return len(idx_row) / (len(breaks) + 1)


class TestIndices:
    def test_shape_and_range(self) -> None:
        idx = b.stationary_bootstrap_indices(50, 3.0, 20, np.random.default_rng(0))
        assert idx.shape == (20, 50)
        assert idx.min() >= 0 and idx.max() < 50

    def test_mean_block_length(self) -> None:
        idx = b.stationary_bootstrap_indices(2000, 5.0, 10, np.random.default_rng(1))
        runs = np.mean([_mean_run_length(r, 2000) for r in idx])
        assert runs == pytest.approx(5.0, rel=0.1)

    def test_block_length_one_is_iid(self) -> None:
        idx = b.stationary_bootstrap_indices(2000, 1.0, 5, np.random.default_rng(2))
        runs = np.mean([_mean_run_length(r, 2000) for r in idx])
        assert runs == pytest.approx(1.0, abs=0.05)

    def test_reproducible(self) -> None:
        a = b.stationary_bootstrap_indices(30, 3.0, 5, np.random.default_rng(7))
        c = b.stationary_bootstrap_indices(30, 3.0, 5, np.random.default_rng(7))
        np.testing.assert_array_equal(a, c)

    @pytest.mark.parametrize(("n", "block", "reps"), [(0, 3.0, 5), (10, 0.5, 5), (10, 3.0, 0)])
    def test_rejects_bad_arguments(self, n: int, block: float, reps: int) -> None:
        with pytest.raises(ValueError):
            b.stationary_bootstrap_indices(n, block, reps, np.random.default_rng(0))


class TestPairedBootstrap:
    def test_estimate_is_statistic_on_data(self) -> None:
        x = np.random.default_rng(3).normal(size=(100, 2))
        res = b.paired_bootstrap(x, lambda d: d.mean(axis=0), 3.0, 200, seed=1)
        np.testing.assert_allclose(res.estimate, x.mean(axis=0))
        assert res.draws.shape == (200, 2)
        assert (res.lower <= res.upper).all()

    def test_pairing_cancels_shared_noise(self) -> None:
        """Column 1 = column 0 + 0.3 exactly: the paired difference has no spread."""
        base = np.random.default_rng(4).normal(size=131)
        x = np.column_stack([base, base + 0.3])
        res = b.paired_bootstrap(x, b.mean_differences(baseline=0), 3.0, 500, seed=2)
        np.testing.assert_allclose(res.lower[1], 0.3)
        np.testing.assert_allclose(res.upper[1], 0.3)
        assert res.estimate[0] == 0.0

    def test_interval_width_shrinks_with_sample_size(self) -> None:
        rng = np.random.default_rng(5)

        def width(n: int) -> float:
            res = b.paired_bootstrap(rng.normal(size=n), lambda d: d.mean(axis=0), 3.0, 500, seed=3)
            return float(res.upper[0] - res.lower[0])

        assert width(2000) < width(100) / 3

    def test_coverage_of_the_mean(self) -> None:
        """Roughly nominal coverage for an i.i.d. mean."""
        rng = np.random.default_rng(6)
        hits = 0
        for s in range(150):
            x = rng.normal(loc=1.0, size=131)
            res = b.paired_bootstrap(x, lambda d: d.mean(axis=0), 1.0, 300, seed=s)
            hits += int(res.lower[0] <= 1.0 <= res.upper[0])
        assert 0.88 <= hits / 150 <= 0.99

    def test_reproducible_with_seed(self) -> None:
        x = np.random.default_rng(7).normal(size=(60, 3))
        r1 = b.paired_bootstrap(x, b.volatility_ratios(0), 21.0, 100, seed=11)
        r2 = b.paired_bootstrap(x, b.volatility_ratios(0), 21.0, 100, seed=11)
        np.testing.assert_array_equal(r1.draws, r2.draws)


class TestStatistics:
    def test_volatility_ratios(self) -> None:
        x = np.random.default_rng(8).normal(size=(500, 2)) * np.array([1.0, 2.0])
        ratios = b.volatility_ratios(baseline=0)(x)
        assert ratios[0] == 1.0
        assert ratios[1] == pytest.approx(2.0, rel=0.1)
