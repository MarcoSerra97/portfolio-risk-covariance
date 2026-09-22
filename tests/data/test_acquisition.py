"""Tests for the acquisition module.

Uses a synthetic ``downloader`` in place of real yfinance calls, so the
pipeline can be exercised end-to-end without network access.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.data import acquisition as acq
from portfolio_risk.data import calendar as cal
from portfolio_risk.data import paths


def _make_fake_downloader(seed: int = 0, fail_first_n_calls: int = 0) -> acq.YfDownloader:
    """Build a synthetic downloader that returns MultiIndex-columned data.

    Mimics yfinance's shape for multi-ticker calls: columns are a
    ``MultiIndex`` with the top level being one of {'Open', 'High', 'Low',
    'Close', 'Volume'} and the second level being the ticker.
    """
    call_count = [0]

    def fake(tickers: list[str], start: date, end: date) -> pd.DataFrame:
        call_count[0] += 1
        if call_count[0] <= fail_first_n_calls:
            raise ConnectionError(f"Simulated failure on call {call_count[0]}")

        calendar = cal.nyse_trading_days(start, end)
        rng = np.random.default_rng(seed)
        n = len(calendar)

        # Random-walk price paths starting at 100
        prices_by_ticker = {t: 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n))) for t in tickers}

        frames = []
        for field in ("Open", "High", "Low", "Close", "Volume"):
            data = {
                t: prices_by_ticker[t] if field != "Volume" else np.ones(n) * 1_000_000
                for t in tickers
            }
            frame = pd.DataFrame(data, index=calendar)
            frame.columns = pd.MultiIndex.from_product([[field], tickers])
            frames.append(frame)

        return pd.concat(frames, axis=1)

    return fake


class TestFetchPrices:
    def test_basic_shape(self) -> None:
        downloader = _make_fake_downloader()
        result = acq.fetch_prices(
            ["AAPL", "MSFT"], date(2019, 1, 2), date(2019, 1, 15), downloader=downloader
        )
        assert list(result.columns) == ["AAPL", "MSFT"]
        assert len(result) == 10
        assert result.index.tz is None

    def test_column_order_preserved(self) -> None:
        downloader = _make_fake_downloader()
        result = acq.fetch_prices(
            ["MSFT", "AAPL", "GOOG"],
            date(2019, 1, 2),
            date(2019, 1, 15),
            downloader=downloader,
        )
        assert list(result.columns) == ["MSFT", "AAPL", "GOOG"]

    def test_batching(self) -> None:
        tickers = [f"T{i}" for i in range(60)]
        downloader = _make_fake_downloader()
        result = acq.fetch_prices(
            tickers,
            date(2019, 1, 2),
            date(2019, 1, 15),
            batch_size=25,
            downloader=downloader,
        )
        assert list(result.columns) == tickers
        assert len(result) == 10

    def test_retry_on_failure(self) -> None:
        downloader = _make_fake_downloader(fail_first_n_calls=2)
        result = acq.fetch_prices(
            ["AAPL"],
            date(2019, 1, 2),
            date(2019, 1, 15),
            retries=3,
            retry_backoff=0.001,
            downloader=downloader,
        )
        assert list(result.columns) == ["AAPL"]

    def test_exhausted_retries_raises(self) -> None:
        downloader = _make_fake_downloader(fail_first_n_calls=100)
        with pytest.raises(RuntimeError, match="All 2 attempts failed"):
            acq.fetch_prices(
                ["AAPL"],
                date(2019, 1, 2),
                date(2019, 1, 15),
                retries=2,
                retry_backoff=0.001,
                downloader=downloader,
            )

    def test_empty_tickers_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            acq.fetch_prices([], date(2019, 1, 2), date(2019, 1, 15))

    def test_end_before_start_raises(self) -> None:
        with pytest.raises(ValueError, match="before start"):
            acq.fetch_prices(["AAPL"], date(2019, 1, 15), date(2019, 1, 2))


class TestExtractClose:
    def test_multiindex_extraction(self) -> None:
        downloader = _make_fake_downloader()
        raw = downloader(["AAPL", "MSFT"], date(2019, 1, 2), date(2019, 1, 15))
        result = acq._extract_close(raw, ["AAPL", "MSFT"])
        assert list(result.columns) == ["AAPL", "MSFT"]

    def test_flat_column_extraction(self) -> None:
        calendar = cal.nyse_trading_days(date(2019, 1, 2), date(2019, 1, 15))
        raw = pd.DataFrame(
            {"Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 1.5, "Volume": 100.0},
            index=calendar,
        )
        result = acq._extract_close(raw, ["AAPL"])
        assert list(result.columns) == ["AAPL"]
        assert (result["AAPL"] == 1.5).all()

    def test_empty_frame_returns_empty(self) -> None:
        raw = pd.DataFrame()
        result = acq._extract_close(raw, ["AAPL", "MSFT"])
        assert list(result.columns) == ["AAPL", "MSFT"]
        assert len(result) == 0

    def test_missing_close_raises(self) -> None:
        raw = pd.DataFrame({"Open": [1.0], "Volume": [100.0]})
        with pytest.raises(KeyError, match="'Close'"):
            acq._extract_close(raw, ["AAPL"])


class TestRunPipeline:
    """End-to-end pipeline against tmp data dirs with a mocked downloader."""

    @pytest.fixture(autouse=True)
    def redirect_data_dirs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        raw = tmp_path / "data" / "raw"
        processed = tmp_path / "data" / "processed"
        manifest_path = tmp_path / "data" / "manifest.yaml"
        monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
        monkeypatch.setattr(paths, "RAW_DIR", raw)
        monkeypatch.setattr(paths, "PROCESSED_DIR", processed)
        monkeypatch.setattr(paths, "MANIFEST_PATH", manifest_path)

    def test_produces_expected_artefacts(self) -> None:
        downloader = _make_fake_downloader()
        artefacts = acq.run_pipeline(
            start=date(2019, 1, 2), end=date(2019, 6, 28), downloader=downloader
        )
        assert set(artefacts.keys()) == {
            "raw_prices",
            "cleaned_prices",
            "returns",
            "anomalies",
            "dropped_dates",
            "manifest",
        }
        for name, path in artefacts.items():
            assert path.exists(), f"{name} artefact missing at {path}"

    def test_raw_prices_shape(self) -> None:
        downloader = _make_fake_downloader()
        artefacts = acq.run_pipeline(
            start=date(2019, 1, 2), end=date(2019, 6, 28), downloader=downloader
        )
        raw = pd.read_parquet(artefacts["raw_prices"])
        assert raw.shape[1] == 75

    def test_returns_computable(self) -> None:
        downloader = _make_fake_downloader()
        artefacts = acq.run_pipeline(
            start=date(2019, 1, 2), end=date(2019, 6, 28), downloader=downloader
        )
        returns = pd.read_parquet(artefacts["returns"])
        assert returns.shape[1] == 75
        assert returns.iloc[0].isna().all()
        assert returns.iloc[1:].notna().sum().sum() > 0

    def test_manifest_written_with_hashes(self) -> None:
        from portfolio_risk.data import manifest as mf

        downloader = _make_fake_downloader()
        artefacts = acq.run_pipeline(
            start=date(2019, 1, 2), end=date(2019, 6, 28), downloader=downloader
        )
        loaded = mf.read_manifest(artefacts["manifest"])
        assert loaded["universe"]["n_instruments"] == 75
        assert "raw_prices_sha256" in loaded["integrity"]
        assert "returns_sha256" in loaded["integrity"]
        assert "cleaned_prices_sha256" in loaded["integrity"]

    def test_pipeline_is_idempotent(self) -> None:
        downloader = _make_fake_downloader(seed=42)
        first = acq.run_pipeline(
            start=date(2019, 1, 2), end=date(2019, 6, 28), downloader=downloader
        )
        prices_first = pd.read_parquet(first["cleaned_prices"])

        downloader2 = _make_fake_downloader(seed=42)
        second = acq.run_pipeline(
            start=date(2019, 1, 2), end=date(2019, 6, 28), downloader=downloader2
        )
        prices_second = pd.read_parquet(second["cleaned_prices"])
        pd.testing.assert_frame_equal(prices_first, prices_second)


class TestMainCli:
    @pytest.fixture(autouse=True)
    def redirect_data_dirs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
        monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "data" / "raw")
        monkeypatch.setattr(paths, "PROCESSED_DIR", tmp_path / "data" / "processed")
        monkeypatch.setattr(paths, "MANIFEST_PATH", tmp_path / "data" / "manifest.yaml")

    def test_returns_zero_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        downloader = _make_fake_downloader()
        monkeypatch.setattr(acq, "_default_downloader", downloader)
        code = acq.main(["--start", "2019-01-02", "--end", "2019-06-28"])
        assert code == 0

    def test_returns_one_on_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def broken(tickers: list[str], start: date, end: date) -> pd.DataFrame:
            raise ConnectionError("boom")

        monkeypatch.setattr(acq, "_default_downloader", broken)
        code = acq.main(["--start", "2019-01-02", "--end", "2019-06-28"])
        assert code == 1

    def test_parses_dates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, date] = {}

        def fake_run(
            start: date,
            end: date | None = None,
            downloader: acq.YfDownloader | None = None,
        ) -> dict[str, Path]:
            captured["start"] = start
            captured["end"] = end or date(1900, 1, 1)
            return {}

        monkeypatch.setattr(acq, "run_pipeline", fake_run)
        acq.main(["--start", "2020-03-15", "--end", "2020-04-01"])
        assert captured["start"] == date(2020, 3, 15)
        assert captured["end"] == date(2020, 4, 1)
