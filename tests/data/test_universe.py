"""Tests for the study universe."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date

import pytest

from portfolio_risk.data import universe as u


class TestCounts:
    """Bucket cardinalities match the design document."""

    def test_total_is_75(self) -> None:
        assert len(u.UNIVERSE) == 75

    def test_equities_38(self) -> None:
        assert len(u.EQUITIES) == 38

    def test_srf_etfs_15(self) -> None:
        assert len(u.SRF_ETFS) == 15

    def test_rate_etfs_8(self) -> None:
        assert len(u.RATE_ETFS) == 8

    def test_commodity_fx_8(self) -> None:
        assert len(u.COMMODITY_FX) == 8

    def test_vol_spread_6(self) -> None:
        assert len(u.VOL_SPREAD) == 6


class TestSubgroupSplit:
    """Sub-classifications within each bucket match the design document."""

    def test_equity_sector_counts(self) -> None:
        counts: dict[str, int] = {}
        for inst in u.EQUITIES:
            counts[inst.subgroup] = counts.get(inst.subgroup, 0) + 1
        assert counts == {
            "Technology": 8,
            "Financials": 8,
            "Healthcare": 8,
            "Industrials": 7,
            "Consumer Staples": 7,
        }

    def test_srf_subgroup_counts(self) -> None:
        counts: dict[str, int] = {}
        for inst in u.SRF_ETFS:
            counts[inst.subgroup] = counts.get(inst.subgroup, 0) + 1
        assert counts == {"US Sector": 9, "Regional": 3, "Factor": 3}


class TestUniqueness:
    def test_no_duplicate_tickers(self) -> None:
        tickers = u.all_tickers()
        assert len(tickers) == len(set(tickers))

    def test_all_uppercase(self) -> None:
        for inst in u.UNIVERSE:
            assert inst.ticker == inst.ticker.upper()

    def test_no_empty_names(self) -> None:
        for inst in u.UNIVERSE:
            assert inst.name.strip()

    def test_no_empty_subgroups(self) -> None:
        for inst in u.UNIVERSE:
            assert inst.subgroup.strip()


class TestInception:
    """Post-2010 launches must have an inception date; pre-2010 must not."""

    POST_2010_TICKERS: frozenset[str] = frozenset({"MTUM", "VIXY", "SPLV", "BKLN"})

    def test_post_2010_have_inception(self) -> None:
        for ticker in self.POST_2010_TICKERS:
            inst = u.get_instrument(ticker)
            assert inst.inception is not None, f"{ticker} missing inception"
            assert inst.inception > date(
                2010, 1, 4
            ), f"{ticker} inception {inst.inception} is not post-2010"

    def test_others_have_no_inception(self) -> None:
        for inst in u.UNIVERSE:
            if inst.ticker not in self.POST_2010_TICKERS:
                assert (
                    inst.inception is None
                ), f"{inst.ticker} has inception {inst.inception} but should be None"


class TestAccessors:
    def test_get_instrument_by_ticker(self) -> None:
        aapl = u.get_instrument("AAPL")
        assert aapl.name == "Apple Inc."
        assert aapl.bucket == "equity"
        assert aapl.subgroup == "Technology"

    def test_get_instrument_case_insensitive(self) -> None:
        assert u.get_instrument("aapl") is u.get_instrument("AAPL")

    def test_get_instrument_missing_raises(self) -> None:
        with pytest.raises(KeyError):
            u.get_instrument("XXXX")

    def test_by_bucket_equity(self) -> None:
        assert len(u.by_bucket("equity")) == 38

    def test_by_bucket_srf(self) -> None:
        assert len(u.by_bucket("srf_etf")) == 15

    def test_by_bucket_returns_tuple(self) -> None:
        result = u.by_bucket("equity")
        assert isinstance(result, tuple)

    def test_by_subgroup_technology(self) -> None:
        techs = u.by_subgroup("Technology")
        assert len(techs) == 8
        assert all(inst.bucket == "equity" for inst in techs)

    def test_by_subgroup_us_sector(self) -> None:
        assert len(u.by_subgroup("US Sector")) == 9

    def test_all_tickers_length(self) -> None:
        assert len(u.all_tickers()) == 75


class TestMetadataFrame:
    def test_shape(self) -> None:
        df = u.metadata_frame()
        assert len(df) == 75
        assert set(df.columns) == {"name", "bucket", "subgroup", "inception", "notes"}
        assert df.index.name == "ticker"

    def test_index_matches_tickers(self) -> None:
        df = u.metadata_frame()
        assert list(df.index) == u.all_tickers()

    def test_inception_column_types(self) -> None:
        df = u.metadata_frame()
        inception = df["inception"].dropna()
        assert len(inception) == 4
        for value in inception:
            assert isinstance(value, date)


class TestImmutability:
    def test_instruments_are_frozen(self) -> None:
        aapl = u.get_instrument("AAPL")
        with pytest.raises(FrozenInstanceError):
            aapl.ticker = "NOPE"  # type: ignore[misc]
