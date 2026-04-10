"""Pytest fixtures: Mock Bloomberg API for tests without Terminal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class MockBlp:
    """Minimal mock mirroring xbbg.blp return shapes for unit tests."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def bdp(
        self,
        tickers: str | list[str],
        flds: str | list[str],
        **kwargs: object,
    ) -> pd.DataFrame:
        if isinstance(tickers, str):
            tickers = [tickers]
        if isinstance(flds, str):
            flds = [flds]
        data: dict[str, np.ndarray] = {}
        for fld in flds:
            fl = fld.lower()
            if "px" in fl or "last" in fl or "price" in fl:
                data[fl] = self.rng.uniform(50, 500, len(tickers))
            else:
                data[fl] = self.rng.uniform(0, 100, len(tickers))
        return pd.DataFrame(data, index=tickers)

    def bdh(
        self,
        tickers: str | list[str],
        flds: str | list[str],
        start_date: str,
        end_date: str | None = None,
        **kwargs: object,
    ) -> pd.DataFrame:
        if isinstance(tickers, str):
            tickers = [tickers]
        if isinstance(flds, str):
            flds = [flds]
        end = end_date or pd.Timestamp.today().strftime("%Y-%m-%d")
        dates = pd.bdate_range(start_date, end)
        cols = pd.MultiIndex.from_product([tickers, [f.lower() for f in flds]])
        arr = self.rng.uniform(50, 200, (len(dates), len(cols)))
        return pd.DataFrame(arr, index=dates, columns=cols)

    def bds(self, ticker: str, fld: str, **kwargs: object) -> pd.DataFrame:
        members = [f"T{i:02d} US Equity" for i in range(20)]
        return pd.DataFrame(
            {"member_ticker_and_exchange_code": members},
            index=[ticker] * len(members),
        )


@pytest.fixture
def mock_blp() -> MockBlp:
    return MockBlp(seed=123)
