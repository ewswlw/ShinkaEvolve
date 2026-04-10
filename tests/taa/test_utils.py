"""Tests for taa.utils."""

from __future__ import annotations

import numpy as np
import pandas as pd

from taa.utils import expanding_zscore, monthly_align


def test_monthly_align() -> None:
    idx = pd.bdate_range("2020-01-01", periods=60)
    df = pd.DataFrame({"a": np.arange(60)}, index=idx)
    m = monthly_align(df)
    assert len(m) < len(df)
    assert isinstance(m.index, pd.DatetimeIndex)


def test_expanding_zscore() -> None:
    s = pd.Series(np.random.randn(100))
    z = expanding_zscore(s, min_periods=10)
    assert len(z) == len(s)
