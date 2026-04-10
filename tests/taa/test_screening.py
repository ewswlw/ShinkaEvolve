"""Tests for screening."""

from __future__ import annotations

import numpy as np
import pandas as pd

from taa.screening import predictability_score, screen_factors


def test_predictability_score() -> None:
    r = pd.Series(np.random.randn(120))
    out = predictability_score(r)
    assert "score" in out
    assert 0 <= out["score"] <= 100


def test_screen_factors() -> None:
    n = 60
    f = pd.DataFrame({"a": np.random.randn(n), "b": np.random.randn(n)})
    y = pd.Series(np.random.randn(n))
    out = screen_factors(f, y, t_threshold=0.1)
    assert "passed" in out
