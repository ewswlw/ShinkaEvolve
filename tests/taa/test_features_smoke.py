"""Smoke test for feature panel with synthetic spliced data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from taa import config
from taa.features import build_feature_panel
from taa.regime import add_regime_features, fit_hmm_regime, macro_panel_from_feature_panel
from taa.walk_forward import walk_forward_taa


def _synthetic_spliced() -> pd.DataFrame:
    idx = pd.bdate_range("2005-01-01", "2010-12-31")
    n = len(idx)
    rng = np.random.default_rng(0)
    data = {}
    for t in config.ETF_TICKERS:
        data[t] = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    return pd.DataFrame(data, index=idx)


def test_build_feature_panel_minimal() -> None:
    spliced = _synthetic_spliced()
    dates = spliced.resample("ME").last().index
    macro_yield = pd.DataFrame(
        {
            "USGG3M Index_px_last": np.linspace(1, 3, len(dates)),
            "USGG2YR Index_px_last": np.linspace(1, 4, len(dates)),
            "USGG5YR Index_px_last": np.linspace(2, 4, len(dates)),
            "USGG10YR Index_px_last": np.linspace(2, 5, len(dates)),
            "USGG30YR Index_px_last": np.linspace(3, 5, len(dates)),
        },
        index=dates,
    )
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(2)
    macro_credit = pd.DataFrame(
        {
            "LF98OAS Index_px_last": rng1.uniform(300, 500, len(dates)),
            "LUACOAS Index_px_last": rng2.uniform(50, 120, len(dates)),
        },
        index=dates,
    )
    macro_vol = pd.DataFrame(
        {
            "VIX Index_px_last": np.random.default_rng(3).uniform(10, 40, len(dates)),
            "MOVE Index_px_last": np.random.default_rng(4).uniform(50, 120, len(dates)),
        },
        index=dates,
    )
    ml = pd.DataFrame(
        {
            "INJCJC Index_px_last": np.random.default_rng(5).uniform(200000, 400000, len(dates)),
            "NAPMPMI Index_px_last": np.random.default_rng(6).uniform(45, 55, len(dates)),
            "IP CHNG Index_px_last": np.zeros(len(dates)),
            "CONCCONF Index_px_last": np.random.default_rng(7).uniform(80, 120, len(dates)),
            "LEI CHNG Index_px_last": np.zeros(len(dates)),
        },
        index=dates,
    )
    infl = pd.DataFrame(
        {
            "CPI YOY Index_px_last": np.ones(len(dates)) * 2.5,
            "USGGBE10 Index_px_last": np.ones(len(dates)) * 2.0,
        },
        index=dates,
    )
    fed = pd.DataFrame(
        {
            "FDTR Index_px_last": np.ones(len(dates)) * 5.0,
            "US0003M Index_px_last": np.ones(len(dates)) * 5.0,
        },
        index=dates,
    )
    liq = pd.DataFrame({"BFCIUS Index_px_last": np.zeros(len(dates))}, index=dates)
    val = pd.DataFrame(
        {"PE_RATIO_px_last": np.ones(len(dates)) * 20, "EARN_YLD_px_last": np.ones(len(dates)) * 0.05},
        index=dates,
    )

    data = {
        "spliced_levels": spliced,
        "macro": {
            "yield_curve": macro_yield,
            "credit": macro_credit,
            "volatility": macro_vol,
            "macro_leading": ml,
            "inflation": infl,
            "fed": fed,
            "liquidity": liq,
        },
        "valuation": val,
        "breadth": pd.DataFrame(),
    }
    panel = build_feature_panel(data)
    assert not panel.empty
    assert "date" in panel.columns
    assert "ticker" in panel.columns

    mh = macro_panel_from_feature_panel(panel)
    _, regime_df = fit_hmm_regime(mh)
    panel2 = add_regime_features(panel, regime_df)
    wf = walk_forward_taa(panel2, regime_df)
    assert not wf.empty
