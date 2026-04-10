"""Central configuration: tickers, proxy maps, hyperparameters, backtest constants."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Reproducibility
RANDOM_STATE: int = 42

# Project paths (repo root = parent of taa/)
PACKAGE_ROOT: Path = Path(__file__).resolve().parent
REPO_ROOT: Path = PACKAGE_ROOT.parent
DATA_DIR: Path = REPO_ROOT / "data" / "taa"
RESULTS_DIR: Path = DATA_DIR / "results"

# Date ranges (Bloomberg pulls start one year before backtest for warmup)
PULL_START: str = "2005-01-01"
BACKTEST_START: str = "2006-01-31"

# ETF universe: ETF ticker -> proxy index ticker, inception (YYYY-MM-DD), asset class
ETF_UNIVERSE: dict[str, dict[str, Any]] = {
    "SPY US Equity": {
        "proxy": "SPXT Index",
        "proxy_name": "S&P 500 Total Return",
        "inception": "1993-01-29",
        "asset_class": "US Large Cap Equity",
    },
    "QQQ US Equity": {
        "proxy": "XNDX Index",
        "proxy_name": "NASDAQ 100 Total Return",
        "inception": "1999-03-10",
        "asset_class": "US Tech / Growth",
    },
    "EFA US Equity": {
        "proxy": "NDDUEAFE Index",
        "proxy_name": "MSCI EAFE Net TR USD",
        "inception": "2001-08-14",
        "asset_class": "Intl Developed Equity",
    },
    "EEM US Equity": {
        "proxy": "NDUEEGF Index",
        "proxy_name": "MSCI EM Net TR USD",
        "inception": "2003-04-07",
        "asset_class": "Emerging Market Equity",
    },
    "AGG US Equity": {
        "proxy": "LBUSTRUU Index",
        "proxy_name": "Bloomberg US Agg Bond TR",
        "inception": "2003-09-22",
        "asset_class": "US Aggregate Bond",
    },
    "TLT US Equity": {
        "proxy": "LUATTRUU Index",
        "proxy_name": "Bloomberg US Treasury 20+Y TR",
        "inception": "2002-07-22",
        "asset_class": "US Long Treasury 20+Y",
    },
    "LQD US Equity": {
        "proxy": "LUACTRUU Index",
        "proxy_name": "Bloomberg US Corporate IG TR",
        "inception": "2002-07-22",
        "asset_class": "US IG Corporate Bond",
    },
    "HYG US Equity": {
        "proxy": "LF98TRUU Index",
        "proxy_name": "Bloomberg US Corporate HY TR",
        "inception": "2007-04-04",
        "asset_class": "US High Yield Corporate",
    },
    "GLD US Equity": {
        "proxy": "XAU Curncy",
        "proxy_name": "Gold Spot USD/oz",
        "inception": "2004-11-18",
        "asset_class": "Gold",
    },
    "DBC US Equity": {
        "proxy": "DBLCDBCE Index",
        "proxy_name": "DBIQ Opt Yield Diversified Commodity",
        "inception": "2006-02-03",
        "asset_class": "Broad Commodities",
    },
    "IYR US Equity": {
        "proxy": "DWRTF Index",
        "proxy_name": "Dow Jones US Real Estate TR",
        "inception": "2000-06-12",
        "asset_class": "US REITs",
    },
    "SHY US Equity": {
        "proxy": "LD12TRUU Index",
        "proxy_name": "Bloomberg US Treasury 1-3Y TR",
        "inception": "2002-07-22",
        "asset_class": "Short-Term Treasury 1-3Y",
    },
}

ETF_TICKERS: list[str] = list(ETF_UNIVERSE.keys())
PROXY_TICKERS: list[str] = [ETF_UNIVERSE[t]["proxy"] for t in ETF_TICKERS]

# Macro / cross-asset Bloomberg tickers (bdh PX_LAST unless noted)
MACRO_TICKERS: dict[str, list[str]] = {
    "yield_curve": [
        "USGG3M Index",
        "USGG2YR Index",
        "USGG5YR Index",
        "USGG10YR Index",
        "USGG30YR Index",
    ],
    "credit": ["LF98OAS Index", "LUACOAS Index"],
    "volatility": ["VIX Index", "MOVE Index"],
    "macro_leading": [
        "INJCJC Index",
        "NAPMPMI Index",
        "IP CHNG Index",
        "CONCCONF Index",
        "LEI CHNG Index",
    ],
    "inflation": ["CPI YOY Index", "USGGBE10 Index"],
    "fed": ["FDTR Index", "US0003M Index"],
    "liquidity": ["BFCIUS Index"],
}

# HMM regime inputs (macro feature column names after features.py naming)
HMM_INPUT_FEATURES: list[str] = [
    "slope_2s10s",
    "hy_oas_zscore",
    "vix_zscore",
    "ism_pmi_chg_3m",
    "claims_yoy_chg",
    "fin_conditions",
    "pct_above_200dma",
]

# LightGBM hyperparameter grid (small; expand cautiously — DSR penalty)
LGBM_GRID: dict[str, list[Any]] = {
    "n_estimators": [100, 200],
    "max_depth": [2, 3],
    "learning_rate": [0.05, 0.1],
    "min_child_samples": [10, 20],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    "reg_alpha": [0.0, 0.1],
    "reg_lambda": [0.0, 0.1],
}

# Purged CV / walk-forward
PURGED_N_SPLITS: int = 5
EMBARGO_MONTHS: int = 2
WF_TRAIN_MONTHS: int = 36
WF_TEST_MONTHS: int = 12
WF_STEP_MONTHS: int = 6

# Backtest
BACKTEST_PARAMS: dict[str, Any] = {
    "cost_bps_normal": 5.0,
    "cost_bps_stress": 10.0,
    "vix_stress_threshold": 30.0,
    "circuit_dd_21d": -0.07,
    "max_one_way_turnover": 0.5,
    "max_single_weight": 0.40,
    "min_shy_weight": 0.05,
    "initial_capital": 1_000_000.0,
}

# Parquet cache filenames
CACHE_FILES: dict[str, str] = {
    "etf_prices": "etf_prices.parquet",
    "index_proxies": "index_proxies.parquet",
    "macro_yield": "macro_yield.parquet",
    "macro_credit": "macro_credit.parquet",
    "macro_vol": "macro_vol.parquet",
    "macro_leading": "macro_leading.parquet",
    "macro_inflation": "macro_inflation.parquet",
    "macro_fed": "macro_fed.parquet",
    "macro_liquidity": "macro_liquidity.parquet",
    "valuation": "valuation.parquet",
    "breadth": "breadth.parquet",
    "spliced_returns": "spliced_total_returns.parquet",
}

# SPY / AGG for benchmarks and cross-asset features
SPY_TICKER: str = "SPY US Equity"
AGG_TICKER: str = "AGG US Equity"
