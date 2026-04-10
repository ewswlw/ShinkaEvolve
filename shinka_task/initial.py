"""
TAA Strategy candidate for ShinkaEvolve optimisation.

The EVOLVE-BLOCK contains the monthly weight allocation logic.
Everything outside the block (data loading, cost model, metric calc) is fixed.

Targets:  CAGR > 15%,  Calmar ratio > 1.0
Universe: 12 ETFs via spliced total-return data (2006-04-2026)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants (fixed — do NOT modify outside EVOLVE-BLOCK)
# ---------------------------------------------------------------------------
def _find_data_path() -> Path:
    """Walk up directory tree to find the spliced returns parquet file.
    
    Robust regardless of where Shinka copies the candidate file to.
    """
    search = Path(__file__).resolve().parent
    target = Path("data") / "taa" / "spliced_total_returns.parquet"
    for _ in range(10):
        candidate = search / target
        if candidate.exists():
            return candidate
        search = search.parent
    raise FileNotFoundError(
        f"Cannot find {target} within 10 parent directories of {Path(__file__).resolve()}"
    )


DATA_PATH = _find_data_path()

TICKERS: list[str] = [
    "SPY US Equity",
    "QQQ US Equity",
    "EFA US Equity",
    "EEM US Equity",
    "AGG US Equity",
    "TLT US Equity",
    "LQD US Equity",
    "HYG US Equity",
    "GLD US Equity",
    "DBC US Equity",
    "IYR US Equity",
    "SHY US Equity",
]
N = len(TICKERS)
BACKTEST_START = pd.Timestamp("2006-01-31")
COST_BPS = 5.0       # one-way transaction cost per trade
MAX_WEIGHT = 0.40    # max single-asset weight
MIN_SHY = 0.05       # floor on short-term treasury (cash-like)


# ---------------------------------------------------------------------------
# EVOLVE-BLOCK-START
# ---------------------------------------------------------------------------
def get_monthly_weights(
    monthly_rets: pd.DataFrame,
    current_date: pd.Timestamp,
) -> np.ndarray:
    """
    Return next-month portfolio weights given historical monthly returns.

    Args:
        monthly_rets: DataFrame of monthly returns, columns = TICKERS,
                      indexed by month-end dates UP TO BUT NOT INCLUDING
                      current_date (i.e. pure lookback, no look-ahead).
        current_date: The month-end date we are allocating FOR.

    Returns:
        np.ndarray of shape (N,) — portfolio weights, will be normalised
        and clipped externally, but should already sum to ~1.
    """
    if len(monthly_rets) < 2:
        return np.ones(N) / N

    # 12-month momentum: rank assets by trailing 12-month cumulative return
    lookback = min(12, len(monthly_rets))
    past = monthly_rets.iloc[-lookback:]
    cumret = (1.0 + past).prod() - 1.0

    # Assign weight proportional to positive excess rank
    ranked = cumret.rank(ascending=True)           # 1 = worst, N = best
    w = np.maximum(ranked.values - N / 2.0, 0.0)  # keep top half only

    if w.sum() < 1e-9:
        # Fallback: equal weight
        return np.ones(N) / N

    w = w / w.sum()

    # Hard floor on SHY (short-duration treasury as safe-haven minimum)
    shy_i = TICKERS.index("SHY US Equity")
    if w[shy_i] < MIN_SHY:
        w[shy_i] = MIN_SHY
        w = w / w.sum()

    return w
# ---------------------------------------------------------------------------
# EVOLVE-BLOCK-END
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Fixed infrastructure — backtest harness (do NOT modify)
# ---------------------------------------------------------------------------
def _load_monthly_returns() -> pd.DataFrame:
    """Load spliced total-return prices, resample to month-end, compute pct change."""
    prices = pd.read_parquet(DATA_PATH)
    prices = prices.reindex(columns=TICKERS).ffill()
    monthly_prices = prices.resample("ME").last()
    monthly_rets = monthly_prices.pct_change()
    monthly_rets = monthly_rets[monthly_rets.index >= BACKTEST_START]
    # Drop rows where ALL assets are NaN (first row after pct_change)
    monthly_rets = monthly_rets.dropna(how="all")
    return monthly_rets


def run_experiment(seed: int | None = None) -> dict:
    """
    Run the full TAA backtest and return performance metrics.

    Returns dict with keys: cagr, calmar, max_dd, sharpe, n_months.
    """
    monthly_rets = _load_monthly_returns()

    portfolio_rets: list[float] = []
    dates: list[pd.Timestamp] = []
    w_prev = np.ones(N) / N
    cost = COST_BPS / 10_000.0

    for i in range(len(monthly_rets)):
        dt = monthly_rets.index[i]
        # Strict look-ahead prevention: history = all data BEFORE index i
        history = monthly_rets.iloc[:i]

        try:
            w = get_monthly_weights(history, dt)
        except Exception:
            w = w_prev.copy()

        # Apply constraints
        w = np.clip(w, 0.0, MAX_WEIGHT)
        if w.sum() < 1e-9:
            w = np.ones(N) / N
        w = w / w.sum()

        # Transaction cost proportional to one-way turnover
        turnover = float(np.sum(np.abs(w - w_prev)) / 2.0)
        row = monthly_rets.iloc[i].reindex(TICKERS).fillna(0.0).values
        gross_ret = float(row @ w)
        net_ret = gross_ret - turnover * cost * 2.0

        portfolio_rets.append(net_ret)
        dates.append(dt)
        w_prev = w.copy()

    r = pd.Series(portfolio_rets, index=dates, name="ret")

    # ---- Performance metrics ----
    n_months = len(r)
    n_years = n_months / 12.0

    if n_years <= 0:
        return {"cagr": 0.0, "calmar": 0.0, "max_dd": 0.0, "sharpe": 0.0, "n_months": 0}

    cagr = float((1.0 + r).prod() ** (1.0 / n_years) - 1.0)
    vol = float(r.std(ddof=1)) * np.sqrt(12.0)
    sharpe = float(cagr / vol) if vol > 1e-9 else 0.0

    cum = (1.0 + r).cumprod()
    max_dd = float((cum / cum.cummax() - 1.0).min())
    calmar = float(-cagr / max_dd) if max_dd < -1e-9 else 0.0

    return {
        "cagr": cagr,
        "calmar": calmar,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "n_months": n_months,
    }


if __name__ == "__main__":
    result = run_experiment()
    print(
        f"CAGR={result['cagr']:.2%}  Calmar={result['calmar']:.2f}  "
        f"MaxDD={result['max_dd']:.2%}  Sharpe={result['sharpe']:.2f}  "
        f"Months={result['n_months']}"
    )
