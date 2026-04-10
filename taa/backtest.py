"""Portfolio simulation, costs, circuit breaker (monthly baseline)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from taa import config

logger = logging.getLogger(__name__)


def apply_turnover_cap(
    w_new: np.ndarray,
    w_old: np.ndarray,
    max_one_way: float,
) -> np.ndarray:
    """Scale move toward w_new if turnover exceeds cap."""
    turn = float(np.sum(np.abs(w_new - w_old)) / 2.0)
    if turn <= max_one_way:
        return w_new
    scale = max_one_way / max(turn, 1e-12)
    return w_old + scale * (w_new - w_old)


def apply_constraints(w: np.ndarray, tickers: list[str]) -> np.ndarray:
    """Max weight, min SHY."""
    w = np.clip(w, 0.0, config.BACKTEST_PARAMS["max_single_weight"])
    w = w / w.sum()
    if "SHY US Equity" in tickers:
        i = tickers.index("SHY US Equity")
        if w[i] < config.BACKTEST_PARAMS["min_shy_weight"]:
            w[i] = config.BACKTEST_PARAMS["min_shy_weight"]
            w = w / w.sum()
    return w


def run_backtest_monthly(
    monthly_returns: pd.Series,
    vix_monthly: pd.Series | None = None,
) -> dict[str, Any]:
    """
    Apply transaction costs and optional stress costs when VIX > threshold.
    monthly_returns: Series indexed by month-end date.
    """
    r = monthly_returns.astype(float).copy()
    cost = config.BACKTEST_PARAMS["cost_bps_normal"] / 10000.0
    stress = config.BACKTEST_PARAMS["cost_bps_stress"] / 10000.0
    thr = config.BACKTEST_PARAMS["vix_stress_threshold"]

    adj = []
    for dt in r.index:
        c = cost
        if vix_monthly is not None and dt in vix_monthly.index:
            if float(vix_monthly.loc[dt]) > thr:
                c = stress
        adj.append(r.loc[dt] - c * 2)

    net = pd.Series(adj, index=r.index, name="ret_net")
    return {"returns_net": net, "returns_gross": r}


def run_benchmarks(
    panel: pd.DataFrame,
) -> dict[str, pd.Series]:
    """SPY, 60/40, equal-weight from same panel."""
    tickers = config.ETF_TICKERS
    spy = config.SPY_TICKER
    agg = config.AGG_TICKER

    def _port_ret(weights: dict[str, float]) -> pd.Series:
        rows: list[float] = []
        idx: list[pd.Timestamp] = []
        for dt in sorted(panel["date"].unique()):
            sub = panel[panel["date"] == dt].set_index("ticker")
            if sub.empty or "ret_1m" not in sub.columns:
                continue
            r = 0.0
            for t, w in weights.items():
                if t in sub.index:
                    r += w * float(sub.loc[t, "ret_1m"])
            rows.append(r)
            idx.append(pd.Timestamp(dt))
        return pd.Series(rows, index=idx)

    w_eq = {t: 1.0 / len(tickers) for t in tickers}
    w_spy = {t: (1.0 if t == spy else 0.0) for t in tickers}
    w_6040 = {t: 0.0 for t in tickers}
    w_6040[spy] = 0.6
    w_6040[agg] = 0.4

    return {
        "spy": _port_ret(w_spy),
        "eq_weight": _port_ret(w_eq),
        "6040": _port_ret(w_6040),
    }
