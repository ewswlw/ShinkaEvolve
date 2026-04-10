"""Walk-forward OOS, PSR, DSR."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from taa import config

logger = logging.getLogger(__name__)


def probabilistic_sharpe_ratio(
    returns: pd.Series,
    benchmark_sr: float = 0.0,
    periods_per_year: int = 12,
) -> dict[str, float]:
    """PSR vs benchmark Sharpe (monthly returns default)."""
    r = returns.dropna().astype(float)
    n = len(r)
    if n < 10:
        return {"psr": 0.0, "sharpe": 0.0}
    mu = float(r.mean())
    sigma = float(r.std(ddof=1)) + 1e-12
    sr = mu / sigma * np.sqrt(periods_per_year)
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))
    z = (sr - benchmark_sr) * np.sqrt(n - 1) / np.sqrt(1.0 - skew * sr + (kurt - 1) / 4.0 * sr**2 + 1e-12)
    psr = float(stats.norm.cdf(z))
    return {"psr": psr, "sharpe": float(sr), "n": float(n)}


def deflated_sharpe_ratio(
    observed_sr: float,
    n_trials: int,
    n_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Harvey-Liu deflated Sharpe approximation."""
    if n_observations < 2:
        return 0.0
    variance_sr = (
        (1.0 - skewness * observed_sr + (kurtosis - 1) / 4.0 * observed_sr**2)
        / (n_observations - 1)
        + 1e-12
    )
    e_max = expected_max_sharpe(n_trials)
    dsr = (observed_sr - e_max) / np.sqrt(variance_sr + 1e-12)
    return float(stats.norm.cdf(dsr))


def expected_max_sharpe(n_trials: int, mean_sharpe: float = 0.0, std_sharpe: float = 1.0) -> float:
    """Expected max SR under multiple testing (approximate)."""
    if n_trials <= 1:
        return mean_sharpe
    return float(mean_sharpe + std_sharpe * (1.0 - np.exp(-np.log(n_trials))))


def rule_based_weights(
    regime: int,
    tickers: list[str],
) -> np.ndarray:
    """Deterministic tilt by regime (no ML)."""
    n = len(tickers)
    w = np.zeros(n)
    eq = ["SPY US Equity", "QQQ US Equity", "EEM US Equity", "LQD US Equity", "HYG US Equity"]
    defen = ["TLT US Equity", "AGG US Equity", "GLD US Equity"]
    crisis = ["SHY US Equity", "TLT US Equity", "GLD US Equity"]

    def _alloc(names: list[str], mass: float) -> None:
        sub = [i for i, t in enumerate(tickers) if t in names]
        if not sub:
            return
        for i in sub:
            w[i] += mass / len(sub)

    if regime == 0:
        _alloc(eq, 0.7)
        _alloc(["AGG US Equity"], 0.15)
        _alloc(["SHY US Equity"], 0.15)
    elif regime == 1:
        _alloc(defen, 0.6)
        _alloc(["SPY US Equity"], 0.2)
        _alloc(["SHY US Equity"], 0.2)
    else:
        _alloc(crisis, 0.85)
        _alloc(["GLD US Equity"], 0.15)
    if w.sum() <= 0:
        w[:] = 1.0 / n
    else:
        w = w / w.sum()
    return w


def blend_regime_weights(
    regime_probs: np.ndarray,
    tickers: list[str],
) -> np.ndarray:
    """w = sum_k p_k * w_k."""
    w = np.zeros(len(tickers))
    for k in range(3):
        w += regime_probs[k] * rule_based_weights(k, tickers)
    return w / w.sum()


def walk_forward_taa(
    panel: pd.DataFrame,
    regime_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Expanding-window walk-forward: monthly weights from blended rule weights.
    Returns DataFrame of monthly portfolio returns (OOS concatenated).
    """
    tickers = config.ETF_TICKERS
    r = regime_df.sort_index()
    r.index = pd.to_datetime(r.index)
    dates = sorted(pd.to_datetime(panel["date"].unique()))
    rets: list[float] = []
    out_dates: list[pd.Timestamp] = []

    for i, dt in enumerate(dates[:-1]):
        nxt = dates[i + 1]
        ts = pd.Timestamp(dt)
        if ts in r.index:
            row = r.loc[ts]
        else:
            pos = r.index.searchsorted(ts) - 1
            if pos < 0:
                continue
            row = r.iloc[pos]
        probs = np.array(
            [
                float(row["hmm_expansion_prob"]),
                float(row["hmm_contraction_prob"]),
                float(row["hmm_crisis_prob"]),
            ],
        )
        w = blend_regime_weights(probs, tickers)
        # Realized return next month: weighted sum of ETF ret_1m at nxt
        sub = panel[panel["date"] == nxt].set_index("ticker")
        if sub.empty:
            continue
        port_r = 0.0
        for j, t in enumerate(tickers):
            if t in sub.index and "ret_1m" in sub.columns:
                port_r += w[j] * float(sub.loc[t, "ret_1m"])
        rets.append(port_r)
        out_dates.append(pd.Timestamp(nxt))

    return pd.DataFrame({"date": out_dates, "ret": rets})


def evaluate_walk_forward(
    wf: pd.DataFrame,
    n_trials: int = 100,
) -> dict[str, Any]:
    """PSR / DSR gates on OOS monthly returns."""
    r = wf["ret"].dropna()
    psr_out = probabilistic_sharpe_ratio(r, benchmark_sr=0.0)
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))
    dsr = deflated_sharpe_ratio(
        psr_out["sharpe"],
        n_trials=n_trials,
        n_observations=len(r),
        skewness=skew,
        kurtosis=kurt,
    )
    return {
        "psr": psr_out["psr"],
        "dsr": dsr,
        "sharpe": psr_out["sharpe"],
        "n": len(r),
        "pass_psr": psr_out["psr"] > 0.95,
        "pass_dsr": dsr > 0.95,
    }
