"""
TAA Strategy candidate for ShinkaEvolve optimisation.

The EVOLVE-BLOCK contains the monthly weight allocation logic.
Everything outside the block (data loading, cost model, metric calc) is fixed.

Targets:  CAGR > 15%,  Calmar ratio > 1.0
Universe: 12 ETFs via spliced total-return data (2006-04-2026)

NOTE: MAX_WEIGHT = 1.0 — no per-asset concentration cap is enforced.
The strategy may concentrate fully in a single asset if desired.
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
MAX_WEIGHT = 1.0     # no per-asset cap — strategy controls concentration directly
MIN_SHY = 0.05       # floor on short-term treasury (cash-like)


# ---------------------------------------------------------------------------
# EVOLVE-BLOCK-START
# ---------------------------------------------------------------------------
def get_monthly_weights(
    monthly_rets: pd.DataFrame,
    current_date: pd.Timestamp,
) -> np.ndarray:
    """
    Implements a layered multi-factor TAA strategy with highly dynamic parameters
    including granular volatility-adaptive lookbacks and momentum blending.

    Seed: best strategy from prior evolution run (gen 210, score 0.879).
    MAX_WEIGHT constraint removed — strategy may now concentrate freely.
    """
    # --- 1. Parameters ---
    MIN_HISTORY_REQUIRED = 12
    SMA_PERIOD = 10

    # SPY Volatility Thresholds for Regime, Momentum Blending, and Volatility Lookback
    SPY_VOL_THRESHOLD_LOW = 0.10      # Threshold for "low spy vol" signal
    SPY_VOL_THRESHOLD_VERY_LOW = 0.07 # For ultra-aggressive mom blend / longest vol lookback
    SPY_VOL_THRESHOLD_MODERATE = 0.15 # For balancing mom blend / standard vol lookback
    SPY_VOL_THRESHOLD_HIGH = 0.20     # For defensive mom blend / shorter vol lookback

    # Volatility Lookback Periods (dynamic, in months)
    VOL_LOOKBACK_FAST = 3
    VOL_LOOKBACK_NORMAL = 6
    VOL_LOOKBACK_SLOW = 9
    VOL_LOOKBACK_VERY_SLOW = 12

    # Momentum Threshold for "Super Bull" sub-regime
    SPY_STRONG_MOM_3M_THRESHOLD = 0.04 # 4% return in 3 months (approx 16% annualized)

    # Market Breadth Signal Parameters
    BREADTH_CANARIES = ["SPY US Equity", "QQQ US Equity", "EFA US Equity", "EEM US Equity", "HYG US Equity"]
    BREADTH_THRESHOLD_COUNT = 3 # At least 3 of 5 must be bullish for a strong breadth signal

    # --- 2. Handle insufficient history ---
    if len(monthly_rets) < MIN_HISTORY_REQUIRED:
        return np.ones(N) / N

    # --- 3. Define key tickers and indices ---
    spy_ticker = "SPY US Equity"
    tlt_ticker = "TLT US Equity"
    shy_ticker = "SHY US Equity"
    shy_idx = TICKERS.index(shy_ticker)

    # --- 4. Calculate Core Signals for All Assets ---
    prices = (1 + monthly_rets).cumprod()

    # Individual Asset Trend Filter (Price > 10m SMA for each asset)
    is_asset_bull = pd.Series(False, index=TICKERS)
    if len(prices) >= SMA_PERIOD:
        sma10 = prices.rolling(window=SMA_PERIOD).mean().iloc[-1]
        is_asset_bull = prices.iloc[-1] > sma10

    # Raw Momentum Scores
    mom3 = ((1 + monthly_rets.iloc[-3:]).prod() - 1).fillna(0.0)
    mom6 = ((1 + monthly_rets.iloc[-6:]).prod() - 1).fillna(0.0)
    mom12 = ((1 + monthly_rets.iloc[-12:]).prod() - 1).fillna(0.0)

    # SPY 3-month annualized volatility for dynamic blending and regime detection
    spy_vol_3m_ann = monthly_rets[spy_ticker].iloc[-3:].std(ddof=1) * np.sqrt(12)

    # --- 5. Calculate Regime Signals ---
    is_spy_bull = is_asset_bull.get(spy_ticker, False)
    is_intermarket_risk_on = mom6.get(spy_ticker, 0) > mom6.get(tlt_ticker, 0) and mom6.get(spy_ticker, 0) > 0
    is_low_spy_vol_signal = spy_vol_3m_ann < SPY_VOL_THRESHOLD_MODERATE # Used for risk_score

    # Market Breadth Signal
    strong_breadth_count = sum(1 for ticker in BREADTH_CANARIES if is_asset_bull.get(ticker, False))
    is_market_breadth_strong = strong_breadth_count >= BREADTH_THRESHOLD_COUNT

    # --- 6. Determine Regime Parameters ---
    # Primary Crisis Filter: Severe Downtrend (long-term momentum)
    is_critical_crisis = not is_spy_bull and mom12.get(spy_ticker, 0) < -0.05

    w = np.zeros(N)
    shy_floor = MIN_SHY # Default value, will be overridden
    current_weighting_mode = '' # Will be 'inv_vol' or 'risk_adj_mom'
    mom_weights = [1/3, 1/3, 1/3] # Default, will be overridden
    current_vol_lookback = VOL_LOOKBACK_NORMAL # Default, will be overridden

    if is_critical_crisis:
        # State: Critical Crisis - Heavily defensive with a high SHY floor
        n_select = 2
        shy_floor = 0.85
        candidate_universe = ["AGG US Equity", "TLT US Equity", "LQD US Equity", "GLD US Equity", shy_ticker]
        mom_weights = [0.1, 0.2, 0.7] # Very long-term focused momentum blending
        current_weighting_mode = 'inv_vol'
        current_vol_lookback = VOL_LOOKBACK_FAST # Fast reaction to current risk
    else:
        # Secondary Risk Score (0-4) for graded, non-critical crisis environments
        risk_score = (
            int(is_spy_bull) +
            int(is_intermarket_risk_on) +
            int(is_low_spy_vol_signal) +
            int(is_market_breadth_strong)
        )

        # Pre-Crisis Warning Logic: A specific defensive stage when risk_score is low but not yet a full critical crash.
        is_pre_crisis_warning = (
            risk_score <= 1 and # Low risk score (0 or 1)
            not is_spy_bull and # SPY below its 10m SMA
            mom6.get(spy_ticker, 0) < 0 and # SPY 6m momentum is negative
            mom12.get(spy_ticker, 0) >= -0.05 # But not yet critical crisis (m12 < -0.05)
        )

        if is_pre_crisis_warning:
            # State: Pre-Crisis Warning - Moderate defensive stance
            n_select = 2
            shy_floor = 0.50 # Moderate SHY floor
            candidate_universe = ["SPY US Equity", "AGG US Equity", "TLT US Equity", "LQD US Equity", "GLD US Equity", shy_ticker]
            mom_weights = [0.2, 0.4, 0.4] # Balanced defensive momentum blend
            current_weighting_mode = 'inv_vol'
            current_vol_lookback = VOL_LOOKBACK_FAST # React faster in warning state
        elif risk_score == 4:  # Max Risk-On (Highest Bull)
            # Determine concentration based on 12m trend strength
            spy_mom12 = mom12.get(spy_ticker, 0)
            if spy_mom12 > 0.25:
                n_select = 5 # Broaden in very strong, established trends
            elif spy_mom12 > 0.15:
                n_select = 4 # Standard strong trend
            else:
                n_select = 3 # Concentrate in weaker bull markets

            # Define "Super Bull" sub-regime for even more aggressive stance
            is_super_bull = (spy_vol_3m_ann < SPY_VOL_THRESHOLD_VERY_LOW and
                             mom3.get(spy_ticker, 0) > SPY_STRONG_MOM_3M_THRESHOLD)

            if is_super_bull:
                shy_floor = MIN_SHY
                candidate_universe = ["QQQ US Equity", "SPY US Equity", "EFA US Equity", "EEM US Equity", "HYG US Equity", "IYR US Equity", "DBC US Equity"]
                mom_weights = [0.9, 0.07, 0.03] # Extremely aggressive short-term momentum
                current_vol_lookback = VOL_LOOKBACK_VERY_SLOW # Very stable, use longest vol lookback
            else: # Standard Max Risk-On (still aggressive)
                shy_floor = MIN_SHY
                candidate_universe = ["QQQ US Equity", "SPY US Equity", "EFA US Equity", "EEM US Equity", "HYG US Equity", "IYR US Equity", "DBC US Equity", "GLD US Equity"]
                if spy_vol_3m_ann < SPY_VOL_THRESHOLD_VERY_LOW:
                    mom_weights = [0.8, 0.15, 0.05]
                    current_vol_lookback = VOL_LOOKBACK_SLOW
                elif spy_vol_3m_ann < SPY_VOL_THRESHOLD_LOW:
                    mom_weights = [0.7, 0.2, 0.1]
                    current_vol_lookback = VOL_LOOKBACK_SLOW
                else:
                    mom_weights = [0.6, 0.3, 0.1]
                    current_vol_lookback = VOL_LOOKBACK_NORMAL
            current_weighting_mode = 'risk_adj_mom'
        elif risk_score == 3:  # Strong Risk-On
            n_select = 4
            shy_floor = MIN_SHY
            candidate_universe = [t for t in TICKERS if t not in ["AGG US Equity", "TLT US Equity", "LQD US Equity", shy_ticker]]
            if spy_vol_3m_ann < SPY_VOL_THRESHOLD_LOW:
                mom_weights = [0.6, 0.3, 0.1] # Aggressive
                current_vol_lookback = VOL_LOOKBACK_SLOW
            else:
                mom_weights = [0.5, 0.3, 0.2] # Moderately aggressive
                current_vol_lookback = VOL_LOOKBACK_NORMAL
            current_weighting_mode = 'risk_adj_mom'
        elif risk_score == 2:  # Moderate
            n_select = 3
            shy_floor = 0.15
            candidate_universe = TICKERS
            if spy_vol_3m_ann < SPY_VOL_THRESHOLD_MODERATE:
                mom_weights = [0.4, 0.4, 0.2] # Balanced
                current_vol_lookback = VOL_LOOKBACK_NORMAL
            else:
                mom_weights = [0.3, 0.4, 0.3] # Slightly defensive
                current_vol_lookback = VOL_LOOKBACK_FAST
            current_weighting_mode = 'risk_adj_mom'
        elif risk_score == 1:  # Cautious - Use Inv Vol for defensive posture
            n_select = 2
            shy_floor = 0.40
            candidate_universe = ["SPY US Equity", "AGG US Equity", "TLT US Equity", "LQD US Equity", "GLD US Equity", shy_ticker]
            if spy_vol_3m_ann > SPY_VOL_THRESHOLD_HIGH:
                mom_weights = [0.2, 0.3, 0.5] # Defensive, long-term
                current_vol_lookback = VOL_LOOKBACK_FAST
            else:
                mom_weights = [0.3, 0.4, 0.3] # Balanced defensive
                current_vol_lookback = VOL_LOOKBACK_NORMAL
            current_weighting_mode = 'inv_vol'
        else:  # risk_score == 0 (Max Defensive, *not* pre-crisis)
            n_select = 2
            shy_floor = 0.70
            candidate_universe = ["AGG US Equity", "TLT US Equity", "LQD US Equity", "GLD US Equity", shy_ticker]
            mom_weights = [0.1, 0.2, 0.7] # Very defensive, very long-term
            current_weighting_mode = 'inv_vol'
            current_vol_lookback = VOL_LOOKBACK_FAST

    # --- 7. Calculate Blended Momentum & Volatility ---
    blended_mom = (mom_weights[0] * mom3 + mom_weights[1] * mom6 + mom_weights[2] * mom12)

    # Use dynamically determined volatility lookback for calculating asset volatilities
    vols = monthly_rets.iloc[-current_vol_lookback:].std(ddof=1) * np.sqrt(12)
    vols = vols.clip(lower=0.01).fillna(vols.mean()).fillna(0.01)

    # --- 7a. Correlation Adjustment for Hedging in Defensive Regimes ---
    selection_scores = blended_mom.copy()
    is_defensive_boost_regime = is_critical_crisis or is_pre_crisis_warning or (risk_score <= 1)

    if is_defensive_boost_regime and len(monthly_rets) >= 12:
        corrs_with_spy = monthly_rets.iloc[-12:].corr(numeric_only=True).get(spy_ticker, pd.Series(0.5, index=TICKERS))
        for asset, corr in corrs_with_spy.items():
            if asset not in candidate_universe:
                continue
            if corr < -0.2:  # Bonus for significantly negatively correlated assets
                bonus_factor = 1.0 - (corr * 2.0)
                selection_scores[asset] *= bonus_factor
            elif corr > 0.5 and asset != shy_ticker:  # Penalize highly positively correlated assets
                penalty_factor = 1.0 + (corr - 0.5) * 1.5
                selection_scores[asset] /= penalty_factor

    # --- 8. Asset Selection based on a Multi-Filter Process ---
    eligible_scores = selection_scores.loc[candidate_universe]

    if current_weighting_mode == 'risk_adj_mom':
        eligible_scores = eligible_scores[eligible_scores > 0]

    eligible_scores = eligible_scores[is_asset_bull[eligible_scores.index] | (eligible_scores.index == shy_ticker)]

    if eligible_scores.empty:
        w[shy_idx] = 1.0; return w

    selected_tickers = eligible_scores.nlargest(n_select).index

    # --- 9. Weighting and Allocation ---
    if current_weighting_mode == 'inv_vol':
        inv_vols = 1.0 / vols.loc[selected_tickers]
        if inv_vols.sum() < 1e-9:
            w[shy_idx] = 1.0; return w
        raw_weights = inv_vols / inv_vols.sum()
    else: # 'risk_adj_mom'
        risk_adj_scores = (blended_mom.loc[selected_tickers] / vols.loc[selected_tickers]).clip(lower=0)
        if risk_adj_scores.sum() < 1e-9:
            w[shy_idx] = 1.0; return w
        raw_weights = risk_adj_scores / risk_adj_scores.sum()

    # --- 10. Enforce SHY Floor and Finalize Weights ---
    current_shy_weight = raw_weights.get(shy_ticker, 0.0)

    if current_shy_weight < shy_floor:
        w[shy_idx] = shy_floor
        remaining_capital = 1.0 - shy_floor

        other_assets_weights = raw_weights.drop(shy_ticker, errors='ignore')
        if not other_assets_weights.empty and remaining_capital > 0 and other_assets_weights.sum() > 1e-9:
            other_assets_weights /= other_assets_weights.sum()
            for ticker, weight in other_assets_weights.items():
                w[TICKERS.index(ticker)] = weight * remaining_capital
        elif remaining_capital > 0:
            w[shy_idx] = 1.0
    else:
        for ticker, weight in raw_weights.items():
            w[TICKERS.index(ticker)] = weight

    # Final normalization
    if w.sum() < 1e-9:
        w = np.zeros(N)
        w[shy_idx] = 1.0
    elif abs(w.sum() - 1.0) > 1e-9:
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
