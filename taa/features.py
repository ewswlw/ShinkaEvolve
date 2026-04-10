"""Feature engineering: monthly panel and macro features (F01–F55)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from taa import config
from taa.utils import expanding_zscore, monthly_align

logger = logging.getLogger(__name__)


def flatten_bdh(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns to lowercase single names."""
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        new_cols = []
        for c in df.columns:
            if isinstance(c, tuple) and len(c) > 1:
                new_cols.append(f"{c[0]}_{c[1]}".lower().replace(" ", "_"))
            else:
                new_cols.append(str(c).lower().replace(" ", "_"))
        out = df.copy()
        out.columns = new_cols
        return out
    out = df.copy()
    out.columns = [str(c).lower().replace(" ", "_") for c in out.columns]
    return out


def _daily_returns(levels: pd.Series) -> pd.Series:
    return levels.astype(float).pct_change()


def compute_etf_features(
    spliced_daily: pd.DataFrame,
) -> pd.DataFrame:
    """
    F01–F12 per ETF at monthly frequency. Returns long-format DataFrame:
    columns: date, ticker, ret_1m, ret_3m, ... plus feature names.
    """
    rows: list[dict[str, Any]] = []
    monthly_last = spliced_daily.resample("ME").last()

    for ticker in spliced_daily.columns:
        lev = spliced_daily[ticker].dropna()
        if lev.empty:
            continue
        mlev = monthly_last[ticker].dropna()
        mret = mlev.pct_change()

        vol_daily = _daily_returns(lev)
        for dt in mlev.index:
            if dt < pd.Timestamp(config.BACKTEST_START):
                continue
            loc = mlev.index.get_loc(dt)
            def mom(k: int) -> float:
                if loc - k < 0:
                    return float("nan")
                a = float(mlev.iloc[loc])
                b = float(mlev.iloc[loc - k])
                return a / b - 1.0 if b else float("nan")

            ret_1m = float(mret.loc[dt]) if dt in mret.index else float("nan")
            ret_3m, ret_6m, ret_12m = mom(3), mom(6), mom(12)
            ret_12m_1m = (
                float(mlev.iloc[loc - 1] / mlev.iloc[loc - 12] - 1.0)
                if loc >= 12
                else float("nan")
            )

            d0 = lev.index[lev.index <= dt]
            d0 = d0[-252:] if len(d0) > 252 else d0
            d3 = lev.index[lev.index <= dt]
            d3 = d3[-63:] if len(d3) > 63 else d3
            v3 = (
                float(vol_daily.loc[d3].std() * np.sqrt(252))
                if len(d3) > 5
                else float("nan")
            )
            v12 = (
                float(vol_daily.loc[d0].std() * np.sqrt(252))
                if len(d0) > 20
                else float("nan")
            )
            vol_ratio = v3 / v12 if v12 and v12 == v12 and v12 != 0 else float("nan")

            # max drawdown 126 trading days
            win = lev.loc[:dt].iloc[-126:]
            if len(win) > 5:
                peak = win.cummax()
                dd = (win / peak - 1.0).min()
                max_dd_6m = float(dd)
            else:
                max_dd_6m = float("nan")

            sma10 = float(mlev.loc[:dt].rolling(10, min_periods=3).mean().iloc[-1])
            sma3 = float(mlev.loc[:dt].rolling(3, min_periods=2).mean().iloc[-1])
            px = float(mlev.loc[dt])
            price_sma10 = px / sma10 if sma10 else float("nan")
            price_sma3 = px / sma3 if sma3 else float("nan")

            dv = spliced_daily[ticker].reindex(lev.index)
            volu = dv  # placeholder if no volume column
            if hasattr(spliced_daily, "columns"):
                pass
            # Volume ratio: need volume series — skip if only levels in frame
            vol_ratio_feat = float("nan")

            rows.append(
                {
                    "date": dt,
                    "ticker": ticker,
                    "ret_1m": ret_1m,
                    "ret_3m": ret_3m,
                    "ret_6m": ret_6m,
                    "ret_12m": ret_12m,
                    "ret_12m_1m": ret_12m_1m,
                    "vol_3m": v3,
                    "vol_12m": v12,
                    "vol_ratio": vol_ratio,
                    "max_dd_6m": max_dd_6m,
                    "price_sma_ratio_10m": price_sma10,
                    "price_sma_ratio_3m": price_sma3,
                    "volume_sma_ratio": vol_ratio_feat,
                },
            )

    return pd.DataFrame(rows)


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Match column case-insensitively by substring (Bloomberg mnemonic or partial)."""
    key = name.lower().replace(" ", "_")
    for c in df.columns:
        cn = str(c).lower().replace(" ", "_")
        if key in cn or (len(key) >= 4 and key[:4] in cn):
            return pd.to_numeric(df[c], errors="coerce")
    for tok in key.split("_"):
        if len(tok) < 4:
            continue
        for c in df.columns:
            if tok in str(c).lower():
                return pd.to_numeric(df[c], errors="coerce")
    raise KeyError(f"{name} in {list(df.columns)}")


def compute_yield_curve_features(yc: pd.DataFrame) -> pd.DataFrame:
    yc = flatten_bdh(yc)
    m = monthly_align(yc)
    out = pd.DataFrame(index=m.index)
    out["ust_2y"] = _col(m, "usgg2yr")
    out["ust_10y"] = _col(m, "usgg10yr")
    out["slope_2s10s"] = out["ust_10y"] - _col(m, "usgg2yr")
    out["slope_3m10y"] = out["ust_10y"] - _col(m, "usgg3m")
    out["curve_curvature"] = 2 * _col(m, "usgg5yr") - _col(m, "usgg2yr") - out["ust_10y"]
    out["slope_2s10s_chg_3m"] = out["slope_2s10s"] - out["slope_2s10s"].shift(3)
    out["slope_2s10s_zscore"] = expanding_zscore(out["slope_2s10s"])
    return out


def compute_credit_features(cr: pd.DataFrame) -> pd.DataFrame:
    cr = flatten_bdh(cr)
    m = monthly_align(cr)
    out = pd.DataFrame(index=m.index)
    out["hy_oas"] = _col(m, "lf98oas")
    out["ig_oas"] = _col(m, "luacoas")
    out["hy_oas_chg_1m"] = out["hy_oas"].diff(1)
    out["hy_oas_chg_3m"] = out["hy_oas"].diff(3)
    out["hy_oas_zscore"] = expanding_zscore(out["hy_oas"])
    out["credit_quality_spread"] = out["hy_oas"] - out["ig_oas"]
    return out


def compute_volatility_features(vol_df: pd.DataFrame) -> pd.DataFrame:
    vol_df = flatten_bdh(vol_df)
    m = monthly_align(vol_df)
    out = pd.DataFrame(index=m.index)
    out["vix_level"] = _col(m, "vix")
    out["vix_sma_ratio"] = out["vix_level"] / out["vix_level"].rolling(3).mean()
    out["vix_zscore"] = expanding_zscore(out["vix_level"])
    out["move_level"] = _col(m, "move")
    out["move_zscore"] = expanding_zscore(out["move_level"])
    return out


def compute_macro_features(macro: pd.DataFrame) -> pd.DataFrame:
    macro = flatten_bdh(macro)
    m = monthly_align(macro)
    out = pd.DataFrame(index=m.index)
    out["ism_pmi"] = _col(m, "napmpmi")
    out["ism_pmi_chg_3m"] = out["ism_pmi"].diff(3)
    out["ism_above_50"] = (out["ism_pmi"] > 50).astype(float)
    claims = _col(m, "injcjc")
    out["claims_4wma"] = claims.rolling(4, min_periods=1).mean()
    out["claims_yoy_chg"] = out["claims_4wma"] / out["claims_4wma"].shift(12) - 1.0
    out["consumer_conf"] = _col(m, "conccf")
    out["consumer_conf_chg_3m"] = out["consumer_conf"].diff(3)
    out["lei_chg"] = _col(m, "lei")
    return out


def compute_inflation_features(infl: pd.DataFrame) -> pd.DataFrame:
    infl = flatten_bdh(infl)
    m = monthly_align(infl)
    out = pd.DataFrame(index=m.index)
    out["cpi_yoy"] = _col(m, "cpi yoy")
    out["breakeven_10y"] = _col(m, "usggbe10")
    out["breakeven_chg_3m"] = out["breakeven_10y"].diff(3)
    return out


def compute_fed_features(fed: pd.DataFrame) -> pd.DataFrame:
    fed = flatten_bdh(fed)
    m = monthly_align(fed)
    out = pd.DataFrame(index=m.index)
    out["fed_funds"] = _col(m, "fdtr")
    out["fed_funds_chg_6m"] = out["fed_funds"].diff(6)
    return out


def compute_breadth_features(breadth: pd.DataFrame) -> pd.DataFrame:
    if breadth.empty or "pct_above_200dma" not in breadth.columns:
        return pd.DataFrame()
    m = breadth.copy()
    if not isinstance(m.index, pd.DatetimeIndex):
        m.index = pd.to_datetime(m.index)
    out = pd.DataFrame(index=m.index)
    out["pct_above_200dma"] = m["pct_above_200dma"]
    out["pct_above_200dma_chg_1m"] = out["pct_above_200dma"].diff(1)
    return out


def compute_valuation_features(val: pd.DataFrame, yc: pd.DataFrame) -> pd.DataFrame:
    val = flatten_bdh(val)
    yc = flatten_bdh(yc)
    m = monthly_align(val)
    out = pd.DataFrame(index=m.index)
    out["spx_pe"] = _col(m, "pe_ratio")
    out["spx_earnings_yield"] = _col(m, "earn_yld")
    ym = monthly_align(yc)
    ust10 = _col(ym, "usgg10yr")
    out["equity_risk_premium"] = out["spx_earnings_yield"] - ust10.reindex(out.index) / 100.0
    return out


def compute_liquidity_features_v2(
    liq: pd.DataFrame,
    fed: pd.DataFrame,
    yc: pd.DataFrame,
) -> pd.DataFrame:
    liq = flatten_bdh(liq)
    fed = flatten_bdh(fed)
    yc = flatten_bdh(yc)
    m = monthly_align(liq)
    fm = monthly_align(fed)
    ym = monthly_align(yc)
    out = pd.DataFrame(index=m.index)
    out["fin_conditions"] = _col(m, "bfcius")
    out["fin_conditions_chg_3m"] = out["fin_conditions"].diff(3)
    libor = _col(fm, "us0003m")
    tb = _col(ym, "usgg3m")
    out["ted_spread"] = libor.reindex(out.index) - tb.reindex(out.index)
    return out


def compute_cross_asset_features(etf_long: pd.DataFrame) -> pd.DataFrame:
    """F52–F55 from long-format ETF features."""
    p = etf_long.pivot(index="date", columns="ticker", values="ret_3m")
    p12 = etf_long.pivot(index="date", columns="ticker", values="ret_12m")
    out = pd.DataFrame(index=p.index)
    spy = config.SPY_TICKER
    agg = config.AGG_TICKER
    dbc = "DBC US Equity"
    gld = "GLD US Equity"
    if spy in p.columns and agg in p.columns:
        out["equity_bond_rel_mom_3m"] = p[spy] - p[agg]
    if spy in p12.columns and agg in p12.columns:
        out["equity_bond_rel_mom_12m"] = p12[spy] - p12[agg]
    if dbc in p.columns and spy in p.columns:
        out["commodity_equity_ratio_3m"] = p[dbc] - p[spy]
    if gld in p.columns and spy in p.columns:
        out["gold_equity_ratio_3m"] = p[gld] - p[spy]
    return out


def build_feature_panel(data: dict[str, Any]) -> pd.DataFrame:
    """Merge ETF long panel with macro time-series features on date."""
    spliced = data.get("spliced_levels")
    if spliced is None or spliced.empty:
        raise ValueError("spliced_levels required")

    etf_long = compute_etf_features(spliced)
    macro = data.get("macro", {})

    parts: list[pd.DataFrame] = []
    yc = macro.get("yield_curve", pd.DataFrame())
    if not yc.empty:
        parts.append(compute_yield_curve_features(yc))
    cr = macro.get("credit", pd.DataFrame())
    if not cr.empty:
        parts.append(compute_credit_features(cr))
    vo = macro.get("volatility", pd.DataFrame())
    if not vo.empty:
        parts.append(compute_volatility_features(vo))
    ml = macro.get("macro_leading", pd.DataFrame())
    if not ml.empty:
        parts.append(compute_macro_features(ml))
    inf = macro.get("inflation", pd.DataFrame())
    if not inf.empty:
        parts.append(compute_inflation_features(inf))
    fd = macro.get("fed", pd.DataFrame())
    if not fd.empty:
        parts.append(compute_fed_features(fd))
    lq = macro.get("liquidity", pd.DataFrame())
    if not lq.empty and not yc.empty:
        parts.append(compute_liquidity_features_v2(lq, fd, yc))

    val = data.get("valuation", pd.DataFrame())
    if not val.empty and not yc.empty:
        parts.append(compute_valuation_features(val, yc))

    br = data.get("breadth", pd.DataFrame())
    if not br.empty:
        parts.append(compute_breadth_features(br))

    macro_panel = pd.concat(parts, axis=1) if parts else pd.DataFrame()
    macro_panel = macro_panel.loc[:, ~macro_panel.columns.duplicated()]
    cross = compute_cross_asset_features(etf_long)
    macro_panel = macro_panel.join(cross, how="outer")

    mp = macro_panel.reset_index()
    if "index" in mp.columns:
        mp = mp.rename(columns={"index": "date"})
    elif mp.columns[0] != "date":
        mp = mp.rename(columns={mp.columns[0]: "date"})

    merged = etf_long.merge(mp, on="date", how="left")
    return merged.sort_values(["date", "ticker"])
