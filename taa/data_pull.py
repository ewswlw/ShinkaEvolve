"""Bloomberg data pulls, Parquet cache, ETF/proxy splice."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xbbg import blp

from taa import config
from taa.utils import bbg, compute_tracking_error, ensure_datetime_index

logger = logging.getLogger(__name__)


def _ensure_data_dir() -> Path:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    return config.DATA_DIR


def _cache_path(name: str) -> Path:
    return _ensure_data_dir() / config.CACHE_FILES.get(name, f"{name}.parquet")


def _today_str() -> str:
    return pd.Timestamp.today().strftime("%Y-%m-%d")


def _load_or_pull(
    cache_name: str,
    pull_fn: Any,
    reload: bool,
    offline: bool,
) -> pd.DataFrame:
    path = _cache_path(cache_name)
    if path.exists() and not reload:
        logger.info("Loading cache %s", path)
        return pd.read_parquet(path)
    if offline:
        raise FileNotFoundError(
            f"Offline mode: missing cache {path}. Run without --offline first.",
        )
    df = pull_fn()
    df.to_parquet(path, index=True)
    return df


def pull_etf_prices(
    reload: bool = False,
    offline: bool = False,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Daily ETF levels: TOT_RETURN_INDEX_GROSS_DVDS, PX_LAST, PX_VOLUME."""

    def _pull() -> pd.DataFrame:
        tickers = config.ETF_TICKERS
        flds = ["TOT_RETURN_INDEX_GROSS_DVDS", "PX_LAST", "PX_VOLUME"]
        raw = blp.bdh(
            tickers,
            flds,
            start_date=start or config.PULL_START,
            end_date=end or _today_str(),
            Per="D",
            Fill="P",
            adjust="all",
        )
        raw = ensure_datetime_index(raw)
        return raw

    return _load_or_pull("etf_prices", _pull, reload, offline)


def pull_index_proxies(
    reload: bool = False,
    offline: bool = False,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Daily index proxy levels (total return style indices use PX_LAST)."""

    def _pull() -> pd.DataFrame:
        tickers = config.PROXY_TICKERS
        raw = blp.bdh(
            tickers,
            "PX_LAST",
            start_date=start or config.PULL_START,
            end_date=end or _today_str(),
            Per="D",
            Fill="P",
        )
        raw = ensure_datetime_index(raw)
        return raw

    return _load_or_pull("index_proxies", _pull, reload, offline)


def _pull_macro_block(
    cache_key: str,
    tickers: list[str],
    reload: bool,
    offline: bool,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    def _pull() -> pd.DataFrame:
        raw = blp.bdh(
            tickers,
            "PX_LAST",
            start_date=start or config.PULL_START,
            end_date=end or _today_str(),
            Per="D",
            Fill="P",
        )
        return ensure_datetime_index(raw)

    return _load_or_pull(cache_key, _pull, reload, offline)


def pull_macro(
    reload: bool = False,
    offline: bool = False,
) -> dict[str, pd.DataFrame]:
    """All macro category panels."""
    key_to_cache: dict[str, str] = {
        "yield_curve": "macro_yield",
        "credit": "macro_credit",
        "volatility": "macro_vol",
        "macro_leading": "macro_leading",
        "inflation": "macro_inflation",
        "fed": "macro_fed",
        "liquidity": "macro_liquidity",
    }
    out: dict[str, pd.DataFrame] = {}
    for key, tickers in config.MACRO_TICKERS.items():
        cache_key = key_to_cache[key]
        out[key] = _pull_macro_block(cache_key, tickers, reload, offline)
    return out


def pull_valuation(
    reload: bool = False,
    offline: bool = False,
) -> pd.DataFrame:
    """SPX PE_RATIO and EARN_YLD via bdh (monthly alignment in features)."""

    def _pull() -> pd.DataFrame:
        raw = blp.bdh(
            "SPX Index",
            ["PE_RATIO", "EARN_YLD"],
            start_date=config.PULL_START,
            end_date=_today_str(),
            Per="D",
            Fill="P",
        )
        return ensure_datetime_index(raw)

    return _load_or_pull("valuation", _pull, reload, offline)


def pull_breadth(
    reload: bool = False,
    offline: bool = False,
) -> pd.DataFrame:
    """
    Monthly % of SPX members above 200-DMA (expensive: one-time bds + bdp batch).
    Cached as time series of pct_above_200dma.
    """
    path = _cache_path("breadth")
    if path.exists() and not reload:
        return pd.read_parquet(path)
    if offline:
        raise FileNotFoundError(f"Missing breadth cache {path}")

    members_df = blp.bds("SPX Index", "INDX_MEMBERS")
    if members_df.empty:
        logger.warning("INDX_MEMBERS empty; breadth series empty")
        empty = pd.DataFrame(columns=["pct_above_200dma"])
        empty.to_parquet(path)
        return empty

    col = "member_ticker_and_exchange_code"
    if col not in members_df.columns:
        col = members_df.columns[0]
    members = members_df[col].astype(str).str.strip().unique().tolist()
    # Batch bdp in chunks
    batch_size = 300
    rows: list[dict[str, Any]] = []
    for i in range(0, len(members), batch_size):
        batch = members[i : i + batch_size]
        ref = blp.bdp(batch, ["PX_LAST", "MOV_AVG_200D"])
        ref = ref.replace("N/A", np.nan)
        for t in ref.index:
            px = ref.loc[t, "px_last"] if "px_last" in ref.columns else np.nan
            ma = (
                ref.loc[t, "mov_avg_200d"]
                if "mov_avg_200d" in ref.columns
                else np.nan
            )
            try:
                px_f = float(px)
                ma_f = float(ma)
            except (TypeError, ValueError):
                continue
            rows.append(
                {
                    "ticker": t,
                    "above": 1.0 if px_f > ma_f else 0.0,
                },
            )
    if not rows:
        out = pd.DataFrame(columns=["pct_above_200dma"])
        out.to_parquet(path)
        return out

    pct = float(np.mean([r["above"] for r in rows]))
    # Single snapshot — historical breadth needs point-in-time history (simplified stub)
    ts = pd.DataFrame(
        {"pct_above_200dma": [pct]},
        index=[pd.Timestamp.today().normalize()],
    )
    ts.to_parquet(path)
    meta_path = config.DATA_DIR / "breadth_meta.json"
    meta_path.write_text(
        json.dumps(
            {"n_members": len(members), "snapshot_date": _today_str()},
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.warning(
        "Breadth is a single snapshot; for full history use a precomputed index or "
        "extend pull_breadth with historical member panels.",
    )
    return ts


def _level_to_returns(levels: pd.Series) -> pd.Series:
    return levels.astype(float).pct_change().dropna()


def _get_etf_tr_series(etf_df: pd.DataFrame, etf: str) -> pd.Series:
    """Extract total-return level series for one ETF from bdh MultiIndex frame."""
    if isinstance(etf_df.columns, pd.MultiIndex):
        for c in etf_df.columns:
            if c[0] != etf:
                continue
            name = str(c[1]).lower()
            if "tot_return" in name or name == "px_last":
                return etf_df[c].astype(float)
        # fallback: first column for this ticker
        sub = etf_df.loc[:, etf_df.columns.get_level_values(0) == etf]
        if sub.shape[1]:
            return sub.iloc[:, 0].astype(float)
    elif etf in etf_df.columns:
        return etf_df[etf].astype(float)
    raise KeyError(f"No column for ETF {etf}")


def _get_proxy_series(proxy_df: pd.DataFrame, proxy_tik: str) -> pd.Series:
    if isinstance(proxy_df.columns, pd.MultiIndex):
        lev1 = proxy_df.columns.get_level_values(1)
        if (lev1 == "px_last").any():
            sub = proxy_df.xs("px_last", axis=1, level=1)
        else:
            sub = proxy_df.copy()
            sub.columns = sub.columns.get_level_values(0)
        if proxy_tik in sub.columns:
            return sub[proxy_tik].astype(float)
    if proxy_tik in proxy_df.columns:
        return proxy_df[proxy_tik].astype(float)
    raise KeyError(f"No column for proxy {proxy_tik}")


def splice_etf_with_proxy(
    etf_df: pd.DataFrame,
    proxy_df: pd.DataFrame,
    te_threshold_bps: float = 50.0,
) -> pd.DataFrame:
    """
    Build continuous total-return level series per ETF using proxy before inception.

    After inception date, ETF levels are used; before, proxy levels are scaled to
    match at the splice point.
    """
    spliced_levels: dict[str, pd.Series] = {}
    te_report: dict[str, float] = {}

    for etf, meta in config.ETF_UNIVERSE.items():
        proxy_tik = meta["proxy"]
        inception = pd.Timestamp(meta["inception"])

        try:
            e = _get_etf_tr_series(etf_df, etf).sort_index()
            p = _get_proxy_series(proxy_df, proxy_tik).sort_index()
        except KeyError as err:
            logger.warning("Splice skip %s: %s", etf, err)
            continue

        e = e[~e.index.duplicated(keep="last")]
        p = p[~p.index.duplicated(keep="last")]

        # Tracking error in overlap (post-inception) for reporting
        overlap_idx = e.index.intersection(p.index)
        overlap_idx = overlap_idx[overlap_idx >= inception]
        if len(overlap_idx) > 20:
            re = _level_to_returns(e.loc[overlap_idx])
            rp = _level_to_returns(p.loc[overlap_idx])
            te = compute_tracking_error(re, rp, annualize=True)
            te_report[etf] = float(te) if not np.isnan(te) else 0.0
            _ = te_threshold_bps  # reserved for haircut extension

        # Proxy segment strictly before inception; ETF from inception onward
        p_pre = p.loc[p.index < inception]
        e_post = e.loc[e.index >= inception]
        if e_post.empty:
            logger.warning("No ETF data post-inception for %s", etf)
            continue
        if p_pre.empty:
            combined = e_post
        else:
            anchor_e = float(e_post.iloc[0])
            anchor_p = float(p_pre.loc[: inception].iloc[-1])
            if anchor_p <= 0 or anchor_e <= 0:
                combined = e_post
            else:
                scale = anchor_e / anchor_p
                p_scaled = p_pre * scale
                # Drop duplicate inception timestamp if present
                combined = pd.concat([p_scaled, e_post]).sort_index()
                combined = combined[~combined.index.duplicated(keep="last")]

        spliced_levels[etf] = combined

    out = pd.DataFrame(spliced_levels)
    out = ensure_datetime_index(out)
    meta_path = config.DATA_DIR / "splice_te_report.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(te_report, indent=2), encoding="utf-8")
    return out


def pull_all(
    reload: bool = False,
    offline: bool = False,
) -> dict[str, Any]:
    """Pull ETF, proxies, macro blocks, valuation, breadth; splice returns."""
    _ensure_data_dir()
    etf = pull_etf_prices(reload=reload, offline=offline)
    proxies = pull_index_proxies(reload=reload, offline=offline)
    macro = pull_macro(reload=reload, offline=offline)
    val = pull_valuation(reload=reload, offline=offline)
    try:
        breadth = pull_breadth(reload=reload, offline=offline)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Breadth pull failed: %s", exc)
        breadth = pd.DataFrame()

    spliced = splice_etf_with_proxy(etf, proxies)
    spath = _cache_path("spliced_returns")
    # Store spliced levels
    spliced.to_parquet(spath)

    return {
        "etf_prices": etf,
        "index_proxies": proxies,
        "macro": macro,
        "valuation": val,
        "breadth": breadth,
        "spliced_levels": spliced,
    }
