"""Predictability gate and |t|>3 factor screening."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from taa import config

logger = logging.getLogger(__name__)


def _hurst_rs(ts: np.ndarray) -> float:
    """Simplified Hurst via R/S."""
    ts = np.asarray(ts, dtype=float)
    ts = ts[~np.isnan(ts)]
    if len(ts) < 32:
        return 0.5
    lags = range(2, min(len(ts) // 2, 100))
    rs_vals: list[float] = []
    for lag in lags:
        n = len(ts) // lag * lag
        seg = ts[:n].reshape(-1, lag)
        mean = seg.mean(axis=1, keepdims=True)
        dev = seg - mean
        cum = np.cumsum(dev, axis=1)
        r = cum.max(axis=1) - cum.min(axis=1)
        s = seg.std(axis=1) + 1e-12
        rs = np.mean(r / s)
        rs_vals.append(float(np.log(rs)))
    if len(rs_vals) < 2:
        return 0.5
    x = np.log(list(lags[: len(rs_vals)]))
    y = np.array(rs_vals)
    slope = np.polyfit(x, y, 1)[0]
    return float(np.clip(slope, 0, 1))


def predictability_score(returns: pd.Series) -> dict[str, Any]:
    """Composite 0–100 score from Hurst, VR, first-order autocorr."""
    r = returns.dropna().astype(float)
    if len(r) < 60:
        return {"score": 0.0, "recommendation": "STOP", "components": {}}

    h = _hurst_rs(r.values)
    # Variance ratio ~1 if random walk
    vr = float(r.var() / (r.diff().var() + 1e-12))
    ac = float(r.autocorr(lag=1)) if len(r) > 2 else 0.0

    score = 50.0 * (h - 0.45) / 0.2 + 20.0 * (1.0 - min(abs(np.log(vr + 1e-12)), 2.0)) + 30.0 * abs(ac)
    score = float(np.clip(score, 0.0, 100.0))

    if score < 20:
        rec = "STOP"
    elif score < 40:
        rec = "CAUTION"
    else:
        rec = "PROCEED"

    return {
        "score": score,
        "recommendation": rec,
        "components": {"hurst": h, "variance_ratio": vr, "acf1": ac},
    }


def screen_factors(
    factors: pd.DataFrame,
    forward_returns: pd.Series,
    t_threshold: float = 3.0,
) -> dict[str, Any]:
    """Univariate rank IC vs forward returns; |t| on mean IC."""
    details: list[dict[str, Any]] = []
    passed: list[str] = []
    failed: list[str] = []

    for col in factors.columns:
        if col in ("date", "ticker"):
            continue
        s = pd.to_numeric(factors[col], errors="coerce")
        aligned = pd.DataFrame({"f": s, "y": forward_returns}).dropna()
        if len(aligned) < 30:
            failed.append(col)
            details.append({"feature": col, "n": len(aligned), "t_stat": 0.0})
            continue
        ic = aligned["f"].corr(aligned["y"], method="spearman")
        n = len(aligned)
        if ic is None or np.isnan(ic) or abs(ic) >= 1.0:
            t_stat = 0.0
        else:
            t_stat = float(ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2))
        details.append({"feature": col, "n": n, "ic": float(ic), "t_stat": t_stat})
        if abs(t_stat) > t_threshold:
            passed.append(col)
        else:
            failed.append(col)

    return {"passed": passed, "failed": failed, "details": pd.DataFrame(details)}


def log_discovery(results: dict[str, Any], path: Path | None = None) -> None:
    path = path or (config.DATA_DIR / "discovery_memory.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(results, default=str, indent=2)
    path.write_text(payload, encoding="utf-8")


def run_screening(
    feature_panel: pd.DataFrame,
    target_ticker: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Predictability on SPY; factor screen vs forward 1m ETF return per row."""
    target_ticker = target_ticker or config.SPY_TICKER
    fp = feature_panel.copy()
    if "ret_1m" not in fp.columns:
        raise ValueError("feature_panel must contain ret_1m")

    spy = fp[fp["ticker"] == target_ticker].sort_values("date")
    fwd_spy = spy["ret_1m"].shift(-1).dropna()
    pred = predictability_score(fwd_spy)

    fp = fp.sort_values(["ticker", "date"])
    fp["_fwd"] = fp.groupby("ticker")["ret_1m"].shift(-1)
    y = fp["_fwd"]

    num_cols = [
        c
        for c in fp.columns
        if c not in ("date", "ticker", "ret_1m", "_fwd")
        and str(fp[c].dtype).startswith(("float", "int"))
    ]
    sf = screen_factors(
        fp[num_cols].apply(pd.to_numeric, errors="coerce"),
        y,
    )

    log_discovery({"predictability": pred, "screening": sf})

    keep = ["date", "ticker", "ret_1m"] + [c for c in sf["passed"] if c in fp.columns]
    keep = [c for c in keep if c in fp.columns]
    filtered = fp[keep].drop(columns=["_fwd"], errors="ignore") if sf["passed"] else fp.drop(
        columns=["_fwd"],
        errors="ignore",
    )
    report = {"predictability": pred, "screening": sf}
    return filtered, report
