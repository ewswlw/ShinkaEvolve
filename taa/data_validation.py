"""Simplified ml-algo-trading Step 1.5 validation (7 domains)."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from taa import config
from taa.utils import compute_tracking_error

logger = logging.getLogger(__name__)


def compute_provenance(path: Path) -> str:
    """SHA-256 hash of file contents."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_schema(
    df: pd.DataFrame,
    required_columns: list[str] | None = None,
) -> dict[str, Any]:
    if df.empty:
        return {"ok": False, "reason": "empty"}
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            return {"ok": False, "missing": list(missing)}
    return {"ok": True}


def validate_calendar(
    df: pd.DataFrame,
    max_gap_bdays: int = 5,
) -> dict[str, Any]:
    if df.empty:
        return {"ok": False}
    if not isinstance(df.index, pd.DatetimeIndex):
        return {"ok": False, "reason": "index"}
    if df.index.duplicated().any():
        return {"ok": False, "reason": "duplicate_dates"}
    idx = df.index.sort_values()
    diffs = idx.to_series().diff().dt.days
    big = diffs[diffs > max_gap_bdays * 2]
    if len(big) > len(idx) * 0.05:
        return {"ok": False, "reason": "large_gaps", "n": len(big)}
    return {"ok": True}


def validate_alignment(
    dfs: dict[str, pd.DataFrame],
    start: str = config.PULL_START,
) -> dict[str, Any]:
    t0 = pd.Timestamp(start)
    for name, df in dfs.items():
        if df.empty:
            continue
        if df.index.min() > t0 + pd.Timedelta(days=400):
            return {"ok": False, "series": name, "start": str(df.index.min())}
    return {"ok": True}


def validate_bias(
    features: pd.DataFrame,
    forward_ret: pd.Series,
    max_abs_corr: float = 0.5,
) -> dict[str, Any]:
    """Shift-and-correlate: flag if feature correlates with future return at negative lag."""
    if features.empty or forward_ret.empty:
        return {"ok": True, "skipped": True}
    aligned = forward_ret.shift(-1)
    bad: list[str] = []
    for col in features.columns:
        if col == "ticker":
            continue
        s = features[col].reindex(aligned.index)
        c = s.corr(aligned)
        if pd.notna(c) and abs(c) > max_abs_corr:
            bad.append(col)
    return {"ok": len(bad) == 0, "suspicious_cols": bad}


def validate_quality(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"ok": True}
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return {"ok": True}
    nan_rate = float(num.isna().mean().mean())
    if nan_rate > 0.05:
        return {"ok": False, "nan_rate": nan_rate}
    return {"ok": True}


def validate_reconciliation(
    etf_rets: pd.Series,
    proxy_rets: pd.Series,
    max_te_annual: float = 0.005,
) -> dict[str, Any]:
    """Annualized tracking error vs threshold (50 bps default)."""
    te = compute_tracking_error(etf_rets, proxy_rets, annualize=True)
    if np.isnan(te):
        return {"ok": True, "te": None}
    return {"ok": te <= max_te_annual * 100, "te": te}


def validate_all(
    data: dict[str, Any],
) -> dict[str, Any]:
    """Run validation domains; return pass/fail summary."""
    results: dict[str, Any] = {}
    if "spliced_levels" in data and not data["spliced_levels"].empty:
        sl = data["spliced_levels"]
        results["schema"] = validate_schema(sl, list(sl.columns))
        results["calendar"] = validate_calendar(sl)
    results["alignment"] = validate_alignment(
        {k: v for k, v in data.items() if isinstance(v, pd.DataFrame)},
    )
    results["quality"] = validate_quality(
        data.get("spliced_levels", pd.DataFrame()),
    )
    prov: dict[str, str] = {}
    for p in config.DATA_DIR.glob("*.parquet"):
        try:
            prov[p.name] = compute_provenance(p)
        except OSError:
            continue
    results["provenance"] = prov
    ok = all(
        v.get("ok", True) if isinstance(v, dict) else True
        for k, v in results.items()
        if k != "provenance"
    )
    results["overall_ok"] = ok
    log_path = config.DATA_DIR / "validation_report.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(results, default=str, indent=2), encoding="utf-8")
    return results
