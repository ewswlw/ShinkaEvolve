"""Shared helpers: Bloomberg dispatcher, monthly alignment, z-scores."""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def monthly_align(
    df: pd.DataFrame,
    how: str = "last",
) -> pd.DataFrame:
    """Resample daily data to month-end (last business day in each month)."""
    if df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("monthly_align expects DatetimeIndex")
    out = df.resample("ME").apply(lambda x: x.iloc[-1] if len(x) else np.nan)
    out = out.dropna(how="all")
    return out


def expanding_zscore(
    s: pd.Series,
    min_periods: int = 24,
) -> pd.Series:
    """Expanding-window z-score with minimum history."""
    mu = s.expanding(min_periods=min_periods).mean()
    sigma = s.expanding(min_periods=min_periods).std()
    return (s - mu) / sigma.replace(0, np.nan)


def bbg(
    tickers: Union[str, list[str]],
    flds: Union[str, list[str]],
    start_date: str | None = None,
    end_date: str | None = None,
    per: str = "D",
    fill: str = "P",
    adjust: str | None = None,
    batch_size: int = 400,
    max_retries: int = 3,
    flat: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Smart Bloomberg dispatcher: bdp if no start_date, else bdh.

    Auto-batches tickers, retries with backoff, flattens single-field bdh MultiIndex.
    """
    from xbbg import blp

    if isinstance(tickers, str):
        tickers = [tickers]
    if isinstance(flds, str):
        flds = [flds]

    single_field = len(flds) == 1

    def _call(batch: list[str]) -> pd.DataFrame:
        for attempt in range(max_retries):
            try:
                if start_date is not None:
                    kw: dict[str, Any] = {"Per": per, "Fill": fill}
                    if adjust:
                        kw["adjust"] = adjust
                    kw.update(kwargs)
                    return blp.bdh(
                        batch,
                        flds,
                        start_date=start_date,
                        end_date=end_date
                        or pd.Timestamp.today().strftime("%Y-%m-%d"),
                        **kw,
                    )
                return blp.bdp(batch, flds, **kwargs)
            except Exception as exc:  # noqa: BLE001
                if attempt == max_retries - 1:
                    raise
                wait = 2**attempt
                logger.warning(
                    "Bloomberg error (attempt %s): %s. Retrying in %ss.",
                    attempt + 1,
                    exc,
                    wait,
                )
                time.sleep(wait)
        return pd.DataFrame()

    chunks = [tickers[i : i + batch_size] for i in range(0, len(tickers), batch_size)]
    results = [_call(c) for c in chunks]
    df = pd.concat(results) if results else pd.DataFrame()

    if start_date is not None and single_field and flat and not df.empty:
        try:
            df = df.droplevel(1, axis=1)
            df.columns.name = None
        except (ValueError, KeyError, IndexError):
            pass

    return df


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize index to timezone-naive DatetimeIndex."""
    if df.empty:
        return df
    idx = pd.to_datetime(df.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    out = df.copy()
    out.index = idx
    out = out.sort_index()
    return out


def compute_tracking_error(
    etf_rets: pd.Series,
    proxy_rets: pd.Series,
    annualize: bool = True,
) -> float:
    """Annualized tracking error from overlapping return series."""
    aligned = pd.concat([etf_rets, proxy_rets], axis=1).dropna()
    if len(aligned) < 12:
        return float("nan")
    diff = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    te = float(diff.std() * math.sqrt(12)) if annualize else float(diff.std())
    return te
