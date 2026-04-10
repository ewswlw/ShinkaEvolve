"""HMM regime detection and triple-barrier labels."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from taa import config

logger = logging.getLogger(__name__)


def fit_hmm_regime(
    macro_features: pd.DataFrame,
    n_states: int = 3,
) -> tuple[Any, pd.DataFrame]:
    """
    Fit Gaussian HMM on macro feature subset; label states by mean slope_2s10s.
    """
    from hmmlearn.hmm import GaussianHMM

    cols = [c for c in config.HMM_INPUT_FEATURES if c in macro_features.columns]
    if len(cols) < 3:
        raise ValueError("Not enough HMM input columns")

    X = macro_features[cols].dropna().astype(float).values
    if len(X) < 48:
        raise ValueError("Insufficient history for HMM")

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=200,
        random_state=config.RANDOM_STATE,
    )
    model.fit(X)
    states = model.predict(X)
    probs = model.predict_proba(X)

    idx = macro_features[cols].dropna().index
    slope_col = "slope_2s10s" if "slope_2s10s" in macro_features.columns else cols[0]
    state_slopes: list[tuple[int, float]] = []
    for s in range(n_states):
        mask = states == s
        if mask.any() and slope_col in macro_features.columns:
            m = float(macro_features.loc[idx[mask], slope_col].mean())
        elif mask.any():
            m = float(np.mean(X[mask, 0]))
        else:
            m = 0.0
        state_slopes.append((s, m))
    state_slopes.sort(key=lambda x: x[1], reverse=True)
    ordered_raw = [x[0] for x in state_slopes][:n_states]
    remap = {ordered_raw[i]: i for i in range(len(ordered_raw))}

    hmm_state = np.array([remap[s] for s in states])
    out = pd.DataFrame(index=idx)
    out["hmm_state"] = hmm_state
    # Probabilities aligned to remapped labels 0=expansion ... 2=crisis
    prob_by_label = np.zeros_like(probs)
    for new_lbl in range(min(n_states, len(ordered_raw))):
        raw = ordered_raw[new_lbl]
        prob_by_label[:, new_lbl] = probs[:, raw]
    out["hmm_expansion_prob"] = prob_by_label[:, 0]
    out["hmm_contraction_prob"] = prob_by_label[:, 1]
    out["hmm_crisis_prob"] = prob_by_label[:, 2]
    return model, out


def triple_barrier_labels(
    prices: pd.Series,
    pt_sl: tuple[float, float] = (1.5, 1.0),
    max_holding: int = 1,
) -> pd.DataFrame:
    """Monthly triple-barrier labels (simplified: one step forward return)."""
    ret = prices.pct_change().shift(-1)
    vol = prices.pct_change().rolling(20).std()
    upper = pt_sl[0] * vol
    lower = pt_sl[1] * vol
    lab = pd.DataFrame(index=prices.index)
    lab["ret_fwd"] = ret
    lab["label"] = 0
    lab.loc[ret > upper.shift(-1), "label"] = 1
    lab.loc[ret < -lower.shift(-1), "label"] = -1
    return lab


def add_regime_features(
    feature_panel: pd.DataFrame,
    regime_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge F56–F59 regime columns onto panel by date."""
    r = regime_df.sort_index().copy()
    r.index.name = "date"
    r = r.reset_index()
    r["date"] = pd.to_datetime(r["date"])
    fp = feature_panel.copy()
    fp["date"] = pd.to_datetime(fp["date"])
    return fp.merge(r, on="date", how="left", suffixes=("", "_regime"))


def macro_panel_from_feature_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per date with HMM input columns (missing columns filled with 0)."""
    d = panel.drop_duplicates(subset=["date"], keep="last").set_index("date")
    out = pd.DataFrame(index=d.index)
    for c in config.HMM_INPUT_FEATURES:
        if c in d.columns:
            out[c] = pd.to_numeric(d[c], errors="coerce")
        else:
            out[c] = 0.0
    return out.sort_index().ffill().bfill()
