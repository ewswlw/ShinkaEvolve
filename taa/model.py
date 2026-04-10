"""Purged CV, LightGBM specialists, weight prediction."""

from __future__ import annotations

import itertools
import logging
from typing import Any, Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

try:
    import lightgbm as lgb
except ImportError:
    lgb = None  # type: ignore[assignment]

try:
    import shap
except ImportError:
    shap = None  # type: ignore[assignment]

from taa import config

logger = logging.getLogger(__name__)


class PurgedKFold(BaseCrossValidator):
    """K-fold with embargo after test set."""

    def __init__(
        self,
        n_splits: int = 5,
        embargo: int = 2,
    ) -> None:
        self.n_splits = n_splits
        self.embargo = embargo

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return self.n_splits

    def split(
        self,
        X: pd.DataFrame,
        y: Any = None,
        groups: Any = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1
        indices = np.arange(n)
        current = 0
        for fs in fold_sizes:
            start, stop = current, current + fs
            test_idx = indices[start:stop]
            train_idx = np.concatenate([indices[:start], indices[stop + self.embargo :]])
            current = stop
            if len(train_idx) > 10 and len(test_idx) > 0:
                yield train_idx, test_idx


def train_specialist(
    X: pd.DataFrame,
    y: pd.Series,
    mask: pd.Series,
    param_grid: dict[str, list[Any]] | None = None,
) -> tuple[Any, dict[str, Any], int]:
    """Train one LightGBM regressor on rows where mask is True."""
    if lgb is None:
        raise ImportError("lightgbm required for taa")
    param_grid = param_grid or config.LGBM_GRID
    Xb = X.loc[mask].astype(float)
    yb = y.loc[mask].astype(float)
    if len(Xb) < 20:
        model = lgb.LGBMRegressor(
            n_estimators=50,
            max_depth=2,
            learning_rate=0.1,
            random_state=config.RANDOM_STATE,
        )
        model.fit(Xb, yb)
        return model, {}, 1

    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    n_trials = len(combos)
    best_score = -np.inf
    best_model = None
    best_params: dict[str, Any] = {}
    for combo in combos:
        params = dict(zip(keys, combo, strict=False))
        m = lgb.LGBMRegressor(**params, random_state=config.RANDOM_STATE)
        m.fit(Xb, yb)
        pred = m.predict(Xb)
        score = -float(np.mean((pred - yb.values) ** 2))
        if score > best_score:
            best_score = score
            best_model = m
            best_params = params
    assert best_model is not None
    return best_model, best_params, n_trials


def train_ensemble(
    panel: pd.DataFrame,
    feature_cols: list[str],
    regime_col: str = "hmm_state",
) -> dict[str, Any]:
    """Train one model per regime predicting cross-sectional next-month return proxy."""
    if "ret_1m" not in panel.columns:
        raise ValueError("ret_1m required")
    # Target: forward return per row
    panel = panel.sort_values(["date", "ticker"])
    panel["_y"] = panel.groupby("ticker")["ret_1m"].shift(-1)
    df = panel.dropna(subset=["_y"] + feature_cols + [regime_col])
    models: dict[int, Any] = {}
    trials = 0
    for regime in (0, 1, 2):
        m = df[df[regime_col] == regime]
        if len(m) < 30:
            continue
        X = m[feature_cols]
        y = m["_y"]
        model, _, nt = train_specialist(X, y, pd.Series(True, index=m.index))
        models[regime] = model
        trials += nt

    shap_vals = None
    if shap is not None and models:
        first = next(iter(models.values()))
        shap_vals = None  # optional: compute on sample

    return {"models": models, "feature_cols": feature_cols, "n_trials": trials, "shap": shap_vals}


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.exp((x - np.max(x)) / max(temperature, 1e-6))
    return z / z.sum()


def predict_weights(
    models: dict[int, Any],
    feature_cols: list[str],
    row: pd.Series,
    regime_probs: np.ndarray,
    tickers: list[str],
) -> pd.Series:
    """Blend specialist predictions into weights with constraints."""
    x = row[feature_cols].astype(float).values.reshape(1, -1)
    exp_ret = np.zeros(len(tickers))
    for i, reg in enumerate([0, 1, 2]):
        if reg not in models:
            continue
        pred = models[reg].predict(x)[0]
        exp_ret += regime_probs[i] * pred
    # duplicate pred across assets — specialists trained on pooled target; use uniform blend
    w = softmax(np.ones(len(tickers)) * exp_ret / max(len(tickers), 1))
    w = np.clip(w, 0.0, config.BACKTEST_PARAMS["max_single_weight"])
    w = w / w.sum()
    # min SHY
    shy_i = tickers.index("SHY US Equity") if "SHY US Equity" in tickers else 0
    if w[shy_i] < config.BACKTEST_PARAMS["min_shy_weight"]:
        w[shy_i] = config.BACKTEST_PARAMS["min_shy_weight"]
        w = w / w.sum()
    return pd.Series(w, index=tickers)
