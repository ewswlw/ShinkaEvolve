"""Performance tables and plots."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

from taa import config

logger = logging.getLogger(__name__)


def _cagr(returns: pd.Series, periods_per_year: int = 12) -> float:
    r = returns.dropna().astype(float)
    if r.empty:
        return float("nan")
    years = len(r) / periods_per_year
    return float((1 + r).prod() ** (1 / max(years, 1e-9)) - 1)


def _max_dd(returns: pd.Series) -> tuple[float, pd.Timestamp | None]:
    wealth = (1 + returns.fillna(0)).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    m = float(dd.min())
    trough = dd.idxmin() if len(dd) else None
    return m, trough


def compute_summary_stats(
    returns: pd.Series,
    name: str = "strategy",
) -> dict[str, Any]:
    r = returns.dropna().astype(float)
    cagr = _cagr(r)
    mdd, _ = _max_dd(r)
    calmar = cagr / abs(mdd) if mdd and mdd != 0 else float("nan")
    vol = float(r.std() * np.sqrt(12))
    sharpe = float(r.mean() / (r.std(ddof=1) + 1e-12) * np.sqrt(12))
    return {
        "name": name,
        "cagr": cagr,
        "calmar": calmar,
        "sharpe": sharpe,
        "vol_ann": vol,
        "max_dd": mdd,
        "n_months": len(r),
    }


def generate_tearsheet(
    strategy_ret: pd.Series,
    benchmarks: dict[str, pd.Series],
    output_dir: Path | None = None,
) -> Path:
    """Write markdown summary + tables + plots."""
    output_dir = output_dir or (config.RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = [compute_summary_stats(strategy_ret, "strategy")]
    for k, s in benchmarks.items():
        stats.append(compute_summary_stats(s, k))

    keys = ["name", "cagr", "calmar", "sharpe", "vol_ann", "max_dd", "n_months"]
    table = tabulate(
        [[s.get(k) for k in keys] for s in stats],
        headers=keys,
        tablefmt="github",
    )
    md = f"# TAA Tearsheet\n\n## Summary\n\n```\n{table}\n```\n"
    (output_dir / "tearsheet.md").write_text(md, encoding="utf-8")

    # Equity curve
    eq = (1 + strategy_ret.fillna(0)).cumprod()
    plt.figure(figsize=(10, 5))
    eq.plot(label="strategy")
    for k, s in benchmarks.items():
        (1 + s.fillna(0).reindex(strategy_ret.index).fillna(0)).cumprod().plot(
            label=k,
            alpha=0.7,
        )
    plt.legend()
    plt.title("Equity curve (normalized)")
    plt.savefig(output_dir / "equity_curve.png", dpi=120)
    plt.close()

    # Drawdown
    w = (1 + strategy_ret.fillna(0)).cumprod()
    peak = w.cummax()
    dd = w / peak - 1.0
    plt.figure(figsize=(10, 3))
    dd.plot()
    plt.title("Drawdown")
    plt.savefig(output_dir / "drawdown.png", dpi=120)
    plt.close()

    return output_dir / "tearsheet.md"
