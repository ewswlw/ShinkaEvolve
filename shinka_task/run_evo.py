#!/usr/bin/env python3
"""
Runner for TAA ShinkaEvolve task.

Usage (from repo root):
    uv run python shinka_task/run_evo.py --config_path shinka_task/shinka.yaml
"""

import argparse
import os
from pathlib import Path

import yaml

from shinka.core import EvolutionConfig, ShinkaEvolveRunner
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

TASK_SYS_MSG = """
You are an expert quantitative portfolio manager optimising a Python tactical asset allocation strategy.

GOAL
----
Improve CAGR and Calmar ratio of a monthly-rebalanced ETF portfolio.
Targets: CAGR > 15%  AND  Calmar ratio > 1.0 (from 2006 to 2026, no leverage).

UNIVERSE (12 ETFs)
------------------
SPY, QQQ, EFA, EEM  — equities (US large, US tech, intl developed, EM)
AGG, TLT, LQD, HYG  — fixed income (US agg, long treasury, IG corp, HY corp)
GLD, DBC, IYR, SHY  — alternatives/cash (gold, commodities, REITs, short treasury)

MODIFIABLE REGION
-----------------
Only modify the function `get_monthly_weights(monthly_rets, current_date)` inside
EVOLVE-BLOCK-START / EVOLVE-BLOCK-END.  Do NOT touch imports, constants, or the
backtest harness below the block.

IDEAS TO EXPLORE
----------------
- Trend-following / moving-average crossovers (e.g. price > 10-month SMA → hold)
- Volatility targeting (scale weights by inverse realized vol)
- Momentum with crash filter (e.g. go to cash if SPY < 200-day MA)
- Risk-parity allocation (equalise vol contribution per asset)
- Correlation-adjusted diversification (reduce correlated clusters)
- Regime-conditional allocation (detect bear/bull from SMA or drawdown signals)
- Ensemble of multiple sub-signals with adaptive weighting
- Kelly-like position sizing with Sharpe-proportional weights

HARD CONSTRAINTS (enforced externally — you don't need to code these)
----------------------------------------------------------------------
- Max single-asset weight: 40%
- Min SHY (safe-haven treasury) weight: 5%
- No leverage (weights sum to 1, all non-negative)
- Transaction cost: 5 bps one-way per traded notional

EVALUATION METRIC
-----------------
combined_score = (CAGR_score + Calmar_score) / 2
  where CAGR_score   = min(cagr / 0.15, 2.0)
        Calmar_score = min(calmar / 1.00, 2.0)
Score of 1.0 means targets are exactly met. Score > 1.0 means targets are beaten.
""".strip()


def main(config_path: str) -> None:
    cfg_file = Path(config_path)
    # Run from the task directory so relative paths (DB, results) land there
    task_dir = cfg_file.parent
    os.chdir(task_dir)

    with open(cfg_file.name, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["evo_config"]["task_sys_msg"] = TASK_SYS_MSG

    evo_config = EvolutionConfig(**config["evo_config"])
    job_config = LocalJobConfig(
        eval_program_path="evaluate.py",
        time="08:00:00",
    )
    db_config = DatabaseConfig(**config["db_config"])

    runner = ShinkaEvolveRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        max_evaluation_jobs=config["max_evaluation_jobs"],
        max_proposal_jobs=config["max_proposal_jobs"],
        max_db_workers=config["max_db_workers"],
        debug=False,
        verbose=True,
    )
    # runner.run() is the sync convenience wrapper that calls asyncio.run() internally
    runner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TAA ShinkaEvolve optimisation")
    parser.add_argument(
        "--config_path",
        type=str,
        default="shinka_task/shinka.yaml",
        help="Path to shinka.yaml (relative to repo root or absolute)",
    )
    args = parser.parse_args()
    main(args.config_path)
