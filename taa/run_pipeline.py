"""End-to-end TAA pipeline CLI."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from taa import config
from taa.backtest import run_backtest_monthly, run_benchmarks
from taa.data_pull import pull_all
from taa.data_validation import validate_all
from taa.features import build_feature_panel
from taa.regime import add_regime_features, fit_hmm_regime, macro_panel_from_feature_panel
from taa.screening import run_screening
from taa.tearsheet import generate_tearsheet
from taa.walk_forward import evaluate_walk_forward, walk_forward_taa

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="TAA pipeline")
    p.add_argument("--offline", action="store_true", help="Use cached Parquet only")
    p.add_argument("--reload", action="store_true", help="Force Bloomberg re-pull")
    p.add_argument("--skip-screening", action="store_true")
    args = p.parse_args(argv)

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Pulling / loading data...")
    data = pull_all(reload=args.reload, offline=args.offline)

    logger.info("Validating...")
    validate_all(data)

    logger.info("Building feature panel...")
    panel = build_feature_panel(data)

    if not args.skip_screening:
        logger.info("Screening...")
        try:
            panel, _ = run_screening(panel)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Screening skipped: %s", exc)

    logger.info("Regime (HMM)...")
    try:
        macro_hmm = macro_panel_from_feature_panel(panel).ffill().bfill()
        _, regime_df = fit_hmm_regime(macro_hmm)
    except Exception as exc:  # noqa: BLE001
        logger.warning("HMM failed (%s); using neutral regime probs", exc)
        dates = sorted(panel["date"].unique())
        regime_df = pd.DataFrame(
            {
                "hmm_expansion_prob": [1.0 / 3] * len(dates),
                "hmm_contraction_prob": [1.0 / 3] * len(dates),
                "hmm_crisis_prob": [1.0 / 3] * len(dates),
                "hmm_state": [0] * len(dates),
            },
            index=pd.to_datetime(dates),
        )
    panel = add_regime_features(panel, regime_df)

    logger.info("Walk-forward...")
    wf = walk_forward_taa(panel, regime_df)
    wf.to_parquet(config.RESULTS_DIR / "walk_forward_returns.parquet", index=False)

    ev = evaluate_walk_forward(wf, n_trials=100)
    logger.info("Walk-forward eval: %s", ev)

    strat_ret = wf.set_index("date")["ret"].sort_index()
    vol = data.get("macro", {}).get("volatility")
    vix_m = None
    if vol is not None and not vol.empty:
        from taa.features import flatten_bdh, monthly_align

        v = flatten_bdh(vol)
        if "vix" in "".join(str(c).lower() for c in v.columns):
            col = [c for c in v.columns if "vix" in str(c).lower()][0]
            vix_m = monthly_align(v[[col]]).iloc[:, 0]

    bt = run_backtest_monthly(strat_ret, vix_m)
    net = bt["returns_net"]

    bms = run_benchmarks(panel)
    out = generate_tearsheet(net, bms, output_dir=config.RESULTS_DIR)
    logger.info("Tearsheet written to %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
