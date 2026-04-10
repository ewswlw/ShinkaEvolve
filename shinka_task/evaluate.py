"""
Evaluator for ShinkaEvolve TAA task.

Calls run_experiment() from the candidate program, validates the result,
and maps CAGR + Calmar into a single combined_score for evolution.

Fitness mapping
---------------
  cagr_score   = min(cagr  / 0.15, 2.0)   — 1.0 = exactly on target, 2.0 = 2x target
  calmar_score = min(calmar / 1.00, 2.0)   — 1.0 = exactly on target, 2.0 = 2x target
  combined_score = (cagr_score + calmar_score) / 2

Score of 1.0 means both targets are exactly met.
Score > 1.0 means both targets are beaten.
"""

from __future__ import annotations

import argparse

from shinka.core import run_shinka_eval

TARGET_CAGR: float = 0.15
TARGET_CALMAR: float = 1.0
MIN_MONTHS: int = 100


def get_kwargs(run_idx: int) -> dict:
    """Deterministic experiment — no randomness needed."""
    return {}


def aggregate_fn(results: list) -> dict:
    """Aggregate single-run result into Shinka metrics dict."""
    result = results[0]

    cagr: float = float(result.get("cagr", 0.0))
    calmar: float = float(result.get("calmar", 0.0))
    max_dd: float = float(result.get("max_dd", -1.0))
    sharpe: float = float(result.get("sharpe", 0.0))
    n_months: int = int(result.get("n_months", 0))

    # Scalarize: progress toward each target, capped at 2× to avoid runaway fitness
    cagr_score = min(cagr / TARGET_CAGR, 2.0) if TARGET_CAGR > 0 and cagr > 0 else max(cagr / TARGET_CAGR, -1.0)
    calmar_score = min(calmar / TARGET_CALMAR, 2.0) if TARGET_CALMAR > 0 and calmar > 0 else max(calmar / TARGET_CALMAR, -1.0)
    combined_score = float((cagr_score + calmar_score) / 2.0)

    both_targets_met = (cagr >= TARGET_CAGR) and (calmar >= TARGET_CALMAR)

    text_feedback = (
        f"CAGR={cagr:.2%} (target≥{TARGET_CAGR:.0%}), "
        f"Calmar={calmar:.2f} (target≥{TARGET_CALMAR:.1f}), "
        f"MaxDD={max_dd:.2%}, Sharpe={sharpe:.2f}, "
        f"Months={n_months}, "
        f"Score={combined_score:.3f}, "
        f"{'✓ TARGETS MET' if both_targets_met else '✗ targets not yet met'}"
    )

    return {
        "combined_score": combined_score,
        "cagr": cagr,
        "calmar": calmar,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "n_months": n_months,
        "both_targets_met": both_targets_met,
        "public": {
            "cagr": cagr,
            "calmar": calmar,
            "sharpe": sharpe,
        },
        "private": {
            "max_dd": max_dd,
            "n_months": n_months,
        },
        "extra_data": {},
        "text_feedback": text_feedback,
    }


def validate_fn(result) -> tuple[bool, str | None]:
    """Reject catastrophically broken candidates early."""
    if not isinstance(result, dict):
        return False, f"result must be dict, got {type(result)}"
    n = result.get("n_months", 0)
    if n < MIN_MONTHS:
        return False, f"too few months ({n} < {MIN_MONTHS}) — backtest likely crashed"
    cagr = result.get("cagr", 0.0)
    if cagr < -0.30:
        return False, f"catastrophic CAGR {cagr:.1%} — strategy is broken"
    return True, None


def main(program_path: str, results_dir: str) -> None:
    metrics, correct, err = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_experiment",
        num_runs=1,
        get_experiment_kwargs=get_kwargs,
        aggregate_metrics_fn=aggregate_fn,
        validate_fn=validate_fn,
    )
    print(f"\n{'='*60}")
    print(f"combined_score : {metrics.get('combined_score', 'N/A'):.4f}")
    print(f"CAGR           : {metrics.get('cagr', 'N/A'):.2%}")
    print(f"Calmar         : {metrics.get('calmar', 'N/A'):.2f}")
    print(f"MaxDD          : {metrics.get('max_dd', 'N/A'):.2%}")
    print(f"Sharpe         : {metrics.get('sharpe', 'N/A'):.2f}")
    print(f"correct        : {correct}")
    if err:
        print(f"error          : {err}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a TAA candidate program")
    parser.add_argument("--program_path", required=True, help="Path to candidate .py")
    parser.add_argument("--results_dir", required=True, help="Directory for JSON outputs")
    args = parser.parse_args()
    main(args.program_path, args.results_dir)
