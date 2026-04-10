# Project Constitution — Tactical Asset Allocation (TAA)

**Last Updated:** 2026-04-10 15:47 ET
**Status:** Approved specification — do not modify without explicit user sign-off.

---

## Technology Stack

- **Python** >=3.11, <3.14 (matches ShinkaEvolve constraint)
- **uv** for dependency management (`uv run python ...`, never bare `python`)
- **xbbg** 0.8.x + **blpapi** (Bloomberg data; already in `pyproject.toml`)
- **vectorbt** (backtest engine)
- **LightGBM** (primary ML model)
- **hmmlearn** (HMM regime detection)
- **scikit-learn** (purged CV, preprocessing)
- **SHAP** (feature importance)
- **gplearn** (optional symbolic regression for factor discovery)
- **tabulate** (tearsheet tables)
- **matplotlib** (equity curve plots; already in `pyproject.toml`)

## Project Structure

```
taa/                        # Standalone package at repo root
├── __init__.py
├── config.py               # All tickers, proxy maps, hyperparams, constants
├── data_pull.py            # Bloomberg bdh/bds pulls, Parquet caching
├── data_validation.py      # ml-algo-trading Step 1.5 validation layer
├── features.py             # ~57 feature computations from validated data
├── screening.py            # Predictability gate + |t|>3 factor screen
├── regime.py               # HMM regime detection + regime-conditional specialists
├── model.py                # LightGBM per-regime specialists, purged CV, SHAP
├── walk_forward.py         # Expanding-window walk-forward, PSR, DSR
├── backtest.py             # vectorbt portfolio sim, circuit breaker, turnover cap
├── tearsheet.py            # Performance tables + equity curve + results doc
├── utils.py                # Shared helpers (monthly alignment, z-score, etc.)
└── run_pipeline.py         # End-to-end orchestrator (Phases 2-10 in sequence)

data/taa/                   # Parquet cache (gitignored)
├── etf_prices.parquet
├── index_proxies.parquet
├── macro.parquet
├── credit.parquet
├── volatility.parquet
├── breadth.parquet
└── valuation.parquet

tests/taa/                  # Test suite
├── test_data_pull.py
├── test_features.py
├── test_screening.py
├── test_regime.py
├── test_model.py
├── test_backtest.py
└── conftest.py             # MockBlp fixtures (from xbbg skill)
```

## Executable Commands

```bash
# Install TAA dependencies
uv add --optional taa vectorbt hmmlearn gplearn shap tabulate lightgbm

# Run full pipeline (requires Bloomberg Terminal on localhost:8194)
uv run python -m taa.run_pipeline

# Run with cached data only (no Bloomberg)
uv run python -m taa.run_pipeline --offline

# Run tests (uses MockBlp, no Bloomberg needed)
uv run pytest tests/taa/ -v

# Run tests with coverage
uv run pytest tests/taa/ --cov=taa --cov-report=html
```

## Hard Boundaries

1. **No leverage.** Position weights must sum to <= 1.0 at all times. No shorting, no margin.
2. **No look-ahead bias.** All features must use only data available on or before the rebalance date. Validated by ml-algo-trading Step 1.5 shift-and-correlate + AST audit.
3. **No modification of `shinka/` source code.** TAA is a standalone package. It does not import from or extend the ShinkaEvolve core.
4. **No secrets in version control.** Bloomberg credentials, API keys, and `.env` files are gitignored.
5. **Honest reporting.** If target metrics (>15% CAGR, Calmar >1) are not achievable, document the best-attainable frontier and do not overfit to hit the targets.
6. **Full ml-algo-trading pipeline compliance.** Every strategy variant must pass: data validation → predictability gate → |t|>3 screening → purged CV → walk-forward → PSR > 0.95 → DSR > 0.95. No gates may be skipped.
7. **Reproducibility.** Random seeds are fixed. Parquet cache is deterministic. Pipeline output must be identical given the same cached data.
8. **Transaction costs always on.** Never report backtests without costs. Minimum 5 bps round-trip; 10 bps in stress (VIX > 30).
