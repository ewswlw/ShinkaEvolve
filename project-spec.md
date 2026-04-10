<project_specification>
  <project_name>TAA — Tactical Asset Allocation via Regime-Conditional ML</project_name>

  <overview>
    Build a no-leverage, monthly-rebalanced tactical asset allocation strategy using
    US-listed macro ETFs (with index TR proxies for pre-inception history). The strategy
    targets >15% CAGR with Calmar ratio >1 from January 2006 to present. It uses a hybrid
    approach: transparent allocation rules (vol targeting, turnover caps, circuit breaker)
    combined with an ML regime-detection layer (HMM + LightGBM) that drives weight tilting
    via three regime-conditional specialist portfolios blended by posterior probabilities.
    The full ml-algo-trading pipeline (data validation, predictability gate, |t|>3 screening,
    purged CV, walk-forward, PSR, DSR) is mandatory. If targets are not achievable without
    overfitting, the project documents the best-attainable risk/return frontier honestly.
    Success is measured by: (1) strategy passes all ml-algo-trading gates, (2) CAGR and
    Calmar meet or approach targets, (3) results are fully reproducible from cached data.
  </overview>

  <technology_stack>
    - Python >=3.11, <3.14 (uv-managed)
    - xbbg 0.8.x + blpapi (Bloomberg data)
    - vectorbt (backtesting)
    - LightGBM (ML model)
    - hmmlearn (HMM regime detection)
    - scikit-learn (purged CV, preprocessing, metrics)
    - SHAP (feature importance)
    - gplearn (optional symbolic regression)
    - tabulate (tearsheet tables)
    - matplotlib (plots)
    - pandas, numpy, scipy, statsmodels (core computation)
  </technology_stack>

  <assumptions>
    1. Bloomberg Terminal is available on localhost:8194 for initial data pull.
    2. All ETFs in the universe are tradeable with sufficient liquidity (>$50M ADV).
    3. Index TR proxies are acceptable stand-ins for ETF returns before ETF inception.
       A tracking-error haircut is applied (see data_validation).
    4. Monthly rebalance is executable at month-end closing prices (no slippage beyond
       the cost model).
    5. The 5 bps round-trip cost assumption holds for normal markets; 10 bps in stress.
    6. No tax optimization or wash-sale considerations.
    7. Portfolio is fully invested or in SHY (no partial cash positions outside circuit
       breaker events).
    8. Macro data releases are point-in-time correct as reported by Bloomberg's revision
       history (no look-ahead into revised figures).
  </assumptions>

  <out_of_scope>
    - Intraday trading or sub-monthly rebalancing (except circuit breaker overrides)
    - Leveraged or inverse ETFs
    - Short selling or derivatives
    - International (non-US-listed) ETFs
    - Live execution / order management system integration
    - Tax-loss harvesting
    - ShinkaEvolve evolutionary wrapper (deferred to optional Phase 11)
    - Real-time Bloomberg streaming (live() / subscribe())
  </out_of_scope>

  <etf_universe>
    All tickers use Bloomberg format: "TICKER Exchange AssetClass"

    <asset ticker="SPY US Equity" class="US Large Cap Equity"
           proxy="SPXT Index" proxy_name="S&P 500 Total Return"
           inception="1993-01-29" />
    <asset ticker="QQQ US Equity" class="US Tech / Growth"
           proxy="XNDX Index" proxy_name="NASDAQ 100 Total Return"
           inception="1999-03-10" />
    <asset ticker="EFA US Equity" class="Intl Developed Equity"
           proxy="NDDUEAFE Index" proxy_name="MSCI EAFE Net TR USD"
           inception="2001-08-14" />
    <asset ticker="EEM US Equity" class="Emerging Market Equity"
           proxy="NDUEEGF Index" proxy_name="MSCI EM Net TR USD"
           inception="2003-04-07" />
    <asset ticker="AGG US Equity" class="US Aggregate Bond"
           proxy="LBUSTRUU Index" proxy_name="Bloomberg US Agg Bond TR"
           inception="2003-09-22" />
    <asset ticker="TLT US Equity" class="US Long Treasury 20+Y"
           proxy="LUATTRUU Index" proxy_name="Bloomberg US Treasury 20+Y TR"
           inception="2002-07-22" />
    <asset ticker="LQD US Equity" class="US IG Corporate Bond"
           proxy="LUACTRUU Index" proxy_name="Bloomberg US Corporate IG TR"
           inception="2002-07-22" />
    <asset ticker="HYG US Equity" class="US High Yield Corporate"
           proxy="LF98TRUU Index" proxy_name="Bloomberg US Corporate HY TR"
           inception="2007-04-04" />
    <asset ticker="GLD US Equity" class="Gold"
           proxy="XAU Curncy" proxy_name="Gold Spot USD/oz"
           inception="2004-11-18" />
    <asset ticker="DBC US Equity" class="Broad Commodities"
           proxy="DBLCDBCE Index" proxy_name="DBIQ Opt Yield Diversified Commodity"
           inception="2006-02-03" />
    <asset ticker="IYR US Equity" class="US REITs"
           proxy="DWRTF Index" proxy_name="Dow Jones US Real Estate TR"
           inception="2000-06-12" />
    <asset ticker="SHY US Equity" class="Short-Term Treasury 1-3Y (risk-off asset)"
           proxy="LD12TRUU Index" proxy_name="Bloomberg US Treasury 1-3Y TR"
           inception="2002-07-22" />

    NOTE: Proxy tickers should be verified on Bloomberg terminal before first pull.
    Some index tickers may use slightly different mnemonics depending on entitlements.
    The data_pull module includes a proxy_verification step that logs any ticker failures.
  </etf_universe>

  <macro_data_tickers>
    All pulled via blp.bdh() with daily frequency, then aligned to monthly (last biz day).

    <category name="US Yield Curve">
      <ticker bbg="USGG3M Index" field="PX_LAST" description="3-month Treasury yield" />
      <ticker bbg="USGG2YR Index" field="PX_LAST" description="2-year Treasury yield" />
      <ticker bbg="USGG5YR Index" field="PX_LAST" description="5-year Treasury yield" />
      <ticker bbg="USGG10YR Index" field="PX_LAST" description="10-year Treasury yield" />
      <ticker bbg="USGG30YR Index" field="PX_LAST" description="30-year Treasury yield" />
    </category>

    <category name="Credit Spreads">
      <ticker bbg="LF98OAS Index" field="PX_LAST" description="Bloomberg US HY OAS (bps)" />
      <ticker bbg="LUACOAS Index" field="PX_LAST" description="Bloomberg US IG OAS (bps)" />
    </category>

    <category name="Volatility">
      <ticker bbg="VIX Index" field="PX_LAST" description="CBOE VIX" />
      <ticker bbg="MOVE Index" field="PX_LAST" description="ICE BofAML MOVE (rate vol)" />
    </category>

    <category name="Macro Leading Indicators">
      <ticker bbg="INJCJC Index" field="PX_LAST" description="Initial Jobless Claims (weekly, use last of month)" />
      <ticker bbg="NAPMPMI Index" field="PX_LAST" description="ISM Manufacturing PMI" />
      <ticker bbg="IP CHNG Index" field="PX_LAST" description="Industrial Production MoM %" />
      <ticker bbg="CONCCONF Index" field="PX_LAST" description="Conference Board Consumer Confidence" />
      <ticker bbg="LEI CHNG Index" field="PX_LAST" description="Leading Economic Index MoM change" />
    </category>

    <category name="Inflation">
      <ticker bbg="CPI YOY Index" field="PX_LAST" description="US CPI Year-over-Year %" />
      <ticker bbg="USGGBE10 Index" field="PX_LAST" description="10-year Breakeven Inflation" />
    </category>

    <category name="Fed Policy">
      <ticker bbg="FDTR Index" field="PX_LAST" description="Fed Funds Target Rate (upper bound)" />
      <ticker bbg="US0003M Index" field="PX_LAST" description="3-month USD LIBOR/SOFR" />
    </category>

    <category name="Liquidity / Financial Conditions">
      <ticker bbg="BFCIUS Index" field="PX_LAST" description="Bloomberg US Financial Conditions" />
    </category>

    <category name="Valuation (S&amp;P 500)">
      <ticker bbg="SPX Index" field="PE_RATIO" description="S&amp;P 500 trailing P/E" />
      <ticker bbg="SPX Index" field="EARN_YLD" description="S&amp;P 500 earnings yield (E/P)" />
    </category>

    <category name="Market Breadth (derived)">
      <description>
        Compute monthly: pull SPX Index members via blp.bds('SPX Index','INDX_MEMBERS'),
        then blp.bdp(members, 'PX_LAST') and blp.bdp(members, 'MOV_AVG_200D').
        Feature = count(PX_LAST > MOV_AVG_200D) / total_members.
        Cache the result; do not re-pull members every run (membership changes slowly).
        For history before 2010, use Bloomberg MMTH Index as proxy if available.
      </description>
    </category>
  </macro_data_tickers>

  <bloomberg_pull_specification>
    <function>blp.bdh()</function>
    <date_range start="2005-01-01" end="today"
                note="Start 1 year before backtest (2006-01-01) to allow lookback warmup" />
    <fields>TOT_RETURN_INDEX_GROSS_DVDS for ETFs; PX_LAST for indices/macro</fields>
    <frequency>Daily pull, then resample to monthly (last business day) in features.py</frequency>
    <adjustments>adjust='all' for ETF price series used in return computation</adjustments>
    <batching>batch_size=400 via the bbg() dispatcher from the xbbg skill Section 16</batching>
    <caching>Parquet files in data/taa/; reload flag to force re-pull</caching>
    <offline_mode>If Bloomberg unavailable, load from cached Parquet only</offline_mode>
  </bloomberg_pull_specification>

  <data_validation_specification>
    Implements ml-algo-trading Step 1.5. Runs after data pull, before feature engineering.

    <domain name="Schema">Verify all expected columns present; dtypes correct</domain>
    <domain name="Calendar">No duplicate dates; business-day aligned; no gaps > 5 biz days
           except known holidays</domain>
    <domain name="Alignment">All ETF/proxy series cover 2005-01-01 to present; macro series
           may start later but must cover 2006-01-01+</domain>
    <domain name="Bias">
      - Look-ahead: shift-and-correlate test on all features vs forward returns
      - Survivorship: ETF universe is fixed (no additions/removals based on performance)
      - Backfill: macro indicators use first-release values (Bloomberg revision history)
      - Corporate actions: adjust='all' applied to all ETF price series
    </domain>
    <domain name="Quality">
      - NaN rate per column must be < 5% after forward-fill
      - Outlier detection: flag daily returns > 15% or < -15% for manual review
    </domain>
    <domain name="Reconciliation">
      - For each ETF↔proxy pair: compute tracking error in overlapping period
      - If annualized TE > 50 bps: flag as "low confidence", apply TE haircut to proxy returns
      - Log all reconciliation results
    </domain>
    <domain name="Provenance">SHA-256 hash chain on all cached Parquet files</domain>
  </data_validation_specification>

  <feature_definitions>
    All features are computed monthly (last business day). Lookback windows are in months
    unless noted. All z-scores use expanding window with minimum 24 months.

    <group name="ETF-Level (per ETF, per month)">
      <feature id="F01" name="ret_1m"  formula="price[t] / price[t-1] - 1" />
      <feature id="F02" name="ret_3m"  formula="price[t] / price[t-3] - 1" />
      <feature id="F03" name="ret_6m"  formula="price[t] / price[t-6] - 1" />
      <feature id="F04" name="ret_12m" formula="price[t] / price[t-12] - 1" />
      <feature id="F05" name="ret_12m_1m" formula="price[t-1] / price[t-12] - 1"
               note="12-1 month momentum; skip most recent month to avoid reversal" />
      <feature id="F06" name="vol_3m"
               formula="std(daily_returns[t-63:t]) * sqrt(252)"
               note="Annualized 3-month realized vol from daily returns" />
      <feature id="F07" name="vol_12m"
               formula="std(daily_returns[t-252:t]) * sqrt(252)" />
      <feature id="F08" name="vol_ratio"
               formula="vol_3m / vol_12m"
               note="Rising ratio = vol regime change" />
      <feature id="F09" name="max_dd_6m"
               formula="max trailing drawdown over 126 trading days" />
      <feature id="F10" name="price_sma_ratio_10m"
               formula="price[t] / SMA(price, 10 months)"
               note="Trend signal: >1 = uptrend" />
      <feature id="F11" name="price_sma_ratio_3m"
               formula="price[t] / SMA(price, 3 months)" />
      <feature id="F12" name="volume_sma_ratio"
               formula="mean(daily_volume, 21 days) / mean(daily_volume, 252 days)" />
    </group>

    <group name="Yield Curve">
      <feature id="F13" name="ust_2y" formula="USGG2YR level" />
      <feature id="F14" name="ust_10y" formula="USGG10YR level" />
      <feature id="F15" name="slope_2s10s" formula="USGG10YR - USGG2YR" />
      <feature id="F16" name="slope_3m10y" formula="USGG10YR - USGG3M" />
      <feature id="F17" name="curve_curvature" formula="2*USGG5YR - USGG2YR - USGG10YR" />
      <feature id="F18" name="slope_2s10s_chg_3m" formula="slope_2s10s[t] - slope_2s10s[t-3]" />
      <feature id="F19" name="slope_2s10s_zscore" formula="z_score(slope_2s10s, expanding, min_periods=24)" />
    </group>

    <group name="Credit">
      <feature id="F20" name="hy_oas" formula="LF98OAS level" />
      <feature id="F21" name="ig_oas" formula="LUACOAS level" />
      <feature id="F22" name="hy_oas_chg_1m" formula="hy_oas[t] - hy_oas[t-1]" />
      <feature id="F23" name="hy_oas_chg_3m" formula="hy_oas[t] - hy_oas[t-3]" />
      <feature id="F24" name="hy_oas_zscore" formula="z_score(hy_oas, expanding, min_periods=24)" />
      <feature id="F25" name="credit_quality_spread" formula="hy_oas - ig_oas" />
    </group>

    <group name="Volatility">
      <feature id="F26" name="vix_level" formula="VIX Index close" />
      <feature id="F27" name="vix_sma_ratio" formula="VIX / SMA(VIX, 3 months)" />
      <feature id="F28" name="vix_zscore" formula="z_score(VIX, expanding, min_periods=24)" />
      <feature id="F29" name="move_level" formula="MOVE Index close" />
      <feature id="F30" name="move_zscore" formula="z_score(MOVE, expanding, min_periods=24)" />
    </group>

    <group name="Macro Leading">
      <feature id="F31" name="ism_pmi" formula="NAPMPMI level" />
      <feature id="F32" name="ism_pmi_chg_3m" formula="ism_pmi[t] - ism_pmi[t-3]" />
      <feature id="F33" name="ism_above_50" formula="1 if ism_pmi > 50 else 0" />
      <feature id="F34" name="claims_4wma" formula="SMA(INJCJC, 4 weeks) — use last monthly value" />
      <feature id="F35" name="claims_yoy_chg" formula="claims_4wma[t] / claims_4wma[t-12] - 1" />
      <feature id="F36" name="consumer_conf" formula="CONCCONF level" />
      <feature id="F37" name="consumer_conf_chg_3m" formula="consumer_conf[t] - consumer_conf[t-3]" />
      <feature id="F38" name="lei_chg" formula="LEI CHNG level (already MoM %)" />
    </group>

    <group name="Inflation">
      <feature id="F39" name="cpi_yoy" formula="CPI YOY level" />
      <feature id="F40" name="breakeven_10y" formula="USGGBE10 level" />
      <feature id="F41" name="breakeven_chg_3m" formula="breakeven_10y[t] - breakeven_10y[t-3]" />
    </group>

    <group name="Fed Policy">
      <feature id="F42" name="fed_funds" formula="FDTR level" />
      <feature id="F43" name="fed_funds_chg_6m" formula="fed_funds[t] - fed_funds[t-6]"
               note="Positive = hiking cycle; negative = cutting" />
    </group>

    <group name="Breadth">
      <feature id="F44" name="pct_above_200dma"
               formula="count(SPX members where PX_LAST > MOV_AVG_200D) / total_members" />
      <feature id="F45" name="pct_above_200dma_chg_1m"
               formula="pct_above_200dma[t] - pct_above_200dma[t-1]" />
    </group>

    <group name="Valuation">
      <feature id="F46" name="spx_pe" formula="SPX Index PE_RATIO" />
      <feature id="F47" name="spx_earnings_yield" formula="SPX Index EARN_YLD" />
      <feature id="F48" name="equity_risk_premium"
               formula="spx_earnings_yield - ust_10y / 100"
               note="Equity E/P minus 10Y yield; higher = equities cheaper vs bonds" />
    </group>

    <group name="Liquidity">
      <feature id="F49" name="ted_spread"
               formula="US0003M - USGG3M"
               note="3M LIBOR/SOFR minus 3M T-Bill; wider = funding stress" />
      <feature id="F50" name="fin_conditions" formula="BFCIUS level" />
      <feature id="F51" name="fin_conditions_chg_3m" formula="fin_conditions[t] - fin_conditions[t-3]" />
    </group>

    <group name="Cross-Asset Relative">
      <feature id="F52" name="equity_bond_rel_mom_3m" formula="ret_3m(SPY) - ret_3m(AGG)" />
      <feature id="F53" name="equity_bond_rel_mom_12m" formula="ret_12m(SPY) - ret_12m(AGG)" />
      <feature id="F54" name="commodity_equity_ratio_3m" formula="ret_3m(DBC) - ret_3m(SPY)" />
      <feature id="F55" name="gold_equity_ratio_3m" formula="ret_3m(GLD) - ret_3m(SPY)" />
    </group>

    <group name="Regime (computed in regime.py, added to panel)">
      <feature id="F56" name="hmm_state"
               formula="HMM.predict(macro_features) — categorical: 0=expansion, 1=contraction, 2=crisis" />
      <feature id="F57" name="hmm_expansion_prob" formula="HMM.predict_proba()[:, 0]" />
      <feature id="F58" name="hmm_contraction_prob" formula="HMM.predict_proba()[:, 1]" />
      <feature id="F59" name="hmm_crisis_prob" formula="HMM.predict_proba()[:, 2]" />
    </group>

    Total: 59 features before screening gate.
    Expected ~15-25 to survive |t| > 3.0 screening.
  </feature_definitions>

  <regime_model_specification>
    <model>Gaussian HMM (hmmlearn.hmm.GaussianHMM)</model>
    <n_states>3 (expansion, contraction, crisis)</n_states>
    <input_features>
      Subset of macro features for regime detection (to avoid circularity with ETF features):
      slope_2s10s, hy_oas_zscore, vix_zscore, ism_pmi_chg_3m, claims_yoy_chg,
      fin_conditions, pct_above_200dma
    </input_features>
    <fitting>
      - Fit on expanding window (minimum 36 months)
      - Re-fit monthly with all available history up to rebalance date
      - n_iter=200, covariance_type='full', random_state=42
      - Label states post-hoc by sorting on mean(slope_2s10s) within each state:
        highest slope = expansion, middle = contraction, lowest = crisis
    </fitting>
    <output>
      Monthly: hmm_state (int), hmm_expansion_prob, hmm_contraction_prob, hmm_crisis_prob
    </output>
  </regime_model_specification>

  <regime_conditional_specialist_ensemble>
    Three specialist LightGBM models, each trained on data from its regime:

    <specialist name="Expansion">
      <training_data>Months where hmm_state == 0 (expansion)</training_data>
      <bias>Overweight equities (SPY, QQQ, EEM), credit (LQD, HYG); underweight bonds, gold</bias>
      <output>Weight vector w_expansion (12 ETFs, sums to 1.0)</output>
    </specialist>

    <specialist name="Contraction">
      <training_data>Months where hmm_state == 1 (contraction)</training_data>
      <bias>Overweight long bonds (TLT, AGG), gold (GLD); underweight equities</bias>
      <output>Weight vector w_contraction (12 ETFs, sums to 1.0)</output>
    </specialist>

    <specialist name="Crisis">
      <training_data>Months where hmm_state == 2 (crisis)</training_data>
      <bias>Max allocation to SHY/TLT; minimal equity; gold as hedge</bias>
      <output>Weight vector w_crisis (12 ETFs, sums to 1.0)</output>
    </specialist>

    <blending>
      w_final = p_expansion * w_expansion + p_contraction * w_contraction + p_crisis * w_crisis
      where p_* = HMM posterior probabilities for current month.
      Soft blending avoids whipsaw from hard regime switches.
    </blending>

    <constraints>
      - All weights >= 0 (no shorting)
      - sum(w_final) == 1.0 (fully invested)
      - No single ETF > 40% weight
      - Minimum 5% in SHY at all times (liquidity buffer)
    </constraints>
  </regime_conditional_specialist_ensemble>

  <ml_model_specification>
    <model>LightGBM (lgb.LGBMRegressor or LGBMClassifier)</model>
    <target>Forward 1-month return for each ETF (regression) or top-quintile indicator (classification)</target>
    <hyperparameters>
      Tuned via purged CV grid search:
      - n_estimators: [100, 200, 500]
      - max_depth: [2, 3, 4] (shallow to prevent overfitting on ~220 monthly obs)
      - learning_rate: [0.01, 0.05, 0.1]
      - min_child_samples: [10, 20, 30]
      - subsample: [0.7, 0.8]
      - colsample_bytree: [0.6, 0.8]
      - reg_alpha: [0.0, 0.1, 1.0]
      - reg_lambda: [0.0, 0.1, 1.0]
      - random_state: 42
    </hyperparameters>
    <cross_validation>
      PurgedKFold from ml-algo-trading references/validation-backtesting.md:
      - n_splits: 5
      - embargo: 2 months (to prevent leakage from overlapping feature windows)
    </cross_validation>
    <feature_selection>
      SHAP-based: after initial fit, retain top 8-12 features by mean |SHAP value|.
      Verify economic sense of top features before proceeding.
    </feature_selection>
    <turnover_penalty>
      Add -2 bps per 1% of portfolio turnover to the objective function.
    </turnover_penalty>
  </ml_model_specification>

  <screening_specification>
    <predictability_gate>
      Run predictability_score() from ml-algo-trading references/predictability-analysis.md
      on forward 1-month returns of SPY (proxy for asset class).
      - Score < 20: STOP — no exploitable signal
      - Score 20-40: CAUTION — regime-switching only, stricter thresholds
      - Score > 40: PROCEED
    </predictability_gate>

    <factor_screening_gate>
      For each of the 59 features:
      1. Compute rank IC vs forward 1-month returns (cross-sectional, per month)
      2. Compute t-statistic of mean IC: t = mean(IC) / (std(IC) / sqrt(N_months))
      3. Gate: |t| > 3.0 required to enter the model
      4. Log all results (pass and fail) in discovery memory
      Expected: ~15-25 features survive
    </factor_screening_gate>
  </screening_specification>

  <walk_forward_specification>
    <method>Expanding window</method>
    <initial_training_window>36 months minimum (2006-01 to 2008-12)</initial_training_window>
    <test_window>12 months</test_window>
    <step_size>6 months</step_size>
    <pass_condition>
      Strategy achieves positive annualized return in >= 60% of walk-forward windows.
      If predictability score was 20-40: require >= 70%.
    </pass_condition>
    <psr>
      Probabilistic Sharpe Ratio on concatenated OOS returns.
      PSR > 0.95 required. Benchmark SR = 0 (zero skill).
    </psr>
    <dsr>
      Deflated Sharpe Ratio accounting for all trials tested.
      n_trials = number of hyperparameter combinations * number of feature subsets tested.
      DSR > 0.95 required.
    </dsr>
  </walk_forward_specification>

  <backtest_specification>
    <engine>vectorbt</engine>
    <date_range start="2006-01-31" end="today" />
    <rebalance_frequency>Monthly (last business day)</rebalance_frequency>
    <initial_capital>1000000 (for realistic position sizing)</initial_capital>

    <cost_model>
      - Normal markets (VIX <= 30): 5 bps round-trip per trade
      - Stress markets (VIX > 30): 10 bps round-trip per trade
      - No additional slippage model (ETFs are highly liquid)
    </cost_model>

    <circuit_breaker>
      Monitored daily even though rebalance is monthly.
      Trigger conditions (OR):
        1. Portfolio trailing 21-day drawdown exceeds -7%
        2. VIX closes above 30
      Action: Immediately shift to 100% SHY (short-term Treasury)
      Duration: Hold SHY until next scheduled monthly rebalance
      Cost: Apply stress cost model (10 bps) for circuit breaker trades
    </circuit_breaker>

    <turnover_cap>
      Maximum one-way turnover per rebalance: 50% of portfolio.
      If ML-recommended weights would exceed this, scale the weight changes proportionally
      toward the current weights until turnover = 50%.
    </turnover_cap>

    <weight_constraints>
      - All weights >= 0 (no shorting)
      - sum(weights) == 1.0 (fully invested, no leverage)
      - No single ETF > 40%
      - Minimum 5% in SHY at all times
    </weight_constraints>

    <benchmarks>
      1. SPY buy-and-hold (100% US equity)
      2. 60/40 portfolio (60% SPY + 40% AGG, monthly rebalanced)
      3. Equal-weight monthly rebalanced (1/12 per ETF)
    </benchmarks>
  </backtest_specification>

  <tearsheet_specification>
    <tables>
      1. Summary statistics: CAGR, Calmar, Sharpe, Sortino, max drawdown, max DD date,
         time underwater, volatility, skewness, kurtosis
      2. Annual returns (strategy vs benchmarks)
      3. Monthly return heatmap (year x month)
      4. Drawdown analysis: top 5 drawdowns with start, trough, recovery dates
      5. Rolling 12-month Sharpe ratio
      6. Regime allocation over time (% in expansion/contraction/crisis specialists)
      7. Feature importance (SHAP top 10)
      8. Walk-forward window results
      9. PSR and DSR results with n_trials
      10. Turnover statistics (annual, per-rebalance)
      11. Circuit breaker activation log
      12. Overfitting prevention checklist (all items pass/fail)
    </tables>
    <plots>
      1. Equity curve: strategy vs benchmarks (log scale)
      2. Drawdown chart
      3. Regime state timeline (color-coded)
      4. Weight allocation over time (stacked area)
    </plots>
    <output_format>
      Tables: plain text via tabulate (also saved as HTML)
      Plots: matplotlib PNG saved to data/taa/results/
      Summary: Markdown report at data/taa/results/tearsheet.md
    </output_format>
  </tearsheet_specification>

  <implementation_steps>
    Phase 1 — Scaffold taa/ package
      - Create taa/ directory structure per project-constitution.md
      - Add optional-deps group [taa] to pyproject.toml
      - Create taa/config.py with all ticker maps, proxy pairs, hyperparams, constants
      - Create data/taa/ directory (gitignored)
      - Estimated: 1 session

    Phase 2 — Bloomberg data pipeline
      - Implement taa/data_pull.py using xbbg bbg() dispatcher
      - Pull ETF prices (TOT_RETURN_INDEX_GROSS_DVDS + PX_LAST + PX_VOLUME)
      - Pull index TR proxies for pre-inception history
      - Pull all macro/credit/vol/breadth/valuation tickers
      - Parquet caching with reload flag
      - Monthly alignment (last business day)
      - Offline mode (load from cache only)
      - Dependency: Phase 1
      - Estimated: 1-2 sessions

    Phase 3 — Data validation
      - Implement taa/data_validation.py
      - Run ml-algo-trading Step 1.5 validate() on all panels
      - Proxy tracking-error reconciliation with haircut
      - Bias checks (look-ahead shift-and-correlate, survivorship, backfill)
      - Provenance hashing
      - Dependency: Phase 2
      - Estimated: 1 session

    Phase 4 — Feature engineering
      - Implement taa/features.py
      - Compute all 59 features from feature_definitions
      - Monthly panel output: DatetimeIndex x (ETF x feature) or long format
      - Fractional differentiation on non-stationary series (test ADF)
      - Dependency: Phase 3
      - Estimated: 1-2 sessions

    Phase 5 — Predictability gate + factor screening
      - Implement taa/screening.py
      - Run predictability_score() on forward returns
      - Run |t|>3 univariate IC screen on all 59 features
      - Log results in discovery memory (JSON)
      - Reduce to surviving feature set
      - Dependency: Phase 4
      - Estimated: 1 session

    Phase 6 — Regime model + labels
      - Implement taa/regime.py
      - Fit Gaussian HMM (3 states) on macro feature subset
      - Post-hoc state labeling by yield curve slope
      - Triple-barrier labels on monthly ETF returns
      - Add regime features (F56-F59) to panel
      - Dependency: Phase 5
      - Estimated: 1 session

    Phase 7 — ML model + purged CV
      - Implement taa/model.py
      - Three LightGBM specialists (one per regime)
      - PurgedKFold with 2-month embargo
      - Hyperparameter grid search within purged CV
      - SHAP feature importance + economic sense check
      - Turnover penalty in objective
      - Weight vector output per month
      - Dependency: Phase 6
      - Estimated: 2 sessions

    Phase 8 — Walk-forward + PSR/DSR
      - Implement taa/walk_forward.py
      - Expanding-window walk-forward (36mo train, 12mo test, 6mo step)
      - Concatenate OOS returns
      - Compute PSR (>0.95 gate)
      - Compute DSR with n_trials (>0.95 gate)
      - Pass/fail decision
      - Dependency: Phase 7
      - Estimated: 1 session

    Phase 9 — Backtest + circuit breaker
      - Implement taa/backtest.py
      - vectorbt Portfolio simulation
      - Cost model (5/10 bps normal/stress)
      - Circuit breaker (-7% DD or VIX>30 → SHY)
      - Turnover cap (50% max)
      - Weight constraints enforced
      - Benchmark portfolios
      - Dependency: Phase 8
      - Estimated: 1-2 sessions

    Phase 10 — Tearsheet + results doc
      - Implement taa/tearsheet.py
      - All 12 tables + 4 plots per tearsheet_specification
      - Overfitting prevention checklist (all 13 items)
      - Honest reporting: if targets not met, document best frontier
      - Save to data/taa/results/
      - Dependency: Phase 9
      - Estimated: 1 session

    Phase 11 — (Optional) ShinkaEvolve wrapper
      - Create examples/taa_evolve/initial.py (baseline strategy)
      - Create examples/taa_evolve/evaluate.py (backtest harness returning fitness)
      - ShinkaEvolve evolves strategy logic/features/thresholds
      - DSR recalculated with full evolution trial count
      - Dependency: Phase 10
      - Estimated: 1-2 sessions
  </implementation_steps>

  <success_criteria>
    <functional>
      - Pipeline runs end-to-end from Bloomberg pull to tearsheet without errors
      - All ml-algo-trading gates produce pass/fail results (even if some fail)
      - Circuit breaker activates correctly on historical stress events
      - Backtest matches vectorbt reference implementation
    </functional>
    <ux>
      - Single command to run full pipeline: uv run python -m taa.run_pipeline
      - Offline mode works without Bloomberg
      - Tearsheet is readable and actionable for a portfolio manager
      - Clear error messages when Bloomberg is unavailable or data is missing
    </ux>
    <technical>
      - All features use only data available at rebalance time (no look-ahead)
      - Random seeds fixed for full reproducibility
      - Test coverage >= 80% on taa/ package (using MockBlp)
      - No hardcoded absolute paths (all paths relative to project root or config)
      - Code follows project style: snake_case, type hints, docstrings
    </technical>
    <performance_targets>
      Primary (aspirational, not guaranteed):
        - CAGR > 15%
        - Calmar ratio > 1.0
      Secondary (must achieve for strategy to be considered valid):
        - PSR > 0.95
        - DSR > 0.95
        - Walk-forward: positive return in >= 60% of windows
        - All features pass |t| > 3.0 screening gate
      If primary targets are not achievable:
        - Document the efficient frontier (CAGR vs max DD)
        - Report the best Calmar ratio achievable without overfitting
        - Suggest constraint relaxations (e.g., weekly rebalance, wider universe)
    </performance_targets>
  </success_criteria>
</project_specification>
