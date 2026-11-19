# Forecasting in practice and backtesting model-based strategies

Use when the goal is an out-of-sample forecast (yields, sales, returns, macro series) or when a forecast is turned into a decision rule (a trading signal, an allocation, an inventory order). Code: `scripts/forecasting.py` and `scripts/backtest.py`. Theory of ARIMA/VAR is in `time-series.md`; this file is about doing it without fooling yourself.

## Contents
1. The two rules that prevent most errors
2. Automatic ARIMA
3. ARIMAX: regressors that are actually known
4. VAR forecasts
5. Walk-forward evaluation
6. Benchmarks and accuracy tests
7. From forecast to strategy: backtest conventions
8. Time-series momentum
9. Reading a backtest honestly
10. Checklist
11. Armstrong's forecasting checklist (before trusting any forecast)
12. Labels that look ahead: overlap, purging and embargo (v2.7.0)
13. Intervals for any forecaster: conformal (v2.7.0)
14. A machine-learning or foundation model as a challenger (v2.7.0)

## 1. The two rules

1. **Information set.** A forecast of y_{t+h} may use only what was known at t. This excludes: contemporaneous regressors (today's VIX to "forecast" today's return), full-sample scalers, parameters estimated on the whole sample, smoothed (two-sided) filters, revised data you would not have had (use real-time vintages for macro data such as GDP).
2. **Timing of P&L.** A position decided with data up to the close of t earns the return of t+1. In code: `strategy_return = position.shift(1) * return`. `backtest.run_backtest` does this by default; `lag=0` is only for forecasts already stored at their target date.

A good self-test: replace the signal with the *future* return. If the backtest does not become absurdly profitable, the timing is wrong somewhere; if it does with `lag=0` but not with `lag=1`, the pipeline is correct (the test suite does exactly this).

## 2. Automatic ARIMA

`forecasting.auto_arima(y, max_p=5, max_q=5, seasonal_period=12)` replicates the logic of `pmdarima.auto_arima`/R's `forecast::auto.arima` with statsmodels only:

- **d** from repeated KPSS tests (`select_d`, H0 = stationary), as in pmdarima and R's `forecast`: the series is differenced only when there is evidence against stationarity. `test="adf"` is available; see `time-series.md` §3 for reading ADF and KPSS together.
- **Trend**: constant if d = 0; drift if d = 1 and the mean change is significant; none otherwise. A drift in a differenced model is a deterministic trend in levels: dangerous for long horizons.
- **(p, q)** by exhaustive search ranked by AICc (small-sample corrected); BIC is also reported because it picks more parsimonious models that often forecast better.
- **Residual check**: Ljung-Box on residuals with degrees of freedom adjusted for p + q. Autocorrelation left over means the search space or the regressors are insufficient.

Caveats:
- The model with the lowest AIC is not necessarily the best forecaster; compare the top 2–3 out of sample.
- For financial returns the selected model is usually ARMA(0,0) or something barely better: returns are close to a martingale difference. A selected ARMA(3,2) on daily returns is usually overfitting noise.
- For yields and rates in levels, d = 1 is typical; the random walk is a hard benchmark to beat (Meese-Rogoff for exchange rates; Duffee for yields).

## 3. ARIMAX

`SARIMAX(y, exog=X)` estimates a regression with ARMA errors. For **forecasting**, X must be known at the forecast origin:

```python
X = forecasting.lag_exog(macro_df, lags=1)        # VIX_{t-1}, oil_{t-1}, ...
fc = forecasting.rolling_origin(y, order=(1,0,1), exog=X, start=0.8, refit_every=20)
```

- With contemporaneous X the "forecast" needs X_{t+1}, which is unknown: the out-of-sample gain is an illusion. Either lag X, or forecast X too (then the uncertainty of X enters), or state explicitly that the exercise is conditional ("if oil is at…").
- Put regressors in the same transformation as y: returns with returns or changes, not returns with price **levels** (a non-stationary regressor in a stationary equation is a misspecification; see spurious regression in `time-series.md`).
- Compare ARIMAX against the same ARIMA without X with a Diebold-Mariano test over the same forecast sample. "RMSE went down" without a test is not evidence.
- For h > 1, regressors must be lagged at least h periods, or forecast themselves.

## 4. VAR forecasts

`forecasting.rolling_var(df, target="ret", start=0.8, maxlags=10, ic="aic", refit_every=20)`:

- Variables must be stationary (returns, changes, spreads, realized volatility); check with `diagnostics.stationarity_many`. If they are I(1) and cointegrated, use a VECM.
- Lag order is re-selected at each refit (information-set rule). AIC tends to pick long lags; BIC/HQ are more parsimonious.
- Output is indexed by target date. For trading, the forecast of r_t made at t−1 multiplies r_t directly (`lag=0`) or, equivalently, shift it to the origin date and use the default `lag=1`. Applying the forecast of r_t to r_{t+1} is a common one-day misalignment that silently destroys any edge.
- Variables such as volume change or intraday range help forecast **volatility** far more than the **sign** of returns. Test the volatility equation separately before concluding that a VAR "does not work".

## 5. Walk-forward evaluation

```python
fc = forecasting.rolling_origin(y, order, start=0.8, h=1, refit_every=12, window=None)
```

- `start`: first forecast origin (share or index). Use at least 30–50 out-of-sample points for any test.
- `refit_every`: re-estimate parameters every k origins and only update the state in between (`res.append`). Refit every period is the purist option but slow; refitting monthly on daily data is standard.
- `window=None` = expanding window (more data, slower adaptation); a fixed window adapts to breaks at the cost of variance. Try both: if conclusions flip, report it.
- For model selection inside the training window, nest a second walk-forward; never pick hyper-parameters on the test window.
- `backtest.walk_forward_splits` gives the train/test indices for any model (machine learning included). When the label of a row looks ahead (a 21-day return, "did the index rise over the next month"), pass `horizon=` so the training rows whose labels reach into the test window are purged (§12).
- Never `KFold(shuffle=True)` on a time series: neighbouring rows land in train and test and the score measures memory, not skill (§12).

## 6. Benchmarks and accuracy tests

`forecasting.benchmarks(y, start, season)` returns, indexed by target date:

| Benchmark | Right for |
|---|---|
| `naive` (y_{t-h}) | Levels: prices, yields, exchange rates, any near-random-walk series |
| `seasonal_naive` | Strongly seasonal series (retail sales, energy) |
| `drift` | Trending series |
| `hist_mean` (expanding mean) | Returns. Campbell-Thompson (2008) out-of-sample R² is measured against it |

`forecasting.evaluate(actual, {"naive": ..., "model": ...})` reports:
- **RMSE, MAE, bias, MAPE** (MAPE only for strictly positive series far from zero), **Theil U** = RMSE / RMSE_benchmark (< 1 beats the benchmark), **MASE** if an in-sample scale is supplied.
- **Diebold-Mariano with HLN correction** vs the benchmark. For nested models (ARIMAX vs ARIMA without X) DM is conservative; Clark-West (2007) is the right test when the small model is nested in the large one.
- **Directional accuracy** and the **Pesaran-Timmermann** test. A 55% hit rate can be significant on 1,000 forecasts and pure noise on 60. For levels, direction is the sign of the predicted *change* (pass `last_actual`), not the sign of the level.

Out-of-sample R² for returns: `R2_oos = 1 - sum((r - f)^2) / sum((r - hist_mean)^2)`. Values of 0.5–1% at the monthly frequency are economically meaningful (Campbell-Thompson); large values are a red flag for look-ahead.

## 7. From forecast to strategy

```python
pos = backtest.forecast_position(fc["forecast"].shift(-1), threshold=0.0005)  # forecast of t+1 stored at t
res = backtest.run_backtest(pos, returns, cost_bps=5)
backtest.performance(res["strategy_return"], turnover=res["turnover"])
```

Conventions in `run_backtest`: positions are exposures decided at t, held during t+1; costs are charged on turnover |Δposition| in basis points one-way. Include realistic costs: index futures 0.5–2 bp, liquid large caps 5–10 bp, small caps or emerging markets 20+ bp, plus borrow costs on shorts.

The threshold avoids trading on forecasts smaller than costs: set it at least at the round-trip cost.

## 8. Time-series momentum

`backtest.tsmom_position(prices, lookback=252, skip=21, vol_target=0.10)` implements Moskowitz, Ooi & Pedersen (2012): position = sign of the past 12-month return (skipping the last month), scaled to a constant ex-ante volatility. Notes:
- Volatility scaling is part of the definition; without it, exposure is concentrated in high-volatility periods and Sharpe falls.
- Single-asset TSMOM is noisy; the documented premium comes from diversified portfolios across asset classes (futures).
- The signal uses `shift(skip)/shift(lookback)` of prices: only past data. A lookback chosen after trying 20 values is a selection-biased parameter (§9).

## 9. Reading a backtest honestly

- **Out-of-sample split**: report in-sample and out-of-sample separately; a 70/30 split is fine for a first look, walk-forward is better.
- **Multiple testing**: every parameter combination tried is a test. With N trials, the best Sharpe is inflated; Bailey & López de Prado's deflated Sharpe ratio and Harvey, Liu & Zhu (2016) suggest t-statistics > 3 for new factors. Report how many variants were tried.
- **Uncertainty of the Sharpe**: `backtest.bootstrap_sharpe_ci` (block bootstrap). Annualized Sharpe has a standard error of roughly sqrt((1 + SR²/2) / years): 5 years of data gives a standard error of about 0.47 around a Sharpe of 0.5.
- **Benchmark**: compare with buy-and-hold and with the same exposure taken randomly. A long-biased strategy in a bull market looks good for that reason alone.
- **Regime dependence**: split results by volatility regime (`regimes.py`). Most momentum strategies lose in sharp reversals; most mean-reversion strategies lose in trends and crises.
- **Survivorship and data quality**: index constituents today were not the constituents 10 years ago; free sources (Yahoo Finance) adjust prices retroactively and have gaps.

## 10. Checklist

- [ ] Every input known at the forecast origin (lagged X, no full-sample scaling or smoothing).
- [ ] Positions shifted one bar before multiplying returns (or `lag=0` with target-dated forecasts).
- [ ] Benchmarks: naive/seasonal naive for levels, historical mean for returns, buy-and-hold for strategies.
- [ ] DM (or Clark-West) and Pesaran-Timmermann tests, not just RMSE or hit rate.
- [ ] Costs and turnover included; threshold ≥ round-trip cost.
- [ ] Number of variants tried reported; Sharpe with a confidence interval.
- [ ] Results by subperiod or regime.
- [ ] Labels that look ahead: `backtest.label_overlap` run; splits with `horizon=`; accuracy judged against the base rate (§12).
- [ ] Every forecast with an interval: the model's own, or `forecasting.conformal_intervals`, with its empirical coverage (§13).

## 11. Armstrong's forecasting checklist

From the evidence-based principles of Armstrong (2001, *Principles of Forecasting*) and the "golden rule" of Armstrong, Green and Graefe (2015): **be conservative**, that is, stay close to what is known unless strong, validated evidence says otherwise. Go through it before a forecast leaves the conversation.

- [ ] **Problem**: the decision the forecast serves, the horizon and the loss (which errors cost more) are written down; the metric matches the loss (RMSE for a mean, MAE for a median, MASE across series).
- [ ] **Knowledge**: what is known about the causal forces (growth, decay, mean reversion, policy) is used; a forecast that contradicts them needs a reason.
- [ ] **Simple first**: a naive, seasonal naive or damped-trend benchmark was run; a complex model is kept only if it beats them out of sample (Green and Armstrong 2015: complexity rarely pays).
- [ ] **Conservative trends**: trends are damped at long horizons and when the causal forces conflict or the series is volatile; no filtered trend (HP) is extrapolated (`time-series.md` §2).
- [ ] **Combine**: forecasts from methods that differ in data or assumptions are combined, with equal weights unless an evaluation window clearly favours others (`ts_univariate.combine_forecasts`).
- [ ] **Breaks and regimes**: the sample was checked for breaks (`ts_univariate.sup_wald`); a model estimated across a regime change is not extrapolated as if nothing happened.
- [ ] **Out-of-sample validation**: rolling-origin errors on data not used for fitting, at the horizons of the decision (`forecasting.rolling_origin`), with a test against the benchmark (Diebold-Mariano).
- [ ] **Uncertainty**: prediction intervals from the out-of-sample errors (`forecasting.conformal_intervals`, any model) or a bootstrap with parameter uncertainty (`ts_univariate.ar_bootstrap_forecast`), not only the analytic normal ones; their coverage was checked. Intervals are usually too narrow: say so.
- [ ] **Judgement**: any manual adjustment is recorded with its reason and kept separate from the model forecast.
- [ ] **Disclosure**: data vintage, method, parameters and code are given so that someone else can reproduce the forecast (`reporting.run_manifest`).

## 12. Labels that look ahead: overlap, purging and embargo

A label built from the next h periods (a monthly return computed every day from daily prices, a churn flag over the next 90 days) overlaps its neighbours in h − 1 periods. Three consequences, each with its tool:

- **Less information than rows.** `backtest.label_overlap(h, n)` reports the labels left without overlap (⌈n/h⌉), their average uniqueness (López de Prado 2018, §4.4) and by how much a naive standard error of a mean of such labels is too small (about √h with i.i.d. shocks; 4.5 for 240 monthly labels at h = 21). Use HAC errors with h − 1 lags (Hansen and Hodrick 1980) or keep one label every h periods.
- **Leakage between training and test.** The last h − 1 training rows have labels that use test-period data. `walk_forward_splits(..., horizon=h)` drops them (the same idea as sklearn's `TimeSeriesSplit(gap=h-1)`, whose folds are anchored at the end of the sample). For cross-validation (choosing a model or its tuning), `backtest.purged_kfold_splits(n, n_splits, horizon=h, embargo=e)` keeps contiguous blocks, purges every training row whose label shares a period with a test label and, after each block, embargoes e more rows (López de Prado 2018, §7.4). In simulations with labels over 21 periods and no skill at all (v2.7.0, `tests/test_backtest.py`), a 1-nearest-neighbour classifier scored 90% accuracy under shuffled K-fold, 52% under contiguous blocks without purging and 50% with `purged_kfold_splits`.
- **The base rate.** A "64% accuracy" is compared with the share of the majority class in the same test window (the accuracy of "always up"), not with 50%, together with the AUC. For a logit or probit fitted here, `discrete_choice.binary_effects` prints both. For an outside classifier (an LSTM, a boosted tree) no function of this skill tests accuracy against the base rate with overlapping labels: compute both on the same test window and bootstrap their difference in blocks of at least h periods, and say the code is your own.

The final evaluation stays walk-forward with the same horizon; purged K-fold is for choices made before it.

## 13. Intervals for any forecaster: conformal

`forecasting.conformal_intervals(actual, forecast, method="aci", h=h)` turns the out-of-sample errors of any model (target-dated, as `rolling_origin` returns them) into prediction intervals, with no model of the error:

- **What it uses.** The interval for a target forecast at origin t uses only errors already observed at t: for h > 1, the errors of the last h − 1 targets are not yet known and are left out.
- **Methods.** `"split"` (one calibration block; Lei et al. 2018) and `"rolling"` (recent errors at each origin) cover at least 1 − α only if the errors are exchangeable. `"aci"` (default; Gibbs and Candès 2021) moves the level after each miss so that the share of misses tracks α over time, with no assumption on the data; for h > 1 the update waits for the error to arrive.
- **What was measured** (v2.7.0, 1000 simulations per cell, 500 targets): with overlapping (h = 4) or persistent (AR 0.9) errors, 95% split and rolling intervals covered 84–95%, worst with 19 calibration errors; ACI covered 93–95%, h up to 12. With a short record ACI is no better: 60 targets, h = 12 and AR(0.9) errors gave about 90% for both.
- **What is guaranteed.** For one-step forecasts (h = 1), the share of misses is within (max(α, 1 − α) + γ)/(Tγ) of α over any T forecasts (Gibbs and Candès 2021, Prop. 4.1): 0.38 for T = 500 and the paper's γ = 0.005, so the guarantee is long-run. For h > 1 the paper's result does not apply as stated; the same argument with the delayed update gives (max(α, 1 − α) + hγ)/(Tγ), proved for this skill and not published. The function prints the slack; report the empirical coverage (with its 95% Wilson interval) as measured, not as guaranteed, and say "no informative interval" for rows whose interval is infinite or empty (the function counts them). **Long horizons (v2.7.1):** in simulations with h = 21 and 120 to 1000 targets, the default left 18–28% of the intervals infinite on average (the longest run up to ~300 rows at the 90th percentile), because the misses of overlapping errors come in runs and ACI sees each one h steps late; with h = 1 or 5, at most 1.6%. The infinite rows count as covered, so report `coverage_finite` (the finite intervals, empty ones included; 84–94% in those simulations) next to the total. In the simulations a smaller γ removed them, but the coverage fell to that of `method="rolling"` (90–94%), and no setting gave finite intervals at the nominal level, because overlapping errors leave about T/h independent ones. On the user's data, give the measured numbers (the finding prints them, and the long-horizon advice when h > 1 and at least 5% of the intervals are infinite), compare with `method="rolling"`, and give the measured coverage as the interval's real level.
- **The next forecast.** Append the target not yet observed as a row with `actual` = NaN (its forecast filled): it gets an interval and stays out of the coverage. The index must be in time order, one row per target.
- Intervals are symmetric around the forecast: a biased forecaster gets wide intervals. Check the bias first (`forecast_metrics`).

## 14. A machine-learning or foundation model as a challenger

A gradient-boosting model, an LSTM or a pre-trained foundation model (TimesFM, Chronos, Moirai…) is evaluated like any other forecaster, never as the default:

1. Same targets, same origins, same information set: produce its forecasts in the walk-forward loop (`walk_forward_splits` with `horizon=`), scaler and features fitted inside each training window.
2. The simple benchmarks first (§6): naive, seasonal naive, historical mean for returns. Report MASE and Theil U.
3. A Diebold-Mariano test against the best benchmark over the same forecast sample (`forecasting.evaluate`); for classification, accuracy against the base rate and the AUC.
4. An interval: `forecasting.conformal_intervals` on its out-of-sample errors, with the empirical coverage.
5. Count the variants tried (architectures, feature sets, hyper-parameters, seeds) and report the number; a leaderboard's winner is selected on the validation score and needs a test window used once (`review-checklist.md`).
6. Licence of the weights: some foundation models are open, others are for non-commercial use only (TimesFM 3.0 weights, for example); state it when a client will use the forecast.

