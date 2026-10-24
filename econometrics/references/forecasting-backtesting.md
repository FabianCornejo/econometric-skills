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
- `backtest.walk_forward_splits` gives the train/test indices for any model (machine learning included).

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
