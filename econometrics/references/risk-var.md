# Value-at-Risk, Expected Shortfall and their backtests

Use for market-risk questions: "how much can this position lose in a day/month at 99%?", "is our VaR model adequate?", "how much worse is the tail than a normal model says?". Code: `scripts/risk.py`. Volatility models themselves (GARCH family, stylized facts) are in `volatility.md`.

## 1. Definitions and conventions

- **VaR_α**: the loss not exceeded with probability 1 − α over the horizon. With returns r, VaR = −q_α(r). Reported as a positive number (0.031 = 3.1%).
- **Expected Shortfall (ES, CVaR)**: average loss given that the loss exceeds VaR, ES = −E[r | r ≤ q_α]. ES is coherent (sub-additive) and is the Basel FRTB measure at 97.5%; VaR at 99% remains the backtesting object.
- **Violation**: r_t < −VaR_t. Under a correct model, violations are Bernoulli(α), i.i.d.
- **Information set**: VaR_t uses returns up to t−1 only. All functions in `risk.py` shift accordingly; check this whenever you write your own.
- **Horizon**: the square-root-of-time rule (VaR_h = VaR_1·√h) assumes i.i.d. normal returns; with GARCH, aggregate the h-step variance forecasts or simulate.
- **Returns**: log returns aggregate over time, simple returns aggregate across assets. For a single position, either works at a daily horizon; be consistent.

## 2. Estimators

| Method | `risk.py` | Pros | Cons |
|---|---|---|---|
| Historical simulation | `var_historical(r, α, 250)` | No distributional assumption | Slow to react; ghost effects when a crash leaves the window; few tail points (2.5 obs at 1% with 250 days) |
| Parametric normal | `var_parametric(r, α, 250)` | Simple, smooth | Underestimates fat tails; constant volatility within the window |
| Parametric t | `var_parametric(..., dist="t")` | Fat tails | Still slow to react |
| EWMA / RiskMetrics | `var_ewma(r, α, lam=0.94)` | Reacts quickly; one parameter | Normal quantile too thin; λ fixed, not estimated |
| Filtered historical simulation | `var_ewma(..., method="fhs")` | EWMA volatility × empirical quantile of standardized returns: fat tails without assuming them | Needs ~500 observations of standardized returns |
| GARCH(-t / GJR) | `var_garch(r, α, window=1000, refit_every=20, o=1)` | Volatility clustering, leverage, fat tails; estimated parameters | Estimation risk; slower; sensitive to the error distribution |
| Conditional EVT (McNeil-Frey) | `var_evt(r, α, window=1000, q=0.90, filter="ewma")` | EWMA filter for clustering + GPD tail for the standardized losses: the best simple model at 99%+ | Threshold choice; needs ~1000 observations |

Implementation notes:
- `ewma_volatility` seeds the recursion with the variance of the first 30 observations and returns NaN before that, so the seed never leaks out of sample. Using the full-sample variance as the seed (a common shortcut) uses future data.
- `var_garch` re-estimates every `refit_every` days and filters with fixed parameters in between: almost identical results to daily refitting at a fraction of the time. Returns are scaled by 100 internally for the optimizer.
- The Student-t quantile must be rescaled to unit variance: `t.ppf(α, ν)·√((ν−2)/ν)`. Forgetting it overstates VaR for small ν.

### 2.1 Extreme value theory for the far tail

For 99.5%–99.9% VaR, stress losses or "how bad is a 1-in-1000-days loss?", the empirical quantile has a handful of points and the normal or t models impose a tail shape. Peaks over threshold (Balkema-de Haan-Pickands): losses above a high threshold u follow approximately a **generalized Pareto distribution** with shape ξ (the tail index; ξ = 1/ν for a Student-t, ξ ≈ 0.2–0.4 for daily equity returns) and scale β.

1. **Choose the threshold**: `mean_excess(r)` gives the mean excess e(u) = E[L − u | L > u]; above a good threshold it is roughly linear (upward slope means ξ > 0). Rule of thumb: the 90–95% quantile, with at least 50–100 exceedances.
2. **Fit**: `gpd_fit(r, q=0.95, alpha=0.001)` returns ξ (with the Smith 1987 SE), β, VaR and ES with delta-method CIs, the empirical VaR for comparison, and a **threshold-stability table**: ξ and VaR should barely move across nearby thresholds. If they drift, the threshold is too low.
3. **Read ξ**: ξ > 0 with a CI excluding 0 is a heavy (power-law) tail: moments of order ≥ 1/ξ do not exist, and a normal model understates far-tail losses by a large factor (in the skill's tests, a t(4) 99.9% VaR is 0.072; EVT gives 0.073, the normal 0.044). ξ ≥ 1 makes ES infinite.
4. **Through time**: `var_evt(..., filter="ewma")` standardizes by EWMA volatility before fitting (McNeil & Frey 2000): EVT alone is unconditional and its violations cluster.
5. The delta-method CI for a far-tail VaR is symmetric while the true uncertainty is skewed upward; for α well below 1/n_u say so, or bootstrap the exceedances.

## 3. Backtesting

`risk.backtest_var(r, {"hist": v1, "ewma": v2, "garch": v3}, alpha=0.01)` evaluates all models **on their common sample** (otherwise the comparison is unfair) and reports:

| Test | H0 | Reads |
|---|---|---|
| Kupiec POF (LR_uc ~ χ²₁) | violation rate = α | Too many violations: risk underestimated. Too few: capital wasted |
| Christoffersen independence (LR_ind ~ χ²₁) | violations are not clustered | Rejection: the model reacts too slowly to volatility (typical of historical and static models) |
| Conditional coverage (LR_cc ~ χ²₂) | both | Overall verdict |
| Basel traffic light (last 250 days, 99%) | — | Green 0–4 violations, yellow 5–9, red 10+ |

Also reported: P(violation | violation yesterday), which should be close to α; values of 5–10% reveal clustering.

Choosing among models: discard those that fail coverage or independence; among the rest, prefer the one with the **lower average VaR** (same protection, less capital). For ES, see §3.1.

With 250 observations at 1%, tests have very low power (2.5 expected violations). Use 2–4 years of daily data, or backtest at 2.5–5% as a complement.

### 3.1 Backtesting Expected Shortfall

ES is the regulatory measure (FRTB, 97.5%) but it is not elicitable on its own: no loss function ranks ES forecasts alone, so it is backtested with tests on the tail and ranked jointly with VaR.

| Function | Test | Reads |
|---|---|---|
| `es_backtest(r, var, es, alpha=0.025)` | **McNeil-Frey** exceedance residuals (L − ES on violation days, optionally divided by σ), bootstrap-t p-value | Positive mean: losses beyond VaR are larger than the ES forecast. The plain t-test is conservative because the residuals are right-skewed (about 2% size at nominal 5% in the skill's simulations); the bootstrap-t is about 3.5% |
| same | **Acerbi-Szekely Z2** = Σ r_t I_t / (T α ES_t) + 1, simulated p-value under a scaled-t null matched to the forecasts' ES/VaR ratio | Z2 < 0: ES underestimated; it penalizes too many violations and too-small ES together. For α = 2.5% and ~250 days, zones green > −0.70, yellow > −1.8, red below |
| `fz_loss`, `compare_es_models(r, {"evt": df1, "garch": df2}, alpha)` | FZ0 loss (Fissler-Ziegel; Patton, Ziegel & Chen 2019), Diebold-Mariano vs the best | Lower mean loss is better; ranks (VaR, ES) pairs fairly on a common sample |

Order of work: `backtest_var` (coverage and independence of the VaR) → `es_backtest` (tail magnitude) → `compare_es_models` among the survivors. In the skill's GARCH-t simulations, conditional EVT passes all four tests and ranks first by FZ0; EWMA-normal fails coverage and both ES tests.

## 4. Typical results and how to read them

On daily equity, FX or crypto returns the usual ranking is:
1. Static historical and normal VaR: too many violations, clustered in crises (fail independence).
2. EWMA-normal: clustering fixed, but too many violations at 99% because the normal tail is too thin.
3. EWMA-FHS, GARCH-t and conditional EVT: pass both tests; conditional EVT is usually best at 99%+ and on ES.

Comparing a model only with "historical VaR" on a chart is not a backtest. Run the tests.

## 5. Portfolio VaR and dependence

- Variance-covariance: VaR_p = z·√(w'Σw) — assumes joint normality and constant correlation.
- Correlations rise in crises; a Gaussian copula has zero tail dependence and understates joint crashes. For two or a few assets, fit copulas (`dependence.fit_copulas`), simulate with `dependence.simulate_copula`, map back with the empirical or GARCH-filtered marginals and read the portfolio quantile (see `dependence-copulas.md` §5).
- For many assets: DCC-GARCH or filtered historical simulation on the portfolio return.

## 6. Limitations to state in a report

- VaR says nothing about the size of losses beyond it: report ES as well, with its backtest (§3.1).
- All these models are backward looking; stress tests and scenario analysis complement them.
- Liquidity: VaR on mid prices ignores the cost of exiting large positions.
- Model risk: report at least two methods; if they differ widely, that is information.
