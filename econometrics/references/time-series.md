# Time series

## Contents
1. Decision tree
2. Decomposition and filters
3. Stationarity and unit roots
4. Structural breaks
5. Univariate models (ARIMA/SARIMA, ETS, ARFIMA)
6. Time-series regression (ADL, ARIMAX) and spurious regression
7. Cointegration and error-correction models (ECM/VECM)
8. VAR: estimation, Granger, impulse responses, variance decomposition
9. Structural VAR (SVAR) and identification
10. Bayesian VAR and data-rich VARs
11. Forecasting: evaluation, intervals, combination
12. Common pitfalls

## 1. Decision tree

```
A single series, goal = forecast?
 └─ SARIMA / ETS (section 5) → out-of-sample evaluation (section 11)
Y explained by X, effect of interest?
 ├─ All I(0)              → OLS/ADL with HAC errors (section 6)
 ├─ All I(1)
 │   ├─ Cointegrated      → ECM/VECM (section 7)
 │   └─ Not cointegrated  → model in differences (section 6)
 └─ Mixed orders          → difference the I(1) ones or use ARDL/Pesaran bounds test
Variables that affect each other (rates, FX, inflation, sales)?
 └─ VAR (I(0) or differences) / VECM (cointegrated) → sections 8–10
Financial returns, risk → volatility.md
```

General protocol (modeling cycle): handle non-stationarity → initial model from ACF/PACF and theory → estimate → diagnose residuals → respecify until diagnostics pass → only then forecast.

## 2. Decomposition and filters

Series = deterministic component (trend, seasonality — no statistics needed) + stochastic component (what is modeled).

- Seasonality: seasonal dummies in the regression; `statsmodels.tsa.seasonal.STL` to decompose; X-13 (`x13_arima_analysis`, requires the Census binary) for official adjustment.
- **HP filter** (`statsmodels.tsa.filters.hp_filter.hpfilter(y, lamb=...)`): λ = 1600 quarterly, 129600 monthly, 6.25 annual (Ravn-Uhlig). Widely used by central banks for the output gap. Important caveats: it distorts the end of the sample (the last point depends on unobserved future data — bad for real-time decisions), can generate spurious cycles and fails to properly remove stochastic trends (Phillips and Jin, 2015; Hamilton, 2018). Alternative: Hamilton's filter — regress y_{t+h} on y_t, …, y_{t−3} (h = 8 quarters or 24 months) and use the residual as the cycle.
- `ts_univariate.detrend(y, method="hp" | "hamilton" | "bk" | "cf" | "poly")` returns trend and cycle with λ (or h, or the band) set from the frequency. With `one_sided=True` the HP trend at each date uses only data up to that date (what a decision maker had); for the two-sided filter it reports the end-point revision, the average gap between the real-time and the final trend, next to the SD of the cycle. Baxter-King loses K observations at each end; Christiano-Fitzgerald and the two-sided HP revise their last values. None of them forecasts: do not extrapolate a filtered trend.
- Detrending and differencing are not interchangeable: a line removed from a random walk leaves a persistent, spurious "cycle"; differencing a trend-stationary series leaves a negative lag-1 autocorrelation of (φ − 1)/2 (`simulation.demo_trend_vs_difference`).
- Year-over-year change (ln y_t − ln y_{t−12}) removes stable seasonality and is intuitive for business, but introduces MA(11) autocorrelation in the changes.

## 3. Stationarity and unit roots

Why it matters: with a unit root, shocks have permanent effects (the impulse response does not decay), the variance grows over time and the usual asymptotics (t, F) fail — regressions between independent I(1) series look significant (spurious regression).

Use **ADF and KPSS together**, because they have opposite nulls:

| ADF (H0: unit root) | KPSS (H0: stationary) | Conclusion |
|---|---|---|
| rejects | does not reject | Stationary |
| does not reject | rejects | Unit root → difference |
| rejects | rejects | Possible deterministic trend or break; test with a trend and Zivot-Andrews |
| does not reject | does not reject | Inconclusive: low power, high persistence or **long memory** (ARFIMA, section 5) |

`diagnostics.stationarity(y, alpha=0.05)` applies this table with DF-GLS in place of the ADF, at level `alpha`; `multivariate.var_analysis` and `vecm_fit` pass their own `alpha` to it (v2.4.7.2). The KPSS p-value is interpolated from a table that only covers [0.01, 0.10] ("≤ 0.01" or "≥ 0.10" at the bounds): a bound value of 0.01 counts as a rejection at any alpha ≥ 0.01, and an alpha outside [0.01, 0.10] gets a warning because the KPSS decision is then not exact.

Choosing tests (Stock, *Handbook of Econometrics* ch. 46):
- **DF-GLS (Elliott-Rothenberg-Stock)** has much more power than the ADF (e.g., 60% vs. 22% against ρ = 0.9) while keeping good size control → use it as the main unit-root test (`arch.unitroot.DFGLS`).
- ADF: correct size, low power. Phillips-Perron: large size distortions when there is a negative MA component (Schwert) — avoid it as the only test.
- ADF lags: by AIC, BIC, the modified AIC (Ng-Perron) or the general-to-specific rule (start with a large p and reduce until the t-statistic of the last lag is ≥ 1.645), always on the sample of the largest model (`ts_univariate.adf_lags(y, rule=...)`). Too few lags distort size; too many reduce power. In simulations with the sieve bootstrap (n = 150), AIC kept the size (6%; 2.5% with MA(−0.5) errors) with full power against AR(0.5); BIC over-rejected with MA errors (10%); the t rule and MAIC kept the size but often stopped at long lags and lost a third of the power. AIC is the default.
- Deterministics: a constant if the series fluctuates around a level; constant + trend if it has a visible trend. Getting this wrong changes the result.
- Bootstrap critical values in small samples: estimate under H0 (Δy_t AR(p)), resample centered residuals, rebuild y*, recompute the statistic B times (5,000+ for a final report). `ts_univariate.adf_bootstrap(y, trend="c" | "ct", B=...)` does it and shows the asymptotic MacKinnon p-value next to the bootstrap one.
- **Power is low near the unit root**: against φ = 0.95 with 100 observations the ADF rejects in about 15% of samples (`simulation.demo_unit_root_power`). "Does not reject" is not evidence of a unit root; the forecast consequences differ (a stationary AR's forecast variance stops at σ²/(1 − φ²); a random walk's grows like h·σ², `simulation.demo_ar1_forecast_variance`).
- An unmodeled structural break makes a stationary series look I(1) → Zivot-Andrews (`arch.unitroot.ZivotAndrews`). Order: clean the series, look for breaks (section 4; on the differences if the series may be I(1), since a break test in levels assumes stationarity), then test for unit roots with the break modelled (`simulation.demo_break_unit_root`: a mean shift of 4 error SDs makes the ADF reject a stationary AR(0.7) in under 10% of samples; with the level dummy, in over 90%).

Practical view: for the most common macro series, the order of integration is usually known from the literature (interest rates and inflation often treated as persistent I(0) under inflation targeting; GDP, price levels and exchange rates in logs as I(1)) — do not reinvent the wheel. The choice is more methodological than statistical; report robustness under both versions.

`scripts/diagnostics.py stationarity` runs ADF, KPSS and DF-GLS in levels and differences, with a constant and with a trend, and suggests the order of integration.

## 4. Structural breaks

Breaks (pandemic, exchange-rate regime change, methodology change, a new product line) violate the constant-parameter assumption. Linked to the Lucas critique: parameters depend on the "rules of the game".

| Situation | Test | Python |
|---|---|---|
| Known date | Chow (F comparing subsamples); Wald with HAC covariances (robust) | `scripts/diagnostics.py chow` |
| Unknown date | Andrews sup-Wald (maximum Wald over all dates in [0.15T, 0.85T]; non-standard critical values) | `ts_univariate.sup_wald(y, X)` (sieve-bootstrap p-value, CUSUM, recursive coefficients) |
| Several breaks | Bai-Perron | `ruptures` package (detection) |
| Gradual instability | CUSUM / CUSUM-squared | `statsmodels.stats.diagnostic.breaks_cusumolsresid`, `recursive_olsresiduals` |
| Break near the end of the sample | Andrews (2003) end-of-sample test via subsampling | manual implementation |
| Unit root with a break | Zivot-Andrews | `arch.unitroot.ZivotAndrews` |

With autocorrelated residuals, use a **sieve bootstrap**: estimate the model without the break (the null), fit an AR(p) by BIC to its residuals, generate bootstrap errors from the AR, add them to the fitted values and recompute the statistic (fitting the AR to the residuals of the model with the break instead understates persistence and over-rejects: 12% at 5% with AR(0.8) errors and n = 150); `sup_wald` does this, choosing the AR order on a common sample. The date of the maximum is the break-date estimate only when the test rejects. In levels the test assumes a stationary series: on a random walk it rejects "no break" far too often (`sup_wald` warns when Zivot-Andrews does not reject a unit root); test the differences or use Zivot-Andrews. The OLS-CUSUM must be scaled by the long-run SD of the errors, not their iid SD: with AR(0.6) errors the iid version rejects a stable series about 60% of the time (`sup_wald` uses the long-run SD from its sieve). When the sup-F does not reject but that CUSUM does, the parameters may drift or change in several steps: say so instead of calling the series stable. A date estimated at the edge of the trimmed window (the last 15% of the sample) means the end of the sample differs, with an imprecise date and a short last regime; `sup_wald` warns.

Treatment: level/slope dummies, estimating only on the current regime, or regime-switching models (Markov switching: `statsmodels.tsa.regime_switching.markov_regression.MarkovRegression`).

## 5. Univariate models

**Box-Jenkins protocol**:
1. Transform (log if the variance grows with the level; Box-Cox).
2. Determine d (section 3) and seasonal D.
3. ACF/PACF of the stationary series (`ts_univariate.ts_identify(y)`: Bartlett bands for the ACF, ±1.96/√n for the PACF, Ljung-Box, cutoff suggestions and a BIC grid):

| Model | ACF | PACF |
|---|---|---|
| AR(p) | decays exponentially | cuts off after lag p |
| MA(q) | cuts off after lag q | decays exponentially |
| ARMA(p,q) | decays from lag q | decays from lag p |

4. Start with a large ARMA, estimate the smaller combinations and choose by **BIC** (more parsimonious and consistent; AIC tends to choose larger models). Do not drop a theoretically relevant lag just because it is not significant.
5. Check stationarity (AR roots outside the unit circle) and invertibility (MA roots outside the unit circle).
6. Residuals must be white noise: Ljung-Box (`acorr_ljungbox(res.resid, lags=[12, 24], model_df=p+q)`), no remaining ARCH effect.

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX
m = SARIMAX(y, order=(p, d, q), seasonal_order=(P, D, Q, 12)).fit(disp=False)
```

- A pure AR(p) can be estimated by OLS; with MA terms, maximum likelihood is needed. A long AR approximates any invertible ARMA — a practical shortcut.
- **ETS / exponential smoothing** (`statsmodels.tsa.exponential_smoothing.ets.ETSModel` or `holtwinters.ExponentialSmoothing`): in the M competitions (Makridakis and Hibon, 2000), simple methods such as damped-trend exponential smoothing and the Theta method matched or beat sophisticated models. Always compare.
- **ARFIMA (long memory)**: when neither ADF nor KPSS rejects (a classic case is monthly CPI inflation) and the ACF decays hyperbolically. (1 − L)^d y_t with −0.5 < d < 0.5 is stationary and invertible. In Python: estimate d with the Geweke-Porter-Hudak method (log-periodogram regression at low frequencies) or the local Whittle estimator (more efficient), filter (1 − L)^d with weights b_0 = 1, b_j = b_{j−1}(j − 1 − d)/j, and fit the ARMA on the filtered series: `ts_univariate.long_memory(y)` (both estimates with SEs π/√(24m) and 1/(2√m)), `frac_diff(y, d)` and `arfima(y, order=(p, q))`. The bandwidth trades bias against variance; compare a few. Short-run persistence biases d upward (an AR(1) with φ = 0.6 and 320 observations gives a local Whittle d near 0.2), so `long_memory` simulates the fitted short-memory AR(p) and reads d as long memory only when that is rejected; the check is conservative, has modest power in a few hundred observations, and never overrides a non-stationary reading.

## 6. Time-series regression

**Spurious regression** — what happens with OLS of y on x:

| y and x | OLS / t-test | Fix |
|---|---|---|
| Trend-stationary | R² and t inflated by the common trend | Include a trend or detrend both |
| I(1) with drift, not cointegrated | t does not converge, R² → 1, spurious correlation | Model in differences |
| I(1) without drift, not cointegrated | Invalid inference | Differences |
| I(1) cointegrated | Superconsistent coefficient; non-standard t | ECM (section 7) |

Warning sign: R² > Durbin-Watson.

**ADL** — effects that take time:

```python
df["d_sales"] = df["log_sales"].diff()
for k in range(0, 7):
    df[f"d_rate_l{k}"] = df["policy_rate"].diff().shift(k)
terms = " + ".join(f"d_rate_l{k}" for k in range(7))
m = smf.ols(f"d_sales ~ d_sales.shift(1) + {terms} + C(month)", data=df)\
       .fit(cov_type="HAC", cov_kwds={"maxlags": 6})
```

- Long-run multiplier = Σβ / (1 − Σφ); it is a ratio, so its standard error needs the delta method (a `t_test` on Σβ is only right without lags of y). `specification.long_run_effect(m, x_terms=[...], y_lag_terms=[...])` returns effect, SE, CI and p from the model's (HAC) covariance.
- **Finite distributed lag** (no lags of y): the long-run propensity LRP = Σβ_j is the effect of a permanent change; `long_run_effect(m, ["pe", "pe_1", "pe_2"])` reproduces Wooldridge Ex. 10.4 (.101, se .030). Lags of x are usually collinear, so individual β_j are imprecise while the LRP is well estimated — report the LRP.
- **Geometric (Koyck) lag**: y_t = α + γx_t + ρy_{t−1} + v_t implies weights γρ^j and LRP = γ/(1−ρ). The transformation makes v_t = u_t − ρu_{t−1} correlated with y_{t−1}, so OLS is inconsistent; instrument y_{t−1} with x_{t−1} (and test for serial correlation). Rational distributed lags (add x_{t−1}) relax the single-ρ shape.
- Lags by BIC or theory (monetary policy typically takes 6 to 12 months to affect activity).
- Underestimating p (the AR order) makes estimators inconsistent; overestimating only costs efficiency.
- **ARIMAX** (`SARIMAX(y, exog=X, order=...)`) when the residual has ARMA structure.

**Granger causality**: x Granger-causes y if lags of x improve the forecast of y. It is **predictive precedence**, not causation: a central bank that anticipates inflation makes rates "cause" inflation in the Granger sense. Write "x helps predict y".

## 7. Cointegration and ECM/VECM

Two I(1) series are cointegrated if a linear combination is I(0) — there is a long-run equilibrium. Examples: consumption and income, prices and dividends, PPP (exchange rate and prices), prices of substitutes (ethanol and gasoline, two share classes of the same company — the basis of long-short statistical arbitrage).

**Engle-Granger (three steps)**:
1. Confirm all series are I(1) (cointegration requires the same order).
2. Estimate the vector by OLS in levels (superconsistent); with different drifts, include a trend.
3. Test for a unit root in the residuals with MacKinnon's cointegration critical values — **not the ordinary ADF ones**. `statsmodels.tsa.stattools.coint(y, x)` does this.

**Johansen** (several series, several relationships): rewrites the VAR as ΔY_t = ΠY_{t−1} + ΣΓ_iΔY_{t−i} + ε_t and tests the rank of Π = αβ' (Granger representation theorem).

```python
from statsmodels.tsa.vector_ar.vecm import select_order, select_coint_rank, VECM
p = select_order(data, maxlags=12, deterministic="ci").bic       # lags in differences
r = select_coint_rank(data, det_order=0, k_ar_diff=p, method="trace", signif=0.05).rank
vecm = VECM(data, k_ar_diff=p, coint_rank=r, deterministic="ci").fit()
vecm.beta, vecm.alpha
```

- Most common deterministic cases: (a) no drift in the series, constant only in the cointegrating vector (`"ci"`); (b) series with drift and a constant (`"co"` or `"ci"` + `"lo"`); (c) constant and trend in the vector (`"li"`). The choice changes the result — justify it from the plot.
- Exogenous variables in Johansen: use them **centered and detrended**, including seasonal dummies (subtract the mean).
- Normalize β on the variable of interest and impose theoretical restrictions when they exist.
- **α** = adjustment speed; −0.2 → 20% of the deviation corrected per period; half-life = ln 0.5 / ln(1 − 0.2) ≈ 3.1 periods. The sign must correct the deviation.
- The ECM separates short and long run and reveals long-horizon predictability that simple regressions miss (Campbell-Shiller: dividend yield predicting returns).
- `scripts/diagnostics.py cointegration` runs Engle-Granger and Johansen (trace and max-eigenvalue).
- `multivariate.johansen_trace(df, k_ar_diff, deterministic="ci")` runs the trace test for the stated case with critical values simulated for it. statsmodels' `select_coint_rank` has no restricted-constant case: with `det_order=0` (unrestricted constant) on two cointegrated random walks without drift, it declared full rank ("stationary") in 9 of 20 samples; the simulated case-'ci' test rejected the true rank in 5% of 300. Cases: 'ci' (restricted constant), 'co' (unrestricted constant, levels drift; critical values simulated with the data's drift, slightly oversized in samples of a few hundred), 'li' (trend in the relation, constant outside: statsmodels' `"coli"`; statsmodels' `"li"` has no constant), 'n'. `multivariate.vecm_fit(df)` chooses the lags and rank, then checks that the system returns to β'y = 0: the implied levels VAR must have exactly k − r unit roots and the rest inside the unit circle (eig(I + β'α) inside the unit circle is the special case without lagged differences; with them it is neither necessary nor sufficient). Weak exogeneity is a joint Wald test that all loadings of a variable's equation are zero. With one relation, α_iβ_i < 0 means variable i moves to close the gap; with r > 1, β is identified only up to a rotation, so single α and β values depend on the normalisation until restrictions with economic meaning are imposed.

## 8. VAR

Y_t = c + Φ₁Y_{t−1} + … + Φ_pY_{t−p} + ε_t, with a non-diagonal Σ = Var(ε_t). Equation-by-equation OLS equals system GLS (same regressors).

```python
from statsmodels.tsa.api import VAR
var = VAR(stationary_data, exog=exog_vars)          # exogenous: commodities, dummies
print(var.select_order(12).summary())               # AIC, BIC, HQ, FPE
res = var.fit(p, trend="c")
res.is_stable()                                      # eigenvalues < 1 in modulus
res.test_whiteness(nlags=12, adjusted=True).summary()
res.test_normality().summary()
res.test_causality("sales", ["policy_rate"], kind="f").summary()
irf = res.irf(24); irf.plot(orth=True); irf.plot_cum_effects(orth=True)
res.fevd(24).summary()
```

- **Lags**: BIC is consistent, AIC tends to choose more. But if residuals remain autocorrelated, add lags — central banks do this in practice (e.g., a sectoral-inflation VAR using 2 lags even though BIC suggested 1, because of serial correlation).
- **Parameters grow fast**: k variables × p lags × k equations. A VAR with 9 variables and 4 lags has 333 parameters. With 120 months, keep k ≤ 4–5.
- **Orthogonalized (Cholesky) impulse responses**: the order matters — the first variable does not respond contemporaneously to the others. Put the more "exogenous" variables first and justify it (e.g., regulated prices → food → goods → services). Show robustness to other orderings.
- **Generalized IRF (Pesaran-Shin)**: does not depend on ordering, but does not allow reading "a 1% shock to x causes β% in y". Formula: ψ_j(h) = Φ_h Σ e_j / √σ_jj (not built into statsmodels; compute with `res.ma_rep(h)` and `res.sigma_u`).
- **IRF intervals**: `res.irf(h).plot(orth=True, stderr_type="mc", repl=1000)` (Monte Carlo) or bootstrap. Simple bootstrap intervals for IRFs can have poor coverage in small samples; prefer the bias-corrected bootstrap (Kilian, 1998) when the conclusion depends on it.
- **Variance decomposition**: which shocks explain the forecast-error variance of each variable; also depends on the orthogonalization.
- `multivariate.var_analysis(df)` runs the selection, stability, whiteness, normality and the Granger matrix, and warns on parameters per observation and on unit roots (by the joint ADF/DF-GLS/KPSS verdict of `diagnostics.stationarity`; `integration_order` holds it per series, and a trend-stationary series in a VAR without a trend is flagged); `irf_bands(res, h, ordering=[...])` gives Kilian bias-corrected bootstrap bands; `fevd_table(res, h, orderings=[...])` repeats the decomposition under each ordering and flags the shares that move by more than 10 points.
- **Price puzzle**: rates rise and prices rise afterwards, because the central bank anticipates inflation; adding commodity prices (or expectations) to the VAR mitigates it.
- A VAR with I(1) series in levels estimates consistently, but Granger tests and long-horizon IRFs become problematic; with cointegration, use a VECM; without it, use differences.

## 9. SVAR and identification

Structural form: B Y_t = A₀ + A₁Y_{t−1} + … + u_t, with uncorrelated structural shocks u_t (diagonal Ω). We estimate the reduced form and recover B by imposing restrictions.

| Strategy | Idea | Python | Cautions |
|---|---|---|---|
| Recursive (Cholesky) | Triangular B | `res.irf(...).orth_irfs` | The ordering needs an economic justification |
| Non-recursive short-run restrictions | Zeros in B from theory (e.g., the central bank sees output only with a lag) | `statsmodels.tsa.vector_ar.svar_model.SVAR(data, svar_type="AB", A=..., B=...)` | Count parameters: needs ≥ k(k−1)/2 restrictions |
| Long run (Blanchard-Quah) | A shock has no permanent effect (money neutrality): C₁₂(1) = 0 | Manual: C(1) = (I − ΣΦ_i)⁻¹; B₀⁻¹ = C(1)⁻¹·chol(C(1) Σ C(1)') | Requires knowing the exact unit root; does not pin down the sign; fragile with a very persistent I(0) variable; sensitive to levels vs. differences (Kilian, 2011) |
| Sign restrictions (Faust, Uhlig) | Demand shock: p↑ and q↑; supply: p↑ and q↓ | Manual: orthogonal rotations Q (QR decomposition of normal matrices), keep those satisfying the signs | Only set-identified; the pointwise median does not correspond to any model (Kilian); report the set or the most likely model |
| Exogenous regressors | Variables entering one equation and not another | VAR with `exog` | Exogeneity in macro is controversial (Sims, 1980) |
| Heteroskedasticity, external instruments (proxy-SVAR) | Volatility changes or an external shock series | Manual | Advanced literature |

`multivariate.svar_identify(k, A, B)` counts free parameters (order condition) and checks the rank of the Jacobian of vech(Σ) at random values (a model can pass the count and still not be identified); `svar_longrun(res)` does Blanchard-Quah; `svar_sign(res, {"demand": {"p": 1, "q": 1}})` keeps the rotations that satisfy the signs and reports the acceptance share, the pointwise median and the Fry-Pagan median-target model.

Example of a justified recursive identification (fuel market): gasoline price (most exogenous, set by policy) → sugar price → ethanol quantity → ethanol price (most endogenous).

General limitations: VAR shocks may reflect omitted variables; Lucas critique — IRFs hold within the same policy regime; breaks and time-varying parameters call for subsamples or TVP-VARs.

## 10. Bayesian VAR and data-rich VARs

- **BVAR with a Minnesota (Litterman) prior**: zero mean for coefficients except 1 on the own first lag (random walk); smaller variance for long lags and for other variables. Shrinks the excess of parameters; historically very good for forecasting. Normal-Wishart allows an analytical solution. Python: a simple manual implementation (OLS with artificial "dummy observations") or external packages.
- **FAVAR**: summarize dozens of series into a few factors (principal components) and include them in the VAR (Bernanke, Boivin and Eliasz).

## 11. Forecasting

**Evaluation**
- In-sample fit does not measure predictive ability. Use a **rolling origin** (expanding or sliding window), re-estimating at each step; for a simple split, a common rule is 80% estimation, 20% evaluation.

```python
import numpy as np
errors = []
for end in range(n_initial, len(y) - h + 1):
    m = SARIMAX(y.iloc[:end], order=order, seasonal_order=seasonal).fit(disp=False)
    errors.append(y.iloc[end:end + h].values - m.forecast(h).values)
rmse_by_horizon = np.sqrt(np.mean(np.square(errors), axis=0))
```

- Always compare with: naive (last value), seasonal naive (same month last year), mean, ETS. If the model does not beat the seasonal naive, say so.
- **The metric must match what you want to forecast** (Petropoulos et al., 2022): MSE/RMSE ↔ conditional mean; MAE ↔ median; MAPE undefined with zeros and asymmetric; MASE to compare different series. Evaluating the same point forecast with many metrics adds little.
- Intervals: check coverage (the share of outcomes inside a 95% interval should be ≈ 95%) and the *interval score*.
- **Diebold-Mariano test** (with the Harvey-Leybourne-Newbold small-sample correction) to tell whether the difference between two models is significant: `scripts/diagnostics.py` has `diebold_mariano`. For nested models DM is conservative (use Clark-West).
- Revised data (GDP, activity indices) and publication lags: use the series as available at the forecast date (vintages) to avoid overstating accuracy.

**Forecast intervals**
- Analytical formula: Var(e_{T+h}) = σ²(1 + ψ₁² + … + ψ²_{h−1}) — converges to the unconditional variance; assumes normality and ignores parameter uncertainty (tends to be too narrow).
- **Bootstrap**: estimate the AR(p), resample residuals, generate artificial series, re-estimate and forecast B times; the α/2 and 1 − α/2 percentiles form the interval. Captures non-normality and parameter uncertainty. `ts_univariate.ar_bootstrap_forecast(y, h)` shows it next to the analytic interval, with residuals rescaled for the degrees of freedom and Kilian's (1998) bias correction of the AR slopes (OLS slopes are biased toward zero in short samples, so plain bootstrap paths revert too fast). In simulations for v2.4.6 (AR(1), φ = 0.8, h = 4, n = 30) the 95% bootstrap interval covered 92% of outcomes, against 86% for the analytic interval and for the bootstrap without the correction; at h = 1 all were near 95%.

**Forecast combination**
- Combining forecasts from different methods usually beats the best individual method (M3; Petropoulos et al.). Equal weights are hard to beat in practice ("forecast combination puzzle").
- Alternatives: weights minimizing the combined error variance (Bates-Granger), Granger-Ramanathan regression (y on forecasts, with weights summing to 1), BIC/Akaike weights (a Bayesian approximation), including consensus forecasts (surveys of professional forecasters).
- `ts_univariate.combine_forecasts(actual, forecasts, method="equal" | "inverse_mse" | "min_variance" | "granger_ramanathan")` estimates the weights on a training window and reports every model and the combination on the rows after it; estimated weights can be negative and noisy, so keep equal weights unless the evaluation window says otherwise.

## 12. Common pitfalls

- Spurious regression in trending levels.
- Look-ahead: using revised data or data not yet published at the forecast date.
- Overfitting with many lags/variables in a short sample.
- Ignoring seasonality → autocorrelation at lags 12/24.
- An untreated break (2020) distorting coefficients and unit-root tests.
- Over-differencing: a differenced I(0) series gains a non-invertible MA (ACF of −0.5 at lag 1).
- Using the HP filter at the end of the sample for real-time decisions.
- Interpreting Granger causality as causation and Cholesky IRFs without justifying the ordering.
