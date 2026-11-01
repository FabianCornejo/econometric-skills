# Workflow (Consultant mode)

The seven steps behind every analysis. Follow them in order; each exists because skipping it produces a specific kind of error. Explain to the user what is being done at each step and why. Interpretation and reporting rules are in `interpretation-rules.md`.

## 0. Write the study card (before any estimate)

Five to ten lines, shown to the user and kept at the top of the report. It is a small pre-analysis plan: it fixes the choices that would otherwise drift toward whatever gives a significant result.

| Field | Content |
|---|---|
| Question | One sentence, with the decision it informs |
| Estimand | ATE, ATT, LATE, effect at a cutoff, CATE tau(x), elasticity, forecast at horizon h, VaR at level a |
| Design | Selection on observables, IV, RD, DiD/event study, synthetic control, time-series model, experiment |
| Identifying assumption | In plain words (what must be true for the number to mean what we say) |
| Variables | y; treatment or regressor of interest; **X** = effect modifiers (where the effect may vary); **W** = confounders (controls that only remove bias); **Z** = instruments |
| Data | Source, unit, period, frequency, N, level of clustering |
| Threats | Confounders not measured, reverse causality, selection, measurement error, look-ahead |
| Robustness plan | At least two checks chosen now (step 6) |

Separating X from W (EconML's convention) prevents a common error: reading the coefficient of a control as if it were an effect. A control is in the model to clean the coefficient of interest, not to be interpreted.

Follow the steps in order; each exists because skipping it produces a specific kind of error. Explain to the user what is being done at each step and why.

## 1. Frame the question and the hypothesis

Before touching the data, write down: dependent variable, regressors, expected sign of each coefficient and why (theory or business logic). This prevents "fishing" across specifications until something is significant.

Classify the goal, because it changes the right model:
- **Estimate an effect** (elasticity, policy impact): the focus is an unbiased coefficient → identification, endogeneity, omitted variables.
- **Estimate a causal effect without an experiment**: choose the design first (selection on observables, RD, DiD/event study, synthetic control, IV) and write its identifying assumption in words → `causal-inference.md`.
  - **DiD/event study: settle the treatment date before estimating.** Ask the user whether the adoption date is exact (and its source), whether it was announced or anticipated, and whether adoption was gradual within units. Never assume the date is right because it is in the data. Then run `causal.did_timing_sensitivity` (or `did_uncertain_dates` when only a window is known) next to `att_gt`, and report the date-robust result with the main one (`causal-inference.md` §6.1). If `att_gt` or `event_study` prints a timing WARNING, tell the user in plain words before reporting any effect.
- **Estimate for whom the effect is larger, or whom to treat** (targeting a discount, a program, an intervention): the focus is the CATE tau(x) and a policy rule, with the same identification as the ATE plus evidence that the heterogeneity is real → `heterogeneous-effects.md`.
- **Forecast**: the focus is out-of-sample error → time-based validation and comparison with naive benchmarks. A model can forecast well without its coefficients having a causal interpretation.

## 2. Identify the data structure

| Structure / type of y | Example | Reference |
|---|---|---|
| Cross-section, continuous y | 500 stores in one month | `ols-regression.md` |
| Time series | Monthly sales 2015–2026 | `time-series.md` |
| Panel | 80 stores × 48 months | `panel-iv.md` |
| Binary, count or censored y | Default, number of orders, spending with zeros | `limited-dependent.md` |
| Price and quantity | Elasticity, promotion | `demand-elasticities.md` |
| Effect of a policy, law, decision or program (non-random) | Effect of a reform on case duration | `causal-inference.md` |
| Time until an event, with censored spells | Time to decision, tenure, churn, default | `survival.md` |
| Documents (decisions, contracts, tenders, news) | Outcome of health-litigation decisions | `text-as-data.md` |
| Binary y with segments that behave differently | Churn drivers by customer type | `limited-dependent.md` §9 |
| Effect that varies across units; whom to treat | Which customers respond to a discount; policy rule with a cost | `heterogeneous-effects.md` |
| Likelihood or nonlinear model not packaged; functions of parameters; few clusters | Custom count model, elasticity at a point, 8 states | `estimation-theory.md` |
| Groups and experiments; robust, quantile or permutation inference; many regressors | A/B/n test, outliers, effect on the tails of demand | `ols-regression.md` §15 |
| Beta, cost of equity, factor model, event study, fund performance, portfolio | WACC beta from comparables, CAR around an M&A announcement | `empirical-finance.md` |
| Raw files, web pages, PDFs | TSE CSVs, court websites | `data-collection-wrangling.md` |
| Returns, volatility | Daily stock index | `volatility.md` |
| Market risk | VaR/ES of a position or portfolio | `risk-var.md` |
| Joint extremes | Two assets crashing together, contagion | `dependence-copulas.md` |
| Changing relationships or regimes | Time-varying hedge ratio, calm vs. crisis | `state-space-regimes.md` |
| Forecast or trading signal | Yield forecast, VAR signal, momentum | `forecasting-backtesting.md` |

## 3. Obtain and audit the data

- Sources in `data-sources.md`. Record source, series code, extraction date and unit.
- **Frequency and alignment**: aggregate consistently (monthly average policy rate ≠ end-of-month rate).
- **Nominal vs. real**: deflate monetary values over long horizons (CPI for consumption).
- **Seasonality, holidays, events and breaks** (pandemic, methodology changes, a new menu or product line).
- **Sample size**: with fewer than ~30 observations per parameter, warn that inference is fragile.
- **Transformations**: logs for elasticities and variance; differences for non-stationary series.
- **Missing values**: run `data_prep.missing_report(df, cols, predictors)` before estimating. Dropping incomplete rows is consistent if missingness depends only on the regressors, biased if it depends on y; under MAR in the regressors use multiple imputation (`data-collection-wrangling.md`, "Missing data"). Never mean-impute by default; report N per model.
- Keep the raw data and do all cleaning in code (documented exclusions).

## 4. Specify and estimate

Start with the simplest model that answers the question; add complexity only when a diagnostic or theory justifies it.

```python
import numpy as np, statsmodels.formula.api as smf
m = smf.ols("np.log(sales) ~ np.log(price) + np.log(income) + C(month)", data=df).fit(cov_type="HC3")
# time series: cov_type="HAC", cov_kwds={"maxlags": int(4*(len(df)/100)**(2/9))}
# panel/groups: clustered errors (see panel-iv.md); fewer than ~40 clusters: estimation.wild_cluster_bootstrap
```

Robust errors by default: heteroskedasticity and autocorrelation do not bias coefficients, but they invalidate standard errors and p-values. HC3 is preferable to HC1 (Hansen, 2020).

## 5. Diagnose

```bash
python3 scripts/diagnostics.py ols data.csv --y log_sales --x log_price,log_income [--hac 6 | --ts]
python3 scripts/diagnostics.py influence data.csv --y log_sales --x log_price,log_income
python3 scripts/diagnostics.py stationarity data.csv --cols policy_rate,cpi,log_sales
python3 scripts/diagnostics.py cointegration data.csv --cols log_p,log_fx --lags 2
```
or `from diagnostics import ...` in Python. For forecasting and strategies:
```bash
python3 scripts/forecasting.py data.csv --col yield --season 12      # auto-ARIMA + walk-forward vs naive
```
```python
from risk import var_garch, var_ewma, backtest_var            # VaR models + Kupiec/Christoffersen
from dependence import fit_copulas                            # tail dependence beyond correlation
from state_space import fit_kalman_regression                 # time-varying beta / hedge ratio
from regimes import markov_switching                          # regimes with filtered probabilities
from backtest import run_backtest, performance                # one-bar lag, costs, Sharpe, drawdown
from specification import fgls_het, prediction_intervals, long_run_effect, oster, sensitivity
from diagnostics import mundlak_test, count_diagnostics       # FE vs RE (clustered); Poisson variance
```

Order (heteroskedasticity and autocorrelation are often symptoms of a wrong specification, so functional form comes first):

| Assumption | Test | If it fails |
|---|---|---|
| Functional form | Ramsey RESET | Logs, quadratics, interactions; re-estimate before moving on |
| Homoskedasticity | Breusch-Pagan, White (full or with ŷ and ŷ²) | HC3; WLS/FGLS if the form is known |
| No autocorrelation (time series only; row order is arbitrary in a cross-section) | Breusch-Godfrey, Durbin-Watson (`time_series=True`) | HAC; model the dynamics (lags) |
| No ARCH (time series) | Engle ARCH-LM on residuals | HC/HAC still valid; GARCH if intervals or risk matter (`volatility.md`) |
| Normality | Jarque-Bera | Only matters in small samples |
| Multicollinearity | VIF, condition number | Combine/drop variables, or accept it if it only affects controls |
| Influence | Leverage, DFBETA, max\|h·ẽ\| | Check for data errors; report with and without |
| Stationarity (time series) | ADF + KPSS + DF-GLS | Difference or test for cointegration |
| Stability | Chow, CUSUM, sup-Wald | Break dummies or subsamples |
| Exogeneity | Theory + Durbin-Wu-Hausman with an instrument | IV/2SLS, panel FE, quasi-experiment |
| FE vs. RE (panel) | Mundlak/CRE with clustered Wald (`mundlak_test`); classical Hausman only as a check | FE, or report the CRE model |
| Poisson variance (counts) | Pearson dispersion, Cameron-Trivedi (`count_diagnostics`) | Robust or quasi-Poisson SEs; NB if count probabilities matter |

**Spurious regression**: trending series (revenue and GDP in levels) almost always yield a high R² and significant t-statistics with no real relationship. Warning sign: R² > Durbin-Watson.

## 6. Test robustness

A conclusion is reliable only if it survives reasonable variations. Do at least two:
- alternative specification (controls, functional form, lags);
- subsample (without the pandemic; separate periods);
- another estimator (OLS vs. IV; FE vs. RE; 2SLS vs. LIML; alternative Cholesky ordering);
- another variable definition or aggregation level;
- for causal claims resting on observed controls: sensitivity to unobserved confounders (`specification.oster` and/or `sensitivity`, `ols-regression.md` §14);
- for forecasting: rolling-origin evaluation against naive, seasonal naive and ETS, with a Diebold-Mariano test;
- for strategies and risk models: out-of-sample period, realistic costs, number of variants tried, VaR backtests (see `forecasting-backtesting.md` §9 and `risk-var.md` §3).

Present them side by side (`comparison_table` in the script, paper style).

## 7. Interpret and report

Structure:
1. **Question and short answer**, with magnitude and interval: "A 1 p.p. increase in the policy rate is associated with a 0.8% to 1.4% drop in sales after 6 months (95% CI)."
2. **Data**: sources, period, frequency, transformations, exclusions and why.
3. **Model**: equation, estimator, standard-error type, identification strategy.
4. **Results**: coefficients, standard errors, CIs, N, adjusted R² (or pseudo-R²), information criteria when comparing models.
5. **Diagnostics**: test, statistic, p-value, conclusion.
6. **Robustness**: what changed and what did not.
7. **Limitations**: what the model does not prove.

Then apply every rule in `interpretation-rules.md`.
