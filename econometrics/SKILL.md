---
name: econometrics
description: Rigorous applied econometrics workflow in Python (statsmodels, linearmodels, arch) — regression with full diagnostics (RESET, heteroskedasticity, autocorrelation, influence), endogeneity and instrumental variables, panel data, logit/probit/tobit, time series (unit roots, structural breaks, ARIMA, VAR/SVAR, cointegration/VECM), volatility (GARCH), forecasting with out-of-sample evaluation, and demand and price-elasticity estimation, with data from central banks and statistical offices (FRED, World Bank, IMF, OECD, Brazil's BCB/IBGE/Ipeadata). Use whenever the user asks for a regression, a price elasticity, the effect of one variable on another, the impact of interest rates, inflation or exchange rates on sales, a time-series forecast, a stationarity or cointegration test, a panel model, a default probability, a promotion's effect, or any "does X explain/cause Y?" question, even without the word "econometrics".
---

# Applied econometrics

This skill turns an economic question ("do interest rates hurt my sales?", "what is the price elasticity of this product?") into an estimated, diagnosed and defensible model. The value is in the **rigor of the process**: anyone can run `sm.OLS(...).fit()`; what separates a reliable analysis from a misleading one is the specification, the assumption tests, the identification strategy, and honesty about what the model does not prove.

The methodology follows the reference literature (Wooldridge, Greene, Hansen, Angrist-Pischke, Cameron-Trivedi, Hamilton, Enders, Stock, Kilian, Huse-Salvo) and the protocols taught in graduate econometrics and time-series courses. `references/bibliography.md` points to where each topic can be studied further.

Answer in the user's language; the skill's instructions are in English but the analysis, tables and report should match the conversation.

## Resources in this skill

| File | When to read |
|---|---|
| `scripts/diagnostics.py` | Whenever you estimate a model: test batteries with plain-language conclusions (OLS, influence, Chow, stationarity with ADF/KPSS/DF-GLS, cointegration, endogeneity, overidentification, Hausman, panel LM, Tobit, Diebold-Mariano) |
| `references/ols-regression.md` | Every regression: functional forms, dummies, diagnostic order, HC/HAC/GLS, multicollinearity, influence, measurement error |
| `references/time-series.md` | Data with a time dimension: unit roots, breaks, ARIMA/ETS/ARFIMA, ADL, cointegration/VECM, VAR/SVAR/BVAR, forecast evaluation and combination |
| `references/panel-iv.md` | Endogeneity, instruments, simultaneous equations, panel data (FE/RE), difference-in-differences, dynamic panels |
| `references/limited-dependent.md` | Binary, count, censored, fractional or ordered y (LPM, logit, probit, tobit, Heckman, PPML) |
| `references/demand-elasticities.md` | Prices, promotions, cannibalization, cost pass-through, elasticities |
| `references/volatility.md` | Financial returns, exchange rates, VaR |
| `references/data-sources.md` | Fetching data (FRED, World Bank, IMF, OECD, BCB/SGS, IBGE/SIDRA, Ipeadata) |
| `references/software-equivalents.md` | Translating Stata/EViews procedures into Python |
| `references/bibliography.md` | Going deeper on a topic in the reference literature |

Read only the references the case needs. The script is a starting point: read the code and adapt it when needed.

## Combining with other skills

- A statistical-analysis/EDA skill, if available, for the initial exploration (distributions, outliers, correlation ≠ causation caveats); this skill takes over at the modeling stage.
- The available data-visualization skill for charts (observed vs. fitted, residuals, impulse responses, forecasts with intervals).
- The skill for the delivery format (spreadsheet, document, slides) when the result becomes a file.
- Any "no invented numbers" skill, if available, as reinforcement: no figure without a computation or a source.

## Environment

```bash
python3 -c "import statsmodels, linearmodels, arch" 2>/dev/null || \
  pip install --break-system-packages -q statsmodels linearmodels arch
```
Optional depending on the case: `pandas-datareader`/`fredapi`, `wbgapi`, `python-bcb`, `sidrapy` (data), `ruptures` (multiple breaks), `pydynpd` (dynamic panels), `scikit-learn` (ROC/AUC, cross-validation).

## Workflow

Follow the steps in order; each exists because skipping it produces a specific kind of error. Explain to the user what is being done at each step and why.

### 1. Frame the question and the hypothesis

Before touching the data, write down: dependent variable, regressors, expected sign of each coefficient and why (theory or business logic). This prevents "fishing" across specifications until something is significant.

Classify the goal, because it changes the right model:
- **Estimate an effect** (elasticity, policy impact): the focus is an unbiased coefficient → identification, endogeneity, omitted variables.
- **Forecast**: the focus is out-of-sample error → time-based validation and comparison with naive benchmarks. A model can forecast well without its coefficients having a causal interpretation.

### 2. Identify the data structure

| Structure / type of y | Example | Reference |
|---|---|---|
| Cross-section, continuous y | 500 stores in one month | `ols-regression.md` |
| Time series | Monthly sales 2015–2026 | `time-series.md` |
| Panel | 80 stores × 48 months | `panel-iv.md` |
| Binary, count or censored y | Default, number of orders, spending with zeros | `limited-dependent.md` |
| Price and quantity | Elasticity, promotion | `demand-elasticities.md` |
| Returns | Daily stock index | `volatility.md` |

### 3. Obtain and audit the data

- Sources in `data-sources.md`. Record source, series code, extraction date and unit.
- **Frequency and alignment**: aggregate consistently (monthly average policy rate ≠ end-of-month rate).
- **Nominal vs. real**: deflate monetary values over long horizons (CPI for consumption).
- **Seasonality, holidays, events and breaks** (pandemic, methodology changes, a new menu or product line).
- **Sample size**: with fewer than ~30 observations per parameter, warn that inference is fragile.
- **Transformations**: logs for elasticities and variance; differences for non-stationary series.
- Keep the raw data and do all cleaning in code (documented exclusions).

### 4. Specify and estimate

Start with the simplest model that answers the question; add complexity only when a diagnostic or theory justifies it.

```python
import numpy as np, statsmodels.formula.api as smf
m = smf.ols("np.log(sales) ~ np.log(price) + np.log(income) + C(month)", data=df).fit(cov_type="HC3")
# time series: cov_type="HAC", cov_kwds={"maxlags": int(4*(len(df)/100)**(2/9))}
# panel/groups: clustered errors (see panel-iv.md)
```

Robust errors by default: heteroskedasticity and autocorrelation do not bias coefficients, but they invalidate standard errors and p-values. HC3 is preferable to HC1 (Hansen, 2020).

### 5. Diagnose

```bash
python3 scripts/diagnostics.py ols data.csv --y log_sales --x log_price,log_income [--hac 6]
python3 scripts/diagnostics.py influence data.csv --y log_sales --x log_price,log_income
python3 scripts/diagnostics.py stationarity data.csv --cols policy_rate,cpi,log_sales
python3 scripts/diagnostics.py cointegration data.csv --cols log_p,log_fx --lags 2
```
or `from diagnostics import ...` in Python.

Order (heteroskedasticity and autocorrelation are often symptoms of a wrong specification, so functional form comes first):

| Assumption | Test | If it fails |
|---|---|---|
| Functional form | Ramsey RESET | Logs, quadratics, interactions; re-estimate before moving on |
| Homoskedasticity | Breusch-Pagan, White (full or with ŷ and ŷ²) | HC3; WLS/FGLS if the form is known |
| No autocorrelation | Breusch-Godfrey, Durbin-Watson | HAC; model the dynamics (lags) |
| Normality | Jarque-Bera | Only matters in small samples |
| Multicollinearity | VIF, condition number | Combine/drop variables, or accept it if it only affects controls |
| Influence | Leverage, DFBETA, max\|h·ẽ\| | Check for data errors; report with and without |
| Stationarity (time series) | ADF + KPSS + DF-GLS | Difference or test for cointegration |
| Stability | Chow, CUSUM, sup-Wald | Break dummies or subsamples |
| Exogeneity | Theory + Durbin-Wu-Hausman with an instrument | IV/2SLS, panel FE, quasi-experiment |

**Spurious regression**: trending series (revenue and GDP in levels) almost always yield a high R² and significant t-statistics with no real relationship. Warning sign: R² > Durbin-Watson.

### 6. Test robustness

A conclusion is reliable only if it survives reasonable variations. Do at least two:
- alternative specification (controls, functional form, lags);
- subsample (without the pandemic; separate periods);
- another estimator (OLS vs. IV; FE vs. RE; 2SLS vs. LIML; alternative Cholesky ordering);
- another variable definition or aggregation level;
- for forecasting: rolling-origin evaluation against naive, seasonal naive and ETS, with a Diebold-Mariano test.

Present them side by side (`comparison_table` in the script, paper style).

### 7. Interpret and report

Structure:
1. **Question and short answer**, with magnitude and interval: "A 1 p.p. increase in the policy rate is associated with a 0.8% to 1.4% drop in sales after 6 months (95% CI)."
2. **Data**: sources, period, frequency, transformations, exclusions and why.
3. **Model**: equation, estimator, standard-error type, identification strategy.
4. **Results**: coefficients, standard errors, CIs, N, adjusted R² (or pseudo-R²), information criteria when comparing models.
5. **Diagnostics**: test, statistic, p-value, conclusion.
6. **Robustness**: what changed and what did not.
7. **Limitations**: what the model does not prove.

Interpretation rules:
- **By functional form** (table in `ols-regression.md`): log-log → elasticity; log-lin → 100·β% (exact: 100·(e^β − 1)); lin-log → β/100 per 1%; quadratic → effect β₁ + 2β₂x.
- **In nonlinear models** (logit, probit, tobit), report average marginal effects, not raw coefficients.
- **Statistical ≠ economic significance**: report magnitude and interval, not only the p-value.
- **Association ≠ causation**: causal language only with an identification strategy (IV, DiD, experiment, discontinuity). Granger causality is predictive precedence. Otherwise, "is associated with".
- **Direction of bias**: when identification is imperfect, say which way the bias probably goes.
- **Honest precision**: 2–3 significant digits; forecasts always with intervals; elasticities hold for small changes within the observed range.
- **No invented numbers**: every coefficient, statistic or data point comes from a computation run in the session or a cited source. If data are missing, say so.

## Deliverables

- Quick analysis: answer in the conversation with the main table and summarized diagnostics.
- Analysis to keep or share: a report with the sections above and a reproducible `.py` script.
- For study or reuse: a step-by-step commented `.ipynb` notebook.
- Charts that communicate the result: observed vs. fitted, residuals over time, impulse responses with bands, forecasts with intervals, marginal effects.
