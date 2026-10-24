---
name: econometrics
description: Rigorous applied econometrics in Python (statsmodels, linearmodels, arch) — regression with full diagnostics, IV, panel data, logit/probit/tobit, causal designs (matching/IPW/AIPW, double ML, regression discontinuity, staggered DiD/event studies, synthetic control), survival/Cox models, text as data (regex, TF-IDF, topics, classifiers, validation), time series (auto-ARIMA/ARIMAX, VAR, cointegration), GARCH, VaR/ES backtests, copulas, Kalman, regime switching, look-ahead-free backtests, demand elasticities, data collection and cleaning (BCB/IBGE/TSE/CNJ, scraping, PDFs), publication tables. Use for any regression, effect of X on Y, policy or law evaluation, elasticity, forecast, duration or time-to-event question, coding of documents into variables, VaR or tail risk, hedge ratio, regimes, strategy backtest, or "does X explain/cause/predict Y?" question, even without the word econometrics.
---

# Applied econometrics

This skill turns an economic question ("do interest rates hurt my sales?", "what is the price elasticity of this product?") into an estimated, diagnosed and defensible model. The value is in the **rigor of the process**: anyone can run `sm.OLS(...).fit()`; what separates a reliable analysis from a misleading one is the specification, the assumption tests, the identification strategy, and honesty about what the model does not prove.

The methodology follows the reference literature (Wooldridge, Greene, Hansen, Angrist-Pischke, Cameron-Trivedi, Hamilton, Enders, Stock, Kilian, Huse-Salvo) and the protocols taught in graduate econometrics and time-series courses. `references/bibliography.md` points to where each topic can be studied further.

Answer in the user's language; the skill's instructions are in English but the analysis, tables and report should match the conversation.

## Resources in this skill

| File | When to read |
|---|---|
| `scripts/diagnostics.py` | Whenever you estimate a model: test batteries with plain-language conclusions (OLS, influence, Chow, stationarity with ADF/KPSS/DF-GLS, cointegration, endogeneity, overidentification, Hausman, panel LM, Tobit, Diebold-Mariano) |
| `scripts/forecasting.py` | Forecasts: auto-ARIMA (statsmodels only), ARIMAX with lagged regressors, VAR forecasts, walk-forward evaluation, naive/seasonal/historical-mean benchmarks, DM and Pesaran-Timmermann tests |
| `scripts/backtest.py` | Turning a forecast or signal into positions: one-bar-lag backtest with costs, performance table, walk-forward splits, time-series momentum, Sharpe bootstrap CI |
| `scripts/risk.py` | VaR/ES (historical, normal/t, EWMA, filtered historical simulation, GARCH) and Kupiec/Christoffersen/Basel backtests |
| `scripts/dependence.py` | Rank correlations, tail dependence, copulas (Gaussian, t, Clayton, Gumbel, Frank, rotations) by MLE, copula simulation, mutual information, distance correlation |
| `scripts/state_space.py` | Kalman filter with time-varying coefficients (MLE of noise variances), half-life, Hurst, cointegration pretest, pairs positions and returns |
| `scripts/regimes.py` | Markov-switching models, Gaussian HMM with filtered probabilities, K-means on features, regime summaries |
| `scripts/causal.py` | Balance, propensity score, IPW, AIPW, matching, double ML, RD (bandwidth, robustness, density test, fuzzy), TWFE event study, Callaway-Sant'Anna, synthetic control with placebo inference |
| `scripts/survival.py` | Kaplan-Meier, log-rank, Cox with hazard ratios and clustered SE, proportional-hazards test, person-period expansion |
| `scripts/text_data.py` | Regex coding with audit snippets, TF-IDF, LDA, text clustering, TF-IDF + logit classifier, validation against hand coding, misclassification correction, similarity to a reference group |
| `scripts/data_prep.py` | Reading awkward files (encodings, `;` CSVs, XML/HTML ".xls", JSON, large CSVs), audit, safe merges, Brazilian numbers, mojibake, PDF text/OCR, polite scraping |
| `scripts/reporting.py` | Publication-style regression tables (outreg2/esttab style) with FE rows, odds/hazard ratios, export to Markdown/Excel/LaTeX |
| `references/ols-regression.md` | Every regression: functional forms, dummies, diagnostic order, HC/HAC/GLS, multicollinearity, influence, measurement error |
| `references/time-series.md` | Data with a time dimension: unit roots, breaks, ARIMA/ETS/ARFIMA, ADL, cointegration/VECM, VAR/SVAR/BVAR, forecast evaluation and combination |
| `references/panel-iv.md` | Endogeneity, instruments, simultaneous equations, panel data (FE/RE), difference-in-differences, dynamic panels |
| `references/limited-dependent.md` | Binary, count, censored, fractional or ordered y (LPM, logit, probit, tobit, Heckman, PPML) |
| `references/demand-elasticities.md` | Prices, promotions, cannibalization, cost pass-through, elasticities |
| `references/volatility.md` | Financial returns, exchange rates: stylized facts and GARCH family |
| `references/risk-var.md` | VaR and Expected Shortfall: estimators, backtests, model choice |
| `references/forecasting-backtesting.md` | Any out-of-sample forecast or model-based strategy: auto-ARIMA, ARIMAX, VAR forecasts, walk-forward, benchmarks, backtest conventions, momentum, multiple testing |
| `references/dependence-copulas.md` | Joint behavior beyond correlation: tail dependence, copulas, mutual information |
| `references/state-space-regimes.md` | Time-varying coefficients (Kalman), pairs trading, Markov switching, HMM, clustering of regimes |
| `references/causal-inference.md` | Any causal question without randomization: choosing a design, matching/weighting, double ML, RD, staggered DiD, synthetic control, law-and-economics designs (judge leniency, movers) |
| `references/survival.md` | Time until an event, with censoring: Kaplan-Meier, Cox, PH checks, competing risks |
| `references/text-as-data.md` | Turning documents into variables and validating them; using text-based measures in regressions |
| `references/data-collection-wrangling.md` | Messy files, large files, merges, scraping, PDFs/OCR, Brazilian administrative and judicial data |
| `references/data-sources.md` | Fetching data (FRED, World Bank, IMF, OECD, BCB/SGS, IBGE/SIDRA, Ipeadata) |
| `references/software-equivalents.md` | Translating Stata/EViews procedures into Python |
| `references/bibliography.md` | Going deeper on a topic in the reference literature |

Read only the references the case needs. The scripts are starting points: read the code and adapt it when needed. Import them with `sys.path.insert(0, '<skill>/scripts')`; `forecasting.evaluate` imports `diagnostics`.

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
For causal designs and text: `scikit-learn` (required by `causal.aipw/dml_plr` and `text_data`); `rdrobust` (recommended for RD inference). Optional depending on the case: `pandas-datareader`/`fredapi`, `wbgapi`, `python-bcb`, `sidrapy`, `yfinance` (data), `ruptures` (multiple breaks), `pydynpd` (dynamic panels), `scikit-learn` (ROC/AUC, cross-validation, K-means, mutual information), `hmmlearn` (hidden Markov models). `pmdarima` is not needed: `forecasting.auto_arima` covers it.

## Workflow

Follow the steps in order; each exists because skipping it produces a specific kind of error. Explain to the user what is being done at each step and why.

### 1. Frame the question and the hypothesis

Before touching the data, write down: dependent variable, regressors, expected sign of each coefficient and why (theory or business logic). This prevents "fishing" across specifications until something is significant.

Classify the goal, because it changes the right model:
- **Estimate an effect** (elasticity, policy impact): the focus is an unbiased coefficient → identification, endogeneity, omitted variables.
- **Estimate a causal effect without an experiment**: choose the design first (selection on observables, RD, DiD/event study, synthetic control, IV) and write its identifying assumption in words → `causal-inference.md`.
- **Forecast**: the focus is out-of-sample error → time-based validation and comparison with naive benchmarks. A model can forecast well without its coefficients having a causal interpretation.

### 2. Identify the data structure

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
| Raw files, web pages, PDFs | TSE CSVs, court websites | `data-collection-wrangling.md` |
| Returns, volatility | Daily stock index | `volatility.md` |
| Market risk | VaR/ES of a position or portfolio | `risk-var.md` |
| Joint extremes | Two assets crashing together, contagion | `dependence-copulas.md` |
| Changing relationships or regimes | Time-varying hedge ratio, calm vs. crisis | `state-space-regimes.md` |
| Forecast or trading signal | Yield forecast, VAR signal, momentum | `forecasting-backtesting.md` |

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
```

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
- for forecasting: rolling-origin evaluation against naive, seasonal naive and ETS, with a Diebold-Mariano test;
- for strategies and risk models: out-of-sample period, realistic costs, number of variants tried, VaR backtests (see `forecasting-backtesting.md` §9 and `risk-var.md` §3).

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
- **No look-ahead**: anything used to forecast, signal or risk-manage at t is computed with data up to t (lagged regressors, training-only scalers and fits, filtered not smoothed probabilities, positions applied to the next period's return).
- **No invented numbers**: every coefficient, statistic or data point comes from a computation run in the session or a cited source. If data are missing, say so.

Tables: `reporting.regression_table({"(1)": m1, "(2)": m2}, keep=[...], fe_rows={"Year FE": ["YES", "YES"]})` produces the side-by-side table (coefficients, SEs in parentheses, stars, N, R², FE rows; `exponentiate=True` for odds/hazard ratios) and `export_table` writes Markdown, Excel or LaTeX. Rescale regressors whose coefficients print as 1e-05 (`reporting.rescale_hint`), and mark a fixed effect YES only where it is in that column's model.

## Deliverables

- Quick analysis: answer in the conversation with the main table and summarized diagnostics.
- Analysis to keep or share: a report with the sections above and a reproducible `.py` script.
- For study or reuse: a step-by-step commented `.ipynb` notebook.
- Charts that communicate the result: observed vs. fitted, residuals over time, impulse responses with bands, forecasts with intervals, marginal effects.
