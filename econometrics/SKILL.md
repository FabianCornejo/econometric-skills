---
name: econometrics
description: Rigorous applied econometrics in Python, as consultant, teacher or reviewer - regression with full diagnostics, IV, panel data, logit/probit/tobit, causal designs (matching/AIPW, double ML, RD, staggered DiD, synthetic control), heterogeneous effects (CATE, causal forests, meta-learners, BLP/GATES, targeting policies), custom maximum likelihood, delta method, wild cluster bootstrap, survival, text as data and LLM coding with valid inference, time series (ARIMA, VAR, cointegration), GARCH, VaR/ES, copulas, Kalman, regimes, backtests, demand elasticities, Brazilian data, publication tables. Use for any regression, effect of X on Y, for whom an effect is larger or whom to target, policy or law evaluation, elasticity, forecast, time-to-event, coding documents into variables, churn segments, tail risk, strategy backtest, teaching or explaining an econometric method, or reviewing someone else's empirical analysis, even without the word econometrics.
---

# Applied econometrics

This skill turns an economic question into an estimated, diagnosed and defensible answer, and explains or audits such answers. The value is in the **rigor of the process**: anyone can run `sm.OLS(...).fit()`; what makes an analysis reliable is the specification, the assumption tests, the identification strategy and honesty about what the model does not prove. Methods follow Wooldridge, Greene, Hansen, Angrist-Pischke, Cameron-Trivedi, Hamilton, Chernozhukov et al. and EconML (`references/bibliography.md`).

Answer in the user's language; these instructions are in English.

## 1. Pick the mode

| Mode | When | Deliverable | Read |
|---|---|---|---|
| **Consultant** (default) | Estimate, test, forecast, evaluate a policy | Study card, analysis, report + reproducible script | `references/workflow.md`, `references/interpretation-rules.md` |
| **Teacher** | "Explain", "teach", "why does...", a class, exam or exercises | Method card: intuition, model, assumptions, estimator, derivation, simulation, code, diagnostics, interpretation, pitfalls, exercise | `references/teaching.md` |
| **Reviewer** | Review, audit or referee a paper, report, expert opinion or model | Opinion: claim, verdict, findings by severity, what would change the verdict | `references/review-checklist.md` |

Modes change the form of the answer, never the computations: every number comes from the same scripts and rules.

## 2. Route the question

Answer four questions, then read only the files they point to: (1) the **goal**: an effect, a causal effect without an experiment, for whom the effect is larger or whom to treat, a forecast, a risk measure; (2) the **data structure and type of y**; (3) the **source of identification**: experiment, observed confounders, instrument, cutoff, adoption date, single treated unit; (4) **can the effect vary across units?**

| Situation | Reference | Main functions |
|---|---|---|
| Cross-section, continuous y | `ols-regression.md` | `diagnostics.ols_diagnostics`, `specification.*` |
| Time series (levels, lags, cointegration, VAR) | `time-series.md` | `diagnostics.stationarity`, `cointegration` |
| Panel data, endogeneity, instruments | `panel-iv.md` | `diagnostics.mundlak_test`, `endogeneity_test`, `overid_test` |
| Binary, count, censored, fractional or ordered y | `limited-dependent.md` | `diagnostics.tobit`, `count_diagnostics` |
| Prices and quantities, elasticities, pass-through | `demand-elasticities.md` | IV recipes |
| Effect of a policy, law, decision or program (non-random) | `causal-inference.md` | `causal.aipw`, `dml_plr`, `rdd`, `att_gt`, `synth` |
| For whom the effect is larger; whom to treat | `heterogeneous-effects.md` | `hte.cate_fit`, `blp_test`, `gates`, `policy_tree` |
| Custom likelihood, NLS, function of parameters, few clusters | `estimation-theory.md` | `estimation.mle`, `delta_method`, `wild_cluster_bootstrap` |
| Time until an event, with censoring | `survival.md` | `survival.cox`, `ph_test` |
| Documents into variables (incl. LLM coding) | `text-as-data.md` | `text_data.llm_classify`, `dsl_regression` |
| Binary y with segments whose drivers differ (prediction) | `limited-dependent.md` §9 | `segments.logit_leaf_model` |
| Messy files, scraping, PDFs; fetching data | `data-collection-wrangling.md`, `data-sources.md` | `data_prep.*` |
| Returns and volatility | `volatility.md` | `arch` recipes |
| VaR and Expected Shortfall | `risk-var.md` | `risk.*` |
| Joint extremes, contagion | `dependence-copulas.md` | `dependence.fit_copulas` |
| Time-varying relations, regimes, pairs | `state-space-regimes.md` | `state_space.*`, `regimes.*` |
| Forecasts and model-based strategies | `forecasting-backtesting.md` | `forecasting.*`, `backtest.*` |
| Translating Stata, EViews or R code | `software-equivalents.md` | |

## 3. Consultant workflow

Full detail, code and the diagnostic order are in `references/workflow.md`. Explain each step and why it is done.

0. **Study card** before any estimate: question, estimand, design, identifying assumption in words, y, X (effect modifiers), W (confounders), Z (instruments), data, threats, robustness plan.
1. **Frame** the hypothesis and expected signs; classify the goal (effect, causal effect, heterogeneity, forecast).
2. **Identify the structure** of the data (table above).
3. **Obtain and audit** the data: source and code, frequency, nominal vs real, breaks, missing values (`data_prep.missing_report`), N per parameter.
4. **Specify and estimate**: the simplest model that answers the question; robust SEs by default.
5. **Diagnose** in order: functional form, heteroskedasticity, autocorrelation, ARCH, normality, collinearity, influence, stationarity, stability, exogeneity.
6. **Test robustness**: at least two variations (specification, subsample, estimator, definition, sensitivity to unobserved confounding).
7. **Interpret and report**: answer with magnitude and interval, data, model, results, diagnostics, robustness, limitations.

## 4. Non-negotiable rules

Full versions and reasons in `references/interpretation-rules.md`.

- **Design before data**: choose the estimand, design, covariates, bandwidths and windows before looking at outcomes.
- **DiD**: ask whether the adoption date is exact (and its source), announced in advance or gradual; run `causal.did_timing_sensitivity` next to `att_gt`; relay any timing WARNING in plain words.
- **Robust SEs by default** (HC3; HAC for time series). **Cluster where the regressor of interest varies** (a judge's sex: by judge, not by court), the same level in text and code; report the number of clusters; with fewer than ~40, add `estimation.wild_cluster_bootstrap`.
- **Association is not causation**: causal language only with an identification strategy; Granger causality is predictive precedence; say which way remaining bias probably goes.
- **Statistical is not economic significance**: magnitude and interval, 2-3 significant digits, forecasts with intervals.
- **Interpret by functional form** (log-log elasticity, log-lin 100*(e^b - 1)%); **average marginal effects** in nonlinear models (Tobit: say which margin); **long-run effects** with delta-method SEs for lags; **prediction vs confidence intervals**, with smearing for log-y.
- **No look-ahead** in anything used to forecast, signal or manage risk at time t.
- **LLM-coded variables are measurements with error**: hand-code a random sample with known probabilities and use `text_data.dsl_regression`; report naive and corrected side by side.
- **Segmented logit**: benchmarks must contain every regressor of the leaf model; report segments only if they recur across seeds.
- **Heterogeneous effects only with evidence**: `hte.blp_test` rejects and `hte.gates` separates groups, on out-of-fold predictions; compare a linear and a flexible model; value policies on held-out units.
- **Run more than one model** for causal claims (simple structural and flexible); disagreement is a finding.
- **Advice must match the tools**: never recommend a step and show code that does something else; if no function exists, say so and write the code.
- **No invented numbers**: every figure comes from a computation run in the session or a cited source.

## 5. Output contract

New and refactored functions return the same shapes so results can be compared and tabulated: a single estimand is a dict with `estimand`, `estimate`, `se`, `ci_low`, `ci_high`, `p`, `n` (plus `method` and `warnings`); several parameters come as a DataFrame `table` with columns `estimate`, `se`, `z` or `t`, `p`, `ci_low`, `ci_high`. Relay every entry of `warnings` to the user. Tables for reports: `reporting.regression_table` and `export_table` (Markdown, Excel, LaTeX).

## 6. Scripts

Import with `sys.path.insert(0, '<skill>/scripts')`. They are tested starting points: read and adapt them.

| Script | Contents |
|---|---|
| `diagnostics.py` | OLS battery, influence, Chow, ADF/KPSS/DF-GLS, cointegration, endogeneity, overidentification, Hausman, panel LM, Mundlak/CRE, Tobit with three margins, count overdispersion, Diebold-Mariano |
| `specification.py` | FGLS, CI vs prediction intervals with smearing, long-run effects, standardized coefficients, exact OVB, Oster delta, Cinelli-Hazlett robustness value |
| `estimation.py` | Generic MLE (Newton/BFGS; Hessian, OPG, sandwich, cluster SEs), LR/Wald/LM trinity, NLS, delta method, pairs/cluster bootstrap, wild cluster bootstrap-t with CI |
| `causal.py` | Balance, propensity score, IPW, AIPW, matching, double ML, RD with robustness, TWFE event study, Callaway-Sant'Anna, DiD timing sensitivity, synthetic control |
| `hte.py` | CATE (DR-learner, linear DML, forest, T/S/X-learners), BLP calibration test, GATES, CLAN, TOC/AUTOC/Qini, R-score, policy trees and policy value |
| `forecasting.py`, `backtest.py` | Auto-ARIMA, ARIMAX, VAR forecasts, walk-forward, benchmarks, DM/PT tests; look-ahead-free backtests, momentum, Sharpe CI |
| `risk.py`, `dependence.py` | VaR/ES (historical, parametric, EWMA, FHS, GARCH) and backtests; copulas, tail dependence, mutual information |
| `state_space.py`, `regimes.py` | Kalman time-varying coefficients, half-life, pairs; Markov switching, HMM, K-means regimes |
| `survival.py` | Kaplan-Meier, log-rank, Cox with clustered SEs, PH test, person-period data |
| `text_data.py` | Regex coding, TF-IDF, LDA, classifiers, validation, LLM coding with probabilities and uncertainty, DSL regression |
| `segments.py` | Logit leaf model with honest splits, benchmarks, temporal holdout, calibration, stability |
| `data_prep.py`, `reporting.py` | Awkward files, audits, missing-data report, safe merges, Brazilian numbers, PDFs, scraping; regression tables |

## 7. Environment

```bash
python3 -c "import statsmodels, linearmodels, arch, sklearn" 2>/dev/null || \
  pip install --break-system-packages -q statsmodels linearmodels arch scikit-learn
```
Optional by case: `rdrobust` (RD inference), `hmmlearn`, `ruptures`, `pydynpd`, `econml` (causal forests with CIs, IV and dynamic CATE), `transformers` + `torch` (local LLM coding), data clients (`python-bcb`, `sidrapy`, `wbgapi`, `fredapi`, `yfinance`).

## 8. Combining with other skills and deliverables

Use an EDA skill for initial exploration, the data-visualization skill for charts (observed vs fitted, residuals, impulse responses, GATES, TOC curves, forecasts with intervals), the delivery-format skill when the result becomes a file, and any "no invented numbers" skill as reinforcement. Deliverables: a quick answer in the conversation with the main table and summarized diagnostics; a report with the sections of step 7 plus a reproducible `.py` script for work to keep or share; a commented `.ipynb` notebook for study or reuse.
