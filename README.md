# econometric-skills

[![tests](https://github.com/FabianCornejo/econometric-skills/actions/workflows/tests.yml/badge.svg)](https://github.com/FabianCornejo/econometric-skills/actions/workflows/tests.yml)

Agent skills for Claude focused on applied econometrics.

## `econometrics`

A rigorous applied-econometrics workflow in Python (statsmodels, linearmodels, arch): regression with full diagnostics, endogeneity and instrumental variables, panel data, logit/probit/tobit, time series (unit roots, structural breaks, ARIMA, VAR/SVAR, cointegration/VECM), volatility (GARCH), forecasting with out-of-sample evaluation, and demand and price-elasticity estimation, with data from FRED, the World Bank, IMF, OECD and Brazil's BCB, IBGE and Ipeadata.

**Financial econometrics (v1.1):**
- **Forecasting:** auto-ARIMA without pmdarima, ARIMAX with properly lagged regressors, walk-forward VAR forecasts, naive/seasonal/historical-mean benchmarks, Diebold-Mariano and Pesaran-Timmermann tests.
- **Risk:** VaR and Expected Shortfall (historical, normal/t, EWMA, filtered historical simulation, rolling GARCH) with Kupiec, Christoffersen and Basel traffic-light backtests.
- **Dependence:** rank correlations, tail dependence, copulas (Gaussian, Student-t, Clayton, Gumbel, Frank and rotations) by maximum likelihood, copula simulation, mutual information on a correlation scale, distance correlation.
- **State space and regimes:** Kalman-filter time-varying regression with ML-estimated noise, pairs-trading layer, Markov-switching models, HMMs with filtered (real-time) probabilities, K-means regimes.
- **Backtesting:** one-bar-lag convention, transaction costs, performance table, walk-forward splits, time-series momentum, Sharpe confidence intervals.

**Causal inference, survival, text and data (v1.2):**
- **Causal designs:** balance tables, propensity scores, IPW, doubly robust AIPW, nearest-neighbour matching, double/debiased ML, regression discontinuity (rdrobust/IK bandwidths, fuzzy RD, robustness package, density test), TWFE event studies, Callaway-Sant'Anna staggered DiD, synthetic control with placebo inference.
- **DiD timing robustness (v1.3):** anticipation in Callaway-Sant'Anna, automatic warnings when the outcome moves before the recorded date, a date-shift sensitivity table with a date-robust ATT, and estimation when each unit's adoption date is only known within a window (laws phased in court by court).
- **Survival:** Kaplan-Meier, log-rank, Cox with hazard ratios and clustered SEs, proportional-hazards test, discrete-time hazard data.
- **Text as data:** regex coding with audit snippets, TF-IDF, LDA, clustering with representative documents, classifiers with proper validation, agreement with hand coding, misclassification correction, similarity to a reference group.
- **Data and reporting:** robust readers for Brazilian files (`;`, latin-1, decimal comma, XML/HTML ".xls"), large CSVs, safe merges, PDF/OCR, polite scraping; outreg2/esttab-style regression tables.

**Textbook replication and core fixes (v1.4):**
- **Replication suite:** `tests/test_textbook_replication.py` checks the skill's functions and the recipes in its references against published results from Wooldridge's *Introductory Econometrics* (Examples 8.4, 13.3, 14.4, 15.1, 15.5, 15.7, 15.8, 17.2, 17.3), alongside the known-truth simulations.
- **Diagnostics:** Durbin-Watson and Breusch-Godfrey run only for time series (in a cross-section the row order is arbitrary); time series also get Engle's ARCH-LM. CLI flag `--ts`.
- **Tobit:** lower, upper or two-limit censoring; the three average marginal effects (total, intensive and extensive margins) with delta-method standard errors.
- **Stata parity:** documented `debiased=True` for textbook/Stata `small` IV standard errors.

**Specification and sensitivity tools (v1.5):**
- **New module `specification.py`:** FGLS for heteroskedasticity of unknown form; confidence vs. prediction intervals (with Duan smearing for log-y models); long-run effects of distributed lags and ADL models with delta-method SEs; standardized (beta) coefficients; exact omitted-variable-bias decomposition; Oster (2019) coefficient stability; Cinelli-Hazlett (2020) robustness values with covariate benchmarks.
- **Diagnostics:** Mundlak/correlated-random-effects test (FE vs RE with clustered errors); Poisson overdispersion (Pearson dispersion, Cameron-Trivedi NB1/NB2 tests, excess zeros).
- **Validated** in `tests/test_specification_tools.py` against Wooldridge Examples 6.1, 6.5–6.6, 8.7, 10.4, 14.4 and 17.3, the Darfur example of Cinelli and Hazlett (2020), and known-truth simulations.

**Packaging, CI and examples (v1.6):**
- **Worked examples** (`examples/`): five end-to-end analyses on Wooldridge data following the 7-step workflow — returns to education with sensitivity analysis, IV, panel FE/RE/CRE, distributed lags, Tobit and Poisson — as scripts and executed notebooks.
- **Missing data:** `data_prep.missing_report` (patterns, complete-case N, test of whether observables predict missingness) and guidance on listwise deletion, multiple imputation and what to avoid, verified by simulation.
- **Dependencies:** `requirements.txt` with lower bounds that the whole test suite was actually run on; `requirements-dev.txt` for tests and examples.
- **Continuous integration:** GitHub Actions runs every test suite and every example on minimum and latest dependency versions, and checks that `dist/econometrics.skill` matches the source.

**LLM coding with valid inference and segmented logit (v1.7):**
- **LLM coding of documents:** answers constrained to a closed label set with a probability per category (`llm_classify`, any backend that returns logits; `hf_backend` runs a Hugging Face model locally, so personal data stay on the machine), a flag for documents where the model wanted to answer outside the codebook, and token-level aleatoric/epistemic uncertainty (LogTokU, Ma et al. 2025).
- **Valid regressions on LLM-coded variables:** `dsl_regression` (OLS or logit, coded outcome and/or regressors) combines LLM codes for every document with a hand-coded subsample drawn with known probabilities (design-based supervised learning, Egami et al. 2023), with sandwich or clustered SEs; `dsl_sampling_probs` oversamples uncertain documents without breaking the design. In simulations with group-dependent coding errors, treating LLM codes as truth understates the effect by 13–32%; the corrected estimator is unbiased with nominal coverage.
- **Logit leaf model** (`segments.py`; De Caigny, Coussement & De Bock 2018): a shallow tree defines segments and a logit is fit in each, with honest sample splitting for valid leaf SEs, AMEs by segment, and holdout AUC and top-decile lift against a pooled logit and the tree.
- **Validated** in `tests/test_llm_annotation_segments.py` (40 checks: closed-form LogTokU, exact identities of the estimator, Monte Carlo bias and coverage, known two-segment process); `hf_backend` was also run end to end on Qwen2.5-0.5B-Instruct.

**Robustness of the v1.7 tools (v1.8):**
- **`dsl_regression`:** checks that every regressor varies and the design has full rank among the hand-coded documents (and, for a logit, that both outcomes appear), naming the cause instead of failing with a singular matrix; the classic trigger is a draw seed shared with the data-generating code. A separated hand-coded logit no longer yields a spurious design effect near zero: the benchmark is reported as NaN with a warning and the DSL estimate is kept.
- **Logit leaf model:** `benchmarks` scores theory-based logits (e.g. interactions with customer type) on the same holdout, `res["best"]` names the model to report, `segment_by` warns when the tree never splits on the groups the user asked about (CART splits on outcome rates, not slopes), and `leaf_model_stability` reports split-variable frequency and model ranking across seeds.
- **Validated** in `tests/test_llm_annotation_segments.py` (55 checks), including a churn process whose slopes differ by customer type: the tree never splits on it and the interaction benchmark wins in every seed.

**Temporal holdout, calibration and categorical segments (v1.9):**
- **Logit leaf model:** `time_col` makes the holdout the most recent period (cut at a distinct value, no period split), so drift shows up instead of being averaged away; `performance` adds the Brier score, calibration slope and mean bias, with a calibration table by decile per model (`segments.calibration` is also available on its own); `segment_by` accepts text or categorical columns (dummies created, rebuilt on prediction, unseen categories flagged); benchmarks that omit a regressor of the leaf model are flagged as not like for like.
- **Instructions:** cluster at the level where the regressor of interest varies (the judge for a judge's characteristic) in both the text and the code; never recommend a step without code that performs it.
- **Evals** (`evals/`): blind-test prompts with objective pass criteria, run in a fresh conversation before each release.
- **Validated** in `tests/test_llm_annotation_segments.py` (66 checks): holdout strictly later than training, drift detected as negative mean bias, calibration slope 0.99 for true probabilities and 0.50 for doubled logits, dummy splits recognized.

**Consultant, Teacher and Reviewer modes; estimation engine; heterogeneous effects (v2.0):**
- **`SKILL.md` as a router** (214 → 111 lines): three modes (Consultant by default, Teacher, Reviewer), a routing table from the question to the reference and functions, the non-negotiable rules in one line each, and a shared output contract. The full workflow (now with a step 0 "study card": estimand, design, identifying assumption, effect modifiers X vs confounders W vs instruments Z) and the interpretation rules moved to `references/workflow.md` and `references/interpretation-rules.md`, unchanged in substance.
- **Teacher mode** (`references/teaching.md`): an eleven-block method card (intuition → model → assumptions → estimator → derivation → simulation → code → diagnostics → interpretation → pitfalls → exercise), three levels, a catalogue of simulations that show each assumption failing, and a curriculum order following Hill-Griffiths-Lim. **Reviewer mode** (`references/review-checklist.md`): a referee checklist and an opinion format with findings by severity.
- **New module `estimation.py`:** maximum likelihood from any per-observation log-likelihood (Newton-Raphson with step halving or BFGS; Hessian, OPG, sandwich and clustered covariances; convergence and gradient checks), LR/Wald/LM tests of linear restrictions, nonlinear least squares, delta method for any vector function of the parameters, pairs and cluster bootstrap, and the **restricted wild cluster bootstrap-t** with Webb weights and a test-inversion CI — the skill already recommended it for few clusters but had no function for it.
- **New module `hte.py`** (after EconML and grf, numpy/scikit-learn only): CATE by DR-learner and linear DML with valid CIs, a forest DR-learner and T/S/X meta-learners, all with cross-fitted nuisances and out-of-fold predictions; validation by the BLP/calibration test, GATES, CLAN (Holm-adjusted), TOC/AUTOC/Qini with bootstrap SEs and the R-score; exhaustive depth-1/2 policy trees on doubly robust scores, learned on one half and valued on the other, with the cost of treatment. `references/heterogeneous-effects.md` covers estimands, identification, choosing an estimator, reading each tool, policy learning and using EconML/grf predictions with these validation tools.
- **Validated** in `tests/test_estimation.py` (28 checks: Wooldridge Examples 17.1 probit and 17.3 Poisson by hand-written likelihood, equality with statsmodels for coefficients, Hessian and sandwich SEs, LR and Wald statistics; NLS = OLS for a linear f; delta method vs simulation; with 8 clusters, clustered SEs reject a true null ~20% of the time and the wild cluster bootstrap ~5–7%) and `tests/test_hte.py` (40 checks: recovery of tau(x) = 1 + x for binary and continuous treatments, ATE CI coverage 94% and BLP size 3.5% in Monte Carlo, no false heterogeneity under a constant effect, a linear CATE missing symmetric heterogeneity that the forest detects, a policy tree recovering "treat if x1 > 0"), on both the minimum and the latest dependency versions.

Every rolling estimate uses only data available at the time. Look-ahead traps (contemporaneous regressors, full-sample scaling, smoothed probabilities, filtered Kalman residuals, signal/return misalignment) are avoided in code and documented in the references.

The skill's instructions are in English; Claude answers in the user's language.

```
econometrics/
├── SKILL.md                         router: modes, routing table, rules, output contract
├── scripts/
│   ├── diagnostics.py               test batteries with plain-language conclusions
│   ├── specification.py             FGLS, prediction intervals, long-run effects, OVB and sensitivity
│   ├── estimation.py                MLE, LR/Wald/LM, NLS, delta method, bootstrap, wild cluster bootstrap
│   ├── forecasting.py               auto-ARIMA, ARIMAX, VAR forecasts, walk-forward evaluation
│   ├── backtest.py                  look-ahead-free backtests, performance, momentum
│   ├── risk.py                      VaR/ES estimators and backtests
│   ├── dependence.py                copulas, tail dependence, mutual information
│   ├── state_space.py               Kalman filter, half-life, pairs trading
│   ├── regimes.py                   Markov switching, HMM, K-means regimes
│   ├── causal.py                    matching/IPW/AIPW, double ML, RD, event study, CS-DiD + timing checks, synthetic control
│   ├── hte.py                       CATE learners, BLP, GATES, CLAN, TOC/Qini, R-score, policy trees
│   ├── survival.py                  Kaplan-Meier, Cox, PH test
│   ├── text_data.py                 regex, TF-IDF, topics, classifiers, validation, LLM coding, DSL regression
│   ├── segments.py                  logit leaf model, theory benchmarks, temporal holdout, calibration, stability
│   ├── data_prep.py                 reading, auditing, merging, PDFs, scraping
│   └── reporting.py                 publication-style regression tables
└── references/
    ├── workflow.md                  Consultant mode: study card + 7 steps
    ├── interpretation-rules.md
    ├── teaching.md                  Teacher mode
    ├── review-checklist.md          Reviewer mode
    ├── ols-regression.md
    ├── time-series.md
    ├── panel-iv.md
    ├── limited-dependent.md
    ├── demand-elasticities.md
    ├── volatility.md
    ├── risk-var.md
    ├── forecasting-backtesting.md
    ├── dependence-copulas.md
    ├── state-space-regimes.md
    ├── causal-inference.md
    ├── heterogeneous-effects.md
    ├── estimation-theory.md
    ├── survival.md
    ├── text-as-data.md
    ├── data-collection-wrangling.md
    ├── data-sources.md
    ├── software-equivalents.md      Stata/EViews/R → Python
    └── bibliography.md
tests/                               284 checks: textbook replications, known-truth simulations, Monte Carlo size and coverage
```

### Installation

- **Claude.ai / desktop app:** download [`econometrics.skill`](https://github.com/FabianCornejo/econometric-skills/raw/main/dist/econometrics.skill) (also in the [latest release](../../releases/latest)) and upload it under Settings → Capabilities → Skills (or zip the `econometrics` folder yourself).
- **Claude Code:** copy the `econometrics` folder into `~/.claude/skills/`.

Python dependencies: `pip install -r requirements.txt` (optional extras are listed in that file; `pip install -r requirements-dev.txt` adds what the tests and examples need).

### Quick test of the script

```bash
python3 econometrics/scripts/diagnostics.py ols your_data.csv --y log_sales --x log_price,log_income
python3 econometrics/scripts/diagnostics.py stationarity your_data.csv --cols policy_rate,cpi
python3 econometrics/scripts/forecasting.py your_data.csv --col yield --season 12
python3 tests/test_financial_modules.py
python3 tests/test_causal_text_survival.py
python3 tests/test_textbook_replication.py   # needs: pip install wooldridge
python3 tests/test_specification_tools.py    # needs: pip install wooldridge
python3 tests/test_missing_data.py
python3 tests/test_estimation.py          # needs: pip install wooldridge
python3 tests/test_hte.py
```

## Feedback and contributions

The skill is open for use, testing and suggestions:

- **Found a bug or a strange result?** Open an [issue](../../issues) describing the question asked, the data (or a simulated example) and the answer you got.
- **Suggestion for a method, test or data source?** Also via an issue.
- **Want to fix or extend something?** Fork the repository and open a pull request.

## License

[MIT](LICENSE) — free to use, including commercially, keeping the copyright notice.
