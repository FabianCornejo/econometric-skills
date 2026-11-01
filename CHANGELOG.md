# Changelog

All releases of the `econometrics` skill, newest first. Each release is tagged on GitHub with the packaged `econometrics.skill` attached ([releases](https://github.com/FabianCornejo/econometric-skills/releases)).

## v2.1 — Classical linear models, empirical finance, randomization inference and a simulation lab

- **New module `linear_models.py`** (Faraway; Dalpiaz): ANOVA types I–III with partial eta-squared, Tukey/Holm/BH pairwise comparisons, ANCOVA with a parallel-slopes test, Huber/bisquare robust regression, quantile regression with a bootstrap test of equal slopes, Freedman-Lane permutation test, pure-error lack-of-fit test, Box-Cox with exact profile-likelihood bounds, partial residuals, ridge/lasso/elastic net/PCR/PLS compared by nested CV, post-double-selection lasso (Belloni-Chernozhukov-Hansen) and best subsets.
- **New module `finance_empirics.py`** (Ang 2015; Tsay 2013; Campbell-Lo-MacKinlay): total and log returns, market model and CAPM with HAC SEs, rolling betas, Blume and Vasicek adjustments, Hamada unlever/relever with debt beta, factor models (Fama-French, NEFIN for Brazil), event studies with cross-sectional t, Patell, BMP and Corrado tests, performance ratios (Sharpe with Lo SE, Sortino, Treynor, IR, Jensen), Markowitz frontier with Ledoit-Wolf covariance. `references/empirical-finance.md` covers WACC betas from comparables, event-study design and Brazilian data (B3, CDI, NEFIN).
- **`causal.randomization_inference`**: Fisher randomization p-value and test-inversion CI for a DiD with few treated units (exact enumeration when assignments are few), for the TWFE statistic or any user statistic — the check that the v2.0 reviewer eval asked for.
- **New module `simulation.py`** for Teacher mode: seeded demos of omitted-variable bias, bad controls, measurement error, heteroskedasticity, spurious regression, weak instruments, few clusters and staggered DiD, a generic `monte_carlo` engine and `power_mde`.
- **Validated** in `tests/test_linear_models.py` (29 checks), `tests/test_finance_empirics.py` (32) and `tests/test_simulation_ri.py` (19): agreement with scipy/statsmodels, recovery of known parameters, Monte Carlo sizes and coverage (permutation test 6%, Box-Cox CI 97%, post-double-selection 95%, randomization inference 5.5% with 6 of 27 units treated), and the classic event-study result that Patell over-rejects under event-induced variance (31%) while BMP does not (4%). Along the way, the Box-Cox CI was fixed to use exact bounds instead of grid points, and the documented wild-bootstrap size with 8 clusters was corrected to ~7% from 1,000 replications.

## v2.0 — Consultant, Teacher and Reviewer modes; estimation engine; heterogeneous effects

- **`SKILL.md` as a router** (214 → 111 lines): three modes (Consultant by default, Teacher, Reviewer), a routing table from the question to the reference and functions, the non-negotiable rules in one line each, and a shared output contract. The full workflow (now with a step 0 "study card": estimand, design, identifying assumption, effect modifiers X vs confounders W vs instruments Z) and the interpretation rules moved to `references/workflow.md` and `references/interpretation-rules.md`, unchanged in substance.
- **Teacher mode** (`references/teaching.md`): an eleven-block method card (intuition → model → assumptions → estimator → derivation → simulation → code → diagnostics → interpretation → pitfalls → exercise), three levels, a catalogue of simulations that show each assumption failing, and a curriculum order following Hill-Griffiths-Lim. **Reviewer mode** (`references/review-checklist.md`): a referee checklist and an opinion format with findings by severity.
- **New module `estimation.py`:** maximum likelihood from any per-observation log-likelihood (Newton-Raphson with step halving or BFGS; Hessian, OPG, sandwich and clustered covariances; convergence and gradient checks), LR/Wald/LM tests of linear restrictions, nonlinear least squares, delta method for any vector function of the parameters, pairs and cluster bootstrap, and the **restricted wild cluster bootstrap-t** with Webb weights and a test-inversion CI — the skill already recommended it for few clusters but had no function for it.
- **New module `hte.py`** (after EconML and grf, numpy/scikit-learn only): CATE by DR-learner and linear DML with valid CIs, a forest DR-learner and T/S/X meta-learners, all with cross-fitted nuisances and out-of-fold predictions; validation by the BLP/calibration test, GATES, CLAN (Holm-adjusted), TOC/AUTOC/Qini with bootstrap SEs and the R-score; exhaustive depth-1/2 policy trees on doubly robust scores, learned on one half and valued on the other, with the cost of treatment. `references/heterogeneous-effects.md` covers estimands, identification, choosing an estimator, reading each tool, policy learning and using EconML/grf predictions with these validation tools.
- **Validated** in `tests/test_estimation.py` (28 checks: Wooldridge Examples 17.1 probit and 17.3 Poisson by hand-written likelihood, equality with statsmodels for coefficients, Hessian and sandwich SEs, LR and Wald statistics; NLS = OLS for a linear f; delta method vs simulation; with 8 clusters, clustered SEs reject a true null ~20% of the time and the wild cluster bootstrap ~5–7%) and `tests/test_hte.py` (40 checks: recovery of tau(x) = 1 + x for binary and continuous treatments, ATE CI coverage 94% and BLP size 3.5% in Monte Carlo, no false heterogeneity under a constant effect, a linear CATE missing symmetric heterogeneity that the forest detects, a policy tree recovering "treat if x1 > 0"), on both the minimum and the latest dependency versions.

## v1.9 — Temporal holdout, calibration and categorical segments

- **Logit leaf model:** `time_col` makes the holdout the most recent period (cut at a distinct value, no period split), so drift shows up instead of being averaged away; `performance` adds the Brier score, calibration slope and mean bias, with a calibration table by decile per model (`segments.calibration` is also available on its own); `segment_by` accepts text or categorical columns (dummies created, rebuilt on prediction, unseen categories flagged); benchmarks that omit a regressor of the leaf model are flagged as not like for like.
- **Instructions:** cluster at the level where the regressor of interest varies (the judge for a judge's characteristic) in both the text and the code; never recommend a step without code that performs it.
- **Evals** (`evals/`): blind-test prompts with objective pass criteria, run in a fresh conversation before each release.
- **Validated** in `tests/test_llm_annotation_segments.py` (66 checks): holdout strictly later than training, drift detected as negative mean bias, calibration slope 0.99 for true probabilities and 0.50 for doubled logits, dummy splits recognized.

## v1.8 — Robustness of the v1.7 tools

- **`dsl_regression`:** checks that every regressor varies and the design has full rank among the hand-coded documents (and, for a logit, that both outcomes appear), naming the cause instead of failing with a singular matrix; the classic trigger is a draw seed shared with the data-generating code. A separated hand-coded logit no longer yields a spurious design effect near zero: the benchmark is reported as NaN with a warning and the DSL estimate is kept.
- **Logit leaf model:** `benchmarks` scores theory-based logits (e.g. interactions with customer type) on the same holdout, `res["best"]` names the model to report, `segment_by` warns when the tree never splits on the groups the user asked about (CART splits on outcome rates, not slopes), and `leaf_model_stability` reports split-variable frequency and model ranking across seeds.
- **Validated** in `tests/test_llm_annotation_segments.py` (55 checks), including a churn process whose slopes differ by customer type: the tree never splits on it and the interaction benchmark wins in every seed.

## v1.7 — LLM coding with valid inference and segmented logit

- **LLM coding of documents:** answers constrained to a closed label set with a probability per category (`llm_classify`, any backend that returns logits; `hf_backend` runs a Hugging Face model locally, so personal data stay on the machine), a flag for documents where the model wanted to answer outside the codebook, and token-level aleatoric/epistemic uncertainty (LogTokU, Ma et al. 2025).
- **Valid regressions on LLM-coded variables:** `dsl_regression` (OLS or logit, coded outcome and/or regressors) combines LLM codes for every document with a hand-coded subsample drawn with known probabilities (design-based supervised learning, Egami et al. 2023), with sandwich or clustered SEs; `dsl_sampling_probs` oversamples uncertain documents without breaking the design. In simulations with group-dependent coding errors, treating LLM codes as truth understates the effect by 13–32%; the corrected estimator is unbiased with nominal coverage.
- **Logit leaf model** (`segments.py`; De Caigny, Coussement & De Bock 2018): a shallow tree defines segments and a logit is fit in each, with honest sample splitting for valid leaf SEs, AMEs by segment, and holdout AUC and top-decile lift against a pooled logit and the tree.
- **Validated** in `tests/test_llm_annotation_segments.py` (40 checks: closed-form LogTokU, exact identities of the estimator, Monte Carlo bias and coverage, known two-segment process); `hf_backend` was also run end to end on Qwen2.5-0.5B-Instruct.

## v1.6 — Packaging, CI and examples

- **Worked examples** (`examples/`): five end-to-end analyses on Wooldridge data following the 7-step workflow — returns to education with sensitivity analysis, IV, panel FE/RE/CRE, distributed lags, Tobit and Poisson — as scripts and executed notebooks.
- **Missing data:** `data_prep.missing_report` (patterns, complete-case N, test of whether observables predict missingness) and guidance on listwise deletion, multiple imputation and what to avoid, verified by simulation.
- **Dependencies:** `requirements.txt` with lower bounds that the whole test suite was actually run on; `requirements-dev.txt` for tests and examples.
- **Continuous integration:** GitHub Actions runs every test suite and every example on minimum and latest dependency versions, and checks that `dist/econometrics.skill` matches the source.

## v1.5 — Specification and sensitivity tools

- **New module `specification.py`:** FGLS for heteroskedasticity of unknown form; confidence vs. prediction intervals (with Duan smearing for log-y models); long-run effects of distributed lags and ADL models with delta-method SEs; standardized (beta) coefficients; exact omitted-variable-bias decomposition; Oster (2019) coefficient stability; Cinelli-Hazlett (2020) robustness values with covariate benchmarks.
- **Diagnostics:** Mundlak/correlated-random-effects test (FE vs RE with clustered errors); Poisson overdispersion (Pearson dispersion, Cameron-Trivedi NB1/NB2 tests, excess zeros).
- **Validated** in `tests/test_specification_tools.py` against Wooldridge Examples 6.1, 6.5–6.6, 8.7, 10.4, 14.4 and 17.3, the Darfur example of Cinelli and Hazlett (2020), and known-truth simulations.

## v1.4 — Textbook replication and core fixes

- **Replication suite:** `tests/test_textbook_replication.py` checks the skill's functions and the recipes in its references against published results from Wooldridge's *Introductory Econometrics* (Examples 8.4, 13.3, 14.4, 15.1, 15.5, 15.7, 15.8, 17.2, 17.3), alongside the known-truth simulations.
- **Diagnostics:** Durbin-Watson and Breusch-Godfrey run only for time series (in a cross-section the row order is arbitrary); time series also get Engle's ARCH-LM. CLI flag `--ts`.
- **Tobit:** lower, upper or two-limit censoring; the three average marginal effects (total, intensive and extensive margins) with delta-method standard errors.
- **Stata parity:** documented `debiased=True` for textbook/Stata `small` IV standard errors.

## v1.2 — Causal inference, survival, text and data

- **Causal designs:** balance tables, propensity scores, IPW, doubly robust AIPW, nearest-neighbour matching, double/debiased ML, regression discontinuity (rdrobust/IK bandwidths, fuzzy RD, robustness package, density test), TWFE event studies, Callaway-Sant'Anna staggered DiD, synthetic control with placebo inference.
- **DiD timing robustness (v1.3):** anticipation in Callaway-Sant'Anna, automatic warnings when the outcome moves before the recorded date, a date-shift sensitivity table with a date-robust ATT, and estimation when each unit's adoption date is only known within a window (laws phased in court by court).
- **Survival:** Kaplan-Meier, log-rank, Cox with hazard ratios and clustered SEs, proportional-hazards test, discrete-time hazard data.
- **Text as data:** regex coding with audit snippets, TF-IDF, LDA, clustering with representative documents, classifiers with proper validation, agreement with hand coding, misclassification correction, similarity to a reference group.
- **Data and reporting:** robust readers for Brazilian files (`;`, latin-1, decimal comma, XML/HTML ".xls"), large CSVs, safe merges, PDF/OCR, polite scraping; outreg2/esttab-style regression tables.

## v1.1 — Financial econometrics

- **Forecasting:** auto-ARIMA without pmdarima, ARIMAX with properly lagged regressors, walk-forward VAR forecasts, naive/seasonal/historical-mean benchmarks, Diebold-Mariano and Pesaran-Timmermann tests.
- **Risk:** VaR and Expected Shortfall (historical, normal/t, EWMA, filtered historical simulation, rolling GARCH) with Kupiec, Christoffersen and Basel traffic-light backtests.
- **Dependence:** rank correlations, tail dependence, copulas (Gaussian, Student-t, Clayton, Gumbel, Frank and rotations) by maximum likelihood, copula simulation, mutual information on a correlation scale, distance correlation.
- **State space and regimes:** Kalman-filter time-varying regression with ML-estimated noise, pairs-trading layer, Markov-switching models, HMMs with filtered (real-time) probabilities, K-means regimes.
- **Backtesting:** one-bar-lag convention, transaction costs, performance table, walk-forward splits, time-series momentum, Sharpe confidence intervals.

## v1.0 — Initial release

A rigorous applied-econometrics workflow in Python (statsmodels, linearmodels, arch): regression with full diagnostics, endogeneity and instrumental variables, panel data, logit/probit/tobit, time series (unit roots, structural breaks, ARIMA, VAR/SVAR, cointegration/VECM), volatility (GARCH), forecasting with out-of-sample evaluation, and demand and price-elasticity estimation, with data from FRED, the World Bank, IMF, OECD and Brazil's BCB, IBGE and Ipeadata.
