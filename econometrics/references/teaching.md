# Teacher mode

Use when the user wants to understand a method, prepare a class or an exam, check their reasoning, or asks "why" a test or estimator exists. The goal is understanding that transfers to new problems, not a recipe. The same scripts and the same rules as in Consultant mode apply: a class never shows numbers the code did not compute.

## 1. Choose the level

| Level | Reader | Math | Typical references |
|---|---|---|---|
| Basic | First contact; manager or student | Words, pictures, one equation | Tamhane & Dunlop, *Statistics and Data Analysis*, and Bussab & Morettin, *Estatística Básica* (Module 0); Dalpiaz, *Applied Statistics with R*; Colonescu, *Principles of Econometrics with R* (Hill, Griffiths & Lim); Angrist & Pischke, *Mastering 'Metrics* |
| Intermediate | Graduate course, practitioner | Matrix notation, assumptions and proofs sketched | Wooldridge, *Introductory Econometrics*; Faraway, *Linear Models with R*; Huntington-Klein, *The Effect* |
| Advanced | Researcher | Asymptotics, influence functions, semiparametric efficiency | Hansen, *Econometrics*; Wooldridge, *Cross Section and Panel Data*; Chernozhukov et al. (2018); EconML documentation |

Ask the level only if it is not clear from the question; otherwise infer it and say which level you chose.

## 2. The method card

Every method is explained with the same eleven blocks, in this order. Skip a block only if it is truly empty for that method.

1. **Intuition**: the problem the method solves, in two or three sentences and one everyday example.
2. **Model**: the equation, every symbol defined.
3. **Assumptions**: numbered, each with what goes wrong if it fails.
4. **Estimator**: the formula (display LaTeX) and what it computes in words.
5. **Short derivation**: the two or three steps that show why the estimator works under the assumptions (e.g. OLS unbiasedness from E[u|X] = 0; the IV estimator as a ratio of two reduced forms; the DR score's double robustness).
6. **Simulation**: a small Monte Carlo that shows the method working and then failing when one assumption is broken (§3). Seeing the bias appear is what makes the assumption stick.
7. **Code**: the skill's function on real or simulated data, with the output. The code itself appears in the answer (or is delivered as a downloadable file): the student must be able to rerun the simulation and change its parameters. Never cite a path inside the sandbox.
8. **Diagnostics**: the tests that check the assumptions that can be checked, and what to do when they fail.
9. **Interpretation**: one sentence for each number that matters, with units and the interval.
10. **Pitfalls**: the three most common mistakes.
11. **Exercise**: one problem with data or a scenario, and **always** its complete worked solution below a separator ("Solution"). A closing comprehension question does not replace the solved exercise.

## 3. Simulations that teach

Ready-made demos are in `scripts/simulation.py` (each returns a table and a one-line lesson; show their code and let the student change one parameter): `demo_clt`, `demo_ci_coverage`, `demo_estimator_mse`, `demo_multiple_testing` (Module 0), `demo_omitted_variable`, `demo_bad_control`, `demo_measurement_error`, `demo_heteroskedasticity`, `demo_spurious_regression`, `demo_weak_iv`, `demo_few_clusters`, `demo_staggered_did`, `demo_did_covariates`; `monte_carlo(dgp, estimators, R, truth)` builds any other one, and `power_mde` answers "how large a sample?" or "how small an effect can we detect?" (for a DiD, `mde_did` and the simulation `power_did_sim`, which shows how serial correlation erodes power). Good and bad controls are taught with `dag.py`: draw the graph, let `adjustment_report` classify each variable, then simulate the linear model and show OLS with a minimal set recovering the effect while a collider or mediator biases it. The heterogeneity lesson uses `hte` (a linear CATE vs a forest on a symmetric effect).

Write them in a few lines of numpy with a fixed seed; repeat 500–1000 times; show the distribution of the estimate against the truth (histogram or a table of mean, bias, SD, coverage).

| Lesson | Simulation | What the student sees |
|---|---|---|
| Sampling distribution and CLT | `demo_clt`: means of n draws from a skewed population | SD = sigma/sqrt(n) for every n; the normal shape and the t interval's coverage arrive only gradually |
| What 95% means | `demo_ci_coverage`: thousands of intervals | About 95% contain the truth for a normal mean with t; fewer with z and an estimated SD, skewed data or the Wald interval for a proportion; Wilson stays close |
| Bias, variance, MSE | `demo_estimator_mse`: S^2 vs the MLE of a variance; method of moments vs MLE for a uniform | The unbiased estimator is not the most precise; MSE = variance + bias^2 |
| Many tests | `demo_multiple_testing`: 20 tests per study | About 64% of studies find a "significant" null without correction; Holm and Benjamini-Hochberg fix it |
| Unbiasedness and sampling variation | OLS on repeated samples (Colonescu 2.5; `monte_carlo`) | Estimates spread around the true beta; spread falls with n |
| CI coverage in regression | 95% CIs over 1,000 samples (Colonescu 3.4; `demo_heteroskedasticity`) | About 950 contain the truth; with heteroskedasticity and classical SEs, fewer |
| Omitted variable bias | Drop a confounder correlated with x | Bias = beta_z · delta, with the predicted sign |
| Bad control | Control for a variable caused by the treatment | The effect shrinks or flips |
| Measurement error | Add noise to x | Attenuation toward zero |
| Spurious regression | Two independent random walks | High R², "significant" t, DW near zero |
| Weak instruments | IV with a weak first stage | 2SLS biased toward OLS, CIs that do not cover |
| Few clusters | Clustered SEs with 8 clusters | Over-rejection; `estimation.wild_cluster_bootstrap` fixes it |
| Staggered DiD | TWFE with growing effects and staggered dates | TWFE wrong (even the sign); `causal.att_gt` right |
| Heterogeneity | Effect larger at both extremes of x | A linear CATE sees nothing; a forest finds it (`hte`) |
| Good and bad controls | Butterfly or M-bias graph (`dag.py`), linear SEM | A minimal set recovers the effect; adding a collider or using an incomplete set biases it |
| Power and serial correlation | `simulation.power_did_sim` with AR(1) ρ = 0 vs 0.8 | Same analytic power, much lower simulated power |
| Fat tails | `risk.gpd_fit` on Student-t returns | Normal 99.9% VaR far too small; EVT close to the truth |
| Look-ahead | Scale with full-sample statistics in a backtest | Spectacular Sharpe that vanishes out of sample |

## 4. Curriculum order

For a course or a self-study plan, start with Module 0 when the student has not had statistics (or is rusty), then follow the order of Hill, Griffiths & Lim (as in Colonescu) and extend it:
0. **Module 0, statistics foundations** (`statistics-foundations.md`): probability and Bayes → distributions → sampling distributions and the CLT → estimators (moments, maximum likelihood) → confidence intervals → tests and power → tables, goodness of fit and many tests. Code: `inference.py`.
1. Simple regression → inference → prediction and functional form.
2. Multiple regression → further inference (F tests, omitted variables, collinearity).
3. Dummy variables, interactions, Chow test, linear probability model, DiD.
4. Heteroskedasticity → time series with stationary variables (distributed lags, serial correlation).
5. Random regressors and IV → simultaneous equations.
6. Nonstationary series, cointegration, VEC/VAR → ARCH/GARCH.
7. Panel data → qualitative and limited dependent variables.
8. Extensions: modern causal designs (`causal-inference.md`), heterogeneous effects (`heterogeneous-effects.md`), maximum likelihood and the bootstrap (`estimation-theory.md`), forecasting and risk.

## 5. Style

- Build from what the student already knows; one new idea per step.
- Ask the student to predict the result of a simulation before running it.
- After the explanation, a short check of understanding (two or three questions); offer a quiz when the user wants practice.
- Use the user's language; keep the standard technical terms in English in parentheses when they are the ones used in the literature.

## 6. Curriculum map

Where each topic of a standard sequence (statistics, econometrics, time series: the Insper MPE courses, Wooldridge, Hill-Griffiths-Lim) lives in the skill. Teacher mode uses it to pick the function behind each method card; `tests/test_curriculum_map.py` checks that every function and section named here exists, so the map cannot go stale. A topic with only a reference section has no dedicated function yet: write the code in the answer.

| Topic | Module | Where in the skill |
|---|---|---|
| Probability, Bayes, Simpson's paradox | 0 | `statistics-foundations.md §1`, `inference.contingency` |
| Distributions and probability calculations | 0 | `statistics-foundations.md §2` |
| Sampling distributions, CLT | 0 | `statistics-foundations.md §3`, `simulation.demo_clt` |
| Bias, variance, MSE; method of moments; maximum likelihood | 0 | `statistics-foundations.md §4`, `simulation.demo_estimator_mse`, `estimation.mle`, `estimation.delta_method` |
| Confidence intervals: mean, proportion, variance; sample size | 0 | `statistics-foundations.md §5`, `inference.one_sample`, `inference.proportion_ci`, `inference.variance_test`, `inference.n_for_proportion`, `simulation.demo_ci_coverage` |
| Tests: logic, errors, power | 0 | `statistics-foundations.md §6`, `inference.power_curve`, `simulation.power_mde` |
| Two means: Welch, pooled, paired | 0 | `inference.two_sample` |
| Rank tests | 0 | `inference.nonparametric` |
| Two variances | 0 | `inference.variance_test` |
| Proportions | 0 | `inference.proportion_test` |
| Goodness of fit, normality | 0 | `statistics-foundations.md §8`, `inference.chisq_fit`, `inference.normality` |
| Contingency tables, odds ratios | 0 | `statistics-foundations.md §9`, `inference.contingency` |
| Correlation | 0 | `inference.correlation_test` |
| Multiple testing | 0 | `statistics-foundations.md §10`, `inference.adjust_pvalues`, `simulation.demo_multiple_testing` |
| Exploratory analysis | 0 | `statistics-foundations.md §11`, `inference.describe_dist`, `data_prep.audit`, `data_prep.missing_report` |
| Simple and multiple regression, Gauss-Markov | 1-2 | `ols-regression.md §1`, `diagnostics.ols_diagnostics`, `simulation.monte_carlo` |
| Prediction vs confidence intervals | 1 | `ols-regression.md §12`, `specification.prediction_intervals` |
| Linear restrictions, F tests | 2 | `ols-regression.md §4` |
| Dummies, interactions, Chow test | 3 | `ols-regression.md §3`, `diagnostics.chow` |
| Functional forms | 3 | `ols-regression.md §2` |
| Multicollinearity | 3 | `ols-regression.md §8`, `diagnostics.vif_table` |
| Omitted variables, measurement error | 2-5 | `ols-regression.md §14`, `specification.ovb_decomposition`, `simulation.demo_omitted_variable`, `simulation.demo_measurement_error` |
| Heteroskedasticity: tests, robust SEs, WLS/FGLS | 4 | `ols-regression.md §6`, `diagnostics.compare_standard_errors`, `specification.fgls_het`, `simulation.demo_heteroskedasticity` |
| Serial correlation: DW, BG, HAC, GLS | 4 | `ols-regression.md §7`, `diagnostics.newey_west_lags` |
| Specification: RESET, model selection | 4 | `ols-regression.md §11`, `diagnostics.ols_diagnostics` |
| IV and 2SLS, endogeneity and overidentification tests | 5 | `panel-iv.md §2`, `panel-iv.md §3`, `diagnostics.endogeneity_test`, `diagnostics.overid_test`, `simulation.demo_weak_iv` |
| Simultaneous equations, identification | 5 | `panel-iv.md §5` |
| Panel data: FE, RE, Hausman, Mundlak | 7 | `panel-iv.md §7`, `panel-iv.md §9`, `diagnostics.hausman`, `diagnostics.mundlak_test` |
| Dynamic panels | 7 | `panel-iv.md §11` |
| LPM, logit, probit, marginal effects | 7 | `limited-dependent.md §1`, `limited-dependent.md §3` |
| Tobit, Heckman | 7 | `limited-dependent.md §5`, `limited-dependent.md §6`, `diagnostics.tobit` |
| Counts | 7 | `limited-dependent.md §7`, `diagnostics.count_diagnostics` |
| Likelihood tests, delta method, bootstrap | 8 | `estimation-theory.md §2`, `estimation.trinity_test`, `estimation.bootstrap`, `estimation.wild_cluster_bootstrap` |
| Decomposition, filters, seasonality | TS | `time-series.md §2` |
| Stationarity, unit roots | TS | `time-series.md §3`, `diagnostics.stationarity`, `simulation.demo_spurious_regression` |
| Structural breaks | TS | `time-series.md §4`, `diagnostics.chow` |
| ARMA and Box-Jenkins | TS | `time-series.md §5`, `forecasting.auto_arima`, `forecasting.select_d` |
| Forecast evaluation, combination | TS | `time-series.md §11`, `forecasting.rolling_origin`, `forecasting.evaluate`, `diagnostics.diebold_mariano` |
| Cointegration, ECM/VECM | TS | `time-series.md §7`, `diagnostics.cointegration` |
| VAR, SVAR, impulse responses | TS | `time-series.md §8`, `time-series.md §9`, `forecasting.rolling_var` |
| ARCH/GARCH, stylized facts | TS | `volatility.md §1`, `volatility.md §3`, `risk.var_garch` |
| Causal designs (matching, DML, RD, DiD, synthetic control) | 8 | `causal-inference.md §2`, `causal.aipw`, `causal.rdd`, `causal.att_gt`, `causal.synth` |

