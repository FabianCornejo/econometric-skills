# Interpretation and reporting rules

Apply to every analysis, in every mode. The one-line versions in `SKILL.md` are the non-negotiable core; the full rules and their reasons are here.

## Core rules
- **By functional form** (table in `ols-regression.md`): log-log → elasticity; log-lin → 100·β% (exact: 100·(e^β − 1)); lin-log → β/100 per 1%; quadratic → effect β₁ + 2β₂x.
- **In nonlinear models** (logit, probit, tobit), report average marginal effects, not raw coefficients. For Tobit, say which margin answers the question: total E(y|x), intensive E(y|y>0,x) or extensive P(y>0|x).
- **Replicating published or Stata results**: in `linearmodels` IV, `debiased=True` gives the n−k correction of textbooks and Stata's `small` option.
- **Statistical ≠ economic significance**: report magnitude and interval, not only the p-value.
- **Association ≠ causation**: causal language only with an identification strategy (IV, DiD, experiment, discontinuity). Granger causality is predictive precedence. Otherwise, "is associated with".
- **Direction of bias**: when identification is imperfect, say which way the bias probably goes.
- **Honest precision**: 2–3 significant digits; forecasts always with intervals; elasticities hold for small changes within the observed range.
- **Which interval**: a forecast for one unit uses the prediction interval, an average uses the CI of the mean (`specification.prediction_intervals`); from a log-y model, report the smearing-corrected mean in levels, not exp(ŷ).
- **Lags**: report the long-run effect (Σβ, or Σβ/(1−Σφ) with lags of y) with its delta-method SE (`specification.long_run_effect`), not the individual lag coefficients.
- **No look-ahead**: anything used to forecast, signal or risk-manage at t is computed with data up to t (lagged regressors, training-only scalers and fits, filtered not smoothed probabilities, positions applied to the next period's return).
- **LLM-coded variables are measurements with error**: never put them in a regression as if they were the truth. Hand-code a random sample drawn with known probabilities and estimate with `text_data.dsl_regression`; report the naive and corrected estimates side by side (`text-as-data.md` §6–7).
- **Segmented logit vs pooled**: when the user asks whether drivers differ across groups, run `segments.logit_leaf_model` with `segment_by=` the groups they named (categorical columns are fine) and `benchmarks=` a logit with those interactions **and every regressor the leaf model uses** (otherwise the comparison is not like for like; the function warns), then `leaf_model_stability`. If the model will score future periods, pass `time_col=` so the holdout is the most recent period. Report the model with the best holdout AUC and top-decile lift, its calibration (slope and mean bias), and the segments only if they reappear across seeds (`limited-dependent.md` §9).
- **Advice must match the tools**: never recommend a step (a temporal split, a calibration check, a test) and then show code that does something else. If a recommended step has no function in the skill, say so and write the code for it.
- **Cluster at the level where the regressor of interest varies**: a judge's sex varies by judge, so cluster by judge, not by court or chamber (a court with three judges is one cluster, not three); a policy adopted by state, by state. Use the same level in the text and in the code, and report the number of clusters, not only the number of observations; with fewer than ~40 clusters, say that the SEs are fragile (wild cluster bootstrap).
- **No invented numbers**: every coefficient, statistic or data point comes from a computation run in the session or a cited source. If data are missing, say so.

Tables: `reporting.regression_table({"(1)": m1, "(2)": m2}, keep=[...], fe_rows={"Year FE": ["YES", "YES"]})` produces the side-by-side table (coefficients, SEs in parentheses, stars, N, R², FE rows; `exponentiate=True` for odds/hazard ratios) and `export_table` writes Markdown, Excel or LaTeX. Rescale regressors whose coefficients print as 1e-05 (`reporting.rescale_hint`), and mark a fixed effect YES only where it is in that column's model.

## Rules added in v2.0

- **Tie every recommendation to a tool.** Each step you recommend names the function of this skill that performs it (e.g. `estimation.wild_cluster_bootstrap`, `causal.att_gt`, `hte.blp_test`); Stata or R commands may follow as equivalents, never replace it. When the user has not sent data yet, deliver the plan together with a runnable script that calls those functions, preferably executed on simulated data with the same structure (same unit, treatment, outcome and N), so the plan and the code cannot diverge and the user can run it on the real file. Show the code in the answer or deliver the script as a file the user can download; a path inside the sandbox is not a deliverable.

- **Run more than one model and compare** (EconML FAQ). For any causal estimate, report a simple structural model (OLS, LinearDML, linear DR-learner) next to a flexible one (forest, gradient boosting nuisances). Agreement is reassuring; a large gap means misspecification or fragile identification, and the gap itself is a finding.
- **Heterogeneous effects need evidence before they are reported.** Report subgroup or individual effects only when `hte.blp_test` rejects "no heterogeneity" (one-sided p < 0.05) and `hte.gates` shows separated groups; otherwise report the ATE and say that the data do not support differences. Effect modifiers X are chosen before looking at outcomes. `hte.clan` describes who is most affected; it does not explain why.
- **CATE models without pointwise CIs** (forest, meta-learners) are judged by BLP, GATES, TOC/AUTOC and R-score on out-of-fold predictions, never by in-sample fit or by the spread of tau(x).
- **Policies are valued on units not used to learn them** (`hte.policy_tree` does it), and with the cost of treatment in the same units as y.
- **Few clusters**: with fewer than ~40 clusters, report the wild cluster bootstrap p-value and CI (`estimation.wild_cluster_bootstrap`) next to the clustered SE.
- **Hand-written likelihoods**: use `estimation.mle` and report the sandwich SEs when the density is a convenient approximation (QMLE); check `converged` and `grad_max`; test restrictions with `estimation.trinity_test` (LR preferred when the three disagree).
- **Functions of parameters** (elasticity at a point, long-run multiplier, ratio of coefficients, turning point) get a delta-method SE (`estimation.delta_method`); bootstrap ratios whose denominator is imprecise.

## Rules added in v2.1

- **Betas come with their choices and their uncertainty**: market proxy, frequency, window, adjustment, and the CI; check stability with `finance_empirics.rolling_beta`. For a valuation, prefer the relevered median asset beta of comparables to a single company regression.
- **Event studies**: report the event-date source, windows and normal-return model, and the CAAR with the cross-sectional t, BMP and Corrado tests; BMP is the default when the announcement may raise volatility (Patell then over-rejects).
- **Few treated units in a DiD**: report `causal.randomization_inference` next to the clustered and wild-bootstrap results, and say whether treated and control units are plausibly exchangeable.
- **After many pairwise comparisons, adjust** (Tukey, Holm or BH); after variable selection, do not read the selected model's p-values as if it had been pre-specified (use `post_double_selection` for one effect).
- **Teaching simulations** use a fixed seed, show their code, and report the quantity the lesson is about (bias, coverage, rejection rate), not only a single draw.

- **Betas for a valuation** come from comparables whose windows start after any break in their risk profile (change of control, privatization, new regulation); report the unlevering method (Hamada/Conine or Harris-Pringle, with or without debt beta), and use `finance_empirics.dimson_beta` or weekly returns for thinly traded stocks.
