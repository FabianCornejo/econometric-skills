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
- **Scripts must run on the user's machine and on real data**: a `SKILL_DIR` variable for this skill's folder that contains `scripts/`, whose default is the user's path, never a sandbox path, not even with the user's path in a comment (skeleton below), and a test on simulated data that include the cases real data bring (events without enough history, missing values, repeated units). In event studies, use `finance_empirics.car_regression` for the cross-section of CARs rather than hand-merging results.

## Rules added in v2.2

- **Choose controls with a graph, not by availability.** Before estimating an effect from observational data, write the causal graph and run `dag.adjustment_report`: control for a minimal (or the optimal) adjustment set, never for mediators, colliders, post-treatment variables or pure instruments; if no valid set exists, say the effect is not identified by adjustment. Report the graph and the result of `dag.test_implications`.
- **Causal claims from observables carry a sensitivity analysis**: report the robustness value (OLS: `specification.sensitivity`; DML: `causal.dml_sensitivity`) benchmarked on the strongest observed covariates (`causal.dml_benchmark`), in words: "a confounder k times as strong as X would move the estimate to ...".
- **An insignificant effect needs its MDE**: report the minimum detectable effect of the design (`simulation.mde_did`; with serial correlation or few treated units, `simulation.power_did_sim`) and the largest effect the CI excludes. Plan power before collecting data.
- **Far-tail risk (99.5% and beyond) uses EVT** (`risk.gpd_fit`, `risk.var_evt`) with the threshold-stability table, and states ξ with its CI; **ES forecasts are backtested** (`risk.es_backtest`) and ranked with `risk.compare_es_models`, not judged by eye.

## Rules added in v2.3

- **DiD with covariates: levels, not TWFE.** When parallel trends need covariates, estimate with `causal.att_gt(covariates=[...])` (doubly robust by default), never by adding the covariates to a TWFE regression: unit fixed effects difference them away (hidden linearity bias; Caetano & Callaway 2024). Before reporting, run `causal.ccc_test` on the untreated observations and say what it selected; if `group` or `twoway`, say that TWFE and Callaway-Sant'Anna with covariates are biased and point to DID-INT. When someone else's TWFE with covariates is under review, run `causal.twfe_covariate_diagnostic` and report its verdict.
- **Headline numbers are recomputed by an independent route** (the dynare-mcp principle of solving the same model with a second solver). Use the routes the skill already computes (`att_gt`'s DR vs outcome regression vs IPW in `res["cross_check"]`; DML vs OLS with the same controls; historical vs filtered VaR; a formula vs its simulation) or `reporting.cross_check` for any pair. Agreement goes in the report in one line; disagreement beyond one SE is a finding to explain, never a reason to pick the more convenient number.
- **Deliverables carry a run manifest.** Every script that produces numbers for a report ends with `reporting.run_manifest(data={...}, files=[...], seed=..., results={...}, path="manifest.json")`: data hashes, package and skill versions, seed. A number that cannot be traced to a manifest is not reproducible.
- **Robust Wald tests with many restrictions** (dozens of cells or groups) over-reject with their χ² reference under heteroskedasticity; use a wild bootstrap imposed under the null (as `ccc_test` does by default) or say that the p-value is anticonservative.
- **Basel traffic light**: the zones (green 0–4, yellow 5–9, red ≥ 10 violations in 250 days) and the 2.5 expected violations belong to the **99%** VaR. At 97.5% the expectation is 6.25 violations in 250 days and the traffic light does not apply; FRTB's desk-level backtesting sets its own limits at both levels (desks lose model approval with more than 12 exceptions at 99% or 30 at 97.5% in the last 250 days; Basel MAR32 — confirm the local rule).

Script skeleton for every deliverable (v2.3.2; install message v2.3.5; trash skipped v2.3.8). The user should run it as received, with no paths to edit and no variables to set:

```python
import os, sys
from pathlib import Path

def _skill_dir():
    """Find the econometrics skill: ECONOMETRICS_SKILL_DIR, the Claude Code plugin, ~/.claude/skills
    (including the copy synced from the Claude app), your clone."""
    skills = Path.home() / ".claude" / "skills"
    synced = sorted((f for f in skills.glob("**/econometrics/scripts/causal.py")   # skip hidden folders: .trash
                     if not any(p.startswith(".") for p in f.relative_to(skills).parts)),   # holds removed skills
                    key=lambda f: f.stat().st_mtime, reverse=True) if skills.is_dir() else []
    for c in (os.environ.get("ECONOMETRICS_SKILL_DIR"),
              Path.home() / ".claude/plugins/marketplaces/econometric-skills/econometrics",
              *[f.parents[1] for f in synced],
              r"<path to your clone>/econometric-skills/econometrics"):
        if c and Path(c, "scripts", "causal.py").exists():
            return str(c)
    sys.exit("Econometrics skill not found: set ECONOMETRICS_SKILL_DIR to the skill folder (the one with scripts/).")

SKILL_DIR = _skill_dir()
sys.path.insert(0, f"{SKILL_DIR}/scripts")
try:
    import causal, reporting   # the modules this script uses
except ModuleNotFoundError as e:   # name a command that starts THIS Python ('python' may be the Windows Store alias)
    import shutil
    py = (f"py -{sys.version_info[0]}.{sys.version_info[1]}" if os.name == "nt" and sys.prefix == sys.base_prefix
          and shutil.which("py") else f'"{sys.executable}"')
    sys.exit(f"Missing Python package '{e.name}'. Run once:  {py} \"{SKILL_DIR}/scripts/setup_env.py\" --install")
```

# ... then load data (or simulate data with the same structure), estimate, diagnose, and end with
# reporting.run_manifest(data={"panel": df}, seed=SEED, results={"ATT": res["overall"]}, path="manifest.json")

To run it in your own environment, set `ECONOMETRICS_SKILL_DIR` to the skill folder there; never write that path, a sandbox path or the versioned plugin cache path (`plugins/cache/econometric-skills/econometrics/<version>/`, which changes at every release) into the file. List the modules the script really uses in the `import` line of the block.

## Rules added in v2.3.1

- **Cluster the DiD bootstrap at the assignment level** with `att_gt(cluster=...)` (and the same `cluster=` in the date checks); never aggregate units to the cluster level just to cluster, because the estimand changes. With fewer than ~30 clusters or ~6 treated ones, add `causal.randomization_inference`.
- **Independent checks need two routes**: pass `att_gt`'s `res["cross_check"]` Series to `reporting.cross_check` as it is; a check with one route verified nothing and must not be reported as agreement.
- **Relay every diagnostic verdict as printed**, including `twfe_covariate_diagnostic`'s "inconclusive (low power)"; a script whose output contradicts the text is a failed deliverable.

## Rules added in v2.3.7

- **A number from an estimator that did not converge is not reported.** Read every optimizer warning (statsmodels' `IterationLimitWarning`, `ConvergenceWarning`, a singular or non-invertible Hessian, `converged` False in `estimation.mle`) and resolve it before the number enters a table or the summary: more iterations, another solver, fewer, rescaled or less collinear regressors. `linear_models.quantile_regression` fits with 5000 iterations, retries with 4x as many, reports `converged` per tau (and a table column), drops bootstrap replications that did not converge (`B_used`) and flags the equality test `valid: False` when an end tau did not converge. In WAGE1, tau = 0.1 with eight regressors stopped at 1000 iterations with educ 0.0389 instead of 0.0386. A tau, model or replication that did not converge is said to have failed and left out; never report it with a footnote.
- **The independent route uses a different method.** A check means something only when the second route can disagree: robust (Huber) vs OLS, pairs-bootstrap vs analytic CI, DR vs outcome regression vs IPW, DML vs OLS, a formula vs its simulation; pass `se=` to `reporting.cross_check` (kind "method"). The same estimator computed by two implementations (statsmodels vs `numpy` normal equations, the skill vs `csdid`) agrees to machine precision by construction: `cross_check` without `se` labels it an implementation check (kind "implementation"), which verifies the arithmetic, not the method. Report it as such, never as an independent confirmation of the result.


## Rules added in v2.4.0

- **Not rejecting is not accepting.** A test that does not reject says the data are compatible with H0 and with every value in the interval; it never shows that H0 is true or that "there is no effect". Write "does not reject at 5%", give the interval and, for a claim of no effect, the effect the test could have detected (`inference.power_curve`, `simulation.power_mde`). Every `inference` verdict is worded this way; relay it as printed.
- **Welch by default for two means.** The pooled t test assumes equal variances; with unequal variances and unequal group sizes its level is wrong. Use `inference.two_sample` (Welch) unless the design guarantees equal variances; `equal_var=True` warns when Brown-Forsythe rejects. Paired data are analysed as pairs (`paired=True`), never as two independent samples.
- **Proportions near 0 or 1, or small n: Wilson, not Wald.** The textbook interval p +/- z sqrt(p(1-p)/n) covers far less than 95% when n p or n (1 - p) is below about 10 (`simulation.demo_ci_coverage`). `inference.proportion_ci` reports Wilson by default and shows the others side by side; small expected counts in tables call for Fisher's exact test.
- **Variance tests need normal data.** Chi-square intervals for a variance, the F test of two variances and Bartlett's test are not robust to heavy tails; check with `describe_dist` and prefer Brown-Forsythe before choosing a pooled t.
- **Goodness of fit loses one degree of freedom per estimated parameter**: df = k - a - 1 after pooling cells with expected counts below 5 (`inference.chisq_fit`).
- **Kurtosis has two conventions.** The moment ratio is 3 for a normal; Excel's KURT and pandas' `.kurt()` report *excess* kurtosis (0 for a normal). State which one is reported; `describe_dist` gives both.
- **Stratify before reading a crude table.** When a third variable could drive both the row and the column (case mix, age, size), run `inference.contingency(..., strata=)`; a reversal inside every stratum (Simpson's paradox) means the crude comparison is confounded.
- **Odds ratios and log coefficients in exact percentages**: an odds ratio of 1.39 is 39% higher odds; a dummy coefficient b in a log model is 100 (e^b - 1)%.
- **Declare the family of tests** before looking at results and adjust (`inference.adjust_pvalues`: Holm for the family-wise error, Benjamini-Hochberg for the false discovery rate); report how many tests were run.

## Rules added in v2.4.2

- **Restriction tests by the SSR when y changes.** The R² form of the F test is valid only when both models explain the same y. A restricted model fitted to a transformed y (y − x to impose β = 1) has another total sum of squares; use the SSR form or the Wald test (`specification.linear_restriction_test`). Under heteroskedasticity neither form holds: use the robust Wald F (`cov_type="HC3"`).
- **Binary outcomes: report effects on the probability, and judge classification against the base rate.** Logit and probit coefficients are not effects on the probability and differ in scale between the two; compare average marginal effects with the LPM slopes (`discrete_choice.binary_effects`). An accuracy figure means something only next to the accuracy of always predicting the majority class; report AUC, and a cutoff chosen by the base rate or the costs of errors, not a fixed 0.5.
- **Weak instruments get the Anderson-Rubin set.** With a first-stage F below 10 the 2SLS estimate and its t-interval are not credible; between 10 and 104.7 the t-interval can still undercover. Report the Anderson-Rubin set (`diagnostics.iv_report`) and say how strong the first stage is. An overidentification test exists only with more instruments than endogenous regressors, and passing it does not prove validity.
- **Heckman needs an exclusion restriction.** Without a variable that moves selection but not the outcome, identification rests on the normal functional form only; say so (`discrete_choice.heckman` warns). A significant λ is evidence of selection; an insignificant one is not proof that there is none. Heckman is a selection model, not a Tobit estimator.
- **Prais-Winsten and Cochrane-Orcutt need strictly exogenous regressors.** With a lagged y or feedback from y to x, FGLS for AR(1) errors is inconsistent; keep OLS with HAC errors and model the dynamics.
- **Non-nested tests can reject both models or neither.** Report both directions of the J test; "the data favour model 2" is a conclusion only when model 1 is rejected and model 2 is not (`specification.nonnested_test`).

## Rules added in v2.4.6

- **Sample size for a given effect: compute it, never read it off a table.** For "how many would we need to detect this difference?", run `inference.n_for_power(effect, sd)` with that effect. A power table built for other effect sizes (0.5 or 0.8 SD) answers another question: in the v2.4.5 test a difference of 8.3 points (d ≈ 0.94) was quoted as needing 26 per group, the figure for 0.8 SD; the right number is 19. Plugging in the observed difference answers "if the truth were what we saw"; say so.
- **A unit-root test that does not reject is not evidence of a unit root.** Report the ADF (bootstrap p-value in short samples, `ts_univariate.adf_bootstrap`) with KPSS, the plot and a break test; near φ = 1 the ADF has little power, and a break makes a stationary series look I(1). State the consequence of each choice (differencing vs a trend) for the forecast.
- **Filters do not see the future.** A two-sided HP or band-pass trend at the end of the sample will be revised; for a real-time reading use the one-sided HP or the Hamilton filter and report the size of the end-point revision (`ts_univariate.detrend`). Do not extrapolate a filtered trend.
- **A break date is an estimate only when the break test rejects.** Report the sup-Wald p-value (bootstrap) before naming a date; with no rejection, say "no evidence of a single break", not "the series is stable", and look at CUSUM and the recursive coefficients. A break test in levels assumes a stationary series: for one that may be I(1), test the differences or use Zivot-Andrews.
- **Forecast intervals with parameter uncertainty.** Analytic AR intervals assume known coefficients and normal errors; in short samples show the bias-corrected bootstrap interval next to them (`ts_univariate.ar_bootstrap_forecast`), and say that both can undercover. A combination of forecasts uses equal weights unless an out-of-sample window clearly favours others (`ts_univariate.combine_forecasts`).

## Rules added in v2.4.6.1

From the first test on a real series (monthly IPCA 2000–2026):

- **A positive d is not long memory until short memory is ruled out.** GPH and local Whittle are biased upward by ordinary AR persistence; report the short-memory check of `ts_univariate.long_memory` (an AR(p) simulated under short memory) and write "no evidence of long memory beyond short-run persistence" when it does not reject.
- **Break tests that disagree are a finding, if both are valid.** A sup-F that does not reject next to a CUSUM that does means gradual or multiple change, not stability; but a CUSUM scaled by the iid SD of autocorrelated residuals rejects stable series most of the time (on the IPCA, p = 0.0025 with the iid SD, 0.34 with the long-run SD): use the long-run version (`ts_univariate.sup_wald`). A break dated at the edge of the trimmed window means the end (or start) of the sample differs, with an imprecise date.
- **An agreement check is only as good as its tolerance.** When two routes "agree within 1 SE" and the SE is a large share of the estimate, report the spread between them, not the word agree (`reporting.cross_check` flags it as weak).

## Rules added in v2.4.7

- **Dynamic panel: no AR(2), Hansen J and instrument count, no result.** AR(1) in the differenced residuals is expected; AR(2) must not reject. Hansen must not reject, but a p-value near 1 with more instruments than groups means a weakened test. ρ should lie between within and pooled OLS (`systems.dynamic_panel`).
- **A Cholesky IRF is an assumption about ordering.** State why each variable cannot respond within the period to those after it, and show the FEVD under other plausible orderings (`multivariate.fevd_table`); shares that move by more than 10 points across orderings are not findings.
- **Granger causality is predictive precedence**, and it is unreliable in a VAR in levels with I(1) series.
- **Sign restrictions identify a set, not a model.** Report the acceptance share and the median-target model; the pointwise median of accepted draws is not the response of any single model.
- **In a VECM, read α only after the error-correction check.** The implied levels VAR must have exactly k − r unit roots and the rest inside the unit circle; a significant α of the wrong sign (one relation) means the long-run relation is not supported, whatever the trace test said. With several relations, single loadings depend on the normalisation: use the joint weak-exogeneity test. State the deterministic case; statsmodels' unrestricted constant over-rejects for series without drift.
- **SUR equals OLS with identical regressors; 3SLS equals 2SLS when every equation is exactly identified.** Say so instead of presenting them as different estimates.
- **GARCH names**: GJR = threshold in the variance; EGARCH = log-variance (not TGARCH). Persistence ≥ 0.99 deserves a break check before it is reported as a property of the asset.
- **Fama-MacBeth with estimated betas** needs the Shanken correction (or a bootstrap) and Newey-West on the λ_t; the plain FM SE overstates precision.
