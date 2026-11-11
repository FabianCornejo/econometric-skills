# Causal inference with observational data

Use when the question is "what is the effect of X on Y?" and X was not randomized: a policy, a law, a court decision, a program, a promotion, a merger, a judge's experience. Code: `scripts/causal.py`. IV and basic difference-in-differences are in `panel-iv.md`; this file covers design choice, selection on observables, regression discontinuity, synthetic control, modern DiD and ML-based estimators.

## Contents
1. Potential outcomes and the design-first rule
2. Choosing a design (2.1 choosing controls with a causal graph)
3. Selection on observables: regression, matching, weighting, doubly robust
4. Double/debiased machine learning (4.1 sensitivity to omitted confounders)
5. Regression discontinuity
6. Difference-in-differences and event studies with staggered adoption (6.2 power)
7. Synthetic control
8. Designs common in law and economics
9. Reporting checklist

## 1. Potential outcomes

Each unit has Y(1) and Y(0); only one is observed. Estimands:
- **ATE** = E[Y(1) − Y(0)] for the whole population.
- **ATT** = E[Y(1) − Y(0) | treated]: usually the policy-relevant one ("what did the program do for those who got it?").
- **LATE**: effect for compliers (IV, fuzzy RD).

Naive difference = ATT + selection bias, where selection bias = E[Y(0)|T=1] − E[Y(0)|T=0]. Every design below is a way to make that term zero under an explicit, defensible assumption. **Write the assumption in words before estimating**; it is what readers will (and should) attack.

Decide the design before looking at outcomes: choose covariates, bandwidths, comparison groups and the event window on pre-treatment information or a pre-analysis plan. Specification searching on the outcome invalidates p-values.

## 2. Choosing a design

| Situation | Design | Key assumption | `causal.py` |
|---|---|---|---|
| Treatment depends only on observed characteristics | Regression / matching / IPW / AIPW | Unconfoundedness + overlap | `aipw`, `ipw`, `nn_match`, `balance_table` |
| Many controls, flexible confounding | Double ML | Unconfoundedness; nuisance functions learnable | `dml_plr` |
| Treatment assigned by a score crossing a threshold (age, grade, vote share, value of claim) | Regression discontinuity | Continuity of potential outcomes at the cutoff; no precise manipulation | `rdd`, `rdd_robustness`, `density_test` |
| Policy hits some units at a known date; untreated comparison exists | Difference-in-differences / event study | Parallel trends | `event_study`, `att_gt` |
| Single treated unit (a state, a country, a court), many untreated, long pre-period | Synthetic control | Treated unit's counterfactual = convex combination of donors | `synth` |
| One (or two or three) treated units against one or a few controls, short series (a store promotion over 24 months) | DiD on the gap series | Parallel trends of the gap; AR(1) dynamics of its deviations | `did_single_treated` (not HAC on the gap) |
| An instrument shifts treatment but not the outcome directly | IV / 2SLS | Relevance, exclusion, monotonicity | `panel-iv.md` |
| Random assignment of cases to decision makers (judges, examiners) | Judge/examiner IV ("leniency") | As-good-as-random assignment, exclusion, monotonicity | `panel-iv.md` + §8 |
| Any of the above, and the question is for whom the effect is larger or whom to treat | CATE + policy learning | Same as the design, plus overlap within subgroups | `heterogeneous-effects.md` (`hte.py`) |

When several designs are possible, run more than one; agreement across designs with different assumptions is the strongest evidence available.

### 2.1 Choosing controls with a causal graph

"Control for everything available" is wrong: some controls remove bias, others create it. Write the causal graph (DAG) of the problem, from institutional knowledge, before looking at outcomes, and let `scripts/dag.py` do the graph algebra (Pearl 2009; Cinelli, Forney & Pearl 2022):

```python
import dag
g = dag.DAG("income -> price_sensitivity; income -> discount; discount -> purchase; "
            "price_sensitivity -> purchase; discount -> visits -> purchase; purchase -> review; discount -> review; "
            "region -> discount; season -> purchase")      # "a <-> b" adds an unobserved common cause
rep = dag.adjustment_report(g, "discount", "purchase", df=data)   # sets {income} or {price_sensitivity};
                                                                  # optimal {price_sensitivity, season}
```

Always include the edge x -> y being estimated: a graph without it claims the effect is zero (the report warns). It returns the **minimal adjustment sets** (backdoor criterion), the **optimal set** (Henckel, Perković & Maathuis 2022: among valid sets, the one with the smallest variance; it adds outcome predictors and drops pure treatment predictors), the **role of every variable** and, with data, **tests of the graph's implications**.

| Role (`classify_controls`) | Example | Control? |
|---|---|---|
| Confounder (causes x and y) | income for a discount's effect on purchases | Yes (or a set that blocks its path) |
| Mediator (x → v → y) | site visits caused by the discount | No: removes part of the effect (unless the question is the direct effect) |
| Descendant of x / collider | a delivery delay caused by the discount campaign | No: post-treatment, can open a path |
| Descendant of y (consequence of the outcome) | a review written after the purchase; the market reaction to a premium paid | No: conditioning on it biases the effect, even though it raises R² |
| Collider of two causes, not a descendant of x (M-bias) | a variable caused by two independent causes of x and y | No: controlling opens a path |
| Instrument-like (affects y only through x) | region of a randomized rollout | No: raises variance and amplifies any remaining bias; use it as an instrument |
| Outcome predictor (not a cause of x) | season for purchases | Optional: improves precision |

`test_implications` checks the conditional independences the graph implies (partial correlations, Holm-adjusted): a rejection says an edge is missing; passing does not prove the graph. If no valid set exists (an unobserved confounder opens every backdoor path), the effect is **not identified by adjustment**: say so and move to an instrument, a panel design or a front-door path. Show the graph in the report (`g.to_mermaid()` gives a Mermaid diagram).

## 3. Selection on observables

**Assumptions**: (i) unconfoundedness — conditional on X, treatment is as good as random; (ii) overlap — 0 < P(T=1|X) < 1. Choose X among **pre-treatment** variables that affect both treatment and outcome; never control for variables affected by the treatment ("bad controls", Angrist-Pischke).

Workflow:
1. `balance_table(df, "t", covs)`: standardized mean differences. |SMD| > 0.1 signals imbalance; > 0.25 means regression adjustment alone relies heavily on functional form.
2. `propensity_score(df, "t", covs)`: logit score, prints the overlap of treated and controls. Units with scores near 0 or 1 have no comparable counterpart: trim to [0.1, 0.9] (Crump et al., 2009) or estimate the ATT, which only needs controls comparable to the treated.
3. Estimate with at least two methods:
   - `aipw(...)` — **doubly robust**: consistent if either the propensity or the outcome model is correct; standard error from the influence function. Default choice.
   - `ipw(..., estimand="ATE"/"ATT")` — normalized weights; reports effective sample size and the largest weight share. One unit carrying 5%+ of the weight is a red flag.
   - `nn_match(...)` — nearest-neighbour matching on the logit of the score with a 0.2-SD caliper (Austin, 2011); intuitive for non-technical readers; its SE ignores the matching step (use as a check).
4. Re-run `balance_table` with the weights (or on the matched sample): balance after adjustment is the diagnostic that matters.
5. **Sensitivity to unobserved confounding**: report how strong a hidden confounder would need to be to explain away the result (Oster, 2019, coefficient stability δ: `specification.oster`; Cinelli-Hazlett, 2020, robustness value: `specification.sensitivity`, with observed covariates as benchmarks). Unconfoundedness cannot be tested; it can only be stress-tested. Details and interpretation in `ols-regression.md` §14.

## 4. Double/debiased machine learning

`dml_plr(df, y, d, X)` estimates θ in Y = θD + g(X) + U with flexible g (random forest by default; any scikit-learn regressor, e.g. `LassoCV`, gradient boosting):
1. Cross-fitting: split the sample in K folds; predict Y and D from X with models trained on the other folds.
2. Regress the residual of Y on the residual of D (Frisch-Waugh-Lovell with ML).
3. Repeat over several splits and take the median (`n_rep`).

Use it when there are many controls or nonlinear confounding. It does **not** solve unobserved confounding: DML is selection on observables with better functional form. Prediction-oriented ML (lazy comparisons of 30 classifiers, maximizing accuracy) answers a different question: good prediction of Y is not an effect of D.

### 4.1 Sensitivity to omitted confounders

DML removes functional-form bias, not omitted-variable bias. `dml_sensitivity(res, cf_y, cf_d, rho=1)` (Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis 2022, "Long Story Short"; `rho` is a correlation in [−1, 1], anything else is a `ValueError` since v2.6.1) bounds what a latent confounder A would do:

|θ_short − θ_long| ≤ |ρ| · S · √(cf_y · cf_d / (1 − cf_d)),  S² = E[(Y − θD − ℓ(X))²] · E[α²],  α = (D − m(X)) / E[(D − m(X))²]

- **cf_y**: share of the outcome's residual variance (given D and X) that A would explain; **cf_d**: share of the treatment's residual variance (given X) that A would explain; **ρ**: how aligned the two omitted pieces are (1 = worst case).
- It returns bounds, confidence bounds (sampling error added through influence functions), a table of the lower bound over a grid of (cf_y, cf_d) for a contour plot, and the **robustness values** RV (strength, equal on both sides, that moves θ to zero) and RVa (that makes it insignificant). With `q ≠ 1` (q > 0) the target is (1 − q)·θ: RV moves θ there and RVa puts that value inside the confidence bounds. Both are searched on cf ∈ (0, 1): RV = 100% means no confounder short of 100% reaches the target (always so with ρ = 0, where the bias bound is zero; with a tiny |ρ| the bound gets there only as cf → 1), and when the reported CI already contains the target the printed text says so instead of quoting RVa. With linear learners these reproduce Cinelli-Hazlett's OLS values exactly (tested).
- **Benchmark** with `dml_benchmark(df, y, d, X, ["income", "age"], res=res)`: it refits without each observed covariate and reports its implied cf_y, cf_d, ρ and the change in θ. "A confounder as strong as income would reduce θ from 0.21 to 0.12" is a sentence the reader can judge; RV = 3% alone is not. A covariate whose estimated share of the residual variance of Y or of D is zero or below (cf_y = 0 or cf_d = 0 after clipping, ρ undefined) is no benchmark: `dml_benchmark` warns, its cf_y and cf_d give a bias bound of zero (not a robustness result), and its NaN ρ is refused by `dml_sensitivity`. Since v2.6.2 `cf_y` and `cf_d` are the paper's **gains** (Appendix F, eq. 22–23, of arXiv 2112.13398v5; Appendix E in v6; as DoubleML's `sensitivity_benchmark`): the added explained variance over what the full model leaves unexplained, G_Y = (σ²_without − σ²_with)/σ²_with and G_D = (ν²_with − ν²_without)/ν²_without, per sample split and then the median, cut at 1 (ρ the same way); with the same splits they equal DoubleML's. Pass `dml_benchmark` the same seed, `n_folds` and `n_rep` as `dml_plr`; otherwise the splits cannot be paired and it warns and falls back to ratios of medians. Before, they were the partial R² G/(1 + G), now `r2_y` and `r2_d`, always smaller, so "as strong as X" was less conservative than in the paper.
- **"A confounder k times as strong as X"**: `dml_bounds(res, bench, kd=[1, 2, 3])` with `bench` from `dml_benchmark` (v2.6.2). The paper scales the gains by k: cf_y = ky·G_Y, cf_d = kd·G_D; ρ is the benchmark's observed one by default (the paper's benchmark for ρ, in the same appendix) or a number in [−1, 1] for every line (`rho=1`, the worst case). Conventions of the skill, which the paper does not state: cf_y above 1 is held at 1 (one warning per benchmark); kd ≥ `k_d_max` = 1/G_D is impossible (default kd: skipped with a warning; a kd you give: `ValueError`). Each line gives the bounds, the confidence bounds (the same computation as `dml_sensitivity`) and whether they include zero, as a worst case. **Never multiply `cf_y` and `cf_d` by k by hand** (the v2.6.1 validation did, with the partial R², ρ = 1 and a cut at 0.999 of its own). The gains and ρ are estimates whose own uncertainty is not propagated (as in the paper).

When the effect may vary with characteristics X, the same cross-fitting machinery estimates tau(x) instead of a single theta (`hte.cate_fit(..., method="linear_dml")`, the R-learner); see `heterogeneous-effects.md`.

## 5. Regression discontinuity

Sharp RD: treatment switches on at x = c. Effect at the cutoff = jump in E[Y|x].

```python
r = rdd(df, "y", "score", cutoff=50)                  # local linear, triangular kernel, MSE-optimal h
rob = rdd_robustness(df, "y", "score", 50, covs_to_test=["age", "female"])
r_f = rdd(df, "y", "score", 50, fuzzy="treated")      # fuzzy RD: local IV, LATE for compliers
```

- **Bandwidth**: `select_bandwidth` uses rdrobust's MSE-optimal `mserd` when installed, Imbens-Kalyanaraman otherwise, and says so: `rdd` then carries a warning to relay (v2.5.0.1). Conventional CIs from a MSE-optimal bandwidth under-cover; for the headline estimate use `rdrobust` (robust bias-corrected CIs, Calonico-Cattaneo-Titiunik 2014), the optional extra `rd` of `setup_env.py`, installed with the user's permission (SKILL.md §7).
- **Local linear, not high-order global polynomials** (Gelman & Imbens, 2019): global cubic/quartic fits produce noisy, misleading estimates.
- **Validity checks** (all in `rdd_robustness`): estimate across 0.5h–2h; local quadratic; placebo cutoffs where nothing should happen; no jumps in predetermined covariates; density test for manipulation (`density_test`, McCrary 2008; `rddensity` is the modern version). Bunching just above a threshold (e.g. claim values just below a small-claims court limit) invalidates the design.
- **Discrete running variable** (age in years, few distinct scores): cluster by value of the running variable is not a fix (Kolesár & Rothe, 2018); use their honest CIs or widen the interpretation.
- External validity: the effect is local to units near the cutoff.

## 6. DiD and event studies with staggered adoption

The 2×2 DiD is in `panel-iv.md` §10. With **units treated at different dates and effects that change over time**, two-way fixed effects (TWFE) compares late-treated units with already-treated ones, giving "forbidden comparisons" and even sign reversals (Goodman-Bacon, 2021; de Chaisemartin & D'Haultfœuille, 2020).

**Start with the design** (v2.5.0): `did_design(df, unit, time, d)` takes the treatment column `d` (binary or a dose, per unit and period) and classifies the panel into the five designs of de Chaisemartin & D'Haultfœuille's guide ("Which DiD estimator should I use?"). It also reports reversals, stayers (units whose treatment never changes), never-treated and always-treated units, cohorts and their pre-periods, balance and assignment below the cluster, and returns the `gvar` and `control` to use (only for a binary absorbing treatment).

| Design (`did_design`) | What defines it | In the skill | Outside the skill |
|---|---|---|---|
| `canonical` | binary, absorbing, one adoption date, never-treated units | `att_gt(control="never")`, `did_imputation`, `event_study`; TWFE acceptable without covariates | — |
| `staggered` | binary, absorbing, several adoption dates | `att_gt(control="notyet")` (the guide's recommended comparison group), `did_imputation`, `did_compare`; `bacon_decomposition` and `twfe_weights` to diagnose a TWFE | R/Stata `csdid`, `did_imputation`, `did_multiplegt_dyn` |
| `heterogeneous_adoption` | everyone untreated at the start, one date, different doses | only `twfe_weights` (diagnosis) | `did_had` |
| `first_switch` | non-binary or reversible treatment, first changes at different dates, with units not yet switched | only `twfe_weights` (diagnosis) | `did_multiplegt_dyn`; with a binary treatment and no lagged effects, also imputation estimators |
| `other` | no switcher has a comparable unit that has not switched (e.g. the treatment changes every period) | only `twfe_weights`: the TWFE coefficient is not a robust causal effect, say so | none, per the guide |

The guide does not define stayers or quasi-untreated units; `did_design` documents the definitions it uses. The Python versions of `did_had` and `did_multiplegt_dyn` listed by the guide were not checked.

- `event_study(df, unit, time, y, event_time, leads=4, lags=4)`: TWFE leads and lags, reference period −1, endpoints binned, errors clustered by unit, joint test that pre-period coefficients are zero. Fine with a single adoption date or homogeneous effects; a visual check otherwise.
- `att_gt(df, unit, time, y, gvar, control="never" | "notyet")`: Callaway & Sant'Anna (2021) group-time ATTs using only clean comparisons, aggregated to an overall ATT and to event time, with unit bootstrap SEs. Use this as the main estimate under staggered adoption; report the TWFE event study next to it. With `covariates=[...]` it estimates under conditional parallel trends (§6.0). With `cluster=` (the level of assignment, e.g. the state when units are municipalities) the bootstrap resamples whole clusters: in the skill's Monte Carlo with shared state shocks, 95% CIs covered 55% of the time resampling municipalities and 92.5% resampling states, with the same point estimate. Do not aggregate to the state to get clustered inference: that turns a municipality average into a state average, weighting large and small municipalities differently.
- **Joint pre-trend test of `att_gt`** (v2.5.0; `res["pretrend"]`): Wald test that the pre-period cells are zero, with the bootstrap covariance and an F(q, n − q) reference, n = the smaller of the resampled units (clusters) and the draws. A covariance estimated from n units supports few restrictions, so the test uses the most detailed level with q ≤ (n − 1)/10: every pre cell, the pre-period event-time averages, or their mean (one restriction); a coarser level is warned, because a pre-trend that averages out passes it. Size of a 5% test in the skill's simulations: 7.5% with 100 to 300 units and 9 cells, 5–7% with 30 to 80 clusters, about 9% with 8 clusters (one restriction; the cluster bootstrap SE is too small with so few clusters, and `att_gt` warns). The χ² reference gave 13% with 100 units and 9 cells, and 83% with 8 clusters and 8 restrictions.
- `did_imputation(df, unit, time, y, gvar)`: the imputation estimator of Borusyak, Jaravel & Spiess (2024). Unit and period effects are fitted on the untreated observations only, Y(0) is imputed for every treated observation, and the ATT (and event study) averages τ̂ = Y − Ŷ(0). SEs: the paper's clustered variance (Theorem 3; `cluster=` the assignment level). The panel may be unbalanced; treated observations that cannot be imputed (always-treated units, periods with no untreated unit) are dropped and counted. Its pre-trend test uses only untreated observations, separate from estimation, and leaves out leads that the fixed effects span (without never-treated units, possibly all). Replicated against R `didimputation` 0.5.1 (ATT, SEs, unbalanced panel, event-study horizons). Its leads start at e = −1, while R's `pretrends=TRUE` uses e = −1 as the reference, so lead coefficients differ between the two by construction.
- `did_compare(df, unit, time, y, gvar, control=...)`: Callaway-Sant'Anna against the imputation estimator on the post-period cells both define, with the same weights. Both routes run without covariates and on a balanced panel, and `control` defaults to `"notyet"` (not `att_gt`'s `"never"`): pass the headline specification's control, and say so when the headline `att_gt` uses covariates; the difference is tested at alpha with a paired bootstrap (both routes recomputed on the same resampled units). The guide reports that the two families stay close except where pre-trend tests reject: a significant gap is a finding. It has modest power against pre-trends (43% against a linear pre-trend of 0.2 per period in the skill's tests), so it complements the pre-trend tests and does not replace them. `reporting.cross_check(res["cross_check"], se=res["diff_se"], tol_se=Φ⁻¹(1 − α/2))` records it as an independent check.
- `twfe_weights(df, unit, time, d, y=None)`: the weights of de Chaisemartin & D'Haultfœuille (2020) behind a TWFE coefficient (binary or dose, unbalanced allowed). β_fe = Σ w·Δ over treated cells, with weights that sum to one but can be negative (early adopters late in the panel act as controls). It reports the negative weights and `sigma_fe`, the standard deviation of effects across treated cells under which the average effect could be zero while TWFE reports β_fe. Replicated against R `TwoWayFEWeights` 2.1.0 (β, counts and sums of weights, `sigma_fe`, binary and dose); `sigma_fe` follows that package's N/(N − 1) convention.
- `bacon_decomposition(df, unit, time, y, gvar)`: the Goodman-Bacon (2021) decomposition of the TWFE coefficient into every 2×2 and its weight (sums exactly to TWFE). The weight on **later vs earlier (control)** comparisons, which use already-treated units as controls, measures how much of TWFE can be contaminated by effects that grow with exposure; the treated weights can all be positive while this share is large.
- **Pre-trends**: a non-significant pre-trend test is weak evidence (low power; Roth, 2022). Plot the leads with CIs; consider HonestDiD-style sensitivity (Rambachan & Roth, 2023): how large a violation of parallel trends would overturn the result?
- **Anticipation**: if behaviour changes before the formal date (law announced before it takes effect), either shift the event date to the announcement or use `att_gt(..., anticipation=δ)`, which takes the base period δ+1 periods before the date. Both `att_gt` and `event_study` print a WARNING when the outcome already moves in one of the two periods right before the recorded date (`flags["effect_before_date"]`).

### 6.0 Covariates: parallel trends that hold only conditional on X

Often treated and comparison units differ in characteristics that drive outcome **trends** (income, firm size, urbanization, age structure), so parallel trends hold only among units with the same covariates. Adding the covariates to a TWFE regression does not fix this, and can make it worse (Caetano & Callaway, 2024; Karim & Webb, 2024):

- **Hidden linearity bias.** Unit fixed effects difference the covariates too, so TWFE uses only their changes ΔX. If trends depend on covariate **levels**, that information is thrown away, and time-invariant covariates drop out completely. In the skill's simulation (covariate effect rising from 1 to 2, treated units with higher X, true ATT 1.0), the unadjusted DiD gave 1.98, TWFE with the covariate 1.99, and the regression adjustment on levels 1.00.
- **Common causal covariates (CCC).** TWFE and Callaway-Sant'Anna with covariates also assume the covariate's effect is the same across groups and periods. If it differs across groups, every estimator below is biased.

Workflow, in this order:

1. **Choose covariates by design** (`dag.adjustment_report`): pre-treatment characteristics or time-varying ones that the treatment cannot affect. A covariate the treatment moves (employment after a hiring subsidy) is a bad control here too.
2. **Test CCC** with `ccc_test(df, y, covariates, group, time, gvar=..., cluster=unit)`. It uses only untreated observations, so treatment cannot contaminate the slopes. `group` is the level of assignment (state, region, cohort), with several observations per group-period cell. The default inference is a **wild bootstrap imposed under each null**: with many restrictions and heteroskedastic errors the asymptotic χ² over-rejects (10–25% for a nominal 5% in the skill's tests, cells of 40–120 observations), and the bootstrap brings it to 4–7%. Read the selection:

| `selected` | Meaning | Estimate with |
|---|---|---|
| `homogeneous` | Slopes look equal across groups and periods (not proof: power falls with small cells) | `att_gt(covariates=...)`; TWFE acceptable on this front |
| `time` | Slopes change over time, common across groups | `att_gt(covariates=...)` (each group-time cell is adjusted on levels; `x_periods="base"` for baseline or time-invariant covariates, `"both"` only for covariates that vary over time and that treatment cannot affect); TWFE with covariates is biased |
| `group` / `twoway` | Slopes differ across groups | Intersection DiD (DID-INT; Karim & Webb: Julia `DiDInt.jl`, R `didintrjl`), not implemented here; or interact covariates with group and say the estimand changed |

3. **Estimate** with `att_gt(df, unit, time, y, gvar, covariates=[...], est_method="dr")`. Covariates enter in **levels** in each group-time 2×2:
   - `est_method="dr"` (default): doubly robust (Sant'Anna & Zhao, 2020), consistent if either the outcome regression or the propensity score is right.
   - `"reg"`: outcome regression (Heckman, Ichimura & Todd, 1997).
   - `"ipw"`: normalized inverse probability weighting (Abadie, 2005).

   `x_periods="base"` uses the base-period values (safe when X could respond to treatment). `"both"` adds the change in X up to t (time-varying covariates not affected by treatment). The bootstrap re-estimates every nuisance model. Cell by cell, the estimates match the `csdid`/DRDID implementation to 10⁻¹¹ (tests).
4. **Read the built-in independent check.** With `verify=True` (default), the other two estimators are recomputed and stored in `res["cross_check"]`; a WARNING fires when they differ by more than one SE. Outcome and propensity models that disagree mean at least one is misspecified: add flexibility (squares, interactions, logs) or check overlap. Comparison units with propensity ≥ 0.995 get zero weight (warned with the count): the treated units there have no comparable controls.
5. **If a TWFE regression with covariates is on the table** (the client's, a paper's), run `twfe_covariate_diagnostic(df, unit, time, y, gvar, covariates=<their controls>, cs_covariates=<valid pre-treatment confounders>, cluster=<assignment level>)`. Put the controls under review in `covariates` even if some are bad controls, and only the valid ones in `cs_covariates`: the benchmark and the hidden-linearity test use the latter. It reports:
   - the joint test of covariate **levels** in untreated first-difference trends (the hidden linearity test);
   - the share of treated observations with negative implicit TWFE weight (Frisch-Waugh);
   - the Goodman-Bacon weight and average of the forbidden (already-treated control) comparisons;
   - the gap to `att_gt` with covariates, in SE units.

   The verdict is "unreliable" if any check fails; then report `att_gt`. "Inconclusive (low power)" means the benchmark CI is too wide to detect the gap: relay it as such, never as support for TWFE.
6. **Pass the same covariates and cluster** to `did_timing_sensitivity` and `did_uncertain_dates` (both accept `covariates`, `est_method`, `x_periods`, `cluster`), so the date checks use the main specification and inference.
7. **Verify** with `reporting.cross_check(res["cross_check"], se=res["overall"]["se"])`: `res["cross_check"]` is a Series (dr, reg, ipw) that the function takes as is; with fewer than two routes it reports that nothing was checked.

Covariates do not rescue a design whose parallel trends fail for unobserved reasons; the pre-period cells of `att_gt` (event time < 0) remain the check, now conditional on X.

### 6.1 When the treatment date itself is uncertain

A law phased in court by court, a policy with imprecise records, an announcement weeks before entry into force: the estimate can be driven by the date rather than the policy. A date that is too late puts treated periods in the base and attenuates the ATT; a date that is too early puts untreated periods in the post period. **Before running any DiD, ask the user:** (1) is the adoption date exact, and from what source? (2) was it announced, or did behaviour start, before that date? (3) did adoption happen at once or gradually within each unit? Then:

| Situation | Function | What it reports |
|---|---|---|
| Date recorded, but could be off by up to k periods | `did_timing_sensitivity(df, unit, time, y, gvar, max_shift=k)` | ATT with all dates shifted −k…+k, and a **date-robust ATT** (event date g+k, anticipation 2k: the base precedes and the post follows every date in the window). Verdict: survives or depends on the exact date |
| Each unit's date known only within a window | `did_uncertain_dates(df, unit, time, y, g_low, g_high)` | **Window-robust ATT** (event date g_high, anticipation = widest window) and the spread of the ATT across random dates inside the windows |
| Known announcement lag of δ periods | `att_gt(..., anticipation=δ)` | ATT using base g−δ−1 |

Read the shift table as mechanical, not as a test: shifting a correct date also moves the ATT, because treated periods enter the base. The verdict is the date-robust or window-robust estimate. It estimates the effect from the latest plausible date onward (a later-exposure ATT when effects grow over time) and drops cohorts whose earlier base falls outside the panel, so the CI is wider. If it does not exclude zero, report the result as dependent on the date and say what records would settle it.

Not covered: intersection DiD for covariate effects that differ across groups (DID-INT; use `DiDInt.jl` or R `didintrjl`, after `ccc_test` selects `group` or `twoway`), estimation with treatment intensity that varies within a unit or with reversals (dose; de Chaisemartin & D'Haultfœuille, 2024; Callaway, Goodman-Bacon & Sant'Anna, 2024: `did_design` identifies these designs and `twfe_weights` diagnoses a TWFE on them, but the estimator is R/Stata `DIDmultiplegtDYN` or `did_had`) and formal HonestDiD bounds (Rambachan & Roth, 2023); use R's `did`, `DIDmultiplegtDYN` and `HonestDiD` for those.
- **Clustering**: at the level of treatment assignment (state, court, municipality); with fewer than ~30 clusters, use wild cluster bootstrap (`estimation.wild_cluster_bootstrap`); with few **treated** clusters (about 1–5 adopting states), add randomization inference with `causal.randomization_inference(df, unit, time, y, gvar, cluster=...)` (reassigns the observed adoption dates to random sets of clusters, the level of assignment, and ranks the actual estimate; p-value and test-inversion CI), because the wild bootstrap also degrades there. With one to three treated units in total, no permutation has enough assignments: use `did_single_treated` (inference from the gap's time series). The full set of checks, each with its MDE, is `did_falsification` (v2.5.1). It assumes the treated units are exchangeable with the others (similar error variance and dynamics); if adopters are systematically more volatile, use Ferman-Pinto or Conley-Taber.

### 6.2 Power before the study

A DiD with an insignificant result is uninformative unless it could have detected a relevant effect. Before collecting data or choosing the sample, compute the minimum detectable effect:
- `simulation.mde_did(n_units=..., n_pre=m, n_post=r, sd=..., rho=...)`: analytic MDE (McKenzie 2012) with equicorrelated errors: Var = σ²(1 − ρ)(1/m + 1/r) / (P(1 − P)N). More periods and higher ρ lower the MDE.
- `simulation.did_power_inputs(history, unit, time, y)`: estimates the simulation inputs from the pre-period panel (SD of the two-way FE residuals, which is usually much smaller than the total SD a user quotes, and their AR(1) coefficient with the Nickell correction). Feed them to:
- `simulation.power_did_sim(n_units, T, share_treated, effect, adoption, rho_ar, ...)`: simulates the panel you will have (unit effects, common shocks, **AR(1) errors**, simultaneous or staggered adoption) and the inference you will use (TWFE with unit-clustered SEs, t(G−1) critical values); reports power, size and coverage. `mde_did_sim` searches the effect with 80% power.
- With decaying serial correlation the analytic formula overstates power (Burlig, Preonas & Woerman 2020): in the skill's tests, a design with 80% analytic power has 51% simulated power at AR(1) ρ = 0.8, and the MDE rises by 40%. Use the simulation for the final number; with fewer than ~10 treated units, plan randomization inference (the clustered SE over-rejects: 10% size with 3 treated of 12).
- When the outcome is a stock that responds slowly (portfolio default rate, inventory), persistence is high and power low: prefer a flow measure that only the new policy touches (default by origination cohort), and count the months it needs to mature.
- After the study, report the MDE next to an insignificant estimate ("the design could detect effects above X; the CI excludes effects above Y").

## 7. Synthetic control

`synth(df, unit, time, y, treated="SP", t0=2016)`:
- Weights on donor units (non-negative, sum to one) chosen to reproduce the treated unit's **pre-treatment** path; the post-treatment gap is the estimated effect.
- Check the pre-period fit (RMSPE) before reading the post-period: a poor fit means no credible counterfactual.
- **Inference by placebo-in-space**: re-run the method treating each donor as if treated; p-value = rank of the treated unit's post/pre RMSPE ratio. With J donors, the smallest attainable p-value is 1/(J+1).
- Exclude donors exposed to similar shocks or to spillovers from the treatment; restrict to units similar in level and trend.
- Robustness: leave-one-out on donors with positive weight; in-time placebo (pretend treatment happened earlier).
- Extensions: augmented synthetic control (Ben-Michael, Feller & Rothstein, 2021) when the fit is imperfect; synthetic DiD (Arkhangelsky et al., 2021) for several treated units.

## 8. Designs common in law and economics

- **Random assignment of cases to judges / examiners**: the judge's leave-one-out leniency (share of other cases she decided a certain way) is an instrument for the decision in a given case (Kling, 2006; Dobbie, Goldin & Yang, 2018). Conditions: random assignment within court × period (test by regressing leniency on case characteristics), exclusion (the judge affects outcomes only through the decision), monotonicity. Cluster by judge.
- **Movers designs**: when decision makers move between courts/units, comparing their behaviour before and after the move, relative to the old and new peers, separates individual style from local norms (in the spirit of Abowd-Kramarz-Margolis and of the Finkelstein-Gentzkow-Williams place effects). Event-study around the move with mover fixed effects; **placebo with future peers** (similarity to the peers of the court the judge will join, measured before the move, should not change) is a strong check against reverse causality and selection of destination.
- **Text-based outcomes** (similarity of a decision to peers' decisions, topic shares, legal citations): validate the measure (`text-as-data.md`) and treat it as an outcome with measurement error. Similarity measures move mechanically with document length and boilerplate; control for them.
- **Changes in law with staggered adoption across states/courts**: §6.
- **Thresholds in procedural rules** (value of the claim for small-claims courts, deadlines, age limits): RD, with a density test for strategic filing.
- **Selection into litigation** (Priest-Klein): the cases that reach a decision are not a random sample of disputes; outcomes at trial can move towards 50% regardless of the legal standard. State this limitation when inferring from decided cases.

Qualitative evidence (interviews, case studies, process tracing) complements these designs: it explains mechanisms, suggests confounders and checks that the quantitative variables mean what we think they mean.

## 9. Reporting checklist

- [ ] Estimand (ATE, ATT, LATE, effect at the cutoff) stated.
- [ ] Identifying assumption in plain words, with the evidence for it (balance, pre-trends, density test, pre-period fit, first stage).
- [ ] Main estimate with CI; at least one alternative estimator; robustness (bandwidths, donors, controls, comparison groups).
- [ ] Placebo outcomes or placebo timing.
- [ ] Clustering at the level of assignment.
- [ ] Controls justified by a causal graph (no mediators, colliders or post-treatment variables); the graph shown.
- [ ] Sensitivity to unobserved confounding (Oster, Cinelli-Hazlett or `dml_sensitivity`, benchmarked on observed covariates) or to violations of parallel trends.
- [ ] Power or MDE of the design, especially next to an insignificant result.
- [ ] What the design cannot say (external validity, mechanisms).
