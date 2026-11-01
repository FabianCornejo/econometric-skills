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

It returns the **minimal adjustment sets** (backdoor criterion), the **optimal set** (Henckel, Perković & Maathuis 2022: among valid sets, the one with the smallest variance; it adds outcome predictors and drops pure treatment predictors), the **role of every variable** and, with data, **tests of the graph's implications**.

| Role (`classify_controls`) | Example | Control? |
|---|---|---|
| Confounder (causes x and y) | income for a discount's effect on purchases | Yes (or a set that blocks its path) |
| Mediator (x → v → y) | site visits caused by the discount | No: removes part of the effect (unless the question is the direct effect) |
| Descendant of x / collider | a review written after the purchase | No: post-treatment, can open a path |
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

DML removes functional-form bias, not omitted-variable bias. `dml_sensitivity(res, cf_y, cf_d, rho=1)` (Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis 2022, "Long Story Short") bounds what a latent confounder A would do:

|θ_short − θ_long| ≤ |ρ| · S · √(cf_y · cf_d / (1 − cf_d)),  S² = E[(Y − θD − ℓ(X))²] · E[α²],  α = (D − m(X)) / E[(D − m(X))²]

- **cf_y**: share of the outcome's residual variance (given D and X) that A would explain; **cf_d**: share of the treatment's residual variance (given X) that A would explain; **ρ**: how aligned the two omitted pieces are (1 = worst case).
- It returns bounds, confidence bounds (sampling error added through influence functions), a table of the lower bound over a grid of (cf_y, cf_d) for a contour plot, and the **robustness values** RV (strength, equal on both sides, that moves θ to zero) and RVa (that makes it insignificant). With linear learners these reproduce Cinelli-Hazlett's OLS values exactly (tested).
- **Benchmark** with `dml_benchmark(df, y, d, X, ["income", "age"], res=res)`: it refits without each observed covariate and reports its implied cf_y, cf_d, ρ and the change in θ. "A confounder as strong as income would reduce θ from 0.21 to 0.12" is a sentence the reader can judge; RV = 3% alone is not.

When the effect may vary with characteristics X, the same cross-fitting machinery estimates tau(x) instead of a single theta (`hte.cate_fit(..., method="linear_dml")`, the R-learner); see `heterogeneous-effects.md`.

## 5. Regression discontinuity

Sharp RD: treatment switches on at x = c. Effect at the cutoff = jump in E[Y|x].

```python
r = rdd(df, "y", "score", cutoff=50)                  # local linear, triangular kernel, MSE-optimal h
rob = rdd_robustness(df, "y", "score", 50, covs_to_test=["age", "female"])
r_f = rdd(df, "y", "score", 50, fuzzy="treated")      # fuzzy RD: local IV, LATE for compliers
```

- **Bandwidth**: `select_bandwidth` uses rdrobust's MSE-optimal `mserd` when installed, Imbens-Kalyanaraman otherwise. Conventional CIs from a MSE-optimal bandwidth under-cover; for the headline estimate use `rdrobust` (robust bias-corrected CIs, Calonico-Cattaneo-Titiunik 2014): `pip install rdrobust`.
- **Local linear, not high-order global polynomials** (Gelman & Imbens, 2019): global cubic/quartic fits produce noisy, misleading estimates.
- **Validity checks** (all in `rdd_robustness`): estimate across 0.5h–2h; local quadratic; placebo cutoffs where nothing should happen; no jumps in predetermined covariates; density test for manipulation (`density_test`, McCrary 2008; `rddensity` is the modern version). Bunching just above a threshold (e.g. claim values just below a small-claims court limit) invalidates the design.
- **Discrete running variable** (age in years, few distinct scores): cluster by value of the running variable is not a fix (Kolesár & Rothe, 2018); use their honest CIs or widen the interpretation.
- External validity: the effect is local to units near the cutoff.

## 6. DiD and event studies with staggered adoption

The 2×2 DiD is in `panel-iv.md` §10. With **units treated at different dates and effects that change over time**, two-way fixed effects (TWFE) compares late-treated units with already-treated ones, giving "forbidden comparisons" and even sign reversals (Goodman-Bacon, 2021; de Chaisemartin & D'Haultfœuille, 2020).

- `event_study(df, unit, time, y, event_time, leads=4, lags=4)`: TWFE leads and lags, reference period −1, endpoints binned, errors clustered by unit, joint test that pre-period coefficients are zero. Fine with a single adoption date or homogeneous effects; a visual check otherwise.
- `att_gt(df, unit, time, y, gvar, control="never" | "notyet")`: Callaway & Sant'Anna (2021) group-time ATTs using only clean comparisons, aggregated to an overall ATT and to event time, with unit bootstrap SEs. Use this as the main estimate under staggered adoption; report the TWFE event study next to it.
- **Pre-trends**: a non-significant pre-trend test is weak evidence (low power; Roth, 2022). Plot the leads with CIs; consider HonestDiD-style sensitivity (Rambachan & Roth, 2023): how large a violation of parallel trends would overturn the result?
- **Anticipation**: if behaviour changes before the formal date (law announced before it takes effect), either shift the event date to the announcement or use `att_gt(..., anticipation=δ)`, which takes the base period δ+1 periods before the date. Both `att_gt` and `event_study` print a WARNING when the outcome already moves in one of the two periods right before the recorded date (`flags["effect_before_date"]`).

### 6.1 When the treatment date itself is uncertain

A law phased in court by court, a policy with imprecise records, an announcement weeks before entry into force: the estimate can be driven by the date rather than the policy. A date that is too late puts treated periods in the base and attenuates the ATT; a date that is too early puts untreated periods in the post period. **Before running any DiD, ask the user:** (1) is the adoption date exact, and from what source? (2) was it announced, or did behaviour start, before that date? (3) did adoption happen at once or gradually within each unit? Then:

| Situation | Function | What it reports |
|---|---|---|
| Date recorded, but could be off by up to k periods | `did_timing_sensitivity(df, unit, time, y, gvar, max_shift=k)` | ATT with all dates shifted −k…+k, and a **date-robust ATT** (event date g+k, anticipation 2k: the base precedes and the post follows every date in the window). Verdict: survives or depends on the exact date |
| Each unit's date known only within a window | `did_uncertain_dates(df, unit, time, y, g_low, g_high)` | **Window-robust ATT** (event date g_high, anticipation = widest window) and the spread of the ATT across random dates inside the windows |
| Known announcement lag of δ periods | `att_gt(..., anticipation=δ)` | ATT using base g−δ−1 |

Read the shift table as mechanical, not as a test: shifting a correct date also moves the ATT, because treated periods enter the base. The verdict is the date-robust or window-robust estimate. It estimates the effect from the latest plausible date onward (a later-exposure ATT when effects grow over time) and drops cohorts whose earlier base falls outside the panel, so the CI is wider. If it does not exclude zero, report the result as dependent on the date and say what records would settle it.

Not covered: treatment intensity that varies continuously within a unit (dose; de Chaisemartin & D'Haultfœuille, 2024; Callaway, Goodman-Bacon & Sant'Anna, 2024) and formal HonestDiD bounds (Rambachan & Roth, 2023); use R's `did`, `DIDmultiplegtDYN` and `HonestDiD` for those.
- **Clustering**: at the level of treatment assignment (state, court, municipality); with fewer than ~30 clusters, use wild cluster bootstrap (`estimation.wild_cluster_bootstrap`); with few **treated** clusters (about 1–5 adopting states), add randomization inference with `causal.randomization_inference(df, unit, time, y, gvar)` (reassigns the observed adoption dates to random sets of units and ranks the actual estimate; p-value and test-inversion CI), because the wild bootstrap also degrades there. It assumes the treated units are exchangeable with the others (similar error variance and dynamics); if adopters are systematically more volatile, use Ferman-Pinto or Conley-Taber.

### 6.2 Power before the study

A DiD with an insignificant result is uninformative unless it could have detected a relevant effect. Before collecting data or choosing the sample, compute the minimum detectable effect:
- `simulation.mde_did(n_units=..., n_pre=m, n_post=r, sd=..., rho=...)`: analytic MDE (McKenzie 2012) with equicorrelated errors: Var = σ²(1 − ρ)(1/m + 1/r) / (P(1 − P)N). More periods and higher ρ lower the MDE.
- `simulation.power_did_sim(n_units, T, share_treated, effect, adoption, rho_ar, ...)`: simulates the panel you will have (unit effects, common shocks, **AR(1) errors**, simultaneous or staggered adoption) and the inference you will use (TWFE with unit-clustered SEs, t(G−1) critical values); reports power, size and coverage. `mde_did_sim` searches the effect with 80% power.
- With decaying serial correlation the analytic formula overstates power (Burlig, Preonas & Woerman 2020): in the skill's tests, a design with 80% analytic power has 51% simulated power at AR(1) ρ = 0.8, and the MDE rises by 40%. Use the simulation for the final number; with fewer than ~10 treated units, plan randomization inference (the clustered SE over-rejects: 10% size with 3 treated of 12).
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
