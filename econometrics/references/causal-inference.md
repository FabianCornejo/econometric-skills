# Causal inference with observational data

Use when the question is "what is the effect of X on Y?" and X was not randomized: a policy, a law, a court decision, a program, a promotion, a merger, a judge's experience. Code: `scripts/causal.py`. IV and basic difference-in-differences are in `panel-iv.md`; this file covers design choice, selection on observables, regression discontinuity, synthetic control, modern DiD and ML-based estimators.

## Contents
1. Potential outcomes and the design-first rule
2. Choosing a design
3. Selection on observables: regression, matching, weighting, doubly robust
4. Double/debiased machine learning
5. Regression discontinuity
6. Difference-in-differences and event studies with staggered adoption
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

When several designs are possible, run more than one; agreement across designs with different assumptions is the strongest evidence available.

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
5. **Sensitivity to unobserved confounding**: report how strong a hidden confounder would need to be to explain away the result (Oster, 2019, coefficient stability δ; Cinelli-Hazlett, 2020, robustness value via the `sensemakr` approach). Unconfoundedness cannot be tested; it can only be stress-tested.

## 4. Double/debiased machine learning

`dml_plr(df, y, d, X)` estimates θ in Y = θD + g(X) + U with flexible g (random forest by default; any scikit-learn regressor, e.g. `LassoCV`, gradient boosting):
1. Cross-fitting: split the sample in K folds; predict Y and D from X with models trained on the other folds.
2. Regress the residual of Y on the residual of D (Frisch-Waugh-Lovell with ML).
3. Repeat over several splits and take the median (`n_rep`).

Use it when there are many controls or nonlinear confounding. It does **not** solve unobserved confounding: DML is selection on observables with better functional form. Prediction-oriented ML (lazy comparisons of 30 classifiers, maximizing accuracy) answers a different question: good prediction of Y is not an effect of D.

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
- **Anticipation**: if behaviour changes before the formal date (law announced before it takes effect), shift the event date to the announcement.
- **Clustering**: at the level of treatment assignment (state, court, municipality); with fewer than ~30 clusters, use wild cluster bootstrap.

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
- [ ] Sensitivity to unobserved confounding or to violations of parallel trends.
- [ ] What the design cannot say (external validity, mechanisms).
