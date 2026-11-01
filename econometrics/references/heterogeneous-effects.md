# Heterogeneous treatment effects and treatment policies

Use when the question is not only "what is the effect?" but "for whom is it larger?" or "whom should we treat?": which customers respond to a discount, which defendants are affected by a procedural rule, where a program pays for its cost. Code: `scripts/hte.py` (numpy/scikit-learn only). The design follows EconML (PyWhy) and grf; their estimators can replace the ones here (§8).

## Contents
1. Estimands and when to use this file
2. Identification: the same as the ATE, plus overlap everywhere
3. Choosing an estimator
4. Workflow
5. Reading the validation tools
6. Policy learning
7. Worked example
8. Using EconML or grf
9. Pitfalls
10. Reporting checklist

## 1. Estimands

- **CATE** tau(x) = E[Y(1) − Y(0) | X = x]: the average effect for units with characteristics x. Not the effect for one person (that is never identified), but for everyone who shares x.
- **GATE**: the average effect in a group (e.g. a quintile of predicted tau).
- **Policy value**: the average gain of a rule pi(x) ∈ {treat, do not treat}, net of the treatment cost.

Which file to use:

| Question | File |
|---|---|
| Average effect of a policy or decision | `causal-inference.md` |
| Does the effect differ across groups the user named (e.g. by sex, region)? | Interactions with pre-specified groups (`ols-regression.md` §3) first; this file when there are many candidate modifiers |
| For whom is the effect larger? Whom to target? | This file |
| Predict an outcome and find segments with different drivers (no treatment) | `limited-dependent.md` §9 (logit leaf model): predictive, not causal |

## 2. Identification

Exactly the same assumptions as the ATE under selection on observables (`causal-inference.md` §3): unconfoundedness given (X, W) and overlap. Heterogeneity adds two demands:
- **Overlap must hold within the subgroups** you report: an effect for "large firms" is extrapolated if almost all large firms are treated. `cate_fit` warns when more than 5% of propensity scores are clipped.
- **X versus W.** X are the effect modifiers, chosen before looking at outcomes, few and interpretable (the CATE is a function of them). W are further confounders used only to remove bias. A variable can be in both. Never put post-treatment variables in either.

With an instrument instead of unconfoundedness the target is a conditional LATE (EconML's `OrthoIV`, `DRIV`, grf's `instrumental_forest`); with an experiment, unconfoundedness holds by design and W only improves precision.

## 3. Choosing an estimator

All methods cross-fit the nuisance models E[Y|X,W], E[T|X,W] (and E[Y|T=t,X,W] for binary t) over K folds, so that flexible learners do not bias the final stage (Chernozhukov et al. 2018).

| `method` | Treatment | Form of tau(x) | Pointwise CIs | When | EconML equivalent |
|---|---|---|---|---|---|
| `"dr"` | Binary | Linear in X (best linear projection of the true tau) | Yes (HC1 on DR scores) | Default for binary t: interpretable coefficients, valid inference even if tau is nonlinear (then it is the best linear approximation) | `LinearDRLearner` |
| `"linear_dml"` | Binary or continuous | Linear in X (assumed) | Yes | Continuous treatment (price, dose, hours); efficient when tau is linear | `LinearDML` |
| `"forest"` | Binary | Flexible (random forest on DR scores) | No | Nonlinear or interactive heterogeneity, many modifiers | `ForestDRLearner`; `CausalForestDML` adds CIs |
| `"t_learner"` | Binary | Flexible (difference of two outcome models) | No | Large samples, balanced arms; baseline | `TLearner` |
| `"s_learner"` | Binary | Flexible (one model with t as a feature) | No | Shrinks effects toward zero: a conservative check | `SLearner` |
| `"x_learner"` | Binary | Flexible | No | Unbalanced arms (few treated) | `XLearner` |

Default nuisance learners are random forests; pass any scikit-learn estimator (`outcome_learner=`, `ps_learner=`, a classifier for a binary t). With few observations or a linear world, linear learners are faster and as good. **Fit a linear and a flexible model on the same seed and compare them** (`compare_cate_models`): agreement is reassuring; a linear model can completely miss symmetric heterogeneity (effect larger at both extremes of x), which a forest catches.

## 4. Workflow

1. **Study card** (`workflow.md` §0): estimand, X, W, why unconfoundedness is plausible, the cost of treatment if a policy is wanted.
2. **Overlap**: read `res["overlap"]`; trim or redefine the population if the warning fires.
3. **Fit two models** with the same `seed`: `dr` (or `linear_dml`) and `forest`.
4. **Is the heterogeneity real?** `blp_test` on each. If the differential coefficient is not significantly positive, stop: report the ATE and say the data do not show differences.
5. **How large and where?** `gates` (monotone groups, top − bottom test), `clan` (who is in the top and bottom groups), the `coef` table of the linear model for direction.
6. **Does the ranking target well?** `toc` (AUTOC, Qini) and `rscore`; `compare_cate_models` puts everything side by side.
7. **Policy** (if asked): `policy_tree` with the cost per treated unit, valued on held-out units.
8. **Robustness**: another seed, other nuisance learners, another K; drop the most influential modifier; sensitivity to unobserved confounding for the ATE (`specification.oster` / `sensitivity` on the linear analogue).

## 5. Reading the validation tools

All validation uses **out-of-fold** predictions `res["tau"]`: a model scored on the data it was trained on always looks heterogeneous.

| Tool | What it estimates | How to read it |
|---|---|---|
| `blp_test` | Regression of Y − m(X) on (T − e)·mean(tau) and (T − e)·(tau − mean(tau)) | `mean_prediction` ≈ 1: the average is right. `differential_prediction` > 0 with one-sided p < 0.05: the predicted heterogeneity is real; ≈ 1: well calibrated; between 0 and 1: real but overstated (shrink toward the ATE) |
| `gates` | ATE within quantile groups of predicted tau, with DR scores | Monotone increase and a significant top − bottom difference are the clearest evidence; report the table, not only the extremes |
| `clan` | Mean characteristics of the most vs least affected group | Describes who benefits; Holm-adjusted p-values because many variables are compared. Not a causal explanation of why |
| `toc` | TOC(q) = ATE among the top q share − overall ATE; AUTOC = area; Qini = q-weighted area | > 0 and significant: the ranking finds units with larger effects. AUTOC emphasizes the very top, Qini the whole ranking |
| `rscore` | 1 − R-loss(model) / R-loss(constant effect) | > 0: better than assuming one effect for all; compare models on the same residuals (`compare_cate_models` does it) |

The ATE in `res["ate"]` is the cross-fitted AIPW estimate for every binary-treatment method, so models differ only in tau(x); for a continuous treatment it is the linear CATE at the sample mean of X, i.e. the average marginal effect.

## 6. Policy learning

`policy_tree(res, df, features, cost, depth=2)` searches all depth-1/2 trees over quantile thresholds for the rule that maximizes the sum of doubly robust gains psi_i − cost among treated units (Athey & Wager 2021). It learns on half of the units and reports, on the other half:
- value of the rule relative to treating no one;
- value of treating everyone;
- gain of the rule over treating everyone (the usual business question).

Rules of use: the cost is per treated unit and in the units of y (margin lost to a discount, cost of a program); restrict `features` to variables that can legally and practically be used to assign treatment; depth 2 is usually the right balance between value and explainability; report the rule as text (`res["rules"]`).

## 7. Worked example

```python
import sys; sys.path.insert(0, "<skill>/scripts")
import hte
X = ["tenure", "monthly_spend", "n_complaints"]          # effect modifiers, chosen in advance
W = ["region_code", "age", "channel_code"]               # extra confounders
dr = hte.cate_fit(df, "churn_90d", "got_discount", X, W, method="dr", seed=1)
rf = hte.cate_fit(df, "churn_90d", "got_discount", X, W, method="forest", seed=1)
print(dr["ate"]); print(dr["coef"])                      # average effect; direction of heterogeneity
print(hte.blp_test(rf)["verdict"])                       # is it real?
print(hte.gates(rf)["table"]); print(hte.clan(rf, df, X + W))
print(hte.compare_cate_models({"linear DR": dr, "forest": rf}))
pt = hte.policy_tree(rf, df, X, cost=0.02, depth=2)      # cost in units of y
print(pt["rules"]); print(pt["value_test"])
```

## 8. Using EconML or grf

EconML (`pip install econml`) offers causal forests with valid pointwise CIs (`CausalForestDML`, bootstrap of little bags), IV versions (`OrthoIV`, `DRIV`, `CausalIVForest`), dynamic treatments over time (`DynamicDML`), sparse high-dimensional final stages (`SparseLinearDML`), SHAP explanations of tau(x) and its own `DRTester` (BLP, calibration, uplift). R's grf adds `rank_average_treatment_effect` and `policy_tree`. To validate their predictions with this module, feed **out-of-fold** predictions:

```python
from sklearn.model_selection import KFold
from econml.dml import CausalForestDML
tau_oof = np.zeros(len(df))
for tr, te in KFold(5, shuffle=True, random_state=1).split(df):
    est = CausalForestDML(discrete_treatment=True, random_state=1).fit(
        df[y].iloc[tr], df[t].iloc[tr], X=df[X].iloc[tr], W=df[W].iloc[tr])
    tau_oof[te] = est.effect(df[X].iloc[te])
res = hte.cate_fit(df, y, t, X, W, method="dr", seed=1)      # nuisances and DR scores
print(hte.blp_test(res, tau=tau_oof)); print(hte.gates(res, tau=tau_oof)["table"])
```
(Rows of `df` must have no missing values in y, t, X, W so that both use the same sample.)

## 9. Pitfalls

- **Fishing in subgroups**: with 20 candidate subgroups, one will look significant by chance. Pre-specify X; let BLP and GATES speak for the whole ranking; adjust CLAN p-values (Holm).
- **In-sample tau**: the spread of fitted tau(x) is not evidence of heterogeneity; regularized ML spreads predictions even when the true effect is constant.
- **Confusing prediction of Y with effect of T**: a variable that predicts churn strongly need not modify the effect of the discount.
- **Bad controls**: post-treatment variables in X or W bias the CATE as they bias the ATE.
- **Weak overlap within groups**: effects for groups that are almost always (or never) treated are extrapolations.
- **Policy valued on training data**: optimistic by construction; use the held-out value.
- **Small samples**: with a few hundred units, flexible CATE models are noise; prefer pre-specified interactions in a linear model.

## 10. Reporting checklist

- [ ] Estimand, X (modifiers) and W (confounders) stated before estimation, with the identifying assumption in words.
- [ ] Overlap reported (share of clipped propensity scores).
- [ ] ATE with CI; linear CATE coefficients with CIs.
- [ ] BLP test (both coefficients) and its verdict; GATES table with CIs and the top − bottom test.
- [ ] CLAN with adjusted p-values; TOC/AUTOC or R-score comparison of at least one linear and one flexible model.
- [ ] For a policy: rule as text, cost used, held-out value vs treating everyone, share treated.
- [ ] Robustness to seed and nuisance learners; limits (unconfoundedness, external validity of the rule).
