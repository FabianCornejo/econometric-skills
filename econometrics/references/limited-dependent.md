# Qualitative and limited dependent variables

Use when y is binary (bought/did not, default/no default), a count (number of orders), censored (spending that is zero for many), fractional (a share between 0 and 1), or ordered/multinomial.

## Contents
1. Linear probability model (LPM)
2. Logit and probit
3. Marginal effects and odds ratios
4. Tests and goodness of fit
5. Tobit (censoring)
6. Sample selection (Heckman) and two-part models
7. Count, fractional, ordered and multinomial outcomes
8. Pitfalls
9. Heterogeneous segments: the logit leaf model

## 1. Linear probability model

Binary y estimated by OLS: P(y=1\|x) = xβ, so β_j = change in probability.
- Advantages: direct interpretation; good for average effects near the mean of the X's; a base for IV with an endogenous regressor.
- Problems: predictions outside [0,1]; heteroskedastic by construction, Var(ε\|x) = p(x)(1 − p(x)) → **always use robust errors (HC3)**; constant effects, unrealistic at the extremes.
- Use as a robustness check alongside logit/probit: signs and significance should agree.

## 2. Logit and probit

Latent-variable model: y* = xβ + ε, y = 1 if y* > 0. Probit: normal ε; logit: logistic ε. Estimated by maximum likelihood.

```python
import statsmodels.formula.api as smf
logit  = smf.logit("default ~ income + age + C(region)", data=df).fit(disp=False)
probit = smf.probit("default ~ income + age + C(region)", data=df).fit(disp=False)
# robust (QML) errors: .fit(cov_type="HC1") ; clustered: .fit(cov_type="cluster", cov_kwds={"groups": df["store"]})
```

- Logit and probit give nearly identical probabilities except in the tails; coefficients differ by scale (β_logit ≈ 1.6·β_probit). Compare marginal effects, never raw coefficients.
- Logit: ln[p/(1−p)] = xβ (linear log-odds), which gives the odds-ratio interpretation.
- Perfect separation (a variable predicts y exactly) makes ML diverge; statsmodels warns. Merge categories or use regularization/Firth.

## 3. Marginal effects and odds ratios

The coefficients are **not** effects on the probability. The effect of x_j is f(xβ)·β_j, which depends on x.

```python
me = logit.get_margeff(at="overall", method="dydx")   # average marginal effect (AME) — preferred
me_mean = logit.get_margeff(at="mean")                 # effect at the mean of X (MEM)
me.summary()
# Dummies: dummy=True computes the discrete change P(y=1|d=1) − P(y=1|d=0) instead of the derivative
logit.get_margeff(at="overall", dummy=True)
# Counts: count=True
```

- The **AME** (average of individual effects) is more robust than the effect at the mean (MEM), especially with dummies — the "average individual" may not exist.
- For a specific profile (like Stata's `margins, at()`): compute `logit.predict(profile)` and `profile+Δ` and take the difference; for intervals, bootstrap or the delta method.
- **Odds ratio (logit)**: exp(β_j) = multiplier of the odds when x_j increases by 1 unit. E.g., exp(0.11) = 1.116 → odds increase by 11.6% per unit. Do not confuse odds with probability.

## 4. Tests and goodness of fit

- Wald: `res.wald_test("x1 = 0, x2 = 0")`; linear hypotheses such as `"x1 = x2"`.
- Likelihood ratio (restricted vs. unrestricted): `LR = 2*(unr.llf - restr.llf)`, χ² with g degrees of freedom. statsmodels reports `llr` and `llr_pvalue` against the constant-only model.
- McFadden pseudo-R²: 1 − llf/llnull (`res.prsquared`). Values of 0.2 to 0.4 already indicate a good fit; do not compare with OLS R².
- Classification table: `res.pred_table(threshold=0.5)`. The 0.5 cutoff is arbitrary; with imbalanced classes, choose the cutoff by the cost of errors (false positive vs. false negative) or the sample proportion.
- ROC curve and AUC (`sklearn.metrics.roc_auc_score`) measure discrimination regardless of the cutoff.
- Calibration: group by predicted-probability deciles and compare with observed frequencies (Hosmer-Lemeshow test).
- For prediction, evaluate **out of sample** (cross-validation).

## 5. Tobit (censoring)

Use when y is continuous and positive but piles up at zero (or at a ceiling): hours worked, spending on an item, volume purchased.

y* = xβ + ε, ε ~ N(0, σ²); y = max(0, y*). With a ceiling (top-coded income, capacity limits, scores capped at 100) or both: y = min(max(a, y*), b).

- OLS on y > 0 only, or on all data, is biased and inconsistent.
- statsmodels has no built-in Tobit: `scripts/diagnostics.py` has `tobit(y, X, lower_limit=0, upper_limit=None)` (ML with the censored likelihood). `lower_limit=None, upper_limit=c` handles top-coding; give both for two limits. It reproduces Wooldridge Example 17.2 exactly (`tests/test_textbook_replication.py`).
- Effects — pick by the question; `res["marginal_effects"]` has the three averages with delta-method SEs:
  - on the latent variable: β_j (only meaningful if y* itself is of interest, e.g. desired hours);
  - on E(y\|x) (unconditional, column `E[y|x]`): Φ(xβ/σ)·β_j — the total effect on observed y, the most used in business;
  - on E(y\|y>0, x) (intensive margin, `E[y|uncensored,x]`): β_j·[1 − λ(c)(c + λ(c))], with c = xβ/σ and λ(c) = φ(c)/Φ(c) (inverse Mills ratio) — "how much more, among those who already buy";
  - on P(y>0\|x) (extensive margin, `P(uncensored|x)`): φ(c)·β_j/σ — "how much more likely to buy at all".
  - McDonald-Moffitt: for each observation, the unconditional effect = P(y>0\|x)·(intensive effect) + E(y\|y>0,x)·(extensive effect). It holds observation by observation, so decompose before averaging (the product of the averages does not add up). Reporting both margins shows *where* the effect comes from.
  - Effects are averaged over the sample (AME); for a dummy regressor they approximate the discrete change.
- Tobit assumes **the same process** decides whether to buy and how much. If they are distinct decisions (very common), use a two-part model (section 6). It is also sensitive to heteroskedasticity and non-normality.

## 6. Sample selection (Heckman) and two-part models

- **Heckman (two-step)**: when y is observed only for a non-randomly selected sample (wages only for those who work). Step 1: probit of selection with all variables, including at least one that affects selection but not the outcome (exclusion restriction — without it, identification relies only on nonlinearity, which is fragile). Step 2: OLS of y on the subsample including λ̂ = φ(ẑγ)/Φ(ẑγ). A significant λ̂ coefficient indicates selection bias. Step-2 standard errors need correction (bootstrap the whole procedure).
- **Two-part (hurdle) model**: part 1 = logit/probit of buying or not (extensive margin); part 2 = OLS/GLM of the amount among buyers (intensive margin). E(y\|x) = P(y>0\|x)·E(y\|y>0, x). Lets price affect the purchase decision and the quantity differently.
- **Zeros and logs**: ln(y+1) is not a neutral fix (results depend on the scale). Prefer PPML — Poisson pseudo-maximum likelihood, `smf.glm("y ~ x", data, family=sm.families.Poisson()).fit(cov_type="HC1")` — which accepts zeros and gives coefficients interpretable as semi-elasticities.

## 7. Other kinds of y

| y | Model | statsmodels |
|---|---|---|
| Count (0, 1, 2…) | Poisson; negative binomial if overdispersed (Var > mean) | `smf.poisson`, `smf.negativebinomial`; or GLM |
| Excess zeros | Zero-inflated Poisson/NB | `statsmodels.discrete.count_model.ZeroInflatedPoisson` |
| Fraction/share (0–1) | Papke-Wooldridge fractional response | `smf.glm(..., family=sm.families.Binomial()).fit(cov_type="HC1")` |
| Ordered (rating 1–5) | Ordered probit/logit | `statsmodels.miscmodels.ordinal_model.OrderedModel` |
| Unordered categories | Multinomial logit | `smf.mnlogit` (beware the independence of irrelevant alternatives assumption) |
| Time to event | Survival (Kaplan-Meier, Cox, discrete-time hazard) | `scripts/survival.py`; see `survival.md` |

**Counts — check the variance after a Poisson fit**: `diagnostics.count_diagnostics(poisson_res)` reports the Pearson dispersion σ̂² (1 under Poisson; Wooldridge Ex. 17.3 gives 1.52), Cameron-Trivedi tests for NB2 (Var = μ + αμ²) and NB1 (Var = (1+α)μ), and observed vs. Poisson-implied zeros. With overdispersion the Poisson *coefficients* stay consistent if the mean is right (quasi-MLE): keep them and fix the SEs — robust (`cov_type="HC1"`) or quasi-Poisson (× σ̂, returned as `quasi_poisson_se`). Move to negative binomial when predicted count probabilities matter; to hurdle/zero-inflated models when zeros come from a separate decision.

## 8. Pitfalls

- Interpreting a logit/probit coefficient as a change in probability.
- Comparing coefficients between logit and probit or across different samples (the scale σ changes).
- Using OLS R² or accuracy at a 0.5 cutoff as the main metric on an imbalanced sample (1% defaulters → 99% accuracy by always predicting "no").
- Forgetting that in nonlinear models interaction effects are not the interaction coefficient; compute the double difference in probabilities.
- Endogeneity in a binary model: 2SLS on the LPM is the simplest and most transparent fix; probit with a control function (include the first-stage residual) is the nonlinear alternative.

## 9. Heterogeneous segments: the logit leaf model

When the effect of x on P(y=1) changes across groups that nobody wrote down in advance (churn drivers differ between new and old customers, default drivers between secured and unsecured loans), a pooled logit averages opposite effects away and a decision tree approximates each linear effect with many steps. The **logit leaf model** (De Caigny, Coussement & De Bock, 2018) combines them: a shallow tree defines a few segments, then a separate logit is estimated in each leaf.

```python
from segments import logit_leaf_model, predict_leaf_model
res = logit_leaf_model(df, "churn", ["tenure", "usage", "complaints", "price_increase"],
                       max_depth=2, min_leaf=100, select="forward_aic", honest=True)
res["leaves"]        # rule, n, positive rate per segment
res["ame"]           # average marginal effects by segment
res["performance"]   # holdout AUC and top-decile lift vs pooled logit and tree
p = predict_leaf_model(res["model"], new_customers)
```

When the user names the groups whose drivers may differ ("between customer types"), pass them and the theory model too, then check stability:

```python
res = logit_leaf_model(df, "churn", x, max_depth=2, min_leaf=100, segment_by=["b2b"],
                       benchmarks={"interactions": "churn ~ tenure + usage*b2b + complaints + price_increase*b2b"})
res["best"], res["warnings"], res["split_variables"]
stab = leaf_model_stability(df, "churn", x, seeds=range(10), max_depth=2, min_leaf=100,
                            benchmarks={"interactions": "churn ~ tenure + usage*b2b + complaints + price_increase*b2b"})
stab["split_frequency"]; stab["summary"]      # share of seeds each variable splits; mean AUC/lift, share best
```

- **Categorical groups**: `segment_by=["customer_type"]` accepts a text or categorical column; it becomes dummies (first level = base) added to x, a split on any dummy counts, and `predict_leaf_model` rebuilds them on new data (an unseen category is scored as the base, with a warning). In a benchmark formula use `C(customer_type)`.
- **Like-for-like benchmarks**: the benchmark formula must contain every regressor the leaf model uses, plus the interactions; a formula that drops one is flagged in `res["warnings"]`.
- **Temporal holdout**: when the model will score future customers, pass `time_col="month"` (any sortable column). The holdout becomes the most recent `test_size` share, cut at a distinct value so no period is split. A random holdout overstates performance whenever behaviour drifts. The honest split inside training remains random; with `time_col`, `leaf_model_stability` keeps the same holdout and varies only the segmentation.
- **Calibration**: `performance` reports the Brier score, the calibration slope (logit of y on logit p: 1 = calibrated, < 1 = predictions too extreme, typical of overfitting) and the mean bias (mean p − mean y), and `res["calibration"]` has a table by decile per model. For retention budgets, offers or provisioning, a well-ranked but badly calibrated score misstates how many customers will leave; recalibrate (Platt or isotonic on a recent window) before using the probabilities as levels. Under drift, a negative mean bias on the recent holdout is the usual sign that the base rate has risen.

- **When it earns its place**: prediction or targeting with readable output (a few rules plus one coefficient table per segment). The original study found it beats its building blocks and matches random forests on churn data, judged by AUC and top-decile lift. If it does not beat the pooled logit on the holdout, report the pooled logit.
- **Settings**: depth 2–3 and at least ~100 observations per leaf (the R package default) so each leaf supports a logit; forward selection by AIC keeps each leaf model short. sklearn's CART replaces the C4.5 tree of the original; use `ccp_alpha` to prune.
- **Inference**: the segments are chosen by the data. With `honest=True` the tree and the leaf logits use different halves of the training sample, so the leaf standard errors are valid *given* the segments (Athey & Imbens, 2016). With `honest=False`, the leaf p-values are too optimistic; use them only descriptively. In both cases, a segment is a hypothesis about heterogeneity: refit with other seeds and samples and report only segments that reappear.
- **The tree splits on purity, not on slopes.** CART cuts where the outcome *rate* differs. If the groups differ in *how* x affects y but not much in the average rate (usage lowers churn for consumers and raises it for business accounts), the tree may never split on that variable and the leaf model gains little over the pooled logit. In the test suite's simulation of exactly this case the tree split on tenure and price in every seed, never on the customer type; the logit with theory interactions reached AUC 0.80 against 0.78 (leaf) and 0.76 (pooled). Hence `segment_by` (warns when the named groups are not used), `benchmarks` (the interaction model on the same holdout) and `res["best"]`.
- **Which model to report**: the one with the best holdout AUC and top-decile lift, averaged across seeds (`leaf_model_stability`). Report segments only if the same split variables reappear in most seeds.
- **Not causal**: segment-specific coefficients are associations, like any logit. For heterogeneous *treatment* effects use causal designs with interactions specified in advance or causal forests (`causal-inference.md`).
- **Alternatives**: interactions written from theory (always preferable when theory exists); model-based recursive partitioning (`partykit::glmtree` in R; Zeileis, Hothorn & Hornik, 2008), which chooses splits by testing parameter instability in the logit itself rather than by node purity, and is the more principled choice when the goal is to find where coefficients change.
