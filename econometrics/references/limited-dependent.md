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

y* = xβ + ε, ε ~ N(0, σ²); y = max(0, y*).

- OLS on y > 0 only, or on all data, is biased and inconsistent.
- statsmodels has no built-in Tobit: `scripts/diagnostics.py` has `tobit(y, X, lower_limit=0)` (ML with the censored likelihood).
- Three marginal effects; pick by the question:
  - on the latent variable: β_j;
  - on E(y\|x) (unconditional): Φ(xβ/σ)·β_j — the most used in business;
  - on E(y\|y>0, x): β_j·[1 − λ(c)(c + λ(c))], with c = xβ/σ and λ(c) = φ(c)/Φ(c) (inverse Mills ratio).
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

## 8. Pitfalls

- Interpreting a logit/probit coefficient as a change in probability.
- Comparing coefficients between logit and probit or across different samples (the scale σ changes).
- Using OLS R² or accuracy at a 0.5 cutoff as the main metric on an imbalanced sample (1% defaulters → 99% accuracy by always predicting "no").
- Forgetting that in nonlinear models interaction effects are not the interaction coefficient; compute the double difference in probabilities.
- Endogeneity in a binary model: 2SLS on the LPM is the simplest and most transparent fix; probit with a control function (include the first-stage residual) is the nonlinear alternative.
