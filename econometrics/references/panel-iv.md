# Endogeneity, instrumental variables and panel data

## Contents
1. Sources of endogeneity
2. Instrumental variables and 2SLS
3. Tests: relevance, endogeneity, overidentification
4. Weak instruments
5. Simultaneous equations and identification
6. Panel structure and preparation
7. Panel estimators (pooled, FE, RE, FD)
8. Panel standard errors and number of clusters
9. Hausman, Mundlak and Breusch-Pagan LM
10. Difference-in-differences
11. Dynamic panels
12. Pitfalls

## 1. Sources of endogeneity

E(ε\|X) ≠ 0 because of: (i) an omitted relevant variable correlated with a regressor; (ii) a wrong functional form; (iii) measurement error in X (attenuation bias); (iv) simultaneity — y and x determined jointly (price and quantity; income and education; sales and advertising). OLS becomes biased **and inconsistent**, and one endogenous regressor contaminates the other coefficients.

Direction of bias: plim β̂ = β + Cov(x, ε)/Var(x). Thinking about the likely sign of the covariance is often enough to tell whether the true effect is larger or smaller than the estimate.

## 2. Instrumental variables and 2SLS

Instrument z: (a) **exogenous** — Cov(z, ε) = 0 (not testable with a single instrument; it is an economic argument — "more art than science"); (b) **relevant** — Cov(z, x) ≠ 0 (testable in the first stage). Exogenous regressors are their own instruments; OLS is the special case where all X's are exogenous.

- Exactly identified (number of instruments = number of endogenous regressors): β̂_IV = (Z'X)⁻¹Z'y.
- Overidentified: **2SLS** — 1st stage: each endogenous variable on all instruments (reduced forms); 2nd stage: y on the fitted values. β̂ = (X'P_Z X)⁻¹X'P_Z y.
- Do not run the 2nd stage "by hand" with OLS for inference: the standard errors will be wrong. Use the ready-made estimator.

```python
from linearmodels.iv import IV2SLS, IVLIML, IVGMM
f = "log_q ~ 1 + log_income + C(month) + [log_price ~ log_input_cost + log_freight]"
iv = IV2SLS.from_formula(f, data=df).fit(cov_type="robust")      # or "clustered", clusters=df.store
print(iv.summary)
print(iv.first_stage)        # partial F, partial R² of excluded instruments
iv.wu_hausman(); iv.durbin()  # endogeneity (H0: regressor exogenous)
iv.sargan; iv.basmann        # overidentification (homoskedasticity)
gmm = IVGMM.from_formula(f, data=df).fit(); gmm.j_stat           # Hansen J (robust)
liml = IVLIML.from_formula(f, data=df).fit(cov_type="robust")    # preferable with weak instruments
# debiased=True: n−k small-sample correction and t/F instead of z/χ² (= Stata's `small` option and
# the textbook formulas). The default divides by n (= Stata's default). Use it when replicating a
# published table; with large N the difference is negligible.
```

Costs of IV: less efficient than OLS (larger variance); biased in finite samples toward OLS; its small-sample distribution can differ a lot from the asymptotic one.

## 3. Tests

**Relevance**: F of the excluded instruments in the 1st stage (robust, if applicable). Always report it.

**Endogeneity (Durbin-Wu-Hausman, regression version)**:
1. Regress the suspect endogenous variable on all exogenous variables and instruments; keep the residual v̂.
2. Add v̂ to the structural equation estimated by OLS (with robust errors if needed).
3. If v̂'s coefficient is significant, the variable is endogenous → use IV; if not, OLS is consistent and more efficient.
`scripts/diagnostics.py` has `endogeneity_test` with this procedure.

**Overidentification** (only with more instruments than endogenous regressors; q = number of instruments − number of endogenous regressors):
- Regress the 2SLS residuals on all exogenous variables and instruments; nR² ~ χ²(q) (Wooldridge) or Sargan (n − k)R². Rejecting → at least one instrument is invalid.
- With heteroskedasticity, use Hansen's J (`IVGMM ... .j_stat`).
- Not rejecting does not prove validity: if all instruments are invalid in the same direction, the test cannot detect it.
- `scripts/diagnostics.py` has `overid_test`.

## 4. Weak instruments

- Staiger-Stock rule of thumb: first-stage F > 10. The stricter criterion of Lee et al. (2022) for a t-test with correct 5% size: F > 104.7. With F between 10 and 100, also report Anderson-Rubin intervals.
- Weak instruments: 2SLS biased toward OLS and intervals with wrong coverage; many weak instruments make the bias worse.
- **LIML** is approximately median-unbiased in overidentified models and hardly suffers from extra useless instruments; exactly identified 2SLS is also approximately median-unbiased (Angrist-Pischke). With several instruments, compare 2SLS and LIML: if they diverge, be suspicious.

## 5. Simultaneous equations and identification

- **Structural form** (behavior: demand, supply) vs. **reduced form** (endogenous variables as functions of exogenous ones only). The reduced form can be estimated by OLS; structural parameters are recoverable only if the equation is identified.
- **Order condition** (necessary): number of exogenous variables excluded from the equation ≥ number of included endogenous variables − 1. Equal → exactly identified (indirect least squares works); greater → overidentified (use 2SLS; ILS would give several solutions).
- **Rank condition** (necessary and sufficient): from the coefficients of the variables excluded from the equation but present in the others, at least one non-zero determinant of order (M−1) can be formed.
- Classic example: demand Q = a₁ + b₁P + c₁Income + u₁ and supply Q = a₂ + b₂P + c₂Rain + u₂ — income identifies supply, rain identifies demand. Retail applications in `demand-elasticities.md`.
- Recursive models (triangular B, uncorrelated errors) can be estimated equation by equation by OLS.

## 6. Panel structure

`linearmodels` requires a MultiIndex (entity, time) in **long** format (one row per entity-period). To convert from wide: `pd.wide_to_long` or `melt`.

```python
df = df.set_index(["store", "month"])
df.groupby(level=0).size().describe()     # balance (equivalent to xtdescribe)
# Overall, between and within variation (equivalent to xtsum):
x = df["price"]
between = x.groupby(level=0).mean().std()
within = (x - x.groupby(level=0).transform("mean")).std()
```

- Fixed effects use only **within**-unit variation; variables that barely change over time are poorly estimated and time-invariant ones (region, store format) are absorbed.
- An unbalanced panel is acceptable, but investigate why units drop out (closed stores = survivorship bias).

## 7. Estimators

| Estimator | Key assumption | When |
|---|---|---|
| Pooled OLS | Heterogeneity uncorrelated with X | Benchmark; requires clustered errors |
| Fixed effects (`EntityEffects`) | Time-invariant heterogeneity may correlate with X | Default for effects in store/firm data |
| Two-way fixed effects (+ `TimeEffects`) | Shocks common to all units each period | Almost always advisable (absorbs seasonality and macro) |
| Random effects | Heterogeneity uncorrelated with X | Random sample of units; estimates time-invariant variables |
| First differences | Like FE; better if the error is a random walk | Small T with a persistent error |
| Between | Unit means | Long-run/cross-sectional relationships |

```python
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS, FirstDifferenceOLS, BetweenOLS
fe = PanelOLS.from_formula("log_sales ~ 1 + log_price + promo + EntityEffects + TimeEffects", df)\
         .fit(cov_type="clustered", cluster_entity=True)
fe.f_pooled               # F: fixed effects jointly significant (FE vs pooled)
re = RandomEffects.from_formula("log_sales ~ 1 + log_price + promo", df).fit()
re.theta, re.variance_decomposition   # weight of the individual component
```

## 8. Standard errors and clusters

- Observations of the same unit are correlated: clustering by entity is the default (`cluster_entity=True`); common time shocks → two-way clustering (`cluster_time=True` too) or Driscoll-Kraay (`cov_type="kernel"`).
- **The number of clusters is the effective sample size** (Hansen): G = 20 is a very small sample.
- **Few clusters** (Angrist-Pischke: fewer than ~42): clustered errors understate uncertainty. Options: use t with G − k degrees of freedom; BRL/CR2 bias correction; wild cluster bootstrap; report the larger of the robust and conventional errors.
- **At what level to cluster** (Hansen): too fine → bias (errors too small); too aggregate → imprecise errors. There is no definitive rule; cluster at the level where shocks are shared and where treatment varies, checking there are still enough clusters.

## 9. Hausman, Mundlak and Breusch-Pagan LM

- **Pooled vs. RE**: Breusch-Pagan LM (H0: individual-effect variance = 0) — `scripts/diagnostics.py` has `panel_bp_lm`.
- **FE vs. pooled**: `fe.f_pooled`.
- **FE vs. RE (Hausman)**: H0 = RE consistent (and efficient). H = (b_FE − b_RE)'[V_FE − V_RE]⁻¹(b_FE − b_RE) ~ χ²(k), using common coefficients and **non-robust** covariances. `scripts/diagnostics.py hausman`. If V_FE − V_RE is not positive semidefinite, the statistic can be negative (Stata works around it with `sigmamore`); in that case use Mundlak.
- **Mundlak (robust, preferable with clustering)**: add the unit means of the X's to the RE model; if they are jointly significant (clustered Wald), use FE.
- In business data, FE almost always wins: location and management correlate with price and promotion.

## 10. Difference-in-differences

Effect of an intervention affecting one group and not another (store remodels, a new pricing policy, a tax change):

```python
df["treated_post"] = df["treated"] * df["post"]
did = PanelOLS.from_formula("y ~ treated_post + EntityEffects + TimeEffects", df)\
          .fit(cov_type="clustered", cluster_entity=True)
```

- **Parallel trends**: without treatment, the groups would have evolved alike. Check with an *event study* (dummies for time relative to treatment): pre-treatment coefficients ≈ 0. Run placebos (fake date, fake group).
- **Staggered adoption** (units treated at different dates) with heterogeneous effects: simple TWFE can be seriously biased (Goodman-Bacon). Use Callaway-Sant'Anna, Sun-Abraham or de Chaisemartin-D'Haultfœuille, or at least state the limitation.
- Few treated groups → few-clusters problem (section 8).

For staggered adoption, event studies with clean comparisons (Callaway-Sant'Anna), synthetic control and RD, see `causal-inference.md` and `scripts/causal.py` (`event_study`, `att_gt`, `synth`).

## 11. Dynamic panels

- A lagged y with fixed effects creates **Nickell bias** (order 1/T), relevant when T < ~20.
- Large T (≥ 30): ordinary FE acceptable, with a caveat.
- Small T, large N: Arellano-Bond (difference) or Blundell-Bond (system) GMM, package `pydynpd`. Equivalents to Stata's `xtabond`: strictly exogenous variables as their own instruments; **predetermined** (`pre()`) instrumented by lags from t−1; **endogenous** (`endogenous()`) from t−2. Use two-step with the Windmeijer correction (robust errors), report AR(2) in differences (should not reject) and Hansen J, and limit the number of instruments (< N).

## 12. Pitfalls

- Pooled OLS with unclustered errors → inflated significance.
- Interpreting an FE coefficient as a difference between units (it is a within-unit effect over time).
- *Bad controls*: controlling for variables affected by the treatment.
- Causal language with FE without discussing time-varying variables left out of the model.
- IV with a weak instrument or an implausible exclusion — worse than honest OLS with caveats about the direction of bias.
