# Linear regression (OLS) — specification, diagnostics and fixes

## Contents
1. Assumptions and what each violation causes
2. Functional forms and interpretation
3. Dummy variables and interactions
4. Restriction tests (F, Wald, LR)
5. Recommended order of diagnostics
6. Heteroskedasticity: tests and fixes (HC, WLS, FGLS)
7. Autocorrelation: tests and fixes (HAC, GLS)
8. Multicollinearity
9. Influential observations
10. Measurement error
11. Model selection

## 1. Assumptions and consequences

| Assumption | If it fails | Consequence for OLS |
|---|---|---|
| Linear in parameters, correct functional form | Misspecification | Bias and inconsistency |
| E(ε\|X) = 0 (exogeneity) | Omitted variable, simultaneity, measurement error in X | Bias and inconsistency — and it contaminates **all** coefficients, not just the endogenous variable's |
| Full rank of X | Perfect multicollinearity | Cannot estimate |
| Homoskedasticity | Heteroskedasticity | Coefficients unbiased and consistent, but standard errors wrong and OLS no longer BLUE |
| No autocorrelation | Serial correlation | Same; standard errors usually understated |
| Normality | Non-normality | Only matters for exact inference in small samples |

Key point: **heteroskedasticity and autocorrelation are often symptoms of a wrong functional form or an omitted variable**. That is why diagnostics start with specification (RESET), not with fixing standard errors.

Exogeneity comes in two strengths: **strict** exogeneity, E(ε_t\|X) = 0 for all periods, guarantees unbiasedness; **weak/contemporaneous** exogeneity, E(x_t ε_t) = 0, guarantees only consistency. Models with a lagged y violate strict exogeneity, so OLS is biased in small samples but consistent if there is no residual autocorrelation.

## 2. Functional forms

| Model | Equation | Interpretation of β | Elasticity |
|---|---|---|---|
| Linear | y = α + βx | Δy per unit of x | β·x/y (evaluate at the mean) |
| Log-log | ln y = α + β ln x | Constant elasticity: %Δy per 1% in x | β |
| Log-lin (semi-log) | ln y = α + βx | 100·β ≈ %Δy per unit of x (exact: 100·(e^β − 1)) | β·x |
| Lin-log | y = α + β ln x | β/100 = Δy per 1% in x | β/y |
| Reciprocal | y = α + β(1/x) | Diminishing effect; asymptote at α | −β/(x·y) |
| Log-inverse | ln y = α − β/x | S-shaped curve | β/x |
| Quadratic | y = α + β₁x + β₂x² | Marginal effect β₁ + 2β₂x; turning point at −β₁/(2β₂) | (β₁ + 2β₂x)·x/y |

Cautions:
- R² of models in y and in ln y are not directly comparable; compare with information criteria computed on the same scale or with out-of-sample forecasts.
- When predicting y from a ln y model, exp(predicted ln y) understates E(y); correct with Duan's smearing factor (mean of exp(residuals)) or exp(σ²/2) under normality.
- Dummies in log models: exact percentage effect = 100·(exp(β) − 1).

## 3. Dummies and interactions

- **Dummy trap**: with an intercept, include g − 1 dummies for g categories. In formulas, `C(category)` does this automatically; choose the base category with `C(x, Treatment(reference="..."))`.
- **Intercept dummy**: shifts the level. **Slope dummy** (interaction `x:D`): changes the effect of x across groups.
- Coding categories as 1, 2, 3, 4 in a single variable imposes equally spaced effects — almost always wrong.
- Interaction between continuous variables: the effect of x₁ depends on x₂ (β₁ + β₁₂·x₂); report it at relevant values of x₂ (mean, quartiles).
- Seasonal dummies (`C(month)`) for monthly data and break/event dummies (pandemic, methodology change).
- Equivalent Chow test: model with all interactions with the group dummy; joint F on the interactions and the dummy.

## 4. Restriction tests

- **Partial F (restricted vs. unrestricted)**: `res.f_test("x1 = 0, x2 = 0")` or `res.compare_f_test(restricted)`.
- General linear hypotheses: `res.f_test("x1 = x2")`, `res.t_test("x1 = 0.25")`, `res.wald_test(...)`.
- With robust errors, `f_test`/`wald_test` automatically use the robust covariance (the classical SSR-based F is invalid under heteroskedasticity).
- Maximum-likelihood models (logit/probit/tobit): likelihood ratio `LR = 2(llf_unrestricted − llf_restricted) ~ χ²(g)`.

## 5. Recommended order of diagnostics

1. Plots: y vs. each x; residuals vs. fitted; residuals vs. order/time; squared residuals vs. each x.
2. **RESET** (functional form). If it rejects, respecify before anything else.
3. Heteroskedasticity (Breusch-Pagan; full White or the simplified version with ŷ and ŷ²).
4. Autocorrelation (Durbin-Watson only as an indicator; Breusch-Godfrey with p lags).
5. Normality (Jarque-Bera) — relevant only in small samples.
6. Multicollinearity (VIF, condition number) and influential observations.
7. Stability (Chow with a known date, CUSUM) when there is time or groups.
8. Exogeneity (theory + Hausman test when an instrument exists) → `panel-iv.md`.

`scripts/diagnostics.py` covers 2–6 in one call (`ols_diagnostics`), plus `influence` and `chow`.

## 6. Heteroskedasticity

Tests:
- **Breusch-Pagan**: regress ê² on the X's; LM = n·R² ~ χ²(k−1).
- **White**: regress ê² on the X's, their squares and cross-products; catches unknown forms but uses many degrees of freedom. Economical version: ê² = δ₀ + δ₁ŷ + δ₂ŷ² + v, H0: δ₁ = δ₂ = 0 (`white_simplified` in the script).

Fixes, from most to least common:
1. **Robust (HC) standard errors** — the default. Hansen (2020) recommends **HC2 or HC3** over HC1 (Stata's default): HC2 is unbiased under homoskedasticity and HC3 is conservative. They only differ much when there are high-leverage observations (h_ii near 1). Use `cov_type="HC3"`.
2. **WLS** when the form is known, Var(ε\|x) = σ²h(x): divide everything by √h(x). `sm.WLS(y, X, weights=1/h)`. Interpret coefficients in the original model.
3. **FGLS/FWLS** with an unknown form (Wooldridge): estimate OLS; regress ln(ê²) on the X's; ĥ = exp(ĝ); estimate WLS with weights 1/ĥ. Also report robust errors in the WLS, since ĥ may be misspecified.
4. Transforming the variable (logs) often removes heteroskedasticity and improves the functional form at the same time.

R² is not affected by heteroskedasticity.

## 7. Autocorrelation

- In a cross-section, autocorrelated residuals (ordered by some variable) signal a **wrong functional form or an omitted variable**.
- Tests: Durbin-Watson (≈2 without autocorrelation; invalid with a lagged y), **Breusch-Godfrey** (LM = n·R² ~ χ²(p); valid with a lagged y), Ljung-Box on residuals.
- Fixes: **HAC/Newey-West** (`cov_type="HAC", cov_kwds={"maxlags": L}`, L ≈ 4(T/100)^(2/9)); model the dynamics (add lags — ADL); GLS with AR(p) errors (`sm.GLSAR(y, X, rho=1).iterative_fit()`), equivalent to Cochrane-Orcutt/Prais-Winsten.
- Prefer modeling the dynamics over "fixing" the error: residual autocorrelation almost always hides omitted lags.

## 8. Multicollinearity

- Biases nothing; inflates variances. Classic symptom: high R² and significant F with weak individual t's, "wrong" signs, coefficients that are unstable when observations are dropped.
- VIF_j = 1/(1 − R²_j): > 5 watch, > 10 worrying. A condition number of X above 30 suggests a problem.
- Remedies: more data; combine variables (index, ratio); drop one of the near-redundant variables (beware of omitted-variable bias); theoretical restrictions; accept it when the collinearity is only among controls and does not affect the coefficient of interest.

## 9. Influential observations

- **Leverage** h_ii: X far from its mean. High leverage gives *potential* influence.
- Actual influence: change in the coefficient when the observation is dropped (**DFBETA** = change / standard error) and in the fitted value, |h_ii·ẽ_i| (Hansen's "Influence" statistic = max |h_ii·ẽ_i|).
- Cook's distance and DFFITS exist, but Hansen warns they lack a firm statistical basis; focus on "does the observation materially change the coefficient of interest?".
- What to do: check for typing/recording errors first; if it is an error, correct or drop it **documented in the script** (keep the raw data); if it is legitimate, report results with and without it. Do not drop it just because it is inconvenient.
- `scripts/diagnostics.py influence` lists the most influential observations.

## 10. Measurement error

- In **y**: coefficients unbiased, just less precise.
- In **x** (classical): **attenuation bias** — plim β̂ = β·σ²_x*/(σ²_x* + σ²_error), always toward zero. Fix: instrumental variables (another measure of the same variable is a good instrument).

## 11. Model selection

- Omitting a relevant variable → bias; including an irrelevant one → inefficiency. When in doubt about a theoretically relevant control, keep it.
- General-to-specific: start with the broadest justifiable model and simplify with joint F tests, checking diagnostics at each step.
- Adjusted R², AIC, BIC (BIC penalizes more and picks smaller models; preferable for identifying the model; AIC tends to be better for forecasting).
- Never use isolated significance as the only inclusion criterion; report how many specifications were tried.
- **Bad controls** (Angrist-Pischke): do not control for variables that are themselves outcomes of the treatment (e.g., when measuring a promotion's effect on revenue, do not control for the number of customers, which the promotion affects).
