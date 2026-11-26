# Estimation engine: maximum likelihood, NLS, delta method, bootstraps and GMM

Use when a model is not packaged (a custom count or duration model, a structural demand equation, a likelihood from a paper), when the quantity of interest is a function of parameters (elasticity at a point, long-run multiplier, willingness to pay), or when standard errors are doubtful (few clusters, small samples, unusual statistics). Code: `scripts/estimation.py`. The theory is in Greene (chs. 7, 14), Cameron & Trivedi (chs. 5, 7, 11), Hansen (chs. 10, 22) and Sargent & Stachurski, *Intermediate Quantitative Economics with Python* (lecture "Maximum Likelihood Estimation").

## 1. Maximum likelihood with `mle`

Write the log-likelihood **per observation** (an array of n contributions), not its sum: the contributions are what robust covariances are built from.

```python
import numpy as np
from scipy.special import gammaln
import estimation as est

def ll_poisson(b, X, y):                       # one value per observation
    xb = X @ b
    return y * xb - np.exp(xb) - gammaln(y + 1)

fit = est.mle(ll_poisson, np.zeros(X.shape[1]), args=(X, y), names=names, cov_type="sandwich")
fit["table"]; fit["converged"]; fit["grad_max"]; fit["aic"]
```

- **Optimizer**: `method="newton"` (Newton-Raphson with step halving: fast and exact near the optimum; falls back to BFGS if the Hessian is not negative definite) or `"bfgs"`. Try several starting values when the likelihood may have local maxima (mixtures, switching models).
- **Covariance** (`cov_type`):

| Option | Formula | Valid when |
|---|---|---|
| `"hessian"` | (−H)⁻¹ | The density is correctly specified |
| `"opg"` | (G'G)⁻¹ (BHHH) | Same; worse in small samples |
| `"sandwich"` | H⁻¹ (G'G) H⁻¹ | The density is a working approximation but the moment it targets is right (QMLE: Poisson for overdispersed counts, normal for non-normal errors) |
| `"cluster"` | H⁻¹ (Σ_g s_g s_g') H⁻¹ · G/(G−1) | Dependence within clusters |

  A large gap between Hessian and sandwich SEs is itself a misspecification signal (White's information matrix idea).
- **Analytic scores** (`score=`) are faster and more precise than numerical ones; numerical derivatives are central differences with a step scaled to each parameter.
- **Checks to report**: convergence, max |gradient| at the solution, AIC/BIC when comparing non-nested models, and the sandwich SEs when in doubt.

## 2. The trinity: LR, Wald, LM (`trinity_test`)

Test linear restrictions R·theta = r:
- **LR** = 2(ℓ_u − ℓ_r): needs both models; invariant to reparametrization; usually the most reliable.
- **Wald** = (Rθ̂ − r)'[R V R']⁻¹(Rθ̂ − r): only the unrestricted model; not invariant (testing b = 1 and 1/b = 1 can disagree).
- **LM (score)** = s(θ̃)'(−H̃)⁻¹ s(θ̃): only the restricted model; useful when the unrestricted one is hard to fit.

All are asymptotically chi²(q). Close values: the asymptotics are working. Far apart: small sample or awkward likelihood; trust LR, or bootstrap. The restricted model is fit by reparametrizing theta = theta_p + N·phi, with N a basis of the null space of R.

## 3. Nonlinear least squares (`nls`)

For y = f(x, theta) + u with a known f (CES production, logistic diffusion curves, exponential decay, Box-Cox regressions). SEs from the Jacobian J: classical s²(J'J)⁻¹ or HC1 (default). With a linear f the function reproduces OLS exactly. Warnings: a Jacobian condition number above 1e8 means weak identification or bad scaling; results that change with the starting values mean the data do not pin down the parameters.

## 4. Delta method (`delta_method`)

Var(g(θ̂)) ≈ J V J', J = ∂g/∂θ'. Typical uses:

| Quantity | g(theta) |
|---|---|
| Percentage effect in a log-lin model | exp(b) − 1 |
| Long-run multiplier in an ADL | Σb / (1 − Σφ) (also `specification.long_run_effect`) |
| Turning point of a quadratic | −b1 / (2 b2) |
| Willingness to pay, marginal rate of substitution | b_attribute / (−b_price) |
| Elasticity at the means | b · x̄ / ȳ |

The normal approximation is poor for ratios with an imprecise denominator (|b/se| below ~3): bootstrap instead, or report a Fieller interval.

## 5. Bootstrap

- `bootstrap(stat_fn, df, B, cluster=None)`: pairs bootstrap (rows) or cluster bootstrap (whole clusters). Returns SE, percentile and normal CIs and the bias. Resample at the level of independence. Not for time series (use blocks), not for statistics on the boundary of the parameter space, and unreliable with fewer than ~30 clusters.
- `wild_cluster_bootstrap(df, formula, coef, cluster)`: restricted wild cluster bootstrap-t (WCR) for one OLS coefficient. With few clusters the usual CR1 SEs over-reject badly; WCR keeps the size close to nominal with as few as 6–10 clusters. It flips each cluster's restricted residuals by a random weight (Rademacher ±1, or Webb's six-point weights when G < 12, chosen automatically), recomputes the clustered t, and inverts the test for a CI. In the skill's simulations with 8 clusters, CR1 with normal critical values rejected a true null about 20–25% of the time; WCR about 7% (1,000 replications), a small residual over-rejection that grows as clusters get fewer or more unequal. Report the number of clusters and both p-values.

  **Few treated clusters.** For a treatment assigned at the cluster level (a law adopted by some states), the binding number is the treated clusters, not all clusters. With roughly 1–5 treated clusters the wild bootstrap itself can under- or over-reject (MacKinnon & Webb 2017, 2018). Add randomization inference (`causal.randomization_inference`): re-estimate with the treatment reassigned to every (or many random) sets of the same number of clusters and report the rank of the actual estimate; with unequal variances across states, Conley & Taber (2011) or Ferman & Pinto (2019).

Equivalents: Stata `boottest`, R `fwildclusterboot`; MacKinnon, Nielsen & Webb (2023), "Cluster-robust inference: a guide to empirical practice", *Journal of Econometrics*.

## 6. Numerical good practice

- Scale variables so that coefficients are of similar magnitude (`reporting.rescale_hint`): badly scaled problems give inaccurate Hessians.
- Solve linear systems instead of inverting (X'X); for OLS use QR or `lstsq` (statsmodels already does).
- Check identification: a singular or nearly singular information matrix means some parameters are not identified by the data.
- Compare with a packaged estimator when one exists (statsmodels `GenericLikelihoodModel`, `Poisson`, `Probit`): the test suite does this for Wooldridge Examples 17.1 and 17.3.

## 7. Generalized method of moments (`gmm`, v2.8.0)

For a model that delivers moment conditions E[g_i(θ)] = 0 rather than a likelihood: an Euler equation E_t[β (C_t+1/C_t)^(−γ) R_t+1 − 1] = 0 times instruments known at t (Hansen and Singleton 1982), a structural demand equation with instruments, any estimating equation from a paper. Write the contributions, one row per observation:

```python
import numpy as np
import estimation as est

def euler(th, R, cg, Z):                       # n x m: pricing error times each instrument
    gamma, beta = th
    return (beta * cg ** (-gamma) * R - 1.0)[:, None] * Z

fit = est.gmm(euler, [1.0, 0.99], args=(R, cg, Z), names=["gamma", "beta"])   # two-step by default
fit["table"]; fit["j"], fit["j_df"], fit["j_p"]; fit["rank_p"]; fit["warnings"]
```

- **Steps.** `"two-step"` (default): identity (or `weight=`) first, then W = S⁻¹ at the first-step estimate; `"iterated"`: until the estimate stops moving; `"one-step"`: the given weight with sandwich SEs and no J. The first step depends on the units of the moments: scale them, or pass `weight=np.linalg.inv(Z.T @ Z / n)` (the 2SLS weight; statsmodels' NonlinearIVGMM default).
- **S.** Without `hac_lags`, the covariance of the contributions (independent observations, any heteroskedasticity); with `hac_lags=L`, Newey-West (Bartlett) or `kernel="truncated"` (Hansen-Singleton's known MA order; can be indefinite, then floored and flagged). An n-period Euler equation has MA(n − 1) errors: `hac_lags ≥ n − 1`. `center=True` (default) demeans the contributions; statsmodels and Hansen-Singleton do not (`center=False` reproduces them).
- **J** = n ḡ'Wḡ with the last weight, χ²(m − k): **the p-value is the right tail** (`j_p`). Old tables' "Prob" is the CDF (`j_cdf`): Hansen-Singleton's Prob of 0.88 is p = 0.12. With m = k there is no test.
- **Identification.** `rank_stat`, `rank_df` = m − k + 1 and `rank_p` are the Cragg-Donald (1997) minimum-Wald test of rank(D) < k on the Jacobian D = E[∂g/∂θ'] (with `hac_lags`, always on a Bartlett long-run covariance). Not rejecting means the parameters may not be identified: do not report them. **Rejecting does not mean strong identification**: the statistic grows with n under any alternative, so it is not a first-stage F and has no cut of 10 (a linear IV with a first-stage F of 1e8 gives about 40 with n = 100). Judge weakness by the precision you get and by the criterion: plot it over a grid of the parameters, fix one from theory, or use an identification-robust test (Stock and Wright 2000). On Hansen and Singleton's 239 months (the lecture's reconstruction) the rank test rejects with 2 to 6 instrument lags, yet the 95% interval of γ contains 0: that imprecision is what the lecture calls weak identification. **Size with persistent series:** the test's variance is iid without `hac_lags` and Bartlett with them, so persistent derivatives make it over-reject. Measured with an irrelevant instrument, AR(ρ) instruments and regressor, n = 240 (1000 to 2000 samples per cell): at 5%, 0.04–0.07 with iid data, 0.20–0.21 at ρ = 0.5 and 0.85–0.86 at ρ = 0.9 without lags, 0.14–0.15 at ρ = 0.7 and 0.37–0.40 at ρ = 0.9 with 4 lags, 0.16–0.18 at ρ = 0.9 with 12 lags (n = 1000 alike); with n = 240 and 12 lags it is 0.09–0.11 even with little persistence (too many lags for n). `estimation.gmm` warns when the largest lag-1 autocorrelation of the derivatives (`used["rank_test_ac1"]`) is at least 0.4 without lags or 0.6 with them and the statistic, divided by the AR(1) long-run variance factor the kernel misses, no longer rejects: then the rejection is not evidence of identification. In those cells a rejection without the warning happened in at most 0.004 of the samples without lags, 0.010 with 4 lags and 0.036 with 12. Hansen and Singleton's derivatives have a lag-1 autocorrelation of about 0.5: without lags the rejection gets the warning; with 4 Bartlett lags it still rejects (p < 1e-4) without it.
- **Bounds.** `bounds=[None, (0, 1)]` keeps β ≤ 1; an estimate at or within one SE of a bound gets a warning: no normal interval there (Andrews 1999).
- **Checks.** `tests/test_estimation.py`: linear two-step GMM against its closed form, exactly identified = IV, HAC against the double sum, Euler equation against statsmodels' NonlinearIVGMM, and a Monte Carlo with γ = 2, β = 0.995 (400 samples of 900 months: 95% CI coverage 0.945 for γ and 0.938 for β, J size 0.035); `tests/test_quantecon_replication.py`: Hansen-Singleton (1982) on the lecture's data. Stata: `gmm`; R: `gmm::gmm`, `momentfit`.

## 8. Likelihood ratio between nested fits (`lr_test`, v2.8.0)

`trinity_test` handles linear restrictions Rθ = r. For restrictions of any form, fit both models with `mle` on the same observations and pass the two results:

```python
fu = est.mle(ll_var, theta_u0, args=(data,))        # unrestricted (e.g. a VAR)
fr = est.mle(ll_model, theta_r0, args=(data,))      # the theory's restricted version
est.lr_test(fu, fr)                                 # LR = 2(l_u - l_r), df = k_u - k_r, right-tail p
```

Hansen and Singleton (1983) test the cross-equation restrictions of a lognormal asset-pricing model against an unrestricted VAR this way (df = the number of restrictions the theory imposes); they print the χ² CDF in parentheses, `chi2_cdf` here. Both fits must use the same rows (a VAR with more lags loses initial rows: trim them first); `lr_test` refuses different n, but cannot see whether two fits with the same n used the same rows. A null on the boundary (a variance of 0): `boundary=True` (50:50 mixture with one restriction, Self and Liang 1987).

