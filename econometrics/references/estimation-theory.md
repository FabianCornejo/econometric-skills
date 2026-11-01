# Estimation engine: maximum likelihood, NLS, delta method and bootstraps

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
- `wild_cluster_bootstrap(df, formula, coef, cluster)`: restricted wild cluster bootstrap-t (WCR) for one OLS coefficient. With few clusters the usual CR1 SEs over-reject badly; WCR keeps the size close to nominal with as few as 6–10 clusters. It flips each cluster's restricted residuals by a random weight (Rademacher ±1, or Webb's six-point weights when G < 12, chosen automatically), recomputes the clustered t, and inverts the test for a CI. In the skill's simulations with 8 clusters, CR1 with normal critical values rejected a true null about 20% of the time; WCR about 5–7%. Report the number of clusters and both p-values.

Equivalents: Stata `boottest`, R `fwildclusterboot`; MacKinnon, Nielsen & Webb (2023), "Cluster-robust inference: a guide to empirical practice", *Journal of Econometrics*.

## 6. Numerical good practice

- Scale variables so that coefficients are of similar magnitude (`reporting.rescale_hint`): badly scaled problems give inaccurate Hessians.
- Solve linear systems instead of inverting (X'X); for OLS use QR or `lstsq` (statsmodels already does).
- Check identification: a singular or nearly singular information matrix means some parameters are not identified by the data.
- Compare with a packaged estimator when one exists (statsmodels `GenericLikelihoodModel`, `Poisson`, `Probit`): the test suite does this for Wooldridge Examples 17.1 and 17.3.
