# Dependence: rank correlation, tail dependence, copulas and mutual information

Use when the question is "how do these variables move together?", especially in the tails (joint crashes, contagion, credit defaults, insurance claims), or when a linear correlation may be hiding nonlinear dependence. Code: `scripts/dependence.py`.

## 1. Why Pearson correlation is not enough

- It measures **linear** association and depends on the marginal distributions: the same dependence structure can give different Pearson values after a monotone transformation.
- It can be zero under perfect dependence (y = x²).
- It says nothing about the **tails**: two assets with correlation 0.5 can have very different probabilities of crashing together.
- It is not robust: a few extreme days dominate it.

`dependence.rank_correlations(df)` shows Pearson, Spearman ρ and Kendall τ side by side. Rank correlations are invariant to monotone transformations, so they measure the dependence structure itself (the copula).

## 2. Copulas in one paragraph

Sklar's theorem: any joint distribution F(x, y) = C(F_X(x), F_Y(y)), where C is a copula (a joint distribution of uniforms). Fit the marginals and the dependence separately. In practice the marginals are replaced by **pseudo-observations** u = rank/(n+1) (semi-parametric, "canonical maximum likelihood"), and the copula parameters are estimated by maximum likelihood on (u, v).

Apply copulas to **i.i.d. data**. For daily returns with volatility clustering, fit the copula to GARCH standardized residuals (`res.std_resid` from `arch`), not to raw returns; otherwise volatility co-movement is mistaken for dependence (this is the copula-GARCH approach of Patton, 2006).

## 3. Families

| Family | Parameter | Kendall τ | Tail dependence λ_L / λ_U | Use when |
|---|---|---|---|---|
| Gaussian | ρ | (2/π) arcsin ρ | 0 / 0 | Benchmark; no joint extremes |
| Student-t | ρ, ν | (2/π) arcsin ρ | symmetric, > 0, larger for small ν | Joint crashes and joint booms (financial returns) |
| Clayton | θ > 0 | θ/(θ+2) | 2^(−1/θ) / 0 | Joint crashes only (equity losses, credit) |
| Gumbel | θ ≥ 1 | 1 − 1/θ | 0 / 2 − 2^(1/θ) | Joint extremes in the upper tail (insurance claims, losses coded as positive) |
| Frank | θ ≠ 0 | 1 − 4/θ·(1 − D₁(θ)) | 0 / 0 | Symmetric dependence without tail clustering; allows negative dependence |
| Survival Clayton / Gumbel | as above | as above | tails swapped | Upper-tail Clayton or lower-tail Gumbel |

Clayton and Gumbel only model positive dependence. For negative dependence (e.g. an asset and its hedge), flip one variable (use −x) or use Gaussian, t or Frank.

## 4. Fitting and choosing

```python
from dependence import fit_copulas, empirical_tail_dependence
tab = fit_copulas(std_resid_a, std_resid_b)     # ranked by AIC
```

The output reports for each family the parameters, log-likelihood, AIC/BIC, implied τ (compare with the empirical τ) and the implied tail coefficients, plus the empirical tail co-exceedance P(V ≤ q | U ≤ q) at q = 5% (independence gives 0.05).

Reading:
- An AIC gain of more than 10 over the Gaussian copula is strong evidence that a correlation matrix understates joint tail risk.
- The empirical co-exceedance at 5% is a noisy, model-free check of the tails: if it is 0.4 in the lower tail and 0.15 in the upper tail, an asymmetric (Clayton / survival Gumbel) family should win.
- The t copula's ν: below ~10, tail dependence is material. Its λ is the same in both tails; if the data are clearly asymmetric, it will lose to Clayton-type families.
- Formal goodness-of-fit (Cramér-von Mises on the empirical copula with parametric bootstrap) is the next step for publication-grade work; AIC ranking is enough for most applied questions.

Gaussian-copula parameter shortcuts: ρ = sin(πτ/2) (inversion of Kendall's τ, robust) or the correlation of normal scores Φ⁻¹(u). Both approximate the MLE; `fit_copula(..., "gaussian")` gives the MLE and its log-likelihood, so it is comparable with the other families by AIC.

## 5. Simulation and portfolio risk

```python
uv = simulate_copula("student_t", [rho, nu], n=100_000)
ra = np.quantile(ret_a, uv[:, 0]); rb = np.quantile(ret_b, uv[:, 1])   # empirical marginals
port = 0.6 * ra + 0.4 * rb
var99 = -np.quantile(port, 0.01); es99 = -port[port <= -var99].mean()
```

Compare the portfolio VaR/ES from the best copula against the Gaussian copula: the gap is the tail risk a correlation-based model misses. With GARCH marginals, simulate standardized residuals through the copula and rescale by the forecast volatilities.

## 6. Mutual information and distance correlation

- **Mutual information (MI)** I(X;Y) ≥ 0, zero only under independence; captures any functional form. `dependence._mi` uses the Kraskov kNN estimator (scikit-learn), in nats.
- MI is not on a 0–1 scale, and MI(X,X) is infinite for continuous data. **Do not set the diagonal of an MI matrix to 1** and compare it with off-diagonal values in nats. Use the **Linfoot information coefficient** √(1 − e^(−2·MI)), which equals |ρ| for a bivariate normal, so it can be read next to |Pearson|. `mutual_information_matrix` returns that scale with a NaN diagonal.
- `dependence_table(df)` lists, per pair, Pearson, Spearman, Linfoot-normalized MI, distance correlation (Székely-Rizzo, zero iff independent) and the **nonlinear gap** = normalized MI − |Pearson|. Large gaps flag dependence that a correlation matrix misses (volatility co-movement, nonlinearity, tail dependence).
- Significance: `mi_permutation_test(x, y)`. Permutations assume i.i.d. data; for time series, use returns or residuals, or block permutations.
- kNN MI has a bias of about ±0.02 nats in samples of a few thousand; treat gaps below ~0.05 on the Linfoot scale as noise.

For daily stock returns, a positive nonlinear gap usually reflects shared volatility regimes rather than an exploitable nonlinear relationship in the mean. Check by recomputing on GARCH standardized residuals: if the gap disappears, it was volatility.

## 7. Pitfalls

- Fitting copulas on prices or on autocorrelated / heteroskedastic series.
- Concluding "no dependence" from a Pearson correlation near zero.
- Comparing families by log-likelihood without penalizing the extra parameter of the t copula (use AIC/BIC).
- Calling a Gaussian copula "the copula model": its tail dependence is zero by construction.
- Reading MI in nats as if it were a correlation.
