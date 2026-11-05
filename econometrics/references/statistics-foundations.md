# Statistics foundations (Module 0)

The layer under every regression: probability, distributions, sampling, estimation and tests. Use it for questions on one or two variables (compare groups, a proportion, a variance, an association in a table, "is this normal?"), for Teacher mode's first module, and whenever a regression result needs its statistical meaning explained. Functions: `scripts/inference.py`; teaching demos: `simulation.demo_clt`, `demo_ci_coverage`, `demo_estimator_mse`, `demo_multiple_testing`; estimation engine: `estimation.mle`, `delta_method`, `bootstrap`.

## Contents
1. Probability
2. Random variables and distributions
3. Sampling distributions and the central limit theorem
4. Estimators and how to judge them
5. Confidence intervals
6. Hypothesis tests
7. Which test? (decision table)
8. Goodness of fit and normality
9. Categorical data
10. Many tests at once
11. Exploratory analysis before any model
12. Traps found in course material
13. Bridge to regression

## 1. Probability

- **Rules.** P(A or B) = P(A) + P(B) - P(A and B); conditional P(A|B) = P(A and B) / P(B); independence means P(A and B) = P(A) P(B), equivalently P(A|B) = P(A). Mutually exclusive events with positive probabilities are *dependent* (one rules out the other).
- **Total probability and Bayes.** P(A) = sum_i P(A|B_i) P(B_i) over a partition; P(B|A) = P(A|B) P(B) / P(A). Typical uses: credit class given a default, condition given a positive test, fraud given an alert. The base rate decides: a 95%-accurate test for a condition with 1% prevalence gives a positive predictive value near 16%.
- **Randomized response** (sensitive questions): with a known randomization probability, the share answering yes identifies the sensitive share (for a fair coin that sends half the people to an innocuous yes/no question, theta = 2 * share_yes - 0.5 when the innocuous yes rate is 1/2).
- **Simpson's paradox.** An association can reverse once a third variable is held fixed (treatment A better within small and large kidney stones, worse overall, because A was used on the harder cases). The crude comparison is confounded; report the stratified one (`inference.contingency(..., strata=)`, Mantel-Haenszel). It is the categorical face of omitted-variable bias (`ols-regression.md` §14).

## 2. Random variables and distributions

- **Expectation and variance rules.** E(a + bX) = a + b E(X); Var(a + bX) = b^2 Var(X); Var(X + Y) = Var(X) + Var(Y) + 2 Cov(X, Y). Law of iterated expectations E(X) = E(E(X|Y)); Var(X) = E(Var(X|Y)) + Var(E(X|Y)). Portfolio of two assets: Var(w X + (1 - w) Y) is minimized at w* = (s_Y^2 - s_XY) / (s_X^2 + s_Y^2 - 2 s_XY).
- **Correlation** is covariance scaled to [-1, 1]; it measures linear association only (zero correlation does not mean independence: Y = X^2 with symmetric X).

| Distribution | Use | Mean | Variance | scipy |
|---|---|---|---|---|
| Bernoulli(p), binomial(n, p) | successes in n independent trials | n p | n p (1 - p) | `stats.binom` |
| Poisson(lambda) | counts of rare events in a fixed window | lambda | lambda (equidispersion) | `stats.poisson` |
| Geometric(p) | failures before the first success, x = 0, 1, ... | (1 - p) / p | (1 - p) / p^2 | `stats.geom(p, loc=-1)` |
| Uniform(a, b) | equal chance on an interval | (a + b)/2 | (b - a)^2 / 12 | `stats.uniform` |
| Exponential(lambda) | waiting time between Poisson events; memoryless | 1/lambda | 1/lambda^2 | `stats.expon(scale=1/lambda)` |
| Normal(mu, sigma^2) | sums and averages (CLT); measurement error | mu | sigma^2 | `stats.norm` |
| Lognormal | positive, right-skewed: incomes, prices, claim sizes | exp(mu + s^2/2) | (e^{s^2} - 1) e^{2 mu + s^2} | `stats.lognorm` |
| Gamma, Pareto | positive skewed amounts; Pareto for power-law tails | | | `stats.gamma`, `stats.pareto` |
| t(df), chi-square(df), F(d1, d2) | sampling distributions of t statistics, variances and their ratios | | | `stats.t`, `chi2`, `f` |

- **Probability calculations** go through the distribution object: `stats.binom(10, 0.55).sf(2)` is P(X >= 3); `stats.poisson(5.3).sf(5)` is P(X > 5); the inverse problem (the reorder point with a 5% stock-out risk) is `stats.norm(mu, sd).ppf(0.95)`.
- **Sums.** The sum of independent Poissons is Poisson (rates add); a linear combination of independent normals is normal; the number of events in t units of time is Poisson(lambda t).

## 3. Sampling distributions and the central limit theorem

- A statistic computed from a random sample has its own distribution across samples. For the mean: E(X̄) = mu and SD(X̄) = sigma / sqrt(n), for any population, which is why precision grows with sqrt(n), not n.
- **CLT**: for i.i.d. draws with finite variance, (X̄ - mu) / (sigma / sqrt(n)) tends to N(0, 1). How fast depends on skewness: a few observations for symmetric data, hundreds for very skewed ones (`simulation.demo_clt` shows the t interval covering less than 95% for a skewed population until n is large). Dependent data (time series) need the long-run variance instead of sigma^2 (`time-series.md`).
- **Normal samples**: (n - 1) S^2 / sigma^2 ~ chi-square(n - 1); (X̄ - mu) / (S / sqrt(n)) ~ t(n - 1); S1^2 / S2^2 over sigma1^2 / sigma2^2 ~ F(n1 - 1, n2 - 1). The t result is robust to moderate non-normality; the chi-square and F results for variances are not.

## 4. Estimators and how to judge them

- **Bias** E(theta_hat) - theta; **variance**; **MSE = variance + bias^2**; **consistency** (theta_hat converges to theta as n grows); **efficiency** (smallest variance in a class; Gauss-Markov for OLS). Unbiased is not best: dividing the sum of squares by n (the MLE) gives a variance estimator with lower MSE than S^2 (`simulation.demo_estimator_mse`).
- **Method of moments**: equate sample moments to their population formulas (uniform(0, theta): theta_hat = 2 X̄; gamma: shape = mean^2 / variance). Simple, consistent, often inefficient.
- **Maximum likelihood**: maximize the log likelihood; the score is its derivative; the Fisher information I(theta) = -E[second derivative]; asymptotically theta_hat ~ N(theta, I(theta)^{-1}), which gives the Wald interval theta_hat +/- z / sqrt(I(theta_hat)). Examples: Poisson lambda_hat = X̄; exponential lambda_hat = 1 / X̄; Pareto with known minimum m, theta_hat = n / sum ln(x_i / m); uniform(0, theta), theta_hat = max(x) (biased down; (n + 1)/n max(x) is unbiased). Any likelihood: `estimation.mle`; a function of parameters (P(X > 20) = (m/20)^theta): `estimation.delta_method`; LR/Wald/LM tests: `estimation.trinity_test` (`estimation-theory.md`).
- **Bayes with a conjugate normal prior**: prior mu ~ N(m0, t0^2), data X̄ ~ N(mu, sigma^2 / n): the posterior mean is the precision-weighted average (m0 / t0^2 + n X̄ / sigma^2) / (1 / t0^2 + n / sigma^2) with variance 1 / (1 / t0^2 + n / sigma^2). The data dominate as n grows.

## 5. Confidence intervals

- A 95% interval is a procedure that covers the true value in 95% of samples when its assumptions hold; a given interval either contains it or not. Never "95% probability that mu is in [a, b]" (that is a Bayesian credible interval).
- **Mean**: X̄ +/- t_{n-1} S / sqrt(n) (`inference.one_sample`); z only with sigma truly known. Difference of means: Welch (`inference.two_sample`); paired data: the interval of the differences (`paired=True`).
- **Proportion**: Wilson by default (`inference.proportion_ci`); the textbook Wald interval p +/- z sqrt(p(1 - p)/n) undercovers badly when n p or n (1 - p) is below about 10 (`simulation.demo_ci_coverage`); Clopper-Pearson is exact and conservative; p +/- z / (2 sqrt(n)) is the worst case. Difference of proportions: Newcombe (`inference.proportion_test([x1, x2], [n1, n2])`).
- **Variance**: [(n - 1) S^2 / chi2_{1-alpha/2}, (n - 1) S^2 / chi2_{alpha/2}]; ratio of two variances with F quantiles (`inference.variance_test`), both only for normal data.
- **Sample size** for a margin of error E: n = z^2 p (1 - p) / E^2 (p = 0.5 if unknown; 1068 for E = 3 points at 95%), with the finite-population correction n / (1 + (n - 1)/N) (`inference.n_for_proportion`); for means and experiments, `simulation.power_mde`.

## 6. Hypothesis tests

- **Logic.** H0 is the claim tested (no effect, a given value); the test statistic measures the distance from H0 in standard errors; the p-value is the probability, *if H0 were true*, of a statistic at least as extreme as the one observed. It is not the probability that H0 is true, and not the size of the effect.
- **Errors.** Type I (reject a true H0) has probability alpha, chosen in advance; type II (fail to reject a false H0) has probability beta; power = 1 - beta depends on the true effect, the noise, n and alpha (`inference.power_curve`). A study with low power that does not reject has learned little. For the n needed to detect a given effect, compute it for that effect (`inference.n_for_power(effect, sd)`); do not read it off a table built for other effect sizes.
- **Never "accept H0".** Not rejecting means the data are compatible with H0 and with every value inside the interval. Write "does not reject at 5%", give the interval and, when the question is "is there no effect?", the effect the study could have detected (MDE) or an equivalence test against a margin that matters.
- **One- vs two-sided**: one-sided only when the direction was fixed before seeing the data; for symmetric statistics (t, z) the one-sided p is half the two-sided one when the estimate falls on the predicted side (not for exact binomial, Fisher, chi-square or F tests).
- **Tests and intervals agree** when the interval inverts the same test: a two-sided t test at alpha rejects mu = mu0 exactly when mu0 lies outside the 1 - alpha t interval. Different methods (an exact binomial p-value next to a Wilson interval) can disagree near the boundary.
- **Statistical vs practical significance**: with large n tiny effects reject; with small n large effects do not. Report magnitude, interval and units (`interpretation-rules.md`).

## 7. Which test? (decision table)

| Question | Data | Function | Notes |
|---|---|---|---|
| Is the mean equal to a value? | one sample | `inference.one_sample` | t; z only with known sigma |
| Do two means differ? | two independent samples | `inference.two_sample` | Welch by default; pooled only with equal variances checked |
| Before/after, matched pairs | paired | `inference.two_sample(paired=True)` | removes between-unit variation |
| Same, but skewed data, small n or ordinal scale | | `inference.nonparametric` | Wilcoxon signed-rank / Mann-Whitney; compares distributions |
| Three or more groups | | `linear_models.anova_table`, `pairwise` | Welch ANOVA or Kruskal-Wallis when variances differ |
| Is a variance equal to a value? Two variances equal? | | `inference.variance_test` | chi-square / F need normal data; Brown-Forsythe is robust |
| A proportion; two proportions | counts | `inference.proportion_ci`, `proportion_test` | Wilson, exact binomial, Newcombe, Fisher |
| Are two categorical variables associated? | table | `inference.contingency` | chi-square, Cramer's V, Fisher (2x2), strata (Simpson) |
| Does a model (Poisson, normal...) fit? | data or counts | `inference.chisq_fit`, `normality` | df = k - a - 1 |
| Are two numeric variables associated? | pairs | `inference.correlation_test` | Pearson, Spearman, Kendall with CI |
| Effect of X on Y with controls | | regression (`ols-regression.md`) | the t test is a regression on a dummy |

## 8. Goodness of fit and normality

- **Chi-square goodness of fit** (`inference.chisq_fit`): compare observed and expected counts per cell; pool cells with expected counts below 5; degrees of freedom k - a - 1, where a is the number of parameters estimated from the data (a Poisson fit with lambda estimated loses one more degree of freedom than a fully specified model). For continuous data the result depends on the bins: prefer Lilliefors/Shapiro-Wilk and a QQ plot.
- **Normality** (`inference.normality`): Shapiro-Wilk (best overall power up to n = 5000), Lilliefors (Kolmogorov-Smirnov with estimated parameters; plain KS with estimated parameters is wrong), Jarque-Bera (skewness and kurtosis; asymptotic, poor in small samples), D'Agostino-Pearson. QQ plot: points on the line = normal; an S shape = heavy tails; a curve = skewness.
- **When normality matters**: exact small-sample t, chi-square and F inference on variances, prediction intervals for single outcomes, VaR and other tail measures. **When it does not**: consistency of OLS and large-sample inference with robust SEs; test the residuals, not y.

## 9. Categorical data

- **Independence test**: chi-square = sum (O - E)^2 / E with E = row total x column total / n, df = (r - 1)(c - 1); valid when expected counts are mostly >= 5; otherwise Fisher's exact test (2x2) or pooling. **Cramer's V** = sqrt(chi2 / (n (min(r, c) - 1))) measures strength (0.1 small, 0.3 medium, 0.5 large, as a rough guide). **Adjusted residuals** beyond +/-2 show which cells drive the association.
- **2x2 tables**: difference in proportions (Newcombe CI), relative risk, odds ratio (exp of a logit coefficient; an odds ratio of 1.39 means odds 39% higher, not 139%). Stratified tables: Cochran-Mantel-Haenszel test, Mantel-Haenszel common odds ratio, Breslow-Day test of homogeneity.

## 10. Many tests at once

- With m independent tests of true nulls at 5%, P(at least one rejection) = 1 - 0.95^m (64% for m = 20). This is how "significant" subgroups, factors and specifications appear by chance (`simulation.demo_multiple_testing`).
- **Family-wise error rate** (any false rejection): Bonferroni (p x m) or Holm (step-down, always at least as powerful). **False discovery rate** (expected share of false rejections among rejections): Benjamini-Hochberg; Benjamini-Yekutieli under arbitrary dependence. `inference.adjust_pvalues`; for group comparisons, `linear_models.pairwise`.
- Decide the family and the correction before looking at results, and report how many tests were run. Pre-registration and holdout samples are the stronger fixes.

## 11. Exploratory analysis before any model

Run `inference.describe_dist` (and `data_prep.audit`, `missing_report`) on every variable that enters a model:
1. Units, frequency, nominal vs real, coding of categories and missing values (999, 0, blank).
2. Location and spread: mean vs median, SD vs IQR; a mean far from the median signals skewness.
3. Shape: skewness, kurtosis (normal = 3; *excess* kurtosis, normal = 0, is what Excel's KURT and pandas return), histogram, QQ plot.
4. Outliers by Tukey's fences: check each one (typo, unit error, real extreme); never drop silently; compare results with and without.
5. Pairs: scatter plots and correlations (Pearson and Spearman); cross-tables for categories, with the third variable that could reverse them.
6. Time: plot the series; trends, seasonality and breaks change every later step (`time-series.md`).

## 12. Traps found in course material

Errors seen in real slides and answer keys; Reviewer and Teacher modes catch them:
- "We do not reject H0, therefore H0 is true" (it is compatible with H0; see §6).
- Excel's KURT is *excess* kurtosis: a value near 2 is already heavier tails than normal, not lighter.
- Excel's F test prints "P(F <= f) one-tail" for what is the upper-tail probability; the two-sided p-value is twice the smaller tail.
- A geometric distribution written as theta (1 - theta)^x must start at x = 0 (failures before the first success); starting at x = 1 the probabilities do not sum to 1.
- A dummy's coefficient in a log model is 100 (e^b - 1)%, not 100 b% (-0.2074 means -18.7%, not -20.7%).
- An odds ratio of 1.39 means 39% higher odds, not 139%.
- Testing a restricted model with the R^2 form of the F test after transforming the dependent variable is wrong; use the sums of squared residuals.
- Correct classification at a 0.5 cutoff is not goodness of fit when one class dominates; compare with the base rate.

## 13. Bridge to regression

- A two-sample t test (pooled) is OLS of y on a group dummy; Welch has the standard error of that regression with HC2 SEs (its degrees of freedom differ). One-way ANOVA is OLS on g - 1 dummies with the overall F test.
- The correlation r is the slope of the standardized regression, and R^2 of a simple regression is r^2.
- Every regression t test, interval and F test inherits the logic of §5–§6; every regression assumption is a statement about a sampling distribution (§3). Next module: simple and multiple regression (`ols-regression.md`).
