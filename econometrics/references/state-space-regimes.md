# State-space models, Kalman filter, pairs and regime switching

Use when parameters change over time (a hedge ratio, a beta, a pass-through coefficient), when a trend or signal must be extracted from noisy data, or when the series alternates between regimes (calm/turbulent markets, expansions/recessions). Code: `scripts/state_space.py` and `scripts/regimes.py`.

## 1. Kalman filter for time-varying coefficients

Model:

```
y_t = a_t + b_t x_t + e_t,              e_t ~ N(0, R)
(a_t, b_t) = (a_{t-1}, b_{t-1}) + w_t,  w_t ~ N(0, Q),  Q = δ/(1−δ)·I
```

δ controls how fast the coefficients can move (δ → 0: constant OLS coefficients; larger δ: faster adaptation and more noise).

```python
from state_space import fit_kalman_regression
kf, par = fit_kalman_regression(log_p_y, log_p_x)     # δ and R by maximum likelihood
kf[["b_pred", "b_filt", "z"]]
```

Outputs:
- `a_pred, b_pred`: coefficients **predicted** for t using data up to t−1 (use these to hedge or forecast at t).
- `a_filt, b_filt`: coefficients **updated** with y_t.
- `e`: one-step prediction error (innovation) y_t − (a_pred + b_pred·x_t); `sqrt_S`: its standard deviation; `z = e / sqrt_S`.

Rules:
- **Signals use the innovation z**, never the residual computed with the filtered coefficients. The filtered residual has been partly absorbed by the update, so it is mechanically shrunk towards zero; z-scores built from it with a rolling mean and std mix scales and inflate backtests. Under a correct model, z ~ N(0,1) and is serially uncorrelated: `fit_kalman_regression` prints its mean and sd as a specification check.
- **Estimate δ and R by maximum likelihood** (prediction-error decomposition) rather than tuning them on backtest P&L, which is overfitting with extra steps.
- Prefer log prices: b_t is then an elasticity, and the spread is a log price ratio.
- For richer models (local level, local linear trend, time-varying regressions with several coefficients and estimated variances), use `statsmodels.tsa.statespace` (`UnobservedComponents`, or a custom `MLEModel`) and its smoothed estimates for description only. Smoothed (two-sided) estimates use future data.

## 2. Pairs trading built on the filter

1. **Economic rationale first**: same industry, same risk factors, a reason for the spread to revert.
2. **Cointegration pretest**: `cointegration_pretest(y, x)` runs Engle-Granger in both directions (it is not symmetric) and Johansen. Without cointegration, "mean reversion" in the spread is an assumption. A time-varying hedge ratio can make any spread look stationary in sample; that is precisely why the pretest and out-of-sample evaluation matter.
3. **Mean-reversion speed**: `half_life(spread)` from Δs_t = c + φ s_{t−1}; half-life = −ln2 / ln(1+φ). Half-lives longer than the holding horizon make the strategy impractical. `hurst_exponent` < 0.5 supports mean reversion.
4. **Signals**: `pairs_position(z, entry=2, exit=0.5, stop=4)`; the stop closes positions when the relationship may be breaking.
5. **Returns**: `pairs_returns(y_price, x_price, hedge=kf["b_pred"], position)`: one unit of y against b units of x decided at t, P&L over t+1, divided by gross exposure, with costs on both legs.
6. **Evaluation**: as in `forecasting-backtesting.md` (out of sample, costs, number of variants, Sharpe CI).

If the filter is run on log prices, the hedge in units is b·(P_y/P_x); `pairs_returns` expects the hedge in units of x per unit of y, so convert when needed.

## 3. Markov-switching models (Hamilton, 1989)

```python
from regimes import markov_switching
ms = markov_switching(returns, k_regimes=2)                 # switching mean and variance
ms = markov_switching(gdp_growth, k_regimes=2, order=4)     # Markov-switching AR (business cycle)
```

Reports, with regimes ordered from lowest to highest variance: the mean and variance of each regime, the transition matrix, expected durations 1/(1 − p_ii), and filtered and smoothed probabilities.

- **Filtered** P(s_t | data up to t): usable in real time (signals, risk switches).
- **Smoothed** P(s_t | all data): best for dating past regimes (recessions); uses the future.
- Use `search_reps` (random starting values): the likelihood is multimodal.
- Number of regimes: standard LR tests do not apply (nuisance parameters unidentified under H0). Compare AIC/BIC and check that each regime is economically interpretable and persistent.

## 4. Hidden Markov models and clustering on features

When regimes should depend on several features (returns, realized volatility, drawdown, trend, volume), use `regime_features(prices)` and then:

- `hmm_regimes(X, n_states, train_end)`: Gaussian HMM (hmmlearn) fitted on the training window, with **filtered** probabilities computed by the forward recursion for every date. hmmlearn's `predict`/`predict_proba` use the whole sequence (Viterbi / smoothed), which looks ahead; they are fine for describing the training sample only.
- `kmeans_regimes(X, k, train_end)`: K-means on standardized features. Scaler and centroids are fitted on the training window. Validity: silhouette (> 0.5 strong, < 0.25 weak structure), Davies-Bouldin (lower is better), Calinski-Harabasz. `choose_k` tabulates them for several k.
- K-means ignores time: consecutive days can flip between clusters. HMMs and Markov switching impose persistence through the transition matrix, which is usually what "regime" means.
- Labels are arbitrary between fits; the functions relabel them by increasing volatility (0 = calmest).
- `compare_labels(a, b)`: adjusted Rand index (1 = identical partitions, 0 = chance) and a cross-tab.
- `regime_summary(returns, labels)`: frequency, annualized mean, volatility, Sharpe and average duration per regime. A useful regime split shows clearly different volatility and persistent durations.

Deep-learning embeddings (autoencoders, transformers) before clustering add many parameters and few guarantees; use them only after a Markov-switching or HMM baseline, with the same train/test discipline, and check that they beat it out of sample.

## 5. Pitfalls

- Using smoothed probabilities or full-sample HMM/K-means fits in a backtest.
- Fitting the scaler on the whole sample before splitting.
- Reading "regime 2" as the same regime across different fits.
- Hand-tuning δ, the number of regimes or thresholds on backtest performance.
- Trading a spread without a cointegration rationale, or measuring its z-score with the filtered residual.
