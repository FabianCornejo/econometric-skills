# Volatility (GARCH family)

Use for asset returns, exchange rates, market interest rates or any series whose variance changes over time in *clusters*.

## 1. Stylized facts of returns (check before modeling)

1. E(r_t\|past) ≈ 0: returns nearly unpredictable in the mean (martingale difference; weak-form market efficiency).
2. But E(r_t) > 0 over the long run (risk premium).
3. **Fat tails**: kurtosis > 3 (often 7 to 15 for emerging-market stocks).
4. **Volatility clustering**: squared returns are autocorrelated.
5. Higher variance after weekends/holidays.
6. Higher intraday volatility at the open and the close.
7. **Leverage effect**: declines raise volatility more than rises.

Procedure: the price level looks trending → ACF of Δln p shows no structure (ARIMA(0,1,0)) → ACF of r² shows structure → model the variance.

## 2. Preparation
- Log returns in %: `r = 100 * np.log(p).diff().dropna()` (the scale helps `arch`'s optimizer). Use prices adjusted for dividends and splits.
- ARCH-effect test: `het_arch(r - r.mean(), nlags=10)` (H0: no ARCH) and Ljung-Box on r². With no ARCH effect, constant variance is enough.

## 3. Specify and estimate

```python
from arch import arch_model
m = arch_model(r, mean="Constant", vol="GARCH", p=1, o=0, q=1, dist="t")
res = m.fit(disp="off")
print(res.summary())
```

| Model | `arch_model` | When |
|---|---|---|
| GARCH(1,1) | `vol="GARCH", p=1, q=1` | Starting point; usually enough (pure ARCH would need many lags) |
| GJR/TARCH | `vol="GARCH", p=1, o=1, q=1` | Leverage effect (dummy for negative shocks) |
| EGARCH | `vol="EGARCH", p=1, o=1, q=1` | Asymmetry in log-variance, no positivity constraints; significant γ = leverage |
| AR mean / with regressors | `mean="AR", lags=1` or `mean="ARX", x=...` | Autocorrelation or day-of-week effect in the mean |
| Distribution | `dist="normal" / "t" / "skewt" / "ged"` | Compare by AIC/BIC; `t` usually beats the normal |

`arch` does not accept regressors in the variance equation; `volatility.garch_fit(r, "garch" | "gjr", variance_dummies=D)` estimates GARCH/GJR with regressors in the variance (day of week, post-holiday, an event) by its own likelihood with Bollerslev-Wooldridge SEs; without regressors it reproduces `arch` (parameters within 0.002, SEs within 3%).

`volatility.stylized_facts(r)` checks the facts of section 1 (Jarque-Bera, Ljung-Box on r and r², ARCH-LM, leverage correlations); `volatility.garch_fit(r, vol)` reports persistence, half-life, the unconditional variance, Ljung-Box on z and z², the Engle-Ng sign-bias tests (`sign_bias_test`) and AIC/BIC, and warns on persistence ≥ 0.99. For a symmetric GARCH it also fits the GJR(1,1,1) with the same mean and distribution and reports the LR test of γ = 0 (`asymmetry_lr`); a rejection is flagged as omitted asymmetry.

**Names**: GJR-GARCH is the threshold GARCH in the variance; Zakoian's TGARCH models the conditional SD; EGARCH (Nelson) models the log-variance. Calling EGARCH a TGARCH is wrong.

Diagnostics:
- Persistence α + β (+ γ/2 in GJR) < 1; close to 1 = very persistent shocks. Half-life = ln 0.5 / ln(α + β).
- Standardized residuals (`res.std_resid`): no autocorrelation in them or their squares; distribution consistent with the chosen one.
- Omitted asymmetry: the LR of the GJR against the GARCH (χ²(1): γ can be negative, so γ = 0 is interior while α > 0; with α̂ near zero the χ²(1) is conservative; like any LR it assumes the chosen error distribution) is the primary test; if the GJR does not converge, `garch_fit` says so and the sign-bias tests are the fallback. The Engle-Ng sign-bias tests are a complement: with fat-tailed returns they have little power (on the Ibovespa 2010–2026 the joint p was 0.77 while the LR was 31.5, p = 2e-8; in simulations with t(5) errors and γ = 0.08 the LR rejected in every sample and the sign-bias tests in about one in ten). A sign-bias test that does not reject does not show symmetry.

## 4. Forecasting and risk

```python
f = res.forecast(horizon=10, reindex=False)
vol = np.sqrt(f.variance.values[-1])           # % per period
vol_annual = vol * np.sqrt(252)                 # daily → annual
nu = res.params["nu"]
q01 = res.model.distribution.ppf(0.01, [nu])    # quantile of the standardized t
var_99 = -(res.params["mu"] + vol[0] * q01)     # 1-day 99% VaR, in %
```

VaR backtesting: share of violations ≈ 1% (Kupiec) and violations not clustered (Christoffersen). Evaluate volatility forecasts against realized volatility (QLIKE loss).

For rolling out-of-sample VaR/ES (GARCH refit every k days, EWMA, filtered historical simulation) and the backtests themselves, use `scripts/risk.py` and `risk-var.md`.

**EWMA (RiskMetrics)** is the restricted IGARCH(1,1) σ²_t = λσ²_{t−1} + (1−λ)r²_{t−1} with ω = 0 and λ fixed (0.94 daily): no mean reversion of volatility, so long-horizon forecasts stay flat at today's level. `risk.ewma_volatility` seeds it with an initial window only, never with the full-sample variance.

## 5. Limitations
- ARCH models are **backward looking**: they react after the shock has happened. In professional risk management, implied volatility from derivatives (forward looking), Monte Carlo simulation and historical bootstrap have replaced much of the GARCH use.
- Very sensitive to the assumed error distribution.
- Need many observations (> 500 daily, ideally).
- Crises and breaks inflate the estimated persistence; test subsamples.
- Modeling prices instead of returns is a conceptual error.
