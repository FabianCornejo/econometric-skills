# econometric-skills

Agent skills for Claude focused on applied econometrics.

## `econometrics`

A rigorous applied-econometrics workflow in Python (statsmodels, linearmodels, arch): regression with full diagnostics, endogeneity and instrumental variables, panel data, logit/probit/tobit, time series (unit roots, structural breaks, ARIMA, VAR/SVAR, cointegration/VECM), volatility (GARCH), forecasting with out-of-sample evaluation, and demand and price-elasticity estimation, with data from FRED, the World Bank, IMF, OECD and Brazil's BCB, IBGE and Ipeadata.

**Financial econometrics (v1.1):**
- **Forecasting:** auto-ARIMA without pmdarima, ARIMAX with properly lagged regressors, walk-forward VAR forecasts, naive/seasonal/historical-mean benchmarks, Diebold-Mariano and Pesaran-Timmermann tests.
- **Risk:** VaR and Expected Shortfall (historical, normal/t, EWMA, filtered historical simulation, rolling GARCH) with Kupiec, Christoffersen and Basel traffic-light backtests.
- **Dependence:** rank correlations, tail dependence, copulas (Gaussian, Student-t, Clayton, Gumbel, Frank and rotations) by maximum likelihood, copula simulation, mutual information on a correlation scale, distance correlation.
- **State space and regimes:** Kalman-filter time-varying regression with ML-estimated noise, pairs-trading layer, Markov-switching models, HMMs with filtered (real-time) probabilities, K-means regimes.
- **Backtesting:** one-bar-lag convention, transaction costs, performance table, walk-forward splits, time-series momentum, Sharpe confidence intervals.

Every rolling estimate uses only data available at the time. Look-ahead traps (contemporaneous regressors, full-sample scaling, smoothed probabilities, filtered Kalman residuals, signal/return misalignment) are avoided in code and documented in the references.

The skill's instructions are in English; Claude answers in the user's language.

```
econometrics/
├── SKILL.md                         7-step workflow
├── scripts/
│   ├── diagnostics.py               test batteries with plain-language conclusions
│   ├── forecasting.py               auto-ARIMA, ARIMAX, VAR forecasts, walk-forward evaluation
│   ├── backtest.py                  look-ahead-free backtests, performance, momentum
│   ├── risk.py                      VaR/ES estimators and backtests
│   ├── dependence.py                copulas, tail dependence, mutual information
│   ├── state_space.py               Kalman filter, half-life, pairs trading
│   └── regimes.py                   Markov switching, HMM, K-means regimes
└── references/
    ├── ols-regression.md
    ├── time-series.md
    ├── panel-iv.md
    ├── limited-dependent.md
    ├── demand-elasticities.md
    ├── volatility.md
    ├── risk-var.md
    ├── forecasting-backtesting.md
    ├── dependence-copulas.md
    ├── state-space-regimes.md
    ├── data-sources.md
    ├── software-equivalents.md      Stata/EViews/R → Python
    └── bibliography.md
tests/test_financial_modules.py      37 known-truth checks on simulated data
```

### Installation

- **Claude.ai / desktop app:** zip the `econometrics` folder (or use a release `.skill` file) and upload it under Settings → Capabilities → Skills.
- **Claude Code:** copy the `econometrics` folder into `~/.claude/skills/`.

Python dependencies: `pip install statsmodels linearmodels arch scikit-learn` (optional: `hmmlearn`, `yfinance`).

### Quick test of the script

```bash
python3 econometrics/scripts/diagnostics.py ols your_data.csv --y log_sales --x log_price,log_income
python3 econometrics/scripts/diagnostics.py stationarity your_data.csv --cols policy_rate,cpi
python3 econometrics/scripts/forecasting.py your_data.csv --col yield --season 12
python3 tests/test_financial_modules.py
```

## Feedback and contributions

The skill is open for use, testing and suggestions:

- **Found a bug or a strange result?** Open an [issue](../../issues) describing the question asked, the data (or a simulated example) and the answer you got.
- **Suggestion for a method, test or data source?** Also via an issue.
- **Want to fix or extend something?** Fork the repository and open a pull request.

## License

[MIT](LICENSE) — free to use, including commercially, keeping the copyright notice.
