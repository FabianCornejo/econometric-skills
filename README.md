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

**Causal inference, survival, text and data (v1.2):**
- **Causal designs:** balance tables, propensity scores, IPW, doubly robust AIPW, nearest-neighbour matching, double/debiased ML, regression discontinuity (rdrobust/IK bandwidths, fuzzy RD, robustness package, density test), TWFE event studies, Callaway-Sant'Anna staggered DiD, synthetic control with placebo inference.
- **DiD timing robustness (v1.3):** anticipation in Callaway-Sant'Anna, automatic warnings when the outcome moves before the recorded date, a date-shift sensitivity table with a date-robust ATT, and estimation when each unit's adoption date is only known within a window (laws phased in court by court).
- **Survival:** Kaplan-Meier, log-rank, Cox with hazard ratios and clustered SEs, proportional-hazards test, discrete-time hazard data.
- **Text as data:** regex coding with audit snippets, TF-IDF, LDA, clustering with representative documents, classifiers with proper validation, agreement with hand coding, misclassification correction, similarity to a reference group.
- **Data and reporting:** robust readers for Brazilian files (`;`, latin-1, decimal comma, XML/HTML ".xls"), large CSVs, safe merges, PDF/OCR, polite scraping; outreg2/esttab-style regression tables.

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
│   ├── regimes.py                   Markov switching, HMM, K-means regimes
│   ├── causal.py                    matching/IPW/AIPW, double ML, RD, event study, CS-DiD + timing checks, synthetic control
│   ├── survival.py                  Kaplan-Meier, Cox, PH test
│   ├── text_data.py                 regex, TF-IDF, topics, clustering, classifiers, validation
│   ├── data_prep.py                 reading, auditing, merging, PDFs, scraping
│   └── reporting.py                 publication-style regression tables
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
    ├── causal-inference.md
    ├── survival.md
    ├── text-as-data.md
    ├── data-collection-wrangling.md
    ├── data-sources.md
    ├── software-equivalents.md      Stata/EViews/R → Python
    └── bibliography.md
tests/                               91 known-truth checks on simulated data
```

### Installation

- **Claude.ai / desktop app:** download [`econometrics.skill`](https://github.com/FabianCornejo/econometric-skills/raw/main/dist/econometrics.skill) (also in the [latest release](../../releases/latest)) and upload it under Settings → Capabilities → Skills (or zip the `econometrics` folder yourself).
- **Claude Code:** copy the `econometrics` folder into `~/.claude/skills/`.

Python dependencies: `pip install statsmodels linearmodels arch scikit-learn` (optional: `rdrobust`, `hmmlearn`, `yfinance`, `pymupdf`, `sentence-transformers`).

### Quick test of the script

```bash
python3 econometrics/scripts/diagnostics.py ols your_data.csv --y log_sales --x log_price,log_income
python3 econometrics/scripts/diagnostics.py stationarity your_data.csv --cols policy_rate,cpi
python3 econometrics/scripts/forecasting.py your_data.csv --col yield --season 12
python3 tests/test_financial_modules.py
python3 tests/test_causal_text_survival.py
```

## Feedback and contributions

The skill is open for use, testing and suggestions:

- **Found a bug or a strange result?** Open an [issue](../../issues) describing the question asked, the data (or a simulated example) and the answer you got.
- **Suggestion for a method, test or data source?** Also via an issue.
- **Want to fix or extend something?** Fork the repository and open a pull request.

## License

[MIT](LICENSE) — free to use, including commercially, keeping the copyright notice.
