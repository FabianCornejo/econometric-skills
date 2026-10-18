# econometric-skills

Agent skills for Claude focused on applied econometrics.

## `econometrics`

A rigorous applied-econometrics workflow in Python (statsmodels, linearmodels, arch): regression with full diagnostics, endogeneity and instrumental variables, panel data, logit/probit/tobit, time series (unit roots, structural breaks, ARIMA, VAR/SVAR, cointegration/VECM), volatility (GARCH), forecasting with out-of-sample evaluation, and demand and price-elasticity estimation, with data from FRED, the World Bank, IMF, OECD and Brazil's BCB, IBGE and Ipeadata.

The skill's instructions are in English; Claude answers in the user's language.

```
.claude-plugin/
├── plugin.json                    plugin manifest
└── marketplace.json               lets the repo act as its own marketplace
skills/econometrics/
├── SKILL.md                       7-step workflow
├── scripts/diagnostics.py         test batteries with plain-language conclusions
└── references/
    ├── ols-regression.md
    ├── time-series.md
    ├── panel-iv.md
    ├── limited-dependent.md
    ├── demand-elasticities.md
    ├── volatility.md
    ├── data-sources.md
    ├── software-equivalents.md    Stata/EViews → Python
    └── bibliography.md
```

### Installation

- **Claude Code (plugin):**

  ```
  /plugin install econometrics --marketplace fabiancornejo/econometric-skills
  ```

  On older Claude Code versions, add the marketplace first: `/plugin marketplace add fabiancornejo/econometric-skills`, then `/plugin install econometrics@econometric-skills`.
- **Claude Code (manual):** copy the `skills/econometrics` folder into `~/.claude/skills/`.
- **Claude.ai / desktop app:** zip the `skills/econometrics` folder (or use a release `.skill` file) and upload it under Settings → Capabilities → Skills.

Python dependencies: `pip install statsmodels linearmodels arch`.

### Quick test of the script

```bash
python3 skills/econometrics/scripts/diagnostics.py ols your_data.csv --y log_sales --x log_price,log_income
python3 skills/econometrics/scripts/diagnostics.py stationarity your_data.csv --cols policy_rate,cpi
```

## Feedback and contributions

The skill is open for use, testing and suggestions:

- **Found a bug or a strange result?** Open an [issue](../../issues) describing the question asked, the data (or a simulated example) and the answer you got.
- **Suggestion for a method, test or data source?** Also via an issue.
- **Want to fix or extend something?** Fork the repository and open a pull request.

## License

[MIT](LICENSE) — free to use, including commercially, keeping the copyright notice.
