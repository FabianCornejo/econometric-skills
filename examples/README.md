# Worked examples

Each example applies the skill's 7-step workflow (`econometrics/SKILL.md`) to a real question with
Wooldridge's textbook data, using the skill's own scripts. They show what a full analysis looks like:
a hypothesis written before estimating, diagnostics that change the model, robustness checks, and a
report that says what the estimate does *not* prove.

| Example | Question | Methods and skill functions |
|---|---|---|
| [01 Returns to education](01_returns_to_education.ipynb) | How much does a year of schooling raise wages, and how fragile is it to omitted ability? | log-lin OLS, HC3, RESET-driven respecification, `ols_diagnostics`, `missing_report`, `oster`, `sensitivity`, `prediction_intervals` |
| [02 Instrumental variables](02_iv_returns_to_education.ipynb) | Is the OLS return biased? 2SLS with parents' education | `IV2SLS` (`debiased=True`), `endogeneity_test`, `overid_test`, first-stage strength |
| [03 Panel: union premium](03_panel_union_premium.ipynb) | Union wage premium net of fixed worker differences | pooled, RE, FE, clustered SEs, `mundlak_test` (CRE) |
| [04 Distributed lag](04_distributed_lag_fertility.ipynb) | Long-run effect of a tax exemption on fertility | `stationarity_many`, spurious-regression signs, first differences, HAC, `long_run_effect` |
| [05 Limited dependent variables](05_limited_dependent.ipynb) | Hours worked with corner solutions; counts of arrests | `tobit` (three margins, McDonald-Moffitt), Poisson, `count_diagnostics` |

The `.py` files are the source (cells marked with `# %%`, runnable as scripts and checked in CI);
the notebooks are generated from them with outputs:

```bash
pip install -r requirements-dev.txt jupytext nbconvert ipykernel
examples/build_notebooks.sh
```

Every number quoted in the text of an example comes from the output printed right above it.
