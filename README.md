# econometric-skills

[![tests](https://github.com/FabianCornejo/econometric-skills/actions/workflows/tests.yml/badge.svg)](https://github.com/FabianCornejo/econometric-skills/actions/workflows/tests.yml)

Agent skills for Claude focused on applied econometrics.

## `econometrics`

An econometric agent for Claude that works as a **consultant**, a **teacher** or a **reviewer**. It turns a plain question ("do interest rates hurt my sales?", "for which customers does the discount work?", "what beta should go in this WACC?") into an estimated, diagnosed and defensible answer, explains the method behind it, or audits someone else's analysis. Everything runs in Python (statsmodels, linearmodels, arch, scikit-learn) through tested scripts, and Claude answers in the user's language.

### Three modes, one tested core

| Mode | When | What you get |
|---|---|---|
| **Consultant** (default) | Estimate, test, forecast, evaluate a policy | A study card (estimand, design, identifying assumption), the analysis with diagnostics and robustness, a report and a reproducible script |
| **Teacher** | "Explain", "teach", "why does...", a class, an exam | A method card: intuition, model, assumptions, estimator, derivation, a seeded simulation that shows the assumption failing, code, diagnostics, interpretation, pitfalls, and an exercise with its solution |
| **Reviewer** | Review, audit or referee a paper, report or expert opinion | An opinion with the claim, a verdict, findings by severity, and the skill function that runs each check |

The mode changes the form of the answer, never the computations: every number comes from the same scripts.

### What it covers

| Area | Methods | Script |
|---|---|---|
| Regression and diagnostics | OLS with full test batteries, robust/HAC SEs, influence, FGLS, prediction intervals, omitted-variable bias, Oster and Cinelli-Hazlett sensitivity | `diagnostics.py`, `specification.py` |
| Classical linear models | ANOVA and multiple comparisons, ANCOVA, robust and quantile regression, permutation tests, Box-Cox, lasso/ridge/PCR/PLS, post-double-selection | `linear_models.py` |
| Estimation engine | Maximum likelihood from any likelihood, LR/Wald/LM, NLS, delta method, bootstrap, wild cluster bootstrap | `estimation.py` |
| Causal graphs | Minimal and optimal adjustment sets, good vs bad controls (confounders, mediators, colliders, instruments), tests of the graph's implications | `dag.py` |
| Causal inference | Matching, IPW, AIPW, double ML with omitted-confounder bounds and benchmarks, regression discontinuity, staggered DiD (Callaway-Sant'Anna) with timing checks, synthetic control, randomization inference for few treated units | `causal.py` |
| Heterogeneous effects | CATE (DR-learner, linear DML, forests, meta-learners), BLP/GATES/CLAN, TOC/Qini, policy trees with costs | `hte.py` |
| Panel, IV, limited dependent variables | FE/RE/Mundlak, IV/2SLS with weak-instrument checks, logit/probit/Tobit with marginal effects, count models | `diagnostics.py` + references |
| Time series and forecasting | Unit roots, breaks, ARIMA/ARIMAX, VAR/SVAR, cointegration, walk-forward evaluation, Diebold-Mariano | `forecasting.py` |
| Finance and risk | Betas for WACC (rolling, Dimson, Blume/Vasicek, Hamada/Harris-Pringle), factor models (Fama-French, NEFIN), event studies, performance ratios, Markowitz, GARCH, VaR/ES with extreme value theory, VaR and ES backtests, copulas, Kalman, regimes, look-ahead-free backtests | `finance_empirics.py`, `risk.py`, `dependence.py`, `state_space.py`, `regimes.py`, `backtest.py` |
| Survival | Kaplan-Meier, Cox with clustered SEs, PH tests | `survival.py` |
| Text as data | Regex coding, topics, classifiers, LLM coding with valid downstream inference (DSL) | `text_data.py` |
| Segments | Logit leaf model with honest splits, temporal holdout and calibration | `segments.py` |
| Teaching lab | Monte Carlo demos (omitted variable, bad control, measurement error, heteroskedasticity, spurious regression, weak IV, few clusters, staggered DiD), power and MDE, including simulated power for DiD with serial correlation | `simulation.py` |
| Data and reporting | Awkward Brazilian files, audits, missing data, safe merges, PDFs, scraping; FRED, World Bank, IMF, OECD, BCB, IBGE, Ipeadata, NEFIN; publication tables | `data_prep.py`, `reporting.py` |

### How it is validated

- **438 checks** in `tests/`: replications of published results (Wooldridge's *Introductory Econometrics* examples, Cinelli-Hazlett's Darfur example), agreement with statsmodels and scipy, recovery of known parameters, and Monte Carlo checks of bias, CI coverage and test size.
- **Continuous integration** runs every test suite and worked example on the minimum and the latest supported dependency versions, and checks that `dist/econometrics.skill` matches the source.
- **Blind evals** (`evals/`): plain-language prompts with objective must / must-not criteria, run in a fresh conversation before each release.
- **Look-ahead discipline**: every rolling estimate uses only data available at the time; the usual traps (contemporaneous regressors, full-sample scaling, smoothed probabilities, signal/return misalignment) are avoided in code and documented.

Release notes for every version are in [CHANGELOG.md](CHANGELOG.md).

### Repository layout

```
econometrics/
├── SKILL.md               router: modes, routing table, rules, output contract
├── scripts/               20 tested modules (see the table above)
└── references/
    ├── workflow.md            Consultant mode: study card + 7 steps
    ├── interpretation-rules.md
    ├── teaching.md            Teacher mode
    ├── review-checklist.md    Reviewer mode
    ├── ols-regression.md      time-series.md        panel-iv.md
    ├── limited-dependent.md   demand-elasticities.md
    ├── causal-inference.md    heterogeneous-effects.md
    ├── estimation-theory.md   empirical-finance.md
    ├── volatility.md          risk-var.md           dependence-copulas.md
    ├── state-space-regimes.md forecasting-backtesting.md
    ├── survival.md            text-as-data.md
    ├── data-collection-wrangling.md  data-sources.md
    ├── software-equivalents.md       Stata / EViews / R / EconML -> Python
    └── bibliography.md
tests/        test suites (python3 tests/test_<name>.py)
examples/     five worked analyses on Wooldridge data (scripts and notebooks)
evals/        blind-test prompts and results history
dist/         econometrics.skill, the packaged skill
```

### Installation

- **Claude app (web or desktop):** download [`econometrics.skill`](https://github.com/FabianCornejo/econometric-skills/raw/main/dist/econometrics.skill) (also attached to the [latest release](../../releases/latest)) and upload it under **Customize → Skills**. Under **Settings → Capabilities**, keep **Run code in the cloud** on (required for skills) and allow network egress so missing Python packages can be installed.
- **Claude Code:** copy the `econometrics` folder into `~/.claude/skills/`.
- **Recommended:** Claude decides from the descriptions whether to load a skill, and may answer conceptual questions without data ("how should we validate our VaR model?") from general knowledge, without the skill's tested functions and script. Add a standing instruction under **Settings → Account → Instructions for Claude**, for example: *"For any econometrics, statistics or quantitative-risk question (VaR/ES, backtests, betas and WACC, event studies, causal effects, choice of controls, power, forecasting, reviewing empirical work), always load the econometrics skill before answering, even for conceptual questions without data."* See [evals/README.md](evals/README.md#known-limitation-skill-loading-on-conceptual-questions).

Python dependencies: `pip install -r requirements.txt` (optional extras are listed in that file); `pip install -r requirements-dev.txt` adds what the tests and examples need.

### Quick start

```bash
python3 econometrics/scripts/diagnostics.py ols your_data.csv --y log_sales --x log_price,log_income
python3 econometrics/scripts/diagnostics.py stationarity your_data.csv --cols policy_rate,cpi
python3 econometrics/scripts/forecasting.py your_data.csv --col yield --season 12
for t in tests/test_*.py; do python3 "$t"; done     # the full suite (needs requirements-dev.txt)
```

## Feedback and contributions

The skill is open for use, testing and suggestions:

- **Found a bug or a strange result?** Open an [issue](../../issues) describing the question asked, the data (or a simulated example) and the answer you got.
- **Suggestion for a method, test or data source?** Also via an issue.
- **Want to fix or extend something?** Fork the repository and open a pull request.

## License

[MIT](LICENSE) — free to use, including commercially, keeping the copyright notice.
