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
| Statistics foundations | Describing data (skewness, kurtosis in both conventions, outliers), one- and two-sample t (Welch by default, paired), variances, proportions (Wilson, exact, Newcombe) and sample size, goodness of fit and normality, contingency tables with Fisher, Cramer's V and a Simpson's-paradox check, rank tests with Hodges-Lehmann intervals, correlations, power curves, multiple-testing adjustments | `inference.py` |
| Regression and diagnostics | OLS with full test batteries, robust/HAC SEs, influence, linear restriction tests (SSR form, restricted estimates), Prais-Winsten/Cochrane-Orcutt, J and encompassing tests for non-nested models, FGLS, prediction intervals, omitted-variable bias, Oster and Cinelli-Hazlett sensitivity (with formal bounds: a confounder k times as strong as a covariate) | `diagnostics.py`, `specification.py` |
| Classical linear models | ANOVA and multiple comparisons, ANCOVA, robust and quantile regression, permutation tests, Box-Cox, lasso/ridge/PCR/PLS, post-double-selection | `linear_models.py` |
| Estimation engine | Maximum likelihood from any likelihood, LR/Wald/LM, NLS, delta method, bootstrap, wild cluster bootstrap | `estimation.py` |
| Causal graphs | Minimal and optimal adjustment sets, good vs bad controls (confounders, mediators, colliders, instruments), tests of the graph's implications | `dag.py` |
| Causal inference | Matching, IPW, AIPW, double ML with omitted-confounder bounds and benchmarks, regression discontinuity, staggered DiD (Callaway-Sant'Anna) with timing checks and covariates (doubly robust, conditional parallel trends), the common causal covariates test, a TWFE hidden-linearity diagnostic, the Goodman-Bacon decomposition, cluster bootstrap at the assignment level, synthetic control, randomization inference for few treated units (permuting at the cluster level), DiD with one treated unit (calibrated inference on the gap series), a falsification matrix with the MDE of every check | `causal.py` |
| Heterogeneous effects | CATE (DR-learner, linear DML, forests, meta-learners), BLP/GATES/CLAN, TOC/Qini, policy trees with costs | `hte.py` |
| Panel, IV, limited dependent variables | FE/RE/Mundlak, an IV report (first-stage strength, endogeneity and overidentification tests, Anderson-Rubin set), LPM/probit/logit side by side with marginal effects, odds ratios and classification against the base rate, Heckman two-step with corrected SEs, Tobit, count models; SUR and 3SLS with identification checks; Arellano-Bond/Blundell-Bond GMM with AR(2), Hansen J and instrument counts | `diagnostics.py`, `discrete_choice.py`, `systems.py` |
| Time series and forecasting | Correlogram and filters, unit roots with bootstrap p-values, breaks, long memory, ARIMA/ARIMAX, bootstrap forecast intervals and combination; VAR with Kilian bootstrap IRF bands, FEVD under several orderings, SVAR (short-run, long-run, signs), Johansen and VECM with the error-correction check; walk-forward evaluation, Diebold-Mariano | `ts_univariate.py`, `multivariate.py`, `forecasting.py` |
| Finance and risk | Betas for WACC (rolling, Dimson, Blume/Vasicek, Hamada/Harris-Pringle), factor models (Fama-French, NEFIN), Fama-MacBeth risk premia, event studies, performance ratios, Markowitz, GARCH/GJR/EGARCH with regressors in the variance and sign-bias tests, VaR/ES with extreme value theory, VaR and ES backtests, copulas, Kalman, regimes, look-ahead-free backtests | `finance_empirics.py`, `volatility.py`, `risk.py`, `dependence.py`, `state_space.py`, `regimes.py`, `backtest.py` |
| Survival | Kaplan-Meier, Cox with clustered SEs, PH tests | `survival.py` |
| Text as data | Regex coding, topics, classifiers, LLM coding with valid downstream inference (DSL) | `text_data.py` |
| Segments | Logit leaf model with honest splits, temporal holdout and calibration | `segments.py` |
| Teaching lab | Monte Carlo demos (central limit theorem, CI coverage, bias vs MSE, multiple testing, simultaneity, LPM bounds, multicollinearity, omitted variable, bad control, measurement error, heteroskedasticity, spurious regression, weak IV, few clusters, staggered DiD, DiD with covariates), power and MDE, including simulated power for DiD with serial correlation | `simulation.py` |
| Data and reporting | Awkward Brazilian files, audits, missing data, safe merges, PDFs, scraping; FRED, World Bank, IMF, OECD, BCB, IBGE, Ipeadata, NEFIN; publication tables, run manifests (data hashes, versions, seed) and independent cross-checks | `data_prep.py`, `reporting.py` |

### How it is validated

- **2578 checks** in `tests/`: replications of published results (Wooldridge's *Introductory Econometrics* examples, Cinelli-Hazlett's Darfur example), agreement with statsmodels, scipy and the `csdid`/DRDID port, a curriculum map whose every function and section must exist, recovery of known parameters, and Monte Carlo checks of bias, CI coverage and test size.
- **Validation diary** ([VALIDATION.md](VALIDATION.md), since v2.6.7): one row per public function (289), with its level (replicated against a fixed reference, known truth, execution only, no test), the reference, the tolerance its test uses, the largest difference observed against external references (`tools/validation_diffs.py`), and what is not covered; `tests/test_validation_md.py` keeps it in sync with the code and the tests.
- **Continuous integration** runs every test suite and worked example on the minimum (core packages and optional extras pinned at their minimums) and the latest supported dependency versions, on Python 3.12, 3.13 and 3.14 and on Windows, and checks that `dist/econometrics.skill` matches the source.
- **Blind evals** (`evals/`): plain-language prompts with objective must / must-not criteria, run in a fresh conversation before each release.
- **Routing evals** (`econometrics/evals/routing/`, run with `python tools/routing_evals.py`): `claude plugin eval` sends twenty-two plain requests (two groups, a quick proportion interval, controls, a weak instrument, VaR validation, six reviews (a null result, a staggered TWFE read as the ATT, a reversible treatment estimated as absorbing, HAC inference with one treated store, parallel trends declared confirmed by one pre-trend test, standard errors clustered below the level of assignment), a lesson, a larger analysis, a promotion's effect on two stores' sales, a beta for a valuation's WACC, a credit programme's effect with a post-treatment control and self-selected participants (the identification checklist must surface both), a sales forecast whose answer must NOT run that checklist, and five unrelated ones that must not trigger the skill: an e-mail, a tip calculation, a spreadsheet formula, a cash budget and a legal-risk review), each twice: with the plugin alone and with 48 decoy data and statistics skills that overflow Claude Code's skill listing. Since v2.5.0.2, `--arm nameonly` adds about 840 decoys, so that econometrics is listed by name only, as on a computer with hundreds of skills. The summary reports how often the skill was reached and how often it fired by mistake in each setting; in v2.4.5 it was reached in all positive cases in both settings, with no false triggers (CHANGELOG). The cases need no shell tools, so they also run on native Windows; each run is a real model call on your account (about $0.10).
- **Independent verification**: headline numbers are recomputed by a second route (DR vs outcome regression vs IPW inside `att_gt`, Callaway-Sant'Anna vs the imputation estimator in `did_compare`, or `reporting.cross_check`), and deliverable scripts end with a run manifest, so a number can be traced and reproduced.
- **Look-ahead discipline**: every rolling estimate uses only data available at the time; the usual traps (contemporaneous regressors, full-sample scaling, smoothed probabilities, signal/return misalignment) are avoided in code and documented.

Release notes for every version are in [CHANGELOG.md](CHANGELOG.md).

### Repository layout

```
.claude-plugin/
└── marketplace.json       Claude Code plugin marketplace (plugin source: econometrics/)
econometrics/
├── SKILL.md               router: modes, routing table, rules, output contract
├── statistics/SKILL.md    entry point for statistics questions (Claude Code only)
├── modes/                 /econometrics:consultant, :teacher, :reviewer commands (Claude Code only)
├── agents/                econometrician subagent with the skill preloaded (Claude Code only)
├── evals/routing/         routing cases for `claude plugin eval` (Claude Code only)
├── scripts/               26 tested modules (see the table above) and workspace.py
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
tools/        routing_evals.py: routing evals in a clean and a crowded skill listing
dist/         econometrics.skill, the packaged skill
```

### Installation

- **Claude app (web or desktop):** download [`econometrics.skill`](https://github.com/FabianCornejo/econometric-skills/raw/main/dist/econometrics.skill) (also attached to the [latest release](../../releases/latest)) and upload it under **Customize → Skills**. Under **Settings → Capabilities**, keep **Run code in the cloud** on (required for skills) and allow network egress so missing Python packages can be installed.
- **Claude Code (plugin, recommended):** one command in a Claude Code session:
  ```
  /plugin install econometrics --marketplace FabianCornejo/econometric-skills
  ```
  (Claude Code 2.1.275 or later; on older versions, or from a terminal: `claude plugin marketplace add FabianCornejo/econometric-skills`, then `claude plugin install econometrics@econometric-skills`). Restart Claude Code and that is all. You need Python 3.10 or later on the machine. On Windows, `python` often opens the Microsoft Store alias ("Python was not found") even with Python installed; the skill detects it and gives you commands that work (usually `py -3.13`).

  *What happens the first time.* Before the first analysis in a folder, Claude runs `setup_env.py --doctor`, which changes nothing and lists in one place what a complete, up-to-date setup still needs: the Python to use (on Windows `python` often opens the Microsoft Store alias, and a machine may have several Pythons; the doctor names the one that already has the packages), missing packages, whether this skill is behind the latest release, copies of the skill at different versions (plugin and Claude app), plugin auto-update, and an overcrowded skill listing. Claude then asks you **once** about everything on that list and does only what you approve. Packages are installed only through `setup_env.py --install`: numpy, pandas, statsmodels, linearmodels, arch, scikit-learn, and optional extras when a task needs them (`excel` for .xlsx files, `plots` for charts, `rd`, `regimes`, `web`, `data-br`, `data-intl`…); large ones (`ml`, `pdf`, `ocr`, `text`, `llm`, about 50 to 205 MB) get a question of their own with the size. The scripts it gives you find the skill by themselves (the plugin folder, any copy under `~/.claude/skills`, including the one synced from the Claude app, or a clone) (removed skills, which Claude Code keeps in a hidden `.trash` folder, are skipped), so you run them as delivered, with no paths to edit. If a script ever reports a missing package, run the command it prints.

  *If Claude does not use the skill.* Type `/skills`: `econometrics` should be listed with its description. The plugin offers two entry points to the same skill: `econometrics:econometrics` and `econometrics:statistics` (for tests of means, proportions, variances, tables, normality and correlation); the second only sends Claude to the first, so the name alone points statistics questions to the skill. With hundreds of skills installed, Claude Code keeps every name but drops descriptions to fit its listing budget (1% of the context window), **starting with the skills you invoke least** ([Claude Code docs, "Skill descriptions are cut short"](https://code.claude.com/docs/en/skills), read on 2026-11-07), so a newly installed or rarely used skill is the first to be listed by name only (`/doctor` shows the listing's cost; `claude --debug` logs "Skill listing over budget"). On a computer with 865 skills (2026-11-07), only 14 kept a description, econometrics was listed by name only, and a difference-in-differences request was answered without the skill.

  *The session-start note (since v2.5.0.2).* The plugin has one hook: at the start of each new session (and after `/clear` or compaction; not on resume, which is assumed to replay the saved note, not verified) it prints one line of 394 characters telling Claude that statistical work (hypothesis tests, confidence intervals, sample size, regression, the effect of a promotion or policy, forecasts from statistical models, VaR or ES) goes to `econometrics:econometrics`, even when the listing shows only its name, and that the note does not apply to anything else, such as budgets, valuations or legal risk. It runs no program besides `echo` and needs no Python. Measured with `tools/routing_evals.py --arm nameonly` on Claude Code 2.1.319 (855 skills, 11 with a description, econometrics by name only in every name-only-arm run), 4 statistics requests and 5 unrelated ones, 2 runs each: without the note the skill was used in 2 of 8 runs, with it in 8 of 8; it never fired on an e-mail, a sum, a spreadsheet formula, a cash budget or a legal-risk review (0 of 10 runs with many skills, 0 of 10 with the plugin alone). These are few runs of few requests, and the measurement leaves out the hooks of other plugins. To turn it off, turn the plugin off in that folder. Other ways out: a clean analysis folder (`workspace.py`, below) or a `.claude/settings.json` there holding `{"enabledPlugins": {"<plugin>@<marketplace>": false}}` for the plugins you do not need; raise the budget with the `skillListingBudgetFraction` setting; keep the skill in one place (a copy uploaded to the Claude app appears in Claude Code too, as `anthropic-skills:econometrics`, a third name for the same skill); or ask for it by name: "use the econometrics skill".

  *Commands that always reach the skill.* `/econometrics:consultant` (then the question), `/econometrics:teacher` (the topic) and `/econometrics:reviewer` (the material) load the skill in the Consultant, Teacher or Reviewer mode. They do not depend on Claude choosing the skill, so they work however many plugins are installed. Claude never calls them on its own (`disable-model-invocation`), so they do not compete with other skills for requests.

  *A dedicated agent.* `@agent-econometrics:econometrician <task>` hands a self-contained piece of work (an analysis on a file, a model check, a forecast, VaR, a review) to a subagent that starts with the whole skill preloaded, in its own context, so it does not depend on the skill listing either; it returns the main table, the diagnostics and the paths of the scripts it wrote. Claude may also delegate to it by itself. Teaching works better in the main conversation (`/econometrics:teacher`). Use the @-mention or delegation: `claude --agent econometrics:econometrician` starts a session *as* the agent, which does not preload skills (the agent then loads the skill itself).

  *A clean analysis folder.* `python <skill folder>/scripts/workspace.py <folder>` writes `<folder>/.claude/settings.json` turning off, in that folder only, every enabled plugin except econometrics (`--keep <plugin>@<marketplace>` keeps others; `--dry-run` shows the file first). Other plugins' skills then stop competing for statistics requests and their hooks stop running there, while the rest of the computer is unchanged. `claude plugin list`, run inside the folder, shows what is on.

  *Updates.* Plugins from third-party marketplaces do not update by themselves unless auto-update is on for the marketplace (it is off by default). Turn it on once: `/plugin` → Marketplaces → `econometric-skills` → **Enable auto-update** (or let Claude do it with `setup_env.py --enable-autoupdate` when the doctor offers it, which writes it to `~/.claude/settings.json` after a backup). Otherwise update by hand: `/plugin` → Marketplaces → `econometric-skills` → `u`, or `claude plugin marketplace update econometric-skills` and `claude plugin update econometrics@econometric-skills`; then restart Claude Code. `claude plugin list` shows the version, and `setup_env.py --doctor` compares it with the latest release. If you also uploaded the skill to the Claude app, Claude Code shows that copy too, synced from your account: update both at every release, or keep the skill in only one place; the doctor warns when their versions differ.

  *Optional.* To use the skill's functions in your own code outside Claude, run `python ~/.claude/plugins/marketplaces/econometric-skills/econometrics/scripts/setup_env.py --check`: it confirms the packages and prints the folder to add to `sys.path` (or to `ECONOMETRICS_SKILL_DIR`). That folder is the stable copy of the plugin; the versioned cache folder changes at every release.
- **Claude Code (manual):** copy the `econometrics` folder into `~/.claude/skills/`.
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
