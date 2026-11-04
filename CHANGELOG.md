# Changelog

All releases of the `econometrics` skill, newest first. Each release is tagged on GitHub with the packaged `econometrics.skill` attached ([releases](https://github.com/FabianCornejo/econometric-skills/releases)).

## v2.3.7 — No number from an estimator that did not converge, and independent checks that are independent

Two gaps found while reviewing the deliverable of the v2.3.6 clean-install test (return to education, WAGE1). The headline results were right (the textbook's Example 7.6 reproduced), but: (1) `linear_models.quantile_regression` used statsmodels' default of 1000 iterations and did not check convergence. At tau = 0.1 with eight regressors it stopped at the limit with educ 0.038897; the solution is 0.038610 (reached at 5000 iterations and stable at 20000). The report gave 4.0% instead of 3.9%, with only a footnote about the warning. (2) The agent passed the same OLS computed by statsmodels and by numpy to `reporting.cross_check` and reported it as an independent confirmation, which checks the arithmetic, not the method.

- **`quantile_regression` reaches the solution or says it did not.** Each tau is fitted with `max_iter=5000` (new argument) and retried with 4x as many if the solver hits its limit. `result["converged"]` and a `converged` table column say which taus reached the solution, and a tau that did not converge is warned about (`warnings`, printed unless `show=False`). Bootstrap replications that did not converge are dropped and counted (`equality_test["B_used"]`); the equality test is flagged `valid: False` when an end tau did not converge. Converged results are unchanged.
- **`cross_check` says what it checked** (`result["kind"]`). With `se`, the routes are different methods (kind `"method"`), printed as an independent check across methods. Without `se`, they must agree to numerical precision, which only two implementations of one estimator do (kind `"implementation"`): printed as an implementation check, with a note that it verifies the arithmetic, not the method, and is not an independent confirmation. Agreement verdicts are unchanged.
- **Rules** (`SKILL.md` §4 and `interpretation-rules.md`, "Rules added in v2.3.7"): a number from an estimator that did not converge is not reported (resolve every optimizer warning first, or leave the number out); the independent route uses a different method or assumptions, and the same estimator computed twice is never reported as an independent confirmation.
- New `tests/test_v237.py` (579 checks in total).

## v2.3.6 — Optional packages go through the same permission question

In the v2.3.5 clean-install test Claude asked to install the eight core packages and then ran a separate `pip install openpyxl`, which the user had not approved, to write an Excel table. `setup_env.py` did not know optional packages, and without openpyxl the skill stopped with pandas' `Missing optional dependency`. openpyxl is needed in three places: `reporting.export_table` to `.xlsx`, `data_prep.read_any` on `.xlsx`/`.xlsm` files and the `diagnostics.py` command line with an Excel file. No change to estimators or results.

- **Optional extras in `setup_env.py`.** `--extra excel` (openpyxl) works with `--check` (exit 1 if the extra is missing) and with `--install`, through the same pip logic as the core packages. `--check` always reports each extra on its own line, without failing the core check. Minimums are kept in sync with `requirements-dev.txt`, which now includes openpyxl.
- **One clear line instead of pandas' error.** `export_table`, `read_any` and the command line call `setup_env.require_extra` right before they need openpyxl; when it is missing they stop with the exact command for the machine (for example `py -3.13 ".../setup_env.py" --install --extra excel`). SpreadsheetML and HTML files saved as `.xls`, which the skill reads without openpyxl, are unchanged; legacy binary `.xls` (xlrd) is out of scope.
- **`SKILL.md` §7**: the permission question names every package to be installed, optional extras included; Claude adds `--extra excel` when the deliverable reads or writes Excel files and never installs a package with a separate pip command. README too.
- `tests/test_setup_env.py` covers the extra's minimum, `--check` and `--install --extra excel`, an unknown extra, the three messages with openpyxl absent, and an `.xlsx` round trip (567 checks in total).

## v2.3.5 — Claude Code sees the skill again

A clean install on a Windows notebook (Claude Code 2.1.317, plugin v2.3.4) showed that Claude never used the skill. Asked for the return to education with diagnostics, it wrote its own statsmodels script in 7 minutes, tested it in a temporary environment and ended with `pip install ...` and `python analise.py`, which on that machine opens the Microsoft Store alias. The debug log (`claude --debug`) had the cause: `Failed to parse YAML frontmatter in .../SKILL.md`. Since v2.2.3 the description held `: ` without quotes ("which do not estimate: VaR ...", "Methods: OLS ..."). YAML reads `: ` as a new key, so the whole header failed, Claude Code loaded the skill with no name and no description, and the skill was never offered to the model, on any machine. Claude Code only logs a warning, and the tests never read the header. No change to estimators or results.

- **The `SKILL.md` header is valid YAML**: the description is quoted. Text and length (1017 characters) are unchanged.
- **New `tests/test_skill_metadata.py`** reads the header with a strict YAML parser and checks the rules for skill metadata: only `name` and `description`; `name` lowercase with hyphens, at most 64 characters and equal to the folder; `description` non-empty, at most 1024 characters, with no angle brackets; the same header in `dist/econometrics.skill`. PyYAML joins `requirements-dev.txt`.
- **Commands that run on Windows.** `setup_env.py --check` prints `Run scripts with:`, a command that starts the Python it checked: `py -X.Y` on Windows when the launcher starts that same interpreter (it works in PowerShell, cmd and Git Bash), else `python` or `python3` when they are that interpreter, else the quoted full path. It adds a note when `python` is the Store alias. `SKILL.md` §7 tells Claude to try `py` first on Windows and to use that command in every instruction to the user; the script skeleton's install message and the virtual-environment hint name it too. `tests/test_setup_env.py` covers the launcher, a virtual environment, the Store alias and macOS/Linux.
- **README**: the Windows note, and what to do when Claude does not pick the skill. With hundreds of skills installed, Claude Code shortens every description to fit its budget (`Skill listing over budget` in the debug log); a `.claude/settings.json` in the analysis folder can turn off plugins there only.
- Total: 557 checks.

## v2.3.4 — Console output no longer stops an analysis on Windows

Running the full test suite on a Windows notebook (Python 3.13, PowerShell) showed that the skill could stop in the middle of an analysis with `UnicodeEncodeError`. When output is captured, as when Claude Code runs a delivered script or a pipe or log file takes it, Python on Windows encodes stdout with the ANSI code page (cp1252), which has no arrow (→), no ≈ and no Greek letters. Twenty-nine `print()` calls in seven modules (`causal`, `data_prep`, `diagnostics`, `forecasting`, `reporting`, `specification`, `state_space`) print such characters; `causal.propensity_score` failed on its overlap line. The CI runs only on Linux, where the streams are UTF-8, so it never saw it. No change to estimators or results.

- **New `scripts/_console.py`.** On import it checks stdout and stderr once. A stream whose error handler raises on such characters (`strict`, or `surrogateescape`, which a PowerShell pipe gives) and whose encoding cannot write them switches to `errors="replace"`: an unencodable character prints as `?` and the run goes on. Output on UTF-8 streams (Linux, macOS, Windows with `PYTHONUTF8=1`) is unchanged; streams that are already tolerant and streams without `reconfigure` (StringIO, notebooks) are left alone. The seven modules import it after their own imports, and so do `examples/_setup.py` and the five test files whose check names hold such characters.
- **Tests read and write text as UTF-8.** Eight `read_text()`, `write_text()` and `open()` calls in four test files gave no encoding and used cp1252 on Windows; `test_did_covariates.py` failed reading `interpretation-rules.md`.
- **New `tests/test_windows_console.py`** forces cp1252 output on any OS, so the Linux CI now catches a regression: each affected module imports and prints an arrow, `causal.propensity_score` runs under `cp1252` and `cp1252:surrogateescape`, UTF-8 output is unchanged, and a module, test or example that holds such a character without importing `_console` fails the suite (540 checks in total).

## v2.3.3 — Scripts also find the copy synced from the Claude app

The first real run of the v2.3.2 flow on Windows (Claude Code 2.1.316) worked: Claude loaded the skill, ran `setup_env.py --check`, listed the eight missing packages for Python 3.14 and 3.12, and asked before installing. It also showed where Claude Code keeps the skill synced from the Claude app: `~/.claude/skills/synced/<id>/econometrics`. The script locator did not search there. A user with only the app copy, and no plugin, would have seen "skill not found".

- The locator block (`interpretation-rules.md`, referenced by the delivery rule) now looks for any `econometrics/scripts/causal.py` under `~/.claude/skills`, newest first. The order is: the variable, the plugin's stable copy, copies under `~/.claude/skills` (including the synced one), the clone. `SKILL.md` and the README say so.
- `tests/test_setup_env.py` gains two checks: a home with only the synced copy is found, and when the plugin and the synced copy both exist, the plugin copy is used (522 checks in total).

## v2.3.2 — Zero-step setup for Claude Code users

A clean install of the plugin, done as a new user would (fresh configuration, Claude Code 2.1.316), worked and showed version 2.3.1. It also showed that, after installing, a user still had to install Python packages by hand and edit or set paths before a delivered script would run. v2.3.2 removes those steps. The user runs one install command; Claude checks the environment, asks before installing anything, and delivered scripts run as received. No change to estimators or results.

- **New `scripts/setup_env.py`**, the same on Windows, macOS and Linux:
  - `--check` reports the Python version (3.10 or later is needed), missing or outdated packages and the skill folder scripts should use.
  - `--install` installs what is missing. It tries a plain pip first and adds `--break-system-packages` only where the system Python is externally managed; the old command used that flag unconditionally, which older pip versions, common on Windows, reject.
  - `--skill-dir` prints the folder to use.

  `SKILL.md` §7 tells Claude to run `--check` before the first script and to ask before `--install` on the user's machine. The plugin copies only `econometrics/`, without `requirements.txt`, so the README's `pip install -r requirements.txt` had been impossible for plugin users, and a clean Python failed at the first import with `ModuleNotFoundError: numpy`.
- **Delivered scripts find the skill themselves.** The script skeleton (`interpretation-rules.md`, referenced in the delivery rule) now starts with a locator block that tries, in order:
  1. `ECONOMETRICS_SKILL_DIR`;
  2. the stable plugin copy, `~/.claude/plugins/marketplaces/econometric-skills/econometrics`;
  3. `~/.claude/skills/econometrics`;
  4. the user's clone.

  The stable plugin copy is a git clone of `main` whose path stays the same across updates; the versioned `plugins/cache/.../<version>/` folder changes at every release, so it must never be written into a script. The block also turns a missing package into a one-line instruction to run `setup_env.py --install`. Users no longer set variables or edit paths.
- **README.** Claude Code installation is one command plus "what happens the first time", updates by menu or terminal (`claude plugin marketplace update econometric-skills`, `claude plugin update econometrics@econometric-skills`), and the synced `anthropic-skills:econometrics` copy for users who also uploaded the skill to the app. A wrong claim was removed: `claude plugin list` shows the version, not the path.
- **Validated** in `tests/test_setup_env.py`:
  - in a fresh virtual environment without packages, `--check` exits 1 and names numpy, and the skeleton stops with the install instruction instead of a traceback;
  - with a fake home holding only the plugin layout and no variable set, the skeleton finds the skill and imports it;
  - the variable takes precedence;
  - the package minimums equal `requirements.txt`;
  - the versioned cache path resolves to the stable copy.

  The version test now reads the CHANGELOG instead of a hard-coded number.

## v2.3.1 — Fixes from the v2.3.0 blind evals: cluster bootstrap, Goodman-Bacon, a diagnostic that cannot be misread

Both v2.3.0 evals passed. The scripts they produced, run against the release, exposed three gaps in the skill itself, and v2.3.1 fixes them. v2.3.0 and v2.3.1 are published together in one release.

- **`causal.att_gt(cluster=...)`**: the bootstrap resamples whole clusters (the state when a law varies by state and the units are municipalities).
  - The point estimate is unchanged and the output reports the cluster count.
  - In a Monte Carlo with state-level shocks, 95% CIs covered 48% of the time resampling municipalities and 95% resampling states.
  - It warns with fewer than ~30 clusters or ~6 treated ones, pointing to `randomization_inference`.
  - `did_timing_sensitivity` and `did_uncertain_dates` pass `cluster=` through.
  - The eval-2 answer had to aggregate to the state level to cluster, which changes the estimand from a municipality average to a state average. The rule now says not to do that.
- **`causal.bacon_decomposition`** (Goodman-Bacon 2021): the TWFE coefficient as a weighted average of every 2×2, typed as treated vs never, earlier vs later, and later vs earlier (already-treated control, forbidden).
  - The weights sum to one and reproduce the TWFE coefficient to 1e-15, with never-treated, always-treated or no never-treated units.
  - It warns when forbidden comparisons carry ≥ 10% of the weight. The eval-2 answer had to send the user to R/Stata for this.
- **`causal.twfe_covariate_diagnostic`**:
  - New `cs_covariates`: the controls under review go in `covariates`, bad ones included, and only the valid pre-treatment confounders feed the benchmark and the hidden-linearity test. The eval-2 script had to pass the parecer's bad control to both, which masked the hidden-linearity signal.
  - Also new: `cs_x_periods` and `cluster`.
  - The Goodman-Bacon check is part of the verdict.
  - A new verdict, **"inconclusive (low power)"**, is given when the benchmark CI is too wide to detect the gap; it previously said "no evidence against TWFE".
  - In the eval-2 scenario, the correct call now flags hidden linearity (p = 3e-19; level coefficient −0.027 for a true −0.025).
- **`reporting.cross_check`** takes `att_gt`'s `res["cross_check"]` Series as it is, and dicts of result dicts. With fewer than two routes it returns `agree=None` and a warning that nothing was verified. The eval-2 script had passed one route and got "agree".
- **`ccc_test` advice for `time`**: `x_periods="base"` for baseline or time-invariant covariates, and `"both"` only for covariates that vary over time and that treatment cannot affect. The eval-1 answer reasoned this correctly against the old text.
- **Rules and checklist**:
  - Cluster the DiD bootstrap at the assignment level.
  - Independent checks need two routes.
  - Relay every diagnostic verdict as printed.
  - The reviewer checklist states what a CCC rejection implies (DID-INT) and asks the authors for a run manifest and an independent recomputation.
- **Evals recorded**: did-covariates-municipal-program passed (the script reproduced every number; true effect +4.0%, DR +4.4%). review-twfe-with-controls passed with gaps (the three above).
- **Validated** in `tests/test_did_v231.py` (25 checks). 507 checks in total, on the minimum and the latest dependency versions.

## v2.3.0 — DiD with covariates, common causal covariates test, independent verification and run manifests

Sources evaluated for this release: Caetano & Callaway (arXiv:2406.15288) and Karim's CCC test (`SunnyRKarim/ccctest`, with Karim & Webb, arXiv:2412.14447) were adopted; the design of `dynare-mcp` (a second, independent solver; model and solver hashes) and Chan's `statespace-toolkit` (byte-for-byte archives, draw-for-draw tests) shaped the verification rules. The Bayesian state-space and bounded-outcome modules are planned for v2.4.

- **`causal.att_gt(covariates=...)`**: Callaway-Sant'Anna under parallel trends that hold only conditional on covariates.
  - Covariates enter in levels in each group-time 2×2, never differenced away.
  - Three estimators via `est_method`:
    - `"dr"` (default): doubly robust, Sant'Anna & Zhao 2020.
    - `"reg"`: outcome regression, Heckman-Ichimura-Todd.
    - `"ipw"`: normalized IPW, Abadie 2005.
  - `x_periods="base"` uses pre-treatment values; `"both"` adds the change in time-varying covariates not affected by treatment (Caetano & Callaway's setting).
  - The bootstrap re-estimates every nuisance model.
  - Comparison units with propensity ≥ 0.995 get zero weight, with a warning (count and largest propensity).
  - New `warnings` and `method` keys follow the output contract.
  - Cell by cell, the three estimators equal the `csdid`/DRDID port to 3e-11, with never-treated and not-yet-treated controls.
  - Without covariates the function is unchanged: the existing suite gives identical output line by line.
  - `did_timing_sensitivity` and `did_uncertain_dates` accept the same covariate options.
- **Independent recomputation built in (`verify=True`)**: `att_gt` with covariates also computes the other two estimators on the full sample (`res["cross_check"]`). It warns when they differ by more than one SE: the outcome and propensity models disagree, so one is misspecified. This is the dynare-mcp principle of solving the same model twice.
- **`causal.ccc_test`**: Karim & Webb's common causal covariates test, written from the method's description. The authors' `ccctest` licence forbids incorporating its code.
  - It uses untreated observations only: never-treated units, and treated units before adoption.
  - Stage 1 tests whether covariate slopes are equal in every group-period cell; stage 2 tests slopes equal across groups and across periods, with Holm adjustment.
  - It selects homogeneous, time, group or twoway, and explains what each implies for estimation (time: `att_gt` with covariates; group/twoway: intersection DiD, `DiDInt.jl`/`didintrjl`).
  - Two deliberate differences from `ccctest`:
    - **CR1 clustered covariances** for repeated units (its HC1 ignores serial dependence).
    - **Wild-bootstrap p-values imposed under each null, by default**, re-centred within cells. With 29 restrictions and heteroskedastic errors the asymptotic χ² rejected a true null 28% of the time (cells of 40 observations); the wild bootstrap 5%.
  - It falls back to the classical F statistic (bootstrapped) when restrictions exceed clusters.
  - Its HC1 and CR1 Wald statistics equal statsmodels' dummy regressions.
- **`causal.twfe_covariate_diagnostic`**: is a TWFE regression with covariates trustworthy? It reports three checks and a verdict:
  - the joint test of covariate levels in untreated first-difference trends (hidden linearity; the coefficient recovers the drift of the covariate effect, 0.39 for a true 0.4);
  - the share of treated observations with negative implicit TWFE weight (Frisch-Waugh);
  - the gap to `att_gt` with covariates, in SEs.

  It also warns that time-invariant covariates drop out of TWFE.
- **`simulation.demo_did_covariates`** (Teacher mode): the true ATT is 1. Unadjusted DiD is biased by 0.74, TWFE with the covariate by 0.97, and Callaway-Sant'Anna DR with the covariate in levels by 0.04. With `beta_drift=0` the three agree.
- **Reproducibility** (`reporting.py`):
  - `run_manifest` records data hashes (DataFrames by content, files byte by byte), Python, package and skill versions, seed, platform and headline results, optionally as JSON.
  - `cross_check` compares the same number computed by independent routes, within k SEs or to a relative tolerance.
  - CI checks that `reporting.SKILL_VERSION` matches this CHANGELOG.
- **Rules**:
  - DiD with covariates: levels, not TWFE; test CCC.
  - Headline numbers are recomputed by an independent route.
  - Deliverables end with a run manifest.
  - Robust Wald tests with many restrictions use a wild bootstrap.
  - New routing row, §6.0 in `causal-inference.md`, reviewer checklist items, Stata/R equivalents (`did`, `csdid`, `drdid`, `ccctest`, `DiDInt.jl`) and bibliography.
- **Fixes from the v2.2.3 eval**:
  - Scripts now start with `SKILL_DIR = os.environ.get("ECONOMETRICS_SKILL_DIR", "<path to your clone>/econometric-skills/econometrics")`. The default is the user's folder, never a sandbox path, not even in a comment. To run the script elsewhere, set the variable instead of editing the default. A script skeleton is in `interpretation-rules.md`.
  - The Basel traffic light and its 2.5 expected violations are stated for the 99% VaR. At 97.5% the expectation is 6.25, and FRTB's desk limits (12 at 99%, 30 at 97.5%) are mentioned.
- Two new blind-eval cases: DiD with covariates for a municipal credit program, and the review of a TWFE-with-controls report.
- **Worked example 06** (`examples/06_did_covariates_minimum_wage`): Callaway & Sant'Anna's county panel (`mpdta`, downloaded on first run). Unconditional ATT −0.0400 and DR ATT conditional on log population −0.0418, the published figures. The CCC test is not rejected (p = 0.29, 13 untreated cohort-year cells, clustered by county). All 12 group-time cells equal `csdid` to 5e-15. The TWFE diagnostic flags the absorbed covariate and 7% negative weights. The conclusion survives date errors of ±1 year, and the script ends with a run manifest.
- **Validated** in `tests/test_did_covariates.py` (44 checks):
  - the 2×2 estimators equal an independent statsmodels implementation, and `att_gt` equals `csdid` (now in `requirements-dev.txt`, so CI compares them on every push);
  - unbiasedness under conditional parallel trends (DR −0.02, reg −0.02, IPW −0.04, unadjusted +0.70);
  - double robustness: a wrong outcome model leaves DR at +0.007 while regression is at +0.56; a wrong propensity model leaves DR at −0.002 while IPW is at +0.17;
  - DR bootstrap CI coverage 92%;
  - `ccc_test` size and selection (time, group, twoway chosen 96–100%);
  - the TWFE diagnostic's three checks;
  - manifests and the new instructions.

  482 checks in total, on the minimum and the latest dependency versions.

## v2.2.4 — Claude Code plugin

- The repository is now a Claude Code plugin marketplace: `.claude-plugin/marketplace.json` lists the plugin `econometrics`, whose source is the existing `econometrics/` folder (the entry is the manifest and declares the folder itself as the skill), so nothing moved and `dist/econometrics.skill` is unchanged. Install with `/plugin install econometrics --marketplace FabianCornejo/econometric-skills`; `/plugin update` follows the version in the entry. Validated with `claude plugin validate` and an install on Claude Code 2.1.316: one skill, only the skill folder copied to the plugin cache. CI now fails if the plugin version differs from the latest CHANGELOG version, and the README documents the plugin install. This replaces the October `claude/plugin-packaging` branch, which moved the skill folder and is superseded.

## v2.2.3 — Trigger on method and planning questions without data

- Two more retests of the ES/FRTB eval: with the `market-risk` skill installed it was chosen again; with it removed, the model's reasoning showed it weighed loading this skill and skipped it, answering from general knowledge. The description required "numbers to be estimated or tested from data", which a method question with no data attached did not seem to meet (the evals that did load the skill asked for a number or a decision). The description now says to load the skill for any question on how to estimate, test, validate or choose a model, including advice and planning before any data exist, instead of answering from general knowledge or with finance or risk skills. Some method names were dropped from its list to stay within 1024 characters (1017); they remain in the routing table. No code changes.

## v2.2.2 — Description that leads with when to use the skill

- The v2.2.1 retest of the ES/FRTB eval failed in the same way: a corporate `market-risk` skill, whose name matches the question, was chosen and this skill was never loaded. The description listed about 500 characters of methods before the first risk word. It now starts with what the skill is (applied econometrics and quantitative risk, tested code) and when to use it, "instead of general finance, valuation or market-risk skills, which do not estimate", with VaR/ES backtests and validation (FRTB), tail risk, WACC betas and announcement returns first; the methods follow (1009 characters). No code changes.

## v2.2.1 — Fixes from the v2.2 blind evals

- **Control choice (eval passed with one gap):** the eval's graph omitted the edge treatment -> outcome, so the optimal set was silently unavailable. `dag.adjustment_report` now warns that such a graph assumes a zero effect (`warnings` key); `classify_controls` gives descendants of the outcome their own role (the bad control that raises R²); `specification.sensitivity` groups the dummies of a categorical or fixed effect into one joint partial R² benchmark (equal to the refit identity) instead of listing each year separately.
- **ES backtesting for FRTB (eval failed on delivery):** a corporate `market-risk` skill answered and this skill was not loaded, so no function or script appeared. The description and the coexistence rule now name market-risk skills, VaR/ES model validation and FRTB (description 1020 characters; Kalman, regimes, delta method and cointegration remain in the routing table). New `risk.dq_test` (Engle-Manganelli dynamic quantile), reported by `backtest_var` as `p_dq` (size 6%, power 97% against a static historical VaR on GARCH returns, where Christoffersen's independence test reaches 22%); `risk-var.md` §3.2 on FRTB model validation (hypothetical P&L, VaR tests at 97.5% and 99%, candidates, ES ≈ VaR 99% under normality, regulatory open points).
- **Power for a DiD (eval passed):** numbers reproduced exactly; the script's `SKILL_DIR` pointed at the repository root, so the delivery rule now says it must point to the folder containing `scripts/`. New `simulation.did_power_inputs` estimates `sd_unit`, `time_sd`, `sd_e` and a Nickell-corrected `rho_ar` from the historical panel (raw 0.55 → corrected 0.60 for a true 0.6); `causal-inference.md` §6.2 adds flow vs stock outcomes.
- 7 new checks (438 in total).

## v2.2 — Causal graphs, sensitivity for double ML, power for DiD, extreme value theory and ES backtests

- **New module `dag.py`** (Pearl; Cinelli, Forney & Pearl 2022): write the causal graph as a string (`"Z -> X; X -> Y; X <-> Y"`), get every minimal adjustment set by the backdoor criterion, the optimal set of Henckel, Perković & Maathuis (2022) (adds outcome predictors, drops instruments), the role of every variable with a recommendation (confounder, mediator, descendant/collider, M-bias collider, instrument-like, outcome predictor), the conditional independences the graph implies with partial-correlation tests on data (Holm), and a Mermaid diagram. A graph with an open unobserved confounder is reported as not identified by adjustment.
- **`causal.dml_sensitivity` and `causal.dml_benchmark`** (Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis 2022, "Long Story Short"): omitted-confounder bounds for double ML in terms of partial R² with the outcome and the treatment, confidence bounds through influence functions, robustness values and a bound table for contour plots; benchmarks refit without each observed covariate and report its implied strength and the change in theta. `dml_plr` now keeps its cross-fitted residuals.
- **DiD power in `simulation.py`**: analytic `mde_did` (McKenzie 2012) and `power_did_sim`/`mde_did_sim`, which simulate the planned panel (unit effects, common shocks, AR(1) errors, simultaneous or staggered adoption) and the inference to be used (TWFE with unit-clustered CR1 SEs and t(G−1)), reporting power, size and coverage.
- **Extreme value theory and ES backtests in `risk.py`**: `mean_excess`, `gpd_fit` (peaks over threshold with Smith 1987 SEs, tail VaR/ES with delta-method CIs, threshold-stability table), rolling `var_evt` (unconditional or McNeil-Frey conditional EVT); `es_backtest` (McNeil-Frey exceedance residuals with a bootstrap-t p-value, Acerbi-Szekely Z2 with a simulated p-value and the Basel-style zones), `fz_loss` and `compare_es_models` (FZ0 loss with Diebold-Mariano against the best model).
- **Rules and references**: choose controls with a graph; causal claims from observables carry a benchmarked sensitivity analysis; an insignificant effect needs its MDE; far tails use EVT and ES forecasts are backtested (`interpretation-rules.md`, `causal-inference.md` §2.1, §4.1, §6.2, `risk-var.md` §2.1, §3.1, workflow study card, reviewer checklist, teaching catalogue, Stata/R/dagitty/DoubleML/evir/esback equivalents, bibliography). Three new blind-eval cases (control choice, ES backtesting for FRTB, power for a DiD).
- **Validated** in `tests/test_dag.py` (18 checks: textbook graphs, OLS with a minimal set unbiased while butterfly bias, no controls and a collider are not, the O-set with the smallest SD, implications rejected when an edge is missing), `tests/test_dml_sensitivity.py` (12: the bound with the true confounder strength reproduces the actual bias within 5%; with linear learners the robustness value equals Cinelli-Hazlett's to four decimals; benchmarks equal OLS partial R²), `tests/test_power_did.py` (10: CR1 SE equal to statsmodels; simulated SD equal to McKenzie's formula; power falls from 80% to 51% under AR(1) ρ = 0.8) and `tests/test_evt_es.py` (18: Smith SE coverage 93%; t(4) tail index and 99.9% VaR/ES inside their CIs; ES backtest sizes 3.0–3.5% and power 89%; conditional EVT passes every VaR and ES test on GARCH-t returns and ranks first by FZ0). The plain McNeil-Frey t-test turned out to be conservative (≈2% size), so the skill uses a bootstrap-t. 431 checks in total, on the minimum and the latest dependency versions.

## v2.1 — Classical linear models, empirical finance, randomization inference and a simulation lab

- **New module `linear_models.py`** (Faraway; Dalpiaz): ANOVA types I–III with partial eta-squared, Tukey/Holm/BH pairwise comparisons, ANCOVA with a parallel-slopes test, Huber/bisquare robust regression, quantile regression with a bootstrap test of equal slopes, Freedman-Lane permutation test, pure-error lack-of-fit test, Box-Cox with exact profile-likelihood bounds, partial residuals, ridge/lasso/elastic net/PCR/PLS compared by nested CV, post-double-selection lasso (Belloni-Chernozhukov-Hansen) and best subsets.
- **New module `finance_empirics.py`** (Ang 2015; Tsay 2013; Campbell-Lo-MacKinlay): total and log returns, market model and CAPM with HAC SEs, rolling betas, Blume and Vasicek adjustments, Hamada unlever/relever with debt beta, factor models (Fama-French, NEFIN for Brazil), event studies with cross-sectional t, Patell, BMP and Corrado tests, performance ratios (Sharpe with Lo SE, Sortino, Treynor, IR, Jensen), Markowitz frontier with Ledoit-Wolf covariance. `references/empirical-finance.md` covers WACC betas from comparables, event-study design and Brazilian data (B3, CDI, NEFIN).
- **`causal.randomization_inference`**: Fisher randomization p-value and test-inversion CI for a DiD with few treated units (exact enumeration when assignments are few), for the TWFE statistic or any user statistic — the check that the v2.0 reviewer eval asked for.
- **New module `simulation.py`** for Teacher mode: seeded demos of omitted-variable bias, bad controls, measurement error, heteroskedasticity, spurious regression, weak instruments, few clusters and staggered DiD, a generic `monte_carlo` engine and `power_mde`.
- **Validated** in `tests/test_linear_models.py` (29 checks), `tests/test_finance_empirics.py` (32) and `tests/test_simulation_ri.py` (19): agreement with scipy/statsmodels, recovery of known parameters, Monte Carlo sizes and coverage (permutation test 6%, Box-Cox CI 97%, post-double-selection 95%, randomization inference 5.5% with 6 of 27 units treated), and the classic event-study result that Patell over-rejects under event-induced variance (31%) while BMP does not (4%). Along the way, the Box-Cox CI was fixed to use exact bounds instead of grid points, and the documented wild-bootstrap size with 8 clusters was corrected to ~7% from 1,000 replications.
- **Fixes from the v2.1 blind evals (before the tag):** the "deliver code, not only a plan" rule is now a visible rule in `SKILL.md` (a runnable script with the skill's functions, on simulated data when none is attached); when a valuation or cost-of-capital skill is also in use, this skill does the estimation; `unlever_beta`/`relever_beta` gain `method="harris_pringle"` (Hamada with debt beta now named as Conine 1980); new `finance_empirics.dimson_beta` for thinly traded stocks; `empirical-finance.md` covers windows after a change of control (privatizations), Vasicek, coherence between international betas and a global CAPM, and the return convention for CAARs. Running the event-study script produced by the eval exposed a crash when events lack history: `event_study` now accepts an `event_id` (repeated acquirers) and the new `car_regression` matches CARs to deal characteristics, drops skipped events and clusters on the rows used; scripts must use a `SKILL_DIR` variable instead of a sandbox path and be tested on awkward cases. 9 new checks (373 in total).

## v2.0 — Consultant, Teacher and Reviewer modes; estimation engine; heterogeneous effects

- **`SKILL.md` as a router** (214 → 111 lines): three modes (Consultant by default, Teacher, Reviewer), a routing table from the question to the reference and functions, the non-negotiable rules in one line each, and a shared output contract. The full workflow (now with a step 0 "study card": estimand, design, identifying assumption, effect modifiers X vs confounders W vs instruments Z) and the interpretation rules moved to `references/workflow.md` and `references/interpretation-rules.md`, unchanged in substance.
- **Teacher mode** (`references/teaching.md`): an eleven-block method card (intuition → model → assumptions → estimator → derivation → simulation → code → diagnostics → interpretation → pitfalls → exercise), three levels, a catalogue of simulations that show each assumption failing, and a curriculum order following Hill-Griffiths-Lim. **Reviewer mode** (`references/review-checklist.md`): a referee checklist and an opinion format with findings by severity.
- **New module `estimation.py`:** maximum likelihood from any per-observation log-likelihood (Newton-Raphson with step halving or BFGS; Hessian, OPG, sandwich and clustered covariances; convergence and gradient checks), LR/Wald/LM tests of linear restrictions, nonlinear least squares, delta method for any vector function of the parameters, pairs and cluster bootstrap, and the **restricted wild cluster bootstrap-t** with Webb weights and a test-inversion CI — the skill already recommended it for few clusters but had no function for it.
- **New module `hte.py`** (after EconML and grf, numpy/scikit-learn only): CATE by DR-learner and linear DML with valid CIs, a forest DR-learner and T/S/X meta-learners, all with cross-fitted nuisances and out-of-fold predictions; validation by the BLP/calibration test, GATES, CLAN (Holm-adjusted), TOC/AUTOC/Qini with bootstrap SEs and the R-score; exhaustive depth-1/2 policy trees on doubly robust scores, learned on one half and valued on the other, with the cost of treatment. `references/heterogeneous-effects.md` covers estimands, identification, choosing an estimator, reading each tool, policy learning and using EconML/grf predictions with these validation tools.
- **Validated** in `tests/test_estimation.py` (28 checks: Wooldridge Examples 17.1 probit and 17.3 Poisson by hand-written likelihood, equality with statsmodels for coefficients, Hessian and sandwich SEs, LR and Wald statistics; NLS = OLS for a linear f; delta method vs simulation; with 8 clusters, clustered SEs reject a true null ~20% of the time and the wild cluster bootstrap ~5–7%) and `tests/test_hte.py` (40 checks: recovery of tau(x) = 1 + x for binary and continuous treatments, ATE CI coverage 94% and BLP size 3.5% in Monte Carlo, no false heterogeneity under a constant effect, a linear CATE missing symmetric heterogeneity that the forest detects, a policy tree recovering "treat if x1 > 0"), on both the minimum and the latest dependency versions.

## v1.9 — Temporal holdout, calibration and categorical segments

- **Logit leaf model:** `time_col` makes the holdout the most recent period (cut at a distinct value, no period split), so drift shows up instead of being averaged away; `performance` adds the Brier score, calibration slope and mean bias, with a calibration table by decile per model (`segments.calibration` is also available on its own); `segment_by` accepts text or categorical columns (dummies created, rebuilt on prediction, unseen categories flagged); benchmarks that omit a regressor of the leaf model are flagged as not like for like.
- **Instructions:** cluster at the level where the regressor of interest varies (the judge for a judge's characteristic) in both the text and the code; never recommend a step without code that performs it.
- **Evals** (`evals/`): blind-test prompts with objective pass criteria, run in a fresh conversation before each release.
- **Validated** in `tests/test_llm_annotation_segments.py` (66 checks): holdout strictly later than training, drift detected as negative mean bias, calibration slope 0.99 for true probabilities and 0.50 for doubled logits, dummy splits recognized.

## v1.8 — Robustness of the v1.7 tools

- **`dsl_regression`:** checks that every regressor varies and the design has full rank among the hand-coded documents (and, for a logit, that both outcomes appear), naming the cause instead of failing with a singular matrix; the classic trigger is a draw seed shared with the data-generating code. A separated hand-coded logit no longer yields a spurious design effect near zero: the benchmark is reported as NaN with a warning and the DSL estimate is kept.
- **Logit leaf model:** `benchmarks` scores theory-based logits (e.g. interactions with customer type) on the same holdout, `res["best"]` names the model to report, `segment_by` warns when the tree never splits on the groups the user asked about (CART splits on outcome rates, not slopes), and `leaf_model_stability` reports split-variable frequency and model ranking across seeds.
- **Validated** in `tests/test_llm_annotation_segments.py` (55 checks), including a churn process whose slopes differ by customer type: the tree never splits on it and the interaction benchmark wins in every seed.

## v1.7 — LLM coding with valid inference and segmented logit

- **LLM coding of documents:** answers constrained to a closed label set with a probability per category (`llm_classify`, any backend that returns logits; `hf_backend` runs a Hugging Face model locally, so personal data stay on the machine), a flag for documents where the model wanted to answer outside the codebook, and token-level aleatoric/epistemic uncertainty (LogTokU, Ma et al. 2025).
- **Valid regressions on LLM-coded variables:** `dsl_regression` (OLS or logit, coded outcome and/or regressors) combines LLM codes for every document with a hand-coded subsample drawn with known probabilities (design-based supervised learning, Egami et al. 2023), with sandwich or clustered SEs; `dsl_sampling_probs` oversamples uncertain documents without breaking the design. In simulations with group-dependent coding errors, treating LLM codes as truth understates the effect by 13–32%; the corrected estimator is unbiased with nominal coverage.
- **Logit leaf model** (`segments.py`; De Caigny, Coussement & De Bock 2018): a shallow tree defines segments and a logit is fit in each, with honest sample splitting for valid leaf SEs, AMEs by segment, and holdout AUC and top-decile lift against a pooled logit and the tree.
- **Validated** in `tests/test_llm_annotation_segments.py` (40 checks: closed-form LogTokU, exact identities of the estimator, Monte Carlo bias and coverage, known two-segment process); `hf_backend` was also run end to end on Qwen2.5-0.5B-Instruct.

## v1.6 — Packaging, CI and examples

- **Worked examples** (`examples/`): five end-to-end analyses on Wooldridge data following the 7-step workflow — returns to education with sensitivity analysis, IV, panel FE/RE/CRE, distributed lags, Tobit and Poisson — as scripts and executed notebooks.
- **Missing data:** `data_prep.missing_report` (patterns, complete-case N, test of whether observables predict missingness) and guidance on listwise deletion, multiple imputation and what to avoid, verified by simulation.
- **Dependencies:** `requirements.txt` with lower bounds that the whole test suite was actually run on; `requirements-dev.txt` for tests and examples.
- **Continuous integration:** GitHub Actions runs every test suite and every example on minimum and latest dependency versions, and checks that `dist/econometrics.skill` matches the source.

## v1.5 — Specification and sensitivity tools

- **New module `specification.py`:** FGLS for heteroskedasticity of unknown form; confidence vs. prediction intervals (with Duan smearing for log-y models); long-run effects of distributed lags and ADL models with delta-method SEs; standardized (beta) coefficients; exact omitted-variable-bias decomposition; Oster (2019) coefficient stability; Cinelli-Hazlett (2020) robustness values with covariate benchmarks.
- **Diagnostics:** Mundlak/correlated-random-effects test (FE vs RE with clustered errors); Poisson overdispersion (Pearson dispersion, Cameron-Trivedi NB1/NB2 tests, excess zeros).
- **Validated** in `tests/test_specification_tools.py` against Wooldridge Examples 6.1, 6.5–6.6, 8.7, 10.4, 14.4 and 17.3, the Darfur example of Cinelli and Hazlett (2020), and known-truth simulations.

## v1.4 — Textbook replication and core fixes

- **Replication suite:** `tests/test_textbook_replication.py` checks the skill's functions and the recipes in its references against published results from Wooldridge's *Introductory Econometrics* (Examples 8.4, 13.3, 14.4, 15.1, 15.5, 15.7, 15.8, 17.2, 17.3), alongside the known-truth simulations.
- **Diagnostics:** Durbin-Watson and Breusch-Godfrey run only for time series (in a cross-section the row order is arbitrary); time series also get Engle's ARCH-LM. CLI flag `--ts`.
- **Tobit:** lower, upper or two-limit censoring; the three average marginal effects (total, intensive and extensive margins) with delta-method standard errors.
- **Stata parity:** documented `debiased=True` for textbook/Stata `small` IV standard errors.

## v1.2 — Causal inference, survival, text and data

- **Causal designs:** balance tables, propensity scores, IPW, doubly robust AIPW, nearest-neighbour matching, double/debiased ML, regression discontinuity (rdrobust/IK bandwidths, fuzzy RD, robustness package, density test), TWFE event studies, Callaway-Sant'Anna staggered DiD, synthetic control with placebo inference.
- **DiD timing robustness (v1.3):** anticipation in Callaway-Sant'Anna, automatic warnings when the outcome moves before the recorded date, a date-shift sensitivity table with a date-robust ATT, and estimation when each unit's adoption date is only known within a window (laws phased in court by court).
- **Survival:** Kaplan-Meier, log-rank, Cox with hazard ratios and clustered SEs, proportional-hazards test, discrete-time hazard data.
- **Text as data:** regex coding with audit snippets, TF-IDF, LDA, clustering with representative documents, classifiers with proper validation, agreement with hand coding, misclassification correction, similarity to a reference group.
- **Data and reporting:** robust readers for Brazilian files (`;`, latin-1, decimal comma, XML/HTML ".xls"), large CSVs, safe merges, PDF/OCR, polite scraping; outreg2/esttab-style regression tables.

## v1.1 — Financial econometrics

- **Forecasting:** auto-ARIMA without pmdarima, ARIMAX with properly lagged regressors, walk-forward VAR forecasts, naive/seasonal/historical-mean benchmarks, Diebold-Mariano and Pesaran-Timmermann tests.
- **Risk:** VaR and Expected Shortfall (historical, normal/t, EWMA, filtered historical simulation, rolling GARCH) with Kupiec, Christoffersen and Basel traffic-light backtests.
- **Dependence:** rank correlations, tail dependence, copulas (Gaussian, Student-t, Clayton, Gumbel, Frank and rotations) by maximum likelihood, copula simulation, mutual information on a correlation scale, distance correlation.
- **State space and regimes:** Kalman-filter time-varying regression with ML-estimated noise, pairs-trading layer, Markov-switching models, HMMs with filtered (real-time) probabilities, K-means regimes.
- **Backtesting:** one-bar-lag convention, transaction costs, performance table, walk-forward splits, time-series momentum, Sharpe confidence intervals.

## v1.0 — Initial release

A rigorous applied-econometrics workflow in Python (statsmodels, linearmodels, arch): regression with full diagnostics, endogeneity and instrumental variables, panel data, logit/probit/tobit, time series (unit roots, structural breaks, ARIMA, VAR/SVAR, cointegration/VECM), volatility (GARCH), forecasting with out-of-sample evaluation, and demand and price-elasticity estimation, with data from FRED, the World Bank, IMF, OECD and Brazil's BCB, IBGE and Ipeadata.
