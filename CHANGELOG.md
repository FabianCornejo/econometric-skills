# Changelog

All releases of the `econometrics` skill, newest first. Each release is tagged on GitHub with the packaged `econometrics.skill` attached ([releases](https://github.com/FabianCornejo/econometric-skills/releases)).

## v2.5.0.2 — The skill is reached when the skill listing is crowded

Origin: the v2.5.0.1 validation on a second computer. Its skill listing held 865 skills, only 14 with a description; econometrics was listed by name only, and a difference-in-differences request with one treated store was answered without the skill (a loose `pip install`, p < 0.001 from a HAC error with one treated unit). Everything v2.5.0.1 added (the doctor, the single question) runs only after the skill loads. Claude Code drops descriptions starting with the skills invoked least, so a new or rarely used skill is the first to lose its description; the listing also grows by itself when Claude Code syncs the plugins of the user's claude.ai account (15 plugins arrived at one session start on that computer).

- **Session-start note** (`econometrics/hooks/hooks.json`, the plugin's first hook): one `SessionStart` command, on startup, `/clear` and compaction (not on resume, assumed to replay the saved note; not verified), that echoes one line of 394 characters: statistical work (hypothesis tests, confidence intervals, sample size, regression, the effect of a promotion or policy, forecasts from statistical models, VaR or ES) goes to `econometrics:econometrics`, even when the listing shows only its name; the note does not apply to anything else, such as budgets, valuations or legal risk. A plain `echo` of a single-quoted line, the same in bash and PowerShell, with no Python; 10-second timeout. Measured on Claude Code 2.1.319 (Windows) with the new name-only arm (855 skills, 11 with a description, econometrics by name only in every name-only run), 2 runs per request: the skill was used in 2 of 8 runs over 4 statistics requests without the note and in 8 of 8 with it; no false trigger over 5 unrelated requests (a cash budget and a legal-risk review among them) in 10 name-only and 10 clean runs. Runs of one request are not independent, so these counts are the evidence, not a p-value. A session-start reminder tried in v2.4.x did not help (0 of 4); it was measured in another setup, and the reason for the difference was not established. The app package (`dist`) leaves `hooks/` out, as the app runs no plugin hooks; CI checks it.
- **`tools/routing_evals.py --arm nameonly`** (and `--arm all`): about 850 decoy skills over 12 decoy plugins, so that Claude Code keeps only a few descriptions, as on the user's computer. The existing crowded arm (48 decoys) never checked whether the skill lost its description and could not reproduce the failure. The listing is not in the trace `claude plugin eval` gives graders but is in each run's own session transcript, so this arm runs with `--keep-temp`, reads the first `skill_listing` of each run, records whether it was captured (a positive control) and whether econometrics was listed by name only, counts routing only over the runs where both hold (or says the arm did not reproduce the failure), saves `nameonly-diagnostics.json` and removes the kept run directories. `--no-hooks` measures the plugin without its note in the same commit.
- **New routing cases**: `two-stores-promo`, the request of the second computer's test, word for word; `wacc-beta-valuation`, a beta for the WACC of a valuation (statistical work the note must not exclude as "valuation"); and two boundary negatives, `cash-budget` and `legal-risk-review`, that the note must leave alone.
- **README**: why the skill can be listed by name only (least-invoked first), the session-start note, its measurement and how to turn it off, and the duplicate name a copy in the Claude app adds.

## v2.5.0.1 — Installation and updates, plug-and-play

The skill now checks the whole setup before the first analysis on a machine and asks the user once for everything that is missing or out of date. Origin: setting up a second computer found the plugin and the Claude-app copy nine versions behind with nothing saying so, plugin auto-update off, two Pythons with the packages split between them, about 300 skills in Claude Code's listing, and optional packages with no permitted way to install them.

- **`setup_env.py --doctor`** (new): one report that installs, writes and changes nothing, ending with a numbered "To do" list for a single question to the user. It never crashes: each probe that fails is reported as "not checked". It reports:
  - the Python to use, and other interpreters on the machine whose core packages are present at the minimum versions (so it does not reinstall them in the wrong one); the probe of another interpreter runs from the temporary folder, so a module in the user's folder cannot shadow it;
  - missing core packages and every optional extra;
  - each loaded copy of the skill (plugin, copy synced from the Claude app, manual copy, or the clone it runs from) against the latest GitHub release, with the update route for that kind of copy (plugin commands and a restart; Customize → Skills for the app copy; replace or `git pull` otherwise). When an update is already installed but the session still runs the old version, it says to restart Claude Code. It makes one request to the public releases endpoint with a short timeout; offline it says "not checked"; `--offline` skips it. Copies are compared by `SKILL_VERSION`, not by file hash: the Claude app repackages an uploaded `.skill` file. Removed copies in `.trash` are ignored, and the marketplace checkout is not a loaded copy;
  - plugin auto-update for the `econometric-skills` marketplace, read in the order Claude Code documents (settings `extraKnownMarketplaces`, then `known_marketplaces.json`, then the default, off for third-party marketplaces), including the environment variables that turn it off;
  - a crowded skill listing (50 skills or more: the routing evals overflowed Claude Code's listing with the plugin plus 48 decoy skills, about 35,000 characters of descriptions), offering a clean analysis folder (`workspace.py`).
  - `--json` prints the report (ASCII-escaped) for the agent. Output is safe on Windows consoles whose code page lacks a character of a user path.
  - External commands (`claude plugin list --json`, `py -0p`, other interpreters) run with a timeout that kills the whole process tree, writing to a temporary file instead of a pipe, so a wrapper such as `claude.cmd` cannot hang the doctor (on Windows the whole process tree is killed with `taskkill /T`; elsewhere the direct child). On Windows the other Pythons are started through the `py` launcher by tag (`-V:3.14`), so a user folder with accents never has to be decoded from its output.
- **`setup_env.py --enable-autoupdate`** (new), with the user's permission only: writes `"autoUpdate": true` on the marketplace's `extraKnownMarketplaces` entry in Claude Code's `settings.json`.
  - It keeps the marketplace's own source (from settings or from `known_marketplaces.json`), so a fork or a local folder is never switched to GitHub. It refuses a marketplace Claude Code does not know.
  - It keeps every other key, backs the file up first (never overwriting an earlier backup), writes atomically, keeps the file's permissions and a symlinked `settings.json` (it writes the target), accepts a UTF-8 BOM, and refuses a file that is not a JSON object, leaving it untouched.
  - It notes when an environment variable keeps auto-update off. The other documented way is `/plugin` → Marketplaces → econometric-skills → Enable auto-update; there is no shell command for it.
- **Optional extras for every optional package** (`--extra NAME`): `rd` (rdrobust), `regimes` (hmmlearn, ruptures), `panel` (pydynpd), `ml` (econml), `pdf` (PyMuPDF), `ocr` (easyocr), `web` (requests), `text` (sentence-transformers), `llm` (transformers, torch), `data-br` (python-bcb, sidrapy, ipeadatapy), `data-intl` (wbgapi, fredapi, yfinance, pandas-datareader), plus `excel` and `plots`.
  - Large extras (`ml` ~50 MB, `pdf` ~81 MB, `llm` and `text` ~160 MB, `ocr` ~205 MB, measured on Windows on top of the core packages) are asked in a question of their own, with the size, only when a function needs them.
  - Minimums are the lowest wheels that install together with the core minimums (all support Python 3.10). CI's minimum job pins every one of them, and a test imports each one there; the PDF reader is also exercised. `ocr`, `text` and `llm` (torch-based) are not installed in CI, and their minimums are documented, not tested.
  - The `plots` minimum rises from 3.8 to 3.9.2: matplotlib 3.8 requires numpy < 2, below the skill's numpy 2.1. `hmmlearn` goes to 0.3.3: 0.3.2 installs and imports with numpy 2 but fits the regimes wrongly (the minimum CI job caught it). `econml` stays at 0.16 (0.17 needs scikit-learn 1.6).
  - Packages that pip cannot install on the running Python are skipped by `--install` with the reason (ruptures has no Python 3.14 release); hmmlearn has no Python 3.14 wheel and is flagged as needing a compiler.
  - A function that needs an extra now also checks its minimum version: an older copy (hmmlearn 0.3.0, sentence-transformers 2.x, transformers below 4.40) stops with the install line instead of running untested.
  - `require_extra(..., only=...)`: a function that needs part of an extra checks only that part (`hmm_regimes` needs hmmlearn, not ruptures).
- **No more bare pip commands**: `regimes.hmm_regimes`, `data_prep.pdf_text` (OCR only for pages without a text layer), `data_prep.polite_get`, `text_data.embed` and `text_data.hf_backend` stop with the `setup_env.py --install --extra …` line; the references point to the extras.
- **`causal.select_bandwidth` no longer switches silently**: without rdrobust it still uses Imbens-Kalyanaraman, but says why (not installed, with the exact install command; failed to import; or failed), and `rdd` carries it as a warning to relay.
- **SKILL.md section 7**: `--doctor` once per machine (Consultant and Reviewer before estimating, Teacher before the first runnable example); one question covering every item that applies; nothing installed, updated or changed before the answer; auto-update only on a yes; with no Python at all, the system installer goes in the question. Section 4 is unchanged (a test pins its SHA-256). The `econometrician` agent runs `--doctor` and reports its list instead of acting on it.
- **CI**: a job on Python 3.14 with the latest dependencies (the latest job already runs pandas 3.1). The extras CI tests are installed in every job, except ruptures on 3.14.

## v2.5.0 — DiD: design, imputation (BJS) and pre-trend tests

Difference-in-differences that starts from the design: what kind of treatment it is, which estimators fit it, a second route with a different method for staggered adoption, and pre-trend tests whose reference distribution is corrected for an estimated covariance (their remaining over-rejection with few clusters is measured and warned). Source: the guide of de Chaisemartin & D'Haultfœuille, "Which DiD estimator should I use?" (Credible Answers). The new estimators are replicated against R.

- **`causal.did_design`** (new): the first step of any DiD. From the treatment column (binary or a dose), it classifies the panel into the guide's five designs (canonical, staggered, heterogeneous adoption, first switch, other). It reports:
  - reversals, stayers and never-treated units;
  - units treated from the first period;
  - cohorts and their pre-periods;
  - balance, irregular periods, and assignment below the cluster.

  A treatment with two values {0, c} is classified as binary, with a warning to rescale it. A switch observed across a gap in the panel is flagged as having an uncertain date. The "other" design never returns a `gvar`.

  It returns the estimators that fit and the comparison group (not-yet-treated for staggered adoption, as the guide recommends). It returns a `gvar` only for a binary absorbing treatment: a reversible treatment or a dose never becomes a first-treated date.
- **`causal.did_imputation`** (new): the imputation estimator of Borusyak, Jaravel & Spiess (2024).
  - Unit and period effects are fitted on untreated observations only, and Y(0) is imputed for treated ones.
  - It returns the ATT and the event study, with the paper's clustered SEs (Theorem 3).
  - Unbalanced panels are allowed. Observations that cannot be imputed are dropped and counted.
  - The pre-trend test uses untreated observations only and leaves out leads that the fixed effects span.
  - With `anticipation`, the periods g − δ … g − 1 stay out of the fixed-effects fit and are reported as their own event-study rows, not averaged into the ATT, which covers t ≥ g as in `att_gt`.
  - In simulation, the mean Theorem 3 SE equals the Monte Carlo SD of (estimate − sample ATT): ratio 1.00 over 200 panels.
  - Matches R `didimputation` 0.5.1: estimates within 2e-7, SEs within 1e-8 (event-study SEs within 1e-7).
- **`causal.did_compare`** (new): Callaway-Sant'Anna against the imputation estimator. Both run without covariates, and `control` defaults to `"notyet"`; pass the headline `att_gt`'s control.
  - The comparison uses only the post-period cells both define, with the same weights.
  - The difference is tested at alpha with a paired bootstrap. A significant gap is a finding that points to the pre-trends.
  - Power against pre-trends is modest (43% against a linear pre-trend of 0.2 per period in the tests), so it does not replace the pre-trend tests.
- **`causal.twfe_weights`** (new): the weights of de Chaisemartin & D'Haultfœuille (2020) behind a TWFE coefficient, for a binary or a non-binary treatment and an unbalanced panel.
  - It reports the negative weights and `sigma_fe`.
  - Matches R `TwoWayFEWeights` 2.1.0 to 1e-9.
- **`causal.att_gt`**
  - **Joint pre-trend test** (`pretrend`, `pretrend_p`, `pretrend_df`): a Wald test with the bootstrap covariance and an F(q, n − q) reference, n = the smaller of the resampled units (or clusters) and the draws. It uses the most detailed level with q ≤ (n − 1)/10: every pre cell, the event-time averages, or their mean. A coarser level is warned.
  - Size of a 5% test in simulation: 7.5% with 100 to 300 units, 5–7% with 30 to 80 clusters, about 9% with 8 clusters (warned). With the χ² reference it was 13% with 100 units and 9 cells, and 83% with 8 clusters and 8 restrictions.
  - **`alpha`** for the CIs, the early-effect warning and the pre-trend test, also passed by `did_timing_sensitivity` and `did_uncertain_dates`.
  - **Fixes:**
    - with no never-treated unit and `control="never"`, it raised a pandas error; now it says what is missing;
    - a bootstrap draw in which no cell is defined (no treated cluster, or no control) is now left out of the SEs and counted in a warning; this is a guard, not the fix of an observed crash.
  - Outputs on designs that ran before are unchanged: bit-identical in 7 configurations.
- **`volatility.garch_fit`**: `half_life_ci` from the limits of the persistence CI, with an infinite upper limit when that CI reaches 1. A persistence ≤ 0 gives a half-life of 0 (before: infinite). The EGARCH half-life is labelled as that of the log-variance.
- **Rules**: `interpretation-rules.md` gains six rules (see its v2.5.0 section). `causal-inference.md` §6 gains a table of the five designs with what the skill covers in each.
- **`SKILL.md` §4**: the DiD rule starts with `did_design`. Cross-cutting rules that lived only in `interpretation-rules.md` are now in §4, shortened, with their content unchanged:
  - one alpha per analysis;
  - declare the family of tests;
  - many-restriction Wald tests;
  - the tolerance of an agreement and agreement compatible with zero;
  - relay verdicts as printed.
- **Evals**: two find-the-error routing cases (`review-twfe-staggered`, `review-reversible-treatment`).
- **Tests**: `tests/test_v250.py` (81 checks). It covers:
  - hand-built panels for each design;
  - a closed form in the canonical design;
  - a TWFE sign reversal;
  - coverage, size and power by simulation;
  - a hand example of the weights;
  - the identity TWFE = Σ w·Δ on noise-free panels;
  - the R replication (`tools/r_reference/`, fixed panel `tests/data/did_v250_panel.csv`).

## v2.4.7.2 — Pending items of v2.4.7.1: one alpha, persistence CI, skewed-t GJR

Pending items of v2.4.7.1 and of the second GARCH validation (Ibovespa, 2026-11-06). No estimator changes its point estimates except the GJR persistence with skewed-t errors.

- **`volatility.garch_fit`**
  - **Persistence with uncertainty**: `persistence_se` and `persistence_ci` (level 1 − alpha) by the delta method with the parameter covariance (arch's, or the QML sandwich when there are regressors in the variance). A CI that reaches 1 gets a warning: an integrated variance is not ruled out. In 200 simulated GJR samples (t(6) errors, persistence 0.95, n = 2000) the 95% CI covered the true value in 94%.
  - **GJR with skewed-t errors**: the persistence is α + κγ + β with κ = E[z²·1(z<0)] of the fitted Hansen distribution (`neg_weight`), not γ/2, which holds only for symmetric errors. On the Ibovespa the difference was about 0.001; with λ = −0.36 and η ≈ 6 it is about 0.12·γ. κ is checked against a numerical integral of the density.
  - **No more `ConvergenceWarning` on stderr**: arch's `fit(show_warning=True)` installs its own "always" filter, which overrode the skill's "ignore"; both fits now pass `show_warning=False`. Convergence is still read from the convergence flag and reported.
- **One alpha per analysis**
  - **`reporting.cross_check`**: new `alpha` (default 0.05); the "compatible with zero" threshold is z = Φ⁻¹(1 − alpha/2) instead of a fixed 1.96.
  - **`diagnostics.stationarity`** (and `stationarity_many`): new `alpha` for the joint ADF/DF-GLS/KPSS verdict. A KPSS p-value at the lower table bound (≤ 0.01) counts as a rejection at any alpha ≥ 0.01, and one at the upper bound (≥ 0.10) never does; an alpha outside [0.01, 0.10], where the KPSS table is exact, gets a warning.
  - **`multivariate.var_analysis`**: its `alpha` reaches the joint verdict, and the ADF-only note uses it instead of a fixed 0.10. This changes the output at the default alpha: a series with ADF p between 0.05 and 0.10 and a joint verdict I(0) now gets the ADF-only note.
  - **`multivariate.vecm_fit`**: in the full-rank case the unit-root cross-check uses the joint verdict at `alpha` (before: ADF alone at a fixed 0.10), and series stationary only around a trend that the deterministic case lacks are flagged.
- **Rules**: report the persistence with its CI; κ, not γ/2, with skewed errors; one alpha per analysis (`interpretation-rules.md`, `volatility.md`, `time-series.md`).
- **Tests**: `tests/test_v2472.py` (27 checks): closed-form SEs, CI coverage by simulation, κ and its gradient by numerical integration of Hansen's density written in the test, boundary cases for alpha, the Selic series of the v2.4.7 validation and a forced arch failure with the warnings recorded.

## v2.4.7.1 — Fixes from the v2.4.7 validation

Three fixes found by the first real-data tests of v2.4.7 (a VAR of IPCA, Selic and the exchange rate; a GARCH of the Ibovespa).

- **`volatility.garch_fit`**: a symmetric GARCH is now tested for omitted asymmetry by the LR of the GJR(1,1,1) with the same mean and distribution (`asymmetry_lr`), and a rejection is flagged. The Engle-Ng sign-bias tests stay as a complement. On the Ibovespa the sign-bias joint p was 0.77 while the LR was 31.5 (p = 2e-8); in simulations with t(5) errors and γ = 0.08 the LR rejected in every sample and the sign-bias tests in about one in ten, with no false rejections of the LR on symmetric data. If the GJR fit fails, the LR is reported as unavailable (not as "no asymmetry") and the sign-bias tests remain the check.
- **`multivariate.var_analysis`**: the unit-root warning follows the joint ADF/DF-GLS/KPSS verdict of `diagnostics.stationarity` (`integration_order` in the output). When only the ADF fails to reject, as for the Selic 2010–2026 (ADF p = 0.27, joint verdict I(0)), the series is reported as persistent, with Toda-Yamamoto as the safe Granger check. A series stationary only around a trend, in a VAR without a trend, gets a warning to add it.
- **`reporting.cross_check`**: when the routes agree and all are compatible with zero, the agreement is flagged as uninformative (`null`), so a VAR and a local projection that both cannot tell the effect from zero are no longer reported as a confirmation. `null` and `weak` no longer fire together.
- **Rules**: a sign-bias test that does not reject does not show symmetry; agreement between routes compatible with zero is not a confirmation, and only the same quantity is compared; unit-root warnings follow the joint verdict (`interpretation-rules.md`, `volatility.md`, `time-series.md`).
- **Python 3.10 and 3.11**: `multivariate.py` (v2.4.7) used a nested f-string that only Python 3.12 accepts, so the module failed to import on 3.10 and 3.11; fixed, and CI now compiles every script on Python 3.10.
- **Tests**: `tests/test_v2471.py` (17 checks), with the Selic series of the validation, seeded GJR simulations (power, size and recovery of γ) and a forced GJR failure.

## v2.4.7 — Multivariate time series, volatility, systems and dynamic panels

Three new modules, plus Fama-MacBeth, with the checks that decide whether an estimate can be read.

- **`scripts/multivariate.py`**
  - **`var_analysis`**: lag order by four criteria, stability (companion eigenvalues), residual whiteness and normality, and the Granger matrix. It warns on parameters per observation and on unit roots in levels.
  - **`irf_bands`**: impulse responses with Kilian (1998) bias-corrected bootstrap bands, recursive in a chosen ordering or Blanchard-Quah. Draws that stay non-stationary after the correction are drawn again, and their count is reported. In 30 persistent VARs (φ = 0.9, T = 80), the 90% bands covered the true response at least as often as plain bands.
  - **`fevd_table`**: the decomposition under several orderings; shares that move by more than 10 points are flagged.
  - **`svar_identify`**: the order condition and a numerical rank condition with a relative tolerance, so the verdict does not depend on units. A model can pass the count and still fail the rank.
  - **`svar_longrun`** (Blanchard-Quah) and **`svar_sign`**: sign restrictions by QR rotations, with the acceptance share and the Fry-Pagan median-target model.
  - **`johansen_trace`**: the trace test for the stated deterministic case ('ci', 'co', 'li' or 'n'), with critical values simulated for it.
    - **Why:** statsmodels has no restricted-constant case. With its unrestricted constant, two cointegrated random walks without drift were declared stationary in 9 of 20 samples. The simulated 'ci' test rejected the true rank in 5% of 300.
    - **Case 'co':** simulated with the drift measured in the data; it is slightly oversized (11% at 5% with T = 250).
    - **Agreement with statsmodels:** the eigenvalues match for every case ('li' corresponds to statsmodels' 'coli').
  - **`vecm_fit`**: chooses the lags and the rank, and fits the same deterministic case it tested.
    - **Stability:** it checks that the implied levels VAR has exactly k − r unit roots and the rest inside the unit circle. eig(I + β'α) is not enough with lagged differences; the tests include a stable VECM where that check fails.
    - **Weak exogeneity:** a joint Wald test per variable.
    - **Readings:** α signs are read only with one relation. With more relations, single loadings depend on the normalisation.
- **`scripts/volatility.py`**
  - **`stylized_facts`**: fat tails, Ljung-Box on r and r², ARCH-LM and leverage correlations.
  - **`garch_fit`**: GARCH, GJR or EGARCH, with persistence, half-life, the unconditional variance, Ljung-Box on z and z², sign bias and AIC/BIC.
  - **`variance_dummies`**: regressors in the variance equation (day of week, events), estimated by the module's own likelihood with Bollerslev-Wooldridge SEs. Without regressors it reproduces `arch` to 3-4 decimals.
  - **`sign_bias_test`**: the Engle-Ng tests.
- **`scripts/systems.py`**
  - **`sur_system`**: SUR with the Breusch-Pagan LM test. It says when SUR equals OLS (identical regressors).
  - **`three_sls`**: the order condition per equation. It refuses an unidentified equation, and a variable that is endogenous in one equation but listed as exogenous in another. It reports 3SLS next to 2SLS, and the first-stage F. It says when 3SLS equals 2SLS (exact identification).
  - **`dynamic_panel`**: Arellano-Bond difference GMM or Blundell-Bond system GMM, one-step robust and two-step with Windmeijer SEs. It reports AR(1)/AR(2), Hansen J, the instrument count against the number of groups (with collapse), and the within/OLS bounds for ρ. When the instruments start at lag 3, the level instrument moves to Δy_t−2.
    - **Agreement with linearmodels:** IV2SLS and two-step IVGMM on the stacked differences give the same estimates.
    - **Simulations:** both estimators are centred on the true ρ. With ρ = 0.9, system GMM was far less dispersed than difference GMM. AR(2) rejects when the errors are MA(1).
    - **Level equation:** exogenous regressors enter it through Δx, because x in levels is usually correlated with the fixed effect.
- **`finance_empirics.fama_macbeth`**: risk premia from characteristics or first-pass betas, with Newey-West SEs and the Shanken correction (applied to the Newey-West variance) for estimated betas. In 60 simulated samples the Shanken interval covered the true premium more often than the uncorrected one.
- **References**:
  - `time-series.md` §7–9, `volatility.md`, `panel-iv.md` §5 and §11, `empirical-finance.md`;
  - eight interpretation rules for v2.4.7 (dynamic-panel checks, Cholesky ordering, Granger, sign restrictions, VECM, SUR/3SLS equivalences, GARCH names, Fama-MacBeth SEs);
  - also updated: the curriculum map, `software-equivalents.md` and the bibliography.
- **Independent review before release:** it found three high-severity issues in the first draft, all fixed and tested.
  - The 'li' case was fitted without a constant.
  - The 'co' critical values ignored the drift.
  - α readings changed with the column order when r > 1.
  - Also fixed: the stability check with lagged differences, the system-GMM level instrument with later lag limits, an endogenous variable reused as an instrument in 3SLS, the unit-dependent rank tolerance, Kilian draws replaced by the generating VAR, and the convergence flag and mean argument of the variance-regressor GARCH.
- New `tests/test_v247.py` (53 checks; 965 in total).

## v2.4.6.2 — Fixes from the v2.4.6.1 validation (IPCA, second run)

The same IPCA request, run again on v2.4.6.1, confirmed the five fixes:
- the short-memory check gave p = 0.126;
- the CUSUM with the long-run SD gave p = 0.34;
- the edge warning fired on the 2022 break;
- the weak-agreement warning fired, and the agent explained it to the user;
- the answer reported the bias, the 91.7% coverage and the sensitivity to the window.

It also found three small issues, fixed here:

- **`ts_univariate.sup_wald` with X that already contains a constant.** The agent passed all 12 month dummies; the function added its own constant, which made the design rank deficient and the Chow F use 13 restrictions instead of 12. The bootstrap p-value was unaffected (0.028), but the reported sup-F was wrong (3.89 instead of 4.24). The function now drops its constant when X spans it, says so, and refuses other collinear designs; the ranks are computed on column-normalised matrices, so a regressor in large units (nominal GDP) does not make the constant look redundant.
- **`_console`: logs written to a file are UTF-8.** On Windows, `py script.py > log.txt` wrote cp1252: arrows and ≥ became "?", and an editor showed —, ± and • as "�". The agent also stripped the accents from its Portuguese output. When stdout or stderr is a regular file (`>` in cmd or in Git Bash, the shell Claude Code uses on Windows), it is now switched to UTF-8, keeping its error handler; pipes and consoles keep the previous behaviour, and so does PowerShell, which redirects through a pipe.
- **Language.** One progress message came out in Spanish in a Portuguese session; the final answer was in Portuguese. `SKILL.md` now says the user's language applies to progress messages too, and to the accents in scripts and logs.
- New `tests/test_v2462.py` (10 checks; 909 in total).

## v2.4.6.1 — Fixes from the first real-series test (monthly IPCA)

The v2.4.6 functions were tested in Claude Code on the monthly IPCA (BCB/SGS 433, 2000–2026) with `/econometrics:consultant`. The analysis was sound, and every headline number was reproduced independently. In several places, though, the skill relied on the agent to notice a problem, and the independent review of this patch found that one of the test's findings was itself an artefact.

- **`ts_univariate.long_memory`**: it read d = 0.285 (local Whittle) as "stationary long memory", although ordinary AR(1) persistence (φ ≈ 0.58) produces a d that large in about 12% of samples; the agent had to run that Monte Carlo by hand.
  - **The check:** when 0 < d < 0.5, the function now fits a short-memory AR(p) by BIC, simulates it, and reports the p-value of the observed d. It reads long memory only when short memory is rejected. It never overrides a non-stationary reading, and it skips series whose AR is near a unit root.
  - **How it performs (simulations):** it is conservative: about 1 in 100 AR(1) series with φ = 0.6 was read as long memory. Its power is modest: about 25% for d = 0.35 with n = 320, 40–50% with n = 1,000, and about 70% for d = 0.3 with n = 2,000.
- **`ts_univariate.sup_wald`**:
  - **OLS-CUSUM:** it is now scaled by the long-run SD of the errors, taken from the sieve AR. With autocorrelated errors the iid version rejects stable series most of the time: about 60% for AR(0.6) at n = 320, against 2.5% for the new version. On the IPCA mean, the "CUSUM rejects, p = 0.0025" seen in the test was this artefact. With the long-run SD, p = 0.34. Both statistics are reported. The verdict says when a valid CUSUM rejects and the sup-F does not.
  - **Window edge:** it warns when the estimated date is at the edge of the trimmed window, as the 2022 break in the IPCA AR(1) was, and points to an end-of-sample test.
  - **Unit-root warning:** the warning that y may have a unit root comes from Zivot-Andrews, which allows one break, when there are no regressors. In 30 simulations it fired for 97% of random walks and for none of the stationary series with a break.
  - **Sieve:** it stays fitted to the residuals under no break. Fitting it to the residuals of the model with the break was tried and rejected in review: the size under AR(0.8) errors rose to 12% at n = 150 and 24% at n = 60. With the null sieve it was about 4%.
- **`reporting.cross_check`**: it flags an agreement as weak when the tolerance is more than a quarter of an estimate that is clearly away from zero, and always prints the spread. On the IPCA, two forecasts "agreed" within an SE of 2.1 points on 5.3%. Disagreements and statistically null estimates are not flagged.
- **`setup_env`**: new optional extra `plots` (matplotlib). Following `SKILL.md`, a delivered script asks for it in the same install question, or says that its chart is optional instead of promising a PNG it cannot write.
- **References**: `time-series.md`, `workflow.md` and three interpretation rules for v2.4.6.1.
- New `tests/test_v2461.py` (27 checks, with the IPCA series as a fixture; 899 in total).

## v2.4.6 — Univariate time series

A new module for one series, four Teacher-mode demos, and the sample size for a given effect.

- **`scripts/ts_univariate.py`**:
  - `ts_identify`: correlogram with Bartlett bands for the ACF and ±1.96/√n for the PACF, Ljung-Box, the orders the correlogram suggests and a BIC grid of small ARMA models.
  - `detrend` (settings from the frequency; daily or weekly data need them passed explicitly): polynomial trend, HP (λ from the frequency; two-sided or one-sided, the real-time version), Hamilton, Baxter-King, Christiano-Fitzgerald. It reports what each filter loses or revises at the end of the sample; for the two-sided HP, the average gap between the real-time and the final trend.
  - `adf_lags` and `adf_bootstrap`: ADF with lags by AIC (default), BIC, the general-to-specific t ≥ 1.645 rule or the modified AIC, all on a common sample, and a sieve-bootstrap p-value under H0 next to MacKinnon's. AIC is the default because, in simulations with the bootstrap (n = 150), it kept the size (6%; 2.5% with MA(−0.5) errors) with full power against AR(0.5), while BIC over-rejected with MA errors (10%) and the t rule and MAIC lost a third of the power.
  - `ar_bootstrap_forecast`: AR(p) forecasts with bootstrap intervals that re-estimate the coefficients (parameter uncertainty), with residuals rescaled for the degrees of freedom and Kilian's bias correction of the slopes, next to the analytic normal interval. In simulations (AR(1), φ = 0.8, n = 30, h = 4) the 95% interval covered 92% of outcomes; the analytic interval and the uncorrected bootstrap covered 86%. Forecast labels continue the series' dates, periods or years.
  - `sup_wald`: Andrews sup-F test for one break at an unknown date, with a sieve-bootstrap p-value (AR order by BIC on a common sample, so it does not depend on the units of y), plus OLS-CUSUM and recursive coefficients. It warns when the residuals are as persistent as a random walk, because a break test in levels assumes stationarity.
  - Long memory: `long_memory` (GPH and local Whittle d with SEs), `frac_diff`, two-step `arfima`.
  - `combine_forecasts`: equal weights (default), inverse MSE, Bates-Granger (with the matrix of mean squared errors, so a biased forecast is penalised) and Granger-Ramanathan, estimated on a training window and evaluated after it.
- **Demos** (`simulation.py`):
  - `demo_trend_vs_difference`: detrending a random walk vs differencing a trend-stationary series.
  - `demo_ar1_forecast_variance`: the h-step variance of an AR(1) and of a random walk.
  - `demo_unit_root_power`: the ADF against φ from 1 to 0.8.
  - `demo_break_unit_root`: a mean shift that makes a stationary series look I(1), and the ADF with the break modelled.
- **`inference.n_for_power(effect, sd)`**: the smallest n whose exact t-test power reaches the target, for the effect asked. In the v2.4.5 Windows test the agent quoted 26 per group, read off a 0.8-SD power table, for an observed difference of d ≈ 0.94; the right number is 19. A new interpretation rule says to compute n for the effect, never read it off a table.
- **References**:
  - `time-series.md` §2–5 and §11 point to the new functions.
  - `forecasting-backtesting.md` gains Armstrong's forecasting checklist (be conservative, simple first, damp trends, combine, validate out of sample, honest intervals, disclose).
  - `workflow.md` sets the order for a series: clean it, look for breaks (on the differences if it may be I(1)), then test for unit roots with the break modelled.
  - Also updated: the interpretation rules for v2.4.6, the curriculum map and simulation table in `teaching.md`, `software-equivalents.md` and the bibliography.
- An independent review before release found and fixed: Bates-Granger weights that used the centred covariance, a sieve BIC that depended on the units of y, bootstrap forecast intervals narrower than claimed, `sup_wald` failing on an X array with dated y, daily data given quarterly settings, a negative d read as "no long memory", and future labels that restarted at the row count.
- New `tests/test_v246.py`: every estimator against statsmodels or a closed form written in the test (ADF τ, AIC lags, AutoReg forecasts and SEs, hpfilter, Hamilton by OLS, Chow F at the estimated date, Bates-Granger and constrained-LS weights, `TTestIndPower`). The bootstrap procedures are checked for size, power and coverage, and long memory by recovering d from an ARFIMA built by its MA(∞) weights (66 checks; 872 in total).

## v2.4.5 — Quick and conceptual questions reach the skill; deliverables stay in the working folder

Measured with the routing evals of v2.4.4. No change to estimators or results.

- **Descriptions** (`SKILL.md` and the `econometrics:statistics` entry point). The v2.4.4 baseline showed requests that never reached the skill, even with the plugin alone: a quick interval for a proportion and the review of a "not significant, so no effect" conclusion (0%), and a lesson (50%). Claude answered them from general knowledge. The descriptions now ask for the skill before answering any question that tests, estimates or judges numbers, "even a quick one you could do by hand (an interval for a proportion, a sample size, a p-value) or a conceptual one", and to review a conclusion, report or paper, and they say why: the tested functions and rules catch errors of answers from memory (the statistics entry names the Wald interval). The main description stays within 1,024 characters ("Applied", "LLM coding" and EVT were dropped from its method list; VaR/ES and documents-into-variables requests are still named or routed by the `SKILL.md` table).
- **Result** (2 runs per case and setting, Claude Code 2.1.318, same harness as the baseline): the skill was reached in **100%** of the 8 positive cases with the plugin alone (baseline 62%) and **100% in the crowded listing** (baseline 50%; the two-groups comparison went from 0% to 100% there). The crowded setting is over Claude Code's listing budget (debug log: "69 skills, 45,907 chars > 30,000 budget — descriptions will be truncated"); the clean one is not. **No false triggers**: two new unrelated requests, plain arithmetic (a 15% tip) and a spreadsheet formula (SUMIF), join the e-mail; none fired the skill in either setting. A clean-listing check with 3 runs on the five changed cases gave the same 100% / 0%.
- **Deliverables in the working folder.** In the v2.4.4 test on Windows the `econometrician` agent computed every number right but saved its script and run manifest in its scratchpad, a temporary folder Claude Code offers for intermediate files. The agent and `SKILL.md` now say that a delivered script and its manifest are the product, not scratch files: they go to the working folder even when a scratchpad is offered, and a system or read-only folder means asking where to save.
- `tools/routing_evals.py`: `--arm clean|crowded` to measure one setting, and any case with a `skill-not-fired` grader counts as an unrelated request, reported as a false-trigger rate.
- New `tests/test_v245.py` (13 checks): the save rule in the agent and `SKILL.md`; the descriptions' triggers, reasons and length; the three unrelated requests and their graders; the false-trigger summary (805 checks in total).

## v2.4.4 — A dedicated agent and routing evals in a crowded skill listing

Layers 4 and 5 of the robustness plan: a way into the skill that does not depend on the skill listing, and a repeatable measurement of how often Claude reaches the skill, with the plugin alone and in a crowded installation. No change to estimators or results.

- **`econometrics:econometrician` agent** (`econometrics/agents/econometrician.md`): a plugin subagent with the skill preloaded (`skills: [econometrics:econometrics]`), invoked with `@agent-econometrics:econometrician <task>` or by delegation. It works in its own context and returns the main table, the diagnostics and the paths of the scripts it wrote. Checked live: spawned through the Agent tool it starts with the skill's text in context; started as the main session with `claude --agent` it does not (skills are preloaded only into subagents), so its first instruction is to load the skill when the text is missing.
- **Routing evals** (`econometrics/evals/routing/`, nine cases for `claude plugin eval`): two groups, a quick proportion interval, controls with a post-treatment variable, a weak instrument, VaR validation, a null result to review, a lesson in English, a larger analysis where delegation to the agent counts, and an unrelated e-mail that must not trigger the skill. Graders: the skill (or the agent) was used, and a cheap regex on the answer; no judge model, no shell tools, so the suite runs on native Windows.
- **`tools/routing_evals.py`**: `claude plugin eval` loads only the plugins a case names, so a crowded installation cannot be reproduced with the plugin alone. The script builds a harness where every case runs twice, with the plugin alone and with 48 decoy data and statistics skills (about 35,000 characters of descriptions, more than the 37,000-character listing that dropped descriptions on the notebook in v2.4.0 once our own skills are added), and prints the routing rate in each setting. `--build-only` writes the harness without model calls.
- **Baseline** (2 runs per case and setting, Claude Code 2.1.318, about $4.60): the skill was reached in 62% of the positive cases with the plugin alone and 50% in the crowded listing. Always reached: controls, weak instrument (crowded), VaR validation, the larger analysis. Never reached in either setting: the quick proportion interval and the review of a null result, which Claude answered from general knowledge (both answers passed the content check). The two-groups comparison went from 100% to 0% in the crowded listing, the failure first seen on the notebook. The unrelated e-mail never triggered the skill. These numbers are the reference for future description changes; `/econometrics:*` commands and the agent are the ways in that do not depend on them.
- CI and packaging: `dist/econometrics.skill` (the app upload) excludes `agents/` and `evals/` too; eval results are git-ignored.
- **Fix found by the Windows CI job** (`regimes.markov_switching`): statsmodels draws the random starting values of `search_reps` from fresh OS entropy (0.15) or NumPy's global generator (0.14), so the fit changed from run to run and could fail with "SVD did not converge" or "Steady-state probabilities could not be constructed". The search now runs under `seed=0` (as `rng`, or as the global seed on 0.14, whose state is restored), is retried with the next seeds and then without the random search when a start fails, and the result records `fit_attempt`.
- New `tests/test_v244.py` (26 checks): the agent's fields, preload and instructions; the routing cases and their graders (the skill pattern matches every entry point and no other plugin's skill; the negative case is scored in both arms); the harness built without model calls (both settings, the plugin copy's manifest equal to the marketplace entry, 48 valid decoys over 30,000 characters); the summary; the Markov-switching fit identical whatever the global random state, and retried after a failure; packaging and README (792 checks in total).

## v2.4.3 — English command names; answers that read well in a terminal

From the v2.4.2 validation on Windows, where `/econometrics:professor multicolinearidade` produced a correct lesson (numbers from `demo_multicollinearity`, checked against the closed-form formulas) with three problems in how it was delivered. No change to estimators or results.

- **Commands renamed to English**, matching the modes in `SKILL.md` and the rest of the repository: `/econometrics:consultant`, `/econometrics:teacher` and `/econometrics:reviewer` (were `consultor`, `professor`, `revisor`). Answers still come in the user's language.
- **Terminal rendering**: in Claude Code, formulas in plain text with Unicode symbols, not LaTeX, and no HTML (the lesson printed `$$\operatorname{Var}...$$` and a `<details>` block literally). LaTeX stays for the app, notebooks and documents. Rule in `SKILL.md` and a *Rendering* paragraph in the method card (`teaching.md` §2).
- **Where scripts are saved**: in Claude Code, in the user's working folder, with the full path; when that is a system folder (Claude Code had been started in `C:\Windows\System32`) or not writable, ask, instead of using a temporary folder the system may delete.
- **Found in the same test, outside the skill**: another plugin's PreToolUse hook (ECC's "Fact-Forcing Gate") denied the first shell command and file write. A folder prepared with `scripts/workspace.py` keeps such hooks out of the analysis; README already explains it.
- New `tests/test_v243.py` (10 checks): command names equal the `SKILL.md` modes, no Portuguese command names left, the rendering and save-location rules (766 checks in total).

## v2.4.2 — Classical regression and discrete choice; three commands and a clean workspace

Second step of the v2.4 plan, plus the first three layers of the robustness plan for crowded Claude Code installations (the v2.4.0 and v2.4.1 validations found other plugins hiding the skill's description and running broken hooks in every session). No change to existing estimators or results.

- **`specification.linear_restriction_test`**: H0: Rβ = r from a formula and a constraint string, by the SSR form with the restricted least-squares coefficients, the R² form shown only when valid, the robust Wald F with `cov_type="HC3"`, and for one restriction θ = Rβ − r with SE, t (t² = F) and CI. Reproduces Wooldridge Example 4.9 (F = 9.55).
- **`specification.prais_winsten`**: iterated Prais-Winsten and Cochrane-Orcutt for AR(1) errors, OLS vs FGLS side by side, DW before and after, warnings for ρ̂ near 1 and the strict-exogeneity requirement. Reproduces Wooldridge Example 12.4 (ρ̂ = 0.293) and equals exact GLS with the AR(1) covariance.
- **`specification.nonnested_test`**: Davidson-MacKinnon J test in both directions and the encompassing F test, with the four possible readings.
- **New `scripts/discrete_choice.py`**: `binary_effects` (LPM with robust SEs, probit and logit side by side; average marginal effects per variable, computed by changing the variable in the data so that polynomials, interactions and every dummy of a factor move together, with the discrete change for 0/1 variables, each factor level against the base and delta-method SEs; odds ratios with the exact change in the odds, LR tests, pseudo-R², LPM fitted values outside [0, 1], classification at the base-rate cutoff against the majority-class accuracy, AUC; reproduces Wooldridge Example 17.1) and `heckman` (two-step with Heckman's corrected SEs, ρ and σ, comparison with OLS on the selected sample, warnings without an exclusion restriction; reproduces Wooldridge Example 17.5 and Stata's `heckman, twostep`).
- **`diagnostics.iv_report`**: 2SLS beside OLS, first-stage partial F and partial R² labelled weak / acceptable (report Anderson-Rubin) / strong, the endogeneity test, the overidentification test only when overidentified, and the Anderson-Rubin confidence set for one endogenous regressor with its shape (bounded, unbounded, disjoint). Reproduces Wooldridge Examples 15.1 and 15.7.
- **`finance_empirics.market_model`**: HAC tests of α = 0 and β = 1 (`beta_null=`) and their joint Wald test in `result["tests"]`.
- **Teaching demos**: `demo_simultaneity` (OLS mixes supply and demand; ILS = 2SLS with a demand shifter), `demo_lpm_bounds` (the LPM slope tracks the average marginal effect while fitted values leave [0, 1]), `demo_multicollinearity` (unbiased, SE × √VIF, joint F still rejects).
- **References and rules**: `ols-regression.md` §4, §7, §11; `panel-iv.md` §3-5; `limited-dependent.md` §3, §6; `empirical-finance.md`; "Rules added in v2.4.2" in `interpretation-rules.md` (SSR form when y changes, binary effects on the probability and classification against the base rate, Anderson-Rubin with weak instruments, Heckman needs an exclusion restriction, FGLS needs strict exogeneity, J tests can reject both or neither); reviewer checks; Stata equivalents; bibliography; the curriculum map points to the new functions.
- **Commands that always reach the skill** (Claude Code): `/econometrics:consultor`, `/econometrics:professor` and `/econometrics:revisor` (`econometrics/modes/`, declared in `marketplace.json`) load the main skill in the chosen mode. They are user-invoked only (`disable-model-invocation`), so they work however crowded the skill listing is and do not compete with other skills. The app package keeps one skill, and CI checks that.
- **A clean analysis folder**: `scripts/workspace.py <folder>` writes a project `.claude/settings.json` that turns off, in that folder only, every enabled plugin except econometrics (`--keep`, `--dry-run`); other folders are unchanged.
- **CI on Windows**: a new job runs the whole suite and the worked examples on `windows-latest` from a checkout path with a space in it, the conditions that broke other plugins on the user's machine.
- **Independent review before the PR** (a reviewer who had not seen the code, deriving each formula and probing edge cases) found five defects, all fixed and now tested: the robust first-stage statistic of `iv_report` was linearmodels' Wald χ² compared with F thresholds (with three instruments, a weak first stage, F = 4.1, showed as 12.3, "acceptable"); `binary_effects` took statsmodels' marginal effects column by column, so a three-level factor flipped one dummy with the others held (0.187 instead of 0.224) and x with x² ignored the square; `prais_winsten` quasi-differenced across rows dropped for missing values and across shuffled rows without a word (it now refuses, and takes `time=`); `nonnested_test` counted a variable written two ways (`C(d)` and `d`) twice in the encompassing test; an IV formula without a constant was compared with an OLS with one. Also: Heckman's exclusion check by the data rather than by names, positional rows against duplicate indexes, R² of the restricted model on the same (centred or uncentred) definition, and the Anderson-Rubin set reported as its pieces.
- New `tests/test_v242.py` (63 checks against textbook results, Stata output, statsmodels/linearmodels/sklearn, exact GLS, finite differences, a bootstrap of the Heckman procedure and Anderson-Rubin by hand; plus the commands, the workspace helper and the Windows job): 756 checks in total.

## v2.4.1 — Statistics questions find the skill more often

The v2.4.0 validation on Windows asked for a two-group comparison, a 2x2 table and a normality check. With many plugins installed, Claude Code's skill listing went over its budget (962 skills, 37,000 characters), the descriptions were dropped and the skill was offered by name only: the agent wrote its own analysis, without the inference module, and suggested a loose `pip install`. With one plugin turned off in that folder (564 skills, descriptions back), the same request loaded the skill first and used `inference.py` correctly. No change to estimators or results.

- **The description names the statistical tasks** (describing data, hypothesis tests, two groups or means with t/Welch/paired/Mann-Whitney, proportions, variances, chi-square and Fisher, normality, correlation, power and sample size), still within 1024 characters. Trimmed: quantile regression, CATE, bootstrap and survival, which are reached through other words ("for whom an effect is larger", "time-to-event").
- **A second entry point, `econometrics:statistics`** (`econometrics/statistics/SKILL.md`, declared in `marketplace.json`). It only sends the agent to the main skill and to `inference.py`, so a statistics request has a skill whose name matches. Claude Code only: `dist/econometrics.skill` (the app upload) keeps one skill, and CI checks that.
- **README**: why Claude may not pick the skill (listing over budget, names only) and the three ways out: turn off unneeded plugins in the analysis folder, raise `skillListingBudgetFraction`, or name the skill.
- **Measured on the notebook** (`claude -p`, the installed 2.4.0 against this version, 2 runs each, plugin `ecc` on): the full request used the skill in both versions; a two-class comparison from a file used it with 2.4.0 and, with descriptions dropped, not with this version (once it chose `data:statistical-analysis`), while with the budget raised to 3% this version used it both times through `econometrics:statistics`; a confidence interval for 47/120 with a sample size was answered by hand, with no tool, in every setup. A SessionStart hook reminding the model of the skill did not help (0 of 4) and was not shipped. The lasting fix for a crowded listing is on the user's side (fewer plugins in the folder, or a larger budget).
- New `tests/test_v241.py` (17 checks): the description's statistical terms and length, the entry point's header, name, paths and functions, the plugin declaration, the one-skill app package and the README (692 checks in total).

## v2.4.0 — Statistics foundations: Module 0 and the inference building blocks

First step of the v2.4 plan (fundamentals before new analytical capabilities). A review of the Insper MPE course material (Statistics, Econometrics, Time Series), FEA-USP Microeconometrics, Wooldridge and four other texts found that the skill had no layer below regression: no functions for one- and two-sample tests, variances, proportions, goodness of fit, normality, contingency tables or rank tests, a Teacher-mode curriculum that started at regression, and two lessons in the simulation table ("Unbiasedness", "CI coverage") with no function behind them. No change to existing estimators or results.

- **New `scripts/inference.py`** (14 functions, every one with a verdict that never accepts H0): `describe_dist` (skewness, kurtosis in both conventions, with Excel's KURT named as excess kurtosis, Tukey outliers), `one_sample` and `two_sample` (Welch by default, pooled on request with a Brown-Forsythe warning, paired; Cohen's d and Hedges' g), `variance_test` (chi-square, F, Brown-Forsythe, Bartlett, with a heavy-tails warning), `proportion_ci` (Wilson by default, side by side with Clopper-Pearson, Agresti-Coull, Wald and the conservative interval), `proportion_test` (exact binomial, score z, pooled z with Newcombe's CI, Fisher), `n_for_proportion`, `chisq_fit` (df = k - a - 1, pooling of small cells), `normality` (Shapiro-Wilk, Lilliefors, Jarque-Bera, D'Agostino, QQ data), `contingency` (chi-square, Cramer's V, adjusted residuals, Fisher and odds ratio for 2x2; with strata, Mantel-Haenszel, Breslow-Day and a Simpson's-paradox flag), `nonparametric` (Wilcoxon and Mann-Whitney with Hodges-Lehmann estimates and distribution-free CIs, rank-biserial correlation), `correlation_test` (Pearson, Spearman, Kendall with Fisher-z CIs; warns when Pearson and Spearman disagree), `power_curve` (noncentral t and z), `adjust_pvalues` (Bonferroni, Holm, BH, BY).
- **Teaching demos** in `simulation.py`: `demo_clt`, `demo_ci_coverage` (t vs z with an estimated SD, skewed data, Wald vs Wilson), `demo_estimator_mse` (S^2 vs the MLE; method of moments vs MLE for a uniform), `demo_multiple_testing` (family-wise error and false discovery rate).
- **New `references/statistics-foundations.md`** (Module 0): probability and Bayes, distributions, sampling distributions and the CLT, estimators (moments, ML, Fisher information, conjugate normal), intervals, test logic and power, a test-choice table, goodness of fit, categorical data, multiple testing, an exploratory checklist, traps found in real course material and the bridge to regression.
- **Routing and rules**: a `SKILL.md` route for one- and two-variable questions; a non-negotiable rule that not rejecting H0 is not evidence for it; `describe_dist` in the audit step; "Rules added in v2.4.0" in `interpretation-rules.md` (Welch by default, Wilson for proportions, variance tests need normal data, df of goodness-of-fit tests, two kurtosis conventions, stratify before reading a crude table, exact percentages for odds ratios and log dummies, declare the family of tests); new reviewer checks; Stata/R equivalents; bibliography.
- **Teacher mode**: Module 0 opens the curriculum; the simulation table gains four lessons; a **curriculum map** (`teaching.md` §6) links about 45 topics of the statistics, econometrics and time-series sequence to functions and reference sections.
- New `tests/test_v240.py` (81 checks against scipy and statsmodels where the skill computes the formula itself, textbook results such as the horse-kick Poisson fit and the kidney-stone Simpson's paradox, and simulated coverage) and `tests/test_curriculum_map.py` (every function and section in the map exists; every inference function and simulation demo is reachable): 675 checks in total.

## v2.3.8 — Delivered scripts skip removed skills; B=0 is not a convergence problem

Two small fixes found in the v2.3.7 validation on Windows. No change to estimators or results.

- **The script skeleton skips Claude Code's trash.** Removing the skill in the Claude app moves the copy synced to Claude Code into `~/.claude/skills/.trash/<id>/econometrics`. The locator searched every `~/.claude/skills/**/econometrics`, newest first, so a user with only the app copy could run a removed version, or an older one trashed after a newer upload (a plugin install was not affected: its copy comes first). The locator now skips hidden folders under `~/.claude/skills`. Scripts delivered before v2.3.8 keep the old locator; regenerate them if the skill was removed from the app. README too.
- **`quantile_regression` with `B=0`** said "an end tau or the bootstrap did not converge". It now says that no bootstrap was asked for and the equality test was not computed; convergence warnings appear only when there was a bootstrap.
- New `tests/test_v238.py`: an active synced copy beats a newer trashed one; only a trashed copy gives the one-line "skill not found"; `B=0` gives the plain message and keeps the taus' convergence; `B>0` stays valid (586 checks in total).

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
