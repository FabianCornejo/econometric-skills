# Validation diary

How each public function of the skill is checked, what it is compared with, how close it has to be, and what is not checked. One row per public function of `econometrics/scripts/` (the setup and workspace scripts are not estimators and are left out). Started in v2.6.7; every new function enters this table in the same pull request.

Check it is in sync with the code and the tests: `python tests/test_validation_md.py`.

## Levels

- **A, replicated against a fixed reference.** The test compares the function with a published value (textbook with the example number, a published table), with the output of other software stored in the repository (R, Stata, DoubleML, lifelines) or run in the test (csdid, statsmodels, scipy), or with a closed form computed in the test by another route. A number the function returns about its own output (a "theory" column next to a simulated one) does not count as a reference.
- **B, known truth.** Simulated or constructed data whose answer is known by construction (a planted effect, a size or coverage target, a graph solved by hand).
- **C, execution only.** The test calls the function and checks shape, keys, warnings or messages, not a value.
- **D, no test.** No test file refers to the function (by `module.function` or by importing it).

A function gets the level of the strongest test that refers to it by name. A function exercised only inside another one keeps its own level (the note says where it is exercised).

## Tolerance rule (from v2.6.7)

The tolerance depends on the reference:

- a published, rounded value (textbook, article): half a unit of the last digit printed in the source (Wooldridge with three decimals: |Δ| ≤ 0.0005);
- the output of other software at full precision (R, Stata, DoubleML, statsmodels, lifelines): relative ≤ 1e-6, unless the algorithms differ (optimizers, sample splits, bootstrap), and then the test says why;
- a closed form computed in the test by another route: relative ≤ 1e-10;
- a known truth: a band fixed before running, in Monte Carlo standard errors, with at least 200 replications for size and coverage and 1000 in the cells at the limits a function accepts.

The *Tolerance* column is the one the test uses, read from the test. Many tests written before v2.6.7 are looser than this rule, some because they use numpy's `isclose`/`allclose` defaults (relative 1e-5, absolute 1e-8); their rows say so in the note, and they are tightened when their module is revisited. The same holds for size and coverage checks with fewer than 200 replications.

## Largest observed difference

For the level-A rows whose reference is outside the Python scientific stack the skill is built on (published values, and R, Stata, DoubleML, csdid, lifelines), `tools/validation_diffs.py` redoes the test's comparison and records the largest difference and the tolerance of that comparison, written as *difference of tolerance*. Measured on 2026-11-13 with Python 3.13.18, numpy 2.5.4, scipy 1.18.1, statsmodels 0.15.0, linearmodels 7.1, scikit-learn 1.9.1; output in `tools/validation_diffs.json`. It reads data, designs and reference values from the tests themselves, and CI runs it with `--check` (every comparison within its tolerance) on the latest dependency versions. Rows compared with software run in the test (scipy, statsmodels, linearmodels, arch, pandas) or with a closed form computed in the test are not measured.

## Summary

289 public functions: level A 88, level B 142, level C 21, level D 38.

| module | A | B | C | D |
|---|---|---|---|---|
| `backtest` | 0 | 3 | 1 | 6 |
| `causal` | 8 | 19 | 2 | 0 |
| `dag` | 1 | 7 | 0 | 0 |
| `data_prep` | 0 | 7 | 1 | 3 |
| `dependence` | 0 | 5 | 1 | 5 |
| `diagnostics` | 9 | 1 | 0 | 11 |
| `discrete_choice` | 2 | 0 | 0 | 0 |
| `estimation` | 4 | 2 | 0 | 0 |
| `finance_empirics` | 9 | 4 | 1 | 0 |
| `forecasting` | 0 | 7 | 1 | 1 |
| `hte` | 0 | 10 | 0 | 1 |
| `inference` | 15 | 0 | 0 | 0 |
| `linear_models` | 1 | 11 | 0 | 0 |
| `multivariate` | 4 | 4 | 0 | 0 |
| `regimes` | 0 | 2 | 4 | 2 |
| `reporting` | 0 | 3 | 1 | 2 |
| `risk` | 3 | 12 | 1 | 2 |
| `segments` | 1 | 3 | 1 | 0 |
| `simulation` | 2 | 23 | 0 | 1 |
| `specification` | 12 | 0 | 0 | 0 |
| `state_space` | 0 | 3 | 3 | 1 |
| `survival` | 1 | 4 | 0 | 0 |
| `systems` | 2 | 1 | 0 | 0 |
| `text_data` | 4 | 9 | 4 | 2 |
| `ts_univariate` | 8 | 2 | 0 | 0 |
| `volatility` | 2 | 0 | 0 | 1 |

## What this validation does not cover

- The text that `show=True` prints is tested for the sensitivity functions (v2.5.3) and for 20 functions with interpretation sentences (`tests/test_show_text.py`, v2.6.7); the other functions with `show` are not.
- Only the designs in the tests. A function can be right on them and wrong elsewhere.
- Dependency versions outside the CI jobs (minimum versions on Python 3.12, latest on 3.13 and 3.14, Windows).
- Rules of thumb in warnings and verdicts (thresholds such as F > 10 or 1.2x excess zeros) are checked for what they print on each side of the threshold, not for being the right threshold.
- Several Wooldridge examples are checked against numbers without the edition stated in the test (noted per row); `tests/test_v242.py` says 2e. Numbering can differ between editions.
- Level D functions have no test; level C functions have no check of their value.
- Public classes are not rows: `dag.DAG` is exercised through the `dag` functions, which take it or build it.

## `backtest`

| function | level | reference | tolerance | largest observed difference | test | note |
|---|---|---|---|---|---|---|
| `backtest.simple_returns` | D | — | — | — | — |  |
| `backtest.log_returns` | D | — | — | — | — |  |
| `backtest.run_backtest` | B | simulated series with a planted look-ahead signal | mean strategy return < 0.002 with the lag; > 0.007 without | — | test_financial_modules.py: backtest lag kills look-ahead |  |
| `backtest.performance` | C | — | — | — | test_financial_modules.py: performance table | only the keys of the table are checked |
| `backtest.compare` | D | — | — | — | — |  |
| `backtest.walk_forward_splits` | B | split structure known by construction | exact (no overlap, 5 splits) | — | test_financial_modules.py: walk-forward splits never overlap |  |
| `backtest.tsmom_position` | B | trending simulated series | cumulative strategy return > 0.2 | — | test_financial_modules.py: TSMOM profits on a trending series | sign check, not a value |
| `backtest.forecast_position` | D | — | — | — | — |  |
| `backtest.buy_and_hold` | D | — | — | — | — |  |
| `backtest.bootstrap_sharpe_ci` | D | — | — | — | — |  |

## `causal`

| function | level | reference | tolerance | largest observed difference | test | note |
|---|---|---|---|---|---|---|
| `causal.balance_table` | B | simulated confounding with known imbalance | max \|SMD\| > 0.3 before, < 0.1 after weighting | — | test_causal_text_survival.py: IPW weights balance covariates |  |
| `causal.propensity_score` | C | — | — | — | test_causal_text_survival.py: IPW recovers ATE≈2 | exercised through ipw/aipw; its own output is not compared |
| `causal.ipw` | B | simulated ATE = 2 | \|estimate - 2\| < 0.2 | — | test_causal_text_survival.py: IPW recovers ATE≈2 |  |
| `causal.aipw` | B | simulated ATE = 2 | CI covers 2 | — | test_causal_text_survival.py: AIPW recovers ATE≈2 |  |
| `causal.nn_match` | B | simulated ATT = 2 (homogeneous effect) | \|estimate - 2\| < 0.25 | — | test_causal_text_survival.py: NN matching ATT≈2 |  |
| `causal.dml_plr` | B | simulated theta = 2; omitted-confounder bias by covariance algebra | \|theta - 2\| < 0.2; bias bound within 5% of the actual bias | — | test_causal_text_survival.py: DML-PLR theta≈2; test_dml_sensitivity.py: bound with the true cf_y, cf_d | numbers also frozen against v2.5.2.1 (tests/data/v253_refs_7c25644.json, a regression anchor, not an external reference) |
| `causal.dml_sensitivity` | A | closed form S·sqrt(cf_y·cf_d/(1−cf_d)) (Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis 2022); Cinelli-Hazlett OLS RV for linear learners | relative 1e-12 (closed form); 0.005 absolute against the OLS RV | not measured (software run in the test) | test_v261.py: rho = 1: bias bound = S·sqrt(cf_y·cf_d/(1 - cf_d)); test_dml_sensitivity.py: linear learners: RV equals Cinelli-Hazlett OLS RV | the ρ = 1 closed form uses the S the function returns; the independent route is the Cinelli-Hazlett OLS RV |
| `causal.dml_benchmark` | A | DoubleML 0.11.4 sensitivity_benchmark (tests/data/doubleml_benchmark_v262.json) | cf_y, cf_d within 2% relative (3% for x3), rho equal, delta_theta within 0.01 | 0.00022 of 0.01 | test_v262.py: DoubleML 0.11.4 | wider than the D2 rule (1e-6) because the two runs use different sample splits |
| `causal.dml_bounds` | B | planted confounder with known (kd, ky) | near end of the bounds within 0.03 of theta at k*; equal to dml_sensitivity by hand within 1e-12 at k = 1 | — | test_v262.py: planted confounder; test_v262.py: k = 1 with the benchmark's rho equals dml_sensitivity called by hand |  |
| `causal.ik_bandwidth` | B | simulated sharp RD, jump 0.8 | \|estimate - 0.8\| < 0.12 | — | test_causal_text_survival.py: IK bandwidth gives a sensible estimate too |  |
| `causal.select_bandwidth` | C | — | — | — | test_v2501.py: select_bandwidth without rdrobust | checks the method chosen and its label (rdrobust mserd or IK fallback), not the bandwidth value |
| `causal.rdd` | B | simulated sharp RD, jump 0.8; fuzzy LATE 1 | CI covers 0.8; \|LATE - 1\| < 0.3 | — | test_causal_text_survival.py: sharp RD recovers 0.8; test_show_text.py: rdd |  |
| `causal.density_test` | B | simulated bunching at the cutoff | p < 0.01 with bunching; p > 0.05 without | — | test_causal_text_survival.py: density test detects bunching |  |
| `causal.rdd_robustness` | B | simulated RD without effects at placebo cutoffs | placebo p > 0.01; density p > 0.05 | — | test_causal_text_survival.py: placebo cutoffs ≈ 0 |  |
| `causal.did_design` | B | planted adoption dates and designs | exact (design, control group, gvar) | — | test_v250.py: did_design |  |
| `causal.event_study` | B | simulated staggered DiD, effect 1 at k = 0 | \|coef(k=0) - 1\| < 0.25; pre-trend p > 0.05 | — | test_causal_text_survival.py: event study k=0 ≈ 1; test_show_text.py: event_study |  |
| `causal.att_gt` | A | csdid (DRDID port) cell by cell, 3 methods x 2 control groups | max \|difference\| < 1e-8 | 3.1e-11 of 1e-08 | test_did_covariates.py: att_gt with covariates equals csdid (DRDID) cell by cell | skipped when csdid is not installed (it is in requirements-dev.txt); known-truth checks in test_causal_text_survival.py |
| `causal.did_timing_sensitivity` | B | misdated simulated adoption, true ATT 2 | \|ATT - 2\| < 0.25 | — | test_causal_text_survival.py: misdated: date-robust ATT |  |
| `causal.did_uncertain_dates` | B | simulated adoption windows, true ATT 2 | \|ATT - 2\| < 0.25 | — | test_causal_text_survival.py: uncertain dates |  |
| `causal.did_imputation` | A | R didimputation 0.5.1 (tools/r_reference/did_v250_reference.R -> output/did_v250_reference.json) | estimate within 1e-6, SE within 1e-8 (event-study SEs 1e-7) | 9.9e-09 of 1e-08 | test_v250.py: R didimputation, static ATT | the state-clustered SE sits at 99% of its tolerance (about 1e-7 relative); also closed-form DiD of means in the canonical design (1e-10) |
| `causal.did_compare` | B | simulated panels with and without a linear pre-trend | difference significant in <= 12% of 80 panels without pre-trend; >= 70% of 30 with | — | test_v250.py: did_compare | below the 200-panel rule of the guide for size statements; measured before that rule |
| `causal.ccc_test` | A | statsmodels cell-dummy regression (HC1) and cluster Wald (CR1), computed another way | relative 1e-6 | not measured (software run in the test) | test_did_covariates.py: ccc stage 1 HC1 Wald = statsmodels cell-dummy regression | size of the wild bootstrap <= 9% in 120 reps (below the 200-rep rule) |
| `causal.bacon_decomposition` | A | TWFE coefficient by OLS (the weighted 2x2s must add up to it) | 1e-10 | not measured (software run in the test) | test_did_v231.py: Bacon: weighted 2x2s equal the TWFE coefficient exactly |  |
| `causal.twfe_weights` | A | R TwoWayFEWeights 2.1.0 (tools/r_reference/output/did_v250_reference.json) | 1e-9 | 3.6e-15 of 1e-09 | test_v250.py: R TwoWayFEWeights | also a hand example (1, -1/2, 1/2) |
| `causal.twfe_covariate_diagnostic` | B | simulated drifting covariate effect (0.4 per period) | \|level coefficient - 0.4\| < 0.06 | — | test_did_covariates.py: TWFE diagnostic |  |
| `causal.synth` | B | simulated treated unit = 0.6 u0 + 0.4 u1, post gap 3 | weights within 0.15; \|gap - 3\| < 0.4 | — | test_causal_text_survival.py: synthetic control weights |  |
| `causal.randomization_inference` | A | exact p by hand enumeration (C(8,2) = 28 and C(6,k) designs); TWFE coefficient | 1e-12 (exact p); 1e-8 (statistic) | not measured (closed form in the test) | test_v252.py: exact randomization p; test_simulation_ri.py: RI statistic = TWFE DiD coefficient | size within [2%, 8%] in test_simulation_ri.py |
| `causal.did_single_treated` | B | validation DGP of 2026-11-07 (seed 20261107); Toeplitz lag weights | size in [0.03, 0.08]; estimate within 1e-6 of the recorded 2x2 DiD | — | test_v251.py: size on the 2026-11-07 validation DGP; test_v268.py: clipping rates of each check re-measured |  |
| `causal.did_falsification` | B | planted pre-trend and planted placebo-outcome effect | the planted row is a finding; clean design has none | — | test_v251.py: falsification |  |

## `dag`

| function | level | reference | tolerance | largest observed difference | test | note |
|---|---|---|---|---|---|---|
| `dag.d_separated` | B | textbook graphs (chain, fork, collider) solved by hand | exact | — | test_dag.py: chain A->B->C |  |
| `dag.is_valid_adjustment` | B | M-bias and butterfly graphs solved by hand | exact | — | test_dag.py: butterfly |  |
| `dag.adjustment_sets` | B | confounding, M-bias, butterfly and latent-confounder graphs solved by hand | exact | — | test_dag.py: confounding: minimal set {Z} |  |
| `dag.optimal_adjustment_set` | B | O-set of Henckel, Perković & Maathuis (2022) on hand-solved graphs | exact | — | test_dag.py: butterfly O-set = {M, Z2} |  |
| `dag.classify_controls` | B | hand-solved graphs (mediator, collider, instrument) | exact | — | test_dag.py: mediator M and post-treatment K |  |
| `dag.implied_independencies` | B | simulated linear-Gaussian data from the graph | all implications pass on the true graph; the missing edge is rejected | — | test_dag.py: true graph |  |
| `dag.test_implications` | A | Holm adjustment by statsmodels.stats.multitest | 1e-12 | not measured (software run in the test) | test_show_text.py: test_implications: Holm-adjusted p equal to statsmodels.multitest | known-truth checks in test_dag.py |
| `dag.adjustment_report` | B | hand-solved graphs | exact | — | test_dag.py: adjustment_report bundles sets; test_show_text.py: adjustment_report |  |

## `data_prep`

| function | level | reference | tolerance | largest observed difference | test | note |
|---|---|---|---|---|---|---|
| `data_prep.read_spreadsheetml` | D | — | — | — | — | exercised through read_any (test_causal_text_survival.py: SpreadsheetML .xls read) |
| `data_prep.read_any` | B | small files written by the test | exact | — | test_causal_text_survival.py: read_any sniffs |  |
| `data_prep.read_csv_filtered` | B | file written by the test | exact row count | — | test_causal_text_survival.py: read_csv_filtered |  |
| `data_prep.audit` | D | — | — | — | — |  |
| `data_prep.missing_report` | B | simulated MCAR and MAR missingness | complete-case N and shares exact (1e-12); MAR p < 1e-6 | — | test_missing_data.py: complete-case N is exact; test_show_text.py: missing_report |  |
| `data_prep.fix_mojibake` | B | known string | exact | — | test_causal_text_survival.py: fix_mojibake |  |
| `data_prep.br_number` | B | known strings | exact | — | test_causal_text_survival.py: br_number parses '1.234,56' |  |
| `data_prep.safe_merge` | B | frames built by the test | exact (the merge must raise) | — | test_causal_text_survival.py: safe_merge blocks many-to-many |  |
| `data_prep.pdf_text` | B | PDF written by the test | exact text read back | — | test_v2501.py: pdf_text at the 'pdf' minimum or later |  |
| `data_prep.robots_allowed` | D | — | — | — | — |  |
| `data_prep.polite_get` | C | — | — | — | test_v2501.py: polite_get without requests | only the missing-extra message |

## `dependence`

| function | level | reference | tolerance | largest observed difference | test | note |
|---|---|---|---|---|---|---|
| `dependence.pseudo_obs` | D | — | — | — | — | exercised through fit_copulas |
| `dependence.rank_correlations` | D | — | — | — | — |  |
| `dependence.empirical_tail_dependence` | D | — | — | — | — | exercised through fit_copulas; its printed values are checked against the ranks in test_show_text.py |
| `dependence.fit_copula` | B | simulated copulas with known parameter | relative 0.15 on the parameter | — | test_financial_modules.py: <family>: sim tau ≈ implied, MLE recovers param |  |
| `dependence.fit_copulas` | B | simulated Clayton, Frank, Gumbel, Student t and Gaussian data (family and tail sentence known); Clayton λ_L = 2^(−1/θ) | 1e-6 on λ_L (θ stored rounded to 6 decimals) | — | test_show_text.py: fit_copulas: Clayton lambda_L = 2^(-1/theta) | the λ check uses the same expression as the implementation, so it is not an independent route; with negative dependence the printed opposite-corner coefficient is checked against the t formula at −ρ and the empirical co-exceedance |
| `dependence.simulate_copula` | B | Kendall tau implied by the parameter | \|tau - implied\| < 0.03 | — | test_financial_modules.py: <family>: sim tau ≈ implied |  |
| `dependence.linfoot` | B | Gaussian pair with rho = 0.7 | \|Linfoot(MI) - 0.7\| < 0.05 | — | test_financial_modules.py: Linfoot(MI) ≈ \|rho\| for Gaussian |  |
| `dependence.mutual_information_matrix` | C | — | — | — | test_financial_modules.py: MI matrix diagonal NaN, symmetric | shape and symmetry only |
| `dependence.distance_correlation` | D | — | — | — | — |  |
| `dependence.mi_permutation_test` | D | — | — | — | — |  |
| `dependence.dependence_table` | B | simulated nonlinear and independent pairs | nonlinear gap > 0.5; independent MI < 0.2 | — | test_financial_modules.py: MI flags x~y nonlinear gap |  |

## `diagnostics`

| function | level | reference | tolerance | largest observed difference | test | note |
|---|---|---|---|---|---|---|
| `diagnostics.newey_west_lags` | D | — | — | — | — |  |
| `diagnostics.white_simplified` | D | — | — | — | — |  |
| `diagnostics.ols_diagnostics` | A | Wooldridge, Introductory Econometrics, Example 8.4 (Breusch-Pagan LM 14.09, p .0028) | 0.01 on LM, 0.0001 on p (looser than half the last printed digit: 0.005 and 0.00005) | 0.0024 of 0.01 | test_textbook_replication.py: W8.4 Breusch-Pagan | edition not stated in the test; tolerance looser than the v2.6.7 rule |
| `diagnostics.vif_table` | D | — | — | — | — |  |
| `diagnostics.influence` | D | — | — | — | — |  |
| `diagnostics.chow` | A | Chow F from three least-squares fits (numpy), computed in the test | 1e-8 absolute (looser than the 1e-10 rule for closed forms) | not measured (closed form in the test) | test_show_text.py: chow: F label F(2, 196) and F equal to the three-fit Chow F | tolerance looser than the v2.6.7 rule |
| `diagnostics.compare_standard_errors` | D | — | — | — | — |  |
| `diagnostics.comparison_table` | D | — | — | — | — |  |
| `diagnostics.stationarity` | B | simulated AR(0.9), random walk and white noise; Selic series | verdict at each alpha (exact) | — | test_v2472.py: stationarity |  |
| `diagnostics.stationarity_many` | D | — | — | — | — |  |
| `diagnostics.cointegration` | D | — | — | — | — |  |
| `diagnostics.endogeneity_test` | A | Wooldridge, Introductory Econometrics, Example 15.7 (Durbin-Wu-Hausman t = 1.67) | 0.02 on F = t², about 0.006 on t = 1.67 (looser than half the last printed digit, 0.005) | 0.0037 of 0.02 | test_textbook_replication.py: W15.7 Durbin-Wu-Hausman; test_v268.py: endogeneity_test, exactly identified | edition not stated in the test; tolerance looser than the v2.6.7 rule |
| `diagnostics.overid_test` | A | Wooldridge, Introductory Econometrics, Example 15.8 (nR² = .378, p = .539) | 0.0006 (looser than half the last printed digit, 0.0005) | 0.00036 of 0.0006 | test_textbook_replication.py: W15.8 overidentification | edition not stated in the test; nR² also equal to n·R² of the auxiliary regression (1e-9, test_show_text.py); tolerance looser than the v2.6.7 rule |
| `diagnostics.iv_report` | A | Wooldridge, Introductory Econometrics, Example 15.1 (educ .059 (.035)); Example 15.7 overidentification | 0.0003 on the coefficient and SE; 0.01 on the overidentification nR² and p (looser than half the last printed digit, 0.0005) | 4e-05 of 0.0003 | test_v242.py: 2SLS reproduces Wooldridge Example 15.1; test_v268.py: iv_report, exactly identified and weak | the test says 2e; the overidentification comparison is not measured; test_v242.py calls the overidentification nR² = .378 Example 15.7, test_textbook_replication.py calls the same number W15.8; the editions are not stated; tolerance looser than the v2.6.7 rule |
| `diagnostics.hausman` | D | — | — | — | — |  |
| `diagnostics.panel_bp_lm` | D | — | — | — | — |  |
| `diagnostics.mundlak_test` | A | linearmodels PanelOLS fixed effects on the data of Wooldridge, Introductory Econometrics, Example 14.4 (CRE = FE identity) | 1e-6 | not measured (software run in the test) | test_specification_tools.py: Mundlak/CRE coefficients equal FE | the book's FE numbers are compared with linearmodels, not with this function, in test_textbook_replication.py |
| `diagnostics.count_diagnostics` | A | Wooldridge, Introductory Econometrics, Example 17.3 (Pearson dispersion σ̂² = 1.52) | 0.005 | 0.0032 of 0.005 | test_specification_tools.py: W17.3 Pearson dispersion | edition not stated in the test |
| `diagnostics.tobit` | A | Wooldridge, Introductory Econometrics, Example 17.2 (educ 80.65, kidslt6 −894.02, σ 1122.02, log L −3819.09) | 0.01 to 0.05 (looser than half the last printed digit, 0.005) | 0.0046 of 0.01 | test_textbook_replication.py: W17.2 Tobit | edition not stated in the test; tolerance looser than the v2.6.7 rule |
| `diagnostics.diebold_mariano` | A | with h = 1, the OLS t of the loss difference on a constant (statsmodels) | 1e-9 on the statistic, 1e-12 on p | not measured (software run in the test) | test_show_text.py: diebold_mariano: with h = 1 the HLN statistic is the OLS t | h > 1 has no external reference |
| `diagnostics.main` | D | — | — | — | — | command-line entry point |

## `discrete_choice`

| function | level | reference | tolerance | largest observed difference | test | note |
|---|---|---|---|---|---|---|
| `discrete_choice.binary_effects` | A | Wooldridge, Introductory Econometrics, Example 17.1 (LPM) | 6e-5 to 6e-4 (looser than half the last printed digit, 5e-5 to 5e-4) | 0.00019 of 0.0006 | test_v242.py: LPM reproduces Wooldridge Example 17.1 | the test says 2e; AMEs also against closed forms (1e-8 to 1e-10); tolerance looser than the v2.6.7 rule |
| `discrete_choice.heckman` | A | Wooldridge, Introductory Econometrics, Example 17.5 and Stata heckman twostep | 0.0002 on coefficients and SEs; 0.0003 on se(λ), ρ and σ (looser than half the last printed digit, 5e-5) | 3.8e-05 of 0.0002 | test_v242.py: Heckit reproduces Wooldridge Example 17.5 and Stata heckman twostep | the test says 2e; tolerance looser than the v2.6.7 rule |

## `estimation`

| function | level | reference | tolerance | largest observed difference | test | note |
|---|---|---|---|---|---|---|
| `estimation.mle` | A | Wooldridge, Introductory Econometrics, Example 17.3 (Poisson); statsmodels Poisson | 0.0011 against the book (looser than half the last printed digit, 0.0005 or 0.00005 for inc86); 1e-5 against statsmodels | 0.00049 of 0.0011 | test_estimation.py: W17.3 Poisson coefficients match the book | edition not stated in the test; tolerance looser than the v2.6.7 rule |
| `estimation.trinity_test` | A | statsmodels LR and Wald | 1e-4 (LR), 1e-3 relative (Wald) (looser than the 1e-6 rule for other software) | not measured (software run in the test) | test_estimation.py: LR statistic = 2(llf_u - llf_r) from statsmodels | tolerance looser than the v2.6.7 rule |
| `estimation.nls` | A | statsmodels OLS (NLS with a linear f) | 1e-7 on coefficients, 1e-5 relative on HC1 SEs (SEs looser than the 1e-6 rule for other software) | not measured (software run in the test) | test_estimation.py: NLS with a linear f = OLS | tolerance looser than the v2.6.7 rule |
| `estimation.delta_method` | A | exact variance a'Va of a linear function | 1e-9 | not measured (closed form in the test) | test_estimation.py: delta method of a linear function is exact |  |
| `estimation.bootstrap` | B | HC3 and CR1 SEs under simulated heteroskedasticity and clustering | within 15% | — | test_estimation.py: pairs bootstrap SE ~ HC3 SE |  |
| `estimation.wild_cluster_bootstrap` | B | simulated 8-cluster design | size in [2%, 9%]; power > 70% | — | test_estimation.py: 8 clusters: wild cluster bootstrap size |  |

## `finance_empirics`

| function | level | reference | tolerance | largest observed difference | test | note |
|---|---|---|---|---|---|---|
| `finance_empirics.returns` | A | log return = ln(1 + simple return); compounding identities | numpy allclose defaults (rtol 1e-5); 1e-12 | not measured (closed form in the test) | test_finance_empirics.py: log return = ln(1 + simple return) | the log identity uses the function's own simple returns; tolerance looser than the v2.6.7 rule (numpy defaults) |
| `finance_empirics.market_model` | A | t of β = 1 and joint Wald from the HAC fit, computed in the test | 1e-9, 1e-8 absolute (looser than the 1e-10 rule for closed forms) | not measured (software run in the test) | test_v242.py: market model: t of β = 1 and joint Wald | known-truth beta in test_finance_empirics.py; tolerance looser than the v2.6.7 rule |
| `finance_empirics.rolling_beta` | A | OLS on the same window | 1e-10 (beta), 1e-8 (SE) | not measured (software run in the test) | test_finance_empirics.py: last rolling beta = OLS beta on the same window | the beta is compared with fe.market_model (the skill's own function); the independent route is the classical SE from sm.OLS |
| `finance_empirics.adjusted_beta` | A | Blume 0.67·β + 0.33 (closed form) | 1e-12 | not measured (closed form in the test) | test_finance_empirics.py: Blume |  |
| `finance_empirics.unlever_beta` | A | Hamada and Harris-Pringle formulas | 1e-12 | not measured (closed form in the test) | test_finance_empirics.py: Hamada formula |  |
| `finance_empirics.relever_beta` | A | Hamada and Harris-Pringle formulas (round trip) | 1e-12 | not measured (closed form in the test) | test_finance_empirics.py: Hamada: unlever then relever |  |
| `finance_empirics.dimson_beta` | B | simulated non-synchronous trading, true beta 1 | CI covers 1.0 | — | test_finance_empirics.py: non-synchronous trading |  |
| `finance_empirics.factor_model` | B | simulated loadings 1.1, 0.5, −0.3 | each CI covers its loading | — | test_finance_empirics.py: factor loadings |  |
| `finance_empirics.fama_macbeth` | A | mean of per-period OLS slopes and SD/√T, and the Shanken formula, computed in the test | numpy allclose defaults | not measured (software run in the test) | test_v247.py: fama_macbeth: premia = mean of per-period OLS slopes | the Shanken check uses the function's own premium and NW SE; the independent route is the mean of per-period OLS slopes; tolerance looser than the v2.6.7 rule (numpy defaults) |
| `finance_empirics.event_study` | B | simulated abnormal return 0.02 on day 0 | \|CAAR - 0.02\| < 0.008; tests reject at 1% | — | test_finance_empirics.py: event study: CAAR(-1,1) ~ 0.02 |  |
| `finance_empirics.car_regression` | B | simulated deal effect −0.03 | CI covers −0.03 | — | test_finance_empirics.py: car_regression |  |
| `finance_empirics.performance_ratios` | A | Sharpe, information ratio, Treynor and Jensen alpha by their formulas | 1e-12 (1e-10 for Jensen) | not measured (closed form in the test) | test_finance_empirics.py: Sharpe = mean/sd * sqrt(252) | the Jensen alpha is compared with fe.market_model (the skill's own function) |
| `finance_empirics.shrink_cov` | C | — | — | — | test_finance_empirics.py: Ledoit-Wolf covariance is symmetric positive definite | properties only (symmetric, positive definite, shrinkage in (0, 1)) |
| `finance_empirics.efficient_frontier` | A | GMV weights Σ⁻¹1/(1'Σ⁻¹1) (closed form) | numpy allclose defaults | not measured (closed form in the test) | test_finance_empirics.py: GMV weights | tolerance looser than the v2.6.7 rule (numpy defaults) |

## `forecasting`

| function | level | reference | tolerance | largest observed difference | test | note |
|---|---|---|---|---|---|---|
| `forecasting.select_d` | B | simulated random walk | exact (d = 1) | — | test_financial_modules.py: select_d = 1 for a random walk |  |
| `forecasting.auto_arima` | B | simulated ARMA(1,1) | order in a small accepted set, d = 0 | — | test_financial_modules.py: auto_arima recovers ARMA(1,1) |  |
| `forecasting.lag_exog` | B | simulated series driven by lagged x | ARIMAX with lagged x wins, DM p < 0.01 | — | test_financial_modules.py: ARIMAX with lagged x wins |  |
| `forecasting.rolling_origin` | C | — | — | — | test_financial_modules.py: rolling_origin length | length only; used by the known-truth forecast checks |
| `forecasting.rolling_var` | B | simulated VAR(1) | lag 1 chosen most often; target dates exact | — | test_financial_modules.py: rolling_var picks lag 1 mostly |  |
| `forecasting.benchmarks` | B | simulated ARMA data | ARMA RMSE below the naive one | — | test_financial_modules.py: ARMA beats naive on ARMA data |  |
| `forecasting.pesaran_timmermann` | B | simulated forecasts with direction skill | one-sided p < 0.01 | — | test_financial_modules.py: Pesaran-Timmermann detects direction skill |  |
| `forecasting.forecast_metrics` | D | — | — | — | — | exercised through evaluate |
| `forecasting.evaluate` | B | simulated forecasts with known accuracy ranking | DM p < 0.01 for the better model | — | test_financial_modules.py: ARIMAX with lagged x wins and DM significant; test_show_text.py: evaluate |  |

## `hte`

| function | level | reference | tolerance | largest observed difference | test | note |
|---|---|---|---|---|---|---|
| `hte.cate_fit` | B | simulated CATE = 1 + x1 | CIs cover the true values | — | test_hte.py: <learner>: ATE CI contains 1 | ATE CI coverage in [0.91, 0.98] by Monte Carlo |
| `hte.cate_predict` | B | simulated CATE | CIs cover tau at x1 = −1, 0, 1 | — | test_hte.py: cate_predict: CIs cover tau |  |
| `hte.blp_test` | B | simulated heterogeneous and homogeneous effects | p < 0.001 with heterogeneity; p > 0.05 without | — | test_hte.py: BLP |  |
| `hte.gates` | B | simulated group effects | each GATE CI covers the true group mean | — | test_hte.py: each GATE CI covers the true mean tau |  |
| `hte.clan` | B | simulated modifier x1 and non-modifiers x3, x4 | Holm p < 0.001 for x1, > 0.05 for x3, x4 | — | test_hte.py: CLAN |  |
| `hte.toc` | B | TOC at q = 1 is zero by construction; simulated heterogeneity | 1e-12 (q = 1); p < 0.001 | — | test_hte.py: TOC |  |
| `hte.rscore` | B | simulated heterogeneous and homogeneous effects | > 0.02 and < 0.01 | — | test_hte.py: R-score |  |
| `hte.compare_cate_models` | B | simulated symmetric heterogeneity | forest R-score above the linear learner's by 0.03 | — | test_hte.py: compare_cate_models |  |
| `hte.policy_tree` | B | simulated rule 'treat if x1 > 0'; value E[max(x1, 0)] = 0.399 | agreement > 90%; value within 0.1 | — | test_hte.py: policy tree recovers 'treat if x1 > 0' |  |
| `hte.policy_predict` | B | simulated rule 'treat if x1 > 0' | agreement > 90% | — | test_hte.py: policy tree recovers 'treat if x1 > 0' |  |
| `hte.policy_value` | D | — | — | — | — |  |

## `inference`

| function | level | reference | tolerance | largest observed difference | test | note |
|---|---|---|---|---|---|---|
| `inference.describe_dist` | A | pandas skew and kurt (Excel SKEW, KURT) | rtol and atol 1e-8 | not measured (software run in the test) | test_v240.py: describe_dist: skewness equals pandas/Excel SKEW |  |
| `inference.one_sample` | A | scipy.stats ttest_1samp and its confidence interval | rtol and atol 1e-8 | not measured (software run in the test) | test_v240.py: one_sample: t, p and CI equal scipy |  |
| `inference.two_sample` | A | scipy.stats ttest_ind (Welch and pooled) | rtol and atol 1e-8 | not measured (software run in the test) | test_v240.py: two_sample: Welch is the default |  |
| `inference.variance_test` | A | chi-square and F formulas; scipy levene(center='median') | rtol and atol 1e-8 | not measured (software run in the test) | test_v240.py: variance_test |  |
| `inference.proportion_ci` | A | statsmodels proportion_confint (Wilson, Clopper-Pearson, Agresti-Coull, Wald) | 1e-7 absolute | not measured (software run in the test) | test_v240.py: proportion_ci |  |
| `inference.proportion_test` | A | scipy binomtest; statsmodels score and pooled z; Newcombe CI | 1e-7 to 1e-8 | not measured (software run in the test) | test_v240.py: proportion_test |  |
| `inference.n_for_proportion` | A | textbook sample sizes (1068 at ±3 points, 95%; 965 with N = 10,000) | exact | 0 (exact) | test_v240.py: n_for_proportion |  |
| `inference.chisq_fit` | A | scipy chisquare; pooled cells by hand | rtol and atol 1e-8 | not measured (software run in the test) | test_v240.py: chisq_fit with given probabilities equals scipy chisquare |  |
| `inference.normality` | A | scipy shapiro | rtol and atol 1e-8 | not measured (software run in the test) | test_v240.py: normality: Shapiro-Wilk equals scipy |  |
| `inference.contingency` | A | chi-square and Cramér's V by hand; scipy fisher_exact; Mantel-Haenszel by hand | rtol and atol 1e-8 | not measured (software run in the test) | test_v240.py: contingency |  |
| `inference.nonparametric` | A | scipy mannwhitneyu and wilcoxon; Hodges-Lehmann as the median of differences | rtol and atol 1e-8 | not measured (software run in the test) | test_v240.py: nonparametric: Mann-Whitney p equals scipy | HL CI coverage in [0.91, 0.985] |
| `inference.correlation_test` | A | Pearson r and Fisher-z CI by hand | rtol and atol 1e-8 | not measured (closed form in the test) | test_v240.py: correlation_test: Pearson r and Fisher-z CI |  |
| `inference.power_curve` | A | statsmodels TTestIndPower and TTestPower; normal-power closed form | 1e-6 (statsmodels), 1e-5 (closed form) (closed form looser than the 1e-10 rule) | not measured (software run in the test) | test_v240.py: power_curve two-sample t equals statsmodels | tolerance looser than the v2.6.7 rule |
| `inference.n_for_power` | A | ceil of statsmodels solve_power | exact | not measured (software run in the test) | test_v246.py: n_for_power two-sample |  |
| `inference.adjust_pvalues` | A | statsmodels multipletests (Bonferroni, Holm, BH, BY) | rtol and atol 1e-8 | not measured (software run in the test) | test_v240.py: adjust_pvalues: Bonferroni, Holm, BH and BY equal statsmodels |  |

## `linear_models`

| function | level | reference | tolerance | largest observed difference | test | note |
|---|---|---|---|---|---|---|
| `linear_models.anova_table` | A | scipy f_oneway | 1e-8 | not measured (software run in the test) | test_linear_models.py: one-way ANOVA F = scipy f_oneway |  |
| `linear_models.pairwise` | B | simulated groups with known differences | the planted pair rejects, the null pair does not | — | test_linear_models.py: Tukey: a vs d significant, a vs b not |  |
| `linear_models.ancova` | B | simulated adjusted difference 0.5 | within 3 SE of 0.5; equal to the group coefficient (1e-10) | — | test_linear_models.py: ANCOVA adjusted difference |  |
| `linear_models.robust_regression` | B | simulated slope 2 with planted outliers | robust error below a third of the OLS error | — | test_linear_models.py: bisquare slope closer to 2 than OLS |  |
| `linear_models.quantile_regression` | B | simulated slopes 1 + Φ⁻¹(τ) | within 3 SE | — | test_linear_models.py: quantile slopes within 3 SE | convergence to the solution checked in test_v237.py (1e-5) |
| `linear_models.permutation_test` | B | simulated effect and null with t(2) errors | p < 0.01 with the effect; size in [2%, 8%] | — | test_linear_models.py: permutation size |  |
| `linear_models.lack_of_fit_test` | B | simulated quadratic and linear truths | p < 0.001 and p > 0.05 | — | test_linear_models.py: linear fit to a quadratic: lack of fit rejected |  |
| `linear_models.boxcox_response` | B | simulated log-normal and linear responses | \|lambda\| < 0.15; CI coverage in [0.90, 0.99] | — | test_linear_models.py: Box-Cox on log-normal y |  |
| `linear_models.partial_residuals` | B | simulated quadratic and linear truths | curvature p < 0.001 and > 0.05 | — | test_linear_models.py: partial residuals flag curvature |  |
| `linear_models.shrinkage_compare` | B | simulated sparse truth (four regressors) | lasso keeps the four; CV RMSE below OLS | — | test_linear_models.py: sparse truth |  |
| `linear_models.post_double_selection` | B | simulated effect 0.5 with 60 candidate controls | coverage >= 0.88; mean within 0.03 | — | test_linear_models.py: post-double-selection |  |
| `linear_models.select_subsets` | B | simulated true model {a, c} | ranked first by BIC | — | test_linear_models.py: best subsets by BIC |  |

## `multivariate`

| function | level | reference | tolerance | largest observed difference | test | note |
|---|---|---|---|---|---|---|
| `multivariate.var_analysis` | A | statsmodels VAR select_order and test_causality | numpy isclose defaults | not measured (software run in the test) | test_v247.py: var_analysis: lag order and criteria = statsmodels select_order | known-truth Granger direction also checked; tolerance looser than the v2.6.7 rule (numpy defaults) |
| `multivariate.irf_bands` | A | statsmodels orth_irfs (both orderings) | numpy allclose defaults | not measured (software run in the test) | test_v247.py: irf_bands: point IRF = statsmodels orth_irfs | band coverage >= 0.8 for a persistent VAR; tolerance looser than the v2.6.7 rule (numpy defaults) |
| `multivariate.fevd_table` | A | statsmodels fevd | numpy allclose defaults | not measured (software run in the test) | test_v247.py: fevd_table: shares = statsmodels fevd | tolerance looser than the v2.6.7 rule (numpy defaults) |
| `multivariate.svar_identify` | B | identification of textbook restriction patterns | exact verdicts | — | test_v247.py: svar_identify |  |
| `multivariate.svar_longrun` | B | B0 B0' = Σ_u and a lower-triangular long-run matrix (defining identities) | numpy allclose defaults; 1e-10 | — | test_v247.py: svar_longrun | tolerance looser than the v2.6.7 rule (numpy defaults) |
| `multivariate.svar_sign` | B | imposed sign pattern; rotation of the Cholesky factor | signs exact; norm 1 (isclose) | — | test_v247.py: svar_sign |  |
| `multivariate.johansen_trace` | B | simulated cointegrated systems | size of the rank tests <= 10% to 15% | — | test_v247.py: johansen_trace | fewer replications than the 200-panel rule; written before it |
| `multivariate.vecm_fit` | A | statsmodels VECM beta | numpy allclose defaults | not measured (software run in the test) | test_v247.py: vecm_fit: rank 1, β = statsmodels VECM β | known cointegrating vector (1, −2) within 0.05; tolerance looser than the v2.6.7 rule (numpy defaults) |

## `regimes`

| function | level | reference | tolerance | largest observed difference | test | note |
|---|---|---|---|---|---|---|
| `regimes.markov_switching` | B | simulated two-regime series, durations 50 | regime accuracy > 0.9; duration in (30, 80) | — | test_financial_modules.py: MS filtered regimes match truth |  |
| `regimes.regime_features` | C | — | — | — | test_financial_modules.py: HMM filtered labels match truth | used to build the HMM features; its own output is not compared |
| `regimes.kmeans_regimes` | C | — | — | — | test_financial_modules.py (called, no check of its own) |  |
| `regimes.choose_k` | D | — | — | — | — |  |
| `regimes.forward_filter` | D | — | — | — | — |  |
| `regimes.hmm_regimes` | B | simulated two-regime series | label accuracy > 0.85, also out of sample | — | test_financial_modules.py: HMM filtered labels match truth |  |
| `regimes.regime_summary` | C | — | — | — | test_financial_modules.py (called, no check of its own) |  |
| `regimes.compare_labels` | C | — | — | — | test_financial_modules.py (called, no check of its own) |  |

## `reporting`

| function | level | reference | tolerance | largest observed difference | test | note |
|---|---|---|---|---|---|---|
| `reporting.coef_table` | D | — | — | — | — |  |
| `reporting.regression_table` | C | — | — | — | test_causal_text_survival.py: regression table layout | layout only |
| `reporting.export_table` | B | table exported and read back | equal after rounding to 6 decimals | — | test_setup_env.py: with openpyxl: a table exported to .xlsx reads back unchanged |  |
| `reporting.rescale_hint` | D | — | — | — | — |  |
| `reporting.run_manifest` | B | same and changed data | exact hashes; fields written | — | test_did_covariates.py: manifest |  |
| `reporting.cross_check` | B | constructed estimates and SEs | exact verdicts | — | test_v2461.py: cross_check |  |

## `risk`

| function | level | reference | tolerance | largest observed difference | test | note |
|---|---|---|---|---|---|---|
| `risk.ewma_volatility` | A | EWMA recursion σ²_t = 0.94 σ²_{t−1} + 0.06 r²_{t−1}, computed in the test | numpy isclose defaults | not measured (closed form in the test) | test_financial_modules.py: EWMA uses only past returns | the recursion is checked on the function's own previous value, not against a series built in the test; tolerance looser than the v2.6.7 rule (numpy defaults) |
| `risk.realized_volatility` | D | — | — | — | — |  |
| `risk.var_historical` | B | simulated GARCH returns | static historical VaR rejected by DQ in > 85% of samples | — | test_evt_es.py: DQ test detects a static historical VaR |  |
| `risk.expected_shortfall_historical` | D | — | — | — | — |  |
| `risk.var_parametric` | B | simulated GARCH-t returns | static normal VaR violations cluster (Christoffersen p below the GARCH-t one) | — | test_financial_modules.py: Static normal VaR violations cluster or exceed |  |
| `risk.var_ewma` | B | simulated heavy-tailed returns | EWMA-normal fails Kupiec (p < 0.05) | — | test_evt_es.py: conditional EVT passes Kupiec and Christoffersen |  |
| `risk.var_garch` | B | simulated GARCH-t returns | Kupiec p > 0.05; ES >= VaR | — | test_financial_modules.py: GARCH-t VaR has correct coverage |  |
| `risk.kupiec_pof` | B | 10 hits in 1000 at 1% (exactly the expected rate) | p > 0.99 | — | test_financial_modules.py: Kupiec: 10 hits in 1000 at 1% |  |
| `risk.christoffersen` | C | — | — | — | test_evt_es.py: DQ test detects a static historical VaR | used only as the comparison for DQ; its p-value is not checked |
| `risk.dq_test` | B | true VaR (size) and static VaR on GARCH data (power) | size 2%–9% at 5%; power > 85% | — | test_evt_es.py: DQ test size with the true VaR |  |
| `risk.basel_traffic_light` | A | Basel Committee (1996) zones for 250 days at 99%: 0–4 green, 5–9 yellow, 10+ red | exact | 0 (exact) | test_financial_modules.py: Basel: 4 viol green, 5 yellow, 10 red |  |
| `risk.backtest_var` | B | simulated returns with correct and wrong VaR models | verdicts per model | — | test_evt_es.py: conditional EVT passes Kupiec and Christoffersen |  |
| `risk.mean_excess` | B | t(4) and normal samples | slope > 0.1 and < 0 | — | test_evt_es.py: mean excess |  |
| `risk.gpd_fit` | B | simulated GPD with xi = 0.3; t(4) with xi = 0.25 | mean xi within 0.02; CI coverage in [0.91, 0.98] | — | test_evt_es.py: GPD: mean xi-hat within 0.02 of 0.3 |  |
| `risk.var_evt` | B | simulated heavy-tailed returns | Kupiec and Christoffersen p > 0.05 | — | test_evt_es.py: conditional EVT passes Kupiec and Christoffersen |  |
| `risk.es_backtest` | A | Acerbi-Szekely Z2 = Σ r I / (T α ES) + 1, computed in the test | 1e-12 | not measured (closed form in the test) | test_evt_es.py: Z2 matches sum r I / (T alpha ES) + 1 | size 1.5%–8% and power > 75% by simulation |
| `risk.fz_loss` | B | true (VaR, ES) against six perturbations | the true pair has the lowest mean loss | — | test_evt_es.py: FZ0: true (VaR, ES) has lower mean loss |  |
| `risk.compare_es_models` | B | simulated returns with a correct and two wrong models | the correct model ranks first; the others p < 0.05 | — | test_evt_es.py: FZ0 ranks conditional EVT first |  |

## `segments`

| function | level | reference | tolerance | largest observed difference | test | note |
|---|---|---|---|---|---|---|
| `segments.top_decile_lift` | A | lift of a perfect score = 1 / base rate | numpy isclose defaults | not measured (closed form in the test) | test_llm_annotation_segments.py: top-decile lift of a perfect score | tolerance looser than the v2.6.7 rule (numpy defaults) |
| `segments.calibration` | B | true and doubled-logit probabilities | slope within 0.08 of 1 and 0.06 of 0.5 | — | test_llm_annotation_segments.py: calibration of the true probabilities |  |
| `segments.logit_leaf_model` | B | simulated two-segment process with opposite effects | split at x1 = 0 within 0.15; AMEs of opposite signs | — | test_llm_annotation_segments.py: tree finds the true segment boundary |  |
| `segments.leaf_model_stability` | B | simulated two-segment process | the true split variable in every seed | — | test_llm_annotation_segments.py: stability |  |
| `segments.predict_leaf_model` | C | — | — | — | test_llm_annotation_segments.py: predictions are probabilities | range and warnings only |

## `simulation`

| function | level | reference | tolerance | largest observed difference | test | note |
|---|---|---|---|---|---|---|
| `simulation.monte_carlo` | D | — | — | — | — | exercised by every demo_* function |
| `simulation.demo_omitted_variable` | B | simulated omitted confounder; the bias the function predicts (γδ/(δ²+1)) | 0.02 | — | test_simulation_ri.py: omitted variable | the predicted bias is returned by the function, not computed in the test |
| `simulation.demo_bad_control` | B | simulated mediator | total effect bias < 0.03; mediator-controlled bias > 0.4 | — | test_simulation_ri.py: bad control |  |
| `simulation.demo_measurement_error` | B | simulated classical measurement error; the attenuation factor the function returns | 0.02 | — | test_simulation_ri.py: measurement error | the attenuation factor is returned by the function |
| `simulation.demo_heteroskedasticity` | B | simulated heteroskedastic errors | classical coverage < 0.90; HC3 >= 0.93 | — | test_simulation_ri.py: heteroskedasticity |  |
| `simulation.demo_spurious_regression` | B | independent random walks | true null rejected > 60% | — | test_simulation_ri.py: spurious regression |  |
| `simulation.demo_weak_iv` | B | simulated weak and strong instruments | weak 2SLS biased toward OLS; strong ~ unbiased | — | test_simulation_ri.py: weak IV |  |
| `simulation.demo_few_clusters` | B | simulated few-cluster design | CR1 size > 15%; wild bootstrap in [2%, 11%] | — | test_simulation_ri.py: few clusters |  |
| `simulation.demo_staggered_did` | B | simulated staggered adoption | TWFE bias > 1; Callaway-Sant'Anna within 0.3 | — | test_simulation_ri.py: staggered DiD |  |
| `simulation.demo_did_covariates` | B | simulated drifting covariate effect | TWFE bias > 0.5; CS doubly robust within 0.1 | — | test_did_covariates.py: demo |  |
| `simulation.power_mde` | A | statsmodels per-group n; design effect 1 + (m − 1)ρ | within 1 unit of n; 1e-12 on the design effect | not measured (software run in the test) | test_simulation_ri.py: power_mde: total n = 2 x statsmodels per-group n |  |
| `simulation.mde_did` | B | its MDE used as the effect size: simulated power ≈ 80% (McKenzie 2012 SE checked against power_did_sim, not against this function) | power within 0.06 of 80% | — | test_power_did.py: equicorrelated errors | the returned MDE is not compared with the formula |
| `simulation.power_did_sim` | B | equicorrelated and AR(1) panels | power within 0.06 of 80%; size 2.5%–7% | — | test_power_did.py: equicorrelated errors: simulated power |  |
| `simulation.mde_did_sim` | B | AR(1) panel | power at the MDE in [0.78, 0.85] | — | test_power_did.py: mde_did_sim |  |
| `simulation.did_power_inputs` | B | simulated AR(1) = 0.6, SD 0.7 | within 0.03 and 0.05 | — | test_power_did.py: did_power_inputs |  |
| `simulation.demo_clt` | B | simulated means against the σ/√n column the function returns | relative 0.06 | — | test_v240.py: demo_clt: SD of the mean equals sigma/sqrt(n) | both columns come from the function |
| `simulation.demo_ci_coverage` | B | nominal 95% intervals | within 0.015 of 0.95 | — | test_v240.py: demo_ci_coverage |  |
| `simulation.demo_estimator_mse` | B | known bias of S² and of the uniform MLE | bias within 0.02 to 0.05 | — | test_v240.py: demo_estimator_mse |  |
| `simulation.demo_multiple_testing` | B | family-wise error 1 − 0.95^20 ≈ 64% (hand-derived), 300 simulations | in (0.55, 0.73) | — | test_v240.py: demo_multiple_testing | a band around a hand-derived value |
| `simulation.demo_simultaneity` | A | hand-derived probability limit of OLS (0.25 with c = 2) against the one the function returns; simulated estimates | 1e-12 on the probability limit; 0.05 on the simulated mean | not measured (software run in the test) | test_v242.py: simultaneity |  |
| `simulation.demo_lpm_bounds` | B | true average marginal effects | LPM slope within 5% of the AME | — | test_v242.py: LPM bounds |  |
| `simulation.demo_multicollinearity` | B | SE ratio against √VIF, with the VIF the function returns | relative 10% | — | test_v242.py: multicollinearity |  |
| `simulation.demo_trend_vs_difference` | B | lag-1 ACF of the differenced trend-stationary AR, (φ − 1)/2 (hand-derived) | 0.05 | — | test_v246.py: demo_trend_vs_difference | a band around a hand-derived value |
| `simulation.demo_ar1_forecast_variance` | B | simulated forecast variance against the theory column the function returns | relative 10% | — | test_v246.py: demo_ar1_forecast_variance | both columns come from the function |
| `simulation.demo_unit_root_power` | B | simulated AR(φ) with φ = 1, 0.95, 0.8 | size < 0.1; power ordering | — | test_v246.py: demo_unit_root_power |  |
| `simulation.demo_break_unit_root` | B | stationary AR with a mean shift | rejection < 0.1 ignoring the break, > 0.9 with the dummy | — | test_v246.py: demo_break_unit_root |  |

## `specification`

| function | level | reference | tolerance | largest observed difference | test | note |
|---|---|---|---|---|---|---|
| `specification.fgls_het` | A | Wooldridge, Introductory Econometrics, Example 8.7 (log(income) 1.30 (.44), educ −.463 (.120), restaurn −3.46 (.80)) | 0.006 (looser than half the last printed digit, 0.005) | 0.0048 of 0.006 | test_specification_tools.py: W8.7 FGLS | edition not stated in the test; tolerance looser than the v2.6.7 rule |
| `specification.prediction_intervals` | A | Wooldridge, Introductory Econometrics, Examples 6.5 and 6.6 | 0.005 | 0.0017 of 0.005 | test_specification_tools.py: W6.5 prediction | edition not stated in the test; smearing coverage within 0.01 of 95% by simulation |
| `specification.long_run_effect` | A | Wooldridge, Introductory Econometrics, Example 10.4 (LRP .101, se .030) | 0.0006 (looser than half the last printed digit, 0.0005) | 0.00028 of 0.0006 | test_specification_tools.py: W10.4 LRP | edition not stated in the test; tolerance looser than the v2.6.7 rule |
| `specification.standardized_coefficients` | A | Wooldridge, Introductory Econometrics, Example 6.1 | 0.0006 (looser than half the last printed digit, 0.0005) | 0.00045 of 0.0006 | test_specification_tools.py: W6.1 standardized coefficients | edition not stated in the test; tolerance looser than the v2.6.7 rule |
| `specification.ovb_decomposition` | A | short and long coefficients from two separate statsmodels OLS fits; identity short = long + γ·δ | 1e-10 | not measured (software run in the test) | test_specification_tools.py: OVB: short and long coefficients equal two separate statsmodels OLS fits; test_specification_tools.py: OVB identity |  |
| `specification.oster` | A | R robomit 1.0.7, which follows Stata psacalc (tools/r_reference/oster_v252_reference.R -> output/oster_v252_reference.json; mtcars) | 2e-6 (looser than half the last of the 6 decimals robomit prints, 5e-7) | 4.4e-07 of 2e-06 | test_v252.py: robomit; test_v268.py: (a) one control, R² gain at rounding level | the exact estimator; population checks with n = 400 000 within 0.01; v2.6.8: alternatives split by side; the spurious root with several controls documented (still classified by its distance to the pole); known limitations near β° = β̃ in the docstring; tolerance looser than the v2.6.7 rule |
| `specification.robustness_value` | A | Cinelli & Hazlett (2020), Darfur example (partial R² 2.2%, RV 13.9%, RVa 7.6%) | 0.0006 (looser than half the last printed digit, 0.0005) | 0.00036 of 0.0006 | test_specification_tools.py: Cinelli-Hazlett Darfur | definitional checks to 4 ulp in test_v264.py; tolerance looser than the v2.6.7 rule |
| `specification.sensitivity` | A | robustness_value on the t and df read from the fit; SSR refit of grouped benchmarks | 1e-12; 1e-10 | not measured (software run in the test) | test_specification_tools.py: sensitivity() reads t and df from a fitted model | the published Darfur numbers are checked through robustness_value; RV is compared with robustness_value, which the implementation calls |
| `specification.ovb_bounds` | A | R sensemakr 0.1.6 (tools/r_reference/ovb_bounds_v26_reference.R -> output/ovb_bounds_v26_reference.json) and the published Darfur table | every column equal to sensemakr (tolerances in test_v26.py); 0.01 on the published table (looser than half the last printed digit, 0.005) | 1.2e-14 (relative) of 1e-08 | test_v26.py: R | the measured differences cover the sensemakr comparisons (some made through private helpers); the published-table check is not measured; Darfur data (GPL-3) are not in the repository; tolerance looser than the v2.6.7 rule |
| `specification.linear_restriction_test` | A | Wooldridge, Introductory Econometrics, Example 4.9 (F = 9.55); statsmodels f_test and t_test | 0.01 against the book (looser than half the last printed digit, 0.005); 1e-8 against statsmodels | 0.00025 of 0.01 | test_v242.py: exclusion F equals statsmodels f_test | the test says 2e; tolerance looser than the v2.6.7 rule |
| `specification.prais_winsten` | A | Wooldridge, Introductory Econometrics, Example 12.4 (ρ̂ = .293; lchempi 2.94 (.63), lgas 1.05 (.98)) | 0.0006 on ρ̂, 0.006 on every coefficient and SE (looser than half the last printed digit: 0.0005, and 0.005 or 0.0005) | 0.0036 of 0.006 | test_v242.py: Prais-Winsten reproduces Wooldridge Example 12.4 | the test says 2e; tolerance looser than the v2.6.7 rule |
| `specification.nonnested_test` | A | J statistic = t of the rival's fitted values, computed in the test | 1e-8 absolute (looser than the 1e-10 rule for closed forms) | not measured (closed form in the test) | test_v242.py: J statistic = t of the rival's fitted values | tolerance looser than the v2.6.7 rule |

## `state_space`

| function | level | reference | tolerance | largest observed difference | test | note |
|---|---|---|---|---|---|---|
| `state_space.kalman_regression` | D | — | — | — | — | exercised through fit_kalman_regression |
| `state_space.fit_kalman_regression` | B | simulated time-varying beta | tracking error < 0.05; innovations ~ N(0, 1) within 0.1 | — | test_financial_modules.py: Kalman tracks time-varying beta |  |
| `state_space.half_life` | B | OU process with AR coefficient 0.95 (half-life ln 2 / −ln 0.95 ≈ 13.5) | in (10, 18) | — | test_financial_modules.py: half-life |  |
| `state_space.hurst_exponent` | B | OU and random walk | < 0.4 and in (0.4, 0.6) | — | test_financial_modules.py: Hurst < 0.5 for OU |  |
| `state_space.pairs_position` | C | — | — | — | test_financial_modules.py: pairs returns computed, no NaN |  |
| `state_space.pairs_returns` | C | — | — | — | test_financial_modules.py: pairs returns computed, no NaN |  |
| `state_space.cointegration_pretest` | C | — | — | — | test_financial_modules.py (called, no check of its own) |  |

## `survival`

| function | level | reference | tolerance | largest observed difference | test | note |
|---|---|---|---|---|---|---|
| `survival.kaplan_meier` | B | simulated groups with different hazards | median survival ordering | — | test_causal_text_survival.py: KM: x=1 survives less |  |
| `survival.logrank` | B | simulated groups with different hazards | p < 0.001 | — | test_causal_text_survival.py: log-rank rejects |  |
| `survival.cox` | A | R survival 3.8.12 coxph(robust) with tied times, Efron and Breslow, strata and clusters (tools/r_reference/cox_v267_reference.R -> output/cox_v267_reference.json); lifelines 0.30.3 without ties (tools/py_reference/cox_robust_v267_lifelines.py -> tests/data/cox_robust_lifelines_v267.json) | 1e-6 on coefficients, 1e-6 relative on SEs | 5.1e-07 of 1e-06 | test_show_text.py: cox R; test_show_text.py: cox | lifelines is not a reference with ties (its score residuals do not handle them); known HR exp(0.5) and CI coverage (200 samples); score residuals: sum 0 at the maximum, unchanged by a shift of x, hand formula in a stratum with one event |
| `survival.ph_test` | B | true proportional hazards and crossing hazards | p > 0.01 and p < 0.001 | — | test_causal_text_survival.py: PH test detects crossing hazards |  |
| `survival.person_period` | B | simulated durations | at most one event row per id | — | test_causal_text_survival.py: person-period expansion |  |

## `systems`

| function | level | reference | tolerance | largest observed difference | test | note |
|---|---|---|---|---|---|---|
| `systems.sur_system` | A | Breusch-Pagan LM = T·r² of the OLS residuals, computed in the test; SUR = OLS with identical regressors | numpy isclose/allclose defaults | not measured (software run in the test) | test_v247.py: sur_system | 'SUR = OLS' compares two outputs of the function; the independent route is the Breusch-Pagan LM; tolerance looser than the v2.6.7 rule (numpy defaults) |
| `systems.three_sls` | B | exactly identified system (3SLS = 2SLS) and a known over-identified demand equation | numpy allclose defaults; exact refusals | — | test_v247.py: three_sls | the 2SLS column is the function's own; no external reference; tolerance looser than the v2.6.7 rule (numpy defaults) |
| `systems.dynamic_panel` | A | linearmodels IV2SLS on the stacked differences and IVGMM two-step | numpy allclose defaults | not measured (software run in the test) | test_v247.py: dynamic_panel (difference GMM, collapsed, identity first step) = linearmodels IV2SLS | known ρ = 0.9 and 0.5 recovered within 0.03; tolerance looser than the v2.6.7 rule (numpy defaults) |

## `text_data`

| function | level | reference | tolerance | largest observed difference | test | note |
|---|---|---|---|---|---|---|
| `text_data.normalize_text` | D | — | — | — | — |  |
| `text_data.regex_flags` | B | small texts written by the test | exact | — | test_causal_text_survival.py: regex flags |  |
| `text_data.regex_extract` | B | small texts written by the test | exact | — | test_causal_text_survival.py: regex_extract parties |  |
| `text_data.tfidf` | C | — | — | — | test_causal_text_survival.py: text clusters recover the two groups | used to build the clustering features; its own output is not compared |
| `text_data.lda_topics` | C | — | — | — | test_causal_text_survival.py: LDA returns topic shares summing to 1 | shares sum to 1 only |
| `text_data.embed` | C | — | — | — | test_v2501.py: embed without sentence-transformers | only the missing-extra message |
| `text_data.choose_k` | D | — | — | — | — |  |
| `text_data.cluster_texts` | B | two known groups of documents | adjusted Rand index > 0.9 | — | test_causal_text_survival.py: text clusters recover the two groups |  |
| `text_data.train_classifier` | B | separable classes | macro F1 > 0.95 | — | test_causal_text_survival.py: classifier learns separable classes |  |
| `text_data.validate_coding` | B | a coding rule applied without error | kappa > 0.99 | — | test_causal_text_survival.py: validate_coding perfect on clean rule |  |
| `text_data.rogan_gladen` | A | Rogan-Gladen formula (p + Sp − 1)/(Se + Sp − 1) | 1e-9 | not measured (closed form in the test) | test_causal_text_survival.py: Rogan-Gladen corrects prevalence |  |
| `text_data.similarity_to_reference` | B | in-group and out-group documents | in-group mean similarity higher | — | test_causal_text_survival.py: similarity higher for in-group docs |  |
| `text_data.label_probs_from_logits` | A | softmax after a −inf mask, computed in the test | numpy allclose defaults | not measured (closed form in the test) | test_llm_annotation_segments.py: restricted softmax | tolerance looser than the v2.6.7 rule (numpy defaults) |
| `text_data.logtoku_uncertainty` | A | LogTokU equations 4, 5 and 9 computed by hand | numpy isclose defaults | not measured (closed form in the test) | test_llm_annotation_segments.py: LogTokU aleatoric uncertainty matches Eq. 4 | tolerance looser than the v2.6.7 rule (numpy defaults) |
| `text_data.llm_classify` | B | fake logits with known labels | exact labels; probabilities sum to 1 | — | test_llm_annotation_segments.py: llm_classify returns the right labels |  |
| `text_data.hf_backend` | C | — | — | — | test_v2501.py: hf_backend without transformers/torch | only the missing-extra message |
| `text_data.dsl_sampling_probs` | B | expected sample size 300 of 2000; floor and cap | sum equal (isclose); bounds 1e-12 | — | test_llm_annotation_segments.py: inclusion probabilities sum to the expected sample size |  |
| `text_data.draw_validation_sample` | B | Bernoulli draws with known inclusion probabilities | mean size within 3 of 300 | — | test_llm_annotation_segments.py: Bernoulli draws hit the expected size |  |
| `text_data.dsl_regression` | A | OLS with HC0 on the true data when every document is hand-coded (statsmodels) | numpy allclose defaults | not measured (software run in the test) | test_llm_annotation_segments.py: all documents hand-coded (pi = 1) -> OLS on the true data | unbiased and CI coverage in [0.92, 0.98] by simulation; tolerance looser than the v2.6.7 rule (numpy defaults) |

## `ts_univariate`

| function | level | reference | tolerance | largest observed difference | test | note |
|---|---|---|---|---|---|---|
| `ts_univariate.ts_identify` | A | statsmodels acf; white-noise band 1.96/√n | numpy allclose defaults; 1e-6 | not measured (software run in the test) | test_v246.py: ts_identify: ACF = statsmodels acf | tolerance looser than the v2.6.7 rule (numpy defaults) |
| `ts_univariate.detrend` | A | statsmodels hpfilter; Hamilton regression by statsmodels OLS | numpy allclose defaults | not measured (software run in the test) | test_v246.py: detrend hp: λ = 1600 for quarterly data and trend = statsmodels hpfilter | tolerance looser than the v2.6.7 rule (numpy defaults) |
| `ts_univariate.adf_lags` | A | AIC minimum and the t ≥ 1.645 rule computed by hand (statsmodels OLS) | exact lag | not measured (software run in the test) | test_v246.py: adf_lags default (AIC) = the AIC minimum computed by hand |  |
| `ts_univariate.adf_bootstrap` | A | statsmodels adfuller statistic | numpy isclose defaults | not measured (software run in the test) | test_v246.py: adf_bootstrap τ = statsmodels adfuller | size <= 15% in 60 replications (below the 200-rep rule); tolerance looser than the v2.6.7 rule (numpy defaults) |
| `ts_univariate.ar_bootstrap_forecast` | A | statsmodels AutoReg forecasts and prediction SE | numpy allclose defaults; SE relative 0.02 | not measured (software run in the test) | test_v246.py: ar_bootstrap_forecast: point forecasts = statsmodels AutoReg | interval coverage >= 0.88 in 120 reps; tolerance looser than the v2.6.7 rule (numpy defaults) |
| `ts_univariate.sup_wald` | A | Chow F at the estimated date (statsmodels OLS); statsmodels breaks_cusumolsresid | numpy isclose defaults | not measured (software run in the test) | test_v246.py: sup_wald: sup-F = Chow F at the estimated date | tolerance looser than the v2.6.7 rule (numpy defaults) |
| `ts_univariate.long_memory` | B | ARFIMA(0, 0.3, 0) built by its MA(∞) weights; white noise | \|d − 0.3\| < 0.05; \|d\| < 0.05 | — | test_v246.py: long_memory recovers d = 0.3 |  |
| `ts_univariate.frac_diff` | A | (1 − L)^(−d)(1 − L)^d x = x; d = 1 is the first difference | numpy allclose defaults | not measured (closed form in the test) | test_v246.py: frac_diff | tolerance looser than the v2.6.7 rule (numpy defaults) |
| `ts_univariate.arfima` | B | ARFIMA with d = 0.3 | \|d − 0.3\| < 0.1 | — | test_v246.py: arfima |  |
| `ts_univariate.combine_forecasts` | A | Bates-Granger closed form; constrained least squares by its Lagrangian | numpy isclose/allclose defaults | not measured (closed form in the test) | test_v246.py: combine min_variance: Bates-Granger closed form | tolerance looser than the v2.6.7 rule (numpy defaults) |

## `volatility`

| function | level | reference | tolerance | largest observed difference | test | note |
|---|---|---|---|---|---|---|
| `volatility.stylized_facts` | A | scipy kurtosis | numpy isclose defaults | not measured (software run in the test) | test_v247.py: stylized_facts | tolerance looser than the v2.6.7 rule (numpy defaults) |
| `volatility.sign_bias_test` | D | — | — | — | — | exercised through garch_fit (its sign_bias block is tested in test_v247.py) |
| `volatility.garch_fit` | A | asymmetry LR from arch directly; persistence SE √(w'Σw) from arch's covariance | 1e-6 (LR); relative 1e-4 (SE) (SE looser than the 1e-6 rule for other software) | not measured (software run in the test) | test_v2471.py: garch_fit: asymmetry LR = 2·(llf GJR − llf GARCH) computed with arch directly; test_v2472.py: garch_fit GARCH: persistence SE = √(Var α + Var β + 2 Cov) from arch's covariance | persistence CI coverage in [0.90, 0.99] over 200 simulations; tolerance looser than the v2.6.7 rule |
