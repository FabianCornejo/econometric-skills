# Stata / EViews → Python equivalents

Much of the econometrics teaching material (textbooks, courses, problem sets) uses Stata and EViews. This table translates the commands into Python to reproduce the procedures.

Imports used below:
```python
import statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.stats import diagnostic as dg
from linearmodels.iv import IV2SLS, IVLIML, IVGMM
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS
```

## Regression and tests

| Stata | EViews | Python |
|---|---|---|
| `regress y x1 x2` | `ls y c x1 x2` | `smf.ols("y ~ x1 + x2", df).fit()` |
| `regress ..., vce(robust)` | Options → White | `.fit(cov_type="HC1")` (HC1 = Stata; HC3 preferable) |
| `vce(hc3)` | — | `.fit(cov_type="HC3")` |
| `vce(cluster id)` | — | `.fit(cov_type="cluster", cov_kwds={"groups": df["id"]})` |
| Newey-West | Options → HAC | `.fit(cov_type="HAC", cov_kwds={"maxlags": L})` |
| `test x1 x2` | Wald → coefficient restrictions | `res.f_test("x1 = 0, x2 = 0")` |
| `test x1 = x2` | `c(2)=c(3)` | `res.f_test("x1 = x2")` |
| `estat ovtest` | Stability diagnostics → Ramsey RESET | `dg.linear_reset(res, power=3, use_f=True)` |
| `estat hettest` | Heteroskedasticity → Breusch-Pagan | `dg.het_breuschpagan(res.resid, res.model.exog)` |
| `estat imtest, white` | Heteroskedasticity → White | `dg.het_white(res.resid, res.model.exog)` |
| `estat bgodfrey, lags(p)` | Serial correlation LM test | `dg.acorr_breusch_godfrey(res, nlags=p)` |
| `estat dwatson` | (in the output) | `statsmodels.stats.stattools.durbin_watson(res.resid)` |
| `estat vif` | — | `statsmodels.stats.outliers_influence.variance_inflation_factor` |
| `dfbeta`, `predict, cooksd` | Influence statistics | `res.get_influence().summary_frame()` |
| `predict r, resid` / `predict yhat` | `resid` / forecast | `res.resid` / `res.fittedvalues` |
| `prais y x` | AR(1) error term | `sm.GLSAR(y, X, rho=1).iterative_fit()` |
| WLS `[aweight=w]` | Weighted LS | `smf.wls("y ~ x", df, weights=w).fit()` |

## IV and simultaneous equations

| Stata | EViews | Python |
|---|---|---|
| `ivregress 2sls y (x = z1 z2) w, vce(robust)` | `tsls y c x w @ c z1 z2 w` | `IV2SLS.from_formula("y ~ 1 + w + [x ~ z1 + z2]", df).fit(cov_type="robust")` |
| `ivregress 2sls ..., small` | — | add `debiased=True` to `.fit(...)` (n−k correction, t and F statistics) |
| `predict, stdp` / `predict, stdf` (CI vs. prediction interval) | forecast std. errors | `specification.prediction_intervals(res, newdata)` |
| `lincom b1 + b2 + b3` / `nlcom (_b[x]+_b[L.x])/(1-_b[L.y])` | `c(2)+c(3)` Wald | `specification.long_run_effect(res, x_terms, y_lag_terms)` |
| `reg ..., beta` | — | `specification.standardized_coefficients(res)` |
| `psacalc delta d` / `psacalc beta d` | — | `specification.oster(res_uncontrolled, res_controlled, "d")` (approximation) |
| `sensemakr` | — | `specification.sensitivity(res, "d")` |
| `xtreg ... , re` + unit means (`mundlak`, `xthybrid`) | — | `diagnostics.mundlak_test(...)` |
| `estat gof` / overdispersion after `poisson` | — | `diagnostics.count_diagnostics(res)` |
| `..., first` | — | `res.first_stage` |
| `estat endogenous` | Endogeneity test | `res.wu_hausman()`, `res.durbin()` |
| `estat overid` | Instrument orthogonality | `res.sargan`; robust: `IVGMM(...).fit().j_stat` |
| `ivregress liml` | LIML | `IVLIML.from_formula(...)` |
| `ivregress gmm` | GMM | `IVGMM.from_formula(...)` |
| `reg3` / `sureg` | System | `linearmodels.system.IV3SLS`, `SUR` |

## Panel data

| Stata | Python |
|---|---|
| `xtset id t` | `df.set_index(["id", "t"])` |
| `reshape long` | `pd.wide_to_long` / `df.melt` |
| `xtsum` (between/within) | see `panel-iv.md`, section 6 |
| `regress ..., vce(cluster id)` (pooled) | `PooledOLS.from_formula(...).fit(cov_type="clustered", cluster_entity=True)` |
| `xtreg ..., fe vce(cluster id)` | `PanelOLS.from_formula("y ~ x + EntityEffects", df).fit(cov_type="clustered", cluster_entity=True)` |
| `xtreg ..., re` | `RandomEffects.from_formula("y ~ 1 + x", df).fit()` |
| `hausman FE RE, sigmamore` | `hausman` in `scripts/diagnostics.py` (or Mundlak) |
| `xttest0` (Breusch-Pagan LM) | `panel_bp_lm` in `scripts/diagnostics.py` |
| `xtabond ..., twostep vce(robust) pre() endogenous()` | `pydynpd` package |

## Qualitative and limited dependent variables

| Stata | EViews | Python |
|---|---|---|
| `probit` / `logit` | Binary choice | `smf.probit(...)`, `smf.logit(...)` |
| `margins, dydx(*)` | — | `res.get_margeff(at="overall").summary()` |
| `margins, dydx(x) atmeans` | — | `res.get_margeff(at="mean")` |
| `margins, dydx(x) at(...)` | — | `res.predict` on profiles and take the difference (see `limited-dependent.md`) |
| `margins` with `i.var` (discrete effect) | — | `get_margeff(..., dummy=True)` |
| `lrtest A B` | LR test | `2*(A.llf - B.llf)` vs χ² |
| `constraint` + `lrtest` | — | estimate the restricted model by reparametrizing + LR |
| `estat classification, cutoff(c)` | Expectation-prediction table | `res.pred_table(threshold=c)` |
| `tobit y x, ll(0)` | Censored | `tobit` in `scripts/diagnostics.py` |
| `heckman` | — | manual two-step (see `limited-dependent.md`) |
| `poisson`, `nbreg` | Count | `smf.poisson`, `smf.negativebinomial` |
| `oprobit`, `mlogit` | Ordered / multinomial | `OrderedModel`, `smf.mnlogit` |
| R `LLM::llm`, `predict.llm` (logit leaf model) | — | `segments.logit_leaf_model`, `segments.predict_leaf_model` |
| R `partykit::glmtree` (model-based partitioning) | — | no direct equivalent; `segments.logit_leaf_model` splits by purity, not parameter instability |

## Time series

| EViews | Python |
|---|---|
| Correlogram (ACF/PACF) | `statsmodels.graphics.tsaplots.plot_acf / plot_pacf`; `acf`, `pacf` |
| `ls y c ar(1) ma(1)` | `SARIMAX(y, order=(1,0,1), trend="c")` |
| Unit root test ADF / PP / KPSS / DF-GLS | `arch.unitroot.ADF / PhillipsPerron / KPSS / DFGLS` |
| Breakpoint test (Chow) | `chow` in `scripts/diagnostics.py` |
| Multiple breakpoints (Bai-Perron) | `ruptures` (detection) |
| Recursive estimates / CUSUM | `dg.recursive_olsresiduals`, `dg.breaks_cusumolsresid` |
| Hodrick-Prescott filter | `statsmodels.tsa.filters.hp_filter.hpfilter(y, lamb=129600)` |
| ARCH/GARCH/TARCH/EGARCH | `arch.arch_model` |
| ARFIMA | GPH + fractional filtering (see `time-series.md`) |
| VAR, lag length criteria | `VAR(data).select_order(12)` |
| Granger causality (pairwise) | `res.test_causality(...)` or `grangercausalitytests` |
| Impulse response (Cholesky / generalized) | `res.irf(h).orth_irfs` / Pesaran-Shin formula |
| Variance decomposition | `res.fevd(h)` |
| Structural VAR (A/B, long-run) | `SVAR(..., svar_type="AB")`; Blanchard-Quah manually |
| Bayesian VAR (Litterman) | manual implementation |
| Cointegration test (Johansen) | `coint_johansen`, `select_coint_rank` |
| Engle-Granger | `statsmodels.tsa.stattools.coint` |
| VEC | `statsmodels.tsa.vector_ar.vecm.VECM` |
| ARIMA automatic selection | `forecasting.auto_arima` (statsmodels-only replacement for `pmdarima.auto_arima`) |
| Recursive / rolling forecasts | `forecasting.rolling_origin`, `forecasting.rolling_var` |
| Markov switching regression | `regimes.markov_switching` (`sm.tsa.MarkovRegression`) |
| State space / Kalman filter (Sspace object) | `state_space.kalman_regression`; general models: `statsmodels.tsa.statespace` |

## Risk, dependence and finance (R / MATLAB → Python)

| R / MATLAB | Python |
|---|---|
| `rugarch::ugarchroll` | `risk.var_garch` (rolling refit) |
| `rugarch::VaRTest` (Kupiec, Christoffersen) | `risk.kupiec_pof`, `risk.christoffersen`, `risk.backtest_var` |
| `copula::fitCopula`, `VineCopula::BiCopSelect` | `dependence.fit_copula`, `dependence.fit_copulas` |
| `copula::rCopula` | `dependence.simulate_copula` |
| `MSwM::msmFit`, MATLAB `msVAR` | `regimes.markov_switching` |
| `depmixS4`, MATLAB `hmmtrain` | `regimes.hmm_regimes` (hmmlearn) |
| `PerformanceAnalytics::table.AnnualizedReturns`, `quantstats` | `backtest.performance` |
| `forecast::auto.arima`, `forecast::dm.test` | `forecasting.auto_arima`, `diagnostics.diebold_mariano` |

## Causal designs, survival, text and tables (Stata → Python)

| Stata | Python |
|---|---|
| `teffects ipw`, `teffects aipw` | `causal.ipw`, `causal.aipw` |
| `teffects psmatch`, `psmatch2` | `causal.nn_match` (+ `causal.balance_table`, like `pstest`) |
| `ddml`, `pystacked` | `causal.dml_plr` |
| `rdrobust`, `rdbwselect`, `rddensity` | `rdrobust` (Python package), `causal.rdd`, `causal.select_bandwidth`, `causal.density_test` |
| `eventdd`, `reghdfe` with leads/lags | `causal.event_study` |
| `csdid` | `causal.att_gt` |
| `synth` | `causal.synth` |
| `stset` + `sts graph`, `sts test` | `survival.kaplan_meier`, `survival.logrank` |
| `stcox, vce(cluster c)`, `estat phtest` | `survival.cox(..., cluster=)`, `survival.ph_test` |
| `outreg2`, `esttab` | `reporting.regression_table` + `reporting.export_table` |
| `import delimited, delimiter(";") encoding(latin1)` | `data_prep.read_any` |
| `merge m:1 key, assert(match)` | `data_prep.safe_merge(..., validate="m:1")` |
| `regexm`, `regexs` | `text_data.regex_flags`, `text_data.regex_extract` |
| R `dsl::dsl` (design-based supervised learning) | `text_data.dsl_regression` (+ `dsl_sampling_probs`, `draw_validation_sample`) |
| Hugging Face `LogitsProcessor` restricting the next token to labels | `text_data.llm_classify` (+ `hf_backend`, `label_probs_from_logits`) |

## Estimation engine and heterogeneous effects (Stata / R / EconML → Python)

| Stata / R / EconML | Python (this skill) |
|---|---|
| Stata `ml model lf`, R `maxLik`, statsmodels `GenericLikelihoodModel` | `estimation.mle(loglik, theta0, args, cov_type=...)` |
| `ml ... vce(robust)`, `vce(cluster c)` | `estimation.mle(..., cov_type="sandwich" / "cluster", cluster=...)` |
| `lrtest`, `test`, `testnl` | `estimation.trinity_test`, `estimation.delta_method` |
| `nl`, R `nls` | `estimation.nls` |
| `nlcom` | `estimation.delta_method` |
| `bootstrap, reps() cluster()` | `estimation.bootstrap(stat_fn, df, B, cluster=...)` |
| `boottest` (Roodman et al.), R `fwildclusterboot` | `estimation.wild_cluster_bootstrap` |
| EconML `LinearDRLearner`, `LinearDML` | `hte.cate_fit(method="dr")`, `hte.cate_fit(method="linear_dml")` |
| EconML `ForestDRLearner`, `TLearner`, `SLearner`, `XLearner` | `hte.cate_fit(method="forest" / "t_learner" / "s_learner" / "x_learner")` |
| EconML `CausalForestDML`, R `grf::causal_forest` | `econml` itself; validate its out-of-fold predictions with `hte.blp_test(res, tau=...)` |
| R `grf::test_calibration`; GenericML BLP | `hte.blp_test` |
| GenericML GATES / CLAN | `hte.gates`, `hte.clan` |
| R `grf::rank_average_treatment_effect` (AUTOC, Qini) | `hte.toc` |
| EconML `RScorer` | `hte.rscore`, `hte.compare_cate_models` |
| EconML `DRPolicyTree`, R `policytree::policy_tree` | `hte.policy_tree`, `hte.policy_value` |
