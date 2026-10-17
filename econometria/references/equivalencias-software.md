# Equivalências Stata / EViews → Python

Grande parte do material didático de econometria (livros, cursos, listas de exercício) usa Stata e EViews. Esta tabela traduz os comandos para Python para reproduzir os procedimentos.

Importações usadas abaixo:
```python
import statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.stats import diagnostic as dg
from linearmodels.iv import IV2SLS, IVLIML, IVGMM
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS
```

## Regressão e testes

| Stata | EViews | Python |
|---|---|---|
| `regress y x1 x2` | `ls y c x1 x2` | `smf.ols("y ~ x1 + x2", df).fit()` |
| `regress ..., vce(robust)` | Options → White | `.fit(cov_type="HC1")` (HC1 = Stata; HC3 preferível) |
| `vce(hc3)` | — | `.fit(cov_type="HC3")` |
| `vce(cluster id)` | — | `.fit(cov_type="cluster", cov_kwds={"groups": df["id"]})` |
| Newey-West | Options → HAC | `.fit(cov_type="HAC", cov_kwds={"maxlags": L})` |
| `test x1 x2` | Wald → coefficient restrictions | `res.f_test("x1 = 0, x2 = 0")` |
| `test x1 = x2` | `c(2)=c(3)` | `res.f_test("x1 = x2")` |
| `estat ovtest` | Stability diagnostics → Ramsey RESET | `dg.linear_reset(res, power=3, use_f=True)` |
| `estat hettest` | Heteroskedasticity → Breusch-Pagan | `dg.het_breuschpagan(res.resid, res.model.exog)` |
| `estat imtest, white` | Heteroskedasticity → White | `dg.het_white(res.resid, res.model.exog)` |
| `estat bgodfrey, lags(p)` | Serial correlation LM test | `dg.acorr_breusch_godfrey(res, nlags=p)` |
| `estat dwatson` | (no output) | `statsmodels.stats.stattools.durbin_watson(res.resid)` |
| `estat vif` | — | `statsmodels.stats.outliers_influence.variance_inflation_factor` |
| `dfbeta`, `predict, cooksd` | Influence statistics | `res.get_influence().summary_frame()` |
| `predict r, resid` / `predict yhat` | `resid` / forecast | `res.resid` / `res.fittedvalues` |
| `prais y x` | AR(1) no erro | `sm.GLSAR(y, X, rho=1).iterative_fit()` |
| WLS `[aweight=w]` | Weighted LS | `smf.wls("y ~ x", df, weights=w).fit()` |

## IV e equações simultâneas

| Stata | EViews | Python |
|---|---|---|
| `ivregress 2sls y (x = z1 z2) w, vce(robust)` | `tsls y c x w @ c z1 z2 w` | `IV2SLS.from_formula("y ~ 1 + w + [x ~ z1 + z2]", df).fit(cov_type="robust")` |
| `..., first` | — | `res.first_stage` |
| `estat endogenous` | Endogeneity test | `res.wu_hausman()`, `res.durbin()` |
| `estat overid` | Instrument orthogonality | `res.sargan`; robusto: `IVGMM(...).fit().j_stat` |
| `ivregress liml` | LIML | `IVLIML.from_formula(...)` |
| `ivregress gmm` | GMM | `IVGMM.from_formula(...)` |
| `reg3` / `sureg` | System | `linearmodels.system.IV3SLS`, `SUR` |

## Painel

| Stata | Python |
|---|---|
| `xtset id t` | `df.set_index(["id", "t"])` |
| `reshape long` | `pd.wide_to_long` / `df.melt` |
| `xtsum` (between/within) | ver `painel-iv.md`, seção 6 |
| `regress ..., vce(cluster id)` (pooled) | `PooledOLS.from_formula(...).fit(cov_type="clustered", cluster_entity=True)` |
| `xtreg ..., fe vce(cluster id)` | `PanelOLS.from_formula("y ~ x + EntityEffects", df).fit(cov_type="clustered", cluster_entity=True)` |
| `xtreg ..., re` | `RandomEffects.from_formula("y ~ 1 + x", df).fit()` |
| `hausman FE RE, sigmamore` | `scripts/diagnosticos.py hausman` (ou Mundlak) |
| `xttest0` (LM de Breusch-Pagan) | `scripts/diagnosticos.py lm_breusch_pagan_painel` |
| `xtabond ..., twostep vce(robust) pre() endogenous()` | pacote `pydynpd` |

## Escolha qualitativa e limitada

| Stata | EViews | Python |
|---|---|---|
| `probit` / `logit` | Binary choice | `smf.probit(...)`, `smf.logit(...)` |
| `margins, dydx(*)` | — | `res.get_margeff(at="overall").summary()` |
| `margins, dydx(x) atmeans` | — | `res.get_margeff(at="mean")` |
| `margins, dydx(x) at(...)` | — | `res.predict` em perfis e diferença (ver `dependente-limitada.md`) |
| `margins` com `i.var` (efeito discreto) | — | `get_margeff(..., dummy=True)` |
| `lrtest A B` | LR test | `2*(A.llf - B.llf)` vs χ² |
| `constraint` + `lrtest` | — | estimar restrito reparametrizando + LR |
| `estat classification, cutoff(c)` | Expectation-prediction table | `res.pred_table(threshold=c)` |
| `tobit y x, ll(0)` | Censored | `scripts/diagnosticos.py tobit` |
| `heckman` | — | dois passos manual (ver `dependente-limitada.md`) |
| `poisson`, `nbreg` | Count | `smf.poisson`, `smf.negativebinomial` |
| `oprobit`, `mlogit` | Ordered / multinomial | `OrderedModel`, `smf.mnlogit` |

## Séries temporais

| EViews (curso) | Python |
|---|---|
| Correlogram (FAC/FACP) | `statsmodels.graphics.tsaplots.plot_acf / plot_pacf`; `acf`, `pacf` |
| `ls y c ar(1) ma(1)` | `SARIMAX(y, order=(1,0,1), trend="c")` |
| Unit root test ADF / PP / KPSS / DF-GLS | `arch.unitroot.ADF / PhillipsPerron / KPSS / DFGLS` |
| Breakpoint test (Chow) | `scripts/diagnosticos.py chow` |
| Multiple breakpoints (Bai-Perron) | `ruptures` (detecção) |
| Recursive estimates / CUSUM | `dg.recursive_olsresiduals`, `dg.breaks_cusumolsresid` |
| Hodrick-Prescott filter | `statsmodels.tsa.filters.hp_filter.hpfilter(y, lamb=129600)` |
| ARCH/GARCH/TARCH/EGARCH | `arch.arch_model` |
| ARFIMA | GPH + filtragem fracionária (ver `series-temporais.md`) |
| VAR, lag length criteria | `VAR(dados).select_order(12)` |
| Granger causality (pairwise) | `res.test_causality(...)` ou `grangercausalitytests` |
| Impulse response (Cholesky / generalized) | `res.irf(h).orth_irfs` / fórmula de Pesaran-Shin |
| Variance decomposition | `res.fevd(h)` |
| Structural VAR (A/B, long-run) | `SVAR(..., svar_type="AB")`; BQ manual |
| Bayesian VAR (Litterman) | implementação manual |
| Cointegration test (Johansen) | `coint_johansen`, `select_coint_rank` |
| Engle-Granger | `statsmodels.tsa.stattools.coint` |
| VEC | `statsmodels.tsa.vector_ar.vecm.VECM` |
