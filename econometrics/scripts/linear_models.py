"""
Classical linear-model toolkit (Faraway, *Linear Models with R*; Dalpiaz, *Applied Statistics
with R*): analysis of variance, robust and quantile regression, exact permutation inference,
lack-of-fit and transformation checks, shrinkage, and inference after variable selection.

Experiments and groups
    anova_table(model, typ)                 ANOVA (type I/II/III) with partial eta-squared; warns on
                                            unbalanced designs with type I
    pairwise(df, y, group, method)          Tukey HSD or Welch t-tests with Holm/Bonferroni/BH adjustment
    ancova(df, y, group, covariate)         adjusted group means + test of parallel slopes

Robustness of the fit
    robust_regression(df, formula, norm)    Huber / Tukey bisquare M-estimation vs OLS, down-weighted rows
    quantile_regression(df, formula, taus)  coefficients across quantiles + bootstrap test of equal slopes
    permutation_test(df, formula, coef)     Freedman-Lane permutation p-value (no normality needed)
    lack_of_fit_test(model, df, x_cols)     pure-error F test when x values repeat
    boxcox_response(df, formula)            profile likelihood for the Box-Cox lambda with a 95% CI
    partial_residuals(model, df, var)       component-plus-residual data + curvature test for one regressor

Many regressors
    shrinkage_compare(df, y, X)             OLS, ridge, lasso, elastic net, PCR, PLS by cross-validated RMSE
    post_double_selection(df, y, d, X)      Belloni-Chernozhukov-Hansen: valid CI for d after lasso selection
    select_subsets(df, y, X, criterion)     best subsets by BIC / AIC / adjusted R2 / Cp (descriptive)

Shrinkage and subset selection are tools for PREDICTION: their coefficients carry no valid
standard errors. For the effect of one variable with many candidate controls, use
post_double_selection (or causal.dml_plr).
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05


# =============================================================================
# Experiments and groups
# =============================================================================
def anova_table(model, typ: int = 2) -> pd.DataFrame:
    """ANOVA table for a fitted statsmodels OLS formula model, with partial eta-squared
    (SS_effect / (SS_effect + SS_residual)).

    typ=1 (sequential) depends on the order of terms when the design is unbalanced; typ=2 tests
    each main effect after the others (no interactions assumed); typ=3 needs sum-to-zero contrasts,
    e.g. C(g, Sum), to be meaningful with interactions."""
    import statsmodels.api as sm
    tab = sm.stats.anova_lm(model, typ=typ)
    ss_res = tab.loc["Residual", "sum_sq"]
    tab["partial_eta_sq"] = tab["sum_sq"] / (tab["sum_sq"] + ss_res)
    tab.loc["Residual", "partial_eta_sq"] = np.nan
    if typ == 1:
        cats = [c for c in model.model.data.frame.columns
                if c in model.model.formula and model.model.data.frame[c].dtype == object]
        for c in cats:
            if model.model.data.frame[c].value_counts().nunique() > 1:
                warnings.warn("type I ANOVA with an unbalanced design: results depend on the order of terms; "
                              "use typ=2 or typ=3")
                break
    return tab


def pairwise(df: pd.DataFrame, y: str, group: str, method: str = "tukey", alpha: float = ALPHA) -> pd.DataFrame:
    """All pairwise comparisons of group means with a family-wise (or FDR) correction.

    method "tukey" (Tukey HSD, equal-variance), or "holm" / "bonferroni" / "fdr_bh" applied to
    Welch t-tests (robust to unequal variances). Never read unadjusted pairwise p-values after an
    ANOVA: with k groups there are k(k-1)/2 tests."""
    d = df[[y, group]].dropna()
    if method == "tukey":
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        res = pairwise_tukeyhsd(d[y], d[group], alpha=alpha)
        t = pd.DataFrame(res._results_table.data[1:], columns=res._results_table.data[0])
        return t.rename(columns={"meandiff": "difference", "p-adj": "p_adj", "lower": "ci_low", "upper": "ci_high"})
    from statsmodels.stats.multitest import multipletests
    rows = []
    for a, b in itertools.combinations(sorted(d[group].unique(), key=str), 2):
        ya, yb = d.loc[d[group] == a, y], d.loc[d[group] == b, y]
        tt = stats.ttest_ind(yb, ya, equal_var=False)
        rows.append({"group1": a, "group2": b, "difference": yb.mean() - ya.mean(), "p": tt.pvalue})
    out = pd.DataFrame(rows)
    rej, padj, _, _ = multipletests(out["p"], alpha=alpha, method=method)
    out["p_adj"], out["reject"] = padj, rej
    return out


def ancova(df: pd.DataFrame, y: str, group: str, covariate: str, controls: list[str] | None = None) -> dict:
    """Analysis of covariance: group differences in y holding a continuous covariate fixed.

    1. Test parallel slopes (group x covariate interaction, F test): if rejected, the group
       difference depends on the covariate and a single adjusted difference is misleading.
    2. Adjusted means: predicted y for each group at the covariate's overall mean."""
    import statsmodels.formula.api as smf
    ctrl = " + " + " + ".join(controls) if controls else ""
    d = df[[y, group, covariate] + (controls or [])].dropna()
    m0 = smf.ols(f"{y} ~ C({group}) + {covariate}{ctrl}", d).fit()
    m1 = smf.ols(f"{y} ~ C({group}) * {covariate}{ctrl}", d).fit()
    f, p, df_diff = m1.compare_f_test(m0)
    grid = pd.DataFrame({group: sorted(d[group].unique(), key=str)})
    grid[covariate] = d[covariate].mean()
    for c in controls or []:
        grid[c] = d[c].mean()
    pr = m0.get_prediction(grid).summary_frame(alpha=ALPHA)
    adj = pd.DataFrame({"group": grid[group], "adjusted_mean": pr["mean"].to_numpy(),
                        "ci_low": pr["mean_ci_lower"].to_numpy(), "ci_high": pr["mean_ci_upper"].to_numpy()})
    verdict = ("slopes differ across groups: report the interaction model, not one adjusted difference"
               if p < ALPHA else "parallel slopes not rejected: adjusted means are interpretable")
    return {"method": "ANCOVA", "model": m0, "interaction_model": m1, "adjusted_means": adj,
            "parallel_slopes_F": float(f), "parallel_slopes_p": float(p), "verdict": verdict, "n": int(m0.nobs)}


# =============================================================================
# Robustness of the fit
# =============================================================================
def robust_regression(df: pd.DataFrame, formula: str, norm: str = "huber") -> dict:
    """M-estimation that down-weights large residuals (Huber: monotone, efficient under normality;
    Tukey bisquare: redescending, ignores gross outliers but can have several solutions).
    Compare with OLS: a large difference means a few observations drive the OLS fit; find them
    (res["downweighted"]) and check whether they are errors before choosing an estimator.
    Robust regression protects against outliers in y, not against high-leverage x values."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    norms = {"huber": sm.robust.norms.HuberT(), "bisquare": sm.robust.norms.TukeyBiweight()}
    if norm not in norms:
        raise ValueError("norm must be 'huber' or 'bisquare'")
    ols = smf.ols(formula, df).fit(cov_type="HC3")
    rlm = smf.rlm(formula, df, M=norms[norm]).fit()
    tab = pd.DataFrame({"ols": ols.params, "ols_se": ols.bse, "robust": rlm.params, "robust_se": rlm.bse})
    tab["difference_in_ols_se"] = (tab["robust"] - tab["ols"]) / tab["ols_se"]
    w = pd.Series(rlm.weights, index=rlm.model.data.row_labels)
    return {"method": f"robust regression ({norm})", "table": tab, "weights": w,
            "downweighted": w[w < 0.5].sort_values(), "ols": ols, "rlm": rlm, "n": int(rlm.nobs)}


def quantile_regression(df: pd.DataFrame, formula: str, taus=(0.1, 0.25, 0.5, 0.75, 0.9), B: int = 200,
                        seed: int = 0, test_coef: str | None = None) -> dict:
    """Linear quantile regressions (Koenker & Bassett 1978) across taus.

    Coefficients that change with tau mean the regressor shifts the spread or shape of the
    distribution, not only its center (e.g. scenario analysis: effect on the 10th and 90th
    percentiles of demand). Reports, for test_coef (default: the first slope), a bootstrap test
    that its coefficient is equal at the lowest and highest tau (rows resampled jointly).
    statsmodels' default (kernel-based) SEs are slightly optimistic in the tails (in the skill's
    simulations, ~90% coverage of nominal 95% CIs at tau = 0.1 with n = 3000): for extreme taus
    in a headline result, also report a pairs-bootstrap CI (estimation.bootstrap)."""
    import statsmodels.formula.api as smf
    taus = list(taus)
    fits = {t: smf.quantreg(formula, df).fit(q=t) for t in taus}
    names = list(fits[taus[0]].params.index)
    test_coef = test_coef or next(n for n in names if n != "Intercept")
    rows = []
    for t, f in fits.items():
        ci = f.conf_int(ALPHA)
        for n in names:
            rows.append({"tau": t, "term": n, "estimate": f.params[n], "se": f.bse[n],
                         "ci_low": ci.loc[n, 0], "ci_high": ci.loc[n, 1]})
    tab = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    lo, hi = min(taus), max(taus)
    diff0 = fits[hi].params[test_coef] - fits[lo].params[test_coef]
    diffs = []
    d = df.reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(B):
            s = d.iloc[rng.integers(0, len(d), len(d))]
            try:
                diffs.append(smf.quantreg(formula, s).fit(q=hi).params[test_coef]
                             - smf.quantreg(formula, s).fit(q=lo).params[test_coef])
            except Exception:
                pass
    se = float(np.std(diffs, ddof=1))
    z = diff0 / se
    test = {"estimand": f"{test_coef}: tau {hi} minus tau {lo}", "estimate": float(diff0), "se": se,
            "ci_low": float(diff0 - 1.96 * se), "ci_high": float(diff0 + 1.96 * se),
            "p": float(2 * stats.norm.sf(abs(z))), "n": len(d)}
    return {"method": "quantile regression", "table": tab, "fits": fits, "equality_test": test}


def permutation_test(df: pd.DataFrame, formula: str, coef: str, n_perm: int = 4999, seed: int = 0) -> dict:
    """Freedman-Lane permutation test of H0: coef = 0 in an OLS model (Faraway 3.3).

    Residuals of the model WITHOUT coef are permuted and added back to its fitted values; the
    full model is re-fit and the t statistic recomputed. Exact for exchangeable errors, no
    normality needed: use in small samples or with heavy-tailed errors. Not valid for clustered
    or time-series data (permute blocks instead)."""
    import statsmodels.formula.api as smf
    base = smf.ols(formula, df).fit()
    names = list(base.model.exog_names)
    if coef not in names:
        raise KeyError(f"{coef!r} not in {names}")
    X = base.model.exog
    y = base.model.endog
    k = names.index(coef)
    Xr = np.delete(X, k, axis=1)
    br = np.linalg.lstsq(Xr, y, rcond=None)[0]
    fit_r, e_r = Xr @ br, y - Xr @ br
    XtX_inv = np.linalg.inv(X.T @ X)
    n, K = X.shape

    def tstat(yy):
        b = XtX_inv @ X.T @ yy
        e = yy - X @ b
        return b[k] / np.sqrt(e @ e / (n - K) * XtX_inv[k, k])

    t0 = tstat(y)
    rng = np.random.default_rng(seed)
    E = np.column_stack([e_r[rng.permutation(n)] for _ in range(n_perm)])
    Yp = fit_r[:, None] + E
    Bp = XtX_inv @ X.T @ Yp
    Ep = Yp - X @ Bp
    tp = Bp[k] / np.sqrt((Ep ** 2).sum(0) / (n - K) * XtX_inv[k, k])
    p = (1 + np.sum(np.abs(tp) >= abs(t0))) / (n_perm + 1)
    return {"estimand": coef, "method": "Freedman-Lane permutation test", "estimate": float(base.params[coef]),
            "t": float(t0), "p": float(p), "p_classical": float(base.pvalues[coef]), "n_perm": n_perm, "n": n}


def lack_of_fit_test(model, df: pd.DataFrame, x_cols: list[str]) -> dict:
    """Pure-error lack-of-fit F test (Faraway 6.3): needs repeated x values (replicates).
    SS_residual = SS_pure_error (variation among rows with identical x) + SS_lack_of_fit.
    A small p-value says the mean function is wrong (missing curvature or interactions)."""
    rows = model.model.data.row_labels
    d = df.loc[rows, x_cols].copy()
    d["_e"] = model.resid.to_numpy() if hasattr(model.resid, "to_numpy") else model.resid
    d["_y"] = model.model.endog
    g = d.groupby(x_cols, observed=True)["_y"]
    c = g.ngroups
    n = len(d)
    p = int(model.df_model + 1)
    if c >= n:
        raise ValueError("no replicated x values: the pure-error test is not available (use RESET or partial residuals)")
    sspe = float(((d["_y"] - g.transform("mean")) ** 2).sum())
    sslf = float(model.ssr - sspe)
    if c <= p:
        raise ValueError("as many distinct x values as parameters: lack of fit cannot be tested")
    F = (sslf / (c - p)) / (sspe / (n - c))
    pv = stats.f.sf(F, c - p, n - c)
    return {"method": "pure-error lack-of-fit test", "F": float(F), "df": (c - p, n - c), "p": float(pv),
            "ss_pure_error": sspe, "ss_lack_of_fit": sslf, "distinct_x": c, "n": n,
            "verdict": "mean function inadequate" if pv < ALPHA else "no evidence of lack of fit"}


def boxcox_response(df: pd.DataFrame, formula: str, lambdas=None) -> dict:
    """Profile log-likelihood of the Box-Cox lambda for y (y > 0) in a linear model, with the
    Jacobian term, and a 95% CI from the chi2(1) likelihood-ratio cut-off (Faraway 7.1).
    Read it for a sensible transformation, not an exact lambda: CI containing 0 -> log(y);
    containing 1 -> no transformation; 0.5 -> square root."""
    import statsmodels.formula.api as smf
    base = smf.ols(formula, df).fit()
    y = base.model.endog
    X = base.model.exog
    if np.any(y <= 0):
        raise ValueError("Box-Cox needs y > 0")
    from scipy import optimize
    lambdas = np.linspace(-2, 2, 161) if lambdas is None else np.asarray(lambdas, float)
    n = len(y)
    sly = np.log(y).sum()

    def prof(lam):
        z = np.log(y) if abs(lam) < 1e-12 else (y ** lam - 1) / lam
        b = np.linalg.lstsq(X, z, rcond=None)[0]
        return -n / 2 * np.log(np.sum((z - X @ b) ** 2) / n) + (lam - 1) * sly

    ll = np.array([prof(lam) for lam in lambdas])
    i = int(np.argmax(ll))
    a, b_ = lambdas[max(i - 1, 0)], lambdas[min(i + 1, len(lambdas) - 1)]
    opt = optimize.minimize_scalar(lambda v: -prof(v), bounds=(a, b_), method="bounded")
    lam_hat, ll_max = float(opt.x), -float(opt.fun)
    cut = ll_max - stats.chi2.ppf(0.95, 1) / 2
    g = lambda v: prof(v) - cut
    lo = optimize.brentq(g, lambdas[0], lam_hat) if g(lambdas[0]) < 0 else float(lambdas[0])
    hi = optimize.brentq(g, lam_hat, lambdas[-1]) if g(lambdas[-1]) < 0 else float(lambdas[-1])
    if lo <= 0 <= hi:
        sug = "log(y)"
    elif lo <= 1 <= hi:
        sug = "no transformation"
    elif lo <= 0.5 <= hi:
        sug = "sqrt(y)"
    else:
        sug = f"y^{lam_hat:.2f} (no simple power inside the CI)"
    return {"method": "Box-Cox profile likelihood", "lambda_hat": lam_hat, "ci_low": float(lo), "ci_high": float(hi),
            "suggestion": sug, "profile": pd.DataFrame({"lambda": lambdas, "loglik": ll}), "n": n}


def partial_residuals(model, df: pd.DataFrame, var: str) -> dict:
    """Component-plus-residual (partial residual) data for one regressor, e + b*x: plotted
    against x, curvature means the linear term is wrong. Also a test: add x^2 and report its
    p-value (HC3)."""
    import statsmodels.formula.api as smf
    rows = model.model.data.row_labels
    x = df.loc[rows, var].to_numpy(float)
    pr = model.resid.to_numpy() + model.params[var] * x
    f2 = model.model.formula + f" + I({var}**2)"
    m2 = smf.ols(f2, df.loc[rows]).fit(cov_type="HC3")
    term = f"I({var} ** 2)"
    p = float(m2.pvalues[term])
    return {"data": pd.DataFrame({var: x, "partial_residual": pr}, index=rows), "curvature_p": p,
            "verdict": "curvature: try a log, a quadratic or a spline" if p < ALPHA else "linear term adequate"}


# =============================================================================
# Many regressors
# =============================================================================
def shrinkage_compare(df: pd.DataFrame, y: str, X: list[str], cv: int = 10, seed: int = 0) -> dict:
    """Cross-validated out-of-sample RMSE of OLS, ridge, lasso, elastic net, principal
    components regression and partial least squares (Faraway ch. 9). Tuning parameters are
    chosen INSIDE each training fold (nested CV), so the comparison is honest.
    For prediction only: penalized coefficients are biased by design and have no valid SEs."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.decomposition import PCA
    from sklearn.linear_model import ElasticNetCV, LassoCV, LinearRegression, RidgeCV
    from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    d = df[[y] + X].dropna()
    Xm, Y = d[X].to_numpy(float), d[y].to_numpy(float)
    p = Xm.shape[1]
    inner = KFold(5, shuffle=True, random_state=seed)
    outer = KFold(cv, shuffle=True, random_state=seed + 1)
    comps = list(range(1, min(p, 15) + 1))
    models = {
        "OLS": make_pipeline(StandardScaler(), LinearRegression()),
        "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 50))),
        "lasso": make_pipeline(StandardScaler(), LassoCV(cv=inner, random_state=seed, max_iter=20000)),
        "elastic net": make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8], cv=inner,
                                                                    random_state=seed, max_iter=20000)),
        "PCR": GridSearchCV(make_pipeline(StandardScaler(), PCA(), LinearRegression()),
                            {"pca__n_components": comps}, cv=inner, scoring="neg_root_mean_squared_error"),
        "PLS": GridSearchCV(make_pipeline(StandardScaler(), PLSRegression()),
                            {"plsregression__n_components": comps}, cv=inner, scoring="neg_root_mean_squared_error"),
    }
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, m in models.items():
            s = -cross_val_score(m, Xm, Y, cv=outer, scoring="neg_root_mean_squared_error")
            m.fit(Xm, Y)
            info = ""
            if name == "lasso":
                coef = m[-1].coef_
                info = f"{int(np.sum(coef != 0))} of {p} coefficients non-zero"
            elif name == "ridge":
                info = f"alpha = {m[-1].alpha_:.3g}"
            elif name in ("PCR", "PLS"):
                info = f"{list(m.best_params_.values())[0]} components"
            rows.append({"model": name, "cv_rmse": s.mean(), "cv_rmse_se": s.std(ddof=1) / np.sqrt(cv), "tuning": info})
    tab = pd.DataFrame(rows).set_index("model").sort_values("cv_rmse")
    lasso = models["lasso"]
    selected = [v for v, c in zip(X, lasso[-1].coef_) if c != 0]
    return {"method": "nested cross-validated comparison", "table": tab, "lasso_selected": selected,
            "models": models, "n": len(d)}


def post_double_selection(df: pd.DataFrame, y: str, d: str, X: list[str], seed: int = 0,
                          cluster: str | None = None) -> dict:
    """Effect of d on y with many candidate controls (Belloni, Chernozhukov & Hansen 2014):
    1. lasso of y on X; 2. lasso of d on X; 3. OLS of y on d and the UNION of selected controls,
    with HC3 (or clustered) SEs. Selecting only on step 1 (the naive approach) drops confounders
    that predict d strongly but y weakly and biases the estimate; the second selection protects
    against that, which is why the CI is valid. Penalties are chosen by cross-validation here
    (the original paper uses a plug-in rule; CV tends to keep a few more controls)."""
    import statsmodels.api as sm
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler
    cols = [y, d] + X + ([cluster] if cluster else [])
    dd = df[cols].dropna()
    Z = StandardScaler().fit_transform(dd[X].to_numpy(float))
    sel = set()
    picks = {}
    for target in (y, d):
        las = LassoCV(cv=5, random_state=seed, max_iter=20000).fit(Z, dd[target].to_numpy(float))
        picks[target] = [v for v, c in zip(X, las.coef_) if abs(c) > 1e-10]
        sel |= set(picks[target])
    controls = [v for v in X if v in sel]
    R = sm.add_constant(dd[[d] + controls].astype(float))
    if cluster:
        fit = sm.OLS(dd[y].astype(float), R).fit(cov_type="cluster", cov_kwds={"groups": dd[cluster]})
    else:
        fit = sm.OLS(dd[y].astype(float), R).fit(cov_type="HC3")
    ci = fit.conf_int(ALPHA).loc[d]
    return {"estimand": f"effect of {d}", "method": "post-double-selection lasso", "estimate": float(fit.params[d]),
            "se": float(fit.bse[d]), "ci_low": float(ci[0]), "ci_high": float(ci[1]), "p": float(fit.pvalues[d]),
            "n": int(fit.nobs), "selected_for_y": picks[y], "selected_for_d": picks[d], "controls_used": controls,
            "model": fit}


def select_subsets(df: pd.DataFrame, y: str, X: list[str], criterion: str = "bic", top: int = 10,
                   max_vars: int = 15) -> pd.DataFrame:
    """Best-subset search over all combinations of X (up to max_vars candidates), ranked by
    BIC, AIC, adjusted R2 or Mallows' Cp (Faraway 8.3). Descriptive: p-values of the chosen
    model are too small because the same data chose it (post-selection inference). BIC favours
    smaller models than AIC; use it when the goal is the true model, AIC/CV when it is prediction."""
    import statsmodels.api as sm
    if len(X) > max_vars:
        raise ValueError(f"{len(X)} candidates is too many for exhaustive search; use shrinkage_compare")
    d = df[[y] + X].dropna()
    Y = d[y].to_numpy(float)
    full = sm.OLS(Y, sm.add_constant(d[X].astype(float))).fit()
    s2 = full.mse_resid
    n = len(d)
    rows = []
    for k in range(0, len(X) + 1):
        for combo in itertools.combinations(X, k):
            Xm = sm.add_constant(d[list(combo)].astype(float)) if combo else np.ones((n, 1))
            f = sm.OLS(Y, Xm).fit()
            p = k + 1
            rows.append({"variables": ", ".join(combo) or "(intercept only)", "k": k, "bic": f.bic, "aic": f.aic,
                         "adj_r2": f.rsquared_adj, "cp": f.ssr / s2 - n + 2 * p})
    tab = pd.DataFrame(rows)
    asc = criterion != "adj_r2"
    key = {"bic": "bic", "aic": "aic", "adj_r2": "adj_r2", "cp": "cp"}[criterion]
    return tab.sort_values(key, ascending=asc).head(top).reset_index(drop=True)
