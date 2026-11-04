"""Binary outcomes and sample selection: the comparisons a careful analyst reports (v2.4.2).

Use as a module:
    import sys; sys.path.insert(0, "<skill folder>/scripts")
    from discrete_choice import *

Functions
    binary_effects(formula, data)        LPM (robust SEs), probit and logit side by side: average marginal
                                         effects per variable (discrete change for 0/1 variables, level vs
                                         base for factors, polynomials and interactions included), odds
                                         ratios with the exact percentage change, LR tests, pseudo-R², LPM
                                         fitted values outside [0, 1], and classification against the base
                                         rate (AUC and a cutoff at the sample share, not a fixed 0.5)
    heckman(outcome, selection, data)    Heckman two-step ("Heckit"): probit of selection, inverse Mills
                                         ratio in the outcome equation, Heckman/Greene corrected SEs, ρ and σ,
                                         and a warning when there is no exclusion restriction

Every number comes from the fitted models; references: limited-dependent.md (sections 1-4 and 6).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import _console  # noqa: F401  (Windows-safe console output, see _console.py)

ALPHA = 0.05
_INTERCEPT = ("Intercept", "const")


def _fmt(v):
    return f"{v:.4f}" if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v)


def _auc(y, p):
    """Area under the ROC curve = P(score of a random 1 > score of a random 0), ties count 1/2."""
    y = np.asarray(y, bool)
    r = stats.rankdata(p)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


def _classification(y, p, cutoff):
    y = np.asarray(y, bool)
    yhat = np.asarray(p) >= cutoff
    tp, tn = int(np.sum(yhat & y)), int(np.sum(~yhat & ~y))
    fp, fn = int(np.sum(yhat & ~y)), int(np.sum(~yhat & y))
    return {"cutoff": float(cutoff), "accuracy": (tp + tn) / len(y),
            "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
            "specificity": tn / (tn + fp) if tn + fp else np.nan, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _design(res, newdata: pd.DataFrame) -> np.ndarray:
    """Design matrix of a formula model for new data (same columns as the fit)."""
    try:
        X, _ = res._transform_predict_exog(newdata)
        return np.asarray(X, float)
    except AttributeError:                        # very old statsmodels: patsy design_info
        from patsy import dmatrix
        return np.asarray(dmatrix(res.model.data.design_info, newdata, return_type="matrix"), float)


def _formula_variables(formula: str, data: pd.DataFrame) -> list[str]:
    """Data columns named on the right-hand side, in order of appearance."""
    import re
    rhs = formula.split("~", 1)[1]
    seen = []
    for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", rhs):
        if tok in data.columns and tok not in seen:
            seen.append(tok)
    return seen


def _is_factor(s: pd.Series, formula: str, name: str) -> bool:
    import re
    return (s.dtype == object or isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == bool
            or re.search(rf"C\(\s*{re.escape(name)}\s*[,)]", formula) is not None)


def _ames(res, data: pd.DataFrame, formula: str, link: str):
    """Average marginal effects per variable, by changing the variable in the data and predicting, so that
    polynomial and interaction terms and every dummy of a factor move together. Continuous: average derivative
    (central difference); 0/1 numeric: P(x=1) − P(x=0); factor: each level against the first level. SEs by the
    delta method with the model's covariance. link: 'identity' (LPM), 'probit' or 'logit'."""
    b = res.params.to_numpy()
    V = res.cov_params().to_numpy()
    if link == "identity":
        F, f = (lambda z: z), (lambda z: np.ones_like(z))
    elif link == "probit":
        F, f = stats.norm.cdf, stats.norm.pdf
    else:
        F = lambda z: 1 / (1 + np.exp(-z))      # noqa: E731
        f = lambda z: F(z) * (1 - F(z))           # noqa: E731
    rows = {}

    def contrast(d1, d0, scale=1.0):
        X1, X0 = _design(res, d1), _design(res, d0)
        z1, z0 = X1 @ b, X0 @ b
        est = float(np.mean(F(z1) - F(z0)) / scale)
        g = (f(z1)[:, None] * X1 - f(z0)[:, None] * X0).mean(axis=0) / scale
        return est, float(np.sqrt(g @ V @ g))

    for v in _formula_variables(formula, data):
        col = data[v]
        if _is_factor(col, formula, v):
            levels = list(pd.Series(col.unique()).dropna().sort_values()) if not isinstance(col.dtype, pd.CategoricalDtype) \
                else list(col.cat.categories)
            base = levels[0]
            for lev in levels[1:]:
                est, se = contrast(data.assign(**{v: lev}), data.assign(**{v: base}))
                rows[f"{v}[{lev} vs {base}]"] = (est, se, "factor level vs base")
        elif set(pd.unique(col.dropna())) <= {0, 1}:
            est, se = contrast(data.assign(**{v: 1}), data.assign(**{v: 0}))
            rows[v] = (est, se, "0/1: discrete change")
        else:
            h = 1e-4 * max(1.0, float(col.std()))
            est, se = contrast(data.assign(**{v: col + h}), data.assign(**{v: col - h}), scale=2 * h)
            rows[v] = (est, se, "derivative")
    return rows


def binary_effects(formula: str, data: pd.DataFrame, cov_type: str = "HC1", lpm_cov: str = "HC3",
                   cutoff="base_rate", alpha: float = ALPHA, show: bool = True) -> dict:
    """Linear probability model, probit and logit for a 0/1 outcome, compared on the probability scale.

    Coefficients of probit and logit are not effects on the probability and differ in scale (logit ≈ 1.6
    probit): the table compares the LPM slopes with the average marginal effects (AME) of the two, which
    should agree in sign and roughly in size. Effects are per variable, not per column: the variable is changed
    in the data and the probabilities predicted again, so polynomial and interaction terms move with it and a
    factor (C(region) or a text column) gives each level against the first level (all observations set to that
    level vs all set to the base), not one dummy flipped with the others held. For a variable coded 0/1 the AME
    is the discrete change P(y=1|d=1) − P(y=1|d=0); for the LPM the same averages are the slopes (or their
    combination when x enters several terms). SEs by the delta method. cov_type applies to probit/logit (HC1 = QML-robust);
    the LPM is heteroskedastic by construction and always uses robust errors (lpm_cov).

    Odds ratios (logit): exp(β) multiplies the odds when x rises by one unit; the change in the odds is
    100·(exp(β) − 1)%, not 100·β% (β = 0.33 → OR 1.39 → +39% in the odds, which is not +39 percentage
    points of probability). Classification uses the sample share of ones as the cutoff by default
    (cutoff="base_rate"), and reports the accuracy of always predicting the majority class: with
    unbalanced outcomes a 0.5 cutoff can be "accurate" while never predicting a 1. AUC does not depend
    on the cutoff."""
    lpm = smf.ols(formula, data).fit(cov_type=lpm_cov)
    y = lpm.model.endog
    vals = np.unique(y)
    if not np.all(np.isin(vals, (0, 1))) or len(vals) != 2:
        raise ValueError("the outcome must be coded 0/1 with both values present")
    warn = []
    fits = {}
    for name, fn in (("probit", smf.probit), ("logit", smf.logit)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m = fn(formula, data).fit(disp=False, cov_type=cov_type, maxiter=200)
        if not m.mle_retvals.get("converged", True):
            warn.append(f"{name} did not converge")
        if any("separation" in str(c.message).lower() for c in caught) or np.any(np.abs(m.params) > 30):
            warn.append(f"{name}: possible (quasi-)complete separation; coefficients and SEs are unreliable")
        fits[name] = m
    names = [c for c in lpm.params.index if c not in _INTERCEPT]
    used = data.loc[lpm.model.data.row_labels]          # estimation sample (rows without missing values)
    eff = {name: _ames(m, used, formula, link) for name, m, link in
           (("LPM", lpm, "identity"), ("probit", fits["probit"], "probit"), ("logit", fits["logit"], "logit"))}
    table = pd.DataFrame({
        "LPM": {k: v[0] for k, v in eff["LPM"].items()}, "LPM se": {k: v[1] for k, v in eff["LPM"].items()},
        "probit AME": {k: v[0] for k, v in eff["probit"].items()}, "probit AME se": {k: v[1] for k, v in eff["probit"].items()},
        "logit AME": {k: v[0] for k, v in eff["logit"].items()}, "logit AME se": {k: v[1] for k, v in eff["logit"].items()},
        "kind": {k: v[2] for k, v in eff["logit"].items()}})
    for c in table.index:
        a, b_ = table.loc[c, "LPM"], table.loc[c, "logit AME"]
        if np.sign(a) != np.sign(b_) and min(abs(a / table.loc[c, "LPM se"]), abs(b_ / table.loc[c, "logit AME se"])) > 2:
            warn.append(f"{c}: LPM and logit AME have opposite signs and are both significant; check the specification")
    lg = fits["logit"]
    ci = lg.conf_int(alpha)
    odds = pd.DataFrame({"coef": lg.params, "odds_ratio": np.exp(lg.params),
                         "odds_change_%": 100 * (np.exp(lg.params) - 1),
                         "or_ci_low": np.exp(ci.iloc[:, 0]), "or_ci_high": np.exp(ci.iloc[:, 1]),
                         "p": lg.pvalues}).loc[names]
    fit_stats = pd.DataFrame({
        name: {"loglik": m.llf, "LR stat (all slopes = 0)": m.llr, "LR df": m.df_model, "LR p": m.llr_pvalue,
               "McFadden pseudo-R²": m.prsquared, "AIC": m.aic, "BIC": m.bic} for name, m in fits.items()}).T
    p_lpm = lpm.fittedvalues.to_numpy()
    outside = float(np.mean((p_lpm < 0) | (p_lpm > 1)))
    if outside > 0:
        warn.append(f"{outside:.1%} of LPM fitted probabilities fall outside [0, 1]")
    base = float(y.mean())
    cut = base if cutoff == "base_rate" else float(cutoff)
    cls = {}
    for name, p in (("LPM", p_lpm), ("probit", fits["probit"].predict()), ("logit", lg.predict())):
        c = _classification(y, p, cut)
        c["AUC"] = _auc(y, p)
        cls[name] = c
    cls = pd.DataFrame(cls).T
    majority = max(base, 1 - base)
    out = {"table": table, "odds_ratios": odds, "fit": fit_stats, "classification": cls, "base_rate": base,
           "majority_accuracy": majority, "lpm_outside_01": outside, "n": int(lpm.nobs), "warnings": warn,
           "models": {"lpm": lpm, **fits}}
    if show:
        print(f"\nBinary outcome: n = {out['n']}, share of ones = {base:.3f}")
        cols = ["LPM", "LPM se", "probit AME", "probit AME se", "logit AME", "logit AME se"]
        print("Effects on the probability (LPM slopes vs average marginal effects):")
        print(table[cols].to_string(float_format=_fmt))
        print("\nLogit odds ratios (change in the odds = 100·(OR − 1)%):")
        print(odds[["odds_ratio", "odds_change_%", "or_ci_low", "or_ci_high", "p"]].to_string(float_format=_fmt))
        print("\nFit:")
        print(fit_stats.to_string(float_format=_fmt))
        print(f"\nClassification at cutoff {cut:.3f}"
              f"{' (base rate)' if cutoff == 'base_rate' else ''}; always predicting the majority class "
              f"gives accuracy {majority:.3f}:")
        print(cls[["accuracy", "sensitivity", "specificity", "AUC"]].to_string(float_format=_fmt))
        for w in warn:
            print("Warning:", w)
    return out


def heckman(outcome: str, selection: str, data: pd.DataFrame, alpha: float = ALPHA, show: bool = True) -> dict:
    """Heckman (1979) two-step estimator for an outcome observed only in a selected sample.

    selection: formula of the selection probit, with the 0/1 selection indicator on the left
    ("inlf ~ educ + exper + nwifeinc + kidslt6"). outcome: formula of the outcome equation ("lwage ~ educ
    + exper"), fitted on the rows with selection = 1 after adding λ̂ = φ(wγ̂)/Φ(wγ̂), the inverse Mills
    ratio. A significant λ̂ coefficient (β_λ = ρσ) is evidence of selection bias; its t-test is valid under
    H0: ρ = 0 even with the plain OLS SE.

    Standard errors: OLS SEs of the second step are wrong (heteroskedastic errors and an estimated
    regressor); the corrected covariance is Heckman's (Greene, Econometric Analysis, eq. 19-31):
    σ̂²(X*'X*)⁻¹[X*'(I − ρ̂²Δ)X* + Q](X*'X*)⁻¹, Q = ρ̂²(X*'ΔW)V_γ(W'ΔX*), δ_i = λ_i(λ_i + w_iγ̂),
    σ̂² = e'e/n₁ + δ̄β̂_λ². ρ̂ is truncated to [−1, 1] as Stata does.

    Identification: without a variable in the selection equation that is excluded from the outcome equation,
    λ̂ is nearly collinear with the outcome regressors and identification rests on the normal functional form
    only; the function warns. Heckman is a selection model, not a Tobit: Tobit assumes one process decides
    participation and amount."""
    data = data.reset_index(drop=True)            # positional rows: a duplicate index cannot misalign the steps
    sel = smf.probit(selection, data).fit(disp=False)
    s = sel.model.endog
    if not set(np.unique(s)) <= {0.0, 1.0}:
        raise ValueError("the left side of the selection formula must be the 0/1 selection indicator")
    W_all = sel.model.exog
    sel_rows = sel.model.data.row_labels
    xb_all = W_all @ sel.params.to_numpy()
    lam_all = stats.norm.pdf(xb_all) / stats.norm.cdf(xb_all)
    idx_sel = sel_rows[s == 1]
    d = data.loc[idx_sel].copy()
    d["_mills"] = pd.Series(lam_all[s == 1], index=idx_sel)
    ols = smf.ols(f"{outcome} + _mills", d).fit()
    keep = ols.model.data.row_labels
    pos = sel_rows.get_indexer(keep)
    X = ols.model.exog
    W = W_all[pos]
    lam, xb = lam_all[pos], xb_all[pos]
    delta = lam * (lam + xb)
    n1 = X.shape[0]
    b = ols.params.to_numpy()
    b_lam = float(ols.params["_mills"])
    e = ols.resid.to_numpy()
    sigma2 = float(e @ e / n1 + delta.mean() * b_lam ** 2)
    sigma = np.sqrt(sigma2)
    rho = b_lam / sigma
    rho_c = float(np.clip(rho, -1, 1))
    XtXi = np.linalg.inv(X.T @ X)
    Vg = sel.cov_params().to_numpy()
    XDW = (X * delta[:, None]).T @ W
    Q = rho_c ** 2 * XDW @ Vg @ XDW.T
    mid = X.T @ (X * (1 - rho_c ** 2 * delta)[:, None]) + Q
    V = sigma2 * XtXi @ mid @ XtXi
    se = np.sqrt(np.diag(V))
    z = stats.norm.ppf(1 - alpha / 2)
    names = list(ols.params.index)
    tab = pd.DataFrame({"estimate": b, "se": se, "z": b / se, "p": 2 * stats.norm.sf(np.abs(b / se)),
                        "ci_low": b - z * se, "ci_high": b + z * se, "OLS se (wrong)": ols.bse.to_numpy()},
                       index=[("lambda (inverse Mills)" if c == "_mills" else c) for c in names])
    warn = []
    out_x = [c for c in names if c not in _INTERCEPT and c != "_mills"]
    sel_x = [c for c in sel.params.index if c not in _INTERCEPT]
    Xo = np.delete(X, names.index("_mills"), axis=1)
    r_out = np.linalg.matrix_rank(Xo)
    excluded = [c for j, c in enumerate(sel.params.index) if c not in _INTERCEPT
                and np.linalg.matrix_rank(np.column_stack([Xo, W[:, j]])) > r_out]   # by span, not by name
    if not excluded:
        warn.append("no exclusion restriction: every selection regressor is in the outcome equation, so the "
                    "model is identified only by the normality of the errors; results are fragile")
    r2_lam = sm.OLS(lam, np.delete(X, names.index("_mills"), axis=1)).fit().rsquared
    if r2_lam > 0.9:
        warn.append(f"λ̂ is nearly collinear with the outcome regressors (R² = {r2_lam:.2f}): imprecise estimates")
    if abs(rho) > 1:
        warn.append(f"implied ρ = {rho:.3f} is outside [−1, 1] (truncated in the SEs); consider the ML estimator")
    naive = smf.ols(outcome, d).fit()
    out = {"table": tab, "lambda": {"estimand": "β_λ = ρσ (selection bias)", "estimate": b_lam,
                                    "se": float(tab.iloc[names.index("_mills")]["se"]),
                                    "p": float(tab.iloc[names.index("_mills")]["p"])},
           "rho": rho, "sigma": sigma, "n_total": int(len(s)), "n_selected": n1, "excluded_from_outcome": excluded,
           "selection_model": sel, "second_step": ols, "naive_ols": naive, "warnings": warn,
           "method": "Heckman two-step with corrected SEs"}
    if show:
        print(f"\nHeckman two-step: {n1} selected of {len(s)}; excluded from the outcome equation: "
              f"{', '.join(excluded) if excluded else 'none'}")
        print(tab.to_string(float_format=_fmt))
        print(f"ρ = {rho:.3f}, σ = {sigma:.4f}; λ coefficient p = {out['lambda']['p']:.4f} → "
              + ("evidence of selection bias" if out["lambda"]["p"] < alpha else
                 "no evidence of selection bias (not proof of its absence)"))
        comp = pd.DataFrame({"OLS on selected": naive.params, "Heckman": tab["estimate"]}).dropna()
        print("OLS on the selected sample vs Heckman:")
        print(comp.to_string(float_format=_fmt))
        for w in warn:
            print("Warning:", w)
    return out
