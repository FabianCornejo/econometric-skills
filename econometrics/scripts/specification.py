"""Specification and sensitivity tools for linear models (statsmodels results).

Use as a module:
    import sys; sys.path.insert(0, "<skill folder>/scripts")
    from specification import *

Functions
    fgls_het(formula, data)                      feasible GLS for heteroskedasticity of unknown form
                                                 (Wooldridge's exp-variance procedure), OLS vs FGLS table
    prediction_intervals(res, newdata)           CI for E(y|x) and prediction interval for a new y;
                                                 log_y=True adds Duan smearing and intervals in y units
    long_run_effect(res, x_terms, y_lag_terms)   long-run propensity of a distributed lag (Σβ) or ADL
                                                 multiplier Σβ/(1−Σφ), with delta-method SE
    standardized_coefficients(res)               beta coefficients (SD units), usable in reporting tables
    ovb_decomposition(data, y, focal, omitted)   exact omitted-variable bias: β̃ = β̂ + γ̂·δ̃
    oster(res_uncontrolled, res_controlled, d)   coefficient stability (Oster, 2019): β* and δ for β = 0
    robustness_value(t, dof)                     Cinelli-Hazlett (2020) partial R², RV_q, RV_q,α
    sensitivity(res, treatment)                  the same from a fitted OLS, with covariate benchmarks

All numbers come from the fitted models passed in; nothing is assumed.
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

ALPHA = 0.05
_INTERCEPT = ("Intercept", "const")


def _fmt(v):
    return f"{v:.4f}" if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v)


# =============================================================================
# Heteroskedasticity: feasible GLS
# =============================================================================
def fgls_het(formula: str, data: pd.DataFrame, cov_type: str = "HC3", show: bool = True) -> dict:
    """Feasible GLS when Var(u|x) = σ²·exp(xδ) (Wooldridge, ch. 8.4).

    1) OLS of y on x → residuals û;  2) regress log(û²) on the same x → fitted ĝ;
    3) WLS with weights 1/exp(ĝ).  The variance function is a guess, so by default the FGLS
    standard errors are also heteroskedasticity-robust (cov_type="HC3"); use "nonrobust" to
    reproduce textbook tables. FGLS gains efficiency only if the variance model is roughly right;
    a large change in the coefficients (not just the SEs) signals functional-form problems.
    """
    ols = smf.ols(formula, data).fit(cov_type=cov_type)
    names = ols.model.exog_names
    X = pd.DataFrame(ols.model.exog, columns=names)
    y = pd.Series(ols.model.endog, name=ols.model.endog_names)
    u2 = np.maximum(ols.resid.values ** 2, np.finfo(float).tiny)
    var_model = sm.OLS(np.log(u2), X).fit()
    h = np.exp(var_model.fittedvalues.values)
    fgls = sm.WLS(y, X, weights=1.0 / h).fit(cov_type=cov_type)
    table = pd.DataFrame({"OLS coef": ols.params.values, "OLS se": ols.bse.values,
                          "FGLS coef": fgls.params.values, "FGLS se": fgls.bse.values}, index=names)
    table["se ratio FGLS/OLS"] = table["FGLS se"] / table["OLS se"]
    out = {"ols": ols, "fgls": fgls, "variance_model": var_model, "weights": 1.0 / h, "table": table}
    if show:
        print(f"\nFGLS (variance model: log û² on x, R² = {var_model.rsquared:.3f}; SEs: {cov_type})")
        print(table.to_string(float_format=_fmt))
        print("SE ratio < 1: efficiency gain. Coefficients that move a lot between OLS and FGLS point to "
              "misspecification of the mean, not just of the variance.")
    return out


# =============================================================================
# Prediction: confidence interval for the mean vs prediction interval for y
# =============================================================================
def prediction_intervals(res, newdata: pd.DataFrame, alpha: float = ALPHA, log_y: bool = False) -> pd.DataFrame:
    """Point prediction, CI for E(y|x) and prediction interval (PI) for a new observation.

    The CI reflects only estimation error in β and shrinks with n; the PI adds the error variance
    σ² and does not shrink. Use the PI to forecast one unit (a store, a person), the CI for an average.
    The PI assumes homoskedastic normal errors.

    log_y=True (model for log y): also returns, in y units,
      median_y  = exp(prediction)                     (the naive back-transform, a median),
      mean_y    = exp(prediction)·mean(exp(û))        (Duan's smearing estimate of E(y|x)),
      pi_lower_y, pi_upper_y = exp(PI bounds)          (valid because exp is monotone).
    """
    sf = res.get_prediction(newdata).summary_frame(alpha=alpha)
    out = pd.DataFrame({"prediction": sf["mean"], "se_mean": sf["mean_se"],
                        "ci_lower": sf["mean_ci_lower"], "ci_upper": sf["mean_ci_upper"],
                        "pi_lower": sf["obs_ci_lower"], "pi_upper": sf["obs_ci_upper"]})
    if log_y:
        smear = float(np.mean(np.exp(np.asarray(res.resid))))
        out["median_y"] = np.exp(out["prediction"])
        out["mean_y"] = out["median_y"] * smear
        out["pi_lower_y"] = np.exp(out["pi_lower"])
        out["pi_upper_y"] = np.exp(out["pi_upper"])
        out.attrs["smearing_factor"] = smear
    return out.reset_index(drop=True)


# =============================================================================
# Distributed lags: long-run effect with delta-method standard error
# =============================================================================
def long_run_effect(res, x_terms: list[str], y_lag_terms: list[str] | None = None, alpha: float = ALPHA,
                    show: bool = False) -> dict:
    """Long-run effect of a permanent unit change in x.

    Finite distributed lag (y_lag_terms=None): long-run propensity LRP = Σβ_j over x_terms.
    ADL (y_lag_terms = lags of y): long-run multiplier Σβ_j / (1 − Σφ_k); requires Σφ < 1.
    The SE uses the delta method with the model's covariance (use HAC errors in time series).
    """
    params, V = res.params, res.cov_params()
    terms = list(x_terms) + list(y_lag_terms or [])
    missing = [t for t in terms if t not in params.index]
    if missing:
        raise KeyError(f"Terms not in the model: {missing}")
    sb = float(params[x_terms].sum())
    sphi = float(params[y_lag_terms].sum()) if y_lag_terms else 0.0
    if y_lag_terms and sphi >= 1:
        raise ValueError(f"Σφ = {sphi:.3f} ≥ 1: no finite long-run effect (unit root in y's dynamics).")
    effect = sb / (1 - sphi)
    grad = pd.Series(0.0, index=params.index)
    grad[x_terms] = 1.0 / (1 - sphi)
    if y_lag_terms:
        grad[y_lag_terms] = sb / (1 - sphi) ** 2
    se = float(np.sqrt(grad.values @ np.asarray(V) @ grad.values))
    dfr = getattr(res, "df_resid", None)
    use_t = bool(getattr(res, "use_t", True)) and dfr
    crit = stats.t.ppf(1 - alpha / 2, dfr) if use_t else stats.norm.ppf(1 - alpha / 2)
    z = effect / se
    p = 2 * (stats.t.sf(abs(z), dfr) if use_t else stats.norm.sf(abs(z)))
    out = {"effect": effect, "se": se, "ci_lower": effect - crit * se, "ci_upper": effect + crit * se,
           "p": float(p), "kind": "ADL long-run multiplier" if y_lag_terms else "long-run propensity (Σβ)"}
    if show:
        print(f"\n{out['kind']}: {effect:.4f} (se {se:.4f}), {100*(1-alpha):.0f}% CI "
              f"[{out['ci_lower']:.4f}, {out['ci_upper']:.4f}], p = {p:.4f}")
    return out


# =============================================================================
# Standardized (beta) coefficients
# =============================================================================
def standardized_coefficients(res) -> pd.DataFrame:
    """Beta coefficients: effect of a 1-SD change in x on y, in SDs of y (β·sd(x)/sd(y)).

    Computed from the model's own design matrix, so transformed regressors (logs, squares) are
    standardized as they enter the model. The SE scales the same way; t and p are unchanged.
    Useful to compare regressors measured in different units; less meaningful for dummies
    (flagged in `binary`) and for terms that move together (x and x²). Returns coef/se/p,
    so it can go straight into reporting.regression_table.
    """
    X = pd.DataFrame(res.model.exog, columns=res.model.exog_names)
    sy = float(np.std(res.model.endog, ddof=1))
    rows = {}
    for name in X.columns:
        if name in _INTERCEPT or X[name].std(ddof=1) == 0:
            continue
        k = X[name].std(ddof=1) / sy
        rows[name] = {"coef": res.params[name] * k, "se": res.bse[name] * k, "p": res.pvalues[name],
                      "binary": bool(set(np.unique(X[name])) <= {0.0, 1.0})}
    return pd.DataFrame(rows).T.astype({"coef": float, "se": float, "p": float, "binary": bool})


# =============================================================================
# Omitted variables: exact decomposition and sensitivity
# =============================================================================
def _q(c):
    return f"Q('{c}')"


def ovb_decomposition(data: pd.DataFrame, y: str, focal: str, omitted: str,
                      included: list[str] | None = None, show: bool = False) -> dict:
    """Exact omitted-variable bias: β̃_focal (short) = β̂_focal (long) + γ̂·δ̃.

    γ̂: coefficient of `omitted` in the long regression; δ̃: coefficient of `focal` in the
    regression of `omitted` on focal and the included controls. Same sample for all three.
    Answers "how much of the short-regression estimate is due to leaving `omitted` out".
    """
    inc = list(included or [])
    d = data[[y, focal, omitted] + inc].dropna()
    rhs = " + ".join([_q(focal)] + [_q(c) for c in inc])
    long_ = smf.ols(f"{_q(y)} ~ {rhs} + {_q(omitted)}", d).fit()
    short = smf.ols(f"{_q(y)} ~ {rhs}", d).fit()
    aux = smf.ols(f"{_q(omitted)} ~ {rhs}", d).fit()
    b_long, gamma, delta = long_.params[_q(focal)], long_.params[_q(omitted)], aux.params[_q(focal)]
    out = {"beta_short": float(short.params[_q(focal)]), "beta_long": float(b_long),
           "gamma": float(gamma), "delta": float(delta), "bias": float(gamma * delta), "n": int(len(d))}
    if show:
        print(f"\nOmitted-variable bias for {focal} (omitting {omitted}, N = {len(d)}):")
        print(f"  short {out['beta_short']:.4f} = long {out['beta_long']:.4f} + γ̂ {gamma:.4f} × δ̃ {delta:.4f} "
              f"(bias {out['bias']:.4f})")
    return out


def oster(res_uncontrolled, res_controlled, focal: str, r_max: float | None = None, delta: float = 1.0,
          show: bool = False) -> dict:
    """Coefficient stability (Oster, 2019, approximate estimator).

    res_uncontrolled: y on the focal variable (and controls that are not about selection, if any);
    res_controlled: adds the observed controls. Same sample.
      β* = β̃ − δ(β° − β̃)(R_max − R̃)/(R̃ − R°)      bias-adjusted effect for a given δ
      δ(β=0) = β̃(R̃ − R°) / [(β° − β̃)(R_max − R̃)]  how strong selection on unobservables must be,
                                                       relative to observables, to drive β to zero
    Default R_max = min(1.3·R̃, 1) (Oster's calibration). |δ(β=0)| > 1 is the usual robustness bar.
    This is the closed-form approximation; Stata's `psacalc` solves the exact cubic.
    """
    b0, r0 = float(res_uncontrolled.params[focal]), float(res_uncontrolled.rsquared)
    bt, rt = float(res_controlled.params[focal]), float(res_controlled.rsquared)
    if int(res_uncontrolled.nobs) != int(res_controlled.nobs):
        warnings.warn("The two regressions use different samples; Oster's bounds assume the same sample.")
    if rt <= r0:
        raise ValueError("Adding controls did not raise R²; coefficient-stability bounds are undefined.")
    rmax = min(1.3 * rt, 1.0) if r_max is None else float(r_max)
    if rmax <= rt:
        raise ValueError("r_max must exceed the R² of the controlled regression.")
    move = (b0 - bt) * (rmax - rt) / (rt - r0)
    beta_star = bt - delta * move
    delta_zero = bt / move if move != 0 else np.inf
    out = {"beta_uncontrolled": b0, "beta_controlled": bt, "r2_uncontrolled": r0, "r2_controlled": rt,
           "r_max": rmax, "delta": delta, "beta_star": beta_star, "delta_for_zero": delta_zero,
           "identified_set": tuple(sorted((bt, beta_star)))}
    if show:
        print(f"\nOster (R_max = {rmax:.3f}, δ = {delta}): β̃ = {bt:.4f} → β* = {beta_star:.4f}; "
              f"identified set [{out['identified_set'][0]:.4f}, {out['identified_set'][1]:.4f}]")
        print(f"δ needed for β = 0: {delta_zero:.2f} "
              + ("(> 1: robust by Oster's criterion)" if abs(delta_zero) > 1 else "(≤ 1: fragile)"))
    return out


def robustness_value(t: float, dof: float, q: float = 1.0, alpha: float = ALPHA) -> dict:
    """Cinelli and Hazlett (2020) sensitivity statistics from a classical OLS t-statistic.

    partial_r2: share of the outcome's residual variance explained by the treatment (R²_{Y~D|X}).
    rv_q: minimum partial R² that a confounder would need with BOTH treatment and outcome to
          reduce the estimate by 100·q% (q = 1: to zero).
    rv_qa: the same, to make the reduced estimate no longer significant at level alpha.
    """
    t, dof = abs(float(t)), float(dof)
    f = t / np.sqrt(dof)
    fq = q * f

    def _rv(x):
        return 0.5 * (np.sqrt(x ** 4 + 4 * x ** 2) - x ** 2)

    f_crit = stats.t.ppf(1 - alpha / 2, dof - 1) / np.sqrt(dof - 1)
    fqa = fq - f_crit
    if fqa < 0:
        rv_qa = 0.0
    elif fq > 1 / f_crit:                     # very large effects: closed form of the boundary case
        rv_qa = (fq ** 2 - f_crit ** 2) / (1 + fq ** 2)
    else:
        rv_qa = _rv(fqa)
    return {"partial_r2": t ** 2 / (t ** 2 + dof), "rv_q": float(_rv(fq)), "rv_qa": float(rv_qa),
            "q": q, "alpha": alpha}


def sensitivity(res, treatment: str, q: float = 1.0, alpha: float = ALPHA, show: bool = False) -> dict:
    """Cinelli-Hazlett statistics for `treatment` in a fitted OLS, with informal benchmarks.

    Uses the classical (non-robust) t, as the formulas are partial-R² identities. `benchmarks`
    lists each observed covariate's partial R² with the outcome (the dummies of a categorical
    variable or fixed effect are grouped into one joint partial R²): a confounder would need to be
    stronger than these to reach RV. For formal bounds (k times as strong as a covariate) use the
    authors' sensemakr (R/Stata; Python port PySensemakr, still early-stage).
    """
    base = res if getattr(res, "cov_type", "nonrobust") == "nonrobust" else res.model.fit()
    out = robustness_value(base.tvalues[treatment], base.df_resid, q=q, alpha=alpha)
    others = [c for c in base.params.index if c not in _INTERCEPT and c != treatment]
    groups: dict[str, list[str]] = {}
    for c in others:                              # dummies of one categorical -> one benchmark
        m = re.match(r"^(C\(.+?\))\[", c) or re.match(r"^(.+?)\[T\.", c)
        groups.setdefault(m.group(1) if m else c, []).append(c)
    bench = {}
    for gname, terms in groups.items():
        if len(terms) == 1:
            t_ = base.tvalues[terms[0]]
            bench[gname] = t_ ** 2 / (t_ ** 2 + base.df_resid)
        else:                                     # joint partial R² = qF / (qF + df) (exact for OLS)
            R = np.zeros((len(terms), len(base.params)))
            for i, t_ in enumerate(terms):
                R[i, list(base.params.index).index(t_)] = 1
            F = float(np.squeeze(base.f_test(R).fvalue))
            q = len(terms)
            bench[f"{gname} ({q} dummies, jointly)"] = q * F / (q * F + base.df_resid)
    out["benchmarks"] = pd.Series(bench, name="partial R² with outcome").sort_values(ascending=False)
    if show:
        print(f"\nSensitivity of {treatment} (Cinelli-Hazlett): partial R² = {out['partial_r2']:.3%}, "
              f"RV(q={q}) = {out['rv_q']:.2%}, RV(q={q}, α={alpha}) = {out['rv_qa']:.2%}")
        print(f"A confounder explaining {out['rv_q']:.1%} of the residual variance of both {treatment} and the "
              f"outcome would remove the estimate; {out['rv_qa']:.1%} would make it insignificant.")
        if len(out["benchmarks"]):
            print("Observed covariates (partial R² with outcome), for comparison:")
            print(out["benchmarks"].to_string(float_format=lambda v: f"{v:.2%}"))
    return out
