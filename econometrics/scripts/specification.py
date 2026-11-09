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
import _console  # noqa: F401  (Windows-safe console output, see _console.py)

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


def _oster_inputs(res_uncontrolled, res_controlled, focal: str):
    """σ²_y, σ²_x and τ_x for the exact Oster estimator, from the controlled fit, or (None, reason).

    As in robomit 1.0.7 and psacalc: variances with denominator n − 1; τ_x is the variance of the residual of the
    focal variable regressed on the other regressors of the controlled model (with an intercept). The exact formulas
    assume the uncontrolled model is y on the focal variable alone (plus an intercept) and unweighted OLS."""
    m1, m0 = getattr(res_controlled, "model", None), getattr(res_uncontrolled, "model", None)
    if m1 is None or m0 is None or not hasattr(m1, "exog_names") or not hasattr(m0, "exog_names"):
        return None, "the fits do not expose their data (statsmodels OLS fits are needed)"
    if type(m1).__name__ != "OLS" or type(m0).__name__ != "OLS":
        return None, (f"the exact solution is for unweighted OLS ({type(m1).__name__} given); weighted and panel "
                      "versions are not implemented")
    extra = [c for c in m0.exog_names if c not in _INTERCEPT and c != focal]
    if extra:
        return None, (f"the uncontrolled model has regressors besides {focal} ({', '.join(extra)}); the exact "
                      "solution here assumes y on the focal variable alone (Stata's psacalc handles this with "
                      "mcontrol, not implemented)")
    names = list(m1.exog_names)
    X = np.asarray(m1.exog, float)
    y = np.asarray(m1.endog, float)
    j = names.index(focal)
    x = X[:, j]
    W = np.delete(X, j, axis=1)
    if not any(np.allclose(W[:, k], W[0, k]) for k in range(W.shape[1])):
        W = np.column_stack([np.ones(len(x)), W])
    resid = x - W @ np.linalg.lstsq(W, x, rcond=None)[0]
    return {"sigma_yy": float(np.var(y, ddof=1)), "sigma_xx": float(np.var(x, ddof=1)),
            "tau_x": float(np.var(resid, ddof=1))}, None


def _oster_exact_beta(b0, b1, r0, r1, rmax, delta, syy, sxx, tx):
    """β* by the exact solution (Oster 2019, Proposition 2), as robomit 1.0.7 (based on psacalc) computes it.

    Unknown ν = β̃ − β*. δ = 1: a quadratic, two roots. δ ≠ 1: a cubic (coefficients A, B, C as in robomit),
    three real roots by the trigonometric formula when the discriminant is negative, else one by Cardano.
    Choice of root, exactly as robomit: the β* closest to β̃, unless it moves against the direction β° → β̃
    (Oster's Assumption 3); with two roots the other one is then taken, with three the violating roots go last.
    Cardano (one real root) applies no check. δ ≠ 1 with τ_x ≈ σ²_x (controls unrelated to x) makes the cubic
    degenerate; robomit divides by zero there, and this function solves the remaining quadratic instead.
    Returns (β*, other real roots)."""
    rt_ro = (r1 - r0) * syy
    rm_rt = (rmax - r1) * syy
    d = b0 - b1

    viol = lambda v: np.sign(v - b1) != np.sign(b1 - b0)   # moves against the direction β° → β̃

    def pick2(s1, s2):                       # robomit, δ = 1: the closer root, swapped if it violates the sign
        best, other = (s1, s2) if (s1 - b1) ** 2 < (s2 - b1) ** 2 else (s2, s1)
        return (other, [best]) if viol(best) else (best, [other])

    def pick3(sols):                         # robomit, three real roots: violators pushed to the end, in order
        dist = [(v - b1) ** 2 for v in sols]
        for i in range(3):
            if viol(sols[i]):
                dist[i] = max(dist) + 1
        order = sorted(range(3), key=lambda i: (dist[i], i))
        return sols[order[0]], [sols[order[1]], sols[order[2]]]

    a3 = (delta - 1) * (tx * sxx - tx ** 2)
    # a3 has the units of var(x) squared, so the degeneracy test is relative to sigma_xx**2, never to an
    # absolute floor: with x in small units (sd(x) ~ 1e-3) a floor of 1.0 took the quadratic branch for a
    # cubic case, moving beta* by 3% and switching off the spurious-root filter (v2.5.2 review, chat 3).
    if delta == 1 or abs(a3) <= 1e-12 * sxx ** 2:
        theta = delta * rm_rt * (sxx - tx) - rt_ro * tx - sxx * tx * d ** 2
        a2 = tx * d * sxx * (delta - 2)
        a0 = rm_rt * delta * d * sxx
        if a2 == 0:
            return (b1 - (-a0 / theta) if theta != 0 else np.nan), []
        disc = theta ** 2 - 4 * a2 * a0
        if disc < 0:
            return np.nan, []
        r = np.sqrt(disc)
        return pick2(b1 - (-theta - r) / (2 * a2), b1 - (-theta + r) / (2 * a2))
    A = tx * d * sxx * (delta - 2) / a3
    B = (delta * rm_rt * (sxx - tx) - rt_ro * tx - sxx * tx * d ** 2) / a3
    C = rm_rt * delta * d * sxx / a3
    Q = (A ** 2 - 3 * B) / 9
    R = (2 * A ** 3 - 9 * A * B + 27 * C) / 54
    D = R ** 2 - Q ** 3
    if D < 0:
        th = np.arccos(R / np.sqrt(Q ** 3))
        roots = [-2 * np.sqrt(Q) * np.cos((th + k) / 3) - A / 3 for k in (0.0, 2 * np.pi, -2 * np.pi)]
        return pick3([b1 - v for v in roots])
    t1, t2 = -R + np.sqrt(D), -R - np.sqrt(D)
    nu = np.sign(t1) * abs(t1) ** (1 / 3) + np.sign(t2) * abs(t2) ** (1 / 3) - A / 3
    return b1 - nu, []


def _oster_exact_delta(b0, b1, r0, r1, rmax, beta, syy, sxx, tx):
    """δ that makes the bias-adjusted effect equal `beta` (exact; Oster 2019, as in robomit/psacalc)."""
    bb = b1 - beta
    rt_ro = (r1 - r0) * syy
    rm_rt = (rmax - r1) * syy
    d = b0 - b1
    num = (bb * rt_ro * tx + bb * sxx * tx * d ** 2 + 2 * bb ** 2 * (tx * d * sxx)
           + bb ** 3 * (tx * sxx - tx ** 2))
    den = (rm_rt * d * sxx + bb * rm_rt * (sxx - tx) + bb ** 2 * (tx * d * sxx)
           + bb ** 3 * (tx * sxx - tx ** 2))
    return num / den if den != 0 else np.inf


def oster(res_uncontrolled, res_controlled, focal: str, r_max: float | None = None, delta: float = 1.0,
          exact: bool = True, show: bool = False) -> dict:
    """Coefficient stability (Oster, 2019): the approximate estimator and, with exact=True, the exact one.

    res_uncontrolled: y on the focal variable alone (plus an intercept); res_controlled: adds the observed
    controls. Same sample, unweighted OLS (statsmodels).
    Approximation (keys beta_star, delta_for_zero, identified_set):
      β* = β̃ − δ(β° − β̃)(R_max − R̃)/(R̃ − R°)      bias-adjusted effect for a given δ
      δ(β=0) = β̃(R̃ − R°) / [(β° − β̃)(R_max − R̃)]  how strong selection on unobservables must be,
                                                       relative to observables, to drive β to zero
    Exact (keys beta_star_exact, beta_star_alternatives, delta_for_zero_exact, identified_set_exact,
    exact_inputs): Oster's Proposition 2, computed as Stata's psacalc and R's robomit do; it also uses var(y),
    var(x) and τ_x (the variance of x net of the controls). When δ ≠ 1 the bias-adjusted β solves a cubic; with
    several real roots the one closest to β̃ in the direction of β° → β̃ is reported and the others are listed.
    With a single control the cubic always has one root that does not solve Oster's equation (at the pole of δ*(β),
    where its numerator and denominator both vanish); it is kept apart (beta_star_spurious) and never reported. psacalc and robomit skip this check, so their
    choice is kept apart in beta_star_psacalc; it differs whenever the spurious root is the closest one to β̃,
    which happens with a genuine root available as well as when there is none (then beta_star_exact is nan).
    A warning names the difference, so a number from psacalc or robomit can be told apart from this one.
    beta_star_exact_unstable is True when another admissible root is nearer to β* than β* is to β̃ (a heuristic for
    being near a double root, where β* is unstable); then report δ for β = 0, not β*.
    The exact solution is skipped, with a warning, for weighted or panel fits and when the uncontrolled model has
    other regressors (psacalc's mcontrol); the approximation is still returned.
    Default R_max = min(1.3·R̃, 1) (Oster's calibration, not a fact about the data). δ depends on the R_max chosen
    and is not a probability; |δ(β=0)| > 1 is the usual robustness bar.
    """
    b0, r0 = float(res_uncontrolled.params[focal]), float(res_uncontrolled.rsquared)
    bt, rt = float(res_controlled.params[focal]), float(res_controlled.rsquared)
    w = []
    if int(res_uncontrolled.nobs) != int(res_controlled.nobs):
        w.append("The two regressions use different samples; Oster's bounds assume the same sample.")
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
    if rt - r0 < 0.01:
        w.append(f"The controls raise R² by only {rt - r0:.4f}: they say little about selection, so δ is "
                 "driven by R_max and the noise in R² rather than by the data; do not read δ as robustness.")
    if exact:
        if int(res_uncontrolled.nobs) != int(res_controlled.nobs):
            inp, why = None, "the two regressions use different samples"
        else:
            inp, why = _oster_inputs(res_uncontrolled, res_controlled, focal)
        if inp is None:
            w.append(f"Exact Oster solution not computed: {why}. Only the approximation is reported.")
        else:
            args, S = (b0, bt, r0, rt, rmax), (inp["sigma_yy"], inp["sigma_xx"], inp["tau_x"])
            bp, alt_p = _oster_exact_beta(*args, delta, *S)          # psacalc/robomit's choice and other roots
            dx = _oster_exact_delta(*args, 0.0, *S)
            # Spurious roots. The denominator of δ*(β) factors as [(β° − β̃)σ²_x + ν(σ²_x − τ_x)]·[(R_max − R̃)σ²_y
            # + ν²τ_x], ν = β̃ − β, so δ*(β) has one pole, at β_p = β̃ + (β° − β̃)σ²_x/(σ²_x − τ_x). With a single
            # control the numerator vanishes there too (Frisch-Waugh: (R̃ − R°)σ²_y = (β° − β̃)²σ²_xτ_x/(σ²_x − τ_x)),
            # and the cubic psacalc and robomit solve (numerator − δ·denominator) gains a root at β_p for every δ and
            # R_max that does not solve Oster's equation. They do not check this. (v2.5.2 study, 2026-11-08.)
            roots = [float(r) for r in [bp, *alt_p] if np.isfinite(r)]
            sxx, tx = S[1], S[2]
            pole = bt + (b0 - bt) * sxx / (sxx - tx) if sxx - tx > 1e-12 * sxx else None
            # relative to the coefficients' own scale, for the same reason (an absolute floor made every
            # root look like the pole once |beta| fell below ~1e-8)
            b_scale = max(abs(b0), abs(bt), abs(pole) if pole is not None else 0.0)
            tol = 1e-7 * b_scale if b_scale > 0 else 1e-12
            spurious = [r for r in roots if pole is not None and abs(r - pole) <= tol]
            genuine = [r for r in roots if r not in spurious]
            side = (lambda r: np.sign(r - bt) == np.sign(bt - b0)) if delta > 0 else (lambda r: True)
            adm = sorted((r for r in genuine if side(r)), key=lambda r: (abs(r - bt), r))
            if adm:                                   # Oster's rule: the admissible root closest to β̃
                bx, violates = adm[0], False
            elif genuine:                             # only roots on the side Assumption 3 forbids
                bx, violates = min(genuine, key=lambda r: (abs(r - bt), r)), True
            else:
                bx, violates = np.nan, False
            others = [r for r in genuine if r != bx]
            out.update(beta_star_exact=float(bx), beta_star_alternatives=others, beta_star_spurious=spurious,
                       beta_star_psacalc=float(bp), delta_for_zero_exact=float(dx), exact_inputs=inp,
                       identified_set_exact=tuple(sorted((bt, float(bx)))) if np.isfinite(bx) else None,
                       beta_star_exact_unstable=False)
            if not np.isfinite(bx):
                if spurious:
                    w.append(f"The exact equation has no real solution for this δ and R_max. psacalc and robomit "
                             f"report β* = {bp:.4g}, but that is a spurious root of the cubic they solve (it does not "
                             "satisfy Oster's equation; with a single control it appears for every δ). Do not report "
                             "it; report δ for β = 0 and the approximation, and say why.")
                else:
                    w.append("The exact equation has no real solution for this δ and R_max; only the approximation "
                             "is reported.")
            else:
                if violates:
                    w.append(f"The exact β* ({bx:.4g}) moves away from β̃ in the opposite direction to the move from "
                             f"β° ({b0:.4g}) to β̃ ({bt:.4g}), which Oster's Assumption 3 rules out: no admissible "
                             "real root for this δ and R_max (psacalc and robomit report it anyway). Do not report "
                             "it as the bias-adjusted effect; report δ for β = 0 and the approximation, and say so.")
                # Another admissible root nearer to β* than β* is to β̃: the closest-root rule picks between two
                # admissible answers near a double root, where β* moves a lot with small changes in R², var(y),
                # var(x) or τ_x (v2.5.2 study: δ = 2, n = 400 000, β* off by 2-3%, or no real root at all). A
                # heuristic. Roots on the forbidden side leave no doubt about the choice: they stay in
                # beta_star_alternatives without a warning (δ = 1 often has one; the default must stay quiet).
                near = [a for a in adm[1:] if abs(a - bx) < abs(bx - bt)]
                unstable = bool(near) and not violates
                out["beta_star_exact_unstable"] = unstable
                if len(adm) > 1 and not violates:
                    msg = (f"The exact equation has {len(adm)} admissible real roots: β* = {bx:.4g} is the one "
                           f"closest to β̃ in the direction of the move from β° (Oster's rule); the other(s): "
                           + ", ".join(f"{a:.4g}" for a in adm[1:]) + ".")
                    if unstable:
                        msg += (f" Another admissible root ({near[0]:.4g}) is closer to β* than β* is to β̃: the "
                                "equation is near a double root, where β* is unstable (small changes in the R² or "
                                "the variances move it a lot, even in very large samples). Do not report β* as the "
                                "bias-adjusted effect; report δ for β = 0, which is stable, and say why.")
                    else:
                        msg += " Report the choice."
                    w.append(msg)
                if spurious and any(abs(bp - r) <= tol for r in spurious):
                    w.append(f"psacalc and robomit would report β* = {bp:.4g} here: that is the spurious root of "
                             f"the cubic they solve, not a solution of Oster's equation. The solution is "
                             f"{bx:.4g}; do not reconcile the two.")
                scale = max(abs(bt), abs(b0 - bt), 1e-12)
                if not (violates or unstable) and abs(bx - beta_star) > 0.10 * scale:   # one warning per situation
                    w.append(f"The exact β* ({bx:.4g}) and the approximation ({beta_star:.4g}) differ by more than "
                             "10% of the effect: report the exact one (the approximation assumes the controls "
                             "and the unobservables relate to the treatment in a simple way).")
    out["warnings"] = w
    for m in w:
        warnings.warn(m)
    if show:
        print(f"\nOster (R_max = {rmax:.3f}, δ = {delta}): β̃ = {bt:.4f} → β* = {beta_star:.4f} (approximation); "
              f"identified set [{out['identified_set'][0]:.4f}, {out['identified_set'][1]:.4f}]")
        if "beta_star_exact" in out:
            print(f"Exact: β* = {out['beta_star_exact']:.4f}; δ needed for β = 0: {out['delta_for_zero_exact']:.2f}")
        for m in w:                                   # the exact β* can be one the warnings say not to report
            print("  WARNING: " + m)
        dz = out.get("delta_for_zero_exact", delta_zero)
        print(f"δ needed for β = 0: {delta_zero:.2f} (approximation)"
              + (f"; {dz:.2f} (exact)" if "delta_for_zero_exact" in out else "")
              + (" (> 1: robust by Oster's criterion, for this R_max)" if abs(dz) > 1 else " (≤ 1: fragile)"))
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


# =============================================================================
# Classical regression (v2.4.2): restrictions, AR(1) errors, non-nested models
# =============================================================================
def _restriction_matrix(res, restrictions):
    """R, r and readable rows from a statsmodels formula fit and a constraint string or (R, r)."""
    names = list(res.params.index)
    if isinstance(restrictions, str):
        try:                                      # statsmodels >= 0.15
            from statsmodels.formula._manager import FormulaManager
            lc = FormulaManager().get_linear_constraints(restrictions, names)
            R, r = np.asarray(lc.constraint_matrix, float), np.asarray(lc.constraint_values, float).ravel()
        except ImportError:                       # statsmodels 0.14 (patsy)
            from patsy import DesignInfo
            lc = DesignInfo(names).linear_constraint(restrictions)
            R, r = np.asarray(lc.coefs, float), np.asarray(lc.constants, float).ravel()
        rows = [s.strip() for s in restrictions.split(",")]
    else:
        R, r = restrictions
        R = np.atleast_2d(np.asarray(R, float))
        r = np.zeros(R.shape[0]) if r is None else np.asarray(r, float).ravel()
        rows = [" + ".join(f"{v:g}*{n}" for v, n in zip(row, names) if v != 0) + f" = {c:g}" for row, c in zip(R, r)]
    if R.shape[1] != len(names) or R.shape[0] != len(r):
        raise ValueError(f"R must have {len(names)} columns and one row per restriction")
    if np.linalg.matrix_rank(R) < R.shape[0]:
        raise ValueError("the restrictions are linearly dependent: drop the redundant rows")
    return R, r, rows


def linear_restriction_test(formula: str, data: pd.DataFrame, restrictions, cov_type: str = "nonrobust",
                            alpha: float = ALPHA, show: bool = True) -> dict:
    """Test H0: Rβ = r in an OLS model, with the restricted estimates (Wooldridge ch. 4.5; Greene ch. 5).

    restrictions: a string in the formula's names ("educ = exper", "x1 = 0, x2 = 0", "x1 + x2 = 1") or (R, r).
    Classical errors (default): F = [(SSR_r − SSR_ur)/q] / [SSR_ur/(n − k)], computed from the sums of
    squared residuals, with the restricted least-squares coefficients β_r = β̂ − (X'X)⁻¹R'[R(X'X)⁻¹R']⁻¹(Rβ̂ − r).
    The R² form [(R²_ur − R²_r)/q]/[(1 − R²_ur)/(n − k)] gives the same F only when both models explain the
    same y; a restricted model fitted to a transformed y (y − x1 to impose β1 = 1) has another total sum of
    squares, and its R² must not be used. Robust cov_type ("HC3", ...): Wald F with that covariance (the SSR
    form is not valid under heteroskedasticity). With one restriction, also the estimate of θ = Rβ − r with
    its SE, t and CI: the t of the reparametrized model, whose square is F."""
    ur = smf.ols(formula, data).fit()
    R, r, rows = _restriction_matrix(ur, restrictions)
    q, n, k = R.shape[0], int(ur.nobs), int(ur.df_model) + 1
    b = ur.params.to_numpy()
    X = ur.model.exog
    XtXi = np.linalg.inv(X.T @ X)
    d = R @ b - r
    b_r = b - XtXi @ R.T @ np.linalg.solve(R @ XtXi @ R.T, d)
    e_r = ur.model.endog - X @ b_r
    ssr_ur, ssr_r = float(ur.ssr), float(e_r @ e_r)
    warn = []
    if cov_type == "nonrobust":
        F = ((ssr_r - ssr_ur) / q) / (ssr_ur / ur.df_resid)
        V = ur.cov_params().to_numpy()
        form = "SSR"
    else:
        fit = ur.model.fit(cov_type=cov_type)
        V = fit.cov_params().to_numpy()
        F = float(d @ np.linalg.solve(R @ V @ R.T, d)) / q
        form = f"Wald with {cov_type} covariance"
        warn.append("robust F: the SSR and R² forms assume homoskedasticity and are not reported as the test")
    p = float(stats.f.sf(F, q, ur.df_resid))
    yv = ur.model.endog
    sst = float(np.sum((yv - yv.mean()) ** 2)) if ur.k_constant else float(yv @ yv)   # same definition as ur.rsquared
    r2_r = 1 - ssr_r / sst
    out = {"F": float(F), "df": (q, float(ur.df_resid)), "p": p, "q": q, "n": n, "form": form, "restrictions": rows,
           "ssr_unrestricted": ssr_ur, "ssr_restricted": ssr_r, "r2_unrestricted": float(ur.rsquared),
           "r2_restricted": r2_r, "restricted_params": pd.Series(b_r, index=ur.params.index),
           "verdict": ("reject H0" if p < alpha else "do not reject H0 (not evidence that it holds)") + f" at {alpha:.0%}",
           "unrestricted": ur, "warnings": warn}
    if q == 1:
        theta = float(d[0])
        se = float(np.sqrt(R[0] @ V @ R[0]))
        t = theta / se
        tq = stats.t.ppf(1 - alpha / 2, ur.df_resid)
        out["theta"] = {"estimand": f"{rows[0]} (θ = Rβ − r)", "estimate": theta, "se": se, "t": t,
                        "p": float(2 * stats.t.sf(abs(t), ur.df_resid)),
                        "ci_low": theta - tq * se, "ci_high": theta + tq * se}
    if show:
        print(f"\nH0: {'; '.join(rows)}   (q = {q}, n = {n}, {form})")
        print(f"F({q}, {ur.df_resid:.0f}) = {F:.4f}, p = {p:.4g} → {out['verdict']}")
        if cov_type == "nonrobust":
            print(f"SSR restricted = {ssr_r:.6g}, unrestricted = {ssr_ur:.6g}")
        if q == 1:
            th = out["theta"]
            print(f"θ = {th['estimate']:.4f} (se {th['se']:.4f}), t = {th['t']:.3f}, "
                  f"CI [{th['ci_low']:.4f}, {th['ci_high']:.4f}]; t² = F")
        for w in warn:
            print("Warning:", w)
    return out


def prais_winsten(formula: str, data: pd.DataFrame, method: str = "prais", max_iter: int = 100, tol: float = 1e-8,
                  cov_type: str = "nonrobust", time: str | None = None, alpha: float = ALPHA, show: bool = True) -> dict:
    """Feasible GLS for AR(1) errors, u_t = ρu_{t−1} + e_t (Wooldridge ch. 12.3), iterated to convergence.

    method="prais" (Prais-Winsten) keeps the first observation, scaled by √(1 − ρ²); method="cochrane"
    (Cochrane-Orcutt) drops it. ρ̂ = Σû_tû_{t−1}/Σû²_{t−1} from the residuals of the current estimates,
    then OLS on the quasi-differenced data, until ρ̂ changes less than tol. Data must be sorted in time
    and without gaps: pass time="<column>" to sort by it; without it the index must be increasing. Rows dropped
    for missing values are allowed only at the start or the end of the sample (lags), never inside it, where the
    quasi-difference would join non-adjacent periods. FGLS needs strictly exogenous regressors: with a lagged y or feedback from y to x it
    is inconsistent, and OLS with Newey-West errors is the safer choice (diagnostics.compare_standard_errors).
    The DW statistic is reported for the OLS residuals and for the transformed (whitened) residuals."""
    if method not in ("prais", "cochrane"):
        raise ValueError("method must be 'prais' or 'cochrane'")
    if time is not None:
        if data[time].isna().any() or not data[time].is_unique:
            raise ValueError(f"'{time}' must have unique, non-missing values")
        data = data.sort_values(time)
    elif not data.index.is_monotonic_increasing or not data.index.is_unique:
        raise ValueError("data are not in time order: sort them, or pass time='<column>'")
    ols = smf.ols(formula, data).fit()
    pos = data.index.get_indexer(ols.model.data.row_labels)
    if np.any(np.diff(pos) != 1):
        raise ValueError("rows with missing values inside the sample break the time sequence; fill or drop them "
                         "explicitly (only leading or trailing rows may be missing)")
    y, X = ols.model.endog, ols.model.exog
    names = list(ols.params.index)
    n = len(y)
    if n < 10:
        raise ValueError("too few observations for an AR(1) error model")
    b = ols.params.to_numpy()
    rho, converged, it = 0.0, False, 0
    for it in range(1, max_iter + 1):
        u = y - X @ b
        rho_new = float(u[1:] @ u[:-1] / (u[:-1] @ u[:-1]))
        if abs(rho_new) >= 1:
            raise ValueError(f"ρ̂ = {rho_new:.3f} is not in (−1, 1): the errors look nonstationary; difference the data")
        ys, Xs = y[1:] - rho_new * y[:-1], X[1:] - rho_new * X[:-1]
        if method == "prais":
            c = np.sqrt(1 - rho_new ** 2)
            ys, Xs = np.r_[c * y[0], ys], np.vstack([c * X[0], Xs])
        b = np.linalg.lstsq(Xs, ys, rcond=None)[0]
        done = abs(rho_new - rho) < tol
        rho = rho_new
        if done:
            converged = True
            break
    fit = sm.OLS(ys, Xs).fit(cov_type=cov_type)
    from statsmodels.stats.stattools import durbin_watson
    tq = stats.t.ppf(1 - alpha / 2, fit.df_resid)
    table = pd.DataFrame({"OLS coef": ols.params.to_numpy(), "OLS se": ols.bse.to_numpy(),
                          "FGLS coef": fit.params, "FGLS se": fit.bse}, index=names)
    table["FGLS ci_low"] = table["FGLS coef"] - tq * table["FGLS se"]
    table["FGLS ci_high"] = table["FGLS coef"] + tq * table["FGLS se"]
    warn = []
    if not converged:
        warn.append(f"ρ̂ did not converge in {max_iter} iterations")
    if abs(rho) > 0.9:
        warn.append(f"ρ̂ = {rho:.3f} is close to 1: consider first differences or a unit-root test")
    out = {"method": "Prais-Winsten" if method == "prais" else "Cochrane-Orcutt", "rho": rho, "iterations": it,
           "converged": converged, "n": int(fit.nobs), "table": table, "dw_ols": float(durbin_watson(ols.resid)),
           "dw_transformed": float(durbin_watson(fit.resid)), "ols": ols, "fgls": fit, "warnings": warn}
    if show:
        print(f"\n{out['method']} FGLS, AR(1) errors: ρ̂ = {rho:.4f} ({it} iterations"
              f"{'' if converged else ', NOT converged'}), n = {out['n']}, SEs: {cov_type}")
        print(table.to_string(float_format=_fmt))
        print(f"Durbin-Watson: OLS {out['dw_ols']:.3f} → transformed {out['dw_transformed']:.3f}")
        for w in warn:
            print("Warning:", w)
    return out


def nonnested_test(formula1: str, formula2: str, data: pd.DataFrame, cov_type: str = "HC3", alpha: float = ALPHA,
                   show: bool = True) -> dict:
    """Choose between two non-nested models of the same y (Wooldridge ch. 9.2; Davidson-MacKinnon, 1981).

    J test, in both directions: add the fitted values of the rival model to each model and t-test them.
    Encompassing (Mizon-Richard) test: the comprehensive model with the regressors of both; F-test the
    regressors that only the rival has. Both, rejected both, or neither can happen: rejecting both says
    neither model is adequate; rejecting neither says the data cannot tell them apart. Both models must
    explain the same y; for y vs log y use a PE test or compare out-of-sample predictions in y units."""
    m1, m2 = smf.ols(formula1, data).fit(), smf.ols(formula2, data).fit()
    if m1.model.endog_names != m2.model.endog_names:
        raise ValueError("the J and encompassing tests need the same dependent variable in both models")
    idx = m1.model.data.row_labels.intersection(m2.model.data.row_labels)
    if len(idx) < len(m1.model.data.row_labels) or len(idx) < len(m2.model.data.row_labels):
        warnings.warn("the models use different rows (missing values); both are refitted on the common rows")
    d = data.loc[idx].copy()
    m1, m2 = smf.ols(formula1, d).fit(), smf.ols(formula2, d).fit()
    rows = []
    for name, base, rival in (("model 1", formula1, m2), ("model 2", formula2, m1)):
        d["_rival_fit"] = rival.fittedvalues
        aug = smf.ols(f"{base} + _rival_fit", d).fit(cov_type=cov_type)
        rows.append({"H0": f"{name} is adequate (J)", "statistic": float(aug.tvalues["_rival_fit"]),
                     "df": 1, "p": float(aug.pvalues["_rival_fit"])})
    X1 = pd.DataFrame(m1.model.exog, columns=m1.params.index, index=idx)
    X2 = pd.DataFrame(m2.model.exog, columns=m2.params.index, index=idx)

    def extra_columns(own, rival):
        """Rival columns not in the span of own's columns (added one at a time while they raise the rank),
        so that a variable written differently in the two formulas (C(d) vs d) is not counted twice."""
        keep, cur = [], own.to_numpy()
        r0 = np.linalg.matrix_rank(cur)
        for c in rival.columns:
            cand = np.column_stack([cur, rival[c].to_numpy()])
            r1 = np.linalg.matrix_rank(cand)
            if r1 > r0:
                keep.append(c); cur, r0 = cand, r1
        return keep

    for name, own, rival in (("model 1", X1, X2), ("model 2", X2, X1)):
        extra = extra_columns(own, rival)
        if not extra:
            rows.append({"H0": f"{name} encompasses the rival (the rival adds nothing)", "statistic": np.nan, "df": 0,
                         "p": np.nan})
            continue
        full_X = pd.concat([own, rival[extra].add_prefix("rival:")], axis=1)
        full = sm.OLS(m1.model.endog, full_X).fit(cov_type=cov_type)
        Rm = np.zeros((len(extra), full_X.shape[1]))
        Rm[:, own.shape[1]:] = np.eye(len(extra))
        w = full.wald_test(Rm, scalar=True, use_f=True)
        rows.append({"H0": f"{name} encompasses the rival (F on {', '.join(extra)})",
                     "statistic": float(w.statistic), "df": len(extra), "p": float(w.pvalue)})
    tab = pd.DataFrame(rows).set_index("H0")
    rej1, rej2 = tab["p"].iloc[0] < alpha, tab["p"].iloc[1] < alpha
    verdict = {(True, True): "J rejects both models: neither is adequate; look for a better specification",
               (False, False): "J rejects neither: the data do not discriminate; prefer the simpler or theory-based model",
               (True, False): "J rejects model 1, not model 2: the data favour model 2",
               (False, True): "J rejects model 2, not model 1: the data favour model 1"}[(rej1, rej2)]
    out = {"table": tab, "verdict": verdict, "n": len(idx), "cov_type": cov_type,
           "adj_r2": {"model 1": float(m1.rsquared_adj), "model 2": float(m2.rsquared_adj)},
           "bic": {"model 1": float(m1.bic), "model 2": float(m2.bic)}, "models": (m1, m2)}
    if show:
        print(f"\nNon-nested comparison (n = {len(idx)}, SEs {cov_type})")
        print(f"  model 1: {formula1}\n  model 2: {formula2}")
        print(tab.to_string(float_format=_fmt))
        print("→", verdict)
        print(f"Adjusted R²: {out['adj_r2']['model 1']:.4f} vs {out['adj_r2']['model 2']:.4f}; "
              f"BIC: {out['bic']['model 1']:.2f} vs {out['bic']['model 2']:.2f} (lower is better)")
    return out
