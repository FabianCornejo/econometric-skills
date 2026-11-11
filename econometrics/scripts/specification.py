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
    ovb_bounds(res, treatment, benchmarks)       Cinelli-Hazlett bounds: a confounder k times as strong as an
                                                 observed covariate (or group) -> adjusted estimate, SE and CI,
                                                 and the k that would remove 100·q% of the estimate

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


def _has_intercept(res) -> bool:
    """Whether a fit has an intercept, by statsmodels' own flag (what decides centered vs uncentered R²),
    with a scan for a constant column as the fallback."""
    m = getattr(res, "model", None)
    if m is None:
        return True                                  # not a statsmodels fit: the caller's other gates handle it
    if getattr(m, "k_constant", None):
        return True
    X = np.asarray(getattr(m, "exog", np.empty((0, 0))), float)
    return X.ndim == 2 and X.shape[0] > 0 and any(
        np.allclose(X[:, k], X[0, k]) for k in range(X.shape[1]))


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
    controls. Same sample, unweighted OLS (statsmodels). Both fits must have an intercept: without one
    statsmodels reports an uncentered R² and every bound here would be wrong, so the function refuses.
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
    # Every number here is a function of R², and statsmodels reports an UNCENTERED R² for a fit without an
    # intercept. Oster's decomposition needs the centered one (its identity R°σ²_y = β°²σ²_x holds only with an
    # intercept), so both the approximation and the exact solution come out wrong, and quietly: with y = 1 + x + 2z
    # the approximation gave 2530 and the exact solution 0.456 against the right 0.702, with no warning at all,
    # and a slightly larger intercept flipped the sign of the effect (v2.5.2 review, chat 3). Refuse instead.
    for res, which in ((res_uncontrolled, "uncontrolled"), (res_controlled, "controlled")):
        if not _has_intercept(res):
            raise ValueError(
                f"The {which} regression has no intercept, so its R² is uncentered and Oster's bounds do not "
                "apply: every quantity here (β*, δ for β = 0, the identified set) would be wrong without "
                "warning. Refit both models with an intercept.")
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
    not_to_report = False                         # the exact β* violates Assumption 3 or is unstable
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
                not_to_report = violates or unstable
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
            ise = out["identified_set_exact"]     # v2.5.3: computed since v2.5.2, never printed before
            if ise is None:
                iset = "no exact identified set (no admissible real root; see the warning)"
            elif not_to_report:                   # a root the warnings say not to report (v2.5.3 review)
                iset = (f"identified set [{ise[0]:.4f}, {ise[1]:.4f}] from that root, not to be reported "
                        "(see the warning)")
            else:
                iset = f"identified set [{ise[0]:.4f}, {ise[1]:.4f}] (exact)"
            print(f"Exact: β* = {out['beta_star_exact']:.4f}; {iset}; "
                  f"δ needed for β = 0: {out['delta_for_zero_exact']:.2f}")
        for m in w:                                   # the exact β* can be one the warnings say not to report
            print("  WARNING: " + m)
        dz = out.get("delta_for_zero_exact", delta_zero)
        print(f"δ needed for β = 0: {delta_zero:.2f} (approximation)"
              + (f"; {dz:.2f} (exact)" if "delta_for_zero_exact" in out else "")
              + (" (> 1: robust by Oster's criterion, for this R_max)" if abs(dz) > 1 else " (≤ 1: fragile)"))
    return out


def _check_q(q):
    """Validate the q of the robustness values (v2.5.3): a real number > 0, finite, not a bool.

    q is the share of the estimate the confounder would remove (q = 1: all of it; q = 2: flip its sign). q <= 0
    has no meaning (q = -1 printed "reduce the estimate by -100% (to 2 times its value)"), and True passed as 1.
    The value is kept as given (an int stays an int, np.float32 keeps its exact value): only the label rounds."""
    import numbers
    if isinstance(q, (bool, np.bool_)) or not isinstance(q, numbers.Real):
        raise TypeError(f"q must be an int or a float > 0 (the share of the estimate to remove; 1 = all of it), "
                        f"got {q!r}")
    try:
        finite = bool(np.isfinite(float(q)))
    except OverflowError:                         # an int too large for a float (2**1100)
        finite = False
    if not finite or q <= 0:
        raise ValueError(f"q must be finite and > 0 (the share of the estimate to remove; 1 = all of it), got {q!r}")
    return int(q) if isinstance(q, numbers.Integral) else float(q)


def _q_label(q) -> str:
    """q as printed: integers as given ("2"), floats to 6 significant digits ("0.1" for np.float32(0.1), "1.0",
    "1.23457e+08"); a float that prints without a point or exponent gets ".0", so 1.0 stays "1.0"."""
    import numbers
    if isinstance(q, numbers.Integral):
        return str(int(q))
    s = f"{float(q):.6g}"
    return s if any(c in s for c in ".en") else s + ".0"


def _q_is_one(q) -> bool:
    """q that prints as 1 reads as q = 1 in the text (the numbers use q as given): the label and the sentence
    always agree (q = 0.9999999 printed RV(q=1.0) with the q != 1 sentence, review of v2.5.3)."""
    return float(_q_label(q)) == 1.0


def _pct(x, digits=1) -> str:
    """A share as a percentage with `digits` decimals; a positive share that would print as zero says
    "less than 0.1%" (it printed "0.0% would make it insignificant" just above the df/df − 1 band)."""
    if 0 < x < 0.5 * 10 ** -(digits + 2):
        return f"less than {10 ** -digits:.{digits}f}%"
    return f"{x:.{digits}%}"


def robustness_value(t: float, dof: float, q: float = 1.0, alpha: float = ALPHA) -> dict:
    """Cinelli and Hazlett (2020) sensitivity statistics from a classical OLS t-statistic.

    partial_r2: share of the outcome's residual variance explained by the treatment (R²_{Y~D|X}).
    rv_q: minimum partial R² that a confounder would need with BOTH treatment and outcome to
          reduce the estimate by 100·q% (q = 1: to zero).
    rv_qa: the same, to make the reduced estimate no longer significant at level alpha.
    q must be a finite real number > 0 (v2.5.3: ValueError otherwise; TypeError for a bool or a non-number).
    """
    q = _check_q(q)
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
    variable or fixed effect are grouped into one joint partial R²): an informal comparison only.
    For formal bounds (a confounder k times as strong as a covariate, with the adjusted estimate
    and CI) use `ovb_bounds(res, treatment, benchmarks)` (v2.6).
    """
    base = res if getattr(res, "cov_type", "nonrobust") == "nonrobust" else res.model.fit()
    out = robustness_value(base.tvalues[treatment], base.df_resid, q=q, alpha=alpha)
    groups = _covariate_groups(base, treatment)
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
            k = len(terms)                        # number of joint restrictions; never reuse `q` (v2.5.2.1)
            bench[f"{gname} ({k} dummies, jointly)"] = k * F / (k * F + base.df_resid)
    out["benchmarks"] = pd.Series(bench, name="partial R² with outcome").sort_values(ascending=False)
    if show:
        qq = out["q"]                             # the label shows the q the statistics were computed with
        lab = _q_label(qq)
        print(f"\nSensitivity of {treatment} (Cinelli-Hazlett): partial R² = {out['partial_r2']:.3%}, "
              f"RV(q={lab}) = {out['rv_q']:.2%}, RV(q={lab}, α={alpha}) = {out['rv_qa']:.2%}")
        level = f"{round((1 - alpha) * 100, 6):g}%"           # confidence level of the interval
        sig = f"{round(alpha * 100, 6):g}%"                    # significance level of the test
        # "already" is judged on the classical CI with df (the one these statistics come from), not on the paper's
        # df − 1 convention (a null confounder costs one degree of freedom): between the two, rv_qa is 0 but that
        # CI excludes the target. The target is (1 − q)·estimate, inside the CI when q·|t| <= t_crit(df); this
        # implies rv_qa = 0 for the same q (up to q only rounding to 1, where the header shows 0.00% anyway).
        one = _q_is_one(qq)
        q_eff = 1.0 if one else float(qq)
        in_reported = q_eff * abs(float(base.tvalues[treatment])) <= stats.t.ppf(1 - alpha / 2, base.df_resid)
        # the band: only a regressor unrelated to both (zero partial R² with D and Y) is sure to push it over, by
        # costing one degree of freedom; one that predicts Y lowers the SE (review of v2.5.3)
        null_reg = "adding a regressor unrelated to both (it only costs one degree of freedom)"
        if one:
            if in_reported:                       # v2.5.3: was "0.0% would make it insignificant"
                tail = f"it is already not significant at the {sig} level, with no confounder"
            elif out["rv_qa"] == 0:
                tail = f"it is only just significant at the {sig} level: {null_reg} would make it insignificant"
            else:                                 # the v2.5.2.1 sentence, byte for byte unless RV_q,α < 0.05%
                tail = f"{_pct(out['rv_qa'])} would make it insignificant"
            print(f"A confounder explaining {_pct(out['rv_q'])} of the residual variance of both {treatment} and the "
                  f"outcome would remove the estimate; {tail}.")
        else:                                     # RV_q reduces the estimate by 100q%; RV_q,α puts (1−q)·estimate in the CI
            pct, rest = f"{float(qq) * 100:.6g}%", f"{1 - float(qq):.6g}"
            if in_reported:
                tail = f"that reduced value is already inside the {level} confidence interval, with no confounder"
            elif out["rv_qa"] == 0:
                tail = f"that reduced value is just outside the {level} confidence interval: {null_reg} would put it inside"
            else:
                tail = f"{_pct(out['rv_qa'])} would put that reduced value inside the {level} confidence interval"
            print(f"A confounder explaining {_pct(out['rv_q'])} of the residual variance of both {treatment} and the "
                  f"outcome would reduce the estimate by {pct} (to {rest} times its value); {tail}.")
        if base is not res:                       # robust fit: the verdicts above are for the classical SE
            t_cl, t_rb = float(base.tvalues[treatment]), float(res.tvalues[treatment])
            se_rb = float(res.bse[treatment])
            rb = (f"your {res.cov_type} fit gives t = {t_rb:.2f}, so its significance can differ from the verdict above"
                  if np.isfinite(t_rb) and np.isfinite(se_rb) and se_rb > 0 else    # v2.6.1: printed "t = 0.00"
                  f"your {res.cov_type} SE is not finite or zero (often observations with leverage 1), so its t is "
                  f"not usable")
            print(f"These statistics and verdicts use the classical SE, as the formulas require (t = {t_cl:.2f}); "
                  f"{rb}.")
        if len(out["benchmarks"]):
            print("Observed covariates (partial R² with outcome), for comparison:")
            print(out["benchmarks"].to_string(float_format=lambda v: f"{v:.2%}"))
            print(f'For formal bounds (a confounder k times as strong as one of these, with the adjusted estimate '
                  f'and CI), run specification.ovb_bounds(res, "{treatment}", benchmarks=[...]).')
    return out


def _covariate_groups(base, treatment: str) -> dict[str, list[str]]:
    """Regressors other than the intercept and the treatment, the dummies of one categorical grouped under its name
    ("C(g)" for C(g)[T.b], "C(g, Treatment('c'))" for its dummies, "g" for g[T.b]), and the dummies of one
    interaction term grouped under the term ("C(g):x1" for C(g)[T.b]:x1, C(g)[T.c]:x1, …), so a joint R² never
    depends on the reference level. Until v2.6 "C(g)[T.b]:x1" joined the C(g) group when C(g) came first in the
    formula but formed its own "x1:C(g)" group the other way round (review of v2.6). Shared by sensitivity and
    ovb_bounds."""
    def factor(part):
        m = re.match(r"^(C\(.*\))\[[^\[\]]*\]$", part) or re.match(r"^(.+?)\[T\.[^\[\]]*\]$", part)
        return m.group(1) if m else part
    groups: dict[str, list[str]] = {}
    for c in base.params.index:
        if c in _INTERCEPT or c == treatment:
            continue
        groups.setdefault(":".join(factor(p) for p in c.split(":")), []).append(c)
    return groups


def _partial_r2_terms(params, cov, df, terms) -> float:
    """Partial R² of a set of regressors from a classical fit: t²/(t² + df) for one, pF/(pF + df) for p of them
    (F the Wald statistic with the classical covariance; exact for OLS, as in sensemakr's group_partial_r2)."""
    b = np.asarray(params.loc[terms], dtype=float)
    V = np.asarray(cov.loc[terms, terms], dtype=float)
    pF = float(b @ np.linalg.solve(V, b))         # = p·F = t² when p = 1
    return pF / (pF + df)


def _check_k(k, name: str) -> np.ndarray:
    """kd or ky as a 1-D float array of finite numbers > 0 (a number or a sequence; bools and strings rejected)."""
    import numbers
    vals = [k] if isinstance(k, numbers.Number) or np.ndim(k) == 0 else list(k)
    if not vals:
        raise ValueError(f"{name} is empty")
    for v in vals:
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real):
            raise TypeError(f"{name} must be a number > 0 or a sequence of them (k times as strong), got {v!r}")
    arr = np.asarray(vals, dtype=float)
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
        raise ValueError(f"{name} must be finite and > 0 (k times as strong as the benchmark), got {k!r}")
    return arr


_R2_ZERO = 64 * np.finfo(float).eps            # a partial R² this small is rounding noise of an exact zero


def _kd_max(r2dxj: float) -> float:
    """Largest admissible kd (exclusive): r2dz.x = kd·r2dxj/(1 − r2dxj) < 1 and sensemakr's second condition
    r2zxj.xd = kd·r2dxj²/((1 − kd·r2dxj)(1 − r2dxj)) < 1 both reduce to kd < (1 − r2dxj)/r2dxj."""
    return np.inf if r2dxj <= 0 else (1 - r2dxj) / r2dxj


def _ovb_partial_r2_bound(r2dxj: float, r2yxj: float, kd: np.ndarray, ky: np.ndarray):
    """Cinelli-Hazlett (2020, §4.4 and sensemakr's ovb_partial_r2_bound): bounds on the partial R² of a confounder
    Z orthogonal to X with D (r2dz_x) and with Y given D (r2yz_dx, before capping at 1), when Z explains kd times
    the residual variance of D and ky times that of Y that the benchmark explains."""
    kd, ky = np.asarray(kd, dtype=float), np.asarray(ky, dtype=float)
    r2dz = kd * r2dxj / (1 - r2dxj)
    r2zxj = kd * r2dxj ** 2 / ((1 - kd * r2dxj) * (1 - r2dxj))
    r2yz = ((np.sqrt(ky) + np.sqrt(r2zxj)) / np.sqrt(1 - r2zxj)) ** 2 * (r2yxj / (1 - r2yxj))
    return r2dz, r2yz


def _ovb_adjust(est: float, se: float, df: float, r2dz, r2yz):
    """Estimate, SE and t adjusted for a confounder with partial R² r2dz (with D) and r2yz (with Y given D), the bias
    pulling the estimate toward zero (sensemakr's reduce = TRUE; an estimate of exactly 0 counts as positive).
    |bias| = √(r2yz·r2dz/(1 − r2dz))·se·√df and SE = √((1 − r2yz)/(1 − r2dz))·se·√(df/(df − 1)) (eq. 12–13)."""
    r2dz, r2yz = np.asarray(r2dz, dtype=float), np.asarray(r2yz, dtype=float)
    bias = np.sqrt(r2yz * r2dz / (1 - r2dz)) * se * np.sqrt(df)
    sign = -1.0 if est < 0 else 1.0
    adj = sign * (abs(est) - bias)
    adj_se = np.sqrt((1 - r2yz) / (1 - r2dz)) * se * np.sqrt(df / (df - 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        adj_t = adj / adj_se
    return adj, adj_se, adj_t


def _k_needed(r2dxj: float, r2yxj: float, est: float, se: float, df: float, q) -> float:
    """Smallest k (kd = ky = k) at which the bound removes 100·q% of the estimate (bias = q·|est|).

    The bias is strictly increasing in k when both partial R² are > 0 (r2dz_x, r2zxj and r2yz_dx increase in k;
    capping r2yz_dx at 1 only holds it constant) and goes to infinity as k approaches the largest admissible kd,
    so the root is unique; brentq finds it (relative tolerance only: an absolute one returned 0 for tiny roots).
    inf when the benchmark explains nothing of D or of Y (partial R² at most 64 eps, the rounding noise of an exact
    zero: no multiple of it biases the estimate) or when the target is only reached at the largest admissible kd in
    floating point; 0 when q·|est| = 0."""
    from scipy.optimize import brentq
    target = q * abs(est)
    if target == 0:
        return 0.0
    if r2dxj <= _R2_ZERO or r2yxj <= _R2_ZERO:   # 3e-33 for a regressor orthogonal by construction
        return np.inf

    def gap(k):
        r2dz, r2yz = _ovb_partial_r2_bound(r2dxj, r2yxj, k, k)
        return float(np.sqrt(min(r2yz, 1.0) * r2dz / (1 - r2dz)) * se * np.sqrt(df)) - target

    hi = _kd_max(r2dxj)
    top = hi * (1 - 1e-12)
    while not gap(top) >= 0:                      # the bias explodes near kd_max; move closer if needed
        nxt = hi - (hi - top) / 1e3
        if nxt >= hi or nxt == top:
            return np.inf                         # not reachable below kd_max in floating point
        top = nxt
    return float(brentq(gap, 0.0, top, xtol=np.finfo(float).tiny, rtol=4 * np.finfo(float).eps, maxiter=5000))


def _k_cap(r2dxj: float, r2yxj: float) -> tuple[float, str]:
    """Largest k (kd = ky = k) at which the bounds are informative: the k where the implied partial R² of the
    confounder with the outcome reaches 1 ("outcome"), or the largest admissible kd if that comes first or the
    benchmark explains nothing of the outcome ("kd_max"). r2yz_dx increases in k, so the root is unique."""
    from scipy.optimize import brentq
    r2dxj = 0.0 if r2dxj <= _R2_ZERO else r2dxj  # rounding noise of an exact zero, as in _k_needed (3e-33)
    kmax = _kd_max(r2dxj)
    if r2yxj <= _R2_ZERO:
        return kmax, "kd_max"

    def h(k):
        return float(_ovb_partial_r2_bound(r2dxj, r2yxj, k, k)[1]) - 1.0

    if np.isfinite(kmax):
        top = kmax * (1 - 1e-12)
        if not h(top) > 0:
            return kmax, "kd_max"
    else:                                         # benchmark unrelated to D: r2yz = k·r2y/(1 − r2y)
        return (1 - r2yxj) / r2yxj, "outcome"
    return float(brentq(h, 0.0, top, xtol=np.finfo(float).tiny, rtol=4 * np.finfo(float).eps, maxiter=5000)), "outcome"


def _k_not_significant(r2dxj: float, r2yxj: float, est: float, se: float, df: float, tcrit: float):
    """Smallest k (kd = ky = k) at which the worst-case adjusted CI reaches zero, i.e. the estimate is no longer
    significant at the call's alpha, with the classical SE (v2.6.1). Returns (k, returns, status, k_cap, cap_by,
    k_again).

    Not a threshold in general: the adjusted SE shrinks as the confounder explains almost all of the outcome (or when
    the benchmark predicts the outcome much better than the treatment), so the CI can exclude zero again, on the same
    side, for a larger k before the cap. `returns` says so and `k_again` is the first such k (NaN otherwise).
    g(k) = sign(est)·adj(k) − t·SE_adj(k); g(0+) = |est| − t·SE·√(df/(df − 1)) is the same for every benchmark.
    status: "already" (the unadjusted CI includes zero; k = 0; `returns` then says that the worst-case CI excludes
    zero from `k_again` on, although the unadjusted one does not), "at once" (g(0+) <= 0: only significant within the
    df vs df − 1 band, so a confounder of vanishing strength, which costs one degree of freedom, removes it; k = 0),
    "found", "not reached" (k = inf: no k below the cap). Both statuses of k = 0 are decided analytically, so they are
    the same for every benchmark of a call. Otherwise g is evaluated on a dense grid on (0, k_cap) — linear, geometric
    from 0 and geometric toward the cap —, every local minimum of g before the first grid point at or below zero is
    refined (bounded minimize on the bracket rescaled to (0, 1), so SciPy's relative tolerance does not stop it near
    the cap), and the first sign change is refined with brentq (from k = 0, where g = g(0+) > 0, if needed)."""
    from scipy.optimize import brentq, minimize_scalar
    cap, cap_by = _k_cap(r2dxj, r2yxj)            # treats a partial R² <= _R2_ZERO as zero
    already = not abs(est) > tcrit * se
    c_ = tcrit * se * np.sqrt(df / (df - 1))
    g0 = abs(est) - c_

    def g(k):
        k = np.asarray(k, dtype=float)
        r2dz, r2yz = _ovb_partial_r2_bound(r2dxj, r2yxj, k, k)
        r2yz = np.minimum(r2yz, 1.0)
        S = np.sqrt((1 - r2yz) / (1 - r2dz))
        bias = np.sqrt(r2yz * r2dz / (1 - r2dz)) * se * np.sqrt(df)
        return abs(est) - bias - c_ * S           # sign(est)·adj − t·SE_adj (eq. 12–13 of the paper)

    def gs(x):
        if x == 0:
            return g0
        with np.errstate(divide="ignore", invalid="ignore"):
            v = float(g(x))
        return v if np.isfinite(v) else -np.inf

    def root(lo, hi):
        return float(brentq(gs, lo, hi, xtol=np.finfo(float).tiny, rtol=4 * np.finfo(float).eps, maxiter=5000))

    span = cap if np.isfinite(cap) else 1e6       # cap is finite unless the benchmark explains nothing of either
    top = span * (1 - 1e-12)
    grid = np.unique(np.concatenate([span * np.linspace(0, 1, 2001)[1:-1], span * np.geomspace(1e-10, 1e-3, 600),
                                     span * (1 - np.geomspace(1e-12, 1e-2, 1400)), [top]]))
    grid = grid[(grid > 0) & (grid <= top)]
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = g(grid)
    vals = np.where(np.isfinite(vals), vals, -np.inf)   # an infinite adjusted SE (r2dz -> 1) includes zero

    def again(m):
        """First k after m (where g <= 0) at which g > 0 again, refined; NaN if none below the cap. A return in the
        last 1e-12 of the cap, past the grid, is seen from g at the cap itself (R²_Y = 1: g = |est| − bias)."""
        pos = np.flatnonzero((grid > m) & (vals > 0))
        if len(pos):
            j = int(pos[0])
            return root(max(m, grid[j - 1]) if j > 0 else m, grid[j])
        if cap_by == "outcome" and np.isfinite(cap) and gs(cap) > 0:
            return root(max(m, grid[-1]), cap)
        return np.nan

    if already:                                   # v2.6.1 review 2: the CI can exclude zero for a larger k
        ka = again(0.0)
        return 0.0, bool(np.isfinite(ka)), "already", cap, cap_by, ka
    if g0 <= 0:
        ka = again(0.0)
        return 0.0, bool(np.isfinite(ka)), "at once", cap, cap_by, ka
    hit = np.flatnonzero(vals <= 0)
    i = int(hit[0]) if len(hit) else len(grid)
    # a dip narrower than the grid step: refine every local minimum before the first grid point at or below zero
    lo_b = hi_b = None
    for j in np.flatnonzero((vals[1:-1] < vals[:-2]) & (vals[1:-1] <= vals[2:])) + 1:
        if j >= i:
            break
        a_, w_ = grid[j - 1], grid[j + 1] - grid[j - 1]
        r = minimize_scalar(lambda u: gs(a_ + u * w_), bounds=(0.0, 1.0), method="bounded",
                            options={"xatol": 1e-12, "maxiter": 500})
        if r.fun <= 0:
            lo_b, hi_b = a_, float(a_ + r.x * w_)
            break
    if lo_b is None:
        if i == len(grid):
            return np.inf, False, "not reached", cap, cap_by, np.nan
        lo_b, hi_b = (grid[i - 1] if i > 0 else 0.0), grid[i]
    k = root(lo_b, hi_b)
    ka = again(hi_b)
    return k, bool(np.isfinite(ka)), "found", cap, cap_by, ka


def _k_list(vals) -> str:
    """kd or ky values for a warning, in the order given (at most six of them)."""
    return ", ".join(f"{v:g}" for v in vals)


def _k_range(vals) -> str:
    """min to max of kd or ky values (one value when they are all equal)."""
    lo, hi = min(vals), max(vals)
    return f"{lo:g}" if f"{lo:g}" == f"{hi:g}" else f"{lo:g} to {hi:g}"


def _r2_above1(v: float) -> str:
    """An implied R² above 1, never printed as "1" (3 significant digits would round 1.000001 to 1)."""
    return f"{v:.3g}" if f"{v:.3g}" != "1" else "just above 1"


def _level_pct(alpha: float) -> str:
    """Confidence level as printed: 6 decimals, more if that would round to 100% (alpha = 1e-9)."""
    s = f"{round((1 - alpha) * 100, 6):g}"
    return (s if s != "100" else f"{(1 - alpha) * 100:.15g}") + "%"


def _share_pct(share: float) -> str:
    """Share of the estimate left, never rounded to a misleading 0% or 100%."""
    v = 100 * share
    if 0 < v < 0.5:
        return "less than 1%"
    if 99.5 <= v < 100:
        return "more than 99%"
    return f"{v:.0f}%"


_KD_DEFAULT = (1, 2, 3)


def ovb_bounds(res, treatment: str, benchmarks, kd=None, ky=None, q: float = 1.0, alpha: float = ALPHA,
               show: bool = False) -> dict:
    """Formal Cinelli-Hazlett (2020) bounds for a fitted OLS: what a confounder k times as strong as an observed
    covariate (or group of covariates) could do to the estimate of `treatment`. Replicates R's
    sensemakr::ovb_bounds (partial-R² bound, reduce = TRUE) and adds the k that would remove 100·q% of it.

    benchmarks  a regressor name, or a list whose items are regressor names, categorical names ("C(g)" or "g":
                the group of its dummies, grouped as `sensitivity` groups them) or dicts {label: [terms]} for
                groups of your own (a term may itself be a categorical's name); or one such dict.
    kd, ky      how many times the benchmark's explained variance of the treatment (kd) and of the outcome (ky)
                the confounder has, given the other covariates; a number or a sequence; ky = None means ky = kd;
                sequences are paired (one number pairs with every element of the other). kd = None (default)
                means (1, 2, 3) and skips, per benchmark, the values that are impossible for it (with a warning, also in
                out["warnings"], and a line in the printed text); a kd you give that is impossible stops the call, as in sensemakr.
    q           for `k_needed`: the share of the estimate the confounder would remove (1: to zero).

    Assumes a confounder orthogonal to the covariates in the model ("or the part of it they do not explain"):
    the bounds are worst cases under that premise, not estimates, and "k times" is in explained variance, not in
    coefficient size. Uses the classical SE (the formulas are partial-R² identities); a robust fit is refit
    classically, as in `sensitivity`. A kd at or beyond (1 − R²_D)/R²_D of a benchmark is impossible; an implied
    R² with the outcome above 1 is capped at 1 with a warning, as in sensemakr (the adjusted SE is then 0 and the
    CI is not informative). Needs a full-rank design (ValueError otherwise).

    Returns {"bounds": DataFrame (one row per benchmark × (kd, ky); ci_verdict in {"same sign", "includes zero",
    "opposite sign", "not informative"}; ci_excludes_zero is False on capped rows), "benchmark_r2": DataFrame (partial R² of each benchmark with D and with Y,
    the df used, the largest admissible kd, k_needed, and since v2.6.1 k_not_significant: the smallest k = kd = ky at
    which the worst-case CI reaches zero at alpha, with the classical SE like the bounds (0 when the estimate is not
    significant, or only within the df vs df − 1 band; inf when no k below k_cap does it), significance_returns (the CI
    excludes zero again, same side, for a larger k below the cap: then k_not_significant is not a threshold),
    k_significant_again (the first such k, NaN otherwise), k_not_significant_status ("already",
    "at once", "found", "not reached"), k_cap and k_cap_by ("outcome": the implied R² with Y reaches 1 there;
    "kd_max")), "estimate", "se", "df", "q", "alpha", "warnings", "classical_refit"}. k_not_significant is about
    zero whatever q; the capped warnings come one per benchmark and call (v2.6.1).
    """
    if not hasattr(res, "model") or type(res.model).__name__ != "OLS":
        raise TypeError("ovb_bounds needs an OLS fit from statsmodels (smf.ols(...).fit() or sm.OLS(...).fit()); "
                        f"got {type(getattr(res, 'model', res)).__name__}")
    q = _check_q(q)
    if not (1e-12 <= alpha < 1):
        raise ValueError(f"alpha must be in [1e-12, 1), got {alpha!r}")
    base = res if getattr(res, "cov_type", "nonrobust") == "nonrobust" else res.model.fit()
    names = list(base.params.index)
    if treatment not in names:
        raise KeyError(f"treatment {treatment!r} is not a regressor of the model; regressors: {names}")
    if treatment in _INTERCEPT:
        raise ValueError("the intercept cannot be the treatment")
    if base.df_resid < 2:
        raise ValueError(f"ovb_bounds needs at least 2 residual degrees of freedom (the confounder costs one); got "
                         f"{base.df_resid:g}")
    exog_np = np.asarray(base.model.exog, dtype=float)
    norms = np.linalg.norm(exog_np, axis=0)
    # rank of the column-scaled design: a regressor in units of 1e12 next to O(1) columns is not collinear
    if (norms == 0).any() or np.linalg.matrix_rank(exog_np / norms) < exog_np.shape[1]:
        raise ValueError("the design matrix is not of full rank (a regressor is a linear combination of others): "
                         "the partial R² are not defined; drop the redundant regressor and refit")
    default_kd = kd is None
    kd_arr = _check_k(_KD_DEFAULT if default_kd else kd, "kd")
    ky_arr = kd_arr.copy() if ky is None else _check_k(ky, "ky")
    if ky is not None and len(ky_arr) != len(kd_arr):
        if len(ky_arr) == 1:
            ky_arr = np.repeat(ky_arr, len(kd_arr))
        elif len(kd_arr) == 1:
            kd_arr = np.repeat(kd_arr, len(ky_arr))
        else:
            raise ValueError(f"kd and ky are paired: give them the same length (or one number), got {len(kd_arr)} "
                             f"and {len(ky_arr)}" + (" (kd not given: its default is 1, 2, 3)" if default_kd else ""))

    # benchmarks -> {label: [regressors]}
    groups = _covariate_groups(base, treatment)

    def resolve(b) -> list[str]:
        if b in names:
            return [b]
        if b in groups:
            return groups[b]
        if f"C({b})" in groups:
            return groups[f"C({b})"]
        hits = [g for g in groups if re.match(r"^C\(\s*" + re.escape(str(b)) + r"\s*[,)]", g)]
        if len(hits) == 1:                        # C(g, Treatment('c')) by "g"
            return groups[hits[0]]
        inner = re.match(r"^C\(\s*([^,()]+?)\s*\)$", str(b))
        if inner and inner.group(1) != b:         # "C(g)" for a string factor g[T.b] or for C(g, Treatment(...))
            try:
                return resolve(inner.group(1))
            except KeyError:
                pass
        raise KeyError(f"benchmark {b!r} is neither a regressor nor a categorical of the model; regressors: {names}")

    items = [benchmarks] if isinstance(benchmarks, (str, dict)) else list(benchmarks)
    spec: dict[str, list[str]] = {}
    for it in items:
        if not isinstance(it, (str, dict)):
            raise TypeError(f"each benchmark is a name or a dict {{label: [terms]}} for a group; got {it!r} (for a group "
                            f"of your own, write {{'label': {list(it) if isinstance(it, (list, tuple)) else it!r}}})")
        if isinstance(it, dict):
            for lab, terms in it.items():
                terms = [terms] if isinstance(terms, str) else list(terms)
                if not terms:
                    raise ValueError(f"benchmark group {lab!r} is empty")
                flat = [t for b in terms for t in (resolve(b) if b != treatment else [b])]
                spec[str(lab)] = list(dict.fromkeys(flat))       # a term listed twice would make V singular
        else:
            spec[str(it)] = resolve(it)
    if not spec:
        raise ValueError("give at least one benchmark")
    for lab, terms in spec.items():
        for t_ in terms:
            if t_ == treatment:
                raise ValueError(f"the treatment {treatment!r} cannot be its own benchmark")
            if t_ in _INTERCEPT or t_ not in names:
                raise KeyError(f"benchmark {lab!r}: {t_!r} is not a covariate of the model; regressors: {names}")

    warns: list[str] = []

    def note(msg):
        warns.append(msg)
        warnings.warn(msg, stacklevel=3)

    inter = [c for c in names if ":" in c and treatment in c.split(":")]
    if inter:
        note(f"{treatment} also enters the interaction(s) {inter}: the bounds are for its coefficient alone, i.e. the "
             f"effect where the interacted variable(s) are 0")

    # treatment regression: D on the other columns of the same design (same sample), df = df_resid + 1
    exog = pd.DataFrame(exog_np, columns=names)
    tmod = sm.OLS(exog[treatment], exog.drop(columns=[treatment])).fit()
    est, se, df = float(base.params[treatment]), float(base.bse[treatment]), float(base.df_resid)
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    sign = -1.0 if est < 0 else 1.0               # as sensemakr: an estimate of exactly 0 counts as positive
    rows, brows = [], []
    for lab, terms in spec.items():
        r2y = _partial_r2_terms(base.params, base.cov_params(), base.df_resid, terms)
        r2d = _partial_r2_terms(tmod.params, tmod.cov_params(), tmod.df_resid, terms)
        kmax = _kd_max(r2d)
        bad = kd_arr >= kmax
        kd_b, ky_b = kd_arr, ky_arr
        if bad.any():
            if not default_kd:
                raise ValueError(f"impossible kd = {kd_arr[bad][0]:g} for benchmark {lab!r}: {lab} explains "
                                 f"{r2d:.4%} of the residual variance of {treatment}, so a confounder kd times as strong "
                                 f"would explain 100% or more; kd must be below {kmax:.6g}. The call stops, as in "
                                 f"sensemakr: lower kd or drop that benchmark")
            note(f"{lab}: kd = {', '.join(f'{v:g}' for v in kd_arr[bad])} skipped (impossible: {lab} explains "
                 f"{r2d:.4%} of the residual variance of {treatment}; kd must be below {kmax:.6g})")
            kd_b, ky_b = kd_arr[~bad], ky_arr[~bad]
        r2dz, r2yz_raw = _ovb_partial_r2_bound(r2d, r2y, kd_b, ky_b)
        capped = r2yz_raw > 1
        r2yz = np.where(capped, 1.0, r2yz_raw)
        ci_ = np.flatnonzero(capped)
        if len(ci_):                              # v2.6.1: one warning per benchmark and call (was one per line)
            pairs = {}                            # (kd, ky) as printed -> implied R², each once, in the order given
            for i_ in ci_:
                pairs.setdefault((f"{kd_b[i_]:g}", f"{ky_b[i_]:g}"), float(r2yz_raw[i_]))
            kds, kys = [float(p_[0]) for p_ in pairs], [float(p_[1]) for p_ in pairs]
            same = [p_[0] for p_ in pairs] == [p_[1] for p_ in pairs]
            if len(pairs) <= 6:
                ks_txt = f"kd = ky = {_k_list(kds)}" if same else f"kd = {_k_list(kds)} with ky = {_k_list(kys)}"
            else:                                 # many lines: ranges, and the count of (kd, ky) pairs
                ks_txt = (f"kd = ky = {_k_range(kds)}" if same else
                          f"kd = {_k_range(kds)} with ky = {_k_range(kys)}") + f" ({len(pairs)} (kd, ky) pairs)"
            r2v = list(pairs.values())
            r2_txt = (", ".join(_r2_above1(v) for v in r2v) if len(r2v) <= 6 else
                      f"{_r2_above1(min(r2v))} to {_r2_above1(max(r2v))}")
            one_ = len(pairs) == 1
            note(f"{lab}, {ks_txt}: the implied partial R² of the confounder with the outcome is {r2_txt}"
                 f"{'' if r2_txt == 'just above 1' else ' > 1'}; "
                 f"capped at 1 (try a lower kd and/or ky); {'its CI is' if one_ else 'the CIs of those lines are'} "
                 f"not informative")
        adj, adj_se, adj_t = _ovb_adjust(est, se, df, r2dz, r2yz)
        for i in range(len(kd_b)):
            kdi, kyi = kd_b[i], ky_b[i]
            mult = f"{round(kdi, 2):g}" if kdi == kyi else f"{round(kdi, 2):g}/{round(kyi, 2):g}"
            lo, hi = adj[i] - tcrit * adj_se[i], adj[i] + tcrit * adj_se[i]
            excl = bool((lo > 0 or hi < 0) and not capped[i])   # a capped row's CI has width 0: no verdict
            if capped[i]:
                verdict = "not informative"
            elif not excl:
                verdict = "includes zero"
            else:
                verdict = "same sign" if (lo > 0) == (sign > 0) else "opposite sign"
            rows.append({"bound_label": f"{mult}x {lab}", "benchmark": lab, "kd": kdi, "ky": kyi,
                         "r2dz_x": r2dz[i], "r2yz_dx": r2yz[i], "r2yz_dx_capped": bool(capped[i]),
                         "adjusted_estimate": adj[i], "adjusted_se": adj_se[i], "adjusted_t": adj_t[i],
                         "adjusted_lower_ci": lo, "adjusted_upper_ci": hi,
                         "share_of_estimate": adj[i] / est if est != 0 else np.nan,
                         "ci_excludes_zero": excl, "ci_verdict": verdict})
        kns, kret, kstat, kcap, kcap_by, kagain = _k_not_significant(r2d, r2y, est, se, df, tcrit)
        brows.append({"benchmark": lab, "terms": list(terms), "r2dxj_x": r2d, "r2yxj_dx": r2y,
                      "df_treatment": float(tmod.df_resid), "df_outcome": df, "kd_max": kmax,
                      "k_needed": _k_needed(r2d, r2y, est, se, df, q), "k_not_significant": kns,
                      "significance_returns": kret, "k_significant_again": kagain,
                      "k_not_significant_status": kstat, "k_cap": kcap, "k_cap_by": kcap_by})
    cols = ["bound_label", "benchmark", "kd", "ky", "r2dz_x", "r2yz_dx", "r2yz_dx_capped", "adjusted_estimate",
            "adjusted_se", "adjusted_t", "adjusted_lower_ci", "adjusted_upper_ci", "share_of_estimate",
            "ci_excludes_zero", "ci_verdict"]
    out = {"bounds": pd.DataFrame(rows, columns=cols), "benchmark_r2": pd.DataFrame(brows).set_index("benchmark"),
           "estimate": est, "se": se, "df": df, "q": q, "alpha": alpha, "warnings": warns,
           "classical_refit": base is not res}

    if show:
        level = _level_pct(alpha)
        orig_lo, orig_hi = est - tcrit * se, est + tcrit * se
        orig_excl = orig_lo > 0 or orig_hi < 0
        se_lab = "classical SE" if base is not res else "SE"   # v2.6.1: label the baseline next to a robust fit
        print(f"\nBounds on omitted-variable bias for {treatment} (Cinelli-Hazlett 2020, OLS): estimate {est:.4g} "
              f"({se_lab} {se:.4g}, df {df:g}; {level} CI [{orig_lo:.4g}, {orig_hi:.4g}]).")
        print("Each line is a worst case for a confounder orthogonal to the covariates in the model that explains up to "
              "k times as much of the residual variance of the treatment, and of the outcome, as the benchmark does "
              "given the other covariates; the partial R² on each line are given all the covariates, so they can "
              "exceed k times the benchmark's. 'k times' is in explained variance, not in coefficient size.")
        for r in rows:
            b = r["benchmark"]
            who = (f"{r['kd']:g}x as strong as {b}" if r["kd"] == r["ky"] else
                   f"{r['kd']:g}x as strong as {b} in explaining {treatment} and {r['ky']:g}x in explaining the outcome")
            cap = ", capped at 100%" if r["r2yz_dx_capped"] else ""
            a = r["adjusted_estimate"]
            if a == est:
                move = f"would leave the estimate at {est:.4g} (no bias)"
            elif est == 0:
                move = f"could move the estimate from 0 to {a:.4g}"
            elif a == 0:
                move = f"could bring the estimate from {est:.4g} to zero"
            elif (a < 0) != (est < 0):
                move = f"could move the estimate from {est:.4g} past zero, to {a:.4g}"
            else:
                move = (f"could move the estimate toward zero, from {est:.4g} to {a:.4g} "
                        f"({_share_pct(r['share_of_estimate'])} of it)")
            v = r["ci_verdict"]
            if v == "not informative":
                ci = ("its CI is not informative (with the outcome fully explained the adjusted SE is 0, "
                      f"so the interval collapses to {a:.4g})")
            else:
                ci = f"{level} CI [{r['adjusted_lower_ci']:.4g}, {r['adjusted_upper_ci']:.4g}], which " + {
                    "includes zero": "includes zero",
                    "same sign": "still excludes zero" if orig_excl else "excludes zero (the unadjusted CI did not)",
                    "opposite sign": "excludes zero on the other side (the sign would flip)"}[v]
            print(f"  {r['bound_label']}: a confounder {who} (partial R² up to {r['r2dz_x']:.1%} with {treatment} and "
                  f"{r['r2yz_dx']:.1%} with the outcome{cap}) {move}; {ci}.")
        for lab, br in out["benchmark_r2"].iterrows():
            if not (out["bounds"]["benchmark"] == lab).any():
                print(f"  {lab}: no line, because no default k (1, 2, 3) is admissible: {lab} explains "
                      f"{br['r2dxj_x']:.1%} of the residual variance of {treatment}, so k must be below "
                      f"{br['kd_max']:.4g}; pass a smaller kd to see bounds for it.")
        qq = out["q"]
        one = _q_is_one(qq)
        goal = "bring the estimate to zero" if one else (
            f"reduce the estimate by {float(qq) * 100:.6g}% (to {1 - float(qq):.6g} times its value)")
        for lab, br in out["benchmark_r2"].iterrows():
            kn = br["k_needed"]
            if kn == 0:
                print(f"  The estimate is exactly 0: no confounder is needed to {goal}.")
            elif np.isinf(kn) and min(br["r2dxj_x"], br["r2yxj_dx"]) <= _R2_ZERO:
                side = "treatment" if br["r2dxj_x"] <= _R2_ZERO else "outcome"
                print(f"  No multiple of {lab} could {goal}: {lab} explains nothing of the {side} "
                      f"(given the other covariates).")
            elif np.isinf(kn):
                print(f"  No admissible multiple of {lab} could {goal} (it would take k at the largest admissible, "
                      f"{br['kd_max']:.6g}).")
            else:
                print(f"  A confounder {kn:.3g}x as strong as {lab} (same k for treatment and outcome) could, in the "
                      f"worst case, {goal}.")
        sig = f"{round(100 * alpha, 6):g}%"
        stats_ = set(out["benchmark_r2"]["k_not_significant_status"])   # "already"/"at once" hold for the whole call
        cls = " (classical SE)" if base is not res else ""
        if "already" in stats_:
            print(f"  The estimate is not significant at the {sig} level{cls} even with no confounder (its {level} CI "
                  f"includes zero).")
        elif "at once" in stats_:
            print(f"  The estimate is only just significant at the {sig} level{cls}: a confounder of vanishing strength "
                  f"already makes it not significant (it costs one degree of freedom).")
        why = "because the worst-case adjusted SE shrinks"
        for lab, br in out["benchmark_r2"].iterrows():
            st, kn, ka = br["k_not_significant_status"], br["k_not_significant"], br["k_significant_again"]
            if st == "found":
                txt = (f"  A confounder {kn:.3g}x as strong as {lab} (same k for treatment and outcome) could, in the "
                       f"worst case, make the estimate not significant at the {sig} level{cls} (its CI reaches zero; "
                       f"the smallest such k on a dense grid).")
                if br["significance_returns"]:
                    txt += (f" From {ka:.3g}x on, before the outcome is fully explained, the worst-case CI excludes zero "
                            f"again on the same side, {why}: read {kn:.3g}x as the first k at which significance is "
                            f"lost, not as a threshold.")
                print(txt)
            elif st == "at once" and br["significance_returns"]:
                print(f"  For {lab}: from {ka:.3g}x on (same k for treatment and outcome) the worst-case CI{cls} "
                      f"excludes zero again, {why}; only a confounder weaker than that makes the estimate not "
                      f"significant.")
            elif st == "already" and br["significance_returns"]:
                print(f"  For {lab}: from {ka:.3g}x on (same k for treatment and outcome) the worst-case CI{cls} "
                      f"excludes zero on the side of the estimate, although the unadjusted CI does not, {why}; this "
                      f"is a property of the worst-case bound, not evidence of an effect.")
            elif st == "not reached" and not np.isfinite(br["k_cap"]):
                print(f"  No multiple of {lab} makes the estimate not significant{cls}: {lab} explains nothing of the "
                      f"treatment or of the outcome (given the other covariates).")
            elif st == "not reached":
                print(f"  No k below {br['k_cap']:.4g} makes the estimate not significant{cls} for {lab} " + (
                      f"(at that k a confounder as strong as {lab} would explain all of the outcome; beyond it the CI "
                      f"of the bounds is not informative)." if br["k_cap_by"] == "outcome" else
                      "(the largest admissible k; a larger one is impossible)."))
        if base is not res:
            t_cl, t_rb = float(base.tvalues[treatment]), float(res.tvalues[treatment])
            rb = (f"your {res.cov_type} fit gives t = {t_rb:.2f}, so its significance can differ" if np.isfinite(t_rb)
                  and np.isfinite(float(res.bse[treatment])) and float(res.bse[treatment]) > 0 else
                  f"your {res.cov_type} SE is not finite or zero (often observations with leverage 1), so its t is "
                  f"not usable")
            print(f"These bounds use the classical SE, as the formulas require (t = {t_cl:.2f}); {rb}.")
        for w in warns:
            print("  WARNING: " + w)
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
