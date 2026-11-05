"""Systems of equations and dynamic panels (v2.4.7). Reference: panel-iv.md sections 5 and 11.

Use as a module:
    import sys; sys.path.insert(0, "<skill folder>/scripts")
    from systems import *

Functions
    sur_system(formulas, data)           SUR (feasible GLS) with the Breusch-Pagan LM test of independent errors,
                                         the residual correlations and the OLS comparison (identical when every
                                         equation has the same regressors)
    three_sls(equations, data)           3SLS with the order condition checked equation by equation, against 2SLS
                                         equation by equation
    dynamic_panel(df, y, x, entity, time) Arellano-Bond difference GMM or Blundell-Bond system GMM: one-step
                                         robust and two-step with the Windmeijer correction, AR(1)/AR(2) tests,
                                         Hansen J, instrument count against groups, and the OLS/within bounds
                                         for the lagged dependent variable

Rule (interpretation-rules.md): a dynamic-panel estimate is not a result without the AR(2) test, the Hansen J
and the instrument count. Every number comes from the data passed in.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import _console  # noqa: F401  (Windows-safe console output, see _console.py)

ALPHA = 0.05


def _fmt(v):
    return f"{v:.4f}" if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v)


# =============================================================================
# SUR
# =============================================================================
def sur_system(formulas: dict, data: pd.DataFrame, cov_type: str = "unadjusted", show: bool = True) -> dict:
    """Seemingly unrelated regressions: equations linked only through correlated errors.

    formulas: {name: "y ~ 1 + x1 + x2", ...}. SUR gains precision over OLS only when the errors are
    correlated across equations AND the regressors differ; with identical regressors SUR equals OLS equation
    by equation. Breusch-Pagan LM: T·Σ_{i<j} r_ij² ~ χ²(M(M−1)/2), from the OLS residual correlations
    (H0: diagonal error covariance, so SUR adds nothing)."""
    from linearmodels.system import SUR
    import statsmodels.formula.api as smf
    if len(formulas) < 2:
        raise ValueError("a system needs at least two equations")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SUR.from_formula(formulas, data).fit(cov_type=cov_type)
    ols = {}
    resid = {}
    for name, f in formulas.items():
        o = smf.ols(f, data).fit()
        ols[name] = o
        resid[name] = o.resid
    R = pd.DataFrame(resid).dropna()
    T = len(R)
    corr = R.corr()
    M = len(formulas)
    lm = float(T * sum(corr.iloc[i, j] ** 2 for i in range(M) for j in range(i)))
    df_lm = M * (M - 1) // 2
    p_lm = float(stats.chi2.sf(lm, df_lm))
    rows = []
    for name, o in ols.items():
        for v in o.params.index:
            key = f"{name}_{v}"
            rows.append({"equation": name, "variable": v, "ols": o.params[v], "ols_se": o.bse[v],
                         "sur": res.params.get(key, np.nan), "sur_se": res.std_errors.get(key, np.nan)})
    comp = pd.DataFrame(rows)
    same_x = len({tuple(sorted(o.params.index)) for o in ols.values()}) == 1
    reading = ("the errors are correlated across equations (SUR can gain precision)" if p_lm < ALPHA else
               "no evidence of correlated errors: SUR adds little over OLS")
    if same_x:
        reading += "; every equation has the same regressors, so SUR equals OLS equation by equation"
    out = {"results": res, "comparison": comp, "residual_corr": corr, "bp_lm": lm, "bp_df": df_lm, "bp_p": p_lm,
           "same_regressors": same_x, "reading": reading, "nobs": T, "method": "SUR (FGLS), linearmodels"}
    if show:
        print(f"\nSUR, {M} equations, T = {T}")
        print(comp.to_string(index=False, float_format=_fmt))
        print("\nResidual correlations (OLS):\n" + corr.to_string(float_format=_fmt))
        print(f"Breusch-Pagan LM = {lm:.3f}, df = {df_lm}, p = {p_lm:.4f} → {reading}")
    return out


# =============================================================================
# 3SLS
# =============================================================================
def three_sls(equations: dict, data: pd.DataFrame, instruments: list | None = None, show: bool = True) -> dict:
    """Three-stage least squares for a simultaneous system, with identification checked first.

    equations: {name: {"y": dependent, "exog": [included exogenous], "endog": [included endogenous]}}; a
    constant is added to every equation. Instruments for every equation: all exogenous variables of the
    system plus `instruments` (excluded variables that appear in no equation). Order condition, per equation:
    excluded instruments ≥ included endogenous (necessary, not sufficient: the excluded variables must also
    move the endogenous ones, which the first-stage F shows). 3SLS is efficient when every equation is
    correctly specified; a misspecified equation contaminates the others, while 2SLS confines the damage, so
    both are reported."""
    from linearmodels.iv import IV2SLS
    from linearmodels.system import IV3SLS
    endog_all = {v for e in equations.values() for v in e.get("endog", [])} | {e["y"] for e in equations.values()}
    clash = sorted({v for e in equations.values() for v in e.get("exog", [])} & endog_all)
    if clash:
        raise ValueError(f"{clash} are endogenous (dependent or endogenous in some equation) but listed as exogenous "
                         "in another: list them under 'endog' there")
    sys_exog = sorted({v for e in equations.values() for v in e.get("exog", [])} | set(instruments or []))
    rows, formulas, eq2sls = [], {}, {}
    for name, e in equations.items():
        exog, endog = list(e.get("exog", [])), list(e.get("endog", []))
        excluded = [v for v in sys_exog if v not in exog]
        status = ("no endogenous regressors" if not endog else "exactly identified" if len(excluded) == len(endog) else
                  "over-identified" if len(excluded) > len(endog) else "NOT identified")
        rows.append({"equation": name, "included endogenous": len(endog), "excluded instruments": len(excluded),
                     "order condition": status})
        rhs = " + ".join(["1"] + exog)
        if endog:
            rhs += f" + [{' + '.join(endog)} ~ {' + '.join(excluded) if excluded else '1'}]"
        formulas[name] = f"{e['y']} ~ {rhs}"
    order = pd.DataFrame(rows)
    if (order["order condition"] == "NOT identified").any():
        bad = order.loc[order["order condition"] == "NOT identified", "equation"].tolist()
        raise ValueError(f"order condition fails for {bad}: add excluded instruments or restrictions")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = IV3SLS.from_formula(formulas, data).fit(cov_type="unadjusted")
        for name, f in formulas.items():
            eq2sls[name] = IV2SLS.from_formula(f, data).fit(cov_type="unadjusted")
    comp = []
    for name, r2 in eq2sls.items():
        for v in r2.params.index:
            key = f"{name}_{v}"
            comp.append({"equation": name, "variable": v, "2sls": r2.params[v], "2sls_se": r2.std_errors[v],
                         "3sls": res.params.get(key, np.nan), "3sls_se": res.std_errors.get(key, np.nan)})
    comp = pd.DataFrame(comp)
    first = {}
    for name, r2 in eq2sls.items():
        try:
            fs = r2.first_stage.diagnostics
            first[name] = fs[["f.stat", "f.pval"]] if "f.stat" in fs else fs
        except Exception:
            first[name] = None
    exact = bool(order["order condition"].isin(["exactly identified", "no endogenous regressors"]).all())
    reading = ("every equation is exactly identified: 3SLS equals 2SLS equation by equation (nothing to gain)"
               if exact else "over-identified equations: 3SLS uses the cross-equation error correlation")
    out = {"results": res, "order_condition": order, "comparison": comp, "first_stage": first, "reading": reading,
           "formulas": formulas, "instruments": sys_exog, "method": "3SLS (linearmodels) with 2SLS per equation"}
    if show:
        print("\nOrder condition by equation:\n" + order.to_string(index=False))
        print("\n2SLS vs 3SLS:\n" + comp.to_string(index=False, float_format=_fmt))
        for name, fs in first.items():
            if fs is not None and len(fs):
                print(f"First stage ({name}):\n" + fs.to_string(float_format=_fmt))
        print("→", reading)
    return out


# =============================================================================
# Dynamic panel GMM
# =============================================================================
def _panel_arrays(df, y, x, entity, time):
    d = df[[entity, time, y] + list(x)].copy()
    d = d.sort_values([entity, time])
    times = np.sort(d[time].unique())
    tpos = {t: i for i, t in enumerate(times)}
    T = len(times)
    units = []
    for g, sub in d.groupby(entity, sort=True):
        Y = np.full(T, np.nan)
        Xm = np.full((T, len(x)), np.nan)
        idx = sub[time].map(tpos).to_numpy()
        Y[idx] = sub[y].to_numpy(float)
        Xm[idx] = sub[list(x)].to_numpy(float)
        units.append((g, Y, Xm))
    return units, times


def _build(units, lags, x, x_kind, system, gmm_min, gmm_max, collapse):
    """Per unit: (yv, Xv, Zrows, n_diff_rows). Returns list and the instrument column labels."""
    kx = len(x)
    col_index: dict = {}

    def col(key):
        if key not in col_index:
            col_index[key] = len(col_index)
        return col_index[key]
    per_unit = []
    for g, Y, Xm in units:
        T = len(Y)
        rows_y, rows_X, rows_Z, eq = [], [], [], []
        dY = np.r_[np.nan, np.diff(Y)]
        dX = np.vstack([np.full((1, kx), np.nan), np.diff(Xm, axis=0)]) if kx else np.empty((T, 0))
        # difference equation
        for t in range(lags + 1, T):
            reg = [dY[t - L] for L in range(1, lags + 1)] + list(dX[t])
            if not np.isfinite(dY[t]) or not np.all(np.isfinite(reg)):
                continue
            z = {}
            hi = t - gmm_min
            lo = 0 if gmm_max is None else max(0, t - gmm_max)
            for s in range(lo, hi + 1):
                if np.isfinite(Y[s]):
                    z[col(("dy", t - s) if collapse else ("dy", t, s))] = Y[s]
            for m, v in enumerate(x):
                kind = x_kind.get(v, "exog")
                if kind == "exog":
                    z[col(("dx", v))] = dX[t, m]
                else:
                    first = 2 if kind == "endog" else 1
                    for s in range(lo, t - first + 1):
                        if np.isfinite(Xm[s, m]):
                            z[col((f"gx_{v}", t - s) if collapse else (f"gx_{v}", t, s))] = Xm[s, m]
            rows_y.append(dY[t])
            rows_X.append(reg + ([0.0] if system else []))
            rows_Z.append(z)
            eq.append(("d", t))
        n_diff = len(rows_y)
        if system:
            for t in range(lags, T):
                reg = [Y[t - L] for L in range(1, lags + 1)] + list(Xm[t])
                if not np.isfinite(Y[t]) or not np.all(np.isfinite(reg)):
                    continue
                z = {}
                # the level instrument is the difference dated where the level instruments of the differenced
                # equation start: Δy_{t-1} with gmm_lags starting at 2, Δy_{t-2} when they start at 3
                lag_l = gmm_min - 1
                if t - lag_l >= 1 and np.isfinite(dY[t - lag_l]):
                    z[col(("ly", 1) if collapse else ("ly", t))] = dY[t - lag_l]
                for m, v in enumerate(x):
                    kind = x_kind.get(v, "exog")
                    if kind == "exog":
                        # Δx_t, not x_t: x in levels is usually correlated with η_i, which is in the level error
                        if np.isfinite(dX[t, m]):
                            z[col(("lx", v))] = dX[t, m]
                    else:
                        lag = 1 if kind == "endog" else 0
                        if t - lag >= 1 and np.isfinite(dX[t - lag, m]):
                            z[col((f"lgx_{v}", 1) if collapse else (f"lgx_{v}", t))] = dX[t - lag, m]
                z[col(("const",))] = 1.0
                rows_y.append(Y[t])
                rows_X.append(reg + [1.0])
                rows_Z.append(z)
                eq.append(("l", t))
        if rows_y:
            per_unit.append((g, np.array(rows_y), np.array(rows_X, float), rows_Z, n_diff, eq))
    return per_unit, col_index


def dynamic_panel(df: pd.DataFrame, y: str, x: list | None = None, entity: str = "entity", time: str = "time",
                  lags: int = 1, method: str = "difference", x_kind: dict | None = None, gmm_lags=(2, None),
                  collapse: bool = False, two_step: bool = True, onestep_weight: str = "H", alpha: float = ALPHA,
                  show: bool = True) -> dict:
    """Dynamic panel y_it = Σ ρ_L y_i,t−L + x_it'β + η_i + ε_it by GMM.

    method 'difference' (Arellano-Bond 1991): first differences remove η_i; Δy_i,t−1 is instrumented by
    y_i,t−2, y_i,t−3, ... (GMM-style, one column per period and lag unless collapse=True). 'system'
    (Blundell-Bond 1998) adds the level equation instrumented by Δy_i,t−1, valid when the deviations from
    the steady state are uncorrelated with η_i; it helps when ρ is near 1 and the lagged levels are weak
    instruments for the differences. x_kind: {var: 'exog' (default: Δx instruments itself), 'predetermined'
    (lags from t−1) or 'endog' (lags from t−2)}. gmm_lags: (first, last) lag of y used as instrument.
    One-step weight: H (2 on the diagonal, −1 next to it) for the differences, identity for the levels and
    zero between them (xtabond2's h(2)); two-step uses the one-step residuals and Windmeijer (2005) corrected
    SEs. Exogenous x enter the level equation through Δx (xtabond2's iv(x) would use x in levels), so system
    GMM numbers differ from an xtabond2 run with defaults; two-step difference GMM does not depend on H.

    Read nothing before the checks: AR(1) in the differenced residuals is expected; AR(2) must not reject (if
    it does, y_t−2 is not a valid instrument: start the instruments at lag 3). Hansen J must not reject, but
    a p-value near 1 with many instruments means the test is weakened (instrument proliferation): instruments
    should not exceed groups (collapse or limit the lags). The estimate of ρ should lie between the within
    estimate (biased down) and pooled OLS (biased up)."""
    x = list(x or [])
    x_kind = dict(x_kind or {})
    if method not in ("difference", "system"):
        raise ValueError("method must be 'difference' or 'system'")
    system = method == "system"
    units, times = _panel_arrays(df, y, x, entity, time)
    per_unit, cols = _build(units, lags, x, x_kind, system, gmm_lags[0], gmm_lags[1], collapse)
    N = len(per_unit)
    L = len(cols)
    names = [f"L{L_}.{y}" for L_ in range(1, lags + 1)] + x + (["const"] if system else [])
    k = len(names)
    Zs, Xs, ys, Hs = [], [], [], []
    for g, yy, XX, rows_Z, n_diff, eq in per_unit:
        Z = np.zeros((len(yy), L))
        for i, z in enumerate(rows_Z):
            for c, v in z.items():
                Z[i, c] = v
        n = len(yy)
        H = np.zeros((n, n))
        for i in range(n_diff):
            H[i, i] = 2.0
            for j in range(n_diff):
                if abs(eq[i][1] - eq[j][1]) == 1:
                    H[i, j] = -1.0
        for i in range(n_diff, n):
            H[i, i] = 1.0
        if onestep_weight == "identity":
            H = np.eye(n)
        Zs.append(Z), Xs.append(XX), ys.append(yy), Hs.append(H)
    keep = np.any([np.any(Z != 0, axis=0) for Z in Zs], axis=0)
    Zs = [Z[:, keep] for Z in Zs]
    L = int(keep.sum())
    ZX = sum(Z.T @ X for Z, X in zip(Zs, Xs))
    Zy = sum(Z.T @ yy for Z, yy in zip(Zs, ys))
    ZHZ = sum(Z.T @ H @ Z for Z, H in zip(Zs, Hs))
    W1 = np.linalg.pinv(ZHZ)
    A1 = np.linalg.inv(ZX.T @ W1 @ ZX)
    b1 = A1 @ ZX.T @ W1 @ Zy
    u1 = [yy - X @ b1 for yy, X in zip(ys, Xs)]
    S1 = sum(Z.T @ np.outer(u, u) @ Z for Z, u in zip(Zs, u1))
    V1 = A1 @ ZX.T @ W1 @ S1 @ W1 @ ZX @ A1
    W2 = np.linalg.pinv(S1)
    A2 = np.linalg.inv(ZX.T @ W2 @ ZX)
    b2 = A2 @ ZX.T @ W2 @ Zy
    u2 = [yy - X @ b2 for yy, X in zip(ys, Xs)]
    Zu2 = sum(Z.T @ u for Z, u in zip(Zs, u2))
    # Windmeijer (2005)
    D = np.zeros((k, k))
    for j in range(k):
        dS = -sum(Z.T @ (np.outer(X[:, j], u) + np.outer(u, X[:, j])) @ Z for Z, X, u in zip(Zs, Xs, u1))
        D[:, j] = -A2 @ ZX.T @ W2 @ dS @ W2 @ Zu2
    V2 = A2 + D @ A2 + A2 @ D.T + D @ V1 @ D.T
    b, V, W, A, u = (b2, V2, W2, A2, u2) if two_step else (b1, V1, W1, A1, u1)
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    tab = pd.DataFrame({"coef": b, "se": se}, index=names)
    tab["z"] = tab["coef"] / tab["se"]
    tab["p"] = 2 * stats.norm.sf(np.abs(tab["z"]))
    J = float(Zu2 @ W2 @ Zu2)
    df_J = L - k
    p_J = float(stats.chi2.sf(J, df_J)) if df_J > 0 else np.nan
    # Arellano-Bond AR tests on the differenced residuals
    ar = {}
    Zu = sum(Z.T @ uu for Z, uu in zip(Zs, u))
    for order in (1, 2):
        num, s_ww, wX = 0.0, 0.0, np.zeros(k)
        cross = np.zeros(L)
        for (g, yy, XX, rows_Z, n_diff, eq), Z, uu in zip(per_unit, Zs, u):
            e = uu[:n_diff]
            ts = [t for _, t in eq[:n_diff]]
            pos = {t: i for i, t in enumerate(ts)}
            w = np.zeros(len(uu))
            for i, t in enumerate(ts):
                if t - order in pos:
                    w[i] = e[pos[t - order]]
            wi_e = float(w @ uu)
            num += wi_e
            s_ww += wi_e ** 2
            wX += w @ XX
            cross += Z.T @ uu * wi_e
        infl = A @ ZX.T @ W
        var = s_ww - 2 * wX @ infl @ cross + wX @ V @ wX
        zstat = num / np.sqrt(var) if var > 0 else np.nan
        ar[order] = {"z": float(zstat), "p": float(2 * stats.norm.sf(abs(zstat)))}
    # bounds: pooled OLS and within for the first lag of y
    rows = []
    for g, Y, Xm in units:
        for t in range(lags, len(Y)):
            reg = [Y[t - L_] for L_ in range(1, lags + 1)] + list(Xm[t])
            if np.isfinite(Y[t]) and np.all(np.isfinite(reg)):
                rows.append([g, Y[t]] + reg)
    P = pd.DataFrame(rows, columns=["g", "y"] + names[:lags + len(x)])
    import statsmodels.api as sm
    ols = sm.OLS(P["y"], sm.add_constant(P[names[:lags + len(x)]])).fit().params[names[0]]
    dm = P.groupby("g").transform(lambda s: s - s.mean())
    fe = sm.OLS(dm["y"], dm[names[:lags + len(x)]]).fit().params[names[0]]
    n_obs = int(sum(len(yy) for yy in ys))
    warn = []
    if ar[2]["p"] < alpha:
        warn.append(f"AR(2) rejects (p = {ar[2]['p']:.4f}): y_t−2 is correlated with Δε_t; instruments from lag 2 "
                    "are invalid; use gmm_lags=(3, ...)")
    if ar[1]["p"] >= alpha:
        warn.append("AR(1) does not reject: unusual for differenced residuals (expected negative first-order "
                    "correlation); check the model")
    if np.isfinite(p_J) and p_J < alpha:
        warn.append(f"Hansen J rejects (p = {p_J:.4f}): the instruments are not jointly valid")
    if L > N:
        warn.append(f"{L} instruments for {N} groups: instrument proliferation overfits the endogenous variables and "
                    "weakens the Hansen test; use collapse=True or limit the lags")
    if np.isfinite(p_J) and p_J > 0.95 and L > N / 2:
        warn.append(f"Hansen p = {p_J:.3f} with many instruments: a p-value near 1 is a symptom of a weakened test, "
                    "not of valid instruments")
    lo_b, hi_b = min(fe, ols), max(fe, ols)
    if not (lo_b - 2 * se[0] <= b[0] <= hi_b + 2 * se[0]):
        warn.append(f"the GMM estimate of {names[0]} ({b[0]:.3f}) is outside the within ({fe:.3f}) - OLS ({ols:.3f}) "
                    "bounds: suspect weak or invalid instruments (Bond 2002)")
    out = {"params": tab, "one_step": pd.DataFrame({"coef": b1, "se_robust": np.sqrt(np.diag(V1))}, index=names),
           "ar1": ar[1], "ar2": ar[2], "hansen_J": J, "hansen_df": df_J, "hansen_p": p_J, "instruments": L,
           "groups": N, "nobs": n_obs, "bounds": {"within": float(fe), "pooled_ols": float(ols)},
           "warnings": warn, "two_step": two_step,
           "method": (f"{'Blundell-Bond system' if system else 'Arellano-Bond difference'} GMM, "
                      + ("two-step with Windmeijer SEs" if two_step else "one-step robust SEs")
                      + (", collapsed instruments" if collapse else ""))}
    if show:
        print(f"\n{out['method']}: {N} groups, {n_obs} rows, {L} instruments")
        print(tab.to_string(float_format=_fmt))
        print(f"AR(1): z = {ar[1]['z']:.3f}, p = {ar[1]['p']:.4f}; AR(2): z = {ar[2]['z']:.3f}, p = {ar[2]['p']:.4f}")
        print(f"Hansen J = {J:.3f}, df = {df_J}, p = {p_J:.4f}")
        print(f"Bounds for {names[0]}: within {fe:.4f} < GMM {b[0]:.4f} < pooled OLS {ols:.4f}?")
        for w in warn:
            print("Warning:", w)
    return out
