"""
Causal-inference designs for observational data (program and policy evaluation,
law and economics, business decisions).

Selection on observables
    balance_table(df, treat, covs, weights)       standardized mean differences (|SMD| < 0.1 = balanced)
    propensity_score(df, treat, covs)             logit propensity score + overlap diagnostics
    ipw(df, y, treat, ps, estimand)               normalized (Hajek) inverse-probability weighting, ATE/ATT
    aipw(df, y, treat, covs, ...)                 doubly robust ATE with influence-function SE (optional cross-fitting)
    nn_match(df, y, treat, ps, ...)               nearest-neighbour propensity matching (ATT) with caliper
    dml_plr(df, y, d, X, ...)                     double/debiased ML, partially linear model (Chernozhukov et al., 2018)
    dml_sensitivity(res, cf_y, cf_d, rho)         omitted-confounder bounds and robustness values for DML-PLR
                                                  (Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis 2022)
    dml_benchmark(df, y, d, X, benchmarks)        implied cf_y, cf_d, rho of observed covariates (calibration)

Regression discontinuity
    select_bandwidth(y, x, cutoff)                rdrobust MSE bandwidth if installed, else Imbens-Kalyanaraman (2012)
    rdd(df, y, x, cutoff, ...)                    local-linear sharp or fuzzy RD, triangular kernel, HC1/cluster SE
    rdd_robustness(df, y, x, cutoff, ...)         bandwidth sensitivity, placebo cutoffs, covariate jumps
    density_test(x, cutoff)                       McCrary (2008)-type manipulation test

Panel designs
    event_study(df, unit, time, y, event_time)    TWFE leads/lags with clustered SE and joint pre-trend test
    att_gt(df, unit, time, y, gvar)               Callaway-Sant'Anna (2021) group-time ATTs, robust to staggered timing;
                                                  anticipation=δ; warns when the effect starts before the recorded date
    did_timing_sensitivity(df, ..., gvar)         ATT with dates shifted ±k + date-robust estimate (true date within ±k)
    did_uncertain_dates(df, ..., g_low, g_high)   adoption known only within a window per unit (phased rollouts)
    synth(df, unit, time, y, treated, t0)         synthetic control (Abadie et al., 2010) + placebo-in-space inference
    randomization_inference(df, unit, time, y, gvar)  Fisher randomization p-value and CI for a DiD with few
                                                  treated units (permutes which units adopt), any statistic

Notes: for production RD use `rdrobust` (robust bias-corrected CIs) if installed;
for staggered DiD the `csdid`/`differences` packages offer more options. The
functions here are transparent reference implementations, tested on simulations.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats

ALPHA = 0.05


def _ci(est, se, alpha=ALPHA):
    z = stats.norm.ppf(1 - alpha / 2)
    return est - z * se, est + z * se


def _out(name, est, se, n, **extra):
    lo, hi = _ci(est, se)
    p = 2 * (1 - stats.norm.cdf(abs(est / se))) if se > 0 else np.nan
    return {"estimand": name, "estimate": est, "se": se, "ci_low": lo, "ci_high": hi, "p": p, "n": n, **extra}


# =============================================================================
# Selection on observables
# =============================================================================
def balance_table(df: pd.DataFrame, treat: str, covs: list[str], weights=None) -> pd.DataFrame:
    """Standardized mean differences (pooled-SD denominator from the UNWEIGHTED sample,
    so before/after comparisons use the same scale) and variance ratios."""
    t = df[treat].astype(bool).to_numpy()
    w = np.ones(len(df)) if weights is None else np.asarray(weights, float)
    rows = []
    for c in covs:
        x = df[c].astype(float).to_numpy()
        sd = np.sqrt((x[t].var(ddof=1) + x[~t].var(ddof=1)) / 2)
        m1 = np.average(x[t], weights=w[t]); m0 = np.average(x[~t], weights=w[~t])
        v1 = np.average((x[t] - m1) ** 2, weights=w[t]); v0 = np.average((x[~t] - m0) ** 2, weights=w[~t])
        rows.append({"covariate": c, "mean_treated": m1, "mean_control": m0,
                     "SMD": (m1 - m0) / sd if sd > 0 else np.nan, "var_ratio": v1 / v0 if v0 > 0 else np.nan})
    tab = pd.DataFrame(rows).set_index("covariate")
    tab["balanced(|SMD|<0.1)"] = tab["SMD"].abs() < 0.1
    return tab


def propensity_score(df: pd.DataFrame, treat: str, covs: list[str] | None = None, formula: str | None = None,
                     show: bool = True) -> pd.Series:
    """Logit propensity score. Pass `formula` (right-hand side) for interactions/polynomials."""
    import statsmodels.formula.api as smf
    rhs = formula or " + ".join(covs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = smf.logit(f"{treat} ~ {rhs}", df).fit(disp=False)
    ps = m.predict(df).rename("pscore")
    if show:
        t = df[treat].astype(bool)
        q = lambda s: f"[{s.min():.3f}, {s.quantile(.05):.3f} … {s.quantile(.95):.3f}, {s.max():.3f}]"
        print(f"\nPropensity score (logit, pseudo-R² {m.prsquared:.3f})")
        print(f"  treated  range/5-95%: {q(ps[t])}")
        print(f"  controls range/5-95%: {q(ps[~t])}")
        extreme = ((ps < 0.1) | (ps > 0.9)).mean()
        print(f"  share with ps outside [0.1, 0.9]: {extreme:.1%} "
              + ("→ weak overlap: trim (Crump et al., 2009) or estimate the ATT only" if extreme > 0.05 else "→ overlap looks fine"))
    return ps


def ipw(df: pd.DataFrame, y: str, treat: str, ps, estimand: str = "ATE", trim: tuple | None = None,
        cluster: str | None = None) -> dict:
    """Normalized IPW. SE from weighted least squares of y on treat (HC1 or clustered),
    which ignores estimation of the score (conservative for the ATE)."""
    import statsmodels.api as sm
    d = df.copy()
    d["_ps"] = np.asarray(ps, float)
    if trim:
        d = d[(d["_ps"] >= trim[0]) & (d["_ps"] <= trim[1])]
    t = d[treat].astype(float)
    e = d["_ps"]
    w = t / e + (1 - t) / (1 - e) if estimand == "ATE" else t + (1 - t) * e / (1 - e)
    X = sm.add_constant(t.rename("treat"))
    kw = {"cov_type": "cluster", "cov_kwds": {"groups": d[cluster]}} if cluster else {"cov_type": "HC1"}
    r = sm.WLS(d[y].astype(float), X, weights=w).fit(**kw)
    ess = w.groupby(t).apply(lambda s: s.sum() ** 2 / (s ** 2).sum())
    return _out(f"{estimand} (IPW)", r.params["treat"], r.bse["treat"], len(d),
                eff_sample_size_treated=ess.get(1.0), eff_sample_size_control=ess.get(0.0),
                max_weight_share=float(w.max() / w.sum()))


def aipw(df: pd.DataFrame, y: str, treat: str, covs: list[str], ps_learner=None, outcome_learner=None,
         n_folds: int = 0, clip: float = 0.01, seed: int = 0) -> dict:
    """Augmented IPW (doubly robust): consistent if EITHER the propensity or the outcome
    model is right. Default learners: logit and OLS. Pass scikit-learn learners and
    n_folds>=2 for cross-fitting (needed when learners are flexible ML models)."""
    from sklearn.linear_model import LinearRegression, LogisticRegression
    X = df[covs].astype(float).to_numpy()
    T = df[treat].astype(int).to_numpy()
    Y = df[y].astype(float).to_numpy()
    n = len(Y)
    psl = ps_learner or LogisticRegression(max_iter=2000, C=1e6)
    ol = outcome_learner or LinearRegression()
    e, m1, m0 = np.zeros(n), np.zeros(n), np.zeros(n)
    folds = [(np.arange(n), np.arange(n))]
    if n_folds and n_folds >= 2:
        from sklearn.model_selection import KFold
        folds = list(KFold(n_folds, shuffle=True, random_state=seed).split(X))
    from sklearn.base import clone
    for tr, te in folds:
        e[te] = clone(psl).fit(X[tr], T[tr]).predict_proba(X[te])[:, 1]
        m1[te] = clone(ol).fit(X[tr][T[tr] == 1], Y[tr][T[tr] == 1]).predict(X[te])
        m0[te] = clone(ol).fit(X[tr][T[tr] == 0], Y[tr][T[tr] == 0]).predict(X[te])
    e = np.clip(e, clip, 1 - clip)
    psi = m1 - m0 + T * (Y - m1) / e - (1 - T) * (Y - m0) / (1 - e)
    ate = psi.mean()
    se = psi.std(ddof=1) / np.sqrt(n)
    return _out("ATE (AIPW)", ate, se, n, cross_fitting=bool(n_folds and n_folds >= 2),
                share_ps_clipped=float(np.mean((e <= clip) | (e >= 1 - clip))))


def nn_match(df: pd.DataFrame, y: str, treat: str, ps, k: int = 1, caliper: float | None = 0.2,
             replace: bool = True) -> dict:
    """ATT by k-nearest-neighbour matching on the logit of the propensity score.
    caliper in SDs of logit(ps) (Austin, 2011: 0.2). SE is conditional on the matches
    (Abadie-Imbens-type corrections are not implemented): use it as a check next to AIPW."""
    d = df[[y, treat]].copy()
    lp = np.log(np.asarray(ps) / (1 - np.asarray(ps)))
    d["_lp"] = lp
    tr = d[d[treat] == 1]
    co = d[d[treat] == 0].copy()
    cal = caliper * np.std(lp) if caliper else np.inf
    diffs, used, dropped = [], {}, 0
    co_lp = co["_lp"].to_numpy()
    co_y = co[y].to_numpy()
    avail = np.ones(len(co), bool)
    for _, r in tr.iterrows():
        dist = np.abs(co_lp - r["_lp"])
        if not replace:
            dist = np.where(avail, dist, np.inf)
        idx = np.argsort(dist)[:k]
        idx = idx[dist[idx] <= cal]
        if len(idx) == 0:
            dropped += 1
            continue
        if not replace:
            avail[idx] = False
        for j in idx:
            used[j] = used.get(j, 0) + 1 / len(idx)
        diffs.append(r[y] - co_y[idx].mean())
    diffs = np.array(diffs)
    att = diffs.mean()
    se = diffs.std(ddof=1) / np.sqrt(len(diffs))
    return _out("ATT (NN matching)", att, se, len(diffs) + len(used), treated_matched=len(diffs),
                treated_dropped_by_caliper=dropped, controls_used=len(used),
                max_control_reuse=max(used.values()) if used else 0)


def dml_plr(df: pd.DataFrame, y: str, d: str, X: list[str], learner_y=None, learner_d=None,
            n_folds: int = 5, n_rep: int = 3, seed: int = 0) -> dict:
    """Double/debiased ML for y = theta*d + g(X) + u, d = m(X) + v.
    Residualize y and d on X with cross-fitted ML, then regress residual on residual.
    Median over n_rep sample splits (Chernozhukov et al., 2018)."""
    from sklearn.base import clone
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold
    Xm = df[X].astype(float).to_numpy()
    Y = df[y].astype(float).to_numpy()
    D = df[d].astype(float).to_numpy()
    ly = learner_y or RandomForestRegressor(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=seed)
    ld = learner_d or RandomForestRegressor(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=seed)
    thetas, ses, resid = [], [], []
    for rep in range(n_rep):
        uy, vd = np.zeros_like(Y), np.zeros_like(D)
        for tr, te in KFold(n_folds, shuffle=True, random_state=seed + rep).split(Xm):
            uy[te] = Y[te] - clone(ly).fit(Xm[tr], Y[tr]).predict(Xm[te])
            vd[te] = D[te] - clone(ld).fit(Xm[tr], D[tr]).predict(Xm[te])
        th = np.sum(vd * uy) / np.sum(vd * vd)
        psi = (uy - th * vd) * vd
        se = np.sqrt(np.mean(psi ** 2) / np.mean(vd ** 2) ** 2 / len(Y))
        thetas.append(th); ses.append(se); resid.append((uy, vd))
    th = float(np.median(thetas))
    se = float(np.median(np.sqrt(np.array(ses) ** 2 + (np.array(thetas) - th) ** 2)))
    return _out("theta (DML-PLR)", th, se, len(Y), n_folds=n_folds, n_rep=n_rep,
                residuals=resid, var_y=float(np.var(Y)), thetas=thetas)


def _dml_sens_rep(uy, vd, cf_y, cf_d, rho):
    """Point estimates and influence functions of theta, S and the bounds for one sample split."""
    n = len(uy)
    Evv = np.mean(vd * vd)
    th = np.sum(vd * uy) / np.sum(vd * vd)
    eps = uy - th * vd
    sig2 = np.mean(eps ** 2)
    nu2 = 1 / Evv                                   # E[alpha^2], Riesz representer alpha = v / E[v^2]
    S = np.sqrt(sig2 * nu2)
    psi_th = eps * vd / Evv
    psi_sig2 = eps ** 2 - sig2
    psi_nu2 = -(vd * vd - Evv) / Evv ** 2
    c = abs(rho) * np.sqrt(cf_y * cf_d / (1 - cf_d)) if cf_d < 1 else np.inf
    bound = c * S
    psi_S = (nu2 * psi_sig2 + sig2 * psi_nu2) / (2 * S)
    psi_lo, psi_hi = psi_th - c * psi_S, psi_th + c * psi_S
    return {"theta": th, "sigma2": sig2, "nu2": nu2, "S": S, "bound": bound,
            "theta_lo": th - bound, "theta_hi": th + bound,
            "se_theta": np.sqrt(np.mean(psi_th ** 2) / n),
            "se_lo": np.sqrt(np.mean(psi_lo ** 2) / n), "se_hi": np.sqrt(np.mean(psi_hi ** 2) / n)}


def dml_sensitivity(res: dict, cf_y: float = 0.03, cf_d: float = 0.03, rho: float = 1.0,
                    alpha: float = ALPHA, q: float = 1.0, show: bool = True) -> dict:
    """Omitted-variable sensitivity for DML-PLR (Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis 2022,
    "Long Story Short"), the machine-learning analogue of Cinelli-Hazlett (2020).

    A latent confounder A, if added, would move theta by at most
        |theta_short - theta_long| <= |rho| * S * sqrt( cf_y * cf_d / (1 - cf_d) ),
    S^2 = E[(Y - theta D - l(X))^2] * E[alpha^2],  alpha = (D - m(X)) / E[(D - m(X))^2], where
      cf_y = partial R^2 of A with the outcome residual, given D and X (share of the residual
             variance of Y that A would explain),
      cf_d = partial R^2 of A with the treatment residual, given X (1 - R^2 of alpha_s on alpha),
      rho  = correlation of the two omitted pieces (1 = worst case, conservative).
    Bounds [theta - B, theta + B] and confidence bounds that add sampling error (influence
    functions of theta and S, median over the sample splits of dml_plr). Robustness values:
    rv = strength (cf_y = cf_d) that drives theta to (1 - q) theta; rva = strength at which the
    confidence bound reaches that value. Calibrate cf_y and cf_d with dml_benchmark: "a
    confounder as strong as income" is a statement the reader can judge."""
    if "residuals" not in res:
        raise ValueError("pass the dict returned by causal.dml_plr")
    z = stats.norm.ppf(1 - alpha / 2)
    reps = [_dml_sens_rep(uy, vd, cf_y, cf_d, rho) for uy, vd in res["residuals"]]
    med = {k: float(np.median([r_[k] for r_ in reps])) for k in reps[0]}
    th, S, B = res["estimate"], med["S"], med["bound"]
    lo, hi = th - B, th + B
    ci_lo, ci_hi = lo - z * med["se_lo"], hi + z * med["se_hi"]

    def rv_for(target_gap, with_se):
        def gap(eta):
            e2 = eta * eta
            c = abs(rho) * e2 / np.sqrt(1 - e2)
            if with_se:
                rr = [_dml_sens_rep(uy, vd, e2, e2, rho) for uy, vd in res["residuals"]]
                sel = np.median([x["se_lo"] if th > 0 else x["se_hi"] for x in rr])
                return c * S + z * sel - target_gap
            return c * S - target_gap
        if gap(1e-9) >= 0:
            return 0.0
        if gap(1 - 1e-9) < 0:
            return 1.0
        return float(optimize.brentq(gap, 1e-9, 1 - 1e-9) ** 2)
    rv = rv_for(q * abs(th), False)
    rva = rv_for(q * abs(th), True)
    grid = np.array([0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30])
    tab = pd.DataFrame(index=pd.Index(grid, name="cf_y \\ cf_d"), columns=grid, dtype=float)
    for cy in grid:
        for cd in grid:
            tab.loc[cy, cd] = th - np.sign(th) * abs(rho) * S * np.sqrt(cy * cd / (1 - cd))
    out = {"estimand": "theta (DML-PLR) under omitted confounding", "estimate": th, "se": res["se"],
           "ci_low": ci_lo, "ci_high": ci_hi, "p": res["p"], "n": res["n"],
           "method": "Chernozhukov et al. (2022) OVB bounds for DML-PLR",
           "cf_y": cf_y, "cf_d": cf_d, "rho": rho, "S": S, "bias_bound": B,
           "theta_lower": lo, "theta_upper": hi, "rv": rv, "rva": rva, "q": q,
           "bound_table": tab, "warnings": []}
    if show:
        print(f"\nDML-PLR sensitivity: theta = {th:.4f} (SE {res['se']:.4f}), S = {S:.4f}")
        print(f"  confounder with cf_y = {cf_y:.1%}, cf_d = {cf_d:.1%}, rho = {rho}: bias <= {B:.4f}")
        print(f"  bounds [{lo:.4f}, {hi:.4f}]; {100 * (1 - alpha):.0f}% confidence bounds [{ci_lo:.4f}, {ci_hi:.4f}]")
        print(f"  RV = {rv:.2%}: a confounder explaining {rv:.1%} of the residual variance of both D and Y "
              f"would move theta to {(1 - q) * th:.4f}; RVa = {rva:.2%} makes it no longer significant.")
        print("  Benchmark these strengths against observed covariates with causal.dml_benchmark.")
    return out


def dml_benchmark(df: pd.DataFrame, y: str, d: str, X: list[str], benchmarks: list[str],
                  res: dict | None = None, **dml_kwargs) -> pd.DataFrame:
    """Implied sensitivity parameters of observed covariates (Chernozhukov et al. 2022, sec. 6):
    refit dml_plr without each benchmark covariate X_j and measure what it contributes.
      cf_y   = (sigma2_short - sigma2_long) / sigma2_short   (gain in the outcome's explained residual variance)
      cf_d   = (nu2_long - nu2_short) / nu2_long              (= R^2 of X_j with D residual, given the rest)
      rho    = (theta_short - theta_long) / (S_short sqrt(cf_y cf_d / (1 - cf_d)))
      delta_theta = theta_short - theta_long
    Pass k * cf_y and k * cf_d to dml_sensitivity for "a confounder k times as strong as X_j".
    Benchmarks are informative only if the latent confounder is comparable to the observed
    ones; they also condition on the remaining covariates, so they are not exact."""
    res = res or dml_plr(df, y, d, X, **dml_kwargs)
    long = [_dml_sens_rep(uy, vd, 0, 0, 1) for uy, vd in res["residuals"]]
    s2l = np.median([r_["sigma2"] for r_ in long]); n2l = np.median([r_["nu2"] for r_ in long])
    rows = []
    for b in benchmarks:
        rs = dml_plr(df, y, d, [x for x in X if x != b], **dml_kwargs)
        sh = [_dml_sens_rep(uy, vd, 0, 0, 1) for uy, vd in rs["residuals"]]
        s2s = np.median([r_["sigma2"] for r_ in sh]); n2s = np.median([r_["nu2"] for r_ in sh])
        Ss = np.median([r_["S"] for r_ in sh])
        cy = max((s2s - s2l) / s2s, 0.0)
        cd = float(np.clip((n2l - n2s) / n2l, 0.0, 0.999))
        dth = rs["estimate"] - res["estimate"]
        denom = Ss * np.sqrt(cy * cd / (1 - cd)) if cy > 0 and cd > 0 else np.nan
        rows.append({"benchmark": b, "cf_y": cy, "cf_d": cd,
                     "rho": float(np.clip(dth / denom, -1, 1)) if np.isfinite(denom) and denom > 0 else np.nan,
                     "theta_without": rs["estimate"], "delta_theta": dth})
    return pd.DataFrame(rows).set_index("benchmark")


# =============================================================================
# Regression discontinuity
# =============================================================================
def _kern(u, kernel="triangular"):
    u = np.abs(u)
    if kernel == "triangular":
        return np.clip(1 - u, 0, None)
    if kernel == "epanechnikov":
        return np.clip(0.75 * (1 - u ** 2), 0, None)
    return (u <= 1).astype(float)                          # uniform


def ik_bandwidth(y, x, cutoff: float = 0.0) -> float:
    """Imbens & Kalyanaraman (2012) bandwidth for sharp local-linear RD, triangular kernel."""
    y = np.asarray(y, float); x = np.asarray(x, float) - cutoff
    ok = np.isfinite(y) & np.isfinite(x); y, x = y[ok], x[ok]
    N = len(x)
    h1 = 1.84 * np.std(x, ddof=1) * N ** (-1 / 5)
    lm, rm = (x < 0) & (x >= -h1), (x >= 0) & (x <= h1)
    nl, nr = lm.sum(), rm.sum()
    s2 = (np.sum((y[lm] - y[lm].mean()) ** 2) + np.sum((y[rm] - y[rm].mean()) ** 2)) / (nl + nr)
    f = (nl + nr) / (2 * N * h1)
    medl, medr = np.median(x[x < 0]), np.median(x[x >= 0])
    w = (x >= medl) & (x <= medr)
    Z = np.column_stack([np.ones(w.sum()), (x[w] >= 0), x[w], x[w] ** 2, x[w] ** 3])
    m3 = 6 * np.linalg.lstsq(Z, y[w], rcond=None)[0][4]
    nL, nR = (x < 0).sum(), (x >= 0).sum()
    h2l = 3.56 * (s2 / (f * max(m3 ** 2, 1e-2))) ** (1 / 7) * nL ** (-1 / 7)
    h2r = 3.56 * (s2 / (f * max(m3 ** 2, 1e-2))) ** (1 / 7) * nR ** (-1 / 7)

    def m2(side, h):
        s = (x < 0) & (x >= -h) if side == "l" else (x >= 0) & (x <= h)
        Z = np.column_stack([np.ones(s.sum()), x[s], x[s] ** 2])
        return 2 * np.linalg.lstsq(Z, y[s], rcond=None)[0][2], s.sum()
    m2l, n2l = m2("l", h2l)
    m2r, n2r = m2("r", h2r)
    rl = 2160 * s2 / (n2l * h2l ** 4)
    rr = 2160 * s2 / (n2r * h2r ** 4)
    return float(3.4375 * ((2 * s2 / f) / ((m2r - m2l) ** 2 + rl + rr)) ** (1 / 5) * N ** (-1 / 5))


def select_bandwidth(y, x, cutoff: float = 0.0) -> tuple[float, str]:
    """MSE-optimal bandwidth: rdrobust's 'mserd' (Calonico-Cattaneo-Titiunik) if the package
    is installed, otherwise Imbens-Kalyanaraman. IK tends to be wider when the two sides
    have similar curvature (its bias term uses the difference of second derivatives)."""
    try:
        from rdrobust import rdbwselect
        yy, xx = np.asarray(y, float), np.asarray(x, float)
        ok = np.isfinite(yy) & np.isfinite(xx)
        bw = rdbwselect(yy[ok], xx[ok], c=cutoff, bwselect="mserd")
        return float(np.asarray(bw.bws)[0, 0]), "rdrobust mserd"
    except Exception:
        return ik_bandwidth(y, x, cutoff), "Imbens-Kalyanaraman"


def rdd(df: pd.DataFrame, y: str, x: str, cutoff: float = 0.0, h: float | None = None,
        kernel: str = "triangular", p: int = 1, covs: list[str] | None = None, cluster: str | None = None,
        fuzzy: str | None = None, show: bool = True) -> dict:
    """Local-polynomial RD. Sharp by default; `fuzzy` = name of the actual treatment
    (then the estimate is the local Wald/2SLS ratio). Bandwidth: IK if None."""
    import statsmodels.api as sm
    d = df.dropna(subset=[y, x] + (covs or []) + ([fuzzy] if fuzzy else [])).copy()
    d["_xc"] = d[x] - cutoff
    bw_method = "user"
    if h is None:
        h, bw_method = select_bandwidth(d[y], d[x], cutoff)
    d = d[d["_xc"].abs() <= h].copy()
    d["_w"] = _kern(d["_xc"] / h, kernel)
    d = d[d["_w"] > 0]
    d["_above"] = (d["_xc"] >= 0).astype(float)
    cols = {"_above": d["_above"]}
    for k in range(1, p + 1):
        cols[f"_x{k}"] = d["_xc"] ** k
        cols[f"_ax{k}"] = d["_above"] * d["_xc"] ** k
    X = pd.DataFrame(cols, index=d.index)
    if covs:
        X = X.join(d[covs])
    X = sm.add_constant(X)
    kw = {"cov_type": "cluster", "cov_kwds": {"groups": d[cluster]}} if cluster else {"cov_type": "HC1"}
    if not fuzzy:
        r = sm.WLS(d[y], X, weights=d["_w"]).fit(**kw)
        est, se = r.params["_above"], r.bse["_above"]
        first = None
    else:
        from linearmodels.iv import IV2SLS
        exog = X.drop(columns="_above")
        iv = IV2SLS(d[y], exog, d[[fuzzy]], d[["_above"]], weights=d["_w"]).fit(
            cov_type="clustered" if cluster else "robust", **({"clusters": d[cluster]} if cluster else {}))
        est, se = iv.params[fuzzy], iv.std_errors[fuzzy]
        fs = sm.WLS(d[fuzzy], X, weights=d["_w"]).fit(**kw)
        first = {"jump_in_treatment": fs.params["_above"], "F": (fs.params["_above"] / fs.bse["_above"]) ** 2}
    out = _out("RD effect" + (" (fuzzy, LATE)" if fuzzy else " (sharp)"), est, se, len(d), bandwidth=h,
               bandwidth_method=bw_method,
               n_left=int((d["_above"] == 0).sum()), n_right=int((d["_above"] == 1).sum()), kernel=kernel, p=p)
    if first:
        out.update(first)
    if show:
        print(f"\nRD at {x} = {cutoff}: {out['estimate']:.4f} (SE {se:.4f}, 95% CI {out['ci_low']:.4f} to "
              f"{out['ci_high']:.4f}), h = {h:.4g} ({bw_method}), N = {out['n_left']} + {out['n_right']}"
              + (f", first-stage jump {first['jump_in_treatment']:.3f} (F = {first['F']:.1f})" if first else ""))
        print("  Conventional CI; for robust bias-corrected inference use rdrobust (Calonico, Cattaneo & Titiunik, 2014).")
    return out


def density_test(x, cutoff: float = 0.0, bins: int | None = None, h: float | None = None) -> dict:
    """McCrary (2008)-type test: log difference of the density at the cutoff, estimated by
    local-linear smoothing of a histogram on each side. H0: no manipulation (continuity)."""
    x = np.asarray(pd.Series(x).dropna(), float) - cutoff
    n = len(x)
    b = 2 * np.std(x, ddof=1) * n ** (-1 / 2) if bins is None else (x.max() - x.min()) / bins
    l = np.floor((x.min()) / b) * b
    r = np.floor((x.max()) / b) * b + b
    edges = np.arange(l, r + b, b)
    centers = (edges[:-1] + edges[1:]) / 2
    counts = np.histogram(x, edges)[0]
    dens = counts / (n * b)
    if h is None:
        h = 2 * np.std(x, ddof=1) * n ** (-1 / 5)       # simple rule; check sensitivity

    def side(mask):
        c, dd = centers[mask], dens[mask]
        w = _kern(c / h)
        k = w > 0
        Z = np.column_stack([np.ones(k.sum()), c[k]])
        W = np.diag(w[k])
        beta = np.linalg.solve(Z.T @ W @ Z, Z.T @ W @ dd[k])
        return max(beta[0], 1e-12)
    fl, fr = side(centers < 0), side(centers >= 0)
    theta = np.log(fr) - np.log(fl)
    se = np.sqrt((1 / (n * h)) * (24 / 5) * (1 / fr + 1 / fl))
    return {"log_density_jump": theta, "se": se, "z": theta / se,
            "p": 2 * (1 - stats.norm.cdf(abs(theta / se))), "f_left": fl, "f_right": fr, "bin": b, "h": h}


def rdd_robustness(df: pd.DataFrame, y: str, x: str, cutoff: float = 0.0, h: float | None = None,
                   covs_to_test: list[str] | None = None, cluster: str | None = None) -> dict:
    """Standard RD robustness package: effect at 0.5h…2h, polynomial order 2, placebo
    cutoffs at the median of each side, 'effects' on predetermined covariates, density test."""
    h = h or select_bandwidth(df[y], df[x], cutoff)[0]
    sens = pd.DataFrame([rdd(df, y, x, cutoff, h=h * m, cluster=cluster, show=False) | {"h_multiple": m}
                         for m in (0.5, 0.75, 1, 1.5, 2)]).set_index("h_multiple")[["estimate", "se", "p", "n"]]
    quad = rdd(df, y, x, cutoff, h=h, p=2, cluster=cluster, show=False)
    left, right = df[df[x] < cutoff], df[df[x] >= cutoff]
    plc = []
    for name, sub in (("placebo_left", left), ("placebo_right", right)):
        c = sub[x].median()
        r = rdd(sub, y, x, c, cluster=cluster, show=False)
        plc.append({"cutoff": c, **{k: r[k] for k in ("estimate", "se", "p")}, "which": name})
    cov = []
    for c in covs_to_test or []:
        r = rdd(df, c, x, cutoff, h=h, cluster=cluster, show=False)
        cov.append({"covariate": c, "jump": r["estimate"], "se": r["se"], "p": r["p"]})
    dens = density_test(df[x], cutoff)
    print("\nRD robustness")
    print("Bandwidth sensitivity:\n" + sens.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"Local quadratic: {quad['estimate']:.4f} (SE {quad['se']:.4f})")
    print("Placebo cutoffs (should be ~0):\n" + pd.DataFrame(plc).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if cov:
        print("Jumps in predetermined covariates (should be ~0):\n" + pd.DataFrame(cov).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"Density test: log jump {dens['log_density_jump']:.3f}, p = {dens['p']:.3f} → "
          + ("possible manipulation of the running variable" if dens["p"] < ALPHA else "no evidence of manipulation"))
    return {"bandwidth_sensitivity": sens, "quadratic": quad, "placebos": pd.DataFrame(plc),
            "covariates": pd.DataFrame(cov), "density": dens}


# =============================================================================
# Panel designs
# =============================================================================
def event_study(df: pd.DataFrame, unit: str, time: str, y: str, event_time: str, leads: int = 4,
                lags: int = 4, ref: int = -1, bin_endpoints: bool = True, controls: list[str] | None = None,
                show: bool = True) -> dict:
    """TWFE event study. event_time = first treated period for each unit (NaN = never treated).
    Relative-time dummies k = t - event_time for k in [-leads, lags], reference k = ref,
    endpoints binned. Unit and time fixed effects, errors clustered by unit.
    With staggered adoption and heterogeneous effects, compare with att_gt."""
    from linearmodels.panel import PanelOLS
    d = df.copy()
    rel = d[time] - d[event_time]
    if bin_endpoints:
        rel = rel.clip(-leads, lags)
    names = []
    for k in range(-leads, lags + 1):
        if k == ref:
            continue
        nm = f"rel_{'m' if k < 0 else 'p'}{abs(k)}"
        d[nm] = ((rel == k) & d[event_time].notna()).astype(float)
        names.append(nm)
    d = d.set_index([unit, time])
    exog = d[names + (controls or [])]
    res = PanelOLS(d[y], exog, entity_effects=True, time_effects=True, drop_absorbed=True).fit(
        cov_type="clustered", cluster_entity=True)
    rows = []
    for k in range(-leads, lags + 1):
        if k == ref:
            rows.append({"k": k, "coef": 0.0, "se": 0.0}); continue
        nm = f"rel_{'m' if k < 0 else 'p'}{abs(k)}"
        rows.append({"k": k, "coef": res.params.get(nm, np.nan), "se": res.std_errors.get(nm, np.nan)})
    tab = pd.DataFrame(rows).set_index("k")
    tab["ci_low"], tab["ci_high"] = _ci(tab["coef"], tab["se"])
    pre = [f"rel_m{abs(k)}" for k in range(-leads, 0) if k != ref and f"rel_m{abs(k)}" in res.params.index]
    wt = res.wald_test(formula=" = ".join(pre) + " = 0") if len(pre) > 1 else (
        res.wald_test(formula=f"{pre[0]} = 0") if pre else None)
    pval = float(wt.pval) if wt is not None else np.nan
    # Timing diagnostics. (i) A jump between the reference period and the lead next to it
    # means the outcome moves before the recorded date (anticipation or misdating).
    # (ii) Post coefficients that keep rising suggest a gradual effect or a phased-in rollout.
    # The two leads nearest the reference, each tested at ALPHA/2 (Bonferroni): a date that is
    # one or two periods too late shows up as a jump in one of them.
    near_ks = sorted((k for k in range(-leads, 0) if k != ref), reverse=True)[:2] if ref < 0 else []
    zc = stats.norm.ppf(1 - ALPHA / 4)
    sig = lambda k: bool(np.isfinite(tab.loc[k, "se"]) and tab.loc[k, "se"] > 0
                         and abs(tab.loc[k, "coef"] / tab.loc[k, "se"]) > zc)
    hits = [k for k in near_ks if sig(k)]
    early = bool(hits)
    near = hits[0] if hits else None
    posts = [k for k in range(0, lags + 1) if k != ref and np.isfinite(tab.loc[k, "coef"])]
    ramp = False
    if len(posts) >= 3:
        a, z = tab.loc[posts[0]], tab.loc[posts[-1]]
        ramp = bool(z["coef"] - a["coef"] > 2 * np.sqrt(a["se"] ** 2 + z["se"] ** 2))
    flags = {"effect_before_date": early, "gradual_effect": ramp}
    if show:
        print(f"\nEvent study (TWFE, ref k = {ref}, clustered by {unit})")
        print(tab.to_string(float_format=lambda v: f"{v:.4f}"))
        print(f"Joint test of pre-period coefficients = 0: p = {pval:.3f} → "
              + ("pre-trends detected: parallel trends doubtful" if pval < ALPHA else "no evidence of pre-trends (low power is possible)"))
        if early:
            print(f"  WARNING: the coefficient at k = {near} differs from the reference k = {ref}: the outcome jumps"
                  " right before the recorded date. Typical causes: anticipation (announcement before entry"
                  " into force) or a misdated/phased-in treatment. Confirm the date; use att_gt(anticipation=...)"
                  " and did_timing_sensitivity.")
        if ramp:
            print("  NOTE: post-treatment coefficients keep rising. Either the effect builds with exposure, or"
                  " adoption was gradual within units (a single date understates early exposure). If the"
                  " rollout was phased, record the phase-in window and use did_uncertain_dates.")
    return {"coefs": tab, "pretrend_p": pval, "flags": flags, "result": res}


def att_gt(df: pd.DataFrame, unit: str, time: str, y: str, gvar: str, control: str = "never",
           n_boot: int = 199, seed: int = 0, anticipation: int = 0, show: bool = True) -> dict:
    """Callaway & Sant'Anna (2021) ATT(g,t) without covariates on a balanced panel.
    gvar = first treated period (0 or NaN = never treated). control = 'never' or 'notyet'.
    anticipation = δ periods in which units may react before g (law announced before it
    takes effect): post cells (t >= g-δ) use base g-δ-1, not-yet controls must be untreated
    until t+δ. Pre-period cells use the short (t-1) base.
    Inference: unit-level bootstrap (resampling units with replacement); n_boot=0 skips it.
    Warns when either of the two pre-period cells right before the (anticipation-adjusted)
    date is significant (Bonferroni, ALPHA/2 each): the effect starts before the recorded date (anticipation or a misdated
    treatment) → check the date and run did_timing_sensitivity."""
    d = df[[unit, time, y, gvar]].copy()
    d[gvar] = d[gvar].fillna(0)
    Y = d.pivot(index=unit, columns=time, values=y)
    if Y.isna().any().any():
        raise ValueError("att_gt needs a balanced panel (drop units with missing periods)")
    G = d.groupby(unit)[gvar].first().reindex(Y.index).to_numpy()
    T = np.array(Y.columns)
    Ym = Y.to_numpy()
    dlt = int(anticipation)
    cohorts = sorted(g for g in np.unique(G) if g > 0 and (g - dlt - 1) in T)
    dropped = sorted(g for g in np.unique(G) if g > 0 and g not in cohorts)

    def compute(idx):
        Yb, Gb = Ym[idx], G[idx]
        cells = []
        for g in cohorts:
            tg = Gb == g
            if tg.sum() == 0:
                continue
            for j, t in enumerate(T):
                base = (g - dlt - 1) if t >= g - dlt else T[j - 1] if j > 0 else None
                if base is None or base not in T:
                    continue
                b = np.where(T == base)[0][0]
                if control == "never":
                    cg = Gb == 0
                else:
                    cg = (Gb == 0) | (Gb > max(t, base) + dlt)
                    cg &= Gb != g
                if cg.sum() == 0:
                    continue
                att = (Yb[tg, j] - Yb[tg, b]).mean() - (Yb[cg, j] - Yb[cg, b]).mean()
                cells.append((g, t, t - g, att, tg.sum()))
        return pd.DataFrame(cells, columns=["g", "t", "e", "att", "n_g"])

    if not cohorts:
        raise ValueError("att_gt: no cohort has the base period g-anticipation-1 inside the panel")
    base = compute(np.arange(len(Ym)))
    post = base[base["e"] >= 0]
    simple = np.average(post["att"], weights=post["n_g"]) if len(post) else np.nan
    dyn = base.groupby("e").apply(lambda s: np.average(s["att"], weights=s["n_g"]), include_groups=False)
    if n_boot and n_boot > 1:
        rng = np.random.default_rng(seed)
        bs_simple, bs_dyn, bs_cells = [], [], []
        for _ in range(n_boot):
            b = compute(rng.integers(0, len(Ym), len(Ym)))
            p = b[b["e"] >= 0]
            bs_simple.append(np.average(p["att"], weights=p["n_g"]) if len(p) else np.nan)
            bs_dyn.append(b.groupby("e").apply(lambda s: np.average(s["att"], weights=s["n_g"]), include_groups=False))
            bs_cells.append(b.set_index(["g", "t"])["att"])
        se_simple = np.nanstd(bs_simple, ddof=1)
        dyn_se = pd.concat(bs_dyn, axis=1).std(axis=1, ddof=1)
        cell_se = pd.concat(bs_cells, axis=1).std(axis=1, ddof=1)
    else:
        se_simple, dyn_se = np.nan, pd.Series(np.nan, index=dyn.index)
        cell_se = pd.Series(np.nan, index=base.set_index(["g", "t"]).index)
    dyn_tab = pd.DataFrame({"att": dyn, "se": dyn_se})
    dyn_tab["ci_low"], dyn_tab["ci_high"] = _ci(dyn_tab["att"], dyn_tab["se"])
    gt = base.set_index(["g", "t"]).join(cell_se.rename("se"))
    # Jump right before the (anticipation-adjusted) date: with the short t-1 base, the cell
    # e = -δ-1 measures the change from g-δ-2 to g-δ-1, which is ~0 if the date is right.
    # Check the two cells right before it (a date one or two periods too late), each at ALPHA/2.
    zc = stats.norm.ppf(1 - ALPHA / 4)
    hits = [k for k in (-dlt - 1, -dlt - 2) if k in dyn_tab.index and np.isfinite(dyn_tab.loc[k, "se"])
            and dyn_tab.loc[k, "se"] > 0 and abs(dyn_tab.loc[k, "att"] / dyn_tab.loc[k, "se"]) > zc]
    flag = bool(hits)
    k_chk = hits[0] if hits else -dlt - 1
    out = {"overall": _out("ATT (CS, simple)", simple, se_simple, len(Ym), anticipation=dlt),
           "dynamic": dyn_tab, "gt": gt, "flags": {"effect_before_date": flag},
           "cohorts_dropped": dropped}
    if show:
        o = out["overall"]
        print(f"\nCallaway-Sant'Anna ({control} controls, {len(cohorts)} cohorts, {n_boot} bootstrap reps"
              + (f", anticipation = {dlt}" if dlt else "") + ")")
        print(f"Overall ATT: {o['estimate']:.4f} (SE {o['se']:.4f}, 95% CI {o['ci_low']:.4f} to {o['ci_high']:.4f})")
        print("Dynamic (event-time) ATT:\n" + dyn_tab.to_string(float_format=lambda v: f"{v:.4f}"))
        if dropped:
            print(f"  Cohorts dropped (base period g-{dlt}-1 outside the panel): {dropped}")
        if flag:
            print(f"  WARNING: the outcome already moves at e = {k_chk} (|z| > {zc:.2f}). The effect starts before"
                  " the recorded date: anticipation (announcement before entry into force) or a misdated"
                  " treatment. Confirm the date; re-run with anticipation=" + str(-k_chk)
                  + " and did_timing_sensitivity before reporting the ATT.")
    return out


def did_timing_sensitivity(df: pd.DataFrame, unit: str, time: str, y: str, gvar: str, max_shift: int = 2,
                           control: str = "never", n_boot: int = 199, seed: int = 0, show: bool = True) -> dict:
    """How much does the Callaway-Sant'Anna result depend on the exact treatment date?

    1. Shift table: re-estimates the overall ATT with every treated unit's date moved by
       k = -max_shift … +max_shift periods. Large swings mean the date itself drives the result.
    2. Date-robust estimate: assumes each true date lies within ±max_shift of the recorded
       one. Uses the latest plausible date (g+max_shift) as the event date and
       anticipation = 2·max_shift, so the base period (g-max_shift-1) is untreated under
       every date in the window and the post periods are treated under every date in it.
       It estimates the effect from g+max_shift onward (a later-exposure ATT when effects
       are dynamic), at the cost of fewer usable cohorts and wider CIs.
    Verdict: 'survives' if the date-robust CI excludes 0 with the same sign as the
    baseline; otherwise the conclusion depends on the exact date."""
    d = df.copy()
    g0 = d[gvar].fillna(0)
    rows = []
    for k in range(-max_shift, max_shift + 1):
        d["_g"] = np.where(g0 > 0, g0 + k, 0)
        try:
            r = att_gt(d, unit, time, y, "_g", control=control, n_boot=n_boot, seed=seed, show=False)
            o = r["overall"]
            rows.append({"shift": k, "att": o["estimate"], "se": o["se"], "ci_low": o["ci_low"],
                         "ci_high": o["ci_high"], "p": o["p"]})
        except ValueError:
            rows.append({"shift": k, "att": np.nan, "se": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan})
    tab = pd.DataFrame(rows).set_index("shift")
    d["_g"] = np.where(g0 > 0, g0 + max_shift, 0)
    rob = att_gt(d, unit, time, y, "_g", control=control, n_boot=n_boot, seed=seed,
                 anticipation=2 * max_shift, show=False)
    ro = rob["overall"]
    b = tab.loc[0]
    same_sign = np.sign(ro["estimate"]) == np.sign(b["att"])
    survives = bool(same_sign and np.isfinite(ro["se"]) and (ro["ci_low"] > 0 or ro["ci_high"] < 0))
    swing = float((tab["att"] - b["att"]).abs().max() / abs(b["att"])) if b["att"] else np.nan
    out = {"shifts": tab, "date_robust": ro, "date_robust_dynamic": rob["dynamic"], "survives": survives,
           "max_relative_swing": swing, "cohorts_dropped_robust": rob["cohorts_dropped"]}
    if show:
        print(f"\nTiming sensitivity (Callaway-Sant'Anna, dates shifted by -{max_shift}…+{max_shift} periods)")
        print(tab.to_string(float_format=lambda v: f"{v:.4f}"))
        print(f"Largest swing vs recorded dates: {swing:.0%} of the baseline ATT (mechanical: a wrong date puts"
              " treated periods in the base or untreated ones in the post period, so swings are expected;"
              " the date-robust line below is the verdict)")
        print(f"Date-robust ATT (true date anywhere within ±{max_shift}): {ro['estimate']:.4f} "
              f"(SE {ro['se']:.4f}, 95% CI {ro['ci_low']:.4f} to {ro['ci_high']:.4f})"
              + (f"; cohorts dropped: {rob['cohorts_dropped']}" if rob["cohorts_dropped"] else ""))
        print("  → " + (f"the conclusion survives date errors of up to ±{max_shift} periods." if survives else
                        f"the conclusion DEPENDS on the exact date: with an error of up to ±{max_shift} periods the "
                        "effect is not distinguishable from zero (or flips sign). Pin the date down with sources "
                        "before reporting a causal effect."))
    return out


def did_uncertain_dates(df: pd.DataFrame, unit: str, time: str, y: str, g_low: str, g_high: str,
                        control: str = "never", n_draws: int = 200, n_boot: int = 199, seed: int = 0,
                        show: bool = True) -> dict:
    """Staggered DiD when each unit's adoption date is only known to lie in a window
    [g_low, g_high] (e.g. a law phased in court by court with imprecise records).
    Never-treated units: g_low = g_high = 0 or NaN.

    1. Bounds-style estimate: event date = g_high, anticipation = max window width, so
       every base period precedes any plausible adoption (Callaway-Sant'Anna, bootstrap CI).
    2. Date draws: n_draws estimates with each unit's date drawn uniformly from its
       window; the spread shows how much date uncertainty alone moves the ATT."""
    d = df.copy()
    lo = d.groupby(unit)[g_low].first().fillna(0)
    hi = d.groupby(unit)[g_high].first().fillna(0)
    if ((hi > 0) & (lo > hi)).any():
        raise ValueError("g_low must be <= g_high for every treated unit")
    lo = lo.where(lo > 0, hi)
    width = int((hi - lo)[hi > 0].max()) if (hi > 0).any() else 0
    d["_g"] = d[unit].map(hi)
    rob = att_gt(d, unit, time, y, "_g", control=control, n_boot=n_boot, seed=seed,
                 anticipation=width, show=False)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_draws):
        gd = pd.Series(np.where(hi > 0, rng.integers(lo.astype(int), hi.astype(int) + 1), 0), index=hi.index)
        d["_g"] = d[unit].map(gd)
        try:
            draws.append(att_gt(d, unit, time, y, "_g", control=control, n_boot=0, show=False)["overall"]["estimate"])
        except ValueError:
            pass
    draws = np.array(draws)
    ro = rob["overall"]
    q05, q50, q95 = np.nanquantile(draws, [0.05, 0.5, 0.95])
    share_same = float(np.mean(np.sign(draws) == np.sign(q50)))
    survives = bool(np.isfinite(ro["se"]) and (ro["ci_low"] > 0 or ro["ci_high"] < 0)
                    and np.sign(ro["estimate"]) == np.sign(q50))
    out = {"window_robust": ro, "window_robust_dynamic": rob["dynamic"], "max_window": width,
           "draws": draws, "draws_q05": q05, "draws_median": q50, "draws_q95": q95,
           "share_same_sign": share_same, "survives": survives, "cohorts_dropped": rob["cohorts_dropped"]}
    if show:
        print(f"\nDiD with uncertain adoption dates (windows up to {width} periods wide, {control} controls)")
        print(f"Window-robust ATT (base before any plausible date): {ro['estimate']:.4f} "
              f"(SE {ro['se']:.4f}, 95% CI {ro['ci_low']:.4f} to {ro['ci_high']:.4f})")
        print(f"ATT across {len(draws)} random dates inside the windows: median {q50:.4f}, "
              f"5–95% {q05:.4f} to {q95:.4f}, same sign in {share_same:.0%} of draws")
        print("  → " + ("the effect survives the date uncertainty." if survives else
                        "the effect does NOT survive the date uncertainty: narrow the windows with better "
                        "records or report the result as inconclusive."))
    return out


def synth(df: pd.DataFrame, unit: str, time: str, y: str, treated, t0, donors: list | None = None,
          placebo: bool = True, show: bool = True) -> dict:
    """Synthetic control fitted on pre-treatment outcomes: weights w >= 0, sum(w) = 1
    minimizing the pre-period MSPE. t0 = first treated period.
    Inference: placebo-in-space (each donor treated in turn); p-value = rank of the
    treated unit's post/pre RMSPE ratio (Abadie, Diamond & Hainmueller, 2010)."""
    W = df.pivot(index=time, columns=unit, values=y).sort_index()
    donors = [u for u in (donors or W.columns) if u != treated]
    pre, post = W.index < t0, W.index >= t0

    def fit(tr, pool):
        y1, Y0 = W[tr].to_numpy(), W[pool].to_numpy()
        k = len(pool)
        obj = lambda w: np.sum((y1[pre] - Y0[pre] @ w) ** 2)
        r = optimize.minimize(obj, np.full(k, 1 / k), method="SLSQP", bounds=[(0, 1)] * k,
                              constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                              options={"maxiter": 1000, "ftol": 1e-12})
        w = r.x
        synth_path = Y0 @ w
        gap = y1 - synth_path
        rm_pre = np.sqrt(np.mean(gap[pre] ** 2))
        rm_post = np.sqrt(np.mean(gap[post] ** 2))
        return w, synth_path, gap, rm_pre, rm_post

    w, sp, gap, rpre, rpost = fit(treated, donors)
    weights = pd.Series(w, index=donors).sort_values(ascending=False)
    res = {"weights": weights[weights > 1e-3], "synthetic": pd.Series(sp, index=W.index),
           "gap": pd.Series(gap, index=W.index), "rmspe_pre": rpre, "rmspe_post": rpost,
           "avg_post_gap": float(gap[post].mean())}
    if placebo:
        ratios = {treated: rpost / rpre}
        pre_fit = {treated: rpre}
        gaps = {treated: gap}
        for u in donors:
            pool = [v for v in donors if v != u]
            _, _, g, a, b = fit(u, pool)
            ratios[u] = b / a if a > 0 else np.inf
            pre_fit[u] = a
            gaps[u] = g
        rr = pd.Series(ratios).sort_values(ascending=False)
        res["placebo_ratios"] = rr
        res["p_value"] = float((rr >= rr[treated]).mean())
        res["placebo_gaps"] = pd.DataFrame(gaps, index=W.index)
    if show:
        print(f"\nSynthetic control for {treated} (treatment from {t0}); donors with weight > 0.001:")
        print(res["weights"].round(3).to_string())
        print(f"Pre-RMSPE {rpre:.4f} | average post gap {res['avg_post_gap']:.4f}"
              + (f" | placebo p-value {res['p_value']:.3f} (rank {int((rr >= rr[treated]).sum())} of {len(rr)})" if placebo else ""))
        if placebo and len(rr) < 20:
            print("  With fewer than 20 donors the smallest attainable p-value is 1/(donors+1).")
    return res


# =============================================================================
# Randomization inference for DiD with few treated units
# =============================================================================
def randomization_inference(df: pd.DataFrame, unit: str, time: str, y: str, gvar: str, n_perm: int = 999,
                            stat=None, ci: bool = True, ci_points: int = 61, seed: int = 0, alpha: float = ALPHA) -> dict:
    """Fisher randomization inference for a difference-in-differences when only a few units are
    treated (a law adopted by 6 of 27 states): clustered SEs and even the wild cluster bootstrap
    become unreliable with very few treated clusters (MacKinnon & Webb 2017, 2018).

    gvar: first treated period of each unit (0 or NaN = never treated); D_it = 1{t >= g_i}.
    Under the sharp null of no effect for anyone, which units adopted is (as if) random, so the
    statistic is recomputed after reassigning the observed adoption dates to random sets of the
    same number of units (all sets when there are no more than n_perm of them). p = share of
    |stat*| >= |stat| (with the observed assignment counted).
    stat=None uses the static TWFE DiD coefficient (unit and time fixed effects), computed fast by
    residualizing on the fixed effects once; pass a function stat(df_perm) -> float for any other
    statistic (e.g. lambda d: att_gt(d, unit, time, y, gvar, n_boot=0, show=False)["overall"]["estimate"]); the CI is then skipped.
    The CI inverts the test for a constant additive effect tau: y - tau*D is permuted for a grid of tau.

    Validity rests on exchangeability: the treated units must be comparable to the others in
    their error variance and dynamics. If the adopters are systematically more volatile (e.g.
    small states), randomization inference over-rejects too; use Ferman & Pinto (2019) or
    Conley & Taber (2011). The smallest attainable p-value is 1 / (number of possible assignments)."""
    from itertools import combinations
    from math import comb
    d = df[[unit, time, y, gvar]].dropna(subset=[unit, time, y]).copy()
    d[gvar] = d[gvar].fillna(0)
    units = pd.Index(sorted(d[unit].unique(), key=str))
    g_unit = d.groupby(unit)[gvar].first().reindex(units)
    treated = list(g_unit[g_unit > 0].index)
    N, N1 = len(units), len(treated)
    if N1 == 0 or N1 == N:
        raise ValueError("need both treated and never-treated units")
    dates_obs = g_unit[treated].to_numpy()
    u_idx = units.get_indexer(d[unit])
    tt = d[time].to_numpy()
    n_assign = comb(N, N1)
    rng = np.random.default_rng(seed)
    if n_assign - 1 <= n_perm:
        sets = [list(c) for c in combinations(range(N), N1)]
        exact = True
    else:
        sets = [list(rng.choice(N, N1, replace=False)) for _ in range(n_perm)]
        exact = False
    obs_set = [units.get_loc(u) for u in treated]

    def D_for(idx_set, dates):
        g = np.zeros(N)
        g[idx_set] = dates
        gi = g[u_idx]
        return ((gi > 0) & (tt >= gi)).astype(float)

    D_obs = D_for(obs_set, dates_obs)
    perms_D = [D_for(sset, rng.permutation(dates_obs)) for sset in sets]
    Y = d[y].to_numpy(float)
    if stat is None:
        U = pd.get_dummies(d[unit]).to_numpy(float)
        Tm = pd.get_dummies(d[time]).to_numpy(float)[:, 1:]
        F = np.column_stack([U, Tm])
        Fp = np.linalg.pinv(F)
        resid = lambda v: v - F @ (Fp @ v)
        y_t = resid(Y)
        Dt_obs = resid(D_obs)
        Dt = np.column_stack([resid(Dp) for Dp in perms_D])
        est = float(Dt_obs @ y_t / (Dt_obs @ Dt_obs))

        def pval(tau):
            yt0 = y_t - tau * Dt_obs
            s0 = Dt_obs @ yt0 / (Dt_obs @ Dt_obs)
            sp = (Dt * yt0[:, None]).sum(0) / (Dt ** 2).sum(0)
            return (1 + np.sum(np.abs(sp) >= abs(s0) - 1e-12)) / (len(sets) + 1)

        p = float(pval(0.0))
        lo = hi = np.nan
        if ci:
            sd_perm = float(np.std((Dt * y_t[:, None]).sum(0) / (Dt ** 2).sum(0)))
            grid = est + sd_perm * np.linspace(-5, 5, ci_points)
            keep = np.array([pval(tv) > alpha for tv in grid])
            if keep.any():
                lo, hi = float(grid[keep].min()), float(grid[keep].max())
                if keep[0] or keep[-1]:
                    warnings.warn("randomization CI reaches the edge of its grid: widen ci_points")
        method = "randomization inference (TWFE DiD statistic)"
    else:
        est = float(stat(d.copy()))
        draws = []
        for sset in sets:
            g = np.zeros(N)
            g[sset] = rng.permutation(dates_obs)
            dp = d.copy()
            dp[gvar] = g[u_idx]
            draws.append(stat(dp))
        draws = np.asarray(draws, float)
        p = float((1 + np.sum(np.abs(draws) >= abs(est) - 1e-12)) / (len(draws) + 1))
        lo = hi = np.nan
        method = "randomization inference (user statistic)"
    warn = []
    if n_assign < 1 / alpha * 2:
        warn.append(f"only {n_assign} possible assignments: the smallest attainable p-value is {1 / n_assign:.3f}")
    for w in warn:
        warnings.warn(w)
    return {"estimand": "ATT (DiD)", "method": method, "estimate": est, "p": p, "ci_low": lo, "ci_high": hi,
            "n_units": N, "n_treated": N1, "possible_assignments": n_assign, "exact": exact,
            "n_perm": len(sets), "n": len(d), "warnings": warn}
