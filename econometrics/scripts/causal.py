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
                                                  anticipation=δ; warns when the effect starts before the recorded date;
                                                  covariates= in levels (conditional parallel trends): doubly robust
                                                  (Sant'Anna-Zhao 2020), outcome regression or IPW, with the other two
                                                  recomputed as an independent check
    twfe_covariate_diagnostic(df, ..., covariates) hidden linearity bias of TWFE with covariates (Caetano-Callaway
                                                  2024): level test on untreated trends, implicit weights, CS benchmark
    ccc_test(df, y, covariates, group, time)      common causal covariates test (Karim-Webb 2024): are covariate
                                                  slopes equal across groups and periods? wild-bootstrap p-values
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


PS_TRIM = 0.995   # comparison units with a propensity above this get zero weight (as DRDID's trim.level)


def _logit_newton(Z: np.ndarray, d: np.ndarray, maxit: int = 60, tol: float = 1e-10):
    """Logit by Newton-Raphson with step halving and a tiny ridge (keeps separated or collinear
    designs finite). Returns (fitted probabilities, converged flag)."""
    k = Z.shape[1]
    b = np.zeros(k)
    ridge = 1e-8 * np.eye(k)

    def nll(beta):
        eta = Z @ beta
        return np.sum(np.logaddexp(0, eta) - d * eta)

    f = nll(b)
    conv = False
    for _ in range(maxit):
        p = 1 / (1 + np.exp(-(Z @ b)))
        grad = Z.T @ (p - d)
        H = (Z * (p * (1 - p))[:, None]).T @ Z + ridge
        step = np.linalg.lstsq(H, grad, rcond=None)[0]
        s = 1.0
        while s > 1e-6:
            bn = b - s * step
            fn = nll(bn)
            if np.isfinite(fn) and fn <= f + 1e-12:
                break
            s /= 2
        b, f_old, f = bn, f, fn
        if abs(f_old - f) < tol * (1 + abs(f)):
            conv = True
            break
    return 1 / (1 + np.exp(-(Z @ b))), conv


def _did_cell_cov(dy: np.ndarray, Z: np.ndarray, treat: np.ndarray, methods=("dr", "reg", "ipw")) -> dict:
    """2x2 ATT with covariates for one (g, t) cell: rows are the cohort g and its comparison group,
    dy = Y_t - Y_base, Z = covariate design with intercept, treat = 1 for cohort g.
    reg: outcome regression of dy on Z among comparisons (Heckman, Ichimura & Todd 1997);
    ipw: normalized (Hajek) inverse probability weighting (Abadie 2005);
    dr: doubly robust, Sant'Anna & Zhao (2020), consistent if either model is right.
    Also returns overlap information (largest comparison propensity, units trimmed)."""
    c = treat == 0
    out = {"ps_max": np.nan, "trimmed": 0, "ps_converged": True}
    beta = np.linalg.lstsq(Z[c], dy[c], rcond=None)[0]
    m0 = Z @ beta
    w1 = treat / treat.mean()
    if "reg" in methods:
        out["reg"] = float(np.sum(w1 * (dy - m0)) / len(dy))
    if "ipw" in methods or "dr" in methods:
        p, conv = _logit_newton(Z, treat.astype(float))
        out["ps_converged"] = conv
        out["ps_max"] = float(p[c].max())
        keep = ~(c & (p >= PS_TRIM))
        out["trimmed"] = int((~keep).sum())
        p = np.clip(p, 1e-12, PS_TRIM)
        w0 = keep * (1 - treat) * p / (1 - p)
        w0 = w0 / w0.mean() if w0.sum() > 0 else w0 * np.nan
        if "ipw" in methods:
            out["ipw"] = float(np.mean(w1 * dy) - np.mean(w0 * dy))
        if "dr" in methods:
            r = dy - m0
            out["dr"] = float(np.mean(w1 * r) - np.mean(w0 * r))
    return out


def att_gt(df: pd.DataFrame, unit: str, time: str, y: str, gvar: str, control: str = "never",
           n_boot: int = 199, seed: int = 0, anticipation: int = 0, covariates: list[str] | None = None,
           est_method: str = "dr", x_periods: str = "base", verify: bool = True, show: bool = True) -> dict:
    """Callaway & Sant'Anna (2021) ATT(g,t) on a balanced panel, with or without covariates.
    gvar = first treated period (0 or NaN = never treated). control = 'never' or 'notyet'.
    anticipation = δ periods in which units may react before g (law announced before it
    takes effect): post cells (t >= g-δ) use base g-δ-1, not-yet controls must be untreated
    until t+δ. Pre-period cells use the short (t-1) base.

    Covariates (parallel trends that hold only conditional on X; Caetano & Callaway 2024):
    covariates = columns entered in LEVELS, never differenced away (that is what removes
    hidden linearity bias from a TWFE regression). est_method = 'dr' (doubly robust,
    Sant'Anna & Zhao 2020; default), 'reg' (outcome regression) or 'ipw' (normalized IPW).
    x_periods = 'base': each cell uses X at its base period (pre-treatment values, safe when X
    could respond to treatment); 'both': X at the base period and its change up to t, for
    time-varying covariates that are NOT affected by treatment (Caetano & Callaway's setting).
    Comparison units with propensity >= 0.995 get zero weight (warned). verify=True also
    computes the other two estimators on the full sample (an independent recomputation): if
    they differ from the chosen one by more than one SE, the outcome and propensity models
    disagree, a sign that at least one is misspecified (warned). Covariate effects that differ
    ACROSS GROUPS (not only over time) bias every estimator here; check with ccc_test.

    Inference: unit-level bootstrap (resampling units with replacement, re-estimating every
    nuisance model); n_boot=0 skips it.
    Warns when either of the two pre-period cells right before the (anticipation-adjusted)
    date is significant (Bonferroni, ALPHA/2 each): the effect starts before the recorded date (anticipation or a misdated
    treatment) → check the date and run did_timing_sensitivity."""
    covs = list(covariates) if covariates else []
    if est_method not in ("dr", "reg", "ipw"):
        raise ValueError("est_method must be 'dr', 'reg' or 'ipw'")
    if x_periods not in ("base", "both"):
        raise ValueError("x_periods must be 'base' or 'both'")
    d = df[[unit, time, y, gvar, *covs]].copy()
    d[gvar] = d[gvar].fillna(0)
    Y = d.pivot(index=unit, columns=time, values=y)
    if Y.isna().any().any():
        raise ValueError("att_gt needs a balanced panel (drop units with missing periods)")
    G = d.groupby(unit)[gvar].first().reindex(Y.index).to_numpy()
    T = np.array(Y.columns)
    Ym = Y.to_numpy()
    # covariates as a (units, periods, k) array; time-invariant columns are simply constant
    if covs:
        Xa = np.stack([d.pivot(index=unit, columns=time, values=c).reindex(index=Y.index, columns=T).to_numpy(float)
                       for c in covs], axis=2)
        if np.isnan(Xa).any():
            raise ValueError("att_gt: covariates have missing values; impute or drop those units first")
    dlt = int(anticipation)
    cohorts = sorted(g for g in np.unique(G) if g > 0 and (g - dlt - 1) in T)
    dropped = [g.item() if hasattr(g, "item") else g for g in sorted(g for g in np.unique(G) if g > 0 and g not in cohorts)]
    diag = {"ps_max": 0.0, "trimmed": 0, "ps_failed": 0, "cells": 0, "small_cells": 0}

    def design(Xb, idx, j, b):
        cols = [np.ones(len(idx))]
        xb = Xb[idx, b, :]
        cols.append(xb)
        if x_periods == "both":
            dx = Xb[idx, j, :] - xb
            keep = dx.std(axis=0) > 1e-12          # time-invariant covariates have no change
            if keep.any():
                cols.append(dx[:, keep])
        return np.column_stack(cols)

    def compute(idx, methods=(est_method,), record=False):
        Yb, Gb = Ym[idx], G[idx]
        Xb = Xa[idx] if covs else None
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
                if not covs:
                    att = (Yb[tg, j] - Yb[tg, b]).mean() - (Yb[cg, j] - Yb[cg, b]).mean()
                    cells.append((g, t, t - g, att, tg.sum()))
                    continue
                rows = np.where(tg | cg)[0]
                Z = design(Xb, rows, j, b)
                dy = Yb[rows, j] - Yb[rows, b]
                res = _did_cell_cov(dy, Z, tg[rows].astype(float), methods)
                if record:
                    diag["cells"] += 1
                    diag["ps_max"] = max(diag["ps_max"], res["ps_max"]) if np.isfinite(res["ps_max"]) else diag["ps_max"]
                    diag["trimmed"] += res["trimmed"]
                    diag["ps_failed"] += int(not res["ps_converged"])
                    diag["small_cells"] += int(min(tg.sum(), cg.sum()) <= Z.shape[1])
                cells.append((g, t, t - g, *[res[m] for m in methods], tg.sum()))
        cols = ["att"] if len(methods) == 1 else list(methods)
        return pd.DataFrame(cells, columns=["g", "t", "e", *cols, "n_g"])

    if not cohorts:
        raise ValueError("att_gt: no cohort has the base period g-anticipation-1 inside the panel")
    all_idx = np.arange(len(Ym))
    if covs:
        meths = ("dr", "reg", "ipw") if verify else (est_method,)
        full = compute(all_idx, meths, record=True)
        base = full[["g", "t", "e", est_method, "n_g"]].rename(columns={est_method: "att"}) if verify else full
    else:
        base = compute(all_idx)
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
    method = "no covariates" if not covs else {"dr": "doubly robust", "reg": "outcome regression",
                                               "ipw": "IPW"}[est_method] + f", X: {', '.join(covs)} ({x_periods})"
    warns = []
    if flag:
        warns.append(f"The outcome already moves at e = {k_chk} (|z| > {zc:.2f}): the effect starts before the recorded"
                     " date (anticipation or a misdated treatment). Confirm the date; re-run with anticipation="
                     f"{-k_chk} and did_timing_sensitivity before reporting the ATT.")
    cross = None
    if covs:
        if diag["trimmed"]:
            warns.append(f"Limited overlap: {diag['trimmed']} comparison-unit cells had propensity >= {PS_TRIM} and got"
                         f" zero weight (largest {diag['ps_max']:.4f}). Treated units with no comparable controls:"
                         " check balance_table and consider fewer or coarser covariates.")
        elif diag["ps_max"] > 0.95:
            warns.append(f"Weak overlap: largest comparison propensity {diag['ps_max']:.3f}; IPW and DR weights are"
                         " concentrated on few units.")
        if diag["ps_failed"]:
            warns.append(f"The propensity logit did not converge in {diag['ps_failed']} cells (near separation).")
        if diag["small_cells"]:
            warns.append(f"{diag['small_cells']} cells have a cohort or comparison group no larger than the number of"
                         " covariate parameters: those cells are fitted exactly and their ATTs are unreliable.")
        if verify:
            fp = full[full["e"] >= 0]
            cross = pd.Series({m: np.average(fp[m], weights=fp["n_g"]) if len(fp) else np.nan
                               for m in ("dr", "reg", "ipw")}, name="overall ATT")
            spread = float(cross.max() - cross.min())
            if np.isfinite(se_simple) and se_simple > 0 and spread > se_simple:
                warns.append(f"Independent recomputation disagrees: DR {cross['dr']:.4f}, outcome regression "
                             f"{cross['reg']:.4f}, IPW {cross['ipw']:.4f} differ by {spread / se_simple:.1f} SE. "
                             "The outcome and propensity models disagree, so at least one is misspecified: add "
                             "flexibility (squares, interactions, logs) or check overlap before reporting.")
    out = {"overall": _out("ATT (CS, simple)", simple, se_simple, len(Ym), anticipation=dlt,
                           method=method, warnings=warns),
           "dynamic": dyn_tab, "gt": gt, "flags": {"effect_before_date": flag},
           "cohorts_dropped": dropped, "warnings": warns}
    if covs:
        out["cross_check"] = cross
        out["overlap"] = {k: diag[k] for k in ("ps_max", "trimmed", "ps_failed", "cells")}
    if show:
        o = out["overall"]
        print(f"\nCallaway-Sant'Anna ({control} controls, {len(cohorts)} cohorts, {n_boot} bootstrap reps"
              + (f", anticipation = {dlt}" if dlt else "") + f"; {method})")
        print(f"Overall ATT: {o['estimate']:.4f} (SE {o['se']:.4f}, 95% CI {o['ci_low']:.4f} to {o['ci_high']:.4f})")
        print("Dynamic (event-time) ATT:\n" + dyn_tab.to_string(float_format=lambda v: f"{v:.4f}"))
        if cross is not None:
            print("Independent recomputation of the overall ATT: "
                  + ", ".join(f"{k} {v:.4f}" for k, v in cross.items()))
        if dropped:
            print(f"  Cohorts dropped (base period g-{dlt}-1 outside the panel): {dropped}")
        for w in warns:
            print("  WARNING: " + w)
    return out


def did_timing_sensitivity(df: pd.DataFrame, unit: str, time: str, y: str, gvar: str, max_shift: int = 2,
                           control: str = "never", n_boot: int = 199, seed: int = 0, covariates: list[str] | None = None,
                           est_method: str = "dr", x_periods: str = "base", show: bool = True) -> dict:
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
    baseline; otherwise the conclusion depends on the exact date.
    covariates / est_method / x_periods: passed to att_gt, so the check uses the same specification."""
    d = df.copy()
    cov_kw = dict(covariates=covariates, est_method=est_method, x_periods=x_periods, verify=False)
    g0 = d[gvar].fillna(0)
    rows = []
    for k in range(-max_shift, max_shift + 1):
        d["_g"] = np.where(g0 > 0, g0 + k, 0)
        try:
            r = att_gt(d, unit, time, y, "_g", control=control, n_boot=n_boot, seed=seed, show=False, **cov_kw)
            o = r["overall"]
            rows.append({"shift": k, "att": o["estimate"], "se": o["se"], "ci_low": o["ci_low"],
                         "ci_high": o["ci_high"], "p": o["p"]})
        except ValueError:
            rows.append({"shift": k, "att": np.nan, "se": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan})
    tab = pd.DataFrame(rows).set_index("shift")
    d["_g"] = np.where(g0 > 0, g0 + max_shift, 0)
    rob = att_gt(d, unit, time, y, "_g", control=control, n_boot=n_boot, seed=seed,
                 anticipation=2 * max_shift, show=False, **cov_kw)
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
                        covariates: list[str] | None = None, est_method: str = "dr", x_periods: str = "base",
                        show: bool = True) -> dict:
    """Staggered DiD when each unit's adoption date is only known to lie in a window
    [g_low, g_high] (e.g. a law phased in court by court with imprecise records).
    Never-treated units: g_low = g_high = 0 or NaN.

    1. Bounds-style estimate: event date = g_high, anticipation = max window width, so
       every base period precedes any plausible adoption (Callaway-Sant'Anna, bootstrap CI).
    2. Date draws: n_draws estimates with each unit's date drawn uniformly from its
       window; the spread shows how much date uncertainty alone moves the ATT.
    covariates / est_method / x_periods: passed to att_gt (same specification as the main estimate)."""
    d = df.copy()
    cov_kw = dict(covariates=covariates, est_method=est_method, x_periods=x_periods, verify=False)
    lo = d.groupby(unit)[g_low].first().fillna(0)
    hi = d.groupby(unit)[g_high].first().fillna(0)
    if ((hi > 0) & (lo > hi)).any():
        raise ValueError("g_low must be <= g_high for every treated unit")
    lo = lo.where(lo > 0, hi)
    width = int((hi - lo)[hi > 0].max()) if (hi > 0).any() else 0
    d["_g"] = d[unit].map(hi)
    rob = att_gt(d, unit, time, y, "_g", control=control, n_boot=n_boot, seed=seed,
                 anticipation=width, show=False, **cov_kw)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_draws):
        gd = pd.Series(np.where(hi > 0, rng.integers(lo.astype(int), hi.astype(int) + 1), 0), index=hi.index)
        d["_g"] = d[unit].map(gd)
        try:
            draws.append(att_gt(d, unit, time, y, "_g", control=control, n_boot=0, show=False,
                                **cov_kw)["overall"]["estimate"])
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


def _wald_block(b: np.ndarray, V: np.ndarray, R: np.ndarray):
    """Wald statistic for R b = 0; returns (W, q, p, ok). ok = False when R V R' is singular."""
    d = R @ b
    C = R @ V @ R.T
    q = R.shape[0]
    if np.linalg.matrix_rank(C) < q or not np.all(np.isfinite(C)):
        return np.nan, q, np.nan, False
    w = float(d @ np.linalg.solve(C, d))
    return w, q, float(stats.chi2.sf(w, q)), True


def _untreated_mask(d: pd.DataFrame, time: str, gvar: str | None, untreated: str | None, anticipation: int = 0):
    if untreated is not None:
        return d[untreated].fillna(0).astype(int).eq(1).to_numpy()
    if gvar is None:
        raise ValueError("Give gvar (first treated period, 0/NaN = never) or an untreated indicator")
    g = d[gvar].fillna(0).to_numpy(float)
    return (g == 0) | (d[time].to_numpy(float) < g - anticipation)


_WEBB6 = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def ccc_test(df: pd.DataFrame, y: str, covariates: list[str], group: str, time: str, gvar: str | None = None,
             untreated: str | None = None, cluster: str | None = None, alpha: float = ALPHA, anticipation: int = 0,
             inference: str = "wild", n_boot: int = 399, seed: int = 0, alltests: bool = False,
             show: bool = True) -> dict:
    """Do covariates affect the outcome the same way in every group and period? A test of the
    common causal covariates (CCC) assumption of Karim & Webb (2024, "Good Controls Gone Bad"),
    written from the method's description (no code from the authors' restricted-license ccctest).

    Uses only untreated observations (never treated, and treated units before g - anticipation),
    so treatment cannot contaminate the covariate slopes. group = the level at which treatment
    is assigned (state, region, cohort), with several observations per group-period cell.
    Stage 1: slopes specific to each group-period cell vs one common slope (cell intercepts
    kept): Wald test of K(U-1) restrictions. Nonrejection selects 'homogeneous'.
    Stage 2 (after a rejection, or alltests=True): additive model with group slopes plus
    period deviations; tests 'slopes equal across groups' and 'no period deviations', Holm-
    adjusted. Selection: group only -> 'group'; time only -> 'time'; both or neither -> 'twoway'.

    Inference. Robust Wald statistics (HC1, or CR1 clustered on `cluster`: with repeated units,
    cluster on the unit, since HC1 ignores serial dependence). With many restrictions their
    chi-square reference over-rejects badly under heteroskedasticity (in the skill's tests,
    10% with 120 obs per cell and 25% with 40, for a nominal 5%), so the default
    inference='wild' takes p-values from a wild bootstrap imposed under each null
    (restricted residuals, Rademacher weights, Webb weights with fewer than 12 clusters,
    re-centred within cells; 5-7% size in the same tests). inference='asymptotic' gives the
    chi-square p-values (warned). When the robust covariance cannot identify every restriction
    (more restrictions than clusters), the classical F statistic is used instead (bootstrapped
    under 'wild'; warned).

    What the selection means for estimation:
      homogeneous -> covariate effects common: TWFE or att_gt with covariates are fine on this front.
      time        -> effects change over time but are common across groups: TWFE with time-varying
                     covariates suffers hidden linearity bias; use att_gt(covariates=...) (levels).
      group/twoway-> effects differ across groups: TWFE and Callaway-Sant'Anna with covariates are
                     biased; use intersection DiD (DID-INT, Karim & Webb; Julia DiDInt.jl, R didintrjl),
                     not implemented here, or interact the covariates with group.
    CCC is not a test of parallel trends, and nonrejection does not prove equal slopes (power
    falls with few observations per cell)."""
    if inference not in ("wild", "asymptotic"):
        raise ValueError("inference must be 'wild' or 'asymptotic'")
    covs = list(covariates)
    k = len(covs)
    d = df.copy().reset_index(drop=True)
    use = _untreated_mask(d, time, gvar, untreated, anticipation)
    cols = [y, group, time, *covs] + ([cluster] if cluster else [])
    use &= d[cols].notna().all(axis=1).to_numpy()
    w = d.loc[use, cols].reset_index(drop=True)
    n = len(w)
    keys = pd.MultiIndex.from_frame(w[[group, time]])
    cell_id, cell_keys = pd.factorize(keys)
    U = len(cell_keys)
    counts = np.bincount(cell_id, minlength=U)
    warns = []
    if (counts < k + 2).any():
        bad = int((counts < k + 2).sum())
        raise ValueError(f"ccc_test: {bad} group-period cells have fewer than {k + 2} untreated observations; "
                         "use a coarser group (region or cohort, not the unit) or fewer covariates")
    Yv = w[y].to_numpy(float)
    X = w[covs].to_numpy(float)

    def centre(v):
        """Within-cell centring: absorbs the cell intercepts (their U parameters stay in the dof)."""
        if v.ndim == 1:
            return v - (np.bincount(cell_id, weights=v, minlength=U) / counts)[cell_id]
        return np.column_stack([centre(v[:, j]) for j in range(v.shape[1])])

    yc, xc = centre(Yv), centre(X)
    if cluster:
        cl = pd.factorize(w[cluster])[0]
        G = int(cl.max() + 1)
    else:
        cl, G = np.arange(n), n

    def sandwich(bread_inv, scores, P):
        if cluster:
            S = np.zeros((G, scores.shape[1]))
            np.add.at(S, cl, scores)
            meat, c = S.T @ S, G / (G - 1) * (n - 1) / (n - P)
        else:
            meat, c = scores.T @ scores, n / (n - P)
        return c * bread_inv @ meat @ bread_inv

    # ---- Stage 1: cell-specific slopes; the design is block-diagonal by cell ----
    P1 = U + k * U
    if n <= P1:
        raise ValueError(f"ccc_test: {n} untreated observations for {P1} parameters")
    Ainv = np.zeros((U, k, k))
    for c in range(U):
        ix = cell_id == c
        A = xc[ix].T @ xc[ix]
        if np.linalg.matrix_rank(A) < k:
            raise ValueError("ccc_test: a covariate does not vary within some group-period cell; drop it or "
                             "use coarser cells")
        Ainv[c] = np.linalg.inv(A)
    bread1 = np.zeros((U * k, U * k))
    for c in range(U):
        bread1[c * k:(c + 1) * k, c * k:(c + 1) * k] = Ainv[c]
    R1 = np.zeros((k * (U - 1), k * U))
    for c in range(1, U):
        R1[(c - 1) * k:c * k, c * k:(c + 1) * k] = np.eye(k)
        R1[(c - 1) * k:c * k, :k] = -np.eye(k)
    cols_k = np.arange(k)

    def stage1(yv, robust=True):
        xy = np.zeros((U, k))
        np.add.at(xy, cell_id, xc * yv[:, None])
        bcell = np.einsum("ukl,ul->uk", Ainv, xy)
        e = yv - np.sum(xc * bcell[cell_id], axis=1)
        ssr_u = float(e @ e)
        if robust:
            sc = np.zeros((n, U * k))
            sc[np.arange(n)[:, None], cell_id[:, None] * k + cols_k[None, :]] = xc * e[:, None]
            Wv, q, _, ok = _wald_block(bcell.ravel(), sandwich(bread1, sc, P1), R1)
            if ok:
                return Wv, True, bcell
        bc_ = np.linalg.lstsq(xc, yv, rcond=None)[0]
        ssr_r = float(np.sum((yv - xc @ bc_) ** 2))
        q = k * (U - 1)
        return ((ssr_r - ssr_u) / q) / (ssr_u / (n - P1)), False, bcell

    W1, ok1, b_cell = stage1(yc)
    q1 = k * (U - 1)
    bc = np.linalg.lstsq(xc, yc, rcond=None)[0]

    def boot_p(stat_fn, fit_r, e_r, obs, robust):
        rng = np.random.default_rng(seed)
        Gb = G if cluster else n
        pool = _WEBB6 if (cluster and G < 12) else np.array([-1.0, 1.0])
        hits = 0
        for _ in range(n_boot):
            v = rng.choice(pool, Gb)[cl]
            ys = centre(fit_r + e_r * v)
            hits += stat_fn(ys, robust)[0] >= obs
        return (1 + hits) / (n_boot + 1)

    if inference == "wild":
        fr = xc @ bc
        p1 = boot_p(stage1, fr, yc - fr, W1, ok1)
    else:
        p1 = float(stats.chi2.sf(W1, q1)) if ok1 else float(stats.f.sf(W1, q1, n - P1))
    kind = lambda ok: ("Wald " + ("CR1" if cluster else "HC1") if ok else "F (classical)") + \
        (", wild bootstrap" if inference == "wild" else "")
    rows = [{"test": "stage 1: slopes equal in every group-period cell", "stat": W1, "df": q1, "p": p1,
             "type": kind(ok1)}]
    if not ok1:
        warns.append(f"Stage 1 has {q1} restrictions, more than the {'clustered ' if cluster else ''}covariance can "
                     f"identify ({G} clusters): the classical F statistic is used"
                     + (" (wild-bootstrapped, so its p-value remains valid under heteroskedasticity"
                        + (" and clustering)." if cluster else ").") if inference == "wild" else
                        " with its F reference, which assumes homoskedastic, independent errors."))

    # ---- Stage 2: additive group slopes + period deviations ----
    gi, glev = pd.factorize(w[group], sort=True)
    ti, tlev = pd.factorize(w[time], sort=True)
    S, Tn = len(glev), len(tlev)
    p_g = p_t = np.nan
    run2 = alltests or (np.isfinite(p1) and p1 <= alpha)
    if run2 and (S < 2 or Tn < 2):
        warns.append("Stage 2 needs at least two groups and two periods among untreated observations.")
        run2 = False
    if run2:
        Z = np.column_stack([xc] + [xc * (gi == s_)[:, None] for s_ in range(1, S)]
                            + [xc * (ti == t_)[:, None] for t_ in range(1, Tn)])
        m = Z.shape[1]
        P2 = U + m
        ZZ = Z.T @ Z
        if np.linalg.matrix_rank(ZZ) < m:
            warns.append("Stage 2 design is rank deficient (a group or period has no covariate variation).")
            run2 = False
    adj = np.array([np.nan, np.nan])
    if run2:
        Zi = np.linalg.inv(ZZ)
        Rg = np.zeros((k * (S - 1), m)); Rg[:, k:k * S] = np.eye(k * (S - 1))
        Rt = np.zeros((k * (Tn - 1), m)); Rt[:, k * S:] = np.eye(k * (Tn - 1))
        for lab, R in (("group", Rg), ("time", Rt)):
            keep = ~R.any(axis=0)
            Zr = Z[:, keep]
            q = R.shape[0]

            def stage2(yv, robust=True, R=R, Zr=Zr, q=q):
                b2 = Zi @ (Z.T @ yv)
                e2 = yv - Z @ b2
                if robust:
                    Wv, _, _, ok = _wald_block(b2, sandwich(Zi, Z * e2[:, None], P2), R)
                    if ok:
                        return Wv, True
                er = yv - Zr @ np.linalg.lstsq(Zr, yv, rcond=None)[0]
                ssr2 = float(e2 @ e2)
                return ((er @ er - ssr2) / q) / (ssr2 / (n - P2)), False

            Wv, ok = stage2(yc)
            if inference == "wild":
                frr = Zr @ np.linalg.lstsq(Zr, yc, rcond=None)[0]
                pv = boot_p(stage2, frr, yc - frr, Wv, ok)
            else:
                pv = float(stats.chi2.sf(Wv, q)) if ok else float(stats.f.sf(Wv, q, n - P2))
            if not ok:
                warns.append(f"Stage 2 '{lab}' test: robust covariance singular ({q} restrictions, {G} clusters); "
                             "classical F statistic used.")
            rows.append({"test": f"stage 2: slopes equal across {'groups' if lab == 'group' else 'periods'}",
                         "stat": Wv, "df": q, "p": pv, "type": kind(ok)})
            if lab == "group":
                p_g = pv
            else:
                p_t = pv
        pr = np.array([p_g, p_t])
        order = np.argsort(pr, kind="stable")
        adj[order] = np.minimum(1, np.maximum.accumulate(pr[order] * np.array([2, 1])))
    tab = pd.DataFrame(rows).set_index("test")
    tab["p_holm"] = np.nan
    if len(tab) == 3:
        tab.iloc[1, tab.columns.get_loc("p_holm")] = adj[0]
        tab.iloc[2, tab.columns.get_loc("p_holm")] = adj[1]
    if inference == "asymptotic" and q1 >= 10:
        warns.append(f"Asymptotic chi-square p-values with {q1} restrictions over-reject under heteroskedasticity "
                     "(see the docstring); prefer inference='wild'.")
    # ---- selection ----
    if not np.isfinite(p1):
        sel, reason = "undetermined", "stage 1 could not be computed"
    elif p1 > alpha:
        sel, reason = "homogeneous", "stage 1 not rejected"
    elif not np.isfinite(adj).all():
        sel, reason = "undetermined", "stage 1 rejected but stage 2 could not be computed"
    else:
        rg, rt = bool(adj[0] <= alpha), bool(adj[1] <= alpha)
        sel, reason = {(True, False): ("group", "Holm: groups only"), (False, True): ("time", "Holm: periods only"),
                       (True, True): ("twoway", "Holm: both"),
                       (False, False): ("twoway", "Holm: neither (stage 1 rejected)")}[(rg, rt)]
    advice = {
        "homogeneous": "Covariate effects look common across groups and periods: TWFE or att_gt with covariates are "
                       "fine on this front (parallel trends still has to be argued; nonrejection is not proof).",
        "time": "Covariate effects change over time but not across groups: a TWFE regression with time-varying "
                "covariates suffers hidden linearity bias; use att_gt(covariates=..., x_periods='both').",
        "group": "Covariate effects differ across groups: TWFE and Callaway-Sant'Anna with covariates are biased. Use "
                 "intersection DiD (DID-INT; DiDInt.jl or R didintrjl) or interact the covariates with group.",
        "twoway": "Covariate effects differ across groups and periods: TWFE and Callaway-Sant'Anna with covariates are "
                  "biased. Use intersection DiD (DID-INT; DiDInt.jl or R didintrjl).",
        "undetermined": "The test could not be completed; see the warnings.",
    }[sel]
    out = {"table": tab, "selected": sel, "reason": reason, "advice": advice, "n": n, "n_cells": int(U),
           "n_clusters": G if cluster else None, "inference": inference,
           "slopes_by_cell": pd.DataFrame(b_cell, columns=covs, index=cell_keys),
           "common_slopes": pd.Series(bc, index=covs), "warnings": warns}
    if show:
        print(f"\nCommon causal covariates test (Karim & Webb), {n} untreated obs in {U} group-period cells"
              + (f", CR1 by {cluster} ({G} clusters)" if cluster else ", HC1")
              + (f", wild bootstrap ({n_boot} draws)" if inference == "wild" else ", asymptotic"))
        print(tab.to_string(float_format=lambda v: f"{v:.4g}"))
        print(f"Selected: {sel} ({reason}). {advice}")
        for m_ in warns:
            print("  WARNING: " + m_)
    return out


def twfe_covariate_diagnostic(df: pd.DataFrame, unit: str, time: str, y: str, gvar: str, covariates: list[str],
                              control: str = "never", est_method: str = "dr", x_periods: str = "both",
                              n_boot: int = 199, seed: int = 0, show: bool = True) -> dict:
    """Is a TWFE regression with covariates trustworthy here? Diagnostics in the spirit of
    Caetano & Callaway (2024), "Difference-in-differences when parallel trends holds conditional
    on covariates", on a balanced panel.

    1. Hidden linearity: unit fixed effects difference the covariates too, so TWFE uses only
       their CHANGES. Among untreated observations, first differences of y are regressed on
       period effects, ΔX and the lagged LEVELS of X (clustered by unit). Rejecting 'levels do
       not matter' means trends depend on covariate levels, which TWFE throws away
       (time-invariant covariates drop out of TWFE entirely).
    2. Implicit weights: by Frisch-Waugh, the TWFE coefficient weights each treated
       observation by its residualized treatment dummy; negative weights mean already-treated
       units serve as controls (forbidden comparisons) and the estimate need not lie between the
       unit-level effects.
    3. Benchmark: att_gt with the same covariates in levels (doubly robust by default); the gap
       is reported in units of its bootstrap SE (approximate: the two estimates are correlated).
    Verdict 'unreliable' if (1) rejects at 5%, or any treated weight is negative, or the gap
    exceeds 2 SE; then report att_gt(covariates=...)."""
    from linearmodels.panel import PanelOLS
    covs = list(covariates)
    d = df[[unit, time, y, gvar, *covs]].copy()
    d[gvar] = d[gvar].fillna(0)
    d = d.sort_values([unit, time]).reset_index(drop=True)
    if d.groupby(unit)[time].count().nunique() != 1 or d[[y, *covs]].isna().any().any():
        raise ValueError("twfe_covariate_diagnostic needs a balanced panel without missing values")
    d["_D"] = ((d[gvar] > 0) & (d[time] >= d[gvar])).astype(float)
    pdf = d.set_index([unit, time])
    tv = [c for c in covs if d.groupby(unit)[c].nunique().max() > 1]
    ti_ = [c for c in covs if c not in tv]
    warns = []
    if ti_:
        warns.append(f"Time-invariant covariates {ti_} are absorbed by the unit fixed effects: TWFE cannot adjust "
                     "for them at all.")
    exog = pdf[["_D", *tv]]
    tw = PanelOLS(pdf[y], exog, entity_effects=True, time_effects=True, drop_absorbed=True).fit(
        cov_type="clustered", cluster_entity=True)
    twfe = _out("TWFE with covariates", float(tw.params["_D"]), float(tw.std_errors["_D"]), len(d))
    # 2. implicit weights (Frisch-Waugh): residualize D on both FEs and the time-varying covariates
    if tv:
        aux = PanelOLS(pdf["_D"], pdf[tv], entity_effects=True, time_effects=True, drop_absorbed=True).fit()
        r = aux.resids.to_numpy()
    else:
        Dm = pdf["_D"].unstack()
        r = (Dm - Dm.mean(axis=1).to_numpy()[:, None] - Dm.mean(axis=0).to_numpy()[None, :] + Dm.values.mean()).stack()
        r = r.reindex(pdf.index).to_numpy()
    treated = pdf["_D"].to_numpy() == 1
    wt = r[treated] / r[treated].sum()
    neg_share = float((wt < 0).mean())
    neg_sum = float(wt[wt < 0].sum())
    if neg_share > 0:
        warns.append(f"{neg_share:.0%} of treated observations get NEGATIVE TWFE weight (sum {neg_sum:.3f}): "
                     "already-treated units act as controls.")
    # 1. hidden linearity test on untreated first differences
    g = d.groupby(unit)
    fd = pd.DataFrame({"_dy": g[y].diff(), "_t": d[time], "_u": d[unit]})
    lev_cols, dif_cols = [], []
    for c in covs:
        fd["L_" + c] = g[c].shift(1)
        lev_cols.append("L_" + c)
        if c in tv:
            fd["d_" + c] = g[c].diff()
            dif_cols.append("d_" + c)
    pre = (d[gvar] == 0) | (d[time] < d[gvar])
    fd = fd[pre.to_numpy() & fd["_dy"].notna().to_numpy()]
    import statsmodels.api as sm
    Xh = pd.concat([pd.get_dummies(fd["_t"], prefix="t", drop_first=False, dtype=float),
                    fd[dif_cols + lev_cols]], axis=1)
    hl = sm.OLS(fd["_dy"], Xh).fit(cov_type="cluster", cov_kwds={"groups": fd["_u"]})
    Rm = np.zeros((len(lev_cols), Xh.shape[1]))
    for i, c in enumerate(lev_cols):
        Rm[i, list(Xh.columns).index(c)] = 1
    wt_ = hl.wald_test(Rm, scalar=True)
    p_hl = float(wt_.pvalue)
    lev_tab = pd.DataFrame({"coef": hl.params[lev_cols], "se": hl.bse[lev_cols], "p": hl.pvalues[lev_cols]})
    if p_hl < 0.05:
        warns.append(f"Hidden linearity: untreated outcome trends depend on covariate LEVELS (joint p = {p_hl:.3g}), "
                     "which TWFE differences away. TWFE with covariates is biased here.")
    # 3. benchmark
    cs = att_gt(df, unit, time, y, gvar, control=control, n_boot=n_boot, seed=seed, covariates=covs,
                est_method=est_method, x_periods=x_periods, show=False)
    co = cs["overall"]
    gap = twfe["estimate"] - co["estimate"]
    gap_se = gap / co["se"] if np.isfinite(co["se"]) and co["se"] > 0 else np.nan
    if np.isfinite(gap_se) and abs(gap_se) > 2:
        warns.append(f"TWFE ({twfe['estimate']:.4f}) and Callaway-Sant'Anna with covariates ({co['estimate']:.4f}) differ "
                     f"by {abs(gap_se):.1f} SE.")
    warns += [w_ for w_ in cs["warnings"]]
    unreliable = bool(p_hl < 0.05 or neg_share > 0 or (np.isfinite(gap_se) and abs(gap_se) > 2))
    out = {"twfe": twfe, "att_gt": co, "gap": gap, "gap_in_se": gap_se, "hidden_linearity_p": p_hl,
           "level_coefficients": lev_tab, "negative_weight_share": neg_share, "negative_weight_sum": neg_sum,
           "verdict": "unreliable" if unreliable else "no evidence against TWFE", "warnings": warns}
    if show:
        print(f"\nTWFE with covariates vs conditional parallel trends ({', '.join(covs)})")
        print(f"TWFE: {twfe['estimate']:.4f} (SE {twfe['se']:.4f})   att_gt ({est_method}, {x_periods}): "
              f"{co['estimate']:.4f} (SE {co['se']:.4f})   gap {gap:+.4f} ({gap_se:+.1f} SE)")
        print(f"Hidden linearity test (levels of X in untreated trends): joint p = {p_hl:.4g}")
        print(lev_tab.to_string(float_format=lambda v: f"{v:.4f}"))
        print(f"Treated observations with negative TWFE weight: {neg_share:.1%}")
        print("Verdict: " + ("TWFE with covariates is NOT reliable here; report att_gt with covariates."
                             if unreliable else "no evidence against TWFE on these checks (report both)."))
        for m_ in warns:
            print("  WARNING: " + m_)
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
