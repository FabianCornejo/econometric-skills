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
    did_design(df, unit, time, d)                 first step of any DiD (v2.5.0): binary or dose, absorbing or
                                                  reversible, one or several dates, never-treated units and stayers;
                                                  the design among the five of de Chaisemartin-D'Haultfœuille's guide,
                                                  the estimators that fit it and gvar (binary absorbing only)
    event_study(df, unit, time, y, event_time)    TWFE leads/lags with clustered SE and joint pre-trend test
    att_gt(df, unit, time, y, gvar)               Callaway-Sant'Anna (2021) group-time ATTs, robust to staggered timing;
                                                  anticipation=δ; warns when the effect starts before the recorded date;
                                                  covariates= in levels (conditional parallel trends): doubly robust
                                                  (Sant'Anna-Zhao 2020), outcome regression or IPW, with the other two
                                                  recomputed as an independent check; joint pre-trend test
                                                  (F reference, fewer restrictions with few clusters; v2.5.0)
    did_imputation(df, unit, time, y, gvar)       Borusyak-Jaravel-Spiess (2024) imputation estimator (v2.5.0): FE fitted
                                                  on untreated observations only, Theorem 3 clustered SEs, event study,
                                                  pre-trend test on untreated observations; unbalanced panels allowed
    did_compare(df, unit, time, y, gvar)          att_gt vs did_imputation on common cells, difference tested with a
                                                  paired bootstrap (v2.5.0): a gap points to the pre-trends
    bacon_decomposition(df, unit, time, y, gvar)  Goodman-Bacon (2021): TWFE as a weighted average of every 2x2,
                                                  with the weight of forbidden (already-treated control) comparisons
    twfe_weights(df, unit, time, d, y)            de Chaisemartin-D'Haultfœuille (2020) weights of a TWFE coefficient
                                                  (binary or dose, unbalanced allowed): negative weights, sigma_fe
    twfe_covariate_diagnostic(df, ..., covariates) hidden linearity bias of TWFE with covariates (Caetano-Callaway
                                                  2024): level test on untreated trends, implicit weights, CS benchmark
    ccc_test(df, y, covariates, group, time)      common causal covariates test (Karim-Webb 2024): are covariate
                                                  slopes equal across groups and periods? wild-bootstrap p-values
    did_timing_sensitivity(df, ..., gvar)         ATT with dates shifted ±k + date-robust estimate (true date within ±k)
    did_uncertain_dates(df, ..., g_low, g_high)   adoption known only within a window per unit (phased rollouts)
    synth(df, unit, time, y, treated, t0)         synthetic control (Abadie et al., 2010) + placebo-in-space inference
    randomization_inference(df, unit, time, y, gvar)  Fisher randomization p-value and CI for a DiD with few
                                                  treated units (permutes which units adopt, or which clusters with
                                                  cluster=), any statistic
    did_single_treated(df, unit, time, y, treated, t0)  DiD with one (or a handful of) treated unit(s) over a short
                                                  series (v2.5.1): inference from the treated−control gap, AR(1) t
                                                  calibrated by simulation (median-unbiased ρ), block-means second
                                                  route, pre-period trend test and placebo
    did_falsification(df, unit, time, y, gvar)    the Reviewer's falsification matrix (v2.5.1): pre-trends (CS and
                                                  imputation), placebo date, placebo outcomes, permutation (cluster=
                                                  for the assignment level), timing, CS vs imputation; each with its
                                                  MDE, never "passed"; gap-series rows with 1-3 treated units

Notes: for production RD use `rdrobust` (robust bias-corrected CIs) if installed;
for staggered DiD the `csdid`/`differences` packages offer more options. The
functions here are transparent reference implementations, tested on simulations.
"""
from __future__ import annotations

import warnings

import importlib.util

import numpy as np
import pandas as pd
from scipy import optimize, stats
import _console  # noqa: F401  (Windows-safe console output, see _console.py)

ALPHA = 0.05


def _ci(est, se, alpha=ALPHA):
    z = stats.norm.ppf(1 - alpha / 2)
    return est - z * se, est + z * se


def _out(name, est, se, n, alpha=ALPHA, **extra):
    lo, hi = _ci(est, se, alpha)
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
    # v2.6.2 review: "learner or default" called len() on an unfitted sklearn ensemble (AttributeError)
    ly = learner_y if learner_y is not None else RandomForestRegressor(n_estimators=300, min_samples_leaf=5, n_jobs=-1,
                                                                       random_state=seed)
    ld = learner_d if learner_d is not None else RandomForestRegressor(n_estimators=300, min_samples_leaf=5, n_jobs=-1,
                                                                       random_state=seed)
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
    return _out("theta (DML-PLR)", th, se, len(Y), n_folds=n_folds, n_rep=n_rep, seed=seed,
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
    if rho == 0 or cf_y == 0:                       # the limit as cf_d -> 1 too (v2.6.4 review: was inf at cf_d = 1)
        c = 0.0
    else:
        c = abs(rho) * np.sqrt(cf_y * cf_d / (1 - cf_d)) if cf_d < 1 else np.inf
    bound = c * S
    psi_S = (nu2 * psi_sig2 + sig2 * psi_nu2) / (2 * S)
    psi_lo, psi_hi = psi_th - c * psi_S, psi_th + c * psi_S
    return {"theta": th, "sigma2": sig2, "nu2": nu2, "S": S, "bound": bound,
            "theta_lo": th - bound, "theta_hi": th + bound,
            "se_theta": np.sqrt(np.mean(psi_th ** 2) / n),
            "se_lo": np.sqrt(np.mean(psi_lo ** 2) / n), "se_hi": np.sqrt(np.mean(psi_hi ** 2) / n)}


def _dml_pct(x: float, d: int) -> str:
    """A share as dml_sensitivity prints it with d decimals: a positive share that would print as zero says so
    ("< 0.01%"). Module-level since v2.6.1 so its thresholds are tested directly: with |rho| <= 1 they are out of
    reach of real data (an RV below 0.005% that is significant needs n of the order of 1e9)."""
    return f"< {10 ** -d:.{d}f}%" if 0 < x < 0.5 * 10 ** -(d + 2) else f"{x:.{d}%}"


def _dml_report_line(rho, bias_name: str, rho_txt: str = "", target: str = "") -> str:
    """v2.6.4, printed after the bounds by dml_sensitivity (its last line) and dml_bounds (before its warnings): in the
    v2.6.3 validation the memo did not say which rho the DML bounds used (the chat did), and called both the bias bound
    and the confidence bounds "interval". rho_txt is rho as the caller already printed it (so the two lines agree;
    rho itself if not given); target is the value the confidence bounds are judged against when it is not zero
    (q != 1 in dml_sensitivity)."""
    if isinstance(rho, str):                  # dml_bounds with rho="benchmark": each line carries its own rho
        which = "each line's rho next to it (the benchmark's observed correlation, as printed on the line)"
    else:
        which = f"rho = {rho_txt or rho}" + (" (the worst case)" if abs(float(rho)) == 1 else "") + " next to the bounds"
    judge = f"say whether the target value {target} is inside them" if target else "exclude or include zero"
    return (f"  In a report, state {which}, and name each interval: {bias_name} is the bias bound (theta plus or minus "
            f"its bias_bound), not accounting for sampling error; the confidence bounds add sampling error and are the "
            f"ones that {judge}.")


def _check_cf(v, name: str) -> float:
    """cf_y or cf_d of dml_sensitivity as a float in [0, 1] (v2.6.4): a sensitivity parameter (cf_y the share of the
    outcome's residual variance, cf_d as in DoubleML), so a bool, a non-number, NaN, inf, a negative value or one above
    1 is refused (TypeError for the first two, ValueError for the rest)."""
    import numbers
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real):
        raise TypeError(f"{name} must be a number in [0, 1] (a sensitivity parameter), got {v!r}")
    x = float(v)
    if not (np.isfinite(x) and 0 <= x <= 1):
        raise ValueError(f"{name} is a sensitivity parameter and must be finite and in [0, 1] (pass dml_benchmark's "
                         f"cf_y and cf_d as they are), got {v!r}"
                         + (" (cf_d = 1 already gives an infinite bound)" if name == "cf_d" and x > 1 else ""))
    return x


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
      rho  = correlation of the two omitted pieces, in [-1, 1] (1 = worst case, conservative; v2.6.1:
             a value outside is a ValueError).
    Bounds [theta - B, theta + B] and confidence bounds that add sampling error (influence
    functions of theta and S, median over the sample splits of dml_plr). Robustness values:
    rv = strength (cf_y = cf_d) that drives theta to (1 - q) theta; rva = strength at which the
    confidence bound reaches that value (q = 1: makes it no longer significant). Both are searched on
    sqrt(cf) in (1e-9, 1 - 1e-9): 1.0 means no strength in that range reaches the target (always so with
    rho = 0, where the bias bound is zero; with a tiny |rho| the bound only gets there as cf -> 1),
    0.0 for rva means the zero-confounder confidence bound already reaches it. Calibrate cf_y
    and cf_d with dml_benchmark: "a confounder as strong as income" is a statement the reader can
    judge. q must be a finite real number > 0 (v2.5.3).
    v2.6.4: cf_y and cf_d are sensitivity parameters in [0, 1]. The values to pass are those dml_benchmark returns as
    cf_y and cf_d (the paper's gains, cut to [0, 1], in the parametrization of DoubleML), never its r2_y and r2_d; a
    ValueError outside [0, 1], a TypeError for a bool or a non-number (a 0-d array or a Decimal too, as for q; and for
    rho, a bool); before, cf_y = 1.5 was used as if valid, negatives gave nan and True read as 1. cf_d = 1 is accepted,
    as dml_benchmark can return it: the bound is then infinite and says nothing, except with cf_y = 0 or rho = 0, where
    it is zero by convention (the limit as cf_d -> 1; at cf_d = 1 itself the treatment has no residual variation). An
    infinite and a zero bias bound now also go to out["warnings"] (and warnings.warn), so they are not lost with
    show=False."""
    import numbers
    from specification import _check_q, _q_is_one
    q = _check_q(q)
    cf_y, cf_d = _check_cf(cf_y, "cf_y"), _check_cf(cf_d, "cf_d")
    if isinstance(rho, (bool, np.bool_)) or not isinstance(rho, numbers.Real):   # v2.6.4 review 2: True printed "rho = True"
        raise TypeError(f"rho must be a number in [-1, 1] (a correlation; 1 is the worst case), got {rho!r}")
    if not np.isfinite(rho):                  # NaN crashed brentq; inf printed RV = RVa = 0.00% (7c25644)
        raise ValueError(f"rho must be finite (got {rho!r})" + (
            "; a benchmark with cf_y = 0 or cf_d = 0 has an undefined rho and is no reference (see the warning of "
            "dml_benchmark)" if np.isnan(rho) else ""))
    if abs(rho) > 1:                          # v2.6.1: a correlation; |rho| > 1 printed bounds no confounder can reach
        raise ValueError(f"rho is a correlation and must be in [-1, 1], got {rho!r} (rho = 1 is the worst case; "
                         "dml_benchmark reports rho already clipped to [-1, 1])")
    if "residuals" not in res:
        raise ValueError("pass the dict returned by causal.dml_plr")
    z = stats.norm.ppf(1 - alpha / 2)
    reps = [_dml_sens_rep(uy, vd, cf_y, cf_d, rho) for uy, vd in res["residuals"]]
    med = {k: float(np.median([r_[k] for r_ in reps])) for k in reps[0]}
    th, S, B = res["estimate"], med["S"], med["bound"]
    lo, hi = th - B, th + B
    ci_lo, ci_hi = lo - z * med["se_lo"], hi + z * med["se_hi"]

    # Why the endpoints decide (v2.6.4, backlog of v2.5.3): in one sample split the confidence bound's SE is
    # ||psi_th − c·psi_S|| / √n, the norm of an affine function of c, hence convex in c; c·S is linear, so gap is convex
    # in c, and c = |rho|·eta²/√(1 − eta²) increases in eta. A convex gap below zero at both ends is below zero in
    # between ("nor does any"), and one that starts below zero and ends above has one root (brentq). With several splits
    # gap uses medians, which need not be convex; tests/test_v264.py checks the endpoint verdict and the root against a
    # dense grid on many designs.
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
    zero_bound = bool(np.isfinite(B) and B == 0)   # keyed on the bound itself (cf_d = 1 gives an infinite one)
    if zero_bound:                                 # v2.6.4: also with show=False (backlog of v2.5.3)
        out["warnings"].append("The bias bound is zero: a confounder with cf_y = 0, cf_d = 0 or rho = 0 changes "
                               "nothing, so these bounds say nothing about confounding (a benchmark with no strength is "
                               "no reference).")
    if np.isinf(B):                                # cf_d = 1 with cf_y > 0 and rho != 0 (v2.6.4 review: keyed on B)
        out["warnings"].append("cf_d = 100% leaves no residual variation in the treatment: the bias bound is infinite "
                               "and says nothing; for 'k times as strong as' a benchmark use dml_bounds, which refuses kd "
                               "at or above k_d_max.")
    for w_ in out["warnings"]:
        warnings.warn(w_, stacklevel=2)
    if show:
        one = _q_is_one(q)                        # q a rounding error from 1 reads as q = 1 (text only)
        target = 0.0 if one else (1 - q) * th   # v2.6.1: was 0.0 * th, printed "-0.0000" for th < 0
        level = f"{round(100 * (1 - alpha), 6):g}%"   # v2.5.3: was :.0f, which printed "100%" for alpha = 0.001
        print(f"\nDML-PLR sensitivity: theta = {th:.4f} (SE {res['se']:.4f}), S = {S:.4f}")
        print(f"  confounder with cf_y = {cf_y:.1%}, cf_d = {cf_d:.1%}, rho = {rho}: bias <= {B:.4f}")
        if zero_bound:                       # the same text as out["warnings"], printed where it always was
            print("  The bias bound is zero: a confounder with cf_y = 0, cf_d = 0 or rho = 0 changes nothing, so these "
                  "bounds say nothing about confounding (a benchmark with no strength is no reference).")
        if np.isinf(B):                      # v2.6.4: was a silent "bias <= inf"
            print("  " + out["warnings"][-1])
        print(f"  bounds [{lo:.4f}, {hi:.4f}]; {level} confidence bounds [{ci_lo:.4f}, {ci_hi:.4f}]")
        pct = _dml_pct
        if rv >= 1.0:                             # the ceiling of the search, cf up to 1 − 1e-9
            rv_txt = (f"RV = 100% (ceiling): no confounder with cf_y = cf_d short of 100% and rho = {rho} moves "
                      f"theta to {target:.4f}")
        else:
            rv_txt = (f"RV = {pct(rv, 2)}: a confounder explaining {pct(rv, 1)} of the residual variance of both D "
                      f"and Y would move theta to {target:.4f}")
        # "already" is judged on the CI the reader sees: theta ± z·SE of dml_plr at this alpha. That SE is the
        # median of sqrt(se_r² + (theta_r − theta)²) over the sample splits, never below the median per-split SE
        # inside rva, so this CI contains the zero-confounder confidence bounds up to rounding: it can contain the
        # target while rva is still > 0. The reverse (rva = 0, the reported CI excluding the target) happens only
        # by an ulp (review of v2.5.3: 30 of 200 seeds with n_rep = 1), so rva = 0 also reads as "already".
        z_rep = stats.norm.ppf(1 - alpha / 2)
        in_reported = (th - z_rep * res["se"] <= target <= th + z_rep * res["se"]) or rva == 0
        sig = f"{round(100 * alpha, 6):g}%"
        if in_reported and one:
            rva_txt = f"the estimate is already not significant at the {sig} level, with no confounder"
        elif in_reported:
            rva_txt = f"that value is already inside the {level} confidence interval, with no confounder"
        elif rva >= 1.0:
            rva_txt = ("nor does any make it no longer significant" if one
                       else f"nor does any put that value inside the {level} confidence bounds")
        elif one:                                 # the v2.5.2.1 sentence, byte for byte (strengths >= 0.05%)
            rva_txt = f"RVa = {pct(rva, 2)} makes it no longer significant"
        else:
            rva_txt = f"RVa = {pct(rva, 2)} would put that value inside the {level} confidence bounds"
        print(f"  {rv_txt}; {rva_txt}.")
        print("  Benchmark these strengths against observed covariates with causal.dml_benchmark.")
        print(_dml_report_line(rho, "'bounds [a, b]'", rho_txt=f"{rho}", target="" if one else f"{target:.4f}"))
    return out


def _split_mismatch(res: dict, rs: dict, n_long: int, n_short: int) -> str:
    """Why the sample splits of res and of a refit cannot be paired by position ('' if they can): v2.6.2 review."""
    if n_long != n_short:
        return f"res has {n_long} sample splits and the refit {n_short}"
    if len(res.get("thetas", ())) != n_long or len(rs.get("thetas", ())) != n_short:
        return "res has no per-split thetas"
    if "seed" not in res:
        return "res does not record its seed (a dml_plr result from before v2.6.2)"
    for key in ("seed", "n_folds"):
        if res.get(key) != rs.get(key):
            return f"res has {key} = {res.get(key)!r} and the refit {rs.get(key)!r}"
    return ""


def dml_benchmark(df: pd.DataFrame, y: str, d: str, X: list[str], benchmarks: list[str],
                  res: dict | None = None, **dml_kwargs) -> pd.DataFrame:
    """Implied sensitivity parameters of observed covariates (Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis 2022;
    Appendix F, eq. 22-23, of arXiv 2112.13398v5, Appendix E in v6). cf_y, cf_d and rho are those of DoubleML's
    sensitivity_benchmark; delta_theta and the NaN rho of a benchmark with no strength are the skill's (below).
    Refit dml_plr without each benchmark covariate X_j and measure the gains it brings, relative to the full model
    (subscript l, with X_j; s, without it), in each sample split r and then as the median over the splits:
      gain_y = median_r (sigma2_s,r - sigma2_l,r) / sigma2_l,r   (G_Y: added explained variance of the outcome over
                                                                  what the full model leaves unexplained)
      gain_d = median_r (nu2_l,r - nu2_s,r) / nu2_s,r            (G_D, nu2 = E[alpha^2] = 1/E[(D - m(X))^2])
      cf_y, cf_d = median_r of the split gains cut to [0, 1]     (DoubleML's cut; cf_d = 1 gives an infinite bound)
      rho  = median_r sign(dth_r) min(|dth_r| / sqrt((sigma2_s,r - sigma2_l,r)(nu2_l,r - nu2_s,r)), 1), with
             dth_r = theta_s,r - theta_l,r and +-1 where that denominator is 0 (DoubleML); NaN when gain_y or gain_d
             is 0 (no strength: the skill's rule since v2.5.3)
      delta_theta = theta_s - theta_l (medians over the splits, as before);  k_d_max = 1/gain_d
      r2_y, r2_d = the partial R² that cf_y and cf_d were before v2.6.2, computed as before (ratios of medians).
    The splits are paired by position, which is valid when res and the refits use the same seed, n_folds and n_rep
    (KFold depends only on the rows and the seed); dml_plr stores its seed since v2.6.2. When they differ, or when res
    has no per-split thetas or no seed, the gains fall back to ratios of medians, with a warning that says why.
    v2.6.2: cf_y and cf_d were the partial R² r2_y, r2_d. In each sample split a gain G > 0 is above the partial R²
    G/(1 + G), so "a confounder as strong as X_j" gave a smaller (less conservative) bound than the paper's benchmark;
    cf_d was also cut at 0.999 silently. The returned columns are different aggregations, though (v2.6.3): cf_y is the
    median of the split gains, r2_y a ratio of medians, so with several paired splits cf_y can be below r2_y (and cf_d
    below r2_d); with one split, or in the fallback to ratios of medians, cf = min(r2 / (1 - r2), 1) (up to rounding).
    For "a confounder k times as strong as X_j" use dml_bounds (the paper scales the gains by k). Benchmarks are
    informative only if the latent confounder is comparable to the observed ones; they also condition on the remaining
    covariates, and their own uncertainty is not propagated (as in the paper)."""
    res = res or dml_plr(df, y, d, X, **dml_kwargs)
    long = [_dml_sens_rep(uy, vd, 0, 0, 1) for uy, vd in res["residuals"]]
    s2l = np.median([r_["sigma2"] for r_ in long]); n2l = np.median([r_["nu2"] for r_ in long])
    rows = []
    for b in benchmarks:
        rs = dml_plr(df, y, d, [x for x in X if x != b], **dml_kwargs)
        sh = [_dml_sens_rep(uy, vd, 0, 0, 1) for uy, vd in rs["residuals"]]
        s2s = np.median([r_["sigma2"] for r_ in sh]); n2s = np.median([r_["nu2"] for r_ in sh])
        ry = max((s2s - s2l) / s2s, 0.0)            # partial R² forms (cf_y, cf_d before v2.6.2), as before
        rd = max((n2l - n2s) / n2l, 0.0)
        dth = rs["estimate"] - res["estimate"]
        why = _split_mismatch(res, rs, len(long), len(sh))
        if not why:
            gys = np.array([max(s_["sigma2"] / l_["sigma2"] - 1, 0.0) for s_, l_ in zip(sh, long)])
            gds = np.array([max(l_["nu2"] / s_["nu2"] - 1, 0.0) for s_, l_ in zip(sh, long)])
            dths = np.array(rs["thetas"]) - np.array(res["thetas"])
            den = np.array([np.sqrt(max(s_["sigma2"] - l_["sigma2"], 0.0) * max(l_["nu2"] - s_["nu2"], 0.0))
                            for s_, l_ in zip(sh, long)])
            with np.errstate(divide="ignore", invalid="ignore"):
                mag = np.where(den > 0, np.minimum(np.abs(dths) / den, 1.0), 1.0)
            rho_b = float(np.median(np.sign(dths) * mag))
        else:
            warnings.warn(f"Benchmark {b}: {why}, so the splits cannot be paired; the gains are ratios of the medians "
                          "over the splits, not medians of the split gains as in DoubleML (pass dml_benchmark the same "
                          "seed, n_folds and n_rep as dml_plr)", stacklevel=2)
            gys = np.array([max(s2s / s2l - 1, 0.0)]); gds = np.array([max(n2l / n2s - 1, 0.0)])
            Ss = np.median([r_["S"] for r_ in sh])
            denom = Ss * np.sqrt(ry * rd / (1 - rd)) if ry > 0 and 0 < rd < 1 else np.nan   # = sqrt(dsigma2 * dnu2)
            rho_b = float(np.clip(dth / denom, -1, 1)) if np.isfinite(denom) and denom > 0 else np.nan
        gy, gd = float(np.median(gys)), float(np.median(gds))
        if gy == 0 or gd == 0:
            rho_b = np.nan
        rows.append({"benchmark": b, "cf_y": float(np.median(np.minimum(gys, 1.0))),
                     "cf_d": float(np.median(np.minimum(gds, 1.0))), "rho": rho_b,
                     "theta_without": rs["estimate"], "delta_theta": dth, "gain_y": gy, "gain_d": gd,
                     "k_d_max": 1 / gd if gd > 0 else np.inf, "r2_y": ry, "r2_d": rd})
        if gy == 0 or gd == 0:                    # v2.5.3: this used to come back silently with rho = NaN
            side = ("of the outcome's and of the treatment's residual variance are" if gy == 0 and gd == 0
                    else "of the outcome's residual variance is" if gy == 0
                    else "of the treatment's residual variance is")
            warnings.warn(f"Benchmark {b}: cf_y = {min(gy, 1.0):.3g}, cf_d = {min(gd, 1.0):.3g}, rho undefined. Given the other "
                          f"covariates, its estimated share {side} zero or below (clipped to 0), so it is no reference "
                          "for how strong a confounder could be: its cf_y and cf_d give dml_sensitivity a bias bound "
                          "of zero, not a robustness result, and its NaN rho is refused. Use covariates that matter "
                          "for both D and Y as benchmarks.", stacklevel=2)
        cfy_, cfd_ = rows[-1]["cf_y"], rows[-1]["cf_d"]
        parts = []                                # v2.6.2 review: state the returned cf, not the uncut median
        if gy > 1:
            parts.append(f"gain_y = {gy:.3g} is above 1 (cf_y, the median of the split gains cut to 1, is {cfy_:.3g}); "
                         "dml_bounds holds ky * gain_y at 1")
        if gd >= 1:
            parts.append(f"gain_d = {gd:.3g} is 1 or more (cf_d = {cfd_:.3g}"
                         + (": the bound of dml_sensitivity is infinite" if cfd_ >= 1 else "") + "); dml_bounds refuses "
                         f"every kd at or above k_d_max = {1 / gd:.3g}")
        if parts:
            warnings.warn(f"Benchmark {b}: " + "; ".join(parts) + ".", stacklevel=2)
    return pd.DataFrame(rows).set_index("benchmark")


def dml_bounds(res: dict, bench: pd.DataFrame, kd=None, ky=None, rho="benchmark", alpha: float = ALPHA,
               show: bool = True) -> dict:
    """Bounds for DML-PLR under a confounder k times as strong as an observed covariate (v2.6.2): the analogue of
    specification.ovb_bounds, following Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis (2022; Appendix F, eq. 22-23,
    of arXiv 2112.13398v5, Appendix E in v6):
    the benchmark's gains are scaled by k, cf_y = ky * gain_y and cf_d = kd * gain_d.

    res    the dict of dml_plr (with its residuals); bench  the DataFrame of dml_benchmark (or any with columns
           gain_y, gain_d and rho, one row per benchmark).
    kd, ky how many times the benchmark's gain in the treatment (kd) and in the outcome (ky); a number or a sequence,
           paired as in ovb_bounds; ky = None means ky = kd; bound_label is "kx X" when kd = ky and "kd/ky x X"
           otherwise, the pattern of ovb_bounds ("2/5x X": kd = 2, ky = 5; not a ratio; ovb_bounds rounds the numbers
           to 2 decimals, as sensemakr, and dml_bounds does not), and the printed line spells it out;
           kd = None means (1, 2, 3) and skips, per benchmark,
           the values at or above k_d_max = 1/gain_d with a warning (a kd you give there stops the call).
    rho    "benchmark" (default): each benchmark's observed rho (the paper's benchmark for rho; a NaN rho is
           skipped, one outside [-1, 1] is a ValueError); or a number in [-1, 1] for every line (1: the worst case,
           the paper's conservative default).

    It scales gain_y and gain_d (the medians of the split gains), not cf_y and cf_d: at k = 1 the two agree unless the
    split gains straddle 1 with an even number of splits. Conventions of the skill (the paper states none): cf_y =
    ky * gain_y above 1 is held at 1 (the confounder would
    explain all of the outcome's residual variance), with one warning per benchmark; kd * gain_d >= 1 is impossible
    (cf_d = 1 gives an infinite bound). The bounds and confidence bounds are those of dml_sensitivity with that cf_y,
    cf_d and rho (influence functions, median over the sample splits); the gains and rho are estimates whose own
    uncertainty is not propagated (as in the paper). Read each line as a worst case under "k times as strong", not as
    a likely value.

    Returns {"bounds": DataFrame (bound_label, benchmark, kd, ky, cf_y, cf_d, cf_y_capped, rho, bias_bound,
    theta_lower, theta_upper, ci_low, ci_high, ci_excludes_zero, ci_verdict in {"excludes zero", "includes zero"}),
    "estimate", "se", "alpha", "rho", "warnings"}."""
    import numbers
    from specification import _check_k, _KD_DEFAULT, _level_pct
    if "residuals" not in res:
        raise ValueError("pass the dict returned by causal.dml_plr")
    if not isinstance(bench, pd.DataFrame) or not {"gain_y", "gain_d", "rho"} <= set(bench.columns):
        raise TypeError("bench must be the DataFrame of causal.dml_benchmark (columns gain_y, gain_d, rho)")
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    if isinstance(rho, str):
        if rho != "benchmark":
            raise ValueError(f"rho is 'benchmark' or a number in [-1, 1], got {rho!r}")
    elif isinstance(rho, (bool, np.bool_)) or not isinstance(rho, numbers.Real):
        raise TypeError(f"rho is 'benchmark' or one number in [-1, 1], got {rho!r}")
    elif not np.isfinite(float(rho)) or abs(float(rho)) > 1:
        raise ValueError(f"rho is a correlation and must be finite and in [-1, 1], got {rho!r} (1 is the worst case)")
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
                             f"and {len(ky_arr)}")
    warns: list[str] = []

    def note(msg):
        warns.append(msg)
        warnings.warn(msg, stacklevel=3)

    z = stats.norm.ppf(1 - alpha / 2)
    th, se = float(res["estimate"]), float(res["se"])
    rows = []
    for lab, b in bench.iterrows():
        gy, gd = float(b["gain_y"]), float(b["gain_d"])
        if not (np.isfinite(gy) and np.isfinite(gd)):
            note(f"{lab}: skipped, its gains are not finite (gain_y = {gy:.3g}, gain_d = {gd:.3g}): missing or broken "
                 "benchmark values, not a strength")
            continue
        if not (gy > 0 and gd > 0):
            note(f"{lab}: skipped, no strength (gain_y = {gy:.3g}, gain_d = {gd:.3g}): no multiple of it biases theta")
            continue
        rb = float(b["rho"]) if rho == "benchmark" else float(rho)
        if not np.isfinite(rb):
            note(f"{lab}: skipped, its rho is undefined (NaN); pass rho=1 (the worst case) to use it")
            continue
        if abs(rb) > 1:
            raise ValueError(f"benchmark {lab!r} has rho = {rb!r}, outside [-1, 1]: rho is a correlation (dml_benchmark "
                             "clips it); fix the bench table or pass a number")
        kmax = 1 / gd
        bad = (kd_arr >= kmax) | (kd_arr * gd >= 1)   # both: kd = fl(1/gd) can give a product just below 1
        kd_b, ky_b = kd_arr, ky_arr
        if bad.any():
            if not default_kd:
                raise ValueError(f"impossible kd = {kd_arr[bad][0]:g} for benchmark {lab!r}: its gain in the treatment "
                                 f"is {gd:.4g}, so kd * gain_d would reach 1 (no residual variation left in the "
                                 f"treatment); kd must be below k_d_max = {kmax:.6g}")
            note(f"{lab}: kd = {', '.join(f'{v:g}' for v in kd_arr[bad])} skipped (impossible: kd must be below "
                 f"k_d_max = {kmax:.6g})")
            kd_b, ky_b = kd_arr[~bad], ky_arr[~bad]
        capped_k = []
        for kdi, kyi in zip(kd_b, ky_b):
            cfy_raw, cfd = kyi * gy, kdi * gd
            cap = cfy_raw > 1
            cfy = 1.0 if cap else cfy_raw
            if cap:
                capped_k.append(f"{kyi:g}")
            reps = [_dml_sens_rep(uy, vd, cfy, cfd, rb) for uy, vd in res["residuals"]]
            med = {k_: float(np.median([r_[k_] for r_ in reps])) for k_ in ("bound", "se_lo", "se_hi")}
            lo, hi = th - med["bound"], th + med["bound"]
            ci_lo, ci_hi = lo - z * med["se_lo"], hi + z * med["se_hi"]
            excl = bool(ci_lo > 0 or ci_hi < 0)
            rows.append({"bound_label": (f"{kdi:g}x {lab}" if kdi == kyi else f"{kdi:g}/{kyi:g}x {lab}"),
                         "benchmark": lab, "kd": kdi, "ky": kyi, "cf_y": cfy, "cf_d": cfd, "cf_y_capped": bool(cap),
                         "rho": rb, "bias_bound": med["bound"], "theta_lower": lo, "theta_upper": hi,
                         "ci_low": ci_lo, "ci_high": ci_hi, "ci_excludes_zero": excl,
                         "ci_verdict": "excludes zero" if excl else "includes zero"})
        if capped_k:
            note(f"{lab}, ky = {', '.join(dict.fromkeys(capped_k))}: ky * gain_y above 1, capped at 1 (the confounder "
                 f"would explain all of the outcome's residual variance)")
    cols = ["bound_label", "benchmark", "kd", "ky", "cf_y", "cf_d", "cf_y_capped", "rho", "bias_bound", "theta_lower",
            "theta_upper", "ci_low", "ci_high", "ci_excludes_zero", "ci_verdict"]
    out = {"bounds": pd.DataFrame(rows, columns=cols), "estimate": th, "se": se, "alpha": alpha, "rho": rho,
           "warnings": warns}
    if show:
        level = _level_pct(alpha)
        print(f"\nDML-PLR bounds for a confounder k times as strong as observed covariates (Chernozhukov et al. 2022, "
              f"benchmarking appendix: the benchmark's gains in explaining the outcome and the treatment, scaled by "
              f"k): theta = {th:.4f} (SE {se:.4f}).")
        print("  rho = the benchmark's observed correlation of the omitted pieces" if rho == "benchmark" else
              f"  rho = {float(rho):g}" + (" (the worst case)" if abs(float(rho)) == 1 else "") + " for every line")
        for _, r in out["bounds"].iterrows():
            cap = " (capped)" if r["cf_y_capped"] else ""
            same = f"{r['kd']:g}" == f"{r['ky']:g}"   # review: kd 0.3, ky 0.1 * 3 print alike with :g
            kd_s, ky_s = ((repr(float(r["kd"])), repr(float(r["ky"]))) if same else (f"{r['kd']:g}", f"{r['ky']:g}"))
            kk = ("" if r["kd"] == r["ky"] else      # v2.6.3: "2/5x" could read as 0.4x
                  f"(kd = {kd_s} for the treatment, ky = {ky_s} for the outcome) ")
            print(f"  {r['bound_label']}: {kk}cf_y {r['cf_y']:.1%}{cap}, cf_d {r['cf_d']:.1%}, rho {r['rho']:.2f}: "
                  f"theta "
                  f"between {r['theta_lower']:.4f} and {r['theta_upper']:.4f}; {level} confidence bounds "
                  f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}], which "
                  f"{'exclude zero' if r['ci_excludes_zero'] else 'include zero'}.")
        if out["bounds"].empty:
            print("  No bound computed: every benchmark or k was skipped (see the warnings).")
        print("  Worst cases under 'k times as strong', not likely values; the gains and rho are estimates whose own "
              "uncertainty is not propagated.")
        if not out["bounds"].empty:
            print(_dml_report_line(rho, "'theta between a and b'",
                                   rho_txt="" if rho == "benchmark" else f"{float(rho):g}"))
        for w in warns:
            print("  WARNING: " + w)
    return out


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
    have similar curvature (its bias term uses the difference of second derivatives).
    The method string says why IK was used (v2.5.0.1: the fallback is no longer silent); rdd() relays it
    as a warning."""
    try:
        from rdrobust import rdbwselect
    except Exception as e:                           # absent, or broken by a dependency at import
        if isinstance(e, ImportError) and importlib.util.find_spec("rdrobust") is None:
            try:
                from setup_env import install_command
                how = install_command("rd")
            except Exception:
                how = "setup_env.py --install --extra rd"
            why = f"rdrobust is not installed: optional extra 'rd', {how}, with the user's permission"
        else:
            why = f"rdrobust failed to import: {type(e).__name__}: {e}"
        return ik_bandwidth(y, x, cutoff), f"Imbens-Kalyanaraman ({why})"
    try:
        yy, xx = np.asarray(y, float), np.asarray(x, float)
        ok = np.isfinite(yy) & np.isfinite(xx)
        bw = rdbwselect(yy[ok], xx[ok], c=cutoff, bwselect="mserd")
        return float(np.asarray(bw.bws)[0, 0]), "rdrobust mserd"
    except Exception as e:
        return ik_bandwidth(y, x, cutoff), f"Imbens-Kalyanaraman (rdrobust failed: {type(e).__name__}: {e})"


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
    rd_warns = []
    if bw_method.startswith("Imbens-Kalyanaraman ("):
        rd_warns.append(f"bandwidth by {bw_method}, not the rdrobust MSE-optimal one: report it as IK")
    out = _out("RD effect" + (" (fuzzy, LATE)" if fuzzy else " (sharp)"), est, se, len(d), bandwidth=h,
               bandwidth_method=bw_method, warnings=rd_warns,
               n_left=int((d["_above"] == 0).sum()), n_right=int((d["_above"] == 1).sum()), kernel=kernel, p=p)
    if first:
        out.update(first)
    if show:
        print(f"\nRD at {x} = {cutoff}: {out['estimate']:.4f} (SE {se:.4f}, 95% CI {out['ci_low']:.4f} to "
              f"{out['ci_high']:.4f}), h = {h:.4g} ({bw_method}), N = {out['n_left']} + {out['n_right']}"
              + (f", first-stage jump {first['jump_in_treatment']:.3f} (F = {first['F']:.1f})" if first else ""))
        print("  Conventional CI; for robust bias-corrected inference use rdrobust (Calonico, Cattaneo & Titiunik, 2014).")
        for w_ in rd_warns:
            print(f"  Warning: {w_}")
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
_DESIGN_LABELS = {
    "canonical": "canonical DiD (binary, absorbing, one adoption date, never-treated units)",
    "staggered": "staggered adoption (binary, absorbing, several adoption dates)",
    "heterogeneous_adoption": "heterogeneous adoption (all untreated at the start, one date, different doses)",
    "first_switch": "staggered first switch (non-binary or non-absorbing treatment; switchers compared with units that "
                    "have not switched yet)",
    "other": "other design (no unit to compare with the switchers: no heterogeneity-robust DiD estimator)",
}


def did_design(df: pd.DataFrame, unit: str, time: str, d: str, cluster: str | None = None,
               show: bool = True) -> dict:
    """First step of any difference-in-differences (v2.5.0): what kind of treatment is this, and which estimators
    fit it? Classifies the panel into the five designs of de Chaisemartin & D'Haultfœuille's guide "Which
    difference-in-differences estimator should I use?" (Credible Answers):

      canonical               binary, absorbing, every treated unit starts at the same date, never-treated units
      staggered               binary, absorbing, adoption at different dates
      heterogeneous_adoption  every unit untreated at the start, treated units switch on at one date with
                              different doses
      first_switch            first treatment changes at different dates, with a non-binary or non-absorbing
                              treatment (staggered adoption is its binary-absorbing special case)
      other                   none of the above: no switcher has a unit with the same starting treatment that has
                              not switched yet (e.g. the treatment changes for every unit every period); the guide
                              lists no heterogeneity-robust estimator for these

    d: the treatment of each unit in each period (binary or a dose), not a first-treated date: gvar is an output.
    Definitions used here (the guide does not define them): a stayer is a unit whose treatment never changes in the
    panel; a switcher changes at least once; its first switch is the first period whose treatment differs from the
    unit's first observed one; a switcher is comparable when some unit with the same starting treatment has not
    switched yet at that date (a stayer or a later switcher). Reversal: a binary treatment going from 1 back to 0.

    Returns: design and label; checks (criterion, value, what it implies); per-unit table (start, number of changes,
    first switch, reversals); cohorts (first-switch date, units, pre-periods in the panel); estimators of the skill
    and outside it; the suggested comparison group (the guide recommends not-yet-treated controls when they exist);
    gvar (first treated period, 0 = never treated, indexed by unit; map it with df[unit].map(res["gvar"])) only for a
    binary absorbing treatment, else None; warnings; ask: the questions about the date (exact? announced in advance?
    gradual?) and spillovers to put to the user for a binary absorbing design, printed as "ASK THE USER" lines (v2.5.1;
    the rule existed in SKILL.md §4 but was skipped in a real session). With 1 to 3 treated units it warns that unit
    bootstrap and permutation are not valid and points to did_single_treated. Nothing is imputed: rows with a missing d are counted and left
    out."""
    cols = [unit, time, d] + ([cluster] if cluster else [])
    x = df[cols].copy()
    n_missing = int(x[d].isna().sum())
    x = x.dropna(subset=[d]).sort_values([unit, time])
    if x.empty:
        raise ValueError("did_design: the treatment column has no observed values")
    if x.duplicated([unit, time]).any():
        raise ValueError(f"did_design: repeated ({unit}, {time}) rows; one row per unit and period is needed")
    vals = np.unique(x[d].to_numpy(float))
    binary = bool(np.isin(vals, [0.0, 1.0]).all())
    scaled = False
    if not binary and len(vals) == 2 and vals[0] == 0.0:
        # two values {0, c}: binary up to scale (v2.5.0 review); classified on d != 0
        x[d] = (x[d] != 0).astype(float)
        binary, scaled, c_val = True, True, float(vals[1])
    periods = np.sort(x[time].unique())
    n_per = x.groupby(unit)[time].size()
    balanced = bool((n_per == len(periods)).all())
    t_num = pd.api.types.is_numeric_dtype(x[time])
    regular = bool(t_num and np.allclose(np.diff(periods.astype(float)), 1.0)) if len(periods) > 1 else True
    # per-unit trajectory
    prev = x.groupby(unit)[d].shift()
    chg = prev.notna() & (x[d] != prev)
    x["_chg"] = chg
    x["_rev"] = chg & (prev == 1) & (x[d] == 0) if binary else False
    g_by = x.groupby(unit)
    start = g_by[d].first()
    first_t = g_by[time].first()
    n_chg = g_by["_chg"].sum().astype(int)
    n_rev = g_by["_rev"].sum().astype(int) if binary else pd.Series(0, index=start.index)
    first_switch = x[x["_chg"]].groupby(unit)[time].first().reindex(start.index)
    units = pd.DataFrame({"start": start, "first_period": first_t, "changes": n_chg, "first_switch": first_switch,
                          "reversals": n_rev})
    stayer = units["changes"] == 0
    switchers = units[~stayer]
    absorbing = bool(binary and (units["reversals"] == 0).all())
    if binary:
        never = stayer & (units["start"] == 0)
        always = units["start"] == 1
    else:
        never = stayer & (units["start"] == 0)
        always = pd.Series(False, index=units.index)
    late_entry = always & (units["first_period"] > periods[0])
    # comparable switchers: a unit with the same start that has not switched yet at the switch date
    observed = set(zip(x[unit], x[time]))
    comp = pd.Series(False, index=switchers.index)
    for u, row in switchers.iterrows():
        f = row["first_switch"]
        same = units[(units["start"] == row["start"]) & (units.index != u)]
        later = same[same["first_switch"].isna() | (same["first_switch"] > f)]
        comp[u] = any((v, f) in observed for v in later.index)     # observed at the switch date
    share_comp = float(comp.mean()) if len(comp) else np.nan
    dates = np.sort(switchers["first_switch"].unique())
    all_untreated_start = bool((units["start"] == 0).all())
    # design
    if len(switchers) == 0:
        design = "other"
    elif binary and absorbing:
        if len(dates) == 1:
            design = "canonical" if never.any() else "other"
        else:
            design = "staggered" if share_comp > 0 else "other"
    elif not binary and all_untreated_start and len(dates) == 1:
        design = "heterogeneous_adoption"
    else:
        design = "first_switch" if share_comp > 0 else "other"
    # cohorts (first-switch dates) and pre-periods
    coh = switchers.groupby("first_switch").size().rename("units").to_frame()
    coh["pre_periods"] = [int((periods < t).sum()) for t in coh.index]
    coh.index.name = "first_switch"
    # clusters
    cl = {}
    if cluster:
        cl_treated = x.groupby(cluster)[d].apply(lambda s: bool((s != 0).any()))
        varies = x.groupby([cluster, time])[d].nunique()
        cl = {"clusters": int(x[cluster].nunique()), "treated_clusters": int(cl_treated.sum()),
              "cluster_periods_with_mixed_treatment": int((varies > 1).sum())}
    # estimators and comparison group
    # the guide recommends not-yet-treated controls (more transparent than imputation, slightly less noisy)
    control = {"canonical": "never", "staggered": "notyet"}.get(design)
    skill, external = [], []
    if design == "canonical":
        skill = ["causal.att_gt(control='never')", "causal.did_imputation", "causal.event_study",
                 "TWFE (acceptable without covariates; with covariates use att_gt(covariates=...))"]
    elif design == "staggered":
        skill = [f"causal.att_gt(control='{control}')", "causal.did_imputation", "causal.did_compare",
                 "causal.did_timing_sensitivity", "for an existing TWFE: causal.bacon_decomposition and "
                 "causal.twfe_weights"]
        external = ["R/Stata csdid, did_imputation, did_multiplegt_dyn"]
    elif design == "heterogeneous_adoption":
        skill = ["for an existing TWFE: causal.twfe_weights (non-binary treatment)"]
        external = ["did_had (R, Stata; Python version listed by the guide, not checked)"]
    elif design == "first_switch":
        skill = ["for an existing TWFE: causal.twfe_weights"]
        external = ["did_multiplegt_dyn (R, Stata; Python version listed by the guide, not checked)"]
        if binary:
            external.append("if lagged effects are ruled out: did_multiplegt_stat, or imputation estimators "
                            "(R/Stata did_imputation, did2s, fect), which the guide says are valid only for a "
                            "binary treatment")
    else:
        skill = ["for an existing TWFE: causal.twfe_weights (how much of it rests on negative weights)"]
    gvar = None
    if binary and absorbing and design != "other":
        gvar = pd.Series(np.where(units["start"] == 1, units["first_period"],
                                  units["first_switch"].fillna(0)), index=units.index, name="gvar")
    # warnings
    w = []
    if scaled:
        w.append(f"The treatment takes two values, 0 and {c_val:g}: classified as binary. Rescale it to 0/1 before "
                 "estimating, or say that effects are per unit of that value.")
    if len(switchers) == 0:
        w.append("No unit changes treatment in the panel: there is no variation over time to estimate a DiD from.")
    prev_obs = x.groupby(unit)[time].shift()
    sw_rows = x[x["_chg"]].groupby(unit).head(1)
    if regular and len(sw_rows):
        gap = (sw_rows[time].astype(float) - prev_obs.loc[sw_rows.index].astype(float)) > 1
        if gap.any():
            w.append(f"{int(gap.sum())} unit(s) switch across a gap in the panel (periods missing right before the "
                     "first switch): the switch date is only known to lie within the gap; consider "
                     "did_uncertain_dates.")
    if n_missing:
        w.append(f"{n_missing} rows have a missing treatment and were left out; check whether they are missing at "
                 "random before estimating.")
    if binary and not absorbing:
        nr = int((units["reversals"] > 0).sum())
        w.append(f"The treatment is reversible: {nr} unit(s) go from 1 back to 0 ({int(units['reversals'].sum())} "
                 "reversals). It must not enter att_gt, did_imputation, bacon_decomposition or "
                 "randomization_inference as a first-treated date (gvar): those assume that treatment, once on, "
                 "stays on. No gvar was built.")
    if not binary:
        w.append(f"The treatment is not binary ({len(vals)} distinct values): the skill's DiD estimators take a "
                 "binary absorbing treatment; none of them may be run on a dichotomized dose without saying that "
                 "the estimand changed.")
    if design == "other" and len(switchers):
        w.append("No heterogeneity-robust DiD estimator applies to this design: a TWFE coefficient here is not a "
                 "robust causal effect, and must not be reported as one. causal.twfe_weights shows how much of it "
                 "rests on negative weights.")
    if design == "first_switch" and share_comp < 1:
        w.append(f"{1 - share_comp:.0%} of the switchers have no unit with the same starting treatment that has not "
                 "switched yet: robust estimators drop them.")
    if always.any():
        w.append(f"{int(always.sum())} unit(s) are treated from their first observed period (no pre-period): they "
                 "cannot identify an effect and serve as controls only in a TWFE, where they are forbidden "
                 "comparisons." + (f" {int(late_entry.sum())} of them enter the panel late already treated: check "
                                   "whether earlier periods are missing." if late_entry.any() else ""))
    if design == "staggered" and not never.any():
        w.append("No never-treated units: use control='notyet'; the last cohort has no comparison group after its "
                 "own date, so its post-period cells are not estimated.")
    if design in ("canonical", "staggered") and len(coh) and (coh["pre_periods"] == 0).any():
        w.append("Some cohorts have no pre-period inside the panel and will be dropped by att_gt.")
    if not balanced:
        w.append(f"Unbalanced panel ({int((n_per < len(periods)).sum())} units miss periods): att_gt and "
                 "bacon_decomposition need a balanced panel; did_imputation does not.")
    if not regular:
        w.append(f"The {time} column is not a sequence of consecutive integers: att_gt uses g − 1 as the base period;"
                 " recode time as 1, 2, 3, … first.")
    if cluster and cl["cluster_periods_with_mixed_treatment"]:
        w.append(f"The treatment varies inside {cl['cluster_periods_with_mixed_treatment']} {cluster}-period cells: it"
                 f" is assigned below the {cluster} level. Cluster at the level of assignment.")
    from math import comb
    n_sw = len(switchers)
    if binary and absorbing and 0 < n_sw <= 3 and design in ("canonical", "staggered"):
        w.append(f"Only {n_sw} treated unit(s): the unit bootstrap of att_gt and the permutation of "
                 f"randomization_inference cannot give a valid p-value (smallest attainable p with "
                 f"{len(units)} units: 1/{comb(len(units), n_sw)}). Use causal.did_single_treated (inference "
                 "from the time series of the treated−control gap) or, with many control units, causal.synth. Do "
                 "not report HAC, Newey-West or Prais-Winsten p-values on the gap: with ~24 periods they reject "
                 "about three times too often.")
        skill.insert(0, "causal.did_single_treated (few treated units: inference from the gap series)")
    if cluster and (cl["clusters"] < 30 or cl["treated_clusters"] < 6):
        w.append(f"{cl['clusters']} clusters ({cl['treated_clusters']} treated): clustered and bootstrap SEs are "
                 "unreliable; plan causal.randomization_inference.")
    checks = pd.DataFrame([
        ("binary treatment", binary, f"values: {', '.join(f'{v:g}' for v in vals[:6])}" + (" …" if len(vals) > 6 else "")),
        ("absorbing (no 1 → 0)", absorbing if binary else None,
         "" if not binary else f"{int(units['reversals'].sum())} reversals in {int((units['reversals'] > 0).sum())} units"),
        ("switchers / stayers", f"{len(switchers)} / {int(stayer.sum())}", ""),
        ("first-switch dates", len(dates), ", ".join(str(t) for t in dates[:8]) + (" …" if len(dates) > 8 else "")),
        ("never-treated units", int(never.sum()), ""),
        ("treated from the first period", int(always.sum()), "no pre-period"),
        ("switchers with a not-yet-switched comparison", f"{share_comp:.0%}" if len(comp) else "—",
         "same starting treatment, not switched at that date"),
        ("all untreated at the start", all_untreated_start, ""),
        ("balanced panel", balanced, f"{len(units)} units × {len(periods)} periods"),
        ("consecutive integer periods", regular, ""),
    ] + [(k.replace("_", " "), v, "") for k, v in cl.items()], columns=["criterion", "value", "note"])
    ask = []
    if binary and absorbing and design in ("canonical", "staggered"):
        ask = ["Is the adoption date exact, and what is its source?",
               "Was the treatment announced, or expected, before it started (anticipation)?",
               "Did it start gradually rather than on one date?",
               "Could it have affected the control units too (spillovers, displaced customers or activity)?"]
    out = {"design": design, "label": _DESIGN_LABELS[design], "binary": binary, "absorbing": absorbing,
           "checks": checks, "units": units, "cohorts": coh, "never_treated": int(never.sum()),
           "always_treated": int(always.sum()), "share_comparable": share_comp, "balanced": balanced,
           "control": control, "estimators": {"skill": skill, "external": external}, "gvar": gvar,
           "n_units": len(units), "n_periods": len(periods), "missing_treatment": n_missing, "warnings": w, "ask": ask,
           **({"clusters": cl} if cluster else {})}
    if show:
        print(f"\nDiD design: {out['label']}")
        print(checks.to_string(index=False))
        if len(coh):
            print("Cohorts (first switch):\n" + coh.to_string())
        print("Estimators in the skill: " + ("; ".join(skill) if skill else "none"))
        if external:
            print("Outside the skill: " + "; ".join(external))
        if control:
            print(f"Comparison group: control='{control}'")
        for m in w:
            print("  WARNING: " + m)
        for m in ask:
            print("  ASK THE USER: " + m)
    return out


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


def _cs_engine(df: pd.DataFrame, unit: str, time: str, y: str, gvar: str, control: str, anticipation: int,
               covs: list[str], est_method: str, x_periods: str, cluster: str | None) -> dict:
    """Data preparation and the cell-by-cell Callaway-Sant'Anna computation of att_gt, separated (v2.5.0) so
    that did_compare can re-estimate CS on the same bootstrap draws as the imputation estimator. Returns the
    balanced outcome matrix, cohorts, cluster members and compute(idx, methods, record), where idx are row
    positions of units (repeated when a unit or cluster is drawn more than once)."""
    d = df[[unit, time, y, gvar, *covs] + ([cluster] if cluster and cluster not in (unit, time, y, gvar, *covs)
                                           else [])].copy()
    d[gvar] = d[gvar].fillna(0)
    Y = d.pivot(index=unit, columns=time, values=y)
    if Y.isna().any().any():
        raise ValueError("att_gt needs a balanced panel (drop units with missing periods)")
    G = d.groupby(unit)[gvar].first().reindex(Y.index).to_numpy()
    eng = {"units": Y.index, "cluster": cluster}
    if cluster:
        if (d.groupby(unit)[cluster].nunique() > 1).any():
            raise ValueError(f"att_gt: cluster '{cluster}' must be constant within each {unit}")
        cl_codes = pd.factorize(d.groupby(unit)[cluster].first().reindex(Y.index))[0]
        eng["cl_members"] = [np.where(cl_codes == c)[0] for c in range(cl_codes.max() + 1)]
        eng["n_cl"] = len(eng["cl_members"])
        eng["n_cl_treated"] = len({cl_codes[i] for i in np.where(G > 0)[0]})
    T = np.array(Y.columns)
    Ym = Y.to_numpy()
    Xa = None
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

    eng.update(T=T, Ym=Ym, G=G, cohorts=cohorts, dropped=dropped, dlt=dlt, diag=diag, compute=compute)
    return eng


def _cs_draw(rng, eng: dict) -> np.ndarray:
    """One bootstrap draw of unit positions: whole clusters when the engine has them, else units."""
    if eng["cluster"]:
        n_cl = eng["n_cl"]
        draw = rng.integers(0, n_cl, n_cl)
        return np.concatenate([eng["cl_members"][c] for c in draw])
    n = len(eng["Ym"])
    return rng.integers(0, n, n)


def _wald_boot(b: np.ndarray, draws: np.ndarray, max_rank: int, n_obs: int | None = None):
    """Wald test of b = 0 with the covariance of bootstrap draws (rows = draws). The rank is computed on the
    correlation scale (so cells with small SEs do not look degenerate) and capped at max_rank (B − 1 draws, G − 1
    clusters); below full rank the pseudo-inverse on the effective rank is used, with df = rank.
    Reference distribution (v2.5.0): with n_obs (the smaller of the resampled units, or clusters, and the draws),
    W·(n − q)/(q·(n − 1)) ~ F(q, n − q), the small-sample form of a Wald test with an estimated covariance; the χ²(q)
    over-rejected in the skill's simulations (5% nominal: 12% with 100 units and 9 restrictions, 8% with 200, 5.7%
    with 800). Without n_obs, or with n ≤ q, χ²(q). Returns (W, df, p, rank, p_chi2, ref)."""
    V = np.cov(draws, rowvar=False, ddof=1).reshape(len(b), len(b))
    sd = np.sqrt(np.clip(np.diag(V), 0, None))
    if not np.all(np.isfinite(V)) or np.any(sd <= 0):
        return np.nan, 0, np.nan, 0, np.nan, None
    C = V / np.outer(sd, sd)
    w, U = np.linalg.eigh(C)
    keep = w > max(w.max(), 0) * len(b) * 1e-10
    rank = int(min(keep.sum(), max_rank))
    if rank < 1:
        return np.nan, 0, np.nan, 0, np.nan, None
    order = np.argsort(w)[::-1][:rank]
    z = b / sd
    proj = U[:, order].T @ z
    W = float(np.sum(proj ** 2 / w[order]))
    p_chi2 = float(stats.chi2.sf(W, rank))
    if n_obs is not None and n_obs > rank + 1:
        p = float(stats.f.sf(W * (n_obs - rank) / (rank * (n_obs - 1)), rank, n_obs - rank))
        return W, rank, p, rank, p_chi2, f"F({rank}, {n_obs - rank})"
    return W, rank, p_chi2, rank, p_chi2, f"chi2({rank})"


def att_gt(df: pd.DataFrame, unit: str, time: str, y: str, gvar: str, control: str = "never",
           n_boot: int = 199, seed: int = 0, anticipation: int = 0, covariates: list[str] | None = None,
           est_method: str = "dr", x_periods: str = "base", verify: bool = True, cluster: str | None = None,
           alpha: float = ALPHA, show: bool = True) -> dict:
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
    computes the other two estimators on the full sample (an independent recomputation, stored as
    res["cross_check"], a pandas Series indexed dr/reg/ipw that reporting.cross_check takes as is): if
    they differ from the chosen one by more than one SE, the outcome and propensity models
    disagree, a sign that at least one is misspecified (warned). Covariate effects that differ
    ACROSS GROUPS (not only over time) bias every estimator here; check with ccc_test.

    Inference: bootstrap resampling units with replacement and re-estimating every nuisance
    model; n_boot=0 skips it. cluster = a column constant within unit (the state when a law
    varies by state, with municipalities as units): the bootstrap then resamples whole
    clusters, so the SEs allow any correlation within them, while the estimand (an average
    over units) is unchanged. Do not aggregate the data to the cluster level to get clustered
    inference: that changes the estimand (unit average -> cluster average). With fewer than
    ~30 clusters or ~6 treated clusters, the cluster bootstrap is unreliable (warned): add
    randomization_inference.
    Warns when either of the two pre-period cells right before the (anticipation-adjusted)
    date is significant (Bonferroni, alpha/2 each): the effect starts before the recorded date (anticipation or a misdated
    treatment) → check the date and run did_timing_sensitivity.

    Joint pre-trend test (v2.5.0; res["pretrend"], res["pretrend_p"], res["pretrend_df"]): Wald test that the
    pre-period effects are zero (cells with t < g − anticipation, short t−1 base), with the covariance of the
    bootstrap draws (those in which every pre cell exists) and an F(q, n − q) reference, n = the smaller of the
    resampled units (clusters with cluster=) and the draws; the χ²(q) p-value is kept as pretrend["p_chi2"]. A
    covariance estimated from n units supports few restrictions, so the test uses the most detailed of three levels
    with q ≤ (n − 1)/10: every pre cell; the pre-period event-time averages (e < −anticipation); their mean (one
    restriction). A coarser level checks less (a pre-trend that averages out passes) and is warned. Below full rank
    the pseudo-inverse on the effective rank is used and df = rank. In the skill's simulations (5% test): 9 cells and
    300 units, 7.5%; 30 to 80 clusters, 5-7%; 8 clusters with the one-restriction level, about 9% (460 panels: the
    pairs-cluster bootstrap SE is too small with so few clusters); the χ² with 9 cells and 100 units, 13%. Not
    rejecting does not show parallel trends (low power; Roth 2022).
    alpha (v2.5.0): level of the CIs, of the joint pre-trend test and of the early-effect warning."""
    covs = list(covariates) if covariates else []
    if est_method not in ("dr", "reg", "ipw"):
        raise ValueError("est_method must be 'dr', 'reg' or 'ipw'")
    if x_periods not in ("base", "both"):
        raise ValueError("x_periods must be 'base' or 'both'")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    eng = _cs_engine(df, unit, time, y, gvar, control, anticipation, covs, est_method, x_periods, cluster)
    Ym, G, cohorts, dropped, dlt, diag, compute = (eng[k] for k in ("Ym", "G", "cohorts", "dropped", "dlt", "diag",
                                                                    "compute"))
    n_cl = eng.get("n_cl")
    n_cl_treated = eng.get("n_cl_treated")
    if not cohorts:
        raise ValueError("att_gt: no cohort has the base period g-anticipation-1 inside the panel")
    all_idx = np.arange(len(Ym))
    if covs:
        meths = ("dr", "reg", "ipw") if verify else (est_method,)
        full = compute(all_idx, meths, record=True)
        base = full[["g", "t", "e", est_method, "n_g"]].rename(columns={est_method: "att"}) if verify else full
    else:
        base = compute(all_idx)
    if base.empty:
        raise ValueError("att_gt: no group-time cell has a comparison group"
                         + (" (no never-treated units: use control='notyet', or check the design with causal.did_design)"
                            if control == "never" else " (no unit is untreated at the time each cohort is compared)"))
    post = base[base["e"] >= 0]
    simple = np.average(post["att"], weights=post["n_g"]) if len(post) else np.nan
    dyn = base.groupby("e").apply(lambda s: np.average(s["att"], weights=s["n_g"]), include_groups=False)
    bs_dyn, bs_cells = [], []
    n_empty = 0
    if n_boot and n_boot > 1:
        rng = np.random.default_rng(seed)
        bs_simple = []
        n_empty = 0
        for _ in range(n_boot):
            b = compute(_cs_draw(rng, eng))
            if b.empty:
                # with few clusters a draw can hold no treated cluster (or no control), so no cell is defined: the
                # draw counts as missing, as np.nanstd already did for the ATT (a guard; no crash was observed on main)
                n_empty += 1
                bs_simple.append(np.nan)
                bs_dyn.append(pd.Series(dtype=float))
                bs_cells.append(pd.Series(dtype=float, index=pd.MultiIndex.from_tuples([], names=["g", "t"])))
                continue
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
    dyn_tab["ci_low"], dyn_tab["ci_high"] = _ci(dyn_tab["att"], dyn_tab["se"], alpha)
    gt = base.set_index(["g", "t"]).join(cell_se.rename("se"))
    # Jump right before the (anticipation-adjusted) date: with the short t-1 base, the cell
    # e = -δ-1 measures the change from g-δ-2 to g-δ-1, which is ~0 if the date is right.
    # Check the two cells right before it (a date one or two periods too late), each at alpha/2.
    zc = stats.norm.ppf(1 - alpha / 4)
    hits = [k for k in (-dlt - 1, -dlt - 2) if k in dyn_tab.index and np.isfinite(dyn_tab.loc[k, "se"])
            and dyn_tab.loc[k, "se"] > 0 and abs(dyn_tab.loc[k, "att"] / dyn_tab.loc[k, "se"]) > zc]
    flag = bool(hits)
    k_chk = hits[0] if hits else -dlt - 1
    # joint pre-trend test (v2.5.0). Three levels, from the most to the least detailed, so that the number of
    # restrictions q stays within (n - 1)/10, n = min(resampled units or clusters, bootstrap draws): a Wald test with
    # a covariance estimated from n units over-rejects badly when q is close to n (8 clusters, 8 restrictions: 83%
    # rejections for a 5% test in the skill's simulations; one restriction: 6%).
    pre = base[base["t"] < base["g"] - dlt].set_index(["g", "t"])["att"].dropna()
    pt = {"stat": np.nan, "df": 0, "p": np.nan, "p_chi2": np.nan, "reference": None, "rank": 0,
          "n_cells": int(len(pre)), "test": None, "draws_used": 0, "max_rank": 0}
    if len(pre) and bs_cells:
        n_res = n_cl if eng["cluster"] else len(Ym)          # resampled units: the n of the F reference
        e_pre = [e for e in dyn.index if e < -dlt]
        Dc = pd.concat(bs_cells, axis=1).reindex(pre.index).T
        Dc = Dc[Dc.notna().all(axis=1)].to_numpy(float)
        De = pd.concat(bs_dyn, axis=1).reindex(e_pre).T
        De = De[De.notna().all(axis=1)].to_numpy(float)
        levels = [("pre-period cells", pre.to_numpy(float), Dc),
                  ("pre-period event-time averages", dyn.reindex(e_pre).to_numpy(float), De),
                  ("average pre-period coefficient", np.array([float(np.mean(dyn.reindex(e_pre)))]),
                   De.mean(axis=1, keepdims=True))]
        for name, est_v, draws_v in levels:
            n_eff = min(n_res, len(draws_v))
            if len(est_v) <= max(1, (n_eff - 1) / 10) or name == levels[-1][0]:
                break
        cap = min(len(draws_v) - 1, n_res - 1)
        if len(draws_v) >= 2 and len(est_v) <= cap:
            W, df_, p_, rk, pc, ref = _wald_boot(est_v, draws_v, cap, n_eff)
            pt.update(stat=W, df=df_, p=p_, p_chi2=pc, reference=ref, rank=rk, test=name,
                      draws_used=len(draws_v), max_rank=cap, n_restrictions=len(est_v), n_event_times=len(e_pre))
        pt["n_resampled"] = n_res
        pt["n_effective"] = min(n_res, pt["draws_used"]) if pt["draws_used"] else 0
    method = "no covariates" if not covs else {"dr": "doubly robust", "reg": "outcome regression",
                                               "ipw": "IPW"}[est_method] + f", X: {', '.join(covs)} ({x_periods})"
    warns = []
    if n_boot and n_boot > 1 and n_empty:
        warns.append(f"{n_empty} of {n_boot} bootstrap draws had no treated {'cluster' if cluster else 'unit'} or no "
                     "control and were left out of the SEs: with this few treated clusters the bootstrap is unreliable;"
                     " use causal.randomization_inference.")
    n_tr_units = int((G > 0).sum())
    if not cluster and 0 < n_tr_units <= 3:
        warns.append(f"Only {n_tr_units} treated unit(s): the unit bootstrap cannot give a valid SE, CI or pre-trend "
                     "test with so few treated units. Use causal.did_single_treated (inference from the time series of "
                     "the treated−control gap) or, with many control units, causal.synth.")
    if cluster and n_boot and n_boot > 1 and (n_cl < 30 or n_cl_treated < 6):
        warns.append(f"Cluster bootstrap with {n_cl} clusters ({n_cl_treated} treated): pairs-cluster bootstrap SEs are "
                     "unreliable with few (treated) clusters. Report causal.randomization_inference (and a wild cluster "
                     "bootstrap of the TWFE event study) next to this CI.")
    if flag:
        warns.append(f"The outcome already moves at e = {k_chk} (|z| > {zc:.2f}): the effect starts before the recorded"
                     " date (anticipation or a misdated treatment). Confirm the date; re-run with anticipation="
                     f"{-k_chk} and did_timing_sensitivity before reporting the ATT.")
    if pt["test"] and pt["test"] != "pre-period cells":
        warns.append(f"Joint pre-trend test run on the {pt['test']} ({pt['n_restrictions']} restriction"
                     f"{'s' if pt['n_restrictions'] > 1 else ''}), not on the {pt['n_cells']} pre-period cells: with "
                     f"{pt['n_effective']} resampled {'clusters' if cluster else 'units'} (or draws) a covariance of more "
                     "restrictions is too noisy for a Wald test. It checks less (a pre-trend that averages out passes).")
    if pt["test"] and pt["rank"] < pt["n_restrictions"]:
        warns.append(f"Pre-trend covariance below full rank: Wald test on rank {pt['rank']} (pseudo-inverse), df = "
                     f"{pt['df']}.")
    if pt["test"] and np.isfinite(pt["p"]) and pt["p"] < alpha:
        warns.append(f"Joint pre-trend test rejects (p = {pt['p']:.3g}, df = {pt['df']}): parallel trends are doubtful."
                     " Look at the pre-period cells, compare with did_imputation (did_compare) and consider anticipation"
                     " or covariates.")
    if pt["test"] and pt["df"] and pt["n_effective"] < 40 * pt["df"] and pt["test"] == "pre-period cells":
        warns.append(f"Joint pre-trend test with {pt['df']} restrictions, {pt['n_resampled']} resampled "
                     f"{'clusters' if cluster else 'units'} and {pt['draws_used']} bootstrap draws: a Wald test with an "
                     f"estimated covariance can over-reject at this size even with the {pt['reference']} reference; read a"
                     " rejection near the threshold with care"
                     + (", raise n_boot (>= 999)" if pt["draws_used"] < 40 * pt["df"] else "")
                     + " and check with causal.randomization_inference.")
    if pt["test"] and cluster and (n_cl < 30 or n_cl_treated < 6):
        warns.append("With few (treated) clusters the bootstrap pre-trend test over-rejects (about 9% for a 5% test "
                     "with 8 clusters in the skill's simulations): read a rejection near the threshold with care and "
                     "check the pre-period cells with causal.randomization_inference.")
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
    out = {"overall": _out("ATT (CS, simple)", simple, se_simple, len(Ym), alpha=alpha, anticipation=dlt,
                           method=method, warnings=warns, cluster=cluster,
                           n_clusters=n_cl if cluster else len(Ym)),
           "dynamic": dyn_tab, "gt": gt, "flags": {"effect_before_date": flag},
           "cohorts_dropped": dropped, "warnings": warns, "pretrend": pt, "pretrend_p": pt["p"],
           "pretrend_df": pt["df"], "alpha": alpha}
    if covs:
        out["cross_check"] = cross
        out["overlap"] = {k: diag[k] for k in ("ps_max", "trimmed", "ps_failed", "cells")}
    if show:
        o = out["overall"]
        print(f"\nCallaway-Sant'Anna ({control} controls, {len(cohorts)} cohorts, {n_boot} bootstrap reps"
              + (f" by {cluster} ({n_cl} clusters, {n_cl_treated} treated)" if cluster else "")
              + (f", anticipation = {dlt}" if dlt else "") + f"; {method})")
        print(f"Overall ATT: {o['estimate']:.4f} (SE {o['se']:.4f}, {100 * (1 - alpha):g}% CI {o['ci_low']:.4f} to "
              f"{o['ci_high']:.4f})")
        print("Dynamic (event-time) ATT:\n" + dyn_tab.to_string(float_format=lambda v: f"{v:.4f}"))
        if pt["test"]:
            print(f"Joint pre-trend test ({pt['test']}, bootstrap covariance): W = {pt['stat']:.3f}, "
                  f"{pt['reference']} p = {pt['p']:.3f} → " + ("pre-trends detected: parallel trends doubtful" if pt["p"] < alpha else
                                            "does not reject; this does not show parallel trends (low power is possible)"))
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
                           est_method: str = "dr", x_periods: str = "base", cluster: str | None = None,
                           alpha: float = ALPHA, show: bool = True) -> dict:
    """How much does the Callaway-Sant'Anna result depend on the exact treatment date?

    1. Shift table: re-estimates the overall ATT with every treated unit's date moved by
       k = -max_shift … +max_shift periods. Large swings mean the date itself drives the result.
    2. Date-robust estimate: assumes each true date lies within ±max_shift of the recorded
       one. Uses the latest plausible date (g+max_shift) as the event date and
       anticipation = 2·max_shift, so the base period (g-max_shift-1) is untreated under
       every date in the window and the post periods are treated under every date in it.
       It estimates the effect from g+max_shift onward (a later-exposure ATT when effects
       are dynamic), at the cost of fewer usable cohorts and wider CIs. cohorts_dropped_robust lists the cohorts it
       leaves out by their ORIGINAL dates (v2.5.1; before, the shifted dates g+max_shift were reported), and
       estimand_note says that the estimand changes.
    Verdict: 'survives' if the date-robust CI excludes 0 with the same sign as the
    baseline; otherwise the conclusion depends on the exact date.
    covariates / est_method / x_periods / cluster / alpha: passed to att_gt, so the check uses the same specification
    and the same (clustered) inference at the same level."""
    d = df.copy()
    cov_kw = dict(covariates=covariates, est_method=est_method, x_periods=x_periods, verify=False,
                  cluster=cluster, alpha=alpha)
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
    dropped_orig = [g - max_shift for g in rob["cohorts_dropped"]]
    note = (f"The date-robust estimate is a later-exposure ATT: it averages the effect from g+{max_shift} onward "
            f"(not from g), so with dynamic effects it is not the same estimand as the baseline ATT."
            + (f" Cohorts left out of it (original dates): {dropped_orig}." if dropped_orig else ""))
    out = {"shifts": tab, "date_robust": ro, "date_robust_dynamic": rob["dynamic"], "survives": survives,
           "max_relative_swing": swing, "cohorts_dropped_robust": dropped_orig,
           "cohorts_dropped_robust_shifted": rob["cohorts_dropped"], "estimand_note": note}
    if show:
        print(f"\nTiming sensitivity (Callaway-Sant'Anna, dates shifted by -{max_shift}…+{max_shift} periods)")
        print(tab.to_string(float_format=lambda v: f"{v:.4f}"))
        print(f"Largest swing vs recorded dates: {swing:.0%} of the baseline ATT (mechanical: a wrong date puts"
              " treated periods in the base or untreated ones in the post period, so swings are expected;"
              " the date-robust line below is the verdict)")
        print(f"Date-robust ATT (true date anywhere within ±{max_shift}): {ro['estimate']:.4f} "
              f"(SE {ro['se']:.4f}, {100 * (1 - alpha):g}% CI {ro['ci_low']:.4f} to {ro['ci_high']:.4f})"
              + (f"; cohorts dropped (original dates): {dropped_orig}" if dropped_orig else ""))
        print("  Note: " + note)
        print("  → " + (f"the conclusion survives date errors of up to ±{max_shift} periods." if survives else
                        f"the conclusion DEPENDS on the exact date: with an error of up to ±{max_shift} periods the "
                        "effect is not distinguishable from zero (or flips sign). Pin the date down with sources "
                        "before reporting a causal effect."))
    return out


def did_uncertain_dates(df: pd.DataFrame, unit: str, time: str, y: str, g_low: str, g_high: str,
                        control: str = "never", n_draws: int = 200, n_boot: int = 199, seed: int = 0,
                        covariates: list[str] | None = None, est_method: str = "dr", x_periods: str = "base",
                        cluster: str | None = None, alpha: float = ALPHA, show: bool = True) -> dict:
    """Staggered DiD when each unit's adoption date is only known to lie in a window
    [g_low, g_high] (e.g. a law phased in court by court with imprecise records).
    Never-treated units: g_low = g_high = 0 or NaN.

    1. Bounds-style estimate: event date = g_high, anticipation = max window width, so
       every base period precedes any plausible adoption (Callaway-Sant'Anna, bootstrap CI).
    2. Date draws: n_draws estimates with each unit's date drawn uniformly from its
       window; the spread shows how much date uncertainty alone moves the ATT.
    covariates / est_method / x_periods / cluster / alpha: passed to att_gt (same specification, inference and level)."""
    d = df.copy()
    cov_kw = dict(covariates=covariates, est_method=est_method, x_periods=x_periods, verify=False,
                  cluster=cluster, alpha=alpha)
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
              f"(SE {ro['se']:.4f}, {100 * (1 - alpha):g}% CI {ro['ci_low']:.4f} to {ro['ci_high']:.4f})")
        print(f"ATT across {len(draws)} random dates inside the windows: median {q50:.4f}, "
              f"5–95% {q05:.4f} to {q95:.4f}, same sign in {share_same:.0%} of draws")
        print("  → " + ("the effect survives the date uncertainty." if survives else
                        "the effect does NOT survive the date uncertainty: narrow the windows with better "
                        "records or report the result as inconclusive."))
    return out


def _fe_design(ui: np.ndarray, ti: np.ndarray, n_u: int, n_t: int, t_ref: int):
    """Sparse two-way fixed-effects design: n_u unit dummies and n_t − 1 period dummies (period t_ref dropped)."""
    from scipy import sparse
    n = len(ui)
    cols_t = np.where(ti < t_ref, ti, ti - 1) + n_u
    keep = ti != t_ref
    rows = np.concatenate([np.arange(n), np.arange(n)[keep]])
    cols = np.concatenate([ui, cols_t[keep]])
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n_u + n_t - 1))


def _bjs_core(y: np.ndarray, ui: np.ndarray, ti: np.ndarray, g_t: np.ndarray, treated: np.ndarray,
              n_u: int, n_t: int):
    """Imputation step of Borusyak, Jaravel & Spiess (2024) on arrays. ui, ti: unit and period codes; treated:
    boolean mask of the treated observations (the rest are untreated, Ω0). Fits unit and period effects by OLS on Ω0
    only, imputes Y(0) for the treated observations and returns their τ̂ = Y − Ŷ(0), plus what the variance needs.
    Treated observations whose unit or period has no untreated observation connected to the rest cannot be imputed
    and are flagged (ident = False)."""
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import spsolve
    u0, t0 = ui[~treated], ti[~treated]
    # connectedness of the unit-period graph of untreated observations
    adj = sparse.csr_matrix((np.ones(len(u0)), (u0, t0 + n_u)), shape=(n_u + n_t, n_u + n_t))
    _, lab = connected_components(adj, directed=False)
    has0 = np.zeros(n_u + n_t, bool)
    has0[u0] = True
    has0[t0 + n_u] = True
    main = np.bincount(lab[has0]).argmax() if has0.any() else -1
    in_main = (lab == main) & has0
    ok0 = in_main[u0] & in_main[t0 + n_u]
    ident = treated & in_main[ui] & in_main[ti + n_u]
    t_ref = int(t0[ok0].min())
    Z0 = _fe_design(u0[ok0], t0[ok0], n_u, n_t, t_ref)
    keep_col = np.asarray(Z0.sum(axis=0)).ravel() > 0
    Z0 = Z0[:, keep_col]
    A = (Z0.T @ Z0).tocsc()
    coef = spsolve(A, Z0.T @ y[~treated][ok0])
    Z1 = _fe_design(ui[ident], ti[ident], n_u, n_t, t_ref)[:, keep_col]
    tau = y[ident] - Z1 @ coef
    resid0 = y[~treated][ok0] - Z0 @ coef
    return {"tau": tau, "ident": ident, "ok0": ok0, "Z0": Z0, "Z1": Z1, "A": A, "resid0": resid0}


def _bjs_weights_v(core: dict, w1: np.ndarray) -> np.ndarray:
    """Implied weights on the untreated observations of the linear estimator Σ w1·τ̂: v0 = −Z0 (Z0'Z0)⁻¹ Z1' w1."""
    from scipy.sparse.linalg import spsolve
    b = spsolve(core["A"], core["Z1"].T @ w1)
    return -(core["Z0"] @ b)


def did_imputation(df: pd.DataFrame, unit: str, time: str, y: str, gvar: str, anticipation: int = 0,
                   cluster: str | None = None, horizons: int | None = None, pretrends: int | None = None,
                   alpha: float = ALPHA, show: bool = True) -> dict:
    """Imputation estimator of Borusyak, Jaravel & Spiess (2024, Review of Economic Studies), the second route for
    staggered adoption next to att_gt (v2.5.0). gvar = first treated period (0 or NaN = never treated); a binary,
    absorbing treatment (check with did_design). The panel may be unbalanced.

    1. Unit and period fixed effects are fitted by OLS on the untreated observations only (never-treated units, and
       treated units before g − anticipation), so treated outcomes never enter the counterfactual model.
    2. Y(0) is imputed for every treated observation and τ̂_it = Y_it − Ŷ_it(0).
    3. The overall ATT is the mean of τ̂ over the treated observations with t ≥ g (each treated unit-period weighs the
       same, as in att_gt's simple aggregation); the event-study coefficient at horizon h ≥ 0 is the mean over
       t − g = h. With anticipation = δ, the periods g − δ … g − 1 are kept out of the fixed-effects fit and their τ̂
       are reported as event-study rows flagged anticipation = True, not averaged into the ATT.
    Treated observations whose unit has no untreated period (always treated) or whose period has no untreated unit
    cannot be imputed: they are dropped and counted (warned).

    Standard errors: the conservative clustered variance of the paper (Theorem 3): every estimate is linear in Y,
    Σ v_it Y_it, with v = w on treated observations and v = −Z0(Z0'Z0)⁻¹Z1'w on untreated ones; the variance is
    Σ_clusters (Σ v_it ε̃_it)², with model residuals on untreated observations and, on treated ones, τ̂ minus the
    average of τ̂ in its cohort-period cell (weighted by v²). It is conservative when effects vary within cells.
    Clusters: units by default, cluster= a coarser column (the level of assignment).

    Pre-trend test (separate from estimation, as in the paper): on the untreated observations only, regress Y on unit
    and period effects and indicators of the pretrends = K periods before adoption (default: all available, cut to
    keep K ≤ (G − 1)/10 with G clusters, warned); Wald test that they are zero with a clustered covariance and an
    F(K, G − K) reference. Not rejecting does not show parallel trends. The leads start at e = −1 (reference: the
    earlier periods); R didimputation's pretrends=TRUE reports e = −2, −3, … with e = −1 as reference, so the lead
    coefficients of the two are not comparable one by one.
    Replication: the ATT, SEs (unit and coarser clusters), unbalanced panel and event-study horizons match R
    didimputation 0.5.1 (estimates within 2e-7; SEs within 1e-8, event-study SEs within 1e-7; tests/test_v250.py,
    tools/r_reference).

    Returns the contract dict in "overall" (estimand, estimate, se, ci_low, ci_high, p, n, method, warnings), the
    event-study table "dynamic" (horizons ≥ 0 from the imputation, < 0 from the pre-trend regression), "pretrend",
    the cell table "gt" (mean τ̂ and count per cohort-period, with an anticipation flag; did_compare recomputes the cells
    on its bootstrap draws), the per-observation "tau" (same flag) and
    "converged" (True: the estimator has no iterative step)."""
    from scipy import sparse
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    cols = [unit, time, y, gvar] + ([cluster] if cluster and cluster not in (unit, time, y, gvar) else [])
    d = df[cols].dropna(subset=[unit, time, y]).copy()
    d[gvar] = d[gvar].fillna(0).astype(float)
    if (d.groupby(unit)[gvar].nunique() > 1).any():
        raise ValueError(f"did_imputation: {gvar} must be constant within each {unit}")
    if d.duplicated([unit, time]).any():
        raise ValueError(f"did_imputation: repeated ({unit}, {time}) rows")
    if cluster and (d.groupby(unit)[cluster].nunique() > 1).any():
        raise ValueError(f"did_imputation: cluster '{cluster}' must be constant within each {unit}")
    dlt = int(anticipation)
    ui, uniq_u = pd.factorize(d[unit], sort=True)
    periods = np.sort(d[time].unique())
    ti = np.searchsorted(periods, d[time].to_numpy())
    tv = d[time].to_numpy(float)
    gv = d[gvar].to_numpy(float)
    treated = (gv > 0) & (tv >= gv - dlt)
    yv = d[y].to_numpy(float)
    if treated.all() or not treated.any():
        raise ValueError("did_imputation: need both treated and untreated observations")
    core = _bjs_core(yv, ui, ti, gv, treated, len(uniq_u), len(periods))
    ident = core["ident"]
    n_drop = int((treated & ~ident).sum())
    e = (tv - gv)[ident]
    tau = core["tau"]
    cl_codes = pd.factorize(d[cluster] if cluster else d[unit])[0]
    n_cl = int(cl_codes.max() + 1)
    cl0 = cl_codes[~treated][core["ok0"]]
    cl1 = cl_codes[ident]
    g1, t1 = gv[ident], tv[ident]
    cell = pd.Series(list(zip(g1, t1)))

    def estimate(w1):
        """Estimate and Theorem 3 SE of Σ w1·τ̂ over the imputed treated observations."""
        est = float(w1 @ tau)
        v0 = _bjs_weights_v(core, w1)
        v2 = pd.Series(w1 ** 2)
        num = (v2 * tau).groupby(cell).transform("sum")
        den = v2.groupby(cell).transform("sum")
        tau_tilde = np.where(den.to_numpy() > 0, (num / den.where(den > 0, 1)).to_numpy(), 0.0)
        eps1 = tau - tau_tilde
        s = np.bincount(cl0, weights=v0 * core["resid0"], minlength=n_cl) + np.bincount(cl1, weights=w1 * eps1,
                                                                                         minlength=n_cl)
        return est, float(np.sqrt(np.sum(s ** 2)))

    # the ATT averages t >= g only; with anticipation the periods g - δ … g - 1 are kept out of the FE sample (they
    # may already respond) and reported as separate event-time rows, not mixed into the ATT (review of v2.5.0)
    post = e >= 0
    n1 = int(post.sum())
    if n1 == 0:
        raise ValueError("did_imputation: no treated observation at or after its adoption date can be imputed")
    est, se = estimate(np.where(post, 1.0 / n1, 0.0))
    hs = sorted(set(e.astype(int)))
    if horizons is not None:
        hs = [h for h in hs if h <= horizons]
    rows = []
    for h in hs:
        m = e == h
        w = np.where(m, 1.0 / m.sum(), 0.0)
        eh, sh = estimate(w)
        rows.append({"e": h, "att": eh, "se": sh, "n_obs": int(m.sum()), "anticipation": bool(h < 0)})
    dyn = pd.DataFrame(rows).set_index("e")
    # ---- pre-trend test on the untreated observations only (separate from estimation, as in the paper). By
    # Frisch-Waugh: the lead indicators and Y are residualized on the unit and period effects, and leads that the
    # effects already span are dropped (without never-treated units the full set of leads is collinear with them),
    # keeping the leads nearest to adoption first.
    pt = {"stat": np.nan, "df": 0, "p": np.nan, "reference": None, "leads": 0, "n_clusters": n_cl,
          "leads_dropped_collinear": 0}
    lead_rows, cut, pre_fail = [], False, None
    try:
        unt_idx = np.where(~treated)[0][core["ok0"]]
        rel = np.where(gv > 0, tv - gv, np.nan)[unt_idx]
        K_all = sorted({int(-r) for r in rel[np.isfinite(rel)] if r < -dlt})
        if pretrends is not None:
            K_all = K_all[:int(pretrends)]
        Z0, A0 = core["Z0"], core["A"]
        from scipy.sparse.linalg import splu
        lu = splu(A0)
        y_r = core["resid0"]                          # Y residualized on the effects (fitted on these observations)
        kept, Lr = [], []
        for k in K_all:
            col = (rel == -k).astype(float)
            r = col - Z0 @ lu.solve(np.asarray(Z0.T @ col).ravel())
            if Lr:
                M = np.column_stack(Lr)
                r = r - M @ np.linalg.lstsq(M, r, rcond=None)[0]
            if r @ r > 1e-10 * max(1.0, col @ col):
                kept.append(k)
                Lr.append(col - Z0 @ lu.solve(np.asarray(Z0.T @ col).ravel()))
        pt["leads_dropped_collinear"] = len(K_all) - len(kept)
        c0 = cl_codes[unt_idx]
        Gc = len(np.unique(c0))
        cap_k = max(1, int((Gc - 1) // 10))
        if len(kept) > cap_k:
            kept, Lr, cut = kept[:cap_k], Lr[:cap_k], True
        if kept:
            Lm = np.column_stack(Lr)
            LtL_inv = np.linalg.inv(Lm.T @ Lm)
            bl = LtL_inv @ (Lm.T @ y_r)
            res = y_r - Lm @ bl
            H = Lm @ LtL_inv
            S = np.vstack([np.bincount(c0, weights=H[:, j] * res, minlength=n_cl) for j in range(len(kept))]).T
            V = (S.T @ S) * Gc / max(Gc - 1, 1)
            q = len(kept)
            W = float(bl @ np.linalg.solve(V, bl))
            if Gc > q + 1:
                p_, ref = float(stats.f.sf(W / q * (Gc - q) / (Gc - 1), q, Gc - q)), f"F({q}, {Gc - q})"
            else:
                p_, ref = float(stats.chi2.sf(W, q)), f"chi2({q})"
            pt.update(stat=W, df=q, p=p_, reference=ref, leads=q, n_clusters=Gc)
            sev = np.sqrt(np.clip(np.diag(V), 0, None))
            for j, k in enumerate(kept):
                lead_rows.append({"e": -k, "att": float(bl[j]), "se": float(sev[j]),
                                  "n_obs": int((rel == -k).sum()), "anticipation": False})
    except Exception as exc:  # noqa: BLE001 — the pre-trend test must never break the estimate
        pre_fail = str(exc)
    if lead_rows:
        dyn = pd.concat([pd.DataFrame(lead_rows).set_index("e"), dyn]).sort_index()
    dyn["ci_low"], dyn["ci_high"] = _ci(dyn["att"], dyn["se"], alpha)
    gt = pd.DataFrame({"g": g1, "t": t1, "tau": tau}).groupby(["g", "t"])["tau"].agg(["mean", "size"])
    gt.columns = ["att", "n_obs"]
    gt["e"] = gt.index.get_level_values("t") - gt.index.get_level_values("g")
    gt["anticipation"] = gt["e"] < 0
    warns = []
    if n_drop:
        warns.append(f"{n_drop} treated observations could not be imputed (their unit has no untreated period, or no "
                     "untreated unit is observed in their period) and were left out of every estimate.")
    if cut:
        warns.append(f"Pre-trend test on the {pt['leads']} lead(s) nearest to adoption only: with {pt['n_clusters']} "
                     "clusters a clustered covariance of more coefficients is too noisy for a Wald test.")
    if pt["leads_dropped_collinear"]:
        warns.append(f"{pt['leads_dropped_collinear']} lead(s) are spanned by the unit and period effects (no "
                     "never-treated units, or too few untreated periods) and were left out of the pre-trend test.")
    if pre_fail:
        warns.append(f"The pre-trend test could not be computed ({pre_fail}); the estimates are unaffected.")
    if np.isfinite(pt["p"]) and pt["p"] < alpha:
        warns.append(f"Pre-trend test rejects (p = {pt['p']:.3g}, {pt['reference']}): parallel trends are doubtful; "
                     "compare with att_gt (did_compare) and look at the leads.")
    if n_cl < 30:
        warns.append(f"{n_cl} clusters: the clustered SEs are unreliable and the pre-trend test over-rejects (about 10% "
                     "for a 5% test with 12 clusters in the skill's simulations); add causal.randomization_inference.")
    if dlt:
        warns.append(f"Anticipation = {dlt}: periods g − {dlt} … g − 1 are left out of the fixed-effects fit and reported "
                     "as separate rows of the event study (anticipation = True); the ATT averages t ≥ g only.")
    if not (pd.api.types.is_numeric_dtype(d[time]) and (len(periods) < 2
                                                         or np.allclose(np.diff(periods.astype(float)), 1.0))):
        warns.append(f"The {time} column is not a sequence of consecutive integers: horizons and anticipation are read "
                     "in its units, not in periods; recode time as 1, 2, 3, … first.")
    method = (f"imputation (Borusyak, Jaravel & Spiess 2024), FE fitted on {int((~treated).sum())} untreated "
              f"observations; SEs clustered by {cluster or unit} ({n_cl} clusters)" + (f", anticipation = {dlt}" if dlt else ""))
    out = {"overall": _out("ATT (imputation)", est, se, int(len(d)), alpha=alpha, method=method, warnings=warns,
                           n_treated_obs=n1, n_clusters=n_cl, anticipation=dlt),
           "dynamic": dyn, "pretrend": pt, "pretrend_p": pt["p"], "gt": gt,
           "tau": pd.DataFrame({unit: uniq_u[ui[ident]], time: t1, gvar: g1, "e": e, "tau": tau, "anticipation": e < 0}),
           "dropped_treated_obs": n_drop, "converged": True, "warnings": warns, "alpha": alpha}
    if show:
        o = out["overall"]
        print(f"\nImputation estimator (BJS; {method})")
        print(f"Overall ATT: {o['estimate']:.4f} (SE {o['se']:.4f}, {100 * (1 - alpha):g}% CI {o['ci_low']:.4f} to "
              f"{o['ci_high']:.4f}); {n1} treated observations")
        print("Event study (leads from the pre-trend regression on untreated observations):\n"
              + dyn.to_string(float_format=lambda v: f"{v:.4f}"))
        if pt["leads"]:
            print(f"Pre-trend test ({pt['leads']} leads, untreated observations only): W = {pt['stat']:.3f}, "
                  f"{pt['reference']} p = {pt['p']:.3f} → " + ("pre-trends detected" if pt["p"] < alpha else
                                                            "does not reject; this does not show parallel trends"))
        for w in warns:
            print("  WARNING: " + w)
    return out


def _bjs_cells(y: np.ndarray, ui: np.ndarray, ti: np.ndarray, tv: np.ndarray, gv: np.ndarray, n_u: int, n_t: int,
               dlt: int) -> pd.DataFrame:
    """Mean τ̂ of the imputation estimator per (cohort, period) cell, without SEs (used inside did_compare's bootstrap)."""
    treated = (gv > 0) & (tv >= gv - dlt)
    core = _bjs_core(y, ui, ti, gv, treated, n_u, n_t)
    idt = core["ident"]
    return pd.DataFrame({"g": gv[idt], "t": tv[idt], "tau": core["tau"]}).groupby(["g", "t"])["tau"].mean()


def did_compare(df: pd.DataFrame, unit: str, time: str, y: str, gvar: str, control: str = "notyet",
                anticipation: int = 0, cluster: str | None = None, n_boot: int = 199, seed: int = 0,
                alpha: float = ALPHA, show: bool = True) -> dict:
    """Callaway-Sant'Anna (att_gt) against the imputation estimator (did_imputation) on the same data: a second route
    with a different method for a staggered DiD (v2.5.0). The guide of de Chaisemartin & D'Haultfœuille reports that
    the two families are always close except where pre-trend tests reject, so a significant gap is a finding that
    points to the pre-trends, not a choice between two numbers.

    Same target: both are averaged over the post-period (cohort, period) cells that both define, with the cohort sizes
    of att_gt as weights (its simple aggregation); cells that only one route estimates (a cohort without a base period
    in att_gt, a period with no untreated unit in the imputation) are listed and left out, so the gap is not
    mechanical. The difference is tested with its own SE from a paired bootstrap: in every draw of units (whole
    clusters with cluster=) both estimators are recomputed on the same resampled data, and the test is at alpha.
    Requires a balanced panel (att_gt). Both routes run without covariates (the imputation estimator here has none),
    and control defaults to 'notyet' (the guide's recommended comparison group), while att_gt defaults to 'never':
    pass the control of the headline att_gt, and if that one uses covariates, say that the comparison checks the
    unconditional specification. Pass the result to reporting.cross_check(res["cross_check"],
    se=res["diff_se"], tol_se=Φ⁻¹(1 − alpha/2)) to record it as an independent check (kind "method")."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    eng = _cs_engine(df, unit, time, y, gvar, control, anticipation, [], "dr", "base", cluster)
    Ym, G, T, dlt = eng["Ym"], eng["G"], eng["T"], eng["dlt"]
    n_u, n_t = Ym.shape
    tv_full = np.tile(T.astype(float), n_u)

    def routes(idx):
        cs = eng["compute"](idx)
        cs = cs[cs["e"] >= 0].set_index(["g", "t"])
        k = len(idx)
        yv = Ym[idx].ravel()
        gv = np.repeat(G[idx].astype(float), n_t)
        ui = np.repeat(np.arange(k), n_t)
        ti = np.tile(np.arange(n_t), k)
        bj = _bjs_cells(yv, ui, ti, np.tile(T.astype(float), k), gv, k, n_t, dlt)
        return cs, bj

    cs0, bj0 = routes(np.arange(n_u))
    bj0.index = pd.MultiIndex.from_arrays([bj0.index.get_level_values(0), bj0.index.get_level_values(1)],
                                          names=["g", "t"])
    cs0.index = pd.MultiIndex.from_arrays([cs0.index.get_level_values(0).astype(float),
                                          cs0.index.get_level_values(1).astype(float)], names=["g", "t"])
    common = cs0.index.intersection(bj0.index)
    only_cs = [tuple(x) for x in cs0.index.difference(common)]
    only_bjs = [tuple(x) for x in bj0.index.difference(common)
                if x[1] >= x[0]]                                  # post cells only
    if len(common) == 0:
        raise ValueError("did_compare: the two estimators share no post-period cell")
    wts = cs0.loc[common, "n_g"].to_numpy(float)

    def agg(cs, bj):
        cs = cs.copy()
        cs.index = pd.MultiIndex.from_arrays([cs.index.get_level_values(0).astype(float),
                                             cs.index.get_level_values(1).astype(float)])
        a = cs["att"].reindex(common).to_numpy(float)
        b = bj.reindex(common).to_numpy(float)
        if np.isnan(a).any() or np.isnan(b).any():
            return np.nan, np.nan
        return float(wts @ a / wts.sum()), float(wts @ b / wts.sum())

    est_cs, est_bj = agg(cs0, bj0)
    diff = est_cs - est_bj
    draws = []
    if n_boot and n_boot > 1:
        rng = np.random.default_rng(seed)
        for _ in range(n_boot):
            idx = _cs_draw(rng, eng)
            try:
                c, b = routes(idx)
            except Exception:  # noqa: BLE001 — a degenerate draw (no untreated observation, singular FE): skipped
                draws.append((np.nan, np.nan))
                continue
            draws.append(agg(c, b))
    D = np.array(draws, float).reshape(-1, 2)
    okd = np.isfinite(D).all(axis=1)
    n_ok = int(okd.sum())
    se_cs = float(np.std(D[okd, 0], ddof=1)) if n_ok > 1 else np.nan
    se_bj = float(np.std(D[okd, 1], ddof=1)) if n_ok > 1 else np.nan
    se_d = float(np.std(D[okd, 0] - D[okd, 1], ddof=1)) if n_ok > 1 else np.nan
    z = diff / se_d if np.isfinite(se_d) and se_d > 0 else np.nan
    p = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan
    diverge = bool(np.isfinite(p) and p < alpha)
    corr = float(np.corrcoef(D[okd].T)[0, 1]) if n_ok > 2 else np.nan
    warns = []
    if diverge:
        warns.append(f"Callaway-Sant'Anna ({est_cs:.4f}) and the imputation estimator ({est_bj:.4f}) differ by "
                     f"{diff:.4f} (SE of the difference {se_d:.4f}, p = {p:.3g}): the two routes disagree, which in "
                     "the guide's meta-analysis happened only when pre-trend tests rejected. Look at att_gt's and "
                     "did_imputation's pre-trend tests and leads before reporting either number.")
    if only_cs or only_bjs:
        warns.append(f"Compared on {len(common)} common post-period cells; left out: {len(only_cs)} cell(s) only in "
                     f"att_gt and {len(only_bjs)} only in did_imputation (each route's overall ATT also averages over "
                     "its own cells).")
    if n_boot and n_ok < n_boot:
        warns.append(f"{n_boot - n_ok} of {n_boot} bootstrap draws could not compute both routes and were skipped.")
    if cluster and (eng["n_cl"] < 30 or eng["n_cl_treated"] < 6):
        warns.append(f"{eng['n_cl']} clusters ({eng['n_cl_treated']} treated): the bootstrap SE of the difference is "
                     "unreliable; read the comparison as descriptive.")
    out = {"cs": est_cs, "bjs": est_bj, "difference": diff, "diff_se": se_d, "z": z, "p": p, "diverge": diverge,
           "se_cs": se_cs, "se_bjs": se_bj, "corr": corr, "cells_common": len(common),
           "cells_only_cs": only_cs, "cells_only_bjs": only_bjs, "draws_used": n_ok, "alpha": alpha,
           "control": control, "cross_check": pd.Series({"Callaway-Sant'Anna": est_cs, "imputation (BJS)": est_bj},
                                                        name="ATT on common cells"),
           "warnings": warns}
    if show:
        print(f"\nCallaway-Sant'Anna ({control} controls) vs imputation (BJS), {len(common)} common post cells, "
              f"{n_ok} paired bootstrap draws" + (f" by {cluster}" if cluster else ""))
        print(f"  CS {est_cs:.4f} (SE {se_cs:.4f}); BJS {est_bj:.4f} (SE {se_bj:.4f}); difference {diff:.4f} "
              f"(SE {se_d:.4f}, p = {p:.3f}) → " + ("the routes DISAGREE at alpha = " + f"{alpha:g}" if diverge else
                                                   f"no significant difference at alpha = {alpha:g}"))
        for w in warns:
            print("  WARNING: " + w)
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
        "time": "Covariate effects change over time but not across groups: a TWFE regression with covariates "
                "suffers hidden linearity bias; use att_gt(covariates=...), which conditions on levels in every "
                "group-time cell (x_periods='base' with baseline or time-invariant covariates; 'both' only for "
                "covariates that change over time and that treatment cannot affect).",
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


def bacon_decomposition(df: pd.DataFrame, unit: str, time: str, y: str, gvar: str, show: bool = True) -> dict:
    """Goodman-Bacon (2021) decomposition of the two-way fixed effects DiD coefficient (no
    covariates, balanced panel, absorbing binary treatment) into every 2x2 comparison and
    its weight. gvar = first treated period; 0/NaN or a date after the panel = never treated;
    a date at or before the first period = always treated (acts only as a control).

    Three kinds of 2x2:
      'treated vs never'            clean;
      'earlier vs later (control)'  clean: the later cohort is still untreated in the window;
      'later vs earlier (control)'  FORBIDDEN: the already-treated earlier cohort is the control.
    If effects grow with exposure, the forbidden comparisons subtract the earlier cohort's
    growing effect: TWFE is attenuated, and can flip sign, even with every weight positive.
    The weights sum to one and sum(weight * estimate) equals the TWFE coefficient exactly
    (checked and reported as 'identity_gap'). Equivalent of R bacondecomp / Stata bacondecomp."""
    d = df[[unit, time, y, gvar]].copy()
    d[gvar] = d[gvar].fillna(0)
    Y = d.pivot(index=unit, columns=time, values=y)
    if Y.isna().any().any():
        raise ValueError("bacon_decomposition needs a balanced panel")
    T = np.array(Y.columns)
    g = d.groupby(unit)[gvar].first().reindex(Y.index).to_numpy(float)
    g = np.where((g == 0) | (g > T.max()), np.inf, np.where(g <= T.min(), T.min(), g))   # inf = never
    Ym = Y.to_numpy(float)
    N, nT = Ym.shape
    groups = sorted(set(g))
    share = {k: np.mean(g == k) for k in groups}
    dbar = {k: (0.0 if np.isinf(k) else np.mean(T >= k)) for k in groups}
    mean_t = {k: Ym[g == k].mean(axis=0) for k in groups}

    def dd(treat, ctrl, pre, post):
        if not pre.any() or not post.any():
            return np.nan
        return ((mean_t[treat][post].mean() - mean_t[treat][pre].mean())
                - (mean_t[ctrl][post].mean() - mean_t[ctrl][pre].mean()))

    rows = []
    timing = [k for k in groups if np.isfinite(k)]
    if np.inf in groups:
        for k in timing:
            nk, nu = share[k], share[np.inf]
            nku = nk / (nk + nu)
            w = (nk + nu) ** 2 * nku * (1 - nku) * dbar[k] * (1 - dbar[k])
            rows.append(("treated vs never", k, np.inf, dd(k, np.inf, T < k, T >= k), w))
    for i, k in enumerate(timing):
        for l in timing[i + 1:]:                         # k earlier than l
            nk, nl = share[k], share[l]
            nkl = nk / (nk + nl)
            Dk, Dl = dbar[k], dbar[l]
            if Dl < 1:   # earlier treated vs later as control, window before l
                w = ((nk + nl) * (1 - Dl)) ** 2 * nkl * (1 - nkl) * ((Dk - Dl) / (1 - Dl)) * ((1 - Dk) / (1 - Dl))
                win = T < l
                rows.append(("earlier vs later (control)", k, l, dd(k, l, win & (T < k), win & (T >= k)), w))
            if Dk > 0:   # later treated vs earlier (already treated) as control, window from k
                w = ((nk + nl) * Dk) ** 2 * nkl * (1 - nkl) * (Dl / Dk) * ((Dk - Dl) / Dk)
                win = T >= k
                rows.append(("later vs earlier (control)", l, k, dd(l, k, win & (T < l), win & (T >= l)), w))
    tab = pd.DataFrame(rows, columns=["type", "treated", "control", "estimate", "weight"])
    tab = tab[(tab["weight"] > 0) & tab["estimate"].notna()].reset_index(drop=True)
    tab["weight"] = tab["weight"] / tab["weight"].sum()
    tab["control"] = tab["control"].replace(np.inf, "never")
    # TWFE coefficient by two-way within transformation
    Dm = (g[:, None] <= T[None, :]).astype(float)
    def within(A):
        return A - A.mean(axis=1, keepdims=True) - A.mean(axis=0, keepdims=True) + A.mean()
    Dw = within(Dm)
    twfe = float((Dw * Ym).sum() / (Dw * Dm).sum())
    recon = float((tab["weight"] * tab["estimate"]).sum())
    by_type = tab.groupby("type").apply(
        lambda s: pd.Series({"weight": s["weight"].sum(), "estimate": np.average(s["estimate"], weights=s["weight"])}),
        include_groups=False)
    forbidden = float(by_type["weight"].get("later vs earlier (control)", 0.0))
    warns = []
    if forbidden >= 0.10:
        clean = tab[tab["type"] != "later vs earlier (control)"]
        fb = by_type.loc["later vs earlier (control)", "estimate"]
        cl = np.average(clean["estimate"], weights=clean["weight"]) if len(clean) else np.nan
        warns.append(f"{forbidden:.0%} of the TWFE weight comes from comparisons that use already-treated units as "
                     f"controls (their average 2x2: {fb:.4g}; clean comparisons: {cl:.4g}). With effects that change "
                     "over time these are biased: report att_gt.")
    out = {"table": tab, "by_type": by_type, "twfe": twfe, "reconstructed": recon,
           "identity_gap": abs(twfe - recon), "forbidden_weight": forbidden, "warnings": warns}
    if show:
        print(f"\nGoodman-Bacon decomposition of the TWFE coefficient ({twfe:.4f}; weighted sum of 2x2s {recon:.4f})")
        print(by_type.to_string(float_format=lambda v: f"{v:.4f}"))
        for w_ in warns:
            print("  WARNING: " + w_)
    return out


def twfe_weights(df: pd.DataFrame, unit: str, time: str, d: str, y: str | None = None, show: bool = True) -> dict:
    """Weights behind a two-way fixed-effects coefficient (de Chaisemartin & D'Haultfœuille 2020, AER; R/Stata
    twowayfeweights, type "feTR"), for a binary or a non-binary treatment (v2.5.0).

    Under parallel trends, the TWFE coefficient of y on d with unit and period effects is a weighted sum of the
    treatment effects of the treated cells: β_fe = Σ_{d ≠ 0} w_it Δ_it, with Δ_it the effect per unit of treatment
    and w_it = d_it ε_it / Σ d ε, where ε is the residual of d on unit and period effects. The weights sum to 1 but can
    be negative: a negative weight means that a treated cell serves as a control for others (typically early adopters
    late in the panel). With heterogeneous effects, β_fe can then lie outside the range of every Δ, even with the
    opposite sign. Unlike bacon_decomposition, this works with a non-binary (dose) and with an unbalanced panel.

    sigma_fe = |β_fe| / σ(w̃), σ(w̃)² = Σ s (w̃ − 1)² · N/(N − 1), s = d / Σ d (the treatment shares), w̃ = w / s,
    N = treated cells: the dispersion of the effects across treated cells under which β_fe is compatible with an
    average effect of zero. The N/(N − 1) factor follows the authors' twowayfeweights (R 2.1.0, matched to 1e-9 in the
    tests), so the value is the population SD of the least dispersed such effects times sqrt((N − 1)/N). A sigma_fe that is small next to plausible heterogeneity means β_fe could have the wrong sign.
    With y, β_fe is also computed (by Frisch-Waugh, Σ ε y / Σ ε d); without it, only the weights are returned.
    Returns: weights table (unit, time, d, weight), sum and count of positive and negative weights, sigma_fe (with y),
    beta (with y), binary flag, warnings."""
    from scipy.sparse.linalg import spsolve
    cols = [unit, time, d] + ([y] if y else [])
    x = df[cols].dropna().copy()
    if x.duplicated([unit, time]).any():
        raise ValueError(f"twfe_weights: repeated ({unit}, {time}) rows")
    ui, uniq_u = pd.factorize(x[unit], sort=True)
    periods = np.sort(x[time].unique())
    ti = np.searchsorted(periods, x[time].to_numpy())
    dv = x[d].to_numpy(float)
    Z = _fe_design(ui, ti, len(uniq_u), len(periods), 0)
    Z = Z[:, np.asarray(Z.sum(axis=0)).ravel() > 0]
    eps = dv - Z @ spsolve((Z.T @ Z).tocsc(), Z.T @ dv)
    tr = dv != 0
    if not tr.any():
        raise ValueError("twfe_weights: no treated cell")
    denom = float(dv @ eps)
    if abs(denom) < 1e-12:
        raise ValueError("twfe_weights: the treatment is fully absorbed by the unit and period effects (no TWFE "
                         "coefficient)")
    w = dv * eps / denom
    s = np.where(tr, dv / dv[tr].sum(), 0.0)
    wt = np.where(tr, w / np.where(tr, s, 1), 0.0)
    n_tr = int(tr.sum())
    # dispersion of the weights with the N/(N − 1) factor of the authors' twowayfeweights (R 2.1.0: its sigma_fe
    # equals |β| / sqrt(Σ s (w̃ − 1)² · N / (N − 1)), N = treated cells; matched to 1e-9 in tests/test_v250.py)
    sig_w = float(np.sqrt(np.sum(s[tr] * (wt[tr] - 1) ** 2) * n_tr / max(n_tr - 1, 1)))
    binary = bool(np.isin(np.unique(dv), [0.0, 1.0]).all())
    tab = pd.DataFrame({unit: uniq_u[ui[tr]], time: x[time].to_numpy()[tr], d: dv[tr], "weight": w[tr]})
    neg = tab["weight"] < 0
    beta = float(eps @ x[y].to_numpy(float) / denom) if y else np.nan
    sigma_fe = float(abs(beta) / sig_w) if y and sig_w > 0 else np.nan
    warns = []
    if neg.any():
        warns.append(f"{int(neg.sum())} of {len(tab)} treated cells get a negative weight (sum {tab.loc[neg, 'weight'].sum():.3f}):"
                     " the TWFE coefficient is not a convex average of the effects; with heterogeneous effects it can "
                     "even have the wrong sign. Report a heterogeneity-robust estimator (causal.did_design says which).")
        if y and np.isfinite(sigma_fe):
            warns.append(f"sigma_fe = {sigma_fe:.4g}: effects varying across treated cells with a standard deviation of "
                         "this size are enough for the average effect to be zero while TWFE reports "
                         f"{beta:.4g}.")
    out = {"weights": tab, "n_treated_cells": int(len(tab)), "n_positive": int((~neg).sum()),
           "n_negative": int(neg.sum()), "sum_positive": float(tab.loc[~neg, "weight"].sum()),
           "sum_negative": float(tab.loc[neg, "weight"].sum()), "sigma_w": sig_w, "sigma_fe": sigma_fe, "beta": beta,
           "binary": binary, "warnings": warns,
           "method": "de Chaisemartin & D'Haultfœuille (2020) weights of the TWFE coefficient (feTR)"}
    if show:
        print(f"\nTWFE weights ({'binary' if binary else 'non-binary'} treatment, {len(tab)} treated cells)")
        print(f"  positive: {out['n_positive']} (sum {out['sum_positive']:.4f}); negative: {out['n_negative']} "
              f"(sum {out['sum_negative']:.4f})")
        if y:
            print(f"  TWFE coefficient {beta:.4f}; sigma_fe {sigma_fe:.4f}")
        for m in warns:
            print("  WARNING: " + m)
    return out


def twfe_covariate_diagnostic(df: pd.DataFrame, unit: str, time: str, y: str, gvar: str, covariates: list[str],
                              control: str = "never", est_method: str = "dr", x_periods: str = "both",
                              n_boot: int = 199, seed: int = 0, cs_covariates: list[str] | None = None,
                              cs_x_periods: str | None = None, cluster: str | None = None,
                              show: bool = True) -> dict:
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
    3. Forbidden comparisons: the Goodman-Bacon decomposition of the TWFE without covariates
       (bacon_decomposition). Treated weights can all be positive while a large share of the
       TWFE weight comes from 2x2s whose control is already treated; with effects that grow
       with exposure those comparisons attenuate TWFE.
    4. Benchmark: att_gt with covariates in levels (doubly robust by default); the gap is
       reported in units of its bootstrap SE (approximate: the two estimates are correlated).

    covariates = the TWFE specification under review (the parecer's controls, bad ones
    included). cs_covariates = the valid pre-treatment confounders for the benchmark and for
    the hidden-linearity test (default: the same list); never put a control the treatment can
    affect there. cs_x_periods defaults to x_periods. cluster = the level of treatment
    assignment (state): TWFE SEs, the hidden-linearity test and the att_gt bootstrap cluster
    on it (default: the unit).
    Verdict 'unreliable' if (1) rejects at 5%, any treated weight is negative, forbidden
    comparisons carry >= 10% of the weight with an average 2x2 that differs from the clean ones
    by more than a quarter, or the gap exceeds 2 SE; then report att_gt(covariates=...). A
    benchmark too noisy to detect the gap is flagged: 'no evidence' then means no power."""
    from linearmodels.panel import PanelOLS
    covs = list(covariates)
    ccovs = list(cs_covariates) if cs_covariates is not None else covs
    cxp = cs_x_periods or x_periods
    allc = list(dict.fromkeys(covs + ccovs))
    extra = [cluster] if cluster and cluster not in (unit, time, y, gvar, *allc) else []
    d = df[[unit, time, y, gvar, *allc, *extra]].copy()
    d[gvar] = d[gvar].fillna(0)
    d = d.sort_values([unit, time]).reset_index(drop=True)
    if d.groupby(unit)[time].count().nunique() != 1 or d[[y, *allc]].isna().any().any():
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
    tw = (PanelOLS(pdf[y], exog, entity_effects=True, time_effects=True, drop_absorbed=True).fit(
        cov_type="clustered", clusters=pd.Series(pd.factorize(pdf[cluster])[0], index=pdf.index, name="cl"))
          if cluster else
          PanelOLS(pdf[y], exog, entity_effects=True, time_effects=True, drop_absorbed=True).fit(
              cov_type="clustered", cluster_entity=True))
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
    fd = pd.DataFrame({"_dy": g[y].diff(), "_t": d[time], "_u": d[cluster] if cluster else d[unit]})
    lev_cols, dif_cols = [], []
    for c in ccovs:
        fd["L_" + c] = g[c].shift(1)
        lev_cols.append("L_" + c)
        if d.groupby(unit)[c].nunique().max() > 1:
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
    # 3. forbidden comparisons (Goodman-Bacon, no covariates)
    bac = bacon_decomposition(d, unit, time, y, gvar, show=False)
    bt = bac["by_type"]
    fb_w = bac["forbidden_weight"]
    forbidden_bias = False
    if fb_w >= 0.10:
        clean = bac["table"][bac["table"]["type"] != "later vs earlier (control)"]
        cl_est = np.average(clean["estimate"], weights=clean["weight"]) if len(clean) else np.nan
        fb_est = bt.loc["later vs earlier (control)", "estimate"]
        forbidden_bias = bool(np.isfinite(cl_est) and abs(fb_est - cl_est) > 0.25 * abs(cl_est))
        if forbidden_bias:
            warns.append(f"Forbidden comparisons: {fb_w:.0%} of the TWFE weight uses already-treated units as controls, "
                         f"and those 2x2s average {fb_est:.4g} against {cl_est:.4g} for the clean ones (Goodman-Bacon). "
                         "Effects that change with exposure are leaking into the TWFE coefficient.")
    # 4. benchmark
    cs = att_gt(df, unit, time, y, gvar, control=control, n_boot=n_boot, seed=seed, covariates=ccovs,
                est_method=est_method, x_periods=cxp, cluster=cluster, show=False)
    co = cs["overall"]
    gap = twfe["estimate"] - co["estimate"]
    gap_se = gap / co["se"] if np.isfinite(co["se"]) and co["se"] > 0 else np.nan
    if np.isfinite(gap_se) and abs(gap_se) > 2:
        warns.append(f"TWFE ({twfe['estimate']:.4f}) and Callaway-Sant'Anna with covariates ({co['estimate']:.4f}) differ "
                     f"by {abs(gap_se):.1f} SE.")
    low_power = bool(np.isfinite(co["se"]) and 2 * co["se"] > 0.5 * max(abs(co["estimate"]), abs(twfe["estimate"])))
    if low_power:
        warns.append(f"Low power of the benchmark comparison: only a gap above 2 x SE = {2 * co['se']:.4g} would be "
                     f"flagged, against estimates of {twfe['estimate']:.4g} (TWFE) and {co['estimate']:.4g} (CS). A "
                     "small gap here is not evidence that TWFE is right.")
    warns += [w_ for w_ in cs["warnings"]]
    unreliable = bool(p_hl < 0.05 or neg_share > 0 or forbidden_bias or (np.isfinite(gap_se) and abs(gap_se) > 2))
    verdict = ("unreliable" if unreliable else
               "inconclusive (low power)" if low_power else "no evidence against TWFE")
    out = {"twfe": twfe, "att_gt": co, "gap": gap, "gap_in_se": gap_se, "hidden_linearity_p": p_hl,
           "level_coefficients": lev_tab, "negative_weight_share": neg_share, "negative_weight_sum": neg_sum,
           "bacon": bt, "forbidden_weight": fb_w, "low_power": low_power, "cs_covariates": ccovs,
           "verdict": verdict, "warnings": warns}
    if show:
        print(f"\nTWFE with covariates ({', '.join(covs)}) vs conditional parallel trends on "
              f"({', '.join(ccovs)})" + (f"; clustered by {cluster}" if cluster else ""))
        print(f"TWFE: {twfe['estimate']:.4f} (SE {twfe['se']:.4f})   att_gt ({est_method}, {cxp}): "
              f"{co['estimate']:.4f} (SE {co['se']:.4f})   gap {gap:+.4f} ({gap_se:+.1f} SE)")
        print(f"Hidden linearity test (levels of X in untreated trends): joint p = {p_hl:.4g}")
        print(lev_tab.to_string(float_format=lambda v: f"{v:.4f}"))
        print(f"Treated observations with negative TWFE weight: {neg_share:.1%}")
        print("Goodman-Bacon (TWFE without covariates):\n" + bt.to_string(float_format=lambda v: f"{v:.4f}"))
        print("Verdict: " + {"unreliable": "TWFE with covariates is NOT reliable here; report att_gt with covariates.",
                             "inconclusive (low power)": "inconclusive: the benchmark is too noisy to confirm or "
                             "reject TWFE (report att_gt and its CI).",
                             "no evidence against TWFE": "no evidence against TWFE on these checks (report both)."}[verdict])
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
                            stat=None, ci: bool = True, ci_points: int = 61, seed: int = 0, alpha: float = ALPHA,
                            cluster: str | None = None) -> dict:
    """Fisher randomization inference for a difference-in-differences when only a few units are
    treated (a law adopted by 6 of 27 states): clustered SEs and even the wild cluster bootstrap
    become unreliable with very few treated clusters (MacKinnon & Webb 2017, 2018).

    gvar: first treated period of each unit (0 or NaN = never treated); D_it = 1{t >= g_i}.
    Under the sharp null of no effect for anyone, which units adopted is (as if) random, so the
    statistic is recomputed after reassigning the observed adoption dates to random sets of the
    same number of units (all sets when there are no more than n_perm of them). p = share of
    |stat*| >= |stat|: over all assignments when they are enumerated (the observed one is among them), and
    (1 + #) / (n_perm + 1) with random draws.
    stat=None uses the static TWFE DiD coefficient (unit and time fixed effects), computed fast by
    residualizing on the fixed effects once; pass a function stat(df_perm) -> float for any other
    statistic (e.g. lambda d: att_gt(d, unit, time, y, gvar, n_boot=0, show=False)["overall"]["estimate"]); the CI is then skipped.
    The CI inverts the test for a constant additive effect tau: y - tau*D is permuted for a grid of tau.

    Validity rests on exchangeability: the treated units must be comparable to the others in
    their error variance and dynamics. If the adopters are systematically more volatile (e.g.
    small states), randomization inference over-rejects too; use Ferman & Pinto (2019) or
    Conley & Taber (2011). The smallest attainable p-value is 1 / (number of possible assignments).
    cluster (v2.5.1): permute at the level of assignment. When a law is adopted by states and the units are counties,
    the observed adoption dates are reassigned to random sets of STATES (every county of a state gets its state's date);
    permuting counties would break the within-state dependence and over-reject. The cluster column must be constant
    within unit, and gvar constant within cluster (an error otherwise). n_units and n_treated then count clusters."""
    from itertools import combinations
    from math import comb, factorial
    d = df[[unit, time, y, gvar]].dropna(subset=[unit, time, y]).copy()
    d[gvar] = d[gvar].fillna(0)
    if cluster:
        d[cluster] = df.loc[d.index, cluster]
        if d[cluster].isna().any():
            raise ValueError(f"randomization_inference: missing values in cluster column '{cluster}'")
        if (d.groupby(unit)[cluster].nunique() > 1).any():
            raise ValueError(f"randomization_inference: cluster '{cluster}' must be constant within each {unit}")
        if (d.groupby(cluster)[gvar].nunique() > 1).any():
            raise ValueError(f"randomization_inference: {gvar} varies within {cluster}; the treatment is not assigned "
                             f"at the {cluster} level, so permuting {cluster}s is not the assignment mechanism")
        key = cluster
    else:
        key = unit
    units = pd.Index(sorted(d[key].unique(), key=str))
    g_unit = d.groupby(key)[gvar].first().reindex(units)
    treated = list(g_unit[g_unit > 0].index)
    N, N1 = len(units), len(treated)
    if N1 == 0 or N1 == N:
        raise ValueError("need both treated and never-treated units" + (f" ({cluster}s)" if cluster else ""))
    dates_obs = g_unit[treated].to_numpy()
    u_idx = units.get_indexer(d[key])
    tt = d[time].to_numpy()
    # The assignment space is the sets of treated units TIMES the distinct ways the observed adoption dates
    # can be dealt out among them. With one date there is a single arrangement and the sets are the space;
    # with staggered dates each set has several arrangements, so both are enumerated (or drawn together).
    # Enumerating only the sets and drawing one arrangement per set left the OBSERVED assignment out of the
    # enumeration, and the exact p of v2.5.2 could come out 0 (v2.5.2 review, chat 3).
    n_date_arr = factorial(N1)
    for c in np.unique(dates_obs, return_counts=True)[1]:
        n_date_arr //= factorial(int(c))
    n_assign = comb(N, N1) * n_date_arr
    rng = np.random.default_rng(seed)
    if n_assign - 1 <= n_perm:
        arrs = ([np.asarray(dates_obs, float)] if n_date_arr == 1 else
                [np.asarray(a, float) for a in _distinct_arrangements(dates_obs)])
        sets = [(list(c), a) for c in combinations(range(N), N1) for a in arrs]
        exact = True
    else:
        sets = [(list(rng.choice(N, N1, replace=False)), rng.permutation(dates_obs)) for _ in range(n_perm)]
        exact = False
    obs_set = [units.get_loc(u) for u in treated]

    def D_for(idx_set, dates):
        g = np.zeros(N)
        g[idx_set] = dates
        gi = g[u_idx]
        return ((gi > 0) & (tt >= gi)).astype(float)

    D_obs = D_for(obs_set, dates_obs)
    perms_D = [D_for(sset, dts) for sset, dts in sets]
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
            hits = np.sum(np.abs(sp) >= abs(s0) - 1e-12)
            # exact: the enumeration contains the observed assignment (v2.5.2: it was counted twice; the
            # arrangements of the dates are enumerated too, so this holds with staggered dates as well)
            return hits / len(sets) if exact else (1 + hits) / (len(sets) + 1)

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
        for sset, dts in sets:
            g = np.zeros(N)
            g[sset] = dts
            dp = d.copy()
            dp[gvar] = g[u_idx]
            draws.append(stat(dp))
        draws = np.asarray(draws, float)
        hits = np.sum(np.abs(draws) >= abs(est) - 1e-12)
        p = float(hits / len(draws) if exact else (1 + hits) / (len(draws) + 1))
        lo = hi = np.nan
        method = "randomization inference (user statistic)"
    warn = []
    if n_assign < 1 / alpha * 2:
        warn.append(f"only {n_assign} possible assignments: the smallest attainable p-value is {1 / n_assign:.3f}")
    for w in warn:
        warnings.warn(w)
    if cluster:
        method += f", dates permuted across {cluster}s"
    return {"estimand": "ATT (DiD)", "method": method, "estimate": est, "p": p, "ci_low": lo, "ci_high": hi,
            "n_units": N, "n_treated": N1, "possible_assignments": n_assign, "exact": exact, "cluster": cluster,
            "n_perm": len(sets), "n": len(d), "warnings": warn}


# =============================================================================
# DiD with one (or very few) treated units and a short time series (v2.5.1)
# =============================================================================
def _sts_lagweights(w: np.ndarray) -> np.ndarray:
    """c_k = Σ_{|i−j|=k} w_i w_j for a contrast w, so that the variance of w'd under a stationary AR(1) with
    innovation variance s² is s²/(1 − ρ²) · Σ_k c_k ρ^k."""
    return np.array([w @ w] + [2 * (w[j:] @ w[:-j]) for j in range(1, len(w))])


def _sts_blocks(d: np.ndarray, T0: int, q: int, alpha: float):
    """Second route: q block means in each segment and a Welch t test between the pre and post block means."""
    pre = np.array([c.mean() for c in np.array_split(d[:T0], q)])
    post = np.array([c.mean() for c in np.array_split(d[T0:], q)])
    m = post.mean() - pre.mean()
    v1, v0 = post.var(ddof=1) / q, pre.var(ddof=1) / q
    se = float(np.sqrt(v1 + v0))
    if se == 0:
        return {"estimate": float(m), "se": 0.0, "df": np.nan, "p": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    df = (v1 + v0) ** 2 / (v1 ** 2 / (q - 1) + v0 ** 2 / (q - 1))
    tc = stats.t.ppf(1 - alpha / 2, df)
    return {"estimate": float(m), "se": se, "df": float(df), "p": float(2 * stats.t.sf(abs(m) / se, df)),
            "ci_low": float(m - tc * se), "ci_high": float(m + tc * se)}


def _sts_design(T: int, T0: int | None):
    """Design of the two contrasts the calibrated route tests. T0 given: the DiD step (a mean for the pre and one for
    the post segment; contrast = post minus pre; ρ pooled within the two segments). T0 None: a linear trend over the
    whole series (intercept and slope; contrast = the slope per period; one segment)."""
    if T0 is not None:
        X = np.column_stack([np.r_[np.ones(T0), np.zeros(T - T0)], np.r_[np.zeros(T0), np.ones(T - T0)]])
        w = np.r_[-np.ones(T0) / T0, np.ones(T - T0) / (T - T0)]
        breaks = [T0]
    else:
        tt = np.arange(T, dtype=float) - (T - 1) / 2
        X = np.column_stack([np.ones(T), tt])
        w = tt / (tt @ tt)
        breaks = []
    return X, w, breaks


_STS_MU_CACHE: dict = {}


def _sts_raw_rho(u: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Lag-1 autocorrelation of residuals u (rows), from the lag products that do not cross a break."""
    num = (u[:, 1:] * u[:, :-1])[:, keep].sum(1)
    den = (u[:, :-1] ** 2)[:, keep].sum(1)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)


def _sts_mu_map(X: np.ndarray, breaks: list[int]):
    """Median-unbiased ρ (Andrews 1993) for this design: the median of the raw ρ̂ is simulated on a grid of true ρ
    (stationary Gaussian AR(1), 2000 series per point, an internal seed so the map does not depend on the user's
    seed) and inverted by interpolation. Cached per design. Returns (grid of medians, grid of ρ)."""
    T = X.shape[0]
    key = (T, tuple(breaks), X.shape[1])
    if key not in _STS_MU_CACHE:
        rng = np.random.default_rng(20261107)
        M = np.eye(T) - X @ np.linalg.pinv(X)
        keep = np.ones(T - 1, bool)
        for bk in breaks:
            keep[bk - 1] = False
        grid = np.linspace(-0.9, 0.95, 38)
        med = np.empty(len(grid))
        for i, g in enumerate(grid):
            z = rng.standard_normal((2000, T))
            x = np.empty_like(z)
            x[:, 0] = z[:, 0] / np.sqrt(1 - g ** 2)
            for t in range(1, T):
                x[:, t] = g * x[:, t - 1] + z[:, t]
            med[i] = np.median(_sts_raw_rho(x @ M.T, keep))
        med = np.maximum.accumulate(med) + np.arange(len(med)) * 1e-9     # strictly increasing for np.interp
        _STS_MU_CACHE[key] = (med, grid)
    return _STS_MU_CACHE[key]


def _sts_general(d: np.ndarray, X: np.ndarray, w: np.ndarray, breaks: list[int]):
    """Contrast w'd, its AR(1) SE and t, vectorized over rows of d. Residuals of the projection on X; ρ from lag-1
    products that do not cross a break, made median-unbiased for this design (_sts_mu_map; Kendall's first-order
    correction left a downward bias of 0.1 to 0.25 with 12 periods and a trend), clipped to ±0.95; innovation
    variance on (lag pairs kept) − k − 1 degrees of freedom; SE from the exact AR(1) variance of the contrast."""
    d = np.atleast_2d(d)
    T = d.shape[1]
    k = X.shape[1]
    M = np.eye(T) - X @ np.linalg.pinv(X)
    u = d @ M.T
    keep = np.ones(T - 1, bool)
    for bk in breaks:
        keep[bk - 1] = False
    med, grid = _sts_mu_map(X, breaks)
    r = np.interp(_sts_raw_rho(u, keep), med, grid, left=grid[0], right=np.inf)
    clipped = np.abs(r) > 0.95
    r = np.clip(r, -0.95, 0.95)
    e = (u[:, 1:] - r[:, None] * u[:, :-1])[:, keep]
    s2 = (e ** 2).sum(1) / max(int(keep.sum()) - k - 1, 1)
    c = _sts_lagweights(w)
    var = s2 / (1 - r ** 2) * ((r[:, None] ** np.arange(T)[None, :]) @ c)
    b = d @ w
    se = np.sqrt(np.maximum(var, 1e-300))
    return b, se, r, clipped, e


def _sts_calibrated(d: np.ndarray, T0: int | None, alpha: float, n_sim: int, rng):
    """Main route: t with the AR(1) standard error, referred to its own distribution simulated under a stationary
    AR(1) at the estimated ρ (ρ re-estimated in every simulated series, so the bias of ρ̂ is carried into the
    reference). p = (1 + #{|t*| ≥ |t|}) / (n_sim + 1); CI = estimate ± q_{1−α}(|t*|)·SE. T0 = None tests a linear
    trend (the slope) instead of the step."""
    T = len(d)
    X, w, breaks = _sts_design(T, T0)
    b, se, r, clipped, e = _sts_general(d[None, :], X, w, breaks)
    b, se, rho = float(b[0]), float(se[0]), float(r[0])
    x = np.empty((n_sim, T))
    z = rng.standard_normal((n_sim, T))
    x[:, 0] = z[:, 0] / np.sqrt(1 - rho ** 2)
    for t in range(1, T):
        x[:, t] = rho * x[:, t - 1] + z[:, t]
    bs, ses, _, _, _ = _sts_general(x, X, w, breaks)
    tabs = np.abs(bs / ses)
    t_obs = abs(b / se)
    crit = float(np.quantile(tabs, 1 - alpha))
    p = float((1 + np.sum(tabs >= t_obs - 1e-12)) / (n_sim + 1))
    return {"estimate": b, "se": se, "rho": rho, "rho_clipped": bool(clipped[0]), "crit": crit, "t": b / se, "p": p,
            "ci_low": b - crit * se, "ci_high": b + crit * se, "innovations": e[0]}


def _sts_ljung_box(e: np.ndarray, T0: int, lags: int):
    """Ljung-Box Q on the AR(1) innovations, autocorrelations pooled within the two segments (no cross-segment
    products); χ² reference with lags − 1 degrees of freedom (one AR coefficient estimated)."""
    seg = [e[:T0 - 1], e[T0 - 1:]]
    n = len(e)
    den = sum((s ** 2).sum() for s in seg)
    q = 0.0
    for k in range(1, lags + 1):
        rk = sum((s[k:] * s[:-k]).sum() for s in seg if len(s) > k) / den
        q += rk ** 2 / (n - k)
    q *= n * (n + 2)
    return float(q), float(stats.chi2.sf(q, lags - 1))


def _sts_regular(idx) -> bool:
    """Equally spaced dates or periods: a PeriodIndex by its ordinals; dates by an inferred frequency, or, failing
    that (mid-month dates, business days with holidays), by equal steps in days, months, quarters or years."""
    if isinstance(idx, pd.PeriodIndex):
        o = np.diff(idx.asi8)
        return bool(len(o) == 0 or (o == o[0]).all())
    di = pd.DatetimeIndex(idx)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if pd.infer_freq(di) is not None:
            return True
    days = np.diff(di.asi8) / 8.64e13
    if np.allclose(days, days[0]):
        return True
    months = np.diff(di.year * 12 + di.month)
    for step in (1, 3, 12):
        if (months == step).all():
            return True
    b = np.diff(np.asarray(pd.bdate_range(di.min(), di.max()).get_indexer(di)))
    return bool((b >= 1).all() and (b <= 2).all() and np.mean(b == 1) > 0.9)


def did_single_treated(df: pd.DataFrame, unit: str, time: str, y: str, treated, t0, controls: list | None = None,
                       alpha: float = ALPHA, n_sim: int = 1999, blocks: int = 4, seed: int = 0,
                       show: bool = True) -> dict:
    """Difference-in-differences when ONE unit (or a handful, averaged) is treated and the comparison comes from one
    or a few control units observed over a short series (a promotion in one of two stores over 24 months; a law in
    one state against a neighbour). With one treated unit, the unit bootstrap of att_gt and the permutation of
    randomization_inference cannot give a valid p-value (two units: two possible assignments, smallest p = 0.5): the
    information is in the time series of the gap g_t = mean(y of treated) − mean(y of controls), and inference has to
    account for its serial correlation with few periods.

    Estimand: mean of g over the post periods (t ≥ t0) minus its mean over the pre periods; with one treated and one
    control unit this is the 2×2 DiD of means. Identifying assumption: without treatment the gap would have stayed at
    its pre-period level (parallel trends between the treated and the controls), with the same dynamics before and
    after; nothing here tests it beyond the pre-period placebo below.

    Main route (calibrated AR(1) t, chosen by simulation in v2.5.1): the SE is the exact variance of the contrast
    under a stationary AR(1) for the gap's deviations, with ρ estimated within the two segments and made
    median-unbiased for this design (Andrews 1993, by simulation); the t ratio is referred to its own distribution simulated at that
    ρ, re-estimating ρ in each simulated series (n_sim draws). In the skill's tests (tests/test_v251.py: T = 24 to 60,
    pre = post or a 6-period pre, AR(1) ρ from 0 to 0.7, 2000 series each), the 5% test rejected 5.1–7.4% under no
    effect and the 95% CI covered 94%; HAC / Newey-West with 2 lags on the gap of the 2026-11-07 two-store case
    rejected 17.5% in the same test (17.8% in the 2000-panel recheck of that session): do not report HAC, Newey-West
    or Prais-Winsten p-values here. At least 6 pre and 6 post periods are required; below 20 periods the test rejected
    6-8% at ρ ≤ 0.4 and 8-9% at ρ = 0.7 (12 to 20 periods; warned).
    Second route: `blocks` block means per segment and a Welch t test between them (no AR(1) assumption; it
    over-rejects when ρ is high: 14% at ρ = 0.7, T = 24). Both have the same point estimate, so they are compared on
    the inference (does each reject at alpha?), not on the number.
    Limits: dynamics beyond AR(1) that are persistent (an AR(2) with coefficients 0.5 and 0.3) make both routes
    reject about 20% of the time under no effect; the Ljung-Box warning below catches it only sometimes with so few
    periods. Seasonality left in the gap makes both conservative (wide CIs): deseasonalize the gap first when the
    two units have different seasonal patterns.

    Pre-period checks, both with the same calibrated AR(1) route and an MDE (≈ (critical value + 0.84)·SE, 80% power)
    that says how large a departure they could have seen; not rejecting does not show parallel trends:
      pre_trend (at least 12 pre periods): slope of the gap on time over the pre periods. This is the check against a
        diverging trend: a step placebo is weak against one, because a trend inside a segment looks like persistence,
        inflates ρ̂ and widens its own CI (a step placebo rejected a pre-trend of 0.35 SD per period less than half
        the time, and a steeper one less often). Size 7-9% with 12 pre periods and ρ from 0 to 0.4, about 11% with ρ = 0.7
        (independent review of v2.5.1);
      pre_placebo (at least 16 pre periods, segments of 8; shorter ones over-rejected about 10%): a fake date at the
        midpoint of the pre periods (a level shift before treatment).

    df: long panel; treated: the label of the treated unit (or a list, averaged); t0: first treated period;
    controls: the control units (default: every other unit, averaged); they must be in the data and must not include a
    treated unit. Time must be a consecutive sequence (the
    AR(1) model assumes equally spaced periods); every unit used needs every period.
    Returns the output contract (estimand, estimate, se, ci_low, ci_high, p, n, method, warnings) for the main
    route, plus rho, critical value, the second route ("blocks"), routes_agree, pre_trend, pre_placebo, mde, the
    gap series and the Ljung-Box check."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    tr = list(treated) if isinstance(treated, (list, tuple, set, np.ndarray, pd.Index)) else [treated]
    units = pd.unique(df[unit])
    missing_tr = [u for u in tr if u not in set(units)]
    if missing_tr:
        raise ValueError(f"did_single_treated: treated unit(s) not in the data: {missing_tr}")
    ctl = [u for u in units if u not in set(tr)] if controls is None else list(controls)
    if not ctl:
        raise ValueError("did_single_treated: no control unit")
    missing_ctl = [u for u in ctl if u not in set(units)]
    if missing_ctl:
        raise ValueError(f"did_single_treated: control unit(s) not in the data: {missing_ctl}")
    both = [u for u in ctl if u in set(tr)]
    if both:
        raise ValueError(f"did_single_treated: unit(s) {both} are both treated and control")
    x = df[df[unit].isin(tr + ctl)][[unit, time, y]].dropna()
    if x.duplicated([unit, time]).any():
        raise ValueError(f"did_single_treated: repeated ({unit}, {time}) rows")
    wide = x.pivot(index=time, columns=unit, values=y).sort_index()
    if wide.isna().any().any():
        bad = list(wide.columns[wide.isna().any()])
        raise ValueError(f"did_single_treated: unit(s) {bad} miss some periods; the gap needs every unit in every "
                         "period (drop the unit or the periods first)")
    periods = wide.index.to_numpy()
    w = []
    if pd.api.types.is_numeric_dtype(wide.index):
        if len(periods) > 1 and not np.allclose(np.diff(periods.astype(float)), np.diff(periods.astype(float))[0]):
            w.append(f"The {time} values are not equally spaced: the AR(1) model assumes one step between periods; "
                     "fill or recode the periods first.")
    elif pd.api.types.is_datetime64_any_dtype(wide.index) or isinstance(wide.index, pd.PeriodIndex):
        if len(periods) > 2 and not _sts_regular(wide.index):
            w.append(f"The {time} dates do not follow a regular frequency (a gap, or mixed spacing): the AR(1) model "
                     "assumes one step between periods; fill or recode the periods first.")
    else:
        w.append(f"The spacing of {time} was not checked (not numeric or dates): the AR(1) model assumes equally "
                 "spaced periods.")
    gap = wide[tr].mean(axis=1) - wide[ctl].mean(axis=1)
    post = periods >= t0
    if post.any() and post.argmax() != (~post).sum():
        raise ValueError("did_single_treated: t0 must split the periods into a pre block followed by a post block")
    T0, T1 = int((~post).sum()), int(post.sum())
    if min(T0, T1) < 6:
        raise ValueError(f"did_single_treated: {T0} pre and {T1} post periods; at least 6 of each are needed: with 5 "
                         "and 5 the calibrated test rejected 10-11% of the time for a 5% test at ρ ≥ 0.4 in the "
                         "skill's checks, and short gaps were often read as a unit root")
    g = gap.to_numpy(float)
    dev = np.r_[g[:T0] - g[:T0].mean(), g[T0:] - g[T0:].mean()]
    if np.max(np.abs(dev)) <= 1e-12 * np.max(np.abs(g)):
        raise ValueError("did_single_treated: the gap does not vary around its pre and post means (a constant or a "
                         "pure step): there is no noise to base inference on; report the difference itself")
    rng = np.random.default_rng(seed)
    main = _sts_calibrated(g, T0, alpha, n_sim, rng)
    q = int(min(blocks, T0, T1))
    blk = _sts_blocks(g, T0, q, alpha) if q >= 2 else None
    lags = int(min(6, max(2, (len(g) - 2) // 5)))
    lb_q, lb_p = _sts_ljung_box(main["innovations"], T0, lags)
    z80 = stats.norm.ppf(0.8)
    mde = (main["crit"] + z80) * main["se"]
    trend = None
    if T0 >= 12:
        tr_res = _sts_calibrated(g[:T0], None, alpha, n_sim, rng)
        trend = {"slope": tr_res["estimate"], "se": tr_res["se"], "p": tr_res["p"], "ci_low": tr_res["ci_low"],
                 "ci_high": tr_res["ci_high"], "mde": (tr_res["crit"] + z80) * tr_res["se"], "rho": tr_res["rho"],
                 "rho_clipped": tr_res["rho_clipped"]}
    placebo = None
    if T0 >= 16:
        h = T0 // 2
        pl = _sts_calibrated(g[:T0], h, alpha, n_sim, rng)
        placebo = {"fake_t0": periods[h], "estimate": pl["estimate"], "se": pl["se"], "p": pl["p"],
                   "ci_low": pl["ci_low"], "ci_high": pl["ci_high"], "mde": (pl["crit"] + z80) * pl["se"],
                   "rho": pl["rho"], "rho_clipped": pl["rho_clipped"]}
    rej_main = main["p"] <= alpha
    agree = None if blk is None or not np.isfinite(blk["p"]) else bool(rej_main == (blk["p"] <= alpha))
    if len(g) < 20:
        w.append(f"Only {len(g)} periods: below 20 the calibrated test rejected 6-8% of the time for a 5% test at "
                 "ρ ≤ 0.4 and 8-9% at ρ = 0.7 in the skill's checks (12 to 20 periods; 5-7% from 24 on); read the "
                 "p-value as a little optimistic.")
    if main["rho_clipped"]:
        w.append("The gap is close to a unit root (estimated ρ above 0.95, clipped): the effect is not distinguishable "
                 "from a drifting gap with this series; the CI is not reliable."
                 + (" With fewer than 20 periods this also happens with moderate persistence: in the skill's checks "
                    "it hit about 18% of gaps with ρ = 0.4 and 4% of white-noise gaps with 12 periods, and 5% and "
                    "0.5% with 16; the series is too short to tell." if len(g) < 20 else ""))
    elif main["rho"] > 0.5:
        w.append(f"The gap is persistent (ρ ≈ {main['rho']:.2f}): the CI is wide, and the block-means route "
                 "over-rejects at this persistence (14% for a 5% test at ρ = 0.7 with 24 periods in the skill's "
                 "simulations); rely on the main route.")
    if lb_p < alpha:
        w.append(f"The AR(1) residuals still show serial correlation (Ljung-Box p = {lb_p:.3f}, {lags} lags): with "
                 "persistent dynamics beyond AR(1) both routes over-reject (about 20% in the skill's simulations); "
                 "treat the p-value as optimistic and look for seasonality or a break in the gap.")
    if agree is False:
        w.append(f"The two routes disagree at alpha = {alpha:g} (main p = {main['p']:.3f}, block means p = "
                 f"{blk['p']:.3f}): the evidence depends on the model for the serial correlation; report both.")
    for nm, chk in (("pre-period trend test", trend), ("pre-period placebo", placebo)):
        if chk is not None and chk["rho_clipped"]:
            # the clip inflates the SE: a rejection is conservative and informative, a non-rejection is not
            if chk["p"] <= alpha:
                w.append(f"The {nm} found the pre-period gap close to a unit root (ρ above 0.95, clipped; "
                         f"{T0} pre periods), which inflates its standard error: it rejects anyway (p = "
                         f"{chk['p']:.3f}), so that finding stands and is if anything understated.")
            else:
                w.append(f"The {nm} found the pre-period gap close to a unit root (ρ above 0.95, clipped; "
                         f"{T0} pre periods): that check did not reject, but it is not informative either. Short "
                         "pre-periods with moderate persistence do this often (about 18% of 12-period gaps with "
                         "ρ = 0.4 in the skill's checks).")
    if trend is not None and trend["p"] <= alpha:
        w.append(f"The gap trends before treatment (slope {trend['slope']:.3g} per period, p = {trend['p']:.3f}): "
                 "parallel trends fail in the pre-period, and a post-minus-pre contrast picks up the trend.")
    if placebo is not None and placebo["p"] <= alpha:
        w.append(f"The pre-period placebo rejects (fake date {placebo['fake_t0']}, p = {placebo['p']:.3f}): the gap "
                 "already moved before treatment; the parallel-trends assumption is in doubt.")
    if n_sim + 1 < 20 / alpha:
        w.append(f"n_sim = {n_sim} is small for alpha = {alpha:g}: p-values move with the seed; use n_sim ≥ "
                 f"{int(20 / alpha) - 1}.")
    method = (f"DiD with {len(tr)} treated and {len(ctl)} control unit(s) on the gap series: AR(1) t calibrated by "
              f"simulation ({n_sim} draws)")
    out = {"estimand": "ATT (DiD of means: post minus pre of the treated−control gap)", "estimate": main["estimate"],
           "se": main["se"], "ci_low": main["ci_low"], "ci_high": main["ci_high"], "p": main["p"], "n": len(g),
           "method": method, "warnings": w, "rho": main["rho"], "critical_value": main["crit"], "t": main["t"],
           "mde": mde, "pre_periods": T0, "post_periods": T1, "treated": tr, "controls": ctl,
           "blocks": blk, "blocks_q": q if blk else None, "routes_agree": agree, "pre_trend": trend, "pre_placebo": placebo,
           "ljung_box": {"q": lb_q, "p": lb_p, "lags": lags}, "gap": gap, "n_sim": n_sim, "alpha": alpha}
    if show:
        lvl = f"{100 * (1 - alpha):g}%"
        print(f"\nDiD with one treated unit ({', '.join(map(str, tr))} vs {len(ctl)} control(s)), {T0} pre and {T1} "
              f"post periods")
        print(f"  Estimate {main['estimate']:.4g} (SE {main['se']:.3g}; {lvl} CI {main['ci_low']:.4g} to "
              f"{main['ci_high']:.4g}; p = {main['p']:.3f}); AR(1) ρ ≈ {main['rho']:.2f}, critical value "
              f"{main['crit']:.2f} instead of {stats.norm.ppf(1 - alpha / 2):.2f}")
        if blk:
            print(f"  Second route, {q} block means per segment + Welch t: {lvl} CI {blk['ci_low']:.4g} to "
                  f"{blk['ci_high']:.4g}; p = {blk['p']:.3f}" + ("" if agree is None else
                                                                ("; same verdict" if agree else "; DIFFERENT verdict")))
        print(f"  MDE (80% power, alpha {alpha:g}): {mde:.3g}")
        if trend:
            print(f"  Pre-period trend: slope {trend['slope']:.3g} per period, p = {trend['p']:.3f}, MDE "
                  f"{trend['mde']:.3g} per period (does not reject ≠ parallel trends hold)")
        if placebo:
            print(f"  Pre-period placebo (fake date {placebo['fake_t0']}): {placebo['estimate']:.4g}, p = "
                  f"{placebo['p']:.3f}, MDE {placebo['mde']:.3g} (does not reject ≠ parallel trends hold)")
        print("  HAC / Newey-West / Prais-Winsten p-values on this series over-reject with so few periods: not used.")
        for m in w:
            print("  WARNING: " + m)
    return out


# =============================================================================
def _distinct_arrangements(values) -> list:
    """Every distinct ordering of a multiset, in lexicographic order, generated directly.

    Deduplicating factorial(n) permutations gives the same list but costs n! steps even when only a
    handful of orderings are distinct: with 14 treated units and two adoption dates that is 87 billion
    steps for 210 arrangements, and randomization_inference hung with no message (v2.5.2 review, chat 3).
    Taking the smallest value still available at each position yields lexicographic order, so the
    arrangements come out in the same order as before and no p-value changes."""
    vals, counts = np.unique(np.asarray(values), return_counts=True)
    vals = [float(v) for v in vals]
    counts = [int(c) for c in counts]
    n, out, cur = len(values), [], []

    def walk():
        if len(cur) == n:
            out.append(tuple(cur))
            return
        for i, v in enumerate(vals):
            if counts[i]:
                counts[i] -= 1
                cur.append(v)
                walk()
                cur.pop()
                counts[i] += 1

    walk()
    return out


# Falsification matrix for a DiD (v2.5.1)
# =============================================================================
def did_falsification(df: pd.DataFrame, unit: str, time: str, y: str, gvar: str,
                      placebo_outcomes: list[str] | None = None, cluster: str | None = None, control: str = "never",
                      anticipation: int = 0, max_shift: int = 1, n_boot: int = 199, n_perm: int = 999,
                      alpha: float = ALPHA, seed: int = 0, show: bool = True) -> dict:
    """The Reviewer's falsification matrix for a binary, absorbing DiD (check the design with did_design first): every
    check the skill has, run on the same data, with what each checks, its p-value, the smallest departure it could
    have detected (MDE ≈ (z_{1−α/2} + z_{0.8})·SE, 80% power, where an SE exists) and what not rejecting does NOT
    show. The matrix never says "passed": a check that does not reject with an MDE at least as large as the headline
    ATT is marked uninformative (it could not have seen a problem of the size of the effect).
    Two kinds of rows: falsification checks (a finding when they reject) and inference checks on the effect itself
    (permutation, timing: a finding when they contradict the headline significance at alpha, e.g. att_gt
    significant and the permutation p above alpha).

    Rows (many treated units or clusters):
      pre-trends (Callaway-Sant'Anna)   joint Wald test of the pre-period cells of att_gt
      pre-trends (imputation)           did_imputation's test on untreated observations only
      placebo date                      att_gt on the periods before the first adoption, every treated unit given a
                                        fake date at their midpoint (at least 4 such periods)
      placebo outcome: <name>           att_gt on each variable in placebo_outcomes (outcomes the treatment cannot
                                        move, e.g. measured before it or unrelated to it), same specification
      permutation                       randomization_inference, dates permuted across units, or across clusters
                                        with cluster= (the level of assignment)
      timing (±max_shift)               did_timing_sensitivity: does the conclusion survive date errors?
      CS vs imputation                  did_compare: a gap at alpha points to the pre-trends
    With 1 to 3 treated units (or clusters) and one adoption date, the unit-level rows cannot work (the permutation
    has a handful of assignments, the unit bootstrap has no variation to draw from); the matrix uses
    did_single_treated instead: its pre-period trend test and step placebo, and placebo outcomes through the same
    route, and says why the other rows are left out.
    Each sub-check that cannot run on these data (a ValueError) is listed with the reason, not dropped; if the
    headline estimate itself cannot be computed, the matrix raises a ValueError saying why. att_gt's warnings are
    passed on. With 1 to 3 treated units adopting at different dates, the unit-level rows run but are flagged as
    unreliable (a warning). anticipation moves the start of the post period back in the gap-series branch too.
    Returns: table (one row per check), headline (att_gt overall or did_single_treated), rejected (names),
    uninformative (names), verdict, warnings."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if anticipation < 0 or int(anticipation) != anticipation:
        raise ValueError("did_falsification: anticipation must be a non-negative integer")
    po = list(placebo_outcomes or [])
    zq = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(0.8)
    d = df.copy()
    d[gvar] = d[gvar].fillna(0)
    key = cluster or unit
    g_key = d.groupby(key)[gvar].max()
    n_tr = int((g_key > 0).sum())
    dates = np.sort(d.loc[d[gvar] > 0, gvar].unique())
    rows, warns = [], []

    def row(check, what, est=np.nan, se=np.nan, p=np.nan, mde=np.nan, note="", ran=True, kind="falsification",
            clipped=False):
        if clipped:
            # clipping ρ at 0.95 inflates the SE, so a rejection is conservative and still informative; only a
            # non-rejection is inconclusive (v2.5.2 review: the note contradicted the row's own finding)
            rej = ran and np.isfinite(p) and p <= alpha
            note = (note + "; " if note else "") + ("ρ clipped at 0.95 (pre-period gap near a unit root): the "
                    "test is conservative, so this rejection stands" if rej else
                    "ρ clipped at 0.95 (pre-period gap near a unit root): not informative")
        rows.append({"check": check, "kind": kind, "what it checks": what, "estimate": est, "se": se, "p": p,
                     "mde": mde, "rejects": (bool(p <= alpha) if ran and np.isfinite(p) else None), "ran": ran,
                     "note": note, "clipped": bool(clipped)})

    few = 0 < n_tr <= 3 and len(dates) == 1
    if few:
        tr_units = list(d.loc[d[gvar] > 0, unit].unique())
        t_all = np.sort(d[time].unique())
        t0 = dates[0]
        if anticipation:
            # the anticipation periods are treated as post: the first possibly-affected period is g − anticipation
            t0 = t_all[max(0, int(np.searchsorted(t_all, dates[0])) - anticipation)]
        try:
            head = did_single_treated(d, unit, time, y, tr_units, t0, alpha=alpha, seed=seed, show=False)
        except ValueError as e:
            raise ValueError(f"did_falsification: the headline estimate (did_single_treated, {n_tr} treated) cannot "
                             f"be computed on these data: {e}") from e
        warns.extend(head["warnings"])
        if anticipation:
            warns.append(f"anticipation = {anticipation}: the gap-series headline puts the {anticipation} period(s) "
                         "before the recorded date in the post mean (an ATT from g − anticipation), while att_gt and "
                         "did_imputation report anticipation periods apart; the estimands differ.")
        headline = {"estimate": head["estimate"], "se": head["se"], "p": head["p"], "method": head["method"]}
        pt = head["pre_trend"]
        if pt:
            span = (head["pre_periods"] + head["post_periods"]) / 2
            row("pre-period trend (gap series)", "slope of the treated−control gap before treatment", pt["slope"],
                pt["se"], pt["p"], pt["mde"] * span,
                f"estimate = slope per period; MDE = smallest detectable slope × {span:g}, the bias it would add to "
                "the post-minus-pre contrast", clipped=pt.get("rho_clipped", False))
        else:
            row("pre-period trend (gap series)", "slope of the gap before treatment", ran=False,
                note="fewer than 12 pre periods (the trend test needs 12)")
        pp = head["pre_placebo"]
        if pp:
            row("placebo date (gap series)", f"level shift at a fake date ({pp['fake_t0']}) before treatment",
                pp["estimate"], pp["se"], pp["p"], pp["mde"], clipped=pp.get("rho_clipped", False))
        else:
            row("placebo date (gap series)", "level shift at a fake date before treatment", ran=False,
                note="fewer than 16 pre periods (the step placebo needs 16)")
        for v in po:
            try:
                r = did_single_treated(d, unit, time, v, tr_units, t0, alpha=alpha, seed=seed, show=False)
                row(f"placebo outcome: {v}", "an effect on an outcome the treatment cannot move", r["estimate"],
                    r["se"], r["p"], r["mde"])
            except ValueError as e:
                row(f"placebo outcome: {v}", "an effect on an outcome the treatment cannot move", ran=False,
                    note=f"not run: {e}")
        n_keys = int(d[key].nunique())
        from math import comb
        row("permutation", "is the effect large among reassignments of the adoption dates?", ran=False, kind="inference",
            note=f"left out: with {n_tr} treated of {n_keys} {'clusters' if cluster else 'units'} there are only "
                 f"{comb(n_keys, n_tr)} assignments (smallest p {1 / comb(n_keys, n_tr):.3g})")
        for name in ("pre-trends (Callaway-Sant'Anna)", "pre-trends (imputation)", "CS vs imputation"):
            row(name, "unit-level estimators", ran=False,
                note="left out: the unit bootstrap and cluster SEs need more treated units; the gap-series rows "
                     "above replace them")
        row(f"timing (±{max_shift})", "does the conclusion survive date errors?", ran=False, kind="inference",
            note=f"left out: the date-robust estimator of did_timing_sensitivity needs more treated units; re-run "
                 f"did_single_treated with t0 moved by ±{max_shift} periods instead")
        warns.append(f"Only {n_tr} treated {'cluster' if cluster else 'unit'}(s): the matrix uses the gap series "
                     "(did_single_treated); unit-level rows are left out and say why."
                     + (f" control='{control}' does not apply there (the controls are every untreated unit)."
                        if control != "never" else ""))
    else:
        kw = dict(control=control, anticipation=anticipation, n_boot=n_boot, seed=seed, cluster=cluster, alpha=alpha,
                  show=False)
        base = att_gt(d, unit, time, y, gvar, **kw)
        o = base["overall"]
        warns.extend(f"att_gt: {m}" for m in base["warnings"])
        if 0 < n_tr <= 3:
            warns.append(f"Only {n_tr} treated {'cluster' if cluster else 'unit'}(s), adopting at {len(dates)} dates: "
                         "the unit-level rows below (bootstrap, permutation, placebo date, timing) are not reliable "
                         "with so few treated units, and the gap-series route needs one date. Analyse each treated "
                         "unit with did_single_treated against the never-treated units, and read this matrix as "
                         "descriptive only.")
        headline = {"estimate": o["estimate"], "se": o["se"], "p": o["p"], "method": "att_gt"}
        row("pre-trends (Callaway-Sant'Anna)", f"joint Wald test of the pre-period cells ({base['pretrend'].get('test')})",
            p=base["pretrend_p"] if base["pretrend_p"] is not None else np.nan,
            note="joint test: no single MDE; with few clusters it over-rejects (see att_gt warnings)")
        try:
            bj = did_imputation(d, unit, time, y, gvar, anticipation=anticipation, cluster=cluster, alpha=alpha,
                                show=False)
            row("pre-trends (imputation)", "leads on untreated observations only (Borusyak-Jaravel-Spiess)",
                p=bj["pretrend_p"] if bj["pretrend_p"] is not None else np.nan, note="joint test: no single MDE")
        except ValueError as e:
            row("pre-trends (imputation)", "leads on untreated observations", ran=False, note=f"not run: {e}")
        pre_t = np.sort(d.loc[d[time] < dates.min(), time].unique())
        if len(pre_t) >= 4:
            fake = pre_t[len(pre_t) // 2]
            dp = d[d[time] < dates.min()].copy()
            dp["_gfake"] = np.where(dp[gvar] > 0, fake, 0)
            try:
                r = att_gt(dp, unit, time, y, "_gfake", **{**kw, "anticipation": 0})["overall"]
                row("placebo date", f"an 'effect' at a fake date ({fake}) before any unit is treated", r["estimate"],
                    r["se"], r["p"], zq * r["se"])
            except ValueError as e:
                row("placebo date", "an effect at a fake date before treatment", ran=False, note=f"not run: {e}")
        else:
            row("placebo date", "an effect at a fake date before treatment", ran=False,
                note=f"only {len(pre_t)} periods before the first adoption (4 needed)")
        for v in po:
            try:
                r = att_gt(d, unit, time, v, gvar, **kw)["overall"]
                row(f"placebo outcome: {v}", "an effect on an outcome the treatment cannot move", r["estimate"],
                    r["se"], r["p"], zq * r["se"])
            except ValueError as e:
                row(f"placebo outcome: {v}", "an effect on an outcome the treatment cannot move", ran=False,
                    note=f"not run: {e}")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ri = randomization_inference(d, unit, time, y, gvar, n_perm=n_perm, ci=False, seed=seed, alpha=alpha,
                                             cluster=cluster)
            row("permutation", "is the effect large among reassignments of the adoption dates" +
                (f" across {cluster}s" if cluster else "") + "? (TWFE statistic)", ri["estimate"], p=ri["p"], kind="inference",
                note=f"{ri['possible_assignments']} possible assignments" + ("; " + "; ".join(ri["warnings"])
                                                                            if ri["warnings"] else ""))
        except ValueError as e:
            row("permutation", "reassignments of the adoption dates", ran=False, note=f"not run: {e}")
        try:
            ts = did_timing_sensitivity(d, unit, time, y, gvar, max_shift=max_shift, control=control, n_boot=n_boot,
                                        seed=seed, cluster=cluster, alpha=alpha, show=False)
            dr = ts["date_robust"]
            rows.append({"clipped": False,         # the one row built by hand: without it the column is NaN
                         "check": f"timing (±{max_shift})", "kind": "inference",
                         "what it checks": "does the conclusion survive date errors?",
                         "estimate": dr["estimate"], "se": dr["se"], "p": dr["p"], "mde": np.nan,
                         "rejects": bool(dr["p"] <= alpha) if np.isfinite(dr["p"]) else None, "ran": True, "note": ("survives" if ts["survives"] else "the conclusion depends on the exact "
                                               "date") + "; " + ts["estimand_note"]})
        except ValueError as e:
            row(f"timing (±{max_shift})", "does the conclusion survive date errors?", ran=False, note=f"not run: {e}",
                kind="inference")
        try:
            cmp_ = did_compare(d, unit, time, y, gvar, control=control, anticipation=anticipation, cluster=cluster,
                               n_boot=n_boot, seed=seed, alpha=alpha, show=False)
            row("CS vs imputation", "gap between two estimators with different comparisons (points to pre-trends)",
                cmp_["difference"], cmp_["diff_se"], cmp_["p"], zq * cmp_["diff_se"])
        except ValueError as e:
            row("CS vs imputation", "gap between two estimators", ran=False, note=f"not run: {e}")
    tab = pd.DataFrame(rows).set_index("check")
    att = abs(headline["estimate"])
    head_sig = bool(np.isfinite(headline["p"]) and headline["p"] <= alpha)
    # a falsification check is a finding when it rejects; an inference check when it contradicts the headline
    tab["finding"] = [None if r["rejects"] is None else (bool(r["rejects"]) if r["kind"] == "falsification"
                                                         else bool(r["rejects"] != head_sig))
                      for _, r in tab.iterrows()]
    tab["informative"] = [None if r["kind"] != "falsification" or not r["ran"] or r["rejects"]
                          else False if r["clipped"]
                          else (bool(r["mde"] < att) if np.isfinite(r["mde"]) else None) for _, r in tab.iterrows()]
    tab = tab.drop(columns="clipped")
    rejected = [c for c, r in tab.iterrows() if r["finding"]]
    uninf = [c for c, r in tab.iterrows() if r["informative"] is False]
    not_run = [c for c, r in tab.iterrows() if not r["ran"]]
    if rejected:
        verdict = (f"{len(rejected)} finding(s) at alpha = {alpha:g}: {', '.join(rejected)} (a falsification check "
                   "that rejects, or an inference check that contradicts the headline). Report them next to the ATT.")
    else:
        verdict = (f"No falsification check rejects and no inference check contradicts the headline at alpha = "
                   f"{alpha:g}. That does not show the design is valid"
                   + (f"; {len(uninf)} check(s) could not have seen a departure the size of the ATT "
                      f"(MDE ≥ |ATT|, or ρ clipped): {', '.join(uninf)}" if uninf else "")
                   + (f"; {len(not_run)} could not run: {', '.join(not_run)}" if not_run else "") + ".")
    out = {"table": tab, "headline": headline, "rejected": rejected, "uninformative": uninf, "not_run": not_run,
           "verdict": verdict, "few_treated": few, "alpha": alpha, "warnings": warns}
    if show:
        print(f"\nFalsification matrix (alpha = {alpha:g}; headline ATT {headline['estimate']:.4g}, "
              f"{headline['method']})")
        show_cols = ["kind", "estimate", "p", "mde", "finding", "informative", "note"]
        with pd.option_context("display.max_colwidth", 70, "display.width", 200):
            print(tab[show_cols].to_string(float_format=lambda v: f"{v:.4g}"))
        print("  → " + verdict)
        print("  Not rejecting a check never proves parallel trends or a valid design.")
        for m in warns:
            print("  WARNING: " + m)
    return out
