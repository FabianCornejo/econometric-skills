"""
Duration (survival / time-to-event) analysis with statsmodels: how long until a case
is decided, a civil servant leaves, a loan defaults, a customer churns, a patient dies.

Why not OLS on durations: spells still in progress at the end of the data are
right-censored (we only know duration > observed time). Dropping them or treating
the censoring time as the event time biases everything downward.

Functions
---------
kaplan_meier(df, duration, event, group)     survival curves, median survival, survival at chosen times
logrank(df, duration, event, group)          H0: identical survival curves across groups
cox(df, formula, duration, event, ...)       Cox PH: hazard ratios with CIs, robust/clustered SE, strata
ph_test(result, df, duration)                proportional-hazards check via scaled Schoenfeld residuals vs time
person_period(df, id, duration, event, ...)  expand to discrete-time format (logit/cloglog hazard models)

Requires statsmodels; `lifelines` (optional) adds parametric AFT models and plots.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats


def kaplan_meier(df: pd.DataFrame, duration: str, event: str, group: str | None = None,
                 times: list[float] | None = None, show: bool = True) -> dict:
    import statsmodels.api as sm
    groups = [(None, df)] if group is None else list(df.groupby(group))
    curves, summ = {}, []
    for g, sub in groups:
        sf = sm.SurvfuncRight(sub[duration].astype(float), sub[event].astype(int))
        tab = sf.summary()
        curves[g if g is not None else "all"] = tab
        row = {"group": g if g is not None else "all", "n": len(sub), "events": int(sub[event].sum()),
               "censored_share": 1 - sub[event].mean()}
        try:
            row["median_survival"] = sf.quantile(0.5)
            lo, hi = sf.quantile_ci(0.5)
            row["median_ci_low"], row["median_ci_high"] = lo, hi
        except Exception:
            row["median_survival"] = np.nan
        for t in times or []:
            s = tab.loc[tab.index <= t, "Surv prob"]
            row[f"S({t})"] = float(s.iloc[-1]) if len(s) else 1.0
        summ.append(row)
    summary = pd.DataFrame(summ).set_index("group")
    if show:
        print("\nKaplan-Meier")
        print(summary.to_string(float_format=lambda v: f"{v:.3f}"))
        if (summary["censored_share"] > 0.5).any():
            print("  More than half the spells are censored: the median may be undefined or imprecise.")
    return {"summary": summary, "curves": curves}


def logrank(df: pd.DataFrame, duration: str, event: str, group: str, strata: str | None = None) -> dict:
    from statsmodels.duration.survfunc import survdiff
    chi2, p = survdiff(df[duration].astype(float), df[event].astype(int), df[group],
                       strata=None if strata is None else df[strata])
    return {"chi2": chi2, "df": df[group].nunique() - 1, "p": p,
            "conclusion": "survival differs across groups" if p < 0.05 else "no significant difference"}


def _cox_score_residuals(time: np.ndarray, status: np.ndarray, X: np.ndarray, beta: np.ndarray,
                         strata: np.ndarray | None = None, ties: str = "efron") -> np.ndarray:
    """Score residuals of the Cox partial likelihood (Therneau & Grambsch 2000, sec. 3.7), by stratum, with Efron or
    Breslow handling of tied event times. For an event time u with d tied events, risk-set sums S0_j, S1_j for
    j = 0..d-1 remove the fraction j/d of the tied events (Efron; j/d = 0 for Breslow). A tied event's own term is
    x_i minus the mean of the d weighted means; the compensator gives full weight to the rest of the risk set and
    weight 1 - j/d to the tied events. A row censored before the first event of its stratum gets 0 (no event time
    at or before it). The residuals add up to the score of the same likelihood (0 at its maximum).
    (statsmodels' PHReg.score_residuals uses Breslow increments with any ties setting and returns NaN for rows
    censored before the first event; v2.6.7 computes them here.)"""
    n, p = X.shape
    out = np.zeros((n, p))
    strata = np.zeros(n) if strata is None else np.asarray(strata)
    efron = ties == "efron"
    for s in pd.unique(strata):
        ix = np.flatnonzero(strata == s)
        t, e, x = time[ix], status[ix].astype(bool), X[ix]
        u_times = np.unique(t[e])
        if len(u_times) == 0:
            continue
        eta = x @ beta
        w = np.exp(eta - eta.max())                        # the scale cancels in every ratio below
        o = np.argsort(t, kind="mergesort")
        ts = t[o]
        cw = np.cumsum(w[o][::-1])[::-1]                   # sum of w over t >= ts[k]
        cwx = np.cumsum((w[o, None] * x[o])[::-1], axis=0)[::-1]
        K = len(u_times)
        a_r, a_d = np.zeros(K), np.zeros(K)
        b_r, b_d, xbar_ev = np.zeros((K, p)), np.zeros((K, p)), np.zeros((K, p))
        for k, u in enumerate(u_times):
            j0 = np.searchsorted(ts, u, side="left")
            tied = (t == u) & e
            dd = int(tied.sum())
            s0_d, s1_d = w[tied].sum(), (w[tied, None] * x[tied]).sum(axis=0)
            f = np.arange(dd) / dd if efron else np.zeros(dd)
            s0 = cw[j0] - f * s0_d
            xb = (cwx[j0][None, :] - f[:, None] * s1_d[None, :]) / s0[:, None]
            a_r[k], b_r[k] = (1 / s0).sum(), (xb / s0[:, None]).sum(axis=0)
            a_d[k], b_d[k] = ((1 - f) / s0).sum(), ((1 - f)[:, None] * xb / s0[:, None]).sum(axis=0)
            xbar_ev[k] = xb.mean(axis=0)
        ca, cb = np.cumsum(a_r), np.cumsum(b_r, axis=0)
        kk = np.searchsorted(u_times, t, side="right")    # number of event times <= t_i
        has = kk > 0
        A = np.where(has, ca[np.maximum(kk - 1, 0)], 0.0)
        B = np.where(has[:, None], cb[np.maximum(kk - 1, 0)], 0.0)
        ke = kk[e] - 1                                     # an event's own time
        A[e] += a_d[ke] - a_r[ke]
        B[e] += b_d[ke] - b_r[ke]
        r = -(w[:, None] * (x * A[:, None] - B))
        r[e] += x[e] - xbar_ev[ke]
        out[ix] = r
    return out


def _cox_sandwich(mod, res, groups, ties: str):
    """Lin-Wei sandwich I^-1 (sum over groups of s_g s_g') I^-1 from _cox_score_residuals. Clusters are coded with
    pd.factorize (labels of any type); a missing cluster label is an error, not a cluster of its own."""
    from statsmodels.duration.hazard_regression import PHRegResults
    codes, _ = pd.factorize(pd.Series(np.asarray(groups, dtype=object)), sort=False)
    if (codes < 0).any():
        raise ValueError(f"cluster has {(codes < 0).sum()} missing label(s): drop those rows or fill the cluster id")
    params = np.asarray(res.params)
    strata = np.asarray(mod.strata) if getattr(mod, "strata", None) is not None else None
    sr = _cox_score_residuals(np.asarray(mod.endog, float), np.asarray(mod.status), np.asarray(mod.exog, float),
                              params, strata, ties)
    sg = np.zeros((codes.max() + 1, sr.shape[1]))
    np.add.at(sg, codes, sr)
    h_inv = np.linalg.inv(mod.hessian(params))
    cov = h_inv @ (sg.T @ sg) @ h_inv
    return PHRegResults(mod, params, cov, covariance_type="robust")


def cox(df: pd.DataFrame, formula: str, duration: str, event: str, ties: str = "efron",
        strata: str | None = None, cluster: str | None = None, robust: bool = True, show: bool = True):
    """Cox proportional hazards. formula = right-hand side only, e.g. "age + C(sex) + C(region)".
    No intercept is estimated (the baseline hazard absorbs it), so do not add '0 +'.
    cluster: groups for a sandwich (Lin-Wei) variance; robust=True uses each row as its
    own group when no cluster is given. Returns (result, hazard-ratio table).

    The sandwich is computed here from score residuals of the same likelihood (v2.6.7, _cox_score_residuals):
    statsmodels' own robust covariance was NaN whenever a row is censored before the first event of its stratum
    (about a quarter of samples with early censoring) and, with tied event times and ties="efron", too small
    (it uses Breslow increments). Checked against R survival::coxph(robust = TRUE) with ties, strata and clusters
    (tools/r_reference/cox_v267_reference.R). A missing cluster label raises an error."""
    import statsmodels.formula.api as smf
    d = df.dropna(subset=[duration, event]).copy()
    kw = {"status": d[event].astype(int), "ties": ties}
    if strata:
        kw["strata"] = d[strata]
    mod = smf.phreg(f"{duration} ~ {formula}", d, **kw)
    fit_kw = {}
    if cluster is not None:
        fit_kw["groups"] = d[cluster]
    elif robust:
        fit_kw["groups"] = np.arange(len(d))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = mod.fit(**fit_kw)
        if "groups" in fit_kw:
            res = _cox_sandwich(mod, res, np.asarray(fit_kw["groups"]), ties)
    names = mod.exog_names
    b = np.asarray(res.params)
    se = np.asarray(res.bse)
    z = stats.norm.ppf(0.975)
    tab = pd.DataFrame({"coef": b, "HR": np.exp(b), "HR_ci_low": np.exp(b - z * se),
                        "HR_ci_high": np.exp(b + z * se), "se": se,
                        "p": 2 * (1 - stats.norm.cdf(np.abs(b / se)))}, index=names)
    if show:
        print(f"\nCox PH ({ties} ties; SE {'clustered by ' + cluster if cluster else 'robust' if robust else 'model-based'}"
              + (f"; stratified by {strata}" if strata else "") + f"); N = {len(d)}, events = {int(d[event].sum())}")
        print(tab.to_string(float_format=lambda v: f"{v:.4f}"))
        print("  HR > 1: higher hazard (event happens sooner); HR < 1: lower hazard (longer duration).")
        print("  A HR of 1.25 means a 25% higher instantaneous event rate, not a 25% shorter duration.")
    res.hr_table = tab
    return res, tab


def ph_test(res, df: pd.DataFrame, duration: str, event: str, transform: str = "rank") -> pd.DataFrame:
    """Proportional-hazards check in the spirit of Grambsch & Therneau (1994):
    correlate each covariate's Schoenfeld residuals (at event times) with a function
    of time. A significant correlation means the hazard ratio changes over time.
    Remedies: stratify on that variable, interact it with time, or split the period."""
    d = df.dropna(subset=[duration, event])
    sch = np.asarray(res.schoenfeld_residuals)
    ev = d[event].astype(int).to_numpy() == 1
    t = d[duration].astype(float).to_numpy()
    names = res.model.exog_names
    rows = []
    for j, nm in enumerate(names):
        r = sch[:, j]
        m = ev & np.isfinite(r)
        tt = stats.rankdata(t[m]) if transform == "rank" else (np.log(t[m]) if transform == "log" else t[m])
        rho, p = stats.pearsonr(r[m], tt)
        rows.append({"covariate": nm, "rho": rho, "p": p, "PH_violated_5%": p < 0.05})
    tab = pd.DataFrame(rows).set_index("covariate")
    print("\nProportional-hazards check (Schoenfeld residuals vs time)")
    print(tab.to_string(float_format=lambda v: f"{v:.4f}"))
    return tab


def person_period(df: pd.DataFrame, id: str, duration: str, event: str, interval: float = 1.0) -> pd.DataFrame:
    """Expand spells to one row per unit and period at risk (discrete-time hazard).
    Then estimate e.g. smf.logit("y ~ C(period) + x", pp) or a cloglog GLM; time-varying
    covariates can be merged on (id, period)."""
    rows = []
    for _, r in df.iterrows():
        k = int(np.ceil(r[duration] / interval))
        for p in range(1, max(k, 1) + 1):
            rows.append({id: r[id], "period": p, "y": int(r[event] == 1 and p == k)})
    pp = pd.DataFrame(rows)
    return pp.merge(df.drop(columns=[duration, event]), on=id, how="left")
