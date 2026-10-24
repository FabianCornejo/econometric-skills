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


def cox(df: pd.DataFrame, formula: str, duration: str, event: str, ties: str = "efron",
        strata: str | None = None, cluster: str | None = None, robust: bool = True, show: bool = True):
    """Cox proportional hazards. formula = right-hand side only, e.g. "age + C(sex) + C(region)".
    No intercept is estimated (the baseline hazard absorbs it), so do not add '0 +'.
    cluster: groups for a sandwich (Lin-Wei) variance; robust=True uses each row as its
    own group when no cluster is given. Returns (result, hazard-ratio table)."""
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
