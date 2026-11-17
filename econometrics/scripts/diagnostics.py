"""Econometric diagnostic batteries with plain-language conclusions.

Use as a module:
    import sys; sys.path.insert(0, "<skill folder>/scripts")
    from diagnostics import *

Use from the command line (CSV or XLSX; optional date column via --date):
    python3 diagnostics.py ols data.csv --y log_sales --x log_price,log_income [--hac 6 | --ts]
    python3 diagnostics.py influence data.csv --y log_sales --x log_price,log_income
    python3 diagnostics.py chow data.csv --y log_sales --x log_price --break 2020-03-01
    python3 diagnostics.py stationarity data.csv --cols policy_rate,cpi,log_sales
    python3 diagnostics.py cointegration data.csv --cols log_p,log_fx,log_pforeign [--lags 2]

Main functions
    ols_diagnostics(res)              RESET, Breusch-Pagan, White, simplified White, JB, VIF;
                                      time_series=True adds DW, Breusch-Godfrey, ARCH-LM
    influence(res)                    leverage, DFBETA, max|h*e~| (Hansen)
    chow(y, X, point)                 Chow test for a break at a known date (plus robust Wald)
    compare_standard_errors(...)      classical vs HC3 vs HAC
    comparison_table({...})           several specifications side by side
    stationarity(series)              ADF + KPSS + DF-GLS, levels and first difference
    cointegration(df)                 Engle-Granger + Johansen
    endogeneity_test(...)             regression-based Durbin-Wu-Hausman (control function)
    overid_test(...)                  Wooldridge nR² and Sargan
    iv_report(formula, data)          2SLS vs OLS, first-stage partial F (weak/acceptable/strong), endogeneity
                                      and overid tests, Anderson-Rubin confidence set (one endogenous regressor;
                                      joint AR test of β = 0 with several)
    hausman(fe, re)                   FE vs RE (linearmodels)
    panel_bp_lm(...)                  pooled vs random effects (Breusch-Pagan LM)
    mundlak_test(df, y, x_vary, ...)  FE vs RE via correlated random effects, clustered Wald
    count_diagnostics(poisson_res)    Pearson dispersion, Cameron-Trivedi NB1/NB2 tests, excess zeros
    tobit(y, X, lower, upper)         Tobit with lower/upper/two-limit censoring; 3 AMEs with delta SEs
    diebold_mariano(e1, e2, h)        equal forecast accuracy (HLN correction)

Default significance level: 5%.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize, stats
from statsmodels.stats.diagnostic import (
    acorr_breusch_godfrey,
    het_arch,
    het_breuschpagan,
    het_white,
    linear_reset,
)
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.tsa.stattools import adfuller, coint, kpss
from statsmodels.tsa.tsatools import add_trend
import _console  # noqa: F401  (Windows-safe console output, see _console.py)

ALPHA = 0.05


def _row(test, h0, stat, p, conclusion):
    return {"Test": test, "H0": h0, "Statistic": stat, "p-value": p, "Conclusion": conclusion}


def _fmt(v):
    return f"{v:.4f}"


def newey_west_lags(n: int) -> int:
    """Newey-West rule of thumb for the number of HAC lags."""
    return max(1, int(np.floor(4 * (n / 100) ** (2 / 9))))


def _plain_ols(res):
    """Re-fit the same model with classical covariance (tests use residuals and X only)."""
    return sm.OLS(res.model.endog, res.model.exog).fit()


# =============================================================================
# OLS
# =============================================================================
def white_simplified(res):
    """White test using ŷ and ŷ² (saves degrees of freedom). Returns (LM, p)."""
    base = _plain_ols(res)
    e2 = base.resid ** 2
    yhat = base.fittedvalues
    Z = sm.add_constant(np.column_stack([yhat, yhat ** 2]))
    aux = sm.OLS(e2, Z).fit()
    lm = len(e2) * aux.rsquared
    return lm, 1 - stats.chi2.cdf(lm, 2)


def ols_diagnostics(res, bg_lags: int | None = None, time_series: bool = False,
                    arch_lags: int = 4, show: bool = True) -> pd.DataFrame:
    """Full battery for a statsmodels OLS result (any cov_type).

    Order: functional form → heteroskedasticity → autocorrelation → normality.
    The chosen standard-error type does not change the diagnostics.

    time_series=False (cross-section): Durbin-Watson and Breusch-Godfrey are NOT run, because
    they depend on the row order, which is arbitrary in a cross-section (sorting the data by a
    regressor would create spurious "autocorrelation"). If the order is meaningful (e.g. units
    sorted by distance), pass time_series=True deliberately.
    time_series=True: adds Durbin-Watson, Breusch-Godfrey, Engle's ARCH-LM on the residuals
    (arch_lags lags) and the R² > DW spurious-regression warning.
    """
    base = _plain_ols(res)
    resid = np.asarray(base.resid)
    exog = np.asarray(base.model.exog)
    n = len(resid)
    bg_lags = bg_lags or (12 if time_series else newey_west_lags(n))
    rows = []

    try:
        reset = linear_reset(base, power=3, use_f=True)
        p = float(reset.pvalue)
        rows.append(_row("Ramsey RESET (ŷ², ŷ³)", "correct functional form", float(reset.fvalue), p,
                         "Likely misspecification: respecify BEFORE fixing standard errors"
                         if p < ALPHA else "No evidence of misspecification"))
    except Exception as e:  # noqa: BLE001
        rows.append(_row("Ramsey RESET", "correct functional form", np.nan, np.nan, f"not computed ({e})"))

    try:   # v2.6.12: statsmodels refuses Breusch-Pagan without a constant column; that stopped the whole battery
        lm, p_bp, _, _ = het_breuschpagan(resid, exog)
        rows.append(_row("Breusch-Pagan", "homoskedasticity", lm, p_bp,
                         "Heteroskedasticity: use HC3 (or WLS/FGLS)" if p_bp < ALPHA else "No evidence"))
    except Exception as e:  # noqa: BLE001
        rows.append(_row("Breusch-Pagan", "homoskedasticity", np.nan, np.nan, f"not computed ({e})"))
    try:
        with warnings.catch_warnings():  # squares of dummies duplicate columns: expected, not a problem
            warnings.simplefilter("ignore")
            lm_w, p_w, _, _ = het_white(resid, exog)
        rows.append(_row("White (full)", "homoskedasticity", lm_w, p_w,
                         "Heteroskedasticity" if p_w < ALPHA else "No evidence"))
    except Exception:  # noqa: BLE001  (fails with many regressors/dummies)
        pass
    lm_ws, p_ws = white_simplified(res)
    rows.append(_row("White simplified (ŷ, ŷ²)", "homoskedasticity", lm_ws, p_ws,
                     "Heteroskedasticity" if p_ws < ALPHA else "No evidence"))

    dw = np.nan
    if time_series:
        dw = durbin_watson(resid)
        rows.append(_row("Durbin-Watson", "no 1st-order autocorrelation (≈2)", dw, np.nan,
                         "Positive autocorrelation likely" if dw < 1.5 else
                         "Negative autocorrelation likely" if dw > 2.5 else "Close to 2"))
        try:
            try:
                bg = acorr_breusch_godfrey(base, nlags=bg_lags, result_object=False)
            except TypeError:  # older statsmodels
                bg = acorr_breusch_godfrey(base, nlags=bg_lags)
            lm_bg, p_bg = bg[0], bg[1]
            rows.append(_row(f"Breusch-Godfrey ({bg_lags} lags)", "no autocorrelation", lm_bg, p_bg,
                             "Autocorrelation: use HAC and/or model the dynamics (or omitted variable/functional form)"
                             if p_bg < ALPHA else "No evidence"))
        except Exception:  # noqa: BLE001
            pass

        try:
            lm_a, p_a, _, _ = het_arch(resid, nlags=arch_lags)
            rows.append(_row(f"ARCH-LM ({arch_lags} lags)", "no conditional heteroskedasticity (ARCH)", lm_a, p_a,
                             "ARCH effects: volatility clustering; HC/HAC SEs remain valid, model the variance "
                             "(GARCH, see volatility.md) if it matters for intervals or risk"
                             if p_a < ALPHA else "No evidence"))
        except Exception:  # noqa: BLE001
            pass

    jb, p_jb, skew, kurt = jarque_bera(resid)
    rows.append(_row("Jarque-Bera", "normal residuals", jb, p_jb,
                     (f"Non-normal (skewness {skew:.2f}, kurtosis {kurt:.2f}); "
                      + ("relevant: small sample" if n < 100 else "of little concern with large N"))
                     if p_jb < ALPHA else "No evidence against normality"))

    if time_series and np.isfinite(dw) and base.rsquared > dw:
        rows.append(_row("R² > DW", "—", base.rsquared, np.nan,
                         "SPURIOUS REGRESSION warning: test stationarity/cointegration"))

    table = pd.DataFrame(rows)
    vif = vif_table(res)
    if show:
        print(f"\nOLS diagnostics (N={n}, R²={base.rsquared:.3f}, adj. R²={base.rsquared_adj:.3f}, "
              f"condition number={np.linalg.cond(exog):.1f})")
        print(table.to_string(index=False, float_format=_fmt))
        if not time_series:
            print("Cross-section: autocorrelation tests skipped (row order is arbitrary). "
                  "For time series, pass time_series=True.")
        if not vif.empty:
            print("\nVIF (>10 worrying; >5 watch):")
            print(vif.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    return table


def vif_table(res) -> pd.DataFrame:
    """VIF of each non-constant regressor, 1/(1 − R²_j), with a column `Kind` saying which one.

    - Model with a constant column (any name): centered VIF, R²_j from the regression of x_j on the other
      non-constant regressors plus a constant (Stata's `estat vif`).
    - Model without a constant column: uncentered VIF, R²_j uncentered, from the regression of x_j on the other
      regressors without a constant (Stata's `estat vif, uncentered`; Stata refuses the centered one after `nocons`).
      With a full set of dummies and no constant the centered VIF is infinite by construction, not collinearity.
    v2.6.11: computed here, the same on every statsmodels version (0.14 returned the uncentered VIF for models
    without a constant, 0.15 the centered one). The rules of thumb (> 5 watch, > 10 worrying) are for the centered
    VIF; the uncentered one is not comparable with it (it can be larger or smaller) and has no such threshold.
    """
    exog = np.asarray(res.model.exog, dtype=float)
    names = list(res.model.exog_names)
    def _is_const(col):   # allclose, tighter than specification._has_intercept's defaults; never all zero
        return bool(np.allclose(col, col[0], rtol=1e-10, atol=1e-12)) and abs(col[0]) > 1e-12
    const_cols = [i for i in range(exog.shape[1]) if _is_const(exog[:, i])]
    centered = len(const_cols) > 0
    keep = [i for i in range(exog.shape[1])
            if i not in const_cols and not np.allclose(exog[:, i], 0, atol=1e-12)]
    rows = []
    for i in keep:
        others = [exog[:, j] for j in keep if j != i]
        x = exog[:, i]
        if centered:
            Z = np.column_stack([np.ones(len(exog))] + others)
            tss = float(((x - x.mean()) ** 2).sum())
        else:
            Z = np.column_stack(others) if others else np.zeros((len(exog), 0))
            tss = float((x ** 2).sum())
        ssr = float(((x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]) ** 2).sum()) if Z.shape[1] else tss
        vif = np.inf if ssr <= 1e-12 * tss else tss / ssr
        rows.append({"Variable": names[i], "VIF": vif, "Kind": "centered" if centered else "uncentered"})
    return pd.DataFrame(rows, columns=["Variable", "VIF", "Kind"])


def influence(res, top: int = 10, show: bool = True) -> pd.DataFrame:
    """Most influential observations: leverage h_ii, |h_ii·ẽ_i| (Hansen) and the largest |DFBETA|.

    ẽ_i = leave-one-out prediction error = ê_i/(1−h_ii). There is no magic threshold:
    the question is whether the observation materially changes the coefficient of interest.
    """
    base = _plain_ols(res)
    inf = base.get_influence()
    h = inf.hat_matrix_diag
    e_loo = base.resid / (1 - h)
    dfb = pd.DataFrame(inf.dfbetas, columns=list(res.model.exog_names),
                       index=getattr(res.model.data, "row_labels", None))
    out = pd.DataFrame({
        "leverage_h": h,
        "influence_|h·ẽ|": np.abs(h * e_loo),
        "max_|DFBETA|": dfb.abs().max(axis=1).values,
        "most_affected_coef": dfb.abs().idxmax(axis=1).values,
        "cook": inf.cooks_distance[0],
    }, index=dfb.index).sort_values("influence_|h·ẽ|", ascending=False)
    if show:
        k, n = base.model.exog.shape[1], len(h)
        print(f"\nInfluence (N={n}). Mean leverage = k/n = {k/n:.3f}; "
              f"watch h > {2*k/n:.3f} (2k/n) and |DFBETA| > {2/np.sqrt(n):.3f} (2/√n).")
        print(f"Hansen influence statistic (max |h·ẽ|) = {out['influence_|h·ẽ|'].iloc[0]:.4f} "
              f"— compare with the standard deviation of y ({np.std(base.model.endog):.4f}).")
        print(out.head(top).to_string(float_format=_fmt))
        print("Check for recording errors; if the observations are legitimate, report results with and without them.")
    return out


def chow(y, X, point, show: bool = True) -> dict:
    """Chow test for a break at a known point.

    y: Series; X: DataFrame (no constant; it is added). point: index label where the
    second regime starts. The classical F is sensitive to heteroskedasticity, so a robust
    version (interactions with a regime dummy + HC3 Wald) is also reported.
    """
    data = pd.concat([y, X], axis=1).dropna()
    yy, XX = data.iloc[:, 0], sm.add_constant(data.iloc[:, 1:])
    post = data.index >= point
    k = XX.shape[1]
    n1, n2 = (~post).sum(), post.sum()
    if min(n1, n2) <= k:
        raise ValueError("Not enough observations in one of the regimes.")
    ssr = sm.OLS(yy, XX).fit().ssr
    ssr1 = sm.OLS(yy[~post], XX[~post]).fit().ssr
    ssr2 = sm.OLS(yy[post], XX[post]).fit().ssr
    F = ((ssr - ssr1 - ssr2) / k) / ((ssr1 + ssr2) / (n1 + n2 - 2 * k))
    p = 1 - stats.f.cdf(F, k, n1 + n2 - 2 * k)
    D = post.astype(float)
    XI = pd.concat([XX, XX.mul(D, axis=0).add_suffix("_x_post")], axis=1)
    rob = sm.OLS(yy, XI).fit(cov_type="HC3")
    R = np.zeros((k, XI.shape[1])); R[:, k:] = np.eye(k)
    w = rob.wald_test(R, scalar=True)
    out = {"F": F, "p": p, "robust_wald": float(w.statistic), "robust_p": float(w.pvalue),
           "n1": int(n1), "n2": int(n2)}
    if show:
        print(f"\nChow at {point}: F({k},{n1+n2-2*k}) = {F:.3f}, p = {p:.4f}; "
              f"robust Wald (HC3) = {out['robust_wald']:.3f}, p = {out['robust_p']:.4f} → robust test: "
              + ("structural break" if out["robust_p"] < ALPHA else "no evidence of a break"))
    return out


def compare_standard_errors(formula: str, data: pd.DataFrame, hac_lags: int | None = None) -> pd.DataFrame:
    """Same model with classical, HC3 and (optionally) HAC errors: shows how much inference changes."""
    import statsmodels.formula.api as smf
    versions = {"Classical": smf.ols(formula, data).fit(),
                "HC3": smf.ols(formula, data).fit(cov_type="HC3")}
    if hac_lags:
        versions[f"HAC({hac_lags})"] = smf.ols(formula, data).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    parts = [pd.DataFrame({"coef": r.params, f"se_{n}": r.bse, f"p_{n}": r.pvalues}) for n, r in versions.items()]
    out = pd.concat(parts, axis=1)
    return out.loc[:, ~out.columns.duplicated()]


def comparison_table(results: dict, stars: bool = True):
    """Paper-style table with several specifications (statsmodels OLS results)."""
    from statsmodels.iolib.summary2 import summary_col
    return summary_col(list(results.values()), model_names=list(results.keys()), stars=stars,
                       info_dict={"N": lambda r: f"{int(r.nobs)}",
                                  "Adj. R²": lambda r: f"{getattr(r, 'rsquared_adj', np.nan):.3f}"})


# =============================================================================
# Stationarity
# =============================================================================
def _tests(x: pd.Series, reg: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf = adfuller(x, regression=reg, autolag="AIC")
        kp = kpss(x, regression=reg, nlags="auto")
        try:
            from arch.unitroot import DFGLS
            dg = DFGLS(x, trend=reg, method="bic")
            dg_stat, dg_p = dg.stat, dg.pvalue
        except Exception:  # noqa: BLE001
            dg_stat = dg_p = np.nan
    return adf[0], adf[1], adf[2], kp[0], kp[1], dg_stat, dg_p


KPSS_P_BOUNDS = (0.01, 0.10)   # statsmodels interpolates the KPSS p-value inside this table only


def _verdict(p_adf, p_kpss, p_dfgls, alpha: float = ALPHA):
    p_ur = p_dfgls if not np.isnan(p_dfgls) else p_adf   # DF-GLS has more power (Stock)
    # the KPSS p-value at the lower bound means "≤ 0.01": a rejection at any alpha ≥ 0.01 (v2.4.7.2)
    # and at the upper bound "≥ 0.10": never a rejection, whatever alpha (the true p-value is unknown above it)
    rej_ur = p_ur < alpha
    lo, hi = KPSS_P_BOUNDS
    if p_kpss >= hi:
        rej_kpss = False
    elif p_kpss <= lo:
        rej_kpss = alpha >= lo
    else:
        rej_kpss = p_kpss < alpha
    if rej_ur and not rej_kpss:
        return "Stationary"
    if not rej_ur and rej_kpss:
        return "Unit root"
    if rej_ur and rej_kpss:
        return "Conflict (trend/break?)"
    return "Inconclusive (high persistence or long memory?)"


def stationarity(series: pd.Series, name: str | None = None, show: bool = True,
                 alpha: float = ALPHA) -> pd.DataFrame:
    """ADF, KPSS and DF-GLS with constant and with trend, in levels and first difference.

    The verdict uses DF-GLS (more power) + KPSS (opposite null), both at level alpha (v2.4.7.2; default 0.05).
    The KPSS p-value is interpolated from a table bounded at [0.01, 0.10]: values at the bounds mean '≤0.01' or
    '≥0.10', so the KPSS decision is only exact for alpha in [0.01, 0.10]; outside that range a warning is
    attached (tab.attrs["warnings"]) and a p-value at the bound is not counted as a rejection.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    warns = []
    lo, hi = KPSS_P_BOUNDS
    if alpha < lo or alpha > hi:
        warns.append(f"alpha = {alpha:g} is outside [{lo:g}, {hi:g}], the range of the KPSS p-value table: a KPSS "
                     "p-value at the table bound cannot be compared with alpha, and the verdict treats it as not "
                     "rejecting")
    name = name or (series.name if series.name is not None else "series")
    x = pd.Series(series).dropna()
    rows = []
    for label, data in (("level", x), ("1st difference", x.diff().dropna())):
        for reg, desc in (("c", "constant"), ("ct", "constant+trend")):
            a, pa, lags, k, pk, g, pg = _tests(data, reg)
            rows.append({"Series": name, "Transform": label, "Deterministics": desc,
                         "ADF": a, "p ADF": pa, "lags": lags, "DF-GLS": g, "p DF-GLS": pg,
                         "KPSS": k, "p KPSS": pk, "Verdict": _verdict(pa, pk, pg, alpha)})
    tab = pd.DataFrame(rows)

    def v(transf, det):
        return tab.loc[(tab["Transform"] == transf) & (tab["Deterministics"] == det), "Verdict"].iloc[0]
    vc, vt = v("level", "constant"), v("level", "constant+trend")
    if vc == "Stationary":
        order = "I(0)"
    elif vt == "Stationary":
        order = ("I(0) around a deterministic trend (or with a break) — with a constant only the tests "
                 "conflict/reject; check that the trend is significant and plausible")
    elif v("1st difference", "constant") == "Stationary":
        # v2.6.12 (v2.6.11 validation): the summary said a bare "I(1)" while a level line said "Conflict"
        order = ("I(1) by the differences, but the level tests conflict (trend or break?): check the level lines"
                 if "Conflict" in (vc + vt) else "I(1)")
    else:
        order = "undetermined — inspect the plot; test for a break (Zivot-Andrews) or long memory (ARFIMA)"
    tab.attrs["suggested_order"] = order
    tab.attrs["alpha"] = alpha
    tab.attrs["warnings"] = warns
    if show:
        print(f"\nStationarity — {name} (N={len(x)}, alpha = {alpha:g}): {order}")
        print(tab.drop(columns="Series").to_string(index=False, float_format=_fmt))
        for w in warns:
            print("Warning:", w)
    return tab


def stationarity_many(df: pd.DataFrame, show: bool = True, alpha: float = ALPHA) -> pd.DataFrame:
    summary = []
    for c in df.columns:
        t = stationarity(df[c], c, show=show, alpha=alpha)
        summary.append({"Series": c, "Suggested order": t.attrs["suggested_order"]})
    out = pd.DataFrame(summary)
    if show:
        print("\nSummary:\n" + out.to_string(index=False))
    return out


# =============================================================================
# Cointegration
# =============================================================================
FULL_RANK_DET0 = ("Full rank with det_order=0 (statsmodels' unrestricted constant): this case declares full rank "
                  "too often for random walks without drift (time-series.md §7), so do not conclude I(0) from it. "
                  "Run multivariate.johansen_trace(df, k_ar_diff, deterministic='ci'), and revisit stationarity.")


def _engle_granger(y, xs, trend: str, autolag: str) -> dict:
    """statsmodels' coint (statistic and MacKinnon p-value) plus the lag its residual ADF chose.

    coint runs adfuller(resid, autolag=..., regression="n") on the residuals of y on xs and the trend terms;
    the same ADF is re-run here only to read the lag. If its statistic does not match coint's, the lag is
    reported as unknown (None) rather than guessed.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p, _ = coint(y, xs, trend=trend, autolag=autolag)
        X = add_trend(np.asarray(xs, dtype=float), trend=trend, prepend=True)
        resid = sm.OLS(np.asarray(y, dtype=float), X).fit().resid
        adf = adfuller(resid, autolag=autolag, regression="n")
    lags = int(adf[2]) if np.isclose(adf[0], stat, rtol=1e-8, atol=1e-10) else None
    return {"statistic": float(stat), "p": float(p), "lags": lags}


def cointegration(df: pd.DataFrame, k_ar_diff: int = 1, det_order: int = 0, show: bool = True) -> dict:
    """Engle-Granger (1st column as dependent) and Johansen (trace and max-eigenvalue, 5%).

    det_order: -1 no deterministics, 0 constant, 1 linear trend.
    k_ar_diff: lags in differences (choose with vecm.select_order).
    Prerequisite: all series I(1).
    Engle-Granger is run with the residual-ADF lags chosen by AIC and by BIC (v2.6.11). The keys
    statistic/p are the AIC run (as before); statistic_bic/p_bic/lags_aic/lags_bic are added. When the two
    disagree at 5%, lag_sensitive is True and the conclusion is "Inconclusive" instead of either verdict
    (Wooldridge 2000, 1st ed., p. 587-588, runs the basic test on r6 and r3 with no ADF lags and finds them clearly
    cointegrated; on the same data, the skill's AIC run picks 6 lags and does not reject).
    """
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    data = df.dropna()
    y, xs = data.iloc[:, 0], data.iloc[:, 1:]
    trend = "c" if det_order <= 0 else "ct"
    eg = {crit: _engle_granger(y, xs, trend, crit) for crit in ("aic", "bic")}
    eg_t, eg_p = eg["aic"]["statistic"], eg["aic"]["p"]
    lag_sensitive = (eg["aic"]["p"] < ALPHA) != (eg["bic"]["p"] < ALPHA)
    if lag_sensitive:
        eg_concl = ("Inconclusive: the conclusion depends on the lag choice of the residual ADF "
                    "(AIC vs BIC); report both")
    else:
        eg_concl = "Cointegrated" if eg_p < ALPHA else "No evidence of cointegration"
    joh = coint_johansen(data.values, det_order, k_ar_diff)
    k = data.shape[1]
    rows, r_tr, r_mx = [], None, None
    for r in range(k):
        tr, tr_cv, mx, mx_cv = joh.lr1[r], joh.cvt[r, 1], joh.lr2[r], joh.cvm[r, 1]
        rows.append({"H0: rank ≤": r, "Trace": tr, "5% CV trace": tr_cv, "Reject (trace)": tr > tr_cv,
                     "Max eigen": mx, "5% CV max": mx_cv, "Reject (max)": mx > mx_cv})
        if r_tr is None and tr <= tr_cv:
            r_tr = r
        if r_mx is None and mx <= mx_cv:
            r_mx = r
    r_tr = k if r_tr is None else r_tr
    r_mx = k if r_mx is None else r_mx
    tab = pd.DataFrame(rows)
    johansen_note = None
    if max(r_tr, r_mx) == k and det_order == 0:
        johansen_note = FULL_RANK_DET0
    elif r_tr == k:
        johansen_note = "Full rank: the series are probably I(0) — revisit stationarity."
    out = {"engle_granger": {"statistic": eg_t, "p": eg_p, "conclusion": eg_concl,
                             "lags_aic": eg["aic"]["lags"], "statistic_bic": eg["bic"]["statistic"],
                             "p_bic": eg["bic"]["p"], "lags_bic": eg["bic"]["lags"],
                             "lag_sensitive": lag_sensitive},
           "johansen": tab, "rank_trace": r_tr, "rank_max_eigen": r_mx, "johansen_note": johansen_note}
    if show:
        def _lags(v):
            return "?" if v is None else str(v)
        print(f"\nEngle-Granger ({y.name} ~ {', '.join(map(str, xs.columns))}), ADF on the residuals "
              f"(MacKinnon critical values):")
        for crit in ("aic", "bic"):
            print(f"  lags chosen by {crit.upper()}: {_lags(eg[crit]['lags'])} → t={eg[crit]['statistic']:.3f}, "
                  f"p={eg[crit]['p']:.4f}")
        print(f"  → {eg_concl}")
        if lag_sensitive:
            print("  Check the autocorrelation of the residual-ADF regression"
                  + (" and run multivariate.johansen_trace(..., deterministic='ci'): the Johansen result below is "
                     "not usable here (see the note)." if johansen_note == FULL_RANK_DET0
                     else " and weigh the Johansen result below."))
        print(f"Johansen (det_order={det_order}, k_ar_diff={k_ar_diff}):")
        print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(f"Cointegration rank: trace = {r_tr}, max eigenvalue = {r_mx}"
              + ("" if r_tr == r_mx else " (they differ: report both and decide on theory)"))
        if johansen_note:
            print(johansen_note)
    return out


# =============================================================================
# Endogeneity and IV
# =============================================================================
def _first_stage_label(f: float) -> str:
    """Label of a first-stage F in endogeneity_test: > 104.7 strong (Lee, McCrary, Moreira & Porter 2022, the
    tF threshold for a 5% test without adjustment), 10 to 104.7 acceptable, below 10 weak (Staiger-Stock). v2.6.12:
    F = 10 is "acceptable", as in iv_report (it was "WEAK" here and "acceptable" there)."""
    if f == np.inf:
        return "perfect first stage (no endogenous variation left; see the warning)"
    return "strong" if f > 104.7 else ("acceptable (10 ≤ F ≤ 104.7); also report Anderson-Rubin" if f >= 10 else "WEAK")


# v2.6.12 (claude/medicoes-v2.6.12.md, E3): with HC0, HC1 or HC2 and fewer residual degrees of freedom than this, a
# first-stage F above 10 on pure noise had probability above 5% (HC0 up to 15 df with 8 instruments); HC3 never did
_ROBUST_SMALL_DF = 30


def _rank_tol(s_max: float, n: int) -> float:
    """Tolerance for a singular value to count as a direction (v2.6.12, E4): eps·s_max·max(100, √n). Rounding of
    columns that are exact combinations stayed below 15·eps·s_max (n up to 1e5), while z + 1e-12·e stayed above
    590·eps·s_max at every n; numpy's default, eps·s_max·max(n, k), dropped z + 1e-10·e when n = 1e5."""
    return float(s_max) * max(100.0, np.sqrt(n)) * np.finfo(float).eps


def _scaled_centred(M):
    """Columns centred and divided by their largest entry (all-zero columns drop out), and the amplification of
    the rounding error that centring a column with a large offset leaves (max|raw| / max|centred|)."""
    M = np.asarray(M, dtype=float)
    C = M - M.mean(axis=0)
    m = np.max(np.abs(C), axis=0) if C.shape[1] else np.zeros(0)
    keep = m > 0
    amp = float(np.max(np.max(np.abs(M), axis=0)[keep] / m[keep])) if np.any(keep) else 1.0
    return C[:, keep] / m[keep], amp


def endogeneity_test(y, X_exog, x_endog, Z, cov_type: str = "HC3", show: bool = True) -> dict:
    """Regression-based Durbin-Wu-Hausman test (control function), as in Wooldridge.

    1) x_endog ~ X_exog + Z (reduced form) → residuals v̂
    2) y ~ x_endog + X_exog + v̂ by OLS; v̂ significant ⇒ x_endog is endogenous.
    y: named Series; X_exog: DataFrame (no constant) or None; x_endog: DataFrame with 1+ columns;
    Z: DataFrame of excluded instruments. Also reports the first-stage F of the excluded instruments.
    Column names must be distinct across y, X_exog, x_endog and Z (v2.6.12: a repeated name raises ValueError;
    before, the instrument was silently dropped).
    Instruments are counted by rank (v2.6.9): columns that are linear combinations of each other or of the
    exogenous regressors add nothing, and "n_instruments" is the number that do; "warnings" says when some don't.
    The first-stage F tests those n_instruments directions (Z net of the exogenous regressors), so a redundant
    column changes neither the F nor the test. With fewer instruments than endogenous regressors the model is not
    identified and the test is NaN (no first-stage line is printed); with no residual degrees of freedom left
    (saturated first stage) the F and the test are NaN too. Rows with NaN are dropped; inf raises ValueError.
    Degenerate data (v2.6.12; they gave "strong" or "endogenous" before):
    - an endogenous regressor that is an exact linear combination of the exogenous regressors and the instruments
      (nothing left in its first-stage residual): first_stage_F = inf, the test is NaN, and a warning says so;
    - a row with leverage 1 in the first stage or in the test regression (an instrument, or a combination of them,
      that lives on that row alone, such as a dummy with a single 1): the robust covariance (HC0-HC3) is
      degenerate there, so the robust F and test are NaN; the warning names the rows and gives the nonrobust F;
    - HC0, HC1 or HC2 with fewer than 30 residual degrees of freedom: a warning that the robust F and test reject
      too often (simulated: P(F > 10) above 5% with no first stage at all); HC3, the default, did not.
    Keys: first_stage_F (dict by endogenous regressor), statistic, p, n_instruments, warnings, high_leverage_rows
    (index labels of the rows with leverage 1; empty if none), perfect_first_stage (list of regressors).
    """
    names = ([y.name] + (list(X_exog.columns) if X_exog is not None else []) + list(x_endog.columns)
             + list(Z.columns))
    if y.name is None:
        raise ValueError("endogeneity_test: y must be a named Series (y.name is used to find it in the data)")
    rep = sorted({str(c) for c in names if names.count(c) > 1})
    if rep:
        raise ValueError("endogeneity_test: column name(s) " + ", ".join(rep) + " appear in more than one of y, "
                         "X_exog, x_endog and Z; a variable is either exogenous, endogenous or an excluded "
                         "instrument (rename the columns if they hold different data)")
    parts = [y, x_endog, Z] if X_exog is None else [y, X_exog, x_endog, Z]
    d = pd.concat(parts, axis=1).dropna()
    yy = d[y.name]
    Xe = sm.add_constant(d[X_exog.columns]) if X_exog is not None and len(X_exog.columns) else \
        pd.DataFrame({"const": np.ones(len(d))}, index=d.index)
    # One rule counts the excluded instruments and picks the directions the first-stage F tests (v2.6.9):
    # - W: orthonormal basis Q_W of the exogenous regressors (centred, the constant put back, columns divided by
    #   their largest entry). Using Q_W keeps the span, so v̂, the F and the test are unchanged, and the leverages
    #   of HC2/HC3 stay accurate with near-collinear W.
    # - Z: centred (the constant is always in W), divided by the largest entry, projected off Q_W.
    # - n_z = number of singular values of that residual above the tolerance (_rank_tol, v2.6.12; and a floor for
    #   the rounding left by centring a column with a large offset, v2.6.9 round 3). z and 2z count once; z and 3w
    #   (w exogenous) count once, and the F is the F of z alone.
    if not np.all(np.isfinite(d.to_numpy(dtype=float))):
        raise ValueError("endogeneity_test: the data contain inf; replace or drop those rows first")
    n_obs = len(d)
    eps = np.finfo(float).eps
    robust = str(cov_type).lower() != "nonrobust"
    Wc, _ = _scaled_centred(Xe.to_numpy(dtype=float))
    Wm = np.column_stack([np.ones(n_obs), Wc])
    UW, sW, _ = np.linalg.svd(Wm, full_matrices=False)
    r_w = int(np.sum(sW > _rank_tol(sW.max(), n_obs))) if sW.size else 0
    QW = UW[:, :r_w]

    def _new_dirs(M, Q, s_ref):
        """Directions of M (scaled, centred) outside span(Q): left singular vectors and their count."""
        Mc, amp = _scaled_centred(M)
        if not Mc.shape[1]:
            return np.zeros((n_obs, 0)), 0
        Mr = Mc - Q @ (Q.T @ Mc)
        U, s, _ = np.linalg.svd(Mr, full_matrices=False)
        s_all = max(s_ref, np.linalg.svd(Mc, compute_uv=False).max())
        tol = max(_rank_tol(s_all, n_obs), 2 * eps * amp * (np.sqrt(n_obs) + np.sqrt(Mc.shape[1])))
        k = int(np.sum(s > tol))
        return U[:, :k], k

    s_w = sW.max() if sW.size else 0.0
    UZ, n_z = _new_dirs(d[Z.columns].to_numpy(dtype=float), QW, s_w)
    n_x = len(x_endog.columns)
    warn = []
    if n_z < len(Z.columns):
        warn.append(f"{len(Z.columns)} instrument column(s), but only {n_z} add information beyond each other and the "
                    "exogenous regressors (the rest are constant or linear combinations): the model is counted with "
                    + (f"{n_z} instrument(s), and the first-stage F tests only those." if n_z >= n_x and n_z else
                       f"{n_z} instrument(s)." if n_z else "no instrument, so there is no first stage to test."))
    fs_saturated = n_z > 0 and n_obs - r_w - n_z <= 0
    test_saturated = n_obs - r_w - 2 * n_x <= 0
    if fs_saturated or (test_saturated and n_z >= n_x):
        warn.append(f"{'the first stage' if fs_saturated else 'the test regression'} is saturated ({n_obs} rows for "
                    f"{r_w} exogenous directions, {n_z} instrument(s) and {n_x} endogenous regressor(s)): no residual "
                    "degrees of freedom, so " + ("the first-stage F and the test are" if fs_saturated else
                                                 "the test is") + " not computed.")
    saturated = fs_saturated or test_saturated
    Wq = pd.DataFrame(QW, index=d.index, columns=[f"_wdir{k + 1}" for k in range(r_w)])
    Zeff = pd.DataFrame(UZ, index=d.index, columns=[f"_zdir{k + 1}" for k in range(n_z)])
    QWZ = np.column_stack([QW, UZ])
    # v2.6.12 (E2): an endogenous regressor with no direction outside span(W, Z) has a first-stage residual made of
    # rounding (F of 1e32 and "endogenous" before); the same rank rule decides it
    perfect = [c for c in x_endog.columns
               if n_z > 0 and _new_dirs(d[[c]].to_numpy(dtype=float), QWZ, np.linalg.svd(QWZ, compute_uv=False).max()
                                        if QWZ.size else 0.0)[1] == 0]
    if perfect:
        warn.append(", ".join(perfect) + (" is" if len(perfect) == 1 else " are") + " an exact linear combination of "
                    "the exogenous regressors and the instruments: the first stage fits perfectly (F = inf), no "
                    "endogenous variation is left, and the Durbin-Wu-Hausman test is not defined (NaN). Check how the "
                    "variable was built.")
    # v2.6.12 (E1): leverages of the first stage are the squared row norms of its orthonormal basis [Q_W, Z dirs]
    lev_rows = []
    if robust and n_z > 0 and not fs_saturated:
        h = np.sum(QWZ ** 2, axis=1)
        lev_rows = list(d.index[h > 1 - 1e-8])
    resids, f_first, f_nonrobust = {}, {}, {}
    with warnings.catch_warnings():         # E5: statsmodels' rank warnings; the function reports its own
        warnings.simplefilter("ignore")
        for c in x_endog.columns:
            if n_z > 0 and not fs_saturated and c not in perfect:
                X1 = pd.concat([Wq, Zeff], axis=1)
                R = np.zeros((n_z, X1.shape[1])); R[:, -n_z:] = np.eye(n_z)
                fs_nr = sm.OLS(d[c], X1).fit()
                f_nonrobust[c] = float(fs_nr.wald_test(R, scalar=True, use_f=True).statistic)
                if lev_rows:
                    f_first[c] = float("nan")
                else:
                    fs = sm.OLS(d[c], X1).fit(cov_type=cov_type)
                    f_first[c] = float(fs.wald_test(R, scalar=True, use_f=True).statistic)
                resids[f"vhat_{c}"] = fs_nr.resid
            else:
                fs_nr = sm.OLS(d[c], pd.concat([Wq, Zeff], axis=1) if n_z > 0 else Wq).fit()
                f_first[c] = float("inf") if c in perfect and not fs_saturated else float("nan")
                resids[f"vhat_{c}"] = fs_nr.resid
        V = pd.DataFrame(resids)
        identified = n_z >= n_x and n_z > 0 and not saturated and not perfect
        test_lev = []
        if identified:
            Xt = pd.concat([Wq, d[x_endog.columns], V], axis=1)
            if robust and not lev_rows:
                Ut, st, _ = np.linalg.svd(Xt.to_numpy(dtype=float), full_matrices=False)
                Ut = Ut[:, : int(np.sum(st > _rank_tol(st.max(), n_obs)))]
                test_lev = list(d.index[np.sum(Ut ** 2, axis=1) > 1 - 1e-8])
            if robust and (lev_rows or test_lev):
                stat_, p_ = float("nan"), float("nan")
            else:
                est = sm.OLS(yy, Xt).fit(cov_type=cov_type)
                Rv = np.zeros((V.shape[1], est.params.shape[0])); Rv[:, -V.shape[1]:] = np.eye(V.shape[1])
                w = est.wald_test(Rv, scalar=True, use_f=True)
                stat_, p_ = float(w.statistic), float(w.pvalue)
        else:
            stat_, p_ = float("nan"), float("nan")
    high_lev = list(dict.fromkeys(lev_rows + test_lev))
    if high_lev:
        shown = ", ".join(str(i) for i in high_lev[:5]) + (" …" if len(high_lev) > 5 else "")
        nr = "; ".join(f"{c}: {f:.2f}" for c, f in f_nonrobust.items())
        warn.append(f"row(s) {shown} have leverage 1 in the {'first stage' if lev_rows else 'test regression'}: an "
                    "instrument (or a combination of them) lives on those rows alone, so its residual there is zero and "
                    f"the {cov_type} covariance is degenerate; the robust "
                    + ("first-stage F and the test are" if lev_rows else "test is") + " not computed (NaN)"
                    + (f" (nonrobust first-stage F: {nr})" if nr else "") + ". Drop those rows or the instrument "
                    "that only they identify, then test again.")
    df_res = n_obs - r_w - n_z
    if (str(cov_type).upper() in ("HC0", "HC1", "HC2") and n_z >= n_x and n_z > 0 and not saturated
            and df_res < _ROBUST_SMALL_DF and not high_lev):
        warn.append(f"{cov_type} with {df_res} residual degrees of freedom in the first stage: the robust F and the "
                    "test reject too often with few degrees of freedom (simulated with no first stage at all, "
                    "P(F > 10) above 5%); use cov_type='HC3' (the default)"
                    + (f"; nonrobust first-stage F: {'; '.join(f'{c}: {f:.2f}' for c, f in f_nonrobust.items())}"
                       if f_nonrobust else "") + ".")
    out = {"first_stage_F": f_first, "statistic": stat_, "p": p_, "n_instruments": n_z,
           "warnings": warn, "high_leverage_rows": high_lev, "perfect_first_stage": perfect}
    if show:
        for m in warn:
            print("\nWarning:", m)
        for c, f in (f_first.items() if n_z >= n_x and n_z > 0 and not fs_saturated else ()):
            if np.isnan(f):
                continue
            print(f"\nFirst stage for {c}: F of excluded instruments = {f:.2f} → {_first_stage_label(f)}")
        weak = any(not f >= 10 for f in f_first.values())          # NaN counts as weak, as its label says
        multi = len(f_first) > 1
        joint = (f" With {len(f_first)} endogenous regressors, each first-stage F above does not show that they are "
                 "jointly identified: check Shea's partial R² (linearmodels IV2SLS(...).fit().first_stage) or the "
                 "Sanderson-Windmeijer F before using 2SLS.") if multi else ""
        if n_z < n_x:
            verdict = (f"not identified: {n_z} excluded instrument(s) for {n_x} endogenous regressors, so IV/2SLS "
                       "is impossible and this test is not informative.")
        elif saturated:
            verdict = "not computed: the regressions are saturated (see the warning)."
        elif perfect:
            verdict = "not defined: the first stage fits perfectly (see the warning)."
        elif high_lev:
            verdict = f"not computed with {cov_type}: rows with leverage 1 (see the warning)."
        else:
            if out["p"] < ALPHA:
                # v2.6.8: LIML equals 2SLS when the model is exactly identified, so it is offered only with more
                # instruments than endogenous regressors (v2.6.7 validation, notebook 1)
                robust_inf = ("weak-instrument-robust inference (Anderson-Rubin) or LIML." if n_z > n_x else
                              "weak-instrument-robust inference (Anderson-Rubin); LIML does not help here: with as "
                              "many instruments as endogenous regressors it equals 2SLS.")
                verdict = ("endogenous, but with weak instruments (first-stage F < 10) 2SLS is unreliable: use "
                           + robust_inf if weak else
                           "endogenous" + ("." if multi else ": use IV/2SLS."))
            else:
                verdict = ("no evidence of endogeneity, but with weak instruments (first-stage F < 10) the test has "
                           "little power." if weak else
                           "no evidence of endogeneity (not proof of exogeneity): OLS is more efficient.")
            verdict += joint
        print(f"Durbin-Wu-Hausman (control function, {cov_type}): F = {out['statistic']:.3f}, p = {out['p']:.4f} → "
              + verdict)
    return out


def overid_test(resid_2sls, included_exog: pd.DataFrame | None, instruments: pd.DataFrame,
                n_endog: int, show: bool = True) -> dict:
    """Overidentifying restrictions test (Wooldridge nR²; Sargan (n−k)R²).

    Regresses the 2SLS residuals on all included exogenous variables and excluded instruments.
    q = number of excluded instruments − number of endogenous regressors (must be ≥ 1).
    Assumes homoskedasticity; with heteroskedasticity use Hansen's J (IVGMM ... .j_stat).
    """
    parts = [instruments] if included_exog is None else [included_exog, instruments]
    Z = sm.add_constant(pd.concat(parts, axis=1))
    aux = sm.OLS(np.asarray(resid_2sls), Z).fit()
    n, k = len(resid_2sls), Z.shape[1]
    q = instruments.shape[1] - n_endog
    if q < 1:
        raise ValueError("Exactly identified model: there are no overidentifying restrictions to test.")
    nr2, sargan = n * aux.rsquared, (n - k) * aux.rsquared
    p = 1 - stats.chi2.cdf(nr2, q)
    if show:
        print(f"\nOveridentification: nR² = {nr2:.3f} (Sargan (n−k)R² = {sargan:.3f}), χ²({q}), p = {p:.4f} → "
              + ("at least one instrument is invalid" if p < ALPHA else
                 "no evidence against validity (this does not prove validity)"))
    return {"nR2": nr2, "sargan": sargan, "q": q, "p": p}


def _ar_pvalue(yv, xv, W, Z, b0, robust):
    """Anderson-Rubin test of H0: β = b0 for one endogenous regressor: regress y − b0·x on W and Z, F on Z."""
    return _ar_test(yv - b0 * xv, W, Z, robust)[1]


def _ar_test(yy, W, Z, robust):
    """F and p of the excluded instruments Z in the regression of yy on [W, Z] (HC1 if robust)."""
    X = np.column_stack([W, Z])
    fit = sm.OLS(yy, X).fit(cov_type="HC1" if robust else "nonrobust")
    R = np.zeros((Z.shape[1], X.shape[1])); R[:, W.shape[1]:] = np.eye(Z.shape[1])
    w = fit.wald_test(R, scalar=True, use_f=True)
    return float(w.statistic), float(w.pvalue)


def _formula_rows(formula: str, data: pd.DataFrame) -> pd.DataFrame:
    """Rows of data with no NaN in the variables the formula uses (v2.6.12). linearmodels 6.1 with formulaic 1.2
    stops on any NaN ("contains null values after evaluation"), and 7.1 drops them only for the IV fit."""
    try:
        from formulaic import Formula
        lhs, rhs = formula.split("~", 1)
        plain = rhs.replace("[", " + ").replace("]", " ").replace("~", " + ")
        used = [v for v in Formula(f"{lhs} ~ {plain}").required_variables if v in data.columns]
    except Exception:  # noqa: BLE001  (an unusual formula: let linearmodels report it)
        return data
    return data.dropna(subset=used) if used else data


def iv_report(formula: str, data: pd.DataFrame, cov_type: str = "robust", alpha: float = ALPHA,
              ar_grid: int = 2001, show: bool = True) -> dict:
    """2SLS with the checks that decide whether to believe it (panel-iv.md sections 2-4).

    formula: linearmodels syntax, "lwage ~ 1 + exper + expersq + [educ ~ motheduc + fatheduc]".
    Reports OLS and 2SLS side by side, on the same rows (v2.6.12: rows with NaN in any variable of the formula are
    dropped first, and OLS is fitted on the rows 2SLS used); for each endogenous regressor the first-stage partial F
    and partial R² of the excluded instruments, classified as weak (F < 10), acceptable but needing the
    Anderson-Rubin interval (10 to 104.7, Lee et al., 2022) or strong; the endogeneity test (robust Wooldridge
    regression test, or Wu-Hausman with cov_type="unadjusted" or "homoskedastic"); the overidentification test only
    when there are more instruments than endogenous regressors (robust Wooldridge score test, or Sargan); and, with
    one endogenous regressor, the Anderson-Rubin confidence set, valid however weak the instruments are (found on a
    grid of ±20 SEs around 2SLS; it can be unbounded, or the complement of an interval; ±inf marks a piece that
    reaches the grid edge). With several endogenous regressors the set is k-dimensional and is not computed; the
    joint Anderson-Rubin test of "all their coefficients are zero" is reported instead (v2.6.12).
    The robust first-stage statistic is reported as an F (Wald χ²/q). With cov_type "unadjusted" or "homoskedastic"
    the first-stage F is the classical one, SSR-based with n − k degrees of freedom (v2.6.12: linearmodels divides
    by n, which made it n/(n − k) too large). The 104.7 threshold is for one instrument. Exogeneity of an instrument
    cannot be tested with just one: it is an argument, not a statistic. Instruments and endogenous regressors are
    counted on the model's columns (`[x ~ z1*z2]` has three instruments: z1, z2 and z1:z2).
    Keys of the result:
      table             OLS and 2SLS coefficients, SEs, 2SLS p and confidence interval;
      first_stage       partial F, its p, partial R² and strength label, one row per endogenous regressor;
      tests             endogeneity and (if overidentified) overidentification: statistic and p;
      anderson_rubin    one endogenous regressor: {"estimand", "ci_low", "ci_high", "pieces", "shape",
                        "p_at_zero", "level"}; several: {"estimand" (list), "joint_F_at_zero", "joint_p_at_zero",
                        "df", "pieces": None, "shape", "level"};
      n, n_dropped      rows used, and rows dropped for missing values;
      endogenous, instruments   the model's columns; overidentified_by; cov_type; warnings;
      iv, ols           the linearmodels results."""
    from linearmodels.iv import IV2SLS
    robust = str(cov_type).lower() not in ("unadjusted", "homoskedastic")
    used = _formula_rows(formula, data)
    n_dropped = len(data) - len(used)
    iv = IV2SLS.from_formula(formula, used).fit(cov_type=cov_type)
    rows = iv.model.dependent.pandas.index
    lhs, rhs = formula.split("~", 1)
    endog_part = rhs[rhs.index("[") + 1: rhs.index("]")]
    endog_names = [e.strip() for e in endog_part.split("~")[0].split("+")]
    exog_rhs = (rhs[:rhs.index("[")] + rhs[rhs.index("]") + 1:]).strip().strip("+").strip()
    ols_rhs = " + ".join([p for p in (exog_rhs, " + ".join(endog_names)) if p])
    warn = []
    if n_dropped or len(rows) < len(used):
        warn.append(f"{len(data) - len(rows)} of {len(data)} rows dropped for missing or undefined values; OLS and 2SLS "
                    "use the same rows")
    W0 = np.asarray(iv.model.exog.ndarray, dtype=float)
    # v2.6.12: "has a constant" means the constant is in the span of the exogenous columns (0 + C(g) has it)
    if W0.shape[1]:
        one = np.ones(W0.shape[0])
        res1 = one - W0 @ np.linalg.lstsq(W0, one, rcond=None)[0]
        has_const = float(np.linalg.norm(res1)) <= 1e-8 * np.sqrt(W0.shape[0])
    else:
        has_const = False
    if not has_const:
        warn.append("the IV model has no constant (write 'y ~ 1 + ...'); OLS is fitted without one too, to compare like "
                    "with like")
        ols_rhs = "0 + " + ols_rhs
    ols = IV2SLS.from_formula(f"{lhs.strip()} ~ {ols_rhs}", used.loc[rows]).fit(cov_type=cov_type)
    table = pd.DataFrame({"OLS": ols.params, "OLS se": ols.std_errors, "2SLS": iv.params, "2SLS se": iv.std_errors,
                          "2SLS p": iv.pvalues})
    ci = iv.conf_int(level=1 - alpha)
    table["2SLS ci_low"], table["2SLS ci_high"] = ci.iloc[:, 0], ci.iloc[:, 1]
    diag = iv.first_stage.diagnostics
    # v2.6.9: instruments and endogenous regressors are counted on the model's columns, not on the formula's
    # terms: [x ~ z1*z2] has 3 instruments (z1, z2, z1:z2), and counting 1 made the partial F three times too
    # large and skipped the overidentification test (v2.6.8 review)
    endog_cols = [str(c) for c in iv.model.endog.cols]
    instr_cols = [str(c) for c in iv.model.instruments.cols]
    n_endog, q_instr = len(endog_cols), len(instr_cols)
    n_used = int(iv.nobs)
    # linearmodels reports the robust first-stage statistic as a Wald chi2(q); F = chi2 / q. Its F (unadjusted,
    # homoskedastic) uses SSR/n; the classical F uses SSR/(n − k), k = first-stage columns, the df of its F(q, n − k)
    fstat, fpval = [], []
    for v, dist, pv in zip(diag["f.stat"], diag["f.dist"], diag["f.pval"]):
        dist = str(dist)
        if dist.lower().startswith("chi2"):
            fstat.append(float(v) / q_instr); fpval.append(float(pv))
        else:
            df2 = float(dist[dist.index(",") + 1: dist.index(")")])
            f_c = float(v) * df2 / n_used
            fstat.append(f_c); fpval.append(float(stats.f.sf(f_c, q_instr, df2)))
    fs = pd.DataFrame({"partial F": fstat, "partial F p": fpval, "partial R²": diag["partial.rsquared"]},
                      index=diag.index)
    fs["strength"] = ["WEAK (F < 10)" if f < 10 else ("acceptable: report Anderson-Rubin (10 ≤ F ≤ 104.7)" if f <= 104.7
                      else "strong (F > 104.7)") for f in fs["partial F"]]
    if q_instr > 1:
        fs["strength"] += " [104.7 is the one-instrument threshold of Lee et al.]"
    if (fs["partial F"] < 10).any():
        # v2.6.8: LIML only when overidentified (exactly identified, LIML equals 2SLS). Counted on the model's
        # columns, not the formula's terms: [x ~ z1*z2] has 3 instruments (review, round 1)
        exact_id = q_instr == n_endog
        ar_what = "Anderson-Rubin set" if n_endog == 1 else "joint Anderson-Rubin test"
        warn.append("weak instrument(s): 2SLS is biased toward OLS and its t-intervals undercover; use the "
                    + (f"{ar_what} and say so (LIML does not help here: exactly identified, it equals 2SLS)"
                       if exact_id else f"{ar_what} (or LIML) and say so"))
    tests = []
    endo = iv.wooldridge_regression if robust else iv.wu_hausman()
    tests.append({"test": "endogeneity (" + ("Wooldridge robust regression" if robust else "Wu-Hausman") + ")",
                  "H0": "regressors are exogenous: OLS is consistent", "statistic": float(endo.stat),
                  "p": float(endo.pval)})
    q = q_instr - n_endog
    if q > 0:
        ov = iv.wooldridge_overid if robust else iv.sargan
        tests.append({"test": "overidentification (" + ("Wooldridge robust score" if robust else "Sargan") + f", {q} df)",
                      "H0": "the extra instruments are valid given that the others are", "statistic": float(ov.stat),
                      "p": float(ov.pval)})
    tests = pd.DataFrame(tests).set_index("test")
    yv = np.asarray(iv.model.dependent.ndarray).ravel()
    W = np.asarray(iv.model.exog.ndarray)
    Z = np.asarray(iv.model.instruments.ndarray)
    if n_endog == 1:
        xv = np.asarray(iv.model.endog.ndarray).ravel()
        b, s = float(iv.params[endog_cols[0]]), float(iv.std_errors[endog_cols[0]])
        grid = np.linspace(b - 20 * s, b + 20 * s, ar_grid)
        pv = np.array([_ar_pvalue(yv, xv, W, Z, g, robust) for g in grid])
        acc = pv > alpha
        pieces = []
        if acc.any():
            idx = np.flatnonzero(acc)
            for seg in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
                pieces.append((float(grid[seg[0]]) if seg[0] > 0 else -np.inf,
                               float(grid[seg[-1]]) if seg[-1] < len(grid) - 1 else np.inf))
        if not pieces:
            shape = "empty on the grid (the model may be misspecified)"
        elif len(pieces) == 1 and np.isfinite(pieces[0][0]) and np.isfinite(pieces[0][1]):
            shape = "bounded interval"
        elif acc[0] and acc[-1] and len(pieces) == 2:
            shape = "complement of an interval (unbounded in both directions)"
        elif acc[0] and acc[-1]:
            shape = "unbounded in both directions"
        elif acc[0] or acc[-1]:
            shape = "unbounded in one direction (or wider than ±20 SEs)"
        else:
            shape = "disjoint (union of intervals)"
        lo = pieces[0][0] if pieces else np.nan
        hi = pieces[-1][1] if pieces else np.nan
        ar = {"estimand": endog_cols[0], "ci_low": lo, "ci_high": hi, "pieces": pieces, "shape": shape,
              "p_at_zero": _ar_pvalue(yv, xv, W, Z, 0.0, robust), "level": 1 - alpha}
        if shape != "bounded interval":
            warn.append(f"Anderson-Rubin set is {shape}: the data do not pin down the effect")
    else:
        # v2.6.12: with k endogenous regressors the AR set is k-dimensional; the joint test of β = 0 is one
        # regression (y on W and Z, F on Z), valid however weak the instruments are
        f0, p0 = _ar_test(yv, W, Z, robust)
        ar = {"estimand": endog_cols, "joint_F_at_zero": f0, "joint_p_at_zero": p0,
              "df": (q_instr, n_used - W.shape[1] - q_instr), "pieces": None,
              "shape": f"not computed (with {n_endog} endogenous regressors the set is {n_endog}-dimensional)",
              "level": 1 - alpha}
    out = {"table": table, "first_stage": fs, "tests": tests, "anderson_rubin": ar, "n": n_used,
           "n_dropped": len(data) - len(rows), "endogenous": endog_cols, "instruments": instr_cols,
           "overidentified_by": q, "cov_type": cov_type, "warnings": warn, "iv": iv, "ols": ols}
    if show:
        print(f"\nIV report (n = {out['n']}, cov_type = {cov_type}); endogenous: {', '.join(endog_cols)}; "
              f"instruments: {', '.join(instr_cols)}")
        print(table.to_string(float_format=_fmt))
        print("\nFirst stage (excluded instruments):")
        print(fs.to_string(float_format=_fmt))
        print("\nTests:")
        print(tests[["statistic", "p"]].to_string(float_format=_fmt))
        if q == 0:
            print("Exactly identified: no overidentification test is possible; instrument validity is an argument.")
        if n_endog == 1:
            txt = " ∪ ".join(f"[{a_:.4f}, {b_:.4f}]" for a_, b_ in ar["pieces"]) or "empty"
            print(f"\nAnderson-Rubin {ar['level']:.0%} set for {ar['estimand']}: {txt} ({ar['shape']}); "
                  f"AR p-value of β = 0: {ar['p_at_zero']:.4f}")
        else:
            print(f"\nAnderson-Rubin, joint test that the coefficients of {', '.join(endog_cols)} are all zero: "
                  f"F({ar['df'][0]}, {ar['df'][1]}) = {ar['joint_F_at_zero']:.3f}, p = {ar['joint_p_at_zero']:.4f} "
                  f"(valid with weak instruments); the confidence set is {ar['shape']}")
        for w in warn:
            print("Warning:", w)
    return out


def hausman(fe_res, re_res, show: bool = True) -> dict:
    """Classical Hausman test FE vs RE (linearmodels). Fit both with cov_type="unadjusted"."""
    common = [c for c in fe_res.params.index if c in re_res.params.index and c.lower() not in ("const", "intercept")]
    b = fe_res.params[common] - re_res.params[common]
    v = fe_res.cov.loc[common, common] - re_res.cov.loc[common, common]
    # v2.6.12: V_FE − V_RE is a difference, so its rounding is relative to V_FE. Eigenvalues within ±1e-8·max eig(V_FE)
    # are zero: the statistic is b'V⁻b with the generalized inverse over the other eigenvalues (negative ones
    # included, as pinv did), df = their number (rank of V, as Stata's hausman), and V is "not positive semidefinite"
    # only below −1e-8·max eig(V_FE). Before: df = number of coefficients, and an absolute −1e-10.
    lam, vec = np.linalg.eigh(v.values)
    tol = 1e-8 * max(float(np.max(np.abs(np.linalg.eigvalsh(fe_res.cov.loc[common, common].values)))), 1e-300)
    keep = np.abs(lam) > tol
    eig = lam
    stat = float(np.sum((vec[:, keep].T @ b.values) ** 2 / lam[keep])) if keep.any() else float("nan")
    df_ = int(keep.sum())
    p = float(stats.chi2.sf(stat, df_)) if df_ and stat >= 0 else (1.0 if df_ else float("nan"))
    valid = bool(np.all(eig > -tol))
    # v2.6.11: the classical statistic needs both fits with the unadjusted covariance. linearmodels keeps the
    # estimator only in the private _cov_type ("Unadjusted", "Robust", "Clustered", "Driscoll-Kraay"; same in
    # 6.1 and 7.1); when it is missing, nothing is claimed (None).
    kinds = [getattr(r, "_cov_type", None) for r in (fe_res, re_res)]
    classical_cov = None if None in kinds else all(str(c).lower() in ("unadjusted", "homoskedastic")
                                                     for c in kinds)
    concl = "Reject H0 → fixed effects" if p < ALPHA else "Do not reject H0 → random effects acceptable"
    if not valid:
        concl += " (WARNING: V_FE − V_RE is not positive semidefinite; prefer the Mundlak test)"
    if stat < 0:   # v2.6.11 review r1: a negative quadratic form is not a χ² draw; p = 1 says nothing
        concl = ("Not interpretable: the statistic is negative"
                 + ("" if valid else " (V_FE − V_RE is not positive semidefinite)") + "; use mundlak_test")
    if classical_cov is False:
        concl = (f"Not interpretable: fitted with {kinds[0]}/{kinds[1]} covariances; the classical Hausman "
                 "needs cov_type='unadjusted' in both (refit them, or use mundlak_test for a robust comparison)")
    out = {"statistic": stat, "df": df_, "p": p, "valid": valid, "classical_cov": classical_cov,
           "conclusion": concl}
    if show:
        print(f"\nHausman: χ²({df_}) = {stat:.3f}, p = {p:.4f} → {concl}")
    return out


def panel_bp_lm(resid_pooled: pd.Series, show: bool = True) -> dict:
    """Breusch-Pagan LM (Stata's xttest0): H0 = variance of the individual effect is zero (pooled OK).

    resid_pooled: pooled OLS residuals with a (entity, time) MultiIndex. Balanced or unbalanced: the statistic
    is Baltagi and Li's (1990) n²/(2(ΣT_i² − n))·[e'De/e'e − 1]², D block-diagonal with T_i×T_i blocks of ones,
    n = ΣT_i, which is NT/(2(T−1))·[…]² when every T_i = T (v2.6.11: the mean T was used before, which over-rejects
    on unbalanced panels).
    """
    e = resid_pooled.dropna()   # review r3: T_i counted the NaN rows that the sums skipped
    groups = e.groupby(level=0)
    Ti = groups.size().to_numpy(float)
    n = Ti.sum()
    num = (groups.sum() ** 2).sum()
    den = (e ** 2).sum()
    lm = (n ** 2 / (2 * ((Ti ** 2).sum() - n)) * (num / den - 1) ** 2
          if (Ti ** 2).sum() - n > 0 and den > 0 else np.nan)
    if not np.isfinite(lm):   # v2.6.11 reviews r2-r3: every T_i = 1, all-zero or empty residuals
        if show:
            print("\nBreusch-Pagan LM (panel): not defined — no entity with two or more observations, "
                  "or residuals all zero")
        return {"LM": np.nan, "p": np.nan}
    p = 1 - stats.chi2.cdf(lm, 1)
    if show:
        print(f"\nBreusch-Pagan LM (panel): {lm:.3f}, p = {p:.4f} → "
              + ("individual effects present: pooled OLS inadequate (use RE or FE)" if p < ALPHA
                 else "pooled OLS acceptable"))
    return {"LM": lm, "p": p}


# =============================================================================
# Tobit
# =============================================================================
def mundlak_test(df: pd.DataFrame, y: str, x_vary: list[str], entity: str, time: str,
                 x_const: list[str] | None = None, time_effects: bool = True, show: bool = True) -> dict:
    """Mundlak / correlated random effects (CRE) test: FE vs RE robust to heteroskedasticity and
    within-unit correlation (clustered by entity) — preferable to the classical Hausman test.

    Estimates RE of y on x_vary, x_const, unit means of x_vary (and time dummies), then a clustered
    Wald test that the means are jointly zero. Rejection ⇒ the unit effect is correlated with x
    ⇒ use FE (or keep the CRE model, whose coefficients on x_vary equal FE in a balanced panel
    while still estimating time-invariant x_const). In unbalanced panels the equality with FE also
    requires the unit means of the time dummies (Wooldridge, 2019); a warning is printed.
    """
    from linearmodels.panel import RandomEffects

    x_const = list(x_const or [])
    d = df[[y, entity, time] + list(x_vary) + x_const].dropna().copy()
    means = []
    for v in x_vary:
        d[f"{v}_mean"] = d.groupby(entity)[v].transform("mean")
        means.append(f"{v}_mean")
    balanced = d.groupby(entity)[time].nunique().nunique() == 1
    d["_period"] = d[time].values
    d = d.set_index([entity, time])
    rhs = ["1"] + list(x_vary) + x_const + means + (["C(_period)"] if time_effects else [])
    res = RandomEffects.from_formula(f"{y} ~ " + " + ".join(rhs), d) \
        .fit(cov_type="clustered", cluster_entity=True)
    w = res.wald_test(formula=", ".join(f"{m} = 0" for m in means))
    out = {"statistic": float(w.stat), "p": float(w.pval), "df": int(w.df), "results": res,
           "coefficients": res.params, "balanced": bool(balanced)}
    if show:
        print(f"\nMundlak / CRE test (RE with unit means, clustered by {entity}): "
              f"Wald χ²({out['df']}) = {out['statistic']:.3f}, p = {out['p']:.4f} → "
              + ("unit effects correlated with x: use FE (or report the CRE model)" if out["p"] < ALPHA
                 else "no evidence against RE: RE (or CRE) is acceptable and more efficient"))
        if not balanced and time_effects:
            print("WARNING: unbalanced panel — add unit means of the time dummies for exact equality with FE.")
    return out


def count_diagnostics(res, show: bool = True) -> dict:
    """Overdispersion checks after a Poisson fit (statsmodels Poisson or GLM Poisson).

    - Pearson dispersion σ̂² = Σ(y−μ̂)²/μ̂ / (n−k): ≈1 under Poisson; >1 overdispersion.
    - Cameron-Trivedi (1990) regression tests, one-sided (H1: α > 0):
        NB2 (Var = μ + αμ²): regress ((y−μ̂)² − y)/μ̂ on μ̂ without a constant;
        NB1 (Var = (1+α)μ):   regress the same on a constant.
    - Zeros: observed share vs the share implied by the Poisson fit (excess zeros → hurdle/ZIP).
    Poisson coefficients stay consistent under overdispersion (QMLE) if the mean is right: the
    usual fix is robust SEs (cov_type="HC0"/"HC1"), or quasi-Poisson SEs (× σ̂), or NB if the
    variance itself matters (predicted probabilities of counts).
    """
    yv = np.asarray(res.model.endog, dtype=float)
    mu = np.asarray(res.predict(), dtype=float)
    n, k = len(yv), len(res.params)
    disp = float(np.sum((yv - mu) ** 2 / mu) / (n - k))
    z = ((yv - mu) ** 2 - yv) / mu
    rows = {}
    for label, X in (("NB2 (Var = μ + αμ²)", mu.reshape(-1, 1)), ("NB1 (Var = (1+α)μ)", np.ones((n, 1)))):
        a = sm.OLS(z, X).fit()
        t = float(a.tvalues[0])
        rows[label] = {"alpha": float(a.params[0]), "t": t, "p": float(stats.norm.sf(t))}
    tests = pd.DataFrame(rows).T
    zeros_obs, zeros_pred = float(np.mean(yv == 0)), float(np.mean(np.exp(-mu)))
    out = {"dispersion": disp, "tests": tests, "zeros_observed": zeros_obs, "zeros_predicted": zeros_pred,
           "quasi_poisson_se": pd.Series(np.asarray(res.bse) * np.sqrt(disp), index=getattr(res.bse, "index", None))}
    if show:
        print(f"\nPoisson overdispersion: Pearson σ̂² = {disp:.3f} (σ̂ = {np.sqrt(disp):.3f}; 1 under Poisson)")
        print(tests.to_string(float_format=_fmt))
        over = (tests["p"] < ALPHA).any()
        print(("Overdispersion: keep Poisson coefficients (QMLE) with robust SEs, or use quasi-Poisson SEs "
               "(× σ̂), or a negative binomial if count probabilities matter.") if over
              else "No evidence of overdispersion: Poisson SEs are fine.")
        print(f"Zeros: observed {zeros_obs:.1%} vs Poisson-implied {zeros_pred:.1%}"
              + (" → excess zeros: consider hurdle or zero-inflated models." if zeros_obs > 1.2 * zeros_pred else ""))
    return out


def _tobit_parts(xb, s, a, b):
    """Pieces of a Tobit with lower limit a and upper limit b (either may be None) at index xb."""
    za = (a - xb) / s if a is not None else np.full_like(xb, -np.inf)
    zb = (b - xb) / s if b is not None else np.full_like(xb, np.inf)
    Pa, Pb = stats.norm.cdf(za), stats.norm.cdf(zb)
    pa, pb = stats.norm.pdf(za), stats.norm.pdf(zb)
    # z·φ(z) → 0 as |z| → ∞; avoid inf·0 = nan
    zpa = np.where(np.isfinite(za), za * pa, 0.0)
    zpb = np.where(np.isfinite(zb), zb * pb, 0.0)
    D = np.clip(Pb - Pa, 1e-300, None)
    lam = (pa - pb) / D
    return za, zb, Pa, Pb, pa, pb, zpa, zpb, D, lam


def _tobit_ames(theta, XX, a, b):
    """Average marginal effects (per unit of each regressor) of a Tobit at parameters theta = (β, log σ).

    Each expectation depends on x only through xβ, so ∂f/∂x_j = f'(xβ)·β_j:
      E(y|x):             f' = Φ(z_b) − Φ(z_a)                          (McDonald-Moffitt)
      E(y|a<y<b, x):      f' = 1 + (z_a φ_a − z_b φ_b)/D − λ²,  λ = (φ_a − φ_b)/D
      P(a<y<b | x):       f' = (φ_a − φ_b)/σ
    """
    beta, s = theta[:-1], np.exp(theta[-1])
    _, _, _, _, pa, pb, zpa, zpb, D, lam = _tobit_parts(XX @ beta, s, a, b)
    d_all = D
    d_in = 1 + (zpa - zpb) / D - lam ** 2
    d_p = (pa - pb) / s
    return np.concatenate([beta * d_all.mean(), beta * d_in.mean(), beta * d_p.mean()])


def tobit(y, X, lower_limit: float | None = 0.0, upper_limit: float | None = None,
          show: bool = True) -> dict:
    """Censored (Tobit) regression by maximum likelihood. X without constant (it is added).

    lower_limit / upper_limit: censoring points (None = no censoring on that side). The default
    (0, None) is the classic corner-solution Tobit; (None, c) handles top-coding; (a, b) two limits.

    Returns latent coefficients, σ, standard errors (inverse numerical Hessian) and the three
    average marginal effects with delta-method standard errors (`marginal_effects`):
      E[y|x]              effect on the observed outcome (participation + intensity);
      E[y|uncensored,x]   effect for units strictly inside the limits (intensive margin);
      P(uncensored|x)     effect on the probability of being uncensored (extensive margin;
                          with only a lower limit, the probability of y > lower_limit).
    The effects are derivatives; for a dummy regressor they approximate the discrete change.
    """
    if lower_limit is None and upper_limit is None:
        raise ValueError("Give at least one censoring limit (lower_limit and/or upper_limit).")
    if lower_limit is not None and upper_limit is not None and not lower_limit < upper_limit:
        raise ValueError("lower_limit must be smaller than upper_limit.")
    d = pd.concat([y, X], axis=1).dropna()
    yy = d.iloc[:, 0].values.astype(float)
    XX = sm.add_constant(d.iloc[:, 1:], has_constant="add").values
    names = ["const"] + list(d.columns[1:])
    a, b = lower_limit, upper_limit
    cens_lo = yy <= a if a is not None else np.zeros(len(yy), bool)
    cens_hi = yy >= b if b is not None else np.zeros(len(yy), bool)
    inside = ~(cens_lo | cens_hi)

    def nll(theta):
        beta, log_s = theta[:-1], theta[-1]
        s = np.exp(log_s)
        xb = XX @ beta
        ll = stats.norm.logpdf((yy[inside] - xb[inside]) / s).sum() - log_s * inside.sum()
        if cens_lo.any():
            ll += stats.norm.logcdf((a - xb[cens_lo]) / s).sum()
        if cens_hi.any():
            ll += stats.norm.logsf((b - xb[cens_hi]) / s).sum()
        return -ll

    b0 = np.linalg.lstsq(XX, yy, rcond=None)[0]
    theta0 = np.r_[b0, np.log(np.std(yy - XX @ b0))]
    opt = optimize.minimize(nll, theta0, method="BFGS")
    if not opt.success:  # BFGS precision loss near the optimum is common; refine with another method
        opt2 = optimize.minimize(nll, opt.x, method="Nelder-Mead",
                                 options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-10})
        if opt2.fun <= opt.fun:
            opt = opt2
    from statsmodels.tools.numdiff import approx_fprime, approx_hess
    grad_norm = float(np.max(np.abs(approx_fprime(opt.x, nll))))
    opt.success = bool(opt.success or grad_norm < 1e-3)
    H = approx_hess(opt.x, nll)
    cov = np.linalg.inv(H)
    se = np.sqrt(np.diag(cov))
    beta, s = opt.x[:-1], np.exp(opt.x[-1])

    k = len(beta)
    ames = _tobit_ames(opt.x, XX, a, b)
    J = approx_fprime(opt.x, lambda th: _tobit_ames(th, XX, a, b), centered=True)
    se_ames = np.sqrt(np.clip(np.diag(J @ cov @ J.T), 0, None))
    cols = ["E[y|x]", "E[y|uncensored,x]", "P(uncensored|x)"]
    me = pd.DataFrame({c: ames[i * k:(i + 1) * k] for i, c in enumerate(cols)}, index=names)
    for i, c in enumerate(cols):
        me[f"se {c}"] = se_ames[i * k:(i + 1) * k]
    me = me.drop(index="const")[[cols[0], f"se {cols[0]}", cols[1], f"se {cols[1]}", cols[2], f"se {cols[2]}"]]

    tab = pd.DataFrame({"latent_coef": beta, "se": se[:-1], "z": beta / se[:-1],
                        "p": 2 * (1 - stats.norm.cdf(np.abs(beta / se[:-1]))),
                        # kept from v1.3 for backward compatibility (= marginal_effects["E[y|x]"])
                        "marginal_effect_E[y|x]": ames[:k]}, index=names)
    out = {"table": tab, "marginal_effects": me, "sigma": s, "se_sigma": float(s * se[-1]),
           "llf": -opt.fun, "converged": opt.success, "n": len(yy),
           "n_censored": int(cens_lo.sum() + cens_hi.sum()),
           "n_censored_lower": int(cens_lo.sum()), "n_censored_upper": int(cens_hi.sum()),
           "lower_limit": a, "upper_limit": b}
    if show:
        lims = " and ".join(f"{side} at {v}" for side, v in (("below", a), ("above", b)) if v is not None)
        print(f"\nTobit (censored {lims}; {cens_lo.sum()} below, {cens_hi.sum()} above, of {len(yy)}), "
              f"σ = {s:.4f}, log-likelihood = {-opt.fun:.2f}, converged = {opt.success}")
        print(tab.drop(columns="marginal_effect_E[y|x]").to_string(float_format=_fmt))
        print("\nAverage marginal effects (delta-method SEs):")
        print(me.to_string(float_format=_fmt))
        print("E[y|x]: total effect on the observed outcome. E[y|uncensored,x]: intensive margin "
              "(units inside the limits). P(uncensored|x): extensive margin. Report the one that "
              "answers the question; latent coefficients are not effects on observed y.")
    return out


# =============================================================================
# Forecasting
# =============================================================================
def diebold_mariano(e1, e2, h: int = 1, power: int = 2, show: bool = True) -> dict:
    """Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample correction.

    e1, e2: forecast errors (actual − forecast) of models 1 and 2 over the same period;
    h: horizon; power 2 = squared loss, 1 = absolute loss.
    Negative statistic ⇒ model 1 more accurate.
    """
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    d = np.abs(e1) ** power - np.abs(e2) ** power
    T = len(d)
    dbar = d.mean()
    gamma = [np.sum((d[k:] - dbar) * (d[:T - k] - dbar)) / T for k in range(h)]
    var_d = (gamma[0] + 2 * sum(gamma[1:])) / T
    dm = dbar / np.sqrt(var_d)
    hln = dm * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    p = 2 * (1 - stats.t.cdf(abs(hln), T - 1))
    if show:
        better = "model 1" if dbar < 0 else "model 2"
        if not np.isfinite(p):
            if not np.isfinite(d).all():
                why = f"{int((~np.isfinite(d)).sum())} missing or infinite forecast error(s); align and drop them first"
            elif np.ptp(d) == 0:
                why = "the loss difference is constant (identical losses): there is nothing to test"
            else:
                why = (f"the long-run variance of the loss difference is not positive (h = {h}, {T} periods); "
                       "use a longer sample" + (" or h = 1" if h > 1 else ""))
            print(f"\nDiebold-Mariano (HLN): not computed: {why}.")
        else:
            print(f"\nDiebold-Mariano (HLN): {hln:.3f}, p = {p:.4f} → "
                  + (f"significant difference, {better} more accurate" if p < ALPHA else "no significant difference"))
    return {"DM_HLN": hln, "p": p, "mean_loss_diff": dbar}


# =============================================================================
# CLI
# =============================================================================
def _read(path, date_col):
    if path.lower().endswith((".xlsx", ".xlsm")):
        from setup_env import require_extra   # optional openpyxl (v2.3.6)
        require_extra("excel", f"Reading {path}")
    df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)
    if date_col and date_col in df.columns:
        raw = df[date_col].astype(str)
        iso = pd.to_datetime(raw, format="ISO8601", errors="coerce")      # 2020-03-01
        dmy = pd.to_datetime(raw, dayfirst=True, errors="coerce")         # 01/03/2020
        df[date_col] = iso if iso.notna().mean() >= dmy.notna().mean() else dmy
        df = df.set_index(date_col).sort_index()
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["ols", "influence", "chow", "stationarity", "cointegration"])
    ap.add_argument("file")
    ap.add_argument("--y")
    ap.add_argument("--x", help="comma-separated regressors")
    ap.add_argument("--cols", help="comma-separated columns")
    ap.add_argument("--date", default="date", help="date column (optional)")
    ap.add_argument("--hac", type=int, help="HAC lags (time series)")
    ap.add_argument("--ts", action="store_true", help="time series: add DW, Breusch-Godfrey, ARCH-LM")
    ap.add_argument("--lags", type=int, default=1, help="Johansen k_ar_diff")
    ap.add_argument("--break", dest="brk", help="date/label where the 2nd regime starts (Chow)")
    a = ap.parse_args()
    df = _read(a.file, a.date)

    if a.action in ("ols", "influence", "chow"):
        xs = a.x.split(",")
        data = df[[a.y] + xs].dropna()
        if a.action == "chow":
            point = pd.Timestamp(a.brk) if isinstance(data.index, pd.DatetimeIndex) else type(data.index[0])(a.brk)
            chow(data[a.y], data[xs], point)
            return
        cov = {"cov_type": "HAC", "cov_kwds": {"maxlags": a.hac}} if a.hac else {"cov_type": "HC3"}
        res = sm.OLS(data[a.y], sm.add_constant(data[xs])).fit(**cov)
        if a.action == "ols":
            print(res.summary())
            ols_diagnostics(res, time_series=bool(a.hac) or a.ts)
        else:
            influence(res)
    elif a.action == "stationarity":
        stationarity_many(df[a.cols.split(",")])
    else:
        cointegration(df[a.cols.split(",")], k_ar_diff=a.lags)


if __name__ == "__main__":
    main()
