"""
Forecasting toolkit: automatic ARIMA order selection, ARIMAX with properly lagged
regressors, VAR forecasts, rolling-origin (walk-forward) evaluation, naive
benchmarks and forecast-accuracy tests.

Conventions
-----------
* Every out-of-sample forecast is produced with data available at the forecast
  origin only (no full-sample refits, no contemporaneous regressors).
* Rolling-origin output is indexed by the TARGET date: row t holds the forecast of
  y_t made at t-h. To trade on it, shift(-h) to the origin date or use
  backtest.run_backtest(..., lag=0).

Functions
---------
select_d(y)                         number of differences from KPSS/ADF (like pmdarima's ndiffs)
auto_arima(y, ...)                  grid search over (p,d,q)[(P,D,Q,s)] by AICc/AIC/BIC + residual checks
lag_exog(X, lags)                   lag regressors so they are known at the forecast origin
rolling_origin(y, order, ...)       h-step walk-forward forecasts from SARIMAX (optionally with exog)
rolling_var(df, target, ...)        1-step walk-forward forecasts of one VAR equation
benchmarks(y, start, ...)           naive, seasonal naive, drift and historical-mean forecasts
forecast_metrics(actual, fc, ...)   RMSE, MAE, MAPE, bias, Theil U, MASE, directional accuracy
pesaran_timmermann(actual, fc)      test that directional accuracy beats chance
evaluate(actual, forecasts)         table for several models + DM test vs the first benchmark
conformal_intervals(actual, fc, ...) prediction intervals for any forecaster from its past errors (ACI, rolling, split)

Requires: statsmodels >= 0.13, scipy. pmdarima is NOT required.
"""
from __future__ import annotations

import bisect
import itertools
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.statespace.sarimax import SARIMAX
import _console  # noqa: F401  (Windows-safe console output, see _console.py)

ALPHA = 0.05


# ---------------------------------------------------------------------------
# Order selection
# ---------------------------------------------------------------------------
def select_d(y: pd.Series, max_d: int = 2, test: str = "kpss", alpha: float = ALPHA) -> int:
    """Smallest d such that the differenced series looks stationary.

    test='kpss' (H0 stationary; difference while rejected) is the pmdarima/forecast
    default and is less prone to over-differencing than ADF.
    """
    x = pd.Series(y).dropna().astype(float)
    for d in range(max_d + 1):
        if len(x) < 20:
            return d
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if test == "kpss":
                p = kpss(x, regression="c", nlags="auto")[1]
                stationary = p > alpha
            else:
                p = adfuller(x, autolag="AIC")[1]
                stationary = p < alpha
        if stationary:
            return d
        x = x.diff().dropna()
    return max_d


def _aicc(res):
    k = len(res.params)
    n = res.nobs
    return res.aic + (2 * k * (k + 1)) / max(n - k - 1, 1)


def auto_arima(y: pd.Series, exog: pd.DataFrame | None = None, max_p: int = 5, max_q: int = 5,
               d: int | None = None, seasonal_period: int | None = None, D: int | None = None,
               max_P: int = 1, max_Q: int = 1, ic: str = "aicc", trend: str | None = None,
               top: int = 5, show: bool = True):
    """Exhaustive (p,q)[(P,Q)] search with d chosen by select_d. Returns (best_result, ranking).

    trend: None = automatic ('c' if d == 0, 't' (drift) if d == 1 and the mean change
    is significant, otherwise 'n').
    """
    y = pd.Series(y).astype(float)
    d = select_d(y) if d is None else d
    if seasonal_period and D is None:
        # Seasonal unit root: crude rule, difference seasonally if the seasonal lag ACF is high
        ys = y.diff(d).dropna() if d else y
        D = int(ys.autocorr(seasonal_period) > 0.64) if len(ys) > 2 * seasonal_period else 0
    D = D or 0
    if trend is None:
        if d + D == 0:
            trend = "c"
        elif d + D == 1:
            dy = y.diff(seasonal_period if D else 1).dropna() if (D or d) else y
            t = dy.mean() / (dy.std(ddof=1) / np.sqrt(len(dy)))
            trend = "t" if abs(t) > 1.96 else "n"
        else:
            trend = "n"
    seas = [(P, Q) for P in range(max_P + 1) for Q in range(max_Q + 1)] if seasonal_period else [(0, 0)]
    rows, fits = [], {}
    for p, q in itertools.product(range(max_p + 1), range(max_q + 1)):
        for P, Q in seas:
            so = (P, D, Q, seasonal_period) if seasonal_period else (0, 0, 0, 0)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = SARIMAX(y, exog=exog, order=(p, d, q), seasonal_order=so, trend=trend,
                                  enforce_stationarity=True, enforce_invertibility=True).fit(disp=False)
            except Exception:
                continue
            if not np.isfinite(res.llf):
                continue
            key = (p, d, q, P, D, Q)
            fits[key] = res
            rows.append({"order": (p, d, q), "seasonal": so if seasonal_period else None,
                         "AICc": _aicc(res), "AIC": res.aic, "BIC": res.bic,
                         "converged": res.mle_retvals.get("converged", True) if res.mle_retvals else True})
    if not rows:
        raise RuntimeError("No ARIMA specification could be estimated.")
    crit = {"aicc": "AICc", "aic": "AIC", "bic": "BIC"}[ic.lower()]
    rank = pd.DataFrame(rows).sort_values(crit).reset_index(drop=True)
    best_row = rank.iloc[0]
    so = best_row["seasonal"] or (0, 0, 0, 0)
    best = fits[(*best_row["order"], so[0], so[1], so[2])]
    if show:
        lb_lag = min(10, max(1, len(y) // 5))
        lb = acorr_ljungbox(best.resid.iloc[d + D * (seasonal_period or 0):], lags=[lb_lag],
                            model_df=sum(best_row["order"][::2]) + so[0] + so[2])
        lbp = float(lb["lb_pvalue"].iloc[0])
        print(f"\nAuto-ARIMA: d = {d}" + (f", D = {D} (s = {seasonal_period})" if seasonal_period else "")
              + f", trend = '{trend}', criterion = {crit}")
        print(rank.head(top).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        print(f"Selected: ARIMA{best_row['order']}" + (f"x{so}" if seasonal_period else "")
              + f" | Ljung-Box({lb_lag}) on residuals p = {lbp:.3f} → "
              + ("residuals look like white noise" if lbp > ALPHA else
                 "residual autocorrelation remains: enlarge the search or add regressors"))
        if crit != "BIC" and rank.iloc[0]["order"] != rank.sort_values("BIC").iloc[0]["order"]:
            print("Note: BIC prefers a more parsimonious model; for forecasting, test both out of sample.")
    best.trend_used = trend
    return best, rank


# ---------------------------------------------------------------------------
# Regressors
# ---------------------------------------------------------------------------
def lag_exog(X: pd.DataFrame | pd.Series, lags: int | list[int] = 1) -> pd.DataFrame:
    """Lag regressors so that row t only contains information from t-1 (or earlier).

    Using contemporaneous regressors (VIX_t to explain r_t) is fine for explaining,
    but a 'forecast' that needs X_t is not a forecast: you do not know X_t at t-1.
    """
    X = X.to_frame() if isinstance(X, pd.Series) else X
    lags = [lags] if isinstance(lags, int) else lags
    out = pd.concat({f"L{l}": X.shift(l) for l in lags}, axis=1)
    out.columns = [f"{c}_L{l[1:]}" for l, c in out.columns]
    return out


# ---------------------------------------------------------------------------
# Walk-forward forecasts
# ---------------------------------------------------------------------------
def rolling_origin(y: pd.Series, order=(1, 0, 0), start: int | float = 0.8, h: int = 1,
                   exog: pd.DataFrame | None = None, seasonal_order=(0, 0, 0, 0),
                   trend: str = "c", refit_every: int = 1, window: int | None = None) -> pd.DataFrame:
    """h-step-ahead walk-forward forecasts.

    start       : first forecast origin, as an index position or a share of the sample.
    refit_every : re-estimate parameters every k origins; in between, the state is
                  updated with the new observation (res.append, refit=False). 1 = refit always.
    window      : rolling estimation window length; None = expanding.
    exog        : must already be lagged (see lag_exog) so it is known at the origin;
                  for h > 1 the regressor values over the horizon are taken from exog,
                  which is only legitimate if they are lagged by >= h.
    Returns DataFrame indexed by target date: actual, forecast, origin.
    """
    y = pd.Series(y).astype(float)
    if exog is not None:
        exog = exog.reindex(y.index)
        ok = exog.notna().all(axis=1)
        y, exog = y[ok], exog[ok]
    n = len(y)
    s = int(start * n) if isinstance(start, float) else int(start)
    rows, res = [], None
    for i, origin in enumerate(range(s, n - h + 1)):
        lo = 0 if window is None else max(0, origin - window)
        y_tr = y.iloc[lo:origin]
        x_tr = None if exog is None else exog.iloc[lo:origin]
        if res is None or i % refit_every == 0 or window is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = SARIMAX(y_tr, exog=x_tr, order=order, seasonal_order=seasonal_order,
                              trend=trend).fit(disp=False)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = res.append(y.iloc[origin - 1:origin],
                                 exog=None if exog is None else exog.iloc[origin - 1:origin])
        x_f = None if exog is None else exog.iloc[origin:origin + h]
        fc = res.forecast(steps=h, exog=x_f)
        rows.append({"target": y.index[origin + h - 1], "origin": y.index[origin - 1],
                     "actual": y.iloc[origin + h - 1], "forecast": float(np.asarray(fc)[-1])})
    return pd.DataFrame(rows).set_index("target")


def rolling_var(df: pd.DataFrame, target: str, start: int | float = 0.8, maxlags: int = 10,
                ic: str = "aic", refit_every: int = 1, window: int | None = None,
                fixed_lag: int | None = None) -> pd.DataFrame:
    """1-step walk-forward forecasts of `target` from a VAR on the columns of df.

    All variables should be stationary (returns, changes, spreads); check with
    diagnostics.stationarity_many first. Lag order is re-selected at each refit.
    """
    from statsmodels.tsa.api import VAR
    df = df.dropna().astype(float)
    n = len(df)
    s = int(start * n) if isinstance(start, float) else int(start)
    j = list(df.columns).index(target)
    rows, res, p = [], None, None
    for i, origin in enumerate(range(s, n)):
        lo = 0 if window is None else max(0, origin - window)
        tr = df.iloc[lo:origin]
        if res is None or i % refit_every == 0:
            mod = VAR(tr)
            if fixed_lag is None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sel = getattr(mod.select_order(maxlags=maxlags), ic)
                p = max(1, int(sel or 1))
            else:
                p = fixed_lag
            res = mod.fit(p)
        fc = res.forecast(tr.values[-p:], steps=1)[0, j]
        rows.append({"target": df.index[origin], "origin": df.index[origin - 1],
                     "actual": df[target].iloc[origin], "forecast": fc, "lag_order": p})
    return pd.DataFrame(rows).set_index("target")


# ---------------------------------------------------------------------------
# Benchmarks and accuracy
# ---------------------------------------------------------------------------
def benchmarks(y: pd.Series, start: int | float = 0.8, h: int = 1,
               season: int | None = None) -> pd.DataFrame:
    """Benchmark forecasts indexed by target date (same convention as rolling_origin)."""
    y = pd.Series(y).astype(float)
    n = len(y)
    s = int(start * n) if isinstance(start, float) else int(start)
    idx = y.index[s + h - 1:]
    out = pd.DataFrame(index=idx)
    out["actual"] = y.loc[idx]
    out["naive"] = y.shift(h).loc[idx]                                     # random walk
    mean_exp = y.expanding().mean().shift(h)
    out["hist_mean"] = mean_exp.loc[idx]                                   # right benchmark for returns
    drift = (y - y.iloc[0]) / np.arange(n).clip(min=1)
    out["drift"] = (y.shift(h) + h * drift.shift(h)).loc[idx]
    if season:
        k = int(np.ceil(h / season)) * season
        out["seasonal_naive"] = y.shift(k).loc[idx]
    return out


def pesaran_timmermann(actual, forecast) -> dict:
    """Pesaran-Timmermann (1992) test. H0: forecast signs are independent of actual signs.

    Returns a dict (v2.7.3: keys exact):
      "hit_rate"       float, share of signs predicted right (rows where actual or forecast is 0 or NaN are left out)
      "hit_rate_under_H0"  float, the share expected by chance
      "PT"             float, the statistic
      "p_one_sided"    float, H1: better than chance (it treats the rows as independent: anticonservative with
                       overlapping horizons)
      "n"              int, rows used
    With n < 10 every value but n is NaN (e.g. a forecast that never predicts a change); PT and p_one_sided are
    also NaN when the forecast (or actual) always has the same sign ("always up"): there is nothing to test.
    Example (r = the result): r["p_one_sided"]; r["hit_rate"]
    """
    a = np.sign(np.asarray(actual, float))
    f = np.sign(np.asarray(forecast, float))
    m = np.isfinite(a) & np.isfinite(f) & (a != 0) & (f != 0)   # v2.7.3 (review r1-B B7): a NaN counted as a sign
    a, f = a[m] > 0, f[m] > 0
    n = len(a)
    if n < 10:                       # e.g. a naive forecast never predicts a change
        return {"hit_rate": np.nan, "hit_rate_under_H0": np.nan, "PT": np.nan, "p_one_sided": np.nan, "n": n}
    p_hat = np.mean(a == f)
    py, px = a.mean(), f.mean()
    p_star = py * px + (1 - py) * (1 - px)
    v_p = p_star * (1 - p_star) / n
    v_ps = ((2 * py - 1) ** 2 * px * (1 - px) / n + (2 * px - 1) ** 2 * py * (1 - py) / n
            + 4 * py * px * (1 - py) * (1 - px) / n ** 2)
    denom = v_p - v_ps
    stat = (p_hat - p_star) / np.sqrt(denom) if denom > 0 else np.nan
    return {"hit_rate": p_hat, "hit_rate_under_H0": p_star, "PT": stat,
            "p_one_sided": 1 - stats.norm.cdf(stat) if np.isfinite(stat) else np.nan, "n": n}


def forecast_metrics(actual, forecast, naive=None, insample_scale: float | None = None,
                     direction: str = "level", last_actual=None) -> dict:
    """Accuracy metrics.

    Returns a dict (v2.7.3: keys exact):
      "RMSE", "MAE", "bias(mean error)"  float (bias = mean of actual − forecast)
      "MAPE_%"         float, only with direction="level" and no actual equal to 0
      "Theil_U(RMSE/RMSE_naive)"  float, only with naive=
      "MASE"           float, only with insample_scale= (non-zero)
      "directional_accuracy", "PT_p_value"  float, only with direction="return" or last_actual=
                       (pesaran_timmermann's hit_rate and p_one_sided)
    An inf or NaN (an infinite input, a squared error that overflows) comes back without a word; evaluate() says
    which (v2.7.3: in tab.attrs["warnings"]).
    Example (r = the result): r["RMSE"]; r["MAE"]

    direction='level'  : directional accuracy of the predicted CHANGE vs the actual change;
                         needs last_actual (value at the origin) — e.g. naive forecast for h=1.
    direction='return' : directional accuracy of the sign itself (series already a return).
    insample_scale     : MAE of the in-sample naive forecast, for MASE.
    """
    if direction not in ("level", "return"):
        # v2.7.2 (review r1-A A1): any other text was accepted and only dropped MAPE_%
        raise ValueError("direction must be 'level' or 'return'")
    a = np.asarray(actual, float)
    f = np.asarray(forecast, float)
    e = a - f
    out = {"RMSE": np.sqrt(np.mean(e ** 2)), "MAE": np.mean(np.abs(e)), "bias(mean error)": e.mean()}
    if np.all(a != 0) and direction == "level":
        out["MAPE_%"] = 100 * np.mean(np.abs(e / a))
    if naive is not None:
        en = a - np.asarray(naive, float)
        out["Theil_U(RMSE/RMSE_naive)"] = out["RMSE"] / np.sqrt(np.mean(en ** 2))
    if insample_scale:
        out["MASE"] = out["MAE"] / insample_scale
    if direction == "return":
        pt = pesaran_timmermann(a, f)
    elif last_actual is not None:
        la = np.asarray(last_actual, float)
        pt = pesaran_timmermann(a - la, f - la)
    else:
        pt = None
    if pt:
        out["directional_accuracy"] = pt["hit_rate"]
        out["PT_p_value"] = pt["p_one_sided"]
    return out


def evaluate(actual: pd.Series, forecasts: dict[str, pd.Series], benchmark: str | None = None,
             direction: str = "level", last_actual: pd.Series | None = None, h: int = 1,
             show: bool = True) -> pd.DataFrame:
    """Compare several forecasts over the common sample; DM test of each vs `benchmark`.

    Returns a DataFrame (v2.7.2), one row per forecast (the dict's names); of the columns
    ["RMSE", "MAE", "bias(mean error)", "MAPE_%", "Theil_U(RMSE/RMSE_naive)", "directional_accuracy", "PT_p_value",
    "DM_vs_<benchmark>", "DM_p"], in this order, "MAPE_%" only with direction="level" and no actual equal to 0;
    "directional_accuracy" and "PT_p_value" only with direction="return" or with last_actual=; the two DM columns
    only with two or more forecasts (NaN in the benchmark's row). Theil's U is relative to the benchmark whatever
    its name. direction is "level" (default) or "return". Example: tab.loc["model", "DM_p"].
      attrs["warnings"]  list of str (v2.7.3), "<model>: <cause>" for each inf or NaN of a row and each DM test not
                         computed (infinite input, squared errors that overflow, a constant loss difference, too few
                         rows, a perfect benchmark for Theil's U, a direction test without enough signs); empty when
                         there is none; filled with show=False and with one forecast too; show prints them
    Example (r = the result): r.loc[r.index[0], "RMSE"]; r.attrs["warnings"]
    """
    from diagnostics import diebold_mariano
    df = pd.concat({"actual": actual, **forecasts}, axis=1).dropna()
    if df.empty:   # v2.7.3 (review r1-B B2): an "overflow" warning (one forecast) or ZeroDivisionError (two) before
        raise ValueError("no row where actual and every forecast are all present: align their indexes (dates) first")
    if last_actual is not None:
        last_actual = last_actual.reindex(df.index)
    benchmark = benchmark or list(forecasts)[0]
    rows = {}
    for k in forecasts:
        m = forecast_metrics(df["actual"], df[k], naive=df[benchmark], direction=direction,
                             last_actual=last_actual)
        if k != benchmark:
            dm = diebold_mariano(df["actual"] - df[k], df["actual"] - df[benchmark], h=h, show=False)
            m["DM_vs_" + benchmark] = dm["DM_HLN"]
            m["DM_p"] = dm["p"]
        rows[k] = m
    tab = pd.DataFrame(rows).T
    # v2.7.3 (review r2 N6 of v2.7.2): the cause of every inf or NaN in the table and of a DM test not computed, per
    # model, also with show=False and with one forecast (before, it was printed only in the DM lines of show=True)
    from diagnostics import _dm_why
    inf_in = {c: bool(np.isinf(df[c].to_numpy(dtype=float)).any()) for c in df.columns}
    a_ = df["actual"].to_numpy(dtype=float)
    warns = []
    for k in forecasts:
        mine = []
        if inf_in["actual"] or inf_in[k]:
            src = "actual" if inf_in["actual"] else "the forecast"
            mine.append(f"{k}: {src} has infinite values (e.g. a percentage change from zero), so its RMSE, MAE and "
                        "bias are not usable; drop or fix those rows first")
        elif not np.isfinite(float(tab.loc[k, "RMSE"])):
            mine.append(f"{k}: the squared errors overflow (errors above ~1e154 in absolute value), so its RMSE is "
                        "inf; rescale the series (e.g. to thousands) first")
        elif k != benchmark and not np.isfinite(float(tab.loc[k, "DM_p"])):
            if inf_in[benchmark] or not np.isfinite(float(tab.loc[benchmark, "RMSE"])):
                mine.append(f"{k}: DM test against {benchmark} not computed: see the warning on {benchmark}"
                            + (" (Theil's U, relative to it, is not usable either)"
                               if "Theil_U(RMSE/RMSE_naive)" in tab.columns else ""))
            else:
                # review r1-B B1: the reason of diebold_mariano itself (constant difference, too few rows, h > 1)
                why = _dm_why((a_ - df[k].to_numpy(dtype=float)) ** 2,
                              (a_ - df[benchmark].to_numpy(dtype=float)) ** 2, h)
                mine.append(f"{k}: DM test not computed (no positive variance of the loss difference with "
                            f"{benchmark}): {why}; compare the RMSE column")
        if not mine or mine[0].startswith(f"{k}: DM test"):
            # review r1-B B3: every other inf or NaN of the row is named too
            other = [c for c in tab.columns if c not in ("DM_vs_" + benchmark, "DM_p")
                     and not np.isfinite(float(tab.loc[k, c]))]
            if "Theil_U(RMSE/RMSE_naive)" in other and float(tab.loc[benchmark, "RMSE"]) == 0:
                mine.append(f"{k}: Theil's U is inf (NaN for a model that is also perfect) because the benchmark "
                            f"{benchmark} has no error in any row (its RMSE is 0)")
                other.remove("Theil_U(RMSE/RMSE_naive)")
            pt = [c for c in other if c in ("directional_accuracy", "PT_p_value")]
            if pt:
                miss = int(last_actual.isna().sum()) if last_actual is not None else 0   # r2 R9b
                mine.append(f"{k}: {' and '.join(pt)} NaN: the direction test needs at least 10 rows where actual and "
                            "the forecast are not zero, and some variation in the forecast's sign"
                            + (f"; last_actual is missing on {miss} of the {len(df)} rows" if miss else ""))
            rest = [c for c in other if c not in pt]
            if rest:
                mine.append(f"{k}: {', '.join(rest)} not finite"
                            + (" (MAPE divides by actual: an actual value very close to 0)" if "MAPE_%" in rest else ""))
        # r2 R9a: with an inf benchmark, the other models' Theil's U is 0 (finite and misleading); said once per model
        if (not mine and k != benchmark and "Theil_U(RMSE/RMSE_naive)" in tab.columns
                and not np.isfinite(float(tab.loc[benchmark, "RMSE"]))):
            mine.append(f"{k}: Theil's U is not usable (the benchmark {benchmark}'s RMSE is not finite)")
        warns += mine
    tab.attrs["warnings"] = warns
    if show:
        print(f"\nOut-of-sample evaluation ({len(df)} forecasts, benchmark = {benchmark})")
        print(tab.to_string(float_format=lambda v: f"{v:.4f}"))
        for k in forecasts:
            if k == benchmark:
                continue
            p = tab.loc[k, "DM_p"]
            if np.isfinite(p):
                better = tab.loc[k, "RMSE"] < tab.loc[benchmark, "RMSE"]
                print(f"  {k}: " + ("significantly " if p < ALPHA else "not significantly ")
                      + ("more" if better else "less") + f" accurate than {benchmark} (DM p = {p:.3f})")
        # v2.7.3: every warning, the DM ones included (one text for the table and the printout)
        for w in warns:
            print("  Warning: " + w)
    return tab


# ---------------------------------------------------------------------------
# Prediction intervals for any forecaster (conformal)
# ---------------------------------------------------------------------------
def _wilson(k: int, n: int, level: float = 0.95):
    z = stats.norm.ppf(1 - (1 - level) / 2)
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    hw = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return c - hw, c + hw


def _pct(x: float) -> str:
    return f"{100 * x:.4g}%"


_MAX_ABS = 1e150          # v2.7.2: largest |actual|, |forecast| or error conformal_intervals accepts (squares stay finite)
_LB_LEVEL = 0.05          # level of the Ljung-Box screen for dependent errors, whatever the interval's alpha


def conformal_intervals(actual, forecast, alpha: float = ALPHA, method: str = "aci", h: int = 1,
                        calib_size: int | None = None, gamma: float = 0.005, show: bool = True) -> dict:
    """Prediction intervals for out-of-sample point forecasts of any model (ARIMA, machine learning, a
    foundation model, a judgement call), from the model's own past errors (conformal inference).

    Returns a dict (v2.7.2: keys exact):
      "table"          DataFrame indexed like actual, columns ["actual", "forecast", "lower", "upper", "covered",
                       "level", "n_calib", "empty"]
      "coverage"       float, share of targets with an actual inside the interval (infinite ones count as covered)
      "coverage_ci"    tuple (low, high), 95% Wilson interval
      "n_evaluated"    int
      "guarantee"      dict, keys ["kind", "slack", "T"]
      "n_infinite"     int
      "coverage_finite"  float, coverage of the finite intervals
      "used"           dict, keys ["method", "alpha", "h", "calib_size", "gamma", "n_rows", "n_future", "warm_up_rows",
                       "min_calib_errors", "coverage_ci_level", "ljung_box_level", "skill_version"]
      "warnings"       list of dicts with keys ["level", "message", "action"]

    actual, forecast : series indexed by TARGET date in time order, one row per consecutive target, as
                       rolling_origin returns them; the forecast of row i was made at origin i - h. To get
                       the interval of a forecast not yet realised, append its row with actual = NaN: rows
                       at the end with no actual get an interval and are left out of the coverage.
                       Values beyond 1e150 in absolute value (actual, forecast or their difference) are refused:
                       their squares can overflow (they do from ~1.3e154) (v2.7.2).
    method   : "aci" (default) — adaptive conformal inference (Gibbs and Candès 2021, eq. 2): the level is
               moved after every observed error, alpha_t = alpha + gamma * sum(alpha - err), so the share
               of misses tracks alpha over time even when errors are dependent or their size drifts;
               "rolling" — split conformal on the most recent known errors at each origin;
               "split" — one calibration block (the first calib_size rows) and a fixed width afterwards
               (Lei et al. 2018, Algorithm 2 and Theorem 2.2).
    h        : forecast horizon. The error of target j is known only once j is observed, so the interval
               for row i uses only errors of rows <= i - h (no look-ahead), and in "aci" the level used
               for row i is the one in force at its origin.
    calib_size : number of recent known errors used at each origin ("rolling", "aci"; None = all known
               so far) or the size of the calibration block ("split"; None = the first half of the rows
               with an actual). The interval half-width is the ceil((m + 1)(1 - level))-th smallest of
               the m absolute errors (at least the smallest). Rows whose origin has fewer than calib_size
               known errors (with None, ceil((1 - alpha) / alpha), 19 for alpha = 0.05) get no interval
               (warm-up).
    gamma    : ACI step (0.005 in the paper).

    Guarantees, and what was measured instead (v2.7.0, development notes):
    - split/rolling: coverage >= 1 - alpha only if the errors are exchangeable (independent, same
      distribution). With overlapping errors (h = 4) or persistent ones (AR coefficient 0.9), 95% intervals
      covered 84-95% in simulations, worst with few calibration errors (19).
    - aci with h = 1: |share of misses - alpha| <= (max(alpha, 1 - alpha) + gamma) / (T gamma) over any T
      forecasts, whatever the data (Proposition 4.1). With gamma = 0.005 that slack is 0.38 for T = 500:
      a long-run promise, uninformative in short samples, so the result reports it and the measured
      coverage. With h > 1 the paper's proposition does not apply as stated; the same argument with the
      update delayed until each error is observed gives (max(alpha, 1 - alpha) + h gamma) / (T gamma)
      (proof in the v2.7.0 review notes, not published). Measured over 500 targets, 93-95% for h from 2
      to 12; over 60 targets with h = 12 and AR(0.9) errors, about 90%, no better than "rolling": with a
      short record ACI behaves like split conformal on all errors so far.
    The dependence screen of "split" and "rolling" runs Ljung-Box on the errors and on their size, at 2.5% each
    (5% in all; it fired on 6-7% of i.i.d. normal series with 40 to 100 errors, about 5% with 200).
    Long horizons (v2.7.1, claude/medicoes-v2.7.1.md, M1; 1000 replications per cell, i.i.d. and GARCH returns,
    the forecast an expanding historical mean): with h = 21 and 120 to 1000 targets, the default gamma left 18-28%
    of the intervals infinite, with a longest run of up to ~300 rows (90th percentile), because ACI raised the level
    to 1 or close to it (the misses of overlapping errors come in runs and each is seen h steps late); total
    coverage 91-95% (the infinite ones count as covered), 84-94% among the finite ones. gamma = 0.005/h or 0.001
    removed them, but the coverage fell to that of "rolling" (90-94%), and calib_size of 100 or 250 did not help.
    With h = 1 or 5 the default left at most 1.6% infinite (coverage 93.7-95.2%). The finding counts the infinite
    rows (all from a level raised above 1 - alpha) and their longest run; `coverage_finite` gives the coverage of
    the finite ones (empty intervals, a sure miss, count among them). The long-horizon advice is printed with h > 1
    and at least 5% of infinite intervals.
    Intervals are symmetric around the forecast (absolute errors). A biased forecaster gets wide intervals:
    check the bias (forecast_metrics) first. With a level that rounds to 1 (ACI after many misses) the
    interval is infinite; with a level at or below 0 it is empty (a sure miss): both are counted in a finding.

    Returns a dict: `table` (actual, forecast, lower, upper, covered, level used, number of calibration
    errors, empty), `coverage` with its 95% Wilson interval and `n_evaluated`, `n_infinite` and `coverage_finite` (the
    coverage of the finite intervals with an actual; v2.7.1), `guarantee`, `used`, `warnings`."""
    from reporting import SKILL_VERSION, _finding, _print_findings
    if method not in ("aci", "rolling", "split"):
        raise ValueError("method must be 'aci', 'rolling' or 'split'")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    def _whole(x, name):
        if isinstance(x, bool) or not isinstance(x, (int, float, np.integer, np.floating)) or not np.isfinite(x) \
                or x != int(x):
            raise ValueError(f"{name} must be a whole number, got {x!r}")
        return int(x)
    h = _whole(h, "h")
    if h < 1:
        raise ValueError("h must be at least 1")
    if calib_size is not None:
        calib_size = _whole(calib_size, "calib_size")
    if not gamma > 0:
        raise ValueError("gamma must be positive")
    a_s = pd.Series(actual, dtype=float) if not isinstance(actual, pd.Series) else actual.astype(float)
    f_s = pd.Series(forecast, dtype=float) if not isinstance(forecast, pd.Series) else forecast.astype(float)
    if len(a_s) != len(f_s) or not a_s.index.equals(f_s.index):
        raise ValueError("actual and forecast must have the same index (target dates, one row per target)")
    if a_s.index.hasnans:
        # v2.7.1 (review NIT of v2.7.0): before, a missing date fell into the "time order" message below
        raise ValueError(f"the index has {int(a_s.index.isna().sum())} missing value(s) (NaN/NaT): every row needs its "
                         "target date; fix or drop those rows first")
    if not (a_s.index.is_monotonic_increasing and a_s.index.is_unique):
        raise ValueError("the index must be in time order with one row per target (oldest first, no repeats): "
                         "otherwise later errors would calibrate earlier forecasts")
    if f_s.isna().any():
        raise ValueError("forecast has missing values; drop or fill those targets first")
    if np.isinf(f_s.to_numpy()).any() or np.isinf(a_s.to_numpy()).any():
        raise ValueError("actual or forecast has infinite values; fix those targets first")
    obs = a_s.notna().to_numpy()
    if obs.any() and not obs[:np.flatnonzero(obs)[-1] + 1].all():
        raise ValueError("actual has missing values before its last observed row; only the trailing rows (the "
                         "future) may be missing")
    y, f = a_s.to_numpy(), f_s.to_numpy()
    n = len(y)
    with np.errstate(over="ignore"):
        r = np.abs(y - f)                                   # NaN for future rows
    # v2.7.2 (review r1-A A8 of v2.7.1; r1-B B1 and N4 of v2.7.2): finite actual and forecast whose difference
    # overflows (1.5e308 and −1.5e308) gave infinite intervals in every method, with a finding that blamed ACI even
    # for "rolling" and "split"; with errors above ~1.3e154 the squared errors of the Ljung-Box screen overflow
    # (the screen went silent), and forecast ± half can overflow. One bound, 1e150, keeps all of them finite.
    with np.errstate(invalid="ignore"):
        big = (obs & ((np.abs(y) > _MAX_ABS) | ~(r <= _MAX_ABS))) | (np.abs(f) > _MAX_ABS)
    if big.any():
        raise ValueError(f"values beyond {_MAX_ABS:g} in absolute value (actual, forecast or their difference) at "
                         f"{int(big.sum())} row(s) (first: {a_s.index[np.flatnonzero(big)[0]]!r}): squared errors "
                         "can overflow there; rescale the series (e.g. to thousands) first")
    m_min = int(np.ceil((1 - alpha) / alpha - 1e-12))       # smallest m with ceil((m+1)(1-alpha)) <= m
    if calib_size is not None and int(calib_size) < m_min:
        raise ValueError(f"calib_size = {calib_size} is below {m_min}, the fewest errors that give a finite "
                         f"{_pct(1 - alpha)} interval")

    def q_of(err_sorted, level):
        """ceil((m + 1) level)-th smallest of the m sorted errors (at least the 1st); inf when the rank exceeds m."""
        m = len(err_sorted)
        if abs(level) < 1e-12:                             # ACI: alpha_t = 1 up to rounding of the running sum
            level = 0.0
        elif abs(1 - level) < 1e-12:
            level = 1.0
        if level <= 0:
            return np.nan                                  # empty set (alpha_t >= 1): every target is a miss
        if level >= 1:
            return np.inf
        k = max(1, int(np.ceil((m + 1) * level - 1e-12)))  # a level a hair above 0 is the smallest error, never rank 0
        return np.inf if k > m else float(err_sorted[k - 1])

    half = np.full(n, np.nan)
    level_used = np.full(n, np.nan)
    m_used = np.zeros(n, dtype=int)
    evaluated = np.zeros(n, dtype=bool)
    n_obs = int(obs.sum())
    short_split = False
    if method == "split":
        cs = int(calib_size) if calib_size is not None else n_obs // 2
        if cs > n_obs:
            raise ValueError(f"calib_size = {cs} exceeds the {n_obs} rows with an actual")
        m_need = cs
        # review r2 R4: the record that gives one evaluated target: the block, then h rows; with the default block
        # (half of the rows with an actual) that is at least 2·m_min rows and 2h − 1 rows
        need_split = max(2 * m_min, 2 * h - 1) if calib_size is None else cs + h
        if cs < m_min:
            # v2.7.1 (review NIT of v2.7.0): a record too short for the default block (half of the rows with an actual)
            # gets the same finding as "aci" and "rolling" with too few errors, not an exception; an explicit calib_size
            # below the minimum was refused above
            # the smallest record that works: half of it at least m_min, and a target observed h rows after the block
            short_split = True
        else:
            cal = np.sort(r[:cs])
            q = q_of(cal, 1 - alpha)
            for i in range(cs - 1 + h, n):
                half[i], level_used[i], m_used[i], evaluated[i] = q, 1 - alpha, cs, True
    else:
        m_need = int(calib_size) if calib_size is not None else m_min   # full window before the first interval
        known: list = []                                    # sorted absolute errors known at the current origin
        window: list = []                                   # the same errors in time order (for calib_size)
        csum = np.zeros(n)                                  # running sum of (alpha - err_j) over evaluated, observed rows
        for i in range(n):
            o = i - h                                       # origin of row i: errors of rows <= o are known
            if o >= 0 and obs[o]:
                bisect.insort(known, float(r[o]))
                window.append(float(r[o]))
                if calib_size is not None and len(window) > int(calib_size):
                    known.pop(bisect.bisect_left(known, window.pop(0)))
            prev = csum[i - 1] if i > 0 else 0.0
            csum[i] = prev
            if len(known) < m_need:
                continue
            lev = 1 - alpha if method == "rolling" else 1 - (alpha + gamma * (csum[o] if o >= 0 else 0.0))
            half[i] = q_of(known, lev)
            level_used[i], m_used[i], evaluated[i] = lev, len(known), True
            if obs[i]:
                miss = 1.0 if np.isnan(half[i]) else float(r[i] > half[i])
                csum[i] = prev + (alpha - miss)
    lower = np.where(evaluated, f - half, np.nan)
    upper = np.where(evaluated, f + half, np.nan)
    empty = evaluated & np.isnan(half)                      # ACI level <= 0: the empty set, a sure miss
    with np.errstate(invalid="ignore"):
        hit = np.where(empty, 0.0, (r <= np.where(empty, 0.0, half)).astype(float))
    covered = np.where(evaluated & obs, hit, np.nan)
    table = pd.DataFrame({"actual": y, "forecast": f, "lower": lower, "upper": upper,
                          "covered": covered, "level": level_used, "n_calib": m_used,
                          "empty": empty}, index=a_s.index)
    ev = evaluated & obs
    T = int(ev.sum())
    k = int(np.nansum(covered[ev])) if T else 0
    cov = k / T if T else np.nan
    lo_c, hi_c = _wilson(k, T) if T else (np.nan, np.nan)

    warns = []
    if method == "aci":
        slack = (max(alpha, 1 - alpha) + h * gamma) / (T * gamma) if T else np.inf
        guarantee = {"kind": "long-run share of misses (Gibbs and Candès 2021, Prop. 4.1" +
                     (")" if h == 1 else "; for h > 1 the same argument with the delayed update, proved for this skill, "
                      "not published)"), "slack": float(slack), "T": T}
        if h > 1:
            warns.append(_finding("notice", f"h = {h}: the paper proves the ACI guarantee for one-step forecasts; with errors "
                                  f"that arrive {h - 1} steps late, the bound with h·gamma used here was proved for this "
                                  "skill and is not published",
                                  "report the empirical coverage next to the interval and say it is measured"))
        if T and slack > 0.05:
            warns.append(_finding("notice", f"the long-run guarantee says only that the share of misses is within "
                                  f"±{slack:.2f} of {alpha:g} over these {T} forecasts (gamma {gamma:g}): not informative "
                                  "at this length", f"report the empirical coverage ({cov:.1%} of {T}) as measured, not as "
                                  "guaranteed; the guarantee tightens as forecasts accumulate"))
    else:
        guarantee = {"kind": "finite-sample coverage >= 1 - alpha, only if the errors are exchangeable "
                     "(Lei et al. 2018, Thm 2.2)", "slack": np.nan, "T": T}
        dep = None
        e_obs = (y - f)[obs]
        if len(e_obs) >= 20:
            lags = int(min(10, len(e_obs) // 5))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ps = [float(acorr_ljungbox(x - x.mean(), lags=[lags], return_df=True)["lb_pvalue"].iloc[0])
                      for x in (e_obs, np.abs(e_obs))]
            dep = min(ps)
        reasons = []
        if h > 1:
            reasons.append(f"errors of {h}-step forecasts overlap")
        if dep is not None and dep < _LB_LEVEL / 2:
            reasons.append(f"the errors or their size are autocorrelated (Ljung-Box, smaller p = {dep:.3g}, "
                           f"screen at {_LB_LEVEL:g} split over the two tests)")
        if reasons:
            warns.append(_finding("warning", "; ".join(reasons) + f": the {_pct(1 - alpha)} coverage of '{method}' intervals "
                                  "is not guaranteed (simulated: 84-95%, worst with few calibration errors)",
                                  "use method='aci', or report the empirical coverage as the interval's real level"))
        if 0 < (m_used[evaluated].min() if evaluated.any() else 0) < 50:
            warns.append(_finding("notice", f"intervals use as few as {m_used[evaluated].min()} past errors; with dependent "
                                  "errors such short calibration under-covers most", "use a longer forecast record or "
                                  "method='aci'"))
    is_inf = evaluated & np.isinf(half)
    n_inf = int(is_inf.sum())
    n_empty = int(empty.sum())
    # v2.7.1 (claude/medicoes-v2.7.1.md, M1): why the intervals are infinite. With h = 21 and the default gamma, 18-28% of the
    # intervals were infinite because ACI raised the level (runs of up to ~300 rows), not because of the warm-up: in the
    # validation of v2.7.0, 204 of 465 (176 at level 1, 28 at levels of 0.9925-0.999 with 87-369 past errors; none from the warm-up)
    # every infinite interval comes from a level above 1 - alpha (to 1, or close enough that ceil((m + 1)·level) > m): at
    # the nominal level an interval is computed only from m_min errors on, where the rank is at most m (rolling and split
    # never give one)
    lev_lo = float(np.min(level_used[is_inf])) if n_inf else np.nan
    runs = np.diff(np.flatnonzero(np.diff(np.r_[0, is_inf.astype(int), 0])))[::2]
    longest = int(runs.max()) if runs.size else 0
    fin_obs = evaluated & obs & ~np.isinf(half)            # empty intervals are finite (a sure miss) and count here
    T_fin = int(fin_obs.sum())
    cov_fin = float(np.mean(covered[fin_obs])) if T_fin else np.nan
    if n_inf or n_empty:
        fut = int((evaluated & ~obs & (np.isinf(half) | empty)).sum())
        why = []
        if n_inf:
            why.append(f"{n_inf} because ACI raised the level to {lev_lo:.4g} or more after misses that came in a run "
                       "(with m past errors a level above m/(m + 1) has no finite quantile)"
                       + (f"; each error is seen {h} steps late, so ACI corrects late" if h > 1 else ""))
        if n_empty:
            why.append(f"{n_empty} empty because the level went past 0 (a sure miss)")
        cov_txt = (f"; the infinite ones count as covered: coverage of the {T_fin} finite intervals with an actual is "
                   f"{cov_fin:.1%}, against {cov:.1%} in all" if n_inf and T_fin and T else "")
        share_inf = n_inf / max(int(evaluated.sum()), 1)
        head_txt = (f"{n_inf} of the {int(evaluated.sum())} interval(s) computed ({T} with an actual) are infinite and "
                    f"{n_empty} empty" if n_inf else
                    f"{n_empty} of the {int(evaluated.sum())} interval(s) computed ({T} with an actual) are empty")
        # review r2 R7: with empty intervals only, the advice is about gamma (they need a level pushed past 0)
        # v2.7.3 (v2.7.2 validation): the action was applied to one model only; it now says it is per model
        act_txt = ("report those rows as 'no informative interval', not as an interval, and give the coverage of the "
                   "finite intervals next to the total for these intervals, and do the same for every other model whose "
                   "intervals you report" if n_inf else
                   "report those rows as sure misses, not as intervals; they appear only when gamma is large enough to "
                   "push the level past 0 after a run of covered targets: a smaller gamma makes them rarer")
        warns.append(_finding("notice", head_txt
                              + (f", {fut} of them for targets not yet observed" if fut else "")
                              + (f" (longest run of infinite intervals: {longest} rows)" if n_inf else "")
                              + ": " + "; ".join(why) + cov_txt,
                              act_txt
                              + (". With overlapping errors (h > 1) this is expected: in simulations with h = 21 and 120 to "
                                 "1000 targets, 18-28% of the intervals were infinite on average; a smaller gamma removed "
                                 "them, but the coverage fell to that of method='rolling' (90-94% there), and no setting "
                                 "gave finite intervals at the nominal level, because overlapping errors leave about T/h "
                                 "independent ones. Compare with method='rolling' on these data" if n_inf and h > 1
                                 and share_inf >= 0.05 else "; a longer record makes them rarer" if n_inf else "")))
    if T and hi_c < 1 - alpha:
        act = ("the intervals are too narrow for these errors: widen them (method='aci') and say so" if method != "aci"
               else "the record is too short for the long-run guarantee to bind, or the errors changed: say the intervals "
                    "are too narrow, give the empirical coverage, and consider a larger gamma or a longer record")
        warns.append(_finding("warning", f"empirical coverage {cov:.1%} ({k} of {T}) is below {_pct(1 - alpha)} beyond its "
                              f"95% Wilson interval [{lo_c:.1%}, {hi_c:.1%}]", act))
    if T == 0:
        warns.append(_finding("warning", "no row could be evaluated: too few past errors before the last observed target"
                              + (f" (method 'split' calibrates on the first half of the {n_obs} rows with an actual, "
                                 f"fewer than the {m_min} errors a {_pct(1 - alpha)} interval needs)" if short_split else ""),
                              f"supply at least {need_split if method == 'split' else m_need + h} forecasts with their "
                              "actual values"))
    used = {"method": method, "alpha": alpha, "h": h, "calib_size": calib_size, "gamma": gamma if method == "aci" else None,
            "n_rows": n, "n_future": int((~obs).sum()), "warm_up_rows": int((~evaluated).sum()),
            "min_calib_errors": m_min, "coverage_ci_level": 0.95, "ljung_box_level": _LB_LEVEL if method != "aci" else None,
            "skill_version": SKILL_VERSION}
    out = {"table": table, "coverage": cov, "coverage_ci": (lo_c, hi_c), "n_evaluated": T, "guarantee": guarantee,
           "n_infinite": n_inf, "coverage_finite": cov_fin, "used": used, "warnings": warns}
    if show:
        print(f"\nConformal prediction intervals ({method}, level {_pct(1 - alpha)}, horizon {h})")
        if T:
            print(f"  empirical coverage: {cov:.1%} ({k} of {T} observed targets; Wilson 95% CI "
                  f"[{lo_c:.1%}, {hi_c:.1%}])")
        if n_inf and T_fin:
            print(f"  coverage of the {T_fin} finite intervals: {cov_fin:.1%} ({n_inf} infinite, counted as covered above)")
        if method == "aci" and T:
            print(f"  long-run guarantee: share of misses within ±{guarantee['slack']:.3f} of {alpha:g} over these {T} forecasts")
        last = table[evaluated].tail(3)
        if len(last):
            print(last[["actual", "forecast", "lower", "upper"]].to_string(float_format=lambda v: f"{v:.4g}"))
        _print_findings(warns)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Auto-ARIMA + walk-forward evaluation vs naive")
    ap.add_argument("file")
    ap.add_argument("--col", required=True)
    ap.add_argument("--date", default="date")
    ap.add_argument("--start", type=float, default=0.8)
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--refit-every", type=int, default=12)
    a = ap.parse_args()
    d = pd.read_csv(a.file, parse_dates=[a.date]).set_index(a.date).sort_index()
    y = d[a.col].dropna()
    n0 = int(a.start * len(y))
    best, _ = auto_arima(y.iloc[:n0], seasonal_period=a.season)
    so = best.model.seasonal_order
    fc = rolling_origin(y, best.model.order, start=a.start, seasonal_order=so,
                        trend=best.trend_used, refit_every=a.refit_every)
    bm = benchmarks(y, start=a.start, season=a.season)
    models = {"naive": bm["naive"]}
    if a.season:
        models["seasonal_naive"] = bm["seasonal_naive"]
    models["auto_arima"] = fc["forecast"]
    evaluate(fc["actual"], models, last_actual=bm["naive"])
