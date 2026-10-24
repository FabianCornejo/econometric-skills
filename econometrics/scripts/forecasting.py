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

Requires: statsmodels >= 0.13, scipy. pmdarima is NOT required.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.statespace.sarimax import SARIMAX

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
    """Pesaran-Timmermann (1992) test. H0: forecast signs are independent of actual signs."""
    a = np.sign(np.asarray(actual, float))
    f = np.sign(np.asarray(forecast, float))
    m = (a != 0) & (f != 0)
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

    direction='level'  : directional accuracy of the predicted CHANGE vs the actual change;
                         needs last_actual (value at the origin) — e.g. naive forecast for h=1.
    direction='return' : directional accuracy of the sign itself (series already a return).
    insample_scale     : MAE of the in-sample naive forecast, for MASE.
    """
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
    """Compare several forecasts over the common sample; DM test of each vs `benchmark`."""
    from diagnostics import diebold_mariano
    df = pd.concat({"actual": actual, **forecasts}, axis=1).dropna()
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
    if show:
        print(f"\nOut-of-sample evaluation ({len(df)} forecasts, benchmark = {benchmark})")
        print(tab.to_string(float_format=lambda v: f"{v:.4f}"))
        for k in forecasts:
            if k == benchmark:
                continue
            p = tab.loc[k, "DM_p"]
            better = tab.loc[k, "RMSE"] < tab.loc[benchmark, "RMSE"]
            print(f"  {k}: " + ("significantly " if p < ALPHA else "not significantly ")
                  + ("more" if better else "less") + f" accurate than {benchmark} (DM p = {p:.3f})")
    return tab


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
