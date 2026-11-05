"""Univariate time series: identify, detrend, test for unit roots and breaks, long memory, bootstrap forecasts
and forecast combination (v2.4.6). References: time-series.md sections 2-5 and 11; workflow rule: clean the
series, then test for breaks, then for unit roots (an unmodelled break looks like a unit root).

Use as a module:
    import sys; sys.path.insert(0, "<skill folder>/scripts")
    from ts_univariate import *

Functions
    ts_identify(y)                        ACF (Bartlett bands) and PACF (±z/sqrt(n)), Ljung-Box, the orders the
                                          correlogram suggests and a BIC grid of small ARMA models
    detrend(y, method)                    polynomial trend, HP (two-sided or one-sided/real-time), Hamilton,
                                          Baxter-King, Christiano-Fitzgerald; reports what each filter loses or
                                          revises at the end of the sample
    adf_lags(y)                           ADF lag order by AIC (default), BIC, the general-to-specific
                                          t >= 1.645 rule or the modified AIC (Ng-Perron)
    adf_bootstrap(y)                      ADF with sieve-bootstrap p-value (residuals of an AR fitted to the
                                          differences under H0) next to the asymptotic MacKinnon p-value
    ar_bootstrap_forecast(y, h)           AR(p) forecasts with bootstrap intervals that include parameter
                                          uncertainty, next to the normal analytic interval
    sup_wald(y, X)                        Andrews sup-Wald (sup-F) test for one break at an unknown date with a
                                          sieve-bootstrap p-value, plus CUSUM and recursive coefficients
    long_memory(y)                        fractional d by GPH (log-periodogram) and local Whittle, with SEs
    frac_diff(y, d)                       (1 - L)^d filter
    arfima(y, order)                      d by local Whittle, then ARMA(p, q) on the fractionally differenced series
    combine_forecasts(actual, forecasts)  equal weights (default), inverse MSE, Bates-Granger minimum variance or
                                          Granger-Ramanathan weights, estimated on a training window and
                                          evaluated after it

Every number comes from the data passed in. Bootstrap functions take B and seed.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats
import _console  # noqa: F401  (Windows-safe console output, see _console.py)

ALPHA = 0.05


def _fmt(v):
    return f"{v:.4f}" if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v)


def _series(y) -> pd.Series:
    s = pd.Series(y).astype(float)
    if s.isna().any():
        inner = s.loc[s.first_valid_index():s.last_valid_index()]
        if inner.isna().any():
            raise ValueError("missing values inside the series: fill or model them before time-series tools")
        s = inner
    if len(s) < 20:
        raise ValueError("too few observations for time-series tools (need at least 20)")
    return s


def _ols(y, X):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    return b, e


# =============================================================================
# Identification
# =============================================================================
def ts_identify(y, nlags: int = 24, alpha: float = ALPHA, max_pq: int = 3, show: bool = True) -> dict:
    """Correlogram of a (stationary) series with the bands that make it readable.

    ACF bands: ±z·sqrt((1 + 2 Σ_{j<k} r_j²)/n) (Bartlett: under an MA(k−1), wider as k grows); PACF bands:
    ±z/sqrt(n). Ljung-Box at 6, 12 and 24 lags (or fewer). Suggested orders: the last lag before the PACF (for an
    AR) or the ACF (for an MA) stays inside its band for three lags in a row, and the (p, q) with the lowest BIC
    in a grid up to max_pq. The correlogram is a starting point; residual checks decide. Difference the series
    first if it has a unit root (adf_bootstrap, diagnostics.stationarity)."""
    from statsmodels.tsa.stattools import acf, pacf
    from statsmodels.stats.diagnostic import acorr_ljungbox
    s = _series(y)
    n = len(s)
    nlags = min(nlags, n // 2 - 1)
    z = stats.norm.ppf(1 - alpha / 2)
    r = acf(s, nlags=nlags, fft=True)[1:]
    ph = pacf(s, nlags=nlags, method="ywm")[1:]
    acf_band = z * np.sqrt((1 + 2 * np.concatenate([[0], np.cumsum(r[:-1] ** 2)])) / n)
    pacf_band = np.full(nlags, z / np.sqrt(n))
    tab = pd.DataFrame({"acf": r, "acf_band": acf_band, "acf_sig": np.abs(r) > acf_band,
                        "pacf": ph, "pacf_band": pacf_band, "pacf_sig": np.abs(ph) > pacf_band},
                       index=pd.RangeIndex(1, nlags + 1, name="lag"))

    def cutoff(sig):
        for k in range(len(sig)):
            if not sig[k:k + 3].any():
                return k                          # lags 1..k significant at most, then three quiet lags
        return None
    p_cut, q_cut = cutoff(tab["pacf_sig"].to_numpy()), cutoff(tab["acf_sig"].to_numpy())
    lags_lb = [k for k in (6, 12, 24) if k <= nlags] or [nlags]
    lb = acorr_ljungbox(s, lags=lags_lb, return_df=True)
    grid = []
    from statsmodels.tsa.arima.model import ARIMA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p_ in range(max_pq + 1):
            for q_ in range(max_pq + 1):
                try:
                    m = ARIMA(s.to_numpy(), order=(p_, 0, q_), trend="c").fit()
                    grid.append({"p": p_, "q": q_, "bic": m.bic, "aic": m.aic})
                except Exception:              # non-invertible or failed fits are skipped
                    continue
    grid = pd.DataFrame(grid).sort_values("bic").reset_index(drop=True)
    best = (int(grid.loc[0, "p"]), int(grid.loc[0, "q"])) if len(grid) else None
    suggestion = []
    if p_cut is not None and p_cut > 0:
        suggestion.append(f"PACF cuts off after lag {p_cut} → AR({p_cut}) candidate")
    if q_cut is not None and q_cut > 0:
        suggestion.append(f"ACF cuts off after lag {q_cut} → MA({q_cut}) candidate")
    if p_cut == 0 and q_cut == 0:
        suggestion.append("no significant autocorrelation: white noise (check Ljung-Box)")
    out = {"table": tab, "ljung_box": lb, "pacf_cutoff": p_cut, "acf_cutoff": q_cut, "bic_grid": grid,
           "bic_best": best, "suggestion": suggestion, "n": n}
    if show:
        print(f"\nCorrelogram (n = {n}; ACF bands Bartlett, PACF bands ±{z:.2f}/√n)")
        print(tab.head(min(nlags, 12)).to_string(float_format=_fmt))
        print("\nLjung-Box:\n" + lb.to_string(float_format=_fmt))
        for line in suggestion:
            print("•", line)
        if best:
            print(f"• BIC grid (p, q ≤ {max_pq}) picks ARMA{best}")
    return out


# =============================================================================
# Trends and filters
# =============================================================================
def detrend(y, method: str = "hp", lamb: float | None = None, freq: str | None = None, one_sided: bool = False,
            degree: int = 1, low: float | None = None, high: float | None = None, K: int | None = None,
            h: int | None = None, show: bool = True) -> dict:
    """Split a series into trend and cycle, and say what the filter does at the end of the sample.

    method: 'poly' (polynomial in time, OLS), 'hp' (Hodrick-Prescott; lamb 1600 quarterly, 129600 monthly,
    6.25 annual by Ravn-Uhlig; one_sided=True recomputes the filter at each date with data up to that date, the
    real-time version), 'hamilton' (Hamilton 2018: cycle = residual of y_{t+h} on y_t..y_{t-3}; h = 8 quarterly,
    24 monthly, 2 annual), 'bk' (Baxter-King band pass, loses K observations at each end), 'cf' (Christiano-
    Fitzgerald asymmetric band pass). Periods for bk/cf: low and high in periods (6-32 quarters by default).

    A filter is not a forecast: the trend at the last date of a two-sided filter uses no future data that exist,
    so it is revised when new data arrive (the end-point problem); `end_revision` measures it by comparing the
    two-sided and the one-sided HP trend over the last observations. Do not extrapolate a filtered trend."""
    s = _series(y)
    n = len(s)
    warn = []
    f = freq or (pd.infer_freq(s.index) if isinstance(s.index, pd.DatetimeIndex) and len(s) >= 3 else None)
    if isinstance(s.index, pd.PeriodIndex) and f is None:
        f = s.index.freqstr
    if f is None:
        f = "Q"
        warn.append("frequency unknown: quarterly settings assumed (λ 1600, Hamilton h 8, band 6-32); pass freq")
    f = f.upper()
    f = f[1:] if f.startswith("B") and f[1:2] in ("M", "Q", "A", "Y") else f   # business month/quarter/year ends
    per_year = 12 if f.startswith("M") else (4 if f.startswith("Q") else (1 if f.startswith(("A", "Y")) else None))
    if per_year is None:
        needed = {"hp": [("lamb", lamb)], "hamilton": [("h", h)], "bk": [("low", low), ("high", high), ("K", K)],
                  "cf": [("low", low), ("high", high)]}.get(method, [])
        missing = [nm for nm, v in needed if v is None]
        if missing:
            raise ValueError(f"no default settings for frequency '{f}': pass {', '.join(missing)} (defaults exist "
                             "for monthly, quarterly and annual data)")
        per_year = 4                               # only used for the end-of-sample window below
    if lamb is None:
        lamb = {12: 129600.0, 4: 1600.0, 1: 6.25}[per_year]
    trend = pd.Series(np.nan, index=s.index)
    from statsmodels.tsa.filters.hp_filter import hpfilter
    if method == "poly":
        t = np.arange(n, dtype=float)
        X = np.vander((t - t.mean()) / t.std(), degree + 1, increasing=True)
        b, e = _ols(s.to_numpy(), X)
        trend[:] = X @ b
    elif method == "hp":
        if one_sided:
            vals = np.full(n, np.nan)
            for t_ in range(9, n):                # HP needs a few points; real-time trend at each date
                vals[t_] = hpfilter(s.iloc[:t_ + 1].to_numpy(), lamb=lamb)[1][-1]
            trend[:] = vals
            warn.append("one-sided HP: the first observations have no estimate (not enough data at that date)")
        else:
            trend[:] = hpfilter(s.to_numpy(), lamb=lamb)[1]
            warn.append("two-sided HP: the trend near the end of the sample uses no future data and will be "
                        "revised; for real-time decisions use one_sided=True or method='hamilton'")
    elif method == "hamilton":
        hh = h or {12: 24, 4: 8, 1: 2}[per_year]
        p_ = 4
        yv = s.to_numpy()
        rows = range(p_ - 1, n - hh)
        X = np.column_stack([np.ones(len(rows))] + [yv[[t_ - j for t_ in rows]] for j in range(p_)])
        target = yv[[t_ + hh for t_ in rows]]
        b, e = _ols(target, X)
        trend.iloc[[t_ + hh for t_ in rows]] = X @ b
        warn.append(f"Hamilton filter: the first {hh + p_ - 1} observations have no cycle; the cycle at t is the "
                    f"surprise relative to the forecast made {hh} periods earlier")
    elif method in ("bk", "cf"):
        lo_p = low or (6 if per_year == 4 else (18 if per_year == 12 else 2))
        hi_p = high or (32 if per_year == 4 else (96 if per_year == 12 else 8))
        from statsmodels.tsa.filters.bk_filter import bkfilter
        from statsmodels.tsa.filters.cf_filter import cffilter
        if method == "bk":
            KK = K or (12 if per_year == 4 else (36 if per_year == 12 else 3))
            cyc = bkfilter(s.to_numpy(), low=lo_p, high=hi_p, K=KK)
            trend.iloc[KK:n - KK] = s.to_numpy()[KK:n - KK] - np.asarray(cyc).ravel()
            warn.append(f"Baxter-King loses {KK} observations at each end (no estimate there)")
        else:
            cyc, _ = cffilter(s.to_numpy(), low=lo_p, high=hi_p, drift=True)
            trend[:] = s.to_numpy() - np.asarray(cyc).ravel()
            warn.append("Christiano-Fitzgerald is asymmetric: the last values use only past data and are revised "
                        "as data arrive")
    else:
        raise ValueError("method must be 'poly', 'hp', 'hamilton', 'bk' or 'cf'")
    cycle = s - trend
    out = {"trend": trend, "cycle": cycle, "method": method, "lamb": lamb if method == "hp" else None,
           "warnings": warn}
    m_end = max(4, per_year * 2)
    if method == "hp" and not one_sided:
        rt = detrend(s, "hp", lamb=lamb, one_sided=True, show=False)["trend"]
        rev = (trend - rt).iloc[-m_end:]
        # real-time (data up to t) vs final (full sample) trend: the revision the latest values will undergo
        out["end_revision"] = {"typical_revision": float((trend - rt).iloc[n // 3: 2 * n // 3].abs().mean()),
                               "revision_to_date_last_window": float(rev.abs().mean()),
                               "cycle_sd": float(cycle.std()), "window": m_end}
    if show:
        print(f"\nDetrend ({method}{', one-sided' if one_sided else ''}; n = {n})")
        print(pd.DataFrame({"y": s, "trend": trend, "cycle": cycle}).tail(6).to_string(float_format=_fmt))
        if "end_revision" in out:
            er = out["end_revision"]
            print(f"End-point revision: the real-time trend differs from the final one by {er['typical_revision']:.4f} "
                  f"on average (middle third), against a cycle SD of {er['cycle_sd']:.4f}; the last values will move "
                  f"by about that much as data arrive")
        for w in warn:
            print("Note:", w)
    return out


# =============================================================================
# Unit roots
# =============================================================================
def _adf_design(y: np.ndarray, p: int, trend: str):
    dy = np.diff(y)
    T = len(dy) - p
    cols = [y[p:-1]]
    for i in range(1, p + 1):
        cols.append(dy[p - i:len(dy) - i])
    if trend in ("c", "ct"):
        cols.append(np.ones(T))
    if trend == "ct":
        cols.append(np.arange(p + 1, p + 1 + T, dtype=float))
    return dy[p:], np.column_stack(cols)


def _adf_stat(y: np.ndarray, p: int, trend: str) -> tuple[float, np.ndarray]:
    target, X = _adf_design(y, p, trend)
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ target
    e = target - X @ b
    s2 = e @ e / (len(target) - X.shape[1])
    se = np.sqrt(s2 * np.diag(XtXi))
    return float(b[0] / se[0]), b / se


def adf_lags(y, trend: str = "c", max_lags: int | None = None, rule: str = "aic") -> int:
    """ADF lag order, all candidates compared on the sample of the largest model (max_lags by Schwert:
    12·(n/100)^(1/4)).

    rule='aic' (default) or 'bic': statsmodels' choice. rule='t16' (general-to-specific, Ng-Perron 1995):
    start at max_lags and drop the last lag while its |t| < 1.645. rule='maic' (Ng-Perron 2001 modified AIC on
    OLS-detrended data, Perron-Qu 2007). With the sieve bootstrap of adf_bootstrap, simulations for v2.4.6
    (n = 150) gave: AIC size 6% (2.5% with MA(−0.5) errors) and power 100% against AR(0.5); BIC over-rejects
    with MA errors (10%); t16 and MAIC keep the size but often stop at long lags under stationarity and lose
    power (65–78% against AR(0.5)). Too few lags distort the size; too many cost power."""
    yv = _series(y).to_numpy()
    n = len(yv)
    pmax = max_lags if max_lags is not None else int(np.floor(12 * (n / 100) ** 0.25))
    if rule == "t16":
        for p in range(pmax, 0, -1):
            _, t = _adf_stat(yv[pmax - p:], p, trend)
            if abs(t[p]) >= 1.645:               # coefficient of the last lagged difference
                return p
        return 0
    if rule == "maic":
        if trend == "n":
            yd = yv - 0.0
        else:
            D = np.column_stack([np.ones(n)] + ([np.arange(n, dtype=float)] if trend == "ct" else []))
            yd = yv - D @ np.linalg.lstsq(D, yv, rcond=None)[0]
        dy = np.diff(yd)
        N = len(dy) - pmax
        ylag = yd[pmax:-1]
        best, best_k = np.inf, 0
        for k in range(pmax + 1):
            X = np.column_stack([ylag] + [dy[pmax - i:len(dy) - i] for i in range(1, k + 1)])
            b, e = _ols(dy[pmax:], X)
            s2 = e @ e / N
            tau = b[0] ** 2 * (ylag @ ylag) / s2
            crit = np.log(s2) + 2 * (tau + k) / N
            if crit < best:
                best, best_k = crit, k
        return best_k
    if rule not in ("aic", "bic"):
        raise ValueError("rule must be 'maic', 't16', 'aic' or 'bic'")
    from statsmodels.tsa.stattools import adfuller
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return int(adfuller(yv, maxlag=pmax, regression=trend, autolag=rule.upper())[2])


def adf_bootstrap(y, trend: str = "c", lags: int | None = None, lag_rule: str = "aic", B: int = 999, seed: int = 0,
                  alpha: float = ALPHA, show: bool = True) -> dict:
    """Augmented Dickey-Fuller test of H0: unit root, with a sieve-bootstrap p-value.

    The statistic is the usual t of γ in Δy_t = [c (+ δt)] + γ y_{t−1} + Σ φ_i Δy_{t−i} + e_t (lags by the
    AIC over a common sample unless given; lag_rule 't16', 'maic' or 'bic' as in adf_lags). Bootstrap under H0 (Park 2003; Chang-Park 2003): fit an AR(p) to Δy, resample
    its centred residuals, rebuild Δy* and y* = cumulated, recompute the statistic B times; p = share of
    bootstrap statistics at or below the observed one. Each replicate uses the lag order chosen on the data. Better size than the asymptotic MacKinnon p-value in
    small samples and with MA errors; it is shown alongside. Not rejecting is not evidence of a unit root:
    unit-root tests have low power against persistent stationary series (simulation.demo_unit_root_power), and
    a break makes a stationary series look I(1) (simulation.demo_break_unit_root). A break test in levels assumes
    stationarity, so run sup_wald on the differences (a break in the drift) or use Zivot-Andrews, which allows a
    break under both hypotheses."""
    from statsmodels.tsa.stattools import adfuller
    s = _series(y)
    yv = s.to_numpy()
    if trend not in ("n", "c", "ct"):
        raise ValueError("trend must be 'n', 'c' or 'ct'")
    p = adf_lags(yv, trend, rule=lag_rule) if lags is None else int(lags)
    tau, _ = _adf_stat(yv, p, trend)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        asym = adfuller(yv, maxlag=p, regression=trend, autolag=None)
    # sieve under H0: Δy_t = μ + Σ a_i Δy_{t-i} + u_t  (μ only with 'ct', where a drift is allowed under H0)
    dy = np.diff(yv)
    if p > 0:
        X = np.column_stack([dy[p - i:len(dy) - i] for i in range(1, p + 1)])
    else:
        X = np.empty((len(dy), 0))
    if trend == "ct":
        X = np.column_stack([np.ones(len(dy) - p), X]) if p > 0 else np.ones((len(dy), 1))
    target = dy[p:]
    a, u = _ols(target, X) if X.shape[1] else (np.zeros(0), target - 0.0)
    u = u - u.mean()
    mu = a[0] if trend == "ct" else 0.0
    phi = a[1:] if trend == "ct" else a
    warn = []
    if p and np.any(np.abs(np.roots(np.r_[1, -phi])) >= 1):
        warn.append("the AR fitted to the differences is not stationary (the series may be I(2)): the bootstrap "
                    "p-value is unreliable; test the differences")
    rng = np.random.default_rng(seed)
    n_dy = len(dy)
    burn = 50
    taus = np.empty(B)
    for b in range(B):
        e = rng.choice(u, size=n_dy + burn, replace=True)
        d = np.zeros(n_dy + burn)
        for t_ in range(n_dy + burn):
            ar = sum(phi[i] * d[t_ - 1 - i] for i in range(p) if t_ - 1 - i >= 0)
            d[t_] = mu + ar + e[t_]
        ystar = np.concatenate([[yv[0]], yv[0] + np.cumsum(d[burn:])])
        taus[b] = _adf_stat(ystar, p, trend)[0]
    p_boot = float((np.sum(taus <= tau) + 1) / (B + 1))
    crit = {f"{q:.0%}": float(np.quantile(taus, q)) for q in (0.01, 0.05, 0.10)}
    verdict = ("rejects the unit root: evidence that the series is stationary" + (" around a trend" if trend == "ct" else "")
               if p_boot < alpha else
               "does not reject the unit root (not evidence that there is one: low power; check KPSS, breaks and the plot)")
    out = {"estimand": "ADF τ (H0: unit root)", "statistic": tau, "p_bootstrap": p_boot, "p_asymptotic": float(asym[1]),
           "lags": p, "lag_rule": lag_rule if lags is None else "given", "trend": trend, "n": len(yv), "B": B,
           "bootstrap_critical_values": crit, "asymptotic_critical_values": asym[4], "verdict": verdict + f" at {alpha:.0%}",
           "warnings": warn, "method": "ADF with sieve bootstrap under H0"}
    if show:
        print(f"\nADF (trend '{trend}', {p} lags by {out['lag_rule']}, n = {len(yv)}): τ = {tau:.3f}")
        print(f"p bootstrap (B = {B}) = {p_boot:.4f}; p asymptotic (MacKinnon) = {asym[1]:.4f}")
        print("Bootstrap critical values: " + ", ".join(f"{k} {v:.3f}" for k, v in crit.items()))
        print("→", out["verdict"])
        for w in warn:
            print("Warning:", w)
    return out


# =============================================================================
# Forecasting with bootstrap intervals
# =============================================================================
def _fit_ar(yv: np.ndarray, p: int):
    X = np.column_stack([np.ones(len(yv) - p)] + [yv[p - i:len(yv) - i] for i in range(1, p + 1)])
    b, e = _ols(yv[p:], X)
    return b, e


def _future_index(index: pd.Index, h: int) -> pd.Index:
    """Labels for the h periods after the last one: dates or periods continue the calendar, integer labels
    (years, say) continue their step, anything else becomes 1..h after the last position."""
    if isinstance(index, pd.PeriodIndex):
        return pd.PeriodIndex([index[-1] + k for k in range(1, h + 1)])
    if isinstance(index, pd.DatetimeIndex):
        freq = index.freq or (pd.infer_freq(index) if len(index) >= 3 else None)
        if freq is not None:
            return pd.date_range(index[-1], periods=h + 1, freq=freq)[1:]
    elif pd.api.types.is_integer_dtype(index) and len(index) >= 2:
        steps = np.diff(np.asarray(index[-3:]))
        if (steps == steps[0]).all() and steps[0] > 0:
            return pd.Index(index[-1] + steps[0] * np.arange(1, h + 1))
    return pd.Index([f"T+{k}" for k in range(1, h + 1)])


def _shrink_to_stationary(coef: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Kilian (1998): subtract the bias from the AR slopes, shrinking the correction until the roots are
    stationary."""
    delta = 1.0
    while delta > 0:
        c = coef - delta * bias
        if not len(c) or np.all(np.abs(np.roots(np.r_[1, -c])) < 1):
            return c
        delta -= 0.01
    return coef


def ar_bootstrap_forecast(y, h: int = 8, p: int | None = None, max_p: int = 8, B: int = 999, alpha: float = ALPHA,
                          bias_correct: bool = True, seed: int = 0, show: bool = True) -> dict:
    """h-step forecasts of a stationary AR(p) with constant, with bootstrap prediction intervals.

    p by BIC (0 to max_p, common sample) unless given. Each bootstrap replicate rebuilds a series from the AR and
    resampled residuals (rescaled by sqrt((n − p)/(n − 2p − 1)), Stine 1987), re-estimates the coefficients on
    it (parameter uncertainty), and forecasts from the actual last p observations with those coefficients and
    future resampled shocks (Thombs-Schucany 1990). OLS slopes of an AR are biased toward zero in short samples,
    which makes bootstrap paths revert too fast and the intervals too narrow; with bias_correct=True (Kilian
    1998, bootstrap-after-bootstrap) a first bootstrap estimates that bias, the series are generated from the
    corrected slopes and each re-estimate is corrected the same way (shrunk to stay stationary). The point
    forecast is the plain OLS one. The analytic interval uses σ̂²Σψ_j² with normal errors and treats the
    coefficients as known. In short samples both intervals undercover; compare them, and check coverage by
    rolling origin (forecasting.rolling_origin). Difference an I(1) series first."""
    s = _series(y)
    yv = s.to_numpy()
    n = len(yv)
    if p is None:
        bics = {}
        for pp in range(0, max_p + 1):
            b_, e_ = _fit_ar(yv[max_p - pp:], pp)
            T = len(e_)
            bics[pp] = T * np.log(e_ @ e_ / T) + (pp + 1) * np.log(T)
        p = min(bics, key=bics.get)
    b, e = _fit_ar(yv, p)
    if p and np.any(np.abs(np.roots(np.r_[1, -b[1:]])) >= 1):
        raise ValueError("the fitted AR has a unit or explosive root: difference the series first")
    sigma = np.sqrt(e @ e / (len(e) - p - 1))
    mean_y = b[0] / (1 - b[1:].sum()) if p else b[0]

    def path(coef, last, shocks):
        hist = list(last)
        out_ = []
        for t_ in range(len(shocks)):
            v = coef[0] + sum(coef[1 + i] * hist[-1 - i] for i in range(p)) + shocks[t_]
            hist.append(v)
            out_.append(v)
        return np.array(out_)
    point = path(b, yv[-p:] if p else [], np.zeros(h))
    psi = np.zeros(h)
    psi[0] = 1.0
    for j in range(1, h):
        psi[j] = sum(b[1 + i] * psi[j - 1 - i] for i in range(min(p, j)))
    se_an = sigma * np.sqrt(np.cumsum(psi ** 2))
    z = stats.norm.ppf(1 - alpha / 2)
    rng = np.random.default_rng(seed)
    infl = np.sqrt((n - p) / max(n - 2 * p - 1, 1))
    ec = (e - e.mean()) * infl
    burn = 100

    def refit(coef):
        ystar = path(coef, yv[:p] if p else [], rng.choice(ec, size=n + burn))[-n:]
        return _fit_ar(ystar, p)[0]

    def with_mean(slopes):                        # keep the sample mean when the slopes change
        return np.r_[mean_y * (1 - slopes.sum()), slopes]
    bias = np.zeros(p)
    gen = b
    if bias_correct and p:
        draws = np.array([refit(b)[1:] for _ in range(min(B, 499))])
        bias = draws.mean(axis=0) - b[1:]
        gen = with_mean(_shrink_to_stationary(b[1:], bias))
    sims = np.empty((B, h))
    for r in range(B):
        bs = refit(gen)
        if p:
            sl = _shrink_to_stationary(bs[1:], bias) if bias_correct else bs[1:]
            if np.any(np.abs(np.roots(np.r_[1, -sl])) >= 1):
                sl = gen[1:]                       # explosive draw: keep the generating slopes
            # keep the replicate's implied mean c*/(1 − Σφ*) when its slopes are corrected
            m_star = bs[0] / (1 - bs[1:].sum()) if abs(1 - bs[1:].sum()) > 1e-8 else mean_y
            bs = np.r_[m_star * (1 - sl.sum()), sl]
        sims[r] = path(bs, yv[-p:] if p else [], rng.choice(ec, size=h))
    tab = pd.DataFrame({"forecast": point, "boot_low": np.quantile(sims, alpha / 2, axis=0),
                        "boot_high": np.quantile(sims, 1 - alpha / 2, axis=0),
                        "analytic_low": point - z * se_an, "analytic_high": point + z * se_an},
                       index=_future_index(s.index, h))
    tab.index.name = "period"
    out = {"table": tab, "p": p, "coef": b, "sigma": sigma, "n": n, "B": B, "level": 1 - alpha,
           "bias_correction": pd.Series(-bias, index=[f"ar.L{i}" for i in range(1, p + 1)], dtype=float),
           "method": "AR(p) by OLS; bootstrap intervals with re-estimation"
                     + (" and Kilian bias correction" if bias_correct and p else "")}
    if show:
        print(f"\nAR({p}) forecasts, {1 - alpha:.0%} intervals (bootstrap B = {B} vs analytic normal), n = {n}")
        print(tab.to_string(float_format=_fmt))
    return out


# =============================================================================
# Structural breaks
# =============================================================================
def _chow_F(yv, X, k, trim_lo, trim_hi):
    n = len(yv)
    b0, e0 = _ols(yv, X)
    ssr0 = e0 @ e0
    Fs = np.full(n, np.nan)
    for tb in range(trim_lo, trim_hi + 1):
        D = (np.arange(n) >= tb).astype(float)[:, None]
        Xb = np.column_stack([X, X * D])
        b1, e1 = _ols(yv, Xb)
        ssr1 = e1 @ e1
        Fs[tb] = ((ssr0 - ssr1) / k) / (ssr1 / (n - 2 * k))
    return Fs


def sup_wald(y, X=None, trim: float = 0.15, B: int = 499, ar_max: int = 4, seed: int = 0, alpha: float = ALPHA,
             show: bool = True) -> dict:
    """Test for one break at an unknown date in y = Xβ + u (X = constant only: a break in the mean).

    Andrews (1993) sup-F: the Chow F for a change in all coefficients at each date in the middle 1 − 2·trim of
    the sample; its maximum has a non-standard distribution, so the p-value comes from a sieve bootstrap under
    H0 (no break): fit an AR(p) by BIC to the residuals of the full-sample regression, rebuild errors from
    resampled AR residuals, add them to the fitted values and recompute sup-F B times. Also the OLS-CUSUM test
    (Ploberger-Krämer) and the recursive coefficients, for gradual change. The date of the maximum is the
    break-date estimate; with no evidence of a break, it means nothing."""
    from statsmodels.stats.diagnostic import breaks_cusumolsresid
    s = _series(y)
    yv = s.to_numpy()
    n = len(yv)
    if X is None:
        Xv = np.ones((n, 1))
        names = ["const"]
    else:
        Xd = pd.DataFrame(X)
        y0 = pd.Series(y)
        if Xd.index.equals(y0.index) or s.index.isin(Xd.index).all() and not isinstance(X, np.ndarray):
            Xd = Xd.loc[s.index]
        elif len(Xd) == len(y0):                  # no shared labels: align by position with y as given
            Xd = Xd.set_axis(y0.index).loc[s.index]
        else:
            raise ValueError("X must have y's index or the same number of rows as y")
        Xd = Xd.astype(float)
        if Xd.isna().any().any():
            raise ValueError("missing values in X")
        Xv = np.column_stack([np.ones(n), Xd.to_numpy()])
        names = ["const"] + list(Xd.columns)
    k = Xv.shape[1]
    lo, hi = int(np.floor(trim * n)), int(np.ceil((1 - trim) * n)) - 1
    if hi - lo < 5 or lo < k + 1:
        raise ValueError("sample too short for this trimming")
    Fs = _chow_F(yv, Xv, k, lo, hi)
    supF = float(np.nanmax(Fs))
    tb = int(np.nanargmax(Fs))
    b0, u = _ols(yv, Xv)
    # sieve AR(p) for the errors under H0
    # sieve order by BIC, every candidate on the common sample u[ar_max:] (otherwise the comparison depends on
    # the units of y), then re-fitted on all the residuals it can use
    T_c = len(u) - ar_max
    bics = []
    for pp in range(0, ar_max + 1):
        Z = (np.column_stack([u[ar_max - i:len(u) - i] for i in range(1, pp + 1)]) if pp
             else np.empty((T_c, 0)))
        r_ = u[ar_max:] - (Z @ _ols(u[ar_max:], Z)[0] if pp else 0.0)
        bics.append(T_c * np.log(r_ @ r_ / T_c) + pp * np.log(T_c))
    pp = int(np.argmin(bics))
    if pp:
        a, r = _ols(u[pp:], np.column_stack([u[pp - i:len(u) - i] for i in range(1, pp + 1)]))
    else:
        a, r = np.zeros(0), u.copy()
    r = r - r.mean()
    warn = []
    root = float(np.max(np.abs(np.roots(np.r_[1, -a])))) if pp else 0.0
    if root > 0.9:
        warn.append(f"the residuals are very persistent (largest AR root {root:.2f}): a break test in levels assumes "
                    "a stationary series, and a random walk looks like a series with breaks; test the differences "
                    "or use a unit-root test that allows a break (Zivot-Andrews)")
    rng = np.random.default_rng(seed)
    fitted = Xv @ b0
    sups = np.empty(B)
    burn = 50
    for bb in range(B):
        eps = rng.choice(r, size=n + burn)
        ustar = np.zeros(n + burn)
        for t_ in range(n + burn):
            ustar[t_] = eps[t_] + sum(a[i] * ustar[t_ - 1 - i] for i in range(pp) if t_ - 1 - i >= 0)
        sups[bb] = np.nanmax(_chow_F(fitted + ustar[burn:], Xv, k, lo, hi))
    p_boot = float((np.sum(sups >= supF) + 1) / (B + 1))
    cus_stat, cus_p, _ = breaks_cusumolsresid(u, ddof=k)
    rec = []
    for t_ in range(max(k + 5, lo // 2), n + 1):
        b_t, _ = _ols(yv[:t_], Xv[:t_])
        rec.append(b_t)
    rec = pd.DataFrame(rec, columns=names, index=s.index[max(k + 5, lo // 2) - 1:n])
    date = s.index[tb]
    out = {"estimand": "sup-F (H0: no break)", "statistic": supF, "p_bootstrap": p_boot, "break_date": date,
           "break_position": tb, "F_path": pd.Series(Fs, index=s.index), "cusum": {"statistic": float(cus_stat),
           "p": float(cus_p)}, "recursive_coefficients": rec, "sieve_ar_order": pp, "trim": trim, "n": n, "B": B,
           "bootstrap_critical_values": {f"{q:.0%}": float(np.quantile(sups, 1 - q)) for q in (0.10, 0.05, 0.01)},
           "verdict": ((f"evidence of a break (most likely at {date})" if p_boot < alpha else
                        "no evidence of a break at an unknown date (not proof of stability; check CUSUM and the "
                        "recursive coefficients)") + f" at {alpha:.0%}"),
           "warnings": warn, "method": "Andrews sup-F with sieve bootstrap; OLS-CUSUM"}
    if show:
        print(f"\nsup-F test for one break (trim {trim:.0%}, k = {k}, n = {n}): sup-F = {supF:.3f} at {date}")
        print(f"p bootstrap (B = {B}, sieve AR({pp})) = {p_boot:.4f}; OLS-CUSUM = {cus_stat:.3f}, p = {cus_p:.4f}")
        print("→", out["verdict"])
        for w in warn:
            print("Warning:", w)
    return out


# =============================================================================
# Long memory
# =============================================================================
def _periodogram(x: np.ndarray, m: int):
    n = len(x)
    xc = x - x.mean()
    j = np.arange(1, m + 1)
    lam = 2 * np.pi * j / n
    f = np.fft.fft(xc)
    I = (np.abs(f[1:m + 1]) ** 2) / (2 * np.pi * n)
    return lam, I


def long_memory(y, power_gph: float = 0.5, power_lw: float = 0.65, alpha: float = ALPHA, show: bool = True) -> dict:
    """Fractional integration order d of a series (y ~ (1 − L)^{−d}·short-memory).

    GPH (Geweke-Porter-Hudak 1983): regress log I(λ_j) on −2·log(2 sin(λ_j/2)) for the first m = n^0.5
    Fourier frequencies; SE = π/sqrt(24 m). Local Whittle (Robinson 1995): minimise
    log(mean(λ_j^{2d} I_j)) − 2d·mean(log λ_j) over m = n^0.65 frequencies; SE = 1/(2 sqrt(m)), more efficient.
    d ≈ 0: short memory; 0 < d < 0.5: stationary long memory (hyperbolic ACF decay; ARFIMA); 0.5 ≤ d < 1:
    non-stationary but mean-reverting; d ≈ 1: unit root. The bandwidth trades bias (short-run dynamics leak in
    as m grows) against variance; compare both estimators and a few bandwidths."""
    s = _series(y)
    x = s.to_numpy()
    n = len(x)
    z = stats.norm.ppf(1 - alpha / 2)
    m1 = int(np.floor(n ** power_gph))
    lam, I = _periodogram(x, m1)
    reg = -2 * np.log(2 * np.sin(lam / 2))
    Xg = np.column_stack([np.ones(m1), reg])
    bg, _ = _ols(np.log(I), Xg)
    d_gph, se_gph = float(bg[1]), float(np.pi / np.sqrt(24 * m1))
    m2 = int(np.floor(n ** power_lw))
    lam2, I2 = _periodogram(x, m2)

    def R(d):
        return np.log(np.mean(lam2 ** (2 * d) * I2)) - 2 * d * np.mean(np.log(lam2))
    res = optimize.minimize_scalar(R, bounds=(-0.49, 1.49), method="bounded")
    d_lw, se_lw = float(res.x), float(1 / (2 * np.sqrt(m2)))
    tab = pd.DataFrame({"d": [d_gph, d_lw], "se": [se_gph, se_lw], "bandwidth_m": [m1, m2]},
                       index=["GPH", "local Whittle"])
    tab["ci_low"], tab["ci_high"] = tab["d"] - z * tab["se"], tab["d"] + z * tab["se"]

    def read(d, se):
        if d <= -0.489:
            return ("d at the lower edge of the search range (−0.49): strongly anti-persistent, typically an "
                    "over-differenced series; the estimate itself is not reliable")
        if d >= 1.489:
            return "d at the upper edge of the search range (1.49): difference the series and estimate again"
        if d + z * se < 0:
            return "anti-persistent (d < 0): often a sign of over-differencing"
        if d - z * se > 0 and d + z * se < 0.5:
            return "stationary long memory (0 < d < 0.5)"
        if d + z * se < 0.5 and d - z * se <= 0:
            return "no evidence of long memory (d compatible with 0)"
        if d - z * se >= 0.5:
            return "non-stationary (d ≥ 0.5): difference or model as I(d)"
        return "d not pinned down between short memory, long memory and non-stationarity"
    out = {"table": tab, "d_gph": d_gph, "d_local_whittle": d_lw, "reading": read(d_lw, se_lw), "n": n,
           "method": "GPH log-periodogram and local Whittle"}
    if show:
        print(f"\nFractional d (n = {n})")
        print(tab.to_string(float_format=_fmt))
        print("→ local Whittle:", out["reading"])
    return out


def frac_diff(y, d: float) -> pd.Series:
    """(1 − L)^d y with weights w_0 = 1, w_j = w_{j−1}(j − 1 − d)/j, truncated at the start of the sample
    (expanding window). Negative d integrates."""
    s = _series(y)
    x = s.to_numpy()
    n = len(x)
    w = np.ones(n)
    for j in range(1, n):
        w[j] = w[j - 1] * (j - 1 - d) / j
    out = np.array([w[:t + 1][::-1] @ x[:t + 1] for t in range(n)])
    return pd.Series(out, index=s.index, name=f"fracdiff_{d:.3f}")


def arfima(y, order=(1, 0), d: float | None = None, show: bool = True) -> dict:
    """Two-step ARFIMA(p, d, q): d by local Whittle (unless given), then ARMA(p, q) with a constant on the
    fractionally differenced series (the first 10% of observations, where the truncated filter is poor, are
    dropped). The SEs of the ARMA part treat d as known."""
    from statsmodels.tsa.arima.model import ARIMA
    s = _series(y)
    lm = long_memory(s, show=False) if d is None else None
    dd = lm["d_local_whittle"] if d is None else float(d)
    fd = frac_diff(s - s.mean(), dd)
    cut = int(0.1 * len(fd))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ARIMA(fd.iloc[cut:].to_numpy(), order=(order[0], 0, order[1]), trend="c").fit()
    out = {"d": dd, "d_source": "local Whittle" if d is None else "given", "arma": fit, "params": fit.params,
           "bse": fit.bse, "long_memory": lm, "n_used": len(fd) - cut}
    if show:
        print(f"\nARFIMA({order[0]}, d, {order[1]}): d = {dd:.4f} ({out['d_source']}); ARMA on (1 − L)^d y, "
              f"n = {out['n_used']}")
        print(pd.DataFrame({"coef": fit.params, "se": fit.bse}, index=fit.param_names).to_string(float_format=_fmt))
    return out


# =============================================================================
# Forecast combination
# =============================================================================
def combine_forecasts(actual, forecasts: pd.DataFrame, method: str = "equal", train_end=None, show: bool = True) -> dict:
    """Combine competing forecasts of the same target and evaluate the combination out of sample.

    forecasts: one column per model, aligned with actual (target-dated). Weights are estimated on the rows up
    to train_end (default: first half) and the RMSE/MAE of every model and of the combination are reported on
    the rows after it. method: 'equal' (default; the "forecast combination puzzle": hard to beat because
    estimated weights are noisy), 'inverse_mse', 'min_variance' (Bates-Granger with the matrix of mean squared
    errors, so a biased forecast is penalised; can give negative weights), 'granger_ramanathan' (regression of y on the forecasts with weights summing to 1)."""
    a = pd.Series(actual).astype(float)
    F = pd.DataFrame(forecasts).astype(float)
    df = pd.concat([a.rename("_y"), F], axis=1).dropna()
    if len(df) < 10:
        raise ValueError("too few rows with the target and every forecast")
    if train_end is None:
        cut = len(df) // 2
    else:
        cut = int((df.index <= train_end).sum())
    tr, te = df.iloc[:cut], df.iloc[cut:]
    if len(tr) < 5 or len(te) < 5:
        raise ValueError("the training and evaluation windows need at least 5 rows each")
    E = tr[F.columns].to_numpy() - tr[["_y"]].to_numpy()
    k = F.shape[1]
    if method == "equal":
        w = np.full(k, 1 / k)
    elif method == "inverse_mse":
        inv = 1 / np.mean(E ** 2, axis=0)
        w = inv / inv.sum()
    elif method == "min_variance":
        # Bates-Granger with the matrix of mean squared errors, E'E/T (not the centred covariance, which would
        # reward a biased but steady forecast)
        S = E.T @ E / len(E)
        w = np.linalg.solve(S, np.ones(k)) if k > 1 else np.ones(1)
        w = w / w.sum()
    elif method == "granger_ramanathan":
        Xf = tr[F.columns].to_numpy()
        yv = tr["_y"].to_numpy()
        # constrained LS: y − f_k = Σ_{j<k} w_j (f_j − f_k), w_k = 1 − Σ w_j
        Z = Xf[:, :-1] - Xf[:, [-1]]
        wj, _ = _ols(yv - Xf[:, -1], Z)
        w = np.r_[wj, 1 - wj.sum()]
    else:
        raise ValueError("method must be 'equal', 'inverse_mse', 'min_variance' or 'granger_ramanathan'")
    weights = pd.Series(w, index=F.columns, name="weight")
    comb_te = te[F.columns].to_numpy() @ w
    rows = {}
    for c in F.columns:
        e = te[c] - te["_y"]
        rows[c] = {"rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e)))}
    e = comb_te - te["_y"].to_numpy()
    rows[f"combination ({method})"] = {"rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e)))}
    tab = pd.DataFrame(rows).T
    best = tab.drop(index=f"combination ({method})")["rmse"].idxmin()
    warn = []
    if method in ("min_variance", "granger_ramanathan") and (w < 0).any():
        warn.append("negative weights: estimated weights are noisy; compare with method='equal'")
    out = {"weights": weights, "evaluation": tab, "best_single": best, "n_train": len(tr), "n_test": len(te),
           "combined_forecast": pd.Series(df[F.columns].to_numpy() @ w, index=df.index, name="combination"),
           "warnings": warn}
    if show:
        print(f"\nForecast combination ({method}); weights from {len(tr)} rows, evaluated on the next {len(te)}")
        print(weights.to_string(float_format=_fmt))
        print(tab.to_string(float_format=_fmt))
        for wn in warn:
            print("Warning:", wn)
    return out
