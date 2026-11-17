"""
State-space tools: Kalman-filter regression with time-varying coefficients
(dynamic hedge ratio), maximum-likelihood estimation of its noise variances,
mean-reversion diagnostics and a pairs-trading layer built on top.

Model (Chan, 2013; Durbin & Koopman, 2012):
    y_t = a_t + b_t x_t + e_t,          e_t ~ N(0, R)          (observation)
    [a_t, b_t]' = [a_{t-1}, b_{t-1}]' + w_t,  w_t ~ N(0, Q),  Q = delta/(1-delta) * I

Key point: the trading signal must use the ONE-STEP-AHEAD prediction error
e_t = y_t - (a_{t|t-1} + b_{t|t-1} x_t) and its variance S_t. Using the residual
after updating with y_t (the filtered spread) mechanically shrinks it towards zero
and inflates backtests.

Functions
---------
kalman_regression(y, x, delta, R)          filter: predicted/filtered coefficients, innovations, z-scores, loglik
fit_kalman_regression(y, x)                MLE of delta and R (prediction-error decomposition)
half_life(spread)                          AR(1)/Ornstein-Uhlenbeck half-life of mean reversion
hurst_exponent(series)                     <0.5 mean-reverting, 0.5 random walk, >0.5 trending
pairs_position(z, entry, exit, stop)       long/short/flat state machine on z-scores
pairs_returns(y, x, hedge, position, cost) returns of the hedged spread portfolio (one-bar lag)
cointegration_pretest(y, x)                Engle-Granger both directions + Johansen trace
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize
import _console  # noqa: F401  (Windows-safe console output, see _console.py)


def kalman_regression(y, x, delta: float = 1e-4, R: float | None = None, init: int = 20,
                      P0: float = 1.0) -> pd.DataFrame:
    """Kalman filter for y_t = a_t + b_t x_t + e_t with random-walk coefficients.

    init : observations used for the OLS starting values (they get no z-score).
    R    : observation noise variance; default = OLS residual variance on the init window.
    Returns a DataFrame with predicted (a_pred, b_pred) and filtered (a_filt, b_filt)
    coefficients, innovation e, its std sqrt_S, z = e / sqrt_S, and the log-likelihood
    in df.attrs['loglik'].
    """
    ys = pd.Series(y).astype(float)
    xs = pd.Series(x, index=ys.index).astype(float)
    Y, X = ys.to_numpy(), xs.to_numpy()
    n = len(Y)
    Xi = np.column_stack([np.ones(init), X[:init]])
    theta, *_ = np.linalg.lstsq(Xi, Y[:init], rcond=None)
    if R is None:
        R = float(np.var(Y[:init] - Xi @ theta, ddof=2))
    P = np.eye(2) * P0 * max(R, 1e-12)
    Q = delta / (1 - delta) * np.eye(2)
    cols = {k: np.full(n, np.nan) for k in ["a_pred", "b_pred", "a_filt", "b_filt", "e", "sqrt_S", "z"]}
    ll = 0.0
    for t in range(init, n):
        th_p = theta                          # random walk: predicted state = last filtered
        P_p = P + Q
        F = np.array([1.0, X[t]])
        e = Y[t] - F @ th_p
        S = F @ P_p @ F + R
        K = P_p @ F / S
        theta = th_p + K * e
        P = (np.eye(2) - np.outer(K, F)) @ P_p
        P = 0.5 * (P + P.T)
        cols["a_pred"][t], cols["b_pred"][t] = th_p
        cols["a_filt"][t], cols["b_filt"][t] = theta
        cols["e"][t], cols["sqrt_S"][t] = e, np.sqrt(S)
        cols["z"][t] = e / np.sqrt(S)
        ll += -0.5 * (np.log(2 * np.pi * S) + e ** 2 / S)
    out = pd.DataFrame(cols, index=ys.index)
    out.attrs.update({"loglik": ll, "delta": delta, "R": R})
    return out


def fit_kalman_regression(y, x, init: int = 20, show: bool = True):
    """Choose delta and R by maximum likelihood instead of hand-tuning (which overfits
    if tuned on backtest P&L). Returns (filter_output, params)."""
    def nll(p):
        d = 1 / (1 + np.exp(-p[0]))           # delta in (0,1)
        r = np.exp(p[1])
        return -kalman_regression(y, x, delta=d, R=r, init=init).attrs["loglik"]
    ys = pd.Series(y).astype(float).to_numpy()
    r0 = np.log(np.var(np.diff(ys[: max(init * 3, 60)])) + 1e-12)
    best = None
    for d0 in (1e-6, 1e-5, 1e-4, 1e-3):
        res = optimize.minimize(nll, [np.log(d0 / (1 - d0)), r0], method="Nelder-Mead",
                                options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 400})
        if best is None or res.fun < best.fun:
            best = res
    d = 1 / (1 + np.exp(-best.x[0]))
    r = float(np.exp(best.x[1]))
    out = kalman_regression(y, x, delta=d, R=r, init=init)
    if show:
        bp = out["b_pred"].dropna()
        print(f"\nKalman regression (MLE): delta = {d:.2e}, R = {r:.4g}, loglik = {out.attrs['loglik']:.1f}")
        print(f"Hedge ratio b_t: start {bp.iloc[0]:.3f}, end {bp.iloc[-1]:.3f}, "
              f"range [{bp.min():.3f}, {bp.max():.3f}]")
        z = out["z"].dropna()
        print(f"Standardized innovations: mean {z.mean():.3f}, sd {z.std():.3f} "
              "(≈0 and ≈1 if the model is well specified)")
    return out, {"delta": d, "R": r, "loglik": out.attrs["loglik"]}


def half_life(spread: pd.Series) -> dict:
    """Δs_t = c + phi s_{t-1} + u_t ; half-life = -ln 2 / ln(1 + phi)."""
    import statsmodels.api as sm
    s = pd.Series(spread).dropna()
    ds = s.diff().dropna()
    lag = s.shift(1).loc[ds.index]
    res = sm.OLS(ds, sm.add_constant(lag)).fit()
    phi = res.params.iloc[1]
    hl = -np.log(2) / np.log(1 + phi) if -1 < phi < 0 else np.inf
    return {"phi": phi, "t_phi": res.tvalues.iloc[1], "half_life": hl,
            "note": "t_phi follows Dickey-Fuller, not normal, critical values (≈ -2.86 at 5%)"}


def hurst_exponent(series, max_lag: int = 100) -> float:
    s = np.asarray(pd.Series(series).dropna(), float)
    lags = np.arange(2, min(max_lag, len(s) // 4))
    tau = [np.std(s[l:] - s[:-l]) for l in lags]
    return float(np.polyfit(np.log(lags), np.log(tau), 1)[0])


def pairs_position(z: pd.Series, entry: float = 2.0, exit: float = 0.5,
                   stop: float | None = 4.0) -> pd.Series:
    """+1 = long spread (long y, short b*x) when z < -entry; -1 = short spread when
    z > entry; flat when |z| < exit or (optional) |z| > stop (relationship breaking)."""
    pos = np.zeros(len(z))
    cur = 0.0
    zz = z.to_numpy()
    for t, v in enumerate(zz):
        if np.isnan(v):
            pos[t] = cur = 0.0
            continue
        if stop is not None and abs(v) > stop:
            cur = 0.0
        elif cur == 0:
            cur = 1.0 if v < -entry else (-1.0 if v > entry else 0.0)
        elif cur == 1 and v > -exit:
            cur = 0.0
        elif cur == -1 and v < exit:
            cur = 0.0
        pos[t] = cur
    return pd.Series(pos, index=z.index, name="spread_position")


def pairs_returns(y_price: pd.Series, x_price: pd.Series, hedge: pd.Series, position: pd.Series,
                  cost_bps: float = 5.0) -> pd.DataFrame:
    """Hedged spread: 1 unit of y against hedge_t units of x, decided at t, held over t+1.
    Return = P&L / gross exposure at t. hedge should be b_pred or b_filt at t (known at t)."""
    pos = position.shift(1).fillna(0.0)
    h = hedge.shift(1)
    py, px = y_price.shift(1), x_price.shift(1)
    pnl = pos * (y_price.diff() - h * x_price.diff())
    gross = py.abs() + (h * px).abs()
    ret = (pnl / gross).fillna(0.0)
    turnover = (pos.diff().abs().fillna(pos.abs()) * 2)        # two legs
    ret = ret - turnover * cost_bps / 1e4
    return pd.DataFrame({"position_held": pos, "hedge_ratio": h, "strategy_return": ret,
                         "turnover": turnover})


def cointegration_pretest(y: pd.Series, x: pd.Series, show: bool = True) -> dict:
    """Engle-Granger is not symmetric: run it both ways and confirm with Johansen.

    v2.6.11: each Engle-Granger direction runs with the residual-ADF lags chosen by AIC (the EG_p_* keys, as
    before) and by BIC (EG_p_*_bic). A direction counts as evidence only when both criteria reject at 5%; when
    they disagree (EG_lag_sensitive) the pretest says so. Johansen uses det_order=0, which declares full rank
    too often for driftless random walks (time-series.md §7): johansen_full_rank flags that case, and then
    cointegrated_5% is False and the pretest gives no verdict (two I(0) series also make Engle-Granger reject).
    """
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    from diagnostics import FULL_RANK_DET0, _engle_granger
    d = pd.concat([y, x], axis=1).dropna()
    eg = {(a, b, crit): _engle_granger(d.iloc[:, a], d.iloc[:, [b]], "c", crit)
          for a, b in ((0, 1), (1, 0)) for crit in ("aic", "bic")}
    p_yx, p_xy = eg[(0, 1, "aic")]["p"], eg[(1, 0, "aic")]["p"]
    p_yx_b, p_xy_b = eg[(0, 1, "bic")]["p"], eg[(1, 0, "bic")]["p"]
    robust = (max(p_yx, p_yx_b) < 0.05) or (max(p_xy, p_xy_b) < 0.05)
    sensitive = any((pa < 0.05) != (pb < 0.05) for pa, pb in ((p_yx, p_yx_b), (p_xy, p_xy_b)))
    j = coint_johansen(d, det_order=0, k_ar_diff=1)
    trace, cv95 = j.lr1[0], j.cvt[0, 1]
    full_rank = bool(j.lr1[0] > j.cvt[0, 1] and j.lr1[-1] > j.cvt[-1, 1])   # rank k: every trace H0 rejected (r2)
    out = {"EG_p_y_on_x": p_yx, "EG_p_x_on_y": p_xy, "EG_p_y_on_x_bic": p_yx_b, "EG_p_x_on_y_bic": p_xy_b,
           "EG_lag_sensitive": sensitive, "Johansen_trace_r0": trace, "Johansen_cv95": cv95,
           "johansen_full_rank": full_rank, "cointegrated_5%": bool(robust and trace > cv95 and not full_rank)}
    if show:
        print(f"\nCointegration pretest: EG p (y~x) = {p_yx:.3f} AIC / {p_yx_b:.3f} BIC, "
              f"EG p (x~y) = {p_xy:.3f} AIC / {p_xy_b:.3f} BIC, "
              f"Johansen trace r=0: {trace:.2f} vs 5% cv {cv95:.2f} → "
              + ("cointegrated" if out["cointegrated_5%"] else
                 "no verdict: Johansen gives full rank, so cointegration is not established (see below)" if full_rank
                 else "NOT clearly cointegrated: a mean-reversion strategy rests on an unproven relationship"))
        if sensitive:
            print("Engle-Granger changes conclusion with the lag choice of the residual ADF (AIC vs BIC): "
                  "only a direction where both reject counts.")
        if full_rank:
            print(FULL_RANK_DET0)
    return out
