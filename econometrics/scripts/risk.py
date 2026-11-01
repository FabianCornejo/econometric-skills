"""
Market-risk econometrics: volatility estimators, Value-at-Risk / Expected Shortfall
and their statistical backtests.

Sign convention: VaR and ES are reported as POSITIVE loss numbers in return units
(0.03 = a 3% loss). A violation (exception) occurs when the realized return is below
-VaR. Every rolling estimate at date t uses returns up to t-1 only.

Functions
---------
ewma_volatility(r, lam)                      RiskMetrics EWMA sigma_t (uses r up to t-1)
var_historical(r, alpha, window)             rolling empirical quantile
var_parametric(r, alpha, window, dist)       rolling normal or Student-t VaR
var_ewma(r, alpha, lam, dist)                EWMA sigma x quantile (filtered-historical option)
var_garch(r, alpha, window, refit_every)     rolling GARCH/GJR one-step VaR and ES
expected_shortfall_historical(...)           rolling empirical ES
kupiec_pof(r, var, alpha)                    unconditional coverage (LR_uc)
christoffersen(r, var, alpha)                independence (LR_ind) and conditional coverage (LR_cc)
basel_traffic_light(n_viol, n=250)           green / yellow / red zone for 99% VaR
dq_test(r, var, alpha, lags=4)               Engle-Manganelli (2004) dynamic quantile test (hits predictable?)
backtest_var(r, var_dict, alpha)             one table for several VaR models

Extreme value theory (v2.2)
mean_excess(r, quantiles)                    mean-excess plot data for choosing the POT threshold
gpd_fit(r, q, alpha)                         peaks over threshold: GPD (xi, beta) with SEs, tail VaR/ES
                                             with delta-method CIs, and a threshold-stability table
var_evt(r, alpha, window, q, filter)         rolling EVT VaR/ES; filter='ewma' is McNeil-Frey (2000)
                                             conditional EVT (GPD on EWMA-standardized returns)

Expected Shortfall backtests and comparison (v2.2)
es_backtest(r, var, es, alpha)               McNeil-Frey exceedance residuals + Acerbi-Szekely (2014) Z2
                                             with a simulated p-value and the Basel-style zones
fz_loss(r, var, es, alpha)                   Fissler-Ziegel FZ0 loss (Patton, Ziegel & Chen 2019), per day
compare_es_models(r, models, alpha)          mean FZ0 loss per model + Diebold-Mariano vs the best

Requires: numpy, pandas, scipy; arch for var_garch.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------
def ewma_volatility(r: pd.Series, lam: float = 0.94, init_window: int = 30,
                    demean: bool = False) -> pd.Series:
    """sigma_t^2 = lam * sigma_{t-1}^2 + (1-lam) * r_{t-1}^2  → forecast for day t.

    Initialized with the variance of the first `init_window` returns; the first
    init_window values are NaN so the seed is not used out of sample.
    lam = 0.94 (daily) / 0.97 (monthly) are the RiskMetrics (1996) values.
    """
    x = r.astype(float)
    if demean:
        x = x - x.expanding().mean().shift(1).fillna(0)
    v = np.full(len(x), np.nan)
    v[init_window] = np.var(x.iloc[:init_window], ddof=1)
    xs = x.to_numpy()
    for t in range(init_window + 1, len(x)):
        v[t] = lam * v[t - 1] + (1 - lam) * xs[t - 1] ** 2
    return pd.Series(np.sqrt(v), index=r.index, name=f"ewma_vol_{lam}")


def realized_volatility(r: pd.Series, window: int = 21) -> pd.Series:
    """Ex-post rolling standard deviation (for evaluating volatility forecasts)."""
    return r.rolling(window).std()


# ---------------------------------------------------------------------------
# VaR / ES
# ---------------------------------------------------------------------------
def _q(dist: str, alpha: float, nu: float | None = None) -> float:
    """alpha-quantile of a unit-variance distribution (negative number for small alpha)."""
    if dist == "normal":
        return stats.norm.ppf(alpha)
    if dist == "t":
        return stats.t.ppf(alpha, nu) * np.sqrt((nu - 2) / nu)
    raise ValueError(dist)


def _es_factor(dist: str, alpha: float, nu: float | None = None) -> float:
    """E[Z | Z <= q_alpha] for a unit-variance distribution (negative)."""
    if dist == "normal":
        return -stats.norm.pdf(stats.norm.ppf(alpha)) / alpha
    q = stats.t.ppf(alpha, nu)
    es = -(stats.t.pdf(q, nu) / alpha) * (nu + q ** 2) / (nu - 1)
    return es * np.sqrt((nu - 2) / nu)


def var_historical(r: pd.Series, alpha: float = 0.01, window: int = 250) -> pd.Series:
    return (-r.rolling(window).quantile(alpha).shift(1)).rename("VaR_hist")


def expected_shortfall_historical(r: pd.Series, alpha: float = 0.01, window: int = 250) -> pd.Series:
    def es(x):
        q = np.quantile(x, alpha)
        return -x[x <= q].mean()
    return r.rolling(window).apply(es, raw=True).shift(1).rename("ES_hist")


def var_parametric(r: pd.Series, alpha: float = 0.01, window: int = 250, dist: str = "normal",
                   with_mean: bool = False) -> pd.DataFrame:
    """Rolling-window variance-covariance VaR; dist='t' fits nu on each window (slower)."""
    mu = r.rolling(window).mean().shift(1) if with_mean else 0.0
    sd = r.rolling(window).std().shift(1)
    if dist == "normal":
        q, esf = _q("normal", alpha), _es_factor("normal", alpha)
        return pd.DataFrame({"VaR": -(mu + sd * q), "ES": -(mu + sd * esf)})
    nus = r.rolling(window).apply(lambda x: max(stats.t.fit(x)[0], 2.1), raw=True).shift(1)
    q = nus.apply(lambda v: _q("t", alpha, v) if pd.notna(v) else np.nan)
    esf = nus.apply(lambda v: _es_factor("t", alpha, v) if pd.notna(v) else np.nan)
    return pd.DataFrame({"VaR": -(mu + sd * q), "ES": -(mu + sd * esf), "nu": nus})


def var_ewma(r: pd.Series, alpha: float = 0.01, lam: float = 0.94, method: str = "normal",
             fhs_window: int = 500) -> pd.DataFrame:
    """EWMA VaR.

    method='normal' : RiskMetrics, VaR = -sigma_t * z_alpha.
    method='fhs'    : filtered historical simulation — empirical quantile of the past
                      standardized returns r/sigma (captures fat tails without assuming them).
    """
    sig = ewma_volatility(r, lam)
    if method == "normal":
        return pd.DataFrame({"VaR": -sig * _q("normal", alpha), "ES": -sig * _es_factor("normal", alpha),
                             "sigma": sig})
    z = (r / sig)
    qz = z.rolling(fhs_window, min_periods=100).quantile(alpha).shift(1)
    esz = z.rolling(fhs_window, min_periods=100).apply(
        lambda x: x[x <= np.quantile(x, alpha)].mean(), raw=True).shift(1)
    return pd.DataFrame({"VaR": -sig * qz, "ES": -sig * esz, "sigma": sig})


def var_garch(r: pd.Series, alpha: float = 0.01, window: int = 1000, refit_every: int = 20,
              vol: str = "GARCH", o: int = 0, dist: str = "t", mean: str = "Constant",
              scale: float = 100.0, expanding: bool = False) -> pd.DataFrame:
    """Rolling one-step-ahead GARCH VaR and ES.

    Parameters are re-estimated every `refit_every` days on the last `window`
    returns (or all past returns if expanding=True); between refits the filter
    is run with fixed parameters, which is standard practice and ~20x faster than
    refitting daily. Returns are multiplied by `scale` for the optimizer and the
    output is converted back to return units.
    o=1 gives GJR-GARCH (leverage effect).
    """
    from arch import arch_model
    x = r.astype(float) * scale
    n = len(x)
    out = pd.DataFrame(index=r.index, columns=["VaR", "ES", "sigma", "nu"], dtype=float)
    params = None
    for t in range(window, n):
        lo = 0 if expanding else t - window
        if params is None or (t - window) % refit_every == 0:
            am = arch_model(x.iloc[lo:t], mean=mean, vol=vol, p=1, o=o, q=1, dist=dist)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                params = am.fit(disp="off", show_warning=False).params
        # filter returns up to t-1 with fixed parameters; the 1-step forecast is for day t
        am = arch_model(x.iloc[lo:t], mean=mean, vol=vol, p=1, o=o, q=1, dist=dist)
        fr = am.fix(params).forecast(horizon=1, reindex=False)
        mu = float(fr.mean.iloc[0, 0])
        s = float(np.sqrt(fr.variance.iloc[0, 0]))
        nu = float(params["nu"]) if "nu" in params else None
        d = "t" if dist == "t" else "normal"
        out.iloc[t] = [-(mu + s * _q(d, alpha, nu)) / scale, -(mu + s * _es_factor(d, alpha, nu)) / scale,
                       s / scale, nu if nu else np.nan]
    return out


# ---------------------------------------------------------------------------
# Backtests
# ---------------------------------------------------------------------------
def _hits(r: pd.Series, var: pd.Series) -> pd.Series:
    df = pd.concat([r, var], axis=1).dropna()
    return (df.iloc[:, 0] < -df.iloc[:, 1]).astype(int)


def kupiec_pof(r: pd.Series, var: pd.Series, alpha: float = 0.01) -> dict:
    """Kupiec (1995) proportion-of-failures test. H0: violation rate = alpha."""
    I = _hits(r, var)
    n, x = len(I), int(I.sum())
    pi = x / n
    ll0 = (n - x) * np.log(1 - alpha) + x * np.log(alpha)
    ll1 = (n - x) * np.log(1 - pi) + x * np.log(pi) if 0 < x < n else 0.0
    lr = -2 * (ll0 - ll1)
    return {"n": n, "violations": x, "expected": alpha * n, "rate": pi, "LR_uc": lr,
            "p_uc": 1 - stats.chi2.cdf(lr, 1)}


def christoffersen(r: pd.Series, var: pd.Series, alpha: float = 0.01) -> dict:
    """Christoffersen (1998): independence of violations and conditional coverage."""
    I = _hits(r, var).to_numpy()
    k = kupiec_pof(r, var, alpha)
    a, b = I[:-1], I[1:]
    n00 = np.sum((a == 0) & (b == 0)); n01 = np.sum((a == 0) & (b == 1))
    n10 = np.sum((a == 1) & (b == 0)); n11 = np.sum((a == 1) & (b == 1))
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def ll(p, z, o):
        return (z * np.log(1 - p) if z and p < 1 else 0) + (o * np.log(p) if o and p > 0 else 0)
    lr_ind = -2 * (ll(pi, n00 + n10, n01 + n11) - ll(pi01, n00, n01) - ll(pi11, n10, n11))
    lr_cc = k["LR_uc"] + lr_ind
    return {**k, "LR_ind": lr_ind, "p_ind": 1 - stats.chi2.cdf(lr_ind, 1),
            "LR_cc": lr_cc, "p_cc": 1 - stats.chi2.cdf(lr_cc, 2),
            "P(viol | viol yesterday)": pi11}


def dq_test(r: pd.Series, var: pd.Series, alpha: float = 0.01, lags: int = 4) -> dict:
    """Engle & Manganelli (2004) dynamic quantile test, out-of-sample version.
    Hit_t = 1{r_t < -VaR_t} - alpha is regressed on a constant, Hit_{t-1..t-lags} and VaR_t;
    DQ = Hit'X (X'X)^-1 X'Hit / (alpha (1 - alpha)) ~ chi2(lags + 2) under a correct model.
    It detects violations that are predictable from past violations or from the VaR level
    itself (dependence of higher order than Christoffersen's first-order Markov test)."""
    df = pd.concat([pd.Series(r).rename("r"), pd.Series(var).rename("v")], axis=1).dropna()
    hit = (df["r"] < -df["v"]).astype(float).to_numpy() - alpha
    v = df["v"].to_numpy()
    T = len(hit)
    rows = [np.ones(T - lags)] + [hit[lags - k:T - k] for k in range(1, lags + 1)] + [v[lags:]]
    X = np.column_stack(rows)
    h = hit[lags:]
    XtX = X.T @ X
    try:
        b = np.linalg.solve(XtX, X.T @ h)
    except np.linalg.LinAlgError:
        b = np.linalg.lstsq(X, h, rcond=None)[0]
    dq = float(h @ X @ b / (alpha * (1 - alpha)))
    k = X.shape[1]
    return {"DQ": dq, "df": k, "p_dq": float(stats.chi2.sf(dq, k)), "n": T - lags, "lags": lags}


def basel_traffic_light(n_viol: int, n: int = 250, alpha: float = 0.01) -> str:
    """Basel zones from the binomial CDF: green < 95%, yellow < 99.99%, red otherwise."""
    c = stats.binom.cdf(n_viol, n, alpha)
    return "green" if c < 0.95 else ("yellow" if c < 0.9999 else "red")


def backtest_var(r: pd.Series, var_dict: dict[str, pd.Series], alpha: float = 0.01,
                 show: bool = True) -> pd.DataFrame:
    """Evaluate several VaR series on their common sample (fair comparison)."""
    common = pd.concat([r.rename("r"), *[v.rename(k) for k, v in var_dict.items()]], axis=1).dropna()
    rows = {}
    for k in var_dict:
        c = christoffersen(common["r"], common[k], alpha)
        c["p_dq"] = dq_test(common["r"], common[k], alpha)["p_dq"]
        last = _hits(common["r"].iloc[-250:], common[k].iloc[-250:]).sum()
        c["avg_VaR"] = common[k].mean()
        c["traffic_light_last_250"] = basel_traffic_light(int(last), min(250, len(common)), alpha) \
            if abs(alpha - 0.01) < 1e-9 else "n/a"
        rows[k] = c
    tab = pd.DataFrame(rows).T[["n", "violations", "expected", "rate", "p_uc", "p_ind", "p_cc", "p_dq",
                                "P(viol | viol yesterday)", "avg_VaR", "traffic_light_last_250"]]
    if show:
        print(f"\nVaR backtest at {100 * (1 - alpha):.1f}% ({len(common)} common days)")
        print(tab.to_string(float_format=lambda v: f"{v:.4f}"))
        for k in rows:
            msgs = []
            if rows[k]["p_uc"] < 0.05:
                msgs.append("too many" if rows[k]["rate"] > alpha else "too few (over-conservative)")
            if rows[k]["p_ind"] < 0.05:
                msgs.append("violations cluster in time")
            if rows[k]["p_dq"] < 0.05:
                msgs.append("violations predictable (DQ)")
            print(f"  {k}: " + ("; ".join(msgs) if msgs else "passes coverage, independence and DQ at 5%"))
        print("  Among models that pass, prefer the lower average VaR (less capital for the same coverage).")
    return tab



# ---------------------------------------------------------------------------
# Extreme value theory: peaks over threshold (v2.2)
# ---------------------------------------------------------------------------
def _losses(r) -> np.ndarray:
    x = -np.asarray(pd.Series(r).dropna(), float)
    return x


def mean_excess(r: pd.Series, quantiles=None) -> pd.DataFrame:
    """Mean excess e(u) = E[L - u | L > u] of the losses L = -r over a grid of thresholds.

    For a GPD tail, e(u) is linear in u with slope xi / (1 - xi): choose the lowest threshold
    above which the plot is roughly linear (upward slope = heavy tail, xi > 0). Too low a
    threshold biases xi (the body of the distribution is not GPD); too high leaves few
    exceedances and a noisy estimate."""
    L = _losses(r)
    qs = np.asarray(quantiles if quantiles is not None else np.arange(0.80, 0.991, 0.01))
    rows = []
    for q in qs:
        u = np.quantile(L, q)
        ex = L[L > u] - u
        m = ex.mean() if len(ex) else np.nan
        se = ex.std(ddof=1) / np.sqrt(len(ex)) if len(ex) > 1 else np.nan
        rows.append({"quantile": q, "threshold": u, "n_exceed": len(ex), "mean_excess": m,
                     "ci_low": m - 1.96 * se, "ci_high": m + 1.96 * se})
    return pd.DataFrame(rows)


def _gpd_mle(y: np.ndarray) -> tuple[float, float]:
    """GPD MLE (location 0) for exceedances y > 0; start from probability-weighted moments."""
    m, v = y.mean(), y.var()
    xi0 = 0.5 * (1 - m ** 2 / v) if v > 0 else 0.1
    xi0 = float(np.clip(xi0, -0.4, 0.9))
    b0 = max(m * (1 - xi0), 1e-8)
    try:
        xi, _, b = stats.genpareto.fit(y, xi0, floc=0, scale=b0)
    except Exception:  # pragma: no cover - scipy fallback
        xi, b = xi0, b0
    return float(xi), float(b)


def _gpd_tail(u, xi, b, zeta, alpha):
    """VaR and ES at tail probability alpha from a GPD fitted above u (zeta = P(L > u))."""
    if abs(xi) < 1e-8:
        var = u + b * np.log(zeta / alpha)
    else:
        var = u + b / xi * ((alpha / zeta) ** (-xi) - 1)
    es = (var + b - xi * u) / (1 - xi) if xi < 1 else np.inf
    return var, es


def gpd_fit(r: pd.Series, q: float = 0.95, alpha: float = 0.01, threshold: float | None = None,
            stability_q=(0.90, 0.925, 0.95, 0.965, 0.975), show: bool = True) -> dict:
    """Peaks-over-threshold estimation of the loss tail (Balkema-de Haan-Pickands theorem).

    Losses L = -r above u (the q-quantile, or `threshold`) follow approximately a generalized
    Pareto distribution with shape xi (tail index; xi = 1/nu for a Student-t) and scale beta.
    Tail estimators (McNeil, Frey & Embrechts 2015, ch. 5), with zeta = N_u / n:
        VaR_alpha = u + beta/xi [ (alpha/zeta)^(-xi) - 1 ],   ES_alpha = (VaR + beta - xi u) / (1 - xi).
    SEs: Smith (1987) asymptotic covariance n_u Cov(xi, beta) = (1+xi)[[1+xi, -beta], [-beta, 2 beta^2]]
    (valid for xi > -1/2), propagated to VaR/ES by the delta method; zeta is treated as known.
    The delta-method CI for a far-tail VaR is symmetric while the true uncertainty is skewed
    upward: for alpha well below 1/n_u prefer a profile-likelihood or bootstrap interval.

    Returns the output contract for xi plus 'VaR', 'ES' (positive loss numbers), their CIs,
    'beta', 'threshold', 'n_exceed' and a threshold-stability table: xi and VaR should be
    stable across nearby thresholds; if they drift, the threshold is too low."""
    L = _losses(r)
    n = len(L)
    u = float(np.quantile(L, q)) if threshold is None else float(threshold)
    y = L[L > u] - u
    nu_ = len(y)
    warn = []
    if nu_ < 50:
        warn.append(f"only {nu_} exceedances: GPD estimates are imprecise (aim for 50-100 or more)")
    xi, b = _gpd_mle(y)
    zeta = nu_ / n
    if alpha >= zeta:
        warn.append(f"alpha={alpha} is not in the tail beyond the threshold (zeta={zeta:.3f}); lower alpha or raise q")
    V = (1 + xi) * np.array([[1 + xi, -b], [-b, 2 * b * b]]) / nu_
    var, es = _gpd_tail(u, xi, b, zeta, alpha)
    se_var = se_es = np.nan
    if xi > -0.5 and xi < 1:
        h = 1e-6
        def g(t):
            return np.array(_gpd_tail(u, t[0], t[1], zeta, alpha))
        J = np.column_stack([(g([xi + h, b]) - g([xi - h, b])) / (2 * h),
                             (g([xi, b + h]) - g([xi, b - h])) / (2 * h)])
        C = J @ V @ J.T
        se_var, se_es = float(np.sqrt(C[0, 0])), float(np.sqrt(C[1, 1]))
    else:
        warn.append("xi outside (-0.5, 1): asymptotic SEs not valid (ES is infinite if xi >= 1)")
    se_xi = float(np.sqrt(V[0, 0]))
    z = stats.norm.ppf(0.975)
    stab = []
    for qq in stability_q:
        uu = float(np.quantile(L, qq))
        yy = L[L > uu] - uu
        if len(yy) < 15:
            continue
        x2, b2 = _gpd_mle(yy)
        v2, e2 = _gpd_tail(uu, x2, b2, len(yy) / n, alpha) if alpha < len(yy) / n else (np.nan, np.nan)
        stab.append({"quantile": qq, "threshold": uu, "n_exceed": len(yy), "xi": x2,
                     "se_xi": np.sqrt((1 + x2) ** 2 / len(yy)) if x2 > -0.5 else np.nan,
                     "VaR": v2, "ES": e2})
    stab = pd.DataFrame(stab)
    emp_var = float(np.quantile(L, 1 - alpha))
    out = {"estimand": "GPD shape xi (loss tail index)", "estimate": xi, "se": se_xi,
           "ci_low": xi - z * se_xi, "ci_high": xi + z * se_xi,
           "p": float(2 * stats.norm.sf(abs(xi / se_xi))), "n": n,
           "method": f"POT-GPD MLE, threshold = {q:.3f} quantile" if threshold is None else "POT-GPD MLE",
           "beta": b, "threshold": u, "n_exceed": nu_, "zeta": zeta, "alpha": alpha,
           "VaR": var, "VaR_se": se_var, "VaR_ci": (var - z * se_var, var + z * se_var),
           "ES": es, "ES_se": se_es, "ES_ci": (es - z * se_es, es + z * se_es),
           "VaR_empirical": emp_var, "stability": stab, "warnings": warn}
    if show:
        print(f"\nPOT-GPD on {n} losses: u = {u:.4f} ({nu_} exceedances, zeta = {zeta:.3f})")
        print(f"  xi = {xi:.3f} (SE {se_xi:.3f})  beta = {b:.4f}   "
              + ("heavy tail (xi > 0)" if xi - z * se_xi > 0 else "tail index not distinguishable from 0 (exponential tail)"))
        print(f"  VaR {100 * (1 - alpha):.2f}% = {var:.4f} [{var - z * se_var:.4f}, {var + z * se_var:.4f}]   "
              f"empirical = {emp_var:.4f}")
        print(f"  ES  {100 * (1 - alpha):.2f}% = {es:.4f} [{es - z * se_es:.4f}, {es + z * se_es:.4f}]")
        if len(stab):
            print("  Threshold stability:")
            print(stab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        for w in warn:
            print("  WARNING:", w)
    return out


def var_evt(r: pd.Series, alpha: float = 0.01, window: int = 1000, q: float = 0.90,
            filter: str = "ewma", lam: float = 0.94, refit_every: int = 20) -> pd.DataFrame:
    """Rolling EVT VaR and ES, using returns up to t-1 only.

    filter='none' : unconditional POT on the raw losses of the window (slow to react to
                    volatility clusters: violations tend to cluster).
    filter='ewma' : McNeil & Frey (2000) conditional EVT. Standardize by the EWMA volatility,
                    fit the GPD to the standardized losses, and scale back:
                    VaR_t = sigma_t VaR_z, ES_t = sigma_t ES_z. Captures both volatility
                    clustering and fat tails; usually the best-backtesting simple model.
    The GPD is refitted every `refit_every` days (the tail shape changes slowly)."""
    r = pd.Series(r).dropna()
    if filter == "ewma":
        sig = ewma_volatility(r, lam)
        z = r / sig
    elif filter == "none":
        sig = pd.Series(1.0, index=r.index)
        z = r
    else:
        raise ValueError("filter must be 'ewma' or 'none'")
    zv = z.to_numpy()
    out = pd.DataFrame(index=r.index, columns=["VaR", "ES", "xi"], dtype=float)
    cur = None
    for t in range(window, len(r)):
        if cur is None or (t - window) % refit_every == 0:
            w = zv[t - window:t]
            w = w[np.isfinite(w)]
            L = -w
            u = np.quantile(L, q)
            y = L[L > u] - u
            xi, b = _gpd_mle(y)
            xi = min(xi, 0.95)
            vz, ez = _gpd_tail(u, xi, b, len(y) / len(L), alpha)
            cur = (vz, ez, xi)
        out.iloc[t] = [cur[0] * sig.iloc[t], cur[1] * sig.iloc[t], cur[2]]
    return out


# ---------------------------------------------------------------------------
# Expected Shortfall backtests (v2.2)
# ---------------------------------------------------------------------------
def _t_nu_from_ratio(ratio: float, alpha: float) -> float:
    """Degrees of freedom of a Student-t whose ES/VaR ratio at alpha equals `ratio`
    (inf = normal). Used to simulate the null distribution of Z2."""
    rn = _es_factor("normal", alpha) / _q("normal", alpha)
    if not np.isfinite(ratio) or ratio <= rn * 1.001:
        return np.inf
    f = lambda nu: _es_factor("t", alpha, nu) / _q("t", alpha, nu) - ratio
    lo, hi = 2.05, 500.0
    if f(lo) < 0:
        return lo
    if f(hi) > 0:
        return np.inf
    from scipy.optimize import brentq
    return float(brentq(f, lo, hi))


def es_backtest(r: pd.Series, var: pd.Series, es: pd.Series, alpha: float = 0.025,
                sigma: pd.Series | None = None, n_sim: int = 5000, n_boot: int = 5000,
                seed: int = 0, show: bool = True) -> dict:
    """Backtests of Expected Shortfall forecasts (positive loss numbers, same dates as r).

    1. McNeil-Frey (2000) exceedance residuals: on violation days, e_t = (L_t - ES_t) / sigma_t
       (sigma_t optional; without it, unscaled). H0: E[e] = 0; one-sided (losses beyond VaR
       larger than the ES forecast). p-value by a bootstrap-t (studentized): exceedance
       residuals are right-skewed, which makes the plain t-test conservative (about 2% size
       at a nominal 5% in the skill's simulations); the bootstrap-t corrects that.
    2. Acerbi-Szekely (2014) Z2 = sum_t r_t I_t / (T alpha ES_t) + 1, with I_t = 1{r_t < -VaR_t}.
       E[Z2] = 0 under a correct ES; Z2 < 0 means ES is underestimated. It tests VaR frequency
       and tail magnitude jointly. p-value by simulation under H0 assuming each day's
       forecast is a scaled Student-t whose nu matches the median ES/VaR ratio (normal when
       the ratio is the normal one). At alpha = 2.5% with T = 250, Acerbi-Szekely report
       nearly distribution-free cut-offs of -0.70 (95%) and -1.8 (99.99%); those zones are
       also reported then (only for 200-300 days: with longer samples they are too lax).
    Report it together with backtest_var (coverage and independence of the VaR)."""
    df = pd.concat([pd.Series(r).rename("r"), pd.Series(var).rename("v"), pd.Series(es).rename("e")]
                   + ([pd.Series(sigma).rename("s")] if sigma is not None else []), axis=1).dropna()
    T = len(df)
    I = (df["r"] < -df["v"]).to_numpy()
    n_v = int(I.sum())
    warn = []
    # McNeil-Frey
    scale = df["s"].to_numpy() if sigma is not None else np.ones(T)
    resid = ((-df["r"].to_numpy() - df["e"].to_numpy()) / scale)[I]
    mf = {"n_violations": n_v, "mean_resid": np.nan, "t": np.nan, "p_t": np.nan, "p_boot": np.nan}
    if n_v >= 3:
        m = resid.mean()
        sd = resid.std(ddof=1)
        tt = m / (sd / np.sqrt(n_v)) if sd > 0 else np.nan
        rng = np.random.default_rng(seed)
        B = resid[rng.integers(0, n_v, (n_boot, n_v))]
        bsd = B.std(axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            tb = (B.mean(axis=1) - m) / (bsd / np.sqrt(n_v))   # bootstrap-t: corrects the skewness
        tb = tb[np.isfinite(tb)]
        mf.update(mean_resid=float(m), t=float(tt), p_t=float(stats.t.sf(tt, n_v - 1)),
                  p_boot=float((1 + np.sum(tb >= tt)) / (len(tb) + 1)))
    else:
        warn.append(f"{n_v} violations: too few for the McNeil-Frey test")
    # Acerbi-Szekely Z2
    rr, vv, ee = df["r"].to_numpy(), df["v"].to_numpy(), df["e"].to_numpy()
    z2 = float(np.sum(rr * I / (T * alpha * ee)) + 1)
    ratio = float(np.median(ee / vv))
    nu = _t_nu_from_ratio(ratio, alpha)
    rng = np.random.default_rng(seed + 1)
    if np.isinf(nu):
        qz = _q("normal", alpha)
        Z = rng.standard_normal((n_sim, T))
    else:
        qz = _q("t", alpha, nu)
        Z = rng.standard_t(nu, (n_sim, T)) * np.sqrt((nu - 2) / nu)
    s_t = vv / (-qz)                                   # day-t scale implied by the VaR
    X = Z * s_t
    Isim = X < -vv
    z2_sim = np.sum(X * Isim / (T * alpha * ee), axis=1) + 1
    p_z2 = float((1 + np.sum(z2_sim <= z2)) / (n_sim + 1))
    zone = "n/a"
    if abs(alpha - 0.025) < 1e-9 and 200 <= T <= 300:      # cut-offs calibrated for one year
        zone = "green" if z2 > -0.70 else ("yellow" if z2 > -1.8 else "red")
    out = {"n": T, "alpha": alpha, "violations": n_v, "expected": alpha * T,
           "mcneil_frey": mf, "Z2": z2, "p_Z2": p_z2, "null_nu": nu, "Z2_zone_AS": zone,
           "warnings": warn}
    if show:
        print(f"\nES backtest at {100 * (1 - alpha):.1f}% ({T} days, {n_v} violations, {alpha * T:.1f} expected)")
        if n_v >= 3:
            print(f"  McNeil-Frey: mean exceedance residual = {mf['mean_resid']:.4f}, "
                  f"p (t) = {mf['p_t']:.4f}, p (bootstrap) = {mf['p_boot']:.4f}")
        print(f"  Acerbi-Szekely Z2 = {z2:.3f}, simulated p = {p_z2:.4f} "
              f"(null: {'normal' if np.isinf(nu) else f'Student-t, nu = {nu:.1f}'})"
              + (f", zone: {zone}" if zone != "n/a" else ""))
        verdict = []
        if p_z2 < 0.05:
            verdict.append("ES underestimated (Z2 rejects)")
        if n_v >= 3 and mf["p_boot"] < 0.05:
            verdict.append("losses beyond VaR exceed the ES forecast")
        print("  " + ("; ".join(verdict) if verdict else "no evidence that ES is underestimated at 5%"))
        for w in warn:
            print("  WARNING:", w)
    return out


def fz_loss(r: pd.Series, var: pd.Series, es: pd.Series, alpha: float = 0.025) -> pd.Series:
    """FZ0 loss of Patton, Ziegel & Chen (2019), the strictly consistent scoring function
    (Fissler & Ziegel 2016) for the pair (VaR, ES); VaR and ES as positive loss numbers.
    With v = -VaR and e = -ES (both negative):  L = -1/(alpha e) 1{r <= v}(v - r) + v/e + log(-e) - 1.
    Lower is better. ES alone is not elicitable, so ES models are ranked with this joint loss."""
    df = pd.concat([pd.Series(r).rename("r"), pd.Series(var).rename("v"), pd.Series(es).rename("e")],
                   axis=1).dropna()
    v, e, y = -df["v"], -df["e"], df["r"]
    hit = (y <= v).astype(float)
    return (-hit * (v - y) / (alpha * e) + v / e + np.log(-e) - 1).rename("FZ0")


def compare_es_models(r: pd.Series, models: dict, alpha: float = 0.025, show: bool = True) -> pd.DataFrame:
    """Rank (VaR, ES) forecasts by mean FZ0 loss on their common sample; models maps a name to
    a DataFrame with columns 'VaR' and 'ES' (or a (var, es) tuple). Each model is compared with
    the best by a Diebold-Mariano test on the loss differences (Newey-West variance)."""
    parts = {}
    for k, m in models.items():
        v, e = (m["VaR"], m["ES"]) if isinstance(m, pd.DataFrame) else m
        parts[k] = fz_loss(r, v, e, alpha)
    L = pd.DataFrame(parts).dropna()
    mean = L.mean().sort_values()
    best = mean.index[0]
    rows = []
    for k in mean.index:
        d = (L[k] - L[best]).to_numpy()
        if k == best:
            rows.append({"model": k, "mean_FZ0": mean[k], "diff_vs_best": 0.0, "DM_t": np.nan, "p": np.nan})
            continue
        T = len(d)
        lag = int(np.floor(4 * (T / 100) ** (2 / 9)))
        dc = d - d.mean()
        lrv = dc @ dc / T
        for j in range(1, lag + 1):
            lrv += 2 * (1 - j / (lag + 1)) * (dc[j:] @ dc[:-j]) / T
        t = d.mean() / np.sqrt(lrv / T) if lrv > 0 else np.nan
        rows.append({"model": k, "mean_FZ0": mean[k], "diff_vs_best": d.mean(), "DM_t": t,
                     "p": float(2 * stats.norm.sf(abs(t)))})
    tab = pd.DataFrame(rows).set_index("model")
    if show:
        print(f"\n(VaR, ES) comparison by FZ0 loss, alpha = {alpha} ({len(L)} common days); lower is better")
        print(tab.to_string(float_format=lambda v: f"{v:.4f}"))
        print(f"  Best: {best}. A model with p < 0.05 is significantly worse than the best.")
    return tab
