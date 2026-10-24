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
backtest_var(r, var_dict, alpha)             one table for several VaR models

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
        last = _hits(common["r"].iloc[-250:], common[k].iloc[-250:]).sum()
        c["avg_VaR"] = common[k].mean()
        c["traffic_light_last_250"] = basel_traffic_light(int(last), min(250, len(common)), alpha) \
            if abs(alpha - 0.01) < 1e-9 else "n/a"
        rows[k] = c
    tab = pd.DataFrame(rows).T[["n", "violations", "expected", "rate", "p_uc", "p_ind", "p_cc",
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
            print(f"  {k}: " + ("; ".join(msgs) if msgs else "passes coverage and independence at 5%"))
        print("  Among models that pass, prefer the lower average VaR (less capital for the same coverage).")
    return tab
