"""
Dependence beyond Pearson correlation: rank correlations, tail dependence,
bivariate copulas (Gaussian, Student-t, Clayton, Gumbel, Frank and survival
rotations) fitted by maximum likelihood, mutual information and distance
correlation.

Why it matters: two assets can have the same Pearson correlation and very
different joint crash risk. Copulas separate the marginals from the dependence
structure; the tail-dependence coefficient says how likely an extreme loss in one
is given an extreme loss in the other.

Functions
---------
pseudo_obs(x, y)                    ranks / (n+1)  (margins-free uniforms)
rank_correlations(df)               Pearson, Spearman, Kendall side by side
empirical_tail_dependence(x, y, q)  P(V<q | U<q) and P(V>1-q | U>1-q)
fit_copula(u, v, family)            MLE for one family -> params, loglik, AIC, BIC, tau, tails
fit_copulas(x, y)                   all families ranked by AIC (with plain-language reading)
simulate_copula(family, params, n)  draws for scenario analysis / portfolio VaR
mutual_information_matrix(df)       kNN MI (nats), Linfoot-normalized to a 0-1 scale
dependence_table(df)                |Pearson| vs normalized MI vs distance correlation, per pair
distance_correlation(x, y)          Szekely-Rizzo (0 iff independent)
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

EPS = 1e-10
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------
def pseudo_obs(x, y=None):
    if y is None:
        x = pd.DataFrame(x)
        return x.rank() / (len(x) + 1)
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    return stats.rankdata(x) / (n + 1), stats.rankdata(y) / (n + 1)


def rank_correlations(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for a, b in itertools.combinations(df.columns, 2):
        d = df[[a, b]].dropna()
        tau, pt = stats.kendalltau(d[a], d[b])
        out.append({"pair": f"{a} ~ {b}", "pearson": d[a].corr(d[b]),
                    "spearman": d[a].corr(d[b], method="spearman"), "kendall_tau": tau,
                    "kendall_p": pt, "n": len(d)})
    return pd.DataFrame(out).set_index("pair")


def empirical_tail_dependence(x, y, q: float = 0.05) -> dict:
    u, v = pseudo_obs(x, y)
    lower = np.mean((u <= q) & (v <= q)) / q
    upper = np.mean((u > 1 - q) & (v > 1 - q)) / q
    return {"q": q, "lambda_L_emp": lower, "lambda_U_emp": upper,
            "independence_benchmark": q}


# ---------------------------------------------------------------------------
# Copula log-densities (u, v in (0,1))
# ---------------------------------------------------------------------------
def _ll_gaussian(p, u, v):
    rho = p[0]
    x, y = stats.norm.ppf(u), stats.norm.ppf(v)
    r2 = 1 - rho ** 2
    return np.sum(-0.5 * np.log(r2) - (rho ** 2 * (x ** 2 + y ** 2) - 2 * rho * x * y) / (2 * r2))


def _ll_t(p, u, v):
    rho, nu = p
    x, y = stats.t.ppf(u, nu), stats.t.ppf(v, nu)
    r2 = 1 - rho ** 2
    log_c2 = (special.gammaln((nu + 2) / 2) - special.gammaln(nu / 2) - np.log(nu * np.pi)
              - 0.5 * np.log(r2) - (nu + 2) / 2 * np.log1p((x ** 2 - 2 * rho * x * y + y ** 2) / (nu * r2)))
    return np.sum(log_c2 - stats.t.logpdf(x, nu) - stats.t.logpdf(y, nu))


def _ll_clayton(p, u, v):
    th = p[0]
    return np.sum(np.log1p(th) + (-1 - th) * (np.log(u) + np.log(v))
                  + (-2 - 1 / th) * np.log(u ** -th + v ** -th - 1))


def _ll_gumbel(p, u, v):
    th = p[0]
    lu, lv = -np.log(u), -np.log(v)
    A = lu ** th + lv ** th
    Ath = A ** (1 / th)
    return np.sum(-Ath - np.log(u) - np.log(v) + (th - 1) * (np.log(lu) + np.log(lv))
                  + (2 / th - 2) * np.log(A) + np.log1p((th - 1) * A ** (-1 / th)))


def _ll_frank(p, u, v):
    th = p[0]
    if abs(th) < 1e-6:
        return 0.0
    num = np.log(abs(th)) + np.log(abs(1 - np.exp(-th))) - th * (u + v)
    den = 2 * np.log(np.abs((1 - np.exp(-th)) - (1 - np.exp(-th * u)) * (1 - np.exp(-th * v))))
    return np.sum(num - den)


FAMILIES = {
    # name: (loglik, x0 from tau, bounds, survival-rotated?)
    "gaussian": (_ll_gaussian, lambda t: [np.sin(np.pi * t / 2)], [(-0.999, 0.999)], False),
    "student_t": (_ll_t, lambda t: [np.sin(np.pi * t / 2), 8.0], [(-0.999, 0.999), (2.05, 100)], False),
    "clayton": (_ll_clayton, lambda t: [max(2 * t / (1 - t), 0.05)], [(1e-3, 50)], False),
    "gumbel": (_ll_gumbel, lambda t: [max(1 / (1 - t), 1.01)], [(1.0001, 50)], False),
    "frank": (_ll_frank, lambda t: [np.sign(t) * max(abs(5 * t), 0.1)], [(-35, 35)], False),
    "survival_clayton": (_ll_clayton, lambda t: [max(2 * t / (1 - t), 0.05)], [(1e-3, 50)], True),
    "survival_gumbel": (_ll_gumbel, lambda t: [max(1 / (1 - t), 1.01)], [(1.0001, 50)], True),
}


def _tails(family, p):
    if family == "student_t":
        rho, nu = p
        lam = 2 * stats.t.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), nu + 1)
        return lam, lam
    if family == "clayton":
        return 2 ** (-1 / p[0]), 0.0
    if family == "survival_clayton":
        return 0.0, 2 ** (-1 / p[0])
    if family == "gumbel":
        return 0.0, 2 - 2 ** (1 / p[0])
    if family == "survival_gumbel":
        return 2 - 2 ** (1 / p[0]), 0.0
    return 0.0, 0.0                       # gaussian (rho<1), frank


def _tau(family, p):
    if family in ("gaussian", "student_t"):
        return 2 / np.pi * np.arcsin(p[0])
    if "clayton" in family:
        return p[0] / (p[0] + 2)
    if "gumbel" in family:
        return 1 - 1 / p[0]
    if family == "frank":
        th = p[0]
        if abs(th) < 1e-6:
            return 0.0
        k = np.linspace(1e-8, abs(th), 2000)
        debye = _trapz(k / np.expm1(k), k) / abs(th)
        return np.sign(th) * (1 - 4 / abs(th) * (1 - debye))
    return np.nan


def fit_copula(u, v, family: str = "student_t") -> dict:
    u = np.clip(np.asarray(u, float), EPS, 1 - EPS)
    v = np.clip(np.asarray(v, float), EPS, 1 - EPS)
    ll, x0f, bounds, rot = FAMILIES[family]
    if rot:
        u, v = 1 - u, 1 - v
    tau_emp = stats.kendalltau(u, v)[0]
    if family in ("clayton", "gumbel", "survival_clayton", "survival_gumbel") and tau_emp <= 0:
        return {"family": family, "params": None, "loglik": np.nan, "AIC": np.nan, "BIC": np.nan,
                "note": "needs positive dependence (rotate the data 90° for negative dependence)"}
    x0 = np.clip(x0f(tau_emp), [b[0] for b in bounds], [b[1] for b in bounds])

    def nll(p):
        val = -ll(np.atleast_1d(p), u, v)
        return val if np.isfinite(val) else 1e12          # penalty instead of inf/NaN

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        if len(bounds) == 1:                               # 1-parameter families: bounded scalar search
            lo, hi = bounds[0]
            if family == "frank":                          # search on the side of the empirical tau
                lo, hi = (1e-4, hi) if tau_emp >= 0 else (lo, -1e-4)
            r1 = optimize.minimize_scalar(nll, bounds=(lo, hi), method="bounded",
                                          options={"xatol": 1e-6})
            res = optimize.OptimizeResult(x=np.array([r1.x]), fun=r1.fun, success=r1.success)
        else:
            res = optimize.minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
    p = res.x
    k, n = len(p), len(u)
    if res.fun >= 1e12:
        return {"family": family, "params": None, "loglik": np.nan, "AIC": np.nan, "BIC": np.nan,
                "note": "likelihood not finite"}
    lam_l, lam_u = _tails(family, p)
    return {"family": family, "params": np.round(p, 6).tolist(), "loglik": -res.fun,
            "AIC": 2 * k + 2 * res.fun, "BIC": k * np.log(n) + 2 * res.fun,
            "implied_tau": _tau(family, p), "lambda_L": lam_l, "lambda_U": lam_u,
            "converged": bool(res.success)}


def fit_copulas(x, y, families: list[str] | None = None, show: bool = True) -> pd.DataFrame:
    """Fit all families on pseudo-observations of (x, y), rank by AIC."""
    d = pd.concat([pd.Series(np.asarray(x, float)), pd.Series(np.asarray(y, float))], axis=1).dropna()
    u, v = pseudo_obs(d.iloc[:, 0], d.iloc[:, 1])
    fams = families or list(FAMILIES)
    tab = pd.DataFrame([fit_copula(u, v, f) for f in fams]).set_index("family").sort_values("AIC")
    if show:
        tau = stats.kendalltau(u, v)[0]
        emp = empirical_tail_dependence(d.iloc[:, 0], d.iloc[:, 1], 0.05)
        print(f"\nCopula fit on {len(u)} pseudo-observations | Kendall tau = {tau:.3f} | "
              f"empirical tail co-exceedance at 5%: lower {emp['lambda_L_emp']:.2f}, "
              f"upper {emp['lambda_U_emp']:.2f} (0.05 under independence)")
        cols = ["params", "loglik", "AIC", "BIC", "implied_tau", "lambda_L", "lambda_U"]
        print(tab[[c for c in cols if c in tab]].to_string(float_format=lambda v: f"{v:.4f}"))
        best = tab.index[0]
        g = tab.loc["gaussian", "AIC"] if "gaussian" in tab.index else np.nan
        print(f"Best by AIC: {best}. "
              + (f"AIC gain over Gaussian = {g - tab.loc[best, 'AIC']:.1f} "
                 "(>10 is strong evidence that the Gaussian copula understates joint tail risk)."
                 if best != "gaussian" and pd.notna(g) else ""))
    return tab


def simulate_copula(family: str, params, n: int = 10000, seed: int = 0) -> np.ndarray:
    """Draw (u, v) from a fitted copula. Map back with the empirical or fitted marginal
    quantile functions (e.g. np.quantile(returns, u)) to simulate joint returns."""
    rng = np.random.default_rng(seed)
    if family == "gaussian":
        rho = params[0]
        z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)
        return stats.norm.cdf(z)
    if family == "student_t":
        rho, nu = params
        z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], n)
        w = np.sqrt(nu / rng.chisquare(nu, n))[:, None]
        return stats.t.cdf(z * w, nu)
    if "clayton" in family:
        th = params[0]
        g = rng.gamma(1 / th, 1, n)[:, None]
        e = rng.exponential(1, (n, 2))
        uv = (1 + e / g) ** (-1 / th)
        return 1 - uv if family.startswith("survival") else uv
    if family == "frank":
        th = params[0]
        u, w = rng.uniform(size=n), rng.uniform(size=n)
        v = -np.log1p(w * np.expm1(-th) / (w + (1 - w) * np.exp(-th * u))) / th
        return np.column_stack([u, v])
    if "gumbel" in family:
        # Marshall-Olkin with a positive stable frailty (Chambers-Mallows-Stuck)
        th = params[0]
        a = 1 / th
        U = rng.uniform(0, np.pi, n)
        W = rng.exponential(1, n)
        S = (np.sin(a * U) / np.sin(U) ** (1 / a)) * (np.sin((1 - a) * U) / W) ** ((1 - a) / a)
        e = rng.exponential(1, (n, 2))
        uv = np.exp(-(e / S[:, None]) ** a)
        return 1 - uv if family.startswith("survival") else uv
    raise ValueError(family)


# ---------------------------------------------------------------------------
# Information-theoretic and distance-based dependence
# ---------------------------------------------------------------------------
def _mi(x, y, k=3, seed=0):
    from sklearn.feature_selection import mutual_info_regression
    return float(mutual_info_regression(np.asarray(x).reshape(-1, 1), np.asarray(y),
                                        n_neighbors=k, random_state=seed)[0])


def linfoot(mi: float) -> float:
    """Information coefficient of correlation sqrt(1 - exp(-2 MI)): equals |rho| for a
    bivariate normal, so it is directly comparable with |Pearson|."""
    return float(np.sqrt(1 - np.exp(-2 * max(mi, 0.0))))


def mutual_information_matrix(df: pd.DataFrame, k: int = 3, normalized: bool = True) -> pd.DataFrame:
    """Symmetric kNN (Kraskov) MI matrix. Diagonal is left NaN: MI(X,X) is infinite for
    continuous data, so setting it to 1 (a common mistake) mixes scales."""
    cols = list(df.columns)
    M = pd.DataFrame(np.nan, index=cols, columns=cols)
    d = df.dropna()
    for a, b in itertools.combinations(cols, 2):
        mi = 0.5 * (_mi(d[a], d[b], k) + _mi(d[b], d[a], k))
        val = linfoot(mi) if normalized else mi
        M.loc[a, b] = M.loc[b, a] = val
    return M


def distance_correlation(x, y) -> float:
    x = np.asarray(x, float)[:, None]
    y = np.asarray(y, float)[:, None]
    if len(x) > 5000:
        idx = np.random.default_rng(0).choice(len(x), 5000, replace=False)
        x, y = x[idx], y[idx]
    a = np.abs(x - x.T)
    b = np.abs(y - y.T)
    A = a - a.mean(0) - a.mean(1)[:, None] + a.mean()
    B = b - b.mean(0) - b.mean(1)[:, None] + b.mean()
    dcov = (A * B).mean()
    return float(np.sqrt(max(dcov, 0) / np.sqrt((A * A).mean() * (B * B).mean())))


def mi_permutation_test(x, y, n_perm: int = 200, k: int = 3, seed: int = 0) -> dict:
    """H0: independence. Shuffling y breaks any dependence (iid assumption: for
    autocorrelated series use block permutations or residuals)."""
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    obs = _mi(x, y, k)
    null = [_mi(x, rng.permutation(y), k) for _ in range(n_perm)]
    return {"MI": obs, "p": (1 + np.sum(np.array(null) >= obs)) / (n_perm + 1)}


def dependence_table(df: pd.DataFrame, k: int = 3, show: bool = True) -> pd.DataFrame:
    """Per pair: |Pearson|, Spearman, Linfoot-normalized MI and distance correlation.
    'nonlinear_gap' = normalized MI - |Pearson|: large positive values flag dependence
    that a correlation matrix misses (nonlinearity, tail or volatility co-movement)."""
    d = df.dropna()
    rows = []
    for a, b in itertools.combinations(d.columns, 2):
        mi = 0.5 * (_mi(d[a], d[b], k) + _mi(d[b], d[a], k))
        r = d[a].corr(d[b])
        rows.append({"pair": f"{a} ~ {b}", "pearson": r, "spearman": d[a].corr(d[b], method="spearman"),
                     "MI_nats": mi, "MI_linfoot": linfoot(mi), "distance_corr": distance_correlation(d[a], d[b]),
                     "nonlinear_gap": linfoot(mi) - abs(r)})
    tab = pd.DataFrame(rows).set_index("pair").sort_values("nonlinear_gap", ascending=False)
    if show:
        print("\nDependence measures (MI_linfoot is on the |correlation| scale)")
        print(tab.to_string(float_format=lambda v: f"{v:.3f}"))
        print("kNN MI has a small-sample bias of order ±0.02 nats; treat gaps below ~0.05 as noise.")
    return tab
