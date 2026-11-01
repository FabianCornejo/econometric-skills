"""
Monte Carlo laboratory for Teacher mode: small, seeded simulations that show an estimator
working and then failing when one assumption breaks, plus power and minimum detectable effects.
Every demo returns a table (one row per estimator or scenario) and a one-line lesson.

Engine
    monte_carlo(dgp, estimators, R, truth)     bias, SD, RMSE, CI coverage and rejection rate of any estimators

Demos (assumption that breaks -> what the student sees)
    demo_omitted_variable()        E[u|x] != 0 by an omitted confounder -> bias = gamma * delta
    demo_bad_control()             controlling for a variable caused by the treatment -> wrong effect
    demo_measurement_error()       noise in x -> attenuation by var(x*) / (var(x*) + var(noise))
    demo_heteroskedasticity()      classical vs HC3 standard errors -> CI coverage
    demo_spurious_regression()     two independent random walks -> "significant" t, high R2, low DW
    demo_weak_iv()                 weak first stage -> 2SLS pulled toward OLS, bad coverage
    demo_few_clusters()            8 clusters -> CR1 over-rejects; wild cluster bootstrap does not
    demo_staggered_did()           staggered adoption with growing effects -> TWFE wrong; Callaway-Sant'Anna right

Design
    power_mde(...)                 minimum detectable effect or sample size, two-sample or clustered

All functions take R (replications) and seed; defaults are small enough to run in seconds.
Show the code with the result: the student should rerun it and change one number.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05


def _ols(y, X):
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    n, k = X.shape
    V_cl = XtX_inv * (e @ e) / (n - k)
    V_hc3 = XtX_inv @ (X.T * (e / (1 - np.einsum("ij,jk,ik->i", X, XtX_inv, X))) ** 2) @ X @ XtX_inv
    return b, V_cl, V_hc3, e


def monte_carlo(dgp, estimators: dict, R: int = 1000, truth: float | None = None, seed: int = 0,
                alpha: float = ALPHA) -> pd.DataFrame:
    """Run R replications. dgp(rng) -> data; estimators: {name: f(data) -> estimate or (estimate, se)}.
    Reports mean, bias, SD, RMSE and, when an SE is returned, CI coverage of `truth` and the
    rejection rate of H0: parameter = truth (the test's size when truth is right)."""
    rng = np.random.default_rng(seed)
    z = stats.norm.ppf(1 - alpha / 2)
    out = {k: [] for k in estimators}
    for _ in range(R):
        data = dgp(rng)
        for k, f in estimators.items():
            v = f(data)
            out[k].append(v if isinstance(v, tuple) else (v, np.nan))
    rows = []
    for k, vals in out.items():
        a = np.array(vals, float)
        est, se = a[:, 0], a[:, 1]
        row = {"estimator": k, "mean": est.mean(), "median": np.median(est), "sd": est.std(ddof=1)}
        if truth is not None:
            row.update({"bias": est.mean() - truth, "median_bias": np.median(est) - truth,
                        "rmse": np.sqrt(np.mean((est - truth) ** 2))})
            if np.isfinite(se).all():
                cover = np.abs(est - truth) <= z * se
                row.update({"coverage": cover.mean(), "rejection_rate": 1 - cover.mean()})
        rows.append(row)
    return pd.DataFrame(rows).set_index("estimator")


def demo_omitted_variable(R: int = 500, n: int = 500, beta: float = 1.0, gamma: float = 0.8, delta: float = 0.5,
                          seed: int = 0) -> dict:
    """y = beta x + gamma z + u, x = delta z + v (var(z) = var(v) = 1). Omitting z biases the slope
    on x by gamma * Cov(x, z) / Var(x) = gamma * delta / (delta^2 + 1)."""
    def dgp(r):
        zz = r.normal(size=n)
        x = delta * zz + r.normal(size=n)
        return x, zz, beta * x + gamma * zz + r.normal(size=n)

    def short(d):
        x, _, y = d
        b, V, _, _ = _ols(y, np.column_stack([np.ones(n), x]))
        return b[1], np.sqrt(V[1, 1])

    def long(d):
        x, zz, y = d
        b, V, _, _ = _ols(y, np.column_stack([np.ones(n), x, zz]))
        return b[1], np.sqrt(V[1, 1])

    tab = monte_carlo(dgp, {"short regression (z omitted)": short, "long regression (z included)": long}, R, beta, seed)
    predicted = gamma * delta / (delta ** 2 + 1)
    return {"table": tab, "predicted_bias": predicted,
            "lesson": f"Omitting a confounder biases the slope by gamma*Cov(x,z)/Var(x) = {predicted:.3f}; "
                      "more data does not help, and the CIs of the short regression almost never cover the truth."}


def demo_bad_control(R: int = 500, n: int = 1000, effect: float = 1.0, seed: int = 0) -> dict:
    """Randomized treatment t raises a mediator m, and m raises y: y = 0.5 t + m + u, m = 0.5 t + e.
    The total effect of t is 1.0. Controlling for m (caused by t) leaves only the direct 0.5,
    and with a confounder of m and y it biases even that."""
    def dgp(r):
        t = r.integers(0, 2, n).astype(float)
        c = r.normal(size=n)                                   # unobserved: affects m and y
        m = 0.5 * t + c + r.normal(size=n)
        return t, m, 0.5 * t + m + c + r.normal(size=n)

    def total(d):
        t, _, y = d
        b, V, _, _ = _ols(y, np.column_stack([np.ones(n), t]))
        return b[1], np.sqrt(V[1, 1])

    def bad(d):
        t, m, y = d
        b, V, _, _ = _ols(y, np.column_stack([np.ones(n), t, m]))
        return b[1], np.sqrt(V[1, 1])

    tab = monte_carlo(dgp, {"y on t (correct: total effect)": total, "y on t and m (bad control)": bad}, R, effect, seed)
    return {"table": tab, "lesson": "A variable caused by the treatment is not a control: conditioning on it removes part "
                                    "of the effect and, with a common cause of m and y, adds bias (collider)."}


def demo_measurement_error(R: int = 500, n: int = 1000, beta: float = 1.0, noise_sd: float = 1.0, seed: int = 0) -> dict:
    """x observed = x* + noise. OLS converges to beta * lambda, lambda = 1 / (1 + noise_sd^2)."""
    def dgp(r):
        xs = r.normal(size=n)
        return xs, xs + noise_sd * r.normal(size=n), beta * xs + r.normal(size=n)

    def true_x(d):
        xs, _, y = d
        b, V, _, _ = _ols(y, np.column_stack([np.ones(n), xs]))
        return b[1], np.sqrt(V[1, 1])

    def noisy(d):
        _, xo, y = d
        b, V, _, _ = _ols(y, np.column_stack([np.ones(n), xo]))
        return b[1], np.sqrt(V[1, 1])

    lam = 1 / (1 + noise_sd ** 2)
    tab = monte_carlo(dgp, {"true x": true_x, "x measured with error": noisy}, R, beta, seed)
    return {"table": tab, "attenuation_factor": lam,
            "lesson": f"Classical measurement error in x shrinks the slope toward zero by lambda = {lam:.2f}."}


def demo_heteroskedasticity(R: int = 1000, n: int = 200, seed: int = 0) -> dict:
    """y = x + u with sd(u) = |x|: the slope is unbiased, but classical SEs are wrong; HC3 fixes them."""
    def dgp(r):
        x = r.normal(size=n)
        return x, x + np.abs(x) * r.normal(size=n)

    def classical(d):
        x, y = d
        b, V, _, _ = _ols(y, np.column_stack([np.ones(n), x]))
        return b[1], np.sqrt(V[1, 1])

    def hc3(d):
        x, y = d
        b, _, V, _ = _ols(y, np.column_stack([np.ones(n), x]))
        return b[1], np.sqrt(V[1, 1])

    tab = monte_carlo(dgp, {"classical SE": classical, "HC3 SE": hc3}, R, 1.0, seed)
    return {"table": tab, "lesson": "Heteroskedasticity leaves the coefficient unbiased but makes classical CIs too narrow; "
                                    "robust (HC3) SEs restore the coverage."}


def demo_spurious_regression(R: int = 500, T: int = 200, seed: int = 0) -> dict:
    """Regress one random walk on another, independent one (Granger & Newbold 1974)."""
    rng = np.random.default_rng(seed)
    rej, r2, dw = [], [], []
    for _ in range(R):
        y = np.cumsum(rng.normal(size=T))
        x = np.cumsum(rng.normal(size=T))
        b, V, _, e = _ols(y, np.column_stack([np.ones(T), x]))
        rej.append(abs(b[1] / np.sqrt(V[1, 1])) > 1.96)
        r2.append(1 - e @ e / np.sum((y - y.mean()) ** 2))
        dw.append(np.sum(np.diff(e) ** 2) / (e @ e))
    tab = pd.DataFrame({"rejection_rate_of_true_null": [np.mean(rej)], "mean_R2": [np.mean(r2)],
                        "mean_Durbin_Watson": [np.mean(dw)]}, index=["levels regression"])
    return {"table": tab, "lesson": "Independent trending series look strongly related in levels: the t-test rejects most "
                                    "of the time. Warning sign: R2 > Durbin-Watson. Test for unit roots and cointegration."}


def demo_weak_iv(R: int = 1000, n: int = 500, pi_strong: float = 0.5, pi_weak: float = 0.05, rho: float = 0.6,
                 seed: int = 0) -> dict:
    """y = x + u, x = pi z + v, corr(u, v) = rho. Compares OLS and 2SLS with a strong and a weak
    instrument; reports medians (the just-identified IV has no finite mean) and the first-stage F."""
    rng = np.random.default_rng(seed)
    res = {k: [] for k in ("OLS", "2SLS strong", "2SLS weak")}
    F = {"2SLS strong": [], "2SLS weak": []}
    cover = {"2SLS strong": [], "2SLS weak": []}
    for _ in range(R):
        z = rng.normal(size=n)
        u = rng.normal(size=n)
        v = rho * u + np.sqrt(1 - rho ** 2) * rng.normal(size=n)
        for name, pi in (("2SLS strong", pi_strong), ("2SLS weak", pi_weak)):
            x = pi * z + v
            y = x + u
            zc, xc, yc = z - z.mean(), x - x.mean(), y - y.mean()
            b = (zc @ yc) / (zc @ xc)
            e = yc - b * xc
            se = np.sqrt((e @ e / (n - 2)) * (zc @ zc) / (zc @ xc) ** 2)
            res[name].append(b)
            cover[name].append(abs(b - 1) <= 1.96 * se)
            pihat = (zc @ xc) / (zc @ zc)
            ev = xc - pihat * zc
            F[name].append(pihat ** 2 * (zc @ zc) / (ev @ ev / (n - 2)))
            if name == "2SLS strong":
                res["OLS"].append((xc @ yc) / (xc @ xc))
    rows = []
    for k, v_ in res.items():
        a = np.array(v_)
        rows.append({"estimator": k, "median": np.median(a), "median_bias": np.median(a) - 1,
                     "iqr": np.subtract(*np.percentile(a, [75, 25])),
                     "median_first_stage_F": np.median(F[k]) if k in F else np.nan,
                     "coverage": np.mean(cover[k]) if k in cover else np.nan})
    return {"table": pd.DataFrame(rows).set_index("estimator"),
            "lesson": "With a weak first stage, 2SLS is centered near OLS (biased toward it) and very dispersed; check "
                      "the effective first-stage F (Montiel Olea-Pflueger; F > 104.7 for reliable t-tests, Lee et al. "
                      "2022) and use Anderson-Rubin intervals when it is low."}


def demo_few_clusters(R: int = 200, G: int = 8, m: int = 30, B: int = 199, seed: int = 0) -> dict:
    """Regressor and errors correlated within G clusters; true slope 0. Rejection rates at 5% of
    CR1 with normal critical values, CR1 with t(G-1), and the wild cluster bootstrap."""
    import statsmodels.formula.api as smf
    import estimation
    rng = np.random.default_rng(seed)
    rej = {"CR1, normal critical values": [], "CR1, t(G-1)": [], "wild cluster bootstrap": []}
    for rep in range(R):
        g = np.repeat(np.arange(G), m)
        x = np.repeat(rng.normal(size=G), m) + 0.5 * rng.normal(size=G * m)
        u = np.repeat(rng.normal(size=G), m) + rng.normal(size=G * m)
        d = pd.DataFrame({"g": g, "x": x, "y": u})
        f = smf.ols("y ~ x", d).fit(cov_type="cluster", cov_kwds={"groups": d["g"]})
        t = f.params["x"] / f.bse["x"]
        rej["CR1, normal critical values"].append(abs(t) > 1.96)
        rej["CR1, t(G-1)"].append(abs(t) > stats.t.ppf(0.975, G - 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = estimation.wild_cluster_bootstrap(d, "y ~ x", "x", "g", B=B, seed=rep, ci=False)
        rej["wild cluster bootstrap"].append(w["p"] < 0.05)
    tab = pd.DataFrame({"rejection_rate_of_true_null": {k: np.mean(v) for k, v in rej.items()}})
    return {"table": tab, "lesson": f"With {G} clusters, clustered SEs understate uncertainty; the number of clusters, "
                                    "not N, is the sample size. The wild cluster bootstrap brings the size much closer to 5% "
                                    "(about 7% with 8 clusters in this design, against ~25% for CR1)."}


def demo_staggered_did(R: int = 30, n_units: int = 60, T: int = 10, seed: int = 0) -> dict:
    """Three cohorts adopt at t = 3, 6 and never; the effect grows by 1 per period since adoption.
    TWFE compares late adopters with already-treated units and is badly biased; Callaway-Sant'Anna
    uses only clean comparisons."""
    import causal
    rng = np.random.default_rng(seed)
    twfe, cs, truth_all = [], [], []
    for rep in range(R):
        gcohort = rng.choice([3, 6, 0], n_units, p=[0.4, 0.4, 0.2])
        rows = []
        a = rng.normal(size=n_units)
        for i in range(n_units):
            for t in range(1, T + 1):
                k = t - gcohort[i] if gcohort[i] > 0 and t >= gcohort[i] else -1
                eff = (k + 1) if k >= 0 else 0.0
                rows.append((i, t, gcohort[i], a[i] + 0.1 * t + eff + rng.normal(), eff))
        d = pd.DataFrame(rows, columns=["unit", "time", "g", "y", "eff"])
        d["D"] = ((d.g > 0) & (d.time >= d.g)).astype(float)
        truth_all.append(d.loc[d.D == 1, "eff"].mean())
        U = pd.get_dummies(d.unit).to_numpy(float)
        Tm = pd.get_dummies(d.time).to_numpy(float)[:, 1:]
        X = np.column_stack([d.D.to_numpy(), U, Tm])
        twfe.append(np.linalg.lstsq(X, d.y.to_numpy(), rcond=None)[0][0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = causal.att_gt(d, "unit", "time", "y", "g", n_boot=0, show=False)
        cs.append(res["overall"]["estimate"])
    truth = float(np.mean(truth_all))
    tab = pd.DataFrame({"mean": [np.mean(twfe), np.mean(cs)], "bias": [np.mean(twfe) - truth, np.mean(cs) - truth]},
                       index=["TWFE", "Callaway-Sant'Anna"])
    return {"table": tab, "truth_att": truth,
            "lesson": "With staggered adoption and effects that grow over time, the TWFE coefficient is a badly weighted "
                      "average (some weights negative); Callaway-Sant'Anna recovers the ATT."}


def power_mde(n: float | None = None, mde: float | None = None, sd: float = 1.0, alpha: float = ALPHA,
              power: float = 0.8, share_treated: float = 0.5, cluster_size: float = 1.0, icc: float = 0.0) -> dict:
    """Minimum detectable effect (given n) or total sample size (given mde) for a two-sided
    two-group comparison of means, with a design effect 1 + (m - 1) ICC for cluster randomization:
        MDE = (z_{1-alpha/2} + z_power) * sd * sqrt(DE / (p (1 - p) n)).
    For a DiD or a regression with controls, sd is the residual SD after the controls (smaller),
    which lowers the MDE; for few clusters, use simulation instead of this normal approximation."""
    k = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    de = 1 + (cluster_size - 1) * icc
    p = share_treated
    if (n is None) == (mde is None):
        raise ValueError("give exactly one of n or mde")
    if n is not None:
        val = k * sd * np.sqrt(de / (p * (1 - p) * n))
        return {"mde": float(val), "mde_in_sd": float(val / sd), "n": n, "design_effect": de, "alpha": alpha, "power": power}
    nn = (k * sd / mde) ** 2 * de / (p * (1 - p))
    return {"n": float(np.ceil(nn)), "mde": mde, "design_effect": de, "alpha": alpha, "power": power,
            "n_clusters": float(np.ceil(nn / cluster_size)) if cluster_size > 1 else None}
