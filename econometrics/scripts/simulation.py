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
    demo_did_covariates()          trends depend on covariate levels -> TWFE with covariates wrong (hidden linearity
                                   bias); Callaway-Sant'Anna with covariates in levels right

Statistics foundations (v2.4.0)
    demo_clt()                     sampling distribution of the mean: SD = sigma/sqrt(n) always; normal shape and
                                   t coverage only gradually for skewed populations
    demo_ci_coverage()             what 95% means: t vs z-with-s, skewed data, Wald vs Wilson for a proportion
    demo_estimator_mse()           bias, variance and MSE: S^2 vs MLE of a variance; MoM vs MLE for a uniform(0, theta)
    demo_multiple_testing()        m tests: family-wise error and false discovery rate, raw vs Bonferroni/Holm/BH

Classical regression (v2.4.2)
    demo_simultaneity()            supply and demand: OLS mixes the curves; ILS = 2SLS with a demand shifter
    demo_lpm_bounds()              logit truth: LPM slope ~ average marginal effect, fitted values leave [0, 1]
    demo_multicollinearity()       corr(x1, x2) -> 1: unbiased, SE x sqrt(VIF), joint F still rejects

Design
    power_mde(...)                 minimum detectable effect or sample size, two-sample or clustered
    mde_did(...)                   analytic MDE / units for a DiD with m pre and r post periods
                                   (McKenzie 2012, equicorrelated errors)
    power_did_sim(...)             simulated power, size and coverage of a TWFE DiD with clustered SEs
                                   under AR(1) errors, unit effects, simultaneous or staggered adoption
    mde_did_sim(...)               smallest effect with the target simulated power (bisection)
    did_power_inputs(df, unit, time, y)  sd_unit, sd_e, rho_ar, time_sd for power_did_sim, estimated from
                                   the pre-period panel (two-way FE residuals, Nickell-corrected AR(1))

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


def demo_did_covariates(R: int = 30, n_units: int = 400, T: int = 4, beta_drift: float = 0.5,
                        seed: int = 0) -> dict:
    """Parallel trends hold only conditional on X: the covariate's effect rises by beta_drift per
    period, treated units (adopting at t = 3) have higher X, and X changes over time without being
    affected by treatment. True ATT = 1. Unadjusted DiD and TWFE with the covariate are biased by
    about the same amount (fixed effects keep only the changes of X, not its levels: hidden
    linearity bias, Caetano & Callaway 2024); Callaway-Sant'Anna with X in levels is not.
    Set beta_drift=0 to see every estimator agree."""
    import causal
    rng = np.random.default_rng(seed)
    out = {"DiD, no covariates": [], "TWFE with X": [], "CS doubly robust, X in levels": []}
    for _ in range(R):
        g = np.where(rng.random(n_units) < 0.5, 3, 0)
        a = rng.normal(0.5 * (g > 0), 1)
        x = rng.normal(1.0 * (g > 0), 1)
        rows = []
        for t in range(1, T + 1):
            x = x + rng.normal(0.2, 0.4, n_units)
            yv = a + (1 + beta_drift * t) * x + 1.0 * ((g > 0) & (t >= g)) + rng.normal(0, 1, n_units)
            rows.append(pd.DataFrame({"unit": np.arange(n_units), "time": t, "y": yv, "x": x, "g": g}))
        d = pd.concat(rows, ignore_index=True)
        d["D"] = ((d.g > 0) & (d.time >= d.g)).astype(float)
        dm = d.copy()                                  # two-way within transformation (balanced panel)
        for c in ("y", "D", "x"):
            dm[c] = d[c] - d.groupby("unit")[c].transform("mean") - d.groupby("time")[c].transform("mean") + d[c].mean()
        out["TWFE with X"].append(np.linalg.lstsq(dm[["D", "x"]].to_numpy(), dm["y"].to_numpy(), rcond=None)[0][0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out["DiD, no covariates"].append(causal.att_gt(d, "unit", "time", "y", "g", n_boot=0,
                                                            show=False)["overall"]["estimate"])
            out["CS doubly robust, X in levels"].append(causal.att_gt(
                d, "unit", "time", "y", "g", n_boot=0, covariates=["x"], x_periods="both", verify=False,
                show=False)["overall"]["estimate"])
    tab = pd.DataFrame({"mean": {k: np.mean(v) for k, v in out.items()},
                        "sd": {k: np.std(v, ddof=1) for k, v in out.items()}})
    tab["bias"] = tab["mean"] - 1.0
    return {"table": tab, "truth_att": 1.0,
            "lesson": "When trends depend on covariate levels, adding the covariate to a TWFE regression does not help: "
                      "the fixed effects keep only its changes. Condition on levels, group-time by group-time "
                      "(att_gt with covariates), and test whether covariate effects differ across groups (ccc_test)."}


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



# =============================================================================
# Power for difference-in-differences (v2.2)
# =============================================================================
def mde_did(n_units: float | None = None, mde: float | None = None, n_pre: int = 1, n_post: int = 1,
            share_treated: float = 0.5, sd: float = 1.0, rho: float = 0.0, alpha: float = ALPHA,
            power: float = 0.8) -> dict:
    """Analytic MDE (given n_units) or number of units (given mde) for a DiD comparing the
    average of n_pre pre-periods with n_post post-periods, two-sided test (McKenzie 2012):
        Var(DiD) = sd^2 (1 - rho) (1/m + 1/r) / (P (1 - P) N),
    where sd is the total SD of the outcome and rho its constant (equicorrelated) autocorrelation:
    unit effects with an i.i.d. shock give exactly this structure, and more periods help.
    With decaying (AR(1)) autocorrelation this formula overstates power (Burlig, Preonas &
    Woerman 2020): use power_did_sim, which simulates the actual error process and the
    clustered standard errors you will use."""
    k = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    p = share_treated
    f = sd ** 2 * (1 - rho) * (1 / n_pre + 1 / n_post) / (p * (1 - p))
    if (n_units is None) == (mde is None):
        raise ValueError("give exactly one of n_units or mde")
    if n_units is not None:
        v = k * np.sqrt(f / n_units)
        return {"mde": float(v), "mde_in_sd": float(v / sd), "n_units": n_units, "n_pre": n_pre,
                "n_post": n_post, "rho": rho, "alpha": alpha, "power": power}
    return {"n_units": float(np.ceil(f * (k / mde) ** 2)), "mde": mde, "n_pre": n_pre, "n_post": n_post,
            "rho": rho, "alpha": alpha, "power": power}


def _did_panel(rng, n_units, T, treat_start, effect, sd_unit, sd_e, rho_ar, time_sd):
    """Balanced panel: y = a_i + l_t + effect * D_it + e_it, e AR(1) with stationary SD sd_e."""
    a = rng.normal(0, sd_unit, n_units)[:, None]
    lam = rng.normal(0, time_sd, T)[None, :]
    e = np.empty((n_units, T))
    e[:, 0] = rng.normal(0, sd_e, n_units)
    innov = sd_e * np.sqrt(1 - rho_ar ** 2)
    for t in range(1, T):
        e[:, t] = rho_ar * e[:, t - 1] + rng.normal(0, innov, n_units)
    tt = np.arange(1, T + 1)[None, :]
    D = ((treat_start[:, None] > 0) & (tt >= treat_start[:, None])).astype(float)
    return a + lam + effect * D + e, D


def _twfe_cluster(Y, D):
    """TWFE coefficient on D with CR1 unit-clustered SE, by two-way demeaning (balanced panel)."""
    def dm(M):
        return M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + M.mean()
    y, d = dm(Y), dm(D)
    G, T = Y.shape
    sdd = np.sum(d * d)
    b = np.sum(d * y) / sdd
    u = y - b * d
    sg = np.sum(d * u, axis=1)
    k = 1 + (G - 1) + (T - 1) + 1          # intercept, unit and time dummies, D
    c = G / (G - 1) * (G * T - 1) / (G * T - k)
    se = np.sqrt(c * np.sum(sg ** 2)) / sdd
    return b, se


def power_did_sim(n_units: int = 100, T: int = 10, share_treated: float = 0.5, effect: float = 0.2,
                  adoption=None, sd_unit: float = 1.0, sd_e: float = 1.0, rho_ar: float = 0.0,
                  time_sd: float = 0.5, alpha: float = ALPHA, R: int = 1000, seed: int = 0,
                  with_size: bool = True) -> dict:
    """Simulated power of the static TWFE DiD with unit-clustered (CR1) SEs and t(G-1) critical
    values, for the design you plan, with the same seeds for every effect size.

    adoption: None = all treated units adopt at period T//2 + 1; an int = that period; a list
    of periods = staggered adoption, assigned to treated units in turn. AR(1) errors with
    autocorrelation rho_ar and stationary SD sd_e, unit effects (SD sd_unit), common time shocks.
    The effect is constant, so TWFE is unbiased here; with staggered adoption and effects that
    change over time, simulate att_gt with monte_carlo instead (demo_staggered_did shows why).
    Returns power, size (effect 0, same draws) and CI coverage; with fewer than ~10 treated
    units clustered SEs over-reject: plan inference by randomization_inference instead."""
    rng0 = np.random.default_rng(seed)
    n_t = int(round(share_treated * n_units))
    if adoption is None:
        adoption = [T // 2 + 1]
    elif np.isscalar(adoption):
        adoption = [int(adoption)]
    adoption = list(adoption)
    warn = []
    if n_t < 10:
        warn.append(f"{n_t} treated units: clustered SEs over-reject; plan randomization inference")
    rej, rej0, cover, ests = [], [], [], []
    crit = stats.t.ppf(1 - alpha / 2, n_units - 1)
    for _ in range(R):
        rng = np.random.default_rng(rng0.integers(2 ** 32))
        start = np.zeros(n_units)
        idx = rng.permutation(n_units)[:n_t]
        start[idx] = [adoption[j % len(adoption)] for j in range(n_t)]
        state = rng.bit_generator.state
        Y, D = _did_panel(rng, n_units, T, start, effect, sd_unit, sd_e, rho_ar, time_sd)
        b, se = _twfe_cluster(Y, D)
        rej.append(abs(b / se) > crit)
        cover.append(abs(b - effect) <= crit * se)
        ests.append(b)
        if with_size:
            rng.bit_generator.state = state
            Y0, _ = _did_panel(rng, n_units, T, start, 0.0, sd_unit, sd_e, rho_ar, time_sd)
            b0, se0 = _twfe_cluster(Y0, D)
            rej0.append(abs(b0 / se0) > crit)
    ests = np.array(ests)
    return {"power": float(np.mean(rej)), "size": float(np.mean(rej0)) if with_size else None,
            "coverage": float(np.mean(cover)), "mean_estimate": float(ests.mean()), "sd_estimate": float(ests.std()),
            "effect": effect, "n_units": n_units, "T": T, "n_treated": n_t, "rho_ar": rho_ar, "R": R,
            "mc_se_power": float(np.sqrt(np.mean(rej) * (1 - np.mean(rej)) / R)), "warnings": warn}


def mde_did_sim(target_power: float = 0.8, lo: float = 0.0, hi: float | None = None, tol: float = 0.005,
                **kwargs) -> dict:
    """Smallest constant effect whose simulated power (power_did_sim, same seeds) reaches
    target_power, by bisection; kwargs are passed to power_did_sim (R, n_units, T, rho_ar, ...).
    Start `hi` at a few times the analytic mde_did if known."""
    kwargs = {**kwargs, "with_size": False}
    if hi is None:
        hi = 1.0
        while power_did_sim(effect=hi, **kwargs)["power"] < target_power:
            hi *= 2
            if hi > 1e6:
                raise RuntimeError("power never reaches the target")
    while hi - lo > tol * max(hi, 1e-12):
        mid = 0.5 * (lo + hi)
        if power_did_sim(effect=mid, **kwargs)["power"] >= target_power:
            hi = mid
        else:
            lo = mid
    pw = power_did_sim(effect=hi, **{**kwargs, "with_size": True})
    return {"mde": hi, "power_at_mde": pw["power"], "size": pw["size"], "target_power": target_power,
            **{k: kwargs[k] for k in ("n_units", "T", "rho_ar") if k in kwargs}}



def did_power_inputs(df: pd.DataFrame, unit: str, time: str, y: str) -> dict:
    """Estimate the inputs of power_did_sim from historical (pre-treatment) panel data:
      sd_unit : SD of the unit means (permanent differences, absorbed by unit FE)
      time_sd : SD of the time means (common shocks, absorbed by time FE)
      sd_e    : SD of the two-way FE residuals (the noise the DiD must see through)
      rho_ar  : AR(1) coefficient of those residuals, corrected for the Nickell (1981) bias of
                demeaned panels: plim(rho_hat) - rho ~ -(1 + rho) / (T - 1), so
                rho = (rho_hat (T - 1) + 1) / (T - 2).
    Uses a balanced panel (units missing periods are dropped). The SD a user quotes is often the
    total SD across units; the relevant one is sd_e, usually much smaller."""
    w = df.pivot_table(index=unit, columns=time, values=y, aggfunc="mean").dropna()
    Y = w.to_numpy(float)
    G, T = Y.shape
    if T < 4 or G < 3:
        raise ValueError("need at least 3 units and 4 periods of balanced data")
    um, tm = Y.mean(1), Y.mean(0)
    E = Y - um[:, None] - tm[None, :] + Y.mean()
    k = (G - 1) + (T - 1) + 1
    sd_e = float(np.sqrt(np.sum(E ** 2) / (G * T - k)))
    a, b = E[:, :-1].ravel(), E[:, 1:].ravel()
    rho_hat = float(a @ b / (a @ a))
    rho = float(np.clip((rho_hat * (T - 1) + 1) / (T - 2), -0.99, 0.99))
    warn = [] if T >= 10 else [f"only {T} periods: the AR(1) estimate is imprecise; check power for a range of rho_ar"]
    return {"sd_unit": float(um.std(ddof=1)), "time_sd": float(tm.std(ddof=1)), "sd_e": sd_e,
            "rho_ar": rho, "rho_ar_uncorrected": rho_hat, "n_units": G, "T": T,
            "sd_total": float(Y.std(ddof=1)), "warnings": warn}


# =============================================================================
# Statistics foundations (v2.4.0): sampling distributions, coverage, estimator quality, many tests
# =============================================================================
def _population(dist: str, rng, size):
    """Draws from a few populations with known mean and SD (returned with them)."""
    if dist == "normal":
        return rng.normal(0.0, 1.0, size), 0.0, 1.0
    if dist == "exponential":
        return rng.exponential(1.0, size), 1.0, 1.0
    if dist == "bernoulli":
        return (rng.random(size) < 0.1).astype(float), 0.1, float(np.sqrt(0.09))
    if dist == "poisson":
        return rng.poisson(0.5, size).astype(float), 0.5, float(np.sqrt(0.5))
    if dist == "lognormal":
        m, v = np.exp(0.5), (np.exp(1) - 1) * np.exp(1)
        return rng.lognormal(0.0, 1.0, size), float(m), float(np.sqrt(v))
    raise ValueError("dist must be normal, exponential, bernoulli, poisson or lognormal")


def demo_clt(dist: str = "exponential", n_values=(1, 2, 5, 30, 100), R: int = 4000, alpha: float = ALPHA,
             seed: int = 0) -> dict:
    """Central limit theorem: the sampling distribution of the mean of n draws from a skewed population.
    For each n: the mean and SD of the R sample means against mu and sigma / sqrt(n), their skewness
    (which shrinks like 1/sqrt(n)), the Kolmogorov distance to the normal approximation and the coverage
    of the t interval. The student sees that the SD formula holds for every n, while the normal shape,
    and with it the interval's coverage, arrive only gradually, more slowly the more skewed the population."""
    rng = np.random.default_rng(seed)
    rows = []
    for n in n_values:
        x, mu, sd = _population(dist, rng, (R, n))
        means = x.mean(axis=1)
        row = {"n": n, "mean_of_means": means.mean(), "sd_of_means": means.std(ddof=1),
               "sigma/sqrt(n)": sd / np.sqrt(n), "skewness": float(stats.skew(means)),
               "ks_distance_to_normal": float(stats.kstest(means, stats.norm(mu, sd / np.sqrt(n)).cdf).statistic)}
        if n >= 2:
            s = x.std(axis=1, ddof=1)
            half = stats.t.ppf(1 - alpha / 2, n - 1) * s / np.sqrt(n)
            row["t_interval_coverage"] = float(np.mean(np.abs(means - mu) <= half))
        else:
            row["t_interval_coverage"] = np.nan
        rows.append(row)
    tab = pd.DataFrame(rows).set_index("n")
    return {"table": tab, "population": dist,
            "lesson": "The SD of the mean is sigma/sqrt(n) for any n; its distribution turns normal as n grows, faster "
                      "for symmetric populations. Until it does, nominal 95% t intervals for a skewed population cover "
                      "less than 95%, and they approach 95% only slowly."}


def demo_ci_coverage(n_values=(10, 30, 100), R: int = 4000, alpha: float = ALPHA, seed: int = 0) -> dict:
    """What '95% confidence' means: the share of R intervals that contain the truth.
    Scenarios: t interval for a normal mean (exact); z interval with the sample SD (too short in small samples);
    t interval for a lognormal mean (skewness breaks it); Wald and Wilson intervals for a proportion of 0.1."""
    rng = np.random.default_rng(seed)
    z = stats.norm.ppf(1 - alpha / 2)
    rows = []
    for n in n_values:
        tq = stats.t.ppf(1 - alpha / 2, n - 1)
        x = rng.normal(0.0, 1.0, (R, n))
        m, s = x.mean(1), x.std(1, ddof=1)
        rows.append({"scenario": "normal mean, t interval", "n": n,
                     "coverage": np.mean(np.abs(m) <= tq * s / np.sqrt(n)), "mean_width": np.mean(2 * tq * s / np.sqrt(n))})
        rows.append({"scenario": "normal mean, z with sample SD", "n": n,
                     "coverage": np.mean(np.abs(m) <= z * s / np.sqrt(n)), "mean_width": np.mean(2 * z * s / np.sqrt(n))})
        x, mu, _ = _population("lognormal", rng, (R, n))
        m, s = x.mean(1), x.std(1, ddof=1)
        rows.append({"scenario": "lognormal mean, t interval", "n": n,
                     "coverage": np.mean(np.abs(m - mu) <= tq * s / np.sqrt(n)), "mean_width": np.mean(2 * tq * s / np.sqrt(n))})
        k = rng.binomial(n, 0.1, R)
        ph = k / n
        h = z * np.sqrt(ph * (1 - ph) / n)
        rows.append({"scenario": "proportion 0.1, Wald", "n": n,
                     "coverage": np.mean((ph - h <= 0.1) & (0.1 <= ph + h)), "mean_width": np.mean(2 * h)})
        den = 1 + z ** 2 / n
        c = (ph + z ** 2 / (2 * n)) / den
        hw = z * np.sqrt(ph * (1 - ph) / n + z ** 2 / (4 * n ** 2)) / den
        rows.append({"scenario": "proportion 0.1, Wilson", "n": n,
                     "coverage": np.mean((c - hw <= 0.1) & (0.1 <= c + hw)), "mean_width": np.mean(2 * hw)})
    tab = pd.DataFrame(rows).set_index(["scenario", "n"]).sort_index()
    return {"table": tab, "nominal": 1 - alpha,
            "lesson": "A 95% interval is a procedure that covers the truth in 95% of samples when its assumptions hold. "
                      "z with an estimated SD, skewed data in small samples and the Wald interval for a proportion "
                      "near 0 or 1 all cover less; Wilson stays close to 95%."}


def demo_estimator_mse(n: int = 10, R: int = 20000, theta: float = 5.0, seed: int = 0) -> dict:
    """Bias, variance and mean squared error (MSE = variance + bias^2) of competing estimators.
    (1) The variance of a normal sample with divisors n - 1 (unbiased S^2), n (the MLE) and n + 1 (minimum MSE).
    (2) theta of a uniform(0, theta): method of moments 2 * mean, the MLE max(x) (biased down) and the
    unbiased (n + 1)/n * max(x). Unbiased is not best: the MLE of the variance has a smaller MSE than S^2,
    and both estimators based on max(x) beat the method of moments by far."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, (R, n))
    ss = ((x - x.mean(1, keepdims=True)) ** 2).sum(1)

    def summ(est, truth):
        return {"mean": est.mean(), "bias": est.mean() - truth, "variance": est.var(ddof=1),
                "mse": np.mean((est - truth) ** 2)}

    var_tab = pd.DataFrame({f"divide by {lab}": summ(ss / d, 1.0) for lab, d in
                            (("n - 1 (S^2, unbiased)", n - 1), ("n (MLE)", n), ("n + 1", n + 1))}).T
    u = rng.uniform(0.0, theta, (R, n))
    mx = u.max(1)
    uni_tab = pd.DataFrame({"method of moments 2*mean": summ(2 * u.mean(1), theta),
                            "MLE max(x)": summ(mx, theta),
                            "unbiased (n+1)/n*max(x)": summ((n + 1) / n * mx, theta)}).T
    return {"variance_table": var_tab, "uniform_table": uni_tab, "n": n,
            "lesson": "Judge estimators by MSE = variance + bias^2, not by bias alone: S^2 is unbiased but the MLE "
                      "(divide by n) has a lower MSE; for a uniform(0, theta) the MLE max(x) is biased yet far more "
                      "precise than the method of moments, and a small correction removes its bias."}


def demo_multiple_testing(m: int = 20, n: int = 50, R: int = 1000, share_true: float = 0.25, effect: float = 0.5,
                          alpha: float = ALPHA, seed: int = 0) -> dict:
    """m two-sample t tests per study, repeated R times, first with every null true and then with a share of
    real effects of `effect` SDs. For raw p-values, Bonferroni, Holm and Benjamini-Hochberg: the family-wise
    error rate (chance of at least one false rejection), the false discovery rate and the power."""
    from inference import adjust_pvalues

    rng = np.random.default_rng(seed)
    rows = []
    for label, share in (("all nulls true", 0.0), (f"{share_true:.0%} real effects", share_true)):
        k_true = int(round(share * m))
        is_true = np.arange(m) < k_true
        stats_acc = {meth: {"any_false": 0, "fdp": [], "power": []} for meth in ("raw", "bonferroni", "holm", "bh")}
        for _ in range(R):
            a = rng.normal(0.0, 1.0, (m, n))
            b = rng.normal(0.0, 1.0, (m, n)) + effect * is_true[:, None]
            p = stats.ttest_ind(b, a, axis=1).pvalue
            for meth in stats_acc:
                rej = p < alpha if meth == "raw" else adjust_pvalues(p, meth, alpha)["reject"].to_numpy()
                false = rej & ~is_true
                acc = stats_acc[meth]
                acc["any_false"] += bool(false.any())
                acc["fdp"].append(false.sum() / max(rej.sum(), 1))
                if k_true:
                    acc["power"].append((rej & is_true).sum() / k_true)
        for meth, acc in stats_acc.items():
            rows.append({"scenario": label, "method": meth, "fwer": acc["any_false"] / R,
                         "fdr": float(np.mean(acc["fdp"])),
                         "power": float(np.mean(acc["power"])) if acc["power"] else np.nan})
    tab = pd.DataFrame(rows).set_index(["scenario", "method"])
    return {"table": tab, "m": m,
            "lesson": f"With {m} true nulls tested at {alpha:.0%} each, the chance of at least one 'significant' result "
                      f"is about 1 - (1 - {alpha:g})^{m} = {1 - (1 - alpha) ** m:.0%}. Holm keeps that chance at most "
                      f"{alpha:.0%} (with more power than Bonferroni); Benjamini-Hochberg keeps the expected share of false "
                      f"discoveries at most {alpha:.0%} and keeps more power."}


# =============================================================================
# Classical regression (v2.4.2)
# =============================================================================
def demo_simultaneity(R: int = 1000, n: int = 200, b_demand: float = 1.0, d_supply: float = 0.5,
                      c_income: float = 1.0, sd_d: float = 1.0, sd_s: float = 1.0, seed: int = 0) -> dict:
    """Supply and demand: demand q = −b·p + c·income + u_d, supply q = d·p + u_s, prices clear the market.
    OLS of q on p mixes the two curves (simultaneity bias). Income shifts demand only, so it traces out the
    supply curve: indirect least squares (π_q/π_p from the two reduced forms) and 2SLS with income as the
    instrument recover d and are the same number when exactly identified. The demand slope is not identified
    without a variable that shifts supply only."""
    rng = np.random.default_rng(seed)
    rows, gap = [], []
    for _ in range(R):
        inc = rng.normal(size=n)
        ud, us = rng.normal(0, sd_d, n), rng.normal(0, sd_s, n)
        p = (c_income * inc + ud - us) / (b_demand + d_supply)
        q = d_supply * p + us
        X = np.column_stack([np.ones(n), p])
        Z = np.column_stack([np.ones(n), inc])
        b_ols = np.linalg.lstsq(X, q, rcond=None)[0][1]
        pi_q = np.linalg.lstsq(Z, q, rcond=None)[0][1]
        pi_p = np.linalg.lstsq(Z, p, rcond=None)[0][1]
        ils = pi_q / pi_p
        Pz = Z @ np.linalg.solve(Z.T @ Z, Z.T @ X)
        tsls = np.linalg.solve(Pz.T @ X, Pz.T @ q)[1]
        rows.append((b_ols, ils, tsls))
        gap.append(abs(ils - tsls))
    a = np.array(rows)
    tab = pd.DataFrame({"median": np.median(a, 0), "mean": a.mean(0), "sd": a.std(0, ddof=1)},
                       index=["OLS of q on p", "indirect least squares", "2SLS (income)"])
    tab["truth (supply slope)"] = d_supply
    tab["median_bias"] = tab["median"] - d_supply
    var_p_d, var_p_s = sd_d ** 2, sd_s ** 2
    plim = (d_supply * (c_income ** 2 + var_p_d) - b_demand * var_p_s) / (c_income ** 2 + var_p_d + var_p_s)
    return {"table": tab, "ols_probability_limit": plim, "max_abs_ils_minus_2sls": float(max(gap)),
            "lesson": "Price and quantity are set together, so OLS estimates neither curve: its limit is a weighted "
                      "mix of the supply and demand slopes. A demand shifter excluded from supply (income) identifies "
                      "the supply slope; ILS and 2SLS give the same estimate when exactly identified."}


def demo_lpm_bounds(slopes=(0.5, 1.0, 2.0, 4.0), R: int = 500, n: int = 500, intercept: float = 0.0,
                    seed: int = 0) -> dict:
    """True model: logit P(y=1|x) = Λ(a + b·x), x ~ N(0, 1). For each slope b: the share of LPM fitted values
    outside [0, 1], the LPM slope against the true average marginal effect E[b·Λ(1 − Λ)], and the logit AME.
    The student sees that the LPM slope tracks the average effect well, while its predictions leave [0, 1]
    as the true curve gets steeper: fine for average effects, poor for individual probabilities."""
    import statsmodels.api as sm
    rng = np.random.default_rng(seed)
    rows = []
    for b in slopes:
        outside, lpm, true_ame, logit_ame = [], [], [], []
        for _ in range(R):
            x = rng.normal(size=n)
            pr = 1 / (1 + np.exp(-(intercept + b * x)))
            y = (rng.uniform(size=n) < pr).astype(float)
            X = np.column_stack([np.ones(n), x])
            coef = np.linalg.lstsq(X, y, rcond=None)[0]
            fit = X @ coef
            outside.append(np.mean((fit < 0) | (fit > 1)))
            lpm.append(coef[1])
            true_ame.append(np.mean(b * pr * (1 - pr)))
            try:
                m = sm.Logit(y, X).fit(disp=False)
                ph = m.predict()
                logit_ame.append(np.mean(m.params[1] * ph * (1 - ph)))
            except Exception:                    # perfect separation in a draw: skip it
                logit_ame.append(np.nan)
        rows.append({"logit slope b": b, "LPM fitted outside [0,1]": np.mean(outside),
                     "true AME": np.mean(true_ame), "LPM slope (mean)": np.mean(lpm),
                     "logit AME (mean)": np.nanmean(logit_ame)})
    tab = pd.DataFrame(rows).set_index("logit slope b")
    return {"table": tab,
            "lesson": "The LPM slope is close to the average marginal effect even when the truth is a logit; what "
                      "breaks is the fitted probability, which leaves [0, 1] more often as the effect grows. Use the "
                      "LPM (with robust SEs) for average effects and logit/probit for individual probabilities."}


def demo_multicollinearity(rhos=(0.0, 0.5, 0.9, 0.99), R: int = 1000, n: int = 100, beta1: float = 1.0,
                           beta2: float = 1.0, sd_u: float = 3.0, alpha: float = ALPHA, seed: int = 0) -> dict:
    """y = β1·x1 + β2·x2 + u with corr(x1, x2) = ρ. OLS stays unbiased; the SE of β̂1 grows like √VIF,
    VIF = 1/(1 − ρ²); individual t-tests stop rejecting while R² does not fall and the joint F still rejects. The remedy is
    more data or a different question (the combined effect β1 + β2 is precise), not dropping a variable,
    which brings omitted-variable bias."""
    rng = np.random.default_rng(seed)
    rows = []
    se0 = None
    for rho in rhos:
        b1, se1, rej1, rejF, r2, se_sum = [], [], [], [], [], []
        for _ in range(R):
            x1 = rng.normal(size=n)
            x2 = rho * x1 + np.sqrt(1 - rho ** 2) * rng.normal(size=n)
            y = beta1 * x1 + beta2 * x2 + rng.normal(0, sd_u, n)
            X = np.column_stack([np.ones(n), x1, x2])
            b, V, _, e = _ols(y, X)
            b1.append(b[1]); se1.append(np.sqrt(V[1, 1]))
            rej1.append(abs(b[1] / np.sqrt(V[1, 1])) > stats.t.ppf(1 - alpha / 2, n - 3))
            Rm = np.array([[0, 1, 0], [0, 0, 1]], float)
            F = (Rm @ b) @ np.linalg.solve(Rm @ V @ Rm.T, Rm @ b) / 2
            rejF.append(F > stats.f.ppf(1 - alpha, 2, n - 3))
            r2.append(1 - e @ e / np.sum((y - y.mean()) ** 2))
            se_sum.append(np.sqrt(V[1, 1] + V[2, 2] + 2 * V[1, 2]))
        row = {"rho": rho, "VIF": 1 / (1 - rho ** 2), "mean b1": np.mean(b1), "mean se(b1)": np.mean(se1),
               "reject b1 = 0": np.mean(rej1), "reject b1 = b2 = 0 (F)": np.mean(rejF), "mean R2": np.mean(r2),
               "mean se(b1 + b2)": np.mean(se_sum)}
        if se0 is None:
            se0 = row["mean se(b1)"]
        row["se ratio vs rho=first"] = row["mean se(b1)"] / se0
        rows.append(row)
    tab = pd.DataFrame(rows).set_index("rho")
    return {"table": tab, "truth": beta1,
            "lesson": "Correlated regressors do not bias OLS; they make the separate effects imprecise (SE × √VIF), "
                      "so individual t-tests fail while the joint F rejects. The sum β1 + β2 stays precise. Dropping "
                      "one regressor trades imprecision for omitted-variable bias."}
