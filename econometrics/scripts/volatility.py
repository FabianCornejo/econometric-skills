"""Volatility: stylized facts of returns, GARCH / GJR / EGARCH with diagnostics, regressors (dummies) in the
variance, and the Engle-Ng sign-bias test (v2.4.7). Reference: volatility.md.

Use as a module:
    import sys; sys.path.insert(0, "<skill folder>/scripts")
    from volatility import *

Functions
    stylized_facts(r)                    mean, SD, skewness, excess kurtosis and Jarque-Bera; Ljung-Box on returns
                                         and on squared returns; ARCH-LM; leverage correlations corr(r_t, r²_{t+j})
    sign_bias_test(z)                    Engle-Ng (1993) sign and size bias tests on standardized residuals
    garch_fit(r, vol)                    GARCH, GJR (threshold) or EGARCH by arch, with persistence, half-life,
                                         unconditional variance, Ljung-Box on z and z², sign bias, AIC/BIC; with
                                         variance_dummies, a GARCH/GJR whose variance equation has regressors
                                         (day of week, holidays, an event), by its own maximum likelihood with
                                         robust (Bollerslev-Wooldridge) SEs

Names matter: GJR-GARCH (Glosten-Jagannathan-Runkle) is the threshold GARCH in the variance (TGARCH in
Zakoian's form models the SD); EGARCH (Nelson) models the log-variance and is neither. Returns in % help the
optimiser. Every number comes from the data passed in.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import optimize, special, stats
import _console  # noqa: F401  (Windows-safe console output, see _console.py)

ALPHA = 0.05


def _fmt(v):
    return f"{v:.4f}" if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v)


def _returns(r) -> pd.Series:
    s = pd.Series(r).astype(float).dropna()
    if len(s) < 100:
        raise ValueError("too few returns for volatility models (need at least 100; ideally > 500 daily)")
    return s


# =============================================================================
# Stylized facts
# =============================================================================
def stylized_facts(r, lags: int = 10, leverage_lags: int = 5, alpha: float = ALPHA, show: bool = True) -> dict:
    """The facts to check before modelling the variance of returns.

    Fat tails (excess kurtosis > 0, Jarque-Bera); little autocorrelation in returns (Ljung-Box on r) but a
    lot in squared returns (Ljung-Box on r², ARCH-LM): volatility clustering; leverage: negative corr(r_t,
    r²_{t+j}) means falls are followed by higher volatility. If r² shows no autocorrelation and ARCH-LM does
    not reject, a constant variance is enough."""
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    s = _returns(r)
    x = s.to_numpy()
    n = len(x)
    jb = stats.jarque_bera(x)
    lb_r = acorr_ljungbox(x, lags=[lags], return_df=True).iloc[0]
    lb_r2 = acorr_ljungbox((x - x.mean()) ** 2, lags=[lags], return_df=True).iloc[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lm, lm_p, _, _ = het_arch(x - x.mean(), nlags=lags)
    lev = {}
    for j in range(1, leverage_lags + 1):
        lev[j] = float(np.corrcoef(x[:-j], x[j:] ** 2)[0, 1])
    se_lev = 1 / np.sqrt(n)
    facts = pd.DataFrame([
        {"fact": "mean (per period)", "value": float(x.mean()), "p": float(2 * stats.t.sf(abs(x.mean() / (x.std(ddof=1) / np.sqrt(n))), n - 1)),
         "reading": "mean return different from zero" if 2 * stats.t.sf(abs(x.mean() / (x.std(ddof=1) / np.sqrt(n))), n - 1) < alpha else "mean not different from zero"},
        {"fact": "SD (per period)", "value": float(x.std(ddof=1)), "p": np.nan, "reading": ""},
        {"fact": "skewness", "value": float(stats.skew(x)), "p": np.nan, "reading": ""},
        {"fact": "excess kurtosis (normal = 0)", "value": float(stats.kurtosis(x)), "p": float(jb.pvalue),
         "reading": "fat tails (Jarque-Bera rejects normality)" if jb.pvalue < alpha else "tails compatible with the normal"},
        {"fact": f"Ljung-Box on r ({lags} lags)", "value": float(lb_r["lb_stat"]), "p": float(lb_r["lb_pvalue"]),
         "reading": "autocorrelation in returns (model the mean: AR terms)" if lb_r["lb_pvalue"] < alpha else "no autocorrelation in returns"},
        {"fact": f"Ljung-Box on r² ({lags} lags)", "value": float(lb_r2["lb_stat"]), "p": float(lb_r2["lb_pvalue"]),
         "reading": "volatility clustering" if lb_r2["lb_pvalue"] < alpha else "no clustering"},
        {"fact": f"ARCH-LM ({lags} lags)", "value": float(lm), "p": float(lm_p),
         "reading": "ARCH effects: model the variance" if lm_p < alpha else "no ARCH effects: a constant variance is enough"},
    ])
    lev_tab = pd.DataFrame({"corr(r_t, r²_t+j)": lev, "approx 95% band": 1.96 * se_lev})
    lev_tab.index.name = "j"
    leverage = any(v < -1.96 * se_lev for v in lev.values())
    out = {"facts": facts, "leverage": lev_tab, "leverage_effect": leverage, "n": n,
           "clustering": bool(lb_r2["lb_pvalue"] < alpha or lm_p < alpha)}
    if show:
        print(f"\nStylized facts (n = {n})")
        print(facts.to_string(index=False, float_format=_fmt))
        print("\nLeverage (negative = falls raise later volatility):")
        print(lev_tab.to_string(float_format=_fmt))
        print("→ leverage effect" if leverage else "→ no clear leverage effect")
    return out


# =============================================================================
# Engle-Ng sign bias
# =============================================================================
def sign_bias_test(z, show: bool = True) -> dict:
    """Engle and Ng (1993): regress z_t² on a constant, S⁻_{t-1} (1 if z_{t-1} < 0), S⁻_{t-1}·z_{t-1} and
    S⁺_{t-1}·z_{t-1} (standardized residuals in the size terms too; Engle-Ng write them with ε_{t-1}, and both
    forms are used), where z are the standardized residuals of a volatility model (or demeaned returns
    divided by their SD before any model). Significant terms mean the model misses an asymmetry: sign bias
    (negative shocks matter differently), negative or positive size bias (large shocks of one sign matter
    more). Joint test: n·R² ~ χ²(3)."""
    import statsmodels.api as sm
    zz = pd.Series(z).astype(float).dropna().to_numpy()
    y = zz[1:] ** 2
    zl = zz[:-1]
    neg = (zl < 0).astype(float)
    X = sm.add_constant(np.column_stack([neg, neg * zl, (1 - neg) * zl]))
    fit = sm.OLS(y, X).fit()
    names = ["sign bias", "negative size bias", "positive size bias"]
    tab = pd.DataFrame({"coef": fit.params[1:], "t": fit.tvalues[1:], "p": fit.pvalues[1:]}, index=names)
    joint = float(len(y) * fit.rsquared)
    p_joint = float(stats.chi2.sf(joint, 3))
    out = {"table": tab, "joint_stat": joint, "joint_p": p_joint,
           "verdict": ("asymmetry left in the residuals: try GJR or EGARCH" if p_joint < ALPHA or (tab["p"] < ALPHA).any()
                       else "no asymmetry left in the residuals")}
    if show:
        print("\nEngle-Ng sign-bias tests on standardized residuals")
        print(tab.to_string(float_format=_fmt))
        print(f"Joint n·R² = {joint:.3f}, p = {p_joint:.4f} → {out['verdict']}")
    return out


# =============================================================================
# GARCH with regressors in the variance (own likelihood)
# =============================================================================
def _backcast(e: np.ndarray) -> float:
    tau = min(75, len(e))
    w = 0.94 ** np.arange(tau)
    return float(np.sum(e[:tau] ** 2 * w / w.sum()))


def _sigma2(theta, e, D, gjr, bc):
    k = D.shape[1]
    omega, alpha, gamma, beta = theta[1], theta[2], (theta[3] if gjr else 0.0), theta[3 + gjr]
    delta = theta[4 + gjr:4 + gjr + k]
    n = len(e)
    s2 = np.empty(n)
    exo = D @ delta if k else np.zeros(n)
    prev_e2, prev_neg, prev_s2 = bc, 0.5 * bc, bc
    for t in range(n):
        s2[t] = omega + exo[t] + alpha * prev_e2 + gamma * prev_neg + beta * prev_s2
        prev_e2 = e[t] ** 2
        prev_neg = prev_e2 * (e[t] < 0)
        prev_s2 = s2[t]
    return s2


def _ll_obs(theta, r, D, gjr, dist):
    mu = theta[0]
    e = r - mu
    s2 = _sigma2(theta, e, D, gjr, _backcast(e))
    if np.any(s2 <= 0) or not np.all(np.isfinite(s2)):
        return np.full(len(r), -1e6)
    if dist == "t":
        nu = theta[-1]
        return (special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
                - 0.5 * np.log(s2) - (nu + 1) / 2 * np.log(1 + e ** 2 / (s2 * (nu - 2))))
    return -0.5 * (np.log(2 * np.pi) + np.log(s2) + e ** 2 / s2)


def _garch_mle(r: np.ndarray, D: np.ndarray, gjr: bool, dist: str, return_cov: bool = False):
    from statsmodels.tools.numdiff import approx_fprime, approx_hess
    k = D.shape[1]
    v = r.var()
    names = ["mu", "omega", "alpha[1]"] + (["gamma[1]"] if gjr else []) + ["beta[1]"]
    theta0 = [r.mean(), 0.05 * v, 0.05] + ([0.05] if gjr else []) + [0.88] + [0.0] * k + ([8.0] if dist == "t" else [])
    bounds = ([(None, None), (1e-8 * v, 10 * v), (0, 1)] + ([(-1, 1)] if gjr else []) + [(0, 1)]
              + [(None, None)] * k + ([(2.05, 200)] if dist == "t" else []))

    lo = np.array([b[0] if b[0] is not None else -np.inf for b in bounds])
    hi = np.array([b[1] if b[1] is not None else np.inf for b in bounds])

    def nll(th):
        if np.any(th < lo) or np.any(th > hi):          # Nelder-Mead ignores bounds
            return 1e10
        pers = th[2] + (th[3] / 2 if gjr else 0) + th[3 + gjr]
        if pers >= 0.9999 or (gjr and th[2] + th[3] < 0):
            return 1e10
        val = -np.sum(_ll_obs(th, r, D, gjr, dist))
        return val if np.isfinite(val) else 1e10
    best = None
    for start in (theta0, [r.mean(), 0.1 * v, 0.1] + ([0.0] if gjr else []) + [0.8] + [0.0] * k + ([6.0] if dist == "t" else [])):
        fit = optimize.minimize(nll, np.array(start, float), method="L-BFGS-B", bounds=bounds)
        fit2 = optimize.minimize(nll, fit.x, method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-10})
        cand = fit2 if fit2.fun < fit.fun else fit
        if best is None or cand.fun < best.fun:
            best = cand
    th = best.x
    H = approx_hess(th, lambda t: np.sum(_ll_obs(t, r, D, gjr, dist)))
    S = approx_fprime(th, lambda t: _ll_obs(t, r, D, gjr, dist))
    try:
        Hi = np.linalg.inv(-H)
        cov = Hi @ (S.T @ S) @ Hi
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        cov = np.full((len(th), len(th)), np.nan)
        se = np.full(len(th), np.nan)
    e = r - th[0]
    s2 = _sigma2(th, e, D, gjr, _backcast(e))
    out = (th, se, -best.fun, s2, names, bool(best.success) or best.nit > 0)
    return out + (cov,) if return_cov else out


_SKEWT = ("skewt", "skewstudent")


def _neg_weight(dist: str, params: pd.Series) -> float:
    """κ = E[z²·1(z < 0)] of the standardized error distribution: the share of the GJR term γ·ε²·1(ε < 0) that
    enters the persistence α + κγ + β. κ = 1/2 for any symmetric distribution (normal, t, GED); for the skewed t
    of Hansen (arch 'skewt') it depends on η and λ (v2.4.7.2: before, 1/2 was used for every distribution)."""
    if dist not in _SKEWT:
        return 0.5
    try:
        from arch.univariate.distribution import SkewStudent
        k = float(SkewStudent().partial_moment(2, 0.0, [float(params["eta"]), float(params["lambda"])]))
    except Exception:  # noqa: BLE001 — reported by garch_fit, which then falls back to 1/2
        return np.nan
    return k if 0 < k < 1 else np.nan


def _persistence(vol: str, dist: str, params: pd.Series, strict: bool = False) -> float:
    """strict=True (used for the delta-method gradient) raises instead of falling back to κ = 1/2, so a step
    outside the distribution's support is detected rather than mixed with the symmetric formula."""
    a = params.get("alpha[1]", np.nan)
    b = params.get("beta[1]", np.nan)
    if vol == "egarch":
        return float(b)
    if vol == "gjr":
        k = _neg_weight(dist, params)
        if not np.isfinite(k):
            if strict:
                raise ValueError("κ not computable at these parameters")
            k = 0.5
        return float(a + b + k * params.get("gamma[1]", 0.0))
    return float(a + b)


def _persistence_delta(vol: str, dist: str, params: pd.Series, cov, level: float):
    """SE and CI of the persistence by the delta method: numerical gradient of _persistence with respect to every
    parameter (with 'skewt' the persistence also moves with η and λ), sandwiched with the parameter covariance."""
    cov = np.asarray(cov, float)
    if cov.shape != (len(params), len(params)) or not np.all(np.isfinite(cov)):
        return np.nan, (np.nan, np.nan)
    base = params.astype(float)
    grad = np.zeros(len(base))
    for i, name in enumerate(base.index):
        if vol == "egarch" and name != "beta[1]":
            continue
        if name not in ("alpha[1]", "beta[1]", "gamma[1]", "eta", "lambda"):
            continue
        h = 1e-5 * max(1.0, abs(base.iloc[i]))
        up, dn = base.copy(), base.copy()
        up.iloc[i] += h
        dn.iloc[i] -= h
        vals = {}
        for key, par in (("up", up), ("dn", dn), ("c", base)):
            try:
                vals[key] = _persistence(vol, dist, par, strict=True)
            except Exception:  # noqa: BLE001 — a step outside the support (e.g. λ near ±1): handled below
                vals[key] = np.nan
        if np.isfinite(vals["up"]) and np.isfinite(vals["dn"]):
            grad[i] = (vals["up"] - vals["dn"]) / (2 * h)
        elif np.isfinite(vals["c"]) and np.isfinite(vals["dn"]):     # one-sided, towards the interior
            grad[i] = (vals["c"] - vals["dn"]) / h
        elif np.isfinite(vals["c"]) and np.isfinite(vals["up"]):
            grad[i] = (vals["up"] - vals["c"]) / h
        else:
            return np.nan, (np.nan, np.nan)
    var = float(grad @ cov @ grad)
    if not np.isfinite(var) or var < 0:
        return np.nan, (np.nan, np.nan)
    se = float(np.sqrt(var))
    p = _persistence(vol, dist, base)
    zq = float(stats.norm.ppf(0.5 + level / 2))
    return se, (p - zq * se, p + zq * se)


def _half_life(p) -> float:
    """Half-life of a variance shock, ln 0.5 / ln(p), for a persistence p. Increasing in p on (0, 1), so the limits of a
    persistence CI map to the limits of the half-life CI in the same order (v2.5.0). p >= 1: infinite (integrated);
    p <= 0: 0 (the shock is gone after one period; a negative EGARCH β oscillates); NaN stays NaN."""
    p = float(p)
    if not np.isfinite(p):
        return np.inf if p == np.inf else np.nan
    if p >= 1:
        return np.inf
    if p <= 0:
        return 0.0
    return float(np.log(0.5) / np.log(p))


def garch_fit(r, vol: str = "garch", dist: str = "t", mean: str = "Constant", variance_dummies=None,
              lags: int = 10, alpha: float = ALPHA, show: bool = True) -> dict:
    """GARCH(1,1), GJR-GARCH(1,1) or EGARCH(1,1,1) with the diagnostics that decide whether it is usable.

    vol: 'garch', 'gjr' (threshold: γ adds to α after negative shocks) or 'egarch' (log-variance; γ < 0 is
    the leverage effect). Persistence: α + β (GARCH), α + κγ + β (GJR, κ = E[z²·1(z<0)]: 1/2 for symmetric errors,
    from the fitted η and λ for 'skewt'), β (EGARCH), with a delta-method SE and a (1 - alpha) CI
    (persistence_se, persistence_ci; v2.4.7.2);
    half-life of a variance shock ln 0.5 / ln(persistence), with half_life_ci from the limits of persistence_ci (same
    order: a higher persistence gives a longer half-life; infinite upper limit when the persistence CI reaches 1;
    v2.5.0); for EGARCH it is the half-life of a shock to the log-variance, not comparable with a GARCH or GJR one;
    unconditional variance ω/(1 - persistence) for GARCH/GJR. Diagnostics on the standardized residuals z: Ljung-Box on z (mean) and z² (variance), the
    Engle-Ng sign-bias tests. For vol='garch' the asymmetry is tested by the LR of the GJR against the
    symmetric GARCH (γ = 0, χ²(1)): the sign-bias tests are a complement with little power under fat tails, and
    a sign-bias p-value above 0.05 does not show symmetry. With variance_dummies (a DataFrame of regressors aligned with r, e.g. a Monday or
    post-holiday dummy), the GARCH/GJR is estimated by its own likelihood with the regressors in the variance
    equation and Bollerslev-Wooldridge (QML sandwich) SEs; arch has no regressors in the variance."""
    from statsmodels.stats.diagnostic import acorr_ljungbox
    s = _returns(r)
    if vol not in ("garch", "gjr", "egarch"):
        raise ValueError("vol must be 'garch', 'gjr' or 'egarch'")
    if dist not in ("normal", "t") and variance_dummies is not None:
        raise ValueError("with variance_dummies, dist must be 'normal' or 't'")
    warn = []
    if s.abs().median() < 0.05:
        warn.append("returns look like decimals (median |r| < 0.05): multiply by 100 so the optimiser works in %")
    if variance_dummies is None:
        from arch import arch_model
        spec = {"garch": dict(vol="GARCH", p=1, o=0, q=1), "gjr": dict(vol="GARCH", p=1, o=1, q=1),
                "egarch": dict(vol="EGARCH", p=1, o=1, q=1)}[vol]
        am = arch_model(s, mean=mean, dist=dist, **spec)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = am.fit(disp="off", show_warning=False)   # arch's own filter would override "ignore"
        params, se = res.params, res.std_err
        z = res.std_resid.dropna().to_numpy()
        llf, k_par = float(res.loglikelihood), len(params)
        conv = int(res.convergence_flag) == 0
        pers = _persistence(vol, dist, params)
        param_cov = res.param_cov
        omega = params.get("omega", np.nan)
        cond_var = res.conditional_volatility ** 2
        dummies = None
    else:
        if vol == "egarch":
            raise ValueError("variance_dummies are supported for 'garch' and 'gjr'")
        Dd = pd.DataFrame(variance_dummies).astype(float).reindex(s.index)
        if Dd.isna().any().any():
            raise ValueError("variance_dummies must cover every date of r")
        if mean != "Constant":
            raise ValueError("with variance_dummies the mean is a constant (mean='Constant')")
        th, se_, llf, s2, names, conv_flag, param_cov = _garch_mle(s.to_numpy(), Dd.to_numpy(), vol == "gjr", dist,
                                                                   return_cov=True)
        names = names + [f"delta[{c}]" for c in Dd.columns] + (["nu"] if dist == "t" else [])
        params, se = pd.Series(th, index=names), pd.Series(se_, index=names)
        z = (s.to_numpy() - th[0]) / np.sqrt(s2)
        k_par, conv = len(th), bool(conv_flag)
        pers = _persistence(vol, dist, params)   # normal or t here: κ = 1/2
        omega = params["omega"] + float(Dd.mean().to_numpy() @ params[[f"delta[{c}]" for c in Dd.columns]].to_numpy())
        cond_var = pd.Series(s2, index=s.index)
        dummies = pd.DataFrame({"coef": params[[f"delta[{c}]" for c in Dd.columns]],
                                "se": se[[f"delta[{c}]" for c in Dd.columns]]})
        dummies["z"] = dummies["coef"] / dummies["se"]
        dummies["p"] = 2 * stats.norm.sf(np.abs(dummies["z"]))
    n = len(s)
    kappa = _neg_weight(dist, params) if vol == "gjr" else np.nan
    pers_se, pers_ci = _persistence_delta(vol, dist, params, param_cov, 1 - alpha)
    half_life = _half_life(pers)
    half_life_ci = (_half_life(pers_ci[0]), _half_life(pers_ci[1]))
    uncond = float(omega / (1 - pers)) if vol != "egarch" and pers < 1 else np.nan
    lb_z = acorr_ljungbox(z, lags=[lags], return_df=True).iloc[0]
    lb_z2 = acorr_ljungbox(z ** 2, lags=[lags], return_df=True).iloc[0]
    sb = sign_bias_test(z, show=False)
    asym = None
    if vol == "garch" and variance_dummies is None:
        # the asymmetry test that has power: LR of the GJR against this GARCH (γ = 0). The sign-bias tests lose
        # power with fat tails (v2.4.7.1: Ibovespa 2010-2026, sign bias p = 0.77, LR p = 2e-8)
        from arch import arch_model
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rg = arch_model(s, mean=mean, dist=dist, vol="GARCH", p=1, o=1, q=1).fit(disp="off",
                                                                                       show_warning=False)
            ok_fit = rg.convergence_flag == 0 and float(rg.loglikelihood) >= llf - 1e-4
        except Exception:  # noqa: BLE001 — the asymmetry check must never break the main fit
            rg, ok_fit = None, False
        if ok_fit:
            lr = max(0.0, 2 * (float(rg.loglikelihood) - llf))
            p_lr = float(stats.chi2.sf(lr, 1))
            asym = {"lr": lr, "p": p_lr, "gamma": float(rg.params["gamma[1]"]),
                    "gamma_se": float(rg.std_err["gamma[1]"]), "gjr_bic": float(rg.bic)}
        else:
            asym = {"lr": np.nan, "p": np.nan, "gamma": np.nan, "gamma_se": np.nan, "gjr_bic": np.nan}
            warn.append("asymmetry LR not available: the GJR fit did not converge (rescale the returns, e.g. to "
                        "percent); rely on the sign-bias tests and fit 'gjr' directly")
        if ok_fit and p_lr < alpha:
            warn.append(f"asymmetry omitted: the GJR improves on this symmetric GARCH (LR = {lr:.2f}, p = {p_lr:.2g}, "
                        f"γ = {asym['gamma']:.4f}); use 'gjr' or 'egarch'"
                        + (f" — the sign-bias tests miss it here (joint p = {sb['joint_p']:.3f}): they have little "
                           "power with fat tails" if sb["joint_p"] >= alpha else ""))
    if vol == "gjr" and dist in _SKEWT and not np.isfinite(kappa):
        warn.append("E[z²·1(z<0)] of the fitted skewed t could not be computed: the persistence uses γ/2, which is "
                    "exact only for symmetric errors")
    if pers < 1 and np.isfinite(pers_ci[1]) and pers_ci[1] >= 1:
        warn.append(f"the {100 * (1 - alpha):g}% CI of the persistence reaches 1 ({pers_ci[0]:.4f} to {pers_ci[1]:.4f}): an "
                    "integrated variance (IGARCH) is not ruled out, and the half-life is imprecise")
    if pers >= 1:
        warn.append(f"persistence {pers:.4f} ≥ 1: integrated (IGARCH-like); unconditional variance undefined. Check "
                    "for breaks, which inflate persistence")
    elif pers > 0.99:
        warn.append(f"persistence {pers:.4f}: shocks last for months (half-life {half_life:.0f} periods); breaks "
                    "or regime changes can inflate it: estimate on subsamples")
    if lb_z2["lb_pvalue"] < alpha:
        warn.append("Ljung-Box rejects on z²: volatility dynamics not captured")
    if lb_z["lb_pvalue"] < alpha:
        warn.append("Ljung-Box rejects on z: autocorrelation left in the mean (add AR terms)")
    if sb["joint_p"] < alpha and vol == "garch" and (asym is None or not asym["p"] < alpha):
        warn.append("sign-bias tests reject: the symmetric GARCH misses an asymmetry; try 'gjr' or 'egarch'")
    if not conv:
        warn.append("the optimiser did not report convergence")
    tab = pd.DataFrame({"coef": params, "se": se})
    tab["z"] = tab["coef"] / tab["se"]
    tab["p"] = 2 * stats.norm.sf(np.abs(tab["z"]))
    out = {"params": tab, "persistence": float(pers), "persistence_se": pers_se,
           "persistence_ci": (float(pers_ci[0]), float(pers_ci[1])), "neg_weight": kappa, "half_life": half_life, "half_life_ci": half_life_ci,
           "unconditional_variance": uncond,
           "llf": llf, "aic": -2 * llf + 2 * k_par, "bic": -2 * llf + k_par * np.log(n), "n": n,
           "ljung_box_z": {"stat": float(lb_z["lb_stat"]), "p": float(lb_z["lb_pvalue"])},
           "ljung_box_z2": {"stat": float(lb_z2["lb_stat"]), "p": float(lb_z2["lb_pvalue"])},
           "sign_bias": sb, "std_resid": pd.Series(z, index=s.index[-len(z):]), "conditional_variance": cond_var,
           "variance_dummies": dummies, "asymmetry_lr": asym, "warnings": warn,
           "method": {"garch": "GARCH(1,1)", "gjr": "GJR-GARCH(1,1) (threshold in the variance)",
                      "egarch": "EGARCH(1,1) (log-variance; not a threshold GARCH)"}[vol]
           + f", {dist} errors" + (", regressors in the variance by own QML (Bollerslev-Wooldridge SEs)"
                                   if variance_dummies is not None else ", arch")}
    if show:
        print(f"\n{out['method']} (n = {n})")
        print(tab.to_string(float_format=_fmt))
        print(f"Persistence {pers:.4f} (SE {pers_se:.4f}; {100 * (1 - alpha):g}% CI {pers_ci[0]:.4f} to {pers_ci[1]:.4f})"
              + (f" [κ = E[z²·1(z<0)] = {kappa:.4f}]" if vol == "gjr" and dist in _SKEWT else "")
              + f"; half-life {half_life:.1f} periods ({100 * (1 - alpha):g}% CI {half_life_ci[0]:.1f} to "
              + f"{half_life_ci[1]:.1f})"
              + (" [log-variance: not comparable with a GARCH/GJR half-life]" if vol == "egarch" else "")
              + (f"; unconditional variance {uncond:.4f}" if np.isfinite(uncond) else ""))
        print(f"Ljung-Box({lags}) z: p = {lb_z['lb_pvalue']:.4f}; z²: p = {lb_z2['lb_pvalue']:.4f}; sign bias joint "
              f"p = {sb['joint_p']:.4f}; AIC {out['aic']:.2f}, BIC {out['bic']:.2f}")
        if asym is not None:
            print(f"Asymmetry (LR, GJR vs this GARCH): LR = {asym['lr']:.2f}, p = {asym['p']:.4g}; γ = {asym['gamma']:.4f} "
                  f"(SE {asym['gamma_se']:.4f})")
        if dummies is not None:
            print("Regressors in the variance:\n" + dummies.to_string(float_format=_fmt))
        for w in warn:
            print("Warning:", w)
    return out
