"""
General-purpose estimation engine: write a likelihood or a regression function, get estimates,
standard errors and tests with the same care as the packaged estimators.

Maximum likelihood
    mle(loglik, theta0, args, ...)            per-observation log-likelihood -> estimates, SEs
                                              (Hessian, OPG, sandwich or clustered sandwich),
                                              Newton-Raphson with step halving or BFGS, AIC/BIC
    trinity_test(loglik, theta0, R, r, ...)   LR, Wald and LM tests of linear restrictions R theta = r

Least squares and functions of parameters
    nls(f, theta0, y, args, ...)              nonlinear least squares with classical or HC1 SEs
    delta_method(g, theta, cov, ...)          SE and CI of any (vector) function of the parameters

Resampling
    bootstrap(stat_fn, df, B, cluster, ...)   pairs or cluster (block) bootstrap: SE, percentile and normal CIs
    wild_cluster_bootstrap(df, formula, coef, cluster, ...)
                                              restricted wild cluster bootstrap-t (Cameron, Gelbach &
                                              Miller 2008; Roodman et al. 2019) for few clusters, with a
                                              test-inversion confidence interval

Output contract (shared with the other modules): a single estimand is a dict with
estimand, estimate, se, ci_low, ci_high, p, n (+ method, warnings); several parameters are a
dict whose "table" is a DataFrame with columns estimate, se, z (or t), p, ci_low, ci_high.

Numerical derivatives use central differences with a step scaled to each parameter. Pass an
analytic per-observation score (`score=`) when one is available: it is faster and more accurate.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats

ALPHA = 0.05


# =============================================================================
# Numerical derivatives
# =============================================================================
def _steps(theta):
    return (np.finfo(float).eps ** (1 / 3)) * np.maximum(np.abs(theta), 1.0)


def _obs_scores(loglik, theta, args):
    """n x k matrix of per-observation scores by central differences."""
    theta = np.asarray(theta, float)
    h = _steps(theta)
    cols = []
    for j in range(len(theta)):
        e = np.zeros_like(theta)
        e[j] = h[j]
        cols.append((np.asarray(loglik(theta + e, *args)) - np.asarray(loglik(theta - e, *args))) / (2 * h[j]))
    return np.column_stack(cols)


def _hessian(total_grad, theta):
    """Hessian of the total log-likelihood from central differences of its gradient (symmetrized)."""
    theta = np.asarray(theta, float)
    h = _steps(theta) * 10  # gradient is itself a difference: a larger outer step limits noise
    k = len(theta)
    H = np.zeros((k, k))
    for j in range(k):
        e = np.zeros(k)
        e[j] = h[j]
        H[:, j] = (total_grad(theta + e) - total_grad(theta - e)) / (2 * h[j])
    return (H + H.T) / 2


def _jacobian(fun, theta):
    """Jacobian of a vector function (m outputs) w.r.t. theta (k): m x k."""
    theta = np.asarray(theta, float)
    h = _steps(theta)
    f0 = np.atleast_1d(np.asarray(fun(theta), float))
    J = np.zeros((len(f0), len(theta)))
    for j in range(len(theta)):
        e = np.zeros_like(theta)
        e[j] = h[j]
        J[:, j] = (np.atleast_1d(fun(theta + e)) - np.atleast_1d(fun(theta - e))) / (2 * h[j])
    return J


def _table(est, cov, names, alpha=ALPHA, dist="z", df_resid=None):
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = est / se
    if dist == "t" and df_resid:
        crit = stats.t.ppf(1 - alpha / 2, df_resid)
        p = 2 * stats.t.sf(np.abs(stat), df_resid)
    else:
        crit = stats.norm.ppf(1 - alpha / 2)
        p = 2 * stats.norm.sf(np.abs(stat))
    return pd.DataFrame({"estimate": est, "se": se, dist: stat, "p": p,
                         "ci_low": est - crit * se, "ci_high": est + crit * se}, index=names)


def _cluster_meat(G, groups):
    """Sum over clusters of (sum of scores in cluster)(...)'. Returns meat and number of clusters."""
    g = pd.Series(groups).astype("category").cat.codes.to_numpy()
    n_g = g.max() + 1
    S = np.zeros((n_g, G.shape[1]))
    np.add.at(S, g, G)
    return S.T @ S, n_g


# =============================================================================
# Maximum likelihood
# =============================================================================
def mle(loglik, theta0, args=(), names=None, method="newton", cov_type="hessian", cluster=None,
        score=None, maxiter=200, tol=1e-8, alpha=ALPHA) -> dict:
    """Maximum likelihood from a per-observation log-likelihood.

    loglik(theta, *args) must return an array of n log-likelihood contributions (not their sum):
    the contributions are what the OPG, sandwich and clustered covariances are built from.
    score(theta, *args), optional, returns the n x k matrix of per-observation scores.

    method   "newton": Newton-Raphson with step halving (falls back to BFGS if the Hessian is
             not negative definite); "bfgs": scipy's BFGS on the negative total log-likelihood.
    cov_type "hessian" (inverse of the observed information; correct if the model is right),
             "opg" (outer product of gradients, BHHH), "sandwich" (robust to misspecification of
             the density, QMLE), or "cluster" (sandwich with scores summed within `cluster`).
    Report the sandwich when the likelihood is a convenient approximation (e.g. Poisson QMLE for
    overdispersed counts): the point estimate stays consistent if the mean is right, the Hessian
    SEs do not."""
    theta = np.asarray(theta0, float).copy()
    k = len(theta)
    names = list(names) if names is not None else [f"theta{j}" for j in range(k)]
    warn = []

    def scores(th):
        return np.asarray(score(th, *args), float) if score is not None else _obs_scores(loglik, th, args)

    def total(th):
        v = np.asarray(loglik(th, *args), float)
        return v.sum() if np.all(np.isfinite(v)) else -np.inf

    def total_grad(th):
        return scores(th).sum(0)

    converged, it = False, 0
    if method == "newton":
        ll = total(theta)
        if not np.isfinite(ll):
            raise ValueError("log-likelihood is not finite at theta0: choose other starting values")
        for it in range(1, maxiter + 1):
            g = total_grad(theta)
            H = _hessian(total_grad, theta)
            try:
                np.linalg.cholesky(-H)
            except np.linalg.LinAlgError:
                warn.append("Hessian not negative definite during Newton steps: switched to BFGS")
                method = "bfgs"
                break
            step = np.linalg.solve(-H, g)
            s, ll_new = 1.0, total(theta + step)
            while (not np.isfinite(ll_new) or ll_new < ll - 1e-12) and s > 1e-8:
                s /= 2
                ll_new = total(theta + s * step)
            theta_new = theta + s * step
            done = abs(ll_new - ll) < tol * (1 + abs(ll)) and np.max(np.abs(s * step)) < 1e-6 * (1 + np.max(np.abs(theta)))
            theta, ll = theta_new, ll_new
            if done:
                converged = True
                break
    if method == "bfgs":
        res = optimize.minimize(lambda th: -total(th), theta, jac=lambda th: -total_grad(th),
                                method="BFGS", options={"maxiter": maxiter * 50, "gtol": 1e-7})
        theta, converged, it = res.x, bool(res.success), int(res.nit)
        if not res.success:
            warn.append(f"BFGS: {res.message}")

    ll_obs = np.asarray(loglik(theta, *args), float)
    n = len(ll_obs)
    G = scores(theta)
    H = _hessian(total_grad, theta)
    grad_max = float(np.max(np.abs(G.sum(0))))
    if grad_max > 1e-3 * max(1.0, n ** 0.5):
        warn.append(f"gradient at the solution is not ~0 (max |g| = {grad_max:.2e}): check convergence")
    try:
        Hinv = np.linalg.inv(-H)
    except np.linalg.LinAlgError:
        Hinv = np.linalg.pinv(-H)
        warn.append("singular information matrix: parameters may not be identified")
    n_clusters = None
    if cov_type == "hessian":
        V = Hinv
    elif cov_type == "opg":
        V = np.linalg.inv(G.T @ G)
    elif cov_type == "sandwich":
        V = Hinv @ (G.T @ G) @ Hinv
    elif cov_type == "cluster":
        if cluster is None:
            raise ValueError("cov_type='cluster' needs cluster= (one label per observation)")
        meat, n_clusters = _cluster_meat(G, cluster)
        V = Hinv @ meat @ Hinv * n_clusters / (n_clusters - 1)
        if n_clusters < 40:
            warn.append(f"only {n_clusters} clusters: clustered SEs are fragile (see wild_cluster_bootstrap)")
    else:
        raise ValueError("cov_type must be 'hessian', 'opg', 'sandwich' or 'cluster'")
    if not converged:
        warn.append("optimizer did not report convergence")
    llf = float(ll_obs.sum())
    for w in warn:
        warnings.warn(w)
    return {"method": f"MLE ({method})", "cov_type": cov_type, "params": pd.Series(theta, index=names),
            "table": _table(theta, V, names, alpha), "cov": pd.DataFrame(V, index=names, columns=names),
            "loglik": llf, "n": n, "k": k, "aic": -2 * llf + 2 * k, "bic": -2 * llf + k * np.log(n),
            "converged": converged, "iterations": it, "grad_max": grad_max, "n_clusters": n_clusters,
            "hessian": H, "warnings": warn}


def _restricted_mle(loglik, theta0, R, r, args, method, score):
    """Maximize under R theta = r by reparametrizing theta = theta_p + N phi (N spans null(R))."""
    R = np.atleast_2d(np.asarray(R, float))
    r = np.zeros(R.shape[0]) if r is None else np.atleast_1d(np.asarray(r, float))
    theta_p = np.linalg.lstsq(R, r, rcond=None)[0]
    _, s, Vt = np.linalg.svd(R)
    rank = int(np.sum(s > 1e-10 * s.max()))
    N = Vt[rank:].T
    phi0 = np.linalg.lstsq(N, np.asarray(theta0, float) - theta_p, rcond=None)[0]

    def ll_phi(phi, *a):
        return loglik(theta_p + N @ phi, *a)

    sc = None if score is None else (lambda phi, *a: np.asarray(score(theta_p + N @ phi, *a)) @ N)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = mle(ll_phi, phi0, args=args, method=method, score=sc)
    return theta_p + N @ fit["params"].to_numpy(), fit["loglik"], rank


def trinity_test(loglik, theta0, R, r=None, args=(), names=None, method="newton", score=None,
                 unrestricted: dict | None = None) -> pd.DataFrame:
    """Likelihood-ratio, Wald and Lagrange-multiplier tests of H0: R theta = r.

    All three are asymptotically chi2(q), q = rank(R). Large disagreements between them signal a
    small sample or a badly behaved likelihood (Wald is not invariant to reparametrization; LR is
    usually the most reliable). LM needs only the restricted model: useful when the unrestricted
    one is hard to estimate. Pass `unrestricted=` an mle() result to avoid refitting it."""
    R = np.atleast_2d(np.asarray(R, float))
    r_vec = np.zeros(R.shape[0]) if r is None else np.atleast_1d(np.asarray(r, float))
    if unrestricted is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            unrestricted = mle(loglik, theta0, args=args, names=names, method=method, score=score)
    th_u = unrestricted["params"].to_numpy()
    V_u = np.linalg.inv(-unrestricted["hessian"])
    th_r, ll_r, q = _restricted_mle(loglik, th_u, R, r_vec, args, method, score)

    lr = 2 * (unrestricted["loglik"] - ll_r)
    d = R @ th_u - r_vec
    wald = float(d @ np.linalg.solve(R @ V_u @ R.T, d))

    def tg(th):
        G = np.asarray(score(th, *args)) if score is not None else _obs_scores(loglik, th, args)
        return G.sum(0)

    s_r = tg(th_r)
    H_r = _hessian(tg, th_r)
    lm = float(s_r @ np.linalg.solve(-H_r, s_r))
    out = pd.DataFrame({"statistic": [lr, wald, lm], "df": q,
                        "p": [stats.chi2.sf(v, q) for v in (lr, wald, lm)]},
                       index=["LR", "Wald", "LM (score)"])
    out.attrs["restricted_params"] = th_r
    return out


# =============================================================================
# Nonlinear least squares and the delta method
# =============================================================================
def nls(f, theta0, y, args=(), names=None, cov_type="HC1", alpha=ALPHA) -> dict:
    """Nonlinear least squares: minimize sum (y - f(theta, *args))^2.

    cov_type "classical" (s^2 (J'J)^-1, homoskedastic errors) or "HC1" (heteroskedasticity-robust,
    the default). Inference is asymptotic (t with n - k df). Try several starting values: NLS
    objectives often have local minima, and a solution that depends on theta0 is not identified
    by the data."""
    y = np.asarray(y, float)
    theta0 = np.asarray(theta0, float)
    k = len(theta0)
    names = list(names) if names is not None else [f"theta{j}" for j in range(k)]
    res = optimize.least_squares(lambda th: y - np.asarray(f(th, *args), float), theta0,
                                 method="lm" if len(y) >= k else "trf", xtol=1e-12, ftol=1e-12, gtol=1e-12)
    th = res.x
    e = y - np.asarray(f(th, *args), float)
    n = len(y)
    J = _jacobian(lambda t: f(t, *args), th)
    JtJ_inv = np.linalg.inv(J.T @ J)
    if cov_type == "classical":
        V = JtJ_inv * (e @ e) / (n - k)
    elif cov_type == "HC1":
        V = JtJ_inv @ (J.T * e ** 2) @ J @ JtJ_inv * n / (n - k)
    else:
        raise ValueError("cov_type must be 'classical' or 'HC1'")
    warn = [] if res.success else [f"least_squares: {res.message}"]
    cond = float(np.linalg.cond(J))
    if cond > 1e8:
        warn.append(f"Jacobian condition number {cond:.1e}: parameters weakly identified or badly scaled")
    return {"method": "NLS", "cov_type": cov_type, "params": pd.Series(th, index=names),
            "table": _table(th, V, names, alpha, dist="t", df_resid=n - k),
            "cov": pd.DataFrame(V, index=names, columns=names), "resid": e, "ssr": float(e @ e),
            "n": n, "k": k, "jacobian_cond": cond, "converged": bool(res.success), "warnings": warn}


def delta_method(g, theta, cov, names=None, alpha=ALPHA) -> pd.DataFrame:
    """SE and CI of g(theta) by the delta method: Var(g) ~= J V J', J = dg/dtheta' at theta-hat.

    theta, cov: estimates and their covariance (Series/DataFrame from any fit, or arrays).
    g returns a scalar or a vector (several functions at once, with their joint covariance in
    .attrs["cov"]). Typical uses: elasticity at a point, long-run multiplier b/(1-phi), ratio of
    coefficients (willingness to pay), turning point -b1/(2 b2), exp(b) - 1. For ratios whose
    denominator is imprecise (|denominator/se| < ~3) the normal approximation is poor: bootstrap."""
    th = np.asarray(theta, float)
    V = np.asarray(cov, float)
    est = np.atleast_1d(np.asarray(g(th), float))
    J = _jacobian(g, th)
    Vg = J @ V @ J.T
    idx = list(names) if names is not None else [f"g{j}" for j in range(len(est))]
    out = _table(est, Vg, idx, alpha)
    out.attrs["cov"] = Vg
    return out


# =============================================================================
# Bootstrap
# =============================================================================
def bootstrap(stat_fn, df: pd.DataFrame, B: int = 999, cluster: str | None = None, seed: int = 0,
              alpha: float = ALPHA) -> dict:
    """Pairs bootstrap (resample rows) or cluster bootstrap (resample whole clusters).

    stat_fn(df) -> float, array or Series (e.g. lambda d: smf.ols(...).fit().params["x"]).
    Returns the estimate on the original data, the bootstrap SE, percentile and normal CIs.
    Resample at the level of independence: rows for i.i.d. data, clusters otherwise. Not valid
    for time series (use a block bootstrap) or for statistics at the boundary (max, min, a
    parameter on the edge of its space). With few clusters prefer wild_cluster_bootstrap."""
    rng = np.random.default_rng(seed)
    est = stat_fn(df)
    names = list(est.index) if isinstance(est, pd.Series) else None
    est = np.atleast_1d(np.asarray(est, float))
    if cluster is not None:
        groups = {g: ix for g, ix in df.groupby(cluster, sort=False).indices.items()}
        keys = list(groups)
    draws, failed = [], 0
    for _ in range(B):
        if cluster is None:
            sample = df.iloc[rng.integers(0, len(df), len(df))]
        else:
            pick = rng.integers(0, len(keys), len(keys))
            sample = df.iloc[np.concatenate([groups[keys[i]] for i in pick])]
        try:
            v = np.atleast_1d(np.asarray(stat_fn(sample), float))
            if np.all(np.isfinite(v)):
                draws.append(v)
            else:
                failed += 1
        except Exception:
            failed += 1
    D = np.vstack(draws)
    se = D.std(axis=0, ddof=1)
    z = stats.norm.ppf(1 - alpha / 2)
    tab = pd.DataFrame({"estimate": est, "se": se,
                        "ci_low": np.quantile(D, alpha / 2, axis=0), "ci_high": np.quantile(D, 1 - alpha / 2, axis=0),
                        "ci_low_normal": est - z * se, "ci_high_normal": est + z * se,
                        "bias": D.mean(0) - est},
                       index=names or [f"stat{j}" for j in range(len(est))])
    warn = []
    if failed > 0.05 * B:
        warn.append(f"{failed} of {B} bootstrap replications failed: the statistic is unstable")
    if cluster is not None and len(keys) < 30:
        warn.append(f"only {len(keys)} clusters: the cluster bootstrap is unreliable, use wild_cluster_bootstrap")
    return {"method": "cluster bootstrap" if cluster else "pairs bootstrap", "table": tab, "draws": D,
            "B_effective": len(draws), "n": len(df), "warnings": warn}


_WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def wild_cluster_bootstrap(df: pd.DataFrame, formula: str, coef: str, cluster: str, null: float = 0.0,
                           B: int = 9999, weights: str = "auto", seed: int = 0, alpha: float = ALPHA,
                           ci: bool = True, ci_points: int = 81) -> dict:
    """Restricted wild cluster bootstrap-t (WCR) for one OLS coefficient.

    With few clusters (G below ~30-40, or very unequal cluster sizes) the usual clustered SEs
    over-reject; WCR keeps the test size close to nominal even with G ~ 6-10 (Cameron, Gelbach &
    Miller 2008; MacKinnon & Webb 2017; Roodman, MacKinnon, Nielsen & Webb 2019).
    1. fit OLS, t = (b - null) / se_CR1;
    2. re-fit imposing the null, keep restricted fitted values and residuals;
    3. for each draw, flip each cluster's residuals by v_g (Rademacher +-1, or Webb's six-point
       weights when G < 12, since Rademacher allows only 2^G distinct draws) and recompute t*;
    4. p = share of |t*| >= |t|.
    The confidence interval inverts the test over a grid of nulls with the same draws.
    Report the number of clusters, not only N.
    Caveat: what matters for a binary treatment that varies at the cluster level is the number of
    TREATED clusters. With very few treated clusters (roughly 1-5) every cluster-based method,
    WCR included, can over- or under-reject (MacKinnon & Webb 2017, 2018): add randomization
    inference (permute the treatment across clusters) or Conley-Taber / Ferman-Pinto."""
    import statsmodels.formula.api as smf
    base = smf.ols(formula, df).fit()
    names = list(base.model.exog_names)
    if coef not in names:
        raise KeyError(f"{coef!r} not in {names}")
    X = base.model.exog
    y = base.model.endog
    rows = base.model.data.row_labels
    g_codes = pd.Series(df.loc[rows, cluster]).astype("category").cat.codes.to_numpy()
    G = int(g_codes.max() + 1)
    n, K = X.shape
    k = names.index(coef)
    A = np.linalg.solve(X.T @ X, X.T)          # K x n
    a_k = A[k]
    c = G / (G - 1) * (n - 1) / (n - K)
    rng = np.random.default_rng(seed)
    if weights == "auto":
        weights = "webb" if G < 12 else "rademacher"
    if weights == "rademacher":
        V = rng.choice([-1.0, 1.0], size=(G, B))
    elif weights == "webb":
        V = rng.choice(_WEBB, size=(G, B))
    else:
        raise ValueError("weights must be 'rademacher', 'webb' or 'auto'")
    Vn = V[g_codes]                              # n x B

    def se_k(E):
        S = np.zeros((G, E.shape[1]) if E.ndim == 2 else G)
        np.add.at(S, g_codes, (a_k[:, None] * E) if E.ndim == 2 else a_k * E)
        return np.sqrt(c * (S ** 2).sum(0))

    b = A @ y
    se0 = float(se_k(y - X @ b))
    Xr = np.delete(X, k, axis=1)
    Ar = np.linalg.pinv(Xr) if Xr.shape[1] else None    # (K-1) x n, restricted OLS operator

    def pval(h0):
        t_obs = (b[k] - h0) / se0
        ys = y - h0 * X[:, k]
        fit_r = Xr @ (Ar @ ys) if Ar is not None else np.zeros(n)
        u_r = ys - fit_r
        Yb = (fit_r + h0 * X[:, k])[:, None] + u_r[:, None] * Vn
        Bb = A @ Yb
        tb = (Bb[k] - h0) / se_k(Yb - X @ Bb)
        return float(np.mean(np.abs(tb) >= abs(t_obs))), t_obs

    p, t_obs = pval(null)
    lo = hi = np.nan
    if ci:
        grid = b[k] + se0 * np.linspace(-6, 6, ci_points)
        keep = np.array([pval(h)[0] > alpha for h in grid])
        if keep.any():
            lo, hi = grid[keep].min(), grid[keep].max()
            if keep[0] or keep[-1]:
                warnings.warn("CI reaches the edge of the search grid: widen it (ci_points or the ±6 SE range)")
    p_cr1 = 2 * stats.t.sf(abs(t_obs), G - 1)
    warn = []
    if weights == "rademacher" and 2 ** G < 2 * B:
        warn.append(f"Rademacher weights with G={G} give only {2 ** G} distinct draws: use weights='webb'")
    return {"estimand": coef, "method": f"wild cluster bootstrap-t (restricted, {weights})",
            "estimate": float(b[k]), "se": se0, "t": float(t_obs), "p": p, "ci_low": lo, "ci_high": hi,
            "p_cr1_t_G-1": float(p_cr1), "null": null, "n": n, "n_clusters": G, "B": B, "warnings": warn}
