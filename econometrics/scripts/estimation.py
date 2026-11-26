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


def _jacobian(fun, theta, h=None):
    """Jacobian of a vector function (m outputs) w.r.t. theta (k): m x k. h: steps (default _steps(theta))."""
    theta = np.asarray(theta, float)
    h = _steps(theta) if h is None else np.asarray(h, float)
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

    Returns a dict (v2.7.3: keys exact):
      "method"         str, "pairs bootstrap" or "cluster bootstrap"
      "table"          DataFrame, one row per statistic (the Series' labels, else "stat0", "stat1", …), columns
                       ["estimate", "se", "ci_low", "ci_high", "ci_low_normal", "ci_high_normal", "bias"]; ci_low/high
                       are percentile limits
      "draws"          array (B_effective × number of statistics), the replications that gave finite values
      "B_effective"    int, replications kept (failed or non-finite ones are dropped)
      "n"              int, rows of df
      "warnings"       list of str (many failed replications; fewer than 30 clusters)
    Example (r = the result): r["table"]["se"].iloc[0]; r["table"]["ci_low"]

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
    # v2.7.3 (review r1-A A13): "if cluster" called cluster=0 a pairs bootstrap
    return {"method": "cluster bootstrap" if cluster is not None else "pairs bootstrap", "table": tab, "draws": D,
            "B_effective": len(draws), "n": len(df), "warnings": warn}


_WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def wild_cluster_bootstrap(df: pd.DataFrame, formula: str, coef: str, cluster: str, null: float = 0.0,
                           B: int = 9999, weights: str = "auto", seed: int = 0, alpha: float = ALPHA,
                           ci: bool = True, ci_points: int = 81) -> dict:
    """Restricted wild cluster bootstrap-t (WCR) for one OLS coefficient.

    Returns a dict (v2.7.3: keys exact):
      "estimand"       str, the coefficient's name (coef)
      "method"         str, "wild cluster bootstrap-t (restricted, <weights>)", the weights used ("rademacher" or
                       "webb", never "auto")
      "estimate", "se"  float, OLS coefficient and its CR1 standard error
      "t"              float, (estimate − null) / se
      "p"              float, WCR bootstrap p-value of H0: coefficient = null
      "ci_low", "ci_high"  float, the CI by test inversion; NaN with ci=False (or when no point of the grid is
                       accepted)
      "p_cr1_t_G-1"    float, the analytic CR1 p with t(G − 1), for comparison (note the hyphen in the name)
      "null"           float, as passed
      "n", "n_clusters", "B"  int
      "warnings"       list of str (Rademacher weights with few clusters); a CI at the grid edge is a Python
                       warning instead
    Example (r = the result): r["p"]; r["ci_low"], r["ci_high"]; r["p_cr1_t_G-1"]

    With few clusters (G below ~30-40, or very unequal cluster sizes) the usual clustered SEs
    over-reject; WCR keeps the test size close to nominal even with G ~ 6-10 (Cameron, Gelbach &
    Miller 2008; MacKinnon & Webb 2017; Roodman, MacKinnon, Nielsen & Webb 2019).
    1. fit OLS, t = (b - null) / se_CR1;
    2. re-fit imposing the null, keep restricted fitted values and residuals;
    3. for each draw, flip each cluster's residuals by v_g (Rademacher +-1, or Webb's six-point
       weights when G < 12, since Rademacher allows only 2^G distinct draws) and recompute t*;
    4. p = share of |t*| >= |t|.
    The confidence interval inverts the test over a grid of nulls with the same draws.
    Report the number of clusters, not only N. A missing cluster label raises ValueError (v2.7.4).
    Caveat: what matters for a binary treatment that varies at the cluster level is the number of
    TREATED clusters. With very few treated clusters (roughly 1-5) every cluster-based method,
    WCR included, can over- or under-reject (MacKinnon & Webb 2017, 2018): add randomization
    inference (causal.randomization_inference) or Conley-Taber / Ferman-Pinto."""
    import statsmodels.formula.api as smf
    base = smf.ols(formula, df).fit()
    names = list(base.model.exog_names)
    if coef not in names:
        raise KeyError(f"{coef!r} not in {names}")
    X = base.model.exog
    y = base.model.endog
    rows = base.model.data.row_labels
    g_lab = pd.Series(df.loc[rows, cluster])
    if g_lab.isna().any():
        # v2.7.4: cat.codes gave -1 to a missing label and those rows were merged into the last cluster, silently
        raise ValueError(f"cluster {cluster!r} has {int(g_lab.isna().sum())} missing label(s) among the rows used: "
                         "drop those rows or fill the cluster id (a missing label is not a cluster of its own)")
    g_codes = pd.factorize(g_lab, sort=True)[0]   # review r1 S3: unused categories of a categorical are not clusters
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


# =============================================================================
# Generalized method of moments (v2.8.0)
# =============================================================================
_GMM_STEPS = ("one-step", "two-step", "iterated")
_GMM_KERNELS = ("bartlett", "truncated")


def _gmm_S(G, lags, kernel, center):
    """Long-run covariance of the moment contributions G (n x m): Gamma_0 + sum_j w_j (Gamma_j + Gamma_j')."""
    n = G.shape[0]
    U = G - G.mean(0) if center else G
    S = U.T @ U / n
    for j in range(1, (lags or 0) + 1):
        w = 1.0 - j / (lags + 1.0) if kernel == "bartlett" else 1.0
        Gam = U[j:].T @ U[:-j] / n
        S = S + w * (Gam + Gam.T)
    return (S + S.T) / 2


def _gmm_pilot_steps(th, th0):
    """Pilot central-difference steps: eps^(1/3) max(|theta|, |theta0|), 1 in place of the scale when both are 0."""
    sc = np.maximum(np.abs(np.asarray(th, float)), np.abs(np.asarray(th0, float)))
    return np.finfo(float).eps ** (1 / 3) * np.where(sc > 0, sc, 1.0)


def _gmm_steps(G, th, th0):
    """Central-difference steps for gmm's derivatives, chosen per parameter by stability (v2.8.0 review r3, S2-2).

    Candidate scales s: |theta_j|, |theta0_j|, 1 and the response scale tau_j = rms(g_i) / rms(dg_i/dtheta_j) (the
    change in theta_j that moves the contributions by their own size, with the derivative from the pilot step). For
    each, the per-observation derivatives with h = eps^(1/3) s and with h/2; the step kept is the one whose two
    estimates agree best (rounding grows and truncation shrinks as h falls, and their disagreement tracks the error).
    A single scale failed somewhere: relative to theta with theta0 far from the estimate (a rate of 2e-4 started at
    1: Jacobian off by 7.6e-3, the estimate by 1.4e-4); relative to the estimate when it is 0 by chance (a slope at
    1e-14: off by 100%); the response scale with offsets (the residual is a cancellation of large terms) and far from
    the estimate (terms that do not depend on theta inflate rms(g)). Measured against analytic Jacobians in eleven
    designs (development repository, claude/desenho-r3-v2.8.0.md): best or tied everywhere. A candidate whose two
    differences are exactly 0 (a step below the resolution of the moments) or not finite is skipped; among those
    that agree about as well as the best (10x, or to 1e-11), the largest step is kept (less rounding); with none
    left, the pilot eps^(1/3) max(|theta|, |theta0|) (1 when both are 0). G returns the n x m contributions."""
    th = np.asarray(th, float)
    th0 = np.asarray(th0, float)
    k = len(th)
    hp = _gmm_pilot_steps(th, th0)
    out = hp.copy()
    eps3 = np.finfo(float).eps ** (1 / 3)
    with np.errstate(all="ignore"):                                 # a trial step outside the domain gives NaN
        try:
            g0 = np.asarray(G(th), float)
        except (ValueError, FloatingPointError, OverflowError, ZeroDivisionError):
            return hp
        rg = float(np.sqrt(np.mean(g0 ** 2)))
        for j in range(k):
            def diff(h, j=j):
                e = np.zeros(k)
                e[j] = h
                try:
                    gp, gm = np.asarray(G(th + e), float), np.asarray(G(th - e), float)
                except (ValueError, FloatingPointError, OverflowError, ZeroDivisionError):
                    return None, None
                return (gp - gm) / (2 * h), (gp.mean(0) - gm.mean(0)) / (2 * h)
            cands = [abs(th[j]), abs(th0[j]), 1.0]
            dobs, _ = diff(hp[j])
            if dobs is not None:
                rd = float(np.sqrt(np.mean(dobs ** 2)))
                if np.isfinite(rd) and rd > 0 and np.isfinite(rg) and rg > 0:
                    cands.append(rg / rd)
            qs = []
            for sc in sorted(set(c for c in cands if np.isfinite(c) and c > 0)):
                h = eps3 * sc
                # agreement of the per-observation derivatives, which the rank test uses: on the mean the rounding
                # noise of a residual that cancels large terms (an offset of 1e4) averages out, and the mean picked a
                # step 3000x too small for the rank test (review r4 cycle)
                d1, _ = diff(h)
                d2, _ = diff(h / 2)
                if d1 is None or d2 is None or not (np.all(np.isfinite(d1)) and np.all(np.isfinite(d2))):
                    continue
                den = max(float(np.max(np.abs(d1))), float(np.max(np.abs(d2))))
                if den == 0:
                    continue
                qs.append((float(np.max(np.abs(d1 - d2))) / den, h))
            if qs:
                # among the steps that agree about as well as the best one (within 10x, or to 1e-11), the largest:
                # truncation there is already negligible and a larger step rounds less (moments linear in theta,
                # where every step is exact up to rounding and the smallest disagreement is chance)
                qmin = min(q for q, _ in qs)
                out[j] = max(h for q, h in qs if q <= max(10 * qmin, 1e-11))
    return out


def _gmm_rank_test(G_of, th, lags, h=None):
    """Cragg-Donald (1997) minimum-Wald test of H0: rank(D) <= k - 1, D = E[dg_i/dtheta'] (m x k).

    For each direction c (|c| = 1), the Wald statistic of E[dg_i/dtheta' c] = 0 uses the long-run covariance of
    the per-observation derivatives along c; the statistic is its minimum over c, chi2(m - k + 1) under H0. It does
    not depend on the units of theta or on invertible recombinations of the moments. It is a test of the RANK, not
    a measure of strength: under any fixed alternative it grows with n (its variance is centred, not taken under
    H0), so it is not a first-stage F and has no rule-of-thumb cut (v2.8.0 review r1).

    The covariance of vec(dg_i/dtheta') is computed once (Omega, mk x mk) and the variance along c is the quadratic
    form (c x I)' Omega (c x I). With lags it always uses the Bartlett kernel, which is positive semidefinite (a
    truncated kernel made the variance indefinite and the statistic negative; review r1). A direction in which the
    derivatives do not vary (variance ~0) but whose mean is not 0 is identified exactly: the statistic is inf.
    Moments that do not depend on some parameters (a block structure, a parameter outside the moments) are handled
    exactly within the search budget: the minimum is taken also inside each moment's null space, where that moment
    is dropped (search in _gmm_rank_search; v2.8.0 reviews r3 and r4). A parameter outside all moments gives 0.
    Precision: with a regressor stored with an offset up to 1e4, with or without HAC lags, the statistic matches the
    well-scaled one to 1e-6 and does not depend on the random starts of the search; with 3e4 and a small statistic
    (~1) the finite differences leave ~1e-5 (with exact derivatives 1e-7), and 1e5 matches to 1e-5. A regressor in
    units of 1e9 (a scale within one moment) gives the same statistic as in units of 1. Centre a regressor with a
    large offset.
    Returns (stat, df, ac1, capped); (nan, df, nan, False) when the moments are not finite next to the estimate;
    capped is True when the candidate subspaces passed their cap (the statistic is then an upper bound). ac1 is the
    largest absolute first-order autocorrelation among the columns of vec(dg_i/dtheta') that vary: the persistence
    that makes the test over-reject (v2.8.0 review r2, S2-4; sizes measured in references/estimation-theory.md
    section 7)."""
    th = np.asarray(th, float)
    h = _steps(th) if h is None else np.asarray(h, float)
    k = len(th)
    cols = []
    try:
        for j in range(k):
            e = np.zeros(k)
            e[j] = h[j]
            cols.append((G_of(th + e) - G_of(th - e)) / (2 * h[j]))
    except ValueError:
        return float("nan"), None, float("nan"), False
    Di = np.stack(cols, axis=2)                                    # n x m x k
    n, m, _ = Di.shape
    if not np.all(np.isfinite(Di)):
        return float("nan"), m - k + 1, float("nan"), False
    Dbar = Di.mean(0)
    flat = Di.reshape(n, m * k)                                    # column j*k + a: moment j, parameter a
    Uc = flat - Dbar.reshape(1, m * k)
    ss = np.sum(Uc ** 2, 0)
    rms = np.sqrt(np.mean(flat ** 2, 0))
    var_ok = ss > (1e-12 * rms) ** 2 * n
    ac1 = float(np.max(np.abs(np.sum(Uc[1:, var_ok] * Uc[:-1, var_ok], 0) / ss[var_ok]))) if var_ok.any() and n > 2 \
        else 0.0
    Om = _gmm_S(flat, lags, "bartlett", True).reshape(m, k, m, k)
    M2 = (flat.T @ flat / n).reshape(m, k, m, k)                   # uncentred second moments: the scale of D_i
    best, capped = _gmm_rank_search(Di, Dbar, Om, M2)
    return best, m - k + 1, ac1, capped


# Singular-value ratio under which a moment's derivative matrix (n x k, each column divided by its own root mean square,
# so the ratio does not depend on the units of the parameters within the moment) is taken not to move in a direction.
# Measured on both sides (development repository, claude/desenho-r3-v2.8.0.md, P0, and desenho-r4): finite-difference
# noise in a structural null direction <= 1.3e-10; genuine directions >= 6.7e-6 with a regressor offset of 1e6, where
# the fit itself already breaks. The columns are equilibrated per moment because a global scale (v2.8.0 review r4,
# S1-a) took a regressor in units of 1e9 for a null direction and gave "not identified" (5.97 instead of 153.9).
_RANK_NULL_TOL = 1e-8
_RANK_MAX_SUBSPACES = 200
_RANK_BUDGET_S = 30.0


def _gmm_rank_search(Di, Dbar, Om, M2):
    """Minimum over directions c of the Wald statistic of E[dg_i/dtheta' c] = 0 (see _gmm_rank_test); (min, capped).

    The minimum is discontinuous where a moment stops depending on c: next to such a direction the moment's share is
    n mean^2 / variance of its derivative along c, whatever the distance, and in it the moment does not move and has
    no share. So the infimum lies on a set of measure zero (the moment's structural null space) that a search over all
    directions does not find (v2.8.0 review r3: a block of moments in theta1 alone hid the weak identification of
    theta2, 140.6 instead of 5.64; k = 3 gave 176 instead of 0.54; a parameter outside the moments crashed). The
    search therefore runs on candidates: the whole space and every non-trivial intersection of the moments' null
    spaces (each judged with that moment's columns equilibrated: review r4, S1-a), where the moments that vanish are
    dropped explicitly (none left: the statistic is 0). Inside each candidate the directions are searched in
    coordinates preconditioned by A = sum_j M2_jj / tr M2_jj restricted to it: a regressor stored with a large offset
    made a valley of width ~1/offset, and the result depended on the random starts (review r3, S2-1). The candidates
    beyond the whole space are searched from the largest down within a budget of _RANK_BUDGET_S seconds and
    _RANK_MAX_SUBSPACES subspaces (fixed-effects designs with group dummies have many; review r4, S2-b); past either,
    capped is True and the statistic is an upper bound."""
    import time as _time
    n, m, k = Di.shape
    M2j = [M2[j, :, j, :] for j in range(m)]
    tr = np.array([np.trace(B) for B in M2j])
    dead = ~(tr > 0)                                                 # a moment that does not depend on theta at all
    A = sum(M2j[j] / tr[j] for j in range(m) if not dead[j]) if (~dead).any() else np.eye(k)
    A = (A + A.T) / 2
    tol = _RANK_NULL_TOL
    Dj = [Di[:, j, :] for j in range(m)]                             # n x k, original coordinates

    def eq_null(j, S):
        # orthonormal basis (in the coordinates of S) of the directions of span(S) along which moment j does not move,
        # judged with the columns of Dj S equilibrated (each divided by its root mean square; a zero column is null)
        B = Dj[j] @ S
        sc = np.sqrt(np.mean(B ** 2, axis=0))
        live_c = sc > 0
        d = S.shape[1]
        if not live_c.any():
            return np.eye(d)
        Be = B[:, live_c] / sc[live_c]
        _, sv, vt = np.linalg.svd(Be, full_matrices=True)
        sv = np.concatenate([sv, np.zeros(Be.shape[1] - len(sv))])
        null_e = vt.T[:, sv <= tol * sv[0]]                          # in equilibrated coordinates of the live columns
        basis = []
        for v in null_e.T:
            a_ = np.zeros(d)
            a_[live_c] = v / sc[live_c]
            basis.append(a_)
        for i in np.flatnonzero(~live_c):
            e_ = np.zeros(d)
            e_[i] = 1.0
            basis.append(e_)
        if not basis:
            return np.zeros((d, 0))
        return np.linalg.qr(np.column_stack(basis))[0]

    # the scale of each moment for the "does it move along c" test: the largest singular value of its equilibrated
    # matrix and the column scales, so the test is the same relative singular value as in eq_null
    colsc = [np.sqrt(np.diag(M2j[j])) for j in range(m)]
    topeq = np.zeros(m)
    for j in range(m):
        if not dead[j]:
            lc = colsc[j] > 0
            topeq[j] = np.linalg.norm(Dj[j][:, lc] / colsc[j][lc], 2) / np.sqrt(n)

    def moves(j, c):
        if dead[j]:
            return False
        lc = colsc[j] > 0
        ce = np.linalg.norm(colsc[j][lc] * c[lc])
        return np.linalg.norm(Dj[j] @ c) / np.sqrt(n) > tol * topeq[j] * ce if ce > 0 else False

    def dropped_on(S):
        return frozenset(j for j in range(m) if dead[j] or eq_null(j, S).shape[1] == S.shape[1])

    eye = np.eye(k)
    cands = [(eye, dropped_on(eye))]
    projs = [eye]
    queue, capped = [0], False
    while queue and not capped:
        S, drop = cands[queue.pop(0)]
        for j in range(m):
            if j in drop:
                continue
            a = eq_null(j, S)
            if a.shape[1] == 0 or a.shape[1] == S.shape[1]:
                continue
            Sn = np.linalg.qr(S @ a)[0]
            P = Sn @ Sn.T
            if any(np.max(np.abs(p - P)) < 1e-8 for p in projs):
                continue
            if len(cands) >= _RANK_MAX_SUBSPACES:
                capped = True
                break
            projs.append(P)
            cands.append((Sn, dropped_on(Sn)))
            queue.append(len(cands) - 1)

    def wald(c, drop):
        # Each moment is standardized by its own scale along c (root of the uncentred second moment of its derivative)
        # before any cut, so the cuts are scale-free per moment (review r2, S1-1); the Wald is invariant to that
        # diagonal rescaling.
        c = np.asarray(c, float)
        nc = np.linalg.norm(c)
        if not np.isfinite(nc) or nc == 0:
            return np.inf
        c = c / nc
        live = np.array([j not in drop and moves(j, c) for j in range(m)])
        if not live.any():
            return 0.0                                              # no moment moves along c: D c = 0
        V = np.einsum("iajb,a,b->ij", Om, c, c)[np.ix_(live, live)]
        dm = (Dbar @ c)[live]
        s = np.sqrt(np.maximum(np.einsum("iaib,a,b->i", M2, c, c)[live], 0.0))
        if not np.all(s > 0):
            return np.inf                                           # cannot happen: live moments move along c
        Vs = V / np.outer(s, s)
        ds = dm / s
        if np.any((np.diag(Vs) <= 1e-12) & (np.abs(ds) > 1e-6)):
            return np.inf                                           # a moment moves deterministically along c
        ev_, vec_ = np.linalg.eigh((Vs + Vs.T) / 2)
        proj = vec_.T @ ds
        keep = ev_ > 1e-12 * max(float(np.abs(ev_).max()), 1.0)
        if np.any(np.abs(proj[~keep]) > 1e-6):
            return np.inf                                           # a recombination of moments does
        return float(n * np.sum(proj[keep] ** 2 / ev_[keep]))

    def precond(S):
        # basis of span(S) in which the average second moment of the derivatives is the identity
        As = S.T @ A @ S
        ev, Q = np.linalg.eigh((As + As.T) / 2)
        return S @ (Q / np.sqrt(np.maximum(ev, max(float(ev.max()), np.finfo(float).tiny) * 1e-14)))

    # valleys: where a moment's mean derivative cancels (Dbar_j c = 0) its share drops from n mean^2 / var to ~0 over a
    # width ~ (its smallest relative dependence); a moment that depends on theta2 through 1e-9 theta2 w makes a valley
    # 1e-9 wide that no grid finds (review r4, S3-2: the statistic was overstated and hid a weak theta2). Each such
    # hyperplane is searched too, the moment kept (it may still vary there), and k = 2 polishes around its direction.
    hyper = []
    if k >= 2:
        for j in range(m):
            row = Dbar[j]
            if dead[j] or not np.any(row):
                continue
            H = np.linalg.svd(row[None, :], full_matrices=True)[2][1:].T          # k x (k - 1), orthonormal
            P = H @ H.T
            if any(np.max(np.abs(p - P)) < 1e-12 for p in projs):
                continue
            projs.append(P)
            hyper.append(H)

    rng = np.random.default_rng(0)
    best = np.inf
    t_start = _time.time()
    cands = cands[:1] + [(H, frozenset(j for j in range(m) if dead[j])) for H in hyper] + cands[1:]
    n_hyper = len(hyper)
    order = [0] + list(range(1, 1 + n_hyper)) + sorted(range(1 + n_hyper, len(cands)),
                                                        key=lambda i: -cands[i][0].shape[1])
    # the Wald is inf along directions in which a moment moves deterministically: BFGS then differences inf and warns
    # (RuntimeWarning, noise in the user's console; review r2); the values it returns are still compared
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for idx in order:
            if idx > 0 and _time.time() - t_start > _RANK_BUDGET_S:
                capped = True
                break
            S, drop = cands[idx]
            Pm = precond(S)
            d = S.shape[1]
            f = (lambda a, Pm=Pm, drop=drop: wald(Pm @ np.atleast_1d(np.asarray(a, float)), drop))
            if d == 1:
                val = f([1.0])
                if k == 2:
                    # polish around the direction: the valley may sit slightly off the plane (the moment's variance and
                    # its correlation with the others); bracket +-1e-3 rad, rescaled to (0, 1)
                    c0 = Pm[:, 0] / np.linalg.norm(Pm[:, 0])
                    a0 = float(np.arctan2(c0[1], c0[0]))
                    res = optimize.minimize_scalar(
                        lambda q, a0=a0, drop=drop: wald(np.array([np.cos(a0 - 1e-3 + 2e-3 * q),
                                                                   np.sin(a0 - 1e-3 + 2e-3 * q)]), drop),
                        bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-12})
                    val = min(val, float(res.fun))
            elif d == 2:
                g2 = (lambda t, f=f: f([np.cos(t), np.sin(t)]))
                grid = np.linspace(0.0, np.pi, 361)
                vals = [g2(t) for t in grid]
                i = int(np.argmin(vals))
                # the bracket rescaled to (0, 1): the bounded method's tolerance is relative to |x| (sqrt(eps) |x|),
                # 2e-8 rad next to pi/2, wider than the valley of a parameter in units of 1e8 (review r4, S1-a;
                # same lesson as v2.6.1)
                lo_, hi_ = grid[max(i - 1, 0)], grid[min(i + 1, 360)]
                res = optimize.minimize_scalar(lambda q, f=g2, lo_=lo_, hi_=hi_: f(lo_ + (hi_ - lo_) * q),
                                               bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-12})
                val = min(float(res.fun), float(vals[i]))
            else:
                # the statistic does not depend on |u|: unconstrained minimisation, from the axes and random starts
                val = np.inf
                for a0 in [np.eye(d)[j] for j in range(d)] + [rng.standard_normal(d) for _ in range(3 * d)]:
                    f0 = f(a0)
                    if not np.isfinite(f0):
                        continue
                    res = optimize.minimize(f, a0, method="BFGS",
                                            options={"gtol": 1e-10 * max(1.0, f0), "maxiter": 400})
                    val = min(val, float(res.fun), f0)
                    res = optimize.minimize(f, res.x, method="Nelder-Mead",
                                            options={"xatol": 1e-9, "fatol": 1e-10, "maxiter": 200 * d})
                    val = min(val, float(res.fun))
            best = min(best, val)
    return best, capped


def _fd_column(fun, th, j, h, lo, hi):
    """d fun / d theta_j by finite differences with step h, inside [lo, hi]: central when theta +- h fit, else the
    one-sided second-order formula (-3 f0 + 4 f(h) - f(2h)) / 2h toward the free side; nan when neither fits."""
    e = np.zeros_like(th)
    e[j] = 1.0
    if th[j] - h >= lo[j] and th[j] + h <= hi[j]:
        return (fun(th + h * e) - fun(th - h * e)) / (2 * h)
    for sgn in (1.0, -1.0):
        if lo[j] <= th[j] + 2 * sgn * h <= hi[j]:
            f0, f1, f2 = fun(th), fun(th + sgn * h * e), fun(th + 2 * sgn * h * e)
            return sgn * (-3 * f0 + 4 * f1 - f2) / (2 * h)
    return np.full(np.shape(fun(th)), np.nan)


def _gmm_probe_moves(gbar, th0, j, h, lo, hi, gs):
    """Which mean moments move when theta_j is pushed outward from theta0, geometrically (1e4 h, 1e6 h, 1e8 h and the
    scale of the parameter, max(|theta0_j|, 1)), inside the bounds; each moment on its own scale gs (by more than
    4 eps). A structural zero stays flat everywhere; a step function or a kink moves somewhere (v2.8.0 review r4)."""
    res = 4 * np.finfo(float).eps
    th0 = np.asarray(th0, float)
    with np.errstate(all="ignore"):
        g0_ = np.asarray(gbar(th0), float) / gs
        moves = np.zeros(len(g0_), bool)
        for H in (1e4 * h, 1e6 * h, 1e8 * h, max(abs(th0[j]), 1.0)):
            for sgn in (1.0, -1.0):
                e_ = np.zeros_like(th0)
                e_[j] = sgn * H
                if lo[j] <= th0[j] + sgn * H <= hi[j]:
                    gH = np.asarray(gbar(th0 + e_), float) / gs
                    moves |= ~np.isfinite(gH) | (np.abs(gH - g0_) > res)
    return moves


def _gmm_check_jacobian(gbar, Du, th0, lo, hi, tol=1e-3, ref_tol=1e-5, steps0=None, g_scale=1.0):
    """Compares a user Jacobian with finite differences at theta0, column by column (review r2, S1-2: on the whole
    matrix, a wrong column in small units hid behind a large one). The step is gmm's step at theta0 (chosen by
    stability, steps0; review r3, S2-2) and stays inside the bounds (S2-2, S2-3). Refuses only when the
    disagreement is clear: gap above tol of the column's largest entry AND the finite differences reliable there
    (steps h and h/4 agree to ref_tol of that scale); otherwise a disagreement is a warning (the reference itself
    is unstable: a kink, an undefined neighbourhood). Differences that are 0 at both steps, up to rounding (|d| h
    <= 4 eps g_scale, g_scale the root mean square of the contributions at theta0), are no reference (v2.8.0
    review r3, S2-5): moments that are step functions (an indicator, as in GMM for a quantile) or a kink at
    theta0 give 0 there while the right Jacobian (a density estimate, a subgradient) is not. When the moment
    moves over a step H = 1e4 h on either side (it is not a structural zero, which stays flat on both sides and
    is still compared), those entries are left out of the comparison and give a warning, never a refusal; the
    other entries of the column are still checked. An entry of the user's that also moves the moments by less
    than rounding (a true zero with noise) passes silently.
    Returns {"refuse", "warn", "col", "gap", "ref_err", "flat"} (flat: the columns whose differences are 0)."""
    th0 = np.asarray(th0, float)
    steps0 = _gmm_pilot_steps(th0, th0) if steps0 is None else np.asarray(steps0, float)
    tiny = np.finfo(float).tiny
    out = {"refuse": False, "warn": None, "col": None, "gap": 0.0, "ref_err": 0.0, "flat": []}
    unchecked, flat = [], []
    for j in range(len(th0)):
        h = steps0[j]
        d1 = _fd_column(gbar, th0, j, h, lo, hi)
        d2 = _fd_column(gbar, th0, j, h / 4, lo, hi)
        du = Du[:, j]
        if not (np.all(np.isfinite(d1)) and np.all(np.isfinite(d2))):
            unchecked.append(j)
            continue
        # each moment (row) on the scale of its own contributions at theta0, so the units of one moment do not hide
        # another (review r4, S1-b: with rms(G0) over all moments a Poisson block of 1e12 swallowed a wrong sign in a
        # unit-scale block; moments in units of 1e10 did the same)
        gs = np.asarray(g_scale, float) * np.ones(len(d1))
        gs = np.where(np.isfinite(gs) & (gs > 0), gs, 1.0)
        d1, d2, du = d1 / gs, d2 / gs, du / gs
        res = 4 * np.finfo(float).eps                               # rounding of a moment over the step, its scale
        dead = (np.abs(d1) * h <= res) & (np.abs(d2) * h / 4 <= res)
        if np.any(dead):
            # a structural zero (the moment does not depend on theta_j) stays flat on both sides however far one looks;
            # a step function or a kink moves somewhere. Probe outward geometrically up to the scale of the parameter
            # (a fixed 1e4 h found no data point with few observations and refused a right kernel-density Jacobian;
            # review r4, S2-a)
            dead = dead & _gmm_probe_moves(gbar, th0, j, h, lo, hi, gs)
        if np.any(dead & (np.abs(du) * h > res)):
            flat.append(j)
        keep = ~dead
        if not keep.any():
            continue
        d1, d2, du = d1[keep], d2[keep], du[keep]
        # floor: the smallest derivative that moves a moment by more than rounding over the step (a true zero column
        # with noise in the analytic one is not a disagreement; review r3, S3)
        scale = max(float(np.max(np.abs(d1))), float(np.max(np.abs(du))), res / h, tiny)
        gap = float(np.max(np.abs(du - d1))) / scale
        ref_err = float(np.max(np.abs(d1 - d2))) / scale
        if gap > tol and (out["col"] is None or gap > out["gap"]):
            out.update(col=j, gap=gap, ref_err=ref_err, refuse=ref_err <= ref_tol)
    if out["col"] is not None and not out["refuse"]:
        out["warn"] = (f"jacobian disagrees with finite differences at theta0 in column {out['col']} (gap "
                       f"{out['gap']:.3g} of its largest entry), but the finite differences are unstable there "
                       f"(steps h and h/4 differ by {out['ref_err']:.1g}), so the check is not conclusive")
    elif unchecked and out["col"] is None:
        out["warn"] = (f"jacobian not checked in column(s) {unchecked}: the moments are not finite next to theta0 "
                       "(within the bounds)")
    out["flat"] = flat
    if flat and out["col"] is None and out["warn"] is None:
        out["warn"] = (f"jacobian not checked in column(s) {flat}: the finite differences of the mean moment are "
                       "0 at theta0 (up to rounding) where your jacobian is not; the moments may be step functions "
                       "(indicators, quantiles) or have a kink at theta0, where finite differences are no reference")
    return out


def _gmm_inv(S):
    """Inverse of a covariance that should be positive definite; (inverse, smallest eigenvalue, ok).

    S is first put on the scale of its correlation matrix (each moment divided by its own standard deviation), so
    the check and the floor do not depend on the units of the moments: a moment that is a large offset times a
    constant plus a small variation made S look singular on its own scale and moved the estimate (v2.8.0 review r2;
    the inverse is the same when nothing is floored). The smallest eigenvalue returned is that of the correlation
    matrix. When it is not positive definite (a truncated kernel can give negative eigenvalues), the eigenvalues of
    the correlation matrix are floored at 1e-10 times the largest absolute one, so the weight stays positive
    definite."""
    d = np.sqrt(np.abs(np.diag(S)))
    d = np.where(d > 0, d, 1.0)
    C = S / np.outer(d, d)
    ev, vec = np.linalg.eigh((C + C.T) / 2)
    top = max(float(np.abs(ev).max()), np.finfo(float).tiny)
    ok = bool(ev.min() > 1e-12 * top)
    fl = ev if ok else np.maximum(ev, 1e-10 * top)
    inv = (vec / fl) @ vec.T / np.outer(d, d)
    return (inv + inv.T) / 2, float(ev.min()), ok


def gmm(moments, theta0, args=(), names=None, steps: str = "two-step", weight=None, hac_lags: int | None = None,
        kernel: str = "bartlett", center: bool = True, jacobian=None, bounds=None, maxiter: int = 100,
        tol: float = 1e-8, alpha: float = ALPHA, show: bool = True) -> dict:
    """Generalized method of moments from moment conditions E[g_i(theta)] = 0 that the user writes.

    Returns a dict (v2.8.0: keys exact):
      "estimand"   "theta"
      "method"     str, e.g. "GMM (two-step, Bartlett HAC with 4 lags)"
      "params"     Series of estimates, indexed by names
      "table"      DataFrame indexed by names, columns ["estimate", "se", "z", "p", "ci_low", "ci_high"]
      "cov"        DataFrame, covariance of the estimates (already divided by n)
      "j"          float, Hansen's J = n g' W g at the estimate (None with steps="one-step": not a test there)
      "j_df"       int, m - k (0 when exactly identified: J is 0 by construction, not a test)
      "j_p"        float, RIGHT-tail p-value, chi2.sf(J, m - k); this is the p-value
      "j_cdf"      float, chi2.cdf(J, m - k) = 1 - j_p: what some old tables print as "Prob" (not a p-value)
      "n", "m", "k"  observations, moments, parameters
      "weight"     ndarray m x m, final weighting matrix; "S" ndarray m x m, long-run covariance at the estimate
      "jacobian"   ndarray m x k, d mean(g) / d theta' at the estimate
      "objective"  float, mean(g)' W mean(g) at the estimate
      "converged"  bool; "iterations" int (weight updates)
      "rank_stat", "rank_df", "rank_p"  Cragg-Donald (1997) minimum-Wald test of rank(D) < k, chi2(m - k + 1);
                   a test of the rank only: rejecting it does not show strong identification (it grows with n
                   under any alternative; not a first-stage F). NaN when the moments are undefined next to the estimate
      "used"       dict, keys ["steps", "first_weight", "hac_lags", "kernel", "center", "bounds", "n", "m", "k",
                   "iterations", "converged", "rank_test_cov", "rank_test_ac1", "skill_version"]
      "warnings"   list of dicts with keys ["level", "message", "action"]

    moments(theta, *args) returns the n x m matrix of moment contributions g_i(theta), one row per
    observation (not their mean), e.g. for an Euler equation with instruments z_t:
    (beta * c_growth**(-gamma) * R - 1)[:, None] * Z. jacobian(theta, *args), optional, returns the
    m x k Jacobian of the MEAN moment (rows = moments, columns = parameters). It is checked once at theta0,
    column by column, against finite differences inside the bounds: refused when a column clearly disagrees
    (e.g. transposed, or one column wrong), a warning when the finite differences are unstable or zero there (a
    kink, a step function; the rank test is then not computed). Without it, central differences whose step is
    chosen per parameter by stability (among scales |theta|, |theta0|, 1 and that of the response, the one where
    steps h and h/2 agree best): it depends neither on the units of theta nor on how far theta0 is from the
    estimate. A trial step of the search where the moments are NaN is skipped (the trust region shrinks); theta0
    and the estimate must be finite.
    steps    "two-step" (default): first step with `weight` (identity if None; the first step depends on
             the units of the moments: scale them), then W = S^-1 at the first-step estimate; "iterated":
             repeat until the estimate stops moving (Hansen, Heaton and Yaron 1996); "one-step": `weight`
             only, sandwich SEs, and no J (J needs the efficient weight).
    hac_lags None: S = covariance of the contributions (independent observations, any heteroskedasticity);
             an integer L: HAC with L lags, `kernel` "bartlett" (Newey-West 1987, always positive
             semidefinite) or "truncated" (equal weights, Hansen-Singleton 1982 with a known MA(L) error;
             may be indefinite). Multi-period Euler equations have MA(n - 1) errors: hac_lags >= n - 1.
    center   True: S from the demeaned contributions (Hall 2000; keeps J consistent under misspecification).
    bounds   optional list of (low, high) per parameter (None for no bound), e.g. a discount factor in (0, 1].
    Standard errors: (D'WD)^-1 D'WSWD (D'WD)^-1 / n with the final W and S at the estimate (= (D'S^-1 D)^-1 / n
    at the efficient weight when S is re-estimated there).
    J: n mean(g)'W mean(g) with the weight of the last step; chi2(m - k) under correct specification. Its p-value
    is the RIGHT tail (j_p). "Prob" columns in Hansen-Singleton (1982, 1983) are the CDF (j_cdf): a Prob of 0.967
    is p = 0.033, a rejection at 5%.
    Findings: exactly identified; a parameter at or within one SE of a bound (J, Wald and LR leave the chi2;
    Andrews 1999); a Jacobian whose rank test does not reject rank < k (may not be identified); fewer than 10
    observations per moment; S not positive definite; no convergence (report no number). The rank test (with lags,
    always on a Bartlett long-run covariance) only detects rank deficiency: identification can be weak even when it
    rejects. Weakness is judged by the criterion over a grid or an identification-robust S test (Stock and Wright
    2000), not by a cut on this statistic. Try several starting values: GMM criteria of
    nonlinear models have flat valleys (gamma and beta of an Euler equation trade off)."""
    from reporting import SKILL_VERSION, _finding, _print_findings
    if steps not in _GMM_STEPS:
        raise ValueError(f"steps must be one of {_GMM_STEPS}")
    if kernel not in _GMM_KERNELS:
        raise ValueError(f"kernel must be one of {_GMM_KERNELS}")
    if hac_lags is not None and (int(hac_lags) != hac_lags or hac_lags < 0):
        raise ValueError("hac_lags must be None or a non-negative integer")
    theta0 = np.atleast_1d(np.asarray(theta0, float)).copy()
    k = len(theta0)
    names = list(names) if names is not None else [f"theta{j}" for j in range(k)]
    if len(names) != k:
        raise ValueError("names must have one entry per parameter")

    def G_raw(th):
        G = np.asarray(moments(th, *args), float)
        return G[:, None] if G.ndim == 1 else G

    def G_of(th):
        G = G_raw(th)
        if not np.all(np.isfinite(G)):
            raise ValueError(f"moments returned NaN or inf at theta = {np.array2string(np.asarray(th), precision=6)}: "
                             "the search reached values where they are undefined; set bounds or other starting values")
        return G

    try:
        G0 = G_of(theta0)
    except ValueError:
        raise ValueError("moments are not finite at theta0 (missing values in the data, or a bad start)") from None
    n, m = G0.shape
    if m < k:
        raise ValueError(f"{m} moments for {k} parameters: the model is not identified (need m >= k)")
    if hac_lags is not None and hac_lags >= n:
        raise ValueError("hac_lags must be smaller than the number of observations")

    if bounds is not None:
        if len(bounds) != k:
            raise ValueError("bounds must have one (low, high) pair per parameter")
        lo = np.array([-np.inf if b is None or b[0] is None else b[0] for b in bounds], float)
        hi = np.array([np.inf if b is None or b[1] is None else b[1] for b in bounds], float)
    else:
        lo, hi = np.full(k, -np.inf), np.full(k, np.inf)

    def gbar(th):
        # no error inside the search: a trial step where the moments are undefined returns NaN and the trust region
        # of least_squares shrinks (review r2, S2-1); theta0 and the estimate are checked by G_of. Nor numpy's
        # RuntimeWarnings of that step (the user's 1/theta at 0, then the matmul of NaN): noise in the user's console
        # (review r3, S3)
        with np.errstate(all="ignore"):
            return G_raw(th).mean(0)

    def D_user(th):
        Du = np.asarray(jacobian(th, *args), float)
        if Du.ndim == 1 and k == 1 and Du.shape[0] == m:
            Du = Du[:, None]
        if Du.shape != (m, k):
            raise ValueError(f"jacobian must return an array of shape (m, k) = ({m}, {k}) (rows = moments, "
                             f"columns = parameters); it returned shape {Du.shape}")
        return Du

    step_cache = {}

    def D_of(th, final=False):
        if jacobian is not None:
            return D_user(th)
        # the steps are chosen again only when a parameter has moved by a factor of 2 (or changed sign, or left 0)
        # since they were last chosen, and always at the estimate: choosing them takes ~10 evaluations of the moments
        # per parameter, and at every Jacobian of the search a fit made 6x as many calls (review r4, S3-1)
        th = np.asarray(th, float)
        ref = step_cache.get("th")
        if final or ref is None or np.any((np.abs(th) > 2 * np.abs(ref)) | (np.abs(th) < 0.5 * np.abs(ref))
                                          | (np.sign(th) != np.sign(ref))):
            step_cache["th"], step_cache["h"] = th.copy(), _gmm_steps(G_raw, th, theta0)
        return _jacobian(gbar, th, step_cache["h"])

    warns = []
    flat_cols = []
    gs0 = np.sqrt(np.mean(G0 ** 2, axis=0))
    gs0 = np.where(np.isfinite(gs0) & (gs0 > 0), gs0, 1.0)
    if jacobian is None:
        # step-function moments (indicators, quantiles) have numerical derivatives of exactly 0: the fit stayed at
        # theta0 with SE 0 and converged=True (v2.8.0 review r4, S2-e). A column that is 0 at theta0 but whose moments
        # move further out is such a function; one that never moves is a parameter outside the moments (rank test).
        h0 = _gmm_steps(G_raw, theta0, theta0)
        D0 = _jacobian(gbar, theta0, h0)
        for j in range(k):
            if not np.any(D0[:, j]) and np.any(_gmm_probe_moves(gbar, theta0, j, h0[j], lo, hi, gs0)):
                raise ValueError(f"the moments do not change when {names[j]} moves a little from its start, but do "
                                 "further out: they "
                                 "are step functions (indicators, quantiles), whose numerical derivatives are 0; pass "
                                 "jacobian= (e.g. a kernel density estimate times the instruments) and a start "
                                 "inside the data")
    if jacobian is not None:
        chk = _gmm_check_jacobian(gbar, D_user(theta0), theta0, lo, hi, steps0=_gmm_steps(G_raw, theta0, theta0),
                                  g_scale=gs0)
        if chk["refuse"]:
            raise ValueError(f"jacobian disagrees with finite differences at theta0 in column {chk['col']} "
                             f"({names[chk['col']]}): largest gap {chk['gap']:.3g} of that column's largest entry, "
                             f"while the finite differences agree with themselves to {chk['ref_err']:.1g}; check it, "
                             "and its orientation (rows = moments, columns = parameters)")
        flat_cols = chk["flat"]
        if chk["warn"]:
            warns.append(_finding("warning", chk["warn"],
                                  "the jacobian you passed was used; check it at theta0, or run without it and compare"))

    def fit(W, start):
        # n * gbar' W gbar = || sqrt(n) L' gbar ||^2 with W = L L': least squares on r(theta) = sqrt(n) L' gbar
        L = np.linalg.cholesky((W + W.T) / 2)
        rn = np.sqrt(n)
        start = np.clip(start, np.nextafter(lo, np.inf), np.nextafter(hi, -np.inf))
        def jac_fit(th):
            # the Jacobian is taken only at accepted points; non-finite there (a difference step crossed into a
            # region where the moments are undefined) is not recoverable by the trust region
            D = D_of(th)
            if not np.all(np.isfinite(D)):
                raise ValueError(f"moments returned NaN or inf at theta = "
                                 f"{np.array2string(np.asarray(th), precision=6)} plus a difference step (in the "
                                 "Jacobian): the search "
                                 "reached values where they are undefined; set bounds or other starting values")
            return rn * (L.T @ D)

        def resid(th):
            with np.errstate(all="ignore"):
                return rn * (L.T @ gbar(th))

        res = optimize.least_squares(resid, start, jac=jac_fit, bounds=(lo, hi),
                                     method="trf", x_scale="jac", xtol=1e-12, ftol=1e-14, gtol=1e-12,
                                     max_nfev=200 * (k + 1))
        return res.x, bool(res.status > 0)

    W1 = np.eye(m) if weight is None else np.asarray(weight, float)
    if W1.shape != (m, m):
        raise ValueError(f"weight must be {m} x {m}")
    try:
        np.linalg.cholesky((W1 + W1.T) / 2)
    except np.linalg.LinAlgError:
        raise ValueError("weight must be positive definite") from None
    th, conv = fit(W1, theta0)
    W, it = W1, 0
    S_pd_ok, s_min_all = True, np.inf
    if steps in ("two-step", "iterated"):
        cap = 1 if steps == "two-step" else maxiter
        for it in range(1, cap + 1):
            S_w = _gmm_S(G_of(th), hac_lags, kernel, center)
            W, s_w, ok = _gmm_inv(S_w)
            S_pd_ok &= ok
            s_min_all = min(s_min_all, s_w)
            th_new, conv = fit(W, th)
            moved = np.max(np.abs(th_new - th) / (1.0 + np.abs(th)))
            th = th_new
            if steps == "iterated" and moved < tol:
                break
        else:
            if steps == "iterated":
                conv = False
                warns.append(_finding("warning", f"iterated GMM did not settle in {maxiter} weight updates",
                                      "report the two-step estimate, or raise maxiter and check the starting values"))
    Gh = G_of(th)
    g = Gh.mean(0)
    D = D_of(th, final=True)
    S = _gmm_S(Gh, hac_lags, kernel, center)
    Sinv_fin, s_min, ok = _gmm_inv(S)
    S_pd_ok &= ok
    s_min_all = min(s_min_all, s_min)
    if not ok:                                   # the floored S, the one whose inverse was used
        S = np.linalg.inv(Sinv_fin)
    # pseudo-inverse on the scale of the parameters (Jacobi): a regressor stored with an offset makes D'WD
    # ill-conditioned (1.4e16 at offset 1e4) and the plain pinv cut a direction, moving the SE by 0.2% (review r2)
    M_dwd = D.T @ W @ D
    q_dwd = np.sqrt(np.abs(np.diag(M_dwd)))
    q_dwd = np.where(q_dwd > 0, q_dwd, 1.0)
    A = np.linalg.pinv(M_dwd / np.outer(q_dwd, q_dwd)) / np.outer(q_dwd, q_dwd)
    V = A @ D.T @ W @ S @ W @ D @ A / n
    V = (V + V.T) / 2
    tab = _table(th, V, names, alpha)
    se = tab["se"].to_numpy()
    df_j = m - k
    if steps == "one-step":
        J = None
        j_p = j_cdf = None
    else:
        J = float(n * g @ W @ g)
        j_p = float(stats.chi2.sf(J, df_j)) if df_j > 0 else None
        j_cdf = float(stats.chi2.cdf(J, df_j)) if df_j > 0 else None
    # findings
    if not conv:
        warns.append(_finding("warning", "the optimizer did not report convergence",
                              "do not report these numbers: try other starting values, rescale the moments or set bounds"))
    if df_j == 0:
        warns.append(_finding("notice", f"exactly identified ({m} moments, {k} parameters): J is 0 by construction",
                              "J tests nothing here; add moments (instruments) to test the model's restrictions"))
    elif steps == "one-step":
        warns.append(_finding("notice", "one-step GMM: no J statistic (J needs the efficient weight S^-1)",
                              "use steps='two-step' or 'iterated' to test the overidentifying restrictions"))
    for j in range(k):
        for side, b in (("lower", lo[j]), ("upper", hi[j])):
            if np.isfinite(b) and np.isfinite(se[j]) and abs(th[j] - b) <= max(se[j], 1e-8 * (1 + abs(b))):
                warns.append(_finding(
                    "warning", f"{names[j]} = {th[j]:.6g} is within one SE ({se[j]:.3g}) of its {side} bound {b:g}",
                    "near a bound the t-ratio, the CI and the chi2 references of J, Wald and LR are not valid "
                    "(Andrews 1999); report the bound as binding and use a boundary-aware test or a bootstrap"))
    if flat_cols and np.array_equal(th, np.clip(theta0, np.nextafter(lo, np.inf), np.nextafter(hi, -np.inf))):
        warns.append(_finding(
            "warning", "the estimate is exactly theta0: with step-function moments the search could not move from the "
            "start (the mean moment is flat between data points)",
            "start inside the data (e.g. at a sample quantile) and compare with a grid of the criterion"))
    if flat_cols:
        # the rank test differentiates each contribution numerically: with step-function moments those differences
        # are 0 and the statistic would say "not identified" for any data (v2.8.0 review r3, S2-5)
        rank_stat, rank_df, rank_ac1, rank_capped = float("nan"), m - k + 1, float("nan"), False
    else:
        rank_stat, rank_df, rank_ac1, rank_capped = _gmm_rank_test(G_of, th, hac_lags, _gmm_steps(G_raw, th, theta0))
    if rank_df is None:
        rank_df = m - k + 1
    if rank_capped:
        warns.append(_finding(
            "notice", f"the rank test did not search every subspace in which some moments do not depend on the "
            f"parameters (past {_RANK_MAX_SUBSPACES} subspaces or {_RANK_BUDGET_S:g} s; designs with many group "
            "dummies have many): the statistic is an upper bound and may overstate identification",
            "check identification also by the criterion over a grid of the parameters"))
    rank_p = float(stats.chi2.sf(rank_stat, rank_df))              # nan when the test was not computed
    if not np.isfinite(rank_p) and flat_cols:
        warns.append(_finding(
            "notice", "the rank test of the Jacobian was not computed: it differentiates the moments numerically, and "
            "their finite differences are 0 (step functions or a kink)",
            "judge identification by your jacobian (its rank and size relative to the SEs) and by the criterion over "
            "a grid of the parameters"))
    elif not np.isfinite(rank_p):
        warns.append(_finding(
            "notice", "the rank test of the Jacobian was not computed: the moments are undefined next to the estimate",
            "check identification by the criterion over a grid of the parameters"))
    elif not rank_p < alpha:
        warns.append(_finding(
            "warning", f"the parameters may not be identified: the rank test of the Jacobian does not reject "
            f"rank < {k} (Cragg-Donald minimum Wald {rank_stat:.3g}, df {rank_df}, p = {rank_p:.3g})",
            "do not report the estimates or their SEs; add informative moments or fix a parameter from theory"))
    # persistence of the derivatives (review r2, S2-4): with persistent dg_i/dtheta' the iid or short-HAC variance of
    # the rank test is too small and its size is well above 5%. Warn when the largest lag-1 autocorrelation passes the
    # cut AND the statistic, divided by the AR(1) factor of the long-run variance the kernel misses, no longer rejects.
    # The row order is read as time order: a cross-section sorted by a variable gives a lag-1 autocorrelation of 0.97
    # with iid data, so the text says the warning holds only for rows in time order (review r3, S2-3; decision of the
    # user, 2026-11-26: always warn, with that condition in the text)
    ac_cut = 0.4 if not hac_lags else 0.6
    if np.isfinite(rank_p) and rank_p < alpha and np.isfinite(rank_ac1) and rank_ac1 >= ac_cut:
        r_ = min(rank_ac1, 0.99)
        L_ = int(hac_lags or 0)
        f_res = ((1 + r_) / (1 - r_)) / (1 + 2 * sum((1 - j / (L_ + 1)) * r_ ** j for j in range(1, L_ + 1)))
        p_defl = float(stats.chi2.sf(rank_stat / f_res, rank_df))
        if p_defl >= alpha:
            lag_txt_r = "without hac_lags" if not hac_lags else f"with hac_lags = {hac_lags}"
            warns.append(_finding(
                "warning", f"the rank test rejects rank < {k} (p = {rank_p:.3g}), but the derivatives of the moments "
                f"are persistent (largest lag-1 autocorrelation {rank_ac1:.2f}) and it was computed {lag_txt_r}: its "
                f"size can be well above 5% (with n = 240, about 0.20 at 0.5 without lags and 0.40 at 0.9 with 4 "
                f"lags), and divided by the long-run variance factor it misses ({f_res:.3g}) the statistic gives "
                f"p = {p_defl:.3g}; if the rows are in time order, the rejection is not reliable evidence of "
                f"identification (in a cross-section the row order means nothing and this warning does not apply)",
                "for a time series: pass hac_lags (or more lags), and judge identification also by the precision of "
                "the estimates and by the criterion over a grid of the parameters; the check sees only lag-1 "
                "persistence, so its absence does not validate a rejection with seasonal persistence"))
    if n < 10 * m:
        warns.append(_finding("notice", f"{n} observations for {m} moments (< 10 per moment): S is estimated with "
                              "noise and J tends to over-reject",
                              "use fewer moments (fewer instrument lags) or report J with that caveat"))
    if not S_pd_ok:
        warns.append(_finding("warning", f"S is not positive definite (smallest eigenvalue of its correlation matrix "
                              f"{s_min_all:.3g}): those eigenvalues were floored at 1e-10 of the largest",
                              "use kernel='bartlett' (always positive semidefinite) or drop redundant moments"))
    lag_txt = "" if hac_lags is None else f", {kernel.capitalize()} HAC with {hac_lags} lags"
    method = f"GMM ({steps}{lag_txt})"
    used = {"steps": steps, "first_weight": "identity" if weight is None else "user", "hac_lags": hac_lags,
            "kernel": kernel if hac_lags is not None else None, "center": center,
            "bounds": None if bounds is None else [tuple(b) if b is not None else None for b in bounds],
            "n": n, "m": m, "k": k, "iterations": it, "converged": conv,
            "rank_test_cov": "iid" if not hac_lags else f"Bartlett HAC with {hac_lags} lags",
            "rank_test_ac1": None if not np.isfinite(rank_ac1) else float(rank_ac1),
            "skill_version": SKILL_VERSION}
    out = {"estimand": "theta", "method": method, "params": pd.Series(th, index=names), "table": tab,
           "cov": pd.DataFrame(V, index=names, columns=names), "j": J, "j_df": df_j, "j_p": j_p, "j_cdf": j_cdf,
           "n": n, "m": m, "k": k, "weight": W, "S": S, "jacobian": D, "objective": float(g @ W @ g),
           "converged": conv, "iterations": it, "rank_stat": rank_stat, "rank_df": rank_df,
           "rank_p": rank_p, "used": used, "warnings": warns}
    if show:
        print(f"\n{method}: n = {n}, {m} moments, {k} parameters")
        print(tab.to_string(float_format=lambda v: f"{v:.6g}"))
        if J is not None and df_j > 0:
            print(f"Hansen J = {J:.4f}, df = {df_j}, p (right tail) = {j_p:.4f}"
                  f"   [chi2 CDF = {j_cdf:.4f}: what old tables print as 'Prob'; not the p-value]")
        print(f"Identification (rank of the Jacobian): Cragg-Donald minimum Wald = {rank_stat:.4g}, df = {rank_df}, "
              f"p = {rank_p:.4g} (tests rank < {k} only; a rejection does not rule out weak identification)")
        _print_findings(warns)
    return out


def lr_test(unrestricted: dict, restricted: dict, df: int | None = None, boundary: bool = False,
            alpha: float = ALPHA, show: bool = True) -> dict:
    """Likelihood-ratio test between two nested models fitted by `mle` on the SAME sample, for restrictions of
    any form (trinity_test handles only linear R theta = r). Example: Hansen and Singleton (1983) test the
    cross-equation restrictions of an asset-pricing model against an unrestricted VAR.

    Returns a dict (v2.8.0: keys exact):
      "estimand"  "LR"
      "statistic" float, 2 (loglik_unrestricted - loglik_restricted)
      "df"        int (default: k_unrestricted - k_restricted)
      "p"         float, RIGHT-tail p-value (with boundary=True and df=1, the 50:50 mixture of chi2(0) and chi2(1))
      "chi2_cdf"  float, chi2.cdf(statistic, df) = 1 - p without boundary: what H-S print in parentheses
      "reject"    bool, p < alpha
      "used"      dict, keys ["df", "boundary", "n", "alpha", "skill_version"]
      "warnings"  list of dicts with keys ["level", "message", "action"]

    unrestricted, restricted : results of `mle` (they carry "loglik", "n", "k", "converged"). Both
    log-likelihoods must be evaluated on the same observations: an LR between different effective samples
    (e.g. a VAR with more lags losing initial rows) has no chi2 distribution. The function refuses fits with a
    different n; it cannot see whether two fits with the same n used the same rows: check that yourself.
    boundary : True when the null puts a parameter on the edge of its space (a variance of 0, an ARCH
    coefficient of 0): with one such parameter the reference is the 50:50 mixture (Self and Liang 1987);
    with more, the mixture weights depend on the model and the chi2 p-value is conservative."""
    from reporting import SKILL_VERSION, _finding, _print_findings
    for name, f in (("unrestricted", unrestricted), ("restricted", restricted)):
        if not isinstance(f, dict) or "loglik" not in f or "n" not in f:
            raise ValueError(f"{name} must be an mle() result (a dict with 'loglik' and 'n')")
    if int(unrestricted["n"]) != int(restricted["n"]):
        raise ValueError(f"the fits use different samples (n = {unrestricted['n']} and {restricted['n']}): "
                         "evaluate both on the same observations")
    if df is None:
        df = int(unrestricted.get("k", 0)) - int(restricted.get("k", 0))
    if int(df) != df or df < 1:
        raise ValueError("df must be a positive integer (the number of independent restrictions)")
    df = int(df)
    lr = 2.0 * (float(unrestricted["loglik"]) - float(restricted["loglik"]))
    warns = []
    for name, f in (("unrestricted", unrestricted), ("restricted", restricted)):
        if f.get("converged") is False:
            warns.append(_finding("warning", f"the {name} fit did not converge",
                                  "do not report the LR until both fits converge"))
    if lr < -1e-6 * max(1.0, abs(float(unrestricted["loglik"]))):
        warns.append(_finding("warning", f"the restricted fit has the higher log-likelihood (LR = {lr:.4g} < 0): "
                              "the unrestricted optimum was not found or the models are not nested",
                              "refit the unrestricted model from the restricted estimates"))
    lr_pos = max(lr, 0.0)
    cdf = float(stats.chi2.cdf(lr_pos, df))
    if boundary and df == 1:
        p = float(0.5 * stats.chi2.sf(lr_pos, 1)) if lr_pos > 0 else 1.0
        warns.append(_finding("notice", "boundary null with one restriction: p from the 50:50 mixture of chi2(0) "
                              "and chi2(1) (Self and Liang 1987), half the chi2(1) p-value",
                              "say that the null is on the boundary when you report the test"))
    else:
        p = float(stats.chi2.sf(lr_pos, df))
        if boundary:
            warns.append(_finding("warning", f"boundary null with {df} restrictions: the chi2({df}) p-value is "
                                  "conservative (the true reference is a mixture whose weights depend on the model)",
                                  "report it as an upper bound on the p-value, or bootstrap the LR under the null"))
    used = {"df": df, "boundary": boundary, "n": int(unrestricted["n"]), "alpha": alpha,
            "skill_version": SKILL_VERSION}
    out = {"estimand": "LR", "statistic": lr, "df": df, "p": p, "chi2_cdf": cdf, "reject": bool(p < alpha),
           "used": used, "warnings": warns}
    if show:
        print(f"\nLR = {lr:.4f}, df = {df}, p (right tail) = {p:.4f}   [chi2 CDF = {cdf:.4f}: not the p-value]")
        _print_findings(warns)
    return out
