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


def _gmm_rank_test(G_of, th, lags):
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
    Returns (stat, df); (nan, df) when the moments are not finite next to the estimate."""
    th = np.asarray(th, float)
    h = _steps(th)
    k = len(th)
    cols = []
    try:
        for j in range(k):
            e = np.zeros(k)
            e[j] = h[j]
            cols.append((G_of(th + e) - G_of(th - e)) / (2 * h[j]))
    except ValueError:
        return float("nan"), None
    Di = np.stack(cols, axis=2)                                    # n x m x k
    n, m, _ = Di.shape
    if not np.all(np.isfinite(Di)):
        return float("nan"), m - k + 1
    Dbar = Di.mean(0)
    flat = Di.reshape(n, m * k)                                    # column j*k + a: moment j, parameter a
    Om = _gmm_S(flat, lags, "bartlett", True).reshape(m, k, m, k)
    M2 = (flat.T @ flat / n).reshape(m, k, m, k)                   # uncentred second moments: the scale of D_i

    def wald(c):
        c = np.asarray(c, float)
        nc = np.linalg.norm(c)
        if not np.isfinite(nc) or nc == 0:
            return np.inf
        c = c / nc
        V = np.einsum("iajb,a,b->ij", Om, c, c)
        ref = float(np.trace(np.einsum("iajb,a,b->ij", M2, c, c))) / m
        dm = Dbar @ c
        ev, vec = np.linalg.eigh((V + V.T) / 2)
        tol = 1e-12 * max(float(np.abs(ev).max()), ref, np.finfo(float).tiny)
        proj = vec.T @ dm
        keep = ev > tol
        if np.any(np.abs(proj[~keep]) > 1e-6 * np.sqrt(max(ref, np.finfo(float).tiny))):
            return np.inf                                          # the moments move deterministically along c
        return float(n * np.sum(proj[keep] ** 2 / ev[keep]))

    if k == 1:
        return wald([1.0]), m
    rng = np.random.default_rng(0)
    if k == 2:
        grid = np.linspace(0.0, np.pi, 361)
        vals = [wald([np.cos(a), np.sin(a)]) for a in grid]
        i = int(np.argmin(vals))
        res = optimize.minimize_scalar(lambda a: wald([np.cos(a), np.sin(a)]),
                                       bounds=(grid[max(i - 1, 0)], grid[min(i + 1, 360)]), method="bounded",
                                       options={"xatol": 1e-10})
        best = min(float(res.fun), float(vals[i]))
    else:
        # the statistic does not depend on |c|: unconstrained minimisation over c, from the axes and random starts
        starts = [np.eye(k)[j] for j in range(k)] + [rng.standard_normal(k) for _ in range(3 * k)]
        best = np.inf
        for c0 in starts:
            f0 = wald(c0)
            if not np.isfinite(f0):
                continue
            res = optimize.minimize(wald, c0, method="BFGS", options={"gtol": 1e-10 * max(1.0, f0), "maxiter": 400})
            best = min(best, float(res.fun), f0)
            res = optimize.minimize(wald, res.x, method="Nelder-Mead",
                                    options={"xatol": 1e-9, "fatol": 1e-10, "maxiter": 200 * k})
            best = min(best, float(res.fun))
    return best, m - k + 1


def _gmm_inv(S):
    """Inverse of a covariance that should be positive definite; (inverse, smallest eigenvalue, ok).

    When S is not positive definite (a truncated kernel can give negative eigenvalues), its eigenvalues are
    floored at 1e-10 times the largest absolute one before inverting, so the weight stays positive definite."""
    ev, vec = np.linalg.eigh(S)
    top = max(float(np.abs(ev).max()), np.finfo(float).tiny)
    ok = bool(ev.min() > 1e-12 * top)
    if ok:
        return np.linalg.inv(S), float(ev.min()), True
    fl = np.maximum(ev, 1e-10 * top)
    return (vec / fl) @ vec.T, float(ev.min()), False


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
                   "iterations", "converged", "rank_test_cov", "skill_version"]
      "warnings"   list of dicts with keys ["level", "message", "action"]

    moments(theta, *args) returns the n x m matrix of moment contributions g_i(theta), one row per
    observation (not their mean), e.g. for an Euler equation with instruments z_t:
    (beta * c_growth**(-gamma) * R - 1)[:, None] * Z. jacobian(theta, *args), optional, returns the
    m x k Jacobian of the MEAN moment (rows = moments, columns = parameters; checked once against central
    differences at theta0, and refused if it disagrees, e.g. transposed); without it, central differences.
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

    def G_of(th):
        G = np.asarray(moments(th, *args), float)
        if G.ndim == 1:
            G = G[:, None]
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

    def gbar(th):
        return G_of(th).mean(0)

    def D_user(th):
        Du = np.asarray(jacobian(th, *args), float)
        if Du.ndim == 1 and k == 1 and Du.shape[0] == m:
            Du = Du[:, None]
        if Du.shape != (m, k):
            raise ValueError(f"jacobian must return an array of shape (m, k) = ({m}, {k}) (rows = moments, "
                             f"columns = parameters); it returned shape {Du.shape}")
        return Du

    def D_of(th):
        if jacobian is not None:
            return D_user(th)
        return _jacobian(gbar, th)

    if jacobian is not None:
        Du0, Dn0 = D_user(theta0), _jacobian(gbar, theta0)
        gap = float(np.max(np.abs(Du0 - Dn0))) / max(float(np.max(np.abs(Dn0))), np.finfo(float).tiny)
        if not gap < 1e-3:
            raise ValueError(f"jacobian disagrees with central differences at theta0 (largest gap {gap:.3g} of the "
                             "largest entry): check it, and its orientation (rows = moments, columns = parameters)")

    if bounds is not None:
        if len(bounds) != k:
            raise ValueError("bounds must have one (low, high) pair per parameter")
        lo = np.array([-np.inf if b is None or b[0] is None else b[0] for b in bounds], float)
        hi = np.array([np.inf if b is None or b[1] is None else b[1] for b in bounds], float)
    else:
        lo, hi = np.full(k, -np.inf), np.full(k, np.inf)

    def fit(W, start):
        # n * gbar' W gbar = || sqrt(n) L' gbar ||^2 with W = L L': least squares on r(theta) = sqrt(n) L' gbar
        L = np.linalg.cholesky((W + W.T) / 2)
        rn = np.sqrt(n)
        start = np.clip(start, np.nextafter(lo, np.inf), np.nextafter(hi, -np.inf))
        res = optimize.least_squares(lambda th: rn * (L.T @ gbar(th)), start,
                                     jac=lambda th: rn * (L.T @ D_of(th)), bounds=(lo, hi), method="trf",
                                     x_scale="jac", xtol=1e-12, ftol=1e-14, gtol=1e-12, max_nfev=200 * (k + 1))
        return res.x, bool(res.status > 0)

    warns = []
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
    D = D_of(th)
    S = _gmm_S(Gh, hac_lags, kernel, center)
    Sinv_fin, s_min, ok = _gmm_inv(S)
    S_pd_ok &= ok
    s_min_all = min(s_min_all, s_min)
    if not ok:                                   # the floored S, the one whose inverse was used
        S = np.linalg.inv(Sinv_fin)
    A = np.linalg.pinv(D.T @ W @ D)
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
    rank_stat, rank_df = _gmm_rank_test(G_of, th, hac_lags)
    if rank_df is None:
        rank_df = m - k + 1
    rank_p = float(stats.chi2.sf(rank_stat, rank_df))              # nan when the test was not computed
    if not np.isfinite(rank_p):
        warns.append(_finding(
            "notice", "the rank test of the Jacobian was not computed: the moments are undefined next to the estimate",
            "check identification by the criterion over a grid of the parameters"))
    elif not rank_p < alpha:
        warns.append(_finding(
            "warning", f"the parameters may not be identified: the rank test of the Jacobian does not reject "
            f"rank < {k} (Cragg-Donald minimum Wald {rank_stat:.3g}, df {rank_df}, p = {rank_p:.3g})",
            "do not report the estimates or their SEs; add informative moments or fix a parameter from theory"))
    if n < 10 * m:
        warns.append(_finding("notice", f"{n} observations for {m} moments (< 10 per moment): S is estimated with "
                              "noise and J tends to over-reject",
                              "use fewer moments (fewer instrument lags) or report J with that caveat"))
    if not S_pd_ok:
        warns.append(_finding("warning", f"S is not positive definite (smallest eigenvalue {s_min_all:.3g}): "
                              "its eigenvalues were floored at 1e-10 of the largest",
                              "use kernel='bartlett' (always positive semidefinite) or drop redundant moments"))
    lag_txt = "" if hac_lags is None else f", {kernel.capitalize()} HAC with {hac_lags} lags"
    method = f"GMM ({steps}{lag_txt})"
    used = {"steps": steps, "first_weight": "identity" if weight is None else "user", "hac_lags": hac_lags,
            "kernel": kernel if hac_lags is not None else None, "center": center,
            "bounds": None if bounds is None else [tuple(b) if b is not None else None for b in bounds],
            "n": n, "m": m, "k": k, "iterations": it, "converged": conv,
            "rank_test_cov": "iid" if not hac_lags else f"Bartlett HAC with {hac_lags} lags",
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
    (e.g. a VAR with more lags losing initial rows) has no chi2 distribution, and the function refuses it.
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
