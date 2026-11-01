"""
Heterogeneous treatment effects (CATE) and treatment policies: for whom is the effect larger,
is that heterogeneity real, and whom should we treat?

Inspired by EconML (PyWhy) and grf, written with numpy/scikit-learn only so it runs wherever the
rest of the skill runs. EconML's estimators can be used instead when installed (see
references/heterogeneous-effects.md); the validation tools here accept any vector of
out-of-fold CATE predictions.

Estimation  (X = effect modifiers, W = extra controls used only for confounding)
    cate_fit(df, y, t, X, W, method)          cross-fitted nuisances + CATE model
        method "dr"          DR-learner, linear final stage, valid CIs (Kennedy 2023; binary t)
               "linear_dml"  R-learner / LinearDML, valid CIs, binary or continuous t (Nie & Wager 2021)
               "forest"      DR-learner with a random-forest final stage (flexible, no pointwise CIs)
               "t_learner", "s_learner", "x_learner"   meta-learners (Kuenzel et al. 2019), no CIs
    cate_predict(res, df_new)                 tau(x) for new units, with CIs for linear final stages

Validation  (always on out-of-fold predictions, never on in-sample fits)
    blp_test(res)                             best linear predictor / calibration test: is the predicted
                                              heterogeneity real? (Chernozhukov, Demirer, Duflo &
                                              Fernandez-Val 2018; grf's test_calibration)
    gates(res, n_groups)                      sorted group average treatment effects + top - bottom test
    clan(res, df, covs)                       who are the most and least affected? (Holm-adjusted)
    toc(res)                                  targeting operator characteristic, AUTOC and Qini
                                              coefficient with bootstrap SEs (Yadlowsky et al. 2025)
    rscore(res) / compare_cate_models(dict)   R-loss score: does a model beat a constant effect?

Policy
    policy_tree(res, df, features, cost)      depth-1/2 treatment rule by exhaustive search on doubly
                                              robust scores (Athey & Wager 2021), learned on one half
                                              and valued on the other
    policy_predict(tree, df)                  apply the rule
    policy_value(res, treat, cost)            value of any rule vs treating everyone / no one

The ATE reported by cate_fit is always the cross-fitted AIPW estimate (binary t) or the average
of the linear CATE (continuous t): methods differ in tau(x), not in how the average is identified.
Identification is the same as for the ATE: unconfoundedness given (X, W) and overlap.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05
METHODS = ("dr", "linear_dml", "forest", "t_learner", "s_learner", "x_learner")


def _out(name, est, se, n, **extra):
    z = stats.norm.ppf(1 - ALPHA / 2)
    p = 2 * stats.norm.sf(abs(est / se)) if se > 0 else np.nan
    return {"estimand": name, "estimate": float(est), "se": float(se), "ci_low": float(est - z * se),
            "ci_high": float(est + z * se), "p": float(p), "n": int(n), **extra}


def _ols_hc1(Z, y):
    """OLS coefficients and HC1 covariance (no intercept added)."""
    ZtZ_inv = np.linalg.pinv(Z.T @ Z)
    b = ZtZ_inv @ Z.T @ y
    e = y - Z @ b
    n, k = Z.shape
    V = ZtZ_inv @ (Z.T * e ** 2) @ Z @ ZtZ_inv * n / max(n - k, 1)
    return b, V


def _default_learners(binary, seed):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    reg = RandomForestRegressor(n_estimators=300, min_samples_leaf=10, max_features=0.5, random_state=seed)
    if binary:
        ps = RandomForestClassifier(n_estimators=300, min_samples_leaf=10, max_features=0.5, random_state=seed)
    else:
        ps = RandomForestRegressor(n_estimators=300, min_samples_leaf=10, max_features=0.5, random_state=seed)
    return reg, ps


# =============================================================================
# Estimation
# =============================================================================
def cate_fit(df: pd.DataFrame, y: str, t: str, X: list[str], W: list[str] | None = None, method: str = "dr",
             n_folds: int = 5, seed: int = 0, outcome_learner=None, ps_learner=None, final_learner=None,
             clip: float = 0.01) -> dict:
    """Estimate tau(x) = E[Y(1) - Y(0) | X = x] (binary t) or the marginal effect dE[Y|T,X]/dT
    (continuous t, linear_dml only).

    X: few, pre-specified effect modifiers chosen before looking at outcomes (the CATE is
       reported as a function of these). W: further confounders, used only in the nuisance models.
    Nuisances E[Y|X,W], E[T|X,W] and, for binary t, E[Y|T=1,X,W], E[Y|T=0,X,W] are cross-fitted
    over n_folds folds (default learners: random forests; pass any scikit-learn estimator; a
    classifier with predict_proba for a binary treatment). The final stage is also cross-fitted,
    so res["tau"] holds OUT-OF-FOLD predictions: use them for every validation tool."""
    from sklearn.base import clone
    from sklearn.model_selection import KFold
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    W = list(W or [])
    X = list(X)
    cols = list(dict.fromkeys(X + W))
    d = df[[y, t] + cols].dropna()
    Y = d[y].to_numpy(float)
    T = d[t].to_numpy(float)
    Xm = d[X].to_numpy(float)
    Zm = d[cols].to_numpy(float)
    n = len(Y)
    binary = set(np.unique(T)) <= {0.0, 1.0}
    if not binary and method != "linear_dml":
        raise ValueError(f"method={method!r} needs a binary 0/1 treatment; use method='linear_dml' for a "
                         "continuous treatment")
    ol_d, ps_d = _default_learners(binary, seed)
    ol = outcome_learner if outcome_learner is not None else ol_d
    psl = ps_learner if ps_learner is not None else ps_d
    warn = []

    folds = list(KFold(n_folds, shuffle=True, random_state=seed).split(Zm))
    m_hat, e_hat = np.zeros(n), np.zeros(n)
    mu1, mu0 = np.zeros(n), np.zeros(n)
    for tr, te in folds:
        m_hat[te] = clone(ol).fit(Zm[tr], Y[tr]).predict(Zm[te])
        if binary:
            e_hat[te] = clone(psl).fit(Zm[tr], T[tr]).predict_proba(Zm[te])[:, 1]
            t1, t0 = tr[T[tr] == 1], tr[T[tr] == 0]
            mu1[te] = clone(ol).fit(Zm[t1], Y[t1]).predict(Zm[te])
            mu0[te] = clone(ol).fit(Zm[t0], Y[t0]).predict(Zm[te])
        else:
            e_hat[te] = clone(psl).fit(Zm[tr], T[tr]).predict(Zm[te])
    overlap = {}
    if binary:
        overlap = {"e_min": float(e_hat.min()), "e_max": float(e_hat.max()),
                   "share_clipped": float(np.mean((e_hat < clip) | (e_hat > 1 - clip)))}
        if overlap["share_clipped"] > 0.05:
            warn.append(f"{overlap['share_clipped']:.0%} of propensity scores outside [{clip}, {1 - clip}]: weak "
                        "overlap, effects for those units are extrapolated (trim the sample or redefine the population)")
        e_hat = np.clip(e_hat, clip, 1 - clip)
        psi = mu1 - mu0 + T * (Y - mu1) / e_hat - (1 - T) * (Y - mu0) / (1 - e_hat)
    else:
        psi = None
    y_res, t_res = Y - m_hat, T - e_hat

    Phi = np.column_stack([np.ones(n), Xm])
    tau_oof = np.zeros(n)
    final, coef, V = None, None, None
    kind = method

    def fit_final(idx):
        """Fit the final stage on rows idx; return a predictor of tau for a feature matrix."""
        if method == "dr":
            b, _ = _ols_hc1(Phi[idx], psi[idx])
            return lambda Xn: np.column_stack([np.ones(len(Xn)), Xn]) @ b
        if method == "linear_dml":
            b, _ = _ols_hc1(t_res[idx, None] * Phi[idx], y_res[idx])
            return lambda Xn: np.column_stack([np.ones(len(Xn)), Xn]) @ b
        if method == "forest":
            from sklearn.ensemble import RandomForestRegressor
            fl = final_learner if final_learner is not None else RandomForestRegressor(
                n_estimators=300, min_samples_leaf=20, max_features=0.5, random_state=seed)
            f = clone(fl).fit(Xm[idx], psi[idx])
            return f.predict
        if method == "t_learner":
            i1, i0 = idx[T[idx] == 1], idx[T[idx] == 0]
            f1, f0 = clone(ol).fit(Zm[i1], Y[i1]), clone(ol).fit(Zm[i0], Y[i0])
            return lambda Zn: f1.predict(Zn) - f0.predict(Zn)
        if method == "s_learner":
            f = clone(ol).fit(np.column_stack([Zm[idx], T[idx]]), Y[idx])
            return lambda Zn: (f.predict(np.column_stack([Zn, np.ones(len(Zn))]))
                               - f.predict(np.column_stack([Zn, np.zeros(len(Zn))])))
        if method == "x_learner":
            i1, i0 = idx[T[idx] == 1], idx[T[idx] == 0]
            f1, f0 = clone(ol).fit(Zm[i1], Y[i1]), clone(ol).fit(Zm[i0], Y[i0])
            g1 = clone(ol).fit(Xm[i1], Y[i1] - f0.predict(Zm[i1]))
            g0 = clone(ol).fit(Xm[i0], f1.predict(Zm[i0]) - Y[i0])
            pe = clone(psl).fit(Zm[idx], T[idx])
            return lambda XZ: (np.clip(pe.predict_proba(XZ[1])[:, 1], clip, 1 - clip) * g0.predict(XZ[0])
                               + (1 - np.clip(pe.predict_proba(XZ[1])[:, 1], clip, 1 - clip)) * g1.predict(XZ[0]))

    def feats(rows):
        if method in ("dr", "linear_dml", "forest"):
            return Xm[rows]
        if method == "x_learner":
            return (Xm[rows], Zm[rows])
        return Zm[rows]

    for tr, te in folds:
        tau_oof[te] = fit_final(tr)(feats(te))
    final = fit_final(np.arange(n))
    if method == "dr":
        coef, V = _ols_hc1(Phi, psi)
    elif method == "linear_dml":
        coef, V = _ols_hc1(t_res[:, None] * Phi, y_res)

    if binary:
        ate = _out("ATE (cross-fitted AIPW)", psi.mean(), psi.std(ddof=1) / np.sqrt(n), n)
    else:
        pbar = Phi.mean(0)
        ate = _out("average marginal effect (linear CATE at sample mean of X)", pbar @ coef,
                   np.sqrt(pbar @ V @ pbar), n)
    coef_table = None
    if coef is not None:
        z = stats.norm.ppf(1 - ALPHA / 2)
        se = np.sqrt(np.diag(V))
        coef_table = pd.DataFrame({"estimate": coef, "se": se, "z": coef / se,
                                   "p": 2 * stats.norm.sf(np.abs(coef / se)),
                                   "ci_low": coef - z * se, "ci_high": coef + z * se},
                                  index=["const"] + X)
    if method in ("forest", "t_learner", "s_learner", "x_learner"):
        warn.append(f"{method}: no valid pointwise CIs for tau(x); judge heterogeneity with blp_test, gates and toc")
    for w in warn:
        warnings.warn(w)
    return {"method": method, "binary_treatment": binary, "n": n, "X": X, "W": W, "ate": ate,
            "coef": coef_table, "tau": tau_oof, "psi": psi, "y_res": y_res, "t_res": t_res,
            "e_hat": e_hat, "overlap": overlap, "n_folds": n_folds, "seed": seed, "warnings": warn,
            "_final": final, "_cov": V, "_coef": coef, "_index": d.index, "_cols": cols}


def cate_predict(res: dict, df_new: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    """tau(x) for new units (full-sample final stage). CIs only for 'dr' and 'linear_dml'."""
    Xn = df_new[res["X"]].to_numpy(float)
    m = res["method"]
    if m in ("dr", "linear_dml", "forest"):
        tau = res["_final"](Xn)
    elif m == "x_learner":
        tau = res["_final"]((Xn, df_new[res["_cols"]].to_numpy(float)))
    else:
        tau = res["_final"](df_new[res["_cols"]].to_numpy(float))
    out = pd.DataFrame({"tau": tau}, index=df_new.index)
    if res["_cov"] is not None:
        P = np.column_stack([np.ones(len(Xn)), Xn])
        se = np.sqrt(np.einsum("ij,jk,ik->i", P, res["_cov"], P))
        z = stats.norm.ppf(1 - alpha / 2)
        out["se"], out["ci_low"], out["ci_high"] = se, tau - z * se, tau + z * se
    return out


# =============================================================================
# Validation
# =============================================================================
def blp_test(res: dict, tau: np.ndarray | None = None) -> dict:
    """Calibration / best-linear-predictor test of the CATE.

    Regress Y - m(X) on [T - e(X)] * mean(tau) and [T - e(X)] * (tau - mean(tau)) (grf's
    test_calibration; equivalent to the BLP of Chernozhukov et al.). Coefficients:
      mean_prediction ~ 1          the average prediction is right;
      differential_prediction > 0  (one-sided test) the predicted heterogeneity is real;
      differential_prediction ~ 1  the heterogeneity is well calibrated (< 1: overstated).
    Report heterogeneity only if the differential coefficient is significantly positive."""
    tau = res["tau"] if tau is None else np.asarray(tau, float)
    tbar = tau.mean()
    Z = np.column_stack([res["t_res"] * tbar, res["t_res"] * (tau - tbar)])
    b, V = _ols_hc1(Z, res["y_res"])
    se = np.sqrt(np.diag(V))
    zc = stats.norm.ppf(1 - ALPHA / 2)
    tab = pd.DataFrame({"estimate": b, "se": se, "z": b / se,
                        "p_one_sided": stats.norm.sf(b / se), "ci_low": b - zc * se, "ci_high": b + zc * se},
                       index=["mean_prediction", "differential_prediction"])
    p_het = float(tab.loc["differential_prediction", "p_one_sided"])
    bd = tab.loc["differential_prediction", "estimate"]
    if p_het >= 0.05:
        verdict = "no evidence that the predicted heterogeneity is real: report the ATE, not subgroups"
    elif tab.loc["differential_prediction", "ci_high"] < 1:
        verdict = f"heterogeneity detected but overstated (slope {bd:.2f} < 1): shrink tau(x) toward the ATE"
    else:
        verdict = f"heterogeneity detected and reasonably calibrated (slope {bd:.2f})"
    return {"method": "BLP / calibration test", "table": tab, "p_heterogeneity": p_het,
            "verdict": verdict, "n": res["n"]}


def _groups(tau, n_groups):
    return pd.qcut(pd.Series(tau).rank(method="first"), n_groups, labels=False).to_numpy()


def _group_effect(res, mask):
    if res["binary_treatment"]:
        v = res["psi"][mask]
        return v.mean(), v.std(ddof=1) / np.sqrt(mask.sum())
    tr, yr = res["t_res"][mask], res["y_res"][mask]
    b, V = _ols_hc1(tr[:, None], yr)
    return b[0], np.sqrt(V[0, 0])


def gates(res: dict, n_groups: int = 5, tau: np.ndarray | None = None) -> dict:
    """Sorted group average treatment effects: split units into n_groups by predicted tau
    (out of fold) and estimate the effect in each group with the doubly robust scores.
    Monotone, well-separated groups are the clearest evidence of heterogeneity; the top - bottom
    difference has its own test."""
    tau = res["tau"] if tau is None else np.asarray(tau, float)
    g = _groups(tau, n_groups)
    rows = []
    for k in range(n_groups):
        mk = g == k
        est, se = _group_effect(res, mk)
        rows.append({"group": k + 1, "n": int(mk.sum()), "mean_predicted_tau": float(tau[mk].mean()),
                     **{kk: v for kk, v in _out("GATE", est, se, mk.sum()).items()
                        if kk in ("estimate", "se", "ci_low", "ci_high", "p")}})
    tab = pd.DataFrame(rows).set_index("group")
    diff = tab["estimate"].iloc[-1] - tab["estimate"].iloc[0]
    se_d = np.sqrt(tab["se"].iloc[-1] ** 2 + tab["se"].iloc[0] ** 2)
    rho = stats.spearmanr(tab.index, tab["estimate"])[0]
    return {"method": "GATES", "table": tab, "top_minus_bottom": _out("GATE top - bottom", diff, se_d, res["n"]),
            "spearman_monotonicity": float(rho), "groups": g, "n": res["n"]}


def clan(res: dict, df: pd.DataFrame, covs: list[str], n_groups: int = 5, tau: np.ndarray | None = None) -> pd.DataFrame:
    """Classification analysis: mean characteristics of the most vs least affected group
    (top vs bottom of predicted tau), Welch t-tests with Holm-adjusted p-values. Descriptive:
    it says who benefits most, not why."""
    tau = res["tau"] if tau is None else np.asarray(tau, float)
    g = _groups(tau, n_groups)
    d = df.loc[res["_index"], covs]
    top, bot = d[g == n_groups - 1], d[g == 0]
    rows = []
    for c in covs:
        tt = stats.ttest_ind(top[c], bot[c], equal_var=False)
        rows.append({"variable": c, "mean_least_affected": bot[c].mean(), "mean_most_affected": top[c].mean(),
                     "difference": top[c].mean() - bot[c].mean(), "p": tt.pvalue})
    out = pd.DataFrame(rows).set_index("variable")
    order = np.argsort(out["p"].to_numpy())
    m = len(out)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * out["p"].iloc[i]))
        adj[i] = running
    out["p_holm"] = adj
    return out


def toc(res: dict, tau: np.ndarray | None = None, q_grid=None, B: int = 200, seed: int = 0) -> dict:
    """Targeting operator characteristic: TOC(q) = ATE among the top q share by predicted tau
    minus the overall ATE. AUTOC (area under TOC) and the Qini coefficient (q-weighted area)
    summarize how much better than random the ranking targets the treatment; > 0 with a small
    p-value means the ranking finds units with larger effects. Bootstrap SEs over units holding
    the predictions fixed. Binary treatment only."""
    if not res["binary_treatment"]:
        raise ValueError("toc needs a binary treatment (doubly robust scores)")
    tau = res["tau"] if tau is None else np.asarray(tau, float)
    q_grid = np.linspace(0.05, 1.0, 20) if q_grid is None else np.asarray(q_grid, float)
    psi = res["psi"]
    n = len(psi)

    def curve(order_psi):
        cs = np.cumsum(order_psi)
        tops = np.maximum(1, np.ceil(q_grid * len(order_psi)).astype(int))
        return cs[tops - 1] / tops - order_psi.mean()

    order = np.argsort(-tau, kind="stable")
    tc = curve(psi[order])
    autoc, qini = tc.mean(), (q_grid * tc).mean()
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        o = idx[np.argsort(-tau[idx], kind="stable")]
        c = curve(psi[o])
        bs.append((c.mean(), (q_grid * c).mean()))
    bs = np.array(bs)
    tab = pd.DataFrame({"q": q_grid, "toc": tc, "ate_top_q": tc + psi.mean()})
    return {"method": "TOC / RATE", "table": tab,
            "autoc": _out("AUTOC", autoc, bs[:, 0].std(ddof=1), n),
            "qini": _out("Qini coefficient", qini, bs[:, 1].std(ddof=1), n), "B": B}


def rscore(res: dict, tau: np.ndarray | None = None, ref: dict | None = None) -> float:
    """R-score = 1 - R-loss(tau) / R-loss(constant effect), with R-loss = mean((Y-m) - tau (T-e))^2
    (Nie & Wager 2021; EconML's RScorer). > 0: the model predicts effect variation better than a
    constant; <= 0: no better than the ATE. Uses out-of-fold tau and the residuals of `ref`
    (default: res itself) so that several models are scored on the same residuals."""
    ref = ref or res
    tau = res["tau"] if tau is None else np.asarray(tau, float)
    yr, tr = ref["y_res"], ref["t_res"]
    b0 = (tr @ yr) / (tr @ tr)
    l0 = np.mean((yr - b0 * tr) ** 2)
    l1 = np.mean((yr - tau * tr) ** 2)
    return float(1 - l1 / l0)


def compare_cate_models(results: dict) -> pd.DataFrame:
    """Side-by-side table for CATE models fit on the same data (same seed and folds): ATE, R-score
    (on the first model's residuals), BLP heterogeneity slope and p, AUTOC. Agreement between a
    simple (linear) and a flexible (forest) model is reassuring; large disagreement means the
    heterogeneity is fragile."""
    names = list(results)
    ref = results[names[0]]
    rows = []
    for k in names:
        r = results[k]
        if r["n"] != ref["n"]:
            raise ValueError("models were fit on different samples")
        b = blp_test(r, tau=r["tau"])
        row = {"model": k, "method": r["method"], "ATE": r["ate"]["estimate"], "ATE_se": r["ate"]["se"],
               "R_score": rscore(r, ref=ref),
               "BLP_slope": b["table"].loc["differential_prediction", "estimate"], "BLP_p": b["p_heterogeneity"]}
        if r["binary_treatment"]:
            a = toc(r, B=100)["autoc"]
            row.update({"AUTOC": a["estimate"], "AUTOC_p": a["p"]})
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


# =============================================================================
# Policy learning
# =============================================================================
def _best_split(F, g, thresholds, min_leaf):
    """Best single split of rows with gains g (treat a leaf if its gain sum > 0)."""
    best = (max(g.sum(), 0.0), None, None, g.sum() > 0, g.sum() > 0)
    for j in range(F.shape[1]):
        for c in thresholds[j]:
            left = F[:, j] <= c
            nl = left.sum()
            if nl < min_leaf or len(g) - nl < min_leaf:
                continue
            sl, sr = g[left].sum(), g[~left].sum()
            v = max(sl, 0.0) + max(sr, 0.0)
            if v > best[0] + 1e-12:
                best = (v, j, c, sl > 0, sr > 0)
    return best


def policy_tree(res: dict, df: pd.DataFrame, features: list[str], cost: float = 0.0, depth: int = 2,
                n_thresholds: int = 20, train_frac: float = 0.5, min_leaf: int | None = None, seed: int = 0) -> dict:
    """Interpretable treatment rule: a tree of depth 1 or 2 chosen by exhaustive search to maximize
    the sum of doubly robust gains psi_i - cost over treated leaves (Athey & Wager 2021; EconML's
    DRPolicyTree, grf's policy_tree). Learned on a random train_frac of the units and valued on the
    rest, so the reported value is honest. cost = cost of treating one unit, in units of y.
    Binary treatment only."""
    if not res["binary_treatment"]:
        raise ValueError("policy_tree needs a binary treatment")
    if depth not in (1, 2):
        raise ValueError("depth must be 1 or 2")
    F = df.loc[res["_index"], features].to_numpy(float)
    gain = res["psi"] - cost
    n = len(gain)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    ntr = int(train_frac * n)
    tr, te = np.sort(perm[:ntr]), np.sort(perm[ntr:])
    min_leaf = min_leaf or max(20, int(0.05 * ntr))
    qs = np.linspace(0, 1, n_thresholds + 2)[1:-1]
    thr = [np.unique(np.quantile(F[tr, j], qs)) for j in range(F.shape[1])]

    def leaf(treat):
        return {"treat": bool(treat)}

    Ft, gt = F[tr], gain[tr]
    if depth == 1:
        v, j, c, tl, trr = _best_split(Ft, gt, thr, min_leaf)
        tree = leaf(gt.sum() > 0) if j is None else {"feature": features[j], "threshold": float(c),
                                                     "left": leaf(tl), "right": leaf(trr)}
    else:
        best_v, tree = max(gt.sum(), 0.0), leaf(gt.sum() > 0)
        for j in range(F.shape[1]):
            for c in thr[j]:
                left = Ft[:, j] <= c
                if left.sum() < min_leaf or (~left).sum() < min_leaf:
                    continue
                bl = _best_split(Ft[left], gt[left], thr, min_leaf)
                br = _best_split(Ft[~left], gt[~left], thr, min_leaf)
                if bl[0] + br[0] > best_v + 1e-12:
                    best_v = bl[0] + br[0]

                    def sub(b):
                        if b[1] is None:
                            return leaf(b[3])
                        return {"feature": features[b[1]], "threshold": float(b[2]),
                                "left": leaf(b[3]), "right": leaf(b[4])}
                    tree = {"feature": features[j], "threshold": float(c), "left": sub(bl), "right": sub(br)}

    def rules(node, path):
        if "treat" in node:
            return [f"{' and '.join(path) if path else 'everyone'} -> {'TREAT' if node['treat'] else 'do not treat'}"]
        f, c = node["feature"], node["threshold"]
        return rules(node["left"], path + [f"{f} <= {c:.4g}"]) + rules(node["right"], path + [f"{f} > {c:.4g}"])

    assign = policy_predict(tree, df.loc[res["_index"]])
    val = policy_value(res, assign, cost=cost, idx=te)
    return {"method": f"policy tree (depth {depth}, exhaustive search)", "tree": tree,
            "rules": rules(tree, []), "value_test": val, "share_treated": float(assign.mean()),
            "train_idx": tr, "test_idx": te, "cost": cost}


def policy_predict(tree: dict, df: pd.DataFrame) -> np.ndarray:
    """Apply a policy_tree rule: boolean array, True = treat."""
    def go(node, rows):
        if "treat" in node:
            return np.full(len(rows), node["treat"])
        x = rows[node["feature"]].to_numpy(float)
        out = np.empty(len(rows), bool)
        lm = x <= node["threshold"]
        out[lm] = go(node["left"], rows[lm])
        out[~lm] = go(node["right"], rows[~lm])
        return out
    return go(tree, df)


def policy_value(res: dict, treat, cost: float = 0.0, idx=None) -> pd.DataFrame:
    """Doubly robust value of a treatment rule, per unit and relative to treating no one:
    mean(pi_i * (psi_i - cost)). Also the gain over treating everyone. Evaluate on units not used
    to learn the rule (policy_tree does this for you)."""
    if not res["binary_treatment"]:
        raise ValueError("policy_value needs a binary treatment")
    pi = np.asarray(treat, float)
    g = res["psi"] - cost
    idx = np.arange(len(g)) if idx is None else np.asarray(idx)
    pi, g = pi[idx], g[idx]
    rows = []
    for name, v in [("policy vs treat no one", pi * g), ("treat everyone vs no one", g),
                    ("policy vs treat everyone", (pi - 1) * g)]:
        o = _out(name, v.mean(), v.std(ddof=1) / np.sqrt(len(v)), len(v))
        rows.append({k: o[k] for k in ("estimand", "estimate", "se", "ci_low", "ci_high", "p", "n")})
    out = pd.DataFrame(rows).set_index("estimand")
    out.attrs["share_treated"] = float(pi.mean())
    return out
