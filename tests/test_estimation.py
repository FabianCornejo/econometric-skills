"""Tests for estimation.py (v2.0): generic MLE, trinity tests, NLS, delta method, bootstraps.

Benchmarks: Wooldridge, *Introductory Econometrics*, Example 17.1 (probit, MROZ) and Example 17.3
(Poisson, CRIME1); statsmodels as an independent implementation; exact algebraic identities;
Monte Carlo size of the wild cluster bootstrap with few clusters.

Run from the repository root:  python3 tests/test_estimation.py
Needs: statsmodels, scipy, wooldridge (pip install wooldridge).
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "econometrics" / "scripts"))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import optimize, stats
from scipy.special import gammaln

import estimation as est

try:
    import wooldridge as woo
except ImportError:  # pragma: no cover
    sys.exit("SKIP: install the `wooldridge` package (pip install wooldridge) to run these tests")

ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


# ---------------------------------------------------------------------------------------------
# Poisson by hand — Wooldridge Example 17.3 (CRIME1, narr86)
# ---------------------------------------------------------------------------------------------
cr = woo.data("crime1")
xs = ["pcnv", "avgsen", "tottime", "ptime86", "qemp86", "inc86", "black", "hispan", "born60"]
X = sm.add_constant(cr[xs]).to_numpy(float)
y = cr["narr86"].to_numpy(float)
names = ["const"] + xs


def ll_pois(b, X, y):
    xb = X @ b
    return y * xb - np.exp(xb) - gammaln(y + 1)


def sc_pois(b, X, y):
    return (y - np.exp(X @ b))[:, None] * X


m_n = est.mle(ll_pois, np.zeros(X.shape[1]), args=(X, y), names=names, method="newton")
m_b = est.mle(ll_pois, np.zeros(X.shape[1]), args=(X, y), names=names, method="bfgs")
sm_p = sm.Poisson(y, X).fit(disp=0)
book = {"const": -0.600, "pcnv": -0.402, "avgsen": -0.024, "tottime": 0.024, "ptime86": -0.099,
        "qemp86": -0.038, "inc86": -0.0081, "black": 0.661, "hispan": 0.500, "born60": -0.051}
check("W17.3 Poisson coefficients match the book",
      all(abs(m_n["params"][k] - v) < 0.0011 for k, v in book.items()),
      f"black {m_n['params']['black']:.3f}, pcnv {m_n['params']['pcnv']:.3f}")
check("Poisson: Newton = BFGS = statsmodels (coefficients)",
      np.allclose(m_n["params"], sm_p.params, atol=1e-5) and np.allclose(m_b["params"], sm_p.params, atol=1e-4))
check("Poisson: Hessian SEs = statsmodels", np.allclose(m_n["table"]["se"], sm_p.bse, rtol=1e-4))
check("Poisson: log-likelihood = statsmodels", abs(m_n["loglik"] - sm_p.llf) < 1e-6)
m_s = est.mle(ll_pois, np.zeros(X.shape[1]), args=(X, y), names=names, cov_type="sandwich")
sm_r = sm.Poisson(y, X).fit(disp=0, cov_type="HC0")
check("Poisson QMLE sandwich SEs = statsmodels HC0", np.allclose(m_s["table"]["se"], sm_r.bse, rtol=1e-3))
m_a = est.mle(ll_pois, np.zeros(X.shape[1]), args=(X, y), names=names, score=sc_pois)
check("analytic score gives the same estimates and SEs",
      np.allclose(m_a["params"], m_n["params"], atol=1e-6) and np.allclose(m_a["table"]["se"], m_n["table"]["se"], rtol=1e-4))
m_c = est.mle(ll_pois, np.zeros(X.shape[1]), args=(X, y), names=names, cov_type="cluster",
              cluster=np.arange(len(y)))
n = len(y)
check("one observation per cluster: clustered = sandwich x n/(n-1)",
      np.allclose(np.diag(m_c["cov"]), np.diag(m_s["cov"]) * n / (n - 1), rtol=1e-8))
check("overdispersed counts: sandwich SE of pcnv larger than Hessian SE",
      m_s["table"].loc["pcnv", "se"] > m_n["table"].loc["pcnv", "se"])

# ---------------------------------------------------------------------------------------------
# Probit by hand — Wooldridge Example 17.1 (MROZ, inlf)
# ---------------------------------------------------------------------------------------------
mz = woo.data("mroz")
pv = ["nwifeinc", "educ", "exper", "expersq", "age", "kidslt6", "kidsge6"]
Xp = sm.add_constant(mz[pv]).to_numpy(float)
yp = mz["inlf"].to_numpy(float)


def ll_probit(b, X, y):
    q = 2 * y - 1
    return stats.norm.logcdf(q * (X @ b))


pr = est.mle(ll_probit, np.zeros(Xp.shape[1]), args=(Xp, yp), names=["const"] + pv)
sm_pr = sm.Probit(yp, Xp).fit(disp=0)
bookp = {"const": 0.270, "nwifeinc": -0.012, "educ": 0.131, "exper": 0.123, "expersq": -0.0019,
         "age": -0.053, "kidslt6": -0.868, "kidsge6": 0.036}
check("W17.1 probit coefficients match the book",
      all(abs(pr["params"][k] - v) < 0.0011 for k, v in bookp.items()), f"kidslt6 {pr['params']['kidslt6']:.3f}")
check("probit SEs = statsmodels", np.allclose(pr["table"]["se"], sm_pr.bse, rtol=1e-3))

# ---------------------------------------------------------------------------------------------
# Trinity: H0 black = hispan = 0 in the Poisson model
# ---------------------------------------------------------------------------------------------
R = np.zeros((2, len(names)))
R[0, names.index("black")] = 1
R[1, names.index("hispan")] = 1
tri = est.trinity_test(ll_pois, np.zeros(len(names)), R, args=(X, y), unrestricted=m_n)
keep = [c for c in names if c not in ("black", "hispan")]
sm_rest = sm.Poisson(y, sm.add_constant(cr[[c for c in xs if c not in ("black", "hispan")]]).to_numpy(float)).fit(disp=0)
lr_sm = 2 * (sm_p.llf - sm_rest.llf)
check("LR statistic = 2(llf_u - llf_r) from statsmodels", abs(tri.loc["LR", "statistic"] - lr_sm) < 1e-4,
      f"LR {tri.loc['LR', 'statistic']:.3f}")
wald_sm = float(np.asarray(sm_p.wald_test(R, scalar=True).statistic))
check("Wald statistic = statsmodels wald_test", abs(tri.loc["Wald", "statistic"] - wald_sm) < 1e-3 * max(1, wald_sm))
check("restricted estimates satisfy R theta = 0", np.allclose(R @ tri.attrs["restricted_params"], 0, atol=1e-10))
check("LR, Wald and LM agree in a large sample (within 10%)",
      tri["statistic"].max() / tri["statistic"].min() < 1.10, tri["statistic"].round(2).to_dict())
check("all three reject H0 at 1%", (tri["p"] < 0.01).all())

# ---------------------------------------------------------------------------------------------
# NLS
# ---------------------------------------------------------------------------------------------
w1 = woo.data("wage1")
Xw = sm.add_constant(w1[["educ", "exper", "tenure"]]).to_numpy(float)
nl = est.nls(lambda b, X: X @ b, np.zeros(4), np.log(w1["wage"]).to_numpy(), args=(Xw,))
ols = sm.OLS(np.log(w1["wage"]).to_numpy(), Xw).fit(cov_type="HC1")
check("NLS with a linear f = OLS (coefficients)", np.allclose(nl["params"], ols.params, atol=1e-7))
check("NLS HC1 SEs = statsmodels OLS HC1", np.allclose(nl["table"]["se"], ols.bse, rtol=1e-5))

rng = np.random.default_rng(11)
xx = rng.uniform(0, 3, 2000)
yy = 2.0 * np.exp(0.7 * xx) + rng.normal(0, 1 + xx, 2000)          # heteroskedastic
fe = est.nls(lambda t, x: t[0] * np.exp(t[1] * x), [1.0, 0.1], yy, args=(xx,), names=["a", "b"])
check("NLS exponential recovers a=2, b=0.7 (inside 95% CI)",
      fe["table"].loc["a", "ci_low"] < 2 < fe["table"].loc["a", "ci_high"]
      and fe["table"].loc["b", "ci_low"] < 0.7 < fe["table"].loc["b", "ci_high"],
      fe["params"].round(3).to_dict())

# ---------------------------------------------------------------------------------------------
# Delta method
# ---------------------------------------------------------------------------------------------
th = ols.params
Vc = ols.cov_params()
a = np.array([0, 1, 2, 0.0])
dl = est.delta_method(lambda t: a @ t, th, Vc)
check("delta method of a linear function is exact", abs(dl["se"].iloc[0] - np.sqrt(a @ Vc @ a)) < 1e-9)
# ratio exper/educ: compare with simulation from the asymptotic normal
dr = est.delta_method(lambda t: t[2] / t[1], th, Vc)
sims = rng.multivariate_normal(th, Vc, 200_000)
check("delta-method SE of a ratio ~ simulated SE (within 5%)",
      abs(dr["se"].iloc[0] / (sims[:, 2] / sims[:, 1]).std() - 1) < 0.05)
vec = est.delta_method(lambda t: np.array([np.exp(t[1]) - 1, t[2] + t[3]]), th, Vc, names=["pct_educ", "exp_ten"])
check("vector delta method returns a joint covariance", vec.attrs["cov"].shape == (2, 2) and list(vec.index) == ["pct_educ", "exp_ten"])

# ---------------------------------------------------------------------------------------------
# Pairs and cluster bootstrap
# ---------------------------------------------------------------------------------------------
n = 600
x = rng.normal(size=n)
dfh = pd.DataFrame({"x": x, "y": 1 + 0.5 * x + rng.normal(size=n) * (0.5 + np.abs(x))})
bs = est.bootstrap(lambda d: smf.ols("y ~ x", d).fit().params["x"], dfh, B=499, seed=1)
hc = smf.ols("y ~ x", dfh).fit(cov_type="HC3").bse["x"]
check("pairs bootstrap SE ~ HC3 SE under heteroskedasticity (within 15%)",
      abs(bs["table"]["se"].iloc[0] / hc - 1) < 0.15, f"boot {bs['table']['se'].iloc[0]:.4f} vs HC3 {hc:.4f}")
G, m = 120, 10
g = np.repeat(np.arange(G), m)
xc = np.repeat(rng.normal(size=G), m) + rng.normal(size=G * m)
dfc = pd.DataFrame({"g": g, "x": xc, "y": 0.3 * xc + np.repeat(rng.normal(size=G), m) + rng.normal(size=G * m)})
bc = est.bootstrap(lambda d: smf.ols("y ~ x", d).fit().params["x"], dfc, B=499, cluster="g", seed=2)
cr1 = smf.ols("y ~ x", dfc).fit(cov_type="cluster", cov_kwds={"groups": dfc["g"]}).bse["x"]
iid = smf.ols("y ~ x", dfc).fit().bse["x"]
check("cluster bootstrap SE ~ CR1 SE (within 15%) and above the i.i.d. SE",
      abs(bc["table"]["se"].iloc[0] / cr1 - 1) < 0.15 and bc["table"]["se"].iloc[0] > 1.2 * iid,
      f"boot {bc['table']['se'].iloc[0]:.4f}, CR1 {cr1:.4f}, iid {iid:.4f}")

# ---------------------------------------------------------------------------------------------
# Wild cluster bootstrap: size with 8 clusters, power, CI coherent with the test
# ---------------------------------------------------------------------------------------------
def few_cluster_sample(r, beta, G=8, m=30):
    gg = np.repeat(np.arange(G), m)
    xg = np.repeat(r.normal(size=G), m) + 0.5 * r.normal(size=G * m)
    u = np.repeat(r.normal(size=G), m) * 1.0 + r.normal(size=G * m)
    return pd.DataFrame({"g": gg, "x": xg, "y": beta * xg + u})


rej_cr1, rej_wcr = [], []
for rep in range(300):
    r = np.random.default_rng(1000 + rep)
    d = few_cluster_sample(r, 0.0)
    fit = smf.ols("y ~ x", d).fit(cov_type="cluster", cov_kwds={"groups": d["g"]})
    rej_cr1.append(fit.pvalues["x"] < 0.05)
    w = est.wild_cluster_bootstrap(d, "y ~ x", "x", "g", B=399, seed=rep, ci=False)
    rej_wcr.append(w["p"] < 0.05)
size_cr1, size_wcr = np.mean(rej_cr1), np.mean(rej_wcr)
check("8 clusters: CR1 with normal critical values over-rejects (size > 10%)", size_cr1 > 0.10, f"{size_cr1:.3f}")
check("8 clusters: wild cluster bootstrap size within [2%, 9%]", 0.02 <= size_wcr <= 0.09, f"{size_wcr:.3f}")

pw = []
for rep in range(60):
    d = few_cluster_sample(np.random.default_rng(5000 + rep), 1.5)
    pw.append(est.wild_cluster_bootstrap(d, "y ~ x", "x", "g", B=399, seed=rep, ci=False)["p"] < 0.05)
check("wild cluster bootstrap has power against a large effect (> 70%)", np.mean(pw) > 0.70, f"{np.mean(pw):.2f}")

coherent = []
for s, beta in [(77, 1.5), (78, 0.0)]:   # one clear rejection, one clear non-rejection (p far from 5%)
    w = est.wild_cluster_bootstrap(few_cluster_sample(np.random.default_rng(s), beta), "y ~ x", "x", "g",
                                   B=1999, seed=3)
    coherent.append(((w["ci_low"] <= 0 <= w["ci_high"]) == (w["p"] > 0.05), round(w["p"], 3)))
check("test-inversion CI is coherent with the p-value for H0: beta = 0 (reject and non-reject cases)",
      all(c for c, _ in coherent), f"p-values {[p for _, p in coherent]}")
check("Webb weights chosen automatically with 8 clusters", "webb" in w["method"] and w["n_clusters"] == 8)


# --- v2.7.4: a missing cluster label in the wild cluster bootstrap ---------------------------------------------
rng74 = np.random.default_rng(7402)
G74 = 20
w74 = pd.DataFrame({"g": np.repeat(np.arange(G74), 15).astype(float)})
ug = rng74.normal(size=G74)
w74["x"] = rng74.normal(size=len(w74)) + ug[w74["g"].astype(int)]
w74["y"] = 0.3 * w74["x"] + rng74.normal(size=len(w74)) + rng74.normal(size=G74)[w74["g"].astype(int)]
w74n = w74.copy(); w74n.loc[w74n["g"] == 0, "g"] = np.nan
try:
    est.wild_cluster_bootstrap(w74n, "y ~ x", "x", "g", B=199, ci=False); msg = "no error"; okr = False
except ValueError as e_:
    msg = str(e_); okr = "15 missing label" in msg and "'g'" in msg
check("v2.7.4 wild_cluster_bootstrap: a missing cluster label raises (before: merged into the last cluster)", okr, msg)
# boundary: a missing label only on rows the formula drops anyway (y missing there) is not an error, and the result is
# the one on the complete rows
w74d = w74n.copy(); w74d.loc[w74d["g"].isna(), "y"] = np.nan
r_d = est.wild_cluster_bootstrap(w74d, "y ~ x", "x", "g", B=199, ci=False)
r_c = est.wild_cluster_bootstrap(w74d.dropna(), "y ~ x", "x", "g", B=199, ci=False)
check("v2.7.4 wild_cluster_bootstrap: a missing label on rows the formula drops is ignored (same as complete rows)",
      r_d["n_clusters"] == 19 and r_d["n"] == 285 and abs(r_d["se"] - r_c["se"]) < 1e-14 and r_d["p"] == r_c["p"])


# review r1 S3: a categorical cluster with unused categories (a filter before the call): clusters are the labels used
w74c = w74.copy()
w74c["g"] = pd.Categorical(w74c["g"].astype(int) * 2, categories=list(range(40)))
r_cat = est.wild_cluster_bootstrap(w74c, "y ~ x", "x", "g", B=199, ci=False)
r_pl = est.wild_cluster_bootstrap(w74, "y ~ x", "x", "g", B=199, ci=False)
check("v2.7.4 review r1 S3: unused categories do not count as clusters (n_clusters 20, the same se, p and CR1 p as "
      "with plain labels)", r_cat["n_clusters"] == 20 and abs(r_cat["se"] - r_pl["se"]) < 1e-14
      and r_cat["p_cr1_t_G-1"] == r_pl["p_cr1_t_G-1"], (r_cat["n_clusters"], r_cat["se"], r_pl["se"]))


# =============================================================================================
# v2.8.0: generic GMM (estimation.gmm) and the LR test between nested mle fits (estimation.lr_test)
# References are written here by other routes: the closed form of linear GMM, the IV formula, the HAC sum
# over all pairs, statsmodels' NonlinearIVGMM (independent code), the normal-regression LR n log(SSR_r/SSR_u),
# and a design with known truth (Euler equation with gamma = 2, beta = 0.995; QuantEcon "Estimating Euler
# Equations by GMM", lecture-python.myst 0dc4fe7, described there; the simulator below is ours).
# =============================================================================================
import contextlib
import io


def _lin_moments(th, y, X, Z):
    return (y - X @ th)[:, None] * Z


rng80 = np.random.default_rng(80)
n80 = 600
z80 = rng80.standard_normal((n80, 3))
v80 = rng80.standard_normal(n80)
x80 = 0.8 * z80.sum(1) + v80
u80 = (0.6 * v80 + rng80.standard_normal(n80)) * np.sqrt(0.5 + z80[:, 0] ** 2)   # endogenous and heteroskedastic
y80 = 1.0 + 2.0 * x80 + u80
X80 = np.column_stack([np.ones(n80), x80])
Z80 = np.column_stack([np.ones(n80), z80])

# closed form of two-step linear GMM (centered S, identity first step), written without the function
A80 = Z80.T @ X80 / n80
c80 = Z80.T @ y80 / n80


def _cf_step(W):
    return np.linalg.solve(A80.T @ W @ A80, A80.T @ W @ c80)


def _cf_S(th):
    g = (y80 - X80 @ th)[:, None] * Z80
    gc = g - g.mean(0)
    return gc.T @ gc / n80


th1 = _cf_step(np.eye(4))
W2 = np.linalg.inv(_cf_S(th1))
th2 = _cf_step(W2)
gbar2 = ((y80 - X80 @ th2)[:, None] * Z80).mean(0)
J_cf = n80 * gbar2 @ W2 @ gbar2
B80 = np.linalg.inv(A80.T @ W2 @ A80)
V_cf = B80 @ A80.T @ W2 @ _cf_S(th2) @ W2 @ A80 @ B80 / n80
g80 = est.gmm(_lin_moments, [0.0, 0.0], args=(y80, X80, Z80), names=["const", "x"], show=False)
check("v2.8.0 gmm: two-step linear GMM equals the closed form (A'WA)^-1 A'Wc with W = S(theta1)^-1 (rel <= 1e-9)",
      np.allclose(g80["params"].to_numpy(), th2, rtol=1e-9, atol=0), (g80["params"].to_numpy(), th2))
check("v2.8.0 gmm: sandwich covariance with S at the estimate equals the closed form (rel <= 1e-7)",
      np.allclose(g80["cov"].to_numpy(), V_cf, rtol=1e-7, atol=0))
check("v2.8.0 gmm: J = n gbar' W gbar with the step-2 weight, df = m - k = 2 (rel <= 1e-7)",
      abs(g80["j"] / J_cf - 1) < 1e-7 and g80["j_df"] == 2, (g80["j"], J_cf))
check("v2.8.0 gmm: j_p is the RIGHT tail chi2.sf(J, 2) and j_cdf = 1 - j_p",
      abs(g80["j_p"] - stats.chi2.sf(J_cf, 2)) < 1e-9 and abs(g80["j_p"] + g80["j_cdf"] - 1) < 1e-12)

# iterated: at the fixed point the estimate solves the problem with W = S(theta)^-1 at itself
g80i = est.gmm(_lin_moments, [0.0, 0.0], args=(y80, X80, Z80), steps="iterated", tol=1e-12, show=False)
th_i = g80i["params"].to_numpy()
check("v2.8.0 gmm: iterated GMM is a fixed point, theta = argmin with W = S(theta)^-1 (rel <= 1e-8)",
      np.allclose(_cf_step(np.linalg.inv(_cf_S(th_i))), th_i, rtol=1e-8, atol=0) and g80i["converged"])

# one-step: the user's weight, sandwich SEs, no J
W_zz = np.linalg.inv(Z80.T @ Z80 / n80)
g80o = est.gmm(_lin_moments, [0.0, 0.0], args=(y80, X80, Z80), steps="one-step", weight=W_zz, show=False)
th_o = _cf_step(W_zz)
Bo = np.linalg.inv(A80.T @ W_zz @ A80)
V_o = Bo @ A80.T @ W_zz @ _cf_S(th_o) @ W_zz @ A80 @ Bo / n80
check("v2.8.0 gmm: one-step with W = (Z'Z/n)^-1 is 2SLS, with the sandwich covariance (rel <= 1e-7)",
      np.allclose(g80o["params"].to_numpy(), th_o, rtol=1e-9, atol=0)
      and np.allclose(g80o["cov"].to_numpy(), V_o, rtol=1e-7, atol=0))
check("v2.8.0 gmm: one-step reports no J and says why",
      g80o["j"] is None and g80o["j_p"] is None
      and any(w["level"] == "notice" and "no J" in w["message"] for w in g80o["warnings"]))

# exactly identified: the IV formula, J = 0 and a notice
Zx = Z80[:, :2]
g80e = est.gmm(_lin_moments, [0.0, 0.0], args=(y80, X80, Zx), show=False)
iv = np.linalg.solve(Zx.T @ X80, Zx.T @ y80)
check("v2.8.0 gmm: exactly identified GMM equals the IV estimator (Z'X)^-1 Z'y and J = 0 (no p-value)",
      np.allclose(g80e["params"].to_numpy(), iv, rtol=1e-9, atol=0) and g80e["j_df"] == 0 and g80e["j"] < 1e-12
      and g80e["j_p"] is None)
check("v2.8.0 gmm: exactly identified gets the notice that J tests nothing",
      any(w["level"] == "notice" and "exactly identified" in w["message"] for w in g80e["warnings"])
      and not any("exactly identified" in w["message"] for w in g80["warnings"]))

# HAC S against the double sum over all pairs (|t - s| <= L, weight 1 - |t - s|/(L + 1) or 1)
rng_h = np.random.default_rng(81)
Gh = rng_h.standard_normal((60, 2))
Gh[1:] += 0.6 * Gh[:-1]
for kern in ("bartlett", "truncated"):
    U = Gh - Gh.mean(0)
    Sd = np.zeros((2, 2))
    for t in range(60):
        for s_ in range(60):
            lag = abs(t - s_)
            if lag <= 3:
                Sd += (1 - lag / 4 if kern == "bartlett" else 1.0) * np.outer(U[t], U[s_])
    Sd /= 60
    check(f"v2.8.0 gmm: the {kern} HAC long-run covariance equals the double sum over pairs (rel <= 1e-12)",
          np.allclose(est._gmm_S(Gh, 3, kern, True), Sd, rtol=1e-12, atol=0))


# Euler equation with known truth
def _euler_sim(n, gamma=2.0, beta=0.995, seed=0):
    r = np.random.default_rng(seed)
    burn = 200
    e = r.standard_normal(n + burn)
    dc = np.empty(n + burn)
    dc[0] = 0.0015
    for t in range(1, n + burn):
        dc[t] = 0.0015 * 0.6 + 0.4 * dc[t - 1] + 0.006 * e[t]
    cg = np.exp(dc[burn:])
    sdf = beta * cg ** (-gamma)
    xi = np.exp(0.02 * r.standard_normal(n) - 0.5 * 0.02 ** 2)     # mean one: E[sdf R] = 1
    return xi / sdf, cg


def _euler_arrays(R, cg, p):
    n = len(R)
    Z = [np.ones(n - p)]
    for j in range(1, p + 1):
        Z += [R[p - j:n - j], cg[p - j:n - j]]
    return R[p:], cg[p:], np.column_stack(Z)


def _euler_moments(th, R, cg, Z):
    return (th[1] * cg ** (-th[0]) * R - 1.0)[:, None] * Z


Re, ce = _euler_sim(5000, seed=3)
Rr, cr, Ze = _euler_arrays(Re, ce, 2)
ge = est.gmm(_euler_moments, [1.0, 0.99], args=(Rr, cr, Ze), names=["gamma", "beta"], center=False,
             weight=np.linalg.inv(Ze.T @ Ze / len(Ze)), show=False)
from statsmodels.sandbox.regression.gmm import NonlinearIVGMM


def _sm_err(params, exog):
    return params[1] * exog[:, 1] ** (-params[0]) * exog[:, 0] - 1.0


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    sm_res = NonlinearIVGMM(np.zeros(len(Rr)), np.column_stack([Rr, cr]), Ze, _sm_err).fit(
        start_params=np.array([1.0, 0.99]), maxiter=2, optim_method="bfgs", optim_args={"disp": False, "gtol": 1e-12},
        wargs={"centered": False})
check("v2.8.0 gmm: Euler equation, same estimates as statsmodels NonlinearIVGMM two-step (rel <= 1e-8; "
      "different optimizers)", np.allclose(ge["params"].to_numpy(), sm_res.params, rtol=1e-8, atol=0),
      (ge["params"].to_numpy(), sm_res.params))
check("v2.8.0 gmm: Euler equation, same SEs and J as NonlinearIVGMM (rel <= 1e-6; numerical Jacobians differ)",
      np.allclose(ge["table"]["se"].to_numpy(), sm_res.bse, rtol=1e-6, atol=0)
      and abs(ge["j"] / sm_res.jtest()[0] - 1) < 1e-6, (ge["table"]["se"].to_numpy(), sm_res.bse))
check("v2.8.0 gmm: Euler equation with n = 4998: the rank test rejects rank < 2 (p < 1e-10) and there is no "
      "identification warning", ge["rank_p"] < 1e-10
      and not any("may not be identified" in w["message"] for w in ge["warnings"]), ge["rank_p"])

# Monte Carlo: 400 samples of 900 months, two instrument lags (5 moments), default options.
# Bands fixed before running: coverage of the 95% CI in [0.92, 0.98] (MC SE 1.1%); size of J at 5% in [0.02, 0.085]
# (catches a 2x inflation).
cov_g, cov_b, rej = [], [], []
for rep in range(400):
    R_, c_ = _euler_sim(900, seed=1000 + rep)
    a_ = _euler_arrays(R_, c_, 2)
    f_ = est.gmm(_euler_moments, [1.0, 0.99], args=a_, show=False)
    t_ = f_["table"]
    cov_g.append(t_["ci_low"].iloc[0] <= 2.0 <= t_["ci_high"].iloc[0])
    cov_b.append(t_["ci_low"].iloc[1] <= 0.995 <= t_["ci_high"].iloc[1])
    rej.append(f_["j_p"] < 0.05)
mc = (np.mean(cov_g), np.mean(cov_b), np.mean(rej))
check("v2.8.0 gmm: Euler Monte Carlo (400 x 900): 95% CI coverage of gamma and beta in [0.92, 0.98]",
      0.92 <= mc[0] <= 0.98 and 0.92 <= mc[1] <= 0.98, mc)
check("v2.8.0 gmm: Euler Monte Carlo: J rejects at 5% in [0.02, 0.085] of the samples (size)", 0.02 <= mc[2] <= 0.085, mc)

# bound: beta capped at 0.99 when the truth is 0.995 -> the estimate sits on the bound -> warning
gb = est.gmm(_euler_moments, [1.0, 0.98], args=(Rr, cr, Ze), names=["gamma", "beta"], bounds=[None, (0.0, 0.99)],
             show=False)
check("v2.8.0 gmm: an estimate on its bound gets the boundary warning (Andrews 1999)",
      abs(gb["params"]["beta"] - 0.99) < 1e-9 and any(w["level"] == "warning" and "upper bound" in w["message"]
                                                     for w in gb["warnings"]))
gb2 = est.gmm(_euler_moments, [1.0, 0.98], args=(Rr, cr, Ze), names=["gamma", "beta"], bounds=[None, (0.0, 1.5)],
              show=False)
check("v2.8.0 gmm: a bound far from the estimate (many SEs) gives no boundary warning",
      not any("bound" in w["message"] for w in gb2["warnings"]))


# weak and missing identification in a linear IV with one endogenous regressor (constant + 3 instruments)
def _iv_case(pi, seed, n=500):
    r = np.random.default_rng(seed)
    z = r.standard_normal((n, 3))
    v = r.standard_normal(n)
    x = pi * z.sum(1) + v + 1.0
    y = x + 0.5 * v + r.standard_normal(n)
    return y, np.column_stack([np.ones(n), x]), np.column_stack([np.ones(n), z])


def _fs_F(y, X, Z):
    return float(np.squeeze(sm.OLS(X[:, 1], Z).fit().f_test(np.eye(4)[1:]).fvalue))


strong = est.gmm(_lin_moments, [0.0, 0.0], args=_iv_case(1.0, 1), show=False)
check("v2.8.0 gmm: strong instruments (first-stage F in the hundreds): rank test rejects, no identification warning",
      not any("identif" in w["message"] for w in strong["warnings"]) and strong["rank_p"] < 1e-10)
check("v2.8.0 gmm: rank test df = m - k + 1 = 3 with 4 moments and 2 parameters, p = chi2.sf(stat, 3)",
      strong["rank_df"] == 3 and abs(strong["rank_p"] - stats.chi2.sf(strong["rank_stat"], 3)) < 1e-15,
      strong["rank_df"])

# v2.8.0 review r1 (S1-1): the rank statistic grows with n under any alternative, so it is not a first-stage F and
# carries no cut of 10. Before the fix, an instrument with a first-stage F of 1e8 (n = 100, 8 instruments) got
# "rank statistic / df" of 5 and a "weak identification" warning.
r_s = np.random.default_rng(1)
z_s = r_s.standard_normal((100, 8))
v_s = r_s.standard_normal(100)
x_s = 1000.0 * z_s.sum(1) + v_s
y_s = 1.0 + 2.0 * x_s + 0.5 * v_s + r_s.standard_normal(100)
X_s, Z_s = np.column_stack([np.ones(100), x_s]), np.column_stack([np.ones(100), z_s])
F_s = float(np.squeeze(sm.OLS(x_s, Z_s).fit().f_test(np.eye(9)[1:]).fvalue))
sat = est.gmm(_lin_moments, [0.0, 0.0], args=(y_s, X_s, Z_s), show=False)
check("v2.8.0 gmm: first-stage F of 1e8 -> rank statistic about 40 (estimation-theory.md), no identification warning "
      "and no 'ident_F' key (the old cut called it weak)",
      F_s > 1e7 and 35 < sat["rank_stat"] < 45 and sat["rank_p"] < 0.05 and "ident_F" not in sat
      and not any("identif" in w["message"] for w in sat["warnings"]), (F_s, sat["rank_stat"]))
# the old seeds on both sides of the dropped cut: both reject rank < 2, neither gets a warning now
wk_below = est.gmm(_lin_moments, [0.0, 0.0], args=_iv_case(0.08, 0), show=False)
check("v2.8.0 gmm: a weak but rank-identified design (pi 0.08) -> rank test rejects, no 'weak identification' "
      "verdict from the statistic", wk_below["rank_p"] < 0.05
      and not any("weak identification" in w["message"] for w in wk_below["warnings"]), wk_below["rank_p"])
none_ = est.gmm(_lin_moments, [0.0, 0.0], args=_iv_case(0.0, 0), show=False)
check("v2.8.0 gmm: irrelevant instruments (rank test p = 0.38) -> 'may not be identified' warning",
      none_["rank_p"] >= 0.05 and any("may not be identified" in w["message"] for w in none_["warnings"]),
      none_["rank_p"])
# size of the rank test under H0 (pi = 0: rank(D) = 1 < 2), 300 samples; band fixed before running: [0.02, 0.085]
rej0 = np.mean([est.gmm(_lin_moments, [0.0, 0.0], args=_iv_case(0.0, 500 + sd), show=False)["rank_p"] < 0.05
                for sd in range(300)])
check("v2.8.0 gmm: rank test rejects rank < 2 in [0.02, 0.085] of 300 samples when pi = 0 (size at 5%)",
      0.02 <= rej0 <= 0.085, rej0)

# the rank statistic with HAC, against a formula written here (S2-5: a mutation that dropped the lags survived).
# k = 1, moments z_t (y_t - x_t theta): the derivative is -z_t x_t, free of theta, so the statistic is
# n dbar' Omega^-1 dbar with Omega the Bartlett long-run covariance of z_t x_t (double sum over pairs). The rank
# test always uses Bartlett, even with kernel="truncated" (S1-2: the truncated variance went negative).
r_h = np.random.default_rng(83)
nh = 150
zh = r_h.standard_normal((nh, 3))
for t in range(1, nh):
    zh[t] += 0.6 * zh[t - 1]
xh = 0.3 * zh.sum(1) + r_h.standard_normal(nh)
yh = 1.5 * xh + r_h.standard_normal(nh)
dh = zh * xh[:, None]
Uh = dh - dh.mean(0)
Om_h = np.zeros((3, 3))
for t in range(nh):
    for s_ in range(nh):
        if abs(t - s_) <= 4:
            Om_h += (1 - abs(t - s_) / 5) * np.outer(Uh[t], Uh[s_])
Om_h /= nh
rk_cf = nh * dh.mean(0) @ np.linalg.solve(Om_h, dh.mean(0))
mom_h = lambda th, y, x, z: (y - x * th[0])[:, None] * z
for kern in ("bartlett", "truncated"):
    gh_ = est.gmm(mom_h, [0.0], args=(yh, xh, zh), hac_lags=4, kernel=kern, show=False)
    check(f"v2.8.0 gmm: rank statistic with hac_lags=4, kernel={kern} = n dbar' Omega_Bartlett^-1 dbar (rel <= 1e-6)",
          abs(gh_["rank_stat"] / rk_cf - 1) < 1e-6 and gh_["rank_df"] == 3
          and gh_["used"]["rank_test_cov"] == "Bartlett HAC with 4 lags", (gh_["rank_stat"], rk_cf))
gi_ = est.gmm(mom_h, [0.0], args=(yh, xh, zh), show=False)
Ui = dh - dh.mean(0)
check("v2.8.0 gmm: rank statistic without lags uses the iid covariance (differs from the HAC one here)",
      abs(gi_["rank_stat"] / (nh * dh.mean(0) @ np.linalg.solve(Ui.T @ Ui / nh, dh.mean(0))) - 1) < 1e-6
      and abs(gi_["rank_stat"] / rk_cf - 1) > 0.05 and gi_["used"]["rank_test_cov"] == "iid")

# S1-2: the review's design (differenced instruments, n = 120, truncated kernel, 3 lags): the old rank statistic was
# negative in 40 of 40 seeds (-6.4e3 to -4.7e16) with a false 'may not be identified'
neg_rk, notid = 0, 0
for sd in range(40):
    r_t = np.random.default_rng(sd)
    zt = r_t.standard_normal((121, 2))
    zt = zt[1:] - zt[:-1]
    et = r_t.standard_normal(121)
    vt = et[1:] - et[:-1]
    xt = 0.5 * zt.sum(1) + vt
    yt = 1.0 + xt + vt + r_t.standard_normal(120)
    gt_ = est.gmm(_lin_moments, [0.0, 0.0], args=(yt, np.column_stack([np.ones(120), xt]),
                                                   np.column_stack([np.ones(120), zt])),
                  hac_lags=3, kernel="truncated", show=False)
    neg_rk += not gt_["rank_stat"] >= 0
    notid += any("may not be identified" in w["message"] for w in gt_["warnings"])
check("v2.8.0 gmm: truncated kernel, review design, 40 seeds: no negative rank statistic and no false "
      "'may not be identified'", neg_rk == 0 and notid == 0, (neg_rk, notid))

# S3-1: a direction in which the derivatives do not vary (the moment x - theta has derivative -1 on every row):
# exactly identified along it; the statistic is inf, not 1e31 of rounding nor 0 from a pseudo-inverse
xd = np.random.default_rng(84).standard_normal(300)
gd_ = est.gmm(lambda th, x: np.column_stack([x - th[0], (x - th[0]) ** 2 - 1]), [0.1], args=(xd,), show=False)
check("v2.8.0 gmm: derivatives constant across rows -> rank statistic inf, p = 0, no identification warning",
      gd_["rank_stat"] == np.inf and gd_["rank_p"] == 0.0
      and not any("identif" in w["message"] for w in gd_["warnings"]), gd_["rank_stat"])

# review r2, S1-1: the rank statistic is invariant to invertible recombinations of the moments. An exogenous w + off
# is the moment of w plus off times the moment of the constant; an instrument times s is a rescaled moment. The cut
# of eigenvalues used one scale for all moments and dropped the small one (off 1000 -> 39.8 instead of 78.1;
# instrument x 1e-6 -> 0.17, p 0.92 and "may not be identified"). Reference: the same minimum Wald on the
# well-scaled design (off 0, scale 1), computed here by Powell from 30 starts, which needs no cut.
def _inv_design(off=0.0, s1=1.0, s2=1.0, seed=0, n=500):
    r = np.random.default_rng(seed)
    z1, z2, v = r.standard_normal(n), r.standard_normal(n), r.standard_normal(n)
    w = r.integers(0, 3, n).astype(float)
    x = 0.5 * z1 + v
    y = 1 + 2 * x + 0.5 * v + r.standard_normal(n)
    return y, np.column_stack([np.ones(n), w + off, x]), np.column_stack([np.ones(n), w + off, s1 * z1, s2 * z2])


def _min_wald_ref(y, X, Z, lags=None):
    d = -(X[:, None, :] * Z[:, :, None])          # n x m x k: d g_i / d theta' for g_i = (y - X theta) Z_i
    nn = len(y)

    def f(c):
        c = c / np.linalg.norm(c)
        u = d @ c
        S = _cf_hac(u - u.mean(0), lags)
        return nn * u.mean(0) @ np.linalg.solve(S, u.mean(0))
    rr = np.random.default_rng(5)
    return min(optimize.minimize(f, rr.standard_normal(X.shape[1]), method="Powell",
                                 options={"xtol": 1e-10, "ftol": 1e-13}).fun for _ in range(30))


def _cf_hac(U, lags):
    S = U.T @ U / len(U)
    for j in range(1, (lags or 0) + 1):
        G = U[j:].T @ U[:-j] / len(U)
        S = S + (1 - j / (lags + 1)) * (G + G.T)
    return S


y_i0, X_i0, Z_i0 = _inv_design()
ref_i0 = _min_wald_ref(y_i0, X_i0, Z_i0)
ref_i3 = _min_wald_ref(y_i0, X_i0, Z_i0, lags=3)
inv_bad = []
for off_, s1_, s2_ in [(0.0, 1.0, 1.0), (1e3, 1.0, 1.0), (3e3, 1.0, 1.0), (1e4, 1.0, 1.0), (0.0, 1e-6, 1.0),
                       (0.0, 1.0, 1e6), (1e3, 1e-6, 1.0), (1e3, 1e-3, 1e4)]:
    yi, Xi, Zi = _inv_design(off_, s1_, s2_)
    for lg_, ref_ in ((None, ref_i0), (3, ref_i3)):
        ri = est.gmm(_lin_moments, [0.0, 0.0, 0.0], args=(yi, Xi, Zi), weight=np.diag(1 / np.mean(Zi ** 2, 0)),
                     hac_lags=lg_, steps="one-step", show=False)
        if not abs(ri["rank_stat"] / ref_ - 1) < 1e-7 or any("identif" in w["message"] for w in ri["warnings"]):
            inv_bad.append((off_, s1_, s2_, lg_, ri["rank_stat"], ref_))
check("v2.8.0 gmm (r2 S1-1, r3 S2-1): rank statistic invariant to an exogenous offset (1e3, 3e3 and 1e4, with and "
      "without 3 HAC lags), an instrument x 1e-6 or x 1e6 and both together: equal to the well-scaled minimum Wald found by Powell "
      "(rel <= 1e-7), never 'may not be identified'", not inv_bad and 50 < ref_i0 < 120, (inv_bad, ref_i0, ref_i3))

# review r3, S2-1: with an offset of 1e4 and HAC the search over directions got stuck in a valley of width ~1/offset
# and the result depended on the random starts (2/12 seeds gave 84.35 instead of 80.66; once in CI). The search is now
# in preconditioned coordinates: the statistic is the same for 12 seeds of the starts (the generator inside the search
# is replaced for the call, then restored), at 1e4 with 3 lags and at 3e4 without lags, identified and not.
_real_rng = np.random.default_rng
seed_bad = []
for off_, lg_, pi_ in [(1e4, 3, 0.5), (3e4, None, 0.5), (1e4, 3, 0.0)]:
    r_ = _real_rng(0)
    z1s, z2s, vs = r_.standard_normal(500), r_.standard_normal(500), r_.standard_normal(500)
    ws = r_.integers(0, 3, 500).astype(float)
    xs = pi_ * z1s + vs
    ys = 1 + 2 * xs + 0.5 * vs + r_.standard_normal(500)
    Xs, Zs = np.column_stack([np.ones(500), ws + off_, xs]), np.column_stack([np.ones(500), ws + off_, z1s, z2s])
    ths = np.linalg.solve(Xs.T @ Xs, Xs.T @ ys)
    Gs = lambda t, Xs=Xs, Zs=Zs, ys=ys: (ys - Xs @ t)[:, None] * Zs
    vals_s = []
    for sd_ in range(12):
        np.random.default_rng = (lambda seed=None, sd_=sd_: _real_rng(1000 + sd_))
        try:
            vals_s.append(est._gmm_rank_test(Gs, ths, lg_, est._gmm_steps(Gs, ths, ths))[0])
        finally:
            np.random.default_rng = _real_rng
    if not max(vals_s) - min(vals_s) <= 1e-8 * max(vals_s):
        seed_bad.append((off_, lg_, pi_, min(vals_s), max(vals_s)))
check("v2.8.0 gmm (r3 S2-1): rank statistic with an offset of 1e4 and 3 lags, and 3e4 without lags: the same for 12 "
      "seeds of the search's random starts (rel 1e-8: the optimizer's tolerance; the valley gave 3%)", not seed_bad and np.random.default_rng is _real_rng, seed_bad)

# review r3, S1-1/S1-3/S1-5/S1-2: moments that do not depend on some parameters. The minimum Wald lies in a moment's
# null space, where that moment is dropped; a search over all directions missed it (block of moments in theta1 alone
# and a weak block in theta2: 140.6 and no warning, instead of 5.64 and the warning; k = 3 with the strong moment in
# theta1 + theta3: 176 instead of 0.54; a parameter outside the moments crashed with k = 1 and 3). Reference: the
# Wald of the weak block alone, written here (k = 1: n dbar' Var^-1 dbar of d_i = -(z2, z3) x2).
r_b = np.random.default_rng(1)
z1b = r_b.standard_normal(500); x1b = z1b + r_b.standard_normal(500); y1b = x1b + r_b.standard_normal(500)
z2b, z3b = r_b.standard_normal(500), r_b.standard_normal(500)
x2b = 0.02 * z2b + r_b.standard_normal(500); y2b = 0.5 * x2b + r_b.standard_normal(500)
dwb = -np.column_stack([z2b * x2b, z3b * x2b])
ref_weak = float(500 * dwb.mean(0) @ np.linalg.solve(np.cov(dwb.T, bias=True), dwb.mean(0)))
blk_bad = []
for name_, mom_, th0_ in [
        ("block", lambda t: np.column_stack([z1b * (y1b - t[0] * x1b), z2b * (y2b - t[1] * x2b), z3b * (y2b - t[1] * x2b)]),
         [0.5, 0.2]),
        ("sum", lambda t: np.column_stack([z1b * (y1b - (t[0] + t[1]) * x1b), z2b * (y2b - t[1] * x2b),
                                           z3b * (y2b - t[1] * x2b)]), [0.5, 0.2]),
        ("nonlinear", lambda t: np.column_stack([z1b * (y1b - np.exp(1e-4 * t[0] + t[1]) * x1b),
                                                 z2b * (y2b - np.sin(t[1]) * x2b), z3b * (y2b - np.sin(t[1]) * x2b)]),
         [-1000.0, 0.2])]:
    gb_ = est.gmm(mom_, th0_, show=False)
    # nonlinear case: the weak block's derivative is cos(theta2) times the linear one; the Wald does not change
    ok_ = abs(gb_["rank_stat"] / ref_weak - 1) < 1e-6 and any("may not be identified" in w["message"]
                                                                for w in gb_["warnings"])
    if not ok_:
        blk_bad.append((name_, gb_["rank_stat"], ref_weak))
check("v2.8.0 gmm (r3 S1-1): a block of moments in theta1 alone (or in theta1 + theta2, or nonlinear in "
      "1e-4 theta1 + theta2) and a weak block in theta2: rank statistic = Wald of the weak block alone (rel 1e-6), "
      "p > 0.05 and the 'may not be identified' warning", not blk_bad and 5 < ref_weak < 6.5, (blk_bad, ref_weak))

r_c = np.random.default_rng(2)
zac, zbc = r_c.standard_normal(500), r_c.standard_normal(500)
xac = zac + r_c.standard_normal(500); xbc = zbc + r_c.standard_normal(500)
yac = xac + r_c.standard_normal(500); ybc = xbc + r_c.standard_normal(500)
z2c, z3c = r_c.standard_normal(500), r_c.standard_normal(500)
x2c = 0.02 * z2c + r_c.standard_normal(500); y2c = 0.5 * x2c + r_c.standard_normal(500)
dwc = -np.column_stack([z2c * x2c, z3c * x2c])
ref_k3 = float(500 * dwc.mean(0) @ np.linalg.solve(np.cov(dwc.T, bias=True), dwc.mean(0)))
k3_vals = []
for rot_ in (False, True):
    mk3 = (lambda t, rot_=rot_: np.column_stack([zac * (yac - (t[0] + (t[2] if rot_ else 0.0)) * xac),
                                                 zbc * (ybc - t[1] * xbc), z2c * (y2c - t[2] * x2c),
                                                 z3c * (y2c - t[2] * x2c)]))
    k3_vals.append(est._gmm_rank_test(mk3, np.array([1.0, 1.0, 0.1]), None)[0])
check("v2.8.0 gmm (r3 S1-5): k = 3, weak block in theta3, strong moments in theta1 (or theta1 + theta3) and theta2: "
      "rank statistic = Wald of the weak block alone (rel 1e-6; the search over all directions gave 176)",
      all(abs(v / ref_k3 - 1) < 1e-6 for v in k3_vals) and ref_k3 < 2, (k3_vals, ref_k3))

r_u = np.random.default_rng(0)
zu = r_u.standard_normal((300, 3)); xu = zu.sum(1) + r_u.standard_normal(300); yu = 2 * xu + r_u.standard_normal(300)
un_res = []
for k_, mom_, th0_ in [(1, lambda t: (yu - 2.0 * xu)[:, None] * zu, [0.3]),
                       (2, lambda t: (yu - t[0] * xu)[:, None] * zu, [1.0, 0.5]),
                       (3, lambda t: (yu - t[0] * xu - t[1] * zu[:, 0])[:, None] * np.column_stack([zu, zu[:, 0] ** 2]),
                        [1.0, 0.0, 0.5])]:
    try:
        gu_ = est.gmm(mom_, th0_, show=False)
        un_res.append((k_, gu_["rank_stat"], any("may not be identified" in w["message"] for w in gu_["warnings"])))
    except Exception as e_:
        un_res.append((k_, repr(e_)[:80], False))
check("v2.8.0 gmm (r3 S1-2): a parameter that does not enter the moments, k = 1, 2, 3: rank statistic 0 and the "
      "'may not be identified' warning, no exception", all(v == 0.0 and w for _, v, w in un_res), un_res)

# review r2 (finding before r1): with the offset, S of the moments looked singular on its own scale, was floored,
# and the estimate moved (2.0486 at off 1000 against 2.0212). Iterated GMM is a fixed point and invariant to the
# recombination: the estimate and SEs at off 1e3 and 1e4 equal those at off 0, without the positive-definiteness warning
it_ref = est.gmm(_lin_moments, [0.0, 0.0, 0.0], args=(y_i0, X_i0, Z_i0), steps="iterated", tol=1e-12, show=False)
it_bad = []
for off_ in (1e3, 1e4):
    yi, Xi, Zi = _inv_design(off_)
    ri = est.gmm(_lin_moments, [0.0, 0.0, 0.0], args=(yi, Xi, Zi), weight=np.diag(1 / np.mean(Zi ** 2, 0)),
                 steps="iterated", tol=1e-12, show=False)
    dx = abs(ri["params"].iloc[2] - it_ref["params"].iloc[2])
    dse = abs(ri["table"]["se"].iloc[2] / it_ref["table"]["se"].iloc[2] - 1)
    if not (dx < 1e-6 and dse < 1e-5) or any("positive definite" in w["message"] for w in ri["warnings"]):
        it_bad.append((off_, ri["params"].iloc[2], it_ref["params"].iloc[2], dse))
check("v2.8.0 gmm (r2): iterated GMM with an exogenous offset of 1e3 and 1e4: same slope (abs <= 1e-6) and SE "
      "(rel <= 1e-5) as with no offset, and no 'S is not positive definite' warning", not it_bad, it_bad)

# review r2, S2-4: with persistent derivatives the rank test over-rejects (size 0.21 at a lag-1 autocorrelation of 0.5
# without lags, 0.39 at 0.9 with 4 lags; claude/medicoes-v2.8.0.md M7). Warning when the largest lag-1 autocorrelation
# of vec(dg_i/dtheta') passes 0.4 (no lags) or 0.6 (lags) and the statistic divided by the AR(1) long-run variance
# factor the kernel misses no longer rejects. used["rank_test_ac1"] against np.corrcoef on the lagged columns.
def _ar_p(r, n, rho, k):
    e = r.standard_normal((n + 200, k))
    x = np.zeros_like(e)
    for t in range(1, n + 200):
        x[t] = rho * x[t - 1] + e[t]
    return x[200:]


def _persist_fit(rho, pi, lags, seed, n=240, data=False):
    r = np.random.default_rng(seed)
    z = _ar_p(r, n, rho, 3)
    x = pi * z.sum(1) + _ar_p(r, n, rho, 1)[:, 0] + 1.0
    X, Z = np.column_stack([np.ones(n), x]), np.column_stack([np.ones(n), z])
    y = x + r.standard_normal(n)
    if data:
        return y, X, Z
    return est.gmm(_lin_moments, [0.0, 1.0], args=(y, X, Z), hac_lags=lags, steps="one-step", show=False), X, Z


per_h0, Xp, Zp = _persist_fit(0.9, 0.0, None, 1)
Dp = -(Zp[:, :, None] * Xp[:, None, :]).reshape(240, -1)
ac_ref = max(abs(np.corrcoef(Dp[1:, j], Dp[:-1, j])[0, 1]) for j in range(Dp.shape[1]) if Dp[:, j].std() > 0)
pw = lambda g: any(w["level"] == "warning" and "persistent" in w["message"] for w in g["warnings"])
check("v2.8.0 gmm (r2 S2-4): AR(0.9) instruments and regressor, irrelevant instruments, no lags: the rank test "
      "rejects but gets the persistence warning; used['rank_test_ac1'] equals the lag-1 autocorrelation computed "
      "by np.corrcoef (abs 0.01)", per_h0["rank_p"] < 0.05 and pw(per_h0)
      and abs(per_h0["used"]["rank_test_ac1"] - ac_ref) < 0.01, (per_h0["rank_p"], per_h0["used"]["rank_test_ac1"], ac_ref))
# review r3, S2-3/S2-4 (decision of 2026-11-26: always warn, with the condition in the text): the check reads the
# row order as time order (a cross-section sorted by x has a lag-1 autocorrelation of 0.97 with iid data) and sees
# only lag-1 persistence (seasonal persistence escapes it). The message and the action say both; and printed by show.
pw_msg = [w for w in per_h0["warnings"] if "persistent" in w["message"]]
check("v2.8.0 gmm (r3 S2-3/S2-4): the persistence warning says it holds only if the rows are in time order and not "
      "in a cross-section, and its action that the check sees only lag-1 persistence (seasonal escapes it)",
      len(pw_msg) == 1 and "if the rows are in time order" in pw_msg[0]["message"]
      and "cross-section" in pw_msg[0]["message"] and "lag-1" in pw_msg[0]["action"]
      and "seasonal" in pw_msg[0]["action"], [w["message"][-160:] for w in pw_msg])
_buf_pw = io.StringIO()
with contextlib.redirect_stdout(_buf_pw):
    est.gmm(_lin_moments, [0.0, 1.0], args=_persist_fit(0.9, 0.0, None, 1, data=True), steps="one-step", show=True)
check("v2.8.0 gmm (r3 S2-3): show prints the persistence warning with its time-order condition",
      "if the rows are in time order" in _buf_pw.getvalue(), _buf_pw.getvalue()[-300:])
per_iid, _, _ = _persist_fit(0.0, 0.3, None, 1)
per_mod, _, _ = _persist_fit(0.5, 0.3, 4, 1)
check("v2.8.0 gmm (r2 S2-4): strong instruments with iid data, and with AR(0.5) data and 4 lags: rejects, no "
      "persistence warning", per_iid["rank_p"] < 0.05 and not pw(per_iid) and per_mod["rank_p"] < 0.05
      and not pw(per_mod), (per_iid["rank_p"], per_mod["rank_p"], per_mod["used"]["rank_test_ac1"]))
per_b, _, _ = _persist_fit(0.8, 0.3, 8, 0)
check("v2.8.0 gmm (r2 S2-4): AR(0.8) data, informative instruments, 8 Bartlett lags (lag-1 autocorrelation 0.84, p "
      "2.7e-4): deflated only by the long-run variance the 8 lags miss, the statistic still rejects -> no warning "
      "(without the credit for the lags it would warn)", per_b["rank_p"] < 0.05 and per_b["used"]["rank_test_ac1"] >= 0.6
      and not pw(per_b), (per_b["rank_p"], per_b["used"]["rank_test_ac1"]))
check("v2.8.0 gmm (r2 S2-4): the Euler equation simulated with n = 4998 (lag-1 autocorrelation about 0.4, p 1e-193) "
      "gets no persistence warning: divided by the long-run factor the statistic still rejects", not pw(ge), ge["rank_p"])

# review r2, S3: a year-like regressor next to the constant makes the Wald inf near an axis; BFGS then printed 15
# RuntimeWarnings to the user's console (the reviewer's e5.py, seed 0)
r_y = np.random.default_rng(0)
yr_ = r_y.integers(2018, 2021, 600).astype(float)
z1y, z2y, vy = r_y.standard_normal(600), r_y.standard_normal(600), r_y.standard_normal(600)
xy_ = 0.4 * z1y + 0.3 * z2y + vy
yy_ = 1 + 0.1 * (yr_ - 2019) + 2 * xy_ + 0.5 * vy + r_y.standard_normal(600)
with warnings.catch_warnings(record=True) as rec_y:
    warnings.simplefilter("always")
    gy_ = est.gmm(_lin_moments, [0.0, 0.0, 0.0], args=(yy_, np.column_stack([np.ones(600), yr_, xy_]),
                                                       np.column_stack([np.ones(600), yr_, z1y, z2y])), show=False)
check("v2.8.0 gmm (r2 S3): rank test with a year-like regressor (k = 3): no RuntimeWarning reaches the user, and the "
      "statistic is finite and rejects", not any(issubclass(q.category, RuntimeWarning) for q in rec_y)
      and np.isfinite(gy_["rank_stat"]) and gy_["rank_p"] < 0.05,
      ([str(q.message)[:60] for q in rec_y], gy_["rank_stat"]))

# S2-2: k >= 3 in seconds (k = 5, m = 10, n = 2000 took 397 s), and the minimum over directions against an
# independent optimizer (Powell from 40 random starts on the Wald of D(c) computed here)
r_k = np.random.default_rng(85)
nk = 2000
zk = r_k.standard_normal((nk, 9))
vk = r_k.standard_normal((nk, 4))
xk = 0.5 * zk[:, :1] + 0.5 * zk[:, :4] + vk
yk = 1.0 + xk.sum(1) + 0.5 * vk[:, 0] + r_k.standard_normal(nk)
Xk, Zk = np.column_stack([np.ones(nk), xk]), np.column_stack([np.ones(nk), zk])
import time as _time
t0_ = _time.time()
gk_ = est.gmm(_lin_moments, np.zeros(5), args=(yk, Xk, Zk), show=False)
dt_ = _time.time() - t0_
Dk = -(Zk[:, :, None] * Xk[:, None, :])                     # derivative of z_i (y_i - x_i'theta): -z_i x_i'


def _wald_k(c):
    c = c / np.linalg.norm(c)
    d = Dk @ c
    u = d - d.mean(0)
    return nk * d.mean(0) @ np.linalg.solve(u.T @ u / nk, d.mean(0))


rk_ref = min(optimize.minimize(_wald_k, r_k.standard_normal(5), method="Powell").fun for _ in range(40))
check("v2.8.0 gmm: k = 5, m = 10, n = 2000: the whole fit in under 60 s and the rank statistic equals the minimum "
      "found by Powell from 40 starts (rel <= 1e-6)", dt_ < 60 and abs(gk_["rank_stat"] / rk_ref - 1) < 1e-6,
      (dt_, gk_["rank_stat"], rk_ref))

# truncated kernel with an indefinite S (an MA(1) with a unit root in the MA part, two lags; seed 1)
r_neg = np.random.default_rng(1)
e_neg = r_neg.standard_normal(81)
x_neg = e_neg[1:] - e_neg[:-1]
gneg = est.gmm(lambda th, x: (x - th[0])[:, None], [0.0], args=(x_neg,), hac_lags=2, kernel="truncated", show=False)
check("v2.8.0 gmm: truncated kernel with a negative S -> warning, a positive weight, finite estimate",
      any(w["level"] == "warning" and "not positive definite" in w["message"] for w in gneg["warnings"])
      and np.isfinite(gneg["params"].iloc[0]) and np.linalg.eigvalsh(gneg["weight"]).min() > 0)
gpos = est.gmm(lambda th, x: (x - th[0])[:, None], [0.0], args=(x_neg,), hac_lags=2, kernel="bartlett", show=False)
check("v2.8.0 gmm: the same data with the Bartlett kernel -> no positive-definiteness warning",
      not any("positive definite" in w["message"] for w in gpos["warnings"]))

# iterated GMM that is not allowed to settle
g_ns = est.gmm(_lin_moments, [0.0, 0.0], args=(y80, X80, Z80), steps="iterated", maxiter=1, tol=0.0, show=False)
check("v2.8.0 gmm: iterated GMM stopped by maxiter -> converged False and the warning",
      g_ns["converged"] is False and any("did not settle" in w["message"] for w in g_ns["warnings"]))

# few observations per moment
few = est.gmm(_lin_moments, [0.0, 0.0], args=(y80[:39], X80[:39], Z80[:39]), show=False)
enough = est.gmm(_lin_moments, [0.0, 0.0], args=(y80[:40], X80[:40], Z80[:40]), show=False)
check("v2.8.0 gmm: 39 observations for 4 moments -> notice; 40 -> no notice",
      any("per moment" in w["message"] for w in few["warnings"])
      and not any("per moment" in w["message"] for w in enough["warnings"]))

# stdout: labels the reader sees
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    est.gmm(_lin_moments, [0.0, 0.0], args=_iv_case(0.0, 0), names=["const", "x"])
txt = buf.getvalue()
check("v2.8.0 gmm show: prints the method, the table, J with 'p (right tail)', the CDF labelled as not the "
      "p-value, the identification line saying a rejection does not rule out weak identification, and each finding "
      "with its action",
      "GMM (two-step)" in txt and "estimate" in txt and "p (right tail)" in txt and "not the p-value" in txt
      and "Cragg-Donald" in txt and "does not rule out weak identification" in txt
      and "WARNING: the parameters may not be identified" in txt and "action:" in txt
      and "per df" not in txt and "weak below" not in txt, txt[-500:])
check("v2.8.0 gmm: 'used' records the choices actually applied", g80["used"]["steps"] == "two-step"
      and g80["used"]["first_weight"] == "identity" and g80["used"]["n"] == n80 and g80["used"]["m"] == 4
      and g80o["used"]["first_weight"] == "user" and g80["used"]["skill_version"])

# refusals
for bad, kw in (("steps", dict(steps="three")), ("kernel", dict(kernel="parzen", hac_lags=2)),
                ("hac_lags", dict(hac_lags=-1))):
    try:
        est.gmm(_lin_moments, [0.0, 0.0], args=(y80, X80, Z80), show=False, **kw)
        refused = False
    except ValueError:
        refused = True
    check(f"v2.8.0 gmm: refuses a bad {bad}", refused)
try:
    est.gmm(_lin_moments, [0.0, 0.0, 0.0], args=(y80, X80[:, :1].repeat(3, 1), Z80[:, :2]), show=False)
    under = False
except ValueError:
    under = True
check("v2.8.0 gmm: fewer moments than parameters is refused (not identified)", under)

# S2-3: a user Jacobian is checked once against central differences; transposed or wrong is refused (it was reshaped
# silently and gave other estimates and SEs with converged=True)
jac_ok = lambda th, y, X, Z: -(Z.T @ X) / len(y)                 # d mean(z (y - X theta)) / d theta' : m x k
gj = est.gmm(_lin_moments, [0.0, 0.0], args=(y80, X80, Z80), jacobian=jac_ok, show=False)
check("v2.8.0 gmm: the right analytic Jacobian is accepted and gives the closed-form estimate (rel <= 1e-9)",
      np.allclose(gj["params"].to_numpy(), th2, rtol=1e-9, atol=0))
for lab, jf in (("transposed (k x m)", lambda th, y, X, Z: (-(Z.T @ X) / len(y)).T),
                ("wrong values, right shape", lambda th, y, X, Z: -2.0 * (Z.T @ X) / len(y))):
    try:
        est.gmm(_lin_moments, [0.0, 0.0], args=(y80, X80, Z80), jacobian=jf, show=False)
        msg = ""
    except ValueError as err:
        msg = str(err)
    check(f"v2.8.0 gmm: a {lab} Jacobian is refused with a message about shape or disagreement",
          ("must return an array of shape (m, k) = (4, 2)" in msg) if "transposed" in lab else ("disagrees" in msg), msg)
Zsq = Z80[:, :2]                                                  # m = k = 2: a transposed square Jacobian
try:
    est.gmm(_lin_moments, [0.0, 0.0], args=(y80, X80, Zsq), jacobian=lambda th, y, X, Z: (-(Z.T @ X) / len(y)).T,
            show=False)
    sq = ""
except ValueError as err:
    sq = str(err)
check("v2.8.0 gmm: a transposed square Jacobian (m = k) is refused by the check against central differences",
      "disagrees" in sq, sq)

# S3-2: moments that become NaN during the search stop with a message that says so (it was numpy's error)
xn = np.random.default_rng(86).standard_normal(200) + 2.0
try:
    est.gmm(lambda th, x: np.column_stack([x - th[0], (x - th[0]) * np.where(th[0] > 1.5, np.nan, 1.0)]), [0.0],
            args=(xn,), show=False)
    nan_msg = ""
except ValueError as err:
    nan_msg = str(err)
check("v2.8.0 gmm: moments that turn NaN in the search -> ValueError naming theta and suggesting bounds",
      "NaN or inf at theta" in nan_msg and "bounds" in nan_msg, nan_msg)

# review r2, S1-2: the Jacobian check compared the whole matrix, so a wrong column in small units hid behind a
# large one (regressor in units of 1e-4, second column x 2: accepted, SE 172.7 instead of 345.3). Now per column.
r_j = np.random.default_rng(0)
zj, vj = r_j.standard_normal((400, 3)), r_j.standard_normal(400)
xj = zj.sum(1) + vj
Xj = np.column_stack([np.ones(400), 1e-4 * xj])
Zj = np.column_stack([np.ones(400), zj])
yj = 1 + 2.0 * xj + 0.5 * vj + r_j.standard_normal(400)
jac_j = lambda th, y, X, Z: -(Z.T @ X) / len(y)


def _jac_col2(th, y, X, Z):
    D = -(Z.T @ X) / len(y)
    D[:, 1] *= 2.0
    return D


try:
    est.gmm(_lin_moments, [0.0, 0.0], args=(yj, Xj, Zj), jacobian=_jac_col2, show=False)
    col_msg = ""
except ValueError as err:
    col_msg = str(err)
gj_ok = est.gmm(_lin_moments, [0.0, 0.0], args=(yj, Xj, Zj), jacobian=jac_j, show=False)
gj_num = est.gmm(_lin_moments, [0.0, 0.0], args=(yj, Xj, Zj), show=False)
check("v2.8.0 gmm (r2 S1-2): a Jacobian with one wrong column in small units is refused and the message names "
      "the column; the right one is accepted with the SEs of the numerical fit (rel <= 1e-6)",
      "column 1" in col_msg and "disagrees" in col_msg
      and np.allclose(gj_ok["table"]["se"], gj_num["table"]["se"], rtol=1e-6, atol=0), col_msg)

# S2-3 and the steps: a parameter of order 1e-4 (an exponential rate). The right analytic Jacobian was refused
# (finite-difference step with an absolute floor), and without a Jacobian the fit moved by 1.4e-4 (estimate) and
# 0.3% (SE). Reference: the same moments with theta in units of 1e-4 (lam = 1e4 theta, a parameter near 2).
x_r = np.random.default_rng(0).exponential(1e4, size=2000)
mom_r = lambda th, x: np.column_stack([x - 1 / th[0], x ** 2 - 2 / th[0] ** 2])
jac_r = lambda th, x: np.array([[1 / th[0] ** 2], [4 / th[0] ** 3]])


def _gmm_or_failed(*a, **kw):
    """est.gmm, or a stand-in with NaN numbers when it raises, so the checks below fail one by one instead of
    stopping the file (review r3, S3: two mutants were caught only by the exception)."""
    try:
        return est.gmm(*a, **kw)
    except Exception as e_:                       # noqa: BLE001
        print(f"  (gmm raised {type(e_).__name__}: {str(e_)[:100]})")
        k_ = len(np.atleast_1d(a[1]))
        return {"params": pd.Series(np.full(k_, np.nan)), "table": pd.DataFrame({"se": np.full(k_, np.nan)}),
                "converged": False, "warnings": [], "rank_p": np.nan, "rank_stat": np.nan}


g_rn = _gmm_or_failed(mom_r, [2e-4], args=(x_r,), show=False)
g_ra = _gmm_or_failed(mom_r, [2e-4], args=(x_r,), jacobian=jac_r, show=False)
g_rl = est.gmm(lambda la, x: mom_r(np.asarray(la) * 1e-4, x), [2.0], args=(x_r,), show=False)
eq_r = [(g["params"].iloc[0] / (g_rl["params"].iloc[0] * 1e-4) - 1, g["table"]["se"].iloc[0] /
         (g_rl["table"]["se"].iloc[0] * 1e-4) - 1) for g in (g_rn, g_ra)]
check("v2.8.0 gmm (r2 S2-3): theta of order 1e-4: the right analytic Jacobian is accepted, and with or without it "
      "the estimate and SE equal those with theta rescaled to order 1 (rel <= 1e-7 and 1e-5)",
      all(abs(a) < 1e-7 and abs(b) < 1e-5 for a, b in eq_r), eq_r)

# review r3, S2-2: steps relative to max(|theta|, |theta0|) brought the error back when theta0 is far from the
# estimate (the usual start of 1: estimate off by 1.4e-4, SE by 3.4e-3); steps relative to the estimate would fail
# when it is 0 by chance. The steps are now on the scale of the response. (a) rate of order 1e-4 started at 1.
with warnings.catch_warnings(record=True) as rec_r1:
    warnings.simplefilter("always")
    g_r1 = _gmm_or_failed(mom_r, [1.0], args=(x_r,), bounds=[(1e-12, None)], show=False)
check("v2.8.0 gmm (r3 S3): started at 1 the search tries steps where 1/theta is undefined: no RuntimeWarning reaches "
      "the user (it printed the user's 'divide by zero' and a matmul 'invalid value')",
      not any(issubclass(q.category, RuntimeWarning) for q in rec_r1), [str(q.message)[:60] for q in rec_r1])
eq_r1 = (g_r1["params"].iloc[0] / (g_rl["params"].iloc[0] * 1e-4) - 1,
         g_r1["table"]["se"].iloc[0] / (g_rl["table"]["se"].iloc[0] * 1e-4) - 1)
check("v2.8.0 gmm (r3 S2-2): theta of order 1e-4 started at theta0 = 1: estimate and SE equal those with theta "
      "rescaled to order 1 (rel <= 1e-7 and 1e-5)", g_r1["converged"] and abs(eq_r1[0]) < 1e-7 and abs(eq_r1[1]) < 1e-5,
      eq_r1)
# (b) exactly identified IV whose slope estimate is 0 up to rounding (the error is made orthogonal to the
# instrument in the sample), started at (1, 1): SEs against the robust IV formula written here,
# V = (Z'X)^-1 (sum e_i^2 z_i z_i') (X'Z)^-1
r_z = np.random.default_rng(4)
nz = 800
zz = r_z.standard_normal(nz)
xz = zz + r_z.standard_normal(nz)
Zz = np.column_stack([np.ones(nz), zz])
Xz = np.column_stack([np.ones(nz), xz])
ez = r_z.standard_normal(nz) * (1 + 0.5 * np.abs(zz))
ez = ez - Zz @ np.linalg.lstsq(Zz, ez, rcond=None)[0]
yz = 3.0 + ez
g_z = est.gmm(_lin_moments, [1.0, 1.0], args=(yz, Xz, Zz), show=False)
bz = np.linalg.solve(Zz.T @ Xz, Zz.T @ yz)
uz = yz - Xz @ bz
Az = np.linalg.inv(Zz.T @ Xz)
Vz = Az @ (Zz.T * uz ** 2) @ Zz @ Az.T
se_z = np.sqrt(np.diag(Vz))
check("v2.8.0 gmm (r3 S2-2): slope estimated at 0 up to rounding (|b| < 1e-12), started at (1, 1): estimates and "
      "SEs equal the IV closed form (abs 1e-10, rel 1e-7)", abs(bz[1]) < 1e-12
      and np.allclose(g_z["params"], bz, rtol=0, atol=1e-10)
      and np.allclose(g_z["table"]["se"], se_z, rtol=1e-7, atol=0), (bz, g_z["params"].tolist(), g_z["table"]["se"].tolist(), se_z))

# S2-1: a NaN at a trial step of the search is recoverable (the trust region shrinks); it raised ValueError from a
# start of 2e-4 (the search tried theta = 0, where 1/theta is undefined)
g_s1 = _gmm_or_failed(mom_r, [1e-4], args=(x_r,), show=False)
check("v2.8.0 gmm (r2 S2-1): start 2e-4 with x - 1/theta converges to the same estimate as a start of 1e-4 "
      "(rel <= 1e-8), no error", g_rn["converged"] and abs(g_rn["params"].iloc[0] / g_s1["params"].iloc[0] - 1) < 1e-8,
      (g_rn["params"].iloc[0], g_s1["params"].iloc[0]))

# review r3, S2-5: finite differences that are exactly 0 at both steps are no reference. GMM for a median (an
# indicator moment) with a kernel-density Jacobian, the main reason to pass one, was refused as a "clear"
# disagreement; a subgradient at a kink at theta0 too. Now a warning, and the rank test (which differentiates the
# contributions numerically: 0 for step functions) is not computed, with a notice; a true zero column with noise
# of 1e-20 in the analytic one passes silently.
r_md = np.random.default_rng(0)
y_md = r_md.standard_normal(400)
z_md = np.column_stack([np.ones(400), r_md.standard_normal(400)])
mom_md = lambda t: ((y_md <= t[0]) - 0.5)[:, None] * z_md
jac_md = lambda t: (stats.gaussian_kde(y_md)(t[0])[0] * z_md.mean(0))[:, None]
md_res = []
for th0_ in (0.1, 0.0):
    try:
        g_md = est.gmm(mom_md, [th0_], jacobian=jac_md, steps="one-step", show=False)
        frac = float(np.mean(y_md <= g_md["params"].iloc[0]))
        md_res.append((0.45 <= frac <= 0.55,
                       any("finite differences of the mean moment are 0" in w["message"] for w in g_md["warnings"]),
                       np.isnan(g_md["rank_p"]) and any("rank test of the Jacobian was not computed" in w["message"]
                                                         for w in g_md["warnings"]),
                       not any("may not be identified" in w["message"] for w in g_md["warnings"])))
    except Exception as e_:
        md_res.append(repr(e_)[:120])
check("v2.8.0 gmm (r3 S2-5): GMM for a median with a kernel-density jacobian, started at 0.1 and at 0: accepted, "
      "estimate between the 45% and 55% sample quantiles, a 'not checked' warning, the rank test not computed (with "
      "its notice, no 'may not be identified')", all(r_ == (True, True, True, True) for r_ in md_res), md_res)
r_kk = np.random.default_rng(1)
e_kk = r_kk.standard_normal(300)
mom_kk = lambda t: np.column_stack([np.abs(t[0] - 0.3) + 0.0 * e_kk, e_kk + 1.0 - t[0]])
jac_kk = lambda t: np.array([[1.0], [-1.0]])          # a subgradient of |theta - 0.3| at 0.3
try:
    g_kk = est.gmm(mom_kk, [0.3], jacobian=jac_kk, steps="one-step", show=False)
    kk_ok = any("not checked in column(s) [0]" in w["message"] for w in g_kk["warnings"])
except Exception as e_:
    kk_ok, g_kk = False, repr(e_)[:120]
check("v2.8.0 gmm (r3 S2-5): a subgradient at a kink at theta0 (|theta - 0.3| started at 0.3): not refused, a 'not "
      "checked' warning", kk_ok, g_kk if not kk_ok else None)
r_zc = np.random.default_rng(2)
zz_ = r_zc.standard_normal((300, 3)); xz_ = zz_.sum(1) + r_zc.standard_normal(300); yz_ = 2 * xz_ + r_zc.standard_normal(300)
mom_zc = lambda t: (yz_ - t[0] * xz_)[:, None] * zz_
jac_zc = lambda t: np.column_stack([-(zz_ * xz_[:, None]).mean(0), np.full(3, 1e-20)])
g_zc = est.gmm(mom_zc, [1.0, 0.5], jacobian=jac_zc, show=False)
check("v2.8.0 gmm (r3 S3): a parameter outside the moments, with noise of 1e-20 in its analytic column: no jacobian "
      "warning (the column moves the moments by less than rounding), and the 'may not be identified' warning",
      not any("jacobian" in w["message"] for w in g_zc["warnings"])
      and any("may not be identified" in w["message"] for w in g_zc["warnings"]),
      [w["message"][:70] for w in g_zc["warnings"]])

# the other side: a wrong entry where the true Jacobian is a structural 0 (each moment depends on one parameter;
# the user's has 0.5 off the diagonal) is still refused: that 0 is flat over a large step on both sides
r_sz = np.random.default_rng(3)
zsz = r_sz.standard_normal((400, 2)); x1sz = zsz[:, 0] + r_sz.standard_normal(400); x2sz = zsz[:, 1] + r_sz.standard_normal(400)
y1sz, y2sz = x1sz + r_sz.standard_normal(400), x2sz + r_sz.standard_normal(400)
mom_sz = lambda t: np.column_stack([zsz[:, 0] * (y1sz - t[0] * x1sz), zsz[:, 1] * (y2sz - t[1] * x2sz)])
jac_sz = lambda t: np.array([[-(zsz[:, 0] * x1sz).mean(), 0.5], [0.5, -(zsz[:, 1] * x2sz).mean()]])
try:
    est.gmm(mom_sz, [0.5, 0.5], jacobian=jac_sz, show=False)
    sz_msg = "accepted"
except ValueError as e_:
    sz_msg = str(e_)
check("v2.8.0 gmm (r3 S2-5): a non-zero entry where the true Jacobian is a structural 0 is still refused",
      "disagrees with finite differences" in sz_msg, sz_msg[:120])

# S2-2: the check stays inside the bounds (log theta with theta0 = 1e-6 next to a lower bound of 1e-8 raised)
x_g = np.random.default_rng(0).gamma(2.0, 1.0, size=1000)
mom_g = lambda th, x: np.column_stack([np.log(x) - np.log(th[0]), x / th[0] - 2.0])
jac_g = lambda th, x: np.array([[-1 / th[0]], [-np.mean(x) / th[0] ** 2]])
g_b1 = _gmm_or_failed(mom_g, [1e-6], args=(x_g,), jacobian=jac_g, bounds=[(1e-8, None)], show=False)
g_b2 = _gmm_or_failed(mom_g, [0.5], args=(x_g,), jacobian=jac_g, bounds=[(1e-8, None)], show=False)
check("v2.8.0 gmm (r2 S2-2): analytic Jacobian, start 1e-6 next to the bound 1e-8: no error, same estimate as "
      "from 0.5 (rel <= 1e-8)", g_b1["converged"] and abs(g_b1["params"].iloc[0] / g_b2["params"].iloc[0] - 1) < 1e-8,
      (g_b1["params"].iloc[0], g_b2["params"].iloc[0]))
lo_b, hi_b = np.array([1e-6]), np.array([np.inf])
fd_b = est._fd_column(lambda t: mom_g(t, x_g).mean(0), np.array([1e-6]), 0, 1e-6 * 6e-6, lo_b, hi_b)
# a moment undefined below its bound (NaN for theta < 0.5) and smooth at it: the difference must not step outside
f_bd = lambda t: np.where(t[0] < 0.5, np.nan, np.array([1.0 - t[0] ** 2, 2.0 - 3 * t[0] ** 2]))
fd_bd = est._fd_column(f_bd, np.array([0.5]), 0, 0.5 * 6.06e-6, np.array([0.5]), np.array([np.inf]))
check("v2.8.0 gmm (r2 S2-2): at a lower bound below which the moments are undefined, the difference stays inside "
      "(finite, equal to the analytic -2 theta and -6 theta, rel 1e-8)",
      np.all(np.isfinite(fd_bd)) and np.allclose(fd_bd, [-1.0, -3.0], rtol=1e-8, atol=0), fd_bd)
check("v2.8.0 gmm: at a lower bound the finite difference is one-sided (second order) and equals the analytic "
      "derivative (rel <= 1e-6)", np.allclose(fd_b, jac_g(np.array([1e-6]), x_g)[:, 0], rtol=1e-6, atol=0), fd_b)

# a disagreement where the finite differences are themselves unstable is a warning, not a refusal (decision of
# 2026-11-26): moments read from a table with a resolution of 1e-5 near the size of the step
q_t = lambda t: np.round(t * 1e5) / 1e5
x_t = np.random.default_rng(3).normal(2, 1, 400)
mom_t = lambda th, x: np.column_stack([x - q_t(th[0]), x ** 2 - q_t(th[0]) ** 2 - 1])
g_t = est.gmm(mom_t, [2.000003], args=(x_t,), jacobian=lambda th, x: np.array([[-1.0], [-2 * th[0]]]), show=False)
check("v2.8.0 gmm: Jacobian disagreeing with unstable finite differences -> warning 'not conclusive', no refusal",
      any(w["level"] == "warning" and "not conclusive" in w["message"] for w in g_t["warnings"]),
      [w["message"] for w in g_t["warnings"]])


# ---------------------------------------------------------------------------------------------
# lr_test: normal linear regression, where LR = n log(SSR_r / SSR_u) exactly at the MLE
# ---------------------------------------------------------------------------------------------
def _ll_normal(th, y, X):
    b, ls = th[:-1], th[-1]
    e = y - X @ b
    return -0.5 * np.log(2 * np.pi) - ls - 0.5 * e ** 2 * np.exp(-2 * ls)


rng_l = np.random.default_rng(82)
nl = 300
Xl = np.column_stack([np.ones(nl), rng_l.standard_normal((nl, 3))])
yl = Xl @ np.array([1.0, 0.5, 0.15, 0.0]) + rng_l.standard_normal(nl)
fu = est.mle(_ll_normal, np.r_[np.zeros(4), 0.0], args=(yl, Xl))
fr = est.mle(_ll_normal, np.r_[np.zeros(2), 0.0], args=(yl, Xl[:, :2]))
ssr = lambda X: float(np.sum((yl - X @ np.linalg.lstsq(X, yl, rcond=None)[0]) ** 2))
lr_cf = nl * np.log(ssr(Xl[:, :2]) / ssr(Xl))
lt = est.lr_test(fu, fr, show=False)
check("v2.8.0 lr_test: equals n log(SSR_r/SSR_u) of the normal regression (rel <= 1e-6), df = 2",
      abs(lt["statistic"] / lr_cf - 1) < 1e-6 and lt["df"] == 2, (lt["statistic"], lr_cf))
check("v2.8.0 lr_test: p is the right tail chi2.sf and chi2_cdf = 1 - p",
      abs(lt["p"] - stats.chi2.sf(lt["statistic"], 2)) < 1e-12 and abs(lt["p"] + lt["chi2_cdf"] - 1) < 1e-12
      and lt["reject"] == (lt["p"] < 0.05))
fr_short = est.mle(_ll_normal, np.r_[np.zeros(2), 0.0], args=(yl[:-1], Xl[:-1, :2]))
try:
    est.lr_test(fu, fr_short, show=False)
    same_n = False
except ValueError:
    same_n = True
check("v2.8.0 lr_test: fits on different samples are refused", same_n)
lb1 = est.lr_test(fu, fr, df=1, boundary=True, show=False)
check("v2.8.0 lr_test: boundary with one restriction -> p = half the chi2(1) p (Self-Liang), with a notice",
      abs(lb1["p"] - 0.5 * stats.chi2.sf(lb1["statistic"], 1)) < 1e-12
      and any(w["level"] == "notice" for w in lb1["warnings"]))
lb2 = est.lr_test(fu, fr, boundary=True, show=False)
check("v2.8.0 lr_test: boundary with two restrictions -> chi2(2) p and the 'conservative' warning",
      abs(lb2["p"] - lt["p"]) < 1e-15 and any("conservative" in w["message"] for w in lb2["warnings"]))
lneg = est.lr_test(fr, fu, df=2, show=False)
check("v2.8.0 lr_test: arguments swapped (LR < 0) -> warning and p = 1",
      lneg["statistic"] < 0 and lneg["p"] == 1.0
      and any("higher log-likelihood" in w["message"] for w in lneg["warnings"]))
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    est.lr_test(fu, fr)
check("v2.8.0 lr_test show: prints LR, df, 'p (right tail)' and the CDF labelled as not the p-value",
      "LR =" in buf.getvalue() and "p (right tail)" in buf.getvalue() and "not the p-value" in buf.getvalue())

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
