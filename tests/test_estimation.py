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
from scipy import stats
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
check("v2.8.0 gmm: Euler equation with n = 4998 is strongly identified (rank test per df far above 10; no "
      "identification warning)", ge["ident_F"] > 50 and ge["rank_p"] < 1e-10
      and not any("identif" in w["message"] for w in ge["warnings"]), ge["ident_F"])

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
check("v2.8.0 gmm: strong instruments (first-stage F in the hundreds): no identification warning",
      not any("identif" in w["message"] for w in strong["warnings"]) and strong["ident_F"] > 10)

# both sides of the weak-identification cut (10), seeds copied from claude/scripts/v280/seed_search.py
wk_below = est.gmm(_lin_moments, [0.0, 0.0], args=_iv_case(0.08, 0), show=False)
wk_above = est.gmm(_lin_moments, [0.0, 0.0], args=_iv_case(0.10, 10), show=False)
check("v2.8.0 gmm: rank statistic per df 7.2 (below 10, rank test rejects) -> weak-identification warning",
      7 < wk_below["ident_F"] < 10 and wk_below["rank_p"] < 0.05
      and any(w["level"] == "warning" and "weak identification" in w["message"] for w in wk_below["warnings"]),
      wk_below["ident_F"])
check("v2.8.0 gmm: rank statistic per df 10.8 (just above 10) -> no identification warning",
      10 < wk_above["ident_F"] < 13 and not any("identif" in w["message"] for w in wk_above["warnings"]),
      wk_above["ident_F"])
none_ = est.gmm(_lin_moments, [0.0, 0.0], args=_iv_case(0.0, 0), show=False)
check("v2.8.0 gmm: irrelevant instruments (rank test p = 0.38) -> 'may not be identified' warning, not 'weak'",
      none_["rank_p"] >= 0.05 and any("may not be identified" in w["message"] for w in none_["warnings"])
      and not any("weak identification" in w["message"] for w in none_["warnings"]), none_["rank_p"])
ratios = []
for pi_ in (0.05, 0.10):
    for sd in range(20):
        a_ = _iv_case(pi_, sd)
        ratios.append(est.gmm(_lin_moments, [0.0, 0.0], args=a_, show=False)["ident_F"] / _fs_F(*a_))
check("v2.8.0 gmm: where identification is weak (pi 0.05 and 0.1, 40 samples) the rank statistic per df is the "
      "first-stage F (median ratio within 5% of 1)", abs(np.median(ratios) - 1) < 0.05, np.median(ratios))

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
    est.gmm(_lin_moments, [0.0, 0.0], args=_iv_case(0.08, 0), names=["const", "x"])
txt = buf.getvalue()
check("v2.8.0 gmm show: prints the method, the table, J with 'p (right tail)', the CDF labelled as not the "
      "p-value, the identification line and each finding with its action",
      "GMM (two-step)" in txt and "estimate" in txt and "p (right tail)" in txt and "not the p-value" in txt
      and "Cragg-Donald" in txt and "WARNING: weak identification" in txt and "action:" in txt, txt[-400:])
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
