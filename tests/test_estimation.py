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

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
