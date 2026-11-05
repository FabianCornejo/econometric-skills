"""Tests for v2.4.7: multivariate time series, volatility, systems of equations, dynamic panels, Fama-MacBeth.

Each estimator is checked against an independent computation (statsmodels, arch or linearmodels, or a formula
written here), and the procedures that make claims about size, coverage or bias by small simulations.

Run from the repository root:  python3 tests/test_v247.py
"""
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "econometrics"
sys.path.insert(0, str(SKILL / "scripts"))
import _console  # noqa: E402,F401
import finance_empirics as fe  # noqa: E402
import multivariate as mv  # noqa: E402
import systems as sy  # noqa: E402
import volatility as vo  # noqa: E402

warnings.filterwarnings("ignore")
ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


def var_data(T=200, A=((0.5, 0.1), (0.2, 0.4)), L=((1, 0), (0.5, 0.8)), seed=0):
    rng = np.random.default_rng(seed)
    A, L = np.array(A), np.array(L)
    Y = np.zeros((T + 100, len(A)))
    for t in range(1, T + 100):
        Y[t] = A @ Y[t - 1] + L @ rng.normal(size=len(A))
    return pd.DataFrame(Y[100:], columns=[f"y{i + 1}" for i in range(len(A))])


# ---------------------------------------------------------------- 1. VAR analysis
from statsmodels.tsa.api import VAR  # noqa: E402
d = var_data()
va = mv.var_analysis(d, show=False)
res = va["results"]
sel = VAR(d).select_order(8)
check("var_analysis: lag order and criteria = statsmodels select_order", va["selected"] == {k: int(v) for k, v in sel.selected_orders.items()}
      and va["lag_order"] == sel.selected_orders["bic"])
check("var_analysis: Granger p-values = statsmodels test_causality",
      np.isclose(va["granger_pvalues"].loc["y2", "y1"], res.test_causality("y2", "y1", kind="f").pvalue)
      and np.isclose(va["granger_pvalues"].loc["y1", "y2"], res.test_causality("y1", "y2", kind="f").pvalue))
check("var_analysis: largest companion root = 1 / smallest |root| of the reverse characteristic polynomial",
      np.isclose(va["max_root"], 1 / np.min(np.abs(res.roots))))
check("var_analysis: true DGP (y1 → y2 only) gives Granger p < 0.05 for y1 → y2 and > 0.05 for y2 → y1",
      va["granger_pvalues"].loc["y2", "y1"] < 0.05 and va["granger_pvalues"].loc["y1", "y2"] > 0.05)
rw = pd.DataFrame(np.cumsum(np.random.default_rng(1).normal(size=(200, 2)), axis=0), columns=["a", "b"])
check("var_analysis warns on unit roots in levels", any("unit root" in w for w in mv.var_analysis(rw, show=False)["warnings"]))

# ---------------------------------------------------------------- 2. IRF bands and FEVD
ib = mv.irf_bands(res, h=8, B=99, show=False)
check("irf_bands: point IRF = statsmodels orth_irfs", np.allclose(ib["irf"], res.irf(8).orth_irfs))
ib_r = mv.irf_bands(res, h=8, B=49, ordering=["y2", "y1"], show=False)
res_r = VAR(d[["y2", "y1"]]).fit(res.k_ar)
check("irf_bands: another ordering = statsmodels on the reordered data", np.allclose(ib_r["irf"], res_r.irf(8).orth_irfs))
# coverage of the true IRF, persistent VAR, short sample
Atrue = np.array([[0.9, 0.0], [0.3, 0.6]])
Ltrue = np.eye(2)
true_irf = np.zeros((5, 2, 2))
P = np.eye(2)
for s in range(5):
    true_irf[s] = P
    P = Atrue @ P
cov_bc, cov_pl = [], []
for s in range(30):
    dd = var_data(T=80, A=Atrue, L=Ltrue, seed=100 + s)
    r_ = VAR(dd).fit(1)
    a = mv.irf_bands(r_, h=4, B=199, seed=s, show=False)
    b = mv.irf_bands(r_, h=4, B=199, seed=s, bias_correct=False, show=False)
    cov_bc.append(a["lower"][4, 0, 0] <= true_irf[4, 0, 0] <= a["upper"][4, 0, 0])
    cov_pl.append(b["lower"][4, 0, 0] <= true_irf[4, 0, 0] <= b["upper"][4, 0, 0])
check("irf_bands: 90% Kilian bands cover the true h = 4 own response of a persistent VAR (φ = 0.9, T = 80) "
      "at least as often as plain bands, and ≥ 80%", np.mean(cov_bc) >= np.mean(cov_pl) and np.mean(cov_bc) >= 0.8,
      f"Kilian {np.mean(cov_bc):.2f}, plain {np.mean(cov_pl):.2f}")
fv = mv.fevd_table(res, h=8, show=False)
decomp = res.fevd(8).decomp
check("fevd_table: shares = statsmodels fevd (ordering y1 > y2), rows sum to 1",
      np.allclose(fv["tables"]["y1 > y2"].xs(8, level="horizon").to_numpy(), decomp[:, 7, :])
      and np.allclose(fv["tables"]["y2 > y1"].sum(axis=1), 1))
d_ind = var_data(L=((1, 0), (0, 1)), seed=3)
fv_ind = mv.fevd_table(VAR(d_ind).fit(1), h=8, show=False)
check("fevd_table: ordering-sensitive shares flagged with correlated errors, none with independent errors",
      fv["sensitivity"]["ordering_sensitive"].any() and not fv_ind["sensitivity"]["ordering_sensitive"].any())

# ---------------------------------------------------------------- 3. SVAR identification
r1 = mv.svar_identify(3, show=False)
A = np.array([[1, 0, 0], [np.nan, 1, 0], [np.nan, np.nan, 1]])
r2 = mv.svar_identify(3, A, np.diag([np.nan] * 3), show=False)
Bbad = np.where(np.array([[1, 1, 0], [1, 1, 0], [1, 1, 0]]) == 1, np.nan, 0.0)
r3 = mv.svar_identify(3, None, Bbad, show=False)
A100 = np.where(np.isnan(A), np.nan, A * 100)
check("svar_identify: the rank verdict does not depend on units (diagonal of A fixed at 1 or at 100)",
      mv.svar_identify(3, A100, np.diag([np.nan] * 3), show=False)["rank_condition"])
check("svar_identify: full B not identified; recursive A with diagonal B exactly identified with full rank; "
      "6 free entries in the wrong places pass the count but fail the rank",
      r1["order_condition"] == "NOT identified" and r2["order_condition"] == "exactly identified" and r2["rank_condition"]
      and r3["order_condition"] == "exactly identified" and not r3["rank_condition"])
lr = mv.svar_longrun(res, bands=False, show=False)
Sig = res.sigma_u.to_numpy() if hasattr(res.sigma_u, "to_numpy") else np.asarray(res.sigma_u)
check("svar_longrun: B0 B0' = Σ_u and the long-run matrix is lower triangular",
      np.allclose(lr["impact"].to_numpy() @ lr["impact"].to_numpy().T, Sig) and abs(lr["long_run"].iloc[0, 1]) < 1e-10)
sg = mv.svar_sign(res, {"s": {"y1": 1, "y2": 1}}, h=8, draws=1000, show=False)
mt = sg["median_target"]
check("svar_sign: accepted draws satisfy the signs; median-target impact column has the imposed signs",
      sg["accepted"] > 0 and mt[0, 0, 0] > 0 and mt[0, 1, 0] > 0 and np.all(sg["lower"][0, :, 0] >= 0))
Pm = np.linalg.cholesky(Sig)
mt_imp = mt[0]
check("svar_sign: the median-target impact is a rotation of the Cholesky factor (its column has the norm of a "
      "Σ-orthonormal shock)", np.isclose(np.linalg.norm(np.linalg.solve(Pm, mt_imp[:, 0])), 1.0))

# ---------------------------------------------------------------- 4. Johansen and VECM
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen  # noqa: E402
rng = np.random.default_rng(3)
x = np.cumsum(rng.normal(size=300))
yv = 2 * x + rng.normal(size=300)
dc = pd.DataFrame({"y": yv, "x": x})
for case, det in (("n", -1), ("co", 0)):
    lam, tr = mv._johansen(dc.to_numpy(), 1, case)
    cj = coint_johansen(dc, det, 1)
    check(f"Johansen case '{case}': eigenvalues and trace = statsmodels coint_johansen",
          np.allclose(tr, cj.lr1) and np.allclose(lam, np.real(cj.eig)))
from statsmodels.tsa.vector_ar import vecm as _V  # noqa: E402
gdr = np.random.default_rng(1)
xd = np.cumsum(0.3 + gdr.normal(size=300))
dd2 = pd.DataFrame({"y": 2 * xd + 0.01 * np.arange(300) + gdr.normal(size=300), "x": xd})
same = []
for case, smc in (("ci", "ci"), ("li", "coli")):
    lam, _ = mv._johansen(dd2.to_numpy(), 1, case)
    _, dyy, yl1, dxl = _V._endog_matrices(dd2.to_numpy().T, None, None, 1, smc)
    same.append(np.allclose(lam, np.sort(_V._sij(dxl, dyy, yl1)[5])[::-1][:2]))
check("Johansen cases 'ci' and 'li' (restricted trend, unrestricted constant): eigenvalues = statsmodels 'ci' and 'coli'",
      all(same))
vli = mv.vecm_fit(dd2, deterministic="li", rank=1, k_ar_diff=1, show=False)
check("vecm_fit 'li' fits the model it tested (statsmodels 'coli': a constant outside the relation)",
      vli["results"].det_coef is not None and np.size(vli["results"].det_coef) > 0)
rej_co = []
for s in range(100):
    g = np.random.default_rng(3000 + s)
    xx = np.cumsum(0.5 + g.normal(size=250))
    rej_co.append(mv.johansen_trace(pd.DataFrame({"y": 2 * xx + g.normal(size=250), "x": xx}), k_ar_diff=0,
                                    deterministic="co", show=False)["table"]["reject"].iloc[1])
check("johansen_trace case 'co' (simulated with the data's drift) on drifting cointegrated data: size of rank ≤ 1 at most 15% (measured 11%; the asymptotic tables gave 13%)",
      np.mean(rej_co) <= 0.15, f"{np.mean(rej_co):.2f}")
rej_r1, rej_r0 = [], []
for s in range(100):
    g = np.random.default_rng(2000 + s)
    xx = np.cumsum(g.normal(size=250))
    rej_r1.append(mv.johansen_trace(pd.DataFrame({"y": 2 * xx + g.normal(size=250), "x": xx}), k_ar_diff=0,
                                    reps=1000, show=False)["table"]["reject"].iloc[1])
    rej_r0.append(mv.johansen_trace(pd.DataFrame(np.cumsum(g.normal(size=(250, 2)), axis=0)), k_ar_diff=0,
                                    reps=1000, show=False)["table"]["reject"].iloc[0])
check("johansen_trace (case 'ci', simulated critical values): size of rank ≤ 1 and of rank 0 near 5% (100 samples)",
      np.mean(rej_r1) <= 0.10 and np.mean(rej_r0) <= 0.10, f"{np.mean(rej_r1):.2f}, {np.mean(rej_r0):.2f}")
vf = mv.vecm_fit(dc, show=False)
ref = VECM(dc, k_ar_diff=vf["k_ar_diff"], coint_rank=1, deterministic="ci").fit()
check("vecm_fit: rank 1, β = statsmodels VECM β, cointegrating vector near (1, −2)",
      vf["rank"] == 1 and np.allclose(vf["beta"].to_numpy(), ref.beta) and abs(vf["beta"].iloc[1, 0] + 2) < 0.05)
adj = vf["adjustment"].set_index("variable")["reading"]
wx = vf["weak_exogeneity"].set_index("variable")["p"]
check("vecm_fit: y adjusts toward equilibrium, x is weakly exogenous (joint Wald); the levels VAR has 1 unit root and the rest inside",
      adj["y"] == "adjusts toward equilibrium" and adj["x"].startswith("does not adjust") and wx["y"] < 0.05 and wx["x"] > 0.05
      and vf["error_correcting"] and vf["unit_roots"] == 1)
# r = 2, three variables: y and z both correct toward x; readings must not depend on the column order
g3 = np.random.default_rng(11)
x3 = np.cumsum(g3.normal(size=400))
y3, z3 = np.zeros(400), np.zeros(400)
for t in range(1, 400):
    y3[t] = y3[t - 1] - 0.4 * (y3[t - 1] - x3[t - 1]) + g3.normal()
    z3[t] = z3[t - 1] - 0.3 * (z3[t - 1] + x3[t - 1]) + g3.normal()
d3 = pd.DataFrame({"y": y3, "z": z3, "x": x3})
wa = mv.vecm_fit(d3, rank=2, k_ar_diff=0, show=False)["weak_exogeneity"].set_index("variable")
wb = mv.vecm_fit(d3[["x", "y", "z"]], rank=2, k_ar_diff=0, show=False)["weak_exogeneity"].set_index("variable")
check("vecm_fit with r = 2: weak exogeneity by joint Wald is the same whatever the column order (x weakly exogenous, y and z adjust)",
      np.allclose(wa.loc[["x", "y", "z"], "p"], wb.loc[["x", "y", "z"], "p"]) and wa.loc["x", "p"] > 0.05
      and wa.loc["y", "p"] < 0.05 and wa.loc["z", "p"] < 0.05)
# a stable VECM whose eig(I + beta'alpha) is outside the unit circle once there are lagged differences
g4 = np.random.default_rng(5)
n4 = 1500
Y4 = np.zeros((n4, 2))
al, Gm = np.array([-0.62, -1.65]), np.array([[-0.22, 0.78], [-0.55, 0.25]])
dprev = np.zeros(2)
for t in range(1, n4):
    dY_ = al * (Y4[t - 1, 0] - Y4[t - 1, 1]) + Gm @ dprev + g4.normal(size=2) * 0.5
    Y4[t] = Y4[t - 1] + dY_
    dprev = dY_
v4 = mv.vecm_fit(pd.DataFrame(Y4, columns=["a", "b"]), rank=1, k_ar_diff=1, deterministic="n", show=False)
b4 = v4["beta"].to_numpy()[:, 0]
a4 = v4["alpha"].to_numpy()[:, 0]
check("vecm_fit: the stability check uses the levels VAR (exactly k − r unit roots), not eig(I + β'α), which is outside "
      "the unit circle here although the system is stable", abs(1 + b4 @ a4) > 1 and v4["error_correcting"] and v4["unit_roots"] == 1,
      f"1 + β'α = {1 + b4 @ a4:.3f}")

# ---------------------------------------------------------------- 5. volatility
from arch import arch_model  # noqa: E402
sim = arch_model(None, vol="GARCH", p=1, o=1, q=1, dist="t").simulate([0.05, 0.05, 0.02, 0.15, 0.86, 7.0], 3000,
                                                                       burn=500)
r = pd.Series(sim["data"].to_numpy(), index=pd.bdate_range("2012-01-02", periods=3000))
for gjr in (False, True):
    th, se_, llf, s2, names, _conv = vo._garch_mle(r.to_numpy(), np.empty((len(r), 0)), gjr, "t")
    ref = arch_model(r, vol="GARCH", p=1, o=int(gjr), q=1, dist="t").fit(disp="off")
    check(f"own GARCH likelihood ({'GJR' if gjr else 'GARCH'}, t) = arch: parameters, robust SEs, log-likelihood",
          np.allclose(th, ref.params.to_numpy(), atol=2e-3) and np.allclose(se_, ref.std_err.to_numpy(), rtol=0.03)
          and abs(llf - ref.loglikelihood) < 0.05)
g = vo.garch_fit(r, "garch", show=False)
j = vo.garch_fit(r, "gjr", show=False)
check("sign-bias tests: reject for a symmetric GARCH on GJR data, not for the GJR",
      g["sign_bias"]["joint_p"] < 0.05 and j["sign_bias"]["joint_p"] > 0.05,
      f"{g['sign_bias']['joint_p']:.4f} vs {j['sign_bias']['joint_p']:.4f}")
check("garch_fit: persistence = α + γ/2 + β for GJR and half-life = ln 0.5 / ln(persistence)",
      np.isclose(j["persistence"], j["params"].loc["alpha[1]", "coef"] + j["params"].loc["gamma[1]", "coef"] / 2
                 + j["params"].loc["beta[1]", "coef"]) and np.isclose(j["half_life"], np.log(0.5) / np.log(j["persistence"])))
sf_g = vo.stylized_facts(r, show=False)
sf_n = vo.stylized_facts(pd.Series(np.random.default_rng(5).normal(size=3000)), show=False)
from scipy import stats  # noqa: E402
check("stylized_facts: clustering and fat tails in GARCH data, neither in iid normal data; kurtosis = scipy",
      sf_g["clustering"] and not sf_n["clustering"]
      and np.isclose(sf_g["facts"].set_index("fact").loc["excess kurtosis (normal = 0)", "value"], stats.kurtosis(r)))
est = []
for s in range(6):
    gg = np.random.default_rng(40 + s)
    n = 2500
    idx = pd.bdate_range("2010-01-04", periods=n)
    mon = (idx.dayofweek == 0).astype(float)
    e = gg.standard_t(8, size=n) / np.sqrt(8 / 6)
    xr = np.empty(n)
    pe2, ps2 = 1.0, 1.0
    for t in range(n):
        s2_ = 0.1 + 1.5 * mon[t] + 0.05 * pe2 + 0.85 * ps2
        xr[t] = np.sqrt(s2_) * e[t]
        pe2, ps2 = xr[t] ** 2, s2_
    fit = vo.garch_fit(pd.Series(xr, idx), variance_dummies=pd.DataFrame({"monday": mon}, index=idx), show=False)
    est.append(fit["variance_dummies"].loc["delta[monday]"].to_numpy())
est = np.array(est)
check("garch_fit with a Monday dummy in the variance: δ recovered (mean of 6 within 0.5 of 1.5) and significant in every sample",
      abs(est[:, 0].mean() - 1.5) < 0.5 and np.all(est[:, 3] < 0.05), f"mean δ {est[:, 0].mean():.2f}")

# ---------------------------------------------------------------- 6. SUR and 3SLS
rng = np.random.default_rng(0)
n = 400
x1, x2, x3 = rng.normal(size=(3, n))
e = rng.multivariate_normal([0, 0], [[1, .7], [.7, 1]], size=n)
ds = pd.DataFrame({"y1": 1 + 2 * x1 + e[:, 0], "y2": -1 + 0.5 * x2 + e[:, 1], "x1": x1, "x2": x2, "x3": x3})
su = sy.sur_system({"a": "y1 ~ 1 + x1", "b": "y2 ~ 1 + x2"}, ds, show=False)
u1 = sm.OLS(ds["y1"], sm.add_constant(ds["x1"])).fit().resid
u2 = sm.OLS(ds["y2"], sm.add_constant(ds["x2"])).fit().resid
check("sur_system: Breusch-Pagan LM = T·r² from OLS residuals; SUR SEs of the slopes below OLS",
      np.isclose(su["bp_lm"], n * np.corrcoef(u1, u2)[0, 1] ** 2)
      and (su["comparison"].query("variable != 'Intercept'").eval("sur_se < ols_se")).all())
su2 = sy.sur_system({"a": "y1 ~ 1 + x1 + x2", "b": "y2 ~ 1 + x1 + x2"}, ds, show=False)
check("sur_system: identical regressors → SUR = OLS, and it says so",
      np.allclose(su2["comparison"]["sur"], su2["comparison"]["ols"]) and su2["same_regressors"]
      and "equals OLS" in su2["reading"])
inc, w, rain = rng.normal(size=(3, n))
u, v = rng.multivariate_normal([0, 0], [[1, .5], [.5, 1]], size=n).T
p = (8 + inc - w - rain + u - v) / 2
q = 2 + p + w + rain + v
dq = pd.DataFrame({"q": q, "p": p, "inc": inc, "w": w, "rain": rain})
t_ex = sy.three_sls({"demand": {"y": "q", "exog": ["inc"], "endog": ["p"]},
                     "supply": {"y": "q", "exog": ["w"], "endog": ["p"]}}, dq.drop(columns="rain"), show=False)
check("three_sls: exactly identified system → 3SLS = 2SLS, and it says so",
      np.allclose(t_ex["comparison"]["2sls"], t_ex["comparison"]["3sls"]) and "equals 2SLS" in t_ex["reading"])
t_ov = sy.three_sls({"demand": {"y": "q", "exog": ["inc"], "endog": ["p"]},
                     "supply": {"y": "q", "exog": ["w", "rain"], "endog": ["p"]}}, dq, show=False)
oc = t_ov["order_condition"].set_index("equation")["order condition"]
check("three_sls: the demand equation (2 excluded instruments, 1 endogenous) is over-identified and the slope ≈ −1 is recovered",
      oc["demand"] == "over-identified" and abs(t_ov["comparison"].query("equation == 'demand' and variable == 'p'")["3sls"].iloc[0] + 1) < 0.2)
try:
    sy.three_sls({"demand": {"y": "q", "exog": ["inc"], "endog": ["p"]},
                  "other": {"y": "rain", "exog": ["p", "w"], "endog": []}}, dq, show=False)
    clash = False
except ValueError as e_:
    clash = "endogenous" in str(e_)
check("three_sls refuses a variable that is endogenous in one equation and listed as exogenous in another", clash)
try:
    sy.three_sls({"demand": {"y": "q", "exog": ["inc", "w"], "endog": ["p"]},
                  "supply": {"y": "q", "exog": ["w", "inc"], "endog": ["p"]}}, dq, show=False)
    refused = False
except ValueError:
    refused = True
check("three_sls refuses an equation that fails the order condition", refused)


# ---------------------------------------------------------------- 7. dynamic panel GMM
def panel(N=500, T=7, rho=0.5, seed=0, ma=0.0, burn=50):
    g = np.random.default_rng(seed)
    rows = []
    for i in range(N):
        eta = g.normal()
        yy = eta / (1 - rho)
        prev = 0.0
        for t in range(-burn, T):
            xx = g.normal() + 0.5 * eta
            vv = g.normal()
            eps = vv + ma * prev
            prev = vv
            yy = rho * yy + xx + eta + eps
            if t >= 0:
                rows.append((i, t, yy, xx))
    return pd.DataFrame(rows, columns=["i", "t", "y", "x"])


from linearmodels.iv import IV2SLS, IVGMM  # noqa: E402
dp = panel()
one = sy.dynamic_panel(dp, "y", ["x"], "i", "t", collapse=True, onestep_weight="identity", show=False)
s_ = dp.sort_values(["i", "t"]).copy()
gr = s_.groupby("i")
s_["dy"], s_["dx"] = gr["y"].diff(), gr["x"].diff()
s_["ldy"] = s_.groupby("i")["dy"].shift(1)
for L in range(2, 7):
    s_[f"z{L}"] = gr["y"].shift(L).fillna(0)
s_ = s_.dropna(subset=["dy", "ldy", "dx"])
Z = s_[[f"z{L}" for L in range(2, 7)]]
Z = Z.loc[:, (Z != 0).any()]
iv = IV2SLS(s_["dy"], s_[["dx"]], s_[["ldy"]], Z).fit()
gm = IVGMM(s_["dy"], s_[["dx"]], s_[["ldy"]], Z, weight_type="clustered", clusters=s_["i"]).fit(
    iter_limit=2, cov_type="clustered", clusters=s_["i"])
check("dynamic_panel (difference GMM, collapsed, identity first step) = linearmodels IV2SLS on the stacked differences",
      np.allclose(one["one_step"]["coef"].to_numpy(), iv.params[["ldy", "dx"]].to_numpy()))
check("dynamic_panel two-step (clustered weight) = linearmodels IVGMM two-step",
      np.allclose(one["params"]["coef"].to_numpy(), gm.params[["ldy", "dx"]].to_numpy()))
est_d, est_s, ar2_rej = [], [], []
for s in range(8):
    dd = panel(rho=0.9, seed=10 + s)
    a = sy.dynamic_panel(dd, "y", ["x"], "i", "t", collapse=True, show=False)
    b = sy.dynamic_panel(dd, "y", ["x"], "i", "t", method="system", collapse=True, show=False)
    est_d.append(a["params"]["coef"].iloc[0]), est_s.append(b["params"]["coef"].iloc[0]), ar2_rej.append(a["ar2"]["p"] < 0.05)
check("dynamic_panel with ρ = 0.9: system GMM centred on 0.9 and less dispersed than difference GMM",
      abs(np.mean(est_s) - 0.9) < 0.03 and np.std(est_s) < np.std(est_d), f"system {np.mean(est_s):.3f} (SD {np.std(est_s):.3f}), "
      f"difference {np.mean(est_d):.3f} (SD {np.std(est_d):.3f})")
est5 = [sy.dynamic_panel(panel(seed=30 + s), "y", ["x"], "i", "t", collapse=True, show=False)["params"]["coef"].iloc[0]
        for s in range(8)]
check("dynamic_panel with ρ = 0.5: difference GMM centred on 0.5 (mean of 8 within 0.03)", abs(np.mean(est5) - 0.5) < 0.03,
      f"{np.mean(est5):.3f}")
ma_rej = [sy.dynamic_panel(panel(seed=50 + s, ma=0.6), "y", ["x"], "i", "t", collapse=True, show=False)["ar2"]["p"] < 0.05
          for s in range(6)]
check("AR(2) test: rarely rejects with valid instruments, rejects with MA(1) errors (instruments from lag 2 invalid)",
      np.mean(ar2_rej) <= 0.25 and np.mean(ma_rej) >= 0.8, f"{np.mean(ar2_rej):.2f} vs {np.mean(ma_rej):.2f}")
many = sy.dynamic_panel(panel(N=40, T=15, seed=7), "y", ["x"], "i", "t", show=False)
ma_s = [sy.dynamic_panel(panel(seed=70 + s, ma=0.6), "y", ["x"], "i", "t", method="system", collapse=True,
                         gmm_lags=(3, None), show=False)["params"]["coef"].iloc[0] for s in range(6)]
check("system GMM with MA(1) errors and gmm_lags=(3, ...): the level instrument moves to Δy_t−2, so ρ = 0.5 is recovered",
      abs(np.mean(ma_s) - 0.5) < 0.06, f"{np.mean(ma_s):.3f}")
check("dynamic_panel warns when instruments exceed groups (N = 40, T = 15, uncollapsed)",
      many["instruments"] > many["groups"] and any("instrument proliferation" in w for w in many["warnings"]))
check("dynamic_panel: GMM estimate between within and pooled OLS (Bond bounds) in a valid design",
      one["bounds"]["within"] < one["params"]["coef"].iloc[0] < one["bounds"]["pooled_ols"])

# ---------------------------------------------------------------- 8. Fama-MacBeth
rng = np.random.default_rng(0)
Tm, N = 240, 100
rows = []
for t in range(Tm):
    size = rng.normal(size=N)
    ret = 0.01 - 0.003 * size + rng.normal(0, 0.05, N)
    rows += [(t, ret[i], size[i]) for i in range(N)]
Pn = pd.DataFrame(rows, columns=["t", "ret", "size"])
fm = fe.fama_macbeth(Pn, "ret", ["size"], "t", show=False)
lam = np.array([np.linalg.lstsq(np.column_stack([np.ones(N), g["size"]]), g["ret"], rcond=None)[0] for _, g in Pn.groupby("t")])
check("fama_macbeth: premia = mean of per-period OLS slopes; FM SE = SD/√T (computed here)",
      np.allclose(fm["table"]["premium"], lam.mean(axis=0)) and np.allclose(fm["table"]["se_fm"], lam.std(axis=0, ddof=1) / np.sqrt(Tm)))
check("fama_macbeth: the size premium (−0.003) is recovered within 2 SE", abs(fm["table"].loc["size", "premium"] + 0.003)
      < 2 * fm["table"].loc["size", "se_nw"])
f = rng.normal(0.006, 0.04, size=Tm)
bet = rng.uniform(0.5, 1.5, N)
Rm = pd.DataFrame(np.outer(f, bet) + rng.normal(0, 0.05, size=(Tm, N)))
Fm = pd.DataFrame({"mkt": f})
fb = fe.fama_macbeth(returns=Rm, factors=Fm, show=False)
c_ = fb["table"].loc["mkt", "premium"] ** 2 / np.var(f, ddof=1)
se_sh = np.sqrt((1 + c_) * fb["table"].loc["mkt", "se_nw"] ** 2 + np.var(f, ddof=1) / Tm)
check("fama_macbeth with estimated betas: Shanken SE = √((1 + λ²/σ_f²)·Var_NW + σ_f²/T), larger than the NW SE",
      np.isclose(fb["table"].loc["mkt", "se_shanken"], se_sh) and se_sh > fb["table"].loc["mkt", "se_nw"])
cov_sh, cov_nw = [], []
for s in range(60):
    g5 = np.random.default_rng(500 + s)
    f5 = g5.normal(0.006, 0.04, size=120)
    b5 = g5.uniform(0.5, 1.5, 60)
    R5 = pd.DataFrame(np.outer(f5, b5) + g5.normal(0, 0.08, size=(120, 60)))
    t5 = fe.fama_macbeth(returns=R5, factors=pd.DataFrame({"mkt": f5}), nw_lags=0, show=False)["table"].loc["mkt"]
    cov_sh.append(abs(t5["premium"] - 0.006) <= 1.96 * t5["se_shanken"])
    cov_nw.append(abs(t5["premium"] - 0.006) <= 1.96 * t5["se_nw"])
check("fama_macbeth: with estimated betas the Shanken interval covers the true premium more often than the uncorrected one "
      "(60 samples)", np.mean(cov_sh) >= np.mean(cov_nw) and np.mean(cov_sh) >= 0.85,
      f"Shanken {np.mean(cov_sh):.2f}, uncorrected {np.mean(cov_nw):.2f}")

# ---------------------------------------------------------------- 9. documentation and version
skill = (SKILL / "SKILL.md").read_text(encoding="utf-8")
rules = (SKILL / "references" / "interpretation-rules.md").read_text(encoding="utf-8")
teach = (SKILL / "references" / "teaching.md").read_text(encoding="utf-8")
check("SKILL.md routes and scripts table name the new modules",
      all(k in skill for k in ("multivariate.py", "volatility.py", "systems.py", "multivariate.var_analysis",
                               "systems.sur_system", "dynamic_panel", "volatility.stylized_facts", "garch_fit", "fama_macbeth")))
check("interpretation rules for v2.4.7 (dynamic panel checks, Cholesky ordering, sign restrictions, VECM, GARCH names)",
      "## Rules added in v2.4.7" in rules and all(k in rules for k in ("no AR(2), Hansen J and instrument count",
                                                                      "assumption about ordering", "identify a set",
                                                                      "error-correction check", "EGARCH = log-variance")))
check("curriculum map lists the new functions", all(k in teach for k in ("systems.dynamic_panel", "multivariate.irf_bands",
                                                                          "volatility.garch_fit", "multivariate.vecm_fit")))
import json  # noqa: E402
import reporting  # noqa: E402
mk = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))
top = re.search(r"^## v(\S+)", (ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), re.M).group(1)
check("same version (2.4.7 or later) in marketplace, reporting and CHANGELOG",
      mk["plugins"][0]["version"] == reporting.SKILL_VERSION == top and tuple(map(int, top.split("."))) >= (2, 4, 7),
      f"{mk['plugins'][0]['version']}, {reporting.SKILL_VERSION}, {top}")

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
