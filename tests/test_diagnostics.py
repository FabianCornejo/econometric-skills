"""Replications for six functions of diagnostics.py that had no test (VALIDATION.md level D until v2.6.11).

Each function is checked against a published number (the edition and page are fixed below) and against the same
quantity computed here by another route, written from the textbook definition and not from the implementation.

References (read in the texts on 2026-11-16; the extraction of the PDFs drops minus signs, so the signs below
come from the definitions):
  Wooldridge, J.M. (2000), Introductory Econometrics: A Modern Approach, 1st ed., South-Western.
      Example 8.5, p. 260 (special White test, log housing price equation 8.18): R² = .0392, LM ≈ 3.45, p = .178.
      Example 9.8, p. 301-302 (RDCHEM): with the 32 firms, sales .000053 (.000044), R² = .0761; without the
      largest firm (n = 31), sales .000186 (.000084), profmarg .0478 (.0445), R² = .1728; largest residual about
      6.37, residual of the largest firm 1.62, sigma-hat 1.82.
      Section 18.4, p. 587-588 (INTQRT): the basic Engle-Granger test regresses u_t on u_{t-1} (no lags);
      5% critical value -3.34 (Table 18.4). Example 18.6, p. 591: OLS of r6 on r3 gives 1.026 (.0077).
  Baltagi, B.H. (2013), Econometric Analysis of Panel Data, 5th ed., Wiley (Grunfeld data, 10 firms):
      p. 81, Hausman m1 = 2.33; p. 108, Table 4.2, Breusch-Pagan LM_mu = 798.162.
  Greene, W.H. (2012), Econometric Analysis, 7th ed., Pearson: eq. (11-42) and Example 11.6, p. 376-377
      (n = 595, T = 7, sum of squared group-mean residuals 53.824384, total sum of squared residuals 607.1265,
      LM = 3881.34); Example 11.8, p. 380 (the classical Hausman maintains the random-effects assumptions).

Data: Grunfeld from statsmodels.datasets (public domain; the 11-firm file, American Steel dropped to get Baltagi's 10
firms); HPRICE1, RDCHEM and INTQRT from the wooldridge package. Tolerance for published numbers: half the last
printed digit (v2.6.7 rule). Tolerance for the other routes: 1e-10 relative (closed-form rule).

Run from the repository root:  python3 tests/test_diagnostics.py
"""
import contextlib
import io
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "econometrics" / "scripts"))
import _console  # noqa: E402,F401  (non-ASCII text on a Windows console)

import diagnostics as dg  # noqa: E402
import state_space as ss  # noqa: E402
import statsmodels.api as sm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
import wooldridge as woo  # noqa: E402
from linearmodels.panel import PanelOLS, PooledOLS, RandomEffects  # noqa: E402

warnings.filterwarnings("ignore")
results = []


def check(name, cond, info=""):
    results.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info)


def close(a, b, tol):
    return abs(float(a) - float(b)) <= tol


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


def quiet(f, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = f(*a, **k)
    return out, buf.getvalue()


# ============================================================================== Grunfeld (Baltagi, 10 firms)
gr = sm.datasets.grunfeld.load_pandas().data
gr = gr[gr.firm != "American Steel"].copy()
gr["year"] = gr.year.astype(int)
gr = gr.set_index(["firm", "year"]).sort_index()
check("Grunfeld: 10 firms x 20 years after dropping American Steel (Baltagi's sample)",
      gr.index.get_level_values(0).nunique() == 10 and len(gr) == 200, gr.shape)
GX = sm.add_constant(gr[["value", "capital"]])
fe_u = PanelOLS(gr.invest, GX, entity_effects=True).fit(cov_type="unadjusted")
re_u = RandomEffects(gr.invest, GX).fit(cov_type="unadjusted")

# -------------------------------------------------------------------------------------------- 1. hausman
h, h_txt = quiet(dg.hausman, fe_u, re_u)
check("B2.33 Hausman FE vs RE on Grunfeld reproduces Baltagi (2013, 5th ed., p. 81) m1 = 2.33",
      close(h["statistic"], 2.33, 0.005) and h["df"] == 2, h)


def fe_re_by_hand(d, y, xs):
    """Within and Swamy-Arora random-effects estimators written from Baltagi (2013, ch. 2) for a balanced panel."""
    ids = d.index.get_level_values(0)
    N, T = ids.nunique(), int(d.groupby(level=0).size().iloc[0])
    Y, X = d[y].to_numpy(float), d[xs].to_numpy(float)
    ym = d[y].groupby(level=0).transform("mean").to_numpy()
    Xm = d[xs].groupby(level=0).transform("mean").to_numpy()
    Yw, Xw = Y - ym, X - Xm
    b_w = np.linalg.solve(Xw.T @ Xw, Xw.T @ Yw)
    e_w = Yw - Xw @ b_w
    k = X.shape[1]
    s2_e = e_w @ e_w / (N * T - N - k)
    V_fe = s2_e * np.linalg.inv(Xw.T @ Xw)
    yb = d[y].groupby(level=0).mean().to_numpy()
    Xb = np.column_stack([np.ones(N), d[xs].groupby(level=0).mean().to_numpy()])
    bb = np.linalg.lstsq(Xb, yb, rcond=None)[0]
    eb = yb - Xb @ bb
    s2_1 = T * (eb @ eb) / (N - k - 1)                 # sigma_1^2 = T sigma_u^2 + sigma_e^2
    theta = 1 - np.sqrt(s2_e / s2_1)
    Ys = Y - theta * ym
    Xs = np.column_stack([np.full(len(Y), 1 - theta), X - theta * Xm])
    b_re = np.linalg.solve(Xs.T @ Xs, Xs.T @ Ys)
    e_re = Ys - Xs @ b_re
    s2_re = e_re @ e_re / (len(Y) - k - 1)
    V_re = s2_re * np.linalg.inv(Xs.T @ Xs)
    d_ = b_w - b_re[1:]
    return float(d_ @ np.linalg.inv(V_fe - V_re[1:, 1:]) @ d_)


H_hand = fe_re_by_hand(gr, "invest", ["value", "capital"])
check("hausman: equals the statistic from FE and Swamy-Arora RE written by hand (relative 1e-10)",
      rel(h["statistic"], H_hand) < 1e-10, (h["statistic"], H_hand))
check("hausman: valid, classical covariances, 'Do not reject' and the printed line",
      h["valid"] and h["classical_cov"] is True and h["conclusion"].startswith("Do not reject H0")
      and "Hausman: χ²(2) = 2.330, p = 0.3119" in h_txt, (h, h_txt))
fe_r = PanelOLS(gr.invest, GX, entity_effects=True).fit(cov_type="robust")
re_r = RandomEffects(gr.invest, GX).fit(cov_type="robust")
hr = dg.hausman(fe_r, re_r, show=False)
check("hausman with robust covariances: classical_cov False and no verdict ('Not interpretable', names the "
      "covariances and mundlak_test; no 'random effects acceptable')",
      hr["classical_cov"] is False and hr["conclusion"].startswith("Not interpretable")
      and "Robust/Robust" in hr["conclusion"] and "mundlak_test" in hr["conclusion"]
      and "acceptable" not in hr["conclusion"], hr)
fe_c = PanelOLS(gr.invest, GX, entity_effects=True).fit(cov_type="clustered", cluster_entity=True)
hc = dg.hausman(fe_c, re_u, show=False)
check("hausman: one clustered fit is enough to withhold the verdict", hc["classical_cov"] is False
      and hc["conclusion"].startswith("Not interpretable") and "Clustered/Unadjusted" in hc["conclusion"], hc)
idx = ["const", "x1", "x2"]


def fake(b, V):
    return SimpleNamespace(params=pd.Series(b, index=idx), cov=pd.DataFrame(V, index=idx, columns=idx))


fa = fake([1.0, 0.5, 0.2], np.diag([1.0, 0.04, 0.03]))
fb = fake([1.0, 0.3, 0.1], np.diag([1.0, 0.02, 0.01]))
hn = dg.hausman(fa, fb, show=False)
H_n = 0.2 ** 2 / 0.02 + 0.1 ** 2 / 0.02
check("hausman: results without _cov_type → classical_cov None, verdict kept; closed form 0.2²/0.02 + 0.1²/0.02",
      hn["classical_cov"] is None and close(hn["statistic"], H_n, 1e-12) and hn["df"] == 2
      and not hn["conclusion"].startswith("Not interpretable"), hn)
hv = dg.hausman(fake([1.0, 0.5, 0.2], np.diag([1.0, 0.01, 0.03])), fb, show=False)
check("hausman: V_FE − V_RE not positive semidefinite → valid False and the Mundlak warning",
      hv["valid"] is False and "not positive semidefinite" in hv["conclusion"], hv)

# -------------------------------------------------------------------------------------------- 2. panel_bp_lm
pooled = PooledOLS(gr.invest, GX).fit()
bp, bp_txt = quiet(dg.panel_bp_lm, pooled.resids)
check("B798 Breusch-Pagan LM on Grunfeld pooled residuals reproduces Baltagi (2013, 5th ed., p. 108) 798.162",
      close(bp["LM"], 798.162, 0.0005), bp)
e = pooled.resids.to_numpy()
N_g, T_g = 10, 20
D = np.kron(np.eye(N_g), np.ones((T_g, T_g)))                # I_N ⊗ J_T (Baltagi's matrix form)
lm_mat = N_g * T_g / (2 * (T_g - 1)) * (e @ D @ e / (e @ e) - 1) ** 2
check("panel_bp_lm: equals the matrix form e'(I_N⊗J_T)e/e'e (relative 1e-10)", rel(bp["LM"], lm_mat) < 1e-10,
      (bp["LM"], lm_mat))
check("panel_bp_lm: printed verdict when p < 5%", "individual effects present" in bp_txt, bp_txt)
# Greene's Example 11.6 has no open data; residuals with the published sufficient statistics give the same LM.
n6, T6, sq_means, ssr6 = 595, 7, 53.824384, 607.1265
m = np.full(n6, np.sqrt(sq_means / n6))
w_ss = ssr6 - T6 * sq_means
pat = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 0.0])
dev = np.tile(pat * np.sqrt(w_ss / (n6 * (pat @ pat))), n6)
e6 = pd.Series(np.repeat(m, T6) + dev,
               index=pd.MultiIndex.from_product([range(n6), range(T6)], names=["i", "t"]))
check("G11.6 residuals built with Greene's sums (595 x 7; 53.824384; 607.1265)",
      close((e6.groupby(level=0).mean() ** 2).sum(), sq_means, 1e-9) and close((e6 ** 2).sum(), ssr6, 1e-9))
g6 = dg.panel_bp_lm(e6, show=False)
# eq. (11-42) with Greene's printed sums gives 3881.345; the printed 3881.34 is 0.005 away (1.3e-6 relative), more
# than the rounding of the printed sums explains. Tolerance 0.01, looser than half the last digit.
check("G11.6 Breusch-Pagan LM reproduces Greene (2012, 7th ed., Example 11.6, p. 377) 3881.34",
      close(g6["LM"], 3881.34, 0.01), g6)
small = pd.Series([0.3, -0.3, 0.1, -0.1, -0.2, 0.2],
                  index=pd.MultiIndex.from_product([range(3), range(2)]))
s_out, s_txt = quiet(dg.panel_bp_lm, small)
check("panel_bp_lm: no group effect in the residuals → LM = 3*2/(2*1)*(0 − 1)² = 3, verdict 'pooled OLS acceptable'",
      close(s_out["LM"], 3.0, 1e-12) and "pooled OLS acceptable" in s_txt, (s_out, s_txt))

# -------------------------------------------------------------------------------------------- 3. white_simplified
hp = woo.data("hprice1")
r818 = smf.ols("lprice ~ llotsize + lsqrft + bdrms", hp).fit()
lm, p = dg.white_simplified(r818)
check("W8.5 special White test reproduces Wooldridge (2000, 1st ed., Example 8.5, p. 260): LM ≈ 3.45, p = .178, "
      "R² = .0392", close(lm, 3.45, 0.005) and close(p, 0.178, 0.0005) and close(lm / len(hp), 0.0392, 0.00005),
      (lm, p))
Xh = np.column_stack([np.ones(len(hp)), hp.llotsize, hp.lsqrft, hp.bdrms])   # the logged columns of the file
yh = hp.lprice.to_numpy(float)
bh = np.linalg.lstsq(Xh, yh, rcond=None)[0]
fit = Xh @ bh
u2 = (yh - fit) ** 2
Z = np.column_stack([np.ones(len(hp)), fit, fit ** 2])
g = np.linalg.lstsq(Z, u2, rcond=None)[0]
r2_aux = 1 - ((u2 - Z @ g) ** 2).sum() / ((u2 - u2.mean()) ** 2).sum()
check("white_simplified: LM = n·R² of û² on (1, ŷ, ŷ²) computed by hand (relative 1e-10)",
      rel(lm, len(hp) * r2_aux) < 1e-10, (lm, len(hp) * r2_aux))
lm_r, _ = dg.white_simplified(smf.ols("lprice ~ llotsize + lsqrft + bdrms", hp).fit(cov_type="HC3"))
check("white_simplified: same LM from a fit with robust covariance (uses residuals and X only)",
      close(lm_r, lm, 1e-12), (lm_r, lm))

# -------------------------------------------------------------------------------------------- 4. vif_table
rlev = smf.ols("price ~ lotsize + sqrft + bdrms", hp).fit()
v = dg.vif_table(rlev)
Xc = hp[["lotsize", "sqrft", "bdrms"]].to_numpy(float)
vif_corr = np.diag(np.linalg.inv(np.corrcoef(Xc.T)))
check("vif_table: equals the diagonal of the inverse correlation matrix (relative 1e-10); no Intercept row",
      list(v.Variable) == ["lotsize", "sqrft", "bdrms"] and rel(v.VIF, vif_corr) < 1e-10, (v, vif_corr))
v0 = dg.vif_table(smf.ols("price ~ lotsize + sqrft + bdrms - 1", hp).fit())
check("vif_table: a model without a constant gives the same centered VIF (statsmodels 0.14 gave 1.86/17.06/16.79)",
      list(v0.Variable) == list(v.Variable) and rel(v0.VIF, vif_corr) < 1e-10, v0)
hp2 = hp.assign(k=5.0, sqrft2=2 * hp.sqrft)
vc = dg.vif_table(sm.OLS(hp2.price, hp2[["k", "lotsize", "sqrft", "sqrft2"]]).fit())
check("vif_table: a constant column under another name is skipped; perfect collinearity gives inf",
      list(vc.Variable) == ["lotsize", "sqrft", "sqrft2"] and np.isinf(vc.VIF.iloc[1]) and np.isinf(vc.VIF.iloc[2])
      and np.isfinite(vc.VIF.iloc[0]), vc)
v1 = dg.vif_table(smf.ols("price ~ sqrft", hp).fit())
check("vif_table: a single regressor has VIF 1", len(v1) == 1 and close(v1.VIF.iloc[0], 1.0, 1e-12), v1)
v_none = dg.vif_table(smf.ols("price ~ 1", hp).fit())
check("vif_table: a model with only a constant gives an empty table with the two columns",
      v_none.empty and list(v_none.columns) == ["Variable", "VIF"], v_none)

# -------------------------------------------------------------------------------------------- 5. influence
rd = woo.data("rdchem")
r98 = smf.ols("rdintens ~ sales + profmarg", rd).fit()
check("W9.8 data: the full-sample fit matches the book (sales .000053 (.000044), R² = .0761, n = 32)",
      close(r98.params["sales"], 0.000053, 5e-7) and close(r98.bse["sales"], 0.000044, 5e-7)
      and close(r98.rsquared, 0.0761, 0.00005) and r98.nobs == 32, r98.params)
inf, inf_txt = quiet(dg.influence, r98, top=3)
big = int(rd.sales.idxmax())
check("influence: the largest firm (sales ≈ $40 billion) is the most influential observation",
      inf.index[0] == big and inf.loc[big, "most_affected_coef"] == "sales", inf.head(3))
# the deleted-observation coefficient rebuilt from the function's own columns (h, DFBETAS)
Xr = r98.model.exog
n_r, k_r = Xr.shape
XtXi = np.linalg.inv(Xr.T @ Xr)
e_r = r98.resid.to_numpy()
h_b = inf.loc[big, "leverage_h"]
s_del = np.sqrt((e_r @ e_r - e_r[big] ** 2 / (1 - h_b)) / (n_r - k_r - 1))
j = list(r98.model.exog_names).index("sales")
b_del = r98.params["sales"] - inf.loc[big, "max_|DFBETA|"] * np.sign(e_r[big] * (XtXi @ Xr[big])[j]) \
    * s_del * np.sqrt(XtXi[j, j])
check("W9.8 influence: the sales coefficient without the largest firm, rebuilt from the function's DFBETA and "
      "leverage, reproduces Wooldridge (2000, 1st ed., Example 9.8, p. 301) .000186",
      close(b_del, 0.000186, 5e-7), b_del)
r31 = smf.ols("rdintens ~ sales + profmarg", rd.drop(index=big)).fit()
check("W9.8 data: without the largest firm the book's numbers (.000186 (.000084), .0478 (.0445), R² = .1728)",
      close(r31.params["sales"], 0.000186, 5e-7) and close(r31.bse["sales"], 0.000084, 5e-7)
      and close(r31.params["profmarg"], 0.0478, 0.00005) and close(r31.bse["profmarg"], 0.0445, 0.00005)
      and close(r31.rsquared, 0.1728, 0.00005), r31.params)
# p. 302 also prints sigma-hat = 1.82, which no usual definition gives with these data (sqrt(SSR/29) = 1.862,
# sqrt(SSR/32) = 1.772) while the coefficients and R² match: not used (probably a misprint).
check("W9.8 data: largest residual about 6.37, residual of the largest firm −1.62 (p. 302, read on the page image)",
      close(e_r.max(), 6.37, 0.005) and close(e_r[big], -1.62, 0.005), (e_r.max(), e_r[big]))
# every column by refitting without each observation (n regressions), and h_ii from a QR decomposition
Q, _ = np.linalg.qr(Xr)
h_qr = (Q ** 2).sum(axis=1)
y_r = r98.model.endog
dfb_ref, cook_ref = np.zeros((n_r, k_r)), np.zeros(n_r)
for i in range(n_r):
    keep = np.arange(n_r) != i
    b_i = np.linalg.lstsq(Xr[keep], y_r[keep], rcond=None)[0]
    ei = y_r[keep] - Xr[keep] @ b_i
    s_i = np.sqrt(ei @ ei / (n_r - 1 - k_r))
    dfb_ref[i] = (r98.params.to_numpy() - b_i) / (s_i * np.sqrt(np.diag(XtXi)))
    db = r98.params.to_numpy() - b_i
    cook_ref[i] = db @ Xr.T @ Xr @ db / (k_r * r98.scale)
loo = e_r / (1 - h_qr)
ref = pd.DataFrame({"h": h_qr, "infl": np.abs(h_qr * loo), "dfb": np.abs(dfb_ref).max(axis=1), "cook": cook_ref})
got = inf.sort_index()
check("influence: leverage (QR), |h·ẽ|, max |DFBETA| and Cook's distance equal n refits without each observation "
      "(relative 1e-10)", max(rel(got["leverage_h"], ref.h), rel(got["influence_|h·ẽ|"], ref.infl),
                             rel(got["max_|DFBETA|"], ref.dfb), rel(got["cook"], ref.cook)) < 1e-10,
      (rel(got["leverage_h"], ref.h), rel(got["max_|DFBETA|"], ref.dfb), rel(got["cook"], ref.cook)))
check("influence: rows sorted by |h·ẽ| (descending)", inf["influence_|h·ẽ|"].is_monotonic_decreasing)
check("influence: printed thresholds k/n = 0.094, 2k/n = 0.188, 2/√n = 0.354, the top 3 rows and the advice",
      "Mean leverage = k/n = 0.094" in inf_txt and "h > 0.188 (2k/n)" in inf_txt and "> 0.354 (2/√n)" in inf_txt
      and "report results with and without them" in inf_txt and inf_txt.count("\n") >= 6, inf_txt)

# -------------------------------------------------------------------------------------------- 6. cointegration
iq = woo.data("intqrt")
ci, ci_txt = quiet(dg.cointegration, iq[["r6", "r3"]], k_ar_diff=1)
eg = ci["engle_granger"]
u = sm.OLS(iq.r6.to_numpy(), sm.add_constant(iq.r3.to_numpy())).fit()
check("W18.6 data: OLS of r6 on r3 gives 1.026 (.0077) (Wooldridge 2000, 1st ed., p. 591)",
      close(u.params[1], 1.026, 0.0005) and close(u.bse[1], 0.0077, 0.00005), u.params)
uh = u.resid
du, ul = np.diff(uh), uh[:-1]
t_basic = (ul @ du / (ul @ ul) - 0) / np.sqrt(((du - ul * (ul @ du / (ul @ ul))) ** 2).sum()
                                                / (len(du) - 1) / (ul @ ul))
check("W18.4 cointegration: BIC picks no lags, and the statistic is the book's basic test (Δû on û₋₁, no constant) "
      "computed by hand (relative 1e-10)", eg["lags_bic"] == 0 and rel(eg["statistic_bic"], t_basic) < 1e-10,
      (eg["lags_bic"], eg["statistic_bic"], t_basic))
check("W18.4 cointegration: the basic test rejects at the book's 5% critical value −3.34 (Table 18.4, p. 588)",
      eg["statistic_bic"] < -3.34 and eg["p_bic"] < 0.05, eg)
check("cointegration: AIC picks 6 lags and does not reject (t = −3.132, p = 0.082): lag_sensitive, 'Inconclusive'",
      eg["lags_aic"] == 6 and close(eg["statistic"], -3.132, 0.0005) and eg["p"] > 0.05 and eg["lag_sensitive"]
      and eg["conclusion"].startswith("Inconclusive"), eg)
from statsmodels.tsa.stattools import coint  # noqa: E402
c_aic, c_bic = coint(iq.r6, iq.r3, trend="c", autolag="aic"), coint(iq.r6, iq.r3, trend="c", autolag="bic")
check("cointegration: statistic and p are statsmodels' coint with autolag aic and bic",
      close(eg["statistic"], c_aic[0], 1e-12) and close(eg["p"], c_aic[1], 1e-12)
      and close(eg["statistic_bic"], c_bic[0], 1e-12) and close(eg["p_bic"], c_bic[1], 1e-12), (eg, c_aic, c_bic))
check("cointegration: the printed lines give both lag choices and the inconclusive verdict",
      "lags chosen by AIC: 6 → t=-3.132, p=0.0821" in ci_txt and "lags chosen by BIC: 0 → t=-8.389" in ci_txt
      and "→ Inconclusive: the conclusion depends on the lag choice" in ci_txt, ci_txt)


def johansen_by_hand(Y, k_ar_diff):
    """Trace statistics of Johansen (1988) with an unrestricted constant: reduced-rank regression of ΔY_t and
    Y_{t-1} on (1, ΔY_{t-1}, ..., ΔY_{t-k}), then -T Σ log(1 - λ_i)."""
    dY = np.diff(Y, axis=0)
    T_ = len(dY) - k_ar_diff
    Zl = np.column_stack([np.ones(T_)] + [dY[k_ar_diff - i - 1:len(dY) - i - 1] for i in range(k_ar_diff)])
    R0 = dY[k_ar_diff:] - Zl @ np.linalg.lstsq(Zl, dY[k_ar_diff:], rcond=None)[0]
    L = Y[k_ar_diff:-1]
    R1 = L - Zl @ np.linalg.lstsq(Zl, L, rcond=None)[0]
    S00, S11, S01 = R0.T @ R0 / T_, R1.T @ R1 / T_, R0.T @ R1 / T_
    lam = np.sort(np.linalg.eigvals(np.linalg.solve(S11, S01.T @ np.linalg.solve(S00, S01))).real)[::-1]
    return np.array([-T_ * np.log(1 - lam[r:]).sum() for r in range(Y.shape[1])])


tr_hand = johansen_by_hand(iq[["r6", "r3"]].to_numpy(float), 1)
check("cointegration: Johansen trace statistics equal the reduced-rank regression written by hand (relative 1e-10)",
      rel(ci["johansen"]["Trace"], tr_hand) < 1e-10, (ci["johansen"]["Trace"].to_list(), tr_hand))
check("cointegration: full rank with det_order=0 → the note says not to conclude I(0) and names johansen_trace 'ci'",
      ci["rank_trace"] == 2 and ci["johansen_note"] == dg.FULL_RANK_DET0
      and "do not conclude I(0)" in ci_txt and "deterministic='ci'" in ci_txt
      and "probably I(0)" not in ci_txt, ci_txt)
rng = np.random.default_rng(20261117)
wn = pd.DataFrame(rng.normal(size=(300, 2)), columns=["a", "b"])
c1 = dg.cointegration(wn, det_order=1, show=False)
check("cointegration: full rank with det_order=1 keeps the old note ('probably I(0)')",
      c1["rank_trace"] == 2 and c1["johansen_note"].startswith("Full rank: the series are probably I(0)"), c1)


def ecm_pair(rng, n, a):
    """y and x I(1); y − x I(0) with adjustment a (a = 0: no cointegration)."""
    x = np.cumsum(rng.normal(size=n))
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = y[t - 1] + a * (y[t - 1] - x[t - 1]) + rng.normal()
    return pd.DataFrame({"y": y, "x": x})


# Known truth, bands fixed before the first run: cointegrated (a = −0.2, n = 200): "Cointegrated" in at least 80% of
# 200 samples; independent random walks: "Cointegrated" in at most 10%.
rng = np.random.default_rng(611)
conc = {"coint": [], "none": []}
for _ in range(200):
    for key, a in (("coint", -0.2), ("none", 0.0)):
        conc[key].append(dg.cointegration(ecm_pair(rng, 200, a), show=False)["engle_granger"]["conclusion"])
rate_c = np.mean([c == "Cointegrated" for c in conc["coint"]])
rate_n = np.mean([c == "Cointegrated" for c in conc["none"]])
inc = np.mean([c.startswith("Inconclusive") for c in conc["coint"] + conc["none"]])
print(f"   known truth: 'Cointegrated' {rate_c:.3f} (cointegrated), {rate_n:.3f} (independent walks); "
      f"inconclusive {inc:.3f} overall")
check("cointegration known truth: 'Cointegrated' in ≥ 80% of cointegrated samples and ≤ 10% of independent walks "
      "(200 each)", rate_c >= 0.80 and rate_n <= 0.10, (rate_c, rate_n))
agree = dg.cointegration(ecm_pair(np.random.default_rng(5), 400, -0.5), show=False)["engle_granger"]
check("cointegration: when AIC and BIC agree the verdict is firm ('Cointegrated', lag_sensitive False)",
      agree["conclusion"] == "Cointegrated" and agree["lag_sensitive"] is False, agree)
orig_adf = dg.adfuller
try:
    dg.adfuller = lambda *a, **k: (orig_adf(*a, **k)[0] + 1.0,) + tuple(orig_adf(*a, **k)[1:])
    odd, odd_txt = quiet(dg.cointegration, iq[["r6", "r3"]])
finally:
    dg.adfuller = orig_adf
check("cointegration: when the re-run ADF does not match coint, the lag is reported as unknown (None, '?')",
      odd["engle_granger"]["lags_aic"] is None and "lags chosen by AIC: ? →" in odd_txt, odd["engle_granger"])

# -------------------------------------------------------------------------------------------- 7. cointegration_pretest
pt, pt_txt = quiet(ss.cointegration_pretest, iq.r6, iq.r3)
check("cointegration_pretest on INTQRT: EG by AIC and BIC both ways; lag-sensitive; cointegrated through the "
      "direction where both reject; the full-rank caveat",
      close(pt["EG_p_y_on_x"], c_aic[1], 1e-12) and close(pt["EG_p_y_on_x_bic"], c_bic[1], 1e-12)
      and pt["EG_p_x_on_y"] < 0.05 and pt["EG_p_x_on_y_bic"] < 0.05 and pt["EG_lag_sensitive"]
      and pt["cointegrated_5%"] and pt["johansen_full_rank"] and "do not conclude I(0)" in pt_txt
      and "only a direction where both reject counts" in pt_txt, pt)
rw = ecm_pair(np.random.default_rng(12), 300, 0.0)
pr = ss.cointegration_pretest(rw.y, rw.x, show=False)
check("cointegration_pretest: independent random walks are not cointegrated",
      not pr["cointegrated_5%"] and min(pr["EG_p_y_on_x"], pr["EG_p_x_on_y"]) > 0.05, pr)

orig_eg = dg._engle_granger
try:
    dg._engle_granger = lambda y, xs, trend, crit: {"statistic": -1.0, "p": 0.01 if crit == "aic" else 0.20,
                                                    "lags": 0}
    pf = ss.cointegration_pretest(iq.r6, iq.r3, show=False)
finally:
    dg._engle_granger = orig_eg
check("cointegration_pretest: AIC rejects and BIC does not in both directions → not cointegrated, lag-sensitive",
      not pf["cointegrated_5%"] and pf["EG_lag_sensitive"], pf)

passed = sum(results)
print(f"\n{passed}/{len(results)} checks passed")
sys.exit(0 if passed == len(results) else 1)
