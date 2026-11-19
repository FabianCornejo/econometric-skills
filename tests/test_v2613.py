"""v2.6.13: leverage in iv_report, robust IV tests that hold their size, and the degenerate-data leftovers.

Plan, measurements and reproduction scripts: claude/plano-v2.6.13.md, claude/medicoes-v2.6.13.md and
claude/scripts/v2613/ in the development repository. Every case below gave a misleading number or an error on main
24f8506 (v2.6.12); each boundary has a case on each side. The references do not reuse the implementation's
arithmetic: leverages from the hat matrix written here (or the closed form for one instrument), Wald statistics from
sandwiches written here, F statistics from two sums of squared residuals, Breusch-Pagan as n·R² of the squared
residuals, and the control-function test assembled by hand.

Run from the repository root:  python3 tests/test_v2613.py
"""
import contextlib
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "econometrics" / "scripts"))
import _console  # noqa: E402,F401  (non-ASCII text on a Windows console)
import causal  # noqa: E402
import diagnostics as dg  # noqa: E402
import specification as sp  # noqa: E402
import statsmodels.api as sm  # noqa: E402

results = []


def check(name, cond, info=""):
    results.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info)


def rel(a, b):
    return abs(float(a) - float(b)) / max(abs(float(b)), 1e-300)


def quiet(f, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = f(*a, **k)
    return out, buf.getvalue()


def hat(X):
    """Leverages h_ii = diag(X (X'X)⁻¹ X'), written here."""
    return np.sum((X @ np.linalg.inv(X.T @ X)) * X, axis=1)


def wald_hc(y, X, cols, kind):
    """F = Wald/q of the columns `cols` with an HC0, HC1 or HC3 sandwich written here."""
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    if kind == "HC0":
        w = e ** 2
    elif kind == "HC1":
        w = e ** 2 * n / (n - k)
    else:
        w = (e / (1 - hat(X))) ** 2
    V = XtXi @ (X.T * w) @ X @ XtXi
    bb = b[cols]
    return float(bb @ np.linalg.solve(V[np.ix_(cols, cols)], bb)) / len(cols)


def f_ssr(y, X_r, X_u):
    def ssr(X):
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        return float(np.sum((y - X @ b) ** 2))
    q = X_u.shape[1] - X_r.shape[1]
    return ((ssr(X_r) - ssr(X_u)) / q) / (ssr(X_u) / (len(y) - X_u.shape[1]))


def with_leverage(z, h):
    """z with its first entry set so that row 0 has leverage h in the regression on [1, z] (closed form for one
    regressor: h_00 = 1/n + (z_0 − z̄)² / Σ(z − z̄)²)."""
    z = z.copy(); n = len(z)

    def lev(c):
        zz = z.copy(); zz[0] = c; zc = zz - zz.mean()
        return 1 / n + zc[0] ** 2 / (zc ** 2).sum() - h
    z[0] = brentq(lev, z[1:].mean(), z[1:].mean() + 1e8 * z[1:].std())
    return z


# ============================================================ 1. iv_report: an instrument that lives on one row
rng = np.random.default_rng(261301)
n = 300
z, w, u, v = rng.normal(size=(4, n))
x = 0.6 * z + 0.3 * w + v + 0.5 * u
pil = np.zeros(n); pil[57] = 1.0                        # a dummy with a single 1 (the v2.6.12 validation case)
d = pd.DataFrame(dict(y=1 + 0.5 * x + 0.3 * w + u, x=x, z=z, w=w, pil=pil))
f = "y ~ 1 + w + [x ~ z + pil]"
r1, t1 = quiet(dg.iv_report, f, d)
fs = r1["first_stage"].iloc[0]
X1 = np.column_stack([np.ones(n), w, z, pil])
check("dummy with one 1 as instrument: its row has leverage 1 (hat matrix written here)", abs(hat(X1)[57] - 1) < 1e-10)
check("robust: first-stage F and p NaN, label 'not computed' (main: F of hundreds, 'strong')",
      np.isnan(fs["partial F"]) and np.isnan(fs["partial F p"]) and fs["strength"].startswith("not computed"), fs)
check("robust: AR not computed (NaN, no pieces), endogeneity and overidentification NaN, row 57 named",
      np.isnan(r1["anderson_rubin"]["p_at_zero"]) and r1["anderson_rubin"]["pieces"] == []
      and r1["anderson_rubin"]["shape"].startswith("not computed") and r1["tests"]["p"].isna().all()
      and r1["high_leverage_rows"] == [57], (r1["anderson_rubin"], r1["tests"], r1["high_leverage_rows"]))
f_cls = f_ssr(d.x.to_numpy(), X1[:, :2], X1)
lev_w = [m for m in r1["warnings"] if "leverage 1" in m]
check("the warning names row 57 and gives the classical first-stage F (from two SSRs, 4 decimals)",
      len(lev_w) == 1 and "row(s) 57" in lev_w[0] and f"x: {f_cls:.2f}" in lev_w[0], (lev_w, f_cls))
check("the 2SLS table is kept (decision 4) and the printed text says the AR set is not computed",
      np.isfinite(r1["table"].loc["x", "2SLS"]) and "Anderson-Rubin 95% set for x: not computed" in t1, t1[-600:])
check("no 'Anderson-Rubin set is ...: the data do not pin down the effect' warning when it is not computed",
      not any("do not pin down" in m for m in r1["warnings"]), r1["warnings"])
r1u, _ = quiet(dg.iv_report, f, d, cov_type="unadjusted")
check("unadjusted: nothing refused; classical F = the F from two SSRs (relative 1e-8), AR computed",
      rel(r1u["first_stage"]["partial F"].iloc[0], f_cls) < 1e-8 and r1u["high_leverage_rows"] == []
      and np.isfinite(r1u["anderson_rubin"]["p_at_zero"]), r1u["first_stage"])
d2 = d.copy(); d2.loc[57, "pil"] = 0.0; d2.loc[[57, 58], "pil"] = 1.0       # two rows: leverage below 1
r2, _ = quiet(dg.iv_report, f, d2)
check("the other side: a dummy with two 1s (leverage about 0.5 each) is not refused",
      r2["high_leverage_rows"] == [] and np.isfinite(r2["first_stage"]["partial F"].iloc[0]), r2["warnings"])
d3 = d.copy(); d3.index = d3.index + 1000
r3, _ = quiet(dg.iv_report, f, d3)
check("the rows are named by their index labels (1057)", r3["high_leverage_rows"] == [1057], r3["high_leverage_rows"])

# ============================================================ 2. iv_report: high (not 1) leverage, both sides of 0.5
rng = np.random.default_rng(261302)
n = 200
w, u, v = rng.normal(size=(3, n))
for h_set, expect in ((0.45, False), (0.55, True), (1 - 1e-6, True)):
    z = with_leverage(rng.normal(size=n), h_set)
    x = 0.4 * z + v
    d = pd.DataFrame(dict(y=1 + x + u, x=x, z=z))
    rh, th = quiet(dg.iv_report, "y ~ 1 + [x ~ z]", d)
    Xh = np.column_stack([np.ones(n), z])
    hh = hat(Xh)[0]
    warned = any("from the instruments" in m and "HC3" in m for m in rh["warnings"])
    fs = rh["first_stage"]
    ok = abs(hh - h_set) < 1e-9 and warned == expect and rh["high_leverage_rows"] == []
    if expect:
        f3 = wald_hc(d.x.to_numpy(), Xh, [1], "HC3")
        f0 = wald_hc(d.x.to_numpy(), Xh, [1], "HC0")
        # with 1 − h = 1e-6 the HC3 weights amplify the rounding of h: skill, this sandwich and statsmodels differ by
        # about 2e-8 there, so that case gets 1e-6
        tol = 1e-8 if h_set < 0.9 else 1e-6
        ok = ok and rel(fs["partial F (HC3)"].iloc[0], f3) < tol and rel(fs["partial F"].iloc[0], f0) < 1e-8
        lab_f = min(f0, f3)
        want = "WEAK" if lab_f < 10 else ("acceptable" if lab_f <= 104.7 else "strong")
        ok = ok and fs["strength"].iloc[0].startswith(want) and "[HC3 F, high-leverage row]" in fs["strength"].iloc[0]
    else:
        ok = ok and "partial F (HC3)" not in fs.columns
    check(f"leverage {h_set:g} from the instrument: warning {'with the HC3 F (hand sandwich, 1e-8) and label from the '
          'smaller F' if expect else 'absent'}; HC0 F = hand sandwich", ok, (hh, rh["warnings"], fs))

# leverage that comes from W only does not count (a group dummy with one row is exogenous, not an instrument)
rng = np.random.default_rng(261303)
n = 200
z, u, v = rng.normal(size=(3, n))
g1 = np.zeros(n); g1[3] = 1.0
x = 0.6 * z + v
d = pd.DataFrame(dict(y=1 + x + 0.2 * g1 + u, x=x, z=z, g1=g1))
rw, _ = quiet(dg.iv_report, "y ~ 1 + g1 + [x ~ z]", d)
check("leverage 1 from an exogenous dummy (W), not from the instruments: nothing refused, no leverage warning",
      rw["high_leverage_rows"] == [] and not any("leverage" in m for m in rw["warnings"])
      and np.isfinite(rw["first_stage"]["partial F"].iloc[0]), rw["warnings"])

# ============================================================ 3. Anderson-Rubin with HC3 (by hand), one and two endog.
rng = np.random.default_rng(261304)
n = 150
z1, z2, w, u, v = rng.normal(size=(5, n))
x = 0.5 * z1 + 0.3 * z2 + v + 0.5 * u
d = pd.DataFrame(dict(y=2 + 0.7 * x + 0.4 * w + u * (1 + np.abs(z1)), x=x, z1=z1, z2=z2, w=w))
ra, ta = quiet(dg.iv_report, "y ~ 1 + w + [x ~ z1 + z2]", d)
Xa = np.column_stack([np.ones(n), w, z1, z2])
f_ar0 = wald_hc(d.y.to_numpy(), Xa, [2, 3], "HC3")
check("AR p at β = 0 = HC3 Wald written here (relative 1e-8); key cov = 'HC3'",
      rel(ra["anderson_rubin"]["p_at_zero"], stats.f.sf(f_ar0, 2, n - 4)) < 1e-8 and ra["anderson_rubin"]["cov"] == "HC3",
      (ra["anderson_rubin"]["p_at_zero"], stats.f.sf(f_ar0, 2, n - 4)))
lo, hi = ra["anderson_rubin"]["ci_low"], ra["anderson_rubin"]["ci_high"]
inside = 0.5 * (lo + hi)
p_in = stats.f.sf(wald_hc(d.y.to_numpy() - inside * x, Xa, [2, 3], "HC3"), 2, n - 4)
p_out = stats.f.sf(wald_hc(d.y.to_numpy() - (hi + 0.05 * (hi - lo)) * x, Xa, [2, 3], "HC3"), 2, n - 4)
check("AR set by HC3: its midpoint is accepted and a point 5% of its width past the upper end is rejected (hand)",
      p_in > 0.05 and p_out < 0.05, (lo, hi, p_in, p_out))
ru, _ = quiet(dg.iv_report, "y ~ 1 + w + [x ~ z1 + z2]", d, cov_type="unadjusted")
check("unadjusted AR unchanged: F at β = 0 from two SSRs (relative 1e-8)",
      rel(ru["anderson_rubin"]["p_at_zero"], stats.f.sf(f_ssr(d.y.to_numpy(), Xa[:, :2], Xa), 2, n - 4)) < 1e-8)

# ============================================================ 4. endogeneity in iv_report: control function with HC1
rng = np.random.default_rng(261305)
n = 120
W = rng.normal(size=(n, 5)); z1, z2, v = rng.normal(size=(3, n))
x = 0.6 * z1 + 0.5 * z2 + 0.3 * W[:, 0] + v
y = x + W.sum(axis=1) * 0.2 + 0.4 * v + rng.normal(size=n) * np.exp(0.4 * z1)
d = pd.DataFrame(np.column_stack([y, x, W, z1, z2]), columns=["y", "x"] + [f"w{j}" for j in range(5)] + ["z1", "z2"])
fe = "y ~ 1 + " + " + ".join(f"w{j}" for j in range(5)) + " + [x ~ z1 + z2]"
re_, te = quiet(dg.iv_report, fe, d)
Wc = np.column_stack([np.ones(n), W])
Xfs = np.column_stack([Wc, z1, z2])
vhat = x - Xfs @ np.linalg.lstsq(Xfs, x, rcond=None)[0]
Xt = np.column_stack([Wc, x, vhat])
f_cf = wald_hc(y, Xt, [Xt.shape[1] - 1], "HC1")
row = re_["tests"].loc["endogeneity (control function, HC1)"]
check("robust endogeneity = control-function test with HC1 assembled here (relative 1e-8), p from F(1, n − k)",
      rel(row["statistic"], f_cf) < 1e-8 and rel(row["p"], stats.f.sf(f_cf, 1, n - Xt.shape[1])) < 1e-8, (row, f_cf))
lm = re_["iv"].wooldridge_regression
check("linearmodels' Wooldridge robust regression test kept in 'endogeneity_linearmodels'",
      re_["endogeneity_linearmodels"]["test"] == "Wooldridge robust regression (linearmodels)"
      and rel(re_["endogeneity_linearmodels"]["statistic"], float(lm.stat)) < 1e-12
      and rel(re_["endogeneity_linearmodels"]["p"], float(lm.pval)) < 1e-12, re_["endogeneity_linearmodels"])
ru, _ = quiet(dg.iv_report, fe, d, cov_type="unadjusted")
check("unadjusted: Wu-Hausman as before, 'endogeneity_linearmodels' None",
      "endogeneity (Wu-Hausman)" in ru["tests"].index and ru["endogeneity_linearmodels"] is None
      and rel(ru["tests"].loc["endogeneity (Wu-Hausman)", "statistic"], float(ru["iv"].wu_hausman().stat)) < 1e-12)
rk, tk = quiet(dg.iv_report, fe, d, cov_type="kernel")
check("kernel: the printout says the control-function test uses HC1 (heteroskedasticity only)",
      "The control-function test uses HC1 (heteroskedasticity only), not the kernel covariance." in tk, tk[-500:])
# perfect first stage: the control-function test is not defined
rng = np.random.default_rng(261306)
n = 200
z1, z2, u = rng.normal(size=(3, n))
# (exactly x = z1 + 2·z2 is refused by linearmodels itself: "do not have full column rank")
d = pd.DataFrame(dict(y=1 + z1 + u, x=z1 + 2 * z2 + 1e-9 * rng.normal(size=n), z1=z1, z2=z2))
rp, _ = quiet(dg.iv_report, "y ~ 1 + [x ~ z1 + z2]", d)
check("x = z1 + 2·z2 + 1e-9·e: robust endogeneity test NaN with a warning (main: a p-value, 0.41, from a test regression made of rounding)",
      np.isnan(rp["tests"]["p"].iloc[0]) and any("perfect first stage" in m for m in rp["warnings"]), rp["tests"])

# ============================================================ 5. overidentification: warning from k/n = 0.15
rng = np.random.default_rng(261307)
for n, k_w, expect in ((200, 26, False), (180, 26, True)):         # k = 1 + k_w + 2: 29/200 = 0.145, 29/180 = 0.161
    W = rng.normal(size=(n, k_w)); z1, z2, v, u = rng.normal(size=(4, n))
    x = z1 + z2 + v
    d = pd.DataFrame(np.column_stack([x + u, x, W, z1, z2]),
                     columns=["y", "x"] + [f"w{j}" for j in range(k_w)] + ["z1", "z2"])
    ro, _ = quiet(dg.iv_report, "y ~ 1 + " + " + ".join(f"w{j}" for j in range(k_w)) + " + [x ~ z1 + z2]", d)
    warned = any("robust overidentification test rejects too often" in m for m in ro["warnings"])
    check(f"overidentification, k/n = {29 / n:.3f}: warning {'present' if expect else 'absent'}", warned == expect,
          ro["warnings"])
ro2, _ = quiet(dg.iv_report, "y ~ 1 + " + " + ".join(f"w{j}" for j in range(26)) + " + [x ~ z1 + z2]", d,
               cov_type="unadjusted")
check("unadjusted (Sargan): no such warning", not any("overidentification test rejects" in m for m in ro2["warnings"]))

# ============================================================ 6. endogeneity_test: HC0 with average leverage ≥ 0.15
rng = np.random.default_rng(261308)
for n, k_w, expect in ((200, 27, False), (180, 27, True)):           # k = 1 + 27 + 1 = 29
    W = pd.DataFrame(rng.normal(size=(n, k_w)), columns=[f"w{j}" for j in range(k_w)])
    z, v, u = rng.normal(size=(3, n))
    x = z + v
    o, _ = quiet(dg.endogeneity_test, pd.Series(x + u, name="y"), W, pd.DataFrame({"x": x}), pd.DataFrame({"z": z}),
                 cov_type="HC0")
    warned = any("average leverage k/n" in m for m in o["warnings"])
    o1, _ = quiet(dg.endogeneity_test, pd.Series(x + u, name="y"), W, pd.DataFrame({"x": x}), pd.DataFrame({"z": z}),
                  cov_type="HC1")
    check(f"endogeneity_test HC0, k/n = {29 / n:.3f}: warning {'present' if expect else 'absent'}; HC1 never "
          "gets the k/n reason", warned == expect and not any("average leverage" in m for m in o1["warnings"]),
          (o["warnings"], o1["warnings"]))

# ============================================================ 7. per-column offset floor (v2.6.12 review r1-A N4)
rng = np.random.default_rng(91)
n = 2000
z1, e, z3, w, u, v = rng.normal(size=(6, n))
x = 0.3 * z1 + 0.3 * z3 + v + u
ref = None
for off in (0.0, 1e9, 1e12):
    d = pd.DataFrame(dict(y=x + u, x=x, z1=z1, z2=z1 + 1e-6 * e, z3=off + z3, w=w))
    o, _ = quiet(dg.endogeneity_test, d.y, d[["w"]], d[["x"]], d[["z1", "z2", "z3"]], cov_type="nonrobust")
    if ref is None:
        ref = o["first_stage_F"]["x"]
        X_u = np.column_stack([np.ones(n), w, z1, z1 + 1e-6 * e, z3])
        check("offset 0: F of three instruments = F from two SSRs (relative 1e-6)",
              rel(ref, f_ssr(x, X_u[:, :2], X_u)) < 1e-6, ref)
    check(f"z3 with offset {off:g}: still three instruments and the same F (relative 1e-5; main: two, F 90.68 at 1e12)",
          o["n_instruments"] == 3 and rel(o["first_stage_F"]["x"], ref) < 1e-5, (o["n_instruments"], o["first_stage_F"]))
for off in (0.0, 1e12):
    d = pd.DataFrame(dict(y=x + u, x=x, z1=z1, z2=z1 + 1e-15 * e, z3=off + z3, w=w))
    o, _ = quiet(dg.endogeneity_test, d.y, d[["w"]], d[["x"]], d[["z1", "z2", "z3"]], cov_type="nonrobust")
    check(f"the other side: z2 = z1 + 1e-15·e (rounding) is merged, offset {off:g} on z3: two instruments",
          o["n_instruments"] == 2, o["n_instruments"])
d = pd.DataFrame(dict(y=x + u, x=x, z1=z1, z3=z3, w=1e8 + w, w2=2 * (1e8 + w)))
o, _ = quiet(dg.endogeneity_test, d.y, d[["w", "w2"]], d[["x"]], d[["z1", "z3"]], cov_type="nonrobust")
check("W with an offset column and its double (w2 = 2w): still one W direction, two instruments, finite F",
      o["n_instruments"] == 2 and np.isfinite(o["first_stage_F"]["x"]), o)

# ============================================================ 8. Breusch-Pagan and White with a constant from dummies
rng = np.random.default_rng(261309)
n = 600
g = rng.integers(0, 3, n); xr = rng.normal(size=n)
D = np.eye(3)[g]
y = D @ np.array([1.0, 2.0, 3.0]) + xr + rng.normal(size=n) * np.exp(xr)
res_d = sm.OLS(y, np.column_stack([D, xr])).fit()
res_c = sm.OLS(y, np.column_stack([np.ones(n), D[:, 1:], xr])).fit()
td, _ = quiet(dg.ols_diagnostics, res_d)
tc, _ = quiet(dg.ols_diagnostics, res_c)
td, tc = td.set_index("Test"), tc.set_index("Test")
e2 = res_c.resid ** 2
Xc = np.column_stack([np.ones(n), D[:, 1:], xr])
fit2 = np.linalg.lstsq(Xc, e2, rcond=None)[0]
r2 = 1 - np.sum((e2 - Xc @ fit2) ** 2) / np.sum((e2 - e2.mean()) ** 2)
check("BP by hand (n·R² of û² on the regressors, Koenker) = the explicit-constant model's BP (relative 1e-10)",
      rel(tc.loc["Breusch-Pagan", "Statistic"], n * r2) < 1e-10, (tc.loc["Breusch-Pagan"], n * r2))
check("0 + D1 + D2 + D3 + x: BP computed and equal to the explicit-constant model (relative 1e-10; main: not computed),"
      " with the note", rel(td.loc["Breusch-Pagan", "Statistic"], tc.loc["Breusch-Pagan", "Statistic"]) < 1e-10
      and "written with a constant column" in td.loc["Breusch-Pagan", "Conclusion"], td.loc["Breusch-Pagan"])
check("White (full) present and equal too (relative 1e-10; main: the row vanished)",
      "White (full)" in td.index and rel(td.loc["White (full)", "Statistic"], tc.loc["White (full)", "Statistic"]) < 1e-10,
      td.index.tolist())
res_n = sm.OLS(y, np.column_stack([D[:, 1:], xr])).fit()              # no constant in the span
tn, _ = quiet(dg.ols_diagnostics, res_n)
tn = tn.set_index("Test")
check("no constant in the span: BP and White (full) rows say 'not computed' (none vanishes)",
      str(tn.loc["Breusch-Pagan", "Conclusion"]).startswith("not computed")
      and str(tn.loc["White (full)", "Conclusion"]).startswith("not computed"), tn)

# ============================================================ 9. one constant rule (iv_report, vif_table, oster)
rng = np.random.default_rng(261310)
n = 300
z, u, v, e = rng.normal(size=(4, n))
x = z + v
for off, expect in ((1e7, False), (1e10, True)):
    d = pd.DataFrame(dict(y=1 + x + u, x=x, z=z, w=off + e))
    rc, _ = quiet(dg.iv_report, "y ~ 0 + w + [x ~ z]", d)
    near = any("w varies by less than 1e-08" in m for m in rc["warnings"])
    noc = any("no constant" in m for m in rc["warnings"])
    check(f"iv_report, 0 + w with w = {off:g} + N(0, 1): near-constant warning {'present' if expect else 'absent'}, "
          "'no constant' warning present (main: none at 1e10)", near == expect and noc, rc["warnings"])
X = np.column_stack([1e11 + e, x])
vt = dg.vif_table(sm.OLS(1 + x + u, X).fit())
check("vif_table: 1e11 + N(0, 1) is not a constant: both columns listed, uncentered (main: centered, column dropped)",
      len(vt) == 2 and (vt["Kind"] == "uncentered").all(), vt)
vt1 = dg.vif_table(sm.OLS(1 + x + u, np.column_stack([np.ones(n), e, x])).fit())
check("vif_table: an explicit constant column still makes it centered", len(vt1) == 2 and (vt1["Kind"] == "centered").all())
rng = np.random.default_rng(261311)
n = 500
w6 = 1e6 + rng.normal(size=n); xo = rng.normal(size=n); c = 0.5 * xo + rng.normal(size=n)
yo = 2 + xo + c + rng.normal(size=n)
ru_ = sm.OLS(yo, pd.DataFrame({"w": w6, "x": xo})).fit()
rc_ = sm.OLS(yo, pd.DataFrame({"w": w6, "x": xo, "c": c})).fit()
try:
    sp.oster(ru_, rc_, "x")
    raised = False
except ValueError as ex:
    raised = "no intercept" in str(ex)
check("oster, w = 1e6 + N(0, 1) and no constant (statsmodels k_constant 0, uncentered R²): ValueError (main: ran)",
      raised and ru_.model.k_constant == 0)
g = rng.integers(0, 3, n)
rd_ = sm.OLS(yo, np.column_stack([np.eye(3)[g], xo])).fit()
check("_has_intercept follows statsmodels: a full set of dummies counts (k_constant 1)",
      sp._has_intercept(rd_) and rd_.model.k_constant == 1)

# ============================================================ 10. rdd fuzzy with perfect compliance
rng = np.random.default_rng(261312)
n = 400
xr = rng.uniform(-1, 1, n)
d = pd.DataFrame(dict(x=xr))
d["t"] = (d.x >= 0).astype(float)
d["y"] = 0.5 * d.t + d.x + rng.normal(size=n)
rf, tf = quiet(causal.rdd, d, "y", "x", 0.0, h=0.8, fuzzy="t")
rs, _ = quiet(causal.rdd, d, "y", "x", 0.0, h=0.8)
check("perfect compliance: F = inf, the warning, and the fuzzy estimate = the sharp one (relative 1e-8; main F 1e31)",
      rf["F"] == np.inf and any("perfect compliance" in m for m in rf["warnings"])
      and rel(rf["estimate"], rs["estimate"]) < 1e-8 and "F = inf" in tf, (rf["F"], rf["warnings"]))
t2 = d.t.to_numpy().copy(); t2[np.flatnonzero(d.x.to_numpy() >= 0)[:40]] = 0.0
d["t2"] = t2
rf2, _ = quiet(causal.rdd, d, "y", "x", 0.0, h=0.8, fuzzy="t2")
check("the other side: 40 no-shows above the cutoff → finite F, no warning",
      np.isfinite(rf2["F"]) and not any("perfect compliance" in m for m in rf2["warnings"]), rf2["F"])

# ============================================================ 11. documentation
rules = (ROOT / "econometrics" / "references" / "interpretation-rules.md").read_text(encoding="utf-8")
sec = rules.split("## Rules added in v2.6.13", 1)
check("interpretation-rules.md: a v2.6.13 section with the exact conversion and the large-sample IV bias",
      len(sec) == 2 and "100·(e^b − 1)" in sec[1] and "Cov(z, u)/Cov(z, x)" in sec[1]
      and "biased toward OLS" in sec[1] and "never both" in sec[1])
piv = (ROOT / "econometrics" / "references" / "panel-iv.md").read_text(encoding="utf-8")
check("panel-iv.md: the large-sample bias of IV and the pointer to the rule", "Cov(z, u)/Cov(z, x)" in piv)
doc = dg.iv_report.__doc__
check("iv_report docstring names the new keys and HC3 for the AR",
      "high_leverage_rows" in doc and "endogeneity_linearmodels" in doc and "(HC3 when cov_type is robust or kernel)" in doc)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
