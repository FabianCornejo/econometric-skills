"""v2.6.12: degenerate data in IV (endogeneity_test, iv_report) and three small v2.6.11 leftovers.

Plan, measurements and reproduction scripts: claude/plano-v2.6.12.md, claude/medicoes-v2.6.12.md and
claude/scripts/v2612/ in the development repository. Every case below gave a misleading number or an error on main
874e060 (v2.6.11); each boundary has a case on each side. The references do not reuse the implementation's
arithmetic: F statistics from two sums of squared residuals (np.linalg.lstsq) or from statsmodels' f_test on the
original columns, the robust Wald from a sandwich written here, OLS refitted by hand on the complete rows.

Run from the repository root:  python3 tests/test_v2612.py
"""
import contextlib
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "econometrics" / "scripts"))
import _console  # noqa: E402,F401  (non-ASCII text on a Windows console)
import diagnostics as dg  # noqa: E402
import statsmodels.api as sm  # noqa: E402

results = []


def check(name, cond, info=""):
    results.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info)


def rel(a, b):
    return abs(float(a) - float(b)) / max(abs(float(b)), 1e-300)


def quiet(f, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = f(*a, **k)
    return out, buf.getvalue(), [str(x.category.__name__) for x in w]


def f_ssr(y, X_r, X_u):
    """Classical F of the columns in X_u beyond X_r, from the two sums of squared residuals."""
    def ssr(X):
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        return float(np.sum((y - X @ b) ** 2))
    q = np.linalg.matrix_rank(X_u) - np.linalg.matrix_rank(X_r)
    return ((ssr(X_r) - ssr(X_u)) / q) / (ssr(X_u) / (len(y) - np.linalg.matrix_rank(X_u)))


# ============================================================ 1. endogeneity_test: endogenous regressor in span(W, Z)
rng = np.random.default_rng(26121)
n = 200
z, w, u, e = rng.normal(size=(4, n))
for lab, x in (("x = 2z + w", 2 * z + w), ("x = 2z + w + 1e-15·e", 2 * z + w + 1e-15 * e)):
    d = pd.DataFrame(dict(y=x + u, x=x, z=z, w=w))
    r, t, ws = quiet(dg.endogeneity_test, d.y, d[["w"]], d[["x"]], d[["z"]], cov_type="HC3")
    check(f"{lab}: perfect first stage → F = inf, test NaN, listed, warned, verdict 'not defined', no 'endogenous', "
          "no statsmodels warning", r["first_stage_F"]["x"] == np.inf and np.isnan(r["statistic"]) and np.isnan(r["p"])
          and r["perfect_first_stage"] == ["x"] and any("exact linear combination" in m for m in r["warnings"])
          and "not defined: the first stage fits perfectly" in t and "endogenous" not in t.split("→")[-1]
          and "SingularMatrixWarning" not in ws, (r, t, ws))
for dl in (1e-4, 1e-11):
    x = 2 * z + w + dl * e
    d = pd.DataFrame(dict(y=x + u, x=x, z=z, w=w))
    r, t, _ = quiet(dg.endogeneity_test, d.y, d[["w"]], d[["x"]], d[["z"]], cov_type="nonrobust")
    ref = f_ssr(d.x.to_numpy(), np.column_stack([np.ones(n), w]), np.column_stack([np.ones(n), w, z]))
    # tolerance: x carries dl of signal on entries of size ~3, so its first-stage residual has a relative rounding of
    # about 3·eps/dl (4e-5 at 1e-11); 1e-3 covers it, 1e-6 at 1e-4
    tol_ = 1e-6 if dl > 1e-6 else 1e-3
    check(f"x = 2z + w + {dl:g}·e (the other side): not perfect, F = F from two SSRs (relative {tol_:g}), test computed",
          r["perfect_first_stage"] == [] and rel(r["first_stage_F"]["x"], ref) < tol_ and np.isfinite(r["statistic"]),
          (r["first_stage_F"], ref))

# ============================================================ 2. endogeneity_test: leverage 1 (instrument on one row)
rng = np.random.default_rng(4)                         # the design of the v2.6.9 review (lev4.py, seed 4)
n = 1000
z, w, u, v = rng.normal(size=(4, n))
zb = np.zeros(n); zb[5] = 1.0
x = 0.3 * z + v + u + 2 * zb
d = pd.DataFrame(dict(y=x + u, x=x, z=z, zb=zb, w=w))
nr_ref = f_ssr(d.x.to_numpy(), np.column_stack([np.ones(n), w]), np.column_stack([np.ones(n), w, z, zb]))
for cov in ("HC0", "HC1", "HC2", "HC3"):
    r, t, _ = quiet(dg.endogeneity_test, d.y, d[["w"]], d[["x"]], d[["z", "zb"]], cov_type=cov)
    msg = " ".join(r["warnings"])
    check(f"leverage 1, {cov}: F and test NaN (were 505.7 with HC0), row 5 named, nonrobust F {nr_ref:.2f} in the "
          "warning, verdict 'not computed', no first-stage line",
          np.isnan(r["first_stage_F"]["x"]) and np.isnan(r["statistic"]) and r["high_leverage_rows"] == [5]
          and "row(s) 5 have leverage 1 in the first stage" in msg and f"x: {nr_ref:.2f}" in msg
          and f"not computed with {cov}: rows with leverage 1" in t and "First stage for x" not in t, (r, t))
r, t, _ = quiet(dg.endogeneity_test, d.y, d[["w"]], d[["x"]], d[["z", "zb"]], cov_type="nonrobust")
check("leverage 1, nonrobust: the F is computed (= F from two SSRs, 1e-8) and no leverage warning (the covariance "
      "is not degenerate)", rel(r["first_stage_F"]["x"], nr_ref) < 1e-8 and r["high_leverage_rows"] == []
      and np.isfinite(r["statistic"]), r)
d5 = d.drop(index=5)
r5, _, _ = quiet(dg.endogeneity_test, d5.y, d5[["w"]], d5[["x"]], d5[["z", "zb"]], cov_type="HC3")
rz, _, _ = quiet(dg.endogeneity_test, d5.y, d5[["w"]], d5[["x"]], d5[["z"]], cov_type="HC3")
check("leverage 1: with row 5 dropped (what the warning says) zb is all zero, one instrument, and the HC3 F equals "
      "the F of z alone (1e-12)", r5["n_instruments"] == 1 and rel(r5["first_stage_F"]["x"], rz["first_stage_F"]["x"])
      < 1e-12, (r5, rz))
zb2 = zb.copy(); zb2[6] = 1.0                          # the other side: two rows, leverage below 1
d2 = d.assign(zb=zb2)
r, _, _ = quiet(dg.endogeneity_test, d2.y, d2[["w"]], d2[["x"]], d2[["z", "zb"]], cov_type="HC3")
check("dummy with two 1s (leverage < 1): F and test computed, no leverage warning", np.isfinite(r["first_stage_F"]["x"])
      and np.isfinite(r["statistic"]) and r["high_leverage_rows"] == [], r)
# leverage 1 only in the test regression: x = 2z + w + 3·e_7 exactly; e_7 enters span(W, x, v̂), not span(W, Z)
rng = np.random.default_rng(26122)
n = 300
z, w, u = rng.normal(size=(3, n))
e7 = np.zeros(n); e7[7] = 3.0
x = 2 * z + w + e7
d = pd.DataFrame(dict(y=x + u, x=x, z=z, w=w))
r, t, _ = quiet(dg.endogeneity_test, d.y, d[["w"]], d[["x"]], d[["z"]], cov_type="HC1")
check("leverage 1 only in the test regression (x = 2z + w + 3·e_7): first-stage F computed, test NaN, row 7 named "
      "'in the test regression'", np.isfinite(r["first_stage_F"]["x"]) and np.isnan(r["statistic"])
      and r["high_leverage_rows"] == [7] and any("in the test regression" in m for m in r["warnings"]), (r, t))

# ============================================================ 3. endogeneity_test: few residual df with HC0-HC2
rng = np.random.default_rng(26123)
k_z = 3


def few_df(df_res, cov):
    n_ = 2 + k_z + df_res
    M = rng.normal(size=(n_, 3 + k_z))
    dd = pd.DataFrame(M, columns=["y", "x", "w"] + [f"s{k}" for k in range(k_z)])
    return quiet(dg.endogeneity_test, dd.y, dd[["w"]], dd[["x"]], dd[[f"s{k}" for k in range(k_z)]], cov_type=cov)


for cov in ("HC0", "HC1", "HC2"):
    (r29, t29, _), (r30, t30, _) = few_df(29, cov), few_df(30, cov)
    check(f"{cov}: 29 residual df → warning that the robust F rejects too often, pointing to HC3; 30 df → none",
          any(m.startswith(f"{cov} with 29 residual degrees of freedom") and "cov_type='HC3'" in m for m in r29["warnings"])
          and not any("residual degrees of freedom in the first stage" in m for m in r30["warnings"]), (r29, r30))
r3, _, _ = few_df(2, "HC3")
check("HC3 with 2 residual df: no few-df warning (HC3 never had P(F > 10) above 5.4% in the measurement)",
      not any("residual degrees of freedom in the first stage" in m for m in r3["warnings"]), r3)
rn, _, _ = few_df(2, "nonrobust")
check("nonrobust with 2 residual df: no few-df warning (the F has its own distribution)",
      not any("residual degrees of freedom in the first stage" in m for m in rn["warnings"]), rn)
check("the few-df threshold is 30 (claude/medicoes-v2.6.12.md, E3)", dg._ROBUST_SMALL_DF == 30)

# ============================================================ 4. endogeneity_test: rank tolerance with large n
rng = np.random.default_rng(26124)
n = 100_000
z, w, u, v, e = rng.normal(size=(5, n))
x = 0.3 * z + 0.2 * w + v + u
d = pd.DataFrame(dict(y=x + u, x=x, z=z, w=w))
Xr = np.column_stack([np.ones(n), w])
for dl in (1e-10, 1e-12):
    dd = d.assign(z2=d.z + dl * e)
    r, _, _ = quiet(dg.endogeneity_test, dd.y, dd[["w"]], dd[["x"]], dd[["z", "z2"]], cov_type="nonrobust")
    # reference: the same span written well conditioned, {1, w, z, z + dl·e} = {1, w, z, e} (v2.6.9 lesson); an
    # SSR F on (z, z2) themselves would need numpy's rank rule, the one this version replaces. Tolerance 1e-3: z2 was
    # rounded to eps·|z| against dl of signal in the e direction
    ref = f_ssr(dd.x.to_numpy(), Xr, np.column_stack([Xr, dd.z, e]))
    check(f"n = 1e5, z and z + {dl:g}·e: two instruments (main counted one at 1e-10 and printed 4458.8), F = F of the "
          "same span written as (z, e), from two SSRs (relative 1e-3)",
          r["n_instruments"] == 2 and rel(r["first_stage_F"]["x"], ref) < 1e-3,
          (r["n_instruments"], r["first_stage_F"], ref))
dd = d.assign(z2=d.z + 1e-15 * e)
r, _, _ = quiet(dg.endogeneity_test, dd.y, dd[["w"]], dd[["x"]], dd[["z", "z2"]], cov_type="nonrobust")
check("n = 1e5, z and z + 1e-15·e (the other side, rounding): one instrument", r["n_instruments"] == 1, r)
# (review r1: a check of _rank_tol against its own formula was dropped; the two cases above fix the tolerance by
# behavior, one on each side)

# ============================================================ 5. endogeneity_test: names
rng = np.random.default_rng(26125)
n = 100
z, w, u, v = rng.normal(size=(4, n))
x = 0.5 * z + v + u
d = pd.DataFrame(dict(y=x + u, x=x, z=z, w=w))
for lab, args in (("instrument named like an exogenous column (other data)",
                   (d.y, d[["w"]], d[["x"]], d[["z"]].rename(columns={"z": "w"}))),
                  ("the same column as exogenous and as instrument", (d.y, d[["w", "z"]], d[["x"]], d[["z"]])),
                  ("instrument named like y", (d.y, d[["w"]], d[["x"]], d[["z"]].rename(columns={"z": "y"})))):
    try:
        dg.endogeneity_test(*args, show=False)
        ok, m = False, "no error"
    except ValueError as ex:
        m = str(ex)
        ok = "appear more than once across y, X_exog, x_endog and Z" in m
    check(f"{lab}: ValueError naming the repeated column (main dropped the instrument silently)", ok, m)
try:
    dg.endogeneity_test(pd.Series(d.y.to_numpy()), d[["w"]], d[["x"]], d[["z"]], show=False)
    ok, m = False, "no error"
except ValueError as ex:
    m = str(ex); ok = "named Series" in m
check("unnamed y: ValueError that asks for a named Series", ok, m)

# ============================================================ 6. iv_report: same rows for OLS and 2SLS; NaN
rng = np.random.default_rng(26126)
n = 600
z, w, u, v = rng.normal(size=(4, n))
x = 0.5 * z + v + u
d = pd.DataFrame(dict(y=1 + x + 0.5 * w + u, x=x, z=z, w=w))
d.loc[:199, "z"] = np.nan
d["unused"] = np.nan                                   # a column outside the formula does not drop rows
r, t, _ = quiet(dg.iv_report, "y ~ 1 + w + [x ~ z]", d)
c = d.dropna(subset=["y", "x", "z", "w"])
ols_hand = sm.OLS(c.y, sm.add_constant(c[["w", "x"]])).fit()
check("NaN only in the instrument: 2SLS and OLS on the same 400 rows (main: OLS on 600; linearmodels 6.1 stopped), "
      "n_dropped 200, warning", r["n"] == 400 and int(r["ols"].nobs) == 400 and r["n_dropped"] == 200
      and any(m.startswith("200 of 600 rows dropped") for m in r["warnings"]), (r["n"], r["ols"].nobs, r["warnings"]))
check("OLS of the report = OLS by hand on the complete rows (relative 1e-10)",
      max(rel(r["table"].loc[k, "OLS"], ols_hand.params[k2]) for k, k2 in (("w", "w"), ("x", "x"), ("Intercept", "const")))
      < 1e-10, (r["table"]["OLS"], ols_hand.params))
dc = d.drop(columns="unused").dropna()
rc, _, _ = quiet(dg.iv_report, "y ~ 1 + w + [x ~ z]", dc)
check("complete data (the other side): n_dropped 0, no rows-dropped warning, same 2SLS as with the NaN rows",
      rc["n_dropped"] == 0 and not any("rows dropped" in m for m in rc["warnings"])
      and rel(rc["table"].loc["x", "2SLS"], r["table"].loc["x", "2SLS"]) < 1e-12, rc["warnings"])

# ============================================================ 7. iv_report: homoskedastic, constant, keys
rng = np.random.default_rng(26127)
n = 400
z1, z2, w, u, v = rng.normal(size=(5, n))
x = 0.4 * z1 + 0.3 * z2 + v + u
d = pd.DataFrame(dict(y=1 + x + w + u, x=x, z1=z1, z2=z2, w=w))
ru, _, _ = quiet(dg.iv_report, "y ~ 1 + w + [x ~ z1 + z2]", d, cov_type="unadjusted")
rh, _, _ = quiet(dg.iv_report, "y ~ 1 + w + [x ~ z1 + z2]", d, cov_type="homoskedastic")
check("homoskedastic = unadjusted: Wu-Hausman and Sargan (main: 'Wooldridge robust'), same statistics, same AR p "
      "(main: HC1)", list(rh["tests"].index) == list(ru["tests"].index) and "Wu-Hausman" in rh["tests"].index[0]
      and np.allclose(rh["tests"]["statistic"], ru["tests"]["statistic"], rtol=1e-12)
      and rel(rh["anderson_rubin"]["p_at_zero"] + 1e-300, ru["anderson_rubin"]["p_at_zero"] + 1e-300) < 1e-9,
      (rh["tests"], ru["tests"]))
g = rng.integers(0, 4, n)
dg_ = d.assign(g=g, y=g + d.x + u)
r0, _, _ = quiet(dg.iv_report, "y ~ 0 + C(g) + [x ~ z1]", dg_)
r1, _, _ = quiet(dg.iv_report, "y ~ 0 + w + [x ~ z1]", dg_)
check("0 + C(g): the full dummies span the constant → no 'no constant' warning (main warned)",
      not any("no constant" in m for m in r0["warnings"]), r0["warnings"])
check("0 + w (the other side): no constant in the span → the warning stays",
      any("no constant" in m for m in r1["warnings"]), r1["warnings"])
doc = dg.iv_report.__doc__
check("iv_report docstring lists every key of the result",
      all(k in doc for k in ru if k not in ("iv", "ols")) and "iv, ols" in doc, [k for k in ru if k not in doc])

# ============================================================ 8. iv_report: classical unadjusted F
fs_c = f_ssr(d.x.to_numpy(), np.column_stack([np.ones(n), w]), np.column_stack([np.ones(n), w, z1, z2]))
pf = float(ru["first_stage"]["partial F"].iloc[0])
check("unadjusted partial F = classical F from two SSRs (relative 1e-10; main gave n/(n − k) = 400/396 times it)",
      rel(pf, fs_c) < 1e-10, (pf, fs_c))
check("its p = F(2, 396) upper tail of that F (relative 1e-8)",
      rel(ru["first_stage"]["partial F p"].iloc[0], stats.f.sf(fs_c, 2, n - 4)) < 1e-8)
rr, _, _ = quiet(dg.iv_report, "y ~ 1 + w + [x ~ z1 + z2]", d)
Xf = np.column_stack([np.ones(n), w, z1, z2])
hc0 = sm.OLS(d.x, Xf).fit(cov_type="HC0").wald_test(np.eye(4)[2:], scalar=True, use_f=True).statistic
check("robust partial F unchanged: statsmodels HC0 Wald F (relative 1e-8)",
      rel(rr["first_stage"]["partial F"].iloc[0], float(hc0)) < 1e-8)

# ============================================================ 9. iv_report: joint Anderson-Rubin with two endogenous
rng = np.random.default_rng(26128)
n = 500
z1, z2, z3, w, u, v1, v2 = rng.normal(size=(7, n))
x1 = 0.5 * z1 + 0.2 * z3 + v1 + u
x2 = 0.5 * z2 + v2 + u
d = pd.DataFrame(dict(y=1 + 0.3 * w + x1 + x2 + u, x1=x1, x2=x2, z1=z1, z2=z2, z3=z3, w=w))
ra, ta, _ = quiet(dg.iv_report, "y ~ 1 + w + [x1 + x2 ~ z1 + z2 + z3]", d, cov_type="unadjusted")
W2 = np.column_stack([np.ones(n), w])
ar_ref = f_ssr(d.y.to_numpy(), W2, np.column_stack([W2, z1, z2, z3]))
check("two endogenous, unadjusted: joint AR F at β = 0 = F of the instruments from two SSRs (relative 1e-10), "
      "df (3, n − 5), no set", rel(ra["anderson_rubin"]["joint_F_at_zero"], ar_ref) < 1e-10
      and ra["anderson_rubin"]["df"] == (3, n - 5) and ra["anderson_rubin"]["pieces"] is None
      and rel(ra["anderson_rubin"]["joint_p_at_zero"], stats.f.sf(ar_ref, 3, n - 5)) < 1e-8, ra["anderson_rubin"])
check("two endogenous: the printed AR line says joint test, F(3, 495) and that the set is not computed",
      "Anderson-Rubin, joint test that the coefficients of x1, x2 are all zero: F(3, 495)" in ta
      and "not computed (with 2 endogenous regressors the set is 2-dimensional)" in ta, ta[-400:])
rb, _, _ = quiet(dg.iv_report, "y ~ 1 + w + [x1 + x2 ~ z1 + z2 + z3]", d)
Xa = np.column_stack([W2, z1, z2, z3])
bh = np.linalg.lstsq(Xa, d.y.to_numpy(), rcond=None)[0]
ea = d.y.to_numpy() - Xa @ bh
XtXi = np.linalg.inv(Xa.T @ Xa)
Vh = XtXi @ (Xa.T * ea ** 2) @ Xa @ XtXi * n / (n - Xa.shape[1])        # HC1 sandwich by hand
bz = bh[2:]
wald = float(bz @ np.linalg.solve(Vh[2:, 2:], bz)) / 3
check("two endogenous, robust: joint AR F = HC1 Wald F written by hand (relative 1e-8)",
      rel(rb["anderson_rubin"]["joint_F_at_zero"], wald) < 1e-8, (rb["anderson_rubin"]["joint_F_at_zero"], wald))
# size under H0 (β = 0) with weak instruments: the AR test is exact under normal homoskedastic errors.
# Band fixed before running: 1000 samples, 5% ± 3 Monte Carlo SEs (0.69 pp) = [2.9%, 7.1%]
rng = np.random.default_rng(26129)
rej, reps, nn = 0, 1000, 120
for _ in range(reps):
    z1, z2, z3, uu, a1, a2 = rng.normal(size=(6, nn))
    x1 = 0.05 * z1 + a1 + 0.8 * uu
    x2 = 0.05 * z2 + a2 + 0.8 * uu
    ds = pd.DataFrame(dict(y=1 + uu, x1=x1, x2=x2, z1=z1, z2=z2, z3=z3))
    rej += dg.iv_report("y ~ 1 + [x1 + x2 ~ z1 + z2 + z3]", ds, cov_type="unadjusted",
                        show=False)["anderson_rubin"]["joint_p_at_zero"] < 0.05
check(f"joint AR test, weak instruments (π = 0.05), endogeneity 0.8, β = 0: size {rej / reps:.3f} in [2.9%, 7.1%]",
      0.029 <= rej / reps <= 0.071, rej / reps)
we = [m for m in quiet(dg.iv_report, "y ~ 1 + [x1 + x2 ~ z1 + z2 + z3]", ds)[0]["warnings"] if m.startswith("weak")]
check("two endogenous, weak: the warning points to the joint Anderson-Rubin test, not to a set",
      len(we) == 1 and "joint Anderson-Rubin test" in we[0] and "Anderson-Rubin set" not in we[0], we)

# ============================================================ 10. one convention for F = 10
check("endogeneity_test label: below 10 WEAK, 10 acceptable (the iv_report rule, F < 10 weak)",
      dg._first_stage_label(9.999) == "WEAK" and dg._first_stage_label(10.0).startswith("acceptable"))
rng = np.random.default_rng(261213)
n = 400
z, u, v = rng.normal(size=(3, n))
x = 0.05 * z + v + 0.8 * u
r, t, _ = quiet(dg.endogeneity_test, pd.Series(1 + x + u, name="y"), None, pd.DataFrame({"x": x}),
                pd.DataFrame({"z": z}))
check("weak-instrument verdict prints '(first-stage F < 10)' (it printed '<= 10')",
      r["first_stage_F"]["x"] < 10 and "(first-stage F < 10)" in t and "<= 10" not in t, t[-300:])

# ============================================================ 11. v2.6.11 leftovers
rng = np.random.default_rng(261210)
n = 200
x1, x2, e = rng.normal(size=(3, n))
y = 1 + x1 + 0.5 * x2 + e
res = sm.OLS(y, np.column_stack([x1, x2])).fit()
tab, txt, _ = quiet(dg.ols_diagnostics, res)
bp = tab[tab["Test"] == "Breusch-Pagan"]
check("ols_diagnostics without a constant: runs (main stopped with the Breusch-Pagan ValueError); Breusch-Pagan "
      "'not computed', the other rows present", len(bp) == 1 and np.isnan(bp["p-value"].iloc[0])
      and str(bp["Conclusion"].iloc[0]).startswith("not computed") and (tab["Test"] == "White simplified (ŷ, ŷ²)").any(),
      tab[["Test", "Conclusion"]])
res_c = sm.OLS(y, sm.add_constant(np.column_stack([x1, x2]))).fit()
tab_c, _, _ = quiet(dg.ols_diagnostics, res_c)
bp_c = tab_c[tab_c["Test"] == "Breusch-Pagan"]
check("ols_diagnostics with a constant (the other side): Breusch-Pagan computed (finite p)",
      len(bp_c) == 1 and np.isfinite(bp_c["p-value"].iloc[0]), bp_c)
t_ = np.arange(300)
# trend + level shift + noise (the design of claude/scripts/v2612/repro_pendencias_v2611.py); the first seed from
# 261212 on whose level lines include "Conflict" while the differences are stationary (searched here, bounded)
st, lv = None, []
for sd in range(261212, 261212 + 40):
    rng = np.random.default_rng(sd)
    s = np.cumsum(rng.normal(size=300)) * 0.05 + 0.02 * t_ + np.where(t_ > 150, 3, 0) + rng.normal(size=300)
    cand, _, _ = quiet(dg.stationarity, pd.Series(s, name="spread"), show=False)
    lv_ = list(cand.loc[cand.Transform == "level", "Verdict"])
    d1 = cand.loc[(cand.Transform == "1st difference") & (cand.Deterministics == "constant"), "Verdict"].iloc[0]
    if any(x_.startswith("Conflict") for x_ in lv_) and d1 == "Stationary" and "Stationary" not in lv_:
        st, lv = cand, lv_
        break
check("stationarity: a design with a level line in 'Conflict' and stationary differences exists in 40 seeds",
      st is not None)
st = st if st is not None else cand
check("stationarity: a level line in 'Conflict' and stationary differences → 'I(1)' with the conflict named "
      "(main: bare 'I(1)')", any(x_.startswith("Conflict") for x_ in lv)
      and st.attrs["suggested_order"].startswith("I(1) by the differences, but the level tests conflict"),
      (lv, st.attrs["suggested_order"]))
rw = pd.Series(np.cumsum(np.random.default_rng(261211).normal(size=300)), name="rw")
st2, _, _ = quiet(dg.stationarity, rw)
lv2 = list(st2.loc[st2.Transform == "level", "Verdict"])
check("stationarity: random walk with no 'Conflict' (the other side) → plain 'I(1)'",
      not any(x_.startswith("Conflict") for x_ in lv2) and st2.attrs["suggested_order"] == "I(1)",
      (lv2, st2.attrs["suggested_order"]))

# ============================================================ 12. review round 1 (r1-A and r1-B)
# r1-A B1: a row fitted exactly by W alone (an exogenous group dummy with one observation) is left out: the F equals
# statsmodels on the original columns (the reviewer's design)
rng = np.random.default_rng(1)
n = 300
z, w, u, v = rng.normal(size=(4, n))
g = np.zeros(n); g[3] = 1.0
x = 0.5 * z + v + u + 2 * g
d = pd.DataFrame(dict(y=x + u, x=x, z=z, w=w, g=g))
for cov in ("HC0", "HC3"):
    r, _, _ = quiet(dg.endogeneity_test, d.y, d[["w", "g"]], d[["x"]], d[["z"]], cov_type=cov)
    ref = sm.OLS(d.x, sm.add_constant(d[["w", "g", "z"]])).fit(cov_type=cov).wald_test(
        np.array([[0, 0, 0, 1.0]]), scalar=True, use_f=True).statistic
    check(f"exogenous dummy with one 1, {cov}: F = statsmodels on the original columns (relative 1e-10), test "
          "computed, a note that row 3 is left out, no leverage-1 refusal", rel(r["first_stage_F"]["x"], ref) < 1e-10
          and np.isfinite(r["statistic"]) and r["high_leverage_rows"] == []
          and any(m.startswith("row(s) 3 are fitted exactly by the exogenous regressors") for m in r["warnings"]),
          (r, float(ref)))
# r1-A S1: leverage near 1 from an instrument. HC0/HC1 warned from leverage 0.5 (measured: size above 10%),
# refused within 1e-8 of 1; HC2/HC3 refused only within 1e-12 of 1 (HC3 stays accurate)
rng = np.random.default_rng(4)
n = 1000
z, w, u, v = rng.normal(size=(4, n))
for z6, hc0_nan, hc3_nan in ((1e-4, True, False), (3e-4, False, False)):
    zb = np.zeros(n); zb[5] = 1.0; zb[6] = z6
    x = 0.3 * z + v + u + 2 * zb
    d = pd.DataFrame(dict(y=x + u, x=x, z=z, zb=zb, w=w))
    r0, _, _ = quiet(dg.endogeneity_test, d.y, d[["w"]], d[["x"]], d[["z", "zb"]], cov_type="HC0")
    r3, _, _ = quiet(dg.endogeneity_test, d.y, d[["w"]], d[["x"]], d[["z", "zb"]], cov_type="HC3")
    check(f"leverage 1 − {z6 ** 2:.0e} (zb[6] = {z6:g}): HC0 {'refused' if hc0_nan else 'computed with the high-leverage '
          'warning'}, HC3 computed (≈ 25.8)", (np.isnan(r0["first_stage_F"]["x"]) == hc0_nan)
          and (hc0_nan or any(m.startswith("HC0 with a row with leverage") for m in r0["warnings"]))
          and np.isfinite(r3["first_stage_F"]["x"]) and abs(r3["first_stage_F"]["x"] - 25.8) < 0.1
          and not any("leverage" in m for m in r3["warnings"]), (r0["first_stage_F"], r3["first_stage_F"], r0["warnings"]))


def lev_design(h, cov, seed=261214):
    rng_ = np.random.default_rng(seed)
    n_ = 200
    w_, z_, x_, y_ = rng_.normal(size=(4, n_))
    z_ = z_ - z_.mean(); z_[0] = 0.0
    s_ = np.sum(z_[1:] ** 2)
    z_[0] = np.sqrt((h - 1 / n_) * s_ / ((1 - 1 / n_) * (1 - h + 1 / n_)))
    dd = pd.DataFrame(dict(y=y_, x=x_, z=z_, w=w_))
    return quiet(dg.endogeneity_test, dd.y, dd[["w"]], dd[["x"]], dd[["z"]], cov_type=cov)[0]


for cov in ("HC0", "HC1", "HC2"):
    hi, lo = lev_design(0.52, cov), lev_design(0.45, cov)
    check(f"{cov}: a row with leverage 0.52 → high-leverage warning (use HC3); 0.45 → none (threshold 0.5, "
          "claude/dados/v2612/size_leverage.txt)", any("with a row with leverage 0.5" in m for m in hi["warnings"])
          and not any("with a row with leverage" in m for m in lo["warnings"]), (hi["warnings"], lo["warnings"]))
check("HC3 with leverage 0.9: no high-leverage warning", not any("leverage" in m for m in lev_design(0.9, "HC3")["warnings"]))
check("the high-leverage threshold is 0.5", dg._ROBUST_HIGH_LEVERAGE == 0.5)
# r1-A S2: endogenous regressors jointly degenerate
rng = np.random.default_rng(11)
n = 400
z1, z2, z3, w, u, v = rng.normal(size=(6, n))
x1 = 0.5 * z1 + 0.3 * z2 + v + u
e = rng.normal(size=n)
for lab, x2, deg in (("x2 = x1 + z1", x1 + z1, True), ("x2 = 2·x1", 2 * x1, True),
                     ("x2 = x1 + z1 + 1e-12·e", x1 + z1 + 1e-12 * e, True), ("x2 = x1 + z1 + 0.3·e", x1 + z1 + 0.3 * e, False)):
    d = pd.DataFrame(dict(y=x1 + x2 + u, x1=x1, x2=x2, z1=z1, z2=z2, z3=z3, w=w))
    r, t, _ = quiet(dg.endogeneity_test, d.y, d[["w"]], d[["x1", "x2"]], d[["z1", "z2", "z3"]], cov_type="HC3")
    check(f"{lab}: {'jointly degenerate → test NaN, warning, verdict not defined' if deg else 'the other side: computed'}",
          (np.isnan(r["statistic"]) and any("jointly degenerate" in m for m in r["warnings"])
           and "jointly degenerate (see the warning)" in t) if deg else
          (np.isfinite(r["statistic"]) and not any("jointly degenerate" in m for m in r["warnings"])), (r, t[-200:]))
# r1-A S3: leverage of the test regression does not depend on the scale of x
rng = np.random.default_rng(26122)
n = 300
z, w, u = rng.normal(size=(3, n))
e7 = np.zeros(n); e7[7] = 3.0
for sc in (1e13, 1e15):
    x = sc * (2 * z + w + e7)
    d = pd.DataFrame(dict(y=x / sc + u, x=x, z=z, w=w))
    r, _, _ = quiet(dg.endogeneity_test, d.y, d[["w"]], d[["x"]], d[["z"]], cov_type="HC1")
    check(f"x at scale {sc:g} with leverage 1 in the test regression: row 7 found, test NaN (it gave 170.07)",
          r["high_leverage_rows"] == [7] and np.isnan(r["statistic"]), r)
# r1-A S4: the offset floor applies to W
for n in (200, 5000):
    rng = np.random.default_rng(21)
    z, a, b, u, v = rng.normal(size=(5, n))
    w1 = 1e5 + a; w3 = 0.1 * w1 + 0.7 * b
    x = 0.4 * z + v + u
    d = pd.DataFrame(dict(y=x + u, x=x, z=z, w1=w1, w2=b, w3=w3))
    r, _, _ = quiet(dg.endogeneity_test, d.y, d[["w1", "w2", "w3"]], d[["x"]], d[["z"]], cov_type="nonrobust")
    ref = f_ssr(d.x.to_numpy(), np.column_stack([np.ones(n), a, b]), np.column_stack([np.ones(n), a, b, z]))
    check(f"n = {n}, w3 = 0.1·(1e5 + a) + 0.7·b: W has 3 directions, F = F from two SSRs with (1, a, b) (relative 1e-6)",
          rel(r["first_stage_F"]["x"], ref) < 1e-6, (r["first_stage_F"], ref))
rng = np.random.default_rng(7)
n = 500
z, w, u, e = rng.normal(size=(4, n))
d = pd.DataFrame(dict(y=2 * z + w + u, x=2 * z + w, z=z, w=1e8 + w))
r, _, _ = quiet(dg.endogeneity_test, d.y, d[["w"]], d[["x"]], d[["z"]], cov_type="nonrobust")
check("W stored as 1e8 + w, x = 2z + w: perfect first stage (it gave F = 9.3e19, 'strong')",
      r["perfect_first_stage"] == ["x"] and r["first_stage_F"]["x"] == np.inf and np.isnan(r["statistic"]), r)
d2 = d.assign(x=d.x + 1e-4 * e)
r, _, _ = quiet(dg.endogeneity_test, d2.y, d2[["w"]], d2[["x"]], d2[["z"]], cov_type="nonrobust")
check("the other side, x + 1e-4·e: not perfect, F finite", r["perfect_first_stage"] == []
      and np.isfinite(r["first_stage_F"]["x"]), r)
# r1-A N1, N2, N3
rng = np.random.default_rng(3)
M = rng.normal(size=(4, 5))
d = pd.DataFrame(M, columns=["y", "x", "w", "z1", "z2"])
r, _, _ = quiet(dg.endogeneity_test, d.y, d[["w"]], d[["x"]], d[["z1", "z2"]])
check("saturated first stage: no 'perfect' claim (F NaN)", r["perfect_first_stage"] == []
      and not any("exact linear combination" in m for m in r["warnings"]), r)
rng = np.random.default_rng(26125)
n = 30
z, w, u = rng.normal(size=(3, n))
d = pd.DataFrame(dict(y=2 * z + w + u, x=2 * z + w, z=z, w=w))
r, _, _ = quiet(dg.endogeneity_test, d.y, d[["w"]], d[["x"]], d[["z"]], cov_type="HC0")
check("perfect first stage with HC0 and few df: no few-df warning", r["perfect_first_stage"] == ["x"]
      and not any("reject too often" in m for m in r["warnings"]), r["warnings"])
rng = np.random.default_rng(5)
n = 200
z, z2, u, v, v2 = rng.normal(size=(5, n))
x1, x2 = 0.5 * z + v + u, 0.5 * z2 + v2
d = pd.DataFrame({"y": x1 + x2 + u, "_wdir1": x1, "vhat__wdir1": x2, "z": z, "z2": z2})
r, _, _ = quiet(dg.endogeneity_test, d.y, None, d[["_wdir1", "vhat__wdir1"]], d[["z", "z2"]])
check("endogenous regressors named '_wdir1' and 'vhat__wdir1' (internal names before): no AssertionError, F and test "
      "finite", all(np.isfinite(f) for f in r["first_stage_F"].values()) and np.isfinite(r["statistic"]), r)
# r1-B B2: repeated index labels
rng = np.random.default_rng(26126)
n = 150
z, w, u, v = rng.normal(size=(4, 2 * n))
x = 0.5 * z + v + u
dd = pd.DataFrame(dict(y=1 + x + 0.5 * w + u, x=x, z=z, w=w))
dup = pd.concat([dd.iloc[:n].reset_index(drop=True), dd.iloc[n:].reset_index(drop=True)])
r, _, _ = quiet(dg.iv_report, "y ~ 1 + w + [x ~ z]", dup)
ols_h = sm.OLS(dup.y.to_numpy(), sm.add_constant(dup[["w", "x"]].to_numpy())).fit()
check("repeated index labels (two halves concatenated): OLS on the same 300 rows as 2SLS (it ran on 600), equal to OLS "
      "by hand (relative 1e-10), warning", r["n"] == 300 and int(r["ols"].nobs) == 300
      and rel(r["table"].loc["x", "OLS"], ols_h.params[2]) < 1e-10 and any("repeated labels" in m for m in r["warnings"]),
      (r["n"], r["ols"].nobs))
r, _, _ = quiet(dg.iv_report, "y ~ 1 + w + [x ~ z]", dd)
check("unique index (the other side): no repeated-labels warning, index kept",
      not any("repeated labels" in m for m in r["warnings"]) and list(r["iv"].resids.index[:2]) == [0, 1])
# r1-B N1: constant tolerance; N4 (f): AR with clustered covariance; (g) "Unadjusted"
rng = np.random.default_rng(261215)
n = 400
z, u, v, w = rng.normal(size=(4, n))
x = 0.5 * z + v + u
d = pd.DataFrame(dict(y=1 + x + u, x=x, z=z, w=1e8 + w, g=np.repeat(np.arange(40), 10)))
r, _, _ = quiet(dg.iv_report, "y ~ 0 + w + [x ~ z]", d)
check("0 + w with w = 1e8 + N(0, 1): no constant in the span → the warning (it disappeared with 1e-8)",
      any("no constant" in m for m in r["warnings"]), r["warnings"])
rc, tc, _ = quiet(dg.iv_report, "y ~ 1 + [x ~ z]", d, cov_type="kernel")
check("cov_type='kernel': the AR line says it uses HC1, not the kernel covariance; key cov = 'HC1'",
      "it uses HC1 (heteroskedasticity only), not the kernel covariance" in tc and rc["anderson_rubin"]["cov"] == "HC1", tc[-300:])
rr, tr, _ = quiet(dg.iv_report, "y ~ 1 + [x ~ z]", d, cov_type="robust")
check("cov_type='robust': no such note", "not the" not in tr.split("Anderson-Rubin")[-1] and rr["anderson_rubin"]["cov"] == "HC1")
ru1, _, _ = quiet(dg.iv_report, "y ~ 1 + [x ~ z]", d, cov_type="Unadjusted")
ru2, _, _ = quiet(dg.iv_report, "y ~ 1 + [x ~ z]", d, cov_type="unadjusted")
check("cov_type='Unadjusted' = 'unadjusted' (linearmodels 6.1 raised KeyError)",
      np.allclose(ru1["tests"]["statistic"], ru2["tests"]["statistic"], rtol=1e-12) and ru1["cov_type"] == "unadjusted")

# ============================================================ 13. review round 2 (r2-A and r2-B)
# r2-A B1: an exogenous x stored with an offset came out "endogenous" (F 238 against 0.11): x now enters the test
# regression net of W and on unit columns
rng = np.random.default_rng(12)
n = 400
z1, z2, w, u, v = rng.normal(size=(5, n))
x1 = 0.5 * z1 + 0.3 * z2 + v                          # exogenous
ro = []
for off in (0.0, 1e7, 1e9):
    d = pd.DataFrame(dict(y=x1 + u, x1=x1 + off, z1=z1, z2=z2, w=w))
    ro.append(quiet(dg.endogeneity_test, d.y, d[["w"]], d[["x1"]], d[["z1", "z2"]], cov_type="HC3")[0])
check("exogenous x stored as 1e7 + x and 1e9 + x: the same test as without the offset (relative 1e-6), p > 0.05",
      all(rel(r_["statistic"], ro[0]["statistic"]) < 1e-6 for r_ in ro[1:]) and ro[0]["p"] > 0.05,
      [r_["statistic"] for r_ in ro])
# r2-A S1: leverage from group dummies in W (FE with T = 2) does not trigger the high-leverage warning
rng = np.random.default_rng(5)
G, T_ = 100, 2
gid = np.repeat(np.arange(G), T_)
Wd = pd.get_dummies(gid, prefix="g", drop_first=True, dtype=float)
a_ = rng.normal(size=G)[gid]
z, u, v = rng.normal(size=(3, G * T_))
x = a_ + 0.5 * z + v + u
d = pd.DataFrame(dict(y=a_ + x + u, x=x, z=z))
r, _, _ = quiet(dg.endogeneity_test, d.y, Wd, d[["x"]], d[["z"]], cov_type="HC1")
check("FE with T = 2 (leverage 0.5 from W on every row), HC1: no high-leverage warning (measured size 3–6.5%)",
      not any("leverage" in m for m in r["warnings"]), r["warnings"])
# r2-A S2: a continuous outlier in W (1 − h of order 1e-5 to 1e-9) is not 'fitted exactly': kept, and the HC3 F equals
# statsmodels on the original columns
rng = np.random.default_rng(1)
n = 200
z, u, v, w = rng.normal(size=(4, n))
for big in (1e4, 1e6):
    w2 = w.copy(); w2[0] = big
    x = 0.5 * z + v + u
    d = pd.DataFrame(dict(y=x + u, x=x, z=z, w=w2))
    r, _, _ = quiet(dg.endogeneity_test, d.y, d[["w"]], d[["x"]], d[["z"]], cov_type="HC3")
    ref = sm.OLS(d.x, sm.add_constant(d[["w", "z"]])).fit(cov_type="HC3").wald_test(
        np.array([[0, 0, 1.0]]), scalar=True, use_f=True).statistic
    check(f"outlier {big:g} in an exogenous column: no 'fitted exactly' note, HC3 F = statsmodels (relative 1e-6)",
          not any("fitted exactly" in m for m in r["warnings"]) and rel(r["first_stage_F"]["x"], float(ref)) < 1e-6,
          (r["first_stage_F"], float(ref), r["warnings"]))
# r2-A N: the HC2/HC3 cut (1e-12) and the √eps condition cut, each with a case on each side
rng = np.random.default_rng(4)
n = 1000
z, w, u, v = rng.normal(size=(4, n))
for z6, refused in ((1e-5, False), (1e-7, True)):
    zb = np.zeros(n); zb[5] = 1.0; zb[6] = z6           # leverage of row 5 ≈ 1 − z6²
    x = 0.3 * z + v + u + 2 * zb
    d = pd.DataFrame(dict(y=x + u, x=x, z=z, zb=zb, w=w))
    r, _, _ = quiet(dg.endogeneity_test, d.y, d[["w"]], d[["x"]], d[["z", "zb"]], cov_type="HC3")
    check(f"HC3 with leverage 1 − {z6 ** 2:.0e}: {'refused (within 1e-12)' if refused else 'computed (≈ 25.8)'}",
          (np.isnan(r["first_stage_F"]["x"]) and r["high_leverage_rows"] == [5]) if refused else
          (abs(r["first_stage_F"]["x"] - 25.8) < 0.1 and r["high_leverage_rows"] == []), r)
rng = np.random.default_rng(11)
n = 400
z1, z2, z3, w, u, v = rng.normal(size=(6, n))
x1 = 0.5 * z1 + 0.3 * z2 + v + u
e = rng.normal(size=n)
d = pd.DataFrame(dict(y=2 * x1 + z1 + u, x1=x1, x2=x1 + z1 + 1e-4 * e, z1=z1, z2=z2, z3=z3, w=w))
r, _, _ = quiet(dg.endogeneity_test, d.y, d[["w"]], d[["x1", "x2"]], d[["z1", "z2", "z3"]], cov_type="HC3")
check("x2 = x1 + z1 + 1e-4·e (residuals collinear to 1e-4, above √eps): computed, not jointly degenerate",
      np.isfinite(r["statistic"]) and not any("jointly degenerate" in m for m in r["warnings"]), r)
# r2-B B1: clustered without clusters is refused
rng = np.random.default_rng(261216)
n = 200
z, u, v = rng.normal(size=(3, n))
x = 0.5 * z + v + u
d = pd.DataFrame(dict(y=1 + x + u, x=x, z=z))
try:
    dg.iv_report("y ~ 1 + [x ~ z]", d, cov_type="clustered", show=False)
    ok, m = False, "no error"
except ValueError as ex:
    m = str(ex); ok = "no clusters argument" in m
check("iv_report cov_type='clustered': ValueError (without clusters it was the robust SE under another name)", ok, m)
# r2-B S1: the constant-in-span check on scaled columns
g = np.repeat(np.arange(4), n // 4)
d2 = d.assign(g=g, w=1e7 * rng.normal(size=n))
r, _, _ = quiet(dg.iv_report, "y ~ 0 + C(g) + w + [x ~ z]", d2)
check("0 + C(g) + w with w at 1e7: no false 'no constant' warning", not any("no constant" in m for m in r["warnings"]),
      r["warnings"])
# r2-B S3: with repeated labels, result k is row k of the data, also after a NaN row
rng = np.random.default_rng(26126)
n = 150
z, w, u, v = rng.normal(size=(4, 2 * n))
x = 0.5 * z + v + u
dd = pd.DataFrame(dict(y=1 + x + 0.5 * w + u, x=x, z=z, w=w))
dup = pd.concat([dd.iloc[:n].reset_index(drop=True), dd.iloc[n:].reset_index(drop=True)])
dup.iloc[5, dup.columns.get_loc("z")] = np.nan
r, _, _ = quiet(dg.iv_report, "y ~ 1 + w + [x ~ z]", dup)
idx = r["iv"].resids.index
yfit = dup.y.to_numpy()[np.asarray(idx)]
check("repeated labels and a NaN at position 5: results indexed by position in the data (5 missing, 6 present), and "
      "y at those positions is the 2SLS y", 5 not in idx and 6 in idx and len(idx) == 299
      and np.allclose(yfit, r["iv"].model.dependent.ndarray.ravel()), list(idx[:8]))

# ============================================================ 14. review round 3
# r3 B1: jointly degenerate endogenous regressors stored with offsets (the rank rule with the offset floor; the
# condition check alone let them through and the verdict changed with the environment)
rng = np.random.default_rng(11)
n = 400
z1, z2, z3, w, u, v = rng.normal(size=(6, n))
x1 = 0.5 * z1 + 0.3 * z2 + v + u
e = rng.normal(size=n)
cases = [("x2 = x1 + 3w, w stored as 1e8 + w", dict(x1=x1, x2=x1 + 3 * (w + 1e8), w=w + 1e8, z1=z1), True),
         ("x1 at 1e-3, x2 = 1e6 + 2·x1", dict(x1=1e-3 * x1, x2=1e6 + 2e-3 * x1, w=w, z1=z1), True),
         ("x2 = x1 + z1, z1 stored as 1e9 + z1", dict(x1=x1, x2=x1 + z1 + 1e9, w=w, z1=z1 + 1e9), True),
         ("x2 = x1 + 3w + 0.3·e, w as 1e8 + w (the other side)", dict(x1=x1, x2=x1 + 3 * (w + 1e8) + 0.3 * e, w=w + 1e8,
                                                                     z1=z1), False)]
for lab, cols, deg in cases:
    d = pd.DataFrame(dict(y=2 * x1 + u, z2=z2, z3=z3, **cols))
    r, _, _ = quiet(dg.endogeneity_test, d.y, d[["w"]], d[["x1", "x2"]], d[["z1", "z2", "z3"]], cov_type="HC3")
    check(f"{lab}: {'jointly degenerate, test NaN' if deg else 'computed'}",
          (np.isnan(r["statistic"]) and any("jointly degenerate" in m for m in r["warnings"])) if deg else
          (np.isfinite(r["statistic"]) and not any("jointly degenerate" in m for m in r["warnings"])), r)
# r3 S1: an outlier in W with 1 − h_W = 2e-10 and HC1: the instruments do not reach that row, so no refusal; the F
# equals statsmodels on the original columns (no row dropped, the same n)
rng = np.random.default_rng(1)
n = 200
z, u, v, w = rng.normal(size=(4, n))
w2 = w.copy(); w2[0] = 1e6
x = 0.5 * z + v + u
d = pd.DataFrame(dict(y=x + u, x=x, z=z, w=w2))
r, _, _ = quiet(dg.endogeneity_test, d.y, d[["w"]], d[["x"]], d[["z"]], cov_type="HC1")
ref = sm.OLS(d.x, sm.add_constant(d[["w", "z"]])).fit(cov_type="HC1").wald_test(
    np.array([[0, 0, 1.0]]), scalar=True, use_f=True).statistic
check("outlier 1e6 in W (1 − h_W ≈ 2e-10), HC1: computed, not refused, = statsmodels (relative 1e-6)",
      r["high_leverage_rows"] == [] and rel(r["first_stage_F"]["x"], float(ref)) < 1e-6 and np.isfinite(r["statistic"]),
      (r, float(ref)))

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
