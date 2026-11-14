"""Tests for v2.6.9: iv_report counts instruments and endogenous regressors on the model's columns, not on the
formula's terms, and endogeneity_test counts instruments by rank.

Origin: the v2.6.8 review. Round 1, reviewer B: with [x ~ z1*z2] (three instrument columns: z1, z2, z1:z2) the
robust partial F was divided by 1 instead of 3, so it came out three times too large, and the overidentification
test was skipped (overidentified_by 0 instead of 2). A weak instrument could pass the F > 10 rule. Round 2, NIT 4:
endogeneity_test counted len(Z.columns), so z and 2z made the model "overidentified" and the text offered LIML.
Plan: claude/plano-v2.6.9.md in the development repository. Reproduction and search scripts are in
claude/scripts/ of that repository (repro_iv_contagem_v269.py, busca_endog_colinear_v269.py).

The references do not reuse the implementation's arithmetic. The robust partial F is checked against the
statsmodels HC0 Wald F. The unadjusted partial F is checked against an F from the two sums of squared residuals,
with sigma² = SSR/n, the convention linearmodels uses for its Wald statistic. The Sargan statistic is n·R² of the
2SLS residuals on all instrument columns, computed by hand. A metamorphic check compares [x ~ z1*z2] with the same
three columns written out as [x ~ z1 + z2 + z12]. Every boundary has a case on each side, and every section has its
own generator.

Run from the repository root:  python3 tests/test_v269.py
"""
from __future__ import annotations

import contextlib
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "econometrics"
sys.path.insert(0, str(SKILL / "scripts"))
import _console  # noqa: E402,F401
import diagnostics as dg  # noqa: E402

warnings.filterwarnings("ignore")
ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info)


def close(a, b, tol):
    return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))


def quiet(f, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = f(*a, **k)
    return res, buf.getvalue()


# ------------------------------------------------- 1. iv_report: [x ~ z1*z2] is the same model as z1 + z2 + z12
# The design of the review's reproduction (n = 400, x = .08 z1 + .08 z2 + v): weak, and overidentified by 2.
rng = np.random.default_rng(269)
n = 400
z1, z2, u, v = rng.normal(size=(4, n))
x = 0.08 * z1 + 0.08 * z2 + v + 0.5 * u
dw = pd.DataFrame(dict(y=1 + 0.5 * x + u, x=x, z1=z1, z2=z2, z12=z1 * z2))
for cov in ("robust", "unadjusted"):
    a, ta = quiet(dg.iv_report, "y ~ 1 + [x ~ z1*z2]", dw, cov_type=cov)
    b, tb = quiet(dg.iv_report, "y ~ 1 + [x ~ z1 + z2 + z12]", dw, cov_type=cov)
    fa, fb = float(a["first_stage"]["partial F"].iloc[0]), float(b["first_stage"]["partial F"].iloc[0])
    check(f"{cov}: z1*z2 and z1 + z2 + z12 give the same partial F and partial R² (1e-10)",
          close(fa, fb, 1e-10) and close(a["first_stage"]["partial R²"].iloc[0], b["first_stage"]["partial R²"].iloc[0],
                                         1e-10), (fa, fb))
    if cov == "robust":   # unadjusted: linearmodels already reports an F, which the old count left alone
        check("robust: the partial F of z1*z2 is not three times the right one (the old count of 1 instrument)",
              not close(fa, 3 * fb, 1e-3), (fa, fb))
    check(f"{cov}: both overidentified by 2, with the same tests and statistics (1e-10)",
          a["overidentified_by"] == 2 and b["overidentified_by"] == 2 and len(a["tests"]) == 2
          and np.allclose(a["tests"][["statistic", "p"]].to_numpy(), b["tests"][["statistic", "p"]].to_numpy(),
                          rtol=1e-10, atol=0) and "2 df" in a["tests"].index[1], a["tests"])
    pa, pb = a["anderson_rubin"]["pieces"], b["anderson_rubin"]["pieces"]
    check(f"{cov}: the same Anderson-Rubin set and shape",
          a["anderson_rubin"]["shape"] == b["anderson_rubin"]["shape"] and len(pa) == len(pb)
          and np.allclose(np.array(pa, dtype=float), np.array(pb, dtype=float), rtol=1e-10, atol=0, equal_nan=True),
          (pa, pb))
check("the 'instruments' key lists the model's columns (z1, z2, z1:z2), not the formula's term",
      a["instruments"] == ["z1", "z2", "z1:z2"] and a["endogenous"] == ["x"], a["instruments"])
check("the printed report (unadjusted, the loop's last pass) names the three columns and the 2-df Sargan test",
      "instruments: z1, z2, z1:z2" in ta and "overidentification (Sargan, 2 df)" in ta
      and "Exactly identified" not in ta, ta[:400])

# ----------------------------------------------------- 2. iv_report: the partial F against references by other paths
rng = np.random.default_rng(2691)
z1, z2, u, v = rng.normal(size=(4, n))
x = 0.2 * z1 + 0.2 * z2 + 0.15 * z1 * z2 + v + 0.5 * u
dr = pd.DataFrame(dict(y=1 + 0.5 * x + u, x=x, z1=z1, z2=z2, z12=z1 * z2))
Zf = sm.add_constant(dr[["z1", "z2", "z12"]])
rob, _ = quiet(dg.iv_report, "y ~ 1 + [x ~ z1*z2]", dr)
f_hc0 = float(sm.OLS(dr.x, Zf).fit(cov_type="HC0").wald_test(np.eye(4)[1:], scalar=True, use_f=True).statistic)
check("robust partial F with z1*z2 = statsmodels HC0 Wald F on the 3 columns (1e-8)",
      close(rob["first_stage"]["partial F"].iloc[0], f_hc0, 1e-8), (rob["first_stage"]["partial F"].iloc[0], f_hc0))
una, _ = quiet(dg.iv_report, "y ~ 1 + [x ~ z1*z2]", dr, cov_type="unadjusted")
fu, fr = sm.OLS(dr.x, Zf).fit(), sm.OLS(dr.x, np.ones(n)).fit()
f_ssr = ((fr.ssr - fu.ssr) / 3) / (fu.ssr / n)            # Wald with sigma² = SSR/n, divided by q = 3
check("unadjusted partial F with z1*z2 = F from the two SSRs with q = 3 (sigma² = SSR/n, linearmodels' Wald; 1e-8)",
      close(una["first_stage"]["partial F"].iloc[0], f_ssr, 1e-8), (una["first_stage"]["partial F"].iloc[0], f_ssr))
aux = sm.OLS(np.asarray(una["iv"].resids), Zf).fit()
sargan, p_sargan = n * aux.rsquared, stats.chi2.sf(n * aux.rsquared, 2)
ov = una["tests"].iloc[1]
check("Sargan with z1*z2 = n·R² of the 2SLS residuals on the 3 columns, χ²(2), by hand (1e-8)",
      close(ov["statistic"], sargan, 1e-8) and close(ov["p"], p_sargan, 1e-8), (ov["statistic"], sargan))

# ------------------------------------------------------ 3. iv_report: the boundary of exact identification, both sides
wa = quiet(dg.iv_report, "y ~ 1 + [x ~ z1*z2]", dw)[0]
wz = [m for m in wa["warnings"] if m.startswith("weak instrument")]
check("z1*z2, weak: overidentified, so 'Anderson-Rubin set (or LIML)', and the one-instrument note on 104.7",
      len(wz) == 1 and "(or LIML)" in wz[0] and "exactly identified" not in wz[0]
      and wa["first_stage"]["strength"].iloc[0].endswith("[104.7 is the one-instrument threshold of Lee et al.]"),
      (wz, wa["first_stage"]["strength"].iloc[0]))
e1, t1 = quiet(dg.iv_report, "y ~ 1 + [x ~ z1]", dw)
w1 = [m for m in e1["warnings"] if m.startswith("weak instrument")]
check("z1 alone, weak: exactly identified (other side): LIML named only to say it equals 2SLS, no 104.7 note, "
      "no overidentification test",
      len(w1) == 1 and "exactly identified, it equals 2SLS" in w1[0] and "(or LIML)" not in w1[0]
      and "one-instrument threshold" not in e1["first_stage"]["strength"].iloc[0]
      and e1["overidentified_by"] == 0 and len(e1["tests"]) == 1 and "Exactly identified" in t1, (w1, t1[-300:]))

# ---------------------------------------------------- 4. iv_report: endogenous regressors counted on the columns too
rng = np.random.default_rng(2692)
z1, z2, u, v, wv = rng.normal(size=(5, n))
xx = 0.4 * z1 + 0.3 * z2 + v + 0.5 * u
ww_ = 0.3 * z1 - 0.4 * z1 * z2 + wv + 0.3 * u
de = pd.DataFrame(dict(y=1 + 0.5 * xx + 0.3 * ww_ + u, x=xx, w=ww_, z1=z1, z2=z2, z12=z1 * z2))
two, _ = quiet(dg.iv_report, "y ~ 1 + [x + w ~ z1 + z2 + z12]", de)
check("two endogenous, three instruments: overidentified by 1, overidentification test run, no Anderson-Rubin set",
      two["endogenous"] == ["x", "w"] and two["overidentified_by"] == 1 and len(two["tests"]) == 2
      and "1 df" in two["tests"].index[1] and two["anderson_rubin"] is None, two["tests"])
inter, ti = quiet(dg.iv_report, "y ~ 1 + [x*w ~ z1*z2]", de)
check("[x*w ~ z1*z2]: three endogenous columns (x, w, x:w) and three instruments, exactly identified, no AR set "
      "(the old count saw one endogenous term, 'x*w', and stopped with KeyError: 'x*w')",
      inter["endogenous"] == ["x", "w", "x:w"] and inter["overidentified_by"] == 0 and len(inter["tests"]) == 1
      and inter["anderson_rubin"] is None and "Exactly identified" in ti and "Anderson-Rubin 95%" not in ti,
      (inter["endogenous"], inter["overidentified_by"]))

Zt = sm.add_constant(de[["z1", "z2", "z12"]])
f_two = [float(sm.OLS(de[c], Zt).fit(cov_type="HC0").wald_test(np.eye(4)[1:], scalar=True, use_f=True).statistic)
         for c in ("x", "w")]
check("two endogenous, three instruments: each partial F = statsmodels HC0 Wald F on the 3 columns (1e-8)",
      all(close(a_, b_, 1e-8) for a_, b_ in zip(two["first_stage"]["partial F"], f_two)),
      (two["first_stage"]["partial F"].tolist(), f_two))
check("[x*w ~ z1*z2]: the header names the three endogenous columns, and the 104.7 note is there (3 instruments)",
      "endogenous: x, w, x:w;" in ti
      and inter["first_stage"]["strength"].str.endswith("[104.7 is the one-instrument threshold of Lee et al.]").all(),
      inter["first_stage"]["strength"].tolist())
# exact identification with two endogenous regressors, both sides (weak design; seed 2693)
rng = np.random.default_rng(2693)
z1, z2, u, v, wv = rng.normal(size=(5, n))
x = 0.05 * z1 + 0.04 * z2 + v + 0.5 * u
w = 0.04 * z1 - 0.05 * z2 + 0.05 * z1 * z2 + wv + 0.3 * u
d2w = pd.DataFrame(dict(y=1 + 0.5 * x + 0.3 * w + u, x=x, w=w, z1=z1, z2=z2, z12=z1 * z2))
ex2, _ = quiet(dg.iv_report, "y ~ 1 + [x + w ~ z1 + z2]", d2w)
ov2, _ = quiet(dg.iv_report, "y ~ 1 + [x + w ~ z1 + z2 + z12]", d2w)
we = [m for m in ex2["warnings"] if m.startswith("weak instrument")]
wo = [m for m in ov2["warnings"] if m.startswith("weak instrument")]
check("two endogenous, two instruments, weak: exactly identified, 'LIML does not help', not '(or LIML)'",
      len(we) == 1 and "exactly identified, it equals 2SLS" in we[0] and "(or LIML)" not in we[0]
      and ex2["overidentified_by"] == 0, we)
check("two endogenous, three instruments, weak: '(or LIML)' (other side)",
      len(wo) == 1 and "(or LIML)" in wo[0] and "exactly identified" not in wo[0] and ov2["overidentified_by"] == 1, wo)

# transformed endogenous terms are named by the model with spaces. On main, the '+' inside np.log(xl+10) split it
# into two "endogenous" names (exactly identified, no AR set, no overidentification test), and I(xl*1) stopped with
# KeyError: 'I(xl*1)' (the column is 'I(xl * 1)')
dl = de.assign(xl=de.x + 10.0)
lg, tlg = quiet(dg.iv_report, "y ~ 1 + [np.log(xl+10) ~ z1 + z2]", dl)
li, _ = quiet(dg.iv_report, "y ~ 1 + [I(xl*1) ~ z1 + z2]", dl)
check("transformed endogenous terms (np.log(xl+10), I(xl*1)): one endogenous column each, overidentified by 1, "
      "and the Anderson-Rubin set computed for the model's column",
      lg["endogenous"] == ["np.log(xl + 10)"] and lg["overidentified_by"] == 1 and len(lg["tests"]) == 2
      and lg["anderson_rubin"] is not None and lg["anderson_rubin"]["estimand"] == "np.log(xl + 10)"
      and li["endogenous"] == ["I(xl * 1)"] and li["anderson_rubin"]["estimand"] == "I(xl * 1)",
      (lg["endogenous"], lg["overidentified_by"], li["endogenous"]))

check("overidentified by 1: the 1-df test is printed and 'Exactly identified' is not (boundary q = 0 / q = 1)",
      "1 df" in tlg and "Exactly identified" not in tlg, tlg[-400:])
# instruments with a '+' inside a term or repeated (round 2): on main I(z1+z2) became two "instruments" ('I(z1',
# 'z2)'), so the F was halved (3.75 against 7.51) and an overidentification test with NaN ran; z1 + z1 likewise
rng = np.random.default_rng(5)
z1, z2, u, v = rng.normal(size=(4, n))
x = 0.1 * z1 + 0.1 * np.log(z2 + 10) + v + 0.5 * u
dp = pd.DataFrame(dict(y=1 + 0.5 * x + u, x=x, z1=z1, z2=z2, zs=z1 + z2))
ip, _ = quiet(dg.iv_report, "y ~ 1 + [x ~ I(z1+z2)]", dp)
iz, _ = quiet(dg.iv_report, "y ~ 1 + [x ~ zs]", dp)
i2, _ = quiet(dg.iv_report, "y ~ 1 + [x ~ z1 + z1]", dp)
i1, _ = quiet(dg.iv_report, "y ~ 1 + [x ~ z1]", dp)
check("[x ~ I(z1+z2)] = [x ~ zs] with zs = z1 + z2: one instrument, exactly identified, the same F (1e-10)",
      ip["instruments"] == ["I(z1 + z2)"] and ip["overidentified_by"] == 0 and len(ip["tests"]) == 1
      and close(ip["first_stage"]["partial F"].iloc[0], iz["first_stage"]["partial F"].iloc[0], 1e-10),
      (ip["instruments"], ip["first_stage"]["partial F"].iloc[0], iz["first_stage"]["partial F"].iloc[0]))
check("[x ~ z1 + z1] = [x ~ z1]: one instrument, the same F (1e-10)",
      i2["instruments"] == ["z1"] and i2["overidentified_by"] == 0
      and close(i2["first_stage"]["partial F"].iloc[0], i1["first_stage"]["partial F"].iloc[0], 1e-10),
      (i2["instruments"], i2["first_stage"]["partial F"].iloc[0]))
# the 104.7 note with exactly 2 instruments (boundary q_instr = 1 / 2; round 2)
n2, _ = quiet(dg.iv_report, "y ~ 1 + [x ~ z1 + z2]", dp)
check("two instruments: the one-instrument note on 104.7 is there (one instrument: absent, checked above)",
      n2["first_stage"]["strength"].iloc[0].endswith("[104.7 is the one-instrument threshold of Lee et al.]"),
      n2["first_stage"]["strength"].iloc[0])

# ---------------------------------------------------- 5. endogeneity_test: instruments counted by rank
# Seed 2701: the first design found by claude/scripts/busca_endog_colinear_v269.py with p < .05 and F <= 10.
rng = np.random.default_rng(2701)
nn = 1000
z, u, v = rng.normal(size=(3, nn))
x = 0.08 * z + v + u
dc = pd.DataFrame(dict(y=1 + 0.5 * x + u, x=x, z=z, z2=2 * z))


def endo(d, zs, exog=None):
    return quiet(dg.endogeneity_test, d["y"], None if exog is None else d[exog], d[["x"]], d[zs])


rc, tc = endo(dc, ["z", "z2"])
rs, ts = endo(dc, ["z"])
check("z and 2z: counted as 1 instrument; exactly identified text (LIML does not help), not 'or LIML'",
      rc["n_instruments"] == 1 and rc["p"] < 0.05 and rc["first_stage_F"]["x"] <= 10
      and "LIML does not help here" in tc and "or LIML" not in tc, tc)
check("z and 2z: warned, in the dict and on screen, that only 1 of the 2 columns adds information",
      len(rc["warnings"]) == 1 and "2 instrument column(s), but only 1 add information" in rc["warnings"][0]
      and "Warning: 2 instrument column(s), but only 1" in tc, rc["warnings"])
check("z and 2z: the same F and p as z alone (the statistics were already right; only the count was wrong)",
      close(rc["statistic"], rs["statistic"], 1e-8) and close(rc["p"], rs["p"], 1e-8)
      and close(rc["first_stage_F"]["x"], rs["first_stage_F"]["x"], 1e-6) and rs["warnings"] == []
      and rs["n_instruments"] == 1, (rc, rs))
# other side: two independent instruments (the overidentified design of test_v268, seed 0)
rng = np.random.default_rng(0)
Z = rng.normal(size=(400, 2))
u = rng.normal(size=400)
v = 0.8 * u + 0.6 * rng.normal(size=400)
x = Z @ np.full(2, 0.12) + v
di = pd.DataFrame(dict(y=1 + 0.5 * x + u, x=x, z1=Z[:, 0], z2=Z[:, 1]))
ri, tii = endo(di, ["z1", "z2"])
check("two independent instruments: counted as 2, no warning, 'or LIML' (other side)",
      ri["n_instruments"] == 2 and ri["warnings"] == [] and "(Anderson-Rubin) or LIML." in tii
      and "LIML does not help" not in tii, tii)
# the rank does not depend on units: 1e6·z1 and 1e-6·z2 are still two instruments
dsc = di.assign(z1=di.z1 * 1e6, z2=di.z2 * 1e-6)
rsc, _ = endo(dsc, ["z1", "z2"])
check("rescaled instruments (1e6 and 1e-6): still 2, no warning, same statistic (1e-6)",
      rsc["n_instruments"] == 2 and rsc["warnings"] == [] and close(rsc["statistic"], ri["statistic"], 1e-6),
      (rsc["n_instruments"], rsc["warnings"]))
# an instrument that is a multiple of an exogenous regressor adds nothing
dx = dc.assign(w=rng.normal(size=nn))
dx["z3"] = 3 * dx["w"]
rx, tx = endo(dx, ["z", "z3"], exog=["w"])
check("an instrument equal to 3 × an exogenous regressor is not counted (1 of 2), and it is said",
      rx["n_instruments"] == 1 and len(rx["warnings"]) == 1 and "only 1 add information" in rx["warnings"][0]
      and "or LIML" not in tx, (rx["n_instruments"], rx["warnings"], tx))
check("z, 3w with w exogenous: the first-stage F is the F of z alone (1e-8), not the F of the redundant columns",
      close(rx["first_stage_F"]["x"], endo(dx, ["z"], exog=["w"])[0]["first_stage_F"]["x"], 1e-8),
      (rx["first_stage_F"], endo(dx, ["z"], exog=["w"])[0]["first_stage_F"]))
# the review's design (round 1, reviewer B, seed 5): z weak, w strong in the first stage; with z3 = 3w the F of the
# redundant columns was 196, "strong", against 4.4 with z alone
rng = np.random.default_rng(5)
z, u, v, w = rng.normal(size=(4, nn))
x = 0.11 * z + 0.8 * w + v + u
d5 = pd.DataFrame(dict(y=1 + 0.5 * x + u, x=x, z=z, w=w, z3=3 * w))
r5, t5 = endo(d5, ["z", "z3"], exog=["w"])
r5z, _ = endo(d5, ["z"], exog=["w"])
check("z, 3w (review's design): F = F of z alone (1e-8), labelled WEAK, and no 'strong' nor 'use IV/2SLS'",
      close(r5["first_stage_F"]["x"], r5z["first_stage_F"]["x"], 1e-8) and r5["first_stage_F"]["x"] < 10
      and "→ WEAK" in t5 and "strong" not in t5 and "use IV/2SLS" not in t5,
      (r5["first_stage_F"], r5z["first_stage_F"], t5))
# no excluded instrument left: 'not identified'
rn, tn = endo(dx, ["z3"], exog=["w"])
check("the only instrument is 3 × an exogenous regressor: 0 instruments, 'not identified', F and test NaN, no "
      "first-stage line",
      rn["n_instruments"] == 0 and "not identified: 0 excluded instrument(s) for 1 endogenous" in tn
      and np.isnan(rn["first_stage_F"]["x"]) and np.isnan(rn["statistic"]) and np.isnan(rn["p"])
      and "First stage for" not in tn and "no instrument, so there is no first stage" in rn["warnings"][0], tn)
# units and degenerate columns (review, round 1, reviewer B: mutants of the rank that survived)
rb, _ = endo(di, ["z1", "z2"])
for sc in (1e-10, 1e-170):
    rr, _ = endo(di.assign(z1=di.z1 * sc), ["z1", "z2"])
    check(f"an instrument scaled by {sc:g}: still 2, no warning, the same F and test (1e-8)",
          rr["n_instruments"] == 2 and rr["warnings"] == [] and close(rr["first_stage_F"]["x"], rb["first_stage_F"]["x"],
                                                                       1e-8)
          and close(rr["statistic"], rb["statistic"], 1e-8), (rr["n_instruments"], rr["first_stage_F"]))
r0, _ = endo(di.assign(z0=0.0), ["z1", "z2", "z0"])
check("a column of zeros among the instruments: counted 2 of 3, warned, the same F (1e-8)",
      r0["n_instruments"] == 2 and len(r0["warnings"]) == 1 and "3 instrument column(s), but only 2" in r0["warnings"][0]
      and close(r0["first_stage_F"]["x"], rb["first_stage_F"]["x"], 1e-8), (r0["n_instruments"], r0["warnings"]))
rng = np.random.default_rng(26930)
dww = di.assign(w=rng.normal(size=400))
dww["w2"] = 2 * dww["w"]
rw, _ = endo(dww, ["z1", "z2"], exog=["w", "w2"])
check("collinear exogenous regressors (w, 2w): the instruments still count 2, no warning",
      rw["n_instruments"] == 2 and rw["warnings"] == [], (rw["n_instruments"], rw["warnings"]))
# copies of two highly correlated exogenous regressors among the instruments, and z orthogonal to them in the sample
# (a residualized instrument): the strongest directions of Z lie in the span of W, so the F must be taken on Z net of
# W (taking the top singular directions of Z itself gave 65.8 against 25.8)
rng = np.random.default_rng(26933)
z, u, v, w = rng.normal(size=(4, 400))
w2 = w + 0.1 * rng.normal(size=400)
WW = np.column_stack([np.ones(400), w, w2])
z = z - WW @ np.linalg.lstsq(WW, z, rcond=None)[0]
x = 0.3 * z + 0.5 * w + v + u
dzw = pd.DataFrame(dict(y=1 + 0.5 * x + u, x=x, z=z, w=w, w2=w2, wz=w, w2z=w2))
rzw, _ = endo(dzw, ["wz", "w2z", "z"], exog=["w", "w2"])
rzo, _ = endo(dzw, ["z"], exog=["w", "w2"])
check("copies of correlated exogenous regressors among the instruments, z orthogonal to them: 1 instrument and the "
      "F of z alone (1e-8)",
      rzw["n_instruments"] == 1 and close(rzw["first_stage_F"]["x"], rzo["first_stage_F"]["x"], 1e-8),
      (rzw["first_stage_F"], rzo["first_stage_F"]))
# nearly collinear but distinct columns (2·z1 + 1e-8 noise) count as 2; the F tests both directions and equals the
# classical F from the two sums of squared residuals (nonrobust), computed here with lstsq on the raw columns
rng = np.random.default_rng(26931)
dn = di.assign(z2e=2 * di.z1 + 1e-8 * rng.normal(size=400))
rnn, _ = quiet(dg.endogeneity_test, dn.y, None, dn[["x"]], dn[["z1", "z2e"]], cov_type="nonrobust")
Xn = np.column_stack([np.ones(400), dn.z1, dn.z2e])
ssr_u = float(np.sum((dn.x - Xn @ np.linalg.lstsq(Xn, dn.x, rcond=None)[0]) ** 2))
ssr_r = float(np.sum((dn.x - dn.x.mean()) ** 2))
f_cl = ((ssr_r - ssr_u) / 2) / (ssr_u / 397)
check("z1 and 2·z1 + 1e-8 noise: 2 instruments, and the nonrobust F = the classical F from the two SSRs (1e-6)",
      rnn["n_instruments"] == 2 and rnn["warnings"] == [] and close(rnn["first_stage_F"]["x"], f_cl, 1e-6),
      (rnn["n_instruments"], rnn["first_stage_F"], f_cl))
# two endogenous regressors, z and 2z: 1 instrument for 2, not identified (the old count said 2 and went on)
d2x = di.assign(x2=di.x + np.random.default_rng(26932).normal(size=400), zz=2 * di.z1)
r2x, t2x = quiet(dg.endogeneity_test, d2x.y, None, d2x[["x", "x2"]], d2x[["z1", "zz"]])
check("two endogenous regressors with z and 2z: 1 instrument, 'not identified: 1 excluded instrument(s) for 2'",
      r2x["n_instruments"] == 1 and "not identified: 1 excluded instrument(s) for 2 endogenous" in t2x, t2x)

# --------------------------------------- 5b. endogeneity_test: references by other paths and degenerate designs
# (review, round 2, reviewer D)
from linearmodels.iv import IV2SLS  # noqa: E402

rng = np.random.default_rng(26940)
nr = 300
z1, z2, w, u, v = rng.normal(size=(5, nr))
x = 0.25 * z1 + 0.15 * z2 + 0.4 * w + v + 0.6 * u
dref = pd.DataFrame(dict(y=1 + 0.5 * x + 0.3 * w + u * (1 + 0.5 * np.abs(w)), x=x, z1=z1, z2=z2, w=w))
Xo = sm.add_constant(dref[["w", "z1", "z2"]])
for cov in ("HC3", "HC0", "nonrobust"):
    rr_, _ = endo(dref.assign(), ["z1", "z2"], exog=["w"]) if cov == "HC3" else \
        quiet(dg.endogeneity_test, dref.y, dref[["w"]], dref[["x"]], dref[["z1", "z2"]], cov_type=cov)
    f_ref = float(sm.OLS(dref.x, Xo).fit(cov_type=cov).f_test(np.eye(4)[2:]).fvalue)
    check(f"{cov}: first-stage F = statsmodels F test on the original columns (1e-8)",
          close(rr_["first_stage_F"]["x"], f_ref, 1e-8), (rr_["first_stage_F"], f_ref))
ivm = IV2SLS.from_formula("y ~ 1 + w + [x ~ z1 + z2]", dref)
h0, _ = quiet(dg.endogeneity_test, dref.y, dref[["w"]], dref[["x"]], dref[["z1", "z2"]], cov_type="HC0")
hn, _ = quiet(dg.endogeneity_test, dref.y, dref[["w"]], dref[["x"]], dref[["z1", "z2"]], cov_type="nonrobust")
check("HC0: the Durbin-Wu-Hausman F = linearmodels' robust Wooldridge regression test (1e-8)",
      close(h0["statistic"], ivm.fit(cov_type="robust").wooldridge_regression.stat, 1e-8),
      (h0["statistic"], ivm.fit(cov_type="robust").wooldridge_regression.stat))
vh = sm.OLS(dref.x, Xo).fit().resid                              # control function by hand
Xr = sm.add_constant(dref[["w", "x"]])
ssr_r, fu_ = sm.OLS(dref.y, Xr).fit().ssr, sm.OLS(dref.y, Xr.assign(vh=vh)).fit()
f_dwh = (ssr_r - fu_.ssr) / (fu_.ssr / fu_.df_resid)
check("nonrobust: the Durbin-Wu-Hausman F = the F from the two SSRs with v̂ added by hand (1e-8)",
      close(hn["statistic"], f_dwh, 1e-8), (hn["statistic"], f_dwh))
# fewer instruments than endogenous regressors: not identified, test NaN, no first-stage line; the other side
# (as many) computes the test
dxx = dref.assign(x2=0.3 * dref.z1 - 0.2 * dref.w + rng.normal(size=nr) + 0.4 * u)
ru1, tu1 = quiet(dg.endogeneity_test, dxx.y, dxx[["w"]], dxx[["x", "x2"]], dxx[["z1"]])
ru2, _ = quiet(dg.endogeneity_test, dxx.y, dxx[["w"]], dxx[["x", "x2"]], dxx[["z1", "z2"]])
check("2 endogenous, 1 instrument: 'not identified', statistic and p NaN, no first-stage line",
      ru1["n_instruments"] == 1 and np.isnan(ru1["statistic"]) and np.isnan(ru1["p"])
      and "not identified: 1 excluded instrument(s) for 2" in tu1 and "First stage for" not in tu1, tu1)
check("2 endogenous, 2 instruments (other side): the test is computed",
      ru2["n_instruments"] == 2 and np.isfinite(ru2["statistic"]) and np.isfinite(ru2["p"]), ru2)
# saturated first stage: 10 rows, 1 exogenous regressor, 8 instruments (HC0 gave F = 1e31, "strong")
rng = np.random.default_rng(26941)
ds = pd.DataFrame(rng.normal(size=(10, 11)), columns=["y", "x", "w"] + [f"s{k}" for k in range(8)])
rsat, tsat = quiet(dg.endogeneity_test, ds.y, ds[["w"]], ds[["x"]], ds[[f"s{k}" for k in range(8)]], cov_type="HC0")
check("saturated first stage (n = 10, 2 + 8 columns): F and test NaN, a warning, 'not computed', no 'strong'",
      np.isnan(rsat["first_stage_F"]["x"]) and np.isnan(rsat["statistic"]) and "saturated" in " ".join(rsat["warnings"])
      and "not computed" in tsat and "strong" not in tsat, tsat)
# the other side of the boundary: 11 rows leave 1 residual degree of freedom, and the F and test are computed
rng = np.random.default_rng(26941)
ds11 = pd.DataFrame(rng.normal(size=(11, 11)), columns=["y", "x", "w"] + [f"s{k}" for k in range(8)])
r11, _ = quiet(dg.endogeneity_test, ds11.y, ds11[["w"]], ds11[["x"]], ds11[[f"s{k}" for k in range(8)]],
               cov_type="nonrobust")
check("11 rows, 2 + 8 columns (1 residual degree of freedom): F and test computed, no saturation warning",
      np.isfinite(r11["first_stage_F"]["x"]) and np.isfinite(r11["statistic"])
      and not any("saturated" in m for m in r11["warnings"]), r11)
# the test regression [W, x, v̂] saturates first: 2 endogenous, 2 instruments, 2 exogenous directions; 6 rows leave
# the first stage 2 degrees of freedom but the test none (the clause n − r_W − 2·n_x ≤ 0); 7 rows compute both
for nrow, sat_ in ((6, True), (7, False)):
    rng = np.random.default_rng(26943)
    dd6 = pd.DataFrame(rng.normal(size=(nrow, 6)), columns=["y", "x1", "x2", "w", "s1", "s2"])
    r6, t6 = quiet(dg.endogeneity_test, dd6.y, dd6[["w"]], dd6[["x1", "x2"]], dd6[["s1", "s2"]], cov_type="nonrobust")
    if sat_:
        check("6 rows, 2 endogenous, 2 instruments, 2 exogenous directions: test NaN, 'the test regression is "
              "saturated', the first-stage F still computed and printed",
              np.isnan(r6["statistic"]) and any("the test regression is saturated" in m for m in r6["warnings"])
              and all(np.isfinite(list(r6["first_stage_F"].values()))) and "First stage for x1" in t6
              and "not computed" in t6, (r6, t6))
    else:
        check("7 rows (other side): the test is computed, no saturation warning",
              np.isfinite(r6["statistic"]) and not any("saturated" in m for m in r6["warnings"]), r6)
# the magnitude of the rank tolerance, both sides: z and z + 1e-10·e are two instruments (distinct beyond rounding);
# z and z + 1e5 are one (the offset's rounding is not an instrument; it used to count 2)
rng = np.random.default_rng(26944)
z, w, u, v, ee = rng.normal(size=(5, 400))
x = 0.3 * z + 0.3 * w + v + u
dtl = pd.DataFrame(dict(y=x + u, x=x, z=z, w=w, zn=z + 1e-10 * ee, zo=z + 1e5, wo=w + 1e5))
rt1, _ = endo(dtl, ["z", "zn"], exog=["w"])
rt2, _ = endo(dtl, ["z", "zo"], exog=["w"])
rt3, _ = endo(dtl, ["z", "wo"], exog=["w"])
rt0, _ = endo(dtl, ["z"], exog=["w"])
check("z and z + 1e-10·e: 2 instruments; z and z + 1e5, and z and w + 1e5: 1, with the F of z alone (1e-8)",
      rt1["n_instruments"] == 2 and rt2["n_instruments"] == 1 and rt3["n_instruments"] == 1
      and close(rt2["first_stage_F"]["x"], rt0["first_stage_F"]["x"], 1e-8)
      and close(rt3["first_stage_F"]["x"], rt0["first_stage_F"]["x"], 1e-8),
      (rt1["n_instruments"], rt2["n_instruments"], rt3["n_instruments"]))
# an exogenous regressor with a large offset stays in W (W is centred too): the same F and test as without it
dwo = dtl.assign(wbig=dtl.w + 1e13)
rwo, _ = endo(dwo, ["z"], exog=["wbig"])
check("an exogenous regressor with offset 1e13: the same F and test as with w (1e-3; the offset costs digits)",
      close(rwo["first_stage_F"]["x"], rt0["first_stage_F"]["x"], 1e-3) and close(rwo["statistic"], rt0["statistic"], 1e-3),
      (rwo["first_stage_F"], rt0["first_stage_F"], rwo["statistic"], rt0["statistic"]))
# fewer instruments than endogenous regressors: the warning about a redundant column does not promise a first stage
dun = dxx.assign(z1b=2 * dxx.z1)
run_, tun = quiet(dg.endogeneity_test, dun.y, dun[["w"]], dun[["x", "x2"]], dun[["z1", "z1b"]])
check("not identified with a redundant column: the warning does not say 'the first-stage F tests only those'",
      run_["n_instruments"] == 1 and "tests only those" not in " ".join(run_["warnings"]) and "First stage" not in tun,
      run_["warnings"])

# near-collinear exogenous regressors (cond ~ 1e12): the instrument still counts (rank tolerances of W and [W, Z]
# used to disagree: 'not identified' for a strong z), and the F and test equal those of the same span written
# well conditioned: a + 2e-13·e1 and b + 2e-13·e2 span what a, e1, b, e2 span
rng = np.random.default_rng(26942)
a_, b_, zz, u, v = rng.normal(size=(5, 200))
e1, e2 = rng.normal(size=(2, 200))
x = 0.5 * zz + v + u
dnc = pd.DataFrame(dict(y=x + u, x=x, z=zz, a=a_, a2=a_ + 2e-13 * e1, b=b_, b2=b_ + 2e-13 * e2, e1=e1, e2=e2))
rnc, _ = quiet(dg.endogeneity_test, dnc.y, dnc[["a", "a2", "b", "b2"]], dnc[["x"]], dnc[["z"]], cov_type="HC1")
rn2, _ = quiet(dg.endogeneity_test, dnc.y, dnc[["a", "e1", "b", "e2"]], dnc[["x"]], dnc[["z"]], cov_type="HC1")
# (with cond(W) ~ 1e12 about four digits survive: eps·cond ~ 2e-4)
check("near-collinear exogenous regressors: 1 instrument, and the F and test of the same span well conditioned (1e-3)",
      rnc["n_instruments"] == 1 and close(rnc["first_stage_F"]["x"], rn2["first_stage_F"]["x"], 1e-3)
      and close(rnc["statistic"], rn2["statistic"], 1e-3), (rnc["first_stage_F"], rn2["first_stage_F"],
                                                             rnc["statistic"], rn2["statistic"]))
# an instrument with a large offset (1e14 + z): centring keeps it (without centring it was counted 0, 'not
# identified'); 1e14 + z keeps z to about 0.016, so the F agrees with the F of z1 to about 1e-3
dof = dref.assign(zo=dref.z1 + 1e14)
rof, _ = endo(dof.assign(), ["zo"], exog=["w"])
rz1, _ = endo(dref.assign(), ["z1"], exog=["w"])
check("an instrument with offset 1e14: still 1 instrument, F close to the F of z1 (5e-3; the offset costs digits)",
      rof["n_instruments"] == 1 and close(rof["first_stage_F"]["x"], rz1["first_stage_F"]["x"], 5e-3),
      (rof["first_stage_F"], rz1["first_stage_F"]))
# the review's grid (round 2, reviewer D, seed 8): two pairs of near-collinear exogenous regressors, eps from 1e-14
# to 1e-12; separate rank tolerances for W and [W, Z] counted 0 instruments at eps = 2.04e-13 ('not identified' for a
# strong z). Every point of the grid must count 1.
rng = np.random.default_rng(8)
a_, b_ = rng.normal(size=200), rng.normal(size=200)
cnt = []
for eps_ in np.geomspace(1e-14, 1e-12, 30):
    Wg = pd.DataFrame(dict(w0=a_, w1=a_ + eps_ * rng.normal(size=200), w2=b_, w3=b_ + eps_ * rng.normal(size=200)))
    zg = pd.DataFrame(dict(z0=rng.normal(size=200) * 50))
    u = rng.normal(size=200)
    x = 0.02 * zg.z0 + u + rng.normal(size=200)
    dg_ = pd.concat([pd.DataFrame(dict(y=x + u, x=x)), Wg, zg], axis=1)
    cnt.append(dg.endogeneity_test(dg_.y, Wg, dg_[["x"]], zg, cov_type="HC1", show=False)["n_instruments"])
check("near-collinear exogenous pairs, 30 values of eps (seed 8): the strong instrument counts 1 at every point",
      cnt == [1] * 30, cnt)
try:
    dg.endogeneity_test(dref.y, None, dref[["x"]], dref[["z1"]].assign(z1=dref.z1.where(dref.index != 3, np.inf)),
                        show=False)
    inf_ok = False
except ValueError as e:
    inf_ok = "inf" in str(e)
check("inf in the data: a clear ValueError, not an SVD failure", inf_ok)

# ------------------------------------------------------------------ 6. CHANGELOG and version (found by heading)
import re  # noqa: E402

import reporting  # noqa: E402

log = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
m = re.search(r"^## v2\.6\.9 — .*?(?=^## v)", log, re.M | re.S)
e269 = m.group(0) if m else ""
check("CHANGELOG has a v2.6.9 entry, above v2.6.8, naming both fixes and the consistency test",
      bool(m) and log.index("## v2.6.9") < log.index("## v2.6.8")
      and all(t in e269 for t in ("`[x ~ z1*z2]`", "7.96 instead of 2.65", "`[np.log(x+10) ~ z1 + z2]`", "counted by rank",
                                  "`n_instruments`", "196, \"strong\", against 4.4", "tests/test_release_consistency.py")), e269[:300])
ver = tuple(int(p_) for p_ in reporting.SKILL_VERSION.split("."))
check("SKILL_VERSION is 2.6.9 or later (not pinned as the last version)", ver >= (2, 6, 9), reporting.SKILL_VERSION)

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
