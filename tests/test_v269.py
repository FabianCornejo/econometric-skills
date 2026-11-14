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
lg, _ = quiet(dg.iv_report, "y ~ 1 + [np.log(xl+10) ~ z1 + z2]", dl)
li, _ = quiet(dg.iv_report, "y ~ 1 + [I(xl*1) ~ z1 + z2]", dl)
check("transformed endogenous terms (np.log(xl+10), I(xl*1)): one endogenous column each, overidentified by 1, "
      "and the Anderson-Rubin set computed for the model's column",
      lg["endogenous"] == ["np.log(xl + 10)"] and lg["overidentified_by"] == 1 and len(lg["tests"]) == 2
      and lg["anderson_rubin"] is not None and lg["anderson_rubin"]["estimand"] == "np.log(xl + 10)"
      and li["endogenous"] == ["I(xl * 1)"] and li["anderson_rubin"]["estimand"] == "I(xl * 1)",
      (lg["endogenous"], lg["overidentified_by"], li["endogenous"]))

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
      len(rc["warnings"]) == 1 and "2 instrument columns, but only 1 add information" in rc["warnings"][0]
      and "Warning: 2 instrument columns, but only 1" in tc, rc["warnings"])
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
      r0["n_instruments"] == 2 and len(r0["warnings"]) == 1 and "3 instrument columns, but only 2" in r0["warnings"][0]
      and close(r0["first_stage_F"]["x"], rb["first_stage_F"]["x"], 1e-8), (r0["n_instruments"], r0["warnings"]))
rng = np.random.default_rng(26930)
dww = di.assign(w=rng.normal(size=400))
dww["w2"] = 2 * dww["w"]
rw, _ = endo(dww, ["z1", "z2"], exog=["w", "w2"])
check("collinear exogenous regressors (w, 2w): the instruments still count 2, no warning",
      rw["n_instruments"] == 2 and rw["warnings"] == [], (rw["n_instruments"], rw["warnings"]))
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
