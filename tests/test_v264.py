"""Tests for v2.6.4: reporting rules of the DML bounds and the numerical backlog of v2.5.3.

Origin: the v2.6.3 validation on notebook 1 (the memo gave the DML bounds without their rho and called both the bias
bound of the linear DML and the confidence bounds of the forest "interval") and the backlog of v2.5.3 (q = 1e300 printed
"nan%"; df_resid = 1 printed "nan%"; a zero bias bound of dml_sensitivity missing from out["warnings"] with show=False;
cf_d >= 1 accepted with an infinite bound; "nor does any" decided at the ends of the search). A probe of v2.6.4
(claude/scripts/sonda_v264.py in the development repository) also found rv_q = 0.0, instead of almost 1, for
q·f >~ 1e8: the paper's formula lost every digit by cancellation.

What is checked, each by a route other than the implementation's:
  1. RV_q and RV_q,alpha against references computed in mpmath with 1300 digits from the paper's own formulas
     (claude/scripts/referencia_rv_v264.py; values copied from its output), x = q·f from 1e-300 to 1e300, with
     dof = 100 and dof = 2, to 4 ulp; monotone in q; the old formula agrees where it was accurate (x <= 1); on both
     sides of where it failed (1e7 and 1e9; 1e153 and 1e155); sensitivity prints no "nan" with q = 1e10 and 1e300;
  2. dof: 2 accepted, 1.999, 1, 0, NaN and inf refused (ValueError naming the df); t NaN or inf refused; sensitivity
     on a fit with df_resid = 1 refused and on one with df_resid = 2 printed without "nan";
  3. dml_sensitivity with show=False: a zero bias bound (cf_y = 0, cf_d = 0, rho = 0) gives one warning in
     out["warnings"] and through warnings.warn; a tiny positive one (cf_y = 1e-300) none; the default none; with
     show=True the zero sentence is printed once, as before;
  4. cf_y and cf_d: 0 and 1 accepted for both; nextafter(1, 0) gives a finite bound and no warning; cf_d = 1 an
     infinite bound, a warning and a printed sentence; nextafter(1, 2), -5e-324, NaN and inf refused (ValueError);
     True, np.True_, "0.1" and None refused (TypeError); np.float32 and int accepted;
  5. "nor does any" and the root of RVa: in one sample split gap is convex in c (second differences on a grid of c);
     the root against a closed form (the quadratic identity of the confidence bound's SE, solved for c and then for
     eta^2); the endpoint verdict and the root against a dense grid on designs with 1, 3 and 5 splits and small rho;
  6. the reporting line of dml_sensitivity and dml_bounds (rho stated, the bias bound and the confidence bounds
     named), absent with show=False and when dml_bounds computes no line; the rule of v2.6.4 in
     interpretation-rules.md and the evaluation case review-memo-dml-rho-and-interval.

Run from the repository root:  python3 tests/test_v264.py
Needs: statsmodels, scipy, scikit-learn.
"""
import contextlib
import io
import json
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path.insert(0, str(ROOT / "econometrics" / "scripts"))
import _console  # noqa: E402,F401  (Windows-safe console output, see econometrics/scripts/_console.py)
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.linear_model import LinearRegression

import causal
import specification as sp

ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


def run(fn, *a, **k):
    """(result, stdout, warning messages); an exception is returned as the result."""
    buf = io.StringIO()
    with warnings.catch_warnings(record=True) as w, contextlib.redirect_stdout(buf):
        warnings.simplefilter("always")
        try:
            r = fn(*a, **k)
        except Exception as e:  # noqa: BLE001
            r = e
    return r, buf.getvalue(), [str(x.message) for x in w]


def ulps(a, b):
    return abs(a - b) / np.spacing(max(abs(a), abs(b), np.finfo(float).tiny))


# ------------------------------------------------------------------ 1. RV_q and RV_q,alpha (mpmath references)
# claude/scripts/referencia_rv_v264.py, 1300 digits, the paper's formulas (not the skill's rewriting)
RV_REF = {1e-300: 1e-300, 1e-160: 1e-160, 1e-08: 9.999999950000001e-09, 0.001: 0.0009995001249999923,
          0.1: 0.09512492197250394, 0.5: 0.3903882032022076, 1.0: 0.6180339887498949, 2.0: 0.8284271247461901,
          10.0: 0.9901951359278482, 1e4: 0.9999999900000002, 1e7: 0.99999999999999, 1e8: 0.9999999999999999,
          1e9: 1.0, 1e20: 1.0, 1e153: 1.0, 1e155: 1.0, 1e200: 1.0, 1e300: 1.0}
RVA_REF_100 = {1e-300: 0.0, 1e-160: 0.0, 1e-08: 0.0, 0.001: 0.0, 0.1: 0.0, 0.5: 0.2587805315246982,
               1.0: 0.5418725762151908, 2.0: 0.8017376210846258, 10.0: 0.9897052588347872, 1e4: 0.9999999896023115,
               1e7: 0.9999999999999896, 1e8: 0.9999999999999999, 1e9: 1.0, 1e20: 1.0, 1e153: 1.0, 1e155: 1.0,
               1e200: 1.0, 1e300: 1.0}
RVA_REF_2 = {1e-300: 0.0, 1e-160: 0.0, 1e-08: 0.0, 0.001: 0.0, 0.1: 0.0, 0.5: 0.0, 1.0: 0.0, 2.0: 0.0, 10.0: 0.0,
             1e4: 0.9999983755236282, 1e7: 0.9999999999983755, 1e8: 0.9999999999999838, 1e9: 0.9999999999999999,
             1e20: 1.0, 1e153: 1.0, 1e155: 1.0, 1e200: 1.0, 1e300: 1.0}
# f = |t|/sqrt(dof) = 1 exactly, so x = q: t = 10 with dof = 100, t = sqrt(2) with dof = 2
for dof, t_, ref_a in ((100.0, 10.0, RVA_REF_100), (2.0, float(np.sqrt(2.0)), RVA_REF_2)):
    worst_q, worst_a, bad = 0.0, 0.0, []
    for x, ref in RV_REF.items():
        o = sp.robustness_value(t_, dof, q=x)
        eq, ea = ulps(o["rv_q"], ref), ulps(o["rv_qa"], ref_a[x])
        worst_q, worst_a = max(worst_q, eq), max(worst_a, ea)
        if not (eq <= 4 and ea <= 4 and np.isfinite(o["rv_q"]) and np.isfinite(o["rv_qa"])):
            bad.append((x, o["rv_q"], ref, o["rv_qa"], ref_a[x]))
    check(f"dof = {dof:g}: rv_q and rv_qa within 4 ulp of the mpmath references for x = 1e-300 to 1e300",
          not bad, f"worst {worst_q:.1f} / {worst_a:.1f} ulp; {bad[:3]}")

o, _, w_ov = run(sp.robustness_value, 1e10, 100.0, q=1e300)     # f = 1e9, so q·f overflows to inf
check("q·f overflowing to inf (t = 1e10, dof = 100, q = 1e300): rv_q = rv_qa = 1, no nan, no RuntimeWarning",
      o["rv_q"] == 1.0 and o["rv_qa"] == 1.0 and w_ov == [], (o["rv_q"], o["rv_qa"], w_ov))
for t_, dof_, q_ in ((1e10, 100.0, 1e299), (1.7e308, 2.0, 1.0)):   # review 2: x finite above ~9e307 gave nan
    o, _, w_x = run(sp.robustness_value, t_, dof_, q=q_)
    check(f"x = q·f finite near the largest float (t = {t_:g}, dof = {dof_:g}, q = {q_:g}): rv_q = rv_qa = partial_r2 "
          "= 1, no warning", isinstance(o, dict) and o["rv_q"] == 1.0 and o["rv_qa"] == 1.0 and o["partial_r2"] == 1.0
          and w_x == [], (o if not isinstance(o, dict) else (o["rv_q"], o["rv_qa"], o["partial_r2"]), w_x))
o, _, w_t = run(sp.robustness_value, 1e155, 100.0)
check("t = 1e155 (t² overflows): partial_r2 = 1, no OverflowError, no warning",
      isinstance(o, dict) and o["partial_r2"] == 1.0 and w_t == [], repr(o) if not isinstance(o, dict) else w_t)
pr = [sp.robustness_value(t_, 50.0)["partial_r2"] for t_ in (0.0, 0.3, 2.0, 7.0, 40.0)]
check("partial_r2 = t²/(t² + dof) within 4 ulp on usual t (dof = 50)",
      all(ulps(p_, t_ ** 2 / (t_ ** 2 + 50.0)) <= 4 for p_, t_ in zip(pr, (0.0, 0.3, 2.0, 7.0, 40.0))), pr)
qs = np.geomspace(1e-6, 1e300, 400)
rvq = np.array([sp.robustness_value(10.0, 100.0, q=float(q))["rv_q"] for q in qs])
rva_ = np.array([sp.robustness_value(10.0, 100.0, q=float(q))["rv_qa"] for q in qs])
check("rv_q and rv_qa are non-decreasing in q, finite, in [0, 1], and reach 1 (q from 1e-6 to 1e300)",
      np.all(np.diff(rvq) >= 0) and np.all(np.diff(rva_) >= 0) and np.all(np.isfinite(rvq)) and rvq[-1] == 1
      and np.all((rvq >= 0) & (rvq <= 1)) and np.all((rva_ >= 0) & (rva_ <= 1)), f"min diff {np.diff(rvq).min():.2e}")


def old_rv(x):                                    # the formula of v2.6.3 (the paper's, in floating point)
    x = np.float64(x)                             # as in the code (fq = q·f is a numpy float): overflow gives inf
    return 0.5 * (np.sqrt(x ** 4 + 4 * x ** 2) - x ** 2)


xs = np.geomspace(1e-6, 1.0, 300)
d_old = max(ulps(sp.robustness_value(10.0, 100.0, q=float(x))["rv_q"], old_rv(x)) for x in xs)
check("where the old formula was accurate (x <= 1), the new one agrees with it within 4 ulp (no number moves)",
      d_old <= 4, f"max {d_old:.1f} ulp")
# where the old formula failed, both sides of each region: cancellation from x²·eps > 4 (between 1e7 and 1e9);
# x⁴ overflows beyond ~1.3e77 (inf), x² beyond ~1.3e154 (inf − inf = nan)
with np.errstate(all="ignore"):
    olds = {x: old_rv(x) for x in (1e7, 1e9, 1e77, 1e78, 1e154, 1e155)}
    neg = old_rv(825742055.3289121)               # review 2: the old form also went negative (a value near 1)
check("old formula at x = 825742055.3289121: -64.0, not near 1 (the erratic region, negative values included)",
      neg == -64.0, neg)
check("old formula: 1e7 still close, 1e9 already 0 (cancellation), 1e77 finite and 1e78 inf (x⁴ overflow), 1e154 inf "
      "and 1e155 nan (x² overflow): the regions the new tests cover",
      abs(olds[1e7] - RV_REF[1e7]) < 1e-6 and olds[1e9] == 0.0 and np.isfinite(olds[1e77]) and olds[1e78] == np.inf
      and olds[1e154] == np.inf and np.isnan(olds[1e155]), str(olds))
newv = {x: sp.robustness_value(10.0, 100.0, q=x)["rv_q"] for x in (1e7, 1e9, 1e77, 1e78, 1e154, 1e155)}
check("new formula on both sides of each region: within 4 ulp of the references (1 beyond 1e9, where the reference "
      "rounds to 1)", all(ulps(v, RV_REF.get(x, 1.0)) <= 4 for x, v in newv.items()), str(newv))

rng = np.random.default_rng(264)
n = 200
D0 = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
D0["y"] = 0.3 * D0.x + 0.5 * D0.z + rng.normal(size=n)
M0 = smf.ols("y ~ x + z", D0).fit()
for q in (1e10, 1e300):
    o, txt, w = run(sp.sensitivity, M0, "x", q=q, show=True)
    check(f"sensitivity q = {q:g}: RV printed 100.00% (not 'nan%' nor 0.00%), no overflow warning",
          isinstance(o, dict) and "nan" not in txt and txt.count("= 100.00%") == 2
          and not any("overflow" in s for s in w), txt.splitlines()[1] if txt else repr(o))

# ------------------------------------------------------------------ 2. degrees of freedom and t
o = sp.robustness_value(3.0, 2.0)
check("dof = 2 accepted, finite (rv_qa = 0: t = 3 is below t_crit(1) = 12.7)",
      np.isfinite(o["rv_q"]) and o["rv_qa"] == 0.0, str(o))
for dof in (1.999, 1.0, 0.0, -3.0, np.nan, np.inf):
    o, _, _ = run(sp.robustness_value, 3.0, dof)
    check(f"dof = {dof!r} refused with a ValueError naming the degrees of freedom",
          isinstance(o, ValueError) and "2 residual degrees of freedom" in str(o), repr(o))
for t_ in (np.nan, np.inf, -np.inf):
    o, _, _ = run(sp.robustness_value, t_, 100)
    check(f"t = {t_!r} refused (ValueError)", isinstance(o, ValueError) and "t must be finite" in str(o), repr(o))
d3 = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.5], "z": [1.0, 0.0, 2.0, 1.0], "y": [0.1, 1.2, 1.9, 3.8]})
o, txt, _ = run(sp.sensitivity, smf.ols("y ~ x + z", d3).fit(), "x", show=True)
check("sensitivity with df_resid = 1: ValueError 'got dof = 1', nothing printed (was 'nan%')",
      isinstance(o, ValueError) and "got dof = 1" in str(o) and txt == "", repr(o))
d4 = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.5, 4.0], "z": [1.0, 0.0, 2.0, 1.0, 3.0], "y": [0.1, 1.2, 1.9, 3.8, 4.1]})
o, txt, _ = run(sp.sensitivity, smf.ols("y ~ x + z", d4).fit(), "x", show=True)
check("sensitivity with df_resid = 2: runs and prints no 'nan'", isinstance(o, dict) and "nan" not in txt,
      txt.splitlines()[1] if txt else repr(o))

# ------------------------------------------------------------------ DML designs
LIN = dict(learner_y=LinearRegression(), learner_d=LinearRegression())
rng = np.random.default_rng(7)
n = 600
X = rng.normal(size=(n, 3))
Dv = X[:, 0] + 0.5 * X[:, 1] + rng.normal(size=n)
Yv = 0.5 * Dv + X[:, 0] + 0.8 * X[:, 1] + rng.normal(size=n)
DD = pd.DataFrame(X, columns=["x1", "x2", "x3"]).assign(d=Dv, y=Yv)
RES = causal.dml_plr(DD, "y", "d", ["x1", "x2", "x3"], n_folds=5, n_rep=3, seed=3, **LIN)

# ------------------------------------------------------------------ 3. zero bias bound with show=False
for kw in ({"cf_y": 0.0}, {"cf_d": 0.0}, {"rho": 0.0}):
    o, txt, w = run(causal.dml_sensitivity, RES, show=False, **kw)
    zw = [s for s in o["warnings"] if "bias bound is zero" in s]
    check(f"show=False, {kw}: bias bound 0, one warning in out['warnings'] and one through warnings.warn, no stdout",
          o["bias_bound"] == 0 and len(zw) == 1 and len(o["warnings"]) == 1
          and sum("bias bound is zero" in s for s in w) == 1 and txt == "", (o["warnings"], w))
o, txt, w = run(causal.dml_sensitivity, RES, cf_y=1e-300, show=False)
check("show=False, cf_y = 1e-300: bias bound > 0, no warning (the other side of the zero)",
      o["bias_bound"] > 0 and o["warnings"] == [] and not any("bias bound" in s for s in w), o["bias_bound"])
o, txt, w = run(causal.dml_sensitivity, RES, rho=1e-200, show=False)
check("show=False, rho = 1e-200: a bias bound of order 1e-200, still > 0, gives no warning (both sides of exactly 0)",
      0 < o["bias_bound"] < 1e-190 and o["warnings"] == [], o["bias_bound"])
o, txt, w = run(causal.dml_sensitivity, RES, show=False)
check("show=False, defaults: no warning", o["warnings"] == [] and w == [], (o["warnings"], w))
with warnings.catch_warnings(record=True) as wr:
    warnings.simplefilter("always")
    causal.dml_sensitivity(RES, cf_y=0.0, show=False)
check("the warning points at the caller (stacklevel: this file, not causal.py)",
      len(wr) == 1 and Path(wr[0].filename).resolve() == Path(__file__).resolve(), [x.filename for x in wr])
o, txt, w = run(causal.dml_sensitivity, RES, cf_y=0.0, show=True)
check("show=True, cf_y = 0: the zero sentence printed once (as before), not repeated as a WARNING line",
      txt.count("The bias bound is zero") == 1 and "WARNING" not in txt, txt.splitlines()[2])

# ------------------------------------------------------------------ 4. cf_y and cf_d
below1, above1 = float(np.nextafter(1.0, 0.0)), float(np.nextafter(1.0, 2.0))
o, txt, w = run(causal.dml_sensitivity, RES, cf_d=below1, show=True)
check("cf_d = nextafter(1, 0): finite bound, no warning, no 'infinite' sentence",
      isinstance(o, dict) and np.isfinite(o["bias_bound"]) and o["warnings"] == [] and "infinite" not in txt,
      repr(o) if not isinstance(o, dict) else o["bias_bound"])
o, txt, w = run(causal.dml_sensitivity, RES, cf_d=1.0, show=True)
check("cf_d = 1: infinite bound, one warning (out and warnings.warn) and the sentence printed",
      isinstance(o, dict) and o["bias_bound"] == np.inf and len(o["warnings"]) == 1
      and "infinite" in o["warnings"][0] and any("infinite" in s for s in w)
      and "cf_d = 100% leaves no residual variation in the treatment" in txt, o["warnings"] if isinstance(o, dict) else o)
o, txt, w = run(causal.dml_sensitivity, RES, cf_d=1.0, show=False)
check("cf_d = 1 with show=False: the warning is in out['warnings'], nothing printed",
      len(o["warnings"]) == 1 and txt == "", o["warnings"])
for kw in ({"rho": 0.0}, {"cf_y": 0.0}):
    o, txt, w = run(causal.dml_sensitivity, RES, cf_d=1.0, show=True, **kw)
    check(f"cf_d = 1 with {kw}: the bound is its limit, zero (was inf): the zero warning only, no 'infinite' text",
          o["bias_bound"] == 0 and len(o["warnings"]) == 1 and "bias bound is zero" in o["warnings"][0]
          and "infinite" not in txt and "bias <= inf" not in txt and txt.count("The bias bound is zero") == 1,
          (o["bias_bound"], o["warnings"]))
o, _, _ = run(causal.dml_sensitivity, RES, cf_y=1.0, show=False)
check("cf_y = 1 accepted (finite bound, no warning)", isinstance(o, dict) and np.isfinite(o["bias_bound"])
      and o["warnings"] == [], repr(o) if not isinstance(o, dict) else o["bias_bound"])
for name in ("cf_y", "cf_d"):
    o, _, _ = run(causal.dml_sensitivity, RES, show=False, **{name: 0})
    check(f"{name} = 0 (int) accepted", isinstance(o, dict), "" if isinstance(o, dict) else repr(o))
    o, _, _ = run(causal.dml_sensitivity, RES, show=False, **{name: np.float32(0.03)})
    check(f"{name} = np.float32(0.03) accepted, same bound as 0.03 up to float32 rounding",
          isinstance(o, dict) and np.isfinite(o["bias_bound"]), repr(o) if not isinstance(o, dict) else "")
    for v in (above1, -5e-324, np.nan, np.inf, -np.inf):
        o, _, _ = run(causal.dml_sensitivity, RES, show=False, **{name: v})
        check(f"{name} = {v!r} refused (ValueError, 'sensitivity parameter ... [0, 1]')",
              isinstance(o, ValueError) and "sensitivity parameter" in str(o) and "[0, 1]" in str(o)
              and name in str(o), repr(o))
    for v in (True, np.True_, "0.1", None):
        o, _, _ = run(causal.dml_sensitivity, RES, show=False, **{name: v})
        check(f"{name} = {v!r} refused (TypeError)", isinstance(o, TypeError) and name in str(o), repr(o))

# ------------------------------------------------------------------ 5. "nor does any" and the root of RVa
z = stats.norm.ppf(0.975)


def handmade(seed, n_rep, n=80, th=0.6, scale_sd=1.5, mix=2.0):
    """A dml_plr-like result with residuals built by hand: heavy tails, splits of very different scales."""
    r = np.random.default_rng(seed)
    resid = []
    for _ in range(n_rep):
        sc = np.exp(r.normal(0, scale_sd))
        vd = r.standard_t(3, n) * sc
        eps = r.standard_t(3, n) * np.exp(r.normal(0, 1)) + mix * r.normal(0, 1) * vd
        resid.append((th * vd + eps, vd))
    ths = [np.sum(v * u) / np.sum(v * v) for u, v in resid]
    th_ = float(np.median(ths))
    ses = [causal._dml_sens_rep(u, v, 0, 0, 1)["se_theta"] for u, v in resid]
    se_ = float(np.median(np.sqrt(np.array(ses) ** 2 + (np.array(ths) - th_) ** 2)))
    return {"estimate": th_, "se": se_, "p": 0.0, "n": n, "residuals": resid, "thetas": ths}


def gap_on(res, rho, etas, q=1.0):
    th = res["estimate"]
    S = float(np.median([causal._dml_sens_rep(u, v, 0, 0, rho)["S"] for u, v in res["residuals"]]))
    out = []
    for e in etas:
        e2 = e * e
        c = abs(rho) * e2 / np.sqrt(1 - e2)
        rr = [causal._dml_sens_rep(u, v, e2, e2, rho) for u, v in res["residuals"]]
        out.append(c * S + z * np.median([x["se_lo"] if th > 0 else x["se_hi"] for x in rr]) - q * abs(th))
    return np.array(out)


# (a) one split: gap is convex in c (the argument in the code), on a grid of c
r1 = handmade(11, 1)
cs = np.linspace(0, 5, 401)
th1 = r1["estimate"]
u1, v1 = r1["residuals"][0]
S1 = causal._dml_sens_rep(u1, v1, 0, 0, 1)["S"]
etas_c = np.sqrt((-cs ** 2 + np.sqrt(cs ** 4 + 4 * cs ** 2)) / 2)          # eta² with c = eta²/sqrt(1 - eta²)
gc = np.array([c * S1 + z * causal._dml_sens_rep(u1, v1, e * e, e * e, 1.0)["se_lo" if th1 > 0 else "se_hi"]
               for c, e in zip(cs, etas_c)])
sec = np.diff(gc, 2)
check("one split: gap is convex in c (second differences >= -1e-12 relative)",
      np.all(sec >= -1e-12 * np.max(np.abs(gc))), f"min second difference {sec.min():.2e}")

# (b) one split: the root of RVa against a closed form. n·SE(c)² = A − 2cB + c²V (quadratic identity), so
# gap(c) = 0 with |th| − cS >= 0 solves (|th| − cS)² = z²(A − 2cB + c²V)/n; then eta² from c (|rho| = 1 or rho).
for rho in (1.0, 0.3):
    side = "se_lo" if th1 > 0 else "se_hi"
    nn = len(u1)

    def se2(c):                                   # n·SE² at a strength whose c is known
        u_ = (-(c / abs(rho)) ** 2 + np.sqrt((c / abs(rho)) ** 4 + 4 * (c / abs(rho)) ** 2)) / 2
        return nn * causal._dml_sens_rep(u1, v1, u_, u_, rho)[side] ** 2
    A = se2(0.0)
    s1, s2 = se2(1.0 * abs(rho)), se2(2.0 * abs(rho))
    V = (s2 - 2 * s1 + A) / (2 * abs(rho) ** 2)
    B = -(s1 - A - V * abs(rho) ** 2) / (2 * abs(rho))
    a_ = S1 ** 2 - z ** 2 * V / nn
    b_ = -2 * abs(th1) * S1 + 2 * z ** 2 * B / nn
    c_ = th1 ** 2 - z ** 2 * A / nn
    roots = np.roots([a_, b_, c_])
    roots = [float(r.real) for r in roots if abs(r.imag) < 1e-12 and r.real >= 0 and abs(th1) - r.real * S1 >= 0]
    c_star = min(roots)
    k_ = c_star / abs(rho)
    rva_closed = (-k_ ** 2 + np.sqrt(k_ ** 4 + 4 * k_ ** 2)) / 2
    o, _, _ = run(causal.dml_sensitivity, r1, rho=rho, show=False)
    check(f"one split, rho = {rho}: RVa equals the closed form (quadratic identity), rel. error < 1e-8",
          0 < o["rva"] < 1 and abs(o["rva"] - rva_closed) < 1e-8 * rva_closed, f"{o['rva']!r} vs {rva_closed!r}")

# (c) several splits: endpoint verdict and root against a dense grid
ETAS = np.unique(np.concatenate([np.linspace(1e-6, 1 - 1e-6, 600), 1 - np.geomspace(1e-9, 1e-3, 120),
                                 np.geomspace(1e-9, 1e-3, 60)]))
seen, bad = set(), []
for seed, n_rep in ((1, 1), (2, 3), (3, 3), (4, 5), (5, 5), (6, 3)):
    res_ = handmade(seed, n_rep)
    for rho in (1.0, 1e-3, 1e-5):
        o, _, _ = run(causal.dml_sensitivity, res_, rho=rho, show=False)
        g = gap_on(res_, rho, ETAS)
        neg = g < 0
        if not neg[0]:
            gv = 0.0
        elif neg.all():
            gv = 1.0
        else:
            gv = float(ETAS[np.flatnonzero(~neg)[0]] ** 2)
        cls = (lambda v: "zero" if v == 0 else "ceiling" if v >= 1 else "interior")
        seen.add(cls(gv))
        one_change = int(np.sum(neg[1:] != neg[:-1])) <= 1
        if not (one_change and cls(o["rva"]) == cls(gv)
                and (cls(gv) != "interior" or abs(np.sqrt(o["rva"]) - np.sqrt(gv)) < 3e-3)):
            bad.append((seed, n_rep, rho, o["rva"], gv, one_change))
check("1, 3 and 5 splits, rho 1, 1e-3, 1e-5: one sign change at most, and the verdict and the root of RVa agree with "
      "a dense grid", not bad, str(bad[:3]))
check("the grid designs reach both an interior root and the ceiling ('nor does any')",
      {"interior", "ceiling"} <= seen, str(seen))

# ------------------------------------------------------------------ 6. reporting line, rule and case
o, txt, _ = run(causal.dml_sensitivity, RES, rho=1.0, show=True)
last = txt.strip().splitlines()[-1]
check("dml_sensitivity rho = 1.0: last line states 'rho = 1.0 (the worst case)' (as line 2 prints it) and names both "
      "intervals: the bias bound as theta plus or minus bias_bound, not accounting for sampling error",
      "In a report, state rho = 1.0 (the worst case) next to the bounds" in last and "rho = 1.0:" in txt
      and "'bounds [a, b]' is the bias bound (theta plus or minus its bias_bound), not accounting for sampling error"
      in last and "confidence bounds add sampling error and are the ones that exclude or include zero" in last, last)
o, txt, _ = run(causal.dml_sensitivity, RES, rho=0.4, show=True)
last = txt.strip().splitlines()[-1]
check("dml_sensitivity rho = 0.4: 'rho = 0.4 next to the bounds', no 'worst case'",
      "state rho = 0.4 next to the bounds" in last and "worst case" not in last, last)
o, txt, _ = run(causal.dml_sensitivity, RES, rho=-1.0, show=True)
check("dml_sensitivity rho = -1: '(the worst case)' too", "rho = -1.0 (the worst case)" in txt.strip().splitlines()[-1])
o, txt, _ = run(causal.dml_sensitivity, RES, rho=0.123456789, show=True)
check("dml_sensitivity: the reporting line gives rho as line 2 prints it (0.123456789 in both, not 0.123457)",
      "rho = 0.123456789:" in txt and "state rho = 0.123456789 next to" in txt.strip().splitlines()[-1],
      txt.strip().splitlines()[-1])
o, txt, _ = run(causal.dml_sensitivity, RES, q=0.5, show=True)
tgt = f"{0.5 * o['estimate']:.4f}"
check(f"dml_sensitivity q = 0.5: the reporting line judges the confidence bounds against the target {tgt}, not zero",
      f"say whether the target value {tgt} is inside them" in txt.strip().splitlines()[-1]
      and "include zero" not in txt.strip().splitlines()[-1] and f"theta to {tgt}" in txt, txt.strip().splitlines()[-1])
o, txt, _ = run(causal.dml_sensitivity, RES, show=False)
check("dml_sensitivity show=False: nothing printed", txt == "")
for v in (True, np.True_, "1", None):
    o, _, _ = run(causal.dml_sensitivity, RES, rho=v, show=False)
    check(f"dml_sensitivity rho = {v!r} refused (TypeError; a bool printed 'rho = True')",
          isinstance(o, TypeError) and "rho" in str(o), repr(o))

BENCH, _, _ = run(causal.dml_benchmark, DD, "y", "d", ["x1", "x2", "x3"], ["x1", "x2"], res=RES, n_folds=5, n_rep=3,
                  seed=3, **LIN)
for rho, want in (("benchmark", "state each line's rho next to it (the benchmark's observed correlation"),
                  (1.0, "state rho = 1 (the worst case) next to the bounds"),
                  (0.4, "state rho = 0.4 next to the bounds")):
    o, txt, w = run(causal.dml_bounds, RES, BENCH, rho=rho, show=True)   # default kd: x1 (gain_d > 1) skipped, warned
    lines = txt.strip().splitlines()
    rep = [i for i, s in enumerate(lines) if s.startswith("  In a report")]
    warn_i = [i for i, s in enumerate(lines) if s.startswith("  WARNING")]
    good = (isinstance(o, dict) and len(o["bounds"]) == 3 and len(warn_i) >= 1 and len(rep) == 1 and want in lines[rep[0]] and "'theta between a and b' is the bias bound" in lines[rep[0]]
            and lines[rep[0] - 1].startswith("  Worst cases") and all(i > rep[0] for i in warn_i)
            and (("(the worst case)" in lines[rep[0]]) == (rho == 1.0)))
    check(f"dml_bounds rho = {rho!r}: one reporting line, after the 'Worst cases' line and before any WARNING, with "
          "the rho and 'theta between a and b' named as the bias bound ('worst case' only for rho = 1)",
          good, lines[rep[0]] if rep else txt)
o, txt, _ = run(causal.dml_bounds, RES, BENCH, show=False)
check("dml_bounds show=False: nothing printed", txt == "")
empty = pd.DataFrame({"gain_y": [0.0], "gain_d": [0.1], "rho": [0.5]}, index=["z"])
o, txt, _ = run(causal.dml_bounds, RES, empty, show=True)
check("dml_bounds with no line computed: no reporting line (nothing to report)",
      isinstance(o, dict) and o["bounds"].empty and "In a report" not in txt and "No bound computed" in txt)

docs = (ROOT / "econometrics" / "references" / "interpretation-rules.md").read_text(encoding="utf-8")
sec = docs.split("## Rules added in v2.6.4")[1] if "## Rules added in v2.6.4" in docs else ""
check("interpretation-rules.md: 'Rules added in v2.6.4' after v2.6.3, with ρ in the document, the bias bound and the "
      "confidence bounds named", bool(sec) and docs.index("## Rules added in v2.6.3") < docs.index("## Rules added in v2.6.4")
      and "in the document" in sec and "bias bound" in sec and "confidence bounds" in sec and "ρ" in sec
      and "ci_verdict" in sec)
v262 = docs.split("## Rules added in v2.6.2")[1].split("## Rules added in v2.6.3")[0]
check("the v2.6.2 rule 'Say which ρ the bounds use' is kept and points to v2.6.4",
      "Say which ρ the bounds use (since v2.6.4: in the document itself" in v262)
cases = json.loads((ROOT / "evals" / "cases.json").read_text(encoding="utf-8"))["cases"]
c = [c_ for c_ in cases if c_.get("id") == "review-memo-dml-rho-and-interval"]
check("evaluation case review-memo-dml-rho-and-interval: one, pt-BR, prompt with both cells and the dml_bounds lines, "
      "must >= 3, must_not >= 2",
      len(c) == 1 and c[0]["language"] == "pt-BR" and len(c[0]["must"]) >= 3 and len(c[0]["must_not"]) >= 2
      and "−13,8" in c[0]["prompt"] and "−9,4" in c[0]["prompt"] and "theta between" in c[0]["prompt"]
      and "confidence bounds" in c[0]["prompt"])
check("ids of the evaluation cases are unique", len({c_["id"] for c_ in cases}) == len(cases))

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
