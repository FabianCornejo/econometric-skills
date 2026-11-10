"""Tests for v2.5.3: what the `show` blocks of the sensitivity functions say.

The numbers were right; the sentences were not, in cases the functions accept without complaint
(validation of v2.5.2.1 on notebook 1, 2026-11-10, and the independent reviews of v2.5.2.1):
  causal.dml_sensitivity  "RVa ... makes it no longer significant" for every q (with q != 1 it is the strength at
                          which the confidence bound reaches (1 - q)·theta); "100% confidence bounds" with
                          alpha = 0.001; with rho = 0 the RV is the ceiling of the search and the text said a
                          confounder of 100% "would move theta to 0"; a zero bias bound printed as a result;
                          RVa = 0 for an estimate that is already not significant.
  causal.dml_benchmark    cf_y = 0 or cf_d = 0 (rho = NaN) returned silently.
  specification.oster     the exact identified set was computed and never printed.
  specification.sensitivity / robustness_value   q <= 0, non-finite or bool accepted (q = -1 printed "reduce the
                          estimate by -100% (to 2 times its value)"); np.float32 printed 0.10000000149011612;
                          q a rounding error away from 1 took the q != 1 sentence; with q = 1 and RV_q,a = 0,
                          "0.0% would make it insignificant".

What is checked here:
  1. no number moved: references fixed from main 7c25644 (v2.5.2.1) in tests/data/v253_refs_7c25644.json,
     three DML designs x q in {0.5, 1, 2} x alpha in {0.05, 0.001} x rho in {1, 0.5, 0}, the benchmarks, Oster
     in two designs and two deltas, and sensitivity on an insignificant estimate;
  2. the q = 1 DML text is byte-identical to 7c25644 when the estimate is significant;
  3. the sentences state the definitions (Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis 2022): feeding RV
     back as cf_y = cf_d moves the bound to (1 - q)·theta, feeding RVa back moves the confidence bound there;
     "already" is judged on the confidence interval the reader sees (dml_plr's SE), in a case built between
     that interval and the confidence bounds with no confounder, and the sentences that must not appear do not;
  4. benchmarks with no strength warn, by name, in each of the three ways (cf_y = 0, cf_d = 0, both), and a
     strong benchmark does not;
  5. Oster prints the exact identified set, or says there is none;
  6. q is validated in the three functions, its label is readable, and q a rounding error from 1 is q = 1;
  7. sensitivity with q = 1 and RV_q,a = 0: "already not significant" when the reported CI contains zero, and the
     window between the reported CI (df) and the paper's convention (df - 1), checked against statsmodels.

Mutation (done when written): against 7c25644 only the text, warning and validation checks fail; every number
check passes.

Run from the repository root:  python3 tests/test_v253.py
Needs: statsmodels, scipy, scikit-learn.
"""
import contextlib
import io
import json
import re
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "econometrics" / "scripts"))
import _console  # noqa: E402,F401  (Windows-safe console output, see econometrics/scripts/_console.py)
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.linear_model import LinearRegression

import causal
import specification as sp

ok = []
REF = json.loads((HERE / "data" / "v253_refs_7c25644.json").read_text(encoding="utf-8"))
LIN = dict(learner_y=LinearRegression(), learner_d=LinearRegression())
NREP = {"D1": 2, "D2": 3, "D3": 2}
X = ["x1", "x2", "x3", "junk"]


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


def close(a, b, rel=1e-8):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(close(x, y, rel) for x, y in zip(a, b))
    if not (np.isfinite(a) and np.isfinite(b)):
        return (np.isnan(a) and np.isnan(b)) or a == b
    return abs(a - b) <= rel * max(1.0, abs(a), abs(b))


def dml_design(name):           # the designs of the reference file (see its _origin)
    seed, n, eff = {"D1": (1, 600, 0.4), "D2": (7, 400, 0.05), "D3": (11, 500, -0.3)}[name]
    rng = np.random.default_rng(seed)
    x1, x2, x3 = rng.normal(size=(3, n))
    d = 0.5 * x1 + 0.3 * x2 - 0.2 * x3 + rng.normal(size=n)
    y = eff * d + x1 + 0.5 * x2 + 0.3 * x3 + rng.normal(size=n)
    return pd.DataFrame(dict(y=y, d=d, x1=x1, x2=x2, x3=x3, junk=rng.normal(size=n)))


def oster_design(name):
    seed, n = {"O1": (5, 500), "O2": (1, 600)}[name]
    rng = np.random.default_rng(seed)
    if name == "O1":
        z1, z2 = rng.normal(size=(2, n))
        x = 0.6 * z1 + 0.3 * z2 + rng.normal(size=n)
        y = 0.5 * x + z1 + 0.5 * z2 + rng.normal(size=n)
        d, f = pd.DataFrame(dict(y=y, x=x, z1=z1, z2=z2)), "y ~ x + z1 + z2"
    else:
        rng.normal(size=(2, n)); rng.normal(size=n)          # draws of the original script, kept for the stream
        z = rng.normal(size=n)
        x = 0.6 * z + rng.normal(size=n)
        y = 0.5 * x + z + rng.normal(size=n)
        d, f = pd.DataFrame(dict(y=y, x=x, z=z)), "y ~ x + z"
    return smf.ols("y ~ x", d).fit(), smf.ols(f, d).fit()


def run(fn, *a, **k):
    """(result, stdout, warning messages)."""
    buf = io.StringIO()
    with warnings.catch_warnings(record=True) as w, contextlib.redirect_stdout(buf):
        warnings.simplefilter("always")
        out = fn(*a, **k)
    return out, buf.getvalue(), [str(x.message) for x in w]


def rv_line(txt):
    lines = [s for s in txt.splitlines() if s.startswith("  RV")]
    return lines[0] if len(lines) == 1 else "<%d RV lines>" % len(lines)


DF = {nm: dml_design(nm) for nm in NREP}
RES = {nm: causal.dml_plr(DF[nm], "y", "d", X, n_rep=NREP[nm], **LIN) for nm in NREP}

# --- 1. no number moved ---------------------------------------------------------------------------------
for nm in NREP:
    check(f"numbers: dml_plr {nm} unchanged", all(close(RES[nm][k], v) for k, v in REF["dml_plr"][nm].items()))
    bad = []
    for key, ref in REF["dml_sens"].items():
        dn, q, a, rho = key.split("|")
        if dn != nm:
            continue
        o = causal.dml_sensitivity(RES[nm], alpha=float(a), q=float(q), rho=float(rho), show=False)
        t = o["bound_table"]
        got = {k: float(o[k]) for k in ref if k != "cells"}
        got["cells"] = [float(t.loc[0.05, 0.05]), float(t.loc[0.30, 0.01]), float(t.loc[0.01, 0.30])]
        bad += [f"{key}:{k}" for k in ref if not close(got[k], ref[k])]
    check(f"numbers: dml_sensitivity {nm}, 18 cases (q, alpha, rho), 11 numbers each, unchanged", bad == [],
          str(bad[:5]))
    bm, _, _ = run(causal.dml_benchmark, DF[nm], "y", "d", X, ["x1", "junk"], res=RES[nm], n_rep=NREP[nm], **LIN)
    rb = REF["dml_bench"][nm]
    check(f"numbers: dml_benchmark {nm} unchanged",
          all(close(None if not np.isfinite(bm.loc[b, c]) else float(bm.loc[b, c]), rb[b][c])
              for b in rb for c in rb[b]))
for key, ref in REF["oster"].items():
    nm, delta = key.split("|")
    o, _, _ = run(sp.oster, *oster_design(nm), "x", delta=float(delta))
    got = {k: (list(o[k]) if isinstance(o.get(k), tuple) else o.get(k)) for k in ref if k != "n_warnings"}
    got = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in got.items()}
    check(f"numbers: oster {key} unchanged (and the same number of warnings)",
          all(close(got[k], ref[k]) for k in got) and len(o["warnings"]) == ref["n_warnings"])
rng = np.random.default_rng(21)
n = 120
x1, x2 = rng.normal(size=(2, n))
d = 0.5 * x1 + rng.normal(size=n)
y = 0.02 * d + x1 + 0.4 * x2 + rng.normal(size=n)
R_INS = smf.ols("y ~ d + x1 + x2", pd.DataFrame(dict(y=y, d=d, x1=x1, x2=x2))).fit()
check("numbers: sensitivity on an insignificant estimate unchanged (q = 0.5, 1, 2)",
      close(float(R_INS.pvalues["d"]), REF["sens"]["p_d"])
      and all(close([o["partial_r2"], o["rv_q"], o["rv_qa"]], REF["sens"][str(q)])
              for q in (0.5, 1.0, 2.0) for o in [sp.sensitivity(R_INS, "d", q=q)]))

# --- 2. q = 1, significant: DML text byte-identical to 7c25644 ------------------------------------------
for nm in ("D1", "D3"):
    _, txt, _ = run(causal.dml_sensitivity, RES[nm], show=True)
    check(f"text: dml_sensitivity {nm}, q = 1, significant, byte-identical to v2.5.2.1", txt == REF["texts"][nm],
          "" if txt == REF["texts"][nm] else repr(txt))

# --- 3. the sentences state the definitions --------------------------------------------------------------
z95 = stats.norm.ppf(0.975)
for nm in ("D1", "D3"):
    res, th = RES[nm], RES[nm]["estimate"]
    for q in (0.5, 1.0, 2.0):
        o, txt, _ = run(causal.dml_sensitivity, res, q=q, show=True)
        target = (1 - q) * th
        back_rv = causal.dml_sensitivity(res, o["rv"], o["rv"], q=q, show=False)
        back_rva = causal.dml_sensitivity(res, o["rva"], o["rva"], q=q, show=False)
        near = back_rv["theta_lower"] if th > 0 else back_rv["theta_upper"]
        near_ci = back_rva["ci_low"] if th > 0 else back_rva["ci_high"]
        check(f"definition {nm}, q={q}: RV fed back as cf_y = cf_d moves the bound to (1 - q)·theta",
              abs(near - target) < 1e-6 * abs(th), f"{near:.8f} vs {target:.8f}")
        check(f"definition {nm}, q={q}: RVa fed back moves the confidence bound to (1 - q)·theta",
              abs(near_ci - target) < 1e-6 * abs(th), f"{near_ci:.8f} vs {target:.8f}")
        line = rv_line(txt)
        if q == 1.0:
            check(f"text {nm}, q=1: 'makes it no longer significant' is what RVa measures", line.endswith(
                f"RVa = {o['rva']:.2%} makes it no longer significant."), line)
        else:
            check(f"text {nm}, q={q}: RVa puts (1 - q)·theta inside the confidence bounds; no 'no longer significant'",
                  f"would move theta to {target:.4f}" in line
                  and f"RVa = {o['rva']:.2%} would put that value inside the 95% confidence bounds." in line
                  and "no longer significant" not in txt, line)
# the level, printed with :g
_, txt, _ = run(causal.dml_sensitivity, RES["D1"], alpha=0.001, show=True)
check("text: alpha = 0.001 prints 99.9% (not 100%)", "99.9% confidence bounds" in txt and "100%" not in txt, txt)
_, txt, _ = run(causal.dml_sensitivity, RES["D1"], alpha=0.10, show=True)
check("text: alpha = 0.10 prints 90%", "; 90% confidence bounds [" in txt, txt)

# already not significant: D2 (p = 0.91)
o, txt, _ = run(causal.dml_sensitivity, RES["D2"], show=True)
check("text: D2 is already not significant by the reported CI (premise)",
      RES["D2"]["ci_low"] < 0 < RES["D2"]["ci_high"] and o["rva"] == 0.0)
check("text: D2, q = 1: says it is already not significant, with no confounder, and no 'RVa = 0.00% makes it'",
      "already not significant at the 95% level" in rv_line(txt) and "with no confounder" in rv_line(txt)
      and "RVa = 0.00%" not in txt and "makes it no longer significant" not in txt, rv_line(txt))

# the SE the reader sees (dml_plr) is never below the one inside RVa (median per-split SE, no spread across
# splits): sqrt(se_r^2 + (theta_r - theta)^2) >= se_r split by split, so the reported CI contains the confidence
# bounds with no confounder. Only one mismatch can happen: the reported CI already contains the target while RVa
# is still positive. "already" must follow the reported CI.
for nm in NREP:
    b0 = causal.dml_sensitivity(RES[nm], rho=0.0, show=False)
    check(f"premise {nm}: reported CI contains the confidence bounds with no confounder",
          RES[nm]["ci_low"] <= b0["ci_low"] + 1e-12 and b0["ci_high"] <= RES[nm]["ci_high"] + 1e-12)
res, th = RES["D1"], RES["D1"]["estimate"]
b0 = causal.dml_sensitivity(res, rho=0.0, show=False)
mid = 0.5 * (res["ci_low"] + b0["ci_low"])               # inside the reported CI, outside the zero-confounder bounds
q_mid = 1 - mid / th
o, txt, _ = run(causal.dml_sensitivity, res, q=q_mid, show=True)
check("boundary: target between the reported CI and the zero-confounder bounds (premise: RVa > 0)",
      res["ci_low"] < mid < b0["ci_low"] and o["rva"] > 0, f"{res['ci_low']:.6f} < {mid:.6f} < {b0['ci_low']:.6f}")
check("boundary: judged on the reported CI, 'already inside the 95% confidence interval, with no confounder'",
      "is already inside the 95% confidence interval, with no confounder" in rv_line(txt)
      and "would put that value inside" not in txt, rv_line(txt))
o, txt, _ = run(causal.dml_sensitivity, res, q=1 - (res["ci_low"] - 0.002) / th, show=True)   # just outside
check("boundary: a target just outside the reported CI keeps the RVa sentence",
      "would put that value inside the 95% confidence bounds" in rv_line(txt) and "already" not in txt, rv_line(txt))

# rho = 0: the bias bound is zero for every strength, the RV is the ceiling of the search
zero = [causal.dml_sensitivity(res, s, s, rho=0.0, show=False)["bias_bound"] for s in (0.01, 0.3, 0.9, 0.999)]
check("definition: with rho = 0 the bias bound is zero at every strength", all(b == 0 for b in zero), str(zero))
o, txt, _ = run(causal.dml_sensitivity, res, rho=0.0, show=True)
check("text: rho = 0 says no confounder moves theta (ceiling), never 'a confounder explaining 100.0% ... would move'",
      o["rv"] == 1.0 and "no confounder with cf_y = cf_d and rho = 0.0 moves theta to 0.0000" in rv_line(txt)
      and "would move theta" not in txt and "makes it no longer significant" not in txt, rv_line(txt))
check("text: a zero bias bound is flagged as saying nothing about confounding",
      "bias bound is zero" in txt and "say nothing about confounding" in txt, txt)
o, txt, _ = run(causal.dml_sensitivity, res, 0.0, 0.05, show=True)
check("text: cf_y = 0 (a benchmark with no strength) is flagged too", "bias bound is zero" in txt, txt)
o, txt, _ = run(causal.dml_sensitivity, res, show=True)
check("text: a non-zero bound has no such note", "bias bound is zero" not in txt)

# --- 4. benchmarks with no strength warn, by name -----------------------------------------------------------
which = {"D1": "cf_y = 0", "D2": "cf_d = 0", "D3": "cf_y = 0, cf_d = 0"}
for nm in NREP:
    bm, _, w = run(causal.dml_benchmark, DF[nm], "y", "d", X, ["x1", "junk"], res=RES[nm], n_rep=NREP[nm], **LIN)
    mine = [m for m in w if m.startswith("Benchmark ")]
    check(f"benchmark {nm}: one warning, naming junk ({which[nm]}), none for x1",
          len(mine) == 1 and mine[0].startswith("Benchmark junk:") and "x1" not in mine[0]
          and "bias bound of zero" in mine[0], str(mine))
    check(f"benchmark {nm}: the warning says which side is zero ({which[nm]})",
          len(mine) == 1 and (("outcome" in mine[0]) == (bm.loc["junk", "cf_y"] == 0))
          and (("treatment" in mine[0]) == (bm.loc["junk", "cf_d"] == 0)), str(mine))

# --- 5. Oster prints the exact identified set ----------------------------------------------------------------
o, txt, _ = run(sp.oster, *oster_design("O1"), "x", delta=1.0, show=True)
lo, hi = o["identified_set_exact"]
check("oster: the exact identified set is printed", f"identified set [{lo:.4f}, {hi:.4f}] (exact)" in txt, txt)
o, txt, _ = run(sp.oster, *oster_design("O2"), "x", delta=2.0, show=True)
check("oster: no exact root -> says there is no exact identified set (premise: identified_set_exact is None)",
      o["identified_set_exact"] is None and "no exact identified set" in txt, txt)

# --- 6. q is validated; label readable; q a rounding error from 1 is q = 1 -----------------------------------
for bad_q in (0, -1, -0.5, float("nan"), float("inf")):
    errs = []
    for f, args in ((sp.robustness_value, (3.0, 100)), (sp.sensitivity, (R_INS, "d")),
                    (causal.dml_sensitivity, (RES["D1"],))):
        try:
            f(*args, q=bad_q, show=False) if f is not sp.robustness_value else f(*args, q=bad_q)
            errs.append("accepted")
        except ValueError as e:
            errs.append("q" in str(e) and "ValueError")
        except Exception as e:                       # any other error is a failure of the check
            errs.append(type(e).__name__)
    check(f"validation: q = {bad_q} is a ValueError naming q in the three functions", errs == ["ValueError"] * 3,
          str(errs))
for bad_q in (True, "0.5", None):
    errs = []
    for f, args in ((sp.robustness_value, (3.0, 100)), (sp.sensitivity, (R_INS, "d")),
                    (causal.dml_sensitivity, (RES["D1"],))):
        try:
            f(*args, q=bad_q, show=False) if f is not sp.robustness_value else f(*args, q=bad_q)
            errs.append("accepted")
        except TypeError:
            errs.append("TypeError")
        except Exception as e:
            errs.append(type(e).__name__)
    check(f"validation: q = {bad_q!r} is a TypeError in the three functions", errs == ["TypeError"] * 3, str(errs))
r_sig = smf.ols("y ~ d + x1 + x2 + x3", DF["D1"]).fit()
o, txt, _ = run(sp.sensitivity, r_sig, "d", q=np.float32(0.1), show=True)
check("label: np.float32(0.1) prints RV(q=0.1), not 0.10000000149011612",
      re.findall(r"RV\(q=([^,)]+)", txt) == ["0.1", "0.1"] and "0000001" not in txt, txt.splitlines()[1])
check("numbers: np.float32(0.1) computes with the value it was given (same as float(np.float32(0.1)))",
      close(o["rv_q"], sp.sensitivity(r_sig, "d", q=float(np.float32(0.1)))["rv_q"], 1e-14))
o, txt, _ = run(sp.sensitivity, r_sig, "d", q=np.int64(2), show=True)
check("label: np.int64(2) prints RV(q=2)", re.findall(r"RV\(q=([^,)]+)", txt) == ["2", "2"], txt.splitlines()[1])
o, txt, _ = run(sp.sensitivity, r_sig, "d", q=1 - 1e-12, show=True)
check("label: q = 1 - 1e-12 is q = 1: RV(q=1.0) and 'would remove the estimate', not 'to 0 times'",
      re.findall(r"RV\(q=([^,)]+)", txt) == ["1.0", "1.0"] and "would remove the estimate" in txt
      and "times its value" not in txt, txt)
o, txt, _ = run(causal.dml_sensitivity, RES["D1"], q=1 + 1e-12, show=True)
check("dml: q = 1 + 1e-12 is q = 1 (the q = 1 sentence)", "makes it no longer significant." in txt, rv_line(txt))
o, txt, _ = run(sp.sensitivity, r_sig, "d", q=1.0, show=True)
check("label: q = 1.0 still prints RV(q=1.0)", re.findall(r"RV\(q=([^,)]+)", txt) == ["1.0", "1.0"])

# --- 7. sensitivity, q = 1, RV_q,a = 0 -------------------------------------------------------------------------
o, txt, _ = run(sp.sensitivity, R_INS, "d", show=True)
check("q=1, insignificant (p = 0.97): 'already not significant ..., with no confounder', no '0.0% would make it'",
      o["rv_qa"] == 0 and "it is already not significant at the 95% level, with no confounder" in txt
      and "would make it insignificant" not in txt, txt.splitlines()[2])
# the window: |t| between t_df (reported CI excludes 0) and t_{df-1}·sqrt(df/(df-1)) (paper's convention: rv_qa = 0)
rng = np.random.default_rng(253)
n = 15
w = pd.DataFrame(dict(d=rng.normal(size=n), x1=rng.normal(size=n)))
w["e"] = w.x1 + rng.normal(size=n)
base = smf.ols("e ~ d + x1", w).fit()
dof = base.df_resid
t_lo, t_hi = stats.t.ppf(0.975, dof), stats.t.ppf(0.975, dof - 1) * np.sqrt(dof / (dof - 1))
c = 0.5 * (t_lo + t_hi) * base.bse["d"] - base.params["d"]      # adding c·d to y moves the slope by c, same SE
w["y"] = w.e + c * w.d
r_win = smf.ols("y ~ d + x1", w).fit()
ci = r_win.conf_int().loc["d"]
o, txt, _ = run(sp.sensitivity, r_win, "d", show=True)
check("window premise: t between t_df and t_(df-1)·sqrt(df/(df-1)); statsmodels' CI excludes 0; rv_qa = 0",
      t_lo < abs(r_win.tvalues["d"]) < t_hi and (ci[0] > 0 or ci[1] < 0) and o["rv_qa"] == 0,
      f"t = {r_win.tvalues['d']:.4f} in ({t_lo:.4f}, {t_hi:.4f}); CI {list(ci.round(4))}")
check("window: 'only just significant ...: adding any regressor, even one unrelated to both, makes it insignificant'",
      "it is only just significant at the 95% level: adding any regressor, even one unrelated to both, makes it "
      "insignificant" in txt and "already" not in txt and "would make it insignificant" not in txt, txt.splitlines()[2])

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
