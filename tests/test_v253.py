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
    # v2.6.2: cf_y and cf_d became the paper's gains; the partial R² this check fixed are now r2_y and r2_d, and the
    # check still proves that no number moved under its new name (rho, theta_without, delta_theta unchanged)
    old_name = {"cf_y": "r2_y", "cf_d": "r2_d"}
    check(f"numbers: dml_benchmark {nm} unchanged (old cf_y, cf_d as r2_y, r2_d since v2.6.2)",
          all(close(None if not np.isfinite(bm.loc[b, old_name.get(c, c)]) else float(bm.loc[b, old_name.get(c, c)]),
                    rb[b][c]) for b in rb for c in rb[b]))
    check(f"numbers: dml_benchmark {nm}, cf_y and cf_d above the reference partial R² (gains, v2.6.2)",
          all(bm.loc[b, c] >= rb[b][c] for b in rb for c in ("cf_y", "cf_d")))
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
# v2.6.1 (decision 2 of the user): the q = 1 target prints "0.0000" for theta < 0 (D3), not "-0.0000"; that one
# substitution is the only change allowed, so the rest of the text is still byte-identical to v2.5.2.1
for nm in ("D1", "D3"):
    _, txt, _ = run(causal.dml_sensitivity, RES[nm], show=True)
    ref_ = REF["texts"][nm].replace("theta to -0.0000", "theta to 0.0000")
    check(f"text: dml_sensitivity {nm}, q = 1, significant, byte-identical to v2.5.2.1 (D3: '-0.0000' -> '0.0000' "
          f"only, v2.6.1)", txt == ref_ and (nm == "D1") == (ref_ == REF["texts"][nm]),
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
      "already not significant at the 5% level" in rv_line(txt) and "with no confounder" in rv_line(txt)
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
      o["rv"] == 1.0 and "no confounder with cf_y = cf_d short of 100% and rho = 0.0 moves theta to 0.0000" in rv_line(txt)
      and rv_line(txt).endswith("; nor does any make it no longer significant.")
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
      o["rv_qa"] == 0 and "it is already not significant at the 5% level, with no confounder" in txt
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
check("window: 'only just significant ...: adding a regressor unrelated to both (it only costs one degree of freedom) "
      "would make it insignificant'",
      "it is only just significant at the 5% level: adding a regressor unrelated to both (it only costs one degree of "
      "freedom) would make it insignificant" in txt and "already" not in txt and "any regressor" not in txt,
      txt.splitlines()[2])
# ...and the sentence is true: a regressor with zero partial R² with d, x1 and y (built orthogonal to all of them)
# leaves the slope and the SSR unchanged and costs one degree of freedom; one that predicts y does the opposite
Z = np.column_stack([np.ones(n), w.d, w.x1, w.y])
u = rng.normal(size=n)
w["z0"] = u - Z @ np.linalg.lstsq(Z, u, rcond=None)[0]
w["zy"] = w.y - w.d * r_win.params["d"] - r_win.params["Intercept"] + 0.1 * rng.normal(size=n)
p0 = smf.ols("y ~ d + x1 + z0", w).fit().pvalues["d"]
py = smf.ols("y ~ d + x1 + zy", w).fit().pvalues["d"]
check("window, the claim: a regressor unrelated to both makes it insignificant (p >= 0.05; was < 0.05)",
      r_win.pvalues["d"] < 0.05 <= p0, f"p {r_win.pvalues['d']:.4f} -> {p0:.4f}")
check("window, why not 'any regressor': one that predicts y keeps it significant", py < 0.05, f"p {py:.2e}")

# --- 8. review round 1 (2026-11-10): one check per finding, on the claim and not only on the string -------------
# A1: the zero-bound note follows the bound itself (cf_d = 1 gives an infinite bound, not a zero one)
o, txt, _ = run(causal.dml_sensitivity, RES["D1"], 0.0, 1.0, show=True)
check("A1: cf_y = 0, cf_d = 1 (bias bound inf) has no 'bias bound is zero' note",
      not np.isfinite(o["bias_bound"]) and "bias bound is zero" not in txt, txt.splitlines()[2])
o, txt, _ = run(causal.dml_sensitivity, RES["D1"], 0.05, 0.0, show=True)
check("A2c: cf_d = 0 (bias bound 0) is flagged", o["bias_bound"] == 0 and "bias bound is zero" in txt)
# A2a: "already" uses the alpha asked for: at alpha = 0.2 the reported CI is narrower than at 0.05
res1, th1, se1 = RES["D1"], RES["D1"]["estimate"], RES["D1"]["se"]
q80 = 1 - (th1 - 0.5 * (stats.norm.ppf(0.975) + stats.norm.ppf(0.9)) * se1) / th1   # target inside 95%, outside 80%
o, txt, _ = run(causal.dml_sensitivity, res1, q=q80, alpha=0.2, show=True)
check("A2a: target inside the 95% CI but outside the 80% one, alpha = 0.2: not 'already', RVa sentence at 80%",
      "already" not in txt and "would put that value inside the 80% confidence bounds" in rv_line(txt), rv_line(txt))
o, txt, _ = run(causal.dml_sensitivity, res1, q=q80, alpha=0.05, show=True)
check("A2a: the same target with alpha = 0.05 is 'already inside the 95% confidence interval'",
      "is already inside the 95% confidence interval" in rv_line(txt), rv_line(txt))
# A2b/A2d: rho = 0, q = 0.5, significant estimate: both at the ceiling, the q != 1 wording
o, txt, _ = run(causal.dml_sensitivity, res1, q=0.5, rho=0.0, show=True)
check("A2b/d: rho = 0, q = 0.5: ceiling sentence and 'nor does any put that value inside the 95% confidence bounds'",
      o["rv"] == 1.0 and o["rva"] == 1.0 and "(ceiling)" in rv_line(txt)
      and "nor does any put that value inside the 95% confidence bounds" in rv_line(txt) and "RVa =" not in txt,
      rv_line(txt))
# A4: a tiny rho reaches the ceiling of the search without the bound being zero: the text says "short of 100%"
o, txt, _ = run(causal.dml_sensitivity, res1, rho=1e-6, show=True)
check("A4: rho = 1e-6 hits the search ceiling; the text limits the claim to strengths short of 100%",
      o["rv"] == 1.0 and o["bias_bound"] > 0 and "short of 100%" in rv_line(txt) and "bias bound is zero" not in txt,
      rv_line(txt))
# A3: a NaN rho (what dml_benchmark returns for a benchmark with no strength) is a clear ValueError; so is ±inf
for bad_rho in (float("nan"), float("inf"), -float("inf")):
    try:
        causal.dml_sensitivity(res1, 0.0, 0.05, rho=bad_rho, show=False)
        a3 = "accepted"
    except ValueError as e:
        a3 = bool("rho" in str(e) and "finite" in str(e)) and (("benchmark" in str(e)) == bool(np.isnan(bad_rho)))
    check(f"A3: rho = {bad_rho} is a ValueError naming rho (the benchmark hint only for nan)", a3 is True, str(a3))
# DML strengths that would print as zero: until v2.6.1 a large rho (30000, 6000, 3000) reached the "< 0.01%" branches
# end to end. rho is a correlation and |rho| > 1 is now refused, which makes those branches unreachable with real data
# (a significant RV below 0.005% needs n of the order of 1e9); the formatter's thresholds are tested directly on
# causal._dml_pct in test_v261.py, and here the old inputs must be refused.
for big in (30000.0, 6000.0, 3000.0):
    try:
        causal.dml_sensitivity(res1, rho=big, show=False)
        e_ = "accepted"
    except ValueError as ex:
        e_ = str(ex)
    check(f"DML: rho = {big:g} (the old way to reach '< 0.01%') is refused since v2.6.1", "[-1, 1]" in e_, e_)
# A5: alpha set at the p-value (the reported CI ends at 0 up to rounding): RVa = 0 never prints as a strength
bad = []
for s_ in range(100, 130):
    rng_ = np.random.default_rng(s_)
    n_ = 80
    a_, b_ = rng_.normal(size=(2, n_))
    dd_ = 0.5 * a_ + rng_.normal(size=n_)
    df_ = pd.DataFrame(dict(y=0.25 * dd_ + a_ + 0.3 * b_ + rng_.normal(size=n_), d=dd_, a=a_, b=b_))
    r_ = causal.dml_plr(df_, "y", "d", ["a", "b"], n_rep=1, n_folds=2, **LIN)
    o_, t_, _ = run(causal.dml_sensitivity, r_, alpha=float(r_["p"]), show=True)
    if o_["rva"] == 0 and "already not significant" not in rv_line(t_):
        bad.append(s_)
check("A5: alpha at the p-value, 30 designs: whenever RVa = 0 the text says 'already', never a 0% strength", bad == [],
      str(bad))
# B1: a robust fit: the statistics use the classical SE, and the text says so with both t's
rng_ = np.random.default_rng(6)
n_ = 200
dd_, xx_ = rng_.normal(size=(2, n_))
yy_ = 0.12 * dd_ + xx_ + rng_.normal(size=n_) * np.exp(-0.8 * np.abs(dd_))
dfr = pd.DataFrame(dict(y=yy_, d=dd_, x=xx_))
r_hc = smf.ols("y ~ d + x", dfr).fit(cov_type="HC1")
r_cl = smf.ols("y ~ d + x", dfr).fit()
o, txt, _ = run(sp.sensitivity, r_hc, "d", show=True)
txt_hc = txt
check("B1 premise: HC1 says significant, the classical SE does not",
      r_hc.pvalues["d"] < 0.05 < r_cl.pvalues["d"], f"{r_hc.pvalues['d']:.4f} vs {r_cl.pvalues['d']:.4f}")
check("B1: the verdict is labelled as the classical one, with both t's and the cov_type",
      "use the classical SE" in txt and f"t = {r_cl.tvalues['d']:.2f}" in txt and f"HC1 fit gives t = "
      f"{r_hc.tvalues['d']:.2f}" in txt, txt)
o, txt, _ = run(sp.sensitivity, r_cl, "d", show=True)
check("B1: the verdict itself is the classical one (classical p > 0.05): 'already not significant at the 5% level'",
      "it is already not significant at the 5% level, with no confounder" in txt_hc and "only just" not in txt_hc,
      txt_hc.splitlines()[2])
check("B1: a classical fit has no such line", "use the classical SE" not in txt)
# B3: Oster prints the exact identified set only as reportable when the warnings allow it
mt = pd.read_csv(HERE.parents[0] / "tools" / "r_reference" / "output" / "mtcars.csv")
o, txt, _ = run(sp.oster, smf.ols("mpg ~ wt", mt).fit(), smf.ols("mpg ~ wt + hp + qsec", mt).fit(), "wt",
                r_max=0.9, delta=2, show=True)
check("B3: Assumption 3 violated (mtcars, δ = 2): the set is printed as 'not to be reported', never '(exact)'",
      any("Assumption 3" in m for m in o["warnings"]) and "not to be reported" in txt and "] (exact)" not in txt, txt)


def population(n, c1, c2, g1, g2, seed):    # Oster's proportional selection, as in tests/test_v252.py
    rng = np.random.default_rng(seed)
    w1, w2 = rng.standard_normal((2, n))
    x = c1 * w1 + c2 * w2 + rng.standard_normal(n)
    y = x + g1 * w1 + g2 * w2 + rng.standard_normal(n)
    d = pd.DataFrame({"y": y, "x": x, "w1": w1, "w2": w2})
    return (smf.ols("y ~ x", d).fit(), smf.ols("y ~ x + w1", d).fit(),
            smf.ols("y ~ x + w1 + w2", d).fit().rsquared, (c2 / g2) / (c1 / g1))


m0, m1, rm, dl = population(2000, 0.6, 1.2, 1, 1, 0)
o, txt, _ = run(sp.oster, m0, m1, "x", r_max=rm, delta=dl, show=True)
check("B3: unstable root (near a double root): 'not to be reported', never '(exact)'",
      o["beta_star_exact_unstable"] and "not to be reported" in txt and "] (exact)" not in txt, txt)
# B4: a positive RV_q,α that would print as 0.0% says "less than 0.1%"
rng_ = np.random.default_rng(4)
n_ = 200
wb = pd.DataFrame(dict(d=rng_.normal(size=n_), x1=rng_.normal(size=n_)))
wb["e"] = wb.x1 + rng_.normal(size=n_)
bb = smf.ols("e ~ d + x1", wb).fit()
dfb = bb.df_resid
t_top = 1.001 * stats.t.ppf(0.975, dfb - 1) * np.sqrt(dfb / (dfb - 1))           # just above the band
wb["y"] = wb.e + (t_top * bb.bse["d"] - bb.params["d"]) * wb.d
o, txt, _ = run(sp.sensitivity, smf.ols("y ~ d + x1", wb).fit(), "d", show=True)
check("B4: 0 < RV_q,α < 0.05% prints 'less than 0.1% would make it insignificant', not '0.0%'",
      0 < o["rv_qa"] < 0.0005 and "less than 0.1% would make it insignificant" in txt and " 0.0%" not in txt,
      f"{o['rv_qa']:.2e}")
# B5: label and sentence agree near 1; large and tiny q print readably
o, txt, _ = run(sp.sensitivity, r_sig, "d", q=0.9999999, show=True)
check("B5: q = 0.9999999 prints RV(q=1.0) and the q = 1 sentence (they agree)",
      re.findall(r"RV\(q=([^,)]+)", txt) == ["1.0", "1.0"] and "would remove the estimate" in txt, txt.splitlines()[1])
o, txt, _ = run(sp.sensitivity, r_sig, "d", q=0.999998, show=True)
check("B5: q = 0.999998 prints RV(q=0.999998) and the q != 1 sentence",
      re.findall(r"RV\(q=([^,)]+)", txt) == ["0.999998", "0.999998"] and "times its value" in txt)
o, txt, _ = run(sp.sensitivity, r_sig, "d", q=123456789.0, show=True)
check("B5: q = 123456789.0 prints RV(q=1.23457e+08), not 123457000.0",
      re.findall(r"RV\(q=([^,)]+)", txt) == ["1.23457e+08", "1.23457e+08"], txt.splitlines()[1])
o, txt, _ = run(sp.sensitivity, r_sig, "d", q=1e-12, show=True)
check("B5: q = 1e-12 reduces the estimate 'by 1e-10%', not 'by 0%'", "reduce the estimate by 1e-10% (to 1 times" in txt,
      txt.splitlines()[2])
# B6: the TypeError message names what is accepted
from decimal import Decimal
try:
    sp.robustness_value(3.0, 100, q=Decimal("0.5"))
    b6 = "accepted"
except TypeError as e:
    b6 = "int or a float" in str(e)
check("B6: Decimal is a TypeError that says 'int or a float'", b6 is True, str(b6))

# --- 9. review round 2 (2026-11-10) ---------------------------------------------------------------------------
# the lower edge of the band: |t| strictly between t_df and t_{df-1} (the midpoint used above lies above t_{df-1})
def edge_fit(q_, seed=2532):
    rng_ = np.random.default_rng(seed)
    n_ = 15
    w_ = pd.DataFrame(dict(d=rng_.normal(size=n_), x1=rng_.normal(size=n_)))
    w_["e"] = w_.x1 + rng_.normal(size=n_)
    b_ = smf.ols("e ~ d + x1", w_).fit()
    df_ = b_.df_resid
    t_e = 0.5 * (stats.t.ppf(0.975, df_) + stats.t.ppf(0.975, df_ - 1)) / q_       # q·|t| inside (t_df, t_{df-1})
    w_["y"] = w_.e + (t_e * b_.bse["d"] - b_.params["d"]) * w_.d
    return w_, smf.ols("y ~ d + x1", w_).fit(), df_, rng_


for q_ in (1.0, 0.5):
    w_, r_e, df_, rng_ = edge_fit(q_)
    qt = q_ * abs(r_e.tvalues["d"])
    o, txt, _ = run(sp.sensitivity, r_e, "d", q=q_, show=True)
    ci = r_e.conf_int(0.05).loc["d"]
    target = (1 - q_) * r_e.params["d"]
    check(f"band, lower edge (q={q_}): premise t_df < q|t| < t_(df-1); the reported CI excludes the target",
          stats.t.ppf(0.975, df_) < qt < stats.t.ppf(0.975, df_ - 1) and not (ci[0] <= target <= ci[1]), f"{qt:.4f}")
    want = ("only just significant at the 5% level: adding a regressor unrelated to both" if q_ == 1
            else "just outside the 95% confidence interval: adding a regressor unrelated to both")
    check(f"band, lower edge (q={q_}): the band sentence, not 'already'", want in txt and "already" not in txt,
          txt.splitlines()[2])
    Z_ = np.column_stack([np.ones(len(w_)), w_.d, w_.x1, w_.y])
    u_ = rng_.normal(size=len(w_))
    w_["z0"] = u_ - Z_ @ np.linalg.lstsq(Z_, u_, rcond=None)[0]
    ci0 = smf.ols("y ~ d + x1 + z0", w_).fit().conf_int(0.05).loc["d"]
    check(f"band, lower edge (q={q_}), the claim: with a regressor orthogonal to both the target enters the CI",
          ci0[0] <= target <= ci0[1], f"CI {list(np.round(ci0, 4))}, target {target:.4f}")
# the threshold of "less than 0.1%": 0.06% prints as 0.1%, 0.04% does not
check("_pct: 0.0006 -> '0.1%', 0.0004 -> 'less than 0.1%', 0 -> '0.0%', 0.25 -> '25.0%'",
      [sp._pct(v) for v in (0.0006, 0.0004, 0.0, 0.25)] == ["0.1%", "less than 0.1%", "0.0%", "25.0%"],
      str([sp._pct(v) for v in (0.0006, 0.0004, 0.0, 0.25)]))
# the q != 1 'less than 0.1%' path: q·|t| just above the band
q_ = 0.5
wb2 = wb.copy()
wb2["y"] = wb2.e + (t_top / q_ * bb.bse["d"] - bb.params["d"]) * wb2.d
o, txt, _ = run(sp.sensitivity, smf.ols("y ~ d + x1", wb2).fit(), "d", q=q_, show=True)
check("q = 0.5 just above the band: 'less than 0.1% would put that reduced value inside', not '0.0%'",
      0 < o["rv_qa"] < 0.0005 and "less than 0.1% would put that reduced value inside" in txt and " 0.0%" not in txt,
      f"{o['rv_qa']:.2e}")
try:
    sp.robustness_value(3.0, 100, q=2 ** 1100)
    big = "accepted"
except ValueError:
    big = "ValueError"
except Exception as e:
    big = type(e).__name__
check("q = 2**1100 (too large for a float) is a ValueError, not an OverflowError", big == "ValueError", big)

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
