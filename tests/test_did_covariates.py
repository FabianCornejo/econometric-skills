"""Tests for v2.3: DiD with covariates (causal.att_gt covariates=, est_method=, x_periods=, verify=),
the common causal covariates test (causal.ccc_test), the TWFE hidden-linearity diagnostic
(causal.twfe_covariate_diagnostic), the Teacher demo (simulation.demo_did_covariates) and the
reproducibility tools (reporting.run_manifest, reporting.cross_check).

Checks: each group-time 2x2 (outcome regression, IPW, doubly robust) equals an independent
statsmodels implementation, and the whole att_gt equals the csdid/DRDID port cell by cell when
csdid is installed; under conditional parallel trends with a drifting covariate effect, the
covariate-adjusted ATT is unbiased while unadjusted DiD and TWFE with the covariate are not;
double robustness (each nuisance model misspecified in turn); bootstrap CI coverage; the
independent-recomputation and overlap warnings; ccc_test statistics equal statsmodels HC1 and
CR1 Wald tests, the wild bootstrap has correct size where the chi-square over-rejects, and the
selection recovers the data-generating case; the TWFE diagnostic flags hidden linearity and
negative weights and stays quiet in a benign design; manifests are stable and sensitive.

Run from the repository root:  python3 tests/test_did_covariates.py
"""
import io
import json
import contextlib
import sys
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "econometrics" / "scripts"))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

import causal as ca
import reporting as rp
import simulation as sim

ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


def panel(seed, n=600, T=6, tau=1.0, drift=0.4, cohorts=(0, 3, 5), p=(.5, .25, .25)):
    """Staggered panel; parallel trends hold conditional on x (its effect drifts over time)."""
    rng = np.random.default_rng(seed)
    g = rng.choice(cohorts, n, p=p)
    x = rng.normal(0.8 * (g > 0), 1)
    a = rng.normal(0.5 * (g > 0), 1)
    rows = []
    for t in range(1, T + 1):
        x = x + rng.normal(0.2, 0.3, n)
        yv = a + (1 + drift * t) * x + 0.3 * t + tau * ((g > 0) & (t >= g)) + rng.normal(0, 1, n)
        rows.append(pd.DataFrame({"id": np.arange(n), "t": t, "y": yv, "x": x, "g": g}))
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------- 1. the 2x2 cell estimators
rng = np.random.default_rng(11)
n = 800
x1, x2 = rng.normal(size=n), rng.normal(size=n)
D = rng.binomial(1, 1 / (1 + np.exp(-(0.3 + 0.8 * x1 - 0.5 * x2))))
dy = 1.0 * D + 0.6 * x1 - 0.4 * x2 + 0.3 * x1 * x2 + rng.normal(size=n)
Z = np.column_stack([np.ones(n), x1, x2])
mine = ca._did_cell_cov(dy, Z, D.astype(float))
ols = sm.OLS(dy[D == 0], Z[D == 0]).fit()
m0 = Z @ ols.params
ps = sm.Logit(D, Z).fit(disp=0, tol=1e-12).predict(Z)
w1 = D / D.mean()
w0 = (1 - D) * ps / (1 - ps)
w0 = w0 / w0.mean()
ref = {"reg": np.mean(w1 * (dy - m0)), "ipw": np.mean(w1 * dy) - np.mean(w0 * dy),
       "dr": np.mean(w1 * (dy - m0)) - np.mean(w0 * (dy - m0))}
for m in ("reg", "ipw", "dr"):
    check(f"2x2 {m} equals an independent statsmodels OLS/Logit implementation", abs(mine[m] - ref[m]) < 1e-7,
          f"{mine[m]:.10f} vs {ref[m]:.10f}")

# ---------------------------------------------------------------- 2. external implementation
try:
    from csdid.att_gt import ATTgt
    q = panel(1)
    q["x0"] = q.groupby("id")["x"].transform("first")
    q["x0sq"] = q["x0"] ** 2
    worst = 0.0
    for ctrl, cs_ctrl in (("never", "nevertreated"), ("notyet", "notyettreated")):
        for m in ("dr", "reg", "ipw"):
            r = ca.att_gt(q, "id", "t", "y", "g", control=ctrl, n_boot=0, covariates=["x0", "x0sq"], est_method=m,
                          verify=False, show=False)
            with contextlib.redirect_stdout(io.StringIO()):
                a = ATTgt(yname="y", gname="g", idname="id", tname="t", xformla="y~x0+x0sq", data=q,
                          control_group=cs_ctrl).fit(est_method=m)
            cs = pd.DataFrame({"g": a.results["group"], "t": a.results["year"], "cs": a.results["att"]})
            mm = r["gt"].reset_index().merge(cs, on=["g", "t"])
            worst = max(worst, float(np.max(np.abs(mm["att"] - mm["cs"]))))
    check("att_gt with covariates equals csdid (DRDID) cell by cell: 3 methods x 2 control groups",
          worst < 1e-8, f"max |diff| = {worst:.2e}")
except ImportError:
    print("SKIP csdid not installed (pip install csdid to compare against the DRDID port)")

# ---------------------------------------------------------------- 3. unbiasedness under conditional PT
est = {"none": [], "dr": [], "reg": [], "ipw": []}
for s in range(40):
    p = panel(100 + s, n=500)
    est["none"].append(ca.att_gt(p, "id", "t", "y", "g", n_boot=0, show=False)["overall"]["estimate"])
    r = ca.att_gt(p, "id", "t", "y", "g", n_boot=0, covariates=["x"], x_periods="both", show=False)
    for m in ("dr", "reg", "ipw"):
        est[m].append(r["cross_check"][m])
bias = {k: np.mean(v) - 1 for k, v in est.items()}
check("unadjusted DiD is biased when trends depend on covariate levels", bias["none"] > 0.3, f"bias {bias['none']:.3f}")
for m in ("dr", "reg", "ipw"):
    check(f"att_gt covariates ({m}) is unbiased under conditional parallel trends", abs(bias[m]) < 0.06,
          f"bias {bias[m]:+.3f}")
dm = sim.demo_did_covariates(R=12, seed=3)["table"]
check("demo: TWFE with the covariate keeps the bias (hidden linearity)", dm.loc["TWFE with X", "bias"] > 0.5,
      f"{dm.loc['TWFE with X', 'bias']:.3f}")
check("demo: CS doubly robust with X in levels recovers the ATT",
      abs(dm.loc["CS doubly robust, X in levels", "bias"]) < 0.1,
      f"{dm.loc['CS doubly robust, X in levels', 'bias']:+.3f}")


# ---------------------------------------------------------------- 4. double robustness (2 periods)
def two_period(seed, n=3000, wrong="outcome"):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    if wrong == "outcome":            # trend nonlinear in x (linear outcome model wrong), logit propensity right
        D = rng.binomial(1, 1 / (1 + np.exp(-(-0.2 + 1.0 * x))))
        trend = 0.8 * x ** 2
    else:                             # trend linear in x (outcome model right), propensity non-monotone in x
        D = rng.binomial(1, np.where(x > 0.5, 0.85, np.where(x > -1.0, 0.15, 0.6)))
        trend = 0.8 * x
    y0 = x + rng.normal(size=n)
    y1 = y0 + trend + 1.0 * D + rng.normal(size=n)
    return pd.DataFrame({"id": np.r_[np.arange(n), np.arange(n)], "t": np.r_[np.ones(n), 2 * np.ones(n)],
                         "y": np.r_[y0, y1], "x": np.r_[x, x], "g": np.r_[2 * D, 2 * D]})


for wrong in ("outcome", "propensity"):
    vals = {"dr": [], "reg": [], "ipw": []}
    for s in range(30):
        r = ca.att_gt(two_period(s, wrong=wrong), "id", "t", "y", "g", n_boot=0, covariates=["x"], show=False)
        for m in vals:
            vals[m].append(r["cross_check"][m])
    b = {m: np.mean(v) - 1 for m, v in vals.items()}
    bad = "reg" if wrong == "outcome" else "ipw"
    check(f"double robustness: {wrong} model wrong -> DR unbiased", abs(b["dr"]) < 0.06, f"DR bias {b['dr']:+.3f}")
    check(f"double robustness: {wrong} model wrong -> {bad} biased", abs(b[bad]) > 0.10, f"{bad} bias {b[bad]:+.3f}")
r = ca.att_gt(two_period(0, wrong="outcome"), "id", "t", "y", "g", n_boot=99, covariates=["x"], show=False)
check("independent recomputation warns when outcome and propensity models disagree",
      any("Independent recomputation disagrees" in w for w in r["warnings"]), str(r["cross_check"].round(3).to_dict()))
r = ca.att_gt(panel(5), "id", "t", "y", "g", n_boot=99, covariates=["x"], x_periods="both", show=False)
check("no disagreement warning when the models are right",
      not any("disagrees" in w for w in r["warnings"]), str(r["cross_check"].round(3).to_dict()))
check("output contract: method and warnings in overall",
      "method" in r["overall"] and "warnings" in r["overall"] and r["overall"]["method"].startswith("doubly robust"))

# ---------------------------------------------------------------- 5. bootstrap coverage
cover = []
for s in range(50):
    o = ca.att_gt(panel(300 + s, n=400), "id", "t", "y", "g", n_boot=99, seed=s, covariates=["x"],
                  x_periods="both", verify=False, show=False)["overall"]
    cover.append(o["ci_low"] <= 1.0 <= o["ci_high"])
check("bootstrap 95% CI coverage of the DR ATT within 86-100% (50 reps)", 0.86 <= np.mean(cover) <= 1.0,
      f"{np.mean(cover):.0%}")

# ---------------------------------------------------------------- 6. overlap, errors, pass-through
rng = np.random.default_rng(4)
nn = 400
xs = np.r_[rng.uniform(-3, 0, nn - 50), rng.uniform(0, 3, 48), [2.5, 2.8]]
Ds = np.r_[np.zeros(nn - 50), np.ones(48), [0, 0]].astype(int)      # two controls deep in the treated region
sep = pd.DataFrame({"id": np.r_[np.arange(nn), np.arange(nn)], "t": np.r_[np.ones(nn), 2 * np.ones(nn)],
                    "y": rng.normal(size=2 * nn), "x": np.r_[xs, xs], "g": np.r_[2 * Ds, 2 * Ds]})
r = ca.att_gt(sep, "id", "t", "y", "g", n_boot=0, covariates=["x"], show=False)
check("near-separation: overlap warning (trimmed or weak overlap)",
      any("overlap" in w.lower() for w in r["warnings"]), r["warnings"][0][:70] if r["warnings"] else "")
miss = panel(2).copy()
miss.loc[5, "x"] = np.nan
try:
    ca.att_gt(miss, "id", "t", "y", "g", n_boot=0, covariates=["x"], show=False)
    check("missing covariate values raise a clear error", False)
except ValueError as e:
    check("missing covariate values raise a clear error", "covariates have missing" in str(e))
p6 = panel(6, n=300)
ts = ca.did_timing_sensitivity(p6, "id", "t", "y", "g", max_shift=1, n_boot=0, covariates=["x"], x_periods="both",
                               show=False)
base6 = ca.att_gt(p6, "id", "t", "y", "g", n_boot=0, covariates=["x"], x_periods="both", show=False)
check("did_timing_sensitivity passes the covariates (shift 0 = att_gt with covariates)",
      abs(ts["shifts"].loc[0, "att"] - base6["overall"]["estimate"]) < 1e-12)
lo = p6.groupby("id")["g"].transform("first")
ud = ca.did_uncertain_dates(p6.assign(gl=lo, gh=lo), "id", "t", "y", "gl", "gh", n_draws=3, n_boot=0,
                            covariates=["x"], x_periods="both", show=False)
check("did_uncertain_dates with covariates and exact dates = att_gt", abs(ud["draws_median"] - base6["overall"]["estimate"]) < 1e-12)
nocov = ca.att_gt(p6, "id", "t", "y", "g", n_boot=0, show=False)
manual = (p6[(p6.g == 3) & (p6.t == 3)].y.to_numpy() - p6[(p6.g == 3) & (p6.t == 2)].y.to_numpy()).mean() - \
         (p6[(p6.g == 0) & (p6.t == 3)].y.to_numpy() - p6[(p6.g == 0) & (p6.t == 2)].y.to_numpy()).mean()
check("without covariates att_gt is the unchanged difference of means", abs(nocov["gt"].loc[(3, 3), "att"] - manual) < 1e-12)


# ---------------------------------------------------------------- 7. ccc_test
def rcs(seed, kind, S=6, T=5, m=40, units=False):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(S):
        ids = np.arange(m) + 1000 * s
        xu = rng.normal(0.3 * s, 1, m)
        for t in range(1, T + 1):
            x = xu + rng.normal(0, 0.5, m) if units else rng.normal(0.3 * s, 1, m)
            b = 1.0 + (0.4 * t if kind in ("time", "two") else 0) + (0.5 * s if kind in ("group", "two") else 0)
            y = 0.2 * s + 0.1 * t + b * x + rng.normal(0, 1 + 0.3 * np.abs(x), m)
            rows.append(pd.DataFrame({"y": y, "x": x, "s": s, "t": t, "g": 0, "uid": ids}))
    return pd.concat(rows, ignore_index=True)


d7 = rcs(0, "time", m=120)
r = ca.ccc_test(d7, "y", ["x"], "s", "t", gvar="g", inference="asymptotic", alltests=True, show=False)
d7["c"] = d7["s"].astype(str) + "_" + d7["t"].astype(str)
f = smf.ols("y ~ C(c) + C(c):x - 1", d7).fit(cov_type="HC1")
names = [nm for nm in f.params.index if ":x" in nm]
idx = {nm: i for i, nm in enumerate(f.params.index)}
R = np.zeros((len(names) - 1, len(f.params)))
for i in range(1, len(names)):
    R[i - 1, idx[names[i]]], R[i - 1, idx[names[0]]] = 1, -1
w_sm = float(f.wald_test(R, scalar=True).statistic)
check("ccc stage 1 HC1 Wald = statsmodels cell-dummy regression", abs(r["table"]["stat"].iloc[0] - w_sm) < 1e-6 * w_sm,
      f"{r['table']['stat'].iloc[0]:.6f} vs {w_sm:.6f}")
f2 = smf.ols("y ~ C(c) + x + C(s):x + C(t):x - 1", d7).fit(cov_type="HC1")
Rg = np.zeros((5, len(f2.params)))
for i, nm in enumerate([n_ for n_ in f2.params.index if n_.startswith("C(s)") and ":x" in n_]):
    Rg[i, list(f2.params.index).index(nm)] = 1
wg = float(f2.wald_test(Rg, scalar=True).statistic)
check("ccc stage 2 group test = statsmodels additive model", abs(r["table"]["stat"].iloc[1] - wg) < 1e-6 * wg,
      f"{r['table']['stat'].iloc[1]:.6f} vs {wg:.6f}")
du = rcs(3, "none", S=8, T=4, m=30, units=True)
rc = ca.ccc_test(du, "y", ["x"], "s", "t", gvar="g", cluster="uid", inference="asymptotic", alltests=True, show=False)
du["c"] = du["s"].astype(str) + "_" + du["t"].astype(str)
fc = smf.ols("y ~ C(c) + x + C(s):x + C(t):x - 1", du).fit(cov_type="cluster", cov_kwds={"groups": du["uid"]})
Rt = np.zeros((3, len(fc.params)))
for i, nm in enumerate([n_ for n_ in fc.params.index if n_.startswith("C(t)") and ":x" in n_]):
    Rt[i, list(fc.params.index).index(nm)] = 1
wt_ = float(fc.wald_test(Rt, scalar=True).statistic)
check("ccc stage 2 CR1 (clustered by unit) = statsmodels cluster Wald", abs(rc["table"]["stat"].iloc[2] - wt_) < 1e-6 * wt_,
      f"{rc['table']['stat'].iloc[2]:.6f} vs {wt_:.6f}")
rej_a, rej_w = [], []
for k in range(120):
    dd = rcs(500 + k, "none")
    rej_a.append(ca.ccc_test(dd, "y", ["x"], "s", "t", gvar="g", inference="asymptotic", show=False)["table"]["p"].iloc[0] <= .05)
    rej_w.append(ca.ccc_test(dd, "y", ["x"], "s", "t", gvar="g", n_boot=199, seed=k, show=False)["table"]["p"].iloc[0] <= .05)
check("ccc asymptotic chi-square over-rejects with 29 restrictions (why wild is the default)", np.mean(rej_a) > 0.12,
      f"size {np.mean(rej_a):.1%}")
check("ccc wild bootstrap size near 5% (<= 9%, 120 reps)", np.mean(rej_w) <= 0.09, f"size {np.mean(rej_w):.1%}")
for kind, want in (("time", "time"), ("group", "group"), ("two", "twoway")):
    sel = [ca.ccc_test(rcs(800 + k, kind), "y", ["x"], "s", "t", gvar="g", n_boot=99, seed=k, show=False)["selected"]
           for k in range(25)]
    check(f"ccc selects '{want}' when slopes vary by {kind}", np.mean(np.array(sel) == want) >= 0.85,
          f"{np.mean(np.array(sel) == want):.0%}")
few = rcs(9, "none", S=3, T=3, m=20, units=True)
few["cl"] = few["s"]
rf = ca.ccc_test(few, "y", ["x"], "s", "t", gvar="g", cluster="cl", n_boot=99, alltests=True, show=False)
check("ccc falls back to the F statistic when restrictions exceed clusters (warned)",
      "F (classical)" in rf["table"]["type"].iloc[0] and any("classical F" in w for w in rf["warnings"]))
pp = panel(7, n=200)
try:
    ca.ccc_test(pp, "y", ["x"], "id", "t", gvar="g", show=False)
    check("ccc rejects one-observation cells (group = unit) with a clear error", False)
except ValueError as e:
    check("ccc rejects one-observation cells (group = unit) with a clear error", "coarser group" in str(e))
mix = rcs(10, "none", m=60)
mix["g"] = np.where(mix["s"] < 2, 4, 0)
mix["y"] = mix["y"] + 50 * ((mix["g"] > 0) & (mix["t"] >= 4))
rm_ = ca.ccc_test(mix, "y", ["x"], "s", "t", gvar="g", n_boot=99, show=False)
check("ccc uses only untreated observations (treated cells excluded)", rm_["n"] == len(mix) - 2 * 2 * 60,
      f"n = {rm_['n']}")

# ---------------------------------------------------------------- 8. TWFE diagnostic
r_bad = ca.twfe_covariate_diagnostic(panel(1), "id", "t", "y", "g", ["x"], n_boot=49, show=False)
check("TWFE diagnostic: hidden linearity detected and verdict unreliable",
      r_bad["hidden_linearity_p"] < 0.01 and r_bad["verdict"] == "unreliable",
      f"p = {r_bad['hidden_linearity_p']:.2g}, gap {r_bad['gap_in_se']:.1f} SE")
check("TWFE diagnostic: level coefficient estimates the per-period drift of the covariate effect (0.4)",
      abs(r_bad["level_coefficients"]["coef"].iloc[0] - 0.4) < 0.06, f"{r_bad['level_coefficients']['coef'].iloc[0]:.3f}")
rng = np.random.default_rng(3)
nb, Tb = 600, 6
gb = np.where(rng.random(nb) < .5, 4, 0)
ab, xb = rng.normal(0.5 * (gb > 0), 1), rng.normal(0.8 * (gb > 0), 1)
rows = []
for t in range(1, Tb + 1):
    xb = xb + rng.normal(0, .3, nb)
    rows.append(pd.DataFrame({"id": np.arange(nb), "t": t, "x": xb, "g": gb,
                              "y": ab + xb + 0.3 * t + ((gb > 0) & (t >= gb)) + rng.normal(0, 1, nb)}))
r_ok = ca.twfe_covariate_diagnostic(pd.concat(rows), "id", "t", "y", "g", ["x"], n_boot=49, show=False)
check("TWFE diagnostic: quiet in a benign design (constant covariate effect, one cohort)",
      r_ok["verdict"] == "no evidence against TWFE", f"p = {r_ok['hidden_linearity_p']:.2f}")
rng = np.random.default_rng(8)
rows = []
gs = np.repeat([2, 6], 100)
for t in range(1, 9):
    k = np.where(t >= gs, t - gs + 1, 0)
    rows.append(pd.DataFrame({"id": np.arange(200), "t": t, "g": gs, "x": rng.normal(size=200),
                              "y": 0.2 * t + 0.5 * k + rng.normal(size=200)}))
r_neg = ca.twfe_covariate_diagnostic(pd.concat(rows), "id", "t", "y", "g", ["x"], control="notyet", n_boot=0,
                                     show=False)
check("TWFE diagnostic: negative implicit weights with staggered adoption and no never-treated units",
      r_neg["negative_weight_share"] > 0, f"{r_neg['negative_weight_share']:.0%} of treated obs")
ti = panel(2, n=300)
ti["z"] = ti.groupby("id")["x"].transform("first")
r_ti = ca.twfe_covariate_diagnostic(ti, "id", "t", "y", "g", ["x", "z"], n_boot=0, show=False)
check("TWFE diagnostic warns that time-invariant covariates drop out of TWFE",
      any("absorbed by the unit fixed effects" in w for w in r_ti["warnings"]))

# ---------------------------------------------------------------- 9. manifest and cross-check
df9 = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})
with contextlib.redirect_stdout(io.StringIO()):
    m1 = rp.run_manifest({"d": df9}, seed=1)
    m2 = rp.run_manifest({"d": df9.copy()}, seed=1)
    m3 = rp.run_manifest({"d": df9.assign(a=[1.0, 2.0, 3.0000001])}, seed=1)
check("manifest: same data -> same hash; one changed value -> different hash",
      m1["data"]["d"]["sha256"] == m2["data"]["d"]["sha256"] != m3["data"]["d"]["sha256"])
with tempfile.TemporaryDirectory() as tmp:
    fp = Path(tmp) / "data.csv"
    df9.to_csv(fp, index=False)
    with contextlib.redirect_stdout(io.StringIO()):
        m4 = rp.run_manifest(files=[fp], seed=7, results={"ATT": base6["overall"]}, path=str(Path(tmp) / "m.json"))
    back = json.loads((Path(tmp) / "m.json").read_text(encoding="utf-8"))
check("manifest: file hash, versions, seed and results written to JSON",
      len(back["files"][str(fp)]) == 64 and back["packages"]["numpy"] and back["seed"] == 7
      and set(back["results"]["ATT"]) <= {"estimate", "se", "ci_low", "ci_high", "p", "n"}
      and back["skill_version"] == rp.SKILL_VERSION)
with contextlib.redirect_stdout(io.StringIO()):
    c1 = rp.cross_check({"dr": 1.00, "reg": 1.03}, se=0.1)
    c2 = rp.cross_check({"dr": 1.00, "reg": 1.30}, se=0.1)
    c3 = rp.cross_check({"statsmodels": 0.123456789, "mine": 0.123456789 + 1e-12})
check("cross_check: agree within 1 SE, disagree beyond it, exact agreement without SE",
      c1["agree"] and not c2["agree"] and c2["warnings"] and c3["agree"])

# ---------------------------------------------------------------- 10. instructions
skill = (ROOT / "econometrics" / "SKILL.md").read_text(encoding="utf-8")
rules = (ROOT / "econometrics" / "references" / "interpretation-rules.md").read_text(encoding="utf-8")
check("SKILL.md: SKILL_DIR defaults to the user's clone through ECONOMETRICS_SKILL_DIR, never a sandbox path",
      'ECONOMETRICS_SKILL_DIR' in skill and "/mnt/" not in skill and "/home/claude" not in skill)
check("rules: the Basel traffic light is stated for 99% VaR (2.5 expected), 97.5% expects 6.25",
      "**99%** VaR" in rules and "6.25" in rules)

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
