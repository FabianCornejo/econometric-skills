"""Tests for v2.5.1: DiD with one treated unit (did_single_treated), the few-treated warnings and ASK lines of
did_design, randomization_inference(cluster=), did_timing_sensitivity cohorts, did_falsification, and three
"find the error" routing cases.

Bands were fixed before this file was first run (claude/estudo-inferencia-uma-tratada-v251-2026-11-07.md in the
development repository holds the exploratory study that chose the route). Monte Carlo error with R = 2000 at 5% is
about ±1 point, so a size band of [0.03, 0.075] separates a correct test from one that doubles its size.

1. Size of the main route (calibrated AR(1) t) under no effect: T = 24 with ρ = 0, 0.4, 0.7, T = 60 with ρ = 0.4,
   a short pre-period (T0 = 6), and the DGP of the 2026-11-07 validation (two stores, 24 months; copied by script
   from recheck_v2502.py, not retyped).
2. The control: HAC with 2 lags (statsmodels, the agent's choice in that validation) on the same gaps over-rejects;
   the test must see it (≥ 14%), or the size checks above prove nothing.
3. Coverage and power with a planted effect.
4. Pieces checked by another path: the lag weights against a Toeplitz quadratic form; the AR(1) variance of the
   contrast against a brute-force Monte Carlo variance; the bias correction of ρ; the point estimate against the
   2×2 DiD of means from groupby; the second route against scipy's Welch t test on block means.
5. Pre-period checks (trend test size and power; the step placebo is weak against a trend, so the trend test is
   the one tested for power), warnings only where they belong (persistent gap, failed pre-trend test), errors on unusable input, list of treated units.
6. did_design: the few-treated warning and the ASK THE USER lines appear in canonical and staggered designs
   and NOT with many treated units or without a gvar (dose, reversals); att_gt warns with one treated unit.
7. randomization_inference(cluster=): permuting counties over-rejects with state-level assignment, permuting
   states does not; counts of assignments; cluster = unit reproduces the unit-level result; error when gvar varies.
8. did_timing_sensitivity: dropped cohorts by their original dates; the estimand note.
9. did_falsification: findings by kind (a falsification row rejects; an inference row contradicts the headline),
   planted pre-trend and moved placebo outcome found, uninformative placebo flagged, gap-series rows with one
   treated unit.
10. The three new "find the error" cases: valid YAML graders that match a correct review and not the request.
11. After the independent review: controls= validated; degenerate gaps rejected; minimum segment lengths; date
    spacing; short-series warning; Ljung-Box false-alarm rate; unit-root flag on random walks; two treated units;
    did_falsification with treated units at two dates, anticipation in the gap-series branch, headline errors;
    the review-no-placebo grader against a wrong review.

Run from the repository root:  python3 tests/test_v251.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg, stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "econometrics" / "scripts"))
import _console  # noqa: E402,F401
import causal as ca  # noqa: E402

warnings.filterwarnings("ignore")
ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


def ar1(rng, R, T, rho):
    """Stationary AR(1) written here (unit innovation variance), one series per row."""
    z = rng.standard_normal((R, T))
    x = np.empty((R, T))
    x[:, 0] = z[:, 0] / np.sqrt(1 - rho ** 2)
    for t in range(1, T):
        x[:, t] = rho * x[:, t - 1] + z[:, t]
    return x


def panel(gap, t0_index=None, effect=0.0):
    """Long panel with a treated unit A whose outcome is the gap and a control B at zero."""
    T = len(gap)
    tt = np.arange(1, T + 1)
    yA = gap + (effect * (np.arange(T) >= t0_index) if t0_index is not None else 0.0)
    return pd.DataFrame({"u": ["A"] * T + ["B"] * T, "t": np.r_[tt, tt], "y": np.r_[yA, np.zeros(T)]})


def size_main(rng, R, T, T0, rho, n_sim=999):
    d = ar1(rng, R, T, rho)
    return np.mean([ca._sts_calibrated(d[i], T0, 0.05, n_sim, rng)["p"] <= 0.05 for i in range(R)])


SAZ = np.tile([-.06, -.08, -.02, -.03, 0, -.01, .01, 0, -.02, .01, .05, .15], 2)


def v1_dgp(seed, efeito=0.08):          # copied by script from recheck_v2502.py
    rng = np.random.default_rng(seed); linhas = []
    for loja, base in (("A", 100_000), ("B", 80_000)):
        e = np.zeros(24)
        for t in range(24):
            e[t] = (0.4 * e[t - 1] if t else 0) + rng.normal(0, 0.035)
        promo = (loja == "A") & (np.arange(24) >= 12)
        logv = np.log(base) + 0.005 * np.arange(24) + SAZ + e + efeito * promo
        linhas += [{"t": t + 1, "loja": loja, "vendas": round(float(np.exp(v)), 2)} for t, v in enumerate(logv)]
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------- 1. size of the main route
rng = np.random.default_rng(20261108)
R = 2000
for T, T0, rho, hi in ((24, 12, 0.0, 0.075), (24, 12, 0.4, 0.075), (24, 12, 0.7, 0.09), (60, 30, 0.4, 0.075),
                       (24, 6, 0.4, 0.075)):
    s = size_main(rng, R, T, T0, rho)
    check(f"size of the main route, T = {T}, T0 = {T0}, AR(1) ρ = {rho}: in [0.03, {hi}]", 0.03 <= s <= hi,
          f"size {s:.3f} (R = {R})")

# the validation DGP of 2026-11-07 (two stores, log sales, 24 months)
rej_m = rej_hac = 0
import statsmodels.api as sm  # noqa: E402
Rv = 1000
D = (np.arange(1, 25) >= 13).astype(float)
for s in range(Rv):
    df0 = v1_dgp(1000 + s, efeito=0.0)
    df0["ly"] = np.log(df0["vendas"])
    r = ca.did_single_treated(df0, "loja", "t", "ly", "A", 13, n_sim=999, seed=s, show=False)
    rej_m += r["p"] <= 0.05
    m = df0.pivot(index="t", columns="loja", values="ly")
    hac = sm.OLS((m["A"] - m["B"]).to_numpy(), sm.add_constant(D)).fit(cov_type="HAC", cov_kwds={"maxlags": 2},
                                                                        use_t=True)
    rej_hac += hac.pvalues[1] <= 0.05
check("size on the 2026-11-07 validation DGP (did_single_treated, full function): in [0.03, 0.08]",
      0.03 <= rej_m / Rv <= 0.08, f"size {rej_m / Rv:.3f} (R = {Rv})")

# ---------------------------------------------------------------- 2. the control must over-reject
check("control: HAC(2) on the same gaps over-rejects (≥ 0.14), as measured in the validation (17.8%)",
      rej_hac / Rv >= 0.14, f"HAC size {rej_hac / Rv:.3f}")

# ---------------------------------------------------------------- 3. coverage and power
d = ar1(rng, R, 24, 0.4)
cov = pw = 0
for i in range(R):
    r1 = ca._sts_calibrated(d[i] + 0.5 * (np.arange(24) >= 12), 12, 0.05, 999, rng)
    cov += r1["ci_low"] <= 0.5 <= r1["ci_high"]
    pw += ca._sts_calibrated(d[i] + 2.0 * (np.arange(24) >= 12), 12, 0.05, 999, rng)["p"] <= 0.05
check("coverage of the 95% CI, T = 24, ρ = 0.4: in [0.925, 0.97]", 0.925 <= cov / R <= 0.97, f"{cov / R:.3f}")
check("power against an effect of 2 innovation SDs, T = 24, ρ = 0.4: ≥ 0.6", pw / R >= 0.6, f"{pw / R:.3f}")
v1 = v1_dgp(20261107)
v1["ly"] = np.log(v1["vendas"])
rv = ca.did_single_treated(v1, "loja", "t", "ly", "A", 13, show=False)
check("validation data (seed 20261107): estimate = the 2×2 DiD of means recorded there (0.0618888)",
      abs(rv["estimate"] - 0.0618888) < 1e-6, f"{rv['estimate']:.7f}")
check("validation data: the CI is wider than the agent's HAC CI (3.0% to 9.9%)",
      rv["ci_high"] - rv["ci_low"] > 0.099 - 0.030, f"{rv['ci_low']:.4f} to {rv['ci_high']:.4f}")

# ---------------------------------------------------------------- 4. pieces by another path
for T0, T1 in ((12, 12), (6, 18), (24, 12)):
    w = np.r_[-np.ones(T0) / T0, np.ones(T1) / T1]
    rho = 0.55
    G = linalg.toeplitz(rho ** np.arange(T0 + T1)) / (1 - rho ** 2)
    c = ca._sts_lagweights(w)
    check(f"lag weights ({T0}, {T1}): Σ c_k ρ^k/(1−ρ²) = w'Γw from a Toeplitz matrix",
          abs(np.polyval(c[::-1], rho) / (1 - rho ** 2) - w @ G @ w) < 1e-12)
x = ar1(np.random.default_rng(7), 200_000, 24, 0.4)
mc_var = np.var(x[:, 12:].mean(1) - x[:, :12].mean(1))
c = ca._sts_lagweights(np.r_[-np.ones(12) / 12, np.ones(12) / 12])
an_var = np.polyval(c[::-1], 0.4) / (1 - 0.4 ** 2)
check("AR(1) variance of the contrast: formula vs Monte Carlo (200 000 series), within 1.5%",
      abs(an_var / mc_var - 1) < 0.015, f"{an_var:.4f} vs {mc_var:.4f}")
x = ar1(np.random.default_rng(8), 4000, 24, 0.4)
X, w, br = ca._sts_design(24, 12)
_, _, r_corr, _, _ = ca._sts_general(x, X, w, br)
u = np.concatenate([x[:, :12] - x[:, :12].mean(1, keepdims=True), x[:, 12:] - x[:, 12:].mean(1, keepdims=True)], 1)
r_raw = np.array([(np.r_[ui[1:12] * ui[:11], ui[13:] * ui[12:-1]]).sum() / (np.r_[ui[:11] ** 2, ui[12:-1] ** 2]).sum()
                  for ui in u])
check("bias of ρ (step design): raw within-segment estimate is biased down (median < 0.30) and the "
      "median-unbiased one has its median within ±0.04 of 0.4", np.median(r_raw) < 0.30
      and abs(np.median(r_corr) - 0.4) < 0.04, f"raw {np.median(r_raw):.3f}, corrected {np.median(r_corr):.3f}")
Xt12, wt12, brt12 = ca._sts_design(12, None)
_, _, r_tr, _, _ = ca._sts_general(ar1(np.random.default_rng(9), 4000, 12, 0.4), Xt12, wt12, brt12)
check("bias of ρ (trend design, 12 periods): median of the median-unbiased ρ within ±0.05 of 0.4",
      abs(np.median(r_tr) - 0.4) < 0.05, f"{np.median(r_tr):.3f}")
xt = x[:, :12]
Xt, wt, brt = ca._sts_design(12, None)
bt, _, _, _, _ = ca._sts_general(xt + 0.3 * np.arange(12), Xt, wt, brt)
ols = np.polyfit(np.arange(12), (xt + 0.3 * np.arange(12)).T, 1)[0]
check("trend design: the contrast is the OLS slope (numpy polyfit)", np.allclose(bt, ols, atol=1e-12))
pdf = panel(ar1(np.random.default_rng(3), 1, 20, 0.3)[0], 8, effect=1.0)
rr = ca.did_single_treated(pdf, "u", "t", "y", "A", 9, show=False)
gm = pdf.assign(post=pdf.t >= 9).groupby(["u", "post"]).y.mean()
did = (gm["A", True] - gm["A", False]) - (gm["B", True] - gm["B", False])
check("point estimate = 2×2 DiD of group means (groupby)", abs(rr["estimate"] - did) < 1e-12)
g = rr["gap"].to_numpy()
pre_b = [c.mean() for c in np.array_split(g[:8], 4)]
post_b = [c.mean() for c in np.array_split(g[8:], 4)]
tt = stats.ttest_ind(post_b, pre_b, equal_var=False)
check("second route = scipy Welch t test on 4 block means per segment", abs(rr["blocks"]["p"] - tt.pvalue) < 1e-12)
r_a = ca.did_single_treated(pdf, "u", "t", "y", "A", 9, show=False, seed=5)
r_b = ca.did_single_treated(pdf, "u", "t", "y", "A", 9, show=False, seed=5)
check("same seed, same p and CI", r_a["p"] == r_b["p"] and r_a["ci_low"] == r_b["ci_low"])

# ---------------------------------------------------------------- 5. warnings, placebo, errors, lists
rng5 = np.random.default_rng(55)
pers = [ca.did_single_treated(panel(ar1(rng5, 1, 24, 0.85)[0]), "u", "t", "y", "A", 13, n_sim=499, show=False)
        for _ in range(40)]
flat = [ca.did_single_treated(panel(ar1(rng5, 1, 24, 0.0)[0]), "u", "t", "y", "A", 13, n_sim=499, show=False)
        for _ in range(40)]
has = lambda r, s: any(s in m for m in r["warnings"])
check("persistence warning: in most series with ρ = 0.85 (≥ 60%) and rare with ρ = 0 (≤ 10%)",
      np.mean([has(r, "persistent") or has(r, "unit root") for r in pers]) >= 0.6
      and np.mean([has(r, "persistent") for r in flat]) <= 0.10)
# a planted linear pre-trend (0.35 SD per period over the pre periods, flat afterwards), no effect
trend_gap = lambda: ar1(rng5, 1, 24, 0.0)[0] + 0.35 * np.minimum(np.arange(24), 12)
tr_rej = np.mean([ca.did_single_treated(panel(trend_gap()), "u", "t", "y", "A", 13, n_sim=499,
                                        show=False)["pre_trend"]["p"] <= 0.05 for _ in range(200)])
nulls = [ca.did_single_treated(panel(ar1(rng5, 1, 40, 0.4)[0]), "u", "t", "y", "A", 21, n_sim=499, show=False)
         for _ in range(500)]
tr_null = np.mean([r["pre_trend"]["p"] <= 0.05 for r in nulls])
pl_null = np.mean([r["pre_placebo"]["p"] <= 0.05 for r in nulls])
check("pre-period trend test rejects a planted pre-trend (power ≥ 0.6, the criterion of the main route; 200 series)", tr_rej >= 0.6, f"{tr_rej:.3f}")
check("pre-period trend test and step placebo with 20 pre periods: size under AR(1) ρ = 0.4 in [0.02, 0.08] "
      "(500 series)", 0.02 <= tr_null <= 0.08 and 0.02 <= pl_null <= 0.08, f"trend {tr_null:.3f}, step {pl_null:.3f}")
X12, w12, b12 = ca._sts_design(12, None)
d12 = ar1(rng5, 1000, 12, 0.4)
s12 = np.mean([ca._sts_calibrated(d12[i], None, 0.05, 499, rng5)["p"] <= 0.05 for i in range(1000)])
check("pre-period trend test at its minimum (12 pre periods), ρ = 0.4: size in [0.02, 0.10], the 7-9% documented "
      "(band set after the independent review measured 9%)", 0.02 <= s12 <= 0.10, f"{s12:.3f}")
check("a failed pre-trend test produces its warning, and a passed one does not",
      all(has(r, "trends before treatment") == (r["pre_trend"]["p"] <= 0.05) for r in nulls))
short = ca.did_single_treated(panel(ar1(rng5, 1, 12, 0.0)[0]), "u", "t", "y", "A", 7, show=False)
check("no step placebo with fewer than 16 pre periods and no trend test with fewer than 12",
      short["pre_placebo"] is None and short["pre_trend"] is None and nulls[0]["pre_placebo"] is not None)
for bad, kw in (("unbalanced", dict(df=panel(np.zeros(24)).drop(index=3), t0=13)),
                ("too few post periods", dict(df=panel(np.zeros(24)), t0=23)),
                ("four pre periods", dict(df=panel(np.arange(24.0) % 3), t0=5)),
                ("constant gap", dict(df=panel(np.ones(24)), t0=13)),
                ("pure step gap", dict(df=panel(1.0 * (np.arange(24) >= 12)), t0=13)),
                ("unknown treated unit", dict(df=panel(np.zeros(24)), t0=13, treated="Z"))):
    try:
        ca.did_single_treated(kw["df"], "u", "t", "y", kw.get("treated", "A"), kw["t0"], show=False)
        check(f"error on {bad} input", False)
    except ValueError:
        check(f"error on {bad} input", True)
e_ = np.random.default_rng(16).standard_normal(24)
two = pd.concat([panel(e_), panel(e_).query("u == 'A'").assign(u="C")])
two.loc[(two.u == "C") & (two.t >= 13), "y"] += 2.0
rt = ca.did_single_treated(two, "u", "t", "y", ["A", "C"], 13, show=False)
check("list of treated units: their outcomes are averaged (effect 2 on one of two treated → DiD of the noise + 1)",
      abs(rt["estimate"] - (e_[12:].mean() - e_[:12].mean() + 1.0)) < 1e-12 and rt["treated"] == ["A", "C"])
check("output contract keys", all(k in rv for k in ("estimand", "estimate", "se", "ci_low", "ci_high", "p", "n",
                                                    "method", "warnings")))

# ---------------------------------------------------------------- 6. did_design and att_gt (front B and the few-treated warning)
def design_panel(n_units, treated_dates, T=12):
    """Binary absorbing panel: unit i treated from treated_dates[i] (0 = never)."""
    rows = []
    for i in range(n_units):
        g = treated_dates[i] if i < len(treated_dates) else 0
        for t in range(1, T + 1):
            rows.append({"u": i, "t": t, "d": int(g > 0 and t >= g), "y": 0.1 * t + (g > 0 and t >= g)})
    return pd.DataFrame(rows)


one = ca.did_design(design_panel(2, [7]), "u", "t", "d", show=False)
many = ca.did_design(design_panel(40, [7] * 10 + [9] * 10), "u", "t", "d", show=False)
dose = design_panel(6, [7, 7])
dose["d"] = dose["d"] * (1 + dose["u"])
dose_r = ca.did_design(dose, "u", "t", "d", show=False)
rev = design_panel(6, [5, 5])
rev.loc[(rev.u == 0) & (rev.t >= 9), "d"] = 0
rev_r = ca.did_design(rev, "u", "t", "d", show=False)
few = lambda r: any("did_single_treated" in m for m in r["warnings"])
check("did_design: one treated unit → the few-treated warning, did_single_treated first in the estimators, "
      "smallest p 1/2 stated", few(one) and one["estimators"]["skill"][0].startswith("causal.did_single_treated")
      and any("1/2" in m for m in one["warnings"]))
check("did_design: 20 treated units → no few-treated warning", not few(many)
      and not any("did_single_treated" in e for e in many["estimators"]["skill"]))
check("did_design: ASK lines (date, anticipation, gradual, spillovers) in canonical and staggered designs",
      len(one["ask"]) == 4 and len(many["ask"]) == 4 and any("announced" in a for a in one["ask"]))
check("did_design: no ASK lines and no few-treated warning when there is no gvar (dose, reversals)",
      dose_r["ask"] == [] and rev_r["ask"] == [] and not few(dose_r) and not few(rev_r)
      and dose_r["gvar"] is None and rev_r["gvar"] is None)
import io, contextlib  # noqa: E402
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ca.did_design(design_panel(2, [7]), "u", "t", "d", show=True)
check("did_design prints the ASK THE USER lines", buf.getvalue().count("ASK THE USER:") == 4)
p1 = design_panel(8, [7])
p1["g"] = np.where(p1.groupby("u")["d"].transform("max") > 0, 7, 0)
a1 = ca.att_gt(p1, "u", "t", "y", "g", n_boot=19, show=False)
p8 = design_panel(16, [7] * 8)
p8["g"] = np.where(p8.groupby("u")["d"].transform("max") > 0, 7, 0)
a8 = ca.att_gt(p8, "u", "t", "y", "g", n_boot=19, show=False)
check("att_gt: one treated unit → warning pointing to did_single_treated; eight treated → none",
      any("did_single_treated" in m for m in a1["warnings"]) and not any("did_single_treated" in m for m in a8["warnings"]))

# ---------------------------------------------------------------- 7. randomization_inference(cluster=) (front C)
def state_panel(rng, n_states=12, n_treated=4, counties=5, T=8, g=5, effect=0.0):
    """Treatment assigned by state; state-by-period shocks (sd 1) shared by its counties, county noise sd 0.5."""
    rows = []
    for s_ in range(n_states):
        shock = rng.standard_normal(T)
        gs = g if s_ < n_treated else 0
        for c_ in range(counties):
            a = rng.standard_normal()
            for t in range(1, T + 1):
                rows.append({"county": f"{s_}-{c_}", "state": s_, "t": t, "g": gs,
                             "y": a + shock[t - 1] + 0.5 * rng.standard_normal() + effect * (gs > 0 and t >= gs)})
    return pd.DataFrame(rows)


rng7 = np.random.default_rng(77)
rc = rs_ = 0
for rep in range(200):
    sp = state_panel(rng7)
    rc += ca.randomization_inference(sp, "county", "t", "y", "g", n_perm=199, ci=False, seed=rep)["p"] <= 0.05
    rs_ += ca.randomization_inference(sp, "county", "t", "y", "g", n_perm=199, ci=False, seed=rep,
                                      cluster="state")["p"] <= 0.05
check("permutation with state-level assignment and state shocks: permuting counties over-rejects (≥ 0.15), "
      "permuting states (cluster=) does not (≤ 0.09); 200 panels", rc / 200 >= 0.15 and rs_ / 200 <= 0.09,
      f"counties {rc / 200:.3f}, states {rs_ / 200:.3f}")
sp = state_panel(np.random.default_rng(1))
r_st = ca.randomization_inference(sp, "county", "t", "y", "g", n_perm=999, ci=False, seed=0, cluster="state")
from math import comb  # noqa: E402
check("cluster=: assignments counted over clusters (C(12, 4) = 495, exact)", r_st["possible_assignments"] == comb(12, 4)
      and r_st["n_units"] == 12 and r_st["n_treated"] == 4 and r_st["exact"])
sp["unit_as_cluster"] = sp["county"]
r_u = ca.randomization_inference(sp, "county", "t", "y", "g", n_perm=199, ci=False, seed=3)
r_c = ca.randomization_inference(sp, "county", "t", "y", "g", n_perm=199, ci=False, seed=3, cluster="unit_as_cluster")
check("cluster equal to the unit gives the unit-level result", r_u["p"] == r_c["p"] and r_u["estimate"] == r_c["estimate"])
bad = sp.copy()
bad.loc[bad.county == "0-0", "g"] = 6
try:
    ca.randomization_inference(bad, "county", "t", "y", "g", n_perm=99, ci=False, cluster="state")
    check("cluster=: error when gvar varies within a cluster", False)
except ValueError:
    check("cluster=: error when gvar varies within a cluster", True)

# ---------------------------------------------------------------- 8. did_timing_sensitivity: original dates (front C)
rng8 = np.random.default_rng(8)
rows = []
for i in range(30):
    g = 2 if i < 8 else (7 if i < 16 else 0)
    for t in range(1, 11):
        rows.append({"u": i, "t": t, "g": g, "y": 0.1 * t + (g > 0 and t >= g) + rng8.standard_normal()})
tp = pd.DataFrame(rows)
ts = ca.did_timing_sensitivity(tp, "u", "t", "y", "g", max_shift=1, n_boot=49, show=False)
check("timing: the cohort with no date-robust base period (g = 2: base 2 + 1 − 2 − 1 = 0, outside the panel) is "
      "reported by its original date 2, not 3; the shifted dates are kept apart",
      list(ts["cohorts_dropped_robust"]) == [2] and list(ts["cohorts_dropped_robust_shifted"]) == [3],
      f"original {ts['cohorts_dropped_robust']}, shifted {ts['cohorts_dropped_robust_shifted']}")
check("timing: the estimand note says the date-robust ATT is a later-exposure ATT",
      "later-exposure" in ts["estimand_note"])

# ---------------------------------------------------------------- 9. did_falsification (front C)
def stag(rng, effect=1.0, pre_trend=0.0, placebo_effect=0.0, n=40, T=12):
    rows = []
    for i in range(n):
        g = 7 if i < n // 4 else (9 if i < n // 2 else 0)
        a = rng.standard_normal()
        for t in range(1, T + 1):
            on = g > 0 and t >= g
            rows.append({"u": i, "t": t, "g": g,
                         "y": a + 0.1 * t + effect * on + pre_trend * (g > 0) * t + rng.standard_normal(),
                         "z": rng.standard_normal() + placebo_effect * on})
    return pd.DataFrame(rows)


rng9 = np.random.default_rng(9)
clean = ca.did_falsification(stag(rng9), "u", "t", "y", "g", placebo_outcomes=["z"], n_boot=99, n_perm=199,
                             show=False)
tabc = clean["table"]
check("falsification, clean design with a true effect: the permutation agrees with the headline and is not a "
      "finding (inference rows are findings only when they contradict it)",
      bool(tabc.loc["permutation", "rejects"]) and clean["headline"]["p"] <= 0.05
      and not bool(tabc.loc["permutation", "finding"]))
check("falsification, clean design: no falsification row is a finding",
      not any(tabc.loc[tabc["kind"] == "falsification", "finding"].fillna(False)), clean["verdict"][:90])
check("falsification: the verdict never says 'passed'", "passed" not in clean["verdict"].lower())
trended = ca.did_falsification(stag(rng9, pre_trend=0.4), "u", "t", "y", "g", n_boot=99, n_perm=199, show=False)
check("falsification, planted pre-trend in the treated: a pre-trend row is a finding",
      any(c.startswith("pre-trends") for c in trended["rejected"]), str(trended["rejected"]))
moved = ca.did_falsification(stag(rng9, placebo_effect=1.5), "u", "t", "y", "g", placebo_outcomes=["z"], n_boot=99,
                             n_perm=199, show=False)
check("falsification: a placebo outcome moved by the treatment is a finding", "placebo outcome: z" in moved["rejected"])
noisy = stag(rng9)
noisy["w"] = 50 * np.random.default_rng(5).standard_normal(len(noisy))
nz = ca.did_falsification(noisy, "u", "t", "y", "g", placebo_outcomes=["w"], n_boot=99, n_perm=199, show=False)
check("falsification: a placebo outcome with MDE ≥ |ATT| is marked uninformative, not as evidence",
      "placebo outcome: w" in nz["uninformative"] or "placebo outcome: w" in nz["rejected"])
T = 24
one = pd.DataFrame({"u": ["A"] * T + ["B"] * T, "t": list(range(1, T + 1)) * 2,
                    "y": np.random.default_rng(10).standard_normal(2 * T)})
one["g"] = np.where(one.u == "A", 13, 0)
fo = ca.did_falsification(one, "u", "t", "y", "g", show=False)
ds = ca.did_single_treated(one, "u", "t", "y", ["A"], 13, seed=0, show=False)
check("falsification with one treated unit: gap-series rows; unit-level rows left out with a reason",
      fo["few_treated"] and not fo["table"].loc["permutation", "ran"]
      and "assignments" in fo["table"].loc["permutation", "note"]
      and not fo["table"].loc["pre-trends (Callaway-Sant'Anna)", "ran"])
check("falsification with one treated unit: the trend row's MDE is the slope MDE × (T0 + T1)/2",
      abs(fo["table"].loc["pre-period trend (gap series)", "mde"] - ds["pre_trend"]["mde"] * 12) < 1e-9)

# ---------------------------------------------------------------- 10. "find the error" cases (evals/routing)
import re  # noqa: E402
import yaml  # noqa: E402
cases = {
    "review-single-store-hac": "Com uma única loja tratada e 24 meses, o HAC com 2 defasagens rejeita demais: use "
                               "causal.did_single_treated, que calibra a inferência na série da diferença.",
    "review-no-placebo": "Não rejeitar o teste de pré-tendência não prova tendências paralelas; rode placebos "
                         "(causal.did_falsification) e informe o MDE.",
    "review-cluster-wrong-level": "A lei é atribuída por estado: os erros-padrão devem ser clusterizados por estado, "
                                  "com só seis estados use wild bootstrap e randomization_inference(cluster='estado').",
}
for name, good in cases.items():
    folder = ROOT / "econometrics" / "evals" / "routing" / name
    g = yaml.safe_load((folder / "graders" / "answer.md").read_text(encoding="utf-8").split("---")[1])
    text = (folder / "prompt.md").read_text(encoding="utf-8")
    meta_p = yaml.safe_load(text.split("---")[1])
    body = text.split("---", 2)[2]
    rx = re.compile(g["pattern"], re.I)
    check(f"case {name}: the answer grader is valid YAML, matches a correct review and not the request itself",
          g["type"] == "regex" and bool(rx.search(good)) and not rx.search(body)
          and meta_p["allowed_tools"] == ["Read", "Glob", "Grep", "Skill"]
          and (folder / "graders" / "skill-fired.md").exists())

# ---------------------------------------------------------------- 11. after the independent review
pc = panel(ar1(np.random.default_rng(12), 1, 24, 0.2)[0])
pc = pd.concat([pc, pc[pc.u == "B"].assign(u="C", y=lambda x: x.y + 0.1)])
for bad, ctl in (("absent control", ["Z"]), ("control that is treated", ["A", "B"]), ("control = treated", ["A"])):
    try:
        ca.did_single_treated(pc, "u", "t", "y", "A", 13, controls=ctl, show=False)
        check(f"controls= validated: {bad} → ValueError", False)
    except ValueError:
        check(f"controls= validated: {bad} → ValueError", True)
rc_ = ca.did_single_treated(pc, "u", "t", "y", "A", 13, controls=["C"], show=False)
check("controls=['C'] uses C only (gap = A − C)", np.allclose(rc_["gap"].to_numpy(),
      pc[pc.u == "A"].y.to_numpy() - pc[pc.u == "C"].y.to_numpy()) and rc_["controls"] == ["C"])
dt = panel(ar1(np.random.default_rng(13), 1, 24, 0.2)[0])
months = pd.date_range("2024-01-01", periods=26, freq="MS").delete([6, 7])
dt["t"] = np.r_[months, months]
r_dt = ca.did_single_treated(dt, "u", "t", "y", "A", months[12], show=False)
dt2 = dt.copy()
reg = pd.date_range("2024-01-01", periods=24, freq="MS")
dt2["t"] = np.r_[reg, reg]
r_dt2 = ca.did_single_treated(dt2, "u", "t", "y", "A", reg[12], show=False)
check("monthly dates with a two-month hole are warned; regular monthly dates are not",
      any("regular frequency" in m for m in r_dt["warnings"]) and not any("regular frequency" in m for m in r_dt2["warnings"]))
r12 = ca.did_single_treated(panel(ar1(np.random.default_rng(14), 1, 12, 0.2)[0]), "u", "t", "y", "A", 7, show=False)
check("short-series warning with 12 periods, none with 24", any("Only 12 periods" in m for m in r12["warnings"])
      and not any("periods: the calibrated" in m for m in rv["warnings"]))
wn = [ca.did_single_treated(panel(ar1(np.random.default_rng(1000 + i), 1, 24, 0.0)[0]), "u", "t", "y", "A", 13,
                            n_sim=199, show=False)["ljung_box"]["p"] <= 0.05 for i in range(300)]
check("Ljung-Box false-alarm rate on white-noise gaps in [0.01, 0.10] (300 series)", 0.01 <= np.mean(wn) <= 0.10,
      f"{np.mean(wn):.3f}")
rw = [ca.did_single_treated(panel(np.cumsum(np.random.default_rng(2000 + i).standard_normal(24))), "u", "t", "y", "A",
                            13, n_sim=199, show=False) for i in range(100)]
check("random-walk gaps: the unit-root warning appears in at least 20% of 100 series (it relies on the clip flag)",
      np.mean([has(r, "unit root") for r in rw]) >= 0.2, f"{np.mean([has(r, 'unit root') for r in rw]):.2f}")
two_same = ca.did_design(design_panel(6, [7, 7]), "u", "t", "d", show=False)
check("did_design: two treated units at one date also get the few-treated warning", few(two_same))
t2 = design_panel(4, [7, 7], T=24)
t2["g"] = np.where(t2.groupby("u")["d"].transform("max") > 0, 7, 0)
t2["y"] = t2["y"] + np.random.default_rng(17).standard_normal(len(t2))
f2 = ca.did_falsification(t2, "u", "t", "y", "g", show=False)
check("did_falsification: two treated units at one date use the gap-series branch", f2["few_treated"])
t3 = design_panel(10, [6, 8], T=12)
t3["y"] = t3["y"] + np.random.default_rng(15).standard_normal(len(t3))
t3["g"] = np.where(t3.u == 0, 6, np.where(t3.u == 1, 8, 0))
f3 = ca.did_falsification(t3, "u", "t", "y", "g", n_boot=49, n_perm=99, show=False)
check("did_falsification: two treated units at two dates → warning that the unit-level rows are unreliable, and "
      "att_gt's few-treated warning passed on", any("descriptive only" in m for m in f3["warnings"])
      and any(m.startswith("att_gt:") and "did_single_treated" in m for m in f3["warnings"]))
one_a = one.copy()
fa = ca.did_falsification(one_a, "u", "t", "y", "g", anticipation=2, show=False)
da = ca.did_single_treated(one_a, "u", "t", "y", ["A"], 11, seed=0, show=False)
check("did_falsification, one treated unit, anticipation = 2: headline = did_single_treated with t0 moved back 2",
      abs(fa["headline"]["estimate"] - da["estimate"]) < 1e-12 and "timing (±1)" in fa["table"].index
      and not fa["table"].loc["timing (±1)", "ran"])
early = one.copy()
early["g"] = np.where(early.u == "A", 3, 0)
try:
    ca.did_falsification(early, "u", "t", "y", "g", show=False)
    check("did_falsification: a headline that cannot be computed raises a clear ValueError", False)
except ValueError as e:
    check("did_falsification: a headline that cannot be computed raises a clear ValueError",
          "headline estimate" in str(e))
gr = ROOT / "econometrics" / "evals" / "routing" / "review-no-placebo" / "graders" / "answer.md"
rxp = re.compile(yaml.safe_load(gr.read_text(encoding="utf-8").split("---")[1])["pattern"], re.I)
wrong = ("Concordo com o texto: o p conjunto de 0,41 confirma as tendências paralelas, então o efeito de 4,2 pontos é "
         "causal e não é preciso fazer mais nada. Podemos seguir para a conclusão.")
check("review-no-placebo grader: does not match a wrong review that agrees with the authors", not rxp.search(wrong))

from statsmodels.stats.diagnostic import acorr_ljungbox  # noqa: E402
e1 = ar1(np.random.default_rng(18), 1, 30, 0.3)[0]
e1 = e1 - e1.mean()
q_s, p_s = ca._sts_ljung_box(e1, len(e1) + 1, 5)        # one segment holds every innovation
lb_sm = acorr_ljungbox(e1, lags=[5], model_df=1, return_df=True).iloc[0]
check("Ljung-Box on one segment = statsmodels acorr_ljungbox(model_df=1): Q and p",
      abs(q_s - lb_sm["lb_stat"]) < 1e-9 and abs(p_s - lb_sm["lb_pvalue"]) < 1e-12, f"Q {q_s:.4f} vs {lb_sm['lb_stat']:.4f}")

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
