"""Tests for v2.5.1, part A: causal.did_single_treated (DiD with one treated unit and a short series).

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
6. did_design: the few-treated warning and the ASK THE USER lines appear in canonical and staggered designs
   and NOT with many treated units or without a gvar (dose, reversals); att_gt warns with one treated unit.
5. Pre-period checks (trend test size and power; the step placebo is weak against a trend, so the trend test is
   the one tested for power), warnings only where they belong (persistent gap, failed pre-trend test), errors on unusable input, list of treated units.

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
nulls = [ca.did_single_treated(panel(ar1(rng5, 1, 24, 0.4)[0]), "u", "t", "y", "A", 13, n_sim=499, show=False)
         for _ in range(400)]
tr_null = np.mean([r["pre_trend"]["p"] <= 0.05 for r in nulls])
pl_null = np.mean([r["pre_placebo"]["p"] <= 0.05 for r in nulls])
check("pre-period trend test rejects a planted pre-trend (power ≥ 0.6, the criterion of the main route; 200 series)", tr_rej >= 0.6, f"{tr_rej:.3f}")
check("pre-period trend test and step placebo: size under AR(1) ρ = 0.4 at most 0.09 (400 series)",
      tr_null <= 0.09 and pl_null <= 0.09, f"trend {tr_null:.3f}, step {pl_null:.3f}")
check("a failed pre-trend test produces its warning, and a passed one does not",
      all(has(r, "trends before treatment") == (r["pre_trend"]["p"] <= 0.05) for r in nulls))
short = ca.did_single_treated(panel(ar1(rng5, 1, 12, 0.0)[0]), "u", "t", "y", "A", 7, show=False)
check("no pre-period placebo with fewer than 8 pre periods", short["pre_placebo"] is None)
for bad, kw in (("unbalanced", dict(df=panel(np.zeros(24)).drop(index=3), t0=13)),
                ("too few post periods", dict(df=panel(np.zeros(24)), t0=23)),
                ("unknown treated unit", dict(df=panel(np.zeros(24)), t0=13, treated="Z"))):
    try:
        ca.did_single_treated(kw["df"], "u", "t", "y", kw.get("treated", "A"), kw["t0"], show=False)
        check(f"error on {bad} input", False)
    except ValueError:
        check(f"error on {bad} input", True)
two = pd.concat([panel(np.ones(24) * 0.0), panel(np.ones(24)).query("u == 'A'").assign(u="C")])
two.loc[(two.u == "C") & (two.t >= 13), "y"] += 2.0
rt = ca.did_single_treated(two, "u", "t", "y", ["A", "C"], 13, show=False)
check("list of treated units: their outcomes are averaged (effect 2 on one of two treated → 1)",
      abs(rt["estimate"] - 1.0) < 1e-12 and rt["treated"] == ["A", "C"])
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

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
