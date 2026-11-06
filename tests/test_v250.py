"""Tests for v2.5.0: DiD design, imputation (Borusyak-Jaravel-Spiess), pre-trend tests, and the GARCH half-life CI.

1. causal.did_design: one hand-built panel per design of the guide (canonical, staggered with and without
   never-treated units, reversals, heterogeneous adoption, first switch, other), plus unbalanced panels, units treated
   from the first period, treatment varying inside clusters, missing treatment and irregular periods. The expected
   classification and counts come from how each panel was built (planted dates, reversals), not from the function.
2. causal.did_imputation: closed form in the canonical design (group means, pre-period average base), sign reversal
   of TWFE, coverage and bias in simulation, SE against a unit bootstrap written here, unbalanced panel, pre-trend
   test size and power. R didimputation numbers on tests/data/did_v250_panel.csv: pending (tools/r_reference).
3. causal.did_compare: size of the paired difference test, power against a pre-trend, paired draws.
4. causal.att_gt: alpha, and the joint pre-trend test (Wald recomputed by a plain inverse, size and power in
   simulation, the level chosen with few clusters).

8. volatility.garch_fit: half_life_ci from the limits of persistence_ci. The checks never reuse ln 0.5 / ln p: they
   verify that p^h = 0.5 at each limit (the definition of a half-life, inverted), compare with the figures of the
   Ibovespa validation (claude/teste-garch-v2472-2026-11-06.md: CI 0.97138 to 0.99127 → 23.9 and 79.1 trading
   days, computed there by hand), and check the edge cases (CI reaching 1, persistence ≤ 0).

Run from the repository root:  python3 tests/test_v250.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "econometrics"
sys.path.insert(0, str(SKILL / "scripts"))
import _console  # noqa: E402,F401
import volatility as vo  # noqa: E402

warnings.filterwarnings("ignore")
ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


# ---------------------------------------------------------------- 8. half_life_ci
def sim_garch(n, omega, alpha, beta, seed, nu=6.0):
    """GARCH(1,1) with standardized t(nu) errors, written by hand (arch.simulate takes no seed)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_t(nu, n + 500) / np.sqrt(nu / (nu - 2))
    s2 = np.empty(n + 500)
    e = np.empty(n + 500)
    s2[0] = omega / (1 - alpha - beta)
    e[0] = np.sqrt(s2[0]) * z[0]
    for t in range(1, n + 500):
        s2[t] = omega + alpha * e[t - 1] ** 2 + beta * s2[t - 1]
        e[t] = np.sqrt(s2[t]) * z[t]
    return pd.Series(e[500:])


# validation figures: the persistence CI of the Ibovespa GJR-t and the half-lives computed by hand from it
lo_v, hi_v = vo._half_life(0.97138), vo._half_life(0.99127)
check("validation case: CI 0.97138-0.99127 gives 23.9 and 79.1 trading days",
      round(lo_v, 1) == 23.9 and round(hi_v, 1) == 79.1, f"({lo_v:.2f}, {hi_v:.2f})")
check("half-life is the h with p^h = 0.5 (inverse check, not the formula)",
      abs(0.97138 ** lo_v - 0.5) < 1e-12 and abs(0.99127 ** hi_v - 0.5) < 1e-12)
check("same order as the persistence: higher persistence, longer half-life", lo_v < hi_v)
check("edge cases: p ≥ 1 → inf, p ≤ 0 → 0, NaN → NaN",
      vo._half_life(1.0) == np.inf and vo._half_life(1.02) == np.inf and vo._half_life(0.0) == 0.0
      and vo._half_life(-0.3) == 0.0 and np.isnan(vo._half_life(np.nan)))

r = sim_garch(3000, 0.05, 0.08, 0.90, seed=11)
res = vo.garch_fit(r, vol="garch", dist="t", show=False)
plo, phi = res["persistence_ci"]
hlo, hhi = res["half_life_ci"]
check("garch_fit: half_life_ci maps the persistence CI (p^h = 0.5 at each finite limit)",
      np.isfinite(hlo) and abs(plo ** hlo - 0.5) < 1e-9
      and ((phi >= 1 and hhi == np.inf) or (phi < 1 and abs(phi ** hhi - 0.5) < 1e-9)),
      f"persistence CI ({plo:.4f}, {phi:.4f}) → half-life CI ({hlo:.1f}, {hhi:.1f})")
check("garch_fit: point half-life lies inside its CI", hlo <= res["half_life"] <= hhi,
      f"{res['half_life']:.1f}")
r2 = vo.garch_fit(r, vol="gjr", dist="t", alpha=0.10, show=False)
g95 = vo.garch_fit(r, vol="gjr", dist="t", show=False)
check("the half-life CI follows alpha: the 90% CI is inside the 95% CI (same GJR fit)",
      g95["half_life_ci"][0] <= r2["half_life_ci"][0] and r2["half_life_ci"][1] <= g95["half_life_ci"][1],
      f"95% {g95['half_life_ci']}, 90% {r2['half_life_ci']}")

# near-integrated variance: the CI of the persistence reaches 1 → infinite upper half-life, and the warning says so
rn = sim_garch(1500, 0.01, 0.06, 0.937, seed=7)
gn = vo.garch_fit(rn, vol="garch", dist="t", show=False)
reaches = np.isfinite(gn["persistence_ci"][1]) and gn["persistence_ci"][1] >= 1
check("near-integrated DGP (α+β = 0.997): the persistence CI reaches 1", reaches, f"{gn['persistence_ci']}")
check("... and the upper half-life limit is infinite, with the existing warning",
      gn["half_life_ci"][1] == np.inf and any("reaches 1" in w for w in gn["warnings"]))

ge = vo.garch_fit(r, vol="egarch", dist="t", show=False)
check("EGARCH: half_life_ci defined from β's CI", np.isfinite(ge["half_life_ci"][0])
      and abs(ge["persistence_ci"][0] ** ge["half_life_ci"][0] - 0.5) < 1e-9)

# ---------------------------------------------------------------- 4. att_gt: alpha and the joint pre-trend test
import causal as ca  # noqa: E402
from scipy import stats  # noqa: E402


def did_panel(seed, N=300, T=8, cohorts=(0, 4, 5, 6), pretrend=0.0, n_states=None):
    """Staggered panel with a constant effect of 1, unit effects and iid errors. pretrend adds a linear trend to every
    eventually-treated unit (a parallel-trends violation that the pre-period cells see). n_states: units are nested in
    states with a common state shock per period, and adoption is by state."""
    rng = np.random.default_rng(seed)
    if n_states:
        g_state = rng.choice(cohorts, n_states)
        g_state[:2] = 0                       # at least two never-treated states, so control='never' is defined
        state = rng.integers(0, n_states, N)
        g = g_state[state]
        shock = rng.normal(0, 0.5, (n_states, T + 1))
    else:
        g = rng.choice(cohorts, N)
        state = np.zeros(N, int)
        shock = np.zeros((1, T + 1))
    rows = []
    for i in range(N):
        a = rng.normal()
        for t in range(1, T + 1):
            te = 1.0 if g[i] and t >= g[i] else 0.0
            rows.append((i, t, a + 0.2 * t + pretrend * (g[i] > 0) * t + te + shock[state[i], t] + rng.normal(),
                         g[i], state[i]))
    return pd.DataFrame(rows, columns=["id", "t", "y", "g", "state"])


d0 = did_panel(1)
r05 = ca.att_gt(d0, "id", "t", "y", "g", n_boot=99, seed=1, show=False)
r10 = ca.att_gt(d0, "id", "t", "y", "g", n_boot=99, seed=1, alpha=0.10, show=False)
o5, o10 = r05["overall"], r10["overall"]
check("att_gt(alpha=0.10): same estimate and SE, CI = estimate ± Φ⁻¹(0.95)·SE",
      o5["estimate"] == o10["estimate"] and o5["se"] == o10["se"]
      and abs(o10["ci_high"] - o10["estimate"] - 1.6448536 * o10["se"]) < 1e-6)
check("att_gt: the dynamic table also follows alpha",
      np.allclose(r10["dynamic"]["ci_high"] - r10["dynamic"]["att"], 1.6448536 * r10["dynamic"]["se"]))
pt = r05["pretrend"]
check("pre-trend test: 9 pre cells (cohorts 4, 5, 6 × short-base periods), F reference with n = min(units, draws)",
      pt["n_cells"] == 9 and pt["df"] == 9 and pt["reference"] == "F(9, 90)" and pt["n_effective"] == 99,
      f"{pt['reference']}, cells {pt['n_cells']}")
check("pre-trend test: the F p-value is above the χ² one (small-sample form is more conservative)",
      pt["p"] >= pt["p_chi2"])
# Wald recomputed here from the bootstrap draws, by a plain inverse (the implementation uses an eigendecomposition)
eng = ca._cs_engine(d0, "id", "t", "y", "g", "never", 0, [], "dr", "base", None)
rng_ = np.random.default_rng(1)
draws = []
for _ in range(99):
    c = eng["compute"](ca._cs_draw(rng_, eng))
    draws.append(c[c["e"] < 0]["att"].to_numpy())
b_pre = r05["gt"][r05["gt"]["e"] < 0]["att"].to_numpy()
W_here = float(b_pre @ np.linalg.solve(np.cov(np.array(draws), rowvar=False), b_pre))
check("pre-trend Wald equals b'V⁻¹b with V from the same bootstrap draws (plain inverse)",
      abs(W_here - pt["stat"]) < 1e-6 * max(1, W_here), f"{W_here:.4f} vs {pt['stat']:.4f}")

# size and power (no pre-trend vs a planted linear pre-trend)
size_p = [ca.att_gt(did_panel(7000 + s), "id", "t", "y", "g", n_boot=199, seed=s, show=False)["pretrend_p"]
          for s in range(120)]
rej = float(np.mean(np.array(size_p) < 0.05))
check("pre-trend test size (no pre-trend, 300 units, B = 199, 120 panels): within [1%, 11%] for a 5% test",
      0.01 <= rej <= 0.11, f"rejection rate {rej:.3f}")
pow_p = [ca.att_gt(did_panel(8000 + s, pretrend=0.25), "id", "t", "y", "g", n_boot=199, seed=s, show=False)
         for s in range(30)]
pw = float(np.mean([r["pretrend_p"] < 0.05 for r in pow_p]))
check("pre-trend test power (linear pre-trend 0.25 per period): rejects in at least 80% of 30 panels", pw >= 0.8,
      f"power {pw:.2f}")
check("... and the rejection is relayed as a warning", all(any("pre-trend test rejects" in w for w in r["warnings"])
                                                          for r in pow_p if r["pretrend_p"] < 0.05))

def _err(f):
    try:
        f()
    except Exception as e:  # noqa: BLE001
        return e
    return None


# few clusters, many pre cells: n = 8 clusters supports one restriction → the mean pre-period coefficient, F(1, 7)
dc = did_panel(3, N=400, T=12, cohorts=(0, 7, 8, 9, 10), n_states=8)
rc = ca.att_gt(dc, "id", "t", "y", "g", cluster="state", n_boot=199, seed=3, show=False)
pc = rc["pretrend"]
check("8 clusters: the test moves to the mean pre-period coefficient (1 restriction, F(1, 7)) and says so",
      pc["n_cells"] > 7 and pc["test"] == "average pre-period coefficient" and pc["reference"] == "F(1, 7)"
      and any("average pre-period coefficient" in w for w in rc["warnings"]),
      f"cells {pc['n_cells']}, {pc['test']}, {pc['reference']}")
d30 = did_panel(4, N=800, T=8, cohorts=(0, 4, 5, 6), n_states=30)
p30 = ca.att_gt(d30, "id", "t", "y", "g", cluster="state", n_boot=199, seed=4, show=False)["pretrend"]
check("30 clusters: 9 cells and 4 event times > (30 − 1)/10, so the mean pre coefficient is tested (F(1, 29))",
      p30["test"] == "average pre-period coefficient" and p30["reference"] == "F(1, 29)", f"{p30['reference']}")
d80 = did_panel(4, N=800, T=8, cohorts=(0, 4, 5, 6), n_states=80)
p80 = ca.att_gt(d80, "id", "t", "y", "g", cluster="state", n_boot=199, seed=4, show=False)["pretrend"]
check("80 clusters: 4 event times ≤ (80 − 1)/10 < 9 cells, so the event-time averages are tested (F(4, 76))",
      p80["test"] == "pre-period event-time averages" and p80["reference"] == "F(4, 76)", f"{p80['reference']}")
d100 = did_panel(4, N=1000, T=8, cohorts=(0, 4, 5, 6), n_states=100)
p100 = ca.att_gt(d100, "id", "t", "y", "g", cluster="state", n_boot=199, seed=4, show=False)["pretrend"]
check("100 clusters: 9 cells ≤ (100 − 1)/10, so every cell is tested (F(9, 91))",
      p100["test"] == "pre-period cells" and p100["reference"] == "F(9, 91)", f"{p100['reference']}")
check("... with the few-clusters fragility warning", any("fragile" in w for w in rc["warnings"]))
e_nn = _err(lambda: ca.att_gt(did_panel(5, N=200, cohorts=(4, 5, 6)), "id", "t", "y", "g", n_boot=0, show=False))
check("att_gt with no never-treated unit and control='never': a clear error, not a pandas one (crashed before v2.5.0)",
      isinstance(e_nn, ValueError) and "never-treated" in str(e_nn), str(e_nn)[:80])
size_c = [ca.att_gt(did_panel(9000 + s, N=400, T=12, cohorts=(0, 7, 8, 9, 10), n_states=8), "id", "t", "y", "g",
                    cluster="state", n_boot=199, seed=s, show=False)["pretrend_p"] for s in range(60)]
size_c = np.array([p for p in size_c if np.isfinite(p)])
rc_rate = float(np.mean(size_c < 0.05))
check("8-cluster pre-trend test (1 restriction): finite p in most panels and size not above 12% (60 panels)",
      len(size_c) >= 50 and rc_rate <= 0.12, f"{len(size_c)} finite, rejection rate {rc_rate:.3f}")

# ---------------------------------------------------------------- 1. did_design
def treat_panel(paths, T, start=1):
    """Panel from planted treatment paths: paths[u] = list of T treatment values (None = unit not observed)."""
    rows = []
    for u, path in enumerate(paths):
        for k, v in enumerate(path):
            if v is not None:
                rows.append((u, start + k, float(v), u % 4))
    return pd.DataFrame(rows, columns=["id", "t", "d", "cl"])


def adopt(g, T):
    """Binary absorbing path: 0 before g, 1 from g (g = 0: never treated)."""
    return [0 if (g == 0 or t < g) else 1 for t in range(1, T + 1)]


T8 = 8
canon = treat_panel([adopt(0, T8)] * 6 + [adopt(5, T8)] * 6, T8)
r = ca.did_design(canon, "id", "t", "d", show=False)
check("did_design: canonical (one date, never-treated)", r["design"] == "canonical" and r["control"] == "never"
      and r["never_treated"] == 6, r["design"])

g_plant = [0] * 4 + [3] * 3 + [5] * 3 + [6] * 2
stag = treat_panel([adopt(g, T8) for g in g_plant], T8)
r = ca.did_design(stag, "id", "t", "d", show=False)
check("did_design: staggered with never-treated, not-yet controls suggested (the guide's recommendation)",
      r["design"] == "staggered" and r["control"] == "notyet" and list(r["cohorts"].index) == [3, 5, 6]
      and list(r["cohorts"]["units"]) == [3, 3, 2] and list(r["cohorts"]["pre_periods"]) == [2, 4, 5], r["design"])
check("did_design: gvar equals the planted adoption dates (0 = never)",
      r["gvar"] is not None and r["gvar"].tolist() == g_plant)
stag["y"] = np.random.default_rng(0).normal(size=len(stag)) + stag["d"]
stag["g"] = stag["id"].map(r["gvar"])
rr = ca.att_gt(stag, "id", "t", "y", "g", control=r["control"], n_boot=0, show=False)
check("did_design → att_gt: the returned gvar and control run as they are", np.isfinite(rr["overall"]["estimate"]))

nonever = treat_panel([adopt(g, T8) for g in [3] * 3 + [5] * 3 + [7] * 3], T8)
r = ca.did_design(nonever, "id", "t", "d", show=False)
check("did_design: staggered without never-treated → notyet and the last-cohort warning",
      r["design"] == "staggered" and r["never_treated"] == 0 and r["control"] == "notyet"
      and any("No never-treated" in w for w in r["warnings"]) and abs(r["share_comparable"] - 6 / 9) < 1e-12,
      f"comparable {r['share_comparable']:.2f}")

# reversals: 2 units with planted 1 → 0 (one of them twice)
paths = [adopt(0, T8)] * 4 + [adopt(4, T8)] * 4
paths += [[0, 0, 1, 1, 0, 0, 0, 0], [0, 0, 1, 0, 1, 0, 1, 1]]
rev = treat_panel(paths, T8)
r = ca.did_design(rev, "id", "t", "d", show=False)
check("did_design: 2 units with 1 → 0 (3 reversals) → not absorbing, first switch, no gvar",
      not r["absorbing"] and r["design"] == "first_switch" and r["gvar"] is None
      and int(r["units"]["reversals"].sum()) == 3 and int((r["units"]["reversals"] > 0).sum()) == 2
      and any("must not enter" in w for w in r["warnings"]))
# boundary: a single 1 → 0 is enough, and what must NOT happen (a gvar, an absorbing flag)
one = treat_panel([adopt(0, T8)] * 4 + [adopt(4, T8)] * 3 + [[0, 0, 0, 1, 1, 1, 1, 0]], T8)
r1 = ca.did_design(one, "id", "t", "d", show=False)
check("did_design boundary: one reversal in the last period removes absorbing and gvar",
      r1["absorbing"] is False and r1["gvar"] is None and r1["design"] != "staggered" and r1["design"] != "canonical")

dose = treat_panel([[0] * T8] * 3 + [[0] * 4 + [v] * 4 for v in (0.5, 1.0, 2.0, 3.5)], T8)
r = ca.did_design(dose, "id", "t", "d", show=False)
check("did_design: dose at one date, all untreated at the start → heterogeneous adoption, no gvar",
      r["design"] == "heterogeneous_adoption" and not r["binary"] and r["gvar"] is None
      and any("did_had" in e for e in r["estimators"]["external"]))
dose2 = treat_panel([[0] * 4 + [v] * 4 for v in (0.5, 1.0, 2.0, 3.5)], T8)
check("did_design: heterogeneous adoption also without never-treated units",
      ca.did_design(dose2, "id", "t", "d", show=False)["design"] == "heterogeneous_adoption")

rng = np.random.default_rng(1)
every = treat_panel([list(rng.uniform(0, 1, T8)) for _ in range(10)], T8)
r = ca.did_design(every, "id", "t", "d", show=False)
check("did_design: a dose that changes for every unit every period → other, TWFE not robust",
      r["design"] == "other" and any("not a robust causal effect" in w for w in r["warnings"]))
fs = treat_panel([[1] * T8] * 3 + [[1, 1, 1, 2, 2, 2, 2, 2]] * 3 + [[1] * 5 + [3, 3, 3]] * 3, T8)
check("did_design: dose with a common start and first changes at two dates, stayers → first switch",
      ca.did_design(fs, "id", "t", "d", show=False)["design"] == "first_switch")

unb = stag.drop(stag.index[[3, 20, 21]])
r = ca.did_design(unb, "id", "t", "d", show=False)
check("did_design: unbalanced panel flagged (att_gt needs balanced, did_imputation does not)",
      not r["balanced"] and any("Unbalanced" in w for w in r["warnings"]))
alw = treat_panel([adopt(0, T8)] * 4 + [adopt(4, T8)] * 4 + [adopt(1, T8)] * 2, T8)
r = ca.did_design(alw, "id", "t", "d", show=False)
check("did_design: 2 units treated from period 1 are counted and get g = first period (att_gt drops them)",
      r["always_treated"] == 2 and r["gvar"].tolist()[-2:] == [1, 1] and any("first observed period" in w
                                                                         for w in r["warnings"]))
mixed = stag.copy()
mixed["cl"] = mixed["id"] % 2          # treated and untreated units share clusters
r = ca.did_design(mixed, "id", "t", "d", cluster="cl", show=False)
check("did_design: treatment varying inside cluster-period cells is flagged (assignment below the cluster)",
      r["clusters"]["cluster_periods_with_mixed_treatment"] > 0 and any("assigned below" in w for w in r["warnings"]))
check("did_design: 2 clusters → few-clusters warning", any("randomization_inference" in w for w in r["warnings"]))
miss = stag.copy()
miss.loc[miss.index[:5], "d"] = np.nan
check("did_design: missing treatment rows counted, not imputed",
      ca.did_design(miss, "id", "t", "d", show=False)["missing_treatment"] == 5)
gap = stag.copy()
gap["t"] = gap["t"] * 2
check("did_design: periods that are not consecutive integers are flagged",
      any("consecutive integers" in w for w in ca.did_design(gap, "id", "t", "d", show=False)["warnings"]))

# ---------------------------------------------------------------- 2-3. did_imputation and did_compare
def twfe_dummies(d, y="y", D="D"):
    """Static TWFE coefficient by OLS on unit and period dummies (an independent route: no within transformation)."""
    X = pd.get_dummies(d[["id", "t"]].astype(str)).astype(float)
    X[D] = d[D].to_numpy(float)
    return float(np.linalg.lstsq(X.to_numpy(), d[y].to_numpy(float), rcond=None)[0][-1])


# (a) canonical design: the imputation estimate equals a DiD of each treated unit's change from its own pre-period
#     mean against the never-treated units' change (closed form, written here with group means only)
rng = np.random.default_rng(21)
rows = []
for i in range(40):
    a, g = rng.normal(), (5 if i < 15 else 0)
    for t in range(1, 9):
        rows.append((i, t, a + 0.3 * t + (1.0 + 0.4 * (t - g) if g and t >= g else 0) + rng.normal(), g))
can = pd.DataFrame(rows, columns=["id", "t", "y", "g"])
rb = ca.did_imputation(can, "id", "t", "y", "g", show=False)
Y = can.pivot(index="id", columns="t", values="y")
tr_u, nv_u = Y.index[:15], Y.index[15:]
pre, post = [1, 2, 3, 4], [5, 6, 7, 8]
cf = [((Y.loc[tr_u, t] - Y.loc[tr_u, pre].mean(axis=1)) - (Y.loc[nv_u, t].mean() - Y.loc[nv_u, pre].mean(axis=1).mean())).mean()
      for t in post]
check("did_imputation, canonical design: ATT equals the closed-form DiD on group means (pre-period average base)",
      abs(rb["overall"]["estimate"] - np.mean(cf)) < 1e-10, f"{rb['overall']['estimate']:.6f} vs {np.mean(cf):.6f}")
check("... and each event-time coefficient equals its closed-form cell",
      np.allclose(rb["dynamic"].loc[[0, 1, 2, 3], "att"].to_numpy(), cf, atol=1e-10))

# (b) sign reversal: early cohort with an effect that grows fast, late cohort with a small one, no never-treated
def sign_rev(seed):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(100):
        a, g = rng.normal(), (2 if i < 50 else 7)
        for t in range(1, 9):
            tau = ((t - g + 1) if g == 2 else 0.2) if t >= g else 0.0
            rows.append((i, t, a + 0.1 * t + tau + rng.normal(0, 0.5), g, tau, float(t >= g)))
    return pd.DataFrame(rows, columns=["id", "t", "y", "g", "tau", "D"])


sr = sign_rev(22)
tw = twfe_dummies(sr)
rb = ca.did_imputation(sr, "id", "t", "y", "g", show=False)
true_imp = sr[(sr.D == 1) & (sr.t < 7)].tau.mean()          # identified: some unit is untreated in the period (= 3)
check("sign reversal: TWFE is negative while every treated effect is positive", tw < 0 and (sr[sr.D == 1].tau > 0).all(),
      f"TWFE {tw:.3f}")
cs_sr = ca.att_gt(sr, "id", "t", "y", "g", control="notyet", n_boot=0, show=False)["overall"]
check("... one control cohort and one base period: CS (not-yet) and the imputation estimator coincide exactly",
      abs(cs_sr["estimate"] - rb["overall"]["estimate"]) < 1e-10, f"{cs_sr['estimate']:.6f} vs {rb['overall']['estimate']:.6f}")
zs = []
for s_ in range(400):
    o_ = ca.did_imputation(sign_rev(1000 + s_), "id", "t", "y", "g", show=False)["overall"]
    zs.append((o_["estimate"] - true_imp) / o_["se"])
zs = np.array(zs)
# 40 panels were too few: seeds 100-139 gave mean z -0.45 and 10% beyond 2 by chance (0.02 and 6.2% over 400)
check("... and recovers the ATT of the imputable observations: over 400 panels |mean z| < 0.15, |z| > 1.96 in ≤ 9%",
      abs(zs.mean()) < 0.15 and np.mean(np.abs(zs) > 1.96) <= 0.09,
      f"mean z {zs.mean():.3f}, |z|>1.96 {np.mean(np.abs(zs) > 1.96):.3f}")
check("... drops (and counts) the treated observations of periods with no untreated unit",
      rb["dropped_treated_obs"] == int(((sr.D == 1) & (sr.t >= 7)).sum()) and any("could not be imputed" in w
                                                                                for w in rb["warnings"]))
check("... no never-treated units: every lead is spanned by the effects, the pre-trend test says so and the estimate stands",
      rb["pretrend"]["leads"] == 0 and rb["pretrend"]["leads_dropped_collinear"] > 0 and np.isfinite(rb["overall"]["se"]))


# (c) coverage of the Theorem 3 CI, and its SE against a unit bootstrap computed here
def stag_het(seed, N=120, T=8, pretrend=0.0):
    rng = np.random.default_rng(seed)
    g = rng.choice([0, 4, 6], N)
    rows = []
    for i in range(N):
        a = rng.normal()
        for t in range(1, T + 1):
            tau = (1 + 0.3 * (t - g[i])) * (1.5 if g[i] == 4 else 1.0) if g[i] and t >= g[i] else 0.0
            rows.append((i, t, a + 0.2 * t + pretrend * (g[i] > 0) * t + tau + rng.normal(), g[i], tau))
    return pd.DataFrame(rows, columns=["id", "t", "y", "g", "tau"])


cover, ests = [], []
for s_ in range(200):
    dd = stag_het(3000 + s_)
    o = ca.did_imputation(dd, "id", "t", "y", "g", show=False)["overall"]
    true = dd[(dd.g > 0) & (dd.t >= dd.g)].tau.mean()
    cover.append(o["ci_low"] <= true <= o["ci_high"])
    ests.append(o["estimate"] - true)
cov = float(np.mean(cover))
check("did_imputation: 95% CI covers the sample ATT in 92-99.5% of 200 panels (conservative variance)",
      0.92 <= cov <= 0.995, f"coverage {cov:.3f}")
check("did_imputation: no bias (mean error within 3 Monte Carlo SE)",
      abs(np.mean(ests)) < 3 * np.std(ests) / np.sqrt(len(ests)), f"mean error {np.mean(ests):.4f}")
dd = stag_het(77)
o = ca.did_imputation(dd, "id", "t", "y", "g", show=False)["overall"]
ids = dd["id"].unique()
rngb = np.random.default_rng(5)
bs = []
for _ in range(199):
    pick = rngb.choice(ids, len(ids))
    bd = pd.concat([dd[dd.id == u].assign(id=k) for k, u in enumerate(pick)])
    bs.append(ca.did_imputation(bd, "id", "t", "y", "g", show=False)["overall"]["estimate"])
ratio = o["se"] / np.std(bs, ddof=1)
check("did_imputation: Theorem 3 SE against a unit bootstrap computed here (ratio 0.85-1.35)", 0.85 <= ratio <= 1.35,
      f"ratio {ratio:.3f}")

# (d) unbalanced panel
pan = pd.read_csv(ROOT / "tests" / "data" / "did_v250_panel.csv")
ub = pan[pan.keep_unb == 1]
ru = ca.did_imputation(ub, "id", "t", "y", "g", show=False)
check("did_imputation runs on an unbalanced panel and uses every treated row it can impute",
      ru["overall"]["n_treated_obs"] + ru["dropped_treated_obs"] == int((ub.d == 1).sum()) and np.isfinite(ru["overall"]["se"]))

# (e) pre-trend test on untreated observations: size and power
ps0 = [ca.did_imputation(stag_het(4000 + s_, N=200), "id", "t", "y", "g", show=False)["pretrend_p"] for s_ in range(150)]
ps1 = [ca.did_imputation(stag_het(5000 + s_, N=200, pretrend=0.25), "id", "t", "y", "g", show=False)["pretrend_p"]
       for s_ in range(40)]
r0, r1 = float(np.mean(np.array(ps0) < 0.05)), float(np.mean(np.array(ps1) < 0.05))
check("did_imputation pre-trend test: size within [1%, 10%] (150 panels, 200 units)", 0.01 <= r0 <= 0.10, f"{r0:.3f}")
check("did_imputation pre-trend test: power ≥ 80% against a linear pre-trend of 0.25 per period", r1 >= 0.8, f"{r1:.2f}")

# (f) did_compare: same target on common cells; the paired difference keeps its size and flags a pre-trend
cmp0 = [ca.did_compare(stag_het(6000 + s_), "id", "t", "y", "g", n_boot=199, seed=s_, show=False) for s_ in range(80)]
rej0 = float(np.mean([c["diverge"] for c in cmp0]))
check("did_compare: without pre-trends the difference is significant in ≤ 12% of 80 panels (5% test)", rej0 <= 0.12,
      f"{rej0:.3f}")
check("did_compare: the SE of the difference comes from paired draws (correlation of the two routes > 0.5)",
      np.median([c["corr"] for c in cmp0]) > 0.5, f"median corr {np.median([c['corr'] for c in cmp0]):.2f}")
cmp1 = [ca.did_compare(stag_het(7000 + s_, pretrend=0.4), "id", "t", "y", "g", n_boot=199, seed=s_, show=False)
        for s_ in range(30)]
rej1 = float(np.mean([c["diverge"] for c in cmp1]))
check("did_compare: with a linear pre-trend of 0.4 per period the routes diverge in ≥ 70% of 30 panels (43% at 0.2)", rej1 >= 0.7,
      f"{rej1:.2f}")
check("... and the warning points to the pre-trend tests", all(any("pre-trend" in w for w in c["warnings"])
                                                               for c in cmp1 if c["diverge"]))
c_one = cmp0[0]
check("did_compare: the cross_check Series has the two routes for reporting.cross_check",
      list(c_one["cross_check"].index) == ["Callaway-Sant'Anna", "imputation (BJS)"])

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
