"""Tests for v2.1: causal.randomization_inference (DiD with few treated units) and simulation.py
(the Teacher-mode Monte Carlo laboratory and power/MDE).

Checks: randomization-inference size with 6 treated of 27 units under exchangeable errors with
serial correlation, power, exact enumeration when assignments are few, CI coherent with the
p-value, a user-supplied statistic; each teaching demo reproduces its textbook result (omitted
variable bias formula, attenuation factor, HC3 coverage, spurious regression, weak-IV bias toward
OLS, few-cluster over-rejection, TWFE bias under staggered adoption); power_mde agrees with
statsmodels' power calculator.

Run from the repository root:  python3 tests/test_simulation_ri.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "econometrics" / "scripts"))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import causal
import simulation as sim

ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


def state_panel(seed, n_units=27, n_treated=6, T=20, effect=0.0, rho=0.5, staggered=True):
    r = np.random.default_rng(seed)
    treated = r.choice(n_units, n_treated, replace=False)
    g = np.zeros(n_units)
    g[treated] = r.choice([8, 11, 14], n_treated) if staggered else 11
    rows = []
    a = r.normal(size=n_units)
    lam = r.normal(scale=0.5, size=T)
    for i in range(n_units):
        e = 0.0
        for t in range(1, T + 1):
            e = rho * e + r.normal()                        # AR(1) errors, same variance for every unit
            D = float(g[i] > 0 and t >= g[i])
            rows.append((f"s{i}", t, g[i], a[i] + lam[t - 1] + effect * D + e))
    return pd.DataFrame(rows, columns=["state", "t", "g", "y"])


# ---------------------------------------------------------------------------------------------
# Randomization inference
# ---------------------------------------------------------------------------------------------
d = state_panel(1, effect=1.0)
ri = causal.randomization_inference(d, "state", "t", "y", "g", n_perm=999, seed=1)
check("RI: detects an effect of 1.0 (p < 0.05) and CI covers 1",
      ri["p"] < 0.05 and ri["ci_low"] < 1.0 < ri["ci_high"], f"est {ri['estimate']:.3f}, CI ({ri['ci_low']:.2f}, {ri['ci_high']:.2f})")
U = pd.get_dummies(d.state).to_numpy(float)
Tm = pd.get_dummies(d.t).to_numpy(float)[:, 1:]
D = ((d.g > 0) & (d.t >= d.g)).astype(float).to_numpy()
twfe = np.linalg.lstsq(np.column_stack([D, U, Tm]), d.y.to_numpy(), rcond=None)[0][0]
check("RI statistic = TWFE DiD coefficient", abs(ri["estimate"] - twfe) < 1e-8)
check("RI: 296,010 possible assignments for 6 of 27, random draws used", ri["possible_assignments"] == 296010 and not ri["exact"])

rej = []
for rep in range(200):
    dn = state_panel(500 + rep)
    rej.append(causal.randomization_inference(dn, "state", "t", "y", "g", n_perm=199, ci=False, seed=rep)["p"] < 0.05)
check("RI size with 6 treated of 27, AR(1) errors: within [2%, 8%]", 0.02 <= np.mean(rej) <= 0.08, f"{np.mean(rej):.3f}")

small = state_panel(3, n_units=8, n_treated=2, effect=0.0)
rs = causal.randomization_inference(small, "state", "t", "y", "g", n_perm=999, seed=0)
check("RI: exact enumeration when assignments are few (C(8,2) = 28) and a warning on the minimum p",
      rs["exact"] and rs["possible_assignments"] == 28 and rs["n_perm"] == 28 and len(rs["warnings"]) == 1)

coh = []
for s, eff in ((11, 1.0), (12, 0.0)):
    r_ = causal.randomization_inference(state_panel(s, effect=eff), "state", "t", "y", "g", n_perm=999, seed=2)
    coh.append(((r_["ci_low"] <= 0 <= r_["ci_high"]) == (r_["p"] > 0.05), round(r_["p"], 3)))
check("RI: CI coherent with the p-value for H0: no effect (reject and non-reject cases)", all(c for c, _ in coh),
      [p for _, p in coh])


def post_minus_pre(df):
    dd = df.copy()
    dd["D"] = ((dd.g > 0) & (dd.t >= dd.g)).astype(float)
    tr = dd[dd.g > 0]
    return tr.loc[tr.D == 1, "y"].mean() - tr.loc[tr.D == 0, "y"].mean() - (
        dd[dd.g == 0].query("t >= 11").y.mean() - dd[dd.g == 0].query("t < 11").y.mean())


ru = causal.randomization_inference(d, "state", "t", "y", "g", n_perm=299, stat=post_minus_pre, seed=4)
check("RI with a user statistic runs and detects the effect", ru["p"] < 0.05 and "user statistic" in ru["method"], f"p = {ru['p']:.3f}")

# ---------------------------------------------------------------------------------------------
# Teaching demos
# ---------------------------------------------------------------------------------------------
ov = sim.demo_omitted_variable(R=300)
t = ov["table"]
check("omitted variable: short-regression bias = gamma*delta/(delta^2+1) = 0.32",
      abs(t.loc["short regression (z omitted)", "bias"] - ov["predicted_bias"]) < 0.02
      and abs(t.loc["long regression (z included)", "bias"]) < 0.02, f"{t.loc['short regression (z omitted)', 'bias']:.3f}")
check("omitted variable: long regression covers ~95%, short almost never",
      t.loc["long regression (z included)", "coverage"] > 0.92 and t.loc["short regression (z omitted)", "coverage"] < 0.05)

bc = sim.demo_bad_control(R=300)["table"]
check("bad control: total effect unbiased, controlling for the mediator biased",
      abs(bc.iloc[0]["bias"]) < 0.03 and abs(bc.iloc[1]["bias"]) > 0.4, bc["mean"].round(3).to_dict())

me = sim.demo_measurement_error(R=300)
check("measurement error: noisy slope = beta * lambda = 0.5",
      abs(me["table"].loc["x measured with error", "mean"] - me["attenuation_factor"]) < 0.02, f"{me['table'].loc['x measured with error', 'mean']:.3f}")

hs = sim.demo_heteroskedasticity(R=800)["table"]
check("heteroskedasticity: classical coverage < 0.90, HC3 coverage >= 0.93",
      hs.loc["classical SE", "coverage"] < 0.90 and hs.loc["HC3 SE", "coverage"] >= 0.93, hs["coverage"].round(3).to_dict())

sp = sim.demo_spurious_regression(R=300)["table"].iloc[0]
check("spurious regression: true null rejected > 60%, mean R2 > mean DW",
      sp["rejection_rate_of_true_null"] > 0.6 and sp["mean_R2"] > sp["mean_Durbin_Watson"], sp.round(3).to_dict())

wv = sim.demo_weak_iv(R=600)["table"]
check("weak IV: median bias toward OLS (> 25% of OLS bias), IQR > 5x strong IV; strong IV ~ unbiased with F > 50",
      wv.loc["2SLS weak", "median_bias"] > 0.25 * wv.loc["OLS", "median_bias"]
      and wv.loc["2SLS weak", "iqr"] > 5 * wv.loc["2SLS strong", "iqr"]
      and abs(wv.loc["2SLS strong", "median_bias"]) < 0.03 and wv.loc["2SLS strong", "median_first_stage_F"] > 50,
      wv["median"].round(3).to_dict())

fc = sim.demo_few_clusters(R=300, B=199)["table"]["rejection_rate_of_true_null"]
check("few clusters: CR1-normal over-rejects (> 15%); wild bootstrap within [2%, 11%] and below half of CR1",
      fc["CR1, normal critical values"] > 0.15 and 0.02 <= fc["wild cluster bootstrap"] <= 0.11
      and fc["wild cluster bootstrap"] < fc["CR1, normal critical values"] / 2, fc.round(3).to_dict())

sd_ = sim.demo_staggered_did(R=20)
tb = sd_["table"]
check("staggered DiD: TWFE biased by more than 1, Callaway-Sant'Anna within 0.3 of the true ATT",
      abs(tb.loc["TWFE", "bias"]) > 1.0 and abs(tb.loc["Callaway-Sant'Anna", "bias"]) < 0.3,
      tb["bias"].round(2).to_dict())

# ---------------------------------------------------------------------------------------------
# Power and MDE vs statsmodels
# ---------------------------------------------------------------------------------------------
from statsmodels.stats.power import NormalIndPower
n_sm = NormalIndPower().solve_power(effect_size=0.25, alpha=0.05, power=0.8, ratio=1.0)   # per group
pm = sim.power_mde(mde=0.25)
check("power_mde: total n = 2 x statsmodels per-group n (d = 0.25)", abs(pm["n"] - np.ceil(2 * n_sm)) <= 1,
      f"{pm['n']:.0f} vs {2 * n_sm:.1f}")
m1 = sim.power_mde(n=pm["n"])
check("power_mde: MDE at that n returns 0.25", abs(m1["mde"] - 0.25) < 0.002)
mc = sim.power_mde(mde=0.25, cluster_size=20, icc=0.05)
check("power_mde: design effect 1 + 19*0.05 = 1.95 multiplies n", abs(mc["design_effect"] - 1.95) < 1e-12
      and abs(mc["n"] / pm["n"] - 1.95) < 0.01)

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
