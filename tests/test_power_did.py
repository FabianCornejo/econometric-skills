"""Tests for v2.2 power analysis for difference-in-differences (simulation.mde_did,
power_did_sim, mde_did_sim).

Checks: the fast two-way-demeaned TWFE estimate and CR1 clustered SE equal statsmodels' dummy
regression; the analytic McKenzie (2012) variance matches the simulated SD of the estimator
under equicorrelated errors, and simulated power is close to the analytic 80%; positive AR(1)
autocorrelation lowers power below the analytic formula (Burlig, Preonas & Woerman 2020);
mde_did_sim finds an effect with ~80% power; size is near 5% with many treated units and
inflated with three treated units (warning); analytic n and MDE are inverses.

Run from the repository root:  python3 tests/test_power_did.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "econometrics" / "scripts"))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import simulation as sim

ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


rng = np.random.default_rng(1)
start = np.r_[np.full(10, 6), np.full(5, 4), np.zeros(15)]
Y, D = sim._did_panel(rng, 30, 10, start, 0.3, 1.0, 1.0, 0.5, 0.5)
b, se = sim._twfe_cluster(Y, D)
df = pd.DataFrame({"y": Y.ravel(), "d": D.ravel(), "i": np.repeat(np.arange(30), 10), "t": np.tile(np.arange(10), 30)})
r = smf.ols("y ~ d + C(i) + C(t)", df).fit(cov_type="cluster", cov_kwds={"groups": df["i"]})
check("TWFE by two-way demeaning = dummy regression (staggered, balanced)", abs(b - r.params["d"]) < 1e-10,
      f"{b:.6f} vs {r.params['d']:.6f}")
check("CR1 clustered SE = statsmodels cluster SE", abs(se - r.bse["d"]) / r.bse["d"] < 1e-6, f"{se:.6f} vs {r.bse['d']:.6f}")

an = sim.mde_did(n_units=100, n_pre=5, n_post=5, sd=np.sqrt(2), rho=0.5)
se_an = np.sqrt(2 * 0.5 * (1 / 5 + 1 / 5) / (0.25 * 100))
ps = sim.power_did_sim(100, 10, effect=an["mde"], sd_unit=1.0, sd_e=1.0, rho_ar=0.0, R=1000)
check("equicorrelated errors: simulated SD of TWFE within 5% of McKenzie's SE",
      abs(ps["sd_estimate"] / se_an - 1) < 0.05, f"{ps['sd_estimate']:.4f} vs {se_an:.4f}")
check("equicorrelated errors: simulated power within 0.06 of the analytic 80%", abs(ps["power"] - 0.8) < 0.06,
      f"{ps['power']:.3f}")
check("unbiased and size near 5% with 50 treated units", abs(ps["mean_estimate"] - an["mde"]) < 0.02
      and 0.025 <= ps["size"] <= 0.07, f"mean {ps['mean_estimate']:.3f}, size {ps['size']:.3f}")
pa = sim.power_did_sim(100, 10, effect=an["mde"], rho_ar=0.8, R=1000)
check("AR(1) rho = 0.8: power far below the analytic 80% (< 0.65)", pa["power"] < 0.65, f"{pa['power']:.3f}")
m = sim.mde_did_sim(R=400, n_units=100, T=10, rho_ar=0.8)
check("mde_did_sim: MDE larger than the analytic one and power at it ~ 0.80",
      m["mde"] > an["mde"] * 1.2 and 0.78 <= m["power_at_mde"] <= 0.85, f"MDE {m['mde']:.3f}, power {m['power_at_mde']:.3f}")
few = sim.power_did_sim(12, 10, share_treated=0.25, effect=0.0, R=1000)
check("3 treated of 12 units: clustered-SE size inflated (> 7%) and a warning", few["size"] > 0.07 and few["warnings"],
      f"size {few['size']:.3f}")
nn = sim.mde_did(mde=an["mde"], n_pre=5, n_post=5, sd=np.sqrt(2), rho=0.5)
check("mde_did: n for the MDE at n = 100 returns 100", abs(nn["n_units"] - 100) <= 1, f"{nn['n_units']}")
check("mde_did: more pre/post periods and higher rho lower the MDE",
      sim.mde_did(n_units=100, n_pre=10, n_post=10, rho=0.5)["mde"] < sim.mde_did(n_units=100, n_pre=5, n_post=5, rho=0.5)["mde"]
      < sim.mde_did(n_units=100, n_pre=5, n_post=5, rho=0.2)["mde"])

# v2.2 eval: estimate the simulation inputs from a historical panel
est = []
for s_ in range(20):
    rg = np.random.default_rng(300 + s_)
    Yh, _ = sim._did_panel(rg, 120, 36, np.zeros(120), 0.0, 1.0, 0.7, 0.6, 0.5)
    hist = pd.DataFrame({"ag": np.repeat(np.arange(120), 36), "m": np.tile(np.arange(36), 120), "y": Yh.ravel()})
    est.append(sim.did_power_inputs(hist, "ag", "m", "y"))
mr = np.mean([e["rho_ar"] for e in est]); mu = np.mean([e["rho_ar_uncorrected"] for e in est])
ms = np.mean([e["sd_e"] for e in est])
check("did_power_inputs: Nickell-corrected AR(1) recovers 0.6 (within 0.03) where the raw estimate is biased down",
      abs(mr - 0.6) < 0.03 and mu < mr - 0.02, f"corrected {mr:.3f}, raw {mu:.3f}")
check("did_power_inputs: residual SD near the stationary SD 0.7 (within 0.05), total SD larger",
      abs(ms - 0.7) < 0.05 and est[0]["sd_total"] > est[0]["sd_e"], f"sd_e {ms:.3f}, total {est[0]['sd_total']:.3f}")

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
