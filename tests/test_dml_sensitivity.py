"""Tests for v2.2 omitted-variable sensitivity of double ML (causal.dml_sensitivity, dml_benchmark),
following Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis (2022), "Long Story Short".

Checks: with the true partial R2 of a linear latent confounder (rho = 1), the bound equals the
actual omitted-variable bias of the short DML estimate; the true effect lies inside the bounds;
with linear learners the robustness value reproduces Cinelli-Hazlett's OLS robustness value;
benchmarks reproduce partial R2 computed by OLS and the change in theta when a covariate is
dropped; a nonlinear design with random forests stays inside the conservative bound; zero
confounding returns the DML interval.

Run from the repository root:  python3 tests/test_dml_sensitivity.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "econometrics" / "scripts"))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression

import causal
import specification as sp

ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


rng = np.random.default_rng(0)
n = 5000
x1, x2, A = rng.normal(size=(3, n))
D = 0.8 * x1 + 0.5 * x2 + 0.5 * A + rng.normal(size=n)
Y = 1.0 * D + 0.7 * x1 + 0.5 * x2 + 0.5 * A + rng.normal(size=n)
df = pd.DataFrame(dict(y=Y, d=D, x1=x1, x2=x2, A=A))
LIN = dict(learner_y=LinearRegression(), learner_d=LinearRegression(), n_rep=2)
res = causal.dml_plr(df, "y", "d", ["x1", "x2"], **LIN)
bias = res["estimate"] - 1.0

ls, ll = smf.ols("y ~ d + x1 + x2", df).fit(), smf.ols("y ~ d + x1 + x2 + A", df).fit()
cf_y = 1 - ll.ssr / ls.ssr
ds, dl = smf.ols("d ~ x1 + x2", df).fit(), smf.ols("d ~ x1 + x2 + A", df).fit()
cf_d = 1 - dl.ssr / ds.ssr
o = causal.dml_sensitivity(res, cf_y, cf_d, rho=1.0, show=False)
check("short DML estimate is biased by omitting A (bias > 0.15)", bias > 0.15, f"{bias:.3f}")
check("bound with the true cf_y, cf_d (linear confounder, rho = 1) equals the actual bias within 5%",
      abs(o["bias_bound"] - bias) / bias < 0.05, f"bound {o['bias_bound']:.4f} vs bias {bias:.4f}")
check("true theta = 1 inside the confidence bounds", o["ci_low"] <= 1.0 <= o["ci_high"],
      f"[{o['ci_low']:.3f}, {o['ci_high']:.3f}]")
check("confidence bounds are wider than the bounds", o["ci_low"] < o["theta_lower"] and o["ci_high"] > o["theta_upper"])

cr = sp.robustness_value(ls.tvalues["d"], ls.df_resid)
check("linear learners: RV equals Cinelli-Hazlett OLS RV (within 0.005)", abs(o["rv"] - cr["rv_q"]) < 0.005,
      f"{o['rv']:.4f} vs {cr['rv_q']:.4f}")
check("linear learners: RVa close to Cinelli-Hazlett RVa (within 0.01) and below RV",
      abs(o["rva"] - cr["rv_qa"]) < 0.01 and o["rva"] < o["rv"], f"{o['rva']:.4f} vs {cr['rv_qa']:.4f}")
o_half = causal.dml_sensitivity(res, 0.0, 0.0, show=False)
check("zero confounding: bounds collapse to the estimate and CI to the DML CI",
      abs(o_half["bias_bound"]) < 1e-12 and abs(o_half["ci_low"] - res["ci_low"]) < 0.002 * abs(res["estimate"]))
o_rv = causal.dml_sensitivity(res, o["rv"], o["rv"], show=False)
check("at cf_y = cf_d = RV the lower bound is zero", abs(o_rv["theta_lower"]) < 1e-3, f"{o_rv['theta_lower']:.5f}")
tab = o["bound_table"]
check("bound table decreases in both cf_y and cf_d (positive theta)",
      (np.diff(tab.to_numpy(), axis=0) <= 1e-12).all() and (np.diff(tab.to_numpy(), axis=1) <= 1e-12).all())

bm = causal.dml_benchmark(df, "y", "d", ["x1", "x2"], ["x2"], res=res, **LIN)
ols_cy = 1 - smf.ols("y ~ d + x1 + x2", df).fit().ssr / smf.ols("y ~ d + x1", df).fit().ssr
ols_cd = 1 - smf.ols("d ~ x1 + x2", df).fit().ssr / smf.ols("d ~ x1", df).fit().ssr
check("benchmark x2: cf_y and cf_d equal the OLS partial R2 (within 0.01)",
      abs(bm.loc["x2", "cf_y"] - ols_cy) < 0.01 and abs(bm.loc["x2", "cf_d"] - ols_cd) < 0.01,
      f"cf_y {bm.loc['x2', 'cf_y']:.4f}/{ols_cy:.4f}, cf_d {bm.loc['x2', 'cf_d']:.4f}/{ols_cd:.4f}")
ols_dth = smf.ols("y ~ d + x1", df).fit().params["d"] - ls.params["d"]
check("benchmark x2: delta theta equals the OLS change from dropping x2 (within 0.01)",
      abs(bm.loc["x2", "delta_theta"] - ols_dth) < 0.01, f"{bm.loc['x2', 'delta_theta']:.4f} vs {ols_dth:.4f}")

# Nonlinear nuisance functions with random forests: the conservative bound (rho = 1) covers the bias
r2 = np.random.default_rng(1)
n2 = 3000
z1, z2, U = r2.normal(size=(3, n2))
D2 = np.sin(2 * z1) + 0.5 * z2 ** 2 + 0.6 * U + r2.normal(size=n2)
Y2 = 0.5 * D2 + np.cos(z1) + 0.5 * np.abs(z2) + 0.6 * U + r2.normal(size=n2)
d2 = pd.DataFrame(dict(y=Y2, d=D2, z1=z1, z2=z2, U=U))
rf = causal.dml_plr(d2, "y", "d", ["z1", "z2"], n_rep=2)
spec_s = "y ~ d + np.sin(2*z1) + np.cos(z1) + I(z2**2) + np.abs(z2)"
cy2 = 1 - smf.ols(spec_s + " + U", d2).fit().ssr / smf.ols(spec_s, d2).fit().ssr
dsp = "d ~ np.sin(2*z1) + np.cos(z1) + I(z2**2) + np.abs(z2)"
cd2 = 1 - smf.ols(dsp + " + U", d2).fit().ssr / smf.ols(dsp, d2).fit().ssr
o2 = causal.dml_sensitivity(rf, cy2, cd2, show=False)
b2 = rf["estimate"] - 0.5
check("random forests, nonlinear design: |bias| <= 1.15 x bound and true effect inside confidence bounds",
      abs(b2) <= 1.15 * o2["bias_bound"] and o2["ci_low"] <= 0.5 <= o2["ci_high"],
      f"bias {b2:.3f}, bound {o2['bias_bound']:.3f}, CI [{o2['ci_low']:.3f}, {o2['ci_high']:.3f}]")

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
