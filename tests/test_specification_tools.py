"""Tests for the v1.5 specification and sensitivity tools.

specification.py:  fgls_het, prediction_intervals, long_run_effect, standardized_coefficients,
                   ovb_decomposition, oster, robustness_value / sensitivity
diagnostics.py:    mundlak_test, count_diagnostics

Benchmarks: published examples from Wooldridge, *Introductory Econometrics* (6.1, 6.5–6.6, 8.7,
10.4, 14.4, 17.3) and the Darfur example of Cinelli and Hazlett (2020, JRSS-B); plus known-truth
simulations and exact algebraic identities.

Run from the repository root:  python3 tests/test_specification_tools.py
Needs: statsmodels, linearmodels, scipy, wooldridge (pip install wooldridge).
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "econometrics" / "scripts"))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

import diagnostics as d
import specification as sp

try:
    import wooldridge as woo
except ImportError:  # pragma: no cover
    sys.exit("SKIP: install the `wooldridge` package (pip install wooldridge) to run these tests")

ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


def close(a, b, tol):
    return abs(float(a) - float(b)) <= tol


# ---------------------------------------------------------------------------------------------
# Example 6.1 — standardized (beta) coefficients, HPRICE2
# ---------------------------------------------------------------------------------------------
h2 = woo.data("hprice2")
r61 = smf.ols("price ~ nox + crime + rooms + dist + stratio", h2).fit()
st = sp.standardized_coefficients(r61)
book = {"nox": -0.340, "crime": -0.143, "rooms": 0.514, "dist": -0.235, "stratio": -0.270}
check("W6.1 standardized coefficients", all(close(st.loc[k, "coef"], v, 0.0006) for k, v in book.items()),
      f"got {st['coef'].round(3).to_dict()}")
check("W6.1 standardized p-values equal the original ones",
      np.allclose(st["p"].values, r61.pvalues.drop("Intercept").values))

# ---------------------------------------------------------------------------------------------
# Examples 6.5 and 6.6 — confidence interval for E(y|x) and prediction interval for y, GPA2
# ---------------------------------------------------------------------------------------------
g2 = woo.data("gpa2")
r65 = smf.ols("colgpa ~ sat + hsperc + hsize + I(hsize**2)", g2).fit()
pi = sp.prediction_intervals(r65, pd.DataFrame({"sat": [1200], "hsperc": [30], "hsize": [5]}))
check("W6.5 prediction 2.70, CI for the mean 2.66–2.74",
      close(pi["prediction"].iloc[0], 2.70, 0.005) and close(pi["ci_lower"].iloc[0], 2.66, 0.005)
      and close(pi["ci_upper"].iloc[0], 2.74, 0.005), pi.round(3).to_dict("records"))
check("W6.6 prediction interval for an individual 1.60–3.80",
      close(pi["pi_lower"].iloc[0], 1.60, 0.005) and close(pi["pi_upper"].iloc[0], 3.80, 0.005))

# log(y) model: the PI for y is exp of the PI for log y; the mean uses Duan's smearing
rng = np.random.default_rng(5)
xl = rng.uniform(0, 2, 5000)
dl = pd.DataFrame({"x": xl, "y": np.exp(1 + 0.5 * xl + rng.normal(0, 0.8, 5000))})
rl = smf.ols("np.log(y) ~ x", dl).fit()
pil = sp.prediction_intervals(rl, pd.DataFrame({"x": [1.0]}), log_y=True)
true_mean = np.exp(1.5 + 0.8 ** 2 / 2)
check("log-y: smearing-corrected mean ≈ true E(y|x)", close(pil["mean_y"].iloc[0], true_mean, 0.05 * true_mean),
      f"{pil['mean_y'].iloc[0]:.3f} vs {true_mean:.3f}; naive exp {pil['median_y'].iloc[0]:.3f}")
yn =np.exp(1.5 + rng.normal(0, 0.8, 20000))
cov = np.mean((yn >= pil["pi_lower_y"].iloc[0]) & (yn <= pil["pi_upper_y"].iloc[0]))
check("log-y: 95% PI for y covers ≈ 95% of new draws", close(cov, 0.95, 0.01), f"coverage {cov:.3f}")

# ---------------------------------------------------------------------------------------------
# Example 8.7 — FGLS for the cigarette demand equation, SMOKE
# ---------------------------------------------------------------------------------------------
sm_ = woo.data("smoke")
fg = sp.fgls_het("cigs ~ np.log(income) + np.log(cigpric) + educ + age + I(age**2) + restaurn", sm_,
                 cov_type="nonrobust", show=False)
b, s = fg["fgls"].params, fg["fgls"].bse
check("W8.7 FGLS log(income) 1.30 (.44), educ −.463 (.120), restaurn −3.46 (.80)",
      close(b["np.log(income)"], 1.30, 0.006) and close(s["np.log(income)"], 0.44, 0.006)
      and close(b["educ"], -0.463, 0.0006) and close(s["educ"], 0.120, 0.0006)
      and close(b["restaurn"], -3.46, 0.006) and close(s["restaurn"], 0.80, 0.006),
      f"got {b['np.log(income)']:.3f} ({s['np.log(income)']:.3f}), {b['educ']:.3f}, {b['restaurn']:.3f}")
check("FGLS default reports HC3 errors too", sp.fgls_het(
    "cigs ~ np.log(income) + educ", sm_, show=False)["fgls"].cov_type == "HC3")

# ---------------------------------------------------------------------------------------------
# Example 10.4 — long-run propensity of a finite distributed lag, FERTIL3; ADL known truth
# ---------------------------------------------------------------------------------------------
fz = woo.data("fertil3").assign(pe_1=lambda x: x.pe.shift(1), pe_2=lambda x: x.pe.shift(2))
r104 = smf.ols("gfr ~ pe + pe_1 + pe_2 + ww2 + pill", fz).fit()
lr = sp.long_run_effect(r104, ["pe", "pe_1", "pe_2"])
check("W10.4 LRP = .101 (se .030)", close(lr["effect"], 0.101, 0.0006) and close(lr["se"], 0.030, 0.0006),
      f"got {lr['effect']:.4f} ({lr['se']:.4f})")

rng = np.random.default_rng(9)
n = 3000
x = rng.normal(size=n)
y = np.zeros(n)
for t in range(1, n):
    y[t] = 0.6 * y[t - 1] + 1.0 * x[t] + 0.4 * x[t - 1] + rng.normal()
adl = pd.DataFrame({"y": y, "x": x}).assign(y_1=lambda v: v.y.shift(1), x_1=lambda v: v.x.shift(1))
radl = smf.ols("y ~ y_1 + x + x_1", adl).fit()
lra = sp.long_run_effect(radl, ["x", "x_1"], y_lag_terms=["y_1"])
check("ADL long-run multiplier (1.0+0.4)/(1−0.6) = 3.5", close(lra["effect"], 3.5, 3 * lra["se"]),
      f"got {lra['effect']:.3f} ({lra['se']:.3f})")
check("ADL long-run SE by delta method is positive", lra["se"] > 0)

# ---------------------------------------------------------------------------------------------
# Omitted-variable bias: exact identity β̃1 = β̂1 + β̂2·δ̃1 (GPA1)
# ---------------------------------------------------------------------------------------------
gp = woo.data("gpa1")
ovb = sp.ovb_decomposition(gp, y="colGPA", focal="ACT", omitted="hsGPA")
check("OVB identity: short = long + gamma·delta", close(ovb["beta_short"], ovb["beta_long"] + ovb["bias"], 1e-10),
      f"short {ovb['beta_short']:.4f} = {ovb['beta_long']:.4f} + {ovb['bias']:.4f}")
ovb_c = sp.ovb_decomposition(gp, y="colGPA", focal="ACT", omitted="hsGPA", included=["skipped"])
check("OVB identity with other controls", close(ovb_c["beta_short"], ovb_c["beta_long"] + ovb_c["bias"], 1e-10))

# ---------------------------------------------------------------------------------------------
# Oster (2019): known truth with proportional selection (δ = 1)
# ---------------------------------------------------------------------------------------------
rng = np.random.default_rng(3)
bst, bctl = [], []
for _ in range(200):
    n = 2000
    w1, w2 = rng.normal(size=n), rng.normal(size=n)
    dd = w1 + w2 + rng.normal(size=n)
    yy = 1.0 * dd + w1 + w2 + rng.normal(size=n)
    dfo = pd.DataFrame({"y": yy, "d": dd, "w1": w1, "w2": w2})
    full_r2 = smf.ols("y ~ d + w1 + w2", dfo).fit().rsquared
    o = sp.oster(smf.ols("y ~ d", dfo).fit(), smf.ols("y ~ d + w1", dfo).fit(), "d", r_max=full_r2)
    bst.append(o["beta_star"])
    bctl.append(o["beta_controlled"])
check("Oster β* recovers the truth under proportional selection",
      close(np.mean(bst), 1.0, 0.02), f"β* {np.mean(bst):.3f} vs controlled {np.mean(bctl):.3f} (truth 1.0)")
o1 = sp.oster(smf.ols("y ~ d", dfo).fit(), smf.ols("y ~ d + w1", dfo).fit(), "d")
check("Oster default R_max = min(1.3·R̃, 1)",
      close(o1["r_max"], min(1.3 * smf.ols("y ~ d + w1", dfo).fit().rsquared, 1), 1e-12))
o0 = sp.oster(smf.ols("y ~ d", dfo).fit(), smf.ols("y ~ d + w1", dfo).fit(), "d", delta=o1["delta_for_zero"])
check("Oster: δ for β = 0 makes β* = 0", close(o0["beta_star"], 0, 1e-9), f"β* {o0['beta_star']:.2e}")

# ---------------------------------------------------------------------------------------------
# Cinelli and Hazlett (2020): Darfur example, t = 4.18 with 783 degrees of freedom
# ---------------------------------------------------------------------------------------------
rv = sp.robustness_value(t=4.18, dof=783)
check("Cinelli-Hazlett Darfur: partial R² = 2.2%, RV = 13.9%, RV(α=.05) = 7.6%",
      close(rv["partial_r2"], 0.022, 0.0006) and close(rv["rv_q"], 0.139, 0.0006)
      and close(rv["rv_qa"], 0.076, 0.0006), {k: round(v, 4) for k, v in rv.items()})
# RV(q, α) has a second closed form for very large effects (f·q > 1/f_crit); it must join continuously
from scipy import stats as _st  # noqa: E402
dof_b = 783
f_crit = _st.t.ppf(0.975, dof_b - 1) / np.sqrt(dof_b - 1)
t_b = np.sqrt(dof_b) / f_crit
lo, hi = sp.robustness_value(t_b * (1 - 1e-9), dof_b), sp.robustness_value(t_b * (1 + 1e-9), dof_b)
check("RV(q, α) continuous at the large-effect boundary", close(lo["rv_qa"], hi["rv_qa"], 1e-6),
      f"{lo['rv_qa']:.8f} vs {hi['rv_qa']:.8f}")
r_s = smf.ols("colGPA ~ ACT + hsGPA + skipped", gp).fit()
sens = sp.sensitivity(r_s, "skipped")
check("sensitivity() reads t and df from a fitted model",
      close(sens["rv_q"], sp.robustness_value(r_s.tvalues["skipped"], r_s.df_resid)["rv_q"], 1e-12))

# ---------------------------------------------------------------------------------------------
# Mundlak / correlated random effects (diagnostics.py)
# ---------------------------------------------------------------------------------------------
from linearmodels.panel import PanelOLS  # noqa: E402

wp = woo.data("wagepan")
mk = d.mundlak_test(wp, y="lwage", x_vary=["expersq", "union", "married"], entity="nr", time="year",
                    x_const=["educ", "black", "hisp"], show=False)
fe = PanelOLS.from_formula("lwage ~ expersq + union + married + C(year) + EntityEffects",
                           wp.set_index(["nr", "year"], drop=False), drop_absorbed=True).fit()
check("Mundlak/CRE coefficients equal FE (W14.4: union .080, married .047)",
      all(close(mk["coefficients"][v], fe.params[v], 1e-6) for v in ["union", "married", "expersq"]),
      f"CRE union {mk['coefficients']['union']:.4f} vs FE {fe.params['union']:.4f}")

rng = np.random.default_rng(21)
units, T = 300, 6
ids = np.repeat(np.arange(units), T)
alpha = rng.normal(size=units)
for corr, label in [(0.8, "correlated"), (0.0, "uncorrelated")]:
    xm = corr * alpha[ids] + rng.normal(size=units * T)
    pdf = pd.DataFrame({"id": ids, "t": np.tile(np.arange(T), units), "x": xm,
                        "y": 1 + 0.5 * xm + alpha[ids] + rng.normal(size=units * T)})
    res = d.mundlak_test(pdf, y="y", x_vary=["x"], entity="id", time="t", show=False)
    want = res["p"] < 0.01 if corr else res["p"] > 0.05
    check(f"Mundlak test: {label} effects → {'reject' if corr else 'do not reject'}", want, f"p = {res['p']:.3g}")

# ---------------------------------------------------------------------------------------------
# Count models: overdispersion (diagnostics.py) — Example 17.3, CRIME1: σ̂² = 1.52
# ---------------------------------------------------------------------------------------------
c1 = woo.data("crime1")
pois = smf.poisson("narr86 ~ pcnv + avgsen + tottime + ptime86 + qemp86 + inc86 + black + hispan + born60",
                   c1).fit(disp=0)
cd = d.count_diagnostics(pois, show=False)
check("W17.3 Pearson dispersion σ̂² = 1.52 (σ̂ = 1.23)", close(cd["dispersion"], 1.52, 0.005),
      f"got {cd['dispersion']:.3f}")
check("W17.3 quasi-Poisson SE = Poisson SE × σ̂",
      np.allclose(cd["quasi_poisson_se"].values, pois.bse.values * np.sqrt(cd["dispersion"])))

rng = np.random.default_rng(13)
n = 4000
xc = rng.normal(size=n)
mu = np.exp(0.5 + 0.4 * xc)
y_nb = rng.negative_binomial(n=1.0, p=1.0 / (1.0 + mu))    # NB2 with α = 1: Var = μ + μ²
y_po = rng.poisson(mu)
for yv, label, want_reject in [(y_nb, "negative binomial data", True), (y_po, "Poisson data", False)]:
    fit = sm.Poisson(yv, sm.add_constant(xc)).fit(disp=0)
    out = d.count_diagnostics(fit, show=False)
    p = out["tests"].loc["NB2 (Var = μ + αμ²)", "p"]
    check(f"Cameron-Trivedi NB2 test on {label}", (p < 0.01) if want_reject else (p > 0.05), f"p = {p:.3g}")

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
