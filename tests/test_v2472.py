"""Tests for v2.4.7.2: pending items of v2.4.7.1 and of the second GARCH validation (Ibovespa, 2026-11-06).

1. reporting.cross_check: the "compatible with zero" threshold was a fixed 1.96; it is now z = Φ⁻¹(1 − alpha/2).
2. diagnostics.stationarity / var_analysis / vecm_fit: alpha reaches the joint unit-root verdict (ADF, DF-GLS,
   KPSS); the ADF-only note (fixed 0.10 in var_analysis, and the only check in vecm_fit) uses the same alpha.
   A KPSS p-value at the lower table bound (≤ 0.01) counts as a rejection at alpha = 0.01.
3. volatility.garch_fit: delta-method SE and CI of the persistence; with the skewed t the GJR persistence uses
   κ = E[z²·1(z<0)] of the fitted distribution, not γ/2; arch's ConvergenceWarning no longer leaks (arch's
   fit(show_warning=True) installs its own "always" filter, which beat the skill's "ignore").

The checks do not reuse the implementation's formulas: κ comes from a numerical integral of Hansen's density
written here, the GARCH persistence SE from the closed form √(Var α + Var β + 2 Cov), and the CI is judged
by its coverage in a simulation with a known persistence.

Data: the Selic series of tests/test_v2471.py (BCB/SGS 4189, 2010-01 to 2026-09).

Run from the repository root:  python3 tests/test_v2472.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import integrate, special, stats

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "econometrics"
sys.path.insert(0, str(SKILL / "scripts"))
import _console  # noqa: E402,F401
import diagnostics  # noqa: E402
import multivariate as mv  # noqa: E402
import reporting as rp  # noqa: E402
import volatility as vo  # noqa: E402

warnings.filterwarnings("ignore")
ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


# ---------------------------------------------------------------- 1. cross_check: alpha sets the "null" threshold
c05 = rp.cross_check({"a": 0.18, "b": 0.17}, se=0.1, show=False)
c10 = rp.cross_check({"a": 0.18, "b": 0.17}, se=0.1, alpha=0.10, show=False)
check("cross_check: max |est| = 1.8 SE is compatible with zero at alpha 0.05 (z 1.96) but not at 0.10 (z 1.645)",
      c05["null"] and not c10["null"] and c05["agree"] and c10["agree"])
check("cross_check: the default message still states the 1.96 threshold",
      any("1.96 SE" in w for w in c05["warnings"]))
old_rule = [abs(x) < 1.96 * 0.1 for x in (0.19, 0.195, 0.197, 0.2)]
new_rule = [rp.cross_check({"a": x, "b": x - 0.001}, se=0.1, show=False)["null"] for x in (0.19, 0.195, 0.197, 0.2)]
check("cross_check: with the default alpha the decisions equal the old fixed 1.96 rule away from the 4th decimal",
      old_rule == new_rule, f"{new_rule}")
try:
    rp.cross_check({"a": 1, "b": 1}, se=0.1, alpha=0, show=False)
    bad = False
except ValueError:
    bad = True
check("cross_check: alpha outside (0, 1) is refused", bad)


# ---------------------------------------------------------------- 2. stationarity / var_analysis / vecm_fit
def ar1(phi, seed, n=200, burn=20):
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n + burn)
    x = np.zeros(n + burn)
    for t in range(1, n + burn):
        x[t] = phi * x[t - 1] + e[t]
    return pd.Series(x[burn:])


x = ar1(0.9, 18)
t05 = diagnostics.stationarity(x, "x", show=False)
t10 = diagnostics.stationarity(x, "x", show=False, alpha=0.10)
r = t05.iloc[0]
check("stationarity: AR(0.9) with DF-GLS p between 0.05 and 0.10 is inconclusive at 0.05 and stationary at 0.10",
      0.05 < r["p DF-GLS"] < 0.10 and t05.iloc[0]["Verdict"].startswith("Inconclusive")
      and t10.iloc[0]["Verdict"] == "Stationary" and t10.attrs["alpha"] == 0.10,
      f"(p DF-GLS = {r['p DF-GLS']:.3f})")
rw = pd.Series(np.cumsum(np.random.default_rng(3).normal(size=500)))
t01 = diagnostics.stationarity(rw, "rw", show=False, alpha=0.01)
check("stationarity at alpha 0.01: a KPSS p-value at the table bound (≤ 0.01) still rejects (random walk → 'Unit root')",
      t01.iloc[0]["p KPSS"] <= 0.01 and t01.iloc[0]["Verdict"] == "Unit root")
t20 = diagnostics.stationarity(x, "x", show=False, alpha=0.20)
check("stationarity: alpha outside the KPSS table [0.01, 0.10] gives a warning; the default gives none",
      any("outside [0.01, 0.1]" in w for w in t20.attrs["warnings"]) and t05.attrs["warnings"] == [])

SELIC = """
8.65 8.65 8.65 8.72 9.4 9.94 10.32 10.66 10.66 10.66 10.66 10.66
10.85 11.17 11.62 11.74 11.92 12.1 12.25 12.42 11.91 11.7 11.4 10.9
10.7 10.4 9.82 9.35 8.87 8.39 8.07 7.85 7.39 7.23 7.14 7.16
7.11 7.12 7.15 7.26 7.42 7.9 8.23 8.45 8.9 9.25 9.45 9.9
10.17 10.43 10.65 10.87 10.9 10.9 10.9 10.9 10.9 10.92 11.15 11.58
11.82 12.15 12.58 12.68 13.15 13.58 13.69 14.15 14.15 14.15 14.15 14.15
14.15 14.15 14.15 14.15 14.15 14.15 14.15 14.15 14.15 14.05 13.9 13.65
13.17 12.82 12.15 11.59 11.15 10.15 10.01 9.15 8.35 8.01 7.4 7
6.9 6.72 6.58 6.4 6.4 6.4 6.4 6.4 6.4 6.4 6.4 6.4
6.4 6.4 6.4 6.4 6.4 6.4 6.4 5.9 5.71 5.38 4.9 4.59
4.4 4.19 3.95 3.65 3.01 2.58 2.15 1.94 1.9 1.9 1.9 1.9
1.9 1.9 2.23 2.65 3.29 3.76 4.15 5.01 5.43 6.3 7.65 8.76
9.15 10.49 11.15 11.65 12.51 12.89 13.15 13.58 13.65 13.65 13.65 13.65
13.65 13.65 13.65 13.65 13.65 13.65 13.65 13.19 12.97 12.65 12.17 11.87
11.65 11.15 11 10.65 10.46 10.4 10.4 10.4 10.5 10.65 11.04 11.77
12.24 13.15 13.57 14.15 14.55 14.74 14.9 14.9 14.9 14.9 14.9 14.9
14.9 14.9 14.8 14.64 14.4 14.29 14.15 13.94 13.78
"""
selic = pd.Series(np.array(SELIC.split(), float), index=pd.period_range("2010-01", periods=201, freq="M").to_timestamp())
d = pd.DataFrame({"selic": selic, "noise": np.random.default_rng(4).normal(size=201)}, index=selic.index).iloc[1:]
va05 = mv.var_analysis(d, maxlags=4, show=False)
va30 = mv.var_analysis(d, maxlags=4, alpha=0.30, show=False)
check("var_analysis: Selic stays I(0) with the default alpha, and the ADF-only note (ADF p ≈ 0.27) is said",
      va05["integration_order"]["selic"].startswith("I(0)") and any("ADF alone" in w and "selic" in w for w in va05["warnings"]))
check("var_analysis(alpha=0.30): the ADF-only note follows alpha (0.27 < 0.30) and the KPSS range warning arrives",
      not any("ADF alone" in w and "selic" in w for w in va30["warnings"])
      and any("outside [0.01, 0.1]" in w for w in va30["warnings"]))

rng5 = np.random.default_rng(5)
stat2 = pd.DataFrame({"a": rng5.normal(size=300), "b": rng5.normal(size=300)})
rw2 = pd.DataFrame(np.cumsum(rng5.normal(size=(300, 2)), axis=0), columns=["a", "b"])
vs = mv.vecm_fit(stat2, k_ar_diff=1, rank=2, show=False)
vr = mv.vecm_fit(rw2, k_ar_diff=1, rank=2, show=False)
check("vecm_fit, full rank: stationary series → no unit-root disagreement; random walks → joint verdict disagreement",
      "do not classify" not in vs["message"] and "do not classify" in vr["message"] and "DF-GLS" in vr["message"])


# ---------------------------------------------------------------- 3a. persistence SE: closed form for the GARCH
def gjr_sim(seed, n=2000, om=0.05, a=0.03, g=0.08, b=0.88, nu=6.0):
    rng = np.random.default_rng(seed)
    e = rng.standard_t(nu, size=n + 500) / np.sqrt(nu / (nu - 2))
    xx = np.empty(n + 500)
    s2 = om / (1 - a - g / 2 - b)
    for t in range(n + 500):
        xx[t] = np.sqrt(s2) * e[t]
        s2 = om + a * xx[t] ** 2 + g * xx[t] ** 2 * (xx[t] < 0) + b * s2
    return pd.Series(xx[500:])


from arch import arch_model  # noqa: E402
r0 = gjr_sim(0)
fg = vo.garch_fit(r0, "garch", show=False)
ra = arch_model(r0, vol="GARCH", p=1, o=0, q=1, dist="t").fit(disp="off", show_warning=False)
C = ra.param_cov
se_closed = np.sqrt(C.loc["alpha[1]", "alpha[1]"] + C.loc["beta[1]", "beta[1]"] + 2 * C.loc["alpha[1]", "beta[1]"])
check("garch_fit GARCH: persistence SE = √(Var α + Var β + 2 Cov) from arch's covariance",
      abs(fg["persistence_se"] - se_closed) < 1e-4 * se_closed, f"({fg['persistence_se']:.5f} vs {se_closed:.5f})")
fj = vo.garch_fit(r0, "gjr", show=False)
Cj = arch_model(r0, vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off", show_warning=False).param_cov
w = pd.Series({"alpha[1]": 1.0, "gamma[1]": 0.5, "beta[1]": 1.0})
se_j = float(np.sqrt(w @ Cj.loc[w.index, w.index] @ w))
lo, hi = fj["persistence_ci"]
check("garch_fit GJR (t errors): SE = √(w'Σw) with w = (1, ½, 1) and the CI is persistence ± 1.96 SE",
      abs(fj["persistence_se"] - se_j) < 1e-4 * se_j and abs(hi - fj["persistence"] - 1.959964 * se_j) < 1e-3 * se_j)
fe = vo.garch_fit(r0, "egarch", show=False)
check("garch_fit EGARCH: persistence SE is the SE of β",
      abs(fe["persistence_se"] - fe["params"].loc["beta[1]", "se"]) < 1e-6)
Dd = pd.DataFrame({"monday": (np.arange(len(r0)) % 5 == 0).astype(float)}, index=r0.index)
fq = vo.garch_fit(r0, "gjr", variance_dummies=Dd, show=False)
check("garch_fit with variance regressors (own QML): persistence SE and CI are finite",
      np.isfinite(fq["persistence_se"]) and fq["persistence_ci"][0] < fq["persistence"] < fq["persistence_ci"][1])

# ---------------------------------------------------------------- 3b. coverage of the persistence CI
TRUE_P = 0.03 + 0.08 / 2 + 0.88
cover, est, ses = [], [], []
for s_ in range(200):
    f = vo.garch_fit(gjr_sim(100 + s_), "gjr", show=False)
    lo, hi = f["persistence_ci"]
    cover.append(lo <= TRUE_P <= hi)
    est.append(f["persistence"])
    ses.append(f["persistence_se"])
cov_rate = float(np.mean(cover))
ratio = float(np.mean(ses) / np.std(est, ddof=1))
check("persistence CI of the GJR covers the true 0.95 in about 95% of 200 simulations (band 90%–99%)",
      0.90 <= cov_rate <= 0.99, f"(coverage {cov_rate:.0%})")
check("delta-method SE close to the Monte Carlo SD of the estimates (ratio 0.7–1.4)",
      0.7 <= ratio <= 1.4, f"(ratio {ratio:.2f})")


# ---------------------------------------------------------------- 3c. skewed t: κ by numerical integration
def hansen_pdf(z, eta, lam):
    """Hansen (1994) skewed t, standardized, written from the paper (not from arch)."""
    c = special.gamma((eta + 1) / 2) / (np.sqrt(np.pi * (eta - 2)) * special.gamma(eta / 2))
    a = 4 * lam * c * (eta - 2) / (eta - 1)
    b = np.sqrt(1 + 3 * lam ** 2 - a ** 2)
    s = (1 - lam) if z < -a / b else (1 + lam)
    return b * c * (1 + ((b * z + a) / s) ** 2 / (eta - 2)) ** (-(eta + 1) / 2)


def kappa_quad(eta, lam):
    return integrate.quad(lambda z: z ** 2 * hansen_pdf(z, eta, lam), -np.inf, 0, limit=200)[0]


mass = integrate.quad(lambda z: hansen_pdf(z, 5.0, -0.3), -np.inf, np.inf, limit=200)[0]
var1 = integrate.quad(lambda z: z ** 2 * hansen_pdf(z, 5.0, -0.3), -np.inf, np.inf, limit=200)[0]
check("Hansen density written here integrates to 1 with unit variance (η 5, λ −0.3)",
      abs(mass - 1) < 1e-6 and abs(var1 - 1) < 1e-5)
pairs = [(5.0, -0.3), (8.0, 0.2), (4.5, 0.0), (30.0, -0.5)]
diffs = [abs(vo._neg_weight("skewt", pd.Series({"eta": e_, "lambda": l_})) - kappa_quad(e_, l_)) for e_, l_ in pairs]
check("κ = E[z²·1(z<0)] used by garch_fit equals the numerical integral of Hansen's density (4 parameter pairs)",
      max(diffs) < 1e-6, f"(max diff {max(diffs):.1e})")
check("κ = 1/2 for λ = 0 and for symmetric distributions; κ > 1/2 when λ < 0 (left skew)",
      abs(vo._neg_weight("skewt", pd.Series({"eta": 6.0, "lambda": 0.0})) - 0.5) < 1e-9
      and vo._neg_weight("t", pd.Series({"nu": 6.0})) == 0.5
      and vo._neg_weight("skewt", pd.Series({"eta": 6.0, "lambda": -0.3})) > 0.5)


def skew_gjr_sim(seed, n=3000, eta=6.0, lam=-0.35, om=0.05, a=0.03, g=0.08, b=0.88):
    # skewed t draws by inverse CDF on a grid of the density written above
    rng = np.random.default_rng(seed)
    grid = np.linspace(-15, 15, 30001)
    dens = np.array([hansen_pdf(z, eta, lam) for z in grid])
    cdf = np.cumsum(dens) * (grid[1] - grid[0])
    cdf /= cdf[-1]
    e = np.interp(rng.uniform(size=n + 500), cdf, grid)
    e = (e - e.mean()) / e.std()
    xx = np.empty(n + 500)
    s2 = om / (1 - a - g / 2 - b)
    for t in range(n + 500):
        xx[t] = np.sqrt(s2) * e[t]
        s2 = om + a * xx[t] ** 2 + g * xx[t] ** 2 * (xx[t] < 0) + b * s2
    return pd.Series(xx[500:])


rs = skew_gjr_sim(7)
fs = vo.garch_fit(rs, "gjr", dist="skewt", show=False)
P = fs["params"]["coef"]
k_q = kappa_quad(P["eta"], P["lambda"])
check("garch_fit GJR skewt: persistence = α + β + κγ with κ from the integral (not γ/2), and κ differs from 1/2",
      abs(fs["persistence"] - (P["alpha[1]"] + P["beta[1]"] + k_q * P["gamma[1]"])) < 1e-6 and abs(k_q - 0.5) > 0.01,
      f"(λ̂ {P['lambda']:.3f}, κ {k_q:.4f})")
check("garch_fit GJR skewt: persistence SE finite (gradient also in η and λ)", np.isfinite(fs["persistence_se"]))


# ---------------------------------------------------------------- 3d. no ConvergenceWarning leak
from arch.utility.exceptions import ConvergenceWarning  # noqa: E402
rd = r0 * 1e-3   # decimal returns: arch's optimiser fails ("Inequality constraints incompatible")
with warnings.catch_warnings(record=True) as rec:
    warnings.resetwarnings()
    warnings.simplefilter("default")
    fd = vo.garch_fit(rd, "garch", show=False)
leaked = [x for x in rec if issubclass(x.category, ConvergenceWarning)]
with warnings.catch_warnings(record=True) as rec2:
    warnings.resetwarnings()
    warnings.simplefilter("default")
    arch_model(rd, vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off")
arch_leaks = [x for x in rec2 if issubclass(x.category, ConvergenceWarning)]
check("garch_fit on decimal returns: arch fails and warns when called directly, but no ConvergenceWarning "
      "escapes garch_fit (main fit and auxiliary GJR)",
      len(arch_leaks) > 0 and not leaked and any("asymmetry LR not available" in w for w in fd["warnings"]))

# ---------------------------------------------------------------- documentation and version
rules = (SKILL / "references" / "interpretation-rules.md").read_text(encoding="utf-8")
vol = (SKILL / "references" / "volatility.md").read_text(encoding="utf-8")
check("interpretation rules for v2.4.7.2 and the volatility reference cover the persistence CI and κ",
      "## Rules added in v2.4.7.2" in rules and "persistence_ci" in vol and "E[z²·1(z<0)]" in vol)
import json  # noqa: E402
import re  # noqa: E402
mk = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))
top = re.search(r"^## v(\S+)", (ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), re.M).group(1)
check("same version (2.4.7.2 or later) in marketplace, reporting and CHANGELOG",
      mk["plugins"][0]["version"] == rp.SKILL_VERSION == top and tuple(map(int, top.split("."))) >= (2, 4, 7, 2))

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
