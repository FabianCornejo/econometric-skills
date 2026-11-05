"""Tests for v2.4.7.1: fixes from the v2.4.7 validation (a VAR of Selic, IPCA and the exchange rate; a GARCH of
the Ibovespa; Claude Code, 2026-11-05).

1. garch_fit(vol="garch") flagged an omitted asymmetry only through the Engle-Ng sign-bias tests, which have
   little power under fat tails: on the Ibovespa the sign-bias joint p was 0.77 while the LR of the GJR against
   the GARCH was 31.5 (p = 2e-8). The LR is now the primary asymmetry test.
2. var_analysis warned of a unit root in the Selic from the ADF alone (p = 0.27) while the skill's own joint
   verdict (ADF, DF-GLS, KPSS) was I(0). It now uses the joint verdict and says when only the ADF disagrees.
3. reporting.cross_check said "agree" for a VAR and a local projection that were both compatible with zero;
   the agent had to add the caveat. It now flags the agreement as uninformative.

Data: Selic (BCB/SGS 4189, monthly, % a.a.), 2010-01 to 2026-09, as downloaded in the validation.

Run from the repository root:  python3 tests/test_v2471.py
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
import diagnostics  # noqa: E402
import multivariate as mv  # noqa: E402
import reporting as rp  # noqa: E402
import volatility as vo  # noqa: E402

warnings.filterwarnings("ignore")
ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


# ---------------------------------------------------------------- 1. asymmetry: LR of the GJR, not only sign bias
def gjr_sim(seed, n=3000, om=0.04, a=0.03, g=0.08, b=0.90, nu=5):
    rng = np.random.default_rng(seed)
    e = rng.standard_t(nu, size=n + 500) / np.sqrt(nu / (nu - 2))
    x = np.empty(n + 500)
    s2 = om / (1 - a - g / 2 - b)
    for t in range(n + 500):
        x[t] = np.sqrt(s2) * e[t]
        s2 = om + a * x[t] ** 2 + g * x[t] ** 2 * (x[t] < 0) + b * s2
    return pd.Series(x[500:])


from arch import arch_model  # noqa: E402
r0 = gjr_sim(0)
f0 = vo.garch_fit(r0, "garch", show=False)
llg = arch_model(r0, vol="GARCH", p=1, o=0, q=1, dist="t").fit(disp="off").loglikelihood
llj = arch_model(r0, vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off").loglikelihood
check("garch_fit: asymmetry LR = 2·(llf GJR − llf GARCH) computed with arch directly",
      abs(f0["asymmetry_lr"]["lr"] - 2 * (llj - llg)) < 1e-6)
lr_pw, sb_pw = [], []
for s in range(8):
    f = vo.garch_fit(gjr_sim(s), "garch", show=False)
    lr_pw.append(f["asymmetry_lr"]["p"] < 0.05)
    sb_pw.append(f["sign_bias"]["joint_p"] < 0.05)
check("t(5) errors, moderate GJR asymmetry (γ = 0.08): the LR detects it in every sample, the sign-bias tests rarely",
      all(lr_pw) and np.mean(sb_pw) <= 0.25, f"LR {np.mean(lr_pw):.2f}, sign bias {np.mean(sb_pw):.2f}")
check("the warning names the omitted asymmetry and says the sign-bias tests missed it",
      any("asymmetry omitted" in w and "sign-bias tests miss it" in w for w in f0["warnings"]))
g_hat = [vo.garch_fit(gjr_sim(s), "garch", show=False)["asymmetry_lr"]["gamma"] for s in range(8)]
check("the GJR γ reported with the LR recovers the simulated γ = 0.08 (mean of 8 samples within 0.03)",
      abs(np.mean(g_hat) - 0.08) < 0.03, f"mean γ̂ = {np.mean(g_hat):.3f}")
fs = [vo.garch_fit(gjr_sim(100 + s, a=0.07, g=0.0), "garch", show=False) for s in range(8)]
n_warn = sum(any("asymmetry omitted" in w for w in f["warnings"]) for f in fs)
check("symmetric GARCH data: the LR rarely rejects and the asymmetry warning rarely fires (≤ 1 of 8)",
      sum(f["asymmetry_lr"]["p"] < 0.05 for f in fs) <= 1 and n_warn <= 1, f"{n_warn}/8")
check("GJR and EGARCH fits carry no asymmetry LR (nothing to test against)",
      vo.garch_fit(r0, "gjr", show=False)["asymmetry_lr"] is None
      and vo.garch_fit(r0, "egarch", show=False)["asymmetry_lr"] is None)
import arch  # noqa: E402
_real_am = arch.arch_model


def _gjr_fails(*a, **k):
    if k.get("o", 0) == 1:
        raise RuntimeError("simulated GJR failure")
    return _real_am(*a, **k)


arch.arch_model = _gjr_fails
try:
    fd = vo.garch_fit(r0, "garch", show=False)
finally:
    arch.arch_model = _real_am
check("a GJR fit that fails gives LR = nan and says so (not 'no asymmetry'); the GARCH fit itself survives",
      np.isnan(fd["asymmetry_lr"]["p"]) and any("asymmetry LR not available" in w for w in fd["warnings"])
      and not any("asymmetry omitted" in w for w in fd["warnings"]) and np.isfinite(fd["llf"]))

# ---------------------------------------------------------------- 2. unit-root warning in var_analysis
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
IPCA = """0.75 0.78 0.52 0.57 0.43 0 0.01 0.04 0.45 0.75 0.83 0.63 0.83 0.8 0.79 0.77 0.47 0.15 0.16 0.37 0.53 0.43 0.52 0.5 0.56 0.45 0.21 0.64 0.36 0.08 0.43 0.41 0.57 0.59 0.6 0.79 0.86 0.6 0.47 0.55 0.37 0.26 0.03 0.24 0.35 0.57 0.54 0.92 0.55 0.69 0.92 0.67 0.46 0.4 0.01 0.25 0.57 0.42 0.51 0.78 1.24 1.22 1.32 0.71 0.74 0.79 0.62 0.22 0.54 0.82 1.01 0.96 1.27 0.9 0.43 0.61 0.78 0.35 0.52 0.44 0.08 0.26 0.18 0.3 0.38 0.33 0.25 0.14 0.31 -0.23 0.24 0.19 0.16 0.42 0.28 0.44 0.29 0.32 0.09 0.22 0.4 1.26 0.33 -0.09 0.48 0.45 -0.21 0.15 0.32 0.43 0.75 0.57 0.13 0.01 0.19 0.11 -0.04 0.1 0.51 1.15 0.21 0.25 0.07 -0.31 -0.38 0.26 0.36 0.24 0.64 0.86 0.89 1.35 0.25 0.86 0.93 0.31 0.83 0.53 0.96 0.87 1.16 1.25 0.95 0.73 0.54 1.01 1.62 1.06 0.47 0.67 -0.68 -0.36 -0.29 0.59 0.41 0.62 0.53 0.84 0.71 0.61 0.23 -0.08 0.12 0.23 0.26 0.24 0.28 0.56 0.42 0.83 0.16 0.38 0.46 0.21 0.38 -0.02 0.44 0.56 0.39 0.52 0.16 1.31 0.56 0.43 0.26 0.24 0.26 -0.11 0.48 0.09 0.18 0.33 0.33 0.7 0.88 0.67 0.58 0.16 0.07 -0.32 0.82"""
ipca = pd.Series(np.array(IPCA.split(), float), index=selic.index)
st = diagnostics.stationarity(selic.iloc[1:], "selic", show=False)
from statsmodels.tsa.stattools import adfuller  # noqa: E402
check("Selic 2010-02..2026-09: ADF alone does not reject (p ≈ 0.27) but the joint verdict is I(0)",
      adfuller(selic.iloc[1:].to_numpy(), autolag="AIC")[1] > 0.10 and st.attrs["suggested_order"].startswith("I(0)"))
d = pd.DataFrame({"ipca": ipca, "selic": selic}).iloc[1:]
va = mv.var_analysis(d, maxlags=8, show=False)
check("var_analysis: no unit-root warning for the Selic, and the ADF-only disagreement is said",
      not any("do not classify" in w for w in va["warnings"]) and any("ADF alone" in w and "selic" in w for w in va["warnings"]))
rw = pd.DataFrame(np.cumsum(np.random.default_rng(1).normal(size=(200, 2)), axis=0), columns=["a", "b"])
va_rw = mv.var_analysis(rw, show=False)
check("var_analysis: random walks still get the unit-root warning (joint verdict I(1))",
      any("do not classify" in w for w in va_rw["warnings"]) and all(v.startswith("I(1)") for v in va_rw["integration_order"].values()))

tt = np.arange(200)
rng_t = np.random.default_rng(2)
ts = pd.DataFrame({"a": 0.05 * tt + rng_t.normal(size=200), "b": rng_t.normal(size=200)})
va_t = mv.var_analysis(ts, show=False)
check("var_analysis: a trend-stationary series in a VAR with trend 'c' is told to add the trend, not reassured",
      va_t["integration_order"]["a"].startswith("I(0) around a deterministic trend")
      and any("trend='ct'" in w for w in va_t["warnings"]) and not any(w.startswith("ADF alone does not reject a unit root for a") for w in va_t["warnings"]))

# ---------------------------------------------------------------- 3. cross_check with routes compatible with zero
c = rp.cross_check({"VAR": 0.145309, "local projection": 0.0964425}, se=0.1235, label="IPCA h=12", show=False)
check("cross_check: VAR 0.145 and local projection 0.096 with SE 0.12 → agreement flagged as uninformative",
      c["agree"] and c["null"] and any("uninformative agreement" in w for w in c["warnings"]))
c2 = rp.cross_check({"a": 5.32, "b": 5.46}, se=2.093, show=False)
c3 = rp.cross_check({"a": 1.0, "b": 3.0}, se=0.1, show=False)
c4 = rp.cross_check({"a": 0.15, "b": 0.17}, se=0.1, show=False)
check("cross_check: 'null' and 'weak' never fire together (estimates between 1 and 1.96 SE)",
      c4["null"] and not c4["weak"])
c1 = rp.cross_check({"a": 1.0}, se=0.1, show=False)
check("cross_check with a single route returns the null and weak keys (False)", c1["null"] is False and c1["weak"] is False)
check("cross_check: an estimate away from zero is not 'null'; a disagreement is not flagged as uninformative",
      not c2["null"] and c2["weak"] and not c3["agree"] and not c3["null"])

# ---------------------------------------------------------------- documentation and version
rules = (SKILL / "references" / "interpretation-rules.md").read_text(encoding="utf-8")
vol = (SKILL / "references" / "volatility.md").read_text(encoding="utf-8")
check("interpretation rules for v2.4.7.1 and the volatility reference name the LR as the asymmetry test",
      "## Rules added in v2.4.7.1" in rules and "does not show symmetry" in rules and "LR" in vol and "little power" in vol)
import json  # noqa: E402
import re  # noqa: E402
mk = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))
top = re.search(r"^## v(\S+)", (ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), re.M).group(1)
check("same version (2.4.7.1 or later) in marketplace, reporting and CHANGELOG",
      mk["plugins"][0]["version"] == rp.SKILL_VERSION == top and tuple(map(int, top.split("."))) >= (2, 4, 7, 1))

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
