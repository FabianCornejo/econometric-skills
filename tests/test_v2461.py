"""Tests for v2.4.6.1: fixes from the first real-series test of v2.4.6 (monthly IPCA, Claude Code, 2026-11-05).

The analysis was sound, but in five places the skill left it to the agent to notice a problem:
1. long_memory read d = 0.285 as long memory although an AR(1) (phi ~ 0.58) gives such a d in about 12% of samples;
2. the OLS-CUSUM next to sup_wald rejected stability (p = 0.0025) only because it was scaled by the iid SD of
   autocorrelated residuals; scaled by the long-run SD it does not reject (p = 0.34), and the verdict now says
   when a valid CUSUM and the sup-F disagree;
3. sup_wald did not warn that the 2022 break in the AR(1) was dated at the edge of the trimmed window;
4. reporting.cross_check said two 12-month forecasts "agree within 1 SE" with SE = 2.1 points on 5.3%;
5. matplotlib was missing and there was no setup_env extra for it.

The data are the IPCA monthly % changes (IBGE, from BCB/SGS series 433), 2000-01 to 2026-09, as the agent
downloaded them.

Run from the repository root:  python3 tests/test_v2461.py
"""
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "econometrics"
sys.path.insert(0, str(SKILL / "scripts"))
import _console  # noqa: E402,F401
import reporting as rp  # noqa: E402
import setup_env as se  # noqa: E402
import ts_univariate as tu  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


IPCA = """
0.62 0.13 0.22 0.42 0.01 0.23 1.61 1.31 0.23 0.14 0.32 0.59
0.57 0.46 0.38 0.58 0.41 0.52 1.33 0.70 0.28 0.83 0.71 0.65
0.52 0.36 0.60 0.80 0.21 0.42 1.19 0.65 0.72 1.31 3.02 2.10
2.25 1.57 1.23 0.97 0.61 -0.15 0.20 0.34 0.78 0.29 0.34 0.52
0.76 0.61 0.47 0.37 0.51 0.71 0.91 0.69 0.33 0.44 0.69 0.86
0.58 0.59 0.61 0.87 0.49 -0.02 0.25 0.17 0.35 0.75 0.55 0.36
0.59 0.41 0.43 0.21 0.10 -0.21 0.19 0.05 0.21 0.33 0.31 0.48
0.44 0.44 0.37 0.25 0.28 0.28 0.24 0.47 0.18 0.30 0.38 0.74
0.54 0.49 0.48 0.55 0.79 0.74 0.53 0.28 0.26 0.45 0.36 0.28
0.48 0.55 0.20 0.48 0.47 0.36 0.24 0.15 0.24 0.28 0.41 0.37
0.75 0.78 0.52 0.57 0.43 0.00 0.01 0.04 0.45 0.75 0.83 0.63
0.83 0.80 0.79 0.77 0.47 0.15 0.16 0.37 0.53 0.43 0.52 0.50
0.56 0.45 0.21 0.64 0.36 0.08 0.43 0.41 0.57 0.59 0.60 0.79
0.86 0.60 0.47 0.55 0.37 0.26 0.03 0.24 0.35 0.57 0.54 0.92
0.55 0.69 0.92 0.67 0.46 0.40 0.01 0.25 0.57 0.42 0.51 0.78
1.24 1.22 1.32 0.71 0.74 0.79 0.62 0.22 0.54 0.82 1.01 0.96
1.27 0.90 0.43 0.61 0.78 0.35 0.52 0.44 0.08 0.26 0.18 0.30
0.38 0.33 0.25 0.14 0.31 -0.23 0.24 0.19 0.16 0.42 0.28 0.44
0.29 0.32 0.09 0.22 0.40 1.26 0.33 -0.09 0.48 0.45 -0.21 0.15
0.32 0.43 0.75 0.57 0.13 0.01 0.19 0.11 -0.04 0.10 0.51 1.15
0.21 0.25 0.07 -0.31 -0.38 0.26 0.36 0.24 0.64 0.86 0.89 1.35
0.25 0.86 0.93 0.31 0.83 0.53 0.96 0.87 1.16 1.25 0.95 0.73
0.54 1.01 1.62 1.06 0.47 0.67 -0.68 -0.36 -0.29 0.59 0.41 0.62
0.53 0.84 0.71 0.61 0.23 -0.08 0.12 0.23 0.26 0.24 0.28 0.56
0.42 0.83 0.16 0.38 0.46 0.21 0.38 -0.02 0.44 0.56 0.39 0.52
0.16 1.31 0.56 0.43 0.26 0.24 0.26 -0.11 0.48 0.09 0.18 0.33
0.33 0.70 0.88 0.67 0.58 0.16 0.07 -0.32 0.82
"""
v = np.array(IPCA.split(), float)
y = pd.Series(v, index=pd.date_range("2000-01-31", periods=len(v), freq="ME"))
ys = y - y.groupby(y.index.month).transform("mean") + y.mean()     # minus the month means
check("IPCA fixture: 321 months, mean 0.4935, 12-month rate at 2026-09 of 4.58%",
      len(v) == 321 and round(v.mean(), 4) == 0.4935 and round((np.prod(1 + v[-12:] / 100) - 1) * 100, 2) == 4.58)


def ar(g, n, phi, burn=200):
    e = g.normal(size=n + burn)
    z = np.zeros(n + burn)
    for t in range(1, n + burn):
        z[t] = phi * z[t - 1] + e[t]
    return z[burn:]


# ---------------------------------------------------------------- 1. long memory vs short-run persistence
lm = tu.long_memory(ys, show=False)
sm = lm["short_memory_check"]
check("IPCA: d (local Whittle) is unchanged at 0.285", abs(lm["d_local_whittle"] - 0.2849) < 1e-3)
check("IPCA: the short-memory check fits an AR(1) and does not reject (p > 0.05), as the agent's Monte Carlo found",
      sm["ar_order"] == 1 and sm["p_local_whittle"] > 0.05, f"p = {sm['p_local_whittle']:.3f}")
check("IPCA: the reading no longer says 'stationary long memory'",
      "no evidence of long memory beyond" in lm["reading"] and not lm["reading"].startswith("stationary long memory"),
      lm["reading"][:80])
fp = [tu.long_memory(ar(np.random.default_rng(s), 320, 0.6), B=99, seed=s, show=False)["reading"].startswith("stationary long memory")
      for s in range(20)]
check("AR(1) with phi = 0.6, n = 320: never read as long memory (20 samples)", not any(fp), f"{sum(fp)}/20")
from scipy import special  # noqa: E402


def arfima0(g, n, d, burn=2000):
    j = np.arange(n + burn)
    psi = np.exp(special.gammaln(j + d) - special.gammaln(j + 1) - special.gammaln(d))
    return np.convolve(g.normal(size=n + burn), psi)[:n + burn][burn:]
det = [tu.long_memory(arfima0(np.random.default_rng(100 + s), 2000, 0.3), B=99, seed=s, show=False)["short_memory_check"]["p_local_whittle"] < 0.05
       for s in range(20)]
check("ARFIMA(0, 0.3, 0), n = 2000: short memory rejected in at least half of 20 samples", np.mean(det) >= 0.5, f"{np.mean(det):.2f}")
off = tu.long_memory(ys, short_memory_check=False, show=False)
check("short_memory_check=False keeps the v2.4.6 reading", off["short_memory_check"] is None
      and off["reading"].startswith("stationary long memory"))
rw_read = [tu.long_memory(np.cumsum(np.random.default_rng(s).normal(size=400)), show=False)["reading"] for s in range(5)]
check("random walks keep the non-stationary reading (the short-memory check never overrides it)",
      all(r.startswith("non-stationary") for r in rw_read), rw_read[0][:60])
check("arfima still takes d from local Whittle without the simulation", tu.arfima(ys, (1, 0), show=False)["long_memory"]["short_memory_check"] is None)

# ---------------------------------------------------------------- 2–3. break tests that disagree; dates at the edge
sw = tu.sup_wald(ys, B=199, show=False)
check("IPCA mean: the CUSUM scaled by the long-run SD does not reject (p ≈ 0.34); the iid version did (p = 0.0025)",
      sw["p_bootstrap"] >= 0.05 and 0.25 < sw["cusum"]["p"] < 0.45 and sw["cusum"]["p_iid"] < 0.01
      and "OLS-CUSUM" not in sw["verdict"], f"{sw['cusum']['p']:.3f} vs iid {sw['cusum']['p_iid']:.4f}")
from statsmodels.stats.diagnostic import breaks_cusumolsresid  # noqa: E402
w = np.random.default_rng(3).normal(size=200)
sw_w = tu.sup_wald(w, B=9, show=False)
check("CUSUM: with white-noise errors (sieve AR(0)) it equals statsmodels' OLS-CUSUM",
      sw_w["sieve_ar_order"] == 0 and np.isclose(sw_w["cusum"]["statistic"], breaks_cusumolsresid(w - w.mean(), ddof=1)[0]))
cz = [tu.sup_wald(ar(np.random.default_rng(50 + s), 320, 0.6), B=9, show=False)["cusum"]["p"] < 0.05 for s in range(100)]
check("CUSUM size under a stable AR(0.6), n = 320: ≤ 10% (the iid version rejects about 60%)", np.mean(cz) <= 0.10,
      f"{np.mean(cz):.3f}")
v1 = tu._break_verdict(0.30, 0.01, 0.05, "2010-01")
v2 = tu._break_verdict(0.01, 0.01, 0.05, "2010-01")
check("verdict: sup-F not rejecting with a CUSUM rejecting says the series is not shown stable; a rejection does not",
      "OLS-CUSUM rejects stability" in v1 and "do not report the series as stable" in v1 and "OLS-CUSUM" not in v2)
X = pd.DataFrame({"ylag": ys.shift(1)}).iloc[1:]
sw2 = tu.sup_wald(ys.iloc[1:], X, B=199, show=False)
check("IPCA AR(1): break in 2022-07, flagged as dated at the end of the searched window",
      sw2["p_bootstrap"] < 0.05 and str(sw2["break_date"])[:7] == "2022-07"
      and any("end of the searched window" in w and "Andrews 2003" in w for w in sw2["warnings"]), str(sw2["warnings"])[:120])
yb = np.r_[np.random.default_rng(5).normal(0, 1, 100), np.random.default_rng(6).normal(1.5, 1, 100)]
sw3 = tu.sup_wald(yb, B=99, show=False)
pw = [tu.sup_wald(ar(np.random.default_rng(300 + s), 150, 0.5) + np.r_[np.zeros(75), np.ones(75)], B=99, seed=s,
                  show=False)["p_bootstrap"] < 0.05 for s in range(20)]
check("sup_wald: power against a 1 SD shift with AR(0.5) errors, n = 150: ≥ 40%", np.mean(pw) >= 0.4, f"{np.mean(pw):.2f}")
sz = [tu.sup_wald(ar(np.random.default_rng(1900 + s), 150, 0.8), B=99, seed=s, show=False)["p_bootstrap"] < 0.05
      for s in range(40)]
check("sup_wald: size under a persistent stable AR(0.8) ≤ 10% (40 samples; sieve on the residuals under H0)",
      np.mean(sz) <= 0.10, f"{np.mean(sz):.3f}")
w_rw = [any("unit root" in w for w in tu.sup_wald(np.cumsum(np.random.default_rng(s).normal(size=150)), B=9,
                                                  show=False)["warnings"]) for s in range(10)]
w_br = [any("unit root" in w for w in tu.sup_wald(ar(np.random.default_rng(s), 150, 0.5) + np.r_[np.zeros(75), 1.5 * np.ones(75)],
                                                  B=9, show=False)["warnings"]) for s in range(10)]
check("sup_wald unit-root warning (Zivot-Andrews): random walks yes (≥ 8/10), stationary series with a break no (0/10)",
      sum(w_rw) >= 8 and sum(w_br) == 0, f"{sum(w_rw)}/10, {sum(w_br)}/10")
check("a break in the middle: no edge warning, no CUSUM caveat", sw3["p_bootstrap"] < 0.05 and not sw3["warnings"]
      and "OLS-CUSUM" not in sw3["verdict"])

# ---------------------------------------------------------------- 4. cross_check with a loose tolerance
c = rp.cross_check({"combination": 5.32, "AR bootstrap": 5.46}, se=2.093, label="12-month IPCA", show=False)
check("cross_check: the IPCA case is flagged as a weak check, with the spread",
      c["agree"] and c["weak"] and any("weak check" in w and "0.14" in w for w in c["warnings"]))
c2 = rp.cross_check({"dr": 1.00, "reg": 1.03}, se=0.1, show=False)
check("cross_check: a tight tolerance (10% of the estimate) is not flagged", c2["agree"] and not c2["weak"] and not c2["warnings"])
c0 = rp.cross_check({"a": 0.002, "b": 0.0021}, se=0.01, show=False)
check("cross_check: an estimate that is statistically zero is not flagged as a weak check", c0["agree"] and not c0["weak"])
c4 = rp.cross_check({"a": 5.0, "b": 9.0}, se=2.0, show=False)
check("cross_check: a disagreement is reported as such, not as a weak agreement", not c4["agree"] and not c4["weak"])
c3 = rp.cross_check({"a": 0.123456789, "b": 0.123456789 + 1e-12}, show=False)
check("cross_check: implementation checks are never flagged as weak", c3["kind"] == "implementation" and not c3["weak"])

# ---------------------------------------------------------------- 5. matplotlib as an optional extra
check("setup_env: optional extra 'plots' is matplotlib", se.OPTIONAL.get("plots") == {"matplotlib": ("matplotlib", "3.8")}
      and "plots" in se.OPTIONAL_USE)
skill = (SKILL / "SKILL.md").read_text(encoding="utf-8")
check("SKILL.md: charts go through --extra plots in the same install question, or the script says the chart is optional",
      "`--extra plots` (matplotlib)" in skill and "instead of promising a PNG" in skill)

# ---------------------------------------------------------------- documentation
rules = (SKILL / "references" / "interpretation-rules.md").read_text(encoding="utf-8")
ts = (SKILL / "references" / "time-series.md").read_text(encoding="utf-8")
check("interpretation rules for v2.4.6.1 (short memory first, disagreeing break tests, weak agreement)",
      "## Rules added in v2.4.6.1" in rules and "short memory is ruled out" in rules and "Break tests that disagree" in rules
      and "only as good as its tolerance" in rules)
check("time-series.md: the short-memory check, the long-run CUSUM and the edge readings",
      "short-memory AR(p)" in ts and "edge of the trimmed window" in ts and "long-run SD" in ts)

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
