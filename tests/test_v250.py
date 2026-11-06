"""Tests for v2.5.0: DiD design, imputation (Borusyak-Jaravel-Spiess), pre-trend tests, and the GARCH half-life CI.

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

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
