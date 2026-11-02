"""Tests for v2.3.1, the fixes from the v2.3.0 blind evals: cluster bootstrap in causal.att_gt,
causal.bacon_decomposition, the extended causal.twfe_covariate_diagnostic (benchmark covariates,
cluster, Goodman-Bacon check, low-power verdict), reporting.cross_check with att_gt's Series and
with fewer than two routes, and the ccc_test advice for the 'time' selection.

Checks: Goodman-Bacon weights sum to one and reproduce the TWFE coefficient exactly (with
never- and always-treated groups, and without never-treated units); with state-level shocks the
municipality bootstrap under-covers and the state cluster bootstrap does not, with the same point
estimate; the diagnostic flags hidden linearity in the review scenario of the eval when the
benchmark uses the valid covariates, and calls the eval's call inconclusive rather than
"no evidence"; a one-route cross-check reports that nothing was verified.

Run from the repository root:  python3 tests/test_did_v231.py
"""
import contextlib
import io
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "econometrics" / "scripts"))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import causal as ca
import reporting as rp

ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


# ---------------------------------------------------------------- 1. Goodman-Bacon decomposition
def stagger(seed, cohorts, n=150, T=8, slope=0.5):
    rng = np.random.default_rng(seed)
    gs = rng.choice(cohorts, n)
    rows = []
    for i in range(n):
        a = rng.normal()
        for t in range(1, T + 1):
            k = t - gs[i] if gs[i] > 0 and t >= gs[i] else -1
            rows.append((i, t, gs[i], a + 0.1 * t + (slope * (k + 1) if k >= 0 else 0) + rng.normal()))
    d = pd.DataFrame(rows, columns=["id", "t", "g", "y"])
    d["D"] = ((d.g > 0) & (d.t >= d.g)).astype(float)
    return d


for label, cohorts in (("never + always-treated", [0, 1, 3, 5, 7]), ("no never-treated", [2, 4, 6]),
                       ("never-treated only one cohort", [0, 4])):
    d = stagger(1, cohorts)
    r = ca.bacon_decomposition(d, "id", "t", "y", "g", show=False)
    b = smf.ols("y ~ D + C(id) + C(t)", d).fit().params["D"]
    check(f"Bacon: weighted 2x2s equal the TWFE coefficient exactly ({label})",
          abs(r["reconstructed"] - b) < 1e-10 and abs(r["twfe"] - b) < 1e-10, f"gap {abs(r['reconstructed'] - b):.1e}")
    check(f"Bacon: weights sum to one and are positive ({label})",
          abs(r["table"]["weight"].sum() - 1) < 1e-12 and (r["table"]["weight"] > 0).all())
d = stagger(2, [2, 4, 6])
r = ca.bacon_decomposition(d, "id", "t", "y", "g", show=False)
check("Bacon: growing effects without never-treated -> large forbidden weight, warned",
      r["forbidden_weight"] > 0.3 and any("already-treated" in w for w in r["warnings"]), f"{r['forbidden_weight']:.0%}")
d = stagger(3, [0, 4], slope=0.5)
r = ca.bacon_decomposition(d, "id", "t", "y", "g", show=False)
check("Bacon: one cohort plus never-treated has no forbidden comparisons", r["forbidden_weight"] == 0)


# ---------------------------------------------------------------- 2. cluster bootstrap
def state_panel(seed, S=40, m=10, T=6):
    rng = np.random.default_rng(seed)
    gs = rng.choice([0, 3, 5], S, p=[.5, .25, .25])
    shock = rng.normal(0, 0.5, (S, T))
    rows = []
    for s in range(S):
        a = rng.normal(0, 1, m)
        for t in range(1, T + 1):
            y = a + 0.2 * t + shock[s, t - 1] + 1.0 * ((gs[s] > 0) & (t >= gs[s])) + rng.normal(0, 0.5, m)
            rows.append(pd.DataFrame({"id": s * 100 + np.arange(m), "st": s, "t": t, "y": y, "g": gs[s],
                                      "x": rng.normal(size=m)}))
    return pd.concat(rows, ignore_index=True)


cu, cc = [], []
for k in range(40):
    p = state_panel(k)
    ou = ca.att_gt(p, "id", "t", "y", "g", n_boot=99, seed=k, show=False)["overall"]
    oc = ca.att_gt(p, "id", "t", "y", "g", n_boot=99, seed=k, cluster="st", show=False)["overall"]
    if k == 0:
        check("cluster bootstrap leaves the point estimate unchanged", ou["estimate"] == oc["estimate"])
        check("cluster bootstrap reports the cluster and the number of clusters",
              oc["cluster"] == "st" and oc["n_clusters"] == 40)
    cu.append(ou["ci_low"] <= 1 <= ou["ci_high"])
    cc.append(oc["ci_low"] <= 1 <= oc["ci_high"])
check("with state shocks, the unit bootstrap under-covers (< 80%)", np.mean(cu) < 0.8, f"{np.mean(cu):.0%}")
check("with state shocks, the state cluster bootstrap covers (>= 85%)", np.mean(cc) >= 0.85, f"{np.mean(cc):.0%}")
p = state_panel(1, S=12)
r = ca.att_gt(p, "id", "t", "y", "g", n_boot=49, cluster="st", show=False)
check("few clusters -> warning that points to randomization inference",
      any("randomization_inference" in w for w in r["warnings"]))
bad = state_panel(2).copy()
bad.loc[bad.index[0], "st"] = 999
try:
    ca.att_gt(bad, "id", "t", "y", "g", n_boot=9, cluster="st", show=False)
    check("a cluster that varies within unit raises a clear error", False)
except ValueError as e:
    check("a cluster that varies within unit raises a clear error", "constant within" in str(e))
p = state_panel(3)
ts = ca.did_timing_sensitivity(p, "id", "t", "y", "g", max_shift=1, n_boot=19, cluster="st", show=False)
base = ca.att_gt(p, "id", "t", "y", "g", n_boot=19, cluster="st", show=False)
check("did_timing_sensitivity passes cluster= (shift 0 = att_gt with the same cluster bootstrap)",
      abs(ts["shifts"].loc[0, "se"] - base["overall"]["se"]) < 1e-12)


# ---------------------------------------------------------------- 3. TWFE diagnostic, review scenario of eval 2
def review_panel(seed=20261102, n_uf=27, mun=12, anos=range(2006, 2020)):
    rng = np.random.default_rng(seed)
    anos = list(anos)
    ufs = [f"UF{i:02d}" for i in range(n_uf)]
    coortes = {u: 0 for u in ufs}
    for u, g in zip(rng.choice(ufs, 9, replace=False), [2010] * 3 + [2013] * 3 + [2016] * 3):
        coortes[u] = g
    choque = {u: rng.normal(0, 0.08, len(anos)) for u in ufs}
    rows = []
    for u in ufs:
        renda_uf = rng.normal(0, 0.6) + (0.5 if coortes[u] else 0.0)
        for j in range(mun):
            a_m, renda0, jov0 = rng.normal(np.log(30), 0.4), renda_uf + rng.normal(0, 0.3), rng.normal(0.18, 0.02)
            for k, t in enumerate(anos):
                g = coortes[u]
                expo = (t - g + 1) if (g and t >= g) else 0
                ly = a_m - 0.015 * k - 0.025 * k * renda0 - (min(0.04 * expo, 0.2) if expo else 0) \
                    + choque[u][k] + rng.normal(0, 0.15)
                rows.append(dict(municipio=f"{u}_{j}", uf=u, ano=t, y=ly, renda=renda0 + 0.02 * k + rng.normal(0, 0.05),
                                 pop_jovem=jov0 - 0.002 * k + rng.normal(0, 0.003),
                                 desemprego=0.1 + 0.05 * (ly - np.log(30)) + rng.normal(0, 0.01), ano_lei=g))
    return pd.DataFrame(rows)


rv = review_panel()
COV = ["renda", "desemprego", "pop_jovem"]
good = ca.twfe_covariate_diagnostic(rv, "municipio", "ano", "y", "ano_lei", COV, cs_covariates=["renda", "pop_jovem"],
                                    x_periods="base", cluster="uf", n_boot=49, seed=1, show=False)
check("diagnostic with valid benchmark covariates and state clusters: unreliable via hidden linearity",
      good["verdict"] == "unreliable" and good["hidden_linearity_p"] < 0.05,
      f"p = {good['hidden_linearity_p']:.3g}, L_renda {good['level_coefficients'].loc['L_renda', 'coef']:.3f} (true -0.025)")
check("diagnostic: benchmark and hidden-linearity test use cs_covariates only",
      good["cs_covariates"] == ["renda", "pop_jovem"] and "L_desemprego" not in good["level_coefficients"].index)
check("diagnostic reports the Goodman-Bacon weights", set(good["bacon"].index) >= {"treated vs never"})
naive = ca.twfe_covariate_diagnostic(rv, "municipio", "ano", "y", "ano_lei", COV, n_boot=49, seed=1, show=False)
check("diagnostic never says 'no evidence' when the benchmark is too noisy (eval-2 call)",
      naive["verdict"] in ("unreliable", "inconclusive (low power)"), naive["verdict"])
rng = np.random.default_rng(3)
rows = []
gb = np.where(rng.random(600) < .5, 4, 0)
ab, xb = rng.normal(0.5 * (gb > 0), 1), rng.normal(0.8 * (gb > 0), 1)
for t in range(1, 7):
    xb = xb + rng.normal(0, .3, 600)
    rows.append(pd.DataFrame({"id": np.arange(600), "t": t, "x": xb, "g": gb,
                              "y": ab + xb + 0.3 * t + ((gb > 0) & (t >= gb)) + rng.normal(0, 1, 600)}))
benign = ca.twfe_covariate_diagnostic(pd.concat(rows), "id", "t", "y", "g", ["x"], n_boot=49, show=False)
check("diagnostic still quiet in a benign, precise design", benign["verdict"] == "no evidence against TWFE",
      benign["verdict"])

# ---------------------------------------------------------------- 4. cross_check and texts
res = ca.att_gt(rv, "municipio", "ano", "y", "ano_lei", n_boot=19, covariates=["renda", "pop_jovem"], show=False)
with contextlib.redirect_stdout(io.StringIO()):
    c_series = rp.cross_check(res["cross_check"], se=res["overall"]["se"])
    c_one = rp.cross_check({"CS DR": res["overall"]["estimate"]}, se=res["overall"]["se"])
    c_nested = rp.cross_check({"a": {"estimate": 1.0}, "b": {"estimate": 1.0 + 1e-9}})
check("cross_check takes att_gt's cross_check Series as it is (three routes)", len(c_series["table"]) == 3
      and c_series["agree"] in (True, False))
check("cross_check with one route reports that nothing was verified (agree=None, warning)",
      c_one["agree"] is None and "nothing was verified" in c_one["warnings"][0])
check("cross_check accepts dicts of result dicts", c_nested["agree"] is True)
ccc = ca.ccc_test(rv, "y", ["renda"], "uf", "ano", gvar="ano_lei", n_boot=19, show=False)
src = (ROOT / "econometrics" / "scripts" / "causal.py").read_text()
check("ccc 'time' advice: x_periods='base' for baseline covariates, 'both' only for time-varying ones",
      "x_periods='base' with baseline or time-invariant covariates" in src)
import re as _re
_top = _re.search(r"^## v(\S+)", (ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), _re.M).group(1)
check("reporting.SKILL_VERSION matches the latest CHANGELOG version", rp.SKILL_VERSION == _top, f"{rp.SKILL_VERSION} vs {_top}")

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
