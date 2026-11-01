"""Tests for v2.2 tail-risk tools in risk.py: peaks over threshold (GPD), rolling EVT VaR/ES
(McNeil-Frey conditional EVT), Expected Shortfall backtests (McNeil-Frey bootstrap-t,
Acerbi-Szekely Z2) and the FZ0 scoring function.

Checks: GPD parameters and Smith (1987) SEs on exact GPD data (Monte Carlo coverage); tail index
1/nu and far-tail VaR/ES of a Student-t; mean-excess slope sign; threshold stability; ES backtest
size under a correct model and power against normal forecasts of fat-tailed returns; conditional
EVT passes VaR and ES backtests on GARCH-t returns where EWMA-normal fails; FZ0 is minimized at
the true (VaR, ES); Z2 matches its formula.

Run from the repository root:  python3 tests/test_evt_es.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "econometrics" / "scripts"))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats

import risk

ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


def t_var_es(alpha, nu, scale=1.0):
    q = stats.t.ppf(alpha, nu)
    es = (stats.t.pdf(q, nu) / alpha) * (nu + q * q) / (nu - 1)
    return -q * scale, es * scale


# ---------------------------------------------------------------------------------------------
# GPD on exact GPD exceedances: estimates and SE coverage
# ---------------------------------------------------------------------------------------------
xi0, b0 = 0.3, 0.02
cover, xis = [], []
for s in range(200):
    y = stats.genpareto.rvs(xi0, scale=b0, size=500, random_state=s)
    # threshold 0, every point is an exceedance: pass the losses as negative returns
    g = risk.gpd_fit(pd.Series(-y), threshold=0.0, alpha=0.001, show=False)
    xis.append(g["estimate"])
    cover.append(g["ci_low"] <= xi0 <= g["ci_high"])
check("GPD: mean xi-hat within 0.02 of 0.3 (n_u = 500)", abs(np.mean(xis) - xi0) < 0.02, f"{np.mean(xis):.3f}")
check("GPD: Smith (1987) SE gives 95% CI coverage in [0.91, 0.98]", 0.91 <= np.mean(cover) <= 0.98,
      f"{np.mean(cover):.3f}")

# ---------------------------------------------------------------------------------------------
# Student-t tail: xi = 1/nu, far-tail VaR and ES
# ---------------------------------------------------------------------------------------------
nu = 4
r = pd.Series(stats.t.rvs(nu, size=20000, random_state=1) * 0.01)
g = risk.gpd_fit(r, q=0.95, alpha=0.001, show=False)
tv, te = t_var_es(0.001, nu, 0.01)
check("t(4): 95% CI for xi contains 1/nu = 0.25", g["ci_low"] <= 0.25 <= g["ci_high"],
      f"xi = {g['estimate']:.3f} [{g['ci_low']:.3f}, {g['ci_high']:.3f}]")
check("t(4): 99.9% VaR CI contains the true quantile", g["VaR_ci"][0] <= tv <= g["VaR_ci"][1],
      f"{g['VaR']:.4f} vs {tv:.4f}")
check("t(4): 99.9% ES CI contains the true ES", g["ES_ci"][0] <= te <= g["ES_ci"][1], f"{g['ES']:.4f} vs {te:.4f}")
vn = -stats.norm.ppf(0.001) * r.std()
check("t(4): EVT VaR error less than a third of the normal VaR error",
      abs(g["VaR"] - tv) < abs(vn - tv) / 3, f"EVT {g['VaR']:.4f}, normal {vn:.4f}, true {tv:.4f}")
st = g["stability"]
check("threshold stability: xi within 0.1 across thresholds 92.5%-97.5%",
      np.ptp(st.loc[st["quantile"] >= 0.925, "xi"].to_numpy()) < 0.1, st["xi"].round(3).tolist())

me_t = risk.mean_excess(r, np.arange(0.90, 0.99, 0.01))
me_n = risk.mean_excess(pd.Series(np.random.default_rng(2).normal(size=20000)), np.arange(0.90, 0.99, 0.01))
slope = lambda m: np.polyfit(m["threshold"], m["mean_excess"], 1)[0]
check("mean excess: upward slope for t(4), downward for the normal", slope(me_t) > 0.1 and slope(me_n) < 0,
      f"{slope(me_t):.3f}, {slope(me_n):.3f}")

# ---------------------------------------------------------------------------------------------
# ES backtests: size under the true model, power against normal forecasts
# ---------------------------------------------------------------------------------------------
a, nu, T = 0.025, 5, 1000
v, e = t_var_es(a, nu)
pz, pm = [], []
for s in range(400):
    x = pd.Series(np.random.default_rng(5000 + s).standard_t(nu, T))
    o = risk.es_backtest(x, pd.Series(v, index=x.index), pd.Series(e, index=x.index), a,
                         n_sim=500, n_boot=1000, seed=s, show=False)
    pz.append(o["p_Z2"]); pm.append(o["mcneil_frey"]["p_boot"])
check("ES backtest size (true t(5) forecasts): Z2 rejects 1.5%-8% at 5%", 0.015 <= np.mean(np.array(pz) < 0.05) <= 0.08,
      f"{np.mean(np.array(pz) < 0.05):.3f}")
check("ES backtest size: McNeil-Frey bootstrap-t rejects 1.5%-8% at 5%",
      0.015 <= np.mean(np.array(pm) < 0.05) <= 0.08, f"{np.mean(np.array(pm) < 0.05):.3f}")
sd = np.sqrt(nu / (nu - 2))
vn, en = -stats.norm.ppf(a) * sd, stats.norm.pdf(stats.norm.ppf(a)) / a * sd
pm2 = []
for s in range(200):
    x = pd.Series(np.random.default_rng(100 + s).standard_t(nu, T))
    o = risk.es_backtest(x, pd.Series(vn, index=x.index), pd.Series(en, index=x.index), a,
                         n_sim=200, n_boot=1000, seed=s, show=False)
    pm2.append(o["mcneil_frey"]["p_boot"])
check("ES backtest power: normal forecasts of t(5) returns rejected by McNeil-Frey > 75%",
      np.mean(np.array(pm2) < 0.05) > 0.75, f"{np.mean(np.array(pm2) < 0.05):.3f}")

xs = pd.Series([-3.0, 0.5, -1.0, 0.2, -2.5]); vs = pd.Series([2.0] * 5); es_ = pd.Series([2.5] * 5)
o = risk.es_backtest(xs, vs, es_, alpha=0.2, n_sim=100, show=False)
z2_hand = (-3.0 - 2.5) / (5 * 0.2 * 2.5) + 1
check("Z2 matches sum r I / (T alpha ES) + 1", abs(o["Z2"] - z2_hand) < 1e-12, f"{o['Z2']:.4f}")
check("Acerbi-Szekely zones reported only for alpha = 2.5% and about 250 days", o["Z2_zone_AS"] == "n/a")


# ---------------------------------------------------------------------------------------------
# Rolling conditional EVT on GARCH-t returns
# ---------------------------------------------------------------------------------------------
def garch_t(T, seed, nu=5):
    rng = np.random.default_rng(seed)
    w, al, be = 0.02, 0.08, 0.9
    h = w / (1 - al - be)
    out = np.zeros(T)
    z = rng.standard_t(nu, T) * np.sqrt((nu - 2) / nu)
    for t in range(T):
        out[t] = np.sqrt(h) * z[t]
        h = w + al * out[t] ** 2 + be * h
    return pd.Series(out / 100)


rg = garch_t(5000, 3)
ev = risk.var_evt(rg, a, window=1000, q=0.9, filter="ewma")
er = risk.var_evt(rg, a, window=1000, q=0.9, filter="none")
nm = risk.var_ewma(rg, a)
check("var_evt uses only past data: first `window` rows are empty", ev["VaR"].iloc[:1000].isna().all()
      and ev["VaR"].iloc[1000:].notna().all())
bt = risk.backtest_var(rg, {"evt": ev["VaR"], "normal": nm["VaR"]}, a, show=False)
check("conditional EVT passes Kupiec and Christoffersen; EWMA-normal fails Kupiec",
      bt.loc["evt", "p_uc"] > 0.05 and bt.loc["evt", "p_ind"] > 0.05 and bt.loc["normal", "p_uc"] < 0.05,
      bt[["rate", "p_uc", "p_ind"]].round(3).to_dict())
o_ev = risk.es_backtest(rg, ev["VaR"], ev["ES"], a, show=False)
o_nm = risk.es_backtest(rg, nm["VaR"], nm["ES"], a, show=False)
check("ES backtest: conditional EVT not rejected, EWMA-normal rejected by Z2 and McNeil-Frey",
      o_ev["p_Z2"] > 0.05 and o_ev["mcneil_frey"]["p_boot"] > 0.05 and o_nm["p_Z2"] < 0.05
      and o_nm["mcneil_frey"]["p_boot"] < 0.05,
      f"EVT {o_ev['p_Z2']:.3f}/{o_ev['mcneil_frey']['p_boot']:.3f}, normal {o_nm['p_Z2']:.4f}/{o_nm['mcneil_frey']['p_boot']:.4f}")
cmp_ = risk.compare_es_models(rg, {"evt_ewma": ev, "evt_raw": er, "ewma_normal": nm}, a, show=False)
check("FZ0 ranks conditional EVT first; both others significantly worse",
      cmp_.index[0] == "evt_ewma" and (cmp_["p"].iloc[1:] < 0.05).all(), cmp_["mean_FZ0"].round(4).to_dict())

# ---------------------------------------------------------------------------------------------
# FZ0 strict consistency: the true (VaR, ES) has the lowest expected loss
# ---------------------------------------------------------------------------------------------
x = pd.Series(np.random.default_rng(9).standard_t(nu, 400000))
true = risk.fz_loss(x, pd.Series(v, index=x.index), pd.Series(e, index=x.index), a).mean()
pert = [risk.fz_loss(x, pd.Series(v * kv, index=x.index), pd.Series(e * ke, index=x.index), a).mean()
        for kv, ke in [(0.9, 1), (1.1, 1), (1, 0.9), (1, 1.1), (0.9, 0.9), (1.1, 1.1)]]
check("FZ0: true (VaR, ES) has lower mean loss than six perturbations", true < min(pert),
      f"{true:.4f} vs min {min(pert):.4f}")

# ---------------------------------------------------------------------------------------------
# Engle-Manganelli dynamic quantile test (v2.2 eval): size and power against a static model
# ---------------------------------------------------------------------------------------------
rej = []
for s_ in range(300):
    x = pd.Series(np.random.default_rng(s_).standard_normal(1500))
    rej.append(risk.dq_test(x, pd.Series(-stats.norm.ppf(0.01), index=x.index), 0.01)["p_dq"] < 0.05)
check("DQ test size with the true VaR: 2%-9% at 5%", 0.02 <= np.mean(rej) <= 0.09, f"{np.mean(rej):.3f}")


def garch_n(T, seed):
    rng = np.random.default_rng(seed)
    h = 0.02 / (1 - 0.98)
    out = np.zeros(T)
    for t in range(T):
        out[t] = np.sqrt(h) * rng.standard_normal()
        h = 0.02 + 0.08 * out[t] ** 2 + 0.9 * h
    return pd.Series(out)


pdq, pind = [], []
for s_ in range(60):
    rr = garch_n(2250, s_)
    vh = risk.var_historical(rr, 0.01, 250)
    pdq.append(risk.dq_test(rr, vh, 0.01)["p_dq"] < 0.05)
    pind.append(risk.christoffersen(rr, vh, 0.01)["p_ind"] < 0.05)
check("DQ test detects a static historical VaR on GARCH data (> 85%) far more often than Christoffersen",
      np.mean(pdq) > 0.85 and np.mean(pdq) > np.mean(pind) + 0.4, f"DQ {np.mean(pdq):.2f}, LR_ind {np.mean(pind):.2f}")
check("backtest_var reports the DQ p-value", "p_dq" in bt.columns)

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
