"""Tests for v2.4.6: univariate time series (ts_univariate.py), four teaching demos, and n for a given effect.

Every estimator is checked against an independent computation (statsmodels or a closed form written here),
and the bootstrap procedures by small simulations of size, power or coverage. The v2.4.5 Windows test showed
the agent quoting a sample size read off a power table built for another effect (26 per group for 0.8 SD
instead of about 19 for the observed difference): inference.n_for_power computes it for the effect asked.

Run from the repository root:  python3 tests/test_v246.py
"""
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import special

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "econometrics"
sys.path.insert(0, str(SKILL / "scripts"))
import _console  # noqa: E402,F401
import inference  # noqa: E402
import simulation  # noqa: E402
import ts_univariate as tu  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


def ar(rng, n, phis, c=0.0, burn=200):
    e = rng.normal(size=n + burn)
    y = np.zeros(n + burn)
    for t in range(len(phis), n + burn):
        y[t] = c + sum(p * y[t - 1 - i] for i, p in enumerate(phis)) + e[t]
    return y[burn:]


rng = np.random.default_rng(2046)

# ---------------------------------------------------------------- 1. n for a given effect
from statsmodels.stats.power import TTestIndPower, TTestPower  # noqa: E402
for eff, sd in ((8.3, 8.87), (0.8, 1.0), (0.3, 1.0)):
    r = inference.n_for_power(eff, sd, show=False)
    ref = int(np.ceil(TTestIndPower().solve_power(effect_size=eff / sd, alpha=0.05, power=0.8)))
    check(f"n_for_power two-sample d = {eff / sd:.3f}: n = ceil(statsmodels solve_power)", r["n"] == ref, f"{r['n']} vs {ref}")
r = inference.n_for_power(8.3, 8.87, show=False)
check("the v2.4.5 case: 8.3 points with SD 8.87 needs 19 per group, not the 26 of a 0.8-SD table",
      r["n"] == 19 and inference.n_for_power(0.8, 1.0, show=False)["n"] == 26)
rp = inference.n_for_power(0.5, 1.0, design="paired", show=False)
ref = int(np.ceil(TTestPower().solve_power(effect_size=0.5, alpha=0.05, power=0.8)))
check("n_for_power paired = statsmodels TTestPower", rp["n"] == ref, f"{rp['n']} vs {ref}")
check("n_for_power: n is the smallest (power at n − 1 below target)",
      inference.power_curve(8.3, r["n"] - 1, sd=8.87, show=False)["power"].iloc[0] < 0.8 <= r["power"])
r2 = inference.n_for_power(0.5, 1.0, ratio=2.0, show=False)
ref = int(np.ceil(TTestIndPower().solve_power(effect_size=0.5, alpha=0.05, power=0.8, ratio=2.0)))
check("n_for_power with unequal groups (ratio 2) = statsmodels; n_total = n + 2n", r2["n"] == ref and r2["n_total"] == 3 * ref,
      f"{r2['n']} vs {ref}")

# ---------------------------------------------------------------- 2. identification
wn = rng.normal(size=400)
idn = tu.ts_identify(wn, show=False)
check("ts_identify: lag-1 ACF band = 1.96/√n (Bartlett starts at the white-noise band)",
      abs(idn["table"]["acf_band"].iloc[0] - 1.959964 / np.sqrt(400)) < 1e-6)
r_sm = sm.tsa.acf(wn, nlags=5, fft=True)[1:6]
check("ts_identify: ACF = statsmodels acf", np.allclose(idn["table"]["acf"].iloc[:5], r_sm))
check("ts_identify: white noise → Ljung-Box does not reject at 12 lags", idn["ljung_box"].loc[12, "lb_pvalue"] > 0.05)
y_ar2 = ar(rng, 1000, [0.5, 0.3])
idn2 = tu.ts_identify(y_ar2, show=False)
check("ts_identify: AR(2) with n = 1000 → PACF cutoff 2 and BIC grid picks p = 2, q = 0",
      idn2["pacf_cutoff"] == 2 and idn2["bic_best"] == (2, 0), f"{idn2['pacf_cutoff']}, {idn2['bic_best']}")

# ---------------------------------------------------------------- 3. filters
from statsmodels.tsa.filters.hp_filter import hpfilter  # noqa: E402
idx = pd.period_range("1980Q1", periods=160, freq="Q").to_timestamp()
y = pd.Series(np.cumsum(rng.normal(0.5, 1, 160)), index=idx)
d_hp = tu.detrend(y, "hp", show=False)
check("detrend hp: λ = 1600 for quarterly data and trend = statsmodels hpfilter",
      d_hp["lamb"] == 1600 and np.allclose(d_hp["trend"], hpfilter(y.to_numpy(), 1600)[1]))
d_rt = tu.detrend(y, "hp", one_sided=True, show=False)
check("detrend hp one-sided: trend at t = last value of the HP trend on data up to t",
      np.isclose(d_rt["trend"].iloc[99], hpfilter(y.iloc[:100].to_numpy(), 1600)[1][-1])
      and np.isclose(d_rt["trend"].iloc[-1], d_hp["trend"].iloc[-1]))
check("detrend hp: end-point revision reported and warned", d_hp["end_revision"]["typical_revision"] > 0
      and any("revised" in w for w in d_hp["warnings"]))
d_h = tu.detrend(y, "hamilton", show=False)
yv = y.to_numpy()
rows = np.arange(3, 160 - 8)
X = sm.add_constant(np.column_stack([yv[rows - j] for j in range(4)]))
res = sm.OLS(yv[rows + 8], X).fit()
check("detrend hamilton: cycle = residual of y(t+8) on y(t..t−3) (statsmodels OLS)",
      np.allclose(d_h["cycle"].iloc[rows + 8].to_numpy(), res.resid) and d_h["cycle"].iloc[:11].isna().all())
d_bk = tu.detrend(y, "bk", show=False)
check("detrend bk: 12 observations lost at each end", d_bk["cycle"].iloc[:12].isna().all()
      and d_bk["cycle"].iloc[-12:].isna().all() and d_bk["cycle"].iloc[12:-12].notna().all())
try:
    tu.detrend(pd.Series(yv, index=pd.date_range("2020-01-01", periods=160, freq="D")), "hp", show=False)
    raised = False
except ValueError:
    raised = True
check("detrend: daily data without λ raises instead of using quarterly settings", raised)
d_p = tu.detrend(y, "poly", degree=1, show=False)
t_ = np.arange(160)
check("detrend poly: linear trend = OLS on time", np.allclose(d_p["trend"], sm.OLS(yv, sm.add_constant(t_)).fit().fittedvalues))

# ---------------------------------------------------------------- 4. ADF: statistic, lags, bootstrap
from statsmodels.tsa.stattools import adfuller  # noqa: E402
rw = np.cumsum(rng.normal(size=250))
for trend in ("n", "c", "ct"):
    a = tu.adf_bootstrap(rw, trend=trend, lags=3, B=49, show=False)
    check(f"adf_bootstrap τ = statsmodels adfuller (trend {trend}, 3 lags)",
          np.isclose(a["statistic"], adfuller(rw, maxlag=3, regression=trend, autolag=None)[0]))
y_l = ar(rng, 300, [1.3, -0.3])                   # unit root with an AR(1) in the differences
pmax = int(np.floor(12 * (300 / 100) ** 0.25))


def last_t(p):
    yy = y_l[pmax - p:]
    dy = np.diff(yy)
    X = np.column_stack([yy[p:-1]] + [dy[p - i:len(dy) - i] for i in range(1, p + 1)] + [np.ones(len(dy) - p)])
    return sm.OLS(dy[p:], X).fit().tvalues[p]


def aic_k(p):                                       # AIC of the ADF regression with p lags, common sample, by hand
    yy = y_l[pmax - p:]
    dy = np.diff(yy)
    X = np.column_stack([yy[p:-1]] + [dy[p - i:len(dy) - i] for i in range(1, p + 1)] + [np.ones(len(dy) - p)])
    e = sm.OLS(dy[p:], X).fit().resid
    T = len(e)
    return T * np.log(e @ e / T) + 2 * X.shape[1]
check("adf_lags default (AIC) = the AIC minimum computed by hand on the common sample", tu.adf_lags(y_l) == int(np.argmin([aic_k(q) for q in range(pmax + 1)])),
      f"{tu.adf_lags(y_l)} vs {int(np.argmin([aic_k(q) for q in range(pmax + 1)]))}")
p_t = tu.adf_lags(y_l, rule="t16")
check("adf_lags t ≥ 1.645 rule: last lag kept is significant, every longer one is not (statsmodels OLS, common sample)",
      p_t >= 1 and abs(last_t(p_t)) >= 1.645 and all(abs(last_t(q)) < 1.645 for q in range(p_t + 1, pmax + 1)),
      f"p = {p_t}, pmax = {pmax}")
ma_k, bic_k = [], []
for s in range(30):
    e = np.random.default_rng(s).normal(size=201)
    yy = np.cumsum(e[1:] - 0.5 * e[:-1])
    ma_k.append(tu.adf_lags(yy, rule="maic")); bic_k.append(tu.adf_lags(yy, rule="bic"))
check("adf_lags maic: with MA(−0.5) errors under H0 it picks more lags than BIC on average (Ng-Perron)",
      np.mean(ma_k) > np.mean(bic_k), f"{np.mean(ma_k):.1f} vs {np.mean(bic_k):.1f}")
R = 60
rej = [tu.adf_bootstrap(np.cumsum(np.random.default_rng(s).normal(size=100)), B=99, seed=s, show=False)["p_bootstrap"] < 0.05
       for s in range(R)]
check("adf_bootstrap size under a random walk ≈ 5% (60 replications, at most 15%)", np.mean(rej) <= 0.15, f"{np.mean(rej):.3f}")
pw = [tu.adf_bootstrap(ar(np.random.default_rng(100 + s), 150, [0.5]), B=99, seed=s, show=False)["p_bootstrap"] < 0.05
      for s in range(20)]
check("adf_bootstrap (AIC lags) rejects a stationary AR(0.5), n = 150 (power ≥ 90%)", np.mean(pw) >= 0.9, f"{np.mean(pw):.2f}")
a_rw = tu.adf_bootstrap(np.cumsum(np.random.default_rng(31).normal(size=200)), B=99, show=False)
a_st = tu.adf_bootstrap(ar(np.random.default_rng(32), 200, [0.3]), B=99, show=False)
check("adf_bootstrap verdicts: a non-rejection is not called evidence of a unit root; a rejection says stationary",
      a_rw["p_bootstrap"] >= 0.05 and "not evidence" in a_rw["verdict"]
      and a_st["p_bootstrap"] < 0.05 and a_st["verdict"].startswith("rejects the unit root"))
rw0 = np.cumsum(np.random.default_rng(33).normal(size=120))
check("adf_bootstrap with 0 lags (trend 'ct') runs and τ = adfuller",
      all(np.isclose(tu.adf_bootstrap(rw0, trend=tr_, lags=0, B=19, show=False)["statistic"],
                     adfuller(rw0, maxlag=0, regression=tr_, autolag=None)[0]) for tr_ in ("n", "c", "ct")))

# ---------------------------------------------------------------- 5. AR forecasts with bootstrap intervals
y1 = ar(rng, 200, [0.6], c=1.0)
f = tu.ar_bootstrap_forecast(y1, h=5, p=2, B=99, show=False)
ref = sm.tsa.AutoReg(y1, lags=2, trend="c").fit()
check("ar_bootstrap_forecast: point forecasts = statsmodels AutoReg",
      np.allclose(f["table"]["forecast"].to_numpy(), ref.forecast(5)))
pr = ref.get_prediction(start=200, end=204)
se_ref = np.sqrt(pr.var_pred_mean)
half = (f["table"]["analytic_high"] - f["table"]["analytic_low"]).to_numpy() / 2 / 1.959964
check("ar_bootstrap_forecast: analytic SE = statsmodels AutoReg prediction SE (σ̂ with the same df)",
      np.allclose(half, se_ref, rtol=0.02), f"{half[:2]} vs {se_ref[:2]}")
cover = []
for s in range(120):
    g = np.random.default_rng(500 + s)
    yy = ar(g, 81, [0.7])
    fc = tu.ar_bootstrap_forecast(yy[:80], h=1, p=1, B=199, seed=s, show=False)["table"].iloc[0]
    cover.append(fc["boot_low"] <= yy[80] <= fc["boot_high"])
check("ar_bootstrap_forecast: 95% bootstrap interval covers the next value about 95% of the time (120 reps, 88–100%)",
      0.88 <= np.mean(cover), f"{np.mean(cover):.3f}")
cb, ca = [], []
for s in range(150):
    g = np.random.default_rng(1500 + s)
    yy = ar(g, 34, [0.8])
    fc = tu.ar_bootstrap_forecast(yy[:30], h=4, p=1, B=199, seed=s, show=False)["table"].iloc[3]
    cb.append(fc["boot_low"] <= yy[33] <= fc["boot_high"])
    ca.append(fc["analytic_low"] <= yy[33] <= fc["analytic_high"])
check("ar_bootstrap_forecast, n = 30, φ = 0.8, h = 4: the bias-corrected bootstrap covers more than the analytic interval",
      np.mean(cb) > np.mean(ca) and np.mean(cb) >= 0.87, f"bootstrap {np.mean(cb):.3f}, analytic {np.mean(ca):.3f}")
f_y = tu.ar_bootstrap_forecast(pd.Series(y1[:60], index=range(1960, 2020)), h=3, p=1, B=19, show=False)["table"]
f_m = tu.ar_bootstrap_forecast(pd.Series(y1[:60], index=pd.date_range("2015-01-31", periods=60, freq="ME")), h=2, p=1,
                               B=19, show=False)["table"]
check("ar_bootstrap_forecast labels the future periods (years continue, month ends continue)",
      f_y.index.tolist() == [2020, 2021, 2022] and f_m.index[0] == pd.Timestamp("2020-01-31"))
f0 = tu.ar_bootstrap_forecast(np.random.default_rng(34).normal(size=150), h=2, B=19, show=False)
check("ar_bootstrap_forecast: BIC can choose p = 0 for white noise, and the p = 0 path runs", f0["p"] == 0
      and np.allclose(f0["table"]["forecast"], f0["coef"][0]))
try:
    tu.ar_bootstrap_forecast(np.cumsum(rng.normal(size=200)) * 10 + np.arange(200) ** 1.5, p=1, show=False)
    raised = False
except ValueError:
    raised = True
check("ar_bootstrap_forecast refuses an explosive fit (difference first)", raised)

# ---------------------------------------------------------------- 6. breaks
yb = np.r_[rng.normal(0, 1, 120), rng.normal(1.2, 1, 80)]
sw = tu.sup_wald(yb, B=99, show=False)
tb = sw["break_position"]
D = (np.arange(200) >= tb).astype(float)
Xr, Xu = np.ones((200, 1)), np.column_stack([np.ones(200), D])
ssr0, ssr1 = sm.OLS(yb, Xr).fit().ssr, sm.OLS(yb, Xu).fit().ssr
check("sup_wald: sup-F = Chow F at the estimated date (statsmodels OLS)",
      np.isclose(sw["statistic"], (ssr0 - ssr1) / (ssr1 / 198)))
check("sup_wald: finds the mean shift near observation 120 and rejects",
      abs(tb - 120) <= 8 and sw["p_bootstrap"] < 0.05, f"date {tb}, p {sw['p_bootstrap']:.3f}")
check("sup_wald: break searched only in the trimmed middle (15%–85%)",
      np.isnan(sw["F_path"].iloc[:29]).all() and np.isnan(sw["F_path"].iloc[171:]).all())
rej = [tu.sup_wald(ar(np.random.default_rng(900 + s), 150, [0.5]), B=99, seed=s, show=False)["p_bootstrap"] < 0.05
       for s in range(40)]
check("sup_wald with sieve bootstrap: size under a stable AR(0.5) ≤ 15% (40 reps)", np.mean(rej) <= 0.15, f"{np.mean(rej):.3f}")
x = rng.normal(size=200)
yx = 1 + np.where(np.arange(200) < 100, 0.5, 1.5) * x + rng.normal(size=200)
swx = tu.sup_wald(yx, pd.DataFrame({"x": x}), B=99, show=False)
idx_m = pd.date_range("2005-01-31", periods=200, freq="ME")
swd = tu.sup_wald(pd.Series(yx, index=idx_m), x.reshape(-1, 1), B=19, show=False)
check("sup_wald: y with dates and X as an array are aligned by position", swd["break_date"] == idx_m[swd["break_position"]]
      and np.isclose(swd["statistic"], tu.sup_wald(yx, pd.DataFrame({"x": x}), B=19, show=False)["statistic"]))
orders = [tu.sup_wald(ar(np.random.default_rng(41), 150, [0.6]) * sc, B=9, show=False)["sieve_ar_order"] for sc in (1e-3, 1, 1e3)]
check("sup_wald: the sieve AR order does not depend on the units of y", len(set(orders)) == 1, str(orders))
check("sup_wald on a random walk warns that a break test in levels assumes stationarity",
      any("assumes" in w for w in tu.sup_wald(np.cumsum(np.random.default_rng(42).normal(size=150)), B=9, show=False)["warnings"]))
check("sup_wald with a regressor: detects a slope change; recursive coefficients reported",
      swx["p_bootstrap"] < 0.05 and list(swx["recursive_coefficients"].columns) == ["const", "x"])

# ---------------------------------------------------------------- 7. long memory
def arfima0(g, n, d, burn=2000):
    j = np.arange(n + burn)
    psi = np.exp(special.gammaln(j + d) - special.gammaln(j + 1) - special.gammaln(d))
    e = g.normal(size=n + burn)
    return np.convolve(e, psi)[:n + burn][burn:]
est = [tu.long_memory(arfima0(np.random.default_rng(s), 2000, 0.3), show=False) for s in range(30)]
dl, dg = np.mean([e["d_local_whittle"] for e in est]), np.mean([e["d_gph"] for e in est])
check("long_memory recovers d = 0.3 from ARFIMA(0, d, 0) built by its MA(∞) weights (mean of 30, n = 2000)",
      abs(dl - 0.3) < 0.05 and abs(dg - 0.3) < 0.08, f"LW {dl:.3f}, GPH {dg:.3f}")
sd_lw = np.std([e["d_local_whittle"] for e in est])
check("long_memory: local Whittle SE 1/(2√m) close to the simulated SD",
      0.6 < est[0]["table"].loc["local Whittle", "se"] / sd_lw < 1.6, f"{est[0]['table'].loc['local Whittle', 'se']:.3f} vs {sd_lw:.3f}")
lw0 = np.mean([tu.long_memory(np.random.default_rng(s).normal(size=1000), show=False)["d_local_whittle"] for s in range(20)])
check("long_memory: white noise gives d ≈ 0", abs(lw0) < 0.05, f"{lw0:.3f}")
xs = rng.normal(size=300)
check("frac_diff: (1 − L)^{−d}(1 − L)^d x = x", np.allclose(tu.frac_diff(tu.frac_diff(xs, 0.4), -0.4), xs))
fd1 = tu.frac_diff(xs, 1.0).to_numpy()
lmn = tu.long_memory(np.diff(np.random.default_rng(43).normal(size=1001)), show=False)
check("long_memory: an over-differenced series is read as anti-persistent, not as 'no long memory'",
      "anti-persistent" in lmn["reading"], lmn["reading"])
check("frac_diff with d = 1 is the first difference", np.isclose(fd1[0], xs[0]) and np.allclose(fd1[1:], np.diff(xs)))
af = tu.arfima(arfima0(np.random.default_rng(7), 2000, 0.3) + 0 * 1, (0, 0), show=False)
check("arfima: d from local Whittle, ARMA part on the filtered series", abs(af["d"] - 0.3) < 0.1 and af["n_used"] == 1800)

# ---------------------------------------------------------------- 8. forecast combination
n = 200
truth = ar(rng, n, [0.5])
F = pd.DataFrame({"a": truth + rng.normal(0, 0.6, n), "b": truth + 0.3 + rng.normal(0, 1.0, n)})
tr = F.iloc[:100].sub(truth[:100], axis=0)
cmb = tu.combine_forecasts(truth, F, "inverse_mse", show=False)
inv = 1 / (tr ** 2).mean()
check("combine inverse_mse: weights ∝ 1/MSE on the training rows", np.allclose(cmb["weights"], inv / inv.sum()))
cmb = tu.combine_forecasts(truth, F, "min_variance", show=False)
Ev = tr.to_numpy()
m11, m22, m12 = np.mean(Ev[:, 0] ** 2), np.mean(Ev[:, 1] ** 2), np.mean(Ev[:, 0] * Ev[:, 1])
w1 = (m22 - m12) / (m11 + m22 - 2 * m12)
check("combine min_variance: Bates-Granger closed form for two forecasts, with mean squared errors",
      np.isclose(cmb["weights"]["a"], w1))
Fb = pd.DataFrame({"unbiased": truth + rng.normal(0, 1.0, n), "biased": truth + 3 + rng.normal(0, 0.5, n)})
cb2 = tu.combine_forecasts(truth, Fb, "min_variance", show=False)
check("combine min_variance: a biased but steady forecast gets little weight",
      cb2["weights"]["biased"] < 0.2, f"{cb2['weights']['biased']:.3f}")
cmb = tu.combine_forecasts(truth, F, "granger_ramanathan", show=False)
A = np.block([[2 * F.iloc[:100].to_numpy().T @ F.iloc[:100].to_numpy(), np.ones((2, 1))], [np.ones((1, 2)), np.zeros((1, 1))]])
rhs = np.r_[2 * F.iloc[:100].to_numpy().T @ truth[:100], 1.0]
check("combine granger_ramanathan: constrained LS (Lagrangian solution, weights sum to 1)",
      np.allclose(cmb["weights"], np.linalg.solve(A, rhs)[:2]) and np.isclose(cmb["weights"].sum(), 1))
eq = tu.combine_forecasts(truth, F, show=False)
check("combine equal (default): weights 1/k, evaluated after the training window",
      np.allclose(eq["weights"], 0.5) and eq["n_train"] == 100 and eq["n_test"] == 100
      and "combination (equal)" in eq["evaluation"].index)

# ---------------------------------------------------------------- 9. demos
d = simulation.demo_trend_vs_difference(R=200)["table"]
check("demo_trend_vs_difference: detrended random walk lag-1 ACF > 0.8; differenced trend-stationary ≈ (φ − 1)/2",
      d.iloc[0, 1] > 0.8 and abs(d.iloc[1, 0] - (-0.25)) < 0.05)
d = simulation.demo_ar1_forecast_variance(R=4000)["table"]
check("demo_ar1_forecast_variance: simulated variance within 10% of σ²(1 − φ^2h)/(1 − φ²) and h·σ² under a unit root",
      np.all(np.abs(d["simulated var"] / d["theory"] - 1) < 0.1))
d = simulation.demo_unit_root_power(R=200, T_values=(100,))["table"]
check("demo_unit_root_power: size near 5%, power rises as φ falls, low at φ = 0.95, T = 100",
      d["phi = 1.0"].iloc[0] < 0.1 and d["phi = 0.95"].iloc[0] < 0.35 and d["phi = 0.8"].iloc[0] > 0.7
      and d.iloc[0, 1:].is_monotonic_increasing)
d = simulation.demo_break_unit_root(R=150)["table"]
check("demo_break_unit_root: ignoring a large shift, the ADF stops rejecting; with the dummy it rejects",
      d.loc[8.0, "ADF ignoring the break"] < 0.1 and d.loc[8.0, "ADF with the break dummy"] > 0.9
      and d["ADF ignoring the break"].is_monotonic_decreasing)

# ---------------------------------------------------------------- 10. documentation and routing
skill = (SKILL / "SKILL.md").read_text(encoding="utf-8")
tsref = (SKILL / "references" / "time-series.md").read_text(encoding="utf-8")
fb = (SKILL / "references" / "forecasting-backtesting.md").read_text(encoding="utf-8")
rules = (SKILL / "references" / "interpretation-rules.md").read_text(encoding="utf-8")
wf = (SKILL / "references" / "workflow.md").read_text(encoding="utf-8")
teach = (SKILL / "references" / "teaching.md").read_text(encoding="utf-8")
check("SKILL.md lists ts_univariate.py and routes univariate questions to it", "ts_univariate.py" in skill
      and re.search(r"unit root|correlogram|structural break", skill, re.I) is not None)
fns = ["ts_identify", "detrend", "adf_bootstrap", "ar_bootstrap_forecast", "sup_wald", "long_memory", "arfima", "combine_forecasts"]
check("time-series.md points to every new function", all(f"{fn}(" in tsref for fn in fns), str([f for f in fns if f"{f}(" not in tsref]))
check("forecasting-backtesting.md carries an Armstrong checklist and combine_forecasts", "Armstrong" in fb and "combine_forecasts" in fb)
check("workflow.md: clean the series, then test for breaks, then for unit roots",
      re.search(r"break.{0,200}before.{0,40}unit.root|clean.{0,120}break.{0,120}unit.root", wf, re.I | re.S) is not None)
check("interpretation rule: n for an effect is computed with n_for_power, never read off a table",
      "n_for_power" in rules and "never read" in rules)
check("teaching.md curriculum map lists the four new demos",
      all(x in teach for x in ("demo_trend_vs_difference", "demo_ar1_forecast_variance", "demo_unit_root_power",
                               "demo_break_unit_root")))

import json  # noqa: E402
import reporting  # noqa: E402
mk = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))
top = re.search(r"^## v(\S+)", (ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), re.M).group(1)
check("same version (2.4.6 or later) in marketplace, reporting and CHANGELOG",
      mk["plugins"][0]["version"] == reporting.SKILL_VERSION == top and tuple(map(int, top.split("."))) >= (2, 4, 6),
      f"{mk['plugins'][0]['version']}, {reporting.SKILL_VERSION}, {top}")

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
