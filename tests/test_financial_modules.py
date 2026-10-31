"""Known-truth tests for forecasting, backtest, risk, dependence, state_space and regimes.

Run from the repository root:  python3 tests/test_financial_modules.py
Needs: statsmodels, arch, scikit-learn, scipy; hmmlearn for the HMM check.
"""
import sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "econometrics" / "scripts"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats

rng = np.random.default_rng(1)
idx = pd.bdate_range("2010-01-01", periods=2500)
ok = []

def check(name, cond, info=""):
    ok.append(cond); print(("PASS " if cond else "FAIL ") + name, info)

# ---------- backtest timing ----------
import backtest as bt
r = pd.Series(rng.normal(0, 0.01, 500), index=idx[:500])
# oracle position using r itself (look-ahead) vs correct lag
pos_peek = np.sign(r)
res = bt.run_backtest(pos_peek, r)
check("backtest lag kills look-ahead", abs(res["strategy_return"].mean()) < 0.002,
      f"mean {res['strategy_return'].mean():.5f}")
res0 = bt.run_backtest(pos_peek, r, lag=0)
check("lag=0 reproduces the look-ahead (sanity)", res0["strategy_return"].mean() > 0.007)
perf = bt.performance(res["strategy_return"])
check("performance table", {"Sharpe", "max_drawdown", "CAGR"} <= set(perf.index))
# TSMOM on a trending + noise series earns positive returns
tr = pd.Series(np.r_[np.linspace(0, 0.6, 600), np.linspace(0.6, 0.0, 600)], index=idx[:1200])
price = np.exp(tr + np.cumsum(rng.normal(0, 0.004, 1200)))
pos = bt.tsmom_position(pd.Series(price, index=idx[:1200]), lookback=60)
rr = bt.run_backtest(pos, price.pct_change().fillna(0) if isinstance(price, pd.Series) else pd.Series(price, index=idx[:1200]).pct_change().fillna(0))
check("TSMOM profits on a trending series", rr["strategy_return"].sum() > 0.2, f"{rr['strategy_return'].sum():.3f}")
sp = list(bt.walk_forward_splits(100, 50, 10))
check("walk-forward splits never overlap", all(t[0] > tr_[-1] for tr_, t in sp) and len(sp) == 5)

# ---------- forecasting ----------
import forecasting as fc
e = rng.normal(size=600)
y = np.zeros(600)
for t in range(2, 600):
    y[t] = 0.6 * y[t - 1] + e[t] + 0.4 * e[t - 1]
ys = pd.Series(y, index=pd.period_range("1970-01", periods=600, freq="M").to_timestamp())
best, rank = fc.auto_arima(ys, max_p=3, max_q=3, show=True)
check("auto_arima recovers ARMA(1,1) (d=0)", best.model.order[1] == 0 and best.model.order in [(1, 0, 1), (2, 0, 0), (2, 0, 1), (1, 0, 2)],
      str(best.model.order))
rw = pd.Series(np.cumsum(rng.normal(size=400)) + 50, index=ys.index[:400])
check("select_d = 1 for a random walk", fc.select_d(rw) == 1)
t0 = time.time()
ro = fc.rolling_origin(ys, (1, 0, 1), start=0.8, refit_every=12, trend="n")
check("rolling_origin length", len(ro) == 120, f"{len(ro)} in {time.time()-t0:.1f}s")
bm = fc.benchmarks(ys, start=0.8)
tab = fc.evaluate(ro["actual"], {"naive": bm["naive"], "hist_mean": bm["hist_mean"], "arma": ro["forecast"]},
                  benchmark="naive", last_actual=bm["naive"])
check("ARMA beats naive on ARMA data", tab.loc["arma", "RMSE"] < tab.loc["naive", "RMSE"])
# ARIMAX: x helps only if lagged x predicts y
x = pd.Series(rng.normal(size=600), index=ys.index)
y2 = 0.8 * x.shift(1).fillna(0) + rng.normal(scale=0.5, size=600)
X = fc.lag_exog(x, 1)
ro2 = fc.rolling_origin(y2, (0, 0, 0), start=0.8, exog=X, refit_every=12)
ro1 = fc.rolling_origin(y2, (0, 0, 0), start=0.8, refit_every=12)
t2 = fc.evaluate(ro2["actual"], {"no_x": ro1["forecast"], "arimax": ro2["forecast"]}, direction="return")
check("ARIMAX with lagged x wins and DM significant", t2.loc["arimax", "DM_p"] < 0.01)
pt = fc.pesaran_timmermann(ro2["actual"], ro2["forecast"])
check("Pesaran-Timmermann detects direction skill", pt["p_one_sided"] < 0.01, f"{pt}")
# VAR
v = np.zeros((700, 2))
for t in range(1, 700):
    v[t] = [0.5 * v[t-1, 0] + 0.3 * v[t-1, 1], 0.4 * v[t-1, 1]] + rng.normal(size=2)
vdf = pd.DataFrame(v, columns=["a", "b"], index=pd.bdate_range("2000-01-03", periods=700))
rv = fc.rolling_var(vdf, "a", start=0.8, refit_every=20)
check("rolling_var picks lag 1 mostly and indexes target dates", rv["lag_order"].mode()[0] == 1 and rv.index[0] == vdf.index[560])

# ---------- risk ----------
import risk
from arch import arch_model
from arch.univariate import StudentsT
sim = arch_model(None, vol="GARCH", p=1, q=1, dist="t")
# Seeded: without it every CI run simulates new data and the Kupiec check below (a test at 5%)
# fails in about 5% of runs with no bug. Across 40 seeds the GARCH-t VaR passed in 38 (95%),
# mean violation rate 1.09% for a 1% VaR, which is the expected behaviour of a correct model.
sim.distribution = StudentsT(seed=np.random.default_rng(0))
simd = sim.simulate([0.02, 0.05, 0.08, 0.9, 6.0], 3000)
rr = pd.Series(simd["data"].to_numpy() / 100, index=pd.bdate_range("2008-01-01", periods=3000))
ew = risk.ewma_volatility(rr)
check("EWMA uses only past returns", np.isclose(ew.iloc[31] ** 2, 0.94 * ew.iloc[30] ** 2 + 0.06 * rr.iloc[30] ** 2))
a = 0.01
v_h = risk.var_historical(rr, a, 250)
v_n = risk.var_parametric(rr, a, 250)["VaR"]
v_e = risk.var_ewma(rr, a)["VaR"]
v_f = risk.var_ewma(rr, a, method="fhs")["VaR"]
t0 = time.time()
gv = risk.var_garch(rr, a, window=1000, refit_every=50)
print(f"var_garch {time.time()-t0:.1f}s")
tab = risk.backtest_var(rr, {"hist": v_h, "normal": v_n, "ewma": v_e, "ewma_fhs": v_f, "garch_t": gv["VaR"]}, a)
check("GARCH-t VaR has correct coverage", tab.loc["garch_t", "p_uc"] > 0.05, f"rate {tab.loc['garch_t','rate']:.4f}")
check("Static normal VaR violations cluster or exceed", tab.loc["normal", "p_cc"] < tab.loc["garch_t", "p_cc"])
# ES >= VaR
check("ES >= VaR", (gv["ES"].dropna() >= gv["VaR"].dropna()).all())
# Kupiec exact
fake_r = pd.Series([-1.0] * 10 + [0.0] * 990); fake_v = pd.Series([0.5] * 1000)
check("Kupiec: 10 hits in 1000 at 1% -> p≈1", risk.kupiec_pof(fake_r, fake_v, 0.01)["p_uc"] > 0.99)
check("Basel: 4 viol green, 5 yellow, 10 red",
      [risk.basel_traffic_light(n) for n in (4, 5, 10)] == ["green", "yellow", "red"])
# t ES factor numeric check
nu = 5; z = stats.t.rvs(nu, size=2_000_000, random_state=3) * np.sqrt((nu - 2) / nu)
q = np.quantile(z, 0.01)
check("t ES factor matches simulation", abs(risk._es_factor("t", 0.01, nu) - z[z <= q].mean()) < 0.03)

# ---------- dependence ----------
import dependence as dp
for fam, par in [("clayton", [2.0]), ("gumbel", [2.0]), ("frank", [5.0]), ("student_t", [0.6, 4.0]), ("gaussian", [0.6])]:
    uv = dp.simulate_copula(fam, par, 4000, seed=7)
    tau_sim = stats.kendalltau(uv[:, 0], uv[:, 1])[0]
    f = dp.fit_copula(uv[:, 0], uv[:, 1], fam)
    check(f"{fam}: sim tau ≈ implied, MLE recovers param", abs(tau_sim - dp._tau(fam, par)) < 0.03
          and abs(f["params"][0] - par[0]) / abs(par[0]) < 0.15, f"tau {tau_sim:.3f} vs {dp._tau(fam, par):.3f}; fit {f['params']}")
uv = dp.simulate_copula("clayton", [3.0], 3000, seed=1)
xx, yy = stats.t.ppf(uv[:, 0], 4), stats.norm.ppf(uv[:, 1])
tab = dp.fit_copulas(xx, yy)
check("fit_copulas picks a lower-tail family for Clayton data", tab.index[0] in ("clayton", "survival_gumbel"), tab.index[0])
# MI: nonlinear relationship invisible to Pearson
xn = rng.normal(size=2000); yn = xn ** 2 + rng.normal(scale=0.3, size=2000)
dt = dp.dependence_table(pd.DataFrame({"x": xn, "y": yn, "z": rng.normal(size=2000)}))
check("MI flags x~y nonlinear gap, z independent", dt.loc["x ~ y", "nonlinear_gap"] > 0.5 and dt.loc["x ~ z", "MI_linfoot"] < 0.2)
g = rng.multivariate_normal([0, 0], [[1, .7], [.7, 1]], 5000)
check("Linfoot(MI) ≈ |rho| for Gaussian", abs(dp.linfoot(dp._mi(g[:, 0], g[:, 1])) - 0.7) < 0.05)
M = dp.mutual_information_matrix(pd.DataFrame(g, columns=["a", "b"]))
check("MI matrix diagonal NaN, symmetric", np.isnan(M.loc["a", "a"]) and M.loc["a", "b"] == M.loc["b", "a"])

# ---------- state space ----------
import state_space as ss
n = 1500
beta = 1.0 + np.cumsum(rng.normal(0, 0.01, n))
xp = 50 + np.cumsum(rng.normal(0, 1, n))
yp = 2 + beta * xp + rng.normal(0, 1.0, n)
ysr, xsr = pd.Series(yp, index=idx[:n]), pd.Series(xp, index=idx[:n])
out, par = ss.fit_kalman_regression(ysr, xsr)
err = np.abs(out["b_filt"] - beta)[100:].mean()
check("Kalman tracks time-varying beta", err < 0.05, f"MAE {err:.4f}")
z = out["z"].dropna()
check("innovation z ~ N(0,1)", abs(z.mean()) < 0.1 and abs(z.std() - 1) < 0.1, f"{z.mean():.3f}, {z.std():.3f}")
filt_resid = (ysr - out["a_filt"] - out["b_filt"] * xsr).dropna()
check("filtered residual is shrunk vs innovation (the trap)", filt_resid.std() < out["e"].std())
s_ou = pd.Series(np.zeros(2000))
for t in range(1, 2000):
    s_ou[t] = 0.95 * s_ou[t - 1] + rng.normal()
hl = ss.half_life(s_ou)["half_life"]
check("half-life ≈ ln2/0.05 ≈ 13.5", 10 < hl < 18, f"{hl:.1f}")
check("Hurst < 0.5 for OU, ≈0.5 for RW", ss.hurst_exponent(s_ou) < 0.4 and 0.4 < ss.hurst_exponent(np.cumsum(rng.normal(size=3000))) < 0.6)
p = ss.pairs_position(out["z"])
pr = ss.pairs_returns(ysr, xsr, out["b_pred"], p)
check("pairs returns computed, no NaN", pr["strategy_return"].notna().all())
ct = ss.cointegration_pretest(ysr, xsr)

# ---------- regimes ----------
import regimes as rg
n = 2000
s = np.zeros(n, int)
for t in range(1, n):
    s[t] = s[t-1] if rng.random() < 0.98 else 1 - s[t-1]
ret = np.where(s == 0, rng.normal(0.0005, 0.007, n), rng.normal(-0.001, 0.025, n))
rs = pd.Series(ret, index=pd.bdate_range("2005-01-03", periods=n))
ms = rg.markov_switching(rs, 2)
acc = ((ms["filtered"][1] > 0.5).astype(int).to_numpy() == s[-len(ms["filtered"]):]).mean()
check("MS filtered regimes match truth", acc > 0.9, f"acc {acc:.3f}")
check("MS durations ≈ 50", 30 < ms["summary"]["expected_duration"].iloc[0] < 80)
prices = pd.Series(100 * np.exp(np.cumsum(ret)), index=rs.index)
F = rg.regime_features(prices)
km = rg.kmeans_regimes(F[["vol_20", "ret_20"]], 2, train_end=F.index[1200])
hm = rg.hmm_regimes(F[["ret_1", "vol_20"]], 2, train_end=F.index[1200])
truth = pd.Series(s, index=rs.index).loc[hm["labels"].index]
acc_h = (hm["labels"] == truth).mean()
check("HMM filtered labels match truth (out of sample too)", acc_h > 0.85, f"{acc_h:.3f}")
cmp = rg.compare_labels(km["labels"], hm["labels"])
print("ARI kmeans vs hmm:", round(cmp["ARI"], 3))
print(rg.regime_summary(rs, hm["labels"]))

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
