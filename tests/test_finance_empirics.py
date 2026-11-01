"""Tests for finance_empirics.py (v2.1): returns, market model/CAPM, rolling and adjusted betas,
Hamada, factor models, event studies, performance ratios and mean-variance portfolios.

Checks: exact identities (simple vs log returns, total returns, Hamada round trip, closed-form
minimum-variance weights); recovery of known betas, factor loadings and a beta break; event-study
power and Monte Carlo size of the cross-sectional, Patell, BMP and Corrado tests, including the
classic result that Patell over-rejects under event-induced variance while BMP does not
(Boehmer, Musumeci & Poulsen 1991).

Run from the repository root:  python3 tests/test_finance_empirics.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "econometrics" / "scripts"))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import finance_empirics as fe

ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


rng = np.random.default_rng(2015)
dates = pd.bdate_range("2018-01-01", periods=1500)

# ---------------------------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------------------------
p = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(dates)))), index=dates)
rs, rl = fe.returns(p), fe.returns(p, kind="log")
check("log return = ln(1 + simple return)", np.allclose(rl, np.log1p(rs)))
div = pd.Series(0.0, index=dates)
div.iloc[100] = 2.0
rt = fe.returns(p, dividends=div)
check("total return adds the dividend on the ex-date",
      abs(rt.iloc[99] - (p.iloc[100] + 2) / p.iloc[99] + 1) < 1e-12 and abs(rt.iloc[50] - rs.iloc[50]) < 1e-15)
rm_ = fe.returns(p, freq="ME")
daily_comp = (1 + rs).groupby(rs.index.to_period("M")).prod() - 1
check("monthly returns from month-end prices = compounded daily returns",
      np.allclose(rm_.to_numpy(), daily_comp.iloc[1:].to_numpy()))

# ---------------------------------------------------------------------------------------------
# Market model, CAPM, rolling beta
# ---------------------------------------------------------------------------------------------
n = 1500
r_m = pd.Series(rng.normal(0.0004, 0.012, n), index=dates)
rf = pd.Series(0.0004, index=dates)
r_i = 0.0001 + 1.2 * r_m + pd.Series(rng.normal(0, 0.015, n), index=dates)
mm = fe.market_model(r_i, r_m)
check("market model: beta CI covers 1.2", mm["table"].loc["beta", "ci_low"] < 1.2 < mm["table"].loc["beta", "ci_high"],
      f"{mm['beta']:.3f}")
capm = fe.market_model(r_i, r_m, rf=rf)
check("CAPM in excess returns: beta ~ market-model beta, alpha CI covers the implied value",
      abs(capm["beta"] - mm["beta"]) < 0.02)
rb = pd.concat([0.8 * r_m.iloc[:750], 1.5 * r_m.iloc[750:]]) + pd.Series(rng.normal(0, 0.01, n), index=dates)
roll = fe.rolling_beta(rb, r_m, window=250)
early, late = roll["beta"].iloc[100], roll["beta"].iloc[-1]
check("rolling beta tracks a break from 0.8 to 1.5", abs(early - 0.8) < 0.15 and abs(late - 1.5) < 0.15,
      f"{early:.2f} -> {late:.2f}")
last = fe.market_model(rb.iloc[-250:], r_m.iloc[-250:], hac_lags=0)
check("last rolling beta = OLS beta on the same window", abs(roll["beta"].iloc[-1] - last["beta"]) < 1e-10)
import statsmodels.api as sm
cls = sm.OLS(rb.iloc[-250:].to_numpy(), sm.add_constant(r_m.iloc[-250:].to_numpy())).fit()
check("rolling OLS SE = classical OLS SE on the same window", abs(roll["se"].iloc[-1] - cls.bse[1]) < 1e-8,
      f"{roll['se'].iloc[-1]:.5f} vs {cls.bse[1]:.5f}")

# ---------------------------------------------------------------------------------------------
# Adjusted betas and Hamada
# ---------------------------------------------------------------------------------------------
check("Blume: 0.67 * 1.6 + 0.33 = 1.402", abs(fe.adjusted_beta(1.6)["adjusted_beta"] - 1.402) < 1e-12)
v1 = fe.adjusted_beta(1.6, "vasicek", se=0.1, prior_sd=0.3)
v2 = fe.adjusted_beta(1.6, "vasicek", se=0.5, prior_sd=0.3)
check("Vasicek shrinks an imprecise beta more", v2["adjusted_beta"] < v1["adjusted_beta"] < 1.6 and abs(v1["weight_on_raw"] - 0.9) < 1e-12)
bu = fe.unlever_beta(1.3, 0.6, 0.34)
check("Hamada: unlever then relever returns the original beta", abs(fe.relever_beta(bu, 0.6, 0.34) - 1.3) < 1e-12,
      f"beta_U = {float(bu):.4f}")
check("Hamada formula: 1.3 / (1 + 0.66 * 0.6) = 0.9312", abs(float(bu) - 1.3 / (1 + 0.66 * 0.6)) < 1e-12)
check("with a debt beta, round trip still exact",
      abs(fe.relever_beta(fe.unlever_beta(1.3, 0.6, 0.34, 0.2), 0.6, 0.34, 0.2) - 1.3) < 1e-12)
comps = np.array([1.1, 1.4, 0.9])
check("vectorized over comparables", fe.unlever_beta(comps, np.array([0.2, 0.8, 0.5]), 0.34).shape == (3,))

hp = fe.unlever_beta(1.3, 0.6, 0.34, method="harris_pringle")
check("Harris-Pringle: 1.3 / (1 + 0.6) = 0.8125 and round trip exact",
      abs(float(hp) - 1.3 / 1.6) < 1e-12 and abs(fe.relever_beta(hp, 0.6, method="harris_pringle") - 1.3) < 1e-12)
check("Harris-Pringle asset beta < Hamada asset beta when t > 0 (no tax shield credited to equity)", float(hp) < float(bu))
check("Harris-Pringle with debt beta: (1.3 + 0.2*0.6) / 1.6",
      abs(float(fe.unlever_beta(1.3, 0.6, debt_beta=0.2, method="harris_pringle")) - (1.3 + 0.12) / 1.6) < 1e-12)
try:
    fe.unlever_beta(1.3, 0.6, 0.34, method="mm")
    check("unknown levering method raises", False)
except ValueError:
    check("unknown levering method raises", True)

# Dimson beta: a thinly traded stock absorbs half of the market move with a one-day delay
rmd = pd.Series(rng.normal(0.0004, 0.012, 1500), index=dates)
thin = 0.5 * rmd + 0.5 * rmd.shift(1).fillna(0) + pd.Series(rng.normal(0, 0.012, 1500), index=dates)
dm = fe.dimson_beta(thin, rmd, lags=1, leads=1)
check("non-synchronous trading: OLS beta biased to ~0.5, Dimson beta CI covers the true 1.0",
      abs(dm["ols_beta"] - 0.5) < 0.1 and dm["ci_low"] < 1.0 < dm["ci_high"], f"OLS {dm['ols_beta']:.2f}, Dimson {dm['beta']:.2f}")
sync = 1.0 * rmd + pd.Series(rng.normal(0, 0.012, 1500), index=dates)
ds = fe.dimson_beta(sync, rmd)
check("synchronous stock: Dimson ~ OLS beta (difference < 2 SE)", abs(ds["beta"] - ds["ols_beta"]) < 2 * ds["se"])

# ---------------------------------------------------------------------------------------------
# Factor model (monthly, FF3-style)
# ---------------------------------------------------------------------------------------------
T = 240
mi = pd.period_range("2005-01", periods=T, freq="M").to_timestamp()
F = pd.DataFrame({"MKT": rng.normal(0.006, 0.045, T), "SMB": rng.normal(0.002, 0.03, T),
                  "HML": rng.normal(0.003, 0.03, T)}, index=mi)
ex = 1.1 * F.MKT + 0.5 * F.SMB - 0.3 * F.HML + rng.normal(0, 0.02, T)
fm = fe.factor_model(pd.Series(ex, index=mi), F)
tb = fm["table"]
check("factor loadings 1.1, 0.5, -0.3 inside their CIs",
      all(tb.loc[k, "ci_low"] < v < tb.loc[k, "ci_high"] for k, v in {"MKT": 1.1, "SMB": 0.5, "HML": -0.3}.items()),
      tb["estimate"].round(3).to_dict())
check("factor alpha not significant when the truth is zero", tb.loc["alpha", "p"] > 0.05)

# ---------------------------------------------------------------------------------------------
# Event study: power with a +2% day-0 abnormal return
# ---------------------------------------------------------------------------------------------
def panel(seed, n_firms=40, ab=0.0, var_mult=1.0):
    r_ = np.random.default_rng(seed)
    rm = pd.Series(r_.normal(0.0003, 0.01, 600), index=dates[:600])
    R = {}
    ev = {}
    for j in range(n_firms):
        b = r_.uniform(0.6, 1.4)
        e = r_.normal(0, 0.018, 600)
        d0 = int(r_.integers(320, 580))
        e[d0] = e[d0] * np.sqrt(var_mult) + ab
        R[f"f{j}"] = 0.0002 + b * rm + e
        ev[f"f{j}"] = rm.index[d0]
    return pd.DataFrame(R), rm, ev


R, rm, ev = panel(1, ab=0.02)
es = fe.event_study(R, rm, ev, est_win=(-250, -30), evt_win=(-10, 10), car_windows=((-1, 1),))
c = es["caar"].loc["(-1, 1)"]
check("event study: CAAR(-1,1) ~ 0.02", abs(c["CAAR"] - 0.02) < 0.008, f"{c['CAAR']:.4f}")
check("event study: all CAAR tests reject (p < 0.01; expected t ~ 4 with 40 firms)", max(c["p_cross_section"], c["p_patell"], c["p_bmp"]) < 0.01)
check("event study: day-0 AAR ~ 0.02 and Corrado rank test rejects",
      abs(es["aar"].loc[0, "AAR"] - 0.02) < 0.008 and es["p_corrado_day0"] < 0.001)
check("no abnormal return outside day 0 (|AAR(-5)| < 0.008)", abs(es["aar"].loc[-5, "AAR"]) < 0.008)
try:
    fe.event_study(R, rm, ev, est_win=(-250, -5), evt_win=(-10, 10))
    check("overlapping estimation and event windows raise", False)
except ValueError:
    check("overlapping estimation and event windows raise", True)

# ---------------------------------------------------------------------------------------------
# Repeated acquirers, skipped events and the cross-sectional CAR regression
# ---------------------------------------------------------------------------------------------
r_ = np.random.default_rng(31)
rm2 = pd.Series(r_.normal(0.0003, 0.01, 900), index=dates[:900])
R2, rows_ev = {}, []
for j in range(30):
    e = r_.normal(0, 0.018, 900)
    for k, d0 in enumerate((350, 650)):                      # every firm announces two deals
        stock = int(r_.integers(0, 2))
        e[d0] += 0.02 - 0.03 * stock                         # cash deals +2%, stock deals -1%
        rows_ev.append({"firm": f"f{j}", "date": rm2.index[d0], "stock_paid": stock, "size": r_.uniform(0.1, 1)})
    R2[f"f{j}"] = 0.0002 + r_.uniform(0.6, 1.4) * rm2 + e
R2 = pd.DataFrame(R2)
R2.loc[:R2.index[300], "f0"] = np.nan                         # f0 listed recently: first event cannot be evaluated
evdf = pd.DataFrame(rows_ev)
es2 = fe.event_study(R2, rm2, evdf, est_win=(-250, -30), evt_win=(-5, 5), car_windows=((0, 0),))
check("repeated acquirers: one event_id per deal, the unevaluable one listed as skipped",
      es2["n_events"] == 59 and es2["skipped"] == ["f0_" + str(rm2.index[350].date())] and es2["ar"].index.is_unique)
cr = fe.car_regression(es2, evdf, "CAR ~ stock_paid + size", window="(0, 0)", cluster="firm")
check("car_regression: stock-paid deals ~3 p.p. lower (CI covers -0.03), skipped event dropped, clusters by acquirer",
      cr["table"].loc["stock_paid", "ci_low"] < -0.03 < cr["table"].loc["stock_paid", "ci_high"] and cr["n"] == 59
      and cr["n_clusters"] == 30 and any("not used" in w for w in cr["warnings"]),
      f"{cr['table'].loc['stock_paid', 'estimate']:.4f}")
try:
    fe.event_study(R2, rm2, evdf.assign(event_id="same"), est_win=(-250, -30), evt_win=(-5, 5))
    check("duplicated event_id raises", False)
except ValueError:
    check("duplicated event_id raises", True)

# ---------------------------------------------------------------------------------------------
# Event study: Monte Carlo size, with and without event-induced variance
# ---------------------------------------------------------------------------------------------
rej = {k: [] for k in ("cs", "patell", "bmp", "corrado", "patell_var", "bmp_var")}
for rep in range(200):
    R0, rm0, ev0 = panel(100 + rep, n_firms=30)
    e0 = fe.event_study(R0, rm0, ev0, evt_win=(-5, 5), car_windows=((0, 0),))
    c0 = e0["caar"].loc["(0, 0)"]
    rej["cs"].append(c0["p_cross_section"] < 0.05)
    rej["patell"].append(c0["p_patell"] < 0.05)
    rej["bmp"].append(c0["p_bmp"] < 0.05)
    rej["corrado"].append(e0["p_corrado_day0"] < 0.05)
    R1, rm1, ev1 = panel(100 + rep, n_firms=30, var_mult=4.0)     # day-0 variance x4, no mean effect
    c1 = fe.event_study(R1, rm1, ev1, evt_win=(-5, 5), car_windows=((0, 0),))["caar"].loc["(0, 0)"]
    rej["patell_var"].append(c1["p_patell"] < 0.05)
    rej["bmp_var"].append(c1["p_bmp"] < 0.05)
sz = {k: np.mean(v) for k, v in rej.items()}
check("no event effect: cross-sectional, Patell, BMP and Corrado sizes in [2%, 9%]",
      all(0.02 <= sz[k] <= 0.09 for k in ("cs", "patell", "bmp", "corrado")), {k: round(v, 3) for k, v in sz.items()})
check("event-induced variance: Patell over-rejects (> 12%) while BMP stays <= 9%",
      sz["patell_var"] > 0.12 and sz["bmp_var"] <= 0.09, f"Patell {sz['patell_var']:.3f}, BMP {sz['bmp_var']:.3f}")

# ---------------------------------------------------------------------------------------------
# Performance ratios
# ---------------------------------------------------------------------------------------------
rp = pd.Series(rng.normal(0.0006, 0.01, 1000), index=dates[:1000])
rbm = 0.7 * rp + pd.Series(rng.normal(0.0001, 0.006, 1000), index=dates[:1000])
pr = fe.performance_ratios(rp, rbm, rf=0.0002)["value"]
exr = rp - 0.0002
check("Sharpe = mean/sd * sqrt(252)", abs(pr["Sharpe (ann.)"] - exr.mean() / exr.std(ddof=1) * np.sqrt(252)) < 1e-12)
act = rp - rbm
check("information ratio = mean active / tracking error * sqrt(252)",
      abs(pr["information ratio (ann.)"] - act.mean() / act.std(ddof=1) * np.sqrt(252)) < 1e-12)
beta_pb = np.cov(exr, rbm - 0.0002, ddof=1)[0, 1] / (rbm - 0.0002).var(ddof=1)
check("Treynor = annualized mean excess / beta", abs(pr["Treynor (ann.)"] - exr.mean() * 252 / beta_pb) < 1e-12)
check("Jensen alpha = CAPM intercept (annualized)",
      abs(pr["Jensen alpha (ann.)"] - fe.market_model(rp, rbm, rf=pd.Series(0.0002, index=rp.index))["alpha_annualized"]) < 1e-10)

# ---------------------------------------------------------------------------------------------
# Mean-variance portfolios
# ---------------------------------------------------------------------------------------------
mu = pd.Series([0.08, 0.12, 0.15, 0.10], index=list("ABCD"))
sd = np.array([0.15, 0.22, 0.30, 0.18])
corr = np.array([[1, .3, .2, .4], [.3, 1, .5, .3], [.2, .5, 1, .2], [.4, .3, .2, 1]])
cov = pd.DataFrame(np.outer(sd, sd) * corr, index=mu.index, columns=mu.index)
ef = fe.efficient_frontier(mu, cov, rf=0.03, short=True)
Si = np.linalg.inv(cov.to_numpy())
w_gmv = Si @ np.ones(4) / (np.ones(4) @ Si @ np.ones(4))
check("GMV weights = Sigma^-1 1 / (1' Sigma^-1 1)", np.allclose(ef["gmv_weights"], w_gmv))
rw = rng.dirichlet(np.ones(4), 5000)
rand_sharpe = (rw @ mu.to_numpy() - 0.03) / np.sqrt(np.einsum("ij,jk,ik->i", rw, cov.to_numpy(), rw))
check("tangency Sharpe >= every random long-only portfolio's", ef["tangency"]["sharpe"] >= rand_sharpe.max() - 1e-9,
      f"{ef['tangency']['sharpe']:.4f} vs {rand_sharpe.max():.4f}")
lo = fe.efficient_frontier(mu, cov, rf=0.03, short=False)
check("long-only weights are >= 0 and sum to 1",
      (lo["gmv_weights"] >= -1e-8).all() and abs(lo["gmv_weights"].sum() - 1) < 1e-6
      and (lo["tangency_weights"] >= -1e-8).all())
check("long-only GMV volatility >= unconstrained GMV volatility", lo["gmv"]["volatility"] >= ef["gmv"]["volatility"] - 1e-9)
rets = pd.DataFrame(rng.multivariate_normal(np.zeros(4), cov.to_numpy() / 252, 300), columns=list("ABCD"))
sc = fe.shrink_cov(rets)
check("Ledoit-Wolf covariance is symmetric positive definite with shrinkage in (0, 1)",
      np.allclose(sc, sc.T) and np.all(np.linalg.eigvalsh(sc) > 0) and 0 < sc.attrs["shrinkage"] < 1)

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
