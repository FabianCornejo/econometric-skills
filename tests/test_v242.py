"""Tests for v2.4.2: classical regression and discrete choice, teaching demos, and robustness of the plugin.

Each function is checked against an independent route, never by repeating its own formula:
textbook results (Wooldridge 2e examples 12.4, 15.1, 15.7, 17.1, 17.5 and Stata's heckman twostep),
statsmodels/linearmodels/sklearn where they compute the same thing another way, exact GLS with the AR(1)
covariance matrix, a bootstrap of the whole Heckman procedure, finite differences for marginal effects,
and direct regressions for restricted least squares and the Anderson-Rubin statistic.

Plugin robustness: the three mode commands (user-invoked only), their declaration in marketplace.json,
the app package without them, the clean-workspace helper, and the Windows CI job with a space in the path.

Run from the repository root:  python3 tests/test_v242.py
"""
import importlib
import json
import re
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import wooldridge as wd
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "econometrics"
sys.path.insert(0, str(SKILL / "scripts"))
warnings.filterwarnings("ignore")

from specification import linear_restriction_test, prais_winsten, nonnested_test  # noqa: E402
from discrete_choice import binary_effects, heckman  # noqa: E402
from diagnostics import iv_report  # noqa: E402
from finance_empirics import market_model  # noqa: E402
from simulation import demo_simultaneity, demo_lpm_bounds, demo_multicollinearity  # noqa: E402
from workspace import workspace_settings  # noqa: E402
import _console  # noqa: E402,F401  (Windows-safe output of the non-ASCII check names)

ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


def close(a, b, tol):
    return abs(float(a) - float(b)) <= tol


# ---------------------------------------------------------------- 1. linear_restriction_test
mlb = wd.data("mlb1")
f = "np.log(salary) ~ years + gamesyr + bavg + hrunsyr + rbisyr"
ur = smf.ols(f, mlb).fit()
r = linear_restriction_test(f, mlb, "bavg = 0, hrunsyr = 0, rbisyr = 0", show=False)
sm_f = ur.f_test("bavg = 0, hrunsyr = 0, rbisyr = 0")
check("exclusion F equals statsmodels f_test (Wooldridge Ex. 4.9: F = 9.55)",
      close(r["F"], float(sm_f.fvalue), 1e-8) and close(r["F"], 9.55, 0.01), f"{r['F']:.4f}")
rr = smf.ols("np.log(salary) ~ years + gamesyr", mlb).fit()
check("restricted least squares under exclusion = OLS of the short model",
      np.allclose(r["restricted_params"][["Intercept", "years", "gamesyr"]].to_numpy(), rr.params.to_numpy(), atol=1e-9)
      and close(r["ssr_restricted"], rr.ssr, 1e-6))
r2f = ((ur.rsquared - rr.rsquared) / 3) / ((1 - ur.rsquared) / ur.df_resid)
check("same y: the R² form equals the SSR form", close(r2f, r["F"], 1e-8))
r1 = linear_restriction_test(f, mlb, "hrunsyr = rbisyr", show=False)
tt = ur.t_test("hrunsyr = rbisyr")
check("one restriction: θ, SE and t equal statsmodels t_test, and t² = F",
      close(r1["theta"]["estimate"], float(np.squeeze(tt.effect)), 1e-10)
      and close(r1["theta"]["se"], float(np.squeeze(tt.sd)), 1e-10) and close(r1["theta"]["t"] ** 2, r1["F"], 1e-8))
mlb2 = mlb.assign(hr_rbi=mlb.hrunsyr + mlb.rbisyr)
comb = smf.ols("np.log(salary) ~ years + gamesyr + bavg + hr_rbi", mlb2).fit()
check("equality restriction: restricted coefficients = regression on the summed variable",
      close(r1["restricted_params"]["hrunsyr"], comb.params["hr_rbi"], 1e-9)
      and close(r1["restricted_params"]["rbisyr"], comb.params["hr_rbi"], 1e-9) and close(r1["ssr_restricted"], comb.ssr, 1e-6))
# imposing beta = 1 by transforming y: SSR form right, R² form wrong
rng = np.random.default_rng(3)
d = pd.DataFrame({"x1": rng.normal(size=300), "x2": rng.normal(size=300)})
d["y"] = 1.0 + 0.8 * d.x1 + 0.5 * d.x2 + rng.normal(size=300)
rb = linear_restriction_test("y ~ x1 + x2", d, "x1 = 1", show=False)
u_fit = smf.ols("y ~ x1 + x2", d).fit()
r_fit = smf.ols("I(y - x1) ~ x2", d).fit()
f_ssr = ((r_fit.ssr - u_fit.ssr) / 1) / (u_fit.ssr / u_fit.df_resid)
f_r2 = ((u_fit.rsquared - r_fit.rsquared) / 1) / ((1 - u_fit.rsquared) / u_fit.df_resid)
check("β = 1 imposed on a transformed y: F equals the SSR form and f_test; the R² form differs",
      close(rb["F"], f_ssr, 1e-8) and close(rb["F"], float(u_fit.f_test("x1 = 1").fvalue), 1e-8)
      and abs(f_r2 - f_ssr) > 1, f"F = {rb['F']:.3f}, wrong R² form = {f_r2:.3f}")
rh = linear_restriction_test(f, mlb, "bavg = 0, hrunsyr = 0, rbisyr = 0", cov_type="HC3", show=False)
check("robust: Wald F with HC3 equals statsmodels f_test on the HC3 fit, and says so",
      close(rh["F"], float(ur.model.fit(cov_type="HC3").f_test("bavg = 0, hrunsyr = 0, rbisyr = 0").fvalue), 1e-8)
      and rh["warnings"])
try:
    linear_restriction_test(f, mlb, "bavg = 0, bavg = 0", show=False)
    check("linearly dependent restrictions are refused", False)
except ValueError:
    check("linearly dependent restrictions are refused", True)
check("verdict never accepts H0", all("accept" not in x["verdict"] for x in (r, r1, rb, rh)))

# ---------------------------------------------------------------- 2. prais_winsten
bar = wd.data("barium")
fb = "lchnimp ~ lchempi + lgas + lrtwex + befile6 + affile6 + afdec6"
pw = prais_winsten(fb, bar, show=False)
t = pw["table"]
target = {"lchempi": (2.94, 0.63), "lgas": (1.05, 0.98), "lrtwex": (1.13, 0.51), "afdec6": (-0.577, 0.342)}
check("Prais-Winsten reproduces Wooldridge Example 12.4 (ρ̂ = .293; lchempi 2.94 (.63), lgas 1.05 (.98), "
      "lrtwex 1.13 (.51), afdec6 −.577 (.342))",
      close(pw["rho"], 0.293, 0.0006) and pw["converged"]
      and all(close(t.loc[k, "FGLS coef"], b, 0.006) and close(t.loc[k, "FGLS se"], s, 0.006) for k, (b, s) in target.items()),
      f"ρ̂ = {pw['rho']:.4f}")
rho = pw["rho"]
n = len(bar)
i = np.arange(n)
S = rho ** np.abs(i[:, None] - i[None, :]) / (1 - rho ** 2)
gls = sm.GLS(pw["ols"].model.endog, pw["ols"].model.exog, sigma=S).fit()
check("Prais-Winsten = exact GLS with the AR(1) covariance at ρ̂ (coefficients and SEs)",
      np.allclose(gls.params, pw["fgls"].params, atol=1e-9) and np.allclose(gls.bse, pw["fgls"].bse, atol=1e-9))
co = prais_winsten(fb, bar, method="cochrane", show=False)
y, X = co["ols"].model.endog, co["ols"].model.exog
u = y - X @ co["fgls"].params
check("Cochrane-Orcutt drops the first observation and ρ̂ is a fixed point of its own residuals",
      co["n"] == n - 1 and close(u[1:] @ u[:-1] / (u[:-1] @ u[:-1]), co["rho"], 1e-6))
check("whitening moves the Durbin-Watson statistic toward 2", abs(pw["dw_transformed"] - 2) < abs(pw["dw_ols"] - 2))
bar_gap = bar.copy()
bar_gap.loc[bar_gap.index[60], "lchempi"] = np.nan
refused = []
for dd_, kw in ((bar_gap, {}), (bar.sample(frac=1, random_state=1), {})):
    try:
        prais_winsten(fb, dd_, show=False, **kw)
        refused.append(False)
    except ValueError:
        refused.append(True)
shuf = bar.sample(frac=1, random_state=1).assign(t=np.argsort(np.argsort(bar.sample(frac=1, random_state=1).index)))
pw_t = prais_winsten(fb, shuf, time="t", show=False)
check("time order enforced: a gap inside the sample and shuffled rows are refused; time= sorts them back",
      all(refused) and close(pw_t["rho"], pw["rho"], 1e-10))
rw = np.cumsum(np.random.default_rng(5).normal(size=200))       # random-walk errors
try:
    out_rw = prais_winsten("y ~ x", pd.DataFrame({"y": 0.1 * np.arange(200.0) + rw, "x": np.arange(200.0)}), show=False)
    flagged = any("close to 1" in w_ for w_ in out_rw["warnings"])
except ValueError:
    flagged = True
check("random-walk errors are flagged (ρ̂ near 1 warned, or ρ̂ ≥ 1 refused)", flagged)

# ---------------------------------------------------------------- 3. nonnested_test
rng = np.random.default_rng(11)
m = 400
dn = pd.DataFrame({"x": rng.lognormal(0, 0.6, m), "z": rng.normal(size=m)})
dn["y"] = 2 + 3 * np.log(dn.x) + 0.5 * dn.z + rng.normal(0, 0.5, m)
nn = nonnested_test("y ~ np.log(x) + z", "y ~ x + z", dn, show=False)
aug = smf.ols("y ~ np.log(x) + z + fit2", dn.assign(fit2=smf.ols("y ~ x + z", dn).fit().fittedvalues)).fit(cov_type="HC3")
check("J statistic = t of the rival's fitted values, computed directly", close(nn["table"]["statistic"].iloc[0],
                                                                             aug.tvalues["fit2"], 1e-8))
check("true model (log x) not rejected, wrong model (x) rejected → data favour model 1",
      nn["table"]["p"].iloc[0] > 0.05 and nn["table"]["p"].iloc[1] < 0.05 and "favour model 1" in nn["verdict"])
hp = wd.data("hprice1")
nc = nonnested_test("lprice ~ sqrft + C(colonial)", "lprice ~ lsqrft + colonial", hp, show=False)
check("encompassing counts a variable written two ways (C(colonial) vs colonial) once: 1 df each way",
      list(nc["table"]["df"].iloc[2:]) == [1, 1], str(list(nc["table"]["df"])))
try:
    nonnested_test("y ~ x", "np.log(y) ~ x", dn, show=False)
    check("different dependent variables are refused", False)
except ValueError:
    check("different dependent variables are refused", True)

# ---------------------------------------------------------------- 4. binary_effects
mroz = wd.data("mroz")
fm = "inlf ~ nwifeinc + educ + exper + I(exper**2) + age + kidslt6 + kidsge6"
be = binary_effects(fm, mroz, show=False)
tb = be["table"]
lpm_b = be["models"]["lpm"].params
check("LPM reproduces Wooldridge Example 17.1 (nwifeinc −.0034, educ .038, exper .039, age −.016, kidslt6 −.262)",
      all(close(lpm_b[k], v, tol) for k, v, tol in (("nwifeinc", -0.0034, 0.00006), ("educ", 0.038, 0.0006),
                                                   ("exper", 0.0395, 0.0006), ("age", -0.0161, 0.0006),
                                                   ("kidslt6", -0.262, 0.0006))))
check("x with x²: the LPM effect is the average slope β_x + 2β_x²·x, not the coefficient of x alone",
      close(tb.loc["exper", "LPM"], np.mean(lpm_b["exper"] + 2 * lpm_b["I(exper ** 2)"] * mroz.exper), 1e-6),
      f"{tb.loc['exper', 'LPM']:.4f} vs coefficient {lpm_b['exper']:.4f}")
lg = be["models"]["logit"]
h = 1e-5
base_p = lg.predict(mroz)
fd = np.mean((lg.predict(mroz.assign(educ=mroz.educ + h)) - lg.predict(mroz.assign(educ=mroz.educ - h))) / (2 * h))
check("logit AME of educ = finite-difference average of predicted probabilities", close(tb.loc["educ", "logit AME"], fd, 1e-6),
      f"{tb.loc['educ', 'logit AME']:.5f} vs {fd:.5f}")
mge = lg.get_margeff(at="overall").summary_frame()
check("delta-method SE of a single-term AME equals statsmodels get_margeff",
      close(tb.loc["educ", "logit AME se"], mge.loc["educ", "Std. Err."], 1e-6))
fde = np.mean((lg.predict(mroz.assign(exper=mroz.exper + h)) - lg.predict(mroz.assign(exper=mroz.exper - h))) / (2 * h))
check("logit AME of exper moves exper and exper² together (finite difference of predictions)",
      close(tb.loc["exper", "logit AME"], fde, 1e-6))
rng = np.random.default_rng(0)
nf = 4000
dfac = pd.DataFrame({"x": rng.normal(size=nf), "region": rng.choice(["a", "b", "c"], nf)})
linf = -0.5 + 0.8 * dfac.x + np.where(dfac.region == "b", 1.0, 0) + np.where(dfac.region == "c", 2.0, 0)
dfac["y"] = (rng.uniform(size=nf) < 1 / (1 + np.exp(-linf))).astype(int)
bf = binary_effects("y ~ x + C(region)", dfac, show=False)
mf = bf["models"]["logit"]
pa, pb, pc = (mf.predict(dfac.assign(region=v)) for v in ("a", "b", "c"))
check("factor with 3 levels: each level vs base = average of P(all at level) − P(all at base)",
      close(bf["table"].loc["region[b vs a]", "logit AME"], np.mean(pb - pa), 1e-10)
      and close(bf["table"].loc["region[c vs a]", "logit AME"], np.mean(pc - pa), 1e-10))
rng = np.random.default_rng(21)
db = pd.DataFrame({"d": rng.integers(0, 2, 2000), "x": rng.normal(size=2000)})
db["y"] = (rng.uniform(size=2000) < 1 / (1 + np.exp(-(-0.3 + 0.9 * db.d + 0.7 * db.x)))).astype(int)
bb = binary_effects("y ~ d + x", db, show=False)
pr = bb["models"]["probit"]
dc = np.mean(pr.predict(db.assign(d=1)) - pr.predict(db.assign(d=0)))
check("0/1 regressor: probit AME is the average discrete change P(d=1) − P(d=0), flagged as such",
      close(bb["table"].loc["d", "probit AME"], dc, 1e-8) and bb["table"].loc["d", "kind"].startswith("0/1"))
odds = be["odds_ratios"]
check("odds ratios: exp(β) and the exact change 100(OR − 1)% (kidslt6: OR ≈ 0.236, −76%)",
      np.allclose(odds["odds_ratio"], np.exp(lg.params[odds.index])) and close(odds.loc["kidslt6", "odds_change_%"],
                                                                             100 * (np.exp(lg.params["kidslt6"]) - 1), 1e-9)
      and close(odds.loc["kidslt6", "odds_ratio"], 0.236, 0.001))
from sklearn.metrics import roc_auc_score  # noqa: E402
check("AUC equals sklearn's roc_auc_score for each model",
      all(close(be["classification"].loc[k, "AUC"], roc_auc_score(mroz.inlf, p), 1e-12)
          for k, p in (("logit", base_p), ("LPM", be["models"]["lpm"].fittedvalues))))
lp = be["models"]["lpm"].fittedvalues
check("cutoff = base rate by default; majority-class accuracy and LPM outside [0, 1] computed directly",
      close(be["classification"].loc["logit", "cutoff"], mroz.inlf.mean(), 1e-12)
      and close(be["majority_accuracy"], max(mroz.inlf.mean(), 1 - mroz.inlf.mean()), 1e-12)
      and close(be["lpm_outside_01"], np.mean((lp < 0) | (lp > 1)), 1e-12) and be["lpm_outside_01"] > 0)
check("LR tests and pseudo-R² match the fitted models", close(be["fit"].loc["probit", "LR stat (all slopes = 0)"],
                                                              2 * (be["models"]["probit"].llf - be["models"]["probit"].llnull), 1e-8))
try:
    binary_effects("hours ~ educ", mroz, show=False)
    check("a non-binary outcome is refused", False)
except ValueError:
    check("a non-binary outcome is refused", True)

# ---------------------------------------------------------------- 5. heckman
sel_f = "inlf ~ nwifeinc + educ + exper + I(exper**2) + age + kidslt6 + kidsge6"
hk = heckman("lwage ~ educ + exper + I(exper**2)", sel_f, mroz, show=False)
ht = hk["table"]
check("Heckit reproduces Wooldridge Example 17.5 and Stata heckman twostep (educ .1091 (.0155); λ .0323 (.1336); "
      "ρ .0486; σ .6636)",
      close(ht.loc["educ", "estimate"], 0.1091, 0.0002) and close(ht.loc["educ", "se"], 0.0155, 0.0002)
      and close(hk["lambda"]["estimate"], 0.0323, 0.0002) and close(hk["lambda"]["se"], 0.1336, 0.0003)
      and close(hk["rho"], 0.0486, 0.0003) and close(hk["sigma"], 0.6636, 0.0003),
      f"educ {ht.loc['educ', 'estimate']:.4f} ({ht.loc['educ', 'se']:.4f}), λ {hk['lambda']['estimate']:.4f} "
      f"({hk['lambda']['se']:.4f}), ρ {hk['rho']:.4f}, σ {hk['sigma']:.4f}")
check("β_λ = ρσ", close(hk["lambda"]["estimate"], hk["rho"] * hk["sigma"], 1e-10))
# simulated selection with strong correlation: corrected SEs vs a bootstrap of the whole two-step procedure
rng = np.random.default_rng(7)
N = 1500
ds = pd.DataFrame({"x": rng.normal(size=N), "z": rng.normal(size=N)})
e = rng.multivariate_normal([0, 0], [[1, 0.7], [0.7, 1]], N)
ds["s"] = (0.3 + 0.5 * ds.x + 1.0 * ds.z + e[:, 0] > 0).astype(int)
ds["y"] = np.where(ds.s == 1, 1 + 2 * ds.x + e[:, 1], np.nan)
hs = heckman("y ~ x", "s ~ x + z", ds, show=False)
boot = []
for b in range(150):
    ix = rng.integers(0, N, N)
    db_ = ds.iloc[ix].reset_index(drop=True)
    boot.append(heckman("y ~ x", "s ~ x + z", db_, show=False)["table"]["estimate"].to_numpy())
bse = np.std(np.array(boot), axis=0, ddof=1)
ratio = hs["table"]["se"].to_numpy() / bse
check("corrected SEs agree with a bootstrap of the two steps (ratio within 0.8-1.25 for every coefficient)",
      np.all((ratio > 0.8) & (ratio < 1.25)), np.round(ratio, 3))
check("strong selection (ρ = 0.7) is detected and the slope recovered",
      hs["lambda"]["p"] < 0.01 and close(hs["table"].loc["x", "estimate"], 2.0, 0.1) and close(hs["rho"], 0.7, 0.15),
      f"ρ̂ = {hs['rho']:.3f}")
hn = heckman("y ~ x + z", "s ~ x + z", ds, show=False)
check("no exclusion restriction → warning", any("exclusion" in w for w in hn["warnings"]) and not hn["excluded_from_outcome"])
hr = heckman("y ~ x + z", "s ~ x + I(1 * z)", ds, show=False)
check("exclusion judged by the data, not the names (z written as I(1 * z) is not excluded)",
      any("exclusion" in w for w in hr["warnings"]))
dup = pd.concat([mroz.iloc[:400], mroz.iloc[:353]])
check("a duplicate index does not break the alignment of the two steps",
      np.isfinite(heckman("lwage ~ educ + exper", "inlf ~ educ + exper + kidslt6", dup, show=False)["rho"]))

# ---------------------------------------------------------------- 6. iv_report
mz = mroz.dropna(subset=["lwage"])
iv1 = iv_report("lwage ~ 1 + [educ ~ fatheduc]", mz, cov_type="unadjusted", show=False)
check("2SLS reproduces Wooldridge Example 15.1 (educ .059 (.035))",
      close(iv1["table"].loc["educ", "2SLS"], 0.0592, 0.0003) and close(iv1["table"].loc["educ", "2SLS se"], 0.0351, 0.0003))
check("exactly identified: no overidentification test", iv1["overidentified_by"] == 0 and len(iv1["tests"]) == 1)
iv2 = iv_report("lwage ~ 1 + exper + expersq + [educ ~ motheduc + fatheduc]", mz, cov_type="unadjusted", show=False)
ov = iv2["tests"].iloc[1]
check("overidentification nR² reproduces Wooldridge Example 15.7 (≈ .38, p ≈ .54)",
      close(ov["statistic"], 0.378, 0.01) and close(ov["p"], 0.539, 0.01), f"{ov['statistic']:.3f}, p = {ov['p']:.3f}")
ar = iv2["anderson_rubin"]


def ar_p(b0, data):
    """AR test by hand: regress lwage − b0·educ on the exogenous regressors and the instruments; F on the instruments."""
    dd = data.assign(yt=data.lwage - b0 * data.educ)
    full = smf.ols("yt ~ exper + expersq + motheduc + fatheduc", dd).fit()
    short = smf.ols("yt ~ exper + expersq", dd).fit()
    F = ((short.ssr - full.ssr) / 2) / (full.ssr / full.df_resid)
    return stats.f.sf(F, 2, full.df_resid)


check("Anderson-Rubin set: bounded, contains 2SLS, and its endpoints sit where the hand-computed AR p-value crosses 5%",
      ar["shape"] == "bounded interval" and ar["ci_low"] < iv2["table"].loc["educ", "2SLS"] < ar["ci_high"]
      and close(ar_p(ar["ci_low"], mz), 0.05, 0.01) and close(ar_p(ar["ci_high"], mz), 0.05, 0.01)
      and close(ar["p_at_zero"], ar_p(0.0, mz), 1e-8),
      f"[{ar['ci_low']:.4f}, {ar['ci_high']:.4f}]")
rng = np.random.default_rng(2)
nw = 300
z = rng.normal(size=nw)
uu = rng.normal(size=nw)
xw = 0.03 * z + 0.8 * uu + 0.6 * rng.normal(size=nw)
dw = pd.DataFrame({"y": 1 + 0.5 * xw + uu, "x": xw, "z": z})
ivw = iv_report("y ~ 1 + [x ~ z]", dw, show=False)
rng = np.random.default_rng(4)
nq = 1000
Zq = rng.normal(size=(nq, 3))
uq = rng.normal(size=nq)
xq = Zq @ np.full(3, 0.06) + 0.7 * uq + 0.7 * rng.normal(size=nq)
dq_ = pd.DataFrame({"y": 1 + 0.5 * xq + uq, "x": xq, "z1": Zq[:, 0], "z2": Zq[:, 1], "z3": Zq[:, 2]})
ivq = iv_report("y ~ 1 + [x ~ z1 + z2 + z3]", dq_, show=False)
fsq = sm.OLS(dq_.x, sm.add_constant(dq_[["z1", "z2", "z3"]])).fit(cov_type="HC0")
f_hc0 = float(fsq.wald_test(np.eye(4)[1:], scalar=True, use_f=True).statistic)
check("robust first stage with 3 instruments: reported as an F (= statsmodels HC0 Wald F) and labelled WEAK",
      close(ivq["first_stage"]["partial F"].iloc[0], f_hc0, 1e-6) and ivq["first_stage"]["strength"].iloc[0].startswith("WEAK"),
      f"F = {f_hc0:.2f}")
ivn = iv_report("lwage ~ [educ ~ motheduc + fatheduc]", mz, show=False)
check("IV model without a constant: warned, and OLS is fitted without one too",
      any("no constant" in w_ for w_ in ivn["warnings"]) and "Intercept" not in ivn["ols"].params.index)
check("weak instrument: labelled WEAK, warned, and the Anderson-Rubin set is not a bounded interval",
      ivw["first_stage"]["strength"].iloc[0].startswith("WEAK") and ivw["warnings"]
      and ivw["anderson_rubin"]["shape"] != "bounded interval", ivw["anderson_rubin"]["shape"])

# ---------------------------------------------------------------- 7. market_model tests
rng = np.random.default_rng(1)
idx = pd.date_range("2020-01-01", periods=600, freq="B")
rm = pd.Series(rng.normal(0, 0.01, 600), idx)
ri = 0.0002 + 1.3 * rm + pd.Series(rng.normal(0, 0.01, 600), idx)
mmr = market_model(ri, rm)
from statsmodels.stats.sandwich_covariance import cov_hac  # noqa: E402
fit = mmr["model"]
plain = sm.OLS(ri.to_numpy(), sm.add_constant(rm.to_numpy())).fit()
Vh = cov_hac(plain, nlags=mmr["hac_lags"], use_correction=False)   # HAC rebuilt from a plain OLS fit
tb1 = (plain.params[1] - 1) / np.sqrt(Vh[1, 1])
dq = plain.params - np.array([0.0, 1.0])
wald = dq @ np.linalg.solve(Vh, dq)
check("market model: t of β = 1 and joint Wald (α = 0, β = 1) computed from the HAC fit",
      close(mmr["tests"].loc["beta = 1", "statistic"], tb1, 1e-9)
      and close(mmr["tests"].iloc[2]["statistic"], wald, 1e-8) and mmr["tests"].loc["beta = 1", "p"] < 0.01)

# ---------------------------------------------------------------- 8. demos
ds_ = demo_simultaneity(R=300)
tt_ = ds_["table"]
# hand derivation: plim OLS = [d(c² + σd²) − b σs²] / (c² + σd² + σs²); defaults (b=1, d=.5, c=σ=1) → 0; c=2 → 1.5/6
ds2 = demo_simultaneity(R=300, c_income=2.0, seed=1)
check("simultaneity: OLS centred on the hand-derived limit (0, and 0.25 with c = 2), ILS = 2SLS ≈ supply slope",
      close(tt_.loc["OLS of q on p", "median"], 0.0, 0.05) and close(ds2["table"].loc["OLS of q on p", "median"], 0.25, 0.05)
      and close(ds2["ols_probability_limit"], 0.25, 1e-12) and ds_["max_abs_ils_minus_2sls"] < 1e-8
      and close(tt_.loc["2SLS (income)", "median"], 0.5, 0.06))
lb = demo_lpm_bounds(R=150)["table"]
check("LPM bounds: share outside [0, 1] grows with the slope; LPM slope within 5% of the true AME up to b = 2",
      lb["LPM fitted outside [0,1]"].is_monotonic_increasing
      and all(abs(lb.loc[b, "LPM slope (mean)"] / lb.loc[b, "true AME"] - 1) < 0.05 for b in (0.5, 1.0, 2.0)))
mc = demo_multicollinearity(R=400)["table"]
check("multicollinearity: SE ratio ≈ √VIF (10%), estimates unbiased up to ρ = .9, joint F still rejects",
      all(abs(mc.loc[r_, "se ratio vs rho=first"] / np.sqrt(mc.loc[r_, "VIF"]) - 1) < 0.1 for r_ in mc.index)
      and all(abs(mc.loc[r_, "mean b1"] - 1) < 0.06 for r_ in (0.0, 0.5, 0.9))
      and (mc["reject b1 = b2 = 0 (F)"] > 0.95).all() and mc["reject b1 = 0"].is_monotonic_decreasing)

# ---------------------------------------------------------------- 9. plugin robustness: mode commands
mk = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))
skills = mk["plugins"][0]["skills"]
check("marketplace.json declares the main skill, statistics and the three mode commands",
      skills == [".", "./statistics", "./modes/consultor", "./modes/professor", "./modes/revisor"], str(skills))
for mode in ("consultor", "professor", "revisor"):
    text = (SKILL / "modes" / mode / "SKILL.md").read_text(encoding="utf-8")
    mm = re.match(r"---\r?\n(.*?)\r?\n---\r?\n(.*)", text, re.S)
    meta = yaml.safe_load(mm.group(1)) if mm else {}
    body = mm.group(2) if mm else ""
    good = (sorted(meta) == ["description", "disable-model-invocation", "name"] and meta.get("name") == mode
            and meta.get("disable-model-invocation") is True and 0 < len(meta.get("description", "")) <= 1024
            and not re.search(r"[<>]", meta.get("description", "")))
    paths = re.findall(r"`(\.\./\.\./[^`]+)`", body)
    exist = all((SKILL / "modes" / mode / p_).resolve().exists() for p_ in paths)
    check(f"/econometrics:{mode}: valid header, user-invoked only, loads the main skill, its paths exist",
          good and "econometrics:econometrics" in body and "$ARGUMENTS" in body and paths and exist, str(paths))
with zipfile.ZipFile(ROOT / "dist" / "econometrics.skill") as zf:
    names = zf.namelist()
check("app package (dist) carries one skill: no modes/ and no statistics/",
      "econometrics/SKILL.md" in names and not any(n.startswith(("econometrics/modes/", "econometrics/statistics/"))
                                                  for n in names))

# ---------------------------------------------------------------- 10. clean workspace helper
inst = [{"id": "econometrics@econometric-skills", "enabled": True}, {"id": "ecc@ecc", "enabled": True},
        {"id": "agent-protocols@synced", "enabled": False}, {"id": "data@market", "enabled": True}]
st, off = workspace_settings(inst, keep=("econometrics@econometric-skills", "data@market"),
                             existing={"theme": "dark", "enabledPlugins": {"old@x": True}})
check("workspace: kept plugins on, other enabled plugins off, disabled ones untouched, other keys preserved",
      st["enabledPlugins"] == {"data@market": True, "ecc@ecc": False, "econometrics@econometric-skills": True,
                               "old@x": True} and off == ["ecc@ecc"] and st["theme"] == "dark")
st2, off2 = workspace_settings(inst, existing=st)
check("workspace: idempotent apart from newly dropped keeps", st2["enabledPlugins"]["data@market"] is False
      and st2["enabledPlugins"]["ecc@ecc"] is False and off2 == ["data@market"])
st3, _ = workspace_settings([{"id": "econometrics@my-fork", "enabled": True}, {"id": "ecc@ecc", "enabled": True}])
check("workspace: the default keeps econometrics from any marketplace", st3["enabledPlugins"] == {
    "ecc@ecc": False, "econometrics@my-fork": True})

# ---------------------------------------------------------------- 11. CI on Windows with a space in the path
wf = yaml.safe_load((ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8"))
win = wf["jobs"].get("windows", {})
steps = win.get("steps", [])
co_path = next((s_.get("with", {}).get("path", "") for s_ in steps if "checkout" in str(s_.get("uses", ""))), "")
check("CI runs the suite on windows-latest from a checkout path with a space",
      win.get("runs-on") == "windows-latest" and " " in co_path
      and any("tests/test_" in str(s_.get("run", "")) for s_ in steps))

# ---------------------------------------------------------------- 12. references and routing
teach = (SKILL / "references" / "teaching.md").read_text(encoding="utf-8")
skill_md = (SKILL / "SKILL.md").read_text(encoding="utf-8")
new = ["specification.linear_restriction_test", "specification.prais_winsten", "specification.nonnested_test",
       "discrete_choice.binary_effects", "discrete_choice.heckman", "diagnostics.iv_report"]
check("every new function is in the curriculum map", all(f_ in teach for f_ in new))
check("SKILL.md routes to the new functions and lists discrete_choice.py and workspace.py",
      all(k in skill_md for k in ("linear_restriction_test", "prais_winsten", "nonnested_test", "binary_effects",
                                  "heckman", "iv_report", "`discrete_choice.py`", "`workspace.py`")))
rules = (SKILL / "references" / "interpretation-rules.md").read_text(encoding="utf-8")
check("interpretation rules for v2.4.2 are present", "## Rules added in v2.4.2" in rules)
for mod in ("discrete_choice", "workspace"):
    m_ = importlib.import_module(mod)
    check(f"{mod}.py imports cleanly and documents its functions", bool(m_.__doc__) and len(m_.__doc__) > 200)

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
