"""Tests for linear_models.py (v2.1): ANOVA, pairwise comparisons, ANCOVA, robust and quantile
regression, permutation and lack-of-fit tests, Box-Cox, partial residuals, shrinkage,
post-double-selection and best subsets.

Benchmarks: scipy and statsmodels as independent implementations; processes with a known
answer (outliers, location-scale errors, curvature, log-normal y, sparse coefficients,
confounding through many controls); Monte Carlo size of the permutation test.

Run from the repository root:  python3 tests/test_linear_models.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "econometrics" / "scripts"))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

import linear_models as lm

ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------------------------
# One-way ANOVA, pairwise comparisons
# ---------------------------------------------------------------------------------------------
g = np.repeat(["a", "b", "c", "d"], 40)
means = {"a": 0.0, "b": 0.0, "c": 0.8, "d": 1.6}
da = pd.DataFrame({"g": g, "y": [means[k] for k in g] + rng.normal(size=len(g))})
m = smf.ols("y ~ C(g)", da).fit()
at = lm.anova_table(m, typ=2)
f_sc = stats.f_oneway(*[da.loc[da.g == k, "y"] for k in "abcd"])
check("one-way ANOVA F = scipy f_oneway", abs(at.loc["C(g)", "F"] - f_sc.statistic) < 1e-8, f"F = {f_sc.statistic:.2f}")
check("partial eta-squared in (0, 1)", 0 < at.loc["C(g)", "partial_eta_sq"] < 1)
tk = lm.pairwise(da, "y", "g", method="tukey")
pair = lambda t, a, b: t[(t.group1 == a) & (t.group2 == b)].iloc[0]
check("Tukey: a vs d significant, a vs b not", pair(tk, "a", "d")["reject"] and not pair(tk, "a", "b")["reject"])
hm = lm.pairwise(da, "y", "g", method="holm")
check("Holm-adjusted p-values >= raw p-values", (hm["p_adj"] >= hm["p"] - 1e-12).all())
check("Holm: a vs d significant, a vs b not", pair(hm, "a", "d")["reject"] and not pair(hm, "a", "b")["reject"])

# ---------------------------------------------------------------------------------------------
# ANCOVA: parallel and non-parallel slopes
# ---------------------------------------------------------------------------------------------
n = 300
grp = rng.choice(["ctl", "trt"], n)
x = rng.normal(size=n) + (grp == "trt") * 0.8          # groups differ in the covariate
dp = pd.DataFrame({"grp": grp, "x": x, "y": 1 + 0.5 * (grp == "trt") + 2 * x + rng.normal(size=n)})
an = lm.ancova(dp, "y", "grp", "x")
diff = an["adjusted_means"].set_index("group")["adjusted_mean"].diff().iloc[-1]
se_diff = an["model"].bse["C(grp)[T.trt]"]
check("ANCOVA parallel slopes not rejected (p > 0.05)", an["parallel_slopes_p"] > 0.05, f"p = {an['parallel_slopes_p']:.3f}")
check("ANCOVA adjusted difference = group coefficient and within 3 SE of 0.5 (raw ~2.1)",
      abs(diff - an["model"].params["C(grp)[T.trt]"]) < 1e-10 and abs(diff - 0.5) < 3 * se_diff,
      f"adjusted {diff:.2f}, raw {dp.groupby('grp').y.mean().diff().iloc[-1]:.2f}")
dnp = dp.assign(y=1 + 0.5 * (grp == "trt") + (2 + 1.5 * (grp == "trt")) * x + rng.normal(size=n))
check("ANCOVA non-parallel slopes rejected (p < 0.001)", lm.ancova(dnp, "y", "grp", "x")["parallel_slopes_p"] < 0.001)

# ---------------------------------------------------------------------------------------------
# Robust regression: 10% gross outliers in y
# ---------------------------------------------------------------------------------------------
n = 400
x = rng.normal(size=n)
y = 1 + 2 * x + rng.normal(size=n)
bad = rng.choice(n, 40, replace=False)
y[bad] += 15 * (x[bad] > 0)                              # outliers that tilt the OLS slope
dr = pd.DataFrame({"x": x, "y": y})
rr = lm.robust_regression(dr, "y ~ x", norm="bisquare")
check("bisquare slope closer to 2 than OLS under contamination",
      abs(rr["table"].loc["x", "robust"] - 2) < abs(rr["table"].loc["x", "ols"] - 2) / 3,
      f"robust {rr['table'].loc['x', 'robust']:.2f}, OLS {rr['table'].loc['x', 'ols']:.2f}")
check("the planted outliers are down-weighted", len(set(rr["downweighted"].index) & set(bad[x[bad] > 0])) >= 0.8 * (x[bad] > 0).sum())
rh = lm.robust_regression(dr.drop(index=bad), "y ~ x", norm="huber")
check("clean data: Huber ~ OLS (difference < 1 OLS SE)", abs(rh["table"].loc["x", "difference_in_ols_se"]) < 1)

# ---------------------------------------------------------------------------------------------
# Quantile regression: location-scale model y = x + (1 + x) e, slope(tau) = 1 + q_e(tau)
# ---------------------------------------------------------------------------------------------
n = 3000
x = rng.uniform(0, 2, n)
dq = pd.DataFrame({"x": x, "y": x + (1 + x) * rng.normal(size=n)})
qr = lm.quantile_regression(dq, "y ~ x", taus=(0.1, 0.5, 0.9), B=150)
t = qr["table"].set_index(["tau", "term"])
truth = {0.1: 1 + stats.norm.ppf(0.1), 0.5: 1.0, 0.9: 1 + stats.norm.ppf(0.9)}
check("quantile slopes within 3 SE of 1 + Phi^-1(tau) at tau = 0.1, 0.5, 0.9",
      all(abs(t.loc[(k, "x"), "estimate"] - v) < 3 * t.loc[(k, "x"), "se"] for k, v in truth.items()),
      {k: round(t.loc[(k, "x"), "estimate"], 2) for k in truth})
check("equal-slope test rejects under heteroskedasticity", qr["equality_test"]["p"] < 0.001)
dh = pd.DataFrame({"x": x, "y": x + rng.normal(size=n)})
check("equal-slope test does not reject under homoskedasticity",
      lm.quantile_regression(dh, "y ~ x", taus=(0.1, 0.9), B=150)["equality_test"]["p"] > 0.05)

# ---------------------------------------------------------------------------------------------
# Permutation test: power and size with heavy tails in a small sample
# ---------------------------------------------------------------------------------------------
x1 = rng.normal(size=30)
dpm = pd.DataFrame({"x1": x1, "x2": rng.normal(size=30)})
dpm["y"] = 0.8 * dpm.x1 + 0.5 * dpm.x2 + rng.normal(size=30)
pt = lm.permutation_test(dpm, "y ~ x1 + x2", "x1", n_perm=1999)
check("permutation test detects a real effect (p < 0.01)", pt["p"] < 0.01, f"p = {pt['p']:.4f}")
rej = []
for rep in range(300):
    r = np.random.default_rng(900 + rep)
    d0 = pd.DataFrame({"x1": r.normal(size=25), "x2": r.normal(size=25)})
    d0["y"] = 0.5 * d0.x2 + r.standard_t(2, 25)
    rej.append(lm.permutation_test(d0, "y ~ x1 + x2", "x1", n_perm=499, seed=rep)["p"] < 0.05)
check("permutation size with t(2) errors and n = 25 within [2%, 8%]", 0.02 <= np.mean(rej) <= 0.08, f"{np.mean(rej):.3f}")

# ---------------------------------------------------------------------------------------------
# Lack of fit, Box-Cox, partial residuals
# ---------------------------------------------------------------------------------------------
xs = np.repeat(np.linspace(0, 4, 9), 8)
dl = pd.DataFrame({"x": xs, "y": 1 + xs + 0.6 * xs ** 2 + rng.normal(size=len(xs))})
lf = lm.lack_of_fit_test(smf.ols("y ~ x", dl).fit(), dl, ["x"])
check("linear fit to a quadratic: lack of fit rejected", lf["p"] < 0.001, f"F = {lf['F']:.1f}")
lf2 = lm.lack_of_fit_test(smf.ols("y ~ x + I(x**2)", dl).fit(), dl, ["x"])
check("correct quadratic: no lack of fit (p > 0.05)", lf2["p"] > 0.05, f"p = {lf2['p']:.3f}")
try:
    lm.lack_of_fit_test(smf.ols("y ~ x", dr).fit(), dr, ["x"])
    check("no replicates raises an informative error", False)
except ValueError:
    check("no replicates raises an informative error", True)

xb = rng.uniform(0, 2, 800)
dbx = pd.DataFrame({"x": xb, "y": np.exp(0.5 + 0.8 * xb + 0.3 * rng.normal(size=800))})
bc = lm.boxcox_response(dbx, "y ~ x")
check("Box-Cox on log-normal y: lambda near 0 (|lambda| < 0.15), far from 1",
      abs(bc["lambda_hat"]) < 0.15 and bc["ci_high"] < 0.5,
      f"lambda {bc['lambda_hat']:.2f} [{bc['ci_low']:.2f}, {bc['ci_high']:.2f}]")
dlin = pd.DataFrame({"x": xb, "y": 10 + 2 * xb + rng.normal(size=800)})
check("Box-Cox on linear y: CI contains 1", lm.boxcox_response(dlin, "y ~ x")["suggestion"] == "no transformation")

pr = lm.partial_residuals(smf.ols("y ~ x", dl).fit(), dl, "x")
check("partial residuals flag curvature in a quadratic", pr["curvature_p"] < 0.001)
check("partial residuals: linear term adequate when it is", lm.partial_residuals(smf.ols("y ~ x", dlin).fit(), dlin, "x")["curvature_p"] > 0.05)

cov_bc = []
for rep in range(100):
    r = np.random.default_rng(7000 + rep)
    xr = r.uniform(0, 2, 400)
    b = lm.boxcox_response(pd.DataFrame({"x": xr, "y": np.exp(0.5 + 0.8 * xr + 0.3 * r.normal(size=400))}), "y ~ x")
    cov_bc.append(b["ci_low"] <= 0 <= b["ci_high"])
check("Monte Carlo: Box-Cox 95% CI covers lambda = 0 in [0.90, 0.99] of samples", 0.90 <= np.mean(cov_bc) <= 0.99,
      f"{np.mean(cov_bc):.2f}")

# ---------------------------------------------------------------------------------------------
# Shrinkage, post-double-selection, best subsets
# ---------------------------------------------------------------------------------------------
n, p = 200, 40
Xs = rng.normal(size=(n, p))
beta = np.zeros(p)
beta[:4] = [2, -1.5, 1, 0.5]
ds = pd.DataFrame(Xs, columns=[f"v{j}" for j in range(p)])
ds["y"] = Xs @ beta + rng.normal(size=n)
sc = lm.shrinkage_compare(ds, "y", [f"v{j}" for j in range(p)], cv=5)
tab = sc["table"]
check("sparse truth: lasso CV RMSE beats OLS", tab.loc["lasso", "cv_rmse"] < tab.loc["OLS", "cv_rmse"], tab["cv_rmse"].round(3).to_dict())
check("lasso keeps the four true regressors", {"v0", "v1", "v2", "v3"} <= set(sc["lasso_selected"]))

cover, est = [], []
for rep in range(80):
    r = np.random.default_rng(3000 + rep)
    n, p = 400, 60
    Xc = r.normal(size=(n, p))
    dtr = Xc[:, :5] @ np.array([1.0, 1, 1, 0.8, 0.8]) + r.normal(size=n)       # confounders drive d strongly
    yy = 0.5 * dtr + Xc[:, :5] @ np.array([0.3, 0.3, 0.3, 0.2, 0.2]) + Xc[:, 5:8].sum(1) + r.normal(size=n)
    dd = pd.DataFrame(Xc, columns=[f"c{j}" for j in range(p)])
    dd["d"], dd["y"] = dtr, yy
    res = lm.post_double_selection(dd, "y", "d", [f"c{j}" for j in range(p)], seed=rep)
    cover.append(res["ci_low"] < 0.5 < res["ci_high"])
    est.append(res["estimate"])
check("post-double-selection: 95% CI coverage >= 0.88 with 60 candidate controls", np.mean(cover) >= 0.88, f"{np.mean(cover):.3f}")
check("post-double-selection: mean estimate within 0.03 of 0.5", abs(np.mean(est) - 0.5) < 0.03, f"{np.mean(est):.3f}")

w = rng.normal(size=(300, 6))
dsub = pd.DataFrame(w, columns=list("abcdef"))
dsub["y"] = 1 + 2 * dsub.a - dsub.c + rng.normal(size=300)
ss = lm.select_subsets(dsub, "y", list("abcdef"), criterion="bic")
check("best subsets by BIC ranks the true model {a, c} first", ss.loc[0, "variables"] == "a, c", ss.loc[0, "variables"])

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
