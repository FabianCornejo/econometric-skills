"""What the `show` blocks print: the 20 functions with interpretation sentences (v2.6.7).

The numbers these functions return are tested elsewhere; what they print was not (lesson of 2026-11-10: RV(q=5)
printed for q = 1 lived five versions in a `show` block). For each function, on data whose verdict is known by
construction, the captured text is checked for:
  - labels: every number printed is the one in the returned object, at the printed precision, and labels that
    depend on an argument (alpha, cov_type, ties, cluster, strata, honest, time_col, fuzzy, h) follow it;
  - sentences against the definition, computed another way where possible (Holm by statsmodels.multitest, the
    Diebold-Mariano statistic with h = 1 as the OLS t of the loss difference on a constant, Chow's F from three
    least-squares fits, the Clayton tail coefficient 2^(-1/theta), counts from the raw data);
  - each threshold with one case on each side (alpha, by moving alpha around the observed p where the function
    takes alpha; first-stage F = 10 and 104.7; excess zeros 1.2x; 3 violations for McNeil-Frey; 200-300 days and
    alpha = 2.5% for the Acerbi-Szekely zone);
  - what must not be printed: a verdict when the test could not be computed (p = NaN), a tail-risk claim for a
    copula without tail dependence, "use IV/2SLS" with weak instruments, "OLS is consistent" from a non-rejection;
  - every warning in the returned object is relayed in the text.

Text defects fixed in v2.6.7 (each check fails on main 74a41a5 and passes on the branch):
  dag.test_implications       printed "Holm, 5%" whatever alpha was;
  dependence.fit_copulas      said the Gaussian copula "understates joint tail risk" when the best family (Frank)
                              has no tail dependence either;
  causal.event_study          with no lead to test (p = NaN) printed "no evidence of pre-trends";
  diagnostics.diebold_mariano with a non-positive long-run variance (h > 1) printed "no significant difference",
                              and forecasting.evaluate relayed a NaN p as "not significantly ... accurate";
  diagnostics.endogeneity_test said "use IV/2SLS" with weak instruments, and "OLS is consistent" from a
                              non-rejection;
  diagnostics.chow            gave the verdict of the robust test without saying so (it can disagree with the F);
  data_prep.missing_report    printed "Rejection: not MCAR" without saying which columns, if any, reject.

Run from the repository root:  python3 tests/test_show_text.py
Needs: statsmodels, linearmodels, scipy, scikit-learn.
"""
import contextlib
import io
import re
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "econometrics" / "scripts"))
import _console  # noqa: E402,F401  (Windows-safe console output, see econometrics/scripts/_console.py)
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

import causal
import dag
import data_prep
import dependence
import diagnostics as dg
import discrete_choice
import forecasting
import risk
import segments
import survival

ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  [{info}]"))


def run(f, *a, **k):
    """Call f with show on and return (result, captured stdout)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = f(*a, **k)
    return res, buf.getvalue()


def has(text, s):
    return s in text


def relayed(text, warns, prefix):
    """Every warning of the returned object appears in the text with its prefix."""
    return all(f"{prefix}{w}" in text for w in warns)


# =============================================================================
# diagnostics
# =============================================================================
rng = np.random.default_rng(10)
n = 200
X = pd.DataFrame({"x": rng.normal(size=n)})
e = rng.normal(size=n)
y_break = pd.Series(1 + np.where(np.arange(n) >= 100, 4.0, 2.0) * X["x"] + e, name="y")
y_flat = pd.Series(1 + 2.0 * X["x"] + e, name="y")


def chow_F(y, x, point):
    """Chow F from three least-squares fits (numpy), k = 2 with the constant."""
    Z = np.column_stack([np.ones(len(x)), np.asarray(x)])
    yy = np.asarray(y)
    post = np.arange(len(yy)) >= point
    ssr = lambda A, b: float(np.sum((b - A @ np.linalg.lstsq(A, b, rcond=None)[0]) ** 2))  # noqa: E731
    s, s1, s2 = ssr(Z, yy), ssr(Z[~post], yy[~post]), ssr(Z[post], yy[post])
    return ((s - s1 - s2) / 2) / ((s1 + s2) / (len(yy) - 4))


r, t = run(dg.chow, y_break, X, 100)
check("chow: F label F(2, 196) and F equal to the three-fit Chow F",
      has(t, f"F(2,196) = {chow_F(y_break, X['x'], 100):.3f}") and abs(r["F"] - chow_F(y_break, X["x"], 100)) < 1e-8, t)
check("chow: planted break -> the verdict says break and names the robust test",
      r["robust_p"] < 0.05 and has(t, "robust test: structural break"), t)
r, t = run(dg.chow, y_flat, X, 100)
check("chow: no break -> 'no evidence of a break' from the robust test, never 'structural break'",
      r["robust_p"] >= 0.05 and has(t, "robust test: no evidence of a break") and "structural break" not in t, t)
check("chow: printed p values are the returned ones",
      has(t, f"p = {r['p']:.4f}") and has(t, f"(HC3) = {r['robust_wald']:.3f}, p = {r['robust_p']:.4f}"), t)
# heteroskedastic errors where the classical F rejects and the robust Wald does not: the text must not give the
# classical F's verdict as if it were the only one
found = None
for s in range(400):
    g = np.random.default_rng(s)
    xs = g.normal(size=120)
    ee = g.normal(size=120) * np.where(np.arange(120) >= 60, 4.0, 0.5)
    yy = pd.Series(1 + xs + ee, name="y")
    rr, tt = run(dg.chow, yy, pd.DataFrame({"x": xs}), 60)
    if rr["p"] < 0.05 <= rr["robust_p"]:
        found = (rr, tt)
        break
check("chow: classical F rejects, robust does not -> text says 'robust test: no evidence of a break'",
      found is not None and has(found[1], "robust test: no evidence of a break"), found and found[1])

# Diebold-Mariano
e1 = rng.normal(size=120)
e2 = 1.5 * rng.normal(size=120)
r, t = run(dg.diebold_mariano, e1, e2)
dlt = e1 ** 2 - e2 ** 2
t_ols = sm.OLS(dlt, np.ones_like(dlt)).fit()
check("diebold_mariano: with h = 1 the HLN statistic is the OLS t of the loss difference on a constant",
      abs(r["DM_HLN"] - t_ols.tvalues[0]) < 1e-9 and abs(r["p"] - 2 * stats.t.sf(abs(t_ols.tvalues[0]), 119)) < 1e-12)
check("diebold_mariano: model 1 has the smaller loss -> 'significant difference, model 1 more accurate'",
      r["p"] < 0.05 and r["mean_loss_diff"] < 0 and has(t, "significant difference, model 1 more accurate")
      and has(t, f"(HLN): {r['DM_HLN']:.3f}, p = {r['p']:.4f}"), t)
r, t = run(dg.diebold_mariano, e2, e1)
check("diebold_mariano: models swapped -> 'model 2 more accurate'", has(t, "model 2 more accurate"), t)
e3 = e1 + 0.01 * rng.normal(size=120)
r, t = run(dg.diebold_mariano, e1, e3)
check("diebold_mariano: indistinguishable errors -> 'no significant difference', no direction",
      r["p"] >= 0.05 and has(t, "no significant difference") and "more accurate" not in t, t)
# the design of claude/scripts/sonda_v267.py (seed 7): the long-run variance with h = 4 is not positive
g7 = np.random.default_rng(7)
a7, b7 = g7.normal(size=20), g7.normal(size=20)
a7 = a7 * np.where(np.arange(20) % 2, 1.6, 0.4)
r, t = run(dg.diebold_mariano, a7, b7, h=4)
check("diebold_mariano: non-positive long-run variance (h = 4) -> NaN statistic kept in the result",
      not np.isfinite(r["DM_HLN"]) and not np.isfinite(r["p"]), r)
check("diebold_mariano: ... and the text says the test was not computed, never 'no significant difference'",
      "no significant difference" not in t and has(t, "not computed"), t)

# endogeneity_test: first-stage label at each band, and the verdict with weak instruments
n2 = 2000
Z = rng.normal(size=(n2, 2))
u = rng.normal(size=n2)
v = 0.7 * u + rng.normal(size=n2)
Zdf = pd.DataFrame(Z, columns=["z1", "z2"])
outs = {}
for pi in (1.0, 0.12, 0.05):
    xe = pi * Z[:, 0] + pi * Z[:, 1] + v
    outs[pi] = run(dg.endogeneity_test, pd.Series(1 + xe + u, name="y"), None, pd.DataFrame({"xe": xe}), Zdf)
(rs, ts), (ra, ta), (rw, tw) = outs[1.0], outs[0.12], outs[0.05]
fs = lambda r_: r_["first_stage_F"]["xe"]  # noqa: E731
check("endogeneity_test: F > 104.7 -> 'strong', and F printed is the returned one",
      fs(rs) > 104.7 and has(ts, f"= {fs(rs):.2f} → strong"), ts)
check("endogeneity_test: 10 < F <= 104.7 -> 'acceptable (>10); also report Anderson-Rubin'",
      10 < fs(ra) <= 104.7 and has(ta, "acceptable (>10); also report Anderson-Rubin"), ta)
check("endogeneity_test: F <= 10 -> 'WEAK'", fs(rw) <= 10 and has(tw, "→ WEAK"), tw)
check("endogeneity_test: strong instruments and rejection -> 'endogenous: use IV/2SLS'",
      rs["p"] < 0.05 and has(ts, "endogenous: use IV/2SLS"), ts)
check("endogeneity_test: weak instruments -> never 'use IV/2SLS'; says 2SLS is unreliable and points to "
      "weak-instrument-robust inference", rw["p"] < 0.05 and "use IV/2SLS" not in tw
      and has(tw, "weak instruments") and has(tw, "Anderson-Rubin"), tw)
check("endogeneity_test: the printed DWH line carries cov_type and the returned F and p",
      has(ts, f"(control function, HC3): F = {rs['statistic']:.3f}, p = {rs['p']:.4f}"), ts)
# the label of the first stage at the exact thresholds (the helper the show block uses)
lab = getattr(dg, "_first_stage_label", None)
check("endogeneity_test: F = 10 is WEAK, just above 10 is acceptable; 104.7 is acceptable, just above is strong",
      lab is not None and lab(10.0) == "WEAK" and lab(10.0 + 1e-9).startswith("acceptable")
      and lab(104.7).startswith("acceptable") and lab(104.7 + 1e-9) == "strong")
# exogenous regressor, strong instruments: no rejection, and no claim that OLS is consistent
xe = Z[:, 0] + Z[:, 1] + rng.normal(size=n2)
r, t = run(dg.endogeneity_test, pd.Series(1 + xe + u, name="y"), None, pd.DataFrame({"xe": xe}), Zdf,
           cov_type="HC1")
check("endogeneity_test: exogenous x -> 'no evidence of endogeneity', with cov_type HC1 in the label",
      r["p"] >= 0.05 and has(t, "no evidence of endogeneity") and has(t, "(control function, HC1)"), t)
check("endogeneity_test: a non-rejection does not say 'OLS is consistent'; it says the test does not prove "
      "exogeneity", "OLS is consistent" not in t and has(t, "not proof of exogeneity"), t)
xe = 0.05 * Z[:, 0] + 0.05 * Z[:, 1] + rng.normal(size=n2)
r, t = run(dg.endogeneity_test, pd.Series(1 + xe + u, name="y"), None, pd.DataFrame({"xe": xe}), Zdf)
check("endogeneity_test: weak instruments and no rejection -> the test has little power, no OLS recommendation",
      r["p"] >= 0.05 and has(t, "weak instruments") and has(t, "little power") and "OLS is" not in t, t)

# overid_test
import linearmodels.iv as lmiv  # noqa: E402

xe = Z[:, 0] + Z[:, 1] + v
iv_ok = lmiv.IV2SLS(pd.Series(1 + xe + u), pd.DataFrame({"c": np.ones(n2)}), pd.Series(xe, name="xe"), Zdf).fit()
r, t = run(dg.overid_test, iv_ok.resids, None, Zdf, 1)
aux = sm.OLS(np.asarray(iv_ok.resids), sm.add_constant(Zdf)).fit()
check("overid_test: nR2 = n times the R2 of the residuals on the instruments, chi2(1)",
      abs(r["nR2"] - n2 * aux.rsquared) < 1e-9 and has(t, f"nR² = {r['nR2']:.3f}") and has(t, "χ²(1)"), t)
check("overid_test: valid instruments -> 'no evidence against validity (this does not prove validity)'",
      r["p"] >= 0.05 and has(t, "no evidence against validity (this does not prove validity)"), t)
iv_bad = lmiv.IV2SLS(pd.Series(1 + xe + u + 0.3 * Z[:, 1]), pd.DataFrame({"c": np.ones(n2)}),
                     pd.Series(xe, name="xe"), Zdf).fit()
r, t = run(dg.overid_test, iv_bad.resids, None, Zdf, 1)
check("overid_test: z2 in the outcome equation -> 'at least one instrument is invalid'",
      r["p"] < 0.05 and has(t, "at least one instrument is invalid") and "no evidence" not in t, t)

# count_diagnostics
n3 = 1500
xx = rng.normal(size=n3)
mu = np.exp(0.5 + 0.4 * xx)
fit_p = smf.poisson("y ~ x", pd.DataFrame({"y": rng.poisson(mu), "x": xx})).fit(disp=0)
fit_nb = smf.poisson("y ~ x", pd.DataFrame({"y": rng.negative_binomial(2, 2 / (2 + mu)), "x": xx})).fit(disp=0)
r, t = run(dg.count_diagnostics, fit_p)
yv, mh = fit_p.model.endog, fit_p.predict()
disp = np.sum((yv - mh) ** 2 / mh) / (n3 - 2)
check("count_diagnostics: Pearson dispersion from its definition, printed with sigma",
      abs(r["dispersion"] - disp) < 1e-10 and has(t, f"σ̂² = {disp:.3f} (σ̂ = {np.sqrt(disp):.3f}"), t)
check("count_diagnostics: Poisson data -> 'No evidence of overdispersion' and no excess-zeros advice",
      has(t, "No evidence of overdispersion") and "excess zeros" not in t
      and r["zeros_observed"] <= 1.2 * r["zeros_predicted"], t)
check("count_diagnostics: zeros line uses the returned shares",
      has(t, f"Zeros: observed {r['zeros_observed']:.1%} vs Poisson-implied {r['zeros_predicted']:.1%}"), t)
r, t = run(dg.count_diagnostics, fit_nb)
check("count_diagnostics: negative binomial data -> overdispersion advice and excess zeros (> 1.2x)",
      has(t, "Overdispersion: keep Poisson coefficients") and r["zeros_observed"] > 1.2 * r["zeros_predicted"]
      and has(t, "excess zeros: consider hurdle"), t)

# mundlak_test
rows = []
for i in range(150):
    a = rng.normal()
    for tt_ in range(5):
        xv = a + rng.normal()
        rows.append(dict(i=i, t=tt_, x=xv, x2=rng.normal(), y=1 + xv + a + rng.normal()))
pan = pd.DataFrame(rows)
r, t = run(dg.mundlak_test, pan, "y", ["x"], "i", "t")
check("mundlak_test: effect correlated with x -> 'use FE', chi2 with the returned df, clustered by the entity",
      r["p"] < 0.05 and has(t, f"Wald χ²({r['df']}) = {r['statistic']:.3f}, p = {r['p']:.4f}")
      and has(t, "clustered by i") and has(t, "use FE"), t)
check("mundlak_test: balanced panel -> no unbalanced warning", "unbalanced" not in t, t)
pan2 = pan.assign(y=1 + pan["x2"] + rng.normal(size=len(pan)) + np.repeat(rng.normal(size=150), 5))
r, t = run(dg.mundlak_test, pan2, "y", ["x2"], "i", "t")
check("mundlak_test: effect independent of x -> 'no evidence against RE'",
      r["p"] >= 0.05 and has(t, "no evidence against RE") and "use FE" not in t, t)
r, t = run(dg.mundlak_test, pan.iloc[:-2], "y", ["x"], "i", "t")
check("mundlak_test: unbalanced with time effects -> warning", has(t, "WARNING: unbalanced panel"), t)
r, t = run(dg.mundlak_test, pan.iloc[:-2], "y", ["x"], "i", "t", time_effects=False)
check("mundlak_test: unbalanced without time effects -> no warning", "unbalanced" not in t, t)

# =============================================================================
# forecasting.evaluate
# =============================================================================
idx = pd.RangeIndex(150)
act = pd.Series(rng.normal(size=150), index=idx)
f_good = act + 0.5 * pd.Series(rng.normal(size=150), index=idx)
f_bench = act + 1.0 * pd.Series(rng.normal(size=150), index=idx)
r, t = run(forecasting.evaluate, act, {"bench": f_bench, "good": f_good, "near": f_bench + 0.01})
check("evaluate: header with the number of forecasts and the benchmark",
      has(t, "(150 forecasts, benchmark = bench)"), t)
check("evaluate: better model, significant DM -> 'significantly more accurate than bench' with its p",
      has(t, f"good: significantly more accurate than bench (DM p = {r.loc['good', 'DM_p']:.3f})"), t)
check("evaluate: indistinguishable model -> 'not significantly'", r.loc["near", "DM_p"] >= 0.05
      and has(t, "near: not significantly"), t)
r, t = run(forecasting.evaluate, act, {"bench": f_bench, "same": f_bench.copy()})
check("evaluate: identical forecasts (DM p = NaN) -> no accuracy verdict; says the DM test was not computed",
      not np.isfinite(r.loc["same", "DM_p"]) and "accurate than" not in t and has(t, "not computed"), t)

# =============================================================================
# dag
# =============================================================================
n4 = 500
zc = rng.normal(size=n4)
xc = zc + rng.normal(size=n4)
yc = xc + zc + rng.normal(size=n4)
wc = rng.normal(size=n4)
dfc = pd.DataFrame(dict(z=zc, x=xc, y=yc, w=wc))
G = "z -> x; z -> y; x -> y; w -> y"
r, t = run(dag.adjustment_report, G, "x", "y", dfc)
check("adjustment_report: confounder z -> minimal set {z}; optimal set {w, z}",
      r["adjustment_sets"] == [["z"]] or [sorted(s) for s in r["adjustment_sets"]] == [["z"]], r["adjustment_sets"])
check("adjustment_report: printed sets are the returned ones",
      has(t, "Minimal adjustment sets: {z}") and has(t, "Optimal set (smallest variance among valid sets): {w, z}"), t)
check("adjustment_report: data checks line counts the returned tests, none rejected",
      has(t, f"Data checks: {len(r['tests'])} implications, none rejected"), t)
r, t = run(dag.adjustment_report, "z -> x; z -> y; w -> y", "x", "y")
check("adjustment_report: no path x -> y -> warning relayed and optimal set 'not available (no causal path ...)'",
      relayed(t, r["warnings"], "  WARNING: ") and len(r["warnings"]) == 1
      and has(t, "not available (no causal path from x to y)"), t)
r, t = run(dag.adjustment_report, "x -> y; x <-> y", "x", "y")
check("adjustment_report: latent confounder -> 'NOT identified by adjustment' and no minimal set line",
      r["adjustment_sets"] == [] and has(t, "NOT identified by adjustment") and "Minimal adjustment sets" not in t, t)

dfw = dfc.assign(w=dfc["w"] + 0.5 * dfc["x"])
r, t = run(dag.test_implications, G, dfw)
holm = multipletests(r["p"], method="holm")[1]
check("test_implications: Holm-adjusted p equal to statsmodels.multitest (holm)", np.allclose(r["p_holm"], holm, atol=1e-12))
bad = r.loc[~r["consistent"], "implication"].tolist()
check("test_implications: data contradict -> 'Data contradict:' lists exactly the rejected implications",
      len(bad) > 0 and has(t, "Data contradict: " + "; ".join(bad) + " -> revise the graph"), t)
r, t = run(dag.test_implications, G, dfc)
check("test_implications: true graph, alpha = 0.05 -> 'No implication rejected (Holm, 5%)'",
      r["consistent"].all() and has(t, "No implication rejected (Holm, 5%)"), t)
r10, t10 = run(dag.test_implications, G, dfc, alpha=0.10)
check("test_implications: alpha = 0.10 -> the sentence says 10%, never 5%",
      has(t10, "(Holm, 10%)") and "5%" not in t10, t10)
# one case on each side of alpha, on the same data: a weak violation (w = noise + 0.1 x, seed 103, found by search
# in claude/scripts) gives Holm p values strictly inside (0, 1); put alpha just above and just below the smallest
g103 = np.random.default_rng(103)
zb = g103.normal(size=500)
xb = zb + g103.normal(size=500)
yb = xb + zb + g103.normal(size=500)
dfb = pd.DataFrame(dict(z=zb, x=xb, y=yb, w=g103.normal(size=500) + 0.1 * xb))
pm = float(run(dag.test_implications, G, dfb)[0]["p_holm"].min())
ra, ta = run(dag.test_implications, G, dfb, alpha=pm * 1.001)
rb, tb = run(dag.test_implications, G, dfb, alpha=pm * 0.999)
check("test_implications: alpha just above the smallest Holm p rejects it; just below does not",
      0 < pm < 0.5 and (~ra["consistent"]).sum() == 1 and has(ta, "Data contradict") and rb["consistent"].all()
      and has(tb, "No implication rejected"), (pm, ta, tb))
r, t = run(dag.test_implications, "a -> b", dfc)
check("test_implications: no testable implication among the columns -> says so", has(t, "No testable implications"), t)

# =============================================================================
# data_prep.missing_report
# =============================================================================
dm = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300), "c": rng.normal(size=300)})
dm.loc[dm["a"] > 1, "b"] = np.nan
dm.loc[rng.uniform(size=300) < 0.1, "c"] = np.nan
r, t = run(data_prep.missing_report, dm, predictors=["a"])
comp = int(dm.notna().all(axis=1).sum())
check("missing_report: complete rows counted from the data, with the share",
      has(t, f"Missing data: {comp} of 300 rows complete on 3 columns ({comp / 300:.1%})"), t)
check("missing_report: missing share per column printed", has(t, f"b   {dm['b'].isna().mean():.1%}"), t)
rej = [c for c in r["tests"].index if r["tests"].loc[c, "p"] < 0.05]
check("missing_report: names the columns whose missingness is predicted (b, by construction) and not c",
      rej == ["b"] and has(t, "Rejected at 5%: b") and "Rejected at 5%: b, c" not in t, t)
dm2 = dm.assign(b=np.where(rng.uniform(size=300) < 0.12, np.nan, rng.normal(size=300)))   # b missing at random
r, t = run(data_prep.missing_report, dm2, predictors=["a"])
check("missing_report: no column predicted -> says none rejects and never 'Rejection: not MCAR'",
      (r["tests"]["p"] >= 0.05).all() and has(t, "No column rejects at 5%") and "Rejected at 5%" not in t
      and "Rejection: not MCAR" not in t, t)
r, t = run(data_prep.missing_report, dm)
check("missing_report: without predictors -> no test block", "Is missingness predicted" not in t, t)

# =============================================================================
# dependence.fit_copulas
# =============================================================================
th = 3.0
uu, tt2 = rng.uniform(size=1500), rng.uniform(size=1500)
vv = (uu ** (-th) * (tt2 ** (-th / (1 + th)) - 1) + 1) ** (-1 / th)             # Clayton(3), conditional inversion
r, t = run(dependence.fit_copulas, uu, vv)
pu, pv = stats.rankdata(uu) / (len(uu) + 1), stats.rankdata(vv) / (len(vv) + 1)
check("fit_copulas: Kendall tau and the empirical lower-tail co-exceedance from the ranks",
      has(t, f"Kendall tau = {stats.kendalltau(uu, vv)[0]:.3f}")
      and has(t, f"lower {np.mean((pu <= 0.05) & (pv <= 0.05)) / 0.05:.2f}"), t)
check("fit_copulas: best by AIC is the smallest AIC and is named", r.index[0] == r["AIC"].idxmin()
      and has(t, f"Best by AIC: {r.index[0]}."), t)
lamL = float(r.loc["clayton", "lambda_L"])
check("fit_copulas: Clayton lambda_L = 2^(-1/theta) at the fitted theta (theta stored rounded to 6 decimals)",
      abs(lamL - 2 ** (-1 / float(np.ravel(r.loc["clayton", "params"])[0]))) < 1e-6)
check("fit_copulas: Clayton best -> the tail claim names the lower tail with its lambda",
      r.index[0] == "clayton" and has(t, f"lower-tail dependence (lambda_L = {lamL:.2f})")
      and has(t, "understates joint tail risk"), t)
th = 8.0
vf = -1 / th * np.log(1 + tt2 * np.expm1(-th) / (tt2 + (1 - tt2) * np.exp(-th * uu)))   # Frank(8)
r, t = run(dependence.fit_copulas, uu, vf)
check("fit_copulas: Frank best (no tail dependence) -> no tail-risk claim; says neither has tail dependence",
      r.index[0] == "frank" and "understates joint tail risk" not in t and has(t, "no tail dependence"), t)
check("fit_copulas: ... and the AIC gain over the Gaussian is still printed",
      has(t, f"AIC gain over Gaussian = {r.loc['gaussian', 'AIC'] - r.loc['frank', 'AIC']:.1f}"), t)
zg = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], 1500)
r, t = run(dependence.fit_copulas, zg[:, 0], zg[:, 1])
check("fit_copulas: Gaussian best -> no gain sentence", r.index[0] == "gaussian" and "AIC gain" not in t, t)

# =============================================================================
# discrete_choice.heckman
# =============================================================================
def heck_data(rho, seed):
    g = np.random.default_rng(seed)
    nn = 2000
    zz, x1, uu_ = g.normal(size=nn), g.normal(size=nn), g.normal(size=nn)
    ee = rho * uu_ + np.sqrt(1 - rho ** 2) * g.normal(size=nn)
    s_ = (0.3 + x1 + zz + uu_ > 0).astype(int)
    return pd.DataFrame(dict(s=s_, y=np.where(s_ == 1, 1 + x1 + ee, np.nan), x1=x1, z=zz))


dh = heck_data(0.6, 1)
r, t = run(discrete_choice.heckman, "y ~ x1", "s ~ x1 + z", dh)
check("heckman: selected count from the data and the excluded regressor named",
      has(t, f"Heckman two-step: {int(dh['s'].sum())} selected of 2000; excluded from the outcome equation: z"), t)
check("heckman: rho = 0.6 -> 'evidence of selection bias' with the returned rho and p",
      r["lambda"]["p"] < 0.05 and has(t, f"ρ = {r['rho']:.3f}, σ = {r['sigma']:.4f}; λ coefficient p = "
                                          f"{r['lambda']['p']:.4f} → evidence of selection bias"), t)
r, t = run(discrete_choice.heckman, "y ~ x1", "s ~ x1 + z", heck_data(0.0, 2))
check("heckman: rho = 0 -> 'no evidence of selection bias (not proof of its absence)'",
      r["lambda"]["p"] >= 0.05 and has(t, "no evidence of selection bias (not proof of its absence)"), t)
pl = r["lambda"]["p"]
ra, ta = run(discrete_choice.heckman, "y ~ x1", "s ~ x1 + z", heck_data(0.0, 2), alpha=min(0.99, pl * 1.001))
check("heckman: alpha just above the lambda p -> 'evidence of selection bias'",
      has(ta, "→ evidence of selection bias"), ta)
r, t = run(discrete_choice.heckman, "y ~ x1", "s ~ x1", dh)
check("heckman: no exclusion restriction -> 'none' and the warning relayed",
      has(t, "excluded from the outcome equation: none") and relayed(t, r["warnings"], "Warning: ")
      and any("no exclusion restriction" in w for w in r["warnings"]), t)

# =============================================================================
# survival.cox
# =============================================================================
n5 = 400
xs5 = rng.normal(size=n5)
T5 = rng.exponential(1 / np.exp(0.5 * xs5))
C5 = rng.exponential(2, size=n5)
ds = pd.DataFrame({"t": np.minimum(T5, C5), "e": (T5 <= C5).astype(int), "x": xs5, "g": rng.integers(0, 40, n5),
                   "s": rng.integers(0, 2, n5)})
(res, tab), t = run(survival.cox, ds, "x", "t", "e")
check("cox: header with ties, robust SE, N and events counted from the data",
      has(t, f"Cox PH (efron ties; SE robust); N = 400, events = {int((T5 <= C5).sum())}"), t)
z975 = stats.norm.ppf(0.975)
check("cox: HR = exp(coef) and the 95% CI from coef +- 1.96 SE",
      np.allclose(tab["HR"], np.exp(tab["coef"])) and
      np.allclose(tab["HR_ci_low"], np.exp(tab["coef"] - z975 * tab["se"])))
check("cox: the true log-HR 0.5 is inside the printed CI", tab.loc["x", "HR_ci_low"] < np.exp(0.5) < tab.loc["x", "HR_ci_high"])
(res, tab), t = run(survival.cox, ds, "x", "t", "e", ties="breslow", cluster="g", strata="s")
check("cox: breslow ties, clustered by g, stratified by s in the header",
      has(t, "Cox PH (breslow ties; SE clustered by g; stratified by s)"), t)
(res, tab), t = run(survival.cox, ds, "x", "t", "e", robust=False)
check("cox: robust=False -> 'model-based'", has(t, "SE model-based"), t)
check("cox: the HR sentence says hazard, not duration", has(t, "not a 25% shorter duration"), t)

# =============================================================================
# risk
# =============================================================================
T6 = 250
sc = 0.01 / np.sqrt(2)
q = stats.norm.ppf(0.025)
var6 = pd.Series(np.full(T6, -q * sc))
es6 = pd.Series(np.full(T6, stats.norm.pdf(q) / 0.025 * sc))
r_ok = pd.Series(rng.normal(size=T6) * sc)
r, t = run(risk.es_backtest, r_ok, var6, es6, n_sim=500, n_boot=500)
nv = int((r_ok < -var6).sum())
check("es_backtest: header at 97.5% with violations counted from the data and alpha*T expected",
      has(t, f"ES backtest at 97.5% (250 days, {nv} violations, 6.2 expected)"), t)
check("es_backtest: correct model, 250 days -> zone shown and 'no evidence that ES is underestimated at 5%'",
      r["p_Z2"] >= 0.05 and has(t, f"zone: {r['Z2_zone_AS']}") and has(t, "no evidence that ES is underestimated"), t)
check("es_backtest: McNeil-Frey line only with at least 3 violations",
      (nv >= 3) == has(t, "McNeil-Frey:"), t)
r_bad = pd.Series(rng.normal(size=600) * sc * 1.4)
r, t = run(risk.es_backtest, r_bad, pd.Series(np.full(600, var6[0])), pd.Series(np.full(600, es6[0])),
           n_sim=500, n_boot=500)
check("es_backtest: volatility 1.4x the model's, 600 days -> 'ES underestimated (Z2 rejects)', no zone",
      r["p_Z2"] < 0.05 and has(t, "ES underestimated (Z2 rejects)") and "zone:" not in t
      and "no evidence" not in t, t)
r, t = run(risk.es_backtest, r_ok.iloc[:40], var6.iloc[:40], es6.iloc[:40], n_sim=200, n_boot=200)
check("es_backtest: fewer than 3 violations -> no McNeil-Frey line",
      r["violations"] < 3 and "McNeil-Frey:" not in t and relayed(t, r["warnings"], "  WARNING: "), t)
r, t = run(risk.es_backtest, r_ok, var6, es6, alpha=0.01, n_sim=200, n_boot=200)
check("es_backtest: alpha = 1% -> header at 99.0% and no Acerbi-Szekely zone (calibrated for 2.5%)",
      has(t, "ES backtest at 99.0%") and "zone:" not in t, t)

m_ok = pd.DataFrame({"VaR": var6, "ES": es6})
m_low = pd.DataFrame({"VaR": var6 * 0.7, "ES": es6 * 0.7})
r, t = run(risk.compare_es_models, r_ok, {"low": m_low, "ok": m_ok})
check("compare_es_models: the best is the smallest mean FZ0 and is named; sorted table",
      r.index[0] == r["mean_FZ0"].idxmin() and has(t, f"Best: {r.index[0]}.")
      and has(t, "alpha = 0.025 (250 common days)"), t)

# =============================================================================
# segments.logit_leaf_model
# =============================================================================
n7 = 3000
x7 = rng.normal(size=n7)
b2b = rng.integers(0, 2, n7)
p7 = 1 / (1 + np.exp(-(-0.5 + np.where(b2b == 1, 1.5 * x7, -1.0 * x7) + 1.0 * b2b)))
d7 = pd.DataFrame(dict(y=rng.binomial(1, p7), x1=x7, b2b=b2b, t=np.arange(n7)))
r, t = run(segments.logit_leaf_model, d7, "y", ["x1", "b2b"], max_depth=2, min_leaf=200)
check("logit_leaf_model: header with segments, training and holdout rows, honest and random",
      has(t, f"Logit leaf model: {len(r['leaves'])} segments, ") and has(t, "honest: tree and logits on separate halves")
      and has(t, f"{len(r['test_index'])} holdout rows (random)."), t)
check("logit_leaf_model: honest -> no 'do not use them for inference' line", "do not use them for inference" not in t, t)
check("logit_leaf_model: the stability reminder is printed", has(t, "leaf_model_stability"), t)
r, t = run(segments.logit_leaf_model, d7, "y", ["x1", "b2b"], max_depth=2, min_leaf=200, honest=False,
           time_col="t", benchmarks={"inter": "y ~ x1*b2b"})
check("logit_leaf_model: honest=False -> same-rows label and the inference warning",
      has(t, "tree and logits on the same rows") and has(t, "do not use them for inference"), t)
check("logit_leaf_model: time holdout -> 'most recent: t >= <first holdout t>'",
      has(t, f"(most recent: t >= {int(d7.loc[r['test_index'], 't'].min())})"), t)
check("logit_leaf_model: every warning relayed; a better benchmark is named in a warning",
      relayed(t, r["warnings"], "  WARNING: ") and ((r["best"] == "logit_leaf")
                                                     or has(t, f"({r['best']} does): report {r['best']}")), t)
r, t = run(segments.logit_leaf_model, d7.assign(region=rng.integers(0, 3, n7).astype(float)), "y",
           ["x1", "b2b", "region"], max_depth=2, min_leaf=200, segment_by=["region"])
check("logit_leaf_model: no split on segment_by -> warning naming it",
      "region" not in r["split_variables"] and has(t, "No split uses ['region']"), t)

# =============================================================================
# causal: rdd, event_study, did_imputation, did_compare
# =============================================================================
n8 = 1500
xr = rng.uniform(-1, 1, n8)
dr = pd.DataFrame(dict(x=xr, y=1 + 0.5 * xr + 0.8 * (xr >= 0) + rng.normal(scale=0.3, size=n8)))
dr["d"] = (((xr >= 0) & (rng.uniform(size=n8) < 0.8)) | ((xr < 0) & (rng.uniform(size=n8) < 0.2))).astype(float)
dr["y2"] = 1 + 0.5 * xr + dr["d"] + rng.normal(scale=0.3, size=n8)
r, t = run(causal.rdd, dr, "y", "x", 0.0, h=0.4)
nl, nr = int(((xr > -0.4) & (xr < 0)).sum()), int(((xr >= 0) & (xr < 0.4)).sum())
check("rdd: N left + right counted from the data inside the triangular window",
      r["n_left"] == nl and r["n_right"] == nr and has(t, f"N = {nl} + {nr}"), t)
check("rdd: 95% CI = estimate +- 1.96 SE and printed as returned",
      abs(r["ci_low"] - (r["estimate"] - z975 * r["se"])) < 1e-12
      and has(t, f"{r['estimate']:.4f} (SE {r['se']:.4f}, 95% CI {r['ci_low']:.4f} to {r['ci_high']:.4f})"), t)
check("rdd: user bandwidth labelled '(user)', no warning; the true jump 0.8 inside the CI",
      has(t, "h = 0.4 (user)") and "Warning" not in t and r["ci_low"] < 0.8 < r["ci_high"], t)
check("rdd: cutoff label and the conventional-CI caveat",
      has(t, "RD at x = 0.0:") and has(t, "Conventional CI; for robust bias-corrected inference use rdrobust"), t)
r, t = run(causal.rdd, dr, "y2", "x", 0.0, h=0.4, fuzzy="d")
check("rdd: fuzzy -> first-stage jump and F printed as returned",
      has(t, f"first-stage jump {r['jump_in_treatment']:.3f} (F = {r['F']:.1f})"), t)
r, t = run(causal.rdd, dr, "y", "x", 0.0)
check("rdd: bandwidth chosen by the function -> every warning relayed",
      relayed(t, r["warnings"], "  Warning: "), t)


def staggered(seed, anticip=0.0, ramp=0.0, n_units=90):
    g_ = np.random.default_rng(seed)
    rows_ = []
    for i in range(n_units):
        gg = [4, 6, np.nan][i % 3]
        a_ = g_.normal()
        for tt3 in range(1, 9):
            k = tt3 - gg if gg == gg else np.nan
            te = 0.0
            if gg == gg and k >= 0:
                te = 1.0 + ramp * k
            elif gg == gg and k == -2:
                te = anticip
            rows_.append(dict(u=i, t=tt3, g=gg, y=a_ + 0.2 * tt3 + te + g_.normal(scale=0.5)))
    out_ = pd.DataFrame(rows_)
    out_["g0"] = out_["g"].fillna(0)
    return out_


p0 = staggered(30)
r, t = run(causal.event_study, p0, "u", "t", "y", "g", leads=3, lags=3)
check("event_study: header with ref and cluster; joint pre-test p as returned; no pre-trends -> low-power caveat",
      has(t, "Event study (TWFE, ref k = -1, clustered by u)") and r["pretrend_p"] >= 0.05
      and has(t, f"p = {r['pretrend_p']:.3f} → no evidence of pre-trends (low power is possible)"), t)
check("event_study: no planted anticipation or ramp -> no WARNING and no NOTE", "WARNING" not in t and "NOTE" not in t, t)
r, t = run(causal.event_study, staggered(31, anticip=1.5), "u", "t", "y", "g", leads=3, lags=3)
check("event_study: effect at k = -2 -> pre-trends detected and the anticipation warning names k = -2",
      r["pretrend_p"] < 0.05 and has(t, "pre-trends detected") and has(t, "WARNING: the coefficient at k = -2"), t)
r, t = run(causal.event_study, staggered(32, ramp=0.6), "u", "t", "y", "g", leads=3, lags=3)
check("event_study: effect growing with exposure -> NOTE on rising post coefficients",
      r["flags"]["gradual_effect"] and has(t, "NOTE: post-treatment coefficients keep rising"), t)
r, t = run(causal.event_study, p0, "u", "t", "y", "g", leads=1, lags=2, ref=-1)
check("event_study: no lead to test (leads = 1, ref = -1) -> p is NaN in the result",
      not np.isfinite(r["pretrend_p"]), r["pretrend_p"])
check("event_study: ... and the text says there is no pre-period coefficient to test, never 'no evidence'",
      "no evidence of pre-trends" not in t and "p = nan" not in t and has(t, "no pre-period coefficient to test"), t)

pg = staggered(33)
r, t = run(causal.did_imputation, pg, "u", "t", "y", "g0")
o = r["overall"]
check("did_imputation: overall line with the 95% CI as returned and treated observations counted",
      has(t, f"Overall ATT: {o['estimate']:.4f} (SE {o['se']:.4f}, 95% CI {o['ci_low']:.4f} to {o['ci_high']:.4f}); "
             f"{int(((pg['g0'] > 0) & (pg['t'] >= pg['g0'])).sum())} treated observations"), t)
check("did_imputation: CI = estimate +- z(0.975) SE", abs(o["ci_high"] - (o["estimate"] + z975 * o["se"])) < 1e-12)
pt = r["pretrend"]
check("did_imputation: pre-trend line with leads, W, reference and p as returned",
      pt["leads"] > 0 and has(t, f"Pre-trend test ({pt['leads']} leads, untreated observations only): "
                                 f"W = {pt['stat']:.3f}, {pt['reference']} p = {pt['p']:.3f} → "), t)
check("did_imputation: every warning relayed", relayed(t, r["warnings"], "  WARNING: "), t)
pp = pt["p"]
ra, ta = run(causal.did_imputation, pg, "u", "t", "y", "g0", alpha=min(0.99, pp * 1.001))
rb, tb = run(causal.did_imputation, pg, "u", "t", "y", "g0", alpha=pp * 0.999)
check("did_imputation: alpha just above the pre-trend p -> 'pre-trends detected'; just below -> 'does not reject'",
      has(ta, "→ pre-trends detected") and has(tb, "→ does not reject; this does not show parallel trends"), (ta, tb))
check("did_imputation: the CI label follows alpha", has(ta, f"{100 * (1 - min(0.99, pp * 1.001)):g}% CI"), ta)
r, t = run(causal.did_imputation, pg, "u", "t", "y", "g0", pretrends=0)
check("did_imputation: pretrends = 0 -> no pre-trend line and no 'does not reject'",
      "Pre-trend test" not in t and "does not reject" not in t, t)

r, t = run(causal.did_compare, pg, "u", "t", "y", "g0", n_boot=49)
check("did_compare: header with control, common cells and draws as returned",
      has(t, f"Callaway-Sant'Anna (notyet controls) vs imputation (BJS), {r['cells_common']} common post cells, "
             f"{r['draws_used']} paired bootstrap draws"), t)
check("did_compare: estimates, difference, SE and p printed as returned",
      has(t, f"CS {r['cs']:.4f} (SE {r['se_cs']:.4f}); BJS {r['bjs']:.4f} (SE {r['se_bjs']:.4f}); difference "
             f"{r['difference']:.4f} (SE {r['diff_se']:.4f}, p = {r['p']:.3f})"), t)
pc = r["p"]
ra, ta = run(causal.did_compare, pg, "u", "t", "y", "g0", n_boot=49, alpha=min(0.99, pc * 1.001))
rb, tb = run(causal.did_compare, pg, "u", "t", "y", "g0", n_boot=49, alpha=pc * 0.999)
check("did_compare: alpha just above the p -> 'DISAGREE' and the divergence warning; just below -> no difference",
      ra["diverge"] and has(ta, f"the routes DISAGREE at alpha = {min(0.99, pc * 1.001):g}")
      and relayed(ta, ra["warnings"], "  WARNING: ") and not rb["diverge"]
      and has(tb, f"no significant difference at alpha = {pc * 0.999:g}") and "DISAGREE" not in tb, (ta, tb))
r, t = run(causal.did_compare, pg, "u", "t", "y", "g0", n_boot=49, cluster="u")
check("did_compare: cluster -> 'draws by u' in the header", has(t, "paired bootstrap draws by u"), t)

passed = sum(ok)
print(f"\n{passed}/{len(ok)} checks passed")
sys.exit(0 if passed == len(ok) else 1)
