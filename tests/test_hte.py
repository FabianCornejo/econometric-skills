"""Tests for hte.py (v2.0): CATE estimation, validation (BLP, GATES, CLAN, TOC, R-score) and
policy learning, on processes with a known treatment effect function tau(x).

Checks: recovery of a linear tau(x) by the DR-learner and the R-learner (binary and continuous
treatment); CI coverage of the ATE and size of the heterogeneity test by Monte Carlo;
heterogeneity detected when present and not when absent; flexible learners on a step-shaped
tau(x); a policy tree that recovers the rule "treat if x1 > 0" and beats treating everyone.

Run from the repository root:  python3 tests/test_hte.py
Needs: numpy, pandas, scipy, scikit-learn.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "econometrics" / "scripts"))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

import hte

ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


LIN = dict(outcome_learner=LinearRegression(), ps_learner=LogisticRegression(C=1e6, max_iter=2000))


def sim_binary(n, seed, tau=lambda d: 1 + d["x1"]):
    r = np.random.default_rng(seed)
    d = pd.DataFrame({f"x{j}": r.normal(size=n) for j in range(1, 5)})
    d["w1"] = r.normal(size=n)
    e = 1 / (1 + np.exp(-(0.6 * d["x2"] + 0.4 * d["w1"])))
    d["t"] = (r.uniform(size=n) < e).astype(int)
    d["tau_true"] = tau(d)
    d["y"] = d["x2"] + d["w1"] + 0.5 * d["x3"] + d["tau_true"] * d["t"] + r.normal(size=n)
    return d


# ---------------------------------------------------------------------------------------------
# Linear tau(x) = 1 + x1, binary treatment, confounding through x2 and w1
# ---------------------------------------------------------------------------------------------
d = sim_binary(4000, 1)
X = ["x1", "x2", "x3", "x4"]
dr = hte.cate_fit(d, "y", "t", X, W=["w1"], method="dr", **LIN)
ld = hte.cate_fit(d, "y", "t", X, W=["w1"], method="linear_dml", **LIN)
for name, r in [("DR-learner", dr), ("linear DML", ld)]:
    c = r["coef"]
    check(f"{name}: ATE CI contains 1", r["ate"]["ci_low"] < 1 < r["ate"]["ci_high"], f"{r['ate']['estimate']:.3f}")
    check(f"{name}: coefficient on x1 CI contains 1 and excludes 0",
          c.loc["x1", "ci_low"] < 1 < c.loc["x1", "ci_high"] and c.loc["x1", "ci_low"] > 0, f"{c.loc['x1', 'estimate']:.3f}")
    check(f"{name}: coefficients on x3, x4 (no effect modification) CIs contain 0",
          all(c.loc[v, "ci_low"] < 0 < c.loc[v, "ci_high"] for v in ("x3", "x4")))

pred = hte.cate_predict(dr, pd.DataFrame({"x1": [-1.0, 0.0, 1.0], "x2": 0.0, "x3": 0.0, "x4": 0.0, "w1": 0.0}))
check("cate_predict: CIs cover tau at x1 = -1, 0, 1",
      all(pred["ci_low"].iloc[i] < v < pred["ci_high"].iloc[i] for i, v in enumerate([0.0, 1.0, 2.0])),
      pred["tau"].round(2).tolist())
check("out-of-fold tau correlates with the truth (> 0.95)", np.corrcoef(dr["tau"], d["tau_true"])[0, 1] > 0.95)

b = hte.blp_test(dr)
bt = b["table"]
check("BLP: heterogeneity detected (one-sided p < 0.001)", b["p_heterogeneity"] < 0.001, b["verdict"])
check("BLP: mean and differential predictions calibrated (CIs contain 1)",
      bt.loc["mean_prediction", "ci_low"] < 1 < bt.loc["mean_prediction", "ci_high"]
      and bt.loc["differential_prediction", "ci_low"] < 1 < bt.loc["differential_prediction", "ci_high"],
      bt["estimate"].round(3).to_dict())

ga = hte.gates(dr, n_groups=5)
gt = ga["table"]
check("GATES increase monotonically across quintiles", gt["estimate"].is_monotonic_increasing,
      gt["estimate"].round(2).tolist())
check("GATES top - bottom > 2 and significant", ga["top_minus_bottom"]["estimate"] > 2 and ga["top_minus_bottom"]["p"] < 1e-6,
      f"{ga['top_minus_bottom']['estimate']:.2f}")
true_g = d.groupby(ga["groups"])["tau_true"].mean().to_numpy()
check("each GATE CI covers the true mean tau of its group",
      all(gt["ci_low"].iloc[k] < true_g[k] < gt["ci_high"].iloc[k] for k in range(5)))

cl = hte.clan(dr, d, ["x1", "x3", "x4"])
check("CLAN: most affected have higher x1 (Holm p < 0.001); x3, x4 do not differ (Holm p > 0.05)",
      cl.loc["x1", "difference"] > 0 and cl.loc["x1", "p_holm"] < 0.001
      and cl.loc["x3", "p_holm"] > 0.05 and cl.loc["x4", "p_holm"] > 0.05)

tc = hte.toc(dr, B=200)
check("TOC: AUTOC and Qini positive and significant", tc["autoc"]["estimate"] > 0 and tc["autoc"]["p"] < 0.001
      and tc["qini"]["p"] < 0.001, f"AUTOC {tc['autoc']['estimate']:.3f}")
check("TOC at q = 1 is zero by construction", abs(tc["table"]["toc"].iloc[-1]) < 1e-12)
check("R-score positive when the effect varies", hte.rscore(dr) > 0.02, f"{hte.rscore(dr):.3f}")

# ---------------------------------------------------------------------------------------------
# Homogeneous effect: no false heterogeneity
# ---------------------------------------------------------------------------------------------
dh = sim_binary(4000, 2, tau=lambda d: np.ones(len(d)))
rh = hte.cate_fit(dh, "y", "t", X, W=["w1"], method="dr", **LIN)
check("homogeneous effect: BLP does not reject (p > 0.05)", hte.blp_test(rh)["p_heterogeneity"] > 0.05,
      f"p = {hte.blp_test(rh)['p_heterogeneity']:.3f}")
check("homogeneous effect: GATES top - bottom not significant", hte.gates(rh)["top_minus_bottom"]["p"] > 0.05)
check("homogeneous effect: R-score ~ 0 (< 0.01)", hte.rscore(rh) < 0.01, f"{hte.rscore(rh):.4f}")

# ---------------------------------------------------------------------------------------------
# Monte Carlo: ATE coverage and size of the heterogeneity test
# ---------------------------------------------------------------------------------------------
cover, rej = [], []
for rep in range(200):
    dm = sim_binary(1000, 100 + rep, tau=lambda d: np.ones(len(d)))
    rm = hte.cate_fit(dm, "y", "t", X, W=["w1"], method="dr", seed=rep, **LIN)
    cover.append(rm["ate"]["ci_low"] < 1 < rm["ate"]["ci_high"])
    rej.append(hte.blp_test(rm)["p_heterogeneity"] < 0.05)
check("Monte Carlo: ATE 95% CI coverage in [0.91, 0.98]", 0.91 <= np.mean(cover) <= 0.98, f"{np.mean(cover):.3f}")
check("Monte Carlo: BLP heterogeneity test size <= 0.09 under a constant effect", np.mean(rej) <= 0.09, f"{np.mean(rej):.3f}")

# ---------------------------------------------------------------------------------------------
# Flexible learners on a symmetric step effect (tau = 3 if |x1| > 1, else 1) and nonlinear confounding
# ---------------------------------------------------------------------------------------------
r = np.random.default_rng(5)
n = 3000
dn = pd.DataFrame({f"x{j}": r.uniform(-2, 2, n) for j in range(1, 5)})
e = 1 / (1 + np.exp(-np.sin(2 * dn["x2"])))
dn["t"] = (r.uniform(size=n) < e).astype(int)
dn["tau_true"] = 1 + 2 * (dn["x1"].abs() > 1)    # no linear trend in x1: a linear CATE cannot see it
dn["y"] = np.sin(2 * dn["x2"]) + dn["x3"] ** 2 + dn["tau_true"] * dn["t"] + r.normal(size=n)
FLEX = dict(outcome_learner=HistGradientBoostingRegressor(max_iter=150, random_state=0),
            ps_learner=HistGradientBoostingClassifier(max_iter=150, random_state=0))
fits = {m: hte.cate_fit(dn, "y", "t", X, method=m, **FLEX) for m in ("forest", "t_learner", "x_learner", "dr")}
for m in ("forest", "t_learner", "x_learner"):
    cor = np.corrcoef(fits[m]["tau"], dn["tau_true"])[0, 1]
    check(f"{m}: out-of-fold tau tracks the step effect (corr > 0.6)", cor > 0.6, f"{cor:.2f}")
check("linear DR-learner: out-of-fold tau uncorrelated with a symmetric effect (|corr| < 0.2)",
      abs(np.corrcoef(fits["dr"]["tau"], dn["tau_true"])[0, 1]) < 0.2)
check("flexible nuisances: ATE CI contains the true ATE", fits["forest"]["ate"]["ci_low"] < dn["tau_true"].mean()
      < fits["forest"]["ate"]["ci_high"], f"{fits['forest']['ate']['estimate']:.3f} vs {dn['tau_true'].mean():.3f}")
check("forest: BLP detects heterogeneity", hte.blp_test(fits["forest"])["p_heterogeneity"] < 0.001)
cmp = hte.compare_cate_models(fits)
check("compare_cate_models: forest beats the linear DR-learner on a symmetric effect (R-score)",
      cmp.loc["forest", "R_score"] > cmp.loc["dr", "R_score"] + 0.03, cmp["R_score"].round(3).to_dict())
check("a linear CATE misses symmetric heterogeneity (BLP p > 0.05) that the forest detects",
      cmp.loc["dr", "BLP_p"] > 0.05 and cmp.loc["forest", "BLP_p"] < 0.001)
check("compare_cate_models: same ATE for every method (cross-fitted AIPW)", cmp["ATE"].nunique() == 1)

# ---------------------------------------------------------------------------------------------
# Policy tree: treat only when x1 > 0 (tau = x1, cost 0)
# ---------------------------------------------------------------------------------------------
dp = sim_binary(6000, 9, tau=lambda d: d["x1"])
rp = hte.cate_fit(dp, "y", "t", X, W=["w1"], method="dr", **LIN)
pt = hte.policy_tree(rp, dp, ["x1", "x3", "x4"], cost=0.0, depth=2, seed=1)
tv = pt["value_test"]
assign = hte.policy_predict(pt["tree"], dp)
agree = np.mean(assign == (dp["x1"] > 0))
check("policy tree recovers 'treat if x1 > 0' (agreement > 90%)", agree > 0.90, f"{agree:.3f}; {pt['rules']}")
check("policy beats treating everyone on held-out units (p < 0.01)",
      tv.loc["policy vs treat everyone", "estimate"] > 0 and tv.loc["policy vs treat everyone", "p"] < 0.01,
      f"gain {tv.loc['policy vs treat everyone', 'estimate']:.3f}")
check("held-out policy value ~ E[max(x1, 0)] = 0.399", abs(tv.loc["policy vs treat no one", "estimate"] - 0.399) < 0.1,
      f"{tv.loc['policy vs treat no one', 'estimate']:.3f}")
pc = hte.policy_tree(rp, dp, ["x1"], cost=10.0, depth=1, seed=1)
check("prohibitive cost: the rule treats no one", pc["share_treated"] == 0.0, pc["rules"])

# ---------------------------------------------------------------------------------------------
# Continuous treatment: dY/dT = 1 + x1 (linear DML)
# ---------------------------------------------------------------------------------------------
r = np.random.default_rng(13)
n = 4000
dc = pd.DataFrame({f"x{j}": r.normal(size=n) for j in range(1, 4)})
dc["t"] = 0.5 * dc["x2"] + r.normal(size=n)
dc["y"] = dc["x2"] + (1 + dc["x1"]) * dc["t"] + r.normal(size=n)
rc = hte.cate_fit(dc, "y", "t", ["x1", "x2", "x3"], method="linear_dml",
                  outcome_learner=LinearRegression(), ps_learner=LinearRegression())
check("continuous treatment: x1 coefficient CI contains 1", rc["coef"].loc["x1", "ci_low"] < 1 < rc["coef"].loc["x1", "ci_high"],
      f"{rc['coef'].loc['x1', 'estimate']:.3f}")
ame_true = 1 + dc["x1"].mean()                      # sample average of dY/dT
check("continuous treatment: average marginal effect CI contains the sample AME",
      rc["ate"]["ci_low"] < ame_true < rc["ate"]["ci_high"], f"{rc['ate']['estimate']:.3f} vs {ame_true:.3f}")
check("continuous treatment: GATES increasing", hte.gates(rc)["table"]["estimate"].is_monotonic_increasing)
try:
    hte.cate_fit(dc, "y", "t", ["x1"], method="x_learner")
    check("meta-learner with a continuous treatment raises", False)
except ValueError:
    check("meta-learner with a continuous treatment raises", True)

# ---------------------------------------------------------------------------------------------
# Weak overlap is flagged
# ---------------------------------------------------------------------------------------------
dw = sim_binary(3000, 21)
dw["t"] = (dw["x2"] > 0).astype(int) | (np.random.default_rng(3).uniform(size=3000) < 0.01)
with warnings.catch_warnings(record=True) as wl:
    warnings.simplefilter("always")
    rw = hte.cate_fit(dw, "y", "t", X, W=["w1"], method="dr", **LIN)
check("weak overlap produces a warning", any("overlap" in str(w.message) for w in wl),
      f"share clipped {rw['overlap']['share_clipped']:.2f}")

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
