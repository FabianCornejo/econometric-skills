"""Tests for v2.6.2: the SE label of sensitivity numbers in reports, k_needed beyond the cap, and DML bounds for a
confounder k times as strong as an observed covariate.

Origin: the v2.6.1 validation on notebook 1 (a memo mixed the HC3 CI with classical-SE thresholds; the agent built the
"k times as strong" DML bounds by hand, scaling cf_y and cf_d by k as the dml_benchmark docstring said), the note of
reviewer A1 of v2.6.1 (k_needed can exceed k_cap: seed 29, x: 7965 > 7936), and the research of v2.6.2
(claude/fonte-dml-benchmark-k.md in the dev repository): Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis (2022,
Appendix F) and DoubleML 0.11.4 benchmark with *gains*, G = R/(1 - R), and scale them by k; dml_benchmark returned the
partial R² R = G/(1 + G), which made "as strong as X" less conservative than the paper.

What is checked, each by a route other than the implementation's:
  1. ovb_bounds and sensitivity after a robust fit print one report sentence (label the classical-SE numbers, show
     the robust CI separately); a classical fit prints none; HC0, HC3, HAC and cluster all print it;
  2. ovb_bounds k_needed_beyond_cap: seed 29 (x and w) and 840 (x, k_needed about 1) of the v2.6.1 search design;
     the boundary built through q: q* puts k_needed exactly at k_cap (bias at the cap with R²_Y = 1, closed form), and
     q*(1 -+ 1e-6) fall on each side (flag, sentence); a benchmark that explains nothing: False; below the cap the
     v2.6.1 sentence is unchanged;
  3. dml_benchmark returns the paper's gains:
     - against population gains from the covariance matrix of a Gaussian linear design (Schur complements, not the
       function's formula), n = 40000, linear learners, 3% relative;
     - against DoubleML 0.11.4 sensitivity_benchmark (tests/data/doubleml_benchmark_v262.json), 2% relative;
     - the old partial R² survive as r2_y and r2_d, equal to G/(1 + G);
  4. dml_bounds:
     - planted confounder U with known strength k* relative to x1 (population): at (kd, ky) = k* and rho = 1 the near
       end of the bounds is theta_true (the bound is sharp with one linear confounder), within 0.03; at k*/2 it is not;
     - boundaries on a hand-built benchmark (gain_d = gain_y = 0.4, so k_d_max = 2.5): kd = 2.5(1 -+ 1e-9) accepted /
       refused; default kd skips 3 with a warning; ky = 2.5(1 -+ 1e-9) not capped / capped;
     - rho: the benchmark's by default (bias bound proportional to |rho|), a number in [-1, 1], 1.2 refused; a
       benchmark with NaN rho or no strength is skipped with a warning;
     - k = 1 equals dml_sensitivity called by hand with the same cf_y, cf_d and rho (implementation check);
     - the text: one line per row with its numbers, "include zero" / "exclude zero" on both sides.

Run from the repository root:  python3 tests/test_v262.py
Needs: statsmodels, scipy, scikit-learn.
"""
import contextlib
import io
import json
import re
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "econometrics" / "scripts"))
import _console  # noqa: E402,F401  (Windows-safe console output, see econometrics/scripts/_console.py)
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression

import causal
import specification as sp

ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


def run(fn, *a, **k):
    """(result, stdout, warning messages); an exception is returned as the result."""
    buf = io.StringIO()
    with warnings.catch_warnings(record=True) as w, contextlib.redirect_stdout(buf):
        warnings.simplefilter("always")
        try:
            out = fn(*a, **k)
        except Exception as e:                    # noqa: BLE001
            out = e
    return out, buf.getvalue(), [str(x.message) for x in w]


LIN = dict(learner_y=LinearRegression(), learner_d=LinearRegression())
REPORT = "In a report, label these numbers as classical-SE"

# --- 1. the report sentence after a robust fit ------------------------------------------------------------------
rng = np.random.default_rng(6)
n = 300
d_, x_, g_ = rng.normal(size=n), rng.normal(size=n), rng.integers(0, 30, size=n)
y_ = 0.3 * d_ + x_ + rng.normal(size=n) * np.exp(-0.5 * np.abs(d_))
D1 = pd.DataFrame(dict(y=y_, d=d_, x=x_, g=g_))
fits = {"nonrobust": smf.ols("y ~ d + x", D1).fit(),
        "HC0": smf.ols("y ~ d + x", D1).fit(cov_type="HC0"),
        "HC3": smf.ols("y ~ d + x", D1).fit(cov_type="HC3"),
        "HAC": smf.ols("y ~ d + x", D1).fit(cov_type="HAC", cov_kwds={"maxlags": 2}),
        "cluster": smf.ols("y ~ d + x", D1).fit(cov_type="cluster", cov_kwds={"groups": D1["g"]})}
for ct, f in fits.items():
    _, t_ob, _ = run(sp.ovb_bounds, f, "d", "x", show=True)
    _, t_se, _ = run(sp.sensitivity, f, "d", show=True)
    want = 0 if ct == "nonrobust" else 1
    check(f"report sentence, {ct}: ovb_bounds prints it {want} time(s), sensitivity {want} time(s)",
          t_ob.count(REPORT) == want and t_se.count(REPORT) == want, (t_ob.count(REPORT), t_se.count(REPORT)))
_, t_ob, _ = run(sp.ovb_bounds, fits["HC3"], "d", "x", show=True)
line = [s for s in t_ob.splitlines() if REPORT in s]
check("report sentence names the robust CI to show separately and the cov_type",
      len(line) == 1 and "HC3" in line[0] and "separately" in line[0], line)
class _Unusable:                     # a robust fit whose SE is infinite (observations with leverage 1)
    def __init__(self, base):
        self.model, self.params, self.cov_type, self.df_resid = base.model, base.params, "HC3", base.df_resid
        self.bse = base.bse * np.inf
        self.tvalues = base.params / self.bse


for fn, args in ((sp.sensitivity, ("d",)), (sp.ovb_bounds, ("d", "x"))):
    _, tu, _ = run(fn, _Unusable(fits["nonrobust"]), *args, show=True)
    lu = [s_ for s_ in tu.splitlines() if REPORT in s_]
    check(f"report line, {fn.__name__} with an unusable robust SE: says so and never 'show your HC3 CI' (review)",
          len(lu) == 1 and "not usable" in lu[0] and "show your" not in lu[0], lu)
_, t_q, _ = run(sp.ovb_bounds, fits["HC3"], "d", "x", show=False)
check("report sentence: nothing printed with show=False", t_q == "", repr(t_q[:80]))


# --- 2. k_needed beyond the cap ---------------------------------------------------------------------------------
def design(seed):                 # the v2.6.1 search design (claude/scripts/busca_sementes_k_not_significant_v261.py)
    rng_ = np.random.default_rng(seed)
    n_ = int(rng_.integers(30, 600)); a, b, c = rng_.normal(size=3) * rng_.choice([0.3, 1, 3], 3)
    x = rng_.normal(size=n_); w = rng_.normal(size=n_)
    d = a * x + rng_.normal(size=n_); y = rng_.normal() * d + b * x + c * w + rng_.normal(size=n_) * rng_.choice([0.5, 1, 3])
    return pd.DataFrame(dict(y=y, d=d, x=x, w=w))


BEYOND = "only with the outcome fully explained"
for seed, bms in ((29, ("x", "w")), (840, ("x",))):
    m = smf.ols("y ~ d + x + w", design(seed)).fit()
    o, t, _ = run(sp.ovb_bounds, m, "d", list(bms), show=True)
    br = o["benchmark_r2"]
    for b in bms:
        kn, kc = br.loc[b, "k_needed"], br.loc[b, "k_cap"]
        check(f"beyond cap, seed {seed} {b}: k_needed {kn:.6g} > k_cap {kc:.6g} and k_needed_beyond_cap True",
              kn > kc and br.loc[b, "k_cap_by"] == "outcome" and bool(br.loc[b, "k_needed_beyond_cap"]))
        lines = [s for s in t.splitlines() if f"as strong as {b} in explaining the treatment" in s and BEYOND in s]
        check(f"beyond cap, seed {seed} {b}: the sentence says the outcome is fully explained, names k_cap, and never "
              f"'same k for treatment and outcome' for k_needed",
              len(lines) == 1 and f"{kc:.4g}x" in lines[0] and f"{kn:.3g}x" in lines[0]
              and not any(f"{kn:.3g}x as strong as {b} (same k for treatment and outcome) could, in the worst case, "
                          f"bring" in s for s in t.splitlines()), lines)
m29 = smf.ols("y ~ d + x + w", design(29)).fit()
o29 = sp.ovb_bounds(m29, "d", "x")
br = o29["benchmark_r2"].loc["x"]
est, se, df = o29["estimate"], o29["se"], o29["df"]
r2dz_cap = br["k_cap"] * br["r2dxj_x"] / (1 - br["r2dxj_x"])
bias_cap = np.sqrt(r2dz_cap / (1 - r2dz_cap)) * se * np.sqrt(df)        # R²_Y = 1 at the cap (eq. 12 of the paper)
qstar = bias_cap / abs(est)
for side, fac in (("below", 1 - 1e-6), ("above", 1 + 1e-6)):
    ob, tb, _ = run(sp.ovb_bounds, m29, "d", "x", q=qstar * fac, show=True)
    r = ob["benchmark_r2"].loc["x"]
    beyond = side == "above"
    check(f"boundary through q, {side} the cap: k_needed {'>' if beyond else '<'} k_cap, flag {beyond}, beyond "
          f"sentence {'printed' if beyond else 'absent'}",
          ((r["k_needed"] > r["k_cap"]) == beyond) and bool(r["k_needed_beyond_cap"]) == beyond
          and (BEYOND in tb) == beyond and abs(r["k_needed"] / r["k_cap"] - 1) < 1e-4,
          (r["k_needed"], r["k_cap"]))
    if not beyond:
        check("boundary through q, below: the v2.6.1 sentence, unchanged",
              f"A confounder {r['k_needed']:.3g}x as strong as x (same k for treatment and outcome) could, in the "
              f"worst case, reduce the estimate by" in tb)
m0 = smf.ols("y ~ d + x + w", design(0)).fit()
o0, t0, _ = run(sp.ovb_bounds, m0, "d", ["x", "w"], show=True)
check("below the cap (seed 0, x and w): flag False and no beyond sentence",
      not o0["benchmark_r2"]["k_needed_beyond_cap"].any() and BEYOND not in t0
      and (o0["benchmark_r2"]["k_needed"] < o0["benchmark_r2"]["k_cap"]).all())
rng = np.random.default_rng(1)                    # the R4 design of test_v261: z explains nothing of either side
n_ = 200
x4 = rng.normal(size=n_)
d4 = 0.5 * x4 + rng.normal(size=n_)
y4 = 1.0 * d4 + x4 + rng.normal(size=n_)
A_ = np.column_stack([np.ones(n_), d4, x4, y4])
z4 = rng.normal(size=n_)
z4 = z4 - A_ @ np.linalg.lstsq(A_, z4, rcond=None)[0]
oZ = sp.ovb_bounds(sm.OLS(y4, pd.DataFrame(dict(const=1.0, d=d4, x=x4, z=z4))).fit(), "d", "z")
check("a benchmark that explains nothing (k_needed inf, cap 'kd_max'): k_needed_beyond_cap False",
      np.isinf(oZ["benchmark_r2"].loc["z", "k_needed"]) and not oZ["benchmark_r2"].loc["z", "k_needed_beyond_cap"])


# --- 3. dml_benchmark returns gains -----------------------------------------------------------------------------
def gaussian_design(a, b, theta, sd_d=1.0, sd_y=1.0):
    """Covariance of (y, d, x1, x2, u) for d = a·(x1, x2, u) + sd_d·e_d, y = theta·d + b·(x1, x2, u) + sd_y·e_y."""
    M = np.zeros((5, 5))                          # rows: y, d, x1, x2, u; columns: x1, x2, u, e_d, e_y
    M[2, 0] = M[3, 1] = M[4, 2] = 1.0
    M[1, :3], M[1, 3] = a, sd_d
    M[0] = theta * M[1]
    M[0, :3] += b
    M[0, 4] = sd_y
    return M @ M.T


IDX = {"y": 0, "d": 1, "x1": 2, "x2": 3, "u": 4}


def cvar(S, t, given):
    """Var(t | given) for a Gaussian vector (Schur complement)."""
    i, g = IDX[t], [IDX[v] for v in given]
    return float(S[i, i] - S[i, g] @ np.linalg.solve(S[np.ix_(g, g)], S[g, i]))


A_COEF, B_COEF, THETA = np.array([0.5, 0.3, 0.4]), np.array([0.6, 0.3, 0.5]), 1.0
S_ = gaussian_design(A_COEF, B_COEF, THETA)


def pop_gains(bm, S=S_, X=("x1", "x2")):
    rest = [v for v in X if v != bm]
    gy = cvar(S, "y", ["d", *rest]) / cvar(S, "y", ["d", *X]) - 1        # (sigma2_short - sigma2_long)/sigma2_long
    gd = cvar(S, "d", rest) / cvar(S, "d", list(X)) - 1                  # nu2 = 1/Var(D|X): (nu2_l - nu2_s)/nu2_s
    return gy, gd


rng = np.random.default_rng(20261112)
n = 40000
sh = rng.normal(size=(5, n))
xs = sh[:3]
d = A_COEF @ xs + sh[3]
y = THETA * d + B_COEF @ xs + sh[4]
DG = pd.DataFrame(dict(y=y, d=d, x1=xs[0], x2=xs[1], u=xs[2]))     # u is the omitted confounder
RG = causal.dml_plr(DG, "y", "d", ["x1", "x2"], n_rep=2, **LIN)
BG, _, wBG = run(causal.dml_benchmark, DG, "y", "d", ["x1", "x2"], ["x1", "x2"], res=RG, n_rep=2, **LIN)
for b in ("x1", "x2"):
    gy, gd = pop_gains(b)
    check(f"dml_benchmark {b}: gain_y and gain_d within 3% of the population gains (covariance algebra)",
          abs(BG.loc[b, "gain_y"] / gy - 1) < 0.03 and abs(BG.loc[b, "gain_d"] / gd - 1) < 0.03,
          (BG.loc[b, "gain_y"], gy, BG.loc[b, "gain_d"], gd))
    check(f"dml_benchmark {b}: cf_y = min(gain_y, 1), cf_d = min(gain_d, 1), k_d_max = 1/gain_d",
          BG.loc[b, "cf_y"] == min(BG.loc[b, "gain_y"], 1.0) and BG.loc[b, "cf_d"] == min(BG.loc[b, "gain_d"], 1.0)
          and abs(BG.loc[b, "k_d_max"] * BG.loc[b, "gain_d"] - 1) < 1e-12)
    check(f"dml_benchmark {b}: the old partial R² kept as r2_y, r2_d, smaller than the gains",
          BG.loc[b, "r2_y"] < BG.loc[b, "gain_y"] and BG.loc[b, "r2_d"] < BG.loc[b, "gain_d"])
RG1 = causal.dml_plr(DG, "y", "d", ["x1", "x2"], n_rep=1, **LIN)
BG1, _, _ = run(causal.dml_benchmark, DG, "y", "d", ["x1", "x2"], ["x1"], res=RG1, n_rep=1, **LIN)
check("one sample split: r2_y = G/(1 + G) and r2_d = G/(1 + G) exactly (the median is the value)",
      abs(BG1.loc["x1", "r2_y"] - BG1.loc["x1", "gain_y"] / (1 + BG1.loc["x1", "gain_y"])) < 1e-12
      and abs(BG1.loc["x1", "r2_d"] - BG1.loc["x1", "gain_d"] / (1 + BG1.loc["x1", "gain_d"])) < 1e-12)
th_s = float(np.linalg.solve(S_[np.ix_([1, 2, 3], [1, 2, 3])], S_[[1, 2, 3], 0])[0])
cfy_u = 1 - cvar(S_, "y", ["d", "x1", "x2", "u"]) / cvar(S_, "y", ["d", "x1", "x2"])
cfd_u = 1 - cvar(S_, "d", ["x1", "x2", "u"]) / cvar(S_, "d", ["x1", "x2"])
S_pop = np.sqrt(cvar(S_, "y", ["d", "x1", "x2"]) / cvar(S_, "d", ["x1", "x2"]))
check("population route sanity: one linear confounder, |theta_s - theta| = sqrt(cf_y cf_d/(1 - cf_d))·S (sharp)",
      abs(abs(th_s - THETA) - np.sqrt(cfy_u * cfd_u / (1 - cfd_u)) * S_pop) < 1e-12, (th_s, cfy_u, cfd_u))

# review round 1: DoubleML's aggregation (gains and rho per sample split, then the median) and an interior rho,
# recomputed here from the residuals of dml_plr, as doubleml/utils/gain_statistics.py does (not with the skill's code)
from sklearn.ensemble import RandomForestRegressor   # noqa: E402
rng = np.random.default_rng(3)
n = 1500
x1n, x2n, x3n = rng.normal(size=(3, n))
dn = np.sin(2 * x1n) + 0.5 * x2n + 0.3 * x1n * x2n + rng.normal(size=n)
yn = 0.5 * dn + x1n ** 2 + 0.5 * x2n + 0.4 * np.abs(x3n) + rng.normal(size=n)
DN = pd.DataFrame(dict(y=yn, d=dn, x1=x1n, x2=x2n, x3=x3n))
RF = dict(learner_y=RandomForestRegressor(n_estimators=60, min_samples_leaf=10, random_state=0),
          learner_d=RandomForestRegressor(n_estimators=60, min_samples_leaf=10, random_state=0))


oe, _, _ = run(causal.dml_plr, DN, "y", "d", ["x1", "x2", "x3"], n_rep=1, n_folds=3, **RF)
check("dml_plr accepts an unfitted RandomForestRegressor as learner (review: 'learner or default' raised AttributeError)",
      isinstance(oe, dict) and np.isfinite(oe["estimate"]), repr(oe)[:120])


def split_moments(res):
    out = []
    for (uy, vd), th in zip(res["residuals"], res["thetas"]):
        out.append((np.mean((uy - th * vd) ** 2), 1 / np.mean(vd * vd), th))   # sigma2, nu2 = E[alpha^2], theta
    return out


for nrep in (1, 3):
    kwn = dict(n_rep=nrep, n_folds=3, seed=0, **RF)
    RN = causal.dml_plr(DN, "y", "d", ["x1", "x2", "x3"], **kwn)
    BN, _, _ = run(causal.dml_benchmark, DN, "y", "d", ["x1", "x2", "x3"], ["x1", "x2"], res=RN, **kwn)
    lm = split_moments(RN)
    for b in ("x1", "x2"):
        sm_ = split_moments(causal.dml_plr(DN, "y", "d", [v for v in ("x1", "x2", "x3") if v != b], **kwn))
        gy_r = np.array([max(s[0] / l[0] - 1, 0) for s, l in zip(sm_, lm)])
        gd_r = np.array([max(l[1] / s[1] - 1, 0) for s, l in zip(sm_, lm)])
        dth = np.array([s[2] - l[2] for s, l in zip(sm_, lm)])
        den = np.sqrt(np.array([(s[0] - l[0]) * (l[1] - s[1]) for s, l in zip(sm_, lm)]).clip(0))
        rho_r = np.sign(dth) * np.where(den > 0, np.minimum(np.abs(dth) / np.where(den > 0, den, 1), 1), 1)
        ok_ = (abs(BN.loc[b, "gain_y"] - np.median(gy_r)) < 1e-12 and abs(BN.loc[b, "gain_d"] - np.median(gd_r)) < 1e-12
               and abs(BN.loc[b, "cf_y"] - np.median(np.minimum(gy_r, 1))) < 1e-12
               and abs(BN.loc[b, "rho"] - np.median(rho_r)) < 1e-12)
        check(f"DoubleML aggregation, random forests, n_rep = {nrep}, {b}: gains, cf_y and rho = medians of the split "
              f"values", ok_, (BN.loc[b, ["gain_y", "gain_d", "cf_y", "rho"]].tolist(), np.median(gy_r),
                               np.median(gd_r), np.median(rho_r)))
        if nrep == 1 and b == "x1":
            check("interior rho (forests, one split): 0 < |rho| < 1, equal to dth/sqrt(dsigma2·dnu2) of the paper",
                  0 < abs(BN.loc[b, "rho"]) < 1 and abs(BN.loc[b, "rho"] - dth[0] / den[0]) < 1e-12,
                  BN.loc[b, "rho"])

# a negative rho: omitting x1 pulls theta down (d rises with x1, y falls with it)
rng = np.random.default_rng(8)
n = 5000
x1m, x2m = rng.normal(size=(2, n))
dm = 0.8 * x1m + 0.4 * x2m + rng.normal(size=n)
ym = 1.0 * dm - 1.5 * x1m + 0.5 * x2m + rng.normal(size=n)
DM = pd.DataFrame(dict(y=ym, d=dm, x1=x1m, x2=x2m))
RM = causal.dml_plr(DM, "y", "d", ["x1", "x2"], n_rep=1, **LIN)
BM, _, _ = run(causal.dml_benchmark, DM, "y", "d", ["x1", "x2"], ["x1"], res=RM, n_rep=1, **LIN)
check("a benchmark whose omission lowers theta: delta_theta < 0 and rho = -1 (the sign is kept; mutation)",
      BM.loc["x1", "delta_theta"] < 0 and BM.loc["x1", "rho"] == -1.0, BM.loc["x1", ["delta_theta", "rho"]].tolist())
# a split whose refit leaves the residual variances unchanged (denominator 0): DoubleML sets +-1 there
rng = np.random.default_rng(9)
n = 400
splits_long = [(rng.normal(size=n), rng.normal(size=n)) for _ in range(3)]
splits_short = [splits_long[0]] + [(uy + 0.3 * rng.normal(size=n), vd * 1.2) for uy, vd in splits_long[1:]]
FAKE_LONG = {"residuals": splits_long, "thetas": [0.10, 0.12, 0.11], "estimate": 0.11, "se": 0.05}
FAKE_SHORT = {"residuals": splits_short, "thetas": [0.30, 0.20, 0.25], "estimate": 0.25, "se": 0.05}
_orig_plr = causal.dml_plr
causal.dml_plr = lambda *a, **k: FAKE_SHORT
try:
    BF, _, _ = run(causal.dml_benchmark, None, "y", "d", ["x1", "x2"], ["x1"], res=FAKE_LONG)
finally:
    causal.dml_plr = _orig_plr


def mom(uy, vd):
    th = np.sum(vd * uy) / np.sum(vd * vd)
    return np.mean((uy - th * vd) ** 2), 1 / np.mean(vd * vd)


rr = []
for (ul, vl), (us, vs), tl, ts in zip(splits_long, splits_short, FAKE_LONG["thetas"], FAKE_SHORT["thetas"]):
    (s2l_, n2l_), (s2s_, n2s_) = mom(ul, vl), mom(us, vs)
    den_ = np.sqrt(max(s2s_ - s2l_, 0) * max(n2l_ - n2s_, 0))
    rr.append(np.sign(ts - tl) * (min(abs(ts - tl) / den_, 1) if den_ > 0 else 1.0))
check("a split with denominator 0 counts as sign(dth)·1, as in DoubleML (the median stays finite; mutation)",
      np.isfinite(BF.loc["x1", "rho"]) and abs(BF.loc["x1", "rho"] - np.median(rr)) < 1e-12 and rr[0] == 1.0,
      (BF.loc["x1", "rho"], rr))

ref = json.loads((HERE / "data" / "doubleml_benchmark_v262.json").read_text())
rng = np.random.default_rng(20261111)
n = 20000
x1, x2, x3 = rng.normal(size=(3, n))
d = 0.8 * x1 + 0.5 * x2 + rng.normal(size=n)
y = 1.0 * d + 1.5 * x1 + 0.7 * x2 + 0.5 * x3 + rng.normal(size=n)
DR = pd.DataFrame(dict(y=y, d=d, x1=x1, x2=x2, x3=x3))
RR = causal.dml_plr(DR, "y", "d", ["x1", "x2", "x3"], n_folds=5, n_rep=3, seed=0, **LIN)
BR, _, _ = run(causal.dml_benchmark, DR, "y", "d", ["x1", "x2", "x3"], ["x1", "x2", "x3"], res=RR, n_folds=5,
               n_rep=3, seed=0, **LIN)
for b in ("x1", "x2"):
    rb = ref[b]
    check(f"DoubleML 0.11.4, {b}: cf_y, cf_d within 2%, rho equal, delta_theta within 0.01 (different splits)",
          abs(BR.loc[b, "cf_y"] / rb["cf_y"] - 1) < 0.02 and abs(BR.loc[b, "cf_d"] / rb["cf_d"] - 1) < 0.02
          and abs(BR.loc[b, "rho"] - rb["rho"]) < 1e-9 and abs(BR.loc[b, "delta_theta"] - rb["delta_theta"]) < 0.01,
          (BR.loc[b, ["cf_y", "cf_d", "rho", "delta_theta"]].tolist(), rb))
check("DoubleML 0.11.4, x1: the gain in the outcome is above 1 and cut to 1 in both (gain_y > 1, cf_y = 1)",
      BR.loc["x1", "gain_y"] > 1 and BR.loc["x1", "cf_y"] == 1.0 == ref["x1"]["cf_y"], BR.loc["x1", "gain_y"])
check("DoubleML 0.11.4, x3: cf_y within 3%", abs(BR.loc["x3", "cf_y"] / ref["x3"]["cf_y"] - 1) < 0.03,
      (BR.loc["x3", "cf_y"], ref["x3"]["cf_y"]))
_, _, wR = run(causal.dml_benchmark, DR, "y", "d", ["x1", "x2", "x3"], ["x1"], res=RR, n_folds=5, n_rep=3, seed=0,
               **LIN)
check("dml_benchmark warns when a gain is cut to 1 (x1: gain_y > 1), naming the benchmark",
      any("x1" in s and "cut to 1" in s for s in wR), wR)
rng = np.random.default_rng(5)                    # x1 almost determines d: the partial R² with d is above 0.999
n = 2000
x1c, x2c = rng.normal(size=(2, n))
dc = x1c + 0.01 * rng.normal(size=n)
yc = 0.5 * dc + x1c + x2c + rng.normal(size=n)
DC = pd.DataFrame(dict(y=yc, d=dc, x1=x1c, x2=x2c))
RC = causal.dml_plr(DC, "y", "d", ["x1", "x2"], n_rep=1, **LIN)
BC, _, wC = run(causal.dml_benchmark, DC, "y", "d", ["x1", "x2"], ["x1"], res=RC, n_rep=1, **LIN)
check("dml_benchmark: no silent 0.999 cut any more: r2_d above 0.999, gain_d above 1, cf_d = 1, and a warning that "
      "the bound is infinite and names k_d_max",
      BC.loc["x1", "r2_d"] > 0.999 and BC.loc["x1", "gain_d"] > 1 and BC.loc["x1", "cf_d"] == 1.0
      and any("x1" in s_ and "infinite" in s_ and "k_d_max" in s_ for s_ in wC),
      (BC.loc["x1", ["r2_d", "gain_d", "cf_d"]].tolist(), wC))

# --- 4. dml_bounds ------------------------------------------------------------------------------------------------
gy1, gd1 = pop_gains("x1")
k_y_star, k_d_star = cfy_u / gy1, cfd_u / gd1
o, t, w = run(causal.dml_bounds, RG, BG.loc[["x1"]], kd=k_d_star, ky=k_y_star, rho=1.0, show=True)
row = o["bounds"].iloc[0]
near = row["theta_lower"] if th_s > THETA else row["theta_upper"]
check(f"planted confounder: at (kd, ky) = ({k_d_star:.3f}, {k_y_star:.3f}) and rho = 1 the near end of the bounds is "
      f"theta_true = 1 within 0.03 (sharp bound)", abs(near - THETA) < 0.03, (near, row["theta_lower"],
                                                                            row["theta_upper"]))
check("planted confounder: theta_true inside the confidence bounds at k*",
      row["ci_low"] <= THETA <= row["ci_high"], (row["ci_low"], row["ci_high"]))
oh = causal.dml_bounds(RG, BG.loc[["x1"]], kd=k_d_star / 2, ky=k_y_star / 2, rho=1.0, show=False)["bounds"].iloc[0]
check("planted confounder: at k*/2 theta_true is outside the bounds (the k matters)",
      not (oh["theta_lower"] <= THETA <= oh["theta_upper"]), (oh["theta_lower"], oh["theta_upper"]))
# implementation check: k = 1 equals dml_sensitivity by hand
o1 = causal.dml_bounds(RG, BG.loc[["x2"]], kd=1, rho="benchmark", show=False)["bounds"].iloc[0]
ds = causal.dml_sensitivity(RG, cf_y=BG.loc["x2", "cf_y"], cf_d=BG.loc["x2", "cf_d"], rho=BG.loc["x2", "rho"],
                            show=False)
check("k = 1 with the benchmark's rho equals dml_sensitivity called by hand (bounds and confidence bounds)",
      all(abs(o1[a] - ds[b]) < 1e-12 for a, b in (("theta_lower", "theta_lower"), ("theta_upper", "theta_upper"),
                                                   ("ci_low", "ci_low"), ("ci_high", "ci_high"))))
# hand-built benchmark: gain_d = gain_y = 0.4 -> k_d_max = 2.5
HB = pd.DataFrame({"gain_y": [0.4], "gain_d": [0.4], "rho": [0.5]}, index=["b"])
od, td, wd = run(causal.dml_bounds, RG, HB, show=True)
check("default kd (1, 2, 3) with k_d_max = 2.5: rows for 1 and 2, 3 skipped with a warning naming it and 2.5",
      isinstance(od, dict) and od["bounds"]["kd"].tolist() == [1.0, 2.0]
      and any("kd = 3" in s and "2.5" in s for s in wd), wd)
for fac, good in ((1 - 1e-9, True), (1.0, False), (1 + 1e-9, False)):
    oe, _, _ = run(causal.dml_bounds, RG, HB, kd=2.5 * fac, show=False)
    check(f"explicit kd = 2.5·{fac!r}: {'accepted' if good else 'ValueError naming k_d_max'}",
          (isinstance(oe, dict) and oe["bounds"]["cf_d"].iloc[0] < 1) if good else
          (isinstance(oe, ValueError) and "2.5" in str(oe)), repr(oe)[:160])
for fac, capped in ((1 - 1e-9, False), (1 + 1e-9, True)):
    oc, _, wc = run(causal.dml_bounds, RG, HB, kd=1, ky=2.5 * fac, show=False)
    r = oc["bounds"].iloc[0]
    check(f"ky = 2.5·{fac!r}: cf_y {'= 1, capped, with a warning' if capped else '< 1, not capped, no cap warning'}",
          bool(r["cf_y_capped"]) == capped and ((r["cf_y"] == 1.0) == capped)
          and (any("capped" in s for s in wc) == capped), (r["cf_y"], wc))
HE = pd.DataFrame({"gain_y": [0.5], "gain_d": [0.4], "rho": [1.0]}, index=["b"])      # ky * gain_y = 2 * 0.5 = 1 exactly
oq, _, wq = run(causal.dml_bounds, RG, HE, kd=1, ky=2, show=False)
check("ky * gain_y = 1 exactly: cf_y = 1 is admissible, not capped, no cap warning (mutation: '>=' survived)",
      oq["bounds"].iloc[0]["cf_y"] == 1.0 and not oq["bounds"].iloc[0]["cf_y_capped"] and not any("capped" in s_ for s_ in wq),
      wq)
b_half = causal.dml_bounds(RG, HB, kd=1, rho="benchmark", show=False)["bounds"].iloc[0]["bias_bound"]
b_one = causal.dml_bounds(RG, HB, kd=1, rho=1.0, show=False)["bounds"].iloc[0]["bias_bound"]
b_neg = causal.dml_bounds(RG, HB, kd=1, rho=-1.0, show=False)["bounds"].iloc[0]["bias_bound"]
check("rho: the benchmark's (0.5) by default gives half the bias bound of rho = 1; rho = -1 the same as 1",
      abs(b_half - b_one / 2) < 1e-12 and abs(b_neg - b_one) < 1e-12, (b_half, b_one, b_neg))
oe, _, _ = run(causal.dml_bounds, RG, HB.assign(rho=[1.5]), show=False)
check("a bench rho of 1.5 under rho='benchmark' is a ValueError naming the benchmark (review: it printed rho 1.50)",
      isinstance(oe, ValueError) and "'b'" in str(oe) and "rho" in str(oe), repr(oe))
oe, _, _ = run(causal.dml_bounds, RG, HB.assign(rho=[-1.0]), show=False)
check("a bench rho of -1 is accepted (the boundary)", isinstance(oe, dict) and len(oe["bounds"]) == 3 - 1, repr(oe)[:80])
for bad_ in (True, None, [1.0, 0.5]):
    oe, _, _ = run(causal.dml_bounds, RG, HB, rho=bad_, show=False)
    check(f"rho = {bad_!r} is a TypeError (review: True read as 1, None and lists gave numpy errors)",
          isinstance(oe, TypeError), repr(oe))
H3 = pd.DataFrame({"gain_y": [0.3], "gain_d": [0.3], "rho": [1.0]}, index=["b"])
kd_edge = np.nextafter(1 / 0.3, 0)
oe, _, _ = run(causal.dml_bounds, RG, H3, kd=kd_edge, show=False)
check(f"kd = nextafter(1/0.3, 0): refused when kd·gain_d rounds to 1 or more, accepted only if below 1 "
      f"(kd·gd = {kd_edge * 0.3!r}; review: a 7.8e7 bound)",
      isinstance(oe, ValueError) if kd_edge * 0.3 >= 1 else (isinstance(oe, dict) and oe["bounds"]["cf_d"].iloc[0] < 1),
      repr(oe)[:120])
oe, _, _ = run(causal.dml_bounds, RG, H3, kd=1 / 0.3 * (1 - 1e-6), show=False)
check("kd·gain_d = 1 − 1e-6: accepted", isinstance(oe, dict) and oe["bounds"]["cf_d"].iloc[0] < 1)
HNA = pd.DataFrame({"gain_y": [np.nan, 0.3], "gain_d": [0.2, np.inf], "rho": [1.0, 1.0]}, index=["nan_g", "inf_g"])
on_, _, wna = run(causal.dml_bounds, RG, HNA, kd=1, show=False)
check("non-finite gains: skipped with 'not finite', never as 'no strength' (review)",
      on_["bounds"].empty and sum("not finite" in s_ for s_ in wna) == 2 and not any("no strength" in s_ for s_ in wna),
      wna)
_, temp, _ = run(causal.dml_bounds, RG, HNA, kd=1, show=True)
check("no rows: the text says that no bound was computed", "No bound computed" in temp, temp[-300:])
_, tal, _ = run(causal.dml_bounds, RG, HB, kd=1, alpha=1e-12, show=True)
check("alpha = 1e-12: the level is never printed as '100%' (review)", "100% confidence" not in tal
      and "99.9999999999" in tal, [s_ for s_ in tal.splitlines() if "confidence" in s_][:1])
oe, _, _ = run(causal.dml_bounds, RG, HB, rho=1.2, show=False)
check("rho = 1.2 is a ValueError on the argument itself (a correlation), before any benchmark is read",
      isinstance(oe, ValueError) and "must be finite and in [-1, 1]" in str(oe) and "benchmark" not in str(oe), repr(oe))
HN = pd.DataFrame({"gain_y": [0.4, 0.0, 0.3], "gain_d": [0.4, 0.2, 0.3], "rho": [0.5, np.nan, np.nan]},
                  index=["b", "none", "nan_rho"])
on, _, wn = run(causal.dml_bounds, RG, HN, kd=1, show=False)
check("benchmark with no strength (gain 0) and with NaN rho under rho='benchmark': skipped, a warning each",
      on["bounds"]["benchmark"].tolist() == ["b"] and any("none" in s for s in wn)
      and any("nan_rho" in s and "rho" in s for s in wn), wn)
onone, _, wnone = run(causal.dml_bounds, RG, HN.loc[["none"]], kd=1, rho=1.0, show=False)
check("no strength (gain_y = 0) with rho = 1 given: still skipped, and the warning says why (mutation: the skip was "
      "seen only through its NaN rho)", onone["bounds"].empty and any("none" in s_ and "no strength" in s_ for s_ in wnone),
      wnone)
on1 = causal.dml_bounds(RG, HN.loc[["b", "nan_rho"]], kd=1, rho=1.0, show=False)
check("with rho = 1 given, the NaN-rho benchmark is used (its rho is not needed)",
      on1["bounds"]["benchmark"].tolist() == ["b", "nan_rho"])
# text: one line per row with its numbers, include / exclude zero on both sides
BS = pd.concat([BG[["gain_y", "gain_d", "rho"]],     # a strong benchmark, so the verdicts fall on both sides
                pd.DataFrame({"gain_y": [0.9], "gain_d": [0.45], "rho": [1.0]}, index=["strong"])])
ot, tt, _ = run(causal.dml_bounds, RG, BS, kd=[1, 2], rho=1.0, show=True)
rows_ok = []
for _, r in ot["bounds"].iterrows():
    word = "exclude zero" if r["ci_excludes_zero"] else "include zero"
    rows_ok.append(sum(f"{r['bound_label']}:" in s and f"{r['theta_lower']:.4f}" in s and f"{r['theta_upper']:.4f}" in s
                       and f"{r['ci_low']:.4f}" in s and f"{r['ci_high']:.4f}" in s and word in s
                       for s in tt.splitlines()) == 1)
check("text: one line per row, with its bounds, confidence bounds and the verdict word", all(rows_ok), tt)
check("verdicts on both sides in this design (some rows include zero, some exclude it)",
      ot["bounds"]["ci_excludes_zero"].any() and not ot["bounds"]["ci_excludes_zero"].all(),
      ot["bounds"][["bound_label", "ci_low", "ci_high"]].to_string())
# a row whose bounds exclude zero while its confidence bounds include it (k found by bisection on the bounds)
HV = pd.DataFrame({"gain_y": [0.3], "gain_d": [0.3], "rho": [1.0]}, index=["v"])
lo_k, hi_k = 1e-6, 1 / 0.3 * (1 - 1e-9)
for _ in range(200):
    mid = (lo_k + hi_k) / 2
    r_ = causal.dml_bounds(RG, HV, kd=mid, rho=1.0, show=False)["bounds"].iloc[0]
    lo_k, hi_k = (mid, hi_k) if r_["theta_lower"] > 0 else (lo_k, mid)
rv_ = causal.dml_bounds(RG, HV, kd=lo_k, rho=1.0, show=False)["bounds"].iloc[0]
check("a row with bounds above zero and confidence bounds reaching below it: verdict 'includes zero' (judged on the "
      "confidence bounds; mutation: the bounds survived)",
      rv_["theta_lower"] > 0 > rv_["ci_low"] and rv_["ci_verdict"] == "includes zero" and not rv_["ci_excludes_zero"],
      (rv_["theta_lower"], rv_["ci_low"]))
check("verdict matches the confidence bounds on every row",
      all(r["ci_excludes_zero"] == (r["ci_low"] > 0 or r["ci_high"] < 0) for _, r in ot["bounds"].iterrows()))
check("text says rho and that the gains are scaled by k as in the paper's benchmarking appendix",
      "rho = 1" in tt and "benchmarking appendix" in tt and "Appendix F" not in tt)

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
