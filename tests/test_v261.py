"""Tests for v2.6.1: validations of causal.dml_sensitivity and the findings of the v2.6.0 validation.

Origin: decisions 1 and 2 of the user, deferred from v2.6 (claude/plano-v2.6.md), and the three findings of the
v2.6.0 validation on notebook 1 (the agent built "the k that makes it insignificant" by hand on a grid of 0.1 and
rounded toward robustness: 1.1x/1.3x against 1.048x/1.203x; 51.9 KB of cap warnings on a grid of k; a memo that
mixed the HC3 CI with the classical baseline CI of ovb_bounds).

What is checked, each by a route other than the implementation's:
  1. dml_sensitivity: rho must be in [-1, 1] (+-1 accepted, 1 + 1e-12 refused, NaN and inf keep their message);
     the bias bound is proportional to |rho| (closed form: B(0.5) = B(1)/2, B(-1) = B(1));
  2. dml_sensitivity, q = 1: the target prints "0.0000", never "-0.0000", for theta < 0; q = 0.5 keeps the sign;
     _dml_pct (the share formatter, module-level since v2.6.1) on both sides of each threshold;
  3. ovb_bounds k_not_significant (the smallest k = kd = ky at which the worst-case CI reaches zero):
     - the v2.6.0 validation case (claude/scripts/triagem_sensibilidade_v260.py in the dev repository, seed
       20261111): 1.0472x region and 1.2024x size, against a brute force of 1e6 points written here from the
       paper's formulas;
     - known parameter: at the k found, a confounder Z with exactly the implied partial R² is built in the sample;
       the OLS refit with Z gives a CI that ends at zero; at k(1 - 1e-6) / k(1 + 1e-6) it excludes / includes it;
     - 8 designs where significance is lost and comes back before the cap and 20 where it is lost once (seeds
       copied from the output of the search, claude/plano-v2.6.1.md): k and significance_returns against brute force;
     - statuses on both sides: already (not significant), at once (inside the df vs df - 1 band), not reached;
       alpha moves k in the right direction; the sentences state the numbers (parsed back);
  4. cap warnings grouped: one per benchmark and call, naming the first and last k;
  5. header: "classical SE" only after a robust fit, the v2.6.0 bytes otherwise;
  6. sensitivity with a robust SE that is not finite: "not usable", never "t = 0.00".

Run from the repository root:  python3 tests/test_v261.py
Needs: statsmodels, scipy, scikit-learn.
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
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.linear_model import LinearRegression

import causal
import specification as sp

ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


def run(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = fn(*a, **k)
    return out, buf.getvalue(), [str(x.message) for x in w]


def raises(exc, fn, *a, **k):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fn(*a, **k)
    except exc as e:
        return str(e) or True
    except Exception:                            # noqa: BLE001
        return False
    return False


def quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


# ============================================================================== 1-2. dml_sensitivity
LIN = dict(learner_y=LinearRegression(), learner_d=LinearRegression())


def dml_design(seed, n, eff):
    rng = np.random.default_rng(seed)
    x1, x2, x3 = rng.normal(size=(3, n))
    d = 0.5 * x1 + 0.3 * x2 - 0.2 * x3 + rng.normal(size=n)
    y = eff * d + x1 + 0.5 * x2 + 0.3 * x3 + rng.normal(size=n)
    return pd.DataFrame(dict(y=y, d=d, x1=x1, x2=x2, x3=x3))


XD = ["x1", "x2", "x3"]
R_NEG = causal.dml_plr(dml_design(11, 500, -0.3), "y", "d", XD, n_rep=2, **LIN)
R_POS = causal.dml_plr(dml_design(1, 600, 0.4), "y", "d", XD, n_rep=2, **LIN)
check("setup: one negative and one positive significant DML estimate",
      R_NEG["estimate"] < 0 < R_POS["estimate"] and R_NEG["p"] < 0.01 and R_POS["p"] < 0.01,
      (R_NEG["estimate"], R_POS["estimate"]))

# --- 1. rho in [-1, 1]
B = {r: causal.dml_sensitivity(R_NEG, 0.05, 0.05, rho=r, show=False)["bias_bound"] for r in (1.0, -1.0, 0.5, -0.5)}
S = causal.dml_sensitivity(R_NEG, 0.05, 0.05, rho=1.0, show=False)["S"]
check("rho = 1: bias bound = S·sqrt(cf_y·cf_d/(1 - cf_d)) (closed form)",
      abs(B[1.0] - S * np.sqrt(0.05 * 0.05 / 0.95)) <= 1e-12 * B[1.0], B[1.0])
check("rho = -1 is accepted and gives the same bound as rho = 1 (the bound depends on |rho|)", B[-1.0] == B[1.0])
check("rho = +-0.5: exactly half the bound of rho = 1", abs(B[0.5] - B[1.0] / 2) <= 1e-12 * B[1.0]
      and B[-0.5] == B[0.5], (B[0.5], B[1.0]))
for bad in (1 + 1e-12, -1 - 1e-12, 1.5, -2.0, 30000.0):
    e = raises(ValueError, causal.dml_sensitivity, R_NEG, 0.05, 0.05, rho=bad, show=False)
    check(f"rho = {bad!r} is a ValueError that says rho must be in [-1, 1]", isinstance(e, str) and "[-1, 1]" in e, e)
for bad in (float("nan"), float("inf"), -float("inf")):
    e = raises(ValueError, causal.dml_sensitivity, R_NEG, 0.05, 0.05, rho=bad, show=False)
    check(f"rho = {bad!r} keeps the 'finite' message (benchmark hint only for nan)",
          isinstance(e, str) and "finite" in e and ("benchmark" in e) == bool(np.isnan(bad)), e)
o = causal.dml_benchmark(dml_design(11, 500, -0.3), "y", "d", XD, ["x1", "x2"], n_rep=1, **LIN) \
    if hasattr(causal, "dml_benchmark") else None
if o is not None:
    rhos = [float(v) for v in o["rho"] if np.isfinite(v)]
    check("dml_benchmark rho values are inside [-1, 1], so they pass the new validation",
          rhos and all(abs(v) <= 1 for v in rhos)
          and all(np.isfinite(causal.dml_sensitivity(R_NEG, 0.05, 0.05, rho=v, show=False)["bias_bound"]) for v in rhos),
          rhos)

# --- 2. "-0.0000"
def rv_line(txt):
    lines = [s for s in txt.splitlines() if s.startswith("  RV")]
    return lines[0] if len(lines) == 1 else "<%d RV lines>" % len(lines)


_, t_neg, _ = run(causal.dml_sensitivity, R_NEG, show=True)
_, t_pos, _ = run(causal.dml_sensitivity, R_POS, show=True)
check("theta < 0, q = 1: the RV sentence says 'theta to 0.0000', never '-0.0000'",
      "theta to 0.0000" in rv_line(t_neg) and "-0.0000" not in t_neg, rv_line(t_neg))
check("theta > 0, q = 1: 'theta to 0.0000' (unchanged)", "theta to 0.0000" in rv_line(t_pos) and "-0.0000" not in t_pos)
o5, t5, _ = run(causal.dml_sensitivity, R_NEG, q=0.5, show=True)
check("theta < 0, q = 0.5: the target keeps its sign, 0.5·theta (the other side of the fix)",
      f"theta to {0.5 * R_NEG['estimate']:.4f}" in rv_line(t5) and 0.5 * R_NEG["estimate"] < -0.00005, rv_line(t5))
check("q = 1 with q a rounding error from 1 (1 + 1e-15) also prints 0.0000",
      "-0.0000" not in run(causal.dml_sensitivity, R_NEG, q=1 + 1e-15, show=True)[1])

# _dml_pct: the "< 0.01%" threshold is 0.5·10^-(d+2), one case on each side, d = 2 and d = 1
P = causal._dml_pct
check("_dml_pct(4.9e-5, 2) = '< 0.01%' and (5e-5, 2) = '0.01%'", P(4.9e-5, 2) == "< 0.01%" and P(5e-5, 2) == "0.01%",
      (P(4.9e-5, 2), P(5e-5, 2)))
check("_dml_pct(4.9e-4, 1) = '< 0.1%' and (5e-4, 1) = '0.1%'", P(4.9e-4, 1) == "< 0.1%" and P(5e-4, 1) == "0.1%",
      (P(4.9e-4, 1), P(5e-4, 1)))
check("_dml_pct(0, 2) = '0.00%' (zero is zero, not '< 0.01%') and (0.4328, 2) = '43.28%'",
      P(0.0, 2) == "0.00%" and P(0.4328, 2) == "43.28%")
check("with |rho| <= 1 the printed RV and RVa never read 0.00%", "0.00%" not in t_neg and "0.00%" not in t_pos)

# ============================================================================== 3. k_not_significant
# Paper's formulas, written here (Cinelli-Hazlett 2020, eq. 12-13 and §4.4), not imported from specification
def implied_r2(k, r2d, r2y):
    r2dz = k * r2d / (1 - r2d)
    r2zx = k * r2d ** 2 / ((1 - k * r2d) * (1 - r2d))
    r2yz = ((np.sqrt(k) + np.sqrt(r2zx)) / np.sqrt(1 - r2zx)) ** 2 * (r2y / (1 - r2y))
    return r2dz, r2yz


def g_brute(k, r2d, r2y, est, se, df, tc):
    """sign(est)·(adjusted estimate) - t·(adjusted SE): > 0 means the worst-case CI excludes zero on the same side."""
    r2dz, r2yz = implied_r2(k, r2d, r2y)
    bias = np.sqrt(r2yz * r2dz / (1 - r2dz)) * se * np.sqrt(df)
    adj_se = np.sqrt((1 - r2yz) / (1 - r2dz)) * se * np.sqrt(df / (df - 1))
    return abs(est) - bias - tc * adj_se, r2yz


def brute(br, est, se, df, tc, npts=1_000_000):
    """First k on a 1e6-point grid with g <= 0, the step, whether g > 0 again later, all before r2yz = 1."""
    hi = min(br["kd_max"], 1e7)
    ks = np.linspace(0, hi, npts + 1)[1:-1]
    with np.errstate(all="ignore"):
        g, r2yz = g_brute(ks, br["r2dxj_x"], br["r2yxj_dx"], est, se, df, tc)
    keep = r2yz < 1
    ks, g = ks[keep], g[keep]
    hit = np.flatnonzero(g <= 0)
    if not len(hit):
        return np.inf, False, ks[1] - ks[0]
    return ks[hit[0]], bool(np.any(g[hit[0] + 1:] > 0)), ks[1] - ks[0]


def stats_of(m, d):
    return float(m.params[d]), float(m.bse[d]), float(m.df_resid)


# --- the v2.6.0 validation case
def simular(seed=20261111, n=400):                 # copied from claude/scripts/triagem_sensibilidade_v260.py
    rng = np.random.default_rng(seed)
    regiao = rng.integers(1, 7, n)
    efeito_reg = np.array([0, 6, -4, 10, 3, -7])[regiao - 1]
    adesao_reg = np.array([-1.0, 0.8, -0.3, 1.2, 0.0, -0.6])[regiao - 1]
    log_pac = rng.normal(8.0, 0.6, n)
    med = np.clip(rng.normal(1.5, 0.4, n) + 0.3 * (log_pac - 8), 0.3, None)
    z = -0.4 + 1.5 * (log_pac - 8) + 1.2 * (med - 1.5) + adesao_reg
    triagem = rng.binomial(1, 1 / (1 + np.exp(-z)))
    espera = (45 + 9 * (log_pac - 8) - 10 * (med - 1.5) + efeito_reg + -8.0 * triagem + rng.normal(0, 8, n))
    return pd.DataFrame({"espera": espera, "triagem": triagem, "log_pacientes": log_pac,
                         "medicos_por_1000": med, "regiao": regiao})


VAL = simular()
F_VAL = "espera ~ triagem + log_pacientes + medicos_por_1000 + C(regiao)"
M_VAL = smf.ols(F_VAL, VAL).fit()
M_VAL_HC3 = smf.ols(F_VAL, VAL).fit(cov_type="HC3")
check("setup: the validation case reproduces (OLS -6.523246, as in the v2.6.0 record)",
      abs(M_VAL.params["triagem"] - (-6.523246)) < 5e-7, M_VAL.params["triagem"])
O_VAL, T_VAL, W_VAL = run(sp.ovb_bounds, M_VAL_HC3, "triagem", ["C(regiao)", "log_pacientes"], show=True)
BR = O_VAL["benchmark_r2"]
est, se, df = stats_of(M_VAL, "triagem")
TC = stats.t.ppf(0.975, df)
for b, kref in (("C(regiao)", 1.0472), ("log_pacientes", 1.2024)):
    kb, retb, step = brute(BR.loc[b], est, se, df, TC)
    kf = BR.loc[b, "k_not_significant"]
    check(f"validation case, {b}: k_not_significant = {kref} (4 decimals) and on the brute-force grid "
          f"(previous point < k <= first point with the CI reaching zero)",
          round(kf, 4) == kref and kb - step < kf <= kb and BR.loc[b, "k_not_significant_status"] == "found"
          and not BR.loc[b, "significance_returns"] and not retb, (kf, kb, step))

# known parameter: refit with a confounder of exactly the implied R² at k, k(1 - 1e-6) and k(1 + 1e-6)
def resid(v, A):
    return v - A @ np.linalg.lstsq(A, v, rcond=None)[0]


def build_z(res, treatment, r2dz, r2yz, sign=1, seed=0):
    """Z orthogonal to X with R²(D~Z|X) = r2dz and R²(Y~Z|X,D) = r2yz exactly in the sample (as in test_v26)."""
    X, y = res.model.exog, res.model.endog
    j = list(res.params.index).index(treatment)
    dt, yt = resid(X[:, j], np.delete(X, j, axis=1)), resid(y, X)
    e = resid(np.random.default_rng(seed).normal(size=len(y)), np.column_stack([X, y]))
    al, be = np.sqrt(r2dz), np.sqrt(r2yz * (1 - r2dz))
    ga = np.sqrt(max(1 - r2dz - be ** 2, 0.0))
    return al * dt / np.linalg.norm(dt) + sign * be * yt / np.linalg.norm(yt) + ga * e / np.linalg.norm(e)


def refit_near_bound(res, treatment, k, r2d, r2y, tc):
    """CI end nearest zero of the refit with Z pulling the estimate toward zero (t of the original df, as the
    bounds and sensemakr use)."""
    r2dz, r2yz = implied_r2(k, r2d, r2y)
    j = list(res.params.index).index(treatment)
    fits = [sm.OLS(res.model.endog, np.column_stack([res.model.exog, build_z(res, treatment, r2dz, r2yz, s)])).fit()
            for s in (1, -1)]
    f = min(fits, key=lambda f_: abs(f_.params[j]))
    b, s_ = f.params[j], f.bse[j]
    return np.sign(res.params[treatment]) * b - tc * s_


for b in ("C(regiao)", "log_pacientes"):
    k = BR.loc[b, "k_not_significant"]
    r2d, r2y = BR.loc[b, "r2dxj_x"], BR.loc[b, "r2yxj_dx"]
    at = refit_near_bound(M_VAL, "triagem", k, r2d, r2y, TC)
    lo = refit_near_bound(M_VAL, "triagem", k * (1 - 1e-6), r2d, r2y, TC)
    hi = refit_near_bound(M_VAL, "triagem", k * (1 + 1e-6), r2d, r2y, TC)
    check(f"known parameter, {b}: the refit with Z built at k has its CI ending at zero (|end| < 1e-8·SE); "
          f"just below k it excludes zero, just above it includes it",
          abs(at) < 1e-8 * se and lo > 0 and hi < 0, (at, lo, hi))

# --- designs found by the search (seeds copied from its output, claude/plano-v2.6.1.md)
def design(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(30, 600))
    a, b_, c = rng.normal(size=3) * rng.choice([0.3, 1, 3], 3)
    x = rng.normal(size=n)
    w = rng.normal(size=n)
    d = a * x + rng.normal(size=n)
    y = rng.normal() * d + b_ * x + c * w + rng.normal(size=n) * rng.choice([0.5, 1, 3])
    return pd.DataFrame(dict(y=y, d=d, x=x, w=w))


TWO = [(29, "x"), (30, "w"), (212, "x"), (309, "w"), (498, "w"), (736, "w"), (800, "x"), (996, "x")]
ONE = [(0, "x"), (0, "w"), (1, "x"), (1, "w"), (3, "x"), (4, "x"), (4, "w"), (5, "x"), (6, "x"), (7, "x"),
       (8, "x"), (9, "x"), (12, "x"), (13, "x"), (14, "x"), (15, "x"), (17, "x"), (19, "x"), (20, "x"), (20, "w")]
bad2, bad1, txt_bad = [], [], []
for cases, want_ret, bad in ((TWO, True, bad2), (ONE, False, bad1)):
    for s, bm in cases:
        m = smf.ols("y ~ d + x + w", design(s)).fit()
        o, txt, _ = run(sp.ovb_bounds, m, "d", bm, show=True)
        br = o["benchmark_r2"].loc[bm]
        e_, s_, d_ = stats_of(m, "d")
        tc = stats.t.ppf(0.975, d_)
        kb, retb, step = brute(br, e_, s_, d_, tc)
        kf = br["k_not_significant"]
        good = (br["k_not_significant_status"] == "found" and kb - step < kf <= kb
                and bool(br["significance_returns"]) == retb == want_ret)
        if not good:
            bad.append((s, bm, kf, kb, br["significance_returns"], retb))
        has_ret = "not as a threshold" in txt
        if has_ret != want_ret or f"{kf:.3g}x as strong as {bm}" not in txt:
            txt_bad.append((s, bm))
check("8 designs where significance is lost and returns before the cap: k on the brute-force grid and "
      "significance_returns = True, as the brute force says", not bad2, bad2)
check("20 designs where it is lost once: k on the brute-force grid and significance_returns = False", not bad1, bad1)
check("the sentence states k (3 significant digits) and says 'not as a threshold' exactly when it returns",
      not txt_bad, txt_bad)

# --- a dip narrower than any grid step, built on purpose: t shifted to t* (where the minimum of g just touches zero,
# found by bisection on a 2e5-point brute force; values copied from the output of that search) times (1 -+ 1e-7)
TSTAR = {(309, "w"): 2.4001602855603816, (996, "x"): -6.282644600221862}
for (s, bm), ts in TSTAR.items():
    D_ = design(s)
    m0_ = smf.ols("y ~ d + x + w", D_).fit()
    for side, fac in (("just inside (1 - 1e-7)", 1 - 1e-7), ("just outside (1 + 1e-7)", 1 + 1e-7)):
        b = ts * fac * m0_.bse["d"] - m0_.params["d"]
        D2 = D_.assign(y=D_["y"] + b * D_["d"])
        m = smf.ols("y ~ d + x + w", D2).fit()
        br = quiet(sp.ovb_bounds, m, "d", bm)["benchmark_r2"].loc[bm]
        e_, s_, d_ = stats_of(m, "d")
        tc = stats.t.ppf(0.975, d_)
        kb, retb, step = brute(br, e_, s_, d_, tc)
        kf = br["k_not_significant"]
        if fac < 1:
            g_at = g_brute(kf, br["r2dxj_x"], br["r2yxj_dx"], e_, s_, d_, tc)[0]
            g_lo = g_brute(kf * (1 - 1e-9), br["r2dxj_x"], br["r2yxj_dx"], e_, s_, d_, tc)[0]
            good = (br["k_not_significant_status"] == "found" and abs(g_at) < 1e-9 * abs(e_) and g_lo > 0
                    and br["significance_returns"] and kb - step < kf <= kb and 0.5 * br["k_cap"] < kf < br["k_cap"])
        else:
            good = br["k_not_significant_status"] == "not reached" and np.isinf(kf) and np.isinf(kb)
        check(f"narrow dip (seed {s}, {bm}), t {side}: " + ("found at the dip (g = 0 there, > 0 just before; the "
              "brute force agrees) and significance returns" if fac < 1 else "no dip: not reached, as the brute force"),
              good, (br["k_not_significant_status"], kf, kb, br["k_cap"]))

# --- statuses on both sides
def with_t(m0, data, formula, t_target, d="d"):
    """Same data with the outcome shifted by b·d so that the classical t of d is t_target (residuals unchanged)."""
    b = t_target * m0.bse[d] - m0.params[d]
    data = data.copy()
    data["y"] = data["y"] + b * data[d]
    return smf.ols(formula, data).fit()


D0 = design(0)
M0 = smf.ols("y ~ d + x + w", D0).fit()
dfr = M0.df_resid
t_df = stats.t.ppf(0.975, dfr)
band_hi = t_df * np.sqrt(dfr / (dfr - 1))           # g(0+) = |est| - t·SE·sqrt(df/(df-1))
cases = {"already (|t| just below t_crit)": (t_df * (1 - 1e-6), "already", 0.0),
         "at once (just above t_crit, inside the band)": (t_df * (1 + 1e-6), "at once", 0.0),
         "at once (just below the band's upper end)": (band_hi * (1 - 1e-6), "at once", 0.0),
         "found (just above the band's upper end)": (band_hi * (1 + 1e-4), "found", None)}
for nm, (tt, st, kexp) in cases.items():
    m = with_t(M0, D0, "y ~ d + x + w", tt)
    o, txt, _ = run(sp.ovb_bounds, m, "d", "x", show=True)
    br = o["benchmark_r2"].loc["x"]
    good = br["k_not_significant_status"] == st and (kexp is None or br["k_not_significant"] == kexp)
    if st == "found":
        good &= 0 < br["k_not_significant"] < 1e-3
    if st == "already":
        good &= "not significant at the 5% level even with no confounder" in txt
    if st == "at once":
        good &= "only just significant at the 5% level" in txt and "vanishing strength" in txt
    check(f"status {nm}: {st}", good, (br["k_not_significant_status"], br["k_not_significant"], m.tvalues["d"]))
m_neg = with_t(M0, D0, "y ~ d + x + w", -t_df * (1 - 1e-6))
check("status already, negative estimate: also 'already'",
      quiet(sp.ovb_bounds, m_neg, "d", "x")["benchmark_r2"].loc["x", "k_not_significant_status"] == "already")

# not reached: a large t and a benchmark that explains little of D: the bound never reaches zero before r2yz = 1
m_big = with_t(M0, D0, "y ~ d + x + w", 60.0)
o, txt, _ = run(sp.ovb_bounds, m_big, "d", "w", show=True)
br = o["benchmark_r2"].loc["w"]
e_, s_, d_ = stats_of(m_big, "d")
kb, _, _ = brute(br, e_, s_, d_, stats.t.ppf(0.975, d_))
check("not reached: k = inf, cap from the outcome, the sentence names the cap; the brute force agrees",
      br["k_not_significant_status"] == "not reached" and np.isinf(br["k_not_significant"]) and np.isinf(kb)
      and br["k_cap_by"] == "outcome" and f"No k below {br['k_cap']:.4g} makes the estimate not significant" in txt
      and "explain all of the outcome" in txt, (br["k_not_significant"], kb, br["k_cap"]))
r2d_w, r2y_w = br["r2dxj_x"], br["r2yxj_dx"]
_, r2yz_cap = implied_r2(br["k_cap"], r2d_w, r2y_w)
check("k_cap is where the implied R² with the outcome reaches 1 (paper's formula: 1 within 1e-10)",
      abs(r2yz_cap - 1) < 1e-10, r2yz_cap)

# alpha moves k the right way: a stricter test (smaller alpha) loses significance sooner
ks_a = {}
for a in (0.10, 0.05, 0.01):
    ks_a[a] = quiet(sp.ovb_bounds, M_VAL, "triagem", "log_pacientes", alpha=a)["benchmark_r2"].loc[
        "log_pacientes", "k_not_significant"]
check("alpha: k(0.10) > k(0.05) > k(0.01) (a wider CI reaches zero sooner)", ks_a[0.10] > ks_a[0.05] > ks_a[0.01], ks_a)
kb01, _, step01 = brute(BR.loc["log_pacientes"], est, se, df, stats.t.ppf(0.995, df))
check("alpha = 0.01: on the brute-force grid too", kb01 - step01 < ks_a[0.01] <= kb01, (ks_a[0.01], kb01))
check("k_not_significant does not depend on q (it is about zero at alpha; q only moves k_needed)",
      quiet(sp.ovb_bounds, M_VAL, "triagem", "log_pacientes", q=0.5)["benchmark_r2"].loc[
          "log_pacientes", "k_not_significant"] == BR.loc["log_pacientes", "k_not_significant"])
check("the validation text names both k with 3 significant digits and the 'dense grid' caveat",
      "1.05x as strong as C(regiao)" in T_VAL and "1.2x as strong as log_pacientes" in T_VAL
      and T_VAL.count("make the estimate not significant at the 5% level") == 2 and "dense grid" in T_VAL)
check("k_not_significant <= k_needed whenever both are found (the CI reaches zero before the estimate does)",
      all(BR["k_not_significant"] <= BR["k_needed"]))

# --- findings of the v2.6.1 review, round 1
# k_significant_again: in the designs where significance returns, the first k after the loss with g > 0 (brute force)
bad_a = []
for s, bm in TWO:
    m = smf.ols("y ~ d + x + w", design(s)).fit()
    br = quiet(sp.ovb_bounds, m, "d", bm)["benchmark_r2"].loc[bm]
    e_, s_, d_ = stats_of(m, "d")
    tc = stats.t.ppf(0.975, d_)
    hi = min(br["kd_max"], 1e7)
    ks = np.linspace(0, hi, 1_000_001)[1:-1]
    with np.errstate(all="ignore"):
        g_, r2yz_ = g_brute(ks, br["r2dxj_x"], br["r2yxj_dx"], e_, s_, d_, tc)
    keep = r2yz_ < 1
    ks, g_ = ks[keep], g_[keep]
    first_neg = np.flatnonzero(g_ <= 0)[0]
    back = first_neg + np.flatnonzero(g_[first_neg:] > 0)[0]
    ka = br["k_significant_again"]
    if not (ks[back - 1] < ka <= ks[back] and br["k_not_significant"] < ka < br["k_cap"]):
        bad_a.append((s, bm, ka, ks[back]))
check("k_significant_again: on the brute-force grid (last k with g <= 0 < k_again <= first k with g > 0 again)",
      not bad_a, bad_a)
_, t29, _ = run(sp.ovb_bounds, smf.ols("y ~ d + x + w", design(29)).fit(), "d", "x", show=True)
br29 = quiet(sp.ovb_bounds, smf.ols("y ~ d + x + w", design(29)).fit(), "d", "x")["benchmark_r2"].loc["x"]
check("the returning sentence names k_significant_again ('From …x on') and 'not as a threshold'",
      f"From {br29['k_significant_again']:.3g}x on" in t29 and "not as a threshold" in t29, br29["k_significant_again"])
m_one = smf.ols("y ~ d + x + w", design(0)).fit()
check("significance lost once: k_significant_again is NaN",
      np.isnan(quiet(sp.ovb_bounds, m_one, "d", "x")["benchmark_r2"].loc["x", "k_significant_again"]))

# R1 (review): a dip within 1e-6 of the cap — SciPy's bounded minimize had a relative floor on its tolerance and
# stopped at once there; the reviewer's case, checked with the paper's formulas written here
r2d_, r2y_, df_, se_ = 0.0004992435266852946, 6.2740517799191556e-06, 10.0, 1.0
tc_ = stats.t.ppf(0.975, df_)
est_ = 1376.996017
kf, kret, kst, kcap, kby, kag = sp._k_not_significant(r2d_, r2y_, est_, se_, df_, tc_)
g_at = g_brute(kf, r2d_, r2y_, est_, se_, df_, tc_)[0] if np.isfinite(kf) else np.nan
g_lo = g_brute(kf * (1 - 1e-12), r2d_, r2y_, est_, se_, df_, tc_)[0] if np.isfinite(kf) else np.nan
check("review R1: a dip within 1e-6 of the cap is found (k ≈ 2002.01573, g = 0 there and > 0 just before), and "
      "significance returns before the cap",
      kst == "found" and abs(kf - 2002.0157316) < 1e-6 and abs(g_at) < 1e-6 and g_lo > 0 and kret
      and kf < kag < kcap and 1 - kf / kcap < 1e-6, (kst, kf, kag, kcap))

# R2 (review): the benchmark predicts the outcome but hardly the treatment; t in the middle of the df vs df - 1 band:
# "at once", but the CI excludes zero again from a tiny k on. The sentence must say so, not "however weak"
rng = np.random.default_rng(5)
n_ = 400
x_ = rng.normal(size=n_)
w_ = rng.normal(size=n_)
d2 = 0.01 * x_ + 0.5 * w_ + rng.normal(size=n_)
y2 = 2.0 * x_ + 0.3 * w_ + rng.normal(size=n_)
XA = pd.DataFrame(dict(const=1.0, d=d2, x=x_, w=w_))
mA0 = sm.OLS(y2, XA).fit()
dfA = mA0.df_resid
tcA = stats.t.ppf(0.975, dfA)
bandA = tcA * np.sqrt(dfA / (dfA - 1))


def at_t(tt):
    return sm.OLS(y2 + (tt * mA0.bse["d"] - mA0.params["d"]) * d2, XA).fit()


mA = at_t((tcA + bandA) / 2)
oA, tA, _ = run(sp.ovb_bounds, mA, "d", ["x", "w"], kd=[0.001, 0.01, 0.1], show=True)
brA = oA["benchmark_r2"]
eA, sA, _ = stats_of(mA, "d")
ks = np.geomspace(1e-12, 0.2, 400_000)
gA, _ = g_brute(ks, brA.loc["x", "r2dxj_x"], brA.loc["x", "r2yxj_dx"], eA, sA, dfA, tcA)
last_neg = ks[np.flatnonzero(gA <= 0)[-1]]
nxt = ks[np.flatnonzero(gA <= 0)[-1] + 1]
check("review R2: 'at once' for every benchmark of the call (decided at k -> 0, the same for all)",
      set(brA["k_not_significant_status"]) == {"at once"}, brA["k_not_significant_status"].tolist())
check("review R2: for x significance returns, k_significant_again on the brute-force grid, and the 0.01x and 0.1x lines "
      "indeed say the CI excludes zero",
      brA.loc["x", "significance_returns"] and last_neg < brA.loc["x", "k_significant_again"] <= nxt
      and "0.01x x:" in tA and "0.1x x:" in tA and tA.count("which still excludes zero") >= 2,
      (brA.loc["x", "k_significant_again"], last_neg))
check("review R2: the text says only a confounder weaker than k_significant_again makes it insignificant, and no "
      "'however weak'",
      f"For x: from {brA.loc['x', 'k_significant_again']:.3g}x on" in tA and "weaker than that" in tA
      and "however weak" not in tA and "vanishing strength" in tA, [l for l in tA.splitlines() if "For x" in l])

# R3 (review): |t| a hair above the band's upper end: the k = 0 statuses are analytic, so the call never mixes
# "at once" with another status; the tiny k is found from k = 0
mB = at_t(bandA * (1 + 1e-12))
oB, tB, _ = run(sp.ovb_bounds, mB, "d", ["x", "w"], show=True)
stB = oB["benchmark_r2"]["k_not_significant_status"].tolist()
check("review R3: just above the band: no 'at once' (the shared sentence is not printed); w found at a tiny k",
      "at once" not in stB and "only just significant" not in tB
      and oB["benchmark_r2"].loc["w", "k_not_significant_status"] == "found"
      and 0 < oB["benchmark_r2"].loc["w", "k_not_significant"] < 1e-9, (stB, oB["benchmark_r2"]["k_not_significant"].tolist()))
eB, sB, _ = stats_of(mB, "d")
kw = oB["benchmark_r2"].loc["w", "k_not_significant"]
# g(0+) is ~1e-12 of |est| here, so g is only resolved to a factor of k (cancellation in the test's own formula)
gw = g_brute(np.array([kw / 2, kw * 2]), oB["benchmark_r2"].loc["w", "r2dxj_x"],
             oB["benchmark_r2"].loc["w", "r2yxj_dx"], eB, sB, dfA, tcA)[0]
check("review R3: at that tiny k the CI reaches zero (g > 0 at k/2, <= 0 at 2k)", gw[0] > 0 >= gw[1], gw)

# R4 (review): a benchmark that explains nothing of either side: no "No k below inf"
rng = np.random.default_rng(1)
n_ = 200
x_ = rng.normal(size=n_)
d3 = 0.5 * x_ + rng.normal(size=n_)
y3 = 1.0 * d3 + x_ + rng.normal(size=n_)
A_ = np.column_stack([np.ones(n_), d3, x_, y3])
z_ = rng.normal(size=n_)
z_ = z_ - A_ @ np.linalg.lstsq(A_, z_, rcond=None)[0]
oZ, tZ, _ = run(sp.ovb_bounds, sm.OLS(y3, pd.DataFrame(dict(const=1.0, d=d3, x=x_, z=z_))).fit(), "d", "z", show=True)
check("review R4: a benchmark that explains nothing of either: 'explains nothing of the treatment or of the outcome', "
      "never 'No k below inf'",
      oZ["benchmark_r2"].loc["z", "k_not_significant_status"] == "not reached" and "below inf" not in tZ
      and "No multiple of z makes the estimate not significant" in tZ)

# docs review S1: after a robust fit the k sentence says the classical SE; not after a classical fit
check("review S1: after a robust fit the k_not_significant sentence says '(classical SE)'; after a classical one, not",
      "level (classical SE) (its CI reaches zero" in T_VAL
      and "classical SE" not in run(sp.ovb_bounds, M_VAL, "triagem", "log_pacientes", show=True)[1])

# ============================================================================== 4. grouped cap warnings
SIM = pd.read_csv(HERE / "data" / "ovb_v26.csv")
M_SIM = smf.ols("y ~ d + x_strong + x_weak + C(g)", SIM).fit()
grid = list(np.round(np.arange(0.1, 3.01, 0.1), 2))
o, txt, w = run(sp.ovb_bounds, M_VAL, "triagem", ["C(regiao)", "log_pacientes"], kd=grid)
cap_reg = o["bounds"].query("benchmark == 'C(regiao)' and r2yz_dx_capped")["kd"].tolist()
cap_por = o["bounds"].query("benchmark == 'log_pacientes' and r2yz_dx_capped")["kd"].tolist()
check("grid of 30 k: many capped lines but exactly one warning per benchmark (warnings.warn and out['warnings'])",
      len(cap_reg) > 6 and len(cap_por) == 3 and len(w) == 2 and o["warnings"] == w, (len(cap_reg), len(cap_por), len(w)))
check("the grouped warning names the benchmark, the first and the last capped k and the count",
      w and f"C(regiao), kd = ky = {cap_reg[0]:g} to {cap_reg[-1]:g} ({len(cap_reg)} values)" in w[0]
      and "the CIs of those lines are not informative" in w[0] and "capped at 1" in w[0]
      and "log_pacientes, kd = ky = 2.8, 2.9, 3:" in w[1], w)
o1, _, w1 = run(sp.ovb_bounds, M_VAL, "triagem", "log_pacientes", kd=3)
check("one capped line: one warning, 'its CI is not informative', with that k and its implied R²",
      len(w1) == 1 and "log_pacientes, kd = ky = 3:" in w1[0] and "its CI is not informative" in w1[0]
      and f"{implied_r2(3, BR.loc['log_pacientes', 'r2dxj_x'], BR.loc['log_pacientes', 'r2yxj_dx'])[1]:.3g} > 1"
      in w1[0], w1)
o2, _, w2 = run(sp.ovb_bounds, M_VAL, "triagem", "log_pacientes", kd=[1, 3], ky=[1, 2.95])
check("kd != ky: the warning lists kd and ky of the capped line only",
      o2["bounds"]["r2yz_dx_capped"].tolist() == [False, True] and len(w2) == 1
      and "log_pacientes, kd = 3 with ky = 2.95:" in w2[0], w2)
_, _, w5 = run(sp.ovb_bounds, M_VAL, "triagem", "log_pacientes", kd=[2.9, 2.75, 3.0, 2.8, 2.95, 2.85, 2.78, 1.0])
check("review R5: more than six capped k given unsorted: 'min to max (count)', not first to last",
      len(w5) == 1 and "kd = ky = 2.75 to 3 (7 values)" in w5[0], w5)
_, _, w6 = run(sp.ovb_bounds, M_VAL, "triagem", "log_pacientes", kd=[3, 3, 3, 1])
check("review R5: a repeated (kd, ky) pair is named once ('kd = ky = 3:', one R², 'its CI is')",
      len(w6) == 1 and "kd = ky = 3:" in w6[0] and "its CI is not informative" in w6[0] and "3, 3" not in w6[0], w6)
_, _, w0 = run(sp.ovb_bounds, M_SIM, "d", "x_weak", kd=[1, 2])
check("nothing capped: no warning", not w0, w0)

# ============================================================================== 5. header label
o, t_rob, _ = run(sp.ovb_bounds, M_VAL_HC3, "triagem", "log_pacientes", kd=1, show=True)
o, t_cls, _ = run(sp.ovb_bounds, M_VAL, "triagem", "log_pacientes", kd=1, show=True)
lo_, hi_ = est - TC * se, est + TC * se
head_v260 = (f"Bounds on omitted-variable bias for triagem (Cinelli-Hazlett 2020, OLS): estimate {est:.4g} "
             f"(SE {se:.4g}, df {df:g}; 95% CI [{lo_:.4g}, {hi_:.4g}]).")
check("classical fit: the header is the v2.6.0 line, byte for byte", head_v260 in t_cls.splitlines(), t_cls[:200])
check("robust fit: the header says 'classical SE' (the baseline CI is the classical one), the rest unchanged",
      head_v260.replace("(SE ", "(classical SE ") in t_rob.splitlines() and head_v260 not in t_rob.splitlines())

# ============================================================================== 6. sensitivity, robust SE not finite
class _Robust:
    def __init__(self, base, factor):
        self.model, self.cov_type, self.params = base.model, "HC3", base.params
        self.bse = base.bse * factor
        self.tvalues = base.params / self.bse


for nm, fac in (("infinite", np.inf), ("NaN", np.nan), ("zero", 0.0)):
    _, t_, _ = run(sp.sensitivity, _Robust(M_SIM, fac), "d", show=True)
    check(f"sensitivity, robust SE {nm}: 'its t is not usable', never 't = 0.00' or 't = nan'",
          "so its t is not usable" in t_ and "t = 0.00;" not in t_ and "t = nan" not in t_ and "t = inf" not in t_,
          [s for s in t_.splitlines() if "classical SE" in s])
_, t_fin, _ = run(sp.sensitivity, smf.ols("y ~ d + x_strong + x_weak + C(g)", SIM).fit(cov_type="HC3"), "d", show=True)
check("sensitivity, finite HC3: 'your HC3 fit gives t =' and nothing about 'not usable'",
      "your HC3 fit gives t = " in t_fin and "not usable" not in t_fin)

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
