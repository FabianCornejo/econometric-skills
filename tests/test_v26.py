"""Tests for v2.6: specification.ovb_bounds, the Cinelli-Hazlett (2020) bounds "a confounder k times as strong as
an observed covariate" for OLS, and the k that would remove 100·q% of the estimate.

Origin: on notebook 1 the agent wrote these bounds by hand twice (validations of v2.5.2 and v2.5.3) and took the
numbers to a deliverable unverified; the rule in interpretation-rules.md asks for exactly this sentence.

What is checked here, each by a route other than the implementation's formulas:
  1. the published Darfur table (sensemakr README; Cinelli-Hazlett 2020, Table 1 and Figure 2), from the
     summary statistics printed in sensemakr's own example;
  2. R's sensemakr 0.1.6: tools/r_reference/ovb_bounds_v26_reference.R, run on notebook 1; Darfur through the
     summary statistics it exports (the data are GPL-3 and stay out of this MIT repository) and
     tests/data/ovb_v26.csv through the whole function, from a fitted model;
  3. known parameter: a confounder Z orthogonal to X is built in the sample with partial R² exactly r2dz_x (with D)
     and r2yz_dx (with Y given D); refitting the OLS with Z gives the adjusted estimate (Z of one sign) or the
     estimate pushed away from zero (the other sign) and the adjusted SE and CI;
  4. the benchmarks' partial R² against refits (1 - SSR_full / SSR_without), in the outcome and in the treatment
     regression (one more residual df), for single regressors and for groups;
  5. k_needed: at k_needed the refit with the built Z lands on (1 - q)·estimate; just below and just above it the
     bound has not / has reached the target;
  6. boundaries, one case on each side of each limit: kd at the largest admissible value, the implied R² with the
     outcome at 1 (warning and flag only above, also with show=False), the adjusted estimate crossing zero, the
     adjusted CI reaching zero;
  7. the printed sentences state what the row says (numbers parsed back from the text), and the sentences that
     must not appear do not;
  8. validation, robust fits (refit classically, same numbers), equivariance to units, grouping shared with
     sensitivity, and the line sensitivity now prints.

Run from the repository root:  python3 tests/test_v26.py
Needs: statsmodels, scipy.
"""
import contextlib
import io
import json
import re
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path.insert(0, str(ROOT / "econometrics" / "scripts"))
import _console  # noqa: E402,F401  (Windows-safe console output, see econometrics/scripts/_console.py)
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

import specification as sp

ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


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
    except Exception as e:                      # noqa: BLE001
        return False if not isinstance(e, exc) else str(e)
    return False


SIM = pd.read_csv(HERE / "data" / "ovb_v26.csv")
F_SIM = "y ~ d + x_strong + x_weak + C(g)"
M_SIM = smf.ols(F_SIM, SIM).fit()


def resid(v, A):
    return v - A @ np.linalg.lstsq(A, v, rcond=None)[0]


def build_z(res, treatment, r2dz, r2yz, sign=1, seed=0):
    """Z orthogonal to the covariates, with R²(D~Z|X) = r2dz and R²(Y~Z|X,D) = r2yz exactly in the sample:
    Z = √a·d̃ + s·√(b(1−a))·ỹ + √(1 − a − b(1−a))·ẽ (unit vectors; d̃ = D⊥X, ỹ = Y⊥X,D, ẽ ⊥ X,D,Y). sign = +1 pulls
    the estimate toward zero when d̃'ỹ… is positive-oriented; the caller tries both signs."""
    X = res.model.exog
    y = res.model.endog
    j = list(res.params.index).index(treatment)
    Xo = np.delete(X, j, axis=1)
    dt, yt = resid(X[:, j], Xo), resid(y, X)
    e = resid(np.random.default_rng(seed).normal(size=len(y)), np.column_stack([X, y]))
    al, be = np.sqrt(r2dz), np.sqrt(r2yz * (1 - r2dz))
    ga = np.sqrt(max(1 - r2dz - be ** 2, 0.0))
    return al * dt / np.linalg.norm(dt) + sign * be * yt / np.linalg.norm(yt) + ga * e / np.linalg.norm(e)


def refit_with(res, z):
    return sm.OLS(res.model.endog, np.column_stack([res.model.exog, z])).fit()


def both_refits(res, treatment, r2dz, r2yz):
    """(estimate pulled toward zero, estimate pushed away, SE) from refits with Z of each sign."""
    j = list(res.params.index).index(treatment)
    fits = [refit_with(res, build_z(res, treatment, r2dz, r2yz, s)) for s in (1, -1)]
    ests = sorted((f.params[j] for f in fits), key=abs)
    return ests[0], ests[1], fits[0].bse[j], fits[1].bse[j]


# ------------------------------------------------------------------------------ 1. published Darfur table
# sensemakr's own example (ovb_partial_r2_bound.numeric): t of female = -9.789 in the outcome regression and
# -2.680 in the treatment regression, dof = 783, estimate 0.0973, se 0.0232. README table (4 decimals):
PUB = [(1, 0.0092, 0.1246, 0.0752, 0.0219), (2, 0.0183, 0.2493, 0.0529, 0.0204), (3, 0.0275, 0.3741, 0.0304, 0.0187)]
r2y_f = 9.789 ** 2 / (9.789 ** 2 + 783)
r2d_f = 2.680 ** 2 / (2.680 ** 2 + 783)
r2dz_p, r2yz_p = sp._ovb_partial_r2_bound(r2d_f, r2y_f, np.array([1., 2., 3.]), np.array([1., 2., 3.]))
adj_p, se_p, t_p = sp._ovb_adjust(0.0973, 0.0232, 783, r2dz_p, r2yz_p)
# The R² come from the t-values (README rounding, 6e-5); estimate and SE also carry the rounding of the inputs
# 0.0973 and 0.0232 (the fit gives 0.097316 and 0.023257: 0.25% on the SE), so 2e-4 there.
check("published Darfur table (sensemakr README): R2dz.x, R2yz.dx, adjusted estimate and SE for 1x/2x/3x female",
      all(abs(r2dz_p[i] - p[1]) < 6e-5 and abs(r2yz_p[i] - p[2]) < 6e-5 and abs(adj_p[i] - p[3]) < 2e-4
          and abs(se_p[i] - p[4]) < 2e-4 for i, p in enumerate(PUB)),
      [(round(a, 4), round(b, 4), round(c, 4), round(s, 4)) for a, b, c, s in zip(r2dz_p, r2yz_p, adj_p, se_p)])
check("Cinelli-Hazlett Figure 2: adjusted t 3.44 / 2.6 / 1.63; §5.2: 77%, 55%, 32% of the estimate",
      np.allclose(t_p, [3.44, 2.60, 1.63], atol=0.01) and np.allclose(adj_p / 0.0973, [0.77, 0.55, 0.32], atol=0.01),
      (np.round(t_p, 3), np.round(adj_p / 0.0973, 3)))

# ------------------------------------------------------------------------------ 2. R's sensemakr
REFP = ROOT / "tools" / "r_reference" / "output" / "ovb_bounds_v26_reference.json"
COLS = [("r2dz.x", "r2dz_x"), ("r2yz.dx", "r2yz_dx"), ("adjusted_estimate", "adjusted_estimate"),
        ("adjusted_se", "adjusted_se"), ("adjusted_t", "adjusted_t"), ("adjusted_lower_CI", "adjusted_lower_ci"),
        ("adjusted_upper_CI", "adjusted_upper_ci")]
if not REFP.exists():
    check("R reference output present (tools/r_reference/output/ovb_bounds_v26_reference.json)", False,
          "run tools/r_reference/ovb_bounds_v26_reference.R on a machine with R")
else:
    REF = json.loads(REFP.read_text(encoding="utf-8"))
    check("R reference: sensemakr 0.1.6 and the simulated data with the committed MD5",
          "sensemakr: 0.1.6" in (REFP.with_suffix(".txt")).read_text(encoding="utf-8")
          and "6f2c9fedb4c7013f2bda41593c8dc118" in (REFP.with_suffix(".txt")).read_text(encoding="utf-8"))
    dar = REF["darfur"]
    tr = dar["treatment"]["directlyharmed"]
    est_d, se_d, df_d = tr["estimate"], tr["se"], dar["df_outcome"]
    check("Darfur: the treatment regression has one more residual df than the outcome regression (784 vs 783)",
          dar["df_treatment"] == df_d + 1 == 784, (dar["df_treatment"], df_d))

    def darfur_r2(bench):
        """Partial R² of a Darfur benchmark from the t-values R exported (t²/(t²+df) in each regression; for the
        occupation group, R's group_partial_r2)."""
        if bench == "occupation":
            return dar["group_r2d_occupation"], dar["group_r2y_occupation"]
        ty, td = dar["outcome_t"][bench]["t"], dar["treatment_t"][bench]["t"]
        return td ** 2 / (td ** 2 + dar["df_treatment"]), ty ** 2 / (ty ** 2 + df_d)

    check("Darfur: partial R² from the exported t-values equal R's partial_r2 (female, pastvoted)",
          all(rel(darfur_r2(b)[1], dar["r2yxj"][b]) < 1e-10 and rel(darfur_r2(b)[0], dar["r2dxj"][b]) < 1e-10
              for b in ("female", "pastvoted")))
    for run_ in dar["runs"]:
        case = run_["case"]
        bench = "occupation" if "occupation" in case else case.split()[1]
        r2d, r2y = darfur_r2(bench)
        kd = np.atleast_1d(np.asarray(run_["kd"], dtype=float))
        ky = np.atleast_1d(np.asarray(run_["ky"], dtype=float))
        if "error" in run_:
            kmax = sp._kd_max(r2d)
            check(f"R {case}: R stops ({run_['error'][:45]}…) and kd = {kd[0]:g} is beyond our largest kd {kmax:.4g}",
                  "Impossible kd value" in run_["error"] and kd[0] >= kmax)
            continue
        r2dz, r2yz_raw = sp._ovb_partial_r2_bound(r2d, r2y, kd, ky)
        r2yz = np.minimum(r2yz_raw, 1.0)
        adj, adj_se, adj_t = sp._ovb_adjust(est_d, se_d, df_d, r2dz, r2yz)
        tc = stats.t.ppf(0.975, df_d)
        mine = {"r2dz_x": r2dz, "r2yz_dx": r2yz, "adjusted_estimate": adj, "adjusted_se": adj_se, "adjusted_t": adj_t,
                "adjusted_lower_ci": adj - tc * adj_se, "adjusted_upper_ci": adj + tc * adj_se}
        good = len(run_["bounds"]) == len(kd)
        worst = 0.0
        for i, row in enumerate(run_["bounds"]):
            for rc, mc in COLS:
                rv = row[rc] if not isinstance(row[rc], list) else row[rc][0]
                if rv is None or not np.isfinite(mine[mc][i]):
                    good = good and (rv is None) == (not np.isfinite(mine[mc][i]))
                    continue
                worst = max(worst, rel(mine[mc][i], rv) if abs(rv) > 1e-12 else abs(mine[mc][i] - rv))
        check(f"R {case}: every column equal to sensemakr (core, from the exported statistics)",
              good and worst < 1e-8, f"max rel. diff {worst:.2e}")
        capped_r = bool(run_["warnings"])
        check(f"R {case}: R warns about r2yz.dx > 1 exactly when we cap", capped_r == bool(np.any(r2yz_raw > 1)),
              (run_["warnings"], np.round(r2yz_raw, 4)))

    # simulated data, the whole function from the fitted model
    simr = REF["sim"]
    check("R sim: same estimate, SE and df as statsmodels",
          rel(M_SIM.params["d"], simr["treatment"]["d"]["estimate"]) < 1e-10
          and rel(M_SIM.bse["d"], simr["treatment"]["d"]["se"]) < 1e-10 and simr["df_outcome"] == M_SIM.df_resid)
    for run_ in simr["runs"]:
        case = run_["case"]
        bench = case.split()[1]
        kd = np.atleast_1d(np.asarray(run_["kd"], dtype=float))
        ky = np.atleast_1d(np.asarray(run_["ky"], dtype=float))
        mine = sp.ovb_bounds(M_SIM, "d", [bench], kd=list(kd), ky=list(ky))["bounds"]
        worst = 0.0
        for i, row in enumerate(run_["bounds"]):
            for rc, mc in COLS:
                rv = row[rc] if not isinstance(row[rc], list) else row[rc][0]
                worst = max(worst, rel(mine[mc].iloc[i], rv))
        check(f"R {case}: ovb_bounds(res, …) equal to sensemakr on every column", len(mine) == len(run_["bounds"])
              and worst < 1e-8, f"max rel. diff {worst:.2e}")
        labs_r = [row["bound_label"] if not isinstance(row["bound_label"], list) else row["bound_label"][0]
                  for row in run_["bounds"]]
        check(f"R {case}: bound labels as sensemakr's", list(mine["bound_label"]) == labs_r,
              (list(mine["bound_label"]), labs_r))

# ------------------------------------------------------------------------------ 3. known parameter: refit with Z
designs = {}
rng = np.random.default_rng(26)
for name, b_true, n in [("pos", 0.8, 500), ("neg", -0.4, 900)]:
    x1, x2, u = rng.normal(size=(3, n))
    dd = 0.5 * x1 - 0.3 * x2 + 0.4 * u + rng.normal(size=n)
    yy = b_true * dd + 0.7 * x1 + 0.2 * x2 + 0.5 * u + rng.normal(size=n)
    designs[name] = smf.ols("y ~ d + x1 + x2", pd.DataFrame({"y": yy, "d": dd, "x1": x1, "x2": x2})).fit()
cases = [("sim", M_SIM, "d", ["x_strong", "g", "x_weak"], [1, 2, 3]),
         ("pos", designs["pos"], "d", ["x1", "x2"], [0.5, 1, 2]),
         ("neg", designs["neg"], "d", ["x1", "x2"], [1, 2])]
for name, res, tr_, benches, ks in cases:
    out = sp.ovb_bounds(res, tr_, benches, kd=ks)
    b = out["bounds"]
    tc = stats.t.ppf(0.975, res.df_resid)
    worst_e = worst_s = worst_away = 0.0
    for _, r in b[~b["r2yz_dx_capped"]].iterrows():
        toward, away, se1, se2 = both_refits(res, tr_, r["r2dz_x"], r["r2yz_dx"])
        worst_e = max(worst_e, rel(toward, r["adjusted_estimate"]))
        worst_away = max(worst_away, rel(away, 2 * out["estimate"] - r["adjusted_estimate"]))
        worst_s = max(worst_s, rel(se1, r["adjusted_se"]), rel(se2, r["adjusted_se"]))
    check(f"known parameter ({name}): refit with the built Z gives the adjusted estimate ({len(b)} rows)",
          worst_e < 1e-8, f"max rel. diff {worst_e:.1e}")
    check(f"known parameter ({name}): Z of the other sign pushes the estimate away by the same bias", worst_away < 1e-8,
          f"{worst_away:.1e}")
    check(f"known parameter ({name}): refit SE (df − 1, Z costs one df) equals adjusted_se, and CI = est ± t(df)·SE",
          worst_s < 1e-8 and np.allclose(b["adjusted_lower_ci"], b["adjusted_estimate"] - tc * b["adjusted_se"])
          and np.allclose(b["adjusted_t"], b["adjusted_estimate"] / b["adjusted_se"]), f"{worst_s:.1e}")
    # the built Z really has the partial R² claimed (the construction, checked by refits)
    r = b.iloc[0]
    z = build_z(res, tr_, r["r2dz_x"], r["r2yz_dx"])
    X = res.model.exog
    j = list(res.params.index).index(tr_)
    Xo = np.delete(X, j, axis=1)
    ssr = lambda yv, A: float(np.sum(resid(yv, A) ** 2))                      # noqa: E731
    r2dz_chk = 1 - ssr(X[:, j], np.column_stack([Xo, z])) / ssr(X[:, j], Xo)
    r2yz_chk = 1 - ssr(res.model.endog, np.column_stack([X, z])) / ssr(res.model.endog, X)
    check(f"known parameter ({name}): the built Z has R²(D~Z|X) and R²(Y~Z|X,D) as claimed, and is ⊥ X",
          abs(r2dz_chk - r["r2dz_x"]) < 1e-10 and abs(r2yz_chk - r["r2yz_dx"]) < 1e-10
          and np.abs(Xo.T @ z).max() < 1e-8 * np.abs(Xo).sum(), (r2dz_chk, r["r2dz_x"]))

# ------------------------------------------------------------------------------ 4. benchmark R² by refits
ssr_of = lambda f: smf.ols(f, SIM).fit().ssr                                  # noqa: E731
br = sp.ovb_bounds(M_SIM, "d", ["x_strong", "g"], kd=1)["benchmark_r2"]
r2y_x = 1 - M_SIM.ssr / ssr_of("y ~ d + x_weak + C(g)")
r2d_x = 1 - ssr_of("d ~ x_strong + x_weak + C(g)") / ssr_of("d ~ x_weak + C(g)")
r2y_g = 1 - M_SIM.ssr / ssr_of("y ~ d + x_strong + x_weak")
r2d_g = 1 - ssr_of("d ~ x_strong + x_weak + C(g)") / ssr_of("d ~ x_strong + x_weak")
check("benchmark R² (single): outcome and treatment regressions equal the refit SSR identity",
      rel(br.loc["x_strong", "r2yxj_dx"], r2y_x) < 1e-10 and rel(br.loc["x_strong", "r2dxj_x"], r2d_x) < 1e-10,
      (br.loc["x_strong", "r2yxj_dx"], r2y_x))
check("benchmark R² (group of 3 dummies): both regressions equal the refit SSR identity",
      rel(br.loc["g", "r2yxj_dx"], r2y_g) < 1e-10 and rel(br.loc["g", "r2dxj_x"], r2d_g) < 1e-10,
      (br.loc["g", "r2dxj_x"], r2d_g))
check("benchmark_r2 reports df_treatment = df_outcome + 1 and the largest kd = (1 − R²_D)/R²_D",
      (br["df_treatment"] == br["df_outcome"] + 1).all()
      and np.allclose(br["kd_max"], (1 - br["r2dxj_x"]) / br["r2dxj_x"]))

# ------------------------------------------------------------------------------ 5. k_needed
warnings.simplefilter("ignore", UserWarning)     # many k_needed land where R²_Y is capped; the cap is tested in §6
for name, res, tr_, benches, _ in cases:
    for q in (1, 0.5, 2):
        out = sp.ovb_bounds(res, tr_, benches, kd=1, q=q)
        est = out["estimate"]
        for lab, row in out["benchmark_r2"].iterrows():
            k = row["k_needed"]
            at = sp.ovb_bounds(res, tr_, {lab: row["terms"]}, kd=k)["bounds"].iloc[0]
            target = (1 - q) * est
            fine = abs(at["adjusted_estimate"] - target) < 1e-9 * abs(est)
            if not at["r2yz_dx_capped"]:
                toward, _, _, _ = both_refits(res, tr_, at["r2dz_x"], at["r2yz_dx"])
                fine = fine and abs(toward - target) < 1e-8 * abs(est)
            lo = sp.ovb_bounds(res, tr_, {lab: row["terms"]}, kd=k * (1 - 1e-6))["bounds"].iloc[0]
            hi = sp.ovb_bounds(res, tr_, {lab: row["terms"]}, kd=min(k * (1 + 1e-6), row["kd_max"] * (1 - 1e-9)))
            hi = hi["bounds"].iloc[0]
            moved = lambda r: abs(est - r["adjusted_estimate"])                # noqa: E731
            check(f"k_needed ({name}, {lab}, q={q}) = {k:.4g}: lands on (1 − q)·estimate (refit with Z too); "
                  f"just below it the bound falls short, just above it overshoots",
                  fine and moved(lo) < q * abs(est) < moved(hi), (at["adjusted_estimate"], target))
nothing = pd.DataFrame({"y": SIM.y, "d": SIM.d, "x_strong": SIM.x_strong, "zero": 0.0})
nothing["orth"] = resid(np.random.default_rng(3).normal(size=len(SIM)),
                        np.column_stack([np.ones(len(SIM)), SIM.d, SIM.x_strong, SIM.y]))
m_orth = smf.ols("y ~ d + x_strong + orth", nothing).fit()
o_orth, txt_orth, _ = run(sp.ovb_bounds, m_orth, "d", "orth", kd=1, show=True)
check("k_needed = inf for a benchmark that explains nothing (orthogonal to D, X and Y), and the text says no multiple "
      "could do it; its bound leaves the estimate where it is",
      np.isinf(o_orth["benchmark_r2"].loc["orth", "k_needed"]) and "No multiple of orth could" in txt_orth
      and abs(o_orth["bounds"]["adjusted_estimate"].iloc[0] - o_orth["estimate"]) < 1e-12 * abs(o_orth["estimate"]),
      o_orth["benchmark_r2"].loc["orth", ["r2dxj_x", "r2yxj_dx"]].tolist())

warnings.resetwarnings()

# ------------------------------------------------------------------------------ 6. boundaries, both sides
kmax = sp.ovb_bounds(M_SIM, "d", "x_weak", kd=1)["benchmark_r2"].loc["x_weak", "kd_max"]
below = raises(ValueError, sp.ovb_bounds, M_SIM, "d", "x_weak", kd=kmax * (1 - 1e-9))
above = raises(ValueError, sp.ovb_bounds, M_SIM, "d", "x_weak", kd=kmax * (1 + 1e-9))
check(f"kd just below the largest admissible ({kmax:.6g}) runs; just above it is a ValueError that names the limit",
      below is False and isinstance(above, str) and "impossible kd" in above and f"{kmax:.6g}" in above, above)
# r2yz_dx crossing 1: find, on the function's own output, the k where the cap starts (bisection on the flag)
lo_k, hi_k = 1.0, 60.0
assert not sp.ovb_bounds(M_SIM, "d", "x_strong", kd=1, ky=lo_k)["bounds"]["r2yz_dx_capped"].iloc[0]
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    assert sp.ovb_bounds(M_SIM, "d", "x_strong", kd=1, ky=hi_k)["bounds"]["r2yz_dx_capped"].iloc[0]
for _ in range(200):
    mid = 0.5 * (lo_k + hi_k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        capped_mid = sp.ovb_bounds(M_SIM, "d", "x_strong", kd=1, ky=mid)["bounds"]["r2yz_dx_capped"].iloc[0]
    if capped_mid:
        hi_k = mid
    else:
        lo_k = mid
o_lo, t_lo, w_lo = run(sp.ovb_bounds, M_SIM, "d", "x_strong", kd=1, ky=lo_k * (1 - 1e-9))
o_hi, t_hi, w_hi = run(sp.ovb_bounds, M_SIM, "d", "x_strong", kd=1, ky=hi_k * (1 + 1e-9))
rlo, rhi = o_lo["bounds"].iloc[0], o_hi["bounds"].iloc[0]
check("r2yz_dx just below 1: no cap, no warning (neither warnings.warn nor out['warnings'])",
      rlo["r2yz_dx"] < 1 and not rlo["r2yz_dx_capped"] and not w_lo and not o_lo["warnings"], rlo["r2yz_dx"])
check("r2yz_dx just above 1: capped at 1, flagged, warned by warnings.warn and in out['warnings'] with show=False",
      rhi["r2yz_dx"] == 1 and rhi["r2yz_dx_capped"] and len(w_hi) == 1 and o_hi["warnings"] == w_hi
      and "x_strong" in w_hi[0] and "capped at 1" in w_hi[0], w_hi)
o_c, t_c, _ = run(sp.ovb_bounds, M_SIM, "d", "x_strong", kd=1, ky=hi_k * (1 + 1e-9), show=True)
check("the capped line says so, and the warning is printed", "capped at 100%" in t_c and "WARNING:" in t_c)
o_u, t_u, _ = run(sp.ovb_bounds, M_SIM, "d", "x_strong", kd=1, ky=lo_k * (1 - 1e-9), show=True)
check("an uncapped line does not say 'capped'", "capped" not in t_u and "WARNING" not in t_u)

kz = sp.ovb_bounds(M_SIM, "d", "x_strong", kd=1)["benchmark_r2"].loc["x_strong", "k_needed"]
_, t_b, _ = run(sp.ovb_bounds, M_SIM, "d", "x_strong", kd=kz * (1 - 1e-6), show=True)
_, t_a, _ = run(sp.ovb_bounds, M_SIM, "d", "x_strong", kd=kz * (1 + 1e-6), show=True)
check("estimate crossing zero: 'toward zero … of it' just before k_needed, 'past zero' just after (and never both)",
      "toward zero" in t_b and "past zero" not in t_b and "past zero" in t_a and "toward zero" not in t_a)


def ci_upper(k):
    return sp.ovb_bounds(M_SIM, "d", "x_strong", kd=k)["bounds"]["adjusted_upper_ci"].iloc[0]


lo_k, hi_k = 1.0, kz                            # estimate < 0: the upper CI bound reaches 0 before the estimate does
for _ in range(200):
    mid = 0.5 * (lo_k + hi_k)
    lo_k, hi_k = (mid, hi_k) if ci_upper(mid) < 0 else (lo_k, mid)
_, t_ex, _ = run(sp.ovb_bounds, M_SIM, "d", "x_strong", kd=lo_k, show=True)
_, t_in, _ = run(sp.ovb_bounds, M_SIM, "d", "x_strong", kd=hi_k * (1 + 1e-9), show=True)
check("CI reaching zero: 'still excludes zero' just before, 'includes zero' just after",
      ci_upper(lo_k) < 0 <= ci_upper(hi_k * (1 + 1e-9)) and "still excludes zero" in t_ex
      and "which includes zero" in t_in and "still excludes" not in t_in, (ci_upper(lo_k), ci_upper(hi_k)))

# ------------------------------------------------------------------------------ 7. the sentences state the row
NUM = r"(-?[0-9.]+(?:e[-+]?\d+)?)"
for name, res, tr_, benches, ks in cases + [("sim, alpha 0.001", M_SIM, "d", ["x_strong", "g"], [1, 2.5, 3])]:
    alpha = 0.001 if "0.001" in name else 0.05
    out, txt, _ = run(sp.ovb_bounds, res, tr_, benches, kd=ks, alpha=alpha, show=True)
    lines = [ln.strip() for ln in txt.splitlines() if re.match(r"^\s+[0-9.]+(/[0-9.]+)?x ", ln)]
    good = len(lines) == len(out["bounds"])
    est = out["estimate"]
    for ln, (_, r) in zip(lines, out["bounds"].iterrows()):
        good &= ln.startswith(r["bound_label"] + ":")
        m = re.search(rf"CI \[{NUM}, {NUM}\], which (still excludes zero|includes zero)", ln)
        lo_, hi_ = float(m.group(1)), float(m.group(2))
        good &= rel(lo_, r["adjusted_lower_ci"]) < 1e-3 and rel(hi_, r["adjusted_upper_ci"]) < 1e-3
        good &= (m.group(3) == "still excludes zero") == (lo_ > 0 or hi_ < 0)
        a = float(re.search(rf"(?:to|, to) {NUM}(?: \(|;)", ln).group(1))
        good &= rel(a, r["adjusted_estimate"]) < 1e-3
        crossed = (a < 0) != (est < 0)
        good &= ("past zero" in ln) == crossed and ("toward zero" in ln) == (not crossed)
        if not crossed:
            share = float(re.search(r"\((\d+)% of it\)", ln).group(1))
            good &= abs(share - 100 * a / est) <= 0.5 + 1e-9
        pr = re.search(rf"partial R² up to {NUM}% with {tr_} and {NUM}% with the outcome", ln)
        good &= abs(float(pr.group(1)) - 100 * r["r2dz_x"]) <= 0.05 + 1e-9
        good &= abs(float(pr.group(2)) - 100 * r["r2yz_dx"]) <= 0.05 + 1e-9
    level = f"{round((1 - alpha) * 100, 6):g}%"
    check(f"text ({name}): each line restates its row (label, estimate, share, R², CI, verdict on zero)", good,
          f"{len(lines)} lines")
    check(f"text ({name}): the level is '{level} CI' (never '100%')",
          txt.count(f"{level} CI") == len(out["bounds"]) and "100% CI" not in txt)
    check(f"text ({name}): the caution (orthogonal confounder, worst case, explained variance) is printed once",
          txt.count("worst case for a confounder orthogonal to the covariates") == 1
          and "not in coefficient size" in txt)
    kn = re.findall(rf"A confounder {NUM}x as strong as (\S+) \(same k", txt)
    check(f"text ({name}): the k_needed lines restate benchmark_r2['k_needed']",
          len(kn) == len(out["benchmark_r2"]) and all(
              rel(float(k), out["benchmark_r2"].loc[lab, "k_needed"]) < 5e-3 for k, lab in kn), kn)
_, t_q, _ = run(sp.ovb_bounds, M_SIM, "d", "x_strong", kd=1, q=0.5, show=True)
_, t_q1, _ = run(sp.ovb_bounds, M_SIM, "d", "x_strong", kd=1, q=1.0, show=True)
check("k_needed text: q = 0.5 says 'reduce the estimate by 50% (to 0.5 times its value)'; q = 1 'bring the estimate "
      "to zero'", "would reduce the estimate by 50% (to 0.5 times its value)" in t_q
      and "would bring the estimate to zero" in t_q1 and "reduce the estimate by" not in t_q1)
_, t_kdky, _ = run(sp.ovb_bounds, M_SIM, "d", "x_weak", kd=[1, 3], ky=[2, 0.5], show=True)
check("kd ≠ ky: label '1/2x x_weak' and the sentence names both multipliers",
      "1/2x x_weak: a confounder 1x as strong as x_weak in explaining d and 2x in explaining the outcome" in t_kdky
      and "3/0.5x x_weak" in t_kdky)

# ------------------------------------------------------------------------------ 8. validation, robust, units, groups
bad = [("kd = 0", dict(kd=0), ValueError), ("kd < 0", dict(kd=[1, -1]), ValueError), ("kd nan", dict(kd=np.nan), ValueError),
       ("kd inf", dict(kd=[np.inf]), ValueError), ("kd bool", dict(kd=True), TypeError), ("kd str", dict(kd="2"), TypeError),
       ("kd empty", dict(kd=[]), ValueError), ("ky = 0", dict(ky=0), ValueError),
       ("kd/ky lengths 3 vs 2", dict(kd=[1, 2, 3], ky=[1, 2]), ValueError), ("q = 0", dict(q=0), ValueError),
       ("alpha = 1", dict(alpha=1.0), ValueError)]
for lab, kw, exc in bad:
    check(f"validation: {lab} -> {exc.__name__}", raises(exc, sp.ovb_bounds, M_SIM, "d", "x_weak", **kw))
check("validation: unknown benchmark -> KeyError listing the regressors",
      "regressors" in str(raises(KeyError, sp.ovb_bounds, M_SIM, "d", "nope")))
check("validation: the treatment as its own benchmark -> ValueError",
      raises(ValueError, sp.ovb_bounds, M_SIM, "d", {"self": ["d"]}))
check("validation: the intercept as a benchmark -> KeyError",
      raises(KeyError, sp.ovb_bounds, M_SIM, "d", {"c": ["Intercept"]}))
check("validation: unknown treatment -> KeyError", raises(KeyError, sp.ovb_bounds, M_SIM, "dd", "x_weak"))
wls = smf.wls(F_SIM, SIM, weights=np.linspace(1, 2, len(SIM))).fit()
check("validation: a WLS fit -> TypeError (the bounds are for OLS)", raises(TypeError, sp.ovb_bounds, wls, "d", "x_weak"))
o_one = sp.ovb_bounds(M_SIM, "d", "x_weak", kd=[1, 2], ky=3)["bounds"]
check("ky as one number pairs with every kd", list(o_one["ky"]) == [3.0, 3.0] and list(o_one["kd"]) == [1.0, 2.0])

hc = smf.ols(F_SIM, SIM).fit(cov_type="HC3")
o_hc, t_hc, _ = run(sp.ovb_bounds, hc, "d", ["x_strong", "g"], show=True)
o_cl = sp.ovb_bounds(M_SIM, "d", ["x_strong", "g"])
check("robust fit (HC3): same numbers as the classical fit, classical_refit flagged, and the classical-SE line printed",
      np.allclose(o_hc["bounds"].select_dtypes("number"), o_cl["bounds"].select_dtypes("number"), rtol=1e-12)
      and o_hc["classical_refit"] and not o_cl["classical_refit"]
      and "These bounds use the classical SE" in t_hc and "HC3 fit gives t =" in t_hc)

scaled = SIM.assign(y=SIM.y * 1000.0, d=SIM.d * 0.01)
m_sc = smf.ols(F_SIM, scaled).fit()
o_sc, t_sc, _ = run(sp.ovb_bounds, m_sc, "d", ["x_strong", "g"], kd=[1, 3], q=0.5, show=True)
o_or, t_or, _ = run(sp.ovb_bounds, M_SIM, "d", ["x_strong", "g"], kd=[1, 3], q=0.5, show=True)
same_cols = ["r2dz_x", "r2yz_dx", "adjusted_t", "share_of_estimate"]
strip = lambda s: re.sub(NUM, "#", s)                                         # noqa: E731
check("equivariance: y×1000 and d×0.01 leave R², t, shares, k_needed, verdicts and the sentences' words unchanged",
      np.allclose(o_sc["bounds"][same_cols], o_or["bounds"][same_cols], rtol=1e-9)
      and np.allclose(o_sc["benchmark_r2"]["k_needed"], o_or["benchmark_r2"]["k_needed"], rtol=1e-9)
      and (o_sc["bounds"]["ci_excludes_zero"] == o_or["bounds"]["ci_excludes_zero"]).all()
      and strip(t_sc) == strip(t_or)
      and np.allclose(o_sc["bounds"]["adjusted_estimate"], o_or["bounds"]["adjusted_estimate"] * 1e5, rtol=1e-9))

g1 = sp.ovb_bounds(M_SIM, "d", "g", kd=[1, 2])
g2 = sp.ovb_bounds(M_SIM, "d", "C(g)", kd=[1, 2])
g3 = sp.ovb_bounds(M_SIM, "d", {"g": ["C(g)[T.b]", "C(g)[T.c]", "C(g)[T.d]"]}, kd=[1, 2])
num = lambda o: o["bounds"].select_dtypes("number").to_numpy()                # noqa: E731
sens = sp.sensitivity(M_SIM, "d")["benchmarks"]
check("grouping: 'g', 'C(g)' and the explicit dict of dummies give the same bounds; the group's R² with the outcome "
      "equals the one sensitivity prints for 'C(g) (3 dummies, jointly)'",
      np.allclose(num(g1), num(g2)) and np.allclose(num(g1), num(g3))
      and g1["benchmark_r2"].loc["g", "terms"] == ["C(g)[T.b]", "C(g)[T.c]", "C(g)[T.d]"]
      and rel(sens["C(g) (3 dummies, jointly)"], g1["benchmark_r2"].loc["g", "r2yxj_dx"]) < 1e-10)

_, t_sens, _ = run(sp.sensitivity, M_SIM, "d", show=True)
last = t_sens.strip().splitlines()[-1]
check("sensitivity's show ends with the pointer to ovb_bounds, with the treatment's name",
      last == 'For formal bounds (a confounder k times as strong as one of these, with the adjusted estimate and CI), '
              'run specification.ovb_bounds(res, "d", benchmarks=[...]).', last)
lone = smf.ols("y ~ d", SIM).fit()
_, t_lone, _ = run(sp.sensitivity, lone, "d", show=True)
check("sensitivity without covariates prints no pointer (there is nothing to benchmark)", "ovb_bounds" not in t_lone)
check("sensitivity's docstring points to ovb_bounds and no longer sends to R", "ovb_bounds" in sp.sensitivity.__doc__
      and "sensemakr (R" not in sp.sensitivity.__doc__)

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
