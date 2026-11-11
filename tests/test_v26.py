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
def rnum(v):
    """A number from R's JSON: jsonlite writes Inf/-Inf/NaN as null or as strings, depending on its options."""
    v = v[0] if isinstance(v, list) else v
    if v is None:
        return np.nan
    return float({"Inf": "inf", "-Inf": "-inf", "NaN": "nan", "NA": "nan"}.get(v, v)) if isinstance(v, str) else float(v)


def same(mine, theirs, tol=1e-8):
    """Equal to R: both non-finite (R's null stands for any of inf/nan), or within tol (relative, absolute near 0)."""
    if not np.isfinite(theirs) or not np.isfinite(mine):
        return (not np.isfinite(mine)) and (not np.isfinite(theirs) or np.isnan(theirs))
    return (rel(mine, theirs) if abs(theirs) > 1e-12 else abs(mine - theirs)) < tol


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
        if "error" in run_:
            kmax = sp._kd_max(r2d)
            check(f"R {case}: R stops ({run_['error'][:45]}…) and kd = {kd[0]:g} is beyond our largest kd {kmax:.4g}",
                  "Impossible kd value" in run_["error"] and kd[0] >= kmax)
            continue
        ky = np.atleast_1d(np.asarray(run_["ky"], dtype=float))
        r2dz, r2yz_raw = sp._ovb_partial_r2_bound(r2d, r2y, kd, ky)
        r2yz = np.minimum(r2yz_raw, 1.0)
        adj, adj_se, adj_t = sp._ovb_adjust(est_d, se_d, df_d, r2dz, r2yz)
        tc = stats.t.ppf(0.975, df_d)
        mine = {"r2dz_x": r2dz, "r2yz_dx": r2yz, "adjusted_estimate": adj, "adjusted_se": adj_se, "adjusted_t": adj_t,
                "adjusted_lower_ci": adj - tc * adj_se, "adjusted_upper_ci": adj + tc * adj_se}
        good = len(run_["bounds"]) == len(kd)
        bad_cells = [(i, mc, float(mine[mc][i]), rnum(row[rc])) for i, row in enumerate(run_["bounds"])
                     for rc, mc in COLS if not same(float(mine[mc][i]), rnum(row[rc]))]
        check(f"R {case}: every column equal to sensemakr (core, from the exported statistics)",
              good and not bad_cells, bad_cells[:3])
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
        bad_cells = [(i, mc, float(mine[mc].iloc[i]), rnum(row[rc])) for i, row in enumerate(run_["bounds"])
                     for rc, mc in COLS if not same(float(mine[mc].iloc[i]), rnum(row[rc]))]
        check(f"R {case}: ovb_bounds(res, …) equal to sensemakr on every column", len(mine) == len(run_["bounds"])
              and not bad_cells, bad_cells[:3])
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
          worst_s < 1e-8
          and np.allclose(b["adjusted_lower_ci"], b["adjusted_estimate"] - tc * b["adjusted_se"], rtol=1e-12, atol=0)
          and np.allclose(b["adjusted_upper_ci"], b["adjusted_estimate"] + tc * b["adjusted_se"], rtol=1e-12, atol=0)
          and np.allclose(b["adjusted_t"], b["adjusted_estimate"] / b["adjusted_se"], rtol=1e-12, atol=0),
          f"{worst_s:.1e}")
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

# Cinelli-Hazlett eq. (22): R²_{D~Z|X} = kD·f²_{D~Xj|X−j}, with f² = R²/(1 − R²) — the bound with D depends on kd
# only (never on ky), and grows with ky the bound with Y only; R² from the refit above, not from the function.
kk = sp.ovb_bounds(M_SIM, "d", "x_strong", kd=[1, 1, 2, 0.5], ky=[1, 3, 0.5, 2])["bounds"]
f2d = r2d_x / (1 - r2d_x)
check("eq. (22): r2dz_x = kd·f²_D from the refit R² (kd = 1, 1, 2, 0.5 with ky = 1, 3, 0.5, 2), whatever ky is",
      np.allclose(kk["r2dz_x"], np.array([1, 1, 2, 0.5]) * f2d, rtol=1e-10, atol=0), list(kk["r2dz_x"]))
check("for the same kd, a larger ky gives a larger bound with the outcome, and the same bound with the treatment",
      kk["r2yz_dx"].iloc[1] > kk["r2yz_dx"].iloc[0] and kk["r2dz_x"].iloc[1] == kk["r2dz_x"].iloc[0])

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
        m = re.search(rf"CI \[{NUM}, {NUM}\], which (still excludes zero|excludes zero \(the unadjusted CI did not\)|"
                      rf"includes zero|excludes zero on the other side \(the sign would flip\))", ln)
        if r["r2yz_dx_capped"]:
            good &= m is None and "its CI is not informative" in ln and "capped at 100%" in ln
        else:
            lo_, hi_ = float(m.group(1)), float(m.group(2))
            good &= rel(lo_, r["adjusted_lower_ci"]) < 1e-3 and rel(hi_, r["adjusted_upper_ci"]) < 1e-3
            same_side = (lo_ > 0 and est > 0) or (hi_ < 0 and est < 0)
            other_side = (lo_ > 0 and est < 0) or (hi_ < 0 and est > 0)
            tc0 = stats.t.ppf(1 - alpha / 2, res.df_resid)
            orig_excl = abs(est) > tc0 * res.bse[tr_]
            expect = ("includes zero" if not (same_side or other_side) else
                      "excludes zero on the other side (the sign would flip)" if other_side else
                      "still excludes zero" if orig_excl else "excludes zero (the unadjusted CI did not)")
            good &= m.group(3) == expect
        a = float(re.search(rf"(?:to|, to) {NUM}(?: \(|;)", ln).group(1))
        good &= rel(a, r["adjusted_estimate"]) < 1e-3
        crossed = (a < 0) != (est < 0)
        good &= ("past zero" in ln) == crossed and ("toward zero" in ln) == (not crossed)
        if not crossed:
            sh = re.search(r"\((less than 1|more than 99|\d+)% of it\)", ln).group(1)
            v = 100 * r["adjusted_estimate"] / est
            good &= (0 < v < 0.5 if sh == "less than 1" else 99.5 <= v < 100 if sh == "more than 99"
                     else abs(float(sh) - v) <= 0.5 + 1e-9)
        pr = re.search(rf"partial R² up to {NUM}% with {tr_} and {NUM}% with the outcome", ln)
        good &= abs(float(pr.group(1)) - 100 * r["r2dz_x"]) <= 0.05 + 1e-9
        good &= abs(float(pr.group(2)) - 100 * r["r2yz_dx"]) <= 0.05 + 1e-9
    level = f"{round((1 - alpha) * 100, 6):g}%"
    check(f"text ({name}): each line restates its row (label, estimate, share, R², CI, verdict on zero)", good,
          f"{len(lines)} lines")
    check(f"text ({name}): the level is '{level} CI' (never '100%')",
          txt.count(f"{level} CI") == 1 + int((~out["bounds"]["r2yz_dx_capped"]).sum()) and "100% CI" not in txt)
    check(f"text ({name}): the caution (orthogonal confounder, worst case, explained variance) is printed once",
          txt.count("worst case for a confounder orthogonal to the covariates") == 1
          and "not in coefficient size" in txt and "can exceed k times the benchmark's" in txt)
    # v2.6.1: the k_not_significant sentence starts the same way; this check is about the k_needed sentence only
    # v2.6.2: past k_cap the sentence says "in explaining the treatment … only with the outcome fully explained"
    kn = re.findall(rf"A confounder {NUM}x as strong as (\S+) (?:\(same k for treatment and outcome\)|in explaining "
                    rf"the treatment) could, in the worst case, (?:bring|reduce)", txt)
    check(f"text ({name}): a k_needed line beyond the cap exactly where benchmark_r2 says so (v2.6.2)",
          txt.count("only with the outcome fully explained") == int(out["benchmark_r2"]["k_needed_beyond_cap"].sum()))
    check(f"text ({name}): the k_needed lines restate benchmark_r2['k_needed']",
          len(kn) == len(out["benchmark_r2"]) and all(
              rel(float(k), out["benchmark_r2"].loc[lab, "k_needed"]) < 5e-3 for k, lab in kn), kn)
_, t_q, _ = run(sp.ovb_bounds, M_SIM, "d", "x_strong", kd=1, q=0.5, show=True)
_, t_q1, _ = run(sp.ovb_bounds, M_SIM, "d", "x_strong", kd=1, q=1.0, show=True)
check("k_needed text: q = 0.5 says 'reduce the estimate by 50% (to 0.5 times its value)'; q = 1 'bring the estimate "
      "to zero'; both as a worst case ('could, in the worst case'), never 'would'",
      "could, in the worst case, reduce the estimate by 50% (to 0.5 times its value)" in t_q
      and "could, in the worst case, bring the estimate to zero" in t_q1 and "reduce the estimate by" not in t_q1
      and "would bring" not in t_q1)
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


# ------------------------------------------------------------------------------ 9. findings of the v2.6 review
rg9 = np.random.default_rng(900)
n9 = 600
x9, w9 = rg9.normal(size=(2, n9))
d9 = 0.5 * x9 + rg9.normal(size=n9)                          # true effect 0; design found by search (seed 900)
null_df = pd.DataFrame({"y": 0.0 * d9 + 0.3 * x9 + 0.3 * w9 + rg9.normal(size=n9), "d": d9, "x": x9, "w": w9})
m_null = smf.ols("y ~ d + x + w", null_df).fit()
lo0, hi0 = m_null.conf_int().loc["d"]
o9, t9, _ = run(sp.ovb_bounds, m_null, "d", "x", kd=[1, 2], show=True)
crossed_rows = o9["bounds"][np.sign(o9["bounds"]["adjusted_estimate"]) != np.sign(o9["estimate"])]
assert not o9["bounds"]["r2yz_dx_capped"].any()
check("BLOCKER of round 1: original CI includes zero -> no line says 'still excludes zero'; a CI on the other side of "
      "zero says so ('the sign would flip')",
      lo0 < 0 < hi0 and "still excludes" not in t9 and len(crossed_rows) > 0
      and all((v == "opposite sign") == (r_lo > 0 or r_hi < 0) for v, r_lo, r_hi in
              zip(crossed_rows["ci_verdict"], crossed_rows["adjusted_lower_ci"], crossed_rows["adjusted_upper_ci"]))
      and ("the sign would flip" in t9) == (crossed_rows["ci_verdict"] == "opposite sign").any(),
      (round(lo0, 3), round(hi0, 3), list(o9["bounds"]["ci_verdict"])))
# original CI includes zero, adjusted CI excludes it on the same side (the adjusted SE shrinks: large R² with Y, tiny
# with D): the text must not say "still"
osm, tsm, _ = run(sp.ovb_bounds, m_null, "d", "w", kd=0.001, ky=11, show=True)
rsm = osm["bounds"].iloc[0]
check("unadjusted CI includes zero, adjusted CI excludes it on the same side: 'excludes zero (the unadjusted CI did "
      "not)', never 'still'", lo0 < 0 < hi0 and rsm["ci_verdict"] == "same sign" and not rsm["r2yz_dx_capped"]
      and "excludes zero (the unadjusted CI did not)" in tsm and "still" not in tsm,
      (rsm["adjusted_lower_ci"], rsm["adjusted_upper_ci"]))
# a significant estimate whose bound crosses zero with a CI wholly on the other side
o9b = sp.ovb_bounds(M_SIM, "d", "x_strong", kd=[1, 2, 3, 3.6])
r36 = o9b["bounds"].iloc[3]
_, t9b, _ = run(sp.ovb_bounds, M_SIM, "d", "x_strong", kd=[3.6], show=True)
check("crossed and the CI wholly on the other side: verdict 'opposite sign', text 'on the other side', no 'still'",
      r36["adjusted_lower_ci"] > 0 > o9b["estimate"] and r36["ci_verdict"] == "opposite sign"
      and "excludes zero on the other side (the sign would flip)" in t9b and "still" not in t9b, r36["ci_verdict"])
check("ci_verdict column: 'same sign' / 'includes zero' / 'opposite sign' as the CI and the estimate say (x_strong 1-3)",
      list(o9b["bounds"]["ci_verdict"]) == ["same sign", "same sign", "includes zero", "opposite sign"],
      list(o9b["bounds"]["ci_verdict"]))
# capped rows: CI not informative, no verdict
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    oc = sp.ovb_bounds(M_SIM, "d", "x_strong", kd=1, ky=30)
_, tc9, _ = run(sp.ovb_bounds, M_SIM, "d", "x_strong", kd=1, ky=30, show=True)
check("capped row: ci_verdict 'not informative', ci_excludes_zero False, the line says the CI is not informative and "
      "gives no verdict on zero",
      oc["bounds"]["ci_verdict"].iloc[0] == "not informative" and not oc["bounds"]["ci_excludes_zero"].iloc[0]
      and "its CI is not informative" in tc9
      and "excludes zero" not in tc9.split("\n  1/30x")[1].split("\n")[0]
      and "includes zero" not in tc9.split("\n  1/30x")[1].split("\n")[0])
# the header: "k times" given the other covariates; the printed R² given all of them can exceed k times
br9 = o9b["benchmark_r2"].loc["x_strong"]
r1 = o9b["bounds"].iloc[0]
check("header is true of the numbers: at kd = 1 the printed R² with D exceeds the benchmark's (given all covariates), "
      "and the header says it can", r1["r2dz_x"] > br9["r2dxj_x"] and "can exceed k times the benchmark's" in t9b,
      (r1["r2dz_x"], br9["r2dxj_x"]))
# default kd skips impossible values; explicit kd stops, naming the way out
xs = np.random.default_rng(5).normal(size=400)
dds = xs + 1.2 * np.random.default_rng(6).normal(size=400)
ds = pd.DataFrame({"y": dds + xs + np.random.default_rng(7).normal(size=400), "d": dds, "x": xs,
                   "z": np.random.default_rng(8).normal(size=400)})
ms = smf.ols("y ~ d + x + z", ds).fit()
od, td, wd = run(sp.ovb_bounds, ms, "d", ["x", "z"], show=True)
kmx = od["benchmark_r2"].loc["x", "kd_max"]
check("default kd = (1, 2, 3): the impossible kd of a strong benchmark are skipped with a warning, the other benchmark "
      "keeps all three", 1 < kmx < 2 and list(od["bounds"].query("benchmark == 'x'")["kd"]) == [1.0]
      and list(od["bounds"].query("benchmark == 'z'")["kd"]) == [1.0, 2.0, 3.0]
      and any("skipped" in w and "x" in w for w in od["warnings"]) and "WARNING: x: kd = 2, 3 skipped" in td, kmx)
e9 = raises(ValueError, sp.ovb_bounds, ms, "d", ["x", "z"], kd=[1, 2])
check("explicit impossible kd stops the call and says how to go on", isinstance(e9, str) and "drop that benchmark" in e9
      and "lower kd" in e9, e9)
check("kd exactly at the largest admissible value is impossible (>=, as sensemakr's r2dz.x >= 1)",
      raises(ValueError, sp.ovb_bounds, ms, "d", "x", kd=kmx))
# rank-deficient design
rd = SIM.assign(x3=2 * SIM.x_strong + SIM.x_weak)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")                 # statsmodels warns about the singular design when fitting it
    m_rd = smf.ols("y ~ d + x_strong + x_weak + x3", rd).fit()
check("a rank-deficient design is a ValueError (partial R² not defined), not minimum-norm numbers",
      "full rank" in str(raises(ValueError, sp.ovb_bounds, m_rd, "d", "x3")))
# grouping: interactions stay apart, whatever the formula's order; string categoricals; Treatment(reference)
ga = sp.ovb_bounds(smf.ols("y ~ d + C(g)*x_weak", SIM).fit(), "d", "g", kd=1)["benchmark_r2"].loc["g", "terms"]
gb = sp.ovb_bounds(smf.ols("y ~ d + x_weak*C(g)", SIM).fit(), "d", "g", kd=1)["benchmark_r2"].loc["g", "terms"]
sa = sp.sensitivity(smf.ols("y ~ d + C(g)*x_weak", SIM).fit(), "d")["benchmarks"]
sb_ = sp.sensitivity(smf.ols("y ~ d + x_weak*C(g)", SIM).fit(), "d")["benchmarks"]
ia = [k for k in sa.index if ":" in k]
ib = [k for k in sb_.index if ":" in k]
check("grouping: C(g)*x and x*C(g) give the same 3-dummy group of C(g) and one 3-dummy group for the interaction term "
      "(same joint R² in both orders), in ovb_bounds and sensitivity",
      ga == gb == ["C(g)[T.b]", "C(g)[T.c]", "C(g)[T.d]"] and "C(g) (3 dummies, jointly)" in sa.index
      and "C(g) (3 dummies, jointly)" in sb_.index and len(sa) == len(sb_) == 3
      and ia == ["C(g):x_weak (3 dummies, jointly)"] and ib == ["x_weak:C(g) (3 dummies, jointly)"]
      and abs(sa[ia[0]] - sb_[ib[0]]) < 1e-10
      and abs(sa["C(g) (3 dummies, jointly)"] - sb_["C(g) (3 dummies, jointly)"]) < 1e-12, (list(sa.index), list(sb_.index)))
m_int_c = smf.ols("y ~ d + x_weak*C(g, Treatment(reference='c'))", SIM).fit()
s_int_c = sp.sensitivity(m_int_c, "d")["benchmarks"]
ic = [k for k in s_int_c.index if ":" in k]
check("the interaction group's joint R² does not depend on the reference level of the categorical",
      len(ic) == 1 and abs(s_int_c[ic[0]] - sb_[ib[0]]) < 1e-10, (ic, list(s_int_c.index)))
m_str = smf.ols("y ~ d + x_strong + x_weak + g", SIM).fit()
gs = sp.ovb_bounds(m_str, "d", "g", kd=[1, 2])
check("a string categorical without C() (g[T.b]) groups the same way and gives the same bounds as C(g)",
      gs["benchmark_r2"].loc["g", "terms"] == ["g[T.b]", "g[T.c]", "g[T.d]"]
      and np.allclose(num(gs), num(g1), rtol=1e-10)
      and "g (3 dummies, jointly)" in sp.sensitivity(m_str, "d")["benchmarks"].index)
m_ref = smf.ols("y ~ d + x_strong + x_weak + C(g, Treatment(reference='c'))", SIM).fit()
gr = sp.ovb_bounds(m_ref, "d", "g", kd=[1, 2])
check("C(g, Treatment(reference='c')) is reachable as 'g', and the group's R² does not depend on the reference level",
      len(gr["benchmark_r2"].loc["g", "terms"]) == 3 and np.allclose(num(gr), num(g1), rtol=1e-9))
mix = sp.ovb_bounds(M_SIM, "d", ["x_weak", {"G": "C(g)"}, {"both": ["x_weak", "g"]}], kd=1)
check("benchmarks: a list mixing names and dicts; a dict value may be a categorical's name",
      list(mix["benchmark_r2"].index) == ["x_weak", "G", "both"]
      and mix["benchmark_r2"].loc["G", "terms"] == ["C(g)[T.b]", "C(g)[T.c]", "C(g)[T.d]"]
      and len(mix["benchmark_r2"].loc["both", "terms"]) == 4)
# k_needed: a small but real R² gives a finite k; tiny roots are not returned as 0
k_small = sp._k_needed(1e-8, 1e-8, -0.33, 0.037, 793, 1.0)
k_tiny = sp._k_needed(0.2, 0.2, 1e-18, 0.037, 793, 1.0)
check("k_needed: R² = 1e-8 (real, above the zero threshold) gives a finite k; a target of 1e-18 gives a tiny k > 0 "
      "(an absolute xtol of 1e-14 returned 0)", np.isfinite(k_small) and k_small > 0 and 0 < k_tiny < 1e-15,
      (k_small, k_tiny))
check("the R²-zero threshold (64 eps ≈ 1.4e-14) on both sides: R² = 3e-14 gives a finite k, 1e-14 gives inf",
      np.isfinite(sp._k_needed(3e-14, 0.5, 1e-12, 0.037, 793, 1.0))
      and np.isinf(sp._k_needed(1e-14, 0.5, 1e-12, 0.037, 793, 1.0)))
check("k_needed: a target reachable only at kd_max in floating point (estimate 1e12, SE 0.037) gives inf, never "
      "kd_max itself; 1e6 is still reachable just below kd_max",
      np.isinf(sp._k_needed(0.2, 0.2, 1e12, 0.037, 793, 1.0))
      and 3.99 < sp._k_needed(0.2, 0.2, 1e6, 0.037, 793, 1.0) < sp._kd_max(0.2))
check("k_needed: q = 1e-300 converges (more iterations) and lands near 0",
      0 <= sp._k_needed(0.2, 0.2, -0.33, 0.037, 793, 1e-300) < 1e-100)
# explains nothing of only one side: the sentence names the right side
basis = np.column_stack([np.ones(len(SIM)), SIM.d, SIM.x_strong])
only_y = resid(SIM.y.to_numpy(), basis)                       # orthogonal to d (and x): R²_D = 0, R²_Y > 0
e_ = resid(np.random.default_rng(11).normal(size=len(SIM)), np.column_stack([basis, SIM.y]))
dt_ = resid(SIM.d.to_numpy(), np.column_stack([np.ones(len(SIM)), SIM.x_strong]))
only_d = dt_ / np.linalg.norm(dt_) + e_ / np.linalg.norm(e_)  # R²_D > 0; given d and x, only e_, which is ⊥ y
side_df = SIM.assign(only_y=only_y, only_d=only_d)
_, t_oy, _ = run(sp.ovb_bounds, smf.ols("y ~ d + x_strong + only_y", side_df).fit(), "d", "only_y", kd=1, show=True)
_, t_od, _ = run(sp.ovb_bounds, smf.ols("y ~ d + x_strong + only_d", side_df).fit(), "d", "only_d", kd=0.5, show=True)
check("'explains nothing of the treatment' / '… of the outcome' name the side that is zero",
      "only_y explains nothing of the treatment" in t_oy and "only_d explains nothing of the outcome" in t_od,
      (t_oy[-200:], t_od[-200:]))
# kd one number with ky a list; labels rounded to 2 decimals as sensemakr's label_maker
bc = sp.ovb_bounds(M_SIM, "d", "x_weak", kd=1, ky=[1, 2, 3])["bounds"]
check("kd one number pairs with every ky", list(bc["kd"]) == [1.0, 1.0, 1.0] and list(bc["ky"]) == [1.0, 2.0, 3.0]
      and list(bc["bound_label"]) == ["1x x_weak", "1/2x x_weak", "1/3x x_weak"])
check("labels round k to 2 decimals (sensemakr's label_maker)",
      sp.ovb_bounds(M_SIM, "d", "x_weak", kd=1.234)["bounds"]["bound_label"].iloc[0] == "1.23x x_weak")
# an estimate of exactly 0 counts as positive (sensemakr: ifelse(estimate < 0, -1, 1)); zero-bias and tiny shares
a0, _, _ = sp._ovb_adjust(0.0, 0.05, 100, 0.1, 0.1)
check("estimate exactly 0: the bias is subtracted (adjusted < 0), as in sensemakr", float(a0) < 0, a0)
_, t_nb, _ = run(sp.ovb_bounds, m_orth, "d", "orth", kd=1, show=True)
check("a benchmark that explains nothing: the line says the estimate is left where it is (no bias), not 'toward zero'",
      "(no bias)" in t_nb and "toward zero" not in t_nb)
check("shares never print as a misleading 0% or 100%, each threshold on both sides", sp._share_pct(0.001) == "less than 1%"
      and sp._share_pct(0.0049) == "less than 1%" and sp._share_pct(0.0051) == "1%"
      and sp._share_pct(0.9949) == "99%" and sp._share_pct(0.9951) == "more than 99%"
      and sp._share_pct(0.999) == "more than 99%" and sp._share_pct(0.54) == "54%" and sp._share_pct(1.0) == "100%")
_, t_al, _ = run(sp.ovb_bounds, M_SIM, "d", "x_weak", kd=1, alpha=1e-9, show=True)
check("alpha = 1e-9: the level is not printed as 100%", "100% CI" not in t_al and "99.9999999% CI" in t_al)
# treatment in an interaction; intercept as treatment
oi, ti, wi = run(sp.ovb_bounds, smf.ols("y ~ d*x_weak + x_strong", SIM).fit(), "d", "x_strong", kd=1)
check("treatment also in an interaction: warned (bounds are for its coefficient alone)",
      any("interaction" in w for w in oi["warnings"]) and any("interaction" in w for w in wi))
check("the intercept as treatment is a ValueError", raises(ValueError, sp.ovb_bounds, M_SIM, "Intercept", "x_weak"))
# robust fit whose SE is not finite (as statsmodels gives with HC3 and observations of leverage 1, e.g. the singleton
# villages of Darfur): a stand-in result with the same model and an infinite robust SE, so the branch is exercised
class _InfRobust:
    model, cov_type = M_SIM.model, "HC3"
    params = M_SIM.params
    bse = M_SIM.bse.mul(np.inf)
    tvalues = M_SIM.params / bse


_, t_inf, _ = run(sp.ovb_bounds, _InfRobust(), "d", "x_weak", kd=1, show=True)
_, t_fin, _ = run(sp.ovb_bounds, smf.ols(F_SIM, SIM).fit(cov_type="HC3"), "d", "x_weak", kd=1, show=True)
check("robust fit with a non-finite SE: the line says its t is not usable (not 't = 0.00'); a finite one prints its t",
      "not usable" in t_inf and "t = 0.00" not in t_inf and "HC3 fit gives t =" in t_fin and "not usable" not in t_fin)

# ------------------------------------------------------------------------------ 10. findings of review round 2
# a regressor in units of 1e12 is not collinear (rank on the column-scaled design)
big = SIM.assign(big=(SIM.x_weak + 5) * 1e12)
check("full rank judged on scaled columns: a regressor in units of 1e12 runs (as statsmodels fits it)",
      raises(ValueError, sp.ovb_bounds, smf.ols("y ~ d + x_strong + big", big).fit(), "d", "x_strong", kd=1) is False)
# a benchmark with no admissible default k: one printed line says so
xa = np.random.default_rng(3).normal(size=300)
da = xa + 0.3 * np.random.default_rng(4).normal(size=300)
dfa = pd.DataFrame({"y": da + xa + np.random.default_rng(5).normal(size=300), "d": da, "x": xa,
                    "z": np.random.default_rng(6).normal(size=300)})
oa, ta, _ = run(sp.ovb_bounds, smf.ols("y ~ d + x + z", dfa).fit(), "d", ["x", "z"], show=True)
check("default kd with every k impossible for a benchmark: no row for it, and a line says why and what to pass",
      not (oa["bounds"]["benchmark"] == "x").any() and (oa["bounds"]["benchmark"] == "z").sum() == 3
      and "x: no line, because no default k (1, 2, 3) is admissible" in ta and "pass a smaller kd" in ta)
check("alpha below 1e-12 is a ValueError (the CI would print as 100% or infinite)",
      raises(ValueError, sp.ovb_bounds, M_SIM, "d", "x_weak", kd=1, alpha=1e-13)
      and raises(ValueError, sp.ovb_bounds, M_SIM, "d", "x_weak", kd=1, alpha=2e-16))
check("df_resid = 1 is a ValueError, not a ZeroDivisionError",
      "degrees of freedom" in str(raises(ValueError, sp.ovb_bounds, smf.ols("y ~ d + x_strong + x_weak", SIM.iloc[:5]).fit(),
                                         "d", "x_strong", kd=0.1)))
dup = sp.ovb_bounds(M_SIM, "d", {"both": ["x_weak", "x_weak", "g", "C(g)[T.b]"]}, kd=1)
check("a term listed twice in a group counts once (it made the Wald matrix singular)",
      dup["benchmark_r2"].loc["both", "terms"] == ["x_weak", "C(g)[T.b]", "C(g)[T.c]", "C(g)[T.d]"])
check("a nested list is a TypeError that shows the dict to write", "label" in str(
      raises(TypeError, sp.ovb_bounds, M_SIM, "d", [["x_weak", "x_strong"]], kd=1)))
cg_plain = sp.ovb_bounds(m_str, "d", "C(g)", kd=[1, 2])
cg_ref = sp.ovb_bounds(m_ref, "d", "C(g)", kd=[1, 2])
check("'C(g)' also reaches a string factor g[T.b] and C(g, Treatment('c')), with the same bounds as 'g'",
      np.allclose(num(cg_plain), num(g1), rtol=1e-10) and np.allclose(num(cg_ref), num(g1), rtol=1e-9))
# an estimate of exactly 0 (a stand-in result: the same model, the treatment's coefficient set to 0)


class _ZeroEst:
    model, cov_type, df_resid = M_SIM.model, "nonrobust", M_SIM.df_resid
    params = M_SIM.params.copy()
    params["d"] = 0.0
    bse = M_SIM.bse
    tvalues = params / bse

    def cov_params(self):
        return M_SIM.cov_params()


oz, tz, _ = run(sp.ovb_bounds, _ZeroEst(), "d", "x_weak", kd=1, show=True)
check("estimate exactly 0: 'from 0 to …' (bias subtracted, as sensemakr), k_needed 0 and its sentence",
      "could move the estimate from 0 to -" in tz and oz["benchmark_r2"].loc["x_weak", "k_needed"] == 0
      and "The estimate is exactly 0: no confounder is needed" in tz)

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
