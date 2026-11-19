"""Tests of forecasting.py functions added from v2.7.0 on (T4: tests by module): conformal_intervals.

References do not reuse the implementation:
- MAPIE 1.6.0 (BSD-3) SplitConformalRegressor, run outside this repository by
  claude/scripts/v270/ref_mapie_split.py in the development notes, on deterministic data rebuilt here; its
  half-widths are fixed below;
- the split quantile by counting (smallest error with at least ceil((m+1)(1-alpha)) errors at or below it);
- ACI by an explicit loop over time written here, with the level updated when each target is observed;
- known truth: exact finite-sample coverage ceil((m+1)(1-alpha))/(m+1) under i.i.d. errors (Lei et al. 2018,
  Theorem 2.2), and the deterministic long-run bound of Gibbs and Candès (2021, Prop. 4.1) on any sequence;
- Wilson interval against statsmodels' proportion_confint.

Run from the repository root:  python3 tests/test_forecasting.py
"""
import contextlib
import io
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "econometrics" / "scripts"))
import _console  # noqa: E402,F401
import forecasting as fc  # noqa: E402
from statsmodels.stats.proportion import proportion_confint  # noqa: E402

results = []


def check(name, cond, info=""):
    results.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info)


def raises(f, *a, **k):
    try:
        f(*a, **k)
    except ValueError as e:
        return str(e) or True
    return False


def quiet(f, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = f(*a, **k)
    return out, buf.getvalue()


CI = fc.conformal_intervals

# ============================================================ 1. split against MAPIE and against counting
t = np.arange(1, 121)
fcast = np.sin(t / 7.0)
u = np.clip(np.modf(np.abs(np.sin(12.9898 * t) * 43758.5453))[0], 1e-6, 1 - 1e-6)
y = fcast + 0.8 * stats.t.ppf(u, 3)
# the noise goes through scipy's t quantile; older scipy (1.14 in the minimum job) differs in the last digits, so the
# MAPIE comparison is relaxed by exactly the input's difference (lesson of v2.6.4), and that difference must be small
Y_REF = np.array([1.6467635298957353, -1.483645605996513, 0.543009971283822])
d_in = float(np.max(np.abs(y[:3] / Y_REF - 1)))
tol = max(1e-12, 10 * d_in)
check(f"MAPIE data rebuilt: first values as in ref_mapie_split.txt (relative difference {d_in:.1e} < 1e-9)",
      d_in < 1e-9)
MAPIE = {0.05: 3.123412528720399, 0.10: 1.7654884581187231}
for alpha, ref in MAPIE.items():
    r = CI(pd.Series(y), pd.Series(fcast), alpha=alpha, method="split", calib_size=60, show=False)
    half = (r["table"]["upper"] - r["table"]["forecast"]).dropna()
    check(f"split, alpha {alpha}: half-width equal to MAPIE SplitConformalRegressor (rel {tol:.0e}), on every test row",
          len(half) == 60 and np.allclose(half, ref, rtol=tol, atol=0), (half.iloc[0], ref))


def split_by_counting(err, alpha):
    err = np.abs(err)
    need = math.ceil((len(err) + 1) * (1 - alpha))
    cands = [e for e in np.sort(err) if (err <= e).sum() >= need]
    return cands[0] if cands else np.inf


ok = True
rng = np.random.default_rng(27102)
for m in (19, 20, 37, 50, 199):
    for alpha in (0.05, 0.1, 0.2):
        if m < math.ceil((1 - alpha) / alpha):
            continue
        e = rng.standard_normal(m + 5)
        r = CI(pd.Series(e), pd.Series(np.zeros(m + 5)), alpha=alpha, method="split", calib_size=m, show=False)
        ok &= abs(r["table"]["upper"].iloc[-1] - split_by_counting(e[:m], alpha)) == 0
check("split: half-width equal to the counting definition of the conformal quantile (m from 19 to 199)", ok)

# ============================================================ 2. exact finite-sample coverage under i.i.d. errors
# one test point per replication right after a calibration block of m errors: coverage = ceil((m+1)(1-a))/(m+1)
for m, reps in ((19, 6000), (50, 6000)):
    rng = np.random.default_rng(27100 + m)
    hits = 0
    for _ in range(reps):
        e = rng.standard_normal(m + 1)
        r = CI(pd.Series(e), pd.Series(np.zeros(m + 1)), method="split", calib_size=m, show=False)
        hits += int(r["table"]["covered"].iloc[-1])
    exact = math.ceil((m + 1) * 0.95) / (m + 1)
    se = math.sqrt(exact * (1 - exact) / reps)
    check(f"split, m = {m}: coverage {hits / reps:.4f} within 4 MC SE ({4 * se:.4f}) of the exact {exact:.4f}",
          abs(hits / reps - exact) < 4 * se)

# ============================================================ 3. no look-ahead with h > 1 (rolling and ACI)
rng = np.random.default_rng(27103)
n, t0 = 120, 70
base = pd.Series(rng.standard_normal(n))
zero = pd.Series(np.zeros(n))
poisoned = base.copy()
poisoned.iloc[t0:] *= 1000.0           # errors of targets t0, t0+1, ... become huge
for method in ("rolling", "aci"):
    for h in (1, 3, 6):
        a = CI(base, zero, method=method, h=h, show=False)["table"]
        b = CI(poisoned, zero, method=method, h=h, show=False)["table"]
        same = np.allclose(a["upper"].iloc[:t0 + h], b["upper"].iloc[:t0 + h], rtol=0, atol=0, equal_nan=True)
        moved = a["upper"].iloc[t0 + h] != b["upper"].iloc[t0 + h]
        check(f"{method}, h = {h}: rows up to t0 + h - 1 ignore the errors of targets >= t0; row t0 + h uses them",
              same and moved)

# ============================================================ 4. ACI against an explicit loop over time
def aci_reference(y, f, alpha, gamma, h, m_min):
    """Time runs t = 0, 1, ...; at time t the target t is observed and the level moves by gamma (alpha - err_t);
    the forecast of row i is issued at origin i - h with the level in force then and the errors of rows <= i - h."""
    n = len(y)
    err = np.abs(y - f)
    level_at = {}                       # level in force right after time t has been processed
    a_t = alpha
    half = np.full(n, np.nan)
    for i in range(n):
        o = i - h
        lev = 1 - (level_at[o] if o >= 0 else alpha)
        lev = 0.0 if abs(lev) < 1e-12 else lev
        known = np.sort(err[:o + 1]) if o >= 0 else np.array([])
        if len(known) >= m_min:
            m = len(known)
            if lev <= 0:
                half[i] = np.nan
                miss = 1.0
            else:
                # counting definition: the smallest known error with at least (m + 1) * level errors at or below
                # it (levels within 1e-12 of 1 count as 1, of 0 as 0: the running sum is rounded)
                need = (m + 1) * (1.0 if abs(1 - lev) < 1e-12 else lev)
                cands = [e for e in known if (known <= e).sum() >= need - 1e-9]
                half[i] = np.inf if lev >= 1 - 1e-12 or not cands else cands[0]
                miss = float(err[i] > half[i])
            a_t = a_t + gamma * (alpha - miss)
        level_at[i] = a_t
    return half


ok = True
rng = np.random.default_rng(27104)
for h in (1, 2, 5):
    for gamma in (0.005, 0.05):
        e = rng.standard_normal(300) * np.exp(np.linspace(0, 2, 300))     # drifting scale
        r = CI(pd.Series(e), pd.Series(np.zeros(300)), method="aci", h=h, gamma=gamma, show=False)
        ref = aci_reference(e, np.zeros(300), 0.05, gamma, h, 19)
        ours = (r["table"]["upper"]).to_numpy()
        ok &= np.array_equal(np.isnan(ours), np.isnan(ref)) and np.allclose(ours[~np.isnan(ref)], ref[~np.isnan(ref)],
                                                                            rtol=0, atol=0)
check("aci: intervals equal to an explicit loop over time (h = 1, 2, 5; gamma 0.005 and 0.05; drifting errors)", ok)

# ============================================================ 5. ACI long-run bound on any sequence
def worst_excess(seq_errors, gamma, h):
    r = CI(pd.Series(seq_errors), pd.Series(np.zeros(len(seq_errors))), method="aci", h=h, gamma=gamma, show=False)
    T = r["n_evaluated"]
    miss = 1 - r["coverage"]
    bound = (max(0.05, 0.95) + h * gamma) / (T * gamma)
    lev = r["table"]["level"].dropna()
    return abs(miss - 0.05) - bound, lev.min(), lev.max()


rng = np.random.default_rng(27105)
seqs = {
    "iid": rng.standard_normal(2000),
    "AR(1) 0.95": np.asarray(pd.Series(rng.standard_normal(2000)).ewm(alpha=0.05).mean()) * 10,
    "scale x100 jump": np.r_[rng.standard_normal(1000), 100 * rng.standard_normal(1000)],
    "adversarial (miss whenever possible)": np.r_[rng.standard_normal(100), 10.0 ** np.linspace(0, 12, 1900)],
}
ok, info = True, []
for nm, s in seqs.items():
    for h in (1, 4):
        for gamma in (0.05, 0.2):
            ex, lmin, lmax = worst_excess(s, gamma, h)
            ok &= ex <= 1e-12 and lmin >= -h * gamma - 1e-12 and lmax <= 1 + h * gamma + 1e-12
            info.append((nm, h, gamma, round(ex, 4), round(lmin, 3), round(lmax, 3)))
check("aci: |share of misses - alpha| within (max(a, 1-a) + h gamma)/(T gamma), level in [-h gamma, 1 + h gamma], "
      "on i.i.d., persistent, jumping and adversarial errors", ok, info)
ex = worst_excess(seqs["adversarial (miss whenever possible)"], 0.05, 1)
check("aci: the adversarial sequence comes close to the bound (the check can fail)", ex[0] > -0.1, ex)

# ============================================================ 6. coverage under i.i.d. errors (rolling and ACI, h = 1)
rng = np.random.default_rng(27106)
cov = {"rolling": [], "aci": []}
for _ in range(300):
    e = pd.Series(rng.standard_normal(219))
    for mth in cov:
        cov[mth].append(CI(e, e * 0, method=mth, calib_size=50, show=False)["coverage"])
for mth, v in cov.items():
    se = np.std(v, ddof=1) / math.sqrt(len(v))
    check(f"{mth}, i.i.d. errors, window 50: mean coverage {np.mean(v):.4f} within 4 MC SE of 0.95 "
          f"(rolling: exact {math.ceil(51 * 0.95) / 51:.4f})",
          abs(np.mean(v) - (math.ceil(51 * 0.95) / 51 if mth == "rolling" else 0.95)) < 4 * se + (0.006 if mth == "aci" else 0))

# ============================================================ 7. future rows, Wilson interval, used
rng = np.random.default_rng(27107)
e = pd.Series(rng.standard_normal(100), index=pd.date_range("2020-01-01", periods=100, freq="D"))
a = e.copy()
a.iloc[-4:] = np.nan
r = CI(a, e * 0, show=False)
tab = r["table"]
check("future rows (no actual yet) get an interval and stay out of the coverage",
      tab["upper"].iloc[-4:].notna().all() and tab["covered"].iloc[-4:].isna().all() and r["used"]["n_future"] == 4
      and r["n_evaluated"] == int(tab["covered"].notna().sum()))
k = int(np.nansum(tab["covered"]))
lo, hi = proportion_confint(k, r["n_evaluated"], alpha=0.05, method="wilson")
check("coverage interval equal to statsmodels' Wilson interval", abs(r["coverage_ci"][0] - lo) < 1e-12
      and abs(r["coverage_ci"][1] - hi) < 1e-12)
check("warm-up: the first 19 rows have no interval (h = 1, alpha 0.05)",
      tab["upper"].iloc[:19].isna().all() and tab["upper"].iloc[19:].notna().all() and r["used"]["warm_up_rows"] == 19)
check("used records method, alpha, h, gamma, calibration and version",
      r["used"]["method"] == "aci" and r["used"]["gamma"] == 0.005 and r["used"]["min_calib_errors"] == 19
      and "skill_version" in r["used"])

# a level that the running sum leaves a hair above 0 is the empty set, as the exact level 0 (review r1-A S1: it took
# the largest error, a +-100 interval among +-1 ones)
y_ = np.ones(30)
y_[0] = 100.0
t_ = CI(pd.Series(y_), pd.Series(np.zeros(30)), alpha=0.2, gamma=0.5, show=False)["table"]
lv_ = t_["level"]
tiny = lv_[(lv_.abs() < 1e-12)]
check("aci: every level within 1e-12 of 0 gives the empty set, never the largest error (alpha 0.2, gamma 0.5)",
      len(tiny) >= 3 and (tiny > 0).any() and t_.loc[tiny.index, "empty"].all() and t_.loc[tiny.index, "upper"].isna().all(),
      t_.loc[tiny.index])
low = t_[(lv_ > 1e-12) & (lv_ <= 0.6)]
check("aci: with the same data, every level from 0.1 to 0.6 uses the smallest error (half-width 1), not 100",
      len(low) > 10 and ((low["upper"] - low["forecast"]) == 1.0).all())

# ============================================================ 8. refusals
s = pd.Series(np.arange(50.0))
for nm, kw in (("unknown method", dict(method="bootstrap")), ("calib_size below 19", dict(calib_size=18)),
               ("h = 0", dict(h=0)), ("alpha = 1", dict(alpha=1.0)), ("gamma = 0", dict(gamma=0.0))):
    check(f"refuses {nm}", raises(CI, s, s * 0, show=False, **kw))
check("accepts calib_size = 19 (the fewest errors for a finite 95% interval)", not raises(CI, s, s * 0, calib_size=19, show=False))
check("refuses an index out of time order (newest first)", raises(CI, s[::-1], s[::-1] * 0, show=False))
check("refuses a repeated target date", raises(CI, pd.Series(np.arange(50.0), index=[0] * 2 + list(range(1, 49))),
                                              pd.Series(np.zeros(50), index=[0] * 2 + list(range(1, 49))), show=False))
inf_ = s.copy()
inf_.iloc[5] = np.inf
check("refuses infinite actuals and forecasts", raises(CI, inf_, s * 0, show=False) and raises(CI, s, inf_, show=False))
check("refuses h = True and h or calib_size = nan with the whole-number message",
      all("whole number" in str(raises(CI, s, s * 0, show=False, **kw)) for kw in (dict(h=True), dict(h=float("nan")),
                                                                                   dict(calib_size=float("nan")))))
check("refuses a fractional horizon or calibration size", raises(CI, s, s * 0, h=2.9, show=False)
      and raises(CI, s, s * 0, calib_size=25.5, show=False))
holes = s.copy()
holes.iloc[10] = np.nan
check("refuses a missing actual before the last observed row", raises(CI, holes, s * 0, show=False))
f_holes = s * 0
f_holes.iloc[3] = np.nan
check("refuses missing forecasts", raises(CI, s, f_holes, show=False))
check("refuses misaligned indexes", raises(CI, s, pd.Series(np.zeros(50), index=range(1, 51)), show=False))
# v2.7.1: a record too short for the default block (half of the rows with an actual) returns the "no row could be
# evaluated" finding, as aci and rolling do, instead of raising (review NIT of v2.7.0; the check used to expect the
# exception); an explicit calib_size below 19 is still refused
r_short = CI(s.iloc[:30], s.iloc[:30] * 0, method="split", show=False)
check("split with a default calibration block below 19 errors: no row evaluated, with a warning (v2.7.1)",
      r_short["n_evaluated"] == 0 and any("no row could be evaluated" in x["message"] for x in r_short["warnings"])
      and raises(CI, s.iloc[:30], s.iloc[:30] * 0, method="split", calib_size=15, show=False))

# split with h > 1: the first row with an interval is the first whose origin has seen the whole calibration block
e = pd.Series(np.random.default_rng(27110).standard_normal(80))
for h_ in (1, 3):
    tab_ = CI(e, e * 0, method="split", calib_size=40, h=h_, show=False)["table"]
    check(f"split, h = {h_}: rows up to 40 + {h_} - 2 have no interval, row 40 + {h_} - 1 has one",
          tab_["upper"].iloc[:40 + h_ - 1].isna().all() and np.isfinite(tab_["upper"].iloc[40 + h_ - 1]))
# the slack that ACI prints uses h * gamma for h > 1 (and gamma for h = 1)
for h_ in (1, 4):
    r_ = CI(e, e * 0, h=h_, gamma=0.05, show=False)
    want = (0.95 + h_ * 0.05) / (r_["n_evaluated"] * 0.05)
    check(f"aci, h = {h_}: printed slack (0.95 + h·gamma)/(T·gamma) = {want:.4f}", abs(r_["guarantee"]["slack"] - want) < 1e-15)

# ============================================================ 9. findings (T3), one case on each side
lv = lambda r: [w["level"] for w in r["warnings"]]  # noqa: E731
rng = np.random.default_rng(27108)
iid = pd.Series(rng.standard_normal(400))
ar = pd.Series(np.asarray(pd.Series(rng.standard_normal(400)).ewm(alpha=0.1).mean()))
check("rolling, i.i.d. errors, h = 1, window 60: no finding",
      lv(CI(iid, iid * 0, method="rolling", calib_size=60, show=False)) == [])
w = CI(iid, iid * 0, method="rolling", h=4, calib_size=60, show=False)["warnings"]
check("rolling, h = 4: warning that overlapping errors void the guarantee, with the ACI action",
      [x["level"] for x in w] == ["warning"] and "overlap" in w[0]["message"] and "method='aci'" in w[0]["action"])
w = CI(ar, ar * 0, method="split", calib_size=200, show=False)["warnings"]
check("split, autocorrelated errors: warning names Ljung-Box", any(x["level"] == "warning" and "Ljung-Box" in x["message"]
                                                                   for x in w))
check("rolling with a short window (19): notice about few calibration errors",
      "notice" in lv(CI(iid, iid * 0, method="rolling", calib_size=19, show=False)))
check("aci, h = 4: notice that the delayed bound is derived, not published",
      any("not published" in x["message"] and "proved for this" in x["message"] for x in CI(iid, iid * 0, h=4, gamma=0.5, show=False)["warnings"]))
# slack = (0.95 + 0.05) / (T * 0.05) = 20 / T: T = 399 gives 0.0501 (notice), T = 400 gives 0.05 (no notice)
for T_, want in ((399, True), (400, False)):
    e = pd.Series(np.random.default_rng(27109).standard_normal(T_ + 19))
    r = CI(e, e * 0, gamma=0.05, show=False)
    has = any("not informative" in x["message"] for x in r["warnings"])
    check(f"aci, gamma 0.05, T = {r['n_evaluated']}: 'not informative' notice {'present' if want else 'absent'}",
          r["n_evaluated"] == T_ and has == want, r["guarantee"])
# under-coverage beyond the Wilson interval: calibration on small errors, then errors 5 times larger
big = pd.Series(np.r_[rng.standard_normal(100), 5 * rng.standard_normal(100)])
w = CI(big, big * 0, method="split", calib_size=100, show=False)["warnings"]
check("split after a scale jump: warning that empirical coverage is below the level",
      any("empirical coverage" in x["message"] and x["level"] == "warning" for x in w))
# coverage below the level but inside its Wilson interval: no warning (47 of 50 = 94%, Wilson upper bound about 98%)
cal = np.linspace(0.01, 1.0, 100)
tst = np.r_[np.full(47, 0.5), np.full(3, 2.0)]
edge = pd.Series(np.r_[cal, tst])
r_ = CI(edge, edge * 0, method="split", calib_size=100, show=False)
check(f"split, coverage {r_['coverage']:.0%} below 95% but inside its Wilson interval: no under-coverage warning",
      abs(r_["coverage"] - 0.94) < 1e-12 and r_["coverage_ci"][1] > 0.95
      and not any("empirical coverage" in x["message"] for x in r_["warnings"]))
same = pd.Series(rng.standard_normal(200))
check("split, same scale: no under-coverage warning",
      not any("empirical coverage" in x["message"] for x in CI(same, same * 0, method="split", calib_size=100,
                                                               show=False)["warnings"]))

# Ljung-Box screen at a fixed 5%, whatever the interval's alpha; on the errors and on their size (review r1-A S6)
rng = np.random.default_rng(27111)
fires = 0
for _ in range(200):
    e_ = pd.Series(rng.standard_normal(200))
    fires += any("autocorrelated" in x["message"] for x in CI(e_, e_ * 0, alpha=0.2, method="rolling", calib_size=60,
                                                               show=False)["warnings"])
check(f"rolling, alpha 0.2, i.i.d. errors: the dependence warning fires in {fires} of 200 series (at most 5% + 3 SE)",
      fires / 200 <= 0.05 + 3 * math.sqrt(0.05 * 0.95 / 200))
sgn = pd.Series(np.asarray(pd.Series(rng.standard_normal(300)).ewm(alpha=0.2).mean()))
check("split: autocorrelated signed errors are caught by the screen on the errors themselves",
      any("autocorrelated" in x["message"] for x in CI(sgn, sgn * 0, method="split", calib_size=150, show=False)["warnings"]))
alt = pd.Series((-1.0) ** np.arange(300) * (1 + 0.1 * rng.standard_normal(300)))   # sign alternates, size i.i.d.
w_alt = CI(alt, alt * 0, method="split", calib_size=150, show=False)["warnings"]
check("split: errors whose sign alternates (size i.i.d.) are caught by the screen on the signed errors",
      any("autocorrelated" in x["message"] for x in w_alt), w_alt)
# size persistent, sign i.i.d. (volatility clustering): only the screen on |errors| sees it (review r2)
ls_ = np.asarray(pd.Series(rng.standard_normal(300)).ewm(alpha=0.05).mean()) * 4
vol = pd.Series(np.where(rng.random(300) < 0.5, -1.0, 1.0) * np.exp(ls_))
w_vol = CI(vol, vol * 0, method="split", calib_size=150, show=False)["warnings"]
check("split: errors with persistent size and random sign are caught by the screen on their size",
      any("autocorrelated" in x["message"] for x in w_vol), w_vol)
# infinite or empty intervals are counted in a finding
r_ = CI(pd.Series(y_), pd.Series(np.zeros(30)), alpha=0.2, gamma=0.5, show=False)
# v2.7.1: the text of the notice changed ("N empty because the level went past 0 (a sure miss)")
check("aci: a notice counts the empty intervals",
      any("empty because the level went past 0 (a sure miss)" in x["message"] for x in r_["warnings"]))
r_ = CI(iid, iid * 0, method="rolling", calib_size=60, show=False)
check("rolling with finite intervals only: no such notice", not any("infinite" in x["message"] for x in r_["warnings"]))
# no evaluated row: no slack notice with inf or nan, and the action gives the record needed
r_ = CI(pd.Series(np.arange(20.0)), pd.Series(np.zeros(20)), h=3, show=False)
check("aci with nothing evaluated: one warning, asking for calib + h = 22 forecasts, no 'inf' or 'nan' text",
      r_["n_evaluated"] == 0 and [x["level"] for x in r_["warnings"]].count("warning") == 1
      and "22 forecasts" in next(x for x in r_["warnings"] if x["level"] == "warning")["action"] and "nan" not in str(r_["warnings"]) and "inf" not in str(r_["warnings"]),
      r_["warnings"])
# the coverage interval is a 95% Wilson interval whatever alpha; the level prints with its digits (97.5%)
r_ = CI(iid, iid * 0, alpha=0.025, show=False)
lo_, hi_ = proportion_confint(round(r_["coverage"] * r_["n_evaluated"]), r_["n_evaluated"], alpha=0.05, method="wilson")
check("alpha 0.025: coverage interval is the 95% Wilson interval", abs(r_["coverage_ci"][0] - lo_) < 1e-12
      and abs(r_["coverage_ci"][1] - hi_) < 1e-12)
_, out = quiet(CI, iid, iid * 0, alpha=0.025)
check("alpha 0.025: printed as level 97.5%", "level 97.5%" in out, out[:200])
# under-coverage action depends on the method
big2 = pd.Series(np.r_[rng.standard_normal(100), 8 * rng.standard_normal(60)])
w_ = CI(big2, big2 * 0, calib_size=100, show=False)["warnings"]
check("aci under-covering: the action does not tell the user to switch to aci",
      any("empirical coverage" in x["message"] for x in w_) and not any("method='aci'" in x["action"] for x in w_
                                                                          if "empirical coverage" in x["message"]), w_)

# ============================================================ 10. what show prints
_, out = quiet(CI, iid, iid * 0, method="rolling", h=4, calib_size=60)
check("show: header, coverage with its Wilson interval, and the finding with level and action",
      "Conformal prediction intervals (rolling, level 95%, horizon 4)" in out and "Wilson 95% CI" in out
      and "WARNING: errors of 4-step forecasts overlap" in out and "action: use method='aci'" in out, out)
_, out = quiet(CI, iid, iid * 0)
check("show (aci): the long-run guarantee with its slack and the notice", "long-run guarantee: share of misses within ±"
      in out and "NOTICE: the long-run guarantee says only" in out, out)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
