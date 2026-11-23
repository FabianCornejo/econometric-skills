"""v2.7.3: regressions of the items that are not documentation (the "Returns" blocks are in test_api_docs.py).

1. conformal_intervals (v2.7.2 validation): the action of the "infinite intervals" finding says it is per model, so
   the coverage of the finite intervals is given for every model whose intervals are reported, not only the first;
   forecasting-backtesting.md says the same.
2. evaluate (review r2 N6 of v2.7.2): the cause of an inf metric or of a DM test not computed goes to
   tab.attrs["warnings"], also with show=False and with a single forecast; show=True prints the benchmark's own
   warning (before, nothing was said with one forecast, and the cause lived only in the DM lines of show=True).
   The columns and the numbers do not change.
3. ph_test(show=False) prints nothing (decision D4); the default still prints.

Run from the repository root:  python3 tests/test_v273.py
"""
import io
import sys
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "econometrics" / "scripts"))
import _console  # noqa: E402,F401  (non-ASCII text on a Windows console)
import forecasting as fc  # noqa: E402

warnings.simplefilter("ignore")
results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  -> {detail}"))


def run(fn, *a, **k):
    buf = io.StringIO()
    with redirect_stdout(buf), np.errstate(all="ignore"):
        r = fn(*a, **k)
    return r, buf.getvalue()


# ============================================================ 1. conformal_intervals: the action is per model
# 21-period overlapping sums (seed 3 from the search: 114 and 118 infinite intervals with h = 21)
rng = np.random.default_rng(3)
n, h = 300, 21
shocks = rng.normal(size=n + h)
act = pd.Series(np.convolve(shocks, np.ones(h), "valid")[:n])
f_zero = pd.Series(np.zeros(n))
f_noise = pd.Series(0.3 * rng.normal(size=n))
NEW = ("next to the total for these intervals, and do the same for every other model whose intervals you report")
for name, f in (("zero forecast", f_zero), ("noisy forecast", f_noise)):
    r, out = run(fc.conformal_intervals, act, f, h=h)
    fin = [w for w in r["warnings"] if "infinite" in w["message"]]
    check(f"conformal_intervals ({name}): infinite intervals ({r['n_infinite']}) and the finding's action says it is "
          "per model, in the result and in the printed text",
          r["n_infinite"] > 0 and len(fin) == 1 and NEW in fin[0]["action"] and NEW in " ".join(out.split()),
          (r["n_infinite"], [w["action"][:160] for w in fin]))
    check(f"conformal_intervals ({name}): the old fragment ('give the coverage of the finite intervals next to the "
          "total') is kept for the tests that look for it", "give the coverage of the finite intervals next to the total"
          in fin[0]["action"])
r0 = fc.conformal_intervals(act.iloc[:40], f_zero.iloc[:40], h=1, show=False)
check("conformal_intervals: no infinite intervals, no such finding (the new text only comes with them)",
      r0["n_infinite"] == 0 and not any(NEW in w["action"] for w in r0["warnings"]), r0["n_infinite"])
ref = (ROOT / "econometrics" / "references" / "forecasting-backtesting.md").read_text(encoding="utf-8")
check("forecasting-backtesting.md: coverage_finite and the number of infinite intervals for every interval reported",
      "Give `coverage_finite` and the number of infinite intervals for every interval you report, not only the first"
      in ref)

# ============================================================ 2. evaluate: the cause in attrs["warnings"]
rng = np.random.default_rng(273)
a = pd.Series(rng.normal(size=80))
fa, fb = pd.Series(rng.normal(size=80)), pd.Series(rng.normal(size=80))
a_ov = a.copy()
a_ov.iloc[5] = 1e160                    # a finite error whose square overflows
lv = pd.Series(np.r_[0.0, 1 + rng.random(79)])
a_inf = lv.pct_change().fillna(0.0)     # a percentage change from zero: inf in row 1
OVER, INF = "the squared errors overflow", "has infinite values"

cases = (("overflow, two forecasts", a_ov, {"a": fa, "b": fb}, {"a": OVER, "b": OVER}),
         ("overflow, one forecast", a_ov, {"a": fa}, {"a": OVER}),
         ("inf input, two forecasts", a_inf, {"a": fa, "b": fb}, {"a": INF, "b": INF}),
         ("inf input, one forecast", a_inf, {"a": fa}, {"a": INF}),
         ("only the benchmark overflows", a, {"bench": fb.where(fb.index != 7, 1e160), "a": fa},
          {"bench": OVER, "a": "DM test against bench not computed: see the warning on bench"}),
         ("identical forecasts", pd.Series(np.arange(10.0)), {"a": pd.Series(np.arange(10.0) + 1),
                                                              "b": pd.Series(np.arange(10.0) + 1)},
          {"b": "no positive variance"}))
for case, act_, fs, want in cases:
    tab_q, out_q = run(fc.evaluate, act_, fs, show=False)
    tab_s, out_s = run(fc.evaluate, act_, fs)
    w = tab_q.attrs.get("warnings")
    got = {x.split(": ", 1)[0]: x for x in (w or [])}
    check(f"evaluate ({case}), show=False: one warning per model named, with the right cause, and nothing printed",
          isinstance(w, list) and set(got) == set(want) and all(want[k] in got[k] for k in want) and out_q == "",
          (w, out_q[:100]))
    check(f"evaluate ({case}): the same warnings with show=True, and the same table",
          tab_s.attrs["warnings"] == w and tab_s.equals(tab_q))
    bench = list(fs)[0]
    if bench in got:
        check(f"evaluate ({case}), show=True: the benchmark's warning is printed (before: nothing with one forecast)",
              "  Warning: " + got[bench] in out_s, out_s)
    if case.startswith("inf input"):
        check(f"evaluate ({case}): the inf cause does not mention an overflow (review r2 S2 of v2.7.2)",
              all("overflow" not in x for x in w) and "overflow" not in out_s, (w, out_s))
    if case.startswith("overflow"):
        check(f"evaluate ({case}): the overflow cause does not say 'identical forecasts'",
              all("identical" not in x for x in w) and "identical forecasts" not in out_s, (w, out_s))

tab_c, out_c = run(fc.evaluate, a, {"a": fa, "b": fb}, show=False)
check("evaluate: a clean case has attrs['warnings'] == []", tab_c.attrs["warnings"] == [], tab_c.attrs)
tab_1, _ = run(fc.evaluate, a, {"a": fa}, show=False)
check("evaluate: a clean single forecast has attrs['warnings'] == [] too", tab_1.attrs["warnings"] == [])
# the numbers are those of forecast_metrics (unchanged), the attrs are the only addition
m_ = fc.forecast_metrics(a, fb, naive=fa)
check("evaluate: the numbers of a row are forecast_metrics' (nothing changed in the table)",
      all(abs(tab_c.loc["b", k] - v) < 1e-12 for k, v in m_.items()) and list(tab_c.attrs) == ["warnings"],
      (tab_c.loc["b"].to_dict(), m_))
doc = fc.evaluate.__doc__
check("evaluate's block names attrs['warnings'] and says it is filled with show=False and with one forecast",
      'attrs["warnings"]  list of str (v2.7.3)' in doc and "filled with show=False and with one" in " ".join(doc.split()))

# ============================================================ 3. ph_test(show=...) (decision D4)
import survival as sv  # noqa: E402
rng = np.random.default_rng(4)
sd = pd.DataFrame({"x": rng.normal(size=120)})
sd["t"] = np.ceil(rng.exponential(np.exp(-0.5 * sd.x) * 8)).clip(1, 20)
sd["e"] = (rng.uniform(size=120) < 0.7).astype(int)
res, tab = sv.cox(sd, "x", "t", "e", show=False)
r_q, out_q = run(sv.ph_test, res, sd, "t", "e", show=False)
r_d, out_d = run(sv.ph_test, res, sd, "t", "e")
check("ph_test: show=False prints nothing; the default prints the table as before; same numbers",
      out_q == "" and "Proportional-hazards check (Schoenfeld residuals vs time)" in out_d and r_q.equals(r_d),
      (out_q[:80], out_d[:80]))

# ============================================================ 5. review r1-B: evaluate, diebold_mariano, survival, PT
import diagnostics as dg  # noqa: E402

def evw(act_, fs_, **kw):
    t_, o_ = run(fc.evaluate, act_, fs_, **kw)
    return t_, o_, t_.attrs["warnings"]


def all_named(t_, w_):
    """Every inf or NaN outside the DM columns, and every DM NaN outside the benchmark's row, has a warning on its model."""
    for k_ in t_.index:
        cols = [c_ for c_ in t_.columns if not c_.startswith("DM")]
        bad = [c_ for c_ in cols if not np.isfinite(float(t_.loc[k_, c_]))]
        dm_bad = "DM_p" in t_.columns and k_ != t_.index[0] and not np.isfinite(float(t_.loc[k_, "DM_p"]))
        if (bad or dm_bad) and not any(x_.startswith(f"{k_}: ") for x_ in w_):
            return False
    return True


a0 = pd.Series(np.arange(40.0))
t_, o_, w_ = evw(a0, {"b": a0 + 1.0, "m": a0 - 2.0})
check("evaluate (B1): a constant, non-zero loss difference -> 'constant', no 'identical' claim, no DM verdict",
      np.isnan(t_.loc["m", "DM_p"]) and any("loss difference is constant" in x_ and "identical" not in x_ for x_ in w_)
      and "accurate than" not in o_ and all_named(t_, w_), (w_, o_[-300:]))
rng5 = np.random.default_rng(55)
e2_ = rng5.normal(size=200)
e1_ = np.sqrt(e2_ ** 2 + 0.3)                     # loss 1 = loss 2 + 0.3 up to rounding
r_ = dg.diebold_mariano(e1_, e2_, show=False)
check("diebold_mariano (B10): a loss difference constant up to rounding gives NaN (it gave 5.5e16 and p = 0)",
      np.isnan(r_["p"]) and np.isnan(r_["DM_HLN"]), r_)
e1g = np.sqrt(e2_ ** 2 + 0.3 * (1 + 1e-9 * rng5.normal(size=200)))
r_ = dg.diebold_mariano(e1g, e2_, show=False)
check("diebold_mariano: a genuine variation of 1e-9 (relative) is still tested (the other side of the cut)",
      np.isfinite(r_["p"]), r_)
alt = 1 + 0.5 * (-1.0) ** np.arange(60) + 1e-3 * rng5.normal(size=60)    # loss differences alternate: LRV < 0 at h = 2
aa = pd.Series(np.zeros(60))
t_, o_, w_ = evw(aa, {"b": pd.Series(np.ones(60)), "m": pd.Series(np.sqrt(alt))}, h=2)
check("evaluate (B1): a long-run variance that is not positive (h = 2) is named as such, not 'identical forecasts'",
      np.isnan(t_.loc["m", "DM_p"]) and any("long-run variance" in x_ and "(h = 2" in x_ for x_ in w_)
      and not any("identical" in x_ for x_ in w_), w_)
t_, o_, w_ = evw(pd.Series([1.0]), {"b": pd.Series([1.5]), "m": pd.Series([0.2])})
check("evaluate (B1): one common row -> 'only 1 period(s) in common'", any("only 1 period(s)" in x_ for x_ in w_), w_)
try:
    fc.evaluate(pd.Series([1.0, 2.0], index=[0, 1]), {"b": pd.Series([1.0, 2.0], index=[5, 6])}, show=False)
    msg_ = ""
except ValueError as x_:
    msg_ = str(x_)
check("evaluate (B2): no common row is refused with the reason (not an 'overflow' warning)", "no row where" in msg_, msg_)
t_, o_, w_ = evw(a, {"perfect": a.copy(), "m": fa}, show=False)
check("evaluate (B3): a perfect benchmark makes Theil's U inf, and the warning says why",
      np.isinf(t_.loc["m", "Theil_U(RMSE/RMSE_naive)"]) and any("Theil's U is inf because the benchmark perfect" in x_
                                                                  for x_ in w_) and all_named(t_, w_), w_)
up = pd.Series(np.abs(rng5.normal(size=80)) + 0.1)
t_, o_, w_ = evw(pd.Series(rng5.normal(size=80)), {"b": up, "m": up + 0.1}, direction="return", show=False)
check("evaluate: a forecast that is always up gives PT_p_value NaN, and a warning names the direction test",
      np.isnan(t_.loc["b", "PT_p_value"]) and any(x_.startswith("b: ") and "direction test" in x_ for x_ in w_)
      and all_named(t_, w_), w_)
f_inf = fb.copy()
f_inf.iloc[4] = np.inf
t_, o_, w_ = evw(a, {"a": fa, "b": f_inf}, show=False)
check("evaluate: an inf in one forecast only is attributed to 'the forecast', not to actual",
      any(x_.startswith("b: the forecast has infinite values") for x_ in w_)
      and not any(x_.startswith("a: ") and "infinite" in x_ for x_ in w_), w_)
for case, act_, fs, _w in cases:
    t_, o_, w_ = evw(act_, fs, show=False)
    check(f"evaluate ({case}): every inf or NaN of the table has a warning on its model", all_named(t_, w_), w_)
# the numbers from their definitions (review r1-B B9: the check above compared with the implementation)
check("evaluate: RMSE, MAE and bias of a row from their definitions (bias = mean of actual − forecast)",
      abs(tab_c.loc["b", "RMSE"] - np.sqrt(np.mean((a - fb) ** 2))) < 1e-12
      and abs(tab_c.loc["b", "MAE"] - np.mean(np.abs(a - fb))) < 1e-12
      and abs(tab_c.loc["b", "bias(mean error)"] - np.mean(a - fb)) < 1e-12)
check("forecast_metrics: bias is actual − forecast (a forecast 1 too low gives +1)",
      abs(fc.forecast_metrics(a, a - 1.0)["bias(mean error)"] - 1.0) < 1e-12)

# pesaran_timmermann (B7)
a7_ = rng5.normal(size=100)
f7_ = a7_ + rng5.normal(size=100)
a7n = a7_.copy()
a7n[:5] = np.nan
a7z = a7_.copy()
a7z[5:9] = 0.0
check("pesaran_timmermann: NaN and zero rows are left out of n", fc.pesaran_timmermann(a7n, f7_)["n"] == 95
      and fc.pesaran_timmermann(a7z, f7_)["n"] == 96)
r_ = fc.pesaran_timmermann(a7_, np.abs(f7_) + 0.1)
check("pesaran_timmermann: a forecast that is always up gives PT and p NaN with a finite hit rate, as the block says",
      np.isnan(r_["PT"]) and np.isnan(r_["p_one_sided"]) and np.isfinite(r_["hit_rate"]) and r_["n"] == 100
      and '("always up")' in fc.pesaran_timmermann.__doc__, r_)

# person_period (B5, B6, N5)
pp_df = pd.DataFrame({"id": [1, 2, 3], "t": [0.0, 3.0, 4.5], "e": [1, 0, 1], "x": [0.1, 0.2, 0.3]})
r_ = sv.person_period(pp_df, "id", "t", "e", interval=2.0)
check("person_period: duration 0 keeps its event in period 1; interval 2 gives ceil(t/2) periods; the event in the last "
      "period; the id keeps its integer type (B6, N5)",
      r_.groupby("id")["period"].max().tolist() == [1, 2, 3] and r_.groupby("id")["y"].sum().tolist() == [1, 0, 1]
      and r_.loc[r_.id == 3, "y"].tolist() == [0, 0, 1] and pd.api.types.is_integer_dtype(r_["id"]), r_)
for bad_df, why in ((pd.concat([pp_df, pp_df.iloc[:1]]), "repeated id"), (pp_df.assign(y=1.0), "a column named y")):
    try:
        sv.person_period(bad_df, "id", "t", "e")
        ok_ = False
    except ValueError:
        ok_ = True
    check(f"person_period: {why} raises ValueError (B5: the rows were multiplied, events counted twice)", ok_)

# kaplan_meier S(t) by hand, logrank and ph_test thresholds, cox model-based SE (B4, B9)
km_df = pd.DataFrame({"t": [2.0, 3, 3, 5, 6, 8, 8, 9, 11, 12], "e": [1, 1, 0, 1, 1, 0, 1, 1, 0, 1]})
s_hand = 1.0
for tt_ in sorted(km_df.loc[km_df.e == 1, "t"].unique()):
    if tt_ <= 7:
        s_hand *= 1 - ((km_df.t == tt_) & (km_df.e == 1)).sum() / (km_df.t >= tt_).sum()
kmr = sv.kaplan_meier(km_df, "t", "e", times=[7], show=False)
check("kaplan_meier: S(7) equals the product-limit estimate computed by hand",
      abs(kmr["summary"].loc["all", "S(7)"] - s_hand) < 1e-12, (kmr["summary"].loc["all", "S(7)"], s_hand))
rng6 = np.random.default_rng(6)
grp_df = pd.DataFrame({"g": np.repeat([0, 1], 60)})
grp_df["t"] = np.ceil(rng6.exponential(np.where(grp_df.g == 1, 3.0, 9.0)))
grp_df["e"] = 1
lr_ = sv.logrank(grp_df, "t", "e", "g")
check("logrank: the conclusion follows p at 5%", (lr_["p"] < 0.05) == (lr_["conclusion"] == "survival differs across groups"))
check("ph_test: PH_violated_5% is p < 0.05 on every row", (r_q["PH_violated_5%"] == (r_q["p"] < 0.05)).all())
import statsmodels.formula.api as smf  # noqa: E402
res_m, tab_m = sv.cox(sd, "x", "t", "e", robust=False, show=False)
ref_m = smf.phreg("t ~ x", sd, status=sd["e"].astype(int), ties="efron").fit()
check("cox (robust=False): se is the model-based SE of statsmodels' phreg (fitted directly)",
      abs(tab_m.loc["x", "se"] - float(np.asarray(ref_m.bse)[0])) < 1e-10, (tab_m.loc["x", "se"], ref_m.bse))

# ============================================================ 4. version, CHANGELOG, counts (no "be the last version")
import json  # noqa: E402
import re  # noqa: E402
import reporting  # noqa: E402
ver = reporting.SKILL_VERSION
mk = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))
cl = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
top = re.search(r"^## v(\S+) ", cl, re.M)
check("version at least 2.7.3, the same in SKILL_VERSION, the marketplace and the top of the CHANGELOG",
      tuple(int(x_) for x_ in ver.split(".")) >= (2, 7, 3) and mk["plugins"][0]["version"] == ver and top
      and top.group(1) == ver, (ver, mk["plugins"][0]["version"], top and top.group(1)))
i_ = cl.find("\n## v2.7.3 — ")
sec = cl[i_:cl.index("\n## v2.7.2 — ")] if i_ >= 0 else ""
for word in ("Returns", "wild_cluster_bootstrap", "vecm_fit", "cox", "ph_test", "show=False", "attrs[\"warnings\"]",
             "every other model", "returns_block_pending.txt", "test_v272"):
    check(f"CHANGELOG v2.7.3 mentions {word!r}", word in sec)
readme = (ROOT / "README.md").read_text(encoding="utf-8")
m_r = re.search(r"\*\*(\d[\d,]*) checks\*\*", readme)
m_m = re.search(r"(\d[\d,]*) tested checks", mk["plugins"][0]["description"])
check("README and marketplace give the same count of checks",
      m_r and m_m and int(m_r.group(1).replace(",", "")) == int(m_m.group(1).replace(",", "")),
      (m_r and m_r.group(1), m_m and m_m.group(1)))

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
