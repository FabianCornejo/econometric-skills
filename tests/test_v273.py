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

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
