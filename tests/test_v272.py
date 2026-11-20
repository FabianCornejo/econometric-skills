"""v2.7.2: regressions and documents (the docstrings' Returns blocks are tested in tests/test_api_docs.py).

1. conformal_intervals refuses an error |actual − forecast| that overflows to inf (review r1-A A8 of v2.7.1): the
   intervals were infinite in every method and the finding blamed ACI, also for "rolling" and "split".
2. iv_report's leverage warning for the control-function regression, on both sides of its 0.5 cut, with the leverage
   computed by another route (statsmodels' hat matrix; review r1-B B17 of v2.7.1).
3. Version, CHANGELOG and the count of checks.

Run from the repository root:  python3 tests/test_v272.py
"""
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "econometrics" / "scripts"))
import _console  # noqa: E402,F401  (non-ASCII text on a Windows console)
import diagnostics as dg  # noqa: E402
import forecasting as fc  # noqa: E402
import reporting  # noqa: E402

warnings.simplefilter("ignore")
results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  -> {detail}"))


def read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


# ============================================================ 1. conformal_intervals: values whose squares overflow
rng = np.random.default_rng(3)
n = 80
a = pd.Series(rng.normal(size=n), index=pd.RangeIndex(100, 100 + n))     # labels differ from positions (review N1)
f = pd.Series(rng.normal(size=n), index=a.index)
a_o, f_o = a.copy(), f.copy()
a_o.iloc[10:20] = 1.5e308
f_o.iloc[10:20] = -1.5e308          # both finite; their difference, 3e308, is not
check("the overflow case: actual and forecast finite, their difference infinite (the case of review A8 of v2.7.1)",
      np.isfinite(a_o).all() and np.isfinite(f_o).all() and np.isinf(np.abs(a_o.to_numpy() - f_o.to_numpy())).sum() == 10)


def refused(a_, f_, **kw):
    try:
        with np.errstate(all="raise"):        # no floating-point warning on the way to the message
            fc.conformal_intervals(a_, f_, show=False, **kw)
        return ""
    except ValueError as e:
        return str(e)


for m in ("aci", "rolling", "split"):
    msg = refused(a_o, f_o, method=m)
    check(f"{m}: an error that overflows is refused, with the count and the first row's label (110, not position 10)",
          "beyond 1e+150 in absolute value" in msg and "at 10 row(s) (first: 110)" in msg and "rescale" in msg, msg)
# review r1-B B1 of v2.7.2: large but finite errors after the warm-up with a large forecast made forecast ± half
# overflow with no word (n_infinite 0); now refused by the same bound
a_b, f_b = a.copy(), f.copy()
a_b.iloc[0:30] = 1e308
f_b.iloc[0:30] = -7e307
a_b.iloc[50] = f_b.iloc[50] = 1e308
for m in ("aci", "rolling", "split"):
    msg = refused(a_b, f_b, method=m)
    check(f"{m}: finite errors of 1.7e308 and a forecast of 1e308 are refused (31 rows, the first labelled 100)",
          "at 31 row(s) (first: 100)" in msg, msg)
# a forecast beyond the bound on a future row (no actual yet) is refused too: its interval would overflow
a_fu, f_fu = a.copy(), f.copy()
a_fu.iloc[-3:] = np.nan
f_fu.iloc[-1] = 1e200
check("a future row with a forecast beyond the bound is refused (its interval forecast ± half would overflow)",
      "at 1 row(s) (first: 179)" in refused(a_fu, f_fu, method="rolling"))
# the other side: values of 1e149 (errors 2e149) run, with finite bounds and no warning from numpy
a_n, f_n = a.copy(), f.copy()
a_n.iloc[10:20] = 1e149
f_n.iloc[10:20] = -1e149
for m in ("aci", "rolling", "split"):
    with np.errstate(all="raise"):
        r = fc.conformal_intervals(a_n, f_n, method=m, show=False)
    tb = r["table"].dropna(subset=["level"])
    blame = [w_["message"] for w_ in r["warnings"] if "ACI raised" in w_["message"]]
    check(f"{m}, errors of 2e149: runs with every bound finite and no finding that blames ACI",
          r["n_evaluated"] > 0 and np.isfinite(tb[["lower", "upper"]].to_numpy()).all()
          and (m == "aci" or (r["n_infinite"] == 0 and not blame)), (r["n_infinite"], blame))
# the Ljung-Box screen of "rolling" still sees AR(0.9) errors at the scale 1e149 (it went silent above ~1e154)
e_ = np.zeros(300)
for t_ in range(1, 300):
    e_[t_] = 0.9 * e_[t_ - 1] + rng.normal()
for scale in (1.0, 1e149):
    act_ = pd.Series(e_ * scale)
    r = fc.conformal_intervals(act_, act_ * 0.0, method="rolling", show=False)
    check(f"rolling, AR(0.9) errors at scale {scale:g}: the dependence screen still warns",
          any("autocorrelated" in w_["message"] or "dependence" in w_["message"] for w_ in r["warnings"]),
          [w_["message"][:80] for w_ in r["warnings"]])
# rows with no actual yet and an ordinary forecast are not refused
a_fut = a_n.copy()
a_fut.iloc[-3:] = np.nan
r = fc.conformal_intervals(a_fut, f_n, method="rolling", show=False)
check("rows with no actual yet and ordinary forecasts are accepted", r["used"]["n_future"] == 3)

# ============================================================ 1b. evaluate and diebold_mariano: a loss that overflows
import io  # noqa: E402
from contextlib import redirect_stdout  # noqa: E402
a_e = pd.Series(rng.normal(size=80))
a_e.iloc[5] = 1e160                     # a finite error whose square overflows (review r1-B B3)
fa, fb = pd.Series(rng.normal(size=80)), pd.Series(rng.normal(size=80))
buf = io.StringIO()
with redirect_stdout(buf), np.errstate(all="ignore"):
    tab = fc.evaluate(a_e, {"a": fa, "b": fb})
out_ = buf.getvalue()
check("evaluate: a squared loss that overflows is named as such, not as 'identical forecasts'",
      "the squared errors overflow" in out_ and "identical forecasts" not in out_ and np.isnan(tab.loc["b", "DM_p"]), out_)
buf = io.StringIO()
with redirect_stdout(buf), np.errstate(all="ignore"):
    dg.diebold_mariano(a_e - fa, a_e - fb)
out_ = buf.getvalue()
check("diebold_mariano: the reason names a loss that overflows, not only a missing or infinite error",
      "1 missing or infinite loss value(s)" in out_ and "overflows" in out_, out_)
buf = io.StringIO()
with redirect_stdout(buf):
    fc.evaluate(pd.Series(np.arange(10.0)), {"a": pd.Series(np.arange(10.0) + 1), "b": pd.Series(np.arange(10.0) + 1)})
check("evaluate: identical forecasts still get the variance reason", "no positive variance" in buf.getvalue(),
      buf.getvalue())

# ============================================================ 2. iv_report: the control-function leverage cut at 0.5
nn = 120
rng = np.random.default_rng(31)
zc = rng.normal(size=nn)
wc = rng.normal(size=nn)
uc = rng.normal(size=nn)
xc = 0.8 * zc + 0.3 * wc + rng.normal(size=nn)


def ref_leverage(big):
    """Leverage of row 11 beyond w in the control-function regression, by statsmodels' hat matrices."""
    xx = xc.copy()
    xx[11] = big
    vh = sm.OLS(xx, sm.add_constant(np.column_stack([wc, zc]))).fit().resid
    yy = xx + wc + uc
    h_all = sm.OLS(yy, sm.add_constant(np.column_stack([wc, xx, vh]))).fit().get_influence().hat_matrix_diag
    h_w = sm.OLS(yy, sm.add_constant(wc)).fit().get_influence().hat_matrix_diag
    return float(np.max(h_all - h_w)), int(np.argmax(h_all - h_w)), xx, yy


def solve(target):
    lo, hi = 1.0, 100.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ref_leverage(mid)[0] < target else (lo, mid)
    return (lo + hi) / 2


for target in (0.49, 0.4995, 0.5005, 0.51):
    h, row, xx, yy = ref_leverage(solve(target))
    d = pd.DataFrame(dict(y=yy, x=xx, z=zc, w=wc))
    r = dg.iv_report("y ~ 1 + w + [x ~ z]", d, show=False)
    cf = [m_ for m_ in r["warnings"] if "control-function regression" in m_]
    p = float(r["tests"]["p"].iloc[0])
    side = "warning" if target > 0.5 else "no warning"
    # review r1-B N2: the number in the message is the leverage of the control-function regression (statsmodels)
    ok = (cf == [] if target < 0.5 else (len(cf) == 1 and cf[0].startswith(f"row 11 has leverage {h:.2f} ")))
    check(f"control-function leverage {h:.4f} (statsmodels, row {row}): {side}; the p is computed", ok and np.isfinite(p)
          and row == 11 and abs(h - target) < 1e-9, (h, cf, p))
    fs_lev = [m_ for m_ in r["warnings"] if "first stage" in m_ and "leverage" in m_]
    check(f"leverage {target}: the outlier is in x, so no first-stage leverage warning hides or replaces it", not fs_lev,
          fs_lev)

# ============================================================ 3. version, CHANGELOG, counts
ver = reporting.SKILL_VERSION
mk = json.loads(read(".claude-plugin/marketplace.json"))
cl = read("CHANGELOG.md")
top = re.search(r"^## v(\S+) ", cl, re.M)
check("version 2.7.2 in SKILL_VERSION, the marketplace and the top of the CHANGELOG",
      ver == "2.7.2" and mk["plugins"][0]["version"] == "2.7.2" and top and top.group(1) == "2.7.2",
      (ver, mk["plugins"][0]["version"], top and top.group(1)))
sec = cl[cl.index("## v2.7.2"):cl.index("## v2.7.1")]
for word in ("Returns", "test_api_docs", "iv_report", "1e150", "0.5", "groupby", "note", "direction", "diebold_mariano"):
    check(f"CHANGELOG v2.7.2 mentions {word!r}", word in sec)
readme = read("README.md")
m_r = re.search(r"\*\*(\d[\d,]*) checks\*\*", readme)
m_m = re.search(r"(\d[\d,]*) tested checks", mk["plugins"][0]["description"])
check("README and marketplace give the same count of checks",
      m_r and m_m and int(m_r.group(1).replace(",", "")) == int(m_m.group(1).replace(",", "")),
      (m_r and m_r.group(1), m_m and m_m.group(1)))

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
