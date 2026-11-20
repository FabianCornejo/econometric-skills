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


# ============================================================ 1. conformal_intervals: an error that overflows
rng = np.random.default_rng(3)
n = 80
a = pd.Series(rng.normal(size=n))
f = pd.Series(rng.normal(size=n))
a_o, f_o = a.copy(), f.copy()
a_o[10:20] = 1.5e308
f_o[10:20] = -1.5e308          # both finite; their difference, 3e308, is not
check("the overflow case: actual and forecast finite, their difference infinite (the case of the review)",
      np.isfinite(a_o).all() and np.isfinite(f_o).all() and np.isinf(np.abs(a_o.to_numpy() - f_o.to_numpy())).sum() == 10)
for m in ("aci", "rolling", "split"):
    try:
        with np.errstate(all="raise"):        # no floating-point warning on the way to the message
            fc.conformal_intervals(a_o, f_o, method=m, show=False)
        msg = ""
    except ValueError as e:
        msg = str(e)
    check(f"{m}: an error that overflows is refused, with the count and the first row", "overflows to inf at 10 row(s)"
          in msg and "(first: 10)" in msg and "rescale" in msg, msg)
# the other side: 1e308 and −7e307 differ by 1.7e308, finite; the function runs and nothing is infinite
a_n, f_n = a.copy(), f.copy()
a_n[10:20] = 1e308
f_n[10:20] = -7e307
check("the other side: 1e308 − (−7e307) is finite", np.isfinite(np.abs(a_n.to_numpy() - f_n.to_numpy())).all())
for m in ("aci", "rolling", "split"):
    r = fc.conformal_intervals(a_n, f_n, method=m, show=False)
    blame = [w_["message"] for w_ in r["warnings"] if "ACI raised" in w_["message"]]
    if m == "aci":
        check("aci, large finite errors: runs", r["n_evaluated"] > 0)
    else:
        check(f"{m}, large finite errors: runs, no infinite interval and no finding that blames ACI",
              r["n_infinite"] == 0 and not blame, (r["n_infinite"], blame))
# future rows (actual NaN) are not errors: they never count as an overflow
a_fut = a_n.copy()
a_fut.iloc[-3:] = np.nan
r = fc.conformal_intervals(a_fut, f_n, method="rolling", show=False)
check("rows with no actual yet are not taken for an overflow", r["used"]["n_future"] == 3)

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
    ok = (cf == [] if target < 0.5 else (len(cf) == 1 and cf[0].startswith("row 11 has leverage")))
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
for word in ("Returns", "test_api_docs", "iv_report", "overflow", "0.5", "groupby", "note"):
    check(f"CHANGELOG v2.7.2 mentions {word!r}", word in sec)
readme = read("README.md")
m_r = re.search(r"\*\*(\d[\d,]*) checks\*\*", readme)
m_m = re.search(r"(\d[\d,]*) tested checks", mk["plugins"][0]["description"])
check("README and marketplace give the same count of checks",
      m_r and m_m and int(m_r.group(1).replace(",", "")) == int(m_m.group(1).replace(",", "")),
      (m_r and m_r.group(1), m_m and m_m.group(1)))

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
