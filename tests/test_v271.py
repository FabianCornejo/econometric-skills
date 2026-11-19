"""v2.7.1: pending fixes of v2.7.0 and v2.6.13 (regressions and documents).

1. conformal_intervals: infinite intervals are explained by their cause (ACI raised the level after a run of misses),
   with the longest run and the coverage of the finite intervals (`n_infinite`, `coverage_finite`); measured in
   claude/medicoes-v2.7.1.md (M1) of the development repository.
2. The keys of iv_report, endogeneity_test and overid_test are written in references/panel-iv.md, and the text matches
   the results (both ways).
3. IV numerics: an offset in an exogenous regressor or in an instrument no longer merges instruments or makes an
   endogenous regressor "perfect" along directions it does not enter (M3a); iv_report's control-function test reads the
   leverage of its own regression, as endogeneity_test (review r1-A A5 of v2.6.13); endogeneity_test warns whenever HC0
   is used (M3c: 5.9-11.4% at k/n 0.08-0.19).
4. NITs of v2.7.0: study_card compares texts in Unicode NFC; 'split' with too short a record returns the finding of
   'aci' and 'rolling'; a NaT in the index has its own message (conformal_intervals and prais_winsten).

Run from the repository root:  python3 tests/test_v271.py
"""
import io
import json
import os
import re
import sys
import tempfile
import unicodedata
import warnings
from contextlib import redirect_stdout
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
import specification as spec  # noqa: E402

warnings.simplefilter("ignore")
results = []


def check(name, cond, info=""):
    results.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info)


read = lambda p: (ROOT / p).read_text(encoding="utf-8")  # noqa: E731


def raises(fn, pattern):
    try:
        fn()
    except ValueError as e:
        return re.search(pattern, str(e)) is not None
    return False


# ============================================================ 1. version and CHANGELOG
def vtuple(v):
    return tuple(int(x) for x in v.split("."))


mk = json.loads(read(".claude-plugin/marketplace.json"))
ver = mk["plugins"][0]["version"]
check("version at least 2.7.1 in the marketplace and in reporting.SKILL_VERSION (equal to each other)",
      vtuple(ver) >= (2, 7, 1) and reporting.SKILL_VERSION == ver)
cl = read("CHANGELOG.md")
m = re.search(r"^## v2\.7\.1 — (.*?)(?=^## v)", cl, re.S | re.M)
check("CHANGELOG: a v2.7.1 entry, right above v2.7.0", m is not None and cl.index("## v2.7.1") < cl.index("## v2.7.0")
      and "\n## v" not in cl[cl.index("## v2.7.1") + 3:cl.index("## v2.7.0")])
body = m.group(1) if m else ""
check("CHANGELOG v2.7.1 gives the measured numbers of the three measurements",
      all(k in body for k in ("18–28%", "90–94%", "5.9–11.4%", "0 of 400")))


# ============================================================ 2. ACI: infinite intervals and their cause
def series(seed, h=21, T=200):
    """The design of M1: i.i.d. daily returns, target = sum of h returns, forecast = h × expanding mean."""
    rng = np.random.default_rng(seed)
    burn, rows = 250, T + 19 + h
    r = 0.0003 + 0.012 * rng.standard_normal(burn + rows + h + 1)
    cs = np.cumsum(r)
    o = np.arange(burn, burn + rows)
    return pd.Series(cs[o + h] - cs[o], o + h), pd.Series(h * cs[o] / (o + 1), o + h)


# seeds copied from the search output (20261121: 94 infinite intervals; 20261120: none)
a, f = series(20261121)
buf = io.StringIO()
with redirect_stdout(buf):
    out = fc.conformal_intervals(a, f, h=21)
txt = buf.getvalue()
t = out["table"]
ev = t["level"].notna().to_numpy()
inf_rows = ev & np.isinf(t["upper"].to_numpy())
fin = ev & t["actual"].notna().to_numpy() & ~inf_rows & ~t["empty"].to_numpy()
cov_fin_ref = float(np.mean(np.abs(t["actual"] - t["forecast"]).to_numpy()[fin] <= (t["upper"] - t["forecast"]).to_numpy()[fin]))
runs = np.diff(np.flatnonzero(np.diff(np.r_[0, inf_rows.astype(int), 0])))[::2]
msg = next((w["message"] for w in out["warnings"] if "infinite" in w["message"]), "")
act = next((w["action"] for w in out["warnings"] if "infinite" in w["message"]), "")
check("h = 21: n_infinite counts the infinite intervals of the table", out["n_infinite"] == int(inf_rows.sum()) == 94,
      out["n_infinite"])
check("h = 21: every infinite interval is at a level above 1 - alpha (ACI raised it; none from the warm-up)",
      bool(np.all(t["level"].to_numpy()[inf_rows] > 0.95)) and int(t["n_calib"].to_numpy()[inf_rows].min()) > 19)
check("h = 21: coverage_finite is the coverage of the finite intervals with an actual, computed from the table",
      abs(out["coverage_finite"] - cov_fin_ref) < 1e-12 and out["coverage_finite"] < out["coverage"],
      (out["coverage_finite"], cov_fin_ref))
check("h = 21: the finding names the cause, the delay, the longest run and both coverages",
      f"{out['n_infinite']} because ACI raised the level" in msg and "seen 21 steps late" in msg
      and f"longest run of infinite intervals: {int(runs.max())} rows" in msg
      and f"{out['coverage_finite']:.1%}" in msg and f"{out['coverage']:.1%} in all" in msg, msg)
check("h = 21: the action says what a smaller gamma costs (coverage of 'rolling') and that no setting fixes it",
      "a smaller gamma removes them" in act and "method='rolling'" in act and "no setting gives finite intervals" in act
      and "a longer record makes them rarer" not in act, act)
check("h = 21: the old wording ('too few errors for this level', 'a smaller gamma makes them rarer') is gone",
      "too few errors for this level" not in msg and "smaller gamma makes them rarer" not in act)
check("h = 21: show prints the coverage of the finite intervals", f"coverage of the {int(fin.sum())} finite intervals: "
      f"{out['coverage_finite']:.1%} ({out['n_infinite']} infinite, counted as covered above)" in txt, txt[:400])
check("h = 21: show prints the level and action of the finding", "NOTICE: 94 of" in txt and "action: report those rows" in txt)

# two runs of different length (6 and 26 rows; seed copied from the search output): the finding gives the longest
a2, f2 = series(20261131)
o2 = fc.conformal_intervals(a2, f2, h=21, show=False)
m2 = next((w["message"] for w in o2["warnings"] if "infinite" in w["message"]), "")
check("h = 21, two runs of 6 and 26 infinite intervals: the finding gives the longest (26)",
      o2["n_infinite"] == 32 and "longest run of infinite intervals: 26 rows" in m2, m2)

a0, f0 = series(20261120)
o0 = fc.conformal_intervals(a0, f0, h=21, show=False)
check("h = 21 without infinite intervals: no finding about them, n_infinite 0 and coverage_finite = coverage",
      o0["n_infinite"] == 0 and not any("infinite" in w["message"] for w in o0["warnings"])
      and abs(o0["coverage_finite"] - o0["coverage"]) < 1e-12)

# h = 1: a run of 30 growing errors (each one larger than all before, so each is a miss) after 100 small ones pushes the
# ACI level to 1 (deterministic)
e1 = np.r_[np.full(100, 0.1) * np.where(np.arange(100) % 2, 1, -1), 10.0 * np.arange(1, 31), np.full(60, 0.1)]
idx = pd.RangeIndex(len(e1))
o1 = fc.conformal_intervals(pd.Series(e1, idx), pd.Series(np.zeros(len(e1)), idx), h=1, show=False)
m1 = next((w for w in o1["warnings"] if "infinite" in w["message"]), {"message": "", "action": ""})
check("h = 1 with a run of misses: infinite intervals, explained by the raised level, without the delay clause",
      o1["n_infinite"] > 0 and "because ACI raised the level" in m1["message"] and "steps late" not in m1["message"], m1)
check("h = 1: the action is the short one (no long-horizon text)", "a longer record makes them rarer" in m1["action"]
      and "no setting gives" not in m1["action"], m1["action"])
o1r = fc.conformal_intervals(pd.Series(e1, idx), pd.Series(np.zeros(len(e1)), idx), h=1, method="rolling", show=False)
check("'rolling' on the same errors: no infinite interval (the nominal level never has one)", o1r["n_infinite"] == 0
      and not np.isnan(o1r["coverage_finite"]) and abs(o1r["coverage_finite"] - o1r["coverage"]) < 1e-12)

# ============================================================ 3. 'split' with a short record, and NaT in the index
rng = np.random.default_rng(271)
for n_obs in (0, 1, 37):
    aa = pd.Series(np.r_[rng.normal(size=n_obs), [np.nan, np.nan]]); ff = pd.Series(np.zeros(n_obs + 2))
    try:
        oo = fc.conformal_intervals(aa, ff, method="split", show=False)
        w_ = [w for w in oo["warnings"] if w["level"] == "warning"]
        ok = oo["n_evaluated"] == 0 and any("method 'split' calibrates on the first half" in w["message"] for w in w_) \
            and any("at least 38 + 1" in w["action"] or "at least 39" in w["action"] for w in w_)
    except ValueError as e:
        ok, w_ = False, str(e)
    check(f"split with {n_obs} observed row(s): a result with the 'no row could be evaluated' finding, not an exception",
          ok, w_)
aa = pd.Series(np.r_[rng.normal(size=38), [np.nan]]); ff = pd.Series(np.zeros(39))
oo = fc.conformal_intervals(aa, ff, method="split", show=False)
check("split with 38 observed rows (the first that works): rows evaluated, no such finding",
      oo["n_evaluated"] > 0 and not any("calibrates on the first half" in w["message"] for w in oo["warnings"]))
check("split: an explicit calib_size below the minimum or above the observed rows is still refused",
      raises(lambda: fc.conformal_intervals(aa, ff, method="split", calib_size=10, show=False), "below 19")
      and raises(lambda: fc.conformal_intervals(aa, ff, method="split", calib_size=50, show=False), "exceeds the 38"))

dates = pd.date_range("2024-01-01", periods=60, freq="D")
y60 = pd.Series(rng.normal(size=60), dates); f60 = pd.Series(np.zeros(60), dates)
nat_idx = dates.insert(30, pd.NaT).delete(31)
check("conformal_intervals: NaT in the index has its own message",
      raises(lambda: fc.conformal_intervals(y60.set_axis(nat_idx), f60.set_axis(nat_idx), show=False),
             r"1 missing value\(s\) \(NaT\)"))
check("conformal_intervals: an index out of order still gets the time-order message (not the NaT one)",
      raises(lambda: fc.conformal_intervals(y60.iloc[::-1], f60.iloc[::-1], show=False), "time order")
      and not raises(lambda: fc.conformal_intervals(y60.iloc[::-1], f60.iloc[::-1], show=False), "NaT"))
pw = pd.DataFrame({"y": rng.normal(size=60), "x": rng.normal(size=60)}, index=dates)
check("prais_winsten: NaT in the index has its own message",
      raises(lambda: spec.prais_winsten("y ~ x", pw.set_axis(nat_idx), show=False), r"\(NaT\)"))
check("prais_winsten: an index out of order still gets the time-order message",
      raises(lambda: spec.prais_winsten("y ~ x", pw.iloc[::-1], show=False), "not in time order")
      and not raises(lambda: spec.prais_winsten("y ~ x", pw.iloc[::-1], show=False), "NaT"))

# ============================================================ 4. study_card in NFC
nfd = unicodedata.normalize("NFD", "Efeito do preço no café")
nfc = unicodedata.normalize("NFC", nfd)
with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "card.json")
    c1 = reporting.study_card(p, question=nfd, estimand="elasticidade", design="painel", show=False)
    raw = Path(p).read_bytes()
    check("study_card writes texts in NFC", c1["fields"]["question"] == nfc and nfc.encode() in raw
          and nfd.encode() not in raw)
    c2 = reporting.study_card(p, question=nfc, estimand="elasticidade", design="painel", show=False)
    check("study_card: the same text in NFC after NFD is the same plan (unchanged, file untouched)",
          c2["action"] == "unchanged" and Path(p).read_bytes() == raw and c2["sha256"] == c1["sha256"])
    check("study_card: a different text is still refused",
          raises(lambda: reporting.study_card(p, question=nfc + " (revisto)", estimand="elasticidade", design="painel",
                                              show=False), "these fields differ: question"))
    # a card written before v2.7.1 with NFD text (its hash over the NFD content): re-running with NFC is "unchanged"
    p2 = os.path.join(td, "old.json")
    old = {"fields": {"question": nfd, "estimand": "e", "design": "d"}, "created_utc": "2026-11-19T12:00:00+00:00",
           "skill_version": "2.7.0", "amendments": []}
    old["sha256"] = reporting._card_hash(old)
    Path(p2).write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")
    raw2 = Path(p2).read_bytes()
    c3 = reporting.study_card(p2, question=nfc, estimand="e", design="d", show=False)
    check("study_card: an NFD card from v2.7.0 re-run with NFC text is unchanged, intact and not rewritten",
          c3["action"] == "unchanged" and not c3["warnings"] and Path(p2).read_bytes() == raw2)
    with redirect_stdout(io.StringIO()):
        man = reporting.run_manifest(results={"a": 1}, study_card=p2, path=os.path.join(td, "m.json"))
    check("run_manifest still reads that card as intact", man["study_card"]["intact"] is True
          if isinstance(man.get("study_card"), dict) and "intact" in man["study_card"] else
          not any("changed outside" in str(w) for w in man.get("warnings", [])), man.get("study_card"))

# ============================================================ 5. IV: offset in W or Z (M3a)
n = 200
rng = np.random.default_rng(2613)
z1, z2, w0, e, u, v, w1 = rng.normal(size=(7, n))
x = 0.5 * z1 + 0.5 * z2 + 0.3 * w0 + v
y = x + w0 + 0.5 * v + u
Zreal = pd.DataFrame({"z1": z1, "z1b": z1 + 1e-6 * e, "z2": z2})
ref = dg.endogeneity_test(pd.Series(y, name="y"), pd.DataFrame({"w": w0, "w1": w1}), pd.DataFrame({"x": x}), Zreal, show=False)
for off in (1e10, 1e12, 1e14):
    r = dg.endogeneity_test(pd.Series(y, name="y"), pd.DataFrame({"w": off + w0, "w1": w1}), pd.DataFrame({"x": x}), Zreal,
                            show=False)
    check(f"offset {off:g} in an exogenous regressor: instruments that differ by 1e-6·e still count 3 "
          "(review r1-A A4 of v2.6.13; 2 before)", r["n_instruments"] == 3 == ref["n_instruments"], r["n_instruments"])
r = dg.endogeneity_test(pd.Series(y, name="y"), pd.DataFrame({"w": 1e10 + w0, "w1": w1}), pd.DataFrame({"x": x}), Zreal, show=False)
check("offset 1e10 in W: the first-stage F is the one without the offset (1e-6 relative)",
      abs(r["first_stage_F"]["x"] / ref["first_stage_F"]["x"] - 1) < 1e-6, (r["first_stage_F"], ref["first_stage_F"]))
for off in (1e6, 1e10, 1e14):
    Zw = pd.DataFrame({"z1": z1, "zw": 2 * w0 - w1, "z2": z2})       # zw lies in span(W) up to W's rounding
    r = dg.endogeneity_test(pd.Series(y, name="y"), pd.DataFrame({"w": off + w0, "w1": w1}), pd.DataFrame({"x": x}), Zw, show=False)
    check(f"offset {off:g} in W: an instrument that is exactly 2·w − w1 still merges (2 instruments)",
          r["n_instruments"] == 2, r["n_instruments"])
QW_, _, _, _, nz_, _, _, _ = dg._iv_basis(np.column_stack([1e12 + w0, w1]), Zreal.to_numpy(), n)
check("_iv_basis (shared with iv_report): 3 instrument directions with W stored as 1e12 + w", nz_ == 3, nz_)
z3 = rng.normal(size=n)
xr = z1 + 2 * z2 + 1e-6 * e
r = dg.endogeneity_test(pd.Series(xr + u, name="y"), pd.DataFrame({"w": w0}), pd.DataFrame({"x": xr}),
                        pd.DataFrame({"z1": z1, "z2": z2, "z3": 1e12 + z3}), show=False)
check("x = z1 + 2·z2 + 1e-6·e with an instrument stored as 1e12 + z3: x is not 'perfect' (it was before)",
      r["perfect_first_stage"] == [], r["perfect_first_stage"])
xp = 2 * w0 + z1
for off in (0.0, 1e12):
    r = dg.endogeneity_test(pd.Series(xp + u, name="y"), pd.DataFrame({"w": off + w0}), pd.DataFrame({"x": xp}),
                            pd.DataFrame({"z1": z1, "z2": z2}), show=False)
    check(f"x = 2·w + z1 exactly, W with offset {off:g}: still 'perfect'", r["perfect_first_stage"] == ["x"],
          r["perfect_first_stage"])

# ============================================================ 6. iv_report: leverage of the control-function regression
rng = np.random.default_rng(31)
zc = rng.normal(size=n); wc = rng.normal(size=n); uc = rng.normal(size=n)
xc = 0.8 * zc + 0.3 * wc + rng.normal(size=n)
for big, expect in ((None, "none"), (30.0, "warn"), (1e7, "nan")):
    xx = xc.copy()
    if big is not None:
        xx[11] = big
    yy = xx + wc + uc
    d = pd.DataFrame(dict(y=yy, x=xx, z=zc, w=wc))
    r = dg.iv_report("y ~ 1 + w + [x ~ z]", d, show=False)
    et = dg.endogeneity_test(d["y"], d[["w"]], d[["x"]], d[["z"]], cov_type="HC1", show=False)
    p_iv = float(r["tests"]["p"].iloc[0])
    lev_w = [m_ for m_ in r["warnings"] if "control-function regression" in m_]
    if expect == "none":
        check("iv_report, no outlier: no leverage warning for the control-function regression, p equal to "
              "endogeneity_test HC1", not lev_w and abs(p_iv - et["p"]) < 1e-10, (lev_w, p_iv, et["p"]))
    elif expect == "warn":
        # the leverage beyond W of row 11, by another path: statsmodels' hat matrix of [1, w, x, v̂] minus that of [1, w]
        vh = sm.OLS(xx, sm.add_constant(np.column_stack([wc, zc]))).fit().resid
        h_all = sm.OLS(yy, sm.add_constant(np.column_stack([wc, xx, vh]))).fit().get_influence().hat_matrix_diag
        h_w = sm.OLS(yy, sm.add_constant(wc)).fit().get_influence().hat_matrix_diag
        ref_h = float(np.max(h_all - h_w))
        check("iv_report, x[11] = 30: warning with the leverage of the control-function regression (hat matrix by "
              "statsmodels), p unchanged and equal to endogeneity_test HC1",
              len(lev_w) == 1 and f"leverage {ref_h:.2f}" in lev_w[0] and abs(p_iv - et["p"]) < 1e-10,
              (lev_w, ref_h, p_iv, et["p"]))
    else:
        check("iv_report, x[11] = 1e7: leverage 1 in the control-function regression gives NaN and names row 11, "
              "as endogeneity_test", np.isnan(p_iv) and np.isnan(et["p"]) and lev_w and "row(s) 11 " in lev_w[0]
              and et["high_leverage_rows"] == [11], (lev_w, p_iv, et["p"]))

# ============================================================ 7. endogeneity_test with HC0 (M3c)
rng = np.random.default_rng(77)
nn = 120
W5 = pd.DataFrame(rng.normal(size=(nn, 3)), columns=["a", "b", "c"]); Z2 = pd.DataFrame(rng.normal(size=(nn, 2)),
                                                                                       columns=["z1", "z2"])
x5 = Z2.sum(1) * 0.5 + rng.normal(size=nn); y5 = x5 + rng.normal(size=nn)
hc0_txt = "HC0 rejects too often in this test"
for cov, want in (("HC0", True), ("HC1", False), ("HC3", False)):
    r = dg.endogeneity_test(y5.rename("y"), W5, pd.DataFrame({"x": x5}), Z2, cov_type=cov, show=False)
    check(f"endogeneity_test {cov} with k/n = {6 / nn:.2f}: HC0 warning {'present' if want else 'absent'}",
          any(hc0_txt in m_ for m_ in r["warnings"]) == want, r["warnings"])
r = dg.endogeneity_test(y5.rename("y"), W5, pd.DataFrame({"x": x5}), Z2, cov_type="HC0", show=False)
check("the HC0 warning gives the measured sizes and suggests HC3",
      any("5.9-11.4%" in m_ and "cov_type='HC3'" in m_ for m_ in r["warnings"]))
Wbig = pd.DataFrame(rng.normal(size=(60, 9)), columns=[f"w{j}" for j in range(9)])
Zb_ = pd.DataFrame(rng.normal(size=(60, 2)), columns=["z1", "z2"])
xb = Zb_.sum(1) + rng.normal(size=60); yb = xb + rng.normal(size=60)
r = dg.endogeneity_test(yb.rename("y"), Wbig, pd.DataFrame({"x": xb}), Zb_, cov_type="HC0", show=False)
check("HC0 with k/n >= 0.19: one warning (the k/n one), not two",
      sum("HC0" in m_ for m_ in r["warnings"]) == 1 and any("k/n" in m_ for m_ in r["warnings"]), r["warnings"])

# ============================================================ 8. the keys of the IV functions in panel-iv.md
import wooldridge as woo  # noqa: E402

mroz = woo.data("mroz").dropna(subset=["wage"]).assign(expersq=lambda t_: t_.exper ** 2)
piv = read("econometrics/references/panel-iv.md")
blk = piv[piv.index("**What the result holds**"):]
blk = blk[:blk.index("\n\n", blk.index("overid_test(resid_2sls"))] if "overid_test(resid_2sls" in blk else blk
rr = dg.iv_report("lwage ~ 1 + exper + expersq + [educ ~ motheduc + fatheduc]", mroz, show=False)
missing = [k for k in rr if f"`{k}`" not in blk and f'`r["{k}"]`' not in blk]
check("panel-iv.md: every key of iv_report's result is named", not missing, missing)
cols = list(rr["table"].columns) + list(rr["first_stage"].columns) + ["partial F (HC3)"] + list(rr["tests"].columns) \
    + list(rr["anderson_rubin"]) + ["joint_F_at_zero", "joint_p_at_zero", "df"] + list(rr["endogeneity_linearmodels"])
missing = [c for c in cols if f"`{c}`" not in blk]
check("panel-iv.md: every column of table, first_stage and tests and every key of anderson_rubin is named",
      not missing, missing)
for tn in rr["tests"].index:
    check(f"panel-iv.md: the test name '{tn.split(' (')[0]}' is the one in the result",
          f'"{tn.split(" (")[0]} (' in blk, tn)
check("panel-iv.md: the example r[\"tests\"][\"p\"].iloc[0] gives the endogeneity p",
      'r["tests"]["p"].iloc[0]' in blk and rr["tests"].index[0].startswith("endogeneity"))
named = set(re.findall(r"`([^`]+)`", blk[:blk.index("`diagnostics.endogeneity_test")]))
actual = set(rr) | set(cols)
listed_cols = {c for c in named if c in ("OLS", "OLS se", "2SLS", "2SLS se", "2SLS p", "2SLS ci_low", "2SLS ci_high",
                                         "partial F", "partial F p", "partial R²", "strength", "H0", "statistic", "p")}
check("panel-iv.md: the column names it lists exist in the result (the text cannot name a column that is not there)",
      listed_cols <= actual, listed_cols - actual)
et_ = dg.endogeneity_test(mroz.lwage, mroz[["exper", "expersq"]], mroz[["educ"]], mroz[["motheduc", "fatheduc"]],
                          show=False)
ov_ = dg.overid_test(rr["iv"].resids, sm.add_constant(mroz[["exper", "expersq"]]), mroz[["motheduc", "fatheduc"]], 1,
                     show=False)
seg_et = blk[blk.index("`diagnostics.endogeneity_test"):blk.index("`diagnostics.overid_test")]
seg_ov = blk[blk.index("`diagnostics.overid_test"):]
check("panel-iv.md: endogeneity_test's keys, all and only", set(re.findall(r"`(\w+)`", seg_et)) == set(et_), seg_et)
check("panel-iv.md: overid_test's keys, all and only", set(re.findall(r"`(\w+)`", seg_ov)) == set(ov_), seg_ov)
doc = dg.iv_report.__doc__
check("iv_report's docstring names the columns and points to panel-iv.md",
      '"partial F p"' in doc and '"2SLS ci_high"' in doc and "panel-iv.md" in doc and '"H0", "statistic", "p"' in doc)
check("panel-iv.md and the docstring: read the strength from F, not from the HC0 p (6–17%, P(F > 10) at most 0.6%)",
      "6–17%" in blk and "0.6%" in blk and "6-17%" in doc and "0.6%" in doc)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
