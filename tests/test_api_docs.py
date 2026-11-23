"""v2.7.2: the "Returns" block at the top of the docstrings says what the result really holds.

In the validations of v2.6.7 to v2.7.1 the agent read each function's API in its docstring (`grep -n "^def f" -A30`,
or `sed` of the lines below the def), not in the references, and guessed what the docstring did not say: the index of
iv_report's tests ("endogeneity" alone: KeyError), "partial_F" for "partial F", `.bse` on a linearmodels result
(claude/medicoes-v2.7.2.md, M0, in the development repository). For the 13 functions those agents called most, this
file checks, against the result of a real call (never against the source code):
  - the block starts within 12 lines of the `def`, so `grep -A30` shows it;
  - its top-level keys are the result's keys, all and only, in every case listed (both ways);
  - every entry ends within 30 lines of the `def` (review r1-A A2);
  - every `columns [...]`, `keys [...]` and `index [...]` list it gives is the real one, the lists of dicts (warnings)
    included, checked on non-empty lists;
  - each entry marked "only with" an argument is seen absent and present, each "else None" None and not None;
  - the test names of iv_report are tied to the cov_type written next to them, and the special values the block
    names (None, NaN, "unchanged", the high-leverage column) are produced;
  - the access examples it prints or writes run and give the number they claim.

Parsing rule: the block opens with a docstring line that starts with "Returns" and ends at the first blank line; a
top-level entry is a line indented 6 spaces that starts with one or more quoted names; an entry whose text says
"only with" (an argument) may be absent from the result, every other one must be there.

v2.7.3 adds the 21 functions used in a single validation and their twins (claude/medicoes-v2.7.3.md, M1: the agent
guessed wild_cluster_bootstrap's keys with .get(..., fallback) and vecm_fit's beta with a fixed number as fallback),
with the same checks, plus:
  - functions that return a DataFrame: the block's `columns [...]` is the real one, and its `attrs["…"]` entries are
    the result's attrs, all and only;
  - cox's tuple (res, tab): tab's columns, res.hr_table is tab;
  - every "Example (r = the result): …" line runs on a real result (the expressions the agent would copy);
  - the states the blocks name: vecm_fit with rank 0, 1 and k; cointegration's note None and text; panel_bp_lm NaN;
    pesaran_timmermann with n < 10; hausman's classical_cov True, False and None; johansen_trace's column with level
    0.05 and 0.10;
  - rule T (decision D2 of v2.7.3): every public function of econometrics/scripts has a block ("Returns … (v2.7.")
    within 12 lines of its def, unless it is in tests/returns_block_pending.txt, a frozen list that only shrinks.

Run from the repository root:  python3 tests/test_api_docs.py
"""
import ast
import inspect
import io
import os
import re
import sys
import tempfile
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "econometrics" / "scripts"))
import _console  # noqa: E402,F401  (non-ASCII text on a Windows console)
import backtest as bt  # noqa: E402
import estimation as est  # noqa: E402
import multivariate as mv  # noqa: E402
import survival as sv  # noqa: E402
import ts_univariate as tu  # noqa: E402
import data_prep as dp  # noqa: E402
import diagnostics as dg  # noqa: E402
import forecasting as fc  # noqa: E402
import inference as inf  # noqa: E402
import reporting as rp  # noqa: E402

warnings.simplefilter("ignore")
results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  -> {detail}"))


MAX_OFFSET = 12      # lines from the def to the "Returns" line (the shortest grep the agents used was -A12; -A30 typical)
MAX_END = 30         # lines from the def to the block's last entry (review r1-A A2: iv/ols were at +33, outside -A30)
SEEN = {}            # (function, key) -> set of states seen: "present", "absent", "none", "value"


def block(func):
    """(offset of the Returns line from the def, the block's lines) read from the source as the agent sees it."""
    lines, _ = inspect.getsourcelines(func)
    k = next((i for i, l_ in enumerate(lines) if l_.strip().startswith("Returns")), None)
    if k is None:
        return None, []
    out = []
    for l_ in lines[k:]:
        if not l_.strip() or l_.strip() == '"""':
            break
        out.append(l_.rstrip("\n"))
        if l_.rstrip().endswith('"""'):
            break
    return k, out


def entries(blk):
    """{key: text of its entry} for the top-level entries (6-space indent, starting with quoted names)."""
    ent, cur = {}, None
    for l_ in blk[1:]:
        m = re.match(r'^ {6}((?:"[^"]+"(?:, )?)+)(.*)$', l_)
        if m:
            names = re.findall(r'"([^"]+)"', m.group(1))
            cur = names
            for nm in names:
                ent[nm] = m.group(2)
        elif cur and l_.startswith(" " * 7):
            for nm in cur:
                ent[nm] += " " + l_.strip()
    return ent


def lists(text, word):
    """Every Python list literal written after `word` in the text (e.g. columns [...])."""
    out = []
    for m in re.finditer(word + r" (\[[^\]]*\])", text):
        out.append(ast.literal_eval(m.group(1)))
    return out


def head_check(name, func):
    k, blk = block(func)
    check(f"{name}: the Returns block starts within {MAX_OFFSET} lines of the def",
          k is not None and k <= MAX_OFFSET, k)
    # v2.7.3: any entry line (quoted keys, columns, attrs[...], res/tab), its continuations and the Example line
    last = max((i for i, l_ in enumerate(blk) if re.match(r"^ {6}\S", l_) or re.match(r"^ {7}", l_)
                or l_.startswith("    Example (r = the result): ")), default=0)
    check(f"{name}: every entry of the block ends within {MAX_END} lines of the def (grep -A30 shows all of it)",
          k is not None and k + last <= MAX_END, k is not None and k + last)
    return blk


def keys_check(name, blk, res, case=""):
    ent = entries(blk)
    real, doc = set(res), set(ent)
    optional = {k_ for k_, t_ in ent.items() if "only with" in t_}     # present only with an argument, says the entry
    check(f"{name}{case}: the documented keys are the result's keys, all and only",
          real <= doc and doc - real <= optional,
          {"not documented": sorted(real - doc), "not in the result": sorted(doc - real - optional)})
    for k_, t_ in ent.items():
        st = SEEN.setdefault((name, k_), set())
        st.add("absent" if k_ not in res else "present")
        if k_ in res:
            st.add("none" if res[k_] is None else "value")
        # every list of dicts said to have keys [...] (warnings) is checked on the dicts actually returned
        if k_ in res and isinstance(res[k_], list) and "list of dicts with keys" in t_:
            want = lists(t_, "keys")[0]
            bad = [w_ for w_ in res[k_] if not isinstance(w_, dict) or list(w_) != want]
            check(f"{name}{case}: every entry of {k_!r} is a dict with keys {want}", not bad, bad)
            if res[k_]:
                st.add("nonempty")
    return ent


def states_check():
    """Each "only with" entry was seen absent and present, each "else None" entry None and not None, and each list of
    dicts was seen non-empty (review r1-A A7: these were never exercised)."""
    for (fn, k_), st in sorted(SEEN.items()):
        ent = ENTRIES[fn]
        t_ = ent.get(k_, "")
        if "only with" in t_:
            check(f"{fn}: {k_!r} seen absent and present (it is marked 'only with')", {"absent", "present"} <= st, st)
        if "else None" in t_:
            check(f"{fn}: {k_!r} seen None and not None (it says 'else None')", {"none", "value"} <= st, st)
        if "list of dicts with keys" in t_:
            check(f"{fn}: {k_!r} seen with at least one entry (its keys are checked)", "nonempty" in st, st)
        if "none" in st and "None" not in t_ and "as passed" not in t_:
            check(f"{fn}: {k_!r} came back None, so its entry must say when", False, t_)
        if "only with" not in t_ and "absent" in st:
            check(f"{fn}: {k_!r} came back absent although its entry does not say 'only with'", False, t_)


ENTRIES = {}


ENTRIES_BLK = {}


def ent_of(name, blk):
    ENTRIES_BLK[name] = blk
    ENTRIES[name] = entries(blk)
    return ENTRIES[name]


def sub_check(name, ent, key, actual, word="keys", alternatives=False):
    ls = lists(ent.get(key, ""), word)
    ok = (list(actual) in ls) if alternatives else (len(ls) >= 1 and ls[0] == list(actual))
    check(f"{name}: {key!r} {word} as documented", ok, (ls, list(actual)))


# ============================================================ v2.7.3: helpers for DataFrame and tuple results
def df_check(name, blk, res, case=""):
    """columns [...] of the block equal the DataFrame's columns; attrs["…"] entries are its attrs, all and only."""
    txt = " ".join(l_.strip() for l_ in blk)
    cols = lists(txt, "columns")
    check(f"{name}{case}: the block's columns [...] are the DataFrame's columns, in order",
          bool(cols) and cols[0] == list(res.columns), (cols[:1], list(res.columns)))
    doc_attrs = set(re.findall(r'^ {6}attrs\["([^"]+)"\]', "\n".join(blk), flags=re.M))
    check(f"{name}{case}: the attrs the block names are the result's attrs, all and only",
          doc_attrs == set(res.attrs), (sorted(doc_attrs), sorted(res.attrs)))


EXAMPLES = {}        # name -> number of example expressions run


def example_check(name, blk, r, case=""):
    """Each expression of the block's "Example (r = the result): …" line runs on the result r."""
    line = [l_ for l_ in blk if l_.strip().startswith("Example (r = the result): ")]
    exprs = [e_.strip().rstrip(".") for e_ in line[0].split(": ", 1)[1].split("; ")] if line else []
    bad = []
    for e_ in exprs:
        try:
            eval(e_, {"r": r, "np": np, "pd": pd})   # noqa: S307  (the text the agent would copy)
        except Exception as x_:                      # noqa: BLE001
            bad.append((e_, repr(x_)))
    check(f"{name}{case}: the block has an Example line and each of its {len(exprs)} expressions runs on a real result",
          len(line) == 1 and exprs and not bad, (line, bad))
    EXAMPLES[name] = len(exprs)


os.chdir(tempfile.mkdtemp())
rng = np.random.default_rng(72)
n = 300
z1, z2, w, u = (rng.normal(size=n) for _ in range(4))
x = z1 + z2 + w + 0.5 * u + rng.normal(size=n)
y = 0.1 * x + w + u
d = pd.DataFrame(dict(y=y, x=x, z1=z1, z2=z2, w=w))
d.to_csv("d.csv", index=False)

# ============================================================ 1. iv_report
blk = head_check("iv_report", dg.iv_report)
ent_of("iv_report", blk)
r = dg.iv_report("y ~ 1 + w + [x ~ z1 + z2]", d, show=False)
ent = keys_check("iv_report", blk, r)
sub_check("iv_report", ent, "table", r["table"].columns, "columns")
sub_check("iv_report", ent, "first_stage", r["first_stage"].columns, "columns")
sub_check("iv_report", ent, "tests", r["tests"].columns, "columns")
sub_check("iv_report", ent, "anderson_rubin", r["anderson_rubin"].keys(), alternatives=True)
sub_check("iv_report", ent, "endogeneity_linearmodels", r["endogeneity_linearmodels"].keys())
check("iv_report: the tests' index is named as documented", r["tests"].index.name == "test"
      and 'index named "test"' in ent["tests"], r["tests"].index.name)
d2 = d.assign(x2=d.x ** 2 + rng.normal(size=n), z3=rng.normal(size=n))
r2 = dg.iv_report("y ~ 1 + w + [x + x2 ~ z1 + z2 + z3]", d2, show=False)
sub_check("iv_report (two endogenous)", ent, "anderson_rubin", r2["anderson_rubin"].keys(), alternatives=True)
keys_check("iv_report", blk, r2, " (two endogenous)")
check("iv_report (two endogenous): pieces is None and df a pair, as the text below the block says",
      r2["anderson_rubin"]["pieces"] is None and len(r2["anderson_rubin"]["df"]) == 2)
doc = "\n".join(blk)
src = inspect.getdoc(dg.iv_report)
for ct, form in (("robust", "y ~ 1 + w + [x ~ z1 + z2]"), ("unadjusted", "y ~ 1 + w + [x ~ z1 + z2]"),
                 ("robust", "y ~ 1 + w + [x ~ z1]"), ("homoskedastic", "y ~ 1 + w + [x ~ z1 + z2]"),
                 ("heteroskedastic", "y ~ 1 + w + [x ~ z1 + z2]"), ("kernel", "y ~ 1 + w + [x ~ z1 + z2]")):
    rr = dg.iv_report(form, d, cov_type=ct, show=False)
    keys_check("iv_report", blk, rr, f" ({ct}, {form.split('[')[1][:-1]})")
    # review r2 N4: "endogeneity_linearmodels … (robust cov_type), else None" tied to the cov_type
    check(f"iv_report ({ct}): endogeneity_linearmodels is None exactly with unadjusted or homoskedastic",
          (rr["endogeneity_linearmodels"] is None) == (ct in ("unadjusted", "homoskedastic"))
          and "(robust cov_type), else None" in ent["endogeneity_linearmodels"])
    for t_ in rr["tests"].index:
        gen = re.sub(r", \d+ df\)", ", <q> df)", t_)
        # review r1-A A7: the name must be the one written for THIS cov_type: "<name>" (…, ct, …)
        m_ = re.search(re.escape(f'"{gen}"') + r" \(([^)]*)\)", " ".join(l_.strip() for l_ in blk))
        check(f"iv_report ({ct}): the test name {t_!r} is written in the block for cov_type {ct}",
              m_ is not None and ct in [c_.strip() for c_ in m_.group(1).split(",")], (gen, m_ and m_.group(1)))
check("iv_report: the example r[\"tests\"][\"p\"].iloc[0] is in the block and gives the endogeneity p",
      'r["tests"]["p"].iloc[0]' in doc and r["tests"].index[0].startswith("endogeneity (")
      and r["tests"]["p"].iloc[0] == r["tests"].loc["endogeneity (control function, HC1)", "p"])
rx = dg.iv_report("y ~ 1 + w + [x ~ z1]", d, show=False)
check("iv_report: .iloc[1] is the overidentification p when overidentified, and there is no row 1 when exactly "
      "identified ('only then')", ".iloc[1], the overidentification p, only then" in doc
      and r["tests"].index[1].startswith("overidentification (") and len(rx["tests"]) == 1)
check("iv_report: 'iv' and 'ols' are linearmodels results with .std_errors and without .bse, as the block says",
      hasattr(r["ols"], "std_errors") and not hasattr(r["ols"], "bse") and hasattr(r["iv"], "std_errors")
      and ".std_errors" in ent["ols"] and ".bse" in ent["ols"])
# the high-leverage column: added from leverage 0.5 to below 1, before "strength"; not with leverage 1
dh = d.copy()
dh.loc[0, "z1"] = 40.0                                   # one row far out in the instrument: leverage about 0.8
rh = dg.iv_report("y ~ 1 + w + [x ~ z1 + z2]", dh, show=False)
dl = d.assign(zd=0.0)
dl.loc[5, "zd"] = 1.0                                    # an instrument that lives on one row: leverage 1
dl.index = dl.index + 100                                # labels differ from positions (review r2 N4): row label 105
rl = dg.iv_report("y ~ 1 + w + [x ~ z1 + zd]", dl, show=False)
# leverage of row 0 beyond w from the instruments, by statsmodels' hat matrices (another path)
h0 = (sm.add_constant(dh[["w", "z1", "z2"]]).pipe(lambda X: sm.OLS(dh.y, X).fit().get_influence().hat_matrix_diag[0])
      - sm.OLS(dh.y, sm.add_constant(dh[["w"]])).fit().get_influence().hat_matrix_diag[0])
flat = " ".join(src.split())
check(f"iv_report, a row with leverage {h0:.2f} from the instruments: 'partial F (HC3)' added before 'strength', as "
      "the text below the block says", (0.5 <= h0 < 1) and list(rh["first_stage"].columns) ==
      ["partial F", "partial F p", "partial R²", "partial F (HC3)", "strength"]
      and '"partial F (HC3)" is added to first_stage (before "strength") when a row has leverage from 0.5 to below 1'
      in flat, (h0, list(rh["first_stage"].columns)))
check("iv_report, a row with leverage 1: no 'partial F (HC3)' column, the row is in high_leverage_rows",
      "partial F (HC3)" not in rl["first_stage"].columns and rl["high_leverage_rows"] == [105]
      and "list of row labels with leverage 1" in ent["high_leverage_rows"], rl["high_leverage_rows"])
keys_check("iv_report", blk, rh, " (leverage 0.8)")
keys_check("iv_report", blk, rl, " (leverage 1)")
# the printed access line runs and gives each test's p
for form, ct in (("y ~ 1 + w + [x ~ z1 + z2]", "robust"), ("y ~ 1 + w + [x ~ z1]", "unadjusted"),
                 ("y ~ 1 + w + [x + x2 ~ z1 + z2 + z3]", "kernel")):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rs = dg.iv_report(form, d2, cov_type=ct)
    line = [l_ for l_ in buf.getvalue().splitlines() if l_.startswith("In code (r = the result): ")]
    exprs = line[0].split(": ", 1)[1].split("; ") if line else []
    vals = [eval(e_, {"r": rs}) for e_ in exprs]   # noqa: S307  (the text the agent would copy)
    check(f"iv_report show ({ct}, {form.split('[')[1][:-1]}): one 'In code' line, one access per test, each giving "
          "that test's p", len(line) == 1 and len(exprs) == len(rs["tests"])
          and all((v_ == p_) or (np.isnan(v_) and np.isnan(p_)) for v_, p_ in zip(vals, rs["tests"]["p"])), (line, vals))

# ============================================================ 2. endogeneity_test and overid_test
blk = head_check("endogeneity_test", dg.endogeneity_test)
ent_of("endogeneity_test", blk)
et = dg.endogeneity_test(d["y"], d[["w"]], d[["x"]], d[["z1", "z2"]], show=False)
ent = keys_check("endogeneity_test", blk, et)
check("endogeneity_test: first_stage_F is a dict by endogenous column", list(et["first_stage_F"]) == ["x"])
blk = head_check("overid_test", dg.overid_test)
ent_of("overid_test", blk)
ov = dg.overid_test(r["iv"].resids, sm.add_constant(d[["w"]]), d[["z1", "z2"]], 1, show=False)
keys_check("overid_test", blk, ov)

# ============================================================ 3. reporting
blk = head_check("study_card", rp.study_card)
ent = ent_of("study_card", blk)
sc = rp.study_card(question="q", estimand="e", design="d", other_field="x", show=False)   # an unknown name: a notice
keys_check("study_card", blk, sc)
sub_check("study_card", ent, "used", sc["used"].keys())
check("study_card: 'action' is one of the documented values",
      sc["action"] in ("written", "unchanged", "amended") and all(f'"{a_}"' in ent["action"] for a_ in
                                                                  ("written", "unchanged", "amended")))
sc1 = rp.study_card(question="q", estimand="e", design="d", other_field="x", show=False)
check("study_card: run again with the same fields, action 'unchanged'", sc1["action"] == "unchanged")
sc2 = rp.study_card(question="q", estimand="e", design="d2", amend=True, reason="r", show=False)
keys_check("study_card", blk, sc2, " (amended)")
check("study_card (amended): action 'amended' and a list with one amendment",
      sc2["action"] == "amended" and isinstance(sc2["amendments"], list) and len(sc2["amendments"]) == 1)

blk = head_check("run_manifest", rp.run_manifest)
ent = ent_of("run_manifest", blk)
mf = rp.run_manifest(d, files=["d.csv", "nao_existe.csv"], seed=1,
                     results={"b": np.float64(1.5), "iv": {"estimate": 1, "se": 0.1, "partial F": 30.0}},
                     study_card="study_card.json", show=False)
keys_check("run_manifest", blk, mf)
sub_check("run_manifest", ent, "study_card", mf["study_card"].keys())
sub_check("run_manifest", ent, "data", mf["data"]["data"].keys())
check("run_manifest: a missing file is recorded as \"missing\", a DataFrame alone is named \"data\"",
      mf["files"]["nao_existe.csv"] == "missing" and list(mf["data"]) == ["data"])
check("run_manifest: of a dict result only the listed keys are kept; amendments is a count here",
      mf["results"]["iv"] == {"estimate": 1, "se": 0.1} and mf["results"]["b"] == 1.5
      and mf["study_card"]["amendments"] == 1, (mf["results"], mf["study_card"]["amendments"]))
mf0 = rp.run_manifest(d, simulated=True, show=False)
keys_check("run_manifest", blk, mf0, " (no card)")
mf1 = rp.run_manifest(d, simulated=True, study_card="nao_existe.json", show=False)
keys_check("run_manifest", blk, mf1, " (a card that does not exist)")
check("run_manifest: study_card is None without study_card= and with a card that does not exist (with a warning)",
      mf0["study_card"] is None and mf1["study_card"] is None and mf1["warnings"])

blk = head_check("cross_check", rp.cross_check)
ent_of("cross_check", blk)
for case, kw in ((" (two routes, se)", dict(estimates={"a": 1.0, "b": 1.1}, se=0.2)),
                 (" (two routes, no se)", dict(estimates={"a": 1.0, "b": 1.0 + 1e-9})),
                 (" (one route)", dict(estimates={"a": 1.0}, se=0.2)),
                 (" (label)", dict(estimates={"a": 1.0, "b": 1.1}, se=0.2, label="ATT"))):
    cc = rp.cross_check(show=False, **kw)
    ent = keys_check("cross_check", blk, cc, case)
    check(f"cross_check{case}: the table's one column is named by label (default \"estimate\")",
          list(cc["table"].columns) == [kw.get("label", "estimate")])
    check(f"cross_check{case}: kind is one of the documented values",
          cc["kind"] is None or f'"{cc["kind"]}"' in ent["kind"], cc["kind"])
    if case == " (one route)":
        check("cross_check (one route): spread NaN, agree None, note \"\", as documented",
              np.isnan(cc["spread"]) and cc["agree"] is None and cc["note"] == ""
              and "NaN with one route" in ent["spread"] and "None when nothing was checked" in ent["agree"])

# ============================================================ 4. inference and data_prep
blk = head_check("describe_dist", inf.describe_dist)
ent = ent_of("describe_dist", blk)
dd = inf.describe_dist(d["y"], show=False)
keys_check("describe_dist", blk, dd)
sub_check("describe_dist", ent, "summary", dd["summary"].index, "index")
check("describe_dist: estimand 'mean', p NaN, and the CI is the t interval of the mean", dd["estimand"] == "mean"
      and np.isnan(dd["p"]) and abs(dd["ci_high"] - dd["estimate"] - dd["se"] * 1.9679296690653618) < 1e-12,
      dd)  # t(0.975, 299) from scipy

blk = head_check("adjust_pvalues", inf.adjust_pvalues)
ap = inf.adjust_pvalues(pd.Series({"a": 0.01, "b": 0.04, "c": np.nan}))
check("adjust_pvalues: a DataFrame with the documented columns; a Series keeps its index",
      isinstance(ap, pd.DataFrame) and lists(" ".join(blk), "columns")[0] == list(ap.columns)
      and list(ap.index) == ["a", "b", "c"], ap)
try:
    inf.adjust_pvalues({"a": 0.01, "b": 0.04})
    raised = False
except TypeError:
    raised = True
check("adjust_pvalues: a dict is refused, as the block says (pd.Series(d) first)", raised
      and "a dict is not accepted" in " ".join(blk))

blk = head_check("missing_report", dp.missing_report)
ent = ent_of("missing_report", blk)
dm = d.copy()
dm.loc[dm.w > 1, "x"] = np.nan
mr = dp.missing_report(dm, show=False)
keys_check("missing_report", blk, mr, " (no predictors)")
mr2 = dp.missing_report(dm, cols=["y", "x"], predictors=["w"], show=False)
keys_check("missing_report", blk, mr2, " (predictors)")
sub_check("missing_report", ent, "tests", mr2["tests"].columns, "columns")
mr3 = dp.missing_report(dm, cols=["y", "w"], predictors=["w"], show=False)    # no column outside predictors misses
check("missing_report: with no missing column outside predictors, tests is empty and still has its columns (A6)",
      len(mr3["tests"]) == 0 and list(mr3["tests"].columns) == ["share_missing", "F", "p"], mr3["tests"])
check("missing_report: patterns has a column per variable and \"rows\"",
      set(mr["patterns"].columns) == set(dm.columns) | {"rows"} and mr["patterns"].columns[-1] == "rows",
      list(mr["patterns"].columns))

# ============================================================ 5. backtest and forecasting
blk = head_check("walk_forward_splits", bt.walk_forward_splits)
g = bt.walk_forward_splits(100, 50, 10)
sp = list(bt.walk_forward_splits(100, 50, 10))
check("walk_forward_splits: a generator of (train_idx, test_idx) integer position arrays (within 0..n-1), as the "
      "block says", inspect.isgenerator(g) and all(isinstance(t_, tuple) and len(t_) == 2 for t_ in sp)
      and all(np.issubdtype(np.asarray(a_).dtype, np.integer) and 0 <= min(a_) and max(a_) < 100
              for t_ in sp for a_ in t_) and "generator" in " ".join(blk) and ".iloc" in " ".join(blk))

blk = head_check("label_overlap", bt.label_overlap)
ent = ent_of("label_overlap", blk)
lo = bt.label_overlap(21, 500, show=False)
keys_check("label_overlap", blk, lo)
sub_check("label_overlap", ent, "used", lo["used"].keys())

idx = pd.RangeIndex(200)
act = pd.Series(rng.normal(size=200).cumsum() + 50, idx)
f1 = act.shift(1).bfill()
f2 = act.rolling(5, min_periods=1).mean().shift(1).bfill()
blk = head_check("evaluate", fc.evaluate)
txt = " ".join(l_.strip() for l_ in blk)
cols_doc = lists(txt, "columns")[0]
# the rules this test applies below are the ones the block writes (mutation r1: a rule dropped from the text survived)
for rule in ('"MAPE_%" only with direction="level" and no actual equal to 0',
             '"directional_accuracy" and "PT_p_value" only with direction="return" or with last_actual=',
             'the two DM columns only with two or more forecasts (NaN in the benchmark\'s row)'):
    check(f"evaluate: the block writes the rule {rule[:50]!r}…", rule in txt, txt)
act0 = act.copy()
act0.iloc[7] = 0.0
ret = act.diff().fillna(0.1)
for case, a_, fs, kw in (("level, two forecasts", act, {"naive": f1, "ma": f2}, {}),
                         ("level, benchmark ma", act, {"naive": f1, "ma": f2}, {"benchmark": "ma"}),
                         ("level with last_actual", act, {"naive": f1, "ma": f2}, {"last_actual": act.shift(1).bfill()}),
                         ("return", ret, {"naive": ret.shift(1).bfill(), "zero": ret * 0 + 0.01}, {"direction": "return"}),
                         ("an actual equal to 0", act0, {"naive": f1, "ma": f2}, {}),
                         ("one forecast", act, {"naive": f1}, {})):
    ev = fc.evaluate(a_, fs, show=False, **kw)
    bench = kw.get("benchmark", list(fs)[0])
    keep = {"MAPE_%": kw.get("direction", "level") == "level" and bool((a_ != 0).all()),
            "directional_accuracy": kw.get("direction") == "return" or "last_actual" in kw,
            "PT_p_value": kw.get("direction") == "return" or "last_actual" in kw,
            "DM_vs_<benchmark>": len(fs) >= 2, "DM_p": len(fs) >= 2}
    want = [c_.replace("<benchmark>", bench) for c_ in cols_doc if keep.get(c_, True)]
    check(f"evaluate ({case}): the columns the block's rules give, in its order", list(ev.columns) == want,
          (list(ev.columns), want))
    if len(fs) >= 2:
        other = [k_ for k_ in fs if k_ != bench][0]
        check(f"evaluate ({case}): the benchmark's row has NaN in the DM columns; tab.loc[name, 'DM_p'] reads a p",
              ev.loc[bench, [f"DM_vs_{bench}", "DM_p"]].isna().all() and 0 <= ev.loc[other, "DM_p"] <= 1)
try:
    fc.evaluate(act, {"naive": f1, "ma": f2}, direction="sign", show=False)
    msg = ""
except ValueError as e_:
    msg = str(e_)
# v2.7.3: the attrs the block names are the table's, and the Example line runs
doc_attrs = set(re.findall(r'^ {6}attrs\["([^"]+)"\]', "\n".join(blk), flags=re.M))
check("evaluate: the attrs the block names are the table's attrs, all and only (v2.7.3)", doc_attrs == set(ev.attrs),
      (doc_attrs, set(ev.attrs)))
example_check("evaluate", blk, ev)
check("evaluate: a direction other than 'level' or 'return' is refused (it only dropped MAPE_% before)",
      "direction must be 'level' or 'return'" in msg and 'direction is "level" (default) or "return"' in txt, msg)

blk = head_check("conformal_intervals", fc.conformal_intervals)
ent_of("conformal_intervals", blk)
for meth in ("aci", "rolling", "split"):
    ci = fc.conformal_intervals(act, f1, method=meth, show=False)
    ent = keys_check("conformal_intervals", blk, ci, f" ({meth})")
    sub_check(f"conformal_intervals ({meth})", ent, "table", ci["table"].columns, "columns")
    sub_check(f"conformal_intervals ({meth})", ent, "used", ci["used"].keys())
    sub_check(f"conformal_intervals ({meth})", ent, "guarantee", ci["guarantee"].keys())
    check(f"conformal_intervals ({meth}): the table is indexed like actual", ci["table"].index.equals(act.index))

# ============================================================ 6. v2.7.3: diagnostics
rng3 = np.random.default_rng(73)
T3 = 160
xa = rng3.normal(size=T3).cumsum()
ts3 = pd.DataFrame({"y1": xa + rng3.normal(scale=0.5, size=T3), "y2": xa + rng3.normal(scale=0.5, size=T3)},
                   index=pd.date_range("2000-01-31", periods=T3, freq="ME"))
st_ = pd.DataFrame({"s1": rng3.normal(size=T3), "s2": rng3.normal(size=T3)}, index=ts3.index)   # two I(0) series

blk = head_check("stationarity", dg.stationarity)
r_ = dg.stationarity(ts3["y1"], show=False)
df_check("stationarity", blk, r_)
example_check("stationarity", blk, r_)
check("stationarity: 4 rows with index 0-3 in the documented order of Transform and Deterministics",
      list(r_.index) == [0, 1, 2, 3] and list(zip(r_["Transform"], r_["Deterministics"])) ==
      [("level", "constant"), ("level", "constant+trend"), ("1st difference", "constant"),
       ("1st difference", "constant+trend")], list(zip(r_["Transform"], r_["Deterministics"])))
r_w = dg.stationarity(ts3["y1"], show=False, alpha=0.2)
check("stationarity: attrs['warnings'] is a list of str, empty with alpha 0.05 and not empty with alpha 0.2, as the "
      "block says", r_.attrs["warnings"] == [] and r_w.attrs["warnings"] and all(isinstance(w_, str) for w_ in
      r_w.attrs["warnings"]) and "empty unless alpha is outside [0.01, 0.10]" in " ".join(blk))
check("stationarity: suggested_order starts with I(0), I(1) or undetermined, as the block says",
      any(r_.attrs["suggested_order"].startswith(a_) for a_ in ("I(0)", "I(1)", "undetermined")),
      r_.attrs["suggested_order"])

blk = head_check("stationarity_many", dg.stationarity_many)
r_ = dg.stationarity_many(ts3, show=False)
df_check("stationarity_many", blk, r_)
example_check("stationarity_many", blk, r_)
check("stationarity_many: one row per column in order, index 0..k-1, Suggested order = stationarity's attrs",
      list(r_["Series"]) == ["y1", "y2"] and list(r_.index) == [0, 1]
      and r_.loc[0, "Suggested order"] == dg.stationarity(ts3["y1"], show=False).attrs["suggested_order"])

blk = head_check("cointegration", dg.cointegration)
ent = ent_of("cointegration", blk)
for case, dat in ((" (cointegrated random walks)", ts3), (" (two I(0) series)", st_)):
    r_ = dg.cointegration(dat, show=False)
    keys_check("cointegration", blk, r_, case)
    sub_check("cointegration" + case, ent, "engle_granger", r_["engle_granger"].keys())
    sub_check("cointegration" + case, ent, "johansen", r_["johansen"].columns, "columns")
    example_check("cointegration", blk, r_, case)
check("cointegration: johansen_note seen None and as text (full rank with det_order=0)",
      {"none", "value"} <= SEEN[("cointegration", "johansen_note")], SEEN[("cointegration", "johansen_note")])

blk = head_check("chow", dg.chow)
ent_of("chow", blk)
xx = pd.DataFrame({"x": rng3.normal(size=T3)}, index=ts3.index)
yy = 1 + 0.5 * xx["x"] + rng3.normal(size=T3)
r_ = dg.chow(yy, xx, ts3.index[80], show=False)
keys_check("chow", blk, r_)
example_check("chow", blk, r_)
check("chow: n1 + n2 is the sample and n1 the rows before point", r_["n1"] == 80 and r_["n1"] + r_["n2"] == T3)

blk = head_check("diebold_mariano", dg.diebold_mariano)
ent_of("diebold_mariano", blk)
e1, e2 = rng3.normal(size=100), rng3.normal(size=100)
r_ = dg.diebold_mariano(e1, e2, show=False)
keys_check("diebold_mariano", blk, r_)
example_check("diebold_mariano", blk, r_)
r0 = dg.diebold_mariano(e1, e1, show=False)
check("diebold_mariano: identical losses give p NaN with show=False, as the block says", np.isnan(r0["p"])
      and "NaN when not computed" in ent_of("diebold_mariano", blk)["p"])

# panel: Grunfeld-like with an effect correlated with x
from linearmodels.panel import PanelOLS, RandomEffects  # noqa: E402
from types import SimpleNamespace  # noqa: E402
N3, Tp = 12, 10
ids = np.repeat(np.arange(N3), Tp)
tt = np.tile(np.arange(Tp), N3)
a_i = rng3.normal(size=N3)[ids]
xp = a_i + rng3.normal(size=N3 * Tp)
pan = pd.DataFrame({"firm": ids, "year": tt, "x": xp, "z": rng3.normal(size=N3 * Tp)})
pan["y"] = 1 + 0.5 * pan.x + 0.3 * pan.z + a_i + rng3.normal(size=N3 * Tp)
pi = pan.set_index(["firm", "year"])
fe = PanelOLS.from_formula("y ~ 1 + x + z + EntityEffects", pi).fit(cov_type="unadjusted")
re_ = RandomEffects.from_formula("y ~ 1 + x + z", pi).fit(cov_type="unadjusted")
fe_rob = PanelOLS.from_formula("y ~ 1 + x + z + EntityEffects", pi).fit(cov_type="robust")

blk = head_check("hausman", dg.hausman)
ent_of("hausman", blk)
for case, a_, b_ in ((" (unadjusted)", fe, re_), (" (robust FE)", fe_rob, re_),
                     (" (objects without _cov_type)", SimpleNamespace(params=fe.params, cov=fe.cov),
                      SimpleNamespace(params=re_.params, cov=re_.cov)),
                     (" (V_FE = V_RE)", SimpleNamespace(params=fe.params, cov=fe.cov),
                      SimpleNamespace(params=re_.params, cov=fe.cov))):
    r_ = dg.hausman(a_, b_, show=False)
    keys_check("hausman", blk, r_, case)
    example_check("hausman", blk, r_, case)
    if case == " (V_FE = V_RE)":
        check("hausman (V_FE = V_RE): df 0, statistic and p NaN, as the block says",
              r_["df"] == 0 and np.isnan(r_["statistic"]) and np.isnan(r_["p"]), r_)
check("hausman: classical_cov seen True, False and None",
      {True, False, None} == {dg.hausman(fe, re_, show=False)["classical_cov"],
                              dg.hausman(fe_rob, re_, show=False)["classical_cov"],
                              dg.hausman(SimpleNamespace(params=fe.params, cov=fe.cov),
                                         SimpleNamespace(params=re_.params, cov=re_.cov), show=False)["classical_cov"]})

blk = head_check("panel_bp_lm", dg.panel_bp_lm)
ent_of("panel_bp_lm", blk)
pooled = sm.OLS(pan.y, sm.add_constant(pan[["x", "z"]])).fit()
res_p = pd.Series(pooled.resid.to_numpy(), index=pi.index)
r_ = dg.panel_bp_lm(res_p, show=False)
keys_check("panel_bp_lm", blk, r_)
example_check("panel_bp_lm", blk, r_)
one = pd.Series(rng3.normal(size=N3), index=pd.MultiIndex.from_arrays([np.arange(N3), np.zeros(N3, int)]))
r1_ = dg.panel_bp_lm(one, show=False)
check("panel_bp_lm: LM and p finite with a panel, both NaN when every entity has one row, as the block says",
      np.isfinite(r_["LM"]) and np.isfinite(r_["p"]) and np.isnan(r1_["LM"]) and np.isnan(r1_["p"]))

blk = head_check("mundlak_test", dg.mundlak_test)
ent_of("mundlak_test", blk)
r_ = dg.mundlak_test(pan, "y", ["x", "z"], "firm", "year", show=False)
keys_check("mundlak_test", blk, r_)
example_check("mundlak_test", blk, r_)
check("mundlak_test: df = len(x_vary), coefficients = results.params with '<x>_mean', results has .std_errors and no "
      ".bse", r_["df"] == 2 and r_["coefficients"].equals(r_["results"].params)
      and {"x_mean", "z_mean"} <= set(r_["coefficients"].index) and hasattr(r_["results"], "std_errors")
      and not hasattr(r_["results"], "bse"))

# ============================================================ 7. v2.7.3: estimation
pan["g"] = pan.firm
blk = head_check("wild_cluster_bootstrap", est.wild_cluster_bootstrap)
ent_of("wild_cluster_bootstrap", blk)
for case, kw in ((" (ci)", dict(ci=True, ci_points=11)), (" (ci=False)", dict(ci=False)),
                 (" (Rademacher with G = 5)", dict(ci=False, weights="rademacher"))):
    dat = pan[pan.g < 5] if "G = 5" in case else pan        # 2^5 = 32 distinct draws < 2B: the warning
    r_ = est.wild_cluster_bootstrap(dat, "y ~ x + z", "x", "g", B=99, **kw)
    keys_check("wild_cluster_bootstrap", blk, r_, case)
    example_check("wild_cluster_bootstrap", blk, r_, case)
    if case == " (ci=False)":
        check("wild_cluster_bootstrap (ci=False): ci_low and ci_high NaN, as the block says",
              np.isnan(r_["ci_low"]) and np.isnan(r_["ci_high"]))
    if case == " (ci)":
        check("wild_cluster_bootstrap (ci): a finite CI around the estimate, n_clusters 12, the key 'p_cr1_t_G-1'",
              r_["ci_low"] < r_["estimate"] < r_["ci_high"] and r_["n_clusters"] == 12 and "p_cr1_t_G-1" in r_)
    if "Rademacher" in case:
        check("wild_cluster_bootstrap: warnings is a list of str, not empty with Rademacher and G = 5",
              r_["warnings"] and all(isinstance(w_, str) for w_ in r_["warnings"]), r_["warnings"])

blk = head_check("bootstrap", est.bootstrap)
ent = ent_of("bootstrap", blk)
import statsmodels.formula.api as smf  # noqa: E402
for case, fn, kw in ((" (one number, rows)", lambda d_: smf.ols("y ~ x", d_).fit().params["x"], {}),
                     (" (a Series, clusters)", lambda d_: smf.ols("y ~ x + z", d_).fit().params, {"cluster": "g"})):
    r_ = est.bootstrap(fn, pan, B=49, **kw)
    keys_check("bootstrap", blk, r_, case)
    sub_check("bootstrap" + case, ent, "table", r_["table"].columns, "columns")
    if case == " (one number, rows)":
        example_check("bootstrap", blk, r_, case)
        check("bootstrap: one number gives a row named 'stat0'; draws has B_effective rows; method 'pairs bootstrap'",
              list(r_["table"].index) == ["stat0"] and r_["draws"].shape == (r_["B_effective"], 1)
              and r_["method"] == "pairs bootstrap")
    else:
        check("bootstrap: a Series keeps its labels as rows; method 'cluster bootstrap'; a warning with 12 clusters",
              list(r_["table"].index) == ["Intercept", "x", "z"] and r_["method"] == "cluster bootstrap"
              and r_["warnings"] and all(isinstance(w_, str) for w_ in r_["warnings"]))

# ============================================================ 8. v2.7.3: ts_univariate and multivariate
blk = head_check("adf_bootstrap", tu.adf_bootstrap)
ent = ent_of("adf_bootstrap", blk)
for case, kw in ((" (lags by AIC)", {}), (" (lags given)", {"lags": 2})):
    r_ = tu.adf_bootstrap(ts3["y1"], B=49, show=False, **kw)
    keys_check("adf_bootstrap", blk, r_, case)
    sub_check("adf_bootstrap" + case, ent, "bootstrap_critical_values", r_["bootstrap_critical_values"].keys())
    sub_check("adf_bootstrap" + case, ent, "asymptotic_critical_values", r_["asymptotic_critical_values"].keys())
    example_check("adf_bootstrap", blk, r_, case)
    check(f"adf_bootstrap{case}: lag_rule is the rule or 'given'",
          r_["lag_rule"] == ("given" if "lags" in kw else "aic") and r_["verdict"].endswith(" at 5%"), r_["lag_rule"])

blk = head_check("sup_wald", tu.sup_wald)
ent = ent_of("sup_wald", blk)
yb = pd.Series(np.r_[rng3.normal(size=80), 2 + rng3.normal(size=80)], index=ts3.index)
for case, kw in ((" (mean)", {}), (" (with X)", {"X": xx})):
    r_ = tu.sup_wald(yb, B=49, show=False, **kw)
    keys_check("sup_wald", blk, r_, case)
    sub_check("sup_wald" + case, ent, "cusum", r_["cusum"].keys())
    sub_check("sup_wald" + case, ent, "bootstrap_critical_values", r_["bootstrap_critical_values"].keys())
    example_check("sup_wald", blk, r_, case)
    check(f"sup_wald{case}: F_path indexed like y; recursive_coefficients columns 'const' then X's names; break_date "
          "is a label of y", r_["F_path"].index.equals(yb.index) and list(r_["recursive_coefficients"].columns)
          == ["const"] + (list(xx.columns) if "X" in kw else []) and r_["break_date"] in yb.index
          and yb.index[r_["break_position"]] == r_["break_date"])

blk = head_check("johansen_trace", mv.johansen_trace)
ent = ent_of("johansen_trace", blk)
for lev, col in ((0.05, "crit 95% (simulated)"), (0.10, "crit 90% (simulated)")):
    r_ = mv.johansen_trace(ts3, level=lev, reps=49, show=False)
    keys_check("johansen_trace", blk, r_, f" (level {lev})")
    cols_doc = [c_.replace("<1-level>", f"{1 - lev:.0%}") for c_ in lists(ent["table"], "columns")[0]]
    check(f"johansen_trace (level {lev}): the table's columns, the critical-value one named {col!r} as the block says",
          list(r_["table"].columns) == cols_doc and col in r_["table"].columns and f'"{col}"' in ent["table"],
          list(r_["table"].columns))
    example_check("johansen_trace", blk, r_, f" (level {lev})")

blk = head_check("vecm_fit", mv.vecm_fit)
ent = ent_of("vecm_fit", blk)
for rank in (0, 1, 2):
    r_ = mv.vecm_fit(ts3, k_ar_diff=1, rank=rank, reps=49, show=False)
    keys_check("vecm_fit", blk, r_, f" (rank {rank})")
    check(f"vecm_fit (rank {rank}): {'five keys with message' if rank in (0, 2) else 'no message'}, as the block says",
          (len(r_) == 5 and "message" in r_) if rank in (0, 2) else ("message" not in r_ and len(r_) == 17), list(r_))
    if rank == 1:
        b_ = r_["beta"]
        check("vecm_fit (rank 1): beta indexed by df's names with columns ['ect1'], first entry 1; alpha, se_alpha, "
              "p_alpha with the same index and columns",
              list(b_.index) == ["y1", "y2"] and list(b_.columns) == ["ect1"] and abs(b_.iloc[0, 0] - 1) < 1e-12
              and all(r_[k_].index.equals(b_.index) and r_[k_].columns.equals(b_.columns)
                      for k_ in ("alpha", "se_alpha", "p_alpha")), b_)
        sub_check("vecm_fit (rank 1)", ent, "adjustment", r_["adjustment"].columns, "columns")
        sub_check("vecm_fit (rank 1)", ent, "weak_exogeneity", r_["weak_exogeneity"].columns, "columns")
        check("vecm_fit (rank 1): trace_test is johansen_trace's table, the critical-value column as named",
              "crit 95% (simulated)" in r_["trace_test"].columns)
        example_check("vecm_fit", blk, r_, " (rank 1)")

# ============================================================ 8b. v2.7.3 review r1-A: states and values of the blocks
from scipy import stats as _st  # noqa: E402
from statsmodels.tsa.stattools import adfuller  # noqa: E402

# A1: y with a NaN at the start (y.diff()): F_path and break_position refer to y without its NaN ends
yd = pd.Series(np.r_[rng3.normal(size=80), 2 + rng3.normal(size=80)], index=ts3.index).cumsum().diff()
r_ = tu.sup_wald(yd, B=19, show=False)
nn_ = len(yd.dropna())
lo_, hi_ = int(np.floor(0.15 * nn_)), int(np.ceil(0.85 * nn_)) - 1
inside = np.zeros(nn_, bool)
inside[lo_:hi_ + 1] = True
check("sup_wald (y.diff()): F_path indexed like y without its NaN start, break_position its position there, F_path NaN "
      "exactly outside the trimmed window (review r1-A A1, M9)",
      r_["F_path"].index.equals(yd.dropna().index) and r_["F_path"].index[r_["break_position"]] == r_["break_date"]
      and r_["F_path"].isna().to_numpy().tolist() == (~inside).tolist(), (len(r_["F_path"]), r_["break_position"]))
# A3: X that already has a constant (a full set of dummies): no "const" column, and the warning
Xd_ = pd.get_dummies(pd.Series(np.arange(T3) % 4, index=ts3.index), prefix="q").astype(float)
r_ = tu.sup_wald(yb, X=Xd_, B=19, show=False)
check("sup_wald, X with a constant (dummies): recursive_coefficients has only X's names, and the warning says so",
      list(r_["recursive_coefficients"].columns) == list(Xd_.columns)
      and any("already contains a constant" in w_ for w_ in r_["warnings"])
      and "only X's names when X already contains a constant" in ENTRIES["sup_wald"]["recursive_coefficients"])

# A2: a constant, non-zero loss difference: NaN, never "significant"
r_, out_ = None, None
buf = io.StringIO()
with redirect_stdout(buf):
    r_ = dg.diebold_mariano(np.full(50, 2.0), np.full(50, 1.0))
check("diebold_mariano, constant non-zero loss difference: DM_HLN and p NaN, the text says constant, not significant "
      "(review r1-A A2)", np.isnan(r_["p"]) and np.isnan(r_["DM_HLN"]) and "constant" in buf.getvalue()
      and "significant" not in buf.getvalue(), (r_, buf.getvalue()))
# the sign and mean_loss_diff from their definition (A7 M8, M21)
e_bad, e_good = 3 * rng3.normal(size=200), rng3.normal(size=200)
r_ = dg.diebold_mariano(e_bad, e_good, show=False)
check("diebold_mariano: model 1 clearly worse gives a positive statistic (the block: negative = model 1 better), and "
      "mean_loss_diff = mean(e1² − e2²)", r_["DM_HLN"] > 0 and abs(r_["mean_loss_diff"] - np.mean(e_bad ** 2 - e_good ** 2))
      < 1e-10 * np.mean(e_bad ** 2), r_)

# A4: vecm_fit refuses a rank outside 0..k; three series: rank 2 normalised on the first two
try:
    mv.vecm_fit(ts3, k_ar_diff=1, rank=3, reps=19, show=False)
    raised = False
except ValueError:
    raised = True
check("vecm_fit: rank 3 with 2 series is refused (review r1-A A4)", raised)
ts3c = ts3.assign(y3=ts3.y1 + rng3.normal(scale=0.5, size=T3))
r_ = mv.vecm_fit(ts3c, k_ar_diff=1, rank=2, reps=19, show=False)
keys_check("vecm_fit", ENTRIES_BLK["vecm_fit"], r_, " (three series, rank 2)")
check("vecm_fit (three series, rank 2): beta columns ['ect1', 'ect2'], its first 2 × 2 block the identity",
      list(r_["beta"].columns) == ["ect1", "ect2"] and np.allclose(r_["beta"].iloc[:2, :2].to_numpy(), np.eye(2)), r_["beta"])

# A5 and A6: hausman valid False (V_FE − V_RE negative definite) and df from the rank
ns_ = lambda p_, c_: SimpleNamespace(params=p_, cov=c_)   # noqa: E731
r_ = dg.hausman(ns_(fe.params, fe.cov), ns_(re_.params, re_.cov * 0 + fe.cov.to_numpy() * 2), show=False)
check("hausman, V_FE − V_RE not positive semidefinite: valid False, a negative statistic, p 1.0, 'Not interpretable' "
      "(review r1-A A5)", r_["valid"] is False and r_["statistic"] < 0 and r_["p"] == 1.0
      and r_["conclusion"].startswith("Not interpretable") and "can be negative" in ENTRIES["hausman"]["statistic"], r_)
cm_ = [c_ for c_ in fe.params.index if c_ in re_.params.index and c_.lower() not in ("const", "intercept")]
v_ = 0.5 * np.sqrt(np.diag(fe.cov.loc[cm_, cm_]))
cre = fe.cov.copy()
cre.loc[cm_, cm_] = fe.cov.loc[cm_, cm_].to_numpy() - np.outer(v_, v_) * [[1, 0], [0, 0]]
r_ = dg.hausman(ns_(fe.params, fe.cov), ns_(re_.params, cre), show=False)
check("hausman: V_FE − V_RE of rank 1 gives df 1 (the rank, not the number of coefficients; A7 M22)", r_["df"] == 1, r_)

# A6: cointegration lag_sensitive True (seed 8 from the search: AR(0.9) deviations, 120 periods)
rs8 = np.random.default_rng(8)
x8 = rs8.normal(size=120).cumsum()
u8 = np.zeros(120)
for t_ in range(1, 120):
    u8[t_] = 0.9 * u8[t_ - 1] + rs8.normal()
r_ = dg.cointegration(pd.DataFrame({"a": x8 + u8, "b": x8}), show=False)
check("cointegration: lag_sensitive True when AIC and BIC disagree at 5%, and the conclusion says Inconclusive",
      r_["engle_granger"]["lag_sensitive"] is True and r_["engle_granger"]["conclusion"].startswith("Inconclusive")
      and (r_["engle_granger"]["p"] < 0.05) != (r_["engle_granger"]["p_bic"] < 0.05), r_["engle_granger"])

# A6/A7: johansen_trace rank = first r0 not rejected; drift_used 0.0 unless "co"
for det in ("ci", "co"):
    r_ = mv.johansen_trace(ts3, deterministic=det, reps=19, show=False)
    first = next((i_ for i_, rej in enumerate(r_["table"]["reject"]) if not rej), len(r_["table"]))
    check(f"johansen_trace ({det}): rank is the first r0 not rejected; drift_used {'> 0' if det == 'co' else '0.0'}",
          r_["rank"] == first and ((r_["drift_used"] > 0) if det == "co" else (r_["drift_used"] == 0.0)), r_["drift_used"])

# A6: mundlak_test balanced False
r_ = dg.mundlak_test(pan.drop(index=[0, 1, 2]), "y", ["x", "z"], "firm", "year", show=False)
check("mundlak_test: balanced False with an unbalanced panel", r_["balanced"] is False)

# A6/A7: bootstrap with failed replications; percentile limits from the draws
calls = {"n": 0}


def flaky(d_):
    calls["n"] += 1
    if calls["n"] > 1 and calls["n"] % 3 == 0:
        raise RuntimeError("fails")
    return d_["x"].mean()


r_ = est.bootstrap(flaky, pan, B=60, alpha=0.1)
check("bootstrap: failed replications dropped (B_effective < B, draws finite), a warning, ci_low/high the percentile "
      "limits of the draws", r_["B_effective"] == 40 and np.isfinite(r_["draws"]).all() and r_["warnings"]
      and abs(r_["table"]["ci_low"].iloc[0] - np.quantile(r_["draws"][:, 0], 0.05)) < 1e-12
      and abs(r_["table"]["ci_high"].iloc[0] - np.quantile(r_["draws"][:, 0], 0.95)) < 1e-12, r_["B_effective"])

# A7: wild_cluster_bootstrap's t uses null, and p_cr1_t_G-1 is the t(G − 1) p
r_ = est.wild_cluster_bootstrap(pan, "y ~ x + z", "x", "g", null=0.1, B=99, ci=False)
check("wild_cluster_bootstrap, null 0.1: t = (estimate − null)/se; p_cr1_t_G-1 = 2·t(G − 1).sf(|t|)",
      abs(r_["t"] - (r_["estimate"] - 0.1) / r_["se"]) < 1e-12
      and abs(r_["p_cr1_t_G-1"] - 2 * _st.t.sf(abs(r_["t"]), r_["n_clusters"] - 1)) < 1e-12, r_)

# A7: panel_bp_lm p = chi2(1).sf(LM); stationarity's alpha attr and the 0.10 edge; adf p_asymptotic from adfuller
r_ = dg.panel_bp_lm(res_p, show=False)
check("panel_bp_lm: p is the chi²(1) tail of LM", abs(r_["p"] - _st.chi2.sf(r_["LM"], 1)) < 1e-12)
check("stationarity: attrs['alpha'] is the level passed (0.2)", r_w.attrs["alpha"] == 0.2)
check("stationarity: no warning at alpha 0.10 (inside [0.01, 0.10]), a warning just above",
      dg.stationarity(ts3["y1"], show=False, alpha=0.10).attrs["warnings"] == []
      and dg.stationarity(ts3["y1"], show=False, alpha=0.1001).attrs["warnings"] != [])
r_ = tu.adf_bootstrap(ts3["y1"], lags=2, B=19, show=False)
check("adf_bootstrap: p_asymptotic is adfuller's with the same lags and trend",
      abs(r_["p_asymptotic"] - adfuller(ts3["y1"].to_numpy(), maxlag=2, regression="c", autolag=None)[1]) < 1e-12)
y_i2 = pd.Series(np.random.default_rng(5).normal(size=100).cumsum().cumsum())   # seed 5 from the search: I(2)
r_ = tu.adf_bootstrap(y_i2, B=19, show=False)
check("adf_bootstrap: an I(2) series gives the warning about the AR on the differences (a non-empty list of str)",
      r_["warnings"] and all(isinstance(w_, str) for w_ in r_["warnings"]), r_["warnings"])

# A12: stationarity_many with no column keeps its columns
check("stationarity_many: an empty frame gives the two columns and no row",
      list(dg.stationarity_many(ts3.iloc[:, :0], show=False).columns) == ["Series", "Suggested order"])

# ============================================================ 9. v2.7.3: survival
n4 = 150
sd = pd.DataFrame({"id": np.arange(n4), "x": rng3.normal(size=n4), "grp": rng3.integers(0, 2, n4),
                   "reg": rng3.integers(0, 6, n4)})
sd["t"] = np.ceil(rng3.exponential(np.exp(-0.5 * sd.x) * 8)).clip(1, 20)
sd["e"] = (rng3.uniform(size=n4) < 0.7).astype(int)

blk = head_check("cox", sv.cox)
txt = " ".join(l_.strip() for l_ in blk)
for case, kw in ((" (robust)", {}), (" (cluster)", {"cluster": "reg"}), (" (model-based)", {"robust": False})):
    r_ = sv.cox(sd, "x + C(grp)", "t", "e", show=False, **kw)
    check(f"cox{case}: a tuple (res, tab); tab's columns as documented; res.hr_table is tab; tab indexed by the "
          "design's names", isinstance(r_, tuple) and len(r_) == 2 and list(r_[1].columns) == lists(txt, "columns")[0]
          and r_[0].hr_table is r_[1] and list(r_[1].index) == ["C(grp)[T.1]", "x"]
          and "res, tab = cox(" in txt, (type(r_), list(r_[1].index)))
    example_check("cox", blk, r_, case)
check("cox: HR = exp(coef), as the block says", np.allclose(r_[1]["HR"], np.exp(r_[1]["coef"])))

blk = head_check("ph_test", sv.ph_test)
res_c, tab_c = sv.cox(sd, "x + C(grp)", "t", "e", show=False)
buf = io.StringIO()
with redirect_stdout(buf):
    r_ = sv.ph_test(res_c, sd, "t", "e", show=False)
check("ph_test (D4): show=False prints nothing", buf.getvalue() == "", buf.getvalue()[:200])
buf = io.StringIO()
with redirect_stdout(buf):
    r2_ = sv.ph_test(res_c, sd, "t", "e")
check("ph_test: the default still prints the table, with the same numbers",
      "Proportional-hazards check" in buf.getvalue() and r2_.equals(r_))
df_check("ph_test", blk, r_)
example_check("ph_test", blk, r_)
check("ph_test: index named 'covariate' with cox's tab names", r_.index.name == "covariate"
      and list(r_.index) == list(tab_c.index) and 'index named "covariate"' in " ".join(blk))

blk = head_check("person_period", sv.person_period)
r_ = sv.person_period(sd, "id", "t", "e")
check("person_period: columns id, period, y, then df's other columns except duration and event, index 0..N-1",
      list(r_.columns) == ["id", "period", "y", "x", "grp", "reg"] and r_.index.equals(pd.RangeIndex(len(r_)))
      and re.findall(r'"([^"]+)"', " ".join(blk).split("columns:")[1].split("Example")[0]) == ["period", "y"],
      list(r_.columns))   # mutation r0: '"y"' anywhere in the block matched the Example line
check("person_period: y is 1 only in the last period of an event spell; period runs 1..ceil(duration)",
      r_["y"].sum() == sd["e"].sum() and r_.groupby("id")["period"].max().equals(sd.set_index("id")["t"].astype(int)))
example_check("person_period", blk, r_)

blk = head_check("kaplan_meier", sv.kaplan_meier)
ent = ent_of("kaplan_meier", blk)
for case, kw in ((" (overall)", {}), (" (by group, times)", {"group": "grp", "times": [3, 6]})):
    r_ = sv.kaplan_meier(sd, "t", "e", show=False, **kw)
    keys_check("kaplan_meier", blk, r_, case)
    base_cols = lists(ent["summary"], "columns")[0]
    extra = [f"S({t_})" for t_ in kw.get("times", [])]
    check(f"kaplan_meier{case}: summary's columns as documented (+ S(<t>) per time), index named 'group', curves "
          "keyed like it with the documented columns",
          list(r_["summary"].columns) == base_cols + extra and r_["summary"].index.name == "group"
          and list(r_["curves"]) == list(r_["summary"].index)
          and all(list(c_.columns) == lists(ent["curves"], "columns")[0] and c_.index.name == "Time"
                  for c_ in r_["curves"].values()), list(r_["summary"].columns))
    example_check("kaplan_meier", blk, r_, case)
check("kaplan_meier (overall): the index is 'all'", list(sv.kaplan_meier(sd, "t", "e", show=False)["summary"].index)
      == ["all"])
late = pd.DataFrame({"t": np.arange(1, 41, dtype=float), "e": [1] * 5 + [0] * 35})
rk = sv.kaplan_meier(late, "t", "e", show=False)["summary"]
check("kaplan_meier: median and its CI NaN when the curve does not reach 0.5, as the block says",
      rk[["median_survival", "median_ci_low", "median_ci_high"]].isna().all(axis=None)
      and "NaN" in ent["summary"])

blk = head_check("logrank", sv.logrank)
ent_of("logrank", blk)
r_ = sv.logrank(sd, "t", "e", "grp")
keys_check("logrank", blk, r_)
example_check("logrank", blk, r_)
check("logrank: df = groups − 1", r_["df"] == 1)

# ============================================================ 10. v2.7.3: forecasting (PT and forecast_metrics)
blk = head_check("pesaran_timmermann", fc.pesaran_timmermann)
ent_of("pesaran_timmermann", blk)
a5, f5 = rng3.normal(size=120), rng3.normal(size=120)
r_ = fc.pesaran_timmermann(a5, f5)
keys_check("pesaran_timmermann", blk, r_)
example_check("pesaran_timmermann", blk, r_)
rs_ = fc.pesaran_timmermann(a5[:8], f5[:8])
check("pesaran_timmermann: with n < 10 every value but n is NaN, as the block says",
      rs_["n"] == 8 and all(np.isnan(v_) for k_, v_ in rs_.items() if k_ != "n")
      and "With n < 10 every value but n is NaN" in " ".join(blk))

blk = head_check("forecast_metrics", fc.forecast_metrics)
ent_of("forecast_metrics", blk)
lv = pd.Series(50 + rng3.normal(size=100).cumsum())
for case, a_, f_, kw in ((" (level)", lv, lv.shift(1).bfill(), {}),
                         (" (level, naive, MASE, last_actual)", lv, lv.shift(1).bfill() + 0.1,
                          {"naive": lv.shift(1).bfill(), "insample_scale": 1.0, "last_actual": lv.shift(1).bfill()}),
                         (" (return)", pd.Series(a5), pd.Series(f5), {"direction": "return"}),
                         (" (level with a zero actual)", lv.where(lv.index != 5, 0.0), lv.shift(1).bfill(), {})):
    r_ = fc.forecast_metrics(a_, f_, **kw)
    keys_check("forecast_metrics", blk, r_, case)
    example_check("forecast_metrics", blk, r_, case)

# ============================================================ 11. v2.7.3: rule T (decision D2)
def _public_functions():
    """{"module.function": offset of its block's Returns line from the def, or None} from the source (AST, no import)."""
    out = {}
    for f_ in sorted((ROOT / "econometrics" / "scripts").glob("*.py")):
        if f_.name.startswith("_"):
            continue
        src = f_.read_text(encoding="utf-8")
        lines_ = src.splitlines()
        for node in ast.parse(src).body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                body = lines_[node.lineno - 1:node.end_lineno]
                k_ = next((i for i, l_ in enumerate(body) if l_.strip().startswith("Returns") and "(v2.7." in l_), None)
                out[f"{f_.stem}.{node.name}"] = k_
    return out


PENDING = ROOT / "tests" / "returns_block_pending.txt"
pending = [l_.strip() for l_ in PENDING.read_text(encoding="utf-8").splitlines()
           if l_.strip() and not l_.startswith("#")]
pub = _public_functions()
with_block = {k_ for k_, v_ in pub.items() if v_ is not None and v_ <= MAX_OFFSET}
check("rule T: the pending list has no repeated names and is sorted", pending == sorted(set(pending)), pending[:5])
check("rule T: every public function without a Returns block within 12 lines of its def is in "
      "tests/returns_block_pending.txt (a new public function needs a block)",
      set(pub) - with_block <= set(pending), sorted(set(pub) - with_block - set(pending)))
check("rule T: no function with a block is still in the pending list (the list only shrinks: remove it)",
      not (with_block & set(pending)), sorted(with_block & set(pending)))
check("rule T: every name in the pending list is a public function that exists",
      set(pending) <= set(pub), sorted(set(pending) - set(pub)))
check("rule T: a block marked '(v2.7.' further than 12 lines from its def does not count",
      all(v_ is None or v_ <= MAX_OFFSET for v_ in pub.values()),
      {k_: v_ for k_, v_ in pub.items() if v_ is not None and v_ > MAX_OFFSET})
check(f"rule T: the 21 functions of v2.7.3 and the 13 of v2.7.2 have a block ({len(with_block)} of {len(pub)} public "
      "functions)", {"diagnostics.stationarity", "multivariate.vecm_fit",
                     "survival.cox", "forecasting.forecast_metrics", "estimation.bootstrap",
                     "diagnostics.iv_report"} <= with_block and len(with_block) >= 34, len(with_block))   # a count that grows: >=

states_check()
print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
