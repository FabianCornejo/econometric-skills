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
    last = max((i for i, l_ in enumerate(blk) if re.match(r'^ {6}"', l_) or re.match(r"^ {7}", l_)), default=0)
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


def ent_of(name, blk):
    ENTRIES[name] = entries(blk)
    return ENTRIES[name]


def sub_check(name, ent, key, actual, word="keys", alternatives=False):
    ls = lists(ent.get(key, ""), word)
    ok = (list(actual) in ls) if alternatives else (len(ls) >= 1 and ls[0] == list(actual))
    check(f"{name}: {key!r} {word} as documented", ok, (ls, list(actual)))


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
      "partial F (HC3)" not in rl["first_stage"].columns and rl["high_leverage_rows"] == [5], rl["high_leverage_rows"])
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

states_check()
print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
