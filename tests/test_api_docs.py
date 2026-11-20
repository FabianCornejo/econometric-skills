"""v2.7.2: the "Returns" block at the top of the docstrings says what the result really holds.

In the validations of v2.6.7 to v2.7.1 the agent read each function's API in its docstring (`grep -n "^def f" -A30`,
or `sed` of the lines below the def), not in the references, and guessed what the docstring did not say: the index of
iv_report's tests ("endogeneity" alone: KeyError), "partial_F" for "partial F", `.bse` on a linearmodels result
(claude/medicoes-v2.7.2.md, M0, in the development repository). For the 13 functions those agents called most, this
file checks, against the result of a real call (never against the source code):
  - the block starts within 12 lines of the `def`, so `grep -A30` shows it;
  - its top-level keys are the result's keys, all and only, in every case listed (both ways);
  - every `columns [...]`, `keys [...]` and `index [...]` list it gives is the real one;
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
    check(f"{name}: the Returns block starts within {MAX_OFFSET} lines of the def (grep -A30 shows it)",
          k is not None and k <= MAX_OFFSET, k)
    return blk


def keys_check(name, blk, res, case=""):
    ent = entries(blk)
    real, doc = set(res), set(ent)
    optional = {k_ for k_, t_ in ent.items() if "only with" in t_}     # present only with an argument, says the entry
    check(f"{name}{case}: the documented keys are the result's keys, all and only",
          real <= doc and doc - real <= optional,
          {"not documented": sorted(real - doc), "not in the result": sorted(doc - real - optional)})
    return ent


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
r = dg.iv_report("y ~ 1 + w + [x ~ z1 + z2]", d, show=False)
ent = keys_check("iv_report", blk, r)
sub_check("iv_report", ent, "table", r["table"].columns, "columns")
sub_check("iv_report", ent, "first_stage", r["first_stage"].columns, "columns")
sub_check("iv_report", ent, "tests", r["tests"].columns, "columns")
sub_check("iv_report", ent, "anderson_rubin", r["anderson_rubin"].keys(), alternatives=True)
sub_check("iv_report", ent, "endogeneity_linearmodels", r["endogeneity_linearmodels"].keys())
check("iv_report: the tests' index is named as documented", r["tests"].index.name == "test"
      and 'index name "test"' in ent["tests"], r["tests"].index.name)
d2 = d.assign(x2=d.x ** 2 + rng.normal(size=n), z3=rng.normal(size=n))
r2 = dg.iv_report("y ~ 1 + w + [x + x2 ~ z1 + z2 + z3]", d2, show=False)
sub_check("iv_report (two endogenous)", ent, "anderson_rubin", r2["anderson_rubin"].keys(), alternatives=True)
keys_check("iv_report", blk, r2, " (two endogenous)")
doc = "\n".join(blk)
labels = []
for ct, form in (("robust", "y ~ 1 + w + [x ~ z1 + z2]"), ("unadjusted", "y ~ 1 + w + [x ~ z1 + z2]"),
                 ("robust", "y ~ 1 + w + [x ~ z1]"), ("homoskedastic", "y ~ 1 + w + [x ~ z1 + z2]")):
    rr = dg.iv_report(form, d, cov_type=ct, show=False)
    keys_check("iv_report", blk, rr, f" ({ct}, {form.split('[')[1][:-1]})")
    for t_ in rr["tests"].index:
        labels.append(t_)
        gen = re.sub(r", \d+ df\)", ", <q> df)", t_)
        check(f"iv_report ({ct}): the test name {t_!r} is written in the block", f'"{gen}"' in doc, gen)
check("iv_report: the example r[\"tests\"][\"p\"].iloc[0] is in the block and gives the endogeneity p",
      'r["tests"]["p"].iloc[0]' in doc and r["tests"].index[0].startswith("endogeneity (")
      and r["tests"]["p"].iloc[0] == r["tests"].loc["endogeneity (control function, HC1)", "p"])
check("iv_report: r[\"tests\"][\"p\"].iloc[1] is the overidentification p, as the block says",
      'r["tests"]["p"].iloc[1]' in doc and r["tests"].index[1].startswith("overidentification ("))
check("iv_report: 'iv' and 'ols' are linearmodels results with .std_errors and without .bse, as the block says",
      hasattr(r["ols"], "std_errors") and not hasattr(r["ols"], "bse") and hasattr(r["iv"], "std_errors")
      and ".std_errors" in ent["ols"] and ".bse" in ent["ols"])
# the printed access line runs and gives each test's p
buf = io.StringIO()
with redirect_stdout(buf):
    rs = dg.iv_report("y ~ 1 + w + [x ~ z1 + z2]", d)
line = [l_ for l_ in buf.getvalue().splitlines() if l_.startswith("In code: ")]
exprs = line[0][len("In code: "):].split("; ") if line else []
vals = [eval(e_, {"r": rs}) for e_ in exprs]   # noqa: S307  (the text the agent would copy)
check("iv_report show: one 'In code:' line, one access per test, each giving that test's p",
      len(line) == 1 and len(exprs) == len(rs["tests"]) and
      all(v_ == p_ for v_, p_ in zip(vals, rs["tests"]["p"])), (line, vals))
buf = io.StringIO()
with redirect_stdout(buf):
    dg.iv_report("y ~ 1 + w + [x ~ z1]", d, cov_type="unadjusted")
check("iv_report show (exactly identified, unadjusted): the access line names the Wu-Hausman row only",
      'In code: r["tests"].loc["endogeneity (Wu-Hausman)", "p"]\n' in buf.getvalue(), buf.getvalue()[-400:])

# ============================================================ 2. endogeneity_test and overid_test
blk = head_check("endogeneity_test", dg.endogeneity_test)
et = dg.endogeneity_test(d["y"], d[["w"]], d[["x"]], d[["z1", "z2"]], show=False)
ent = keys_check("endogeneity_test", blk, et)
check("endogeneity_test: first_stage_F is a dict by endogenous column", list(et["first_stage_F"]) == ["x"])
blk = head_check("overid_test", dg.overid_test)
ov = dg.overid_test(r["iv"].resids, sm.add_constant(d[["w"]]), d[["z1", "z2"]], 1, show=False)
keys_check("overid_test", blk, ov)

# ============================================================ 3. reporting
blk = head_check("study_card", rp.study_card)
sc = rp.study_card(question="q", estimand="e", design="d", show=False)
ent = keys_check("study_card", blk, sc)
sub_check("study_card", ent, "used", sc["used"].keys())
check("study_card: 'action' is one of the documented values",
      sc["action"] in ("written", "unchanged", "amended") and all(f'"{a_}"' in ent["action"] for a_ in
                                                                  ("written", "unchanged", "amended")))
sc2 = rp.study_card(question="q", estimand="e", design="d2", amend=True, reason="r", show=False)
keys_check("study_card", blk, sc2, " (amended)")
check("study_card (amended): action 'amended' and one amendment", sc2["action"] == "amended" and len(sc2["amendments"]) == 1)

blk = head_check("run_manifest", rp.run_manifest)
mf = rp.run_manifest(d, files=["d.csv", "nao_existe.csv"], seed=1, results={"b": np.float64(1.5)},
                     study_card="study_card.json", show=False)
ent = keys_check("run_manifest", blk, mf)
sub_check("run_manifest", ent, "study_card", mf["study_card"].keys())
sub_check("run_manifest", ent, "data", mf["data"]["data"].keys())
check("run_manifest: a missing file is recorded as \"missing\", a DataFrame alone is named \"data\"",
      mf["files"]["nao_existe.csv"] == "missing" and list(mf["data"]) == ["data"])
check("run_manifest: warnings are dicts with the documented keys",
      mf["warnings"] and all(list(w_) == ["level", "message", "action"] for w_ in mf["warnings"])
      and lists(ent["warnings"], "keys")[0] == ["level", "message", "action"], mf["warnings"])
mf0 = rp.run_manifest(d, simulated=True, show=False)
keys_check("run_manifest", blk, mf0, " (no card)")
check("run_manifest: without study_card= the key is None, as documented", mf0["study_card"] is None
      and "else None" in ent["study_card"])

blk = head_check("cross_check", rp.cross_check)
for case, kw in ((" (two routes, se)", dict(estimates={"a": 1.0, "b": 1.1}, se=0.2)),
                 (" (two routes, no se)", dict(estimates={"a": 1.0, "b": 1.0 + 1e-9})),
                 (" (one route)", dict(estimates={"a": 1.0}, se=0.2))):
    cc = rp.cross_check(show=False, **kw)
    ent = keys_check("cross_check", blk, cc, case)
    sub_check("cross_check" + case, ent, "table", cc["table"].columns, "columns")
    check(f"cross_check{case}: kind is one of the documented values",
          cc["kind"] is None or f'"{cc["kind"]}"' in ent["kind"], cc["kind"])

# ============================================================ 4. inference and data_prep
blk = head_check("describe_dist", inf.describe_dist)
dd = inf.describe_dist(d["y"], show=False)
ent = keys_check("describe_dist", blk, dd)
sub_check("describe_dist", ent, "summary", dd["summary"].index, "index")
check("describe_dist: estimand 'mean' and the CI is the t interval of the mean", dd["estimand"] == "mean"
      and abs(dd["ci_high"] - dd["estimate"] - dd["se"] * 1.9679296690653618) < 1e-12, dd)  # t(0.975, 299), scipy

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
dm = d.copy()
dm.loc[dm.w > 1, "x"] = np.nan
mr = dp.missing_report(dm, show=False)
ent = keys_check("missing_report", blk, mr, " (no predictors)")
mr2 = dp.missing_report(dm, cols=["y", "x"], predictors=["w"], show=False)
keys_check("missing_report", blk, mr2, " (predictors)")
check("missing_report: 'tests' is absent without predictors= and present with it (the only optional key)",
      "tests" not in mr and "tests" in mr2 and {k_ for k_, t_ in ent.items() if "only with" in t_} == {"tests"})
sub_check("missing_report", ent, "tests", mr2["tests"].columns, "columns")
check("missing_report: patterns has a column per variable and \"rows\"",
      list(mr["patterns"].columns) == list(dm.columns) + ["rows"] or
      (set(mr["patterns"].columns) == set(dm.columns) | {"rows"} and mr["patterns"].columns[-1] == "rows"),
      list(mr["patterns"].columns))

# ============================================================ 5. backtest and forecasting
blk = head_check("walk_forward_splits", bt.walk_forward_splits)
g = bt.walk_forward_splits(100, 50, 10)
first = next(iter(g))
check("walk_forward_splits: a generator of (train_idx, test_idx) integer arrays, as the block says",
      inspect.isgenerator(g) and isinstance(first, tuple) and len(first) == 2
      and all(np.issubdtype(np.asarray(a_).dtype, np.integer) for a_ in first) and "generator" in " ".join(blk))

blk = head_check("label_overlap", bt.label_overlap)
lo = bt.label_overlap(21, 500, show=False)
ent = keys_check("label_overlap", blk, lo)
sub_check("label_overlap", ent, "used", lo["used"].keys())

idx = pd.RangeIndex(200)
act = pd.Series(rng.normal(size=200).cumsum(), idx)
f1 = act.shift(1).bfill()
f2 = act.rolling(5, min_periods=1).mean().shift(1).bfill()
blk = head_check("evaluate", fc.evaluate)
txt = " ".join(l_.strip() for l_ in blk)
cols_doc = lists(txt, "columns")[0]
for bench in ("naive", "ma"):
    ev = fc.evaluate(act, {"naive": f1, "ma": f2}, benchmark=bench, show=False)
    want = [c_.replace("<benchmark>", bench) for c_ in cols_doc]
    check(f"evaluate (benchmark {bench}): the documented columns, with DM_vs_<benchmark> filled in",
          list(ev.columns) == want, (list(ev.columns), want))
    other = "ma" if bench == "naive" else "naive"
    check(f"evaluate (benchmark {bench}): the benchmark's row has NaN in the DM columns; the example runs",
          ev.loc[bench, [f"DM_vs_{bench}", "DM_p"]].isna().all() and np.isfinite(ev.loc[other, "DM_p"]))
evs = fc.evaluate(act, {"naive": f1, "ma": f2}, direction="sign", last_actual=act.shift(1).bfill(), show=False)
want_s = [c_.replace("<benchmark>", "naive") for c_ in cols_doc if c_ != "MAPE_%"]
check("evaluate (direction='sign'): no MAPE_%, plus directional_accuracy and PT_p_value, as the block says",
      set(evs.columns) == set(want_s) | {"directional_accuracy", "PT_p_value"}, list(evs.columns))

blk = head_check("conformal_intervals", fc.conformal_intervals)
for meth in ("aci", "rolling", "split"):
    ci = fc.conformal_intervals(act, f1, method=meth, show=False)
    ent = keys_check("conformal_intervals", blk, ci, f" ({meth})")
    sub_check(f"conformal_intervals ({meth})", ent, "table", ci["table"].columns, "columns")
    sub_check(f"conformal_intervals ({meth})", ent, "used", ci["used"].keys())
    check(f"conformal_intervals ({meth}): the table is indexed like actual", ci["table"].index.equals(act.index))
    if meth == "aci":
        sub_check("conformal_intervals (aci)", ent, "guarantee", ci["guarantee"].keys())

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
