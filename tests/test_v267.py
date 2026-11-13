"""Tests for v2.6.7: version, CHANGELOG and README; tools/routing_evals.py keeps the answers.

The output tests of the 20 functions are in tests/test_show_text.py and the sync of VALIDATION.md in
tests/test_validation_md.py (one file per subject, T4). This file keeps what belongs to the version:
  1. the version in the three places, the CHANGELOG entry at the top and the numbers it states, computed here from
     VALIDATION.md and tools/validation_diffs.json (not typed);
  2. README points to VALIDATION.md;
  3. routing_evals.py: the last message of a run is the `result` of the last trace line that has one; it is saved,
     with the graders' verdicts, only from an eval run directory under the temp folder; a trace with no result or a
     directory elsewhere is recorded as unreadable, with no file; main() runs every arm with --keep-temp unless
     --no-responses, writes responses.json, and removes the kept directories.

Run from the repository root:  python3 tests/test_v267.py
"""
import atexit
import importlib.util
import json
import re
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "econometrics" / "scripts"))
import _console  # noqa: E402,F401
import reporting  # noqa: E402

ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  [{info}]"))


# ---------------------------------------------------------------- 1. version and CHANGELOG
cl = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
mk = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))
top = re.search(r"^## v(\S+) — (.*)$", cl, re.M)
check("version 2.6.7 in CHANGELOG (top), reporting.SKILL_VERSION and marketplace.json",
      top.group(1) == "2.6.7" == reporting.SKILL_VERSION == mk["plugins"][0]["version"],
      (top.group(1), reporting.SKILL_VERSION, mk["plugins"][0]["version"]))
entry = cl[top.start():cl.index("\n## v", top.end())]
check("the v2.6.7 entry names the three subjects (show tests, validation diary, Cox SEs) in its title",
      "`show`" in top.group(2) and "validation diary" in top.group(2) and "Cox" in top.group(2), top.group(2))
val = (ROOT / "VALIDATION.md").read_text(encoding="utf-8")
lv = Counter(m.group(1) for m in re.finditer(r"^\| `[a-z_]+\.[A-Za-z0-9_]+` \| ([ABCD]) \|", val, re.M))
diffs = json.loads((ROOT / "tools" / "validation_diffs.json").read_text(encoding="utf-8"))["functions"]
check("CHANGELOG: the level counts are those of VALIDATION.md",
      f"({sum(lv.values())})" in entry and f"A replicated against a fixed reference: {lv['A']}" in entry
      and f"B known truth: {lv['B']}" in entry and f"C execution only: {lv['C']}" in entry
      and f"D no test: {lv['D']}" in entry, dict(lv))
check("CHANGELOG: the number of measured functions is that of tools/validation_diffs.json",
      f"For the {len(diffs)} level-A functions whose reference is outside the Python scientific stack" in entry,
      len(diffs))
worst = max(diffs.items(), key=lambda kv: kv[1]["worst_diff"] / kv[1]["tolerance"] if kv[1]["tolerance"] else 0)
check("CHANGELOG: the closest comparison named is the one with the largest difference/tolerance ratio",
      worst[0] == "causal.did_imputation"
      and re.sub(r"e-0*", "e-", f"{worst[1]['worst_diff']:.2g} of {worst[1]['tolerance']:g}") in entry,
      (worst[0], worst[1]["worst_diff"], worst[1]["tolerance"]))
check("CHANGELOG: the eight sentences fixed are named, each by its function",
      all(f in entry for f in ("dag.test_implications", "dependence.fit_copulas", "causal.event_study",
                               "diagnostics.diebold_mariano", "forecasting.evaluate", "diagnostics.endogeneity_test",
                               "diagnostics.chow", "data_prep.missing_report")))
lref = json.loads((ROOT / "tests" / "data" / "cox_robust_lifelines_v267.json").read_text(encoding="utf-8"))
check("CHANGELOG: the lifelines version cited is the one of the reference file",
      f"lifelines {lref['lifelines']}" in entry, lref["lifelines"])

# ---------------------------------------------------------------- 2. README
readme = (ROOT / "README.md").read_text(encoding="utf-8")
check("README points to VALIDATION.md and to its sync test",
      "[VALIDATION.md](VALIDATION.md)" in readme and "tests/test_validation_md.py" in readme)
n_funcs = sum(lv.values())
check("README: the number of functions in the diary is that of VALIDATION.md", f"one row per public function ({n_funcs})"
      in readme, n_funcs)

# ---------------------------------------------------------------- 3. routing_evals keeps the answers
spec = importlib.util.spec_from_file_location("routing_evals", ROOT / "tools" / "routing_evals.py")
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)


CREATED = []
atexit.register(lambda: [shutil.rmtree(p_, ignore_errors=True) for p_ in CREATED])   # also when a check raises


def kept_run(answer_lines):
    """A kept run directory as the eval tool leaves it: claude-eval-*/out/, with a trace two levels below."""
    d = Path(tempfile.mkdtemp(prefix="claude-eval-", dir=tempfile.gettempdir()))
    CREATED.append(d)
    (d / "out").mkdir()
    tr = d / "out" / "trace.jsonl"
    tr.write_text("\n".join(json.dumps(x) for x in answer_lines) + "\n", encoding="utf-8")
    return d, tr


good, tr_g = kept_run([{"type": "assistant", "message": "thinking"}, {"type": "result", "result": "first"},
                       {"type": "tool", "x": 1}, {"type": "result", "result": "Use a dummy."}, {"other": 2}])
none_, tr_n = kept_run([{"type": "assistant", "message": "no result"}])
outside = Path(tempfile.mkdtemp(prefix="not-eval-"))
CREATED.append(outside)
(outside / "out").mkdir()
tr_o = outside / "out" / "trace.jsonl"
tr_o.write_text(json.dumps({"result": "should not be read"}) + "\n", encoding="utf-8")
check("last message: the result of the LAST trace line that has one (not the first)", rt._last_message(tr_g)
      == "Use a dummy.", rt._last_message(tr_g))
res = {"cases": [{"name": "clean-two/groups", "arms": {"with": [
    {"tracePath": str(tr_g), "graders": [{"name": "skill-fired", "passed": True}, {"name": "answer", "passed": False}]},
    {"tracePath": str(tr_n), "graders": []},
    {"tracePath": str(tr_o), "graders": []},
    {"graders": []}]}}]}
with tempfile.TemporaryDirectory() as tmp:
    recs = rt.save_responses(res, Path(tmp) / "responses")
    files = sorted(p.name for p in (Path(tmp) / "responses").iterdir())
    body = (Path(tmp) / "responses" / files[0]).read_text(encoding="utf-8") if files else ""
check("save_responses: one record per run, only the readable one has a file (name made safe)",
      [r["readable"] for r in recs] == [True, False, False, False] and files == ["clean-two_groups__run0.md"]
      and recs[0]["file"] == files[0] and all(r["file"] is None for r in recs[1:]), (recs, files))
check("save_responses: the file has the case, the verdicts (pass/FAIL) and the answer",
      "# clean-two/groups, run 0" in body and "- skill-fired: pass" in body and "- answer: FAIL" in body
      and body.rstrip().endswith("Use a dummy."), body)
check("save_responses: a trace outside an eval run directory under the temp folder is never read",
      recs[2]["readable"] is False and "should not be read" not in body)

calls = []


def fake_run(cmd, *a, **k):
    calls.append(cmd)
    d, tr = kept_run([{"type": "result", "result": f"answer {len(calls)}"}])
    out = Path(cmd[cmd.index("--json") + 1])
    out.write_text(json.dumps({"cases": [{"name": "clean-a", "arms": {"with": [
        {"tracePath": str(tr), "graders": [{"name": "skill-fired", "passed": True}]}]}}], "costUsd": 0}),
        encoding="utf-8")
    calls.append(d)
    return type("P", (), {"returncode": 0})()


orig = (rt.shutil.which, rt.subprocess.run)
rt.shutil.which, rt.subprocess.run = (lambda name: "/usr/bin/claude"), fake_run
try:
    with tempfile.TemporaryDirectory() as tmp:
        rt.main(["--arm", "both", "--runs", "1", "--case", "two-groups", "--out", str(Path(tmp) / "d")])
        rt.main(["--arm", "both", "--runs", "1", "--case", "two-groups", "--no-responses",
                 "--out", str(Path(tmp) / "n")])
        pi, pf = Path(tmp) / "d" / "responses.json", Path(tmp) / "d" / "responses" / "clean-a__run0.md"
        d_idx = json.loads(pi.read_text(encoding="utf-8")) if pi.exists() else None
        d_file = pf.read_text(encoding="utf-8") if pf.exists() else ""
        n_has = (Path(tmp) / "n" / "responses.json").exists() or (Path(tmp) / "n" / "responses").exists()
        diag = (Path(tmp) / "d" / "nameonly-diagnostics.json").exists()
finally:
    rt.shutil.which, rt.subprocess.run = orig
cmds, dirs = calls[0::2], calls[1::2]
check("main(): every arm runs with --keep-temp by default, and without it with --no-responses (clean and crowded)",
      ["--keep-temp" in c for c in cmds] == [True, False], [("--keep-temp" in c) for c in cmds])
check("main(): the answer is saved and indexed (responses.json) by default, nothing with --no-responses",
      d_idx == [{"case": "clean-a", "run": 0, "readable": True, "graders": {"skill-fired": True},
                 "file": "clean-a__run0.md"}] and d_file.rstrip().endswith("answer 1") and not n_has, (d_idx, n_has))
check("main(): the kept run directory is removed after the answer is saved", not dirs[0].exists(), dirs[0])
check("main(): no name-only diagnostics file without the name-only arm", not diag)

# two case names that clean to the same file name are both kept
_, tr_a = kept_run([{"result": "A"}])
_, tr_b = kept_run([{"result": "B"}])
with tempfile.TemporaryDirectory() as tmp:
    recs2 = rt.save_responses({"cases": [{"name": "a/b", "arms": {"with": [{"tracePath": str(tr_a)}]}},
                                         {"name": "a_b", "arms": {"with": [{"tracePath": str(tr_b)}]}}]},
                              Path(tmp) / "r")
    texts = {p_.name: p_.read_text(encoding="utf-8").rstrip()[-1] for p_ in (Path(tmp) / "r").iterdir()}
check("save_responses: two case names that clean to the same file name do not overwrite each other",
      texts == {"a_b__run0.md": "A", "a_b__run0_2.md": "B"} and [r_["file"] for r_ in recs2]
      == ["a_b__run0.md", "a_b__run0_2.md"], texts)

# an error while saving the answers does not cost the run its summary
calls.clear()


def boom(*a, **k):
    raise OSError("disk full")


orig_save = rt.save_responses
rt.shutil.which, rt.subprocess.run, rt.save_responses = (lambda name: "/usr/bin/claude"), fake_run, boom
try:
    with tempfile.TemporaryDirectory() as tmp:
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rt.main(["--arm", "both", "--runs", "1", "--case", "two-groups", "--out", str(Path(tmp) / "x")])
        summ = (Path(tmp) / "x" / "summary.md").exists()
        px = Path(tmp) / "x" / "responses.json"
        idx = json.loads(px.read_text(encoding="utf-8")) if px.exists() else None
finally:
    rt.shutil.which, rt.subprocess.run = orig
    rt.save_responses = orig_save
check("main(): an error in save_responses is reported, the summary is still written and the kept directory removed",
      "WARNING: answers not saved (OSError: disk full)" in buf.getvalue() and summ and idx == []
      and not calls[1].exists(), buf.getvalue()[-300:])

passed = sum(ok)
print(f"\n{passed}/{len(ok)} checks passed")
sys.exit(0 if passed == len(ok) else 1)
