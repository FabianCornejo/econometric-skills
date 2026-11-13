"""Tests for v2.5.0.2: a routing measurement that reproduces the crowded-listing failure.

On the user's notebook (2026-11-07) the skill listing held 865 skills, only 14 with a description, and econometrics
appeared by name only; a difference-in-differences request with one treated store was answered without the skill.
Claude Code drops descriptions starting with the least-invoked skills, so a new or rarely used skill is the first to
lose its description. The existing "crowded" arm (48 decoys) never checked whether the description was dropped, so
it could not reproduce that failure nor validate a remedy.

`tools/routing_evals.py` gains a "nameonly" arm (about 850 decoy skills over 12 decoy plugins), two diagnostics per
run read from the child session's transcript (the listing was captured; econometrics is listed by name only) and a
summary that counts routing only over the runs where both hold. The request of the notebook test is a new routing
case. The first run on the notebook showed that the listing is not in the trace the eval hands to graders but is in
the kept run's transcript (855 skills, 11 descriptions, econometrics by name only), hence the transcript reader.

Run from the repository root:  python3 tests/test_v2502.py
"""
import importlib.util
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "econometrics"
sys.path.insert(0, str(SKILL / "scripts"))
import _console  # noqa: E402,F401

ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


def front(path):
    m = re.match(r"---\r?\n(.*?)\r?\n---\r?\n?(.*)", path.read_text(encoding="utf-8"), re.S)
    return yaml.safe_load(m.group(1)), m.group(2)


spec = importlib.util.spec_from_file_location("routing_evals", ROOT / "tools" / "routing_evals.py")
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)

# ---------------------------------------------------------------- 1. the decoy listing
decoys = rt.nameonly_decoys()
names = [n for skills in decoys.values() for n, _ in skills]
check("name-only arm: 12 decoy plugins and 800 to 900 decoy skills (the notebook had 865 skills)",
      len(decoys) == 12 and 800 <= len(names) <= 900, f"{len(decoys)} plugins, {len(names)} skills")
check("decoy skill names are unique inside each plugin and valid (lowercase, digits, hyphens, at most 64 chars)",
      all(len({n for n, _ in s}) == len(s) for s in decoys.values())
      and all(re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", n) and len(n) <= 64 for n in names))
check("the decoys are the same on every call (no randomness in the harness)", rt.nameonly_decoys() == decoys)
check("some decoy names sound like statistics work, as in the real listing",
      {"statistical-analysis", "explore-data", "variance-analysis", "ab-test", "forecast"} <= set(names))
check("no decoy plugin is called econometrics or crowd", not {"econometrics", "crowd"} & set(decoys))

# ---------------------------------------------------------------- 2. the harness
cases = sorted(p.name for p in (SKILL / "evals" / "routing").iterdir() if (p / "prompt.md").exists())
with tempfile.TemporaryDirectory() as tmp:
    h = Path(tmp) / "h"
    info = rt.build(h, ["two-stores-promo", "excel-formula"], rt.ARM_CHOICES["nameonly"])
    built = sorted(p.name for p in (h / "evals").iterdir())
    check("--arm nameonly builds only nameonly-<case> cases",
          built == ["nameonly-excel-formula", "nameonly-two-stores-promo"], str(built))
    d = h / "evals" / "nameonly-two-stores-promo"
    pm, body = front(d / "prompt.md")
    check("a name-only case loads econometrics plus the 12 decoy plugins, tagged nameonly",
          pm["plugins"] == ["econometrics", *decoys] and "nameonly" in pm["tags"] and pm["name"] == d.name)
    n_skill = len(list(d.glob("*/skills/*/SKILL.md")))
    check("every decoy is written as a skill of its decoy plugin", n_skill == len(names) == info["nameonly_skills"],
          f"{n_skill} on disk")
    check("decoy descriptions far exceed the listing seen on the notebook (40,644 characters with 14 descriptions)",
          info["nameonly_chars"] > 5 * 40644, f"{info['nameonly_chars']} chars")
    heads = [front(p)[0] for p in list(d.glob("*/skills/*/SKILL.md"))[:50]]
    check("decoy headers parse and name their folder", all(isinstance(m, dict) and m.get("description") for m in heads))
    graders = sorted(p.stem for p in (d / "graders").glob("*.md"))
    check("a name-only case carries the case's own graders (the diagnostics come from the transcript, not graders)",
          graders == ["answer", "skill-fired"], str(graders))
    # the default build stays as it was: clean and crowded only
    h2 = Path(tmp) / "h2"
    rt.build(h2, ["two-groups"])
    check("the default build is unchanged (clean and crowded; the name-only arm only when asked for)",
          sorted(p.name for p in (h2 / "evals").iterdir()) == ["clean-two-groups", "crowded-two-groups"])
    h3 = Path(tmp) / "h3"
    info3 = rt.build(h3, ["two-groups"], rt.ARM_CHOICES["all"])
    check("--arm all builds the three arms",
          sorted(p.name for p in (h3 / "evals").iterdir()) == ["clean-two-groups", "crowded-two-groups",
                                                               "nameonly-two-groups"] and info3["nameonly_skills"])
try:
    rt.build(Path(tempfile.mkdtemp()) / "x", ["two-groups"], ("clean", "lotado"))
    raised = False
except ValueError:
    raised = True
check("an unknown arm is refused instead of building a case without decoys", raised)

# ---------------------------------------------------------------- 3. the listing read from a kept run's transcript
# Format read from the notebook transcripts of 2026-11-07: attachment type "skill_listing", entries "- <name>" without a
# description and "- <name>: <description>" with it.


def listing_line(content, count=None):
    att = {"type": "skill_listing", "content": content, "skillCount": count or content.count("\n- ") + 1}
    return json.dumps({"type": "attachment", "attachment": att})


def sandbox(root, name, lines):
    sb = root / name
    (sb / "out").mkdir(parents=True)
    (sb / "out" / "trace.jsonl").write_text("{}\n", encoding="utf-8")
    proj = sb / "config" / "projects" / "C--tmp-cwd"
    proj.mkdir(parents=True)
    (proj / "s.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sb


name_only = listing_line("- dataviz: Use this skill whenever...\n- data-suite:sql\n- econometrics:econometrics\n"
                         "- econometrics:statistics\n- legal-ops:contract", 855)
with_desc = listing_line("- data-suite:sql\n- econometrics:econometrics: Statistics, econometrics and quantitative risk\n"
                         "- econometrics:statistics: Statistics and hypothesis tests")
other = listing_line("- data-suite:sql\n- econometrics:econometrics-extra")
tool_call = json.dumps({"type": "assistant", "message": {"content": [
    {"type": "tool_use", "name": "Skill", "input": {"skill": "econometrics:econometrics"}}]}})
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    d1 = rt.listing_diagnostics(sandbox(root, "claude-eval-a", ["not json", tool_call, name_only]))
    check("transcript: name-only listing found; econometrics by name only; counts of skills and descriptions",
          d1 == {"listing-captured": True, "description-dropped": True, "skills": 855, "with_description": 1,
                 "econometrics_described": []}, str(d1))
    stat_kept = listing_line("- data-suite:sql\n- econometrics:econometrics\n"
                             "- econometrics:statistics: Statistics and hypothesis tests in Python")
    d1b = rt.listing_diagnostics(sandbox(root, "claude-eval-a2", [stat_kept]))
    check("transcript: econometrics:statistics kept its description -> not the failure under test (not counted)",
          d1b["listing-captured"] and not d1b["description-dropped"]
          and d1b["econometrics_described"] == ["econometrics:statistics"], str(d1b))
    sb_bom = sandbox(root, "claude-eval-a3", [])
    (sb_bom / "config" / "projects" / "C--tmp-cwd" / "s.jsonl").write_bytes(b"\xef\xbb\xbf" + (name_only + "\r\n").encode())
    check("transcript: a UTF-8 BOM and CRLF line ends do not hide the listing",
          rt.listing_diagnostics(sb_bom)["description-dropped"])
    sb_sub = sandbox(root, "claude-eval-a4", [name_only])
    (sb_sub / "config" / "projects" / "C--tmp-cwd" / "s" / "subagents").mkdir(parents=True)
    (sb_sub / "config" / "projects" / "C--tmp-cwd" / "s" / "subagents" / "agent-1.jsonl").write_text(with_desc + "\n")
    check("transcript: the main session's listing is read before a subagent's (shallower file first)",
          rt.listing_diagnostics(sb_sub)["description-dropped"])
    d2 = rt.listing_diagnostics(sandbox(root, "claude-eval-b", [with_desc]))
    check("transcript: econometrics kept its description -> description-dropped False",
          d2["listing-captured"] and not d2["description-dropped"] and d2["with_description"] == 2, str(d2))
    d3 = rt.listing_diagnostics(sandbox(root, "claude-eval-c", [tool_call]))
    check("transcript: no listing -> nothing captured, nothing dropped (a Skill call is not a listing)",
          not d3["listing-captured"] and not d3["description-dropped"], str(d3))
    d4 = rt.listing_diagnostics(sandbox(root, "claude-eval-d", [other]))
    check("transcript: a different skill whose name starts with econometrics:econometrics does not count",
          d4["listing-captured"] and not d4["description-dropped"], str(d4))
    both = listing_line("- econometrics:econometrics\n- econometrics:econometrics-extra: another skill")
    d4b = rt.listing_diagnostics(sandbox(root, "claude-eval-d2", [both]))
    check("transcript: a described entry of the plugin whose name only starts like the skill is not mistaken for it, "
          "and still excludes the run (any econometrics:* description does)",
          not d4b["description-dropped"] and d4b["econometrics_described"] == ["econometrics:econometrics-extra"],
          str(d4b))
    d5 = rt.listing_diagnostics(sandbox(root, "claude-eval-e", [name_only, with_desc]))
    check("transcript: the first listing is the one used (the request was answered with it)",
          d5["description-dropped"], str(d5))
    # the agent file names the skill with two spaces of indentation: never read as a listing entry
    agent = json.dumps({"type": "attachment", "attachment": {"type": "file", "content": "skills:\n  - econometrics:econometrics"}})
    d6 = rt.listing_diagnostics(sandbox(root, "claude-eval-f", [agent]))
    check("transcript: an attachment of another type (the agent file) is not a listing", not d6["listing-captured"])

# a kept run lives under the system temp folder in a claude-eval-* directory with out/; nothing else is touched
tmpd = Path(tempfile.mkdtemp(prefix="claude-eval-"))
(tmpd / "out").mkdir()
outside = Path(tempfile.mkdtemp(prefix="v2502-"))       # under temp, with out/, but not an eval directory
(outside / "out").mkdir()
check("only eval run directories under the temp folder are read and removed",
      rt._is_eval_sandbox(tmpd) and not rt._is_eval_sandbox(outside) and not rt._is_eval_sandbox(ROOT)
      and not rt._is_eval_sandbox(Path("/claude-eval-x")))
shutil.rmtree(tmpd)
shutil.rmtree(outside)

base = Path(tempfile.mkdtemp(prefix="claude-eval-"))     # a fake kept run, as the eval tool leaves it
(base / "out").mkdir()
(base / "out" / "trace.jsonl").write_text("{}\n", encoding="utf-8")
(base / "config" / "projects" / "p").mkdir(parents=True)
(base / "config" / "projects" / "p" / "s.jsonl").write_text(name_only + "\n", encoding="utf-8")
fake_res = {"cases": [
    {"name": "nameonly-two-groups", "arms": {"with": [{"tracePath": str(base / "out" / "trace.jsonl"),
                                                       "graders": [{"name": "skill-fired", "passed": False}]}]}},
    {"name": "clean-two-groups", "arms": {"with": [{"tracePath": str(base / "out" / "trace.jsonl"),
                                                    "graders": [{"name": "skill-fired", "passed": True}]}]}}]}
base2 = Path(tempfile.mkdtemp(prefix="claude-eval-"))    # a kept clean-arm run (--arm all keeps every run)
(base2 / "out").mkdir()
fake_res["cases"][1]["arms"]["with"][0]["tracePath"] = str(base2 / "out" / "trace.jsonl")
recs = rt.annotate_nameonly(fake_res)
g = {x["name"]: x["passed"] for x in fake_res["cases"][0]["arms"]["with"][0]["graders"]}
check("annotate: the name-only run gets listing-captured and description-dropped from its transcript",
      g == {"skill-fired": False, "listing-captured": True, "description-dropped": True}, str(g))
check("annotate: other arms are untouched and one record per name-only run is returned",
      len(fake_res["cases"][1]["arms"]["with"][0]["graders"]) == 1 and len(recs) == 1 and recs[0]["skills"] == 855)
left = rt.remove_kept_dirs(fake_res)
check("remove_kept_dirs: the kept directories of every arm are removed, nothing left to report",
      not base.exists() and not base2.exists() and left == [])
keep_out = Path(tempfile.mkdtemp(prefix="v2502-keep-"))   # a path that is not an eval run directory
(keep_out / "out").mkdir()
odd = {"cases": [{"name": "nameonly-two-groups", "arms": {"with": [
    {"tracePath": str(keep_out / "out" / "trace.jsonl"), "graders": [{"name": "skill-fired", "passed": True}]}]}}]}
rt.annotate_nameonly(odd)
left2 = rt.remove_kept_dirs(odd)
t_odd, _ = rt.summarize(odd)
check("an unreadable kept directory is marked, left in place, reported, and the summary says the harness could not "
      "read it instead of 'did not reproduce' alone",
      odd["cases"][0]["arms"]["with"][0]["harnessUnreadable"] and keep_out.exists() and left2 == [str(keep_out)]
      and "could not read the kept directory of 1 run" in t_odd, t_odd)
shutil.rmtree(keep_out)
miss = {"cases": [{"name": "nameonly-a", "arms": {"with": [{"tracePath": None, "graders": []}]}}]}
rt.annotate_nameonly(miss)
check("annotate: a run without a kept directory counts as not captured (never as name only)",
      {x["name"]: x["passed"] for x in miss["cases"][0]["arms"]["with"][0]["graders"]}
      == {"listing-captured": False, "description-dropped": False})
# review round 2 (SF-2): each half of the deletion guard on its own
fake_temp = Path(tempfile.mkdtemp(prefix="v2502-faketemp-"))    # stands in for the temp folder
elsewhere = Path(tempfile.mkdtemp(prefix="v2502-elsewhere-"))    # a sibling, outside that "temp folder"
far = elsewhere / "claude-eval-far"
(far / "out").mkdir(parents=True)
saved_tempdir = tempfile.tempdir
try:
    tempfile.tempdir = str(fake_temp)
    far_res = {"cases": [{"name": "nameonly-a", "arms": {"with": [{"tracePath": str(far / "out" / "trace.jsonl"),
                                                                   "graders": []}]}}]}
    left_far = rt.remove_kept_dirs(far_res)
    ok_far = not rt._is_eval_sandbox(far) and far.exists() and left_far == [str(far)]
finally:
    tempfile.tempdir = saved_tempdir
    shutil.rmtree(fake_temp, ignore_errors=True)
    shutil.rmtree(elsewhere, ignore_errors=True)
check("deletion guard: a claude-eval-* directory with out/ OUTSIDE the temp folder is refused, kept and reported",
      ok_far, str(left_far))
no_out = Path(tempfile.mkdtemp(prefix="claude-eval-"))     # under temp, right prefix, but no out/
(no_out / "x").mkdir()
nores = {"cases": [{"name": "nameonly-a", "arms": {"with": [{"tracePath": str(no_out / "out" / "trace.jsonl"),
                                                             "graders": []}]}}]}
left_no = rt.remove_kept_dirs(nores)
check("deletion guard: a claude-eval-* directory under temp WITHOUT out/ is refused, kept and reported",
      not rt._is_eval_sandbox(no_out) and no_out.exists() and left_no == [str(no_out)], str(left_no))
shutil.rmtree(no_out)
untr = {"cases": [{"name": "nameonly-a", "arms": {"with": [{"error": "exit 1: limit", "graders": []}]}}]}
left_u = rt.remove_kept_dirs(untr)
check("remove_kept_dirs: a run without a trace path is not silent (one line pointing at leftover claude-eval-* "
      "directories)", len(left_u) == 1 and "claude-eval-*" in left_u[0] and "1 run(s)" in left_u[0], str(left_u))
# N-7: a transcript that cannot be read marks that run only, and the annotation goes on
bad = Path(tempfile.mkdtemp(prefix="claude-eval-")); (bad / "out").mkdir()
good = Path(tempfile.mkdtemp(prefix="claude-eval-")); (good / "out").mkdir()
(good / "config" / "projects" / "p").mkdir(parents=True)
(good / "config" / "projects" / "p" / "s.jsonl").write_text(name_only + "\n", encoding="utf-8")
real_ld = rt.listing_diagnostics
def flaky(kept):
    if kept.resolve() == bad.resolve():
        raise PermissionError("locked")
    return real_ld(kept)
rt.listing_diagnostics = flaky
two = {"cases": [{"name": "nameonly-a", "arms": {"with": [
    {"tracePath": str(bad / "out" / "trace.jsonl"), "graders": []},
    {"tracePath": str(good / "out" / "trace.jsonl"), "graders": []}]}}]}
try:
    recs2 = rt.annotate_nameonly(two)
    ok_flaky = (two["cases"][0]["arms"]["with"][0]["harnessUnreadable"]
                and not two["cases"][0]["arms"]["with"][1]["harnessUnreadable"]
                and [r["description-dropped"] for r in recs2] == [False, True])
except OSError:
    ok_flaky = False
finally:
    rt.listing_diagnostics = real_ld
check("annotate: a transcript that cannot be read (locked on Windows) marks that run unreadable and the next run "
      "is still read", ok_flaky)
rt.remove_kept_dirs(two)
check("the kept directories of that test are removed", not bad.exists() and not good.exists())
src = (ROOT / "tools" / "routing_evals.py").read_text(encoding="utf-8")
# main(): the command it builds, with the CLI and the eval tool replaced by stubs
calls = []


def fake_run(cmd, *a, **k):
    calls.append(cmd)
    out = Path(cmd[cmd.index("--json") + 1])
    out.write_text(json.dumps({"cases": [], "costUsd": 0}), encoding="utf-8")
    return type("P", (), {"returncode": 0})()


orig = (rt.shutil.which, rt.subprocess.run)
rt.shutil.which, rt.subprocess.run = (lambda name: "/usr/bin/claude"), fake_run
try:
    with tempfile.TemporaryDirectory() as tmp:
        # --no-responses (v2.6.7): without it every arm keeps its run directories to save the answers
        for arm in ("both", "nameonly", "all"):
            rt.main(["--arm", arm, "--runs", "1", "--case", "two-groups", "--no-responses", "--out", str(Path(tmp) / arm)])
        rt.main(["--arm", "nameonly", "--no-hooks", "--runs", "1", "--case", "two-groups", "--no-responses",
                 "--out", str(Path(tmp) / "nh")])
        heads = {d: (Path(tmp) / d / "summary.md").read_text(encoding="utf-8").splitlines()[0] for d in ("both", "nh")}
        diag = (Path(tmp) / "nameonly" / "nameonly-diagnostics.json").exists()
finally:
    rt.shutil.which, rt.subprocess.run = orig
check("main() with --no-responses: --keep-temp only when the name-only arm runs (not for both), and the diagnostics "
      "file is written",
      ["--keep-temp" in c for c in calls] == [False, True, True, True] and diag)
check("main(): the summary header says whether the plugin hooks were on",
      "plugin hooks on" in heads["both"] and "plugin hooks off" in heads["nh"], str(heads))

# ---------------------------------------------------------------- 4. the summary


def run(**g):
    return {"graders": [{"name": k.replace("_", "-"), "passed": v} for k, v in g.items()]}


fake = {"cases": [
    {"name": "clean-a", "arms": {"with": [run(skill_fired=True, answer=True)]}},
    {"name": "nameonly-a", "arms": {"with": [
        run(listing_captured=True, description_dropped=True, skill_fired=False, answer=False),
        run(listing_captured=True, description_dropped=True, skill_fired=True, answer=True),
        run(listing_captured=True, description_dropped=False, skill_fired=True, answer=True),   # kept its description
        run(listing_captured=False, description_dropped=False, skill_fired=True, answer=True),  # listing not seen
    ]}},
    {"name": "nameonly-excel-formula", "arms": {"with": [
        run(listing_captured=True, description_dropped=True, skill_not_fired=True)]}},
]}
table, rows = rt.summarize(fake)
check("summary: name-only routing counts only the runs where the listing was captured and econometrics had no "
      "description (1 of 2 here, not 3 of 4)",
      "| a | 4 | 3 | 2 | 1/2 | 1/2 |" in table and "Routing rate, nameonly: 50% over 1 cases" in table, table)
check("summary: a negative case in the name-only arm reports false triggers",
      "False triggers, nameonly: 0% over 1 unrelated requests" in table)
check("summary: the name-only arm stays out of the clean/crowded table and rates",
      all(not k.startswith("nameonly") for r in rows for k in r) and "Routing rate, clean: 100% over 1 cases" in table)
none_valid = {"cases": [{"name": "nameonly-a", "arms": {"with": [
    run(listing_captured=True, description_dropped=False, skill_fired=True)]}}]}
t2, _ = rt.summarize(none_valid)
check("summary: when no run lost the description, it says the arm did not reproduce the failure and gives no rate",
      "did not reproduce the failure" in t2 and "Routing rate, nameonly" not in t2, t2)
errored = {"cases": [{"name": "nameonly-a", "arms": {"with": [
    {"error": "exit 1: You've hit your 5-hour limit", "graders": [{"name": "listing-captured", "passed": True},
                                                                  {"name": "description-dropped", "passed": True}]}]}}]}
t_err, _ = rt.summarize(errored)
check("summary: name-only runs that ended on an API error before grading are reported as such, not as "
      "'did not reproduce' (seen on the notebook: 18 runs hit the 5-hour usage limit)",
      "ended on an error before grading" in t_err and "this arm did not reproduce" not in t_err
      and "Routing rate, nameonly" not in t_err, t_err)
no_listing = {"cases": [{"name": "nameonly-a", "arms": {"with": [
    run(listing_captured=False, description_dropped=False, skill_fired=False),
    run(listing_captured=False, description_dropped=False, skill_fired=True)]}}]}
t_nl, _ = rt.summarize(no_listing)
check("summary (review SF-1): when no transcript showed the listing at all, it reports a harness or format problem, "
      "not 'did not reproduce', and gives no rate",
      "not found in any transcript" in t_nl and "this arm did not reproduce" not in t_nl
      and "Routing rate, nameonly" not in t_nl, t_nl)
err_nolist = {"cases": [{"name": "nameonly-a", "arms": {"with": [
    {"error": "exit 1: You've hit your 5-hour limit", "graders": [{"name": "listing-captured", "passed": False},
                                                                  {"name": "description-dropped", "passed": False}]}]}}]}
t_en, _ = rt.summarize(err_nolist)
check("summary (review SF-1): a run that errored before its listing was captured is reported as an error, "
      "not as 'did not reproduce' nor as a format problem",
      "1 name-only run(s) ended on an error before grading" in t_en and "this arm did not reproduce" not in t_en
      and "not found in any transcript" not in t_en, t_en)
mixed = {"cases": [{"name": "nameonly-a", "arms": {"with": [
    run(listing_captured=True, description_dropped=True, skill_fired=True),
    {"error": "exit 1: limit", "graders": [{"name": "listing-captured", "passed": False},
                                           {"name": "description-dropped", "passed": False}]}]}}]}
t_mx, _ = rt.summarize(mixed)
check("summary: one graded run and one errored run -> the rate uses the graded run only and the error is reported",
      "Routing rate, nameonly: 100% over 1 cases" in t_mx and "1 name-only run(s) ended on an error" in t_mx, t_mx)
errun = {"error": "exit 1: limit", "graders": [{"name": "listing-captured", "passed": False},
                                              {"name": "description-dropped", "passed": False}]}
case_a = {"cases": [{"name": "nameonly-a", "arms": {"with": [
    errun, run(listing_captured=False, description_dropped=False, skill_fired=False),
    run(listing_captured=False, description_dropped=False, skill_fired=True)]}}]}
t_a, _ = rt.summarize(case_a)
check("summary (review round 3, case A): one errored run does not hide that the graded runs never showed the "
      "listing; both are reported, and 'No graded run' is not said",
      "not found in any transcript" in t_a and "1 name-only run(s) ended on an error" in t_a
      and "No graded run" not in t_a and "Routing rate, nameonly" not in t_a, t_a)
case_b = {"cases": [{"name": "nameonly-a", "arms": {"with": [
    errun, run(listing_captured=True, description_dropped=False, skill_fired=True)]}}]}
t_b, _ = rt.summarize(case_b)
check("summary (review round 3, case B): one errored run plus a graded run that kept its description -> "
      "'did not reproduce' and the error line, not 'No graded run'",
      "this arm did not reproduce" in t_b and "1 name-only run(s) ended on an error" in t_b
      and "No graded run" not in t_b, t_b)
t3, _ = rt.summarize({"cases": [{"name": "clean-a", "arms": {"with": [run(skill_fired=True)]}}]})
check("summary without a name-only arm is unchanged (no name-only section)", "Name-only" not in t3)

# ---------------------------------------------------------------- 5. the SessionStart note (remedy under test)
hj = json.loads((SKILL / "hooks" / "hooks.json").read_text(encoding="utf-8"))
ss = hj["hooks"]["SessionStart"]
h0 = ss[0]["hooks"][0]
m = re.fullmatch(r"echo '([^']*)'", h0["command"])
note = m.group(1) if m else ""
check("hook: one SessionStart command hook, the only event the plugin hooks",
      list(hj["hooks"]) == ["SessionStart"] and len(ss) == 1 and len(ss[0]["hooks"]) == 1 and h0["type"] == "command")
check("hook: a plain echo of one single-quoted line, the same in bash and PowerShell (no $, backtick or quotes "
      "inside, no Python needed)", m is not None and not re.search(r"[$`\"'\n]", note))
check("hook: the note names the skill, says to load it with the name only, and to ignore it otherwise",
      "econometrics:econometrics" in note and "only its name" in note and "Ignore this note" in note)
check("hook: the note is limited to statistical work and names what stays out (budgets, valuations, legal risk)",
      note.startswith("Note from the econometrics plugin: statistical work")
      and all(w in note for w in ("budgets", "valuations", "legal risk")) and "judge numbers" not in note)
bash = shutil.which("bash")
if bash and sys.platform != "win32":
    import subprocess  # noqa: E402
    printed = subprocess.run([bash, "-c", h0["command"]], capture_output=True, text=True).stdout
    check("hook: run by bash, it prints exactly the note", printed == note + "\n", repr(printed[:60]))
check("hook: short (at most 400 characters, under 1% of the 40,000-character listing budget seen on the notebook)",
      0 < len(note) <= 400, f"{len(note)} chars")
check("hook: runs on startup, /clear and compaction; not on resume or fork (assumed to replay the saved note: "
      "NOT verified, checked in the real validation)",
      set(ss[0]["matcher"].split("|")) == {"startup", "clear", "compact"})
check("hook: a short timeout (the first answer waits for SessionStart hooks)", 0 < h0.get("timeout", 600) <= 10)
readme = (ROOT / "README.md").read_text(encoding="utf-8")
check("README: describes the session-start note with its real length and how to turn it off",
      f"one line of {len(note)} characters" in readme and "turn the plugin off" in readme
      and "least" in readme and "--arm nameonly" in readme)
with tempfile.TemporaryDirectory() as tmp:
    rt.build(Path(tmp) / "on", ["two-groups"], ("nameonly",))
    rt.build(Path(tmp) / "off", ["two-groups"], ("nameonly",), hooks=False)
    on = Path(tmp) / "on" / "evals" / "nameonly-two-groups" / "econometrics" / "hooks" / "hooks.json"
    off = Path(tmp) / "off" / "evals" / "nameonly-two-groups" / "econometrics" / "hooks"
    check("harness: the plugin copy carries the hook by default and leaves it out with hooks=False (--no-hooks)",
          on.exists() and not off.exists() and "--no-hooks" in src)
import zipfile  # noqa: E402
with zipfile.ZipFile(ROOT / "dist" / "econometrics.skill") as z:
    check("app package (dist) carries no hooks/ (the app does not run plugin hooks)",
          not any(n.startswith("econometrics/hooks/") for n in z.namelist()))
wf = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
check("CI: the dist check excludes hooks/ from the comparison and refuses it in the package",
      "-x hooks" in wf and "evals|hooks)/" in wf and '\\"econometrics/hooks/*\\"' in wf)

# ---------------------------------------------------------------- 6. the real request as a routing case
case = SKILL / "evals" / "routing" / "two-stores-promo"
pm, body = front(case / "prompt.md")
fired = front(case / "graders" / "skill-fired.md")[0]
ref = front(SKILL / "evals" / "routing" / "two-groups" / "graders" / "skill-fired.md")[0]
check("two-stores-promo: the notebook request (2 stores, one with a promotion, 24 months, Excel, simulated data)",
      all(w in body for w in ("2 lojas", "promoção", "24 meses", "Excel", "dados simulados")) and "two-stores-promo" in cases)
check("two-stores-promo: read-only tools, the same skill-fired grader as the other cases",
      pm["allowed_tools"] == ["Read", "Glob", "Grep", "Skill"] and fired == ref)
ans = front(case / "graders" / "answer.md")[0]
pat = re.compile(ans["pattern"])
check("two-stores-promo: the answer check finds a difference-in-differences framing, case-sensitive (no flag i, "
      "so the English word 'did' does not pass)", "flags" not in ans and
      all(pat.search(s) for s in ("Use diferenças em diferenças", "um DiD com a loja controle", "did_design(...)",
                                  "difference-in-differences"))
      and not any(pat.search(s) for s in ("é difícil dizer", "a diferença entre as lojas", "didático",
                                                       "Did the promotion work?", "did it rise?")))

case = SKILL / "evals" / "routing" / "wacc-beta-valuation"
pm, body = front(case / "prompt.md")
check("wacc-beta-valuation: a beta for the WACC inside a valuation, read-only tools, the shared skill-fired grader",
      all(w in body for w in ("valuation", "beta", "WACC")) and pm["allowed_tools"] == ["Read", "Glob", "Grep", "Skill"]
      and front(case / "graders" / "skill-fired.md")[0] == ref and "wacc-beta-valuation" in cases)
ans = front(case / "graders" / "answer.md")[0]
pat = re.compile(ans["pattern"], re.I if "i" in ans.get("flags", "") else 0)
check("wacc-beta-valuation: the answer check asks how reliable the beta is (standard error, interval, R², "
      "Newey-West, Blume or Vasicek), not just that a regression was run",
      all(pat.search(x) for x in ("o erro-padrão do beta é 0,12", "intervalo de confiança de 95%",
                                  "ajuste de Blume", "standard error of the slope", "Newey-West com 3 defasagens"))
      and not any(pat.search(x) for x in ("rode uma regressão dos retornos", "use OLS",
                                          "add the controls", "beta = cov/var")))

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
