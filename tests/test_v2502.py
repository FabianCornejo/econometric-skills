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
          d1 == {"listing-captured": True, "description-dropped": True, "skills": 855, "with_description": 1}, str(d1))
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
    check("transcript: econometrics by name only is found next to a skill whose name only starts like it",
          d4b["description-dropped"], str(d4b))
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
recs = rt.annotate_nameonly(fake_res)
g = {x["name"]: x["passed"] for x in fake_res["cases"][0]["arms"]["with"][0]["graders"]}
check("annotate: the name-only run gets listing-captured and description-dropped from its transcript",
      g == {"skill-fired": False, "listing-captured": True, "description-dropped": True}, str(g))
check("annotate: other arms are untouched and one record per name-only run is returned",
      len(fake_res["cases"][1]["arms"]["with"][0]["graders"]) == 1 and len(recs) == 1 and recs[0]["skills"] == 855)
check("annotate: the kept run directory is removed afterwards", not base.exists())
miss = {"cases": [{"name": "nameonly-a", "arms": {"with": [{"tracePath": None, "graders": []}]}}]}
rt.annotate_nameonly(miss)
check("annotate: a run without a kept directory counts as not captured (never as name only)",
      {x["name"]: x["passed"] for x in miss["cases"][0]["arms"]["with"][0]["graders"]}
      == {"listing-captured": False, "description-dropped": False})
src = (ROOT / "tools" / "routing_evals.py").read_text(encoding="utf-8")
check("the name-only arm runs with --keep-temp and writes nameonly-diagnostics.json",
      '"--keep-temp"' in src and "nameonly-diagnostics.json" in src)

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
t3, _ = rt.summarize({"cases": [{"name": "clean-a", "arms": {"with": [run(skill_fired=True)]}}]})
check("summary without a name-only arm is unchanged (no name-only section)", "Name-only" not in t3)

# ---------------------------------------------------------------- 5. the real request as a routing case
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

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
