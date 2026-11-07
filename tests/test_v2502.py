"""Tests for v2.5.0.2: a routing measurement that reproduces the crowded-listing failure.

On the user's notebook (2026-11-07) the skill listing held 865 skills, only 14 with a description, and econometrics
appeared by name only; a difference-in-differences request with one treated store was answered without the skill.
Claude Code drops descriptions starting with the least-invoked skills, so a new or rarely used skill is the first to
lose its description. The existing "crowded" arm (48 decoys) never checked whether the description was dropped, so
it could not reproduce that failure nor validate a remedy.

`tools/routing_evals.py` gains a "nameonly" arm (about 850 decoy skills over 12 decoy plugins), two diagnostic
graders read from the session trace (the listing was captured; econometrics is listed by name only) and a summary
that counts routing only over the runs where both hold. The request of the notebook test is a new routing case.

Run from the repository root:  python3 tests/test_v2502.py
"""
import importlib.util
import json
import re
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
    check("a name-only case carries the case's graders plus listing-captured and description-dropped",
          graders == ["answer", "description-dropped", "listing-captured", "skill-fired"], str(graders))
    cap = front(d / "graders" / "listing-captured.md")[0]
    drop = front(d / "graders" / "description-dropped.md")[0]
    check("the diagnostic graders are regex graders over the trace, scored in both arms",
          all(g["type"] == "regex" and g["target"] == "trace" and g["arm"] == "both" for g in (cap, drop)))
    check("the diagnostic graders have no body (a body is only for llm graders)",
          front(d / "graders" / "listing-captured.md")[1].strip() == ""
          and front(d / "graders" / "description-dropped.md")[1].strip() == "")
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

# ---------------------------------------------------------------- 3. the diagnostic patterns on trace lines
# The trace is JSON per line; a listing entry is "- <name>" (no description) or "- <name>: <description>"
# (format read from the notebook transcript of 2026-11-07, attachment type "skill_listing").


def line(content):
    return json.dumps({"type": "attachment", "attachment": {"type": "skill_listing", "content": content}})


name_only = line("- dataviz: Use this skill whenever...\n- data-suite:sql\n- econometrics:econometrics\n"
                 "- econometrics:statistics\n- legal-ops:contract")
with_desc = line("- data-suite:sql\n- econometrics:econometrics: Statistics, econometrics and quantitative risk\n"
                 "- econometrics:statistics: Statistics and hypothesis tests")
last_entry = line("- data-suite:sql\n- econometrics:econometrics")
tool_call = json.dumps({"type": "assistant", "message": {"content": [
    {"type": "tool_use", "name": "Skill", "input": {"skill": "econometrics:econometrics"}}]}})
decoy_call = json.dumps({"type": "assistant", "message": {"content": [
    {"type": "tool_use", "name": "Skill", "input": {"skill": "data-suite:statistical-analysis"}}]}})
agent_file = json.dumps({"content": "---\nskills:\n  - econometrics:econometrics\n---"})
pc, pd_ = cap["pattern"], drop["pattern"]
check("listing-captured: found in a listing with the decoy plugins, not in a Skill call (to econometrics or to "
      "a decoy skill)", re.search(pc, name_only) and not re.search(pc, tool_call) and not re.search(pc, decoy_call))
check("description-dropped: found when econometrics is listed by name only, also as the last entry",
      re.search(pd_, name_only) and re.search(pd_, last_entry))
check("description-dropped: NOT found when econometrics kept its description",
      not re.search(pd_, with_desc))
check("description-dropped: NOT found in a Skill call or in the agent file that names the skill",
      not re.search(pd_, tool_call) and not re.search(pd_, agent_file))
skill_text = "\n".join(p.read_text(encoding="utf-8") for p in SKILL.rglob("*.md") if "evals" not in p.parts)
check("no file of the skill contains a line that the diagnostic patterns would take for the listing",
      not re.search(pd_, json.dumps(skill_text)) and not re.search(pc, json.dumps(skill_text)))

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
