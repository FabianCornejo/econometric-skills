"""Tests for v2.4.4: layers 4 and 5 of the robustness plan.

Layer 4, a dedicated agent: `econometrics/agents/econometrician.md` is a plugin subagent with the skill
preloaded (`skills:`), so delegated work does not depend on the skill listing. Checked live during development:
spawned through the Agent tool it starts with the skill's text in context (it quoted "# Applied econometrics"
and "## 7. Environment"); started as the main session with `claude --agent` it does not, hence step 0 of its
instructions.

Layer 5, routing evals: `econometrics/evals/routing/` holds cases for `claude plugin eval`, and
`tools/routing_evals.py` runs each one with the plugin alone and with 48 decoy skills that overflow the
listing. These tests check the files and build the harness without calling a model.

Run from the repository root:  python3 tests/test_v244.py
"""
import importlib.util
import json
import re
import sys
import tempfile
import zipfile
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
    m = re.match(r"---\r?\n(.*?)\r?\n---\r?\n(.*)", path.read_text(encoding="utf-8"), re.S)
    return (yaml.safe_load(m.group(1)), m.group(2)) if m else (None, "")


# ---------------------------------------------------------------- layer 4: the agent
agent = SKILL / "agents" / "econometrician.md"
meta, body = front(agent)
supported = {"name", "description", "model", "effort", "maxTurns", "tools", "disallowedTools", "skills", "memory",
             "background", "omitClaudeMd", "isolation", "color", "experimental"}
check("agent file parses and uses only fields plugin agents support", isinstance(meta, dict) and set(meta) <= supported,
      str(meta and sorted(meta)))
meta = meta or {}
check("agent name 'econometrician' (invoked as econometrics:econometrician)", meta.get("name") == "econometrician")
check("agent preloads the main skill by its namespaced name", meta.get("skills") == ["econometrics:econometrics"])
tools = [t.strip() for t in str(meta.get("tools", "")).split(",")]
check("agent can run the skill's scripts and write them (Read, Write, Edit, Bash) and load a skill (Skill)",
      {"Read", "Write", "Edit", "Bash", "Skill"} <= set(tools), str(tools))
desc = meta.get("description", "")
check("agent description says what to delegate, within 1024 characters, no angle brackets",
      0 < len(desc) <= 1024 and not re.search(r"[<>]", desc) and "Delegate" in desc)
check("agent body: skill-loaded check, mode, environment, files, reply contract",
      all(k in body for k in ("econometrics:econometrics", "# Applied econometrics", "Consultant", "Reviewer",
                               "setup_env.py --check", "working folder", "Your reply is all the caller sees")))
check("agent has no hooks or permissionMode (ignored for plugin agents)",
      "hooks" not in meta and "permissionMode" not in meta)

# ---------------------------------------------------------------- layer 5: the routing cases
cases = sorted(p for p in (SKILL / "evals" / "routing").iterdir() if p.is_dir())
check("at least 9 routing cases", len(cases) >= 9, str([c.name for c in cases]))
names_ok, graders_ok, tools_ok = [], [], []
for c in cases:
    m, prompt = front(c / "prompt.md")
    names_ok.append(isinstance(m, dict) and len(prompt.strip()) > 20 and set(m) <= {
        "description", "tags", "max_turns", "timeout_seconds", "allowed_tools", "name", "plugins", "runs", "model"})
    tools_ok.append(isinstance(m, dict) and not {"Bash", "Write", "Edit", "WebFetch", "WebSearch"} & set(m.get("allowed_tools", [])))
    gs = {g.stem: front(g)[0] for g in (c / "graders").glob("*.md")}
    graders_ok.append(bool(gs) and all(isinstance(v, dict) and v.get("type") in
                                       ("regex", "tool_used", "tool_order", "file_exists", "llm") for v in gs.values()))
check("every case has a valid prompt.md (known fields only)", all(names_ok))
check("every case has graders of known types", all(graders_ok))
check("no case needs a shell or write tool (runs on native Windows, no sandbox needed)", all(tools_ok))
skill_re = None
for c in cases:
    g = c / "graders" / "skill-fired.md"
    if g.exists():
        skill_re = front(g)[0]["input_match"]
        break
call = '{"skill": "econometrics:statistics", "args": ""}'
check("skill-fired matches a namespaced Skill call to any entry point, not another plugin's skill",
      skill_re and re.search(skill_re, call) and re.search(skill_re, '{"skill":"econometrics:econometrics"}')
      and not re.search(skill_re, '{"skill":"data:statistical-analysis"}'))
neg = front(SKILL / "evals" / "routing" / "unrelated-email" / "graders" / "skill-not-fired.md")[0]
check("negative case: the skill must not fire (min 0, max 0, scored in both arms)",
      neg.get("min") == 0 and neg.get("max") == 0 and neg.get("arm") == "both")
reached = front(SKILL / "evals" / "routing" / "regression-delegate" / "graders" / "reached.md")[0]["pattern"]
trace_skill = r'{"type":"tool_use","name":"Skill","input":{"skill":"econometrics:econometrics"}}'
trace_agent = r'"input":{\"subagent_type\":\"econometrics:econometrician\"}'
check("delegation case: 'reached' matches the skill or the econometrician agent in the trace",
      re.search(reached, trace_skill) and re.search(reached, trace_agent) and not re.search(reached, '"skill":"data:x"'))
for c in cases:
    for g in (c / "graders").glob("*.md"):
        gm = front(g)[0]
        if gm.get("type") == "regex":
            re.compile(gm["pattern"])
check("every regex grader compiles", True)

# ---------------------------------------------------------------- the harness (no model calls)
spec = importlib.util.spec_from_file_location("routing_evals", ROOT / "tools" / "routing_evals.py")
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)
with tempfile.TemporaryDirectory() as tmp:
    h = Path(tmp) / "harness"
    info = rt.build(h)
    built = sorted(p.name for p in (h / "evals").iterdir())
    check("harness: every routing case twice, clean and crowded",
          built == sorted([f"clean-{c.name}" for c in cases] + [f"crowded-{c.name}" for c in cases]))
    pm, _ = front(h / "evals" / "crowded-two-groups" / "prompt.md")
    cm, _ = front(h / "evals" / "clean-two-groups" / "prompt.md")
    check("harness: crowded cases load econometrics and the decoys, clean cases econometrics only",
          pm["plugins"] == ["econometrics", "crowd"] and cm["plugins"] == ["econometrics"] and "crowded" in pm["tags"])
    man = json.loads((h / "evals" / "clean-two-groups" / "econometrics" / ".claude-plugin" / "plugin.json").read_text())
    entry = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))["plugins"][0]
    check("harness: the plugin copy declares the marketplace's skills and version, keeps the agent, drops evals/",
          man["skills"] == entry["skills"] and man["version"] == entry["version"]
          and (h / "evals" / "clean-two-groups" / "econometrics" / "agents" / "econometrician.md").exists()
          and not (h / "evals" / "clean-two-groups" / "econometrics" / "evals").exists())
    decoys = list((h / "evals" / "crowded-two-groups" / "crowd" / "skills").glob("*/SKILL.md"))
    check("harness: 48 decoy skills whose descriptions overflow the listing (> 30,000 characters)",
          len(decoys) == 48 and info["crowd_chars"] > 30000, f"{info['crowd_chars']} chars")
    check("harness: every decoy has a valid header", all(isinstance(front(d)[0], dict) and front(d)[0]["name"] == d.parent.name
                                                        for d in decoys))
fake = {"cases": [
    {"name": "clean-a", "arms": {"with": [{"graders": [{"name": "skill-fired", "passed": True}, {"name": "answer", "passed": True}]}]}},
    {"name": "crowded-a", "arms": {"with": [{"graders": [{"name": "skill-fired", "passed": False}, {"name": "answer", "passed": True}]}]}},
    {"name": "clean-unrelated-email", "arms": {"with": [{"graders": [{"name": "skill-not-fired", "passed": True}]}]}}]}
table, rows = rt.summarize(fake)
ra = next(r for r in rows if r["case"] == "a")
neg_row = next(r for r in rows if r["case"] == "unrelated-email")
check("summary: per-arm rates; the negative case reports how often the skill wrongly fired",
      ra["clean skill fired"] == 1 and ra["crowded skill fired"] == 0 and neg_row["clean skill fired"] == 0
      and "Routing rate, clean: 100%" in table)

# ---------------------------------------------------------------- packaging
with zipfile.ZipFile(ROOT / "dist" / "econometrics.skill") as z:
    names = z.namelist()
check("app package (dist) carries one skill: no agents/, evals/, modes/ or statistics/",
      "econometrics/SKILL.md" in names and not any(n.startswith(tuple(f"econometrics/{d}/" for d in
                                                                       ("agents", "evals", "modes", "statistics")))
                                                  for n in names))
gi = (ROOT / ".gitignore").read_text(encoding="utf-8")
check("eval results are not committed", "routing-results/" in gi and "evals/results/" in gi)
readme = (ROOT / "README.md").read_text(encoding="utf-8")
check("README documents the agent and the routing evals",
      "@agent-econometrics:econometrician" in readme and "tools/routing_evals.py" in readme)

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
