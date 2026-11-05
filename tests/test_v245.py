"""Tests for v2.4.5: deliverables saved in the working folder; descriptions that reach quick and conceptual questions.

1. In the v2.4.4 test on Windows, the econometrician agent computed every number right but saved the script and
   the run manifest in its scratchpad, a temporary folder Claude Code offers for intermediate files, instead of
   the working folder. The agent and SKILL.md now say that a delivered script is the product, not a scratch file.
2. The v2.4.4 routing baseline found requests that never reached the skill, even with the plugin alone: a quick
   interval for a proportion and the review of a "not significant, so no effect" conclusion (both 0%), and a
   lesson (50%). The descriptions now name quick calculations, conceptual questions and reviews of a
   conclusion, and say why the skill should be used (its functions and rules catch errors of answers from
   memory, such as the Wald interval). Two more unrelated requests check that the skill does not fire on plain
   arithmetic or a spreadsheet formula.

Run from the repository root:  python3 tests/test_v245.py
"""
import importlib.util
import re
import sys
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


def meta(path):
    m = re.match(r"---\r?\n(.*?)\r?\n---\r?\n(.*)", path.read_text(encoding="utf-8"), re.S)
    return yaml.safe_load(m.group(1)), m.group(2)


# ---------------------------------------------------------------- 1. deliverables in the working folder
agent_meta, agent_body = meta(SKILL / "agents" / "econometrician.md")
check("agent: the delivered script and manifest are the product and go to the working folder, even with a scratchpad",
      all(k in agent_body for k in ("product of the work, not scratch files", "even if your environment offers a scratchpad",
                                    "Use the scratchpad only for throwaway files")))
check("agent: a system or read-only working folder means asking, not falling back to a temporary folder",
      "ask where to save instead of falling back to a temporary folder" in agent_body)
skill_text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
check("SKILL.md: the same rule for every mode", "even when the environment offers a scratchpad or temporary directory: "
      "a delivered script is the product, not a scratch file" in skill_text)

# ---------------------------------------------------------------- 2. descriptions
main, _ = meta(SKILL / "SKILL.md")
d = main["description"]
check("main description within 1024 characters", 0 < len(d) <= 1024, f"{len(d)} chars")
check("main description reaches quick calculations, conceptual questions and reviews of a conclusion",
      all(k in d for k in ("even a quick one you could do by hand", "an interval for a proportion", "a sample size",
                           "a p-value", "a conceptual one", "review a conclusion, report or paper")))
check("main description says why to load it (tested functions catch errors of answers from memory)",
      "catch errors of answers from memory" in d)
check("main description still names what it is for, and what it replaces",
      all(k in d for k in ("Welch", "power and sample size", "effect of X on Y", "VaR", "WACC", "forecast", "DiD",
                           "teaching a method", "not data, finance or risk skills")))
stat, _ = meta(SKILL / "statistics" / "SKILL.md")
sd = stat["description"]
check("statistics entry point: quick calculations, 'not significant' readings and the Wald trap, within 1024 characters",
      0 < len(sd) <= 1024 and all(k in sd for k in ("quick calculations you could do by hand", "confidence interval for a proportion",
                                                    "'not significant' result means no effect", "Wald interval")))

# ---------------------------------------------------------------- 3. guards against false triggers
routing = SKILL / "evals" / "routing"
negatives = [c.name for c in routing.iterdir() if (c / "graders" / "skill-not-fired.md").exists()]
check("at least three unrelated requests must not fire the skill (e-mail, arithmetic, spreadsheet formula)",
      {"unrelated-email", "simple-arithmetic", "excel-formula"} <= set(negatives), str(sorted(negatives)))
for n in negatives:
    g, _ = meta(routing / n / "graders" / "skill-not-fired.md")
    check(f"{n}: Skill must be called 0 times, scored in both arms", g["min"] == 0 and g["max"] == 0 and g["arm"] == "both")
spec = importlib.util.spec_from_file_location("routing_evals", ROOT / "tools" / "routing_evals.py")
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)
fake = {"cases": [
    {"name": "clean-a", "arms": {"with": [{"graders": [{"name": "skill-fired", "passed": True}]}]}},
    {"name": "clean-simple-arithmetic", "arms": {"with": [{"graders": [{"name": "skill-not-fired", "passed": False}]},
                                                          {"graders": [{"name": "skill-not-fired", "passed": True}]}]}}]}
table, rows = rt.summarize(fake)
neg = next(r for r in rows if r["case"] == "simple-arithmetic")
check("summary: any case with skill-not-fired is a negative, reported as a false-trigger rate",
      neg["negative"] and neg["clean skill fired"] == 0.5 and "False triggers, clean: 50%" in table
      and "Routing rate, clean: 100% over 1 cases" in table)

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
