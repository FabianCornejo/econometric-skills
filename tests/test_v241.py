"""Tests for v2.4.1: statistics questions reach the skill.

In the v2.4.0 validation on Windows, a request to compare two means, test a 2x2 table and check
normality was answered without the skill. With many plugins installed, Claude Code's skill listing
went over its budget (962 skills, 37,000 characters), descriptions were dropped and the skill was
offered by name only; "econometrics" did not look like the place for basic statistics, and the
description did not mention hypothesis tests either. With the listing under the budget, the same
request loaded the skill and used the new inference module.

Two fixes, checked here: (1) the description of econometrics/SKILL.md names the statistical tasks;
(2) the plugin exposes a second entry point, econometrics/statistics/SKILL.md ("econometrics:statistics"),
whose name alone points statistics questions to the skill, and whose body only sends the agent to the
main skill and to the inference module. The app package (dist) keeps one skill.

Run from the repository root:  python3 tests/test_v241.py
"""
import importlib
import json
import re
import sys
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "econometrics"
ROUTER = SKILL / "statistics"
sys.path.insert(0, str(SKILL / "scripts"))

ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


def meta_of(path):
    text = path.read_text(encoding="utf-8")
    m = re.match(r"---\r?\n(.*?)\r?\n---\r?\n(.*)", text, re.S)
    if not m:
        return None, ""
    try:
        return yaml.safe_load(m.group(1)), m.group(2)
    except yaml.YAMLError:
        return None, m.group(2)


# ---------------------------------------------------------------- 1. the main description names the statistical tasks
main, _ = meta_of(SKILL / "SKILL.md")
desc = (main or {}).get("description", "")
needed = ["hypothesis", "two groups", "Welch", "proportions", "variances", "chi-square", "normality", "correlation",
          "power and sample size"]
missing = [w for w in needed if w.lower() not in desc.lower()]
check("SKILL.md description names the statistical tasks", not missing, str(missing))
check("SKILL.md description stays within 1024 characters", 0 < len(desc) <= 1024, f"{len(desc)} chars")
check("SKILL.md description still covers the econometrics and risk tasks",
      all(w in desc for w in ("regression", "effect of X on Y", "VaR", "WACC", "forecast", "DiD")))

# ---------------------------------------------------------------- 2. the statistics entry point
rmeta, body = meta_of(ROUTER / "SKILL.md")
check("statistics/SKILL.md has a valid YAML header with only name and description",
      isinstance(rmeta, dict) and sorted(rmeta) == ["description", "name"], str(rmeta and sorted(rmeta)))
rmeta = rmeta or {}
check("its name is 'statistics', equal to its folder", rmeta.get("name") == ROUTER.name == "statistics")
rdesc = rmeta.get("description", "")
check("its description is within 1024 characters, without angle brackets",
      0 < len(rdesc) <= 1024 and not re.search(r"[<>]", rdesc), f"{len(rdesc)} chars")
check("its description names the statistical tasks and the econometrics skill",
      all(w.lower() in rdesc.lower() for w in ("hypothesis tests", "two groups", "normality", "econometrics")))
check("its body sends the agent to the main skill (Skill tool, or ../SKILL.md)",
      "econometrics:econometrics" in body and "../SKILL.md" in body)
inf = importlib.import_module("inference")
named = re.findall(r"\b([a-z_]+)\b", body.split("inference.py", 1)[1].split(")", 1)[0]) if "inference.py" in body else []
named = [n for n in named if "_" in n or n in ("normality", "contingency", "nonparametric")]
absent = [n for n in named if not callable(getattr(inf, n, None))]
check("every inference function the entry point names exists", named and not absent, str(absent or named[:3]))
for rel in re.findall(r"`\.\./([^`]+)`", body):
    check(f"the entry point's path ../{rel} exists", (SKILL / rel).exists())
check("the entry point holds no method instructions of its own (short body)", len(body.split()) < 250,
      f"{len(body.split())} words")

# ---------------------------------------------------------------- 3. the plugin declares both, the app package keeps one
mk = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))
plug = mk["plugins"][0]
check("marketplace.json declares the main skill and the statistics entry point",
      plug.get("source") == "./econometrics" and plug.get("skills") == [".", "./statistics"], str(plug.get("skills")))
with zipfile.ZipFile(ROOT / "dist" / "econometrics.skill") as z:
    names = z.namelist()
check("dist/econometrics.skill (the app upload) carries one skill: no statistics/ folder",
      "econometrics/SKILL.md" in names and not any(n.startswith("econometrics/statistics/") for n in names))
readme = (ROOT / "README.md").read_text(encoding="utf-8")
check("README explains the listing budget and the statistics entry point",
      "skillListingBudgetFraction" in readme and "econometrics:statistics" in readme)

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
