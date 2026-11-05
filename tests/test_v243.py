"""Tests for v2.4.3: English command names, answers that read well in a terminal, scripts saved where they last.

The v2.4.2 validation on Windows ran /econometrics:professor in Claude Code. The lesson was right, but
(1) the command names were Portuguese while the skill, its modes (Consultant, Teacher, Reviewer) and the
repository are English; (2) formulas came out as raw LaTeX ($$\\operatorname{Var}...$$) and the solution inside
<details> HTML, both printed literally by the terminal; (3) Claude Code was started in C:\\Windows\\System32,
so the lesson's script went to a temporary scratchpad folder that the system may delete.

Run from the repository root:  python3 tests/test_v243.py
"""
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "econometrics"
sys.path.insert(0, str(SKILL / "scripts"))
import _console  # noqa: E402,F401  (Windows-safe output of the non-ASCII check names)

ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


# ---------------------------------------------------------------- 1. English command names = the modes in SKILL.md
skill_md = (SKILL / "SKILL.md").read_text(encoding="utf-8")
modes_in_table = [m.lower() for m in re.findall(r"^\| \*\*(\w+)\*\*", skill_md, re.M)]
folders = sorted(p.name for p in (SKILL / "modes").iterdir() if p.is_dir())
check("the mode commands are the SKILL.md modes, in English", sorted(modes_in_table) == folders == ["consultant", "reviewer", "teacher"],
      f"{modes_in_table} vs {folders}")
mk = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))
check("marketplace.json declares /econometrics:consultant, :teacher, :reviewer",
      mk["plugins"][0]["skills"] == [".", "./statistics", "./modes/consultant", "./modes/teacher", "./modes/reviewer"])
for mode in folders:
    text = (SKILL / "modes" / mode / "SKILL.md").read_text(encoding="utf-8")
    meta = yaml.safe_load(re.match(r"---\r?\n(.*?)\r?\n---\r?\n", text, re.S).group(1))
    check(f"{mode}: name matches the folder, the description names the command and the mode",
          meta["name"] == mode and f"/econometrics:{mode}" in meta["description"]
          and f"{mode.capitalize()} mode" in meta["description"] and meta.get("disable-model-invocation") is True)
old = ("consultor", "professor", "revisor")
stale = [str(p.relative_to(ROOT)) for p in list(ROOT.glob("econometrics/**/*")) + [ROOT / "README.md", ROOT / ".claude-plugin" / "marketplace.json"]
         if p.is_file() and p.suffix in (".md", ".json", ".py") and any(f"econometrics:{o}" in p.read_text(encoding="utf-8") for o in old)]
check("no Portuguese command names left outside the CHANGELOG", not stale, str(stale))

# ---------------------------------------------------------------- 2. rendering for the terminal
teach = (SKILL / "references" / "teaching.md").read_text(encoding="utf-8")
rend = re.search(r"\*\*Rendering\.\*\*(.*?)\n\n", teach, re.S)
body = rend.group(1) if rend else ""
check("teaching.md has a Rendering rule: terminal → plain-text Unicode formulas, no LaTeX, no HTML",
      bool(rend) and "terminal" in body and "Unicode" in body and "LaTeX" in body and "<details>" in body)
check("the method card no longer prescribes LaTeX for every medium", "(display LaTeX)" not in teach)
check("SKILL.md tells every mode to avoid LaTeX and HTML in a terminal",
      re.search(r"In a terminal \(Claude Code\) write formulas in plain text with Unicode symbols, not LaTeX, and use no HTML",
                skill_md) is not None)

# ---------------------------------------------------------------- 3. where scripts are saved in Claude Code
check("SKILL.md: scripts go to the user's working folder; a system folder means asking, never a temporary folder",
      re.search(r"save the script[^.]*in the user's working folder", skill_md) is not None and "System32" in skill_md
      and "instead of using a temporary or scratchpad folder" in skill_md)

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
