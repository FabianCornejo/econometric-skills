"""Tests for v2.3.5: the SKILL.md header is valid YAML, so Claude Code can see the skill.

From v2.2.3 the description held ": " ("which do not estimate: VaR ...", "Methods: OLS ...") without
quotes. YAML reads ": " as a new key, the whole header failed to parse, and Claude Code loaded the
skill with no name and no description: the skill was never offered to the model, and Claude answered
without it. Claude Code only logs a warning ("Failed to parse YAML frontmatter"), so nothing visible
failed. The Python tests passed because they never read the header.

Checks: SKILL.md starts with a header between '---' lines; a strict YAML parser reads it; it has
exactly the keys name and description; name is lowercase letters, digits and hyphens (at most 64)
and equals the skill folder; description is a non-empty string of at most 1024 characters with no
angle brackets (no XML tags); the packaged dist/econometrics.skill carries the same header.

Run from the repository root:  python3 tests/test_skill_metadata.py
"""
import re
import sys
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "econometrics"

ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


def header(text):
    m = re.match(r"---\r?\n(.*?)\r?\n---\r?\n", text, re.S)
    return m.group(1) if m else None


def parse(fm):
    try:
        return yaml.safe_load(fm), ""
    except yaml.YAMLError as e:
        return None, str(e).splitlines()[0][:100]


text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
fm = header(text)
check("SKILL.md starts with a header between '---' lines", fm is not None)
meta, err = parse(fm or "")
check("a strict YAML parser reads the header (an unquoted ': ' in the description breaks it)",
      isinstance(meta, dict), err)
meta = meta if isinstance(meta, dict) else {}
check("the header has exactly the keys name and description", sorted(meta) == ["description", "name"], sorted(meta))
name, desc = meta.get("name"), meta.get("description")
check("name is lowercase letters, digits and hyphens, at most 64 characters",
      isinstance(name, str) and re.fullmatch(r"[a-z0-9-]{1,64}", name) is not None, repr(name))
check("name equals the skill folder", name == SKILL.name, f"{name!r} vs {SKILL.name!r}")
check("description is a non-empty string of at most 1024 characters",
      isinstance(desc, str) and 0 < len(desc.strip()) <= 1024, f"{len(desc) if isinstance(desc, str) else type(desc)} chars")
check("description has no angle brackets (no XML tags)", isinstance(desc, str) and not re.search(r"[<>]", desc))
check("description says when to use the skill in its first sentence",
      isinstance(desc, str) and "econometrics" in desc.split(".")[0].lower())

with zipfile.ZipFile(ROOT / "dist" / "econometrics.skill") as z:
    packed = z.read("econometrics/SKILL.md").decode("utf-8")
pmeta, perr = parse(header(packed) or "")
check("the packaged dist/econometrics.skill carries the same, valid header", pmeta == meta and bool(meta), perr)

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
