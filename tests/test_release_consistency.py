"""The numbers a release states in more than one place agree with each other (v2.6.9).

The check count appears in README.md ("**N checks**") and in the marketplace description ("N tested checks"); the
version appears in .claude-plugin/marketplace.json, reporting.SKILL_VERSION and the top entry of CHANGELOG.md.
Before v2.6.8 the two counts had drifted apart (2282 against 2469) without any check noticing. This test does not
fix the count itself, which changes with every version: it only requires the places that state it to agree.
Origin: item #5 of the structural optimization proposal (claude/avaliacao-proposta-otimizacao-2026-11-13.md in the
development repository), approved for v2.6.9.

Run from the repository root:  python3 tests/test_release_consistency.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "econometrics" / "scripts"))
import _console  # noqa: E402,F401

ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info)


def counts(readme: str, description: str) -> tuple[list[int], list[int]]:
    """Every "**N checks**" in the README and every "N tested checks" in the marketplace description."""
    r = [int(m.replace(",", "")) for m in re.findall(r"\*\*([\d,]+) checks\*\*", readme)]
    d = [int(m.replace(",", "")) for m in re.findall(r"\b([\d,]+) tested checks\b", description)]
    return r, d


def agree(readme: str, description: str) -> bool:
    r, d = counts(readme, description)
    return len(r) >= 1 and len(d) >= 1 and len(set(r + d)) == 1


# --------------------------------------------------------------- the rule itself, on texts built here (both sides)
check("the rule accepts equal counts", agree("- **2511 checks** in tests", "Econometric agent. 2511 tested checks."))
check("the rule rejects different counts (the 2282 vs 2469 drift of v2.6.7)",
      not agree("- **2469 checks** in tests", "Econometric agent. 2282 tested checks."))
check("the rule rejects a missing count on either side",
      not agree("- 2511 checks in tests", "2511 tested checks.") and not agree("- **2511 checks**", "tested."))
check("the rule rejects two different counts inside the README",
      not agree("**2511 checks** ... **2469 checks**", "2511 tested checks."))

# --------------------------------------------------------------- the repository
readme = (ROOT / "README.md").read_text(encoding="utf-8")
market = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))
plugins = market["plugins"]
check("every plugin in the marketplace states the README's check count",
      len(plugins) >= 1 and all(agree(readme, p_["description"]) for p_ in plugins),
      [counts(readme, p_["description"]) for p_ in plugins])
plugin = next(p_ for p_ in plugins if p_["name"] == "econometrics")

changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
top = re.search(r"^## v(\S+)", changelog, re.M).group(1)
skill_version = re.search(r'^SKILL_VERSION = "([^"]+)"',
                          (ROOT / "econometrics" / "scripts" / "reporting.py").read_text(encoding="utf-8"), re.M).group(1)
check("marketplace version = reporting.SKILL_VERSION = top of CHANGELOG",
      plugin["version"] == skill_version == top, (plugin["version"], skill_version, top))

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
