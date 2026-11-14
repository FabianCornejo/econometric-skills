"""Tests for v2.6.10 (core): the number rule in SKILL.md section 4, the repaired delivery sentence, the locator paths
kept in one place, and the doctor's closing lines when the latest release was not checked.

Origin: the v2.6.9 validation (notebook 2, 2026-11-14). (1) The reply worked out two numbers outside the tables (a
combined slope and an MDE), one of them flagged as "my own arithmetic"; the v2.6.8 rule had failed the same way in the
v2.6.7 validation. (2) The agent ran `setup_env.py --doctor --offline 2>&1 | Select-Object -Last 25` on a computer with
network; reading the doctor then showed that with --offline and nothing else to do it closed with "the setup is
complete and up to date", although no version had been checked. (3) SKILL.md section 4 had a cut sentence
("... which the system may delete. and test it on data ...", review of 2026-11-13, item #6).

Checks:
  1. section 4: the delivery rule reads "Test the script on data that include the awkward cases"; no ". and test";
     it still points to the locator block and still forbids the versioned cache path and sandbox paths;
  2. what left section 4 (the locator's paths and the install message) is in the "Script skeleton" of
     interpretation-rules.md, literally;
  3. section 4's number rule covers numbers outside the tables and says a disclaimer does not exempt them;
     interpretation-rules.md has "Rules added in v2.6.10", after v2.6.8, with the validation's two numbers and the
     functions it names (which exist);
  4. the doctor's text: never "up to date" when the latest release was not checked; with --offline the last line
     asks to run it again without the flag (so a tail of the output shows it); the JSON actions do not change.

Run from the repository root:  python3 tests/test_v2610.py
"""
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "econometrics"
sys.path.insert(0, str(SKILL / "scripts"))

import estimation  # noqa: E402
import setup_env as se  # noqa: E402
import simulation  # noqa: E402

results = []


def check(name, cond, info=""):
    results.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info)


skill_md = (SKILL / "SKILL.md").read_text(encoding="utf-8")
sec4 = skill_md[skill_md.index("## 4. Non-negotiable rules"):skill_md.index("## 5.")]
rules = (SKILL / "references" / "interpretation-rules.md").read_text(encoding="utf-8")

# ----------------------------------------------------------------------------------- 1. the delivery rule
deliver = next(ln for ln in sec4.splitlines() if ln.startswith("- **Deliver code, not only a plan.**"))
check("section 4: the cut sentence is repaired ('Test the script on data that include the awkward cases')",
      "which the system may delete. Test the script on data that include the awkward cases (skipped events, missing "
      "values, repeated units)." in deliver and ". and test" not in skill_md, deliver[-200:])
check("section 4: the delivery rule still starts the script with the locator block of interpretation-rules.md",
      "Start the script with the locator block in `references/interpretation-rules.md` (\"Script skeleton\")"
      in deliver)
check("section 4: still never a sandbox path or the versioned cache folder, not even in a comment; the variable "
      "for your own environment is named",
      "never put a sandbox path or the versioned `plugins/cache/.../<version>/` folder in it, not even in a comment"
      in deliver and "set `ECONOMETRICS_SKILL_DIR` instead" in deliver)
check("section 4: the locator's paths are no longer repeated in the rule (one place: the Script skeleton)",
      "plugins/marketplaces" not in deliver and "skills/synced" not in deliver and "It also turns" not in deliver)
check("section 4: the delivery rule is shorter than in v2.6.9 (259 words)", len(deliver.split()) < 259,
      len(deliver.split()))

# ----------------------------------------------------------------------------------- 2. what left section 4
skel = rules[rules.index("Script skeleton for every deliverable"):rules.index("## Rules added in v2.3.1")]
for piece in ('os.environ.get("ECONOMETRICS_SKILL_DIR")',
              '".claude/plugins/marketplaces/econometric-skills/econometrics"',
              "including the copy synced from the Claude app",
              "<path to your clone>/econometric-skills/econometrics",
              'scripts/setup_env.py\\" --install',
              "plugins/cache/econometric-skills/econometrics/<version>/"):
    check(f"Script skeleton keeps what left section 4: {piece}", piece in skel)

# ----------------------------------------------------------------------------------- 3. the number rule
num = next(ln for ln in sec4.splitlines() if ln.startswith("- **No invented numbers**"))
check("section 4: every number of the reply, summary and document is printed by the script, including derived ones",
      "Every number in the reply, its summary and the document" in num
      and "a sum of coefficients or a combined slope, an MDE, an interval, a percentage" in num
      and "is printed by the script before it is written" in num)
check("section 4: a disclaimer ('my own arithmetic') does not exempt a number",
      "saying it is your own arithmetic does not exempt it" in num and "v2.6.8 and v2.6.10" in num)
heads = [ln for ln in rules.splitlines() if ln.startswith("## Rules added in v2.6.")]
check("interpretation-rules.md: 'Rules added in v2.6.10' exists once, after v2.6.8",
      heads.count("## Rules added in v2.6.10") == 1 and "## Rules added in v2.6.8" in heads
      and heads.index("## Rules added in v2.6.10") > heads.index("## Rules added in v2.6.8"), heads)
start = rules.index("## Rules added in v2.6.10")
nxt = rules.find("\n## ", start + 5)
v2610 = rules[start:nxt if nxt > 0 else len(rules)]
check("v2.6.10 section: the validation's two numbers (combined slope, MDE) and the flagged disclaimer",
      "−0.5174 + 0.4841" in v2610 and "2.8 × SE" in v2610 and "my own arithmetic" in v2610)
check("v2.6.10 section: outside the tables; a disclaimer does not replace the script; add the print and rerun",
      "outside the tables" in v2610 and "a disclaimer does not replace the script" in v2610
      and "add the print, run the script again" in v2610)
check("v2.6.10 section: the functions it names exist (estimation.delta_method, simulation.power_mde)",
      "`estimation.delta_method`" in v2610 and "`simulation.power_mde`" in v2610
      and callable(getattr(estimation, "delta_method", None)) and callable(getattr(simulation, "power_mde", None)))


# ----------------------------------------------------------------------------------- 4. the doctor's text
def report(**over):
    rep = {"python": {"version": "3.13.1", "exe": "py", "supported": True, "run_with": "py -3.13"},
           "interpreters": [], "core_missing": [], "extras": {},
           "skill": {"version": "2.6.10", "dir": "d"}, "latest": {"version": "2.6.10", "status": "ok"},
           "copies": [], "autoupdate": {"registered": False},
           "listing": {"checked": True, "folder": "f", "skills": 12, "enabled_plugins": 2, "crowded": False},
           "actions": []}
    rep.update(over)
    return rep


def shown(rep):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        se.print_doctor(rep)
    return buf.getvalue()


OFF = {"version": None, "status": "not checked (--offline)"}
AGAIN = "if this computer has network, run --doctor again without --offline before answering"
o_ok = shown(report())
o_off = shown(report(latest=OFF))
o_err = shown(report(latest={"version": None, "status": "not checked (HTTP Error 403)"}))
o_off_act = shown(report(latest=OFF, actions=["crowded skill listing: offer a clean analysis folder"]))
last = lambda o: o.rstrip("\n").splitlines()[-1]  # noqa: E731
check("doctor text: checked and nothing to do -> 'up to date', and no line about --offline",
      "Nothing to do: the setup is complete and up to date." in o_ok and AGAIN not in o_ok, o_ok)
check("doctor text: --offline and nothing to do -> never 'up to date'; says the release was not checked",
      "up to date" not in o_off and "Nothing to do in this setup; the latest release was not checked (--offline), "
      "so the skill may be out of date." in o_off, o_off)
check("doctor text: --offline -> the last line asks to run it again without the flag (a tail still shows it)",
      AGAIN in last(o_off) and AGAIN in last(o_off_act), (last(o_off), last(o_off_act)))
check("doctor text: a release check that failed for another reason -> no 'up to date' and no --offline line",
      "up to date" not in o_err and "not checked (HTTP Error 403)" in o_err and AGAIN not in o_err, o_err)
check("doctor text: the v2.6.6 line for --offline and the To do header stay",
      "--offline is only for a machine without network; with network, run --doctor without it" in o_off
      and "\n  1. crowded skill listing" in o_off_act and "To do (ask the user once" in o_off_act, o_off_act)
calls = []
with tempfile.TemporaryDirectory() as tmp:     # isolated, as in test_v266: no real ~/.claude, no CLI, no network
    rep_off = se.doctor(offline=True, cdir=Path(tmp), env={}, opener=lambda *a, **k: calls.append(a),
                        run=lambda *a, **k: (1, ""), which=lambda n: None, cwd=tmp)
check("doctor --offline: no request; the JSON status and the actions list are what they were (only the text changed)",
      calls == [] and json.loads(json.dumps(rep_off))["latest"] == OFF
      and rep_off["actions"] == se.doctor_actions(rep_off), rep_off["latest"])
o_real = shown(rep_off)
check("doctor --offline end to end: the printed report ends with the line asking for a run without --offline",
      AGAIN in last(o_real) and "complete and up to date" not in o_real, last(o_real))

passed = sum(results)
print(f"\n{passed}/{len(results)} checks passed")
sys.exit(0 if passed == len(results) else 1)
