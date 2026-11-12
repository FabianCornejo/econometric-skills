"""Tests for v2.6.5: the doctor and Claude accounts; rho on the dml_bounds lines; small items of the backlog triage.

Origin: the v2.6.4 validation on notebook 1. Claude Code was signed in to one Claude account, the doctor listed the
skill copy that ANOTHER account had synced to the same computer (~/.claude/skills/synced/<orgId>_<id>, left on disk
at 2.6.3) as "the Claude app copy" and told the user to update it; the agent passed that on. Checked on the notebook
before the fix (development repository, claude/plano-v2.6.5.md, E0 and E1): `claude auth status` prints a JSON object
with the account's "orgId"; each synced folder is named <orgId>_<another UUID>; with both folders on disk Claude Code
listed one anthropic-skills:econometrics, the signed-in account's. Also: dml_bounds printed rho with 2 decimals on its
lines (0.996 as "1.00", which reads as the worst case) while its header and report line print it with :g.

The fixtures copy the shape seen on the notebook (the keys of `claude auth status`, two UUIDs joined by "_" in the
folder name) with made-up UUIDs and e-mail: the real ones stay out of this public repository.

What is checked (expected items written by hand, not derived from the code):
  1. signed_in_org: the notebook's JSON gives the orgId and orgName; None for an exit code other than 0, empty or
     non-JSON output, a JSON list, loggedIn false, no orgId, an orgId that is not a UUID, a timeout (-1, ""), no CLI,
     and a run() that raises; an upper-case orgId is accepted;
  2. synced_account, both sides of each boundary: the signed-in org -> "signed in"; an org differing in its last
     character, or only sharing the first 8 characters -> "other"; upper/lower case ignored; a folder name that is
     the orgId with no "_<id>", or the orgId plus one character before "_" -> "unknown"; no org -> "unknown";
  3. skill_copies on a notebook-shaped folder: plugin, checkout, the two synced copies with their accounts; .trash
     still ignored; without an org every synced copy is "unknown";
  4. doctor_actions: other account's copy behind -> no item at all and its version in none; the signed-in copy
     behind -> the usual item; offline, an older copy of another account -> no "different versions" item; account
     unknown with copies of two accounts -> one item naming both versions and no update order aimed at one of them,
     none when both are current; one synced copy with the account unknown -> the v2.6.4 item, unchanged;
  5. skill_listing: with synced skills of two accounts (3 and 5 SKILL.md), only the signed-in account's count;
     both without an org; projectPath "" is never this folder's install (v2.5.2 NIT: "" resolved to the
     current folder), and a projectPath equal to the folder still is;
  6. doctor end to end with a fake CLI: the account in the report (and in --json), the printed owner of each copy,
     the old To do item gone;
  7. dml_bounds prints rho on each line as :g (0.9726123 -> 0.972612; 0.996 -> 0.996, never 1.00; 1 -> 1; -1 -> -1),
     and with a numeric rho = 1/3 the line, the header and the report line show the same text;
  8. .gitattributes marks *.skill binary (git check-attr when git is there).
Review 1 added: "other" only next to a signed-in copy (the checked case), else "unknown"; the accounts item only when the
account was not read, after the "different versions" item (it hid it), with "unreadable" for a missing version; the
copy running now never silenced; a BOM or a warning line before the JSON; folder names in full (<UUID>_<UUID>); no
`claude auth status` call without a synced folder; rho 0.9999996 printed in full and -0.0 as 0.

Run from the repository root:  python3 tests/test_v265.py
Needs: numpy, pandas, scikit-learn (part 7); git for one check of part 8.
"""
import contextlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path.insert(0, str(ROOT / "econometrics" / "scripts"))
import _console  # noqa: E402,F401  (Windows-safe console output, see econometrics/scripts/_console.py)
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import causal
import setup_env as se

ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


ME = "3f2a9c41-7b5e-4d10-9a6c-0e8f21b4c7d3"            # made-up UUIDs in the notebook's shape
ME_ID = "5c8e1d22-a940-4f3b-b6d7-91e0f4a2c815"
OTHER = "9b7d4e60-2c1a-4e8f-8d35-6a0b9c2e1f47"
OTHER_ID = "d14f7a83-6e2b-4c90-a5d1-3b8e0f6c9a12"
AUTH = {"loggedIn": True, "authMethod": "claude.ai", "apiProvider": "firstParty", "analyticsDisabled": False,
        "projectsDirectory": "C:\\Users\\U\\.claude\\projects", "configDirectory": "C:\\Users\\U\\.claude",
        "loginExpiresAt": "2026-12-11T06:32:59.948Z", "email": "user@example.com", "orgId": ME,
        "orgName": "Org A", "subscriptionType": "team", "storedClaudeAiLogin": True, "planTier": "other"}
CLAUDE = lambda n: "/bin/claude" if n == "claude" else None     # noqa: E731


def runner(auth=None, code=0, plugins="[]", raises=None):
    def run(cmd, timeout):
        if raises:
            raise raises
        if cmd[1:3] == ["auth", "status"]:
            return code, auth if isinstance(auth, str) else json.dumps(auth)
        if cmd[1:4] == ["plugin", "list", "--json"]:
            return 0, plugins
        return 0, ""
    return run


def write_copy(folder: Path, version: str):
    (folder / "scripts").mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text("---\nname: econometrics\n---\n", encoding="utf-8")
    (folder / "scripts" / "reporting.py").write_text(f'SKILL_VERSION = "{version}"\n', encoding="utf-8")


# ----------------------------------------------------------------------------------- 1. signed_in_org
got = se.signed_in_org(run=runner(AUTH), which=CLAUDE)
check("signed_in_org: the notebook's JSON (13 keys, orgId and orgName) gives the org", got == {"org_id": ME, "org_name": "Org A"},
      got)
bad = {"exit code 1": runner(AUTH, code=1), "empty output": runner(""), "not JSON": runner("Logged in as Org A"),
       "a JSON list": runner([AUTH]), "loggedIn false": runner({**AUTH, "loggedIn": False}),
       "no orgId": runner({k: v for k, v in AUTH.items() if k != "orgId"}),
       "orgId not a UUID": runner({**AUTH, "orgId": "Org A"}), "orgId a UUID with a suffix": runner({**AUTH, "orgId": ME + "x"}),
       "orgId a number": runner({**AUTH, "orgId": 7}),
       "a JSON list printed with indent (review 2)": runner(json.dumps([AUTH], indent=2)), "a timeout (_run gives -1, '')": runner("", code=-1),
       "run() raises": runner(raises=OSError("boom"))}
for label, run in bad.items():
    check(f"signed_in_org: {label} -> None (account unknown, no exception)",
          se.signed_in_org(run=run, which=CLAUDE) is None)
for label, txt in (("a UTF-8 BOM before the JSON", "\ufeff" + json.dumps(AUTH, indent=2)),
                   ("a warning line before the JSON", "Note: a newer version is available\n" + json.dumps(AUTH, indent=2))):
    got = se.signed_in_org(run=runner(txt), which=CLAUDE)
    check(f"signed_in_org: {label} -> still read (review 1)", got == {"org_id": ME, "org_name": "Org A"}, got)
check("signed_in_org: no claude CLI on the PATH -> None", se.signed_in_org(run=runner(AUTH), which=lambda n: None) is None)
up = se.signed_in_org(run=runner({**AUTH, "orgId": ME.upper(), "orgName": ""}), which=CLAUDE)
check("signed_in_org: an upper-case orgId is accepted; an empty orgName becomes None",
      up == {"org_id": ME.upper(), "org_name": None}, up)

# ----------------------------------------------------------------------------------- 2. synced_account
parts = lambda folder: ("synced", folder, "econometrics", "SKILL.md")   # noqa: E731
cases = [
    (f"{ME}_{ME_ID}", ME, "signed in", "the signed-in org"),
    (f"{ME}_{ME_ID}".upper(), ME, "signed in", "folder in upper case"),
    (f"{ME}_{ME_ID}", ME.upper(), "signed in", "org in upper case"),
    (f"{OTHER}_{OTHER_ID}", ME, "other", "another org"),
    (f"{ME[:-1]}4_{ME_ID}", ME, "other", "an org differing in its last character"),
    (f"{ME[:8]}-0000-4000-8000-000000000000_{ME_ID}", ME, "other", "an org sharing only the first 8 characters"),
    (ME, ME, "unknown", "the orgId with no '_<id>'"),
    (f"{ME}0_{ME_ID}", ME, "unknown", "the orgId plus one character before '_'"),
    ("acct", ME, "unknown", "a folder name with no orgId"),
    (f"{ME}_x", ME, "unknown", "the orgId and '_' but no second UUID (review 1)"),
    (f"{ME}_", ME, "unknown", "the orgId and '_' only"),
    (f"{OTHER}_{OTHER_ID}x", ME, "unknown", "a second UUID with one character more"),
    (f"{ME}_{ME_ID}", None, "unknown", "org not known"),
]
for folder, org, want, label in cases:
    got = se.synced_account(parts(folder), org)
    check(f"synced_account: {label} -> {want}", got == want, got)
check("synced_account: a path with no synced/ folder (a manual copy) -> unknown",
      se.synced_account(("econometrics", "SKILL.md"), ME) == "unknown")

# ----------------------------------------------------------------------------------- 3. skill_copies
tmp = Path(tempfile.mkdtemp())
cdir = tmp / ".claude"
plug = cdir / "plugins" / "cache" / "econometric-skills" / "econometrics" / "2.6.4"
write_copy(plug, "2.6.4")
write_copy(cdir / "plugins" / "marketplaces" / "econometric-skills" / "econometrics", "2.6.4")
write_copy(cdir / "skills" / "synced" / f"{ME}_{ME_ID}" / "econometrics", "2.6.4")
write_copy(cdir / "skills" / "synced" / f"{OTHER}_{OTHER_ID}" / "econometrics", "2.6.3")
write_copy(cdir / "skills" / ".trash" / "123-x" / "econometrics", "2.6.2")
(cdir / "plugins" / "installed_plugins.json").write_text(json.dumps(
    {"version": 2, "plugins": {se.PLUGIN_ID: [{"scope": "user", "installPath": str(plug), "version": "2.6.4"}]}}),
    encoding="utf-8")
cp = se.skill_copies(cdir, org_id=ME)
syn = {c["version"]: c.get("account") for c in cp if c["kind"] == se.APP_COPY}
check("skill_copies (notebook shape): the signed-in copy 2.6.4 and the other account's 2.6.3, marked",
      syn == {"2.6.4": "signed in", "2.6.3": "other"}, syn)
check("skill_copies: plugin and checkout as before, with no account key",
      [(c["kind"], c["version"]) for c in cp if c["kind"] != se.APP_COPY]
      == [("plugin", "2.6.4"), ("marketplace checkout", "2.6.4")] and all("account" not in c for c in cp
                                                                         if c["kind"] != se.APP_COPY), cp)
check("skill_copies: the copy in .trash is still ignored", not any(c["version"] == "2.6.2" for c in cp))
t2 = Path(tempfile.mkdtemp()) / ".claude"
write_copy(t2 / "skills" / "synced" / f"{OTHER}_{OTHER_ID}" / "econometrics", "2.6.3")
only = se.skill_copies(t2, org_id=ME)
check("skill_copies: signed in, but only another account's folder on disk -> 'unknown', not 'other' (the case not "
      "checked on the notebook; review 1)", [c.get("account") for c in only] == ["unknown"], only)
check("skill_copies: no org -> every synced copy 'unknown'",
      [c.get("account") for c in se.skill_copies(cdir) if c["kind"] == se.APP_COPY] == ["unknown", "unknown"])


# ----------------------------------------------------------------------------------- 4. doctor_actions
def report(copies, latest="2.6.4", org=None, here="/c/plugins/cache/econometric-skills/x/2.6.4"):
    return {"account": {"org_id": org, "org_name": None}, "python": {"exe": "x", "version": "3.13.0", "supported": True, "run_with": "py -3.13"},
            "core_missing": [], "interpreters": [], "skill": {"dir": here, "version": "2.6.4"},
            "latest": {"version": latest, "status": "ok" if latest else "not checked (--offline)"},
            "copies": copies, "autoupdate": {"registered": True, "on": True, "configured": True,
                                             "source": "settings.json", "blocked_by": []},
            "listing": {"checked": True, "skills": 12, "enabled_plugins": 2, "crowded": False}}


P = {"kind": "plugin", "version": "2.6.4", "path": "p"}
app = lambda v, a: {"kind": se.APP_COPY, "version": v, "path": f"s/{a}/{v}", "account": a}   # noqa: E731
acts = se.doctor_actions(report([P, app("2.6.4", "signed in"), app("2.6.3", "other")]))
check("To do (the v2.6.4 notebook case): the other account's 2.6.3 copy -> no item", acts == [], acts)
acts = se.doctor_actions(report([P, app("2.6.3", "other")]))
check("To do: only another account's copy, behind -> no item, and '2.6.3' nowhere", acts == [], acts)
acts = se.doctor_actions(report([P, app("2.6.3", "signed in"), app("2.6.4", "other")]))
check("To do: the signed-in copy behind (the other current) -> exactly the usual Customize -> Skills item",
      len(acts) == 1 and acts[0].startswith("the Claude app copy is 2.6.3, the latest is 2.6.4") and
      "Customize -> Skills" in acts[0], acts)
acts = se.doctor_actions(report([P, app("2.6.4", "signed in"), app("2.6.3", "other")], latest=None))
check("To do offline: an older copy of another account -> no 'different versions' item", acts == [], acts)
acts = se.doctor_actions(report([P, app("2.6.3", "signed in"), app("2.6.4", "other")], latest=None))
check("To do offline: the signed-in copy at another version -> 'different versions' naming 2.6.3, not the other's",
      len(acts) == 1 and "different versions" in acts[0] and "2.6.3" in acts[0] and
      acts[0].count("2.6.4") == 1, acts)
acts = se.doctor_actions(report([P, app("2.6.4", "unknown"), app("2.6.3", "unknown")]))
check("To do, account unknown, copies of two accounts (2.6.4, 2.6.3) -> one item naming both, saying the account "
      "could not be read, and no 'the Claude app copy is 2.6.3' order",
      len(acts) == 1 and "2 Claude app copies are on this computer" in acts[0] and "versions 2.6.4, 2.6.3," in acts[0] and
      "could not be read" in acts[0] and "the Claude app copy is" not in acts[0] and "different versions" not in acts[0],
      acts)
acts = se.doctor_actions(report([P, app("2.6.4", "unknown"), app("2.6.4", "unknown")]))
check("To do, account unknown, two accounts both current -> no item", acts == [], acts)
acts = se.doctor_actions(report([P, app("2.6.4", "unknown"), app("2.6.3", "unknown")], latest=None))
check("To do offline, account unknown, two accounts at different versions -> the one item, not 'different versions'",
      len(acts) == 1 and "2 Claude app copies are on this computer" in acts[0] and "behind 2.6.4," in acts[0], acts)
acts = se.doctor_actions(report([P, app("2.6.4", "unknown"), app("2.6.4", "unknown")], latest=None))
check("To do offline, account unknown, two accounts at the plugin's version -> no item", acts == [], acts)
acts = se.doctor_actions(report([P, app("2.6.3", "unknown"), app("2.6.3", "unknown")], latest=None))
check("To do offline, account unknown, two accounts equal to each other but behind the plugin (2.6.3 < 2.6.4) -> "
      "the one item (the reference is the plugin's version when the release is not checked)",
      len(acts) == 1 and "2 Claude app copies are on this computer" in acts[0] and "versions 2.6.3, 2.6.3," in acts[0], acts)
old = {"kind": se.APP_COPY, "version": "2.6.3", "path": "s"}       # a report without "account" (before v2.6.5)
for label, c in (("one synced copy, account unknown", app("2.6.3", "unknown")), ("a report with no account key", old)):
    acts = se.doctor_actions(report([P, c]))
    check(f"To do: {label} -> the v2.6.4 item, word for word",
          acts == ["the Claude app copy is 2.6.3, the latest is 2.6.4: in the Claude app, Customize -> Skills, upload "
                   f"{se.RELEASE_ASSET} (it reaches Claude Code at its next start)"], acts)
M = {"kind": "manual copy", "version": "2.6.2", "path": "m"}
acts = se.doctor_actions(report([P, M, app("2.6.3", "unknown"), app("2.6.4", "unknown")], latest=None))
check("To do offline, a manual copy 2.6.2 plus two accounts unknown: BOTH the 'different versions' item (naming the "
      "manual copy) and the accounts item (review 1: the second hid the first)",
      len(acts) == 2 and "different versions" in acts[0] and "manual copy 2.6.2" in acts[0]
      and "2 Claude app copies are on this computer" in acts[1], acts)
acts = se.doctor_actions(report([app("2.6.4", "unknown"), app("2.6.3", "unknown")], latest=None,
                                here="/c/clone/econometrics"))
check("To do offline, no plugin, two accounts unknown at different versions -> the accounts item",
      any("2 Claude app copies are on this computer" in a for a in acts), acts)
acts = se.doctor_actions(report([P, app("2.6.4", "unknown"), app("2.6.3", "unknown")], org=ME))
check("To do, account READ but two folder names it cannot place -> no 'could not be read' item; each copy as before "
      "(review 1: the item contradicted the line above it)",
      acts == ["the Claude app copy at s/unknown/2.6.3 is 2.6.3, the latest is 2.6.4: in the Claude app, Customize -> "
               f"Skills, upload {se.RELEASE_ASSET} (it reaches Claude Code at its next start)"], acts)
acts = se.doctor_actions(report([P, app("2.6.3", "unknown"), {"kind": se.APP_COPY, "version": None, "path": "z",
                                                              "account": "unknown"}]))
check("To do, accounts item (one copy behind) with an unreadable version: 'versions 2.6.3, unreadable', never 'None'",
      len(acts) == 1 and "versions 2.6.3, unreadable," in acts[0] and "None" not in acts[0], acts)
acts = se.doctor_actions(report([P, app("2.6.4", "signed in"), app("2.6.3", "other")], here="s/other/2.6.3"))
check("To do: the copy running now is the other account's (2.6.3) -> it is not silenced (review 1)",
      len(acts) == 1 and "2.6.3" in acts[0], acts)
acts = se.doctor_actions(report([app("2.6.3", "unknown"), {"kind": se.APP_COPY, "version": None, "path": "z",
                                                         "account": "unknown"}], latest=None, here="/c/clone/econometrics"))
check("To do offline, no reference, two accounts: 2.6.3 and an unreadable version are not 'different versions' -> []",
      acts == [], acts)
acts = se.doctor_actions(report([P, app("2.6.3", "signed in"), app("2.6.3", "unknown")]))
check("To do: a signed-in copy and one unknown folder name, both behind -> one item each, naming its folder, never two "
      "identical items (review 2)", len(acts) == 2 and acts[0].startswith("the Claude app copy at s/signed in/2.6.3 is 2.6.3")
      and acts[1].startswith("the Claude app copy at s/unknown/2.6.3 is 2.6.3") and acts[0] != acts[1], acts)

# ----------------------------------------------------------------------------------- 5. skill_listing
lt = Path(tempfile.mkdtemp())
lc = lt / ".claude"
for acct, n in ((f"{ME}_{ME_ID}", 3), (f"{OTHER}_{OTHER_ID}", 5)):   # each with its own copy of this skill
    for k in range(n):
        d = lc / "skills" / "synced" / acct / ("econometrics" if k == 0 else f"s{k}")
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text("x", encoding="utf-8")
(lc / "skills" / "mine").mkdir(parents=True)
(lc / "skills" / "mine" / "SKILL.md").write_text("x", encoding="utf-8")
folder = lt / "work"
folder.mkdir()
for org, want, label in ((ME, 3 + 1, "signed in to the 3-skill account"), (OTHER, 5 + 1, "signed in to the other"),
                         (None, 3 + 5 + 1, "account unknown")):
    lst = se.skill_listing(lc, run=runner(), which=CLAUDE, cwd=folder, org_id=org)
    check(f"listing: {label} -> {want} skills (synced skills of another account not counted; a manual one counted)",
          lst["checked"] and lst["skills"] == want, lst.get("skills"))
lm = Path(tempfile.mkdtemp()) / ".claude"
for acct, sub in ((f"{ME}_{ME_ID}", "other-skill"), (f"{OTHER}_{OTHER_ID}", "econometrics")):
    d = lm / "skills" / "synced" / acct / sub
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text("x", encoding="utf-8")
lst = se.skill_listing(lm, run=runner(), which=CLAUDE, cwd=folder, org_id=ME)
cps = se.skill_copies(lm, org_id=ME)
check("listing and copies agree: the signed-in folder has another skill but no copy of this one -> the other "
      "account's econometrics is 'unknown' AND counted (2 skills; review 2: the count left it out)",
      lst["skills"] == 2 and [c.get("account") for c in cps] == ["unknown"], (lst.get("skills"), cps))
lo = Path(tempfile.mkdtemp()) / ".claude"
for k in range(4):
    d = lo / "skills" / "synced" / f"{OTHER}_{OTHER_ID}" / f"s{k}"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text("x", encoding="utf-8")
lst = se.skill_listing(lo, run=runner(), which=CLAUDE, cwd=folder, org_id=ME)
check("listing: signed in, only another account's folder (4 skills) -> counted (4), as in skill_copies (review 1)",
      lst["checked"] and lst["skills"] == 4, lst.get("skills"))
pl = json.dumps([{"id": se.PLUGIN_ID, "enabled": True, "scope": "project", "projectPath": "", "version": "2.0",
                  "installPath": str(lt / "none")}])
lst = se.skill_listing(lc, run=runner(plugins=pl), which=CLAUDE, cwd=Path.cwd())
check("listing: a project install with projectPath '' is not this folder's install (no project_install, run from "
      "the current folder)", lst["checked"] and "project_install" not in lst, lst)
pl = json.dumps([{"id": se.PLUGIN_ID, "enabled": True, "scope": "project", "projectPath": str(folder), "version": "2.0",
                  "installPath": str(lt / "none")}])
lst = se.skill_listing(lc, run=runner(plugins=pl), which=CLAUDE, cwd=folder)
check("listing: the other side, projectPath = this folder -> its project install is reported (pinned at 2.0)",
      lst.get("project_install") == {"version": "2.0", "scope": "project"}, lst.get("project_install"))

# ----------------------------------------------------------------------------------- 6. doctor end to end
rep = se.doctor(offline=True, cdir=cdir, env={}, run=runner(AUTH), which=CLAUDE, os_name="posix")
check("doctor: the report carries the account (org_id and org_name) and is JSON-serializable",
      json.loads(json.dumps(rep))["account"] == {"org_id": ME, "org_name": "Org A"}, rep.get("account"))
check("doctor: the listing it reports counts the signed-in account's synced skill only (1, not 2)",
      rep["listing"].get("checked") and rep["listing"]["skills"] == 1, rep["listing"].get("skills"))
check("doctor offline (notebook case): no item about the other account's 2.6.3 copy",
      not any("2.6.3" in a for a in rep["actions"]), rep["actions"])
rep["latest"] = {"version": "2.6.4", "status": "ok"}
rep["actions"] = se.doctor_actions(rep)
check("doctor with latest 2.6.4 (notebook case): the v2.6.4 finding is gone",
      not any("Claude app copy is 2.6.3" in a for a in rep["actions"]), rep["actions"])
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    se.print_doctor(rep)
out = buf.getvalue()
check("doctor output: 'Claude Code signed in to: Org A (org <id>)'", f"Claude Code signed in to: Org A (org {ME})" in out,
      out[-600:])
lines = [ln for ln in out.splitlines() if ln.startswith("  copy: " + se.APP_COPY)]
check("doctor output: each synced copy says whose it is (the signed-in account's; another account's, nothing to do)",
      len(lines) == 2 and any("2.6.4, the signed-in account's copy" in ln for ln in lines)
      and any("2.6.3, another Claude account's copy" in ln and "nothing to do" in ln for ln in lines), lines)
rep2 = se.doctor(offline=True, cdir=cdir, env={}, run=runner(""), which=CLAUDE, os_name="posix")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    se.print_doctor(rep2)
check("doctor, auth status unreadable: account None, 'Claude Code account: not read', no owner on the copies",
      rep2["account"] == {"org_id": None, "org_name": None} and "Claude Code account: not read" in buf.getvalue()
      and "the signed-in account's copy" not in buf.getvalue(), rep2["account"])
check("doctor, auth status unreadable: the listing counts both accounts' synced skills (2), as before v2.6.5",
      rep2["listing"].get("checked") and rep2["listing"]["skills"] == 2, rep2["listing"].get("skills"))

calls = []


def counting(cmd, timeout):
    calls.append(cmd[1:3])
    return runner(AUTH)(cmd, timeout)


nos = Path(tempfile.mkdtemp()) / ".claude"
write_copy(nos / "plugins" / "cache" / "econometric-skills" / "econometrics" / "2.6.4", "2.6.4")
rep3 = se.doctor(offline=True, cdir=nos, env={}, run=counting, which=CLAUDE, os_name="posix")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    se.print_doctor(rep3)
check("doctor with no skills/synced folder: claude auth status is not run, and no account line is printed (review 1)",
      ["auth", "status"] not in calls and "Claude Code account" not in buf.getvalue()
      and "Claude Code signed in to" not in buf.getvalue(), calls)

calls.clear()
nsk = Path(tempfile.mkdtemp()) / ".claude"
write_copy(nsk / "skills" / "mine" / "econometrics", "2.6.4")      # a manual copy: skills/ but no skills/synced
se.doctor(offline=True, cdir=nsk, env={}, run=counting, which=CLAUDE, os_name="posix")
check("doctor with skills/ but no skills/synced: claude auth status is not run", ["auth", "status"] not in calls, calls)
_isdir = Path.is_dir


def denied(self):
    if self.name == "synced":
        raise PermissionError(13, "Permission denied")
    return _isdir(self)


with mock.patch.object(Path, "is_dir", denied):
    try:
        rp = se.doctor(offline=True, cdir=cdir, env={}, run=runner(AUTH), which=CLAUDE, os_name="posix")
        crashed = None
    except Exception as e:                     # noqa: BLE001
        rp, crashed = None, e
check("doctor: an OS error on skills/synced (permission) does not crash it; the account is then not read (review 2)",
      crashed is None and rp["account"]["org_id"] is None, repr(crashed))
oth = next(c["path"] for c in se.skill_copies(cdir, org_id=ME) if c.get("account") == "other")
with mock.patch.object(se, "SKILL_DIR", Path(oth)):
    rr = se.doctor(offline=True, cdir=cdir, env={}, run=runner(AUTH), which=CLAUDE, os_name="posix")
rr["latest"] = {"version": "2.6.4", "status": "ok"}
rr["actions"] = se.doctor_actions(rr)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    se.print_doctor(rr)
check("doctor run FROM the other account's 2.6.3 copy: that copy is not 'another account's' in the report, the printed "
      "list or the To do (review 2: 'nothing to do here' next to an update item)",
      [c.get("account") for c in rr["copies"] if c["path"] == oth] == ["unknown"]
      and "nothing to do here" not in buf.getvalue() and any("2.6.3" in a for a in rr["actions"]), rr["actions"])

# ----------------------------------------------------------------------------------- 7. rho on the dml_bounds lines
rng = np.random.default_rng(2605)
n = 400
X = rng.normal(size=(n, 2))
d = 0.5 * X[:, 0] + rng.normal(size=n)
y = 1.0 * d + 0.7 * X[:, 0] - 0.4 * X[:, 1] + rng.normal(size=n)
R = causal.dml_plr(pd.DataFrame(dict(y=y, d=d, x1=X[:, 0], x2=X[:, 1])), "y", "d", ["x1", "x2"],
                   learner_y=LinearRegression(), learner_d=LinearRegression(), n_rep=1)
BENCH = pd.DataFrame({"gain_y": [0.05] * 7, "gain_d": [0.05] * 7,
                      "rho": [0.9726123, 0.996, 1.0, -1.0, 0.9999996, -0.0, -0.9999996]},
                     index=["a", "b", "c", "e", "g", "h", "i"])


def printed(**k):
    buf = io.StringIO()
    with warnings.catch_warnings(), contextlib.redirect_stdout(buf):
        warnings.simplefilter("ignore")
        o = causal.dml_bounds(R, BENCH, kd=1, **k)
    return o, buf.getvalue().splitlines()


o, lines = printed()
want = {"a": "rho 0.972612:", "b": "rho 0.996:", "c": "rho 1:", "e": "rho -1:", "g": "rho 0.9999996:", "h": "rho 0:",
        "i": "rho -0.9999996:"}
for b, w in want.items():
    ln = [x for x in lines if x.startswith(f"  1x {b}:")]
    check(f"dml_bounds line of benchmark rho {BENCH.loc[b, 'rho']!r}: '{w}'", len(ln) == 1 and w in ln[0], ln)
check("dml_bounds: 0.996 is never printed as 'rho 1.00' (reads as the worst case); no line has 2-decimal rho",
      not any("rho 1.00" in x or "rho 0.97:" in x for x in lines), [x for x in lines if "rho" in x][:6])
o, lines = printed(rho=1 / 3)
txt = f"{1 / 3:g}"
head = [x for x in lines if x.startswith("  rho = ")]
rows = [x for x in lines if x.startswith("  1x ")]
rep_line = [x for x in lines if x.startswith("  In a report, state")]
check(f"dml_bounds rho = 1/3: header, every line and the report line all print '{txt}'",
      head == [f"  rho = {txt} for every line"] and len(rows) == 7 and all(f"rho {txt}:" in x for x in rows)
      and len(rep_line) == 1 and f"rho = {txt} next to the bounds" in rep_line[0], (head, rows[:1], rep_line))

o, lines = printed(rho=0.9999996)
head = [x for x in lines if x.startswith("  rho = ")]
rep_line = [x for x in lines if x.startswith("  In a report, state")]
check("dml_bounds rho = 0.9999996: the header and report line say 0.9999996 (not '1', the worst case), and no "
      "'(the worst case)'", head == ["  rho = 0.9999996 for every line"] and len(rep_line) == 1
      and "rho = 0.9999996 next to the bounds" in rep_line[0] and "worst case" not in rep_line[0], (head, rep_line))

# ----------------------------------------------------------------------------------- 8. .gitattributes
ga = (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
check(".gitattributes: '*.skill binary' (and the CSV rule kept)", "*.skill binary" in ga
      and "tests/data/*.csv text eol=lf" in ga, ga)
git = shutil.which("git")
if git and (ROOT / ".git").exists():
    r = subprocess.run([git, "check-attr", "binary", "text", "--", "dist/econometrics.skill"], cwd=ROOT,
                       capture_output=True, text=True)
    check("git check-attr: dist/econometrics.skill is binary and not text",
          "binary: set" in r.stdout and "text: unset" in r.stdout, r.stdout.strip())

shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree(lt, ignore_errors=True)
print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
