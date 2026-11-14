"""
v2.5.0.1: installation and updates, plug-and-play.

Origin: the setup of a second notebook (2026-11-06) found the plugin and the Claude-app copy nine versions
behind with nothing saying so, the marketplace without auto-update, two Pythons with the packages split
between them (py = 3.14 complete, python = 3.12 empty), about 300 skills in the listing, and optional
packages with no permitted install path (hmm_regimes told the user to run pip directly; select_bandwidth
switched to IK silently).

Checked here, on simulated machines (every probe of setup_env takes its inputs as arguments):
  1. --doctor: copies of the skill and their versions (the app copy synced to disk, .trash ignored),
     latest release (and offline), plugin auto-update (documented precedence), other interpreters,
     a crowded listing, and the single "To do" list built from all of it.
  2. --enable-autoupdate: writes only that key, keeps the rest, backs the file up, refuses broken JSON.
  3. Extras: every optional package has one, heavy ones say their size, require_extra(only=...),
     packages pip cannot install on this Python are skipped by --install.
  4. The modules: no bare pip command left; rd, regimes, pdf, ocr, web, text and llm stop with the
     setup_env line; select_bandwidth's IK fallback is reported as a warning of rdd.
  5. SKILL.md: section 7 asks once after --doctor; section 4 is unchanged.
Each boundary also asserts what must NOT happen (lesson of v2.4.7.2).
"""
from __future__ import annotations

import hashlib
import importlib
import io
import os
import json
import re
import subprocess
import sys
import tempfile
import urllib.error
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "econometrics"
sys.path.insert(0, str(SKILL / "scripts"))
import _console  # noqa: E402,F401
import setup_env as se  # noqa: E402

results = []


def check(name, cond, info=""):
    results.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{info}]" if info and not cond else ""))


class blocked:
    """Make modules look absent: find_spec returns None and import raises ImportError."""
    def __init__(self, *names):
        self.names = names

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.names:
            raise ImportError(f"blocked {name}")
        return None

    def __enter__(self):
        self.saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k.split(".")[0] in self.names}
        sys.meta_path.insert(0, self)
        self._fs = importlib.util.find_spec

        def fs(name, *a, **k):
            if name.split(".")[0] in self.names:
                return None
            return self._fs(name, *a, **k)
        importlib.util.find_spec = fs
        return self

    def __exit__(self, *a):
        sys.meta_path.remove(self)
        importlib.util.find_spec = self._fs
        sys.modules.update(self.saved)


def write_copy(folder: Path, version: str):
    (folder / "scripts").mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text("---\nname: econometrics\n---\n", encoding="utf-8")
    (folder / "scripts" / "reporting.py").write_text(f'SKILL_VERSION = "{version}"\n', encoding="utf-8")


def fake_opener(payload=None, exc=None, calls=None):
    class R(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def op(req, timeout=None):
        if calls is not None:
            calls.append(req.full_url)
        if exc:
            raise exc
        return R(payload if isinstance(payload, bytes) else json.dumps(payload).encode())
    return op


# ----------------------------------------------------------------------------------- 1. copies of the skill
tmp = Path(tempfile.mkdtemp())
cdir = tmp / ".claude"
plug = cdir / "plugins" / "cache" / "econometric-skills" / "econometrics" / "2.3.3"
write_copy(plug, "2.3.3")
write_copy(cdir / "plugins" / "marketplaces" / "econometric-skills" / "econometrics", "2.3.3")
write_copy(cdir / "skills" / "synced" / "acct" / "econometrics", "2.3.8")
write_copy(cdir / "skills" / ".trash" / "123-x" / "econometrics", "2.3.7")
(cdir / "plugins" / "installed_plugins.json").write_text(json.dumps(
    {"version": 2, "plugins": {"econometrics@econometric-skills": {"scope": "user", "installPath": str(plug),
                                                                   "version": "2.3.3"}}}), encoding="utf-8")
copies = se.skill_copies(cdir)
kinds = {c["kind"]: c["version"] for c in copies}
check("copies: plugin 2.3.3, marketplace checkout 2.3.3 and the Claude app copy synced to disk 2.3.8",
      kinds == {"plugin": "2.3.3", "marketplace checkout": "2.3.3", "Claude app (synced)": "2.3.8"}, kinds)
check("copies: the one in skills/.trash (removed from the app) is not reported",
      not any("2.3.7" == c["version"] or ".trash" in c["path"] for c in copies))
(cdir / "plugins" / "installed_plugins.json").write_text(json.dumps(
    {"version": 2, "plugins": {"econometrics@econometric-skills": [{"scope": "user", "installPath": str(plug),
                                                                    "version": "2.3.3"}]}}), encoding="utf-8")
check("copies: installed_plugins.json with one record per scope (a list) is read too",
      any(c["kind"] == "plugin" and c["version"] == "2.3.3" for c in se.skill_copies(cdir)))
check("copies: an empty Claude folder gives no copies and no error", se.skill_copies(tmp / "nothing") == [])
check("the version is read from SKILL_VERSION (the app repackages the file, so hashes differ)",
      se.read_skill_version(SKILL) == re.search(r'^SKILL_VERSION = "([^"]+)"',
                                                (SKILL / "scripts" / "reporting.py").read_text(encoding="utf-8"),
                                                re.M).group(1))

# ----------------------------------------------------------------------------------- 2. latest release
calls = []
v, st = se.latest_release(opener=fake_opener({"tag_name": "v2.5.1"}, calls=calls))
check("latest release: tag v2.5.1 read as 2.5.1 from the public releases/latest endpoint",
      v == "2.5.1" and st == "ok" and calls == [se.RELEASES_API], (v, st, calls))
v, st = se.latest_release(opener=fake_opener(exc=urllib.error.URLError("offline")))
check("latest release: no network gives 'not checked', not an error", v is None and st.startswith("not checked"), st)
v, st = se.latest_release(opener=fake_opener(b"<html>proxy</html>"))
check("latest release: a non-JSON reply (proxy page) gives 'not checked'", v is None and st.startswith("not checked"))
v, st = se.latest_release(opener=fake_opener({"message": "API rate limit exceeded"}))
check("latest release: a reply without a tag gives 'not checked', never a fake version", v is None, (v, st))
check("version order: 2.5.0.1 is newer than 2.5.0 and older than 2.5.1",
      se._vtuple("2.5.0.1") > se._vtuple("2.5.0") and se._vtuple("2.5.0.1") < se._vtuple("2.5.1"))

# ----------------------------------------------------------------------------------- 3. auto-update
c2 = tmp / "c2"
(c2 / "plugins").mkdir(parents=True)
au = se.marketplace_autoupdate(c2, env={})
check("auto-update: marketplace not registered -> registered False (no action about it)", not au["registered"])
(c2 / "plugins" / "known_marketplaces.json").write_text(json.dumps(
    {"econometric-skills": {"source": {"source": "github", "repo": se.REPO}}}), encoding="utf-8")
au = se.marketplace_autoupdate(c2, env={})
check("auto-update: registered without the key -> off by default (third-party marketplace), as on notebook 2",
      au["registered"] and not au["on"] and au["source"].startswith("default"), au)
(c2 / "plugins" / "known_marketplaces.json").write_text(json.dumps(
    {"econometric-skills": {"source": {}, "autoUpdate": True}}), encoding="utf-8")
au = se.marketplace_autoupdate(c2, env={})
check("auto-update: the /plugin toggle (known_marketplaces.json autoUpdate true) -> on", au["on"], au)
(c2 / "settings.json").write_text(json.dumps(
    {"extraKnownMarketplaces": {"econometric-skills": {"source": {}, "autoUpdate": False}}}), encoding="utf-8")
au = se.marketplace_autoupdate(c2, env={})
check("auto-update: a settings.json entry takes precedence over known_marketplaces.json (documented order)",
      not au["on"] and au["source"] == "settings.json", au)
(c2 / "settings.json").write_text(json.dumps(
    {"extraKnownMarketplaces": {"econometric-skills": {"source": {}, "autoUpdate": True}}}), encoding="utf-8")
au = se.marketplace_autoupdate(c2, env={"DISABLE_AUTOUPDATER": "1"})
check("auto-update: DISABLE_AUTOUPDATER=1 blocks it even when configured on",
      not au["on"] and au["configured"] and au["blocked_by"] == ["DISABLE_AUTOUPDATER"], au)
au = se.marketplace_autoupdate(c2, env={"DISABLE_AUTOUPDATER": "1", "FORCE_AUTOUPDATE_PLUGINS": "1"})
check("auto-update: FORCE_AUTOUPDATE_PLUGINS=1 lifts the block", au["on"] and not au["blocked_by"], au)
au = se.marketplace_autoupdate(c2, env={"DISABLE_AUTOUPDATER": "0"})
check("auto-update: a variable set to 0 does not block", au["on"], au)

# enable_autoupdate
c3 = tmp / "c3"
c3.mkdir()
before = {"enabledPlugins": {"x@y": True}, "theme": "dark", "extraKnownMarketplaces": {
    "econometric-skills": {"source": {"source": "github", "repo": se.REPO}}, "other": {"source": {}}}}
(c3 / "settings.json").write_text(json.dumps(before, indent=2), encoding="utf-8")
original_bytes = (c3 / "settings.json").read_bytes()      # as written on this OS (CRLF on Windows)
path, backup = se.enable_autoupdate(c3, now=0)
after = json.loads(path.read_text(encoding="utf-8"))
exp = json.loads(json.dumps(before))
exp["extraKnownMarketplaces"]["econometric-skills"]["autoUpdate"] = True
check("--enable-autoupdate: only autoUpdate true is added; every other key and marketplace is kept", after == exp)
check("--enable-autoupdate: the previous file is backed up byte for byte",
      backup is not None and backup.read_bytes() == original_bytes)
_p2, backup2 = se.enable_autoupdate(c3, now=0)
check("--enable-autoupdate: a second run in the same second keeps the first backup (new name)",
      backup2 is not None and backup2 != backup and backup.read_bytes() == original_bytes)
check("--enable-autoupdate: the result is read back as on", se.marketplace_autoupdate(c3, env={})["on"])
c4 = tmp / "c4"
c4.mkdir()
(c4 / "settings.json").write_text("{ broken", encoding="utf-8")
try:
    se.enable_autoupdate(c4)
    refused = False
except ValueError:
    refused = True
check("--enable-autoupdate: a settings.json that is not valid JSON is refused and left untouched",
      refused and (c4 / "settings.json").read_text(encoding="utf-8") == "{ broken"
      and not list(c4.glob("settings.json.bak-*")))
c5 = tmp / "c5"
try:
    se.enable_autoupdate(c5)
    refused = False
except ValueError as e:
    refused = "not registered" in str(e)
check("--enable-autoupdate: a marketplace Claude Code does not know is refused, nothing written",
      refused and not (c5 / "settings.json").exists())
(c5 / "plugins").mkdir(parents=True)
fork = {"source": "directory", "path": "/home/dev/econometric-skills"}
(c5 / "plugins" / "known_marketplaces.json").write_text(json.dumps({"econometric-skills": {"source": fork}}),
                                                       encoding="utf-8")
path, backup = se.enable_autoupdate(c5)
check("--enable-autoupdate: no settings.json yet -> created with the marketplace's own source (a local folder "
      "stays a local folder, never switched to GitHub)",
      backup is None and json.loads(path.read_text(encoding="utf-8"))["extraKnownMarketplaces"][
          "econometric-skills"] == {"source": fork, "autoUpdate": True})
for bad, what in (("[1, 2]", "a list"), ('{"extraKnownMarketplaces": null}', "null marketplaces"),
                  ('{"extraKnownMarketplaces": {"econometric-skills": 5}}', "a number as the entry"),
                  ("\ufeff" + json.dumps({"extraKnownMarketplaces": {"econometric-skills": {"source": fork}}}), None)):
    c6 = tmp / f"c6-{abs(hash(bad))}"
    (c6 / "plugins").mkdir(parents=True)
    (c6 / "plugins" / "known_marketplaces.json").write_text(json.dumps({"econometric-skills": {"source": fork}}),
                                                           encoding="utf-8")
    (c6 / "settings.json").write_text(bad, encoding="utf-8")
    try:
        se.enable_autoupdate(c6)
        res = "written"
    except ValueError:
        res = "refused"
    if what:
        check(f"--enable-autoupdate: settings.json with {what} is refused, untouched, no backup",
              res == "refused" and (c6 / "settings.json").read_text(encoding="utf-8") == bad
              and not list(c6.glob("settings.json.bak-*")))
    else:
        check("--enable-autoupdate: a settings.json saved with a UTF-8 BOM is read, not refused",
              res == "written" and se.marketplace_autoupdate(c6, env={})["on"])

# ----------------------------------------------------------------------------------- 4. interpreters
PY0P = (" -V:3.14 *        C:\\Users\\fabia\\AppData\\Local\\Python\\pythoncore-3.14-64\\python.exe\n"
        " -V:3.12          C:\\Users\\fabia\\AppData\\Local\\Programs\\Python\\Python312\\python.exe\n")


def fake_run_py(cmd, timeout):
    return 0, PY0P


cands = se.python_candidates(os_name="nt", run=fake_run_py, which=lambda n: "C:\\Windows\\py.exe" if n == "py" else None)
check("interpreters: 'py -0p' of notebook 2 gives both Pythons by tag, started through the launcher's full path",
      cands == [["C:\\Windows\\py.exe", "-V:3.14"], ["C:\\Windows\\py.exe", "-V:3.12"]], cands)
cands = se.python_candidates(os_name="nt", run=lambda c, t: (0, " -3.12-64        C:\\Users\\Jo\ufffdo\\py312\\python.exe\n"),
                             which=lambda n: "C:\\py.exe")
check("interpreters: the older 'py -0p' format, and a path mangled in decoding does not matter (the tag is used)",
      cands == [["C:\\py.exe", "-V:3.12-64"]], cands)
cands = se.python_candidates(os_name="posix", run=None,
                             which=lambda n: {"python3.12": "/usr/bin/python3.12",
                                              "python3.13": "/usr/bin/python3.13"}.get(n))
check("interpreters: python3.X on the PATH off Windows", cands == [["/usr/bin/python3.12"], ["/usr/bin/python3.13"]])
real = se.probe_interpreter(sys.executable)
check("interpreters: probing a real interpreter returns its version and missing core packages",
      real is not None and real["version"].startswith(f"{sys.version_info[0]}.{sys.version_info[1]}")
      and isinstance(real["missing_core"], list), real)
check("interpreters: an interpreter that does not start is skipped, not an error",
      se.probe_interpreter(str(tmp / "no-python.exe")) is None)
old_env = json.dumps({"exe": "x", "v": [3, 11, 2], "ver": {"numpy": "1.26.4", "pandas": "2.0.3", "scipy": "1.14.1",
                      "statsmodels": "0.14.4", "linearmodels": "6.1", "arch": "7.2.0", "sklearn": "1.5.2",
                      "tabulate": "0.9.0"}})
info = se.probe_interpreter("other", run=lambda cmd, timeout: (0, old_env))
check("interpreters: packages present but below the minimums count as missing (numpy 1.26, pandas 2.0)",
      info["missing_core"] == ["numpy>=2.1", "pandas>=2.2"], info)
check("interpreters: a probe whose output cannot be decoded or parsed is skipped",
      se.probe_interpreter("other", run=lambda cmd, timeout: (0, "\ufffd garbage")) is None
      and se.python_candidates(os_name="nt", run=lambda c, t: (_ for _ in ()).throw(UnicodeDecodeError(
          "cp1252", b"\x81", 0, 1, "x")), which=lambda n: "py") == [])
work = tmp / "analysis"
work.mkdir()
(work / "json.py").write_text("raise SystemExit('shadowed json')\n", encoding="utf-8")
cwd0 = os.getcwd()
os.chdir(work)
try:
    shadow = se.probe_interpreter(sys.executable)
finally:
    os.chdir(cwd0)
check("interpreters: a json.py in the user's folder does not shadow the probe (it runs from the temp folder)",
      shadow is not None, shadow)


# ----------------------------------------------------------------------------------- 5. the To do list
def base_report(**over):
    rep = {"python": {"exe": "C:\\Python312\\python.exe", "version": "3.12.10", "supported": True,
                      "run_with": "py -3.12"},
           "core_missing": [], "interpreters": [],
           "skill": {"dir": "p", "version": "2.5.0.1"}, "latest": {"version": "2.5.0.1", "status": "ok"},
           "copies": [{"kind": "plugin", "version": "2.5.0.1", "path": "p"}],
           "autoupdate": {"registered": True, "on": True, "configured": True, "source": "settings.json",
                          "blocked_by": []},
           "listing": {"checked": True, "skills": 12, "enabled_plugins": 2, "crowded": False}}
    rep.update(over)
    return rep


check("To do: an up-to-date, complete setup gives an empty list", se.doctor_actions(base_report()) == [])
acts = se.doctor_actions(base_report(latest={"version": "2.5.1", "status": "ok"}))
check("To do: a newer release -> the plugin update commands and the restart",
      len(acts) == 1 and "2.5.0.1 to 2.5.1" in acts[0] and "claude plugin update econometrics@econometric-skills"
      in acts[0] and "restart" in acts[0], acts)
acts = se.doctor_actions(base_report(latest={"version": "2.5.1", "status": "ok"}, skill={"dir": "s", "version": "2.5.0.1"},
                                     copies=[{"kind": "Claude app (synced)", "version": "2.5.0.1", "path": "s"}]))
check("To do: only the Claude app copy (no plugin) is behind -> upload in Customize -> Skills, no plugin commands",
      len(acts) == 1 and "Customize -> Skills" in acts[0] and "claude plugin" not in acts[0], acts)
acts = se.doctor_actions(base_report(latest={"version": "2.5.1", "status": "ok"},
                                     skill={"dir": "m", "version": "2.5.1"},
                                     copies=[{"kind": "plugin", "version": "2.5.0.1", "path": "p"},
                                             {"kind": "marketplace checkout", "version": "2.5.1", "path": "m"}]))
check("To do: run from an updated marketplace checkout, the installed plugin behind -> plugin update item",
      any("update the plugin from 2.5.0.1 to 2.5.1" in a for a in acts), acts)
acts = se.doctor_actions(base_report(latest={"version": "2.5.1", "status": "ok"}, skill={"dir": "clone", "version": "2.5.0"},
                                     copies=[{"kind": "plugin", "version": "2.5.1", "path": "p"}]))
check("To do: a running clone behind the release -> git pull item; the up-to-date plugin -> none",
      len(acts) == 1 and "git pull" in acts[0] and "copy in use" in acts[0], acts)
cache_old = "/home/u/.claude/plugins/cache/econometric-skills/econometrics/2.5.0.1"
acts = se.doctor_actions(base_report(latest={"version": "2.5.1", "status": "ok"}, skill={"dir": cache_old, "version": "2.5.0.1"},
                                     copies=[{"kind": "plugin", "version": "2.5.1",
                                              "path": "/home/u/.claude/plugins/cache/econometric-skills/econometrics/2.5.1"}]))
check("To do: auto-update already installed 2.5.1 but the session runs the 2.5.0.1 cache folder -> one item: restart; "
      "never 'replace' or 'git pull' a plugin cache folder", len(acts) == 1 and "restart Claude Code" in acts[0]
      and "git pull" not in acts[0] and "replace" not in acts[0], acts)
check("To do: an older or equal 'latest' never asks for an update",
      se.doctor_actions(base_report(latest={"version": "2.5.0", "status": "ok"})) == []
      and se.doctor_actions(base_report(latest={"version": None, "status": "not checked"})) == [])
acts = se.doctor_actions(base_report(copies=[{"kind": "plugin", "version": "2.5.0.1", "path": "p"},
                                             {"kind": "Claude app (synced)", "version": "2.3.8", "path": "s"}]))
check("To do: the Claude app copy behind the release -> one item (Customize -> Skills), not a second "
      "'different versions' item", len(acts) == 1 and "2.3.8" in acts[0] and "Customize -> Skills" in acts[0], acts)
acts = se.doctor_actions(base_report(latest={"version": None, "status": "not checked (--offline)"},
                                     copies=[{"kind": "plugin", "version": "2.5.0.1", "path": "p"},
                                             {"kind": "Claude app (synced)", "version": "2.3.8", "path": "s"}]))
check("To do: offline, copies at different versions -> one item naming both versions",
      len(acts) == 1 and "plugin 2.5.0.1" in acts[0] and "2.3.8" in acts[0] and "Customize -> Skills" in acts[0], acts)
acts = se.doctor_actions(base_report(copies=[{"kind": "plugin", "version": "2.5.0.1", "path": "p"},
                                             {"kind": "marketplace checkout", "version": "2.5.1", "path": "m"}]))
check("To do: the marketplace checkout (source of the next update) is not a conflicting copy", acts == [], acts)
acts = se.doctor_actions(base_report(autoupdate={"registered": True, "on": False, "configured": False,
                                                 "source": "default", "blocked_by": []}))
check("To do: auto-update off -> offer both documented ways, with the user's permission",
      len(acts) == 1 and "/plugin -> Marketplaces -> econometric-skills -> Enable auto-update" in acts[0]
      and "--enable-autoupdate" in acts[0] and "permission" in acts[0], acts)
acts = se.doctor_actions(base_report(autoupdate={"registered": False, "on": False, "configured": False,
                                                 "source": "default", "blocked_by": []}))
check("To do: no marketplace registered (app or manual copy only) -> no auto-update item", acts == [], acts)
acts = se.doctor_actions(base_report(autoupdate={"registered": True, "on": False, "configured": True,
                                                 "source": "settings.json", "blocked_by": ["DISABLE_UPDATES"]}))
check("To do: auto-update blocked by an environment variable -> says why and how to update by hand, "
      "offers no toggle that cannot work", len(acts) == 1 and "DISABLE_UPDATES" in acts[0]
      and "--enable-autoupdate" not in acts[0] and "claude plugin update" in acts[0], acts)
for n, exp_ in ((se.CROWDED_SKILLS - 1, 0), (se.CROWDED_SKILLS, 1)):
    acts = se.doctor_actions(base_report(listing={"checked": True, "skills": n, "enabled_plugins": 9,
                                                  "crowded": n >= se.CROWDED_SKILLS}))
    check(f"To do: {n} skills in the listing -> {exp_} clean-folder item (threshold {se.CROWDED_SKILLS})",
          len(acts) == exp_ and (exp_ == 0 or "workspace.py" in acts[0]), acts)
check("To do: listing not checked (no claude CLI) -> no clean-folder item",
      se.doctor_actions(base_report(listing={"checked": False, "reason": "x"})) == [])
miss = ["numpy>=2.1", "pandas>=2.2"]
acts = se.doctor_actions(base_report(core_missing=miss, interpreters=[
    {"exe": "C:\\Py314\\python.exe", "version": "3.14.0", "missing_core": [], "supported": True},
    {"exe": "C:\\Py313\\python.exe", "version": "3.13.1", "missing_core": [], "supported": True}]))
check("To do: this Python lacks the packages but another has all of them -> use the newest complete one, "
      "do not install again", len(acts) == 1 and "C:\\Py314\\python.exe" in acts[0]
      and "instead of installing" in acts[0] and "--install" not in acts[0], acts)
acts = se.doctor_actions(base_report(core_missing=miss, interpreters=[
    {"exe": "C:\\Py311\\python.exe", "version": "3.11.0", "missing_core": ["numpy"], "supported": True}]))
check("To do: no complete interpreter -> install the core packages with setup_env",
      len(acts) == 1 and "numpy>=2.1" in acts[0] and "--install" in acts[0], acts)
acts = se.doctor_actions(base_report(python={"exe": "py39", "version": "3.9.7", "supported": False,
                                             "run_with": "python"}, core_missing=miss), os_name="nt")
check("To do: Python too old and no other -> the winget command to install a current Python",
      len(acts) == 1 and "winget install --id Python.Python.3.13 -e" in acts[0], acts)
check("python install hints: brew on macOS, the package manager on Linux",
      "brew install" in se.python_install_hint("posix", "darwin")
      and "apt" in se.python_install_hint("posix", "linux"))

# notebook 2 as found on 2026-11-06, run with the 3.12 that 'python' starts
nb2 = base_report(
    core_missing=[f"{d}>={m}" for d, m in se.CORE.values()],
    interpreters=[{"exe": "C:\\Users\\fabia\\AppData\\Local\\Python\\pythoncore-3.14-64\\python.exe",
                   "version": "3.14.0", "missing_core": [], "supported": True}],
    skill={"dir": "cache/2.5.0.1", "version": "2.5.0.1"}, latest={"version": "2.5.0.1", "status": "ok"},
    copies=[{"kind": "plugin", "version": "2.5.0.1", "path": "p"},
            {"kind": "marketplace checkout", "version": "2.5.0.1", "path": "m"},
            {"kind": "Claude app (synced)", "version": "2.3.8", "path": "s"}],
    autoupdate={"registered": True, "on": False, "configured": False, "source": "default", "blocked_by": []},
    listing={"checked": True, "skills": 296, "enabled_plugins": 65, "crowded": True})
acts = se.doctor_actions(nb2)
check("notebook 2 after the update: one list with four items (use 3.14, app copy, auto-update, clean folder), "
      "and no package install", len(acts) == 4 and "pythoncore-3.14" in acts[0] and "2.3.8" in acts[1]
      and "auto-update" in acts[2] and "workspace.py" in acts[3]
      and not any("--install" in a for a in acts), acts)


# ----------------------------------------------------------------------------------- 6. doctor end to end
def fake_run_listing(n_skills):
    pdir = tmp / f"plugin{n_skills}"
    for k in range(n_skills):
        (pdir / f"s{k}").mkdir(parents=True, exist_ok=True)
        (pdir / f"s{k}" / "SKILL.md").write_text("x", encoding="utf-8")

    def run(cmd, timeout):
        if cmd[1:4] == ["plugin", "list", "--json"]:
            return 0, json.dumps([{"id": "econometrics@econometric-skills", "enabled": True,
                                   "installPath": str(tmp / "none")},
                                  {"id": "big@m", "enabled": True, "installPath": str(pdir)},
                                  {"id": "off@m", "enabled": False, "installPath": str(pdir)}])
        return 0, ""
    return run


lst = se.skill_listing(tmp / "empty", run=fake_run_listing(60), which=lambda n: "/bin/claude")
check("listing: only enabled plugins count (60 skills, the disabled copy not counted), crowded",
      lst["checked"] and lst["skills"] == 60 and lst["enabled_plugins"] == 2 and lst["crowded"]
      and lst["others_enabled"] == ["big@m"], lst)
lst = se.skill_listing(tmp / "empty", run=None, which=lambda n: None)
check("listing: without the claude CLI it is reported as not checked", not lst["checked"] and "PATH" in lst["reason"])
calls = []
rep = se.doctor(offline=True, cdir=cdir, env={}, opener=fake_opener({"tag_name": "v9.9"}, calls=calls),
                run=fake_run_listing(3), which=lambda n: "/bin/claude" if n == "claude" else None, os_name="posix")
check("doctor --offline: no request is made and the release is 'not checked (--offline)'",
      calls == [] and rep["latest"]["status"] == "not checked (--offline)")
check("doctor: the report is JSON-serializable (--json) and names every optional extra",
      json.loads(json.dumps(rep))["extras"].keys() == se.OPTIONAL.keys())
check("doctor: the app copy at another version becomes a To do item",
      any("Claude app (synced) 2.3.8" in a for a in rep["actions"]), rep["actions"])
rep = se.doctor(offline=False, cdir=cdir, env={}, opener=fake_opener({"tag_name": "v99.0"}),
                run=fake_run_listing(3), which=lambda n: None, os_name="posix")
check("doctor: a newer release becomes a To do item", any("to 99.0" in a for a in rep["actions"]), rep["actions"])
buf = io.StringIO()
with redirect_stdout(buf):
    se.print_doctor(rep)
out = buf.getvalue()
check("doctor output: 'Run scripts with:' and a numbered To do list that says to ask once",
      "Run scripts with: " in out and "ask the user once" in out and "\n  1. " in out, out[-300:])
r = subprocess.run([sys.executable, str(SKILL / "scripts" / "setup_env.py"), "--doctor", "--offline", "--json"],
                   capture_output=True, text=True, timeout=300)
try:
    j = json.loads(r.stdout)
    ok = "actions" in j and j["latest"]["status"] == "not checked (--offline)"
except ValueError:
    ok = False
check("command line: --doctor --offline --json prints valid JSON", ok, r.stdout[:200] + r.stderr[-200:])

bad = tmp / "bad"
(bad / "plugins").mkdir(parents=True)
(bad / "plugins" / "installed_plugins.json").write_text('{"plugins": {"econometrics@econometric-skills": '
                                                       '{"installPath": 5}}}', encoding="utf-8")
(bad / "plugins" / "known_marketplaces.json").write_text("[1]", encoding="utf-8")
(bad / "settings.json").write_text('{"extraKnownMarketplaces": [1]}', encoding="utf-8")
try:
    rb = se.doctor(offline=True, cdir=bad, env={}, run=lambda c, t: (0, "[1]"), which=lambda n: "/bin/claude",
                   os_name="posix")
    ok = isinstance(rb["actions"], list) and rb["copies"] == [] and not rb["autoupdate"]["registered"]
except Exception as e:
    ok, rb = False, repr(e)
check("doctor: malformed installed_plugins/known_marketplaces/settings and a strange CLI reply never crash it",
      ok, rb if not ok else "")
snap = {p: p.stat().st_mtime_ns for p in cdir.rglob("*")}
cmds = []


def rec_run(cmd, timeout):
    cmds.append(" ".join(map(str, cmd)))
    return 0, "[]"


se.doctor(offline=True, cdir=cdir, env={}, run=rec_run, which=lambda n: "/bin/" + n, os_name="posix")
check("doctor writes nothing in the Claude folder and never calls pip",
      {p: p.stat().st_mtime_ns for p in cdir.rglob("*")} == snap and not any("pip" in c for c in cmds), cmds)
uni = tmp / "\u0141\u00f3d\u017a" / ".claude"
write_copy(uni / "skills" / "econometrics", "2.3.8")
envc = dict(os.environ, PYTHONIOENCODING="cp1252", CLAUDE_CONFIG_DIR=str(uni))
outs = []
for extra in ([], ["--json"]):
    r = subprocess.run([sys.executable, str(SKILL / "scripts" / "setup_env.py"), "--doctor", "--offline", *extra],
                       capture_output=True, env=envc, timeout=300)
    outs.append((r.returncode, r.stdout, r.stderr))
check("doctor on a cp1252 console with a user path outside the code page (\u0141): no traceback, the copy is listed",
      all(b"Traceback" not in e and rc in (0, 1) for rc, _o, e in outs) and b"manual copy 2.3.8" in outs[0][1]
      and json.loads(outs[1][1].decode("ascii"))["copies"][0]["version"] == "2.3.8",
      [(rc, e[-200:]) for rc, _o, e in outs])
bad2 = tmp / "bad2"
(bad2 / "plugins").mkdir(parents=True)
(bad2 / "plugins" / "installed_plugins.json").write_text(json.dumps(
    {"plugins": {"econometrics@econometric-skills": {"installPath": str(tmp / "nowhere"), "version": 2}}}),
    encoding="utf-8")
try:
    rb2 = se.doctor(offline=False, cdir=bad2, env={}, opener=fake_opener({"tag_name": "v2.6"}), run=lambda c, t: (0, "[]"),
                    which=lambda n: None, os_name="posix")
    ok = [c["version"] for c in rb2["copies"]] == [None] and isinstance(rb2["actions"], list)
except Exception as e:
    ok, rb2 = False, repr(e)
check("doctor: a non-string version in installed_plugins.json with a release online does not crash it", ok, rb2)
check("versions: a commit SHA or a number is not a version (no silent 'up to date')",
      se._version_or_none("7f3a9c21b0") is None and se._version_or_none(2) is None
      and se._version_or_none("2.5.0.1") == "2.5.0.1"
      and se.latest_release(opener=fake_opener({"tag_name": "7f3a9c2"}))[0] is None)
if os.name == "posix":
    c7 = tmp / "c7"
    (c7 / "plugins").mkdir(parents=True)
    (c7 / "plugins" / "known_marketplaces.json").write_text(json.dumps({"econometric-skills": {"source": fork}}),
                                                           encoding="utf-8")
    dot = tmp / "dotfiles"
    dot.mkdir()
    target = dot / "claude-settings.json"
    target.write_text(json.dumps({"env": {"TOKEN": "secret"}}), encoding="utf-8")
    os.chmod(target, 0o600)
    (c7 / "settings.json").symlink_to(target)
    se.enable_autoupdate(c7, now=0)
    check("--enable-autoupdate: a symlinked settings.json stays a symlink, its target gets the change, mode 600 kept",
          (c7 / "settings.json").is_symlink() and json.loads(target.read_text(encoding="utf-8"))[
              "extraKnownMarketplaces"]["econometric-skills"]["autoUpdate"] is True
          and (target.stat().st_mode & 0o777) == 0o600 and not list(dot.glob(".*.tmp-*")))
    import time as _t
    t0 = _t.time()
    code_, out_ = se._run(["sh", "-c", "echo '[1]'; sleep 30 & sleep 30"], 2)
    check("_run: a command whose grandchild keeps the output open returns at the timeout, not after it",
          code_ == -1 and _t.time() - t0 < 10, (code_, round(_t.time() - t0, 1)))
    check("_run: normal output is returned and decoded with replacement",
          se._run(["sh", "-c", "printf 'ok\\377'"], 20) == (0, "ok\ufffd"))
if os.name == "nt":
    wb = tmp / "fake bin"
    wb.mkdir()
    (wb / "claude.cmd").write_text('@echo [{"id": "econometrics@econometric-skills", "enabled": true}]\r\n',
                                   encoding="ascii")
    lst_w = se.skill_listing(tmp / "empty", which=lambda n: str(wb / "claude.cmd"))
    check("Windows: a claude.cmd wrapper in a folder with a space is run and parsed", lst_w.get("checked")
          and lst_w.get("enabled_plugins") == 1, lst_w)
    (wb / "hang.cmd").write_text("@ping -n 30 127.0.0.1 >nul\r\n", encoding="ascii")
    import time as _t
    t0 = _t.time()
    code_, _o = se._run([str(wb / "hang.cmd")], 3)
    check("Windows: a hanging .cmd wrapper returns at the timeout (process tree killed)",
          code_ == -1 and _t.time() - t0 < 15, round(_t.time() - t0, 1))
r = subprocess.run([sys.executable, str(SKILL / "scripts" / "setup_env.py"), "--json"], capture_output=True, text=True)
check("--json without --doctor is an argument error, not silently ignored", r.returncode == 2 and "--doctor" in r.stderr)

# ----------------------------------------------------------------------------------- 7. extras
check("every extra has a use text, heavy and untested extras exist",
      set(se.OPTIONAL_USE) == set(se.OPTIONAL) and set(se.HEAVY_MB) <= set(se.OPTIONAL)
      and set(se.UNTESTED_MINIMUM) <= set(se.HEAVY_MB))
optional_by_case = ["rdrobust", "hmmlearn", "ruptures", "pydynpd", "econml", "transformers", "torch", "python-bcb",
                    "sidrapy", "wbgapi", "fredapi", "yfinance", "pymupdf", "easyocr", "requests",
                    "sentence-transformers"]
dists = {d for pk in se.OPTIONAL.values() for d, _m in pk.values()}
check("every optional package the skill uses or recommends has an extra (no permitted path was missing)",
      set(optional_by_case) <= dists, sorted(set(optional_by_case) - dists))
check("heavy extras: the user's list (torch, easyocr, econml) plus pdf and text, each with a size",
      {"llm", "ocr", "ml", "text", "pdf"} == set(se.HEAVY_MB) and all(v >= 50 for v in se.HEAVY_MB.values()))


def err(fn):
    try:
        fn()
    except ImportError as e:
        return str(e)
    return ""


with blocked("easyocr"):
    msg = err(lambda: se.require_extra("ocr", "OCR"))
check("require_extra (heavy): the install line, the size and 'ask ... on its own'; no bare pip",
      "--install --extra ocr" in msg and "205 MB" in msg and "on its own" in msg and "permission" in msg
      and "pip install" not in msg, msg)
with blocked("openpyxl"):
    msg = err(lambda: se.require_extra("excel", "Writing"))
check("require_extra (light): no size and no separate question", "--extra excel" in msg and "MB" not in msg, msg)
with blocked("ruptures"):
    only_ok = err(lambda: se.require_extra("regimes", "hmm_regimes", only=("hmmlearn",))) == "" \
        if se._installed_version("hmmlearn", "hmmlearn") else None
    whole = err(lambda: se.require_extra("regimes", "hmm_regimes"))
check("require_extra(only=hmmlearn): a missing ruptures does not stop hmm_regimes (without only, it would)",
      only_ok is True and "ruptures" in whole, (only_ok, whole[:80]))
check("caveats: ruptures cannot be installed on Python 3.14 (skipped by --install), it can on 3.13",
      se.not_installable(["ruptures>=1.1"], version=(3, 14)) == ["ruptures>=1.1"]
      and se.not_installable(["ruptures>=1.1"], version=(3, 13)) == [])
check("caveats: hmmlearn on 3.14 is a warning on every platform (no wheel: needs a compiler), not skipped; "
      "nothing on 3.13", se.not_installable(["hmmlearn>=0.3.2"], version=(3, 14), os_name="nt") == []
      and all("compiler" in " ".join(se.extra_caveats("regimes", version=(3, 14), os_name=o)) for o in ("nt", "posix"))
      and not any("hmmlearn" in c for c in se.extra_caveats("regimes", version=(3, 13), os_name="nt")))
installed = []
orig_install, orig_missing = se.install, se.missing_packages
se.install = lambda reqs: installed.append(list(reqs)) or True
se.missing_packages = lambda extras=(), core=True: (["ruptures>=1.1"] if extras else [])
try:
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = se.main(["--install", "--extra", "regimes"])
finally:
    se.install, se.missing_packages = orig_install, orig_missing
if sys.version_info[:2] >= (3, 14):
    check("--install on 3.14: ruptures is skipped with the reason, pip is not called, exit 0",
          installed == [] and "Skipping ruptures" in buf.getvalue() and code == 0, (installed, buf.getvalue()[:200]))
else:
    check("--install below 3.14: ruptures goes to pip as usual", installed == [["ruptures>=1.1"]], installed)

tested = [(n, imp, dist, m) for n, pk in se.OPTIONAL.items() if n not in se.UNTESTED_MINIMUM
          for imp, (dist, m) in pk.items()]
bad_x = []
for n, imp, dist, m in tested:
    if se.not_installable([f"{dist}>={m}"]):
        continue
    try:
        importlib.import_module(imp)
        v = se._installed_version(imp, dist)
        if v is None or se._vtuple(v) < se._vtuple(m):
            bad_x.append(f"{dist} {v} < {m}")
    except Exception as e:
        bad_x.append(f"{dist}: {type(e).__name__}: {str(e)[:60]}")
check("every extra CI installs imports at or above its minimum here (in the minimum job: exactly the minimum)",
      not bad_x, bad_x)

# ----------------------------------------------------------------------------------- 8. the modules
scripts = sorted((SKILL / "scripts").glob("*.py"))
bare = [f"{p.name}:{i}" for p in scripts if p.name != "setup_env.py"
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1) if "pip install" in line]
refs = [f"{p.name}:{i}" for p in (SKILL / "references").glob("*.md")
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1) if "pip install" in line]
check("no module or reference tells the user to run pip directly (SKILL.md section 7)", not bare and not refs,
      bare + refs)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import causal  # noqa: E402
rng = np.random.default_rng(7)
xr = rng.uniform(-1, 1, 800)
yr = 0.5 * xr + 1.0 * (xr >= 0) + rng.normal(0, 0.3, 800)
dfr = pd.DataFrame({"y": yr, "x": xr})
with blocked("rdrobust"):
    h, meth = causal.select_bandwidth(yr, xr, 0.0)
    out = causal.rdd(dfr, "y", "x", 0.0, show=False)
check("select_bandwidth without rdrobust: IK, and the method names the 'rd' extra",
      meth.startswith("Imbens-Kalyanaraman (rdrobust is not installed") and "--extra rd" in meth and h > 0, meth)
check("rdd without rdrobust: the IK fallback is a warning to relay (it was silent)",
      len(out["warnings"]) == 1 and "IK" in out["warnings"][0] and out["bandwidth_method"] == meth, out["warnings"])
if se._installed_version("rdrobust", "rdrobust"):
    out2 = causal.rdd(dfr, "y", "x", 0.0, show=False)
    check("rdd with rdrobust: mserd bandwidth and no warning", out2["bandwidth_method"] == "rdrobust mserd"
          and out2["warnings"] == [], out2["bandwidth_method"])
    import rdrobust as _rr
    real_bw = _rr.rdbwselect
    _rr.rdbwselect = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("singular"))
    try:
        _h, meth3 = causal.select_bandwidth(yr, xr, 0.0)
    finally:
        _rr.rdbwselect = real_bw
    check("select_bandwidth: an rdrobust failure is reported as such, not as 'not installed'",
          "rdrobust failed: RuntimeError: singular" in meth3, meth3)
else:
    check("rdrobust is installed in the test environment (requirements-dev.txt)", False)
check("rdd with a bandwidth given: no warning", causal.rdd(dfr, "y", "x", 0.0, h=0.4, show=False)["warnings"] == [])

import regimes  # noqa: E402
import data_prep  # noqa: E402
import text_data  # noqa: E402
with blocked("hmmlearn"):
    msg = err(lambda: regimes.hmm_regimes(pd.DataFrame({"r": rng.normal(size=50)}), show=False))
check("hmm_regimes without hmmlearn: the setup_env line (and markov_switching), no bare pip",
      "--extra regimes" in msg and "markov_switching" in msg and "pip install" not in msg, msg)
with blocked("pymupdf", "fitz"):
    msg = err(lambda: data_prep.pdf_text(tmp / "x.pdf"))
check("pdf_text without PyMuPDF: the 'pdf' extra with its size", "--extra pdf" in msg and "81 MB" in msg, msg)
if se._installed_version("pymupdf", "pymupdf"):
    import pymupdf
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Selic 10,75% ao ano")
    doc.new_page()                                   # no text layer: a 'scanned' page for the OCR path
    doc.save(str(tmp / "two.pdf"))
    doc.close()
    pages = data_prep.pdf_text(tmp / "two.pdf")
    check("pdf_text at the 'pdf' minimum or later: the text layer is read back", "Selic 10,75%" in pages[0]
          and len(pages) == 2, pages)
    one = pymupdf.open()
    one.new_page().insert_text((72, 72), "texto com camada de texto suficiente para nao precisar de OCR")
    one.save(str(tmp / "one.pdf"))
    one.close()
    with blocked("easyocr"):
        no_need = err(lambda: data_prep.pdf_text(tmp / "one.pdf", ocr=True))
        msg = err(lambda: data_prep.pdf_text(tmp / "two.pdf", ocr=True))
    check("pdf_text(ocr=True): easyocr is not required when every page has text", no_need == "", no_need)
    check("pdf_text(ocr=True) on a page without text and no easyocr: the 'ocr' extra, asked on its own",
          "--extra ocr" in msg and "on its own" in msg, msg)
else:
    check("pymupdf is installed in the test environment (requirements-dev.txt)", False)
with blocked("requests"):
    msg = err(lambda: data_prep.polite_get("https://example.org"))
check("polite_get without requests: the 'web' extra", "--extra web" in msg, msg)
with blocked("sentence_transformers"):
    msg = err(lambda: text_data.embed(["a"]))
check("embed without sentence-transformers: the 'text' extra with its size", "--extra text" in msg and "MB" in msg,
      msg)
with blocked("transformers", "torch"):
    msg = err(lambda: text_data.hf_backend("some/model"))
check("hf_backend without transformers/torch: the 'llm' extra with its size", "--extra llm" in msg and "MB" in msg,
      msg)

# ----------------------------------------------------------------------------------- 9. SKILL.md and the agent
skill_md = (SKILL / "SKILL.md").read_text(encoding="utf-8")
sec7 = skill_md[skill_md.index("## 7. Environment"):skill_md.index("## 8.")]
check("SKILL.md section 7: --doctor once per machine, one question, nothing changed before the answer",
      "setup_env.py --doctor" in sec7 and "Ask the user once" in sec7 and "Never install, update or change settings "
      "before that answer" in sec7)
check("SKILL.md section 7: auto-update only on a yes; heavy extras in a question of their own with the size",
      "--enable-autoupdate" in sec7 and "only on a yes" in sec7 and "question of their own" in sec7
      and "download size" in sec7)
check("SKILL.md section 7: no Python at all -> offer the system installer in the question",
      "winget install --id Python.Python.3.13 -e" in sec7 and "brew install" in sec7)
check("SKILL.md section 7: when to run it in each mode (Consultant, Reviewer, Teacher)",
      all(m in sec7 for m in ("Consultant", "Reviewer", "Teacher")))
sec4 = skill_md[skill_md.index("## 4. Non-negotiable rules"):skill_md.index("## 5.")]
# v2.5.1 added two pieces to the DiD rule (CHANGELOG v2.5.1); remove exactly those, so this check still proves that
# nothing else in section 4 changed since v2.5.0.
V251_ASK = " (relay `did_design`'s ASK THE USER lines)"
V251_DID = (" With one to three treated units, use `causal.did_single_treated` and never report HAC, Newey-West or "
            "Prais-Winsten p-values on the gap series (about three times too many false positives with ~24 periods); "
            "before a causal DiD claim, run `causal.did_falsification` (with `cluster=` when assignment is by cluster) "
            "and report its findings and its uninformative checks.")
# v2.6.6 added three pieces to the sensitivity rule (CHANGELOG v2.6.6): the OLS and DML bounds functions and the
# sentence on stating the boundary with its number; remove exactly those as well.
V266_OVB = " and `ovb_bounds`"
V266_DML = " and `dml_bounds`"
V266_SUM = (" The point where significance is lost is a number (OLS: `k_not_significant`, \"a confounder 1.2x as strong "
            "as X\"; DML: the two `dml_bounds` k values between which the confidence bounds first include zero): say it "
            "with that number in every sentence, the opening summary and the chat reply included, never rounded to 1x "
            "(\"as strong as X\" with no number) or to \"weaker than X\" (`interpretation-rules.md`, v2.6.3 and v2.6.6).")
V266 = (V266_OVB, V266_DML, V266_SUM)
# v2.6.10 (CHANGELOG v2.6.10) added one sentence to the number rule and rewrote three pieces of the delivery rule:
# the locator's paths, listed again in the "Script skeleton" of interpretation-rules.md, left section 4, and the cut
# sentence ". and test it" was repaired. Each pair is (text now, text in v2.5.0); swapping them back rebuilds v2.5.0.
V2610_NUM = (" Every number in the reply, its summary and the document, including the ones worked out from printed "
             "numbers (a sum of coefficients or a combined slope, an MDE, an interval, a percentage), is printed by the "
             "script before it is written; saying it is your own arithmetic does not exempt it "
             "(`interpretation-rules.md`, v2.6.8 and v2.6.10).")
V2610_SUBS = (
    ("(\"Script skeleton\"), which finds the skill so the user runs the file without editing paths and turns a "
     "missing package into a one-line install instruction; never put",
     "(\"Script skeleton\"): it takes `ECONOMETRICS_SKILL_DIR` if set, else finds the Claude Code plugin "
     "(`~/.claude/plugins/marketplaces/econometric-skills/econometrics`), any copy under `~/.claude/skills` (including "
     "the one synced from the Claude app, `~/.claude/skills/synced/.../econometrics`) or the user's clone, so the "
     "user runs the file without editing paths; never put"),
    ("(in your own environment, set `ECONOMETRICS_SKILL_DIR` instead). In Claude Code",
     "(in your own environment, set the variable instead). It also turns a missing package into a one-line "
     "instruction to run `scripts/setup_env.py --install`. In Claude Code"),
    ("which the system may delete. Test the script on data", "which the system may delete. and test it on data"),
)
sec4_v250 = sec4.replace(V251_ASK, "", 1).replace(V251_DID, "", 1)
for piece in V266:
    sec4_v250 = sec4_v250.replace(piece, "", 1)
sec4_v250 = sec4_v250.replace(V2610_NUM, "", 1)
for now, then in V2610_SUBS:
    sec4_v250 = sec4_v250.replace(now, then, 1)
check("SKILL.md section 4 (non-negotiable rules), minus the documented v2.5.1, v2.6.6 and v2.6.10 changes, is "
      "byte-identical to v2.5.0 (SHA-256 pinned, works in a shallow CI checkout)", V251_ASK in sec4 and V251_DID in sec4
      and sec4.count(V2610_NUM) == 1 and all(sec4.count(now) == 1 for now, _ in V2610_SUBS)
      and "a cited source." + V2610_NUM + "\n" in sec4
      and all(sec4.count(piece) == 1 for piece in V266)
      # each piece where the CHANGELOG says it is (review 1 of v2.6.6: moving the sentence passed the hash)
      and "(`specification.sensitivity`" + V266_OVB + ";" in sec4
      and "`dml_benchmark`" + V266_DML + ")." + V266_SUM in sec4
      and hashlib.sha256(sec4_v250.encode("utf-8")).hexdigest()
      == "a820ccbfddb871b324a556bb963a15e75ed594799a292d50c82ec217c83538f1")
agent = (SKILL / "agents" / "econometrician.md").read_text(encoding="utf-8")
check("econometrician agent: --doctor first, changes nothing itself, reports the To do list",
      "setup_env.py --doctor" in agent and "change nothing yourself" in agent and '"To do"' in agent)
check("SKILL.md section 7 names the one network request and how to skip it", "api.github.com" in sec7
      and "--offline" in sec7)

passed = sum(results)
print(f"\n{passed}/{len(results)} checks passed")
sys.exit(0 if passed == len(results) else 1)
