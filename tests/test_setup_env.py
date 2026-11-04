"""Tests for v2.3.2: zero-step setup for Claude Code users (scripts/setup_env.py and the script skeleton
that delivered scripts start with).

Checks: setup_env's package minimums equal requirements.txt; --check passes here; in a fresh virtual
environment without packages, --check exits 1 naming numpy and the skeleton stops with the
setup_env install instruction instead of a traceback; with a fake home that holds only the Claude
Code plugin layout (no variable set) the skeleton finds the skill and imports it; the variable takes
precedence; a versioned plugin-cache location resolves to the stable marketplace copy; and the
skeleton contains no sandbox or versioned cache path.

Run from the repository root:  python3 tests/test_setup_env.py
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "econometrics"
sys.path.insert(0, str(SKILL / "scripts"))
import setup_env as se

ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


# ---------------------------------------------------------------- 1. setup_env.py
req = {}
for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
    m = re.match(r"^([A-Za-z0-9_.-]+)>=([0-9.]+)", line.strip())
    if m:
        req[m.group(1).lower()] = m.group(2)
core = {dist.lower(): v for dist, v in se.CORE.values()}
check("setup_env minimum versions equal requirements.txt", core == req, f"{core} vs {req}")
r = subprocess.run([sys.executable, str(SKILL / "scripts" / "setup_env.py"), "--check"], capture_output=True, text=True)
check("--check passes in the test environment", r.returncode == 0 and "all present" in r.stdout, r.stdout.strip()[:120])
check("version parsing handles suffixes", se._vtuple("0.14.4") > se._vtuple("0.14") and se._vtuple("2.1.0rc1") >= (2, 1)
      and se._vtuple("1.5.2") < se._vtuple("1.10"))

# versioned plugin cache -> stable marketplace copy
with tempfile.TemporaryDirectory() as tmp:
    stable = Path(tmp) / "marketplaces" / "econometric-skills" / "econometrics"
    (stable / "scripts").mkdir(parents=True)
    (stable / "scripts" / "causal.py").write_text("", encoding="utf-8")
    old_skill, old_stable = se.SKILL_DIR, se.STABLE_PLUGIN_DIR
    se.SKILL_DIR = Path(tmp) / "cache" / "econometric-skills" / "econometrics" / "2.3.2"
    se.STABLE_PLUGIN_DIR = stable
    check("a versioned plugin-cache location resolves to the stable marketplace copy", se.recommended_skill_dir() == stable)
    se.SKILL_DIR = SKILL
    check("outside the plugin cache, the skill's own folder is used", se.recommended_skill_dir() == SKILL)
    se.SKILL_DIR, se.STABLE_PLUGIN_DIR = old_skill, old_stable

# ---------------------------------------------------------------- 1b. the command to run Python (v2.3.5)
# On Windows, 'python' and 'python3' often open the Microsoft Store alias ("Python was not found")
# even with Python installed; only the 'py' launcher works. Instructions must name a command that runs.
EXE = r"C:\Users\Ana Silva\AppData\Local\Programs\Python\Python313\python.exe"
STORE = r"C:\Users\Ana Silva\AppData\Local\Microsoft\WindowsApps\python.exe"
V = (3, 13)


def fake_which(table):
    return lambda name: table.get(name)


cmd = se.run_command(exe=EXE, which=fake_which({"py": "py.exe", "python": STORE}), os_name="nt",
                     launcher=lambda tag: EXE, version=V)
check("Windows, py launcher runs this Python: the command is 'py -3.13' (works in PowerShell, cmd, Git Bash)",
      cmd == "py -3.13", cmd)
cmd = se.run_command(exe=r"C:\proj\.venv\Scripts\python.exe",
                     which=fake_which({"py": "py.exe", "python": r"C:\proj\.venv\Scripts\python.exe"}), os_name="nt",
                     launcher=lambda tag: EXE, version=V)
check("Windows virtual environment (py would start another Python): the command is 'python'", cmd == "python", cmd)
cmd = se.run_command(exe=EXE, which=fake_which({"python": STORE, "python3": STORE}), os_name="nt",
                     launcher=lambda tag: None, version=V)
check("Windows, no launcher, 'python' is the Store alias: never 'python', the quoted full path instead",
      cmd == f'"{EXE}"', cmd)
cmd = se.run_command(exe="/usr/bin/python3", which=fake_which({"python3": "/usr/bin/python3"}), os_name="posix",
                     launcher=lambda tag: None, version=V)
check("macOS/Linux: the command is 'python3' when it is this Python", cmd == "python3", cmd)
note = se.store_alias_note(exe=EXE, which=fake_which({"python": STORE}), os_name="nt")
check("Windows: a note when 'python' opens the Microsoft Store alias", "Microsoft Store" in note, note[:80])
check("no note when 'python' is this Python, or off Windows",
      se.store_alias_note(exe=EXE, which=fake_which({"python": EXE}), os_name="nt") == ""
      and se.store_alias_note(exe="/usr/bin/python3", which=fake_which({"python": STORE}), os_name="posix") == "")
r = subprocess.run([sys.executable, str(SKILL / "scripts" / "setup_env.py"), "--check"], capture_output=True, text=True)
check("--check prints the command to run scripts with", "Run scripts with: " in r.stdout,
      next((l for l in r.stdout.splitlines() if l.startswith("Run scripts")), "")[:90])

# ---------------------------------------------------------------- 1c. optional extras (v2.3.6)
# Reading or writing .xlsx needs openpyxl. In a clean-install test Claude asked to install the 8 core
# packages and then ran a separate 'pip install openpyxl' the user had not approved. Optional packages
# now go through setup_env (--extra), in the same permission question, with a clear message when missing.
dev = {}
for line in (ROOT / "requirements-dev.txt").read_text(encoding="utf-8").splitlines():
    m = re.match(r"^([A-Za-z0-9_.-]+)>=([0-9.]+)", line.strip())
    if m:
        dev[m.group(1).lower()] = m.group(2)
opt = {dist.lower(): v for pkgs in getattr(se, "OPTIONAL", {}).values() for dist, v in pkgs.values()}
check("extra 'excel' is openpyxl, with the same minimum as requirements-dev.txt",
      "openpyxl" in opt and all(dev.get(k) == v for k, v in opt.items()), f"{opt} vs dev")


class blocked:
    """Make a module look absent (importlib.util.find_spec returns None, import raises ImportError)."""
    def __init__(self, name):
        self.name, self.saved = name, None

    def __enter__(self):
        self.saved = sys.modules.get(self.name, "absent")
        sys.modules[self.name] = None

    def __exit__(self, *a):
        if self.saved == "absent":
            sys.modules.pop(self.name, None)
        else:
            sys.modules[self.name] = self.saved


import contextlib
import importlib.util
import io

skill_md_text = (SKILL / "SKILL.md").read_text(encoding="utf-8")

with blocked("openpyxl"):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code_plain = se.main(["--check"])
        code_extra = se.main(["--check", "--extra", "excel"])
    out = buf.getvalue()
check("--check names the missing extra and how to get it, without failing the core check",
      "Optional 'excel'" in out and "--extra excel" in out and code_plain == 0, out.strip().splitlines()[-1][:100])
check("--check --extra excel exits 1 when openpyxl is missing", code_extra == 1)
asked = []
old_install = se.install
se.install = lambda reqs: asked.append(list(reqs)) or True
try:
    with blocked("openpyxl"), contextlib.redirect_stdout(io.StringIO()):
        se.main(["--install", "--extra", "excel"])
finally:
    se.install = old_install
check("--install --extra excel installs openpyxl (and only what is missing)",
      asked == [[f"openpyxl>={opt.get('openpyxl')}"]], asked)
try:
    with contextlib.redirect_stderr(io.StringIO()):
        se.main(["--install", "--extra", "nonsense"])
    bad_extra = False
except SystemExit as e:
    bad_extra = e.code == 2
check("an unknown extra is rejected by the argument parser", bad_extra)

import pandas as pd
import reporting
import data_prep
import diagnostics


def message_of(fn):
    try:
        fn()
    except ImportError as e:
        return str(e)
    return ""


with tempfile.TemporaryDirectory() as tmp:
    tab = pd.DataFrame({"model": ["(1)", "(2)"], "educ": [0.083, 0.09]}).set_index("model")
    xlsx = Path(tmp) / "t.xlsx"
    with blocked("openpyxl"):
        m_write = message_of(lambda: reporting.export_table(tab, str(xlsx)))
        xlsx.write_bytes(b"PK\x03\x04 not a real workbook")
        m_read = message_of(lambda: data_prep.read_any(xlsx))
        m_cli = message_of(lambda: diagnostics._read(str(xlsx), None))
    for what, msg in (("reporting.export_table(.xlsx)", m_write), ("data_prep.read_any(.xlsx)", m_read),
                      ("diagnostics command line with .xlsx", m_cli)):
        check(f"no openpyxl: {what} stops with the setup_env --extra excel command",
              "openpyxl" in msg and "setup_env.py" in msg and "--install --extra excel" in msg, msg[:110])
    xlsx.unlink()
    if importlib.util.find_spec("openpyxl"):
        reporting.export_table(tab, str(xlsx))
        back = data_prep.read_any(xlsx).set_index("model")
        check("with openpyxl: a table exported to .xlsx reads back unchanged",
              back.round(6).equals(tab.round(6)), back.to_dict())
    else:
        check("openpyxl is installed in the test environment (requirements-dev.txt)", False)
check("SKILL.md: the permission question lists every package, optional extras included, and installs them via setup_env",
      "--extra excel" in skill_md_text and "every package" in skill_md_text)

# ---------------------------------------------------------------- 2. the script skeleton
rules = (SKILL / "references" / "interpretation-rules.md").read_text(encoding="utf-8")
m = re.search(r"Script skeleton for every deliverable.*?```python\n(.*?)```", rules, re.S)
check("interpretation-rules.md contains the script skeleton", m is not None)
block = m.group(1) if m else ""
check("skeleton has no sandbox or versioned cache path",
      all(s not in block for s in ("/mnt/", "/home/claude", "/tmp/", "plugins/cache")) and "ECONOMETRICS_SKILL_DIR" in block)
skill_md = (SKILL / "SKILL.md").read_text(encoding="utf-8")
check("SKILL.md delivery rule points to the skeleton and setup_env",
      "locator block" in skill_md and "setup_env.py --check" in skill_md and "/mnt/" not in skill_md)
probe = block + "\nprint('FOUND', SKILL_DIR)\n"


def run(py, home, env_var=None):
    env = {k: v for k, v in os.environ.items() if k != "ECONOMETRICS_SKILL_DIR"}
    env["HOME"] = env["USERPROFILE"] = str(home)
    if env_var:
        env["ECONOMETRICS_SKILL_DIR"] = env_var
    with tempfile.TemporaryDirectory() as wd:
        f = Path(wd) / "analysis.py"
        f.write_text(probe, encoding="utf-8")
        return subprocess.run([py, str(f)], capture_output=True, text=True, cwd=wd, env=env)


with tempfile.TemporaryDirectory() as tmp:
    home = Path(tmp) / "home"
    plugin = home / ".claude" / "plugins" / "marketplaces" / "econometric-skills" / "econometrics"
    plugin.parent.mkdir(parents=True)
    shutil.copytree(SKILL, plugin, ignore=shutil.ignore_patterns("__pycache__"))
    r = run(sys.executable, home)
    check("no variable set: the skeleton finds the Claude Code plugin and imports the skill",
          r.returncode == 0 and f"FOUND {plugin}" in r.stdout, (r.stdout + r.stderr).strip()[-120:])
    r = run(sys.executable, home, env_var=str(SKILL))
    check("ECONOMETRICS_SKILL_DIR takes precedence over the plugin", r.returncode == 0 and f"FOUND {SKILL}" in r.stdout)
    synced_home = Path(tmp) / "synced_home"
    synced = synced_home / ".claude" / "skills" / "synced" / "a1b2c3" / "econometrics"
    synced.parent.mkdir(parents=True)
    shutil.copytree(SKILL, synced, ignore=shutil.ignore_patterns("__pycache__"))
    r = run(sys.executable, synced_home)
    check("only the copy synced from the Claude app (~/.claude/skills/synced/...): the skeleton finds it",
          r.returncode == 0 and f"FOUND {synced}" in r.stdout, (r.stdout + r.stderr).strip()[-120:])
    both = Path(tmp) / "both_home"
    shutil.copytree(home, both)
    shutil.copytree(synced_home / ".claude" / "skills", both / ".claude" / "skills")
    r = run(sys.executable, both)
    check("plugin and synced copy both present: the plugin copy is used",
          r.returncode == 0 and "plugins/marketplaces/econometric-skills" in r.stdout.replace("\\", "/"))
    empty = Path(tmp) / "empty_home"
    empty.mkdir()
    r = run(sys.executable, empty)
    check("skill nowhere: a one-line message, not a traceback",
          r.returncode != 0 and "Econometrics skill not found" in r.stderr and "Traceback" not in r.stderr)

    # fresh virtual environment without any package
    venv_dir = Path(tmp) / "bare"
    venv.EnvBuilder(with_pip=False).create(venv_dir)
    py = str(venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
    r = subprocess.run([py, str(SKILL / "scripts" / "setup_env.py"), "--check"], capture_output=True, text=True)
    check("bare Python: --check exits 1 and names the missing packages",
          r.returncode == 1 and "numpy>=2.1" in r.stdout, r.stdout.strip().splitlines()[1][:100] if r.stdout else "")
    r = run(py, home)
    check("bare Python: the skeleton stops with the setup_env install instruction, not a traceback",
          r.returncode != 0 and "setup_env.py" in r.stderr and "--install" in r.stderr and "Traceback" not in r.stderr,
          r.stderr.strip()[:110])
    venv_exe = subprocess.run([py, "-c", "import sys; print(sys.executable)"], capture_output=True, text=True).stdout.strip()
    check("bare Python: the install instruction names this exact Python, not a bare 'python'",
          f'"{venv_exe}"' in r.stderr and "Run once:  python " not in r.stderr, r.stderr.strip()[-110:])

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
