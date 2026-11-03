"""
Environment check and setup for the econometrics skill, the same on Windows, macOS and Linux.

    python setup_env.py --check      report Python version, missing or outdated packages, the skill folder
                                     and the command that runs this Python ("Run scripts with:"; on
                                     Windows usually 'py -X.Y', since 'python' may be the Store alias)
    python setup_env.py --install    install what is missing (plain pip first; adds --break-system-packages
                                     only if pip refuses because the system Python is externally managed)
    python setup_env.py --skill-dir  print the folder that scripts should use as SKILL_DIR

Exit code 0 when everything needed is present (after --install, when it succeeded), 1 otherwise.
In Claude Code, Claude runs --check before the first script on a user's machine and asks before --install.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

MIN_PYTHON = (3, 10)
# import name -> (distribution name, minimum version); kept in sync with requirements.txt (tested)
CORE = {
    "numpy": ("numpy", "2.1"),
    "pandas": ("pandas", "2.2"),
    "scipy": ("scipy", "1.14"),
    "statsmodels": ("statsmodels", "0.14.4"),
    "linearmodels": ("linearmodels", "6.1"),
    "arch": ("arch", "7.2"),
    "sklearn": ("scikit-learn", "1.5"),
    "tabulate": ("tabulate", "0.9"),
}
SKILL_DIR = Path(__file__).resolve().parents[1]
STABLE_PLUGIN_DIR = Path.home() / ".claude" / "plugins" / "marketplaces" / "econometric-skills" / "econometrics"


def _vtuple(v: str) -> tuple:
    out = []
    for part in v.replace("-", ".").split("."):
        num = "".join(ch for ch in part if ch.isdigit())
        if not num:
            break
        out.append(int(num))
    return tuple(out)


def missing_packages() -> list[str]:
    """Requirement strings ('numpy>=2.1') for packages that are absent or older than the minimum."""
    need = []
    for _, (dist, minimum) in CORE.items():
        try:
            installed = metadata.version(dist)
        except metadata.PackageNotFoundError:
            need.append(f"{dist}>={minimum}")
            continue
        if _vtuple(installed) < _vtuple(minimum):
            need.append(f"{dist}>={minimum}")
    return need


def recommended_skill_dir() -> Path:
    """The folder scripts should use: the stable plugin copy when this file runs from the versioned plugin
    cache (whose path changes with every release), otherwise this skill's own folder."""
    if "cache" in SKILL_DIR.parts and STABLE_PLUGIN_DIR.joinpath("scripts", "causal.py").exists():
        return STABLE_PLUGIN_DIR
    return SKILL_DIR


def _same_file(a, b) -> bool:
    if not a or not b:
        return False
    try:
        return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))
    except (OSError, ValueError):
        return False


def _py_launcher_exe(tag: str):
    """The interpreter that the Windows 'py -X.Y' launcher starts, or None."""
    try:
        r = subprocess.run(["py", f"-{tag}", "-c", "import sys; print(sys.executable)"],
                           capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def run_command(exe=None, which=shutil.which, os_name=os.name, launcher=_py_launcher_exe, version=None) -> str:
    """A command that starts this Python from a terminal, for instructions given to the user.

    On Windows, 'python' and 'python3' often open the Microsoft Store alias ("Python was not found")
    even with Python installed, so the 'py -X.Y' launcher comes first when it starts this same
    interpreter (it does not inside a virtual environment); it works in PowerShell, cmd and Git Bash.
    Then 'python' or 'python3' when they are this interpreter; otherwise the quoted full path."""
    exe = exe or sys.executable
    v = version or sys.version_info
    tag = f"{v[0]}.{v[1]}"
    if os_name == "nt" and which("py") and _same_file(launcher(tag), exe):
        return f"py -{tag}"
    for name in ("python", "python3"):
        if _same_file(which(name), exe):
            return name
    return f'"{exe}"' if " " in exe else exe


def store_alias_note(exe=None, which=shutil.which, os_name=os.name) -> str:
    """A warning when 'python' on this Windows machine is the Microsoft Store alias, not this Python."""
    if os_name != "nt":
        return ""
    found = which("python")
    if found and "windowsapps" in found.lower() and not _same_file(found, exe or sys.executable):
        return ("'python' on this machine opens the Microsoft Store alias, not this Python "
                "(it prints 'Python was not found'): use the command above.")
    return ""


def install(reqs: list[str]) -> bool:
    base = [sys.executable, "-m", "pip", "install", "-q", *reqs]
    r = subprocess.run(base, capture_output=True, text=True)
    if r.returncode == 0:
        return True
    if "externally-managed" in (r.stderr + r.stdout):
        r = subprocess.run(base[:4] + ["--break-system-packages"] + base[4:], capture_output=True, text=True)
        if r.returncode == 0:
            return True
    sys.stderr.write(r.stderr[-2000:])
    print(f"pip could not install the packages. Create a virtual environment ({run_command()} -m venv .venv), "
          "activate it and run this script again with --install.")
    return False


def report(reqs: list[str]) -> None:
    ok_py = sys.version_info[:2] >= MIN_PYTHON
    print(f"Python {sys.version.split()[0]} ({sys.executable}): "
          + ("ok" if ok_py else f"TOO OLD, the skill needs {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or later"))
    print("Packages: " + ("all present" if not reqs else "missing or outdated: " + ", ".join(reqs)))
    print(f"Run scripts with: {run_command()}")
    note = store_alias_note()
    if note:
        print(f"  note: {note}")
    rec = recommended_skill_dir()
    print(f"Skill folder for scripts: {rec}")
    env = os.environ.get("ECONOMETRICS_SKILL_DIR")
    if env and Path(env).resolve() != rec.resolve():
        print(f"  note: ECONOMETRICS_SKILL_DIR is set to {env}, a different folder; scripts will use that one.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true")
    g.add_argument("--install", action="store_true")
    g.add_argument("--skill-dir", action="store_true")
    a = ap.parse_args(argv)
    if a.skill_dir:
        print(recommended_skill_dir())
        return 0
    if sys.version_info[:2] < MIN_PYTHON:
        report([])
        return 1
    reqs = missing_packages()
    if a.install and reqs:
        print("Installing: " + " ".join(reqs))
        if not install(reqs):
            return 1
        reqs = missing_packages()
    report(reqs)
    return 0 if not reqs else 1


if __name__ == "__main__":
    sys.exit(main())
