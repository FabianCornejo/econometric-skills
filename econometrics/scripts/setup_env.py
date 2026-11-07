"""
Environment check and setup for the econometrics skill, the same on Windows, macOS and Linux.

    python setup_env.py --check      report Python version, missing or outdated packages, the skill folder
                                     and the command that runs this Python ("Run scripts with:"; on
                                     Windows usually 'py -X.Y', since 'python' may be the Store alias)
    python setup_env.py --doctor     everything needed for a complete, up-to-date setup, in one report and
                                     one list of what to ask the user: the Python to use (other interpreters
                                     on the machine included), packages and every optional extra, this skill's
                                     version against the latest release, other copies of the skill (plugin,
                                     Claude app) and their versions, plugin auto-update, and a crowded skill
                                     listing. Installs nothing. --json for a machine-readable report;
                                     --offline skips the release check (one request to api.github.com).
    python setup_env.py --install    install what is missing (plain pip first; adds --break-system-packages
                                     only if pip refuses because the system Python is externally managed)
    python setup_env.py --skill-dir  print the folder that scripts should use as SKILL_DIR
    python setup_env.py --enable-autoupdate
                                     turn on Claude Code's background auto-update for the econometric-skills
                                     marketplace: writes "autoUpdate": true to its extraKnownMarketplaces entry
                                     in ~/.claude/settings.json (backup first). Only with the user's permission.
    --extra NAME                     with --check or --install: also an optional extra (see OPTIONAL below;
                                     excel = openpyxl for .xlsx, plots = matplotlib for PNG charts)

Exit code 0 when everything needed is present (after --install, when it succeeded), 1 otherwise.
In Claude Code, Claude runs --doctor before the first analysis on a machine and asks the user once
before installing or changing anything.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
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
# Optional packages by feature (--extra NAME), same layout as CORE. They are never installed on their own:
# the agent names them in the permission question when the task needs them (heavy ones in a question of
# their own, with the download size). Minimums of the extras in requirements-dev.txt are tested in CI;
# UNTESTED_MINIMUM lists the extras CI does not install, whose minimums are documented, not tested.
OPTIONAL = {
    "excel": {"openpyxl": ("openpyxl", "3.1")},
    "plots": {"matplotlib": ("matplotlib", "3.8")},
    "rd": {"rdrobust": ("rdrobust", "1.3")},
    "regimes": {"hmmlearn": ("hmmlearn", "0.3"), "ruptures": ("ruptures", "1.1")},
    "panel": {"pydynpd": ("pydynpd", "0.2")},
    "ml": {"econml": ("econml", "0.16")},
    "pdf": {"pymupdf": ("pymupdf", "2.1")},
    "ocr": {"easyocr": ("easyocr", "1.7")},
    "web": {"requests": ("requests", "2.34")},
    "text": {"sentence_transformers": ("sentence-transformers", "3.0")},
    "llm": {"transformers": ("transformers", "4.40"), "torch": ("torch", "2.2")},
    "data-br": {"bcb": ("python-bcb", "0.4"), "sidrapy": ("sidrapy", "0.1"), "ipeadatapy": ("ipeadatapy", "0.1")},
    "data-intl": {"wbgapi": ("wbgapi", "1.0"), "fredapi": ("fredapi", "0.5"), "yfinance": ("yfinance", "1.8"),
                  "pandas_datareader": ("pandas-datareader", "0.11")},
}
OPTIONAL_USE = {
    "excel": "read or write .xlsx/.xlsm files",
    "plots": "save charts (PNG) from a delivered script",
    "rd": "regression discontinuity bandwidths and robust inference (causal.select_bandwidth, rd_estimate)",
    "regimes": "hidden Markov models and change-point detection (regimes.hmm_regimes; ruptures)",
    "panel": "dynamic panel GMM cross-check (pydynpd)",
    "ml": "causal forests with pointwise CIs, IV and dynamic CATE (econml)",
    "pdf": "text from PDF files (data_prep.pdf_text)",
    "ocr": "OCR of scanned PDF pages (data_prep.pdf_text with ocr=True)",
    "web": "polite web requests and scraping (data_prep.polite_get)",
    "text": "sentence embeddings (text_data.embed)",
    "llm": "local LLM coding with token probabilities (text_data.hf_backend)",
    "data-br": "Brazilian data clients: BCB (SGS, Focus), IBGE SIDRA, Ipeadata",
    "data-intl": "international data clients: World Bank, FRED, Yahoo Finance, pandas-datareader",
}
# Extras whose download goes beyond 50 MB on top of the core packages: asked in a question of their own,
# with the size. Measured on 2026-11-06 for Windows (win_amd64), Python 3.13 and 3.14, by resolving each
# extra and summing the files PyPI serves, minus what the core packages already bring.
HEAVY_MB = {"ml": 50, "pdf": 81, "llm": 160, "text": 160, "ocr": 205}
UNTESTED_MINIMUM = ("ocr", "text", "llm")      # torch-based: not installed in CI
# Known gaps of upstream packages for some Pythons (checked 2026-11-06):
# dist -> (from Python version, platform or None for all, pip cannot install it at all, note)
PY_CAVEATS = {
    "ruptures": ((3, 14), None, True, "has no release for Python 3.14 yet (it declares requires-python <3.14)"),
    "hmmlearn": ((3, 14), "nt", False, "has no Windows wheel for Python 3.14: pip builds it from source, which "
                                       "needs Microsoft C++ Build Tools"),
}
SKILL_DIR = Path(__file__).resolve().parents[1]
STABLE_PLUGIN_DIR = Path.home() / ".claude" / "plugins" / "marketplaces" / "econometric-skills" / "econometrics"

REPO = "FabianCornejo/econometric-skills"
MARKETPLACE = "econometric-skills"
PLUGIN_ID = "econometrics@econometric-skills"
RELEASES_API = f"https://api.github.com/repos/{REPO}/releases/latest"
RELEASE_ASSET = f"https://github.com/{REPO}/releases/latest/download/econometrics.skill"
# A skill listing this long overflowed Claude Code's budget in the routing evals (the plugin plus 48 decoy
# skills, tools/routing_evals.py): from here on, descriptions may be dropped and other skills compete.
CROWDED_SKILLS = 50
UPDATE_ENV_OFF = ("DISABLE_UPDATES", "DISABLE_AUTOUPDATER", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC")


def _vtuple(v: str) -> tuple:
    out = []
    for part in v.replace("-", ".").split("."):
        num = "".join(ch for ch in part if ch.isdigit())
        if not num:
            break
        out.append(int(num))
    return tuple(out)


def _installed_version(import_name: str, dist: str):
    """Version of an importable package, or None when it cannot be imported here."""
    try:
        if importlib.util.find_spec(import_name) is None:
            return None
    except (ImportError, ValueError):
        return None
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def missing_packages(extras=(), core: bool = True) -> list[str]:
    """Requirement strings ('numpy>=2.1') for packages that are absent or older than the minimum:
    the core packages (unless core=False) plus those of each optional extra named in `extras`."""
    groups = ([CORE] if core else []) + [OPTIONAL[e] for e in extras]
    need = []
    for group in groups:
        for import_name, (dist, minimum) in group.items():
            installed = _installed_version(import_name, dist)
            if installed is None or _vtuple(installed) < _vtuple(minimum):
                need.append(f"{dist}>={minimum}")
    return need


def _caveat(dist: str, version=None, os_name=os.name):
    """The PY_CAVEATS entry that applies to `dist` on this Python and platform, or None."""
    v = tuple((version or sys.version_info)[:2])
    cav = PY_CAVEATS.get(dist)
    if cav and v >= cav[0] and (cav[1] is None or cav[1] == os_name):
        return cav
    return None


def extra_caveats(extra: str, version=None, os_name=os.name, only=None) -> list[str]:
    """Known reasons why (part of) an extra may not install on this Python (PY_CAVEATS)."""
    out = []
    for imp, (dist, _m) in OPTIONAL[extra].items():
        cav = _caveat(dist, version, os_name)
        if cav and (only is None or imp in only):
            out.append(f"{dist} {cav[3]}")
    return out


def not_installable(reqs: list[str], version=None, os_name=os.name) -> list[str]:
    """The requirement strings pip cannot install on this Python at all (PY_CAVEATS with the flag set)."""
    out = []
    for r in reqs:
        cav = _caveat(r.split(">=")[0], version, os_name)
        if cav and cav[2]:
            out.append(r)
    return out


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


def install_command(extra: str | None = None) -> str:
    """The exact command, for this machine, that installs what is missing (plus one optional extra)."""
    cmd = f'{run_command()} "{recommended_skill_dir() / "scripts" / "setup_env.py"}" --install'
    return cmd + (f" --extra {extra}" if extra else "")


def require_extra(extra: str, purpose: str, only=None) -> None:
    """Raise ImportError with the install command when an optional extra is missing.

    Called by the modules right before the feature that needs it (e.g. reporting.export_table to .xlsx),
    so the user sees one actionable line instead of pandas' 'Missing optional dependency'. For a heavy
    extra the message gives the download size, so the agent asks about it in a question of its own.
    `only`: the import names this feature needs, when it uses part of the extra (hmm_regimes needs hmmlearn,
    not ruptures, which has no Python 3.14 release)."""
    group = OPTIONAL[extra] if only is None else {k: OPTIONAL[extra][k] for k in only}
    missing = []
    for imp, (dist, m) in group.items():
        iv = _installed_version(imp, dist)
        if iv is None or _vtuple(iv) < _vtuple(m):
            missing.append(f"{dist}>={m}")
    if missing:
        size = (f" It is a large download (about {HEAVY_MB[extra]} MB on top of the core packages): "
                "ask the user about it on its own." if extra in HEAVY_MB else "")
        cav = extra_caveats(extra, only=only)
        cav = (" Note: " + "; ".join(cav) + ".") if cav else ""
        raise ImportError(f"{purpose} needs {', '.join(missing)}, which is not installed. "
                          f"Install it once, with the user's permission:  {install_command(extra)}.{size}{cav}")


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


COMMON_EXTRAS = ("excel", "plots")   # the ones a delivered report usually needs; --check lists them


def report(reqs: list[str]) -> None:
    ok_py = sys.version_info[:2] >= MIN_PYTHON
    print(f"Python {sys.version.split()[0]} ({sys.executable}): "
          + ("ok" if ok_py else f"TOO OLD, the skill needs {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or later"))
    print("Packages: " + ("all present" if not reqs else "missing or outdated: " + ", ".join(reqs)))
    for name in COMMON_EXTRAS:
        miss = missing_packages([name], core=False)
        state = "present" if not miss else (f"missing {', '.join(miss)} (only if needed: "
                                            f"--install --extra {name}; ask the user first)")
        print(f"Optional '{name}' ({OPTIONAL_USE[name]}): {state}")
    others = [n for n in OPTIONAL if n not in COMMON_EXTRAS]
    print(f"Other optional extras ({', '.join(others)}): see --doctor")
    print(f"Run scripts with: {run_command()}")
    note = store_alias_note()
    if note:
        print(f"  note: {note}")
    rec = recommended_skill_dir()
    print(f"Skill folder for scripts: {rec}")
    env = os.environ.get("ECONOMETRICS_SKILL_DIR")
    if env and Path(env).resolve() != rec.resolve():
        print(f"  note: ECONOMETRICS_SKILL_DIR is set to {env}, a different folder; scripts will use that one.")


# ---------------------------------------------------------------------------------------------------------
# --doctor: the whole setup in one report (v2.5.0.1). Every probe takes its inputs as arguments, so the
# tests can simulate machines; doctor_actions() turns the report into the one question for the user.
# ---------------------------------------------------------------------------------------------------------

def claude_dir(env=None, home=None) -> Path:
    """Claude Code's configuration folder: CLAUDE_CONFIG_DIR when set, otherwise ~/.claude."""
    env = os.environ if env is None else env
    d = env.get("CLAUDE_CONFIG_DIR")
    return Path(d) if d else Path(home or Path.home()) / ".claude"


def _load_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_skill_version(folder) -> str | None:
    """SKILL_VERSION of the copy of the skill in `folder` (from scripts/reporting.py), or None.

    The version, not a file hash, identifies a copy: the Claude app repackages the uploaded .skill file,
    so the same release has a different hash there (seen on 2026-11-06)."""
    try:
        text = (Path(folder) / "scripts" / "reporting.py").read_text(encoding="utf-8")
    except OSError:
        return None
    m = re.search(r'^SKILL_VERSION = "([^"]+)"', text, re.M)
    return m.group(1) if m else None


def skill_copies(cdir: Path) -> list[dict]:
    """Every copy of the skill Claude Code can see: the installed plugin, the marketplace checkout (the
    source of the next update), copies synced from the Claude app and manual copies under skills/.
    Copies in skills/.trash (removed from the app) are ignored, as Claude Code ignores them."""
    out = []
    inst = _load_json(cdir / "plugins" / "installed_plugins.json") or {}
    entry = (inst.get("plugins") or {}).get(PLUGIN_ID)
    if isinstance(entry, list):                      # one record per scope
        entry = entry[0] if entry else None
    if isinstance(entry, dict) and entry.get("installPath"):
        p = Path(entry["installPath"])
        out.append({"kind": "plugin", "path": str(p), "version": read_skill_version(p) or entry.get("version")})
    mk = cdir / "plugins" / "marketplaces" / MARKETPLACE / "econometrics"
    if (mk / "SKILL.md").exists():
        out.append({"kind": "marketplace checkout", "path": str(mk), "version": read_skill_version(mk)})
    skills = cdir / "skills"
    if skills.is_dir():
        for sk in sorted(skills.glob("**/econometrics/SKILL.md")):
            rel = sk.relative_to(skills).parts
            if ".trash" in rel:
                continue
            kind = "Claude app (synced)" if "synced" in rel else "manual copy"
            out.append({"kind": kind, "path": str(sk.parent), "version": read_skill_version(sk.parent)})
    return out


def latest_release(opener=None, timeout: float = 4.0) -> tuple[str | None, str]:
    """(version, status) of the latest GitHub release; (None, 'not checked: ...') when offline or on error."""
    opener = opener or urllib.request.urlopen
    req = urllib.request.Request(RELEASES_API, headers={"Accept": "application/vnd.github+json",
                                                         "User-Agent": "econometrics-skill-setup-env"})
    try:
        with opener(req, timeout=timeout) as r:
            tag = json.loads(r.read().decode("utf-8")).get("tag_name") or ""
    except Exception as e:                           # no network, proxy, rate limit, bad JSON
        return None, f"not checked ({type(e).__name__})"
    tag = tag[1:] if tag[:1] in "vV" else tag
    return (tag, "ok") if _vtuple(tag) else (None, "not checked (no version in the reply)")


def marketplace_autoupdate(cdir: Path, env=None) -> dict:
    """Is Claude Code's background auto-update on for the econometric-skills marketplace?

    Order, as Claude Code documents it: "autoUpdate" on the extraKnownMarketplaces entry of a settings file,
    then on the known_marketplaces.json entry (what the /plugin toggle writes), then the default, which is
    off for third-party marketplaces. DISABLE_UPDATES, DISABLE_AUTOUPDATER and
    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC turn the whole pass off unless FORCE_AUTOUPDATE_PLUGINS=1.
    Managed (organization) settings can also decide it; they are not read here."""
    env = os.environ if env is None else env

    def truthy(k):
        return str(env.get(k, "")).strip().lower() not in ("", "0", "false", "no")

    blocked = [] if truthy("FORCE_AUTOUPDATE_PLUGINS") else [k for k in UPDATE_ENV_OFF if truthy(k)]
    settings = _load_json(cdir / "settings.json") or {}
    ekm = (settings.get("extraKnownMarketplaces") or {}).get(MARKETPLACE)
    known = (_load_json(cdir / "plugins" / "known_marketplaces.json") or {}).get(MARKETPLACE)
    if isinstance(ekm, dict) and "autoUpdate" in ekm:
        on, source = bool(ekm["autoUpdate"]), "settings.json"
    elif isinstance(known, dict) and "autoUpdate" in known:
        on, source = bool(known["autoUpdate"]), "known_marketplaces.json (/plugin toggle)"
    else:
        on, source = False, "default (off for third-party marketplaces)"
    return {"registered": bool(ekm or known), "on": on and not blocked, "configured": on, "source": source,
            "blocked_by": blocked}


def enable_autoupdate(cdir: Path, now=None) -> tuple[Path, Path | None]:
    """Write "autoUpdate": true on the econometric-skills entry of extraKnownMarketplaces in
    <claude dir>/settings.json, keeping every other key; back the file up first. Returns (file, backup).
    Raise ValueError when the file is not valid JSON (it is then left untouched)."""
    path = cdir / "settings.json"
    backup = None
    settings = {}
    if path.exists():
        settings = _load_json(path)
        if not isinstance(settings, dict):
            raise ValueError(f"{path} is not valid JSON; fix it or use /plugin -> Marketplaces instead.")
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
        backup = path.with_name(f"settings.json.bak-{stamp}")
        shutil.copy2(path, backup)
    ekm = settings.setdefault("extraKnownMarketplaces", {})
    entry = ekm.get(MARKETPLACE) if isinstance(ekm.get(MARKETPLACE), dict) else {}
    entry.setdefault("source", {"source": "github", "repo": REPO})
    entry["autoUpdate"] = True
    ekm[MARKETPLACE] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path, backup


_PROBE = ("import json,sys,importlib.util as u\n"
          "def f(n):\n"
          "    try:\n        return u.find_spec(n) is not None\n"
          "    except Exception:\n        return False\n"
          "print(json.dumps({'exe': sys.executable, 'v': list(sys.version_info[:3]),"
          " 'missing': [n for n in %r if not f(n)]}))")


def python_candidates(os_name=os.name, run=subprocess.run, which=shutil.which) -> list[str]:
    """Other Python executables on the machine: 'py -0p' on Windows, python3.X on the PATH elsewhere."""
    found = []
    if os_name == "nt" and which("py"):
        try:
            out = run(["py", "-0p"], capture_output=True, text=True, timeout=20).stdout
        except (OSError, subprocess.SubprocessError):
            out = ""
        for line in out.splitlines():
            m = re.search(r"([A-Za-z]:\\.*?python(?:w)?\.exe)\s*$", line.strip(), re.I)
            if m:
                found.append(m.group(1))
    else:
        for minor in range(MIN_PYTHON[1], 20):
            p = which(f"python3.{minor}")
            if p:
                found.append(p)
    uniq = []
    for p in found:
        if not any(_same_file(p, q) for q in uniq):
            uniq.append(p)
    return uniq


def probe_interpreter(exe: str, run=subprocess.run) -> dict | None:
    """Version and missing core packages of another interpreter (by import name), or None if it fails."""
    try:
        r = run([exe, "-c", _PROBE % (list(CORE),)], capture_output=True, text=True, timeout=60)
        d = json.loads(r.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None
    return {"exe": exe, "version": ".".join(map(str, d["v"])), "missing_core": d["missing"],
            "supported": tuple(d["v"][:2]) >= MIN_PYTHON}


def skill_listing(cdir: Path, run=subprocess.run, which=shutil.which, cwd=None) -> dict:
    """How crowded Claude Code's skill listing is: enabled plugins and their skills (from
    `claude plugin list --json`, run in `cwd` so a folder's own settings count) plus skills under skills/."""
    settings = _load_json(cdir / "settings.json") or {}
    res = {"checked": False, "budget_fraction": settings.get("skillListingBudgetFraction")}
    exe = which("claude")
    if exe is None:
        res["reason"] = "Claude Code CLI not on the PATH"
        return res
    try:
        out = run([exe, "plugin", "list", "--json"], capture_output=True, text=True, encoding="utf-8",
                  timeout=120, cwd=cwd).stdout
        plugins = json.loads(out)
    except (OSError, subprocess.SubprocessError, ValueError):
        res["reason"] = "claude plugin list --json failed"
        return res
    enabled = [p for p in plugins if isinstance(p, dict) and p.get("enabled")]
    n_skills = 0
    for p in enabled:
        ip = Path(p.get("installPath") or "")
        if ip.is_dir():
            n_skills += sum(1 for _ in ip.glob("**/SKILL.md"))
    sk = cdir / "skills"
    if sk.is_dir():
        n_skills += sum(1 for f in sk.glob("**/SKILL.md") if ".trash" not in f.relative_to(sk).parts)
    res.update(checked=True, enabled_plugins=len(enabled), skills=n_skills,
               crowded=n_skills >= CROWDED_SKILLS,
               others_enabled=[p.get("id") for p in enabled if not str(p.get("id", "")).startswith("econometrics@")])
    return res


def doctor(offline: bool = False, *, cdir=None, env=None, opener=None, run=subprocess.run, which=shutil.which,
           os_name=os.name, cwd=None) -> dict:
    """The full report (a dict). Installs and changes nothing."""
    cdir = Path(cdir) if cdir else claude_dir(env)
    here = sys.version_info[:2] >= MIN_PYTHON
    rep = {
        "python": {"exe": sys.executable, "version": sys.version.split()[0], "supported": here,
                   "run_with": run_command(which=which, os_name=os_name)},
        "core_missing": missing_packages() if here else list(f"{d}>={m}" for d, m in CORE.values()),
        "extras": {n: {"use": OPTIONAL_USE[n], "missing": missing_packages([n], core=False),
                       "heavy_mb": HEAVY_MB.get(n), "minimum_tested": n not in UNTESTED_MINIMUM,
                       "caveats": extra_caveats(n, os_name=os_name)} for n in OPTIONAL},
        "skill": {"dir": str(SKILL_DIR), "version": read_skill_version(SKILL_DIR)},
        "copies": skill_copies(cdir),
        "autoupdate": marketplace_autoupdate(cdir, env),
        "listing": skill_listing(cdir, run=run, which=which, cwd=cwd),
    }
    others = []
    for exe in python_candidates(os_name=os_name, run=run, which=which):
        if _same_file(exe, sys.executable):
            continue
        info = probe_interpreter(exe, run=run)
        if info:
            others.append(info)
    rep["interpreters"] = others
    if offline:
        rep["latest"] = {"version": None, "status": "not checked (--offline)"}
    else:
        v, status = latest_release(opener=opener)
        rep["latest"] = {"version": v, "status": status}
    rep["actions"] = doctor_actions(rep, os_name=os_name)
    return rep


def _update_steps() -> str:
    return ("claude plugin marketplace update econometric-skills, then claude plugin update "
            f"{PLUGIN_ID} (or /plugin -> Marketplaces -> econometric-skills -> u), then restart Claude Code")


def doctor_actions(rep: dict, os_name=os.name) -> list[str]:
    """What to put in the single question to the user, in order. Empty when nothing is needed."""
    acts = []
    py = rep["python"]
    complete = [i for i in rep.get("interpreters", []) if i["supported"] and not i["missing_core"]]
    if not py["supported"]:
        acts.append(f"Python {py['version']} is too old (the skill needs {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+): "
                    + (f"use {complete[0]['exe']} (Python {complete[0]['version']}, packages present)" if complete
                       else "install a current Python: " + python_install_hint(os_name)))
    elif rep["core_missing"]:
        if complete:
            best = max(complete, key=lambda i: _vtuple(i["version"]))
            acts.append(f"this Python ({py['version']}) lacks {', '.join(rep['core_missing'])}, but "
                        f"{best['exe']} (Python {best['version']}) has every core package: run the scripts with "
                        "that one instead of installing again")
        else:
            acts.append("install the core packages: " + ", ".join(rep["core_missing"])
                        + f"  ({install_command()})")
    mine, latest = rep["skill"]["version"], rep["latest"]["version"]
    if mine and latest and _vtuple(latest) > _vtuple(mine):
        acts.append(f"update the skill from {mine} to {latest}: {_update_steps()}")
    vers = {c["version"] for c in rep["copies"] if c["kind"] != "marketplace checkout" and c["version"]}
    if len(vers) > 1:
        desc = "; ".join(f"{c['kind']} {c['version']}" for c in rep["copies"] if c["kind"] != "marketplace checkout")
        app = any(c["kind"] == "Claude app (synced)" for c in rep["copies"])
        acts.append(f"copies of the skill at different versions ({desc}): keep them equal"
                    + (f" (the Claude app copy: Customize -> Skills, upload {RELEASE_ASSET}, or remove it there "
                       "and keep only the plugin)" if app else ""))
    au = rep["autoupdate"]
    if au["registered"] and not au["on"]:
        if au["blocked_by"]:
            acts.append("plugin auto-update is off for every marketplace because "
                        + ", ".join(au["blocked_by"]) + " is set; updates need the commands above")
        else:
            acts.append("plugin auto-update is off for econometric-skills (default for third-party "
                        "marketplaces). With the user's permission: /plugin -> Marketplaces -> econometric-skills "
                        "-> Enable auto-update, or " + f'{py["run_with"]} "{Path(__file__).resolve()}" '
                        "--enable-autoupdate (writes it to ~/.claude/settings.json, with a backup)")
    lst = rep["listing"]
    if lst.get("checked") and lst.get("crowded"):
        acts.append(f"crowded skill listing ({lst['skills']} skills in {lst['enabled_plugins']} enabled plugins): "
                    "offer a clean analysis folder, "
                    + f'{py["run_with"]} "{Path(__file__).resolve().parent / "workspace.py"}" <folder>, '
                    "which turns the other plugins off in that folder only")
    return acts


def python_install_hint(os_name=os.name, platform=sys.platform) -> str:
    """The system command that installs Python, for the permission question when none is usable."""
    if os_name == "nt":
        return "winget install --id Python.Python.3.13 -e (then open a new terminal; 'py -3.13' runs it)"
    if platform == "darwin":
        return "brew install python@3.13 (or the installer from python.org)"
    return "the system package manager, e.g. sudo apt install python3 python3-venv python3-pip"


def print_doctor(rep: dict) -> None:
    py = rep["python"]
    print(f"Python {py['version']} ({py['exe']}): " + ("ok" if py["supported"] else "TOO OLD"))
    print(f"Run scripts with: {py['run_with']}")
    for i in rep["interpreters"]:
        state = "all core packages" if not i["missing_core"] else f"missing {len(i['missing_core'])} core packages"
        print(f"  other Python {i['version']} ({i['exe']}): {state}")
    print("Core packages: " + ("all present" if not rep["core_missing"] else "missing or outdated: "
                               + ", ".join(rep["core_missing"])))
    print("Optional extras (installed only when a task needs them, with the user's permission):")
    for n, e in rep["extras"].items():
        state = "present" if not e["missing"] else "missing " + ", ".join(e["missing"])
        extra = []
        if e["heavy_mb"]:
            extra.append(f"large: ~{e['heavy_mb']} MB, ask on its own")
        if not e["minimum_tested"]:
            extra.append("minimum not tested in CI")
        extra += e["caveats"]
        print(f"  {n:9s} {state}" + (f"  [{'; '.join(extra)}]" if extra else ""))
    sk, lt = rep["skill"], rep["latest"]
    print(f"Skill in use: {sk['version']} ({sk['dir']})")
    print("Latest release: " + (lt["version"] if lt["version"] else lt["status"]))
    for c in rep["copies"]:
        print(f"  copy: {c['kind']} {c['version']} ({c['path']})")
    au = rep["autoupdate"]
    if au["registered"]:
        print("Plugin auto-update: " + ("on" if au["on"] else "off") + f" ({au['source']}"
              + (f"; blocked by {', '.join(au['blocked_by'])}" if au["blocked_by"] else "") + ")")
    lst = rep["listing"]
    if lst.get("checked"):
        print(f"Skill listing: {lst['skills']} skills in {lst['enabled_plugins']} enabled plugins"
              + (" (crowded)" if lst["crowded"] else "")
              + (f"; skillListingBudgetFraction {lst['budget_fraction']}" if lst.get("budget_fraction") else ""))
    else:
        print(f"Skill listing: not checked ({lst.get('reason')})")
    if rep["actions"]:
        print("\nTo do (ask the user once, listing all of these; install or change nothing before the answer):")
        for k, a in enumerate(rep["actions"], 1):
            print(f"  {k}. {a}")
    else:
        print("\nNothing to do: the setup is complete and up to date.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true")
    g.add_argument("--install", action="store_true")
    g.add_argument("--skill-dir", action="store_true")
    g.add_argument("--doctor", action="store_true")
    g.add_argument("--enable-autoupdate", action="store_true")
    ap.add_argument("--extra", action="append", default=[], choices=sorted(OPTIONAL),
                    help="also check/install an optional extra: " +
                         "; ".join(f"{k} = {OPTIONAL_USE[k]}" for k in OPTIONAL))
    ap.add_argument("--offline", action="store_true", help="--doctor: skip the latest-release check")
    ap.add_argument("--json", action="store_true", help="--doctor: print the report as JSON")
    a = ap.parse_args(argv)
    if a.skill_dir:
        print(recommended_skill_dir())
        return 0
    if a.enable_autoupdate:
        try:
            path, backup = enable_autoupdate(claude_dir())
        except ValueError as e:
            print(e)
            return 1
        print(f"Auto-update turned on for {MARKETPLACE} in {path}" + (f" (backup: {backup})" if backup else ""))
        print("It runs in the background after the first message of a session; restart Claude Code to load "
              "an update. /plugin -> Marketplaces shows the setting.")
        return 0
    if a.doctor:
        rep = doctor(offline=a.offline)
        if a.json:
            print(json.dumps(rep, indent=2, ensure_ascii=False))
        else:
            print_doctor(rep)
        return 0 if rep["python"]["supported"] and not rep["core_missing"] else 1
    if sys.version_info[:2] < MIN_PYTHON:
        report([])
        return 1
    reqs = missing_packages(a.extra)
    if a.install and reqs:
        skip = not_installable(reqs)
        for r in skip:
            print(f"Skipping {r}: {_caveat(r.split('>=')[0])[3]}")
        todo = [r for r in reqs if r not in skip]
        if todo:
            print("Installing: " + " ".join(todo))
            if not install(todo):
                return 1
        reqs = missing_packages(a.extra)
    report(reqs)
    skipped = not_installable(reqs)
    return 0 if not [r for r in reqs if r not in skipped] else 1


if __name__ == "__main__":
    sys.exit(main())
