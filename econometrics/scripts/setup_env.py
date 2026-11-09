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
                                     --offline skips the release check (one request to api.github.com);
                                     --folder PATH measures the listing as Claude Code sees it in that
                                     folder (default: $CLAUDE_PROJECT_DIR, else the current folder): run it
                                     for the analysis folder, not from inside the skill's own folder.
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
import functools
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from importlib import metadata
from pathlib import Path

try:                     # Windows-safe console output (user paths with characters outside the code page)
    import _console  # noqa: F401
except ImportError:      # setup_env.py copied alone: make stdout tolerant here
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass

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
# their own, with the download size). The minimums of the extras in requirements-dev.txt are the lowest
# wheels that install with the core minimums (resolved 2026-11-06); CI's minimum job pins them
# (ci/constraints-min.txt) and tests/test_v2501.py imports each one there. UNTESTED_MINIMUM lists the extras
# CI does not install (torch-based), whose minimums are documented, not tested.
OPTIONAL = {
    "excel": {"openpyxl": ("openpyxl", "3.1")},
    "plots": {"matplotlib": ("matplotlib", "3.9.2")},      # 3.8 needs numpy < 2
    "rd": {"rdrobust": ("rdrobust", "1.3")},
    "regimes": {"hmmlearn": ("hmmlearn", "0.3.3"), "ruptures": ("ruptures", "1.1.9")},   # 0.3.2 mis-fits on numpy 2
    "panel": {"pydynpd": ("pydynpd", "0.2")},
    "ml": {"econml": ("econml", "0.16")},                  # 0.17 needs scikit-learn 1.6
    "pdf": {"pymupdf": ("pymupdf", "1.24.3")},             # first release importable as 'pymupdf'
    "ocr": {"easyocr": ("easyocr", "1.7")},
    "web": {"requests": ("requests", "2.31")},
    "text": {"sentence_transformers": ("sentence-transformers", "3.0")},
    "llm": {"transformers": ("transformers", "4.40"), "torch": ("torch", "2.2")},
    "data-br": {"bcb": ("python-bcb", "0.3"), "sidrapy": ("sidrapy", "0.1.4"), "ipeadatapy": ("ipeadatapy", "0.1.9")},
    "data-intl": {"wbgapi": ("wbgapi", "1.0.12"), "fredapi": ("fredapi", "0.5.1"), "yfinance": ("yfinance", "0.2.40"),
                  "pandas_datareader": ("pandas-datareader", "0.10")},
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
    "ruptures": ((3, 14), None, True, "has no release for Python 3.14 (the latest excludes it, older ones need a "
                                      "source build with a C compiler)"),
    "hmmlearn": ((3, 14), None, False, "has no wheel for Python 3.14: pip builds it from source, which needs a "
                                       "C++ compiler (Microsoft C++ Build Tools on Windows, Xcode tools on macOS)"),
}
SKILL_DIR = Path(__file__).resolve().parents[1]
STABLE_PLUGIN_DIR = (Path(os.environ["CLAUDE_CONFIG_DIR"]) if os.environ.get("CLAUDE_CONFIG_DIR")
                     else Path.home() / ".claude") / "plugins" / "marketplaces" / "econometric-skills" / "econometrics"

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


@functools.lru_cache(maxsize=None)
def _py_launcher_exe(tag: str):
    """The interpreter that the Windows 'py -X.Y' launcher starts, or None (cached: one call per run)."""
    py = shutil.which("py")
    if not py:
        return None
    try:
        r = subprocess.run([py, f"-{tag}", "-c", "import sys; print(sys.executable)"],
                           capture_output=True, text=True, errors="replace", timeout=20)
    except (OSError, subprocess.SubprocessError, ValueError):
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


def _load_json(path: Path, strict: bool = False):
    """A JSON object (dict) read from `path` (a UTF-8 BOM is accepted); {} when the file is missing, unreadable
    or not an object. strict=True returns None instead of {} for a file that exists but is not an object."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None if strict and Path(path).exists() else {}
    if isinstance(data, dict):
        return data
    return None if strict else {}


def _dict(x) -> dict:
    return x if isinstance(x, dict) else {}


_VERSION_RE = re.compile(r"^\d+(\.\d+)*$")


def _version_or_none(v):
    """A release-like version string ('2.5.0.1'), or None for anything else (a number, a commit SHA)."""
    return v if isinstance(v, str) and _VERSION_RE.match(v.strip()) else None


def read_skill_version(folder) -> str | None:
    """SKILL_VERSION of the copy of the skill in `folder` (from scripts/reporting.py), or None.

    The version, not a file hash, identifies a copy: the Claude app repackages the uploaded .skill file,
    so the same release has a different hash there (seen on 2026-11-06)."""
    try:
        text = (Path(folder) / "scripts" / "reporting.py").read_text(encoding="utf-8")
    except OSError:
        return None
    m = re.search(r'^SKILL_VERSION = "([^"]+)"', text, re.M)
    return _version_or_none(m.group(1)) if m else None


def skill_copies(cdir: Path) -> list[dict]:
    """Every copy of the skill Claude Code can see: the installed plugin, the marketplace checkout (the
    source of the next update), copies synced from the Claude app and manual copies under skills/.
    Copies in skills/.trash (removed from the app) are ignored, as Claude Code ignores them."""
    out = []
    inst = _load_json(cdir / "plugins" / "installed_plugins.json")
    entry = _dict(inst.get("plugins")).get(PLUGIN_ID)
    if isinstance(entry, list):                      # one record per scope
        entry = next((e for e in entry if isinstance(e, dict)), None)
    if isinstance(entry, dict) and isinstance(entry.get("installPath"), str) and entry["installPath"]:
        p = Path(entry["installPath"])
        out.append({"kind": "plugin", "path": str(p),
                    "version": read_skill_version(p) or _version_or_none(entry.get("version"))})
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
    return (tag, "ok") if _version_or_none(tag) else (None, "not checked (no version in the reply)")


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
    settings = _load_json(cdir / "settings.json")
    ekm = _dict(settings.get("extraKnownMarketplaces")).get(MARKETPLACE)
    known = _load_json(cdir / "plugins" / "known_marketplaces.json").get(MARKETPLACE)
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

    The entry keeps the marketplace's own source: the one already in settings, else the one Claude Code
    recorded in plugins/known_marketplaces.json (a fork or a local folder stays what it is). Raise ValueError,
    leaving everything untouched, when the marketplace is not registered or settings.json is not a JSON
    object with an object under extraKnownMarketplaces."""
    path = cdir / "settings.json"
    known = _load_json(cdir / "plugins" / "known_marketplaces.json").get(MARKETPLACE)
    settings = {}
    if path.exists():
        settings = _load_json(path, strict=True)
        if settings is None:
            raise ValueError(f"{path} is not a valid JSON object; fix it, or use /plugin -> Marketplaces -> "
                             f"{MARKETPLACE} -> Enable auto-update instead.")
    ekm = settings.get("extraKnownMarketplaces", {})
    if not isinstance(ekm, dict):
        raise ValueError(f"extraKnownMarketplaces in {path} is not an object; use /plugin -> Marketplaces instead.")
    entry = ekm.get(MARKETPLACE)
    if entry is not None and not isinstance(entry, dict):
        raise ValueError(f"the {MARKETPLACE} entry in {path} is not an object; use /plugin -> Marketplaces instead.")
    source = (entry or {}).get("source") or _dict(known).get("source")
    if not source:
        raise ValueError(f"the {MARKETPLACE} marketplace is not registered in Claude Code; add it first "
                         f"(claude plugin marketplace add {REPO}).")
    backup = None
    if path.exists():
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
        backup = path.with_name(f"settings.json.bak-{stamp}")
        k = 1
        while backup.exists():                       # never overwrite an earlier backup
            backup = path.with_name(f"settings.json.bak-{stamp}-{k}")
            k += 1
        shutil.copy2(path, backup)
    new_entry = dict(entry or {})
    new_entry["source"] = source
    new_entry["autoUpdate"] = True
    ekm = dict(ekm)
    ekm[MARKETPLACE] = new_entry
    settings = dict(settings)
    settings["extraKnownMarketplaces"] = ekm
    path.parent.mkdir(parents=True, exist_ok=True)
    real = path.resolve() if path.exists() else path  # a symlinked settings.json: write its target, keep the link
    tmp = real.with_name(f".{real.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(json.dumps(settings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if real.exists():
            shutil.copymode(real, tmp)               # keep the permissions (it may hold tokens under "env")
        os.replace(tmp, real)                        # atomic in the same folder
    finally:
        if tmp.exists():
            tmp.unlink()
    return path, backup


def _run(cmd, timeout: float, cwd=None) -> tuple[int, str]:
    """(exit code, stdout) of a command, never hanging past `timeout`.

    Output goes to a temporary file, not a pipe: on Windows a wrapper such as claude.cmd leaves a grandchild
    holding the pipe, and subprocess.run would wait for it after the timeout. On timeout the whole process
    tree is killed (taskkill /T on Windows). Decoding is UTF-8 with replacement, so no output can crash."""
    with tempfile.TemporaryFile() as out:
        try:
            p = subprocess.Popen(cmd, stdout=out, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, cwd=cwd)
        except OSError:
            return -1, ""
        try:
            code = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
            p.kill()
            p.wait()
            return -1, ""
        out.seek(0)
        return code, out.read().decode("utf-8", errors="replace")


_PROBE = ("import json,sys,importlib.util as u\n"
          "from importlib import metadata as m\n"
          "def f(n, d):\n"
          "    try:\n        return m.version(d) if u.find_spec(n) is not None else None\n"
          "    except Exception:\n        return None\n"
          "print(json.dumps({'exe': sys.executable, 'v': list(sys.version_info[:3]),"
          " 'ver': {n: f(n, d) for n, d in %r}}, ensure_ascii=True))")


def python_candidates(os_name=os.name, run=None, which=shutil.which) -> list[list[str]]:
    """Commands that start the other Pythons on the machine. On Windows, the tags listed by 'py -0p' (ASCII,
    such as -V:3.14) are started through the launcher, so a user folder with accents in its path never has to
    be decoded from the launcher's output; elsewhere, python3.X on the PATH. run(cmd, timeout) -> (code, out)."""
    run = run or _run
    found = []
    if os_name == "nt":
        py = which("py")
        if py:
            try:
                out = run([py, "-0p"], 20)[1] or ""
            except Exception:                        # never let a probe crash the doctor
                out = ""
            for line in out.splitlines():
                m = re.match(r"\s*-V:(\S+?)\s", line + " ") or re.match(r"\s*-(\d+\.\d+(?:-\d+)?)\s", line + " ")
                if m:
                    found.append([py, f"-V:{m.group(1)}"])
    else:
        seen = []
        for minor in range(MIN_PYTHON[1], 20):
            p = which(f"python3.{minor}")
            if p and not any(_same_file(p, q) for q in seen):
                seen.append(p)
                found.append([p])
    return found


def probe_interpreter(cmd, run=None) -> dict | None:
    """Executable and version of another interpreter and its core packages that are missing or below the
    minimums ('numpy>=2.1' strings), or None if it does not start. `cmd` starts it (a list, or one path).
    It runs from the temporary folder, so a module in the user's analysis folder (a json.py, say) cannot
    shadow the standard library there; its reply is ASCII-only JSON, so no path is lost in decoding."""
    run = run or (lambda c, timeout: _run(c, timeout, cwd=tempfile.gettempdir()))
    cmd = [cmd] if isinstance(cmd, str) else list(cmd)
    try:
        code, out = run(cmd + ["-c", _PROBE % ([(k, d) for k, (d, _m) in CORE.items()],)], 60)
        d = json.loads(out.strip().splitlines()[-1])
        vers = d["ver"]
        v = [int(x) for x in d["v"]]
        exe = str(d.get("exe") or cmd[0])
    except Exception:
        return None
    need = [f"{dist}>={m}" for imp, (dist, m) in CORE.items()
            if not vers.get(imp) or _vtuple(str(vers[imp])) < _vtuple(m)]
    return {"exe": exe, "version": ".".join(map(str, v)), "missing_core": need,
            "supported": tuple(v[:2]) >= MIN_PYTHON}


def _same_dir(a, b) -> bool:
    try:
        return os.path.normcase(str(Path(a).resolve())) == os.path.normcase(str(Path(b).resolve()))
    except (OSError, ValueError, TypeError):
        return False


def folder_settings(folder) -> dict:
    """Plugins a folder's own .claude/settings.json (and settings.local.json) turns off, as workspace.py writes."""
    off, on = set(), set()
    for name in ("settings.json", "settings.local.json"):
        ep = _load_json(Path(folder) / ".claude" / name).get("enabledPlugins")
        if isinstance(ep, dict):
            off |= {k for k, v in ep.items() if v is False}
            on |= {k for k, v in ep.items() if v is True}
    # workspace.py writes this skill true (KEEP) and every other enabled plugin false. One unrelated plugin
    # turned off by hand is NOT a prepared folder: that folder still needs the clean-folder offer, which the
    # "already prepared" item suppresses (v2.5.2 review).
    prepared = bool(off) and any(str(i).startswith("econometrics@") for i in on)
    return {"prepared": prepared, "plugins_off": len(off), "plugins_on": len(on)}


def skill_listing(cdir: Path, run=None, which=shutil.which, cwd=None) -> dict:
    """How crowded Claude Code's skill listing is in folder `cwd` (default: the current folder): enabled plugins
    and their skills, from `claude plugin list --json` run there so the folder's own settings count, plus skills
    under skills/. The CLI also lists, as enabled, project-scope installs made for OTHER folders (2.1.319): those
    are left out, and an install folder listed twice is counted once. A project-scope install of this skill for
    this folder is reported (`project_install`): it pins the folder to that version."""
    folder = Path(cwd) if cwd else Path.cwd()
    run = run or (lambda cmd, timeout: _run(cmd, timeout, cwd=str(folder)))
    settings = _load_json(cdir / "settings.json")
    res = {"checked": False, "budget_fraction": settings.get("skillListingBudgetFraction"),
           "folder": str(folder), **_safe(lambda: folder_settings(folder), lambda: {"prepared": False})}
    exe = which("claude")
    if exe is None:
        res["reason"] = "Claude Code CLI not on the PATH"
        return res
    try:
        code, out = run([exe, "plugin", "list", "--json"], 60)
        plugins = json.loads(out)
        if not isinstance(plugins, list):
            raise ValueError("not a list")
    except Exception:
        res["reason"] = "claude plugin list --json failed or timed out"
        return res
    enabled, seen, n_skills = [], set(), 0
    for p in plugins:
        if not (isinstance(p, dict) and p.get("enabled")):
            continue
        pp = p.get("projectPath")
        if p.get("scope") in ("project", "local") and isinstance(pp, str) and pp and not _same_dir(pp, folder):
            continue                                  # an install for another folder: not active here
        if str(p.get("id", "")).startswith("econometrics@") and p.get("scope") in ("project", "local") \
                and isinstance(pp, str) and _same_dir(pp, folder):
            res["project_install"] = {"version": p.get("version"), "scope": p.get("scope")}
        if str(p.get("id", "")).startswith("econometrics@") and p.get("scope") == "user":
            res["user_install"] = {"version": p.get("version")}      # what applies if the folder's own goes
        enabled.append(p)
        ip = Path(p["installPath"]) if isinstance(p.get("installPath"), str) and p["installPath"] else None
        key = os.path.normcase(str(ip)) if ip is not None else None
        if ip is not None and key not in seen and ip.is_dir():
            seen.add(key)
            n_skills += sum(1 for _ in ip.glob("**/SKILL.md"))
    sk = cdir / "skills"
    if sk.is_dir():
        n_skills += sum(1 for f in sk.glob("**/SKILL.md") if ".trash" not in f.relative_to(sk).parts)
    ids = {p.get("id") for p in enabled}
    res.update(checked=True, enabled_plugins=len(ids), skills=n_skills,
               crowded=n_skills >= CROWDED_SKILLS,
               others_enabled=sorted(str(i) for i in ids if i and not str(i).startswith("econometrics@")))
    return res


def _safe(fn, default):
    """Run one probe; any error becomes the default plus a 'not checked' note: the doctor never crashes."""
    try:
        return fn()
    except Exception as e:                           # a malformed file, a strange PATH, an OS error
        d = default() if callable(default) else default
        if isinstance(d, dict):
            d["error"] = f"not checked ({type(e).__name__}: {e})"
        return d


def doctor(offline: bool = False, *, cdir=None, env=None, opener=None, run=None, which=shutil.which,
           os_name=os.name, cwd=None) -> dict:
    """The full report (a dict). Installs, writes and changes nothing; the only network access is one
    request to the public GitHub releases endpoint (none with offline=True). `cwd` is the analysis folder whose
    skill listing is measured (default: $CLAUDE_PROJECT_DIR when set, else the current folder)."""
    cdir = Path(cdir) if cdir else claude_dir(env)
    if cwd is None:
        cwd = (os.environ if env is None else env).get("CLAUDE_PROJECT_DIR") or None
    here = sys.version_info[:2] >= MIN_PYTHON
    rep = {
        "claude_dir": str(cdir),
        "python": {"exe": sys.executable, "version": sys.version.split()[0], "supported": here,
                   "run_with": _safe(lambda: run_command(which=which, os_name=os_name), sys.executable)},
        "core_missing": missing_packages() if here else [f"{d}>={m}" for d, m in CORE.values()],
        "extras": {n: {"use": OPTIONAL_USE[n], "missing": missing_packages([n], core=False),
                       "heavy_mb": HEAVY_MB.get(n), "minimum_tested": n not in UNTESTED_MINIMUM,
                       "caveats": extra_caveats(n, os_name=os_name)} for n in OPTIONAL},
        "skill": {"dir": str(SKILL_DIR), "version": read_skill_version(SKILL_DIR)},
        "copies": _safe(lambda: skill_copies(cdir), list),
        "autoupdate": _safe(lambda: marketplace_autoupdate(cdir, env),
                            lambda: {"registered": False, "on": False, "configured": False, "source": "unknown",
                                     "blocked_by": []}),
        "listing": _safe(lambda: skill_listing(cdir, run=run, which=which, cwd=cwd), lambda: {"checked": False}),
    }
    others = []
    for cmd in _safe(lambda: python_candidates(os_name=os_name, run=run, which=which), list):
        info = _safe(lambda: probe_interpreter(cmd, run=run), None)
        if info and not _same_file(info["exe"], sys.executable) \
                and not any(_same_file(info["exe"], o["exe"]) for o in others):
            # the command the user types to start it ('py -3.14' rather than a long path), for the To do items
            info["run_with"] = _safe(lambda: run_command(exe=info["exe"], which=which, os_name=os_name,
                                                         version=_vtuple(info["version"])), info["exe"])
            others.append(info)
    rep["interpreters"] = others
    if offline:
        rep["latest"] = {"version": None, "status": "not checked (--offline)"}
    else:
        v, status = latest_release(opener=opener)
        rep["latest"] = {"version": v, "status": status}
    rep["actions"] = _safe(lambda: doctor_actions(rep, os_name=os_name), list)
    return rep


def _update_steps() -> str:
    return ("claude plugin marketplace update econometric-skills, then claude plugin update "
            f"{PLUGIN_ID} (or /plugin -> Marketplaces -> econometric-skills -> u), then restart Claude Code")


def doctor_actions(rep: dict, os_name=os.name) -> list[str]:
    """What to put in the single question to the user, in order. Empty when nothing is needed."""
    acts = []
    py = rep["python"]
    complete = [i for i in rep.get("interpreters", []) if i["supported"] and not i["missing_core"]]
    best = max(complete, key=lambda i: _vtuple(i["version"])) if complete else None
    # the commands in the items below go to the user: when this Python is too old or incomplete and another one
    # has every core package, they use that one, so the user keeps a single interpreter (v2.5.0.1 NIT)
    use = (best.get("run_with") or best["exe"]) if best and (not py["supported"] or rep["core_missing"]) \
        else py["run_with"]
    if not py["supported"]:
        acts.append(f"Python {py['version']} is too old (the skill needs {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+): "
                    + (f"use {best['exe']} (Python {best['version']}, packages present; run scripts with {use})"
                       if best
                       else "install a current Python: " + python_install_hint(os_name)))
    elif rep["core_missing"]:
        if best:
            acts.append(f"this Python ({py['version']}) lacks {', '.join(rep['core_missing'])}, but "
                        f"{best['exe']} (Python {best['version']}) has every core package: run the scripts with "
                        f"that one ({use}) instead of installing again")
        else:
            acts.append("install the core packages: " + ", ".join(rep["core_missing"])
                        + f"  ({install_command()})")
    latest = rep["latest"]["version"]
    n_before_updates = len(acts)
    if latest:
        # every copy Claude can load, plus the one running if it is none of them (a clone, say); the
        # marketplace checkout is only the source of the next plugin update, so it is not one of them
        loaded = [c for c in rep["copies"] if c["kind"] != "marketplace checkout"]
        here = rep["skill"]
        in_cache = "cache" in Path(here["dir"]).parts and any(
            part == MARKETPLACE for part in Path(here["dir"]).parts)
        plugin = next((c for c in loaded if c["kind"] == "plugin"), None)
        if (here["version"] and in_cache and plugin and plugin["version"]
                and _vtuple(plugin["version"]) > _vtuple(here["version"])
                and not _same_file(plugin["path"], here["dir"])):
            # an update was installed (auto-update, say) but this session still runs the old cache folder
            acts.append(f"the plugin {plugin['version']} is installed but this session still runs "
                        f"{here['version']}: restart Claude Code to load it")
        elif here["version"] and not in_cache and not any(_same_file(c["path"], here["dir"]) for c in rep["copies"]):
            loaded.append({"kind": "copy in use", "version": here["version"], "path": here["dir"]})
        for c in loaded:
            if not c["version"] or _vtuple(latest) <= _vtuple(c["version"]):
                continue
            if c["kind"] == "plugin":
                acts.append(f"update the plugin from {c['version']} to {latest}: {_update_steps()}")
            elif c["kind"] == "Claude app (synced)":
                acts.append(f"the Claude app copy is {c['version']}, the latest is {latest}: in the Claude app, "
                            f"Customize -> Skills, upload {RELEASE_ASSET} (it reaches Claude Code at its next start)")
            else:
                acts.append(f"the {c['kind']} at {c['path']} is {c['version']}, the latest is {latest}: replace it "
                            f"with the release ({RELEASE_ASSET}) or update the clone (git pull)")
    vers = {c["version"] for c in rep["copies"] if c["kind"] != "marketplace checkout" and c["version"]}
    if len(vers) > 1 and len(acts) == n_before_updates:     # not already covered by an update item above
        desc = "; ".join(f"{c['kind']} {c['version']}" for c in rep["copies"] if c["kind"] != "marketplace checkout")
        app = any(c["kind"] == "Claude app (synced)" for c in rep["copies"])
        acts.append(f"copies of the skill at different versions ({desc}): keep them equal"
                    + (f" (the Claude app copy: Customize -> Skills, upload {RELEASE_ASSET}, or remove it there "
                       "and keep only the plugin)" if app else ""))
    au = rep["autoupdate"]
    if au["registered"] and not au["on"]:
        if au["blocked_by"]:
            acts.append("plugin auto-update is off for every marketplace because "
                        + ", ".join(au["blocked_by"]) + " is set; update by hand at each release: "
                        + _update_steps())
        else:
            acts.append("plugin auto-update is off for econometric-skills (default for third-party "
                        "marketplaces). With the user's permission: /plugin -> Marketplaces -> econometric-skills "
                        "-> Enable auto-update, or " + f'{use} "{Path(__file__).resolve()}" '
                        "--enable-autoupdate (writes it to " + str(Path(rep.get("claude_dir", "~/.claude")) / "settings.json")
                        + ", with a backup)")
    lst = rep["listing"]
    pin = lst.get("project_install")
    usr = (lst.get("user_install") or {}).get("version")
    ref = rep["latest"].get("version") or rep["skill"].get("version")
    if pin and pin.get("version") and ref and _vtuple(str(pin["version"])) < _vtuple(str(ref)):
        # Uninstalling the folder's own install only helps when a user-scope install is there to take over and
        # is newer; with no user-scope install that command would remove the skill from the folder
        # (v2.5.2 review). Then the fix is to update it in place.
        if usr and _vtuple(str(usr)) > _vtuple(str(pin["version"])):
            acts.append(f"this folder ({lst.get('folder')}) has its own {pin.get('scope')}-scope install of the "
                        f"skill, pinned at {pin['version']}, while the user-scope install is {usr}: sessions here "
                        "keep loading the old version. With the user's permission, inside the folder: claude "
                        f"plugin uninstall {PLUGIN_ID} --scope {pin.get('scope')} (the {usr} user-scope install "
                        "then applies), then restart Claude Code")
        else:
            acts.append(f"this folder ({lst.get('folder')}) has its own {pin.get('scope')}-scope install of the "
                        f"skill, pinned at {pin['version']} (current {ref}), and it is the only install that "
                        "applies here: sessions here keep loading the old version. With the user's permission, "
                        f"inside the folder: claude plugin update {PLUGIN_ID} --scope {pin.get('scope')}, then "
                        "restart Claude Code (do not uninstall it: there is no user-scope install to take over)")
    if lst.get("checked") and lst.get("crowded") and lst.get("prepared"):
        acts.append(f"skill listing still crowded in this folder although it is already prepared "
                    f"({lst.get('plugins_off')} plugins off in its .claude/settings.json): the {lst['skills']} skills "
                    "come from the plugins still on and from ~/.claude/skills; do not offer another clean folder")
    elif lst.get("checked") and lst.get("crowded"):
        acts.append(f"crowded skill listing ({lst['skills']} skills in {lst['enabled_plugins']} enabled plugins): "
                    "offer a clean analysis folder, "
                    + f'{use} "{Path(__file__).resolve().parent / "workspace.py"}" <folder>, '
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
        print(f"  other Python {i['version']} ({i['exe']}): {state}"
              + (f"; run with {i['run_with']}" if i.get("run_with") and not i["missing_core"] else ""))
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
        print(f"Skill listing in {lst.get('folder')}: {lst['skills']} skills in {lst['enabled_plugins']} enabled plugins"
              + (" (crowded)" if lst["crowded"] else "")
              + (f"; folder prepared ({lst.get('plugins_off')} plugins off here)" if lst.get("prepared") else "")
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
    ap.add_argument("--folder", help="--doctor: the analysis folder whose skill listing is measured "
                                     "(default: $CLAUDE_PROJECT_DIR, else the current folder)")
    a = ap.parse_args(argv)
    if (a.json or a.offline or a.folder) and not a.doctor:
        ap.error("--json, --offline and --folder go with --doctor")
    if a.folder and not Path(a.folder).is_dir():
        ap.error(f"--folder: {a.folder} is not a folder")
    if a.skill_dir:
        print(recommended_skill_dir())
        return 0
    if a.enable_autoupdate:
        cdir = claude_dir()
        try:
            path, backup = enable_autoupdate(cdir)
        except (ValueError, OSError) as e:
            print(f"Auto-update not changed: {e}")
            return 1
        print(f"Auto-update turned on for {MARKETPLACE} in {path}" + (f" (backup: {backup})" if backup else ""))
        blocked = marketplace_autoupdate(cdir)["blocked_by"]
        if blocked:
            print(f"  note: {', '.join(blocked)} is set, which keeps every plugin auto-update off until it is unset "
                  "(or FORCE_AUTOUPDATE_PLUGINS=1).")
        print("Claude Code checks for updates in the background after the first message of a session and "
              "loads them at the next start. /plugin -> Marketplaces shows the setting.")
        return 0
    if a.doctor:
        rep = doctor(offline=a.offline, cwd=a.folder)
        if a.json:
            print(json.dumps(rep, indent=2, ensure_ascii=True))
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
