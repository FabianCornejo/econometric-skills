"""Tests for v2.3.4: the skill's console output cannot stop an analysis on Windows.

On Windows, when a script's output is captured (Claude Code running a delivered script, a pipe, a
log file), Python encodes stdout with the ANSI code page, usually cp1252, and print() of an arrow
raises UnicodeEncodeError in the middle of the run. The modules print such characters in their
summaries. The fix is scripts/_console.py, imported by those modules.

Checks: every module whose print() calls hold a character cp1252 cannot encode imports _console;
so does every test or worked example with such a character in its strings (directly, or through
examples/_setup.py); with stdout forced to strict cp1252 (on any OS, so the Linux CI catches a regression too), each of
those modules imports and prints an arrow without error, and causal.propensity_score prints its
summary and exits 0; with a UTF-8 stream the arrow is printed unchanged; a strict UTF-8 stream keeps
errors="strict", a strict cp1252 stream gets errors="replace"; streams without reconfigure
(StringIO, notebooks) are left alone.

Run from the repository root:  python3 tests/test_windows_console.py
"""
import ast
import io
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "econometrics" / "scripts"
sys.path.insert(0, str(SCRIPTS))

ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


def unencodable_in_prints(tree):
    """Characters in print() string literals that cp1252 cannot encode."""
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "print":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    found |= {c for c in sub.value if c.encode("cp1252", "ignore") == b""}
    return found


def imports_console(tree):
    return any(isinstance(n, ast.Import) and any(a.name == "_console" for a in n.names) for n in tree.body)


def run(code, encoding):
    """Run code in a fresh Python whose stdout and stderr use `encoding` with strict errors."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONUTF8", "PYTHONIOENCODING")}
    env["PYTHONIOENCODING"] = encoding  # no ":errors" part -> stdout errors are "strict"
    env["PYTHONUTF8"] = "0"
    env["PYTHONWARNINGS"] = "ignore"
    prelude = f"import sys; sys.path.insert(0, {str(SCRIPTS)!r})\n"
    return subprocess.run([sys.executable, "-c", prelude + code], capture_output=True, env=env, cwd=str(ROOT))


def last_line(b):
    lines = b.decode("utf-8", "replace").strip().splitlines()
    # ASCII only: this report must not hit the very error the file tests for
    return lines[-1][:110].encode("ascii", "backslashreplace").decode("ascii") if lines else ""


# ---------------------------------------------------------------- 1. which modules need the fix
affected, missing = [], []
for p in sorted(SCRIPTS.glob("*.py")):
    if p.name.startswith("_"):
        continue
    tree = ast.parse(p.read_text(encoding="utf-8"))
    if unencodable_in_prints(tree):
        affected.append(p.stem)
        if not imports_console(tree):
            missing.append(p.name)
check(f"modules that print characters cp1252 cannot encode import _console ({len(affected)} modules)",
      affected and not missing, ("missing: " + ", ".join(missing)) if missing else "")


def unencodable_strings(tree):
    """Characters cp1252 cannot encode in any string literal except docstrings (test names go
    through check(), not print(), so the print() scan above would miss them)."""
    docs = {id(n.body[0].value) for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body
            and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    return {c for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in docs for c in n.value if c.encode("cp1252", "ignore") == b""}


def uses_setup(tree):
    return any(isinstance(n, ast.ImportFrom) and n.module == "_setup" for n in tree.body)


setup_ok = imports_console(ast.parse((ROOT / "examples" / "_setup.py").read_text(encoding="utf-8")))
check("examples/_setup.py imports _console (every worked example goes through it)", setup_ok)
loose = []
for p in sorted([*(ROOT / "tests").glob("test_*.py"), *(ROOT / "examples").glob("0*.py")]):
    if p.name == Path(__file__).name:  # compares such characters, never prints them
        continue
    tree = ast.parse(p.read_text(encoding="utf-8"))
    if unencodable_strings(tree) and not (imports_console(tree) or (uses_setup(tree) and setup_ok)):
        loose.append(p.name)
check("tests and examples with such characters in their strings import _console (directly or via _setup)",
      not loose, ("missing: " + ", ".join(loose)) if loose else "")

# ---------------------------------------------------------------- 2. strict cp1252 output
for m in affected:
    r = run(f"import {m}\nprint('\\u2192 ok')\n", "cp1252")
    check(f"strict cp1252 stdout: import {m}, print an arrow, exit 0",
          r.returncode == 0 and b"? ok" in r.stdout, last_line(r.stderr))

real = """
import numpy as np, pandas as pd
import causal
rng = np.random.default_rng(0)
n = 400
x1, x2 = rng.normal(size=n), rng.normal(size=n)
t = (rng.normal(size=n) + 1.5 * x1 > 0).astype(int)
causal.propensity_score(pd.DataFrame({"t": t, "x1": x1, "x2": x2}), "t", ["x1", "x2"])
print("done")
"""
for spec in ("cp1252", "cp1252:surrogateescape"):  # the second is what a PowerShell pipe gives
    r = run(real, spec)
    check(f"{spec} stdout: causal.propensity_score prints its overlap summary and exits 0",
          r.returncode == 0 and b"overlap" in r.stdout and b"done" in r.stdout, last_line(r.stderr))

# ---------------------------------------------------------------- 3. UTF-8 output is unchanged
r = run("import causal\nprint('\\u2192 ok')\n", "utf-8")
check("UTF-8 stdout: the arrow is printed unchanged",
      r.returncode == 0 and "\u2192 ok" in r.stdout.decode("utf-8", "replace"), last_line(r.stderr))

# ---------------------------------------------------------------- 4. the helper itself
try:
    import _console
except ImportError as e:
    check("scripts/_console.py exists and imports", False, str(e))
else:
    saved = sys.stdout
    try:
        results = {}
        # surrogateescape is what a PowerShell pipe gives on Windows; it fails on an arrow like strict
        for enc, errors in (("utf-8", "strict"), ("cp1252", "strict"), ("cp1252", "surrogateescape"),
                            ("cp1252", "backslashreplace")):
            w = io.TextIOWrapper(io.BytesIO(), encoding=enc, errors=errors)
            sys.stdout = w
            _console.make_output_tolerant()
            results[enc, errors] = w.errors
        sys.stdout = io.StringIO()
        _console.make_output_tolerant()
        sys.stdout.write("\u2192")
        stringio_value = sys.stdout.getvalue()
    finally:
        sys.stdout = saved
    for (enc, errors), expected in ((("utf-8", "strict"), "strict"), (("cp1252", "strict"), "replace"),
                                    (("cp1252", "surrogateescape"), "replace"),
                                    (("cp1252", "backslashreplace"), "backslashreplace")):
        got = results[enc, errors]
        check(f"a {enc} stream with errors='{errors}' ends with errors='{expected}'", got == expected, got)
    check("streams without reconfigure (StringIO, notebooks) are left alone", stringio_value == "\u2192")

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
