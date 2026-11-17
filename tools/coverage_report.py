"""Line coverage of the test suite per public function, next to its VALIDATION.md level (v2.6.11).

Input: the JSON written by `coverage json` after running the tests under coverage (CI: the latest job on Python
3.13, see .github/workflows/tests.yml). Output: a Markdown report on stdout (CI appends it to the job summary).

What it shows:
  - per level (A, B, C, D): number of functions, how many the suite executes at all, and the share of their lines
    that run;
  - level-D functions the suite executes through other functions (no test refers to them by name, but their lines
    run): candidates for a direct test;
  - level A-C functions with no line executed (should not happen: the row says a test calls them);
  - every function with its level and covered lines.

What it does not show: whether the numbers are right (line coverage is not validation), lines run only in a child
process (tests that call a script through subprocess are not traced unless COVERAGE_PROCESS_START is set), and the
modules VALIDATION.md leaves out (_console, setup_env, workspace). There is no threshold: the script never fails the
build on a percentage.

Public functions are found the way tests/test_validation_md.py finds them: top-level `def` without a leading
underscore in econometrics/scripts/*.py. A function's lines are the executable lines (coverage's statements) after
its `def` line, so the `def` itself, which runs on import, does not count.

Run from the repository root:  python tools/coverage_report.py coverage.json [--all]
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "econometrics" / "scripts"
SKIP = {"_console.py", "setup_env.py", "workspace.py"}


def public_functions():
    """{module.function: (file name, first body line, last line)} for every public top-level function."""
    out = {}
    for f in sorted(SCRIPTS.glob("*.py")):
        if f.name in SKIP:
            continue
        for n in ast.parse(f.read_text(encoding="utf-8")).body:
            if isinstance(n, ast.FunctionDef) and not n.name.startswith("_"):
                out[f"{f.stem}.{n.name}"] = (f.name, n.body[0].lineno, n.end_lineno)
    return out


def levels():
    """{module.function: level} from the rows of VALIDATION.md."""
    text = (ROOT / "VALIDATION.md").read_text(encoding="utf-8")
    return {m.group(1): m.group(2) for m in re.finditer(r"^\| `([a-z_]+\.[A-Za-z0-9_]+)` \| ([ABCD]) \|", text, re.M)}


def file_lines(cov: dict):
    """{script file name: (executed lines, all executable lines)} from a coverage JSON."""
    out = {}
    for key, rec in cov.get("files", {}).items():
        p = key.replace("\\", "/")
        if "econometrics/scripts/" not in p and not p.startswith("scripts/"):
            continue
        name = p.rsplit("/", 1)[-1]
        executed = set(rec.get("executed_lines", []))
        out[name] = (executed, executed | set(rec.get("missing_lines", [])))
    return out


def per_function(cov: dict) -> list[dict]:
    lv, fl = levels(), file_lines(cov)
    rows = []
    for fn, (fname, first, last) in public_functions().items():
        executed, statements = fl.get(fname, (set(), set()))
        body = {ln for ln in statements if first <= ln <= last}
        run = body & executed
        rows.append({"function": fn, "level": lv.get(fn, "?"), "lines": len(body), "run": len(run),
                     "share": (len(run) / len(body)) if body else None, "traced": fname in fl})
    return rows


def report(rows: list[dict], show_all: bool = False) -> str:
    out = ["## Line coverage of the test suite, by VALIDATION.md level", "",
           "Lines of each public function executed by `tests/test_*.py` (child processes not traced). "
           "Coverage is not validation: the level says how a function's numbers were checked. No threshold.", "",
           "| level | functions | executed at all | lines run / lines |", "|---|---|---|---|"]
    for lv in "ABCD":
        rs = [r for r in rows if r["level"] == lv]
        lines, run = sum(r["lines"] for r in rs), sum(r["run"] for r in rs)
        share = f"{run}/{lines} ({run / lines:.0%})" if lines else "—"
        out.append(f"| {lv} | {len(rs)} | {sum(r['run'] > 0 for r in rs)} | {share} |")
    untraced = sorted({r["function"].split(".")[0] for r in rows if not r["traced"]})
    if untraced:
        out += ["", f"Modules not in the coverage data (never imported by the suite): {', '.join(untraced)}."]
    d_run = [r for r in rows if r["level"] == "D" and r["run"] > 0]
    out += ["", f"### Level D executed through other functions ({len(d_run)})", "",
            "No test refers to them by name, but the suite runs their lines: candidates for a direct test.", ""]
    out += [f"- `{r['function']}`: {r['run']}/{r['lines']} lines" for r in sorted(d_run, key=lambda r: -r["share"])]
    zero = [r for r in rows if r["level"] in ("A", "B", "C") and r["lines"] and r["run"] == 0]
    out += ["", f"### Level A-C with no line executed ({len(zero)})", ""]
    out += [f"- `{r['function']}` (level {r['level']})" for r in zero] or ["None."]
    if show_all:
        out += ["", "### All public functions", "", "| function | level | lines run / lines |", "|---|---|---|"]
        out += [f"| `{r['function']}` | {r['level']} | {r['run']}/{r['lines']} |" for r in rows]
    return "\n".join(out) + "\n"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    paths = [a for a in argv if not a.startswith("--")]
    if len(paths) != 1:
        print("usage: python tools/coverage_report.py coverage.json [--all]", file=sys.stderr)
        return 2
    cov = json.loads(Path(paths[0]).read_text(encoding="utf-8"))
    sys.stdout.write(report(per_function(cov), show_all="--all" in argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
