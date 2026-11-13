"""VALIDATION.md stays in sync with the code, the tests and tools/validation_diffs.json (v2.6.7).

What is checked:
  1. every public function of econometrics/scripts/ (AST; _console, setup_env and workspace left out) has exactly one
     row, and no row names a function that does not exist;
  2. levels are A, B, C or D, and the summary counts (overall and per module) are those of the rows;
  3. level D if and only if no test refers to the function: a test refers to it through `alias.function`, where alias
     is the module imported in that test, or by importing the function itself (AST, so a word in a comment or a
     function of the same name in another module does not count);
  4. for levels A-C, every test file in the row exists and refers to the function (so when a function gains or
     loses a test, its row has to change), and the check named after each file is in that file (the name up to a
     placeholder such as <case>);
  5. every repository path cited in a row (tools/..., tests/data/...) exists;
  6. the "largest observed difference" column: a row has a number there exactly when the function is in
     tools/validation_diffs.json, the number is the one measured, and every measured comparison is within its
     tolerance.

Run from the repository root:  python3 tests/test_validation_md.py
"""
import ast
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  [{info}]"))


SKIP = {"_console.py", "setup_env.py", "workspace.py"}
funcs = []
for f in sorted((ROOT / "econometrics" / "scripts").glob("*.py")):
    if f.name in SKIP:
        continue
    for n in ast.parse(f.read_text(encoding="utf-8")).body:
        if isinstance(n, ast.FunctionDef) and not n.name.startswith("_"):
            funcs.append(f"{f.stem}.{n.name}")
modules = {k.split(".")[0] for k in funcs}

# references from the tests, by AST
refs = defaultdict(set)
for p in sorted((ROOT / "tests").glob("test_*.py")):
    tree = ast.parse(p.read_text(encoding="utf-8"))
    alias, direct = {}, {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name in modules:
                    alias[a.asname or a.name] = a.name
        elif isinstance(node, ast.ImportFrom) and node.module in modules:
            for a in node.names:
                direct[a.asname or a.name] = f"{node.module}.{a.name}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in alias:
            refs[f"{alias[node.value.id]}.{node.attr}"].add(p.name)
        elif isinstance(node, ast.Name) and node.id in direct:
            refs[direct[node.id]].add(p.name)

text = (ROOT / "VALIDATION.md").read_text(encoding="utf-8")
rows = {}
dups = []
for line in text.splitlines():
    m = re.match(r"^\| `([a-z_]+\.[A-Za-z0-9_]+)` \| ([ABCD]|[^|]*) \|", line)
    if not m:
        continue
    cells = [c.strip() for c in re.split(r"(?<!\\)\|", line)[1:-1]]
    if m.group(1) in rows:
        dups.append(m.group(1))
    rows[m.group(1)] = cells

# 1. one row per function
check(f"every public function has a row ({len(funcs)} functions)", not [k for k in funcs if k not in rows],
      [k for k in funcs if k not in rows][:10])
check("no row names a function that does not exist", not [k for k in rows if k not in funcs],
      [k for k in rows if k not in funcs][:10])
check("no function has two rows", not dups, dups)
check("each row has 7 cells (function, level, reference, tolerance, difference, test, note)",
      all(len(c) == 7 for c in rows.values()), [k for k, c in rows.items() if len(c) != 7][:5])

# 2. levels and counts
levels = {k: c[1] for k, c in rows.items()}
check("levels are A, B, C or D", set(levels.values()) <= set("ABCD"), set(levels.values()))
cnt = Counter(levels[k] for k in funcs if k in levels)
check("summary line: the overall counts are those of the rows",
      f"{len(funcs)} public functions: level A {cnt['A']}, level B {cnt['B']}, level C {cnt['C']}, "
      f"level D {cnt['D']}." in text, dict(cnt))
bad_mod = []
for mod in sorted(modules):
    c = Counter(levels[k] for k in funcs if k.startswith(mod + ".") and k in levels)
    if f"| `{mod}` | {c['A']} | {c['B']} | {c['C']} | {c['D']} |" not in text:
        bad_mod.append(mod)
check("summary table: the counts per module are those of the rows", not bad_mod, bad_mod)

# 3. level D if and only if no test refers to the function
d_wrong = [k for k in funcs if k in levels and (levels[k] == "D") != (not refs.get(k))]
check("level D exactly for the functions no test refers to (a function that gains a test must change level)",
      not d_wrong, [(k, levels[k], sorted(refs.get(k, []))) for k in d_wrong][:8])

# 4. cited test files exist and refer to the function
bad_cite = []
for k, c in rows.items():
    if c[1] == "D":
        continue
    cited = set(re.findall(r"\b(test_[A-Za-z0-9_]+\.py)\b", c[5]))
    if not cited:
        bad_cite.append((k, "no test file cited"))
    for t in cited:
        if not (ROOT / "tests" / t).exists():
            bad_cite.append((k, f"{t} does not exist"))
        elif t not in refs.get(k, set()):
            bad_cite.append((k, f"{t} does not refer to it"))
check("levels A-C: each cited test file exists and refers to the function", not bad_cite, bad_cite[:8])
bad_name = []
for k, c in rows.items():
    for seg in re.split(r";\s*(?=test_[A-Za-z0-9_]+\.py)", c[5]):
        m = re.match(r"(test_[A-Za-z0-9_]+\.py)(?::\s*(.*))?$", seg.strip().replace("\\|", "|"))
        if not m or not m.group(2) or not (ROOT / "tests" / m.group(1)).exists():
            continue
        name = re.split(r"[<…]|\(called", m.group(2))[0].strip().rstrip(":").strip()
        src = (ROOT / "tests" / m.group(1)).read_text(encoding="utf-8")
        if len(name) >= 4 and name not in src:
            bad_name.append((k, m.group(1), name))
check("levels A-C: the check named after each test file is in that file", not bad_name, bad_name[:8])

# 5. cited repository paths exist
paths = set()
for c in rows.values():
    paths.update(re.findall(r"\b((?:tools|tests/data)/[A-Za-z0-9_./-]+\.(?:json|R|py|csv|txt))", " ".join(c)))
missing = sorted(p for p in paths if not (ROOT / p).exists())
check(f"every repository path cited in a row exists ({len(paths)} paths)", not missing, missing)

# 6. the measured differences
diffs = json.loads((ROOT / "tools" / "validation_diffs.json").read_text(encoding="utf-8"))["functions"]


def fmt(x):
    return "0" if x == 0 else f"{x:.2g}"


cell_bad = []
for k, c in rows.items():
    if k in diffs:
        s = diffs[k]
        want = (f"{fmt(s['worst_diff'])}{' (relative)' if s['kind'] == 'rel' else ''} of {fmt(s['tolerance'])}"
                if s["tolerance"] else f"{fmt(s['worst_diff'])} (exact)")
        if c[1] != "A" or c[4] != want:
            cell_bad.append((k, c[1], c[4], want))
    elif c[1] == "A" and c[4] not in ("not measured (closed form in the test)",
                                      "not measured (software run in the test)"):
        cell_bad.append((k, c[4]))
    elif c[1] != "A" and c[4] != "—":
        cell_bad.append((k, c[4]))
check("difference column: the measured value for the functions in tools/validation_diffs.json, level A only",
      not cell_bad, cell_bad[:6])
check("every measured comparison is within the tolerance of its test",
      all(s["all_within_tolerance"] for s in diffs.values()),
      [k for k, s in diffs.items() if not s["all_within_tolerance"]])

passed = sum(ok)
print(f"\n{passed}/{len(ok)} checks passed")
sys.exit(0 if passed == len(ok) else 1)
