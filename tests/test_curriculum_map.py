"""The curriculum map in references/teaching.md (section 6) cannot go stale.

Teacher mode picks the function behind each method card from that table. Every `module.function`
named there must exist in scripts/, and every `file.md §N` must be a numbered section of that
reference. Also: every public function of inference.py and every demo in simulation.py is
reachable from teaching.md, and the map covers the three blocks (statistics, econometrics,
time series).

Run from the repository root:  python3 tests/test_curriculum_map.py
"""
import importlib
import inspect
import re
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "econometrics"
sys.path.insert(0, str(SKILL / "scripts"))
warnings.filterwarnings("ignore")

ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


teach = (SKILL / "references" / "teaching.md").read_text(encoding="utf-8")
m = re.search(r"^## 6\. Curriculum map\n(.*?)(?=^## |\Z)", teach, re.S | re.M)
check("teaching.md has a section 6, Curriculum map", m is not None)
body = m.group(1) if m else ""
rows = [ln for ln in body.splitlines() if ln.startswith("| ") and not ln.startswith("| Topic") and "---" not in ln]
check("the map has at least 40 topics", len(rows) >= 40, f"{len(rows)} rows")

bad_func, bad_sec, empty = [], [], []
modules = {}
for ln in rows:
    cells = [c.strip() for c in ln.strip("|").split("|")]
    tokens = re.findall(r"`([^`]+)`", cells[-1])
    if not tokens:
        empty.append(cells[0])
    for tok in tokens:
        fm = re.fullmatch(r"(\w+)\.(\w+)", tok)
        sm = re.fullmatch(r"([\w-]+\.md) §(\d+)", tok)
        if fm:
            mod, fn = fm.groups()
            try:
                modules.setdefault(mod, importlib.import_module(mod))
                if not callable(getattr(modules[mod], fn, None)):
                    bad_func.append(tok)
            except ModuleNotFoundError:
                bad_func.append(tok)
        elif sm:
            f, sec = sm.groups()
            p = SKILL / "references" / f
            if not p.exists() or not re.search(rf"^## {sec}\. ", p.read_text(encoding="utf-8"), re.M):
                bad_sec.append(tok)
        else:
            bad_func.append(f"unparsed: {tok}")
check("every topic points somewhere", not empty, str(empty))
check("every function named in the map exists", not bad_func, str(bad_func))
check("every section named in the map exists", not bad_sec, str(bad_sec))

blocks = {re.sub(r"\s", "", ln.strip("|").split("|")[1]) for ln in rows}
check("the map covers statistics (0), econometrics (1-8) and time series (TS)",
      "0" in blocks and "TS" in blocks and any(b[0].isdigit() and b != "0" for b in blocks), str(sorted(blocks)))

import inference  # noqa: E402
import simulation  # noqa: E402

public = [n for n, f in inspect.getmembers(inference, inspect.isfunction)
          if not n.startswith("_") and f.__module__ == "inference"]
unmapped = [n for n in public if f"inference.{n}" not in body]
check("every public inference function is in the map", not unmapped, str(unmapped))
demos = [n for n, _ in inspect.getmembers(simulation, inspect.isfunction) if n.startswith("demo_")]
untaught = [n for n in demos if n not in teach]
check("every simulation demo is named in teaching.md", not untaught, str(untaught))

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
