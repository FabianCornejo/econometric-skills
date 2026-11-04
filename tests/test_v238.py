"""Tests for v2.3.8: delivered scripts ignore skills Claude Code moved to its trash, and quantile_regression
says plainly when no bootstrap was asked for.

Found in the v2.3.7 validation on Windows. (1) Removing the skill in the Claude app moves Claude Code's
synced copy to ~/.claude/skills/.trash/<id>/econometrics. The script skeleton's locator searched
~/.claude/skills/**/econometrics newest first, so for a user with only the app copy a delivered script
could run a removed version, or an older one moved to the trash after a newer upload. (2) With B=0,
quantile_regression warned that "an end tau or the bootstrap did not converge", which is false.

Checks: the skeleton skips hidden folders (.trash) under ~/.claude/skills, picks the active synced copy
even when the trashed one is newer, and stops with the one-line "not found" message when only a trashed
copy exists; with B=0 the equality test is not computed, says why, and does not mention convergence;
with B>0 and converged taus the test stays valid.

Run from the repository root:  python3 tests/test_v238.py
"""
import contextlib
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "econometrics"
sys.path.insert(0, str(SKILL / "scripts"))
warnings.filterwarnings("ignore")
import linear_models as lm

ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


# ---------------------------------------------------------------- 1. the locator skips the trash
rules = (SKILL / "references" / "interpretation-rules.md").read_text(encoding="utf-8")
block = re.search(r"Script skeleton for every deliverable.*?```python\n(.*?)```", rules, re.S).group(1)
probe = block + "\nprint('FOUND', SKILL_DIR)\n"


def run(home):
    env = {k: v for k, v in os.environ.items() if k != "ECONOMETRICS_SKILL_DIR"}
    env["HOME"] = env["USERPROFILE"] = str(home)
    with tempfile.TemporaryDirectory() as wd:
        f = Path(wd) / "analysis.py"
        f.write_text(probe, encoding="utf-8")
        return subprocess.run([sys.executable, str(f)], capture_output=True, text=True, cwd=wd, env=env)


def copy_skill(dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SKILL, dst, ignore=shutil.ignore_patterns("__pycache__"))


with tempfile.TemporaryDirectory() as tmp:
    home = Path(tmp) / "home"
    synced = home / ".claude" / "skills" / "synced" / "a1b2" / "econometrics"
    trashed = home / ".claude" / "skills" / ".trash" / "1793801808064-x" / "econometrics"
    copy_skill(synced)
    copy_skill(trashed)
    now = time.time()                         # copytree keeps the source mtimes: set them explicitly
    os.utime(synced / "scripts" / "causal.py", (now - 3600, now - 3600))
    os.utime(trashed / "scripts" / "causal.py", (now, now))   # trashed copy is the newest, as after a removal
    r = run(home)
    check("active synced copy and a newer trashed one: the skeleton uses the active copy",
          r.returncode == 0 and f"FOUND {synced}" in r.stdout, (r.stdout + r.stderr).strip()[-110:])
    only_trash = Path(tmp) / "only_trash"
    copy_skill(only_trash / ".claude" / "skills" / ".trash" / "999-y" / "econometrics")
    r = run(only_trash)
    check("only a trashed copy: 'skill not found' in one line, the removed version does not run",
          r.returncode != 0 and "Econometrics skill not found" in r.stderr and "Traceback" not in r.stderr,
          r.stderr.strip()[:110])
check("the skeleton documents that hidden folders such as .trash are skipped",
      ".trash" in block and "hidden" in block.lower())

# ---------------------------------------------------------------- 2. B=0 is not a convergence problem
rng = np.random.default_rng(3)
n = 600
x = rng.normal(size=n)
d = pd.DataFrame({"x": x, "y": 1 + 0.5 * x + rng.standard_t(5, n)})
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    q0 = lm.quantile_regression(d, "y ~ x", taus=(0.1, 0.9), B=0)
out = buf.getvalue()
t0 = q0["equality_test"]
check("B=0: the equality test is not computed (valid False, p NaN, B_used 0)",
      t0["valid"] is False and np.isnan(t0["p"]) and t0["B_used"] == 0, t0)
check("B=0: the message says no bootstrap was asked for, without mentioning convergence",
      any("B=0" in w for w in q0["warnings"]) and "converge" not in " ".join(q0["warnings"]) and "converge" not in out,
      q0["warnings"])
check("B=0: the taus themselves are still reported as converged", all(q0["converged"].values()))
q1 = lm.quantile_regression(d, "y ~ x", taus=(0.1, 0.9), B=40, show=False)
check("B>0 with converged taus: the test stays valid, with no warning",
      q1["equality_test"]["valid"] is True and not q1["warnings"] and q1["equality_test"]["B_used"] > 30,
      q1["equality_test"]["B_used"])

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
