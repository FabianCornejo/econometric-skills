"""Tests for v2.4.6.2: fixes from the v2.4.6.1 validation (the IPCA request run again, 2026-11-05).

1. sup_wald was given the 12 month dummies; it added its own constant, so the design was rank deficient and the
   Chow F used 13 restrictions instead of 12 (sup-F 3.89 instead of 4.24; the bootstrap p-value was unaffected).
2. A log written with `py script.py > log.txt` on Windows was cp1252: arrows became "?", and the agent stripped
   the accents from its Portuguese output. _console now writes regular files in UTF-8.
3. One progress message came out in Spanish in a Portuguese session.

Run from the repository root:  python3 tests/test_v2462.py
"""
import os
import re
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "econometrics"
sys.path.insert(0, str(SKILL / "scripts"))
import _console  # noqa: E402,F401
import ts_univariate as tu  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


# ---------------------------------------------------------------- 1. X that already spans the constant
src = (ROOT / "tests" / "test_v2461.py").read_text(encoding="utf-8")
v = np.array(re.search(r'IPCA = """(.*?)"""', src, re.S).group(1).split(), float)
y = pd.Series(v, index=pd.date_range("2000-01-01", periods=len(v), freq="MS"))
mes = pd.get_dummies(y.index.month, prefix="m", dtype=float).set_index(y.index)
full = tu.sup_wald(y, mes, B=99, show=False)
eleven = tu.sup_wald(y, mes.iloc[:, 1:], B=99, show=False)
check("12 month dummies: no extra constant, same sup-F as 11 dummies plus a constant (4.24, not 3.89)",
      np.isclose(full["statistic"], eleven["statistic"]) and abs(full["statistic"] - 4.244) < 0.01
      and any("already contains a constant" in w for w in full["warnings"]), f"{full['statistic']:.3f}")
check("12 month dummies: the recursive coefficients are the 12 month means (no 'const' column)",
      list(full["recursive_coefficients"].columns) == list(mes.columns))
D = full["break_position"]
Xr = mes.to_numpy()
Xu = np.column_stack([Xr, Xr * (np.arange(len(v)) >= D)[:, None]])
import statsmodels.api as sm  # noqa: E402
ssr0, ssr1 = sm.OLS(v, Xr).fit().ssr, sm.OLS(v, Xu).fit().ssr
check("12 month dummies: sup-F = Chow F with 12 restrictions at the estimated date (statsmodels OLS)",
      np.isclose(full["statistic"], ((ssr0 - ssr1) / 12) / (ssr1 / (len(v) - 24))))
x = np.random.default_rng(1).normal(size=len(v))
try:
    tu.sup_wald(y, pd.DataFrame({"x": x, "x2": 2 * x}, index=y.index), B=9, show=False)
    raised = False
except ValueError as e:
    raised = "collinear" in str(e)
check("collinear regressors (x and 2x) are refused with a clear message", raised)
big = tu.sup_wald(y, pd.DataFrame({"gdp": 1e13 * (1 + 0.1 * x)}, index=y.index), B=9, show=False)
check("a regressor in large units (1e13) keeps the constant (ranks on normalised columns)",
      list(big["recursive_coefficients"].columns) == ["const", "gdp"] and not any("already contains" in w for w in big["warnings"]))
plain = tu.sup_wald(y, pd.DataFrame({"x": x}, index=y.index), B=9, show=False)
check("an ordinary regressor still gets the constant", list(plain["recursive_coefficients"].columns) == ["const", "x"]
      and not any("already contains" in w for w in plain["warnings"]))

# ---------------------------------------------------------------- 2. logs written to a file are UTF-8
code = ("import sys; sys.path.insert(0, %r); import _console; print('ação → ≥ — ±'); "
        "print(sys.stdout.encoding, file=sys.stderr)" % str(SKILL / "scripts"))
env = dict(os.environ, PYTHONIOENCODING="cp1252")
env.pop("PYTHONUTF8", None)
with tempfile.TemporaryDirectory() as d:
    log = Path(d) / "log.txt"
    with open(log, "wb") as fh:
        r = subprocess.run([sys.executable, "-c", code], stdout=fh, stderr=subprocess.PIPE, env=env)
    text = log.read_bytes().decode("utf-8", errors="strict") if r.returncode == 0 else ""
check("stdout redirected to a file with a cp1252 default is written in UTF-8 (accents and arrows intact)",
      r.returncode == 0 and text.strip() == "ação → ≥ — ±", repr(text))
p = subprocess.run([sys.executable, "-c", code], capture_output=True, env=env)
check("stdout to a pipe keeps the console encoding and the tolerant handler (arrow as '?', accents in cp1252)",
      p.returncode == 0 and p.stdout.strip() == "ação ? ? — ±".encode("cp1252"), repr(p.stdout))

code_err = ("import sys; sys.path.insert(0, %r); import _console; print('x\\udcff', file=sys.stderr); "
            "print(sys.stderr.errors)" % str(SKILL / "scripts"))
with tempfile.TemporaryDirectory() as d:
    errf = Path(d) / "err.txt"
    with open(errf, "wb") as fh:
        r2 = subprocess.run([sys.executable, "-c", code_err], stdout=subprocess.PIPE, stderr=fh, env=env)
    err_text = errf.read_bytes().decode("utf-8", errors="replace")
check("stderr redirected to a file keeps its error handler (backslashreplace): a lone surrogate does not crash",
      r2.returncode == 0 and r2.stdout.strip() == b"backslashreplace" and "\\udcff" in err_text, repr(err_text[:40]))

# ---------------------------------------------------------------- 3. language of progress messages
skill = (SKILL / "SKILL.md").read_text(encoding="utf-8")
check("SKILL.md: the user's language applies to progress messages and to accents in scripts and logs",
      "including the short progress messages between steps" in skill and "keep its accents in the scripts and logs" in skill)

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
