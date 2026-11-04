"""Tests for v2.3.7: no number from an estimator that did not converge, and independent checks that are
independent.

Found in the v2.3.6 clean-install test (return to education, WAGE1). (1) quantile_regression used
statsmodels' default of 1000 iterations and did not check convergence: at tau = 0.1 it stopped at the
limit with educ = 0.038897, while the solution is 0.038610 (reached at 5000 and stable at 20000); the
report gave 4.0% instead of 3.9%. (2) The agent called reporting.cross_check on the same OLS
computed by statsmodels and by numpy and reported it as an independent confirmation: that checks the
arithmetic, not the method.

Checks: quantile_regression reaches the solution at tau = 0.1 on that case, retries with more
iterations and reports converged per tau (table column too); with an iteration budget that is too
small it marks the taus, warns, and the equality test is flagged invalid; the bootstrap reports how
many replications it used; cross_check without se is labelled an implementation check (kind) and says
it is not an independent confirmation, with se it is a check across methods; agreement verdicts do
not change; the rules in SKILL.md and interpretation-rules.md say both things.

Run from the repository root:  python3 tests/test_v237.py
"""
import contextlib
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import wooldridge

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "econometrics"
sys.path.insert(0, str(SKILL / "scripts"))
warnings.filterwarnings("ignore")
import linear_models as lm
import reporting as rp

ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


def run(fn):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn()
    return out, buf.getvalue()


# ---------------------------------------------------------------- 1. quantile regression converges
d = wooldridge.data("wage1")
d = d.assign(lwage=np.log(d.wage))
F = "lwage ~ educ + exper + expersq + tenure + tenursq + female + nonwhite + married"
qr, out = run(lambda: lm.quantile_regression(d, F, taus=(0.1, 0.5), B=30, seed=1, test_coef="educ"))
t = qr["table"]
b10 = float(t.query("tau == 0.1 and term == 'educ'")["estimate"].iloc[0])
check("tau = 0.1 reaches the solution (educ 0.038610), not the 1000-iteration stop (0.038897)",
      abs(b10 - 0.038610) < 1e-5, f"{b10:.6f}")
check("converged is reported per tau, and as a table column", qr.get("converged") == {0.1: True, 0.5: True}
      and "converged" in t.columns and bool(t["converged"].all()), qr.get("converged"))
check("no warning when every tau converged", not qr.get("warnings") and "did not converge" not in out)
check("the bootstrap reports how many replications it used", 0 < qr["equality_test"].get("B_used", 0) <= 30,
      qr["equality_test"].get("B_used"))

bad, out = run(lambda: lm.quantile_regression(d, F, taus=(0.1, 0.5), B=5, seed=1, test_coef="educ", max_iter=20))
check("too small an iteration budget: the taus are marked as not converged",
      bad["converged"] and not all(bad["converged"].values()), bad["converged"])
check("...the function warns that those numbers must not be reported",
      any("did not converge" in w for w in bad["warnings"]) and "WARNING" in out and "did not converge" in out,
      out.strip().splitlines()[-1][:110] if out.strip() else "")
check("...and the equality test is flagged invalid", bad["equality_test"].get("valid") is False)

# ---------------------------------------------------------------- 2. cross_check labels what it checked
impl, out_i = run(lambda: rp.cross_check({"statsmodels": 0.0789, "numpy (X'X)^-1 X'y": 0.0789 + 1e-12},
                                         label="educ"))
check("without se: an implementation check (same estimator), not an independent confirmation",
      impl.get("kind") == "implementation" and "Implementation check" in out_i
      and "not an independent confirmation" in out_i, out_i.strip().splitlines()[0][:110])
meth, out_m = run(lambda: rp.cross_check({"OLS": 0.0789, "Huber robust": 0.0815}, se=0.0077, label="educ"))
check("with se: a check across methods", meth.get("kind") == "method" and "Independent check" in out_m,
      out_m.strip().splitlines()[0][:110])
check("agreement verdicts are unchanged", impl["agree"] is True and meth["agree"] is True
      and rp.cross_check({"a": 1.0, "b": 1.3}, se=0.1, show=False)["agree"] is False)

# ---------------------------------------------------------------- 3. the rules
skill_md = (SKILL / "SKILL.md").read_text(encoding="utf-8")
rules = (SKILL / "references" / "interpretation-rules.md").read_text(encoding="utf-8")
check("rules: a number from an estimator that did not converge is not reported",
      "did not converge" in rules and "not reported" in rules and "converge" in skill_md)
check("rules: the second route uses a different method; one estimator twice is an implementation check",
      "different method" in rules and "implementation check" in rules and "different method" in skill_md)

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
