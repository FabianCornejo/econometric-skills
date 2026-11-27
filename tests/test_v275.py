"""v2.7.5: regressions and documentation checks (rule T4: tests of functions live in the module files; the
replications on the QuantEcon data live in test_quantecon_replication.py).

- the graders of the three new routing cases (review-mixed-samples, review-j-prob-cdf, review-2sls-by-hand; written in
  v2.7.5, first run in the routing round of v2.8.0), measured on the corpora of tests/corpora_v275.py in Python and
  JavaScript, with the prompt echoed back refused (lessons of v2.6.11, v2.7.0 and v2.7.4);
- the rules of v2.7.5 in interpretation-rules.md, review-checklist.md, time-series.md and volatility.md;
- the claim of the initial-condition rule, by simulation with a known parameter: an AR(1) with rho = 0.5 that starts
  at y0 = 10 is estimated near 0.9 by the exact likelihood (statsmodels SARIMAX) and near 0.5 by AutoReg;
- version, CHANGELOG, README and VALIDATION.md.
"""
import json
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "econometrics" / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _console  # noqa: E402,F401  (Windows-safe console output, see econometrics/scripts/_console.py)
import corpora_v275 as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SK = ROOT / "econometrics"
ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info)


# ------------------------------------------------------------------ the three new routing cases
CASES = {"review-mixed-samples": C.MIXED, "review-j-prob-cdf": C.JPROB, "review-2sls-by-hand": C.HAND2SLS}
ref_sf = (SK / "evals" / "routing" / "review-null-result" / "graders" / "skill-fired.md").read_text(encoding="utf-8")
node = shutil.which("node")
for name, corpus in CASES.items():
    folder = SK / "evals" / "routing" / name
    text = (folder / "prompt.md").read_text(encoding="utf-8")
    meta = yaml.safe_load(text.split("---")[1])
    request = text.split("---", 2)[2]
    g = yaml.safe_load((folder / "graders" / "answer.md").read_text(encoding="utf-8").split("---")[1])
    check(f"{name}: valid YAML, reviewer case with the routing tags, regex grader on the last message; "
          "the skill-fired grader of the other review cases",
          {"routing", "reviewer"} <= set(meta["tags"]) and meta["max_turns"] == 8 and g["type"] == "regex"
          and g["target"] == "last_message" and g.get("flags") == "i"
          and (folder / "graders" / "skill-fired.md").read_text(encoding="utf-8") == ref_sf, (meta, g.get("type")))
    pat = re.compile(g["pattern"], re.I)
    good, bad = corpus["good"], corpus["bad"]
    check(f"{name}: corpus of at least 20 right and 20 wrong answers", len(good) >= 20 and len(bad) >= 20,
          (len(good), len(bad)))
    check(f"{name}: the grader accepts each right answer (Python)", all(pat.search(s) for s in good),
          [s for s in good if not pat.search(s)])
    check(f"{name}: the grader refuses each wrong answer and the prompt echoed back (Python)",
          not any(pat.search(s) for s in bad + [request]), [s for s in bad + [request] if pat.search(s)])
    if node:
        js = ("const d=JSON.parse(require('fs').readFileSync(0,'utf8'));const p=new RegExp(d.pat,'i');"
              "console.log(JSON.stringify([d.good.filter(g=>!p.test(g)),d.bad.filter(b=>p.test(b))]))")
        out = subprocess.run([node, "-e", js], input=json.dumps({"pat": g["pattern"], "good": good, "bad": bad + [request]}),
                             capture_output=True, text=True, encoding="utf-8")
        res = json.loads(out.stdout) if out.returncode == 0 else out.stderr
        check(f"{name}: same verdicts in JavaScript (the eval runner's regex engine)", res == [[], []], res)
    else:
        print("SKIP JavaScript check: node not found")
known = C.KNOWN_FALSE_PASSES
for name, key in (("review-mixed-samples", "MIXED"), ("review-j-prob-cdf", "JPROB"), ("review-2sls-by-hand", "HAND2SLS")):
    g = yaml.safe_load((SK / "evals" / "routing" / name / "graders" / "answer.md").read_text(encoding="utf-8").split("---")[1])
    still = sum(bool(re.search(g["pattern"], s, re.I)) for s in known[key])
    print(f"INFO {name}: {still} of {len(known[key])} known false passes (negations) still accepted by the grader")
n_cases = len([d for d in (SK / "evals" / "routing").iterdir() if d.is_dir()])
readme = (ROOT / "README.md").read_text(encoding="utf-8")
check("at least 28 routing cases on disk (v2.8.0: a count that grows); the README names the three new reviews", n_cases >= 28
      and all(w in readme for w in ("reviews (", "Prob(J)", "by hand with OLS")), n_cases)

# ------------------------------------------------------------------ the rules of v2.7.5
ir = (SK / "references" / "interpretation-rules.md").read_text(encoding="utf-8")
sec = ir.split("## Rules added in v2.7.5", 1)[1].split("\n## ", 1)[0] if "## Rules added in v2.7.5" in ir else ""
check("interpretation-rules.md: a v2.7.5 section (found by its title; v2.8.0 added one after it) with the six rules "
      "(cases a-g; h left for E7)",
      sec and all(w in sec for w in (
          "same rows", "8.38", "8.09", "700", "5000",                     # (a) and (c)
          "Prob(J)", "1 − Prob", "0.77",                                    # (b)
          "by hand", "0.1255", "0.1541", "iv_report",                       # (d)
          "boundary", "Self and Liang", "Andrews", "½χ²(0) + ½χ²(1)",       # (e)
          "does **not** hold for a GARCH(1,1)",                             # (e), review r1 S2-3
          "when every other parameter is identified under the null",       # (e), review r2 S3-4
          "prior", "unit root", "point mass",                               # (f)
          "initial condition", "0.90", "0.49", "AutoReg", "ts_identify")),  # (g)
      sec[:200])
check("interpretation-rules.md: no conditional-sum-of-squares method promised for ARIMA; no unverified 0.88; "
      "no private script cited", 'method="css"' not in ir and "has no conditional-sum-of-squares method" in sec
      and "0.88" not in sec and "claude/scripts" not in sec)
rc = (SK / "references" / "review-checklist.md").read_text(encoding="utf-8")
check("review-checklist.md: the checklist asks the four v2.7.5 questions",
      rc.count("v2.7.5") == 4 and all(w in rc for w in ("IV second stage run by hand with OLS", "Prob(J) is the CDF",
                                                          "null on the boundary of the parameter space",
                                                          "Do the numbers set side by side", "does the prior give support to the region",
                                                          "exact likelihood on a sample that starts far from the mean")))
ts = (SK / "references" / "time-series.md").read_text(encoding="utf-8")
vo = (SK / "references" / "volatility.md").read_text(encoding="utf-8")
check("time-series.md (section 5) and volatility.md point to the new rules",
      "Initial condition (v2.7.5)" in ts.split("## 6.")[0].split("## 5.")[1] and "AutoReg" in ts
      and "Testing for ARCH effects against a GARCH (v2.7.5)" in vo and "parametric bootstrap" in vo)

# ------------------------------------------------------------------ the claim of the initial-condition rule
from statsmodels.tsa.ar_model import AutoReg  # noqa: E402
from statsmodels.tsa.statespace.sarimax import SARIMAX  # noqa: E402

rng = np.random.default_rng(275)
ex, cond = [], []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for _ in range(60):
        y = np.empty(51)
        y[0] = 10.0  # far in the tail of the stationary N(0, 1/(1 - 0.25)): about 8.7 SDs
        for t_ in range(1, 51):
            y[t_] = 0.5 * y[t_ - 1] + rng.normal()
        ex.append(SARIMAX(y, order=(1, 0, 0), trend="n").fit(disp=0).params[0])
        cond.append(AutoReg(y, 1, trend="n").fit().params[0])
ex, cond = np.array(ex), np.array(cond)
check("initial condition: with y0 = 10 the exact MLE of rho = 0.5 averages above 0.8, AutoReg within 0.1 of 0.5",
      ex.mean() > 0.8 and abs(cond.mean() - 0.5) < 0.1, (ex.mean(), cond.mean()))
cond0 = []
rng = np.random.default_rng(2751)
for _ in range(60):  # the control: with y0 drawn from the stationary distribution, no pull towards 1
    y = np.empty(51)
    y[0] = rng.normal(0, 1 / np.sqrt(0.75))
    for t_ in range(1, 51):
        y[t_] = 0.5 * y[t_ - 1] + rng.normal()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cond0.append(SARIMAX(y, order=(1, 0, 0), trend="n").fit(disp=0).params[0])
check("initial condition: with a stationary y0 the exact MLE stays within 0.1 of 0.5 (the pull comes from y0)",
      abs(np.mean(cond0) - 0.5) < 0.1, np.mean(cond0))

# ------------------------------------------------------------------ VALIDATION.md: the new replications are listed
val = (ROOT / "VALIDATION.md").read_text(encoding="utf-8")
rows = {f: next((ln for ln in val.splitlines() if ln.startswith(f"| `{f}`")), "")
        for f in ("diagnostics.iv_report", "diagnostics.endogeneity_test", "estimation.mle", "diagnostics.count_diagnostics")}
check("VALIDATION.md: the rows of iv_report, endogeneity_test, mle and count_diagnostics cite the QuantEcon replication",
      all("test_quantecon_replication.py" in r for r in rows.values()) and "control-function" in rows["diagnostics.iv_report"]
      and all(f"test_quantecon_replication.py: {c_}" in rows[f_] for f_, cs in (
          ("diagnostics.iv_report", ("AJR iv_report unadjusted", "AJR first stage", "AJR 2SLS closed form",
                                     "AJR iv_report robust", "AJR Wu-Hausman", "AJR Anderson-Rubin")),
          ("diagnostics.endogeneity_test", ("AJR endogeneity_test (70 rows)",)),
          ("estimation.mle", tuple(f"Treisman Model {i}: estimation.mle" for i in (1, 2, 3))),
          ("diagnostics.count_diagnostics", ("Treisman Model 1: count_diagnostics",))) for c_ in cs)
      and "Acemoglu" in rows["diagnostics.iv_report"] and "Treisman" in rows["estimation.mle"],
      {k: bool("test_quantecon_replication.py" in v) for k, v in rows.items()})

# ------------------------------------------------------------------ version and CHANGELOG (found by title, not "the top")
import reporting  # noqa: E402
cl = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
heads = re.findall(r"^## v(\d+(?:\.\d+)+) — ", cl, re.M)
check("CHANGELOG has a v2.7.5 entry, right before v2.7.4's", "2.7.5" in heads and "2.7.4" in heads
      and heads.index("2.7.5") + 1 == heads.index("2.7.4"), heads[:4])
e275 = cl.split("## v2.7.5 — ", 1)[1].split("\n## v", 1)[0] if "## v2.7.5 — " in cl else ""
check("the v2.7.5 entry names the replications, the functions they cover, the rules and the three cases",
      all(w in e275 for w in ("Acemoglu", "Treisman", "43f32e0", "iv_report", "endogeneity_test", "estimation.mle",
                              "count_diagnostics", "interpretation-rules.md", "review-mixed-samples",
                              "review-j-prob-cdf", "review-2sls-by-hand", "GITHUB_ACTIONS")))
mk = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))
vers = {p_.get("version") for p_ in mk.get("plugins", [])}
v_ok = tuple(int(x) for x in reporting.SKILL_VERSION.split(".")) >= (2, 7, 5)   # a version that grows: >=
check("SKILL_VERSION is 2.7.5 or later and equals the marketplace's version", v_ok and vers == {reporting.SKILL_VERSION},
      (reporting.SKILL_VERSION, vers))

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
