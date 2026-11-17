"""Tests for v2.6.11 (outside the module tests): the line coverage report and its CI step, the evaluator of the routing
case causal-checklist-post-treatment, the eval data kept byte for byte, and the documentation of the fixes.

Origin: items #3 and #9 of the structural optimization review (2026-11-13) and the two pending items of v2.6.10
(routing round R2 and the v2.6.10 validation, 2026-11-16). The replications and the fixes of diagnostics.py and
state_space.py are tested in tests/test_diagnostics.py.

Checks:
  1. tools/coverage_report.py on a coverage JSON built here: lines per function, counts per level, level-D functions
     run through others, level A-C functions with no line run, the exit codes, and no threshold;
  2. the CI step: coverage only in the latest job on Python 3.13, the report in the job summary and an artifact, no
     --fail-under anywhere;
  3. the evaluator regex passes the R2 answer (reconstructed from the routing record's excerpts) and still fails each
     half of the requirement; synonyms of review r1 pass, 'tratamento dos dados' does not count;
  4. evals/data/*.csv is not converted on checkout, the generator writes LF, and the committed file has no CR and the
     SHA-256 the generator expects;
  5. time-series.md and software-equivalents.md describe the new behavior; the CHANGELOG entry.

Run from the repository root:  python3 tests/test_v2611.py
"""
import hashlib
import io
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "econometrics"
sys.path.insert(0, str(SKILL / "scripts"))
sys.path.insert(0, str(ROOT / "tools"))
import _console  # noqa: E402,F401  (non-ASCII text on a Windows console)

import coverage_report as cr  # noqa: E402

results = []


def check(name, cond, info=""):
    results.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info)


# ------------------------------------------------------------------------- 1. tools/coverage_report.py
funcs = cr.public_functions()
lv = cr.levels()
check("coverage_report: same public functions as VALIDATION.md (one level each)",
      set(funcs) == set(lv) and len(funcs) > 250, (len(funcs), len(lv), sorted(set(funcs) ^ set(lv))[:5]))
fname, first, last = funcs["diagnostics.vif_table"]
_, b_first, b_last = funcs["diagnostics.influence"]
body_vif = list(range(first, last + 1))
body_inf = list(range(b_first, b_last + 1))
cov = {"files": {
    "econometrics/scripts/diagnostics.py": {"executed_lines": body_vif + [first - 1],     # def line runs on import
                                            "missing_lines": body_inf},
    "C:\\work\\econometric skills\\econometrics\\scripts\\risk.py": {"executed_lines": [], "missing_lines": []},
    "tests/test_x.py": {"executed_lines": [1, 2, 3], "missing_lines": []}}}
rows = {r["function"]: r for r in cr.per_function(cov)}
check("coverage_report: lines of a function are its body lines (the def line is not counted)",
      rows["diagnostics.vif_table"]["lines"] == len(body_vif) and rows["diagnostics.vif_table"]["run"] == len(body_vif)
      and rows["diagnostics.influence"]["run"] == 0 and rows["diagnostics.influence"]["lines"] == len(body_inf),
      (rows["diagnostics.vif_table"], rows["diagnostics.influence"]))
risk_rows = [r for k, r in rows.items() if k.startswith("risk.")]
check("coverage_report: Windows paths are read; files outside econometrics/scripts are ignored; untraced modules marked",
      risk_rows and all(r["traced"] for r in risk_rows) and not rows["causal.att_gt"]["traced"]
      and not any(k.startswith("test_x") for k in rows), (risk_rows[:1], rows["causal.att_gt"]))
lv_saved = cr.levels
cr.levels = lambda: {**lv_saved(), "diagnostics.vif_table": "D", "diagnostics.influence": "A"}
try:
    txt = cr.report(cr.per_function(cov), show_all=True)
finally:
    cr.levels = lv_saved
check("coverage_report: a level-D function whose lines run is listed as run through other functions",
      "### Level D executed through other functions (1)" in txt
      and f"- `diagnostics.vif_table`: {len(body_vif)}/{len(body_vif)} lines" in txt, txt[:600])
check("coverage_report: a level A-C function with no line run is flagged",
      "- `diagnostics.influence` (level A)" in txt, txt[:900])
check("coverage_report: the per-level table and the full list ('--all') are in the report",
      "| level | functions | executed at all | lines run / lines |" in txt and "### All public functions" in txt
      and f"| `diagnostics.vif_table` | D | {len(body_vif)}/{len(body_vif)} |" in txt)
check("coverage_report: says what it is not (no threshold; coverage is not validation; child processes)",
      "No threshold" in txt and "Coverage is not validation" in txt and "child processes not traced" in txt)
tmp = ROOT / "tests" / "_cov_v2611.json"
try:
    tmp.write_text(json.dumps(cov), encoding="utf-8")
    r_ok = subprocess.run([sys.executable, str(ROOT / "tools" / "coverage_report.py"), str(tmp)],
                          capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
    r_bad = subprocess.run([sys.executable, str(ROOT / "tools" / "coverage_report.py")],
                           capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
finally:
    tmp.unlink(missing_ok=True)
check("coverage_report: exit 0 with a report whatever the percentages; exit 2 and the usage line without input",
      r_ok.returncode == 0 and r_ok.stdout.startswith("## Line coverage") and r_bad.returncode == 2
      and "usage:" in r_bad.stderr, (r_ok.returncode, r_ok.stderr[-300:], r_bad.returncode))
src = (ROOT / "tools" / "coverage_report.py").read_text(encoding="utf-8")
check("coverage_report: no threshold in the code (no fail-under, no exit on a percentage)",
      "fail_under" not in src and "fail-under" not in src and "sys.exit(1" not in src)

# ------------------------------------------------------------------------- 2. the CI step
yml = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
job = yml[:yml.index("\n  windows:")]
check("CI: the suites run under coverage only in the latest job on Python 3.13",
      'if [ "${{ matrix.deps }}" = "latest" ] && [ "${{ matrix.python }}" = "3.13" ]; then' in job
      and 'run="python -m coverage run -p --source=econometrics/scripts"' in job and '$run "$t" || status=1' in job
      and "coverage" not in yml[yml.index("\n  windows:"):])
check("CI: the report step runs coverage_report.py into the job summary, also when a suite failed",
      "name: Line coverage report by VALIDATION.md level (no threshold)" in job
      and "if: ${{ !cancelled() && matrix.deps == 'latest' && matrix.python == '3.13' }}" in job
      and "python tools/coverage_report.py coverage.json --all > coverage-report.md" in job
      and 'cat coverage-report.md >> "$GITHUB_STEP_SUMMARY"' in job)
check("CI: coverage.json, coverage.xml and the report are uploaded as an artifact",
      "actions/upload-artifact@v6" in job and all(f in job for f in ("coverage.json", "coverage.xml",
                                                                      "coverage-report.md")))
check("CI: no coverage threshold anywhere in the workflow", "fail-under" not in yml and "fail_under" not in yml)
gi = (ROOT / ".gitignore").read_text(encoding="utf-8")
check("the coverage files are ignored by git", all(x in gi.splitlines() for x in (".coverage", ".coverage.*",
                                                                                   "coverage.json")))

# ------------------------------------------------------------------------- 3. the evaluator
grader = (SKILL / "evals" / "routing" / "causal-checklist-post-treatment" / "graders" / "answer.md")
pat = re.search(r"^pattern: '(.*)'$", grader.read_text(encoding="utf-8"), re.M).group(1)
g = re.compile(pat, re.I)
# Reconstructed from the excerpts of the R2 answer in the development repository's routing record (the full answer is not
# kept); review r1 renamed it from "recorded".
r2 = ("Não use o número de funcionários um ano depois da adesão como controle: é uma variável medida depois do "
      "tratamento. Como as empresas escolheram participar, há fatores não observados que afetam as duas coisas.")
check("evaluator: the R2 answer (reconstructed from the record's excerpts) now passes", bool(g.search(r2)))
check("evaluator: still fails with only the post-treatment half or only the unobserved half",
      not g.search("A variável é medida depois do tratamento; não use como controle.")
      and not g.search("Há fatores não observados que afetam as duas coisas.")
      and not g.search("Funcionários depois da adesão é pós-tratamento: não use como controle.")
      and g.search("The headcount is a post-treatment variable, and selection on unobserved factors needs Oster.")
      and g.search("Mesmo after the treatment... há fatores não observados."))
check("evaluator (review r1): synonyms pass — após o tratamento, pós tratamento, posterior ao tratamento, variáveis "
      "não observadas, autosseleção, viés de seleção; 'depois do tratamento dos dados' is not post-treatment",
      all(g.search(x) for x in ("É medida após o tratamento; há variáveis não observadas.",
                                 "Variável pós tratamento; cuidado com a autosseleção.",
                                 "Medida posterior ao tratamento, e há viés de seleção.",
                                 "Characteristics measured posttreatment; self-selection matters."))
      and not g.search("Depois do tratamento dos dados, há variáveis não observadas."))
check("evaluator (review r2): the exclusion holds for every form of 'tratamento d…', word anchors (Após ≠ pós, "
      "intermediador ≠ mediador), 'post-treatment period' and 'treatment of'; plural and English synonyms pass",
      not any(g.search(x) for x in ("Após tratamento dos dados, há fatores não observados.",
                                    "Faça o pós-tratamento dos dados e trate as variáveis não observadas.",
                                    "Depois do tratamento de dados, há variáveis não observadas.",
                                    "After the treatment of missing values, check for unobserved heterogeneity.",
                                    "Use the post-treatment period in a DiD; there is no selection bias.",
                                    "O intermediador financeiro é não observável."))
      and all(g.search(x) for x in ("É um mediating variable; unmeasured confounders matter.",
                                    "Pós-tratamento; variáveis omitidas importam.")))
check("evaluator (review r3): the exclusion is only for data treatment — 'pós-tratamento do programa', 'depois do "
      "tratamento do grupo', 'após o tratamento de 2019' pass; 'tratamento estatístico', 'after the treatment period' fail",
      all(g.search(x) for x in ("O emprego é pós-tratamento do programa; há viés de seleção.",
                                 "Foi medido depois do tratamento do grupo; há confundidores não observados.",
                                 "É medida após o tratamento de 2019; existem fatores não observados."))
      and not any(g.search(x) for x in ("Depois do tratamento estatístico dos dados, há fatores não observados.",
                                        "After the treatment period, check for unobserved heterogeneity.")))

# ------------------------------------------------------------------------- 4. eval data byte for byte
ga = (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
check(".gitattributes: evals/data/*.csv is not converted (-text); tests/data keeps text eol=lf",
      "evals/data/*.csv -text" in ga and "tests/data/*.csv text eol=lf" in ga)
gen = (ROOT / "evals" / "data" / "make_treino_regiao_v269.py").read_text(encoding="utf-8")
want = re.search(r'EXPECTED = "([0-9a-f]{64})"', gen).group(1)
raw = (ROOT / "evals" / "data" / "treino_regiao_v269.csv").read_bytes()
check("eval data: the generator writes LF; the committed CSV has no CR and the SHA-256 the generator expects",
      'd.to_csv(out, index=False, lineterminator="\\n")' in gen and b"\r" not in raw
      and hashlib.sha256(raw).hexdigest() == want, hashlib.sha256(raw).hexdigest())

# ------------------------------------------------------------------------- 5. documentation
ts = (SKILL / "references" / "time-series.md").read_text(encoding="utf-8")
sec7 = ts[ts.index("## 7. Cointegration"):ts.index("## 8. VAR")]
check("time-series.md §7: both lag choices, 'Inconclusive', the INTQRT numbers and the det_order=0 caveat",
      "lags chosen by AIC and by BIC" in sec7 and '"Inconclusive"' in sec7 and "t = −8.39" in sec7
      and "AIC picks 6 lags" in sec7 and "a full rank there is not evidence of I(0)" in sec7)
se = (SKILL / "references" / "software-equivalents.md").read_text(encoding="utf-8")
check("software-equivalents.md: estat vif (centered) and estat vif, uncentered map to diagnostics.vif_table (review r1)",
      "| `estat vif` / `estat vif, uncentered` | — | `diagnostics.vif_table(res)` (centered VIF when the model has a "
      "constant, uncentered when it has none" in se)
changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
top = changelog[changelog.index("## v2.6.11"):changelog.index("## v2.6.10")]
check("CHANGELOG v2.6.11: the four fixes, the replications with editions, the coverage report, the evaluator",
      all(k in top for k in ("`diagnostics.cointegration` gave a firm verdict", "`det_order=0`",
                             "`state_space.cointegration_pretest`", "`diagnostics.vif_table` depended on the statsmodels",
                             "`diagnostics.hausman` gave a verdict from robust", "1st ed. (2000)", "5th ed. (2013)",
                             "7th ed. (2012)", "tools/coverage_report.py", "fatores não observados", "-text")))

passed = sum(results)
print(f"\n{passed}/{len(results)} checks passed")
sys.exit(0 if passed == len(results) else 1)
