"""Measure whether Claude Code reaches the econometrics skill, in a clean and in a crowded skill listing.

`claude plugin eval` runs each case in an isolated session with only the plugins the case names. That
measures routing in a clean install, but the failures seen on the user's notebook came from a crowded one:
with hundreds of skills, Claude Code drops the descriptions to fit its listing budget (about 1% of the context
window) and offers skills by name only. This script builds a harness in which every routing case of
`econometrics/evals/routing/` runs twice:

    clean-<case>     the econometrics plugin alone
    crowded-<case>   the econometrics plugin plus 48 decoy skills (many about data and statistics) whose
                     descriptions add up to about 35,000 characters, enough to overflow the listing

and then prints, for each case, how often the skill fired and how often the answer check passed in each arm.
The plugin is copied into each case (the eval harness loads only plugins inside the case directory), with a
`plugin.json` built from the marketplace entry, so the copy declares the same skills, commands and agent.

Usage (from the repository root; needs Claude Code >= 2.1.269 and runs on your account):
    python tools/routing_evals.py                    # 3 runs per case and arm (~$3-6 at list price)
    python tools/routing_evals.py --runs 1 --case two-groups
    python tools/routing_evals.py --build-only out/  # write the harness and stop (no model calls)

The cases grant only read-only tools (no Bash), so they run on native Windows too. Results go to
routing-results/<timestamp>/ (summary.md, routing.json, the harness's report.html).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "econometrics"
CASES = PLUGIN / "evals" / "routing"

DECOYS = [
    ("data-explorer", "explore a dataset, summary statistics, distributions, missing values and outliers"),
    ("stats-helper", "statistical analysis, hypothesis tests, p-values, t-tests and confidence intervals"),
    ("ab-test-analyzer", "A/B tests, conversion rates, significance, lift and sample size for experiments"),
    ("survey-analysis", "survey data, Likert scales, cross-tabulations, chi-square tests and weighting"),
    ("data-quality", "data validation, duplicates, type checks, ranges and reconciliation of tables"),
    ("dashboard-builder", "dashboards, KPIs, charts and metric definitions for business reviews"),
    ("forecast-assistant", "sales forecasts, seasonality, trend lines and scenario projections"),
    ("sql-analyst", "SQL queries, joins, window functions and aggregations over warehouse tables"),
    ("excel-modeler", "spreadsheet models, formulas, pivot tables and sensitivity tables"),
    ("cohort-retention", "cohort analysis, retention curves, churn rates and lifetime value"),
    ("pricing-analyst", "price elasticity, discounts, willingness to pay and margin analysis"),
    ("marketing-mix", "marketing attribution, channel ROI, media mix and campaign effects"),
    ("customer-segments", "customer segmentation, clustering, personas and RFM scores"),
    ("finance-ratios", "financial ratios, margins, returns, leverage and peer comparisons"),
    ("portfolio-risk", "portfolio volatility, drawdowns, correlations, VaR and risk budgets"),
    ("credit-scoring", "credit scores, default probability, scorecards and rejection inference"),
    ("ops-metrics", "operations metrics, throughput, lead times, utilization and bottlenecks"),
    ("hr-analytics", "people analytics, attrition, engagement surveys, training programs and pay equity"),
    ("supply-planner", "inventory, safety stock, demand planning and service levels"),
    ("quality-control", "control charts, process capability, defects and Six Sigma"),
    ("text-insights", "text analytics, sentiment, topics and keyword extraction from feedback"),
    ("geo-analysis", "geospatial data, maps, regions, distances and catchment areas"),
    ("research-notes", "literature notes, summaries of papers, citations and reading lists"),
    ("report-writer", "analytical reports, executive summaries, findings, reviews and recommendations"),
    ("chart-maker", "charts and plots, axis choices, colors, labels and annotations"),
    ("notebook-helper", "Jupyter notebooks, pandas code, cleaning steps and reproducible analysis"),
    ("r-to-python", "translating R or Stata code to Python, packages and idioms"),
    ("ml-trainer", "machine-learning models, train/test splits, cross-validation and feature importance"),
    ("anomaly-finder", "anomaly detection in metrics and time series, alerts and thresholds"),
    ("benchmarking", "benchmarks against competitors, market shares and industry averages"),
    ("budget-tracker", "budgets versus actuals, variances, forecasts of spend and accruals"),
    ("kpi-definitions", "KPI definitions, metric trees, owners and targets"),
    ("experiment-design", "experiment design, randomization, control groups and power"),
    ("causal-helper", "cause and effect questions, drivers, impact of initiatives and attribution"),
    ("regression-tool", "regressions, coefficients, R-squared, instruments and predicting a variable from others"),
    ("time-series-tool", "time series, moving averages, decomposition and autocorrelation"),
    ("risk-register", "risk registers, likelihood and impact scoring, mitigation plans"),
    ("investment-memo", "investment memos, theses, valuation summaries and risks"),
    ("market-sizing", "market sizing, TAM/SAM/SOM, bottom-up and top-down estimates"),
    ("pdf-extractor", "extracting tables and text from PDF reports into data"),
    ("data-catalog", "data catalogs, lineage, column descriptions and documentation"),
    ("metric-alerts", "metric monitoring, alerts, thresholds and incident notes"),
    ("survey-designer", "questionnaire design, question wording, scales and sampling plans"),
    ("sports-stats", "sports statistics, player ratings, rankings and match data"),
    ("health-metrics", "health and wellness metrics, trackers and trends"),
    ("education-data", "student grades, class comparisons, attendance and school indicators"),
    ("energy-analytics", "energy consumption, load profiles, tariffs and savings estimates"),
    ("real-estate-data", "property prices, rents, comparables and neighborhood statistics"),
]
FILLER = ("Use it whenever the user mentions data, numbers, a spreadsheet, a CSV file, a table, a chart, a report, a "
          "metric, a comparison between groups, a trend, a test, a model or an estimate, even without naming this skill. "
          "It covers planning the analysis, cleaning the inputs, choosing the method, running the calculation, checking "
          "the result, explaining it in plain language, writing the summary for managers and suggesting next steps, "
          "with templates, checklists and examples for teams in finance, marketing, operations, sales, product and HR. "
          "It works with Python, R, SQL, Excel and BI tools, and adapts the depth of the explanation to the reader.")


def write_crowd(dst: Path) -> int:
    """The decoy plugin; returns the total length of its skill descriptions."""
    (dst / ".claude-plugin").mkdir(parents=True)
    (dst / ".claude-plugin" / "plugin.json").write_text(json.dumps(
        {"name": "crowd", "version": "1.0.0", "description": "Decoy skills for routing evals (not a real plugin)."},
        indent=2) + "\n", encoding="utf-8")
    total = 0
    for name, topic in DECOYS:
        desc = f"Helps with {topic}. {FILLER}"
        total += len(desc)
        d = dst / "skills" / name
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(f'---\nname: {name}\ndescription: "{desc}"\n---\n\n# {name}\n\nAnswer the request '
                                    f"about {topic} directly, with a short explanation and, when numbers are given, "
                                    "a quick calculation.\n", encoding="utf-8")
    return total


def write_plugin_copy(dst: Path) -> None:
    """Copy of econometrics/ with a plugin.json built from the marketplace entry (no evals inside)."""
    shutil.copytree(PLUGIN, dst, ignore=shutil.ignore_patterns("evals", "__pycache__", "*.pyc"))
    entry = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))["plugins"][0]
    manifest = {k: entry[k] for k in ("name", "version", "description", "author", "skills") if k in entry}
    (dst / ".claude-plugin").mkdir(exist_ok=True)
    (dst / ".claude-plugin" / "plugin.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                                         encoding="utf-8")


def set_frontmatter(text: str, name: str, plugins: list[str], extra_tag: str) -> str:
    m = re.match(r"---\r?\n(.*?)\r?\n---\r?\n(.*)", text, re.S)
    if not m:
        raise ValueError("prompt.md without frontmatter")
    lines = [ln for ln in m.group(1).splitlines() if not ln.startswith(("plugins:", "name:"))]
    lines = [re.sub(r"^tags: \[(.*)\]$", lambda t: f"tags: [{t.group(1)}, {extra_tag}]", ln) for ln in lines]
    lines += [f"name: {name}", "plugins: [" + ", ".join(f'"{p}"' for p in plugins) + "]"]
    return "---\n" + "\n".join(lines) + "\n---\n" + m.group(2)


def build(harness: Path, cases: list[str] | None = None) -> dict:
    """Write the harness plugin; returns {"cases": [...], "crowd_chars": int}."""
    (harness / ".claude-plugin").mkdir(parents=True, exist_ok=True)
    (harness / ".claude-plugin" / "plugin.json").write_text(json.dumps(
        {"name": "routing-harness", "version": "1.0.0", "description": "Harness for econometrics routing evals."},
        indent=2) + "\n", encoding="utf-8")
    names = sorted(p.name for p in CASES.iterdir() if (p / "prompt.md").exists())
    if cases:
        names = [n for n in names if any(re.fullmatch(c.replace("*", ".*"), n) for c in cases)]
    if not names:
        raise SystemExit("no routing case matches")
    crowd_chars = 0
    built = []
    for n in names:
        for arm in ("clean", "crowded"):
            d = harness / "evals" / f"{arm}-{n}"
            shutil.copytree(CASES / n / "graders", d / "graders")
            write_plugin_copy(d / "econometrics")
            plugins = ["econometrics"]
            if arm == "crowded":
                crowd_chars = write_crowd(d / "crowd")
                plugins.append("crowd")
            src = (CASES / n / "prompt.md").read_text(encoding="utf-8")
            (d / "prompt.md").write_text(set_frontmatter(src, f"{arm}-{n}", plugins, arm), encoding="utf-8")
            built.append(f"{arm}-{n}")
    return {"cases": built, "crowd_chars": crowd_chars}


def summarize(result: dict) -> tuple[str, list[dict]]:
    rows = {}
    for c in result.get("cases", []):
        arm, base = c["name"].split("-", 1)
        runs = c.get("arms", {}).get("with", [])
        def rate(grader):
            vals = [g["passed"] for r in runs for g in r.get("graders", []) if g["name"] == grader]
            return sum(vals) / len(vals) if vals else None
        fired = rate("skill-fired")
        if fired is None:                         # delegation case: the skill or the econometrician agent
            fired = rate("reached")
        if fired is None:                         # negative case: the skill must NOT fire
            nf = rate("skill-not-fired")
            fired = None if nf is None else 1 - nf
        rows.setdefault(base, {"case": base})
        rows[base][f"{arm} skill fired"] = fired
        rows[base][f"{arm} answer ok"] = rate("answer")
        rows[base][f"{arm} runs"] = len(runs)
        rows[base][f"{arm} errors"] = sum(1 for r in runs if r.get("error"))
    fmt = lambda v: "—" if v is None else f"{v:.0%}"           # noqa: E731
    out = ["| Case | Skill fired: clean | Skill fired: crowded | Answer ok: clean | Answer ok: crowded |",
           "|---|---|---|---|---|"]
    for r in rows.values():
        neg = " (must not fire)" if r["case"] == "unrelated-email" else ""
        out.append(f"| {r['case']}{neg} | {fmt(r.get('clean skill fired'))} | {fmt(r.get('crowded skill fired'))} | "
                   f"{fmt(r.get('clean answer ok'))} | {fmt(r.get('crowded answer ok'))} |")
    pos = [r for r in rows.values() if r["case"] != "unrelated-email"]
    for arm in ("clean", "crowded"):
        v = [r[f"{arm} skill fired"] for r in pos if r.get(f"{arm} skill fired") is not None]
        if v:
            out.append(f"\nRouting rate, {arm}: {sum(v) / len(v):.0%} over {len(v)} cases")
    return "\n".join(out), list(rows.values())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--case", nargs="*", help="routing case names or globs (default: all)")
    ap.add_argument("--model", help="model under test (pin it to compare releases)")
    ap.add_argument("--max-cost-usd", type=float)
    ap.add_argument("-j", "--concurrency", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("routing-results") / datetime.now().strftime("%Y%m%d-%H%M%S"))
    ap.add_argument("--build-only", type=Path, metavar="DIR", help="write the harness to DIR and stop")
    a = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):      # Windows consoles: never crash on "≈" or "—"
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    if a.build_only:
        info = build(a.build_only, a.case)
        print(f"Harness written to {a.build_only}: {len(info['cases'])} cases; decoy descriptions "
              f"{info['crowd_chars']} characters")
        return 0
    exe = shutil.which("claude")
    if exe is None:
        raise SystemExit("Claude Code CLI ('claude') not found on the PATH")
    a.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="econ-routing-") as tmp:
        harness = Path(tmp) / "harness"
        info = build(harness, a.case)
        print(f"{len(info['cases'])} cases (clean and crowded); decoy descriptions {info['crowd_chars']} characters")
        out_json = (a.out / "routing.json").resolve()
        cmd = [exe, "plugin", "eval", str(harness), "--trust-plugin", "--ablation", "none", "--runs", str(a.runs),
               "--no-publish", "-j", str(a.concurrency), "--output-dir", str(a.out.resolve()), "--json", str(out_json)]
        if a.model:
            cmd += ["--model", a.model]
        if a.max_cost_usd:
            cmd += ["--max-cost-usd", str(a.max_cost_usd)]
        proc = subprocess.run(cmd)
    if not out_json.exists():
        print("claude plugin eval wrote no result; exit code", proc.returncode)
        return proc.returncode or 1
    result = json.loads(out_json.read_text(encoding="utf-8"))
    table, _ = summarize(result)
    head = (f"Routing evals, {datetime.now():%Y-%m-%d %H:%M}, Claude Code {result.get('claudeVersion')}, "
            f"{a.runs} run(s) per case and arm, cost ≈ ${result.get('costUsd', 0):.2f}"
            + (" (PARTIAL: " + str(result.get("partialReason")) + ")" if result.get("partial") else ""))
    text = head + "\n\n" + table + "\n"
    (a.out / "summary.md").write_text(text, encoding="utf-8")
    print("\n" + text + f"\nSaved in {a.out.resolve()}")
    return 0 if proc.returncode in (0, 1) else proc.returncode


if __name__ == "__main__":
    sys.exit(main())
