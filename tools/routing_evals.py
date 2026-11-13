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

A third arm, run only when asked for (`--arm nameonly` or `--arm all`), reproduces the failure seen on the
user's notebook on 2026-11-07 (865 skills, 14 with a description, econometrics listed by name only):

    nameonly-<case>  the econometrics plugin plus about 850 decoy skills spread over 12 decoy plugins, so that
                     Claude Code keeps only a few descriptions (it drops the least-invoked skills' first)

Whether econometrics really lost its description is not assumed. The skill listing is not in the trace that
`claude plugin eval` hands to graders, but it is in the child session's own transcript, so this arm runs with
`--keep-temp`, reads `config/projects/**/*.jsonl` of each kept run (the part the eval tool itself writes), records
two diagnostics per run, `listing-captured` (a skill listing was found, a positive control) and
`description-dropped` (econometrics is listed by name only), and removes the kept directory. The summary reports
routing among the runs where both held; the diagnostics are saved in nameonly-diagnostics.json.
The plugin is copied into each case (the eval harness loads only plugins inside the case directory), with a
`plugin.json` built from the marketplace entry, so the copy declares the same skills, commands and agent.

Usage (from the repository root; needs Claude Code >= 2.1.269 and runs on your account):
    python tools/routing_evals.py                    # 3 runs per case and arm (~$3-6 at list price)
    python tools/routing_evals.py --runs 1 --case two-groups
    python tools/routing_evals.py --build-only out/  # write the harness and stop (no model calls)
    python tools/routing_evals.py --arm nameonly --runs 1 --case two-stores-promo two-groups excel-formula
    python tools/routing_evals.py --arm nameonly --no-hooks ...   # the same without the plugin's SessionStart note

The cases grant only read-only tools (no Bash), so they run on native Windows too. Results go to
routing-results/<timestamp>/ (summary.md, routing.json, the harness's report.html).

Since v2.6.7 the answers are kept: every arm runs with `--keep-temp`, the last message of each run (the `result`
field of the last line of its trace) is written to responses/<case>__run<i>.md with the graders' verdicts, an index
goes to responses.json, and the kept directories are removed. A failed answer can then be read without running the
case again. `--no-responses` turns this off (the name-only arm still keeps its directories to read the listing).
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


# The name-only arm: a listing the size of a real crowded installation. Names follow the "plugin:skill" shape of
# the real listing, and a share of them sound like data and statistics work, as in the user's listing.
NAMEONLY_PLUGINS = ("data-suite", "finance-desk", "research-kit", "legal-ops", "marketing-hub", "dev-workflow",
                    "education-kit", "strategy-kit", "productivity-pack", "analytics-office-x", "career-tools",
                    "consulting-kit")
NAMEONLY_TOPICS = (
    "statistical-analysis", "explore-data", "variance-analysis", "scenario-analysis", "sensitivity", "forecast",
    "experiment", "ab-test", "survey", "cohort", "kpi", "dashboard", "metrics", "data-quality", "sql", "excel",
    "pivot", "budget", "cash-flow", "valuation", "dcf", "comps", "lbo", "credit", "portfolio", "risk", "pricing",
    "market-sizing", "segmentation", "churn", "retention", "attribution", "campaign", "seo", "content", "brand",
    "contract", "compliance", "policy", "board-memo", "meeting-notes", "email", "slides", "report", "proposal",
    "roadmap", "okr", "hiring", "onboarding", "performance", "training", "lesson", "quiz", "rubric", "literature",
    "citation", "paper", "thesis", "code-review", "testing", "debugging", "deploy", "architecture", "security",
    "documentation", "spec", "research", "interview", "workshop", "negotiation")
NAMEONLY_FORMS = ("", "-review", "-builder", "-planner", "-checklist", "-audit", "-assistant", "-tracker",
                  "-guide", "-coach", "-template", "-helper")
NAMEONLY_FILLER = ("Use it when the user brings numbers, files, notes or a goal in this area, even without naming "
                   "the skill: it plans the work, drafts the output, checks it against a checklist and suggests "
                   "next steps, with templates for teams in finance, operations, product and HR.")


def nameonly_decoys() -> dict[str, list[tuple[str, str]]]:
    """{decoy plugin: [(skill name, description)]}, about 850 skills, the same on every call."""
    out: dict[str, list[tuple[str, str]]] = {p: [] for p in NAMEONLY_PLUGINS}
    i = 0
    for form in NAMEONLY_FORMS:
        for topic in NAMEONLY_TOPICS:
            name = topic + form
            words = name.replace("-", " ")
            out[NAMEONLY_PLUGINS[i % len(NAMEONLY_PLUGINS)]].append(
                (name, f"Helps with {words}: frameworks, worked examples and a ready-to-use output. {NAMEONLY_FILLER}"))
            i += 1
    return out


def write_nameonly_crowd(dst: Path) -> tuple[list[str], int, int]:
    """The decoy plugins of the name-only arm under dst/; returns (plugin dirs, skill count, description chars)."""
    dirs, n, chars = [], 0, 0
    for plugin, skills in nameonly_decoys().items():
        root = dst / plugin
        (root / ".claude-plugin").mkdir(parents=True)
        (root / ".claude-plugin" / "plugin.json").write_text(json.dumps(
            {"name": plugin, "version": "1.0.0", "description": "Decoy skills for routing evals (not a real plugin)."},
            indent=2) + "\n", encoding="utf-8")
        for name, desc in skills:
            d = root / "skills" / name
            d.mkdir(parents=True)
            (d / "SKILL.md").write_text(f'---\nname: {name}\ndescription: "{desc}"\n---\n\n# {name}\n\nAnswer the '
                                        "request directly, with a short explanation.\n", encoding="utf-8")
            n += 1
            chars += len(desc)
        dirs.append(plugin)
    return dirs, n, chars


def listing_diagnostics(sandbox: Path) -> dict:
    """Read the skill listing of one kept eval run from its session transcript (config/projects/**/*.jsonl).
    An entry is "- <name>" without a description or "- <name>: <description>" with it (format of the attachment
    "skill_listing", read from a real transcript on 2026-11-07). description-dropped is True only when
    econometrics:econometrics is listed AND no econometrics:* entry (econometrics:statistics included) keeps a
    description: a run in which the statistics entry point kept its description is not the failure under test.
    The main session transcript is read before any subagent transcript (shallower files first)."""
    out = {"listing-captured": False, "description-dropped": False, "skills": None, "with_description": None,
           "econometrics_described": None}
    files = sorted((sandbox / "config" / "projects").rglob("*.jsonl"),
                   key=lambda f: (len(f.relative_to(sandbox).parts), str(f)))
    for f in files:
        for raw in f.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            att = msg.get("attachment") if isinstance(msg, dict) else None
            if not isinstance(att, dict) or att.get("type") != "skill_listing" or not isinstance(att.get("content"), str):
                continue
            entries = [e for e in att["content"].splitlines() if e.startswith("- ")]
            described = sorted(e[2:].split(": ", 1)[0] for e in entries
                               if e.startswith("- econometrics:") and ": " in e[2:])
            out.update({"listing-captured": True, "skills": att.get("skillCount", len(entries)),
                        "with_description": sum(1 for e in entries if ": " in e[2:]),
                        "econometrics_described": described,
                        "description-dropped": "- econometrics:econometrics" in entries and not described})
            return out                            # the first listing is the one the request was answered with
    return out


def _is_eval_sandbox(path: Path) -> bool:
    """Only directories the eval tool creates under the system temp folder may be read and removed."""
    try:
        path.resolve().relative_to(Path(tempfile.gettempdir()).resolve())
    except ValueError:
        return False
    return path.name.startswith("claude-eval-") and (path / "out").is_dir()


def _kept_dir(run: dict) -> Path | None:
    tp = run.get("tracePath")
    return Path(tp).parent.parent if tp else None


def annotate_nameonly(result: dict) -> list[dict]:
    """Add the transcript diagnostics to every run of the name-only cases, as two graders the summary reads.
    A run whose kept directory cannot be read (no tracePath, or a path that is not an eval run directory under the
    temp folder) is marked unreadable, so the summary never reads it as "the failure was not reproduced".
    Returns one record per run (saved next to the results). Directories are removed by remove_kept_dirs."""
    records = []
    for c in result.get("cases", []):
        if not c.get("name", "").startswith("nameonly-"):
            continue
        for i, r in enumerate(c.get("arms", {}).get("with", [])):
            kept = _kept_dir(r)
            readable = kept is not None and _is_eval_sandbox(kept)
            empty = {"listing-captured": False, "description-dropped": False, "skills": None,
                     "with_description": None, "econometrics_described": None}
            try:
                diag = listing_diagnostics(kept) if readable else empty
            except OSError:              # a transcript locked or vanished (Windows): this run only is unreadable
                readable, diag = False, empty
            r["graders"] = [g for g in r.get("graders", []) if g.get("name") not in ("listing-captured",
                                                                                      "description-dropped")]
            r["graders"] += [{"name": "listing-captured", "passed": diag["listing-captured"], "source": "transcript"},
                             {"name": "description-dropped", "passed": diag["description-dropped"],
                              "source": "transcript"}]
            r["harnessUnreadable"] = not readable
            records.append({"case": c["name"], "run": i, "readable": readable,
                            "kept_dir": str(kept) if kept else None, **diag})
    return records


def _last_message(trace: Path) -> str | None:
    """The final answer of a run: the `result` field of the last trace line that has one."""
    try:
        lines = trace.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and isinstance(rec.get("result"), str):
            return rec["result"]
    return None


def save_responses(result: dict, dest: Path) -> list[dict]:
    """Write the last message of every run to dest/<case>__run<i>.md, with the graders' verdicts on top, and return
    one index record per run. Only traces inside an eval run directory under the temp folder are read; any other
    run is recorded with readable False (and no file)."""
    dest.mkdir(parents=True, exist_ok=True)
    records = []
    for c in result.get("cases", []):
        name = c.get("name", "")
        for i, r in enumerate(c.get("arms", {}).get("with", [])):
            kept, tp = _kept_dir(r), r.get("tracePath")
            msg = _last_message(Path(tp)) if tp and kept is not None and _is_eval_sandbox(kept) else None
            graders = {g.get("name"): g.get("passed") for g in r.get("graders", [])}
            rec = {"case": name, "run": i, "readable": msg is not None, "graders": graders, "file": None}
            if msg is not None:
                stem = f"{re.sub(r'[^A-Za-z0-9_.-]', '_', name)}__run{i}"
                f, k = dest / f"{stem}.md", 2
                while f.exists():                      # two case names that clean to the same file name
                    f, k = dest / f"{stem}_{k}.md", k + 1
                head = "\n".join(f"- {k}: {'pass' if v else 'FAIL'}" for k, v in graders.items())
                f.write_text(f"# {name}, run {i}\n\n{head}\n\n---\n\n{msg}\n", encoding="utf-8")
                rec["file"] = f.name
            records.append(rec)
    return records


def remove_kept_dirs(result: dict) -> list[str]:
    """Remove the kept directory of every run, in every arm (--keep-temp keeps them all). Returns the paths that
    could not be removed or were refused (not an eval run directory under the temp folder), to print."""
    left, untraced = [], 0
    for c in result.get("cases", []):
        for r in c.get("arms", {}).get("with", []):
            kept = _kept_dir(r)
            if kept is None:
                untraced += 1
                continue
            if _is_eval_sandbox(kept):
                shutil.rmtree(kept, ignore_errors=True)
            if kept.exists():
                left.append(str(kept))
    if untraced:
        left.append(f"{untraced} run(s) gave no trace path; look for leftover "
                    f"{Path(tempfile.gettempdir()) / 'claude-eval-*'} directories")
    return left


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


def write_plugin_copy(dst: Path, hooks: bool = True) -> None:
    """Copy of econometrics/ with a plugin.json built from the marketplace entry (no evals inside). hooks=False
    leaves out hooks/, to measure the plugin with and without its SessionStart note in the same commit."""
    skip = ("evals", "__pycache__", "*.pyc") + (() if hooks else ("hooks",))
    shutil.copytree(PLUGIN, dst, ignore=shutil.ignore_patterns(*skip))
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


ARMS = ("clean", "crowded", "nameonly")
ARM_CHOICES = {"both": ("clean", "crowded"), "all": ARMS, "clean": ("clean",), "crowded": ("crowded",),
               "nameonly": ("nameonly",)}


def build(harness: Path, cases: list[str] | None = None, arms=("clean", "crowded"), hooks: bool = True) -> dict:
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
    unknown = set(arms) - set(ARMS)
    if unknown:
        raise ValueError(f"unknown arm(s): {sorted(unknown)}")
    crowd_chars = 0
    nameonly = {"skills": 0, "chars": 0}
    built = []
    for n in names:
        for arm in arms:
            d = harness / "evals" / f"{arm}-{n}"
            shutil.copytree(CASES / n / "graders", d / "graders")
            write_plugin_copy(d / "econometrics", hooks=hooks)
            plugins = ["econometrics"]
            if arm == "crowded":
                crowd_chars = write_crowd(d / "crowd")
                plugins.append("crowd")
            elif arm == "nameonly":
                dirs, nameonly["skills"], nameonly["chars"] = write_nameonly_crowd(d)
                plugins += dirs
            src = (CASES / n / "prompt.md").read_text(encoding="utf-8")
            (d / "prompt.md").write_text(set_frontmatter(src, f"{arm}-{n}", plugins, arm), encoding="utf-8")
            built.append(f"{arm}-{n}")
    return {"cases": built, "crowd_chars": crowd_chars, "nameonly_skills": nameonly["skills"],
            "nameonly_chars": nameonly["chars"]}


def summarize(result: dict) -> tuple[str, list[dict]]:
    rows = {}
    for c in result.get("cases", []):
        arm, base = c["name"].split("-", 1)
        if arm == "nameonly":                     # reported apart, over the runs that reproduce the failure
            continue
        runs = c.get("arms", {}).get("with", [])
        def rate(grader):
            vals = [g["passed"] for r in runs for g in r.get("graders", []) if g["name"] == grader]
            return sum(vals) / len(vals) if vals else None
        fired = rate("skill-fired")
        if fired is None:                         # delegation case: the skill or the econometrician agent
            fired = rate("reached")
        negative = False
        if fired is None:                         # negative case: the skill must NOT fire
            nf = rate("skill-not-fired")
            fired = None if nf is None else 1 - nf
            negative = nf is not None
        rows.setdefault(base, {"case": base, "negative": False})
        rows[base]["negative"] |= negative
        rows[base][f"{arm} skill fired"] = fired
        rows[base][f"{arm} answer ok"] = rate("answer")
        rows[base][f"{arm} runs"] = len(runs)
        rows[base][f"{arm} errors"] = sum(1 for r in runs if r.get("error"))
    fmt = lambda v: "—" if v is None else f"{v:.0%}"           # noqa: E731
    out = ["| Case | Skill fired: clean | Skill fired: crowded | Answer ok: clean | Answer ok: crowded |",
           "|---|---|---|---|---|"]
    for r in rows.values():
        neg = " (must not fire)" if r["negative"] else ""
        out.append(f"| {r['case']}{neg} | {fmt(r.get('clean skill fired'))} | {fmt(r.get('crowded skill fired'))} | "
                   f"{fmt(r.get('clean answer ok'))} | {fmt(r.get('crowded answer ok'))} |")
    pos = [r for r in rows.values() if not r["negative"]]
    neg_rows = [r for r in rows.values() if r["negative"]]
    for arm in ("clean", "crowded"):
        v = [r[f"{arm} skill fired"] for r in pos if r.get(f"{arm} skill fired") is not None]
        if v:
            out.append(f"\nRouting rate, {arm}: {sum(v) / len(v):.0%} over {len(v)} cases")
        w = [r[f"{arm} skill fired"] for r in neg_rows if r.get(f"{arm} skill fired") is not None]
        if w:
            out.append(f"False triggers, {arm}: {sum(w) / len(w):.0%} over {len(w)} unrelated requests")
    nameonly = nameonly_summary(result)
    if nameonly:
        out.append("\n" + nameonly)
    return "\n".join(out), list(rows.values())


def _grade(run: dict, name: str):
    """True/False for the grader `name` in one run, None when the run has no such grader."""
    vals = [g["passed"] for g in run.get("graders", []) if g["name"] == name]
    return all(vals) if vals else None


def _errored_before_grading(run: dict) -> bool:
    """A run that ended on an error (for example a usage limit) before any routing or answer grader ran."""
    return bool(run.get("error")) and not any(
        _grade(run, g) is not None for g in ("skill-fired", "reached", "skill-not-fired", "answer"))


def nameonly_summary(result: dict) -> str:
    """Name-only arm: routing counted only over the runs whose transcript shows econometrics listed by name only.
    A run where the listing was not captured, or where econometrics kept its description, does not test the
    failure of 2026-11-07, so it is reported but left out of the rate."""
    lines, pos, neg, unreadable, ungraded, graded, graded_captured = [], [], [], 0, 0, 0, 0
    for c in result.get("cases", []):
        arm, base = c["name"].split("-", 1)
        if arm != "nameonly":
            continue
        runs = c.get("arms", {}).get("with", [])
        captured = sum(1 for r in runs if _grade(r, "listing-captured"))
        unreadable += sum(1 for r in runs if r.get("harnessUnreadable"))
        valid = [r for r in runs if _grade(r, "listing-captured") and _grade(r, "description-dropped")]
        ungraded += sum(1 for r in runs if _errored_before_grading(r))
        graded += sum(1 for r in runs if not _errored_before_grading(r))
        graded_captured += sum(1 for r in runs if not _errored_before_grading(r) and _grade(r, "listing-captured"))
        negative = any(_grade(r, "skill-not-fired") is not None for r in runs)
        fired = []
        for r in valid:
            f = _grade(r, "skill-fired")
            if f is None:
                f = _grade(r, "reached")
            if f is None and _grade(r, "skill-not-fired") is not None:
                f = not _grade(r, "skill-not-fired")
            if f is not None:
                fired.append(f)
        answer = [a for a in (_grade(r, "answer") for r in valid) if a is not None]
        rate = (lambda v: "—" if not v else f"{sum(v)}/{len(v)}")           # noqa: E731
        lines.append(f"| {base}{' (must not fire)' if negative else ''} | {len(runs)} | {captured} | {len(valid)} | "
                     f"{rate(fired)} | {rate(answer)} |")
        if fired:
            (neg if negative else pos).append(sum(fired) / len(fired))
    if not lines:
        return ""
    head = ["Name-only arm (econometrics listed without its description)",
            "",
            "| Case | Runs | Listing in transcript | Name only | Skill fired (name-only runs) | Answer ok (name-only runs) |",
            "|---|---|---|---|---|---|"]
    tail = []
    if pos:
        tail.append(f"\nRouting rate, nameonly: {sum(pos) / len(pos):.0%} over {len(pos)} cases")
    if neg:
        tail.append(f"False triggers, nameonly: {sum(neg) / len(neg):.0%} over {len(neg)} unrelated requests")
    if unreadable:
        tail.append(f"\nThe harness could not read the kept directory of {unreadable} run(s) (path outside the temp "
                    "folder, or a layout it does not know): those runs say nothing about Claude Code.")
    if ungraded:
        tail.append(f"\n{ungraded} name-only run(s) ended on an error before grading (for example a usage limit): "
                    "they are not counted. Re-run them.")
    if not pos and not neg:
        if not graded:
            tail.append("\nNo graded run: no routing rate is reported, and nothing is concluded about the failure.")
        elif not graded_captured:
            tail.append("\nThe skill listing was not found in any transcript: a harness or transcript-format problem "
                        "(for example a new Claude Code version), not a result. Nothing is concluded about the failure.")
        else:
            tail.append("\nNo readable run showed econometrics listed by name only" + (
                "" if not unreadable else " (see the unreadable runs above)") + ": this arm did not reproduce the "
                        "failure, and its routing rate is not reported.")
    tail.append("Rates are per case (mean over cases); routing.json and the eval report count every run, "
                "including those that did not reproduce the failure.")
    return "\n".join(head + lines + tail)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--case", nargs="*", help="routing case names or globs (default: all)")
    ap.add_argument("--model", help="model under test (pin it to compare releases)")
    ap.add_argument("--max-cost-usd", type=float)
    ap.add_argument("-j", "--concurrency", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("routing-results") / datetime.now().strftime("%Y%m%d-%H%M%S"))
    ap.add_argument("--arm", choices=tuple(ARM_CHOICES), default="both",
                    help="both = clean and crowded (default); nameonly = the ~850-skill listing in which econometrics "
                         "keeps only its name; all = the three")
    ap.add_argument("--build-only", type=Path, metavar="DIR", help="write the harness to DIR and stop")
    ap.add_argument("--no-responses", action="store_true",
                    help="do not keep the answers (runs without --keep-temp, except the name-only arm)")
    ap.add_argument("--no-hooks", action="store_true",
                    help="leave the plugin's hooks out of the copies (baseline for measuring the SessionStart note)")
    a = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):      # Windows consoles: never crash on "≈" or "—"
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    if a.build_only:
        info = build(a.build_only, a.case, ARM_CHOICES[a.arm], hooks=not a.no_hooks)
        print(f"Harness written to {a.build_only}: {len(info['cases'])} cases; decoy descriptions "
              f"{info['crowd_chars']} characters" + (f"; name-only arm: {info['nameonly_skills']} decoy skills, "
                                                    f"{info['nameonly_chars']} characters" if info["nameonly_skills"] else ""))
        return 0
    exe = shutil.which("claude")
    if exe is None:
        raise SystemExit("Claude Code CLI ('claude') not found on the PATH")
    a.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="econ-routing-") as tmp:
        harness = Path(tmp) / "harness"
        info = build(harness, a.case, ARM_CHOICES[a.arm], hooks=not a.no_hooks)
        print(f"{len(info['cases'])} cases ({a.arm}); decoy descriptions {info['crowd_chars']} characters"
              + (f"; name-only arm: {info['nameonly_skills']} decoy skills" if info["nameonly_skills"] else ""))
        out_json = (a.out / "routing.json").resolve()
        cmd = [exe, "plugin", "eval", str(harness), "--trust-plugin", "--ablation", "none", "--runs", str(a.runs),
               "--no-publish", "-j", str(a.concurrency), "--output-dir", str(a.out.resolve()), "--json", str(out_json)]
        if "nameonly" in ARM_CHOICES[a.arm] or not a.no_responses:
            cmd.append("--keep-temp")             # answers and listing are read from each run's directory, then removed
        if a.model:
            cmd += ["--model", a.model]
        if a.max_cost_usd:
            cmd += ["--max-cost-usd", str(a.max_cost_usd)]
        proc = subprocess.run(cmd)
    keep = "--keep-temp" in cmd
    if not out_json.exists():
        print("claude plugin eval wrote no result; exit code", proc.returncode)
        if keep:
            print(f"Kept run directories (claude-eval-*) may remain in {tempfile.gettempdir()}: remove them.")
        return proc.returncode or 1
    result = json.loads(out_json.read_text(encoding="utf-8"))
    try:
        if keep and "nameonly" in ARM_CHOICES[a.arm]:
            records = annotate_nameonly(result)
            (a.out / "nameonly-diagnostics.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        if keep and not a.no_responses:
            try:
                resp = save_responses(result, a.out / "responses")
            except Exception as exc:                  # a paid run must still get its summary
                resp = []
                print(f"WARNING: answers not saved ({type(exc).__name__}: {exc})")
            (a.out / "responses.json").write_text(json.dumps(resp, indent=2) + "\n", encoding="utf-8")
            unread = sum(not x["readable"] for x in resp)
            print(f"Answers of {len(resp) - unread} run(s) saved in {(a.out / 'responses').resolve()}"
                  + (f"; {unread} run(s) had no readable trace" if unread else ""))
    finally:
        if keep:
            left = remove_kept_dirs(result)
            paths = [x for x in left if "run(s) gave no trace path" not in x]
            if paths:
                print("Kept run directories not removed (remove them by hand):\n  " + "\n  ".join(paths))
            for x in left:
                if x not in paths:
                    print(x)
    table, _ = summarize(result)
    head = (f"Routing evals, {datetime.now():%Y-%m-%d %H:%M}, Claude Code {result.get('claudeVersion')}, "
            f"{a.runs} run(s) per case and arm, plugin hooks {'off' if a.no_hooks else 'on'}, "
            f"cost ≈ ${result.get('costUsd', 0):.2f}"
            + (" (PARTIAL: " + str(result.get("partialReason")) + ")" if result.get("partial") else ""))
    text = head + "\n\n" + table + "\n"
    (a.out / "summary.md").write_text(text, encoding="utf-8")
    print("\n" + text + f"\nSaved in {a.out.resolve()}")
    return 0 if proc.returncode in (0, 1) else proc.returncode


if __name__ == "__main__":
    sys.exit(main())
