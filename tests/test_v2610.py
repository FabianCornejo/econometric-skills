"""Tests for v2.6.10 (core): the number rule in SKILL.md section 4, the repaired delivery sentence, the locator paths
kept in one place, and the doctor's closing lines when the latest release was not checked.

Origin: the v2.6.9 validation (notebook 2, 2026-11-14). (1) The reply worked out two numbers outside the tables (a
combined slope and an MDE), one of them flagged as "my own arithmetic"; the v2.6.8 rule had failed the same way in the
v2.6.8 validation, there inside a table. (2) The agent ran `setup_env.py --doctor --offline 2>&1 | Select-Object -Last 25` on a computer with
network; reading the doctor then showed that with --offline and nothing else to do it closed with "the setup is
complete and up to date", although no version had been checked. (3) SKILL.md section 4 had a cut sentence
("... which the system may delete. and test it on data ...", review of 2026-11-13, item #6).

Checks:
  1. section 4: the delivery rule reads "Test the script on data that include the awkward cases"; no ". and test";
     it still points to the locator block and still forbids the versioned cache path and sandbox paths;
  2. what left section 4 (the locator's paths and the install message) is in the "Script skeleton" of
     interpretation-rules.md, literally;
  3. section 4's number rule covers numbers outside the tables and says a disclaimer does not exempt them;
     interpretation-rules.md has "Rules added in v2.6.10", after v2.6.8, with the validation's two numbers and the
     functions it names (which exist);
  4. the doctor's text: never "up to date" when the latest release was not checked; with --offline the last line
     asks to run it again without the flag (so a tail of the output shows it); the JSON actions do not change.

Run from the repository root:  python3 tests/test_v2610.py
"""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "econometrics"
sys.path.insert(0, str(SKILL / "scripts"))
import _console  # noqa: E402,F401  (non-ASCII text on a Windows console)

import estimation  # noqa: E402
import setup_env as se  # noqa: E402
import simulation  # noqa: E402

results = []


def check(name, cond, info=""):
    results.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info)


skill_md = (SKILL / "SKILL.md").read_text(encoding="utf-8")
sec4 = skill_md[skill_md.index("## 4. Non-negotiable rules"):skill_md.index("## 5.")]
rules = (SKILL / "references" / "interpretation-rules.md").read_text(encoding="utf-8")

# ----------------------------------------------------------------------------------- 1. the delivery rule
deliver = next((ln for ln in sec4.splitlines() if ln.startswith("- **Deliver code, not only a plan.**")), "")
check("section 4: the delivery rule is found by its title", bool(deliver))
check("section 4: the cut sentence is repaired ('Test the script on data that include the awkward cases')",
      "which the system may delete. Test the script on data that include the awkward cases (skipped events, missing "
      "values, repeated units)." in deliver and ". and test" not in skill_md, deliver[-200:])
check("section 4: the delivery rule still starts the script with the locator block of interpretation-rules.md",
      "Start the script with the locator block in `references/interpretation-rules.md` (\"Script skeleton\")"
      in deliver)
check("section 4: still never a sandbox path or the versioned cache folder, not even in a comment; the variable "
      "for your own environment is named",
      "never put a sandbox path or the versioned `plugins/cache/.../<version>/` folder in it, not even in a comment"
      in deliver and "set `ECONOMETRICS_SKILL_DIR` instead" in deliver)
check("section 4: the locator's paths are no longer repeated in the rule (one place: the Script skeleton)",
      "plugins/marketplaces" not in deliver and "skills/synced" not in deliver and "It also turns" not in deliver)
check("section 4: the delivery rule is shorter than in v2.6.9 (259 words)", len(deliver.split()) < 259,
      len(deliver.split()))

# ----------------------------------------------------------------------------------- 2. what left section 4
skel = rules[rules.index("Script skeleton for every deliverable"):rules.index("## Rules added in v2.3.1")]
for piece in ('os.environ.get("ECONOMETRICS_SKILL_DIR")',
              '".claude/plugins/marketplaces/econometric-skills/econometrics"',
              "including the copy synced from the Claude app",
              "<path to your clone>/econometric-skills/econometrics",
              'scripts/setup_env.py\\" --install',
              "plugins/cache/econometric-skills/econometrics/<version>/"):
    check(f"Script skeleton keeps what left section 4: {piece}", piece in skel)

# ----------------------------------------------------------------------------------- 3. the number rule
num = next((ln for ln in sec4.splitlines() if ln.startswith("- **No invented numbers**")), "")
check("section 4: the number rule is found by its title", bool(num))
check("section 4: every number the reply computes or derives is printed by the script, in reply, summary and document",
      "Every number the reply computes or derives" in num
      and "a sum of coefficients or a combined slope, an MDE, an interval, a percentage or a ratio" in num
      and "in the reply, its summary and the document, is printed by the script before it is written" in num)
check("section 4: a number quoted unchanged from the user or a cited source is attributed, not recomputed "
      "(agrees with the rule's first sentence, 'or a cited source')",
      "a number quoted unchanged from the user or a cited source is attributed to it" in num
      and "comes from a computation run in the session or a cited source." in num)
check("section 4: a disclaimer ('my own arithmetic') does not exempt a number",
      "saying it is your own arithmetic does not exempt it" in num and "v2.6.8 and v2.6.10" in num)
heads = [ln for ln in rules.splitlines() if ln.startswith("## Rules added in v2.6.")]
check("interpretation-rules.md: 'Rules added in v2.6.10' exists once, after v2.6.8",
      heads.count("## Rules added in v2.6.10") == 1 and "## Rules added in v2.6.8" in heads
      and heads.index("## Rules added in v2.6.10") > heads.index("## Rules added in v2.6.8"), heads)
start = rules.index("## Rules added in v2.6.10")
nxt = rules.find("\n## ", start + 5)
v2610 = rules[start:nxt if nxt > 0 else len(rules)]
check("v2.6.10 section: the validation's two numbers (combined slope, MDE) and the flagged disclaimer",
      "−0.5174 + 0.4841" in v2610 and "2.8 × SE" in v2610 and "my own arithmetic" in v2610)
check("v2.6.10 section: outside the tables; a disclaimer does not replace the script; add the print and rerun",
      "outside the tables" in v2610 and "a disclaimer does not replace the script" in v2610
      and "add the print, run the script again" in v2610)
check("v2.6.10 section: the functions it names exist (estimation.delta_method, simulation.power_mde)",
      "`estimation.delta_method(" in v2610 and "`simulation.power_mde`" in v2610
      and callable(getattr(estimation, "delta_method", None)) and callable(getattr(simulation, "power_mde", None)))
check("v2.6.10 section: linearmodels fits (no t_test) use delta_method with res.cov; few clusters -> t(G - 1) "
      "(review F7); the MDE of an estimated coefficient comes from its SE, power_mde only for two means (review F6)",
      "for linearmodels fits (IV, panel), which have no `t_test`" in v2610 and "res.params, res.cov)" in v2610
      and "with few clusters, build the interval from its printed SE with t(G − 1)" in v2610
      and "The minimum detectable effect of a coefficient already estimated, binary or continuous, is printed from "
          "that coefficient's SE inside the script" in v2610
      and "with both z values computed in the script (about 2.8 × SE at 5% and 80% power; never type 2.8 or 1.96)" in v2610
      and "`simulation.power_mde` is for planning, before the data exist" in v2610
      and "the v2.6.8 validation had already shown the rule failing inside a table" in v2610)
NR = "after estimation, (z₁₋α/₂ + z_power) × the coefficient's SE, printed by the script"
check("section 4 and the v2.4.0 rule agree with v2.6.10 on the detectable effect after a non-rejection "
      "(review r3 A SF-2): power_curve; power_mde when planning; after estimation, from the coefficient's SE",
      f"(`inference.power_curve`; `simulation.power_mde` when planning; {NR})." in sec4
      and f"(`inference.power_curve`; `simulation.power_mde` when planning; {NR})." in rules
      and abs(__import__("scipy.stats", fromlist=["norm"]).norm.ppf(0.975)
              + __import__("scipy.stats", fromlist=["norm"]).norm.ppf(0.8) - 2.8) < 0.005
      and "A number quoted unchanged from the user or a cited source is attributed to it." in v2610)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from linearmodels.iv import IV2SLS  # noqa: E402
_rng = np.random.default_rng(26100)
_n = 400
_z, _w, _u = _rng.normal(size=_n), _rng.binomial(1, 0.5, _n), _rng.normal(size=_n)
_x = _z + _u + _rng.normal(size=_n)
_d = pd.DataFrame({"y": 1 + 0.5 * _x + 0.3 * _w + _u, "x": _x, "w": _w, "z": _z})
_d["xw"], _d["zw"] = _d.x * _d.w, _d.z * _d.w
res = IV2SLS.from_formula("y ~ 1 + w + [x + xw ~ z + zw]", _d).fit(cov_type="robust")
i, j = list(res.params.index).index("x"), list(res.params.index).index("xw")
dm = estimation.delta_method(lambda t: t[i] + t[j], res.params, res.cov)      # the call the rule writes
by_hand_se = float(np.sqrt(res.cov.loc[["x", "xw"], ["x", "xw"]].to_numpy().sum()))
check("v2.6.10 section: the documented delta_method call works on a linearmodels IV fit and gives b_x + b_xw with "
      "the SE from the covariance matrix (Var(a + b) = Var a + Var b + 2 Cov)",
      not hasattr(res, "t_test") and abs(dm["estimate"].iloc[0] - (res.params["x"] + res.params["xw"])) < 1e-12
      and abs(dm["se"].iloc[0] - by_hand_se) < 1e-10, (dm.to_dict(), by_hand_se))


# ----------------------------------------------------------------------------------- 4. the doctor's text
def report(**over):
    rep = {"python": {"version": "3.13.1", "exe": "py", "supported": True, "run_with": "py -3.13"},
           "interpreters": [], "core_missing": [], "extras": {},
           "skill": {"version": "2.6.10", "dir": "d"}, "latest": {"version": "2.6.10", "status": "ok"},
           "copies": [], "autoupdate": {"registered": False},
           "listing": {"checked": True, "folder": "f", "skills": 12, "enabled_plugins": 2, "crowded": False},
           "actions": []}
    rep.update(over)
    return rep


def shown(rep):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        se.print_doctor(rep)
    return buf.getvalue()


OFF = {"version": None, "status": "not checked (--offline)"}
AGAIN = "if this computer has network, run --doctor again without --offline before answering"
last = lambda o: (o.rstrip("\n").splitlines() or [""])[-1]  # noqa: E731


class _Reply(io.BytesIO):                                  # a context manager like urlopen's response
    def __exit__(self, *a):
        return False


def produced(opener):
    """(version, status) from the real producer, setup_env.latest_release (review B-1: no invented statuses)."""
    v, s = se.latest_release(opener=opener)
    return {"version": v, "status": s}


def _fail(*a, **k):
    raise urllib.error.URLError("no route")


ERR = produced(_fail)                                                   # "not checked (URLError)"
NOVER = produced(lambda *a, **k: _Reply(b'{"tag_name": ""}'))           # "not checked (no version in the reply)"
OK = produced(lambda *a, **k: _Reply(b'{"tag_name": "v2.6.10"}'))       # ("2.6.10", "ok")
check("doctor text: the statuses come from latest_release (network error, empty tag, a version)",
      ERR == {"version": None, "status": "not checked (URLError)"}
      and NOVER == {"version": None, "status": "not checked (no version in the reply)"}
      and OK == {"version": "2.6.10", "status": "ok"}, (ERR, NOVER, OK))
ACT = ["crowded skill listing: offer a clean analysis folder"]
o_ok = shown(report(latest=OK))
o_off = shown(report(latest=OFF))
o_off_act = shown(report(latest=OFF, actions=ACT))
check("doctor text: checked and nothing to do -> 'up to date', and no line about --offline",
      "Nothing to do: the setup is complete and up to date." in o_ok and AGAIN not in o_ok, o_ok)
check("doctor text: --offline and nothing to do -> never 'up to date'; says the release was not checked",
      "up to date" not in o_off and "Nothing to do in this setup; the latest release was not checked (--offline), "
      "so the skill may be out of date." in o_off, o_off)
check("doctor text: --offline -> the last line asks to run it again without the flag (a tail still shows it)",
      AGAIN in last(o_off) and AGAIN in last(o_off_act), (last(o_off), last(o_off_act)))
for name, lt in (("network error", ERR), ("reply without a version", NOVER)):
    o_none, o_act = shown(report(latest=lt)), shown(report(latest=lt, actions=ACT))
    check(f"doctor text: {name}, with and without To do items -> never 'up to date'; the last line says the "
          "release was not checked; no --offline line",
          "up to date" not in o_none + o_act and AGAIN not in o_none + o_act
          and f"the latest release was {lt['status']}, so the skill" in last(o_none)
          and last(o_act) == f"The latest release was {lt['status']}, so the skill may be out of date.",
          (last(o_none), last(o_act)))
check("doctor text: the v2.6.6 line for --offline and the To do header stay",
      "--offline is only for a machine without network; with network, run --doctor without it" in o_off
      and "\n  1. crowded skill listing" in o_off_act and "To do (ask the user once" in o_off_act, o_off_act)
calls = []
with tempfile.TemporaryDirectory() as tmp:     # isolated, as in test_v266: no real ~/.claude, no CLI, no network
    rep_off = se.doctor(offline=True, cdir=Path(tmp), env={}, opener=lambda *a, **k: calls.append(a),
                        run=lambda *a, **k: (1, ""), which=lambda n: None, cwd=tmp)
    rep_err = se.doctor(offline=False, cdir=Path(tmp), env={}, opener=_fail,
                        run=lambda *a, **k: (1, ""), which=lambda n: None, cwd=tmp)
check("doctor --offline: no request; the JSON status is unchanged and --offline adds no To do item (review B-3: "
      "the isolated setup has none, and a network error gives the same list)",
      calls == [] and json.loads(json.dumps(rep_off))["latest"] == OFF and rep_off["actions"] == []
      and rep_off["actions"] == rep_err["actions"] and rep_err["latest"] == ERR, (rep_off["actions"], rep_err["actions"]))
check("doctor --offline, report built by doctor() and printed: ends with the line asking for a run without --offline",
      AGAIN in last(shown(rep_off)) and "complete and up to date" not in shown(rep_off), last(shown(rep_off)))
import shutil  # noqa: E402


def run_cli(args, **extra_env):
    """setup_env.py run as the agent runs it, isolated: temp HOME/USERPROFILE/CLAUDE_CONFIG_DIR and no `claude` on the
    PATH (any extension Windows tries: review r2 B-2.4). Returns (returncode, non-empty lines of stdout+stderr,
    stdout); a timeout returns code 124 (review r2 B-2.5: a FAIL line, not a traceback)."""
    with tempfile.TemporaryDirectory() as tmp:
        env = {k: v for k, v in os.environ.items()
               if not k.upper().startswith(("CLAUDE", "ECONOMETRICS")) and k.upper() not in ("NO_PROXY",)}
        env.update(HOME=tmp, USERPROFILE=tmp, CLAUDE_CONFIG_DIR=str(Path(tmp) / ".claude"), PYTHONIOENCODING="utf-8",
                   PATH=os.pathsep.join(p for p in env.get("PATH", "").split(os.pathsep)   # r3 B1-4: by file, not
                                        if p and not any((Path(p) / f"claude{x}").is_file()   # which (it also looks
                                                         for x in ("", ".exe", ".cmd", ".bat", ".ps1"))))  # in the cwd)
        env.update(extra_env)
        try:
            cli = subprocess.run([sys.executable, str(SKILL / "scripts" / "setup_env.py"), *args, "--folder", tmp],
                                 capture_output=True, text=True, encoding="utf-8", env=env, cwd=tmp, timeout=300)
        except subprocess.TimeoutExpired:
            return 124, [], "", "TimeoutExpired"
    return cli.returncode, [ln for ln in (cli.stdout + cli.stderr).splitlines() if ln.strip()], cli.stdout, cli.stderr


rc, lines, _, _ = run_cli(["--doctor", "--offline"])
check("doctor --offline from the command line (stdout and stderr): the last line asks for a run without --offline",
      rc == 0 and bool(lines) and AGAIN in lines[-1], (rc, lines[-3:]))
import socket  # noqa: E402
import threading  # noqa: E402

seen = []                                         # review r3 B1-1: a local proxy that records the request, then drops it


def _proxy(srv):
    try:
        conn, _ = srv.accept()
        with conn:
            conn.settimeout(10)
            seen.append(conn.recv(4096).decode("latin-1").split("\r\n", 1)[0])
    except OSError:
        pass


_srv = socket.socket()
_srv.bind(("127.0.0.1", 0))
_srv.listen(1)
_t = threading.Thread(target=_proxy, args=(_srv,), daemon=True)
_t.start()
DEAD = f"http://127.0.0.1:{_srv.getsockname()[1]}"
rc_e, lines_e, _, _ = run_cli(["--doctor"], HTTPS_PROXY=DEAD, https_proxy=DEAD, HTTP_PROXY=DEAD, http_proxy=DEAD)
_t.join(5)
_srv.close()
check("doctor without --offline really asks GitHub for the latest release (through the proxy it was given)",
      bool(seen) and seen[0].startswith("CONNECT api.github.com:443"), seen)
check("doctor with a network error, from the command line: the last line says the release was not checked; "
      "never 'up to date'; no --offline line",
      rc_e == 0 and bool(lines_e) and lines_e[-1].startswith("Nothing to do in this setup; the latest release was not "
                                                             "checked (")
      and lines_e[-1].endswith("so the skill may be out of date.")
      and not any("up to date" in ln for ln in lines_e) and not any(AGAIN in ln for ln in lines_e), (rc_e, lines_e[-2:]))
rc_j, _, out_j, err_j = run_cli(["--doctor", "--offline", "--json"])
try:
    js = json.loads(out_j)
except ValueError:
    js = {}
check("doctor --offline --json from the command line: the same keys as v2.6.9, latest 'not checked (--offline)', no "
      "To do item, and nothing printed after the JSON (review r2 B-2.2)",
      rc_j == 0 and set(js) == {"claude_dir", "account", "python", "core_missing", "extras", "skill", "copies",
                                "autoupdate", "listing", "interpreters", "latest", "actions"}
      and js.get("latest") == OFF and js.get("actions") == [] and err_j.strip() == "",
      (rc_j, sorted(js), js.get("latest"), err_j[-200:]))
check("doctor text: --offline with To do items -> the line before the last says the release was not checked "
      "(review r2 B-2.3)",
      o_off_act.rstrip("\n").splitlines()[-2:-1] == ["The latest release was not checked (--offline), so the skill may "
                                                    "be out of date."], o_off_act.splitlines()[-3:])

# ----------------------------------------------------------------------------------- 5. description (item 3.1)
import yaml  # noqa: E402
_fm = skill_md.splitlines()
_end = _fm.index("---", 1) if _fm and _fm[0] == "---" and "---" in _fm[1:] else 0
desc = (yaml.safe_load("\n".join(_fm[1:_end])) or {}).get("description", "") if _end else ""
check("description: at most 1024 characters (the 990 kept room for v2.7.0; v2.8.1 used it for GMM and sequential tests, test_v281)", 0 < len(desc) <= 1024, len(desc))
kept = ["hypothesis", "two groups", "Welch", "proportions", "variances", "chi-square", "normality", "correlation",
        "power and sample size", "regression", "effect of X on Y", "VaR", "WACC", "forecast", "DiD",
        "even a quick one you could do by hand", "an interval for a proportion", "a sample size", "a p-value",
        "a conceptual one", "review a conclusion, report or paper", "catch errors of answers from memory",
        "teaching a method", "not data, finance or risk skills", "consultant", "teacher", "reviewer", "FRTB",
        "even without the word econometrics", "event studies", "time-to-event", "elasticity", "choice of controls"]
check("description: every phrase the earlier tests and the routing history need is still there",
      all(k in desc for k in kept), [k for k in kept if k not in desc])
check("description: the methods sit inside the tasks; no separate 'Methods:' list",
      "Methods:" not in desc and "any regression (OLS, IV, panel, logit/probit)" in desc
      and "policy evaluation (DiD, RD, synthetic control)" in desc and "forecast (ARIMA, VAR, GARCH)" in desc)

# ----------------------------------------------------------------------------------- 6. unit roots (item 3.2)
rows = [ln for ln in skill_md.splitlines() if ln.startswith("| ")]
one = next((r for r in rows if r.startswith("| One series:")), "")
order = next((r for r in rows if r.startswith("| Regression with time series")), "")
ts_md = (SKILL / "references" / "time-series.md").read_text(encoding="utf-8")
check("routing: one series -> unit root names which function comes first (stationarity, then adf_bootstrap)",
      "unit root (`diagnostics.stationarity` first; `adf_bootstrap` for bootstrap p-values in a short series)" in one)
import specification  # noqa: E402
check("routing: the generic 'Time series (levels, lags, cointegration, VAR)' row became a row that still routes "
      "time-series regression (review F4: section 6, long_run_effect) and says what stationarity and cointegration do",
      not any(r.startswith("| Time series (levels") for r in rows)
      and "spurious regression, distributed lags, ADL and its long-run effect, HAC errors" in order
      and "`time-series.md` §3, §6, §7" in order and "`specification.long_run_effect`" in order
      and "`diagnostics.stationarity`, `cointegration`" in order and "once all series are I(1)" in order
      and "Engle-Granger and Johansen" in order and callable(getattr(specification, "long_run_effect", None)), order)
check("time-series.md section 3 states the same choice of function",
      "Which function (v2.6.10): `diagnostics.stationarity` first" in ts_md
      and "`ts_univariate.adf_bootstrap` when the sample is small enough that the asymptotic ADF p-value is not trusted" in ts_md and "`arch.unitroot.ZivotAndrews` when a break is suspected" in ts_md
      and "If the bootstrap p-value and the DF-GLS/KPSS verdict disagree, report both" in ts_md)

# ----------------------------------------------------------------------------------- 7. --offline in section 7 (3.5)
sec7 = skill_md[skill_md.index("## 7. Environment"):skill_md.index("## 8.")]       # review B-5: section 7 only
check("section 7: --offline only without network, never by default, never because of a sandbox or to save time; "
      "read the whole report (the one rule said once, review r2 A nit 2)",
      "(one request to api.github.com; `--offline` only without network, never by default)" in sec7
      and "Never add `--offline` because of a sandbox or to save time, and read the whole report (no `tail` or "
          "`Select-Object -Last`)." in sec7 and sec7.count("--offline") == 2)
check("the doctor's last line says what to do after --offline (read from the printed report, review B-4)",
      "run --doctor again without --offline" in last(o_off))
check("section 7: what left it is said elsewhere (the doctor lists the extras, the README names them; --install "
      "chooses pip itself); a missing extra still stops with the install line; within the v2.5.1 size",
      "any other extra only when the task needs it; a function missing one stops with the install line." in sec7
      and "break-system-packages" not in sec7
      and all(f"`{x}`" in (ROOT / "README.md").read_text(encoding="utf-8")
              for x in ("rd", "regimes", "panel", "web", "data-br", "data-intl"))
      and all(x in se.OPTIONAL for x in ("rd", "regimes", "panel", "web", "data-br", "data-intl"))
      and "--break-system-packages" in (SKILL / "scripts" / "setup_env.py").read_text(encoding="utf-8")
      and len(sec7) <= 2832, len(sec7))
check("routing: the 'One series' row lists diagnostics.stationarity in its function column too (review r2 A nit 6)",
      one.count("`diagnostics.stationarity`") == 2 and one.split(" | ")[-1].startswith("`diagnostics.stationarity`"))
cases = json.loads((ROOT / "evals" / "cases.json").read_text(encoding="utf-8"))
case = next((c for c in cases.get("cases", []) if c.get("id") == "iv-interaction-numbers"), {})
gen = ROOT / "evals" / "data" / "make_treino_regiao_v269.py"
check("blind eval iv-interaction-numbers: data and generator are in the public repository; the MDE must matches "
      "the rule (SE of the coefficient, not the two-group formula); strength graded by the AR set, not by a word",
      case.get("data", "").startswith("evals/data/treino_regiao_v269.csv") and gen.is_file()
      and (ROOT / "evals" / "data" / "treino_regiao_v269.csv").is_file()
      and "SE of the 2SLS coefficient of training hours (not the OLS SE)" in " ".join(case.get("must", []))
      and "both z values computed in the script" in " ".join(case.get("must", []))
      and "Hard-codes a critical value (1.96, 2.8 or 0.84)" in " ".join(case.get("must_not", []))
      and "any label for the strength is fine if the AR set is reported" in " ".join(case.get("must", []))
      and "{DATA_DIR}" in case.get("prompt", "")
      and "**Cases with data** (a `data` field" in (ROOT / "evals" / "README.md").read_text(encoding="utf-8")
      and "holds **only** the data file" in (ROOT / "evals" / "README.md").read_text(encoding="utf-8"))

passed = sum(results)
print(f"\n{passed}/{len(results)} checks passed")
sys.exit(0 if passed == len(results) else 1)
