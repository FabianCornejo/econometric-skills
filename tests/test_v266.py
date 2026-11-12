"""Tests for v2.6.6: SKILL.md names dml_bounds and ovb_bounds; the boundary goes with its number in the summary;
section 7 per analysis folder, --offline only without network, one question with every item; the same two points in
the text the doctor prints.

Origin (development repository, claude/plano-v2.6.6.md):
  - v2.6.2 / v2.6 reviews: SKILL.md never named causal.dml_bounds, and its sensitivity rule (section 4) and the M&A
    evaluation case named only specification.sensitivity for OLS, not ovb_bounds;
  - v2.6.5 validation on notebook 1: the memo opened with "the result only holds if no unmeasured factor is as
    strong as the clinic's size" (and the chat reply with "weaker than size") while k_not_significant was 1.2x and
    the 1x line still excluded zero; a later sentence said that 2x made the effect "compatible with zero" with any
    method, while the OLS 2x line's CI excludes zero on the other side; the agent ran `--doctor --offline` with
    network and did not ask the one question about the only To do item (a clean folder for a crowded listing);
  - v2.5.2 NIT: section 7 said "once per machine" although the listing and project installs are per folder.

What is checked (expected texts written by hand, not read from the code):
  1. section 2: the robustness line names dml_bounds, and without it is the v2.6.5 line, word for word (one name
     added, the T6 rule of the development repository);
  2. section 4: the three v2.6.6 insertions (ovb_bounds, dml_bounds, the boundary sentence), each exactly once, the
     sentence naming k_not_significant, the opening summary and the chat reply, and the words it forbids;
  3. section 7: "Once per analysis folder" (and "Once per machine" nowhere in SKILL.md), "--offline only without
     network, never by default", non-blocking items "go in that question", and the size cap of tests/test_v252.py;
     the agent's step 2 says "in this analysis folder";
  4. interpretation-rules.md: a last section "Rules added in v2.6.6" with the rule and its origin;
  5. evals/cases.json: the new "find the error" case (pt-BR, must and must_not), unique ids, the M&A case's
     sensitivity criterion names ovb_bounds and dml_bounds; the output lines quoted in the new case are what
     ovb_bounds and dml_bounds (linear learner) print on the validation's design, recomputed here with the skill's
     functions (the data generator below is copied by script from the validation script, not retyped), and the
     facts the case grades are true of those numbers: the 1x line keeps the sign, the 2x OLS line flips it, the
     2x linear DML line includes zero, k_not_significant for size rounds to 1.2. The forest DML line (slow) is
     quoted from the same run and only checked for its rho;
  6. the doctor's printed text: with --offline status the line saying it is only for a machine without network
     (and not without it); the To do header asks once, at the end of the reply, with optional items included
     (and no header when there is nothing to do); the --json status stays "not checked (--offline)".

Run from the repository root:  python3 tests/test_v266.py
Needs: numpy, pandas, statsmodels, scikit-learn.
"""
import contextlib
import io
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
SKILL = ROOT / "econometrics"
sys.path.insert(0, str(SKILL / "scripts"))
import _console  # noqa: E402,F401  (Windows-safe console output, see econometrics/scripts/_console.py)
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression

import causal
import setup_env as se
import specification

ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


skill_md = (SKILL / "SKILL.md").read_text(encoding="utf-8")
sec2 = skill_md[skill_md.index("## 2. Route the question"):skill_md.index("## 3.")]
sec4 = skill_md[skill_md.index("## 4. Non-negotiable rules"):skill_md.index("## 5.")]
sec7 = skill_md[skill_md.index("## 7. Environment"):skill_md.index("## 8.")]

# ------------------------------------------------------------------------------------------------ 1. section 2
V265_LINE = ("| How robust is a causal estimate to a hidden confounder | `causal-inference.md` §4.1, `ols-regression.md` "
             "§14 | `causal.dml_sensitivity`, `dml_benchmark`, `specification.sensitivity` |")
line = [ln for ln in sec2.splitlines() if ln.startswith("| How robust is a causal estimate")]
check("section 2: one robustness line", len(line) == 1, line)
line = line[0] if line else ""
check("section 2: the robustness line names dml_bounds", "`dml_bounds`" in line, line)
check("section 2: without dml_bounds it is the v2.6.5 line word for word (one name added)",
      line.replace(" `dml_bounds`,", "", 1) == V265_LINE, line)
check("section 2: ovb_bounds is not added to the routing table (at most one name per line)", "ovb_bounds" not in sec2)

# ------------------------------------------------------------------------------------------------ 2. section 4
rule = [ln for ln in sec4.splitlines() if ln.startswith("- **Run more than one model**")]
check("section 4: one 'Run more than one model' rule", len(rule) == 1)
rule = rule[0] if rule else ""
check("section 4: OLS sensitivity names sensitivity and ovb_bounds",
      "(`specification.sensitivity` and `ovb_bounds`;" in rule, rule[-500:])
check("section 4: DML sensitivity names dml_sensitivity, dml_benchmark and dml_bounds",
      "`causal.dml_sensitivity` + `dml_benchmark` and `dml_bounds`)" in rule, rule[-500:])
check("section 4: each of the three v2.6.6 pieces appears once in the section",
      sec4.count(" and `ovb_bounds`") == 1 and sec4.count(" and `dml_bounds`") == 1
      and sec4.count("Where significance is lost is a number") == 1)
sent = rule[rule.find("Where significance is lost"):]
check("section 4: the boundary sentence names k_not_significant with a number, the summary and the chat reply",
      "`k_not_significant`" in sent and "1.2x as strong as size" in sent and "opening summary" in sent
      and "chat reply" in sent and "every sentence" in sent, sent)
check("section 4: it forbids 'as strong as size' and 'weaker than size' and points to the rules",
      'never "as strong as size"' in sent and '"weaker than size"' in sent and "interpretation-rules.md" in sent
      and "v2.6.6" in sent, sent)

# ------------------------------------------------------------------------------------------------ 3. section 7
check("section 7: once per analysis folder", "**Once per analysis folder, before the first script**" in sec7)
check("SKILL.md: 'Once per machine' nowhere", "per machine" not in skill_md.lower())
check("section 7: --offline only without network, never by default",
      "`--offline` only without network, never by default" in sec7, sec7[:900])
check("section 7: non-blocking items never hold up the analysis but go in that question",
      "never hold up the analysis but go in that question." in sec7)
check("section 7: still within the v2.5.1 size (tests/test_v252.py caps it at 2,859)", len(sec7) <= 2859, len(sec7))
agent = (SKILL / "agents" / "econometrician.md").read_text(encoding="utf-8")
check("agent: --check when the doctor already ran in this analysis folder, not 'on this machine'",
      "already ran in this analysis folder" in agent and "on this machine" not in agent)

# ------------------------------------------------------------------------------------------------ 4. rules
rules = (SKILL / "references" / "interpretation-rules.md").read_text(encoding="utf-8")
heads = re.findall(r"^## .*$", rules, re.M)
check("interpretation rules: 'Rules added in v2.6.6' is the last section", heads[-1:] == ["## Rules added in v2.6.6"],
      heads[-3:])
r266 = rules[rules.index("## Rules added in v2.6.6"):] if "## Rules added in v2.6.6" in rules else ""
check("interpretation rules v2.6.6: the summary, the slide title and the chat reply carry the number",
      all(k in r266 for k in ("opening line", "executive summary", "slide title", "reply in the chat", "1.2x",
                              "worst case", "classical SE")), r266[:400])
check("interpretation rules v2.6.6: forbids the 1x paraphrases and ties 1x to the 1x line's ci_verdict",
      all(k in r266 for k in ('"as strong as size"', '"weaker than size"', "ci_verdict", "v2.6.5 validation")))

# ------------------------------------------------------------------------------------------------ 5. cases.json
cases = json.loads((ROOT / "evals" / "cases.json").read_text(encoding="utf-8"))["cases"]
check("ids of the evaluation cases are unique", len({c_["id"] for c_ in cases}) == len(cases))
byid = {c_["id"]: c_ for c_ in cases}
new = byid.get("review-memo-summary-boundary", {})
check("new case: present, pt-BR, at least 3 must and 3 must_not",
      new.get("language") == "pt-BR" and len(new.get("must", [])) >= 3 and len(new.get("must_not", [])) >= 3)
must, must_not = " ".join(new.get("must", [])), " ".join(new.get("must_not", []))
check("new case: grades the 1x paraphrase in the summary against k_not_significant 1.2x",
      "1.2x" in must and "k_not_significant" in must and "ci_verdict" in must and "Rounds the boundary to 1x" in must_not)
check("new case: grades the 2x OLS line as a sign flip, not 'compatible with zero' (triage item 28)",
      "[1.351, 3.624]" in must and "sign would flip" in must and "compatible with zero" in must_not)
ma = " ".join(byid.get("dag-controls-advisor-premium", {}).get("must", []))
check("M&A case: the sensitivity criterion names ovb_bounds and dml_bounds and the number of the boundary",
      "specification.ovb_bounds" in ma and "dml_bounds" in ma and "k_not_significant" in ma)


# The validation's design (copied by script from claude/scripts/validacao_v265_triagem_sensibilidade.py of the
# development repository; the same run produced the lines quoted in the new case)
SEED = 20261111
ALPHA = 0.05
N = 400
EF_ESPERA = np.array([0, 6, -4, 10, 3, -7.0])        # efeito da região sobre a espera
EF_ADESAO = np.array([-1, 0.8, -0.3, 1.2, 0, -0.6])  # efeito da região sobre a adesão (logit)


def simular(seed=SEED, n=N):
    rng = np.random.default_rng(seed)
    regiao = rng.integers(0, 6, n)                                   # 6 regiões, equiprováveis
    log_pac = rng.normal(8, 0.6, n)
    med = np.maximum(rng.normal(1.5, 0.4, n) + 0.3 * (log_pac - 8), 0.3)
    lin = -0.4 + 1.5 * (log_pac - 8) + 1.2 * (med - 1.5) + EF_ADESAO[regiao]
    triagem = rng.binomial(1, 1 / (1 + np.exp(-lin)))
    espera = (45 + 9 * (log_pac - 8) - 10 * (med - 1.5) + EF_ESPERA[regiao]
              - 8 * triagem + rng.normal(0, 8, n))
    df = pd.DataFrame({"espera": espera, "triagem": triagem, "log_pac": log_pac, "med": med,
                       "regiao": pd.Categorical(regiao + 1)})
    dummies = pd.get_dummies(df["regiao"], prefix="reg", drop_first=True).astype(float)
    return pd.concat([df, dummies], axis=1)


def shown(fn_, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = fn_(*a, **k)
    return r, buf.getvalue()


prompt = new.get("prompt", "")
quoted = re.findall(r'"([^"]+)"', prompt)
df = simular()
ols = smf.ols("espera ~ triagem + log_pac + med + C(regiao)", df).fit(cov_type="HC3")
import warnings  # noqa: E402
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    bnd, out = shown(specification.ovb_bounds, ols, "triagem", ["C(regiao)", "log_pac"], kd=[1, 2, 3], alpha=0.05,
                     show=True)
lines = [ln.strip() for ln in out.splitlines()]
want = [ln for ln in lines if ln.startswith("1x log_pac:") or ln.startswith("2x log_pac:")
        or ln.startswith("A confounder 1.2x as strong as log_pac")]
check("new case: ovb_bounds prints the three quoted lines (1x, 2x size and the 1.2x boundary), word for word",
      len(want) == 3 and all(w in quoted for w in want), want)
b = bnd["bounds"].set_index("bound_label")
check("ovb_bounds on the design: 1x size keeps the sign (the summary's '1x' is wrong)",
      b.loc["1x log_pac", "ci_verdict"] == "same sign")
check("ovb_bounds on the design: 2x size flips the sign (not 'compatible with zero')",
      b.loc["2x log_pac", "ci_verdict"] == "opposite sign" and b.loc["2x log_pac", "adjusted_lower_ci"] > 0)
kns = bnd["benchmark_r2"].loc["log_pac", "k_not_significant"]
check("ovb_bounds on the design: k_not_significant for size rounds to 1.2 (between 1.15 and 1.25)",
      1.15 < kns < 1.25 and f"{round(kns, 2):g}" == "1.2", kns)
X = ["log_pac", "med"] + [c for c in df.columns if c.startswith("reg_")]
kw = dict(n_folds=5, n_rep=5, seed=SEED, learner_y=LinearRegression(), learner_d=LinearRegression())
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    res = causal.dml_plr(df, "espera", "triagem", X, **kw)
    bench = causal.dml_benchmark(df, "espera", "triagem", X, ["log_pac"], res=res, **kw)
    dbd, dout = shown(causal.dml_bounds, res, bench, kd=[1, 2, 3], rho="benchmark", alpha=0.05, show=True)
dl = [ln.strip() for ln in dout.splitlines() if ln.strip().startswith("2x log_pac:")]
check("new case: the linear DML 2x line is what dml_bounds prints, word for word", len(dl) == 1 and dl[0] in quoted, dl)
row = dbd["bounds"].set_index("bound_label").loc["2x log_pac"]
check("dml_bounds (linear) on the design: 2x size includes zero", row["ci_verdict"] == "includes zero")
fl = [q for q in quoted if q.startswith("2x log_pac:") and "rho 0.967914" in q]
check("new case: the forest DML 2x line is quoted with its observed rho 0.967914 and includes zero",
      len(fl) == 1 and "which include zero" in fl[0], fl)

# ------------------------------------------------------------------------------------------------ 6. doctor text
def report(**over):
    rep = {"python": {"exe": "C:\\Python313\\python.exe", "version": "3.13.17", "supported": True,
                      "run_with": "py -3.13"},
           "core_missing": [], "interpreters": [], "extras": {},
           "skill": {"dir": "p", "version": "2.6.6"}, "latest": {"version": "2.6.6", "status": "ok"},
           "copies": [{"kind": "plugin", "version": "2.6.6", "path": "p"}],
           "autoupdate": {"registered": True, "on": True, "configured": True, "source": "settings.json",
                          "blocked_by": []},
           "listing": {"checked": True, "folder": "f", "skills": 12, "enabled_plugins": 2, "crowded": False},
           "actions": []}
    rep.update(over)
    return rep


OFF = "--offline is only for a machine without network; with network, run --doctor without it"
_, o = shown(se.print_doctor, report(latest={"version": None, "status": "not checked (--offline)"}))
check("doctor text: with --offline, the line saying it is only for a machine without network",
      OFF in o and "Latest release: not checked (--offline)" in o, o)
_, o2 = shown(se.print_doctor, report())
_, o3 = shown(se.print_doctor, report(latest={"version": None, "status": "not checked (HTTP Error 403)"}))
check("doctor text: no such line when the release was checked, nor when the check failed for another reason",
      OFF not in o2 and OFF not in o3 and "Latest release: 2.6.6" in o2, o3)
acts = ["crowded skill listing (1693 skills in 54 enabled plugins): offer a clean analysis folder"]
_, o4 = shown(se.print_doctor, report(actions=acts))
head = [ln for ln in o4.splitlines() if ln.startswith("To do")]
check("doctor text: the To do header asks once, in one question at the end of the reply, optional items included, "
      "nothing changed before the answer",
      len(head) == 1 and "ask the user once" in head[0] and "in one question at the end of your reply" in head[0]
      and "optional ones included" in head[0] and "install or change nothing before the answer" in head[0]
      and "\n  1. crowded skill listing" in o4, head)
check("doctor text: nothing to do -> no To do header", "To do" not in o2 and "Nothing to do" in o2)
rep_off = se.doctor(offline=True, cwd=str(ROOT))
check("doctor --offline: the JSON status is unchanged ('not checked (--offline)')",
      rep_off["latest"]["status"] == "not checked (--offline)")

passed = sum(ok)
print(f"\n{passed}/{len(ok)} checks passed")
sys.exit(0 if passed == len(ok) else 1)
