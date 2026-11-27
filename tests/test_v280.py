"""v2.8.0: regressions and documentation checks (rule T4: the tests of estimation.gmm and lr_test are in
test_estimation.py, of sequential.py in test_sequential.py, the Hansen-Singleton replication in
test_quantecon_replication.py).

- SKILL.md is byte-for-byte the one of v2.7.5 (decision of 2026-11-24: the SKILL.md lines and the routing round go to
  the next release), so this release needs no routing round;
- the graders of the two new routing cases (gmm-euler, review-optional-stopping), measured on the corpora of
  tests/corpora_v280.py in Python and JavaScript, with the prompt echoed back refused;
- the rules and the reference sections of v2.8.0, and the claims in them that are numbers (computed here again);
- version, CHANGELOG, README and VALIDATION.md.
"""
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "econometrics" / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _console  # noqa: E402,F401  (Windows-safe console output, see econometrics/scripts/_console.py)
import corpora_v280 as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SK = ROOT / "econometrics"
ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info)


def read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


# ------------------------------------------------------------------ SKILL.md unchanged (no routing round needed)
SKILL_V275 = "35b7c6e446f8cc409bbeb979f2964972224c98b6861d75ed9e586d23ea88d86d"   # git show bc17002:econometrics/SKILL.md
raw = (SK / "SKILL.md").read_bytes().replace(b"\r\n", b"\n")
check("SKILL.md is the v2.7.5 file (SHA-256 with LF line ends): the routing round moves to the next release",
      hashlib.sha256(raw).hexdigest() == SKILL_V275, hashlib.sha256(raw).hexdigest())

# ------------------------------------------------------------------ the graders of the two new cases
CASES = {"gmm-euler": (C.GMM, {"routing", "gmm"}), "review-optional-stopping": (C.PEEK, {"routing", "reviewer"})}
ref_sf = (SK / "evals" / "routing" / "review-null-result" / "graders" / "skill-fired.md").read_text(encoding="utf-8")
node = shutil.which("node")
for name, (corpus, tags) in CASES.items():
    folder = SK / "evals" / "routing" / name
    text = (folder / "prompt.md").read_text(encoding="utf-8")
    meta = yaml.safe_load(text.split("---")[1])
    request = text.split("---", 2)[2]
    g = yaml.safe_load((folder / "graders" / "answer.md").read_text(encoding="utf-8").split("---")[1])
    check(f"{name}: valid YAML, routing tags, read-only tools, regex grader on the last message; the skill-fired grader "
          "of the other cases", tags <= set(meta["tags"]) and meta["max_turns"] == 8
          and meta["allowed_tools"] == ["Read", "Glob", "Grep", "Skill"] and g["type"] == "regex"
          and g["target"] == "last_message" and g.get("flags") == "i"
          and (folder / "graders" / "skill-fired.md").read_text(encoding="utf-8") == ref_sf, (meta, g.get("type")))
    pat = re.compile(g["pattern"], re.I)
    good, bad = corpus["good"], corpus["bad"]
    check(f"{name}: corpus of at least 40 right and 40 wrong answers, 10 of each held out (written after the grader)",
          len(good) >= 40 and len(bad) >= 40, (len(good), len(bad)))
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
for name, key in (("gmm-euler", "GMM"), ("review-optional-stopping", "PEEK")):
    g = yaml.safe_load((SK / "evals" / "routing" / name / "graders" / "answer.md").read_text(encoding="utf-8").split("---")[1])
    still = sum(bool(re.search(g["pattern"], s, re.I)) for s in C.KNOWN_FALSE_PASSES[key])
    print(f"INFO {name}: {still} of {len(C.KNOWN_FALSE_PASSES[key])} known false passes (negations) still accepted")
n_cases = len([d for d in (SK / "evals" / "routing").iterdir() if d.is_dir()])
readme = read("README.md")
check("at least 30 routing cases on disk; the README names the two new ones", n_cases >= 30
      and "an Euler equation by GMM" in readme and "an A/B test stopped at the first p < 0.05" in readme, n_cases)

# ------------------------------------------------------------------ rules and reference sections
ir = read("econometrics/references/interpretation-rules.md")
sec = ir.split("## Rules added in v2.8.0", 1)[1].split("\n## ", 1)[0] if "## Rules added in v2.8.0" in ir else ""
check("interpretation-rules.md: a v2.8.0 section with the four rules (peeking, GMM's J, LR on one sample, divergences)",
      sec and all(w in sec for w in ("before its end", "8% with 2 looks", "14% with 5", "19% with 10", "Pocock",
                                     "0.0158", "sequential.peeking_size", "sequential.sprt", "continue",
                                     "right tail", "m − k", "zero by construction", "Andrews", "Cragg-Donald",
                                     "not evidence of strong identification", "no cut of 10", "239 months",
                                     "same observations", "lr_test", "cannot see whether", "trinity_test",
                                     "KL(p‖q) ≠ KL(q‖p)", "nats")), sec[:200])
# v2.8.0 review r1: the rank statistic is not a first-stage F; no document may say so again
docs80 = {p_: read(p_) for p_ in ("econometrics/references/interpretation-rules.md", "econometrics/references/"
                                  "estimation-theory.md", "econometrics/references/review-checklist.md", "README.md",
                                  "VALIDATION.md", "CHANGELOG.md", "econometrics/scripts/estimation.py")}
stale = [(p_, w) for p_, t_ in docs80.items() for w in ("ident_F", "per degree of freedom is below 10",
                                                         "is the first-stage F of a linear IV", "within 4%",
                                                         "weak-identification cut", "2p − 1", "refuses fits on different samples")
         if w in (t_.split("## v2.7.5", 1)[0] if p_ == "CHANGELOG.md" else t_)]
check("no document repeats the review-r1 false claims (ident_F, 'first-stage F' reading, the 10 cut, 2p − 1, "
      "'refuses fits on different samples')", not stale, stale)
rc = read("econometrics/references/review-checklist.md")
check("review-checklist.md: the two v2.8.0 questions (peeking; GMM's J, identification and bounds)",
      rc.count("(v2.8.0)") == 2 and "looked at before its planned end" in rc and "right-tail p-value and degrees of "
      "freedom m − k" in rc)
et = read("econometrics/references/estimation-theory.md")
check("estimation-theory.md: section 7 (GMM) and 8 (LR between nested fits) with the functions and the measured numbers",
      "## 7. Generalized method of moments (`gmm`, v2.8.0)" in et and "## 8. Likelihood ratio between nested fits" in et
      and all(w in et for w in ("est.gmm(", "j_p", "j_cdf", "rank_p", "Cragg-Donald", "0.945 for γ and 0.938 for β",
                                "0.035", "Rejecting does not mean strong identification", "about 40 with n = 100",
                                "interval of γ contains 0", "est.lr_test(", "number of restrictions", "Self and Liang")))
sf = read("econometrics/references/statistics-foundations.md")
check("statistics-foundations.md: section 14 in the contents and in the text, with the SPRT, peeking and divergences",
      "14. Sequential tests" in sf.split("## 1.")[0] and "## 14. Sequential tests" in sf
      and all(w in sf for w in ("sq.sprt_design", "sq.sprt(", "sq.peeking_size(10)", "sq.divergence(", "35",
                                "about 16 under H0 and 19 under H1", "overshoot")))
check("software-equivalents.md: Stata gmm and R gmm::gmm point to estimation.gmm",
      "estimation.gmm(moments" in read("econometrics/references/software-equivalents.md"))

# the numbers the documents state, computed again
import sequential as sq  # noqa: E402
from scipy import stats  # noqa: E402

nf = sq.n_for_error(stats.norm(0, 1), stats.norm(0.5, 1), show=False)
check("statistics-foundations.md's example: fixed n 35, SPRT about 16 and 19 (rounded)",
      nf["n_normal"] == 35 and round(nf["sprt_en0"]) == 16 and round(nf["sprt_en1"]) == 19, nf)
pk = {K: sq.peeking_size(K, show=False) for K in (2, 5, 10)}
check("interpretation-rules.md's peeking numbers: 8%, 14% and 19% (rounded to the percent), Pocock 0.0158 for 5 looks",
      [round(100 * pk[K]["size"]) for K in (2, 5, 10)] == [8, 14, 19] and round(pk[5]["per_look_alpha"], 4) == 0.0158,
      {K: pk[K]["size"] for K in pk})

# ------------------------------------------------------------------ VALIDATION.md, CHANGELOG, version
val = read("VALIDATION.md")
new = ("estimation.gmm", "estimation.lr_test", "sequential.sprt_design", "sequential.sprt", "sequential.peeking_size",
       "sequential.divergence", "sequential.divergence_rate", "sequential.n_for_error", "sequential.importance_tail_prob")
check("VALIDATION.md: a row for each of the nine new functions, at level A", all(
    re.search(rf"^\| `{re.escape(f)}` \| A \|", val, re.M) for f in new))
cl = read("CHANGELOG.md")
heads = re.findall(r"^## v(\d+(?:\.\d+)+) — ", cl, re.M)
check("CHANGELOG has a v2.8.0 entry, right before v2.7.5's", "2.8.0" in heads and "2.7.5" in heads
      and heads.index("2.8.0") + 1 == heads.index("2.7.5"), heads[:4])
e280 = cl.split("## v2.8.0 — ", 1)[1].split("\n## v", 1)[0] if "## v2.8.0 — " in cl else ""
check("the v2.8.0 entry names the functions, the replication, the rules, the cases and that SKILL.md is unchanged",
      all(w in e280 for w in ("estimation.gmm", "estimation.lr_test", "sequential.py", "peeking_size", "Cragg-Donald",
                              "NonlinearIVGMM", "hansen_singleton_1982_data.csv", "interpretation-rules.md",
                              "gmm-euler", "review-optional-stopping", "`SKILL.md` is unchanged"))
      and "CHECKS_PLACEHOLDER" not in e280)
import reporting  # noqa: E402
mk = json.loads(read(".claude-plugin/marketplace.json"))
vers = {p_.get("version") for p_ in mk.get("plugins", [])}
v_ok = tuple(int(x) for x in reporting.SKILL_VERSION.split(".")) >= (2, 8, 0)   # a version that grows: >=
check("SKILL_VERSION is 2.8.0 or later and equals the marketplace's version", v_ok and vers == {reporting.SKILL_VERSION},
      (reporting.SKILL_VERSION, vers))

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
