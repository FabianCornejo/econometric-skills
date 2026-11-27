"""v2.8.1: SKILL.md names GMM, the likelihood ratio and the sequential tests (found only by grep in the D6 validation
of v2.8.0), the study card keys are written as they are, and reporting.study_card says which standard field a
non-standard name probably means (D6: identifying_assumption, robustness, tests_family).

- description: GMM and A/B peeking / SPRT, within 1024 characters, the v2.6.10 phrases kept;
- section 2 and section 6: the rows, every function they name exists, every section they cite exists;
- section 3: the card keys listed are exactly reporting.CARD_FIELDS;
- study_card: the suggestion on both sides (synonym, typo, case; a standard key gets no notice; a field already
  given or an unrelated name gets no suggestion; nothing is renamed);
- version, CHANGELOG.
"""
import json
import re
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "econometrics" / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _console  # noqa: E402,F401  (Windows-safe console output, see econometrics/scripts/_console.py)
import estimation  # noqa: E402
import reporting  # noqa: E402
import sequential  # noqa: E402
import skillmd_changes as SC  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SK = ROOT / "econometrics"
ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info)


def read(rel):
    return (ROOT / rel).read_text(encoding="utf-8").replace("\r\n", "\n")


skill = read("econometrics/SKILL.md")
lines = skill.split("\n")
end = lines.index("---", 1)
desc = yaml.safe_load("\n".join(lines[1:end]))["description"]

# ------------------------------------------------------------------------------------------------ 1. description
check("description: at most 1024 characters", 0 < len(desc) <= 1024, len(desc))
check("description: GMM right after the regression list, A/B peeking and SPRT right after power and sample size",
      "logit/probit), GMM, effect of X on Y" in desc and "power and sample size, A/B peeking and SPRT, any regression"
      in desc, desc)

# ------------------------------------------------------------------------------------- 2. sections 2 and 6 rows
check("section 2: the GMM row and the sequential row, once each, right after the custom-likelihood row",
      skill.count(SC.ROW_GMM) == 1 and skill.count(SC.ROW_SEQ) == 1
      and ("`wild_cluster_bootstrap` |\n" + SC.ROW_GMM + SC.ROW_SEQ) in skill)
sec2 = skill.split("## 2. Route the question", 1)[1].split("## 3.", 1)[0]
sec6 = skill.split("## 6. Scripts", 1)[1].split("## 7.", 1)[0]
check("the two rows are in section 2 and the sequential.py row in section 6",
      SC.ROW_GMM in sec2 and SC.ROW_SEQ in sec2 and SC.ROW_SEQ_SCRIPT in sec6 and SC.EST_GMM in sec6)


def named(row):
    """The functions of the last column: 'module.f' sets the module for the bare names that follow."""
    out, mod = [], None
    for tok in re.findall(r"`([\w.]+)`", row.rstrip(" |\n").rsplit("|", 1)[1]):
        if "." in tok:
            mod, tok = tok.split(".", 1)
        out.append((mod, tok))
    return out


mods = {"estimation": estimation, "sequential": sequential}
for row in (SC.ROW_GMM, SC.ROW_SEQ):
    fns = named(row)
    check(f"every function the row names exists and is public: {[m + '.' + f for m, f in fns]}",
          len(fns) >= 2 and all(m in mods and callable(getattr(mods[m], f, None)) and not f.startswith("_")
                                for m, f in fns), fns)
et = read("econometrics/references/estimation-theory.md")
sf = read("econometrics/references/statistics-foundations.md")
check("the sections the rows cite exist: estimation-theory §7 (gmm) and §8 (lr_test), statistics-foundations §14",
      re.search(r"^## 7\. .*`gmm`", et, re.M) and re.search(r"^## 8\. .*`lr_test`", et, re.M)
      and re.search(r"^## 14\. Sequential tests", sf, re.M))
check("§14 and §7-8 use the functions the rows name (no row points to a section about something else)",
      all(f in sf for f in ("sprt_design", "peeking_size", "n_for_error", "divergence"))
      and "gmm(" in et and "lr_test(" in et)
pub = sorted(n for n, v in vars(sequential).items() if callable(v) and not n.startswith("_")
             and getattr(v, "__module__", "") == "sequential")
check("every public function of sequential.py is in the routing row or described in the sequential.py row of §6",
      all(f in SC.ROW_SEQ for f in ("sprt_design", "sprt", "peeking_size", "n_for_error", "divergence"))
      and "Markov chains or VAR(1)" in SC.ROW_SEQ_SCRIPT and "importance sampling" in SC.ROW_SEQ_SCRIPT
      and pub == sorted(["sprt_design", "sprt", "peeking_size", "divergence", "divergence_rate", "n_for_error",
                         "importance_tail_prob"]), pub)

# --------------------------------------------------------------------------------------- 3. card keys in section 3
sec3 = skill.split("## 3. Consultant workflow", 1)[1].split("## 4.", 1)[0]
listed = re.findall(r"`(\w+)`", SC.CARD_KEYS)
check("section 3 lists the study card keys, and they are exactly reporting.CARD_FIELDS",
      SC.CARD_KEYS in sec3 and len(listed) == len(set(listed)) and set(listed) == set(reporting.CARD_FIELDS),
      sorted(set(listed) ^ set(reporting.CARD_FIELDS)))
check("section 4 is not touched by v2.8.1 (no documented v2.8.1 change falls in it)",
      all(now not in skill.split("## 4. Non-negotiable rules", 1)[1].split("## 5.", 1)[0] for now, _ in SC.V281))

# ------------------------------------------------------------------------------------ 4. study_card suggestion
REQ = dict(question="q", estimand="e", design="d")


def notice(**extra):
    with tempfile.TemporaryDirectory() as d:
        out = reporting.study_card(str(Path(d) / "card.json"), show=False, **REQ, **extra)
        on_file = json.loads((Path(d) / "card.json").read_text(encoding="utf-8"))["fields"]
    ws = [w for w in out["warnings"] if "non-standard" in w["message"]]
    return (ws[0] if ws else None), out, on_file


w, out, on_file = notice(identifying_assumption="a", robustness="r", tests_family="t")
check("D6 names: each gets the standard field it means, in the notice",
      w is not None and w["level"] == "notice"
      and "identifying_assumption (did you mean assumption?)" in w["message"]
      and "robustness (did you mean robustness_plan?)" in w["message"]
      and "tests_family (did you mean family_of_tests?)" in w["message"], w)
check("nothing is renamed: the card keeps the names given and the standard fields stay absent",
      {"identifying_assumption", "robustness", "tests_family"} <= set(on_file)
      and not {"assumption", "robustness_plan", "family_of_tests"} & set(on_file)
      and out["fields"] == on_file)
check("the action says a standard field left out stays empty and lists the standard fields",
      w is not None and "stays empty" in w["action"] and all(f in w["action"] for f in reporting.CARD_FIELDS), w)
w, *_ = notice(assumption="a", y="y", robustness_plan="r", family_of_tests=["t1"], horizon=12)
check("standard keys only: no notice at all", w is None, w)
w, *_ = notice(assumption="a", identifying_assumption="b")
check("a field the card already has is never suggested (assumption given, identifying_assumption kept as is)",
      w is not None and "identifying_assumption" in w["message"] and "did you mean" not in w["message"], w)
w, *_ = notice(horizn=3, familly_of_tests="t", x="d")
check("typos and case: horizn -> horizon, familly_of_tests -> family_of_tests, x -> X",
      w is not None and "horizn (did you mean horizon?)" in w["message"]
      and "familly_of_tests (did you mean family_of_tests?)" in w["message"] and "x (did you mean X?)" in w["message"],
      w)
w, *_ = notice(**{"Robustness Plan": "r", "notes": "n", "zz": 1})
check("spacing and capitals of a standard key are matched; unrelated names get no invented suggestion",
      w is not None and "Robustness Plan (did you mean robustness_plan?)" in w["message"]
      and re.search(r"\bnotes\b(?! \(did)", w["message"]) and re.search(r"\bzz\b(?! \(did)", w["message"]), w)
check("every synonym points to a standard field, and no synonym is itself a standard key",
      set(reporting.CARD_SYNONYMS.values()) <= set(reporting.CARD_FIELDS)
      and not set(reporting.CARD_SYNONYMS) & {reporting._card_key(f) for f in reporting.CARD_FIELDS})
doc = reporting.study_card.__doc__
check("study_card docstring: the suggestion is documented, and the Returns block still ends in the first 2500 "
      "characters (lesson of v2.7.3)", "did you mean assumption?" in doc
      and doc.index('"warnings"') < 2500, doc.index('"warnings"'))

# ------------------------------------------------------------------------------------------- 5. version, CHANGELOG
cl = read("CHANGELOG.md")
heads = re.findall(r"^## v(\d+(?:\.\d+)+) — ", cl, re.M)
check("CHANGELOG has a v2.8.1 entry, right before v2.8.0's", "2.8.1" in heads and "2.8.0" in heads
      and heads.index("2.8.1") + 1 == heads.index("2.8.0"), heads[:4])
e281 = cl.split("## v2.8.1 — ", 1)[1].split("\n## v", 1)[0] if "## v2.8.1 — " in cl else ""
check("the v2.8.1 entry names the SKILL.md changes, the card suggestion and the D6 finding",
      all(w_ in e281 for w_ in ("SKILL.md", "estimation.gmm", "lr_test", "sequential", "study_card", "did you mean",
                                "D6", "skillmd_changes.py")))
mk = json.loads(read(".claude-plugin/marketplace.json"))
vers = {p_.get("version") for p_ in mk.get("plugins", [])}
v_ok = tuple(int(x) for x in reporting.SKILL_VERSION.split(".")) >= (2, 8, 1)   # a version that grows: >=
check("SKILL_VERSION is 2.8.1 or later and equals the marketplace's version", v_ok and vers == {reporting.SKILL_VERSION},
      (reporting.SKILL_VERSION, vers))

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
