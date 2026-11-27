"""v2.8.1: SKILL.md names GMM, the likelihood ratio and the sequential tests (found only by grep in the D6 validation
of v2.8.0), the study card keys are written as they are, and reporting.study_card says which standard field a
non-standard name means, from the same key or an unambiguous synonym only (D6: identifying_assumption,
robustness, tests_family; reviews r1-r3).

- description: GMM and A/B peeking / SPRT, within 1024 characters, the v2.6.10 phrases kept;
- section 2 and section 6: the rows, every function they name exists, every section they cite exists;
- section 3: the card keys listed are exactly reporting.CARD_FIELDS;
- study_card: the suggestion on both sides (synonym and case; no guess from spelling; ambiguous names get none; a
  field already given gets none; nothing is renamed), collisions, and an action that can be followed without
  losing fields;
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
import systems  # noqa: E402
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


def cited(row):
    """The section numbers in the row's reference column, read from the row (§7–8 -> [7, 8]; review r1, S3-7)."""
    col = row.split("|")[2]
    out = []
    for a, b in re.findall(r"§(\d+)(?:[–-](\d+))?", col):
        out += list(range(int(a), int(b or a) + 1))
    return col, out


col_g, sec_g = cited(SC.ROW_GMM)
col_s, sec_s = cited(SC.ROW_SEQ)
heads_et = {int(n): t for n, t in re.findall(r"^## (\d+)\. (.*)$", et, re.M)}
heads_sf = {int(n): t for n, t in re.findall(r"^## (\d+)\. (.*)$", sf, re.M)}
check("the sections each row cites, read from the row, are the GMM, LR and sequential sections of the cited file",
      "estimation-theory.md" in col_g and sec_g == [7, 8] and "`gmm`" in heads_et.get(7, "")
      and "`lr_test`" in heads_et.get(8, "") and "statistics-foundations.md" in col_s and sec_s == [14]
      and heads_sf.get(14, "").startswith("Sequential tests"), (col_g, sec_g, col_s, sec_s))
check("key words of the rows that the code backs (review r1, S3-4/5/6/7): right-tail J, the 50:50 mixture for one "
      "restriction, the rank test, simple against simple, Pocock, the pointers to iv_report and dynamic_panel",
      "right-tail p-value" in SC.EST_GMM and "50:50 boundary mixture for one restriction" in SC.EST_GMM
      and "rank test of whether the moments can identify" in SC.ROW_GMM and "`diagnostics.iv_report`, row \"Panel data, endogeneity, instruments\"" in SC.ROW_GMM
      and "`systems.dynamic_panel`" in SC.ROW_GMM and callable(getattr(systems, "dynamic_panel", None))
      and "simple against a simple hypothesis" in SC.ROW_SEQ and "Pocock per-look level" in SC.ROW_SEQ_SCRIPT
      and "with boundary=True and df=1, the 50:50 mixture" in estimation.lr_test.__doc__
      and "Pocock" in sequential.peeking_size.__doc__ and "Wald's SPRT" in sequential.sprt.__doc__
      and "whether the parameters are identified" not in skill)
iv_rows = [ln for ln in sec2.split("\n") if ln.startswith("| Panel data, endogeneity, instruments |")]
check("the row the GMM row names exists in section 2 and routes to diagnostics.iv_report",
      len(iv_rows) == 1 and "`diagnostics.iv_report`" in iv_rows[0], iv_rows)
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
      and "tests_family (did you mean family_of_tests?)" in w["message"] and "all mean" not in w["message"], w)
check("nothing is renamed: the card keeps the names given and the standard fields stay absent",
      {"identifying_assumption", "robustness", "tests_family"} <= set(on_file)
      and not {"assumption", "robustness_plan", "family_of_tests"} & set(on_file)
      and out["fields"] == on_file)
check("the action says a standard field left out stays empty and lists the standard fields",
      w is not None and "a standard field left out stays empty" in w["action"]
      and all(f in w["action"] for f in reporting.CARD_FIELDS), w)
w, *_ = notice(assumption="a", y="y", robustness_plan="r", family_of_tests=["t1"], horizon=12)
check("standard keys only: no notice at all", w is None, w)
w, *_ = notice(assumption="a", identifying_assumption="b")
check("a field the card already has is never suggested (assumption given, identifying_assumption kept as is)",
      w is not None and "identifying_assumption" in w["message"] and "did you mean" not in w["message"], w)
w, *_ = notice(horizn=3, estimador="CS", x="d")
check("case is matched (x -> X); no guess from spelling: horizn and estimador get none (review r2, S2-3)",
      w is not None and "x (did you mean X?)" in w["message"] and "did you mean horizon" not in w["message"]
      and "did you mean estimand" not in w["message"], w)
w, *_ = notice(**{"Robustness Plan": "r", "notes": "n", "zz": 1})
check("spacing and capitals of a standard key are matched; unrelated names get no invented suggestion",
      w is not None and "Robustness Plan (did you mean robustness_plan?)" in w["message"]
      and re.search(r"\bnotes\b(?! \(did)", w["message"]) and re.search(r"\bzz\b(?! \(did)", w["message"]), w)
check("every synonym points to a standard field, and no synonym is itself a standard key",
      set(reporting.CARD_SYNONYMS.values()) <= set(reporting.CARD_FIELDS)
      and not set(reporting.CARD_SYNONYMS) & {reporting._card_key(f) for f in reporting.CARD_FIELDS})
EXPECT = {   # reviews r1, r2 and r3: only names that mean one field in every design
    "treat": "treatment", "treatment_variable": "treatment", "treatment_var": "treatment", "intervention": "treatment",
    "confounder": "W", "confounders": "W", "control_variables": "W",
    "outcome": "y", "outcomes": "y", "outcome_variable": "y", "outcome_var": "y", "dependent_variable": "y",
    "instruments": "Z", "instrument": "Z", "instrumental_variable": "Z", "instrumental_variables": "Z",
    "effect_modifier": "X", "effect_modifiers": "X",
    "identifying_assumption": "assumption", "identification_assumption": "assumption", "assumptions": "assumption",
    "key_assumption": "assumption", "identification_strategy": "design", "research_design": "design",
    "research_question": "question", "robustness": "robustness_plan", "robustness_check": "robustness_plan",
    "robustness_checks": "robustness_plan", "robustness_plans": "robustness_plan", "robustness_tests": "robustness_plan",
    "tests_family": "family_of_tests", "test_family": "family_of_tests", "family_of_test": "family_of_tests",
    "test_families": "family_of_tests", "multiple_tests": "family_of_tests",
    "significance": "alpha", "significance_level": "alpha", "sig_level": "alpha", "alpha_level": "alpha",
    "min_detectable_effect": "mde", "minimum_detectable_effect": "mde", "forecast_horizon": "horizon",
    "loss_function": "loss", "benchmark": "benchmarks", "threat": "threats", "threats_to_validity": "threats",
    "validity_threats": "threats", "threats_to_identification": "threats", "dataset": "data", "data_source": "data",
    "data_sources": "data", "datasource": "data", "Robustness Plan": "robustness_plan",
    "Family-of-Tests": "family_of_tests", "x": "X",
    # no suggestion: a name with another meaning in some design, a spelling that only looks close, or a start
    "control": None, "controls": None, "covariate": None, "covariates": None, "treated": None, "exposure": None,
    "source": None, "risks": None, "family": None, "level": None, "cluster_level": None, "identification": None,
    "tests": None, "sensitivity": None, "estimador": None, "estimado": None, "estimated": None, "treinamento": None,
    "treats": None, "treaties": None, "heats": None, "thres": None, "horizn": None, "threts": None, "bench": None,
    "alphas": None, "losses": None, "datas": None, "notes": None, "comments": None, "weights": None, "clusters": None,
    "design_notes": None, "data_notes": None, "wage": None, "zz": None,
}
got = {n: reporting._card_suggestion(n, {}) for n in EXPECT}
check("suggestions: every name gets the expected field or none (control(s), covariates, treated, family, exposure, "
      "estimador, horizn and the other ambiguous or look-alike names get none)", got == EXPECT,
      {n: (got[n], EXPECT[n]) for n in EXPECT if got[n] != EXPECT[n]})
check("every synonym in CARD_SYNONYMS is covered by the expectations above (a new or changed one needs a case)",
      all(any(reporting._card_key(n) == k for n in EXPECT) for k in reporting.CARD_SYNONYMS),
      [k for k in reporting.CARD_SYNONYMS if not any(reporting._card_key(n) == k for n in EXPECT)])
w, *_ = notice(assumptions="a", identifying_assumption="b", notes="n")
check("two names for the same field: the notice says so and says to merge them (reviews r1 S3-3, r2 S3-6)",
      w is not None and "assumptions and identifying_assumption all mean assumption: merge them into one assumption field" in w["message"]
      and "notes" in w["message"], w)
w, *_ = notice(notes="n")
check("no suggestion: the action asks for no amendment and says the fields are kept as given (review r3, S3-2)",
      w is not None and "did you mean" not in w["message"] and "non-standard fields are kept as given" in w["action"]
      and "amend" not in w["action"], w)
w, *_ = notice(identifying_assumption="a", notes="n")
act = w["action"] if w else ""
check("with a suggestion the action says the truth (reviews r2 S2-4, r3 S1-1, S2-1): the card is written; call again "
      "with ALL the fields and amend=True; a field left out is dropped; the reason is the agent's, not a ready text",
      act.startswith("the card is already written with these names") and "ALL the fields" in act
      and "amend=True" in act and "recorded as an amendment" in act and "dropped from the card" in act
      and "whether results had been seen" in act and "no results seen" not in act, act)
w, *_ = notice(assumptions="a", identifying_assumption="b", key_assumption="c")
check("three names for one field: listed with 'and' and told to merge (review r3, S3-5)",
      w is not None and "assumptions, identifying_assumption and key_assumption all mean assumption: merge them into "
      "one assumption field" in w["message"], w)
with tempfile.TemporaryDirectory() as d:   # review r3, S1-1: follow the action and lose nothing
    path = str(Path(d) / "card.json")
    first = reporting.study_card(path, show=False, **REQ, identifying_assumption="a", W=["x1"], notes="n")
    n0 = next(w_ for w_ in first["warnings"] if "non-standard" in w_["message"])
    sugg = dict(re.findall(r"(\w+) \(did you mean (\w+)\?\)", n0["message"]))
    given = {**REQ, "identifying_assumption": "a", "W": ["x1"], "notes": "n"}
    renamed = {sugg.get(k, k): v for k, v in given.items()}
    second = reporting.study_card(path, show=False, **renamed, amend=True,
                                  reason="names changed to the standard keys; no results had been seen")
    on_file = json.loads(Path(path).read_text(encoding="utf-8"))
check("following the action (all fields, the suggested key in place of the name, amend=True and a reason) keeps "
      "every field and records one amendment", sugg == {"identifying_assumption": "assumption"}
      and on_file["fields"] == {**REQ, "assumption": "a", "W": ["x1"], "notes": "n"}
      and len(on_file.get("amendments", [])) == 1, (sugg, on_file.get("fields"), len(on_file.get("amendments", []))))
with tempfile.TemporaryDirectory() as d:
    a = reporting.study_card(str(Path(d) / "a.json"), show=False, **REQ, identifying_assumption="x", treat="t")
    raw = (Path(d) / "a.json").read_text(encoding="utf-8")
    card = json.loads(raw)
check("the card on file holds only the fields given (no notice text, no renamed key) and its hash is the hash of "
      "its content (the notice is not stored)", "did you mean" not in raw and set(card["fields"]) == {*REQ, "identifying_assumption",
      "treat"} and card["sha256"] == reporting._card_hash(card) and a["fields"] == card["fields"])
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
