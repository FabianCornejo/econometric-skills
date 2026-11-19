"""Tests of reporting.py functions added or changed from v2.7.0 on (T4: tests by module): study_card, the study card
and the missing-file notice in run_manifest, and the finding helpers of the v2.7.0 output contract.

The card's hash is recomputed here from the bytes read from disk (json.loads of the file, the content minus the
`sha256` key, serialized with sorted keys and compact separators, then hashlib), not with the module's helper.

Run from the repository root:  python3 tests/test_reporting.py
"""
import contextlib
import hashlib
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "econometrics" / "scripts"))
import _console  # noqa: E402,F401
import reporting as rp  # noqa: E402

results = []


def check(name, cond, info=""):
    results.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info)


def raises(f, *a, **k):
    try:
        f(*a, **k)
    except ValueError as e:
        return str(e)
    return False


def quiet(f, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = f(*a, **k)
    return out, buf.getvalue()


def hash_from_disk(path):
    card = json.loads(Path(path).read_bytes().decode("utf-8"))
    body = {k: v for k, v in card.items() if k != "sha256"}
    return card["sha256"], hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False,
                                                     separators=(",", ":")).encode("utf-8")).hexdigest()


CARD = dict(question="Does the training program raise wages?", estimand="ATT on log hourly wage",
            design="selection on observables (AIPW)", W=["age", "educ", "prior wage"], alpha=0.05,
            family_of_tests=["ATT", "ATT by sex"], robustness_plan=["OLS with the same W", "trimmed sample"])
tmp = Path(tempfile.mkdtemp())

# ============================================================ 1. writing and the hash
p = tmp / "card.json"
r = rp.study_card(path=p, show=False, **CARD)
stored, recomputed = hash_from_disk(p)
check("study_card: file written, stored hash equal to the hash recomputed from disk", r["action"] == "written"
      and stored == recomputed == r["sha256"])
on_disk = json.loads(p.read_text(encoding="utf-8"))
check("study_card: fields, UTC time, skill version and an empty amendment list in the file",
      on_disk["fields"] == CARD and on_disk["created_utc"].endswith("+00:00") and on_disk["skill_version"] == rp.SKILL_VERSION
      and on_disk["amendments"] == [])
p_pt = tmp / "cartao.json"
rp.study_card(path=p_pt, show=False, question="Efeito do programa sobre o salário?", estimand="ATT", design="AIPW")
check("study_card: non-ASCII text stored as it is and hashed consistently",
      "salário" in p_pt.read_text(encoding="utf-8") and len(set(hash_from_disk(p_pt))) == 1)

# ============================================================ 2. re-running and changing
before = p.read_bytes()
r = rp.study_card(path=p, show=False, **CARD)
check("study_card: re-running with the same fields leaves the file byte for byte unchanged",
      r["action"] == "unchanged" and p.read_bytes() == before)
msg = raises(rp.study_card, path=p, show=False, **{**CARD, "estimand": "ATE on log hourly wage"})
check("study_card: different fields are refused without amend, and the message names the changed field",
      bool(msg) and "estimand" in msg and "amend=True" in msg and p.read_bytes() == before, msg)
check("study_card: amend without a reason is refused",
      bool(raises(rp.study_card, path=p, amend=True, show=False, **{**CARD, "estimand": "ATE"})))
r = rp.study_card(path=p, amend=True, reason="ATE asked by the client before any estimate", show=False,
                  **{**CARD, "estimand": "ATE on log hourly wage"})
on_disk = json.loads(p.read_text(encoding="utf-8"))
check("study_card: amendment keeps the previous fields, their hash and the reason; creation time unchanged",
      r["action"] == "amended" and len(on_disk["amendments"]) == 1
      and on_disk["amendments"][0]["replaced_fields"]["estimand"] == "ATT on log hourly wage"
      and on_disk["amendments"][0]["replaced_sha256"] == stored
      and on_disk["amendments"][0]["reason"].startswith("ATE asked")
      and on_disk["created_utc"] == json.loads(before)["created_utc"])
s2, rc2 = hash_from_disk(p)
check("study_card: the amended file's hash covers the amendment history", s2 == rc2 != stored)

# ============================================================ 3. required fields, values, extra fields
check("study_card: question, estimand and design are required",
      bool(raises(rp.study_card, path=tmp / "x.json", show=False, question="q", estimand="e")))
check("study_card: an empty required field is refused",
      bool(raises(rp.study_card, path=tmp / "x.json", show=False, question="q", estimand=" ", design="d")))
check("study_card: values that are not JSON are refused",
      bool(raises(rp.study_card, path=tmp / "x.json", show=False, question="q", estimand="e", design="d",
                  data=pd.DataFrame({"a": [1]}))))
r = rp.study_card(path=tmp / "extra.json", show=False, question="q", estimand="e", design="d", contorls=["x"])
check("study_card: a misspelt field is kept and reported as a notice with the standard names",
      [w["level"] for w in r["warnings"]] == ["notice"] and "contorls" in r["warnings"][0]["message"])
r = rp.study_card(path=tmp / "fc.json", show=False, question="Next 12 months of sales", estimand="forecast at h = 1..12",
                  design="ARIMA vs seasonal naive, walk-forward", horizon=12, loss="MAE", benchmarks=["seasonal naive"],
                  window="expanding from 2015")
check("study_card: a short forecast card (horizon, loss, benchmarks, window) has no finding", r["warnings"] == [])

# ============================================================ 4. a card edited by hand
q = tmp / "hand.json"
rp.study_card(path=q, show=False, **CARD)
card = json.loads(q.read_text(encoding="utf-8"))
card["fields"]["alpha"] = 0.10                      # changed by hand, hash left as it was
q.write_text(json.dumps(card), encoding="utf-8")
r = rp.study_card(path=q, show=False, **{**CARD, "alpha": 0.10})
check("study_card: a card changed by hand is reported (warning) and not rewritten",
      [w["level"] for w in r["warnings"]] == ["warning"] and r["action"] == "unchanged")

# ============================================================ 5. run_manifest
df = pd.DataFrame({"y": np.arange(5.0), "x": np.ones(5)})
src = tmp / "data.csv"
df.to_csv(src, index=False)
m = rp.run_manifest(data={"d": df}, files=[src], seed=1, study_card=p, show=False)
check("run_manifest: study card hash, creation time and amendments recorded; intact",
      m["study_card"]["sha256"] == s2 and m["study_card"]["intact"] and m["study_card"]["amendments"] == 1
      and m["study_card"]["created_utc"] == json.loads(before)["created_utc"])
check("run_manifest: the amendment is a notice with its reason; no other finding when files= is given",
      [w["level"] for w in m["warnings"]] == ["notice"] and "ATE asked" in m["warnings"][0]["message"])
m = rp.run_manifest(data={"d": df}, files=[src], study_card=q, show=False)
check("run_manifest: a card changed by hand is a warning and intact is False",
      m["study_card"]["intact"] is False and [w["level"] for w in m["warnings"]] == ["warning"])
m = rp.run_manifest(data={"d": df}, files=[src], study_card=tmp / "none.json", show=False)
check("run_manifest: a missing study card is a warning", m["study_card"] is None and
      [w["level"] for w in m["warnings"]] == ["warning"])
m = rp.run_manifest(data={"d": df}, show=False)
check("run_manifest: data without files= gives the notice with the files=[DATA_FILE] action",
      [w["level"] for w in m["warnings"]] == ["notice"] and "files=[DATA_FILE]" in m["warnings"][0]["action"])
m = rp.run_manifest(data={"d": df}, files=[src], show=False)
check("run_manifest: data with files= and no card: no finding, study_card None", m["warnings"] == [] and m["study_card"] is None)
m = rp.run_manifest(show=False)
check("run_manifest: nothing passed, no finding", m["warnings"] == [])
keys = {"created_utc", "skill_version", "python", "platform", "packages", "seed", "data", "files", "results", "notes"}
check("run_manifest: every key of earlier versions is still there", keys <= set(m))
out_p = tmp / "manifest.json"
rp.run_manifest(data={"d": df}, files=[src], study_card=p, path=out_p, show=False)
check("run_manifest: the written manifest is valid JSON with the study card", json.loads(out_p.read_text())["study_card"]["intact"])

# ============================================================ 6. findings helpers and what show prints
check("_finding refuses an unknown level", bool(raises(rp._finding, "error", "m", "a")))
_, out = quiet(rp._print_findings, [rp._finding("notice", "m1", "a1"), "old plain text"])
check("_print_findings: level, message and action for dicts; WARNING for plain strings",
      "NOTICE: m1" in out and "action: a1" in out and "WARNING: old plain text" in out, out)
_, out = quiet(rp.study_card, path=tmp / "show.json", question="q", estimand="e", design="d", contorls=1)
check("study_card show: action, hash, fields and the notice with its action",
      "Study card written:" in out and "sha256" in out and "  question: q" in out and "NOTICE: non-standard" in out
      and "action: check the spelling" in out, out)
_, out = quiet(rp.run_manifest, data={"d": df}, study_card=p)
check("run_manifest show: the study card line and the files= notice with its action",
      "study card" in out and "1 amendment(s)" in out and "NOTICE: data were hashed" in out
      and "action: pass the file(s)" in out, out)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
