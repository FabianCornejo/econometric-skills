"""
Publication-style regression tables (the Python counterpart of Stata's outreg2/esttab):
several models side by side, coefficients with standard errors below, significance
stars, fixed-effects indicator rows, N and fit statistics, exported to Markdown,
Excel, LaTeX or Word-friendly HTML.

Works with statsmodels (OLS/WLS/GLM/Logit/Probit/PHReg/margins), linearmodels
(PanelOLS, IV2SLS, ...) and plain DataFrames with columns coef/se/p.

Functions
---------
coef_table(result)                       Series-friendly coef/se/p for one model
regression_table(models, ...)            the side-by-side table (DataFrame of strings)
export_table(tab, path)                  .md / .xlsx / .tex / .html by extension
rescale_hint(result, X)                  flags coefficients that need unit rescaling (1e-06 etc.)

Reproducibility and verification
--------------------------------
study_card(path, **fields)               the study card written to a file with time and sha256 BEFORE estimating;
                                         changes only with amend=True and a reason (kept in the file)
run_manifest(data, files, seed, ...)     data and file hashes, package and skill versions, seed, platform, the study
                                         card's hash: what another person needs to reproduce a number (JSON-ready dict)
cross_check(estimates, se, ...)          the same quantity computed by independent routes: agreement table
                                         and a warning when they differ by more than tol_se standard errors
"""
from __future__ import annotations

import hashlib
import unicodedata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import _console  # noqa: F401  (Windows-safe console output, see _console.py)

SKILL_VERSION = "2.8.1"
_PACKAGES = ("numpy", "pandas", "scipy", "statsmodels", "linearmodels", "arch", "scikit-learn")


LEVELS = ("warning", "notice")


def _finding(level: str, message: str, action: str) -> dict:
    """One entry of `warnings` in the output of functions added from v2.7.0 on (T3): what was found,
    how serious it is ("warning": the number may mislead if read as it is; "notice": context the reader
    needs) and what to do about it. Relay all three to the user."""
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    return {"level": level, "message": message, "action": action}


def _print_findings(warnings_: list, indent: str = "  ") -> None:
    """Print findings (dicts from `_finding`, or plain strings from older functions) with level and action."""
    for w in warnings_ or []:
        if isinstance(w, dict):
            print(f"{indent}{w['level'].upper()}: {w['message']}")
            print(f"{indent}  action: {w['action']}")
        else:
            print(f"{indent}WARNING: {w}")


def coef_table(res) -> pd.DataFrame:
    """Uniform coef/se/p table from different result objects."""
    if isinstance(res, pd.DataFrame):
        return res[["coef", "se", "p"]]
    if hasattr(res, "summary_frame") and hasattr(res, "margeff"):           # statsmodels margins
        f = res.summary_frame()
        return pd.DataFrame({"coef": f.iloc[:, 0], "se": f.iloc[:, 1], "p": f.iloc[:, 3]})
    names = None
    if hasattr(res, "model") and hasattr(res.model, "exog_names"):
        names = res.model.exog_names
    params = res.params
    se = getattr(res, "std_errors", None)
    if se is None:
        se = res.bse
    p = res.pvalues
    if not isinstance(params, pd.Series):
        params, se, p = (pd.Series(np.asarray(v), index=names) for v in (params, se, p))
    return pd.DataFrame({"coef": params, "se": se, "p": p})


def _stars(p, levels=(0.01, 0.05, 0.1)):
    return "***" if p < levels[0] else "**" if p < levels[1] else "*" if p < levels[2] else ""


def _fmt(v, digits):
    if not np.isfinite(v):
        return ""
    a = abs(v)
    if a != 0 and (a < 10 ** (-digits) or a >= 1e6):
        return f"{v:.{digits - 1}e}"
    return f"{v:.{digits}f}"


def _fit_stats(res):
    out = {}
    n = getattr(res, "nobs", None)
    if n is not None:
        out["Observations"] = f"{int(n):,}"
    for attr, lab in (("rsquared_adj", "Adj. R-squared"), ("rsquared", "R-squared"),
                      ("prsquared", "Pseudo R-squared")):
        v = getattr(res, attr, None)
        if v is not None and not callable(v):
            try:
                out[lab] = f"{float(v):.3f}"
            except (TypeError, ValueError):
                pass
    return out


def regression_table(models: dict, keep: list[str] | None = None, labels: dict | None = None,
                     fe_rows: dict[str, list[str]] | None = None, exponentiate: bool = False,
                     digits: int = 3, se_note: str = "Robust standard errors in parentheses",
                     extra_rows: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """models: {"(1) Base": res1, "(2) + controls": res2, ...}
    keep: coefficient names to show, in order (others, e.g. FE dummies, are hidden).
    labels: {"np.log(price)": "Log price"}.
    fe_rows: {"Year FE": ["YES", "YES", "NO"], ...} — say YES only where the FE is in the model.
    exponentiate: report exp(coef) (odds/hazard ratios) with delta-method SE = exp(b)*se."""
    tabs = {k: coef_table(v) for k, v in models.items()}
    names = keep or list(dict.fromkeys(n for t in tabs.values() for n in t.index))
    rows = []
    for nm in names:
        r1, r2 = {"": (labels or {}).get(nm, nm)}, {"": ""}
        for k, t in tabs.items():
            if nm in t.index:
                b, s, p = t.loc[nm, ["coef", "se", "p"]]
                if exponentiate:
                    b, s = np.exp(b), np.exp(b) * s
                r1[k] = _fmt(b, digits) + _stars(p)
                r2[k] = f"({_fmt(s, digits)})"
            else:
                r1[k] = r2[k] = ""
        rows += [r1, r2]
    rows.append({"": "", **{k: "" for k in models}})
    stats_ = {k: _fit_stats(v) for k, v in models.items()}
    for lab in ("Observations", "R-squared", "Adj. R-squared", "Pseudo R-squared"):
        if any(lab in s for s in stats_.values()):
            rows.append({"": lab, **{k: stats_[k].get(lab, "") for k in models}})
    for lab, vals in {**(fe_rows or {}), **(extra_rows or {})}.items():
        rows.append({"": lab, **dict(zip(models, vals))})
    rows.append({"": se_note, **{k: "" for k in models}})
    rows.append({"": "*** p<0.01, ** p<0.05, * p<0.1", **{k: "" for k in models}})
    return pd.DataFrame(rows).set_index("")


def export_table(tab: pd.DataFrame, path: str):
    p = str(path).lower()
    if p.endswith(".md"):
        open(path, "w", encoding="utf-8").write(tab.to_markdown())
    elif p.endswith(".xlsx"):
        from setup_env import require_extra   # optional openpyxl: one clear line if missing (v2.3.6)
        require_extra("excel", "Writing an .xlsx table")
        tab.to_excel(path)
    elif p.endswith(".tex"):
        open(path, "w", encoding="utf-8").write(tab.to_latex(escape=True))
    elif p.endswith(".html"):
        open(path, "w", encoding="utf-8").write(tab.to_html())
    else:
        tab.to_csv(path)
    return path


def rescale_hint(res, X: pd.DataFrame | None = None) -> pd.DataFrame:
    """Coefficients like 4.6e-05 are unreadable: rescale the regressor (R$ → R$ thousand,
    per capita → per 1,000 inhabitants) so the coefficient is a meaningful change.
    Suggests the power of 10 that brings |coef × SD(x)| to a readable range."""
    t = coef_table(res)
    rows = []
    for nm, r in t.iterrows():
        sd = float(X[nm].std()) if X is not None and nm in X else np.nan
        mag = abs(r["coef"])
        if mag and (mag < 1e-3 or mag > 1e4):
            k = int(np.round(-np.log10(mag)))
            rows.append({"variable": nm, "coef": r["coef"], "sd_x": sd,
                         "suggest": f"express {nm} in units of 10^{k} (coefficient becomes coef × 10^{k})" if k > 0 else f"multiply {nm} by 10^{-k}",
                         "effect_of_1sd": r["coef"] * sd if np.isfinite(sd) else np.nan})
    return pd.DataFrame(rows)


def _hash_frame(df: pd.DataFrame) -> str:
    """Content hash of a DataFrame: values, index, column names and dtypes (row order matters)."""
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
    h.update(json.dumps([str(c) for c in df.columns]).encode())
    h.update(json.dumps([str(t) for t in df.dtypes]).encode())
    return h.hexdigest()


CARD_FIELDS = ("question", "estimand", "design", "assumption", "y", "treatment", "X", "W", "Z", "data", "threats",
               "robustness_plan", "alpha", "family_of_tests", "horizon", "loss", "benchmarks", "window", "mde")
CARD_REQUIRED = ("question", "estimand", "design")

# v2.8.1: names an agent or a user writes for a standard field (the D6 validation of v2.8.0 wrote the first three
# from the prose of SKILL.md). Keys are compared after _card_key(); the card itself is never renamed. Only names that
# mean one field in every design (reviews r1, r2 and r3): "control" and "controls" (the control group in a
# case-control study?), "covariates" (X in a heterogeneity study, W elsewhere), "treated" (the treated group?),
# "level" (cluster level?), "exposure" (a portfolio's exposure?), "source", "risks", "family" and "tests" are left
# out on purpose, and no suggestion comes from a spelling that only looks close ("estimador" is not "estimand").
CARD_SYNONYMS = {
    "identifyingassumption": "assumption", "identificationassumption": "assumption", "assumptions": "assumption",
    "keyassumption": "assumption",
    "identificationstrategy": "design", "researchdesign": "design", "researchquestion": "question",
    "treat": "treatment", "treatmentvariable": "treatment", "treatmentvar": "treatment", "intervention": "treatment",
    "robustness": "robustness_plan", "robustnesscheck": "robustness_plan", "robustnesschecks": "robustness_plan",
    "robustnessplans": "robustness_plan", "robustnesstests": "robustness_plan",
    "testsfamily": "family_of_tests", "testfamily": "family_of_tests", "familyoftest": "family_of_tests",
    "testfamilies": "family_of_tests", "multipletests": "family_of_tests",
    "outcome": "y", "outcomes": "y", "outcomevariable": "y", "outcomevar": "y", "dependentvariable": "y",
    "confounder": "W", "confounders": "W", "controlvariables": "W",
    "instruments": "Z", "instrument": "Z", "instrumentalvariable": "Z", "instrumentalvariables": "Z",
    "effectmodifier": "X", "effectmodifiers": "X",
    "significance": "alpha", "significancelevel": "alpha", "siglevel": "alpha", "alphalevel": "alpha",
    "minimumdetectableeffect": "mde", "mindetectableeffect": "mde", "forecasthorizon": "horizon",
    "lossfunction": "loss", "benchmark": "benchmarks", "threat": "threats", "threatstovalidity": "threats",
    "validitythreats": "threats", "threatstoidentification": "threats", "dataset": "data", "datasource": "data",
    "datasources": "data",
}


def _card_key(name: str) -> str:
    """Lower case, letters and digits only: 'Identifying assumption' -> 'identifyingassumption'."""
    return "".join(ch for ch in unicodedata.normalize("NFC", str(name)).lower() if ch.isalnum())


def _card_suggestion(name: str, given) -> str | None:
    """The standard field a non-standard name means, or None: the same key up to case and separators ("Robustness
    Plan" -> robustness_plan) or a name in CARD_SYNONYMS. Never a field already in the card; no guess from spelling."""
    key = _card_key(name)
    cand = {_card_key(f): f for f in CARD_FIELDS}.get(key) or CARD_SYNONYMS.get(key)
    return cand if cand is not None and cand not in given else None


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _card_hash(card: dict) -> str:
    return hashlib.sha256(_canonical({k: v for k, v in card.items() if k != "sha256"})).hexdigest()


def _plain(x):
    """numpy scalars to Python numbers, for JSON (np.int64 and np.float64 alike)."""
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(f"{type(x).__name__} is not text, a number or a list")


def _read_card(path: Path) -> dict:
    """The card in `path`; ValueError with the reason if it is not a readable card (empty, not JSON, a folder)."""
    try:
        card = json.loads(Path(path).read_bytes().decode("utf-8-sig"))
    except (OSError, UnicodeDecodeError, ValueError) as e:
        raise ValueError(f"{path} could not be read as a study card ({type(e).__name__}: {e})") from None
    if not isinstance(card, dict) or not isinstance(card.get("fields"), dict):
        raise ValueError(f"{path} is not a study card written by reporting.study_card")
    return card


def _write_atomic(path: Path, text: str) -> None:
    import tempfile
    try:
        fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent if str(path.parent) else ".")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except OSError as e:
        raise ValueError(f"the study card could not be written to {path} ({e})") from None


def _nfc(obj, strict: bool = True):
    """Texts (keys and values, in lists too) in Unicode NFC (v2.7.1): "café" typed with a combining accent (NFD, as some
    editors and macOS file names give it) and with a precomposed é is the same text, and must not make a re-run look like
    a changed plan."""
    if isinstance(obj, str):
        return unicodedata.normalize("NFC", obj)
    if isinstance(obj, list):
        return [_nfc(x, strict) for x in obj]
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            kk = _nfc(k)
            if kk in out and strict:
                raise _NameClash(f"two study card fields are the same name once normalised (NFC): {k!r}")
            out[kk] = _nfc(v, strict)
        return out
    return obj


class _NameClash(ValueError):
    pass


def _blank(v) -> bool:
    """A required field must be text, or a list of texts, with something in it (0, False and [" "] are blank)."""
    if isinstance(v, str):
        return not v.strip()
    if isinstance(v, list):
        return not v or all(_blank(x) for x in v)
    return True


def study_card(path: str = "study_card.json", amend: bool = False, reason: str | None = None, show: bool = True,
               **fields) -> dict:
    """Write the study card (references/workflow.md §0) to a file BEFORE any estimate, with the time and a
    sha256 of its content, so that "design before data" and "declare the family of tests" can be checked
    afterwards: run_manifest(study_card=path) records the hash next to the results.

    Returns a dict (v2.7.2: keys exact):
      "path"           str
      "sha256"         str, the hash stored in the card (not that of the content if the card was edited by hand: a
                       warning says so)
      "created_utc"    str
      "fields"         dict, the card's fields (NFC when written by v2.7.1 or later)
      "amendments"     list of the amendments, each with its time and reason (run_manifest gives their count)
      "action"         str: "written", "unchanged" or "amended"
      "used"           dict, keys ["path", "action", "skill_version"]
      "warnings"       list of dicts with keys ["level", "message", "action"]

    fields: question, estimand, design (required, not blank); assumption, y, treatment, X, W, Z, data,
    threats, robustness_plan, alpha, family_of_tests, mde; for a forecast, horizon, loss, benchmarks and
    window. Values are text, finite numbers or lists (JSON; numpy numbers are converted). Other names are
    kept and reported as a notice, which names the standard field a name means when it is the same key up to case
    and separators or a listed synonym ("identifying_assumption (did you mean assumption?)", v2.8.1; never a field
    the card already has; no guess from spelling); nothing is renamed. Texts are
    compared and written in Unicode NFC (v2.7.1), so the same words
    typed with combining accents count as the same plan (a card already on file keeps its text as written, and the
    result's `fields` are the ones on file).

    Re-running the same script is safe: with the same fields the file is left untouched (same time, same
    hash). Different fields are refused, because a card that changes after the results are seen is no
    longer a plan; pass amend=True and a reason to change it: the previous version, its skill version and
    the reason are kept in the file's `amendments`, never deleted, and the manifest reports them. A card
    edited by hand (stored hash different from the content) is reported; amending it records that it had
    been edited by hand, and run_manifest keeps reporting it.

    What the card can and cannot show: the hash catches an edit made outside this function and the
    amendment log keeps every recorded change; it does not stop a deliberate rewrite (anyone can recompute
    a hash or delete the file). It is evidence of order for an honest workflow; for a third party, share
    the card (or commit it to version control) before estimating.

    Returns path, sha256, created_utc, fields, amendments, `used` and `warnings` (level, message, action)."""
    p = Path(path)
    try:
        fields = _nfc(json.loads(json.dumps(fields, default=_plain, allow_nan=False)))
    except _NameClash:
        raise
    except (TypeError, ValueError) as e:
        raise ValueError(f"study card fields must be text, finite numbers or lists: {e}") from None
    missing = [f for f in CARD_REQUIRED if _blank(fields.get(f))]
    if missing:
        raise ValueError(f"the study card needs {', '.join(missing)} (question, estimand and design are required and "
                         "cannot be blank)")
    warns = []
    extra = sorted(set(fields) - set(CARD_FIELDS))
    if extra:
        sugg = {f: _card_suggestion(f, fields) for f in extra}
        named = [f"{f} (did you mean {sugg[f]}?)" if sugg[f] else f for f in extra]
        msg = f"non-standard study card fields kept as given: {', '.join(named)}"
        for target in sorted({v for v in sugg.values() if v}):
            same = [f for f in extra if sugg[f] == target]
            if len(same) > 1:   # reviews r1 S3-3 and r2 S3-6
                msg += (f"; {', '.join(same[:-1])} and {same[-1]} all mean {target}: merge them into one "
                        f"{target} field")
        std = "The standard fields are " + ", ".join(CARD_FIELDS)
        if any(sugg.values()):   # reviews r2 S2-4, r3 S1-1, S2-1, S3-2
            act = ("the card is already written with these names. To use a standard key, call study_card again with "
                   "ALL the fields, each standard key in place of the name above, plus amend=True and a reason that "
                   "says the names changed and whether results had been seen (it is recorded as an amendment); a "
                   "field left out of that call is dropped from the card and kept only in its amendments. Until "
                   "then a standard field left out stays empty. " + std)
        else:
            act = ("check that these names are not standard fields spelled another way; non-standard fields are "
                   "kept as given. " + std)
        warns.append(_finding("notice", msg, act))
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    action = "written"
    if p.exists():
        card = _read_card(p)
        intact = card.get("sha256") == _card_hash(card)
        same = _canonical(_nfc(card["fields"], strict=False)) == _canonical(fields)     # a card written before v2.7.1 may hold NFD
        if not intact:
            warns.append(_finding("warning", f"{p.name} was changed outside reporting.study_card (its stored hash does not "
                                  "match its content)", "say so in the report; restore the card from version control, or "
                                  "record the change with amend=True and a reason (the amendment notes the hand edit)"))
        if same and intact:
            action = "unchanged"
            if amend:
                warns.append(_finding("notice", "amend=True but the fields are the ones already on file: nothing to amend, "
                                      "the reason was not recorded", "pass amend=True only with the changed fields"))
        elif not amend:
            if same:
                action = "unchanged"
            else:
                on_file = _nfc(card["fields"], strict=False)    # review r2 R5: a v2.7.0 card stays readable
                changed = sorted(k for k in set(on_file) | set(fields)
                                 if _canonical(on_file.get(k)) != _canonical(fields.get(k)))
                hand = (" The card on file was edited by hand (its hash does not match): if the script holds the plan "
                        "as written, restore it, or record the difference with amend=True and a reason." if not intact else "")
                raise ValueError(f"{p} already holds a study card (created {card.get('created_utc')}) and these fields "
                                 f"differ: {', '.join(changed)}. A plan changed after seeing results is not a plan: pass "
                                 "amend=True and reason='...' to record the change." + hand)
        else:
            if not (reason and str(reason).strip()):
                raise ValueError("amend=True needs a reason (why the plan changes, and whether results had been seen)")
            card.setdefault("amendments", []).append({
                "replaced_fields": card["fields"], "replaced_sha256": card.get("sha256"),
                "replaced_skill_version": card.get("skill_version"), "edited_by_hand_before": not intact,
                "amended_utc": now, "reason": str(reason)})
            card["fields"] = fields
            card["skill_version"] = SKILL_VERSION
            card["sha256"] = _card_hash(card)
            _write_atomic(p, json.dumps(card, indent=2, ensure_ascii=False))
            action = "amended"
    else:
        card = {"fields": fields, "created_utc": now, "skill_version": SKILL_VERSION, "amendments": []}
        card["sha256"] = _card_hash(card)
        _write_atomic(p, json.dumps(card, indent=2, ensure_ascii=False))
    out = {"path": str(p), "sha256": card["sha256"], "created_utc": card["created_utc"], "fields": card["fields"],
           "amendments": card.get("amendments", []), "action": action,
           "used": {"path": str(p), "action": action, "skill_version": SKILL_VERSION}, "warnings": warns}
    if show:
        print(f"\nStudy card {action}: {p} (created {card['created_utc']}, sha256 {card['sha256'][:16]}…, "
              f"{len(out['amendments'])} amendment(s))")
        for k, v in card["fields"].items():
            print(f"  {k}: {v}")
        _print_findings(warns)
    return out


def run_manifest(data: dict | pd.DataFrame | None = None, files: list | None = None, seed=None,
                 results: dict | None = None, notes: str = "", path: str | None = None, show: bool = True,
                 study_card: str | None = None, simulated: bool = False) -> dict:
    """What another person needs to reproduce a number: sha256 of every data set used (DataFrames
    by content, files byte by byte), the versions of Python, the numerical packages and this
    skill, the random seed, the platform and the time, plus the headline results. Two runs that
    print the same hashes and versions should give the same numbers; if a number changes,
    compare manifests first. path='manifest.json' also writes it to disk.

    Returns a dict (v2.7.2: keys exact):
      "created_utc"    str, ISO time
      "skill_version"  str
      "python"         str
      "platform"       str
      "packages"       dict, package name -> version (None when not installed)
      "seed"           as passed
      "data"           dict, name -> keys ["sha256", "rows", "cols"] (a DataFrame alone is named "data")
      "files"          dict, path -> sha256 of the bytes, or "missing"
      "results"        dict, the results passed as plain numbers; of a dict value only "estimate", "se", "ci_low",
                       "ci_high", "p" and "n" are kept, and what JSON cannot hold becomes text
      "notes"          str
      "simulated"      bool
      "study_card"     dict with keys ["path", "sha256", "sha256_recomputed", "intact", "created_utc", "amendments",
                       "amendment_reasons", "edited_by_hand"] ("amendments" a count here) when study_card= names a
                       card that can be read, else None (a missing or unreadable card is also a warning)
      "warnings"       list of dicts with keys ["level", "message", "action"]

    Since v2.7.0: study_card='study_card.json' records the card's hash, creation time and amendments
    (reporting.study_card), and a card edited by hand is a warning; data passed without the source file
    in files= is a notice, because the DataFrame hash changes with any cleaning step and the file hash is
    what lets another person check they start from the same data; simulated=True says the data were
    simulated in the script (no file to hash: the seed reproduces them) and silences that notice. A path in
    files= that does not exist is recorded as "missing" and is a warning. A card that cannot be read is a
    warning, not an error, so the manifest is still written. Findings are in `warnings`."""
    if not isinstance(simulated, (bool, np.bool_)):
        raise ValueError("simulated must be True or False")
    frames = {"data": data} if isinstance(data, pd.DataFrame) else (data or {})
    man = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "skill_version": SKILL_VERSION,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {},
        "seed": seed,
        "data": {k: {"sha256": _hash_frame(v), "rows": int(v.shape[0]), "cols": int(v.shape[1])}
                 for k, v in frames.items()},
        "files": {},
        "results": {},
        "notes": notes,
        "simulated": bool(simulated),
    }
    for p in _PACKAGES:
        try:
            man["packages"][p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            man["packages"][p] = None
    for f in files or []:
        fp = Path(f)
        man["files"][str(f)] = hashlib.sha256(fp.read_bytes()).hexdigest() if fp.is_file() else "missing"
    for k, v in (results or {}).items():
        if isinstance(v, dict):
            v = {kk: vv for kk, vv in v.items() if kk in ("estimate", "se", "ci_low", "ci_high", "p", "n")}
        man["results"][k] = json.loads(json.dumps(v, default=lambda o: float(o) if np.isscalar(o) else str(o)))
    warns = []
    man["study_card"] = None
    if study_card is not None:
        cp = Path(study_card)
        card = None
        if not cp.is_file():
            warns.append(_finding("warning", f"study card {cp} {'is not a file' if cp.exists() else 'not found'}: the design "
                                  "cannot be shown to predate the results",
                                  "write it with reporting.study_card before estimating, or say in the report that there was none"))
        else:
            try:
                card = _read_card(cp)
            except ValueError as e:
                card = None
                warns.append(_finding("warning", f"study card {cp} could not be read: {e}",
                                      "restore the card written by reporting.study_card, or say in the report that it is lost"))
        if card is not None:
            recomputed = _card_hash(card)
            amends = card.get("amendments", [])
            man["study_card"] = {"path": str(cp), "sha256": card.get("sha256"), "sha256_recomputed": recomputed,
                                 "intact": card.get("sha256") == recomputed, "created_utc": card.get("created_utc"),
                                 "amendments": len(amends),
                                 "amendment_reasons": [f"{a.get('amended_utc')}: {a.get('reason')}" for a in amends],
                                 "edited_by_hand": [a.get("amended_utc") for a in amends if a.get("edited_by_hand_before")]}
            if not man["study_card"]["intact"]:
                warns.append(_finding("warning", f"study card {cp.name} was changed outside reporting.study_card after it was "
                                      "written (hash mismatch)", "report the plan as amended without a record, or restore it"))
            if man["study_card"]["edited_by_hand"]:
                warns.append(_finding("warning", "the study card had been changed by hand before the amendment(s) of "
                                      + ", ".join(man["study_card"]["edited_by_hand"]) + "; the replaced fields recorded "
                                      "there are the hand-edited ones, not necessarily the original plan",
                                      "say so in the report; the original plan is in version control or a shared copy, if any"))
            if amends:
                warns.append(_finding("notice", f"the study card was amended {len(amends)} time(s): "
                                      + "; ".join(man["study_card"]["amendment_reasons"]),
                                      "report each amendment and whether results had been seen before it"))
    missing = [k for k, v in man["files"].items() if v == "missing"]
    if missing:
        warns.append(_finding("warning", f"file(s) not found, recorded as 'missing': {', '.join(missing)}",
                              "pass the path of the file the data were read from (relative to the folder the script runs in)"))
    if frames and not (files or []) and not simulated:
        warns.append(_finding("notice", "data were hashed as DataFrames but no source file was passed in files=",
                              "pass the file(s) the data were read from, files=[DATA_FILE], so the raw input can be checked "
                              "(simulated=True when the data were simulated in the script)"))
    man["warnings"] = warns
    if path:
        Path(path).write_text(json.dumps(man, indent=2))
    if show:
        print(f"\nRun manifest (skill {SKILL_VERSION}, Python {man['python']}, seed {seed})")
        for k, v in man["data"].items():
            print(f"  data {k}: {v['rows']}x{v['cols']}, sha256 {v['sha256'][:16]}…")
        for k, v in man["files"].items():
            print(f"  file {k}: sha256 {v[:16]}…")
        if man["study_card"]:
            sc = man["study_card"]
            print(f"  study card {sc['path']}: sha256 {str(sc['sha256'])[:16]}…, created {sc['created_utc']}, "
                  f"{sc['amendments']} amendment(s){'' if sc['intact'] else ', CHANGED BY HAND'}")
        print("  packages: " + ", ".join(f"{k} {v}" for k, v in man["packages"].items() if v))
        _print_findings(warns)
    return man


def cross_check(estimates, se: float | None = None, tol_se: float = 1.0, rel_tol: float = 1e-6,
                label: str = "estimate", weak_ratio: float = 0.25, alpha: float = 0.05, show: bool = True) -> dict:
    """Independent verification of a headline number (the dynare-mcp idea of solving the same
    model twice): pass the same quantity computed by different routes, e.g. {"DR": ..., "outcome
    regression": ..., "IPW": ...}, {"statsmodels": ..., "linearmodels": ...} or {"formula": ...,
    "simulation": ...}. With se, routes that estimate the same target under different modelling
    assumptions agree if they are within tol_se standard errors; without se, they must agree to
    rel_tol (two implementations of the same estimator). Disagreement is a finding: report it,
    do not pick the most convenient number.

    Returns a dict (v2.7.2: keys exact, the same with one route):
      "table"          DataFrame, one row per route, one column named by label (default: columns ["estimate"])
      "spread"         float, max minus min of the estimates (NaN with one route)
      "agree"          bool, or None when nothing was checked
      "criterion"      str, the rule used ("within 1 SE (0.2)", "not checked", ...)
      "null"           bool, the routes agree and every one is compatible with zero
      "kind"           str: "method" (with a positive se) or "implementation" (without), or None
      "weak"           bool, the tolerance is a large share of the estimate
      "note"           str ("" when there is nothing to add)
      "warnings"       list of str

    estimates: a dict or a pandas Series of numbers, e.g. att_gt's res["cross_check"] passed as
    it is (a Series dr/reg/ipw). At least two routes are needed: with one, nothing was checked,
    and the result says so instead of reporting agreement.

    What was checked (v2.3.7, result["kind"]): with se, routes are different methods or assumptions
    (kind "method"), an independent check of the result. Without se they must agree to rel_tol, which
    only two implementations of the same estimator do (kind "implementation"): that verifies the
    arithmetic, not the method, and is printed as an implementation check, never as an independent
    confirmation. For a headline number, add a route with a different method (robust vs OLS,
    bootstrap vs analytic CI, DR vs outcome regression vs IPW, formula vs simulation).

    A tolerance can be too loose to test anything (v2.4.6.1): when tol_se·se exceeds weak_ratio (25%) of the
    estimate's size, "agree" would have been printed for routes far apart in practical terms (in a 12-month
    inflation forecast of 5.3% with SE 2.1, anything from 3.2% to 7.4% "agreed"). The result then says the check
    is weak and reports the actual spread, which is the number to judge. When every route is compatible with
    zero at level alpha (|estimate| < z·SE, z = Φ⁻¹(1 − alpha/2); 1.96 for the default 0.05; v2.4.7.1, alpha since
    v2.4.7.2), the agreement is flagged as uninformative: two estimates compatible with zero agree with each
    other trivially."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if isinstance(estimates, dict) and any(isinstance(v, dict) for v in estimates.values()):
        estimates = {k: (v.get("estimate") if isinstance(v, dict) else v) for k, v in estimates.items()}
    s = pd.Series(estimates, dtype=float).dropna()
    if len(s) < 2:
        msg = (f"cross_check received {len(s)} route(s) for the {label}: nothing was verified. Pass at least two "
               "independent computations (for att_gt: reporting.cross_check(res['cross_check'], se=...)).")
        if show:
            print("\n  WARNING: " + msg)
        return {"table": s.rename(label).to_frame(), "spread": np.nan, "agree": None, "criterion": "not checked",
                "kind": None, "weak": False, "null": False, "note": "", "warnings": [msg]}
    spread = float(s.max() - s.min())
    if se is not None and np.isfinite(se) and se > 0:
        kind = "method"
        agree = spread <= tol_se * se
        crit = f"within {tol_se:g} SE ({se:.4g})"
    else:
        kind = "implementation"
        scale = max(1.0, float(s.abs().max()))
        agree = spread <= rel_tol * scale
        crit = f"to a relative tolerance of {rel_tol:g}"
    warns = [] if agree else [f"Independent routes disagree on the {label}: " + ", ".join(
        f"{k} {v:.6g}" for k, v in s.items()) + f" (spread {spread:.4g}, required {crit})."]
    weak = False
    null = False
    size = float(s.abs().mean())
    # every route compatible with zero at alpha (|estimate| < z SE): agreeing then says nothing about the effect
    from scipy import stats
    z = float(stats.norm.ppf(1 - alpha / 2))
    if kind == "method" and agree and float(s.abs().max()) < z * se:
        null = True
        warns.append(f"uninformative agreement: every route is compatible with zero at alpha = {alpha:g} "
                     f"(|{label}| < {z:.3g} SE = {z * se:.4g}); the routes agree, but the effect itself is not measured")
    # weak only for an agreement on an estimate clearly away from zero: for a statistically null estimate
    # (|estimate| below the tolerance) the relative size of the tolerance means nothing
    if kind == "method" and agree and not null and size > tol_se * se and tol_se * se > weak_ratio * size:
        weak = True
        warns.append(f"weak check: the tolerance ({tol_se:g} SE = {tol_se * se:.4g}) is {tol_se * se / size:.0%} of the "
                     f"{label}, so routes far apart in practice would still 'agree'; judge the spread ({spread:.4g}) "
                     "against the difference that matters for the decision")
    note = ("" if kind == "method" else
            "same estimator computed twice: this verifies the arithmetic, not the method, and is not an independent "
            "confirmation; for a headline number add a route with a different method (pass se=).")
    out = {"table": s.rename(label).to_frame(), "spread": spread, "agree": bool(agree), "criterion": crit, "null": null,
           "kind": kind, "weak": weak, "note": note, "warnings": warns}
    if show:
        head = "Independent check (different methods)" if kind == "method" else "Implementation check"
        print(f"\n{head} of the {label}: " + ", ".join(f"{k} {v:.6g}" for k, v in s.items())
              + f" → {'agree' if agree else 'DISAGREE'} ({crit}; spread {spread:.4g})")
        if note:
            print("  note: " + note)
        for w in warns:
            print("  WARNING: " + w)
    return out
