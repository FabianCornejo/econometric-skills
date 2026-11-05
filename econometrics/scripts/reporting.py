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
run_manifest(data, files, seed, ...)     data and file hashes, package and skill versions, seed, platform:
                                         what another person needs to reproduce a number (JSON-ready dict)
cross_check(estimates, se, ...)          the same quantity computed by independent routes: agreement table
                                         and a warning when they differ by more than tol_se standard errors
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import _console  # noqa: F401  (Windows-safe console output, see _console.py)

SKILL_VERSION = "2.4.6.2"
_PACKAGES = ("numpy", "pandas", "scipy", "statsmodels", "linearmodels", "arch", "scikit-learn")


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


def run_manifest(data: dict | pd.DataFrame | None = None, files: list | None = None, seed=None,
                 results: dict | None = None, notes: str = "", path: str | None = None, show: bool = True) -> dict:
    """What another person needs to reproduce a number: sha256 of every data set used (DataFrames
    by content, files byte by byte), the versions of Python, the numerical packages and this
    skill, the random seed, the platform and the time, plus the headline results. Two runs that
    print the same hashes and versions should give the same numbers; if a number changes,
    compare manifests first. path='manifest.json' also writes it to disk."""
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
    if path:
        Path(path).write_text(json.dumps(man, indent=2))
    if show:
        print(f"\nRun manifest (skill {SKILL_VERSION}, Python {man['python']}, seed {seed})")
        for k, v in man["data"].items():
            print(f"  data {k}: {v['rows']}x{v['cols']}, sha256 {v['sha256'][:16]}…")
        for k, v in man["files"].items():
            print(f"  file {k}: sha256 {v[:16]}…")
        print("  packages: " + ", ".join(f"{k} {v}" for k, v in man["packages"].items() if v))
    return man


def cross_check(estimates, se: float | None = None, tol_se: float = 1.0, rel_tol: float = 1e-6,
                label: str = "estimate", weak_ratio: float = 0.25, show: bool = True) -> dict:
    """Independent verification of a headline number (the dynare-mcp idea of solving the same
    model twice): pass the same quantity computed by different routes, e.g. {"DR": ..., "outcome
    regression": ..., "IPW": ...}, {"statsmodels": ..., "linearmodels": ...} or {"formula": ...,
    "simulation": ...}. With se, routes that estimate the same target under different modelling
    assumptions agree if they are within tol_se standard errors; without se, they must agree to
    rel_tol (two implementations of the same estimator). Disagreement is a finding: report it,
    do not pick the most convenient number.

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
    is weak and reports the actual spread, which is the number to judge."""
    if isinstance(estimates, dict) and any(isinstance(v, dict) for v in estimates.values()):
        estimates = {k: (v.get("estimate") if isinstance(v, dict) else v) for k, v in estimates.items()}
    s = pd.Series(estimates, dtype=float).dropna()
    if len(s) < 2:
        msg = (f"cross_check received {len(s)} route(s) for the {label}: nothing was verified. Pass at least two "
               "independent computations (for att_gt: reporting.cross_check(res['cross_check'], se=...)).")
        if show:
            print("\n  WARNING: " + msg)
        return {"table": s.rename(label).to_frame(), "spread": np.nan, "agree": None, "criterion": "not checked",
                "warnings": [msg]}
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
    size = float(s.abs().mean())
    # weak only for an agreement on an estimate clearly away from zero: for a statistically null estimate
    # (|estimate| below the tolerance) the relative size of the tolerance means nothing
    if kind == "method" and agree and size > tol_se * se and tol_se * se > weak_ratio * size:
        weak = True
        warns.append(f"weak check: the tolerance ({tol_se:g} SE = {tol_se * se:.4g}) is {tol_se * se / size:.0%} of the "
                     f"{label}, so routes far apart in practice would still 'agree'; judge the spread ({spread:.4g}) "
                     "against the difference that matters for the decision")
    note = ("" if kind == "method" else
            "same estimator computed twice: this verifies the arithmetic, not the method, and is not an independent "
            "confirmation; for a headline number add a route with a different method (pass se=).")
    out = {"table": s.rename(label).to_frame(), "spread": spread, "agree": bool(agree), "criterion": crit,
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
