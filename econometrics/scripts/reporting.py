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
"""
from __future__ import annotations

import numpy as np
import pandas as pd


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
