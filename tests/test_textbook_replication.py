"""Textbook replication tests: the skill's functions and recipes must reproduce published numbers.

Benchmarks come from Wooldridge, *Introductory Econometrics: A Modern Approach* (examples cited
by number), using the datasets shipped with the `wooldridge` package. These complement the
known-truth simulations in the other test files: simulations catch estimators that do not recover
the truth; replications catch convention errors (degrees of freedom, constants, scaling) that a
simulation would not reveal.

Also checks:
- ols_diagnostics runs autocorrelation tests only for time series (row order is arbitrary in a
  cross-section) and adds Engle's ARCH-LM for time series;
- Tobit with lower/upper limits recovers known parameters, and its three average marginal effects
  equal finite-difference derivatives of the corresponding expectations.

Run from the repository root:  python3 tests/test_textbook_replication.py
Needs: statsmodels, linearmodels, scipy, wooldridge (pip install wooldridge).
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "econometrics" / "scripts"))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

import diagnostics as d

try:
    import wooldridge as woo
except ImportError:  # pragma: no cover
    sys.exit("SKIP: install the `wooldridge` package (pip install wooldridge) to run these tests")

ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


def close(a, b, tol):
    return abs(float(a) - float(b)) <= tol


# ---------------------------------------------------------------------------------------------
# 1. Textbook replications
# ---------------------------------------------------------------------------------------------

# Example 8.4 — Breusch-Pagan test on the housing price equation (HPRICE1): LM = 14.09, p = .0028
h = woo.data("hprice1")
r = smf.ols("price ~ lotsize + sqrft + bdrms", h).fit()
tab = d.ols_diagnostics(r, show=False)
bp = tab.loc[tab["Test"] == "Breusch-Pagan"].iloc[0]
check("W8.4 Breusch-Pagan LM = 14.09", close(bp["Statistic"], 14.09, 0.01), f"got {bp['Statistic']:.3f}")
check("W8.4 Breusch-Pagan p = .0028", close(bp["p-value"], 0.0028, 0.0001), f"got {bp['p-value']:.4f}")

# Example 13.3 — difference-in-differences, incinerator and house prices (KIELMC): δ1 = −11,863.9
k = woo.data("kielmc")
did = smf.ols("rprice ~ nearinc*y81", k).fit()
check("W13.3 DiD estimate = −11,863.9", close(did.params["nearinc:y81"], -11863.9, 0.5),
      f"got {did.params['nearinc:y81']:.1f}")

# Example 14.4 — fixed effects wage equation (WAGEPAN): union .080, married .047
from linearmodels.panel import PanelOLS  # noqa: E402

w = woo.data("wagepan").set_index(["nr", "year"], drop=False)
fe = PanelOLS.from_formula("lwage ~ expersq + union + married + C(year) + EntityEffects", w,
                           drop_absorbed=True).fit()
check("W14.4 FE union = .080", close(fe.params["union"], 0.080, 0.0006), f"got {fe.params['union']:.4f}")
check("W14.4 FE married = .047", close(fe.params["married"], 0.047, 0.0006), f"got {fe.params['married']:.4f}")

# Examples 15.1, 15.5, 15.7, 15.8 — return to education for married women (MROZ)
from linearmodels.iv import IV2SLS  # noqa: E402

m = woo.data("mroz").dropna(subset=["wage"]).copy()
m["lwage"] = np.log(m["wage"])
m["expersq"] = m["exper"] ** 2

# debiased=True applies the n−k small-sample correction used by the textbook (Stata: `ivregress ..., small`);
# linearmodels' default divides by n, like Stata's default, and gives slightly smaller standard errors.
iv1 = IV2SLS.from_formula("lwage ~ 1 + [educ ~ fatheduc]", m).fit(cov_type="unadjusted", debiased=True)
check("W15.1 IV educ = .059 (se .035)",
      close(iv1.params["educ"], 0.059, 0.0006) and close(iv1.std_errors["educ"], 0.035, 0.0006),
      f"got {iv1.params['educ']:.4f} ({iv1.std_errors['educ']:.4f})")

iv2 = IV2SLS.from_formula("lwage ~ 1 + exper + expersq + [educ ~ motheduc + fatheduc]", m) \
    .fit(cov_type="unadjusted", debiased=True)
check("W15.5 2SLS educ = .0614 (se .0314)",
      close(iv2.params["educ"], 0.0614, 0.00006) and close(iv2.std_errors["educ"], 0.0314, 0.0001),
      f"got {iv2.params['educ']:.4f} ({iv2.std_errors['educ']:.4f})")

endo = d.endogeneity_test(m["lwage"], m[["exper", "expersq"]], m[["educ"]],
                          m[["motheduc", "fatheduc"]], cov_type="nonrobust", show=False)
check("W15.7 Durbin-Wu-Hausman t = 1.67 (F = t² ≈ 2.79)", close(endo["statistic"], 1.67 ** 2, 0.02),
      f"got F = {endo['statistic']:.3f}")

over = d.overid_test(iv2.resids, m[["exper", "expersq"]], m[["motheduc", "fatheduc"]], n_endog=1, show=False)
check("W15.8 overidentification nR² = .378, p = .539",
      close(over["nR2"], 0.378, 0.0006) and close(over["p"], 0.539, 0.0006),
      f"got {over['nR2']:.3f}, p = {over['p']:.3f}")

# Example 17.2 — Tobit for annual hours worked (MROZ, all 753 women)
mz = woo.data("mroz")
Xt = mz[["nwifeinc", "educ", "exper", "age", "kidslt6", "kidsge6"]].assign(expersq=mz["exper"] ** 2)
tb = d.tobit(mz["hours"], Xt, show=False)
check("W17.2 Tobit educ = 80.65", close(tb["table"].loc["educ", "latent_coef"], 80.65, 0.01),
      f"got {tb['table'].loc['educ', 'latent_coef']:.2f}")
check("W17.2 Tobit kidslt6 = −894.02", close(tb["table"].loc["kidslt6", "latent_coef"], -894.02, 0.05),
      f"got {tb['table'].loc['kidslt6', 'latent_coef']:.2f}")
check("W17.2 Tobit σ = 1122.02, log L = −3819.09",
      close(tb["sigma"], 1122.02, 0.05) and close(tb["llf"], -3819.09, 0.01),
      f"got σ = {tb['sigma']:.2f}, log L = {tb['llf']:.2f}")

# Example 17.3 — Poisson regression for number of arrests (CRIME1): pcnv −.402, black .661
c = woo.data("crime1")
pois = smf.poisson("narr86 ~ pcnv + avgsen + tottime + ptime86 + qemp86 + inc86 + black + hispan + born60",
                   c).fit(disp=0)
check("W17.3 Poisson pcnv = −.402, black = .661",
      close(pois.params["pcnv"], -0.402, 0.0006) and close(pois.params["black"], 0.661, 0.0006),
      f"got {pois.params['pcnv']:.3f}, {pois.params['black']:.3f}")

# ---------------------------------------------------------------------------------------------
# 2. ols_diagnostics: autocorrelation tests only for time series; ARCH-LM for time series
# ---------------------------------------------------------------------------------------------
tests_cs = set(tab["Test"].str.split(" ").str[0])
check("cross-section: no Durbin-Watson / Breusch-Godfrey", not ({"Durbin-Watson", "Breusch-Godfrey"} & tests_cs),
      f"tests: {sorted(tests_cs)}")

# Sorting a cross-section by a regressor must not change the diagnostic battery's conclusions
r_sorted = smf.ols("price ~ lotsize + sqrft + bdrms", h.sort_values("sqrft")).fit()
tab_sorted = d.ols_diagnostics(r_sorted, show=False)
check("cross-section: results invariant to row order",
      np.allclose(tab["Statistic"].values, tab_sorted["Statistic"].values, equal_nan=True))

rng = np.random.default_rng(7)
n = 1500
x = rng.normal(size=n)
e = np.zeros(n)
for t in range(1, n):  # ARCH(1): σ²_t = 0.2 + 0.7 e²_{t−1}
    e[t] = rng.normal() * np.sqrt(0.2 + 0.7 * e[t - 1] ** 2)
ts = pd.DataFrame({"y": 1 + 0.5 * x + e, "x": x}, index=pd.date_range("2000-01-01", periods=n, freq="D"))
tab_ts = d.ols_diagnostics(smf.ols("y ~ x", ts).fit(), time_series=True, show=False)
names_ts = " ".join(tab_ts["Test"])
check("time series: Durbin-Watson and Breusch-Godfrey present",
      "Durbin-Watson" in names_ts and "Breusch-Godfrey" in names_ts)
arch_row = tab_ts[tab_ts["Test"].str.startswith("ARCH-LM")]
check("time series: ARCH-LM detects ARCH(1) errors", len(arch_row) == 1 and arch_row["p-value"].iloc[0] < 0.01,
      f"p = {arch_row['p-value'].iloc[0]:.2e}" if len(arch_row) else "row missing")

ts_iid = pd.DataFrame({"y": 1 + 0.5 * x + rng.normal(size=n), "x": x}, index=ts.index)
arch_iid = d.ols_diagnostics(smf.ols("y ~ x", ts_iid).fit(), time_series=True, show=False)
arch_iid = arch_iid[arch_iid["Test"].str.startswith("ARCH-LM")]
check("time series: ARCH-LM silent under i.i.d. errors", len(arch_iid) == 1 and arch_iid["p-value"].iloc[0] > 0.01,
      f"p = {arch_iid['p-value'].iloc[0]:.3f}" if len(arch_iid) else "row missing")

# ---------------------------------------------------------------------------------------------
# 3. Tobit: known truth with two limits, and marginal effects vs finite differences
# ---------------------------------------------------------------------------------------------
rng = np.random.default_rng(11)
n = 4000
X2 = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.uniform(0, 2, size=n)})
ystar = 2.0 + 1.5 * X2["x1"] - 1.0 * X2["x2"] + rng.normal(scale=2.0, size=n)
y2 = ystar.clip(0, 4).rename("y")
t2 = d.tobit(y2, X2, lower_limit=0.0, upper_limit=4.0, show=False)
b2 = t2["table"]["latent_coef"]
check("two-limit Tobit recovers β and σ",
      close(b2["x1"], 1.5, 0.1) and close(b2["x2"], -1.0, 0.12) and close(t2["sigma"], 2.0, 0.1),
      f"x1 {b2['x1']:.3f}, x2 {b2['x2']:.3f}, σ {t2['sigma']:.3f}")
check("two-limit Tobit counts both censoring sides",
      t2["n_censored_lower"] == int((ystar <= 0).sum()) and t2["n_censored_upper"] == int((ystar >= 4).sum()))

ub = d.tobit(ystar.clip(upper=4).rename("y"), X2, lower_limit=None, upper_limit=4.0, show=False)  # top-coding only
check("upper-only Tobit runs and recovers x1", close(ub["table"].loc["x1", "latent_coef"], 1.5, 0.1),
      f"x1 {ub['table'].loc['x1', 'latent_coef']:.3f}")


def expectations(beta, sigma, XX, a, b):
    """E(y|x), E(y|a<y<b,x), P(a<y<b|x) for a two-limit Tobit."""
    xb = XX @ beta
    za = (a - xb) / sigma if a is not None else np.full_like(xb, -np.inf)
    zb = (b - xb) / sigma if b is not None else np.full_like(xb, np.inf)
    Pa, Pb = stats.norm.cdf(za), stats.norm.cdf(zb)
    pa, pb = stats.norm.pdf(za), stats.norm.pdf(zb)
    p_in = Pb - Pa
    e_in = xb + sigma * (pa - pb) / p_in
    e_all = (a if a is not None else 0) * Pa + p_in * e_in + (b * (1 - Pb) if b is not None else 0)
    return e_all, e_in, p_in


def finite_difference_ames(res, X, a, b, col, h=1e-4):
    beta = res["table"]["latent_coef"].values
    XX = np.column_stack([np.ones(len(X)), X.values])
    j = list(X.columns).index(col) + 1
    hi, lo = XX.copy(), XX.copy()
    hi[:, j] += h
    lo[:, j] -= h
    up, dn = expectations(beta, res["sigma"], hi, a, b), expectations(beta, res["sigma"], lo, a, b)
    return [float(np.mean((u - v) / (2 * h))) for u, v in zip(up, dn)]


for label, res, X, a, b in [("W17.2 (lower only)", tb, Xt, 0.0, None), ("two-limit", t2, X2, 0.0, 4.0)]:
    col = "educ" if "educ" in X.columns else "x1"
    fd = finite_difference_ames(res, X, a, b, col)
    an = res["marginal_effects"].loc[col, ["E[y|x]", "E[y|uncensored,x]", "P(uncensored|x)"]].values
    check(f"Tobit {label}: analytic AMEs = finite differences",
          np.allclose(an, fd, rtol=1e-4, atol=1e-7), f"analytic {np.round(an, 5)}, numeric {np.round(fd, 5)}")
    se = res["marginal_effects"].loc[col, ["se E[y|x]", "se E[y|uncensored,x]", "se P(uncensored|x)"]].values
    check(f"Tobit {label}: delta-method SEs finite and positive", np.all(np.isfinite(se)) and np.all(se > 0))

check("Tobit keeps the v1.3 column (backward compatible)",
      np.allclose(tb["table"]["marginal_effect_E[y|x]"].drop("const"), tb["marginal_effects"]["E[y|x]"]))

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
