"""v2.7.5: replications on published data, from the QuantEcon lectures (rule T4 exception: one file for one data source,
like test_textbook_replication.py).

Data: QuantEcon/data-lectures at a fixed commit, downloaded by the test and checked by SHA-256; the plugin does not
redistribute the files (none has a named licence). Specifications and the published numbers: QuantEcon/
lecture-python.myst at a fixed commit, lectures/ols.md (Acemoglu, Johnson and Robinson 2001, AER 91(5)) and
lectures/mle.md (Treisman 2016, AER 106(5), Table 1, Poisson with HC0 errors). The numbers checked are the lectures'
(statsmodels and linearmodels on the same files), not the papers' printed tables.

Without network, or with a file whose SHA-256 differs: SKIP with exit code 0 on a local run; FAIL on GitHub Actions
(GITHUB_ACTIONS=true), so the test cannot die silently in CI.
"""
import hashlib
import io
import os
import sys
import tempfile
import time
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.special import gammaln

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "econometrics" / "scripts"))
import _console  # noqa: E402,F401  (Windows-safe console output, see econometrics/scripts/_console.py)
import diagnostics as dg  # noqa: E402
import estimation as es  # noqa: E402

DATA_COMMIT = "43f32e0e00c6f1a384ae3210e5836ee0c244ae99"      # QuantEcon/data-lectures, 2026-11-06
LECTURES_COMMIT = "0dc4fe74070fb7437da2c2761740b807f76f2f18"  # QuantEcon/lecture-python.myst (ols.md, mle.md)
URL = f"https://raw.githubusercontent.com/QuantEcon/data-lectures/{DATA_COMMIT}/lectures/{{}}"
FILES = {  # SHA-256 equal to the data-lectures manifests (lectures/<file>.yml, integrity.sha256)
    "maketable1.dta": "7a8687cc96c505a46dd24e98ab8b2332b31495003d68d476f9b280fcf002863b",
    "maketable4.dta": "2b34cce21b58a3a424a804e97a134106ab7010ef936bb68b3a244d3e271af404",
    "fp.dta": "de99b9c0de6fb08cb732e306971ba87a308c5c4ef8ca3ae23fbbb50eabdecd8d",
}
IN_CI = os.environ.get("GITHUB_ACTIONS", "").lower() == "true"
ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info)


def unavailable(reason):
    if IN_CI:
        print("FAIL data unavailable on GitHub Actions:", reason)
        sys.exit(1)
    print("SKIP test_quantecon_replication:", reason)
    sys.exit(0)


def fetch(name):
    """The file's path in a cache folder (ECONOMETRICS_TEST_DATA, or the temporary folder), downloaded once; with a
    read-only cache, the verified bytes in memory."""
    folder = Path(os.environ.get("ECONOMETRICS_TEST_DATA") or Path(tempfile.gettempdir()) / "econometrics-test-data")
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError:
        folder = Path(tempfile.mkdtemp(prefix="econometrics-test-data-"))  # the folder given cannot be created
    path = folder / f"{FILES[name][:16]}-{name}"
    in_memory = False
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != FILES[name]:
        try:
            path.unlink()  # a damaged or edited cached copy: download it again
        except OSError:
            in_memory = True  # it cannot be deleted (read-only cache): download into memory (review r2 S3-1)
    if in_memory or not path.exists():
        blob, err = None, None
        for attempt in range(2):  # one retry after a short wait: a transient network error should not fail CI
            try:
                with urllib.request.urlopen(URL.format(name), timeout=60) as r:
                    blob = r.read()
                break
            except Exception as e:  # noqa: BLE001  (no network, proxy, 404: all mean the data are not available)
                err = e
                if attempt == 0:
                    time.sleep(5)
        if blob is None:
            unavailable(f"{name}: {type(err).__name__}: {err}")
        digest = hashlib.sha256(blob).hexdigest()
        if digest != FILES[name]:
            unavailable(f"{name}: SHA-256 {digest} differs from the fixed one")
        if in_memory:
            return io.BytesIO(blob)
        try:
            tmp = path.with_suffix(".part")
            tmp.write_bytes(blob)
            tmp.replace(path)
        except OSError:
            return io.BytesIO(blob)  # a read-only cache: use the verified bytes in memory (review r2 S3-2)
    return path


def near(a, b, tol):
    return abs(float(a) - float(b)) <= tol


H = 0.00005  # half the last printed digit of the lectures' four-decimal numbers

# ------------------------------------------------------------------ AJR 2001: the data (ols.md, "Simple regression")
df1 = pd.read_stata(fetch("maketable1.dta"))
d1 = df1.dropna(subset=["logpgp95", "avexpr"]).assign(const=1.0)
r1 = sm.OLS(d1["logpgp95"], d1[["const", "avexpr"]]).fit()
check("AJR data: OLS of logpgp95 on avexpr is 0.5319 (0.0406), n 111 (ols.md)",
      near(r1.params["avexpr"], 0.5319, H) and near(r1.bse["avexpr"], 0.0406, H) and int(r1.nobs) == 111,
      (r1.params["avexpr"], r1.bse["avexpr"], r1.nobs))

# case (a) of claude/plano-v2.7.5.md: the lecture's text says 8.38 at "the average 7.07", its code predicts at the
# mean of the base sample (64 rows) with the regression on 111 rows: 8.0916. Mixed samples, two different numbers.
m_all = df1["avexpr"].mean()  # 7.0665: the 121 rows with avexpr, not the 111 of the regression (7.2068)
m_base = d1.loc[d1["baseco"] == 1, "avexpr"].mean()
p_base = float(r1.predict([[1.0, m_base]])[0])
check("AJR case (a): the lecture's .predict is 8.0916 at the base-sample mean 6.5156, not 8.38 at 7.07",
      near(p_base, 8.0916, H) and near(m_base, 6.5156, H) and abs(float(r1.predict([[1.0, m_all]])[0]) - 8.38) < 0.01,
      (p_base, m_base, m_all))

# ------------------------------------------------------------------ AJR 2001: 2SLS with iv_report (ols.md, "Endogeneity")
df4a = pd.read_stata(fetch("maketable4.dta"))
df4 = df4a[df4a["baseco"] == 1].copy()
rep = dg.iv_report("logpgp95 ~ 1 + [avexpr ~ logem4]", df4, cov_type="unadjusted", show=False)
t, fs = rep["table"].loc["avexpr"], rep["first_stage"].iloc[0]
check("AJR iv_report unadjusted: 2SLS 0.9443 (0.1541), n 64 (ols.md, IV2SLS unadjusted)",
      near(t["2SLS"], 0.9443, H) and near(t["2SLS se"], 0.1541, H) and rep["n"] == 64, (t["2SLS"], t["2SLS se"], rep["n"]))

# the first stage by another route: OLS of avexpr on logem4 (statsmodels), -0.6068 (0.1267), F 22.95
d4 = df4.dropna(subset=["logpgp95", "avexpr", "logem4"]).assign(const=1.0)
f1 = sm.OLS(d4["avexpr"], d4[["const", "logem4"]]).fit()
check("AJR first stage: logem4 -0.6068 (0.1267); iv_report's partial F = statsmodels' F 22.9468; 'acceptable'",
      near(f1.params["logem4"], -0.6068, H) and near(f1.bse["logem4"], 0.1267, H)
      and abs(fs["partial F"] - f1.fvalue) < 1e-8 * f1.fvalue and near(f1.fvalue, 22.9468, H)
      and str(fs["strength"]).startswith("acceptable"), (f1.params["logem4"], fs["partial F"], f1.fvalue, fs["strength"]))

# the OLS column of iv_report is the OLS on the 2SLS rows (v2.6.12), not the 111-row 0.5319
o64 = sm.OLS(d4["logpgp95"], d4[["const", "avexpr"]]).fit()
# with cov_type="unadjusted" its SE uses SSR/n, like linearmodels' unadjusted (statsmodels: SSR/(n - k))
check("AJR iv_report's OLS is on the same 64 rows: 0.5221 = statsmodels, SE = statsmodels' times sqrt((n-k)/n); not 0.5319",
      abs(t["OLS"] - o64.params["avexpr"]) < 1e-9
      and abs(t["OLS se"] - o64.bse["avexpr"] * np.sqrt((o64.nobs - 2) / o64.nobs)) < 1e-9
      and abs(t["OLS"] - 0.5319) > 0.005, (t["OLS"], t["OLS se"], o64.params["avexpr"], o64.bse["avexpr"]))

# trap (d): the second stage run by hand with OLS gives the 2SLS point estimate with a wrong SE (0.1255, not 0.1541)
hand = sm.OLS(d4["logpgp95"], d4[["const"]].assign(pred=f1.fittedvalues)).fit()
check("AJR trap (d): 2SLS by hand = 0.9443 with SE 0.1255, smaller than iv_report's 0.1541",
      abs(hand.params["pred"] - t["2SLS"]) < 1e-9 and near(hand.bse["pred"], 0.1255, H) and t["2SLS se"] > hand.bse["pred"] + 0.02,
      (hand.params["pred"], hand.bse["pred"]))

# 2SLS by closed formulas written here (review r1 S2-1: linearmodels, which iv_report calls, is only an extra check):
# b = (Xh'X)^-1 Xh'y with Xh the first-stage fitted [1, x]; unadjusted variance (SSR/n)·(Xh'Xh)^-1 with the residuals
# of the original x; HC0 (Xh'Xh)^-1 Xh' diag(u²) Xh (Xh'Xh)^-1.
Xo = d4[["const", "avexpr"]].to_numpy(float)
Zm = d4[["const", "logem4"]].to_numpy(float)
Xh = Zm @ np.linalg.lstsq(Zm, Xo, rcond=None)[0]
yo = d4["logpgp95"].to_numpy(float)
bh = np.linalg.solve(Xh.T @ Xo, Xh.T @ yo)
uh = yo - Xo @ bh
A = np.linalg.inv(Xh.T @ Xh)
se_un = np.sqrt((uh @ uh / len(yo)) * A[1, 1])
se_h0 = np.sqrt((A @ (Xh.T * uh ** 2) @ Xh @ A)[1, 1])
check("AJR 2SLS closed form: iv_report's estimate and unadjusted SE (SSR/n) equal it",
      abs(t["2SLS"] - bh[1]) < 1e-10 and abs(t["2SLS se"] - se_un) < 1e-10 * se_un, (t["2SLS"], bh[1], t["2SLS se"], se_un))
rob = dg.iv_report("logpgp95 ~ 1 + [avexpr ~ logem4]", df4, cov_type="robust", show=False)
check("AJR iv_report robust: 2SLS SE 0.1761 = the closed-form HC0 sandwich",
      abs(rob["table"].loc["avexpr", "2SLS se"] - se_h0) < 1e-10 * se_h0 and near(se_h0, 0.1761, H),
      (rob["table"].loc["avexpr", "2SLS se"], se_h0))
# Wu-Hausman (unadjusted) = t² of the first-stage residual added to the OLS of y on [1, x], on the same 64 rows
cf = sm.OLS(d4["logpgp95"], d4[["const", "avexpr"]].assign(v=f1.resid)).fit()
row = rep["tests"].loc[[i for i in rep["tests"].index if "Wu-Hausman" in str(i)][0]]
check("AJR Wu-Hausman: iv_report = t² of the control-function regression (statsmodels; F 24.22, p 7e-6)",
      abs(row["statistic"] - cf.tvalues["v"] ** 2) < 1e-8 * row["statistic"]
      and abs(row["p"] - stats.f.sf(cf.tvalues["v"] ** 2, 1, cf.df_resid)) < 1e-10 and near(row["statistic"], 24.22, 0.005),
      (row.to_dict(), cf.tvalues["v"] ** 2))
try:  # an extra check against the library itself
    from linearmodels.iv import IV2SLS
    lm = IV2SLS(d4["logpgp95"], d4[["const"]], d4["avexpr"], d4["logem4"]).fit(cov_type="unadjusted")
    check("AJR (extra): linearmodels gives the same 2SLS SE and Wu-Hausman",
          abs(lm.std_errors["avexpr"] - se_un) < 1e-9 and abs(lm.wu_hausman().stat - row["statistic"]) < 1e-8 * row["statistic"])
except ImportError:
    check("linearmodels is installed (a core package)", False)

# Anderson-Rubin set against the closed form: with one instrument and homoskedastic errors, F(b) <= c is a quadratic
# inequality in b, written here from the AR regression of y - b*x on [1, z] (Anderson and Rubin 1949).
yv, xv, zv = (d4[c].to_numpy(float) for c in ("logpgp95", "avexpr", "logem4"))
yt, xt, zt = yv - yv.mean(), xv - xv.mean(), zv - zv.mean()
n, dfd = len(yv), len(yv) - 2
c = stats.f.ppf(0.95, 1, dfd)
zz, a_, b_ = zt @ zt, zt @ yt, zt @ xt
# (dfd + c)·(a - b·β)²/zz − c·(yy − 2β·xy + β²·xx) ≤ 0
qa = (dfd + c) * b_ ** 2 / zz - c * (xt @ xt)
qb = -2 * (dfd + c) * a_ * b_ / zz + 2 * c * (xt @ yt)
qc = (dfd + c) * a_ ** 2 / zz - c * (yt @ yt)
lo_x, hi_x = sorted(np.roots([qa, qb, qc]).real)
ar = rep["anderson_rubin"]
step = 40 * t["2SLS se"] / 2000  # the grid: 2001 points over ±20 SEs (iv_report's default)
check("AJR Anderson-Rubin: a bounded interval whose grid ends lie inside the exact set, within one grid step",
      qa > 0 and lo_x <= ar["ci_low"] < lo_x + step and hi_x - step < ar["ci_high"] <= hi_x,
      (ar["ci_low"], ar["ci_high"], lo_x, hi_x, step))
u0 = yt - 0 * xt
P0 = (zt @ u0) ** 2 / zz
F0 = dfd * P0 / (u0 @ u0 - P0)
check("AJR Anderson-Rubin p at beta = 0 = the closed-form F test's p",
      abs(ar["p_at_zero"] - stats.f.sf(F0, 1, dfd)) < 1e-10 * max(1.0, stats.f.sf(F0, 1, dfd)) + 1e-15,
      (ar["p_at_zero"], stats.f.sf(F0, 1, dfd)))

# ------------------------------------------------------------------ AJR: the residual test (ols.md, exercise 1)
# The lecture's first stage uses the 74 rows with avexpr and logem4 (no baseco filter); its second regression, the 70
# that also have logpgp95: resid -0.4951 (0.0987). endogeneity_test uses one sample for both steps (70 rows). The t
# statistic does not depend on the first stage's sample: [1, avexpr, avexpr - a - b·logem4] spans the same columns as
# [1, avexpr, logem4] for any (a, b), so only the coefficient on resid changes, not its t.
fa = df4a.dropna(subset=["avexpr", "logem4"]).assign(const=1.0)
ra = sm.OLS(fa["avexpr"], fa[["const", "logem4"]]).fit()
fa = fa.assign(resid=ra.resid).dropna(subset=["logpgp95"])
ha = sm.OLS(fa["logpgp95"], fa[["const", "avexpr", "resid"]]).fit()
check("AJR residual test as in the lecture: resid -0.4951 (0.0987), first stage on 74 rows, test on 70",
      near(ha.params["resid"], -0.4951, H) and near(ha.bse["resid"], 0.0987, H) and int(ra.nobs) == 74 and int(ha.nobs) == 70,
      (ha.params["resid"], ha.bse["resid"], ra.nobs, ha.nobs))
s70 = df4a.dropna(subset=["logpgp95", "avexpr", "logem4"]).assign(const=1.0)
r70 = sm.OLS(s70["avexpr"], s70[["const", "logem4"]]).fit()
h70 = sm.OLS(s70["logpgp95"], s70[["const", "avexpr"]].assign(resid=r70.resid)).fit()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    et = dg.endogeneity_test(s70["logpgp95"], pd.DataFrame(index=s70.index), s70[["avexpr"]], s70[["logem4"]],
                             cov_type="nonrobust", show=False)
check("AJR endogeneity_test (70 rows) = t² of the residual test with the first stage on the same 70 rows",
      abs(et["statistic"] - h70.tvalues["resid"] ** 2) < 1e-8 * et["statistic"] and len(s70) == 70,
      (et["statistic"], h70.tvalues["resid"] ** 2))
check("AJR: the lecture's t² (first stage on 74 rows) equals endogeneity_test's 25.1697; the resid coefficients differ",
      abs(ha.tvalues["resid"] ** 2 - et["statistic"]) < 1e-8 * et["statistic"] and near(et["statistic"], 25.1697, H)
      and abs(ha.params["resid"] - h70.params["resid"]) > 1e-4, (ha.tvalues["resid"] ** 2, et["statistic"],
                                                                 ha.params["resid"], h70.params["resid"]))

# ------------------------------------------------------------------ Treisman 2016, Table 1 (mle.md): estimation.mle
fp = pd.read_stata(fetch("fp.dta"))
fp = fp[fp["year"] == 2008].assign(const=1.0)
base = ["const", "lngdppc", "lnpop", "gattwto08"]
MODELS = {  # the lecture's numbers: statsmodels Poisson, cov_type HC0, 2008 cross-section
    "Model 1": (base, 197, {"const": (-29.0495, 2.5781), "lngdppc": (1.0839, 0.1383), "lnpop": (1.1714, 0.0974),
                            "gattwto08": (0.0060, 0.0069)}),
    "Model 2": (base + ["lnmcap08", "rintr", "topint08"], 131, None),
    "Model 3": (base + ["lnmcap08", "rintr", "topint08", "nrrents", "roflaw"], 131, None),
}


def loglik(b, y, X):
    eta = X @ b
    return y * eta - np.exp(eta) - gammaln(y + 1)  # the Poisson log-likelihood, written here


def score(b, y, X):
    return (y - np.exp(X @ b))[:, None] * X


for name, (cols, n_exp, printed) in MODELS.items():
    d = fp[["numbil0"] + cols].dropna()
    y, X = d["numbil0"].to_numpy(float), d[cols].to_numpy(float)
    ref = sm.Poisson(d["numbil0"], d[cols]).fit(cov_type="HC0", disp=0, maxiter=200)
    start = np.r_[np.log(y.mean()), np.zeros(len(cols) - 1)]
    m = es.mle(loglik, start, args=(y, X), names=cols, cov_type="sandwich", score=score)
    tab = m["table"]
    est = tab.iloc[:, 0].to_numpy(float)
    se = tab.iloc[:, 1].to_numpy(float)
    rel_b = np.max(np.abs(est - ref.params.to_numpy()) / np.maximum(1.0, np.abs(ref.params.to_numpy())))
    rel_s = np.max(np.abs(se - ref.bse.to_numpy()) / ref.bse.to_numpy())
    check(f"Treisman {name}: estimation.mle (sandwich) = statsmodels Poisson HC0 (1e-5), n {n_exp}",
          rel_b < 1e-5 and rel_s < 1e-5 and len(y) == n_exp, (rel_b, rel_s, len(y)))
    if printed:
        bad = {k: (round(est[i], 4), round(se[i], 4)) for i, k in enumerate(cols)
               if not (near(est[i], printed[k][0], H) and near(se[i], printed[k][1], H))}
        check(f"Treisman {name}: coefficients and HC0 SEs equal the lecture's four decimals", not bad, bad)
    if name == "Model 1":
        cd = dg.count_diagnostics(ref, show=False)
        glm = sm.GLM(d["numbil0"], d[cols], family=sm.families.Poisson()).fit()
        check("Treisman Model 1: count_diagnostics' dispersion = statsmodels GLM Pearson chi²/df (6.527), overdispersed",
              abs(cd["dispersion"] - glm.pearson_chi2 / glm.df_resid) < 1e-8 * cd["dispersion"]
              and near(cd["dispersion"], 6.527, 0.0005) and (cd["tests"]["p"] < 0.05).all(),
              (cd["dispersion"], glm.pearson_chi2 / glm.df_resid, cd["tests"].to_dict()))

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
