"""Tests for data_prep.missing_report and the missing-data guidance in data-collection-wrangling.md.

Known truth by simulation:
- missing_report flags missingness that depends on an observed variable (not MCAR) and stays
  silent under MCAR; complete-case N and shares are exact;
- the guidance that listwise deletion is consistent when missingness depends only on x, and
  biased when it depends on y;
- the statsmodels MICE snippet in the reference runs and recovers the slope under MAR.

Run from the repository root:  python3 tests/test_missing_data.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "econometrics" / "scripts"))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

import data_prep as dp

ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


rng = np.random.default_rng(42)
n = 5000
x = rng.normal(size=n)
z = rng.normal(size=n)
y = 1.0 + 2.0 * x + rng.normal(size=n)
base = pd.DataFrame({"y": y, "x": x, "z": z})

# MCAR in z vs missingness in z driven by x
mcar = base.copy()
mcar.loc[rng.random(n) < 0.3, "z"] = np.nan
mar = base.copy()
mar.loc[rng.random(n) < 1 / (1 + np.exp(-(x - 0.5) * 2)), "z"] = np.nan

r_mcar = dp.missing_report(mcar, ["y", "x", "z"], predictors=["x"], show=False)
r_mar = dp.missing_report(mar, ["y", "x", "z"], predictors=["x"], show=False)
check("complete-case N is exact", r_mcar["complete_n"] == int(mcar["z"].notna().sum()))
check("missing shares are exact", abs(r_mcar["shares"]["z"] - mcar["z"].isna().mean()) < 1e-12)
check("MCAR: missingness not predicted by x", r_mcar["tests"].loc["z", "p"] > 0.05,
      f"p = {r_mcar['tests'].loc['z', 'p']:.3f}")
check("x-driven missingness detected (not MCAR)", r_mar["tests"].loc["z", "p"] < 1e-6,
      f"p = {r_mar['tests'].loc['z', 'p']:.2e}")

# Listwise deletion: consistent if missingness depends on x, biased if it depends on y
slopes_x, slopes_y = [], []
for _ in range(200):
    xx = rng.normal(size=2000)
    yy = 1.0 + 2.0 * xx + rng.normal(size=2000)
    keep_x = rng.random(2000) > 1 / (1 + np.exp(-2 * xx))     # selection on the regressor
    keep_y = rng.random(2000) > 1 / (1 + np.exp(-(yy - 1)))   # selection on the outcome
    slopes_x.append(sm.OLS(yy[keep_x], sm.add_constant(xx[keep_x])).fit().params[1])
    slopes_y.append(sm.OLS(yy[keep_y], sm.add_constant(xx[keep_y])).fit().params[1])
check("listwise deletion consistent when missingness depends on x", abs(np.mean(slopes_x) - 2.0) < 0.02,
      f"mean slope {np.mean(slopes_x):.3f} (truth 2.0)")
check("listwise deletion biased when missingness depends on y", abs(np.mean(slopes_y) - 2.0) > 0.15,
      f"mean slope {np.mean(slopes_y):.3f} (truth 2.0)")

# MICE snippet from the reference: x2 missing at random given y and x1
from statsmodels.imputation.mice import MICE, MICEData  # noqa: E402

m = 1500
x1 = rng.normal(size=m)
x2 = 0.6 * x1 + rng.normal(size=m)
yv = 1 + 1.0 * x1 + 1.5 * x2 + rng.normal(size=m)
dm = pd.DataFrame({"y": yv, "x1": x1, "x2": x2})
dm.loc[rng.random(m) < 1 / (1 + np.exp(-(yv - yv.mean()))), "x2"] = np.nan   # MAR given y
cc = smf.ols("y ~ x1 + x2", dm.dropna()).fit()
mi = MICE("y ~ x1 + x2", sm.OLS, MICEData(dm)).fit(n_burnin=10, n_imputations=20)
b_mi = pd.Series(mi.params, index=["Intercept", "x1", "x2"])
check("MICE snippet runs and recovers x2 slope under MAR", abs(b_mi["x2"] - 1.5) < 0.1,
      f"MI {b_mi['x2']:.3f} vs complete cases {cc.params['x2']:.3f} (truth 1.5)")

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
