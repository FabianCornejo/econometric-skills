"""Reference for survival.cox's robust and clustered SEs (v2.6.7), from lifelines (MIT licence).

lifelines' CoxPHFitter(robust=True) computes the Lin-Wei sandwich; with cluster_col it sums the score residuals
by cluster. The designs have continuous times (no ties), so Efron and Breslow coincide. Design "early" has rows
censored before the first event: statsmodels' own robust covariance is NaN there (the defect fixed in v2.6.7).

Run once (not part of the test suite, lifelines is not a dependency of the skill):
    python tools/py_reference/cox_robust_v267_lifelines.py
It writes tests/data/cox_robust_lifelines_v267.json, which tests/test_show_text.py reads.
"""
import json
import sys
from pathlib import Path

import lifelines
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

ROOT = Path(__file__).resolve().parents[2]


def design(seed, n=400):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    w = rng.normal(size=n)
    T = rng.exponential(1 / np.exp(0.5 * x - 0.3 * w))
    C = rng.exponential(2, size=n)
    return pd.DataFrame({"t": np.minimum(T, C), "e": (T <= C).astype(int), "x": x, "w": w,
                         "g": rng.integers(0, 40, n)})


def n_early(d):
    return int(((d["e"] == 0) & (d["t"] < d.loc[d["e"] == 1, "t"].min())).sum())


# the first seed with at least 2 rows censored before the first event, and the first with none
seed_early = next(s for s in range(1000) if n_early(design(s)) >= 2)
seed_none = next(s for s in range(1000) if n_early(design(s)) == 0)
out = {"lifelines": lifelines.__version__, "python": sys.version.split()[0], "designs": {}}
for name, seed in (("early", seed_early), ("none", seed_none)):
    d = design(seed)
    early = n_early(d)
    for kind in ("robust", "cluster"):
        f = CoxPHFitter()
        if kind == "robust":
            f.fit(d[["t", "e", "x", "w"]], "t", "e", robust=True)
        else:
            f.fit(d[["t", "e", "x", "w", "g"]], "t", "e", robust=True, cluster_col="g")
        out["designs"][f"{name}_{kind}"] = {"seed": seed, "censored_before_first_event": early,
                                            "coef": f.params_.to_dict(), "se": f.standard_errors_.to_dict()}
p = ROOT / "tests" / "data" / "cox_robust_lifelines_v267.json"
p.write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
