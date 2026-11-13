"""Data for the R reference of survival.cox's robust SEs with tied event times (v2.6.7).

n = 400, seed 3: x and w with log hazard ratios 0.5 and -0.3, exponential censoring; `t` is the observed time
rounded UP to a grid of 1/4 (heavy ties: about 15 distinct event times), `tc` the same time without rounding (no
ties); `s` three strata, `g` 40 clusters, `id` the row. Run from the repository root:
    python tools/r_reference/make_cox_v267_data.py
It writes tests/data/cox_v267_ties.csv (and prints its MD5).
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
g = np.random.default_rng(3)
n = 400
x, w = g.normal(size=n), g.normal(size=n)
T = g.exponential(1 / np.exp(0.5 * x - 0.3 * w))
C = g.exponential(2, size=n)
tc = np.minimum(T, C)
d = pd.DataFrame({"id": np.arange(1, n + 1), "t": np.ceil(tc * 4) / 4, "tc": tc.round(10), "e": (T <= C).astype(int),
                  "x": x.round(10), "w": w.round(10), "s": g.integers(0, 3, n), "g": g.integers(0, 40, n)})
p = ROOT / "tests" / "data" / "cox_v267_ties.csv"
d.to_csv(p, index=False, lineterminator="\n")
print(p, hashlib.md5(p.read_bytes()).hexdigest(), "events", int(d["e"].sum()), "distinct event times",
      d.loc[d["e"] == 1, "t"].nunique())
