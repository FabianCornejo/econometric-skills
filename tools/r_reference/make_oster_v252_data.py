"""Builds tests/data/oster_v252.csv, the simulated data that tools/r_reference/oster_v252_reference.R reads.

Proportional selection as in Oster (2019): an observed control w1 and an unobserved w2, independent, with
cov(w2, x)/var(w2) = δ·cov(w1, x)/var(w1) for δ = 2 and a true effect of x equal to 1. The R script runs
robomit on y ~ x with control w1 (w2 is left out on purpose). Run from the repository root:
    python tools/r_reference/make_oster_v252_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(20261108)
n = 2000
w1 = rng.standard_normal(n)
w2 = rng.standard_normal(n)
x = 0.6 * w1 + 2.0 * 0.6 * w2 + rng.standard_normal(n)
y = 1.0 * x + w1 + w2 + rng.standard_normal(n)
out = Path("tests/data/oster_v252.csv")
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"y": y, "x": x, "w1": w1, "w2": w2}).round(10).to_csv(out, index=False, lineterminator="\n")
print(f"wrote {out} ({n} rows)")
