"""Builds tests/data/ovb_v26.csv, the simulated data that tools/r_reference/ovb_bounds_v26_reference.R reads.

A continuous treatment d with a negative true effect (-0.5), confounded by an observed strong covariate (x_strong),
a weak one (x_weak), a four-level categorical (g, entered as dummies: the grouped benchmark) and an unobserved u
that is left out of every regression on purpose. The R script runs sensemakr::ovb_bounds on
    y ~ d + x_strong + x_weak + g
with benchmarks x_strong, x_weak and the group of g's dummies. Run from the repository root:
    python tools/r_reference/make_ovb_v26_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(20261110)
n = 800
g = rng.choice(np.array(["a", "b", "c", "d"]), size=n)
g_eff = pd.Series(g).map({"a": 0.0, "b": 0.4, "c": -0.3, "d": 0.8}).to_numpy()
x_strong = rng.standard_normal(n)
x_weak = rng.standard_normal(n)
u = rng.standard_normal(n)
d = 0.35 * x_strong + 0.1 * x_weak + 0.5 * g_eff + 0.4 * u + rng.standard_normal(n)
y = -0.5 * d + 0.3 * x_strong + 0.15 * x_weak + 0.7 * g_eff + 0.5 * u + rng.standard_normal(n)
out = Path("tests/data/ovb_v26.csv")
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"y": y, "d": d, "x_strong": x_strong, "x_weak": x_weak, "g": g}).round(10).to_csv(
    out, index=False, lineterminator="\n")
print(f"wrote {out} ({n} rows)")
