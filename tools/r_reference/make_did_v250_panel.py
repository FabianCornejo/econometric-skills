"""Build the fixed panel used to replicate the v2.5.0 DiD functions against R (didimputation, TwoWayFEWeights).

The CSV is committed (tests/data/did_v250_panel.csv), so the R script and tests/test_v250.py read the same numbers;
re-running this script must reproduce it byte for byte (the seed is fixed).

Design: 60 units, periods 1-10, four groups of 15 (never treated, adoption at 4, 6 and 8), unit and period effects,
and an effect that grows with exposure and differs by cohort, tau = (1 + 0.2 (t - g)) (1 + 0.5 1{g = 4}): the case in
which TWFE is biased and the imputation and Callaway-Sant'Anna estimators are not.
Columns:
    id, t, y        unit, period, outcome
    g               first treated period (0 = never treated)
    d               binary treatment 1{g > 0 and t >= g}
    dose            d times a unit intensity in {0.5, 1, 2} (non-binary treatment for TwoWayFEWeights)
    state           cluster (12 states of 5 consecutive units; all units of a state share its adoption date)
    keep_unb        1 for the rows of the unbalanced version (a fixed set of 23 unit-periods removed)

Run from the repository root:  python tools/r_reference/make_did_v250_panel.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests" / "data" / "did_v250_panel.csv"


def build(seed: int = 250) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n, T = 60, 10
    g = np.repeat([0, 4, 6, 8], 15)
    intensity = np.tile([0.5, 1.0, 2.0], 20)
    state = np.arange(n) // 5
    alpha_i = rng.normal(0, 1, n)
    lam_t = np.cumsum(rng.normal(0.1, 0.3, T))
    rows = []
    for i in range(n):
        for k, t in enumerate(range(1, T + 1)):
            d = int(g[i] > 0 and t >= g[i])
            tau = (1 + 0.2 * (t - g[i])) * (1 + 0.5 * (g[i] == 4)) if d else 0.0
            y = alpha_i[i] + lam_t[k] + tau + rng.normal(0, 0.5)
            rows.append((i + 1, t, round(float(y), 6), int(g[i]), d, round(d * intensity[i], 2), int(state[i]) + 1))
    df = pd.DataFrame(rows, columns=["id", "t", "y", "g", "d", "dose", "state"])
    drop = rng.choice(len(df), 23, replace=False)
    df["keep_unb"] = 1
    df.loc[drop, "keep_unb"] = 0
    return df


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    build().to_csv(OUT, index=False, lineterminator="\n")
    print(f"wrote {OUT}")
