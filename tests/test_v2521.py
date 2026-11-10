"""Tests for v2.5.2.1: the q label of specification.sensitivity.

Bug (v2.2.1–v2.5.2): inside the loop over joint benchmarks, `q = len(terms)` overwrote the parameter `q`,
so with any categorical covariate the `show` block printed RV(q=<number of dummies>) instead of the q the
statistics were computed with. The numbers were right (robustness_value runs before the loop). The second
line also said "would remove the estimate" for every q, which is true only for q = 1.

What is checked here:
  1. the printed q is the q requested and the one in out["q"], with a 6-level categorical in the model;
  2. the joint benchmark keeps its label and equals the partial R² from refitting without the dummies;
  3. no number changed: references fixed from the code at 8df4e07 (v2.5.2), three designs, three q's;
  4. the q = 1 text is byte-identical to v2.5.2 when there is no categorical (the case the bug never hit);
  5. the q ≠ 1 sentence states what Cinelli and Hazlett (2020) define, verified with the bias bound written
     here from the paper, not from the implementation: a confounder with R²_yz = R²_dz = RV_q moves the
     estimate to (1 − q)·τ̂; RV_q,α is the smallest max(R²_yz, R²_dz) whose adjusted CI reaches (1 − q)·τ̂,
     found on a 2-D grid (it does not assume monotonicity: near R²_yz = 1 the adjusted SE collapses),
     including the large-effect branch.

Mutation (done when written): against specification.py of 8df4e07 the label checks of (1) and the sentence
checks of (5) fail; (2), (3), (4) and the definition checks of (5) pass.

Run from the repository root:  python3 tests/test_v2521.py
Needs: statsmodels, scipy.
"""
import contextlib
import io
import re
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "econometrics" / "scripts"))
import _console  # noqa: E402,F401  (Windows-safe console output, see econometrics/scripts/_console.py)
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

import specification as sp

ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


def data(seed, n, levels):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n),
                       "setor": rng.choice(list("ABCDEFGH")[:levels], n)})
    eff = dict(zip("ABCDEFGH", [0, .2, .4, .1, -.3, .5, .25, -.15]))
    df["d"] = 0.5 * df.x1 + 0.2 * df.setor.map(eff) + rng.normal(size=n)
    df["y"] = 0.3 * df.d + 0.4 * df.x1 - 0.2 * df.x2 + df.setor.map(eff) + rng.normal(size=n)
    return df


def printed(res, q, alpha=0.05):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = sp.sensitivity(res, "d", q=q, alpha=alpha, show=True)
    return out, buf.getvalue()


dA = data(2521, 400, 6)
fits = {"A": smf.ols("y ~ d + x1 + x2 + C(setor)", dA).fit(),          # 6 levels -> 5 dummies
        "B": smf.ols("y ~ d + x1 + C(setor)", data(7, 250, 4)).fit(),   # 4 levels -> 3 dummies
        "C": smf.ols("y ~ d + x1 + x2", data(11, 300, 2)).fit()}        # no categorical

# --- 1. the label shows the q requested -------------------------------------------------------------
for q in (1.0, 2.0, 0.5):
    out, txt = printed(fits["A"], q)
    labels = re.findall(r"RV\(q=([^,)]+)", txt)
    check(f"label: q={q} with a 6-level categorical prints RV(q={q}) twice and nothing else",
          labels == [str(q), str(q)], str(labels))
    check(f"label: q={q} never prints the number of dummies (RV(q=5)", "RV(q=5" not in txt)
    check(f"label: q={q} matches out['q']", out["q"] == q and all(float(v) == out["q"] for v in labels))
_, txt_b = printed(fits["B"], 2.0)
check("label: q=2 with a 4-level categorical prints RV(q=2.0), not RV(q=3",
      "RV(q=2.0)" in txt_b and "RV(q=3" not in txt_b)

# --- 2. joint benchmark: label kept, value by refitting ----------------------------------------------
bA = sp.sensitivity(fits["A"], "d")["benchmarks"]
key = "C(setor) (5 dummies, jointly)"
restricted = smf.ols("y ~ d + x1 + x2", dA).fit()
pr2 = 1 - fits["A"].ssr / restricted.ssr                    # partial R² of the 5 dummies, by SSR
check("benchmark: joint label still names 5 dummies", key in bA.index, str(list(bA.index)))
check("benchmark: joint partial R² equals the SSR refit", key in bA.index and abs(bA[key] - pr2) < 1e-10,
      f"{bA.get(key, np.nan):.10f} vs {pr2:.10f}")

# --- 3. no number changed (references from specification.py at 8df4e07, v2.5.2) ---------------------
REF = {  # (partial_r2, rv_q, rv_qa), benchmarks
    ("A", 1.0): (0.04282780639613858, 0.19033574715441084, 0.10587885717181099),
    ("A", 2.0): (0.04282780639613858, 0.34292888173951563, 0.2753787324162799),
    ("A", 0.5): (0.04282780639613858, 0.1003187731057977, 0.006189173490535478),
    ("B", 1.0): (0.09814596825486621, 0.2799332493093697, 0.18386741880778085),
    ("B", 2.0): (0.09814596825486621, 0.47709849602373894, 0.40979633471886956),
    ("B", 0.5): (0.09814596825486621, 0.1519012418355443, 0.03784635219627461),
    ("C", 1.0): (0.07419023132912228, 0.24583595182568607, 0.15489959941190484),
    ("C", 2.0): (0.07419023132912228, 0.42814110039739395, 0.36098612207419845),
    ("C", 0.5): (0.07419023132912228, 0.1318781457451302, 0.026596561406382047),
}
BENCH = {"A": {"x1": 0.13538046261785916, "C(setor) (5 dummies, jointly)": 0.07608067924718526,
               "x2": 0.03979904605426103},
         "B": {"x1": 0.09939193694581919, "C(setor) (3 dummies, jointly)": 0.011912946920026592},
         "C": {"x1": 0.0562584369261155, "x2": 0.04543623145561481}}
for (name, q), ref in REF.items():
    o = sp.sensitivity(fits[name], "d", q=q)
    got = (o["partial_r2"], o["rv_q"], o["rv_qa"])
    same_b = (set(o["benchmarks"].index) == set(BENCH[name])
              and all(abs(o["benchmarks"][k] - v) < 1e-9 for k, v in BENCH[name].items()))
    check(f"numbers unchanged vs v2.5.2: design {name}, q={q}",
          all(abs(g - r) < 1e-9 for g, r in zip(got, ref)) and same_b and o["q"] == q,
          f"{got} vs {ref}")

# --- 4. q = 1 without a categorical: text byte-identical to v2.5.2 -----------------------------------
OUT_C_Q1 = ("\nSensitivity of d (Cinelli-Hazlett): partial R² = 7.419%, RV(q=1.0) = 24.58%, RV(q=1.0, α=0.05) = 15.49%\n"
            "A confounder explaining 24.6% of the residual variance of both d and the outcome would remove the "
            "estimate; 15.5% would make it insignificant.\nObserved covariates (partial R² with outcome), for "
            "comparison:\nx1   5.63%\nx2   4.54%\n")
_, txt_c = printed(fits["C"], 1.0)
check("q=1, no categorical: output byte-identical to v2.5.2", txt_c == OUT_C_Q1, repr(txt_c))
_, txt_a1 = printed(fits["A"], 1.0)
check("q=1 keeps the 'remove the estimate … insignificant' sentence",
      "would remove the estimate" in txt_a1 and "would make it insignificant" in txt_a1)

# --- 5. q ≠ 1 sentence, and the definition it states ------------------------------------------------
for q, pct, to in ((2.0, "200%", "-1"), (0.5, "50%", "0.5")):
    o, txt = printed(fits["A"], q)
    check(f"sentence q={q}: says 'reduce the estimate by {pct} (to {to} times its value)'",
          f"would reduce the estimate by {pct} (to {to} times its value)" in txt, txt.splitlines()[2])
    check(f"sentence q={q}: puts the reduced value inside the 95% CI, not 'remove'/'insignificant'",
          "inside the 95% confidence interval" in txt and "would remove the estimate" not in txt
          and "insignificant" not in txt)
_, txt90 = printed(fits["A"], 2.0, alpha=0.10)
check("sentence: the CI level follows alpha (α=0.10 -> 90%)", "inside the 90% confidence interval" in txt90)


def adjusted(tau, se, dof, r2yz, r2dz):
    """Cinelli and Hazlett (2020, sec. 4): bias bound and adjusted SE of a confounder with partial R²s
    (r2yz with the outcome, r2dz with the treatment). Written from the paper, not from specification.py."""
    bias = se * np.sqrt(dof * r2yz * r2dz / (1 - r2dz))
    se_adj = se * np.sqrt((1 - r2yz) / (1 - r2dz) * dof / (dof - 1))
    return tau - bias, se_adj


def minimax_rv_qa(tau, se, dof, q, alpha, n=1500):
    """Smallest max(R²_yz, R²_dz) on a grid whose adjusted lower CI bound reaches (1 − q)·τ."""
    g = np.linspace(0, 1 - 1e-9, n)
    a, b = np.meshgrid(g, g, indexing="ij")
    est, se_adj = adjusted(tau, se, dof, a, b)
    reach = est - stats.t.ppf(1 - alpha / 2, dof - 1) * se_adj <= (1 - q) * tau
    return float(np.maximum(a, b)[reach].min()) if reach.any() else np.nan


def on_grid(grid_min, rv, n):
    """The grid minimum can never be below the true minimum and is at most one grid step above it."""
    step = (1 - 1e-9) / (n - 1)
    return rv - 1e-12 <= grid_min <= rv + 1.5 * step


resA = fits["A"]
tau, se, dof = float(resA.params["d"]), float(resA.bse["d"]), float(resA.df_resid)
for q in (0.5, 1.0, 2.0):
    o = sp.sensitivity(resA, "d", q=q)
    est, _ = adjusted(tau, se, dof, o["rv_q"], o["rv_q"])
    check(f"definition: R²_yz = R²_dz = RV_q moves the estimate to (1 − q)·τ̂ (q={q})",
          abs(est - (1 - q) * tau) < 1e-9 * abs(tau), f"{est / tau:+.10f} vs {1 - q:+.2f}")
    m = minimax_rv_qa(tau, se, dof, q, 0.05)
    check(f"definition: RV_q,α is the smallest strength whose CI reaches (1 − q)·τ̂ (q={q}, grid 1/1500)",
          on_grid(m, o["rv_qa"], 1500), f"grid {m:.5f} vs {o['rv_qa']:.5f}")
for t_big, dof_big, q in ((30.0, 50, 1.0), (40.0, 30, 2.0)):        # large-effect branch of RV_q,α
    r = sp.robustness_value(t_big, dof_big, q=q)
    fq = q * t_big / np.sqrt(dof_big)
    f_crit = stats.t.ppf(0.975, dof_big - 1) / np.sqrt(dof_big - 1)
    m = minimax_rv_qa(t_big, 1.0, dof_big, q, 0.05, n=3000)
    check(f"definition, large-effect branch (t={t_big}, df={dof_big}, q={q}): grid minimum = RV_q,α",
          fq > 1 / f_crit and on_grid(m, r["rv_qa"], 3000), f"grid {m:.5f} vs {r['rv_qa']:.5f}")

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
