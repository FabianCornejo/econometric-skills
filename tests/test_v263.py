"""Tests for v2.6.3: the text of the DML benchmark, the sentences of ovb_bounds and the NITs of v2.6.2.

Origin: the v2.6.2 validation on notebook 1 (cf_y 0.0784 below r2_y 0.0948 with 5 sample splits, while the
CHANGELOG, the dml_benchmark docstring and causal-inference.md §4.1 said the old partial R² were "always smaller";
the memo said the result "does not survive" a confounder 1x as strong as region, when its own numbers put the
boundary at k_not_significant = 1.05x) and the v2.6.2 reviews (the "not reached" sentence of ovb_bounds said "a
confounder as strong as x" next to a k of 1.108x; the label "2/5x" of kd = 2, ky = 5 can read as 0.4x).

What is checked, each by a route other than the implementation's:
  1. cf (median of the split gains) against r2 (ratio of medians), with sample splits built by hand so that every
     split's sigma2 and nu2 are known exactly: three splits where cf_y = 0.01 < r2_y = 0.2 and cf_d = 0.01 <
     r2_d = 0.4, and three where cf_y = 0.05 > r2_y = 1/21 (both sides); in every split G > G/(1 + G); one split:
     cf = min(r2/(1 - r2), 1) exactly, on both sides of the cut (G = 1 -+ 1e-9 and G = 1), for Y and for D;
  2. the docs no longer say "always smaller/below" and say that the two are different aggregations;
  3. ovb_bounds "not reached" (design seed 1 of claude/scripts/sonda_v263.py, t = 58): the sentence carries the
     multiplier ("at k = 1.108, a confounder 1.108x as strong as w"), its start is unchanged (tests/test_v261.py), and no printed sentence in
     any call of this file says "a confounder as strong as" without a multiplier;
  4. the kd/ky label: sensemakr's "1/2x w" and "3/0.5x w" unchanged; ovb_bounds prints the "not a ratio" note once
     when some line has kd != ky and never when all have kd = ky (also kd = 2 with ky = 2.0); dml_bounds prints
     "(kd = 2 for the treatment, ky = 5 for the outcome)" on that line only (in full, repr, when kd and ky print
     alike with :g, as kd = 0.3, ky = 0.1 * 3), and every line still starts with its label and ":";
  5. the rule of v2.6.3 in interpretation-rules.md and the evaluation case review-memo-k-threshold-paraphrase.

Run from the repository root:  python3 tests/test_v263.py
Needs: statsmodels, scipy, scikit-learn.
"""
import contextlib
import io
import json
import re
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path.insert(0, str(ROOT / "econometrics" / "scripts"))
import _console  # noqa: E402,F401  (Windows-safe console output, see econometrics/scripts/_console.py)
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression

import causal
import specification as sp

ok = []
TEXTS = []                                        # every printed ovb_bounds output of this file (item 3 sweep)


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


def run(fn, *a, **k):
    """(result, stdout, warning messages); an exception is returned as the result."""
    buf = io.StringIO()
    with warnings.catch_warnings(record=True) as w, contextlib.redirect_stdout(buf):
        warnings.simplefilter("always")
        try:
            out = fn(*a, **k)
        except Exception as e:                    # noqa: BLE001
            out = e
    if fn is sp.ovb_bounds:
        TEXTS.append(buf.getvalue())
    return out, buf.getvalue(), [str(x.message) for x in w]


# ------------------------------------------------------------------ 1. cf against r2: splits with known moments
def split_pair(level_y, gy, level_d, gd, n=500, seed=0):
    """One sample split whose moments are exact: theta = 0 in both fits, sigma2_long = level_y,
    sigma2_short = level_y (1 + gy), nu2_long = level_d, nu2_short = level_d / (1 + gd)."""
    r = np.random.default_rng(seed)
    vd = r.normal(size=n)
    vd = vd - vd.mean()
    uy = r.normal(size=n)
    uy = uy - (np.sum(vd * uy) / np.sum(vd * vd)) * vd          # orthogonal to vd: theta = 0, sigma2 = mean(uy²)
    uy = uy * np.sqrt(level_y / np.mean(uy * uy))
    vd = vd * np.sqrt((1 / level_d) / np.mean(vd * vd))         # nu2 = 1 / mean(vd²) = level_d
    return (uy, vd), (uy * np.sqrt(1 + gy), vd * np.sqrt(1 + gd))


def bench_of(levels_y, gys, levels_d, gds):
    pairs = [split_pair(ly, gy, ld, gd, seed=i) for i, (ly, gy, ld, gd) in enumerate(zip(levels_y, gys, levels_d, gds))]
    k = len(pairs)
    long_ = {"residuals": [p[0] for p in pairs], "thetas": [0.0] * k, "estimate": 0.0, "se": 0.05, "seed": 0,
             "n_folds": 5}
    short_ = {"residuals": [p[1] for p in pairs], "thetas": [0.1] * k, "estimate": 0.1, "se": 0.05, "seed": 0,
              "n_folds": 5}
    orig = causal.dml_plr
    causal.dml_plr = lambda *a, **kw: short_
    try:
        b, _, w = run(causal.dml_benchmark, None, "y", "d", ["x1"], ["x1"], res=long_)
    finally:
        causal.dml_plr = orig
    return b.loc["x1"], w


# below: levels (1, 2, 3); the medians pick different splits in the long and the short fits
b_lo, w_lo = bench_of((1, 2, 3), (1.5, 0.01, 0.01), (1, 2, 3), (0.01, 0.01, 1.5))
# by hand: cf_y = median(min(G, 1)) = 0.01; r2_y = (median(2.5, 2.02, 3.03) - 2) / 2.5 = 0.2
#          cf_d = 0.01; nu2 short = (1/1.01, 2/1.01, 3/2.5): median 1.2; r2_d = (2 - 1.2) / 2 = 0.4
check("3 splits built by hand: cf_y = 0.01 below r2_y = 0.2 (median of split gains vs ratio of medians)",
      abs(b_lo["cf_y"] - 0.01) < 1e-9 and abs(b_lo["r2_y"] - 0.2) < 1e-9, b_lo[["cf_y", "r2_y"]].tolist())
check("3 splits built by hand: cf_d = 0.01 below r2_d = 0.4",
      abs(b_lo["cf_d"] - 0.01) < 1e-9 and abs(b_lo["r2_d"] - 0.4) < 1e-9, b_lo[["cf_d", "r2_d"]].tolist())
b_hi, _ = bench_of((1, 2, 3), (0.05, 0.05, 0.05), (1, 2, 3), (0.05, 0.05, 0.05))
check("3 splits, equal gains 0.05: cf_y = 0.05 above r2_y = 0.05/1.05 (the other side)",
      abs(b_hi["cf_y"] - 0.05) < 1e-9 and abs(b_hi["r2_y"] - 0.05 / 1.05) < 1e-9 and b_hi["cf_y"] > b_hi["r2_y"],
      b_hi[["cf_y", "r2_y"]].tolist())
# the per-split statement of the docs, from the built residuals (moments recomputed here, not by the function)
def split_moments(u, v):
    th = np.sum(v * u) / np.sum(v * v)
    return np.mean((u - th * v) ** 2), 1 / np.mean(v * v)


got = []
for i, (ly, gy, ld, gd) in enumerate(zip((1, 2, 3), (1.5, 0.01, 0.01), (1, 2, 3), (0.01, 0.01, 1.5))):
    (ul, vl), (us, vs) = split_pair(ly, gy, ld, gd, seed=i)
    (s2l, n2l), (s2s, n2s) = split_moments(ul, vl), split_moments(us, vs)
    got.append((s2s / s2l - 1, n2l / n2s - 1, (s2s - s2l) / s2s, (n2l - n2s) / n2l))
check("the built splits have the planted moments (a check of the test's own construction, not of the skill)",
      all(abs(a - gy) < 1e-9 and abs(b - gd) < 1e-9 and a > ry and b > rd and abs(ry - a / (1 + a)) < 1e-12
          for (a, b, ry, rd), gy, gd in zip(got, (1.5, 0.01, 0.01), (0.01, 0.01, 1.5))), got)
check("the 'below' design gives no 'cannot be paired' warning (splits paired, as DoubleML)",
      not any("cannot be paired" in s for s in w_lo), w_lo)

# one split: cf = min(r2 / (1 - r2), 1), on both sides of the cut
for g in (0.3, 1 - 1e-9, 1.0, 1 + 1e-9):
    b1, _ = bench_of((2.0,), (g,), (0.5,), (g,))
    exp_y = min(b1["r2_y"] / (1 - b1["r2_y"]), 1.0)
    exp_d = min(b1["r2_d"] / (1 - b1["r2_d"]), 1.0)
    check(f"one split, G = {g!r}: cf_y = min(r2_y/(1 - r2_y), 1) and cf_d = min(r2_d/(1 - r2_d), 1); "
          f"r2 = G/(1 + G); cf = min(G, 1)",
          abs(b1["cf_y"] - exp_y) < 1e-9 and abs(b1["cf_d"] - exp_d) < 1e-9
          and abs(b1["r2_y"] - g / (1 + g)) < 1e-9 and abs(b1["cf_y"] - min(g, 1.0)) < 1e-9
          and abs(b1["r2_d"] - g / (1 + g)) < 1e-9 and abs(b1["cf_d"] - min(g, 1.0)) < 1e-9
          and (b1["cf_y"] == 1.0) == (g >= 1.0 - 1e-12) and (b1["cf_d"] == 1.0) == (g >= 1.0 - 1e-12),
          b1[["cf_y", "r2_y", "cf_d", "r2_d"]].tolist())

# fallback to ratios of medians (res without a seed): cf = min(r2 / (1 - r2), 1) up to rounding (review 2)
_long = {"residuals": [split_pair(ly, gy, ld, gd, seed=i)[0] for i, (ly, gy, ld, gd) in
                       enumerate(zip((1, 2, 3), (1.5, 0.01, 0.01), (1, 2, 3), (0.01, 0.01, 1.5)))],
         "thetas": [0.0] * 3, "estimate": 0.0, "se": 0.05, "n_folds": 5}
_short = {"residuals": [split_pair(ly, gy, ld, gd, seed=i)[1] for i, (ly, gy, ld, gd) in
                        enumerate(zip((1, 2, 3), (1.5, 0.01, 0.01), (1, 2, 3), (0.01, 0.01, 1.5)))],
          "thetas": [0.1] * 3, "estimate": 0.1, "se": 0.05, "seed": 0, "n_folds": 5}
_orig = causal.dml_plr
causal.dml_plr = lambda *a, **kw: _short
try:
    bfb, _, wfb = run(causal.dml_benchmark, None, "y", "d", ["x1"], ["x1"], res=_long)
finally:
    causal.dml_plr = _orig
bfb = bfb.loc["x1"]
check("fallback (res without a seed), same splits: cf_y = 0.25 = r2_y/(1 - r2_y) and cf_d = 2/3 = r2_d/(1 - r2_d), "
      "up to rounding; not the paired 0.01",
      any("cannot be paired" in s for s in wfb) and abs(bfb["cf_y"] - 0.25) < 1e-12 and abs(bfb["cf_d"] - 2 / 3) < 1e-12
      and abs(bfb["cf_y"] - bfb["r2_y"] / (1 - bfb["r2_y"])) < 1e-12
      and abs(bfb["cf_d"] - bfb["r2_d"] / (1 - bfb["r2_d"])) < 1e-12, bfb[["cf_y", "r2_y", "cf_d", "r2_d"]].tolist())

# ------------------------------------------------------------------ 2. the docs
SKILL = ROOT / "econometrics"
docs = {"causal.py (dml_benchmark)": causal.dml_benchmark.__doc__,
        "causal-inference.md": (SKILL / "references" / "causal-inference.md").read_text(encoding="utf-8"),
        "interpretation-rules.md": (SKILL / "references" / "interpretation-rules.md").read_text(encoding="utf-8")}
bad = [k for k, v in docs.items() if re.search(r"always (smaller|below)|the smaller partial", v)]
check("no 'always smaller/below' or 'the smaller partial R²' left in the docstring, §4.1 and the rules", bad == [], bad)
check("the docstring says the two are different aggregations and gives the one-split identity",
      "different aggregations" in causal.dml_benchmark.__doc__ and "min(r2 / (1 - r2), 1)" in causal.dml_benchmark.__doc__)
sec41 = docs["causal-inference.md"]
check("§4.1 says per split and that cf_y can be below r2_y with several splits",
      "in each sample split" in sec41 and "can be below" in sec41)
chg = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
top = chg.split("\n## v")[1]
check("the CHANGELOG entry of v2.6.3 corrects 'always smaller' and 'plus the gain identity' of v2.6.2",
      top.startswith("2.6.3") and "always smaller" in top and "gain identity" in top)

# ------------------------------------------------------------------ 3. ovb_bounds "not reached"
def fit(seed, bd, by, eff, n=400):              # claude/scripts/sonda_v263.py (dev repository)
    r = np.random.default_rng(seed)
    x, w = r.normal(size=(2, n))
    d = 0.6 * x + bd * w + r.normal(size=n)
    y = eff * d + x + by * w + r.normal(size=n)
    return smf.ols("y ~ d + x + w", pd.DataFrame(dict(y=y, d=d, x=x, w=w))).fit()


M_NR = fit(1, 0.05, 1.0, 3.0)
o, t_nr, _ = run(sp.ovb_bounds, M_NR, "d", "w", show=True)
br = o["benchmark_r2"].loc["w"]
kc = f"{br['k_cap']:.4g}"
check("design seed 1: status 'not reached', cap from the outcome, t about 58",
      br["k_not_significant_status"] == "not reached" and br["k_cap_by"] == "outcome" and M_NR.tvalues["d"] > 50,
      (br["k_not_significant_status"], br["k_cap_by"], kc))
check(f"'not reached' names the multiplier: 'at k = {kc}, a confounder {kc}x as strong as w would explain all of "
      "the outcome's residual variance'",
      f"at k = {kc}, a confounder {kc}x as strong as w would explain all of the outcome's residual variance" in t_nr
      and f"at k = {kc}x" not in t_nr,
      [ln for ln in t_nr.splitlines() if "No k below" in ln])
check("its start is unchanged ('No k below … makes the estimate not significant', tests/test_v261.py)",
      f"No k below {kc} makes the estimate not significant for w" in t_nr)
check("the v2.6.2 wording is gone ('at that k a confounder as strong as')", "at that k a confounder" not in t_nr)
# robust fit: the same sentence with "(classical SE)"
o_r, t_r, _ = run(sp.ovb_bounds, M_NR.model.fit(cov_type="HC3"), "d", "w", show=True)
check("after HC3 the 'not reached' sentence says '(classical SE)' and carries the multiplier",
      f"not significant (classical SE) for w (at k = {kc}, a confounder {kc}x as strong as w" in t_r)

# ------------------------------------------------------------------ 4. the kd/ky label
M2 = fit(3, 0.3, 0.8, 0.3)
o2, t2, _ = run(sp.ovb_bounds, M2, "d", "w", kd=[1, 3], ky=[2, 0.5], show=True)
check("ovb_bounds labels as sensemakr's: '1/2x w', '3/0.5x w' (unchanged)",
      list(o2["bounds"]["bound_label"]) == ["1/2x w", "3/0.5x w"], list(o2["bounds"]["bound_label"]))
NOTE = "A label 'kd/ky x' (sensemakr's) gives kd for the treatment and ky for the outcome; it is not a ratio."
check("ovb_bounds with kd != ky: the 'not a ratio' note once", t2.count(NOTE) == 1)
_, t6, _ = run(sp.ovb_bounds, M2, "d", "w", kd=[0.3], ky=[0.1 * 3], show=True)
check("ovb_bounds kd = 0.3, ky = 0.1 * 3: the label stays sensemakr's ('0.3/0.3x w') and the sentence shows them "
      "apart (review 2)", "0.3/0.3x w: a confounder 0.3x as strong as w in explaining d and 0.30000000000000004x in "
      "explaining the outcome" in t6, [ln for ln in t6.splitlines() if "x w:" in ln])
lines2 = [ln.strip() for ln in t2.splitlines() if re.match(r"^\s+[0-9.]+(/[0-9.]+)?x ", ln)]
check("ovb_bounds lines still start with the label and ':' and name both multipliers",
      len(lines2) == 2 and lines2[0].startswith("1/2x w: a confounder 1x as strong as w in explaining d and 2x in "
                                                "explaining the outcome"), lines2)
_, t3, _ = run(sp.ovb_bounds, M2, "d", "w", show=True)
_, t4, _ = run(sp.ovb_bounds, M2, "d", "w", kd=[2], ky=[2.0], show=True)
_, t5, _ = run(sp.ovb_bounds, M2, "d", "w", kd=[1, 2], ky=[1, 3], show=True)
check("no note when every line has kd = ky (default k, and kd = 2 with ky = 2.0); once with one line of kd != ky",
      NOTE not in t3 and NOTE not in t4 and t5.count(NOTE) == 1)
check("…and kd = ky keeps ':g' ('2x as strong as w', never '2.0x')",
      "2x w: a confounder 2x as strong as w (" in t4 and "2.0x" not in t4)

rng = np.random.default_rng(5)
n = 600
x1, x2 = rng.normal(size=(2, n))
d = 0.6 * x1 + 0.4 * x2 + rng.normal(size=n)
y = 0.5 * d + 0.8 * x1 + 0.5 * x2 + rng.normal(size=n)
DD = pd.DataFrame(dict(y=y, d=d, x1=x1, x2=x2))
LIN = dict(learner_y=LinearRegression(), learner_d=LinearRegression())
RD = causal.dml_plr(DD, "y", "d", ["x1", "x2"], n_rep=1, **LIN)
BH = pd.DataFrame({"gain_y": [0.05], "gain_d": [0.1], "rho": [0.5]}, index=["x1"])
ob, tb, _ = run(causal.dml_bounds, RD, BH, kd=[2, 3], ky=[5, 3], rho=1.0, show=True)
lb = [ln.strip() for ln in tb.splitlines() if re.match(r"^\s+[0-9.]+(/[0-9.]+)?x x1:", ln)]
check("dml_bounds labels '2/5x x1' and '3x x1' (the pattern of ovb_bounds)",
      list(ob["bounds"]["bound_label"]) == ["2/5x x1", "3x x1"], list(ob["bounds"]["bound_label"]))
check("dml_bounds spells out kd and ky on the kd != ky line, right after 'label:'",
      len(lb) == 2 and lb[0].startswith("2/5x x1: (kd = 2 for the treatment, ky = 5 for the outcome) cf_y 25.0%"), lb)
check("…and not on the kd = ky line", len(lb) == 2 and lb[1].startswith("3x x1: cf_y 15.0%") and "(kd =" not in lb[1],
      lb)
_, tb2, _ = run(causal.dml_bounds, RD, BH, kd=[2], ky=[2.0], rho=1.0, show=True)
check("dml_bounds kd = 2, ky = 2.0: label '2x x1', no '(kd ='", "2x x1: cf_y" in tb2 and "(kd =" not in tb2)
_, tb3, _ = run(causal.dml_bounds, RD, BH, kd=[0.3], ky=[0.1 * 3], rho=1.0, show=True)
check("dml_bounds kd = 0.3, ky = 0.1 * 3 (equal with :g, not as floats): the parenthesis shows them apart (review)",
      "(kd = 0.3 for the treatment, ky = 0.30000000000000004 for the outcome)" in tb3,
      [ln for ln in tb3.splitlines() if "x x1:" in ln])
# not a repeat of the first dml_bounds check: this one fails if the code always used repr ("kd = 2.0")
_, tb4, _ = run(causal.dml_bounds, RD, BH, kd=[2], ky=[5], rho=1.0, show=True)
check("…and keeps the short format when :g already tells them apart ('kd = 2', 'ky = 5')",
      "(kd = 2 for the treatment, ky = 5 for the outcome)" in tb4)
check("the docstrings say the label is not a ratio",
      "not a ratio" in causal.dml_bounds.__doc__ and "not a ratio" in sp.ovb_bounds.__doc__)

# ------------------------------------------------------------------ 3 (sweep). every printed ovb_bounds text
bare = [ln for t in TEXTS for ln in t.splitlines() if re.search(r"confounder as strong as", ln)]
check(f"sweep over the {len(TEXTS)} printed ovb_bounds outputs of this file: no 'confounder as strong as' without "
      "a multiplier", bare == [] and len(TEXTS) >= 6, bare[:2])

# ------------------------------------------------------------------ 5. rule and evaluation case
rules = docs["interpretation-rules.md"]
sec = rules.split("## Rules added in v2.6.3")[1] if "## Rules added in v2.6.3" in rules else ""
check("interpretation-rules.md has 'Rules added in v2.6.3' with k_not_significant and ci_verdict, after v2.6.2",
      bool(sec) and "k_not_significant" in sec and "ci_verdict" in sec
      and rules.index("## Rules added in v2.6.2") < rules.index("## Rules added in v2.6.3"))
cases = json.loads((ROOT / "evals" / "cases.json").read_text(encoding="utf-8"))["cases"]
c = [c_ for c_ in cases if c_.get("id") == "review-memo-k-threshold-paraphrase"]
check("evaluation case review-memo-k-threshold-paraphrase: one, pt-BR, with prompt, must and must_not",
      len(c) == 1 and c[0]["language"] == "pt-BR" and len(c[0]["must"]) >= 3 and len(c[0]["must_not"]) >= 2
      and "não resiste" in c[0]["prompt"] and "1.05x" in c[0]["prompt"] and "1,2" in c[0]["prompt"])
check("ids of the evaluation cases are unique", len({c_["id"] for c_ in cases}) == len(cases))

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
