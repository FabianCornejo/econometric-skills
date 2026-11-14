"""Tests for v2.6.8: the exact Oster estimator (Assumption 3 with a tolerance, the spurious root by its cause, the
alternatives split), the ρ-clipping rates in did_single_treated, the doctor and versions that are not release
numbers, LIML offered only when overidentified, the rule on numbers worked out in the reply, and the version.

Origin: the v2.5.2 review (deferred NITs: np.sign without a tolerance, spurious root "with a single control",
beta_star_alternatives mixing sides, a non-numeric project-install version), the backlog triage of 2026-11-12
(the "about 18%" of the placebo warning) and the v2.6.7 validation on notebook 1 (LIML in an exactly identified
model; a confidence interval worked out in the reply). Plan: claude/plano-v2.6.8.md in the development repository.

Each boundary has a case on each side and asserts what must NOT happen. The search scripts, the measurements and
the seeds are in claude/scripts/ of the development repository (sonda_oster_a3_v268.py, estudo_espuria_v268.py,
residuo_identidade_v268.py, comparar_oster_v268.py, corte_rho_v268.py, busca_desenhos_v268.py). Every section has
its own generator.

Run from the repository root:  python3 tests/test_v268.py
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "econometrics"
sys.path.insert(0, str(SKILL / "scripts"))
import _console  # noqa: E402,F401
import causal as cz  # noqa: E402
import diagnostics as dg  # noqa: E402
import reporting  # noqa: E402
import setup_env as se  # noqa: E402
import specification as sp  # noqa: E402

warnings.filterwarnings("ignore")
ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info)


def oster(*a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.oster(*a, **k)


# ----------------------------------------------------------- 1. Oster: the calculation is v2.6.7's (item 1 dropped)
# The plan's item 1 (no direction for Assumption 3 when β° and β̃ differ by rounding) was dropped after the second
# round of review: with β° = β̃, β̃ is a 0/0 point of the exact equation, not a solution in general (decision of the
# user, 2026-11-13). The numbers are main's; claude/scripts/comparar_oster_v268.py found no numeric difference on
# 3618 designs. What this version keeps is documented as a known limitation.
# A genuine violation (the R reference data of tests/test_v252.py).
mt = pd.read_csv(ROOT / "tools" / "r_reference" / "output" / "mtcars.csv")
m0_mt, m1_mt = smf.ols("mpg ~ wt", mt).fit(), smf.ols("mpg ~ wt + hp + qsec", mt).fit()
o = oster(m0_mt, m1_mt, "wt", r_max=0.9, delta=2)
check("a genuine violation (mtcars, δ = 2, R_max = 0.9) still says Assumption 3 and 'Do not report'",
      any("Assumption 3" in m and "Do not report" in m for m in o["warnings"]), o["warnings"])


# ------------------------------------- 2. Oster: the spurious root, with two controls too (classification as v2.6.7)
# y = 0.5 x + 0.7 π'W + e, with e orthogonal to (1, x, W): the controls' coefficients in y are proportional to their
# coefficients in x, so the numerator of δ*(β) vanishes at the pole with two controls.
def eps_fits(seed, eps):
    rng = np.random.default_rng(seed)
    n = 400
    W = rng.normal(size=(n, 2))
    x = W @ [0.8, -0.5] + rng.normal(size=n)
    pi = np.linalg.lstsq(np.column_stack([np.ones(n), W]), x, rcond=None)[0][1:]
    e = rng.normal(size=n)
    A = np.column_stack([np.ones(n), x, W])
    e = e - A @ np.linalg.lstsq(A, e, rcond=None)[0]
    y = 1 + 0.5 * x + 0.7 * (W @ pi) + eps * W[:, 1] + e
    d = pd.DataFrame(dict(y=y, x=x, w1=W[:, 0], w2=W[:, 1]))
    return smf.ols("y ~ x", d).fit(), smf.ols("y ~ x + w1 + w2", d).fit(), d


def identity_rel(m0, m1, d):
    """(R̃ − R°)·var(y) against (β° − β̃)²·var(x)·τ_x/(var(x) − τ_x), from the data (not from the function)."""
    vy, vx = np.var(d["y"], ddof=1), np.var(d["x"], ddof=1)
    tau = np.var(smf.ols("x ~ w1 + w2", d).fit().resid, ddof=1)
    gain = (m1.rsquared - m0.rsquared) * vy
    return abs(gain - (m0.params["x"] - m1.params["x"]) ** 2 * vx * tau / (vx - tau)) / gain


m0, m1, d0 = eps_fits(0, 0.0)
o2 = [oster(m0, m1, "x", delta=dl) for dl in (0.5, 2.0, 20.0)]
check("two controls with γ ∝ π: the identity holds to rounding, and a spurious root is found for every δ "
      "(the docstring used to say 'with a single control')",
      identity_rel(m0, m1, d0) < 1e-12 and all(len(o["beta_star_spurious"]) == 1 for o in o2),
      [identity_rel(m0, m1, d0)] + [o["beta_star_spurious"] for o in o2])
nr = [m for o in o2 for m in o["warnings"] if "is the spurious root of the cubic they solve" in m]
check("the spurious warning: no single-control tie, the pole named, no claim that it 'does not satisfy' the equation",
      nr and all("lines up with their relation to the treatment" in m and "at the pole of δ*(β)" in m
                 and "does not satisfy" not in m and "with a single control it appears" not in m for m in nr), nr)


# The rule the review rejected (v2.6.8, round 1): the identity to a tolerance, instead of the distance to the pole.
# These cases keep v2.6.7's answers. (a) one control, R² gain 2.4e-11: the 1e-6 identity rule reported the spurious
# root 0.99999792 as β* with δ = −1; (b) two controls with γ ∝ π, x on a 1e6 scale, controls barely related to x:
# rounding put the identity off by 2.8e-5, and the rule reported the spurious root 5.1e-7 with δ = −1; (c) two
# generic random designs (seeds 2036 and 3077 of claude/scripts/comparar_oster_v268.py) with the identity off by
# 2.2e-5 and 9.4e-4 by chance: v2.6.7 and this version give a genuine β* next to the pole, ruled out by Assumption 3.
def tiny_gain():
    rng = np.random.default_rng(1)
    n = 300
    x = rng.normal(size=n)
    w = 0.6 * x + 0.8 * rng.normal(size=n)
    y = 1 + x + 1e-5 * w + 1e-4 * rng.normal(size=n)
    d = pd.DataFrame(dict(y=y, x=x, w=w))
    return smf.ols("y ~ x", d).fit(), smf.ols("y ~ x + w", d).fit()


def scaled_aligned():
    rng = np.random.default_rng(16)
    n = 300
    W = rng.normal(size=(n, 2))
    x = 1e6 * (0.01 * (W @ [1.0, -0.6]) + rng.normal(size=n))
    pi = np.linalg.lstsq(np.column_stack([np.ones(n), W]), x, rcond=None)[0][1:]
    e = rng.normal(size=n)
    A = np.column_stack([np.ones(n), x, W])
    e = e - A @ np.linalg.lstsq(A, e, rcond=None)[0]
    y = 1 + 0.5 * x / 1e6 + 0.01 * (W @ pi) / 1e6 + e
    d = pd.DataFrame(dict(y=y, x=x, w0=W[:, 0], w1=W[:, 1]))
    return smf.ols("y ~ x", d).fit(), smf.ols("y ~ x + w0 + w1", d).fit()


def generic(k, seed):
    rng = np.random.default_rng(1000 * k + seed)
    n = 300
    W = rng.normal(size=(n, k))
    x = W @ rng.uniform(-1, 1, k) + rng.normal(size=n)
    y = 1 + rng.uniform(-1, 1) * x + W @ rng.uniform(-1, 1, k) + rng.normal(size=n)
    d = pd.DataFrame(dict(y=y, x=x, **{f"w{j}": W[:, j] for j in range(k)}))
    return smf.ols("y ~ x", d).fit(), smf.ols("y ~ x + " + " + ".join(f"w{j}" for j in range(k)), d).fit()


ta = [oster(*tiny_gain(), "x", delta=dl) for dl in (-1.0, 2.0)]
check("(a) one control, R² gain at rounding level: β* nan and the root 0.99999792 spurious, for δ = −1 and 2",
      all(np.isnan(o["beta_star_exact"]) and len(o["beta_star_spurious"]) == 1
          and abs(o["beta_star_spurious"][0] - 0.9999979222) < 1e-9 for o in ta),
      [(o["beta_star_exact"], o["beta_star_spurious"]) for o in ta])
sa = oster(*scaled_aligned(), "x", delta=-1.0)
check("(b) two aligned controls, x on a 1e6 scale: β* nan with δ = −1, the root 5.1e-7 spurious, not reported",
      np.isnan(sa["beta_star_exact"]) and len(sa["beta_star_spurious"]) == 1
      and abs(sa["beta_star_spurious"][0] - 5.0999986e-7) < 1e-13, (sa["beta_star_exact"], sa["beta_star_spurious"]))
ga = [oster(*generic(k, s_), "x", delta=2.0) for k, s_ in ((2, 36), (3, 77))]
check("(c) two generic designs near alignment by chance: a genuine β* next to the pole (0.001776, −0.1123), nothing "
      "spurious, the warning is Assumption 3",
      [round(o["beta_star_exact"], 4) for o in ga] == [0.0018, -0.1123]
      and all(not o["beta_star_spurious"] and any("Assumption 3" in m for m in o["warnings"]) for o in ga),
      [(o["beta_star_exact"], o["beta_star_spurious"], o["warnings"]) for o in ga])
# One control (the R reference data): still spurious, as in v2.5.2.
sim = pd.read_csv(ROOT / "tests" / "data" / "oster_v252.csv")
m0s, m1s = smf.ols("y ~ x", sim).fit(), smf.ols("y ~ x + w1", sim).fit()
os2 = oster(m0s, m1s, "x", r_max=1, delta=2)
check("one control (oster_v252.csv, δ = 2, R_max = 1): the only real root is spurious, β* nan, as in v2.5.2",
      np.isnan(os2["beta_star_exact"]) and len(os2["beta_star_spurious"]) == 1
      and abs(os2["beta_star_spurious"][0] - 2.803677) < 2e-6, os2)
# Generic controls (mtcars): no spurious root.
m0g, m1g = smf.ols("mpg ~ wt", mt).fit(), smf.ols("mpg ~ wt + hp + qsec", mt).fit()
check("mtcars (two controls, generic): no spurious root for δ = 0.5, 1, 2, 3",
      all(oster(m0g, m1g, "wt", r_max=0.9, delta=dl)["beta_star_spurious"] == [] for dl in (0.5, 1, 2, 3)))

# --------------------------------------------------------------- 3. Oster: alternatives split by side
# robomit (tests/test_v252.py): mtcars δ = 1 has a second root, −7.014095; β° = −5.34 and β̃ > β°, so the
# admissible side is above β̃ and −7.014 is on the forbidden side. sim δ = 2, R_max = 0.885: roots 0.550597 and
# 2.803677 besides β* = 1.131751 (2.803677 is the spurious one).
o_mt1 = oster(m0_mt, m1_mt, "wt", r_max=0.9, delta=1)
check("mtcars δ = 1: the second root −7.014 is on the forbidden side, no admissible alternative; the old key unchanged",
      len(o_mt1["beta_star_forbidden_side"]) == 1 and abs(o_mt1["beta_star_forbidden_side"][0] + 7.014095) < 2e-6
      and o_mt1["beta_star_admissible_alternatives"] == [] and o_mt1["beta_star_alternatives"]
      == o_mt1["beta_star_forbidden_side"], o_mt1)
o_s2 = oster(m0s, m1s, "x", r_max=0.885, delta=2)
check("sim δ = 2, R_max = 0.885: 0.5506 is an admissible alternative (below β̃, as β° > β̃), the spurious root in "
      "neither list",
      [round(r, 6) for r in o_s2["beta_star_admissible_alternatives"]] == [0.550597]
      and o_s2["beta_star_forbidden_side"] == [] and len(o_s2["beta_star_spurious"]) == 1, o_s2)
grid = [oster(m, mm, f, r_max=rm, delta=dl) for (m, mm, f) in ((m0s, m1s, "x"), (m0_mt, m1_mt, "wt"))
        for rm in (0.86, 0.885, 0.9, 1.0) for dl in (-1.0, 0.5, 1.0, 2.0, 3.0)]
check("on a grid of 40 (δ, R_max): admissible + forbidden is beta_star_alternatives, element for element",
      all(sorted(o["beta_star_admissible_alternatives"] + o["beta_star_forbidden_side"])
          == sorted(o["beta_star_alternatives"]) for o in grid if "beta_star_alternatives" in o))
check("δ ≤ 0 has no forbidden side (Assumption 3 is about δ > 0)",
      all(o["beta_star_forbidden_side"] == [] for o in grid if o["delta"] <= 0))
check("docstring: both new keys, the cause of the spurious root, δ ≤ 0 without a direction, and the known "
      "limitations (δ near 1; a genuine root within the pole's tolerance; β̃ a 0/0 point when β° = β̃)",
      all(k in sp.oster.__doc__ for k in ("beta_star_admissible_alternatives", "beta_star_forbidden_side",
                                          "proportional to their coefficients in", "within about 1e-7 of 1",
                                          "with δ ≤ 0 no direction is", "called spurious", "0/0 point",
                                          "with δ < 0, where they still apply the sign check"))
      and "With a single control the cubic always has" not in sp.oster.__doc__
      and "it differs whenever the spurious root is the closest one" not in sp.oster.__doc__)
# δ ≤ 0: several admissible roots, and the warning must not claim a direction (review, round 2)
gd = [oster(*generic(2, s_), "x", delta=-0.5) for s_ in (1, 2)]
wd = [m for o in gd for m in o["warnings"] if "admissible real roots" in m]
check("δ = −0.5 with two admissible roots (generic seeds 2001, 2002): 'closest to β̃ (with δ ≤ 0 no direction is "
      "applied)', not 'in the direction of the move'", len(wd) == 2 and all(
          "with δ ≤ 0 no direction is applied" in m and "direction of the move" not in m for m in wd), wd)
d2w = oster(*generic(2, 1), "x", delta=2.0)
check("δ = 2, same data: if several roots are admissible the warning still names Oster's direction (other side)",
      all("direction of the move from β°" in m for m in d2w["warnings"] if "admissible real roots" in m))
ols_md = (SKILL / "references" / "ols-regression.md").read_text(encoding="utf-8")
check("ols-regression.md: the spurious root with several controls, and the two new keys",
      "proportional to their coefficients in x" in ols_md and "beta_star_forbidden_side" in ols_md)

# ------------------------------------------------- 4. did_single_treated: ρ-clipping rates measured for each check
# Rates re-measured here with seeds of this test (20000 stationary AR(1) series per cell, the function's own
# design and clipping); the bands are about ±4 to ±7 Monte Carlo SEs (0.002–0.003 with 20000 series) around the
# measurement in claude/scripts/corte_rho_v268.py, and each quoted figure lies inside its band.
def clip_rate(T, T0, rho, seed, R=20000):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((R, T))
    x = np.empty((R, T))
    x[:, 0] = z[:, 0] / np.sqrt(1 - rho ** 2)
    for t in range(1, T):
        x[:, t] = rho * x[:, t - 1] + z[:, t]
    X, w, br = cz._sts_design(T, T0)
    return float(np.mean(cz._sts_general(x, X, w, br)[3]))


rates = {("trend", 12): (clip_rate(12, None, 0.4, 41), 0.19, 0.23), ("trend", 16): (clip_rate(16, None, 0.4, 42), 0.09, 0.11),
         ("trend", 20): (clip_rate(20, None, 0.4, 43), 0.039, 0.051), ("placebo", 16): (clip_rate(16, 8, 0.4, 44), 0.066, 0.079),
         ("placebo", 20): (clip_rate(20, 10, 0.4, 45), 0.032, 0.040), ("main 12", 0.4): (clip_rate(12, 6, 0.4, 46), 0.163, 0.180),
         ("main 12", 0.0): (clip_rate(12, 6, 0.0, 47), 0.029, 0.038), ("main 16", 0.4): (clip_rate(16, 8, 0.4, 48), 0.062, 0.074),
         ("main 16", 0.0): (clip_rate(16, 8, 0.0, 49), 0.003, 0.0065)}
check("clipping rates of each check re-measured with new seeds fall in the bands of the measurement",
      all(lo <= r <= hi for r, lo, hi in rates.values()), {k: round(v[0], 4) for k, v in rates.items()})


def panel(seed, T0, T1=6, rho=0.4):
    rng = np.random.default_rng(seed)
    T = T0 + T1
    g = np.empty(T)
    g[0] = rng.normal() / np.sqrt(1 - rho ** 2)
    for t in range(1, T):
        g[t] = rho * g[t - 1] + rng.normal()
    return pd.DataFrame([dict(u="t", t=t, y=g[t]) for t in range(T)] + [dict(u="c", t=t, y=0.0) for t in range(T)])


# seeds from claude/scripts/busca_desenhos_v268.py: pre-trend clipped and not rejecting (T0 = 12, seed 0); placebo
# clipped and not rejecting with the trend not clipped (T0 = 16, seed 20)
r_tr = cz.did_single_treated(panel(0, 12), "u", "t", "y", ["t"], t0=12, n_sim=999, seed=1, show=False)
r_pl = cz.did_single_treated(panel(20, 16), "u", "t", "y", ["t"], t0=16, n_sim=999, seed=1, show=False)
w_tr = [m for m in r_tr["warnings"] if m.startswith("The pre-period trend test found")]
w_pl = [m for m in r_pl["warnings"] if m.startswith("The pre-period placebo found")]
check("trend test clipped, not rejecting: its own rates (21% of 12-period, 10% of 16-period, 4.5% with 20)",
      r_tr["pre_trend"]["rho_clipped"] and r_tr["pre_trend"]["p"] > 0.05 and len(w_tr) == 1
      and "about 21% of 12-period and 10% of 16-period pre-periods with ρ = 0.4, 4.5% with 20" in w_tr[0]
      and "18%" not in w_tr[0], w_tr)
check("placebo clipped, not rejecting: its own rates (7% with 16, 3.5% with 20), never the 12-period figure",
      r_pl["pre_placebo"]["rho_clipped"] and r_pl["pre_placebo"]["p"] > 0.05 and len(w_pl) == 1
      and "about 7% of 16-period pre-periods with ρ = 0.4 and 3.5% with 20" in w_pl[0]
      and "12-period" not in w_pl[0] and "18%" not in w_pl[0], w_pl)
# 12 periods (6 + 6), the main step's ρ clipped (seed 0 of claude/scripts/busca_desenhos_v268.py, part (c)); and
# 24 periods, where the sentence must not appear
r_mg = cz.did_single_treated(panel(0, 6, 6), "u", "t", "y", ["t"], t0=6, n_sim=999, seed=1, show=False)
w_mg = [m for m in r_mg["warnings"] if "the series is too short to tell" in m]
r_24 = cz.did_single_treated(panel(0, 12, 12), "u", "t", "y", ["t"], t0=12, n_sim=999, seed=1, show=False)
check("main-gap sentence (12 periods, clipped): 17% and 3% with 12 periods, 7% and 0.5% with 16; old 18%/4%/5% gone; "
      "absent with 24 periods",
      len(w_mg) == 1 and "about 17% of gaps with ρ = 0.4 and 3% of white-noise gaps with 12 periods, and 7% and 0.5% "
      "with 16" in w_mg[0] and "18%" not in w_mg[0] and "5% and" not in w_mg[0]
      and not any("the series is too short to tell" in m for m in r_24["warnings"]), (w_mg, r_24["warnings"]))

# --------------------------------------------------- 5. doctor: a project install whose version is not a release
tmp = Path(tempfile.mkdtemp(prefix="v268 "))
cdir = tmp / "claude"
cdir.mkdir()
folder = tmp / "analises" / "pasta com espaço"
(folder / ".claude").mkdir(parents=True)
eco = tmp / "plugins" / "eco" / "s0"
eco.mkdir(parents=True)
(eco / "SKILL.md").write_text("x", encoding="utf-8")
claude = lambda n: "/bin/claude" if n == "claude" else None
ver = se.read_skill_version(se.SKILL_DIR)


def doc(installs):
    run = lambda cmd, timeout: (0, json.dumps(installs)) if cmd[1:4] == ["plugin", "list", "--json"] else (0, "")
    return se.doctor(offline=True, cdir=cdir, env={}, run=run, which=claude, os_name="posix", cwd=str(folder))["actions"]


user = {"id": "econometrics@econometric-skills", "enabled": True, "scope": "user", "installPath": str(eco.parent),
        "version": ver}
pin = lambda v: dict(user, scope="project", projectPath=str(folder), version=v)
for v in ("abcdef0", "3a98a97", "deadbeef"):
    acts = doc([pin(v), user])
    check(f"pin at '{v}' (not a release): no update or uninstall advice, a line saying it cannot be compared",
          not any("pinned at" in a or "plugin uninstall" in a or "plugin update" in a for a in acts)
          and any(f"at version {v}, which is not a release number" in a and "claude plugin list --json" in a
                  for a in acts), acts)
acts = doc([pin("2.4.0"), user])
check("pin at 2.4.0 with a newer user install (a release): the old advice, uninstall the folder's own (other side)",
      any("pinned at 2.4.0" in a and "plugin uninstall" in a for a in acts)
      and not any("not a release number" in a for a in acts), acts)
acts = doc([pin("2.4.0"), dict(user, version="ffff999")])
check("pin at 2.4.0 with a user install whose version is not a release: update in place, never uninstall, and it "
      "says why truthfully (the user install is there, its version cannot be ordered)",
      any("pinned at 2.4.0" in a and "plugin update" in a and "do not uninstall it: the user-scope install (ffff999) "
          "is not a release number" in a for a in acts)
      and not any("plugin uninstall" in a or "no user-scope install" in a for a in acts), acts)
acts = doc([pin("2.4.0")])
check("pin at 2.4.0 and no user install: update in place, 'there is no user-scope install to take over'",
      any("pinned at 2.4.0" in a and "do not uninstall it: there is no user-scope install to take over" in a
          for a in acts), acts)
acts = doc([pin("2.4.0"), {k: v for k, v in user.items() if k != "version"}])
check("pin at 2.4.0 and a user install with no version field: 'reports no version', not 'no user-scope install' "
      "(review, round 2)", any("do not uninstall it: the user-scope install reports no version" in a for a in acts)
      and not any("no user-scope install" in a or "plugin uninstall" in a for a in acts), acts)
acts = doc([pin("2.4.0"), dict(user, version="2.3.0")])
check("pin at 2.4.0 and an older user install (2.3.0): update in place, 'the user-scope install (2.3.0) is not newer'",
      any("do not uninstall it: the user-scope install (2.3.0) is not newer" in a for a in acts)
      and not any("plugin uninstall" in a for a in acts), acts)
acts = doc([pin(ver), user])
check("pin at the current release: nothing about the folder's install", not any("this folder (" in a for a in acts), acts)

# ----------------------------------------------- 6. IV: LIML only when overidentified (exactly identified = 2SLS)
def ivdata(seed, k):
    rng = np.random.default_rng(seed)
    n = 400
    Z = rng.normal(size=(n, k))
    u = rng.normal(size=n)
    v = 0.8 * u + 0.6 * rng.normal(size=n)
    x = Z @ np.full(k, 0.12) + v
    y = 1 + 0.5 * x + u
    return pd.DataFrame(dict(y=y, x=x, **{f"z{j + 1}": Z[:, j] for j in range(k)}))


def endo_text(d, zs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = dg.endogeneity_test(d["y"], None, d[["x"]], d[zs], show=True)
    return res, buf.getvalue()


d1, d2 = ivdata(2, 1), ivdata(0, 2)                      # seeds from claude/scripts/busca_desenhos_v268.py
r1, t1 = endo_text(d1, ["z1"])
r2, t2 = endo_text(d2, ["z1", "z2"])
check("endogeneity_test, exactly identified, weak and rejecting: Anderson-Rubin only, and says why LIML does not help",
      r1["first_stage_F"]["x"] <= 10 and r1["p"] < 0.05
      and "weak-instrument-robust inference (Anderson-Rubin); LIML does not help here" in t1
      and "it equals 2SLS" in t1 and "or LIML" not in t1, t1)
check("endogeneity_test, overidentified, weak and rejecting: Anderson-Rubin or LIML (other side)",
      r2["first_stage_F"]["x"] <= 10 and r2["p"] < 0.05 and "(Anderson-Rubin) or LIML." in t2
      and "LIML does not help" not in t2, t2)
with contextlib.redirect_stdout(io.StringIO()):
    iv1 = dg.iv_report("y ~ 1 + [x ~ z1]", d1)
    iv2 = dg.iv_report("y ~ 1 + [x ~ z1 + z2]", d2)
w1 = [m for m in iv1["warnings"] if m.startswith("weak instrument")]
w2 = [m for m in iv2["warnings"] if m.startswith("weak instrument")]
check("iv_report, exactly identified and weak: the Anderson-Rubin set, LIML named only to say it equals 2SLS",
      len(w1) == 1 and "Anderson-Rubin set and say so (LIML does not help here: exactly identified, it equals 2SLS)"
      in w1[0] and "(or LIML)" not in w1[0], w1)
check("iv_report, overidentified and weak: 'Anderson-Rubin set (or LIML)' (other side)",
      len(w2) == 1 and "Anderson-Rubin set (or LIML) and say so" in w2[0], w2)
from linearmodels.iv import IV2SLS, IVLIML  # noqa: E402
b_2s = IV2SLS.from_formula("y ~ 1 + [x ~ z1]", d1).fit().params["x"]
b_li = IVLIML.from_formula("y ~ 1 + [x ~ z1]", d1).fit().params["x"]
b_2s2 = IV2SLS.from_formula("y ~ 1 + [x ~ z1 + z2]", d2).fit().params["x"]
b_li2 = IVLIML.from_formula("y ~ 1 + [x ~ z1 + z2]", d2).fit().params["x"]
check("the claim itself (linearmodels): LIML equals 2SLS exactly identified (1e-8), and differs overidentified",
      abs(b_2s - b_li) < 1e-8 and abs(b_2s2 - b_li2) > 1e-4, (b_2s - b_li, b_2s2 - b_li2))
# instruments counted on the model's columns: [x ~ z1*z2] has 3 (z1, z2, z1:z2), so it is overidentified
# (review, round 1: counting the formula's terms called it exactly identified)
d3 = ivdata(0, 2)
d3["x"] = 0.08 * d3["z1"] + 0.08 * d3["z2"] + (d3["x"] - 0.12 * d3["z1"] - 0.12 * d3["z2"])
with contextlib.redirect_stdout(io.StringIO()):
    iv3 = dg.iv_report("y ~ 1 + [x ~ z1*z2]", d3)
w3 = [m for m in iv3["warnings"] if m.startswith("weak instrument")]
check("iv_report with [x ~ z1*z2] (3 instrument columns, weak): overidentified, 'Anderson-Rubin set (or LIML)'",
      len(w3) == 1 and "Anderson-Rubin set (or LIML) and say so" in w3[0] and "exactly identified" not in w3[0], w3)
piv = (SKILL / "references" / "panel-iv.md").read_text(encoding="utf-8")
check("panel-iv.md says the same as the code (line 57 and the code comment)",
      "LIML only with more instruments than endogenous regressors" in piv and "Anderson-Rubin, or LIML)" not in piv
      and "# preferable with weak instruments" not in piv and "# with weak instruments, when overidentified" in piv)

# --------------------------------------------------- 7. the rule on numbers worked out in the reply
rules = (SKILL / "references" / "interpretation-rules.md").read_text(encoding="utf-8")
heads = re.findall(r"^## .*$", rules, re.M)
r268 = rules[rules.index("## Rules added in v2.6.8"):] if "## Rules added in v2.6.8" in rules else ""
_vt = lambda h: tuple(int(p) for p in re.search(r"v([\d.]+)$", h).group(1).split("."))
_i68 = heads.index("## Rules added in v2.6.8") if "## Rules added in v2.6.8" in heads else -1
check("interpretation-rules.md: 'Rules added in v2.6.8' right after v2.6.6, followed only by later versions",
      _i68 > 0 and heads[_i68 - 1] == "## Rules added in v2.6.6"
      and all(h.startswith("## Rules added in v") and _vt(h) > (2, 6, 8) for h in heads[_i68 + 1:]), heads[-3:])
r268 = r268.split("\n## ", 1)[0] if "\n## " in r268 else r268          # the v2.6.8 section only
check("the v2.6.8 rule: printed by the script, the same critical value, the SE of the model described, its origin",
      all(k in r268 for k in ("printed by the script", "never work it out in the reply", "t(G − 1)",
                              "SE of the model the sentence describes", "v2.6.7 validation", "[0.37, 0.65]",
                              "Rounding a printed number to 2–3 significant digits is fine")),
      r268[:500])
check("the rule quotes the reply's interval as the reply's, and no interval worked out by the rule itself (the "
      "registered HR and SE are rounded: they do not reproduce either bound)", "the interval is [" not in r268
      and "gave \"HR 0.49 with an approximate CI of [0.37, 0.65]" in r268, r268[:400])

# ------------------------------------------------------------- 8. version and CHANGELOG
cl = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
mk = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))
first = re.search(r"^## v([\d.]+) — ", cl, re.M)
check("version in CHANGELOG (top), reporting.SKILL_VERSION and marketplace.json agree and are 2.6.8 or later",
      first is not None and first.group(1) == reporting.SKILL_VERSION == mk["plugins"][0]["version"]
      and tuple(int(x) for x in first.group(1).split(".")) >= (2, 6, 8),
      (first and first.group(1), reporting.SKILL_VERSION, mk["plugins"][0]["version"]))
top = re.search(r"^## v2\.6\.8 — ", cl, re.M)
e268 = cl[top.start():cl.index("\n## v", top.end())] if top else ""
check("CHANGELOG v2.6.8 names each item and the measured rates",
      all(k in e268 for k in ("Assumption 3", "beta_star_admissible_alternatives", "beta_star_forbidden_side",
                              "proportional to their coefficients in x", "The classification stays the distance",
                              "trend test 21% with 12 pre-periods, 10% with 16, 4.5% with 20",
                              "placebo 7% with 16, 3.5% with 20", "measured: 17%, 3%, 7% and 0.5%",
                              "not a release number", "LIML equals 2SLS", "`[x ~ z1*z2]` has three",
                              "interpretation-rules.md")),
      e268[:600])

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
