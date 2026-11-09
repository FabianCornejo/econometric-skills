"""Tests for v2.5.2: the exact Oster estimator, the doctor measuring the analysis folder (--folder), ρ clipping warned
in the pre-period checks of did_single_treated and did_falsification, and the exact randomization p-value.

Origin: items D and E of the v2.5.1 plan (deferred), the two findings of the v2.5.1 validation on notebook 2
(2026-11-08: the doctor ran from inside the skill's folder and offered a clean folder that was already in use; a
clipped ρ in the pre-period trend test passed silently), and the v2.5.1 NIT on the exact randomization p.

Oster, by three paths that do not copy the implementation's algebra:
  1. Replication of R's robomit 1.0.7 (which follows Stata's psacalc): tools/r_reference/oster_v252_reference.R,
     run on notebook 1 (R 4.6.2, 2026-11-08). The numbers below were extracted from its JSON output by script, not
     retyped. robomit prints 6 decimals, so the tolerance is 2e-6.
  2. The roots by another route: every solution of δ*(β) = δ found by bracketing and brentq on a grid of β, using
     the δ formula (a different expression from the cubic), against the trigonometric/Cardano solver's roots.
  3. Population: large samples from Oster's proportional-selection DGP with a known δ and true effect 1. The bands
     were fixed after an exploratory study (claude/estudo-oster-exato-v252-2026-11-08.md in the development
     repository) and are checked here on seeds that study did not use. That study found the exact β* ill-conditioned
     near a double root (δ = 2, n = 400 000, 2-3% off), hence the beta_star_exact_unstable flag tested below.
Warnings: one per situation; the common case (one admissible root, exact ≈ approximation) has none; a root that
violates Oster's Assumption 3 is never called "the one to report". Each boundary also asserts what must NOT happen.

Run from the repository root:  python3 tests/test_v252.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import optimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "econometrics" / "scripts"))
import _console  # noqa: E402,F401
import causal as ca  # noqa: E402
import setup_env as se  # noqa: E402
import specification as sp  # noqa: E402

warnings.filterwarnings("ignore")
ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info)


def oster_quiet(*a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.oster(*a, **k)


# ---------------------------------------------------------------------------------------- 1. robomit replication
# (case, delta, R2max, beta*, other real roots, beta uncontrolled, R2 controlled)
ROBOMIT_BETA = [
    ("mtcars", 1, 0.9, -2.000185, [-7.014095], -5.34447157272268, 0.83476776170583),
    ("mtcars", 0.5, 0.9, -3.513622, [], -5.34447157272268, 0.83476776170583),
    ("mtcars", 2, 0.9, -6.406107, [], -5.34447157272268, 0.83476776170583),
    ("sim", 0.5, 0.885, 1.431273, [2.803677, 4.17608], 1.63553961147753, 0.839643666096557),
    ("sim", 1, 0.885, 1.364068, [2.803677], 1.63553961147753, 0.839643666096557),
    ("sim", 2, 0.885, 1.131751, [0.550597, 2.803677], 1.63553961147753, 0.839643666096557),
    ("sim", 2, 1, 2.803677, [], 1.63553961147753, 0.839643666096557),
    ("sim", 3, 0.86, 2.803677, [], 1.63553961147753, 0.839643666096557),
]
# (case, beta, R2max, delta*)
ROBOMIT_DELTA = [
    ("mtcars", 0, 0.9, 1.151136),
    ("sim", 0, 0.885, 1.7412),
    ("sim", 0, 1, 1.474521),
    ("sim", 1, 0.885, 2.14195),
]
TOL = 2e-6

mt = pd.read_csv(ROOT / "tools" / "r_reference" / "output" / "mtcars.csv")
sim = pd.read_csv(ROOT / "tests" / "data" / "oster_v252.csv")
FITS = {"mtcars": (smf.ols("mpg ~ wt", mt).fit(), smf.ols("mpg ~ wt + hp + qsec", mt).fit(), "wt"),
        "sim": (smf.ols("y ~ x", sim).fit(), smf.ols("y ~ x + w1", sim).fit(), "x")}

for case, delta, rmax, bstar, alts, b0, r1 in ROBOMIT_BETA:
    m0, m1, foc = FITS[case]
    o = oster_quiet(m0, m1, foc, r_max=rmax, delta=delta)
    cubic = sorted(([o["beta_star_exact"]] if np.isfinite(o["beta_star_exact"]) else [])
                   + o["beta_star_alternatives"] + o["beta_star_spurious"])
    want = sorted([bstar] + alts)
    check(f"robomit {case} δ={delta} R_max={rmax}: psacalc's choice = {bstar}, and the same real roots of the cubic",
          abs(o["beta_star_psacalc"] - bstar) < TOL and len(cubic) == len(want)
          and all(abs(a - b) < TOL for a, b in zip(cubic, want))
          and abs(o["beta_uncontrolled"] - b0) < 1e-9 and abs(o["r2_controlled"] - r1) < 1e-9,
          (o["beta_star_psacalc"], cubic))
    spur = any(abs(bstar - z) < TOL for z in o["beta_star_spurious"])
    check(f"robomit {case} δ={delta} R_max={rmax}: beta_star_exact is robomit's β* unless that is spurious (then nan)",
          (np.isnan(o["beta_star_exact"]) if spur else abs(o["beta_star_exact"] - bstar) < TOL), o["beta_star_exact"])
for case, beta, rmax, dstar in ROBOMIT_DELTA:
    m0, m1, foc = FITS[case]
    o = oster_quiet(m0, m1, foc, r_max=rmax)
    inp = o["exact_inputs"]
    got = sp._oster_exact_delta(o["beta_uncontrolled"], o["beta_controlled"], o["r2_uncontrolled"],
                                o["r2_controlled"], rmax, beta, inp["sigma_yy"], inp["sigma_xx"], inp["tau_x"])
    check(f"robomit {case} δ* for β={beta}, R_max={rmax}: {dstar}", abs(got - dstar) < TOL, got)
    if beta == 0:
        check(f"robomit {case}: delta_for_zero_exact is that δ* (R_max={rmax})",
              abs(o["delta_for_zero_exact"] - dstar) < TOL, o["delta_for_zero_exact"])


# ---------------------------------------------------------------------- 2. the roots by another route (brentq)
def roots_by_bracketing(o, delta, span=12.0, n=40001):
    """All β with δ*(β) = δ, by sign changes of δ*(β) − δ on a grid and brentq; poles (δ* → ±∞) are discarded."""
    inp = o["exact_inputs"]
    args = (o["beta_uncontrolled"], o["beta_controlled"], o["r2_uncontrolled"], o["r2_controlled"], o["r_max"])
    f = lambda b: sp._oster_exact_delta(*args, b, inp["sigma_yy"], inp["sigma_xx"], inp["tau_x"]) - delta
    bt, sc = o["beta_controlled"], max(abs(o["beta_controlled"]), abs(o["beta_uncontrolled"]), 1.0)
    grid = np.linspace(bt - span * sc, bt + span * sc, n)
    vals = np.array([f(b) for b in grid])
    out = []
    for i in range(n - 1):
        if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) and np.sign(vals[i]) != np.sign(vals[i + 1]):
            r = optimize.brentq(f, grid[i], grid[i + 1], xtol=1e-12)
            if abs(f(r)) < 1e-6:                     # a root, not a pole
                out.append(r)
    return sorted(out)


for case, delta, rmax, *_ in ROBOMIT_BETA:
    m0, m1, foc = FITS[case]
    o = oster_quiet(m0, m1, foc, r_max=rmax, delta=delta)
    solver = sorted(([o["beta_star_exact"]] if np.isfinite(o["beta_star_exact"]) else [])
                    + o["beta_star_alternatives"])
    brk = roots_by_bracketing(o, delta)
    check(f"roots by brentq on δ*(β) = δ = the genuine roots ({case}, δ={delta}, R_max={rmax}): {len(brk)} root(s)",
          len(brk) == len(solver) and all(abs(a - b) < 1e-6 for a, b in zip(brk, solver)), (brk, solver))
    check(f"round trip δ*(β*(δ)) = δ ({case}, δ={delta}, R_max={rmax})",
          all(abs(rt - delta) < 1e-7 for rt in
              [sp._oster_exact_delta(o["beta_uncontrolled"], o["beta_controlled"], o["r2_uncontrolled"],
                                     o["r2_controlled"], rmax, b, *[o["exact_inputs"][k] for k in
                                                                    ("sigma_yy", "sigma_xx", "tau_x")])
               for b in solver]))


# With a single control the cubic has a spurious root at β̃ + (β° − β̃)σ²_x/(σ²_x − τ_x) (by Frisch-Waugh,
# (R̃ − R°)σ²_y = (β° − β̃)²σ²_x τ_x/(σ²_x − τ_x), so numerator and denominator of δ*(β) vanish there): derived by hand
# in the 2026-11-08 study, checked here against what the solver marks. With two controls (mtcars) there is none.
for case in ("sim", "mtcars"):
    m0, m1, foc = FITS[case]
    o = oster_quiet(m0, m1, foc, r_max=0.9, delta=2)
    i = o["exact_inputs"]
    bsp = o["beta_controlled"] + (o["beta_uncontrolled"] - o["beta_controlled"]) * i["sigma_xx"] / (i["sigma_xx"] - i["tau_x"])
    if case == "sim":
        check("one control: the spurious root is the one derived by hand, for several δ and R_max",
              all(len(z["beta_star_spurious"]) == 1 and abs(z["beta_star_spurious"][0] - bsp) < 1e-6
                  for z in (oster_quiet(m0, m1, foc, r_max=rm, delta=dl) for rm in (0.86, 0.9, 1.0)
                            for dl in (0.5, 2, 3))), bsp)
    else:
        check("two controls (mtcars): no spurious root for any δ", all(
            oster_quiet(m0, m1, foc, r_max=0.9, delta=dl)["beta_star_spurious"] == [] for dl in (0.5, 1, 2, 3)))


# ------------------------------------------------------------------------------- 3. population, known δ
def population(n, c1, c2, g1, g2, seed):
    """Oster's proportional selection: w1 observed, w2 not, independent; x = c1 w1 + c2 w2 + u,
    y = 1·x + g1 w1 + g2 w2 + e, so δ = (c2/g2)/(c1/g1) and R_max = R² of the regression with w2."""
    rng = np.random.default_rng(seed)
    w1, w2 = rng.standard_normal((2, n))
    x = c1 * w1 + c2 * w2 + rng.standard_normal(n)
    y = x + g1 * w1 + g2 * w2 + rng.standard_normal(n)
    d = pd.DataFrame({"y": y, "x": x, "w1": w1, "w2": w2})
    m0, m1, mf = smf.ols("y ~ x", d).fit(), smf.ols("y ~ x + w1", d).fit(), smf.ols("y ~ x + w1 + w2", d).fit()
    return m0, m1, mf.rsquared, (c2 / g2) / (c1 / g1)


with warnings.catch_warnings(record=True) as W:
    warnings.simplefilter("always")
    m0, m1, rm, dl = population(400_000, 0.6, 0.6, 1, 1, 52101)
    p1 = sp.oster(m0, m1, "x", r_max=rm, delta=dl)
check("population δ = 1 (n = 400 000): the exact β* recovers the effect within 0.01",
      abs(p1["beta_star_exact"] - 1) < 0.01, p1["beta_star_exact"])
check("population δ = 1, the common case: no warning at all (one admissible root, exact ≈ approximation)",
      p1["warnings"] == [] and len(W) == 0 and not p1["beta_star_exact_unstable"], p1["warnings"])
m0, m1, rm, dl = population(400_000, 0.6, 0.3, 1, 1, 52102)
p05 = oster_quiet(m0, m1, "x", r_max=rm, delta=dl)
check("population δ = 0.5 (n = 400 000): exact within 0.01 of the effect; the approximation misses by > 0.015",
      abs(p05["beta_star_exact"] - 1) < 0.01 and abs(p05["beta_star"] - 1) > 0.015,
      (p05["beta_star_exact"], p05["beta_star"]))
m0, m1, rm, dl = population(2_000_000, 0.3, 0.6, 1, 1, 52103)
p2 = oster_quiet(m0, m1, "x", r_max=rm, delta=dl)
check("population δ = 2, well conditioned (n = 2 000 000): exact within 0.02; the approximation misses by > 0.04; "
      "not flagged unstable", abs(p2["beta_star_exact"] - 1) < 0.02 and abs(p2["beta_star"] - 1) > 0.04
      and not p2["beta_star_exact_unstable"], (p2["beta_star_exact"], p2["beta_star"]))
m0, m1, rm, dl = population(400_000, 0.6, 1.2, 1, 1, 52104)
pu = oster_quiet(m0, m1, "x", r_max=rm, delta=dl)
# Near a double root the pair of genuine roots may also turn complex (it did for this seed; the band first written
# asked only for the unstable flag, and was widened to the other face of the same instability after seeing it).
check("population δ = 2 near a double root: never a β* to report: flagged unstable, or no real solution; told to "
      "report δ for β = 0; never 'report the exact one'",
      (pu["beta_star_exact_unstable"] or np.isnan(pu["beta_star_exact"]))
      and any("δ for β = 0" in w for w in pu["warnings"])
      and not any("report the exact one" in w for w in pu["warnings"]), pu["warnings"])
check("...and the δ for β = 0 there is finite and > 1 (the stable number; the true δ is 2)",
      np.isfinite(pu["delta_for_zero_exact"]) and pu["delta_for_zero_exact"] > 1, pu["delta_for_zero_exact"])
real_pair = None
for s_ in range(52110, 52130):     # the same DGP: the first seed whose pair of genuine roots stays real
    m0, m1, rm, dl = population(400_000, 0.6, 1.2, 1, 1, s_)
    z = oster_quiet(m0, m1, "x", r_max=rm, delta=dl)
    if np.isfinite(z["beta_star_exact"]) and z["beta_star_alternatives"]:
        real_pair = z
        break
check("near a double root with the pair real (first of 20 seeds): flagged unstable, one warning, 'δ for β = 0'",
      real_pair is not None and real_pair["beta_star_exact_unstable"] and len(real_pair["warnings"]) == 1
      and "double root" in real_pair["warnings"][0] and "report the exact one" not in real_pair["warnings"][0],
      None if real_pair is None else real_pair["warnings"])


# ------------------------------------------------------------------------------------- 4. Oster warnings
def oster_case(case, delta, rmax):
    m0, m1, foc = FITS[case]
    return oster_quiet(m0, m1, foc, r_max=rmax, delta=delta)


o = oster_case("mtcars", 2, 0.9)
check("Assumption 3 violated by a genuine root (mtcars, δ=2): one warning, 'do not report', and NOT also "
      "'report the exact one' (the v2.5.2 contradiction)",
      len(o["warnings"]) == 1 and "Assumption 3" in o["warnings"][0] and "Do not report" in o["warnings"][0]
      and np.isfinite(o["beta_star_exact"]) and not o["beta_star_exact_unstable"], o["warnings"])
for delta, rmax in ((2, 1.0), (3, 0.86)):
    o = oster_case("sim", delta, rmax)
    check(f"only the spurious root is real (sim, δ={delta}, R_max={rmax}): β* exact is nan, one warning naming "
          "psacalc/robomit's number as spurious, no identified set",
          np.isnan(o["beta_star_exact"]) and o["identified_set_exact"] is None and len(o["warnings"]) == 1
          and "spurious" in o["warnings"][0] and "2.804" in o["warnings"][0]
          and "report the exact one" not in o["warnings"][0], o["warnings"])
o = oster_case("mtcars", 1, 0.9)
check("δ = 1 with the other root on the forbidden side (mtcars): no 'real roots' warning, the root still listed",
      o["beta_star_alternatives"] and not any("real roots" in w for w in o["warnings"])
      and len(o["warnings"]) == 1 and "differ by more than 10%" in o["warnings"][0], o["warnings"])
o = oster_case("sim", 1, 0.885)
check("δ = 1, other root forbidden, exact ≈ approximation (sim): no warning", o["warnings"] == [], o["warnings"])
o = oster_case("sim", 2, 0.885)
check("two admissible roots, far apart (sim δ=2): one warning naming both and 'Report the choice', not unstable",
      len(o["warnings"]) == 1 and "2 admissible real roots" in o["warnings"][0] and "Report the choice" in o["warnings"][0]
      and "0.5506" in o["warnings"][0] and not o["beta_star_exact_unstable"], o["warnings"])
check("back-compatible keys: beta_star and delta_for_zero are the approximation, unchanged by exact=True",
      all(oster_case("mtcars", 1, 0.9)[k] == oster_quiet(*FITS["mtcars"], r_max=0.9, exact=False)[k]
          for k in ("beta_star", "delta_for_zero", "identified_set")))
o = oster_quiet(*FITS["mtcars"], r_max=0.9, exact=False)
check("exact=False: no exact keys", "beta_star_exact" not in o and "exact_inputs" not in o)

wls0 = sm.WLS(mt["mpg"], sm.add_constant(mt[["wt"]]), weights=mt["hp"]).fit()
wls1 = sm.WLS(mt["mpg"], sm.add_constant(mt[["wt", "hp", "qsec"]]), weights=mt["hp"]).fit()
o = oster_quiet(wls0, wls1, "wt", r_max=0.95)
check("weighted fits: exact not computed, with the reason; the approximation is returned",
      "beta_star_exact" not in o and np.isfinite(o["beta_star"])
      and any("Exact Oster solution not computed" in w and "unweighted OLS" in w for w in o["warnings"]), o["warnings"])
o = oster_quiet(smf.ols("mpg ~ wt + qsec", mt).fit(), smf.ols("mpg ~ wt + qsec + hp", mt).fit(), "wt", r_max=0.9)
check("uncontrolled model with another regressor: exact not computed (psacalc's mcontrol named), approximation kept",
      "beta_star_exact" not in o and any("mcontrol" in w for w in o["warnings"]), o["warnings"])
o = oster_quiet(smf.ols("mpg ~ wt", mt.iloc[:30]).fit(), FITS["mtcars"][1], "wt", r_max=0.9)
check("different samples: exact not computed", "beta_star_exact" not in o
      and any("different samples" in w for w in o["warnings"]))
rng = np.random.default_rng(5)
d = pd.DataFrame({"x": rng.standard_normal(500)})
d["w"] = rng.standard_normal(500)
d["y"] = d["x"] + 0.02 * d["w"] + rng.standard_normal(500)
o = oster_quiet(smf.ols("y ~ x", d).fit(), smf.ols("y ~ x + w", d).fit(), "x")
check("controls that barely raise R²: the δ-is-not-informative warning", any("raise R² by only" in w for w in o["warnings"]))
check("...and NOT in the common population case", not any("raise R² by only" in w for w in p1["warnings"]))


# ------------------------------------------------------------------------------- 5. doctor: the analysis folder
tmp = Path(tempfile.mkdtemp(prefix="v252 "))           # a space in the path, as on the Windows CI job
cdir = tmp / "claude"
cdir.mkdir()
folder = tmp / "analises" / "pasta com espaço"
(folder / ".claude").mkdir(parents=True)
other = tmp / "analises" / "outra"
other.mkdir(parents=True)


def plugin_dir(name, n):
    p = tmp / "plugins" / name
    for k in range(n):
        (p / f"s{k}").mkdir(parents=True, exist_ok=True)
        (p / f"s{k}" / "SKILL.md").write_text("x", encoding="utf-8")
    return str(p)


BIG, SMALL, ECO = plugin_dir("big", 60), plugin_dir("small", 3), plugin_dir("eco", 1)


def fake_run(plugins):
    def run(cmd, timeout):
        if cmd[1:4] == ["plugin", "list", "--json"]:
            return 0, json.dumps(plugins)
        return 0, ""
    return run


claude = lambda n: "/bin/claude" if n == "claude" else None
eco_user = {"id": "econometrics@econometric-skills", "enabled": True, "scope": "user", "installPath": ECO,
            "version": "2.5.1"}

lst = se.skill_listing(cdir, run=fake_run([eco_user, {"id": "small@m", "enabled": True, "scope": "user",
                                                       "installPath": SMALL},
                                           {"id": "big@m", "enabled": True, "scope": "project",
                                            "projectPath": str(other), "installPath": BIG}]),
                       which=claude, cwd=folder)
check("listing: a project-scope install made for ANOTHER folder is left out (its 60 skills not counted)",
      lst["skills"] == 4 and lst["enabled_plugins"] == 2 and "big@m" not in lst["others_enabled"]
      and not lst["crowded"], lst)
lst = se.skill_listing(cdir, run=fake_run([eco_user, {"id": "big@m", "enabled": True, "scope": "project",
                                                       "projectPath": str(folder), "installPath": BIG}]),
                       which=claude, cwd=folder)
check("listing: a project-scope install for THIS folder counts (mutation of the rule above)",
      lst["skills"] == 61 and lst["crowded"] and lst["others_enabled"] == ["big@m"], lst)
lst = se.skill_listing(cdir, run=fake_run([eco_user, {"id": "big@m", "enabled": True, "scope": "user",
                                                       "installPath": BIG},
                                           {"id": "big@m", "enabled": True, "scope": "project",
                                            "projectPath": str(folder), "installPath": BIG}]),
                       which=claude, cwd=folder)
check("listing: the same install folder listed twice is counted once (61 skills, not 121), one plugin id",
      lst["skills"] == 61 and lst["enabled_plugins"] == 2, lst)

crowd = [eco_user, {"id": "big@m", "enabled": True, "scope": "user", "installPath": BIG}]
rep = se.doctor(offline=True, cdir=cdir, env={}, run=fake_run(crowd), which=claude, os_name="posix", cwd=str(folder))
acts = rep["actions"]
check("common folder, crowded listing: the clean folder (workspace.py) is offered, as before",
      rep["listing"]["folder"] == str(folder) and not rep["listing"]["prepared"]
      and any("workspace.py" in a for a in acts), acts)
# What workspace.py writes: this skill (KEEP) true, every other enabled plugin false.
(folder / ".claude" / "settings.json").write_text(
    json.dumps({"enabledPlugins": {"a@m": False, "b@m": False, "econometrics@econometric-skills": True}}),
    encoding="utf-8")
rep = se.doctor(offline=True, cdir=cdir, env={}, run=fake_run(crowd), which=claude, os_name="posix", cwd=str(folder))
acts = rep["actions"]
check("prepared folder (workspace.py's settings), still crowded: says so and does NOT offer another clean folder",
      rep["listing"]["prepared"] and rep["listing"]["plugins_off"] == 2
      and any("already prepared" in a and "do not offer another clean folder" in a for a in acts)
      and not any("workspace.py" in a for a in acts), acts)
# One unrelated plugin turned off by hand is not a prepared folder: the clean-folder offer must survive, or a
# crowded listing loses its only fix (v2.5.2 review, chat 3).
byhand = tmp / "analises" / "a mao"
(byhand / ".claude").mkdir(parents=True)
(byhand / ".claude" / "settings.json").write_text(json.dumps({"enabledPlugins": {"unrelated@m": False}}),
                                                  encoding="utf-8")
rep_bh = se.doctor(offline=True, cdir=cdir, env={}, run=fake_run(crowd), which=claude, os_name="posix",
                   cwd=str(byhand))
check("a folder with one unrelated plugin off by hand is NOT 'prepared': the clean folder is still offered",
      not rep_bh["listing"]["prepared"] and rep_bh["listing"]["plugins_off"] == 1
      and any("workspace.py" in a for a in rep_bh["actions"])
      and not any("already prepared" in a for a in rep_bh["actions"]), rep_bh["actions"])
check("folder_settings reads settings.local.json too, and counts what is on",
      (folder / ".claude" / "settings.local.json").write_text(json.dumps({"enabledPlugins": {"d@m": False}}),
                                                              encoding="utf-8") > 0
      and se.folder_settings(folder)["plugins_off"] == 3 and se.folder_settings(folder)["plugins_on"] == 1)

ver = se.read_skill_version(se.SKILL_DIR)
# Pinned folder, WITH a newer user-scope install to take over: uninstalling the folder's own one is right.
pinned_u = [dict(eco_user, scope="project", projectPath=str(folder), version="2.4.0"), dict(eco_user, version=ver)]
rep = se.doctor(offline=True, cdir=cdir, env={}, run=fake_run(pinned_u), which=claude, os_name="posix",
                cwd=str(folder))
check("folder pinned at an old version with a newer user-scope install: the To-do item names both and uninstalls "
      "the folder's own",
      rep["listing"]["project_install"]["version"] == "2.4.0" and rep["listing"]["user_install"]["version"] == ver
      and any("pinned at 2.4.0" in a and f"user-scope install is {ver}" in a
              and "claude plugin uninstall econometrics@econometric-skills --scope project" in a
              for a in rep["actions"]), rep["actions"])
# Pinned folder and NO user-scope install: uninstalling would remove the skill from the folder (v2.5.2 review).
pinned = [dict(eco_user, scope="project", projectPath=str(folder), version="2.4.0")]
rep = se.doctor(offline=True, cdir=cdir, env={}, run=fake_run(pinned), which=claude, os_name="posix", cwd=str(folder))
check("folder pinned with NO user-scope install: update it in place, and never tell the user to uninstall the "
      "only install that applies there",
      rep["listing"]["project_install"]["version"] == "2.4.0" and "user_install" not in rep["listing"]
      and any("pinned at 2.4.0" in a and "claude plugin update econometrics@econometric-skills --scope project" in a
              and "do not uninstall it" in a for a in rep["actions"])
      and not any("plugin uninstall" in a for a in rep["actions"]), rep["actions"])
same = [dict(eco_user, scope="project", projectPath=str(folder), version=ver)]
rep = se.doctor(offline=True, cdir=cdir, env={}, run=fake_run(same), which=claude, os_name="posix", cwd=str(folder))
check("project install at the current version: no 'pinned' item (mutation)",
      not any("pinned at" in a for a in rep["actions"]), rep["actions"])
rep = se.doctor(offline=True, cdir=cdir, env={}, run=fake_run([dict(pinned[0], projectPath=str(other))]),
                which=claude, os_name="posix", cwd=str(folder))
check("an old project install for ANOTHER folder: no 'pinned' item here", "project_install" not in rep["listing"]
      and not any("pinned at" in a for a in rep["actions"]), rep["actions"])

seen_cwd = []
real_run = se._run
se._run = lambda cmd, timeout, cwd=None: (seen_cwd.append(cwd), (0, json.dumps(crowd)))[1]
try:
    se.skill_listing(cdir, which=claude, cwd=folder)
    rep = se.doctor(offline=True, cdir=cdir, env={"CLAUDE_PROJECT_DIR": str(other)}, which=claude, os_name="posix")
finally:
    se._run = real_run
check("the CLI runs IN the measured folder (a path with spaces)", seen_cwd[:1] == [str(folder)], seen_cwd)
check("no --folder: $CLAUDE_PROJECT_DIR is the folder measured", rep["listing"]["folder"] == str(other)
      and str(other) in seen_cwd, (rep["listing"].get("folder"), seen_cwd))

got = {}
real_doctor = se.doctor
se.doctor = lambda **k: (got.update(k), real_doctor(offline=True, cdir=cdir, env={}, run=fake_run(crowd),
                                                   which=claude, os_name="posix", cwd=k.get("cwd")))[1]
real_print = se.print_doctor
se.print_doctor = lambda rep: None
try:
    with open(os.devnull, "w") as dn:
        so, sys.stdout = sys.stdout, dn
        try:
            se.main(["--doctor", "--offline", "--json", "--folder", str(folder)])
        finally:
            sys.stdout = so
finally:
    se.doctor, se.print_doctor = real_doctor, real_print
check("--folder PATH is passed to doctor", got.get("cwd") == str(folder) and got.get("offline") is True, got)
try:
    with open(os.devnull, "w") as dn:
        se_err, sys.stderr = sys.stderr, dn
        try:
            se.main(["--folder", str(folder)])
            refused = False
        except SystemExit:
            refused = True
        finally:
            sys.stderr = se_err
except Exception:
    refused = False
check("--folder without --doctor is refused, like --json and --offline", refused)


# ------------------------------------------------------------------------- 6. ρ clipping in the pre-period checks
def gap_panel(gap, t0_index):
    T = len(gap)
    tt = np.arange(1, T + 1)
    df = pd.DataFrame({"u": ["A"] * T + ["B"] * T, "t": np.r_[tt, tt], "y": np.r_[gap, np.zeros(T)]})
    df["g"] = np.where(df.u == "A", t0_index + 1, 0)
    return df


T, T0 = 32, 16                    # 16 pre periods: both the trend test (≥ 12) and the step placebo (≥ 16) run
found = None
for s in range(60):               # random-walk pre-period gaps: the first seed whose trend test clips ρ
    rng = np.random.default_rng(7000 + s)
    gap = np.cumsum(rng.standard_normal(T))
    ds = ca.did_single_treated(gap_panel(gap, T0), "u", "t", "y", ["A"], T0 + 1, n_sim=499, seed=0, show=False)
    if ds["pre_trend"]["rho_clipped"] and ds["pre_trend"]["p"] > 0.05:
        found = (s, gap, ds)
        break
check("a random-walk pre-period clips ρ in the trend test within 60 seeds", found is not None)
if found:
    s, gap, ds = found
    check("pre_trend and pre_placebo carry rho_clipped", "rho_clipped" in ds["pre_trend"]
          and "rho_clipped" in ds["pre_placebo"])
    check("clipped trend test: warned as not informative, whatever its p-value",
          any("pre-period trend test found" in w and "not informative" in w for w in ds["warnings"]), ds["warnings"])
    fo = ca.did_falsification(gap_panel(gap, T0), "u", "t", "y", "g", show=False)
    row = fo["table"].loc["pre-period trend (gap series)"]
    check("did_falsification: the clipped trend row says 'not informative' and is listed as uninformative",
          "ρ clipped" in row["note"] and "not informative" in row["note"]
          and "pre-period trend (gap series)" in fo["uninformative"] and "clipped" not in fo["table"].columns,
          (row["note"], fo["uninformative"]))
quiet = None
for s in range(60):               # white-noise gaps: the first seed with neither pre-period check clipped
    rng = np.random.default_rng(8000 + s)
    gap = rng.standard_normal(T)
    ds = ca.did_single_treated(gap_panel(gap, T0), "u", "t", "y", ["A"], T0 + 1, n_sim=499, seed=0, show=False)
    if not ds["pre_trend"]["rho_clipped"] and not ds["pre_placebo"]["rho_clipped"]:
        quiet = (gap, ds)
        break
check("white-noise gap without clipping found", quiet is not None)
if quiet:
    gap, ds = quiet
    fo = ca.did_falsification(gap_panel(gap, T0), "u", "t", "y", "g", show=False)
    check("no clipping: no clipping warning and no 'ρ clipped' note (mutation)",
          not any("pre-period trend test found" in w or "pre-period placebo found" in w for w in ds["warnings"])
          and not any("ρ clipped" in str(n) for n in fo["table"]["note"]), ds["warnings"])


# ---------------------------------------------------------------------- 7. exact randomization p, no double count
# 5 units, 2 periods, 1 treated: 5 assignments. Gains Δ = (10, 1, 2, 3, 4). With one treated unit j the two-period
# TWFE DiD is Δ_j − mean of the others = (5Δ_j − 20)/4 = (7.5, −3.75, −2.5, −1.25, 0). Worked by hand:
#   treated 0: only |7.5| ≥ 7.5                 → p = 1/5 = 0.2   (the old count gave 2/6 = 0.333)
#   treated 1: |7.5| and |3.75| ≥ 3.75          → p = 2/5 = 0.4   (old: 3/6 = 0.5)
def tiny(treated):
    gains = [10, 1, 2, 3, 4]
    rows = [{"u": i, "t": t, "y": (gains[i] if t == 2 else 0.0), "g": (2 if i == treated else 0)}
            for i in range(5) for t in (1, 2)]
    return pd.DataFrame(rows)


for tr, want, old in ((0, 0.2, 2 / 6), (1, 0.4, 3 / 6)):
    r = ca.randomization_inference(tiny(tr), "u", "t", "y", "g", n_perm=999, ci=False, seed=0)
    check(f"exact randomization p (treated {tr}): {want} by hand, the observed assignment counted once",
          r["exact"] and r["n_perm"] == 5 and abs(r["p"] - want) < 1e-12 and abs(r["p"] - old) > 0.05, r["p"])

    def gain_stat(d):
        g = d.pivot(index="u", columns="t", values="y")
        dg = (g[2] - g[1])
        tr_u = d.loc[d["g"] > 0, "u"].unique()
        return float(dg[dg.index.isin(tr_u)].mean() - dg[~dg.index.isin(tr_u)].mean())

    r2 = ca.randomization_inference(tiny(tr), "u", "t", "y", "g", stat=gain_stat, n_perm=999, seed=0)
    check(f"exact p with a user statistic (treated {tr}): the same {want}", abs(r2["p"] - want) < 1e-12, r2["p"])

rng = np.random.default_rng(3)
big = pd.DataFrame([{"u": i, "t": t, "y": rng.standard_normal() + (3.0 if (i == 0 and t == 2) else 0.0),
                     "g": (2 if i == 0 else 0)} for i in range(12) for t in (1, 2)])
r = ca.randomization_inference(big, "u", "t", "y", "g", n_perm=5, ci=False, seed=0)
check("random draws (12 assignments > n_perm + 1): still (1 + hits)/(n_perm + 1), so p ≥ 1/6",
      not r["exact"] and abs(r["p"] * 6 - round(r["p"] * 6)) < 1e-9 and r["p"] >= 1 / 6 - 1e-12, r["p"])


# ------------------------------------------------- 8. H: the To-do commands use the complete interpreter (chat 3)
# v2.5.0.1 NIT: run from an incomplete Python (3.12, no packages) while another one (3.14) had everything, the doctor
# told the user to run --enable-autoupdate and workspace.py with 'py -3.12'. They work (standard library only), but
# the user should keep one interpreter: the one the first item tells them to use.
def rep_h(core_missing, interps, supported=True, version="3.12.10"):
    return {"python": {"exe": "/usr/bin/python3.12", "version": version, "supported": supported,
                       "run_with": "py -3.12"},
            "core_missing": core_missing, "interpreters": interps,
            "latest": {"version": None, "status": "not checked (--offline)"}, "copies": [],
            "skill": {"dir": str(se.SKILL_DIR), "version": ver},
            "autoupdate": {"registered": True, "on": False, "blocked_by": [], "configured": False, "source": "x"},
            "listing": {"checked": True, "crowded": True, "prepared": False, "skills": 900, "enabled_plugins": 80,
                        "folder": str(folder)},
            "claude_dir": str(cdir)}


full = {"exe": "/opt/py314/python3.14", "version": "3.14.8", "supported": True, "missing_core": [],
        "run_with": "py -3.14"}
au_ws = lambda acts: [a for a in acts if "--enable-autoupdate" in a or "workspace.py" in a]
acts = se.doctor_actions(rep_h(["numpy>=2.1"], [full]), os_name="nt")
check("incomplete Python, a complete 3.14 found: the first item names its command, and the auto-update and "
      "workspace.py commands use it, never 'py -3.12'",
      "(py -3.14)" in acts[0] and len(au_ws(acts)) == 2 and all("py -3.14 \"" in a for a in au_ws(acts))
      and not any("py -3.12" in a for a in acts), acts)
acts = se.doctor_actions(rep_h([], [full]), os_name="nt")
check("complete Python: its own command stays in every item, even with a newer complete one around (mutation)",
      len(au_ws(acts)) == 2 and all("py -3.12 \"" in a for a in au_ws(acts)) and not any("py -3.14" in a for a in acts),
      acts)
acts = se.doctor_actions(rep_h(["numpy>=2.1"], [dict(full, missing_core=["scipy>=1.13"])]), os_name="nt")
check("incomplete Python and no complete one: install item, and the commands keep this Python",
      any(a.startswith("install the core packages") for a in acts)
      and all("py -3.12 \"" in a for a in au_ws(acts)), acts)
acts = se.doctor_actions(rep_h([], [full], supported=False, version="3.9.7"), os_name="nt")
check("Python too old, a complete one found: 'run scripts with py -3.14' and the commands use it",
      "run scripts with py -3.14" in acts[0] and all("py -3.14 \"" in a for a in au_ws(acts)), acts)
old_shape = {k: v for k, v in full.items() if k != "run_with"}
acts = se.doctor_actions(rep_h(["numpy>=2.1"], [old_shape]), os_name="nt")
check("an interpreter entry without run_with (report from an older doctor): falls back to its path",
      all("/opt/py314/python3.14 \"" in a for a in au_ws(acts)), acts)

exe_file = tmp / "bin" / "python3.14"
exe_file.parent.mkdir()
exe_file.write_text("", encoding="utf-8")
real_cand, real_probe = se.python_candidates, se.probe_interpreter
se.python_candidates = lambda **k: [str(exe_file)]
se.probe_interpreter = lambda cmd, run=None: {"exe": str(exe_file), "version": "3.14.8", "missing_core": [],
                                              "supported": True}
try:
    w3 = lambda n: str(exe_file) if n == "python3" else claude(n)
    rep = se.doctor(offline=True, cdir=cdir, env={}, run=fake_run(crowd), which=w3, os_name="posix", cwd=str(folder))
    rep2 = se.doctor(offline=True, cdir=cdir, env={}, run=fake_run(crowd), which=claude, os_name="posix",
                     cwd=str(folder))
finally:
    se.python_candidates, se.probe_interpreter = real_cand, real_probe
check("doctor: each other interpreter carries the command that starts it ('python3' when that is it on the PATH)",
      [i.get("run_with") for i in rep["interpreters"]] == ["python3"], rep["interpreters"])
check("doctor: not on the PATH -> its full path, quoted (the folder has a space), never a bare name",
      [i.get("run_with") for i in rep2["interpreters"]] == [f'"{exe_file}"'] and " " in str(exe_file),
      rep2["interpreters"])

# ------------------------------------------------------------- 9. E: the identification checklist (chat 3)
import importlib  # noqa: E402
import re  # noqa: E402

SK = ROOT / "econometrics"
skill_md = (SK / "SKILL.md").read_text(encoding="utf-8")
sec3 = skill_md[skill_md.index("## 3. Consultant workflow"):skill_md.index("## 4. Non-negotiable rules")]
step0 = next(line for line in sec3.splitlines() if line.startswith("0. **Study card**"))
check("SKILL.md §3, step 0: causal questions fill the threats with the checklist in workflow.md §0; forecasts skip it",
      "identification checklist in `references/workflow.md` §0" in step0 and "forecasts skip it" in step0
      and step0.startswith("0. **Study card** before any estimate: question, estimand, design"), step0)
wf = (SK / "references" / "workflow.md").read_text(encoding="utf-8")
m = re.search(r"### Identification checklist \(causal questions only\)\n(.*?)\n## 1\.", wf, re.S)
block = m.group(1) if m else ""
check("workflow.md §0 holds the checklist, with the six threats of the plan and the skip rule",
      m is not None and all(k in block for k in ("Bad controls", "Weak or invalid instrument", "Spillovers",
                                                  "Inference at the wrong level", "Few treated units",
                                                  "Unobserved confounding")) and "Forecasts" in block)
named = sorted(set(re.findall(r"`(\w+)\.(\w+)(?:\(|`)", block)))
missing = []
for mod, fn in named:
    try:
        if not callable(getattr(importlib.import_module(mod), fn, None)):
            missing.append(f"{mod}.{fn}")
    except ImportError:
        missing.append(f"{mod}.{fn}")
check("every function the checklist names exists in the skill (advice must match the tools)",
      len(named) >= 9 and not missing, (named, missing))
check("the check that every named function exists would catch a wrong name (mutation)",
      not callable(getattr(importlib.import_module("causal"), "synthetic_control", None)))


def fm(path):
    t = path.read_text(encoding="utf-8")
    head = t.split("---")[1]
    return {k.strip(): v.strip().strip("'") for k, v in (ln.split(":", 1) for ln in head.strip().splitlines())}


R = SK / "evals" / "routing"
fc, fcf = fm(R / "forecast-no-checklist" / "graders" / "answer.md"), fm(R / "forecast-no-checklist" / "graders" /
                                                                       "forecast.md")
neg = re.compile(fc["pattern"], re.I)
check("forecast case: the answer grader asserts the checklist is ABSENT (match: not_contains)",
      fc["type"] == "regex" and fc.get("match") == "not_contains" and fc.get("target") == "last_message")
check("forecast case: a forecasting answer passes, an answer that runs the checklist fails",
      not neg.search("Use ETS e SARIMA com validação fora da amostra (origem móvel) contra o ingênuo sazonal.")
      and neg.search("Antes, o checklist de identificação: controles ruins, spillovers e Oster.")
      and re.search(fcf["pattern"], "Compare com o ingênuo sazonal fora da amostra.", re.I))
ca_g = re.compile(fm(R / "causal-checklist-post-treatment" / "graders" / "answer.md")["pattern"], re.I)
check("causal case: passes only when the answer flags the post-treatment variable AND unobserved selection",
      ca_g.search("Funcionários depois da adesão é pós-tratamento. Como as empresas escolheram, faça a "
                  "sensibilidade de Oster.")
      and not ca_g.search("Funcionários depois da adesão é pós-tratamento: não use como controle.")
      and not ca_g.search("Faça a sensibilidade a confundidores não observados."))
ref = (R / "two-groups" / "graders" / "skill-fired.md").read_text(encoding="utf-8")
check("both new cases use the standard skill-fired grader",
      all((R / c / "graders" / "skill-fired.md").read_text(encoding="utf-8") == ref
          for c in ("forecast-no-checklist", "causal-checklist-post-treatment")))
p0 = (R / "causal-checklist-post-treatment" / "prompt.md").read_text(encoding="utf-8").lower()
check("causal case does not ask about controls (unlike training-controls): the checklist must bring it up",
      "controle" not in p0 and "depois da adesão" in p0 and "escolheram participar" in p0)

# ----------------------------------------------------- 10. H text: §7 within its v2.5.1 size, and the Oster rule
sec7 = skill_md[skill_md.index("## 7. Environment"):skill_md.index("## 8.")]
check("SKILL.md §7 did not grow (v2.5.1: 2,859 characters)", len(sec7) <= 2859, len(sec7))
check("§7: run the doctor from the analysis folder or --folder; one question at the end; non-blocking items do not "
      "hold up the analysis; --offline only without network; the app copy is updated by the user",
      all(k in sec7 for k in ("from the analysis folder (or pass `--folder PATH`)", "at the end of the reply",
                              "non-blocking items", "never hold up the analysis", "`--offline` only without network",
                              "which the user updates in Customize → Skills",
                              "the complete one the \"To do\" names")), sec7)
rules = (SK / "references" / "interpretation-rules.md").read_text(encoding="utf-8")
r252 = rules[rules.index("## Rules added in v2.5.2"):] if "## Rules added in v2.5.2" in rules else ""
check("interpretation rules v2.5.2: δ depends on R_max and is not a probability; 1.3·R̃ is a calibration",
      all(k in r252 for k in ("not a probability", "depends on R_max", "calibration", "not a fact about the data")))
check("interpretation rules v2.5.2: report the exact solution, and δ for β = 0 when β* is not reportable",
      all(k in r252 for k in ("beta_star_exact", "Assumption 3", "spurious", "beta_star_exact_unstable",
                              "report δ for β = 0")))
ols = (SK / "references" / "ols-regression.md").read_text(encoding="utf-8")
swe = (SK / "references" / "software-equivalents.md").read_text(encoding="utf-8")
check("ols-regression.md §14 and software-equivalents no longer call oster an approximation only",
      "solves the exact version" not in ols and "beta_star_exact" in ols and "robomit" in ols
      and '"d")` (approximation) |' not in swe and "beta_star_exact" in swe)


# ------------------------------- 11. fixes from the independent review, round 1 (chat 3, 2026-11-09)
# Both were absolute floors inside tests that have to be relative, and the exact randomization p-value whose
# enumeration did not contain the observed assignment. Each check below fails with the code as it was.
from math import comb as _comb, factorial as _fact  # noqa: E402

sim_d = pd.read_csv(ROOT / "tests" / "data" / "oster_v252.csv")


def oster_scaled(xs=1.0, ys=1.0, delta=2.0, rmax=0.885):
    df = sim_d.copy()
    df["x"], df["y"] = df["x"] * xs, df["y"] * ys
    u = smf.ols("y ~ x", df).fit()
    c = smf.ols("y ~ x + w1", df).fit()
    with warnings.catch_warnings(record=True) as W:
        warnings.simplefilter("always")
        o = sp.oster(u, c, "x", r_max=rmax, delta=delta)
    return o, [str(w.message) for w in W]


ref, ref_w = oster_scaled()
scales = [1e-2, 1e-3, 3e-4, 1e-6]
rows = [oster_scaled(xs=s) for s in scales]
check("Oster exact is equivariant in the units of x: β* scales as 1/scale(x) (sd(x) down to 1.6e-6), the "
      "spurious root is still found and the warnings do not change",
      all(abs(o["beta_star_exact"] * s - ref["beta_star_exact"]) < 1e-8 * abs(ref["beta_star_exact"])
          and len(o["beta_star_spurious"]) == len(ref["beta_star_spurious"]) == 1
          and len(W) == len(ref_w) == 1 for (o, W), s in zip(rows, scales)),
      [(f"{s:.0e}", o["beta_star_exact"] * s, len(o["beta_star_spurious"]), len(W)) for (o, W), s in zip(rows, scales)])
# The old test was abs(a3) <= 1e-12 * max(1.0, abs(tx*sxx)); a3 has the units of var(x) squared, so the floor
# of 1.0 fired once sd(x) was small. This asserts the old condition WOULD have fired here (mutation): with it,
# the quadratic branch was taken, β* was 3.1% off, the spurious filter was off and the warning said
# "report the exact one". Written from the units, not from the implementation's expression.
o3, _ = oster_scaled(xs=1e-3)
sxx3, tx3 = o3["exact_inputs"]["sigma_xx"], o3["exact_inputs"]["tau_x"]
a3 = (2.0 - 1) * tx3 * (sxx3 - tx3)
check("mutation: at sd(x) ~ 1.6e-3 the old absolute floor would have degenerated the cubic, the relative test "
      "does not", abs(a3) <= 1e-12 * max(1.0, abs(tx3 * sxx3)) and abs(a3) > 1e-12 * sxx3 ** 2,
      (a3, 1e-12 * max(1.0, abs(tx3 * sxx3)), 1e-12 * sxx3 ** 2))
worst = [oster_scaled(xs=s, delta=2.0, rmax=1.0) for s in [1.0, 1e-3, 1e-6]]
check("the case whose only real root is spurious stays nan in every unit of x, and never says "
      "'report the exact one'",
      all(np.isnan(o["beta_star_exact"]) and len(o["beta_star_spurious"]) == 1 and len(W) == 1
          and "spurious" in W[0] and "report the exact one" not in W[0] for o, W in worst),
      [(o["beta_star_exact"], W) for o, W in worst])
ys_rows = [oster_scaled(ys=y) for y in [1e-6, 1e-9]]
check("the spurious-root tolerance is relative to the coefficients: tiny units of y do not turn every root "
      "spurious (it used to give 3 spurious roots and β* = nan below |β̃| ~ 1e-8)",
      all(len(o["beta_star_spurious"]) == 1 and len(o["beta_star_alternatives"]) == 1
          and abs(o["beta_star_exact"] / y - ref["beta_star_exact"]) < 1e-6 * abs(ref["beta_star_exact"])
          for (o, _), y in zip(ys_rows, [1e-6, 1e-9])),
      [(len(o["beta_star_spurious"]), len(o["beta_star_alternatives"])) for o, _ in ys_rows])


def ri_panel(dates, eff=20.0, T=8, N=6, seed=7):
    rng = np.random.default_rng(seed)
    rows_, g = [], {u: dates.get(u, 0) for u in range(N)}
    for u in range(N):
        a = rng.normal(0, 1)
        for t in range(1, T + 1):
            dd = 1.0 if g[u] and t >= g[u] else 0.0
            rows_.append({"unit": u, "time": t, "y": a + 0.3 * t + eff * dd + rng.normal(0, 1), "g": g[u]})
    return pd.DataFrame(rows_)


def arrangements(dates):
    n = _fact(len(dates))
    for c in np.unique(np.asarray(dates), return_counts=True)[1]:
        n //= _fact(int(c))
    return n


for label, dates, n_arr in [("one date", {0: 5, 1: 5}, 1), ("two dates", {0: 3, 1: 5}, 2),
                            ("three dates", {0: 3, 1: 5, 2: 7}, 6)]:
    runs = [ca.randomization_inference(ri_panel(dates), "unit", "time", "y", "g", n_perm=999, ci=False, seed=s)
            for s in range(12)]
    space = _comb(6, len(dates)) * n_arr
    check(f"exact randomization p, {label}: the space is C(6,k) x {n_arr} arrangements and p is never below "
          "1/space (it came out 0.0 in 7 of these 12 seeds with staggered dates)",
          arrangements(list(dates.values())) == n_arr and all(
              r["exact"] and r["possible_assignments"] == space and r["p"] >= 1 / space - 1e-12 and r["p"] > 0
              for r in runs),
          (space, sorted({round(r["p"], 5) for r in runs})))
runs1 = [ca.randomization_inference(ri_panel({0: 5, 1: 5}), "unit", "time", "y", "g", n_perm=999, ci=False,
                                    seed=s)["p"] for s in range(4)]
check("with one adoption date nothing changed: the enumeration is the 15 sets and p = hits / 15, so item I "
      "(the observed assignment counted twice) stays fixed",
      all(abs(p - 1 / 15) < 1e-12 for p in runs1), runs1)


# --------------------- 12. the rest of the independent review, round 1 (chat 3): warnings that mislead
import contextlib  # noqa: E402
import io  # noqa: E402

# psacalc and robomit pick the root closest to β̃ without the spurious check, so they can print the spurious
# root while a genuine solution exists (47 of 342 (δ, R_max) points on this data). The user comparing with
# Stata has to be told, and the docstring used to claim the two only differ when β* is nan.
u_s = smf.ols("y ~ x", sim_d).fit()
c_s = smf.ols("y ~ x + w1", sim_d).fit()
with warnings.catch_warnings(record=True) as W:
    warnings.simplefilter("always")
    o_ps = sp.oster(u_s, c_s, "x", r_max=0.84, delta=-2.0)
w_ps = [str(x.message) for x in W]
check("when psacalc and robomit would print the spurious root while a genuine solution exists, a warning names "
      "both numbers (the docstring used to say the two differ only when β* is nan)",
      np.isfinite(o_ps["beta_star_exact"]) and o_ps["beta_star_spurious"]
      and abs(o_ps["beta_star_psacalc"] - o_ps["beta_star_spurious"][0]) < 1e-9
      and any("psacalc and robomit would report" in m and "spurious" in m for m in w_ps),
      (o_ps["beta_star_exact"], o_ps["beta_star_psacalc"], w_ps))
check("that warning stays away from the common case (δ = 1, default R_max): psacalc agrees there",
      not any("would report" in m for m in oster_quiet(u_s, c_s, "x")["warnings"]),
      oster_quiet(u_s, c_s, "x")["warnings"])
buf = io.StringIO()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with contextlib.redirect_stdout(buf):
        o_sh = sp.oster(u_s, c_s, "x", r_max=0.84, delta=-2.0, show=True)
shown = buf.getvalue()
check("show=True prints the warnings next to the numbers: a β* the warnings say not to report never appears "
      "bare (warnings are filtered in the CI and in the suites, so warnings.warn alone is invisible)",
      o_sh["warnings"] and all("  WARNING: " + m in shown for m in o_sh["warnings"])
      and shown.count("WARNING") == len(o_sh["warnings"]), shown[-300:])

# ρ clipped at 0.95 inflates the SE, so a check that rejects anyway is conservative and its finding stands;
# only a non-rejection is uninformative. The warning used to say "not informative, whatever its p-value",
# contradicting the pre-trend warning next to it and the matrix's own `informative` column.
def clip_panel(trend, seed, T0, T1=8):
    rng = np.random.default_rng(seed)
    T = T0 + T1
    gap = np.cumsum(rng.normal(0, 1, T)) + trend * np.arange(T)
    base = rng.normal(0, 1, T)
    rows_ = []
    for t in range(T):
        rows_.append({"u": "ctrl", "t": t + 1, "y": base[t]})
        rows_.append({"u": "trt", "t": t + 1, "y": base[t] + gap[t]})
    return pd.DataFrame(rows_)


rej_res = ca.did_single_treated(clip_panel(3.0, 0, 12), "u", "t", "y", "trt", 13, show=False)
rej_w = rej_res["warnings"]                      # the output contract: every entry is relayed to the user
pt = rej_res["pre_trend"]
check("a clipped pre-trend test that REJECTS: the warning says the clip inflates the SE and the finding stands, "
      "and never calls it uninformative",
      pt["rho_clipped"] and pt["p"] <= 0.05
      and any("inflates its standard error" in m and "stands" in m for m in rej_w)
      and not any("not informative" in m for m in rej_w), (pt["p"], rej_w))
norej = ca.did_single_treated(clip_panel(0.0, 3, 12), "u", "t", "y", "trt", 13, show=False)
norej_w = norej["warnings"]
pt2 = norej["pre_trend"]
check("a clipped pre-trend test that does NOT reject: still called uninformative (the other half of the rule)",
      pt2["rho_clipped"] and pt2["p"] > 0.05
      and any("did not reject, but it is not informative" in m for m in norej_w)
      and not any("inflates its standard error" in m for m in norej_w), (pt2["p"], norej_w))


print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
