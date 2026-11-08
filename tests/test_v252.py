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
(folder / ".claude" / "settings.json").write_text(json.dumps({"enabledPlugins": {"a@m": False, "b@m": False,
                                                                                  "c@m": True}}), encoding="utf-8")
rep = se.doctor(offline=True, cdir=cdir, env={}, run=fake_run(crowd), which=claude, os_name="posix", cwd=str(folder))
acts = rep["actions"]
check("prepared folder (workspace.py's settings), still crowded: says so and does NOT offer another clean folder",
      rep["listing"]["prepared"] and rep["listing"]["plugins_off"] == 2
      and any("already prepared" in a and "do not offer another clean folder" in a for a in acts)
      and not any("workspace.py" in a for a in acts), acts)
check("folder_settings reads settings.local.json too",
      (folder / ".claude" / "settings.local.json").write_text(json.dumps({"enabledPlugins": {"d@m": False}}),
                                                              encoding="utf-8") > 0
      and se.folder_settings(folder)["plugins_off"] == 3)

ver = se.read_skill_version(se.SKILL_DIR)
pinned = [dict(eco_user, scope="project", projectPath=str(folder), version="2.4.0")]
rep = se.doctor(offline=True, cdir=cdir, env={}, run=fake_run(pinned), which=claude, os_name="posix", cwd=str(folder))
check("folder pinned at an old version by its own project install: a To-do item with the uninstall command",
      rep["listing"]["project_install"]["version"] == "2.4.0"
      and any("pinned at 2.4.0" in a and "claude plugin uninstall econometrics@econometric-skills --scope project" in a
              for a in rep["actions"]), rep["actions"])
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


print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
