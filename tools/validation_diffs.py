"""Largest observed difference against the external references of VALIDATION.md (v2.6.7).

For every level-A function whose reference is external (a published value, or the output of R, Stata, DoubleML,
csdid or lifelines), this script redoes the comparison the test makes and records the largest |difference| and the
tolerance the test allows. Comparisons against a closed form computed inside a test are not measured here.

Two routes:
  - tests that compare through a helper (close(a, b, tol) or same(mine, theirs, tol)) are run as they are, with the
    helper and check() wrapped: each check gets the differences of the helper calls made for it. The tests are not
    modified;
  - the other comparisons are redone below with the same data, calls and references as the test that cites them.

Data, designs and reference values are read from the tests themselves (by AST or by the test's own lines), so a
change in a test changes the measurement.

Run from the repository root:  python tools/validation_diffs.py [--check]
Writes tools/validation_diffs.json and prints a table; with --check it writes nothing and exits 1 if a comparison is
outside its tolerance (used in CI). Needs the packages of requirements-dev.txt (csdid included).
"""
import ast
import contextlib
import datetime
import io
import json
import sys
import warnings
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "econometrics" / "scripts"))
warnings.filterwarnings("ignore")

OUT = {}   # function -> list of (abs or relative diff, tolerance, kind, what)
CHECK = "--check" in sys.argv[1:]


def test_src(fname):
    return (ROOT / "tests" / fname).read_text(encoding="utf-8")


def literal(fname, var):
    """The literal value assigned to `var` at the top level of a test."""
    tree = ast.parse(test_src(fname))
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", "") == var)


def function_src(fname, name):
    """The source of a top-level function of a test."""
    src = test_src(fname)
    return next(ast.get_source_segment(src, n) for n in ast.parse(src).body
                if isinstance(n, ast.FunctionDef) and n.name == name)


def lines_between(fname, first, last):
    """The test's own lines from the one starting with `first` to the one starting with `last` (both included)."""
    lines = test_src(fname).splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith(first))
    j = next(k for k, ln in enumerate(lines) if k >= i and ln.startswith(last))
    return "\n".join(lines[i:j + 1])


def add(fn, diff, tol, what, kind="abs"):
    OUT.setdefault(fn, []).append((float(diff), float(tol), kind, what))


# ------------------------------------------------------------------ 1. tests that compare through a helper
GENERIC = {
    "test_textbook_replication.py": ("close", {
        "diagnostics.ols_diagnostics": ["W8.4"], "diagnostics.endogeneity_test": ["W15.7"],
        "diagnostics.overid_test": ["W15.8"], "diagnostics.tobit": ["W17.2"]}),
    "test_v242.py": ("close", {
        "diagnostics.iv_report": ["2SLS reproduces Wooldridge Example 15.1"],
        "discrete_choice.binary_effects": ["LPM reproduces Wooldridge Example 17.1"],
        "discrete_choice.heckman": ["Heckit reproduces"],
        "specification.linear_restriction_test": ["exclusion F equals statsmodels f_test"],
        "specification.prais_winsten": ["Prais-Winsten reproduces"]}),
    "test_specification_tools.py": ("close", {
        "diagnostics.count_diagnostics": ["W17.3 Pearson dispersion"], "specification.fgls_het": ["W8.7 FGLS"],
        "specification.prediction_intervals": ["W6.5", "W6.6"], "specification.long_run_effect": ["W10.4 LRP"],
        "specification.standardized_coefficients": ["W6.1 standardized coefficients"],
        "specification.robustness_value": ["Cinelli-Hazlett Darfur"]}),
    "test_v26.py": ("same", {
        "specification.ovb_bounds": ["R Darfur", "R sim"]}),
}


def run_instrumented(fname, helper):
    """Run tests/<fname> with <helper> and check wrapped; return {check name: [(diff, tol, kind)]}."""
    path = ROOT / "tests" / fname
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = []
    for node in tree.body:
        body.append(node)
        if isinstance(node, ast.FunctionDef) and node.name in (helper, "check"):
            body.append(ast.parse(f"{node.name} = __vd_wrap_{'helper' if node.name == helper else 'check'}"
                                  f"({node.name})").body[0])
    tree.body = body
    pending, per_check = [], {}

    def wrap_helper(f):
        def inner(a, b, tol=1e-8, *rest, **kw):
            try:
                if helper == "same":
                    a_, b_ = float(a), float(b)
                    if np.isfinite(a_) and np.isfinite(b_):
                        rel = abs(b_) > 1e-12
                        pending.append((abs(a_ - b_) / (abs(b_) if rel else 1.0), tol, "rel" if rel else "abs"))
                else:
                    d = np.max(np.abs(np.asarray(a, float) - np.asarray(b, float)))
                    pending.append((float(d), tol, "abs"))
            except (TypeError, ValueError):
                pass
            return f(a, b, tol, *rest, **kw)
        return inner

    def wrap_check(f):
        def inner(name, cond, *rest, **kw):
            per_check.setdefault(str(name), []).extend(pending)
            pending.clear()
            return f(name, cond, *rest, **kw)
        return inner

    g = {"__file__": str(path), "__name__": "__main__", "__vd_wrap_helper": wrap_helper, "__vd_wrap_check": wrap_check}
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            exec(compile(tree, str(path), "exec"), g)
        except SystemExit:
            pass
    return per_check


for fname, (helper, fmap) in GENERIC.items():
    pc = run_instrumented(fname, helper)
    for fn, prefixes in fmap.items():
        hits = [(nm, x) for nm, xs in pc.items() for x in xs if any(nm.startswith(p) for p in prefixes)]
        assert hits, f"{fn}: no comparison recorded in {fname} for {prefixes}"
        for nm, (d, tol, kind) in hits:
            add(fn, d, tol, f"{fname}: {nm}", kind)

# ------------------------------------------------------------------ 2. comparisons redone here
import causal as ca  # noqa: E402
import estimation as est  # noqa: E402
import inference as inf  # noqa: E402
import risk  # noqa: E402
import specification as sp  # noqa: E402
import statsmodels.api as sm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
import survival  # noqa: E402
from scipy.special import gammaln  # noqa: E402


def quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*a, **k)


# estimation.mle: Wooldridge Example 17.3 (test_estimation.py)
import wooldridge as woo  # noqa: E402

cr = woo.data("crime1")
xs = ["pcnv", "avgsen", "tottime", "ptime86", "qemp86", "inc86", "black", "hispan", "born60"]
X = sm.add_constant(cr[xs]).to_numpy(float)
yv = cr["narr86"].to_numpy(float)
m_n = est.mle(lambda b, X, y: y * (X @ b) - np.exp(X @ b) - gammaln(y + 1), np.zeros(X.shape[1]), args=(X, yv),
              names=["const"] + xs, method="newton")
book = literal("test_estimation.py", "book")
for k, v in book.items():
    add("estimation.mle", abs(m_n["params"][k] - v), 0.0011, f"test_estimation.py: W17.3 {k}")

# did_imputation and twfe_weights: R (test_v250.py)
REF = json.loads((ROOT / "tools" / "r_reference" / "output" / "did_v250_reference.json").read_text(encoding="utf-8"))
pan = pd.read_csv(ROOT / "tests" / "data" / "did_v250_panel.csv")
o = ca.did_imputation(pan, "id", "t", "y", "g", show=False)
r_st, r_cl, r_ub = REF["bjs_static"][0], REF["bjs_static_state"][0], REF["bjs_static_unbalanced"][0]
add("causal.did_imputation", abs(o["overall"]["estimate"] - r_st["estimate"]), 1e-6, "static ATT estimate")
add("causal.did_imputation", abs(o["overall"]["se"] - r_st["std.error"]), 1e-8, "static ATT SE")
oc = ca.did_imputation(pan, "id", "t", "y", "g", cluster="state", show=False)["overall"]
add("causal.did_imputation", abs(oc["se"] - r_cl["std.error"]), 1e-8, "SE clustered by state")
ou = ca.did_imputation(pan[pan.keep_unb == 1], "id", "t", "y", "g", show=False)["overall"]
add("causal.did_imputation", abs(ou["estimate"] - r_ub["estimate"]), 1e-6, "unbalanced estimate")
add("causal.did_imputation", abs(ou["se"] - r_ub["std.error"]), 1e-8, "unbalanced SE")
ev = {int(row["term"]): row for row in REF["bjs_event"] if int(row["term"]) >= 0}
for h in range(7):
    add("causal.did_imputation", abs(o["dynamic"].loc[h, "att"] - ev[h]["estimate"]), 1e-6, f"event h = {h} estimate")
    add("causal.did_imputation", abs(o["dynamic"].loc[h, "se"] - ev[h]["std.error"]), 1e-7, f"event h = {h} SE")
for D, key in (("d", "twfe_weights_binary"), ("dose", "twfe_weights_dose")):
    R = {k: (next(iter(v[0].values())) if isinstance(v, list) and v and isinstance(v[0], dict) else v)
         for k, v in REF[key].items() if k in ("beta", "sensibility", "sum_plus", "sum_minus", "nr_plus", "nr_minus")}
    w = ca.twfe_weights(pan, "id", "t", D, y="y", show=False)
    for mine, theirs in (("beta", "beta"), ("sum_positive", "sum_plus"), ("sum_negative", "sum_minus"),
                         ("sigma_fe", "sensibility")):
        add("causal.twfe_weights", abs(w[mine] - R[theirs]), 1e-9, f"{D}: {mine}")
    add("causal.twfe_weights", abs(w["n_positive"] - R["nr_plus"]) + abs(w["n_negative"] - R["nr_minus"]), 0,
        f"{D}: counts (exact)")

# att_gt: csdid (test_did_covariates.py, section 2; the panel builder is the test's)
try:
    from csdid.att_gt import ATTgt
    src = (ROOT / "tests" / "test_did_covariates.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    panel_src = next(ast.get_source_segment(src, n) for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name == "panel")
    ns = {"np": np, "pd": pd}
    exec(panel_src, ns)
    q = ns["panel"](1)
    q["x0"] = q.groupby("id")["x"].transform("first")
    q["x0sq"] = q["x0"] ** 2
    for ctrl, cs_ctrl in (("never", "nevertreated"), ("notyet", "notyettreated")):
        for m in ("dr", "reg", "ipw"):
            r = ca.att_gt(q, "id", "t", "y", "g", control=ctrl, n_boot=0, covariates=["x0", "x0sq"], est_method=m,
                          verify=False, show=False)
            with contextlib.redirect_stdout(io.StringIO()):
                a = ATTgt(yname="y", gname="g", idname="id", tname="t", xformla="y~x0+x0sq", data=q,
                          control_group=cs_ctrl).fit(est_method=m)
            cs = pd.DataFrame({"g": a.results["group"], "t": a.results["year"], "cs": a.results["att"]})
            mm = r["gt"].reset_index().merge(cs, on=["g", "t"])
            add("causal.att_gt", float(np.max(np.abs(mm["att"] - mm["cs"]))), 1e-8, f"{ctrl}, {m}")
except ImportError:
    print("csdid not installed: causal.att_gt not measured")

# dml_benchmark: DoubleML 0.11.4 (test_v262.py; the same data, learners and splits as the test)
from sklearn.linear_model import LinearRegression  # noqa: E402

LIN = dict(learner_y=LinearRegression(), learner_d=LinearRegression())
dref = json.loads((ROOT / "tests" / "data" / "doubleml_benchmark_v262.json").read_text())
ns262 = {"np": np, "pd": pd}
exec(lines_between("test_v262.py", "rng = np.random.default_rng(20261111)", "DR = pd.DataFrame("), ns262)
DR = ns262["DR"]
assert "LIN = dict(learner_y=LinearRegression(), learner_d=LinearRegression())" in test_src("test_v262.py"), \
    "learners of test_v262"
RR = quiet(ca.dml_plr, DR, "y", "d", ["x1", "x2", "x3"], n_folds=5, n_rep=3, seed=0, **LIN)
BR = quiet(ca.dml_benchmark, DR, "y", "d", ["x1", "x2", "x3"], ["x1", "x2", "x3"], res=RR, n_folds=5, n_rep=3,
           seed=0, **LIN)
for b in ("x1", "x2"):
    for c in ("cf_y", "cf_d"):
        add("causal.dml_benchmark", abs(BR.loc[b, c] / dref[b][c] - 1), 0.02, f"{b} {c}", "rel")
    add("causal.dml_benchmark", abs(BR.loc[b, "delta_theta"] - dref[b]["delta_theta"]), 0.01, f"{b} delta_theta")
add("causal.dml_benchmark", abs(BR.loc["x3", "cf_y"] / dref["x3"]["cf_y"] - 1), 0.03, "x3 cf_y", "rel")

# oster: robomit 1.0.7 (test_v252.py; the table of reference values is read from the test)
tree = ast.parse((ROOT / "tests" / "test_v252.py").read_text(encoding="utf-8"))
RB = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
          and getattr(n.targets[0], "id", "") == "ROBOMIT_BETA")
RD = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
          and getattr(n.targets[0], "id", "") == "ROBOMIT_DELTA")
mt = pd.read_csv(ROOT / "tools" / "r_reference" / "output" / "mtcars.csv")
sim = pd.read_csv(ROOT / "tests" / "data" / "oster_v252.csv")
FITS = {"mtcars": (smf.ols("mpg ~ wt", mt).fit(), smf.ols("mpg ~ wt + hp + qsec", mt).fit(), "wt"),
        "sim": (smf.ols("y ~ x", sim).fit(), smf.ols("y ~ x + w1", sim).fit(), "x")}
for case, delta, rmax, bstar, alts, b0, r1 in RB:
    m0, m1, foc = FITS[case]
    oo = quiet(sp.oster, m0, m1, foc, r_max=rmax, delta=delta)
    add("specification.oster", abs(oo["beta_star_psacalc"] - bstar), 2e-6, f"{case} δ={delta} R_max={rmax}: β*")
for case, beta, rmax, dstar in RD:
    m0, m1, foc = FITS[case]
    oo = quiet(sp.oster, m0, m1, foc, r_max=rmax)
    inp = oo["exact_inputs"]
    got = sp._oster_exact_delta(oo["beta_uncontrolled"], oo["beta_controlled"], oo["r2_uncontrolled"],
                                oo["r2_controlled"], rmax, beta, inp["sigma_yy"], inp["sigma_xx"], inp["tau_x"])
    add("specification.oster", abs(got - dstar), 2e-6, f"{case} δ* for β={beta}, R_max={rmax}")

# survival.cox: lifelines (test_show_text.py)
LREF = json.loads((ROOT / "tests" / "data" / "cox_robust_lifelines_v267.json").read_text(encoding="utf-8"))
nsc = {"np": np, "pd": pd}
exec(function_src("test_show_text.py", "cox_design"), nsc)
for key, ref in LREF["designs"].items():
    dd = nsc["cox_design"](ref["seed"])
    _, tab = quiet(survival.cox, dd, "x + w", "t", "e", **({"cluster": "g"} if key.endswith("cluster") else {}))
    for c in ("x", "w"):
        add("survival.cox", abs(tab.loc[c, "coef"] - ref["coef"][c]), 1e-6, f"lifelines {key} coef {c}")
        add("survival.cox", abs(tab.loc[c, "se"] / ref["se"][c] - 1), 1e-6, f"lifelines {key} SE {c}", "rel")
RREF = json.loads((ROOT / "tools" / "r_reference" / "output" / "cox_v267_reference.json").read_text(encoding="utf-8"))
dties = pd.read_csv(ROOT / "tests" / "data" / "cox_v267_ties.csv")
for key, ref in RREF.items():
    if not isinstance(ref, dict):
        continue
    kw = {"ties": ref["ties"], **({"cluster": "g"} if ref["cluster"] == "g" else {}),
          **({"strata": "s"} if "strata" in key else {})}
    _, tab = quiet(survival.cox, dties, "x + w", "tc" if key.startswith("continuous") else "t", "e", **kw)
    for i, c in enumerate(("x", "w")):
        add("survival.cox", abs(tab.loc[c, "coef"] - ref["coef"][i]), 1e-6, f"R {key} coef {c}")
        add("survival.cox", abs(tab.loc[c, "se"] / ref["robust_se"][i] - 1), 1e-6, f"R {key} SE {c}", "rel")

# exact published values
for nv, zone in ((4, "green"), (5, "yellow"), (10, "red")):          # the cases of test_financial_modules.py
    add("risk.basel_traffic_light", float(risk.basel_traffic_light(nv) != zone), 0, f"{nv} violations -> {zone}")
add("inference.n_for_proportion", abs(quiet(inf.n_for_proportion, 0.03)["n"] - 1068), 0, "±3 points, 95%")
add("inference.n_for_proportion", abs(quiet(inf.n_for_proportion, 0.03, population=10_000)["n"] - 965), 0, "N = 10,000")

# ------------------------------------------------------------------ 3. summary
summary = {}
for fn, rows in sorted(OUT.items()):
    worst = max(rows, key=lambda r: (r[0] / r[1]) if r[1] > 0 else (np.inf if r[0] > 0 else 0))
    within = all((r[0] <= r[1]) if r[1] > 0 else r[0] == 0 for r in rows)
    summary[fn] = {"comparisons": len(rows), "worst_diff": worst[0], "tolerance": worst[1], "kind": worst[2],
                   "what": worst[3], "all_within_tolerance": within}
pkgs = {}
for p in ("numpy", "pandas", "scipy", "statsmodels", "linearmodels", "scikit-learn", "wooldridge", "csdid"):
    try:
        pkgs[p] = metadata.version(p)
    except metadata.PackageNotFoundError:
        pkgs[p] = None
res = {"date": datetime.date.today().isoformat(), "python": sys.version.split()[0], "packages": pkgs,
       "functions": summary}
if not CHECK:
    (ROOT / "tools" / "validation_diffs.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n",
                                                          encoding="utf-8")
for fn, s in summary.items():
    print(f"{fn:42s} {s['comparisons']:3d}  worst {s['worst_diff']:.3g} ({s['kind']}) vs tol {s['tolerance']:g}"
          f"  {'ok' if s['all_within_tolerance'] else 'OUTSIDE'}  [{s['what'][:60]}]")
outside = [fn for fn, s_ in summary.items() if not s_["all_within_tolerance"]]
if outside:
    print("OUTSIDE TOLERANCE:", ", ".join(outside))
sys.exit(1 if (CHECK and outside) else 0)
