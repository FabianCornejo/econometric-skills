"""Static guard (v2.5.3): no function parameter is reassigned inside a loop.

Why: in specification.sensitivity, `q = len(terms)` inside the loop over joint benchmarks overwrote the
parameter `q`, and the `show` block printed RV(q=<number of dummies>) for five versions (v2.2.1-v2.5.2,
fixed in v2.5.2.1). The numbers were right, so no numeric test could see it. This guard turns the one-off
AST scan of v2.5.2.1 into a permanent check over every function in econometrics/scripts.

Rule: a parameter is never the target of an assignment, an augmented assignment, a walrus or a `for`
inside a `for`/`while` body (nested functions, lambdas, classes and comprehensions have their own scope and
are not entered). In functions with a `show` parameter there are no exceptions. Elsewhere the known,
intentional cases are listed in ALLOWED with the reason; a new one fails until it is listed (or renamed).

The guard is also run on a copy of the v2.5.2 code of sensitivity (the bug) and on variants of the pattern,
so this file proves it catches the bug, not only that the current code passes.

Run from the repository root:  python3 tests/test_show_static.py
Needs: nothing outside the standard library.
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "econometrics" / "scripts"))
import _console  # noqa: E402,F401  (Windows-safe console output, see econometrics/scripts/_console.py)

ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


# (module, function, parameter): why reassigning it inside a loop is intended
ALLOWED = {
    ("estimation", "mle", "method"): "falls back to BFGS when the requested optimizer fails; nothing prints it",
    ("simulation", "mde_did_sim", "lo"): "bisection bounds of the search for the minimum detectable effect",
    ("simulation", "mde_did_sim", "hi"): "bisection bounds of the search for the minimum detectable effect",
}
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef,
           ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _names(target):
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, (ast.Tuple, ast.List)):
        for e in target.elts:
            yield from _names(e)
    elif isinstance(target, ast.Starred):
        yield from _names(target.value)


def loop_reassignments(fn):
    """(parameter, line) for every parameter of `fn` reassigned inside a loop of `fn` itself."""
    a = fn.args
    params = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
    params |= {x.arg for x in (a.vararg, a.kwarg) if x is not None}
    hits = []

    def visit(node, in_loop):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, _SCOPES):
                continue
            targets = []
            if isinstance(ch, (ast.For, ast.AsyncFor)):
                targets.append(ch.target)          # the loop variable is reassigned on every pass
            elif in_loop and isinstance(ch, ast.Assign):
                targets.extend(ch.targets)
            elif in_loop and isinstance(ch, (ast.AugAssign, ast.AnnAssign, ast.NamedExpr)):
                targets.append(ch.target)
            hits.extend((n, ch.lineno) for t in targets for n in _names(t) if n in params)
            visit(ch, in_loop or isinstance(ch, (ast.For, ast.AsyncFor, ast.While)))

    visit(fn, False)
    return params, hits


def scan_source(src, module):
    """[(module, function, parameter, line, has_show)] for one source text."""
    found = []
    for fn in ast.walk(ast.parse(src)):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params, hits = loop_reassignments(fn)
            found.extend((module, fn.name, p, line, "show" in params) for p, line in hits)
    return found


# --- 1. the guard catches the bug it is for ------------------------------------------------------------
OLD_SENSITIVITY = '''
def sensitivity(res, treatment, q=1.0, alpha=0.05, show=False):
    out = robustness_value(res.tvalues[treatment], res.df_resid, q=q, alpha=alpha)
    for gname, terms in groups.items():
        if len(terms) == 1:
            pass
        else:
            q = len(terms)
            bench[gname] = q * F / (q * F + res.df_resid)
    if show:
        print(f"RV(q={q}) = {out['rv_q']:.2%}")
    return out
'''
f = scan_source(OLD_SENSITIVITY, "specification")
check("guard catches the v2.5.2 sensitivity bug (q = len(terms) inside the loop, in a show function)",
      [(m, fn, p, s) for m, fn, p, _, s in f] == [("specification", "sensitivity", "q", True)], str(f))

VARIANTS = {
    "loop variable named like a parameter": "def f(x, q=1, show=False):\n    for q in range(3):\n        pass\n",
    "augmented assignment in a while": "def f(n, show=False):\n    while n > 0:\n        n -= 1\n",
    "tuple unpacking inside a nested loop": "def f(a, b, show=False):\n    for i in range(2):\n"
                                            "        for j in range(2):\n            a, c = 1, 2\n",
    "walrus inside a loop": "def f(k, show=False):\n    for i in range(3):\n        if (k := i) > 1:\n            pass\n",
    "nested for whose target is a parameter": "def f(t, show=False):\n    for i in range(2):\n"
                                              "        for t in range(2):\n            pass\n",
}
for label, src in VARIANTS.items():
    check(f"guard catches: {label}", len(scan_source(src, "m")) == 1, str(scan_source(src, "m")))

CLEAN = {
    "normalization at the top (x = np.asarray(x)) is not a loop": "def f(x, show=False):\n    x = list(x)\n"
                                                                  "    for i in x:\n        y = i\n",
    "a comprehension has its own scope": "def f(q, show=False):\n    for i in range(2):\n"
                                         "        z = [q for q in range(3)]\n",
    "a nested function has its own scope": "def f(q, show=False):\n    for i in range(2):\n"
                                           "        def g(q):\n            q = 2\n",
    "a local that is not a parameter": "def f(a, show=False):\n    for i in range(2):\n        k = i\n",
}
for label, src in CLEAN.items():
    check(f"guard stays quiet: {label}", scan_source(src, "m") == [], str(scan_source(src, "m")))

# --- 2. the package ------------------------------------------------------------------------------------
found = []
n_fn = n_show = 0
for path in sorted((ROOT / "econometrics" / "scripts").glob("*.py")):
    src = path.read_text(encoding="utf-8")
    found.extend(scan_source(src, path.stem))
    for fn in ast.walk(ast.parse(src)):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            n_fn += 1
            n_show += any(x.arg == "show" for x in fn.args.args + fn.args.kwonlyargs)
check(f"the scan sees the package ({n_fn} functions, {n_show} with show)", n_fn > 300 and n_show > 100,
      f"{n_fn} functions, {n_show} with show")
in_show = [x for x in found if x[4]]
check("no parameter of a function with `show` is reassigned inside a loop", in_show == [], str(in_show))
unlisted = [x for x in found if (x[0], x[1], x[2]) not in ALLOWED]
check("outside `show` functions, only the listed intentional cases", unlisted == [], str(unlisted))
stale = [k for k in ALLOWED if not any((m, fn, p) == k for m, fn, p, _, _ in found)]
check("every allowed case still exists (a stale entry would hide a new one under an old name)", stale == [],
      str(stale))

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
