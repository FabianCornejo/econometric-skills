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
    ("estimation", "mle", "method"): "falls back to BFGS when the requested optimizer fails; the returned "
                                     "'method' reports the optimizer actually used",
    ("simulation", "mde_did_sim", "lo"): "bisection bounds of the search for the minimum detectable effect",
    ("simulation", "mde_did_sim", "hi"): "bisection bounds of the search for the minimum detectable effect",
}
_FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
_COMPS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _bound(node):
    """Names a single node binds or unbinds in the enclosing function scope (not recursive)."""
    if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
        return [node.id]
    if isinstance(node, ast.ExceptHandler) and node.name:
        return [node.name]
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return [(a.asname or a.name).split(".")[0] for a in node.names]
    match_nodes = tuple(getattr(ast, n) for n in ("MatchAs", "MatchStar") if hasattr(ast, n))
    if match_nodes and isinstance(node, match_nodes) and node.name:
        return [node.name]
    if hasattr(ast, "MatchMapping") and isinstance(node, ast.MatchMapping) and node.rest:
        return [node.rest]
    return []


def loop_reassignments(fn):
    """(parameter, line) for every parameter of `fn` rebound (assignment, augmented or annotated assignment, walrus,
    for target, with/except/import alias, del, match capture) inside a for/while of `fn` itself.

    Nested functions, lambdas and classes are their own scope and are not entered. In a comprehension only a walrus
    binds in the function (its loop variables do not). A loop's `else` block and a for's iterable run once, so they
    are outside the loop; a while's test runs on every pass. Not covered: `nonlocal` rebinding from a nested
    function (needs scope analysis; never used in scripts/)."""
    a = fn.args
    params = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
    params |= {x.arg for x in (a.vararg, a.kwarg) if x is not None}
    hits = []

    def visit(node, in_loop, comp=False):
        if isinstance(node, _FUNCS):              # its own scope; only its name binds here (def q(): ... in a loop)
            name = getattr(node, "name", None)
            if in_loop and not comp and name in params:
                hits.append((name, node.lineno))
            return
        if in_loop:
            if comp:
                names = [node.target.id] if isinstance(node, ast.NamedExpr) else []
            else:
                names = _bound(node)
            hits.extend((n, getattr(node, "lineno", 0)) for n in names if n in params)
        if isinstance(node, (ast.For, ast.AsyncFor)):
            visit(node.iter, in_loop, comp)
            for ch in [node.target, *node.body]:
                visit(ch, True, comp)
            for ch in node.orelse:
                visit(ch, in_loop, comp)
            return
        if isinstance(node, ast.While):
            for ch in [node.test, *node.body]:
                visit(ch, True, comp)
            for ch in node.orelse:
                visit(ch, in_loop, comp)
            return
        for ch in ast.iter_child_nodes(node):
            visit(ch, in_loop, comp or isinstance(ch, _COMPS))

    for stmt in fn.body:
        visit(stmt, False)
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
    "with ... as": "def f(q, show=False):\n    for i in range(2):\n        with open('x') as q:\n            pass\n",
    "except ... as": "def f(q, show=False):\n    for i in range(2):\n        try:\n            pass\n"
                     "        except ValueError as q:\n            pass\n",
    "del": "def f(q, show=False):\n    while True:\n        del q\n",
    "import ... as": "def f(np, show=False):\n    for i in range(2):\n        import numpy as np\n",
    "match capture": "def f(q, show=False):\n    for i in range(2):\n        match i:\n"
                     "            case [q, *_]:\n                pass\n",
    "walrus inside a comprehension (binds in the function)": "def f(q, show=False):\n    for i in range(2):\n"
                                                             "        z = [(q := j) for j in range(3)]\n",
    "a def inside a loop named like a parameter": "def f(g, show=False):\n    for i in range(2):\n"
                                                  "        def g():\n            pass\n",
    "assignment in a while test": "def f(q, show=False):\n    while (q := q - 1) > 0:\n        pass\n",
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
    "the else block of a loop runs once": "def f(q, show=False):\n    for i in range(2):\n        pass\n"
                                          "    else:\n        q = 3\n",
    "a walrus in a for's iterable runs once": "def f(q, show=False):\n    for i in (q := [1, 2]):\n        pass\n",
    "a comprehension's own loop variable": "def f(q, show=False):\n    for i in range(2):\n"
                                           "        z = {q: 1 for q in range(3)}\n",
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
