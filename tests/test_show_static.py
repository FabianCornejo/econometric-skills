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


def _outer_parts(node):
    """The parts of a nested def, lambda or class that run in the enclosing scope (where they are defined):
    decorators, defaults, class bases and keywords, and the annotations of a def's arguments and return (evaluated
    at definition up to Python 3.13; 3.14 refuses a walrus there, so counting them costs nothing)."""
    out = list(getattr(node, "decorator_list", []))
    if isinstance(node, ast.ClassDef):
        return out + list(node.bases) + [k.value for k in node.keywords]
    a = node.args
    out += list(a.defaults) + [d for d in a.kw_defaults if d is not None]
    if not isinstance(node, ast.Lambda):
        args = a.posonlyargs + a.args + a.kwonlyargs + [x for x in (a.vararg, a.kwarg) if x is not None]
        out += [x.annotation for x in args if x.annotation is not None]
        out += [node.returns] if node.returns is not None else []
    return out


def _own_body(f):
    """The nodes of function f's own scope, nested defs, lambdas, classes and comprehensions not entered (each of
    those is returned itself, so its name or its walrus can still be looked at)."""
    stack, out = list(f.body), []
    while stack:
        s = stack.pop()
        out.append(s)
        if not isinstance(s, _FUNCS + _COMPS):
            stack.extend(ast.iter_child_nodes(s))
    return out


def _walrus_targets(node):
    """Names bound by a walrus in `node` in the scope where `node` runs: a comprehension's walrus counts (it binds
    in the enclosing function), a lambda's or a nested def's body does not (it binds there) (review 2)."""
    out, stack = [], [node]
    while stack:
        s = stack.pop()
        if isinstance(s, ast.NamedExpr):
            out.append(s.target.id)
        if isinstance(s, _FUNCS) and s is not node:
            stack.extend(_outer_parts(s))          # its defaults, decorators...: still this scope; not its body
            continue
        stack.extend(ast.iter_child_nodes(s))
    return out


def _nonlocal_rebinds(node, params):
    """(parameter, line) for every outer parameter that a function inside `node` (node included) declares
    `nonlocal` in its own body and then rebinds there (an assignment of any kind, a def or class of that name, a
    walrus in a comprehension; a comprehension's own loop variable does not count). Scopes are followed: a function
    that has its own `p` (a parameter, or bound without `nonlocal p`) hides the outer one from the functions inside
    it, whose `nonlocal p` then means that local (review 1); a class body hides nothing (nonlocal skips it)."""
    hits = []

    def walk(f, live):
        if isinstance(f, ast.Lambda):             # an expression body: no nonlocal, no def inside
            return
        if isinstance(f, ast.ClassDef):           # methods see through the class body
            for s in _own_body(f):
                if isinstance(s, _FUNCS) and s is not f:
                    walk(s, live)
            return
        body = _own_body(f)
        declared = {n for s in body if isinstance(s, ast.Nonlocal) for n in s.names} & live
        bound = set()
        for s in body:
            if isinstance(s, _COMPS):
                names = _walrus_targets(s)
            elif isinstance(s, _FUNCS):            # its name, and a walrus in its defaults, decorators, annotations
                names = [getattr(s, "name", None)] + [n for part in _outer_parts(s) for n in _walrus_targets(part)]
            else:
                names = _bound(s)
            bound |= set(names)
            hits.extend((n, getattr(s, "lineno", 0)) for n in names if n in declared)
        a = f.args
        own = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs} | {x.arg for x in (a.vararg, a.kwarg) if x}
        inner = live - ((own | bound) - declared)     # names still meaning the outer parameters below this function
        for s in body:
            if isinstance(s, _FUNCS):
                walk(s, inner)

    walk(node, set(params))
    return hits


def loop_reassignments(fn):
    """(parameter, line) for every parameter of `fn` rebound (assignment, augmented or annotated assignment, walrus,
    for target, with/except/import alias, del, match capture) inside a for/while of `fn` itself.

    Nested functions, lambdas and classes are their own scope and are not entered. In a comprehension only a walrus
    binds in the function (its loop variables do not). A loop's `else` block and a for's iterable run once, so they
    are outside the loop; a while's test runs on every pass; a comprehension is a loop of its own (a walrus in it
    counts even outside any for/while). v2.6.5: a nested def's, lambda's or class's defaults, decorators, annotations,
    bases and keywords run in this function's scope, so a walrus there counts like one in the body (inside a loop, a
    hit); and a parameter that a nested function rebinds after `nonlocal p` is always a hit, in a loop or not, because
    where the nested function is called is not visible without flow analysis. Scopes are followed: under a function
    with its own `p`, a `nonlocal p` is that function's (see _nonlocal_rebinds)."""
    a = fn.args
    params = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
    params |= {x.arg for x in (a.vararg, a.kwarg) if x is not None}
    hits = []

    def visit(node, in_loop, comp=False):
        if isinstance(node, _FUNCS):              # its own scope; only its name binds here (def q(): ... in a loop)
            name = getattr(node, "name", None)
            if in_loop and not comp and name in params:
                hits.append((name, node.lineno))
            for ch in _outer_parts(node):         # defaults, decorators, bases: evaluated here (v2.6.5)
                visit_part(ch, in_loop, comp)
            hits.extend(_nonlocal_rebinds(node, params))
            return
        if in_loop:
            if comp:
                names = [node.target.id] if isinstance(node, ast.NamedExpr) else []
            else:
                names = _bound(node)
            hits.extend((n, getattr(node, "lineno", 0)) for n in names if n in params)
        if isinstance(node, (ast.For, ast.AsyncFor)):
            visit_part(node.iter, in_loop, comp)
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
            visit_part(ch, in_loop, comp)

    def visit_part(ch, in_loop, comp):
        if isinstance(ch, _COMPS):                # a comprehension is itself a loop: a walrus in it rebinds every pass
            visit(ch, True, True)                 # (review 1: also as a default or a for's iterable)
        else:
            visit(ch, in_loop, comp)

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
    "walrus in a comprehension, outside any for/while (the one-line form of the bug)":
        "def f(q, show=False):\n    z = [(q := len(t)) for t in g]\n    print(q)\n",
    "assignment in a while test": "def f(q, show=False):\n    while (q := q - 1) > 0:\n        pass\n",
    # v2.6.5 (v2.5.3 review NITs): parts of a nested def that run in the outer scope, and nonlocal
    "walrus in the default of a def inside a loop": "def f(q, show=False):\n    for i in range(2):\n"
                                                   "        def g(x=(q := i)):\n            pass\n",
    "walrus in the keyword-only default of a def inside a loop": "def f(q, show=False):\n    for i in range(2):\n"
                                                                 "        def g(*, x=(q := i)):\n            pass\n",
    "walrus in the default of a lambda inside a loop": "def f(q, show=False):\n    for i in range(2):\n"
                                                       "        h = lambda x=(q := i): x\n",
    "walrus in a decorator inside a loop": "def f(q, show=False):\n    for i in range(2):\n"
                                           "        @deco(q := i)\n        def g():\n            pass\n",
    "walrus in a class base inside a loop": "def f(q, show=False):\n    for i in range(2):\n"
                                            "        class C(base(q := i)):\n            pass\n",
    "nonlocal rebinding of a parameter (outside any loop)": "def f(q, show=False):\n    def g():\n"
                                                            "        nonlocal q\n        q = 3\n    g()\n    print(q)\n",
    "nonlocal rebinding with an augmented assignment, called in a loop": "def f(q, show=False):\n    def g():\n"
                                                                         "        nonlocal q\n        q += 1\n"
                                                                         "    for i in range(2):\n        g()\n",
    "nonlocal two levels down": "def f(q, show=False):\n    def g():\n        def h():\n"
                                "            nonlocal q\n            q = 1\n        h()\n    g()\n",
    # review 1 of v2.6.5
    "walrus in a comprehension that is a nested def's default (outside any loop: a comprehension is a loop)":
        "def f(q, show=False):\n    def g(x=[(q := t) for t in y]):\n        pass\n",
    "walrus in a comprehension that is a for's iterable": "def f(q, show=False):\n"
                                                          "    for i in [(q := t) for t in y]:\n        pass\n",
    "walrus in a nested def's argument annotation inside a loop": "def f(q, show=False):\n    for i in range(2):\n"
                                                                  "        def g(x: (q := int)):\n            pass\n",
    "walrus in a nested def's return annotation inside a loop": "def f(q, show=False):\n    for i in range(2):\n"
                                                                "        def g() -> (q := int):\n            pass\n",
    "nonlocal then a def of that name": "def f(q, show=False):\n    def g():\n        nonlocal q\n"
                                        "        def q():\n            pass\n    g()\n",
    "nonlocal then a class of that name": "def f(q, show=False):\n    def g():\n        nonlocal q\n"
                                          "        class q:\n            pass\n    g()\n",
    "nonlocal in a method of a nested class (the class body hides nothing)": "def f(q, show=False):\n"
        "    class C:\n        def m(self):\n            nonlocal q\n            q = 1\n    C().m()\n",
    "nonlocal, then a walrus in a deeper def's default": "def f(q, show=False):\n    def g():\n        nonlocal q\n"
        "        def h(x=(q := 1)):\n            pass\n    g()\n",
    "nonlocal, then a walrus in a deeper def's decorator": "def f(q, show=False):\n    def g():\n        nonlocal q\n"
        "        @deco(q := 1)\n        def h():\n            pass\n    g()\n",
    "nonlocal, then a walrus in a lambda's default": "def f(q, show=False):\n    def g():\n        nonlocal q\n"
        "        h = lambda x=(q := 1): x\n    g()\n",
    "nonlocal, then a walrus in a class base": "def f(q, show=False):\n    def g():\n        nonlocal q\n"
        "        class C(base(q := 1)):\n            pass\n    g()\n",
    "walrus in a nested def's keyword-only annotation inside a loop": "def f(q, show=False):\n    for i in range(2):\n"
        "        def g(*, x: (q := int)):\n            pass\n",
    "walrus in a nested def's *args annotation inside a loop": "def f(q, show=False):\n    for i in range(2):\n"
        "        def g(*a: (q := int)):\n            pass\n",
    "walrus in a nested def's positional-only annotation inside a loop": "def f(q, show=False):\n    for i in range(2):\n"
        "        def g(x: (q := int), /):\n            pass\n",
    "nonlocal then a walrus in a comprehension": "def f(q, show=False):\n    def g():\n        nonlocal q\n"
                                                 "        z = [(q := t) for t in y]\n    g()\n",
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
    "walrus in the default of a nested def outside any loop runs once": "def f(q, show=False):\n"
                                                                        "    def g(x=(q := 1)):\n        pass\n",
    "a default that reads a parameter without rebinding it": "def f(q, show=False):\n    for i in range(2):\n"
                                                             "        def g(x=q):\n            return x\n",
    "nonlocal of a local that is not a parameter": "def f(q, show=False):\n    k = 0\n    def g():\n"
                                                   "        nonlocal k\n        k = 1\n    g()\n",
    "nonlocal of a parameter that is only read": "def f(q, show=False):\n    def g():\n        nonlocal q\n"
                                                 "        return q + 1\n    return g()\n",
    "a nested function's own assignment without nonlocal is local": "def f(q, show=False):\n    def g():\n"
                                                                    "        q = 2\n        return q\n    return g()\n",
    "a local q of an intermediate def, with nonlocal q one level deeper (it refers to that local)":
        "def f(q, show=False):\n    def g():\n        q = 5\n        def h():\n            nonlocal q\n"
        "            q = 6\n        h()\n    g()\n",
    "nonlocal q, then q only as a comprehension's own loop variable": "def f(q, show=False):\n    def g():\n"
                                                                      "        nonlocal q\n        return [0 for q in y]\n"
                                                                      "    return g()\n",
    "nonlocal q, then a walrus inside a lambda inside a comprehension (binds in the lambda)":
        "def f(q, show=False):\n    def g():\n        nonlocal q\n        z = [(lambda: (q := 1)) for t in y]\n    g()\n",
    "an annotation that reads a parameter inside a loop": "def f(q, show=False):\n    for i in range(2):\n"
                                                          "        def g(x: q):\n            pass\n",
}
for label, src in CLEAN.items():
    check(f"guard stays quiet: {label}", scan_source(src, "m") == [], str(scan_source(src, "m")))
SHADOW = ("def f(q, show=False):\n    def g(q):\n        def h():\n            nonlocal q\n            q = 6\n"
          "        h()\n    g(1)\n")
f_ = scan_source(SHADOW, "m")
check("nonlocal q under a def g(q): the hit is g's parameter, never f's (scopes followed; review 1 of v2.6.5)",
      [(fn, p) for _, fn, p, _, _ in f_] == [("g", "q")], str(f_))

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
