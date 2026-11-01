"""
Causal graphs (DAGs) for choosing controls before estimating: which variables to adjust for,
which to leave out (mediators, colliders, descendants of the treatment, instruments), and
which conditional independences the graph implies so that the data can contradict it.

The graph encodes the identifying assumptions (Pearl 2009; Cinelli, Forney & Pearl 2022, "A
crash course in good and bad controls"). The functions are exact graph algorithms; the graph
itself is the analyst's claim and must be defended with institutional knowledge.

Graph input: a string "Z -> X; Z -> Y; X -> Y; X <-> Y" or a list of (parent, child) pairs.
"A <-> B" means an unobserved common cause of A and B (added as a latent node). Names listed
in `latent` (or created by <->) are unobserved and never proposed as controls.

    DAG(edges, latent=())                  parse; .parents/.children/.ancestors/.descendants, .to_mermaid()
    d_separated(g, xs, ys, zs)             d-separation by the moralized ancestral graph
    is_valid_adjustment(g, x, y, Z)        backdoor criterion: Z has no descendant of x and blocks every backdoor path
    adjustment_sets(g, x, y, ...)          all minimal valid adjustment sets among observed variables
    optimal_adjustment_set(g, x, y)        Henckel, Perkovic & Maathuis (2022) O-set: valid set with the smallest
                                           asymptotic variance among valid sets (linear models), when observable
    classify_controls(g, x, y)             role of every other variable + recommendation (control / do not / optional)
    implied_independencies(g)              local Markov implications u _||_ v | pa(u) among observed variables
    test_implications(g, df, alpha)        partial-correlation (Fisher z) tests of those implications on data
    adjustment_report(g, x, y, df=None)    everything above in one printed report

Requires: numpy, pandas, scipy.
"""
from __future__ import annotations

import itertools
import re

import numpy as np
import pandas as pd
from scipy import stats


class DAG:
    def __init__(self, edges, latent=()):
        self.latent = set(latent)
        self.edges: set[tuple[str, str]] = set()
        if isinstance(edges, str):
            parts = [p.strip() for p in re.split(r"[;\n,]", edges) if p.strip()]
            pairs = []
            for p in parts:
                if "<->" in p:
                    a, b = [s.strip() for s in p.split("<->")]
                    u = f"U[{a},{b}]"
                    self.latent.add(u)
                    pairs += [(u, a), (u, b)]
                elif "->" in p:
                    chain = [s.strip() for s in p.split("->")]
                    pairs += list(zip(chain[:-1], chain[1:]))
                elif "<-" in p:
                    chain = [s.strip() for s in p.split("<-")]
                    pairs += list(zip(chain[1:], chain[:-1]))
                else:
                    pairs.append((p, None))
            edges = pairs
        self.nodes: set[str] = set()
        for a, b in edges:
            self.nodes.add(a)
            if b is not None:
                self.nodes.add(b)
                self.edges.add((a, b))
        self.nodes |= self.latent & self.nodes
        self._check_acyclic()

    # -- structure ---------------------------------------------------------------------------
    def parents(self, v):
        return {a for a, b in self.edges if b == v}

    def children(self, v):
        return {b for a, b in self.edges if a == v}

    def ancestors(self, vs):
        vs = {vs} if isinstance(vs, str) else set(vs)
        out, stack = set(vs), list(vs)
        while stack:
            for p in self.parents(stack.pop()):
                if p not in out:
                    out.add(p)
                    stack.append(p)
        return out                                 # includes vs themselves

    def descendants(self, vs):
        vs = {vs} if isinstance(vs, str) else set(vs)
        out, stack = set(vs), list(vs)
        while stack:
            for c in self.children(stack.pop()):
                if c not in out:
                    out.add(c)
                    stack.append(c)
        return out                                 # includes vs themselves

    @property
    def observed(self):
        return self.nodes - self.latent

    def _check_acyclic(self):
        indeg = {v: 0 for v in self.nodes}
        for _, b in self.edges:
            indeg[b] += 1
        q = [v for v, d in indeg.items() if d == 0]
        seen = 0
        while q:
            v = q.pop()
            seen += 1
            for c in self.children(v):
                indeg[c] -= 1
                if indeg[c] == 0:
                    q.append(c)
        if seen != len(self.nodes):
            raise ValueError("the graph has a cycle: a DAG must be acyclic")

    def topological_order(self):
        order, placed = [], set()
        while len(order) < len(self.nodes):
            for v in sorted(self.nodes - placed):
                if self.parents(v) <= placed:
                    order.append(v)
                    placed.add(v)
        return order

    def remove_outgoing(self, x):
        g = DAG([(a, b) for a, b in self.edges if a != x] + [(v, None) for v in self.nodes], self.latent)
        return g

    def to_mermaid(self) -> str:
        def nid(v):
            return re.sub(r"\W", "_", v)
        lines = ["graph LR"]
        for v in sorted(self.nodes):
            lines.append(f"  {nid(v)}([{v}])" if v in self.latent else f"  {nid(v)}[{v}]")
        for a, b in sorted(self.edges):
            lines.append(f"  {nid(a)} {'-.->' if a in self.latent else '-->'} {nid(b)}")
        return "\n".join(lines)

    def __repr__(self):
        return f"DAG({len(self.nodes)} nodes, {len(self.edges)} edges, latent={sorted(self.latent)})"


def _as_dag(g):
    return g if isinstance(g, DAG) else DAG(g)


def _set(v):
    if v is None:
        return set()
    return {v} if isinstance(v, str) else set(v)


def d_separated(g, xs, ys, zs=()) -> bool:
    """True if every path between xs and ys is blocked by zs (Lauritzen's moralization test)."""
    g = _as_dag(g)
    xs, ys, zs = _set(xs), _set(ys), _set(zs)
    keep = g.ancestors(xs | ys | zs)
    adj = {v: set() for v in keep}
    for a, b in g.edges:
        if a in keep and b in keep:
            adj[a].add(b); adj[b].add(a)
    for v in keep:                                 # marry the parents
        ps = [p for p in g.parents(v) if p in keep]
        for p, q in itertools.combinations(ps, 2):
            adj[p].add(q); adj[q].add(p)
    seen, stack = set(xs), list(xs)
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if w in zs or w in seen:
                continue
            if w in ys:
                return False
            seen.add(w)
            stack.append(w)
    return True


def is_valid_adjustment(g, x: str, y: str, Z) -> bool:
    """Backdoor criterion (Pearl 1993): no element of Z descends from x, and Z d-separates x and y
    in the graph without the edges out of x."""
    g = _as_dag(g)
    Z = _set(Z)
    if Z & (g.descendants(x) - {x}) or x in Z or y in Z:
        return False
    return d_separated(g.remove_outgoing(x), x, y, Z)


def adjustment_sets(g, x: str, y: str, observed=None, max_size: int | None = None, max_candidates: int = 18) -> list:
    """All minimal valid adjustment sets made of observed variables (smallest first). An empty
    list means the effect is not identified by adjustment: look for an instrument, a design
    (DiD, RD) or a front-door path."""
    g = _as_dag(g)
    obs = _set(observed) if observed is not None else g.observed
    cand = sorted((obs - (g.descendants(x)) - {y}) & g.ancestors({x, y}))   # non-ancestors never needed
    if len(cand) > max_candidates:
        raise ValueError(f"{len(cand)} candidate controls: pass max_size or fewer observed variables")
    found = []
    for k in range(0, (max_size if max_size is not None else len(cand)) + 1):
        for Z in itertools.combinations(cand, k):
            Zs = set(Z)
            if any(f <= Zs for f in found):
                continue
            if is_valid_adjustment(g, x, y, Zs):
                found.append(Zs)
    return [sorted(f) for f in found]


def _causal_nodes(g, x, y):
    """Nodes on proper causal paths x -> ... -> y, excluding x."""
    return (g.descendants(x) - {x}) & g.ancestors(y)


def optimal_adjustment_set(g, x: str, y: str):
    """O-set of Henckel, Perkovic & Maathuis (2022): parents of the causal nodes, minus the
    forbidden set (descendants of causal nodes and x). Among all valid adjustment sets it gives
    the OLS/AIPW estimator with the smallest asymptotic variance: it adds outcome predictors
    (precision) and leaves out pure treatment predictors (instruments, which inflate variance).
    Returns None if the O-set contains a latent variable or is not valid."""
    g = _as_dag(g)
    cn = _causal_nodes(g, x, y)
    if not cn:
        return None
    forb = set().union(*[g.descendants(c) for c in cn]) | {x}
    O = set().union(*[g.parents(c) for c in cn]) - forb
    if O & g.latent or not is_valid_adjustment(g, x, y, O):
        return None
    return sorted(O)


def classify_controls(g, x: str, y: str) -> pd.DataFrame:
    """Role of every other variable relative to the effect of x on y, with a recommendation.

    Rules, in order: mediator (on a causal path) and other descendants of x are post-treatment:
    never control. Confounder: causes x and causes y other than through x. Instrument-like:
    causes x but y only through x (controlling raises the variance and amplifies any remaining
    bias). Outcome predictor: causes y, not x: optional, improves precision. Collider: not a
    cause of x or y, but controlling for it opens a path that a valid set had closed (M-bias)."""
    g = _as_dag(g)
    desc_x = g.descendants(x) - {x}
    cn = _causal_nodes(g, x, y)
    gx = g.remove_outgoing(x)
    anc_x = g.ancestors(x) - {x}
    anc_y_nox = gx.ancestors(y) - {y}                # causes of y by paths that avoid x
    sets = adjustment_sets(g, x, y) if len(g.observed) <= 20 else []
    base = set(sets[0]) if sets else set()
    O = optimal_adjustment_set(g, x, y) or []
    rows = []
    for v in sorted(g.nodes - {x, y}):
        if v in cn:
            role, rec = "mediator (x -> v -> y)", "do not control: removes part of the effect"
        elif v in desc_x:
            opens = sets and not d_separated(gx, x, y, base | {v})
            role = "descendant of x" + (" and collider (opens a path)" if opens else " (post-treatment)")
            rec = "do not control"
        elif v in anc_x and v in anc_y_nox:
            role, rec = "confounder (causes x and y)", "control, or a set that blocks its path"
        elif v in anc_x:
            role, rec = "instrument-like (affects y only through x)", "do not control: more variance, amplifies bias"
        elif v in anc_y_nox:
            role, rec = "outcome predictor (not a cause of x)", "optional: improves precision"
        elif sets and not d_separated(gx, x, y, base | {v}):
            role, rec = "collider (controlling opens a path: M-bias)", "do not control"
        else:
            role, rec = "not on any x-y path", "irrelevant"
        in_min = any(v in s_ for s_ in sets)
        if in_min and rec.startswith("optional"):
            rec = "control: blocks a backdoor path (also improves precision)"
        if v in g.latent:
            rec = "unobserved" + ("; if no valid set exists, the effect is not identified by adjustment"
                                  if role.startswith("confounder") else "")
        rows.append({"variable": v, "observed": v not in g.latent, "role": role, "recommendation": rec,
                     "in_minimal_set": in_min, "in_optimal_set": v in O})
    return pd.DataFrame(rows).set_index("variable")


def implied_independencies(g, max_cond: int = 4) -> list:
    """Local Markov implications u _||_ v | pa(u) for observed u, v with v a non-descendant
    non-parent of u and pa(u) observed (each pair once). These are testable on data."""
    g = _as_dag(g)
    out, seen = [], set()
    for u in g.topological_order():
        if u in g.latent:
            continue
        pa = g.parents(u)
        if pa & g.latent or len(pa) > max_cond:
            continue
        for v in sorted(g.observed - g.descendants(u) - pa - {u}):
            key = frozenset((u, v))
            if key in seen:
                continue
            if d_separated(g, u, v, pa):
                seen.add(key)
                out.append((u, v, sorted(pa)))
    return out


def test_implications(g, df: pd.DataFrame, alpha: float = 0.05, show: bool = True) -> pd.DataFrame:
    """Test each implied independence with a partial correlation and Fisher's z (linear-Gaussian
    approximation; for binary or nonlinear relations treat it as a screen). A rejection says the
    graph is missing an edge (or the relation is nonlinear); passing does not prove the graph.
    Holm-adjusted p-values account for the number of tests."""
    g = _as_dag(g)
    rows = []
    for u, v, Z in implied_independencies(g):
        if not {u, v, *Z} <= set(df.columns):
            continue
        d = df[[u, v, *Z]].dropna().astype(float)
        if Z:
            Zm = np.column_stack([np.ones(len(d)), d[Z].to_numpy()])
            ru = d[u] - Zm @ np.linalg.lstsq(Zm, d[u], rcond=None)[0]
            rv = d[v] - Zm @ np.linalg.lstsq(Zm, d[v], rcond=None)[0]
        else:
            ru, rv = d[u] - d[u].mean(), d[v] - d[v].mean()
        r = float(np.corrcoef(ru, rv)[0, 1])
        zstat = np.arctanh(np.clip(r, -0.999999, 0.999999)) * np.sqrt(len(d) - len(Z) - 3)
        rows.append({"implication": f"{u} _||_ {v}" + (f" | {', '.join(Z)}" if Z else ""), "partial_r": r,
                     "z": zstat, "p": float(2 * stats.norm.sf(abs(zstat))), "n": len(d)})
    tab = pd.DataFrame(rows)
    if len(tab):
        m = len(tab)
        order = np.argsort(tab["p"].to_numpy())
        adj = np.empty(m)
        run = 0.0
        for rank, i in enumerate(order):
            run = max(run, min(1.0, (m - rank) * tab["p"].iloc[i]))
            adj[i] = run
        tab["p_holm"] = adj
        tab["consistent"] = tab["p_holm"] > alpha
    if show:
        if not len(tab):
            print("\nNo testable implications among the observed columns.")
        else:
            print(f"\nTestable implications of the DAG ({len(tab)}), partial correlation tests:")
            print(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
            bad = tab.loc[~tab["consistent"], "implication"].tolist()
            print("  " + ("Data contradict: " + "; ".join(bad) + " -> revise the graph (missing edge?)" if bad
                          else "No implication rejected (Holm, 5%): the data do not contradict the graph."))
    return tab


def adjustment_report(g, x: str, y: str, df: pd.DataFrame | None = None, show: bool = True) -> dict:
    """Minimal and optimal adjustment sets, the role of every variable and, with data, the tests
    of the graph's implications. Use it before choosing the regression's controls."""
    g = _as_dag(g)
    sets = adjustment_sets(g, x, y)
    O = optimal_adjustment_set(g, x, y)
    roles = classify_controls(g, x, y)
    tests = test_implications(g, df, show=False) if df is not None else None
    if show:
        print(f"\nCausal effect of {x} on {y}: {g}")
        if sets:
            print("  Minimal adjustment sets: " + "; ".join("{" + ", ".join(s) + "}" if s else "{} (no controls needed)"
                                                          for s in sets))
            print("  Optimal set (smallest variance among valid sets): "
                  + ("{" + ", ".join(O) + "}" if O is not None
                     else "not available (it would contain an unobserved variable, or there is no causal path)"))
        else:
            print("  NOT identified by adjustment: an unobserved confounder is open on every set. "
                  "Use an instrument, a panel design (DiD/FE) or a front-door path.")
        print(roles.to_string())
        if tests is not None and len(tests):
            bad = tests.loc[~tests["consistent"], "implication"].tolist()
            print(f"  Data checks: {len(tests)} implications, " + (f"rejected: {bad}" if bad else "none rejected"))
    return {"adjustment_sets": sets, "optimal_set": O, "roles": roles, "tests": tests, "mermaid": g.to_mermaid()}
