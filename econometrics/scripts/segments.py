"""
Segmented binary-response models: the logit leaf model.

A decision tree splits the sample into a few segments (its leaves); a separate logit is
estimated inside each leaf. Trees capture interactions but fit linear effects with many
steps; a pooled logit fits linear effects but needs every interaction written by hand. The
hybrid keeps a short list of readable rules plus one logit per segment (De Caigny, Coussement
& De Bock, 2018, European Journal of Operational Research 269(2): 760-772; R package `LLM`).

Use it when the question is prediction or targeting with heterogeneous segments (churn,
default, dropout, conversion) and you want something more readable than a forest. The
segments are chosen by the data, so they are hypotheses about heterogeneity, not findings:
with honest=True the tree and the leaf logits use different halves of the training data,
which makes the leaf standard errors valid conditional on the segments (Athey & Imbens, 2016).

Functions
---------
logit_leaf_model(df, y, x, ...)    fit tree + leaf logits; holdout AUC and top-decile lift vs pooled logit,
                                   tree and any benchmark formulas (e.g. interactions from theory)
leaf_model_stability(df, y, x)     refit over several seeds: which variables define the segments, and how
                                   often the leaf model beats the pooled logit
predict_leaf_model(model, df)      predicted probabilities for new rows
top_decile_lift(y, p)              share of positives in the top 10% of scores / overall share

The tree splits on node purity (CART), not on where the logit coefficients change. If the
heterogeneity is a change in slopes with little change in the churn rate, the tree can miss it:
pass the variable you expect to define segments in `segment_by` (you are warned if no split
uses it) and the theory-based interaction model in `benchmarks`, and prefer that model when it
wins on the holdout. Model-based recursive partitioning (partykit::glmtree) splits on parameter
instability and is the principled alternative.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def top_decile_lift(y, p, frac: float = 0.10) -> float:
    """Positive rate among the frac highest-scored observations divided by the overall rate."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    k = max(1, int(np.ceil(frac * len(y))))
    top = np.argsort(-p, kind="stable")[:k]
    base = y.mean()
    return float(y[top].mean() / base) if base > 0 else np.nan


def _rules(tree, names):
    """Text rule for each leaf id of a fitted sklearn tree."""
    t = tree.tree_
    out = {}

    def walk(node, conds):
        if t.children_left[node] == -1:
            out[node] = " and ".join(conds) if conds else "all"
            return
        f, thr = names[t.feature[node]], t.threshold[node]
        walk(t.children_left[node], conds + [f"{f} <= {thr:.4g}"])
        walk(t.children_right[node], conds + [f"{f} > {thr:.4g}"])

    walk(0, [])
    return out


def _split_variables(tree, names) -> list[str]:
    """Variables used in at least one split of a fitted sklearn tree, in order of first use."""
    seen = []
    for f in tree.tree_.feature:
        if f >= 0 and names[f] not in seen:
            seen.append(names[f])
    return seen


def _fit_leaf_logit(yl, Xl, names, select):
    """Logit in one leaf; forward selection by AIC if requested. Falls back to a constant
    (all outcomes equal) or to a ridge logit without SEs (perfect separation)."""
    import statsmodels.api as sm
    if yl.min() == yl.max():
        return {"kind": "constant", "p": float(yl.mean()), "vars": [], "res": None}

    def fit(vs):
        X = sm.add_constant(Xl[vs], has_constant="add") if vs else np.ones((len(yl), 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return sm.Logit(yl, X).fit(disp=False, maxiter=200)

    try:
        if select == "forward_aic":
            chosen, best = [], fit([]).aic
            improved = True
            while improved:
                improved = False
                cands = []
                for v in [v for v in names if v not in chosen]:
                    try:
                        cands.append((fit(chosen + [v]).aic, v))
                    except Exception:
                        continue
                if cands:
                    aic, v = min(cands)
                    if aic < best - 1e-9:
                        chosen.append(v); best = aic; improved = True
            vs = chosen
        else:
            vs = list(names)
        res = fit(vs)
        if not np.all(np.isfinite(res.bse)):
            raise np.linalg.LinAlgError("non-finite SEs")
        return {"kind": "logit", "vars": vs, "res": res}
    except Exception:
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(C=1.0, max_iter=2000).fit(Xl[names], yl)
        return {"kind": "ridge", "vars": list(names), "res": m}


def _leaf_predict(fit, Xl):
    import statsmodels.api as sm
    if fit["kind"] == "constant":
        return np.full(len(Xl), fit["p"])
    if fit["kind"] == "ridge":
        return fit["res"].predict_proba(Xl[fit["vars"]])[:, 1]
    X = sm.add_constant(Xl[fit["vars"]], has_constant="add") if fit["vars"] else np.ones((len(Xl), 1))
    return np.asarray(fit["res"].predict(X))


def logit_leaf_model(df: pd.DataFrame, y: str, x: list[str], max_depth: int = 3, min_leaf: int = 100,
                     ccp_alpha: float = 0.0, select: str = "forward_aic", honest: bool = True,
                     test_size: float = 0.3, seed: int = 0, benchmarks: dict[str, str] | None = None,
                     segment_by: list[str] | None = None, show: bool = True) -> dict:
    """Fit a logit leaf model and compare it with a pooled logit and the tree alone on a holdout.

    df, y, x     data, binary outcome (0/1) and numeric regressors (encode categories beforehand)
    max_depth    tree depth (2-3 keeps the rules readable)
    min_leaf     minimum observations per leaf; each leaf must support a logit (the R package uses 100)
    ccp_alpha    cost-complexity pruning (sklearn CART is used instead of C4.5; prune to keep few leaves)
    select       "forward_aic" (forward selection by AIC within each leaf) or "none" (all regressors)
    honest       True: tree on one half of the training data, leaf logits on the other half
    test_size    share of rows kept out for the AUC / top-decile-lift comparison
    benchmarks   {name: patsy formula} of extra logits fit on the same training rows and scored on the
                 same holdout, e.g. {"interactions": "churn ~ tenure + usage*b2b + price*b2b"}. Formulas
                 may use any column of df; rows where they cannot be evaluated raise an error.
    segment_by   variables the user expects to define the segments (customer type, region...). A
                 warning is printed and stored if no split of the tree uses any of them.

    Returns dict: leaves (rule, n, positive rate, kind), coefs (per-leaf coefficients with SEs),
    ame (per-leaf average marginal effects), performance (holdout AUC and lift), best (name of the
    model with the highest holdout AUC), split_variables, warnings, model (for predict_leaf_model),
    and the test indices.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    import statsmodels.api as sm

    d = df[[y] + list(x)].dropna()
    yy = d[y].astype(int).values
    if set(np.unique(yy)) - {0, 1}:
        raise ValueError("y must be binary 0/1.")
    tr_idx, te_idx = train_test_split(np.arange(len(d)), test_size=test_size, random_state=seed, stratify=yy)
    if honest:
        seg_idx, est_idx = train_test_split(tr_idx, test_size=0.5, random_state=seed + 1, stratify=yy[tr_idx])
    else:
        seg_idx = est_idx = tr_idx
    X = d[list(x)].reset_index(drop=True)

    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_leaf, ccp_alpha=ccp_alpha,
                                  random_state=seed).fit(X.iloc[seg_idx], yy[seg_idx])
    rules = _rules(tree, list(x))
    leaf_est = tree.apply(X.iloc[est_idx])

    fits, leaves, coefs, ames = {}, [], [], []
    for leaf in sorted(rules):
        m = leaf_est == leaf
        Xl, yl = X.iloc[est_idx][m], yy[est_idx][m]
        if len(yl) == 0:
            fits[leaf] = {"kind": "constant", "p": float(yy[seg_idx][tree.apply(X.iloc[seg_idx]) == leaf].mean()),
                          "vars": [], "res": None}
        else:
            fits[leaf] = _fit_leaf_logit(yl, Xl, list(x), select)
        f = fits[leaf]
        leaves.append({"leaf": leaf, "rule": rules[leaf], "n_estimation": int(m.sum()),
                       "positive_rate": float(yl.mean()) if len(yl) else np.nan, "kind": f["kind"],
                       "variables": ", ".join(f["vars"]) or "-"})
        if f["kind"] == "logit":
            r = f["res"]
            tab = pd.DataFrame({"coef": r.params, "se": r.bse, "p": r.pvalues})
            tab.index = ["const"] + f["vars"]
            coefs.append(tab.assign(leaf=leaf))
            if f["vars"]:
                me = r.get_margeff(at="overall").summary_frame()
                me.index = f["vars"]
                ames.append(me[["dy/dx", "Std. Err."]].rename(columns={"dy/dx": "ame", "Std. Err.": "se"}).assign(leaf=leaf))

    model = {"tree": tree, "fits": fits, "x": list(x), "rules": rules}
    p_leaf = predict_leaf_model(model, X.iloc[te_idx])

    Xtr = sm.add_constant(X.iloc[tr_idx], has_constant="add")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pooled = sm.Logit(yy[tr_idx], Xtr).fit(disp=False, maxiter=200)
    p_pool = np.asarray(pooled.predict(sm.add_constant(X.iloc[te_idx], has_constant="add")))
    tree_full = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_leaf, ccp_alpha=ccp_alpha,
                                       random_state=seed).fit(X.iloc[tr_idx], yy[tr_idx])
    p_tree = tree_full.predict_proba(X.iloc[te_idx])[:, 1]
    yte = yy[te_idx]
    scores = [("logit_leaf", p_leaf), ("pooled_logit", p_pool), ("tree", p_tree)]
    if benchmarks:
        import statsmodels.formula.api as smf
        full = df.loc[d.index]
        for name, formula in benchmarks.items():
            if name in {m for m, _ in scores}:
                raise ValueError(f"Benchmark name '{name}' clashes with a built-in model name.")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bm = smf.logit(formula, data=full.iloc[tr_idx]).fit(disp=False, maxiter=200)
            pb = np.asarray(bm.predict(full.iloc[te_idx]), float)
            if len(pb) != len(te_idx) or np.isnan(pb).any():
                raise ValueError(f"Benchmark '{name}' cannot be evaluated on every holdout row; "
                                 "drop rows with missing values in its variables first.")
            scores.append((name, pb))
    perf = pd.DataFrame({m: {"AUC": roc_auc_score(yte, p), "top_decile_lift": top_decile_lift(yte, p)}
                         for m, p in scores}).T
    best = perf["AUC"].idxmax()
    split_vars = _split_variables(tree, list(x))
    notes = []
    if segment_by:
        missing = [v for v in segment_by if v not in x]
        if missing:
            raise ValueError(f"segment_by variables {missing} are not among the regressors x.")
        if not set(segment_by) & set(split_vars):
            notes.append(f"No split uses {segment_by}; the tree split on {split_vars or 'nothing'}. The tree "
                         "looks for differences in the outcome rate, not in slopes: if you expect the effects "
                         "to differ by these variables, compare with interactions in `benchmarks`.")
    if best != "logit_leaf":
        notes.append(f"The logit leaf model does not have the best holdout AUC ({best} does): report {best}, "
                     "and treat the segments as descriptive.")

    out = {"leaves": pd.DataFrame(leaves).set_index("leaf"),
           "coefs": pd.concat(coefs) if coefs else pd.DataFrame(),
           "ame": pd.concat(ames) if ames else pd.DataFrame(),
           "performance": perf, "best": best, "split_variables": split_vars, "warnings": notes,
           "model": model, "test_index": d.index[te_idx], "honest": honest}
    if show:
        print(f"\nLogit leaf model: {len(rules)} segments, {len(tr_idx)} training rows "
              f"({'honest: tree and logits on separate halves' if honest else 'tree and logits on the same rows'}), "
              f"{len(te_idx)} holdout rows.")
        print(out["leaves"].to_string())
        if len(out["ame"]):
            print("\nAverage marginal effects by segment (change in probability per unit of x):")
            print(out["ame"].round(4).to_string())
        print("\nHoldout performance:")
        print(perf.round(3).to_string())
        if not honest:
            print("  Leaf SEs ignore that the segments were chosen on the same data: do not use them for inference.")
        for note in notes:
            print("  WARNING: " + note)
        print("  Segments are data-driven hypotheses about heterogeneity; check that they are stable across seeds "
              "(leaf_model_stability).")
    return out


def leaf_model_stability(df: pd.DataFrame, y: str, x: list[str], seeds=range(5), show: bool = True,
                         **kwargs) -> dict:
    """Refit the logit leaf model with several seeds (new train/holdout and honest splits each time).

    A segment is worth reporting only if it reappears; a holdout gain is worth claiming only if it
    is consistent. kwargs go to logit_leaf_model (max_depth, min_leaf, benchmarks, segment_by...).

    Returns dict: by_seed (rules, split variables, AUC and lift of each model per seed),
    split_frequency (share of seeds in which each variable defines a split), and summary (mean AUC
    and lift per model, and the share of seeds in which each model has the best AUC).
    """
    rows = []
    for s in seeds:
        r = logit_leaf_model(df, y, x, seed=s, show=False, **kwargs)
        row = {"seed": s, "rules": " | ".join(r["leaves"]["rule"]), "split_variables": ", ".join(r["split_variables"]),
               "best": r["best"]}
        for m in r["performance"].index:
            row[f"AUC_{m}"] = r["performance"].loc[m, "AUC"]
            row[f"lift_{m}"] = r["performance"].loc[m, "top_decile_lift"]
        rows.append(row)
    by_seed = pd.DataFrame(rows).set_index("seed")
    used = by_seed["split_variables"].str.split(", ")
    freq = pd.Series({v: float(np.mean([v in u for u in used])) for v in x}, name="share_of_seeds").sort_values(
        ascending=False)
    models = [c[4:] for c in by_seed.columns if c.startswith("AUC_")]
    summary = pd.DataFrame({m: {"mean_AUC": by_seed[f"AUC_{m}"].mean(),
                                "mean_top_decile_lift": by_seed[f"lift_{m}"].mean(),
                                "share_best_AUC": float((by_seed["best"] == m).mean())} for m in models}).T
    if show:
        print(f"\nLogit leaf model over {len(by_seed)} seeds.")
        print("Share of seeds in which each variable defines a split:")
        print(freq.round(2).to_string())
        print("\nHoldout performance across seeds:")
        print(summary.round(3).to_string())
        gain = (by_seed["AUC_logit_leaf"] > by_seed["AUC_pooled_logit"]).mean()
        print(f"  The logit leaf model beats the pooled logit in AUC in {gain:.0%} of seeds.")
    return {"by_seed": by_seed, "split_frequency": freq, "summary": summary}


def predict_leaf_model(model: dict, X: pd.DataFrame) -> np.ndarray:
    """Probabilities from a fitted logit leaf model for new rows (same columns as in fitting)."""
    X = X[model["x"]]
    leaf = model["tree"].apply(X)
    p = np.empty(len(X))
    for lf, f in model["fits"].items():
        m = leaf == lf
        if m.any():
            p[m] = _leaf_predict(f, X[m])
    return p
