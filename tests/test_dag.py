"""Tests for v2.2 dag.py: d-separation, backdoor adjustment sets, the optimal adjustment set,
classification of controls and testable implications.

Checks on textbook graphs (Pearl 2009; Cinelli, Forney & Pearl 2022): chain, fork and collider
d-separation; confounding; M-bias (do not control the collider); butterfly bias (two minimal
sets); an unobserved confounder with an instrument (not identified by adjustment, instrument
flagged); a mediator and a post-treatment collider; the Henckel-Perkovic-Maathuis O-set adds
outcome predictors and drops instruments. Simulations of linear SEMs: OLS with a minimal set
recovers the effect while bad controls bias it; the O-set has the smallest SD among valid sets;
implied independencies hold in data from the true graph and one is rejected when an edge is
missing from the graph.

Run from the repository root:  python3 tests/test_dag.py
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "econometrics" / "scripts"))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import dag

ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


# d-separation basics
check("chain A->B->C: A _||_ C | B, not marginally", dag.d_separated("A->B->C", "A", "C", "B")
      and not dag.d_separated("A->B->C", "A", "C"))
check("fork A<-B->C: A _||_ C | B, not marginally", dag.d_separated("B->A; B->C", "A", "C", "B")
      and not dag.d_separated("B->A; B->C", "A", "C"))
check("collider A->B<-C: A _||_ C marginally, not given B or its child",
      dag.d_separated("A->B; C->B; B->D", "A", "C") and not dag.d_separated("A->B; C->B; B->D", "A", "C", "B")
      and not dag.d_separated("A->B; C->B; B->D", "A", "C", "D"))
try:
    dag.DAG("A->B; B->C; C->A")
    check("cycle rejected", False)
except ValueError:
    check("cycle rejected", True)

# Textbook graphs
check("confounding: minimal set {Z}", dag.adjustment_sets("Z->X; Z->Y; X->Y", "X", "Y") == [["Z"]])
gm = dag.DAG("U1->X; U1->M; U2->M; U2->Y; X->Y", latent={"U1", "U2"})
rm = dag.classify_controls(gm, "X", "Y")
check("M-bias: empty set valid, {M} invalid, M classified collider / do not control",
      dag.adjustment_sets(gm, "X", "Y") == [[]] and not dag.is_valid_adjustment(gm, "X", "Y", {"M"})
      and rm.loc["M", "role"].startswith("collider") and rm.loc["M", "recommendation"] == "do not control")
gb = "Z1->X; Z1->M; Z2->M; Z2->Y; M->X; M->Y; X->Y"
check("butterfly: minimal sets {M,Z1} and {M,Z2}; {M} alone invalid",
      dag.adjustment_sets(gb, "X", "Y") == [["M", "Z1"], ["M", "Z2"]] and not dag.is_valid_adjustment(gb, "X", "Y", {"M"}))
giv = dag.DAG("Z->X->Y; X<->Y")
check("unobserved confounder: no valid adjustment set; Z flagged instrument-like",
      dag.adjustment_sets(giv, "X", "Y") == []
      and dag.classify_controls(giv, "X", "Y").loc["Z", "role"].startswith("instrument"))
gmed = "X->M->Y; X->Y; C->X; C->Y; X->K; Y->K"
rmed = dag.classify_controls(gmed, "X", "Y")
check("mediator M and post-treatment K: do not control; C confounder",
      rmed.loc["M", "role"].startswith("mediator") and rmed.loc["K", "recommendation"].startswith("do not control")
      and rmed.loc["K", "role"].startswith("descendant of the outcome")
      and rmed.loc["C", "role"].startswith("confounder") and not dag.is_valid_adjustment(gmed, "X", "Y", {"C", "M"}))
go = "X->Y; W->Y; Z->X; C->X; C->Y"
ro = dag.classify_controls(go, "X", "Y")
check("O-set = {C, W}: adds the outcome predictor, drops the instrument",
      dag.optimal_adjustment_set(go, "X", "Y") == ["C", "W"] and ro.loc["Z", "role"].startswith("instrument")
      and ro.loc["W", "recommendation"].startswith("optional"))
check("butterfly O-set = {M, Z2}", dag.optimal_adjustment_set(gb, "X", "Y") == ["M", "Z2"])
check("mermaid export marks latent nodes with dashed edges", "-.->" in giv.to_mermaid())


# Simulation: linear SEM of the butterfly + bad controls
def ols_coef(df, y, xs):
    X = np.column_stack([np.ones(len(df)), df["X"]] + [df[c] for c in xs])
    return np.linalg.lstsq(X, df[y], rcond=None)[0][1]


def butterfly(n, seed):
    r = np.random.default_rng(seed)
    Z1, Z2 = r.normal(size=(2, n))
    M = Z1 + Z2 + r.normal(size=n)
    X = 0.8 * Z1 + 0.8 * M + r.normal(size=n)
    Y = 0.5 * X + 0.8 * M + 0.8 * Z2 + r.normal(size=n)
    K = X + Y + r.normal(size=n)
    return pd.DataFrame(dict(Z1=Z1, Z2=Z2, M=M, X=X, Y=Y, K=K))


est = {k: [] for k in ["M,Z1", "M,Z2", "M", "M,Z2,K", "none"]}
for s in range(200):
    d = butterfly(1000, s)
    for k in est:
        est[k].append(ols_coef(d, "Y", [] if k == "none" else k.split(",")))
m = {k: np.mean(v) for k, v in est.items()}
sd = {k: np.std(v) for k, v in est.items()}
check("OLS with each minimal set is unbiased (within 0.02 of 0.5)", abs(m["M,Z1"] - 0.5) < 0.02 and abs(m["M,Z2"] - 0.5) < 0.02,
      {k: round(v, 3) for k, v in m.items()})
check("{M} alone (butterfly bias), no controls, and adding the collider K are biased (> 0.05)",
      abs(m["M"] - 0.5) > 0.05 and abs(m["none"] - 0.5) > 0.05 and abs(m["M,Z2,K"] - 0.5) > 0.05)
check("O-set {M, Z2} has a smaller SD than the other valid set {M, Z1}", sd["M,Z2"] < sd["M,Z1"],
      f"{sd['M,Z2']:.4f} vs {sd['M,Z1']:.4f}")

# Testable implications
d = butterfly(3000, 99)
gtrue = dag.DAG(gb + "; X->K; Y->K")
imp = dag.implied_independencies(gtrue)
t_ok = dag.test_implications(gtrue, d, show=False)
check("true graph: every implied independence passes (Holm, 5%)", len(t_ok) > 0 and t_ok["consistent"].all(),
      f"{len(t_ok)} implications, min p = {t_ok['p'].min():.3f}")
gwrong = dag.DAG("Z1->X; Z1->M; Z2->M; M->X; M->Y; X->Y; X->K; Y->K")    # missing Z2 -> Y
t_bad = dag.test_implications(gwrong, d, show=False)
check("graph missing Z2 -> Y: the implication Z2 _||_ Y | ... is rejected",
      (~t_bad["consistent"]).any() and t_bad.loc[~t_bad["consistent"], "implication"].str.contains("Z2").any(),
      t_bad.loc[~t_bad["consistent"], "implication"].tolist())
rep = dag.adjustment_report(gtrue, "X", "Y", df=d, show=False)
check("adjustment_report bundles sets, O-set, roles, tests and mermaid",
      rep["adjustment_sets"] == [["M", "Z1"], ["M", "Z2"]] and rep["optimal_set"] == ["M", "Z2"]
      and len(rep["roles"]) == 4 and rep["mermaid"].startswith("graph LR"))

# v2.2 eval: a graph without the edge x -> y assumes a zero effect and must say so
noedge = "S -> A; S -> P; A -> T; P -> T"
r_no = dag.adjustment_report(noedge, "A", "P", show=False)
r_ok = dag.adjustment_report(noedge + "; A -> P", "A", "P", show=False)
check("missing edge x -> y: warning raised and optimal set unavailable; with the edge, no warning and O-set {S}",
      len(r_no["warnings"]) == 1 and r_no["optimal_set"] is None and not r_ok["warnings"] and r_ok["optimal_set"] == ["S"])


# --- v2.7.4: a mediator that is also a collider (chat of 2026-11-18: Tenure with an unobserved parent M) ----------
base74 = "Age -> X; Age -> Y; X -> Y; X -> Tenure; Tenure -> Y"
c_lat = dag.classify_controls(dag.DAG(base74 + "; M -> Tenure; M -> Y", latent=["M"]), "X", "Y")
r_t = c_lat.loc["Tenure"]
check("v2.7.4 mediator with an unobserved parent that causes y: 'mediator ... and collider', naming M",
      r_t["role"] == "mediator (x -> v -> y) and collider (x -> v <- M -> ... -> y)", r_t["role"])
check("v2.7.4 ... and the recommendation says the direct effect is not identified by adjusting for it",
      r_t["recommendation"].startswith("do not control") and "M unobserved" in r_t["recommendation"]
      and "not even the direct effect" in r_t["recommendation"], r_t["recommendation"])
c_obs = dag.classify_controls(dag.DAG(base74 + "; M -> Tenure; M -> Y"), "X", "Y")
r_o = c_obs.loc["Tenure"]
check("v2.7.4 same graph with M observed: still marked, and the recommendation says to control M for the direct effect",
      "and collider (x -> v <- M" in r_o["role"] and "control M too" in r_o["recommendation"]
      and "unobserved" not in r_o["recommendation"], r_o.to_dict())
c_w = dag.classify_controls(dag.DAG(base74 + "; W -> Tenure"), "X", "Y")
check("v2.7.4 a parent of the mediator with no other path to y (W -> Tenure only) does not make it a collider",
      c_w.loc["Tenure", "role"] == "mediator (x -> v -> y)", c_w.loc["Tenure", "role"])
c_a = dag.classify_controls(dag.DAG(base74 + "; Age -> Tenure"), "X", "Y")
check("v2.7.4 a parent already blocked by the minimal set (the confounder Age -> Tenure) does not make it a collider",
      c_a.loc["Tenure", "role"] == "mediator (x -> v -> y)", c_a.loc["Tenure", "role"])
c_d = dag.classify_controls(dag.DAG(base74 + "; X -> D; D -> Tenure"), "X", "Y")
check("v2.7.4 a parent that descends from x (X -> D -> Tenure) is not a collider parent",
      c_d.loc["Tenure", "role"] == "mediator (x -> v -> y)", c_d.loc["Tenure", "role"])
c_2 = dag.classify_controls(dag.DAG(base74 + "; M -> Tenure; M -> Y; N -> Tenure; N -> Y", latent=["M"]), "X", "Y")
check("v2.7.4 two collider parents are both named, sorted, and only the latent one is called unobserved",
      "x -> v <- M, N ->" in c_2.loc["Tenure", "role"] and "with M unobserved" in c_2.loc["Tenure", "recommendation"],
      c_2.loc["Tenure"].to_dict())
# the claim itself, by simulation: linear SEM of the first graph, regression of Y on X with Age and Tenure
rng74 = np.random.default_rng(7405)
nn = 200_000
age, m_ = rng74.normal(size=nn), rng74.normal(size=nn)
xx = 0.5 * age + rng74.normal(size=nn)
ten = 0.8 * xx + 0.9 * m_ + rng74.normal(size=nn)
yy = 1.0 * xx + 0.5 * ten + 0.7 * age + 0.8 * m_ + rng74.normal(size=nn)
Xd = np.column_stack([np.ones(nn), xx, age, ten])
b_ad = np.linalg.lstsq(Xd, yy, rcond=None)[0][1]
# closed form of the X coefficient given Age and Tenure: direct effect 1 plus the collider bias -0.8*0.9*0.8/(0.9^2+1)
bias = -0.8 * (0.9 * 0.8) / (0.9 ** 2 + 1.0)
check("v2.7.4 simulation: controlling the mediator-collider gives neither the direct effect (1) nor the total; it "
      "matches 1 + the collider bias (closed form)", abs(b_ad - (1 + bias)) < 0.01 and abs(b_ad - 1) > 0.25,
      f"{b_ad:.4f} vs {1 + bias:.4f}")

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
