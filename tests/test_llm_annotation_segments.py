"""Tests for the v1.7 tools: LLM annotation with valid downstream inference, and the logit leaf model.

text_data.py:  label_probs_from_logits, logtoku_uncertainty, llm_classify (with a fake backend),
               dsl_sampling_probs, draw_validation_sample, dsl_regression
segments.py:   logit_leaf_model (benchmarks, segment_by), leaf_model_stability, predict_leaf_model,
               top_decile_lift

Benchmarks: the closed-form LogTokU formulas (Ma et al., 2025, Eqs. 3-5, 9); equivalence of a
restricted softmax with a -inf logits mask; exact identities of the design-based estimator
(all documents hand-coded -> OLS on the truth; perfect surrogate -> full-sample OLS); Monte Carlo
bias and 95% coverage with differential (x-dependent) coding error, where treating LLM codes as
truth is biased; a known two-segment data-generating process for the logit leaf model.
v1.8: identification checks of dsl_regression (incl. the shared-seed trap) and a process whose
slopes differ by a variable the purity-based tree never splits on, where a theory benchmark wins.
No language model is downloaded: llm_classify runs on a deterministic fake backend.

Run from the repository root:  python3 tests/test_llm_annotation_segments.py
Needs: numpy, pandas, scipy, statsmodels, scikit-learn.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "econometrics" / "scripts"))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import digamma

import segments as sg
import text_data as td

ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


def raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


# ---------------------------------------------------------------------------------------------
# Constrained label probabilities
# ---------------------------------------------------------------------------------------------
rng = np.random.default_rng(0)
V = 50
logits = rng.normal(size=V) * 3
ids = [7, 21, 33]
lp = td.label_probs_from_logits(logits, ids)
masked = np.full(V, -np.inf); masked[ids] = logits[ids]
soft = np.exp(masked - masked.max()); soft /= soft.sum()
check("restricted softmax = softmax after a -inf mask on non-label tokens", np.allclose(lp["probs"], soft[ids]))
full = np.exp(logits - logits.max()); full /= full.sum()
check("mass_on_labels = unconstrained probability on the label tokens",
      np.isclose(lp["mass_on_labels"], full[ids].sum()), f"{lp['mass_on_labels']:.4f}")
batch = td.label_probs_from_logits(np.vstack([logits, logits[::-1]]), ids)
check("batch input gives one row per document", batch["probs"].shape == (2, 3) and np.allclose(batch["probs"][0], lp["probs"]))
check("duplicate label token ids are rejected", raises(lambda: td.label_probs_from_logits(logits, [3, 3])))

# ---------------------------------------------------------------------------------------------
# LogTokU (Ma et al., 2025): closed form and the quadrant logic
# ---------------------------------------------------------------------------------------------
lg = np.array([9.0, 4.0, 2.5, 1.0, -3.0, 0.5])
k = 4
a = np.sort(lg)[::-1][:k]; a0 = a.sum()
au_hand = -np.sum(a / a0 * (digamma(a + 1) - digamma(a0 + 1)))
eu_hand = k / np.sum(a + 1)
u = td.logtoku_uncertainty(lg, k=k)
check("LogTokU aleatoric uncertainty matches Eq. 4", np.isclose(u["aleatoric"], au_hand), f"{u['aleatoric']:.4f}")
check("LogTokU epistemic uncertainty matches Eq. 5", np.isclose(u["epistemic"], eu_hand), f"{u['epistemic']:.4f}")
check("reliability = 1/(AU·EU) (Eq. 9)", np.isclose(u["reliability"], 1 / (au_hand * eu_hand)))
one_strong = np.r_[30.0, np.zeros(9)]          # one answer with a lot of evidence
many_strong = np.full(10, 30.0)                 # several answers, each with a lot of evidence
little = np.full(10, 0.3)                       # no answer has evidence
U = td.logtoku_uncertainty(np.vstack([one_strong, many_strong, little]), k=10)
check("several well-supported answers -> higher aleatoric uncertainty than one",
      U["aleatoric"][1] > U["aleatoric"][0], f"{U['aleatoric'][1]:.3f} > {U['aleatoric'][0]:.3f}")
check("little total evidence -> highest epistemic uncertainty",
      U["epistemic"][2] > U["epistemic"][0] > U["epistemic"][1], np.round(U["epistemic"], 3))
check("same softmax, different evidence: probability cannot tell them apart, LogTokU can",
      np.isclose(td._softmax(np.full(10, 0.3))[0], td._softmax(np.full(10, 30.0))[0])
      and U["epistemic"][2] > 5 * U["epistemic"][1])
check("negative logits count as zero evidence (no NaN)",
      np.all(np.isfinite(list(td.logtoku_uncertainty(-np.abs(lg), k=4).values())[:2])))

# ---------------------------------------------------------------------------------------------
# llm_classify with a deterministic fake backend
# ---------------------------------------------------------------------------------------------
vocab = {" A": [10, 99], " B": [11], " C": [12], " AB": [10, 5]}


def encode(s):
    return vocab.get(s, [0])


def fake_logits(prompt):
    z = np.zeros(40)
    doc = prompt.split("TEXT:")[1]
    if "julgo procedente" in doc:
        z[10] = 8.0
    elif "improcedente" in doc:
        z[11] = 8.0
    else:
        z[[10, 11, 12]] = 2.0; z[3] = 9.0      # ambiguous text: the model prefers a non-label token
    return z


texts = ["... julgo procedente o pedido ...", "... julgo improcedente ...", "... extingo sem resolução ..."]
res = td.llm_classify(texts, {"A": "granted", "B": "denied", "C": "other"},
                      "Classify the decision. TEXT:{text}\nAnswer:", fake_logits, encode, show=False)
check("llm_classify returns the right labels", res["label_hat"].tolist()[:2] == ["granted", "denied"],
      res["label_hat"].tolist())
check("label probabilities sum to one", np.allclose(res[["p_granted", "p_denied", "p_other"]].sum(axis=1), 1))
check("ambiguous document is flagged by low mass on labels and high uncertainty",
      res["mass_on_labels"].iloc[2] < 0.5 and res["reliability"].iloc[2] < res["reliability"].iloc[0],
      res["mass_on_labels"].round(3).tolist())
check("codes sharing a first token are rejected",
      raises(lambda: td.llm_classify(texts, ["A", "AB"], "TEXT:{text}", fake_logits, encode, show=False)))

# ---------------------------------------------------------------------------------------------
# Sampling probabilities for hand coding
# ---------------------------------------------------------------------------------------------
score = rng.uniform(0, 1, 2000) ** 3
pi = td.dsl_sampling_probs(score, 300, uniform_share=0.2, max_pi=0.5)
check("inclusion probabilities sum to the expected sample size", np.isclose(pi.sum(), 300), f"{pi.sum():.2f}")
check("every pi is at least the uniform floor and at most max_pi",
      pi.min() >= 0.2 * 300 / 2000 - 1e-12 and pi.max() <= 0.5 + 1e-12, f"[{pi.min():.4f}, {pi.max():.3f}]")
check("pi increases with the uncertainty score", np.all(np.diff(pi[np.argsort(score)]) >= -1e-12))
draws = np.mean([td.draw_validation_sample(pi, s).sum() for s in range(200)])
check("Bernoulli draws hit the expected size on average", abs(draws - 300) < 3, f"{draws:.1f}")

# ---------------------------------------------------------------------------------------------
# dsl_regression: exact identities
# ---------------------------------------------------------------------------------------------
n = 800
x = rng.normal(size=n)
d_true = (rng.random(n) < 0.4).astype(float)
yv = 1 + 0.7 * d_true + 0.5 * x + rng.normal(size=n)
d_hat = np.where(rng.random(n) < 0.2, 1 - d_true, d_true)
df = pd.DataFrame(dict(y=yv, x=x, d_true=d_true, d_hat=d_hat))
ols_true = sm.OLS(yv, sm.add_constant(np.column_stack([d_true, x]))).fit(cov_type="HC0")
allR = td.dsl_regression(df, "y", ["d", "x"], np.ones(n, bool), np.ones(n), coded={"d": ("d_true", "d_hat")}, show=False)
check("all documents hand-coded (pi = 1) -> OLS on the true data, HC0 SEs",
      np.allclose(allR["table"]["coef"].values, ols_true.params) and np.allclose(allR["table"]["se"].values, ols_true.bse),
      allR["table"]["coef"].round(4).tolist())
R = rng.random(n) < 0.2
df_p = df.assign(d_true=np.where(R, d_true, np.nan), d_perfect=d_true)
perfect = td.dsl_regression(df_p, "y", ["d", "x"], R, np.full(n, 0.2), coded={"d": ("d_true", "d_perfect")}, show=False)
check("perfect surrogate -> full-sample OLS whatever the sample drawn",
      np.allclose(perfect["table"]["coef"].values, ols_true.params))
check("perfect surrogate -> large precision gain over hand-coded documents alone",
      perfect["design_effect"]["d"] < 0.4, f"{perfect['design_effect']['d']:.2f}")
check("pi outside (0, 1] is rejected",
      raises(lambda: td.dsl_regression(df_p, "y", ["d", "x"], R, np.zeros(n), coded={"d": ("d_true", "d_hat")}, show=False)))
cl = td.dsl_regression(df_p, "y", ["d", "x"], R, np.full(n, 0.2), coded={"d": ("d_true", "d_hat")},
                       cluster=np.arange(n) // 4, show=False)
check("clustered SEs run and are positive", np.all(cl["table"]["se"] > 0))

# ---------------------------------------------------------------------------------------------
# dsl_regression: Monte Carlo with differential coding error
# ---------------------------------------------------------------------------------------------
def sim_ols(seed):
    r = np.random.default_rng(seed); n = 3000
    x = r.normal(size=n)
    d = (r.random(n) < 1 / (1 + np.exp(-0.5 * x))).astype(float)
    y = 1 + 0.8 * d + 0.5 * x + r.normal(size=n)
    flip = r.random(n) < np.where(d == 1, 0.1 + 0.25 * (x > 0), 0.08)   # misses more positives when x > 0
    dh = np.where(flip, 1 - d, d)
    pi = td.dsl_sampling_probs(np.where(np.abs(x) > 1, 0.5, 0.2), 300)
    Rr = td.draw_validation_sample(pi, seed + 10_000)
    df = pd.DataFrame(dict(y=y, x=x, d_true=np.where(Rr, d, np.nan), d_hat=dh))
    o = td.dsl_regression(df, "y", ["d", "x"], Rr, pi, coded={"d": ("d_true", "d_hat")}, show=False)
    return o["table"].loc["d", "coef"], o["table"].loc["d", "se"], o["naive"].loc["d", "coef"]


def sim_logit(seed):
    r = np.random.default_rng(seed); n = 3000
    x = r.normal(size=n)
    y = (r.random(n) < 1 / (1 + np.exp(-(-0.5 + x)))).astype(float)
    p = np.clip(0.75 * y + 0.15 + 0.1 * (x > 0), 0.01, 0.99)                # LLM probability, error depends on x
    pi = td.dsl_sampling_probs(np.sqrt(p * (1 - p)), 400)
    Rr = td.draw_validation_sample(pi, seed + 10_000)
    df = pd.DataFrame(dict(x=x, y_true=np.where(Rr, y, np.nan), y_hat=p))
    o = td.dsl_regression(df, "y", ["x"], Rr, pi, coded={"y": ("y_true", "y_hat")}, model="logit", show=False)
    return o["table"].loc["x", "coef"], o["table"].loc["x", "se"], o["naive"].loc["x", "coef"]


for label, fn, truth in [("OLS, LLM-coded regressor", sim_ols, 0.8), ("logit, LLM-coded outcome", sim_logit, 1.0)]:
    sims = np.array([fn(s) for s in range(200)])
    b, se, naive = sims[:, 0], sims[:, 1], sims[:, 2]
    cover = np.mean(np.abs(b - truth) <= 1.96 * se)
    check(f"DSL {label}: unbiased", abs(b.mean() - truth) < 0.03, f"mean {b.mean():.3f} vs {truth}")
    check(f"DSL {label}: 95% CI coverage in [0.92, 0.98]", 0.92 <= cover <= 0.98, f"{cover:.3f}")
    check(f"DSL {label}: SE matches the Monte Carlo spread", abs(se.mean() / b.std() - 1) < 0.15,
          f"mean SE {se.mean():.3f}, sd {b.std():.3f}")
    check(f"{label}: treating LLM codes as truth is biased", abs(naive.mean() - truth) > 0.1,
          f"naive mean {naive.mean():.3f}")

# ---------------------------------------------------------------------------------------------
# Logit leaf model
# ---------------------------------------------------------------------------------------------
yb = np.r_[np.ones(10), np.zeros(90)]
check("top-decile lift of a perfect score = 1 / base rate", np.isclose(sg.top_decile_lift(yb, yb), 10.0))
check("top-decile lift of a constant score ≈ 1 (ties kept in original order)",
      np.isclose(sg.top_decile_lift(rng.permutation(np.tile(yb, 20)), np.zeros(2000)), 1.0, atol=0.3))

r = np.random.default_rng(1); n = 6000
x1, x2, x3 = r.normal(size=(3, n))
eta = np.where(x1 > 0, -0.5 + 1.5 * x2, 0.3 - 1.5 * x2)          # opposite slopes in two segments
seg = pd.DataFrame(dict(y=(r.random(n) < 1 / (1 + np.exp(-eta))).astype(int), x1=x1, x2=x2, x3=x3))
llm = sg.logit_leaf_model(seg, "y", ["x1", "x2", "x3"], max_depth=1, min_leaf=200, show=False)
thr = llm["model"]["tree"].tree_.threshold[0]
check("tree finds the true segment boundary (x1 = 0)", llm["model"]["tree"].tree_.feature[0] == 0 and abs(thr) < 0.15,
      f"split x1 at {thr:.3f}")
ame = llm["ame"].reset_index().set_index(["leaf", "index"])["ame"]
leaves = sorted(llm["leaves"].index)
check("leaf logits recover opposite effects of x2",
      ame.loc[(leaves[0], "x2")] < -0.15 and ame.loc[(leaves[1], "x2")] > 0.15,
      f"{ame.loc[(leaves[0], 'x2')]:.3f}, {ame.loc[(leaves[1], 'x2')]:.3f}")
perf = llm["performance"]
check("holdout AUC: logit leaf beats pooled logit and tree by > 0.1",
      perf.loc["logit_leaf", "AUC"] > perf.loc[["pooled_logit", "tree"], "AUC"].max() + 0.1, perf["AUC"].round(3).tolist())
p_new = sg.predict_leaf_model(llm["model"], seg.iloc[:50])
check("predictions are probabilities", p_new.shape == (50,) and np.all((p_new > 0) & (p_new < 1)))
check("honest split: logits estimated on rows not used to grow the tree",
      llm["honest"] and llm["leaves"]["n_estimation"].sum() < 0.5 * 0.7 * n + 5)

r = np.random.default_rng(2); n = 3000
z1, z2 = r.normal(size=(2, n))
yc = np.where(z1 > 1, 0, (r.random(n) < 1 / (1 + np.exp(-z2))).astype(int))   # no positives when z1 > 1
const = sg.logit_leaf_model(pd.DataFrame(dict(y=yc, z1=z1, z2=z2)), "y", ["z1", "z2"], max_depth=2, min_leaf=100,
                            honest=False, select="none", show=False)
check("a leaf with a single outcome falls back to a constant probability",
      "constant" in const["leaves"]["kind"].values, const["leaves"]["kind"].tolist())

# ---------------------------------------------------------------------------------------------
# v1.8: identification checks in dsl_regression
# ---------------------------------------------------------------------------------------------
r = np.random.default_rng(11); n = 1500
g = r.binomial(1, 0.4, n); z = r.normal(size=n)
yd = r.binomial(1, 1 / (1 + np.exp(-(0.5 * g + 0.3 * z))))
dd = pd.DataFrame(dict(g=g, z=z, p_hat=np.clip(yd * 0.8 + 0.1 + r.normal(0, 0.1, n), 0.01, 0.99)))
dd["pi"] = 0.15
lab = (r.random(n) < 0.15) & (g == 0)                      # a draw that never reaches g = 1
dd["y_true"] = np.where(lab, yd, np.nan)
msg = ""
try:
    td.dsl_regression(dd, "y", ["g", "z"], lab, "pi", coded={"y": ("y_true", "p_hat")}, model="logit", show=False)
except ValueError as e:
    msg = str(e)
check("dsl_regression names the regressor that does not vary among hand-coded documents",
      "'g'" in msg and "seed" in msg, msg[:70])

# the trap that motivated the check: one seed used both to generate the data and to draw the sample
rs = np.random.default_rng(2026); gs = rs.binomial(1, 0.4, n)
lab_s = td.draw_validation_sample(np.full(n, 0.15), seed=2026)
check("a draw sharing the data seed is caught (the check fires on the shared-seed trap)",
      gs[lab_s].max() == 0 and raises(lambda: td.dsl_regression(
          pd.DataFrame(dict(g=gs, z=z, p_hat=dd.p_hat, pi=0.15, y_true=np.where(lab_s, yd, np.nan))),
          "y", ["g", "z"], lab_s, "pi", coded={"y": ("y_true", "p_hat")}, model="logit", show=False)))

lab2 = r.random(n) < 0.15
dd["y_const"] = np.where(lab2, 1.0, np.nan)
check("logit DSL refuses a hand-coded outcome with a single value",
      raises(lambda: td.dsl_regression(dd, "y", ["g", "z"], lab2, "pi", coded={"y": ("y_const", "p_hat")},
                                       model="logit", show=False)))
dd["zz"] = 2 * dd["z"]
check("collinear regressors among hand-coded documents are reported",
      raises(lambda: td.dsl_regression(dd.assign(y_true=np.where(lab2, yd, np.nan)), "y", ["z", "zz"], lab2, "pi",
                                       coded={"y": ("y_true", "p_hat")}, model="logit", show=False)))

# separated hand-coded sample: labeled-only benchmark fails, DSL is still reported
lab3 = np.zeros(n, bool); lab3[:60] = True
ysep = np.where(dd.z > 0, 1, 0)
dsep = dd.assign(y_true=np.where(lab3, ysep, np.nan), p_hat=np.clip(0.5 + 0.3 * np.tanh(dd.z), 0.01, 0.99))
with warnings.catch_warnings(record=True) as wlist:
    warnings.simplefilter("always")
    try:
        rsep = td.dsl_regression(dsep, "y", ["z"], lab3, np.full(n, 60 / n), coded={"y": ("y_true", "p_hat")},
                                 model="logit", show=False)
        sep_ok = (np.all(np.isfinite(rsep["table"]["coef"])) and rsep["design_effect"].isna().all()
                  and any("labeled-only benchmark" in str(w.message) for w in wlist))
    except np.linalg.LinAlgError:
        sep_ok = False
check("separated hand-coded sample: DSL reported, design_effect NaN with a warning (not a spurious ~0)", sep_ok)

# ---------------------------------------------------------------------------------------------
# v1.8: benchmarks, segment_by and stability for the logit leaf model
# ---------------------------------------------------------------------------------------------
r = np.random.default_rng(7); n = 6000
ten = r.exponential(24, n); use = r.normal(size=n); comp = r.poisson(1, n)
price = r.binomial(1, 0.3, n); b2b = r.binomial(1, 0.35, n)
eta = (-1.8 + 0.9 * (ten < 12) - 0.03 * np.minimum(ten, 60) + np.where(b2b == 1, 0.6, -0.8) * use
       + 0.35 * comp + np.where(b2b == 1, 0.2, 1.4) * price)       # slopes, not rates, differ by b2b
ch = pd.DataFrame(dict(churn=r.binomial(1, 1 / (1 + np.exp(-eta))), tenure=ten, usage=use, complaints=comp,
                       price_increase=price, b2b=b2b))
xs = ["tenure", "usage", "complaints", "price_increase", "b2b"]
bm = {"interactions": "churn ~ tenure + usage*b2b + complaints + price_increase*b2b"}
rl = sg.logit_leaf_model(ch, "churn", xs, max_depth=2, min_leaf=150, benchmarks=bm, segment_by=["b2b"], show=False)
check("benchmark formula is scored on the same holdout", "interactions" in rl["performance"].index)
check("when the heterogeneity is in slopes, the interaction benchmark wins and is named as best",
      rl["best"] == "interactions", rl["performance"]["AUC"].round(3).tolist())
check("segment_by warns when no split uses the variable", any("No split uses ['b2b']" in w for w in rl["warnings"]),
      rl["split_variables"])
check("a non-winning leaf model is flagged", any("does not have the best holdout AUC" in w for w in rl["warnings"]))
check("segment_by outside x is rejected",
      raises(lambda: sg.logit_leaf_model(ch, "churn", xs, segment_by=["region"], show=False)))
check("benchmark names cannot overwrite built-in models",
      raises(lambda: sg.logit_leaf_model(ch, "churn", xs, benchmarks={"tree": "churn ~ usage"}, show=False)))

r2 = sg.logit_leaf_model(seg, "y", ["x1", "x2", "x3"], max_depth=1, min_leaf=200, segment_by=["x1"], show=False)
check("no segment_by warning when the tree splits on it", not any("No split uses" in w for w in r2["warnings"])
      and r2["split_variables"] == ["x1"])
stab = sg.leaf_model_stability(seg, "y", ["x1", "x2", "x3"], seeds=range(4), max_depth=1, min_leaf=200, show=False)
check("stability: the true segment variable splits in every seed", stab["split_frequency"]["x1"] == 1.0,
      stab["split_frequency"].to_dict())
check("stability: the leaf model is best in every seed for a true two-segment process",
      stab["summary"].loc["logit_leaf", "share_best_AUC"] == 1.0)
stab2 = sg.leaf_model_stability(ch, "churn", xs, seeds=range(3), max_depth=2, min_leaf=150, benchmarks=bm, show=False)
check("stability: the slope-heterogeneity variable is never chosen by the purity-based tree",
      stab2["split_frequency"]["b2b"] == 0.0 and stab2["summary"].loc["interactions", "share_best_AUC"] == 1.0)

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
