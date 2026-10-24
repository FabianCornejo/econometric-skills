"""Known-truth tests for causal, survival, text_data, data_prep and reporting.

Run from the repository root:  python3 tests/test_causal_text_survival.py
Needs: statsmodels, linearmodels, scikit-learn, scipy (rdrobust optional, for a cross-check).
"""
import sys, warnings, tempfile, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "econometrics" / "scripts"))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

rng = np.random.default_rng(11)
ok = []


def check(name, cond, info=""):
    ok.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name, info)


import causal as ca

# ---------- selection on observables: true ATE = 2, confounded ----------
n = 4000
x1, x2 = rng.normal(size=n), rng.normal(size=n)
ps_true = 1 / (1 + np.exp(-(1.2 * x1 - 0.8 * x2)))
T = rng.binomial(1, ps_true)
Y = 2 * T + 2.5 * x1 - 1.5 * x2 + 0.5 * x1 * x2 + rng.normal(size=n)
df = pd.DataFrame({"y": Y, "t": T, "x1": x1, "x2": x2})
naive = df.loc[df.t == 1, "y"].mean() - df.loc[df.t == 0, "y"].mean()
ps = ca.propensity_score(df, "t", ["x1", "x2"])
r_ipw = ca.ipw(df, "y", "t", ps)
r_aipw = ca.aipw(df, "y", "t", ["x1", "x2"])
r_m = ca.nn_match(df, "y", "t", ps)
print(f"naive {naive:.3f} | IPW {r_ipw['estimate']:.3f} | AIPW {r_aipw['estimate']:.3f} | NN ATT {r_m['estimate']:.3f}")
check("naive difference is biased", abs(naive - 2) > 0.5)
check("IPW recovers ATE≈2", abs(r_ipw["estimate"] - 2) < 0.2)
check("AIPW recovers ATE≈2 with CI covering 2", r_aipw["ci_low"] < 2 < r_aipw["ci_high"])
check("NN matching ATT≈2 (homogeneous effect)", abs(r_m["estimate"] - 2) < 0.25)
bt = ca.balance_table(df, "t", ["x1", "x2"])
w = np.where(df.t == 1, 1 / ps, 1 / (1 - ps))
bw = ca.balance_table(df, "t", ["x1", "x2"], weights=w)
check("IPW weights balance covariates (|SMD|<0.1)", bt["SMD"].abs().max() > 0.3 and bw["SMD"].abs().max() < 0.1)
from sklearn.linear_model import LassoCV
r_dml = ca.dml_plr(df.assign(t=df.t.astype(float)), "y", "t", ["x1", "x2"], n_rep=2)
check("DML-PLR theta≈2", abs(r_dml["estimate"] - 2) < 0.2, f"{r_dml['estimate']:.3f}")

# ---------- RDD: jump of 0.8 at 0 ----------
n = 5000
xr = rng.uniform(-1, 1, n)
cov = rng.normal(size=n)
yr = 0.5 + 1.2 * xr - 0.7 * xr ** 2 + 0.8 * (xr >= 0) + 0.3 * cov + rng.normal(0, 0.5, n)
rd = pd.DataFrame({"y": yr, "x": xr, "cov": cov})
r = ca.rdd(rd, "y", "x", 0.0)
check("sharp RD recovers 0.8 (CI covers)", r["ci_low"] < 0.8 < r["ci_high"], f"{r['estimate']:.3f}, h={r['bandwidth']:.3f} {r['bandwidth_method']}")
hik = ca.ik_bandwidth(yr, xr, 0.0)
rik = ca.rdd(rd, "y", "x", 0.0, h=hik, show=False)
check("IK bandwidth gives a sensible estimate too", abs(rik["estimate"] - 0.8) < 0.12, f"{rik['estimate']:.3f}, h_IK={hik:.3f}")
rob = ca.rdd_robustness(rd, "y", "x", 0.0, covs_to_test=["cov"])
check("placebo cutoffs ≈ 0", (rob["placebos"]["p"] > 0.01).all())
check("no density jump without manipulation", rob["density"]["p"] > 0.05)
xm = np.concatenate([xr, rng.uniform(0, 0.05, 800)])            # bunching just above
check("density test detects bunching", ca.density_test(xm, 0)["p"] < 0.01)
# fuzzy: take-up jumps by 0.6, effect of treatment = 1.0
take = (rng.uniform(size=n) < 0.2 + 0.6 * (xr >= 0)).astype(float)
yf = 0.5 + 1.2 * xr + 1.0 * take + rng.normal(0, 0.5, n)
rf = ca.rdd(pd.DataFrame({"y": yf, "x": xr, "d": take}), "y", "x", 0.0, fuzzy="d")
check("fuzzy RD recovers LATE≈1", abs(rf["estimate"] - 1) < 0.3, f"{rf['estimate']:.3f}, first stage {rf['jump_in_treatment']:.2f}")

# ---------- staggered DiD with dynamic effects ----------
units, periods = 300, 10
g = rng.choice([0, 4, 6, 8], units, p=[0.3, 0.25, 0.25, 0.2])
rows = []
ui = rng.normal(size=units)
tf = rng.normal(size=periods) * 0.5
for i in range(units):
    for t in range(1, periods + 1):
        e = t - g[i] if g[i] > 0 else -99
        eff = (1 + 0.5 * e) if (g[i] > 0 and e >= 0) else 0.0      # effect grows with exposure
        rows.append({"id": i, "t": t, "g": g[i] if g[i] > 0 else np.nan,
                     "y": ui[i] + tf[t - 1] + eff + rng.normal(0, 0.5)})
pan = pd.DataFrame(rows)
cs = ca.att_gt(pan.assign(g0=pan.g.fillna(0)), "id", "t", "y", "g0", n_boot=99)
dyn = cs["dynamic"]
check("CS dynamic ATT(e=0)≈1, (e=2)≈2", abs(dyn.loc[0, "att"] - 1) < 0.2 and abs(dyn.loc[2, "att"] - 2) < 0.25,
      f"{dyn.loc[0,'att']:.2f}, {dyn.loc[2,'att']:.2f}")
check("CS pre-period ATTs ≈ 0", dyn.loc[dyn.index < 0, "att"].abs().max() < 0.2)
es = ca.event_study(pan, "id", "t", "y", "g", leads=3, lags=4)
check("event study: no pre-trends flagged", es["pretrend_p"] > 0.05)
check("event study k=0 ≈ 1", abs(es["coefs"].loc[0, "coef"] - 1) < 0.25)

# ---------- synthetic control: treated unit = 0.6 A + 0.4 B, effect +3 after t0 ----------
T_, J = 40, 15
F = np.cumsum(rng.normal(size=(T_, 3)), axis=0)
load = rng.uniform(0, 1, (J, 3))
Yd = F @ load.T + rng.normal(0, 0.3, (T_, J))
treated = 0.6 * Yd[:, 0] + 0.4 * Yd[:, 1] + rng.normal(0, 0.1, T_)
treated[30:] += 3
long = pd.DataFrame([{"unit": f"u{j}", "t": t, "y": Yd[t, j]} for t in range(T_) for j in range(J)] +
                    [{"unit": "TR", "t": t, "y": treated[t]} for t in range(T_)])
sc = ca.synth(long, "unit", "t", "y", "TR", 30)
check("synthetic control weights ≈ 0.6/0.4 on u0/u1", abs(sc["weights"].get("u0", 0) - 0.6) < 0.15 and abs(sc["weights"].get("u1", 0) - 0.4) < 0.15)
check("synthetic control post gap ≈ 3, placebo p minimal", abs(sc["avg_post_gap"] - 3) < 0.4 and sc["p_value"] <= 1 / 16 + 1e-9)

# ---------- survival: true log-HR 0.5 for x, censoring ----------
import survival as sv
n = 3000
xs = rng.binomial(1, 0.5, n)
z = rng.normal(size=n)
lam = 0.1 * np.exp(0.5 * xs + 0.3 * z)
tt = rng.exponential(1 / lam)
cens = rng.exponential(15, n)
d = pd.DataFrame({"time": np.minimum(tt, cens), "event": (tt <= cens).astype(int), "x": xs, "z": z,
                  "grp": rng.integers(0, 50, n)})
res, hr = sv.cox(d, "x + z", "time", "event", cluster="grp")
check("Cox recovers HR exp(0.5)=1.65", abs(hr.loc["x", "HR"] - np.exp(0.5)) < 0.15, f"{hr.loc['x','HR']:.3f}")
km = sv.kaplan_meier(d, "time", "event", group="x", times=[5, 10])
check("KM: x=1 survives less", km["summary"].loc[1, "median_survival"] < km["summary"].loc[0, "median_survival"])
check("log-rank rejects", sv.logrank(d, "time", "event", "x")["p"] < 0.001)
ph = sv.ph_test(res, d, "time", "event")
check("PH not rejected under true PH", (ph["p"] > 0.01).all())
# non-PH: effect reverses over time
xb = rng.binomial(1, 0.5, n)
t1 = np.where(xb == 1, rng.weibull(0.6, n) * 8, rng.weibull(2.0, n) * 8)
dn = pd.DataFrame({"time": t1, "event": 1, "x": xb})
r2, _ = sv.cox(dn, "x", "time", "event", show=False)
check("PH test detects crossing hazards", sv.ph_test(r2, dn, "time", "event")["p"].iloc[0] < 0.001)
pp = sv.person_period(d.head(5).assign(id=range(5)), "id", "time", "event")
check("person-period expansion: one event row max per id", pp.groupby("id")["y"].sum().max() <= 1)

# ---------- text ----------
import text_data as tx
pos = ["o requerente pede medicamento de alto custo e o pedido foi deferido pelo juízo",
       "julgo procedente o pedido para fornecimento do medicamento ao autor", "pedido deferido, fornecimento de remédio"]
neg = ["julgo improcedente o pedido, indeferido o fornecimento", "pedido indeferido por falta de provas do autor",
       "extingo o processo sem resolução do mérito, indeferido"]
vocab_pos = "medicamento fornecimento deferido procedente tutela urgencia saude remedio tratamento".split()
vocab_neg = "improcedente indeferido extincao prescricao ilegitimidade custas honorarios recurso merito".split()
common = "autor reu juizo processo decisao sentenca estado".split()
texts = [rng.choice(pos if i % 2 == 0 else neg) + " " + " ".join(rng.choice(vocab_pos if i % 2 == 0 else vocab_neg, 6))
         + " " + " ".join(rng.choice(common, 4)) for i in range(400)]
labels = [1 if i % 2 == 0 else 0 for i in range(400)]
fl = tx.regex_flags(pd.Series(texts + [None]), {"granted": r"\bdeferid[oa]\b|\bprocedente\b(?<!improcedente)",
                                               "denied": r"indeferid[oa]|improcedente"})
check("regex flags: NaN for missing text, snippet kept", np.isnan(fl["granted"].iloc[-1]) and fl["granted_snippet"].notna().sum() > 0)
ex = tx.regex_extract(pd.Series(["Requerente: JOÃO DA SILVA  Requerido: ESTADO DE SP Juiz(a): X"]),
                      {"plaintiff": r"requerente:\s*(.*?)\s*requerido:", "defendant": r"requerido:\s*(.*?)\s*juiz"})
check("regex_extract parties", ex.iloc[0].tolist() == ["JOÃO DA SILVA", "ESTADO DE SP"])
clf = tx.train_classifier(texts, labels)
check("classifier learns separable classes", clf["test_f1_macro"] > 0.95)
X, vec = tx.tfidf(texts)
cl = tx.cluster_texts(texts, k=2, X=X, vectorizer=vec)
from sklearn.metrics import adjusted_rand_score
check("text clusters recover the two groups", adjusted_rand_score(labels, cl["labels"]) > 0.9)
v = tx.validate_coding(fl["denied"].iloc[:-1].astype(int), [1 - l for l in labels])
check("validate_coding perfect on clean rule", v["kappa"] > 0.99)
check("Rogan-Gladen corrects prevalence", abs(tx.rogan_gladen(0.38, 0.9, 0.95) - (0.38 + 0.95 - 1) / (0.85)) < 1e-9)
sims = tx.similarity_to_reference(X, np.array(labels) == 1)
check("similarity higher for in-group docs", sims[np.array(labels) == 1].mean() > sims[np.array(labels) == 0].mean())
top, theta, _ = tx.lda_topics(texts, k=2)
check("LDA returns topic shares summing to 1", np.allclose(theta.sum(axis=1), 1))

# ---------- data_prep ----------
import data_prep as dp
tmp = tempfile.mkdtemp()
pth = os.path.join(tmp, "x.csv")
pd.DataFrame({"NM": ["José", "Ana"], "VALOR": ["1.234,56", "10,00"]}).to_csv(pth, sep=";", index=False, encoding="latin-1")
r = dp.read_any(pth)
check("read_any sniffs ';' and latin-1", list(r.columns) == ["NM", "VALOR"] and r.NM[0] == "José")
check("br_number parses '1.234,56'", dp.br_number(r.VALOR).tolist() == [1234.56, 10.0])
check("fix_mojibake", dp.fix_mojibake("setor_censitÃ¡rio") == "setor_censitário")
xml = os.path.join(tmp, "o.xls")
open(xml, "w").write('<?xml version="1.0"?><Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet"><Worksheet><Table>'
                     '<Row><Cell><Data ss:Type="String">VARIABLES</Data></Cell><Cell><Data>(1)</Data></Cell></Row>'
                     '<Row><Cell><Data>x</Data></Cell><Cell><Data>0.5***</Data></Cell></Row></Table></Worksheet></Workbook>')
check("SpreadsheetML .xls read", dp.read_any(xml).iloc[1, 1] == "0.5***")
L = pd.DataFrame({"k": [1, 2, 3], "a": 1}); R = pd.DataFrame({"k": [1, 1, 2], "b": 2})
try:
    dp.safe_merge(L, R, "k", validate="m:1"); check("safe_merge blocks many-to-many", False)
except Exception:
    check("safe_merge blocks a non-unique right key", True)
big = os.path.join(tmp, "big.csv")
pd.DataFrame({"cargo": ["Senador", "Deputado"] * 5000, "v": range(10000)}).to_csv(big, index=False)
check("read_csv_filtered", len(dp.read_csv_filtered(big, keep=lambda d: d.cargo.eq("Senador"), chunksize=1000)) == 5000)

# ---------- reporting ----------
import reporting as rp
import statsmodels.formula.api as smf
m1 = smf.ols("y ~ t", df).fit(cov_type="HC1"); m2 = smf.ols("y ~ t + x1 + x2", df).fit(cov_type="HC1")
tab = rp.regression_table({"(1)": m1, "(2)": m2}, keep=["t", "x1", "x2"], labels={"t": "Treatment"},
                          fe_rows={"Controls": ["NO", "YES"]})
check("regression table layout", tab.loc["Treatment"].iloc[1].endswith("***") and "Observations" in tab.index)
lg = smf.logit("t ~ x1 + x2", df).fit(disp=False)
tm = rp.regression_table({"margins": lg.get_margeff()})
check("table accepts margins objects", "x1" in tm.index)
th = rp.regression_table({"cox": res}, exponentiate=True)
check("table with hazard ratios", th.loc["x"].iloc[0].startswith("1.6"))
rp.export_table(tab, os.path.join(tmp, "t.md"))

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
