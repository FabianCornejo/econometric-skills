"""Tests for v2.4.0: statistics foundations (scripts/inference.py) and the Module 0 teaching demos.

Every number is checked against a second route that can disagree: scipy or statsmodels where the
skill computes the formula itself (Welch, Wilson, Newcombe, Hodges-Lehmann, power, p-value
adjustments), closed-form textbook results (sample size 1068, the horse-kick Poisson fit, the
kidney-stone Simpson's paradox) and simulated coverage for the distribution-free intervals.
Also checked: verdicts never say that H0 is accepted or true, the documents route to the new
module, and the demos show their lessons.

Run from the repository root:  python3 tests/test_v240.py
"""
import contextlib
import io
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "econometrics"
sys.path.insert(0, str(SKILL / "scripts"))
warnings.filterwarnings("ignore")
import inference as inf  # noqa: E402
import simulation as sim  # noqa: E402

ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, info)


def quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*a, **k)


def close(a, b, tol=1e-8):
    return np.allclose(np.asarray(a, float), np.asarray(b, float), rtol=tol, atol=tol)


rng = np.random.default_rng(2024)
x = rng.normal(10, 2, 40)
y = rng.normal(9, 3, 31)

# ---------------------------------------------------------------- describe_dist
s = pd.Series(rng.lognormal(size=300))
r = quiet(inf.describe_dist, s, "lognormal")
sm = r["summary"]
check("describe_dist: skewness equals pandas/Excel SKEW", close(sm["skewness"], s.skew()))
check("describe_dist: excess kurtosis equals pandas .kurt() (Excel KURT)", close(sm["excess_kurtosis"], s.kurt()))
big = rng.normal(size=200_000)
rb = quiet(inf.describe_dist, big)
check("describe_dist: moment kurtosis about 3 for a normal sample", abs(rb["summary"]["kurtosis"] - 3) < 0.05,
      f"{rb['summary']['kurtosis']:.3f}")
q1, q3 = np.quantile(s, [0.25, 0.75])
n_out = int(((s < q1 - 1.5 * (q3 - q1)) | (s > q3 + 1.5 * (q3 - q1))).sum())
check("describe_dist: Tukey outliers counted", sm["outliers"] == n_out and any("Tukey" in w for w in r["warnings"]))

# ---------------------------------------------------------------- one_sample
r = quiet(inf.one_sample, x, 9.5)
ref = stats.ttest_1samp(x, 9.5)
ci = ref.confidence_interval(0.95)
check("one_sample: t, p and CI equal scipy", close([r["statistic"], r["p"], r["ci_low"], r["ci_high"]],
                                                  [ref.statistic, ref.pvalue, ci.low, ci.high]))
r = quiet(inf.one_sample, x, 9.5, alternative="greater")
ref = stats.ttest_1samp(x, 9.5, alternative="greater")
check("one_sample: one-sided p and lower bound equal scipy",
      close([r["p"], r["ci_low"]], [ref.pvalue, ref.confidence_interval(0.95).low]) and np.isinf(r["ci_high"]))
r = quiet(inf.one_sample, x, 9.5, sigma=2.0)
z = (x.mean() - 9.5) / (2 / np.sqrt(x.size))
check("one_sample: z test with known sigma", close(r["p"], 2 * stats.norm.sf(abs(z))) and r["warnings"])
check("one_sample: Cohen's d", close(quiet(inf.one_sample, x, 9.5)["cohen_d"], (x.mean() - 9.5) / x.std(ddof=1)))

# ---------------------------------------------------------------- two_sample
r = quiet(inf.two_sample, x, y)
ref = stats.ttest_ind(x, y, equal_var=False)
ci = ref.confidence_interval(0.95)
check("two_sample: Welch is the default; t, df, p, CI equal scipy",
      "Welch" in r["method"] and close([r["statistic"], r["df"], r["p"], r["ci_low"], r["ci_high"]],
                                       [ref.statistic, ref.df, ref.pvalue, ci.low, ci.high]))
refp = stats.ttest_ind(x, y)
check("two_sample: the pooled p-value is reported alongside", close(r["pooled_t"]["p"], refp.pvalue))
r = quiet(inf.two_sample, x, y, equal_var=True)
check("two_sample: pooled t equals scipy", close([r["statistic"], r["p"]], [refp.statistic, refp.pvalue]))
sp = np.sqrt(((x.size - 1) * x.var(ddof=1) + (y.size - 1) * y.var(ddof=1)) / (x.size + y.size - 2))
check("two_sample: Cohen's d with the pooled SD", close(r["cohen_d"], (x.mean() - y.mean()) / sp))
a, b = rng.normal(0, 1, 20), rng.normal(0, 5, 60)
r = quiet(inf.two_sample, a, b, equal_var=True)
check("two_sample: pooled t with unequal variances and sizes warns", any("Welch" in w for w in r["warnings"]))
pa = rng.normal(50, 10, 25)
pb = pa + rng.normal(1.5, 2, 25)
r = quiet(inf.two_sample, pa, pb, paired=True)
ref = stats.ttest_rel(pa, pb)
check("two_sample paired: equals scipy ttest_rel", close([r["statistic"], r["p"]], [ref.statistic, ref.pvalue]))
check("two_sample paired: pairing gains power here",
      r["p"] < quiet(inf.two_sample, pa, pb)["p"], f"{r['p']:.2g}")

# ---------------------------------------------------------------- variances
r = quiet(inf.variance_test, x, sigma2=4.0)
n = x.size
v = x.var(ddof=1)
lo, hi = (n - 1) * v / stats.chi2.ppf(0.975, n - 1), (n - 1) * v / stats.chi2.ppf(0.025, n - 1)
st = (n - 1) * v / 4
pc = 2 * min(stats.chi2.cdf(st, n - 1), stats.chi2.sf(st, n - 1))
check("variance_test one sample: chi-square CI and two-sided p", close([r["ci_low"], r["ci_high"], r["p"]], [lo, hi, pc]))
r = quiet(inf.variance_test, x, y)
f = x.var(ddof=1) / y.var(ddof=1)
d1, d2 = x.size - 1, y.size - 1
check("variance_test: F, its two-sided p and CI",
      close([r["estimate"], r["p"], r["ci_low"]],
            [f, 2 * min(stats.f.cdf(f, d1, d2), stats.f.sf(f, d1, d2)), f / stats.f.ppf(0.975, d1, d2)]))
check("variance_test: Brown-Forsythe equals scipy levene(center='median')",
      close(r["brown_forsythe_p"], stats.levene(x, y, center="median").pvalue))
r = quiet(inf.variance_test, rng.standard_t(3, 200), rng.standard_t(3, 200))
check("variance_test: heavy tails trigger the normality warning", any("kurtosis" in w for w in r["warnings"]))

# ---------------------------------------------------------------- proportions
from statsmodels.stats.proportion import (confint_proportions_2indep, proportion_confint,  # noqa: E402
                                          proportions_ztest)

good = True
for k, nn in [(3, 25), (0, 20), (20, 20), (47, 120), (1, 500)]:
    t = quiet(inf.proportion_ci, k, nn)["table"]
    for mine, theirs in [("wilson", "wilson"), ("clopper_pearson", "beta"), ("agresti_coull", "agresti_coull"),
                         ("wald", "normal")]:
        ref = proportion_confint(k, nn, alpha=0.05, method=theirs)
        if mine in ("wald", "agresti_coull"):
            ref = (max(0, ref[0]), min(1, ref[1]))
        good &= close(t.loc[mine, ["ci_low", "ci_high"]], ref, 1e-7)
check("proportion_ci: Wilson, Clopper-Pearson, Agresti-Coull and Wald equal statsmodels (incl. 0 and n)", good)
check("proportion_ci: small counts warn about Wald", quiet(inf.proportion_ci, 3, 25)["warnings"])
r = quiet(inf.proportion_test, 30, 100, 0.25)
check("proportion_test one sample: exact p equals binomtest",
      close(r["p_exact"], stats.binomtest(30, 100, 0.25).pvalue))
zs, pz = proportions_ztest(30, 100, 0.25, prop_var=0.25)
check("proportion_test one sample: score z equals statsmodels", close([r["statistic"], r["p_score"]], [zs, pz]))
r = quiet(inf.proportion_test, [81, 234], [87, 270])
ref = confint_proportions_2indep(81, 87, 234, 270, method="newcomb")
zz, pp = proportions_ztest([81, 234], [87, 270])
check("proportion_test two samples: Newcombe CI equals statsmodels", close([r["ci_low"], r["ci_high"]], ref, 1e-7))
check("proportion_test two samples: pooled z equals statsmodels", close([r["statistic"], r["p"]], [zz, pp]))
check("n_for_proportion: 3 points at 95% needs 1068", inf.n_for_proportion(0.03)["n"] == 1068)
check("n_for_proportion: finite population of 10,000 needs 965",
      inf.n_for_proportion(0.03, population=10_000)["n"] == 965)

# ---------------------------------------------------------------- goodness of fit
kicks = np.repeat([0, 1, 2, 3, 4], [109, 65, 22, 3, 1])    # Bortkiewicz, Prussian cavalry deaths
r = quiet(inf.chisq_fit, kicks, "poisson")
lam = kicks.mean()
e0, e1 = 200 * np.exp(-lam), 200 * lam * np.exp(-lam)
e2 = 200 - e0 - e1
manual = (109 - e0) ** 2 / e0 + (65 - e1) ** 2 / e1 + (26 - e2) ** 2 / e2
check("chisq_fit Poisson: pooled cells 0, 1, 2+ and df = 3 - 1 - 1",
      r["k_cells"] == 3 and r["df"] == 1 and close(r["statistic"], manual), f"chi2 = {r['statistic']:.4f}")
die = [16, 25, 19, 21, 8, 31]
r = quiet(inf.chisq_fit, observed=die, expected_probs=[1 / 6] * 6)
ref = stats.chisquare(die)
check("chisq_fit with given probabilities equals scipy chisquare", close([r["statistic"], r["p"]], [ref.statistic, ref.pvalue]))
r = quiet(inf.chisq_fit, rng.exponential(size=500), "normal")
check("chisq_fit: exponential data reject a fitted normal", r["p"] < 1e-4)

# ---------------------------------------------------------------- normality
gauss = np.random.default_rng(7).normal(size=200)
r = quiet(inf.normality, gauss)
check("normality: Shapiro-Wilk equals scipy", close(r["tests"].loc["Shapiro-Wilk", "p"], stats.shapiro(gauss).pvalue))
check("normality: a normal sample is not rejected", r["p"] > 0.05, f"p = {r['p']:.3f}")
check("normality: an exponential sample is rejected by every test",
      (quiet(inf.normality, rng.exponential(size=200))["tests"]["p"] < 0.01).all())
check("normality: QQ data for the plot", len(r["qq"]) == 200 and r["qq_r"] > 0.98)

# ---------------------------------------------------------------- contingency and Simpson
rows = []
for size, trt, succ, tot in [("small", "A", 81, 87), ("large", "A", 192, 263), ("small", "B", 234, 270),
                             ("large", "B", 55, 80)]:                                   # Charig et al. (1986)
    rows += [{"size": size, "trt": trt, "out": "success"}] * succ + [{"size": size, "trt": trt, "out": "failure"}] * (tot - succ)
kid = pd.DataFrame(rows)
r = quiet(inf.contingency, data=kid, row="trt", col="out", strata="size")
t = pd.crosstab(kid.trt, kid.out).to_numpy(float)
e = np.outer(t.sum(1), t.sum(0)) / t.sum()
chi2 = ((t - e) ** 2 / e).sum()
check("contingency: chi-square and Cramer's V by hand", close([r["statistic"], r["cramers_v"]], [chi2, np.sqrt(chi2 / t.sum())]))
check("contingency: Fisher exact equals scipy", close(r["p_fisher"], stats.fisher_exact(t).pvalue))
check("contingency: kidney stones flagged as Simpson's paradox", r["simpson"] and any("Simpson" in w for w in r["warnings"]))
num = den = 0.0
for _, g in kid.groupby("size"):
    ts = pd.crosstab(g.trt, g.out).to_numpy(float)
    num += ts[0, 0] * ts[1, 1] / ts.sum()
    den += ts[0, 1] * ts[1, 0] / ts.sum()
check("contingency: Mantel-Haenszel odds ratio by hand", close(r["mh_odds_ratio"], num / den))
kid2 = kid.copy()
kid2["size"] = rng.permutation(kid2["size"].to_numpy())
check("contingency: no Simpson flag when the strata are unrelated", not quiet(inf.contingency, data=kid2, row="trt",
                                                                                col="out", strata="size")["simpson"])
r = quiet(inf.contingency, [[3, 1], [1, 3]])
check("contingency: small expected counts point to Fisher", any("Fisher" in w for w in r["warnings"]))

# ---------------------------------------------------------------- rank tests
xa, xb = rng.normal(0.4, 1, 30), rng.normal(0, 1, 35)
r = quiet(inf.nonparametric, xa, xb)
mw = stats.mannwhitneyu(xa, xb)
check("nonparametric: Mann-Whitney p equals scipy", close(r["p"], mw.pvalue))
check("nonparametric: Hodges-Lehmann shift is the median of all differences",
      close(r["estimate"], np.median(np.subtract.outer(xa, xb))))
ranks = stats.rankdata(np.concatenate([xa, xb]))
u1 = ranks[:30].sum() - 30 * 31 / 2
check("nonparametric: rank-biserial correlation by hand", close(r["rank_biserial"], 2 * u1 / (30 * 35) - 1))
gen = np.random.default_rng(11)
cover = np.mean([(lambda q: q["ci_low"] <= 0.5 <= q["ci_high"])(
    quiet(inf.nonparametric, gen.standard_t(3, 25) + 0.5, gen.standard_t(3, 30))) for _ in range(300)])
check("nonparametric: the Hodges-Lehmann CI covers the true shift about 95% (t3 data)", 0.91 <= cover <= 0.985,
      f"{cover:.3f}")
one = rng.normal(0.3, 1, 25)
r = quiet(inf.nonparametric, one)
walsh = (one[:, None] + one[None, :])[np.triu_indices(25)]
check("nonparametric one sample: Wilcoxon p equals scipy, pseudo-median of Walsh averages",
      close(r["p"], stats.wilcoxon(one).pvalue) and close(r["estimate"], np.median(walsh)))

# ---------------------------------------------------------------- correlation
cx = rng.normal(size=40)
cy = 0.5 * cx + rng.normal(size=40)
r = quiet(inf.correlation_test, cx, cy)
rr = np.corrcoef(cx, cy)[0, 1]
zc = stats.norm.ppf(0.975) / np.sqrt(37)
check("correlation_test: Pearson r and Fisher-z CI",
      close([r["estimate"], r["ci_low"], r["ci_high"]], [rr, np.tanh(np.arctanh(rr) - zc), np.tanh(np.arctanh(rr) + zc)]))
cy2 = cy.copy()
cy2[0] = 40
check("correlation_test: an outlier makes Pearson and Spearman disagree, with a warning",
      any("Spearman" in w for w in quiet(inf.correlation_test, cx, cy2)["warnings"]))

# ---------------------------------------------------------------- power
from statsmodels.stats.power import TTestIndPower, TTestPower  # noqa: E402

pc = quiet(inf.power_curve, [0.2, 0.5, 0.8], [10, 40], design="two_sample")
ref = [TTestIndPower().power(e, n, 0.05) for n in (10, 40) for e in (0.2, 0.5, 0.8)]
check("power_curve two-sample t equals statsmodels", close(pc["power"], ref, 1e-6))
pc = quiet(inf.power_curve, [0.3, 0.6], 25, design="one_sample", alternative="greater")
ref = [TTestPower().power(e, 25, 0.05, alternative="larger") for e in (0.3, 0.6)]
check("power_curve one-sided one-sample t equals statsmodels", close(pc["power"], ref, 1e-6))
pz = quiet(inf.power_curve, 0.4, 50, test="z", design="one_sample")
nc = 0.4 * np.sqrt(50)
check("power_curve z closed form", close(pz["power"].iloc[0], stats.norm.sf(1.959964 - nc) + stats.norm.cdf(-1.959964 - nc), 1e-5))
check("power_curve: power at a zero effect equals alpha",
      close(quiet(inf.power_curve, 0.0, 30)["power"].iloc[0], 0.05, 1e-9))

# ---------------------------------------------------------------- multiple testing
from statsmodels.stats.multitest import multipletests  # noqa: E402

pv = np.r_[rng.uniform(0, 0.01, 4), rng.uniform(0, 1, 16)]
good = True
for mine, theirs in [("bonferroni", "bonferroni"), ("holm", "holm"), ("bh", "fdr_bh"), ("by", "fdr_by")]:
    a_ = inf.adjust_pvalues(pv, mine)
    rej, padj, _, _ = multipletests(pv, 0.05, theirs)
    good &= close(a_["p_adjusted"], padj) and (a_["reject"].to_numpy() == rej).all()
check("adjust_pvalues: Bonferroni, Holm, BH and BY equal statsmodels", good)

# ---------------------------------------------------------------- verdict wording
verdicts = []
for res in [quiet(inf.one_sample, x, x.mean()), quiet(inf.two_sample, x, x + 0.01), quiet(inf.chisq_fit, kicks, "poisson"),
            quiet(inf.proportion_test, 25, 100, 0.25), quiet(inf.variance_test, x, x * 1.01)]:
    verdicts.append(res["verdict"])
text = " ".join(verdicts).lower()
check("verdicts: non-rejections never accept H0 and say it is not evidence for H0",
      "accept" not in text and all("not evidence that h0 is true" in v.lower() for v in verdicts))
src = (SKILL / "scripts" / "inference.py").read_text(encoding="utf-8")
check("inference.py prints no 'H0 is accepted' anywhere", not re.search(r"accept(s|ed)? h0|h0 (is )?accepted", src, re.I))

# ---------------------------------------------------------------- teaching demos
d = sim.demo_clt(R=3000)
tab = d["table"]
check("demo_clt: SD of the mean equals sigma/sqrt(n) for every n",
      np.allclose(tab["sd_of_means"], tab["sigma/sqrt(n)"], rtol=0.06))
check("demo_clt: skewness and the distance to the normal shrink with n",
      tab["skewness"].iloc[-1] < tab["skewness"].iloc[0] / 3 and tab["ks_distance_to_normal"].is_monotonic_decreasing)
check("demo_clt: t coverage rises toward 95% for skewed data",
      tab["t_interval_coverage"].iloc[1] < tab["t_interval_coverage"].iloc[-1] and abs(tab["t_interval_coverage"].iloc[-1] - 0.95) < 0.02)
d = sim.demo_ci_coverage(R=3000)
cov = d["table"]["coverage"]
check("demo_ci_coverage: t interval for a normal mean covers about 95%",
      all(abs(cov.loc[("normal mean, t interval", n)] - 0.95) < 0.015 for n in (10, 30, 100)))
check("demo_ci_coverage: z with the sample SD undercovers at n = 10",
      cov.loc[("normal mean, z with sample SD", 10)] < 0.93)
check("demo_ci_coverage: Wilson beats Wald for p = 0.1, n = 10",
      cov.loc[("proportion 0.1, Wilson", 10)] - cov.loc[("proportion 0.1, Wald", 10)] > 0.2)
d = sim.demo_estimator_mse(R=20000)
vt, ut = d["variance_table"], d["uniform_table"]
check("demo_estimator_mse: S^2 unbiased, MLE biased with a lower MSE",
      abs(vt.iloc[0]["bias"]) < 0.02 and vt.iloc[1]["bias"] < -0.05 and vt.iloc[1]["mse"] < vt.iloc[0]["mse"])
check("demo_estimator_mse: uniform MLE biased down, (n+1)/n max unbiased, both beat the method of moments",
      ut.iloc[1]["bias"] < -0.3 and abs(ut.iloc[2]["bias"]) < 0.05 and ut.iloc[2]["mse"] < ut.iloc[0]["mse"] / 2)
d = sim.demo_multiple_testing(R=300)
mt = d["table"]
check("demo_multiple_testing: about 64% of null studies find something without correction",
      0.55 < mt.loc[("all nulls true", "raw"), "fwer"] < 0.73)
check("demo_multiple_testing: Holm keeps the family-wise error near 5%",
      mt.loc[("all nulls true", "holm"), "fwer"] < 0.09)
real = "25% real effects"
check("demo_multiple_testing: BH keeps FDR near 5% with more power than Holm",
      mt.loc[(real, "bh"), "fdr"] < 0.08 and mt.loc[(real, "bh"), "power"] > mt.loc[(real, "holm"), "power"])

# ---------------------------------------------------------------- documents route to the module
skill = (SKILL / "SKILL.md").read_text(encoding="utf-8")
found = (SKILL / "references" / "statistics-foundations.md").read_text(encoding="utf-8")
teach = (SKILL / "references" / "teaching.md").read_text(encoding="utf-8")
rules = (SKILL / "references" / "interpretation-rules.md").read_text(encoding="utf-8")
check("SKILL.md routes one- and two-variable questions to statistics-foundations.md and lists inference.py",
      "`statistics-foundations.md`" in skill and "| `inference.py` |" in skill)
check("SKILL.md rule: not rejecting H0 is not evidence for H0", "Not rejecting H0 is not evidence for H0" in skill)
check("statistics-foundations.md has sections 1 to 13",
      all(re.search(rf"^## {i}\. ", found, re.M) for i in range(1, 14)))
check("teaching.md starts the curriculum with Module 0", "Module 0, statistics foundations" in teach)
check("interpretation-rules.md has the v2.4.0 rules", "## Rules added in v2.4.0" in rules)
missing = [n for n in ("one_sample", "two_sample", "variance_test", "proportion_ci", "proportion_test", "chisq_fit",
                       "normality", "contingency", "nonparametric", "correlation_test", "power_curve", "adjust_pvalues",
                       "describe_dist", "n_for_proportion") if n not in found]
check("statistics-foundations.md names every inference function", not missing, str(missing))

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
