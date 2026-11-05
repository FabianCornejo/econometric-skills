"""
Statistical inference building blocks: describe a variable, test means, variances and proportions,
check a distribution, test association in tables, rank tests, correlations, power and multiple tests.
The foundation under the regression modules, and the code behind Teacher mode's Module 0
(references/statistics-foundations.md).

Describe
    describe_dist(x)                          n, mean, SD, quantiles, skewness, kurtosis (and Excel's excess
                                              kurtosis), Tukey outliers, Jarque-Bera
Means
    one_sample(x, mu0, sigma=None)            t test (z if sigma is known), CI, Cohen's d / Hedges' g
    two_sample(x, y, paired=False)            Welch t by default, pooled t on request, paired t;
                                              difference, CI, effect size, Brown-Forsythe check of variances
Variances
    variance_test(x, y=None, sigma2=None)     chi-square CI and test for one variance; F test and CI for a
                                              ratio of two, with the robust Brown-Forsythe and Bartlett tests
Proportions
    proportion_ci(count, nobs)                Wilson (default), Clopper-Pearson, Agresti-Coull, Wald, conservative
    proportion_test(count, nobs, value)       one proportion (exact binomial and score test) or two
                                              (pooled z, Newcombe CI for the difference, Fisher for small counts)
    n_for_proportion(margin, p=None)          sample size for a margin of error (with finite-population correction)
Distributions
    chisq_fit(...)                            chi-square goodness of fit with df = k - a - 1 and pooling of small cells
    normality(x)                              Shapiro-Wilk, Lilliefors, Jarque-Bera, D'Agostino, QQ-plot data
Tables
    contingency(table | data, row, col, strata=None)
                                              chi-square independence, Cramer's V, adjusted residuals, Fisher (2x2),
                                              odds ratio; with strata: CMH, Breslow-Day and Simpson's-paradox check
Ranks and correlation
    nonparametric(x, y=None, paired=False)    Wilcoxon signed-rank / Mann-Whitney with Hodges-Lehmann estimate,
                                              its CI and the rank-biserial correlation
    correlation_test(x, y, method)            Pearson, Spearman or Kendall with a Fisher-z CI
Planning and many tests
    power_curve(effect, n)                    power and beta of z/t tests (one-sample, paired, two-sample)
    n_for_power(effect, sd, power=0.8)        smallest n reaching the target power for a given true difference
    adjust_pvalues(p, method)                 Bonferroni, Holm, Benjamini-Hochberg, Benjamini-Yekutieli

Conventions (SKILL.md section 5): a test returns a dict with estimand, estimate, se, ci_low, ci_high, p, n,
method, warnings and a plain-language verdict. A test that does not reject never says that H0 is true:
the verdict says the data are compatible with it and points to the interval (and to power_curve).
Every function prints a short summary unless show=False.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

import _console  # noqa: F401  (Windows-safe console output, see _console.py)

ALPHA = 0.05
_ALTS = ("two-sided", "less", "greater")


# =============================================================================
# Helpers
# =============================================================================
def _clean(x, name="x"):
    s = pd.Series(np.asarray(x, dtype=float).ravel())
    n_missing = int(s.isna().sum())
    a = s.dropna().to_numpy()
    if a.size == 0:
        raise ValueError(f"{name} has no non-missing values")
    return a, n_missing


def _check_alt(alternative):
    if alternative not in _ALTS:
        raise ValueError(f"alternative must be one of {_ALTS}")


def _p_value(stat, dist, alternative):
    """p-value of `stat` under a scipy distribution, for the given alternative."""
    if alternative == "two-sided":
        if getattr(dist, "dist", None) is not None and dist.dist.name in ("chi2", "f"):
            return float(np.minimum(1.0, 2 * np.minimum(dist.cdf(stat), dist.sf(stat))))
        return float(np.minimum(1.0, 2 * dist.sf(abs(stat))))
    return float(dist.sf(stat) if alternative == "greater" else dist.cdf(stat))


def _ci_symmetric(est, se, crit_two, crit_one, alternative):
    if alternative == "two-sided":
        return est - crit_two * se, est + crit_two * se
    if alternative == "greater":
        return est - crit_one * se, np.inf
    return -np.inf, est + crit_one * se


def _verdict(p, alpha, h0):
    if not np.isfinite(p):
        return "no test computed"
    if p < alpha:
        return f"rejects H0 ({h0}) at the {alpha:.0%} level (p = {p:.3g})"
    return (f"does not reject H0 ({h0}) at the {alpha:.0%} level (p = {p:.3g}): the data are compatible "
            "with H0, which is not evidence that H0 is true; report the interval and the power")


def _fmt(v, d=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "inf" if (isinstance(v, float) and np.isinf(v) and v > 0) else (
            "-inf" if (isinstance(v, float) and np.isinf(v)) else "nan")
    return f"{v:.{d}g}"


def _print(res, extra=()):
    lines = [f"\n{res['method']}"]
    if res.get("estimand") is not None and np.isfinite(res.get("estimate", np.nan)):
        ci = ""
        if np.isfinite(res.get("ci_low", np.nan)) or np.isfinite(res.get("ci_high", np.nan)):
            ci = f", {1 - res.get('alpha', ALPHA):.0%} CI [{_fmt(res['ci_low'])}, {_fmt(res['ci_high'])}]"
        lines.append(f"  {res['estimand']}: {_fmt(res['estimate'])}{ci}  (n = {res['n']})")
    lines.extend(f"  {e}" for e in extra)
    if res.get("verdict"):
        lines.append(f"  Verdict: {res['verdict']}")
    for w in res.get("warnings", []):
        lines.append(f"  WARNING: {w}")
    print("\n".join(lines))


def _result(estimand, estimate, se, ci, p, n, method, alpha, h0=None, **extra):
    res = {"estimand": estimand, "estimate": float(estimate), "se": float(se),
           "ci_low": float(ci[0]), "ci_high": float(ci[1]), "p": float(p), "n": n,
           "method": method, "alpha": alpha, "warnings": []}
    if h0 is not None:
        res["h0"] = h0
        res["verdict"] = _verdict(p, alpha, h0)
    res.update(extra)
    return res


# =============================================================================
# Describe
# =============================================================================
def describe_dist(x, name: str = "x", alpha: float = ALPHA, show: bool = True) -> dict:
    """Summary of one variable before any model: location, spread, shape and outliers.

    kurtosis is the moment ratio b2 = m4 / m2^2 (3 for a normal). excess_kurtosis is the bias-adjusted
    G2 (0 for a normal), which is what Excel's KURT and pandas' .kurt() return: a value near 2 there is
    already heavy tails, not "lighter than normal". skewness is the adjusted G1 (Excel's SKEW).
    Outliers use Tukey's fences, Q1 - 1.5 IQR and Q3 + 1.5 IQR (3 IQR for extreme)."""
    a, n_missing = _clean(x, name)
    n = a.size
    if n < 2:
        raise ValueError("describe_dist needs at least 2 values")
    mean, sd = a.mean(), a.std(ddof=1)
    q1, med, q3 = np.quantile(a, [0.25, 0.5, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    out_mask = (a < lo) | (a > hi)
    extreme = (a < q1 - 3 * iqr) | (a > q3 + 3 * iqr)
    skew = float(stats.skew(a, bias=False)) if n > 2 else np.nan
    b2 = float(stats.kurtosis(a, fisher=False, bias=True)) if n > 3 else np.nan
    g2 = float(stats.kurtosis(a, fisher=True, bias=False)) if n > 3 else np.nan
    jb = stats.jarque_bera(a)
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf(1 - alpha / 2, n - 1)
    summary = pd.Series({"n": n, "missing": n_missing, "mean": mean, "sd": sd, "se_mean": se,
                         "cv": sd / mean if mean != 0 else np.nan, "min": a.min(), "q1": q1, "median": med,
                         "q3": q3, "max": a.max(), "iqr": iqr, "skewness": skew, "kurtosis": b2,
                         "excess_kurtosis": g2, "jb_stat": float(jb.statistic), "jb_p": float(jb.pvalue),
                         "outliers": int(out_mask.sum()), "extreme_outliers": int(extreme.sum())}, name=name)
    res = _result("mean", mean, se, (mean - tcrit * se, mean + tcrit * se), np.nan, n,
                  f"Description of {name}", alpha, summary=summary, outlier_values=a[out_mask],
                  fences=(lo, hi))
    w = res["warnings"]
    if n_missing:
        w.append(f"{n_missing} missing value(s) dropped; check whether they are missing at random "
                 "(data_prep.missing_report).")
    if out_mask.any():
        w.append(f"{int(out_mask.sum())} value(s) outside Tukey's fences [{lo:.4g}, {hi:.4g}] "
                 f"({int(extreme.sum())} beyond 3 IQR): check them before modelling; never drop them silently.")
    if np.isfinite(skew) and abs(skew) > 1:
        w.append(f"skewness {skew:.2f}: the mean and t intervals are sensitive to the tail; report the median "
                 "too and consider a log scale or a rank test.")
    if np.isfinite(g2) and g2 > 3:
        w.append(f"excess kurtosis {g2:.2f} (normal = 0): heavy tails; variance tests and normal-based "
                 "intervals for single values (VaR, prediction intervals) are unreliable.")
    if show:
        print(f"\nDescription of {name} (n = {n}, missing = {n_missing})")
        print(summary.drop(["n", "missing"]).to_string(float_format=lambda v: f"{v:.4g}"))
        print("  kurtosis: moment ratio, normal = 3; excess_kurtosis: normal = 0 (Excel KURT, pandas .kurt())")
        for e in w:
            print(f"  WARNING: {e}")
    return res


# =============================================================================
# Means
# =============================================================================
def _cohen_d_ci(d, n_eff, alpha):
    se = np.sqrt(1 / n_eff + d ** 2 / (2 * n_eff))
    z = stats.norm.ppf(1 - alpha / 2)
    return se, (d - z * se, d + z * se)


def one_sample(x, mu0: float = 0.0, sigma: float | None = None, alternative: str = "two-sided",
               alpha: float = ALPHA, show: bool = True) -> dict:
    """Test H0: mean = mu0 and give a CI for the mean.

    t test with the sample SD (df = n - 1); z test only when the population sigma is truly known.
    The CI matches the alternative: two-sided, or a one-sided bound for 'greater'/'less'.
    Effect size: Cohen's d = (mean - mu0) / s and Hedges' g (small-sample corrected), with an approximate
    two-sided CI (whatever the alternative)."""
    _check_alt(alternative)
    a, n_missing = _clean(x)
    n = a.size
    if n < 2:
        raise ValueError("one_sample needs at least 2 values")
    m, s = a.mean(), a.std(ddof=1)
    if s == 0 and sigma is None:
        raise ValueError("all values are equal: no variation, so no t test or interval")
    if sigma is not None:
        se = sigma / np.sqrt(n)
        dist = stats.norm()
        crit2, crit1 = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(1 - alpha)
        method, df = "One-sample z test (sigma known)", np.inf
    else:
        se = s / np.sqrt(n)
        df = n - 1
        dist = stats.t(df)
        crit2, crit1 = stats.t.ppf(1 - alpha / 2, df), stats.t.ppf(1 - alpha, df)
        method = "One-sample t test"
    stat = (m - mu0) / se
    p = _p_value(stat, dist, alternative)
    ci = _ci_symmetric(m, se, crit2, crit1, alternative)
    d = (m - mu0) / s
    g = d * (1 - 3 / (4 * (n - 1) - 1))
    _, d_ci = _cohen_d_ci(d, n, alpha)
    sym = {"two-sided": "=", "greater": "<=", "less": ">="}[alternative]
    res = _result("mean", m, se, ci, p, n, method, alpha, h0=f"mean {sym} {mu0:g}",
                  statistic=float(stat), df=float(df), mu0=mu0, sd=float(s), cohen_d=float(d),
                  hedges_g=float(g), cohen_d_ci=d_ci, alternative=alternative)
    w = res["warnings"]
    if n_missing:
        w.append(f"{n_missing} missing value(s) dropped.")
    sk = stats.skew(a, bias=False) if n > 2 else 0.0
    if n < 30 and abs(sk) > 1:
        w.append(f"small sample (n = {n}) with skewness {sk:.2f}: the t interval can miss; compare with "
                 "nonparametric() or a bootstrap (estimation.bootstrap).")
    if sigma is not None:
        w.append("the z test assumes the population sigma is known; with sigma estimated from the data use the "
                 "t test (sigma=None).")
    if show:
        _print(res, [f"H0: mean {sym} {mu0:g}; statistic = {stat:.4g}" + (f", df = {df:g}" if np.isfinite(df) else "")
                     + f"; Cohen's d = {d:.3g} (Hedges' g = {g:.3g})"])
    return res


def two_sample(x, y, paired: bool = False, equal_var: bool = False, alternative: str = "two-sided",
               alpha: float = ALPHA, labels: tuple = ("x", "y"), show: bool = True) -> dict:
    """Difference of two means: mean(x) - mean(y).

    Independent samples: Welch's t by default (no equal-variance assumption, Satterthwaite df); the pooled
    t only with equal_var=True, and then a Brown-Forsythe test warns when the variances differ. The other
    version's p-value is returned too ('pooled_t' or 'welch_t') because textbooks and Excel show both.
    Paired samples (paired=True): t test on the differences, which removes the between-unit variation.
    Effect size: Cohen's d with the pooled SD (d_z, the mean difference over the SD of differences, when
    paired) and Hedges' g."""
    _check_alt(alternative)
    sym = {"two-sided": "=", "greater": "<=", "less": ">="}[alternative]
    h0 = f"mean({labels[0]}) - mean({labels[1]}) {sym} 0"
    if paired:
        xs, ys = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
        if xs.size != ys.size:
            raise ValueError("paired samples must have the same length")
        keep = ~(np.isnan(xs) | np.isnan(ys))
        dlt = xs[keep] - ys[keep]
        n = dlt.size
        if n < 2:
            raise ValueError("need at least 2 complete pairs")
        m, s = dlt.mean(), dlt.std(ddof=1)
        if s == 0:
            raise ValueError("all paired differences are equal: no variation, so no t test or interval")
        se, df = s / np.sqrt(n), n - 1
        stat = m / se
        p = _p_value(stat, stats.t(df), alternative)
        ci = _ci_symmetric(m, se, stats.t.ppf(1 - alpha / 2, df), stats.t.ppf(1 - alpha, df), alternative)
        dz = m / s
        r = float(np.corrcoef(xs[keep], ys[keep])[0, 1]) if n > 2 else np.nan
        res = _result(f"mean difference ({labels[0]} - {labels[1]}, paired)", m, se, ci, p, n, "Paired t test",
                      alpha, h0=h0, statistic=float(stat), df=float(df), cohen_dz=float(dz),
                      hedges_g=float(dz * (1 - 3 / (4 * (n - 1) - 1))), pair_correlation=r,
                      alternative=alternative)
        if (~keep).any():
            res["warnings"].append(f"{int((~keep).sum())} incomplete pair(s) dropped.")
        if show:
            _print(res, [f"t = {stat:.4g}, df = {df}; d_z = {dz:.3g}; correlation within pairs = {r:.3g} "
                         "(the higher it is, the more pairing gains over two independent samples)"])
        return res

    a, ma = _clean(x, labels[0])
    b, mb = _clean(y, labels[1])
    n1, n2 = a.size, b.size
    if n1 < 2 or n2 < 2:
        raise ValueError("each sample needs at least 2 values")
    m1, m2 = a.mean(), b.mean()
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    diff = m1 - m2
    if v1 == 0 or v2 == 0:
        raise ValueError("a sample has no variation: the t tests are not defined; describe the groups instead")
    se_w = np.sqrt(v1 / n1 + v2 / n2)
    df_w = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
    se_p, df_p = np.sqrt(sp2 * (1 / n1 + 1 / n2)), n1 + n2 - 2
    t_w, t_p = diff / se_w, diff / se_p
    p_w, p_p = _p_value(t_w, stats.t(df_w), alternative), _p_value(t_p, stats.t(df_p), alternative)
    if equal_var:
        se, df, stat, p, method = se_p, df_p, t_p, p_p, "Two-sample t test (pooled variance)"
        other = {"welch_t": {"t": float(t_w), "df": float(df_w), "p": p_w}}
    else:
        se, df, stat, p, method = se_w, df_w, t_w, p_w, "Two-sample Welch t test (unequal variances)"
        other = {"pooled_t": {"t": float(t_p), "df": float(df_p), "p": p_p}}
    ci = _ci_symmetric(diff, se, stats.t.ppf(1 - alpha / 2, df), stats.t.ppf(1 - alpha, df), alternative)
    d = diff / np.sqrt(sp2)
    g = d * (1 - 3 / (4 * (n1 + n2) - 9))
    bf = stats.levene(a, b, center="median")
    res = _result(f"mean difference ({labels[0]} - {labels[1]})", diff, se, ci, p, n1 + n2, method, alpha,
                  h0=h0, statistic=float(stat), df=float(df), n1=n1, n2=n2, means=(float(m1), float(m2)),
                  sds=(float(np.sqrt(v1)), float(np.sqrt(v2))), cohen_d=float(d), hedges_g=float(g),
                  brown_forsythe_p=float(bf.pvalue), alternative=alternative, **other)
    w = res["warnings"]
    if ma or mb:
        w.append(f"{ma + mb} missing value(s) dropped.")
    if equal_var and bf.pvalue < alpha:
        w.append(f"the variances differ (Brown-Forsythe p = {bf.pvalue:.3g})"
                 + (" and the group sizes differ" if n1 != n2 else "")
                 + ": the pooled t test is not valid; use Welch (equal_var=False, the default).")
    if show:
        _print(res, [f"means {m1:.4g} vs {m2:.4g}; SDs {np.sqrt(v1):.4g} vs {np.sqrt(v2):.4g}; n {n1} vs {n2}",
                     f"t = {stat:.4g}, df = {df:.4g}; Cohen's d = {d:.3g} (Hedges' g = {g:.3g}); "
                     f"Brown-Forsythe test of equal variances p = {bf.pvalue:.3g}"])
    return res


# =============================================================================
# Variances
# =============================================================================
def variance_test(x, y=None, sigma2: float | None = None, groups: list | None = None,
                  alternative: str = "two-sided", alpha: float = ALPHA, show: bool = True) -> dict:
    """Inference on variances.

    One sample: CI for sigma^2 (and sigma) from (n-1) s^2 / sigma^2 ~ chi2(n-1); with sigma2, tests H0: sigma^2 = sigma2.
    Two samples: F = s_x^2 / s_y^2 ~ F(n_x-1, n_y-1), with a CI for the ratio, plus the Brown-Forsythe
    (Levene, median-centred) and Bartlett tests. groups=[a, b, c, ...]: k-sample Brown-Forsythe and Bartlett.
    The chi-square and F procedures (and Bartlett) assume normal data and fail badly with heavy tails;
    Brown-Forsythe does not, so it is the default check before choosing a pooled t."""
    _check_alt(alternative)
    if groups is not None:
        gs = [_clean(g, f"group {i}")[0] for i, g in enumerate(groups)]
        bf, bt = stats.levene(*gs, center="median"), stats.bartlett(*gs)
        res = _result("variances equal across groups", np.nan, np.nan, (np.nan, np.nan), float(bf.pvalue),
                      sum(g.size for g in gs), f"Equality of {len(gs)} variances (Brown-Forsythe)", alpha,
                      h0="all group variances are equal", brown_forsythe=float(bf.statistic),
                      brown_forsythe_p=float(bf.pvalue), bartlett=float(bt.statistic), bartlett_p=float(bt.pvalue),
                      variances=[float(g.var(ddof=1)) for g in gs])
        if show:
            _print(res, [f"Brown-Forsythe W = {bf.statistic:.4g} (p = {bf.pvalue:.3g}); Bartlett = {bt.statistic:.4g} "
                         f"(p = {bt.pvalue:.3g}, assumes normality)"])
        return res

    a, _ = _clean(x)
    n1 = a.size
    v1 = a.var(ddof=1)
    if v1 == 0:
        raise ValueError("x has no variation")
    if y is None:
        df = n1 - 1
        lo = df * v1 / stats.chi2.ppf(1 - alpha / 2, df)
        hi = df * v1 / stats.chi2.ppf(alpha / 2, df)
        if alternative == "greater":
            lo, hi = df * v1 / stats.chi2.ppf(1 - alpha, df), np.inf
        elif alternative == "less":
            lo, hi = 0.0, df * v1 / stats.chi2.ppf(alpha, df)
        if sigma2 is not None:
            stat = df * v1 / sigma2
            p = _p_value(stat, stats.chi2(df), alternative)
            sym = {"two-sided": "=", "greater": "<=", "less": ">="}[alternative]
            h0 = f"variance {sym} {sigma2:g}"
        else:
            stat, p, h0 = np.nan, np.nan, None
        res = _result("variance", v1, v1 * np.sqrt(2 / df), (lo, hi), p, n1,
                      "Chi-square inference on one variance", alpha, h0=h0, statistic=float(stat), df=df,
                      sd=float(np.sqrt(v1)), sd_ci=(float(np.sqrt(lo)), float(np.sqrt(hi))))
        extra = [f"SD = {np.sqrt(v1):.4g}, CI [{np.sqrt(lo):.4g}, {np.sqrt(hi):.4g}]"]
        if sigma2 is not None:
            extra.append(f"chi2 = {stat:.4g}, df = {df}")
    else:
        b, _ = _clean(y)
        n2 = b.size
        v2 = b.var(ddof=1)
        if v2 == 0:
            raise ValueError("y has no variation")
        d1, d2 = n1 - 1, n2 - 1
        f = v1 / v2
        p = _p_value(f, stats.f(d1, d2), alternative)
        lo, hi = f / stats.f.ppf(1 - alpha / 2, d1, d2), f / stats.f.ppf(alpha / 2, d1, d2)
        if alternative == "greater":
            lo, hi = f / stats.f.ppf(1 - alpha, d1, d2), np.inf
        elif alternative == "less":
            lo, hi = 0.0, f / stats.f.ppf(alpha, d1, d2)
        bf, bt = stats.levene(a, b, center="median"), stats.bartlett(a, b)
        sym = {"two-sided": "=", "greater": "<=", "less": ">="}[alternative]
        res = _result("variance ratio var(x)/var(y)", f, np.nan, (lo, hi), p, n1 + n2,
                      "F test of two variances", alpha, h0=f"var(x) / var(y) {sym} 1", statistic=float(f),
                      df=(d1, d2), brown_forsythe_p=float(bf.pvalue), bartlett_p=float(bt.pvalue),
                      variances=(float(v1), float(v2)))
        extra = [f"F = {f:.4g} with ({d1}, {d2}) df; robust Brown-Forsythe p = {bf.pvalue:.3g}; "
                 f"Bartlett p = {bt.pvalue:.3g}"]
        a = np.concatenate([a - a.mean(), b - b.mean()])
    g2 = stats.kurtosis(a, fisher=True, bias=False) if a.size > 3 else 0.0
    if abs(g2) > 1:
        res["warnings"].append(f"excess kurtosis {g2:.2f}: chi-square and F inference on variances assume normal "
                               "data and their true level can be far from the nominal one; rely on Brown-Forsythe "
                               "(two or more samples) or a bootstrap CI.")
    if show:
        _print(res, extra)
    return res


# =============================================================================
# Proportions
# =============================================================================
def _wilson(x, n, z):
    p = x / n
    den = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return centre - half, centre + half


def _prop_intervals(x, n, alpha):
    z = stats.norm.ppf(1 - alpha / 2)
    p = x / n
    rows = {}
    h = z * np.sqrt(p * (1 - p) / n)
    rows["wald"] = (max(0.0, p - h), min(1.0, p + h))
    rows["wilson"] = _wilson(x, n, z)
    nt = n + z ** 2
    pt = (x + z ** 2 / 2) / nt
    h = z * np.sqrt(pt * (1 - pt) / nt)
    rows["agresti_coull"] = (max(0.0, pt - h), min(1.0, pt + h))
    lo = 0.0 if x == 0 else stats.beta.ppf(alpha / 2, x, n - x + 1)
    hi = 1.0 if x == n else stats.beta.ppf(1 - alpha / 2, x + 1, n - x)
    rows["clopper_pearson"] = (float(lo), float(hi))
    h = z / (2 * np.sqrt(n))
    rows["conservative"] = (max(0.0, p - h), min(1.0, p + h))
    tab = pd.DataFrame(rows, index=["ci_low", "ci_high"]).T
    tab["width"] = tab["ci_high"] - tab["ci_low"]
    return tab


def proportion_ci(count: int, nobs: int, method: str = "wilson", alpha: float = ALPHA, show: bool = True) -> dict:
    """Confidence interval for a proportion, with every common method side by side.

    wilson (default): inverts the score test, good coverage even for small n or p near 0 or 1.
    clopper_pearson: exact, conservative (coverage at least 1 - alpha). agresti_coull: 'add z^2/2 successes
    and failures'. wald: p +/- z sqrt(p(1-p)/n), taught everywhere but with poor coverage when n p or
    n (1 - p) is small (Brown, Cai and DasGupta 2001). conservative: p +/- z / (2 sqrt(n)), the worst case p = 1/2."""
    if not (0 <= count <= nobs) or nobs <= 0:
        raise ValueError("need 0 <= count <= nobs and nobs > 0")
    tab = _prop_intervals(count, nobs, alpha)
    if method not in tab.index:
        raise ValueError(f"method must be one of {list(tab.index)}")
    p = count / nobs
    res = _result("proportion", p, np.sqrt(p * (1 - p) / nobs), tuple(tab.loc[method, ["ci_low", "ci_high"]]),
                  np.nan, nobs, f"Confidence interval for a proportion ({method})", alpha, count=count, table=tab)
    if min(count, nobs - count) < 10:
        res["warnings"].append(f"n p or n (1 - p) is below 10 ({min(count, nobs - count)}): the Wald interval is "
                               "unreliable here; use Wilson or Clopper-Pearson.")
    if show:
        _print(res, ["all methods:\n" + tab.to_string(float_format=lambda v: f"{v:.4f}")])
    return res


def proportion_test(count, nobs, value: float | None = None, alternative: str = "two-sided",
                    alpha: float = ALPHA, labels: tuple = ("group 1", "group 2"), show: bool = True) -> dict:
    """One proportion (count, nobs scalars; H0: p = value) or the difference of two (count, nobs of length 2;
    H0: p1 = p2).

    One sample: score z test (SE under H0) and the exact binomial p-value; Wilson CI. Two samples: pooled z
    test, Newcombe's hybrid score CI for p1 - p2 and Fisher's exact p-value (use it when counts are small)."""
    _check_alt(alternative)
    sym = {"two-sided": "=", "greater": "<=", "less": ">="}[alternative]
    zc2, zc1 = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(1 - alpha)
    if np.ndim(count) == 0:
        x, n = int(count), int(nobs)
        if value is None or not 0 < value < 1:
            raise ValueError("give value, the proportion under H0, strictly between 0 and 1")
        p = x / n
        se0 = np.sqrt(value * (1 - value) / n)
        z = (p - value) / se0
        p_score = _p_value(z, stats.norm(), alternative)
        p_exact = float(stats.binomtest(x, n, value, alternative=alternative).pvalue)
        lo, hi = _wilson(x, n, zc2 if alternative == "two-sided" else zc1)
        ci = (lo, hi) if alternative == "two-sided" else ((lo, 1.0) if alternative == "greater" else (0.0, hi))
        res = _result("proportion", p, np.sqrt(p * (1 - p) / n), ci, p_exact, n,
                      "One-proportion test (exact binomial; score z alongside)", alpha, h0=f"p {sym} {value:g}",
                      statistic=float(z), p_score=p_score, p_exact=p_exact)
        if show:
            _print(res, [f"x = {x} of {n}; score z = {z:.4g} (p = {p_score:.3g}); exact binomial p = {p_exact:.3g}"])
        return res

    x = np.asarray(count, int)
    n = np.asarray(nobs, int)
    if x.size != 2 or n.size != 2:
        raise ValueError("for two proportions pass count=[x1, x2] and nobs=[n1, n2]")
    p1, p2 = x / n
    diff = p1 - p2
    pp = x.sum() / n.sum()
    se0 = np.sqrt(pp * (1 - pp) * (1 / n[0] + 1 / n[1]))
    z = diff / se0 if se0 > 0 else np.nan
    pz = _p_value(z, stats.norm(), alternative)
    zc = zc2 if alternative == "two-sided" else zc1
    l1, u1 = _wilson(x[0], n[0], zc)
    l2, u2 = _wilson(x[1], n[1], zc)
    lo = diff - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = diff + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    if alternative == "greater":
        hi = 1.0
    elif alternative == "less":
        lo = -1.0
    table = np.array([[x[0], n[0] - x[0]], [x[1], n[1] - x[1]]])
    alt_f = {"two-sided": "two-sided", "greater": "greater", "less": "less"}[alternative]
    p_fisher = float(stats.fisher_exact(table, alternative=alt_f).pvalue)
    se = np.sqrt(p1 * (1 - p1) / n[0] + p2 * (1 - p2) / n[1])
    res = _result(f"difference in proportions ({labels[0]} - {labels[1]})", diff, se, (lo, hi), pz, int(n.sum()),
                  "Two-proportion z test (pooled), Newcombe CI", alpha, h0=f"p1 - p2 {sym} 0", statistic=float(z),
                  proportions=(float(p1), float(p2)), p_fisher=p_fisher)
    if (np.outer(table.sum(1), table.sum(0)) / table.sum() < 5).any():
        res["warnings"].append(f"an expected count is below 5: use Fisher's exact p = {p_fisher:.3g} instead of the z test.")
    if show:
        _print(res, [f"proportions {p1:.4g} vs {p2:.4g}; z = {z:.4g}; Fisher exact p = {p_fisher:.3g}"])
    return res


def n_for_proportion(margin: float, p: float | None = None, alpha: float = ALPHA,
                     population: int | None = None) -> dict:
    """Sample size so that the half-width of a (Wald) CI for a proportion is at most `margin`:
    n0 = z^2 p (1 - p) / margin^2, with p = 0.5 (the worst case) when no prior guess is given, and the
    finite-population correction n = n0 / (1 + (n0 - 1) / N) when the population size N is known."""
    z = stats.norm.ppf(1 - alpha / 2)
    pp = 0.5 if p is None else p
    n0 = z ** 2 * pp * (1 - pp) / margin ** 2
    n = n0 / (1 + (n0 - 1) / population) if population else n0
    return {"n": int(np.ceil(n - 1e-9)), "n0": float(n0), "p_assumed": pp, "margin": margin, "alpha": alpha,
            "population": population,
            "note": "worst case p = 0.5" if p is None else "uses the guessed p; if the true p is nearer 0.5 the margin is wider"}


# =============================================================================
# Distributions
# =============================================================================
def _pool(obs, exp, labels, min_expected):
    obs, exp, labels = list(obs), list(exp), list(labels)
    pooled = False
    while len(exp) > 1 and min(exp) < min_expected:
        i = int(np.argmin(exp))
        j = i + 1 if i == 0 else (i - 1 if i == len(exp) - 1 else (i - 1 if exp[i - 1] <= exp[i + 1] else i + 1))
        a, b = sorted((i, j))
        obs[a:b + 1] = [obs[a] + obs[b]]
        exp[a:b + 1] = [exp[a] + exp[b]]
        labels[a:b + 1] = [f"{str(labels[a]).split('..')[0]}..{str(labels[b]).split('..')[-1]}"]
        pooled = True
    return np.array(obs, float), np.array(exp, float), labels, pooled


def chisq_fit(data=None, dist: str | None = None, observed=None, expected_probs=None, n_params: int | None = None,
              n_trials: int | None = None, bins: int | None = None, min_expected: float = 5.0,
              alpha: float = ALPHA, show: bool = True) -> dict:
    """Chi-square goodness of fit: does a probability model describe the data?

    Either observed counts with expected_probs (categories; n_params = parameters estimated from the data,
    default 0), or raw data with dist in {'poisson', 'binomial' (needs n_trials), 'normal', 'exponential'},
    whose parameters are estimated by maximum likelihood. Cells with expected count below min_expected are
    pooled with a neighbour. Degrees of freedom: k - a - 1 (k cells after pooling, a estimated parameters).
    For continuous data the cells are equal-probability bins of the fitted distribution; the verdict depends
    on the bins, so also look at normality()/QQ plots."""
    if observed is not None:
        obs = np.asarray(observed, float)
        probs = np.asarray(expected_probs, float)
        if probs.shape != obs.shape or abs(probs.sum() - 1) > 1e-6:
            raise ValueError("expected_probs must match observed and sum to 1")
        exp = obs.sum() * probs
        labels = [str(i) for i in range(obs.size)]
        a, name = int(n_params or 0), "given probabilities"
    else:
        x, _ = _clean(data)
        n = x.size
        if dist in ("poisson", "binomial"):
            if np.any(x < 0) or np.any(x != np.round(x)):
                raise ValueError(f"{dist} needs non-negative integer data")
            k_max = int(x.max())
            if dist == "poisson":
                lam = x.mean()
                fitted = stats.poisson(lam)
                name, a, params = f"Poisson(lambda = {lam:.4g})", 1, {"lambda": lam}
                top = k_max
            else:
                if n_trials is None:
                    raise ValueError("binomial needs n_trials")
                if k_max > n_trials:
                    raise ValueError(f"a value ({k_max}) exceeds n_trials ({n_trials})")
                pr = x.mean() / n_trials
                fitted = stats.binom(n_trials, pr)
                name, a, params = f"Binomial(n = {n_trials}, p = {pr:.4g})", 1, {"p": pr}
                top = n_trials
            cells = np.arange(0, top + 1)
            obs = np.array([(x == c).sum() for c in cells], float)
            probs = fitted.pmf(cells)
            if dist == "poisson":
                obs[-1] = (x >= top).sum()
                probs[-1] = fitted.sf(top - 1)
            exp = n * probs
            labels = [str(c) for c in cells]
            if dist == "poisson":
                labels[-1] = f"{top}+"
        elif dist in ("normal", "exponential"):
            if dist == "normal":
                mu, sg = x.mean(), x.std(ddof=0)
                fitted, a, name, params = stats.norm(mu, sg), 2, f"Normal(mu = {mu:.4g}, sigma = {sg:.4g})", {"mu": mu, "sigma": sg}
            else:
                if np.any(x < 0):
                    raise ValueError("exponential needs non-negative data")
                sc = x.mean()
                fitted, a, name, params = stats.expon(scale=sc), 1, f"Exponential(mean = {sc:.4g})", {"mean": sc}
            k = bins or int(np.clip(np.ceil(2 * n ** 0.4), 4, max(4, n // max(min_expected, 1))))
            edges = fitted.ppf(np.linspace(0, 1, k + 1))
            idx = np.clip(np.searchsorted(edges[1:-1], x, side="right"), 0, k - 1)
            obs = np.bincount(idx, minlength=k).astype(float)
            exp = np.full(k, n / k)
            labels = [f"[{edges[i]:.3g}, {edges[i + 1]:.3g})" for i in range(k)]
        else:
            raise ValueError("dist must be 'poisson', 'binomial', 'normal' or 'exponential', or pass observed counts")
        if n_params is not None:
            a = n_params
    obs, exp, labels, pooled = _pool(obs, exp, labels, min_expected)
    k = obs.size
    df = k - a - 1
    if df < 1:
        raise ValueError(f"no degrees of freedom left (k = {k} cells after pooling, {a} estimated parameters)")
    contrib = (obs - exp) ** 2 / exp
    stat = float(contrib.sum())
    p = float(stats.chi2.sf(stat, df))
    table = pd.DataFrame({"observed": obs, "expected": exp, "contribution": contrib}, index=labels)
    res = _result("chi-square statistic", stat, np.nan, (np.nan, np.nan), p, int(obs.sum()),
                  f"Chi-square goodness of fit to {name}", alpha, h0=f"the data follow {name}",
                  statistic=stat, df=df, k_cells=k, n_estimated=a, table=table)
    w = res["warnings"]
    w.append(f"df = k - a - 1 = {k} - {a} - 1 = {df}: each parameter estimated from the data costs one degree of freedom.")
    if pooled:
        w.append(f"cells with expected count below {min_expected:g} were pooled with a neighbour.")
    if observed is None and a > 0:
        w.append("the parameters were estimated from the raw data, not from the cell counts: the statistic then lies "
                 f"between chi2({df}) and chi2({k - 1}) (Chernoff-Lehmann), so the p-value can be somewhat too small"
                 + ("; for continuous data prefer normality() and a QQ plot." if dist in ("normal", "exponential") else "."))
    if (exp < 1).any() or (exp < 5).mean() > 0.2:
        w.append("after pooling, more than 20% of cells still expect fewer than 5: the chi-square approximation is rough.")
    if show:
        _print(res, [table.to_string(float_format=lambda v: f"{v:.3f}"), f"chi2 = {stat:.4g}, df = {df}"])
    return res


def normality(x, alpha: float = ALPHA, show: bool = True) -> dict:
    """Is a variable (or a residual) normal? Shapiro-Wilk (n <= 5000), Lilliefors (KS with estimated mean and
    SD), Jarque-Bera (skewness and kurtosis, asymptotic) and D'Agostino-Pearson (n >= 20), plus QQ-plot data.

    Read the tests with the sample size: with thousands of observations trivial departures reject; with a
    few dozen, large ones do not. Look at the QQ plot and the size of skewness and kurtosis. Normality
    matters for small-sample exact inference (t, chi-square and F on variances), prediction intervals for
    single values and tail risk; OLS coefficients and their robust SEs do not need it in large samples."""
    from statsmodels.stats.diagnostic import lilliefors

    a, n_missing = _clean(x)
    n = a.size
    if n < 8:
        raise ValueError("normality needs at least 8 values")
    rows = {}
    if n <= 5000:
        s = stats.shapiro(a)
        rows["Shapiro-Wilk"] = (float(s.statistic), float(s.pvalue))
    ks, pk = lilliefors(a, dist="norm")
    rows["Lilliefors (KS)"] = (float(ks), float(pk))
    jb = stats.jarque_bera(a)
    rows["Jarque-Bera"] = (float(jb.statistic), float(jb.pvalue))
    if n >= 20:
        dp = stats.normaltest(a)
        rows["D'Agostino-Pearson"] = (float(dp.statistic), float(dp.pvalue))
    tests = pd.DataFrame(rows, index=["statistic", "p"]).T
    (osm, osr), (slope, intercept, r) = stats.probplot(a, dist="norm")
    qq = pd.DataFrame({"theoretical": osm, "sample": osr})
    p_main = tests.iloc[0]["p"]
    res = _result(f"{tests.index[0]} statistic", float(tests.iloc[0]["statistic"]), np.nan, (np.nan, np.nan), p_main, n,
                  "Normality tests", alpha, h0="the data come from a normal distribution", tests=tests, qq=qq,
                  qq_line=(float(intercept), float(slope)), qq_r=float(r),
                  skewness=float(stats.skew(a, bias=False)), excess_kurtosis=float(stats.kurtosis(a, bias=False)))
    w = res["warnings"]
    if n > 5000:
        w.append("Shapiro-Wilk is not computed above n = 5000.")
    if n >= 1000:
        w.append(f"n = {n}: tiny departures reject; judge by the QQ plot and by skewness {res['skewness']:.2f} and "
                 f"excess kurtosis {res['excess_kurtosis']:.2f}.")
    if n < 30:
        w.append(f"n = {n}: these tests have little power; not rejecting does not show normality.")
    if show:
        _print(res, [tests.to_string(float_format=lambda v: f"{v:.4g}"),
                     f"skewness {res['skewness']:.3g}, excess kurtosis {res['excess_kurtosis']:.3g}; "
                     f"QQ correlation {r:.4f} (data in res['qq'] for the plot)"])
    return res


# =============================================================================
# Tables
# =============================================================================
def _odds_ratio_ci(t, alpha):
    a, b, c, d = t.ravel().astype(float)
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    lor = np.log(a * d / (b * c))
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    z = stats.norm.ppf(1 - alpha / 2)
    return float(np.exp(lor)), (float(np.exp(lor - z * se)), float(np.exp(lor + z * se)))


def contingency(table=None, data: pd.DataFrame | None = None, row: str | None = None, col: str | None = None,
                strata: str | None = None, correction: bool = False, alpha: float = ALPHA, show: bool = True) -> dict:
    """Association between two categorical variables.

    Pass a table of counts (array or DataFrame) or data with row and col. Chi-square test of independence,
    Cramer's V (0 = none, 1 = perfect), expected counts and adjusted standardized residuals (beyond +/-2 marks
    the cells that drive the association). For a 2x2 table: Fisher's exact test, the odds ratio with a
    Woolf CI and the difference in the share of the first column between rows.
    strata (with data, 2x2 within each stratum): the comparison inside each stratum, Cochran-Mantel-Haenszel
    test and pooled odds ratio, Breslow-Day test of a common odds ratio, and a flag when the crude
    association reverses inside every stratum (Simpson's paradox)."""
    if table is None:
        if data is None or row is None or col is None:
            raise ValueError("pass a table of counts, or data with row and col")
        tab = pd.crosstab(data[row], data[col])
    else:
        tab = table if isinstance(table, pd.DataFrame) else pd.DataFrame(np.asarray(table))
    t = tab.to_numpy(dtype=float)
    if t.ndim != 2 or min(t.shape) < 2:
        raise ValueError("need at least a 2x2 table")
    n = t.sum()
    chi2, p, dof, exp = stats.chi2_contingency(t, correction=correction)
    rp, cp = t.sum(1) / n, t.sum(0) / n
    adj = (t - exp) / np.sqrt(exp * np.outer(1 - rp, 1 - cp))
    v = float(np.sqrt(chi2 / (n * (min(t.shape) - 1))))
    res = _result("Cramer's V", v, np.nan, (np.nan, np.nan), float(p), int(n),
                  "Chi-square test of independence" + (" (Yates)" if correction else ""), alpha,
                  h0="rows and columns are independent", statistic=float(chi2), df=int(dof), cramers_v=v,
                  observed=tab, expected=pd.DataFrame(exp, index=tab.index, columns=tab.columns),
                  adjusted_residuals=pd.DataFrame(adj, index=tab.index, columns=tab.columns))
    w = res["warnings"]
    extra = [f"chi2 = {chi2:.4g}, df = {dof}; Cramer's V = {v:.3f}"]
    small = (exp < 5).mean()
    if t.shape == (2, 2):
        fe = stats.fisher_exact(t)
        orr, or_ci = _odds_ratio_ci(t, alpha)
        sh = t[:, 0] / t.sum(1)
        res.update({"p_fisher": float(fe.pvalue), "odds_ratio": orr, "odds_ratio_ci": or_ci,
                    "share_first_col": (float(sh[0]), float(sh[1])), "share_difference": float(sh[0] - sh[1])})
        extra.append(f"Fisher exact p = {fe.pvalue:.3g}; odds ratio = {orr:.3g} CI [{or_ci[0]:.3g}, {or_ci[1]:.3g}]; "
                     f"share of '{tab.columns[0]}' by row: {sh[0]:.3f} vs {sh[1]:.3f}")
        if (exp < 5).any():
            w.append(f"an expected count is below 5: use Fisher's exact p = {fe.pvalue:.3g}.")
    elif small > 0.2 or (exp < 1).any():
        w.append(f"{small:.0%} of cells expect fewer than 5: the chi-square approximation is unreliable; pool "
                 "categories (or use an exact test).")
    if strata is not None:
        if data is None:
            raise ValueError("strata needs data with row, col and the strata column")
        from statsmodels.stats.contingency_tables import StratifiedTable
        rows_, tabs = [], []
        r_levels, c_levels = list(tab.index), list(tab.columns)
        if len(r_levels) != 2 or len(c_levels) != 2:
            raise ValueError("the stratified analysis needs a binary row and a binary col")
        for s, g in data.groupby(strata):
            ts = pd.crosstab(g[row], g[col]).reindex(index=r_levels, columns=c_levels, fill_value=0).to_numpy(float)
            sh = ts[:, 0] / ts.sum(1)
            rows_.append({"stratum": s, "n": int(ts.sum()), "share_row1": sh[0], "share_row2": sh[1],
                          "difference": sh[0] - sh[1]})
            tabs.append(ts)
        st = pd.DataFrame(rows_).set_index("stratum")
        crude = res["share_difference"]
        strat = StratifiedTable([x for x in tabs])
        cmh = strat.test_null_odds()
        bd = strat.test_equal_odds()
        res.update({"strata_table": st, "cmh_stat": float(cmh.statistic), "cmh_p": float(cmh.pvalue),
                    "mh_odds_ratio": float(strat.oddsratio_pooled), "breslow_day_p": float(bd.pvalue)})
        signs = np.sign(st["difference"].to_numpy())
        if np.all(signs == signs[0]) and signs[0] != 0 and np.sign(crude) == -signs[0]:
            res["simpson"] = True
            w.append(f"Simpson's paradox: the crude difference ({crude:+.3f}) has the opposite sign to the "
                     f"difference inside every level of '{strata}'; the crude comparison is confounded by it. "
                     "Report the stratified comparison (Mantel-Haenszel).")
        else:
            res["simpson"] = False
        extra.append("by stratum:\n" + st.to_string(float_format=lambda v: f"{v:.3f}"))
        extra.append(f"Cochran-Mantel-Haenszel p = {cmh.pvalue:.3g}; Mantel-Haenszel odds ratio = "
                     f"{strat.oddsratio_pooled:.3g}; Breslow-Day (common odds ratio) p = {bd.pvalue:.3g}")
    if show:
        _print(res, extra)
    return res


# =============================================================================
# Ranks and correlation
# =============================================================================
def nonparametric(x, y=None, paired: bool = False, mu0: float = 0.0, alternative: str = "two-sided",
                  alpha: float = ALPHA, show: bool = True) -> dict:
    """Rank tests that do not assume normality.

    One sample (y=None) or paired: Wilcoxon signed-rank on x - mu0 (or x - y), with the Hodges-Lehmann
    pseudo-median (median of Walsh averages) and its distribution-free CI. Two independent samples:
    Mann-Whitney U with the Hodges-Lehmann shift (median of all differences x_i - y_j) and its CI, the
    probability of superiority P(X > Y) and the rank-biserial correlation.
    Mann-Whitney tests whether one distribution tends to give larger values; it is a test of medians or means
    only if the two shapes are the same. CIs use the normal approximation for the order statistics."""
    _check_alt(alternative)
    z = stats.norm.ppf(1 - alpha / 2) if alternative == "two-sided" else stats.norm.ppf(1 - alpha)

    def _bounds(sorted_vals, k, m):
        lo, hi = float(sorted_vals[k - 1]), float(sorted_vals[m - k])
        return (lo, hi) if alternative == "two-sided" else ((lo, np.inf) if alternative == "greater" else (-np.inf, hi))

    small = False
    if y is None or paired:
        if y is None:
            d, _ = _clean(np.asarray(x, float) - mu0)
            what, h0 = f"pseudo-median of x - {mu0:g}", f"distribution of x symmetric about {mu0:g}"
        else:
            xs, ys = np.asarray(x, float), np.asarray(y, float)
            keep = ~(np.isnan(xs) | np.isnan(ys))
            d = xs[keep] - ys[keep]
            what, h0 = "pseudo-median of x - y (paired)", "differences symmetric about 0"
        dn = d[d != 0]
        n = dn.size
        if n < 2:
            raise ValueError("need at least 2 non-zero differences")
        wres = stats.wilcoxon(dn, alternative=alternative)
        ranks = stats.rankdata(np.abs(dn))
        tp, tm = ranks[dn > 0].sum(), ranks[dn < 0].sum()
        rbc = (tp - tm) / (tp + tm)
        walsh = ((dn[:, None] + dn[None, :]) / 2)[np.triu_indices(n)]
        walsh.sort()
        m = walsh.size
        k = int(np.floor(m / 2 - z * np.sqrt(n * (n + 1) * (2 * n + 1) / 24)))
        small = k < 1
        k = max(k, 1)
        ci = _bounds(walsh, k, m)
        est = float(np.median(walsh)) + (mu0 if y is None else 0.0)
        if y is None:
            ci = (ci[0] + mu0, ci[1] + mu0)
            what = "pseudo-median (Hodges-Lehmann)"
        res = _result(what, est, np.nan, ci, float(wres.pvalue), int(d.size), "Wilcoxon signed-rank test", alpha,
                      h0=h0, statistic=float(wres.statistic), rank_biserial=float(rbc), zeros_dropped=int(d.size - n))
        if d.size - n:
            res["warnings"].append(f"{d.size - n} zero difference(s) dropped (Wilcoxon's convention).")
        extra = [f"W = {wres.statistic:.4g}; rank-biserial correlation = {rbc:.3f}"]
    else:
        a, _ = _clean(x)
        b, _ = _clean(y)
        n1, n2 = a.size, b.size
        mw = stats.mannwhitneyu(a, b, alternative=alternative)
        u1 = float(mw.statistic)
        ps = u1 / (n1 * n2)
        rbc = 2 * ps - 1
        if n1 * n2 <= 4_000_000:
            diffs = np.subtract.outer(a, b).ravel()
        else:
            rng = np.random.default_rng(0)
            diffs = rng.choice(a, 4_000_000) - rng.choice(b, 4_000_000)
        diffs.sort()
        m = diffs.size
        k = int(np.floor(n1 * n2 / 2 - z * np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)))
        small = k < 1
        k = max(k, 1)
        ci = _bounds(diffs, k, m) if n1 * n2 <= 4_000_000 else (np.nan, np.nan)
        res = _result("location shift x - y (Hodges-Lehmann)", float(np.median(diffs)), np.nan, ci, float(mw.pvalue),
                      n1 + n2, "Mann-Whitney U test (Wilcoxon rank-sum)", alpha,
                      h0="P(X > Y) = P(Y > X)", statistic=u1,
                      prob_superiority=float(ps), rank_biserial=float(rbc), n1=n1, n2=n2)
        res["warnings"].append("the test compares whole distributions; read the shift as a difference of medians "
                               "only if the two shapes are similar.")
        extra = [f"U = {u1:.4g}; P(X > Y) + P(X = Y)/2 = {ps:.3f}; rank-biserial correlation = {rbc:.3f}"]
    if small:
        res["warnings"].append(f"the sample is too small for a distribution-free {1 - alpha:.0%} interval: the bounds "
                               "shown are the most extreme possible and cover less than the stated level.")
    if alternative != "two-sided":
        res["warnings"].append("one-sided test: the interval is a one-sided bound matching the alternative.")
    if show:
        _print(res, extra)
    return res


def correlation_test(x, y, method: str = "pearson", alpha: float = ALPHA, show: bool = True) -> dict:
    """Correlation with a test of H0: rho = 0 and a CI by Fisher's z transform.

    pearson: linear association (SE of z: 1/sqrt(n-3)); spearman: monotone association on ranks
    (Bonett-Wright SE sqrt((1 + r^2/2)/(n-3))); kendall: concordance (Fieller SE sqrt(0.437/(n-4))).
    With Pearson, Spearman is computed too: a large gap signals outliers or curvature (look at the scatter)."""
    xs, ys = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    keep = ~(np.isnan(xs) | np.isnan(ys))
    xs, ys = xs[keep], ys[keep]
    n = xs.size
    if n < 5:
        raise ValueError("correlation_test needs at least 5 complete pairs")
    if method == "pearson":
        r, p = stats.pearsonr(xs, ys)
        se_z = 1 / np.sqrt(n - 3)
    elif method == "spearman":
        r, p = stats.spearmanr(xs, ys)
        se_z = np.sqrt((1 + r ** 2 / 2) / (n - 3))
    elif method == "kendall":
        r, p = stats.kendalltau(xs, ys)
        se_z = np.sqrt(0.437 / (n - 4))
    else:
        raise ValueError("method must be 'pearson', 'spearman' or 'kendall'")
    r, p = float(r), float(p)
    zc = stats.norm.ppf(1 - alpha / 2)
    zr = np.arctanh(np.clip(r, -0.999999, 0.999999))
    ci = (float(np.tanh(zr - zc * se_z)), float(np.tanh(zr + zc * se_z)))
    res = _result(f"{method} correlation", r, np.nan, ci, p, n, f"Correlation test ({method})", alpha,
                  h0="rho = 0")
    if (~keep).any():
        res["warnings"].append(f"{int((~keep).sum())} incomplete pair(s) dropped.")
    if method == "pearson":
        rs = float(stats.spearmanr(xs, ys)[0])
        res["spearman"] = rs
        if abs(rs - r) > 0.2:
            res["warnings"].append(f"Pearson {r:.2f} and Spearman {rs:.2f} differ: outliers or a non-linear "
                                   "relation; look at the scatter plot before reading Pearson.")
    res["warnings"].append("a correlation is not an effect: confounders and reverse causation are not addressed.")
    if show:
        _print(res)
    return res


# =============================================================================
# Power and many tests
# =============================================================================
def power_curve(effect, n, sd: float = 1.0, test: str = "t", design: str = "two_sample",
                alternative: str = "two-sided", alpha: float = ALPHA, ratio: float = 1.0, show: bool = True) -> pd.DataFrame:
    """Power (and beta = 1 - power) of a z or t test for every combination of true effects and sample sizes.

    effect: true difference in means (scalar or list), in the units of sd. n: sample size (per group for
    two_sample; pairs for paired), scalar or list. design: 'one_sample', 'paired' (sd = SD of the differences)
    or 'two_sample' (n2 = ratio * n). The t power uses the noncentral t distribution, exact under normality.
    Use it to say how large an effect a non-significant test could have missed, next to simulation.power_mde."""
    _check_alt(alternative)
    rows = []
    for nn in np.atleast_1d(n):
        for ef in np.atleast_1d(effect):
            d = ef / sd
            if design in ("one_sample", "paired"):
                df, nc = nn - 1, d * np.sqrt(nn)
            elif design == "two_sample":
                n2 = ratio * nn
                df, nc = nn + n2 - 2, d / np.sqrt(1 / nn + 1 / n2)
            else:
                raise ValueError("design must be 'one_sample', 'paired' or 'two_sample'")
            if test == "z":
                if alternative == "two-sided":
                    c = stats.norm.ppf(1 - alpha / 2)
                    pw = stats.norm.sf(c - nc) + stats.norm.cdf(-c - nc)
                elif alternative == "greater":
                    pw = stats.norm.sf(stats.norm.ppf(1 - alpha) - nc)
                else:
                    pw = stats.norm.cdf(-stats.norm.ppf(1 - alpha) - nc)
            elif test == "t":
                if alternative == "two-sided":
                    c = stats.t.ppf(1 - alpha / 2, df)
                    pw = stats.nct.sf(c, df, nc) + stats.nct.cdf(-c, df, nc)
                elif alternative == "greater":
                    pw = stats.nct.sf(stats.t.ppf(1 - alpha, df), df, nc)
                else:
                    pw = stats.nct.cdf(-stats.t.ppf(1 - alpha, df), df, nc)
            else:
                raise ValueError("test must be 't' or 'z'")
            rows.append({"n": nn, "effect": ef, "d": d, "power": float(pw), "beta": float(1 - pw)})
    out = pd.DataFrame(rows)
    if show:
        print(f"\nPower of a {design.replace('_', '-')} {test} test ({alternative}, alpha = {alpha:g}, sd = {sd:g})")
        print(out.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    return out


def n_for_power(effect: float, sd: float = 1.0, power: float = 0.8, design: str = "two_sample",
                alternative: str = "two-sided", alpha: float = ALPHA, ratio: float = 1.0, test: str = "t",
                max_n: int = 1_000_000, show: bool = True) -> dict:
    """Smallest sample size (per group for two_sample, pairs for paired) whose power for a true difference
    `effect` reaches `power`, from the same exact noncentral-t calculation as power_curve.

    Use it for "how many would we need to detect a difference this size?": compute the n for that effect,
    never read it off a power table built for other effect sizes. The effect is a guess about the truth;
    plugging in the observed difference answers "if the truth were what we saw", not "what is the truth"."""
    if effect == 0:
        raise ValueError("effect must be non-zero")
    f = lambda nn: power_curve(effect, nn, sd=sd, test=test, design=design, alternative=alternative,  # noqa: E731
                               alpha=alpha, ratio=ratio, show=False)["power"].iloc[0]
    lo, hi = 2, 2
    while f(hi) < power:
        lo, hi = hi, hi * 2
        if hi > max_n:
            raise ValueError(f"power {power} not reached below n = {max_n}")
    while hi - lo > 1:                            # power increases with n: bisect on integers
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if f(mid) < power else (lo, mid)
    n = hi if f(hi) >= power else hi + 1
    out = {"n": int(n), "n_total": int(n + (np.ceil(ratio * n) if design == "two_sample" else 0)),
           "effect": effect, "d": effect / sd, "power": float(f(n)), "target_power": power, "alpha": alpha,
           "design": design, "alternative": alternative,
           "method": f"exact {test} power ({design}, {alternative}), smallest n with power >= {power:g}"}
    if show:
        unit = "per group" if design == "two_sample" else ("pairs" if design == "paired" else "observations")
        print(f"\nTo detect a difference of {effect:g} (d = {effect / sd:.3f}) with power {power:.0%} at alpha "
              f"{alpha:g}: n = {out['n']} {unit} (power {out['power']:.3f})")
    return out


def adjust_pvalues(pvalues, method: str = "holm", alpha: float = ALPHA) -> pd.DataFrame:
    """Adjust p-values for multiple tests. bonferroni and holm control the family-wise error rate (the chance
    of any false rejection; Holm is uniformly more powerful than Bonferroni); bh (Benjamini-Hochberg) controls
    the false discovery rate under independence or positive dependence; by (Benjamini-Yekutieli) under any
    dependence. Returns the adjusted p-values and the rejections at alpha, in the input order; missing p-values
    stay missing and do not count in m."""
    p_all = np.asarray(pvalues, float)
    ok = ~np.isnan(p_all)
    p = p_all[ok]
    m = p.size
    order = np.argsort(p)
    ps = p[order]
    if method == "bonferroni":
        adj_s = np.minimum(ps * m, 1)
    elif method == "holm":
        adj_s = np.minimum(np.maximum.accumulate((m - np.arange(m)) * ps), 1)
    elif method in ("bh", "by"):
        cm = np.sum(1 / np.arange(1, m + 1)) if method == "by" else 1.0
        raw = ps * m * cm / np.arange(1, m + 1)
        adj_s = np.minimum(np.minimum.accumulate(raw[::-1])[::-1], 1)
    else:
        raise ValueError("method must be 'bonferroni', 'holm', 'bh' or 'by'")
    adj_ok = np.empty(m)
    adj_ok[order] = adj_s
    adj = np.full(p_all.size, np.nan)
    adj[ok] = adj_ok
    idx = pvalues.index if isinstance(pvalues, pd.Series) else range(p_all.size)
    return pd.DataFrame({"p": p_all, "p_adjusted": adj, "reject": np.where(ok, adj < alpha, False)}, index=idx)
