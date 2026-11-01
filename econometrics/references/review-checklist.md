# Reviewer mode

Use when the user asks to review, audit or referee someone else's empirical work: a paper, a consulting report, an expert opinion, a model in a spreadsheet, a regression table in a deck. The output is a written opinion: what the work claims, whether the evidence supports it, and what would change the conclusion.

## 1. Output

1. **Summary of the claim** in two sentences: the estimand, the design, the headline number.
2. **Verdict**: supported / partially supported / not supported, with one sentence of why.
3. **Findings**, each with a severity:
   - **Critical**: invalidates the main conclusion (identification fails, wrong estimand, coding error that drives the result).
   - **Major**: the conclusion may survive, but needs a new analysis to know (missing robustness, wrong SEs, unaddressed confounder).
   - **Minor**: presentation, precision, missing detail.
   Each finding states where it is (table, page, cell), why it matters, and the concrete check or fix.
4. **What would change my mind**: the analyses whose results would move the verdict.

Reproduce numbers when data are available (the scripts of this skill); otherwise say which claims could not be checked.

## 2. Checklist

**Question and estimand**
- [ ] Is the estimand stated (ATE, ATT, LATE, elasticity, forecast)? Does the headline sentence match it?
- [ ] Is causal language used without an identification strategy?

**Design and identification**
- [ ] Is the identifying assumption written and defended with evidence (balance, pre-trends, first stage, density test, pre-period fit)?
- [ ] Bad controls (post-treatment variables)? Selection on the outcome? Reverse causality?
- [ ] Staggered DiD estimated with plain TWFE? Treatment dates verified?
- [ ] IV: first-stage F (effective F), exclusion argued, LATE interpreted for compliers?

**Inference**
- [ ] SEs robust to heteroskedasticity; clustered at the level where the regressor of interest varies; number of clusters reported; wild bootstrap with few clusters?
- [ ] Time series: HAC SEs, stationarity, spurious regression (R² > DW)?
- [ ] Multiple testing: how many outcomes, subgroups, specifications were tried?
- [ ] Heterogeneity claims backed by a test (BLP/GATES or pre-specified interactions), not by picking subgroups?

**Specification and data**
- [ ] Functional form checked (RESET, residual plots); units and transformations correct; elasticities read correctly?
- [ ] Missing data handled and N reported per model; outliers and influential points checked?
- [ ] Nominal vs real, seasonality, breaks, frequency alignment?
- [ ] Look-ahead in forecasts, signals or risk models?

**Robustness and magnitude**
- [ ] At least two robustness checks, with results that do not depend on one specification?
- [ ] Economic significance discussed, not only p-values; intervals reported?
- [ ] Sensitivity to unobserved confounding (Oster, Cinelli-Hazlett) for selection-on-observables claims?

**Reproducibility**
- [ ] Data sources, code and sample construction documented? Can the main table be reproduced?

## 3. Tone

Be specific and fair: credit what is done well, separate fatal flaws from improvements, and propose the check that settles each doubt. Never invent a number the work does not report; when a figure cannot be verified, say so.
