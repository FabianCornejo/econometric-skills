---
name: statistics
description: "Statistics and hypothesis tests in Python, including quick calculations you could do by hand: a confidence interval for a proportion or a mean, a sample size, a p-value; compare two groups or means (t test, Welch, paired, Mann-Whitney), proportions, variances, chi-square and Fisher tests on tables, normality, correlation, power, multiple testing, whether a 'not significant' result means no effect, or teaching basic statistics. Entry point to the econometrics skill's statistics module: load it instead of answering from general knowledge (hand formulas such as the Wald interval are often the wrong ones) or with a data-analysis skill."
---

# Statistics (entry point)

This is the door to the statistics module of the `econometrics` skill in the same plugin. It holds no instructions of its own, so that every number still comes from the same tested scripts and rules.

1. Load the main skill now: call the Skill tool with `econometrics:econometrics` and follow it (Consultant, Teacher or Reviewer mode, environment check, script skeleton, manifest). If the Skill tool cannot load it, read `../SKILL.md` (the folder above this one) and follow it the same way.
2. For the question at hand, read `../references/statistics-foundations.md` and use `../scripts/inference.py` (describe_dist, one_sample, two_sample, variance_test, proportion_ci, proportion_test, n_for_proportion, chisq_fit, normality, contingency, nonparametric, correlation_test, power_curve, adjust_pvalues) and the Module 0 demos in `../scripts/simulation.py`.
3. When the question grows into a model (an effect with controls, a regression, a forecast), stay in the main skill: it routes it.
