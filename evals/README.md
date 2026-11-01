# Evals

Blind behavioural tests of the `econometrics` skill. The test suites in `tests/` check that the code computes the right numbers; these evals check that, given a plain question, Claude chooses the right method, uses the right functions and gives advice that the code actually carries out.

## Protocol

1. Install the release candidate (`dist/econometrics.skill`) under Settings → Capabilities → Skills.
2. Open a **fresh conversation** for each case. A conversation that has already discussed the case knows the answer and contaminates the test.
3. Paste only the `prompt` from `cases.json`, with no context.
4. Grade the answer against every `must` and `must_not` criterion. Read the code in the answer against the text: an answer that recommends a step and then shows code doing something else fails.
5. Record the result in the table below. A case passes only if all `must` hold and no `must_not` is triggered.
6. For each failure, decide whether the fix belongs in the code (a missing function or option), in `SKILL.md` (a rule Claude should follow) or in a reference file, and add a criterion here if the failure was new.

## History

| Version | Case | Result | Notes |
|---|---|---|---|
| 1.7 | llm-coded-outcome-judge-sex | pass | DSL and naive vs corrected; found that a shared draw seed failed with an uninformative error (fixed in 1.8) |
| 1.7 | churn-drivers-by-customer-type | pass | Leaf vs pooled by AUC and lift; the purity-based tree missed slope heterogeneity by customer type (benchmarks and `segment_by` added in 1.8) |
| 1.8 | llm-coded-outcome-judge-sex | pass with one error | Text said cluster by judge, code used `cluster="vara"` (rule added to `SKILL.md` in 1.9) |
| 1.8 | churn-drivers-by-customer-type | pass with gaps | Recommended a temporal split and calibration that the function did not do; categorical customer type would have failed; benchmark dropped a regressor (all addressed in 1.9) |
| 2.0 | discount-targeting-heterogeneity | pass with one gap | Non-random assignment, X vs W, bad controls, overlap, BLP/GATES before subgroups, cost converted to churn value, held-out policy value; no data attached, so no code and no `hte.cate_fit` named (rule "without data: plan + runnable script with the skill's functions" added before the v2.0 tag) |
| 2.0 | teach-instrumental-variables | pass with two gaps | Full method card, Monte Carlo of OLS bias, weak and invalid instruments, LATE, effective F and 104.7; exercise had no worked solution and the simulation code was cited as a sandbox path (Teacher mode rules tightened before the v2.0 tag) |
| 2.0 | review-few-clusters-did | pass with one gap | Opinion format, few treated clusters, staggered TWFE, pre-trends, HonestDiD, leave-one-out, randomization inference; recommended `boottest`/`fwildclusterboot` instead of `estimation.wild_cluster_bootstrap` and did not name `causal.att_gt` (reviewer rule and few-treated-clusters caveat added before the v2.0 tag) |
