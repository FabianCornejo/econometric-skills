# %% [markdown]
# # 2. Instrumental variables: return to education for married women (MROZ)
#
# **Question:** is the OLS return to education biased by omitted ability, and what does 2SLS give
# using parents' education as instruments? (Wooldridge Examples 15.1–15.8.)

# %%
import numpy as np
import statsmodels.formula.api as smf
import wooldridge as woo
from linearmodels.iv import IV2SLS

from _setup import *  # noqa: F401,F403
import diagnostics as dg
import reporting as rp

# %% [markdown]
# ## Steps 1–3 — Question, structure, data
# Wages are observed only for women in the labor force (428 of 753), so this is a selected sample;
# the example follows the textbook and conditions on working.

# %%
m = woo.data("mroz").dropna(subset=["wage"]).copy()
m["lwage"] = np.log(m["wage"])
m["expersq"] = m["exper"] ** 2
len(m)

# %% [markdown]
# ## Step 4 — OLS and 2SLS
# `debiased=True` gives the n−k small-sample correction used in the textbook (Stata: `small`).

# %%
ols = smf.ols("lwage ~ educ + exper + expersq", m).fit(cov_type="HC3")
iv = IV2SLS.from_formula("lwage ~ 1 + exper + expersq + [educ ~ motheduc + fatheduc]", m) \
    .fit(cov_type="robust", debiased=True)
print(iv.first_stage)

# %% [markdown]
# ## Step 5 — Instrument diagnostics
# 1. **Relevance:** the first-stage F of the excluded instruments.
# 2. **Endogeneity:** does educ need instrumenting at all? (Durbin-Wu-Hausman, control function.)
# 3. **Overidentification:** two instruments for one endogenous regressor allow a (weak) test.

# %%
endo = dg.endogeneity_test(m["lwage"], m[["exper", "expersq"]], m[["educ"]], m[["motheduc", "fatheduc"]])
over = dg.overid_test(iv.resids, m[["exper", "expersq"]], m[["motheduc", "fatheduc"]], n_endog=1)

# %% [markdown]
# The instruments pass the Staiger-Stock rule (first-stage F ≈ 49 > 10) but not the stricter Lee et
# al. (2022) threshold of 104.7, so an Anderson-Rubin interval would be a useful complement. The
# endogeneity test does not reject at conventional levels (p ≈ 0.12), and the overidentification
# test does not reject either — which does **not** prove the instruments valid: parents' education
# may affect wages through other channels (networks, inherited ability).
#
# ## Step 6 — Robustness: one instrument at a time

# %%
iv_m = IV2SLS.from_formula("lwage ~ 1 + exper + expersq + [educ ~ motheduc]", m).fit(cov_type="robust", debiased=True)
iv_f = IV2SLS.from_formula("lwage ~ 1 + exper + expersq + [educ ~ fatheduc]", m).fit(cov_type="robust", debiased=True)
rp.regression_table({"OLS": ols, "2SLS both": iv, "2SLS mother": iv_m, "2SLS father": iv_f},
                    keep=["educ", "exper", "expersq"], labels={"educ": "Education", "exper": "Experience",
                                                               "expersq": "Experience²"})

# %% [markdown]
# ## Step 7 — Report
# OLS gives 0.108 log points per year of education (about 11%); 2SLS gives 0.061 (about 6%), with a
# confidence interval about 2.5 times wider that runs from slightly below zero to 0.127 — so the 2SLS
# return is **not significant at 5%**. The direction matches the ability-bias story (OLS too high),
# but the 2SLS interval contains the OLS estimate and the endogeneity test does not reject, so these
# data cannot distinguish the two. That is the usual price of IV: less bias, much less precision.
# Instrument validity remains an economic argument, not a test result.
