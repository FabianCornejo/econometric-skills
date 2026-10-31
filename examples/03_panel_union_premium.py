# %% [markdown]
# # 3. Panel data: the union wage premium (WAGEPAN)
#
# **Question:** how much more do union members earn, once fixed differences between workers are
# accounted for? 545 men observed 1980–1987 (Wooldridge Example 14.4).

# %%
import wooldridge as woo
from linearmodels.panel import PanelOLS, PooledOLS, RandomEffects

from _setup import *  # noqa: F401,F403
import diagnostics as dg
import reporting as rp

# %% [markdown]
# ## Steps 1–3
# Expected sign: positive premium. Threat: workers who join unions may differ in unobserved ways
# (skill, sector) that also affect wages → compare pooled OLS, RE and FE, and test FE vs RE.

# %%
w = woo.data("wagepan")
p = w.set_index(["nr", "year"], drop=False)
print(f"{w['nr'].nunique()} workers × {w['year'].nunique()} years; union share {w['union'].mean():.1%}")

# %% [markdown]
# ## Step 4 — Estimators, all with errors clustered by worker

# %%
f_rhs = "expersq + union + married + C(year)"
pooled = PooledOLS.from_formula(f"lwage ~ 1 + educ + black + hisp + {f_rhs}", p) \
    .fit(cov_type="clustered", cluster_entity=True)
re = RandomEffects.from_formula(f"lwage ~ 1 + educ + black + hisp + {f_rhs}", p) \
    .fit(cov_type="clustered", cluster_entity=True)
fe = PanelOLS.from_formula(f"lwage ~ {f_rhs} + EntityEffects", p, drop_absorbed=True) \
    .fit(cov_type="clustered", cluster_entity=True)

# %% [markdown]
# ## Step 5 — FE or RE? The Mundlak / correlated random effects test
# Robust to heteroskedasticity and serial correlation (clustered), unlike the classical Hausman test.

# %%
mk = dg.mundlak_test(w, "lwage", ["expersq", "union", "married"], entity="nr", time="year",
                     x_const=["educ", "black", "hisp"])

# %% [markdown]
# The unit means are jointly significant: unobserved worker effects are correlated with union
# status, marriage and experience, so RE (and pooled OLS) are inconsistent. The CRE model reproduces
# the FE coefficients exactly while still estimating time-invariant variables (education, race).
#
# ## Step 6 — Side by side

# %%
rp.regression_table({"Pooled": pooled, "RE": re, "FE": fe, "CRE (Mundlak)": mk["results"]},
                    keep=["union", "married", "expersq", "educ"],
                    labels={"union": "Union member", "married": "Married", "expersq": "Experience²",
                            "educ": "Education"},
                    fe_rows={"Worker FE": ["NO", "NO", "YES", "means"], "Year FE": ["YES"] * 4},
                    se_note="Standard errors clustered by worker in parentheses")

# %% [markdown]
# ## Step 7 — Report
# Comparing the same worker in union and non-union years, membership is associated with about 8%
# higher wages (FE: 0.080 log points), less than half of the pooled estimate (0.189); RE (0.110) sits
# in between because it only partly removes the worker effects. Much of the raw premium reflects
# who joins unions. The FE estimate still assumes that switching in and out of unions is unrelated to
# time-varying wage shocks.
