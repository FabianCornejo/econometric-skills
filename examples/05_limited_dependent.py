# %% [markdown]
# # 5. Limited dependent variables: hours worked (Tobit) and arrests (Poisson)
#
# Two outcomes OLS handles badly: **hours** pile up at zero (corner solution), and **arrests** are
# small counts. (Wooldridge Examples 17.2 and 17.3.)

# %%
import numpy as np
import statsmodels.formula.api as smf
import wooldridge as woo

from _setup import *  # noqa: F401,F403
import diagnostics as dg

# %% [markdown]
# ## A. Tobit for annual hours of married women (MROZ)
# 325 of 753 women work zero hours. Question: how do young children affect labor supply — through
# whether women work at all (extensive margin) or how much those who work do (intensive margin)?

# %%
mz = woo.data("mroz")
X = mz[["nwifeinc", "educ", "exper", "age", "kidslt6", "kidsge6"]].assign(expersq=mz["exper"] ** 2)
tb = dg.tobit(mz["hours"], X)

# %% [markdown]
# Reading the margins for **kidslt6** (children under 6):
# - `E[y|x]`: total effect on expected hours across all women.
# - `E[y|uncensored,x]`: effect on hours among women who work.
# - `P(uncensored|x)`: effect on the probability of working at all.
#
# A young child reduces the probability of working by about 25 percentage points. The McDonald-Moffitt
# decomposition (computed observation by observation, then averaged) splits the total effect on
# expected hours between the two margins. The latent coefficient (−894 hours) is not an effect on
# observed hours and should not be reported as one.

# %%
from scipy import stats

me = tb["marginal_effects"].loc["kidslt6"]
print(f"kidslt6: total {me['E[y|x]']:.0f} h, among workers {me['E[y|uncensored,x]']:.0f} h, "
      f"P(work) {me['P(uncensored|x)']:+.3f}")
b, s = tb["table"]["latent_coef"].values, tb["sigma"]
XX = np.column_stack([np.ones(len(X)), X.values])
c = XX @ b / s
P, lam = stats.norm.cdf(c), stats.norm.pdf(c) / stats.norm.cdf(c)
bj = tb["table"].loc["kidslt6", "latent_coef"]
intensive = np.mean(P * bj * (1 - lam * (c + lam)))                   # P(work) × change in hours of workers
extensive = np.mean((XX @ b + s * lam) * stats.norm.pdf(c) * bj / s)  # hours of workers × change in P(work)
print(f"McDonald-Moffitt: {intensive:.0f} h from fewer hours among workers + {extensive:.0f} h from leaving work "
      f"= {intensive + extensive:.0f} h ({extensive / (intensive + extensive):.0%} extensive)")

# %% [markdown]
# ## B. Poisson regression for number of arrests in 1986 (CRIME1)

# %%
c = woo.data("crime1")
f = "narr86 ~ pcnv + avgsen + tottime + ptime86 + qemp86 + inc86 + black + hispan + born60"
pois = smf.poisson(f, c).fit(disp=0)
cd = dg.count_diagnostics(pois)

# %% [markdown]
# Variance exceeds the mean (σ̂² ≈ 1.5), so the default Poisson standard errors are too small. The
# coefficients remain consistent (quasi-MLE) if the mean is right: keep them, fix the errors.

# %%
robust = smf.poisson(f, c).fit(disp=0, cov_type="HC1")
b_pcnv = pois.params["pcnv"]
print(f"pcnv: coef {b_pcnv:.3f}; a 0.1 rise in the conviction probability → "
      f"{100*(1-np.exp(0.1*b_pcnv)):.1f}% fewer expected arrests")
print(f"SE of pcnv — Poisson {pois.bse['pcnv']:.3f}, robust {robust.bse['pcnv']:.3f}, "
      f"quasi-Poisson {cd['quasi_poisson_se']['pcnv']:.3f}")

# %% [markdown]
# **Report.** A child under six lowers expected annual hours by about 530; the effect splits roughly
# evenly between women leaving work (a 25-point drop in participation) and fewer hours among those who
# keep working. The deterrence estimate (a higher past conviction rate is associated with fewer
# arrests) survives the correction for overdispersion, with standard errors about 20% larger than
# Poisson's.
