# %% [markdown]
# # 4. Time series: does a tax exemption raise fertility? (FERTIL3)
#
# **Question:** what is the long-run effect of the personal tax exemption (pe) on the U.S. general
# fertility rate (gfr), 1913–1984? Effects plausibly take time → distributed lag.
# (Wooldridge Examples 10.4 and 11.8.)

# %%
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import wooldridge as woo

from _setup import *  # noqa: F401,F403
import diagnostics as dg
import specification as sp

# %% [markdown]
# ## Steps 1–3 — Expected sign positive (a subsidy to children). Annual series, 72 years.

# %%
fz = woo.data("fertil3")
fz.index = pd.Index(fz["year"].astype(int), name="year")
for k in (1, 2):
    fz[f"pe_{k}"] = fz["pe"].shift(k)

# %% [markdown]
# ## Step 5 first: are the series stationary?
# Regressing trending series in levels risks a spurious regression.

# %%
st = dg.stationarity_many(fz[["gfr", "pe"]])

# %% [markdown]
# Both series look integrated of order one, so the levels regression of Example 10.4 is at risk of
# being spurious. We show it, then estimate in first differences (Example 11.8).

# %%
lev = smf.ols("gfr ~ pe + pe_1 + pe_2 + ww2 + pill", fz).fit()
lr_lev = sp.long_run_effect(lev, ["pe", "pe_1", "pe_2"], show=True)
_ = dg.ols_diagnostics(lev, time_series=True)

# %% [markdown]
# R² above Durbin-Watson and strong residual autocorrelation: the warning signs of a spurious
# regression. Now in first differences, with HAC errors.

# %%
for c in ["gfr", "pe", "pe_1", "pe_2"]:
    fz[f"d_{c}"] = fz[c].diff()
nw = dg.newey_west_lags(len(fz))
dif = smf.ols("d_gfr ~ d_pe + d_pe_1 + d_pe_2", fz).fit(cov_type="HAC", cov_kwds={"maxlags": nw})
print(dif.summary().tables[1])
lr_dif = sp.long_run_effect(dif, ["d_pe", "d_pe_1", "d_pe_2"], show=True)
_ = dg.ols_diagnostics(dif, time_series=True)

# %% [markdown]
# ## Step 7 — Report
# In levels, the long-run propensity is about 0.10 births per 1,000 women per dollar of exemption,
# but the regression shows the signature of spuriousness. In first differences, the contemporaneous
# and one-year effects are slightly negative and indistinguishable from zero; only the two-year lag is
# clearly positive. The long-run effect
# is estimated imprecisely, with a confidence interval that includes zero. Conclusion: the evidence
# for a fertility response is fragile and depends on treating trending series correctly. Individual
# lag coefficients are imprecise because the lags are collinear, so report the long-run effect and
# its interval.
