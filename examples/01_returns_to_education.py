# %% [markdown]
# # 1. Returns to education in a cross-section (WAGE1)
#
# A worked example of the skill's 7-step workflow on Wooldridge's WAGE1 data (526 workers, 1976 CPS).
# **Question:** how much does one more year of education raise hourly wages, and how much should we
# trust that number as a causal effect?

# %%
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import wooldridge as woo

from _setup import *  # noqa: F401,F403  (adds the skill's scripts to the path)
import data_prep as dp
import diagnostics as dg
import reporting as rp
import specification as sp

# %% [markdown]
# ## Step 1 — Frame the question and expected signs
# - y = log hourly wage, so coefficients are approximate percentage effects (log-lin).
# - educ: positive (human capital). exper: positive but concave (add exper²). tenure: positive.
# - Goal: *estimate an effect*. The threat is omitted **ability**, which raises both schooling and
#   wages, so OLS is probably biased **upward**. We will quantify how fragile the estimate is.
#
# ## Step 2 — Data structure: cross-section, continuous y → `ols-regression.md`
#
# ## Step 3 — Audit the data

# %%
df = woo.data("wage1")
cols = ["wage", "educ", "exper", "tenure", "female", "married", "nonwhite"]
dp.audit(df[cols], show=False).head(8)

# %%
miss = dp.missing_report(df, cols, show=True)

# %% [markdown]
# ## Step 4 — Specify and estimate (simplest model first, robust errors by default)

# %%
m1 = smf.ols("np.log(wage) ~ educ", df).fit(cov_type="HC3")
m2 = smf.ols("np.log(wage) ~ educ + exper + I(exper**2) + tenure", df).fit(cov_type="HC3")
m3 = smf.ols("np.log(wage) ~ educ + exper + I(exper**2) + tenure + female + married + nonwhite",
             df).fit(cov_type="HC3")

# %% [markdown]
# ## Step 5 — Diagnose (cross-section: no autocorrelation tests, row order is arbitrary)

# %%
diag = dg.ols_diagnostics(m3)

# %% [markdown]
# **RESET rejects** (p ≈ 0.001): the functional form is wrong, and the workflow says to respecify
# *before* worrying about standard errors. Theory suggests two candidates: the marriage premium
# differs by gender (an interaction), and returns to schooling may rise with the level of
# schooling (credentials, college premium), so add educ² and tenure².

# %%
m4 = smf.ols("np.log(wage) ~ educ + I(educ**2) + exper + I(exper**2) + tenure + I(tenure**2)"
             " + female*married + nonwhite", df).fit(cov_type="HC3")
diag4 = dg.ols_diagnostics(m4)

# %% [markdown]
# RESET no longer rejects — and neither do Breusch-Pagan nor the full White test, which both rejected
# before. The "heteroskedasticity" was largely a symptom of the wrong functional form, which is why
# the workflow checks the form first. Only the simplified White test is borderline (p ≈ 0.05); the
# HC3 errors keep inference valid either way. With educ² in the model the return to schooling depends on the level of schooling:
# ∂log(wage)/∂educ = β₁ + 2β₂·educ. Evaluate it where it matters, with delta-method SEs.

# %%
for e in (10, 12, 16):
    t = m4.t_test(f"educ + {2*e}*I(educ ** 2) = 0")
    b, se = float(np.ravel(t.effect)[0]), float(np.ravel(t.sd)[0])
    print(f"Return at {e} years of schooling: {100*(np.exp(b)-1):.1f}% per extra year "
          f"(95% CI {100*(np.exp(b-1.96*se)-1):.1f}% to {100*(np.exp(b+1.96*se)-1):.1f}%)")

# %% [markdown]
# ## Step 6 — Robustness: specifications side by side

# %%
labels = {"educ": "Years of education", "I(educ ** 2)": "Education²", "female": "Female",
          "married": "Married", "female:married": "Female × married"}
rp.regression_table({"(1)": m1, "(2)": m2, "(3)": m3, "(4)": m4},
                    keep=["educ", "I(educ ** 2)", "female", "married", "female:married"], labels=labels)

# %% [markdown]
# In the linear-in-education columns the average return is stable at about 8% across specifications.
# Stability under observed controls is informative but not proof: what about **unobserved** ability?
#
# ### Sensitivity to an omitted confounder
# These tools need a single coefficient, so they use the linear-in-education version of the final
# specification, whose coefficient approximates the average return.
# - **Oster (2019):** how strongly would unobservables have to drive selection, relative to the
#   observed controls, to make the effect zero?
# - **Cinelli-Hazlett (2020):** how much residual variance of both education and wages would a
#   confounder need to explain?

# %%
m4_lin = smf.ols("np.log(wage) ~ educ + exper + I(exper**2) + tenure + I(tenure**2)"
                 " + female*married + nonwhite", df).fit()
osr = sp.oster(smf.ols("np.log(wage) ~ educ", df).fit(), m4_lin, "educ", show=True)
sens = sp.sensitivity(m4_lin, "educ", show=True)

# %% [markdown]
# ## Step 7 — Interpret and report
#
# Prediction for a specific worker (12 years of schooling, 10 of experience, 2 of tenure, male,
# married, white). The **prediction interval** is for one person; the CI is for the average such
# worker. In levels, report the smearing-corrected mean, not exp(prediction).

# %%
new = pd.DataFrame({"educ": [12], "exper": [10], "tenure": [2], "female": [0], "married": [1], "nonwhite": [0]})
sp.prediction_intervals(m4.model.fit(), new, log_y=True).round(2)

# %% [markdown]
# **Report.** The simple log-wage equation fails RESET; allowing returns to rise with schooling and
# the marriage premium to differ by gender fixes it. Returns to schooling increase with its level —
# compare the estimates at 10, 12 and 16 years above — and average about 8% per year. That average
# is stable across specifications; by Oster's criterion unobservables would need to be dozens of
# times as important as the observed controls to erase it, and a confounder would need to explain
# far more residual variance than any observed covariate does. Ability remains the obvious omitted
# variable and probably biases the estimate upward; example 2 addresses it with instruments. For
# one worker, the wage prediction interval is wide: individual wages vary far more than averages.
