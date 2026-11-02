# %% [markdown]
# # 6. DiD with covariates: minimum wage and teen employment (staggered adoption)
#
# **Question.** Did state minimum-wage increases reduce teen employment in the counties of those
# states? Counties were treated in 2004, 2006 or 2007, or never, between 2003 and 2007.
#
# **Data.** `mpdta`, the county panel of Callaway & Sant'Anna (2021): 500 counties, log teen
# employment (`lemp`), log population in 2000 (`lpop`, time-invariant) and the first treated
# year (`first.treat`, 0 = never). It ships with the R package `did` (GPL); to keep this MIT
# repository clean of it, the example downloads the CSV copy distributed with the MIT-licensed
# Python port `csdid` once and caches it.
#
# **Hypothesis, written before estimating.** A higher minimum wage lowers teen employment (a
# negative ATT), growing with exposure. Larger counties may have different employment trends
# (urban labour markets), so parallel trends are more credible **conditional on population**.

# %%
import io
import contextlib
import urllib.request
from pathlib import Path

import numpy as np

from _setup import *  # noqa: F401,F403
import causal as ca
import reporting as rp

URL = "https://raw.githubusercontent.com/d2cml-ai/csdid/main/data/mpdta.csv"
cache = Path(".cache") / "mpdta.csv"          # examples run from the examples/ folder
if not cache.exists():
    cache.parent.mkdir(exist_ok=True)
    urllib.request.urlretrieve(URL, cache)
d = pd.read_csv(cache)
print(d.shape, "| counties by first treated year:", d.groupby("first.treat")["countyreal"].nunique().to_dict())

# %% [markdown]
# ## Step 1. The unconditional benchmark
# Staggered adoption: Callaway-Sant'Anna with never-treated controls, unit bootstrap.

# %%
r0 = ca.att_gt(d, "countyreal", "year", "lemp", "first.treat", n_boot=199)

# %% [markdown]
# The overall ATT is about −0.040 (−4% teen employment), the number in Callaway & Sant'Anna's
# application. The pre-period cell e = −3 is positive and its CI just excludes zero: the
# unconditional trends are not perfectly parallel, a reason to condition on covariates.
#
# ## Step 2. Do covariate effects differ across groups or years? (Karim & Webb's CCC)
# Before adjusting for population, check that its effect on employment is the same in every
# cohort and year among **untreated** county-years. The groups are the adoption cohorts (several
# counties per cell, unlike `countyreal`), with SEs clustered by county and wild-bootstrap
# p-values.

# %%
cc = ca.ccc_test(d, "lemp", ["lpop"], group="first.treat", time="year", gvar="first.treat",
                 cluster="countyreal", n_boot=199)

# %% [markdown]
# Stage 1 does not reject equal slopes across the 13 untreated cohort-year cells, so the
# common-covariate assumption is not contradicted: a covariate-adjusted Callaway-Sant'Anna is
# appropriate (with group-specific slopes, intersection DiD would be needed instead).
#
# ## Step 3. Callaway-Sant'Anna with the covariate in levels, doubly robust
# `lpop` is time-invariant: a TWFE regression could not use it at all (unit fixed effects absorb
# it). In each group-time 2×2, `att_gt` compares treated and comparison counties of similar
# population, through an outcome regression and a propensity score.

# %%
r1 = ca.att_gt(d, "countyreal", "year", "lemp", "first.treat", n_boot=199, covariates=["lpop"])

# %% [markdown]
# The adjusted ATT is about −0.042, slightly larger in magnitude, with the effect growing to
# −10% to −14% two to three years after adoption. The pre-period cell at e = −3 shrinks and its
# CI now includes zero. The built-in independent recomputation (doubly robust, outcome
# regression and IPW) agrees to the third decimal, so the two nuisance models tell the same
# story.
#
# ## Step 4. Verify against an independent implementation
# The same estimator in another code base: the `csdid` port of the R `did`/`DRDID` packages,
# when installed. `reporting.cross_check` with no SE demands agreement to numerical precision.

# %%
try:
    from csdid.att_gt import ATTgt
    with contextlib.redirect_stdout(io.StringIO()):
        a = ATTgt(yname="lemp", gname="first.treat", idname="countyreal", tname="year",
                  xformla="lemp~lpop", data=d).fit(est_method="dr")
    cs = pd.DataFrame({"g": a.results["group"], "t": a.results["year"], "csdid": a.results["att"]})
    both = r1["gt"].reset_index().merge(cs, on=["g", "t"])
    worst = float(np.max(np.abs(both["att"] - both["csdid"])))
    print(f"Largest cell difference vs csdid: {worst:.2e} over {len(both)} group-time cells")
    rp.cross_check({"this skill (cell 2004/2007)": both.query("g == 2004 and t == 2007")["att"].iloc[0],
                    "csdid (cell 2004/2007)": both.query("g == 2004 and t == 2007")["csdid"].iloc[0]},
                   label="ATT(2004, 2007)")
except ImportError:
    print("csdid not installed: pip install csdid to run the external check")

# %% [markdown]
# ## Step 5. What would a TWFE regression with the covariate have said?

# %%
tw = ca.twfe_covariate_diagnostic(d, "countyreal", "year", "lemp", "first.treat", ["lpop"], n_boot=199)

# %% [markdown]
# Three red flags: `lpop` drops out of TWFE entirely; with staggered adoption some treated
# county-years get **negative** weight (already-treated counties serve as controls); and the
# level of population predicts untreated employment trends at the 10% level, the information
# TWFE throws away. Here the TWFE number happens to be close (−0.037), but it rests on
# assumptions the data do not support; the estimate to report is Callaway-Sant'Anna with the
# covariate.
#
# ## Step 6. Is the result driven by the exact dates?

# %%
ts = ca.did_timing_sensitivity(d, "countyreal", "year", "lemp", "first.treat", max_shift=1, n_boot=99,
                               covariates=["lpop"])

# %% [markdown]
# Shifting the dates moves the ATT mechanically, as expected. The verdict is the date-robust
# estimate, which assumes each true date lies within one year of the recorded one: it stays
# negative with a CI that excludes zero, so the conclusion survives date errors of ±1 year.
#
# ## Step 7. Report, with a run manifest
#
# *Minimum-wage increases reduced teen employment in treated counties by about 4% on average
# over 2004–2007 (doubly robust Callaway-Sant'Anna conditional on county population; never-
# treated counties as controls; county bootstrap), with effects growing to 10–14% after two to
# three years. Covariate effects are common across cohorts and years (CCC test not rejected).
# The estimate matches an independent implementation cell by cell. Limits: parallel trends
# conditional on population is an assumption; one pre-period cell is borderline; county-level
# spillovers are ignored; the result's sensitivity to the exact dates is shown above.*

# %%
man = rp.run_manifest(data={"mpdta": d}, files=[cache], seed=0,
                      results={"ATT unconditional": r0["overall"], "ATT conditional on lpop (DR)": r1["overall"]})
