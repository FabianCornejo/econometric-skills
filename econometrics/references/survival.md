# Survival (duration, time-to-event) analysis

Use when the outcome is **how long until something happens**: time to a court decision or settlement, tenure until a civil servant leaves or is promoted, time to default or prepayment, customer churn, firm survival, time to re-offending. Code: `scripts/survival.py` (statsmodels; `lifelines` optional).

## 1. Why not OLS or logit

- **Censoring**: spells still running at the end of the observation window (cases not yet decided, employees still employed) only tell us that the duration exceeds the observed time. Dropping them or using the censoring date as an end date biases durations downward.
- **Time-varying risk**: the chance of the event changes with elapsed time; survival models describe the hazard directly.
- A logit on "event within 10 years" throws away timing and mishandles units observed for less than 10 years. It is acceptable only when everyone is observed for the full window.

Data layout: one row per spell with `duration` (time from origin to event or censoring), `event` (1 = event observed, 0 = censored) and covariates measured at the origin. Define the **origin** (entry into the state) and the **clock** (days, months) explicitly; spells that started before the data window (left truncation) need entry times (`entry=` in statsmodels `PHReg`).

## 2. Describe: Kaplan-Meier and log-rank

```python
km = kaplan_meier(df, "duration", "event", group="court", times=[180, 365, 730])
logrank(df, "duration", "event", "court")
```

Reports, per group, the number of spells and events, the censored share, the median survival with its CI, and S(t) at chosen times. Plot the curves (`km["curves"]`). If more than half the spells are censored, the median may be undefined: report S(t) at meaningful horizons instead.

## 3. Model: Cox proportional hazards

h(t | x) = h₀(t)·exp(x′β). The baseline hazard h₀(t) is left unspecified; β is estimated by partial likelihood.

```python
res, hr = cox(df, "age + C(sex) + C(education) + C(region)", "duration", "event",
              cluster="court", strata=None, ties="efron")
```

- Report **hazard ratios** exp(β) with CIs. HR = 1.25: the instantaneous event rate is 25% higher at every point in time. It is not "25% shorter duration".
- Direction depends on the event: for "decision", HR > 1 means faster resolution; for "exit from public service", HR > 1 means shorter tenure.
- No intercept: the formula needs no `0 +` (harmless, but unnecessary).
- Categorical variables: `C(var, Treatment('reference'))` to choose a meaningful reference category.
- **Clustered (Lin-Wei) SEs** when spells share a court, firm or judge; robust SEs by default.
- **Ties**: Efron is the default and more accurate than Breslow with many tied event times (daily or monthly data).
- **Stratification** (`strata="state"`): separate baseline hazards by stratum, no coefficient for it. Use for variables that violate proportionality and are only controls.

## 4. Check proportional hazards

```python
ph_test(res, df, "duration", "event")      # Schoenfeld residuals vs rank of time
```

A significant correlation means that variable's HR changes over time (e.g. an advantage that fades). Remedies: stratify on it, interact it with a function of time (split the episodes at chosen times with `person_period` and estimate time-specific effects), or report effects by period. With very large samples, tiny departures become significant: judge the magnitude by plotting the residuals over time.

## 5. Alternatives

| Model | When | Tool |
|---|---|---|
| Discrete-time hazard (logit/cloglog on person-period data) | Durations measured in coarse intervals (months, years); time-varying covariates; easy to explain | `person_period` + `smf.logit("y ~ C(period) + x", pp)` |
| Parametric AFT (Weibull, log-normal, log-logistic) | Need predicted durations or extrapolation; effects as time ratios | `lifelines.WeibullAFTFitter` |
| Competing risks (cumulative incidence, Fine-Gray) | Several mutually exclusive ends: settlement vs judgment vs dismissal | `lifelines.AalenJohansenFitter`; cause-specific Cox for each end, treating others as censored |
| Frailty / shared frailty | Unobserved heterogeneity shared within a group | Cluster-robust Cox as a first step |
| Time-varying covariates | Covariates that change during the spell (judge's caseload, interest rate) | Counting-process (start, stop] format; `lifelines.CoxTimeVaryingFitter` |

Competing risks matter often in legal data: a case "ends" by settlement, dismissal or judgment. Treating dismissal as censoring when estimating time to judgment answers "time to judgment in a world without dismissals", which is rarely the question.

## 6. Pitfalls

- Using only completed spells (survivorship bias in reverse: short spells over-represented).
- Covariates measured after the origin that are consequences of the duration (e.g. number of hearings).
- Immortal-time bias: classifying units by something that happens during follow-up (e.g. "judges who were promoted") and counting their time before that event as exposed.
- Reading HRs as effects on mean duration.
- Ignoring competing events.
- Calendar-time changes (procedural reforms, digital filing) confounded with duration: add cohort or filing-year fixed effects or stratify.
