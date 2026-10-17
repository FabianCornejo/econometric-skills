# Demand, prices and elasticities

Read when the question is "how much do sales change if the price changes", a promotion's effect, cannibalization between products, cost pass-through or pricing. It is the most common econometric application in retail and restaurants — and the one that suffers most from endogeneity.

## Contents
1. The identification problem
2. Demand specifications
3. Instruments for price
4. Retail data: practical pitfalls
5. Zeros, promotions and dynamics
6. From coefficient to decision
7. Checklist

## 1. The identification problem

Observed prices and quantities are equilibrium points. If the firm raises prices *because* it expects high demand (holidays, a wealthier neighborhood, a trending product), price correlates with the demand shock in the error. Typical consequence (Huse and Salvo, 2006): **OLS gives an elasticity less negative than the true one** — sometimes even positive ("upward-sloping demand").

Direction of bias: plim γ̂ = γ + Cov(p, ε)/Var(p). If demand shocks raise prices, Cov > 0 and the estimated elasticity is pushed toward zero. Knowing the direction already helps: an OLS estimate of −0.8 suggests the true value is more elastic.

To identify the demand curve you need price variation that **shifts supply** (costs) without shifting demand.

Order condition (simultaneous-equation systems): an equation is identified if the number of exogenous variables excluded from it ≥ number of included endogenous variables − 1. The rank condition is necessary and sufficient. In practice: the demand equation needs at least one supply shifter excluded from it.

## 2. Specifications

| Approach | When | Note |
|---|---|---|
| Log-log with controls | One product or an aggregate category | Direct elasticity (the coefficient on ln p); simplest and most common |
| Log-log with substitute prices | Few related products | Cross elasticities: > 0 substitutes, < 0 complements |
| AIDS / QUAIDS (Deaton-Muellbauer) | System of categories with budget shares | Imposes theoretical restrictions (adding-up, homogeneity, symmetry); gives compensated and uncompensated elasticities |
| Multi-stage budgeting (Hausman) | Many differentiated products | Category → segment → brand; reduces the number of parameters |
| Logit / nested logit / BLP (discrete choice) | Many differentiated products, market-share data | Requires defining market size; substitution patterns through characteristics space |

Minimum controls in a retail log-log demand: seasonality (month, day of week), holidays/events, a trend or time effects, local income/activity (or store and time fixed effects in a panel), substitute prices, promotion/advertising, distribution (item availability).

**Uncompensated** (Marshallian) elasticity includes the income effect; **compensated** (Hicksian) only substitution. For pricing decisions on an item with a small budget share, the two nearly coincide.

## 3. Instruments for price

| Class | Examples | Identifying assumption | Risk |
|---|---|---|---|
| Supply shifters (costs) | Input prices (fish, rice, energy), exchange rate for imports, sector wages, freight | Costs uncorrelated with demand shocks | Little variation; different frequency from prices |
| Prices of the same product in other regions/stores (Hausman) | Average price in the other stores | Demand shocks independent across regions; common cost | National advertising/events violate the assumption |
| Lagged prices | p_{t−1}, Δp_{t−1} | Demand shocks not very persistent | Fails if demand has inertia |
| Rival product characteristics (BLP) | Sum of competing products' characteristics | Exogenous characteristics | Best suited to discrete choice |
| Tax or regulatory changes | Sales-tax changes, price floors/caps | Change not driven by demand | Anticipation by agents |

Always: a strong first-stage F (see `panel-iv.md`), an overidentification test when there are more instruments than endogenous regressors, and an OLS vs. IV comparison (Hausman). If IV moves the elasticity in the expected direction (more negative), that is a coherent sign.

When OLS may be acceptable: prices set in advance and not adjusted to demand shocks the firm observes (e.g., a menu fixed months ahead, promotions planned on a rigid calendar), with good seasonality and event controls. Even then, document the assumption.

A stronger alternative to IV: **experimental variation** (A/B price tests across stores or periods) or quasi-experimental (difference-in-differences when only some stores change prices).

## 4. Retail data: practical pitfalls

- **Average unit price (revenue ÷ quantity)** mixes list price, promotions and pack mix; with "buy 2, pay for 1" promotions, the buyer changes the unit price, creating a mechanical negative correlation between price and quantity. Prefer the list/menu price, or compute the price at an aggregate level (store-week, national) to reduce this bias, as in the Birkbeck (2022) study.
- **Aggregation level**: the elasticity of a single-serve pack in a convenience store ≠ that of a family pack in a supermarket. Aggregate where price sensitivity is homogeneous and test robustness at other levels. Very disaggregated data can be noisy; weekly data can be strongly autocorrelated (aggregating to monthly/quarterly is an option).
- **Availability (stockouts)**: zero sales due to lack of stock are not zero demand; exclude or model them.
- **Unobserved coupons and discounts**: the net price paid differs from the list price.
- **Menu or mix changes** create breaks; use dummies or subsamples.
- **Parameter stability**: over long windows (> 3–4 years) or in transforming sectors (delivery), preferences change; include time effects or test for breaks.
- **Delivery vs. dine-in**: different channels, different elasticities; estimate separately or with interactions.

## 5. Zeros, promotions and dynamics

- **Zeros**: buyers who never buy, who buy intermittently because of inventory, or who only buy on discount. There is no elasticity "from zero"; use two-part models, PPML, or model at an aggregate level (see `limited-dependent.md`).
- **Promotions**: generate spikes followed by dips (purchase acceleration/stockpiling). The net promotion effect = spike − subsequent dip − cannibalization of other items. Include lags and leads of the promotion dummy.
- **Dynamics**: habit and inventory make short-run elasticity differ from long-run. In an ADL model: long run = Σβ_price / (1 − Σφ_lagged y).
- **Cost pass-through**: regressing Δln p on Δln cost (with lags) measures how much and how fast cost becomes price; useful for planning price adjustments.

## 6. From coefficient to decision

- Revenue: with elasticity ε, ΔR/R ≈ (1 + ε)·Δp/p. If |ε| < 1 (inelastic), raising the price raises revenue.
- Margin: the profit-maximizing price (Lerner) satisfies (p − c)/p = 1/|ε| for a firm with market power. If the current margin is above 1/|ε|, the price is likely too high, and vice versa.
- Report **ranges**, not points: "elasticity between −1.1 and −1.6 (95% CI)", and simulate the revenue and margin impact at the extremes.
- Elasticities hold for **small** changes within the observed price range; extrapolating to 30% increases is speculative.

## 7. Checklist

1. Unit of analysis and aggregation level defined and justified.
2. Price measured so the buyer cannot change it (list/menu, or aggregate).
3. Seasonality, events, promotions and availability controlled.
4. Source of price variation explicitly discussed: costs, experiment, policy change — or assumed exogenous with a justification.
5. OLS and IV (or quasi-experiment) compared; first-stage F reported.
6. Own and cross elasticities with intervals; short and long run if there are dynamics.
7. Robustness: another aggregation level, another window, with/without the atypical period.
8. Translation into revenue and margin with uncertainty made explicit.
