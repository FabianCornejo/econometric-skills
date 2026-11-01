# Empirical finance: betas, cost of equity, factor models, event studies, performance, portfolios

Use for the econometric inputs of valuation and transactions (the beta and the cost of equity in a WACC, comparable-company betas, the market reaction to an M&A, IPO or divestiture announcement), for evaluating a fund or strategy, and for mean-variance portfolios. Code: `scripts/finance_empirics.py`. Sources: Ang (2015), Tsay (2013), Campbell, Lo & MacKinlay (1997), Damodaran on cost of capital. Volatility models are in `volatility.md`, VaR/ES in `risk-var.md`, strategy backtests in `forecasting-backtesting.md`.

## 1. Returns

`returns(prices, kind="simple"|"log", dividends=None, freq=None)`.
- **Total, not price, returns** for betas and performance: use adjusted prices or pass dividends.
- **Simple returns aggregate across assets** (portfolio return = weighted average); **log returns aggregate over time** (sum).
- **Frequency**: resample prices to the period end (`freq="W-FRI"`, `"ME"`) before computing returns, so multi-day returns compound.
- **Brazil**: B3 tickers in Yahoo end in `.SA` (`PETR4.SA`, `^BVSP` for the Ibovespa); the risk-free rate is usually the CDI (BCB SGS series 12, daily rate in % per day; convert to decimals and align dates) or the Selic.

## 2. Beta and the cost of equity

`market_model(r_i, r_m, rf=None)` gives alpha, beta, R², idiosyncratic volatility and Newey-West SEs; with `rf` it is the CAPM regression in excess returns.

| Choice | Common practice | Why it matters |
|---|---|---|
| Market proxy | Ibovespa or IBrX-100 for Brazilian stocks; a global index for a global investor | Beta is relative to the proxy |
| Frequency | Weekly (2 years) or monthly (5 years) | Daily betas of illiquid stocks are biased toward zero (non-synchronous trading); use weekly/monthly, or `dimson_beta` (sum of lead/lag coefficients; compare with the OLS beta) |
| Window | 2–5 years | Long windows mix old capital structures; short windows are noisy |
| Adjustment | Blume or Vasicek | Raw betas mean-revert toward 1 |

1. **Stability first**: `rolling_beta(r_i, r_m, window)`. A trend or a break means older data describe another company. Typical breaks: a merger or spin-off, a new capital structure, a new business line, a **change of control** (privatization of a state-owned company, entry of a strategic shareholder) or a new regulatory regime. Start each comparable's window after its break; when the post-break window is too short, use dummies for the event weeks of the process (approvals, auction, offering), give that comparable less weight, and say so. Months dominated by deal news have returns driven by idiosyncratic events, which bias the beta toward zero.
2. **Precision**: report the beta's CI. A beta of 1.1 with SE 0.3 is consistent with anything from 0.5 to 1.7. That is why practitioners prefer **industry betas from comparables** to a single regression.
3. **Comparables**: estimate each comparable's levered beta, unlever it with its own market D/E (averaged over the estimation window, not at the valuation date) and marginal tax rate, take the median or a quality-weighted mean of the asset betas, and relever at the target's long-run capital structure. `unlever_beta`/`relever_beta` offer `method="hamada"` (fixed debt; with a debt beta this is Conine 1980) and `method="harris_pringle"` (debt rebalanced to a constant D/E, no tax term); use the same method both ways and state it. Highly levered sectors (utilities, sanitation, infrastructure) need a debt beta (often 0.1–0.3): ignoring it understates the asset beta. For Brazilian companies under lucro real the marginal rate is 34% (25% IRPJ + 9% CSLL); a target without taxable profit cannot use the full tax shield.
   - **Few or unstable local comparables**: triangulate with an international peer set or an industry table (e.g. Damodaran's), but keep the CAPM coherent: a beta from US peers belongs in a global CAPM (US risk-free rate, US equity risk premium, country risk premium, inflation differential to convert to BRL); a beta from Brazilian peers belongs in a local CAPM with a Brazilian market premium. Do not mix them.
4. **Adjustment**: `adjusted_beta(beta, "blume")` (0.67 β + 0.33, the Bloomberg convention) or `"vasicek"` (shrinks each beta toward the industry mean in proportion to its imprecision: the statistically grounded choice when you have the beta's SE and the cross-sectional SD of industry betas). In regulated, defensive sectors Blume pulls betas up mechanically; show it as a sensitivity rather than the base case.
   - **Deliver a range**: betas across specifications (proxy, frequency, window, method) and propagate the range to the cost of equity. Premiums for size, illiquidity or company-specific risk go on separate lines of the cost of equity, never inside the beta.
5. **Cost of equity**: k_e = r_f + β × ERP (+ country risk premium when the inputs are in USD). The beta is the econometric part; the risk-free rate, the equity risk premium and the country risk come from sources, not regressions. Never estimate an ERP from a short sample of realized returns.

## 3. Factor models

`factor_model(r_i, factors, rf)` regresses excess returns on factor returns with HAC SEs: Fama-French 3 (Mkt-RF, SMB, HML) and 5 factors (+ RMW, CMA), Carhart momentum (Mom). Data: Kenneth French's data library (US, developed and emerging markets); **NEFIN-USP** for Brazil (Market, SMB, HML, WML, IML and a risk-free rate, daily and monthly). Uses: style analysis of a fund, risk-adjusted performance (alpha), cost of equity with several factors. A significant alpha means abnormal performance relative to those factors, or a missing factor.

## 4. Event studies

`event_study(r, r_m, events, est_win=(-250, -30), evt_win=(-5, 5), model="market", car_windows=((-1, 1),))`.

1. **Define the event date** precisely (first public announcement, including leaks and rumors); day 0 is the first trading day on or after it. Announcements after the close belong to the next day: shift their date before calling `event_study`. A firm with several deals needs one row per deal; pass an `event_id` column (built as firm_date otherwise). Results are indexed by `event_id`, and events without enough history appear in `res["skipped"]`.
2. **Estimation window** before the event window, with a gap (e.g. −250 to −30) so that run-up does not contaminate the normal-return model.
3. **Normal-return model**: market model (default), market-adjusted (no estimation; for IPOs and thin histories) or constant mean.
4. **Abnormal returns** AR = r − expected; **CAR** over the event window; **CAAR** across events. Use simple returns for CAARs (the cross-sectional mean of simple returns is a portfolio return; that of log returns is not), or state the convention; the difference is small for short windows.
5. **Tests** (all reported):

| Test | Assumption | Use |
|---|---|---|
| Cross-sectional t | Independent CARs across events | Robust to event-induced variance; low power with few events |
| Patell Z | AR variance unchanged by the event | Over-rejects when the event raises volatility (in the skill's simulations, ~31% instead of 5% with day-0 variance ×4) |
| BMP | Standardized CARs, cross-sectional variance | **Default for announcements** (stayed at ~4% in the same simulations) |
| Corrado rank | Nonparametric | Non-normal returns, small samples; day-0 test |

6. **Cross-section of CARs**: `car_regression(res, events, "CAR ~ np.log(rel_size) + C(payment) + listed_target", window="(-1, 1)", cluster="firm")` matches CARs to deal characteristics by `event_id`, drops skipped events and clusters on the rows actually used (by acquirer when firms repeat). A CAAR near zero can hide opposite effects (cash vs stock payment).
7. **Pitfalls**: event clustering (same date, same industry) correlates ARs and inflates every test: use a portfolio of the affected firms or calendar-time regressions; thin trading (zero returns) biases betas; confounding news on the same day (earnings) must be checked by reading the news; long-horizon BHARs need matched benchmarks and are another design.

## 5. Performance

`performance_ratios(r_p, r_b, rf, periods_per_year)`: Sharpe with the Lo (2002) SE, Sortino, Treynor, information ratio, Jensen's alpha, Roy safety-first. Sharpe ratios are noisy: with 3 years of monthly data, an annualized Sharpe of 0.5 has a standard error of about 0.58 (Lo 2002), so two funds' Sharpes rarely differ significantly. For strategies, drawdowns and costs see `backtest.performance`; for many strategies tried, the deflated Sharpe ratio (`forecasting-backtesting.md` §9).

## 6. Mean-variance portfolios

`efficient_frontier(mu, cov, rf, short)` gives the global minimum-variance and tangency portfolios and frontier points (closed form with short sales; quadratic programming long-only). `shrink_cov(returns)` gives the Ledoit-Wolf covariance. Mean-variance weights are extremely sensitive to the expected returns: small changes in mu flip the weights. Prefer the minimum-variance portfolio (no means needed), long-only or bounded weights, shrunk covariances, and treat the tangency portfolio as illustrative unless the expected returns come from a model you trust (e.g. implied returns, Black-Litterman).

## 7. Reporting checklist

- [ ] Return definition (total vs price, simple vs log), frequency, window, market proxy and risk-free rate stated.
- [ ] Beta with CI, rolling-beta stability check, adjustment used; comparables' D/E and tax rates for unlevering.
- [ ] Event study: event-date source, windows, normal-return model, CAAR with the four tests, number of events and any clustering.
- [ ] Performance ratios annualized consistently, with the Sharpe SE.
- [ ] Portfolio weights with the covariance estimator and constraints used.
