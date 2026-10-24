# Reference bibliography by topic

Use to go deeper on a topic beyond this skill: proofs, worked examples, test variants. If the user has their own material (books, lecture notes, course slides) in a folder or connector available in the session, prefer consulting it.

## General econometrics (regression, inference, diagnostics)
- Wooldridge, *Introductory Econometrics: A Modern Approach* — the standard teaching reference; heteroskedasticity (ch. 8), specification (ch. 9), IV (ch. 15), simultaneous equations (ch. 16), limited dependent variables (ch. 17).
- Greene, *Econometric Analysis* — comprehensive reference for estimators and tests.
- Hansen, *Econometrics* (free author version) — robust HC0–HC3 errors, clustering, influential observations, asymptotic theory.
- Heij, de Boer, Franses, Kloek and van Dijk, *Econometric Methods with Applications in Business and Economics* (2004) — the basis of the Erasmus *Econometrics: Methods and Applications* course (Coursera).
- Gujarati and Porter, *Basic Econometrics*.
- Johnston and DiNardo, *Econometric Methods*.

## Microeconometrics and causal inference
- Angrist and Pischke, *Mostly Harmless Econometrics* (2008) — IV, LIML, bad controls, difference-in-differences, few clusters.
- Cameron and Trivedi, *Microeconometrics: Methods and Applications* (2005) and *Microeconometrics Using Stata*.
- Wooldridge, *Econometric Analysis of Cross Section and Panel Data*.
- Croissant, *Microeconometrics with R* (2024); Adams, *Learning Microeconometrics with R* (2020).
- Lee, McCrary, Moreira and Porter (2022), "Valid t-ratio inference for IV", *American Economic Review* — the F > 104.7 criterion.
- Goodman-Bacon (2021); Callaway and Sant'Anna (2021); Sun and Abraham (2021) — DiD with staggered adoption.

## Time series and forecasting
- Hamilton, *Time Series Analysis* (1994).
- Enders, *Applied Econometric Time Series*.
- Stock, "Unit Roots, Structural Breaks and Trends", *Handbook of Econometrics*, vol. 4, ch. 46 — power of unit-root tests, DF-GLS, breaks.
- Kilian (2011), "Structural Vector Autoregressions" — identification by short-run, long-run and sign restrictions.
- Kilian and Lütkepohl, *Structural Vector Autoregressive Analysis* (2017); Lütkepohl on bootstrapping impulse responses.
- Cochrane, *Time Series for Macroeconomics and Finance* (lecture notes, 1997).
- Makridakis and Hibon (2000), "The M3-Competition", *International Journal of Forecasting*.
- Petropoulos et al. (2022), "Forecasting: theory and practice", *International Journal of Forecasting* — a review of methods and forecast evaluation.
- Hyndman and Athanasopoulos, *Forecasting: Principles and Practice* (free online).
- Hamilton (2018), "Why you should never use the Hodrick-Prescott filter"; Phillips and Jin (2015) on the HP filter.

## Demand, prices and industrial organization
- Huse and Salvo (2006), "Estimação e identificação de demanda e de oferta" (Ipea/Anpec/SDE, in Portuguese) — instruments for price, retail data, logit/BLP, AIDS.
- Deaton and Muellbauer (1980), "An Almost Ideal Demand System".
- Berry, Levinsohn and Pakes (1995); Nevo (2000) — demand for differentiated products.
- Marioni et al. (2022), *Estimating Food and Drink Demand Elasticities* (Birkbeck/Defra) — AIDS, unit-price endogeneity, zeros.
- Aguirregabiria, *Empirical Industrial Organization* (2018).

## Volatility
- Engle (1982); Bollerslev (1986); Glosten, Jagannathan and Runkle (1993); Nelson (1991).
- Andersen, Bollerslev, Christoffersen and Diebold, "Volatility and Correlation Forecasting", *Handbook of Economic Forecasting*.

## Financial econometrics, risk and backtesting
- Tsay, *Analysis of Financial Time Series* (3rd ed., 2010) — returns, GARCH, VaR, state-space models, Markov switching.
- Christoffersen, *Elements of Financial Risk Management* (2nd ed., 2012) — VaR/ES, filtered historical simulation, backtesting, copulas.
- McNeil, Frey and Embrechts, *Quantitative Risk Management* (2015) — ES, extreme values, copulas, tail dependence.
- J.P. Morgan/Reuters (1996), *RiskMetrics — Technical Document* — EWMA, λ = 0.94.
- Kupiec (1995) and Christoffersen (1998), on VaR backtests; Basel Committee (1996) traffic-light framework.
- Patton (2006), "Modelling asymmetric exchange rate dependence", *International Economic Review* — copula-GARCH.
- Joe, *Dependence Modeling with Copulas* (2014); Nelsen, *An Introduction to Copulas* (2006).
- Kraskov, Stögbauer and Grassberger (2004) — kNN mutual information; Székely, Rizzo and Bakirov (2007) — distance correlation.
- Durbin and Koopman, *Time Series Analysis by State Space Methods* (2nd ed., 2012); Harvey, *Forecasting, Structural Time Series Models and the Kalman Filter* (1989).
- Chan, *Algorithmic Trading* (2013) — Kalman-filter hedge ratios and pairs trading.
- Hamilton (1989), "A new approach to the economic analysis of nonstationary time series and the business cycle", *Econometrica* — Markov switching.
- Moskowitz, Ooi and Pedersen (2012), "Time series momentum", *Journal of Financial Economics*.
- Pesaran and Timmermann (1992) — directional accuracy test; Clark and West (2007) — nested forecast comparison; Campbell and Thompson (2008) — out-of-sample R² for returns.
- Bailey and López de Prado (2014), "The deflated Sharpe ratio"; Harvey, Liu and Zhu (2016), "... and the cross-section of expected returns" — multiple testing in backtests.
- Hyndman and Khandakar (2008), "Automatic time series forecasting: the forecast package for R", *Journal of Statistical Software* — the auto-ARIMA algorithm.

## Official data
- FRED, World Bank, IMF, OECD, BIS — international macro data.
- Banco Central do Brasil (SGS, Focus survey via the Olinda API, Inflation Reports with applied VAR boxes); IBGE (SIDRA); Ipeadata — Brazilian data.
