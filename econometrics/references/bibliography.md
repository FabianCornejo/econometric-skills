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
- Cameron and Trivedi, *Microeconometrics: Methods and Applications* (2005) and *Microeconometrics Using Stata*; Cameron and Trivedi (1990), "Regression-based tests for overdispersion in the Poisson model", *Journal of Econometrics* — the NB1/NB2 auxiliary-regression tests.
- Mundlak (1978), "On the pooling of time series and cross section data", *Econometrica*; Wooldridge (2019), "Correlated random effects models with unbalanced panels", *Journal of Econometrics*.
- Little and Rubin, *Statistical Analysis with Missing Data* (3rd ed., 2019); van Buuren, *Flexible Imputation of Missing Data* (2nd ed., 2018, free online) — mechanisms, multiple imputation; Jones (1996), "Indicator and stratification methods for missing explanatory variables in multiple linear regression", *JASA*.
- Duan (1983), "Smearing estimate: a nonparametric retransformation method", *JASA* — predicting y from a model for log y.
- Oster (2019), "Unobservable selection and coefficient stability", *Journal of Business & Economic Statistics*; Cinelli and Hazlett (2020), "Making sense of sensitivity: extending omitted variable bias", *Journal of the Royal Statistical Society B*.
- Wooldridge, *Econometric Analysis of Cross Section and Panel Data*.
- De Caigny, Coussement and De Bock (2018), "A new hybrid classification algorithm for customer churn prediction based on logistic regression and decision trees", *European Journal of Operational Research* 269(2): 760–772 — the logit leaf model; R package `LLM`. Zeileis, Hothorn and Hornik (2008), "Model-based recursive partitioning", *Journal of Computational and Graphical Statistics* (`partykit::glmtree`); Athey and Imbens (2016), "Recursive partitioning for heterogeneous causal effects", *PNAS* — honest sample splitting.
- Croissant, *Microeconometrics with R* (2024); Adams, *Learning Microeconometrics with R* (2020).
- Lee, McCrary, Moreira and Porter (2022), "Valid t-ratio inference for IV", *American Economic Review* — the F > 104.7 criterion.
- Goodman-Bacon (2021); Callaway and Sant'Anna (2021); Sun and Abraham (2021) — DiD with staggered adoption.

## Causal inference and research design
- Angrist and Pischke, *Mastering 'Metrics* (2014) — the five core designs in accessible form.
- Cunningham, *Causal Inference: The Mixtape* (2021, free online); Huntington-Klein, *The Effect* (2021, free online).
- Imbens and Rubin, *Causal Inference for Statistics, Social, and Biomedical Sciences* (2015) — potential outcomes, matching, weighting.
- Cattaneo, Idrobo and Titiunik, *A Practical Introduction to Regression Discontinuity Designs* (2020/2024); Calonico, Cattaneo and Titiunik (2014); Imbens and Kalyanaraman (2012); McCrary (2008); Gelman and Imbens (2019).
- Abadie, Diamond and Hainmueller (2010); Abadie (2021), "Using synthetic controls", *Journal of Economic Literature*.
- Roth, Sant'Anna, Bilinski and Poe (2023), "What's trending in difference-in-differences?", *Journal of Econometrics*; Rambachan and Roth (2023) — sensitivity to parallel-trends violations.
- Callaway and Sant'Anna (2021) — group-time ATTs, including the anticipation adjustment; de Chaisemartin and D'Haultfœuille (2024, *Review of Economics and Statistics*) and Callaway, Goodman-Bacon and Sant'Anna (2024) — non-binary and continuous treatments.
- Chernozhukov et al. (2018), "Double/debiased machine learning", *Econometrics Journal*.
- Crump, Hotz, Imbens and Mitnik (2009) — trimming for overlap; Austin (2011) — calipers; Oster (2019); Cinelli and Hazlett (2020) — sensitivity to unobserved confounding.
- Kling (2006); Dobbie, Goldin and Yang (2018) — judge/examiner leniency designs.

## Heterogeneous treatment effects and policy learning
- Battocchi, Dillon, Hei, Lewis, Oka, Oprescu and Syrgkanis, *EconML: A Python Package for ML-Based Heterogeneous Treatment Effects Estimation* (Microsoft Research / PyWhy; https://www.pywhy.org/EconML/) — estimator comparison, flowchart for choosing an estimator, validation (DRTester, RScorer), policy trees.
- Chernozhukov, Demirer, Duflo and Fernández-Val (2018, rev. 2023), "Generic machine learning inference on heterogeneous treatment effects in randomized experiments", NBER w24678 — BLP, GATES, CLAN.
- Wager and Athey (2018), "Estimation and inference of heterogeneous treatment effects using random forests", *JASA*; Athey, Tibshirani and Wager (2019), "Generalized random forests", *Annals of Statistics* — causal forests, grf.
- Nie and Wager (2021), "Quasi-oracle estimation of heterogeneous treatment effects", *Biometrika* — R-learner, R-loss.
- Kennedy (2023), "Towards optimal doubly robust estimation of heterogeneous causal effects", *Electronic Journal of Statistics* — DR-learner.
- Künzel, Sekhon, Bickel and Yu (2019), "Metalearners for estimating heterogeneous treatment effects using machine learning", *PNAS* — T-, S- and X-learners.
- Yadlowsky, Fleming, Shah, Brunskill and Wager (2025), "Evaluating treatment prioritization rules via rank-weighted average treatment effects", *JASA* — TOC, AUTOC, Qini.
- Athey and Wager (2021), "Policy learning with observational data", *Econometrica*; Sverdrup, Kanodia, Zhou, Athey and Wager (2020), "policytree: policy learning via doubly robust empirical welfare maximization over trees", *JOSS*.
- Chernozhukov, Cinelli, Newey, Sharma and Syrgkanis (2022), "Long story short: omitted variable bias in causal machine learning", NBER w30302 — sensitivity analysis for DML (EconML's `robustness_value`).

## Estimation, inference and the bootstrap
- Greene, *Econometric Analysis*, chs. 7 (nonlinear least squares), 14 (maximum likelihood); Cameron and Trivedi (2005), chs. 5, 7 and 11 — M-estimation, the LR/Wald/LM trinity, the bootstrap.
- White (1982), "Maximum likelihood estimation of misspecified models", *Econometrica* — sandwich covariance.
- Cameron, Gelbach and Miller (2008), "Bootstrap-based improvements for inference with clustered errors", *REStat*; MacKinnon and Webb (2017), "Wild bootstrap inference for wildly different cluster sizes", *JAE*; Webb (2023), "Reworking wild bootstrap-based inference for clustered errors", *Canadian Journal of Economics*; Roodman, MacKinnon, Nielsen and Webb (2019), "Fast and wild: bootstrap inference in Stata using boottest", *Stata Journal*; MacKinnon, Nielsen and Webb (2023), "Cluster-robust inference: a guide to empirical practice", *Journal of Econometrics*.

## Textbooks with code (used for the Teacher mode and the tests)
- Sargent and Stachurski, *Intermediate Quantitative Economics with Python* (QuantEcon, 2025) — linear algebra, LLN/CLT, Kalman filter, Bayesian AR(1), likelihood-ratio processes; replications of Acemoglu-Johnson-Robinson (2001) by IV and of Treisman (2016) by Poisson MLE.
- Colonescu, *Principles of Econometrics with R* (2016) — companion to Hill, Griffiths and Lim; repeated-sample and Monte Carlo teaching of sampling distributions and CI coverage.
- Dalpiaz, *Applied Statistics with R* — regression taught through simulation; ANOVA, model building, logistic regression.
- Faraway, *Linear Models with R* (Chapman & Hall/CRC; 2nd ed. 2014) — diagnostics, permutation tests, lack of fit, robust regression, shrinkage, model uncertainty, missing data, ANOVA and block designs.
- Tsay, *An Introduction to Analysis of Financial Data with R* (Wiley, 2013) — linear time series, GARCH family, stochastic volatility, high-frequency data, VaR with EVT and quantile regression.
- Ang, *Analyzing Financial Data and Implementing Financial Models Using R* (Springer, 2015) — returns, CAPM and market model, Fama-French, event studies, performance ratios, Markowitz.
- Gallic, *Python for Economists* (2019) — introduction to the language for economists.

## Empirical legal studies and text as data
- Machado (org.), *Pesquisar empiricamente o Direito* (Rede de Estudos Empíricos em Direito, 2017) — empirical legal research in Brazil.
- Epstein and Martin, *An Introduction to Empirical Legal Research* (2014).
- Yin, *Case Study Research* — qualitative designs that complement quantitative ones.
- Grimmer, Roberts and Stewart, *Text as Data* (2022); Gentzkow, Kelly and Taddy (2019), "Text as data", *Journal of Economic Literature*.
- Ash and Hansen (2023), "Text algorithms in economics", *Annual Review of Economics*.
- Egami, Hinck, Stewart and Wei (2023), "Using imperfect surrogates for downstream inference: design-based supervised learning for social science applications of large language models", *NeurIPS* — valid inference with LLM/ML-coded variables; R package `dsl`.
- Angelopoulos, Bates, Fannjiang, Jordan and Zrnic (2023), "Prediction-powered inference", *Science*; Zrnic and Candès (2024), "Active statistical inference", *ICML* — the same correction with sampling probabilities driven by model uncertainty.
- Ludwig, Mullainathan and Rambachan (2025), "Large language models: an applied econometric framework" (NBER w33344; arXiv 2412.07031) — when LLM outputs can be used for prediction vs. estimation, and why a validation sample is needed for the latter.
- Ma, Chen, Wang and Zhang (2025), "Estimating LLM uncertainty with logits" (arXiv 2502.00290) — LogTokU: aleatoric and epistemic uncertainty from top-k logits.
- Priest and Klein (1984) — selection of disputes for litigation.

## Survival analysis
- Kleinbaum and Klein, *Survival Analysis: A Self-Learning Text* (3rd ed., 2012).
- Cameron and Trivedi (2005), chapters 17–19 — duration models in econometrics.
- Grambsch and Therneau (1994) — proportional-hazards test with Schoenfeld residuals.

## Python for data work and machine learning
- McKinney, *Python for Data Analysis* (3rd ed., 2022; free online).
- Géron, *Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow*.
- Mitchell, *Web Scraping with Python* (O'Reilly).
- Scavetta and Angelov, *Python and R for the Modern Data Scientist* (2021).

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
