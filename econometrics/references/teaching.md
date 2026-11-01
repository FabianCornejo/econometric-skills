# Teacher mode

Use when the user wants to understand a method, prepare a class or an exam, check their reasoning, or asks "why" a test or estimator exists. The goal is understanding that transfers to new problems, not a recipe. The same scripts and the same rules as in Consultant mode apply: a class never shows numbers the code did not compute.

## 1. Choose the level

| Level | Reader | Math | Typical references |
|---|---|---|---|
| Basic | First contact; manager or student | Words, pictures, one equation | Dalpiaz, *Applied Statistics with R*; Colonescu, *Principles of Econometrics with R* (Hill, Griffiths & Lim); Angrist & Pischke, *Mastering 'Metrics* |
| Intermediate | Graduate course, practitioner | Matrix notation, assumptions and proofs sketched | Wooldridge, *Introductory Econometrics*; Faraway, *Linear Models with R*; Huntington-Klein, *The Effect* |
| Advanced | Researcher | Asymptotics, influence functions, semiparametric efficiency | Hansen, *Econometrics*; Wooldridge, *Cross Section and Panel Data*; Chernozhukov et al. (2018); EconML documentation |

Ask the level only if it is not clear from the question; otherwise infer it and say which level you chose.

## 2. The method card

Every method is explained with the same eleven blocks, in this order. Skip a block only if it is truly empty for that method.

1. **Intuition**: the problem the method solves, in two or three sentences and one everyday example.
2. **Model**: the equation, every symbol defined.
3. **Assumptions**: numbered, each with what goes wrong if it fails.
4. **Estimator**: the formula (display LaTeX) and what it computes in words.
5. **Short derivation**: the two or three steps that show why the estimator works under the assumptions (e.g. OLS unbiasedness from E[u|X] = 0; the IV estimator as a ratio of two reduced forms; the DR score's double robustness).
6. **Simulation**: a small Monte Carlo that shows the method working and then failing when one assumption is broken (§3). Seeing the bias appear is what makes the assumption stick.
7. **Code**: the skill's function on real or simulated data, with the output. The code itself appears in the answer (or is delivered as a downloadable file): the student must be able to rerun the simulation and change its parameters. Never cite a path inside the sandbox.
8. **Diagnostics**: the tests that check the assumptions that can be checked, and what to do when they fail.
9. **Interpretation**: one sentence for each number that matters, with units and the interval.
10. **Pitfalls**: the three most common mistakes.
11. **Exercise**: one problem with data or a scenario, and **always** its complete worked solution below a separator ("Solution"). A closing comprehension question does not replace the solved exercise.

## 3. Simulations that teach

Write them in a few lines of numpy with a fixed seed; repeat 500–1000 times; show the distribution of the estimate against the truth (histogram or a table of mean, bias, SD, coverage).

| Lesson | Simulation | What the student sees |
|---|---|---|
| Unbiasedness and sampling variation | OLS on repeated samples (Colonescu 2.5) | Estimates spread around the true beta; spread falls with n |
| CI coverage | 95% CIs over 1,000 samples (Colonescu 3.4) | About 950 contain the truth; with heteroskedasticity and classical SEs, fewer |
| Omitted variable bias | Drop a confounder correlated with x | Bias = beta_z · delta, with the predicted sign |
| Bad control | Control for a variable caused by the treatment | The effect shrinks or flips |
| Measurement error | Add noise to x | Attenuation toward zero |
| Spurious regression | Two independent random walks | High R², "significant" t, DW near zero |
| Weak instruments | IV with a weak first stage | 2SLS biased toward OLS, CIs that do not cover |
| Few clusters | Clustered SEs with 8 clusters | Over-rejection; `estimation.wild_cluster_bootstrap` fixes it |
| Staggered DiD | TWFE with growing effects and staggered dates | TWFE wrong (even the sign); `causal.att_gt` right |
| Heterogeneity | Effect larger at both extremes of x | A linear CATE sees nothing; a forest finds it (`hte`) |
| Look-ahead | Scale with full-sample statistics in a backtest | Spectacular Sharpe that vanishes out of sample |

## 4. Curriculum order

For a course or a self-study plan, follow the order of Hill, Griffiths & Lim (as in Colonescu) and extend it:
1. Simple regression → inference → prediction and functional form.
2. Multiple regression → further inference (F tests, omitted variables, collinearity).
3. Dummy variables, interactions, Chow test, linear probability model, DiD.
4. Heteroskedasticity → time series with stationary variables (distributed lags, serial correlation).
5. Random regressors and IV → simultaneous equations.
6. Nonstationary series, cointegration, VEC/VAR → ARCH/GARCH.
7. Panel data → qualitative and limited dependent variables.
8. Extensions: modern causal designs (`causal-inference.md`), heterogeneous effects (`heterogeneous-effects.md`), maximum likelihood and the bootstrap (`estimation-theory.md`), forecasting and risk.

## 5. Style

- Build from what the student already knows; one new idea per step.
- Ask the student to predict the result of a simulation before running it.
- After the explanation, a short check of understanding (two or three questions); offer a quiz when the user wants practice.
- Use the user's language; keep the standard technical terms in English in parentheses when they are the ones used in the literature.
