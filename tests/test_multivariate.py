"""Tests of multivariate.py by module (rule T4, since v2.7.0).

v2.7.4: the two VAR interpretation rules of interpretation-rules.md ("Rules added in v2.7.4"), checked against a
population reference that does not use the skill, then against what multivariate.var_analysis reports.
  (1) A VAR fitted on a subset of the variables: a variable that does not Granger-cause another in the full system
      can do so once a common cause is left out, and the subset's projection on its own past is no longer a VAR of
      finite order (Sargent & Stachurski, QuantEcon, ch. 53, edition of 2026-11-12; Sims 1972, 1980).
  (2) Measurement error: a noisy measure of y makes the past of a second, independent measure of the same variable
      predict y, although nothing causes y but its own past (QuantEcon, ch. 54).
Reference: the population autocovariances of the system (Γ0 solves Γ0 = A Γ0 A' + Σ, scipy's discrete Lyapunov
solver; Γh = A^h Γ0), and the linear projection of y_t on p lags of the observed variables from them (normal
equations). The projection is computed here, not by the skill.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "econometrics" / "scripts"))
import _console  # noqa: E402,F401  (Windows-safe console output, see econometrics/scripts/_console.py)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.linalg import solve_discrete_lyapunov  # noqa: E402

import multivariate as mv  # noqa: E402

ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info)


def projection(gamma, target, keep, p):
    """Coefficients of the projection of variable `target` at t on lags 1..p of the variables `keep`, from the
    autocovariance function gamma(h) = E[w_t w_{t-h}'] (h >= 0). Returns an array (p, len(keep)): row j-1 = lag j."""
    m = len(keep)

    def g(h):                                          # E[o_t o_{t-h}'] for the observed subvector, any sign of h
        G = gamma(abs(h))
        G = G if h >= 0 else G.T
        return G[np.ix_(keep, keep)]

    V = np.block([[g(j - i) for j in range(1, p + 1)] for i in range(1, p + 1)])   # Var of (o_{t-1}, …, o_{t-p})
    c = np.concatenate([gamma(i)[target, keep] for i in range(1, p + 1)])         # Cov(y_t, o_{t-i})
    return np.linalg.solve(V, c).reshape(p, m)


# ------------------------------------------------------------------ rule 1: a common cause left out
# w = (x, y, z): z causes x and y; x does not cause y; e ~ N(0, I)
A = np.array([[0.4, 0.0, 0.8],
              [0.0, 0.4, 0.8],
              [0.0, 0.0, 0.8]])
G0 = solve_discrete_lyapunov(A, np.eye(3))


def gamma_var(h):
    return np.linalg.matrix_power(A, h) @ G0


full = projection(gamma_var, 1, [0, 1, 2], 4)
check("rule 1, population: in the full VAR the projection of y on 4 lags of (x, y, z) is the true row of A "
      "(lag 1 = [0, 0.4, 0.8], later lags 0; x's coefficients 0)",
      np.allclose(full[0], A[1], atol=1e-12) and np.allclose(full[1:], 0, atol=1e-12), full)
sub = projection(gamma_var, 1, [0, 1], 4)
check("rule 1, population: with z left out, the past of x enters y's projection (lag-1 coefficient of x above 0.1)",
      sub[0, 0] > 0.1, sub)
sub8 = projection(gamma_var, 1, [0, 1], 8)
check("rule 1, population: the subsystem is not a finite-order VAR: with 8 lags the 8th still has a non-zero "
      "coefficient (above 1e-7; the full system's are 0 to 1e-12)", np.abs(sub8[7]).max() > 1e-7
      and np.allclose(projection(gamma_var, 1, [0, 1, 2], 8)[1:], 0, atol=1e-12), sub8[7])


def sim_var(T, seed, burn=200):
    r = np.random.default_rng(seed)
    e = r.normal(size=(T + burn, 3))
    w = np.zeros((T + burn, 3))
    for t in range(1, T + burn):
        w[t] = A @ w[t - 1] + e[t]
    return pd.DataFrame(w[burn:], columns=["x", "y", "z"])


R = 40
rej_full, rej_sub = [], []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for rep in range(R):
        d = sim_var(400, 74000 + rep)
        rej_full.append(mv.var_analysis(d, lags=2, show=False)["granger_pvalues"].loc["y", "x"] < 0.05)
        rej_sub.append(mv.var_analysis(d[["x", "y"]], lags=2, show=False)["granger_pvalues"].loc["y", "x"] < 0.05)
check(f"rule 1, var_analysis ({R} samples, T = 400, p = 2): with z in the VAR, 'x Granger-causes y' is rejected in "
      "at most 15% of samples (nominal 5%)", np.mean(rej_full) <= 0.15, np.mean(rej_full))
check(f"rule 1, var_analysis ({R} samples): with z left out, the same test rejects in at least 90% of samples",
      np.mean(rej_sub) >= 0.90, np.mean(rej_sub))

# ------------------------------------------------------------------ rule 2: measurement error
# y* is an AR(1); x = y* + u (a second measure of y*; it causes nothing); observed y = y* + v
phi, s2u = 0.9, 1.0
s2star = 1.0 / (1 - phi ** 2)


def gamma_me(s2v):
    def gamma(h):
        G = phi ** h * s2star * np.ones((2, 2))           # (x, y) share y*
        if h == 0:
            G = G + np.diag([s2u, s2v])
        return G
    return gamma


p0 = projection(gamma_me(0.0), 1, [0, 1], 4)
check("rule 2, population: measured without error (y = y*), the past of x does not enter y's projection "
      "(x coefficients 0; y lag 1 = 0.9)", np.allclose(p0[:, 0], 0, atol=1e-12) and abs(p0[0, 1] - phi) < 1e-12, p0)
p1 = projection(gamma_me(1.0), 1, [0, 1], 4)
check("rule 2, population: with measurement error in y (var 1), the past of x enters (lag-1 coefficient above 0.2) "
      "and y's own lag-1 coefficient falls below 0.9", p1[0, 0] > 0.2 and p1[0, 1] < phi, p1)


def sim_me(T, seed, sv, burn=200):
    r = np.random.default_rng(seed)
    s = np.zeros(T + burn)
    e = r.normal(size=T + burn)
    for t in range(1, T + burn):
        s[t] = phi * s[t - 1] + e[t]
    s = s[burn:]
    return pd.DataFrame({"x": s + r.normal(scale=np.sqrt(s2u), size=T), "y": s + sv * r.normal(size=T)})


rej0, rej1 = [], []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for rep in range(R):
        rej0.append(mv.var_analysis(sim_me(400, 74100 + rep, 0.0), lags=2, show=False)
                    ["granger_pvalues"].loc["y", "x"] < 0.05)
        rej1.append(mv.var_analysis(sim_me(400, 74100 + rep, 1.0), lags=2, show=False)
                    ["granger_pvalues"].loc["y", "x"] < 0.05)
check(f"rule 2, var_analysis ({R} samples, T = 400, p = 2): y without error, 'x Granger-causes y' rejected in at most "
      "15% of samples", np.mean(rej0) <= 0.15, np.mean(rej0))
check(f"rule 2, var_analysis ({R} samples): y with error, rejected in at least 90% of samples", np.mean(rej1) >= 0.90,
      np.mean(rej1))

# ------------------------------------------------------------------ the rules are written where the agent reads them
ROOT = Path(__file__).resolve().parents[1] / "econometrics"
ir = (ROOT / "references" / "interpretation-rules.md").read_text(encoding="utf-8")
sec = (ir.split("## Rules added in v2.7.4", 1)[1].split("\n## ", 1)[0] if "## Rules added in v2.7.4" in ir else "").lower()
check("interpretation-rules.md has a 'Rules added in v2.7.4' section with both rules (omitted variables, infinite "
      "order; measurement error and filtered or revised data)",
      "omitted" in sec and "infinite order" in sec and "measurement error" in sec and "filtered" in sec
      and "revised" in sec, sec[:300])
rc = (ROOT / "references" / "review-checklist.md").read_text(encoding="utf-8")
check("review-checklist.md asks, for a VAR, about left-out variables and measured or filtered data",
      any("Granger" in l_ and "left out" in l_ and "measurement error" in l_ for l_ in rc.splitlines()))
ts = (ROOT / "references" / "time-series.md").read_text(encoding="utf-8")
check("time-series.md points to the v2.7.4 rules from the VAR limitations",
      "interpretation-rules.md" in ts and "v2.7.4" in ts.split("General limitations", 1)[-1][:800])

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
