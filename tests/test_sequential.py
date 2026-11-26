"""Tests for sequential.py (v2.8.0): SPRT, peeking, divergences, divergence rates, sample size, importance sampling.

References, all by another route than the function:
- closed forms written here: KL between normals, between Betas (digamma) and between Poissons; Chernoff information of
  two normals with equal variance, (mu1 - mu0)^2 / (8 sigma^2) at phi = 1/2; Jensen-Shannon of probability vectors;
  the log-likelihood ratio of N(0,1) against N(mu,1), mu x - mu^2/2, its exact normal law, the Neyman-Pearson sample
  size ((z_a + z_b)/mu)^2 and the exact tail probabilities;
- published values: QuantEcon lecture-python.myst 0dc4fe7 ("Statistical Divergence Measures", "Likelihood Ratio
  Processes", "A Problem that Stumped Milton Friedman": KL 0.7592, Chernoff 0.1212 at phi 0.5969, Markov rates 0.0588
  and 0.0563, SPRT mean stop 1.59 and errors 0.012 and 0.022, printed in the book of 2026-11-12); Armitage, McPherson
  and Rowe (1969) and Pocock (1977) for peeking;
- simulation written here: long Markov and VAR paths for the rates, z paths for peeking.

Run from the repository root:  python3 tests/test_sequential.py
"""
import contextlib
import io
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "econometrics" / "scripts"))
warnings.filterwarnings("ignore")
import numpy as np
from scipy import special, stats

import sequential as sq

ok = []


def check(name, cond, info=""):
    ok.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info)


def near(a, b, tol):
    return abs(float(a) - float(b)) <= tol


H4 = 0.00005  # half a unit of a four-decimal published value


def out_of(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*a, **k)
    return buf.getvalue()


# ------------------------------------------------------------------------------------------------ divergence
def kl_norm(m1, s1, m2, s2):
    return np.log(s2 / s1) + (s1 ** 2 + (m1 - m2) ** 2) / (2 * s2 ** 2) - 0.5


def kl_beta(a1, b1, a2, b2):
    return (special.betaln(a2, b2) - special.betaln(a1, b1) + (a1 - a2) * special.digamma(a1)
            + (b1 - b2) * special.digamma(b1) + (a2 - a1 + b2 - b1) * special.digamma(a1 + b1))


d = sq.divergence(stats.norm(0.3, 1.2), stats.norm(-0.5, 0.7), show=False)
check("divergence: KL between normals equals the closed form both ways (abs 1e-9)",
      near(d["kl_pq"], kl_norm(0.3, 1.2, -0.5, 0.7), 1e-9) and near(d["kl_qp"], kl_norm(-0.5, 0.7, 0.3, 1.2), 1e-9),
      (d["kl_pq"], kl_norm(0.3, 1.2, -0.5, 0.7)))
de = sq.divergence(stats.norm(0, 2), stats.norm(1.5, 2), show=False)
check("divergence: Chernoff of two normals with equal variance = (mu1 - mu0)^2/(8 sigma^2) at phi = 1/2 "
      "(abs 1e-9; phi 1e-6), equal to Bhattacharyya", near(de["chernoff"], 1.5 ** 2 / 32, 1e-9)
      and near(de["chernoff_phi"], 0.5, 1e-6) and near(de["bhattacharyya"], 1.5 ** 2 / 32, 1e-9), de)
db = sq.divergence(stats.beta(1, 1), stats.beta(3, 1.2), show=False)
check("divergence: KL(Beta(1,1)||Beta(3,1.2)) equals the digamma closed form (abs 1e-8) both ways",
      near(db["kl_pq"], kl_beta(1, 1, 3, 1.2), 1e-8) and near(db["kl_qp"], kl_beta(3, 1.2, 1, 1), 1e-8),
      (db["kl_pq"], kl_beta(1, 1, 3, 1.2)))
check("divergence: the book's KL 0.7592, Chernoff 0.1212 at phi 0.5969 (four decimals)",
      near(db["kl_pq"], 0.7592, H4) and near(db["chernoff"], 0.1212, H4) and near(db["chernoff_phi"], 0.5969, H4), db)
check("divergence: Bhattacharyya <= Chernoff <= KL KL/(KL + KL) <= min KL, and no warning",
      db["bhattacharyya"] <= db["chernoff"] <= db["kl_pq"] * db["kl_qp"] / (db["kl_pq"] + db["kl_qp"])
      <= min(db["kl_pq"], db["kl_qp"]) and not db["warnings"])
dp = sq.divergence(stats.poisson(3.0), stats.poisson(5.0), show=False)
check("divergence: KL between Poissons = l1 log(l1/l2) + l2 - l1 (discrete sum, abs 1e-10)",
      near(dp["kl_pq"], 3 * np.log(3 / 5) + 5 - 3, 1e-10) and dp["used"]["kind"] == "discrete", dp["kl_pq"])
p_, q_ = np.array([0.5, 0.3, 0.2]), np.array([0.2, 0.2, 0.6])
m_ = (p_ + q_) / 2
js_ = 0.5 * np.sum(p_ * np.log(p_ / m_)) + 0.5 * np.sum(q_ * np.log(q_ / m_))
dv = sq.divergence(p_, q_, show=False)
check("divergence: probability vectors, KL and Jensen-Shannon by hand (abs 1e-14), JS <= log 2",
      near(dv["kl_pq"], np.sum(p_ * np.log(p_ / q_)), 1e-14) and near(dv["js"], js_, 1e-14) and dv["js"] <= np.log(2))
dn = sq.divergence(stats.norm(0, 1), stats.norm(0.8, 1), show=False)
check("divergence: JS of two continuous normals equals JS of a fine discretization (abs 1e-6)",
      near(dn["js"], (lambda x: (lambda pp, qq: 0.5 * np.sum(pp * np.log(2 * pp / (pp + qq)))
                                 + 0.5 * np.sum(qq * np.log(2 * qq / (pp + qq))))(
          stats.norm.pdf(x) * (x[1] - x[0]), stats.norm.pdf(x, 0.8) * (x[1] - x[0])))(np.linspace(-12, 13, 250001)), 1e-6))
du = sq.divergence(stats.uniform(0, 1), stats.uniform(0, 0.5), show=False)
check("divergence: p with mass where q has none -> KL(p||q) = inf, the reverse finite, and a notice",
      np.isinf(du["kl_pq"]) and np.isfinite(du["kl_qp"]) and any(w["level"] == "notice" for w in du["warnings"]))
for bad in ((np.array([0.5, 0.6]), np.array([0.5, 0.5])), (stats.norm(0, 1), stats.poisson(2))):
    try:
        sq.divergence(*bad, show=False)
        refused = False
    except ValueError:
        refused = True
    check(f"divergence: refuses {type(bad[1]).__name__} input that is not a valid pair", refused)
txt = out_of(sq.divergence, stats.beta(1, 1), stats.beta(3, 1.2))
check("divergence show: prints KL both ways, JS with its maximum, Chernoff with phi and Bhattacharyya",
      "KL(p||q) = 0.759217" in txt and "JS =" in txt and "0.6931" in txt and "phi = 0.5969" in txt
      and "Bhattacharyya" in txt, txt)

# ------------------------------------------------------------------------------------------- divergence_rate
Pf = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.1, 0.3, 0.6]])
Pg = np.array([[0.5, 0.3, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
rm = sq.divergence_rate(Pf, Pg, show=False)
check("divergence_rate: Markov rates equal the book's 0.0588 and 0.0563 (four decimals)",
      near(rm["rate_pq"], 0.0588, H4) and near(rm["rate_qp"], 0.0563, H4) and rm["kind"] == "markov", rm)
rng = np.random.default_rng(1)
T = 400_000
s = np.empty(T + 1, dtype=int)
s[0] = 0
u = rng.random(T)
cum = np.cumsum(Pf, axis=1)
for t in range(T):
    s[t + 1] = int(np.searchsorted(cum[s[t]], u[t], side="right"))
lr_steps = np.log(Pf[s[:-1], s[1:]] / Pg[s[:-1], s[1:]])
se_m = lr_steps.std() / np.sqrt(T) * 3  # generous for serial dependence
check("divergence_rate: Markov rate = mean log-likelihood-ratio step on a simulated path of 400 000 steps "
      "(within 3 x the iid SE x 3)", abs(lr_steps.mean() - rm["rate_pq"]) < 3 * se_m, (lr_steps.mean(), rm["rate_pq"]))
check("divergence_rate: the stationary distribution solves pi P = pi",
      np.allclose(rm["stationary_p"] @ Pf, rm["stationary_p"], atol=1e-12))
A0 = np.array([[0.8, 0.1], [0.2, 0.7]])
C0 = np.array([[0.3, 0.1], [0.1, 0.3]])
A1 = np.array([[0.6, 0.2], [0.3, 0.5]])
C1 = np.diag([0.4, 0.4])
rv = sq.divergence_rate({"A": A0, "C": C0}, {"A": A1, "C": C1}, show=False)
rv2 = sq.divergence_rate((A0, C0 @ C0.T), {"A": A1, "Sigma": C1 @ C1.T}, show=False)
check("divergence_rate: the three VAR input forms give the same rates", near(rv["rate_pq"], rv2["rate_pq"], 1e-14)
      and near(rv["rate_qp"], rv2["rate_qp"], 1e-14))
rng = np.random.default_rng(2)
T = 200_000
x = np.zeros((T + 1, 2))
for t in range(T):
    x[t + 1] = A0 @ x[t] + C0 @ rng.standard_normal(2)
S0, S1 = C0 @ C0.T, C1 @ C1.T
e0 = x[1:] - x[:-1] @ A0.T
e1 = x[1:] - x[:-1] @ A1.T
ll0 = stats.multivariate_normal(np.zeros(2), S0).logpdf(e0)
ll1 = stats.multivariate_normal(np.zeros(2), S1).logpdf(e1)
diff = ll0[1000:] - ll1[1000:]
check("divergence_rate: VAR rate = mean log-likelihood-ratio step on a simulated path of 200 000 steps "
      "(within 4 iid SEs)", abs(diff.mean() - rv["rate_pq"]) < 4 * diff.std() / np.sqrt(len(diff)),
      (diff.mean(), rv["rate_pq"]))
check("divergence_rate: the VAR stationary covariance solves V = A V A' + Sigma",
      np.allclose(A0 @ rv["stationary_p"] @ A0.T + S0, rv["stationary_p"], atol=1e-12))
try:
    sq.divergence_rate({"A": np.array([[1.01]]), "C": np.array([[1.0]])}, {"A": np.array([[0.5]]), "C": np.array([[1.0]])},
                       show=False)
    unstable = False
except ValueError:
    unstable = True
check("divergence_rate: an unstable VAR is refused", unstable)
Pz = np.array([[0.5, 0.5, 0.0], [0.3, 0.4, 0.3], [0.2, 0.3, 0.5]])
rz = sq.divergence_rate(Pg, Pz, show=False)
check("divergence_rate: a transition allowed by P and forbidden by Q -> infinite rate and a notice",
      np.isinf(rz["rate_pq"]) and np.isfinite(rz["rate_qp"]) and rz["warnings"])
Pred = np.array([[1.0, 0.0], [0.0, 1.0]])
check("divergence_rate: a reducible chain gets the warning",
      any("reducible" in w["message"] for w in sq.divergence_rate(Pred, np.full((2, 2), 0.5), show=False)["warnings"]))
txt = out_of(sq.divergence_rate, Pf, Pg)
check("divergence_rate show: prints the kind, the unit and both rates", "markov, nats per step" in txt
      and "0.058762" in txt and "0.0562548" in txt, txt)

# ------------------------------------------------------------------------------------------------ SPRT design
sd = sq.sprt_design(stats.beta(2, 5), stats.beta(5, 2), alpha=0.05, beta=0.10, n_sim=20000, seed=1, show=False)
check("sprt_design: thresholds log((1-b)/a) = log 18 and log(b/(1-a)) = log(0.1/0.95) (abs 1e-14)",
      near(sd["log_A"], np.log(18), 1e-14) and near(sd["log_B"], np.log(0.1 / 0.95), 1e-14))
check("sprt_design: Wald's bounds alpha/(1-beta) and beta/(1-alpha)",
      near(sd["wald_bound_alpha"], 0.05 / 0.9, 1e-15) and near(sd["wald_bound_beta"], 0.1 / 0.95, 1e-15))
check("sprt_design: book's Beta(2,5) vs Beta(5,2): errors 0.012 and 0.022 within 3 MC SEs + 0.0005, mean stop "
      "1.59 within 0.03 (MC and rounding)",
      near(sd["sim_alpha"], 0.012, 3 * sd["sim_alpha_se"] + 0.0005)
      and near(sd["sim_beta"], 0.022, 3 * sd["sim_beta_se"] + 0.0005) and near(sd["sim_en"], 1.59, 0.03), sd)
check("sprt_design: simulated errors within Wald's bounds; the overshoot notice fires",
      sd["sim_alpha"] <= sd["wald_bound_alpha"] and sd["sim_beta"] <= sd["wald_bound_beta"]
      and any("overshoot" in w["message"] for w in sd["warnings"]))
# small steps: N(0,1) against N(0.2,1); Wald's approximations get close and the errors near the targets
sn = sq.sprt_design(stats.norm(0, 1), stats.norm(0.2, 1), alpha=0.05, beta=0.10, n_sim=8000, seed=2, show=False)
en1_cf = ((0.9 * np.log(18) + 0.1 * np.log(0.1 / 0.95)) / 0.02)
en0_cf = ((0.05 * np.log(18) + 0.95 * np.log(0.1 / 0.95)) / -0.02)
check("sprt_design: normal case, Wald's approximate E[N] equals the closed form with KL = mu^2/2 (rel 1e-8)",
      near(sn["approx_en1"] / en1_cf, 1, 1e-8) and near(sn["approx_en0"] / en0_cf, 1, 1e-8), (sn["approx_en1"], en1_cf))
check("sprt_design: normal case with small steps, simulated E[N] between Wald's approximation and 25% above "
      "(overshoot adds a little)", en1_cf <= sn["sim_en1"] <= 1.25 * en1_cf and en0_cf <= sn["sim_en0"] <= 1.25 * en0_cf,
      (sn["sim_en0"], en0_cf, sn["sim_en1"], en1_cf))
check("sprt_design: normal case, simulated errors near the targets (alpha in [0.03, 0.056], beta in [0.07, 0.106])",
      0.03 <= sn["sim_alpha"] <= 0.056 and 0.07 <= sn["sim_beta"] <= 0.106, (sn["sim_alpha"], sn["sim_beta"]))
st = sq.sprt_design(stats.norm(0, 1), stats.norm(0.01, 1), n_sim=200, max_n=50, seed=3, show=False)
check("sprt_design: runs cut at max_n are reported as undecided (warning)",
      any("without a decision" in w["message"] for w in st["warnings"]))
for a_, b_ in ((0.0, 0.1), (0.6, 0.5)):
    try:
        sq.sprt_design(stats.norm(0, 1), stats.norm(1, 1), alpha=a_, beta=b_, show=False)
        bad = False
    except ValueError:
        bad = True
    check(f"sprt_design: refuses alpha = {a_}, beta = {b_}", bad)
txt = out_of(sq.sprt_design, stats.beta(2, 5), stats.beta(5, 2), n_sim=2000, seed=1)
check("sprt_design show: thresholds, Wald's guarantees, KL and approximate E[N], simulated errors with MC SEs, "
      "the finding with its action", "reject H0 when log L >= 2.8904" in txt and "Wald's guarantees" in txt
      and "approximate E[N]" in txt and "MC SE" in txt and "NOTICE:" in txt and "action:" in txt, txt)

# ------------------------------------------------------------------------------------------------------- sprt
# N(0,1) against N(1,1): log-likelihood-ratio step = x - 1/2
f0, f1 = stats.norm(0, 1), stats.norm(1, 1)
xs = np.array([0.4, 1.1, 0.9, 1.6, 0.2])
r = sq.sprt(xs, f0, f1, show=False)
check("sprt: the path equals cumsum(x - 1/2) (abs 1e-12)", np.allclose(r["llr"].to_numpy(), np.cumsum(xs - 0.5),
                                                                       atol=1e-12, rtol=0))
check("sprt: no crossing -> 'continue', n_stop None, and the notice not to switch to a fixed-sample test",
      r["decision"] == "continue" and r["n_stop"] is None and any("keep sampling" in w["action"] for w in r["warnings"]))
xs3 = np.array([1.5, 1.5, 1.5, -5.0])                               # 1.0, 2.0, 3.0 >= log 18 = 2.89 at the 3rd
r3 = sq.sprt(xs3, f0, f1, show=False)
check("sprt: first crossing of log A at observation 3 -> reject H0, later observations ignored (notice)",
      r3["decision"] == "reject H0" and r3["n_stop"] == 3 and r3["n_seen"] == 4
      and any("do not enter the decision" in w["message"] for w in r3["warnings"]))
r4 = sq.sprt(np.array([-1.0, -0.9]), f0, f1, show=False)              # -1.5, -2.9 <= log(0.1/0.95) = -2.25 at the 2nd
check("sprt: first crossing of log B at observation 2 -> accept H0", r4["decision"] == "accept H0" and r4["n_stop"] == 2)
r5 = sq.sprt(np.array([1.5, 1.5, 1.38]), f0, f1, show=False)        # 1.0, 2.0, 2.88 < 2.8904: just below log A
check("sprt: just below log A (2.88 < 2.8904) -> continue", r5["decision"] == "continue")
try:
    sq.sprt(np.array([0.1, np.nan]), f0, f1, show=False)
    nanok = False
except ValueError:
    nanok = True
check("sprt: missing values are refused", nanok)
txt = out_of(sq.sprt, xs3, f0, f1)
check("sprt show: decision, stopping point and thresholds", "reject H0 after 3 observations" in txt and "log A" in txt, txt)

# -------------------------------------------------------------------------------------------------- peeking
pub = {2: 0.083, 5: 0.142, 10: 0.193, 20: 0.248}       # Armitage, McPherson and Rowe (1969), alpha 0.05 two-sided
pocock = {2: 0.0294, 5: 0.0158, 10: 0.0106}             # Pocock (1977), nominal level per look
pk = {K: sq.peeking_size(K, show=False) for K in (1, 2, 5, 10, 20)}
check("peeking_size: one look is the test itself (size = alpha, per-look = alpha)",
      near(pk[1]["size"], 0.05, 1e-12) and near(pk[1]["per_look_alpha"], 0.05, 1e-6) and not pk[1]["warnings"])
check("peeking_size: Armitage-McPherson-Rowe sizes 0.083, 0.142, 0.193, 0.248 (three decimals)",
      all(near(pk[K]["size"], v, 0.0005) for K, v in pub.items()), {K: pk[K]["size"] for K in pub})
check("peeking_size: Pocock's per-look levels 0.0294, 0.0158, 0.0106 (four decimals)",
      all(near(pk[K]["per_look_alpha"], v, H4) for K, v in pocock.items()), {K: pk[K]["per_look_alpha"] for K in pocock})
mvn3 = stats.multivariate_normal(np.zeros(3), np.sqrt(np.minimum.outer([1, 2, 3], [1, 2, 3])
                                                       / np.maximum.outer([1, 2, 3], [1, 2, 3])), seed=7)
z = stats.norm.isf(0.025)
check("peeking_size: three looks equal 1 - the trivariate normal CDF of the Brownian correlations (abs 1e-4)",
      near(sq.peeking_size(3, show=False)["size"], 1 - mvn3.cdf(np.full(3, z), lower_limit=np.full(3, -z)), 1e-4))
rng = np.random.default_rng(5)
Ssum = np.cumsum(rng.standard_normal((200_000, 5)), axis=1)
Zk = Ssum / np.sqrt(np.arange(1, 6))
sim5 = np.mean(np.any(np.abs(Zk) > z, axis=1))
check("peeking_size: five looks equal the share of 200 000 simulated z paths that cross (within 3 MC SEs)",
      abs(pk[5]["size"] - sim5) < 3 * np.sqrt(sim5 * (1 - sim5) / 200_000), (pk[5]["size"], sim5))
Zk1 = Ssum / np.sqrt(np.arange(1, 6))
sim5o = np.mean(np.any(Zk1 > stats.norm.isf(0.05), axis=1))
po = sq.peeking_size(5, two_sided=False, show=False)
check("peeking_size: one-sided, five looks, against the simulated paths (within 3 MC SEs)",
      abs(po["size"] - sim5o) < 3 * np.sqrt(sim5o * (1 - sim5o) / 200_000), (po["size"], sim5o))
check("peeking_size: with the per-look level the size is alpha again (abs 1e-6)",
      near(sq.peeking_size(10, alpha=pk[10]["per_look_alpha"], show=False)["size"], 0.05, 2e-3)
      and near(pk[10]["per_look_z"], stats.norm.isf(pk[10]["per_look_alpha"] / 2), 1e-9))
check("peeking_size: more than one look -> a warning with the size and the per-look level",
      any(w["level"] == "warning" and "0.193" in w["message"] for w in pk[10]["warnings"])
      and any("0.0106" in w["action"] for w in pk[10]["warnings"]))
txt = out_of(sq.peeking_size, 10)
check("peeking_size show: the looks, the overall size and Pocock's level", "10 equally spaced looks" in txt
      and "overall type I error = 0.193" in txt and "Pocock" in txt, txt)

# ---------------------------------------------------------------------------------------------- n_for_error
mu = 0.5
nf = sq.n_for_error(stats.norm(0, 1), stats.norm(mu, 1), alpha=0.05, beta=0.10, simulate=True, n_sim=20000, seed=4,
                    show=False)
n_cf = int(np.ceil(((stats.norm.isf(0.05) + stats.norm.isf(0.10)) / mu) ** 2))
check("n_for_error: normal means, n = ceil(((z_a + z_b)/mu)^2) = 35 (exact here)", nf["n_normal"] == n_cf == 35, nf)
check("n_for_error: the simulated n is within 1 of the closed form", abs(nf["n_simulated"] - n_cf) <= 1, nf["n_simulated"])
check("n_for_error: Stein exponents KL = mu^2/2 and Chernoff mu^2/8 (abs 1e-9)",
      near(nf["kl_01"], mu ** 2 / 2, 1e-9) and near(nf["chernoff"], mu ** 2 / 8, 1e-9))
check("n_for_error: the SPRT's approximate E[N] is below the fixed n under both hypotheses (16 and 19 against 35)",
      nf["sprt_en0"] < n_cf and nf["sprt_en1"] < n_cf and near(nf["sprt_en0"], en0_cf * 0.02 / (mu ** 2 / 2), 1e-9),
      (nf["sprt_en0"], nf["sprt_en1"]))
nb = sq.n_for_error(stats.bernoulli(0.3), stats.bernoulli(0.4), simulate=True, n_sim=20000, seed=6, show=False)
check("n_for_error: Bernoulli 0.3 vs 0.4, normal approximation within 15% of the simulated n",
      abs(nb["n_normal"] / nb["n_simulated"] - 1) < 0.15, (nb["n_normal"], nb["n_simulated"]))
txt = out_of(sq.n_for_error, stats.norm(0, 1), stats.norm(mu, 1))
check("n_for_error show: fixed n, SPRT expected sizes and the exponents", "n = 35 (normal approximation)" in txt
      and "SPRT" in txt and "Chernoff" in txt, txt)

# ------------------------------------------------------------------------------------- importance sampling
# log L_n ~ N(-n mu^2/2, n mu^2) under H0 and N(n mu^2/2, n mu^2) under H1 for N(0,1) vs N(mu,1)
n_, c_ = 40, 6.0
it1 = sq.importance_tail_prob(stats.norm(0, 1), stats.norm(mu, 1), n=n_, threshold=c_, draws=20000, seed=8, show=False)
ex1 = stats.norm.sf(c_, loc=-n_ * mu ** 2 / 2, scale=np.sqrt(n_) * mu)
check("importance_tail_prob: type I tail equals the exact normal tail 2.52e-4 within 3 SEs, SE < 2% of it",
      abs(it1["estimate"] - ex1) < 3 * it1["se"] and it1["se"] < 0.02 * ex1, (it1["estimate"], it1["se"], ex1))
check("importance_tail_prob: variance at least 50x smaller than plain simulation with the same draws",
      it1["variance_ratio"] > 50, it1["variance_ratio"])
it2 = sq.importance_tail_prob(stats.norm(0, 1), stats.norm(mu, 1), n=n_, threshold=-6.0, error="type2", draws=20000,
                              seed=9, show=False)
ex2 = stats.norm.cdf(-6.0, loc=n_ * mu ** 2 / 2, scale=np.sqrt(n_) * mu)
check("importance_tail_prob: type II tail equals the exact normal tail within 3 SEs", abs(it2["estimate"] - ex2)
      < 3 * it2["se"], (it2["estimate"], it2["se"], ex2))
it3 = sq.importance_tail_prob(stats.norm(0, 1), stats.norm(mu, 1), n=n_, threshold=10.0, draws=2000, seed=10,
                              show=False)
ex3 = stats.norm.sf(10.0, loc=-n_ * mu ** 2 / 2, scale=np.sqrt(n_) * mu)
check("importance_tail_prob: a 1e-6 tail: plain simulation finds no hit (notice), importance sampling within 3 SEs",
      it3["hits_naive"] == 0 and any("plain draws" in w["message"] for w in it3["warnings"])
      and abs(it3["estimate"] - ex3) < 3 * it3["se"], (it3["estimate"], it3["se"], ex3))
it4 = sq.importance_tail_prob(stats.norm(0, 1), stats.norm(mu, 1), n=n_, threshold=60.0, draws=500, seed=11,
                              show=False)
check("importance_tail_prob: a threshold far beyond the sampler leaves few effective draws -> warning",
      it4["ess"] < 30 and any("effective draws" in w["message"] for w in it4["warnings"]), it4["ess"])
txt = out_of(sq.importance_tail_prob, stats.norm(0, 1), stats.norm(mu, 1), n=n_, threshold=c_, draws=2000)
check("importance_tail_prob show: estimate with SE, the plain estimate with its hits, variance ratio",
      "by importance sampling" in txt and "plain simulation" in txt and "hits" in txt and "variance ratio" in txt, txt)

# ------------------------------------------------------------------------------------- v2.8.0 review r1
# S1-4: quad over (-inf, inf) missed a narrow density far from 0 (KL 0 for N(100, 1) vs N(100.5, 1), no warning)
far = []
for loc, sc, dlt in ((40.0, 1.0, 0.5), (100.0, 1.0, 0.5), (100.0, 0.01, 0.1), (1e4, 1.0, 1.0), (-3e5, 2.0, 1.0),
                     (1e6, 0.001, 0.0005)):
    dd = sq.divergence(stats.norm(loc, sc), stats.norm(loc + dlt, sc), show=False)
    rt_ = 1e-8 + 100 * np.finfo(float).eps * abs(loc) / sc    # rounding of (x - loc)/sc in the densities
    kl_cf, ch_cf = dlt ** 2 / (2 * sc ** 2), dlt ** 2 / (8 * sc ** 2)
    far.append(near(dd["kl_pq"], kl_cf, rt_ * kl_cf) and near(dd["kl_qp"], kl_cf, rt_ * kl_cf)
               and near(dd["chernoff"], ch_cf, 10 * rt_ * ch_cf) and not dd["warnings"])
check("divergence: normals far from 0 (locations 40 to 1e6, sd 0.001 to 2): KL = delta^2/(2 sd^2) and Chernoff = "
      "delta^2/(8 sd^2) (rel 1e-8 + 100 eps |loc|/sd), no warning", all(far), far)
jsf = sq.divergence(stats.norm(1e4, 1), stats.norm(1e4 + 1, 1), show=False)["js"]
js0 = sq.divergence(stats.norm(0, 1), stats.norm(1, 1), show=False)["js"]
check("divergence: JS is the same at location 0 and 1e4 (rel 1e-8)", near(jsf, js0, 1e-8 * js0), (jsf, js0))
cn = sq.divergence(stats.cauchy(0, 1), stats.norm(0, 1), show=False)
check("divergence: KL(Cauchy || N(0,1)) is infinite and was -2.43 -> NaN with a warning, never a negative KL",
      np.isnan(cn["kl_pq"]) and any(w["level"] == "warning" and "not computed" in w["message"] for w in cn["warnings"])
      and near(cn["kl_qp"], 0.2592445325, 1e-9), (cn["kl_pq"], cn["kl_qp"]))
# 0.2592445325: KL(N(0,1) || Cauchy(0,1)) by mpmath quad at 30 digits, E_N[log phi(x) + log(pi (1 + x^2))]
nf100 = sq.n_for_error(stats.norm(100, 1), stats.norm(100.5, 1), show=False)
check("n_for_error: N(100,1) vs N(100.5,1) gives the same n = 35 as at 0 (it raised 'cannot convert NaN')",
      nf100["n_normal"] == 35 and near(nf100["sprt_en0"], nf["sprt_en0"], 1e-6), nf100)
sd100 = sq.sprt_design(stats.norm(100, 1), stats.norm(100.2, 1), n_sim=2000, seed=2, show=False)
check("sprt_design: at location 100 KL = 0.02 and Wald's E[N] as at 0 (it gave KL 0 and E[N] NaN)",
      near(sd100["kl_01"], 0.02, 1e-10) and near(sd100["approx_en0"], en0_cf, 1e-6)
      and near(sd100["approx_en1"], en1_cf, 1e-6), (sd100["kl_01"], sd100["approx_en0"]))
try:
    sq.n_for_error(stats.cauchy(0, 1), stats.norm(0, 1), show=False)
    nf_c = ""
except ValueError as err:
    nf_c = str(err)
check("n_for_error: a KL that cannot be integrated -> ValueError that says so", "could not be integrated" in nf_c, nf_c)

# S2-4: n_for_error(simulate=True) built one n_sim x 4n array (2 GiB for N(0,1) vs N(0.05,1)); now in blocks of
# _MAX_BLOCK draws, and skipped with a notice beyond the budget
nf_small = sq.n_for_error(stats.norm(0, 1), stats.norm(0.05, 1), simulate=True, show=False)
check("n_for_error: N(0,1) vs N(0.05,1) with simulate=True -> n_normal 3426, simulation skipped with a notice, "
      "no memory error", nf_small["n_normal"] == int(np.ceil(((stats.norm.isf(0.05) + stats.norm.isf(0.1)) / 0.05) ** 2))
      and nf_small["n_simulated"] is None and any("skipped" in w["message"] for w in nf_small["warnings"]), nf_small)
_mb = sq._MAX_BLOCK
sq._MAX_BLOCK = 500
try:
    nf_blk = sq.n_for_error(stats.norm(0, 1), stats.norm(mu, 1), simulate=True, n_sim=20000, seed=4, show=False)
finally:
    sq._MAX_BLOCK = _mb
check("n_for_error: drawing in blocks of 500 gives the same simulated n as one block (same random stream)",
      nf_blk["n_simulated"] == nf["n_simulated"], (nf_blk["n_simulated"], nf["n_simulated"]))

# S3-4: runs could pass max_n by up to 63 draws and count as decided
smx = sq.sprt_design(stats.norm(0, 1), stats.norm(0.02, 1), n_sim=400, max_n=50, seed=3, show=False)
check("sprt_design: max_n = 50 is respected (longest run 50) and the undecided runs get the warning",
      smx["sim_max_n"] == 50 and any("max_n = 50" in w["message"] for w in smx["warnings"]), smx["sim_max_n"])

# S3-6: a transition matrix as a tuple of tuples was taken for a VAR and crashed
rt = sq.divergence_rate(tuple(map(tuple, Pf)), tuple(map(tuple, Pg)), show=False)
check("divergence_rate: a Markov matrix given as a tuple of tuples = the array result",
      rt["kind"] == "markov" and near(rt["rate_pq"], rm["rate_pq"], 1e-15), rt["kind"])

# S3-5: precision of the recursion against a grid three times finer (one-sided had 4.8e-7 at 50 looks)
z1, z2 = stats.norm.isf(0.05), stats.norm.isf(0.025)
g1 = abs(sq._amr_size(z1, 20, False) - sq._amr_size(z1, 20, False, npt=3601))
g2 = abs(sq._amr_size(z2, 20, True) - sq._amr_size(z2, 20, True, npt=1201))
check("peeking_size: 20 looks, default grid against a 3x finer one: within 1e-8 one-sided and two-sided",
      g1 < 1e-8 and g2 < 1e-8, (g1, g2))

# ------------------------------------------------------------------------------------- v2.8.0 review r2
# S2-5: in the scale of x, densities spread over many decades or infinite at an edge gave silent errors of ~1e-5
# (Gamma(0.1), Beta(0.1, 5)) or NaN (Beta(0.5, 0.5)); KL, JS and the LLR moments are now expectations in the quantile
# scale. References: closed forms (digamma) and, for JS of two Cauchys and the Beta(0.5, 0.5) KL, mpmath quad at 30
# digits (claude/scripts/v280/r2_js_cauchy_ref.py in the development repository).
from scipy import special as _sp


def _kl_gamma(a1, a2):              # unit rate
    return (a1 - a2) * _sp.digamma(a1) - _sp.gammaln(a1) + _sp.gammaln(a2)


def _kl_beta(a1, b1, a2, b2):
    return (_sp.betaln(a2, b2) - _sp.betaln(a1, b1) + (a1 - a2) * _sp.digamma(a1) + (b1 - b2) * _sp.digamma(b1)
            + (a2 - a1 + b2 - b1) * _sp.digamma(a1 + b1))


edge = []
for p_, q_, ref_, rt_ in ((stats.gamma(0.1), stats.gamma(1.0), _kl_gamma(0.1, 1.0), 1e-10),
                          (stats.beta(0.1, 5), stats.beta(1, 1), _kl_beta(0.1, 5, 1, 1), 1e-10),
                          (stats.beta(2, 2), stats.beta(0.5, 0.5), _kl_beta(2, 2, 0.5, 0.5), 1e-10),
                          (stats.beta(0.5, 0.5), stats.beta(2, 2), 1.2223937282822159, 1e-6)):
    r_ = sq.divergence(p_, q_, show=False)
    edge.append((near(r_["kl_pq"], ref_, rt_ * ref_) and not r_["warnings"], r_["kl_pq"], ref_))
check("divergence (r2 S2-5): KL of Gamma(0.1) || Exp(1) and Beta(0.1, 5) || U(0, 1) equal the closed forms (rel 1e-10; "
      "they were off by 1.8e-5), Beta(0.5, 0.5) || Beta(2, 2) within 1e-6 (it was NaN; the limit is the mass within one "
      "ulp of the edge), no warning", all(e[0] for e in edge), edge)
jc = sq.divergence(stats.cauchy(0, 1), stats.cauchy(1, 1), show=False)
check("divergence (r2 S2-5): JS of Cauchy(0, 1) and Cauchy(1, 1) = 0.054230661598 (mpmath; it was NaN), KL = "
      "log(5/4) both ways (rel 1e-10)", near(jc["js"], 0.054230661598185154, 1e-10)
      and near(jc["kl_pq"], np.log(1.25), 1e-10 * np.log(1.25)) and not jc["warnings"], (jc["js"], jc["kl_pq"]))
bh = []
for p_, q_, ref_ in ((stats.gamma(0.1), stats.gamma(1.0), -(_sp.gammaln(0.55) - 0.5 * (_sp.gammaln(0.1) + _sp.gammaln(1)))),
                     (stats.beta(0.5, 0.5), stats.beta(2, 2),
                      -(_sp.betaln(1.25, 1.25) - 0.5 * (_sp.betaln(0.5, 0.5) + _sp.betaln(2, 2))))):
    r_ = sq.divergence(p_, q_, show=False)
    bh.append((near(r_["bhattacharyya"], ref_, 1e-10 * ref_) and r_["chernoff"] >= r_["bhattacharyya"], r_["bhattacharyya"], ref_))
check("divergence: Bhattacharyya of Gamma(0.1) vs Exp(1) and Beta(0.5, 0.5) vs Beta(2, 2) = -log of the closed-form "
      "Bhattacharyya coefficient (rel 1e-10), Chernoff >= Bhattacharyya", all(b[0] for b in bh), bh)
# Chernoff C = max over phi of -log rho(phi) >= the value at 1/2; the search read a NaN rho as a value and gave C < B
cc = sq.divergence(stats.cauchy(0, 1), stats.cauchy(1e6, 1e3), show=False)
check("divergence (r2 S2-5): Cauchy vs Cauchy(1e6, 1e3): Chernoff is NaN with a warning or >= Bhattacharyya (it was "
      "7.493 < 7.717)", (np.isnan(cc["chernoff"]) and any("Chernoff" in w["message"] for w in cc["warnings"]))
      or cc["chernoff"] >= cc["bhattacharyya"], (cc["chernoff"], cc["bhattacharyya"]))
# the LLR moments of n_for_error / sprt_design (the same integral as the KL, fixed with it)
sg = sq.sprt_design(stats.gamma(0.1), stats.gamma(1.0), n_sim=200, seed=1, show=False)
check("sprt_design: Gamma(0.1) vs Exp(1): kl_10 = KL(f1 || f0) closed form (rel 1e-10)",
      near(sg["kl_10"], _kl_gamma(1.0, 0.1), 1e-10 * _kl_gamma(1.0, 0.1)), sg["kl_10"])

# ------------------------------------------------------------------------------------------------- used keys
check("all functions: 'used' carries the skill version and 'warnings' are dicts with level, message and action",
      all(r_["used"]["skill_version"] for r_ in (d, rm, sd, r3, pk[10], nf, it1))
      and all(set(w) == {"level", "message", "action"} for r_ in (du, rz, sd, r3, pk[10], it3) for w in r_["warnings"]))

print(f"\n{sum(ok)}/{len(ok)} checks passed")
sys.exit(0 if all(ok) else 1)
