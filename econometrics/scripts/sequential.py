"""Sequential tests, sample size from divergences, divergences between distributions, and small error
probabilities by importance sampling (v2.8.0). Reference: statistics-foundations.md section 14.

Use as a module:
    import sys; sys.path.insert(0, "<skill folder>/scripts")
    import sequential as sq

Functions
    sprt_design(dist0, dist1, alpha, beta)   Wald's SPRT: log thresholds, Wald's bounds on the errors and approximate
                                             expected sample sizes, and the errors and sample sizes simulated
    sprt(x, dist0, dist1, alpha, beta)       run the SPRT on data in arrival order: decision, stopping point, path
    peeking_size(looks, alpha)               type I error of a fixed-sample z test looked at `looks` times and stopped
                                             at the first p < alpha (equally spaced looks), and the per-look level
                                             that would keep the overall error at alpha (Pocock)
    divergence(p, q)                         KL both ways, Jensen-Shannon, Chernoff information (with the optimal
                                             phi) and Bhattacharyya distance, for scipy distributions or probability
                                             vectors
    divergence_rate(P, Q)                    KL divergence rate between two finite Markov chains or two Gaussian
                                             VAR(1) models, both ways
    n_for_error(dist0, dist1, alpha, beta)   fixed sample size of the likelihood-ratio test for target errors (normal
                                             approximation of the log-likelihood ratio, optionally simulated), next to
                                             the SPRT's approximate expected sample sizes and the Chernoff exponent
    importance_tail_prob(dist0, dist1, n, threshold)
                                             P(sum of n log-likelihood ratios beyond a threshold) under one hypothesis,
                                             by importance sampling from the other (Wald's change of measure), with
                                             the naive estimate from the same number of draws

Distributions are scipy frozen distributions (stats.norm(0, 1), stats.beta(2, 5), stats.poisson(3)); the
observations are independent draws unless the function says otherwise. Every number comes from the arguments
passed in and the seed. Functions follow the v2.7.0 output rules: `warnings` is a list of dicts with level,
message and action, and `used` records the choices applied.

Rule (interpretation-rules.md, v2.8.0): a fixed-sample test looked at before its planned end and stopped at the
first p < alpha does not have size alpha (about 19% instead of 5% with ten looks): use an SPRT or a group-sequential
boundary, or say the p-value is not valid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import integrate, optimize, stats
import _console  # noqa: F401  (Windows-safe console output, see _console.py)

ALPHA = 0.05
_MAX_BLOCK = 2_000_000          # draws per simulation block (memory)
_SIM_BUDGET = 50_000_000        # n_for_error: largest n_sim x n drawn per step of the search (time; v2.8.0 review r1)


def _sv():
    from reporting import SKILL_VERSION
    return SKILL_VERSION


def _is_discrete(d):
    return isinstance(getattr(d, "dist", None), stats.rv_discrete)


def _logp(d, x):
    return d.logpmf(x) if _is_discrete(d) else d.logpdf(x)


def _disc_range(*ds):
    """Integer grid covering the support of discrete distributions up to tails below 1e-17."""
    lo, hi = np.inf, -np.inf
    for d in ds:
        a, b = d.support()
        if not np.isfinite(a):
            a = d.ppf(1e-12)
            while d.cdf(a) > 1e-17:
                a = a - max(10, abs(a))
        if not np.isfinite(b):
            b = d.ppf(1 - 1e-12)
            while d.sf(b) > 1e-17:
                b = b + max(10, abs(b))
        lo, hi = min(lo, a), max(hi, b)
    return np.arange(int(lo), int(hi) + 1)


def _check_dist(d, name):
    if not (hasattr(d, "rvs") and hasattr(d, "dist")):
        raise ValueError(f"{name} must be a scipy frozen distribution, e.g. stats.norm(0, 1) or stats.beta(2, 5)")


def _llr(x, dist0, dist1):
    """log f1(x) - log f0(x), elementwise; +inf where only f1 has mass, -inf where only f0 has."""
    l1 = _logp(dist1, x)
    l0 = _logp(dist0, x)
    with np.errstate(invalid="ignore"):
        out = l1 - l0
    out = np.where(np.isneginf(l0) & np.isfinite(l1), np.inf, out)
    out = np.where(np.isneginf(l1) & np.isfinite(l0), -np.inf, out)
    return out


# =============================================================================
# Divergences
# =============================================================================
def _vec(p):
    p = np.asarray(p, float)
    if p.ndim != 1 or np.any(p < 0) or not np.isfinite(p).all() or abs(p.sum() - 1) > 1e-9:
        raise ValueError("a probability vector must be 1-D, non-negative and sum to 1")
    return p


def _expect_cont(f, a, b, dists=()):
    """Integral of f over (a, b), in pieces between break points of the distributions (v2.8.0 review r1).

    quad over (-inf, inf) maps the line to a finite interval and can miss a narrow density far from 0 (N(100, 1) gave
    a KL of 0); one piece spanning a heavy-tailed core (a Cauchy's spans +-3e11) misses a narrow density inside it.
    Break points: the 1e-12, 1e-6, 0.5, 1 - 1e-6 and 1 - 1e-12 quantiles of each of `dists` within (a, b); finite
    pieces between them, and the two tails beyond. Returns (value, reliable): reliable is False when quad's error
    estimate exceeds 1e-8 of max(1, |value|), or when quad warns on a tail piece (a divergent tail, e.g. the infinite
    KL of a Cauchy against a normal)."""
    import warnings as _w
    bp = set()
    for d in dists:
        for u in (1e-12, 1e-6):
            bp.update((float(d.ppf(u)), float(d.isf(u))))
        bp.add(float(d.median()))
    bp = sorted(x for x in bp if np.isfinite(x) and a < x < b)
    if not bp:
        bp = [float(a), float(b)] if np.isfinite(a) and np.isfinite(b) else [0.0]
    edges = ([a] if a < bp[0] else []) + bp + ([b] if bp[-1] < b else [])
    val = err = 0.0
    ok = True
    for lo_, hi_ in zip(edges[:-1], edges[1:]):
        tail = not (np.isfinite(lo_) and np.isfinite(hi_))
        with _w.catch_warnings(record=True) as rec:
            _w.simplefilter("always", integrate.IntegrationWarning)
            v, e = integrate.quad(f, lo_, hi_, limit=500, epsabs=1e-13, epsrel=1e-11)
        if tail and any(issubclass(r.category, integrate.IntegrationWarning) for r in rec):
            ok = False
        val += v
        err += e
    ok = bool(ok and np.isfinite(val) and np.isfinite(err) and err <= 1e-8 * max(1.0, abs(val)))
    return float(val), ok


_U_EDGES = (0.0, 1e-12, 1e-9, 1e-6, 1e-3, 0.5)


def _expect_under(h, d):
    """E_d[h(X)] = integral over u in (0, 1) of h(F_d^-1(u)), in the quantile scale of d (v2.8.0 review r2).

    In the scale of x, a density whose mass is spread over many decades (Gamma(0.1): 6% of it below 1e-12) or that is
    infinite at an edge of a bounded support (Beta(0.5, 0.5)) gave an error of 1e-5 without warning, or NaN; in the
    quantile scale those are mild singularities at u = 0 or 1. The lower half uses ppf(u), the upper half isf(v),
    v = 1 - u, so the right tail keeps its precision; pieces between 1e-12, 1e-9, 1e-6, 1e-3 and 1/2 on each side.
    Returns (value, reliable): not reliable when quad warns on an outermost piece (a divergent tail, e.g. KL of a
    Cauchy against a normal) or its error estimate exceeds 1e-8 of max(1, |value|)."""
    import warnings as _w
    val = err = 0.0
    ok = True
    for inv in (d.ppf, d.isf):
        g = (lambda u, inv=inv: h(float(inv(u))))
        for i, (lo_, hi_) in enumerate(zip(_U_EDGES[:-1], _U_EDGES[1:])):
            with _w.catch_warnings(record=True) as rec:
                _w.simplefilter("always", integrate.IntegrationWarning)
                v, e = integrate.quad(g, lo_, hi_, limit=500, epsabs=1e-14, epsrel=1e-12)
            if i == 0 and any(issubclass(r.category, integrate.IntegrationWarning) for r in rec):
                ok = False
            val += v
            err += e
    ok = bool(ok and np.isfinite(val) and np.isfinite(err) and err <= 1e-8 * max(1.0, abs(val)))
    return float(val), ok


def _kl_pair(p, q):
    """KL(p || q) for two frozen distributions of the same kind, or two probability vectors."""
    if isinstance(p, np.ndarray):
        mask = p > 0
        if np.any(mask & (q == 0)):
            return np.inf
        return float(np.sum(p[mask] * (np.log(p[mask]) - np.log(q[mask]))))
    if _is_discrete(p):
        x = _disc_range(p)
        lp, lq = p.logpmf(x), q.logpmf(x)
        m = np.isfinite(lp)
        if np.any(m & ~np.isfinite(lq)):
            return np.inf
        return float(np.sum(np.exp(lp[m]) * (lp[m] - lq[m])))
    a, b = p.support()

    def h(x):
        lp = p.logpdf(x)
        if not np.isfinite(lp):
            return 0.0
        lq = q.logpdf(x)
        if not np.isfinite(lq):
            return np.inf
        return lp - lq

    qa, qb = q.support()
    if a < qa or b > qb:
        return np.inf
    val, ok = _expect_under(h, p)
    if not ok or val < -1e-9 * max(1.0, abs(val)):      # KL >= 0 (Gibbs): a negative value is an integration failure
        return float("nan")
    return max(val, 0.0)


def _log_rho(p, q, phi):
    """log of the Chernoff coefficient rho(phi) = integral of p^phi q^(1 - phi)."""
    if isinstance(p, np.ndarray):
        m = (p > 0) & (q > 0)
        if not m.any():
            return -np.inf
        return float(np.log(np.sum(np.exp(phi * np.log(p[m]) + (1 - phi) * np.log(q[m])))))
    if _is_discrete(p):
        x = _disc_range(p, q)
        v = phi * p.logpmf(x) + (1 - phi) * q.logpmf(x)
        v = v[np.isfinite(v)]
        return float(np.log(np.sum(np.exp(v)))) if len(v) else -np.inf
    a = max(p.support()[0], q.support()[0])
    b = min(p.support()[1], q.support()[1])
    if not a < b:
        return -np.inf

    def f(x):
        v = phi * p.logpdf(x) + (1 - phi) * q.logpdf(x)
        return float(np.exp(v)) if np.isfinite(v) else 0.0

    val, ok = _expect_cont(f, a, b, (p, q))
    if not ok:
        return float("nan")
    return float(np.log(val)) if val > 0 else -np.inf


def divergence(p, q, show: bool = True) -> dict:
    """Divergences between two distributions: KL both ways, Jensen-Shannon, Chernoff information and Bhattacharyya.

    Returns a dict (v2.8.0: keys exact):
      "kl_pq", "kl_qp"   float, KL(p || q) = E_p[log p/q] and KL(q || p), in nats (inf if p has mass where q has none)
      "js"               float, Jensen-Shannon divergence, in nats (between 0 and log 2 = 0.6931)
      "chernoff"         float, Chernoff information C = -log min_phi rho(phi), rho(phi) = integral p^phi q^(1-phi)
      "chernoff_phi"     float, the minimizing phi in (0, 1)
      "bhattacharyya"    float, -log rho(1/2) (a lower bound on C)
      "used"             dict, keys ["kind", "skill_version"]
      "warnings"         list of dicts with keys ["level", "message", "action"]

    p, q : two scipy frozen distributions of the same kind (both continuous or both discrete), or two probability
           vectors over the same categories. Continuous KL and JS are expectations computed in the quantile scale
           of the distribution they are taken under (scipy quad); the Chernoff coefficient is integrated in pieces
           between quantiles of both, so a narrow density far from 0 or one spread over many decades is not
           missed, and C is never below the Bhattacharyya distance; discrete ones are
           summed over the support (out to tails below 1e-17 when it is infinite). A value whose integral does not
           converge (e.g. the infinite KL of a Cauchy against a normal) is NaN with a warning, never a guess.
           Precision is limited by rounding in the densities: about eps * |location| / scale relative, and, when a
           density is infinite at an edge of a bounded support, by the mass within one ulp of that edge
           (KL(Beta(0.5, 0.5) || Beta(2, 2)): 3e-7).
    Uses in this skill: KL(f0 || f1) and KL(f1 || f0) are the expected log-likelihood-ratio steps of an SPRT under
    each hypothesis (sprt_design) and Stein's exponents of the error of a fixed-sample test; C is the exponent of the
    smallest attainable average error with equal priors, P_e(n) ~ exp(-n C) (Chernoff 1952); JS is symmetric and
    bounded, so it ranks how far apart several pairs are. Checks printed: Bhattacharyya <= C <=
    KL_pq KL_qp / (KL_pq + KL_qp) <= min(KL_pq, KL_qp) (the bounds in QuantEcon, "Statistical Divergence Measures");
    a violation means the numerical integration failed."""
    from reporting import _finding, _print_findings
    if isinstance(p, (list, tuple, np.ndarray)) or isinstance(q, (list, tuple, np.ndarray)):
        p, q = _vec(p), _vec(q)
        if p.shape != q.shape:
            raise ValueError("probability vectors must have the same length")
        kind = "vectors"
    else:
        _check_dist(p, "p")
        _check_dist(q, "q")
        if _is_discrete(p) != _is_discrete(q):
            raise ValueError("p and q must both be continuous or both discrete")
        kind = "discrete" if _is_discrete(p) else "continuous"
    kl_pq, kl_qp = _kl_pair(p, q), _kl_pair(q, p)
    # Jensen-Shannon with the mixture m = (p + q)/2
    if kind == "vectors":
        mvec = (p + q) / 2
        js = 0.5 * _kl_pair(p, mvec) + 0.5 * _kl_pair(q, mvec)
    elif kind == "discrete":
        x = _disc_range(p, q)
        pp, qq = p.pmf(x), q.pmf(x)
        mm = (pp + qq) / 2
        js = 0.5 * _kl_pair(pp, mm) + 0.5 * _kl_pair(qq, mm)
    else:
        a = min(p.support()[0], q.support()[0])
        b = max(p.support()[1], q.support()[1])

        def half(d, o):
            # log(d / m) with m = (d + o)/2, in logs: log 2 - log(1 + o/d); bounded above by log 2
            def h(x):
                ld, lo_ = d.logpdf(x), o.logpdf(x)
                if not np.isfinite(lo_):
                    return np.log(2.0)
                return float(np.log(2.0) - np.logaddexp(0.0, lo_ - ld))
            return _expect_under(h, d)

        (j1, ok1), (j2, ok2) = half(p, q), half(q, p)
        js = 0.5 * (j1 + j2) if ok1 and ok2 else float("nan")
    bhat = -_log_rho(p, q, 0.5)
    if np.isnan(bhat):
        chern = phi = float("nan")
    else:
        # C = max over phi of -log rho(phi) >= the value at 1/2 (Bhattacharyya). A NaN anywhere in the search makes C
        # NaN (it gave C < B: the bounded search read NaN as a value; v2.8.0 review r2)
        seen_nan = []

        def lr(ph):
            v = _log_rho(p, q, ph)
            if np.isnan(v):
                seen_nan.append(ph)
                return np.inf
            return v
        grid = np.linspace(0.05, 0.95, 19)
        vals = [lr(g) for g in grid]
        i = int(np.argmin(vals))
        res = optimize.minimize_scalar(lr, bounds=(grid[max(i - 1, 0)] if i > 0 else 1e-9,
                                                   grid[min(i + 1, 18)] if i < 18 else 1 - 1e-9),
                                       method="bounded", options={"xatol": 1e-10})
        cands = [(float(res.fun), float(res.x)), (vals[i], float(grid[i])), (-bhat, 0.5)]
        best = min(cands)
        chern, phi = (float("nan"), float("nan")) if seen_nan else (-best[0], best[1])
    warns = []
    bad = [nm for nm, v in (("KL(p||q)", kl_pq), ("KL(q||p)", kl_qp), ("JS", js), ("Chernoff", chern),
                            ("Bhattacharyya", bhat)) if np.isnan(v)]
    if bad:
        warns.append(_finding("warning", f"{', '.join(bad)} not computed: the numerical integral did not converge or "
                              "gave a negative KL (a KL is infinite when p has much heavier tails than q, e.g. Cauchy "
                              "against normal)", "compute the divergence in closed form, or check the tails of p and q; "
                              "do not use the NaN values"))
    if np.isfinite(kl_pq) and np.isfinite(kl_qp) and np.isfinite(chern) and np.isfinite(bhat):
        upper = kl_pq * kl_qp / (kl_pq + kl_qp) if kl_pq + kl_qp > 0 else 0.0
        tol = 1e-7 * max(1.0, upper)
        if not (bhat <= chern + tol and chern <= upper + tol and upper <= min(kl_pq, kl_qp) + tol):
            warns.append(_finding("warning", f"the divergences break the bound Bhattacharyya <= C <= KL KL/(KL + KL) "
                                  f"({bhat:.6g}, {chern:.6g}, {upper:.6g}): the numerical integration failed",
                                  "check the distributions (support, parameters) or compute the divergences in closed form"))
    if np.isinf(kl_pq) or np.isinf(kl_qp):
        warns.append(_finding("notice", "one KL divergence is infinite: one distribution puts mass where the other has "
                              "none, so a single observation there settles the test",
                              "report it; the SPRT then stops at the first such observation"))
    out = {"kl_pq": kl_pq, "kl_qp": kl_qp, "js": js, "chernoff": chern, "chernoff_phi": phi, "bhattacharyya": bhat,
           "used": {"kind": kind, "skill_version": _sv()}, "warnings": warns}
    if show:
        print(f"\nDivergences ({kind}, nats): KL(p||q) = {kl_pq:.6g}, KL(q||p) = {kl_qp:.6g}, JS = {js:.6g} "
              f"(max log 2 = 0.6931)")
        print(f"Chernoff information C = {chern:.6g} at phi = {phi:.4f}; Bhattacharyya distance = {bhat:.6g}")
        _print_findings(warns)
    return out


def _stationary(P):
    w, v = np.linalg.eig(P.T)
    x = np.real(v[:, np.argmin(np.abs(w - 1))])
    return x / x.sum()


def _lyap(A, Sig):
    k = A.shape[0]
    vecV = np.linalg.solve(np.eye(k * k) - np.kron(A, A), Sig.reshape(-1, order="F"))
    V = vecV.reshape(k, k, order="F")
    return (V + V.T) / 2


def _var_model(M, name):
    if isinstance(M, dict):
        A = np.atleast_2d(np.asarray(M["A"], float))
        Sig = (np.atleast_2d(np.asarray(M["Sigma"], float)) if "Sigma" in M
               else np.atleast_2d(np.asarray(M["C"], float)) @ np.atleast_2d(np.asarray(M["C"], float)).T)
    else:
        A, Sig = (np.atleast_2d(np.asarray(z, float)) for z in M)
    if np.max(np.abs(np.linalg.eigvals(A))) >= 1:
        raise ValueError(f"{name}: the VAR is not stable (an eigenvalue of A has modulus >= 1)")
    return A, Sig


def _is_var_spec(M):
    """A VAR is a dict, or a pair (A, Sigma); a transition matrix may also come as nested lists or tuples (review r1:
    ((0.9, 0.1), (0.2, 0.8)) was taken for a VAR and crashed)."""
    if isinstance(M, dict):
        return True
    if not isinstance(M, tuple):
        return False
    try:
        arr = np.asarray(M, float)
    except (ValueError, TypeError):
        return True                                  # ragged: (A, Sigma) of different shapes
    is_markov = (arr.ndim == 2 and arr.shape[0] == arr.shape[1] and arr.shape[0] >= 2 and np.all(arr >= 0)
                 and np.all(np.abs(arr.sum(1) - 1) <= 1e-9))
    return not is_markov


def divergence_rate(P, Q, show: bool = True) -> dict:
    """KL divergence rate between two stationary processes, both ways: finite Markov chains or Gaussian VAR(1).

    Returns a dict (v2.8.0: keys exact):
      "rate_pq", "rate_qp"  float, lim (1/T) E_P[log of the likelihood ratio of P against Q over T steps] and the
                            reverse, in nats per step (inf when P allows a transition Q forbids)
      "kind"                "markov" or "var"
      "stationary_p", "stationary_q"  ndarray, stationary distribution (Markov) or stationary covariance (VAR)
      "used"                dict, keys ["kind", "skill_version"]
      "warnings"            list of dicts with keys ["level", "message", "action"]

    Markov: P and Q are square transition matrices (rows sum to 1); the rate is sum_i pi_i sum_j P_ij log(P_ij/Q_ij)
    with pi the stationary distribution of P (QuantEcon, "Likelihood Ratio Processes"). VAR(1): x_t+1 = A x_t + C w_t+1
    with w standard normal; pass {"A": A, "C": C}, {"A": A, "Sigma": C C'} or (A, Sigma). A transition matrix may be
    an array or nested lists or tuples (a square tuple whose rows sum to 1 is read as a Markov chain). The rate is
    0.5 [log det Sq - log det Sp - k + tr(Sq^-1 Sp) + tr((Ap - Aq)' Sq^-1 (Ap - Aq) Vp)], Vp the stationary covariance
    of x under P (solved from V = A V A' + Sigma). The rate is the drift of log L_t per step: with data from P, an
    SPRT of Q against P needs about log(A)/rate steps, and the error of a fixed-length test falls like exp(-T rate)."""
    from reporting import _finding, _print_findings
    warns = []
    if _is_var_spec(P) or _is_var_spec(Q):
        kind = "var"
        Ap, Sp = _var_model(P, "P")
        Aq, Sq = _var_model(Q, "Q")
        if Ap.shape != Aq.shape or Sp.shape != Sq.shape or Ap.shape != Sp.shape:
            raise ValueError("the two VARs must have the same dimension")
        k = Ap.shape[0]
        Vp, Vq = _lyap(Ap, Sp), _lyap(Aq, Sq)

        def rate(A1, S1, V1, A2, S2):
            S2i = np.linalg.inv(S2)
            dA = A1 - A2
            return float(0.5 * (np.linalg.slogdet(S2)[1] - np.linalg.slogdet(S1)[1] - k + np.trace(S2i @ S1)
                                + np.trace(dA.T @ S2i @ dA @ V1)))

        r_pq, r_qp = rate(Ap, Sp, Vp, Aq, Sq), rate(Aq, Sq, Vq, Ap, Sp)
        st_p, st_q = Vp, Vq
    else:
        kind = "markov"
        P, Q = np.asarray(P, float), np.asarray(Q, float)
        for M, nm in ((P, "P"), (Q, "Q")):
            if M.ndim != 2 or M.shape[0] != M.shape[1] or np.any(M < 0) or np.any(np.abs(M.sum(1) - 1) > 1e-9):
                raise ValueError(f"{nm} must be a square transition matrix with rows summing to 1")
        if P.shape != Q.shape:
            raise ValueError("P and Q must have the same number of states")
        st_p, st_q = _stationary(P), _stationary(Q)

        def rate(M1, M2, pi):
            if np.any((M1 > 0) & (M2 == 0)):
                return np.inf
            m = M1 > 0
            L = np.zeros_like(M1)
            L[m] = np.log(M1[m] / M2[m])
            return float(np.sum(pi[:, None] * M1 * L))

        r_pq, r_qp = rate(P, Q, st_p), rate(Q, P, st_q)
        w = np.linalg.eigvals(P)
        if np.sum(np.abs(w - 1) < 1e-9) > 1:
            warns.append(_finding("warning", "P has more than one stationary distribution (it is reducible): the rate "
                                  "depends on where the chain starts", "compute the rate on each closed class"))
    if not (np.isfinite(r_pq) and np.isfinite(r_qp)):
        warns.append(_finding("notice", "an infinite rate: one model allows a transition the other forbids",
                              "one such transition settles the comparison"))
    out = {"rate_pq": r_pq, "rate_qp": r_qp, "kind": kind, "stationary_p": st_p, "stationary_q": st_q,
           "used": {"kind": kind, "skill_version": _sv()}, "warnings": warns}
    if show:
        print(f"\nKL divergence rate ({kind}, nats per step): P against Q = {r_pq:.6g}; Q against P = {r_qp:.6g}")
        _print_findings(warns)
    return out


# =============================================================================
# Wald's sequential probability ratio test
# =============================================================================
def _thresholds(alpha, beta):
    if not (0 < alpha < 1 and 0 < beta < 1 and alpha + beta < 1):
        raise ValueError("alpha and beta must be in (0, 1) with alpha + beta < 1")
    return float(np.log((1 - beta) / alpha)), float(np.log(beta / (1 - alpha)))


def _simulate_sprt(dist0, dist1, truth, n_runs, logA, logB, max_n, rng):
    """Stopping times and decisions (True = reject H0) of n_runs SPRTs with data from `truth`."""
    N = np.zeros(n_runs, dtype=np.int64)
    rej = np.zeros(n_runs, dtype=bool)
    done = np.zeros(n_runs, dtype=bool)
    L = np.zeros(n_runs)
    t = 0
    while not done.all() and t < max_n:
        idx = np.flatnonzero(~done)
        step = max(1, min(64, _MAX_BLOCK // max(len(idx), 1), max_n - t))
        x = truth.rvs(size=(len(idx), step), random_state=rng)
        inc = _llr(x, dist0, dist1)
        path = L[idx, None] + np.cumsum(inc, axis=1)
        up = path >= logA
        dn = path <= logB
        hit = up | dn
        any_hit = hit.any(axis=1)
        first = np.where(any_hit, hit.argmax(axis=1), step - 1)
        rows = np.arange(len(idx))
        N[idx] = t + first + 1
        rej[idx] = any_hit & up[rows, first]
        done[idx] = any_hit
        L[idx] = path[rows, first]
        t += step
    return N, rej, done


def sprt_design(dist0, dist1, alpha: float = ALPHA, beta: float = 0.10, n_sim: int = 20000, max_n: int = 100_000,
                seed: int = 0, show: bool = True) -> dict:
    """Design of Wald's sequential probability ratio test of H0: f = dist0 against H1: f = dist1.

    Returns a dict (v2.8.0: keys exact):
      "log_A", "log_B"      float, log thresholds: reject H0 when log L_n >= log A = log((1 - beta)/alpha), accept it
                            when log L_n <= log B = log(beta/(1 - alpha)) (Wald's approximation)
      "wald_bound_alpha", "wald_bound_beta"  float, the guarantees alpha' <= alpha/(1 - beta), beta' <= beta/(1 - alpha)
      "kl_01", "kl_10"      float, KL(f0 || f1) and KL(f1 || f0): the mean step of log L under H0 (negative) and H1
      "approx_en0", "approx_en1"  float, Wald's approximate E[N] under H0 and H1 (no overshoot)
      "sim_alpha", "sim_beta"  float, simulated error rates; "sim_alpha_se", "sim_beta_se" their Monte Carlo SEs
      "sim_en0", "sim_en1", "sim_en"  float, simulated mean stopping times under H0, H1 and over both halves
      "sim_max_n"           int, the longest simulated run
      "used"                dict, keys ["alpha", "beta", "n_sim", "max_n", "seed", "skill_version"]
      "warnings"            list of dicts with keys ["level", "message", "action"]

    n_sim runs, half with data from dist0 and half from dist1 (as in QuantEcon, "A Problem that Stumped Milton
    Friedman", where Beta(2,5) against Beta(5,2) with alpha 0.05, beta 0.10 stops after 1.59 draws on average).
    The simulated errors are usually below the targets because log L jumps past the thresholds (overshoot); the
    approximate E[N] ignores the overshoot and understates the sample size when the steps are large.
    A run longer than max_n draws is stopped and counted as undecided (a finding)."""
    from reporting import _finding, _print_findings
    _check_dist(dist0, "dist0")
    _check_dist(dist1, "dist1")
    logA, logB = _thresholds(alpha, beta)
    if int(n_sim) != n_sim or n_sim < 2:
        raise ValueError("n_sim must be an integer >= 2")
    dv = divergence(dist0, dist1, show=False)
    kl01, kl10 = dv["kl_pq"], dv["kl_qp"]
    en0 = (alpha * logA + (1 - alpha) * logB) / (-kl01) if np.isfinite(kl01) and kl01 > 0 else np.nan
    en1 = ((1 - beta) * logA + beta * logB) / kl10 if np.isfinite(kl10) and kl10 > 0 else np.nan
    if np.isnan(kl01) or np.isnan(kl10):
        dv_warn = [w for w in dv["warnings"] if "not computed" in w["message"]]
    else:
        dv_warn = []
    rng = np.random.default_rng(seed)
    h0 = n_sim // 2
    N0, R0, D0 = _simulate_sprt(dist0, dist1, dist0, h0, logA, logB, max_n, rng)
    N1, R1, D1 = _simulate_sprt(dist0, dist1, dist1, n_sim - h0, logA, logB, max_n, rng)
    a_hat = float(np.mean(R0[D0])) if D0.any() else np.nan
    b_hat = float(np.mean(~R1[D1])) if D1.any() else np.nan
    se_a = float(np.sqrt(a_hat * (1 - a_hat) / max(D0.sum(), 1)))
    se_b = float(np.sqrt(b_hat * (1 - b_hat) / max(D1.sum(), 1)))
    warns = list(dv_warn)
    if dv_warn:
        warns.append(_finding("warning", "Wald's approximate E[N] is not computed: a KL divergence could not be "
                              "integrated", "use the simulated stopping times (sim_en0, sim_en1)"))
    und = int((~D0).sum() + (~D1).sum())
    if und:
        warns.append(_finding("warning", f"{und} of {n_sim} simulated runs reached max_n = {max_n} without a decision",
                              "raise max_n; the reported errors and stopping times leave those runs out"))
    bA, bB = alpha / (1 - beta), beta / (1 - alpha)
    for nm, val, se, bound in (("type I", a_hat, se_a, bA), ("type II", b_hat, se_b, bB)):
        if np.isfinite(val) and val - 3 * se > bound:
            warns.append(_finding("warning", f"the simulated {nm} error {val:.4f} exceeds Wald's bound {bound:.4f} by "
                                  "more than 3 Monte Carlo SEs", "check that the observations are independent draws "
                                  "from dist0 or dist1 (Wald's bounds assume it)"))
    if np.isfinite(a_hat) and np.isfinite(b_hat) and (a_hat < 0.5 * alpha or b_hat < 0.5 * beta):
        warns.append(_finding("notice", f"the simulated errors ({a_hat:.4f}, {b_hat:.4f}) are well below the targets "
                              f"({alpha}, {beta}): log L jumps past the thresholds (overshoot)",
                              "report the simulated errors, not the targets, as the test's operating characteristics"))
    out = {"log_A": logA, "log_B": logB, "wald_bound_alpha": bA, "wald_bound_beta": bB, "kl_01": kl01, "kl_10": kl10,
           "approx_en0": float(en0), "approx_en1": float(en1), "sim_alpha": a_hat, "sim_beta": b_hat,
           "sim_alpha_se": se_a, "sim_beta_se": se_b, "sim_en0": float(N0[D0].mean()) if D0.any() else np.nan,
           "sim_en1": float(N1[D1].mean()) if D1.any() else np.nan,
           "sim_en": float(np.concatenate([N0[D0], N1[D1]]).mean()), "sim_max_n": int(max(N0.max(), N1.max())),
           "used": {"alpha": alpha, "beta": beta, "n_sim": int(n_sim), "max_n": int(max_n), "seed": seed,
                    "skill_version": _sv()}, "warnings": warns}
    if show:
        print(f"\nSPRT: reject H0 when log L >= {logA:.4f}, accept H0 when log L <= {logB:.4f} "
              f"(targets alpha = {alpha}, beta = {beta})")
        print(f"Wald's guarantees: alpha' <= {bA:.4f}, beta' <= {bB:.4f}")
        print(f"KL(f0||f1) = {kl01:.4g}, KL(f1||f0) = {kl10:.4g}; Wald's approximate E[N]: {en0:.3g} under H0, "
              f"{en1:.3g} under H1")
        print(f"Simulated ({n_sim} runs, seed {seed}): type I = {a_hat:.4f} (MC SE {se_a:.4f}), type II = {b_hat:.4f} "
              f"(MC SE {se_b:.4f}); mean N = {out['sim_en0']:.3g} under H0, {out['sim_en1']:.3g} under H1, "
              f"{out['sim_en']:.3g} overall; longest run {out['sim_max_n']}")
        _print_findings(warns)
    return out


def sprt(x, dist0, dist1, alpha: float = ALPHA, beta: float = 0.10, show: bool = True) -> dict:
    """Wald's SPRT on observations in arrival order: stop at the first crossing of a threshold.

    Returns a dict (v2.8.0: keys exact):
      "decision"   "reject H0", "accept H0" or "continue" (no threshold crossed yet)
      "n_stop"     int, observations used at the decision (None with "continue")
      "n_seen"     int, observations passed in
      "llr"        Series, log L_n after each observation (index 1..n_seen)
      "log_A", "log_B"  float, thresholds
      "used"       dict, keys ["alpha", "beta", "skill_version"]
      "warnings"   list of dicts with keys ["level", "message", "action"]

    x: observations in the order they arrived (missing values are refused). The decision is the first crossing;
    observations after it do not change it (they are reported in n_seen). "continue" means no decision: keep
    sampling; stopping there and running a fixed-sample test on the same data is the peeking the rules forbid."""
    from reporting import _finding, _print_findings
    _check_dist(dist0, "dist0")
    _check_dist(dist1, "dist1")
    logA, logB = _thresholds(alpha, beta)
    xa = np.asarray(pd.Series(x).to_numpy(), float)
    if xa.ndim != 1 or len(xa) == 0:
        raise ValueError("x must be a non-empty 1-D sequence of observations")
    if not np.isfinite(xa).all():
        raise ValueError("x has missing or infinite values: drop them in arrival order before the test")
    path = np.cumsum(_llr(xa, dist0, dist1))
    hit = np.flatnonzero((path >= logA) | (path <= logB))
    warns = []
    if len(hit):
        i = int(hit[0])
        decision = "reject H0" if path[i] >= logA else "accept H0"
        n_stop = i + 1
        if len(xa) > n_stop:
            warns.append(_finding("notice", f"the test stopped at observation {n_stop}; {len(xa) - n_stop} later "
                                  "observations do not enter the decision",
                                  "report the decision at the stopping point, not a test on all the data"))
    else:
        decision, n_stop = "continue", None
        warns.append(_finding("notice", f"no threshold crossed after {len(xa)} observations (log L = {path[-1]:.4g})",
                              "keep sampling; do not switch to a fixed-sample test on these data"))
    out = {"decision": decision, "n_stop": n_stop, "n_seen": len(xa),
           "llr": pd.Series(path, index=pd.RangeIndex(1, len(xa) + 1), name="log L"), "log_A": logA, "log_B": logB,
           "used": {"alpha": alpha, "beta": beta, "skill_version": _sv()}, "warnings": warns}
    if show:
        print(f"\nSPRT: {decision}" + (f" after {n_stop} observations" if n_stop else "")
              + f" (log A = {logA:.4f}, log B = {logB:.4f}; log L at the end = {path[-1]:.4f})")
        _print_findings(warns)
    return out


# =============================================================================
# Peeking at a fixed-sample test
# =============================================================================
def _amr_size(z, K, two_sided, npt=None):
    """P(some look rejects | H0) for K equally spaced looks at critical value z: Armitage, McPherson and Rowe (1969).

    S_k = sum of k standard normal increments, Z_k = S_k / sqrt(k); the test continues while |S_k| < z sqrt(k)
    (two-sided) or S_k < z sqrt(k). The sub-density of S_k on the continuation interval is carried by convolution
    with the N(0, 1) increment, by Simpson's rule on an odd grid of npt points per look (default 401 two-sided,
    1201 one-sided, whose interval [-9 sqrt(k), z sqrt(k)] is longer)."""
    if K == 1:                                       # the test itself
        return float(2 * stats.norm.sf(z) if two_sided else stats.norm.sf(z))
    if npt is None:
        npt = 401 if two_sided else 1201
    lo_c = (lambda k: -z * np.sqrt(k)) if two_sided else (lambda k: -9.0 * np.sqrt(k))
    x = np.linspace(lo_c(1), z, npt)
    f = stats.norm.pdf(x)
    w = np.ones(npt)
    w[1:-1:2], w[2:-1:2] = 4.0, 2.0
    for k in range(2, K + 1):
        x_new = np.linspace(lo_c(k), z * np.sqrt(k), npt)
        f = stats.norm.pdf(x_new[:, None] - x[None, :]) @ (w * (x[1] - x[0]) / 3.0 * f)
        x = x_new
    return float(1.0 - np.sum(w * (x[1] - x[0]) / 3.0 * f))


def peeking_size(looks: int = 10, alpha: float = ALPHA, two_sided: bool = True, show: bool = True) -> dict:
    """Type I error of a fixed-sample z test computed at `looks` equally spaced interim points and stopped at the
    first p < alpha, and the per-look level that would keep the overall error at alpha (Pocock 1977).

    Returns a dict (v2.8.0: keys exact):
      "size"            float, P(at least one look rejects | H0) when each look uses level alpha
      "per_look_alpha"  float, the constant per-look nominal level with overall size alpha (Pocock's boundary)
      "per_look_z"      float, the matching critical value
      "looks"           int
      "used"            dict, keys ["looks", "alpha", "two_sided", "method", "skill_version"]
      "warnings"        list of dicts with keys ["level", "message", "action"]

    Under H0 the z statistics at information fractions k/K are a scaled Brownian motion (corr(Z_i, Z_j) =
    sqrt(i/j), i < j). The size is computed exactly by the recursion of Armitage, McPherson and Rowe (1969): the
    density of the running sum is carried from look to look by numerical integration (against a grid ten times finer,
    up to 50 looks: within 6e-9 two-sided and 3e-9 one-sided; v2.8.0 measurement), not by a simulation of data.
    Classical values (Armitage, McPherson and Rowe 1969): about 8% with 2 looks, 14% with 5, 19% with 10 for alpha = 5% two-sided."""
    from reporting import _finding, _print_findings
    if int(looks) != looks or looks < 1 or looks > 50:
        raise ValueError("looks must be an integer from 1 to 50")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    K = int(looks)

    def size_at(z):
        return _amr_size(z, K, two_sided)

    z0 = stats.norm.isf(alpha / 2) if two_sided else stats.norm.isf(alpha)
    size = size_at(z0)
    zp = z0 if K == 1 else optimize.brentq(lambda z: size_at(z) - alpha, z0, z0 + 3, xtol=1e-7)
    per = 2 * stats.norm.sf(zp) if two_sided else stats.norm.sf(zp)
    warns = []
    if K > 1:
        warns.append(_finding("warning", f"looking {K} times at level {alpha} and stopping at the first rejection "
                              f"gives a type I error of {size:.3f}, not {alpha}",
                              f"fix the looks in advance and test each at {per:.4f} (Pocock), use an SPRT "
                              "(sequential.sprt) or an alpha-spending boundary, or report that the p-value is not valid"))
    out = {"size": float(size), "per_look_alpha": float(per), "per_look_z": float(zp), "looks": K,
           "used": {"looks": K, "alpha": alpha, "two_sided": two_sided, "method": "Armitage-McPherson-Rowe recursion",
                    "skill_version": _sv()}, "warnings": warns}
    if show:
        print(f"\nPeeking: {K} equally spaced looks at level {alpha} ({'two' if two_sided else 'one'}-sided): "
              f"overall type I error = {size:.4f}")
        print(f"Per-look level for overall {alpha} (Pocock): {per:.5f} (|z| > {zp:.4f})")
        _print_findings(warns)
    return out


# =============================================================================
# Sample size and small error probabilities
# =============================================================================
def _llr_moments(dist0, dist1, under):
    """Mean and variance of log f1/f0 under `under` (numerical)."""
    if _is_discrete(under):
        x = _disc_range(under)
        w = under.pmf(x)
        r = _llr(x, dist0, dist1)
        m = w > 0
        mu = float(np.sum(w[m] * r[m]))
        return mu, float(np.sum(w[m] * (r[m] - mu) ** 2))
    def f(x, k):
        if not np.isfinite(under.logpdf(x)):
            return 0.0
        return float(_llr(np.array([x]), dist0, dist1)[0] ** k)

    # in the quantile scale of `under`, like the KL (v2.8.0 review r2)
    m1, ok1 = _expect_under(lambda x: f(x, 1), under)
    m2, ok2 = _expect_under(lambda x: f(x, 2), under)
    if not (ok1 and ok2):
        return float("nan"), float("nan")
    return float(m1), float(m2 - m1 ** 2)


def n_for_error(dist0, dist1, alpha: float = ALPHA, beta: float = 0.10, simulate: bool = False, n_sim: int = 20000,
                seed: int = 0, show: bool = True) -> dict:
    """Fixed sample size of the likelihood-ratio (Neyman-Pearson) test of dist0 against dist1 with type I error alpha
    and type II error beta, next to what an SPRT needs on average.

    Returns a dict (v2.8.0: keys exact):
      "n_normal"     int, from the normal approximation of the sum of n log-likelihood ratios:
                     sqrt(n) = (z_(1-alpha) s0 + z_(1-beta) s1) / (KL(f0||f1) + KL(f1||f0)), s0, s1 the SDs of one
                     log ratio under each hypothesis (exact when both are normal with the same variance)
      "n_simulated"  int or None, the smallest n whose simulated test (threshold at the simulated 1 - alpha quantile
                     under H0) has type II error <= beta (with simulate=True)
      "sprt_en0", "sprt_en1"  float, Wald's approximate expected sample sizes of the SPRT under H0 and H1
      "kl_01", "kl_10", "chernoff"  float, the exponents: Stein (the other error falls like exp(-n KL)) and Chernoff
      "used"         dict, keys ["alpha", "beta", "simulate", "n_sim", "seed", "skill_version"]
      "warnings"     list of dicts with keys ["level", "message", "action"]

    The SPRT stops, on average, after far fewer observations than the fixed-sample test with the same errors
    (Wald 1947): that saving is the reason to test sequentially, and the price is a random sample size."""
    from reporting import _finding, _print_findings
    _check_dist(dist0, "dist0")
    _check_dist(dist1, "dist1")
    logA, logB = _thresholds(alpha, beta)
    dv = divergence(dist0, dist1, show=False)
    kl01, kl10 = dv["kl_pq"], dv["kl_qp"]
    warns = []
    if np.isnan(kl01) or np.isnan(kl10):
        raise ValueError("a KL divergence could not be integrated numerically (see sequential.divergence): "
                         "no sample size is computed")
    if not (np.isfinite(kl01) and np.isfinite(kl10)):
        raise ValueError("a KL divergence is infinite: one observation can settle the test; no sample-size formula applies")
    _, v0 = _llr_moments(dist0, dist1, dist0)
    _, v1 = _llr_moments(dist0, dist1, dist1)
    if not (np.isfinite(v0) and np.isfinite(v1)):
        raise ValueError("the variance of the log-likelihood ratio could not be integrated numerically: "
                         "no sample size is computed")
    za, zb = stats.norm.isf(alpha), stats.norm.isf(beta)
    n_norm = int(np.ceil(((za * np.sqrt(v0) + zb * np.sqrt(v1)) / (kl01 + kl10)) ** 2))
    n_sim_res = None
    if simulate:
        rng = np.random.default_rng(seed)
        hi = max(4 * n_norm, 8)

        def sums(dist, n):
            # in blocks of at most _MAX_BLOCK draws (v2.8.0 review r1: one n_sim x n array ran out of memory)
            rows = max(1, _MAX_BLOCK // n)
            out_ = np.empty(n_sim)
            for i0 in range(0, n_sim, rows):
                i1 = min(n_sim, i0 + rows)
                out_[i0:i1] = np.sum(_llr(dist.rvs(size=(i1 - i0, n), random_state=rng), dist0, dist1), axis=1)
            return out_

        def type2(n):
            c = np.quantile(sums(dist0, n), 1 - alpha)
            return float(np.mean(sums(dist1, n) <= c))

        if n_sim * hi > _SIM_BUDGET:
            simulate = False
            warns.append(_finding("notice", f"simulation skipped: about {n_sim * hi:.3g} draws per step of the search "
                                  f"(n_sim {n_sim} x n up to {hi}) exceed the budget of {_SIM_BUDGET:.0e}",
                                  "rely on n_normal (at this n the sum of log ratios is close to normal) or lower n_sim"))
        else:
            lo_n = 1
            while type2(hi) > beta and hi < 1_000_000 and n_sim * 2 * hi <= _SIM_BUDGET:
                hi *= 2
            while lo_n < hi:
                mid = (lo_n + hi) // 2
                if type2(mid) <= beta:
                    hi = mid
                else:
                    lo_n = mid + 1
            n_sim_res = int(lo_n)
        if n_sim_res is not None and abs(n_sim_res - n_norm) > max(2, 0.2 * n_norm):
            warns.append(_finding("notice", f"the normal approximation ({n_norm}) and the simulation ({n_sim_res}) "
                                  "differ by more than 20%: the log-likelihood ratio is far from normal at this n",
                                  "use the simulated n"))
    en0 = (alpha * logA + (1 - alpha) * logB) / (-kl01)
    en1 = ((1 - beta) * logA + beta * logB) / kl10
    out = {"n_normal": n_norm, "n_simulated": n_sim_res, "sprt_en0": float(en0), "sprt_en1": float(en1),
           "kl_01": kl01, "kl_10": kl10, "chernoff": dv["chernoff"],
           "used": {"alpha": alpha, "beta": beta, "simulate": simulate, "n_sim": int(n_sim), "seed": seed,
                    "skill_version": _sv()}, "warnings": warns}
    if show:
        print(f"\nFixed-sample likelihood-ratio test with alpha = {alpha}, beta = {beta}: n = {n_norm} (normal "
              "approximation)" + (f", n = {n_sim_res} (simulated)" if n_sim_res is not None else ""))
        print(f"SPRT, Wald's approximate expected sample size: {en0:.3g} under H0, {en1:.3g} under H1")
        print(f"Exponents: KL(f0||f1) = {kl01:.4g}, KL(f1||f0) = {kl10:.4g}, Chernoff = {dv['chernoff']:.4g}")
        _print_findings(warns)
    return out


def importance_tail_prob(dist0, dist1, n: int, threshold: float, error: str = "type1", draws: int = 10000,
                         seed: int = 0, alpha: float = ALPHA, show: bool = True) -> dict:
    """Small error probability of a fixed-sample likelihood-ratio test by importance sampling.

    Returns a dict (v2.8.0: keys exact):
      "estimand"   "P0(log L_n >= threshold)" (error="type1") or "P1(log L_n <= threshold)" (error="type2")
      "estimate", "se", "ci_low", "ci_high"  float, importance-sampling estimate, its SE and normal CI
      "naive", "naive_se"  float, plain Monte Carlo from the true distribution with the same number of draws
      "hits_naive"  int, naive draws in the event (0 means the naive estimate is useless)
      "variance_ratio"  float, naive variance / importance-sampling variance (inf when the naive one is 0)
      "ess"        float, effective sample size of the importance weights inside the event
      "n", "draws"  int
      "used"       dict, keys ["error", "n", "threshold", "draws", "seed", "skill_version"]
      "warnings"   list of dicts with keys ["level", "message", "action"]

    log L_n = sum of n terms log f1(x_i)/f0(x_i). Type I error P0(log L_n >= c) = E1[exp(-log L_n) 1{log L_n >= c}]:
    draw from f1, where the event is common, and weight by f0/f1 (Wald's change of measure; QuantEcon, "Mean of a
    Likelihood Ratio Process"). Type II error P1(log L_n <= c) = E0[exp(log L_n) 1{log L_n <= c}]. Rare events
    (1e-6 and below) are out of reach of plain simulation with a few thousand draws."""
    from reporting import _finding, _print_findings
    _check_dist(dist0, "dist0")
    _check_dist(dist1, "dist1")
    if error not in ("type1", "type2"):
        raise ValueError("error must be 'type1' or 'type2'")
    if int(n) != n or n < 1 or int(draws) != draws or draws < 100:
        raise ValueError("n must be a positive integer and draws an integer >= 100")
    n, draws = int(n), int(draws)
    rng = np.random.default_rng(seed)
    truth, sampler = (dist0, dist1) if error == "type1" else (dist1, dist0)
    S_is = np.sum(_llr(sampler.rvs(size=(draws, n), random_state=rng), dist0, dist1), axis=1)
    if error == "type1":
        ind = S_is >= threshold
        w = np.where(ind, np.exp(-np.where(ind, S_is, 0.0)), 0.0)
    else:
        ind = S_is <= threshold
        w = np.where(ind, np.exp(np.where(ind, S_is, 0.0)), 0.0)
    est = float(w.mean())
    se = float(w.std(ddof=1) / np.sqrt(draws))
    S_n = np.sum(_llr(truth.rvs(size=(draws, n), random_state=rng), dist0, dist1), axis=1)
    hits = (S_n >= threshold) if error == "type1" else (S_n <= threshold)
    naive = float(hits.mean())
    naive_se = float(np.sqrt(naive * (1 - naive) / draws))
    ess = float(w.sum() ** 2 / np.sum(w ** 2)) if np.any(w > 0) else 0.0
    vr = (naive_se ** 2 / se ** 2) if se > 0 else np.inf
    if naive_se == 0:
        vr = np.inf
    z = stats.norm.isf(alpha / 2)
    warns = []
    if hits.sum() == 0:
        warns.append(_finding("notice", f"none of the {draws} plain draws fell in the event: plain simulation says 0",
                              "report the importance-sampling estimate with its SE"))
    if ess < 30:
        warns.append(_finding("warning", f"only {ess:.3g} effective draws in the event: the importance-sampling "
                              "estimate is unstable", "raise draws or choose a threshold nearer the sampler's typical values"))
    name = "P0(log L_n >= threshold)" if error == "type1" else "P1(log L_n <= threshold)"
    out = {"estimand": name, "estimate": est, "se": se, "ci_low": est - z * se, "ci_high": est + z * se,
           "naive": naive, "naive_se": naive_se, "hits_naive": int(hits.sum()), "variance_ratio": float(vr),
           "ess": ess, "n": n, "draws": draws,
           "used": {"error": error, "n": n, "threshold": float(threshold), "draws": draws, "seed": seed,
                    "skill_version": _sv()}, "warnings": warns}
    if show:
        print(f"\n{name} with n = {n}, threshold = {threshold:g}: {est:.4g} (SE {se:.2g}) by importance sampling; "
              f"plain simulation {naive:.4g} (SE {naive_se:.2g}, {int(hits.sum())} hits in {draws}); "
              f"variance ratio {vr:.3g}, effective draws {ess:.3g}")
        _print_findings(warns)
    return out
