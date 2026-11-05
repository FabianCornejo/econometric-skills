"""Multivariate time series: VAR, impulse responses with bootstrap bands, variance decompositions, structural
identification (short-run, long-run, sign restrictions) and VECM (v2.4.7). Reference: time-series.md sections 7-9.

Use as a module:
    import sys; sys.path.insert(0, "<skill folder>/scripts")
    from multivariate import *

Functions
    var_analysis(df)                    lag order by AIC/BIC/HQ/FPE, the fitted VAR, stability (companion
                                        eigenvalues), residual whiteness and normality, the Granger matrix, and
                                        warnings on parameters per observation and unit roots
    irf_bands(res, h)                   impulse responses with bootstrap bands, Kilian (1998) bias-corrected by
                                        default; recursive (Cholesky, in a chosen ordering) or long-run
                                        (Blanchard-Quah) identification
    fevd_table(res, h)                  forecast-error variance decomposition under several Cholesky orderings,
                                        with the shares that depend on the ordering flagged
    svar_identify(k, A, B)              order condition (counting) and local rank condition (numerical Jacobian)
                                        for an AB-model with zeros and free entries
    svar_longrun(res, h)                Blanchard-Quah long-run identification: impact matrix, long-run matrix
                                        and responses (bands through irf_bands)
    svar_sign(res, signs, h)            sign restrictions by random rotations (Rubio-Ramirez, Waggoner and Zha
                                        2010): accepted share, median and percentile bands, and the Fry-Pagan
                                        median-target model
    johansen_trace(df)                  Johansen trace test for the stated deterministic case, with simulated
                                        critical values
    vecm_fit(df)                        Johansen rank, lag order, alpha and beta with SEs, the error-correction
                                        check (does alpha pull the system back to beta'y = 0?) and weak exogeneity

Every number comes from the data passed in. Bootstrap functions take B and seed.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import _console  # noqa: F401  (Windows-safe console output, see _console.py)

ALPHA = 0.05


def _fmt(v):
    return f"{v:.4f}" if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v)


def _frame(df) -> pd.DataFrame:
    d = pd.DataFrame(df).astype(float)
    if d.isna().any().any():
        raise ValueError("missing values: fill or drop them before fitting a VAR")
    if d.shape[1] < 2:
        raise ValueError("a VAR needs at least two series")
    return d


# =============================================================================
# Small VAR machinery (numpy), used by the bootstrap
# =============================================================================
def _lagmat(Y: np.ndarray, p: int, trend: str):
    T, k = Y.shape
    X = [Y[p - i:T - i] for i in range(1, p + 1)]
    X = np.hstack(X) if X else np.empty((T - p, 0))
    if trend == "c":
        X = np.hstack([np.ones((T - p, 1)), X])
    return Y[p:], X


def _var_ols(Y: np.ndarray, p: int, trend: str):
    """Returns (c, A_list, resid, Sigma) with Sigma = E'E/(T - p - kp - const) as statsmodels' sigma_u."""
    Yt, X = _lagmat(Y, p, trend)
    B, *_ = np.linalg.lstsq(X, Yt, rcond=None)
    E = Yt - X @ B
    k = Y.shape[1]
    off = 1 if trend == "c" else 0
    c = B[0] if trend == "c" else np.zeros(k)
    A = [B[off + i * k: off + (i + 1) * k].T for i in range(p)]
    dof = len(Yt) - X.shape[1]
    Sigma = E.T @ E / dof
    return c, A, E, Sigma


def _companion(A: list) -> np.ndarray:
    k, p = A[0].shape[0], len(A)
    C = np.zeros((k * p, k * p))
    C[:k] = np.hstack(A)
    if p > 1:
        C[k:, :-k] = np.eye(k * (p - 1))
    return C


def _max_root(A: list) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(_companion(A)))))


def _ma(A: list, h: int) -> np.ndarray:
    """MA coefficients Phi_0..Phi_h (h+1, k, k)."""
    k, p = A[0].shape[0], len(A)
    Phi = np.zeros((h + 1, k, k))
    Phi[0] = np.eye(k)
    for j in range(1, h + 1):
        for i in range(1, min(j, p) + 1):
            Phi[j] += Phi[j - i] @ A[i - 1]
    return Phi


def _simulate(c, A, E, Y0, T, rng, burn=0):
    """Rebuild a series of length T from the start values Y0 (p x k) and resampled residual rows."""
    p, k = len(A), len(c)
    out = np.zeros((T + burn, k))
    out[:p] = Y0
    idx = rng.integers(0, len(E), size=T + burn)
    for t in range(p, T + burn):
        v = c.copy()
        for i in range(p):
            v = v + A[i] @ out[t - 1 - i]
        out[t] = v + E[idx[t]]
    return out[burn:]


def _shrink(A_hat: list, bias: list) -> list:
    """Kilian (1998): subtract the bias, shrinking the correction until the companion matrix is stationary."""
    if _max_root(A_hat) >= 1:
        return A_hat
    delta = 1.0
    while delta > 0:
        A_c = [a - delta * b for a, b in zip(A_hat, bias)]
        if _max_root(A_c) < 1:
            return A_c
        delta -= 0.01
    return A_hat


def _impact(A: list, Sigma: np.ndarray, identification: str) -> np.ndarray:
    if identification == "cholesky":
        return np.linalg.cholesky(Sigma)
    if identification == "longrun":
        k = Sigma.shape[0]
        C1 = np.linalg.inv(np.eye(k) - sum(A))
        L = np.linalg.cholesky(C1 @ Sigma @ C1.T)
        return np.linalg.solve(C1, L)
    raise ValueError("identification must be 'cholesky' or 'longrun'")


def _res_parts(res):
    names = list(res.names)
    trend = getattr(res, "trend", "c")
    if trend not in ("c", "n"):
        raise ValueError("bootstrap bands support trend 'c' or 'n' (fit the VAR with one of them)")
    Y = np.asarray(res.endog, float)
    return names, trend, Y, int(res.k_ar)


# =============================================================================
# VAR analysis
# =============================================================================
def var_analysis(df, maxlags: int = 8, ic: str = "bic", lags: int | None = None, trend: str = "c",
                 whiteness_lags: int | None = None, alpha: float = ALPHA, show: bool = True) -> dict:
    """Fit a VAR and run the checks a VAR needs before any IRF is read.

    Lag order: AIC, BIC, HQ and FPE for p = 0..maxlags on a common sample (statsmodels select_order); ic picks
    one (BIC is consistent, AIC tends to choose more lags; if the residuals stay autocorrelated, add lags).
    Stability: the largest modulus of the companion eigenvalues must be below 1. Residuals: the adjusted
    portmanteau test of whiteness and the Jarque-Bera test of normality. Granger matrix: p-values of the F test
    that the lags of the column variable do not enter the equation of the row variable (predictive precedence,
    not causation). Warnings: parameters per equation against observations, and series that the joint verdict
    of diagnostics.stationarity (ADF, DF-GLS, KPSS) does not classify as I(0) (a VAR in levels with I(1)
    series: use vecm_fit if cointegrated, differences otherwise); when only the ADF fails to reject, that is
    said separately."""
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.stattools import adfuller
    d = _frame(df)
    k = d.shape[1]
    model = VAR(d)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sel = model.select_order(maxlags, trend=trend)
    ic_tab = pd.DataFrame({name: pd.Series(vals) for name, vals in sel.ics.items()})
    ic_tab.index.name = "p"
    chosen = {name: int(v) for name, v in sel.selected_orders.items()}
    p = int(lags) if lags is not None else max(1, chosen[ic])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit(p, trend=trend)
    T = int(res.nobs)
    root = _max_root(list(res.coefs))
    wl = whiteness_lags or max(p + 4, 10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        white = res.test_whiteness(nlags=wl, adjusted=True)
        normal = res.test_normality()
    names = list(res.names)
    gr = pd.DataFrame(np.nan, index=names, columns=names)
    for caused, causing in itertools.permutations(names, 2):
        gr.loc[caused, causing] = float(res.test_causality(caused, causing, kind="f").pvalue)
    gr.index.name = "equation (caused)"
    gr.columns.name = "lags of (causing)"
    warn = []
    per_eq = k * p + (1 if trend == "c" else 0)
    if T < 10 * per_eq:
        warn.append(f"{per_eq} parameters per equation for {T} observations ({T / per_eq:.1f} per parameter): "
                    "estimates and IRFs will be imprecise; use fewer variables or lags, or a Bayesian VAR")
    if root >= 1:
        warn.append(f"the VAR is not stable (largest companion root {root:.3f}): IRFs do not die out")
    elif root > 0.95:
        warn.append(f"largest companion root {root:.3f}: very persistent; long-horizon IRFs are unreliable")
    if white.pvalue < alpha:
        warn.append(f"the residuals are autocorrelated (portmanteau p = {white.pvalue:.4f}): add lags before "
                    "reading IRFs or Granger tests")
    # unit roots by the joint verdict of diagnostics.stationarity (ADF, DF-GLS, KPSS), the same one the rest of
    # the skill uses; ADF alone has low power near the unit root (v2.4.7.1)
    from diagnostics import stationarity
    unit, adf_only, trend_only, orders = [], [], [], {}
    for c in names:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st = stationarity(d[c], c, show=False)
            orders[c] = st.attrs.get("suggested_order", "")
            p_adf = adfuller(d[c].to_numpy(), autolag="AIC")[1]
        if "I(0)" not in orders[c]:
            unit.append(c)
        elif orders[c].startswith("I(0) around a deterministic trend") and "t" not in trend:
            trend_only.append(c)
        elif p_adf > 0.10:
            adf_only.append(c)
    if unit:
        listed = ", ".join(c + " (" + orders[c].split(" —")[0] + ")" for c in unit)
        warn.append(f"the unit-root tests (ADF, DF-GLS, KPSS) do not classify {listed} "
                    "as stationary (a unit root is not ruled out): a VAR in levels is consistent but Granger tests and long-horizon IRFs are "
                    "unreliable; test cointegration (vecm_fit) or use differences")
    if trend_only:
        warn.append(f"{', '.join(trend_only)} is stationary only around a deterministic trend, and this VAR has trend "
                    f"'{trend}': add the trend (trend='ct') or detrend the series before reading the VAR")
    if adf_only:
        warn.append(f"ADF alone does not reject a unit root for {', '.join(adf_only)}, but the joint verdict of ADF, "
                    "DF-GLS and KPSS is I(0): a persistent series; a VAR in levels is reasonable, and a Toda-Yamamoto "
                    "Granger test is the safe check")
    out = {"results": res, "lag_order": p, "ic_table": ic_tab, "selected": chosen, "max_root": root,
           "stable": root < 1, "whiteness": {"statistic": float(white.test_statistic), "p": float(white.pvalue),
                                              "lags": wl},
           "normality": {"statistic": float(normal.test_statistic), "p": float(normal.pvalue)},
           "granger_pvalues": gr, "nobs": T, "params_per_equation": per_eq, "integration_order": orders,
           "warnings": warn,
           "method": f"VAR({p}) by OLS, trend '{trend}'"}
    if show:
        print(f"\nVAR lag selection (p = 0..{maxlags}); chosen by each criterion: {chosen}; using p = {p} ({ic.upper() if lags is None else 'given'})")
        print(ic_tab.to_string(float_format=_fmt))
        print(f"\nStability: largest companion root {root:.4f} ({'stable' if root < 1 else 'NOT stable'})")
        print(f"Residual whiteness (adjusted portmanteau, {wl} lags): p = {white.pvalue:.4f}; "
              f"normality (JB): p = {normal.pvalue:.4f}")
        print("\nGranger p-values (row: equation; column: lags tested):")
        print(gr.to_string(float_format=_fmt))
        for w in warn:
            print("Warning:", w)
    return out


# =============================================================================
# Impulse responses with bootstrap bands
# =============================================================================
def _irf_from(c, A, Sigma, h, identification):
    return _ma(A, h) @ _impact(A, Sigma, identification)


def irf_bands(res, h: int = 12, B: int = 999, level: float = 0.90, ordering: list | None = None,
              identification: str = "cholesky", bias_correct: bool = True, cumulative: bool = False,
              seed: int = 0, show: bool = True) -> dict:
    """Impulse responses of a fitted statsmodels VAR with bootstrap percentile bands.

    identification 'cholesky' (recursive, in `ordering`, default the column order; the ordering is an economic
    assumption: say why a variable cannot react within the period to the ones after it) or 'longrun'
    (Blanchard-Quah: shock j has no long-run effect on variables before j). The bootstrap resamples the VAR
    residuals, rebuilds the series from the actual first p observations, re-estimates and recomputes the IRF.
    With bias_correct=True (Kilian 1998), OLS slopes are first corrected for their small-sample bias (estimated
    by a first bootstrap, shrunk to keep the VAR stationary), the series are generated from the corrected VAR
    and each re-estimate is corrected the same way; plain percentile bands are too narrow and off-centre when
    the VAR is persistent and the sample short. Draws that stay non-stationary are discarded and drawn again
    (their count is reported). The point IRF is the OLS one; the bands come from the corrected draws.
    Responses are to a one-standard-deviation structural shock."""
    names, trend, Y, p = _res_parts(res)
    if ordering is not None:
        if sorted(ordering) != sorted(names):
            raise ValueError("ordering must list every variable once")
        perm = [names.index(v) for v in ordering]
        Y = Y[:, perm]
        names = list(ordering)
    k = len(names)
    c, A, E, Sigma = _var_ols(Y, p, trend)
    E = E - E.mean(axis=0)
    point = _irf_from(c, A, Sigma, h, identification)
    rng = np.random.default_rng(seed)
    T = len(Y)
    bias = [np.zeros((k, k)) for _ in range(p)]
    gen_A = A
    if bias_correct:
        draws = []
        for _ in range(min(B, 499)):
            Ys = _simulate(c, A, E, Y[:p], T, rng)
            draws.append(_var_ols(Ys, p, trend)[1])
        bias = [np.mean([d[i] for d in draws], axis=0) - A[i] for i in range(p)]
        gen_A = _shrink(A, bias)
    # keep the sample mean when the slopes are corrected: c = (I - sum A) mu
    mu = np.linalg.solve(np.eye(k) - sum(A), c) if trend == "c" else np.zeros(k)
    gen_c = (np.eye(k) - sum(gen_A)) @ mu if trend == "c" else c
    sims = np.empty((B, h + 1, k, k))
    redrawn = 0
    for b in range(B):
        for attempt in range(100):
            Ys = _simulate(gen_c, gen_A, E, Y[:p], T, rng)
            cb, Ab, _, Sb = _var_ols(Ys, p, trend)
            if bias_correct:
                Ab = _shrink(Ab, bias)        # keeps the uncorrected draw when no correction is stationary
            if _max_root(Ab) < 1:
                break
            redrawn += 1                       # a non-stationary draw is replaced by a new draw, not by gen_A
        sims[b] = _irf_from(cb, Ab, Sb, h, identification)
    if cumulative:
        point, sims = np.cumsum(point, axis=0), np.cumsum(sims, axis=1)
    a = (1 - level) / 2
    lo, hi = np.quantile(sims, a, axis=0), np.quantile(sims, 1 - a, axis=0)
    rows = []
    for j, shock in enumerate(names):
        for i, resp in enumerate(names):
            for s in range(h + 1):
                rows.append({"shock": shock, "response": resp, "h": s, "irf": point[s, i, j],
                             "low": lo[s, i, j], "high": hi[s, i, j]})
    tab = pd.DataFrame(rows).set_index(["shock", "response", "h"])
    sig = tab[(tab["low"] > 0) | (tab["high"] < 0)]
    out = {"irf": point, "lower": lo, "upper": hi, "table": tab, "names": names, "level": level, "B": B,
           "identification": identification, "bias_correct": bias_correct, "cumulative": cumulative,
           "excluding_zero": sig.reset_index()[["shock", "response", "h"]], "nonstationary_redraws": redrawn,
           "method": (f"{'Kilian bias-corrected' if bias_correct else 'plain'} residual bootstrap, {B} draws, "
                      f"{level:.0%} percentile bands; {identification} identification"
                      + (f", ordering {names}" if identification == "cholesky" else ""))}
    if show:
        print(f"\nImpulse responses ({out['method']})")
        for j, shock in enumerate(names):
            sub = tab.xs(shock, level="shock")
            wide = sub["irf"].unstack(0)
            hs = [s for s in (0, 1, 2, 4, 8, 12, 24) if s <= h]
            print(f"\nShock: {shock} (one SD)")
            print(wide.loc[hs].to_string(float_format=_fmt))
        print(f"\n{len(sig)} (response, horizon) cells have bands excluding zero")
    return out


# =============================================================================
# Variance decomposition under several orderings
# =============================================================================
def _fevd(A, Sigma, h, perm):
    k = Sigma.shape[0]
    Ap = [a[np.ix_(perm, perm)] for a in A]
    Sp = Sigma[np.ix_(perm, perm)]
    Theta = _ma(Ap, h - 1) @ np.linalg.cholesky(Sp)
    contrib = np.cumsum(Theta ** 2, axis=0)              # (h, k, k): response i, shock m
    total = contrib.sum(axis=2, keepdims=True)
    share = contrib / total
    inv = np.argsort(perm)
    return share[:, inv][:, :, inv]                      # back to the original variable order


def fevd_table(res, h: int = 12, orderings: list | None = None, horizons=(1, 4, 8, 12), threshold: float = 0.10,
               show: bool = True) -> dict:
    """Share of the h-step forecast-error variance of each variable due to each orthogonal shock, under every
    ordering in `orderings` (default: the column order and its reverse; pass more to cover the plausible
    ones). Shares that move by more than `threshold` across orderings are flagged: the decomposition there
    reflects the ordering assumption, not the data."""
    names, trend, Y, p = _res_parts(res)
    c, A, E, Sigma = _var_ols(Y, p, trend)
    if orderings is None:
        orderings = [names, names[::-1]]
    horizons = [s for s in horizons if s <= h]
    tables, shares = {}, []
    for order in orderings:
        perm = [names.index(v) for v in order]
        sh = _fevd(A, Sigma, h, perm)
        shares.append(sh)
        rows = []
        for i, v in enumerate(names):
            for s in horizons:
                rows.append(pd.Series(sh[s - 1, i], index=names, name=(v, s)))
        t = pd.DataFrame(rows)
        t.index = pd.MultiIndex.from_tuples(t.index, names=["variable", "horizon"])
        tables[" > ".join(order)] = t
    spread = np.max(shares, axis=0) - np.min(shares, axis=0)
    rows = []
    for i, v in enumerate(names):
        for m, sname in enumerate(names):
            mx = float(spread[[s - 1 for s in horizons], i, m].max())
            rows.append({"variable": v, "shock": sname, "max_spread": mx, "ordering_sensitive": mx > threshold})
    sens = pd.DataFrame(rows)
    out = {"tables": tables, "sensitivity": sens, "orderings": orderings, "horizons": horizons,
           "method": "orthogonalised FEVD (Cholesky) under each ordering"}
    if show:
        for key, t in tables.items():
            print(f"\nFEVD, ordering {key}")
            print(t.to_string(float_format=_fmt))
        flagged = sens[sens["ordering_sensitive"]]
        print(f"\n{len(flagged)} of {len(sens)} (variable, shock) shares move by more than {threshold:.0%} across "
              "orderings" + (":\n" + flagged.to_string(index=False, float_format=_fmt) if len(flagged) else ""))
    return out


# =============================================================================
# Identification of AB-models
# =============================================================================
def svar_identify(k: int, A=None, B=None, seed: int = 0, show: bool = True) -> dict:
    """Order and rank conditions for the structural VAR A u_t = B e_t, Var(e_t) = I.

    A and B are k x k arrays with numbers for fixed entries and np.nan for free ones (A defaults to I, B to
    a full free matrix). Order condition: free parameters ≤ k(k+1)/2, the distinct elements of Sigma_u.
    Rank (local identification): the Jacobian of vech(A^-1 B B' A^-1') with respect to the free parameters,
    evaluated at random values, must have full column rank. Counting alone can pass with a model that is not
    identified (restrictions in the wrong places)."""
    A = np.eye(k) if A is None else np.array(A, dtype=float)
    B = np.full((k, k), np.nan) if B is None else np.array(B, dtype=float)
    if A.shape != (k, k) or B.shape != (k, k):
        raise ValueError("A and B must be k x k")
    fa, fb = np.isnan(A), np.isnan(B)
    n_free = int(fa.sum() + fb.sum())
    n_eq = k * (k + 1) // 2
    order = "exactly identified" if n_free == n_eq else ("over-identified" if n_free < n_eq else "NOT identified")
    rng = np.random.default_rng(seed)
    il = np.tril_indices(k)

    def vech_sigma(theta):
        Aa, Bb = A.copy(), B.copy()
        Aa[fa] = theta[:fa.sum()]
        Bb[fb] = theta[fa.sum():]
        Ai = np.linalg.solve(Aa, Bb)
        return (Ai @ Ai.T)[il]
    ranks = []
    for _ in range(3):
        theta = rng.normal(size=n_free)
        theta[:fa.sum()] = theta[:fa.sum()] * 0.3        # keep A away from singular
        base = vech_sigma(theta)
        J = np.empty((len(base), n_free))
        for j in range(n_free):
            e = np.zeros(n_free)
            e[j] = 1e-6
            J[:, j] = (vech_sigma(theta + e) - vech_sigma(theta - e)) / 2e-6
        sv = np.linalg.svd(J, compute_uv=False)
        ranks.append(int(np.sum(sv > 1e-6 * sv.max())) if sv.size and sv.max() > 0 else 0)
    rank = max(ranks)
    rank_ok = rank == n_free
    verdict = (f"{order} by counting ({n_free} free parameters, {n_eq} equations); rank condition "
               + ("holds (locally identified)" if rank_ok else f"FAILS (Jacobian rank {rank} < {n_free}): the "
                  "restrictions do not pin down the free parameters"))
    out = {"k": k, "free_parameters": n_free, "equations": n_eq, "restrictions_needed_for_exact":
           2 * k * k - n_eq, "restrictions_imposed": 2 * k * k - n_free, "order_condition": order,
           "jacobian_rank": rank, "rank_condition": rank_ok, "verdict": verdict}
    if show:
        print(f"\nSVAR identification (k = {k}): {verdict}")
    return out


def svar_longrun(res, h: int = 20, bands: bool = True, B: int = 499, level: float = 0.90, seed: int = 0,
                 show: bool = True) -> dict:
    """Blanchard-Quah (1989) long-run identification: the long-run response matrix C(1)·B0 is lower
    triangular, so shock j has no permanent effect on the variables ordered before it. Typical use: output
    growth first, unemployment second; the second shock is 'demand' because it does not move the level of
    output in the long run. The data enter in differences or as stationary series; the impact matrix B0
    satisfies B0 B0' = Sigma_u. The bands are for the responses of the variables as they enter the VAR;
    when the first is a growth rate, call irf_bands(res, identification='longrun', cumulative=True) for the
    response of its level."""
    names, trend, Y, p = _res_parts(res)
    c, A, E, Sigma = _var_ols(Y, p, trend)
    k = len(names)
    C1 = np.linalg.inv(np.eye(k) - sum(A))
    B0 = _impact(A, Sigma, "longrun")
    LR = C1 @ B0
    out = {"impact": pd.DataFrame(B0, index=names, columns=[f"shock {j + 1}" for j in range(k)]),
           "long_run": pd.DataFrame(LR, index=names, columns=[f"shock {j + 1}" for j in range(k)]),
           "check_BBt_equals_Sigma": float(np.max(np.abs(B0 @ B0.T - Sigma))),
           "method": "Blanchard-Quah: lower-triangular long-run matrix"}
    if bands:
        out["irf"] = irf_bands(res, h=h, B=B, level=level, identification="longrun", seed=seed, show=False)
    if show:
        print("\nBlanchard-Quah identification")
        print("Impact matrix B0:\n" + out["impact"].to_string(float_format=_fmt))
        print("Long-run matrix C(1)B0 (lower triangular by construction):\n" + out["long_run"].to_string(float_format=_fmt))
    return out


# =============================================================================
# Sign restrictions
# =============================================================================
def svar_sign(res, signs: dict, h: int = 12, sign_horizons: int = 0, draws: int = 5000, level: float = 0.68,
              seed: int = 0, show: bool = True) -> dict:
    """Sign-restricted structural shocks by random orthogonal rotations of the Cholesky factor.

    signs: {shock_name: {variable: +1 or -1, ...}, ...} imposed on the responses at horizons 0..sign_horizons;
    shocks are taken in the order given and assigned to the first columns of the rotation (a column may be
    flipped). Each draw: Q from the QR decomposition of a standard normal matrix (diag R > 0), impact = P Q;
    kept when every restriction holds. Reported: acceptance share, pointwise median and percentile bands, and
    the Fry-Pagan median target (the accepted model closest to the pointwise median). The pointwise median is
    not the response of any single model, and the bands reflect the identification set, not sampling
    uncertainty (coefficients held at OLS). Few accepted draws means restrictions the data barely allow."""
    names, trend, Y, p = _res_parts(res)
    c, A, E, Sigma = _var_ols(Y, p, trend)
    k = len(names)
    shocks = list(signs)
    if len(shocks) > k:
        raise ValueError("more restricted shocks than variables")
    for s, rest in signs.items():
        bad = [v for v in rest if v not in names]
        if bad:
            raise ValueError(f"unknown variables in the restrictions of {s}: {bad}")
    P = np.linalg.cholesky(Sigma)
    Phi = _ma(A, h)
    rng = np.random.default_rng(seed)
    kept = []
    for _ in range(draws):
        Z = rng.normal(size=(k, k))
        Q, R = np.linalg.qr(Z)
        Q = Q @ np.diag(np.sign(np.diag(R)))
        irf = Phi @ (P @ Q)
        ok = True
        for j, s in enumerate(shocks):
            rest = signs[s]
            resp = irf[:sign_horizons + 1, :, j]
            good = all(np.all(np.sign(resp[:, names.index(v)]) == sg) for v, sg in rest.items())
            bad = all(np.all(np.sign(-resp[:, names.index(v)]) == sg) for v, sg in rest.items())
            if good:
                continue
            if bad:
                irf[:, :, j] = -irf[:, :, j]
                continue
            ok = False
            break
        if ok:
            kept.append(irf)
    n_ok = len(kept)
    out = {"accepted": n_ok, "draws": draws, "acceptance_share": n_ok / draws, "shocks": shocks, "names": names,
           "level": level, "method": "sign restrictions by QR rotations (Rubio-Ramirez, Waggoner and Zha 2010); "
           "coefficients at OLS"}
    if n_ok == 0:
        out["warning"] = "no rotation satisfies the restrictions: they are incompatible with the estimated VAR"
        if show:
            print("\nSign restrictions:", out["warning"])
        return out
    K = np.array(kept)
    a = (1 - level) / 2
    med = np.median(K, axis=0)
    lo, hi = np.quantile(K, a, axis=0), np.quantile(K, 1 - a, axis=0)
    sd = K.std(axis=0)
    sd[sd == 0] = 1.0
    dist = (((K - med) / sd)[:, :, :, :len(shocks)] ** 2).sum(axis=(1, 2, 3))
    mt = K[int(np.argmin(dist))]
    out.update({"median": med[:, :, :len(shocks)], "lower": lo[:, :, :len(shocks)], "upper": hi[:, :, :len(shocks)],
                "median_target": mt[:, :, :len(shocks)]})
    warn = []
    if n_ok < 100:
        warn.append(f"only {n_ok} accepted rotations: bands are noisy and the restrictions are barely compatible "
                    "with the data")
    warn.append("the pointwise median is not the response of a single model; report the median-target model or "
                "the whole set")
    out["warnings"] = warn
    if show:
        print(f"\nSign restrictions: {n_ok} of {draws} rotations accepted ({n_ok / draws:.1%})")
        for j, s in enumerate(shocks):
            hs = [x for x in (0, 1, 2, 4, 8, 12) if x <= h]
            t = pd.DataFrame({f"{v} median": med[hs, i, j] for i, v in enumerate(names)}, index=hs)
            print(f"\nShock '{s}' (median of accepted, impact one SD):")
            print(t.to_string(float_format=_fmt))
        for w in warn:
            print("Note:", w)
    return out


# =============================================================================
# VECM
# =============================================================================
def _johansen(Y: np.ndarray, k_ar_diff: int, case: str):
    """Johansen reduced-rank regression. case: 'n' (no deterministic terms), 'ci' (constant restricted to the
    cointegrating relation), 'co' (unrestricted constant), 'li' (trend restricted, constant unrestricted).
    Returns the eigenvalues (descending) and the trace statistics for r0 = 0..k-1."""
    dY = np.diff(Y, axis=0)
    T_eff = len(dY) - k_ar_diff
    Z0 = dY[k_ar_diff:]
    Z1 = Y[k_ar_diff:-1]
    Z2 = [dY[k_ar_diff - i:len(dY) - i] for i in range(1, k_ar_diff + 1)]
    Z2 = np.hstack(Z2) if Z2 else np.empty((T_eff, 0))
    one = np.ones((T_eff, 1))
    trend = np.arange(1, T_eff + 1, dtype=float)[:, None]
    if case == "ci":
        Z1 = np.hstack([Z1, one])
    elif case == "co":
        Z2 = np.hstack([Z2, one])
    elif case == "li":
        Z1 = np.hstack([Z1, trend])
        Z2 = np.hstack([Z2, one])
    elif case != "n":
        raise ValueError("deterministic must be 'n', 'ci', 'co' or 'li'")

    def resid(Z):
        if Z2.shape[1] == 0:
            return Z
        b, *_ = np.linalg.lstsq(Z2, Z, rcond=None)
        return Z - Z2 @ b
    R0, R1 = resid(Z0), resid(Z1)
    S00, S11, S01 = R0.T @ R0 / T_eff, R1.T @ R1 / T_eff, R0.T @ R1 / T_eff
    M = np.linalg.solve(S11, S01.T @ np.linalg.solve(S00, S01))
    lam = np.sort(np.real(np.linalg.eigvals(M)))[::-1][:Y.shape[1]]
    lam = np.clip(lam, 0, 1 - 1e-12)
    k = Y.shape[1]
    tr = np.array([-T_eff * np.sum(np.log(1 - lam[r:])) for r in range(k)])
    return lam, tr


_CRIT_CACHE: dict = {}


def _johansen_crit(T: int, k_ar_diff: int, m: int, case: str, level: float, reps: int, seed: int,
                   drift: float = 0.0) -> float:
    """Critical value of the trace statistic for k - r = m common trends, simulated with m random walks without
    drift of the same length and lagged differences. Valid for 'n', 'ci' (invariant to the level) and 'li'
    (invariant to the drift). For 'co' the distribution depends on the drift, so one common trend gets a drift
    of the size measured in the data (in SDs of the increments). For r0 > 0 it simulates the m-variable
    system, an approximation to the k-variable one."""
    key = (T, k_ar_diff, m, case, level, reps, seed, round(drift, 6))
    if key not in _CRIT_CACHE:
        rng = np.random.default_rng(seed + 1000 * m)
        stats_ = np.empty(reps)
        for b in range(reps):
            inc = rng.normal(size=(T, m))
            inc[:, 0] += drift                 # 'co': one common trend drifts, as in the data
            W = np.cumsum(inc, axis=0)
            stats_[b] = _johansen(W, k_ar_diff, case)[1][0]
        _CRIT_CACHE[key] = float(np.quantile(stats_, 1 - level))
    return _CRIT_CACHE[key]


def johansen_trace(df, k_ar_diff: int = 1, deterministic: str = "ci", level: float = ALPHA, reps: int = 2000,
                   seed: int = 0, show: bool = True) -> dict:
    """Johansen trace test with the deterministic case stated and critical values simulated for it.

    'ci' (default): the levels have no drift and the cointegrating relation has an intercept (most macro and
    price series in levels around a mean); 'co': the levels drift (unrestricted constant); 'li': the
    relation has a trend (constant unrestricted; statsmodels' 'coli'); 'n': no deterministic terms. Critical
    values are simulated for the case; for 'co' with the drift measured in the data, since that distribution
    depends on it (an approximation: two drifting cointegrated series, T = 250, rejected the true rank in 11%
    of 100 samples at 5%, against 13% with the asymptotic tables; prefer 'ci' unless the levels clearly drift). statsmodels' select_coint_rank has no restricted-constant case; using
    'co' on series without drift over-rejects the last null (rank k - 1 against k):
    in simulations for v2.4.7 two cointegrated random walks without drift were declared stationary in 9 of
    20 samples. Rank: the first r0 whose trace statistic is below its critical value."""
    d = _frame(df)
    Y = d.to_numpy()
    k = Y.shape[1]
    lam, tr = _johansen(Y, k_ar_diff, deterministic)
    T_sim = len(Y)
    drift = 0.0
    if deterministic == "co":
        dY = np.diff(Y, axis=0)
        drift = float(np.linalg.norm(dY.mean(axis=0) / dY.std(axis=0, ddof=1)))
    crit = [_johansen_crit(T_sim, k_ar_diff, k - r, deterministic, level, reps, seed, drift) for r in range(k)]
    crit_src = "simulated"
    rank = k
    for r in range(k):
        if tr[r] < crit[r]:
            rank = r
            break
    tab = pd.DataFrame({"r0": range(k), "eigenvalue": lam, "trace": tr, f"crit {1 - level:.0%} ({crit_src})": crit,
                        "reject": tr >= np.array(crit)})
    out = {"table": tab, "rank": rank, "deterministic": deterministic, "k_ar_diff": k_ar_diff, "reps": reps,
           "drift_used": drift,
           "method": f"Johansen trace test, case '{deterministic}', critical values from {reps} simulations"
                     + (f" with a drift of {drift:.2f} SD per period in one common trend" if deterministic == "co" else "")}
    if show:
        print(f"\nJohansen trace test (case '{deterministic}', {k_ar_diff} lagged differences):")
        print(tab.to_string(index=False, float_format=_fmt))
        print(f"→ cointegration rank {rank}")
    return out


def vecm_fit(df, k_ar_diff: int | None = None, rank: int | None = None, deterministic: str = "ci",
             maxlags: int = 8, ic: str = "bic", alpha: float = ALPHA, reps: int = 2000, seed: int = 0,
             show: bool = True) -> dict:
    """Vector error-correction model with the checks that make alpha and beta readable.

    Lags of the differences by ic (statsmodels select_order) unless given; cointegration rank by the Johansen
    trace test at `alpha` for the deterministic case (johansen_trace) unless given (rank 0: no long-run
    relation, use a VAR in differences; rank = k: the series are stationary, use a VAR in levels). beta is
    normalised on the first r variables. Error-correction check: the implied levels VAR must have exactly
    k - r unit roots and its other roots inside the unit circle (eig(I + beta'alpha) inside the unit circle is
    the special case without lagged differences, and is not enough with them). Weak exogeneity of a
    variable: joint Wald test that all r loadings of its equation are zero. With r = 1, alpha_i·beta_i < 0
    means variable i moves to close the gap; with r > 1 single loadings depend on the normalisation."""
    from statsmodels.tsa.vector_ar.vecm import VECM, select_order
    d = _frame(df)
    k = d.shape[1]
    if deterministic not in ("n", "ci", "co", "li"):
        raise ValueError("deterministic must be 'n', 'ci', 'co' or 'li'")
    # statsmodels' 'li' has no constant; Johansen's restricted-trend case (ours) is its 'coli'
    sm_det = {"n": "n", "ci": "ci", "co": "co", "li": "coli"}[deterministic]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if k_ar_diff is None:
            so = select_order(d, maxlags=maxlags, deterministic=sm_det)
            k_ar_diff = int({"aic": so.aic, "bic": so.bic, "hqic": so.hqic, "fpe": so.fpe}[ic])
    jt = johansen_trace(d, k_ar_diff=k_ar_diff, deterministic=deterministic, level=alpha, reps=reps, seed=seed,
                        show=False)
    trace = jt["table"]
    r = int(jt["rank"]) if rank is None else int(rank)
    out = {"k_ar_diff": k_ar_diff, "rank": r, "trace_test": trace, "deterministic": deterministic}
    if r == 0 or r == k:
        msg = ("no cointegration at the chosen level: model the differences (VAR in differences)" if r == 0 else
               "full rank: the series behave as stationary; a VAR in levels is appropriate")
        if r == k:
            from statsmodels.tsa.stattools import adfuller
            unit = [c for c in d.columns if adfuller(d[c].to_numpy(), autolag="AIC")[1] > 0.10]
            if unit:
                msg += (f"; but ADF does not reject a unit root for {', '.join(map(str, unit))}: the rank test and "
                        "the unit-root tests disagree; check the deterministic case and the lags")
        out["message"] = msg
        if show:
            print(f"\nJohansen trace test (case '{deterministic}', k_ar_diff = {k_ar_diff}):\n"
                  + trace.to_string(index=False, float_format=_fmt))
            print("→", msg)
        return out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = VECM(d, k_ar_diff=k_ar_diff, coint_rank=r, deterministic=sm_det).fit()
    names = list(d.columns)
    beta = pd.DataFrame(res.beta, index=names, columns=[f"ect{j + 1}" for j in range(r)])
    alpha_ = pd.DataFrame(res.alpha, index=names, columns=beta.columns)
    se_a = pd.DataFrame(res.stderr_alpha, index=names, columns=beta.columns)
    t_a = alpha_ / se_a
    p_a = pd.DataFrame(2 * stats.norm.sf(np.abs(t_a.to_numpy())), index=names, columns=beta.columns)
    # Stability of the implied levels VAR: exactly k - r unit roots, the rest inside the unit circle. The 1 x r
    # check eig(I + beta'alpha) is only right without lagged differences (the short-run Gamma also matters).
    A_lev = list(res.var_rep)
    mods = np.sort(np.abs(np.linalg.eigvals(_companion(A_lev))))[::-1]
    n_unit = int(np.sum(np.abs(mods - 1) < 1e-6))
    explosive = bool(np.any(mods > 1 + 1e-6))
    rest = mods[np.abs(mods - 1) >= 1e-6]
    corrects = n_unit == k - r and not explosive
    # weak exogeneity of each variable: Wald test that all r loadings of its equation are zero (beta treated
    # as known, being superconsistent; auxiliary OLS of the VECM equation with HC1 SEs)
    import statsmodels.api as sm
    Yv = d.to_numpy()
    dY = np.diff(Yv, axis=0)
    ect = (Yv[:-1] @ res.beta)[k_ar_diff:]
    lagd = [dY[k_ar_diff - i:len(dY) - i] for i in range(1, k_ar_diff + 1)]
    Xaux = np.hstack([ect] + lagd + [np.ones((len(ect), 1))]) if deterministic != "n" else np.hstack([ect] + lagd)
    if deterministic == "li":
        Xaux = np.hstack([Xaux, np.arange(len(ect), dtype=float)[:, None]])
    wex = []
    for i, v in enumerate(names):
        f_ = sm.OLS(dY[k_ar_diff:, i], Xaux).fit(cov_type="HC1")
        R = np.zeros((r, Xaux.shape[1]))
        R[:, :r] = np.eye(r)
        wt = f_.wald_test(R, scalar=True)
        wex.append({"variable": v, "wald": float(wt.statistic), "df": r, "p": float(wt.pvalue),
                    "reading": "adjusts (not weakly exogenous)" if wt.pvalue < alpha else "weakly exogenous (does not adjust)"})
    wex = pd.DataFrame(wex)
    rows = []
    for j in range(r):
        for i, v in enumerate(names):
            a_ij, b_ij = res.alpha[i, j], res.beta[i, j]
            sig = p_a.iloc[i, j] < alpha
            if r > 1:
                role = "see the joint weak-exogeneity test (single loadings depend on the normalisation)"
            elif not sig:
                role = "does not adjust (weakly exogenous)"
            elif a_ij * b_ij < 0:
                role = "adjusts toward equilibrium"
            else:
                role = "moves away from equilibrium (wrong sign)"
            rows.append({"relation": f"ect{j + 1}", "variable": v, "beta": b_ij, "alpha": a_ij,
                         "se_alpha": se_a.iloc[i, j], "p_alpha": p_a.iloc[i, j], "reading": role})
    adj = pd.DataFrame(rows)
    warn = []
    if not corrects:
        warn.append(f"the implied levels VAR does not have exactly {k - r} unit roots with the rest inside the unit "
                    f"circle (moduli {', '.join(f'{m:.3f}' for m in mods[:min(len(mods), 6)])}): the system does not "
                    "return to beta'y = 0; check the rank, the deterministic terms and the lags before reading alpha")
    if r == 1 and (adj["reading"] == "moves away from equilibrium (wrong sign)").any():
        warn.append("a significant alpha has the wrong sign: that variable moves away from the long-run relation; "
                    "an unusual finding to check, not to explain away")
    if not (wex["p"] < alpha).any():
        warn.append("no variable adjusts significantly: the long-run relation is not supported by the adjustment")
    if r > 1:
        warn.append(f"with {r} relations, beta is identified only up to a rotation (normalised here on the first "
                    f"{r} variables): read single alpha and beta values only after imposing restrictions with "
                    "economic meaning; weak exogeneity is the joint test")
    out.update({"results": res, "beta": beta, "alpha": alpha_, "se_alpha": se_a, "p_alpha": p_a,
                "adjustment": adj, "weak_exogeneity": wex, "error_correcting": corrects,
                "levels_var_root_moduli": mods, "unit_roots": n_unit, "largest_other_root": float(rest.max()) if len(rest) else np.nan,
                "warnings": warn,
                "method": f"VECM by Johansen ML, rank {r}, {k_ar_diff} lagged differences, deterministic '{deterministic}'"})
    if show:
        print(f"\nJohansen trace test (case '{deterministic}', k_ar_diff = {k_ar_diff}):\n"
              + trace.to_string(index=False, float_format=_fmt))
        print(f"→ rank {r}" + (" (given)" if rank is not None else " at the chosen level"))
        print("\nbeta (normalised):\n" + beta.to_string(float_format=_fmt))
        print("\nAdjustment:\n" + adj.to_string(index=False, float_format=_fmt))
        print("\nWeak exogeneity (all loadings of the equation zero):\n" + wex.to_string(index=False, float_format=_fmt))
        print(f"Levels VAR: {n_unit} unit roots (expected {k - r}), largest other modulus "
              f"{out['largest_other_root']:.3f} → " + ("returns to equilibrium" if corrects else "does NOT return to equilibrium"))
        for w in warn:
            print("Warning:", w)
    return out
