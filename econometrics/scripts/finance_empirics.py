"""
Empirical finance for valuation, transactions and portfolio work (Ang, *Analyzing Financial Data
and Implementing Financial Models Using R*, 2015; Tsay, *An Introduction to Analysis of Financial
Data with R*, 2013; Campbell, Lo & MacKinlay, *The Econometrics of Financial Markets*, 1997).

Returns
    returns(prices, kind, dividends, freq)        simple or log, price or total return, resampled

Betas and cost of equity
    market_model(r_i, r_m, rf)                    alpha, beta (CAPM if rf is given), HAC SEs, R2, idiosyncratic vol
    rolling_beta(r_i, r_m, window, rf)            beta over a moving window with a 95% band
    adjusted_beta(beta, method, ...)              Blume (0.67 b + 0.33) or Vasicek (Bayesian shrinkage by SE)
    unlever_beta / relever_beta                   Hamada/Conine or Harris-Pringle (optional debt beta): comparables
    dimson_beta(r_i, r_m, lags, leads)            beta for thinly traded stocks (sum of lead/lag coefficients)
    factor_model(r_i, factors, rf)                Fama-French 3/5, Carhart, NEFIN (Brazil) loadings and alpha

Events and performance
    event_study(r, r_m, events, ...)              abnormal returns, CAR/CAAR, cross-sectional t, Patell,
                                                  BMP (robust to event-induced variance) and Corrado rank tests
    performance_ratios(r_p, r_b, rf)              Sharpe (with Lo SE), Sortino, Treynor, information ratio,
                                                  Jensen's alpha, Roy safety-first

Portfolios
    shrink_cov(returns)                           Ledoit-Wolf covariance (less estimation error than the sample one)
    efficient_frontier(mu, cov, rf, short)        minimum-variance and tangency portfolios, frontier points

Conventions: returns as decimals (0.012 = 1.2%), aligned on the same dates; rf per period in the
same frequency (e.g. daily CDI for daily returns; SGS series 12 is the daily CDI rate in % per day).
Annualization uses periods_per_year (252 daily, 52 weekly, 12 monthly).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats

ALPHA = 0.05


def _nw_lags(n):
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def _align(*series):
    d = pd.concat(series, axis=1, join="inner").dropna()
    return [d.iloc[:, i] for i in range(d.shape[1])]


# =============================================================================
# Returns
# =============================================================================
def returns(prices, kind: str = "simple", dividends=None, freq: str | None = None):
    """Returns from a price Series or DataFrame (one column per asset).

    kind "simple" (P_t / P_{t-1} - 1: aggregates across assets in a portfolio) or "log"
    (ln P_t - ln P_{t-1}: aggregates over time). dividends (same shape, paid on the ex-date) turn
    price returns into total returns: (P_t + D_t) / P_{t-1} - 1. freq resamples prices first
    ("W-FRI" weekly, "ME" month-end) using the last price of each period, so multi-day returns
    compound correctly. Use total returns for betas and performance, adjusted prices or dividends."""
    p = prices.copy()
    if freq is not None:
        if dividends is not None:
            dividends = dividends.resample(freq).sum()
        p = p.resample(freq).last()
    num = p + (dividends.reindex(p.index).fillna(0) if dividends is not None else 0)
    r = num / p.shift(1) - 1
    if kind == "log":
        r = np.log1p(r)
    elif kind != "simple":
        raise ValueError("kind must be 'simple' or 'log'")
    return r.iloc[1:]


# =============================================================================
# Betas and cost of equity
# =============================================================================
def market_model(r_i, r_m, rf=None, hac_lags: int | None = None, periods_per_year: int = 252) -> dict:
    """OLS of the asset's return on the market's: r_i = alpha + beta r_m + e (market model);
    with rf, the CAPM regression in excess returns (alpha = Jensen's alpha). SEs are
    Newey-West HAC (default lag rule floor(4 (n/100)^(2/9))).

    Choices that move the beta more than the estimator does (Ang ch. 5; Damodaran): the market
    proxy (Ibovespa vs IBrX-100 vs MSCI), frequency (daily betas of illiquid stocks are biased
    toward zero by non-synchronous trading: use weekly or monthly), and window (2 years weekly or
    5 years monthly are common). Report them with the beta."""
    import statsmodels.api as sm
    if rf is not None:
        ri, rm, f = _align(r_i, r_m, rf if isinstance(rf, pd.Series) else pd.Series(rf, index=r_i.index))
        y, x = ri - f, rm - f
    else:
        y, x = _align(r_i, r_m)
    n = len(y)
    lags = _nw_lags(n) if hac_lags is None else hac_lags
    fit = sm.OLS(y.to_numpy(), sm.add_constant(x.to_numpy())).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    ci = fit.conf_int(ALPHA)
    tab = pd.DataFrame({"estimate": fit.params, "se": fit.bse, "t": fit.tvalues, "p": fit.pvalues,
                        "ci_low": ci[:, 0], "ci_high": ci[:, 1]}, index=["alpha", "beta"])
    resid_sd = float(np.std(fit.resid, ddof=2))
    warn = []
    if n < 36:
        warn.append(f"only {n} observations: beta is imprecise")
    return {"method": "CAPM (excess returns)" if rf is not None else "market model", "table": tab,
            "beta": float(fit.params[1]), "beta_se": float(fit.bse[1]), "r2": float(fit.rsquared),
            "alpha_annualized": float(fit.params[0] * periods_per_year),
            "idiosyncratic_vol_annualized": resid_sd * np.sqrt(periods_per_year), "hac_lags": lags, "n": n,
            "warnings": warn, "model": fit}


def rolling_beta(r_i, r_m, window: int = 60, rf=None) -> pd.DataFrame:
    """Beta over a moving window (Ang 5.3) with an OLS standard error and 95% band. Use it to see
    whether the beta is stable before choosing an estimation window; a trend or a break (merger,
    new capital structure) means older data describe a different company."""
    if rf is not None:
        ri, rm, f = _align(r_i, r_m, rf if isinstance(rf, pd.Series) else pd.Series(rf, index=r_i.index))
        y, x = ri - f, rm - f
    else:
        y, x = _align(r_i, r_m)
    cov = y.rolling(window).cov(x)
    var = x.rolling(window).var()
    beta = cov / var
    alpha = y.rolling(window).mean() - beta * x.rolling(window).mean()
    resid_var = (y.rolling(window).var() - beta ** 2 * var) * (window - 1) / (window - 2)
    se = np.sqrt(resid_var / (var * (window - 1)))
    out = pd.DataFrame({"beta": beta, "se": se, "ci_low": beta - 1.96 * se, "ci_high": beta + 1.96 * se,
                        "alpha": alpha})
    return out.dropna()


def adjusted_beta(beta: float, method: str = "blume", se: float | None = None, prior_mean: float = 1.0,
                  prior_sd: float | None = None) -> dict:
    """Shrink a raw beta toward the market (betas mean-revert; extreme estimates are partly noise).

    blume:   0.67 * beta + 0.33 * 1 (Blume 1971/1975; the Bloomberg 'adjusted beta').
    vasicek: weight on the raw beta = prior_sd^2 / (prior_sd^2 + se^2): imprecise betas are shrunk
             more (Vasicek 1973). prior_mean and prior_sd: mean and cross-sectional SD of betas in the
             industry or market (e.g. from the comparables)."""
    if method == "blume":
        w = 0.67
        adj = w * beta + (1 - w) * 1.0
    elif method == "vasicek":
        if se is None or prior_sd is None:
            raise ValueError("Vasicek needs se (of the raw beta) and prior_sd (cross-sectional SD of betas)")
        w = prior_sd ** 2 / (prior_sd ** 2 + se ** 2)
        adj = w * beta + (1 - w) * prior_mean
    else:
        raise ValueError("method must be 'blume' or 'vasicek'")
    return {"method": method, "raw_beta": float(beta), "adjusted_beta": float(adj), "weight_on_raw": float(w)}


def _lever_factor(debt_to_equity, tax_rate, method):
    de = np.asarray(debt_to_equity, float)
    if method == "hamada":
        return (1 - tax_rate) * de
    if method == "harris_pringle":
        return de
    raise ValueError("method must be 'hamada' or 'harris_pringle'")


def unlever_beta(beta_levered, debt_to_equity, tax_rate: float = 0.0, debt_beta: float = 0.0,
                 method: str = "hamada"):
    """Asset (unlevered) beta from an equity beta.

    method "hamada" (fixed amount of debt, tax shield as risky as the debt; with debt_beta > 0 this
    is Conine's 1980 extension): beta_U = (beta_L + beta_D (1 - t) D/E) / (1 + (1 - t) D/E).
    method "harris_pringle" (debt rebalanced to a constant D/E, tax shield as risky as the assets;
    tax_rate is ignored): beta_U = (beta_L + beta_D D/E) / (1 + D/E).
    Use market values of D and E averaged over the beta's estimation window, and each comparable's
    own D/E and marginal tax rate (34% for Brazilian companies under lucro real: 25% IRPJ + 9% CSLL).
    Use the same method to unlever and to relever."""
    k = _lever_factor(debt_to_equity, tax_rate, method)
    return (np.asarray(beta_levered, float) + debt_beta * k) / (1 + k)


def relever_beta(beta_unlevered, debt_to_equity, tax_rate: float = 0.0, debt_beta: float = 0.0,
                 method: str = "hamada"):
    """Equity beta at a target capital structure (same methods as unlever_beta):
    beta_L = beta_U + (beta_U - beta_D) k, with k = (1 - t) D/E (Hamada/Conine) or D/E (Harris-Pringle)."""
    k = _lever_factor(debt_to_equity, tax_rate, method)
    return np.asarray(beta_unlevered, float) + (np.asarray(beta_unlevered, float) - debt_beta) * k


def dimson_beta(r_i, r_m, lags: int = 1, leads: int = 1, rf=None, hac_lags: int | None = None) -> dict:
    """Dimson (1979) beta for thinly traded stocks: regress r_i on the market return at t and its
    lags and leads; beta = sum of the coefficients (SE of the sum from the HAC covariance).
    When a stock does not trade every day (many zero returns, common among Brazilian small caps),
    its price reacts to market news with a delay and the plain OLS beta is biased toward zero.
    Compare with market_model: a Dimson beta well above the OLS beta signals that bias; weekly or
    monthly returns are the other remedy."""
    import statsmodels.api as sm
    if rf is not None:
        ri, rm, f = _align(r_i, r_m, rf if isinstance(rf, pd.Series) else pd.Series(rf, index=r_i.index))
        y, x = ri - f, rm - f
    else:
        y, x = _align(r_i, r_m)
    cols = {"m(t)": x}
    for k in range(1, lags + 1):
        cols[f"m(t-{k})"] = x.shift(k)
    for k in range(1, leads + 1):
        cols[f"m(t+{k})"] = x.shift(-k)
    X = pd.DataFrame(cols)
    d = pd.concat([y.rename("_y"), X], axis=1).dropna()
    n = len(d)
    hl = _nw_lags(n) if hac_lags is None else hac_lags
    fit = sm.OLS(d["_y"], sm.add_constant(d[list(cols)])).fit(cov_type="HAC", cov_kwds={"maxlags": hl})
    a = np.array([0.0] + [1.0] * len(cols))
    b = float(a @ fit.params.to_numpy())
    se = float(np.sqrt(a @ fit.cov_params().to_numpy() @ a))
    ols = market_model(y, x, hac_lags=hl)
    return {"method": f"Dimson beta ({lags} lag(s), {leads} lead(s))", "beta": b, "se": se,
            "ci_low": b - 1.96 * se, "ci_high": b + 1.96 * se, "coefficients": fit.params.drop("const"),
            "ols_beta": ols["beta"], "share_zero_returns": float((y == 0).mean()), "n": n}


def factor_model(r_i, factors: pd.DataFrame, rf=None, hac_lags: int | None = None,
                 periods_per_year: int = 12) -> dict:
    """Time-series factor regression: (r_i - rf) = alpha + sum_k b_k f_k + e, HAC SEs.

    factors: DataFrame of factor returns on the same dates (e.g. Fama-French Mkt-RF, SMB, HML,
    RMW, CMA, Mom from Kenneth French's library; for Brazil, NEFIN-USP's Market, SMB, HML, WML, IML
    and its risk-free rate). Pass rf=None when r_i is already an excess return. A significant alpha
    is abnormal performance relative to the factors, or a missing factor."""
    import statsmodels.api as sm
    y = r_i if rf is None else r_i - (rf if isinstance(rf, pd.Series) else rf)
    d = pd.concat([y.rename("_y"), factors], axis=1, join="inner").dropna()
    n = len(d)
    lags = _nw_lags(n) if hac_lags is None else hac_lags
    fit = sm.OLS(d["_y"], sm.add_constant(d[factors.columns])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    ci = fit.conf_int(ALPHA)
    tab = pd.DataFrame({"estimate": fit.params, "se": fit.bse, "t": fit.tvalues, "p": fit.pvalues,
                        "ci_low": ci[0], "ci_high": ci[1]}).rename(index={"const": "alpha"})
    return {"method": "factor model", "table": tab, "r2": float(fit.rsquared),
            "alpha_annualized": float(fit.params["const"] * periods_per_year), "hac_lags": lags, "n": n, "model": fit}


# =============================================================================
# Event study
# =============================================================================
def event_study(r: pd.DataFrame, r_m: pd.Series | None, events, est_win=(-250, -30), evt_win=(-5, 5),
                model: str = "market", car_windows=((-1, 1),)) -> dict:
    """Short-run event study (MacKinlay 1997; Campbell, Lo & MacKinlay ch. 4; Ang 5.5).

    r: returns, one column per firm, indexed by trading date; r_m: market returns (None for the
    constant-mean model); events: dict {firm: date} or DataFrame with columns firm, date (the event
    day is the first trading day on or after the date). Windows are in trading days relative to
    day 0; the estimation window must end before the event window starts.
    model "market" (alpha + beta r_m), "market_adjusted" (r - r_m, no estimation) or "mean".

    Tests of the cumulative average abnormal return (CAAR):
      cross-sectional t  : uses the dispersion of CARs across events (robust to event-induced variance);
      Patell Z           : standardizes each AR by its estimation-window SD; over-rejects if the event
                           itself raises volatility;
      BMP (Boehmer, Musumeci & Poulsen 1991): standardized CARs with cross-sectional variance; the
                           default for announcements, which typically raise volatility;
      Corrado rank       : nonparametric test of the day-0 average AR (robust to non-normal returns).
    Overlapping event windows of firms in the same industry and date (clustering) make all of these
    over-reject: use a portfolio approach or calendar-time regressions then."""
    if isinstance(events, dict):
        ev = pd.DataFrame({"firm": list(events), "date": pd.to_datetime(list(events.values()))})
    else:
        ev = events.rename(columns=str.lower)[["firm", "date"]].copy()
        ev["date"] = pd.to_datetime(ev["date"])
    if est_win[1] >= evt_win[0]:
        raise ValueError("the estimation window must end before the event window starts")
    idx = r.index
    days = np.arange(evt_win[0], evt_win[1] + 1)
    L1 = est_win[1] - est_win[0] + 1
    rows, ar_mat, sar_mat, rank_day0, skipped = [], [], [], [], []
    for _, e in ev.iterrows():
        f = e["firm"]
        pos = idx.searchsorted(e["date"])
        if f not in r.columns or pos + evt_win[1] >= len(idx) or pos + est_win[0] < 0:
            skipped.append(f)
            continue
        est = slice(pos + est_win[0], pos + est_win[1] + 1)
        evw = slice(pos + evt_win[0], pos + evt_win[1] + 1)
        ye, yv = r[f].iloc[est].to_numpy(float), r[f].iloc[evw].to_numpy(float)
        if np.isnan(ye).mean() > 0.2 or np.isnan(yv).any():
            skipped.append(f)
            continue
        if model == "market":
            xe, xv = r_m.iloc[est].to_numpy(float), r_m.iloc[evw].to_numpy(float)
            ok = ~np.isnan(ye) & ~np.isnan(xe)
            X = np.column_stack([np.ones(ok.sum()), xe[ok]])
            b = np.linalg.lstsq(X, ye[ok], rcond=None)[0]
            res_e = ye[ok] - X @ b
            s2 = res_e @ res_e / (ok.sum() - 2)
            ar = yv - (b[0] + b[1] * xv)
            xbar, sxx = xe[ok].mean(), ((xe[ok] - xe[ok].mean()) ** 2).sum()
            adj = 1 + 1 / ok.sum() + (xv - xbar) ** 2 / sxx          # prediction-error correction
            ar_est = res_e
        elif model == "market_adjusted":
            ar_est = ye - r_m.iloc[est].to_numpy(float)
            ar = yv - r_m.iloc[evw].to_numpy(float)
            s2 = np.nanvar(ar_est, ddof=1)
            adj = np.ones(len(days))
        elif model == "mean":
            mu = np.nanmean(ye)
            ar_est = ye - mu
            ar = yv - mu
            s2 = np.nanvar(ye, ddof=1)
            adj = np.ones(len(days)) * (1 + 1 / L1)
        else:
            raise ValueError("model must be 'market', 'market_adjusted' or 'mean'")
        sar = ar / np.sqrt(s2 * adj)
        ar_mat.append(ar)
        sar_mat.append(sar)
        allar = np.concatenate([ar_est[~np.isnan(ar_est)], ar])
        rk = stats.rankdata(allar)
        rank_day0.append((rk[len(allar) - len(ar) + list(days).index(0)] - (len(allar) + 1) / 2,
                          rk - (len(allar) + 1) / 2, len(ar)))
        rows.append({"firm": f, "event_date": idx[pos], "s2": s2, "L1": int(np.sum(~np.isnan(ar_est)))})
    if not ar_mat:
        raise ValueError("no event could be evaluated (check dates, windows and data coverage)")
    AR = pd.DataFrame(ar_mat, columns=days, index=[x["firm"] for x in rows])
    SAR = pd.DataFrame(sar_mat, columns=days, index=AR.index)
    N = len(AR)
    aar = AR.mean()
    aar_t = aar / (AR.std(ddof=1) / np.sqrt(N))
    aar_tab = pd.DataFrame({"AAR": aar, "t_cross_section": aar_t, "p": 2 * stats.t.sf(np.abs(aar_t), N - 1)})

    def caar_tests(w):
        cols = [d for d in days if w[0] <= d <= w[1]]
        car = AR[cols].sum(1)
        T = len(cols)
        scar = SAR[cols].sum(1) / np.sqrt(T)
        caar = car.mean()
        t_cs = caar / (car.std(ddof=1) / np.sqrt(N))
        L = np.array([x["L1"] for x in rows])
        patell = scar.sum() / np.sqrt(np.sum((L - 2) / (L - 4)))
        bmp = scar.mean() / (scar.std(ddof=1) / np.sqrt(N))
        return {"window": f"({w[0]}, {w[1]})", "CAAR": float(caar), "N": N,
                "t_cross_section": float(t_cs), "p_cross_section": float(2 * stats.t.sf(abs(t_cs), N - 1)),
                "patell_z": float(patell), "p_patell": float(2 * stats.norm.sf(abs(patell))),
                "bmp_t": float(bmp), "p_bmp": float(2 * stats.t.sf(abs(bmp), N - 1))}, car

    caar_rows, cars = [], {}
    for w in [tuple(evt_win)] + [tuple(x) for x in car_windows]:
        res, car = caar_tests(w)
        caar_rows.append(res)
        cars[res["window"]] = car
    # Corrado (1989) rank test for day 0, with the SD of average ranks over the whole sample period
    K0 = np.array([x[0] for x in rank_day0])
    maxlen = max(len(x[1]) for x in rank_day0)
    Kbar = np.nanmean(np.array([np.pad(x[1], (maxlen - len(x[1]), 0), constant_values=np.nan) for x in rank_day0]), axis=0)
    s_k = np.sqrt(np.nanmean(Kbar ** 2))
    corrado = K0.mean() / s_k
    return {"method": f"event study ({model} model)", "caar": pd.DataFrame(caar_rows).set_index("window"),
            "aar": aar_tab, "ar": AR, "car": pd.DataFrame(cars), "corrado_day0_z": float(corrado),
            "p_corrado_day0": float(2 * stats.norm.sf(abs(corrado))), "n_events": N, "skipped": skipped}


# =============================================================================
# Performance
# =============================================================================
def performance_ratios(r_p: pd.Series, r_b: pd.Series | None = None, rf=0.0, periods_per_year: int = 252,
                       mar: float = 0.0) -> pd.DataFrame:
    """Risk-adjusted performance (Ang ch. 6), annualized.

    Sharpe = mean excess / SD (with the Lo 2002 i.i.d. SE: sqrt((1 + SR^2/2)/n) per period);
    Sortino = mean excess / downside deviation below mar; Treynor = mean excess / beta vs the
    benchmark; information ratio = mean active return / tracking error; Jensen's alpha from the
    CAPM regression on the benchmark; Roy safety-first = (mean - mar) / SD.
    For a strategy's full performance table (drawdowns, turnover) see backtest.performance."""
    k = periods_per_year
    rp = r_p.dropna()
    if r_b is not None:
        rp, rb = _align(r_p, r_b)
    f = rf.reindex(rp.index) if isinstance(rf, pd.Series) else rf
    ex = rp - f
    n = len(ex)
    sr = ex.mean() / ex.std(ddof=1)
    rows = {"mean excess return (ann.)": ex.mean() * k, "volatility (ann.)": rp.std(ddof=1) * np.sqrt(k),
            "Sharpe (ann.)": sr * np.sqrt(k), "Sharpe SE (ann., Lo 2002)": np.sqrt((1 + sr ** 2 / 2) / n) * np.sqrt(k)}
    down = np.minimum(rp - mar, 0)
    rows["Sortino (ann.)"] = (ex.mean() * k) / (np.sqrt((down ** 2).mean()) * np.sqrt(k))
    rows["Roy safety-first"] = (rp.mean() - mar) / rp.std(ddof=1)
    if r_b is not None:
        exb = rb - f
        beta = np.cov(ex, exb, ddof=1)[0, 1] / exb.var(ddof=1)
        alpha = ex.mean() - beta * exb.mean()
        active = rp - rb
        rows.update({"beta vs benchmark": beta, "Treynor (ann.)": ex.mean() * k / beta,
                     "Jensen alpha (ann.)": alpha * k,
                     "tracking error (ann.)": active.std(ddof=1) * np.sqrt(k),
                     "information ratio (ann.)": active.mean() / active.std(ddof=1) * np.sqrt(k)})
    out = pd.Series(rows, name="value").to_frame()
    out.attrs["n"] = n
    return out


# =============================================================================
# Portfolios
# =============================================================================
def shrink_cov(rets: pd.DataFrame) -> pd.DataFrame:
    """Ledoit-Wolf shrinkage covariance: the sample covariance of many assets is noisy and its
    inverse (used by mean-variance weights) amplifies the noise. Shrinkage toward a structured
    target gives more stable weights."""
    from sklearn.covariance import LedoitWolf
    d = rets.dropna()
    lw = LedoitWolf().fit(d.to_numpy())
    out = pd.DataFrame(lw.covariance_, index=d.columns, columns=d.columns)
    out.attrs["shrinkage"] = float(lw.shrinkage_)
    return out


def efficient_frontier(mu, cov, rf: float | None = None, short: bool = True, n_points: int = 50) -> dict:
    """Markowitz mean-variance frontier (Ang ch. 7).

    short=True: closed form (Merton 1972) for the global minimum-variance (GMV) portfolio, the
    tangency portfolio (max Sharpe, needs rf) and the frontier; short=False: long-only weights by
    quadratic programming (SLSQP). mu and cov in the same period units (e.g. annualized).
    Estimated means are very noisy: the optimizer magnifies their errors ("error maximization").
    Prefer shrink_cov for the covariance, long-only or bounded weights, and treat the tangency
    portfolio as illustrative unless the mean estimates are well founded."""
    mu = pd.Series(mu, dtype=float)
    S = pd.DataFrame(cov).loc[mu.index, mu.index].to_numpy(float)
    m = mu.to_numpy()
    k = len(m)
    names = list(mu.index)
    one = np.ones(k)

    def stats_of(w):
        ret, vol = float(w @ m), float(np.sqrt(w @ S @ w))
        return ret, vol, (ret - rf) / vol if rf is not None else np.nan

    if short:
        Si = np.linalg.inv(S)
        A, B, C = one @ Si @ m, m @ Si @ m, one @ Si @ one
        w_gmv = Si @ one / C
        targets = np.linspace(min(m.min(), w_gmv @ m), max(m.max(), w_gmv @ m) * 1.5, n_points)
        D = B * C - A ** 2
        front = []
        for tgt in targets:
            lam, gam = (C * tgt - A) / D, (B - A * tgt) / D
            w = Si @ (lam * m + gam * one)
            front.append((tgt, np.sqrt(w @ S @ w)))
        w_tan = None
        if rf is not None:
            z = Si @ (m - rf * one)
            w_tan = z / z.sum()
    else:
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        bnds = [(0, 1)] * k
        w0 = one / k
        r = optimize.minimize(lambda w: w @ S @ w, w0, bounds=bnds, constraints=cons, method="SLSQP")
        w_gmv = r.x
        front = []
        for tgt in np.linspace(w_gmv @ m, m.max(), n_points):
            c2 = cons + [{"type": "eq", "fun": lambda w, t=tgt: w @ m - t}]
            rr = optimize.minimize(lambda w: w @ S @ w, w0, bounds=bnds, constraints=c2, method="SLSQP")
            if rr.success:
                front.append((tgt, np.sqrt(rr.fun)))
        w_tan = None
        if rf is not None:
            rr = optimize.minimize(lambda w: -(w @ m - rf) / np.sqrt(w @ S @ w), w0, bounds=bnds,
                                   constraints=cons, method="SLSQP")
            w_tan = rr.x
    out = {"method": "mean-variance (" + ("short sales allowed" if short else "long only") + ")",
           "gmv_weights": pd.Series(w_gmv, index=names), "frontier": pd.DataFrame(front, columns=["return", "volatility"])}
    g = stats_of(w_gmv)
    out["gmv"] = {"return": g[0], "volatility": g[1], "sharpe": g[2]}
    if w_tan is not None:
        t = stats_of(w_tan)
        out["tangency_weights"] = pd.Series(w_tan, index=names)
        out["tangency"] = {"return": t[0], "volatility": t[1], "sharpe": t[2]}
        if short and (w_tan < -0.5).any():
            warnings.warn("tangency portfolio has large short positions: mean estimates are probably driving it")
    return out
