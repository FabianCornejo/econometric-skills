"""
Backtesting utilities for signals built from econometric models.

The single most common bug in model-based trading research is timing: a signal
computed with information up to the close of day t can only earn the return of
day t+1. Everything here enforces that convention:

    position_t = f(information up to t)          (decided at the close of t)
    strategy_return_{t+1} = position_t * r_{t+1} - cost * |position_t - position_{t-1}|

Functions
---------
simple_returns / log_returns     price -> returns
run_backtest(position, returns)   strategy returns with one-bar lag and costs
performance(strategy_returns)     CAGR, vol, Sharpe, Sortino, max drawdown, Calmar, hit rate, turnover
walk_forward_splits(n, ...)       expanding or rolling train/test index windows
tsmom_position(prices, ...)       time-series momentum (sign of past return, optional vol targeting)
forecast_position(forecast, ...)  turn a one-step-ahead forecast into a long/short/flat position
buy_and_hold(returns)             benchmark
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS = {"D": 252, "W": 52, "M": 12, "Q": 4}


def simple_returns(prices: pd.Series | pd.DataFrame):
    return prices.pct_change().iloc[1:]


def log_returns(prices: pd.Series | pd.DataFrame):
    return np.log(prices).diff().iloc[1:]


def run_backtest(position: pd.Series, returns: pd.Series, cost_bps: float = 0.0,
                 lag: int = 1) -> pd.DataFrame:
    """Apply positions decided at t to the return of t+lag.

    position : target exposure (e.g. -1, 0, 1, or fractional) indexed like returns,
               where position[t] uses information up to and including t.
    returns  : simple returns of the traded asset, same index.
    cost_bps : one-way transaction cost in basis points per unit of turnover.
    """
    position = position.reindex(returns.index).astype(float)
    held = position.shift(lag).fillna(0.0)            # exposure actually held during t
    turnover = held.diff().abs().fillna(held.abs())
    cost = turnover * cost_bps / 1e4
    strat = held * returns - cost
    return pd.DataFrame({"position_held": held, "asset_return": returns,
                         "turnover": turnover, "cost": cost, "strategy_return": strat})


def _drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def performance(strategy_returns: pd.Series, freq: str = "D", rf: float = 0.0,
                turnover: pd.Series | None = None) -> pd.Series:
    """Standard performance table. rf is the annual risk-free rate (decimal)."""
    r = strategy_returns.dropna()
    k = PERIODS.get(freq, freq) if isinstance(freq, str) else freq
    n = len(r)
    if n == 0:
        return pd.Series(dtype=float)
    equity = (1 + r).cumprod()
    years = n / k
    cagr = equity.iloc[-1] ** (1 / years) - 1 if equity.iloc[-1] > 0 else np.nan
    excess = r - rf / k
    vol = r.std(ddof=1) * np.sqrt(k)
    sharpe = excess.mean() / r.std(ddof=1) * np.sqrt(k) if r.std(ddof=1) > 0 else np.nan
    downside = np.sqrt((np.minimum(excess, 0) ** 2).mean()) * np.sqrt(k)
    sortino = excess.mean() * k / downside if downside > 0 else np.nan
    dd = _drawdown(equity)
    mdd = dd.min()
    active = r[r != 0]
    out = {
        "observations": n,
        "total_return": equity.iloc[-1] - 1,
        "CAGR": cagr,
        "ann_volatility": vol,
        "Sharpe": sharpe,
        "Sharpe_t_stat": sharpe * np.sqrt(years) if pd.notna(sharpe) else np.nan,
        "Sortino": sortino,
        "max_drawdown": mdd,
        "Calmar": cagr / abs(mdd) if mdd < 0 and pd.notna(cagr) else np.nan,
        "hit_rate_active_days": (active > 0).mean() if len(active) else np.nan,
        "exposure_share": (r != 0).mean(),
        "skewness": r.skew(),
        "excess_kurtosis": r.kurt(),
    }
    if turnover is not None:
        out["ann_turnover"] = turnover.reindex(r.index).sum() / years
    return pd.Series(out)


def compare(results: dict[str, pd.Series], freq: str = "D") -> pd.DataFrame:
    """Side-by-side performance of several return streams {name: returns}."""
    return pd.DataFrame({k: performance(v, freq) for k, v in results.items()})


def walk_forward_splits(n: int, train_size: int, test_size: int, expanding: bool = False,
                        step: int | None = None):
    """Yield (train_idx, test_idx) integer arrays. Never shuffles; test always follows train."""
    step = step or test_size
    start = 0
    while True:
        tr_start = 0 if expanding else start
        tr_end = start + train_size
        te_end = tr_end + test_size
        if te_end > n:
            break
        yield np.arange(tr_start, tr_end), np.arange(tr_end, te_end)
        start += step


def tsmom_position(prices: pd.Series, lookback: int = 252, skip: int = 0,
                   threshold: float = 0.0, vol_target: float | None = None,
                   vol_window: int = 60, freq: str = "D", max_leverage: float = 2.0,
                   long_only: bool = False) -> pd.Series:
    """Time-series momentum (Moskowitz, Ooi & Pedersen, 2012).

    Signal at t = sign of the return from t-lookback to t-skip (uses only past prices).
    vol_target (annual, e.g. 0.10) scales the position by target / ex-ante volatility,
    which is how TSMOM is defined in the literature; without it, exposure is ±1.
    """
    past = prices.shift(skip) / prices.shift(lookback) - 1
    pos = pd.Series(np.where(past > threshold, 1.0, np.where(past < -threshold, -1.0, 0.0)),
                    index=prices.index)
    pos[past.isna()] = 0.0
    if long_only:
        pos = pos.clip(lower=0)
    if vol_target is not None:
        k = PERIODS.get(freq, 252)
        sigma = prices.pct_change().rolling(vol_window).std() * np.sqrt(k)
        pos = (pos * (vol_target / sigma)).clip(-max_leverage, max_leverage).fillna(0.0)
    return pos


def forecast_position(forecast: pd.Series, threshold: float = 0.0,
                      long_only: bool = False) -> pd.Series:
    """forecast[t] must be the forecast of r_{t+1} made with data up to t.

    Pass the result straight to run_backtest (which applies the one-bar lag).
    If your forecast series is indexed by the *target* date (forecast of r_t
    stored at t), shift it by -1 first, or call run_backtest(..., lag=0).
    """
    pos = pd.Series(np.where(forecast > threshold, 1.0,
                             np.where(forecast < -threshold, -1.0, 0.0)), index=forecast.index)
    return pos.clip(lower=0) if long_only else pos


def buy_and_hold(returns: pd.Series) -> pd.Series:
    return returns.copy()


def bootstrap_sharpe_ci(strategy_returns: pd.Series, freq: str = "D", n_boot: int = 2000,
                        block: int = 20, alpha: float = 0.05, seed: int = 0):
    """Stationary-ish moving-block bootstrap CI for the annualized Sharpe ratio."""
    r = strategy_returns.dropna().to_numpy()
    k = PERIODS.get(freq, 252)
    n = len(r)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    stats = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, nb)
        s = np.concatenate([r[i:i + block] for i in starts])[:n]
        sd = s.std(ddof=1)
        stats.append(s.mean() / sd * np.sqrt(k) if sd > 0 else np.nan)
    lo, hi = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"sharpe": r.mean() / r.std(ddof=1) * np.sqrt(k), "ci_low": lo, "ci_high": hi,
            "block": block, "n_boot": n_boot}
