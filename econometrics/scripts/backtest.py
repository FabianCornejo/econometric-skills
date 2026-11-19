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
walk_forward_splits(n, ...)       expanding or rolling train/test index windows (horizon= purges look-ahead labels)
purged_kfold_splits(n, ...)       K-fold CV with purging and embargo for labels that look ahead
label_overlap(horizon, n)         how much overlapping labels shrink the information and the naive SE
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


def _whole(x, name: str) -> int:
    """x as an int, refusing fractions (horizon=2.9 must not become 2 silently)."""
    if isinstance(x, bool) or x != int(x):
        raise ValueError(f"{name} must be a whole number, got {x!r}")
    return int(x)


def walk_forward_splits(n: int, train_size: int, test_size: int, expanding: bool = False,
                        step: int | None = None, horizon: int = 1):
    """Yield (train_idx, test_idx) integer arrays. Never shuffles; test always follows train.

    horizon (v2.7.0): how many periods each label looks ahead. The label of row i is built from
    periods i, ..., i + horizon - 1 (a 21-day return, "did the index rise over the next month" computed
    from daily prices). The last horizon - 1 training rows then have labels that run into the test
    window, so they are dropped from the training set (purging, López de Prado 2018, §7.4.1); the
    training window still ends where it would without them, so the test windows do not move. horizon=1
    (the default) gives exactly the splits of earlier versions. An embargo changes nothing here, because
    training never follows the test window; it matters in K-fold cross-validation (purged_kfold_splits).
    The splits equal sklearn's TimeSeriesSplit(test_size=test_size, gap=horizon - 1) when step = test_size and
    n - train_size is a multiple of test_size (rolling windows: max_train_size = train_size - horizon + 1);
    TimeSeriesSplit anchors its folds at the end of the sample, this function at the start.
    Invalid arguments raise ValueError when the function is called, before any split is drawn."""
    horizon = _whole(horizon, "horizon")
    if horizon < 1:
        raise ValueError("horizon must be at least 1 (the label of row i uses periods i .. i + horizon - 1)")
    if train_size - (horizon - 1) < 1:
        raise ValueError(f"train_size = {train_size} leaves no training row after purging horizon - 1 = "
                         f"{horizon - 1} rows; use a longer training window")
    step = step or test_size

    def _gen():
        start = 0
        while True:
            tr_start = 0 if expanding else start
            tr_end = start + train_size
            te_end = tr_end + test_size
            if te_end > n:
                break
            yield np.arange(tr_start, tr_end - (horizon - 1)), np.arange(tr_end, te_end)
            start += step
    return _gen()


def purged_kfold_splits(n: int, n_splits: int = 5, horizon: int = 1, embargo: int = 0,
                        show: bool = True) -> dict:
    """K-fold cross-validation for labels that look ahead (López de Prado 2018, §7.4: purging and embargo).

    The rows are cut into n_splits contiguous blocks, never shuffled; each block is the test set once.
    The label of row i uses periods i, ..., i + horizon - 1. A training row is dropped when its label
    shares a period with a test label (purging) and, after each test block, the `embargo` rows that
    follow the end of the last test label are dropped as well, because features and labels of the
    periods right after the test window are correlated with it. Training rows before and after the test
    block are kept: this is cross-validation for choosing a model or its tuning, not a backtest; the
    final evaluation of a forecast stays walk-forward (walk_forward_splits with the same horizon).

    Returns a dict with `splits` (list of (train_idx, test_idx)), `table` (per fold: test rows, training
    rows kept, rows purged, rows embargoed), `used` and `warnings` (level, message, action)."""
    from reporting import SKILL_VERSION, _finding, _print_findings
    n, n_splits, horizon, embargo = (_whole(n, "n"), _whole(n_splits, "n_splits"), _whole(horizon, "horizon"),
                                     _whole(embargo, "embargo"))
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if n < n_splits:
        raise ValueError(f"n = {n} rows cannot be cut into {n_splits} blocks")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if embargo < 0:
        raise ValueError("embargo must be 0 or more rows")
    idx = np.arange(n)
    splits, rows = [], []
    for k, test in enumerate(np.array_split(idx, n_splits)):
        a, b = int(test[0]), int(test[-1])
        # train row j is purged if [j, j + horizon - 1] meets [a, b + horizon - 1], i.e. a - horizon + 1 <= j <= b + horizon - 1;
        # the embargo extends the end of the test labels by `embargo` rows
        lo, hi_purge = a - horizon + 1, b + horizon - 1
        hi = hi_purge + embargo
        keep = (idx < lo) | (idx > hi)
        purged = int(((idx >= lo) & (idx <= hi_purge) & ((idx < a) | (idx > b))).sum())
        embargoed = int(((idx > hi_purge) & (idx <= hi)).sum())
        splits.append((idx[keep], test))
        rows.append({"fold": k + 1, "test_start": a, "test_end": b, "n_test": len(test),
                     "n_train": int(keep.sum()), "purged": purged, "embargoed": embargoed})
    table = pd.DataFrame(rows).set_index("fold")
    if (table["n_train"] == 0).any():
        raise ValueError(f"purging and embargo leave no training row in fold(s) "
                         f"{list(table.index[table['n_train'] == 0])}: use fewer blocks, a shorter horizon or embargo")
    warns = []
    lost = float((1 - table["n_train"] / (n - table["n_test"])).max())     # worst fold, its own rows
    if lost > 0.5:
        warns.append(_finding("warning", f"purging and embargo remove up to {lost:.0%} of the training rows in a fold "
                              f"(horizon {horizon}, embargo {embargo}, {n} rows in {n_splits} blocks)",
                              "use fewer, larger blocks, a shorter label horizon, or a longer sample"))
    if horizon > 1 and embargo == 0:
        warns.append(_finding("notice", f"no embargo: rows right after each test block are kept for training although "
                              f"their features may still carry information from it",
                              f"set embargo to a few rows (about 1% of the sample, {max(1, n // 100)} here) unless the "
                              f"features are known not to look back into the test window"))
    used = {"n": n, "n_splits": n_splits, "horizon": horizon, "embargo": embargo, "shuffle": False,
            "skill_version": SKILL_VERSION}
    if show:
        print(f"\nPurged K-fold: {n} rows, {n_splits} contiguous blocks, label horizon {horizon}, embargo {embargo}")
        print(table.to_string())
        _print_findings(warns)
    return {"splits": splits, "table": table, "used": used, "warnings": warns}


def label_overlap(horizon: int, n: int, show: bool = True) -> dict:
    """How much n labels that each look `horizon` periods ahead overlap (López de Prado 2018, §4.2-4.4;
    Hansen and Hodrick 1980 for the moving-average error).

    The label of row i uses periods i, ..., i + horizon - 1, so neighbouring labels share horizon - 1
    periods: monthly returns computed every day from daily prices are 21-day overlapping sums. Returns
    - `independent_labels`: ceil(n / horizon), the labels left when only non-overlapping ones are kept;
    - `average_uniqueness`: the mean over labels of the average of 1/c_t over each label's periods, c_t
      being the number of labels that use period t (1 without overlap, about 1/horizon with it);
    - `se_ratio`: by how much the naive standard error of the mean of the labels (SD / sqrt(n)) is too
      small when the labels are sums of horizon i.i.d. shocks, sqrt(sum_t c_t^2 / (n * horizon)), about
      sqrt(horizon) for long samples (against the true SD of a label; the SD computed from the same
      overlapping labels is itself too small, so a naive SE built from it is smaller still). A
      t-statistic computed as if the labels were independent is inflated by at least that factor; with real returns (not i.i.d.) the factor differs, so use HAC errors with
      horizon - 1 lags or non-overlapping labels rather than dividing by it."""
    from reporting import SKILL_VERSION, _finding, _print_findings
    horizon, n = _whole(horizon, "horizon"), _whole(n, "n")
    if horizon < 1 or n < 1:
        raise ValueError("horizon and n must be at least 1")
    span = n + horizon - 1                                    # periods 0 .. n + horizon - 2
    c = np.zeros(span)
    for j in range(horizon):                                  # c_t = number of labels i with i <= t <= i + horizon - 1
        c[j:j + n] += 1
    inv = 1.0 / c
    csum = np.concatenate([[0.0], np.cumsum(inv)])
    uniq = (csum[horizon:horizon + n] - csum[:n]) / horizon  # mean of 1/c_t over periods i .. i + horizon - 1
    se_ratio = float(np.sqrt((c ** 2).sum() / (n * horizon)))
    out = {"n": n, "horizon": horizon, "independent_labels": int(np.ceil(n / horizon)),
           "average_uniqueness": float(uniq.mean()), "se_ratio": se_ratio, "uniqueness": uniq}
    warns = []
    if horizon > 1 and n > 1:
        warns.append(_finding(
            "warning",
            f"labels overlap: each shares {horizon - 1} of its {horizon} periods with its neighbour, so {n} labels carry "
            f"about the information of {out['independent_labels']} independent ones, and a standard error that treats them "
            f"as independent is about {se_ratio:.2f} times too small (i.i.d. shocks)",
            f"split with backtest.walk_forward_splits(horizon={horizon}) or purged_kfold_splits(horizon={horizon}); "
            f"use HAC standard errors with {horizon - 1} lags, or keep one label every {horizon} periods; judge accuracy against "
            "the base rate"))
    out["used"] = {"n": n, "horizon": horizon, "skill_version": SKILL_VERSION}
    out["warnings"] = warns
    if show:
        print(f"\nLabel overlap: {n} labels, each over {horizon} periods")
        print(f"  independent labels: {out['independent_labels']}; average uniqueness {out['average_uniqueness']:.3f}; "
              f"naive SE too small by x{se_ratio:.2f}")
        _print_findings(warns)
    return out


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
