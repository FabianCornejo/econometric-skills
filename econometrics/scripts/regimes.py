"""
Regime detection: Markov-switching models (Hamilton, 1989), Gaussian hidden Markov
models and K-means clustering of market features, with the out-of-sample discipline
these methods usually lack.

Look-ahead traps handled here:
  * SMOOTHED probabilities P(s_t | all data) use the future. For real-time use
    (signals, backtests) use FILTERED probabilities P(s_t | data up to t).
  * Scalers, HMMs and K-means centroids must be fitted on the training window only.
  * Regime labels are arbitrary (state 0 in one fit can be state 1 in another);
    they are relabelled here by increasing volatility so 0 = calmest.

Functions
---------
markov_switching(y, k, ...)         statsmodels MarkovRegression/MarkovAutoregression + summary
regime_features(prices, ...)        returns, realized vol, drawdown, trend, range (no 'ta' needed)
kmeans_regimes(X, k, train_end)     K-means fitted on train, applied to all; validity indices
hmm_regimes(X, k, train_end)        Gaussian HMM (hmmlearn) with FILTERED state probabilities
forward_filter(loglik, pi0, P)      generic HMM forward recursion (filtered probabilities)
regime_summary(returns, labels)     frequency, mean, vol, Sharpe, average duration per regime
compare_labels(a, b)                adjusted Rand index + cross-tabulation
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def _relabel_by(values: pd.Series, labels: np.ndarray) -> tuple[np.ndarray, dict]:
    """Map arbitrary labels to 0..k-1 ordered by the mean of `values` within each label."""
    order = pd.Series(values.to_numpy()).groupby(labels).mean().sort_values().index.tolist()
    mapping = {old: new for new, old in enumerate(order)}
    return np.vectorize(mapping.get)(labels), mapping


# ---------------------------------------------------------------------------
# Markov switching (econometric workhorse)
# ---------------------------------------------------------------------------
def markov_switching(y: pd.Series, k_regimes: int = 2, order: int = 0, trend: str = "c",
                     switching_variance: bool = True, exog=None, search_reps: int = 20,
                     show: bool = True):
    """Markov-switching mean/variance (order=0) or autoregression (order>0).

    Returns dict with the fitted result, transition matrix, expected durations and
    filtered/smoothed probabilities (columns ordered from lowest to highest variance).
    """
    import statsmodels.api as sm
    y = pd.Series(y).dropna().astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if order == 0:
            mod = sm.tsa.MarkovRegression(y, k_regimes=k_regimes, trend=trend, exog=exog,
                                          switching_variance=switching_variance)
        else:
            mod = sm.tsa.MarkovAutoregression(y, k_regimes=k_regimes, order=order, trend=trend,
                                              switching_variance=switching_variance)
        res = mod.fit(search_reps=search_reps, disp=False)
    names = list(res.model.param_names)
    prm = np.asarray(res.params)
    sig = [prm[names.index(f"sigma2[{i}]")] if f"sigma2[{i}]" in names else prm[names.index("sigma2")]
           for i in range(k_regimes)]
    order_idx = np.argsort(sig)
    P = res.regime_transition[:, :, 0]               # P[i, j] = P(s_t = i | s_{t-1} = j)
    P = P[np.ix_(order_idx, order_idx)].T            # rows = from, cols = to
    durations = 1 / (1 - np.diag(P))
    filt = res.filtered_marginal_probabilities
    smth = res.smoothed_marginal_probabilities
    filt = pd.DataFrame(np.asarray(filt)[:, order_idx], index=y.index[-len(filt):])
    smth = pd.DataFrame(np.asarray(smth)[:, order_idx], index=y.index[-len(smth):])
    regimes = []
    for new, old in enumerate(order_idx):
        c = prm[names.index(f"const[{old}]")] if f"const[{old}]" in names else np.nan
        regimes.append({"regime": new, "mean": c, "variance": sig[old], "sd": np.sqrt(sig[old]),
                        "expected_duration": durations[new], "share_time_smoothed": smth[new].mean()})
    summary = pd.DataFrame(regimes).set_index("regime")
    if show:
        print(f"\nMarkov switching, {k_regimes} regimes (0 = lowest variance) | loglik {res.llf:.1f}, AIC {res.aic:.1f}")
        print(summary.to_string(float_format=lambda v: f"{v:.4f}"))
        print("Transition matrix (row = from, col = to):")
        print(pd.DataFrame(P).to_string(float_format=lambda v: f"{v:.3f}"))
        print("Use filtered probabilities for anything real-time; smoothed ones look into the future.")
    return {"result": res, "summary": summary, "transition": pd.DataFrame(P),
            "filtered": filt, "smoothed": smth}


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def regime_features(prices: pd.Series, vol_window: int = 20, trend_window: int = 50,
                    high: pd.Series | None = None, low: pd.Series | None = None) -> pd.DataFrame:
    """Backward-looking features (each row uses data up to that date only)."""
    p = prices.astype(float)
    r = np.log(p).diff()
    f = pd.DataFrame(index=p.index)
    f["ret_1"] = r
    f["ret_20"] = np.log(p / p.shift(20))
    f["vol_20"] = r.rolling(vol_window).std() * np.sqrt(252)
    f["vol_ratio"] = r.rolling(5).std() / r.rolling(60).std()
    f["drawdown"] = p / p.cummax() - 1
    f["trend"] = p / p.rolling(trend_window).mean() - 1
    if high is not None and low is not None:
        f["range"] = (high - low) / p
    return f.replace([np.inf, -np.inf], np.nan).dropna()


# ---------------------------------------------------------------------------
# K-means
# ---------------------------------------------------------------------------
def kmeans_regimes(X: pd.DataFrame, k: int = 3, train_end=None, order_by: str | None = None,
                   seed: int = 42, show: bool = True) -> dict:
    """Fit scaler + K-means on X.loc[:train_end], assign every row. Labels are ordered
    by the training mean of `order_by` (default: first column containing 'vol')."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
    from sklearn.preprocessing import StandardScaler
    X = X.dropna()
    tr = X.loc[:train_end] if train_end is not None else X
    sc = StandardScaler().fit(tr)
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(sc.transform(tr))
    lab_all = km.predict(sc.transform(X))
    key = order_by or next((c for c in X.columns if "vol" in c), X.columns[0])
    _, mapping = _relabel_by(tr[key], km.labels_)
    labels = pd.Series(np.vectorize(mapping.get)(lab_all), index=X.index, name="kmeans_regime")
    Ztr = sc.transform(tr)
    scores = {"silhouette": silhouette_score(Ztr, km.labels_, sample_size=min(5000, len(Ztr)), random_state=seed),
              "davies_bouldin": davies_bouldin_score(Ztr, km.labels_),
              "calinski_harabasz": calinski_harabasz_score(Ztr, km.labels_)}
    if show:
        print(f"\nK-means (k={k}) fitted on {len(tr)} training rows, ordered by {key}: "
              f"silhouette {scores['silhouette']:.3f} (>0.5 strong, <0.25 weak structure), "
              f"Davies-Bouldin {scores['davies_bouldin']:.2f} (lower is better)")
    return {"labels": labels, "scores": scores, "model": km, "scaler": sc}


def choose_k(X: pd.DataFrame, ks=range(2, 7), train_end=None, seed: int = 42) -> pd.DataFrame:
    rows = []
    for k in ks:
        r = kmeans_regimes(X, k, train_end, seed=seed, show=False)
        rows.append({"k": k, **r["scores"], "inertia": r["model"].inertia_})
    return pd.DataFrame(rows).set_index("k")


# ---------------------------------------------------------------------------
# Hidden Markov model
# ---------------------------------------------------------------------------
def forward_filter(loglik: np.ndarray, pi0: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Filtered probabilities P(s_t | x_1..x_t). loglik[t, j] = log p(x_t | s_t = j);
    P[i, j] = P(s_t = j | s_{t-1} = i)."""
    T, k = loglik.shape
    alpha = np.zeros((T, k))
    prior = pi0
    for t in range(T):
        m = loglik[t].max()
        a = prior * np.exp(loglik[t] - m)
        alpha[t] = a / a.sum()
        prior = alpha[t] @ P
    return alpha


def hmm_regimes(X: pd.DataFrame, n_states: int = 3, train_end=None, covariance_type: str = "diag",
                order_by: str | None = None, seed: int = 42, show: bool = True) -> dict:
    """Gaussian HMM (hmmlearn) fitted on the training window. Returns FILTERED state
    probabilities for every row (real-time usable) and the Viterbi path on the training
    window (in-sample description only)."""
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError as e:
        raise ImportError("pip install --break-system-packages hmmlearn  "
                          "(or use markov_switching, which only needs statsmodels)") from e
    from scipy.stats import multivariate_normal
    from sklearn.preprocessing import StandardScaler
    X = X.dropna()
    tr = X.loc[:train_end] if train_end is not None else X
    sc = StandardScaler().fit(tr)
    Ztr, Z = sc.transform(tr), sc.transform(X)
    best = None
    for s in range(5):                                  # EM is local: several starts
        m = GaussianHMM(n_components=n_states, covariance_type=covariance_type, n_iter=500,
                        random_state=seed + s).fit(Ztr)
        sc_ = m.score(Ztr)
        if best is None or sc_ > best[1]:
            best = (m, sc_)
    m = best[0]
    covs = m.covars_
    ll = np.column_stack([multivariate_normal(m.means_[j], covs[j], allow_singular=True).logpdf(Z)
                          for j in range(n_states)])
    filt = forward_filter(ll, m.startprob_, m.transmat_)
    key = order_by or next((c for c in X.columns if "vol" in c), X.columns[0])
    _, mapping = _relabel_by(tr[key], m.predict(Ztr))
    inv = np.argsort([mapping.get(j, j) for j in range(n_states)])
    filt = pd.DataFrame(filt[:, inv], index=X.index)
    labels = filt.idxmax(axis=1).rename("hmm_regime")
    P = pd.DataFrame(m.transmat_[np.ix_(inv, inv)])
    if show:
        print(f"\nGaussian HMM ({n_states} states) on {len(tr)} training rows | loglik {best[1]:.1f}")
        print("Transition matrix (row = from, col = to), states ordered by", key)
        print(P.to_string(float_format=lambda v: f"{v:.3f}"))
        print("Expected durations:", np.round(1 / (1 - np.diag(P.to_numpy())), 1))
    return {"labels": labels, "filtered": filt, "transition": P, "model": m, "scaler": sc}


# ---------------------------------------------------------------------------
# Reading the regimes
# ---------------------------------------------------------------------------
def regime_summary(returns: pd.Series, labels: pd.Series, periods: int = 252) -> pd.DataFrame:
    d = pd.concat([returns.rename("r"), labels.rename("s")], axis=1).dropna()
    runs = (d["s"] != d["s"].shift()).cumsum()
    dur = d.groupby(runs)["s"].agg(["first", "size"]).groupby("first")["size"].mean()
    g = d.groupby("s")["r"]
    out = pd.DataFrame({"share_time": g.size() / len(d), "ann_mean": g.mean() * periods,
                        "ann_vol": g.std() * np.sqrt(periods)})
    out["sharpe"] = out["ann_mean"] / out["ann_vol"]
    out["avg_duration"] = dur
    return out


def compare_labels(a: pd.Series, b: pd.Series) -> dict:
    from sklearn.metrics import adjusted_rand_score
    d = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    return {"ARI": adjusted_rand_score(d["a"], d["b"]), "crosstab": pd.crosstab(d["a"], d["b"])}
