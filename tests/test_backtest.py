"""Tests of backtest.py functions added or changed from v2.7.0 on (T4: tests by module).

walk_forward_splits(horizon=), purged_kfold_splits and label_overlap. References do not reuse the implementation:
- splits: brute force over label intervals, row by row, and sklearn's TimeSeriesSplit(gap=) / KFold(shuffle=False);
- label_overlap: the standard error ratio from the autocovariance of overlapping sums, max(h - |k|, 0), summed over
  all pairs of labels (the implementation counts concurrent labels per period), a hand-worked case, and a simulation;
- known truth: a classifier with no skill (labels are signs of future returns) scored by 1-nearest neighbour in time.
  The bands were fixed from claude/medicoes-v2.7.0.md (M2, 200 replications) before the test was written.

Run from the repository root:  python3 tests/test_backtest.py
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold, TimeSeriesSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "econometrics" / "scripts"))
import _console  # noqa: E402,F401
import backtest as bt  # noqa: E402

results = []


def check(name, cond, info=""):
    results.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name, "" if cond else info)


def raises(f, *a, **k):
    try:
        f(*a, **k)
    except ValueError:
        return True
    return False


def quiet(f, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = f(*a, **k)
    return out, buf.getvalue()


# ============================================================ 1. walk_forward_splits(horizon=)
def wf_reference(n, train_size, test_size, expanding, step, h):
    """Brute force: the windows of the original walk-forward; a training row j is kept only if every period of its
    label, j .. j + h - 1, is before the first test row."""
    step = step or test_size
    out, start = [], 0
    while start + train_size + test_size <= n:
        tr0 = 0 if expanding else start
        te = list(range(start + train_size, start + train_size + test_size))
        tr = [j for j in range(tr0, start + train_size) if all(p < te[0] for p in range(j, j + h))]
        out.append((tr, te))
        start += step
    return out


ok = True
cases = 0
for n in (30, 57, 120):
    for train_size in (10, 25):
        for test_size in (3, 7):
            for expanding in (False, True):
                for step in (None, 2):
                    for h in (1, 2, 5, 10):
                        got = [(a.tolist(), b.tolist()) for a, b in bt.walk_forward_splits(n, train_size, test_size,
                                                                                           expanding, step, horizon=h)]
                        ok &= got == wf_reference(n, train_size, test_size, expanding, step, h)
                        cases += 1
check(f"walk_forward_splits: equal to the brute-force purge in {cases} configurations", ok)

# horizon = 1 keeps the splits of earlier versions: train [start, start + train_size), test right after
got = [(a.tolist(), b.tolist()) for a, b in bt.walk_forward_splits(20, 8, 4)]
check("walk_forward_splits: default horizon gives the pre-v2.7.0 windows",
      got == [(list(range(0, 8)), list(range(8, 12))), (list(range(4, 12)), list(range(12, 16))),
              (list(range(8, 16)), list(range(16, 20)))])

# second route: sklearn TimeSeriesSplit(gap = h - 1), expanding and rolling (max_train_size)
ok = True
for h in (1, 3, 8):
    for train_size, test_size, k in ((20, 5, 6), (40, 10, 3)):
        n = train_size + k * test_size
        ours = [(a.tolist(), b.tolist()) for a, b in bt.walk_forward_splits(n, train_size, test_size, True, horizon=h)]
        sk = [(a.tolist(), b.tolist()) for a, b in TimeSeriesSplit(n_splits=k, test_size=test_size, gap=h - 1).split(np.zeros(n))]
        ok &= ours == sk
        ours = [(a.tolist(), b.tolist()) for a, b in bt.walk_forward_splits(n, train_size, test_size, False, horizon=h)]
        sk = [(a.tolist(), b.tolist()) for a, b in TimeSeriesSplit(n_splits=k, test_size=test_size, gap=h - 1,
                                                                   max_train_size=train_size - (h - 1)).split(np.zeros(n))]
        ok &= ours == sk
check("walk_forward_splits: same splits as sklearn TimeSeriesSplit(gap=horizon-1), expanding and rolling", ok)

# boundary of the training window: train_size - (h - 1) = 1 is accepted, 0 is refused; horizon 0 refused
check("walk_forward_splits: one training row left after purging is accepted",
      [a.tolist() for a, _ in bt.walk_forward_splits(12, 4, 2, horizon=4)][0] == [0])
check("walk_forward_splits: no training row left after purging is refused",
      raises(lambda: list(bt.walk_forward_splits(12, 4, 2, horizon=5))))
check("walk_forward_splits: horizon 0 is refused", raises(lambda: list(bt.walk_forward_splits(12, 4, 2, horizon=0))))
check("walk_forward_splits: the test windows do not move with the horizon",
      [b.tolist() for _, b in bt.walk_forward_splits(40, 10, 5, horizon=1)]
      == [b.tolist() for _, b in bt.walk_forward_splits(40, 10, 5, horizon=6)])


# ============================================================ 2. purged_kfold_splits
def kfold_reference(n, k, h, e):
    """Row by row: a training row j is dropped if its label [j, j+h-1] shares a period with the label of any test row
    i, [i, i+h-1], or if it starts within the e rows after the last period of the last test label."""
    out = []
    for te in np.array_split(np.arange(n), k):
        te = te.tolist()
        last = te[-1] + h - 1
        tr = []
        for j in range(n):
            if j in te:
                continue
            if any(j <= i + h - 1 and i <= j + h - 1 for i in te):
                continue
            if last < j <= last + e:
                continue
            tr.append(j)
        out.append((tr, te))
    return out


ok, cases = True, 0
for n in (23, 50, 101):
    for k in (2, 3, 5):
        for h in (1, 2, 4, 9):
            for e in (0, 1, 3):
                r = bt.purged_kfold_splits(n, k, horizon=h, embargo=e, show=False)
                ok &= [(a.tolist(), b.tolist()) for a, b in r["splits"]] == kfold_reference(n, k, h, e)
                t = r["table"]
                ok &= bool(((t["n_train"] + t["n_test"] + t["purged"] + t["embargoed"]) == n).all())
                cases += 1
check(f"purged_kfold_splits: equal to the row-by-row reference in {cases} configurations; table adds up to n", ok)

ok = True
for n, k in ((20, 4), (23, 5), (101, 3)):
    ours = [(a.tolist(), b.tolist()) for a, b in bt.purged_kfold_splits(n, k, show=False)["splits"]]
    sk = [(a.tolist(), b.tolist()) for a, b in KFold(n_splits=k, shuffle=False).split(np.zeros(n))]
    ok &= ours == sk
check("purged_kfold_splits: horizon 1 and no embargo is sklearn's unshuffled KFold", ok)

r = bt.purged_kfold_splits(20, 4, horizon=3, embargo=1, show=False)
check("purged_kfold_splits: hand-worked fold (test 5..9, purge 3-4 and 10-11, embargo 12)",
      r["splits"][1][0].tolist() == [0, 1, 2] + list(range(13, 20)))
check("purged_kfold_splits: no embargo after the last block (nothing follows it)",
      int(r["table"].loc[4, "embargoed"]) == 0 and int(r["table"].loc[4, "purged"]) == 2)
check("purged_kfold_splits: used records the choices",
      r["used"]["horizon"] == 3 and r["used"]["embargo"] == 1 and r["used"]["shuffle"] is False
      and "skill_version" in r["used"])
for bad in (dict(n=10, n_splits=1), dict(n=3, n_splits=4), dict(n=10, horizon=0), dict(n=10, embargo=-1)):
    check(f"purged_kfold_splits: refuses {bad}", raises(bt.purged_kfold_splits, **bad, show=False))

# findings: the embargo notice appears with overlapping labels and no embargo, and only then
lv = lambda r: [w["level"] for w in r["warnings"]]  # noqa: E731
check("purged_kfold_splits: notice when labels overlap and there is no embargo",
      lv(bt.purged_kfold_splits(100, 5, horizon=5, show=False)) == ["notice"])
check("purged_kfold_splits: no notice with an embargo, nor with horizon 1",
      lv(bt.purged_kfold_splits(100, 5, horizon=5, embargo=1, show=False)) == []
      and lv(bt.purged_kfold_splits(100, 5, show=False)) == [])
# the 'too much lost' warning (middle fold of 5 blocks of 20 rows; 80 rows outside the test block): horizon 11 and
# embargo 10 remove 30 (no warning); horizon 21 without embargo removes 40, exactly 50% (no warning); horizon 21 with
# embargo 1 removes 41 (warning). One case on each side of the 50% line.
w_no = bt.purged_kfold_splits(100, 5, horizon=11, embargo=10, show=False)
w_edge = bt.purged_kfold_splits(100, 5, horizon=21, embargo=0, show=False)
w_yes = bt.purged_kfold_splits(100, 5, horizon=21, embargo=1, show=False)
check("purged_kfold_splits: middle-fold removals are 30, 40 and 41 of 80 rows",
      [int(80 - w["table"].loc[3, "n_train"]) for w in (w_no, w_edge, w_yes)] == [30, 40, 41])
check("purged_kfold_splits: no 'rows removed' warning at 30 of 80 nor at exactly 50%",
      "warning" not in lv(w_no) and "warning" not in lv(w_edge), (lv(w_no), lv(w_edge)))
check("purged_kfold_splits: 'rows removed' warning above 50% (41 of 80)", "warning" in lv(w_yes), w_yes["table"])

# ============================================================ 3. label_overlap
def se_ratio_reference(n, h):
    """Variance of the mean of n overlapping sums of h i.i.d. unit shocks: (1/n^2) * sum over pairs of
    max(h - |i - k|, 0); the naive variance is h / n."""
    d = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    return np.sqrt(np.maximum(h - d, 0).sum() / n ** 2 / (h / n))


ok = True
for n in (1, 2, 7, 50, 333):
    for h in (1, 2, 5, 21, 60):
        o = bt.label_overlap(h, n, show=False)
        ok &= abs(o["se_ratio"] - se_ratio_reference(n, h)) <= 1e-10 * se_ratio_reference(n, h)
        ok &= o["independent_labels"] == -(-n // h)
check("label_overlap: SE ratio equal to the autocovariance formula (rel 1e-10) and ceil(n/h) labels", ok)
o = bt.label_overlap(2, 3, show=False)
check("label_overlap: hand-worked uniqueness, n = 3, h = 2 (c = 1, 2, 2, 1): 0.75, 0.5, 0.75",
      np.allclose(o["uniqueness"], [0.75, 0.5, 0.75], rtol=0, atol=1e-15) and abs(o["average_uniqueness"] - 2 / 3) < 1e-15)
o1 = bt.label_overlap(1, 40, show=False)
check("label_overlap: no overlap, no finding, uniqueness 1 and ratio 1",
      o1["warnings"] == [] and o1["average_uniqueness"] == 1.0 and abs(o1["se_ratio"] - 1) < 1e-15)
check("label_overlap: interior labels have uniqueness exactly 1/h",
      np.allclose(bt.label_overlap(5, 30, show=False)["uniqueness"][4:26], 0.2, rtol=0, atol=1e-15))
check("label_overlap: refuses horizon 0", raises(bt.label_overlap, 0, 10, show=False))

# simulation: the spread of the mean of overlapping labels across samples, against the naive SE with the true label
# SD, sqrt(h / n). Band: 4 Monte Carlo SEs of a sample SD, 4 / sqrt(2 (reps - 1)), about 6.3% with 2000 samples.
rng = np.random.default_rng(27001)
n, h, reps = 240, 21, 2000
means, sd2 = [], []
for _ in range(reps):
    s = np.convolve(rng.standard_normal(n + h - 1), np.ones(h), mode="valid")
    means.append(s.mean())
    sd2.append(s.var(ddof=1))
sim_ratio = np.std(means, ddof=1) / np.sqrt(h / n)
target = bt.label_overlap(h, n, show=False)["se_ratio"]
band = 4 / np.sqrt(2 * (reps - 1))
check(f"label_overlap: simulated SE ratio {sim_ratio:.3f} within {band:.1%} of the formula {target:.3f} ({reps} samples)",
      abs(sim_ratio / target - 1) < band)
check(f"label_overlap: the in-sample SD of overlapping labels is biased down (mean variance {np.mean(sd2) / h:.3f} of the "
      "true one), so an SE built from it is smaller still", np.mean(sd2) / h < 0.97)

# ============================================================ 4. known truth: no skill, overlapping labels
def nn_accuracy(y, splits):
    hit = tot = 0
    for tr, te in splits:
        pos = np.searchsorted(tr, te)
        left = tr[np.clip(pos - 1, 0, len(tr) - 1)]
        right = tr[np.clip(pos, 0, len(tr) - 1)]
        nb = np.where(np.abs(te - left) <= np.abs(right - te), left, right)
        hit += int((y[nb] == y[te]).sum())
        tot += len(te)
    return hit / tot


rng = np.random.default_rng(27002)
n, h, reps = 1000, 21, 200
acc = {"shuffled": [], "contiguous": [], "purged": []}
for _ in range(reps):
    y = (np.convolve(rng.standard_normal(n + h - 1), np.ones(h), mode="valid") > 0).astype(int)
    folds = np.array_split(rng.permutation(n), 5)
    acc["shuffled"].append(nn_accuracy(y, [(np.setdiff1d(np.arange(n), f), np.sort(f)) for f in folds]))
    acc["contiguous"].append(nn_accuracy(y, bt.purged_kfold_splits(n, 5, show=False)["splits"]))
    acc["purged"].append(nn_accuracy(y, bt.purged_kfold_splits(n, 5, horizon=h, embargo=h, show=False)["splits"]))
m = {k: float(np.mean(v)) for k, v in acc.items()}
se_p = float(np.std(acc["purged"], ddof=1) / np.sqrt(reps))
check(f"no skill, labels over 21 periods: shuffled K-fold accuracy {m['shuffled']:.3f} > 0.85 (M2: 0.898)",
      m["shuffled"] > 0.85)
check(f"no skill: purged K-fold accuracy {m['purged']:.4f} within 4 MC SE ({4 * se_p:.4f}) of 0.5",
      abs(m["purged"] - 0.5) < 4 * se_p)
diff = np.array(acc["contiguous"]) - np.array(acc["purged"])
check(f"no skill: contiguous blocks without purging gain {diff.mean():.4f} over purged ones (M2: 0.023; band > 0.01)",
      diff.mean() > 0.01)

# ============================================================ 5. what show prints (T3: level and action)
_, out = quiet(bt.label_overlap, 21, 240)
check("label_overlap show: level, message and action printed",
      "WARNING: labels overlap" in out and "action: split with backtest.walk_forward_splits(horizon=21)" in out
      and "about 4.52 times too small" in out, out)
_, out = quiet(bt.purged_kfold_splits, 100, 5, horizon=5)
check("purged_kfold_splits show: table and the embargo notice with its action",
      "Purged K-fold: 100 rows, 5 contiguous blocks, label horizon 5, embargo 0" in out and "NOTICE: no embargo" in out
      and "action: set embargo to a few rows" in out and "embargoed" in out, out)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
