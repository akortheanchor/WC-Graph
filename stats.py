"""Statistical validation utilities.

* Validity: one-sided one-sample t-test of per-repetition coverage against 1-alpha
  (H0: E[coverage] >= 1-alpha), with Holm correction across methods, plus the
  pooled exact binomial test on all test points.
* Efficiency: paired Wilcoxon signed-rank test of per-repetition mean set size
  (proposed vs each baseline), Holm-corrected, with Cliff's delta effect size.
* Overall ranking: Friedman test over blocks (dataset x setting) and the Nemenyi
  critical difference (Demsar, 2006).
"""
from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy import stats


def mean_ci(x: np.ndarray, level: float = 0.95):
    x = np.asarray(x, float)
    m = x.mean()
    if len(x) < 2:
        return m, np.nan, np.nan
    h = stats.t.ppf(0.5 + level / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def holm(pvals: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvals), float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def coverage_test(cov_reps: np.ndarray, target: float) -> float:
    """p-value of H0: E[cov] >= target vs H1: E[cov] < target (one-sided t)."""
    cov_reps = np.asarray(cov_reps, float)
    if cov_reps.std(ddof=1) == 0:
        return 0.0 if cov_reps.mean() < target else 1.0
    return float(stats.ttest_1samp(cov_reps, target, alternative="less").pvalue)


def binom_pooled(k: int, n: int, target: float) -> float:
    return float(stats.binomtest(k, n, target, alternative="less").pvalue)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a), np.asarray(b)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (len(a) * len(b)))


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a) - np.asarray(b)
    if np.allclose(d, 0):
        return 1.0
    return float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)


def friedman_nemenyi(rank_table: pd.DataFrame, alpha: float = 0.05) -> Dict:
    """rank_table: rows = blocks, columns = methods, values = scores (lower=better)."""
    X = rank_table.to_numpy(float)
    N, k = X.shape
    ranks = np.apply_along_axis(stats.rankdata, 1, X)
    avg = pd.Series(ranks.mean(0), index=rank_table.columns).sort_values()
    chi2, p = stats.friedmanchisquare(*[X[:, j] for j in range(k)])
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6.0 * N))
    return {"avg_rank": avg, "chi2": float(chi2), "p": float(p), "cd": float(cd), "N": N, "k": k}


def validity_aware_score(cov: float, size: float, target: float, tol: float, n_leaves: int) -> float:
    """Ranking score (lower = better): valid methods ranked by normalised set size;
    methods whose mean coverage falls below target - tol are ranked after every
    valid method, ordered by their coverage shortfall."""
    if cov >= target - tol:
        return size / n_leaves
    return 1.0 + (target - cov)
