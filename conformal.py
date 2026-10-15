"""Conformal calibration primitives and trajectory nonconformity scores.

Notation (per input x):
  S[e]   step score of edge e, s_e(x) = 1 - p_agent(child | parent, x)
  tau    a root-to-terminal path (trajectory); |tau| its length
Trajectory scores implemented here:
  max      S_max(x, tau) = max_{e in tau} s_e(x)                         (PASC-style)
  prod     S_prod(x, tau) = 1 - prod_{e in tau} (1 - s_e(x))             (answer-level LAC)
  la-max   S_la(x, tau) = (max_{e in tau} F_{d(e)}(s_e(x)))^{|tau|}       (ours)
where F_d is the depth-d empirical CDF of *true-step* scores on a held-out tuning split.
If step ranks were i.i.d. U(0,1), max of |tau| of them has CDF v^{|tau|}, so S_la is
~U(0,1) for every length: trajectories of different length become comparable.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .graph import Taxonomy


# ---------------------------------------------------------------------------
# Quantiles
# ---------------------------------------------------------------------------
def split_conformal_quantile(cal_scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-CP threshold: the ceil((n+1)(1-alpha))-th smallest score."""
    n = len(cal_scores)
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return np.inf
    return float(np.sort(cal_scores)[k - 1])


def weighted_conformal_quantile(cal_scores: np.ndarray, cal_w: np.ndarray,
                                test_w: np.ndarray, alpha: float) -> np.ndarray:
    """Weighted split-CP threshold for each test point (Tibshirani et al., 2019).

    q(x) = Quantile_{1-alpha}( sum_i p_i(x) delta_{V_i} + p_{n+1}(x) delta_{+inf} ),
    p_i(x) = w(X_i) / (sum_j w(X_j) + w(x)).   Returns an array of thresholds.
    """
    order = np.argsort(cal_scores, kind="stable")
    v = cal_scores[order]
    cw = np.cumsum(cal_w[order])
    W = cw[-1]
    target = (1 - alpha) * (W + test_w)
    k = np.searchsorted(cw, target - 1e-12 * max(W, 1.0), side="left")
    q = np.full(len(test_w), np.inf)
    ok = k < len(v)
    q[ok] = v[k[ok]]
    return q


# ---------------------------------------------------------------------------
# Trajectory score matrices: (n, L) score of every candidate trajectory
# ---------------------------------------------------------------------------
def path_scores_max(S: np.ndarray, tax: Taxonomy) -> np.ndarray:
    return np.stack([S[:, p].max(axis=1) for p in tax.leaf_paths], axis=1)


def path_scores_prod(S: np.ndarray, tax: Taxonomy) -> np.ndarray:
    P = np.clip(1.0 - S, 1e-12, 1.0)
    return np.stack([1.0 - np.exp(np.log(P[:, p]).sum(axis=1)) for p in tax.leaf_paths], axis=1)


class DepthCDF:
    """Depth-wise empirical CDF of true-step scores, strictly increasing beyond the max."""

    def __init__(self, S_tune: np.ndarray, y_tune: np.ndarray, tax: Taxonomy):
        mask = tax.true_edge_mask(y_tune)
        self.sorted = {}
        for d in range(tax.max_depth):
            cols = tax.edge_depth == d
            vals = S_tune[:, cols][mask[:, cols]]
            self.sorted[d] = np.sort(vals) if len(vals) else np.array([0.0])
        self.edge_depth = tax.edge_depth

    def transform(self, S: np.ndarray) -> np.ndarray:
        U = np.empty_like(S, dtype=float)
        for d, srt in self.sorted.items():
            cols = np.where(self.edge_depth == d)[0]
            if len(cols) == 0:
                continue
            x = S[:, cols]
            m = len(srt)
            r = np.searchsorted(srt, x, side="right") / m
            beyond = x > srt[-1]
            r = np.where(beyond, 1.0 + (x - srt[-1]), r)
            U[:, cols] = r
        return U


def path_scores_length_adjusted(U: np.ndarray, tax: Taxonomy) -> np.ndarray:
    out = []
    for p in tax.leaf_paths:
        m = U[:, p].max(axis=1)
        out.append(np.power(m, len(p)))
    return np.stack(out, axis=1)


# ---------------------------------------------------------------------------
# Sets from per-depth thresholds (union-bound family)
# ---------------------------------------------------------------------------
def depthwise_thresholds(S_cal: np.ndarray, y_cal: np.ndarray, tax: Taxonomy,
                         alphas: np.ndarray) -> np.ndarray:
    """q_d = split-CP quantile at level alpha_d of true-step scores at depth d,
    computed among calibration trajectories that reach depth d."""
    mask = tax.true_edge_mask(y_cal)
    q = np.full(tax.max_depth, np.inf)
    for d in range(tax.max_depth):
        cols = tax.edge_depth == d
        vals = S_cal[:, cols][mask[:, cols]]
        if len(vals) and alphas[d] > 0:
            q[d] = split_conformal_quantile(vals, alphas[d])
    return q


def sets_from_depth_thresholds(S: np.ndarray, tax: Taxonomy, q: np.ndarray) -> np.ndarray:
    bad = (S > q[tax.edge_depth][None, :]).astype(np.int32)
    return (bad @ tax.incidence.astype(np.int32)) == 0


def depth_reach_probs(y: np.ndarray, tax: Taxonomy) -> np.ndarray:
    lens = tax.path_len[y]
    return np.array([(lens > d).mean() for d in range(tax.max_depth)])
