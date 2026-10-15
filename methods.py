"""The proposed method (WC-Graph), seven baselines and two ablations.

Every method maps a calibration context to a boolean matrix ``sets`` of shape
(n_test, L): sets[i, l] is True iff trajectory l is in the prediction set C(x_i).
All methods receive the same labelled source budget n_cal. Methods that need a
tuning split (TRAQ-style allocation, WC-Graph) carve it out of that same budget.

Baselines
  B1 Indep-CP        per-step split CP at level alpha (no joint guarantee)
  B2 Bonferroni-CP   per-step split CP at alpha/D (union bound)
  B3 PASC            joint max-score split CP (Kotte, 2026)
  B4 Answer-SCP      final-answer split CP with LAC score 1-prod p (structure-agnostic)
  B5 TRAQ-Alloc      union bound with tuned per-step budgets (Li et al., 2024 style)
  B6 MiCP-Alloc      union bound with turn-decaying budgets (Zhou et al., 2026 style)
  B7 WCP-Answer      weighted CP under covariate shift on the answer score
                     (Tibshirani et al., 2019)
Proposed
  WC-Graph           density-ratio weighted CP, stratified by trajectory length
                     (Mondrian), on the trajectory-likelihood score
Ablations
  -weights           length-Mondrian CP without density-ratio weights
  -Mondrian          weighted CP without stratification -- this is exactly B7
                     (WCP-Answer), so it is not run twice
  max-score          WC-Graph with the PASC max-step score
  LA-score           single weighted quantile on a length-adjusted, depth-normalised
                     max score (the first design we tried; reported for completeness)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np

from .conformal import (DepthCDF, depthwise_thresholds, path_scores_length_adjusted,
                        path_scores_max, path_scores_prod, sets_from_depth_thresholds,
                        split_conformal_quantile, weighted_conformal_quantile)
from .graph import Taxonomy


@dataclass
class CalContext:
    tax: Taxonomy
    alpha: float
    S_cal: np.ndarray          # full calibration step scores (n_cal, E)
    y_cal: np.ndarray
    w_cal: np.ndarray          # density-ratio weights of calibration inputs
    S_test: np.ndarray
    w_test: np.ndarray
    tune_frac: float = 0.3
    seed: int = 0
    class_prior: Optional[np.ndarray] = None   # label frequencies from *training* data
    n_min: int = 50                            # min expected calibration size per stratum

    def __post_init__(self):
        n = len(self.y_cal)
        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(n)
        k = int(round(self.tune_frac * n))
        self.tune_idx, self.calb_idx = perm[:k], perm[k:]

    @property
    def present_depths(self) -> np.ndarray:
        lens = self.tax.path_len[self.y_cal]
        return np.array([(lens > d).any() for d in range(self.tax.max_depth)])


def _sets_from_scores(A_test: np.ndarray, q) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q.ndim == 0:
        return A_test <= q
    return A_test <= q[:, None]


# ------------------------------- baselines ---------------------------------
def indep_cp(c: CalContext) -> np.ndarray:
    a = np.full(c.tax.max_depth, c.alpha)
    q = depthwise_thresholds(c.S_cal, c.y_cal, c.tax, a)
    return sets_from_depth_thresholds(c.S_test, c.tax, q)


def bonferroni_cp(c: CalContext) -> np.ndarray:
    D = int(c.present_depths.sum())
    a = np.where(c.present_depths, c.alpha / D, 0.0)
    q = depthwise_thresholds(c.S_cal, c.y_cal, c.tax, a)
    return sets_from_depth_thresholds(c.S_test, c.tax, q)


def pasc(c: CalContext) -> np.ndarray:
    A_cal = path_scores_max(c.S_cal, c.tax)[np.arange(len(c.y_cal)), c.y_cal]
    q = split_conformal_quantile(A_cal, c.alpha)
    return _sets_from_scores(path_scores_max(c.S_test, c.tax), q)


def answer_scp(c: CalContext) -> np.ndarray:
    A_cal = path_scores_prod(c.S_cal, c.tax)[np.arange(len(c.y_cal)), c.y_cal]
    q = split_conformal_quantile(A_cal, c.alpha)
    return _sets_from_scores(path_scores_prod(c.S_test, c.tax), q)


def traq_alloc(c: CalContext, n_candidates: int = 200) -> np.ndarray:
    """Search per-step budgets on the tuning split, calibrate on the rest."""
    pres = c.present_depths
    D = int(pres.sum())
    rng = np.random.default_rng(c.seed + 17)
    cands = [np.full(D, 1.0 / D)] + list(rng.dirichlet(np.ones(D), size=n_candidates - 1))
    St, yt = c.S_cal[c.tune_idx], c.y_cal[c.tune_idx]
    best, best_size = None, np.inf
    for frac in cands:
        a = np.zeros(c.tax.max_depth)
        a[pres] = c.alpha * frac
        q = depthwise_thresholds(St, yt, c.tax, a)
        size = sets_from_depth_thresholds(St, c.tax, q).sum(1).mean()
        if size < best_size:
            best, best_size = a, size
    q = depthwise_thresholds(c.S_cal[c.calb_idx], c.y_cal[c.calb_idx], c.tax, best)
    return sets_from_depth_thresholds(c.S_test, c.tax, q)


def micp_alloc(c: CalContext, decay: float = 0.5) -> np.ndarray:
    pres = c.present_depths
    raw = np.where(pres, decay ** np.arange(c.tax.max_depth), 0.0)
    a = c.alpha * raw / raw.sum()
    q = depthwise_thresholds(c.S_cal, c.y_cal, c.tax, a)
    return sets_from_depth_thresholds(c.S_test, c.tax, q)


def wcp_answer(c: CalContext) -> np.ndarray:
    A_cal = path_scores_prod(c.S_cal, c.tax)[np.arange(len(c.y_cal)), c.y_cal]
    q = weighted_conformal_quantile(A_cal, c.w_cal, c.w_test, c.alpha)
    return _sets_from_scores(path_scores_prod(c.S_test, c.tax), q)


# ------------------------------ proposed -----------------------------------
def length_strata(tax: Taxonomy, class_prior: Optional[np.ndarray], n_cal: int,
                  n_min: int) -> Dict[int, int]:
    """Partition trajectory lengths into contiguous strata so that each stratum has
    an *expected* calibration size >= n_min. Expected sizes come from training-label
    frequencies, so the partition is fixed before the calibration data are seen and
    the per-stratum guarantee is preserved."""
    prior = (np.full(tax.n_leaves, 1.0 / tax.n_leaves) if class_prior is None
             else np.asarray(class_prior, float) / np.sum(class_prior))
    lengths = np.unique(tax.path_len)
    exp_n = {l: n_cal * prior[tax.path_len == l].sum() for l in lengths}
    strata, sid, acc = {}, 0, 0.0
    for l in lengths:
        strata[int(l)] = sid
        acc += exp_n[l]
        if acc >= n_min:
            sid, acc = sid + 1, 0.0
    if acc < n_min and sid > 0:            # merge an under-filled tail into the previous stratum
        for l in lengths:
            if strata[int(l)] == sid:
                strata[int(l)] = sid - 1
    return strata


def _mondrian_by_length(c: CalContext, A_cal_full: np.ndarray, A_test: np.ndarray,
                        weighted: bool) -> np.ndarray:
    """Length-stratified (Mondrian) calibration.

    A candidate trajectory of length l is compared only with calibration examples
    whose *true* trajectory has length l, using a (weighted) quantile computed
    inside that stratum. This yields P(tau* in C(X) | |tau*| = l) >= 1 - alpha for
    every length l (exactly under known covariate-shift weights). Rare lengths are
    merged with their neighbours (see ``length_strata``); the guarantee then holds
    per merged stratum.
    """
    strata = length_strata(c.tax, c.class_prior, len(c.y_cal), c.n_min)
    leaf_stratum = np.array([strata[int(l)] for l in c.tax.path_len])
    cal_stratum = leaf_stratum[c.y_cal]
    sets = np.zeros_like(A_test, dtype=bool)
    for st in np.unique(leaf_stratum):
        cand = np.where(leaf_stratum == st)[0]
        m = cal_stratum == st
        if m.sum() == 0:
            sets[:, cand] = True          # no evidence: keep every candidate
            continue
        if weighted:
            q = weighted_conformal_quantile(A_cal_full[m], c.w_cal[m], c.w_test, c.alpha)
            sets[:, cand] = A_test[:, cand] <= q[:, None]
        else:
            q = split_conformal_quantile(A_cal_full[m], c.alpha)
            sets[:, cand] = A_test[:, cand] <= q
    return sets


def wc_graph(c: CalContext) -> np.ndarray:
    """WC-Graph: weighted + length-Mondrian calibration of the trajectory
    likelihood score S(x, tau) = 1 - prod_{e in tau} p_e(x) (equivalently the
    additive score sum_e -log p_e(x))."""
    A_cal = path_scores_prod(c.S_cal, c.tax)[np.arange(len(c.y_cal)), c.y_cal]
    return _mondrian_by_length(c, A_cal, path_scores_prod(c.S_test, c.tax), weighted=True)


def wc_graph_no_weights(c: CalContext) -> np.ndarray:
    A_cal = path_scores_prod(c.S_cal, c.tax)[np.arange(len(c.y_cal)), c.y_cal]
    return _mondrian_by_length(c, A_cal, path_scores_prod(c.S_test, c.tax), weighted=False)


def wc_graph_max_score(c: CalContext) -> np.ndarray:
    A_cal = path_scores_max(c.S_cal, c.tax)[np.arange(len(c.y_cal)), c.y_cal]
    return _mondrian_by_length(c, A_cal, path_scores_max(c.S_test, c.tax), weighted=True)


def wc_graph_length_adjusted(c: CalContext) -> np.ndarray:
    """Alternative explored first: a single weighted quantile on the
    length-adjusted depth-normalised max score (no stratification)."""
    ti, ci = c.tune_idx, c.calb_idx
    F = DepthCDF(c.S_cal[ti], c.y_cal[ti], c.tax)
    A_cal = path_scores_length_adjusted(F.transform(c.S_cal[ci]), c.tax)
    A_cal = A_cal[np.arange(len(ci)), c.y_cal[ci]]
    A_test = path_scores_length_adjusted(F.transform(c.S_test), c.tax)
    q = weighted_conformal_quantile(A_cal, c.w_cal[ci], c.w_test, c.alpha)
    return _sets_from_scores(A_test, q)


BASELINES: Dict[str, Callable] = {
    "Indep-CP": indep_cp,
    "Bonferroni-CP": bonferroni_cp,
    "PASC": pasc,
    "Answer-SCP": answer_scp,
    "TRAQ-Alloc": traq_alloc,
    "MiCP-Alloc": micp_alloc,
    "WCP-Answer": wcp_answer,
}
PROPOSED: Dict[str, Callable] = {"WC-Graph (ours)": wc_graph}
ABLATIONS: Dict[str, Callable] = {
    "Abl: -weights": wc_graph_no_weights,          # Mondrian, unweighted
    "Abl: max-score": wc_graph_max_score,          # Mondrian + weights, max score
    "Abl: LA-score": wc_graph_length_adjusted,     # first design, no stratification
}
ALL_METHODS: Dict[str, Callable] = {**BASELINES, **PROPOSED, **ABLATIONS}
MAIN_METHODS = list(BASELINES) + list(PROPOSED)


def evaluate_sets(sets: np.ndarray, y: np.ndarray, path_len: np.ndarray) -> dict:
    cov = sets[np.arange(len(y)), y]
    size = sets.sum(1)
    out = {
        "coverage": float(cov.mean()),
        "size": float(size.mean()),
        "size_median": float(np.median(size)),
        "empty_rate": float((size == 0).mean()),
        "singleton_rate": float((size == 1).mean()),
    }
    lens = path_len[y]
    for L in np.unique(lens):
        m = lens == L
        out[f"cov_len{L}"] = float(cov[m].mean())
        out[f"n_len{L}"] = int(m.sum())
    return out
