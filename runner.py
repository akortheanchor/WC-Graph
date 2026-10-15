"""Repetition loop shared by all experiment scripts."""
from __future__ import annotations

import os
import pickle
import time
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from .data import Prepared, prepare, sample_rep
from .methods import ALL_METHODS, CalContext, evaluate_sets, wc_graph

RAW_WEIGHTS = "Abl: raw weights"
from .weights import DomainClassifierRatio, effective_sample_size


def load_prepared(name: str, data_dir: str, cache_dir: str, seed: int = 0) -> Prepared:
    os.makedirs(cache_dir, exist_ok=True)
    f = os.path.join(cache_dir, f"{name}_seed{seed}.pkl")
    if os.path.exists(f):
        return pickle.load(open(f, "rb"))
    p = prepare(name, data_dir, seed)
    pickle.dump(p, open(f, "wb"))
    return p


def run_block(p: Prepared, beta: float, alpha: float, reps: int,
              methods: Iterable[str], seed: int = 0, setting: str = "",
              n_min: int = 50) -> pd.DataFrame:
    rows: List[Dict] = []
    for r in range(reps):
        rng = np.random.default_rng([seed, r, int(beta * 100), int(alpha * 1000)])
        idx = sample_rep(p, beta, rng)
        wfun = DomainClassifierRatio().fit(p.F_train[idx["dom_src"]], p.F[idx["unl"]])
        w_cal, w_test = wfun(p.F[idx["cal"]]), wfun(p.F[idx["test"]])
        raw_ctx = None
        ctx = CalContext(tax=p.tax, alpha=alpha, S_cal=p.S[idx["cal"]], y_cal=p.y[idx["cal"]],
                         w_cal=w_cal, S_test=p.S[idx["test"]], w_test=w_test,
                         seed=int(rng.integers(1 << 31)), class_prior=p.class_prior,
                         n_min=n_min)
        y_test = p.y[idx["test"]]
        for m in methods:
            t0 = time.perf_counter()
            if m == RAW_WEIGHTS:                    # ablation: unstabilised weights
                if raw_ctx is None:
                    rf = DomainClassifierRatio(Cs=(1.0,), clip_q=None).fit(
                        p.F_train[idx["dom_src"]], p.F[idx["unl"]])
                    raw_ctx = CalContext(tax=p.tax, alpha=alpha, S_cal=ctx.S_cal, y_cal=ctx.y_cal,
                                         w_cal=rf(p.F[idx["cal"]]), S_test=ctx.S_test,
                                         w_test=rf(p.F[idx["test"]]), seed=ctx.seed,
                                         class_prior=p.class_prior, n_min=n_min)
                sets = wc_graph(raw_ctx)
            else:
                sets = ALL_METHODS[m](ctx)
            dt = time.perf_counter() - t0
            row = {"dataset": p.name, "setting": setting, "beta": beta, "alpha": alpha,
                   "rep": r, "method": m, "n_cal": len(idx["cal"]), "n_test": len(y_test),
                   "ess": effective_sample_size(w_cal), "time_s": dt}
            row.update(evaluate_sets(sets, y_test, p.tax.path_len))
            row["n_covered"] = int(round(row["coverage"] * len(y_test)))
            rows.append(row)
    return pd.DataFrame(rows)


def run_synthetic(env, delta: float, alpha: float, reps: int, methods: Iterable[str],
                  seed: int = 0, n_cal: int = 1000, n_test: int = 1000, n_unl: int = 1000,
                  n_min: int = 50) -> pd.DataFrame:
    """Synthetic tool-use environment: adds 'WC-Graph (oracle w)' and the exact
    coverage-gap term Delta for the estimated weights."""
    from .methods import wc_graph
    from .weights import weight_error_bound
    agent, tax = env.agent, env.tax
    big_rng = np.random.default_rng([seed, 999, int(delta * 100)])
    X_big = big_rng.normal(size=(20000, env.d))           # source sample for Delta
    w_true_big = env.true_ratio(X_big, delta)
    from .methods import length_strata
    strata = length_strata(tax, env.class_prior, n_cal, n_min)
    st_big = np.array([strata[int(l)] for l in tax.path_len[env.route(X_big, big_rng)]])
    rows = []
    for r in range(reps):
        rng = np.random.default_rng([seed, r, int(delta * 100), int(alpha * 1000)])
        Xc, yc = env.sample(n_cal, 0.0, rng)
        Xt, yt = env.sample(n_test, delta, rng)
        Xu, _ = env.sample(n_unl, delta, rng)
        dom = env.X_train[rng.choice(len(env.X_train), 2000, replace=False)]
        wfun = DomainClassifierRatio().fit(dom, Xu)
        Sc, St = agent.step_scores(Xc), agent.step_scores(Xt)
        ctx = CalContext(tax=tax, alpha=alpha, S_cal=Sc, y_cal=yc, w_cal=wfun(Xc),
                         S_test=St, w_test=wfun(Xt), seed=int(rng.integers(1 << 31)),
                         class_prior=env.class_prior, n_min=n_min)
        octx = CalContext(tax=tax, alpha=alpha, S_cal=Sc, y_cal=yc,
                          w_cal=env.true_ratio(Xc, delta), S_test=St,
                          w_test=env.true_ratio(Xt, delta), seed=ctx.seed,
                          class_prior=env.class_prior, n_min=n_min)
        wb = wfun(X_big)
        delta_hat = weight_error_bound(wb, w_true_big)
        delta_k = max(weight_error_bound(wb[st_big == k], w_true_big[st_big == k])
                      for k in np.unique(st_big))
        todo = [(m, ALL_METHODS[m], ctx) for m in methods] + [("WC-Graph (oracle w)", wc_graph, octx)]
        for name, fn, c in todo:
            sets = fn(c)
            row = {"dataset": "ToolGraph-Syn", "delta_shift": delta, "alpha": alpha, "rep": r,
                   "method": name, "Delta": delta_hat, "Delta_strata_max": delta_k,
                   "ess": effective_sample_size(c.w_cal)}
            row.update(evaluate_sets(sets, yt, tax.path_len))
            rows.append(row)
    return pd.DataFrame(rows)
