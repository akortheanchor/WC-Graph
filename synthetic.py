"""Synthetic tool-use environment with a *known* density ratio.

A random tool graph (rooted tree, branching 2-4, depth 1-5) is generated. The
correct tool at each node is chosen by a hidden noisy linear router acting on
the task embedding x in R^d, so P(trajectory | x) is identical in source and
target (pure covariate shift). The agent learns one logistic 'tool selector'
per node from source data. Source inputs X ~ N(0, I_d); target inputs
X ~ N(mu, I_d) with mu = delta * u for a fixed unit vector u, hence the exact
density ratio is w(x) = exp(mu.x - |mu|^2 / 2). This allows (i) oracle-weight
WC-Graph and (ii) exact evaluation of the coverage-gap term Delta of Theorem 1.
"""
from __future__ import annotations

import numpy as np

from .data import HierarchicalAgent, Prepared
from .graph import Taxonomy


def random_tool_tree(rng: np.random.Generator, max_depth: int = 5,
                     p_stop=(0.0, 0.15, 0.3, 0.45, 0.7)) -> Taxonomy:
    children, leaves = {}, []
    nid = [1]

    def grow(v, d):
        if d >= max_depth or (d > 0 and rng.random() < p_stop[d]):
            leaves.append(v)
            return
        k = int(rng.integers(2, 5))
        kids = list(range(nid[0], nid[0] + k))
        nid[0] += k
        children[v] = kids
        for c in kids:
            grow(c, d + 1)

    grow(0, 0)
    return Taxonomy(children=children, root=0, leaf_class={v: i for i, v in enumerate(leaves)})


class ToolEnv:
    def __init__(self, seed: int = 0, d: int = 8, router_noise: float = 0.6):
        self.rng = np.random.default_rng(seed)
        while True:
            self.tax = random_tool_tree(self.rng)
            if 20 <= self.tax.n_leaves <= 45:
                break
        self.d, self.noise = d, router_noise
        self.W = {v: self.rng.normal(size=(len(k), d)) * 1.5
                  for v, k in self.tax.children.items()}
        self.u = np.eye(d)[0]

    def route(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Ground-truth trajectory (returned as the leaf/class index)."""
        node = np.zeros(len(X), dtype=int)
        done = np.zeros(len(X), dtype=bool)
        for _ in range(self.tax.max_depth + 1):
            for v, kids in self.tax.children.items():
                m = (node == v) & ~done
                if m.any():
                    logits = X[m] @ self.W[v].T + self.noise * rng.gumbel(size=(m.sum(), len(kids)))
                    node[m] = np.asarray(kids)[logits.argmax(1)]
            done = ~np.isin(node, list(self.tax.children))
        return np.array([self.tax.leaf_class[v] for v in node])

    def sample(self, n: int, delta: float, rng: np.random.Generator):
        X = rng.normal(size=(n, self.d)) + delta * self.u
        return X, self.route(X, rng)

    def true_ratio(self, X: np.ndarray, delta: float) -> np.ndarray:
        mu = delta * self.u
        return np.exp(X @ mu - 0.5 * mu @ mu)

    def fit_agent(self, n_train: int = 6000, seed: int = 1) -> HierarchicalAgent:
        rng = np.random.default_rng(seed)
        X, y = self.sample(n_train, 0.0, rng)
        self.X_train = X
        self.class_prior = np.bincount(y, minlength=self.tax.n_leaves) / len(y)
        self.agent = HierarchicalAgent(self.tax, C=1.0).fit(X, y)
        return self.agent
