"""Action graphs and the combinatorial algorithms behind conformal trajectory sets.

An *action graph* G = (V, E) is a rooted DAG. A trajectory is a root-to-terminal
path; its length |tau| is the number of edges and may differ between trajectories.
Every edge e carries a step nonconformity score s_e(x) for a given input x.

This module provides
  * ``Taxonomy``   - the rooted-tree special case used by the hierarchical agents,
                     with vectorised path/edge incidence for fast calibration;
  * ``DAG``        - a general layered DAG used by the complexity experiments and
                     the synthetic tool-use environment;
  * exact algorithms for the trajectory-set problems analysed in the paper:
      - count_paths_max          O(|V|+|E|)      (Theorem 2a)
      - count_paths_per_length   O(D(|V|+|E|))   (Theorem 2a, length-Mondrian sets)
      - reachable_answers_sum_per_length O(D(|V|+|E|)) (Theorem 2b, length-Mondrian)
      - reachable_answers_max    O(|V|+|E|)
      - reachable_answers_sum    O(|V|+|E|)      DAG shortest path (Theorem 2b)
      - count_paths_sum_pseudo   O(|E| B)        pseudo-polynomial (Theorem 2c)
      - brute-force enumerators for verification.
Counting trajectories under an *additive* score is #P-complete (Theorem 2c);
see ``knapsack_reduction`` for the parsimonious reduction from #Knapsack.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Rooted tree (taxonomy) used by the hierarchical sequential agents
# ---------------------------------------------------------------------------
@dataclass
class Taxonomy:
    """Rooted tree whose leaves are the terminal answers (class labels).

    children : mapping internal node -> ordered list of child nodes
    root     : root node id
    leaf_class : mapping leaf node -> class index in [0, L)
    """

    children: Dict[Hashable, List[Hashable]]
    root: Hashable
    leaf_class: Dict[Hashable, int]
    edges: List[Tuple[Hashable, Hashable]] = field(init=False)
    edge_depth: np.ndarray = field(init=False)
    leaf_paths: List[np.ndarray] = field(init=False)
    path_len: np.ndarray = field(init=False)
    incidence: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.edges, depth = [], []
        self.child_edges: Dict[Hashable, List[int]] = {}
        stack = [(self.root, 0)]
        while stack:
            node, d = stack.pop()
            kids = self.children.get(node, [])
            if kids:
                self.child_edges[node] = []
                for c in kids:
                    self.child_edges[node].append(len(self.edges))
                    self.edges.append((node, c))
                    depth.append(d)
                    stack.append((c, d + 1))
        self.edge_depth = np.asarray(depth, dtype=int)
        parent_edge = {c: i for i, (_, c) in enumerate(self.edges)}
        L = len(self.leaf_class)
        self.leaf_paths = [None] * L
        for leaf, cls in self.leaf_class.items():
            path, node = [], leaf
            while node in parent_edge:
                e = parent_edge[node]
                path.append(e)
                node = self.edges[e][0]
            self.leaf_paths[cls] = np.asarray(path[::-1], dtype=int)
        self.path_len = np.array([len(p) for p in self.leaf_paths], dtype=int)
        self.incidence = np.zeros((len(self.edges), L), dtype=bool)
        for c, p in enumerate(self.leaf_paths):
            self.incidence[p, c] = True
        self.max_depth = int(self.path_len.max())

    @property
    def n_leaves(self) -> int:
        return len(self.leaf_paths)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def internal_nodes(self) -> List[Hashable]:
        return list(self.child_edges.keys())

    def leaves_under(self, node) -> List[int]:
        out, stack = [], [node]
        while stack:
            v = stack.pop()
            if v in self.leaf_class:
                out.append(self.leaf_class[v])
            stack.extend(self.children.get(v, []))
        return out

    def true_edge_mask(self, y: np.ndarray) -> np.ndarray:
        """Boolean (n, E): edge e lies on the true trajectory of example i."""
        return self.incidence[:, y].T

    def describe(self) -> dict:
        return {
            "leaves": self.n_leaves,
            "edges": self.n_edges,
            "internal_nodes": len(self.child_edges),
            "min_len": int(self.path_len.min()),
            "max_len": int(self.path_len.max()),
            "mean_len": float(self.path_len.mean()),
        }


def compress_unary(children: Dict, root) -> Dict:
    """Collapse chains of single-child internal nodes (keeps leaves)."""
    out: Dict = {}

    def walk(v):
        kids = children.get(v, [])
        new = []
        for c in kids:
            while len(children.get(c, [])) == 1:
                c = children[c][0]
            new.append(c)
            walk(c)
        if new:
            out[v] = new

    walk(root)
    return out


# ---------------------------------------------------------------------------
# General DAG and the exact algorithms of Theorem 2
# ---------------------------------------------------------------------------
@dataclass
class DAG:
    """Rooted DAG. ``order`` is a topological order starting at ``root``.

    edges[k] = (u, v); terminals = nodes with no outgoing edge (or flagged);
    answer[t] = answer label of terminal t (many terminals may share a label).
    """

    n_nodes: int
    edges: List[Tuple[int, int]]
    root: int = 0
    answer: Dict[int, int] | None = None

    def __post_init__(self) -> None:
        self.out = defaultdict(list)
        self.inc = defaultdict(list)
        for k, (u, v) in enumerate(self.edges):
            self.out[u].append(k)
            self.inc[v].append(k)
        indeg = np.zeros(self.n_nodes, dtype=int)
        for _, v in self.edges:
            indeg[v] += 1
        order, stack = [], [i for i in range(self.n_nodes) if indeg[i] == 0]
        while stack:
            u = stack.pop()
            order.append(u)
            for k in self.out[u]:
                v = self.edges[k][1]
                indeg[v] -= 1
                if indeg[v] == 0:
                    stack.append(v)
        if len(order) != self.n_nodes:
            raise ValueError("graph has a cycle")
        self.order = order
        self.terminals = [v for v in range(self.n_nodes) if not self.out[v]]
        if self.answer is None:
            self.answer = {t: i for i, t in enumerate(self.terminals)}


def count_paths_max(g: DAG, s: np.ndarray, q: float) -> int:
    """#root->terminal paths whose every edge score is <= q. O(|V|+|E|)."""
    ways = np.zeros(g.n_nodes, dtype=object)
    ways[g.root] = 1
    for u in g.order:
        if ways[u] == 0:
            continue
        for k in g.out[u]:
            if s[k] <= q:
                ways[g.edges[k][1]] += ways[u]
    return int(sum(ways[t] for t in g.terminals))


def count_paths_per_length(g: DAG, s: np.ndarray, thr: Dict[int, float]) -> int:
    """#paths tau whose edges all satisfy s_e <= thr[|tau|] (length-Mondrian sets).

    The admissible threshold depends on the *final* length, so for every length l a
    DP over (node, steps-taken) counts paths of exact length l whose edges are all
    <= thr[l]. Total cost O(D (|V| + |E|)) with D = number of distinct lengths.
    """
    total = 0
    for ell, t in thr.items():
        ways = [defaultdict(int) for _ in range(g.n_nodes)]
        ways[g.root][0] = 1
        for v in g.order:
            for d, w in list(ways[v].items()):
                if d >= ell:
                    continue
                for k in g.out[v]:
                    if s[k] <= t:
                        ways[g.edges[k][1]][d + 1] += w
        total += sum(ways[x].get(ell, 0) for x in g.terminals)
    return total


def reachable_answers_sum_per_length(g: DAG, s: np.ndarray, thr: Dict[int, float]) -> set:
    """Answer set for length-Mondrian *additive* scores: answer a is in the set iff
    some path of length l ending at a terminal with answer a has sum_e s_e <= thr[l].
    DP of the minimum additive score over (node, length): O(D (|V| + |E|))."""
    D = max(thr) if thr else 0
    best = np.full((g.n_nodes, D + 1), np.inf)
    best[g.root, 0] = 0.0
    for v in g.order:
        for d in range(D):
            if np.isfinite(best[v, d]):
                for k in g.out[v]:
                    u = g.edges[k][1]
                    best[u, d + 1] = min(best[u, d + 1], best[v, d] + s[k])
    return {g.answer[t] for t in g.terminals
            for l, q in thr.items() if l <= D and best[t, l] <= q + 1e-12}


def reachable_answers_max(g: DAG, s: np.ndarray, q: float) -> set:
    """Answer set {answer(t): exists path to t with all edges <= q}. O(|V|+|E|)."""
    reach = np.zeros(g.n_nodes, dtype=bool)
    reach[g.root] = True
    for u in g.order:
        if reach[u]:
            for k in g.out[u]:
                if s[k] <= q:
                    reach[g.edges[k][1]] = True
    return {g.answer[t] for t in g.terminals if reach[t]}


def shortest_path_dag(g: DAG, s: np.ndarray) -> np.ndarray:
    dist = np.full(g.n_nodes, np.inf)
    dist[g.root] = 0.0
    for u in g.order:
        if np.isfinite(dist[u]):
            for k in g.out[u]:
                v = g.edges[k][1]
                dist[v] = min(dist[v], dist[u] + s[k])
    return dist


def reachable_answers_sum(g: DAG, s: np.ndarray, q: float) -> set:
    """Answer set under the additive score sum_e s_e <= q via DAG shortest path."""
    dist = shortest_path_dag(g, s)
    return {g.answer[t] for t in g.terminals if dist[t] <= q + 1e-12}


def count_paths_sum_pseudo(g: DAG, w: np.ndarray, B: int) -> int:
    """#paths with sum of non-negative integer edge weights <= B. O(|E| B)."""
    w = np.asarray(w, dtype=int)
    ways = np.zeros((g.n_nodes, B + 1), dtype=object)
    ways[g.root, 0] = 1
    for u in g.order:
        row = ways[u]
        if not any(row):
            continue
        for k in g.out[u]:
            v, wk = g.edges[k][1], w[k]
            if wk > B:
                continue
            ways[v, wk:] = ways[v, wk:] + row[: B + 1 - wk]
    return int(sum(ways[t].sum() for t in g.terminals))


def enumerate_paths(g: DAG) -> List[List[int]]:
    """All root->terminal paths as lists of edge ids (exponential; tests only)."""
    out = []

    def rec(v, acc):
        if not g.out[v]:
            out.append(list(acc))
            return
        for k in g.out[v]:
            acc.append(k)
            rec(g.edges[k][1], acc)
            acc.pop()

    rec(g.root, [])
    return out


def knapsack_reduction(weights: Sequence[int]) -> DAG:
    """Parsimonious reduction #Knapsack <= #AdditiveTrajectorySet.

    Item i becomes a gadget between chain nodes c_{i-1} and c_i with two routes:
    'skip' (edge of weight 0) and 'take' (two edges via a midpoint, weights w_i and 0).
    Root-to-terminal paths are in bijection with subsets, and the path weight equals
    the subset weight, so #{paths with sum <= B} = #{subsets with weight <= B}.
    Returns the DAG; edge weights are stored on attribute ``w``.
    """
    edges, w = [], []
    n = len(weights)
    # chain nodes 0..n, midpoints n+1..2n
    for i, wi in enumerate(weights):
        a, b, m = i, i + 1, n + 1 + i
        edges.append((a, b)); w.append(0)          # skip
        edges.append((a, m)); w.append(int(wi))    # take (part 1)
        edges.append((m, b)); w.append(0)          # take (part 2)
    g = DAG(n_nodes=2 * n + 1, edges=edges, root=0)
    g.w = np.asarray(w, dtype=int)
    return g


def random_layered_dag(n_layers: int, width: int, p: float, rng: np.random.Generator,
                       n_answers: int | None = None) -> DAG:
    """Random layered DAG: root -> layer_1 ... layer_n -> (terminals = last layer).

    Every node keeps at least one outgoing edge so all nodes reach a terminal.
    """
    nodes = [[0]]
    nid = 1
    for _ in range(n_layers):
        nodes.append(list(range(nid, nid + width)))
        nid += width
    edges = []
    for li in range(len(nodes) - 1):
        for u in nodes[li]:
            tgt = [v for v in nodes[li + 1] if rng.random() < p]
            if not tgt:
                tgt = [int(rng.choice(nodes[li + 1]))]
            edges += [(u, v) for v in tgt]
    g = DAG(n_nodes=nid, edges=edges, root=0)
    if n_answers is not None:
        g.answer = {t: int(rng.integers(n_answers)) for t in g.terminals}
    return g
