"""Datasets, taxonomies, hierarchical sequential agents and shift protocols.

Five real benchmarks are used. On each, an *agent* answers by walking a class
taxonomy from the root: at every internal node it calls a node-specific model
(a "tool") that chooses one child, and it stops at a leaf. Trajectory lengths
vary with the leaf, so this is an instance of the variable-length action-graph
setting of the paper. Step score: s_e(x) = 1 - p(child | node, x).

  20NG      20 Newsgroups (text), taxonomy = newsgroup name hierarchy,
            *natural* shift: senders with .edu addresses (source) vs all others (target)
  Letter    26 classes, taxonomy = Ward clustering of class centroids
  PenDigits 10 classes, Ward taxonomy
  MNIST     10 classes (20k subsample, PCA-50), Ward taxonomy
  SatImage  6 classes, Ward taxonomy
Tabular datasets use exponential-tilting covariate shift on the first principal
component, P_target(x) proportional to exp(beta * z(x)) P_source(x) (Tibshirani et al., 2019).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .graph import Taxonomy, compress_unary

PMLB_URL = "https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/datasets/{0}/{0}.tsv.gz"
NG_URL = "https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json"
TABULAR = {"Letter": "letter", "PenDigits": "pendigits", "MNIST": "mnist", "SatImage": "satimage"}
DATASETS = ["20NG", "Letter", "PenDigits", "MNIST", "SatImage"]


def _download(url: str, path: str) -> str:
    if not os.path.exists(path):
        import urllib.request
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


# ------------------------------- taxonomies --------------------------------
def ward_taxonomy(X: np.ndarray, y: np.ndarray, n_classes: int) -> Taxonomy:
    cents = np.stack([X[y == c].mean(0) for c in range(n_classes)])
    Z = linkage(cents, method="ward")
    children = {}
    for i, (a, b, *_rest) in enumerate(Z):
        children[n_classes + i] = [int(a), int(b)]
    root = n_classes + len(Z) - 1
    return Taxonomy(children=children, root=root,
                    leaf_class={c: c for c in range(n_classes)})


def name_taxonomy(names: List[str]) -> Taxonomy:
    """Trie over dot-separated names with unary chains compressed."""
    children: Dict[str, List[str]] = {}
    for nm in names:
        parts = nm.split(".")
        for k in range(len(parts)):
            parent = "ROOT" if k == 0 else ".".join(parts[:k])
            node = ".".join(parts[: k + 1])
            children.setdefault(parent, [])
            if node not in children[parent]:
                children[parent].append(node)
    children = compress_unary(children, "ROOT")
    return Taxonomy(children=children, root="ROOT",
                    leaf_class={nm: i for i, nm in enumerate(names)})


# ------------------------------- the agent ---------------------------------
class _ConstantModel:
    def __init__(self, j: int, k: int):
        self.classes_ = np.arange(k)
        self.j = j

    def predict_proba(self, X):
        P = np.full((X.shape[0], len(self.classes_)), 1e-3)
        P[:, self.j] = 1.0
        return P / P.sum(1, keepdims=True)


class HierarchicalAgent:
    """Top-down sequential agent: one probabilistic 'tool' per internal node."""

    def __init__(self, tax: Taxonomy, C: float = 1.0):
        self.tax, self.C = tax, C

    def fit(self, X, y):
        self.models = {}
        for node, eids in self.tax.child_edges.items():
            kids = [self.tax.edges[e][1] for e in eids]
            leaf2child = {}
            for j, k in enumerate(kids):
                for c in self.tax.leaves_under(k):
                    leaf2child[c] = j
            m = np.isin(y, list(leaf2child))
            t = np.array([leaf2child[c] for c in y[m]])
            if len(np.unique(t)) < 2:                      # degenerate node
                self.models[node] = _ConstantModel(t[0] if len(t) else 0, len(kids))
                continue
            clf = LogisticRegression(C=self.C, max_iter=3000)
            clf.fit(X[m], t)
            self.models[node] = clf
        return self

    def step_scores(self, X) -> np.ndarray:
        """S (n, E): s_e(x) = 1 - p(child | parent, x) for every edge."""
        n = X.shape[0]
        S = np.ones((n, self.tax.n_edges))
        for node, eids in self.tax.child_edges.items():
            clf = self.models[node]
            P = clf.predict_proba(X)
            for j, e in enumerate(eids):
                col = np.where(clf.classes_ == j)[0]
                S[:, e] = 1.0 - (P[:, col[0]] if len(col) else 0.0)
        return S

    def greedy_accuracy(self, S: np.ndarray, y: np.ndarray) -> float:
        from .conformal import path_scores_prod
        return float((path_scores_prod(S, self.tax).argmin(1) == y).mean())


# ------------------------------ prepared data ------------------------------
@dataclass
class Prepared:
    name: str
    tax: Taxonomy
    S: np.ndarray              # step scores for the held-out pool
    y: np.ndarray
    F: np.ndarray              # features for the domain classifier (pool)
    F_train: np.ndarray        # same features for training (source) inputs
    z: Optional[np.ndarray]    # tilt projection (tabular) else None
    source: Optional[np.ndarray] = None   # natural-shift masks (20NG)
    target: Optional[np.ndarray] = None
    agent_acc: float = float("nan")
    info: dict = field(default_factory=dict)
    class_prior: Optional[np.ndarray] = None   # training-label frequencies


def _strip_headers(doc: str) -> str:
    head, _, body = doc.partition("\n\n")
    return body


def prepare_20ng(data_dir: str, seed: int = 0) -> Prepared:
    path = _download(NG_URL, os.path.join(data_dir, "newsgroups.json"))
    d = json.load(open(path))
    keys = list(d["content"].keys())
    docs = [d["content"][k] for k in keys]
    names_doc = [d["target_names"][k] for k in keys]
    classes = sorted(set(names_doc))
    y = np.array([classes.index(n) for n in names_doc])
    tld = []
    for doc in docs:
        m = re.search(r"^From:.*?@([\w\.-]+)", doc, re.M)
        tld.append(m.group(1).lower().rstrip(".").split(".")[-1] if m else "none")
    edu = np.array([t == "edu" for t in tld])
    bodies = [_strip_headers(doc) for doc in docs]
    idx_edu = np.where(edu)[0]
    tr, hold_src = train_test_split(idx_edu, train_size=0.5, random_state=seed, stratify=y[idx_edu])
    hold_tgt = np.where(~edu)[0]
    vec = TfidfVectorizer(sublinear_tf=True, max_features=30000, min_df=2,
                          stop_words="english")
    Xtr = vec.fit_transform([bodies[i] for i in tr])
    pool = np.r_[hold_src, hold_tgt]
    Xpool = vec.transform([bodies[i] for i in pool])
    tax = name_taxonomy(classes)
    agent = HierarchicalAgent(tax, C=10.0).fit(Xtr, y[tr])
    S = agent.step_scores(Xpool)
    svd = TruncatedSVD(n_components=100, random_state=seed).fit(Xtr)
    F_train, F = svd.transform(Xtr), svd.transform(Xpool)
    src = np.r_[np.ones(len(hold_src), bool), np.zeros(len(hold_tgt), bool)]
    p = Prepared("20NG", tax, S, y[pool], F, F_train, None, src, ~src)
    p.class_prior = np.bincount(y[tr], minlength=len(classes)) / len(tr)
    p.agent_acc = agent.greedy_accuracy(S[~src], y[pool][~src])
    p.info = {"n_train": len(tr), "n_source_pool": len(hold_src), "n_target_pool": len(hold_tgt),
              "acc_source": agent.greedy_accuracy(S[src], y[pool][src]), "classes": classes}
    return p


def prepare_tabular(name: str, data_dir: str, seed: int = 0, train_frac: float = 0.4) -> Prepared:
    key = TABULAR[name]
    path = _download(PMLB_URL.format(key), os.path.join(data_dir, f"{key}.tsv.gz"))
    df = pd.read_csv(path, sep="\t")
    X = df.drop(columns=["target"]).to_numpy(float)
    y_raw = df["target"].to_numpy()
    classes = np.unique(y_raw)
    y = np.searchsorted(classes, y_raw)
    rng = np.random.default_rng(seed)
    if name == "MNIST":
        keep = rng.choice(len(y), 20000, replace=False)
        X, y = X[keep], y[keep]
    tr, pool = train_test_split(np.arange(len(y)), train_size=train_frac,
                                random_state=seed, stratify=y)
    sc = StandardScaler().fit(X[tr])
    Xs = sc.transform(X)
    if name == "MNIST":
        pca50 = PCA(50, random_state=seed).fit(Xs[tr])
        Xs = pca50.transform(Xs)
        Xs = Xs / Xs[tr].std(0)
    tax = ward_taxonomy(Xs[tr], y[tr], len(classes))
    agent = HierarchicalAgent(tax, C=1.0).fit(Xs[tr], y[tr])
    S = agent.step_scores(Xs[pool])
    pc1 = PCA(1, random_state=seed).fit(Xs[tr])
    z_all = pc1.transform(Xs)[:, 0]
    z_all = (z_all - z_all[tr].mean()) / z_all[tr].std()
    p = Prepared(name, tax, S, y[pool], Xs[pool], Xs[tr], z_all[pool])
    p.class_prior = np.bincount(y[tr], minlength=len(classes)) / len(tr)
    p.agent_acc = agent.greedy_accuracy(S, y[pool])
    p.info = {"n_train": len(tr), "n_pool": len(pool), "dim": X.shape[1]}
    return p


def prepare(name: str, data_dir: str, seed: int = 0) -> Prepared:
    return prepare_20ng(data_dir, seed) if name == "20NG" else prepare_tabular(name, data_dir, seed)


# ------------------------------ shift sampling ------------------------------
def gumbel_topk(logw: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Weighted sampling without replacement with probabilities proportional to exp(logw)."""
    g = logw + rng.gumbel(size=len(logw))
    return np.argpartition(-g, k)[:k]


def sample_rep(p: Prepared, beta: float, rng: np.random.Generator,
               n_cal: int = 1000, n_test: int = 1000, n_unl: int = 1000,
               n_dom_src: int = 2000) -> dict:
    """One repetition: (calibration from source, test & unlabeled from target)."""
    if p.source is not None:                       # natural shift (20NG)
        src, tgt = np.where(p.source)[0], np.where(p.target)[0]
        n_cal = min(n_cal, len(src))
        cal = rng.choice(src, n_cal, replace=False)
        tt = rng.choice(tgt, min(n_test + n_unl, len(tgt)), replace=False)
        test, unl = tt[:n_test], tt[n_test:]
    else:                                          # exponential tilting (tabular)
        perm = rng.permutation(len(p.y))
        h = len(perm) // 2
        src, tgt = perm[:h], perm[h:]
        n_cal = min(n_cal, len(src))
        m = min(n_test, int(0.4 * len(tgt)))
        cal = rng.choice(src, n_cal, replace=False)
        sel = tgt[gumbel_topk(beta * p.z[tgt], 2 * m, rng)]
        rng.shuffle(sel)
        test, unl = sel[:m], sel[m:]
    dom_src = rng.choice(len(p.F_train), min(n_dom_src, len(p.F_train)), replace=False)
    return {"cal": cal, "test": test, "unl": unl, "dom_src": dom_src}
