"""Adapter for real LLM-agent traces (e.g. ReAct / tool-calling logs).

Run WC-Graph on your own agent logs by exporting one JSON object per episode:

    {"episode_id": "q17",
     "split": "cal" | "test" | "unlabeled",
     "features": [0.12, -0.3, ...],          # task embedding (for density ratio)
     "steps": [                              # the agent's decision points, in order
        {"node": "root",
         "candidates": {"search": 0.71, "calculator": 0.20, "answer": 0.09}},
        {"node": "root/search",
         "candidates": {"open_doc_3": 0.55, "open_doc_1": 0.40, "answer": 0.05}}
     ],
     "gold_path": ["search", "open_doc_1", "answer:Paris"]}

Candidate probabilities come from the LLM (normalised token log-probs of each
tool call, or sampling frequencies over k rollouts). ``gold_path`` is the
verified correct trajectory (only needed for 'cal' and, for evaluation, 'test').

``load_traces`` builds the action tree from the union of observed prefixes and
returns step-score matrices compatible with ``wcgraph.methods.CalContext``.
Candidates that were not expanded in an episode receive score 1 (probability 0).
"""
from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np

from .graph import Taxonomy


def load_traces(path: str) -> Tuple[Taxonomy, Dict[str, dict]]:
    eps = [json.loads(line) for line in open(path) if line.strip()]
    children: Dict[str, List[str]] = {}
    leaves: List[str] = []

    def add(parent: str, child: str):
        children.setdefault(parent, [])
        if child not in children[parent]:
            children[parent].append(child)

    for ep in eps:
        for st in ep["steps"]:
            for a in st["candidates"]:
                add(st["node"], f'{st["node"]}/{a}')
        if "gold_path" in ep:
            node = "root"
            for a in ep["gold_path"]:
                add(node, f"{node}/{a}")
                node = f"{node}/{a}"
    internal = set(children)
    for kids in children.values():
        leaves += [k for k in kids if k not in internal]
    leaves = sorted(set(leaves))
    tax = Taxonomy(children=children, root="root", leaf_class={l: i for i, l in enumerate(leaves)})
    eid = {e: k for k, e in enumerate(tax.edges)}

    out: Dict[str, dict] = {}
    for split in ("cal", "test", "unlabeled"):
        sel = [ep for ep in eps if ep.get("split") == split]
        S = np.ones((len(sel), tax.n_edges))
        y = np.full(len(sel), -1)
        F = np.array([ep.get("features", []) for ep in sel], float)
        for i, ep in enumerate(sel):
            for st in ep["steps"]:
                for a, p in st["candidates"].items():
                    S[i, eid[(st["node"], f'{st["node"]}/{a}')]] = 1.0 - float(p)
            if "gold_path" in ep:
                leaf = "root/" + "/".join(ep["gold_path"])
                y[i] = tax.leaf_class.get(leaf, -1)
        out[split] = {"S": S, "y": y, "F": F, "ids": [ep["episode_id"] for ep in sel]}
    return tax, out
