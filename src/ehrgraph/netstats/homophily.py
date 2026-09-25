"""Homophily and Dirichlet energy (issue #13, PROJECT_GUIDE.md Module D).

Two related but different questions:

  homophily          Do connected nodes tend to share the same *outcome
                      label* (e.g. both "long stay" or both not)? Naive
                      edge homophily is biased toward whichever class is
                      larger -- if 75% of nodes are class 0, most edges
                      are 0-0 just by base rate. `class_insensitive_homophily`
                      is Lim et al. (2021)'s correction: each class's
                      same-class neighbor rate minus that class's share of
                      nodes, so a randomly wired graph scores ~0 at any
                      prevalence. (PROJECT_GUIDE.md Module D cites Zhu et
                      al. 2020 for the idea; Lim et al. give the exact
                      imbalance-corrected formula used here.)

  dirichlet_energy    A continuous version of the same question, using
                      CLMBR's real-valued fingerprints instead of a
                      label: how "bumpy" is that signal across the
                      graph's edges? Low energy = connected patients
                      already have very similar fingerprints = the graph
                      structure agrees with what CLMBR already knows.

**Two corrections this module exists to get right** (PROJECT_GUIDE.md
Module D): `nx.laplacian_matrix` orders rows by `G.nodes()` insertion
order, which will NOT match a feature matrix `X`'s row order unless you
force it -- silently producing a number that runs without error but means
nothing. `nodelist=` must be passed explicitly, and the caller must build
`X` in that exact same order (this module asserts it, rather than trusting
the caller got it right). Second, raw `trace(Xᵀ L X)` scales with edge
count and weight magnitude, so it's not comparable across the three graph
conditions (they have different edge counts) -- normalized by the
feature matrix's squared Frobenius norm to stay scale-free.
"""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
import numpy as np


def edge_homophily(graph: nx.Graph, labels: dict) -> float:
    """Naive edge homophily: fraction of edges whose endpoints share a
    label. Reported only as a reference point -- it rises with the
    majority class's share of nodes, so it can't be compared across
    tasks with different prevalence (0.25 for LOS vs 0.14 for
    Hypertension)."""
    m = graph.number_of_edges()
    if m == 0:
        return float("nan")
    return sum(1 for u, v in graph.edges() if labels[u] == labels[v]) / m


def class_insensitive_homophily(graph: nx.Graph, labels: dict) -> dict:
    """Class-insensitive homophily (Lim et al., NeurIPS 2021, eq. 2):

        h_k = (same-class neighbor slots of class-k nodes)
              / (all neighbor slots of class-k nodes)
            = 2 * e_kk / (2 * e_kk + e_k,other)          (for 2 classes)
        h_hat = 1/(C-1) * sum_k max(0, h_k - p_k)

    where p_k is class k's share of all nodes. Subtracting p_k is the whole
    point: in a graph wired at random, a class-k node's neighbors are class
    k with probability p_k, so h_k ~= p_k and h_hat ~= 0 regardless of how
    imbalanced the classes are. h_hat = 1 only when every edge is
    same-class. (Correction, Sep 25: the first draft used e_kk / (e_kk +
    e_k,other) with no p_k baseline, which still scores ~0.37 on a random
    graph at prevalence 0.25 and so moves with prevalence -- exactly the
    confound this metric exists to remove, and fatal for a cross-task
    homophily-vs-gain comparison.)

    `labels`: {node: class}, and must cover every node in the graph.
    Returns per-class h_k, class proportions p_k, and h_hat.
    """
    missing = set(graph.nodes) - set(labels)
    if missing:
        raise ValueError(f"{len(missing)} graph node(s) have no label.")
    node_labels = [labels[n] for n in graph.nodes]
    classes = sorted(set(node_labels))
    n = len(node_labels)
    proportion = {c: node_labels.count(c) / n for c in classes}

    same_slots = dict.fromkeys(classes, 0)
    all_slots = dict.fromkeys(classes, 0)
    for u, v in graph.edges():
        cu, cv = labels[u], labels[v]
        all_slots[cu] += 1
        all_slots[cv] += 1
        if cu == cv:
            same_slots[cu] += 2

    per_class = {
        c: (same_slots[c] / all_slots[c] if all_slots[c] > 0 else float("nan")) for c in classes
    }
    if len(classes) < 2:
        h_hat = float("nan")
    else:
        h_hat = sum(
            max(0.0, per_class[c] - proportion[c]) for c in classes if not np.isnan(per_class[c])
        ) / (len(classes) - 1)

    return {
        "per_class": per_class,
        "class_proportion": proportion,
        "class_insensitive_homophily": float(h_hat),
    }


def dirichlet_energy(graph: nx.Graph, nodelist: Sequence, features: np.ndarray) -> float:
    """Scale-free Dirichlet energy: trace(X^T L_norm X) / ||X||_F^2.

    `nodelist` must be the exact row order of `features` -- asserted here,
    not assumed, since a silent mismatch here produces a real-looking but
    meaningless number (PROJECT_GUIDE.md Module D correction).
    """
    nodelist = list(nodelist)
    if len(nodelist) != features.shape[0]:
        raise ValueError(
            f"nodelist has {len(nodelist)} entries but features has {features.shape[0]} rows "
            "-- they must be the same length and the same order."
        )
    if set(nodelist) != set(graph.nodes):
        raise ValueError("nodelist must contain exactly the graph's nodes, no more, no fewer.")

    laplacian = nx.normalized_laplacian_matrix(graph, nodelist=nodelist, weight="weight")
    # trace(X^T L X) computed as sum(X * (L @ X)) -- avoids ever
    # materializing a dense n x n Laplacian, which at a few thousand nodes
    # would be needlessly slow and memory-heavy for a number that reduces
    # to a single scalar.
    lx = laplacian @ features
    numerator = float(np.sum(features * lx))
    denominator = float(np.linalg.norm(features, ord="fro") ** 2)
    return numerator / denominator
