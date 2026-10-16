"""Spreading and partitioning on the patient networks (course Lectures 4-5).

Three small tools:

  sir_curves            the SIR epidemic model: how fast does "something"
                        spread through a network? Clustered networks with
                        long paths spread slower than random ones.
  cascade_reach         an independent cascade started from patients WITH the
                        outcome: who does it reach? If similar patients share
                        outcomes, the reached patients should have the outcome
                        more often than the base rate. (This is the same idea
                        Project 2's label propagation uses.)
  spectral_vs_leiden    graph partitioning by spectral clustering vs Leiden
                        communities on the largest connected piece.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score


def _neighbors(graph: nx.Graph) -> tuple[list, list[np.ndarray]]:
    nodes = list(graph.nodes)
    index = {n: i for i, n in enumerate(nodes)}
    return nodes, [np.array([index[m] for m in graph.neighbors(n)], dtype=int) for n in nodes]


def sir_curves(
    graph: nx.Graph,
    beta: float = 0.2,
    gamma: float = 0.1,
    n_seeds: int = 10,
    steps: int = 60,
    runs: int = 20,
    seed: int = 0,
) -> np.ndarray:
    """Average share of nodes ever infected (infected + recovered) after each
    step, over `runs` simulations. Each step: every infected node infects each
    susceptible neighbor with probability `beta`, then recovers with
    probability `gamma`. Seeds are random nodes in the largest piece."""
    rng = np.random.default_rng(seed)
    nodes, nbrs = _neighbors(graph)
    lcc = max(nx.connected_components(graph), key=len)
    index = {n: i for i, n in enumerate(nodes)}
    # sorted: set order changes with Python's hash seed, so without this the
    # same `seed` picks different starting nodes in different runs
    lcc_idx = np.sort([index[n] for n in lcc])
    n = len(nodes)
    curves = np.zeros((runs, steps + 1))
    for r in range(runs):
        state = np.zeros(n, dtype=np.int8)  # 0 S, 1 I, 2 R
        state[rng.choice(lcc_idx, size=min(n_seeds, len(lcc_idx)), replace=False)] = 1
        curves[r, 0] = (state > 0).mean()
        for t in range(1, steps + 1):
            infected = np.flatnonzero(state == 1)
            for i in infected:
                nb = nbrs[i]
                if len(nb):
                    hit = nb[(state[nb] == 0) & (rng.random(len(nb)) < beta)]
                    state[hit] = 1
            recover = infected[rng.random(len(infected)) < gamma]
            state[recover] = 2
            curves[r, t] = (state > 0).mean()
    return curves.mean(axis=0)


def cascade_reach(
    graph: nx.Graph,
    labels: dict,
    p: float = 0.3,
    seed_share: float = 0.05,
    runs: int = 50,
    seed: int = 0,
) -> dict:
    """Independent cascade seeded from a random `seed_share` of the patients
    WITH the outcome. Each newly reached patient gets one chance to pass it on
    to each neighbor with probability `p`. Returns the average number of
    patients reached (seeds excluded) and the share of them that have the
    outcome, next to the overall base rate."""
    rng = np.random.default_rng(seed)
    nodes, nbrs = _neighbors(graph)
    y = np.array([labels[n] for n in nodes])
    positives = np.flatnonzero(y == 1)
    k = max(1, int(round(seed_share * len(positives))))
    reached_n, reached_pos = [], []
    for _ in range(runs):
        seeds = rng.choice(positives, size=k, replace=False)
        active = np.zeros(len(nodes), dtype=bool)
        active[seeds] = True
        frontier = list(seeds)
        while frontier:
            new = []
            for i in frontier:
                nb = nbrs[i]
                if len(nb):
                    hit = nb[~active[nb] & (rng.random(len(nb)) < p)]
                    active[hit] = True
                    new.extend(hit.tolist())
            frontier = new
        active[seeds] = False
        reached_n.append(active.sum())
        reached_pos.append(y[active].mean() if active.any() else np.nan)
    return {
        "cascade_reached": float(np.mean(reached_n)),
        "cascade_positive_share": float(np.nanmean(reached_pos)),
        "base_rate": float(y.mean()),
    }


def spectral_vs_leiden(graph: nx.Graph, labels: dict, k: int = 20, seed: int = 0) -> dict:
    """Spectral clustering into `k` groups vs Leiden communities, both on the
    largest connected piece. Reports each partition's modularity, how much of
    the outcome it explains (eta squared), and how much the two agree (ARI)."""
    from ehrgraph.netstats.communities import leiden_partition

    lcc = graph.subgraph(max(nx.connected_components(graph), key=len)).copy()
    nodes = list(lcc.nodes)
    adjacency = sp.csr_matrix(nx.to_scipy_sparse_array(lcc, nodelist=nodes, weight="weight"))
    # scikit-learn's spectral clustering only accepts 32-bit sparse indices
    adjacency.indices = adjacency.indices.astype(np.int32)
    adjacency.indptr = adjacency.indptr.astype(np.int32)
    spectral = SpectralClustering(
        n_clusters=k, affinity="precomputed", random_state=seed, assign_labels="discretize"
    ).fit_predict(adjacency)
    leiden = leiden_partition(lcc, seed=seed)
    leiden_labels = np.array([leiden[n] for n in nodes])
    y = np.array([labels[n] for n in nodes])

    def modularity(groups):
        parts = {}
        for n, g in zip(nodes, groups, strict=True):
            parts.setdefault(g, set()).add(n)
        return nx.community.modularity(lcc, parts.values(), weight="weight")

    def eta_squared(groups):
        grand = y.mean()
        between = sum(
            (groups == g).sum() * (y[groups == g].mean() - grand) ** 2 for g in np.unique(groups)
        )
        total = ((y - grand) ** 2).sum()
        return between / total if total else float("nan")

    return {
        "spectral_modularity": modularity(spectral),
        "leiden_modularity_lcc": modularity(leiden_labels),
        "spectral_eta_squared": eta_squared(spectral),
        "leiden_eta_squared_lcc": eta_squared(leiden_labels),
        "spectral_vs_leiden_ari": adjusted_rand_score(spectral, leiden_labels),
        "n_leiden_groups_lcc": float(len(np.unique(leiden_labels))),
    }
