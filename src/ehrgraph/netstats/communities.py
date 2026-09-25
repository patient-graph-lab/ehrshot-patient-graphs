"""Community detection: Louvain + Leiden, agreement check (issue #12,
PROJECT_GUIDE.md Module C).

Two different, independently-implemented algorithms, run on the same
graph -- if they land on substantially the same grouping, that's a real,
robust pattern in the data, not one algorithm's arbitrary opinion. Reported
via the adjusted Rand index (0 = no better than random agreement, 1 =
identical partitions).
"""

from __future__ import annotations

import community as community_louvain
import igraph as ig
import leidenalg
import networkx as nx
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def louvain_partition(graph: nx.Graph, seed: int = 0) -> dict:
    """{node: community_id}, via python-louvain."""
    return community_louvain.best_partition(graph, weight="weight", random_state=seed)


def leiden_partition(graph: nx.Graph, seed: int = 0) -> dict:
    """{node: community_id}, via igraph + leidenalg.

    `ig.Graph.from_networkx` preserves each node's original identity in a
    `_nx_name` vertex attribute (verified against a real toy graph, not
    assumed) -- required to map the partition (indexed by igraph's own
    internal vertex order) back to our actual node IDs correctly.
    """
    ig_graph = ig.Graph.from_networkx(graph)
    partition = leidenalg.find_partition(
        ig_graph, leidenalg.ModularityVertexPartition, weights="weight", seed=seed
    )
    return {
        ig_graph.vs[i]["_nx_name"]: community_id
        for i, community_id in enumerate(partition.membership)
    }


def compare_partitions(graph: nx.Graph, seed: int = 0) -> dict:
    """Runs both methods and reports their agreement. Returns a dict with
    both partitions (as pandas Series, node-indexed) plus the adjusted
    Rand index between them, each partition's community count, and each
    partition's weighted modularity (so "Leiden found a better split"
    is a number, not an impression)."""
    louvain = louvain_partition(graph, seed=seed)
    leiden = leiden_partition(graph, seed=seed)

    nodes = list(graph.nodes)
    louvain_labels = [louvain[n] for n in nodes]
    leiden_labels = [leiden[n] for n in nodes]

    return {
        "louvain": pd.Series(louvain, name="louvain_community"),
        "leiden": pd.Series(leiden, name="leiden_community"),
        "adjusted_rand_index": adjusted_rand_score(louvain_labels, leiden_labels),
        "n_louvain_communities": len(set(louvain.values())),
        "n_leiden_communities": len(set(leiden.values())),
        "louvain_modularity": _modularity(graph, louvain),
        "leiden_modularity": _modularity(graph, leiden),
    }


def _modularity(graph: nx.Graph, partition: dict) -> float:
    groups: dict = {}
    for node, community_id in partition.items():
        groups.setdefault(community_id, set()).add(node)
    return nx.community.modularity(graph, groups.values(), weight="weight")
